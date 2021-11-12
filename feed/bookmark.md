
## 2021-11-12

### [[2111.06080] Catching Unusual Traffic Behavior using TF-IDF-based Port Access Statistics Analysis](http://arxiv.org/abs/2111.06080)


  Detecting the anomalous behavior of traffic is one of the important actions
for network operators. In this study, we applied term frequency - inverse
document frequency (TF-IDF), which is a popular method used in natural language
processing, to detect unusual behavior from network access logs. We mapped the
term and document concept to the port number and daily access history,
respectively, and calculated the TF-IDF. With this approach, we could obtain
ports frequently observed in fewer days compared to other port access
activities. Such access behaviors are not always malicious activities; however,
such information is a good indicator for starting a deeper analysis of traffic
behavior. Using a real-life dataset, we could detect two bot-oriented accesses
and one unique UDP traffic.

    

### [[2111.06087] Classification of URL bitstreams using Bag of Bytes](http://arxiv.org/abs/2111.06087)


  Protecting users from accessing malicious web sites is one of the important
management tasks for network operators. There are many open-source and
commercial products to control web sites users can access. The most traditional
approach is blacklist-based filtering. This mechanism is simple but not
scalable, though there are some enhanced approaches utilizing fuzzy matching
technologies. Other approaches try to use machine learning (ML) techniques by
extracting features from URL strings. This approach can cover a wider area of
Internet web sites, but finding good features requires deep knowledge of trends
of web site design. Recently, another approach using deep learning (DL) has
appeared. The DL approach will help to extract features automatically by
investigating a lot of existing sample data. Using this technique, we can build
a flexible filtering decision module by keep teaching the neural network module
about recent trends, without any specific expert knowledge of the URL domain.
In this paper, we apply a mechanical approach to generate feature vectors from
URL strings. We implemented our approach and tested with realistic URL access
history data taken from a research organization and data from the famous
archive site of phishing site information, this http URL. Our approach achieved
2~3% better accuracy compared to the existing DL-based approach.

    

### [[2111.06161] Understanding mobility in networks: A node embedding approach](http://arxiv.org/abs/2111.06161)


  Motivated by the growing number of mobile devices capable of connecting and
exchanging messages, we propose a methodology aiming to model and analyze node
mobility in networks. We note that many existing solutions in the literature
rely on topological measurements calculated directly on the graph of node
contacts, aiming to capture the notion of the node's importance in terms of
connectivity and mobility patterns beneficial for prototyping, design, and
deployment of mobile networks. However, each measure has its specificity and
fails to generalize the node importance notions that ultimately change over
time. Unlike previous approaches, our methodology is based on a node embedding
method that models and unveils the nodes' importance in mobility and
connectivity patterns while preserving their spatial and temporal
characteristics. We focus on a case study based on a trace of group meetings.
The results show that our methodology provides a rich representation for
extracting different mobility and connectivity patterns, which can be helpful
for various applications and services in mobile networks.

    

### [[2111.06263] Towards Live Video Analytics with On-Drone Deeper-yet-Compatible Compression](http://arxiv.org/abs/2111.06263)


  In this work, we present DCC(Deeper-yet-Compatible Compression), one enabling
technique for real-time drone-sourced edge-assisted video analytics built on
top of the existing codec. DCC tackles an important technical problem to
compress streamed video from the drone to the edge without scarifying accuracy
and timeliness of video analytical tasks performed at the edge. DCC is inspired
by the fact that not every bit in streamed video is equally valuable to video
analytics, which opens new compression room over the conventional
analytics-oblivious video codec technology. We exploit drone-specific context
and intermediate hints from object detection to pursue adaptive fidelity needed
to retain analytical quality. We have prototyped DCC in one showcase
application of vehicle detection and validated its efficiency in representative
scenarios. DCC has reduced transmission volume by 9.5-fold over the baseline
approach and 19-683% over the state-of-the-art with comparable detection
accuracy.

    

### [[2111.06352] Performance of Queueing Models for MISO Content-Centric Networks](http://arxiv.org/abs/2111.06352)


  MISO networks have garnered attention in wireless content-centric networks
due to the additional degrees of freedoms they provide. Several beamforming
techniques such as NOMA, OMA, SDMA and Rate splitting have been proposed for
such networks. These techniques utilise the redundancy in the content requests
across users and leverage the spatial multicast and multiplexing gains of
multi-antenna transmit beamforming to improve the content delivery rate.
However, queueing delays and user traffic dynamics which significantly affect
the performance of these schemes, have generally been ignored. We study
queueing delays in the downlink for several scheduling and beamforming schemes
in content-centric networks, with one base-station possessing multiple transmit
antennas. These schemes are studied along with a recently proposed Simple
Multicast Queue, to improve the delay performance of the network. This work is
particularly relevant for content delivery in 5G and eMBB networks.

    

### [[2111.05874] A Hierarchy for Replica Quantum Advantage](http://arxiv.org/abs/2111.05874)


  We prove that given the ability to make entangled measurements on at most $k$
replicas of an $n$-qubit state $\rho$ simultaneously, there is a property of
$\rho$ which requires at least order $2^n / k^2$ measurements to learn.
However, the same property only requires one measurement to learn if we can
make an entangled measurement over a number of replicas polynomial in $k, n$.
Because the above holds for each positive integer $k$, we obtain a hierarchy of
tasks necessitating progressively more replicas to be performed efficiently. We
introduce a powerful proof technique to establish our results, and also use
this to provide new bounds for testing the mixedness of a quantum state.

    

### [[2111.05881] Exponential separations between learning with and without quantum memory](http://arxiv.org/abs/2111.05881)


  We study the power of quantum memory for learning properties of quantum
systems and dynamics, which is of great importance in physics and chemistry.
Many state-of-the-art learning algorithms require access to an additional
external quantum memory. While such a quantum memory is not required a priori,
in many cases, algorithms that do not utilize quantum memory require much more
data than those which do. We show that this trade-off is inherent in a wide
range of learning problems. Our results include the following:
(1) We show that to perform shadow tomography on an $n$-qubit state rho with
$M$ observables, any algorithm without quantum memory requires $\Omega(\min(M,
2^n))$ samples of rho in the worst case. Up to logarithmic factors, this
matches the upper bound of [HKP20] and completely resolves an open question in
[Aar18, AR19].
(2) We establish exponential separations between algorithms with and without
quantum memory for purity testing, distinguishing scrambling and depolarizing
evolutions, as well as uncovering symmetry in physical dynamics. Our
separations improve and generalize prior work of [ACQ21] by allowing for a
broader class of algorithms without quantum memory.
(3) We give the first tradeoff between quantum memory and sample complexity.
We prove that to estimate absolute values of all $n$-qubit Pauli observables,
algorithms with $k < n$ qubits of quantum memory require at least
$\Omega(2^{(n-k)/3})$ samples, but there is an algorithm using $n$-qubit
quantum memory which only requires $O(n)$ samples.
The separations we show are sufficiently large and could already be evident,
for instance, with tens of qubits. This provides a concrete path towards
demonstrating real-world advantage for learning algorithms with quantum memory.

    

### [[2111.05885] Predicting Lattice Phonon Vibrational Frequencies Using Deep Graph Neural Networks](http://arxiv.org/abs/2111.05885)


  Lattice vibration frequencies are related to many important materials
properties such as thermal and electrical conductivity as well as
superconductivity. However, computational calculation of vibration frequencies
using density functional theory (DFT) methods is too computationally demanding
for a large number of samples in materials screening. Here we propose a deep
graph neural network-based algorithm for predicting crystal vibration
frequencies from crystal structures with high accuracy. Our algorithm addresses
the variable dimension of vibration frequency spectrum using the zero padding
scheme. Benchmark studies on two data sets with 15,000 and 35,552 samples show
that the aggregated $R^2$ scores of the prediction reaches 0.554 and 0.724
respectively. Our work demonstrates the capability of deep graph neural
networks to learn to predict phonon spectrum properties of crystal structures
in addition to phonon density of states (DOS) and electronic DOS in which the
output dimension is constant.

    

### [[2111.05894] Graph Neural Network Training with Data Tiering](http://arxiv.org/abs/2111.05894)


  Graph Neural Networks (GNNs) have shown success in learning from
graph-structured data, with applications to fraud detection, recommendation,
and knowledge graph reasoning. However, training GNN efficiently is challenging
because: 1) GPU memory capacity is limited and can be insufficient for large
datasets, and 2) the graph-based data structure causes irregular data access
patterns. In this work, we provide a method to statistical analyze and identify
more frequently accessed data ahead of GNN training. Our data tiering method
not only utilizes the structure of input graph, but also an insight gained from
actual GNN training process to achieve a higher prediction result. With our
data tiering method, we additionally provide a new data placement and access
strategy to further minimize the CPU-GPU communication overhead. We also take
into account of multi-GPU GNN training as well and we demonstrate the
effectiveness of our strategy in a multi-GPU system. The evaluation results
show that our work reduces CPU-GPU traffic by 87-95% and improves the training
speed of GNN over the existing solutions by 1.6-2.1x on graphs with hundreds of
millions of nodes and billions of edges.

    

### [[2111.05895] A Generic Deep Learning Based Cough Analysis System from Clinically Validated Samples for Point-of-Need Covid-19 Test and Severity Levels](http://arxiv.org/abs/2111.05895)


  We seek to evaluate the detection performance of a rapid primary screening
tool of Covid-19 solely based on the cough sound from 8,380 clinically
validated samples with laboratory molecular-test (2,339 Covid-19 positives and
6,041 Covid-19 negatives). Samples were clinically labeled according to the
results and severity based on quantitative RT-PCR (qRT-PCR) analysis, cycle
threshold, and lymphocytes count from the patients. Our proposed generic method
is an algorithm based on Empirical Mode Decomposition (EMD) with subsequent
classification based on a tensor of audio features and a deep artificial neural
network classifier with convolutional layers called DeepCough'. Two different
versions of DeepCough based on the number of tensor dimensions, i.e.
DeepCough2D and DeepCough3D, have been investigated. These methods have been
deployed in a multi-platform proof-of-concept Web App CoughDetect to administer
this test anonymously. Covid-19 recognition results rates achieved a promising
AUC (Area Under Curve) of 98.800.83%, sensitivity of 96.431.85%, and
specificity of 96.201.74%, and 81.08%5.05% AUC for the recognition of three
severity levels. Our proposed web tool and underpinning algorithm for the
robust, fast, point-of-need identification of Covid-19 facilitates the rapid
detection of the infection. We believe that it has the potential to
significantly hamper the Covid-19 pandemic across the world.

    

### [[2111.05897] Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters](http://arxiv.org/abs/2111.05897)


  Deep learning based models have dominated the current landscape of production
recommender systems. Furthermore, recent years have witnessed an exponential
growth of the model scale--from Google's 2016 model with 1 billion parameters
to the latest Facebook's model with 12 trillion parameters. Significant quality
boost has come with each jump of the model capacity, which makes us believe the
era of 100 trillion parameters is around the corner. However, the training of
such models is challenging even within industrial scale data centers. This
difficulty is inherited from the staggering heterogeneity of the training
computation--the model's embedding layer could include more than 99.99% of the
total model size, which is extremely memory-intensive; while the rest neural
network is increasingly computation-intensive. To support the training of such
huge models, an efficient distributed training system is in urgent need. In
this paper, we resolve this challenge by careful co-design of both the
optimization algorithm and the distributed system architecture. Specifically,
in order to ensure both the training efficiency and the training accuracy, we
design a novel hybrid training algorithm, where the embedding layer and the
dense neural network are handled by different synchronization mechanisms; then
we build a system called Persia (short for parallel recommendation training
system with hybrid acceleration) to support this hybrid training algorithm.
Both theoretical demonstration and empirical study up to 100 trillion
parameters have conducted to justified the system design and implementation of
Persia. We make Persia publicly available (at
this https URL) so that anyone would be able to easily
train a recommender model at the scale of 100 trillion parameters.

    

### [[2111.05898] Beyond Importance Scores: Interpreting Tabular ML by Visualizing Feature Semantics](http://arxiv.org/abs/2111.05898)


  Interpretability is becoming an active research topic as machine learning
(ML) models are more widely used to make critical decisions. Tabular data is
one of the most commonly used modes of data in diverse applications such as
healthcare and finance. Much of the existing interpretability methods used for
tabular data only report feature-importance scores -- either locally (per
example) or globally (per model) -- but they do not provide interpretation or
visualization of how the features interact. We address this limitation by
introducing Feature Vectors, a new global interpretability method designed for
tabular datasets. In addition to providing feature-importance, Feature Vectors
discovers the inherent semantic relationship among features via an intuitive
feature visualization technique. Our systematic experiments demonstrate the
empirical utility of this new method by applying it to several real-world
datasets. We further provide an easy-to-use Python package for Feature Vectors.

    

### [[2111.05917] Recognition of Patient Groups with Sleep Related Disorders using Bio-signal Processing and Deep Learning](http://arxiv.org/abs/2111.05917)


  Accurately diagnosing sleep disorders is essential for clinical assessments
and treatments. Polysomnography (PSG) has long been used for detection of
various sleep disorders. In this research, electrocardiography (ECG) and
electromayography (EMG) have been used for recognition of breathing and
movement-related sleep disorders. Bio-signal processing has been performed by
extracting EMG features exploiting entropy and statistical moments, in addition
to developing an iterative pulse peak detection algorithm using synchrosqueezed
wavelet transform (SSWT) for reliable extraction of heart rate and
breathing-related features from ECG. A deep learning framework has been
designed to incorporate EMG and ECG features. The framework has been used to
classify four groups: healthy subjects, patients with obstructive sleep apnea
(OSA), patients with restless leg syndrome (RLS) and patients with both OSA and
RLS. The proposed deep learning framework produced a mean accuracy of 72% and
weighted F1 score of 0.57 across subjects for our formulated four-class
problem.

    

### [[2111.05934] A soft thumb-sized vision-based sensor with accurate all-round force perception](http://arxiv.org/abs/2111.05934)


  Vision-based haptic sensors have emerged as a promising approach to robotic
touch due to affordable high-resolution cameras and successful computer-vision
techniques. However, their physical design and the information they provide do
not yet meet the requirements of real applications. We present a robust, soft,
low-cost, vision-based, thumb-sized 3D haptic sensor named Insight: it
continually provides a directional force-distribution map over its entire
conical sensing surface. Constructed around an internal monocular camera, the
sensor has only a single layer of elastomer over-molded on a stiff frame to
guarantee sensitivity, robustness, and soft contact. Furthermore, Insight is
the first system to combine photometric stereo and structured light using a
collimator to detect the 3D deformation of its easily replaceable flexible
outer shell. The force information is inferred by a deep neural network that
maps images to the spatial distribution of 3D contact force (normal and shear).
Insight has an overall spatial resolution of 0.4 mm, force magnitude accuracy
around 0.03 N, and force direction accuracy around 5 degrees over a range of
0.03--2 N for numerous distinct contacts with varying contact area. The
presented hardware and software design concepts can be transferred to a wide
variety of robot parts.

    

### [[2111.05935] A Meta-Method for Portfolio Management Using Machine Learning for Adaptive Strategy Selection](http://arxiv.org/abs/2111.05935)


  This work proposes a novel portfolio management technique, the Meta Portfolio
Method (MPM), inspired by the successes of meta approaches in the field of
bioinformatics and elsewhere. The MPM uses XGBoost to learn how to switch
between two risk-based portfolio allocation strategies, the Hierarchical Risk
Parity (HRP) and more classical Naïve Risk Parity (NRP). It is demonstrated
that the MPM is able to successfully take advantage of the best characteristics
of each strategy (the NRP's fast growth during market uptrends, and the HRP's
protection against drawdowns during market turmoil). As a result, the MPM is
shown to possess an excellent out-of-sample risk-reward profile, as measured by
the Sharpe ratio, and in addition offers a high degree of interpretability of
its asset allocation decisions.

    

### [[2111.05936] SPA-GCN: Efficient and Flexible GCN Accelerator with an Application for Graph Similarity Computation](http://arxiv.org/abs/2111.05936)


  While there have been many studies on hardware acceleration for deep learning
on images, there has been a rather limited focus on accelerating deep learning
applications involving graphs. The unique characteristics of graphs, such as
the irregular memory access and dynamic parallelism, impose several challenges
when the algorithm is mapped to a CPU or GPU. To address these challenges while
exploiting all the available sparsity, we propose a flexible architecture
called SPA-GCN for accelerating Graph Convolutional Networks (GCN), the core
computation unit in deep learning algorithms on graphs. The architecture is
specialized for dealing with many small graphs since the graph size has a
significant impact on design considerations. In this context, we use SimGNN, a
neural-network-based graph matching algorithm, as a case study to demonstrate
the effectiveness of our architecture. The experimental results demonstrate
that SPA-GCN can deliver a high speedup compared to a multi-core CPU
implementation and a GPU implementation, showing the efficiency of our design.

    

### [[2111.05939] A study on Channel Popularity in Twitch](http://arxiv.org/abs/2111.05939)


  In the past few decades, there has been an increasing need for Internet users
to host real time events online and to share their experiences with live,
interactive audiences. Online streaming services like Twitch have attracted
millions of users to stream and to spectate. There have been few studies about
the prediction of streamers' popularity on Twitch. In this paper, we look at
potential factors that can contribute to the popularity of streamers. Streamer
data was collected through consistent tracking using Twitch's API during a 4
weeks period. Each user's streaming information such as the number of current
viewers and followers, the genre of the stream etc., were collected. From the
results, we found that the frequency of streaming sessions, the types of
content and the length of the streams are major factors in determining how much
viewers and subscribers streamers can gain during sessions.

    

### [[2111.05941] Generalizable Cross-Graph Embedding for GNN-based Congestion Prediction](http://arxiv.org/abs/2111.05941)


  Presently with technology node scaling, an accurate prediction model at early
design stages can significantly reduce the design cycle. Especially during
logic synthesis, predicting cell congestion due to improper logic combination
can reduce the burden of subsequent physical implementations. There have been
attempts using Graph Neural Network (GNN) techniques to tackle congestion
prediction during the logic synthesis stage. However, they require informative
cell features to achieve reasonable performance since the core idea of GNNs is
built on the message passing framework, which would be impractical at the early
logic synthesis stage. To address this limitation, we propose a framework that
can directly learn embeddings for the given netlist to enhance the quality of
our node features. Popular random-walk based embedding methods such as
Node2vec, LINE, and DeepWalk suffer from the issue of cross-graph alignment and
poor generalization to unseen netlist graphs, yielding inferior performance and
costing significant runtime. In our framework, we introduce a superior
alternative to obtain node embeddings that can generalize across netlist graphs
using matrix factorization methods. We propose an efficient mini-batch training
method at the sub-graph level that can guarantee parallel training and satisfy
the memory restriction for large-scale netlists. We present results utilizing
open-source EDA tools such as DREAMPLACE and OPENROAD frameworks on a variety
of openly available circuits. By combining the learned embedding on top of the
netlist with the GNNs, our method improves prediction performance, generalizes
to new circuit lines, and is efficient in training, potentially saving over $90
\%$ of runtime.

    

### [[2111.05949] How to See Hidden Patterns in Metamaterials with Interpretable Machine Learning](http://arxiv.org/abs/2111.05949)


  Metamaterials are composite materials with engineered geometrical micro- and
meso-structures that can lead to uncommon physical properties, like negative
Poisson's ratio or ultra-low shear resistance. Periodic metamaterials are
composed of repeating unit-cells, and geometrical patterns within these
unit-cells influence the propagation of elastic or acoustic waves and control
dispersion. In this work, we develop a new interpretable, multi-resolution
machine learning framework for finding patterns in the unit-cells of materials
that reveal their dynamic properties. Specifically, we propose two new
interpretable representations of metamaterials, called shape-frequency features
and unit-cell templates. Machine learning models built using these feature
classes can accurately predict dynamic material properties. These feature
representations (particularly the unit-cell templates) have a useful property:
they can operate on designs of higher resolutions. By learning key coarse scale
patterns that can be reliably transferred to finer resolution design space via
the shape-frequency features or unit-cell templates, we can almost freely
design the fine resolution features of the unit-cell without changing coarse
scale physics. Through this multi-resolution approach, we are able to design
materials that possess target frequency ranges in which waves are allowed or
disallowed to propagate (frequency bandgaps). Our approach yields major
benefits: (1) unlike typical machine learning approaches to materials science,
our models are interpretable, (2) our approaches leverage multi-resolution
properties, and (3) our approach provides design flexibility.

    

### [[2111.05950] Self-Compression in Bayesian Neural Networks](http://arxiv.org/abs/2111.05950)


  Machine learning models have achieved human-level performance on various
tasks. This success comes at a high cost of computation and storage overhead,
which makes machine learning algorithms difficult to deploy on edge devices.
Typically, one has to partially sacrifice accuracy in favor of an increased
performance quantified in terms of reduced memory usage and energy consumption.
Current methods compress the networks by reducing the precision of the
parameters or by eliminating redundant ones. In this paper, we propose a new
insight into network compression through the Bayesian framework. We show that
Bayesian neural networks automatically discover redundancy in model parameters,
thus enabling self-compression, which is linked to the propagation of
uncertainty through the layers of the network. Our experimental results show
that the network architecture can be successfully compressed by deleting
parameters identified by the network itself while retaining the same level of
accuracy.

    

### [[2111.05953] Robust Learning via Ensemble Density Propagation in Deep Neural Networks](http://arxiv.org/abs/2111.05953)


  Learning in uncertain, noisy, or adversarial environments is a challenging
task for deep neural networks (DNNs). We propose a new theoretically grounded
and efficient approach for robust learning that builds upon Bayesian estimation
and Variational Inference. We formulate the problem of density propagation
through layers of a DNN and solve it using an Ensemble Density Propagation
(EnDP) scheme. The EnDP approach allows us to propagate moments of the
variational probability distribution across the layers of a Bayesian DNN,
enabling the estimation of the mean and covariance of the predictive
distribution at the output of the model. Our experiments using MNIST and
CIFAR-10 datasets show a significant improvement in the robustness of the
trained models to random noise and adversarial attacks.

    

### [[2111.05955] Keys to Accurate Feature Extraction Using Residual Spiking Neural Networks](http://arxiv.org/abs/2111.05955)


  Spiking neural networks (SNNs) have become an interesting alternative to
conventional artificial neural networks (ANN) thanks to their temporal
processing capabilities and their low-SWaP (Size, Weight, and Power) and energy
efficient implementations in neuromorphic hardware. However the challenges
involved in training SNNs have limited their performance in terms of accuracy
and thus their applications. Improving learning algorithms and neural
architectures for a more accurate feature extraction is therefore one of the
current priorities in SNN research. In this paper we present a study on the key
components of modern spiking architectures. We empirically compare different
techniques in image classification datasets taken from the best performing
networks. We design a spiking version of the successful residual network
(ResNet) architecture and test different components and training strategies on
it. Our results provide a state of the art guide to SNN design, which allows to
make informed choices when trying to build the optimal visual feature
extractor. Finally, our network outperforms previous SNN architectures in
CIFAR-10 (94.1%) and CIFAR-100 (74.5%) datasets and matches the state of the
art in DVS-CIFAR10 (71.3%), with less parameters than the previous state of the
art and without the need for ANN-SNN conversion. Code available at
this https URL.

    

### [[2111.05956] Feature Generation for Long-tail Classification](http://arxiv.org/abs/2111.05956)


  The visual world naturally exhibits an imbalance in the number of object or
scene instances resulting in a \emph{long-tailed distribution}. This imbalance
poses significant challenges for classification models based on deep learning.
Oversampling instances of the tail classes attempts to solve this imbalance.
However, the limited visual diversity results in a network with poor
representation ability. A simple counter to this is decoupling the
representation and classifier networks and using oversampling only to train the
classifier. In this paper, instead of repeatedly re-sampling the same image
(and thereby features), we explore a direction that attempts to generate
meaningful features by estimating the tail category's distribution. Inspired by
ideas from recent work on few-shot learning, we create calibrated distributions
to sample additional features that are subsequently used to train the
classifier. Through several experiments on the CIFAR-100-LT (long-tail) dataset
with varying imbalance factors and on mini-ImageNet-LT (long-tail), we show the
efficacy of our approach and establish a new state-of-the-art. We also present
a qualitative analysis of generated features using t-SNE visualizations and
analyze the nearest neighbors used to calibrate the tail class distributions.
Our code is available at this https URL.

    

### [[2111.05962] Adversarial sampling of unknown and high-dimensional conditional distributions](http://arxiv.org/abs/2111.05962)


  Many engineering problems require the prediction of
realization-to-realization variability or a refined description of modeled
quantities. In that case, it is necessary to sample elements from unknown
high-dimensional spaces with possibly millions of degrees of freedom. While
there exist methods able to sample elements from probability density functions
(PDF) with known shapes, several approximations need to be made when the
distribution is unknown. In this paper the sampling method, as well as the
inference of the underlying distribution, are both handled with a data-driven
method known as generative adversarial networks (GAN), which trains two
competing neural networks to produce a network that can effectively generate
samples from the training set distribution. In practice, it is often necessary
to draw samples from conditional distributions. When the conditional variables
are continuous, only one (if any) data point corresponding to a particular
value of a conditioning variable may be available, which is not sufficient to
estimate the conditional distribution. This work handles this problem using an
a priori estimation of the conditional moments of a PDF. Two approaches,
stochastic estimation, and an external neural network are compared here for
computing these moments; however, any preferred method can be used. The
algorithm is demonstrated in the case of the deconvolution of a filtered
turbulent flow field. It is shown that all the versions of the proposed
algorithm effectively sample the target conditional distribution with minimal
impact on the quality of the samples compared to state-of-the-art methods.
Additionally, the procedure can be used as a metric for the diversity of
samples generated by a conditional GAN (cGAN) conditioned with continuous
variables.

    

### [[2111.05968] Linear Speedup in Personalized Collaborative Learning](http://arxiv.org/abs/2111.05968)


  Personalization in federated learning can improve the accuracy of a model for
a user by trading off the model's bias (introduced by using data from other
users who are potentially different) against its variance (due to the limited
amount of data on any single user). In order to develop training algorithms
that optimally balance this trade-off, it is necessary to extend our
theoretical foundations. In this work, we formalize the personalized
collaborative learning problem as stochastic optimization of a user's objective
$f_0(x)$ while given access to $N$ related but different objectives of other
users $\{f_1(x), \dots, f_N(x)\}$. We give convergence guarantees for two
algorithms in this setting -- a popular personalization method known as
\emph{weighted gradient averaging}, and a novel \emph{bias correction} method
-- and explore conditions under which we can optimally trade-off their bias for
a reduction in variance and achieve linear speedup w.r.t.\ the number of users
$N$. Further, we also empirically study their performance confirming our
theoretical insights.

    

### [[2111.05969] PowerGridworld: A Framework for Multi-Agent Reinforcement Learning in Power Systems](http://arxiv.org/abs/2111.05969)


  We present the PowerGridworld software package to provide users with a
lightweight, modular, and customizable framework for creating
power-systems-focused, multi-agent Gym environments that readily integrate with
existing training frameworks for reinforcement learning (RL). Although many
frameworks exist for training multi-agent RL (MARL) policies, none can rapidly
prototype and develop the environments themselves, especially in the context of
heterogeneous (composite, multi-device) power systems where power flow
solutions are required to define grid-level variables and costs. PowerGridworld
is an open-source software package that helps to fill this gap. To highlight
PowerGridworld's key features, we present two case studies and demonstrate
learning MARL policies using both OpenAI's multi-agent deep deterministic
policy gradient (MADDPG) and RLLib's proximal policy optimization (PPO)
algorithms. In both cases, at least some subset of agents incorporates elements
of the power flow solution at each time step as part of their reward (negative
cost) structures.

    

### [[2111.05972] Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training](http://arxiv.org/abs/2111.05972)


  With deep learning models rapidly growing in size, systems-level solutions
for large-model training are required. We present Amazon SageMaker model
parallelism, a software library that integrates with PyTorch, and enables easy
training of large models using model parallelism and other memory-saving
features. In contrast to existing solutions, the implementation of the
SageMaker library is much more generic and flexible, in that it can
automatically partition and run pipeline parallelism over arbitrary model
architectures with minimal code change, and also offers a general and
extensible framework for tensor parallelism, which supports a wider range of
use cases, and is modular enough to be easily applied to new training scripts.
The library also preserves the native PyTorch user experience to a much larger
degree, supporting module re-use and dynamic graphs, while giving the user full
control over the details of the training step. We evaluate performance over
GPT-3, RoBERTa, BERT, and neural collaborative filtering, and demonstrate
competitive performance over existing solutions.

    

### [[2111.05973] Soft Sensing Transformer: Hundreds of Sensors are Worth a Single Word](http://arxiv.org/abs/2111.05973)


  With the rapid development of AI technology in recent years, there have been
many studies with deep learning models in soft sensing area. However, the
models have become more complex, yet, the data sets remain limited: researchers
are fitting million-parameter models with hundreds of data samples, which is
insufficient to exercise the effectiveness of their models and thus often fail
to perform when implemented in industrial applications. To solve this
long-lasting problem, we are providing large scale, high dimensional time
series manufacturing sensor data from Seagate Technology to the public. We
demonstrate the challenges and effectiveness of modeling industrial big data by
a Soft Sensing Transformer model on these data sets. Transformer is used
because, it has outperformed state-of-the-art techniques in Natural Language
Processing, and since then has also performed well in the direct application to
computer vision without introduction of image-specific inductive biases. We
observe the similarity of a sentence structure to the sensor readings and
process the multi-variable sensor readings in a time series in a similar manner
of sentences in natural language. The high-dimensional time-series data is
formatted into the same shape of embedded sentences and fed into the
transformer model. The results show that transformer model outperforms the
benchmark models in soft sensing field based on auto-encoder and long
short-term memory (LSTM) models. To the best of our knowledge, we are the first
team in academia or industry to benchmark the performance of original
transformer model with large-scale numerical soft sensing data.

    

### [[2111.05976] Classification of the Chess Endgame problem using Logistic Regression, Decision Trees, and Neural Networks](http://arxiv.org/abs/2111.05976)


  In this study we worked on the classification of the Chess Endgame problem
using different algorithms like logistic regression, decision trees and neural
networks. Our experiments indicates that the Neural Networks provides the best
accuracy (85%) then the decision trees (79%). We did these experiments using
Microsoft Azure Machine Learning as a case-study on using Visual Programming in
classification. Our experiments demonstrates that this tool is powerful and
save a lot of time, also it could be improved with more features that increase
the usability and reduce the learning curve. We also developed an application
for dataset visualization using a new programming language called Ring, our
experiments demonstrates that this language have simple design like Python
while integrates RAD tools like Visual Basic which is good for GUI development
in the open-source world

    

### [[2111.05978] Trustworthy Medical Segmentation with Uncertainty Estimation](http://arxiv.org/abs/2111.05978)


  Deep Learning (DL) holds great promise in reshaping the healthcare systems
given its precision, efficiency, and objectivity. However, the brittleness of
DL models to noisy and out-of-distribution inputs is ailing their deployment in
the clinic. Most systems produce point estimates without further information
about model uncertainty or confidence. This paper introduces a new Bayesian
deep learning framework for uncertainty quantification in segmentation neural
networks, specifically encoder-decoder architectures. The proposed framework
uses the first-order Taylor series approximation to propagate and learn the
first two moments (mean and covariance) of the distribution of the model
parameters given the training data by maximizing the evidence lower bound. The
output consists of two maps: the segmented image and the uncertainty map of the
segmentation. The uncertainty in the segmentation decisions is captured by the
covariance matrix of the predictive distribution. We evaluate the proposed
framework on medical image segmentation data from Magnetic Resonances Imaging
and Computed Tomography scans. Our experiments on multiple benchmark datasets
demonstrate that the proposed framework is more robust to noise and adversarial
attacks as compared to state-of-the-art segmentation models. Moreover, the
uncertainty map of the proposed framework associates low confidence (or
equivalently high uncertainty) to patches in the test input images that are
corrupted with noise, artifacts or adversarial attacks. Thus, the model can
self-assess its segmentation decisions when it makes an erroneous prediction or
misses part of the segmentation structures, e.g., tumor, by presenting higher
values in the uncertainty map.

    

### [[2111.05986] SyMetric: Measuring the Quality of Learnt Hamiltonian Dynamics Inferred from Vision](http://arxiv.org/abs/2111.05986)


  A recently proposed class of models attempts to learn latent dynamics from
high-dimensional observations, like images, using priors informed by
Hamiltonian mechanics. While these models have important potential applications
in areas like robotics or autonomous driving, there is currently no good way to
evaluate their performance: existing methods primarily rely on image
reconstruction quality, which does not always reflect the quality of the learnt
latent dynamics. In this work, we empirically highlight the problems with the
existing measures and develop a set of new measures, including a binary
indicator of whether the underlying Hamiltonian dynamics have been faithfully
captured, which we call Symplecticity Metric or SyMetric. Our measures take
advantage of the known properties of Hamiltonian dynamics and are more
discriminative of the model's ability to capture the underlying dynamics than
reconstruction error. Using SyMetric, we identify a set of architectural
choices that significantly improve the performance of a previously proposed
model for inferring latent dynamics from pixels, the Hamiltonian Generative
Network (HGN). Unlike the original HGN, the new HGN++ is able to discover an
interpretable phase space with physically meaningful latents on some datasets.
Furthermore, it is stable for significantly longer rollouts on a diverse range
of 13 datasets, producing rollouts of essentially infinite length both forward
and backwards in time with no degradation in quality on a subset of the
datasets.

    

### [[2111.05987] Tight bounds for minimum l1-norm interpolation of noisy data](http://arxiv.org/abs/2111.05987)


  We provide matching upper and lower bounds of order $\sigma^2/\log(d/n)$ for
the prediction error of the minimum $\ell_1$-norm interpolator, a.k.a. basis
pursuit. Our result is tight up to negligible terms when $d \gg n$, and is the
first to imply asymptotic consistency of noisy minimum-norm interpolation for
isotropic features and sparse ground truths. Our work complements the
literature on "benign overfitting" for minimum $\ell_2$-norm interpolation,
where asymptotic consistency can be achieved only when the features are
effectively low-dimensional.

    

### [[2111.05992] On the Use and Misuse of Absorbing States in Multi-agent Reinforcement Learning](http://arxiv.org/abs/2111.05992)


  The creation and destruction of agents in cooperative multi-agent
reinforcement learning (MARL) is a critically under-explored area of research.
Current MARL algorithms often assume that the number of agents within a group
remains fixed throughout an experiment. However, in many practical problems, an
agent may terminate before their teammates. This early termination issue
presents a challenge: the terminated agent must learn from the group's success
or failure which occurs beyond its own existence. We refer to propagating value
from rewards earned by remaining teammates to terminated agents as the
Posthumous Credit Assignment problem. Current MARL methods handle this problem
by placing these agents in an absorbing state until the entire group of agents
reaches a termination condition. Although absorbing states enable existing
algorithms and APIs to handle terminated agents without modification, practical
training efficiency and resource use problems exist.
In this work, we first demonstrate that sample complexity increases with the
quantity of absorbing states in a toy supervised learning task for a fully
connected network, while attention is more robust to variable size input. Then,
we present a novel architecture for an existing state-of-the-art MARL algorithm
which uses attention instead of a fully connected layer with absorbing states.
Finally, we demonstrate that this novel architecture significantly outperforms
the standard architecture on tasks in which agents are created or destroyed
within episodes as well as standard multi-agent coordination tasks.

    

### [[2111.06003] Detecting Fake Points of Interest from Location Data](http://arxiv.org/abs/2111.06003)


  The pervasiveness of GPS-enabled mobile devices and the widespread use of
location-based services have resulted in the generation of massive amounts of
geo-tagged data. In recent times, the data analysis now has access to more
sources, including reviews, news, and images, which also raises questions about
the reliability of Point-of-Interest (POI) data sources. While previous
research attempted to detect fake POI data through various security mechanisms,
the current work attempts to capture the fake POI data in a much simpler way.
The proposed work is focused on supervised learning methods and their
capability to find hidden patterns in location-based data. The ground truth
labels are obtained through real-world data, and the fake data is generated
using an API, so we get a dataset with both the real and fake labels on the
location data. The objective is to predict the truth about a POI using the
Multi-Layer Perceptron (MLP) method. In the proposed work, MLP based on data
classification technique is used to classify location data accurately. The
proposed method is compared with traditional classification and robust and
recent deep neural methods. The results show that the proposed method is better
than the baseline methods.

    

### [[2111.06005] Agent Spaces](http://arxiv.org/abs/2111.06005)


  Exploration is one of the most important tasks in Reinforcement Learning, but
it is not well-defined beyond finite problems in the Dynamic Programming
paradigm (see Subsection 2.4). We provide a reinterpretation of exploration
which can be applied to any online learning method. We come to this definition
by approaching exploration from a new direction. After finding that concepts of
exploration created to solve simple Markov decision processes with Dynamic
Programming are no longer broadly applicable, we reexamine exploration. Instead
of extending the ends of dynamic exploration procedures, we extend their means.
That is, rather than repeatedly sampling every state-action pair possible in a
process, we define the act of modifying an agent to itself be explorative. The
resulting definition of exploration can be applied in infinite problems and
non-dynamic learning methods, which the dynamic notion of exploration cannot
tolerate. To understand the way that modifications of an agent affect learning,
we describe a novel structure on the set of agents: a collection of distances
(see footnote 7) $d_{a} \in A$, which represent the perspectives of each agent
possible in the process. Using these distances, we define a topology and show
that many important structures in Reinforcement Learning are well behaved under
the topology induced by convergence in the agent space.

    

### [[2111.06008] Near-Optimal No-Regret Learning for Correlated Equilibria in Multi-Player General-Sum Games](http://arxiv.org/abs/2111.06008)


  Recently, Daskalakis, Fishelson, and Golowich (DFG) (NeurIPS`21) showed that
if all agents in a multi-player general-sum normal-form game employ Optimistic
Multiplicative Weights Update (OMWU), the external regret of every player is
$O(\textrm{polylog}(T))$ after $T$ repetitions of the game. We extend their
result from external regret to internal regret and swap regret, thereby
establishing uncoupled learning dynamics that converge to an approximate
correlated equilibrium at the rate of $\tilde{O}(T^{-1})$. This substantially
improves over the prior best rate of convergence for correlated equilibria of
$O(T^{-3/4})$ due to Chen and Peng (NeurIPS`20), and it is optimal -- within
the no-regret framework -- up to polylogarithmic factors in $T$.
To obtain these results, we develop new techniques for establishing
higher-order smoothness for learning dynamics involving fixed point operations.
Specifically, we establish that the no-internal-regret learning dynamics of
Stoltz and Lugosi (Mach Learn`05) are equivalently simulated by
no-external-regret dynamics on a combinatorial space. This allows us to trade
the computation of the stationary distribution on a polynomial-sized Markov
chain for a (much more well-behaved) linear transformation on an
exponential-sized set, enabling us to leverage similar techniques as DGF to
near-optimally bound the internal regret.
Moreover, we establish an $O(\textrm{polylog}(T))$ no-swap-regret bound for
the classic algorithm of Blum and Mansour (BM) (JMLR`07). We do so by
introducing a technique based on the Cauchy Integral Formula that circumvents
the more limited combinatorial arguments of DFG. In addition to shedding
clarity on the near-optimal regret guarantees of BM, our arguments provide
insights into the various ways in which the techniques by DFG can be extended
and leveraged in the analysis of more involved learning algorithms.

    

### [[2111.06011] Climate Modeling with Neural Diffusion Equations](http://arxiv.org/abs/2111.06011)


  Owing to the remarkable development of deep learning technology, there have
been a series of efforts to build deep learning-based climate models. Whereas
most of them utilize recurrent neural networks and/or graph neural networks, we
design a novel climate model based on the two concepts, the neural ordinary
differential equation (NODE) and the diffusion equation. Many physical
processes involving a Brownian motion of particles can be described by the
diffusion equation and as a result, it is widely used for modeling climate. On
the other hand, neural ordinary differential equations (NODEs) are to learn a
latent governing equation of ODE from data. In our presented method, we combine
them into a single framework and propose a concept, called neural diffusion
equation (NDE). Our NDE, equipped with the diffusion equation and one more
additional neural network to model inherent uncertainty, can learn an
appropriate latent governing equation that best describes a given climate
dataset. In our experiments with two real-world and one synthetic datasets and
eleven baselines, our method consistently outperforms existing baselines by
non-trivial margins.

    

### [[2111.06012] Kronecker Factorization for Preventing Catastrophic Forgetting in Large-scale Medical Entity Linking](http://arxiv.org/abs/2111.06012)


  Multi-task learning is useful in NLP because it is often practically
desirable to have a single model that works across a range of tasks. In the
medical domain, sequential training on tasks may sometimes be the only way to
train models, either because access to the original (potentially sensitive)
data is no longer available, or simply owing to the computational costs
inherent to joint retraining. A major issue inherent to sequential learning,
however, is catastrophic forgetting, i.e., a substantial drop in accuracy on
prior tasks when a model is updated for a new task. Elastic Weight
Consolidation is a recently proposed method to address this issue, but scaling
this approach to the modern large models used in practice requires making
strong independence assumptions about model parameters, limiting its
effectiveness. In this work, we apply Kronecker Factorization--a recent
approach that relaxes independence assumptions--to prevent catastrophic
forgetting in convolutional and Transformer-based neural networks at scale. We
show the effectiveness of this technique on the important and illustrative task
of medical entity linking across three datasets, demonstrating the capability
of the technique to be used to make efficient updates to existing methods as
new medical data becomes available. On average, the proposed method reduces
catastrophic forgetting by 51% when using a BERT-based model, compared to a 27%
reduction using standard Elastic Weight Consolidation, while maintaining
spatial complexity proportional to the number of model parameters.

    

### [[2111.06023] HMD-AMP: Protein Language-Powered Hierarchical Multi-label Deep Forest for Annotating Antimicrobial Peptides](http://arxiv.org/abs/2111.06023)


  Identifying the targets of an antimicrobial peptide is a fundamental step in
studying the innate immune response and combating antibiotic resistance, and
more broadly, precision medicine and public health. There have been extensive
studies on the statistical and computational approaches to identify (i) whether
a peptide is an antimicrobial peptide (AMP) or a non-AMP and (ii) which targets
are these sequences effective to (Gram-positive, Gram-negative, etc.). Despite
the existing deep learning methods on this problem, most of them are unable to
handle the small AMP classes (anti-insect, anti-parasite, etc.). And more
importantly, some AMPs can have multiple targets, which the previous methods
fail to consider. In this study, we build a diverse and comprehensive
multi-label protein sequence database by collecting and cleaning amino acids
from various AMP databases. To generate efficient representations and features
for the small classes dataset, we take advantage of a protein language model
trained on 250 million protein sequences. Based on that, we develop an
end-to-end hierarchical multi-label deep forest framework, HMD-AMP, to annotate
AMP comprehensively. After identifying an AMP, it further predicts what targets
the AMP can effectively kill from eleven available classes. Extensive
experiments suggest that our framework outperforms state-of-the-art models in
both the binary classification task and the multi-label classification task,
especially on the minor classes.The model is robust against reduced features
and small perturbations and produces promising results. We believe HMD-AMP
contributes to both the future wet-lab investigations of the innate structural
properties of different antimicrobial peptides and build promising empirical
underpinnings for precise medicine with antibiotics.

    

### [[2111.06025] Adapting Surprise Minimizing Reinforcement Learning Techniques for Transactive Control](http://arxiv.org/abs/2111.06025)


  Optimizing prices for energy demand response requires a flexible controller
with ability to navigate complex environments. We propose a reinforcement
learning controller with surprise minimizing modifications in its architecture.
We suggest that surprise minimization can be used to improve learning speed,
taking advantage of predictability in peoples' energy usage. Our architecture
performs well in a simulation of energy demand response. We propose this
modification to improve functionality and save in a large scale experiment.

    

### [[2111.06027] Towards Theoretical Understanding of Flexible Transmitter Networks via Approximation and Local Minima](http://arxiv.org/abs/2111.06027)


  Flexible Transmitter Network (FTNet) is a recently proposed bio-plausible
neural network and has achieved competitive performance with the
state-of-the-art models when handling temporal-spatial data. However, there
remains an open problem about the theoretical understanding of FTNet. This work
investigates the theoretical properties of one-hidden-layer FTNet from the
perspectives of approximation and local minima. Under mild assumptions, we show
that: i) FTNet is a universal approximator; ii) the approximation complexity of
FTNet can be exponentially smaller than those of real-valued neural networks
with feedforward/recurrent architectures and is of the same order in the worst
case; iii) any local minimum of FTNet is the global minimum, which suggests
that it is possible for local search algorithms to converge to the global
minimum. Our theoretical results indicate that FTNet can efficiently express
target functions and has no concern about local minima, which complements the
theoretical blank of FTNet and exhibits the possibility for ameliorating the
FTNet.

    

### [[2111.06029] Causal KL: Evaluating Causal Discovery](http://arxiv.org/abs/2111.06029)


  The two most commonly used criteria for assessing causal model discovery with
artificial data are edit-distance and Kullback-Leibler divergence, measured
from the true model to the learned model. Both of these metrics maximally
reward the true model. However, we argue that they are both insufficiently
discriminating in judging the relative merits of false models. Edit distance,
for example, fails to distinguish between strong and weak probabilistic
dependencies. KL divergence, on the other hand, rewards equally all
statistically equivalent models, regardless of their different causal claims.
We propose an augmented KL divergence, which we call Causal KL (CKL), which
takes into account causal relationships which distinguish between
observationally equivalent models. Results are presented for three variants of
CKL, showing that Causal KL works well in practice.

    

### [[2111.06032] Benefit-aware Early Prediction of Health Outcomes on Multivariate EEG Time Series](http://arxiv.org/abs/2111.06032)


  Given a cardiac-arrest patient being monitored in the ICU (intensive care
unit) for brain activity, how can we predict their health outcomes as early as
possible? Early decision-making is critical in many applications, e.g.
monitoring patients may assist in early intervention and improved care. On the
other hand, early prediction on EEG data poses several challenges: (i)
earliness-accuracy trade-off; observing more data often increases accuracy but
sacrifices earliness, (ii) large-scale (for training) and streaming (online
decision-making) data processing, and (iii) multi-variate (due to multiple
electrodes) and multi-length (due to varying length of stay of patients) time
series. Motivated by this real-world application, we present BeneFitter that
infuses the incurred savings from an early prediction as well as the cost from
misclassification into a unified domain-specific target called benefit.
Unifying these two quantities allows us to directly estimate a single target
(i.e. benefit), and importantly, dictates exactly when to output a prediction:
when benefit estimate becomes positive. BeneFitter (a) is efficient and fast,
with training time linear in the number of input sequences, and can operate in
real-time for decision-making, (b) can handle multi-variate and variable-length
time-series, suitable for patient data, and (c) is effective, providing up to
2x time-savings with equal or better accuracy as compared to competitors.

    

### [[2111.06036] CubeTR: Learning to Solve The Rubiks Cube Using Transformers](http://arxiv.org/abs/2111.06036)


  Since its first appearance, transformers have been successfully used in wide
ranging domains from computer vision to natural language processing.
Application of transformers in Reinforcement Learning by reformulating it as a
sequence modelling problem was proposed only recently. Compared to other
commonly explored reinforcement learning problems, the Rubiks cube poses a
unique set of challenges. The Rubiks cube has a single solved state for
quintillions of possible configurations which leads to extremely sparse
rewards. The proposed model CubeTR attends to longer sequences of actions and
addresses the problem of sparse rewards. CubeTR learns how to solve the Rubiks
cube from arbitrary starting states without any human prior, and after move
regularisation, the lengths of solutions generated by it are expected to be
very close to those given by algorithms used by expert human solvers. CubeTR
provides insights to the generalisability of learning algorithms to higher
dimensional cubes and the applicability of transformers in other relevant
sparse reward scenarios.

    

### [[2111.06037] Constrained Stochastic Submodular Maximization with State-Dependent Costs](http://arxiv.org/abs/2111.06037)


  In this paper, we study the constrained stochastic submodular maximization
problem with state-dependent costs. The input of our problem is a set of items
whose states (i.e., the marginal contribution and the cost of an item) are
drawn from a known probability distribution. The only way to know the realized
state of an item is to select that item. We consider two constraints, i.e.,
\emph{inner} and \emph{outer} constraints. Recall that each item has a
state-dependent cost, and the inner constraint states that the total
\emph{realized} cost of all selected items must not exceed a give budget. Thus,
inner constraint is state-dependent. The outer constraint, one the other hand,
is state-independent. It can be represented as a downward-closed family of sets
of selected items regardless of their states. Our objective is to maximize the
objective function subject to both inner and outer constraints. Under the
assumption that larger cost indicates larger "utility", we present a constant
approximate solution to this problem.

    

### [[2111.06057] Characterization of Frequent Online Shoppers using Statistical Learning with Sparsity](http://arxiv.org/abs/2111.06057)


  Developing shopping experiences that delight the customer requires businesses
to understand customer taste. This work reports a method to learn the shopping
preferences of frequent shoppers to an online gift store by combining ideas
from retail analytics and statistical learning with sparsity. Shopping activity
is represented as a bipartite graph. This graph is refined by applying
sparsity-based statistical learning methods. These methods are interpretable
and reveal insights about customers' preferences as well as products driving
revenue to the store.

    

### [[2111.06060] Exploiting the Power of Levenberg-Marquardt Optimizer with Anomaly Detection in Time Series](http://arxiv.org/abs/2111.06060)


  The Levenberg-Marquardt (LM) optimization algorithm has been widely used for
solving machine learning problems. Literature reviews have shown that the LM
can be very powerful and effective on moderate function approximation problems
when the number of weights in the network is not more than a couple of hundred.
In contrast, the LM does not seem to perform as well when dealing with pattern
recognition or classification problems, and inefficient when networks become
large (e.g. with more than 500 weights). In this paper, we exploit the true
power of LM algorithm using some real world aircraft datasets. On these
datasets most other commonly used optimizers are unable to detect the anomalies
caused by the changing conditions of the aircraft engine. The challenging
nature of the datasets are the abrupt changes in the time series data. We find
that the LM optimizer has a much better ability to approximate abrupt changes
and detect anomalies than other optimizers. We compare the performance, in
addressing this anomaly/change detection problem, of the LM and several other
optimizers. We assess the relative performance based on a range of measures
including network complexity (i.e. number of weights), fitting accuracy, over
fitting, training time, use of GPUs and memory requirement etc. We also discuss
the issue of robust LM implementation in MATLAB and Tensorflow for promoting
more popular usage of the LM algorithm and potential use of LM optimizer for
large-scale problems.

    

### [[2111.06061] Edge-Cloud Polarization and Collaboration: A Comprehensive Survey](http://arxiv.org/abs/2111.06061)


  Influenced by the great success of deep learning via cloud computing and the
rapid development of edge chips, research in artificial intelligence (AI) has
shifted to both of the computing paradigms, i.e., cloud computing and edge
computing. In recent years, we have witnessed significant progress in
developing more advanced AI models on cloud servers that surpass traditional
deep learning models owing to model innovations (e.g., Transformers, Pretrained
families), explosion of training data and soaring computing capabilities.
However, edge computing, especially edge and cloud collaborative computing, are
still in its infancy to announce their success due to the resource-constrained
IoT scenarios with very limited algorithms deployed. In this survey, we conduct
a systematic review for both cloud and edge AI. Specifically, we are the first
to set up the collaborative learning mechanism for cloud and edge modeling with
a thorough review of the architectures that enable such mechanism. We also
discuss potentials and practical experiences of some on-going advanced edge AI
topics including pretraining models, graph neural networks and reinforcement
learning. Finally, we discuss the promising directions and challenges in this
field.

    

### [[2111.06063] On the Equivalence between Neural Network and Support Vector Machine](http://arxiv.org/abs/2111.06063)


  Recent research shows that the dynamics of an infinitely wide neural network
(NN) trained by gradient descent can be characterized by Neural Tangent Kernel
(NTK) \citep{jacot2018neural}. Under the squared loss, the infinite-width NN
trained by gradient descent with an infinitely small learning rate is
equivalent to kernel regression with NTK \citep{arora2019exact}. However, the
equivalence is only known for ridge regression currently
\citep{arora2019harnessing}, while the equivalence between NN and other kernel
machines (KMs), e.g. support vector machine (SVM), remains unknown. Therefore,
in this work, we propose to establish the equivalence between NN and SVM, and
specifically, the infinitely wide NN trained by soft margin loss and the
standard soft margin SVM with NTK trained by subgradient descent. Our main
theoretical results include establishing the equivalence between NN and a broad
family of $\ell_2$ regularized KMs with finite-width bounds, which cannot be
handled by prior work, and showing that every finite-width NN trained by such
regularized loss functions is approximately a KM. Furthermore, we demonstrate
our theory can enable three practical applications, including (i)
\textit{non-vacuous} generalization bound of NN via the corresponding KM; (ii)
\textit{non-trivial} robustness certificate for the infinite-width NN (while
existing robustness verification methods would provide vacuous bounds); (iii)
intrinsically more robust infinite-width NNs than those from previous kernel
regression. Our code for the experiments are available at
\url{this https URL}.

    

### [[2111.06067] Solving Multi-Arm Bandit Using a Few Bits of Communication](http://arxiv.org/abs/2111.06067)


  The multi-armed bandit (MAB) problem is an active learning framework that
aims to select the best among a set of actions by sequentially observing
rewards. Recently, it has become popular for a number of applications over
wireless networks, where communication constraints can form a bottleneck.
Existing works usually fail to address this issue and can become infeasible in
certain applications. In this paper we address the communication problem by
optimizing the communication of rewards collected by distributed agents. By
providing nearly matching upper and lower bounds, we tightly characterize the
number of bits needed per reward for the learner to accurately learn without
suffering additional regret. In particular, we establish a generic reward
quantization algorithm, QuBan, that can be applied on top of any (no-regret)
MAB algorithm to form a new communication-efficient counterpart, that requires
only a few (as low as 3) bits to be sent per iteration while preserving the
same regret bound. Our lower bound is established via constructing hard
instances from a subgaussian distribution. Our theory is further corroborated
by numerically experiments.

    

### [[2111.06077] A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations](http://arxiv.org/abs/2111.06077)


  This two-part comprehensive survey is devoted to a computing framework most
commonly known under the names Hyperdimensional Computing and Vector Symbolic
Architectures (HDC/VSA). Both names refer to a family of computational models
that use high-dimensional distributed representations and rely on the algebraic
properties of their key operations to incorporate the advantages of structured
symbolic representations and vector distributed representations. Notable models
in the HDC/VSA family are Tensor Product Representations, Holographic Reduced
Representations, Multiply-Add-Permute, Binary Spatter Codes, and Sparse Binary
Distributed Representations but there are other models too. HDC/VSA is a highly
interdisciplinary area with connections to computer science, electrical
engineering, artificial intelligence, mathematics, and cognitive science. This
fact makes it challenging to create a thorough overview of the area. However,
due to a surge of new researchers joining the area in recent years, the
necessity for a comprehensive survey of the area has become extremely
important. Therefore, amongst other aspects of the area, this Part I surveys
important aspects such as: known computational models of HDC/VSA and
transformations of various input data types to high-dimensional distributed
representations. Part II of this survey is devoted to applications, cognitive
computing and architectures, as well as directions for future work. The survey
is written to be useful for both newcomers and practitioners.

    

### [[2111.06119] Fine-Grained Image Analysis with Deep Learning: A Survey](http://arxiv.org/abs/2111.06119)


  Fine-grained image analysis (FGIA) is a longstanding and fundamental problem
in computer vision and pattern recognition, and underpins a diverse set of
real-world applications. The task of FGIA targets analyzing visual objects from
subordinate categories, e.g., species of birds or models of cars. The small
inter-class and large intra-class variation inherent to fine-grained image
analysis makes it a challenging problem. Capitalizing on advances in deep
learning, in recent years we have witnessed remarkable progress in deep
learning powered FGIA. In this paper we present a systematic survey of these
advances, where we attempt to re-define and broaden the field of FGIA by
consolidating two fundamental fine-grained research areas -- fine-grained image
recognition and fine-grained image retrieval. In addition, we also review other
key issues of FGIA, such as publicly available benchmark datasets and related
domain-specific applications. We conclude by highlighting several research
directions and open problems which need further exploration from the community.

    

### [[2111.06142] Reducing Data Complexity using Autoencoders with Class-informed Loss Functions](http://arxiv.org/abs/2111.06142)


  Available data in machine learning applications is becoming increasingly
complex, due to higher dimensionality and difficult classes. There exists a
wide variety of approaches to measuring complexity of labeled data, according
to class overlap, separability or boundary shapes, as well as group morphology.
Many techniques can transform the data in order to find better features, but
few focus on specifically reducing data complexity. Most data transformation
methods mainly treat the dimensionality aspect, leaving aside the available
information within class labels which can be useful when classes are somehow
complex.
This paper proposes an autoencoder-based approach to complexity reduction,
using class labels in order to inform the loss function about the adequacy of
the generated variables. This leads to three different new feature learners,
Scorer, Skaler and Slicer. They are based on Fisher's discriminant ratio, the
Kullback-Leibler divergence and least-squares support vector machines,
respectively. They can be applied as a preprocessing stage for a binary
classification problem. A thorough experimentation across a collection of 27
datasets and a range of complexity and classification metrics shows that
class-informed autoencoders perform better than 4 other popular unsupervised
feature extraction techniques, especially when the final objective is using the
data for a classification task.

    

### [[2111.06146] FedGreen: Federated Learning with Fine-Grained Gradient Compression for Green Mobile Edge Computing](http://arxiv.org/abs/2111.06146)


  Federated learning (FL) enables devices in mobile edge computing (MEC) to
collaboratively train a shared model without uploading the local data. Gradient
compression may be applied to FL to alleviate the communication overheads but
current FL with gradient compression still faces great challenges. To deploy
green MEC, we propose FedGreen, which enhances the original FL with
fine-grained gradient compression to efficiently control the total energy
consumption of the devices. Specifically, we introduce the relevant operations
including device-side gradient reduction and server-side element-wise
aggregation to facilitate the gradient compression in FL. According to a public
dataset, we investigate the contributions of the compressed local gradients
with respect to different compression ratios. After that, we formulate and
tackle a learning accuracy-energy efficiency tradeoff problem where the optimal
compression ratio and computing frequency are derived for each device.
Experiments results demonstrate that given the 80% test accuracy requirement,
compared with the baseline schemes, FedGreen reduces at least 32% of the total
energy consumption of the devices.

    

### [[2111.06150] Improving Novelty Detection using the Reconstructions of Nearest Neighbours](http://arxiv.org/abs/2111.06150)


  We show that using nearest neighbours in the latent space of autoencoders
(AE) significantly improves performance of semi-supervised novelty detection in
both single and multi-class contexts. Autoencoding methods detect novelty by
learning to differentiate between the non-novel training class(es) and all
other unseen classes. Our method harnesses a combination of the reconstructions
of the nearest neighbours and the latent-neighbour distances of a given input's
latent representation. We demonstrate that our nearest-latent-neighbours (NLN)
algorithm is memory and time efficient, does not require significant data
augmentation, nor is reliant on pre-trained networks. Furthermore, we show that
the NLN-algorithm is easily applicable to multiple datasets without
modification. Additionally, the proposed algorithm is agnostic to autoencoder
architecture and reconstruction error method. We validate our method across
several standard datasets for a variety of different autoencoding architectures
such as vanilla, adversarial and variational autoencoders using either
reconstruction, residual or feature consistent losses. The results show that
the NLN algorithm grants up to a 17% increase in Area Under the Receiver
Operating Characteristics (AUROC) curve performance for the multi-class case
and 8% for single-class novelty detection.

    

### [[2111.06152] Longitudinal patient stratification of electronic health records with flexible adjustment for clinical outcomes](http://arxiv.org/abs/2111.06152)


  The increase in availability of longitudinal electronic health record (EHR)
data is leading to improved understanding of diseases and discovery of novel
phenotypes. The majority of clustering algorithms focus only on patient
trajectories, yet patients with similar trajectories may have different
outcomes. Finding subgroups of patients with different trajectories and
outcomes can guide future drug development and improve recruitment to clinical
trials. We develop a recurrent neural network autoencoder to cluster EHR data
using reconstruction, outcome, and clustering losses which can be weighted to
find different types of patient clusters. We show our model is able to discover
known clusters from both data biases and outcome differences, outperforming
baseline models. We demonstrate the model performance on $29,229$ diabetes
patients, showing it finds clusters of patients with both different
trajectories and different outcomes which can be utilized to aid clinical
decision making.

    

### [[2111.06155] A Novel Approach for Deterioration and Damage Identification in Building Structures Based on Stockwell-Transform and Deep Convolutional Neural Network](http://arxiv.org/abs/2111.06155)


  In this paper, a novel deterioration and damage identification procedure
(DIP) is presented and applied to building models. The challenge associated
with applications on these types of structures is related to the strong
correlation of responses, which gets further complicated when coping with real
ambient vibrations with high levels of noise. Thus, a DIP is designed utilizing
low-cost ambient vibrations to analyze the acceleration responses using the
Stockwell transform (ST) to generate spectrograms. Subsequently, the ST outputs
become the input of two series of Convolutional Neural Networks (CNNs)
established for identifying deterioration and damage to the building models. To
the best of our knowledge, this is the first time that both damage and
deterioration are evaluated on building models through a combination of ST and
CNN with high accuracy.

    

### [[2111.06171] Convergence and Stability of the Stochastic Proximal Point Algorithm with Momentum](http://arxiv.org/abs/2111.06171)


  Stochastic gradient descent with momentum (SGDM) is the dominant algorithm in
many optimization scenarios, including convex optimization instances and
non-convex neural network training. Yet, in the stochastic setting, momentum
interferes with gradient noise, often leading to specific step size and
momentum choices in order to guarantee convergence, set aside acceleration.
Proximal point methods, on the other hand, have gained much attention due to
their numerical stability and elasticity against imperfect tuning. Their
stochastic accelerated variants though have received limited attention: how
momentum interacts with the stability of (stochastic) proximal point methods
remains largely unstudied. To address this, we focus on the convergence and
stability of the stochastic proximal point algorithm with momentum (SPPAM), and
show that SPPAM allows a faster linear convergence rate compared to stochastic
proximal point algorithm (SPPA) with a better contraction factor, under proper
hyperparameter tuning. In terms of stability, we show that SPPAM depends on
problem constants more favorably than SGDM, allowing a wider range of step size
and momentum that lead to convergence.

    

### [[2111.06173] Uncertainty quantification of a 3D In-Stent Restenosis model with surrogate modelling](http://arxiv.org/abs/2111.06173)


  In-Stent Restenosis is a recurrence of coronary artery narrowing due to
vascular injury caused by balloon dilation and stent placement. It may lead to
the relapse of angina symptoms or to an acute coronary syndrome. An uncertainty
quantification of a model for In-Stent Restenosis with four uncertain
parameters (endothelium regeneration time, the threshold strain for smooth
muscle cells bond breaking, blood flow velocity and the percentage of
fenestration in the internal elastic lamina) is presented. Two quantities of
interest were studied, namely the average cross-sectional area and the maximum
relative area loss in a vessel. Due to the computational intensity of the model
and the number of evaluations required in the uncertainty quantification, a
surrogate model, based on Gaussian process regression with proper orthogonal
decomposition, was developed which subsequently replaced the original In-Stent
Restenosis model in the uncertainty quantification. A detailed analysis of the
uncertainty propagation and sensitivity analysis is presented. Around 11% and
16% of uncertainty are observed on the average cross-sectional area and maximum
relative area loss respectively, and the uncertainty estimates show that a
higher fenestration mainly determines uncertainty in the neointimal growth at
the initial stage of the process. On the other hand, the uncertainty in blood
flow velocity and endothelium regeneration time mainly determine the
uncertainty in the quantities of interest at the later, clinically relevant
stages of the restenosis process. The uncertainty in the threshold strain is
relatively small compared to the other uncertain parameters.

    

### [[2111.06175] Training neural networks with synthetic electrocardiograms](http://arxiv.org/abs/2111.06175)


  We present a method for training neural networks with synthetic
electrocardiograms that mimic signals produced by a wearable single lead
electrocardiogram monitor. We use domain randomization where the synthetic
signal properties such as the waveform shape, RR-intervals and noise are varied
for every training example. Models trained with synthetic data are compared to
their counterparts trained with real data. Detection of r-waves in
electrocardiograms recorded during different physical activities and in atrial
fibrillation is used to compare the models. By allowing the randomization to
increase beyond what is typically observed in the real-world data the
performance is on par or superseding the performance of networks trained with
real data. Experiments show robust performance with different seeds and
training examples on different test sets without any test set specific tuning.
The method makes possible to train neural networks using practically
free-to-collect data with accurate labels without the need for manual
annotations and it opens up the possibility of extending the use of synthetic
data on cardiac disease classification when disease specific a priori
information is used in the electrocardiogram generation. Additionally the
distribution of data can be controlled eliminating class imbalances that are
typically observed in health related data and additionally the generated data
is inherently private.

    

### [[2111.06178] BOiLS: Bayesian Optimisation for Logic Synthesis](http://arxiv.org/abs/2111.06178)


  Optimising the quality-of-results (QoR) of circuits during logic synthesis is
a formidable challenge necessitating the exploration of exponentially sized
search spaces. While expert-designed operations aid in uncovering effective
sequences, the increase in complexity of logic circuits favours automated
procedures. Inspired by the successes of machine learning, researchers adapted
deep learning and reinforcement learning to logic synthesis applications.
However successful, those techniques suffer from high sample complexities
preventing widespread adoption. To enable efficient and scalable solutions, we
propose BOiLS, the first algorithm adapting modern Bayesian optimisation to
navigate the space of synthesis operations. BOiLS requires no human
intervention and effectively trades-off exploration versus exploitation through
novel Gaussian process kernels and trust-region constrained acquisitions. In a
set of experiments on EPFL benchmarks, we demonstrate BOiLS's superior
performance compared to state-of-the-art in terms of both sample efficiency and
QoR values.

    

### [[2111.06195] Towards Domain-Independent and Real-Time Gesture Recognition Using mmWave Signal](http://arxiv.org/abs/2111.06195)


  Human gesture recognition using millimeter wave (mmWave) signals provides
attractive applications including smart home and in-car interface. While
existing works achieve promising performance under controlled settings,
practical applications are still limited due to the need of intensive data
collection, extra training efforts when adapting to new domains (i.e.
environments, persons and locations) and poor performance for real-time
recognition. In this paper, we propose DI-Gesture, a domain-independent and
real-time mmWave gesture recognition system. Specifically, we first derive the
signal variation corresponding to human gestures with spatial-temporal
processing. To enhance the robustness of the system and reduce data collecting
efforts, we design a data augmentation framework based on the correlation
between signal patterns and gesture variations. Furthermore, we propose a
dynamic window mechanism to perform gesture segmentation automatically and
accurately, thus enable real-time recognition. Finally, we build a lightweight
neural network to extract spatial-temporal information from the data for
gesture classification. Extensive experimental results show DI-Gesture achieves
an average accuracy of 97.92%, 99.18% and 98.76% for new users, environments
and locations, respectively. In real-time scenario, the accuracy of DI-Gesutre
reaches over 97% with average inference time of 2.87ms, which demonstrates the
superior robustness and effectiveness of our system.

    

### [[2111.06206] Towards Axiomatic, Hierarchical, and Symbolic Explanation for Deep Models](http://arxiv.org/abs/2111.06206)


  This paper proposes a hierarchical and symbolic And-Or graph (AOG) to
objectively explain the internal logic encoded by a well-trained deep model for
inference. We first define the objectiveness of an explainer model in game
theory, and we develop a rigorous representation of the And-Or logic encoded by
the deep model. The objectiveness and trustworthiness of the AOG explainer are
both theoretically guaranteed and experimentally verified. Furthermore, we
propose several techniques to boost the conciseness of the explanation.

    

### [[2111.06211] Model-Based Reinforcement Learning for Stochastic Hybrid Systems](http://arxiv.org/abs/2111.06211)


  Optimal control of general nonlinear systems is a central challenge in
automation. Data-driven approaches to control, enabled by powerful function
approximators, have recently had great success in tackling challenging robotic
applications. However, such methods often obscure the structure of dynamics and
control behind black-box over-parameterized representations, thus limiting our
ability to understand the closed-loop behavior. This paper adopts a
hybrid-system view of nonlinear modeling and control that lends an explicit
hierarchical structure to the problem and breaks down complex dynamics into
simpler localized units. Therefore, we consider a sequence modeling paradigm
that captures the temporal structure of the data and derive an
expecation-maximization (EM) algorithm that automatically decomposes nonlinear
dynamics into stochastic piecewise affine dynamical systems with nonlinear
boundaries. Furthermore, we show that these time-series models naturally admit
a closed-loop extension that we use to extract locally linear or polynomial
feedback controllers from nonlinear experts via imitation learning. Finally, we
introduce a novel hybrid realtive entropy policy search (Hb-REPS) technique
that incorporates the hierarchical nature of hybrid systems and optimizes a set
of time-invariant local feedback controllers derived from a locally polynomial
approximation of a global value function.

    

### [[2111.06222] ARISE: ApeRIodic SEmi-parametric Process for Efficient Markets without Periodogram and Gaussianity Assumptions](http://arxiv.org/abs/2111.06222)


  Mimicking and learning the long-term memory of efficient markets is a
fundamental problem in the interaction between machine learning and financial
economics to sequential data. Despite the prominence of this issue, current
treatments either remain largely limited to heuristic techniques or rely
significantly on periodogram or Gaussianty assumptions. In this paper, we
present the ApeRIodic SEmi-parametric (ARISE) process for investigating
efficient markets. The ARISE process is formulated as an infinite-sum function
of some known processes and employs the aperiodic spectrum estimation to
determine the key hyper-parameters, thus possessing the power and potential of
modeling the price data with long-term memory, non-stationarity, and aperiodic
spectrum. We further theoretically show that the ARISE process has the
mean-square convergence, consistency, and asymptotic normality without
periodogram and Gaussianity assumptions. In practice, we apply the ARISE
process to identify the efficiency of real-world markets. Besides, we also
provide two alternative ARISE applications: studying the long-term memorability
of various machine-learning models and developing a latent state-space model
for inference and forecasting of time series. The numerical experiments confirm
the superiority of our proposed approaches.

    

### [[2111.06223] Data-Centric Engineering: integrating simulation, machine learning and statistics. Challenges and Opportunities](http://arxiv.org/abs/2111.06223)


  Recent advances in machine learning, coupled with low-cost computation,
availability of cheap streaming sensors, data storage and cloud technologies,
has led to widespread multi-disciplinary research activity with significant
interest and investment from commercial stakeholders. Mechanistic models, based
on physical equations, and purely data-driven statistical approaches represent
two ends of the modelling spectrum. New hybrid, data-centric engineering
approaches, leveraging the best of both worlds and integrating both simulations
and data, are emerging as a powerful tool with a transformative impact on the
physical disciplines. We review the key research trends and application
scenarios in the emerging field of integrating simulations, machine learning,
and statistics. We highlight the opportunities that such an integrated vision
can unlock and outline the key challenges holding back its realisation. We also
discuss the bottlenecks in the translational aspects of the field and the
long-term upskilling requirements of the existing workforce and future
university graduates.

    

### [[2111.06236] Discovering and Explaining the Representation Bottleneck of DNNs](http://arxiv.org/abs/2111.06236)


  This paper explores the bottleneck of feature representations of deep neural
networks (DNNs), from the perspective of the complexity of interactions between
input variables encoded in DNNs. To this end, we focus on the multi-order
interaction between input variables, where the order represents the complexity
of interactions. We discover that a DNN is more likely to encode both too
simple interactions and too complex interactions, but usually fails to learn
interactions of intermediate complexity. Such a phenomenon is widely shared by
different DNNs for different tasks. This phenomenon indicates a cognition gap
between DNNs and human beings, and we call it a representation bottleneck. We
theoretically prove the underlying reason for the representation bottleneck.
Furthermore, we propose a loss to encourage/penalize the learning of
interactions of specific complexities, and analyze the representation
capacities of interactions of different complexities.

    

### [[2111.06240] Improvements to short-term weather prediction with recurrent-convolutional networks](http://arxiv.org/abs/2111.06240)


  The Weather4cast 2021 competition gave the participants a task of predicting
the time evolution of two-dimensional fields of satellite-based meteorological
data. This paper describes the author's efforts, after initial success in the
first stage of the competition, to improve the model further in the second
stage. The improvements consisted of a shallower model variant that is
competitive against the deeper version, adoption of the AdaBelief optimizer,
improved handling of one of the predicted variables where the training set was
found not to represent the validation set well, and ensembling multiple models
to improve the results further. The largest quantitative improvements to the
competition metrics can be attributed to the increased amount of training data
available in the second stage of the competition, followed by the effects of
model ensembling. Qualitative results show that the model can predict the time
evolution of the fields, including the motion of the fields over time, starting
with sharp predictions for the immediate future and blurring of the outputs in
later frames to account for the increased uncertainty.

    

### [[2111.06254] Detecting COVID-19 from Chest Computed Tomography Scans using AI-Driven Android Application](http://arxiv.org/abs/2111.06254)


  The COVID-19 (coronavirus disease 2019) pandemic affected more than 186
million people with over 4 million deaths worldwide by June 2021. The magnitude
of which has strained global healthcare systems. Chest Computed Tomography (CT)
scans have a potential role in the diagnosis and prognostication of COVID-19.
Designing a diagnostic system which is cost-efficient and convenient to operate
on resource-constrained devices like mobile phones would enhance the clinical
usage of chest CT scans and provide swift, mobile, and accessible diagnostic
capabilities. This work proposes developing a novel Android application that
detects COVID-19 infection from chest CT scans using a highly efficient and
accurate deep learning algorithm. It further creates an attention heatmap,
augmented on the segmented lung parenchyma region in the CT scans through an
algorithm developed as a part of this work, which shows the regions of
infection in the lungs. We propose a selection approach combined with
multi-threading for a faster generation of heatmaps on Android Device, which
reduces the processing time by about 93%. The neural network trained to detect
COVID-19 in this work is tested with F1 score and accuracy, both of 99.58% and
sensitivity of 99.69%, which is better than most of the results in the domain
of COVID diagnosis from CT scans. This work will be beneficial in high volume
practices and help doctors triage patients in the early diagnosis of the
COVID-19 quickly and efficiently.

    

### [[2111.06257] Branch and Bound in Mixed Integer Linear Programming Problems: A Survey of Techniques and Trends](http://arxiv.org/abs/2111.06257)


  In this paper, we surveyed the existing literature studying different
approaches and algorithms for the four critical components in the general
branch and bound (B&B) algorithm, namely, branching variable selection, node
selection, node pruning, and cutting-plane selection. However, the complexity
of the B&B algorithm always grows exponentially with respect to the increase of
the decision variable dimensions. In order to improve the speed of B&B
algorithms, learning techniques have been introduced in this algorithm
recently. We further surveyed how machine learning can be used to improve the
four critical components in B&B algorithms. In general, a supervised learning
method helps to generate a policy that mimics an expert but significantly
improves the speed. An unsupervised learning method helps choose different
methods based on the features. In addition, models trained with reinforcement
learning can beat the expert policy, given enough training and a supervised
initialization. Detailed comparisons between different algorithms have been
summarized in our survey. Finally, we discussed some future research directions
to accelerate and improve the algorithms further in the literature.

    

### [[2111.06259] Learning via Long Short-Term Memory (LSTM) network for predicting strains in Railway Bridge members under train induced vibration](http://arxiv.org/abs/2111.06259)


  Bridge health monitoring using machine learning tools has become an efficient
and cost-effective approach in recent times. In the present study, strains in
railway bridge member, available from a previous study conducted by IIT
Guwahati has been utilized. These strain data were collected from an existing
bridge while trains were passing over the bridge. LSTM is used to train the
network and to predict strains in different members of the railway bridge.
Actual field data has been used for the purpose of predicting strain in
different members using strain data from a single member, yet it has been
observed that they are quite agreeable to those of ground truth values. This is
in spite of the fact that a lot of noise existed in the data, thus showing the
efficacy of LSTM in training and predicting even from noisy field data. This
may easily open up the possibility of collecting data from the bridge with a
much lesser number of sensors and predicting the strain data in other members
through LSTM network.

    

### [[2111.06265] Dense Unsupervised Learning for Video Segmentation](http://arxiv.org/abs/2111.06265)


  We present a novel approach to unsupervised learning for video object
segmentation (VOS). Unlike previous work, our formulation allows to learn dense
feature representations directly in a fully convolutional regime. We rely on
uniform grid sampling to extract a set of anchors and train our model to
disambiguate between them on both inter- and intra-video levels. However, a
naive scheme to train such a model results in a degenerate solution. We propose
to prevent this with a simple regularisation scheme, accommodating the
equivariance property of the segmentation task to similarity transformations.
Our training objective admits efficient implementation and exhibits fast
training convergence. On established VOS benchmarks, our approach exceeds the
segmentation accuracy of previous work despite using significantly less
training data and compute power.

    

### [[2111.06266] AlphaDDA: game artificial intelligence with dynamic difficulty adjustment using AlphaZero](http://arxiv.org/abs/2111.06266)


  An artificial intelligence (AI) player has obtained superhuman skill for
games like Go, Chess, and Othello (Reversi). In other words, the AI player
becomes too strong as an opponent of human players. Then, we will not enjoy
playing board games with the AI player. In order to entertain human players,
the AI player is required to balance its skill with the human player's one
automatically. To address this issue, I propose AlphaDDA, an AI player with
dynamic difficulty adjustment based on AlphaZero. AlphaDDA consists of a deep
neural network (DNN) and Monte Carlo tree search like AlphaZero. AlphaDDA
estimates the value of the game state form only the board state using the DNN
and changes its skill according to the value. AlphaDDA can adjust AlphaDDA's
skill using only the state of a game without prior knowledge about an opponent.
In this study, AlphaDDA plays Connect4, 6x6 Othello, which is Othello using a
6x6 size board, and Othello with the other AI agents. The other AI agents are
AlphaZero, Monte Carlo tree search, Minimax algorithm, and a random player.
This study shows that AlphaDDA achieves to balance its skill with the other AI
agents except for a random player. AlphaDDA's DDA ability is derived from the
accurate estimation of the value from the state of a game. We will be able to
use the approach of AlphaDDA for any games in that the DNN can estimate the
value from the state.

    

### [[2111.06268] Raman spectroscopy in open world learning settings using the Objectosphere approach](http://arxiv.org/abs/2111.06268)


  Raman spectroscopy in combination with machine learning has significant
promise for applications in clinical settings as a rapid, sensitive, and
label-free identification method. These approaches perform well in classifying
data that contains classes that occur during the training phase. However, in
practice, there are always substances whose spectra have not yet been taken or
are not yet known and when the input data are far from the training set and
include new classes that were not seen at the training stage, a significant
number of false positives are recorded which limits the clinical relevance of
these algorithms. Here we show that these obstacles can be overcome by
implementing recently introduced Entropic Open Set and Objectosphere loss
functions. To demonstrate the efficiency of this approach, we compiled a
database of Raman spectra of 40 chemical classes separating them into 20
biologically relevant classes comprised of amino acids, 10 irrelevant classes
comprised of bio-related chemicals, and 10 classes that the Neural Network has
not seen before, comprised of a variety of other chemicals. We show that this
approach enables the network to effectively identify the unknown classes while
preserving high accuracy on the known ones, dramatically reducing the number of
false positives while preserving high accuracy on the known classes, which will
allow this technique to bridge the gap between laboratory experiments and
clinical applications.

    

### [[2111.06283] DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks](http://arxiv.org/abs/2111.06283)


  This paper studies Dropout Graph Neural Networks (DropGNNs), a new approach
that aims to overcome the limitations of standard GNN frameworks. In DropGNNs,
we execute multiple runs of a GNN on the input graph, with some of the nodes
randomly and independently dropped in each of these runs. Then, we combine the
results of these runs to obtain the final result. We prove that DropGNNs can
distinguish various graph neighborhoods that cannot be separated by message
passing GNNs. We derive theoretical bounds for the number of runs required to
ensure a reliable distribution of dropouts, and we prove several properties
regarding the expressive capabilities and limits of DropGNNs. We experimentally
validate our theoretical findings on expressiveness. Furthermore, we show that
DropGNNs perform competitively on established GNN benchmarks.

    

### [[2111.06312] Implicit SVD for Graph Representation Learning](http://arxiv.org/abs/2111.06312)


  Recent improvements in the performance of state-of-the-art (SOTA) methods for
Graph Representational Learning (GRL) have come at the cost of significant
computational resource requirements for training, e.g., for calculating
gradients via backprop over many data epochs. Meanwhile, Singular Value
Decomposition (SVD) can find closed-form solutions to convex problems, using
merely a handful of epochs. In this paper, we make GRL more computationally
tractable for those with modest hardware. We design a framework that computes
SVD of \textit{implicitly} defined matrices, and apply this framework to
several GRL tasks. For each task, we derive linear approximation of a SOTA
model, where we design (expensive-to-store) matrix $\mathbf{M}$ and train the
model, in closed-form, via SVD of $\mathbf{M}$, without calculating entries of
$\mathbf{M}$. By converging to a unique point in one step, and without
calculating gradients, our models show competitive empirical test performance
over various graphs such as article citation and biological interaction
networks. More importantly, SVD can initialize a deeper model, that is
architected to be non-linear almost everywhere, though behaves linearly when
its parameters reside on a hyperplane, onto which SVD initializes. The deeper
model can then be fine-tuned within only a few epochs. Overall, our procedure
trains hundreds of times faster than state-of-the-art methods, while competing
on empirical test performance. We open-source our implementation at:
this https URL


### [[2111.06316] Unsupervised Noise Adaptive Speech Enhancement by Discriminator-Constrained Optimal Transport](http://arxiv.org/abs/2111.06316)


  This paper presents a novel discriminator-constrained optimal transport
network (DOTN) that performs unsupervised domain adaptation for speech
enhancement (SE), which is an essential regression task in speech processing.
The DOTN aims to estimate clean references of noisy speech in a target domain,
by exploiting the knowledge available from the source domain. The domain shift
between training and testing data has been reported to be an obstacle to
learning problems in diverse fields. Although rich literature exists on
unsupervised domain adaptation for classification, the methods proposed,
especially in regressions, remain scarce and often depend on additional
information regarding the input data. The proposed DOTN approach tactically
fuses the optimal transport (OT) theory from mathematical analysis with
generative adversarial frameworks, to help evaluate continuous labels in the
target domain. The experimental results on two SE tasks demonstrate that by
extending the classical OT formulation, our proposed DOTN outperforms previous
adversarial domain adaptation frameworks in a purely unsupervised manner.

    

### [[2111.06318] Multi-agent Reinforcement Learning for Cooperative Lane Changing of Connected and Autonomous Vehicles in Mixed Traffic](http://arxiv.org/abs/2111.06318)


  Autonomous driving has attracted significant research interests in the past
two decades as it offers many potential benefits, including releasing drivers
from exhausting driving and mitigating traffic congestion, among others.
Despite promising progress, lane-changing remains a great challenge for
autonomous vehicles (AV), especially in mixed and dynamic traffic scenarios.
Recently, reinforcement learning (RL), a powerful data-driven control method,
has been widely explored for lane-changing decision makings in AVs with
encouraging results demonstrated. However, the majority of those studies are
focused on a single-vehicle setting, and lane-changing in the context of
multiple AVs coexisting with human-driven vehicles (HDVs) have received scarce
attention. In this paper, we formulate the lane-changing decision making of
multiple AVs in a mixed-traffic highway environment as a multi-agent
reinforcement learning (MARL) problem, where each AV makes lane-changing
decisions based on the motions of both neighboring AVs and HDVs. Specifically,
a multi-agent advantage actor-critic network (MA2C) is developed with a novel
local reward design and a parameter sharing scheme. In particular, a
multi-objective reward function is proposed to incorporate fuel efficiency,
driving comfort, and safety of autonomous driving. Comprehensive experimental
results, conducted under three different traffic densities and various levels
of human driver aggressiveness, show that our proposed MARL framework
consistently outperforms several state-of-the-art benchmarks in terms of
efficiency, safety and driver comfort.

    

### [[2111.06328] Stationary Behavior of Constant Stepsize SGD Type Algorithms: An Asymptotic Characterization](http://arxiv.org/abs/2111.06328)


  Stochastic approximation (SA) and stochastic gradient descent (SGD)
algorithms are work-horses for modern machine learning algorithms. Their
constant stepsize variants are preferred in practice due to fast convergence
behavior. However, constant step stochastic iterative algorithms do not
converge asymptotically to the optimal solution, but instead have a stationary
distribution, which in general cannot be analytically characterized. In this
work, we study the asymptotic behavior of the appropriately scaled stationary
distribution, in the limit when the constant stepsize goes to zero.
Specifically, we consider the following three settings: (1) SGD algorithms with
smooth and strongly convex objective, (2) linear SA algorithms involving a
Hurwitz matrix, and (3) nonlinear SA algorithms involving a contractive
operator. When the iterate is scaled by $1/\sqrt{\alpha}$, where $\alpha$ is
the constant stepsize, we show that the limiting scaled stationary distribution
is a solution of an integral equation. Under a uniqueness assumption (which can
be removed in certain settings) on this equation, we further characterize the
limiting distribution as a Gaussian distribution whose covariance matrix is the
unique solution of a suitable Lyapunov equation. For SA algorithms beyond these
cases, our numerical experiments suggest that unlike central limit theorem type
results: (1) the scaling factor need not be $1/\sqrt{\alpha}$, and (2) the
limiting distribution need not be Gaussian. Based on the numerical study, we
come up with a formula to determine the right scaling factor, and make
insightful connection to the Euler-Maruyama discretization scheme for
approximating stochastic differential equations.

    

### [[2111.06331] Towards an Efficient Voice Identification Using Wav2Vec2.0 and HuBERT Based on the Quran Reciters Dataset](http://arxiv.org/abs/2111.06331)


  Current authentication and trusted systems depend on classical and biometric
methods to recognize or authorize users. Such methods include audio speech
recognitions, eye, and finger signatures. Recent tools utilize deep learning
and transformers to achieve better results. In this paper, we develop a deep
learning constructed model for Arabic speakers identification by using
Wav2Vec2.0 and HuBERT audio representation learning tools. The end-to-end
Wav2Vec2.0 paradigm acquires contextualized speech representations learnings by
randomly masking a set of feature vectors, and then applies a transformer
neural network. We employ an MLP classifier that is able to differentiate
between invariant labeled classes. We show several experimental results that
safeguard the high accuracy of the proposed model. The experiments ensure that
an arbitrary wave signal for a certain speaker can be identified with 98% and
97.1% accuracies in the cases of Wav2Vec2.0 and HuBERT, respectively.

    

### [[2111.06334] Identification of Fine-Grained Location Mentions in Crisis Tweets](http://arxiv.org/abs/2111.06334)


  Identification of fine-grained location mentions in crisis tweets is central
in transforming situational awareness information extracted from social media
into actionable information. Most prior works have focused on identifying
generic locations, without considering their specific types. To facilitate
progress on the fine-grained location identification task, we assemble two
tweet crisis datasets and manually annotate them with specific location types.
The first dataset contains tweets from a mixed set of crisis events, while the
second dataset contains tweets from the global COVID-19 pandemic. We
investigate the performance of state-of-the-art deep learning models for
sequence tagging on these datasets, in both in-domain and cross-domain
settings.

    

### [[2111.06345] Poisoning Knowledge Graph Embeddings via Relation Inference Patterns](http://arxiv.org/abs/2111.06345)


  We study the problem of generating data poisoning attacks against Knowledge
Graph Embedding (KGE) models for the task of link prediction in knowledge
graphs. To poison KGE models, we propose to exploit their inductive abilities
which are captured through the relationship patterns like symmetry, inversion
and composition in the knowledge graph. Specifically, to degrade the model's
prediction confidence on target facts, we propose to improve the model's
prediction confidence on a set of decoy facts. Thus, we craft adversarial
additions that can improve the model's prediction confidence on decoy facts
through different inference patterns. Our experiments demonstrate that the
proposed poisoning attacks outperform state-of-art baselines on four KGE models
for two publicly available datasets. We also find that the symmetry pattern
based attacks generalize across all model-dataset combinations which indicates
the sensitivity of KGE models to this pattern.

    

### [[2111.06349] Unsupervised Part Discovery from Contrastive Reconstruction](http://arxiv.org/abs/2111.06349)


  The goal of self-supervised visual representation learning is to learn
strong, transferable image representations, with the majority of research
focusing on object or scene level. On the other hand, representation learning
at part level has received significantly less attention. In this paper, we
propose an unsupervised approach to object part discovery and segmentation and
make three contributions. First, we construct a proxy task through a set of
objectives that encourages the model to learn a meaningful decomposition of the
image into its parts. Secondly, prior work argues for reconstructing or
clustering pre-computed features as a proxy to parts; we show empirically that
this alone is unlikely to find meaningful parts; mainly because of their low
resolution and the tendency of classification networks to spatially smear out
information. We suggest that image reconstruction at the level of pixels can
alleviate this problem, acting as a complementary cue. Lastly, we show that the
standard evaluation based on keypoint regression does not correlate well with
segmentation quality and thus introduce different metrics, NMI and ARI, that
better characterize the decomposition of objects into parts. Our method yields
semantic parts which are consistent across fine-grained but visually distinct
categories, outperforming the state of the art on three benchmark datasets.
Code is available at the project page:
this https URL.

    

### [[2111.06353] Learning from Mistakes -- A Framework for Neural Architecture Search](http://arxiv.org/abs/2111.06353)


  Learning from one's mistakes is an effective human learning technique where
the learners focus more on the topics where mistakes were made, so as to deepen
their understanding. In this paper, we investigate if this human learning
strategy can be applied in machine learning. We propose a novel machine
learning method called Learning From Mistakes (LFM), wherein the learner
improves its ability to learn by focusing more on the mistakes during revision.
We formulate LFM as a three-stage optimization problem: 1) learner learns; 2)
learner re-learns focusing on the mistakes, and; 3) learner validates its
learning. We develop an efficient algorithm to solve the LFM problem. We apply
the LFM framework to neural architecture search on CIFAR-10, CIFAR-100, and
Imagenet. Experimental results strongly demonstrate the effectiveness of our
model.

    

### [[2111.06376] Quantum Model-Discovery](http://arxiv.org/abs/2111.06376)


  Quantum computing promises to speed up some of the most challenging problems
in science and engineering. Quantum algorithms have been proposed showing
theoretical advantages in applications ranging from chemistry to logistics
optimization. Many problems appearing in science and engineering can be
rewritten as a set of differential equations. Quantum algorithms for solving
differential equations have shown a provable advantage in the fault-tolerant
quantum computing regime, where deep and wide quantum circuits can be used to
solve large linear systems like partial differential equations (PDEs)
efficiently. Recently, variational approaches to solving non-linear PDEs also
with near-term quantum devices were proposed. One of the most promising general
approaches is based on recent developments in the field of scientific machine
learning for solving PDEs. We extend the applicability of near-term quantum
computers to more general scientific machine learning tasks, including the
discovery of differential equations from a dataset of measurements. We use
differentiable quantum circuits (DQCs) to solve equations parameterized by a
library of operators, and perform regression on a combination of data and
equations. Our results show a promising path to Quantum Model Discovery (QMoD),
on the interface between classical and quantum machine learning approaches. We
demonstrate successful parameter inference and equation discovery using QMoD on
different systems including a second-order, ordinary differential equation and
a non-linear, partial differential equation.

    

### [[2111.06383] Distilling Motion Planner Augmented Policies into Visual Control Policies for Robot Manipulation](http://arxiv.org/abs/2111.06383)


  Learning complex manipulation tasks in realistic, obstructed environments is
a challenging problem due to hard exploration in the presence of obstacles and
high-dimensional visual observations. Prior work tackles the exploration
problem by integrating motion planning and reinforcement learning. However, the
motion planner augmented policy requires access to state information, which is
often not available in the real-world settings. To this end, we propose to
distill a state-based motion planner augmented policy to a visual control
policy via (1) visual behavioral cloning to remove the motion planner
dependency along with its jittery motion, and (2) vision-based reinforcement
learning with the guidance of the smoothed trajectories from the behavioral
cloning agent. We evaluate our method on three manipulation tasks in obstructed
environments and compare it against various reinforcement learning and
imitation learning baselines. The results demonstrate that our framework is
highly sample-efficient and outperforms the state-of-the-art algorithms.
Moreover, coupled with domain randomization, our policy is capable of zero-shot
transfer to unseen environment settings with distractors. Code and videos are
available at this https URL


### [[2111.06387] Learning Signal-Agnostic Manifolds of Neural Fields](http://arxiv.org/abs/2111.06387)


  Deep neural networks have been used widely to learn the latent structure of
datasets, across modalities such as images, shapes, and audio signals. However,
existing models are generally modality-dependent, requiring custom
architectures and objectives to process different classes of signals. We
leverage neural fields to capture the underlying structure in image, shape,
audio and cross-modal audiovisual domains in a modality-independent manner. We
cast our task as one of learning a manifold, where we aim to infer a
low-dimensional, locally linear subspace in which our data resides. By
enforcing coverage of the manifold, local linearity, and local isometry, our
model -- dubbed GEM -- learns to capture the underlying structure of datasets
across modalities. We can then travel along linear regions of our manifold to
obtain perceptually consistent interpolations between samples, and can further
use GEM to recover points on our manifold and glean not only diverse
completions of input images, but cross-modal hallucinations of audio or image
signals. Finally, we show that by walking across the underlying manifold of
GEM, we may generate new samples in our signal domains. Code and additional
results are available at this https URL.

    

### [[2111.06389] Full-Body Visual Self-Modeling of Robot Morphologies](http://arxiv.org/abs/2111.06389)


  Internal computational models of physical bodies are fundamental to the
ability of robots and animals alike to plan and control their actions. These
"self-models" allow robots to consider outcomes of multiple possible future
actions, without trying them out in physical reality. Recent progress in fully
data-driven self-modeling has enabled machines to learn their own forward
kinematics directly from task-agnostic interaction data. However,
forward-kinema\-tics models can only predict limited aspects of the morphology,
such as the position of end effectors or velocity of joints and masses. A key
challenge is to model the entire morphology and kinematics, without prior
knowledge of what aspects of the morphology will be relevant to future tasks.
Here, we propose that instead of directly modeling forward-kinematics, a more
useful form of self-modeling is one that could answer space occupancy queries,
conditioned on the robot's state. Such query-driven self models are continuous
in the spatial domain, memory efficient, fully differentiable and kinematic
aware. In physical experiments, we demonstrate how a visual self-model is
accurate to about one percent of the workspace, enabling the robot to perform
various motion planning and control tasks. Visual self-modeling can also allow
the robot to detect, localize and recover from real-world damage, leading to
improved machine resiliency. Our project website is at:
this https URL


### [[2111.06393] Super-resolving Dark Matter Halos using Generative Deep Learning](http://arxiv.org/abs/2111.06393)


  Generative deep learning methods built upon Convolutional Neural Networks
(CNNs) provide a great tool for predicting non-linear structure in cosmology.
In this work we predict high resolution dark matter halos from large scale, low
resolution dark matter only simulations. This is achieved by mapping lower
resolution to higher resolution density fields of simulations sharing the same
cosmology, initial conditions and box-sizes. To resolve structure down to a
factor of 8 increase in mass resolution, we use a variation of U-Net with a
conditional GAN, generating output that visually and statistically matches the
high resolution target extremely well. This suggests that our method can be
used to create high resolution density output over Gpc/h box-sizes from low
resolution simulations with negligible computational effort.

    

### [[2111.06394] The Emergence of Objectness: Learning Zero-Shot Segmentation from Videos](http://arxiv.org/abs/2111.06394)


  Humans can easily segment moving objects without knowing what they are. That
objectness could emerge from continuous visual observations motivates us to
model grouping and movement concurrently from unlabeled videos. Our premise is
that a video has different views of the same scene related by moving
components, and the right region segmentation and region flow would allow
mutual view synthesis which can be checked from the data itself without any
external supervision. Our model starts with two separate pathways: an
appearance pathway that outputs feature-based region segmentation for a single
image, and a motion pathway that outputs motion features for a pair of images.
It then binds them in a conjoint representation called segment flow that pools
flow offsets over each region and provides a gross characterization of moving
regions for the entire scene. By training the model to minimize view synthesis
errors based on segment flow, our appearance and motion pathways learn region
segmentation and flow estimation automatically without building them up from
low-level edges or optical flows respectively. Our model demonstrates the
surprising emergence of objectness in the appearance pathway, surpassing prior
works on zero-shot object segmentation from an image, moving object
segmentation from a video with unsupervised test-time adaptation, and semantic
image segmentation by supervised fine-tuning. Our work is the first truly
end-to-end zero-shot object segmentation from videos. It not only develops
generic objectness for segmentation and tracking, but also outperforms
prevalent image-based contrastive learning methods without augmentation
engineering.

    

### [[2111.06395] Kalman Filtering with Adversarial Corruptions](http://arxiv.org/abs/2111.06395)


  Here we revisit the classic problem of linear quadratic estimation, i.e.
estimating the trajectory of a linear dynamical system from noisy measurements.
The celebrated Kalman filter gives an optimal estimator when the measurement
noise is Gaussian, but is widely known to break down when one deviates from
this assumption, e.g. when the noise is heavy-tailed. Many ad hoc heuristics
have been employed in practice for dealing with outliers. In a pioneering work,
Schick and Mitter gave provable guarantees when the measurement noise is a
known infinitesimal perturbation of a Gaussian and raised the important
question of whether one can get similar guarantees for large and unknown
perturbations.
In this work we give a truly robust filter: we give the first strong provable
guarantees for linear quadratic estimation when even a constant fraction of
measurements have been adversarially corrupted. This framework can model
heavy-tailed and even non-stationary noise processes. Our algorithm robustifies
the Kalman filter in the sense that it competes with the optimal algorithm that
knows the locations of the corruptions. Our work is in a challenging Bayesian
setting where the number of measurements scales with the complexity of what we
need to estimate. Moreover, in linear dynamical systems past information decays
over time. We develop a suite of new techniques to robustly extract information
across different time steps and over varying time scales.

    

### [[1907.03809] Competing Models](http://arxiv.org/abs/1907.03809)


  Different agents need to make a prediction. They observe identical data, but
have different models: they predict using different explanatory variables. We
study which agent believes they have the best predictive ability -- as measured
by the smallest subjective posterior mean squared prediction error -- and show
how it depends on the sample size. With small samples, we present results
suggesting it is an agent using a low-dimensional model. With large samples, it
is generally an agent with a high-dimensional model, possibly including
irrelevant variables, but never excluding relevant ones. We apply our results
to characterize the winning model in an auction of productive assets, to argue
that entrepreneurs and investors with simple models will be over-represented in
new sectors, and to understand the proliferation of "factors" that explain the
cross-sectional variation of expected stock returns in the asset-pricing
literature.

    

### [[1909.01795] Stochastic Submodular Probing with State-Dependent Costs](http://arxiv.org/abs/1909.01795)


  In this paper, we study a new stochastic submodular maximization problem with
state-dependent costs and rejections. The input of our problem is a budget
constraint $B$, and a set of items whose states (i.e., the marginal
contribution and the cost of an item) are drawn from a known probability
distribution. The only way to know the realized state of an item is to probe
that item. We allow rejections, i.e., after probing an item and knowing its
actual state, we must decide immediately and irrevocably whether to add that
item to our solution or not. Our objective is to sequentially probe/selet a
best group of items subject to a budget constraint on the total cost of the
selected items. We present a constant approximate solution to this problem. We
show that our solution can be extended to an online setting.

    

### [[1910.07294] Reinforcement Learning for Robotic Manipulation using Simulated Locomotion Demonstrations](http://arxiv.org/abs/1910.07294)


  Mastering robotic manipulation skills through reinforcement learning (RL)
typically requires the design of shaped reward functions. Recent developments
in this area have demonstrated that using sparse rewards, i.e. rewarding the
agent only when the task has been successfully completed, can lead to better
policies. However, state-action space exploration is more difficult in this
case. Recent RL approaches to learning with sparse rewards have leveraged
high-quality human demonstrations for the task, but these can be costly, time
consuming or even impossible to obtain. In this paper, we propose a novel and
effective approach that does not require human demonstrations. We observe that
every robotic manipulation task could be seen as involving a locomotion task
from the perspective of the object being manipulated, i.e. the object could
learn how to reach a target state on its own. In order to exploit this idea, we
introduce a framework whereby an object locomotion policy is initially obtained
using a realistic physics simulator. This policy is then used to generate
auxiliary rewards, called simulated locomotion demonstration rewards (SLDRs),
which enable us to learn the robot manipulation policy. The proposed approach
has been evaluated on 13 tasks of increasing complexity, and can achieve higher
success rate and faster learning rates compared to alternative algorithms.
SLDRs are especially beneficial for tasks like multi-object stacking and
non-rigid object manipulation.

    

### [[2002.08410] Gaussian Mixture Reduction with Composite Transportation Divergence](http://arxiv.org/abs/2002.08410)


  Gaussian mixture reduction (GMR) is the problem of approximating a high order
Gaussian mixture by one with lower order. It is widely used in density
estimation, recursive tracking in hidden Markov model, and belief propagation.
In this work, we show that the GMR can be formulated as an optimization problem
which minimizes the composite transportation divergence (CTD) between two
mixtures. The optimization problem can be solved by an easy-to-implement
Majorization-Minimization (MM) algorithm. We show that the MM algorithm
converges under general conditions. One popular computationally efficient
approach for GMR is the clustering based iterative algorithms. However, these
algorithms lack a theoretical guarantee whether they converge or attain some
optimality targets when they do. We show that existing clustering-based
algorithms are special cases of our MM algorithm can their theoretical
properties are therefore established. We further show the performance of the
clustering-based algorithms can be further improved by choosing various cost
function in the CTD. Numerical experiments are conducted to illustrate the
effectiveness of our proposed extension.

    

### [[2005.08510] Improving Learning Efficiency for Wireless Resource Allocation with Symmetric Prior](http://arxiv.org/abs/2005.08510)


  Improving learning efficiency is paramount for learning resource allocation
with deep neural networks (DNNs) in wireless communications over highly dynamic
environments. Incorporating domain knowledge into learning is a promising way
of dealing with this issue, which is an emerging topic in the wireless
community. In this article, we first briefly summarize two classes of
approaches to using domain knowledge: introducing mathematical models or prior
knowledge to deep learning. Then, we consider a kind of symmetric prior,
permutation equivariance, which widely exists in wireless tasks. To explain how
such a generic prior is harnessed to improve learning efficiency, we resort to
ranking, which jointly sorts the input and output of a DNN. We use power
allocation among subcarriers, probabilistic content caching, and interference
coordination to illustrate the improvement of learning efficiency by exploiting
the property. From the case study, we find that the required training samples
to achieve given system performance decreases with the number of subcarriers or
contents, owing to an interesting phenomenon: "sample hardening". Simulation
results show that the training samples, the free parameters in DNNs and the
training time can be reduced dramatically by harnessing the prior knowledge.
The samples required to train a DNN after ranking can be reduced by $15 \sim
2,400$ folds to achieve the same system performance as the counterpart without
using prior.

    

### [[2008.01976] Robust Deep Reinforcement Learning through Adversarial Loss](http://arxiv.org/abs/2008.01976)


  Recent studies have shown that deep reinforcement learning agents are
vulnerable to small adversarial perturbations on the agent's inputs, which
raises concerns about deploying such agents in the real world. To address this
issue, we propose RADIAL-RL, a principled framework to train reinforcement
learning agents with improved robustness against $l_p$-norm bounded adversarial
attacks. Our framework is compatible with popular deep reinforcement learning
algorithms and we demonstrate its performance with deep Q-learning, A3C and
PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to
show the effectiveness of our robust training algorithm. Our RADIAL-RL agents
consistently outperform prior methods when tested against attacks of varying
strength and are more computationally efficient to train. In addition, we
propose a new evaluation method called Greedy Worst-Case Reward (GWC) to
measure attack agnostic robustness of deep RL agents. We show that GWC can be
evaluated efficiently and is a good estimate of the reward under the worst
possible sequence of adversarial attacks. All code used for our experiments is
available at this https URL.

    

### [[2008.02066] Follow the Object: Curriculum Learning for Manipulation Tasks with Imagined Goals](http://arxiv.org/abs/2008.02066)


  Learning robot manipulation through deep reinforcement learning in
environments with sparse rewards is a challenging task. In this paper we
address this problem by introducing a notion of imaginary object goals. For a
given manipulation task, the object of interest is first trained to reach a
desired target position on its own, without being manipulated, through
physically realistic simulations. The object policy is then leveraged to build
a predictive model of plausible object trajectories providing the robot with a
curriculum of incrementally more difficult object goals to reach during
training. The proposed algorithm, Follow the Object (FO), has been evaluated on
7 MuJoCo environments requiring increasing degree of exploration, and has
achieved higher success rates compared to alternative algorithms. In
particularly challenging learning scenarios, e.g. where the object's initial
and target positions are far apart, our approach can still learn a policy
whereas competing methods currently fail.

    

### [[2010.15987] AutoAtlas: Neural Network for 3D Unsupervised Partitioning and Representation Learning](http://arxiv.org/abs/2010.15987)


  We present a novel neural network architecture called AutoAtlas for fully
unsupervised partitioning and representation learning of 3D brain Magnetic
Resonance Imaging (MRI) volumes. AutoAtlas consists of two neural network
components: one neural network to perform multi-label partitioning based on
local texture in the volume, and a second neural network to compress the
information contained within each partition. We train both of these components
simultaneously by optimizing a loss function that is designed to promote
accurate reconstruction of each partition, while encouraging spatially smooth
and contiguous partitioning, and discouraging relatively small partitions. We
show that the partitions adapt to the subject specific structural variations of
brain tissue while consistently appearing at similar spatial locations across
subjects. AutoAtlas also produces very low dimensional features that represent
local texture of each partition. We demonstrate prediction of metadata
associated with each subject using the derived feature representations and
compare the results to prediction using features derived from FreeSurfer
anatomical parcellation. Since our features are intrinsically linked to
distinct partitions, we can then map values of interest, such as
partition-specific feature importance scores onto the brain for visualization.

    

### [[2102.09390] A Machine Learning Approach for Early Detection of Fish Diseases by Analyzing Water Quality](http://arxiv.org/abs/2102.09390)


  Early detection of fish diseases and identifying the underlying causes are
crucial for farmers to take necessary steps to mitigate the potential outbreak
and thus to avert financial losses with apparent negative implications to the
national economy. Typically, fish diseases are caused by viruses and bacteria;
according to biochemical studies, the presence of certain bacteria and viruses
may affect the level of pH, DO, BOD, COD, TSS, TDS, EC, PO43-, NO3-N, and NH3-N
in water, resulting in the death of fishes. Besides, natural processes, e.g.,
photosynthesis, respiration, and decomposition, also contribute to the
alteration of water quality that adversely affects fish health. Being motivated
by the recent successes of machine learning techniques, a state-of-art machine
learning algorithm has been adopted in this paper to detect and predict the
degradation of water quality timely and accurately. Thus, it helps to take
preemptive steps against potential fish diseases. The experimental results show
high accuracy in detecting fish diseases specific to water quality based on the
algorithm with real datasets.

    

### [[2102.12967] A statistical framework for efficient out of distribution detection in deep neural networks](http://arxiv.org/abs/2102.12967)


  Background. Commonly, Deep Neural Networks (DNNs) generalize well on samples
drawn from a distribution similar to that of the training set. However, DNNs'
predictions are brittle and unreliable when the test samples are drawn from a
dissimilar distribution. This is a major concern for deployment in real-world
applications, where such behavior may come at a considerable cost, such as
industrial production lines, autonomous vehicles, or healthcare applications.
Contributions. We frame Out Of Distribution (OOD) detection in DNNs as a
statistical hypothesis testing problem. Tests generated within our proposed
framework combine evidence from the entire network. Unlike previous OOD
detection heuristics, this framework returns a $p$-value for each test sample.
It is guaranteed to maintain the Type I Error (T1E - mistakenly identifying OOD
samples as ID) for test data. Moreover, this allows combining several detectors
while maintaining the T1E. Building on this framework, we suggest a novel OOD
procedure based on low-order statistics. Our method achieves comparable or
better results than state-of-the-art methods on well-accepted OOD benchmarks,
without retraining the network parameters or assuming prior knowledge on the
test distribution -- and at a fraction of the computational cost.

    

### [[2102.13098] Toward Instance-Optimal State Certification With Incoherent Measurements](http://arxiv.org/abs/2102.13098)


  We revisit the basic problem of quantum state certification: given copies of
unknown mixed state $\rho\in\mathbb{C}^{d\times d}$ and the description of a
mixed state $\sigma$, decide whether $\sigma = \rho$ or $\|\sigma -
\rho\|_{\mathsf{tr}} \ge \epsilon$. When $\sigma$ is maximally mixed, this is
mixedness testing, and it is known that $\Omega(d^{\Theta(1)}/\epsilon^2)$
copies are necessary, where the exact exponent depends on the type of
measurements the learner can make [OW15, BCL20], and in many of these settings
there is a matching upper bound [OW15, BOW19, BCL20].
Can one avoid this $d^{\Theta(1)}$ dependence for certain kinds of mixed
states $\sigma$, e.g. ones which are approximately low rank? More ambitiously,
does there exist a simple functional $f:\mathbb{C}^{d\times
d}\to\mathbb{R}_{\ge 0}$ for which one can show that
$\Theta(f(\sigma)/\epsilon^2)$ copies are necessary and sufficient for state
certification with respect to any $\sigma$? Such instance-optimal bounds are
known in the context of classical distribution testing, e.g. [VV17].
Here we give the first bounds of this nature for the quantum setting, showing
(up to log factors) that the copy complexity for state certification using
nonadaptive incoherent measurements is essentially given by the copy complexity
for mixedness testing times the fidelity between $\sigma$ and the maximally
mixed state. Surprisingly, our bound differs substantially from instance
optimal bounds for the classical problem, demonstrating a qualitative
difference between the two settings.

    

### [[2103.10226] Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations](http://arxiv.org/abs/2103.10226)


  Explainability for machine learning models has gained considerable attention
within the research community given the importance of deploying more reliable
machine-learning systems. In computer vision applications, generative
counterfactual methods indicate how to perturb a model's input to change its
prediction, providing details about the model's decision-making. Current
methods tend to generate trivial counterfactuals about a model's decisions, as
they often suggest to exaggerate or remove the presence of the attribute being
classified. For the machine learning practitioner, these types of
counterfactuals offer little value, since they provide no new information about
undesired model or data biases. In this work, we identify the problem of
trivial counterfactual generation and we propose DiVE to alleviate it. DiVE
learns a perturbation in a disentangled latent space that is constrained using
a diversity-enforcing loss to uncover multiple valuable explanations about the
model's prediction. Further, we introduce a mechanism to prevent the model from
producing trivial explanations. Experiments on CelebA and Synbols demonstrate
that our model improves the success rate of producing high-quality valuable
explanations when compared to previous state-of-the-art methods. Code is
available at this https URL.

    

### [[2104.08736] Stochastic Optimization of Areas UnderPrecision-Recall Curves with Provable Convergence](http://arxiv.org/abs/2104.08736)


  Areas under ROC (AUROC) and precision-recall curves (AUPRC) are common
metrics for evaluating classification performance for imbalanced problems.
Compared with AUROC, AUPRC is a more appropriate metric for highly imbalanced
datasets. While stochastic optimization of AUROC has been studied extensively,
principled stochastic optimization of AUPRC has been rarely explored. In this
work, we propose a principled technical method to optimize AUPRC for deep
learning. Our approach is based on maximizing the averaged precision (AP),
which is an unbiased point estimator of AUPRC. We cast the objective into a sum
of {\it dependent compositional functions} with inner functions dependent on
random variables of the outer level. We propose efficient adaptive and
non-adaptive stochastic algorithms named SOAP with {\it provable convergence
guarantee under mild conditions} by leveraging recent advances in stochastic
compositional optimization. Extensive experimental results on image and graph
datasets demonstrate that our proposed method outperforms prior methods on
imbalanced problems in terms of AUPRC. To the best of our knowledge, our work
represents the first attempt to optimize AUPRC with provable convergence. The
SOAP has been implemented in the libAUC library at~\url{this https URL}.

    

### [[2105.05115] Analysis of One-Hidden-Layer Neural Networks via the Resolvent Method](http://arxiv.org/abs/2105.05115)


  In this work, we investigate the asymptotic spectral density of the random
feature matrix $M = Y Y^\ast$ with $Y = f(WX)$ generated by a
single-hidden-layer neural network, where $W$ and $X$ are random rectangular
matrices with i.i.d. centred entries and $f$ is a non-linear smooth function
which is applied entry-wise. We prove that the Stieltjes transform of the
limiting spectral distribution approximately satisfies a quartic
self-consistent equation, which is exactly the equation obtained by
[Pennington, Worah] and [Benigni, Péché] with the moment method. We extend
the previous results to the case of additive bias $Y=f(WX+B)$ with $B$ being an
independent rank-one Gaussian random matrix, closer modelling the neural
network infrastructures encountered in practice. Our key finding is that in the
case of additive bias it is impossible to choose an activation function
preserving the layer-to-layer singular value distribution, in sharp contrast to
the bias-free case where a simple integral constraint is sufficient to achieve
isospectrality. To obtain the asymptotics for the empirical spectral density we
follow the resolvent method from random matrix theory via the cumulant
expansion. We find that this approach is more robust and less combinatorial
than the moment method and expect that it will apply also for models where the
combinatorics of the former become intractable. The resolvent method has been
widely employed, but compared to previous works, it is applied here to
non-linear random matrices.

    

### [[2106.00474] Gaussian Processes with Differential Privacy](http://arxiv.org/abs/2106.00474)


  Gaussian processes (GPs) are non-parametric Bayesian models that are widely
used for diverse prediction tasks. Previous work in adding strong privacy
protection to GPs via differential privacy (DP) has been limited to protecting
only the privacy of the prediction targets (model outputs) but not inputs. We
break this limitation by introducing GPs with DP protection for both model
inputs and outputs. We achieve this by using sparse GP methodology and
publishing a private variational approximation on known inducing points. The
approximation covariance is adjusted to approximately account for the added
uncertainty from DP noise. The approximation can be used to compute arbitrary
predictions using standard sparse GP techniques. We propose a method for
hyperparameter learning using a private selection protocol applied to
validation set log-likelihood. Our experiments demonstrate that given
sufficient amount of data, the method can produce accurate models under strong
privacy protection.

    

### [[2106.02412] Resource Allocation in Disaggregated Data Centre Systems with Reinforcement Learning](http://arxiv.org/abs/2106.02412)


  Resource-disaggregated data centres (RDDC) propose a resource-centric, and
high-utilisation architecture for data centres (DC), avoiding resource
fragmentation and enabling arbitrarily sized resource pools to be allocated to
tasks, rather than server-sized ones. RDDCs typically impose greater demand on
the network, requiring more infrastructure and increasing cost and power, so
new resource allocation algorithms that co-manage both server and networks
resources are essential to ensure that allocation is not bottlenecked by the
network, and that requests can be served successfully with minimal networking
resources. We apply reinforcement learning (RL) to this problem for the first
time and show that an RL policy based on graph neural networks can learn
resource allocation policies end-to-end that outperform previous
hand-engineered heuristics by up to 22.0\%, 42.6\% and 22.6\% for acceptance
ratio, CPU and memory utilisation respectively, maintain performance when
scaled up to RDDC topologies with $10^2\times$ more nodes than those seen
during training and can achieve comparable performance to the best baselines
while using $5.3\times$ less network resources.

    

### [[2106.11230] Can contrastive learning avoid shortcut solutions?](http://arxiv.org/abs/2106.11230)


  The generalization of representations learned via contrastive learning
depends crucially on what features of the data are extracted. However, we
observe that the contrastive loss does not always sufficiently guide which
features are extracted, a behavior that can negatively impact the performance
on downstream tasks via "shortcuts", i.e., by inadvertently suppressing
important predictive features. We find that feature extraction is influenced by
the difficulty of the so-called instance discrimination task (i.e., the task of
discriminating pairs of similar points from pairs of dissimilar ones). Although
harder pairs improve the representation of some features, the improvement comes
at the cost of suppressing previously well represented features. In response,
we propose implicit feature modification (IFM), a method for altering positive
and negative samples in order to guide contrastive models towards capturing a
wider variety of predictive features. Empirically, we observe that IFM reduces
feature suppression, and as a result improves performance on vision and medical
imaging tasks. The code is available at: \url{this https URL}.

    

### [[2111.00698] Influential Prototypical Networks for Few Shot Learning: A Dermatological Case Study](http://arxiv.org/abs/2111.00698)


  Prototypical network (PN) is a simple yet effective few shot learning
strategy. It is a metric-based meta-learning technique where classification is
performed by computing Euclidean distances to prototypical representations of
each class. Conventional PN attributes equal importance to all samples and
generates prototypes by simply averaging the support sample embeddings
belonging to each class. In this work, we propose a novel version of PN that
attributes weights to support samples corresponding to their influence on the
support sample distribution. Influence weights of samples are calculated based
on maximum mean discrepancy (MMD) between the mean embeddings of sample
distributions including and excluding the sample. Comprehensive evaluation of
our proposed influential PN (IPNet) is performed by comparing its performance
with other baseline PNs on three different benchmark dermatological datasets.
IPNet outperforms all baseline models with compelling results across all three
datasets and various N-way, K-shot classification tasks. Findings from
cross-domain adaptation experiments further establish the robustness and
generalizability of IPNet.

    

### [[2111.06166] G-GPU: A Fully-Automated Generator of GPU-like ASIC Accelerators](http://arxiv.org/abs/2111.06166)


  Modern Systems on Chip (SoC), almost as a rule, require accelerators for
achieving energy efficiency and high performance for specific tasks that are
not necessarily well suited for execution in standard processing units.
Considering the broad range of applications and necessity for specialization,
the design of SoCs has thus become expressively more challenging. In this
paper, we put forward the concept of G-GPU, a general-purpose GPU-like
accelerator that is not application-specific but still gives benefits in energy
efficiency and throughput. Furthermore, we have identified an existing gap for
these accelerators in ASIC, for which no known automated generation
platform/tool exists. Our solution, called GPUPlanner, is an open-source
generator of accelerators, from RTL to GDSII, that addresses this gap. Our
analysis results show that our automatically generated G-GPU designs are
remarkably efficient when compared against the popular CPU architecture RISC-V,
presenting speed-ups of up to 223 times in raw performance and up to 11 times
when the metric is performance derated by area. These results are achieved by
executing a design space exploration of the GPU-like accelerators, where the
memory hierarchy is broken in a smart fashion and the logic is pipelined on
demand. Finally, tapeout-ready layouts of the G-GPU in 65nm CMOS are presented.

    

### [[2009.05334] An Open-Source Platform for High-Performance Non-Coherent On-Chip Communication](http://arxiv.org/abs/2009.05334)


  On-chip communication infrastructure is a central component of modern
systems-on-chip (SoCs), and it continues to gain importance as the number of
cores, the heterogeneity of components, and the on-chip and off-chip bandwidth
continue to grow. Decades of research on on-chip networks enabled
cache-coherent shared-memory multiprocessors. However, communication fabrics
that meet the needs of heterogeneous many-cores and accelerator-rich SoCs,
which are not, or only partially, coherent, are a much less mature research
area.
In this work, we present a modular, topology-agnostic, high-performance
on-chip communication platform. The platform includes components to build and
link subnetworks with customizable bandwidth and concurrency properties and
adheres to a state-of-the-art, industry-standard protocol. We discuss
microarchitectural trade-offs and timing/area characteristics of our modules
and show that they can be composed to build high-bandwidth (e.g., 2.5 GHz and
1024 bit data width) end-to-end on-chip communication fabrics (not only network
switches but also DMA engines and memory controllers) with high degrees of
concurrency. We design and implement a state-of-the-art ML training
accelerator, where our communication fabric scales to 1024 cores on a die,
providing 32 TB/s cross-sectional bandwidth at only 24 ns round-trip latency
between any two cores.

    

### [[2111.05979] A Visual Analytics Framework for Distributed Data Analysis Systems](http://arxiv.org/abs/2111.05979)


  This paper proposes a visual analytics framework that addresses the complex
user interactions required through a command-line interface to run analyses in
distributed data analysis systems. The visual analytics framework facilitates
the user to manage access to the distributed servers, incorporate data from the
source, run data-driven analysis, monitor the progress, and explore the result
using interactive visualizations. We provide a user interface embedded with
generalized functionalities and access protocols and integrate it with a
distributed analysis system. To demonstrate our proof of concept, we present
two use cases from the earth science and Sustainable Human Building Ecosystem
research domain.

    

### [[2111.06064] Fairness-aware Crowdsourcing of IoT Energy Services](http://arxiv.org/abs/2111.06064)


  We propose a Novel Fairness-Aware framework for Crowdsourcing Energy Services
(FACES) to efficiently provision crowdsourced IoT energy services. Typically,
efficient resource provisioning might incur an unfair resource sharing for some
requests. FACES, however, maximizes the utilization of the available energy
services by maximizing fairness across all requests. We conduct a set of
preliminary experiments to assess the effectiveness of the proposed framework
against traditional fairness-aware resource allocation algorithms. Results
demonstrate that the IoT energy utilization of FACES is better than FCFS and
similar to Max-min fair scheduling. Experiments also show that better fairness
is achieved among the provisioned requests using FACES compared toFCFS and
Max-min fair scheduling.

    

### [[2111.06157] At the Edge of a Seamless Cloud Experience](http://arxiv.org/abs/2111.06157)


  There is a growing need for low latency for many devices and users. The
traditional cloud computing paradigm can not meet this requirement,
legitimizing the need for a new paradigm. Edge computing proposes to move
computing capacities to the edge of the network, closer to where data is
produced and consumed. However, edge computing raises new challenges. At the
edge, devices are more heterogeneous than in the data centre, where everything
is optimized to achieve economies of scale. Edge devices can be mobile, like a
car, which complicates architecture with dynamic topologies. IoT devices
produce a considerable amount of data that can be processed at the Edge. In
this paper, we discuss the main challenges to be met in edge computing and
solutions to achieve a seamless cloud experience. We propose to use
technologies like containers and WebAssembly to manage applications' execution
on heterogeneous devices.

    

### [[2111.06290] Fairness, Integrity, and Privacy in a Scalable Blockchain-based Federated Learning System](http://arxiv.org/abs/2111.06290)


  Federated machine learning (FL) allows to collectively train models on
sensitive data as only the clients' models and not their training data need to
be shared. However, despite the attention that research on FL has drawn, the
concept still lacks broad adoption in practice. One of the key reasons is the
great challenge to implement FL systems that simultaneously achieve fairness,
integrity, and privacy preservation for all participating clients. To
contribute to solving this issue, our paper suggests a FL system that
incorporates blockchain technology, local differential privacy, and
zero-knowledge proofs. Our implementation of a proof-of-concept with multiple
linear regression illustrates that these state-of-the-art technologies can be
combined to a FL system that aligns economic incentives, trust, and
confidentiality requirements in a scalable and transparent system.

    

### [[2111.06339] Supporting and Controlling Complex Concurrency in Fault- Tolerant Distributed Systems](http://arxiv.org/abs/2111.06339)


  Distributed computing often gives rise to complex concurrent and interacting
activities. In some cases several concurrent activities may be working
together, i.e. cooperating, to solve a given problem; in other cases, the
activities may be independent but needing to share common system resources for
which they must compete. Many difficulties and limitations occur in the widely
advocated objects and (trans)actions model when it is supposed to support
cooperating activities. We have introduced previously the concept of
coordinated atomic (CA) actions [Xu et al. 1995]; this paper analyzes and
examines the derived objects and CA actions model for constructing
fault-tolerant distributed systems and providing unified support for both
cooperative and competitive concurrency. Our investigation reveals and
clarifies several significant problems that have not previously been studied
extensively, including the problem of ensuring consistent access to shared
objects from a joint action as opposed to a set of independent actions.
Conceptual and implementation-related solutions are proposed and illustrated.

    

### [[2008.00553] A Unifying Framework for Parallel and Distributed Processing in R using Futures](http://arxiv.org/abs/2008.00553)


  A future is a programming construct designed for concurrent and asynchronous
evaluation of code, making it particularly useful for parallel processing. The
future package implements the Future API for programming with futures in R.
This minimal API provides sufficient constructs for implementing parallel
versions of well-established, high-level map-reduce APIs. The future ecosystem
supports exception handling, output and condition relaying, parallel random
number generation, and automatic identification of globals lowering the
threshold to parallelize code. The Future API bridges parallel frontends with
parallel backends following the philosophy that end-users are the ones who
choose the parallel backend while the developer focuses on what to parallelize.
A variety of backends exist and third-party contributions meeting the
specifications, which ensure that the same code works on all backends, are
automatically supported. The future framework solves several problems not
addressed by other parallel frameworks in R.

    

### [[2101.12682] Self-stabilisation of cellular automata on tilings](http://arxiv.org/abs/2101.12682)


  Given a finite set of local constraints, we seek a cellular automaton (i.e.,
a local and uniform algorithm) that self-stabilises on the configurations that
satisfy these constraints. More precisely, starting from a finite perturbation
of a valid configuration, the cellular automaton must eventually fall back to
the space of valid configurations where it remains still. We allow the cellular
automaton to use extra symbols, but in that case, the extra symbols can also
appear in the initial finite perturbation. For several classes of local
constraints (e.g., $k$-colourings with $k\neq 3$, and North-East deterministic
constraints), we provide efficient self-stabilising cellular automata with or
without additional symbols that wash out finite perturbations in linear or
quadratic time, but also show that there are examples of local constraints for
which the self-stabilisation problem is inherently hard. We note that the
optimal self-stabilisation speed is the same for all local constraints that are
isomorphic to one another. We also consider probabilistic cellular automata
rules and show that in some cases, the use of randomness simplifies the
problem. In the deterministic case, we show that if finite perturbations are
corrected in linear time, then the cellular automaton self-stabilises even
starting from a random perturbation of a valid configuration, that is, when
errors in the initial configuration occur independently with a sufficiently low
density.

    

### [[2103.10114] AGCM-3DLF: Accelerating Atmospheric General Circulation Model via 3D Parallelization and Leap-Format](http://arxiv.org/abs/2103.10114)


  The Atmospheric General Circulation Model (AGCM) has been an important
research tool in the study of climate change for decades. As the demand for
high-resolution simulation is becoming urgent, the scalability and simulation
efficiency is faced with great challenges, especially for the
latitude-longitude mesh-based models. In this paper, we propose a highly
scalable 3D atmospheric general circulation model based on leap-format, namely
AGCM-3DLF. Firstly, it utilizes a 3D decomposition method allowing for
parallelism release in all three physical dimensions. Then the leap-format
difference computation scheme is adopted to maintain computational stability in
grid updating and avoid additional filtering at the high latitudes. A novel
shifting window communication algorithm is designed for parallelization of the
unified model. Furthermore, a series of optimizations are conducted to improve
the effectiveness of large-scale simulations. Experiment results in different
platforms demonstrate good efficiency and scalability of the model. AGCM-3DLF
scales up to the entire CAS-Xiandao1 supercomputer (196,608 CPU cores),
attaining the speed of 11.1 simulation-year-per-day (SYPD) at a high resolution
of 25KM. In addition, simulations conducted on the Sunway TaihuLight
supercomputer exhibit a 1.06 million cores scalability with 36.1% parallel
efficiency.

    

### [[2111.05884] Search in Imperfect Information Games](http://arxiv.org/abs/2111.05884)


  From the very dawn of the field, search with value functions was a
fundamental concept of computer games research. Turing's chess algorithm from
1950 was able to think two moves ahead, and Shannon's work on chess from $1950$
includes an extensive section on evaluation functions to be used within a
search. Samuel's checkers program from 1959 already combines search and value
functions that are learned through self-play and bootstrapping. TD-Gammon
improves upon those ideas and uses neural networks to learn those complex value
functions -- only to be again used within search. The combination of
decision-time search and value functions has been present in the remarkable
milestones where computers bested their human counterparts in long standing
challenging games -- DeepBlue for Chess and AlphaGo for Go. Until recently,
this powerful framework of search aided with (learned) value functions has been
limited to perfect information games. As many interesting problems do not
provide the agent perfect information of the environment, this was an
unfortunate limitation. This thesis introduces the reader to sound search for
imperfect information games.

    

### [[2111.05901] An Extensive Study of User Identification via Eye Movements across Multiple Datasets](http://arxiv.org/abs/2111.05901)


  Several studies have reported that biometric identification based on eye
movement characteristics can be used for authentication. This paper provides an
extensive study of user identification via eye movements across multiple
datasets based on an improved version of method originally proposed by George
and Routray. We analyzed our method with respect to several factors that affect
the identification accuracy, such as the type of stimulus, the IVT parameters
(used for segmenting the trajectories into fixation and saccades), adding new
features such as higher-order derivatives of eye movements, the inclusion of
blink information, template aging, age and gender.We find that three methods
namely selecting optimal IVT parameters, adding higher-order derivatives
features and including an additional blink classifier have a positive impact on
the identification accuracy. The improvements range from a few percentage
points, up to an impressive 9 % increase on one of the datasets.

    

### [[2111.05937] Recent Advances in Automated Question Answering In Biomedical Domain](http://arxiv.org/abs/2111.05937)


  The objective of automated Question Answering (QA) systems is to provide
answers to user queries in a time efficient manner. The answers are usually
found in either databases (or knowledge bases) or a collection of documents
commonly referred to as the corpus. In the past few decades there has been a
proliferation of acquisition of knowledge and consequently there has been an
exponential growth in new scientific articles in the field of biomedicine.
Therefore, it has become difficult to keep track of all the information in the
domain, even for domain experts. With the improvements in commercial search
engines, users can type in their queries and get a small set of documents most
relevant for answering their query, as well as relevant snippets from the
documents in some cases. However, it may be still tedious and time consuming to
manually look for the required information or answers. This has necessitated
the development of efficient QA systems which aim to find exact and precise
answers to user provided natural language questions in the domain of
biomedicine. In this paper, we introduce the basic methodologies used for
developing general domain QA systems, followed by a thorough investigation of
different aspects of biomedical QA systems, including benchmark datasets and
several proposed approaches, both using structured databases and collection of
texts. We also explore the limitations of current systems and explore potential
avenues for further advancement.

    

### [[2111.05944] Multi-Objective Optimization for Value-Sensitive and Sustainable Basket Recommendations](http://arxiv.org/abs/2111.05944)


  Sustainable consumption aims to minimize the environmental and societal
impact of the use of services and products. Over-consumption of services and
products leads to potential natural resource exhaustion and societal
inequalities, as access to goods and services becomes more challenging. In
everyday life, a person can simply achieve more sustainable purchases by
drastically changing their lifestyle choices and potentially going against
their personal values or wishes. Conversely, achieving sustainable consumption
while accounting for personal values is a more complex task, as potential
trade-offs arise when trying to satisfy environmental and personal goals. This
article focuses on value-sensitive design of recommender systems, which enable
consumers to improve the sustainability of their purchases while respecting
their personal values. Value-sensitive recommendations for sustainable
consumption are formalized as a multi-objective optimization problem, where
each objective represents different sustainability goals and personal values.
Novel and existing multi-objective algorithms calculate solutions to this
problem. The solutions are proposed as personalized sustainable basket
recommendations to consumers. These recommendations are evaluated on a
synthetic dataset, which comprises three established real-world datasets from
relevant scientific and organizational reports. The synthetic dataset contains
quantitative data on product prices, nutritional values and environmental
impact metrics, such as greenhouse gas emissions and water footprint. The
recommended baskets are highly similar to consumer purchased baskets and
aligned with both sustainability goals and personal values relevant to health,
expenditure and taste. Even when consumers would accept only a fraction of
recommendations, a considerable reduction of environmental impact is observed.

    

### [[2111.05988] Cross-language Information Retrieval](http://arxiv.org/abs/2111.05988)


  Two key assumptions shape the usual view of ranked retrieval: (1) that the
searcher can choose words for their query that might appear in the documents
that they wish to see, and (2) that ranking retrieved documents will suffice
because the searcher will be able to recognize those which they wished to find.
When the documents to be searched are in a language not known by the searcher,
neither assumption is true. In such cases, Cross-Language Information Retrieval
(CLIR) is needed. This chapter reviews the state of the art for cross-language
information retrieval and outlines some open research questions.

    

### [[2111.06014] AlphaGarden: Learning to Autonomously Tend a Polyculture Garden](http://arxiv.org/abs/2111.06014)


  This paper presents AlphaGarden: an autonomous polyculture garden that prunes
and irrigates living plants in a 1.5m x 3.0m physical testbed. AlphaGarden uses
an overhead camera and sensors to track the plant distribution and soil
moisture. We model individual plant growth and interplant dynamics to train a
policy that chooses actions to maximize leaf coverage and diversity. For
autonomous pruning, AlphaGarden uses two custom-designed pruning tools and a
trained neural network to detect prune points. We present results for four
60-day garden cycles. Results suggest AlphaGarden can autonomously achieve 0.96
normalized diversity with pruning shears while maintaining an average canopy
coverage of 0.86 during the peak of the cycle. Code, datasets, and supplemental
material can be found at this https URL.

    

### [[2111.06046] Music Score Expansion with Variable-Length Infilling](http://arxiv.org/abs/2111.06046)


  In this paper, we investigate using the variable-length infilling (VLI)
model, which is originally proposed to infill missing segments, to "prolong"
existing musical segments at musical boundaries. Specifically, as a case study,
we expand 20 musical segments from 12 bars to 16 bars, and examine the degree
to which the VLI model preserves musical boundaries in the expanded results
using a few objective metrics, including the Register Histogram Similarity we
newly propose. The results show that the VLI model has the potential to address
the expansion task.

    

### [[2111.06070] Explainable Sentence-Level Sentiment Analysis for Amazon Product Reviews](http://arxiv.org/abs/2111.06070)


  In this paper, we conduct a sentence level sentiment analysis on the product
reviews from Amazon and thorough analysis on the model interpretability. For
the sentiment analysis task, we use the BiLSTM model with attention mechanism.
For the study of interpretability, we consider the attention weights
distribution of single sentence and the attention weights of main aspect terms.
The model has an accuracy of up to 0.96. And we find that the aspect terms have
the same or even more attention weights than the sentimental words in
sentences.

    

### [[2111.06086] A Chinese Multi-type Complex Questions Answering Dataset over Wikidata](http://arxiv.org/abs/2111.06086)


  Complex Knowledge Base Question Answering is a popular area of research in
the past decade. Recent public datasets have led to encouraging results in this
field, but are mostly limited to English and only involve a small number of
question types and relations, hindering research in more realistic settings and
in languages other than English. In addition, few state-of-the-art KBQA models
are trained on Wikidata, one of the most popular real-world knowledge bases. We
propose CLC-QuAD, the first large scale complex Chinese semantic parsing
dataset over Wikidata to address these challenges. Together with the dataset,
we present a text-to-SPARQL baseline model, which can effectively answer
multi-type complex questions, such as factual questions, dual intent questions,
boolean questions, and counting questions, with Wikidata as the background
knowledge. We finally analyze the performance of SOTA KBQA models on this
dataset and identify the challenges facing Chinese KBQA.

    

### [[2111.06103] Towards Robust Knowledge Graph Embedding via Multi-task Reinforcement Learning](http://arxiv.org/abs/2111.06103)


  Nowadays, Knowledge graphs (KGs) have been playing a pivotal role in
AI-related applications. Despite the large sizes, existing KGs are far from
complete and comprehensive. In order to continuously enrich KGs, automatic
knowledge construction and update mechanisms are usually utilized, which
inevitably bring in plenty of noise. However, most existing knowledge graph
embedding (KGE) methods assume that all the triple facts in KGs are correct,
and project both entities and relations into a low-dimensional space without
considering noise and knowledge conflicts. This will lead to low-quality and
unreliable representations of KGs. To this end, in this paper, we propose a
general multi-task reinforcement learning framework, which can greatly
alleviate the noisy data problem. In our framework, we exploit reinforcement
learning for choosing high-quality knowledge triples while filtering out the
noisy ones. Also, in order to take full advantage of the correlations among
semantically similar relations, the triple selection processes of similar
relations are trained in a collective way with multi-task learning. Moreover,
we extend popular KGE models TransE, DistMult, ConvE and RotatE with the
proposed framework. Finally, the experimental validation shows that our
approach is able to enhance existing KGE models and can provide more robust
representations of KGs in noisy scenarios.

    

### [[2111.06213] Enhanced Fast Boolean Matching based on Sensitivity Signatures Pruning](http://arxiv.org/abs/2111.06213)


  Boolean matching is significant to digital integrated circuits design. An
exhaustive method for Boolean matching is computationally expensive even for
functions with only a few variables, because the time complexity of such an
algorithm for an n-variable Boolean function is $O(2^{n+1}n!)$. Sensitivity is
an important characteristic and a measure of the complexity of Boolean
functions. It has been used in analysis of the complexity of algorithms in
different fields. This measure could be regarded as a signature of Boolean
functions and has great potential to help reduce the search space of Boolean
matching.
In this paper, we introduce Boolean sensitivity into Boolean matching and
design several sensitivity-related signatures to enhance fast Boolean matching.
First, we propose some new signatures that relate sensitivity to Boolean
equivalence. Then, we prove that these signatures are prerequisites for Boolean
matching, which we can use to reduce the search space of the matching problem.
Besides, we develop a fast sensitivity calculation method to compute and
compare these signatures of two Boolean functions. Compared with the
traditional cofactor and symmetric detection methods, sensitivity is a series
of signatures of another dimension. We also show that sensitivity can be easily
integrated into traditional methods and distinguish the mismatched Boolean
functions faster. To the best of our knowledge, this is the first work that
introduces sensitivity to Boolean matching. The experimental results show that
sensitivity-related signatures we proposed in this paper can reduce the search
space to a very large extent, and perform up to 3x speedup over the
state-of-the-art Boolean matching methods.

    

### [[2111.06366] Answer Set Programming Made Easy](http://arxiv.org/abs/2111.06366)


  We take up an idea from the folklore of Answer Set Programming, namely that
choices, integrity constraints along with a restricted rule format is
sufficient for Answer Set Programming. We elaborate upon the foundations of
this idea in the context of the logic of Here-and-There and show how it can be
derived from the logical principle of extension by definition. We then provide
an austere form of logic programs that may serve as a normalform for logic
programs similar to conjunctive normalform in classical logic. Finally, we take
the key ideas and propose a modeling methodology for ASP beginners and
illustrate how it can be used.

    

### [[2111.06390] Full Characterization of Adaptively Strong Majority Voting in Crowdsourcing](http://arxiv.org/abs/2111.06390)


  A commonly used technique for quality control in crowdsourcing is to task the
workers with examining an item and voting on whether the item is labeled
correctly. To counteract possible noise in worker responses, one solution is to
keep soliciting votes from more workers until the difference between the
numbers of votes for the two possible outcomes exceeds a pre-specified
threshold {\delta}. We show a way to model such {\delta}-margin voting
consensus aggregation process using absorbing Markov chains. We provide
closed-form equations for the key properties of this voting process -- namely,
for the quality of the results, the expected number of votes to completion, the
variance of the required number of votes, and other moments of the
distribution. Using these results, we show further that one can adapt the value
of the threshold {\delta} to achieve quality-equivalence across voting
processes that employ workers of different accuracy levels. We then use this
result to provide efficiency-equalizing payment rates for groups of workers
characterized by different levels of response accuracy. Finally, we perform a
set of simulated experiments using both fully synthetic data as well as
real-life crowdsourced votes. We show that our theoretical model characterizes
the outcomes of the consensus aggregation process well.

    

### [[2003.06507] The Conflict Between People's Urge to Punish AI and Legal Systems](http://arxiv.org/abs/2003.06507)


  Regulating artificial intelligence (AI) has become necessary in light of its
deployment in high-risk scenarios. This paper explores the proposal to extend
legal personhood to AI and robots, which had not yet been examined through the
lens of the general public. We present two studies (N = 3,559) to obtain
people's views of electronic legal personhood vis-à-vis existing liability
models. Our study reveals people's desire to punish automated agents even
though these entities are not recognized any mental state. Furthermore, people
did not believe automated agents' punishment would fulfill deterrence nor
retribution and were unwilling to grant them legal punishment preconditions,
namely physical independence and assets. Collectively, these findings suggest a
conflict between the desire to punish automated agents and its perceived
impracticability. We conclude by discussing how future design and legal
decisions may influence how the public reacts to automated agents' wrongdoings.

    

### [[2010.01729] Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch](http://arxiv.org/abs/2010.01729)


  Spiking Neural Networks (SNNs) have recently emerged as an alternative to
deep learning owing to sparse, asynchronous and binary event (or spike) driven
processing, that can yield huge energy efficiency benefits on neuromorphic
hardware. However, training high-accuracy and low-latency SNNs from scratch
suffers from non-differentiable nature of a spiking neuron. To address this
training issue in SNNs, we revisit batch normalization and propose a temporal
Batch Normalization Through Time (BNTT) technique. Most prior SNN works till
now have disregarded batch normalization deeming it ineffective for training
temporal SNNs. Different from previous works, our proposed BNTT decouples the
parameters in a BNTT layer along the time axis to capture the temporal dynamics
of spikes. The temporally evolving learnable parameters in BNTT allow a neuron
to control its spike rate through different time-steps, enabling low-latency
and low-energy training from scratch. We conduct experiments on CIFAR-10,
CIFAR-100, Tiny-ImageNet and event-driven DVS-CIFAR10 datasets. BNTT allows us
to train deep SNN architectures from scratch, for the first time, on complex
datasets with just few 25-30 time-steps. We also propose an early exit
algorithm using the distribution of parameters in BNTT to reduce the latency at
inference, that further improves the energy-efficiency.

    

### [[2012.15234] Artificial Intelligence Development Races in Heterogeneous Settings](http://arxiv.org/abs/2012.15234)


  Regulation of advanced technologies such as Artificial Intelligence (AI) has
become increasingly more important given their potential implications such as
associated risks and ethical issues. With the great benefits promised from
being able to first supply such technologies, safety precautions and societal
consequences might be ignored or shortchanged in exchange for speeding up the
development, therefore engendering a racing narrative among the developers.
Starting from a game-theoretical model describing an idealised technology race
in a well-mixed world of players, here we investigate how different interaction
structures among race participants can alter collective choices and
requirements for regulatory actions. Our findings indicate that, when
participants portray a strong diversity in terms of connections and
peer-influence (e.g., when scale-free networks shape interactions among
parties), the conflicts that exist in homogeneous settings are significantly
reduced, thereby lessening the need for regulatory actions. Furthermore, our
results suggest that technology governance and regulation may profit from the
world's patent heterogeneity and inequality among firms and nations, so as to
enable the design and implementation of meticulous interventions on a minority
of participants, which is capable of influencing an entire population towards
an ethical and sustainable use of advanced technologies.

    

### [[2101.12051] Edge Federated Learning Via Unit-Modulus Over-The-Air Computation](http://arxiv.org/abs/2101.12051)


  Edge federated learning (FL) is an emerging paradigm that trains a global
parametric model from distributed datasets based on wireless communications.
This paper proposes a unit-modulus over-the-air computation (UMAirComp)
framework to facilitate efficient edge federated learning, which simultaneously
uploads local model parameters and updates global model parameters via analog
beamforming. The proposed framework avoids sophisticated baseband signal
processing, leading to low communication delays and implementation costs.
Training loss bounds of UMAirComp FL systems are derived and two low-complexity
large-scale optimization algorithms, termed penalty alternating minimization
(PAM) and accelerated gradient projection (AGP), are proposed to minimize the
nonconvex nonsmooth loss bound. Simulation results show that the proposed
UMAirComp framework with PAM algorithm achieves a smaller mean square error of
model parameters' estimation, training loss, and test error compared with other
benchmark schemes. Moreover, the proposed UMAirComp framework with AGP
algorithm achieves satisfactory performance while reduces the computational
complexity by orders of magnitude compared with existing optimization
algorithms. Finally, we demonstrate the implementation of UMAirComp in a
vehicle-to-everything autonomous driving simulation platform. It is found that
autonomous driving tasks are more sensitive to model parameter errors than
other tasks since the neural networks for autonomous driving contain sparser
model parameters.

    

### [[2105.14426] ICDAR 2021 Competition on Scientific Table Image Recognition to LaTeX](http://arxiv.org/abs/2105.14426)


  Tables present important information concisely in many scientific documents.
Visual features like mathematical symbols, equations, and spanning cells make
structure and content extraction from tables embedded in research documents
difficult. This paper discusses the dataset, tasks, participants' methods, and
results of the ICDAR 2021 Competition on Scientific Table Image Recognition to
LaTeX. Specifically, the task of the competition is to convert a tabular image
to its corresponding LaTeX source code. We proposed two subtasks. In Subtask 1,
we ask the participants to reconstruct the LaTeX structure code from an image.
In Subtask 2, we ask the participants to reconstruct the LaTeX content code
from an image. This report describes the datasets and ground truth
specification, details the performance evaluation metrics used, presents the
final results, and summarizes the participating methods. Submission by team
VCGroup got the highest Exact Match accuracy score of 74% for Subtask 1 and 55%
for Subtask 2, beating previous baselines by 5% and 12%, respectively. Although
improvements can still be made to the recognition capabilities of models, this
competition contributes to the development of fully automated table recognition
systems by challenging practitioners to solve problems under specific
constraints and sharing their approaches; the platform will remain available
for post-challenge submissions at
this https URL .

    

### [[2106.15047] Benchmarking Knowledge-driven Zero-shot Learning](http://arxiv.org/abs/2106.15047)


  External knowledge (a.k.a. side information) plays a critical role in
zero-shot learning (ZSL) which aims to predict with unseen classes that have
never appeared in training data. Several kinds of external knowledge, such as
text and attribute, have been widely investigated, but they alone are limited
with incomplete semantics. Some very recent studies thus propose to use
Knowledge Graph (KG) due to its high expressivity and compatibility for
representing kinds of knowledge. However, the ZSL community is still in short
of standard benchmarks for studying and comparing different external knowledge
settings and different KG-based ZSL methods. In this paper, we proposed six
resources covering three tasks, i.e., zero-shot image classification (ZS-IMGC),
zero-shot relation extraction (ZS-RE), and zero-shot KG completion (ZS-KGC).
Each resource has a normal ZSL benchmark and a KG containing semantics ranging
from text to attribute, from relational knowledge to logical expressions. We
have clearly presented these resources including their construction,
statistics, data formats and usage cases w.r.t. different ZSL methods. More
importantly, we have conducted a comprehensive benchmarking study, with two
general and state-of-the-art methods, two setting-specific methods and one
interpretable method. We discussed and compared different ZSL paradigms w.r.t.
different external knowledge settings, and found that our resources have great
potential for developing more advanced ZSL methods and more solutions for
applying KGs for augmenting machine learning. All the resources are available
at this https URL.

    

### [[2111.05923] The Decidability and Complexity of Interleaved Bidirected Dyck Reachability](http://arxiv.org/abs/2111.05923)


  Dyck reachability is the standard formulation of a large domain of static
analyses, as it achieves the sweet spot between precision and efficiency, and
has thus been studied extensively. Interleaved Dyck reachability (denoted
$D_k\odot D_k$) uses two Dyck languages for increased precision (e.g., context
and field sensitivity) but is well-known to be undecidable. As many static
analyses yield a certain type of bidirected graphs, they give rise to
interleaved bidirected Dyck reachability problems. Although these problems have
seen numerous applications, their decidability and complexity has largely
remained open. In a recent work, Li et al. made the first steps in this
direction, showing that (i) $D_1\odot D_1$ reachability (i.e., when both Dyck
languages are over a single parenthesis and act as counters) is computable in
$O(n^7)$ time, while (ii) $D_k\odot D_k$ reachability is NP-hard.
In this work we address the decidability and complexity of all variants of
interleaved bidirected Dyck reachability. First, we show that $D_1\odot D_1$
reachability can be computed in $O(n^3\cdot \alpha(n))$ time, significantly
improving over the existing $O(n^7)$ bound. Second, we show that $D_k\odot D_1$
reachability (i.e., when one language acts as a counter) is decidable, in
contrast to the non-bidirected case where decidability is open. We further
consider $D_k\odot D_1$ reachability where the counter remains linearly
bounded. Our third result shows that this bounded variant can be solved in
$O(n^2\cdot \alpha(n))$ time, while our fourth result shows that the problem
has a (conditional) quadratic lower bound, and thus our upper bound is
essentially optimal. Fifth, we show that full $D_k\odot D_k$ reachability is
undecidable. This improves the recent NP-hardness lower-bound, and shows that
the problem is equivalent to the non-bidirected case.

    