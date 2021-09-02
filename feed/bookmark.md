
## 2021-9-2

### [<title>How to use Early Stopping in an sklearn pipeline when the pipeline handles type conversion - XGBoost</title>](https://discuss.xgboost.ai/t/how-to-use-early-stopping-in-an-sklearn-pipeline-when-the-pipeline-handles-type-conversion/1704/2)

### [<title>Conditional jump or move depends on uninitialised value(s) - XGBoost</title>](https://discuss.xgboost.ai/t/conditional-jump-or-move-depends-on-uninitialised-value-s/2456/1)

### [[2109.00228] Deployable Networks for Public Safety in 5G and Beyond: A Coverage and Interference Study](http://arxiv.org/abs/2109.00228)


  Deployable networks are foreseen to be one of the key technologies for public
safety in fifth generation (5G) mobile communications and beyond. They can be
used to complement the existing public cellular networks to provide temporary
and on-demand connectivity in emergency situations. However, operating
deployable networks in coexistence with public cellular networks can be
challenging from an interference perspective. To gain insights on the
deployment strategy for deployable networks, in this article, we present an
extensive numerical study of coverage and interference analysis, considering
four different co-existence scenarios and different types of deployable base
stations (BSs), i.e., BS on a truck and BS on an Unmanned Aerial Vehicle (UAV).
Our simulation results show that deploying deployable BSs in rural scenarios
can provide good coverage to meet the service requirement for mission critical
(MC) users. In addition, the interference impact is only substantial when the
deployable and public networks are close to each other. Finally, allowing the
MC users to access the public network can be of vital importance to guarantee
their service when the interference level between public and deployable network
is very high.

    

### [[2109.00369] Decentralized Collaborative Video Caching in 5G Small-Cell Base Station Cellular Networks](http://arxiv.org/abs/2109.00369)


  We consider the problem of video caching across a set of 5G small-cell base
stations (SBS) connected to each other over a high-capacity short-delay
back-haul link, and linked to a remote server over a long-delay connection.
Even though the problem of minimizing the overall video delivery delay is
NP-hard, the Collaborative Caching Algorithm (CCA) that we present can
efficiently compute a solution close to the optimal, where the degree of
sub-optimality depends on the worst case video-to-cache size ratio. The
algorithm is naturally amenable to distributed implementation that requires
zero explicit coordination between the SBSs, and runs in $O(N + K \log K)$
time, where $N$ is the number of SBSs (caches) and $K$ the maximum number of
videos. We extend CCA to an online setting where the video popularities are not
known a priori but are estimated over time through a limited amount of periodic
information sharing between SBSs. We demonstrate that our algorithm closely
approaches the optimal integral caching solution as the cache size increases.
Moreover, via simulations carried out on real video access traces, we show that
our algorithm effectively uses the SBS caches to reduce the video delivery
delay and conserve the remote server's bandwidth, and that it outperforms two
other reference caching methods adapted to our system setting.

    

### [[2109.00376] Clover: an Anonymous Transaction Relay Protocol for the Bitcoin P2P Network](http://arxiv.org/abs/2109.00376)


  The Bitcoin P2P network currently represents a reference benchmark for modern
cryptocurrencies. Its underlying protocol defines how transactions and blocks
are distributed through all participating nodes. To protect user privacy, the
identity of the node originating a message is kept hidden. However, an
adversary observing the whole network can analyze the spread pattern of a
transaction to trace it back to its source. This is possible thanks to the
so-called rumor centrality, which is caused by the symmetry in the spreading of
gossip-like protocols.
Recent works try to address this issue by breaking the symmetry of the
Diffusion protocol, currently used in Bitcoin, and leveraging proxied
broadcast. Nonetheless, the complexity of their design can be a barrier to
their adoption in real life. In this work, we propose Clover, a novel
transaction relay protocol that protects the source of transaction messages
with a simple, yet effective, design. Compared to previous solutions, our
protocol does not require building propagation graphs, and reduces the ability
of the adversary to gain precision by opening multiple connections towards the
same node. Experimental results show that the deanonymization accuracy of an
eavesdropper adversary against Clover is up to 10 times smaller compared to
Diffusion.

    

### [[2109.00395] The Internet with Privacy Policies: Measuring The Web Upon Consent](http://arxiv.org/abs/2109.00395)


  To protect users' privacy, legislators have regulated the usage of tracking
technologies, mandating the acquisition of users' consent before collecting
data. Consequently, websites started showing more and more consent management
modules -- i.e., Privacy Banners -- the visitors have to interact with to
access the website content. They challenge the automatic collection of Web
measurements, primarily to monitor the extensiveness of tracking technologies
but also to measure Web performance in the wild. Privacy Banners in fact limit
crawlers from observing the actual website content.
In this paper, we present a thorough measurement campaign focusing on popular
websites in Europe and the US, visiting both landing and internal pages from
different countries around the world. We engineer Priv-Accept, a Web crawler
able to accept the privacy policies, as most users would do in practice. This
let us compare how webpages change before and after. Our results show that all
measurements performed not dealing with the Privacy Banners offer a very biased
and partial view of the Web. After accepting the privacy policies, we observe
an increase of up to 70 trackers, which in turn slows down the webpage load
time by a factor of 2x-3x.

    

### [[2109.00020] Working Memory Connections for LSTM](http://arxiv.org/abs/2109.00020)


  Recurrent Neural Networks with Long Short-Term Memory (LSTM) make use of
gating mechanisms to mitigate exploding and vanishing gradients when learning
long-term dependencies. For this reason, LSTMs and other gated RNNs are widely
adopted, being the standard de facto for many sequence modeling tasks. Although
the memory cell inside the LSTM contains essential information, it is not
allowed to influence the gating mechanism directly. In this work, we improve
the gate potential by including information coming from the internal cell
state. The proposed modification, named Working Memory Connection, consists in
adding a learnable nonlinear projection of the cell content into the network
gates. This modification can fit into the classical LSTM gates without any
assumption on the underlying task, being particularly effective when dealing
with longer sequences. Previous research effort in this direction, which goes
back to the early 2000s, could not bring a consistent improvement over vanilla
LSTM. As part of this paper, we identify a key issue tied to previous
connections that heavily limits their effectiveness, hence preventing a
successful integration of the knowledge coming from the internal cell state. We
show through extensive experimental evaluation that Working Memory Connections
constantly improve the performance of LSTMs on a variety of tasks. Numerical
results suggest that the cell state contains useful information that is worth
including in the gate structure.

    

### [[2109.00024] Machine-Learning media bias](http://arxiv.org/abs/2109.00024)


  We present an automated method for measuring media bias. Inferring which
newspaper published a given article, based only on the frequencies with which
it uses different phrases, leads to a conditional probability distribution
whose analysis lets us automatically map newspapers and phrases into a bias
space. By analyzing roughly a million articles from roughly a hundred
newspapers for bias in dozens of news topics, our method maps newspapers into a
two-dimensional bias landscape that agrees well with previous bias
classifications based on human judgement. One dimension can be interpreted as
traditional left-right bias, the other as establishment bias. This means that
although news bias is inherently political, its measurement need not be.

    

### [[2109.00025] Sense representations for Portuguese: experiments with sense embeddings and deep neural language models](http://arxiv.org/abs/2109.00025)


  Sense representations have gone beyond word representations like Word2Vec,
GloVe and FastText and achieved innovative performance on a wide range of
natural language processing tasks. Although very useful in many applications,
the traditional approaches for generating word embeddings have a strict
drawback: they produce a single vector representation for a given word ignoring
the fact that ambiguous words can assume different meanings. In this paper, we
explore unsupervised sense representations which, different from traditional
word embeddings, are able to induce different senses of a word by analyzing its
contextual semantics in a text. The unsupervised sense representations
investigated in this paper are: sense embeddings and deep neural language
models. We present the first experiments carried out for generating sense
embeddings for Portuguese. Our experiments show that the sense embedding model
(Sense2vec) outperformed traditional word embeddings in syntactic and semantic
analogies task, proving that the language resource generated here can improve
the performance of NLP tasks in Portuguese. We also evaluated the performance
of pre-trained deep neural language models (ELMo and BERT) in two transfer
learning approaches: feature based and fine-tuning, in the semantic textual
similarity task. Our experiments indicate that the fine tuned Multilingual and
Portuguese BERT language models were able to achieve better accuracy than the
ELMo model and baselines.

    

### [[2109.00036] Half-Space and Box Constraints as NUV Priors: First Results](http://arxiv.org/abs/2109.00036)


  Normals with unknown variance (NUV) can represent many useful priors and
blend well with Gaussian models and message passing algorithms. NUV
representations of sparsifying priors have long been known, and NUV
representations of binary (and M-level) priors have been proposed very
recently. In this document, we propose NUV representations of half-space
constraints and box constraints, which allows to add such constraints to any
linear Gaussian model with any of the previously known NUV priors without
affecting the computational tractability.

    

### [[2109.00046] Scalable Spatiotemporally Varying Coefficient Modeling with Bayesian Kernelized Tensor Regression](http://arxiv.org/abs/2109.00046)


  As a regression technique in spatial statistics, spatiotemporally varying
coefficient model (STVC) is an important tool to discover nonstationary and
interpretable response-covariate associations over both space and time.
However, it is difficult to apply STVC for large-scale spatiotemporal analysis
due to the high computational cost. To address this challenge, we summarize the
spatiotemporally varying coefficients using a third-order tensor structure and
propose to reformulate the spatiotemporally varying coefficient model as a
special low-rank tensor regression problem. The low-rank decomposition can
effectively model the global patterns of the large data with substantially
reduced number of parameters. To further incorporate the local spatiotemporal
dependencies among the samples, we place Gaussian process (GP) priors on the
spatial and temporal factor matrices to better encode local spatial and
temporal processes on each factor component. We refer to the overall framework
as Bayesian Kernelized Tensor Regression (BKTR). For model inference, we
develop an efficient Markov chain Monte Carlo (MCMC) algorithm, which uses
Gibbs sampling to update factor matrices and slice sampling to update kernel
hyperparameters. We conduct extensive experiments on both synthetic and
real-world data sets, and our results confirm the superior performance and
efficiency of BKTR for model estimation and parameter inference.

    

### [[2109.00060] Data-Driven Reduced-Order Modeling of Spatiotemporal Chaos with Neural Ordinary Differential Equations](http://arxiv.org/abs/2109.00060)


  Dissipative partial differential equations that exhibit chaotic dynamics tend
to evolve to attractors that exist on finite-dimensional manifolds. We present
a data-driven reduced order modeling method that capitalizes on this fact by
finding the coordinates of this manifold and finding an ordinary differential
equation (ODE) describing the dynamics in this coordinate system. The manifold
coordinates are discovered using an undercomplete autoencoder -- a neural
network (NN) that reduces then expands dimension. Then the ODE, in these
coordinates, is approximated by a NN using the neural ODE framework. Both of
these methods only require snapshots of data to learn a model, and the data can
be widely and/or unevenly spaced. We apply this framework to the
Kuramoto-Sivashinsky for different domain sizes that exhibit chaotic dynamics.
With this system, we find that dimension reduction improves performance
relative to predictions in the ambient space, where artifacts arise. Then, with
the low-dimensional model, we vary the training data spacing and find excellent
short- and long-time statistical recreation of the true dynamics for widely
spaced data (spacing of ~0.7 Lyapunov times). We end by comparing performance
with various degrees of dimension reduction, and find a "sweet spot" in terms
of performance vs. dimension.

    

### [[2109.00074] Effectiveness of Deep Networks in NLP using BiDAF as an example architecture](http://arxiv.org/abs/2109.00074)


  Question Answering with NLP has progressed through the evolution of advanced
model architectures like BERT and BiDAF and earlier word, character, and
context-based embeddings. As BERT has leapfrogged the accuracy of models, an
element of the next frontier can be the introduction of deep networks and an
effective way to train them. In this context, I explored the effectiveness of
deep networks focussing on the model encoder layer of BiDAF. BiDAF with its
heterogeneous layers provides the opportunity not only to explore the
effectiveness of deep networks but also to evaluate whether the refinements
made in lower layers are additive to the refinements made in the upper layers
of the model architecture. I believe the next greatest model in NLP will in
fact fold in a solid language modeling like BERT with a composite architecture
which will bring in refinements in addition to generic language modeling and
will have a more extensive layered architecture. I experimented with the Bypass
network, Residual Highway network, and DenseNet architectures. In addition, I
evaluated the effectiveness of ensembling the last few layers of the network. I
also studied the difference character embeddings make in adding them to the
word embeddings, and whether the effects are additive with deep networks. My
studies indicate that deep networks are in fact effective in giving a boost.
Also, the refinements in the lower layers like embeddings are passed on
additively to the gains made through deep networks.

    

### [[2109.00087] It's not Rocket Science : Interpreting Figurative Language in Narratives](http://arxiv.org/abs/2109.00087)


  Figurative language is ubiquitous in English. Yet, the vast majority of NLP
research focuses on literal language. Existing text representations by design
rely on compositionality, while figurative language is often non-compositional.
In this paper, we study the interpretation of two non-compositional figurative
languages (idioms and similes). We collected datasets of fictional narratives
containing a figurative expression along with crowd-sourced plausible and
implausible continuations relying on the correct interpretation of the
expression. We then trained models to choose or generate the plausible
continuation. Our experiments show that models based solely on pre-trained
language models perform substantially worse than humans on these tasks. We
additionally propose knowledge-enhanced models, adopting human strategies for
interpreting figurative language: inferring meaning from the context and
relying on the constituent word's literal meanings. The knowledge-enhanced
models improve the performance on both the discriminative and generative tasks,
further bridging the gap from human performance.

    

### [[2109.00092] GFINNs: GENERIC Formalism Informed Neural Networks for Deterministic and Stochastic Dynamical Systems](http://arxiv.org/abs/2109.00092)


  We propose the GENERIC formalism informed neural networks (GFINNs) that obey
the symmetric degeneracy conditions of the GENERIC formalism. GFINNs comprise
two modules, each of which contains two components. We model each component
using a neural network whose architecture is designed to satisfy the required
conditions. The component-wise architecture design provides flexible ways of
leveraging available physics information into neural networks. We prove
theoretically that GFINNs are sufficiently expressive to learn the underlying
equations, hence establishing the universal approximation theorem. We
demonstrate the performance of GFINNs in three simulation problems: gas
containers exchanging heat and volume, thermoelastic double pendulum and the
Langevin dynamics. In all the examples, GFINNs outperform existing methods,
hence demonstrating good accuracy in predictions for both deterministic and
stochastic systems.

    

### [[2109.00095] Quantized convolutional neural networks through the lens of partial differential equations](http://arxiv.org/abs/2109.00095)


  Quantization of Convolutional Neural Networks (CNNs) is a common approach to
ease the computational burden involved in the deployment of CNNs, especially on
low-resource edge devices. However, fixed-point arithmetic is not natural to
the type of computations involved in neural networks. In this work, we explore
ways to improve quantized CNNs using PDE-based perspective and analysis. First,
we harness the total variation (TV) approach to apply edge-aware smoothing to
the feature maps throughout the network. This aims to reduce outliers in the
distribution of values and promote piece-wise constant maps, which are more
suitable for quantization. Secondly, we consider symmetric and stable variants
of common CNNs for image classification, and Graph Convolutional Networks
(GCNs) for graph node-classification. We demonstrate through several
experiments that the property of forward stability preserves the action of a
network under different quantization rates. As a result, stable quantized
networks behave similarly to their non-quantized counterparts even though they
rely on fewer parameters. We also find that at times, stability even aids in
improving accuracy. These properties are of particular interest for sensitive,
resource-constrained, low-power or real-time applications like autonomous
driving.

    

### [[2109.00101] Position-based Hash Embeddings For Scaling Graph Neural Networks](http://arxiv.org/abs/2109.00101)


  Graph Neural Networks (GNNs) bring the power of deep representation learning
to graph and relational data and achieve state-of-the-art performance in many
applications. GNNs compute node representations by taking into account the
topology of the node's ego-network and the features of the ego-network's nodes.
When the nodes do not have high-quality features, GNNs learn an embedding layer
to compute node embeddings and use them as input features. However, the size of
the embedding layer is linear to the graph size and does not scale to graphs
with hundreds of millions of nodes. To reduce the memory associated with this
embedding layer, hashing-based approaches, commonly used in applications like
NLP and recommender systems, can potentially be used. However, a direct
application of these ideas fails to exploit the fact that in many real-world
graphs, nodes that are topologically close will tend to be related to each
other (homophily) and as such their representations will be similar.
In this work, we present approaches that take advantage of the nodes'
position in the graph to dramatically reduce the memory required, with minimal
if any degradation in the quality of the resulting GNN model. Our approaches
decompose a node's embedding into two components: a position-specific component
and a node-specific component. The position-specific component models homophily
and the node-specific component models the node-to-node variation. Extensive
experiments using different datasets and GNN models show that in nearly all
cases, our methods are able to reduce the memory requirements by 86% to 97%
while achieving better classification accuracy than other competing approaches,
including the full embeddings.

    

### [[2109.00103] Automatic non-invasive Cough Detection based on Accelerometer and Audio Signals](http://arxiv.org/abs/2109.00103)


  We present an automatic non-invasive way of detecting cough events based on
both accelerometer and audio signals.
The acceleration signals are captured by a smartphone firmly attached to the
patient's bed, using its integrated accelerometer.
The audio signals are captured simultaneously by the same smartphone using an
external microphone.
We have compiled a manually-annotated dataset containing such
simultaneously-captured acceleration and audio signals for approximately 6000
cough and 68000 non-cough events from 14 adult male patients in a tuberculosis
clinic.
LR, SVM and MLP are evaluated as baseline classifiers and compared with deep
architectures such as CNN, LSTM, and Resnet50 using a leave-one-out
cross-validation scheme.
We find that the studied classifiers can use either acceleration or audio
signals to distinguish between coughing and other activities including
sneezing, throat-clearing, and movement on the bed with high accuracy.
However, in all cases, the deep neural networks outperform the shallow
classifiers by a clear margin and the Resnet50 offers the best performance by
achieving an AUC exceeding 0.98 and 0.99 for acceleration and audio signals
respectively.
While audio-based classification consistently offers a better performance
than acceleration-based classification, we observe that the difference is very
small for the best systems.
Since the acceleration signal requires less processing power, and since the
need to record audio is sidestepped and thus privacy is inherently secured, and
since the recording device is attached to the bed and not worn, an
accelerometer-based highly accurate non-invasive cough detector may represent a
more convenient and readily accepted method in long-term cough monitoring.

    

### [[2109.00110] MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics](http://arxiv.org/abs/2109.00110)


  We present miniF2F, a dataset of formal Olympiad-level mathematics problems
statements intended to provide a unified cross-system benchmark for neural
theorem proving. The miniF2F benchmark currently targets Metamath, Lean, and
Isabelle and consists of 488 problem statements drawn from the AIME, AMC, and
the International Mathematical Olympiad (IMO), as well as material from
high-school and undergraduate mathematics courses. We report baseline results
using GPT-f, a neural theorem prover based on GPT-3 and provide an analysis of
its performance. We intend for miniF2F to be a community-driven effort and hope
that our benchmark will help spur advances in neural theorem proving.

    

### [[2109.00115] Uncertainty Quantified Deep Learning for Predicting Dice Coefficient of Digital Histopathology Image Segmentation](http://arxiv.org/abs/2109.00115)


  Deep learning models (DLMs) can achieve state of the art performance in
medical image segmentation and classification tasks. However, DLMs that do not
provide feedback for their predictions such as Dice coefficients (Dice) have
limited deployment potential in real world clinical settings. Uncertainty
estimates can increase the trust of these automated systems by identifying
predictions that need further review but remain computationally prohibitive to
deploy. In this study, we use a DLM with randomly initialized weights and Monte
Carlo dropout (MCD) to segment tumors from microscopic Hematoxylin and Eosin
(H&E) dye stained prostate core biopsy RGB images. We devise a novel approach
that uses multiple clinical region based uncertainties from a single image
(instead of the entire image) to predict Dice of the DLM model output by linear
models. Image level uncertainty maps were generated and showed correspondence
between imperfect model segmentation and high levels of uncertainty associated
with specific prostate tissue regions with or without tumors. Results from this
study suggest that linear models can learn coefficients of uncertainty
quantified deep learning and correlations ((Spearman's correlation (p<0.05)) to
predict Dice scores of specific regions of medical images.

    

### [[2109.00125] A Weight Initialization Based on the Linear Product Structure for Neural Networks](http://arxiv.org/abs/2109.00125)


  Weight initialization plays an important role in training neural networks and
also affects tremendous deep learning applications. Various weight
initialization strategies have already been developed for different activation
functions with different neural networks. These initialization algorithms are
based on minimizing the variance of the parameters between layers and might
still fail when neural networks are deep, e.g., dying ReLU. To address this
challenge, we study neural networks from a nonlinear computation point of view
and propose a novel weight initialization strategy that is based on the linear
product structure (LPS) of neural networks. The proposed strategy is derived
from the polynomial approximation of activation functions by using theories of
numerical algebraic geometry to guarantee to find all the local minima. We also
provide a theoretical analysis that the LPS initialization has a lower
probability of dying ReLU comparing to other existing initialization
strategies. Finally, we test the LPS initialization algorithm on both fully
connected neural networks and convolutional neural networks to show its
feasibility, efficiency, and robustness on public datasets.

    

### [[2109.00126] Online Dynamic Window (ODW) Assisted Two-stage LSTM Frameworks for Indoor Localization](http://arxiv.org/abs/2109.00126)


  Internet of Things (IoT)-based indoor localization has gained significant
popularity recently to satisfy the ever-increasing requirements of indoor
Location-based Services (LBS). In this context, Inertial Measurement Unit
(IMU)-based localization is of interest as it provides a scalable solution
independent of any proprietary sensors/modules. Existing IMU-based
methodologies, however, are mainly developed based on statistical heading and
step length estimation techniques that suffer from cumulative error issues and
have extensive computational time requirements limiting their application for
real-time indoor positioning. To address the aforementioned issues, we propose
the Online Dynamic Window (ODW)-assisted two-stage Long Short Term Memory
(LSTM) localization framework. Three ODWs are proposed, where the first model
uses a Natural Language Processing (NLP)-inspired Dynamic Window (DW) approach,
which significantly reduces the required computational time. The second
framework is developed based on a Signal Processing Dynamic Windowing (SP-DW)
approach to further reduce the required processing time of the two-stage
LSTM-based model. The third ODW, referred to as the SP-NLP, combines the first
two windowing mechanisms to further improve the overall achieved accuracy.
Compared to the traditional LSTM-based positioning approaches, which suffer
from either high tensor computation requirements or low accuracy, the proposed
ODW-assisted models can perform indoor localization in a near-real time fashion
with high accuracy. Performances of the proposed ODW-assisted models are
evaluated based on a real Pedestrian Dead Reckoning (PDR) dataset. The results
illustrate potentials of the proposed ODW-assisted techniques in achieving high
classification accuracy with significantly reduced computational time, making
them applicable for near real-time implementations.

    

### [[2109.00137] Implicit Behavioral Cloning](http://arxiv.org/abs/2109.00137)


  We find that across a wide range of robot policy learning scenarios, treating
supervised policy learning with an implicit model generally performs better, on
average, than commonly used explicit models. We present extensive experiments
on this finding, and we provide both intuitive insight and theoretical
arguments distinguishing the properties of implicit models compared to their
explicit counterparts, particularly with respect to approximating complex,
potentially discontinuous and multi-valued (set-valued) functions. On robotic
policy learning tasks we show that implicit behavioral cloning policies with
energy-based models (EBM) often outperform common explicit (Mean Square Error,
or Mixture Density) behavioral cloning policies, including on tasks with
high-dimensional action spaces and visual image inputs. We find these policies
provide competitive results or outperform state-of-the-art offline
reinforcement learning methods on the challenging human-expert tasks from the
D4RL benchmark suite, despite using no reward information. In the real world,
robots with implicit policies can learn complex and remarkably subtle behaviors
on contact-rich tasks from human demonstrations, including tasks with high
combinatorial complexity and tasks requiring 1mm precision.

    

### [[2109.00138] Deep Dual Support Vector Data Description for Anomaly Detection on Attributed Networks](http://arxiv.org/abs/2109.00138)


  Networks are ubiquitous in the real world such as social networks and
communication networks, and anomaly detection on networks aims at finding nodes
whose structural or attributed patterns deviate significantly from the majority
of reference nodes. However, most of the traditional anomaly detection methods
neglect the relation structure information among data points and therefore
cannot effectively generalize to the graph structure data. In this paper, we
propose an end-to-end model of Deep Dual Support Vector Data description based
Autoencoder (Dual-SVDAE) for anomaly detection on attributed networks, which
considers both the structure and attribute for attributed networks.
Specifically, Dual-SVDAE consists of a structure autoencoder and an attribute
autoencoder to learn the latent representation of the node in the structure
space and attribute space respectively. Then, a dual-hypersphere learning
mechanism is imposed on them to learn two hyperspheres of normal nodes from the
structure and attribute perspectives respectively. Moreover, to achieve joint
learning between the structure and attribute of the network, we fuse the
structure embedding and attribute embedding as the final input of the feature
decoder to generate the node attribute. Finally, abnormal nodes can be detected
by measuring the distance of nodes to the learned center of each hypersphere in
the latent structure space and attribute space respectively. Extensive
experiments on the real-world attributed networks show that Dual-SVDAE
consistently outperforms the state-of-the-arts, which demonstrates the
effectiveness of the proposed method.

    

### [[2109.00150] Federated Reconnaissance: Efficient, Distributed, Class-Incremental Learning](http://arxiv.org/abs/2109.00150)


  We describe federated reconnaissance, a class of learning problems in which
distributed clients learn new concepts independently and communicate that
knowledge efficiently. In particular, we propose an evaluation framework and
methodological baseline for a system in which each client is expected to learn
a growing set of classes and communicate knowledge of those classes efficiently
with other clients, such that, after knowledge merging, the clients should be
able to accurately discriminate between classes in the superset of classes
observed by the set of clients. We compare a range of learning algorithms for
this problem and find that prototypical networks are a strong approach in that
they are robust to catastrophic forgetting while incorporating new information
efficiently. Furthermore, we show that the online averaging of prototype
vectors is effective for client model merging and requires only a small amount
of communication overhead, memory, and update time per class with no
gradient-based learning or hyperparameter tuning. Additionally, to put our
results in context, we find that a simple, prototypical network with four
convolutional layers significantly outperforms complex, state of the art
continual learning algorithms, increasing the accuracy by over 22% after
learning 600 Omniglot classes and over 33% after learning 20 mini-ImageNet
classes incrementally. These results have important implications for federated
reconnaissance and continual learning more generally by demonstrating that
communicating feature vectors is an efficient, robust, and effective means for
distributed, continual learning.

    

### [[2109.00151] Asynchronous Federated Learning for Sensor Data with Concept Drift](http://arxiv.org/abs/2109.00151)


  Federated learning (FL) involves multiple distributed devices jointly
training a shared model without any of the participants having to reveal their
local data to a centralized server. Most of previous FL approaches assume that
data on devices are fixed and stationary during the training process. However,
this assumption is unrealistic because these devices usually have varying
sampling rates and different system configurations. In addition, the underlying
distribution of the device data can change dynamically over time, which is
known as concept drift. Concept drift makes the learning process complicated
because of the inconsistency between existing and upcoming data. Traditional
concept drift handling techniques such as chunk based and ensemble
learning-based methods are not suitable in the federated learning frameworks
due to the heterogeneity of local devices. We propose a novel approach,
FedConD, to detect and deal with the concept drift on local devices and
minimize the effect on the performance of models in asynchronous FL. The drift
detection strategy is based on an adaptive mechanism which uses the historical
performance of the local models. The drift adaptation is realized by adjusting
the regularization parameter of objective function on each local device.
Additionally, we design a communication strategy on the server side to select
local updates in a prudent fashion and speed up model convergence. Experimental
evaluations on three evolving data streams and two image datasets show that
\model~detects and handles concept drift, and also reduces the overall
communication cost compared to other baseline methods.

    

### [[2109.00157] A Survey of Exploration Methods in Reinforcement Learning](http://arxiv.org/abs/2109.00157)


  Exploration is an essential component of reinforcement learning algorithms,
where agents need to learn how to predict and control unknown and often
stochastic environments. Reinforcement learning agents depend crucially on
exploration to obtain informative data for the learning process as the lack of
enough information could hinder effective learning. In this article, we provide
a survey of modern exploration methods in (Sequential) reinforcement learning,
as well as a taxonomy of exploration methods.

    

### [[2109.00172] Task-Oriented Communication for Multi-Device Cooperative Edge Inference](http://arxiv.org/abs/2109.00172)


  This paper investigates task-oriented communication for multi-device
cooperative edge inference, where a group of distributed low-end edge devices
transmit the extracted features of local samples to a powerful edge server for
inference. While cooperative edge inference can overcome the limited sensing
capability of a single device, it substantially increases the communication
overhead and may incur excessive latency. To enable low-latency cooperative
inference, we propose a learning-based communication scheme that optimizes
local feature extraction and distributed feature encoding in a task-oriented
manner, i.e., to remove data redundancy and transmit information that is
essential for the downstream inference task rather than reconstructing the data
samples at the edge server. Specifically, we leverage an information bottleneck
(IB) principle to extract the task-relevant feature at each edge device and
adopt a distributed information bottleneck (DIB) framework to formalize a
single-letter characterization of the optimal rate-relevance tradeoff for
distributed feature encoding. To admit flexible control of the communication
overhead, we extend the DIB framework to a distributed deterministic
information bottleneck (DDIB) objective that explicitly incorporates the
representational costs of the encoded features. As the IB-based objectives are
computationally prohibitive for high-dimensional data, we adopt variational
approximations to make the optimization problems tractable. To compensate the
potential performance loss due to the variational approximations, we also
develop a selective retransmission (SR) mechanism to identify the redundancy in
the encoded features of multiple edge devices to attain additional
communication overhead reduction. Extensive experiments evidence that the
proposed task-oriented communication scheme achieves a better rate-relevance
tradeoff than baseline methods.

    

### [[2109.00173] FADE: FAir Double Ensemble Learning for Observable and Counterfactual Outcomes](http://arxiv.org/abs/2109.00173)


  Methods for building fair predictors often involve tradeoffs between fairness
and accuracy and between different fairness criteria, but the nature of these
tradeoffs varies. Recent work seeks to characterize these tradeoffs in specific
problem settings, but these methods often do not accommodate users who wish to
improve the fairness of an existing benchmark model without sacrificing
accuracy, or vice versa. These results are also typically restricted to
observable accuracy and fairness criteria. We develop a flexible framework for
fair ensemble learning that allows users to efficiently explore the
fairness-accuracy space or to improve the fairness or accuracy of a benchmark
model. Our framework can simultaneously target multiple observable or
counterfactual fairness criteria, and it enables users to combine a large
number of previously trained and newly trained predictors. We provide
theoretical guarantees that our estimators converge at fast rates. We apply our
method on both simulated and real data, with respect to both observable and
counterfactual accuracy and fairness criteria. We show that, surprisingly,
multiple unfairness measures can sometimes be minimized simultaneously with
little impact on accuracy, relative to unconstrained predictors or existing
benchmark models.

    

### [[2109.00177] Problem Learning: Towards the Free Will of Machines](http://arxiv.org/abs/2109.00177)


  A machine intelligence pipeline usually consists of six components: problem,
representation, model, loss, optimizer and metric. Researchers have worked hard
trying to automate many components of the pipeline. However, one key component
of the pipeline--problem definition--is still left mostly unexplored in terms
of automation. Usually, it requires extensive efforts from domain experts to
identify, define and formulate important problems in an area. However,
automatically discovering research or application problems for an area is
beneficial since it helps to identify valid and potentially important problems
hidden in data that are unknown to domain experts, expand the scope of tasks
that we can do in an area, and even inspire completely new findings.
This paper describes Problem Learning, which aims at learning to discover and
define valid and ethical problems from data or from the machine's interaction
with the environment. We formalize problem learning as the identification of
valid and ethical problems in a problem space and introduce several possible
approaches to problem learning. In a broader sense, problem learning is an
approach towards the free will of intelligent machines. Currently, machines are
still limited to solving the problems defined by humans, without the ability or
flexibility to freely explore various possible problems that are even unknown
to humans. Though many machine learning techniques have been developed and
integrated into intelligent systems, they still focus on the means rather than
the purpose in that machines are still solving human defined problems. However,
proposing good problems is sometimes even more important than solving problems,
because a good problem can help to inspire new ideas and gain deeper
understandings. The paper also discusses the ethical implications of problem
learning under the background of Responsible AI.

    

### [[2109.00183] Deep $\mathcal{L}^1$ Stochastic Optimal Control Policies for Planetary Soft-landing](http://arxiv.org/abs/2109.00183)


  In this paper, we introduce a novel deep learning based solution to the
Powered-Descent Guidance (PDG) problem, grounded in principles of nonlinear
Stochastic Optimal Control (SOC) and Feynman-Kac theory. Our algorithm solves
the PDG problem by framing it as an $\mathcal{L}^1$ SOC problem for minimum
fuel consumption. Additionally, it can handle practically useful control
constraints, nonlinear dynamics and enforces state constraints as
soft-constraints. This is achieved by building off of recent work on deep
Forward-Backward Stochastic Differential Equations (FBSDEs) and differentiable
non-convex optimization neural-network layers based on stochastic search. In
contrast to previous approaches, our algorithm does not require convexification
of the constraints or linearization of the dynamics and is empirically shown to
be robust to stochastic disturbances and the initial position of the
spacecraft. After training offline, our controller can be activated once the
spacecraft is within a pre-specified radius of the landing zone and at a
pre-specified altitude i.e., the base of an inverted cone with the tip at the
landing zone. We demonstrate empirically that our controller can successfully
and safely land all trajectories initialized at the base of this cone while
minimizing fuel consumption.

    

### [[2109.00185] Adapted End-to-End Coreference Resolution System for Anaphoric Identities in Dialogues](http://arxiv.org/abs/2109.00185)


  We present an effective system adapted from the end-to-end neural coreference
resolution model, targeting on the task of anaphora resolution in dialogues.
Three aspects are specifically addressed in our approach, including the support
of singletons, encoding speakers and turns throughout dialogue interactions,
and knowledge transfer utilizing existing resources. Despite the simplicity of
our adaptation strategies, they are shown to bring significant impact to the
final performance, with up to 27 F1 improvement over the baseline. Our final
system ranks the 1st place on the leaderboard of the anaphora resolution track
in the CRAC 2021 shared task, and achieves the best evaluation results on all
four datasets.

    

### [[2109.00190] Approximation Properties of Deep ReLU CNNs](http://arxiv.org/abs/2109.00190)


  This paper is devoted to establishing $L^2$ approximation properties for deep
ReLU convolutional neural networks (CNNs) on two-dimensional space. The
analysis is based on a decomposition theorem for convolutional kernels with
large spatial size and multi-channel. Given that decomposition and the property
of the ReLU activation function, a universal approximation theorem of deep ReLU
CNNs with classic structure is obtained by showing its connection with ReLU
deep neural networks (DNNs) with one hidden layer. Furthermore, approximation
properties are also obtained for neural networks with ResNet, pre-act ResNet,
and MgNet architecture based on connections between these networks.

    

### [[2109.00194] Boosting Cross-Lingual Transfer via Self-Learning with Uncertainty Estimation](http://arxiv.org/abs/2109.00194)


  Recent multilingual pre-trained language models have achieved remarkable
zero-shot performance, where the model is only finetuned on one source language
and directly evaluated on target languages. In this work, we propose a
self-learning framework that further utilizes unlabeled data of target
languages, combined with uncertainty estimation in the process to select
high-quality silver labels. Three different uncertainties are adapted and
analyzed specifically for the cross lingual transfer: Language
Heteroscedastic/Homoscedastic Uncertainty (LEU/LOU), Evidential Uncertainty
(EVI). We evaluate our framework with uncertainties on two cross-lingual tasks
including Named Entity Recognition (NER) and Natural Language Inference (NLI)
covering 40 languages in total, which outperforms the baselines significantly
by 10 F1 on average for NER and 2.5 accuracy score for NLI.

    

### [[2109.00201] An Empirical Study on the Joint Impact of Feature Selection and Data Resampling on Imbalance Classification](http://arxiv.org/abs/2109.00201)


  Real-world datasets often present different degrees of imbalanced (i.e.,
long-tailed or skewed) distributions. While the majority (a.k.a., head or
frequent) classes have sufficient samples, the minority (a.k.a., tail or rare)
classes can be under-represented by a rather limited number of samples. On one
hand, data resampling is a common approach to tackling class imbalance. On the
other hand, dimension reduction, which reduces the feature space, is a
conventional machine learning technique for building stronger classification
models on a dataset. However, the possible synergy between feature selection
and data resampling for high-performance imbalance classification has rarely
been investigated before. To address this issue, this paper carries out a
comprehensive empirical study on the joint influence of feature selection and
resampling on two-class imbalance classification. Specifically, we study the
performance of two opposite pipelines for imbalance classification, i.e.,
applying feature selection before or after data resampling. We conduct a large
amount of experiments (a total of 9225 experiments) on 52 publicly available
datasets, using 9 feature selection methods, 6 resampling approaches for class
imbalance learning, and 3 well-known classification algorithms. Experimental
results show that there is no constant winner between the two pipelines, thus
both of them should be considered to derive the best performing model for
imbalance classification. We also find that the performance of an imbalance
classification model depends on the classifier adopted, the ratio between the
number of majority and minority samples (IR), as well as on the ratio between
the number of samples and features (SFR). Overall, this study should provide
new reference value for researchers and practitioners in imbalance learning.

    

### [[2109.00202] Federated Learning: Issues in Medical Application](http://arxiv.org/abs/2109.00202)


  Since the federated learning, which makes AI learning possible without moving
local data around, was introduced by google in 2017 it has been actively
studied particularly in the field of medicine. In fact, the idea of machine
learning in AI without collecting data from local clients is very attractive
because data remain in local sites. However, federated learning techniques
still have various open issues due to its own characteristics such as non
identical distribution, client participation management, and vulnerable
environments. In this presentation, the current issues to make federated
learning flawlessly useful in the real world will be briefly overviewed. They
are related to data/system heterogeneity, client management, traceability, and
security. Also, we introduce the modularized federated learning framework, we
currently develop, to experiment various techniques and protocols to find
solutions for aforementioned issues. The framework will be open to public after
development completes.

    

### [[2109.00207] Fairness based Multi-Preference Resource Allocation in Decentralised Open Markets](http://arxiv.org/abs/2109.00207)


  In this work, we focus on resource allocation in a decentralised open market.
In decentralised open markets consists of multiple vendors and multiple
dynamically-arriving buyers, thus makes the market complex and dynamic.
Because, in these markets, negotiations among vendors and buyers take place
over multiple conflicting issues such as price, scalability, robustness, delay,
etc. As a result, optimising the resource allocation in such open markets
becomes directly dependent on two key decisions, which are; incorporating a
different kind of buyers' preferences, and fairness based vendor elicitation
strategy. Towards this end, in this work, we propose a three-step resource
allocation approach that employs a reverse-auction paradigm. At the first step,
priority label is attached to each bidding vendor based on the proposed
priority mechanism. Then, at the second step, the preference score is
calculated for all the different kinds of preferences of the buyers. Finally,
at the third step, based on the priority label of the vendor and the preference
score winner is determined. Finally, we compare the proposed approach with two
state-of-the-art resource pricing and allocation strategies. The experimental
results show that the proposed approach outperforms the other two resource
allocation approaches in terms of the independent utilities of buyers and the
overall utility of the open market.

    

### [[2109.00238] Complexity Measures for Multi-objective Symbolic Regression](http://arxiv.org/abs/2109.00238)


  Multi-objective symbolic regression has the advantage that while the accuracy
of the learned models is maximized, the complexity is automatically adapted and
need not be specified a-priori. The result of the optimization is not a single
solution anymore, but a whole Pareto-front describing the trade-off between
accuracy and complexity. In this contribution we study which complexity
measures are most appropriately used in symbolic regression when performing
multi- objective optimization with NSGA-II. Furthermore, we present a novel
complexity measure that includes semantic information based on the function
symbols occurring in the models and test its effects on several benchmark
datasets. Results comparing multiple complexity measures are presented in terms
of the achieved accuracy and model length to illustrate how the search
direction of the algorithm is affected.

    

### [[2109.00267] The Impact of Reinitialization on Generalization in Convolutional Neural Networks](http://arxiv.org/abs/2109.00267)


  Recent results suggest that reinitializing a subset of the parameters of a
neural network during training can improve generalization, particularly for
small training sets. We study the impact of different reinitialization methods
in several convolutional architectures across 12 benchmark image classification
datasets, analyzing their potential gains and highlighting limitations. We also
introduce a new layerwise reinitialization algorithm that outperforms previous
methods and suggest explanations of the observed improved generalization.
First, we show that layerwise reinitialization increases the margin on the
training examples without increasing the norm of the weights, hence leading to
an improvement in margin-based generalization bounds for neural networks.
Second, we demonstrate that it settles in flatter local minima of the loss
surface. Third, it encourages learning general rules and discourages
memorization by placing emphasis on the lower layers of the neural network. Our
takeaway message is that the accuracy of convolutional neural networks can be
improved for small datasets using bottom-up layerwise reinitialization, where
the number of reinitialized layers may vary depending on the available compute
budget.

    

### [[2109.00343] Exploring deep learning methods for recognizing rare diseases and their clinical manifestations from texts](http://arxiv.org/abs/2109.00343)


  Although rare diseases are characterized by low prevalence, approximately 300
million people are affected by a rare disease. The early and accurate diagnosis
of these conditions is a major challenge for general practitioners, who do not
have enough knowledge to identify them. In addition to this, rare diseases
usually show a wide variety of manifestations, which might make the diagnosis
even more difficult. A delayed diagnosis can negatively affect the patient's
life. Therefore, there is an urgent need to increase the scientific and medical
knowledge about rare diseases. Natural Language Processing (NLP) and Deep
Learning can help to extract relevant information about rare diseases to
facilitate their diagnosis and treatments. The paper explores the use of
several deep learning techniques such as Bidirectional Long Short Term Memory
(BiLSTM) networks or deep contextualized word representations based on
Bidirectional Encoder Representations from Transformers (BERT) to recognize
rare diseases and their clinical manifestations (signs and symptoms) in the
RareDis corpus. This corpus contains more than 5,000 rare diseases and almost
6,000 clinical manifestations. BioBERT, a domain-specific language
representation based on BERT and trained on biomedical corpora, obtains the
best results. In particular, this model obtains an F1-score of 85.2% for rare
diseases, outperforming all the other models.

    

### [[2109.00442] Position Masking for Improved Layout-Aware Document Understanding](http://arxiv.org/abs/2109.00442)


  Natural language processing for document scans and PDFs has the potential to
enormously improve the efficiency of business processes. Layout-aware word
embeddings such as LayoutLM have shown promise for classification of and
information extraction from such documents. This paper proposes a new
pre-training task called that can improve performance of layout-aware word
embeddings that incorporate 2-D position embeddings. We compare models
pre-trained with only language masking against models pre-trained with both
language masking and position masking, and we find that position masking
improves performance by over 5% on a form understanding task.

    

### [[2109.00498] The Second International Verification of Neural Networks Competition (VNN-COMP 2021): Summary and Results](http://arxiv.org/abs/2109.00498)


  This report summarizes the second International Verification of Neural
Networks Competition (VNN-COMP 2021), held as a part of the 4th Workshop on
Formal Methods for ML-Enabled Autonomous Systems that was collocated with the
33rd International Conference on Computer-Aided Verification (CAV). Twelve
teams participated in this competition. The goal of the competition is to
provide an objective comparison of the state-of-the-art methods in neural
network verification, in terms of scalability and speed. Along this line, we
used standard formats (ONNX for neural networks and VNNLIB for specifications),
standard hardware (all tools are run by the organizers on AWS), and tool
parameters provided by the tool authors. This report summarizes the rules,
benchmarks, participating tools, results, and lessons learned from this
competition.

    

### [[1905.10330] Dirac Delta Regression: Conditional Density Estimation with Clinical Trials](http://arxiv.org/abs/1905.10330)


  Personalized medicine seeks to identify the causal effect of treatment for a
particular patient as opposed to a clinical population at large. Most
investigators estimate such personalized treatment effects by regressing the
outcome of a randomized clinical trial (RCT) on patient covariates. The
realized value of the outcome may however lie far from the conditional
expectation. We therefore introduce a method called Dirac Delta Regression
(DDR) that estimates the entire conditional density from RCT data in order to
visualize the probabilities across all possible outcome values. DDR transforms
the outcome into a set of asymptotically Dirac delta distributions and then
estimates the density using non-linear regression. The algorithm can identify
significant differences in patient-specific outcomes even when no population
level effect exists. Moreover, DDR outperforms state-of-the-art algorithms in
conditional density estimation by a large margin even in the small sample
regime. An R package is available at this https URL.

    

### [[1910.09040] Landing Probabilities of Random Walks for Seed-Set Expansion in Hypergraphs](http://arxiv.org/abs/1910.09040)


  We describe the first known mean-field study of landing probabilities for
random walks on hypergraphs. In particular, we examine clique-expansion and
tensor methods and evaluate their mean-field characteristics over a class of
random hypergraph models for the purpose of seed-set community expansion. We
describe parameter regimes in which the two methods outperform each other and
propose a hybrid expansion method that uses partial clique-expansion to reduce
the projection distortion and low-complexity tensor methods applied directly on
the partially expanded hypergraphs.

    

### [[2001.05317] CycleCluster: Modernising Clustering Regularisation for Deep Semi-Supervised Classification](http://arxiv.org/abs/2001.05317)


  Given the potential difficulties in obtaining large quantities of labelled
data, many works have explored the use of deep semi-supervised learning, which
uses both labelled and unlabelled data to train a neural network architecture.
The vast majority of SSL approaches focus on implementing the low-density
separation assumption or consistency assumption, the idea that decision
boundaries should lie in low density regions. However, they have implemented
this assumption by making local changes to the decision boundary at each data
point, ignoring the global structure of the data. In this work, we explore an
alternative approach using the global information present in the clustered data
to update our decision boundaries. We propose a novel framework, CycleCluster,
for deep semi-supervised classification. Our core optimisation is driven by a
new clustering based regularisation along with a graph based pseudo-labels and
a shared deep network. Demonstrating that direct implementation of the cluster
assumption is a viable alternative to the popular consistency based
regularisation. We demonstrate the predictive capability of our technique
through a careful set of numerical results.

    

### [[2003.11435] Preferential Batch Bayesian Optimization](http://arxiv.org/abs/2003.11435)


  Most research in Bayesian optimization (BO) has focused on \emph{direct
feedback} scenarios, where one has access to exact values of some
expensive-to-evaluate objective. This direction has been mainly driven by the
use of BO in machine learning hyper-parameter configuration problems. However,
in domains such as modelling human preferences, A/B tests, or recommender
systems, there is a need for methods that can replace direct feedback with
\emph{preferential feedback}, obtained via rankings or pairwise comparisons. In
this work, we present preferential batch Bayesian optimization (PBBO), a new
framework that allows finding the optimum of a latent function of interest,
given any type of parallel preferential feedback for a group of two or more
points. We do so by using a Gaussian process model with a likelihood specially
designed to enable parallel and efficient data collection mechanisms, which are
key in modern machine learning. We show how the acquisitions developed under
this framework generalize and augment previous approaches in Bayesian
optimization, expanding the use of these techniques to a wider range of
domains. An extensive simulation study shows the benefits of this approach,
both with simulated functions and four real data sets.

    

### [[2005.02392] Deep Constraint-based Propagation in Graph Neural Networks](http://arxiv.org/abs/2005.02392)


  The popularity of deep learning techniques renewed the interest in neural
architectures able to process complex structures that can be represented using
graphs, inspired by Graph Neural Networks (GNNs). We focus our attention on the
originally proposed GNN model of Scarselli et al. 2009, which encodes the state
of the nodes of the graph by means of an iterative diffusion procedure that,
during the learning stage, must be computed at every epoch, until the fixed
point of a learnable state transition function is reached, propagating the
information among the neighbouring nodes. We propose a novel approach to
learning in GNNs, based on constrained optimization in the Lagrangian
framework. Learning both the transition function and the node states is the
outcome of a joint process, in which the state convergence procedure is
implicitly expressed by a constraint satisfaction mechanism, avoiding iterative
epoch-wise procedures and the network unfolding. Our computational structure
searches for saddle points of the Lagrangian in the adjoint space composed of
weights, nodes state variables and Lagrange multipliers. This process is
further enhanced by multiple layers of constraints that accelerate the
diffusion process. An experimental analysis shows that the proposed approach
compares favourably with popular models on several benchmarks.

    

### [[2005.11115] Consistency of Extreme Learning Machines and Regression under Non-Stationarity and Dependence for ML-Enhanced Moving Objects](http://arxiv.org/abs/2005.11115)


  Supervised learning by extreme learning machines resp. neural networks with
random weights is studied under a non-stationary spatial-temporal sampling
design which especially addresses settings where an autonomous object moving in
a non-stationary spatial environment collects and analyzes data. The stochastic
model especially allows for spatial heterogeneity and weak dependence. As
efficient and computationally cheap learning methods (unconstrained) least
squares, ridge regression and $\ell_s$-penalized least squares (including the
LASSO) are studied. Consistency and asymptotic normality of the least squares
and ridge regression estimates as well as corresponding consistency results for
the $\ell_s$-penalty are shown under weak conditions. The results also cover
bounds for the sample squared predicition error.

    

### [[2006.02915] Continuous-time system identification with neural networks: Model structures and fitting criteria](http://arxiv.org/abs/2006.02915)


  This paper presents tailor-made neural model structures and two custom
fitting criteria for learning dynamical systems. The proposed framework is
based on a representation of the system behavior in terms of continuous-time
state-space models. The sequence of hidden states is optimized along with the
neural network parameters in order to minimize the difference between measured
and estimated outputs, and at the same time to guarantee that the optimized
state sequence is consistent with the estimated system dynamics. The
effectiveness of the approach is demonstrated through three case studies,
including two public system identification benchmarks based on experimental
data.

    

### [[2011.08463] Meta Automatic Curriculum Learning](http://arxiv.org/abs/2011.08463)


  A major challenge in the Deep RL (DRL) community is to train agents able to
generalize their control policy over situations never seen in training.
Training on diverse tasks has been identified as a key ingredient for good
generalization, which pushed researchers towards using rich procedural task
generation systems controlled through complex continuous parameter spaces. In
such complex task spaces, it is essential to rely on some form of Automatic
Curriculum Learning (ACL) to adapt the task sampling distribution to a given
learning agent, instead of randomly sampling tasks, as many could end up being
either trivial or unfeasible. Since it is hard to get prior knowledge on such
task spaces, many ACL algorithms explore the task space to detect progress
niches over time, a costly tabula-rasa process that needs to be performed for
each new learning agents, although they might have similarities in their
capabilities profiles. To address this limitation, we introduce the concept of
Meta-ACL, and formalize it in the context of black-box RL learners, i.e.
algorithms seeking to generalize curriculum generation to an (unknown)
distribution of learners. In this work, we present AGAIN, a first instantiation
of Meta-ACL, and showcase its benefits for curriculum generation over classical
ACL in multiple simulated environments including procedurally generated parkour
environments with learners of varying morphologies. Videos and code are
available at this https URL .

    

### [[2011.11261] Hierarchically Decoupled Spatial-Temporal Contrast for Self-supervised Video Representation Learning](http://arxiv.org/abs/2011.11261)


  We present a novel technique for self-supervised video representation
learning by: (a) decoupling the learning objective into two contrastive
subtasks respectively emphasizing spatial and temporal features, and (b)
performing it hierarchically to encourage multi-scale understanding. Motivated
by their effectiveness in supervised learning, we first introduce
spatial-temporal feature learning decoupling and hierarchical learning to the
context of unsupervised video learning. We show by experiments that
augmentations can be manipulated as regularization to guide the network to
learn desired semantics in contrastive learning, and we propose a way for the
model to separately capture spatial and temporal features at multiple scales.
We also introduce an approach to overcome the problem of divergent levels of
instance invariance at different hierarchies by modeling the invariance as loss
weights for objective re-weighting. Experiments on downstream action
recognition benchmarks on UCF101 and HMDB51 show that our proposed
Hierarchically Decoupled Spatial-Temporal Contrast (HDC) makes substantial
improvements over directly learning spatial-temporal features as a whole and
achieves competitive performance when compared with other state-of-the-art
unsupervised methods. Code will be made available.

    

### [[2012.09156] Learning Accurate Long-term Dynamics for Model-based Reinforcement Learning](http://arxiv.org/abs/2012.09156)


  Accurately predicting the dynamics of robotic systems is crucial for
model-based control and reinforcement learning. The most common way to estimate
dynamics is by fitting a one-step ahead prediction model and using it to
recursively propagate the predicted state distribution over long horizons.
Unfortunately, this approach is known to compound even small prediction errors,
making long-term predictions inaccurate. In this paper, we propose a new
parametrization to supervised learning on state-action data to stably predict
at longer horizons -- that we call a trajectory-based model. This
trajectory-based model takes an initial state, a future time index, and control
parameters as inputs, and directly predicts the state at the future time index.
Experimental results in simulated and real-world robotic tasks show that
trajectory-based models yield significantly more accurate long term
predictions, improved sample efficiency, and the ability to predict task
reward. With these improved prediction properties, we conclude with a
demonstration of methods for using the trajectory-based model for control.

    

### [[2102.10343] Measuring $\ell_\infty$ Attacks by the $\ell_2$ Norm](http://arxiv.org/abs/2102.10343)


  Deep Neural Networks (DNNs) could be easily fooled by Adversarial Examples
(AEs) with the imperceptible difference to original samples in human eyes. To
keep the difference imperceptible, the existing attacking bound the adversarial
perturbations by the $\ell_\infty$ norm, which is then served as the standard
to align different attacks for a fair comparison. However, when investigating
attack transferability, i.e., the capability of the AEs from attacking one
surrogate DNN to cheat other black-box DNN, we find that only using the
$\ell_\infty$ norm is not sufficient to measure the attack strength, according
to our comprehensive experiments concerning 7 transfer-based attacks, 4
white-box surrogate models, and 9 black-box victim models. Specifically, we
find that the $\ell_2$ norm greatly affects the transferability in
$\ell_\infty$ attacks. Since larger-perturbed AEs naturally bring about better
transferability, we advocate that the strength of all attacks should be
measured by both the widely used $\ell_\infty$ and also the $\ell_2$ norm.
Despite the intuitiveness of our conclusion and advocacy, they are very
necessary for the community, because common evaluations (bounding only the
$\ell_\infty$ norm) allow tricky enhancements of the "attack transferability"
by increasing the "attack strength" ($\ell_2$ norm) as shown by our simple
counter-example method, and the good transferability of several existing
methods may be due to their large $\ell_2$ distances.

    

### [[2103.14079] Domain Specific Concept Drift Detectors for Predicting Financial Time Series](http://arxiv.org/abs/2103.14079)


  Concept drift detectors allow learning systems to maintain good accuracy on
non-stationary data streams. Financial time series are an instance of
non-stationary data streams whose concept drifts (market phases) are so
important to affect investment decisions worldwide. This paper studies how
concept drift detectors behave when applied to financial time series. General
results are: a) concept drift detectors usually improve the runtime over
continuous learning, b) their computational cost is usually a fraction of the
learning and prediction steps of even basic learners, c) it is important to
study concept drift detectors in combination with the learning systems they
will operate with, and d) concept drift detectors can be directly applied to
the time series of raw financial data and not only to the model's accuracy one.
Moreover, the study introduces three simple concept drift detectors, tailored
to financial time series, and shows that two of them can be at least as
effective as the most sophisticated ones from the state of the art when applied
to financial time series. Currently submitted to Pattern Recognition

    

### [[2104.13369] Explaining in Style: Training a GAN to explain a classifier in StyleSpace](http://arxiv.org/abs/2104.13369)


  Image classification models can depend on multiple different semantic
attributes of the image. An explanation of the decision of the classifier needs
to both discover and visualize these properties. Here we present StylEx, a
method for doing this, by training a generative model to specifically explain
multiple attributes that underlie classifier decisions. A natural source for
such attributes is the StyleSpace of StyleGAN, which is known to generate
semantically meaningful dimensions in the image. However, because standard GAN
training is not dependent on the classifier, it may not represent these
attributes which are important for the classifier decision, and the dimensions
of StyleSpace may represent irrelevant attributes. To overcome this, we propose
a training procedure for a StyleGAN, which incorporates the classifier model,
in order to learn a classifier-specific StyleSpace. Explanatory attributes are
then selected from this space. These can be used to visualize the effect of
changing multiple attributes per image, thus providing image-specific
explanations. We apply StylEx to multiple domains, including animals, leaves,
faces and retinal images. For these, we show how an image can be modified in
different ways to change its classifier output. Our results show that the
method finds attributes that align well with semantic ones, generate meaningful
image-specific explanations, and are human-interpretable as measured in
user-studies.

    

### [[2105.07464] Few-NERD: A Few-Shot Named Entity Recognition Dataset](http://arxiv.org/abs/2105.07464)


  Recently, considerable literature has grown up around the theme of few-shot
named entity recognition (NER), but little published benchmark data
specifically focused on the practical and challenging task. Current approaches
collect existing supervised NER datasets and re-organize them to the few-shot
setting for empirical study. These strategies conventionally aim to recognize
coarse-grained entity types with few examples, while in practice, most unseen
entity types are fine-grained. In this paper, we present Few-NERD, a
large-scale human-annotated few-shot NER dataset with a hierarchy of 8
coarse-grained and 66 fine-grained entity types. Few-NERD consists of 188,238
sentences from Wikipedia, 4,601,160 words are included and each is annotated as
context or a part of a two-level entity type. To the best of our knowledge,
this is the first few-shot NER dataset and the largest human-crafted NER
dataset. We construct benchmark tasks with different emphases to
comprehensively assess the generalization capability of models. Extensive
empirical results and analysis show that Few-NERD is challenging and the
problem requires further research. We make Few-NERD public at
this https URL.

    

### [[2105.13929] Quantifying and Localizing Private Information Leakage from Neural Network Gradients](http://arxiv.org/abs/2105.13929)


  Empirical attacks on collaborative learning show that the gradients of deep
neural networks can not only disclose private latent attributes of the training
data but also be used to reconstruct the original data. While prior works tried
to quantify the privacy risk stemming from gradients, these measures do not
establish a theoretically grounded understanding of gradient leakages, do not
generalize across attackers, and can fail to fully explain what is observed
through empirical attacks in practice. In this paper, we introduce
theoretically-motivated measures to quantify information leakages in both
attack-dependent and attack-independent manners. Specifically, we present an
adaptation of the $\mathcal{V}$-information, which generalizes the empirical
attack success rate and allows quantifying the amount of information that can
leak from any chosen family of attack models. We then propose
attack-independent measures, that only require the shared gradients, for
quantifying both original and latent information leakages. Our empirical
results, on six datasets and four popular models, reveal that gradients of the
first layers contain the highest amount of original information, while the
(fully-connected) classifier layers placed after the (convolutional) feature
extractor layers contain the highest latent information. Further, we show how
techniques such as gradient aggregation during training can mitigate
information leakages. Our work paves the way for better defenses such as
layer-based protection or strong aggregation.

    

### [[2106.08122] Sequence-Level Training for Non-Autoregressive Neural Machine Translation](http://arxiv.org/abs/2106.08122)


  In recent years, Neural Machine Translation (NMT) has achieved notable
results in various translation tasks. However, the word-by-word generation
manner determined by the autoregressive mechanism leads to high translation
latency of the NMT and restricts its low-latency applications.
Non-Autoregressive Neural Machine Translation (NAT) removes the autoregressive
mechanism and achieves significant decoding speedup through generating target
words independently and simultaneously. Nevertheless, NAT still takes the
word-level cross-entropy loss as the training objective, which is not optimal
because the output of NAT cannot be properly evaluated due to the multimodality
problem. In this article, we propose using sequence-level training objectives
to train NAT models, which evaluate the NAT outputs as a whole and correlates
well with the real translation quality. Firstly, we propose training NAT models
to optimize sequence-level evaluation metrics (e.g., BLEU) based on several
novel reinforcement algorithms customized for NAT, which outperforms the
conventional method by reducing the variance of gradient estimation. Secondly,
we introduce a novel training objective for NAT models, which aims to minimize
the Bag-of-Ngrams (BoN) difference between the model output and the reference
sentence. The BoN training objective is differentiable and can be calculated
efficiently without doing any approximations. Finally, we apply a three-stage
training strategy to combine these two methods to train the NAT model. We
validate our approach on four translation tasks (WMT14 En$\leftrightarrow$De,
WMT16 En$\leftrightarrow$Ro), which shows that our approach largely outperforms
NAT baselines and achieves remarkable performance on all translation tasks. The
source code is available at this https URL.

    

### [[2106.13516] Multi-Domain Active Learning: A Comparative Study](http://arxiv.org/abs/2106.13516)


  Building classifiers on multiple domains is a practical problem in the real
life. Instead of building classifiers one by one, multi-domain learning (MDL)
simultaneously builds classifiers on all the domains. MDL utilizes the
information shared among the domains to improve the performance. As a
supervised learning problem, the labeling effort is still high in MDL problems.
Usually, this high labeling cost issue could be relieved by using active
learning. Thus, it is natural to utilize active learning to reduce the labeling
effort in MDL, and we refer this setting as multi-domain active learning
(MDAL). However, there are only few works which are built on this setting. And
when the researchers have to face this problem, there is no off-the-shelf
solution. Under this circumstance, combining the current multi-domain learning
models and single-domain active learning strategies might be a preliminary
solution for MDAL problem. To find out the potential of this preliminary
solution, a comparative study over 5 models and 4 active learning strategies is
made in this paper. To the best of our knowledge, this is the first work
provides the formal definition of MDAL. Besides, this is the first comparative
work for MDAL problem. From the results, the Multinomial Adversarial Networks
(MAN) model with a simple best vs second best (BvSB) uncertainty strategy shows
its superiority in most cases. We take this combination as our off-the-shelf
recommendation for the MDAL problem.

    

### [[2107.01034] Weather-based forecasting of energy generation, consumption and price for electrical microgrids management](http://arxiv.org/abs/2107.01034)


  The Intergovernmental Panel on Climate Change proposes different mitigation
strategies to achieve the net emissions reductions that would be required to
follow a pathway that limits global warming to 1.5°C with no or limited
overshoot. The transition towards a carbon-free society goes through an
inevitable increase of the share of renewable generation in the energy mix and
a drastic decrease in terms of the total consumption of fossil fuels.
Therefore, this thesis studies the integration of renewables in power systems
by investigating forecasting and decision-making tools. Indeed, in contrast to
conventional power plants, renewable energy is subject to uncertainty. Most of
the generation technologies based on renewable sources are non-dispatchable,
and their production is stochastic and hard to predict in advance. A high share
of renewables is a great challenge for power systems that have been designed
and sized for dispatchable units. In this context, probabilistic forecasts,
which aim at modeling the distribution of all possible future realizations,
have become an important tool to equip decision-makers, hopefully leading to
better decisions in energy applications. This thesis focus on two main research
questions: (1) How to produce reliable probabilistic forecasts of renewable
generation, consumption, and electricity prices? (2) How to make decisions with
uncertainty using probabilistic forecasts? The thesis perimeter is the energy
management of "small" systems such as microgrids at a residential scale on a
day-ahead basis. It is divided into two main parts to propose directions to
address both research questions (1) a forecasting part; (2) a planning and
control part.

    

### [[2108.13179] Reachability Is NP-Complete Even for the Simplest Neural Networks](http://arxiv.org/abs/2108.13179)


  We investigate the complexity of the reachability problem for (deep) neural
networks: does it compute valid output given some valid input? It was recently
claimed that the problem is NP-complete for general neural networks and
conjunctive input/output specifications. We repair some flaws in the original
upper and lower bound proofs. We then show that NP-hardness already holds for
restricted classes of simple specifications and neural networks with just one
layer, as well as neural networks with minimal requirements on the occurring
parameters.

    

### [[2108.13298] E-Commerce Promotions Personalization via Online Multiple-Choice Knapsack with Uplift Modeling](http://arxiv.org/abs/2108.13298)


  Promotions and discounts are essential components of modern e-commerce
platforms, where they are often used to incentivize customers towards purchase
completion. Promotions also affect revenue and may incur a monetary loss that
is often limited by a dedicated promotional budget. We study the Online
Constrained Multiple-Choice Promotions Personalization Problem, where the
optimization goal is to select for each customer which promotion to present in
order to maximize purchase completions, while also complying with global budget
limitations. Our work formalizes the problem as an Online Multiple Choice
Knapsack Problem and extends the existent literature by addressing cases with
negative weights and values. We provide a real-time adaptive method that
guarantees budget constraints compliance and achieves above 99.7% of the
optimal promotional impact on various datasets. Our method is evaluated on a
large-scale experimental study at one of the leading online travel platforms in
the world.

    

### [[1403.4997] Universal and Distinct Properties of Communication Dynamics: How to Generate Realistic Inter-event Times](http://arxiv.org/abs/1403.4997)


  With the advancement of information systems, means of communications are
becoming cheaper, faster and more available. Today, millions of people carrying
smart-phones or tablets are able to communicate at practically any time and
anywhere they want. Among others, they can access their e-mails, comment on
weblogs, watch and post comments on videos, make phone calls or text messages
almost ubiquitously. Given this scenario, in this paper we tackle a fundamental
aspect of this new era of communication: how the time intervals between
communication events behave for different technologies and means of
communications? Are there universal patterns for the inter-event time
distribution (IED)? In which ways inter-event times behave differently among
particular technologies? To answer these questions, we analyze eight different
datasets from real and modern communication data and we found four well defined
patterns that are seen in all the eight datasets. Moreover, we propose the use
of the Self-Feeding Process (SFP) to generate inter-event times between
communications. The SFP is extremely parsimonious point process that requires
at most two parameters and is able to generate inter-event times with all the
universal properties we observed in the data. We show the potential application
of SFP by proposing a framework to generate a synthetic dataset containing
realistic communication events of any one of the analyzed means of
communications (e.g. phone calls, e-mails, comments on blogs) and an algorithm
to detect anomalies.

    

### [[2109.00474] Leaking Control Flow Information via the Hardware Prefetcher](http://arxiv.org/abs/2109.00474)


  Modern processor designs use a variety of microarchitectural methods to
achieve high performance. Unfortunately, new side-channels have often been
uncovered that exploit these enhanced designs. One area that has received
little attention from a security perspective is the processor's hard-ware
prefetcher, a critical component used to mitigate DRAM latency in today's
systems. Prefetchers, like branch predictors, hold critical state related to
the execution of the application, and have the potential to leak secret
information. But up to now, there has not been a demonstration of a generic
prefetcher side-channel that could be actively exploited in today's hardware.
In this paper, we present AfterImage, a new side-channel that exploits the
Intel Instruction Pointer-based stride prefetcher. We observe that, when the
execution of the processor switches between different private domains, the
prefetcher trained by one domain can be triggered in another. To the best of
our knowledge, this work is the first to publicly demonstrate a methodology
that is both algorithm-agnostic and also able to leak kernel data into
userspace. AfterImage is different from previous works, as it leaks data on the
non-speculative path of execution. Because of this, a large class of work that
has focused on protecting transient, branch-outcome-based data will be unable
to block this side-channel. By reverse-engineering the IP-stride prefetcher in
modern Intel processors, we have successfully developed three variants of
AfterImage to leak control flow information across code regions, processes and
the user-kernel boundary. We find a high level of accuracy in leaking
information with our methodology (from 91%, up to 99%), and propose two
mitigation techniques to block this side-channel, one of which can be used on
hardware systems today.

    

### [[2101.08744] Enabling Large Neural Networks on Tiny Microcontrollers with Swapping](http://arxiv.org/abs/2101.08744)


  Running neural networks (NNs) on microcontroller units (MCUs) is becoming
increasingly important, but is very difficult due to the tiny SRAM size of MCU.
Prior work proposes many algorithm-level techniques to reduce NN memory
footprints, but all at the cost of sacrificing accuracy and generality, which
disqualifies MCUs for many important use cases. We investigate a system
solution for MCUs to execute NNs out of core: dynamically swapping NN data
chunks between an MCU's tiny SRAM and its large, low-cost external flash.
Out-of-core NNs on MCUs raise multiple concerns: execution slowdown, storage
wear out, energy consumption, and data security. We present a study showing
that none is a showstopper; the key benefit -- MCUs being able to run large NNs
with full accuracy and generality -- triumphs the overheads. Our findings
suggest that MCUs can play a much greater role in edge intelligence.

    

### [[2109.00082] Plan-based Job Scheduling for Supercomputers with Shared Burst Buffers](http://arxiv.org/abs/2109.00082)


  The ever-increasing gap between compute and I/O performance in HPC platforms,
together with the development of novel NVMe storage devices (NVRAM), led to the
emergence of the burst buffer concept - an intermediate persistent storage
layer logically positioned between random-access main memory and a parallel
file system. Despite the development of real-world architectures as well as
research concepts, resource and job management systems, such as Slurm, provide
only marginal support for scheduling jobs with burst buffer requirements, in
particular ignoring burst buffers when backfilling. We investigate the impact
of burst buffer reservations on the overall efficiency of online job scheduling
for common algorithms: First-Come-First-Served (FCFS) and Shortest-Job-First
(SJF) EASY-backfilling. We evaluate the algorithms in a detailed simulation
with I/O side effects. Our results indicate that the lack of burst buffer
reservations in backfilling may significantly deteriorate scheduling. We also
show that these algorithms can be easily extended to support burst buffers.
Finally, we propose a burst-buffer-aware plan-based scheduling algorithm with
simulated annealing optimisation, which improves the mean waiting time by over
20% and mean bounded slowdown by 27% compared to the burst-buffer-aware
SJF-EASY-backfilling.

    

### [[2109.00416] LightChain: Scalable DHT-Based Blockchain](http://arxiv.org/abs/2109.00416)


  As an append-only distributed database, blockchain is utilized in a vast
variety of applications including the cryptocurrency and Internet-of-Things
(IoT). The existing blockchain solutions show downsides in communication and
storage scalability, as well as decentralization. In this article, we propose
LightChain , which is the first blockchain architecture that operates over a
Distributed Hash Table (DHT) of participating peers. LightChain is a
permissionless blockchain that provides addressable blocks and transactions
within the network, which makes them efficiently accessible by all peers. Each
block and transaction is replicated within the DHT of peers and is retrieved in
an on-demand manner. Hence, peers in LightChain are not required to retrieve or
keep the entire ledger. LightChain is fair as all of the participating peers
have a uniform chance of being involved in the consensus regardless of their
influence such as hashing power or stake. We provide formal mathematical
analysis and experimental results (simulations and cloud deployment) to
demonstrate the security, efficiency, and fairness of LightChain , and show
that LightChain is the only existing blockchain that can provide integrity
under the corrupted majority power of peers. As we experimentally demonstrate,
compared to the mainstream blockchains such as Bitcoin and Ethereum, LightChain
requires around 66 times smaller per node storage, and is around 380 times
faster on bootstrapping a new node to the system, and each LightChain node is
rewarded equally likely for participating in the protocol.

    

### [[2109.00465] Fast Abstracts and Student Forum Proceedings, 17th European Dependable Computing Conference -- EDCC 2021](http://arxiv.org/abs/2109.00465)


  Collection of manuscript accepted for presentation at the Student Forum and
Fast Abstracts track of the 17th European Dependable Computing Conference (EDCC
2021).

    

### [[2109.00485] Accelerating an Iterative Eigensolver for Nuclear Structure Configuration Interaction Calculations on GPUs using OpenACC](http://arxiv.org/abs/2109.00485)


  To accelerate the solution of large eigenvalue problems arising from
many-body calculations in nuclear physics on distributed-memory parallel
systems equipped with general-purpose Graphic Processing Units (GPUs), we
modified a previously developed hybrid MPI/OpenMP implementation of an
eigensolver written in FORTRAN 90 by using an OpenACC directives based
programming model. Such an approach requires making minimal changes to the
original code and enables a smooth migration of large-scale nuclear structure
simulations from a distributed-memory many-core CPU system to a distributed GPU
system. However, in order to make the OpenACC based eigensolver run efficiently
on GPUs, we need to take into account the architectural differences between a
many-core CPU and a GPU device. Consequently, the optimal way to insert OpenACC
directives may be different from the original way of inserting OpenMP
directives. We point out these differences in the implementation of sparse
matrix-matrix multiplications (SpMM), which constitutes the main cost of the
eigensolver, as well as other differences in the preconditioning step and dense
linear algebra operations. We compare the performance of the OpenACC based
implementation executed on multiple GPUs with the performance on
distributed-memory many-core CPUs, and demonstrate significant speedup achieved
on GPUs compared to the on-node performance of a many-core CPU. We also show
that the overall performance improvement of the eigensolver on multiple GPUs is
more modest due to the communication overhead among different MPI ranks.

    

### [[2103.15860] Twine: An Embedded Trusted Runtime for WebAssembly](http://arxiv.org/abs/2103.15860)


  WebAssembly is an increasingly popular lightweight binary instruction format,
which can be efficiently embedded and sandboxed. Languages like C, C++, Rust,
Go, and many others can be compiled into WebAssembly. This paper describes
Twine, a WebAssembly trusted runtime designed to execute unmodified,
language-independent applications. We leverage Intel SGX to build the runtime
environment without dealing with language-specific, complex APIs. While SGX
hardware provides secure execution within the processor, Twine provides a
secure, sandboxed software runtime nested within an SGX enclave, featuring a
WebAssembly system interface (WASI) for compatibility with unmodified
WebAssembly applications. We evaluate Twine with a large set of general-purpose
benchmarks and real-world applications. In particular, we used Twine to
implement a secure, trusted version of SQLite, a well-known full-fledged
embeddable database. We believe that such a trusted database would be a
reasonable component to build many larger application services. Our evaluation
shows that SQLite can be fully executed inside an SGX enclave via WebAssembly
and existing system interface, with similar average performance overheads. We
estimate that the performance penalties measured are largely compensated by the
additional security guarantees and its full compatibility with standard
WebAssembly. An in-depth analysis of our results indicates that performance can
be greatly improved by modifying some of the underlying libraries. We describe
and implement one such modification in the paper, showing up to $4.1\times$
speedup. Twine is open-source, available at GitHub along with instructions to
reproduce our experiments.

    

### [[2109.00031] Deep DNA Storage: Scalable and Robust DNA Storage via Coding Theory and Deep Learning](http://arxiv.org/abs/2109.00031)


  The concept of DNA storage was first suggested in 1959 by Richard Feynman who
shared his vision regarding nanotechnology in the talk "There is plenty of room
at the bottom". Later, towards the end of the 20-th century, the interest in
storage solutions based on DNA molecules was increased as a result of the human
genome project which in turn led to a significant progress in sequencing and
assembly methods. DNA storage enjoys major advantages over the well-established
magnetic and optical storage solutions. As opposed to magnetic solutions, DNA
storage does not require electrical supply to maintain data integrity and is
superior to other storage solutions in both density and durability. Given the
trends in cost decreases of DNA synthesis and sequencing, it is now
acknowledged that within the next 10-15 years DNA storage may become a highly
competitive archiving technology and probably later the main such technology.
With that said, the current implementations of DNA based storage systems are
very limited and are not fully optimized to address the unique pattern of
errors which characterize the synthesis and sequencing processes. In this work,
we propose a robust, efficient and scalable solution to implement DNA-based
storage systems. Our method deploys Deep Neural Networks (DNN) which
reconstruct a sequence of letters based on imperfect cluster of copies
generated by the synthesis and sequencing processes. A tailor-made
Error-Correcting Code (ECC) is utilized to combat patterns of errors which
occur during this process. Since our reconstruction method is adapted to
imperfect clusters, our method overcomes the time bottleneck of the noisy DNA
copies clustering process by allowing the use of a rapid and scalable
pseudo-clustering instead. Our architecture combines between convolutions and
transformers blocks and is trained using synthetic data modelled after real
data statistics.

    

### [[2109.00066] Informing Autonomous Deception Systems with Cyber Expert Performance Data](http://arxiv.org/abs/2109.00066)


  The performance of artificial intelligence (AI) algorithms in practice
depends on the realism and correctness of the data, models, and feedback
(labels or rewards) provided to the algorithm. This paper discusses methods for
improving the realism and ecological validity of AI used for autonomous cyber
defense by exploring the potential to use Inverse Reinforcement Learning (IRL)
to gain insight into attacker actions, utilities of those actions, and
ultimately decision points which cyber deception could thwart. The Tularosa
study, as one example, provides experimental data of real-world techniques and
tools commonly used by attackers, from which core data vectors can be leveraged
to inform an autonomous cyber defense system.

    

### [[2109.00100] Proceedings of KDD 2021 Workshop on Data-driven Humanitarian Mapping: Harnessing Human-Machine Intelligence for High-Stake Public Policy and Resilience Planning](http://arxiv.org/abs/2109.00100)


  Humanitarian challenges, including natural disasters, food insecurity,
climate change, racial and gender violence, environmental crises, the COVID-19
coronavirus pandemic, human rights violations, and forced displacements,
disproportionately impact vulnerable communities worldwide. According to UN
OCHA, 235 million people will require humanitarian assistance in 20211 .
Despite these growing perils, there remains a notable paucity of data science
research to scientifically inform equitable public policy decisions for
improving the livelihood of at-risk populations. Scattered data science efforts
exist to address these challenges, but they remain isolated from practice and
prone to algorithmic harms concerning lack of privacy, fairness,
interpretability, accountability, transparency, and ethics. Biases in
data-driven methods carry the risk of amplifying inequalities in high-stakes
policy decisions that impact the livelihood of millions of people.
Consequently, proclaimed benefits of data-driven innovations remain
inaccessible to policymakers, practitioners, and marginalized communities at
the core of humanitarian actions and global development. To help fill this gap,
we propose the Data-driven Humanitarian Mapping Research Program, which focuses
on developing novel data science methodologies that harness human-machine
intelligence for high-stakes public policy and resilience planning.

    

### [[2109.00127] Cognitive science as a source of forward and inverse models of human decisions for robotics and control](http://arxiv.org/abs/2109.00127)


  Those designing autonomous systems that interact with humans will invariably
face questions about how humans think and make decisions. Fortunately,
computational cognitive science offers insight into human decision-making using
tools that will be familiar to those with backgrounds in optimization and
control (e.g., probability theory, statistical machine learning, and
reinforcement learning). Here, we review some of this work, focusing on how
cognitive science can provide forward models of human decision-making and
inverse models of how humans think about others' decision-making. We highlight
relevant recent developments, including approaches that synthesize blackbox and
theory-driven modeling, accounts that recast heuristics and biases as forms of
bounded optimality, and models that characterize human theory of mind and
communication in decision-theoretic terms. In doing so, we aim to provide
readers with a glimpse of the range of frameworks, methodologies, and
actionable insights that lie at the intersection of cognitive science and
control research.

    

### [[2109.00181] CTAL: Pre-training Cross-modal Transformer for Audio-and-Language Representations](http://arxiv.org/abs/2109.00181)


  Existing audio-language task-specific predictive approaches focus on building
complicated late-fusion mechanisms. However, these models are facing challenges
of overfitting with limited labels and low model generalization abilities. In
this paper, we present a Cross-modal Transformer for Audio-and-Language, i.e.,
CTAL, which aims to learn the intra-modality and inter-modality connections
between audio and language through two proxy tasks on a large amount of
audio-and-language pairs: masked language modeling and masked cross-modal
acoustic modeling. After fine-tuning our pre-trained model on multiple
downstream audio-and-language tasks, we observe significant improvements across
various tasks, such as, emotion classification, sentiment analysis, and speaker
verification. On this basis, we further propose a specially-designed fusion
mechanism that can be used in fine-tuning phase, which allows our pre-trained
model to achieve better performance. Lastly, we demonstrate detailed ablation
studies to prove that both our novel cross-modality fusion component and
audio-language pre-training methods significantly contribute to the promising
results.

    

### [[2109.00217] Multi-Sample based Contrastive Loss for Top-k Recommendation](http://arxiv.org/abs/2109.00217)


  The top-k recommendation is a fundamental task in recommendation systems
which is generally learned by comparing positive and negative pairs. The
Contrastive Loss (CL) is the key in contrastive learning that has received more
attention recently and we find it is well suited for top-k recommendations.
However, it is a problem that CL treats the importance of the positive and
negative samples as the same. On the one hand, CL faces the imbalance problem
of one positive sample and many negative samples. On the other hand, positive
items are so few in sparser datasets that their importance should be
emphasized. Moreover, the other important issue is that the sparse positive
items are still not sufficiently utilized in recommendations. So we propose a
new data augmentation method by using multiple positive items (or samples)
simultaneously with the CL loss function. Therefore, we propose a Multi-Sample
based Contrastive Loss (MSCL) function which solves the two problems by
balancing the importance of positive and negative samples and data
augmentation. And based on the graph convolution network (GCN) method,
experimental results demonstrate the state-of-the-art performance of MSCL. The
proposed MSCL is simple and can be applied in many methods. We will release our
code on GitHub upon the acceptance.

    

### [[2109.00256] Extracting all Aspect-polarity Pairs Jointly in a Text with Relation Extraction Approach](http://arxiv.org/abs/2109.00256)


  Extracting aspect-polarity pairs from texts is an important task of
fine-grained sentiment analysis. While the existing approaches to this task
have gained many progresses, they are limited at capturing relationships among
aspect-polarity pairs in a text, thus degrading the extraction performance.
Moreover, the existing state-of-the-art approaches, namely token-based
se-quence tagging and span-based classification, have their own defects such as
polarity inconsistency resulted from separately tagging tokens in the former
and the heterogeneous categorization in the latter where aspect-related and
polarity-related labels are mixed. In order to remedy the above defects,
in-spiring from the recent advancements in relation extraction, we propose to
generate aspect-polarity pairs directly from a text with relation extraction
technology, regarding aspect-pairs as unary relations where aspects are
enti-ties and the corresponding polarities are relations. Based on the
perspective, we present a position- and aspect-aware sequence2sequence model
for joint extraction of aspect-polarity pairs. The model is characterized with
its ability to capture not only relationships among aspect-polarity pairs in a
text through the sequence decoding, but also correlations between an aspect and
its polarity through the position- and aspect-aware attentions. The
experi-ments performed on three benchmark datasets demonstrate that our model
outperforms the existing state-of-the-art approaches, making significant
im-provement over them.

    

### [[2109.00287] Complex Event Forecasting with Prediction Suffix Trees: Extended Technical Report](http://arxiv.org/abs/2109.00287)


  Complex Event Recognition (CER) systems have become popular in the past two
decades due to their ability to "instantly" detect patterns on real-time
streams of events. However, there is a lack of methods for forecasting when a
pattern might occur before such an occurrence is actually detected by a CER
engine. We present a formal framework that attempts to address the issue of
Complex Event Forecasting (CEF). Our framework combines two formalisms: a)
symbolic automata which are used to encode complex event patterns; and b)
prediction suffix trees which can provide a succinct probabilistic description
of an automaton's behavior. We compare our proposed approach against
state-of-the-art methods and show its advantage in terms of accuracy and
efficiency. In particular, prediction suffix trees, being variable-order Markov
models, have the ability to capture long-term dependencies in a stream by
remembering only those past sequences that are informative enough. Our
experimental results demonstrate the benefits, in terms of accuracy, of being
able to capture such long-term dependencies. This is achieved by increasing the
order of our model beyond what is possible with full-order Markov models that
need to perform an exhaustive enumeration of all possible past sequences of a
given order. We also discuss extensively how CEF solutions should be best
evaluated on the quality of their forecasts.

    

### [[2109.00318] Intrinsic Argument Strength in Structured Argumentation: a Principled Approach](http://arxiv.org/abs/2109.00318)


  Abstract argumentation provides us with methods such as gradual and Dung
semantics with which to evaluate arguments after potential attacks by other
arguments. Some of these methods can take intrinsic strengths of arguments as
input, with which to modulate the effects of attacks between arguments. Coming
from abstract argumentation, these methods look only at the relations between
arguments and not at the structure of the arguments themselves. In structured
argumentation the way an argument is constructed, by chaining inference rules
starting from premises, is taken into consideration. In this paper we study
methods for assigning an argument its intrinsic strength, based on the
strengths of the premises and inference rules used to form said argument. We
first define a set of principles, which are properties that strength assigning
methods might satisfy. We then propose two such methods and analyse which
principles they satisfy. Finally, we present a generalised system for creating
novel strength assigning methods and speak to the properties of this system
regarding the proposed principles.

    

### [[2109.00381] Building a Legal Dialogue System: Development Process, Challenges and Opportunities](http://arxiv.org/abs/2109.00381)


  This paper presents key principles and solutions to the challenges faced in
designing a domain-specific conversational agent for the legal domain. It
includes issues of scope, platform, architecture and preparation of input data.
It provides functionality in answering user queries and recording user
information including contact details and case-related information. It utilises
deep learning technology built upon Amazon Web Services (AWS) LEX in
combination with AWS Lambda. Due to lack of publicly available data, we
identified two methods including crowdsourcing experiments and archived
enquiries to develop a number of linguistic resources. This includes a training
dataset, set of predetermined responses for the conversational agent, a set of
regression test cases and a further conversation test set. We propose a
hierarchical bot structure that facilitates multi-level delegation and report
model accuracy on the regression test set. Additionally, we highlight features
that are added to the bot to improve the conversation flow and overall user
experience.

    

### [[2109.00388] Boolean proportions](http://arxiv.org/abs/2109.00388)


  Analogy-making is at the core of human intelligence and creativity with
applications to such diverse tasks as commonsense reasoning, learning, language
acquisition, and story telling. This paper studies analogical proportions
between booleans of the form `$a$ is to $b$ what $c$ is to $d$' called boolean
proportions. Technically, we instantiate an abstract algebraic framework of
analogical proportions -- recently introduced by the author -- in the boolean
domain consisting of the truth values true and false together with boolean
functions. It turns out that our notion of boolean proportions has appealing
mathematical properties and that it coincides with a prominent model of boolean
proportions in the general case. In a broader sense, this paper is a further
step towards a theory of analogical reasoning and learning systems with
potential applications to fundamental AI-problems like commonsense reasoning
and computational learning and creativity.

    

### [[2109.00412] Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis](http://arxiv.org/abs/2109.00412)


  In multimodal sentiment analysis (MSA), the performance of a model highly
depends on the quality of synthesized embeddings. These embeddings are
generated from the upstream process called multimodal fusion, which aims to
extract and combine the input unimodal raw data to produce a richer multimodal
representation. Previous work either back-propagates the task loss or
manipulates the geometric property of feature spaces to produce favorable
fusion results, which neglects the preservation of critical task-related
information that flows from input to the fusion results. In this work, we
propose a framework named MultiModal InfoMax (MMIM), which hierarchically
maximizes the Mutual Information (MI) in unimodal input pairs (inter-modality)
and between multimodal fusion result and unimodal input in order to maintain
task-related information through multimodal fusion. The framework is jointly
trained with the main task (MSA) to improve the performance of the downstream
MSA task. To address the intractable issue of MI bounds, we further formulate a
set of computationally simple parametric and non-parametric methods to
approximate their truth value. Experimental results on the two widely used
datasets demonstrate the efficacy of our approach. The implementation of this
work is publicly available at
this https URL.

    

### [[2109.00414] Balancing Performance and Human Autonomy with Implicit Guidance Agent](http://arxiv.org/abs/2109.00414)


  The human-agent team, which is a problem in which humans and autonomous
agents collaborate to achieve one task, is typical in human-AI collaboration.
For effective collaboration, humans want to have an effective plan, but in
realistic situations, they might have difficulty calculating the best plan due
to cognitive limitations. In this case, guidance from an agent that has many
computational resources may be useful. However, if an agent guides the human
behavior explicitly, the human may feel that they have lost autonomy and are
being controlled by the agent. We therefore investigated implicit guidance
offered by means of an agent's behavior. With this type of guidance, the agent
acts in a way that makes it easy for the human to find an effective plan for a
collaborative task, and the human can then improve the plan. Since the human
improves their plan voluntarily, he or she maintains autonomy. We modeled a
collaborative agent with implicit guidance by integrating the Bayesian Theory
of Mind into existing collaborative-planning algorithms and demonstrated
through a behavioral experiment that implicit guidance is effective for
enabling humans to maintain a balance between improving their plans and
retaining autonomy.

    

### [[2109.00417] Masked Adversarial Generation for Neural Machine Translation](http://arxiv.org/abs/2109.00417)


  Attacking Neural Machine Translation models is an inherently combinatorial
task on discrete sequences, solved with approximate heuristics. Most methods
use the gradient to attack the model on each sample independently. Instead of
mechanically applying the gradient, could we learn to produce meaningful
adversarial attacks ? In contrast to existing approaches, we learn to attack a
model by training an adversarial generator based on a language model. We
propose the Masked Adversarial Generation (MAG) model, that learns to perturb
the translation model throughout the training process. The experiments show
that it improves the robustness of machine translation models, while being
faster than competing methods.

    

### [[2109.00430] M^2-MedDialog: A Dataset and Benchmarks for Multi-domain Multi-service Medical Dialogues](http://arxiv.org/abs/2109.00430)


  Medical dialogue systems (MDSs) aim to assist doctors and patients with a
range of professional medical services, i.e., diagnosis, consultation, and
treatment. However, one-stop MDS is still unexplored because: (1) no dataset
has so large-scale dialogues contains both multiple medical services and
fine-grained medical labels (i.e., intents, slots, values); (2) no model has
addressed a MDS based on multiple-service conversations in a unified framework.
In this work, we first build a Multiple-domain Multiple-service medical
dialogue (M^2-MedDialog)dataset, which contains 1,557 conversations between
doctors and patients, covering 276 types of diseases, 2,468 medical entities,
and 3 specialties of medical services. To the best of our knowledge, it is the
only medical dialogue dataset that includes both multiple medical services and
fine-grained medical labels. Then, we formulate a one-stop MDS as a
sequence-to-sequence generation problem. We unify a MDS with causal language
modeling and conditional causal language modeling, respectively. Specifically,
we employ several pretrained models (i.e., BERT-WWM, BERT-MED, GPT2, and MT5)
and their variants to get benchmarks on M^2-MedDialog dataset. We also propose
pseudo labeling and natural perturbation methods to expand M2-MedDialog dataset
and enhance the state-of-the-art pretrained models. We demonstrate the results
achieved by the benchmarks so far through extensive experiments on
M2-MedDialog. We release the dataset, the code, as well as the evaluation
scripts to facilitate future research in this important research direction.

    

### [[2109.00435] Proceedings of KDD 2020 Workshop on Data-driven Humanitarian Mapping: Harnessing Human-Machine Intelligence for High-Stake Public Policy and Resilience Planning](http://arxiv.org/abs/2109.00435)


  Humanitarian challenges, including natural disasters, food insecurity,
climate change, racial and gender violence, environmental crises, the COVID-19
coronavirus pandemic, human rights violations, and forced displacements,
disproportionately impact vulnerable communities worldwide. According to UN
OCHA, 235 million people will require humanitarian assistance in 20211 .
Despite these growing perils, there remains a notable paucity of data science
research to scientifically inform equitable public policy decisions for
improving the livelihood of at-risk populations. Scattered data science efforts
exist to address these challenges, but they remain isolated from practice and
prone to algorithmic harms concerning lack of privacy, fairness,
interpretability, accountability, transparency, and ethics. Biases in
data-driven methods carry the risk of amplifying inequalities in high-stakes
policy decisions that impact the livelihood of millions of people.
Consequently, proclaimed benefits of data-driven innovations remain
inaccessible to policymakers, practitioners, and marginalized communities at
the core of humanitarian actions and global development. To help fill this gap,
we propose the Data-driven Humanitarian Mapping Research Program, which focuses
on developing novel data science methodologies that harness human-machine
intelligence for high-stakes public policy and resilience planning.

    

### [[2109.00449] Planning from video game descriptions](http://arxiv.org/abs/2109.00449)


  This project proposes a methodology for the automatic generation of action
models from video game dynamics descriptions, as well as its integration with a
planning agent for the execution and monitoring of the plans. Planners use
these action models to get the deliberative behaviour for an agent in many
different video games and, combined with a reactive module, solve deterministic
and no-deterministic levels. Experimental results validate the methodology and
prove that the effort put by a knowledge engineer can be greatly reduced in the
definition of such complex domains. Furthermore, benchmarks of the domains has
been produced that can be of interest to the international planning community
to evaluate planners in international planning competitions.

    

### [[2109.00460] From Movement Kinematics to Object Properties: Online Recognition of Human Carefulness](http://arxiv.org/abs/2109.00460)


  When manipulating objects, humans finely adapt their motions to the
characteristics of what they are handling. Thus, an attentive observer can
foresee hidden properties of the manipulated object, such as its weight,
temperature, and even whether it requires special care in manipulation. This
study is a step towards endowing a humanoid robot with this last capability.
Specifically, we study how a robot can infer online, from vision alone, whether
or not the human partner is careful when moving an object. We demonstrated that
a humanoid robot could perform this inference with high accuracy (up to 81.3%)
even with a low-resolution camera. Only for short movements without obstacles,
carefulness recognition was insufficient. The prompt recognition of movement
carefulness from observing the partner's action will allow robots to adapt
their actions on the object to show the same degree of care as their human
partners.

    

### [[2109.00484] Impossibility Results in AI: A Survey](http://arxiv.org/abs/2109.00484)


  An impossibility theorem demonstrates that a particular problem or set of
problems cannot be solved as described in the claim. Such theorems put limits
on what is possible to do concerning artificial intelligence, especially the
super-intelligent one. As such, these results serve as guidelines, reminders,
and warnings to AI safety, AI policy, and governance researchers. These might
enable solutions to some long-standing questions in the form of formalizing
theories in the framework of constraint satisfaction without committing to one
option. In this paper, we have categorized impossibility theorems applicable to
the domain of AI into five categories: deduction, indistinguishability,
induction, tradeoffs, and intractability. We found that certain theorems are
too specific or have implicit assumptions that limit application. Also, we
added a new result (theorem) about the unfairness of explainability, the first
explainability-related result in the induction category. We concluded that
deductive impossibilities deny 100%-guarantees for security. In the end, we
give some ideas that hold potential in explainability, controllability, value
alignment, ethics, and group decision-making. They can be deepened by further
investigation.

    

### [[1909.04559] Learning Hierarchically Structured Concepts](http://arxiv.org/abs/1909.04559)


  We study the question of how concepts that have structure get represented in
the brain. Specifically, we introduce a model for hierarchically structured
concepts and we show how a biologically plausible neural network can recognize
these concepts, and how it can learn them in the first place. Our main goal is
to introduce a general framework for these tasks and prove formally how both
(recognition and learning) can be achieved.
We show that both tasks can be accomplished even in presence of noise. For
learning, we analyze Oja's rule formally, a well-known biologically-plausible
rule for adjusting the weights of synapses. We complement the learning results
with lower bounds asserting that, in order to recognize concepts of a certain
hierarchical depth, neural networks must have a corresponding number of layers.

    

### [[2101.06177] Hierarchical Width-Based Planning and Learning](http://arxiv.org/abs/2101.06177)


  Width-based search methods have demonstrated state-of-the-art performance in
a wide range of testbeds, from classical planning problems to image-based
simulators such as Atari games. These methods scale independently of the size
of the state-space, but exponentially in the problem width. In practice,
running the algorithm with a width larger than 1 is computationally
intractable, prohibiting IW from solving higher width problems. In this paper,
we present a hierarchical algorithm that plans at two levels of abstraction. A
high-level planner uses abstract features that are incrementally discovered
from low-level pruning decisions. We illustrate this algorithm in classical
planning PDDL domains as well as in pixel-based simulator domains. In classical
planning, we show how IW(1) at two levels of abstraction can solve problems of
width 2. For pixel-based domains, we show how in combination with a learned
policy and a learned value function, the proposed hierarchical IW can
outperform current flat IW-based planners in Atari games with sparse rewards.

    

### [[2103.01834] A Data-Centric Framework for Composable NLP Workflows](http://arxiv.org/abs/2103.01834)


  Empirical natural language processing (NLP) systems in application domains
(e.g., healthcare, finance, education) involve interoperation among multiple
components, ranging from data ingestion, human annotation, to text retrieval,
analysis, generation, and visualization. We establish a unified open-source
framework to support fast development of such sophisticated NLP workflows in a
composable manner. The framework introduces a uniform data representation to
encode heterogeneous results by a wide range of NLP tasks. It offers a large
repository of processors for NLP tasks, visualization, and annotation, which
can be easily assembled with full interoperability under the unified
representation. The highly extensible framework allows plugging in custom
processors from external off-the-shelf NLP and deep learning libraries. The
whole framework is delivered through two modularized yet integratable
open-source projects, namely Forte (for workflow infrastructure and NLP
function processors) and Stave (for user interaction, visualization, and
annotation).

    

### [[2104.03149] Beyond Question-Based Biases: Assessing Multimodal Shortcut Learning in Visual Question Answering](http://arxiv.org/abs/2104.03149)


  We introduce an evaluation methodology for visual question answering (VQA) to
better diagnose cases of shortcut learning. These cases happen when a model
exploits spurious statistical regularities to produce correct answers but does
not actually deploy the desired behavior. There is a need to identify possible
shortcuts in a dataset and assess their use before deploying a model in the
real world. The research community in VQA has focused exclusively on
question-based shortcuts, where a model might, for example, answer "What is the
color of the sky" with "blue" by relying mostly on the question-conditional
training prior and give little weight to visual evidence. We go a step further
and consider multimodal shortcuts that involve both questions and images. We
first identify potential shortcuts in the popular VQA v2 training set by mining
trivial predictive rules such as co-occurrences of words and visual elements.
We then introduce VQA-CounterExamples (VQA-CE), an evaluation protocol based on
our subset of CounterExamples i.e. image-question-answer triplets where our
rules lead to incorrect answers. We use this new evaluation in a large-scale
study of existing approaches for VQA. We demonstrate that even state-of-the-art
models perform poorly and that existing techniques to reduce biases are largely
ineffective in this context. Our findings suggest that past work on
question-based biases in VQA has only addressed one facet of a complex issue.
The code for our method is available at
this https URL.

    

### [[2108.12536] DASH: Modularized Human Manipulation Simulation with Vision and Language for Embodied AI](http://arxiv.org/abs/2108.12536)


  Creating virtual humans with embodied, human-like perceptual and actuation
constraints has the promise to provide an integrated simulation platform for
many scientific and engineering applications. We present Dynamic and Autonomous
Simulated Human (DASH), an embodied virtual human that, given natural language
commands, performs grasp-and-stack tasks in a physically-simulated cluttered
environment solely using its own visual perception, proprioception, and touch,
without requiring human motion data. By factoring the DASH system into a vision
module, a language module, and manipulation modules of two skill categories, we
can mix and match analytical and machine learning techniques for different
modules so that DASH is able to not only perform randomly arranged tasks with a
high success rate, but also do so under anthropomorphic constraints and with
fluid and diverse motions. The modular design also favors analysis and
extensibility to more complex manipulation skills.

    

### [[2109.00311] Termination Analysis for the $π$-Calculus by Reduction to Sequential Program Termination](http://arxiv.org/abs/2109.00311)


  We propose an automated method for proving termination of $\pi$-calculus
processes, based on a reduction to termination of sequential programs: we
translate a $\pi$-calculus process to a sequential program, so that the
termination of the latter implies that of the former. We can then use an
off-the-shelf termination verification tool to check termination of the
sequential program. Our approach has been partially inspired by Deng and
Sangiorgi's termination analysis for the $\pi$-calculus, and checks that there
is no infinite chain of communications on replicated input channels, by
converting such a chain of communications to a chain of recursive function
calls in the target sequential program. We have implemented an automated tool
based on the proposed method and confirmed its effectiveness.

    

### [[2109.00445] Linked visualisations via Galois dependencies](http://arxiv.org/abs/2109.00445)


  We present new language-based dynamic analysis techniques for linking
visualisations and other structured outputs to data in a fine-grained way,
allowing a user to interactively explore how data attributes map to visual or
other output elements by selecting (focusing on) substructures of interest.
This can help both programmers and end-users understand how data sources and
complex outputs are related, which can be a challenge even for someone with
expert knowledge of the problem domain. Our approach builds on bidirectional
program slicing techiques based on Galois connections, which provide desirable
round-tripping properties.
Unlike the prior work in program slicing, our approach allows selections to
be negated. In a setting with negation, the bidirectional analysis has a De
Morgan dual, which can be used to link different outputs generated from the
same input. This offers a principled language-based foundation for a popular
interactive visualisation feature called brushing and linking where selections
in one chart automatically select corresponding elements in another related
chart. Although such view coordination features are valuable comprehension
aids, they tend be to hard-coded into specific applications or libraries, or
require programmer effort.

    