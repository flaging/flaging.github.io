
## 2021-12-1

### [[2111.14994] A General Purpose Data and Query Privacy Preserving Protocol for Wireless Sensor Networks](http://arxiv.org/abs/2111.14994)


  Wireless Sensor Networks (WSNs) are composed of a large number of spatially
distributed devices equipped with sensing technology and interlinked via radio
signaling. A WSN deployed for monitoring purposes can provide a ubiquitous view
over the monitored environment. However, the management of collected data is
very resource-consuming and raises security and privacy issues. In this paper,
we propose a privacy preserving protocol for collecting aggregated data from
WSNs. The protocol relies on the Onion Routing technique to provide uniformly
distributed network traffic and confine the knowledge a foreign actor can gain
from monitoring messages traveling the network. Our solution employs the
computing power of nodes in the network by conveying them general-purpose
computer code for in-situ processing and aggregation of data sourcing from
multiple sensor nodes. We complement our work with a simulation of the proposed
solution using the network simulator ns-3. Results of the simulation give an
overview of the scalability of the solution and highlight potential
constraints.

    

### [[2111.15013] DeepCQ+: Robust and Scalable Routing with Multi-Agent Deep Reinforcement Learning for Highly Dynamic Networks](http://arxiv.org/abs/2111.15013)


  Highly dynamic mobile ad-hoc networks (MANETs) remain as one of the most
challenging environments to develop and deploy robust, efficient, and scalable
routing protocols. In this paper, we present DeepCQ+ routing protocol which, in
a novel manner integrates emerging multi-agent deep reinforcement learning
(MADRL) techniques into existing Q-learning-based routing protocols and their
variants and achieves persistently higher performance across a wide range of
topology and mobility configurations. While keeping the overall protocol
structure of the Q-learning-based routing protocols, DeepCQ+ replaces
statically configured parameterized thresholds and hand-written rules with
carefully designed MADRL agents such that no configuration of such parameters
is required a priori. Extensive simulation shows that DeepCQ+ yields
significantly increased end-to-end throughput with lower overhead and no
apparent degradation of end-to-end delays (hop counts) compared to its
Q-learning based counterparts. Qualitatively, and perhaps more significantly,
DeepCQ+ maintains remarkably similar performance gains under many scenarios
that it was not trained for in terms of network sizes, mobility conditions, and
traffic dynamics. To the best of our knowledge, this is the first successful
application of the MADRL framework for the MANET routing problem that
demonstrates a high degree of scalability and robustness even under
environments that are outside the trained range of scenarios. This implies that
our MARL-based DeepCQ+ design solution significantly improves the performance
of Q-learning based CQ+ baseline approach for comparison and increases its
practicality and explainability because the real-world MANET environment will
likely vary outside the trained range of MANET scenarios. Additional techniques
to further increase the gains in performance and scalability are discussed.

    

### [[2111.15029] Reinforcement Learning Algorithm for Traffic Steering in Heterogeneous Network](http://arxiv.org/abs/2111.15029)


  Heterogeneous radio access networks require efficient traffic steering
methods to reach near-optimal results in order to maximize network capacity.
This paper aims to propose a novel traffic steering algorithm for usage in
HetNets, which utilizes a reinforcement learning algorithm in combination with
an artificial neural network to maximize total user satisfaction in the
simulated cellular network. The novel algorithm was compared with two reference
algorithms using network simulation results. The results prove that the novel
algorithm provides noticeably better efficiency in comparison with reference
algorithms, especially in terms of the number of served users with limited
frequency resources of the radio access network.

    

### [[2111.15087] MAMRL: Exploiting Multi-agent Meta Reinforcement Learning in WAN Traffic Engineering](http://arxiv.org/abs/2111.15087)


  Traffic optimization challenges, such as load balancing, flow scheduling, and
improving packet delivery time, are difficult online decision-making problems
in wide area networks (WAN). Complex heuristics are needed for instance to find
optimal paths that improve packet delivery time and minimize interruptions
which may be caused by link failures or congestion. The recent success of
reinforcement learning (RL) algorithms can provide useful solutions to build
better robust systems that learn from experience in model-free settings.
In this work, we consider a path optimization problem, specifically for
packet routing, in large complex networks. We develop and evaluate a model-free
approach, applying multi-agent meta reinforcement learning (MAMRL) that can
determine the next-hop of each packet to get it delivered to its destination
with minimum time overall. Specifically, we propose to leverage and compare
deep policy optimization RL algorithms for enabling distributed model-free
control in communication networks and present a novel meta-learning-based
framework, MAMRL, for enabling quick adaptation to topology changes. To
evaluate the proposed framework, we simulate with various WAN topologies. Our
extensive packet-level simulation results show that compared to classical
shortest path and traditional reinforcement learning approaches, MAMRL
significantly reduces the average packet delivery time even when network demand
increases; and compared to a non-meta deep policy optimization algorithm, our
results show the reduction of packet loss in much fewer episodes when link
failures occur while offering comparable average packet delivery time.

    

### [[2111.15098] Technical Report: Edge-centric Programming for IoT Applications with EdgeProg](http://arxiv.org/abs/2111.15098)


  IoT application development usually involves separate programming at the
device side and server side. While separate programming style is sufficient for
many simple applications, it is not suitable for many complex applications that
involve complex interactions and intensive data processing. We propose
EdgeProg, an edge-centric programming approach to simplify IoT application
programming, motivated by the increasing popularity of edge computing. With
EdgeProg, users could write application logic in a centralized manner with an
augmented If-This-Then-That (IFTTT) syntax and virtual sensor mechanism. The
program can be processed at the edge server, which can automatically generate
the actual application code and intelligently partition the code into device
code and server code, for achieving the optimal latency. EdgeProg employs
dynamic linking and loading to deploy the device code on a variety of IoT
devices, which do not run any application-specific codes at the start. Results
show that EdgeProg achieves an average reduction of 20.96%, 27.8% and 79.41% in
terms of execution latency, energy consumption, and lines of code compared with
state-of-the-art approaches.

    

### [[2111.15276] COREATTACK: Breaking Up the Core Structure of Graphs](http://arxiv.org/abs/2111.15276)


  The concept of k-core in complex networks plays a key role in many
applications, e.g., understanding the global structure, or identifying
central/critical nodes, of a network. A malicious attacker with jamming ability
can exploit the vulnerability of the k-core structure to attack the network and
invalidate the network analysis methods, e.g., reducing the k-shell values of
nodes can deceive graph algorithms, leading to the wrong decisions. In this
paper, we investigate the robustness of the k-core structure under adversarial
attacks by deleting edges, for the first time. Firstly, we give the general
definition of targeted k-core attack, map it to the set cover problem which is
NP-hard, and further introduce a series of evaluation metrics to measure the
performance of attack methods. Then, we propose $Q$ index theoretically as the
probability that the terminal node of an edge does not belong to the innermost
core, which is further used to guide the design of our heuristic attack
methods, namely COREATTACK and GreedyCOREATTACK. The experiments on a variety
of real-world networks demonstrate that our methods behave much better than a
series of baselines, in terms of much smaller Edge Change Rate (ECR) and False
Attack Rate (FAR), achieving state-of-the-art attack performance. More
impressively, for certain real-world networks, only deleting one edge from the
k-core may lead to the collapse of the innermost core, even if this core
contains dozens of nodes. Such a phenomenon indicates that the k-core structure
could be extremely vulnerable under adversarial attacks, and its robustness
thus should be carefully addressed to ensure the security of many graph
algorithms.

    

### [[2111.15296] BrainScaleS Large Scale Spike Communication using Extoll](http://arxiv.org/abs/2111.15296)


  The BrainScaleS Neuromorphic Computing System is currently connected to a
compute cluster via Gigabit-Ethernet network technology. This is convenient for
the currently used experiment mode, where neuronal networks cover at most one
wafer module. When modelling networks of larger size, as for example a full
sized cortical microcircuit model, one has to think about connecting neurons
across wafer modules to larger networks. This can be done, using the Extoll
networking technology, which provides high bandwidth and low latencies, as well
as a low overhead packet protocol format.

    

### [[2111.15301] Optical Wireless Sytems for Spine and Leaf Data Center Downlinks](http://arxiv.org/abs/2111.15301)


  The continually growing demands for traffic as a result of advanced
technologies in 5G and 6G systems offering services with intensive demands such
as IoT and virtual reality applications has resulted in significant performance
expectations of data center networks (DCNs). More specifically, DCNs are
expected to meet high bandwidth connectivity, high throughput, low latency, and
high scalability requirements. However, the current wired DCN architectures
introduce large cabling requirements and limit the ability to reconfigure data
centres as they expand. To that end, wireless technologies such as Optical
Wireless Communication (OWC) have been proposed as a viable and cost-effective
solution to meet the aforementioned requirements. This paper proposes the use
of Infrared (IR) OWC systems that employ Wavelength Division Multiplexing (WDM)
to enhance the DCN communication in the downlink direction; i.e. from Access
Points (APs) in the ceiling, connected to spine switches, to receivers attached
to the top of the racks representing leaf switches. The proposed systems
utilize Angle Diversity Transmitters (ADTs) mounted on the room ceiling to
facilitate inter-rack communication. Two different optical receiver types are
considered, namely Angle Diversity Receivers (ADRs) and Wide Field-of-View
Receivers (WFOVR). The simulation (i.e. channel modeling) results show that our
proposed data center links achieve good data rates in the data centre up to 15
Gbps.

    

### [[2111.15451] Large-Scale Video Analytics through Object-Level Consolidation](http://arxiv.org/abs/2111.15451)


  As the number of installed cameras grows, so do the compute resources
required to process and analyze all the images captured by these cameras. Video
analytics enables new use cases, such as smart cities or autonomous driving. At
the same time, it urges service providers to install additional compute
resources to cope with the demand while the strict latency requirements push
compute towards the end of the network, forming a geographically distributed
and heterogeneous set of compute locations, shared and resource-constrained.
Such landscape (shared and distributed locations) forces us to design new
techniques that can optimize and distribute work among all available locations
and, ideally, make compute requirements grow sublinearly with respect to the
number of cameras installed. In this paper, we present FoMO (Focus on Moving
Objects). This method effectively optimizes multi-camera deployments by
preprocessing images for scenes, filtering the empty regions out, and composing
regions of interest from multiple cameras into a single image that serves as
input for a pre-trained object detection model. Results show that overall
system performance can be increased by 8x while accuracy improves 40% as a
by-product of the methodology, all using an off-the-shelf pre-trained model
with no additional training or fine-tuning.

    

### [[2111.15502] Popcorns-Pro: A Cooperative Network-Server Approach for Data Center Energy Optimization](http://arxiv.org/abs/2111.15502)


  Data centers have become a popular computing platform for various
applications, and they account for nearly 2% of total US energy consumption.
Therefore, it has become important to optimize data center power, and reduce
their energy footprint. Most existing work optimizes power in servers and
networks independently and does not address them together in a holistic fashion
that has the potential to achieve greater power savings. In this article, we
present PopcornsPro, a cooperative server network framework for energy
optimization. We present a comprehensive power model for heterogeneous data
center switches along with low power mode designs in combination with the
server power model. We design job scheduling algorithms that place tasks onto
servers in a power-aware manner, such that servers and network switches can
take effective advantage of low power state and available network link
capacities. Our experimental results show that we are able to achieve
significantly higher savings up to 80% compared to the previously well-known
server and network power optimization policies.

    

### [[2111.14829] Nonparametric Topological Layers in Neural Networks](http://arxiv.org/abs/2111.14829)


  Various topological techniques and tools have been applied to neural networks
in terms of network complexity, explainability, and performance. One
fundamental assumption of this line of research is the existence of a global
(Euclidean) coordinate system upon which the topological layer is constructed.
Despite promising results, such a \textit{topologization} method has yet to be
widely adopted because the parametrization of a topologization layer takes a
considerable amount of time and more importantly, lacks a theoretical
foundation without which the performance of the neural network only achieves
suboptimal performance. This paper proposes a learnable topological layer for
neural networks without requiring a Euclidean space; Instead, the proposed
construction requires nothing more than a general metric space except for an
inner product, i.e., a Hilbert space. Accordingly, the according
parametrization for the proposed topological layer is free of user-specified
hyperparameters, which precludes the costly parametrization stage and the
corresponding possibility of suboptimal networks.

    

### [[2111.14831] MIST-net: Multi-domain Integrative Swin Transformer network for Sparse-View CT Reconstruction](http://arxiv.org/abs/2111.14831)


  The deep learning-based tomographic image reconstruction have been attracting
much attention among these years. The sparse-view data reconstruction is one of
typical underdetermined inverse problems, how to reconstruct high-quality CT
images from dozens of projections is still a challenge in practice. To address
this challenge, in this article we proposed a Multi-domain Integrative Swin
Transformer network (MIST-net). First, the proposed MIST-net incorporated
lavish domain features from data, residual-data, image, and residual-image
using flexible network architectures. Here, the residual-data and
residual-image domains network components can be considered as the data
consistency module to eliminate interpolation errors in both residual data and
image domains, and then further retain image details. Second, to detect the
image features and further protect image edge, the trainable Sobel Filter was
incorporated into the network to improve the encode-decode ability. Third, with
the classical Swin transformer, we further designed the high-quality
reconstruction transformer (i.e., Recformer) to improve the reconstruction
performance. The Recformer inherited the power of Swin transformer to capture
the global and local features of the reconstructed image. The experiments on
the numerical datasets with 48 views demonstrated our proposed MIST-net
provided higher reconstructed image quality with small feature recovery and
edge protection than other competitors including the advanced unrolled
networks. The quantitative results show that our MIST-net also obtained the
best performance. The trained network was transferred to the real cardiac CT
dataset with 48 views, the reconstruction results further validated the
advantages of our MIST-net and further demonstrated the good robustness of our
MIST in clinical applications.

    

### [[2111.14833] Adversarial Attacks in Cooperative AI](http://arxiv.org/abs/2111.14833)


  Single-agent reinforcement learning algorithms in a multi-agent environment
are inadequate for fostering cooperation. If intelligent agents are to interact
and work together to solve complex problems, methods that counter
non-cooperative behavior are needed to facilitate the training of multiple
agents. This is the goal of cooperative AI. Recent work in adversarial machine
learning, however, shows that models (e.g., image classifiers) can be easily
deceived into making incorrect decisions. In addition, some past research in
cooperative AI has relied on new notions of representations, like public
beliefs, to accelerate the learning of optimally cooperative behavior. Hence,
cooperative AI might introduce new weaknesses not investigated in previous
machine learning research. In this paper, our contributions include: (1)
arguing that three algorithms inspired by human-like social intelligence
introduce new vulnerabilities, unique to cooperative AI, that adversaries can
exploit, and (2) an experiment showing that simple, adversarial perturbations
on the agents' beliefs can negatively impact performance. This evidence points
to the possibility that formal representations of social behavior are
vulnerable to adversarial attacks.

    

### [[2111.14834] Self-supervised Autoregressive Domain Adaptation for Time Series Data](http://arxiv.org/abs/2111.14834)


  Unsupervised domain adaptation (UDA) has successfully addressed the domain
shift problem for visual applications. Yet, these approaches may have limited
performance for time series data due to the following reasons. First, they
mainly rely on large-scale dataset (i.e., ImageNet) for the source pretraining,
which is not applicable for time-series data. Second, they ignore the temporal
dimension on the feature space of the source and target domains during the
domain alignment step. Last, most of prior UDA methods can only align the
global features without considering the fine-grained class distribution of the
target domain. To address these limitations, we propose a Self-supervised
Autoregressive Domain Adaptation (SLARDA) framework. In particular, we first
design a self-supervised learning module that utilizes forecasting as an
auxiliary task to improve the transferability of the source features. Second,
we propose a novel autoregressive domain adaptation technique that incorporates
temporal dependency of both source and target features during domain alignment.
Finally, we develop an ensemble teacher model to align the class-wise
distribution in the target domain via a confident pseudo labeling approach.
Extensive experiments have been conducted on three real-world time series
applications with 30 cross-domain scenarios. Results demonstrate that our
proposed SLARDA method significantly outperforms the state-of-the-art
approaches for time series domain adaptation.

    

### [[2111.14836] Low-bit Quantization of Recurrent Neural Network Language Models Using Alternating Direction Methods of Multipliers](http://arxiv.org/abs/2111.14836)


  The high memory consumption and computational costs of Recurrent neural
network language models (RNNLMs) limit their wider application on resource
constrained devices. In recent years, neural network quantization techniques
that are capable of producing extremely low-bit compression, for example,
binarized RNNLMs, are gaining increasing research interests. Directly training
of quantized neural networks is difficult. By formulating quantized RNNLMs
training as an optimization problem, this paper presents a novel method to
train quantized RNNLMs from scratch using alternating direction methods of
multipliers (ADMM). This method can also flexibly adjust the trade-off between
the compression rate and model performance using tied low-bit quantization
tables. Experiments on two tasks: Penn Treebank (PTB), and Switchboard (SWBD)
suggest the proposed ADMM quantization achieved a model size compression factor
of up to 31 times over the full precision baseline RNNLMs. Faster convergence
of 5 times in model training over the baseline binarized RNNLM quantization was
also obtained. Index Terms: Language models, Recurrent neural networks,
Quantization, Alternating direction methods of multipliers.

    

### [[2111.14837] p2pGNN: A Decentralized Graph Neural Network for Node Classification in Peer-to-Peer Networks](http://arxiv.org/abs/2111.14837)


  In this work, we aim to classify nodes of unstructured peer-to-peer networks
with communication uncertainty, such as users of decentralized social networks.
Graph Neural Networks (GNNs) are known to improve the accuracy of simpler
classifiers in centralized settings by leveraging naturally occurring network
links, but graph convolutional layers are challenging to implement in
decentralized settings when node neighbors are not constantly available. We
address this problem by employing decoupled GNNs, where base classifier
predictions and errors are diffused through graphs after training. For these,
we deploy pre-trained and gossip-trained base classifiers and implement
peer-to-peer graph diffusion under communication uncertainty. In particular, we
develop an asynchronous decentralized formulation of diffusion that converges
at the same predictions linearly with respect to communication rate. We
experiment on three real-world graphs with node features and labels and
simulate peer-to-peer networks with uniformly random communication frequencies;
given a portion of known labels, our decentralized graph diffusion achieves
comparable accuracy to centralized GNNs.

    

### [[2111.14838] Evaluating Privacy-Preserving Machine Learning in Critical Infrastructures: A Case Study on Time-Series Classification](http://arxiv.org/abs/2111.14838)


  With the advent of machine learning in applications of critical
infrastructure such as healthcare and energy, privacy is a growing concern in
the minds of stakeholders. It is pivotal to ensure that neither the model nor
the data can be used to extract sensitive information used by attackers against
individuals or to harm whole societies through the exploitation of critical
infrastructure. The applicability of machine learning in these domains is
mostly limited due to a lack of trust regarding the transparency and the
privacy constraints. Various safety-critical use cases (mostly relying on
time-series data) are currently underrepresented in privacy-related
considerations. By evaluating several privacy-preserving methods regarding
their applicability on time-series data, we validated the inefficacy of
encryption for deep learning, the strong dataset dependence of differential
privacy, and the broad applicability of federated methods.

    

### [[2111.14839] PCA-based Category Encoder for Categorical to Numerical Variable Conversion](http://arxiv.org/abs/2111.14839)


  Increasing the cardinality of categorical variables might decrease the
overall performance of ML algorithms. This paper presents a novel computational
preprocessing method to convert categorical to numerical variables for machine
learning (ML) algorithms. In this method, We select and convert three
categorical features to numerical features. First, we choose the threshold
parameter based on the distribution of categories in variables. Then, we use
conditional probabilities to convert each categorical variable into two new
numerical variables, resulting in six new numerical variables in total. After
that, we feed these six numerical variables to the Principal Component Analysis
(PCA) algorithm. Next, we select the whole or partial numbers of Principal
Components (PCs). Finally, by applying binary classification with ten different
classifiers, We measured the performance of the new encoder and compared it
with the other 17 well-known category encoders. The proposed technique achieved
the highest performance related to accuracy and Area under the curve (AUC) on
high cardinality categorical variables using the well-known cybersecurity
NSLKDD dataset. Also, we defined harmonic average metrics to find the best
trade-off between train and test performance and prevent underfitting and
overfitting. Ultimately, the number of newly created numerical variables is
minimal. Consequently, this data reduction improves computational processing
time which might reduce processing data in 5G future telecommunication
networks.

    

### [[2111.14842] Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?](http://arxiv.org/abs/2111.14842)


  Spoken language understanding (SLU) tasks are usually solved by first
transcribing an utterance with automatic speech recognition (ASR) and then
feeding the output to a text-based model. Recent advances in self-supervised
representation learning for speech data have focused on improving the ASR
component. We investigate whether representation learning for speech has
matured enough to replace ASR in SLU. We compare learned speech features from
wav2vec 2.0, state-of-the-art ASR transcripts, and the ground truth text as
input for a novel speech-based named entity recognition task, a cardiac arrest
detection task on real-world emergency calls and two existing SLU benchmarks.
We show that learned speech features are superior to ASR transcripts on three
classification tasks. For machine translation, ASR transcripts are still the
better choice. We highlight the intrinsic robustness of wav2vec 2.0
representations to out-of-vocabulary words as key to better performance.

    

### [[2111.14843] Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds](http://arxiv.org/abs/2111.14843)


  Audio-visual navigation combines sight and hearing to navigate to a
sound-emitting source in an unmapped environment. While recent approaches have
demonstrated the benefits of audio input to detect and find the goal, they
focus on clean and static sound sources and struggle to generalize to unheard
sounds. In this work, we propose the novel dynamic audio-visual navigation
benchmark which requires to catch a moving sound source in an environment with
noisy and distracting sounds. We introduce a reinforcement learning approach
that learns a robust navigation policy for these complex settings. To achieve
this, we propose an architecture that fuses audio-visual information in the
spatial feature space to learn correlations of geometric information inherent
in both local maps and audio signals. We demonstrate that our approach
consistently outperforms the current state-of-the-art by a large margin across
all tasks of moving sounds, unheard sounds, and noisy environments, on two
challenging 3D scanned real-world environments, namely Matterport3D and
Replica. The benchmark is available at this http URL.

    

### [[2111.14844] Evaluation of Machine Learning Techniques for Forecast Uncertainty Quantification](http://arxiv.org/abs/2111.14844)


  Producing an accurate weather forecast and a reliable quantification of its
uncertainty is an open scientific challenge. Ensemble forecasting is, so far,
the most successful approach to produce relevant forecasts along with an
estimation of their uncertainty. The main limitations of ensemble forecasting
are the high computational cost and the difficulty to capture and quantify
different sources of uncertainty, particularly those associated with model
errors. In this work proof-of-concept model experiments are conducted to
examine the performance of ANNs trained to predict a corrected state of the
system and the state uncertainty using only a single deterministic forecast as
input. We compare different training strategies: one based on a direct training
using the mean and spread of an ensemble forecast as target, the other ones
rely on an indirect training strategy using a deterministic forecast as target
in which the uncertainty is implicitly learned from the data. For the last
approach two alternative loss functions are proposed and evaluated, one based
on the data observation likelihood and the other one based on a local
estimation of the error. The performance of the networks is examined at
different lead times and in scenarios with and without model errors.
Experiments using the Lorenz'96 model show that the ANNs are able to emulate
some of the properties of ensemble forecasts like the filtering of the most
unpredictable modes and a state-dependent quantification of the forecast
uncertainty. Moreover, ANNs provide a reliable estimation of the forecast
uncertainty in the presence of model error.

    

### [[2111.14874] Weighing the Milky Way and Andromeda with Artificial Intelligence](http://arxiv.org/abs/2111.14874)


  We present new constraints on the masses of the halos hosting the Milky Way
and Andromeda galaxies derived using graph neural networks. Our models, trained
on thousands of state-of-the-art hydrodynamic simulations of the CAMELS
project, only make use of the positions, velocities and stellar masses of the
galaxies belonging to the halos, and are able to perform likelihood-free
inference on halo masses while accounting for both cosmological and
astrophysical uncertainties. Our constraints are in agreement with estimates
from other traditional methods.

    

### [[2111.14889] Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems](http://arxiv.org/abs/2111.14889)


  Koopman operators are infinite-dimensional operators that globally linearize
nonlinear dynamical systems, making their spectral information useful for
understanding dynamics. However, Koopman operators can have continuous spectra
and infinite-dimensional invariant subspaces, making computing their spectral
information a considerable challenge. This paper describes data-driven
algorithms with rigorous convergence guarantees for computing spectral
information of Koopman operators from trajectory data. We introduce residual
dynamic mode decomposition (ResDMD), which provides the first scheme for
computing the spectra and pseudospectra of general Koopman operators from
snapshot data without spectral pollution. Using the resolvent operator and
ResDMD, we also compute smoothed approximations of spectral measures associated
with measure-preserving dynamical systems. We prove explicit convergence
theorems for our algorithms, which can achieve high-order convergence even for
chaotic systems, when computing the density of the continuous spectrum and the
discrete spectrum. We demonstrate our algorithms on the tent map, Gauss
iterated map, nonlinear pendulum, double pendulum, Lorenz system, and an
$11$-dimensional extended Lorenz system. Finally, we provide kernelized
variants of our algorithms for dynamical systems with a high-dimensional
state-space. This allows us to compute the spectral measure associated with the
dynamics of a protein molecule that has a 20,046-dimensional state-space, and
compute nonlinear Koopman modes with error bounds for turbulent flow past
aerofoils with Reynolds number $>10^5$ that has a 295,122-dimensional
state-space.

    

### [[2111.14905] Bounding the Last Mile: Efficient Learned String Indexing](http://arxiv.org/abs/2111.14905)


  We introduce the RadixStringSpline (RSS) learned index structure for
efficiently indexing strings. RSS is a tree of radix splines each indexing a
fixed number of bytes. RSS approaches or exceeds the performance of traditional
string indexes while using 7-70$\times$ less memory. RSS achieves this by using
the minimal string prefix to sufficiently distinguish the data unlike most
learned approaches which index the entire string. Additionally, the
bounded-error nature of RSS accelerates the last mile search and also enables a
memory-efficient hash-table lookup accelerator. We benchmark RSS on several
real-world string datasets against ART and HOT. Our experiments suggest this
line of research may be promising for future memory-intensive database
applications.

    

### [[2111.14911] Optimizing High-Dimensional Physics Simulations via Composite Bayesian Optimization](http://arxiv.org/abs/2111.14911)


  Physical simulation-based optimization is a common task in science and
engineering. Many such simulations produce image- or tensor-based outputs where
the desired objective is a function of those outputs, and optimization is
performed over a high-dimensional parameter space. We develop a Bayesian
optimization method leveraging tensor-based Gaussian process surrogates and
trust region Bayesian optimization to effectively model the image outputs and
to efficiently optimize these types of simulations, including a radio-frequency
tower configuration problem and an optical design problem.

    

### [[2111.14923] Equitable modelling of brain imaging by counterfactual augmentation with morphologically constrained 3D deep generative models](http://arxiv.org/abs/2111.14923)


  We describe Countersynth, a conditional generative model of diffeomorphic
deformations that induce label-driven, biologically plausible changes in
volumetric brain images. The model is intended to synthesise counterfactual
training data augmentations for downstream discriminative modelling tasks where
fidelity is limited by data imbalance, distributional instability, confounding,
or underspecification, and exhibits inequitable performance across distinct
subpopulations. Focusing on demographic attributes, we evaluate the quality of
synthesized counterfactuals with voxel-based morphometry, classification and
regression of the conditioning attributes, and the Fréchet inception
distance. Examining downstream discriminative performance in the context of
engineered demographic imbalance and confounding, we use UK Biobank magnetic
resonance imaging data to benchmark CounterSynth augmentation against current
solutions to these problems. We achieve state-of-the-art improvements, both in
overall fidelity and equity. The source code for CounterSynth is available
online.

    

### [[2111.14924] Architecture Matters: Investigating the Influence of Differential Privacy on Neural Network Design](http://arxiv.org/abs/2111.14924)


  One barrier to more widespread adoption of differentially private neural
networks is the entailed accuracy loss. To address this issue, the relationship
between neural network architectures and model accuracy under differential
privacy constraints needs to be better understood. As a first step, we test
whether extant knowledge on architecture design also holds in the
differentially private setting. Our findings show that it does not;
architectures that perform well without differential privacy, do not
necessarily do so with differential privacy. Consequently, extant knowledge on
neural network architecture design cannot be seamlessly translated into the
differential privacy context. Future research is required to better understand
the relationship between neural network architectures and model accuracy to
enable better architecture design choices under differential privacy
constraints.

    

### [[2111.14932] Learning with Noisy Labels by Efficient Transition Matrix Estimation to Combat Label Miscorrection](http://arxiv.org/abs/2111.14932)


  Recent studies on learning with noisy labels have shown remarkable
performance by exploiting a small clean dataset. In particular, model agnostic
meta-learning-based label correction methods further improve performance by
correcting noisy labels on the fly. However, there is no safeguard on the label
miscorrection, resulting in unavoidable performance degradation. Moreover,
every training step requires at least three back-propagations, significantly
slowing down the training speed. To mitigate these issues, we propose a robust
and efficient method that learns a label transition matrix on the fly.
Employing the transition matrix makes the classifier skeptical about all the
corrected samples, which alleviates the miscorrection issue. We also introduce
a two-head architecture to efficiently estimate the label transition matrix
every iteration within a single back-propagation, so that the estimated matrix
closely follows the shifting noise distribution induced by label correction.
Extensive experiments demonstrate that our approach shows the best performance
in training efficiency while having comparable or better accuracy than existing
methods.

    

### [[2111.14934] GAN-CNMP: An Interactive Generative Drawing Tool](http://arxiv.org/abs/2111.14934)


  Sketches are abstract representations of visual perception and visuospatial
construction. In this work, we proposed a new framework, GAN-CNMP, that
incorporates a novel adversarial loss on CNMP to increase sketch smoothness and
consistency. Through the experiments, we show that our model can be trained
with few unlabeled samples, can construct distributions automatically in the
latent space, and produces better results than the base model in terms of shape
consistency and smoothness.

    

### [[2111.14938] Distribution Shift in Airline Customer Behavior during COVID-19](http://arxiv.org/abs/2111.14938)


  Traditional AI approaches in customized (personalized) contextual pricing
applications assume that the data distribution at the time of online pricing is
similar to that observed during training. However, this assumption may be
violated in practice because of the dynamic nature of customer buying patterns,
particularly due to unanticipated system shocks such as COVID-19. We study the
changes in customer behavior for a major airline during the COVID-19 pandemic
by framing it as a covariate shift and concept drift detection problem. We
identify which customers changed their travel and purchase behavior and the
attributes affecting that change using (i) Fast Generalized Subset Scanning and
(ii) Causal Forests. In our experiments with simulated and real-world data, we
present how these two techniques can be used through qualitative analysis.

    

### [[2111.14948] Image denoising by Super Neurons: Why go deep?](http://arxiv.org/abs/2111.14948)


  Classical image denoising methods utilize the non-local self-similarity
principle to effectively recover image content from noisy images. Current
state-of-the-art methods use deep convolutional neural networks (CNNs) to
effectively learn the mapping from noisy to clean images. Deep denoising CNNs
manifest a high learning capacity and integrate non-local information owing to
the large receptive field yielded by numerous cascade of hidden layers.
However, deep networks are also computationally complex and require large data
for training. To address these issues, this study draws the focus on the
Self-organized Operational Neural Networks (Self-ONNs) empowered by a novel
neuron model that can achieve a similar or better denoising performance with a
compact and shallow model. Recently, the concept of super-neurons has been
introduced which augment the non-linear transformations of generative neurons
by utilizing non-localized kernel locations for an enhanced receptive field
size. This is the key accomplishment which renders the need for a deep network
configuration. As the integration of non-local information is known to benefit
denoising, in this work we investigate the use of super neurons for both
synthetic and real-world image denoising. We also discuss the practical issues
in implementing the super neuron model on GPUs and propose a trade-off between
the heterogeneity of non-localized operations and computational complexity. Our
results demonstrate that with the same width and depth, Self-ONNs with super
neurons provide a significant boost of denoising performance over the networks
with generative and convolutional neurons for both denoising tasks. Moreover,
results demonstrate that Self-ONNs with super neurons can achieve a competitive
and superior synthetic denoising performances than well-known deep CNN
denoisers for synthetic and real-world denoising, respectively.

    

### [[2111.14951] Expressive Communication: A Common Framework for Evaluating Developments in Generative Models and Steering Interfaces](http://arxiv.org/abs/2111.14951)


  There is an increasing interest from ML and HCI communities in empowering
creators with better generative models and more intuitive interfaces with which
to control them. In music, ML researchers have focused on training models
capable of generating pieces with increasing long-range structure and musical
coherence, while HCI researchers have separately focused on designing steering
interfaces that support user control and ownership. In this study, we
investigate through a common framework how developments in both models and user
interfaces are important for empowering co-creation where the goal is to create
music that communicates particular imagery or ideas (e.g., as is common for
other purposeful tasks in music creation like establishing mood or creating
accompanying music for another media). Our study is distinguished in that it
measures communication through both composer's self-reported experiences, and
how listeners evaluate this communication through the music. In an evaluation
study with 26 composers creating 100+ pieces of music and listeners providing
1000+ head-to-head comparisons, we find that more expressive models and more
steerable interfaces are important and complementary ways to make a difference
in composers communicating through music and supporting their creative
empowerment.

    

### [[2111.14955] Privacy-Preserving Serverless Edge Learning with Decentralized Small Data](http://arxiv.org/abs/2111.14955)


  In the last decade, data-driven algorithms outperformed traditional
optimization-based algorithms in many research areas, such as computer vision,
natural language processing, etc. However, extensive data usages bring a new
challenge or even threat to deep learning algorithms, i.e., privacy-preserving.
Distributed training strategies have recently become a promising approach to
ensure data privacy when training deep models. This paper extends conventional
serverless platforms with serverless edge learning architectures and provides
an efficient distributed training framework from the networking perspective.
This framework dynamically orchestrates available resources among heterogeneous
physical units to efficiently fulfill deep learning objectives. The design
jointly considers learning task requests and underlying infrastructure
heterogeneity, including last-mile transmissions, computation abilities of
mobile devices, edge and cloud computing centers, and devices battery status.
Furthermore, to significantly reduce distributed training overheads,
small-scale data training is proposed by integrating with a general, simple
data classifier. This low-load enhancement can seamlessly work with various
distributed deep models to improve communications and computation efficiencies
during the training phase. Finally, open challenges and future research
directions encourage the research community to develop efficient distributed
deep learning techniques.

    

### [[2111.14956] Third-Party Hardware IP Assurance against Trojans through Supervised Learning and Post-processing](http://arxiv.org/abs/2111.14956)


  System-on-chip (SoC) developers increasingly rely on pre-verified hardware
intellectual property (IP) blocks acquired from untrusted third-party vendors.
These IPs might contain hidden malicious functionalities or hardware Trojans to
compromise the security of the fabricated SoCs. Recently, supervised machine
learning (ML) techniques have shown promising capability in identifying nets of
potential Trojans in third party IPs (3PIPs). However, they bring several major
challenges. First, they do not guide us to an optimal choice of features that
reliably covers diverse classes of Trojans. Second, they require multiple
Trojan-free/trusted designs to insert known Trojans and generate a trained
model. Even if a set of trusted designs are available for training, the suspect
IP could be inherently very different from the set of trusted designs, which
may negatively impact the verification outcome. Third, these techniques only
identify a set of suspect Trojan nets that require manual intervention to
understand the potential threat. In this paper, we present VIPR, a systematic
machine learning (ML) based trust verification solution for 3PIPs that
eliminates the need for trusted designs for training. We present a
comprehensive framework, associated algorithms, and a tool flow for obtaining
an optimal set of features, training a targeted machine learning model,
detecting suspect nets, and identifying Trojan circuitry from the suspect nets.
We evaluate the framework on several Trust-Hub Trojan benchmarks and provide a
comparative analysis of detection performance across different trained models,
selection of features, and post-processing techniques. The proposed
post-processing algorithms reduce false positives by up to 92.85%.

    

### [[2111.14971] Classification of animal sounds in a hyperdiverse rainforest using Convolutional Neural Networks](http://arxiv.org/abs/2111.14971)


  To protect tropical forest biodiversity, we need to be able to detect it
reliably, cheaply, and at scale. Automated species detection from passively
recorded soundscapes via machine-learning approaches is a promising technique
towards this goal, but it is constrained by the necessity of large training
data sets. Using soundscapes from a tropical forest in Borneo and a
Convolutional Neural Network model (CNN) created with transfer learning, we
investigate i) the minimum viable training data set size for accurate
prediction of call types ('sonotypes'), and ii) the extent to which data
augmentation can overcome the issue of small training data sets. We found that
even relatively high sample sizes (> 80 per call type) lead to mediocre
accuracy, which however improves significantly with data augmentation,
including at extremely small sample sizes, regardless of taxonomic group or
call characteristics. Our results suggest that transfer learning and data
augmentation can make the use of CNNs to classify species' vocalizations
feasible even for small soundscape-based projects with many rare species. Our
open-source method has the potential to enable conservation initiatives become
more evidence-based by using soundscape data in the adaptive management of
biodiversity.

    

### [[2111.14973] MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction](http://arxiv.org/abs/2111.14973)


  Predicting the future behavior of road users is one of the most challenging
and important problems in autonomous driving. Applying deep learning to this
problem requires fusing heterogeneous world state in the form of rich
perception signals and map information, and inferring highly multi-modal
distributions over possible futures. In this paper, we present MultiPath++, a
future prediction model that achieves state-of-the-art performance on popular
benchmarks. MultiPath++ improves the MultiPath architecture by revisiting many
design choices. The first key design difference is a departure from dense
image-based encoding of the input world state in favor of a sparse encoding of
heterogeneous scene elements: MultiPath++ consumes compact and efficient
polylines to describe road features, and raw agent state information directly
(e.g., position, velocity, acceleration). We propose a context-aware fusion of
these elements and develop a reusable multi-context gating fusion component.
Second, we reconsider the choice of pre-defined, static anchors, and develop a
way to learn latent anchor embeddings end-to-end in the model. Lastly, we
explore ensembling and output aggregation techniques -- common in other ML
domains -- and find effective variants for our probabilistic multimodal output
representation. We perform an extensive ablation on these design choices, and
show that our proposed model achieves state-of-the-art performance on the
Argoverse Motion Forecasting Competition and the Waymo Open Dataset Motion
Prediction Challenge.

    

### [[2111.14991] Bayesian Optimization for auto-tuning GPU kernels](http://arxiv.org/abs/2111.14991)


  Finding optimal parameter configurations for tunable GPU kernels is a
non-trivial exercise for large search spaces, even when automated. This poses
an optimization task on a non-convex search space, using an expensive to
evaluate function with unknown derivative. These characteristics make a good
candidate for Bayesian Optimization, which has not been applied to this problem
before. However, the application of Bayesian Optimization to this problem is
challenging. We demonstrate how to deal with the rough, discrete, constrained
search spaces, containing invalid configurations. We introduce a novel
contextual variance exploration factor, as well as new acquisition functions
with improved scalability, combined with an informed acquisition function
selection mechanism. By comparing the performance of our Bayesian Optimization
implementation on various test cases to the existing search strategies in
Kernel Tuner, as well as other Bayesian Optimization implementations, we
demonstrate that our search strategies generalize well and consistently
outperform other search strategies by a wide margin.

    

### [[2111.14998] Harnessing expressive capacity of Machine Learning modeling to represent complex coupling of Earth's auroral space weather regimes](http://arxiv.org/abs/2111.14998)


  We develop multiple Deep Learning (DL) models that advance the
state-of-the-art predictions of the global auroral particle precipitation. We
use observations from low Earth orbiting spacecraft of the electron energy flux
to develop a model that improves global nowcasts (predictions at the time of
observation) of the accelerated particles. Multiple Machine Learning (ML)
modeling approaches are compared, including a novel multi-task model, models
with tail- and distribution-based loss functions, and a spatio-temporally
sparse 2D-convolutional model. We detail the data preparation process as well
as the model development that will be illustrative for many similar time series
global regression problems in space weather and across domains. Our ML
improvements are three-fold: 1) loss function engineering; 2) multi-task
learning; and 3) transforming the task from time series prediction to
spatio-temporal prediction. Notably, the ML models improve prediction of the
extreme events, historically obstinate to accurate specification and indicate
that increased expressive capacity provided by ML innovation can address grand
challenges in the science of space weather.

    

### [[2111.15000] Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes](http://arxiv.org/abs/2111.15000)


  Machine learning has been widely adopted in many domains, including
high-stakes applications such as healthcare, finance, and criminal justice. To
address concerns of fairness, accountability and transparency, predictions made
by machine learning models in these critical domains must be interpretable. One
line of work approaches this challenge by integrating the power of deep neural
networks and the interpretability of case-based reasoning to produce accurate
yet interpretable image classification models. These models generally classify
input images by comparing them with prototypes learned during training,
yielding explanations in the form of "this looks like that." However, methods
from this line of work use spatially rigid prototypes, which cannot explicitly
account for pose variations. In this paper, we address this shortcoming by
proposing a case-based interpretable neural network that provides spatially
flexible prototypes, called a deformable prototypical part network (Deformable
ProtoPNet). In a Deformable ProtoPNet, each prototype is made up of several
prototypical parts that adaptively change their relative spatial positions
depending on the input image. This enables each prototype to detect object
features with a higher tolerance to spatial transformations, as the parts
within a prototype are allowed to move. Consequently, a Deformable ProtoPNet
can explicitly capture pose variations, improving both model accuracy and the
richness of explanations provided. Compared to other case-based interpretable
models using prototypes, our approach achieves competitive accuracy, gives an
explanation with greater context, and is easier to train, thus enabling wider
use of interpretable models for computer vision.

    

### [[2111.15024] A Highly Configurable Hardware/Software Stack for DNN Inference Acceleration](http://arxiv.org/abs/2111.15024)


  This work focuses on an efficient Agile design methodology for
domain-specific accelerators. We employ feature-by-feature enhancement of a
vertical development stack and apply it to the TVM/VTA inference accelerator.
We have enhanced the VTA design space and enabled end-to-end support for
additional workloads. This has been accomplished by augmenting the VTA
micro-architecture and instruction set architecture (ISA), as well as by
enhancing the TVM compilation stack to support a wide range of VTA configs.
The VTA tsim implementation (CHISEL-based) has been enhanced with fully
pipelined versions of the ALU/GEMM execution units. In tsim, memory width can
now range between 8-64 bytes. Field widths have been made more flexible to
support larger scratchpads. New instructions have been added: element-wise
8-bit multiplication to support depthwise convolution, and load with a choice
of pad values to support max pooling. Support for more layers and better double
buffering has also been added.
Fully pipelining ALU/GEMM helps significantly: 4.9x fewer cycles with minimal
area change to run ResNet-18 under the default config. Configs featuring a
further 11.5x decrease in cycle count at a cost of 12x greater area can be
instantiated. Many points on the area-performance pareto curve are shown,
showcasing the balance of execution unit sizing, memory interface width, and
scratchpad sizing. Finally, VTA is now able to run Mobilenet 1.0 and all layers
for ResNets, including the previously disabled pooling and fully connected
layers.
The TVM/VTA architecture has always featured end-to-end workload evaluation
on RTL in minutes. With our modifications, it now offers a much greater number
of feasible configurations with a wide range of cost vs. performance. All
capabilities mentioned are available in opensource forks while a subset of
these capabilities have already been upstreamed.

    

### [[2111.15031] MOTIF: A Large Malware Reference Dataset with Ground Truth Family Labels](http://arxiv.org/abs/2111.15031)


  Malware family classification is a significant issue with public safety and
research implications that has been hindered by the high cost of expert labels.
The vast majority of corpora use noisy labeling approaches that obstruct
definitive quantification of results and study of deeper interactions. In order
to provide the data needed to advance further, we have created the Malware
Open-source Threat Intelligence Family (MOTIF) dataset. MOTIF contains 3,095
malware samples from 454 families, making it the largest and most diverse
public malware dataset with ground truth family labels to date, nearly 3x
larger than any prior expert-labeled corpus and 36x larger than the prior
Windows malware corpus. MOTIF also comes with a mapping from malware samples to
threat reports published by reputable industry sources, which both validates
the labels and opens new research opportunities in connecting opaque malware
samples to human-readable descriptions. This enables important evaluations that
are normally infeasible due to non-standardized reporting in industry. For
example, we provide aliases of the different names used to describe the same
malware family, allowing us to benchmark for the first time accuracy of
existing tools when names are obtained from differing sources. Evaluation
results obtained using the MOTIF dataset indicate that existing tasks have
significant room for improvement, with accuracy of antivirus majority voting
measured at only 62.10% and the well-known AVClass tool having just 46.78%
accuracy. Our findings indicate that malware family classification suffers a
type of labeling noise unlike that studied in most ML literature, due to the
large open set of classes that may not be known from the sample under
consideration

    

### [[2111.15037] CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data](http://arxiv.org/abs/2111.15037)


  Hyperbolic space can embed tree metric with little distortion, a desirable
property for modeling hierarchical structures of real-world data and semantics.
While high-dimensional embeddings often lead to better representations, most
hyperbolic models utilize low-dimensional embeddings, due to non-trivial
optimization as well as the lack of a visualization for high-dimensional
hyperbolic data.
We propose CO-SNE, extending the Euclidean space visualization tool, t-SNE,
to hyperbolic space. Like t-SNE, it converts distances between data points to
joint probabilities and tries to minimize the Kullback-Leibler divergence
between the joint probabilities of high-dimensional data $X$ and
low-dimensional embeddings $Y$. However, unlike Euclidean space, hyperbolic
space is inhomogeneous: a volume could contain a lot more points at a location
far from the origin. CO-SNE thus uses hyperbolic normal distributions for $X$
and hyberbolic \underline{C}auchy instead of t-SNE's Student's t-distribution
for $Y$, and it additionally attempts to preserve $X$'s individual distances to
the \underline{O}rigin in $Y$.
We apply CO-SNE to high-dimensional hyperbolic biological data as well as
unsupervisedly learned hyperbolic representations. Our results demonstrate that
CO-SNE deflates high-dimensional hyperbolic data into a low-dimensional space
without losing their hyperbolic characteristics, significantly outperforming
popular visualization tools such as PCA, t-SNE, UMAP, and HoroPCA, the last of
which is specifically designed for hyperbolic data.

    

### [[2111.15041] Online Learning for Receding Horizon Control with Provable Regret Guarantees](http://arxiv.org/abs/2111.15041)


  We address the problem of learning to control an unknown linear dynamical
system with time varying cost functions through the framework of online
Receding Horizon Control (RHC). We consider the setting where the control
algorithm does not know the true system model and has only access to a
fixed-length (that does not grow with the control horizon) preview of the
future cost functions. We characterize the performance of an algorithm using
the metric of dynamic regret, which is defined as the difference between the
cumulative cost incurred by the algorithm and that of the best sequence of
actions in hindsight. We propose two different online RHC algorithms to address
this problem, namely Certainty Equivalence RHC (CE-RHC) algorithm and
Optimistic RHC (O-RHC) algorithm. We show that under the standard stability
assumption for the model estimate, the CE-RHC algorithm achieves
$\mathcal{O}(T^{2/3})$ dynamic regret. We then extend this result to the
setting where the stability assumption hold only for the true system model by
proposing the O-RHC algorithm. We show that O-RHC algorithm achieves
$\mathcal{O}(T^{2/3})$ dynamic regret but with some additional computation.

    

### [[2111.15060] Second-order Approximation of Minimum Discrimination Information in Independent Component Analysis](http://arxiv.org/abs/2111.15060)


  Independent Component Analysis (ICA) is intended to recover the mutually
independent sources from their linear mixtures, and F astICA is one of the most
successful ICA algorithms. Although it seems reasonable to improve the
performance of F astICA by introducing more nonlinear functions to the
negentropy estimation, the original fixed-point method (approximate Newton
method) in F astICA degenerates under this circumstance. To alleviate this
problem, we propose a novel method based on the second-order approximation of
minimum discrimination information (MDI). The joint maximization in our method
is consisted of minimizing single weighted least squares and seeking unmixing
matrix by the fixed-point method. Experimental results validate its efficiency
compared with other popular ICA algorithms.

    

### [[2111.15072] Transition Motion Tensor: A Data-Driven Approach for Versatile and Controllable Agents in Physically Simulated Environments](http://arxiv.org/abs/2111.15072)


  This paper proposes the Transition Motion Tensor, a data-driven framework
that creates novel and physically accurate transitions outside of the motion
dataset. It enables simulated characters to adopt new motion skills efficiently
and robustly without modifying existing ones. Given several physically
simulated controllers specializing in different motions, the tensor serves as a
temporal guideline to transition between them. Through querying the tensor for
transitions that best fit user-defined preferences, we can create a unified
controller capable of producing novel transitions and solving complex tasks
that may require multiple motions to work coherently. We apply our framework on
both quadrupeds and bipeds, perform quantitative and qualitative evaluations on
transition quality, and demonstrate its capability of tackling complex motion
planning problems while following user control directives.

    

### [[2111.15080] SurvODE: Extrapolating Gene Expression Distribution for Early Cancer Identification](http://arxiv.org/abs/2111.15080)


  With the increasingly available large-scale cancer genomics datasets, machine
learning approaches have played an important role in revealing novel insights
into cancer development. Existing methods have shown encouraging performance in
identifying genes that are predictive for cancer survival, but are still
limited in modeling the distribution over genes. Here, we proposed a novel
method that can simulate the gene expression distribution at any given time
point, including those that are out of the range of the observed time points.
In order to model the irregular time series where each patient is one
observation, we integrated a neural ordinary differential equation (neural ODE)
with cox regression into our framework. We evaluated our method on eight cancer
types on TCGA and observed a substantial improvement over existing approaches.
Our visualization results and further analysis indicate how our method can be
used to simulate expression at the early cancer stage, offering the possibility
for early cancer identification.

    

### [[2111.15090] The Geometric Occam's Razor Implicit in Deep Learning](http://arxiv.org/abs/2111.15090)


  In over-parameterized deep neural networks there can be many possible
parameter configurations that fit the training data exactly. However, the
properties of these interpolating solutions are poorly understood. We argue
that over-parameterized neural networks trained with stochastic gradient
descent are subject to a Geometric Occam's Razor; that is, these networks are
implicitly regularized by the geometric model complexity. For one-dimensional
regression, the geometric model complexity is simply given by the arc length of
the function. For higher-dimensional settings, the geometric model complexity
depends on the Dirichlet energy of the function. We explore the relationship
between this Geometric Occam's Razor, the Dirichlet energy and other known
forms of implicit regularization. Finally, for ResNets trained on CIFAR-10, we
observe that Dirichlet energy measurements are consistent with the action of
this implicit Geometric Occam's Razor.

    

### [[2111.15097] EAGAN: Efficient Two-stage Evolutionary Architecture Search for GANs](http://arxiv.org/abs/2111.15097)


  Generative Adversarial Networks (GANs) have been proven hugely successful in
image generation tasks, but GAN training has the problem of instability. Many
works have improved the stability of GAN training by manually modifying the GAN
architecture, which requires human expertise and extensive trial-and-error.
Thus, neural architecture search (NAS), which aims to automate the model
design, has been applied to search GANs on the task of unconditional image
generation. The early NAS-GAN works only search generators for reducing the
difficulty. Some recent works have attempted to search both generator (G) and
discriminator (D) to improve GAN performance, but they still suffer from the
instability of GAN training during the search. To alleviate the instability
issue, we propose an efficient two-stage evolutionary algorithm (EA) based NAS
framework to discover GANs, dubbed \textbf{EAGAN}. Specifically, we decouple
the search of G and D into two stages and propose the weight-resetting strategy
to improve the stability of GAN training. Besides, we perform evolution
operations to produce the Pareto-front architectures based on multiple
objectives, resulting in a superior combination of G and D. By leveraging the
weight-sharing strategy and low-fidelity evaluation, EAGAN can significantly
shorten the search time. EAGAN achieves highly competitive results on the
CIFAR-10 (IS=8.81$\pm$0.10, FID=9.91) and surpasses previous NAS-searched GANs
on the STL-10 dataset (IS=10.44$\pm$0.087, FID=22.18).

    

### [[2111.15099] Trust the Critics: Generatorless and Multipurpose WGANs with Initial Convergence Guarantees](http://arxiv.org/abs/2111.15099)


  Inspired by ideas from optimal transport theory we present Trust the Critics
(TTC), a new algorithm for generative modelling. This algorithm eliminates the
trainable generator from a Wasserstein GAN; instead, it iteratively modifies
the source data using gradient descent on a sequence of trained critic
networks. This is motivated in part by the misalignment which we observed
between the optimal transport directions provided by the gradients of the
critic and the directions in which data points actually move when parametrized
by a trainable generator. Previous work has arrived at similar ideas from
different viewpoints, but our basis in optimal transport theory motivates the
choice of an adaptive step size which greatly accelerates convergence compared
to a constant step size. Using this step size rule, we prove an initial
geometric convergence rate in the case of source distributions with densities.
These convergence rates cease to apply only when a non-negligible set of
generated data is essentially indistinguishable from real data. Resolving the
misalignment issue improves performance, which we demonstrate in experiments
that show that given a fixed number of training epochs, TTC produces higher
quality images than a comparable WGAN, albeit at increased memory requirements.
In addition, TTC provides an iterative formula for the transformed density,
which traditional WGANs do not. Finally, TTC can be applied to map any source
distribution onto any target; we demonstrate through experiments that TTC can
obtain competitive performance in image generation, translation, and denoising
without dedicated algorithms.

    

### [[2111.15101] A novel data-driven algorithm to predict anomalous prescription based on patient's feature set](http://arxiv.org/abs/2111.15101)


  Appropriate dosing of radiation is crucial to patient safety in radiotherapy.
Current quality assurance depends heavily on a peer-review process, where the
physicians' peer review on each patient's treatment plan, including dose and
fractionation. However, such a process is manual and laborious. Physicians may
not identify errors due to time constraints and caseload. We designed a novel
prescription anomaly detection algorithm that utilizes historical data to
predict anomalous cases. Such a tool can serve as an electronic peer who will
assist the peer-review process providing extra safety to the patients. In our
primary model, we created two dissimilarity metrics, R and F. R defining how
far a new patient's prescription is from historical prescriptions. F represents
how far away a patient's feature set is from the group with an identical or
similar prescription. We flag prescription if either metric is greater than
specific optimized cut-off values. We used thoracic cancer patients (n=2356) as
an example and extracted seven features. Here, we report our testing f1 score,
between 75%-94% for different treatment technique groups. We also independently
validate our results by conducting a mock peer review with three thoracic
specialists. Our model has a lower type 2 error rate compared to manual
peer-review physicians. Our model has many advantages over traditional machine
learning algorithms, particularly in that it does not suffer from class
imbalance. It can also explain why it flags each case and separate prescription
and non-prescription-related features without learning from the data.

    

### [[2111.15106] MAPLE: Microprocessor A Priori for Latency Estimation](http://arxiv.org/abs/2111.15106)


  Modern deep neural networks must demonstrate state-of-the-art accuracy while
exhibiting low latency and energy consumption. As such, neural architecture
search (NAS) algorithms take these two constraints into account when generating
a new architecture. However, efficiency metrics such as latency are typically
hardware dependent requiring the NAS algorithm to either measure or predict the
architecture latency. Measuring the latency of every evaluated architecture
adds a significant amount of time to the NAS process. Here we propose
Microprocessor A Priori for Latency Estimation MAPLE that does not rely on
transfer learning or domain adaptation but instead generalizes to new hardware
by incorporating a prior hardware characteristics during training. MAPLE takes
advantage of a novel quantitative strategy to characterize the underlying
microprocessor by measuring relevant hardware performance metrics, yielding a
fine-grained and expressive hardware descriptor. Moreover, the proposed MAPLE
benefits from the tightly coupled I/O between the CPU and GPU and their
dependency to predict DNN latency on GPUs while measuring microprocessor
performance hardware counters from the CPU feeding the GPU hardware. Through
this quantitative strategy as the hardware descriptor, MAPLE can generalize to
new hardware via a few shot adaptation strategy where with as few as 3 samples
it exhibits a 3% improvement over state-of-the-art methods requiring as much as
10 samples. Experimental results showed that, increasing the few shot
adaptation samples to 10 improves the accuracy significantly over the
state-of-the-art methods by 12%. Furthermore, it was demonstrated that MAPLE
exhibiting 8-10% better accuracy, on average, compared to relevant baselines at
any number of adaptation samples.

    

### [[2111.15112] AugLiChem: Data Augmentation Library ofChemical Structures for Machine Learning](http://arxiv.org/abs/2111.15112)


  Machine learning (ML) has demonstrated the promise for accurate andefficient
property prediction of molecules and crystalline materials. Todevelop highly
accurate ML models for chemical structure property pre-diction, datasets with
sufficient samples are required. However, obtainingclean and sufficient data of
chemical properties can be expensive andtime-consuming, which greatly limits
the performance of ML models.Inspired by the success of data augmentations in
computer vision andnatural language processing, we developed AugLiChem: the
data aug-mentation library for chemical structures. Augmentation methods
forboth crystalline systems and molecules are introduced, which can beutilized
for fingerprint-based ML models and Graph Neural Networks(GNNs). We show that
using our augmentation strategies significantlyimproves the performance of ML
models, especially when using this http URL addition, the augmentations that we
developed can be used as adirect plug-in module during training and have
demonstrated the effec-tiveness when implemented with different GNN models
through theAugliChem library. The Python-based package for our implementa-tion
of Auglichem: Data augmentation library for chemical structures,is publicly
available at: this https URL


### [[2111.15114] ePose: Let's Make EfficientPose More Generally Applicable](http://arxiv.org/abs/2111.15114)


  EfficientPose is an impressive 3D object detection model. It has been
demonstrated to be quick, scalable, and accurate, especially when considering
that it uses only RGB inputs. In this paper we try to improve on EfficientPose
by giving it the ability to infer an object's size, and by simplifying both the
data collection and loss calculations. We evaluated ePose using the Linemod
dataset and a new subset of it called "Occlusion 1-class". We also outline our
current progress and thoughts about using ePose with the NuScenes and the 2017
KITTI 3D Object Detection datasets. The source code is available at
this https URL.

    

### [[2111.15124] In-Bed Human Pose Estimation from Unseen and Privacy-Preserving Image Domains](http://arxiv.org/abs/2111.15124)


  Medical applications have benefited from the rapid advancement in computer
vision. For patient monitoring in particular, in-bed human posture estimation
provides important health-related metrics with potential value in medical
condition assessments. Despite great progress in this domain, it remains a
challenging task due to substantial ambiguity during occlusions, and the lack
of large corpora of manually labeled data for model training, particularly with
domains such as thermal infrared imaging which are privacy-preserving, and thus
of great interest. Motivated by the effectiveness of self-supervised methods in
learning features directly from data, we propose a multi-modal conditional
variational autoencoder (MC-VAE) capable of reconstructing features from
missing modalities seen during training. This approach is used with HRNet to
enable single modality inference for in-bed pose estimation. Through extensive
evaluations, we demonstrate that body positions can be effectively recognized
from the available modality, achieving on par results with baseline models that
are highly dependent on having access to multiple modes at inference time. The
proposed framework supports future research towards self-supervised learning
that generates a robust model from a single source, and expects it to
generalize over many unknown distributions in clinical environments.

    

### [[2111.15129] Anonymization for Skeleton Action Recognition](http://arxiv.org/abs/2111.15129)


  The skeleton-based action recognition attracts practitioners and researchers
due to the lightweight, compact nature of datasets. Compared with
RGB-video-based action recognition, skeleton-based action recognition is a
safer way to protect the privacy of subjects while having competitive
recognition performance. However, due to the improvements of skeleton
estimation algorithms as well as motion- and depth-sensors, more details of
motion characteristics can be preserved in the skeleton dataset, leading to a
potential privacy leakage from the dataset. To investigate the potential
privacy leakage from the skeleton datasets, we first train a classifier to
categorize sensitive private information from a trajectory of joints.
Experiments show the model trained to classify gender can predict with 88%
accuracy and re-identify a person with 82% accuracy. We propose two variants of
anonymization algorithms to protect the potential privacy leakage from the
skeleton dataset. Experimental results show that the anonymized dataset can
reduce the risk of privacy leakage while having marginal effects on the action
recognition performance.

    

### [[2111.15133] LossPlot: A Better Way to Visualize Loss Landscapes](http://arxiv.org/abs/2111.15133)


  Investigations into the loss landscapes of deep neural networks are often
laborious. This work documents our user-driven approach to create a platform
for semi-automating this process. LossPlot accepts data in the form of a csv,
and allows multiple trained minimizers of the loss function to be manipulated
in sync. Other features include a simple yet intuitive checkbox UI, summary
statistics, and the ability to control clipping which other methods do not
offer.

    

### [[2111.15135] Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs](http://arxiv.org/abs/2111.15135)


  Coordinate-MLPs are emerging as an effective tool for modeling
multidimensional continuous signals, overcoming many drawbacks associated with
discrete grid-based approximations. However, coordinate-MLPs with ReLU
activations, in their rudimentary form, demonstrate poor performance in
representing signals with high fidelity, promoting the need for positional
embedding layers. Recently, Sitzmann et al. proposed a sinusoidal activation
function that has the capacity to omit positional embedding from
coordinate-MLPs while still preserving high signal fidelity. Despite its
potential, ReLUs are still dominating the space of coordinate-MLPs; we
speculate that this is due to the hyper-sensitivity of networks -- that employ
such sinusoidal activations -- to the initialization schemes. In this paper, we
attempt to broaden the current understanding of the effect of activations in
coordinate-MLPs, and show that there exists a broader class of activations that
are suitable for encoding signals. We affirm that sinusoidal activations are
only a single example in this class, and propose several non-periodic functions
that empirically demonstrate more robust performance against random
initializations than sinusoids. Finally, we advocate for a shift towards
coordinate-MLPs that employ these non-traditional activation functions due to
their high performance and simplicity.

    

### [[2111.15139] Breaking the Linear Iteration Cost Barrier for Some Well-known Conditional Gradient Methods Using MaxIP Data-structures](http://arxiv.org/abs/2111.15139)


  Conditional gradient methods (CGM) are widely used in modern machine
learning. CGM's overall running time usually consists of two parts: the number
of iterations and the cost of each iteration. Most efforts focus on reducing
the number of iterations as a means to reduce the overall running time. In this
work, we focus on improving the per iteration cost of CGM. The bottleneck step
in most CGM is maximum inner product search (MaxIP), which requires a linear
scan over the parameters. In practice, approximate MaxIP data-structures are
found to be helpful heuristics. However, theoretically, nothing is known about
the combination of approximate MaxIP data-structures and CGM. In this work, we
answer this question positively by providing a formal framework to combine the
locality sensitive hashing type approximate MaxIP data-structures with CGM
algorithms. As a result, we show the first algorithm, where the cost per
iteration is sublinear in the number of parameters, for many fundamental
optimization algorithms, e.g., Frank-Wolfe, Herding algorithm, and policy
gradient.

    

### [[2111.15141] Path Integral Sampler: a stochastic control approach for sampling](http://arxiv.org/abs/2111.15141)


  We present Path Integral Sampler~(PIS), a novel algorithm to draw samples
from unnormalized probability density functions. The PIS is built on the
Schrödinger bridge problem which aims to recover the most likely evolution of
a diffusion process given its initial distribution and terminal distribution.
The PIS draws samples from the initial distribution and then propagates the
samples through the Schrödinger bridge to reach the terminal distribution.
Applying the Girsanov theorem, with a simple prior diffusion, we formulate the
PIS as a stochastic optimal control problem whose running cost is the control
energy and terminal cost is chosen according to the target distribution. By
modeling the control as a neural network, we establish a sampling algorithm
that can be trained end-to-end. We provide theoretical justification of the
sampling quality of PIS in terms of Wasserstein distance when sub-optimal
control is used. Moreover, the path integrals theory is used to compute
importance weights of the samples to compensate for the bias induced by the
sub-optimality of the controller and time-discretization. We experimentally
demonstrate the advantages of PIS compared with other start-of-the-art sampling
methods on a variety of tasks.

    

### [[2111.15144] Decoding the Protein-ligand Interactions Using Parallel Graph Neural Networks](http://arxiv.org/abs/2111.15144)


  Protein-ligand interactions (PLIs) are fundamental to biochemical research
and their identification is crucial for estimating biophysical and biochemical
properties for rational therapeutic design. Currently, experimental
characterization of these properties is the most accurate method, however, this
is very time-consuming and labor-intensive. A number of computational methods
have been developed in this context but most of the existing PLI prediction
heavily depends on 2D protein sequence data. Here, we present a novel parallel
graph neural network (GNN) to integrate knowledge representation and reasoning
for PLI prediction to perform deep learning guided by expert knowledge and
informed by 3D structural data. We develop two distinct GNN architectures, GNNF
is the base implementation that employs distinct featurization to enhance
domain-awareness, while GNNP is a novel implementation that can predict with no
prior knowledge of the intermolecular interactions. The comprehensive
evaluation demonstrated that GNN can successfully capture the binary
interactions between ligand and proteins 3D structure with 0.979 test accuracy
for GNNF and 0.958 for GNNP for predicting activity of a protein-ligand
complex. These models are further adapted for regression tasks to predict
experimental binding affinities and pIC50 is crucial for drugs potency and
efficacy. We achieve a Pearson correlation coefficient of 0.66 and 0.65 on
experimental affinity and 0.50 and 0.51 on pIC50 with GNNF and GNNP,
respectively, outperforming similar 2D sequence-based models. Our method can
serve as an interpretable and explainable artificial intelligence (AI) tool for
predicted activity, potency, and biophysical properties of lead candidates. To
this end, we show the utility of GNNP on SARS-Cov-2 protein targets by
screening a large compound library and comparing our prediction with the
experimentally measured data.

    

### [[2111.15146] Molecular Attributes Transfer from Non-Parallel Data](http://arxiv.org/abs/2111.15146)


  Optimizing chemical molecules for desired properties lies at the core of drug
development. Despite initial successes made by deep generative models and
reinforcement learning methods, these methods were mostly limited by the
requirement of predefined attribute functions or parallel data with manually
pre-compiled pairs of original and optimized molecules. In this paper, for the
first time, we formulate molecular optimization as a style transfer problem and
present a novel generative model that could automatically learn internal
differences between two groups of non-parallel data through adversarial
training strategies. Our model further enables both preservation of molecular
contents and optimization of molecular properties through combining auxiliary
guided-variational autoencoders and generative flow techniques. Experiments on
two molecular optimization tasks, toxicity modification and synthesizability
improvement, demonstrate that our model significantly outperforms several
state-of-the-art methods.

    

### [[2111.15155] gCastle: A Python Toolbox for Causal Discovery](http://arxiv.org/abs/2111.15155)


  $\texttt{gCastle}$ is an end-to-end Python toolbox for causal structure
learning. It provides functionalities of generating data from either simulator
or real-world dataset, learning causal structure from the data, and evaluating
the learned graph, together with useful practices such as prior knowledge
insertion, preliminary neighborhood selection, and post-processing to remove
false discoveries. Compared with related packages, $\texttt{gCastle}$ includes
many recently developed gradient-based causal discovery methods with optional
GPU acceleration. $\texttt{gCastle}$ brings convenience to researchers who may
directly experiment with the code as well as practitioners with graphical user
interference. Three real-world datasets in telecommunications are also provided
in the current version. $\texttt{gCastle}$ is available under Apache License
2.0 at \url{this https URL}.

    

### [[2111.15159] CycleTransGAN-EVC: A CycleGAN-based Emotional Voice Conversion Model with Transformer](http://arxiv.org/abs/2111.15159)


  In this study, we explore the transformer's ability to capture
intra-relations among frames by augmenting the receptive field of models.
Concretely, we propose a CycleGAN-based model with the transformer and
investigate its ability in the emotional voice conversion task. In the training
procedure, we adopt curriculum learning to gradually increase the frame length
so that the model can see from the short segment till the entire speech. The
proposed method was evaluated on the Japanese emotional speech dataset and
compared to several baselines (ACVAE, CycleGAN) with objective and subjective
evaluations. The results show that our proposed model is able to convert
emotion with higher strength and quality.

    

### [[2111.15160] Mitigating Adversarial Attacks by Distributing Different Copies to Different Users](http://arxiv.org/abs/2111.15160)


  Machine learning models are vulnerable to adversarial attacks. In this paper,
we consider the scenario where a model is to be distributed to multiple users,
among which a malicious user attempts to attack another user. The malicious
user probes its copy of the model to search for adversarial samples and then
presents the found samples to the victim's model in order to replicate the
attack. We point out that by distributing different copies of the model to
different users, we can mitigate the attack such that adversarial samples found
on one copy would not work on another copy. We first observed that training a
model with different randomness indeed mitigates such replication to certain
degree. However, there is no guarantee and retraining is computationally
expensive. Next, we propose a flexible parameter rewriting method that directly
modifies the model's parameters. This method does not require additional
training and is able to induce different sets of adversarial samples in
different copies in a more controllable manner. Experimentation studies show
that our approach can significantly mitigate the attacks while retaining high
classification accuracy. From this study, we believe that there are many
further directions worth exploring.

    

### [[2111.15176] Learning Large-Time-Step Molecular Dynamics with Graph Neural Networks](http://arxiv.org/abs/2111.15176)


  Molecular dynamics (MD) simulation predicts the trajectory of atoms by
solving Newton's equation of motion with a numeric integrator. Due to physical
constraints, the time step of the integrator need to be small to maintain
sufficient precision. This limits the efficiency of simulation. To this end, we
introduce a graph neural network (GNN) based model, MDNet, to predict the
evolution of coordinates and momentum with large time steps. In addition, MDNet
can easily scale to a larger system, due to its linear complexity with respect
to the system size. We demonstrate the performance of MDNet on a 4000-atom
system with large time steps, and show that MDNet can predict good equilibrium
and transport properties, well aligned with standard MD simulations.

    

### [[2111.15178] Sparse deep computer-generated holography for optical microscopy](http://arxiv.org/abs/2111.15178)


  Computer-generated holography (CGH) has broad applications such as
direct-view display, virtual and augmented reality, as well as optical
microscopy. CGH usually utilizes a spatial light modulator that displays a
computer-generated phase mask, modulating the phase of coherent light in order
to generate customized patterns. The algorithm that computes the phase mask is
the core of CGH and is usually tailored to meet different applications. CGH for
optical microscopy usually requires 3D accessibility (i.e., generating
overlapping patterns along the $z$-axis) and micron-scale spatial precision.
Here, we propose a CGH algorithm using an unsupervised generative model
designed for optical microscopy to synthesize 3D selected illumination. The
algorithm, named sparse deep CGH, is able to generate sparsely distributed
points in a large 3D volume with higher contrast than conventional CGH
algorithms.

    

### [[2111.15179] A Highly Effective Low-Rank Compression of Deep Neural Networks with Modified Beam-Search and Modified Stable Rank](http://arxiv.org/abs/2111.15179)


  Compression has emerged as one of the essential deep learning research
topics, especially for the edge devices that have limited computation power and
storage capacity. Among the main compression techniques, low-rank compression
via matrix factorization has been known to have two problems. First, an
extensive tuning is required. Second, the resulting compression performance is
typically not impressive. In this work, we propose a low-rank compression
method that utilizes a modified beam-search for an automatic rank selection and
a modified stable rank for a compression-friendly training. The resulting BSR
(Beam-search and Stable Rank) algorithm requires only a single hyperparameter
to be tuned for the desired compression ratio. The performance of BSR in terms
of accuracy and compression ratio trade-off curve turns out to be superior to
the previously known low-rank compression methods. Furthermore, BSR can perform
on par with or better than the state-of-the-art structured pruning methods. As
with pruning, BSR can be easily combined with quantization for an additional
compression.

    

### [[2111.15182] Easy Semantification of Bioassays](http://arxiv.org/abs/2111.15182)


  Biological data and knowledge bases increasingly rely on Semantic Web
technologies and the use of knowledge graphs for data integration, retrieval
and federated queries. We propose a solution for automatically semantifying
biological assays. Our solution juxtaposes the problem of automated
semantification as classification versus clustering where the two methods are
on opposite ends of the method complexity spectrum. Characteristically modeling
our problem, we find the clustering solution significantly outperforms a deep
neural network state-of-the-art classification approach. This novel
contribution is based on two factors: 1) a learning objective closely modeled
after the data outperforms an alternative approach with sophisticated semantic
modeling; 2) automatically semantifying biological assays achieves a high
performance F1 of nearly 83%, which to our knowledge is the first reported
standardized evaluation of the task offering a strong benchmark model.

    

### [[2111.15186] Automatic Synthesis of Diverse Weak Supervision Sources for Behavior Analysis](http://arxiv.org/abs/2111.15186)


  Obtaining annotations for large training sets is expensive, especially in
behavior analysis settings where domain knowledge is required for accurate
annotations. Weak supervision has been studied to reduce annotation costs by
using weak labels from task-level labeling functions to augment ground truth
labels. However, domain experts are still needed to hand-craft labeling
functions for every studied task. To reduce expert effort, we present AutoSWAP:
a framework for automatically synthesizing data-efficient task-level labeling
functions. The key to our approach is to efficiently represent expert knowledge
in a reusable domain specific language and domain-level labeling functions,
with which we use state-of-the-art program synthesis techniques and a small
labeled dataset to generate labeling functions. Additionally, we propose a
novel structural diversity cost that allows for direct synthesis of diverse
sets of labeling functions with minimal overhead, further improving labeling
function data efficiency. We evaluate AutoSWAP in three behavior analysis
domains and demonstrate that AutoSWAP outperforms existing approaches using
only a fraction of the data. Our results suggest that AutoSWAP is an effective
way to automatically generate labeling functions that can significantly reduce
expert effort for behavior analysis.

    

### [[2111.15187] HyperPCA: a Powerful Tool to Extract Elemental Maps from Noisy Data Obtained in LIBS Mapping of Materials](http://arxiv.org/abs/2111.15187)


  Laser-induced breakdown spectroscopy is a preferred technique for fast and
direct multi-elemental mapping of samples under ambient pressure, without any
limitation on the targeted element. However, LIBS mapping data have two
peculiarities: an intrinsically low signal-to-noise ratio due to single-shot
measurements, and a high dimensionality due to the high number of spectra
acquired for imaging. This is all the truer as lateral resolution gets higher:
in this case, the ablation spot diameter is reduced, as well as the ablated
mass and the emission signal, while the number of spectra for a given surface
increases. Therefore, efficient extraction of physico-chemical information from
a noisy and large dataset is a major issue. Multivariate approaches were
introduced by several authors as a means to cope with such data, particularly
Principal Component Analysis. Yet, PCA is known to present theoretical
constraints for the consistent reconstruction of the dataset, and has therefore
limitations to efficient interpretation of LIBS mapping data. In this paper, we
introduce HyperPCA, a new analysis tool for hyperspectral images based on a
sparse representation of the data using Discrete Wavelet Transform and
kernel-based sparse PCA to reduce the impact of noise on the data and to
consistently reconstruct the spectroscopic signal, with a particular emphasis
on LIBS data. The method is first illustrated using simulated LIBS mapping
datasets to emphasize its performances with highly noisy and/or highly
interfered spectra. Comparisons to standard PCA and to traditional univariate
data analyses are provided. Finally, it is used to process real data in two
cases that clearly illustrate the potential of the proposed algorithm. We show
that the method presents advantages both in quantity and quality of the
information recovered, thus improving the physico-chemical characterisation of
analysed surfaces.

    

### [[2111.15196] PGNets: Planet mass prediction using convolutional neural networks for radio continuum observations of protoplanetary disks](http://arxiv.org/abs/2111.15196)


  We developed Convolutional Neural Networks (CNNs) to rapidly and directly
infer the planet mass from radio dust continuum images. Substructures induced
by young planets in protoplanetary disks can be used to infer the potential
young planets' properties. Hydrodynamical simulations have been used to study
the relationships between the planet's properties and these disk features.
However, these attempts either fine-tuned numerical simulations to fit one
protoplanetary disk at a time, which was time-consuming, or azimuthally
averaged simulation results to derive some linear relationships between the gap
width/depth and the planet mass, which lost information on asymmetric features
in disks. To cope with these disadvantages, we developed Planet Gap neural
Networks (PGNets) to infer the planet mass from 2D images. We first fit the
gridded data in Zhang et al. (2018) as a classification problem. Then, we
quadrupled the data set by running additional simulations with near-randomly
sampled parameters, and derived the planet mass and disk viscosity together as
a regression problem. The classification approach can reach an accuracy of
92\%, whereas the regression approach can reach 1$\sigma$ as 0.16 dex for
planet mass and 0.23 dex for disk viscosity. We can reproduce the degeneracy
scaling $\alpha$ $\propto$ $M_p^3$ found in the linear fitting method, which
means that the CNN method can even be used to find degeneracy relationship. The
gradient-weighted class activation mapping effectively confirms that PGNets use
proper disk features to constrain the planet mass. We provide programs for
PGNets and the traditional fitting method from Zhang et al. (2018), and discuss
each method's advantages and disadvantages.

    

### [[2111.15205] New Datasets for Dynamic Malware Classification](http://arxiv.org/abs/2111.15205)


  Nowadays, malware and malware incidents are increasing daily, even with
various anti-viruses systems and malware detection or classification
methodologies. Many static, dynamic, and hybrid techniques have been presented
to detect malware and classify them into malware families. Dynamic and hybrid
malware classification methods have advantages over static malware
classification methods by being highly efficient. Since it is difficult to mask
malware behavior while executing than its underlying code in static malware
classification, machine learning techniques have been the main focus of the
security experts to detect malware and determine their families dynamically.
The rapid increase of malware also brings the necessity of recent and updated
datasets of malicious software. We introduce two new, updated datasets in this
work: One with 9,795 samples obtained and compiled from VirusSamples and the
one with 14,616 samples from VirusShare. This paper also analyzes multi-class
malware classification performance of the balanced and imbalanced version of
these two datasets by using Histogram-based gradient boosting, Random Forest,
Support Vector Machine, and XGBoost models with API call-based dynamic malware
classification. Results show that Support Vector Machine, achieves the highest
score of 94% in the imbalanced VirusSample dataset, whereas the same model has
91% accuracy in the balanced VirusSample dataset. While XGBoost, one of the
most common gradient boosting-based models, achieves the highest score of 90%
and 80%.in both versions of the VirusShare dataset. This paper also presents
the baseline results of VirusShare and VirusSample datasets by using the four
most widely known machine learning techniques in dynamic malware classification
literature. We believe that these two datasets and baseline results enable
researchers in this field to test and validate their methods and approaches.

    

### [[2111.15207] NeeDrop: Self-supervised Shape Representation from Sparse Point Clouds using Needle Dropping](http://arxiv.org/abs/2111.15207)


  There has been recently a growing interest for implicit shape
representations. Contrary to explicit representations, they have no resolution
limitations and they easily deal with a wide variety of surface topologies. To
learn these implicit representations, current approaches rely on a certain
level of shape supervision (e.g., inside/outside information or
distance-to-shape knowledge), or at least require a dense point cloud (to
approximate well enough the distance-to-shape). In contrast, we introduce
{\method}, an self-supervised method for learning shape representations from
possibly extremely sparse point clouds. Like in Buffon's needle problem, we
"drop" (sample) needles on the point cloud and consider that, statistically,
close to the surface, the needle end points lie on opposite sides of the
surface. No shape knowledge is required and the point cloud can be highly
sparse, e.g., as lidar point clouds acquired by vehicles. Previous
self-supervised shape representation approaches fail to produce good-quality
results on this kind of data. We obtain quantitative results on par with
existing supervised approaches on shape reconstruction datasets and show
promising qualitative results on hard autonomous driving datasets such as
KITTI.

    

### [[2111.15228] Global Convergence Using Policy Gradient Methods for Model-free Markovian Jump Linear Quadratic Control](http://arxiv.org/abs/2111.15228)


  Owing to the growth of interest in Reinforcement Learning in the last few
years, gradient based policy control methods have been gaining popularity for
Control problems as well. And rightly so, since gradient policy methods have
the advantage of optimizing a metric of interest in an end-to-end manner, along
with being relatively easy to implement without complete knowledge of the
underlying system. In this paper, we study the global convergence of
gradient-based policy optimization methods for quadratic control of
discrete-time and model-free Markovian jump linear systems (MJLS). We surmount
myriad challenges that arise because of more than one states coupled with lack
of knowledge of the system dynamics and show global convergence of the policy
using gradient descent and natural policy gradient methods. We also provide
simulation studies to corroborate our claims.

    

### [[2111.15242] ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regularized Domain Concatenation](http://arxiv.org/abs/2111.15242)


  Transferring knowledge learned from the labeled source domain to the raw
target domain for unsupervised domain adaptation (UDA) is essential to the
scalable deployment of an autonomous driving system. State-of-the-art
approaches in UDA often employ a key concept: utilize joint supervision signals
from both the source domain (with ground-truth) and the target domain (with
pseudo-labels) for self-training. In this work, we improve and extend on this
aspect. We present ConDA, a concatenation-based domain adaptation framework for
LiDAR semantic segmentation that: (1) constructs an intermediate domain
consisting of fine-grained interchange signals from both source and target
domains without destabilizing the semantic coherency of objects and background
around the ego-vehicle; and (2) utilizes the intermediate domain for
self-training. Additionally, to improve both the network training on the source
domain and self-training on the intermediate domain, we propose an
anti-aliasing regularizer and an entropy aggregator to reduce the detrimental
effects of aliasing artifacts and noisy target predictions. Through extensive
experiments, we demonstrate that ConDA is significantly more effective in
mitigating the domain gap compared to prior arts.

    

### [[2111.15256] Approximate Spectral Decomposition of Fisher Information Matrix for Simple ReLU Networks](http://arxiv.org/abs/2111.15256)


  We investigate the Fisher information matrix (FIM) of one hidden layer
networks with the ReLU activation function and obtain an approximate spectral
decomposition of FIM under certain conditions. From this decomposition, we can
approximate the main eigenvalues and eigenvectors. We confirmed by numerical
simulation that the obtained decomposition is approximately correct when the
number of hidden nodes is about 10000.

    

### [[2111.15258] DeepAL: Deep Active Learning in Python](http://arxiv.org/abs/2111.15258)


  We present DeepAL, a Python library that implements several common strategies
for active learning, with a particular emphasis on deep active learning. DeepAL
provides a simple and unified framework based on PyTorch that allows users to
easily load custom datasets, build custom data handlers, and design custom
strategies without much modification of codes. DeepAL is open-source on Github
and welcome any contribution.

    

### [[2111.15264] EdiBERT, a generative model for image editing](http://arxiv.org/abs/2111.15264)


  Advances in computer vision are pushing the limits of im-age manipulation,
with generative models sampling detailed images on various tasks. However, a
specialized model is often developed and trained for each specific task, even
though many image edition tasks share similarities. In denoising, inpainting,
or image compositing, one always aims at generating a realistic image from a
low-quality one. In this paper, we aim at making a step towards a unified
approach for image editing. To do so, we propose EdiBERT, a bi-directional
transformer trained in the discrete latent space built by a vector-quantized
auto-encoder. We argue that such a bidirectional model is suited for image
manipulation since any patch can be re-sampled conditionally to the whole
image. Using this unique and straightforward training objective, we show that
the resulting model matches state-of-the-art performances on a wide variety of
tasks: image denoising, image completion, and image composition.

    

### [[2111.15298] Generating Rich Product Descriptions for Conversational E-commerce Systems](http://arxiv.org/abs/2111.15298)


  Through recent advancements in speech technologies and introduction of smart
assistants, such as Amazon Alexa, Apple Siri and Google Home, increasing number
of users are interacting with various applications through voice commands.
E-commerce companies typically display short product titles on their webpages,
either human-curated or algorithmically generated, when brevity is required.
However, these titles are dissimilar from natural spoken language. For example,
"Lucky Charms Gluten Free Break-fast Cereal, 20.5 oz a box Lucky Charms Gluten
Free" is acceptable to display on a webpage, while a similar title cannot be
used in a voice based text-to-speech application. In such conversational
systems, an easy to comprehend sentence, such as "a 20.5 ounce box of lucky
charms gluten free cereal" is preferred. Compared to display devices, where
images and detailed product information can be presented to users, short titles
for products which convey the most important information, are necessary when
interfacing with voice assistants. We propose eBERT, a sequence-to-sequence
approach by further pre-training the BERT embeddings on an e-commerce product
description corpus, and then fine-tuning the resulting model to generate short,
natural, spoken language titles from input web titles. Our extensive
experiments on a real-world industry dataset, as well as human evaluation of
model output, demonstrate that eBERT summarization outperforms comparable
baseline models. Owing to the efficacy of the model, a version of this model
has been deployed in real-world setting.

    

### [[2111.15309] Deep Auto-encoder with Neural Response](http://arxiv.org/abs/2111.15309)


  Artificial intelligence and neuroscience are deeply interactive. Artificial
neural networks (ANNs) have been a versatile tool to study the neural
representation in the ventral visual stream, and the knowledge in neuroscience
in return inspires ANN models to improve performance in the task. However, how
to merge these two directions into a unified model has less studied. Here, we
propose a hybrid model, called deep auto-encoder with the neural response
(DAE-NR), which incorporates the information from the visual cortex into ANNs
to achieve better image reconstruction and higher neural representation
similarity between biological and artificial neurons. Specifically, the same
visual stimuli (i.e., natural images) are input to both the mice brain and
DAE-NR. The DAE-NR jointly learns to map a specific layer of the encoder
network to the biological neural responses in the ventral visual stream by a
mapping function and to reconstruct the visual input by the decoder. Our
experiments demonstrate that if and only if with the joint learning, DAE-NRs
can (i) improve the performance of image reconstruction and (ii) increase the
representational similarity between biological neurons and artificial neurons.
The DAE-NR offers a new perspective on the integration of computer vision and
visual neuroscience.

    

### [[2111.15317] AutoDrop: Training Deep Learning Models with Automatic Learning Rate Drop](http://arxiv.org/abs/2111.15317)


  Modern deep learning (DL) architectures are trained using variants of the SGD
algorithm that is run with a $\textit{manually}$ defined learning rate
schedule, i.e., the learning rate is dropped at the pre-defined epochs,
typically when the training loss is expected to saturate. In this paper we
develop an algorithm that realizes the learning rate drop
$\textit{automatically}$. The proposed method, that we refer to as AutoDrop, is
motivated by the observation that the angular velocity of the model parameters,
i.e., the velocity of the changes of the convergence direction, for a fixed
learning rate initially increases rapidly and then progresses towards soft
saturation. At saturation the optimizer slows down thus the angular velocity
saturation is a good indicator for dropping the learning rate. After the drop,
the angular velocity "resets" and follows the previously described pattern - it
increases again until saturation. We show that our method improves over SOTA
training approaches: it accelerates the training of DL models and leads to a
better generalization. We also show that our method does not require any extra
hyperparameter tuning. AutoDrop is furthermore extremely simple to implement
and computationally cheap. Finally, we develop a theoretical framework for
analyzing our algorithm and provide convergence guarantees.

    

### [[2111.15318] DiffSDFSim: Differentiable Rigid-Body Dynamics With Implicit Shapes](http://arxiv.org/abs/2111.15318)


  Differentiable physics is a powerful tool in computer vision and robotics for
scene understanding and reasoning about interactions. Existing approaches have
frequently been limited to objects with simple shape or shapes that are known
in advance. In this paper, we propose a novel approach to differentiable
physics with frictional contacts which represents object shapes implicitly
using signed distance fields (SDFs). Our simulation supports contact point
calculation even when the involved shapes are nonconvex. Moreover, we propose
ways for differentiating the dynamics for the object shape to facilitate shape
optimization using gradient-based methods. In our experiments, we demonstrate
that our approach allows for model-based inference of physical parameters such
as friction coefficients, mass, forces or shape parameters from trajectory and
depth image observations in several challenging synthetic scenarios and a real
image sequence.

    

### [[2111.15340] MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning](http://arxiv.org/abs/2111.15340)


  Self-supervised pretraining is the method of choice for natural language
processing models and is rapidly gaining popularity in many vision tasks.
Recently, self-supervised pretraining has shown to outperform supervised
pretraining for many downstream vision applications, marking a milestone in the
area. This superiority is attributed to the negative impact of incomplete
labelling of the training images, which convey multiple concepts, but are
annotated using a single dominant class label. Although Self-Supervised
Learning (SSL), in principle, is free of this limitation, the choice of pretext
task facilitating SSL is perpetuating this shortcoming by driving the learning
process towards a single concept output. This study aims to investigate the
possibility of modelling all the concepts present in an image without using
labels. In this aspect the proposed SSL frame-work MC-SSL0.0 is a step towards
Multi-Concept Self-Supervised Learning (MC-SSL) that goes beyond modelling
single dominant label in an image to effectively utilise the information from
all the concepts present in it. MC-SSL0.0 consists of two core design concepts,
group masked model learning and learning of pseudo-concept for data token using
a momentum encoder (teacher-student) framework. The experimental results on
multi-label and multi-class image classification downstream tasks demonstrate
that MC-SSL0.0 not only surpasses existing SSL methods but also outperforms
supervised transfer learning. The source code will be made publicly available
for community to train on bigger corpus.

    

### [[2111.15341] ZZ-Net: A Universal Rotation Equivariant Architecture for 2D Point Clouds](http://arxiv.org/abs/2111.15341)


  In this paper, we are concerned with rotation equivariance on 2D point cloud
data. We describe a particular set of functions able to approximate any
continuous rotation equivariant and permutation invariant function. Based on
this result, we propose a novel neural network architecture for processing 2D
point clouds and we prove its universality for approximating functions
exhibiting these symmetries.
We also show how to extend the architecture to accept a set of 2D-2D
correspondences as indata, while maintaining similar equivariance properties.
Experiments are presented on the estimation of essential matrices in stereo
vision.

    

### [[2111.15344] Material Classification Using Active Temperature Controllable Robotic Gripper](http://arxiv.org/abs/2111.15344)


  Recognition techniques allow robots to make proper planning and control
strategies to manipulate various objects. Object recognition is more reliable
when made by combining several percepts, e.g., vision and haptics. One of the
distinguishing features of each object's material is its heat properties, and
classification can exploit heat transfer, similarly to human thermal sensation.
Thermal-based recognition has the advantage of obtaining contact surface
information in realtime by simply capturing temperature change using a tiny and
cheap sensor. However, heat transfer between a robot surface and a contact
object is strongly affected by the initial temperature and environmental
conditions. A given object's material cannot be recognized when its temperature
is the same as the robotic grippertip. We present a material classification
system using active temperature controllable robotic gripper to induce heat
flow. Subsequently, our system can recognize materials independently from their
ambient temperature. The robotic gripper surface can be regulated to any
temperature that differentiates it from the touched object's surface. We
conducted some experiments by integrating the temperature control system with
the Academic SCARA Robot, classifying them based on a long short-term memory
(LSTM) using temperature data obtained from grasping target objects.

    

### [[2111.15347] Adversarial Factor Models for the Generation of Improved Autism Diagnostic Biomarkers](http://arxiv.org/abs/2111.15347)


  Discovering reliable measures that inform on autism spectrum disorder (ASD)
diagnosis is critical for providing appropriate and timely treatment for this
neurodevelopmental disorder. In this work we present applications of
adversarial linear factor models in the creation of improved biomarkers for ASD
diagnosis. First, we demonstrate that an adversarial linear factor model can be
used to remove confounding information from our biomarkers, ensuring that they
contain only pertinent information on ASD. Second, we show this same model can
be used to learn a disentangled representation of multimodal biomarkers that
results in an increase in predictive performance. These results demonstrate
that adversarial methods can address both biomarker confounds and improve
biomarker predictive performance.

    

### [[2111.15348] Overcoming limited battery data challenges: A coupled neural network approach](http://arxiv.org/abs/2111.15348)


  The Electric Vehicle (EV) Industry has seen extraordinary growth in the last
few years. This is primarily due to an ever increasing awareness of the
detrimental environmental effects of fossil fuel powered vehicles and
availability of inexpensive Lithium-ion batteries (LIBs). In order to safely
deploy these LIBs in Electric Vehicles, certain battery states need to be
constantly monitored to ensure safe and healthy operation. The use of Machine
Learning to estimate battery states such as State-of-Charge and State-of-Health
have become an extremely active area of research. However, limited availability
of open-source diverse datasets has stifled the growth of this field, and is a
problem largely ignored in literature. In this work, we propose a novel method
of time-series battery data augmentation using deep neural networks. We
introduce and analyze the method of using two neural networks working together
to alternatively produce synthetic charging and discharging battery profiles.
One model produces battery charging profiles, and another produces battery
discharging profiles. The proposed approach is evaluated using few public
battery datasets to illustrate its effectiveness, and our results show the
efficacy of this approach to solve the challenges of limited battery data. We
also test this approach on dynamic Electric Vehicle drive cycles as well.

    

### [[2111.15363] Voint Cloud: Multi-View Point Cloud Representation for 3D Understanding](http://arxiv.org/abs/2111.15363)


  Multi-view projection methods have demonstrated promising performance on 3D
understanding tasks like 3D classification and segmentation. However, it
remains unclear how to combine such multi-view methods with the widely
available 3D point clouds. Previous methods use unlearned heuristics to combine
features at the point level. To this end, we introduce the concept of the
multi-view point cloud (Voint cloud), representing each 3D point as a set of
features extracted from several view-points. This novel 3D Voint cloud
representation combines the compactness of 3D point cloud representation with
the natural view-awareness of multi-view representation. Naturally, we can
equip this new representation with convolutional and pooling operations. We
deploy a Voint neural network (VointNet) with a theoretically established
functional form to learn representations in the Voint space. Our novel
representation achieves state-of-the-art performance on 3D classification and
retrieval on ScanObjectNN, ModelNet40, and ShapeNet Core55. Additionally, we
achieve competitive performance for 3D semantic segmentation on ShapeNet Parts.
Further analysis shows that VointNet improves the robustness to rotation and
occlusion compared to other methods.

    

### [[2111.15365] Expert Aggregation for Financial Forecasting](http://arxiv.org/abs/2111.15365)


  Machine learning algorithms dedicated to financial time series forecasting
have gained a lot of interest over the last few years. One difficulty lies in
the choice between several algorithms, as their estimation accuracy may be
unstable through time. In this paper, we propose to apply an online
aggregation-based forecasting model combining several machine learning
techniques to build a portfolio which dynamically adapts itself to market
conditions. We apply this aggregation technique to the construction of a
long-short-portfolio of individual stocks ranked on their financial
characteristics and we demonstrate how aggregation outperforms single
algorithms both in terms of performances and of stability.

    

### [[2111.15366] AI and the Everything in the Whole Wide World Benchmark](http://arxiv.org/abs/2111.15366)


  There is a tendency across different subfields in AI to valorize a small
collection of influential benchmarks. These benchmarks operate as stand-ins for
a range of anointed common problems that are frequently framed as foundational
milestones on the path towards flexible and generalizable AI systems.
State-of-the-art performance on these benchmarks is widely understood as
indicative of progress towards these long-term goals. In this position paper,
we explore the limits of such benchmarks in order to reveal the construct
validity issues in their framing as the functionally "general" broad measures
of progress they are set up to be.

    

### [[2111.15367] A Review on Graph Neural Network Methods in Financial Applications](http://arxiv.org/abs/2111.15367)


  Keeping the individual features and the complicated relations, graph data are
widely utilized and investigated. Being able to capture the structural
information by updating and aggregating nodes' representations, graph neural
network (GNN) models are gaining popularity. In the financial context, the
graph is constructed based on real-world data, which leads to complex graph
structure and thus requires sophisticated methodology. In this work, we provide
a comprehensive review of GNN models in recent financial context. We first
categorize the commonly-used financial graphs and summarize the feature
processing step for each node. Then we summarize the GNN methodology for each
graph type, application in each area, and propose some potential research
areas.

    

### [[2111.15379] Text classification problems via BERT embedding method and graph convolutional neural network](http://arxiv.org/abs/2111.15379)


  This paper presents the novel way combining the BERT embedding method and the
graph convolutional neural network. This combination is employed to solve the
text classification problem. Initially, we apply the BERT embedding method to
the texts (in the BBC news dataset and the IMDB movie reviews dataset) in order
to transform all the texts to numerical vector. Then, the graph convolutional
neural network will be applied to these numerical vectors to classify these
texts into their ap-propriate classes/labels. Experiments show that the
performance of the graph convolutional neural network model is better than the
perfor-mances of the combination of the BERT embedding method with clas-sical
machine learning models.

    

### [[2111.15382] Continuous Control With Ensemble Deep Deterministic Policy Gradients](http://arxiv.org/abs/2111.15382)


  The growth of deep reinforcement learning (RL) has brought multiple exciting
tools and methods to the field. This rapid expansion makes it important to
understand the interplay between individual elements of the RL toolbox. We
approach this task from an empirical perspective by conducting a study in the
continuous control setting. We present multiple insights of fundamental nature,
including: an average of multiple actors trained from the same data boosts
performance; the existing methods are unstable across training runs, epochs of
training, and evaluation runs; a commonly used additive action noise is not
required for effective training; a strategy based on posterior sampling
explores better than the approximated UCB combined with the weighted Bellman
backup; the weighted Bellman backup alone cannot replace the clipped double
Q-Learning; the critics' initialization plays the major role in ensemble-based
actor-critic exploration. As a conclusion, we show how existing tools can be
brought together in a novel way, giving rise to the Ensemble Deep Deterministic
Policy Gradients (ED2) method, to yield state-of-the-art results on continuous
control tasks from OpenAI Gym MuJoCo. From the practical side, ED2 is
conceptually straightforward, easy to code, and does not require knowledge
outside of the existing RL toolbox.

    

### [[2111.15397] NeuralProphet: Explainable Forecasting at Scale](http://arxiv.org/abs/2111.15397)


  We introduce NeuralProphet, a successor to Facebook Prophet, which set an
industry standard for explainable, scalable, and user-friendly forecasting
frameworks. With the proliferation of time series data, explainable forecasting
remains a challenging task for business and operational decision making. Hybrid
solutions are needed to bridge the gap between interpretable classical methods
and scalable deep learning models. We view Prophet as a precursor to such a
solution. However, Prophet lacks local context, which is essential for
forecasting the near-term future and is challenging to extend due to its Stan
backend.
NeuralProphet is a hybrid forecasting framework based on PyTorch and trained
with standard deep learning methods, making it easy for developers to extend
the framework. Local context is introduced with auto-regression and covariate
modules, which can be configured as classical linear regression or as Neural
Networks. Otherwise, NeuralProphet retains the design philosophy of Prophet and
provides the same basic model components.
Our results demonstrate that NeuralProphet produces interpretable forecast
components of equivalent or superior quality to Prophet on a set of generated
time series. NeuralProphet outperforms Prophet on a diverse collection of
real-world datasets. For short to medium-term forecasts, NeuralProphet improves
forecast accuracy by 55 to 92 percent.

    

### [[2111.15414] Neuron with Steady Response Leads to Better Generalization](http://arxiv.org/abs/2111.15414)


  Regularization can mitigate the generalization gap between training and
inference by introducing inductive bias. Existing works have already proposed
various inductive biases from diverse perspectives. However, to the best of our
knowledge, none of them explores inductive bias from the perspective of
class-dependent response distribution of individual neurons. In this paper, we
conduct a substantial analysis of the characteristics of such distribution.
Based on the analysis results, we articulate the Neuron Steadiness Hypothesis:
the neuron with similar responses to instances of the same class leads to
better generalization. Accordingly, we propose a new regularization method
called Neuron Steadiness Regularization to reduce neuron intra-class response
variance. We conduct extensive experiments on Multilayer Perceptron,
Convolutional Neural Network, and Graph Neural Network with popular benchmark
datasets of diverse domains, which show that our Neuron Steadiness
Regularization consistently outperforms the vanilla version of models with
significant gain and low additional overhead.

    

### [[2111.15422] Hierarchical Prototype Networks for Continual Graph Representation Learning](http://arxiv.org/abs/2111.15422)


  Despite significant advances in graph representation learning, little
attention has been paid to the more practical continual learning scenario in
which new categories of nodes (e.g., new research areas in citation networks,
or new types of products in co-purchasing networks) and their associated edges
are continuously emerging, causing catastrophic forgetting on previous
categories. Existing methods either ignore the rich topological information or
sacrifice plasticity for stability. To this end, we present Hierarchical
Prototype Networks (HPNs) which extract different levels of abstract knowledge
in the form of prototypes to represent the continuously expanded graphs.
Specifically, we first leverage a set of Atomic Feature Extractors (AFEs) to
encode both the elemental attribute information and the topological structure
of the target node. Next, we develop HPNs to adaptively select relevant AFEs
and represent each node with three levels of prototypes. In this way, whenever
a new category of nodes is given, only the relevant AFEs and prototypes at each
level will be activated and refined, while others remain uninterrupted to
maintain the performance over existing nodes. Theoretically, we first
demonstrate that the memory consumption of HPNs is bounded regardless of how
many tasks are encountered. Then, we prove that under mild constraints,
learning new tasks will not alter the prototypes matched to previous data,
thereby eliminating the forgetting problem. The theoretical results are
supported by experiments on five datasets, showing that HPNs not only
outperform state-of-the-art baseline techniques but also consume relatively
less memory.

    

### [[2111.15426] Efficient and robust high-dimensional sparse logistic regression via nonlinear primal-dual hybrid gradient algorithms](http://arxiv.org/abs/2111.15426)


  Logistic regression is a widely used statistical model to describe the
relationship between a binary response variable and predictor variables in data
sets. It is often used in machine learning to identify important predictor
variables. This task, variable selection, typically amounts to fitting a
logistic regression model regularized by a convex combination of $\ell_1$ and
$\ell_{2}^{2}$ penalties. Since modern big data sets can contain hundreds of
thousands to billions of predictor variables, variable selection methods depend
on efficient and robust optimization algorithms to perform well.
State-of-the-art algorithms for variable selection, however, were not
traditionally designed to handle big data sets; they either scale poorly in
size or are prone to produce unreliable numerical results. It therefore remains
challenging to perform variable selection on big data sets without access to
adequate and costly computational resources. In this paper, we propose a
nonlinear primal-dual algorithm that addresses these shortcomings.
Specifically, we propose an iterative algorithm that provably computes a
solution to a logistic regression problem regularized by an elastic net penalty
in $O(T(m,n)\log(1/\epsilon))$ operations, where $\epsilon \in (0,1)$ denotes
the tolerance and $T(m,n)$ denotes the number of arithmetic operations required
to perform matrix-vector multiplication on a data set with $m$ samples each
comprising $n$ features. This result improves on the known complexity bound of
$O(\min(m^2n,mn^2)\log(1/\epsilon))$ for first-order optimization methods such
as the classic primal-dual hybrid gradient or forward-backward splitting
methods.

    

### [[2111.15430] The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration](http://arxiv.org/abs/2111.15430)


  In spite of the dominant performances of deep neural networks, recent works
have shown that they are poorly calibrated, resulting in over-confident
predictions. Miscalibration can be exacerbated by overfitting due to the
minimization of the cross-entropy during training, as it promotes the predicted
softmax probabilities to match the one-hot label assignments. This yields a
pre-softmax activation of the correct class that is significantly larger than
the remaining activations. Recent evidence from the literature suggests that
loss functions that embed implicit or explicit maximization of the entropy of
predictions yield state-of-the-art calibration performances. We provide a
unifying constrained-optimization perspective of current state-of-the-art
calibration losses. Specifically, these losses could be viewed as
approximations of a linear penalty (or a Lagrangian) imposing equality
constraints on logit distances. This points to an important limitation of such
underlying equality constraints, whose ensuing gradients constantly push
towards a non-informative solution, which might prevent from reaching the best
compromise between the discriminative performance and calibration of the model
during gradient-based optimization. Following our observations, we propose a
simple and flexible generalization based on inequality constraints, which
imposes a controllable margin on logit distances. Comprehensive experiments on
a variety of image classification, semantic segmentation and NLP benchmarks
demonstrate that our method sets novel state-of-the-art results on these tasks
in terms of network calibration, without affecting the discriminative
performance. The code is available at this https URL .

    

### [[2111.15431] Binary Independent Component Analysis via Non-stationarity](http://arxiv.org/abs/2111.15431)


  We consider independent component analysis of binary data. While fundamental
in practice, this case has been much less developed than ICA for continuous
data. We start by assuming a linear mixing model in a continuous-valued latent
space, followed by a binary observation model. Importantly, we assume that the
sources are non-stationary; this is necessary since any non-Gaussianity would
essentially be destroyed by the binarization. Interestingly, the model allows
for closed-form likelihood by employing the cumulative distribution function of
the multivariate Gaussian distribution. In stark contrast to the
continuous-valued case, we prove non-identifiability of the model with few
observed variables; our empirical results imply identifiability when the number
of observed variables is higher. We present a practical method for binary ICA
that uses only pairwise marginals, which are faster to compute than the full
multivariate likelihood.

    

### [[2111.15432] TiWS-iForest: Isolation Forest in Weakly Supervised and Tiny ML scenarios](http://arxiv.org/abs/2111.15432)


  Unsupervised anomaly detection tackles the problem of finding anomalies
inside datasets without the labels availability; since data tagging is
typically hard or expensive to obtain, such approaches have seen huge
applicability in recent years. In this context, Isolation Forest is a popular
algorithm able to define an anomaly score by means of an ensemble of peculiar
trees called isolation trees. These are built using a random partitioning
procedure that is extremely fast and cheap to train. However, we find that the
standard algorithm might be improved in terms of memory requirements, latency
and performances; this is of particular importance in low resources scenarios
and in TinyML implementations on ultra-constrained microprocessors. Moreover,
Anomaly Detection approaches currently do not take advantage of weak
supervisions: being typically consumed in Decision Support Systems, feedback
from the users, even if rare, can be a valuable source of information that is
currently unexplored. Beside showing iForest training limitations, we propose
here TiWS-iForest, an approach that, by leveraging weak supervision is able to
reduce Isolation Forest complexity and to enhance detection performances. We
showed the effectiveness of TiWS-iForest on real word datasets and we share the
code in a public repository to enhance reproducibility.

    

### [[2111.15449] A Softmax-free Loss Function Based on Predefined Optimal-distribution of Latent Features for CNN Classifier](http://arxiv.org/abs/2111.15449)


  In the field of pattern classification, the training of convolutional neural
network classifiers is mostly end-to-end learning, and the loss function is the
constraint on the final output (posterior probability) of the network, so the
existence of Softmax is essential. In the case of end-to-end learning, there is
usually no effective loss function that completely relies on the features of
the middle layer to restrict learning, resulting in the distribution of sample
latent features is not optimal, so there is still room for improvement in
classification accuracy. Based on the concept of Predefined Evenly-Distributed
Class Centroids (PEDCC), this article proposes a Softmax-free loss function
(POD Loss) based on predefined optimal-distribution of latent features. The
loss function only restricts the latent features of the samples, including the
cosine distance between the latent feature vector of the sample and the center
of the predefined evenly-distributed class, and the correlation between the
latent features of the samples. Finally, cosine distance is used for
classification. Compared with the commonly used Softmax Loss and the typical
Softmax related AM-Softmax Loss, COT-Loss and PEDCC-Loss, experiments on
several commonly used datasets on a typical network show that the
classification performance of POD Loss is always better and easier to converge.
Code is available in this https URL.

    

### [[2111.15452] On the Generalization of Agricultural Drought Classification from Climate Data](http://arxiv.org/abs/2111.15452)


  Climate change is expected to increase the likelihood of drought events, with
severe implications for food security. Unlike other natural disasters, droughts
have a slow onset and depend on various external factors, making drought
detection in climate data difficult. In contrast to existing works that rely on
simple relative drought indices as ground-truth data, we build upon soil
moisture index (SMI) obtained from a hydrological model. This index is directly
related to insufficiently available water to vegetation. Given ERA5-Land
climate input data of six months with land use information from MODIS satellite
observation, we compare different models with and without sequential inductive
bias in classifying droughts based on SMI. We use PR-AUC as the evaluation
measure to account for the class imbalance and obtain promising results despite
a challenging time-based split. We further show in an ablation study that the
models retain their predictive capabilities given input data of coarser
resolutions, as frequently encountered in climate models.

    

### [[2111.15464] Energy-Efficient Design for a NOMA assisted STAR-RIS Network with Deep Reinforcement Learning](http://arxiv.org/abs/2111.15464)


  Simultaneous transmitting and reflecting reconfigurable intelligent surfaces
(STAR-RISs) has been considered as a promising auxiliary device to enhance the
performance of the wireless network, where users located at the different sides
of the surfaces can be simultaneously served by the transmitting and reflecting
signals. In this paper, the energy efficiency (EE) maximization problem for a
non-orthogonal multiple access (NOMA) assisted STAR-RIS downlink network is
investigated. Due to the fractional form of the EE, it is challenging to solve
the EE maximization problem by the traditional convex optimization solutions.
In this work, a deep deterministic policy gradient (DDPG)-based algorithm is
proposed to maximize the EE by jointly optimizing the transmission beamforming
vectors at the base station and the coefficients matrices at the STAR-RIS.
Simulation results demonstrate that the proposed algorithm can effectively
maximize the system EE considering the time-varying channels.

    

### [[2111.15466] Citation network applications in a scientific co-authorship recommender system](http://arxiv.org/abs/2111.15466)


  The problem of co-authors selection in the area of scientific collaborations
might be a daunting one. In this paper, we propose a new pipeline that
effectively utilizes citation data in the link prediction task on the
co-authorship network. In particular, we explore the capabilities of a
recommender system based on data aggregation strategies on different graphs.
Since graph neural networks proved their efficiency on a wide range of tasks
related to recommendation systems, we leverage them as a relevant method for
the forecasting of potential collaborations in the scientific community.

    

### [[2111.15473] New Approaches to Long Document Summarization: Fourier Transform Based Attention in a Transformer Model](http://arxiv.org/abs/2111.15473)


  In this work, we extensively redesign the newly introduced method of token
mixing using Fourier Transforms (FNET) to replace the computationally expensive
self-attention mechanism in a full transformer implementation on a long
document summarization task (> 512 tokens). As a baseline, we also carried out
long document summarization using established methods such as Longformer and
Big Bird transformer models that are capable of processing over 8000 tokens and
are currently the state of the art methods for these type of problems. The
original FNET paper implemented this in an encoder only architecture while
abstractive summarization requires both an encoder and a decoder. Since such a
pretrained transformer model does not currently exist in the public domain, we
decided to implement a full transformer based on this Fourier token mixing
approach in an encoder/decoder architecture which we trained starting with
Glove embeddings for the individual words in the corpus. We investigated a
number of different extensions to the original FNET architecture and evaluated
them on their Rouge F1-score performance on a summarization task. All
modifications showed better performance on the summarization task than when
using the original FNET encoder in a transformer architecture.

    

### [[2111.15481] Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs](http://arxiv.org/abs/2111.15481)


  In recent years, the proliferation of unmanned aerial vehicles (UAVs) has
increased dramatically. UAVs can accomplish complex or dangerous tasks in a
reliable and cost-effective way but are still limited by power consumption
problems, which pose serious constraints on the flight duration and completion
of energy-demanding tasks. The possibility of providing UAVs with advanced
decision-making capabilities in an energy-effective way would be extremely
beneficial. In this paper, we propose a practical solution to this problem that
exploits deep learning on the edge. The developed system integrates an OpenMV
microcontroller into a DJI Tello Micro Aerial Vehicle (MAV). The
microcontroller hosts a set of machine learning-enabled inference tools that
cooperate to control the navigation of the drone and complete a given mission
objective. The goal of this approach is to leverage the new opportunistic
features of TinyML through OpenMV including offline inference, low latency,
energy efficiency, and data security. The approach is successfully validated on
a practical application consisting of the onboard detection of people wearing
protection masks in a crowded environment.

    

### [[2111.15486] Playing Ping Pong with Light: Directional Emission of White Light](http://arxiv.org/abs/2111.15486)


  Over the last decades, light-emitting diodes (LED) have replaced common light
bulbs in almost every application, from flashlights in smartphones to
automotive headlights. Illuminating nightly streets requires LEDs to emit a
light spectrum that is perceived as pure white by the human eye. The power
associated with such a white light spectrum is not only distributed over the
contributing wavelengths but also over the angles of vision. For many
applications, the usable light rays are required to exit the LED in forward
direction, namely under small angles to the perpendicular. In this work, we
demonstrate that a specifically designed multi-layer thin film on top of a
white LED increases the power of pure white light emitted in forward direction.
Therefore, the deduced multi-objective optimization problem is reformulated via
a real-valued physics-guided objective function that represents the
hierarchical structure of our engineering problem. Variants of Bayesian
optimization are employed to maximize this non-deterministic objective function
based on ray tracing simulations. Eventually, the investigation of optical
properties of suitable multi-layer thin films allowed to identify the mechanism
behind the increased directionality of white light: angle and wavelength
selective filtering causes the multi-layer thin film to play ping pong with
rays of light.

    

### [[2111.15487] FROB: Few-shot ROBust Model for Classification and Out-of-Distribution Detection](http://arxiv.org/abs/2111.15487)


  Nowadays, classification and Out-of-Distribution (OoD) detection in the
few-shot setting remain challenging aims due to rarity and the limited samples
in the few-shot setting, and because of adversarial attacks. Accomplishing
these aims is important for critical systems in safety, security, and defence.
In parallel, OoD detection is challenging since deep neural network classifiers
set high confidence to OoD samples away from the training data. To address such
limitations, we propose the Few-shot ROBust (FROB) model for classification and
few-shot OoD detection. We devise FROB for improved robustness and reliable
confidence prediction for few-shot OoD detection. We generate the support
boundary of the normal class distribution and combine it with few-shot Outlier
Exposure (OE). We propose a self-supervised learning few-shot confidence
boundary methodology based on generative and discriminative models. The
contribution of FROB is the combination of the generated boundary in a
self-supervised learning manner and the imposition of low confidence at this
learned boundary. FROB implicitly generates strong adversarial samples on the
boundary and forces samples from OoD, including our boundary, to be less
confident by the classifier. FROB achieves generalization to unseen OoD with
applicability to unknown, in the wild, test sets that do not correlate to the
training datasets. To improve robustness, FROB redesigns OE to work even for
zero-shots. By including our boundary, FROB reduces the threshold linked to the
model's few-shot robustness; it maintains the OoD performance approximately
independent of the number of few-shots. The few-shot robustness analysis
evaluation of FROB on different sets and on One-Class Classification (OCC) data
shows that FROB achieves competitive performance and outperforms benchmarks in
terms of robustness to the outlier few-shot sample population and variability.

    

### [[2111.15496] Bayesian Modelling of Multivalued Power Curves from an Operational Wind Farm](http://arxiv.org/abs/2111.15496)


  Power curves capture the relationship between wind speed and output power for
a specific wind turbine. Accurate regression models of this function prove
useful in monitoring, maintenance, design, and planning. In practice, however,
the measurements do not always correspond to the ideal curve: power
curtailments will appear as (additional) functional components. Such
multivalued relationships cannot be modelled by conventional regression, and
the associated data are usually removed during pre-processing. The current work
suggests an alternative method to infer multivalued relationships in curtailed
power data. Using a population-based approach, an overlapping mixture of
probabilistic regression models is applied to signals recorded from turbines
within an operational wind farm. The model is shown to provide an accurate
representation of practical power data across the population.

    

### [[2111.15498] Assessment of Data Consistency through Cascades of Independently Recurrent Inference Machines for fast and robust accelerated MRI reconstruction](http://arxiv.org/abs/2111.15498)


  Interpretability and robustness are imperative for integrating Machine
Learning methods for accelerated Magnetic Resonance Imaging (MRI)
reconstruction in clinical applications. Doing so would allow fast high-quality
imaging of anatomy and pathology. Data Consistency (DC) is crucial for
generalization in multi-modal data and robustness in detecting pathology. This
work proposes the Cascades of Independently Recurrent Inference Machines
(CIRIM) to assess DC through unrolled optimization, implicitly by gradient
descent and explicitly by a designed term. We perform extensive comparison of
the CIRIM to other unrolled optimization methods, being the End-to-End
Variational Network (E2EVN) and the RIM, and to the UNet and Compressed Sensing
(CS). Evaluation is done in two stages. Firstly, learning on multiple trained
MRI modalities is assessed, i.e., brain data with ${T_1}$-weighting and FLAIR
contrast, and ${T_2}$-weighted knee data. Secondly, robustness is tested on
reconstructing pathology through white matter lesions in 3D FLAIR MRI data of
relapsing remitting Multiple Sclerosis (MS) patients. Results show that the
CIRIM performs best when implicitly enforcing DC, while the E2EVN requires
explicitly formulated DC. The CIRIM shows the highest lesion contrast
resolution in reconstructing the clinical MS data. Performance improves by
approximately 11% compared to CS, while the reconstruction time is twenty times
reduced.

    

### [[2111.15506] Towards a comprehensive visualization of structure in data](http://arxiv.org/abs/2111.15506)


  Dimensional data reduction methods are fundamental to explore and visualize
large data sets. Basic requirements for unsupervised data exploration are
simplicity, flexibility and scalability. However, current methods show complex
parameterizations and strong computational limitations when exploring large
data structures across scales. Here, we focus on the t-SNE algorithm and show
that a simplified parameter setup with a single control parameter, namely the
perplexity, can effectively balance local and global data structure
visualization. We also designed a chunk\&mix protocol to efficiently
parallelize t-SNE and explore data structure across a much wide range of scales
than currently available. Our parallel version of the BH-tSNE, namely pt-SNE,
converges to good global embedding, comparable to state-of-the-art solutions,
though the chunk\&mix protocol adds little noise and decreases the accuracy at
the local scale. Nonetheless, we show that simple post-processing can
efficiently restore local scale visualization, without any loss of precision at
the global scales. We expect the same approach to apply to faster embedding
algorithms other than BH-tSNE, like FIt-SNE or UMAP, thus, extending the
state-of-the-art and leading to more comprehensive data structure visualization
and analysis.

    

### [[2111.15512] What Do You See in this Patient? Behavioral Testing of Clinical NLP Models](http://arxiv.org/abs/2111.15512)


  Decision support systems based on clinical notes have the potential to
improve patient care by pointing doctors towards overseen risks. Predicting a
patient's outcome is an essential part of such systems, for which the use of
deep neural networks has shown promising results. However, the patterns learned
by these networks are mostly opaque and previous work revealed flaws regarding
the reproduction of unintended biases. We thus introduce an extendable testing
framework that evaluates the behavior of clinical outcome models regarding
changes of the input. The framework helps to understand learned patterns and
their influence on model decisions. In this work, we apply it to analyse the
change in behavior with regard to the patient characteristics gender, age and
ethnicity. Our evaluation of three current clinical NLP models demonstrates the
concrete effects of these characteristics on the models' decisions. They show
that model behavior varies drastically even when fine-tuned on the same data
and that allegedly best-performing models have not always learned the most
medically plausible patterns.

    

### [[2111.15518] Detecting Adversaries, yet Faltering to Noise? Leveraging Conditional Variational AutoEncoders for Adversary Detection in the Presence of Noisy Images](http://arxiv.org/abs/2111.15518)


  With the rapid advancement and increased use of deep learning models in image
identification, security becomes a major concern to their deployment in
safety-critical systems. Since the accuracy and robustness of deep learning
models are primarily attributed from the purity of the training samples,
therefore the deep learning architectures are often susceptible to adversarial
attacks. Adversarial attacks are often obtained by making subtle perturbations
to normal images, which are mostly imperceptible to humans, but can seriously
confuse the state-of-the-art machine learning models. What is so special in the
slightest intelligent perturbations or noise additions over normal images that
it leads to catastrophic classifications by the deep neural networks? Using
statistical hypothesis testing, we find that Conditional Variational
AutoEncoders (CVAE) are surprisingly good at detecting imperceptible image
perturbations. In this paper, we show how CVAEs can be effectively used to
detect adversarial attacks on image classification networks. We demonstrate our
results over MNIST, CIFAR-10 dataset and show how our method gives comparable
performance to the state-of-the-art methods in detecting adversaries while not
getting confused with noisy images, where most of the existing methods falter.

    

### [[2111.15519] Gram Barcodes for Histopathology Tissue Texture Retrieval](http://arxiv.org/abs/2111.15519)


  Recent advances in digital pathology have led to the need for Histopathology
Image Retrieval (HIR) systems that search through databases of biopsy images to
find similar cases to a given query image. These HIR systems allow pathologists
to effortlessly and efficiently access thousands of previously diagnosed cases
in order to exploit the knowledge in the corresponding pathology reports. Since
HIR systems may have to deal with millions of gigapixel images, the extraction
of compact and expressive image features must be available to allow for
efficient and accurate retrieval. In this paper, we propose the application of
Gram barcodes as image features for HIR systems. Unlike most feature generation
schemes, Gram barcodes are based on high-order statistics that describe tissue
texture by summarizing the correlations between different feature maps in
layers of convolutional neural networks. We run HIR experiments on three public
datasets using a pre-trained VGG19 network for Gram barcode generation and
showcase highly competitive results.

    

### [[2111.15521] Node-Level Differentially Private Graph Neural Networks](http://arxiv.org/abs/2111.15521)


  Graph Neural Networks (GNNs) are a popular technique for modelling
graph-structured data that compute node-level representations via aggregation
of information from the local neighborhood of each node. However, this
aggregation implies increased risk of revealing sensitive information, as a
node can participate in the inference for multiple nodes. This implies that
standard privacy preserving machine learning techniques, such as differentially
private stochastic gradient descent (DP-SGD) - which are designed for
situations where each data point participates in the inference for one point
only - either do not apply, or lead to inaccurate solutions. In this work, we
formally define the problem of learning 1-layer GNNs with node-level privacy,
and provide an algorithmic solution with a strong differential privacy
guarantee. Even though each node can be involved in the inference for multiple
nodes, by employing a careful sensitivity analysis anda non-trivial extension
of the privacy-by-amplification technique, our method is able to provide
accurate solutions with solid privacy parameters. Empirical evaluation on
standard benchmarks demonstrates that our method is indeed able to learn
accurate privacy preserving GNNs, while still outperforming standard
non-private methods that completely ignore graph information.

    

### [[2111.15527] Embedding Principle: a hierarchical structure of loss landscape of deep neural networks](http://arxiv.org/abs/2111.15527)


  We prove a general Embedding Principle of loss landscape of deep neural
networks (NNs) that unravels a hierarchical structure of the loss landscape of
NNs, i.e., loss landscape of an NN contains all critical points of all the
narrower NNs. This result is obtained by constructing a class of critical
embeddings which map any critical point of a narrower NN to a critical point of
the target NN with the same output function. By discovering a wide class of
general compatible critical embeddings, we provide a gross estimate of the
dimension of critical submanifolds embedded from critical points of narrower
NNs. We further prove an irreversiblility property of any critical embedding
that the number of negative/zero/positive eigenvalues of the Hessian matrix of
a critical point may increase but never decrease as an NN becomes wider through
the embedding. Using a special realization of general compatible critical
embedding, we prove a stringent necessary condition for being a "truly-bad"
critical point that never becomes a strict-saddle point through any critical
embedding. This result implies the commonplace of strict-saddle points in wide
NNs, which may be an important reason underlying the easy optimization of wide
NNs widely observed in practice.

    

### [[2111.15537] Model-Free $μ$ Synthesis via Adversarial Reinforcement Learning](http://arxiv.org/abs/2111.15537)


  Motivated by the recent empirical success of policy-based reinforcement
learning (RL), there has been a research trend studying the performance of
policy-based RL methods on standard control benchmark problems. In this paper,
we examine the effectiveness of policy-based RL methods on an important robust
control problem, namely $\mu$ synthesis. We build a connection between robust
adversarial RL and $\mu$ synthesis, and develop a model-free version of the
well-known $DK$-iteration for solving state-feedback $\mu$ synthesis with
static $D$-scaling. In the proposed algorithm, the $K$ step mimics the
classical central path algorithm via incorporating a recently-developed
double-loop adversarial RL method as a subroutine, and the $D$ step is based on
model-free finite difference approximation. Extensive numerical study is also
presented to demonstrate the utility of our proposed model-free algorithm. Our
study sheds new light on the connections between adversarial RL and robust
control.

    

### [[2111.15542] Learning to Transfer for Traffic Forecasting via Multi-task Learning](http://arxiv.org/abs/2111.15542)


  Deep neural networks have demonstrated superior performance in short-term
traffic forecasting. However, most existing traffic forecasting systems assume
that the training and testing data are drawn from the same underlying
distribution, which limits their practical applicability. The NeurIPS 2021
Traffic4cast challenge is the first of its kind dedicated to benchmarking the
robustness of traffic forecasting models towards domain shifts in space and
time. This technical report describes our solution to this challenge. In
particular, we present a multi-task learning framework for temporal and
spatio-temporal domain adaptation of traffic forecasting models. Experimental
results demonstrate that our multi-task learning approach achieves strong
empirical performance, outperforming a number of baseline domain adaptation
methods, while remaining highly efficient. The source code for this technical
report is available at this https URL.

    

### [[2111.15546] Black box tests for algorithmic stability](http://arxiv.org/abs/2111.15546)


  Algorithmic stability is a concept from learning theory that expresses the
degree to which changes to the input data (e.g., removal of a single data
point) may affect the outputs of a regression algorithm. Knowing an algorithm's
stability properties is often useful for many downstream applications -- for
example, stability is known to lead to desirable generalization properties and
predictive inference guarantees. However, many modern algorithms currently used
in practice are too complex for a theoretical analysis of their stability
properties, and thus we can only attempt to establish these properties through
an empirical exploration of the algorithm's behavior on various data sets. In
this work, we lay out a formal statistical framework for this kind of "black
box testing" without any assumptions on the algorithm or the data distribution,
and establish fundamental bounds on the ability of any black box test to
identify algorithmic stability.

    

### [[2111.15569] Scalable Machine Learning Architecture for Neonatal Seizure Detection on Ultra-Edge Devices](http://arxiv.org/abs/2111.15569)


  Neonatal seizures are a commonly encountered neurological condition. They are
the first clinical signs of a serious neurological disorder. Thus, rapid
recognition and treatment are necessary to prevent serious fatalities. The use
of electroencephalography (EEG) in the field of neurology allows precise
diagnosis of several medical conditions. However, interpreting EEG signals
needs the attention of highly specialized staff since the infant brain is
developmentally immature during the neonatal period. Detecting seizures on time
could potentially prevent the negative effects on the neurocognitive
development of the infants. In recent years, neonatal seizure detection using
machine learning algorithms have been gaining traction. Since there is a need
for the classification of bio-signals to be computationally inexpensive in the
case of seizure detection, this research presents a machine learning (ML) based
architecture that operates with comparable predictive performance as previous
models but with minimum level configuration. The proposed classifier was
trained and tested on a public dataset of NICU seizures recorded at the
Helsinki University Hospital. Our architecture achieved a best sensitivity of
87%, which is 6% more than that of the standard ML model chosen in this study.
The model size of the ML classifier is optimized to just 4.84 KB with minimum
prediction time of 182.61 milliseconds, thus enabling it to be deployed on
wearable ultra-edge devices for quick and accurate response and obviating the
need for cloud-based and other such exhaustive computational methods.

    

### [[2111.15571] An Exact Algorithm for Semi-supervised Minimum Sum-of-Squares Clustering](http://arxiv.org/abs/2111.15571)


  The minimum sum-of-squares clustering (MSSC), or k-means type clustering, is
traditionally considered an unsupervised learning task. In recent years, the
use of background knowledge to improve the cluster quality and promote
interpretability of the clustering process has become a hot research topic at
the intersection of mathematical optimization and machine learning research.
The problem of taking advantage of background information in data clustering is
called semi-supervised or constrained clustering. In this paper, we present a
new branch-and-bound algorithm for semi-supervised MSSC, where background
knowledge is incorporated as pairwise must-link and cannot-link constraints.
For the lower bound procedure, we solve the semidefinite programming relaxation
of the MSSC discrete optimization model, and we use a cutting-plane procedure
for strengthening the bound. For the upper bound, instead, by using integer
programming tools, we propose an adaptation of the k-means algorithm to the
constrained case. For the first time, the proposed global optimization
algorithm efficiently manages to solve real-world instances up to 800 data
points with different combinations of must-link and cannot-link constraints and
with a generic number of features. This problem size is about four times larger
than the one of the instances solved by state-of-the-art exact algorithms.

    

### [[2111.15592] MapReader: A Computer Vision Pipeline for the Semantic Exploration of Maps at Scale](http://arxiv.org/abs/2111.15592)


  We present MapReader, a free, open-source software library written in Python
for analyzing large map collections (scanned or born-digital). This library
transforms the way historians can use maps by turning extensive, homogeneous
map sets into searchable primary sources. MapReader allows users with little or
no computer vision expertise to i) retrieve maps via web-servers; ii)
preprocess and divide them into patches; iii) annotate patches; iv) train,
fine-tune, and evaluate deep neural network models; and v) create structured
data about map content. We demonstrate how MapReader enables historians to
interpret a collection of $\approx$16K nineteenth-century Ordnance Survey map
sheets ($\approx$30.5M patches), foregrounding the challenge of translating
visual markers into machine-readable data. We present a case study focusing on
British rail infrastructure and buildings as depicted on these maps. We also
show how the outputs from the MapReader pipeline can be linked to other,
external datasets, which we use to evaluate as well as enrich and interpret the
results. We release $\approx$62K manually annotated patches used here for
training and evaluating the models.

    

### [[2111.15597] Surrogate-based optimization using an artificial neural network for a parameter identification in a 3D marine ecosystem model](http://arxiv.org/abs/2111.15597)


  Parameter identification for marine ecosystem models is important for the
assessment and validation of marine ecosystem models against observational
data. The surrogate-based optimization (SBO) is a computationally efficient
method to optimize complex models. SBO replaces the computationally expensive
(high-fidelity) model by a surrogate constructed from a less accurate but
computationally cheaper (low-fidelity) model in combination with an appropriate
correction approach, which improves the accuracy of the low-fidelity model. To
construct a computationally cheap low-fidelity model, we tested three different
approaches to compute an approximation of the annual periodic solution (i.e., a
steady annual cycle) of a marine ecosystem model: firstly, a reduced number of
spin-up iterations (several decades instead of millennia), secondly, an
artificial neural network (ANN) approximating the steady annual cycle and,
finally, a combination of both approaches. Except for the low-fidelity model
using only the ANN, the SBO yielded a solution close to the target and reduced
the computational effort significantly. If an ANN approximating appropriately a
marine ecosystem model is available, the SBO using this ANN as low-fidelity
model presents a promising and computational efficient method for the
validation.

    

### [[2111.15602] Fine-grained prediction of food insecurity using news streams](http://arxiv.org/abs/2111.15602)


  Anticipating the outbreak of a food crisis is crucial to efficiently allocate
emergency relief and reduce human suffering. However, existing food insecurity
early warning systems rely on risk measures that are often delayed, outdated,
or incomplete. Here, we leverage recent advances in deep learning to extract
high-frequency precursors to food crises from the text of a large corpus of
news articles about fragile states published between 1980 and 2020. Our text
features are causally grounded, interpretable, validated by existing data, and
allow us to predict 32% more food crises than existing models up to three
months ahead of time at the district level across 15 fragile states. These
results could have profound implications on how humanitarian aid gets allocated
and open new avenues for machine learning to improve decision making in
data-scarce environments.

    

### [[2111.15605] Synthetic weather radar using hybrid quantum-classical machine learning](http://arxiv.org/abs/2111.15605)


  The availability of high-resolution weather radar images underpins effective
forecasting and decision-making. In regions beyond traditional radar coverage,
generative models have emerged as an important synthetic capability, fusing
more ubiquitous data sources, such as satellite imagery and numerical weather
models, into accurate radar-like products. Here, we demonstrate methods to
augment conventional convolutional neural networks with quantum-assisted models
for generative tasks in global synthetic weather radar. We show that quantum
kernels can, in principle, perform fundamentally more complex tasks than
classical learning machines on the relevant underlying data. Our results
establish synthetic weather radar as an effective heuristic benchmark for
quantum computing capabilities and set the stage for detailed quantum advantage
benchmarking on a high-impact operationally relevant problem.

    

### [[2111.15611] The Power of Communication in a Distributed Multi-Agent System](http://arxiv.org/abs/2111.15611)


  Single-Agent (SA) Reinforcement Learning systems have shown outstanding
re-sults on non-stationary problems. However, Multi-Agent Reinforcement
Learning(MARL) can surpass SA systems generally and when scaling. Furthermore,
MAsystems can be super-powered by collaboration, which can happen through
ob-serving others, or a communication system used to share information
betweencollaborators. Here, we developed a distributed MA learning mechanism
withthe ability to communicate based on decentralised partially observable
Markovdecision processes (Dec-POMDPs) and Graph Neural Networks (GNNs).
Minimis-ing the time and energy consumed by training Machine Learning models
whileimproving performance can be achieved by collaborative MA mechanisms.
Wedemonstrate this in a real-world scenario, an offshore wind farm, including a
set ofdistributed wind turbines, where the objective is to maximise collective
efficiency.Compared to a SA system, MA collaboration has shown significantly
reducedtraining time and higher cumulative rewards in unseen and scaled
scenarios.

    

### [[2111.15615] Semi-Local Convolutions for LiDAR Scan Processing](http://arxiv.org/abs/2111.15615)


  A number of applications, such as mobile robots or automated vehicles, use
LiDAR sensors to obtain detailed information about their three-dimensional
surroundings. Many methods use image-like projections to efficiently process
these LiDAR measurements and use deep convolutional neural networks to predict
semantic classes for each point in the scan. The spatial stationary assumption
enables the usage of convolutions. However, LiDAR scans exhibit large
differences in appearance over the vertical axis. Therefore, we propose semi
local convolution (SLC), a convolution layer with reduced amount of
weight-sharing along the vertical dimension. We are first to investigate the
usage of such a layer independent of any other model changes. Our experiments
did not show any improvement over traditional convolution layers in terms of
segmentation IoU or accuracy.

    

### [[2111.15623] Towards Modularity Optimization Using Reinforcement Learning to Community Detection in Dynamic Social Networks](http://arxiv.org/abs/2111.15623)


  The identification of community structure in a social network is an important
problem tackled in the literature of network analysis. There are many solutions
to this problem using a static scenario, when facing a dynamic scenario some
solutions may be adapted but others simply do not fit, moreover when
considering the demand to analyze constantly growing networks. In this context,
we propose an approach to the problem of community detection in dynamic
networks based on a reinforcement learning strategy to deal with changes on big
networks using a local optimization on the modularity score of the changed
entities. An experiment using synthetic and real-world dynamic network data
shows results comparable to static scenarios.

    

### [[2111.15624] Image Style Transfer and Content-Style Disentanglement](http://arxiv.org/abs/2111.15624)


  We propose a way of learning disentangled content-style representation of
image, allowing us to extrapolate images to any style as well as interpolate
between any pair of styles. By augmenting data set in a supervised setting and
imposing triplet loss, we ensure the separation of information encoded by
content and style representation. We also make use of cycle-consistency loss to
guarantee that images could be reconstructed faithfully by their
representation.

    

### [[2111.15629] DiPD: Disruptive event Prediction Dataset from Twitter](http://arxiv.org/abs/2111.15629)


  Riots and protests, if gone out of control, can cause havoc in a country. We
have seen examples of this, such as the BLM movement, climate strikes, CAA
Movement, and many more, which caused disruption to a large extent. Our motive
behind creating this dataset was to use it to develop machine learning systems
that can give its users insight into the trending events going on and alert
them about the events that could lead to disruption in the nation. If any event
starts going out of control, it can be handled and mitigated by monitoring it
before the matter escalates. This dataset collects tweets of past or ongoing
events known to have caused disruption and labels these tweets as 1. We also
collect tweets that are considered non-eventful and label them as 0 so that
they can also be used to train a classification system. The dataset contains
94855 records of unique events and 168706 records of unique non-events, thus
giving the total dataset 263561 records. We extract multiple features from the
tweets, such as the user's follower count and the user's location, to
understand the impact and reach of the tweets. This dataset might be useful in
various event related machine learning problems such as event classification,
event recognition, and so on.

    

### [[2111.15631] Neural Symplectic Integrator with Hamiltonian Inductive Bias for the Gravitational $N$-body Problem](http://arxiv.org/abs/2111.15631)


  The gravitational $N$-body problem, which is fundamentally important in
astrophysics to predict the motion of $N$ celestial bodies under the mutual
gravity of each other, is usually solved numerically because there is no known
general analytical solution for $N>2$. Can an $N$-body problem be solved
accurately by a neural network (NN)? Can a NN observe long-term conservation of
energy and orbital angular momentum? Inspired by Wistom & Holman (1991)'s
symplectic map, we present a neural $N$-body integrator for splitting the
Hamiltonian into a two-body part, solvable analytically, and an interaction
part that we approximate with a NN. Our neural symplectic $N$-body code
integrates a general three-body system for $10^{5}$ steps without diverting
from the ground truth dynamics obtained from a traditional $N$-body integrator.
Moreover, it exhibits good inductive bias by successfully predicting the
evolution of $N$-body systems that are no part of the training set.

    

### [[2111.15635] Improving random walk rankings with feature selection and imputation](http://arxiv.org/abs/2111.15635)


  The Science4cast Competition consists of predicting new links in a semantic
network, with each node representing a concept and each edge representing a
link proposed by a paper relating two concepts. This network contains
information from 1994-2017, with a discretization of days (which represents the
publication date of the underlying papers). Team Hash Brown's final submission,
\emph{ee5a}, achieved a score of 0.92738 on the test set. Our team's score
ranks \emph{second place}, 0.01 below the winner's score. This paper details
our model, its intuition, and the performance of its variations in the test
set.

    

### [[2111.15638] Radio-Frequency Multi-Mode OAM Detection Based on UCA Samples Learning](http://arxiv.org/abs/2111.15638)


  Orbital angular momentum (OAM) at radio-frequency provides a novel approach
of multiplexing a set of orthogonal modes on the same frequency channel to
achieve high spectral efficiencies. However, classical phase gradient-based OAM
mode detection methods require perfect alignment of transmit and receive
antennas, which greatly challenges the practical application of OAM
communications. In this paper, we first show the effect of non-parallel
misalignment on the OAM phase structure, and then propose the OAM mode
detection method based on uniform circular array (UCA) samples learning for the
more general alignment or non-parallel misalignment case. Specifically, we
applied three classifiers: K-nearest neighbor (KNN), support vector machine
(SVM), and back-propagation neural network (BPNN) to both single-mode and
multi-mode OAM detection. The simulation results validate that the proposed
learning-based OAM mode detection methods are robust to misalignment errors and
especially BPNN classifier has the best generalization performance.

    

### [[2111.15639] DeDUCE: Generating Counterfactual Explanations Efficiently](http://arxiv.org/abs/2111.15639)


  When an image classifier outputs a wrong class label, it can be helpful to
see what changes in the image would lead to a correct classification. This is
the aim of algorithms generating counterfactual explanations. However, there is
no easily scalable method to generate such counterfactuals. We develop a new
algorithm providing counterfactual explanations for large image classifiers
trained with spectral normalisation at low computational cost. We empirically
compare this algorithm against baselines from the literature; our novel
algorithm consistently finds counterfactuals that are much closer to the
original inputs. At the same time, the realism of these counterfactuals is
comparable to the baselines. The code for all experiments is available at
this https URL.

    

### [[2111.15645] Survey Descent: A Multipoint Generalization of Gradient Descent for Nonsmooth Optimization](http://arxiv.org/abs/2111.15645)


  For strongly convex objectives that are smooth, the classical theory of
gradient descent ensures linear convergence relative to the number of gradient
evaluations. An analogous nonsmooth theory is challenging: even when the
objective is smooth at every iterate, the corresponding local models are
unstable, and traditional remedies need unpredictably many cutting planes. We
instead propose a multipoint generalization of the gradient descent iteration
for local optimization. While designed with general objectives in mind, we are
motivated by a "max-of-smooth" model that captures subdifferential dimension at
optimality. We prove linear convergence when the objective is itself
max-of-smooth, and experiments suggest a more general phenomenon.

    

### [[2111.15646] Exponentially Tilted Gaussian Prior for Variational Autoencoder](http://arxiv.org/abs/2111.15646)


  An important propertyfor deep neural networks to possess is the ability to
perform robust out of distribution detection (OOD) on previously unseen data.
This property is essential for safety purposes when deploying models for real
world applications. Recent studies show that probabilistic generative models
can perform poorly on this task, which is surprising given that they seek to
estimate the likelihood of training data. To alleviate this issue, we propose
the exponentially tilted Gaussian prior distribution for the Variational
Autoencoder (VAE). With this prior, we are able to achieve state-of-the art
results using just the negative log likelihood that the VAE naturally assigns,
while being orders of magnitude faster than some competitive methods. We also
show that our model produces high quality image samples which are more crisp
than that of a standard Gaussian VAE. The new prior distribution has a very
simple implementation which uses a Kullback Leibler divergence that compares
the difference between a latent vector's length, and the radius of a sphere.

    

### [[2111.15651] Leveraging The Topological Consistencies of Learning in Deep Neural Networks](http://arxiv.org/abs/2111.15651)


  Recently, methods have been developed to accurately predict the testing
performance of a Deep Neural Network (DNN) on a particular task, given
statistics of its underlying topological structure. However, further leveraging
this newly found insight for practical applications is intractable due to the
high computational cost in terms of time and memory. In this work, we define a
new class of topological features that accurately characterize the progress of
learning while being quick to compute during running time. Additionally, our
proposed topological features are readily equipped for backpropagation, meaning
that they can be incorporated in end-to-end training. Our newly developed
practical topological characterization of DNNs allows for an additional set of
applications. We first show we can predict the performance of a DNN without a
testing set and without the need for high-performance computing. We also
demonstrate our topological characterization of DNNs is effective in estimating
task similarity. Lastly, we show we can induce learning in DNNs by actively
constraining the DNN's topological structure. This opens up new avenues in
constricting the underlying structure of DNNs in a meta-learning framework.

    

### [[2111.15655] Studying Hadronization by Machine Learning Techniques](http://arxiv.org/abs/2111.15655)


  Hadronization is a non-perturbative process, which theoretical description
can not be deduced from first principles. Modeling hadron formation, requires
several assumptions and various phenomenological approaches. Utilizing
state-of-the-art Computer Vision and Deep Learning algorithms, it is eventually
possible to train neural networks to learn non-linear and non-perturbative
features of the physical processes. In this study, results of two ResNet
networks are presented by investigating global and kinematical quantities,
indeed jet- and event-shape variables. The widely used Lund string
fragmentation model is applied as a baseline in $\sqrt{s}= 7$ TeV proton-proton
collisions to predict the most relevant observables at further LHC energies.

    

### [[2111.15664] Donut: Document Understanding Transformer without OCR](http://arxiv.org/abs/2111.15664)


  Understanding document images (e.g., invoices) has been an important research
topic and has many applications in document processing automation. Through the
latest advances in deep learning-based Optical Character Recognition (OCR),
current Visual Document Understanding (VDU) systems have come to be designed
based on OCR. Although such OCR-based approach promise reasonable performance,
they suffer from critical problems induced by the OCR, e.g., (1) expensive
computational costs and (2) performance degradation due to the OCR error
propagation. In this paper, we propose a novel VDU model that is end-to-end
trainable without underpinning OCR framework. To this end, we propose a new
task and a synthetic document image generator to pre-train the model to
mitigate the dependencies on large-scale real document images. Our approach
achieves state-of-the-art performance on various document understanding tasks
in public benchmark datasets and private industrial service datasets. Through
extensive experiments and analysis, we demonstrate the effectiveness of the
proposed model especially with consideration for a real-world application.

    

### [[1910.00943] Models under which random forests perform badly; consequences for applications](http://arxiv.org/abs/1910.00943)


  We give examples of data-generating models under which Breiman's random
forest may be extremely slow to converge to the optimal predictor or even fail
to be consistent. The evidence provided for these properties is based on mostly
intuitive arguments, similar to those used earlier with simpler examples, and
on numerical experiments. Although one can always choose models under which
random forests perform very badly, we show that simple methods based on
statistics of `variable use' and `variable importance' can often be used to
construct a much better predictor based on a `many-armed' random forest
obtained by forcing initial splits on variables which the default version of
the algorithm tends to ignore.

    

### [[2005.02607] Towards quantum advantage via topological data analysis](http://arxiv.org/abs/2005.02607)


  Even after decades of quantum computing development, examples of generally
useful quantum algorithms with exponential speedups over classical counterparts
are scarce. Recent progress in quantum algorithms for linear-algebra positioned
quantum machine learning (QML) as a potential source of such useful exponential
improvements. Yet, in an unexpected development, a recent series of
"dequantization" results has equally rapidly removed the promise of exponential
speedups for several QML algorithms. This raises the critical question whether
exponential speedups of other linear-algebraic QML algorithms persist. In this
paper, we study the quantum-algorithmic methods behind the algorithm for
topological data analysis of Lloyd, Garnerone and Zanardi through this lens. We
provide evidence that the problem solved by this algorithm is classically
intractable by showing that its natural generalization is as hard as simulating
the one clean qubit model -- which is widely believed to require
superpolynomial time on a classical computer -- and is thus very likely immune
to dequantizations. Based on this result, we provide a number of new quantum
algorithms for problems such as rank estimation and complex network analysis,
along with complexity-theoretic evidence for their classical intractability.
Furthermore, we analyze the suitability of the proposed quantum algorithms for
near-term implementations. Our results provide a number of useful applications
for full-blown, and restricted quantum computers with a guaranteed exponential
speedup over classical methods, recovering some of the potential for
linear-algebraic QML to become one of quantum computing's killer applications.

    

### [[2006.14551] Prediction with Approximated Gaussian Process Dynamical Models](http://arxiv.org/abs/2006.14551)


  The modeling and simulation of dynamical systems is a necessary step for many
control approaches. Using classical, parameter-based techniques for modeling of
modern systems, e.g., soft robotics or human-robot interaction, is often
challenging or even infeasible due to the complexity of the system dynamics. In
contrast, data-driven approaches need only a minimum of prior knowledge and
scale with the complexity of the system. In particular, Gaussian process
dynamical models (GPDMs) provide very promising results for the modeling of
complex dynamics. However, the control properties of these GP models are just
sparsely researched, which leads to a "blackbox" treatment in modeling and
control scenarios. In addition, the sampling of GPDMs for prediction purpose
respecting their non-parametric nature results in non-Markovian dynamics making
the theoretical analysis challenging. In this article, we present approximated
GPDMs which are Markov and analyze their control theoretical properties. Among
others, the approximated error is analyzed and conditions for boundedness of
the trajectories are provided. The outcomes are illustrated with numerical
examples that show the power of the approximated models while the the
computational time is significantly reduced.

    

### [[2009.03864] Contraction $\mathcal{L}_1$-Adaptive Control using Gaussian Processes](http://arxiv.org/abs/2009.03864)


  We present $\mathcal{CL}_1$-$\mathcal{GP}$, a control framework that enables
safe simultaneous learning and control for systems subject to uncertainties.
The two main constituents are contraction theory-based $\mathcal{L}_1$
($\mathcal{CL}_1$) control and Bayesian learning in the form of Gaussian
process (GP) regression. The $\mathcal{CL}_1$ controller ensures that control
objectives are met while providing safety certificates. Furthermore,
$\mathcal{CL}_1$-$\mathcal{GP}$ incorporates any available data into a GP model
of uncertainties, which improves performance and enables the motion planner to
achieve optimality safely. This way, the safe operation of the system is always
guaranteed, even during the learning transients. We provide a few illustrative
examples for the safe learning and control of planar quadrotor systems in a
variety of environments.

    

### [[2010.09470] Dos and Don'ts of Machine Learning in Computer Security](http://arxiv.org/abs/2010.09470)


  With the growing processing power of computing systems and the increasing
availability of massive datasets, machine learning algorithms have led to major
breakthroughs in many different areas. This development has influenced computer
security, spawning a series of work on learning-based security systems, such as
for malware detection, vulnerability discovery, and binary code analysis.
Despite great potential, machine learning in security is prone to subtle
pitfalls that undermine its performance and render learning-based systems
potentially unsuitable for security tasks and practical deployment. In this
paper, we look at this problem with critical eyes. First, we identify common
pitfalls in the design, implementation, and evaluation of learning-based
security systems. We conduct a study of 30 papers from top-tier security
conferences within the past 10 years, confirming that these pitfalls are
widespread in the current security literature. In an empirical analysis, we
further demonstrate how individual pitfalls can lead to unrealistic performance
and interpretations, obstructing the understanding of the security problem at
hand. As a remedy, we propose actionable recommendations to support researchers
in avoiding or mitigating the pitfalls where possible. Furthermore, we identify
open problems when applying machine learning in security and provide directions
for further research.

    

### [[2012.09831] On Episodes, Prototypical Networks, and Few-shot Learning](http://arxiv.org/abs/2012.09831)


  Episodic learning is a popular practice among researchers and practitioners
interested in few-shot learning. It consists of organising training in a series
of learning problems (or episodes), each divided into a small training and
validation subset to mimic the circumstances encountered during evaluation. But
is this always necessary? In this paper, we investigate the usefulness of
episodic learning in methods which use nonparametric approaches, such as
nearest neighbours, at the level of the episode. For these methods, we not only
show how the constraints imposed by episodic learning are not necessary, but
that they in fact lead to a data-inefficient way of exploiting training
batches. We conduct a wide range of ablative experiments with Matching and
Prototypical Networks, two of the most popular methods that use nonparametric
approaches at the level of the episode. Their "non-episodic" counterparts are
considerably simpler, have less hyperparameters, and improve their performance
in multiple few-shot classification datasets.

    

### [[2101.00307] Quantifying Spatial Homogeneity of Urban Road Networks via Graph Neural Networks](http://arxiv.org/abs/2101.00307)


  Quantifying the topological similarities of different parts of urban road
networks (URNs) enables us to understand the urban growth patterns. While
conventional statistics provide useful information about characteristics of
either a single node's direct neighbors or the entire network, such metrics
fail to measure the similarities of subnetworks considering local indirect
neighborhood relationships. In this study, we propose a graph-based
machine-learning method to quantify the spatial homogeneity of subnetworks. We
apply the method to 11,790 urban road networks across 30 cities worldwide to
measure the spatial homogeneity of road networks within each city and across
different cities. We find that intra-city spatial homogeneity is highly
associated with socioeconomic statuses such as GDP and population growth.
Moreover, inter-city spatial homogeneity obtained by transferring the model
across different cities, reveals the inter-city similarity of urban network
structures originating in Europe, passed on to cities in the US and Asia.
Socioeconomic development and inter-city similarity revealed using our method
can be leveraged to understand and transfer insights across cities. It also
enables us to address urban policy challenges including network planning in
rapidly urbanizing areas and combating regional inequality.

    

### [[2101.11174] Graph Neural Network for Traffic Forecasting: A Survey](http://arxiv.org/abs/2101.11174)


  Traffic forecasting is important for the success of intelligent
transportation systems. Deep learning models, including convolution neural
networks and recurrent neural networks, have been extensively applied in
traffic forecasting problems to model spatial and temporal dependencies. In
recent years, to model the graph structures in transportation systems as well
as contextual information, graph neural networks have been introduced and have
achieved state-of-the-art performance in a series of traffic forecasting
problems. In this survey, we review the rapidly growing body of research using
different graph neural networks, e.g. graph convolutional and graph attention
networks, in various traffic forecasting problems, e.g. road traffic flow and
speed forecasting, passenger flow forecasting in urban rail transit systems,
and demand forecasting in ride-hailing platforms. We also present a
comprehensive list of open data and source resources for each problem and
identify future research directions. To the best of our knowledge, this paper
is the first comprehensive survey that explores the application of graph neural
networks for traffic forecasting problems. We have also created a public GitHub
repository where the latest papers, open data, and source resources will be
updated.

    

### [[2102.07005] Clustering Interval-Censored Time-Series for Disease Phenotyping](http://arxiv.org/abs/2102.07005)


  Unsupervised learning is often used to uncover clusters in data. However,
different kinds of noise may impede the discovery of useful patterns from
real-world time-series data. In this work, we focus on mitigating the
interference of interval censoring in the task of clustering for disease
phenotyping. We develop a deep generative, continuous-time model of time-series
data that clusters time-series while correcting for censorship time. We provide
conditions under which clusters and the amount of delayed entry may be
identified from data under a noiseless model.

    

### [[2104.07639] Robust Optimization for Multilingual Translation with Imbalanced Data](http://arxiv.org/abs/2104.07639)


  Multilingual models are parameter-efficient and especially effective in
improving low-resource languages by leveraging crosslingual transfer. Despite
recent advance in massive multilingual translation with ever-growing model and
data, how to effectively train multilingual models has not been well
understood. In this paper, we show that a common situation in multilingual
training, data imbalance among languages, poses optimization tension between
high resource and low resource languages where the found multilingual solution
is often sub-optimal for low resources. We show that common training method
which upsamples low resources can not robustly optimize population loss with
risks of either underfitting high resource languages or overfitting low
resource ones. Drawing on recent findings on the geometry of loss landscape and
its effect on generalization, we propose a principled optimization algorithm,
Curvature Aware Task Scaling (CATS), which adaptively rescales gradients from
different tasks with a meta objective of guiding multilingual training to
low-curvature neighborhoods with uniformly low loss for all languages. We ran
experiments on common benchmarks (TED, WMT and OPUS-100) with varying degrees
of data imbalance. CATS effectively improved multilingual optimization and as a
result demonstrated consistent gains on low resources ($+0.8$ to $+2.2$ BLEU)
without hurting high resources. In addition, CATS is robust to
overparameterization and large batch size training, making it a promising
training method for massive multilingual models that truly improve low resource
languages.

    

### [[2105.14557] Robust Dynamic Network Embedding via Ensembles](http://arxiv.org/abs/2105.14557)


  Dynamic Network Embedding (DNE) has recently attracted considerable attention
due to the advantage of network embedding in various fields and the dynamic
nature of many real-world networks. An input dynamic network to DNE is often
assumed to have smooth changes over snapshots, which however would not hold for
all real-world scenarios. It is natural to ask if existing DNE methods can
perform well for an input dynamic network without smooth changes. To quantify
it, an index called Degree of Changes (DoCs) is suggested so that the smaller
DoCs indicates the smoother changes. Our comparative study shows several DNE
methods are not robust enough to different DoCs even if the corresponding input
dynamic networks come from the same dataset, which would make these methods
unreliable and hard to use for unknown real-world applications. To propose an
effective and more robust DNE method, we follow the notion of ensembles where
each base learner adopts an incremental Skip-Gram embedding model. To further
boost the performance, a simple yet effective strategy is designed to enhance
the diversity among base learners at each timestep by capturing different
levels of local-global topology. Extensive experiments demonstrate the superior
effectiveness and robustness of the proposed method compared to
state-of-the-art DNE methods, as well as the benefits of special designs in the
proposed method and its scalability.

    

### [[2106.08138] Learning Full Configuration Interaction Electron Correlations with Deep Learning](http://arxiv.org/abs/2106.08138)


  In this report, we present a deep learning framework termed the Electron
Correlation Potential Neural Network (eCPNN) that can learn succinct and
compact potential functions. These functions can effectively describe the
complex instantaneous spatial correlations among electrons in many--electron
atoms. The eCPNN was trained in an unsupervised manner with limited information
from Full Configuration Interaction (FCI) one--electron density functions
within predefined limits of accuracy. Using the effective correlation potential
functions generated by eCPNN, we can predict the total energies of each of the
studied atomic systems with a remarkable accuracy when compared to FCI
energies.

    

### [[2106.13700] ViTAS: Vision Transformer Architecture Search](http://arxiv.org/abs/2106.13700)


  Vision transformers (ViTs) inherited the success of NLP but their structures
have not been sufficiently investigated and optimized for visual tasks. One of
the simplest solutions is to directly search the optimal one via the widely
used neural architecture search (NAS) in CNNs. However, we empirically find
this straightforward adaptation would encounter catastrophic failures and be
frustratingly unstable for the training of superformer. In this paper, we argue
that since ViTs mainly operate on token embeddings with little inductive bias,
imbalance of channels for different architectures would worsen the
weight-sharing assumption and cause the training instability as a result.
Therefore, we develop a new cyclic weight-sharing mechanism for token
embeddings of the ViTs, which enables each channel could more evenly contribute
to all candidate architectures. Besides, we also propose identity shifting to
alleviate the many-to-one issue in superformer and leverage weak augmentation
and regularization techniques for more steady training empirically. Based on
these, our proposed method, ViTAS, has achieved significant superiority in both
DeiT- and Twins-based ViTs. For example, with only $1.4$G FLOPs budget, our
searched architecture has $3.3\%$ ImageNet-$1$k accuracy than the baseline
DeiT. With $3.0$G FLOPs, our results achieve $82.0\%$ accuracy on
ImageNet-$1$k, and $45.9\%$ mAP on COCO$2017$ which is $2.4\%$ superior than
other ViTs.

    

### [[2106.15535] Subgroup Generalization and Fairness of Graph Neural Networks](http://arxiv.org/abs/2106.15535)


  Despite enormous successful applications of graph neural networks (GNNs),
theoretical understanding of their generalization ability, especially for
node-level tasks where data are not independent and identically-distributed
(IID), has been sparse. The theoretical investigation of the generalization
performance is beneficial for understanding fundamental issues (such as
fairness) of GNN models and designing better learning methods. In this paper,
we present a novel PAC-Bayesian analysis for GNNs under a non-IID
semi-supervised learning setup. Moreover, we analyze the generalization
performances on different subgroups of unlabeled nodes, which allows us to
further study an accuracy-(dis)parity-style (un)fairness of GNNs from a
theoretical perspective. Under reasonable assumptions, we demonstrate that the
distance between a test subgroup and the training set can be a key factor
affecting the GNN performance on that subgroup, which calls special attention
to the training node selection for fair learning. Experiments across multiple
GNN models and datasets support our theoretical results.

    

### [[2107.02170] On Model Calibration for Long-Tailed Object Detection and Instance Segmentation](http://arxiv.org/abs/2107.02170)


  Vanilla models for object detection and instance segmentation suffer from the
heavy bias toward detecting frequent objects in the long-tailed setting.
Existing methods address this issue mostly during training, e.g., by
re-sampling or re-weighting. In this paper, we investigate a largely overlooked
approach -- post-processing calibration of confidence scores. We propose
NorCal, Normalized Calibration for long-tailed object detection and instance
segmentation, a simple and straightforward recipe that reweighs the predicted
scores of each class by its training sample size. We show that separately
handling the background class and normalizing the scores over classes for each
proposal are keys to achieving superior performance. On the LVIS dataset,
NorCal can effectively improve nearly all the baseline models not only on rare
classes but also on common and frequent classes. Finally, we conduct extensive
analysis and ablation studies to offer insights into various modeling choices
and mechanisms of our approach. Our code is publicly available at
this https URL.

    

### [[2111.07819] Testing the Generalization of Neural Language Models for COVID-19 Misinformation Detection](http://arxiv.org/abs/2111.07819)


  A drastic rise in potentially life-threatening misinformation has been a
by-product of the COVID-19 pandemic. Computational support to identify false
information within the massive body of data on the topic is crucial to prevent
harm. Researchers proposed many methods for flagging online misinformation
related to COVID-19. However, these methods predominantly target specific
content types (e.g., news) or platforms (e.g., Twitter). The methods'
capabilities to generalize were largely unclear so far. We evaluate fifteen
Transformer-based models on five COVID-19 misinformation datasets that include
social media posts, news articles, and scientific papers to fill this gap. We
show tokenizers and models tailored to COVID-19 data do not provide a
significant advantage over general-purpose ones. Our study provides a realistic
assessment of models for detecting COVID-19 misinformation. We expect that
evaluating a broad spectrum of datasets and models will benefit future research
in developing misinformation detection systems.

    

### [[2111.11249] LeQua@CLEF2022: Learning to Quantify](http://arxiv.org/abs/2111.11249)


  LeQua 2022 is a new lab for the evaluation of methods for "learning to
quantify" in textual datasets, i.e., for training predictors of the relative
frequencies of the classes of interest in sets of unlabelled textual documents.
While these predictions could be easily achieved by first classifying all
documents via a text classifier and then counting the numbers of documents
assigned to the classes, a growing body of literature has shown this approach
to be suboptimal, and has proposed better methods. The goal of this lab is to
provide a setting for the comparative evaluation of methods for learning to
quantify, both in the binary setting and in the single-label multiclass
setting. For each such setting we provide data either in ready-made vector form
or in raw document form.

    

### [[2111.14580] Amortized Implicit Differentiation for Stochastic Bilevel Optimization](http://arxiv.org/abs/2111.14580)


  We study a class of algorithms for solving bilevel optimization problems in
both stochastic and deterministic settings when the inner-level objective is
strongly convex. Specifically, we consider algorithms based on inexact implicit
differentiation and we exploit a warm-start strategy to amortize the estimation
of the exact gradient. We then introduce a unified theoretical framework
inspired by the study of singularly perturbed systems (Habets, 1974) to analyze
such amortized algorithms. By using this framework, our analysis shows these
algorithms to match the computational complexity of oracle methods that have
access to an unbiased estimate of the gradient, thus outperforming many
existing results for bilevel optimization. We illustrate these findings on
synthetic experiments and demonstrate the efficiency of these algorithms on
hyper-parameter optimization experiments involving several thousands of
variables.

    

### [[2111.14693] SAGCI-System: Towards Sample-Efficient, Generalizable, Compositional, and Incremental Robot Learning](http://arxiv.org/abs/2111.14693)


  Building general-purpose robots to perform an enormous amount of tasks in a
large variety of environments at the human level is notoriously complicated. It
requires the robot learning to be sample-efficient, generalizable,
compositional, and incremental. In this work, we introduce a systematic
learning framework called SAGCI-system towards achieving these above four
requirements. Our system first takes the raw point clouds gathered by the
camera mounted on the robot's wrist as the inputs and produces initial modeling
of the surrounding environment represented as a URDF. Our system adopts a
learning-augmented differentiable simulation that loads the URDF. The robot
then utilizes the interactive perception to interact with the environments to
online verify and modify the URDF. Leveraging the simulation, we propose a new
model-based RL algorithm combining object-centric and robot-centric approaches
to efficiently produce policies to accomplish manipulation tasks. We apply our
system to perform articulated object manipulation, both in the simulation and
the real world. Extensive experiments demonstrate the effectiveness of our
proposed learning framework. Supplemental materials and videos are available on
this https URL.

    

### [[2111.14746] Dynamic Inference](http://arxiv.org/abs/2111.14746)


  Traditional statistical estimation, or statistical inference in general, is
static, in the sense that the estimate of the quantity of interest does not
change the future evolution of the quantity. In some sequential estimation
problems however, we encounter the situation where the future values of the
quantity to be estimated depend on the estimate of its current value. Examples
include stock price prediction by big investors, interactive product
recommendation, and behavior prediction in multi-agent systems. We may call
such problems as dynamic inference. In this work, a formulation of this problem
under a Bayesian probabilistic framework is given, and the optimal estimation
strategy is derived as the solution to minimize the overall inference loss. How
the optimal estimation strategy works is illustrated through two examples,
stock trend prediction and vehicle behavior prediction. When the underlying
models for dynamic inference are unknown, we can consider the problem of
learning for dynamic inference. This learning problem can potentially unify
several familiar machine learning problems, including supervised learning,
imitation learning, and reinforcement learning.

    

### [[2106.00311] What's a good imputation to predict with missing values?](http://arxiv.org/abs/2106.00311)


  How to learn a good predictor on data with missing values? Most efforts focus
on first imputing as well as possible and second learning on the completed data
to predict the outcome. Yet, this widespread practice has no theoretical
grounding. Here we show that for almost all imputation functions, an
impute-then-regress procedure with a powerful learner is Bayes optimal. This
result holds for all missing-values mechanisms, in contrast with the classic
statistical results that require missing-at-random settings to use imputation
in probabilistic modeling. Moreover, it implies that perfect conditional
imputation is not needed for good prediction asymptotically. In fact, we show
that on perfectly imputed data the best regression function will generally be
discontinuous, which makes it hard to learn. Crafting instead the imputation so
as to leave the regression function unchanged simply shifts the problem to
learning discontinuous imputations. Rather, we suggest that it is easier to
learn imputation and regression jointly. We propose such a procedure, adapting
NeuMiss, a neural network capturing the conditional links across observed and
unobserved variables whatever the missing-value pattern. Experiments confirm
that joint imputation and regression through NeuMiss is better than various two
step procedures in our experiments with finite number of samples.

    

### [[2108.00670] Identify Light-Curve Signals with Deep Learning Based Object Detection Algorithm. I. Transit Detection](http://arxiv.org/abs/2108.00670)


  Deep learning techniques have been well explored in the transiting exoplanet
field; however, previous work mainly focuses on classification and inspection.
In this work, we develop a novel detection algorithm based on a well proven
object detection framework in the computer vision field. Through training the
network on the light curves of the confirmed Kepler exoplanets, our model
yields about 90% precision and recall for identifying transits with
signal-to-noise ratio higher than 6 (set the confidence threshold to 0.6).
Giving a slightly lower confidence threshold, recall can reach higher than 95%.
We also transfer the trained model to the TESS data and obtain similar
performance. The results of our algorithm match the intuition of the human
visual perception and make it useful to find single-transiting candidates.
Moreover, the parameters of the output bounding boxes can also help to find
multiplanet systems. Our network and detection functions are implemented in the
Deep-Transit toolkit, which is an open-source Python package hosted on GitHub
and PyPI.

    

### [[2111.15286] PERCIVAL: Open-Source Posit RISC-V Core with Quire Capability](http://arxiv.org/abs/2111.15286)


  The posit representation for real numbers is an alternative to the ubiquitous
IEEE 754 floating-point standard. In this work, we present PERCIVAL, an
application-level posit capable RISC-V core based on CVA6 that can execute all
posit instructions, including the quire fused operations. This solves the
obstacle encountered by previous works, which only included partial posit
support or which had to emulate posits in software, thus limiting the scope or
the scalability of their applications. In addition, Xposit, a RISC-V extension
for posit instructions is incorporated into LLVM. Therefore, PERCIVAL is the
first work that integrates the complete posit instruction set in hardware.
These elements allow for the native execution of posit instructions as well as
the standard floating-point ones, further permitting the comparison of these
representations. FPGA and ASIC synthesis show the hardware cost of implementing
32-bit posits and highlight the significant overhead of including a quire
accumulator. However, results comparing posits and IEEE floats show that the
quire enables a more accurate execution of dot products. In general matrix
multiplications, the accuracy error is reduced up to 4 orders of magnitude when
compared with single-precision floats. Furthermore, performance comparisons
show that these accuracy improvements do not hinder their execution, as posits
run as fast as single-precision floats and exhibit better timing than
double-precision floats, thus potentially providing an alternative
representation.

    

### [[2111.14946] Verifying Transactional Consistency of MongoDB](http://arxiv.org/abs/2111.14946)


  MongoDB is a popular general-purpose, document-oriented, distributed NoSQL
database. It supports transactions in three different deployments:
single-document transactions utilizing the WiredTiger storage engine in a
standalone node, multi-document transactions in a replica set which consists of
a primary node and several secondary nodes, and distributed transactions in a
sharded cluster which is a group of multiple replica sets, among which data is
sharded. A natural and fundamental question about MongoDB transactions is: What
transactional consistency guarantee do MongoDB Transactions in each deployment
provide? However, it lacks both concise pseudocode of MongoDB transactions in
each deployment and formal specification of the consistency guarantees which
MongoDB claimed to provide. In this work, we formally specify and verify the
transactional consistency protocols of MongoDB. Specifically, we provide a
concise pseudocode for the transactional consistency protocols in each MongoDB
deployment, namely WIREDTIGER, REPLICASET, and SHARDEDCLUSTER, based on the
official documents and source code. We then prove that WIREDTIGER, REPLICASET,
and SHARDEDCLUSTER satisfy different variants of snapshot isolation, namely
Strong-SI, Realtime-SI, and Session-SI, respectively. We also propose and
evaluate efficient white-box checking algorithms for MongoDB transaction
protocols against their consistency guarantees, effectively circumventing the
NP-hard obstacle in theory.

    

### [[2111.15071] Communication-Efficient Federated Learning via Quantized Compressed Sensing](http://arxiv.org/abs/2111.15071)


  In this paper, we present a communication-efficient federated learning
framework inspired by quantized compressed sensing. The presented framework
consists of gradient compression for wireless devices and gradient
reconstruction for a parameter server (PS). Our strategy for gradient
compression is to sequentially perform block sparsification, dimensional
reduction, and quantization. Thanks to gradient sparsification and
quantization, our strategy can achieve a higher compression ratio than one-bit
gradient compression. For accurate aggregation of the local gradients from the
compressed signals at the PS, we put forth an approximate minimum mean square
error (MMSE) approach for gradient reconstruction using the
expectation-maximization generalized-approximate-message-passing (EM-GAMP)
algorithm. Assuming Bernoulli Gaussian-mixture prior, this algorithm
iteratively updates the posterior mean and variance of local gradients from the
compressed signals. We also present a low-complexity approach for the gradient
reconstruction. In this approach, we use the Bussgang theorem to aggregate
local gradients from the compressed signals, then compute an approximate MMSE
estimate of the aggregated gradient using the EM-GAMP algorithm. We also
provide a convergence rate analysis of the presented framework. Using the MNIST
dataset, we demonstrate that the presented framework achieves almost identical
performance with the case that performs no compression, while significantly
reducing communication overhead for federated learning.

    

### [[2111.15130] Thermal entropy based hesitant fuzzy linguistic term set analysis in energy efficient opportunistic clustering](http://arxiv.org/abs/2111.15130)


  Limited energy resources and sensor nodes adaptability with the surrounding
environment play a significant role in the sustainable Wireless Sensor
Networks. This paper proposes a novel, dynamic, self-organizing opportunistic
clustering using Hesitant Fuzzy Linguistic Term Analysis-based Multi-Criteria
Decision Modeling methodology in order to overcome the CH decision making
problems and network lifetime bottlenecks. The asynchronous sleep/awake cycle
strategy could be exploited to make an opportunistic connection between sensor
nodes using opportunistic connection random graph. Every node in the network
observe the node gain degree, energy welfare, relative thermal entropy, link
connectivity, expected optimal hop, link quality factor etc. to form the
criteria for Hesitant Fuzzy Linguistic Term Set. It makes the node to evaluate
its current state and make the decision about the required action (CH, CM or
relay). The simulation results reveal that our proposed scheme leads to an
improvement in network lifetime, packet delivery ratio and overall energy
consumption against existing benchmarks.

    

### [[2111.15259] Privacy-Preserving Decentralized Exchange Marketplaces](http://arxiv.org/abs/2111.15259)


  Decentralized exchange markets leveraging blockchain have been proposed
recently to provide open and equal access to traders, improve transparency and
reduce systemic risk of centralized exchanges. However, they compromise on the
privacy of traders with respect to their asset ownership, account balance,
order details and their identity. In this paper, we present Rialto, a fully
decentralized privacy-preserving exchange marketplace with support for matching
trade orders, on-chain settlement and market price discovery. Rialto provides
confidentiality of order rates and account balances and unlinkability between
traders and their trade orders, while retaining the desirable properties of a
traditional marketplace like front-running resilience and market fairness. We
define formal security notions and present a security analysis of the
marketplace. We perform a detailed evaluation of our solution, demonstrate that
it scales well and is suitable for a large class of goods and financial
instruments traded in modern exchange markets.

    

### [[2111.15399] Evaluating Blockchain Application Requirements and their Satisfaction in Hyperledger Fabric](http://arxiv.org/abs/2111.15399)


  Blockchain applications may offer better fault-tolerance, integrity,
traceability and transparency compared to centralized solutions. Despite these
benefits, few businesses switch to blockchain-based applications. Industries
worry that the current blockchain implementations do not meet their
requirements, e.g., when it comes to scalability, throughput or latency.
Hyperledger Fabric (HLF) is a permissioned blockchain infrastructure that aims
to meet enterprise needs and provides a highly modular and well-conceived
architecture. In this paper, we survey and analyse requirements of blockchain
applications in respect to their underlying infrastructure by focusing mainly
on performance and resilience characteristics. Subsequently, we discuss to what
extent Fabric's current design allows it to meet these requirements. We further
evaluate the performance of Hyperledger Fabric 2.2 simulating different use
case scenarios by comparing single with multi ordering service performance and
conducting an evaluation with mixed workloads.

    

### [[2111.15471] Robust and Automated Method for Spike Detection and Removal in Magnetic Resonance Imaging](http://arxiv.org/abs/2111.15471)


  Radio frequency (RF) spike noise is a common source of exogenous image
corruption in MRI. Spikes occur as point-like disturbances of $k$-space that
lead to global sinusoidal intensity errors in the image domain. Depending on
the amplitude of the disturbances and their locations in $k$-space, the effect
of a spike can be significant, often ruining the reconstructed images. Here we
present both a spike detection method and a related data correction method for
automatic correction of RF spike noise. To detect spikes, we found the
$k$-space points that have the most significant effect on the total variation
of the image. To replace the spikes, we used a compressed sensing
reconstruction in which only the points thought to be corrupted are
unconstrained. We demonstrated our technique in two cases: (1) in vivo gradient
echo brain data with artificially corrupted points and (2) actual, complex
scanner data from a whole-body fat-water imaging gradient echo protocol
corrupted by spikes at uncertain locations. Our method allowed near-perfect
detection and correction with no human intervention. We calculated Matthews
correlation coefficients and sensitivities above 0.95 for a maximum of 0.78\%
corruption in synthetically corrupted in vivo brain data. We also found
specificities above 0.9994.

    

### [[2111.15480] Search by a Metamorphic Robotic System in a Finite 3D Cubic Grid](http://arxiv.org/abs/2111.15480)


  We consider search in a finite 3D cubic grid by a metamorphic robotic system
(MRS), that consists of anonymous modules. A module can perform a sliding and
rotation while the whole modules keep connectivity. As the number of modules
increases, the variety of actions that the MRS can perform increases. The
search problem requires the MRS to find a target in a given finite field. Doi
et al. (SSS 2018) demonstrate a necessary and sufficient number of modules for
search in a finite 2D square grid. We consider search in a finite 3D cubic grid
and investigate the effect of common knowledge. We consider three different
settings. First, we show that three modules are necessary and sufficient when
all modules are equipped with a common compass, i.e., they agree on the
direction and orientation of the $x$, $y$, and $z$ axes. Second, we show that
four modules are necessary and sufficient when all modules agree on the
direction and orientation of the vertical axis. Finally, we show that five
modules are necessary and sufficient when all modules are not equipped with a
common compass. Our results show that the shapes of the MRS in the 3D cubic
grid have richer structure than those in the 2D square grid.

    

### [[1705.04042] Robust Routing Made Easy](http://arxiv.org/abs/1705.04042)


  With the increasing scale of communication networks, the likelihood of
failures grows as well. Since these networks form a critical backbone of our
digital society, it is important that they rely on robust routing algorithms
which ensure connectivity despite such failures. While most modern
communication networks feature robust routing mechanisms, these mechanisms are
often fairly complex to design and verify, as they need to account for the
effects of failures and rerouting on communication.
This paper revisits the design of robust routing mechanisms, with the aim to
avoid such complexity. In particular, we showcase simple and generic blackbox
transformations that increase resilience of routing against independently
distributed failures, which allows to simulate the routing scheme on the
original network, even in the presence of non-benign node failures (henceforth
called faults). This is attractive as the system specification and routing
policy can simply be preserved.
We present a scheme for constructing such a reinforced network, given an
existing (synchronous) network and a routing scheme. We prove that this
algorithm comes with small constant overheads, and only requires a minimal
amount of additional node and edge resources. At the same time, it allows to
tolerate a large number of independent random (node) faults, asymptotically
almost surely. We complement our analytical results with simulations on
different real-world topologies.

    

### [[2103.10366] Fast Consensus via the Unconstrained Undecided State Dynamics](http://arxiv.org/abs/2103.10366)


  We consider the plurality consensus problem among $n$ agents. Initially, each
agent has one of $k$ different opinions. Agents choose random interaction
partners and revise their state according to a fixed transition function,
depending on their own state and the state of the interaction partners. The
goal is to reach a consensus configuration in which all agents agree on the
same opinion, and if there is initially a sufficiently large bias towards one
opinion, that opinion should prevail.
We analyze a synchronized variant of the undecided state dynamics defined as
follows. The agents act in phases, consisting of a decision and a boosting
part. In the decision part, any agent that encounters an agent with a different
opinion becomes undecided. In the boosting part, undecided agents adopt the
first opinion they encounter.
We consider this dynamics in the population model and the gossip model. For
the population model, our protocol reaches consensus (w.h.p.) in $O(\log^2 n)$
parallel time, providing the first polylogarithmic result for $k > 2$ (w.h.p.)
in this model. Without any assumption on the bias, fast consensus has only been
shown for $k = 2$ for the unsynchronized version of the undecided state
dynamics [Clementi et al., MFCS'18]. We show that the synchronized variant of
the undecided state dynamics reaches consensus (w.h.p.) in time $O(\log^2 n)$,
independently of the initial number, bias, or distribution of opinions. In both
models, we guarantee that if there is an initial bias of $\Omega(\sqrt{n \log
n})$, then (w.h.p.) that opinion wins.
A simple extension of our protocol in the gossip model yields a dynamics that
does not depend on $n$ or $k$, is anonymous, and has (w.h.p.) runtime $O(\log^2
n)$. This solves an open problem formulated by Becchetti et al.~[Distributed
Computing,~2017].

    

### [[2111.15020] US-Rule: Discovering Utility-driven Sequential Rules](http://arxiv.org/abs/2111.15020)


  Utility-driven mining is an important task in data science and has many
applications in real life. High utility sequential pattern mining (HUSPM) is
one kind of utility-driven mining. HUSPM aims to discover all sequential
patterns with high utility. However, the existing algorithms of HUSPM can not
provide an accurate probability to deal with some scenarios for prediction or
recommendation. High-utility sequential rule mining (HUSRM) was proposed to
discover all sequential rules with high utility and high confidence. There is
only one algorithm proposed for HUSRM, which is not enough efficient. In this
paper, we propose a faster algorithm, called US-Rule, to efficiently mine
high-utility sequential rules. It utilizes rule estimated utility co-occurrence
pruning strategy (REUCP) to avoid meaningless computation. To improve the
efficiency on dense and long sequence datasets, four tighter upper bounds
(LEEU, REEU, LERSU, RERSU) and their corresponding pruning strategies (LEEUP,
REEUP, LERSUP, RERSUP) are proposed. Besides, US-Rule proposes rule estimated
utility recomputing pruning strategy (REURP) to deal with sparse datasets. At
last, a large number of experiments on different datasets compared to the
state-of-the-art algorithm demonstrate that US-Rule can achieve better
performance in terms of execution time, memory consumption and scalability.

    

### [[2111.15026] Anomaly Rule Detection in Sequence Data](http://arxiv.org/abs/2111.15026)


  Analyzing sequence data usually leads to the discovery of interesting
patterns and then anomaly detection. In recent years, numerous frameworks and
methods have been proposed to discover interesting patterns in sequence data as
well as detect anomalous behavior. However, existing algorithms mainly focus on
frequency-driven analytic, and they are challenging to be applied in real-world
settings. In this work, we present a new anomaly detection framework called
DUOS that enables Discovery of Utility-aware Outlier Sequential rules from a
set of sequences. In this pattern-based anomaly detection algorithm, we
incorporate both the anomalousness and utility of a group, and then introduce
the concept of utility-aware outlier sequential rule (UOSR). We show that this
is a more meaningful way for detecting anomalies. Besides, we propose some
efficient pruning strategies w.r.t. upper bounds for mining UOSR, as well as
the outlier detection. An extensive experimental study conducted on several
real-world datasets shows that the proposed DUOS algorithm has a better
effectiveness and efficiency. Finally, DUOS outperforms the baseline algorithm
and has a suitable scalability.

    

### [[2111.15040] X-ray Dissectography Enables Stereotography to Improve Diagnostic Performance](http://arxiv.org/abs/2111.15040)


  X-ray imaging is the most popular medical imaging technology. While x-ray
radiography is rather cost-effective, tissue structures are superimposed along
the x-ray paths. On the other hand, computed tomography (CT) reconstructs
internal structures but CT increases radiation dose, is complicated and
expensive. Here we propose "x-ray dissectography" to extract a target
organ/tissue digitally from few radiographic projections for stereographic and
tomographic analysis in the deep learning framework. As an exemplary
embodiment, we propose a general X-ray dissectography network, a dedicated
X-ray stereotography network, and the X-ray imaging systems to implement these
functionalities. Our experiments show that x-ray stereography can be achieved
of an isolated organ such as the lungs in this case, suggesting the feasibility
of transforming conventional radiographic reading to the stereographic
examination of the isolated organ, which potentially allows higher sensitivity
and specificity, and even tomographic visualization of the target. With further
improvements, x-ray dissectography promises to be a new x-ray imaging modality
for CT-grade diagnosis at radiation dose and system cost comparable to that of
radiographic or tomosynthetic imaging.

    

### [[2111.15108] Interval-valued q-Rung Orthopair Fuzzy Choquet Integral Operators and Its Application in Group Decision Making](http://arxiv.org/abs/2111.15108)


  It is more flexible for decision makers to evaluate by interval-valued q-rung
orthopair fuzzy set (IVq-ROFS),which offers fuzzy decision-making more
applicational space. Meanwhile, Choquet integralses non-additive set function
(fuzzy measure) to describe the interaction between attributes this http URL
particular, there are a large number of practical issues that have relevance
between attributes.Therefore,this paper proposes the correlation operator and
group decision-making method based on the interval-valued q-rung orthopair
fuzzy set Choquet integral.First,interval-valued q-rung orthopair fuzzy Choquet
integral average operator (IVq-ROFCA) and interval-valued q-rung orthopair
fuzzy Choquet integral geometric operator (IVq-ROFCG) are inves-tigated,and
their basic properties are proved.Furthermore, several operators based on
IVq-ROFCA and IVq-ROFCG are developed. Then, a group decision-making method
based on IVq-ROFCA is developed,which can solve the decision making problems
with interaction between attributes.Finally,through the implementation of the
warning management system for hypertension,it is shown that the operator and
group decision-making method proposed in this paper can handle complex
decision-making cases in reality, and the decision result is consistent with
the doctor's diagnosis result.Moreover,the comparison with the results of other
operators shows that the proposed operators and group decision-making method
are correct and effective,and the decision result will not be affected by the
change of q value.

    

### [[2111.15119] Aerial Images Meet Crowdsourced Trajectories: A New Approach to Robust Road Extraction](http://arxiv.org/abs/2111.15119)


  Land remote sensing analysis is a crucial research in earth science. In this
work, we focus on a challenging task of land analysis, i.e., automatic
extraction of traffic roads from remote sensing data, which has widespread
applications in urban development and expansion estimation. Nevertheless,
conventional methods either only utilized the limited information of aerial
images, or simply fused multimodal information (e.g., vehicle trajectories),
thus cannot well recognize unconstrained roads. To facilitate this problem, we
introduce a novel neural network framework termed Cross-Modal Message
Propagation Network (CMMPNet), which fully benefits the complementary different
modal data (i.e., aerial images and crowdsourced trajectories). Specifically,
CMMPNet is composed of two deep Auto-Encoders for modality-specific
representation learning and a tailor-designed Dual Enhancement Module for
cross-modal representation refinement. In particular, the complementary
information of each modality is comprehensively extracted and dynamically
propagated to enhance the representation of another modality. Extensive
experiments on three real-world benchmarks demonstrate the effectiveness of our
CMMPNet for robust road extraction benefiting from blending different modal
data, either using image and trajectory data or image and Lidar data. From the
experimental results, we observe that the proposed approach outperforms current
state-of-the-art methods by large margins.

    

### [[2111.15185] SamplingAug: On the Importance of Patch Sampling Augmentation for Single Image Super-Resolution](http://arxiv.org/abs/2111.15185)


  With the development of Deep Neural Networks (DNNs), plenty of methods based
on DNNs have been proposed for Single Image Super-Resolution (SISR). However,
existing methods mostly train the DNNs on uniformly sampled LR-HR patch pairs,
which makes them fail to fully exploit informative patches within the image. In
this paper, we present a simple yet effective data augmentation method. We
first devise a heuristic metric to evaluate the informative importance of each
patch pair. In order to reduce the computational cost for all patch pairs, we
further propose to optimize the calculation of our metric by integral image,
achieving about two orders of magnitude speedup. The training patch pairs are
sampled according to their informative importance with our method. Extensive
experiments show our sampling augmentation can consistently improve the
convergence and boost the performance of various SISR architectures, including
EDSR, RCAN, RDN, SRCNN and ESPCN across different scaling factors (x2, x3, x4).
Code is available at this https URL


### [[2111.15208] HRNET: AI on Edge for mask detection and social distancing](http://arxiv.org/abs/2111.15208)


  The purpose of the paper is to provide innovative emerging technology
framework for community to combat epidemic situations. The paper proposes a
unique outbreak response system framework based on artificial intelligence and
edge computing for citizen centric services to help track and trace people
eluding safety policies like mask detection and social distancing measure in
public or workplace setup. The framework further provides implementation
guideline in industrial setup as well for governance and contact tracing tasks.
The adoption will thus lead in smart city planning and development focusing on
citizen health systems contributing to improved quality of life. The conceptual
framework presented is validated through quantitative data analysis via
secondary data collection from researcher's public websites, GitHub
repositories and renowned journals and further benchmarking were conducted for
experimental results in Microsoft Azure cloud environment. The study includes
selective AI-models for benchmark analysis and were assessed on performance and
accuracy in edge computing environment for large scale societal setup. Overall
YOLO model Outperforms in object detection task and is faster enough for mask
detection and HRNetV2 outperform semantic segmentation problem applied to solve
social distancing task in AI-Edge inferencing environmental setup. The paper
proposes new Edge-AI algorithm for building technology-oriented solutions for
detecting mask in human movement and social distance. The paper enriches the
technological advancement in artificial intelligence and edge-computing applied
to problems in society and healthcare systems. The framework further equips
government agency, system providers to design and constructs
technology-oriented models in community setup to Increase the quality of life
using emerging technologies into smart urban environments.

    

### [[2111.15210] Point Cloud Instance Segmentation with Semi-supervised Bounding-Box Mining](http://arxiv.org/abs/2111.15210)


  Point cloud instance segmentation has achieved huge progress with the
emergence of deep learning. However, these methods are usually data-hungry with
expensive and time-consuming dense point cloud annotations. To alleviate the
annotation cost, unlabeled or weakly labeled data is still less explored in the
task. In this paper, we introduce the first semi-supervised point cloud
instance segmentation framework (SPIB) using both labeled and unlabelled
bounding boxes as supervision. To be specific, our SPIB architecture involves a
two-stage learning procedure. For stage one, a bounding box proposal generation
network is trained under a semi-supervised setting with perturbation
consistency regularization (SPCR). The regularization works by enforcing an
invariance of the bounding box predictions over different perturbations applied
to the input point clouds, to provide self-supervision for network learning.
For stage two, the bounding box proposals with SPCR are grouped into some
subsets, and the instance masks are mined inside each subset with a novel
semantic propagation module and a property consistency graph module. Moreover,
we introduce a novel occupancy ratio guided refinement module to refine the
instance masks. Extensive experiments on the challenging ScanNet v2 dataset
demonstrate our method can achieve competitive performance compared with the
recent fully-supervised methods.

    

### [[2111.15246] Hallucinated Neural Radiance Fields in the Wild](http://arxiv.org/abs/2111.15246)


  Neural Radiance Fields (NeRF) has recently gained popularity for its
impressive novel view synthesis ability. This paper studies the problem of
hallucinated NeRF: i.e. recovering a realistic NeRF at a different time of day
from a group of tourism images. Existing solutions adopt NeRF with a
controllable appearance embedding to render novel views under various
conditions, but cannot render view-consistent images with an unseen appearance.
To solve this problem, we present an end-to-end framework for constructing a
hallucinated NeRF, dubbed as H-NeRF. Specifically, we propose an appearance
hallucination module to handle time-varying appearances and transfer them to
novel views. Considering the complex occlusions of tourism images, an
anti-occlusion module is introduced to decompose the static subjects for
visibility accurately. Experimental results on synthetic data and real tourism
photo collections demonstrate that our method can not only hallucinate the
desired appearances, but also render occlusion-free images from different
views. The project and supplementary materials are available at
this https URL.

    

### [[2111.15255] Double Fuzzy Probabilistic Interval Linguistic Term Set and a Dynamic Fuzzy Decision Making Model based on Markov Process with tts Application in Multiple Criteria Group Decision Making](http://arxiv.org/abs/2111.15255)


  The probabilistic linguistic term has been proposed to deal with probability
distributions in provided linguistic evaluations. However, because it has some
fundamental defects, it is often difficult for decision-makers to get
reasonable information of linguistic evaluations for group decision making. In
addition, weight information plays a significant role in dynamic information
fusion and decision making process. However, there are few research methods to
determine the dynamic attribute weight with time. In this paper, I propose the
concept of double fuzzy probability interval linguistic term set (DFPILTS).
Firstly, fuzzy semantic integration, DFPILTS definition, its preference
relationship, some basic algorithms and aggregation operators are defined.
Then, a fuzzy linguistic Markov matrix with its network is developed. Then, a
weight determination method based on distance measure and information entropy
to reducing the inconsistency of DFPILPR and obtain collective priority vector
based on group consensus is developed. Finally, an aggregation-based approach
is developed, and an optimal investment case from a financial risk is used to
illustrate the application of DFPILTS and decision method in multi-criteria
decision making.

    

### [[2111.15257] ARTSeg: Employing Attention for Thermal images Semantic Segmentation](http://arxiv.org/abs/2111.15257)


  The research advancements have made the neural network algorithms deployed in
the autonomous vehicle to perceive the surrounding. The standard exteroceptive
sensors that are utilized for the perception of the environment are cameras and
Lidar. Therefore, the neural network algorithms developed using these
exteroceptive sensors have provided the necessary solution for the autonomous
vehicle's perception. One major drawback of these exteroceptive sensors is
their operability in adverse weather conditions, for instance, low illumination
and night conditions. The useability and affordability of thermal cameras in
the sensor suite of the autonomous vehicle provide the necessary improvement in
the autonomous vehicle's perception in adverse weather conditions. The
semantics of the environment benefits the robust perception, which can be
achieved by segmenting different objects in the scene. In this work, we have
employed the thermal camera for semantic segmentation. We have designed an
attention-based Recurrent Convolution Network (RCNN) encoder-decoder
architecture named ARTSeg for thermal semantic segmentation. The main
contribution of this work is the design of encoder-decoder architecture, which
employ units of RCNN for each encoder and decoder block. Furthermore, additive
attention is employed in the decoder module to retain high-resolution features
and improve the localization of features. The efficacy of the proposed method
is evaluated on the available public dataset, showing better performance with
other state-of-the-art methods in mean intersection over union (IoU).

    

### [[2111.15275] Emotions as abstract evaluation criteria in biological and artificial intelligences](http://arxiv.org/abs/2111.15275)


  Biological as well as advanced artificial intelligences (AIs) need to decide
which goals to pursue. We review nature's solution to the time allocation
problem, which is based on a continuously readjusted categorical weighting
mechanism we experience introspectively as emotions. One observes
phylogenetically that the available number of emotional states increases hand
in hand with the cognitive capabilities of animals and that raising levels of
intelligence entail ever larger sets of behavioral options. Our ability to
experience a multitude of potentially conflicting feelings is in this view not
a leftover of a more primitive heritage, but a generic mechanism for
attributing values to behavioral options that can not be specified at birth. In
this view, emotions are essential for understanding the mind.
For concreteness, we propose and discuss a framework which mimics emotions on
a functional level. Based on time allocation via emotional stationarity (TAES),
emotions are implemented as abstract criteria, such as satisfaction, challenge
and boredom, which serve to evaluate activities that have been carried out. The
resulting timeline of experienced emotions is compared with the `character' of
the agent, which is defined in terms of a preferred distribution of emotional
states. The long-term goal of the agent, to align experience with character, is
achieved by optimizing the frequency for selecting individual tasks. Upon
optimization, the statistics of emotion experience becomes stationary.

    

### [[2111.15323] The signature and cusp geometry of hyperbolic knots](http://arxiv.org/abs/2111.15323)


  We introduce a new real-valued invariant called the natural slope of a
hyperbolic knot in the 3-sphere, which is defined in terms of its cusp
geometry. We show that twice the knot signature and the natural slope differ by
at most a constant times the hyperbolic volume divided by the cube of the
injectivity radius. This inequality was discovered using machine learning to
detect relationships between various knot invariants. It has applications to
Dehn surgery and to 4-ball genus. We also show a refined version of the
inequality where the upper bound is a linear function of the volume, and the
slope is corrected by terms corresponding to short geodesics that link the knot
an odd number of times.

    

### [[2111.15361] Seeking Salient Facial Regions for Cross-Database Micro-Expression Recognition](http://arxiv.org/abs/2111.15361)


  This paper focuses on the research of cross-database micro-expression
recognition, in which the training and test micro-expression samples belong to
different microexpression databases. Mismatched feature distributions between
the training and testing micro-expression feature degrade the performance of
most well-performing micro-expression methods. To deal with cross-database
micro-expression recognition, we propose a novel domain adaption method called
Transfer Group Sparse Regression (TGSR). TGSR learns a sparse regression matrix
for selecting salient facial local regions and the corresponding relationship
of the training set and test set. We evaluate our TGSR model in CASME II and
SMIC databases. Experimental results show that the proposed TGSR achieves
satisfactory performance and outperforms most state-of-the-art subspace
learning-based domain adaption methods.

    

### [[2111.15445] Asymptotics for Pull on the Complete Graph](http://arxiv.org/abs/2111.15445)


  Consider the following model to study adversarial effects on opinion forming.
A set of initially selected experts form their binary opinion while being
influenced by an adversary, who may convince some of them of the falsehood. All
other participants in the network then take the opinion of the majority of
their neighbouring experts. Can the adversary influence the experts in such a
way that the majority of the network believes the falsehood? Alon et al. [1]
conjectured that in this context an iterative dissemination process will always
be beneficial to the adversary. This work provides a counterexample to that
conjecture.
[1] N. Alon, M. Feldman, O. Lev, and M. Tennenholtz. How Robust Is the Wisdom
of the Crowds? In Proceedings of the 24th International Joint Conference on
Artificial Intelligence (IJCAI 2015), pages 2055-2061, 2015.

    

### [[2111.15446] TEGDetector: A Phishing Detector that Knows Evolving Transaction Behaviors](http://arxiv.org/abs/2111.15446)


  Recently, phishing scams have posed a significant threat to blockchains.
Phishing detectors direct their efforts in hunting phishing addresses. Most of
the detectors extract target addresses' transaction behavior features by random
walking or constructing static subgraphs. The random walking
methods,unfortunately, usually miss structural information due to limited
sampling sequence length, while the static subgraph methods tend to ignore
temporal features lying in the evolving transaction behaviors. More
importantly, their performance undergoes severe degradation when the malicious
users intentionally hide phishing behaviors. To address these challenges, we
propose TEGDetector, a dynamic graph classifier that learns the evolving
behavior features from transaction evolution graphs (TEGs). First, we cast the
transaction series into multiple time slices, capturing the target address's
transaction behaviors in different periods. Then, we provide a fast
non-parametric phishing detector to narrow down the search space of suspicious
addresses. Finally, TEGDetector considers both the spatial and temporal
evolutions towards a complete characterization of the evolving transaction
behaviors. Moreover, TEGDetector utilizes adaptively learnt time coefficient to
pay distinct attention to different periods, which provides several novel
insights. Extensive experiments on the large-scale Ethereum transaction dataset
demonstrate that the proposed method achieves state-of-the-art detection
performance.

    

### [[2111.15626] Variational Autoencoders for Studying the Manifold of Precoding Matrices with High Spectral Efficiency](http://arxiv.org/abs/2111.15626)


  In multiple-input multiple-output (MIMO) wireless communications systems,
neural networks have been employed for channel decoding, detection, channel
estimation, and resource management. In this paper, we look at how to use a
variational autoencoder to find a precoding matrix with a high Spectral
Efficiency (SE). To identify efficient precoding matrices, an optimization
approach is used. Our objective is to create a less time-consuming algorithm
with minimum quality degradation. To build precoding matrices, we employed two
forms of variational autoencoders: conventional variational autoencoders (VAE)
and conditional variational autoencoders (CVAE). Both methods may be used to
study a wide range of optimal precoding matrices. To the best of our knowledge,
the development of precoding matrices for the spectral efficiency objective
function (SE) utilising VAE and CVAE methods is being published for the first
time.

    

### [[2111.15636] Generating gapless land surface temperature with a high spatio-temporal resolution by fusing multi-source satellite-observed and model-simulated data](http://arxiv.org/abs/2111.15636)


  Land surface temperature (LST) is a key parameter when monitoring land
surface processes. However, cloud contamination and the tradeoff between the
spatial and temporal resolutions greatly impede the access to high-quality
thermal infrared (TIR) remote sensing data. Despite the massive efforts made to
solve these dilemmas, it is still difficult to generate LST estimates with
concurrent spatial completeness and a high spatio-temporal resolution. Land
surface models (LSMs) can be used to simulate gapless LST with a high temporal
resolution, but this usually comes with a low spatial resolution. In this
paper, we present an integrated temperature fusion framework for
satellite-observed and LSM-simulated LST data to map gapless LST at a 60-m
spatial resolution and half-hourly temporal resolution. The global linear model
(GloLM) model and the diurnal land surface temperature cycle (DTC) model are
respectively performed as preprocessing steps for sensor and temporal
normalization between the different LST data. The Landsat LST, Moderate
Resolution Imaging Spectroradiometer (MODIS) LST, and Community Land Model
Version 5.0 (CLM 5.0)-simulated LST are then fused using a filter-based
spatio-temporal integrated fusion model. Evaluations were implemented in an
urban-dominated region (the city of Wuhan in China) and a natural-dominated
region (the Heihe River Basin in China), in terms of accuracy, spatial
variability, and diurnal temporal dynamics. Results indicate that the fused LST
is highly consistent with actual Landsat LST data (in situ LST measurements),
in terms of a Pearson correlation coefficient of 0.94 (0.97-0.99), a mean
absolute error of 0.71-0.98 K (0.82-3.17 K), and a root-mean-square error of
0.97-1.26 K (1.09-3.97 K).

    

### [[2009.07734] TreeGAN: Incorporating Class Hierarchy into Image Generation](http://arxiv.org/abs/2009.07734)


  Conditional image generation (CIG) is a widely studied problem in computer
vision and machine learning. Given a class, CIG takes the name of this class as
input and generates a set of images that belong to this class. In existing CIG
works, for different classes, their corresponding images are generated
independently, without considering the relationship among classes. In
real-world applications, the classes are organized into a hierarchy and their
hierarchical relationships are informative for generating high-fidelity images.
In this paper, we aim to leverage the class hierarchy for conditional image
generation. We propose two ways of incorporating class hierarchy: prior control
and post constraint. In prior control, we first encode the class hierarchy,
then feed it as a prior into the conditional generator to generate images. In
post constraint, after the images are generated, we measure their consistency
with the class hierarchy and use the consistency score to guide the training of
the generator. Based on these two ideas, we propose a TreeGAN model which
consists of three modules: (1) a class hierarchy encoder (CHE) which takes the
hierarchical structure of classes and their textual names as inputs and learns
an embedding for each class; the embedding captures the hierarchical
relationship among classes; (2) a conditional image generator (CIG) which takes
the CHE-generated embedding of a class as input and generates a set of images
belonging to this class; (3) a consistency checker which performs hierarchical
classification on the generated images and checks whether the generated images
are compatible with the class hierarchy; the consistency score is used to guide
the CIG to generate hierarchy-compatible images. Experiments on various
datasets demonstrate the effectiveness of our method.

    

### [[2101.03805] A Conflict-Based Search Framework for Multi-Objective Multi-Agent Path Finding](http://arxiv.org/abs/2101.03805)


  Conventional multi-agent path planners typically compute an ensemble of paths
while optimizing a single objective, such as path length. However, many
applications may require multiple objectives, say fuel consumption and
completion time, to be simultaneously optimized during planning and these
criteria may not be readily compared and sometimes lie in competition with each
other. Naively applying existing multi-objective search algorithms, such as
multi-objective A* (MOA*), to multi-agent path finding may prove to be
inefficient as the size of the space of possible solutions, i.e., the
Pareto-optimal set, can grow exponentially with the number of agents (the
dimension of the search space). This article presents an approach named
Multi-Objective Conflict-Based Search (MO-CBS) that bypasses this so-called
curse of dimensionality by leveraging prior Conflict-Based Search (CBS), a
well-known algorithm for single-objective multi-agent path finding, and
principles of dominance from multi-objective optimization literature. We also
develop several variants of MO-CBS to further improve its performance. We prove
that MO-CBS and its variants are able to compute the entire Pareto-optimal set.
Numerical results show that MO-CBS outperforms both MOA* as well as MOM*, a
recently developed state-of-the-art multi-objective multi-agent planner.

    

### [[2104.06744] Defending Against Adversarial Denial-of-Service Data Poisoning Attacks](http://arxiv.org/abs/2104.06744)


  Data poisoning is one of the most relevant security threats against machine
learning and data-driven technologies. Since many applications rely on
untrusted training data, an attacker can easily craft malicious samples and
inject them into the training dataset to degrade the performance of machine
learning models. As recent work has shown, such Denial-of-Service (DoS) data
poisoning attacks are highly effective. To mitigate this threat, we propose a
new approach of detecting DoS poisoned instances. In comparison to related
work, we deviate from clustering and anomaly detection based approaches, which
often suffer from the curse of dimensionality and arbitrary anomaly threshold
selection. Rather, our defence is based on extracting information from the
training data in such a generalized manner that we can identify poisoned
samples based on the information present in the unpoisoned portion of the data.
We evaluate our defence against two DoS poisoning attacks and seven datasets,
and find that it reliably identifies poisoned instances. In comparison to
related work, our defence improves false positive / false negative rates by at
least 50%, often more.

    

### [[2111.14562] Instance-wise Occlusion and Depth Orders in Natural Scenes](http://arxiv.org/abs/2111.14562)


  In this paper, we introduce a new dataset, named InstaOrder, that can be used
to understand the spatial relationships of instances in a 3D space. The dataset
consists of 2.9M annotations of geometric orderings for class-labeled instances
in 101K natural scenes. The scenes were annotated by 3,659 crowd-workers
regarding (1) occlusion order that identifies occluder/occludee and (2) depth
order that describes ordinal relations that consider relative distance from the
camera. The dataset provides joint annotation of two kinds of orderings for the
same instances, and we discover that the occlusion order and depth order are
complementary. We also introduce a geometric order prediction network called
InstaOrderNet, which is superior to state-of-the-art approaches. Moreover, we
propose InstaDepthNet that uses auxiliary geometric order loss to boost the
instance-wise depth prediction accuracy of MiDaS. These contributions to
geometric scene understanding will help to improve the accuracy of various
computer vision tasks.

    

### [[2111.14799] UBoCo : Unsupervised Boundary Contrastive Learning for Generic Event Boundary Detection](http://arxiv.org/abs/2111.14799)


  Generic Event Boundary Detection (GEBD) is a newly suggested video
understanding task that aims to find one level deeper semantic boundaries of
events. Bridging the gap between natural human perception and video
understanding, it has various potential applications, including interpretable
and semantically valid video parsing. Still at an early development stage,
existing GEBD solvers are simple extensions of relevant video understanding
tasks, disregarding GEBD's distinctive characteristics. In this paper, we
propose a novel framework for unsupervised/supervised GEBD, by using the
Temporal Self-similarity Matrix (TSM) as the video representation. The new
Recursive TSM Parsing (RTP) algorithm exploits local diagonal patterns in TSM
to detect boundaries, and it is combined with the Boundary Contrastive (BoCo)
loss to train our encoder to generate more informative TSMs. Our framework can
be applied to both unsupervised and supervised settings, with both achieving
state-of-the-art performance by a huge margin in GEBD benchmark. Especially,
our unsupervised method outperforms the previous state-of-the-art "supervised"
model, implying its exceptional efficacy.

    

### [[2111.14917] A Separation Logic for Negative Dependence](http://arxiv.org/abs/2111.14917)


  Formal reasoning about hashing-based probabilistic data structures often
requires reasoning about random variables where when one variable gets larger
(such as the number of elements hashed into one bucket), the others tend to be
smaller (like the number of elements hashed into the other buckets). This is an
example of negative dependence, a generalization of probabilistic independence
that has recently found interesting applications in algorithm design and
machine learning. Despite the usefulness of negative dependence for the
analyses of probabilistic data structures, existing verification methods cannot
establish this property for randomized programs.
To fill this gap, we design LINA, a probabilistic separation logic for
reasoning about negative dependence. Following recent works on probabilistic
separation logic using separating conjunction to reason about the probabilistic
independence of random variables, we use separating conjunction to reason about
negative dependence. Our assertion logic features two separating conjunctions,
one for independence and one for negative dependence. We generalize the logic
of bunched implications (BI) to support multiple separating conjunctions, and
provide a sound and complete proof system. Notably, the semantics for
separating conjunction relies on a non-deterministic, rather than partial,
operation for combining resources. By drawing on closure properties for
negative dependence, our program logic supports a Frame-like rule for negative
dependence and monotone operations. We demonstrate how LINA can verify
probabilistic properties of hash-based data structures and balls-into-bins
processes.

    

### [[2111.14947] An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs](http://arxiv.org/abs/2111.14947)


  While loop reordering and fusion can make big impacts on the constant-factor
performance of dense tensor programs, the effects on sparse tensor programs are
asymptotic, often leading to orders of magnitude performance differences in
practice. Sparse tensors also introduce a choice of compressed storage formats
that can have asymptotic effects. Research into sparse tensor compilers has led
to simplified languages that express these tradeoffs, but the user is expected
to provide a schedule that makes the decisions. This is challenging because
schedulers must anticipate the interaction between sparse formats, loop
structure, potential sparsity patterns, and the compiler itself. Automating
this decision making process stands to finally make sparse tensor compilers
accessible to end users.
We present, to the best of our knowledge, the first automatic asymptotic
scheduler for sparse tensor programs. We provide an approach to abstractly
represent the asymptotic cost of schedules and to choose between them. We
narrow down the search space to a manageably small "Pareto frontier" of
asymptotically undominated kernels. We test our approach by compiling these
kernels with the TACO sparse tensor compiler and comparing them with those
generated with the default TACO schedules. Our results show that our approach
reduces the scheduling space by orders of magnitude and that the generated
kernels perform asymptotically better than those generated using the default
schedules.

    

### [[2111.15149] SteelCore: An Extensible Concurrent Separation Logic for Effectful Dependently Typed Programs](http://arxiv.org/abs/2111.15149)


  Much recent research has been devoted to modeling effects within type theory.
Building on this work, we observe that effectful type theories can provide a
foundation on which to build semantics for more complex programming constructs
and program logics, extending the reasoning principles that apply within the
host effectful type theory itself. Concretely, our main contribution is a
semantics for concurrent separation logic (CSL) within the F* proof assistant
in a manner that enables dependently typed, effectful F* programs to make use
of concurrency and to be specified and verified using a full-featured,
extensible CSL. In contrast to prior approaches, we directly derive the
partial-correctness Hoare rules for CSL from the denotation of computations in
the effectful semantics of non-deterministically interleaved atomic actions.
Demonstrating the flexibility of our semantics, we build generic, verified
libraries that support various concurrency constructs, ranging from dynamically
allocated, storable spin locks, to protocol-indexed channels. We conclude that
our effectful semantics provides a simple yet expressive basis on which to
layer domain-specific languages and logics for verified, concurrent
programming.

    

### [[2111.15456] Towards Denotational Semantics of AD for Higher-Order, Recursive, Probabilistic Languages](http://arxiv.org/abs/2111.15456)


  Automatic differentiation (AD) aims to compute derivatives of user-defined
functions, but in Turing-complete languages, this simple specification does not
fully capture AD's behavior: AD sometimes disagrees with the true derivative of
a differentiable program, and when AD is applied to non-differentiable or
effectful programs, it is unclear what guarantees (if any) hold of the
resulting code. We study an expressive differentiable programming language,
with piecewise-analytic primitives, higher-order functions, and general
recursion. Our main result is that even in this general setting, a version of
Lee et al. [2020]'s correctness theorem (originally proven for a first-order
language without partiality or recursion) holds: all programs denote so-called
$\omega$PAP functions, and AD computes correct intensional derivatives of them.
Mazza and Pagani [2021]'s recent theorem, that AD disagrees with the true
derivative of a differentiable recursive program at a measure-zero set of
inputs, can be derived as a straightforward corollary of this fact. We also
apply the framework to study probabilistic programs, and recover a recent
result from Mak et al. [2021] via a novel denotational argument.

    

### [[1901.05750] TaDA Live: Compositional Reasoning for Termination of Fine-grained Concurrent Programs](http://arxiv.org/abs/1901.05750)


  We present TaDA Live, a concurrent separation logic for reasoning
compositionally about the termination of blocking fine-grained concurrent
programs. The crucial challenge is how to deal with abstract atomic blocking:
that is, abstract atomic operations that have blocking behaviour arising from
busy-waiting patterns as found in, for example, fine-grained spin locks. Our
fundamental innovation is with the design of abstract specifications that
capture this blocking behaviour as liveness assumptions on the environment. We
design a logic that can reason about the termination of clients which use such
operations without breaking their abstraction boundaries, and the correctness
of the implementations of the operations with respect to their abstract
specifications. We introduce a novel semantic model using layered subjective
obligations to express liveness invariants, and a proof system that is sound
with respect to the model. The subtlety of our specifications and reasoning is
illustrated using several case studies.

    

### [[2101.09038] A Decentralized Analysis of Multiparty Protocols](http://arxiv.org/abs/2101.09038)


  Protocols provide the unifying glue in concurrent and distributed software
today; verifying that message-passing programs conform to such governing
protocols is important but difficult. Static approaches based on multiparty
session types (MPST) use protocols as types to avoid protocol violations and
deadlocks in programs. An elusive problem for MPST is to ensure both protocol
conformance and deadlock freedom for implementations with interleaved and
delegated protocols.
We propose a decentralized analysis of multiparty protocols, specified as
global types and implemented as interacting processes in an asynchronous
$\pi$-calculus. Our solution rests upon two novel notions: router processes and
relative types. While router processes use the global type to enable the
composition of participant implementations in arbitrary process networks,
relative types extract from the global type the intended interactions and
dependencies between pairs of participants. In our analysis, processes are
typed using APCP, a type system that ensures protocol conformance and deadlock
freedom with respect to binary protocols, developed in prior work. Our
decentralized, router-based analysis enables the sound and complete
transference of protocol conformance and deadlock freedom from APCP to
multiparty protocols.

    