
## 2021-11-29

### [[2111.12851] Coverage Analysis for Satellite Downlink Networks](http://arxiv.org/abs/2111.12851)


  Satellite networks are promising to provide ubiquitous and high-capacity
global wireless connectivity. Traditionally, satellite networks are modeled by
placing satellites on a grid of multiple circular orbit geometries. Such a
network model, however, requires intricate system-level simulations to evaluate
coverage performance, and analytical understanding of the satellite network is
limited. Continuing the success of stochastic geometry in a tractable analysis
for terrestrial networks, in this paper, we develop novel models that are
tractable for the coverage analysis of satellite networks using stochastic
geometry. By modeling the locations of satellites and users using Poisson point
processes on the surfaces of concentric spheres, we characterize analytical
expressions for the coverage probability of a typical downlink user as a
function of relevant parameters, including path-loss exponent, satellite
height, density, and Nakagami fading parameter. Then, we also derive a tight
lower bound of the coverage probability in closed-form expression while keeping
full generality. Leveraging the derived expression, we identify the optimal
density of satellites in terms of the height and the path-loss exponent. Our
key finding is that the optimal average number of satellites decreases
logarithmically with the network height to maximize the coverage performance.
Simulation results verify the exactness of the derived expressions.

    

### [[2111.13124] An Architecture for Meeting Quality-of-Service Requirements in Multi-User Quantum Networks](http://arxiv.org/abs/2111.13124)


  Quantum communication can enhance internet technology by enabling novel
applications that are provably impossible classically. The successful execution
of such applications relies on the generation of quantum entanglement between
different users of the network which meets stringent performance requirements.
Alongside traditional metrics such as throughput and jitter, one must ensure
the generated entanglement is of sufficiently high quality. Meeting such
performance requirements demands a careful orchestration of many devices in the
network, giving rise to a fundamentally new scheduling problem. Furthermore,
technological limitations of near-term quantum devices impose significant
constraints on scheduling methods hoping to meet performance requirements. In
this work, we propose the first end-to-end design of a centralized quantum
network with multiple users that orchestrates the delivery of entanglement
which meets quality-of-service (QoS) requirements of applications. We achieve
this by using a centrally constructed schedule that manages usage of devices
and ensures the coordinated execution of different quantum operations
throughout the network. We use periodic task scheduling and
resource-constrained project scheduling techniques, including a novel
heuristic, to construct the schedules. Our simulations of four small networks
using hardware-validated network parameters, and of a real-world fiber topology
using futuristic parameters, illustrate trade-offs between traditional and
quantum performance metrics.

    

### [[2111.13414] Optimizing Packet Reception Rates for Low Duty-Cycle BLE Relay Nodes](http://arxiv.org/abs/2111.13414)


  In order to achieve the full potential of the Internet-of-Things,
connectivity between devices should be ubiquitous and efficient. Wireless mesh
networks are a critical component to achieve this ubiquitous connectivity for a
wide range of services, and are composed of terminal devices (i.e., nodes),
such as sensors of various types, and wall powered gateway devices, which
provide further internet connectivity (e..g, via WiFi). When considering large
indoor areas, such as hospitals or industrial scenarios, the mesh must cover a
large area, which introduces concerns regarding range and the number of
gateways needed and respective wall cabling infrastructure. Solutions for mesh
networks implemented over different wireless protocols exist, like the recent
Bluetooth Low Energy (BLE) 5.1. Besides range concerns, choosing which nodes
forward data through the mesh has a large impact on performance and power
consumption. We address the area coverage issue via a battery powered BLE relay
device of our own design, which acts as a range extender by forwarding packets
from end nodes to gateways. We present the relay's design and experimentally
determine the packet forwarding efficiency for several scenarios and
configurations. In the best case, up to 35% of the packets transmitted by 11
nodes can be forwarded to a gateway by a single relay under continuous
operation. A battery lifetime of 1 year can be achieved with a relay duty cycle
of 20%.

    

### [[2111.13525] CoinPrune: Shrinking Bitcoin's Blockchain Retrospectively](http://arxiv.org/abs/2111.13525)


  Popular cryptocurrencies continue to face serious scalability issues due to
their ever-growing blockchains. Thus, modern blockchain designs began to prune
old blocks and rely on recent snapshots for their bootstrapping processes
instead. Unfortunately, established systems are often considered incapable of
adopting these improvements. In this work, we present CoinPrune, our
block-pruning scheme with full Bitcoin compatibility, to revise this popular
belief. CoinPrune bootstraps joining nodes via snapshots that are periodically
created from Bitcoin's set of unspent transaction outputs (UTXO set). Our
scheme establishes trust in these snapshots by relying on CoinPrune-supporting
miners to mutually reaffirm a snapshot's correctness on the blockchain. This
way, snapshots remain trustworthy even if adversaries attempt to tamper with
them. Our scheme maintains its retrospective deployability by relying on
positive feedback only, i.e., blocks containing invalid reaffirmations are not
rejected, but invalid reaffirmations are outpaced by the benign ones created by
an honest majority among CoinPrune-supporting miners. Already today, CoinPrune
reduces the storage requirements for Bitcoin nodes by two orders of magnitude,
as joining nodes need to fetch and process only 6 GiB instead of 271 GiB of
data in our evaluation, reducing the synchronization time of powerful devices
from currently 7 h to 51 min, with even larger potential drops for less
powerful devices. CoinPrune is further aware of higher-level application data,
i.e., it conserves otherwise pruned application data and allows nodes to
obfuscate objectionable and potentially illegal blockchain content from their
UTXO set and the snapshots they distribute.

    

### [[2111.13573] Semi-supervised t-SNE for Millimeter-wave Wireless Localization](http://arxiv.org/abs/2111.13573)


  We consider the mobile localization problem in future millimeter-wave
wireless networks with distributed Base Stations (BSs) based on multi-antenna
channel state information (CSI). For this problem, we propose a Semi-supervised
tdistributed Stochastic Neighbor Embedding (St-SNE) algorithm to directly embed
the high-dimensional CSI samples into the 2D geographical map. We evaluate the
performance of St-SNE in a simulated urban outdoor millimeter-wave radio access
network. Our results show that St-SNE achieves a mean localization error of 6.8
m with only 5% of labeled CSI samples in a 200*200 m^2 area with a ray-tracing
channel model. St-SNE does not require accurate synchronization among multiple
BSs, and is promising for future large-scale millimeter-wave localization.

    

### [[1709.03033] Robust Routing in Interdependent Networks](http://arxiv.org/abs/1709.03033)


  We consider a model of two interdependent networks, where every node in one
network depends on one or more supply nodes in the other network and a node
fails if it loses all of its supply nodes. We develop algorithms to compute the
failure probability of a path, and obtain the most reliable path between a pair
of nodes in a network, under the condition that each supply node fails
independently with a given probability. Our work generalizes the classical
shared risk group model, by considering multiple risks associated with a node
and letting a node fail if all the risks occur. Moreover, we study the diverse
routing problem by considering two paths between a pair of nodes. We define two
paths to be $d$-failure resilient if at least one path survives after removing
$d$ or fewer supply nodes, which generalizes the concept of disjoint paths in a
single network, and risk-disjoint paths in a classical shared risk group model.
We compute the probability that both paths fail, and develop algorithms to
compute the most reliable pair of paths.

    

### [[1901.02636] On the Robustness of Distributed Computing Networks](http://arxiv.org/abs/1901.02636)


  Traffic flows in a distributed computing network require both transmission
and processing, and can be interdicted by removing either communication or
computation resources. We study the robustness of a distributed computing
network under the failures of communication links and computation nodes. We
define cut metrics that measure the connectivity, and show a non-zero gap
between the maximum flow and the minimum cut. Moreover, we study a network flow
interdiction problem that minimizes the maximum flow by removing communication
and computation resources within a given budget. We develop mathematical
programs to compute the optimal interdiction, and polynomial-time approximation
algorithms that achieve near-optimal interdiction in simulation.

    

### [[2111.11332] Experimental demonstration of entanglement delivery using a quantum network stack](http://arxiv.org/abs/2111.11332)


  Scaling current quantum communication demonstrations to a large-scale quantum
network will require not only advancements in quantum hardware capabilities,
but also robust control of such devices to bridge the gap to user demand.
Moreover, the abstraction of tasks and services offered by the quantum network
should enable platform-independent applications to be executed without
knowledge of the underlying physical implementation. Here we experimentally
demonstrate, using remote solid-state quantum network nodes, a link layer and a
physical layer protocol for entanglement-based quantum networks. The link layer
abstracts the physical-layer entanglement attempts into a robust,
platform-independent entanglement delivery service. The system is used to run
full state tomography of the delivered entangled states, as well as preparation
of a remote qubit state on a server by its client. Our results mark a clear
transition from physics experiments to quantum communication systems, which
will enable the development and testing of components of future quantum
networks.

    

### [[2111.12749] FCMpy: A Python Module for Constructing and Analyzing Fuzzy Cognitive Maps](http://arxiv.org/abs/2111.12749)


  FCMpy is an open source package in Python for building and analyzing Fuzzy
Cognitive Maps. More specifically, the package allows 1) deriving fuzzy causal
weights from qualitative data, 2) simulating the system behavior, 3) applying
machine learning algorithms (e.g., Nonlinear Hebbian Learning, Active Hebbian
Learning, Genetic Algorithms and Deterministic Learning) to adjust the FCM
causal weight matrix and to solve classification problems, and 4) implementing
scenario analysis by simulating hypothetical interventions (i.e., analyzing
what-if scenarios).

    

### [[2111.12763] Sparse is Enough in Scaling Transformers](http://arxiv.org/abs/2111.12763)


  Large Transformer models yield impressive results on many tasks, but are
expensive to train, or even fine-tune, and so slow at decoding that their use
and study becomes out of reach. We address this problem by leveraging sparsity.
We study sparse variants for all layers in the Transformer and propose Scaling
Transformers, a family of next generation Transformer models that use sparse
layers to scale efficiently and perform unbatched decoding much faster than the
standard Transformer as we scale up the model size. Surprisingly, the sparse
layers are enough to obtain the same perplexity as the standard Transformer
with the same number of parameters. We also integrate with prior sparsity
approaches to attention and enable fast inference on long sequences even with
limited memory. This results in performance competitive to the state-of-the-art
on long text summarization.

    

### [[2111.12772] JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints](http://arxiv.org/abs/2111.12772)


  Physical products are often complex assemblies combining a multitude of 3D
parts modeled in computer-aided design (CAD) software. CAD designers build up
these assemblies by aligning individual parts to one another using constraints
called joints. In this paper we introduce JoinABLe, a learning-based method
that assembles parts together to form joints. JoinABLe uses the weak
supervision available in standard parametric CAD files without the help of
object class labels or human guidance. Our results show that by making network
predictions over a graph representation of solid models we can outperform
multiple baseline methods with an accuracy (79.53%) that approaches human
performance (80%). Finally, to support future research we release the Fusion
360 Gallery assembly dataset, containing assemblies with rich information on
joints, contact surfaces, holes, and the underlying assembly graph structure.

    

### [[2111.12776] IMBENS: Ensemble Class-imbalanced Learning in Python](http://arxiv.org/abs/2111.12776)


  imbalanced-ensemble, abbreviated as imbens, is an open-source Python toolbox
for quick implementing and deploying ensemble learning algorithms on
class-imbalanced data. It provides access to multiple state-of-art ensemble
imbalanced learning (EIL) methods, visualizer, and utility functions for
dealing with the class imbalance problem. These ensemble methods include
resampling-based, e.g., under/over-sampling, and reweighting-based ones, e.g.,
cost-sensitive learning. Beyond the implementation, we also extend conventional
binary EIL algorithms with new functionalities like multi-class support and
resampling scheduler, thereby enabling them to handle more complex tasks. The
package was developed under a simple, well-documented API design follows that
of scikit-learn for increased ease of use. imbens is released under the MIT
open-source license and can be installed from Python Package Index (PyPI).
Source code, binaries, detailed documentation, and usage examples are available
at this https URL.

    

### [[2111.12786] Differentially Private Nonparametric Regression Under a Growth Condition](http://arxiv.org/abs/2111.12786)


  Given a real-valued hypothesis class $\mathcal{H}$, we investigate under what
conditions there is a differentially private algorithm which learns an optimal
hypothesis from $\mathcal{H}$ given i.i.d. data. Inspired by recent results for
the related setting of binary classification (Alon et al., 2019; Bun et al.,
2020), where it was shown that online learnability of a binary class is
necessary and sufficient for its private learnability, Jung et al. (2020)
showed that in the setting of regression, online learnability of $\mathcal{H}$
is necessary for private learnability. Here online learnability of
$\mathcal{H}$ is characterized by the finiteness of its $\eta$-sequential fat
shattering dimension, ${\rm sfat}_\eta(\mathcal{H})$, for all $\eta > 0$. In
terms of sufficient conditions for private learnability, Jung et al. (2020)
showed that $\mathcal{H}$ is privately learnable if $\lim_{\eta \downarrow 0}
{\rm sfat}_\eta(\mathcal{H})$ is finite, which is a fairly restrictive
condition. We show that under the relaxed condition $\lim \inf_{\eta \downarrow
0} \eta \cdot {\rm sfat}_\eta(\mathcal{H}) = 0$, $\mathcal{H}$ is privately
learnable, establishing the first nonparametric private learnability guarantee
for classes $\mathcal{H}$ with ${\rm sfat}_\eta(\mathcal{H})$ diverging as
$\eta \downarrow 0$. Our techniques involve a novel filtering procedure to
output stable hypotheses for nonparametric function classes.

    

### [[2111.12787] Algorithm and Hardware Co-design for Reconfigurable CNN Accelerator](http://arxiv.org/abs/2111.12787)


  Recent advances in algorithm-hardware co-design for deep neural networks
(DNNs) have demonstrated their potential in automatically designing neural
architectures and hardware designs. Nevertheless, it is still a challenging
optimization problem due to the expensive training cost and the time-consuming
hardware implementation, which makes the exploration on the vast design space
of neural architecture and hardware design intractable. In this paper, we
demonstrate that our proposed approach is capable of locating designs on the
Pareto frontier. This capability is enabled by a novel three-phase co-design
framework, with the following new features: (a) decoupling DNN training from
the design space exploration of hardware architecture and neural architecture,
(b) providing a hardware-friendly neural architecture space by considering
hardware characteristics in constructing the search cells, (c) adopting
Gaussian process to predict accuracy, latency and power consumption to avoid
time-consuming synthesis and place-and-route processes. In comparison with the
manually-designed ResNet101, InceptionV2 and MobileNetV2, we can achieve up to
5% higher accuracy with up to 3x speed up on the ImageNet dataset. Compared
with other state-of-the-art co-design frameworks, our found network and
hardware configuration can achieve 2% ~ 6% higher accuracy, 2x ~ 26x smaller
latency and 8.5x higher energy efficiency.

    

### [[2111.12791] Towards Inter-class and Intra-class Imbalance in Class-imbalanced Learning](http://arxiv.org/abs/2111.12791)


  Imbalanced Learning (IL) is an important problem that widely exists in data
mining applications. Typical IL methods utilize intuitive class-wise resampling
or reweighting to directly balance the training set. However, some recent
research efforts in specific domains show that class-imbalanced learning can be
achieved without class-wise manipulation. This prompts us to think about the
relationship between the two different IL strategies and the nature of the
class imbalance. Fundamentally, they correspond to two essential imbalances
that exist in IL: the difference in quantity between examples from different
classes as well as between easy and hard examples within a single class, i.e.,
inter-class and intra-class imbalance. Existing works fail to explicitly take
both imbalances into account and thus suffer from suboptimal performance. In
light of this, we present Duple-Balanced Ensemble, namely DUBE , a versatile
ensemble learning framework. Unlike prevailing methods, DUBE directly performs
inter-class and intra-class balancing without relying on heavy distance-based
computation, which allows it to achieve competitive performance while being
computationally efficient. We also present a detailed discussion and analysis
about the pros and cons of different inter/intra-class balancing strategies
based on DUBE . Extensive experiments validate the effectiveness of the
proposed method. Code and examples are available at
this https URL.

    

### [[2111.12795] Picasso: Model-free Feature Visualization](http://arxiv.org/abs/2111.12795)


  Today, Machine Learning (ML) applications can have access to tens of
thousands of features. With such feature sets, efficiently browsing and
curating subsets of most relevant features is a challenge. In this paper, we
present a novel approach to visualize up to several thousands of features in a
single image. The image not only shows information on individual features, but
also expresses feature interactions via the relative positioning of features.

    

### [[2111.12796] Out-of-Category Document Identification Using Target-Category Names as Weak Supervision](http://arxiv.org/abs/2111.12796)


  Identifying outlier documents, whose content is different from the majority
of the documents in a corpus, has played an important role to manage a large
text collection. However, due to the absence of explicit information about the
inlier (or target) distribution, existing unsupervised outlier detectors are
likely to make unreliable results depending on the density or diversity of the
outliers in the corpus. To address this challenge, we introduce a new task
referred to as out-of-category detection, which aims to distinguish the
documents according to their semantic relevance to the inlier (or target)
categories by using the category names as weak supervision. In practice, this
task can be widely applicable in that it can flexibly designate the scope of
target categories according to users' interests while requiring only the
target-category names as minimum guidance. In this paper, we present an
out-of-category detection framework, which effectively measures how confidently
each document belongs to one of the target categories based on its
category-specific relevance score. Our framework adopts a two-step approach;
(i) it first generates the pseudo-category label of all unlabeled documents by
exploiting the word-document similarity encoded in a text embedding space, then
(ii) it trains a neural classifier by using the pseudo-labels in order to
compute the confidence from its target-category prediction. The experiments on
real-world datasets demonstrate that our framework achieves the best detection
performance among all baseline methods in various scenarios specifying
different target categories.

    

### [[2111.12797] ReAct: Out-of-distribution Detection With Rectified Activations](http://arxiv.org/abs/2111.12797)


  Out-of-distribution (OOD) detection has received much attention lately due to
its practical importance in enhancing the safe deployment of neural networks.
One of the primary challenges is that models often produce highly confident
predictions on OOD data, which undermines the driving principle in OOD
detection that the model should only be confident about in-distribution
samples. In this work, we propose ReAct--a simple and effective technique for
reducing model overconfidence on OOD data. Our method is motivated by novel
analysis on internal activations of neural networks, which displays highly
distinctive signature patterns for OOD distributions. Our method can generalize
effectively to different network architectures and different OOD detection
scores. We empirically demonstrate that ReAct achieves competitive detection
performance on a comprehensive suite of benchmark datasets, and give
theoretical explication for our method's efficacy. On the ImageNet benchmark,
ReAct reduces the false positive rate (FPR95) by 25.05% compared to the
previous best method.

    

### [[2111.12798] Geometric Priors for Scientific Generative Models in Inertial Confinement Fusion](http://arxiv.org/abs/2111.12798)


  In this paper, we develop a Wasserstein autoencoder (WAE) with a
hyperspherical prior for multimodal data in the application of inertial
confinement fusion. Unlike a typical hyperspherical generative model that
requires computationally inefficient sampling from distributions like the von
Mis Fisher, we sample from a normal distribution followed by a projection layer
before the generator. Finally, to determine the validity of the generated
samples, we exploit a known relationship between the modalities in the dataset
as a scientific constraint, and study different properties of the proposed
model.

    

### [[2111.12805] Application of deep learning to camera trap data for ecologists in planning / engineering -- Can captivity imagery train a model which generalises to the wild?](http://arxiv.org/abs/2111.12805)


  Understanding the abundance of a species is the first step towards
understanding both its long-term sustainability and the impact that we may be
having upon it. Ecologists use camera traps to remotely survey for the presence
of specific animal species. Previous studies have shown that deep learning
models can be trained to automatically detect and classify animals within
camera trap imagery with high levels of confidence. However, the ability to
train these models is reliant upon having enough high-quality training data.
What happens when the animal is rare or the data sets are non-existent? This
research proposes an approach of using images of rare animals in captivity
(focusing on the Scottish wildcat) to generate the training dataset. We explore
the challenges associated with generalising a model trained on captivity data
when applied to data collected in the wild. The research is contextualised by
the needs of ecologists in planning/engineering. Following precedents from
other research, this project establishes an ensemble for object detection,
image segmentation and image classification models which are then tested using
different image manipulation and class structuring techniques to encourage
model generalisation. The research concludes, in the context of Scottish
wildcat, that models trained on captivity imagery cannot be generalised to wild
camera trap imagery using existing techniques. However, final model
performances based on a two-class model Wildcat vs Not Wildcat achieved an
overall accuracy score of 81.6% and Wildcat accuracy score of 54.8% on a test
set in which only 1% of images contained a wildcat. This suggests using
captivity images is feasible with further research. This is the first research
which attempts to generate a training set based on captivity data and the first
to explore the development of such models in the context of ecologists in
planning/engineering.

    

### [[2111.12823] Fairness for AUC via Feature Augmentation](http://arxiv.org/abs/2111.12823)


  We study fairness in the context of classification where the performance is
measured by the area under the curve (AUC) of the receiver operating
characteristic. AUC is commonly used when both Type I (false positive) and Type
II (false negative) errors are important. However, the same classifier can have
significantly varying AUCs for different protected groups and, in real-world
applications, it is often desirable to reduce such cross-group differences. We
address the problem of how to select additional features to most greatly
improve AUC for the disadvantaged group. Our results establish that the
unconditional variance of features does not inform us about AUC fairness but
class-conditional variance does. Using this connection, we develop a novel
approach, fairAUC, based on feature augmentation (adding features) to mitigate
bias between identifiable groups. We evaluate fairAUC on synthetic and
real-world (COMPAS) datasets and find that it significantly improves AUC for
the disadvantaged group relative to benchmarks maximizing overall AUC and
minimizing bias between groups.

    

### [[2111.12835] SchemaDB: Structures in Relational Datasets](http://arxiv.org/abs/2111.12835)


  In this paper we introduce the SchemaDB data-set; a collection of relational
database schemata in both sql and graph formats. Databases are not commonly
shared publicly for reasons of privacy and security, so schemata are not
available for study. Consequently, an understanding of database structures in
the wild is lacking, and most examples found publicly belong to common
development frameworks or are derived from textbooks or engine benchmark
designs. SchemaDB contains 2,500 samples of relational schemata found in public
repositories which we have standardised to MySQL syntax. We provide our
gathering and transformation methodology, summary statistics, and structural
analysis, and discuss potential downstream research tasks in several domains.

    

### [[2111.12840] Explaining machine-learned particle-flow reconstruction](http://arxiv.org/abs/2111.12840)


  The particle-flow (PF) algorithm is used in general-purpose particle
detectors to reconstruct a comprehensive particle-level view of the collision
by combining information from different subdetectors. A graph neural network
(GNN) model, known as the machine-learned particle-flow (MLPF) algorithm, has
been developed to substitute the rule-based PF algorithm. However,
understanding the model's decision making is not straightforward, especially
given the complexity of the set-to-set prediction task, dynamic graph building,
and message-passing steps. In this paper, we adapt the layerwise-relevance
propagation technique for GNNs and apply it to the MLPF algorithm to gauge the
relevant nodes and features for its predictions. Through this process, we gain
insight into the model's decision-making.

    

### [[2111.12843] Animal Behavior Classification via Accelerometry Data and Recurrent Neural Networks](http://arxiv.org/abs/2111.12843)


  We study the classification of animal behavior using accelerometry data
through various recurrent neural network (RNN) models. We evaluate the
classification performance and complexity of the considered models, which
feature long short-time memory (LSTM) or gated recurrent unit (GRU)
architectures with varying depths and widths, using four datasets acquired from
cattle via collar or ear tags. We also include two state-of-the-art
convolutional neural network (CNN)-based time-series classification models in
the evaluations. The results show that the RNN-based models can achieve similar
or higher classification accuracy compared with the CNN-based models while
having less computational and memory requirements. We also observe that the
models with GRU architecture generally outperform the ones with LSTM
architecture in terms of classification accuracy despite being less complex. A
single-layer uni-directional GRU model with 64 hidden units appears to offer a
good balance between accuracy and complexity making it suitable for
implementation on edge/embedded devices.

    

### [[2111.12849] Particle Graph Autoencoders and Differentiable, Learned Energy Mover's Distance](http://arxiv.org/abs/2111.12849)


  Autoencoders have useful applications in high energy physics in anomaly
detection, particularly for jets - collimated showers of particles produced in
collisions such as those at the CERN Large Hadron Collider. We explore the use
of graph-based autoencoders, which operate on jets in their "particle cloud"
representations and can leverage the interdependencies among the particles
within a jet, for such tasks. Additionally, we develop a differentiable
approximation to the energy mover's distance via a graph neural network, which
may subsequently be used as a reconstruction loss function for autoencoders.

    

### [[2111.12861] A Deep Learning Approach for Macroscopic Energy Consumption Prediction with Microscopic Quality for Electric Vehicles](http://arxiv.org/abs/2111.12861)


  This paper presents a machine learning approach to model the electric
consumption of electric vehicles at macroscopic level, i.e., in the absence of
a speed profile, while preserving microscopic level accuracy. For this work, we
leveraged a high-performance, agent-based transportation tool to model trips
that occur in the Greater Chicago region under various scenario changes, along
with physics-based modeling and simulation tools to provide high-fidelity
energy consumption values. The generated results constitute a very large
dataset of vehicle-route energy outcomes that capture variability in vehicle
and routing setting, and in which high-fidelity time series of vehicle speed
dynamics is masked. We show that although all internal dynamics that affect
energy consumption are masked, it is possible to learn aggregate-level energy
consumption values quite accurately with a deep learning approach. When
large-scale data is available, and with carefully tailored feature engineering,
a well-designed model can overcome and retrieve latent information. This model
has been deployed and integrated within POLARIS Transportation System
Simulation Tool to support real-time behavioral transportation models for
individual charging decision-making, and rerouting of electric vehicles.

    

### [[2111.12865] Multi-fidelity Stability for Graph Representation Learning](http://arxiv.org/abs/2111.12865)


  In the problem of structured prediction with graph representation learning
(GRL for short), the hypothesis returned by the algorithm maps the set of
features in the \emph{receptive field} of the targeted vertex to its label. To
understand the learnability of those algorithms, we introduce a weaker form of
uniform stability termed \emph{multi-fidelity stability} and give learning
guarantees for weakly dependent graphs. We testify that
~\citet{london2016stability}'s claim on the generalization of a single sample
holds for GRL when the receptive field is sparse. In addition, we study the
stability induced bound for two popular algorithms: \textbf{(1)} Stochastic
gradient descent under convex and non-convex landscape. In this example, we
provide non-asymptotic bounds that highly depend on the sparsity of the
receptive field constructed by the algorithm. \textbf{(2)} The constrained
regression problem on a 1-layer linear equivariant GNN. In this example, we
present lower bounds for the discrepancy between the two types of stability,
which justified the multi-fidelity design.

    

### [[2111.12867] Back to Reality for Imitation Learning](http://arxiv.org/abs/2111.12867)


  Imitation learning, and robot learning in general, emerged due to
breakthroughs in machine learning, rather than breakthroughs in robotics. As
such, evaluation metrics for robot learning are deeply rooted in those for
machine learning, and focus primarily on data efficiency. We believe that a
better metric for real-world robot learning is time efficiency, which better
models the true cost to humans. This is a call to arms to the robot learning
community to develop our own evaluation metrics, tailored towards the long-term
goals of real-world robotics.

    

### [[2111.12873] Quantised Transforming Auto-Encoders: Achieving Equivariance to Arbitrary Transformations in Deep Networks](http://arxiv.org/abs/2111.12873)


  In this work we investigate how to achieve equivariance to input
transformations in deep networks, purely from data, without being given a model
of those transformations. Convolutional Neural Networks (CNNs), for example,
are equivariant to image translation, a transformation that can be easily
modelled (by shifting the pixels vertically or horizontally). Other
transformations, such as out-of-plane rotations, do not admit a simple analytic
model. We propose an auto-encoder architecture whose embedding obeys an
arbitrary set of equivariance relations simultaneously, such as translation,
rotation, colour changes, and many others. This means that it can take an input
image, and produce versions transformed by a given amount that were not
observed before (e.g. a different point of view of the same object, or a colour
variation). Despite extending to many (even non-geometric) transformations, our
model reduces exactly to a CNN in the special case of translation-equivariance.
Equivariances are important for the interpretability and robustness of deep
networks, and we demonstrate results of successful re-rendering of transformed
versions of input images on several synthetic and real datasets, as well as
results on object pose estimation.

    

### [[2111.12876] Time-independent Generalization Bounds for SGLD in Non-convex Settings](http://arxiv.org/abs/2111.12876)


  We establish generalization error bounds for stochastic gradient Langevin
dynamics (SGLD) with constant learning rate under the assumptions of
dissipativity and smoothness, a setting that has received increased attention
in the sampling/optimization literature. Unlike existing bounds for SGLD in
non-convex settings, ours are time-independent and decay to zero as the sample
size increases. Using the framework of uniform stability, we establish
time-independent bounds by exploiting the Wasserstein contraction property of
the Langevin diffusion, which also allows us to circumvent the need to bound
gradients using Lipschitz-like assumptions. Our analysis also supports variants
of SGLD that use different discretization methods, incorporate Euclidean
projections, or use non-isotropic noise.

    

### [[2111.12877] A Letter on Convergence of In-Parameter-Linear Nonlinear Neural Architectures with Gradient Learnings](http://arxiv.org/abs/2111.12877)


  This letter summarizes and proves the concept of bounded-input bounded-state
(BIBS) stability for weight convergence of a broad family of
in-parameter-linear nonlinear neural architectures as it generally applies to a
broad family of incremental gradient learning algorithms. A practical BIBS
convergence condition results from the derived proofs for every individual
learning point or batches for real-time applications.

    

### [[2111.12896] SLA$^2$P: Self-supervised Anomaly Detection with Adversarial Perturbation](http://arxiv.org/abs/2111.12896)


  Anomaly detection is a fundamental yet challenging problem in machine
learning due to the lack of label information. In this work, we propose a novel
and powerful framework, dubbed as SLA$^2$P, for unsupervised anomaly detection.
After extracting representative embeddings from raw data, we apply random
projections to the features and regard features transformed by different
projections as belonging to distinct pseudo classes. We then train a classifier
network on these transformed features to perform self-supervised learning. Next
we add adversarial perturbation to the transformed features to decrease their
softmax scores of the predicted labels and design anomaly scores based on the
predictive uncertainties of the classifier on these perturbed features. Our
motivation is that because of the relatively small number and the decentralized
modes of anomalies, 1) the pseudo label classifier's training concentrates more
on learning the semantic information of normal data rather than anomalous data;
2) the transformed features of the normal data are more robust to the
perturbations than those of the anomalies. Consequently, the perturbed
transformed features of anomalies fail to be classified well and accordingly
have lower anomaly scores than those of the normal samples. Extensive
experiments on image, text and inherently tabular benchmark datasets back up
our findings and indicate that SLA$^2$P achieves state-of-the-art results on
unsupervised anomaly detection tasks consistently.

    

### [[2111.12906] Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity](http://arxiv.org/abs/2111.12906)


  Adversarial examples can easily degrade the classification performance in
neural networks. Empirical methods for promoting robustness to such examples
have been proposed, but often lack both analytical insights and formal
guarantees. Recently, some robustness certificates have appeared in the
literature based on system theoretic notions. This work proposes an incremental
dissipativity-based robustness certificate for neural networks in the form of a
linear matrix inequality for each layer. We also propose an equivalent spectral
norm bound for this certificate which is scalable to neural networks with
multiple layers. We demonstrate the improved performance against adversarial
attacks on a feed-forward neural network trained on MNIST and an Alexnet
trained using CIFAR-10.

    

### [[2111.12907] Fragment-based molecular generative model with high generalization ability and synthetic accessibility](http://arxiv.org/abs/2111.12907)


  Deep generative models are attracting great attention for molecular design
with desired properties. Most existing models generate molecules by
sequentially adding atoms. This often renders generated molecules with less
correlation with target properties and low synthetic accessibility. Molecular
fragments such as functional groups are more closely related to molecular
properties and synthetic accessibility than atoms. Here, we propose a
fragment-based molecular generative model which designs new molecules with
target properties by sequentially adding molecular fragments to any given
starting molecule. A key feature of our model is a high generalization ability
in terms of property control and fragment types. The former becomes possible by
learning the contribution of individual fragments to the target properties in
an auto-regressive manner. For the latter, we used a deep neural network that
predicts the bonding probability of two molecules from the embedding vectors of
the two molecules as input. The high synthetic accessibility of the generated
molecules is implicitly considered while preparing the fragment library with
the BRICS decomposition method. We show that the model can generate molecules
with the simultaneous control of multiple target properties at a high success
rate. It also works equally well with unseen fragments even in the property
range where the training data is rare, verifying the high generalization
ability. As a practical application, we demonstrated that the model can
generate potential inhibitors with high binding affinities against the 3CL
protease of SARS-COV-2 in terms of docking score.

    

### [[2111.12922] Clustering Effect of (Linearized) Adversarial Robust Models](http://arxiv.org/abs/2111.12922)


  Adversarial robustness has received increasing attention along with the study
of adversarial examples. So far, existing works show that robust models not
only obtain robustness against various adversarial attacks but also boost the
performance in some downstream tasks. However, the underlying mechanism of
adversarial robustness is still not clear. In this paper, we interpret
adversarial robustness from the perspective of linear components, and find that
there exist some statistical properties for comprehensively robust models.
Specifically, robust models show obvious hierarchical clustering effect on
their linearized sub-networks, when removing or replacing all non-linear
components (e.g., batch normalization, maximum pooling, or activation layers).
Based on these observations, we propose a novel understanding of adversarial
robustness and apply it on more tasks including domain adaption and robustness
boosting. Experimental evaluations demonstrate the rationality and superiority
of our proposed clustering strategy.

    

### [[2111.12925] ContourletNet: A Generalized Rain Removal Architecture Using Multi-Direction Hierarchical Representation](http://arxiv.org/abs/2111.12925)


  Images acquired from rainy scenes usually suffer from bad visibility which
may damage the performance of computer vision applications. The rainy scenarios
can be categorized into two classes: moderate rain and heavy rain scenes.
Moderate rain scene mainly consists of rain streaks while heavy rain scene
contains both rain streaks and the veiling effect (similar to haze). Although
existing methods have achieved excellent performance on these two cases
individually, it still lacks a general architecture to address both heavy rain
and moderate rain scenarios effectively. In this paper, we construct a
hierarchical multi-direction representation network by using the contourlet
transform (CT) to address both moderate rain and heavy rain scenarios. The CT
divides the image into the multi-direction subbands (MS) and the semantic
subband (SS). First, the rain streak information is retrieved to the MS based
on the multi-orientation property of the CT. Second, a hierarchical
architecture is proposed to reconstruct the background information including
damaged semantic information and the veiling effect in the SS. Last, the
multi-level subband discriminator with the feedback error map is proposed. By
this module, all subbands can be well optimized. This is the first architecture
that can address both of the two scenarios effectively. The code is available
in this https URL.

    

### [[2111.12933] ML-Decoder: Scalable and Versatile Classification Head](http://arxiv.org/abs/2111.12933)


  In this paper, we introduce ML-Decoder, a new attention-based classification
head. ML-Decoder predicts the existence of class labels via queries, and
enables better utilization of spatial data compared to global average pooling.
By redesigning the decoder architecture, and using a novel group-decoding
scheme, ML-Decoder is highly efficient, and can scale well to thousands of
classes. Compared to using a larger backbone, ML-Decoder consistently provides
a better speed-accuracy trade-off. ML-Decoder is also versatile - it can be
used as a drop-in replacement for various classification heads, and generalize
to unseen classes when operated with word queries. Novel query augmentations
further improve its generalization ability. Using ML-Decoder, we achieve
state-of-the-art results on several classification tasks: on MS-COCO
multi-label, we reach 91.4% mAP; on NUS-WIDE zero-shot, we reach 31.1% ZSL mAP;
and on ImageNet single-label, we reach with vanilla ResNet50 backbone a new top
score of 80.7%, without extra data or distillation. Public code is available
at: this https URL


### [[2111.12940] Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation](http://arxiv.org/abs/2111.12940)


  Self-training has greatly facilitated domain adaptive semantic segmentation,
which iteratively generates pseudo labels on the target domain and retrains the
network. However, since the realistic segmentation datasets are highly
imbalanced, target pseudo labels are typically biased to the majority classes
and basically noisy, leading to an error-prone and sub-optimal model. To
address this issue, we propose a region-based active learning approach for
semantic segmentation under a domain shift, aiming to automatically query a
small partition of image regions to be labeled while maximizing segmentation
performance. Our algorithm, Active Learning via Region Impurity and Prediction
Uncertainty (AL-RIPU), introduces a novel acquisition strategy characterizing
the spatial adjacency of image regions along with the prediction confidence. We
show that the proposed region-based selection strategy makes more efficient use
of a limited budget than image-based or point-based counterparts. Meanwhile, we
enforce local prediction consistency between a pixel and its nearest neighbor
on a source image. Further, we develop a negative learning loss to enhance the
discriminative representation learning on the target domain. Extensive
experiments demonstrate that our method only requires very few annotations to
almost reach the supervised performance and substantially outperforms
state-of-the-art methods.

    

### [[2111.12945] Correcting the Laplace Method with Variational Bayes](http://arxiv.org/abs/2111.12945)


  Approximate inference methods like the Laplace method, Laplace approximations
and variational methods, amongst others, are popular methods when exact
inference is not feasible due to the complexity of the model or the abundance
of data. In this paper we propose a hybrid approximate method namely Low-Rank
Variational Bayes correction (VBC), that uses the Laplace method and
subsequently a Variational Bayes correction to the posterior mean. The cost is
essentially that of the Laplace method which ensures scalability of the method.
We illustrate the method and its advantages with simulated and real data, on
small and large scale.

    

### [[2111.12950] Few-shot Deep Representation Learning based on Information Bottleneck Principle](http://arxiv.org/abs/2111.12950)


  In a standard anomaly detection problem, a detection model is trained in an
unsupervised setting, under an assumption that the samples were generated from
a single source of normal data. In practice, however, normal data often consist
of multiple classes. In such settings, learning to differentiate between normal
instances and anomalies among discrepancies between normal classes without
large-scale labeled data presents a significant challenge. In this work, we
attempt to overcome this challenge by preparing few examples from each normal
class, which is not excessively costly. The above setting can also be described
as a few-shot learning for multiple, normal classes, with the goal of learning
a useful representation for anomaly detection. In order to utilize the limited
labeled examples in training, we integrate the inter-class distances among the
labeled examples in the deep feature space into the MAP loss. We derive their
relations from an information-theoretic principle. Our empirical study shows
that the proposed model improves the segmentation of normal classes in the deep
feature space which contributes to identifying the anomaly class examples.

    

### [[2111.12951] Reliable Graph Neural Networks for Drug Discovery Under Distributional Shift](http://arxiv.org/abs/2111.12951)


  The concern of overconfident mis-predictions under distributional shift
demands extensive reliability research on Graph Neural Networks used in
critical tasks in drug discovery. Here we first introduce CardioTox, a
real-world benchmark on drug cardio-toxicity to facilitate such efforts. Our
exploratory study shows overconfident mis-predictions are often distant from
training data. That leads us to develop distance-aware GNNs: GNN-SNGP. Through
evaluation on CardioTox and three established benchmarks, we demonstrate
GNN-SNGP's effectiveness in increasing distance-awareness, reducing
overconfident mis-predictions and making better calibrated predictions without
sacrificing accuracy performance. Our ablation study further reveals the
representation learned by GNN-SNGP improves distance-preservation over its base
architecture and is one major factor for improvements.

    

### [[2111.12952] AutoHEnsGNN: Winning Solution to AutoGraph Challenge for KDD Cup 2020](http://arxiv.org/abs/2111.12952)


  Graph Neural Networks (GNNs) have become increasingly popular and achieved
impressive results in many graph-based applications. However, extensive manual
work and domain knowledge are required to design effective architectures, and
the results of GNN models have high variance with different training setups,
which limits the application of existing GNN models. In this paper, we present
AutoHEnsGNN, a framework to build effective and robust models for graph tasks
without any human intervention. AutoHEnsGNN won first place in the AutoGraph
Challenge for KDD Cup 2020, and achieved the best rank score of five real-life
datasets in the final phase. Given a task, AutoHEnsGNN first applies a fast
proxy evaluation to automatically select a pool of promising GNN models. Then
it builds a hierarchical ensemble framework: 1) We propose graph self-ensemble
(GSE), which can reduce the variance of weight initialization and efficiently
exploit the information of local and global neighborhoods; 2) Based on GSE, a
weighted ensemble of different types of GNN models is used to effectively learn
more discriminative node representations. To efficiently search the
architectures and ensemble weights, we propose AutoHEnsGNN$_{\text{Gradient}}$,
which treats the architectures and ensemble weights as architecture parameters
and uses gradient-based architecture search to obtain optimal configurations,
and AutoHEnsGNN$_{\text{Adaptive}}$, which can adaptively adjust the ensemble
weight based on the model accuracy. Extensive experiments on node
classification, graph classification, edge prediction and KDD Cup challenge
demonstrate the effectiveness and generality of AutoHEnsGNN

    

### [[2111.12953] Learn Zero-Constraint-Violation Policy in Model-Free Constrained Reinforcement Learning](http://arxiv.org/abs/2111.12953)


  In the trial-and-error mechanism of reinforcement learning (RL), a notorious
contradiction arises when we expect to learn a safe policy: how to learn a safe
policy without enough data and prior model about the dangerous region? Existing
methods mostly use the posterior penalty for dangerous actions, which means
that the agent is not penalized until experiencing danger. This fact causes
that the agent cannot learn a zero-violation policy even after convergence.
Otherwise, it would not receive any penalty and lose the knowledge about
danger. In this paper, we propose the safe set actor-critic (SSAC) algorithm,
which confines the policy update using safety-oriented energy functions, or the
safety indexes. The safety index is designed to increase rapidly for
potentially dangerous actions, which allows us to locate the safe set on the
action space, or the control safe set. Therefore, we can identify the dangerous
actions prior to taking them, and further obtain a zero constraint-violation
policy after convergence.We claim that we can learn the energy function in a
model-free manner similar to learning a value function. By using the energy
function transition as the constraint objective, we formulate a constrained RL
problem. We prove that our Lagrangian-based solutions make sure that the
learned policy will converge to the constrained optimum under some assumptions.
The proposed algorithm is evaluated on both the complex simulation environments
and a hardware-in-loop (HIL) experiment with a real controller from the
autonomous vehicle. Experimental results suggest that the converged policy in
all environments achieves zero constraint violation and comparable performance
with model-based baselines.

    

### [[2111.12961] Distributed Policy Gradient with Variance Reduction in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2111.12961)


  This paper studies a distributed policy gradient in collaborative multi-agent
reinforcement learning (MARL), where agents over a communication network aim to
find the optimal policy to maximize the average of all agents' local returns.
Due to the non-concave performance function of policy gradient, the existing
distributed stochastic optimization methods for convex problems cannot be
directly used for policy gradient in MARL. This paper proposes a distributed
policy gradient with variance reduction and gradient tracking to address the
high variances of policy gradient, and utilizes importance weight to solve the
non-stationary problem in the sampling process. We then provide an upper bound
on the mean-squared stationary gap, which depends on the number of iterations,
the mini-batch size, the epoch size, the problem parameters, and the network
topology. We further establish the sample and communication complexity to
obtain an $\epsilon$-approximate stationary point. Numerical experiments on the
control problem in MARL are performed to validate the effectiveness of the
proposed algorithm.

    

### [[2111.12963] Error Bounds for a Matrix-Vector Product Approximation with Deep ReLU Neural Networks](http://arxiv.org/abs/2111.12963)


  Among the several paradigms of artificial intelligence (AI) or machine
learning (ML), a remarkably successful paradigm is deep learning. Deep
learning's phenomenal success has been hoped to be interpreted via fundamental
research on the theory of deep learning. Accordingly, applied research on deep
learning has spurred the theory of deep learning-oriented depth and breadth of
developments. Inspired by such developments, we pose these fundamental
questions: can we accurately approximate an arbitrary matrix-vector product
using deep rectified linear unit (ReLU) feedforward neural networks (FNNs)? If
so, can we bound the resulting approximation error? In light of these
questions, we derive error bounds in Lebesgue and Sobolev norms that comprise
our developed deep approximation theory. Guided by this theory, we have
successfully trained deep ReLU FNNs whose test results justify our developed
theory. The developed theory is also applicable for guiding and easing the
training of teacher deep ReLU FNNs in view of the emerging teacher-student AI
or ML paradigms that are essential for solving several AI or ML problems in
wireless communications and signal processing; network science and graph signal
processing; and network neuroscience and brain physics.

    

### [[2111.12984] Demystifying Graph Neural Network Explanations](http://arxiv.org/abs/2111.12984)


  Graph neural networks (GNNs) are quickly becoming the standard approach for
learning on graph structured data across several domains, but they lack
transparency in their decision-making. Several perturbation-based approaches
have been developed to provide insights into the decision making process of
GNNs. As this is an early research area, the methods and data used to evaluate
the generated explanations lack maturity. We explore these existing approaches
and identify common pitfalls in three main areas: (1) synthetic data generation
process, (2) evaluation metrics, and (3) the final presentation of the
explanation. For this purpose, we perform an empirical study to explore these
pitfalls along with their unintended consequences and propose remedies to
mitigate their effects.

    

### [[2111.12986] A-Muze-Net: Music Generation by Composing the Harmony based on the Generated Melody](http://arxiv.org/abs/2111.12986)


  We present a method for the generation of Midi files of piano music. The
method models the right and left hands using two networks, where the left hand
is conditioned on the right hand. This way, the melody is generated before the
harmony. The Midi is represented in a way that is invariant to the musical
scale, and the melody is represented, for the purpose of conditioning the
harmony, by the content of each bar, viewed as a chord. Finally, notes are
added randomly, based on this chord representation, in order to enrich the
generated audio. Our experiments show a significant improvement over the state
of the art for training on such datasets, and demonstrate the contribution of
each of the novel components.

    

### [[2111.12990] Learning Algebraic Representation for Systematic Generalization in Abstract Reasoning](http://arxiv.org/abs/2111.12990)


  Is intelligence realized by connectionist or classicist? While connectionist
approaches have achieved superhuman performance, there has been growing
evidence that such task-specific superiority is particularly fragile in
systematic generalization. This observation lies in the central debate between
connectionist and classicist, wherein the latter continually advocates an
algebraic treatment in cognitive architectures. In this work, we follow the
classicist's call and propose a hybrid approach to improve systematic
generalization in reasoning. Specifically, we showcase a prototype with
algebraic representation for the abstract spatial-temporal reasoning task of
Raven's Progressive Matrices (RPM) and present the ALgebra-Aware
Neuro-Semi-Symbolic (ALANS) learner. The ALANS learner is motivated by abstract
algebra and the representation theory. It consists of a neural visual
perception frontend and an algebraic abstract reasoning backend: the frontend
summarizes the visual information from object-based representation, while the
backend transforms it into an algebraic structure and induces the hidden
operator on the fly. The induced operator is later executed to predict the
answer's representation, and the choice most similar to the prediction is
selected as the solution. Extensive experiments show that by incorporating an
algebraic treatment, the ALANS learner outperforms various pure connectionist
models in domains requiring systematic generalization. We further show that the
algebraic representation learned can be decoded by isomorphism to generate an
answer.

    

### [[2111.12991] Non Parametric Data Augmentations Improve Deep-Learning based Brain Tumor Segmentation](http://arxiv.org/abs/2111.12991)


  Automatic brain tumor segmentation from Magnetic Resonance Imaging (MRI) data
plays an important role in assessing tumor response to therapy and personalized
treatment stratification.Manual segmentation is tedious and
subjective.Deep-learning-based algorithms for brain tumor segmentation have the
potential to provide objective and fast tumor segmentation.However, the
training of such algorithms requires large datasets which are not always
available. Data augmentation techniques may reduce the need for large
datasets.However current approaches are mostly parametric and may result in
suboptimal performance.We introduce two non-parametric methods of data
augmentation for brain tumor segmentation: the mixed structure regularization
(MSR) and shuffle pixels noise (SPN).We evaluated the added value of the MSR
and SPN augmentation on the brain tumor segmentation (BraTS) 2018 challenge
dataset with the encoder-decoder nnU-Net architecture as the segmentation
algorithm.Both MSR and SPN improve the nnU-Net segmentation accuracy compared
to parametric Gaussian noise augmentation.Mean dice score increased from 80% to
82% and p-values=0.0022, 0.0028 when comparing MSR to non-parametric
augmentation for the tumor core and whole tumor experiments respectively.The
proposed MSR and SPN augmentations have the potential to improve
neural-networks performance in other tasks as well.

    

### [[2111.12993] PolyViT: Co-training Vision Transformers on Images, Videos and Audio](http://arxiv.org/abs/2111.12993)


  Can we train a single transformer model capable of processing multiple
modalities and datasets, whilst sharing almost all of its learnable parameters?
We present PolyViT, a model trained on image, audio and video which answers
this question. By co-training different tasks on a single modality, we are able
to improve the accuracy of each individual task and achieve state-of-the-art
results on 5 standard video- and audio-classification datasets. Co-training
PolyViT on multiple modalities and tasks leads to a model that is even more
parameter-efficient, and learns representations that generalize across multiple
domains. Moreover, we show that co-training is simple and practical to
implement, as we do not need to tune hyperparameters for each combination of
datasets, but can simply adapt those from standard, single-task training.

    

### [[2111.12995] Learning Low-Dimensional Quadratic-Embeddings of High-Fidelity Nonlinear Dynamics using Deep Learning](http://arxiv.org/abs/2111.12995)


  Learning dynamical models from data plays a vital role in engineering design,
optimization, and predictions. Building models describing dynamics of complex
processes (e.g., weather dynamics, or reactive flows) using empirical knowledge
or first principles are onerous or infeasible. Moreover, these models are
high-dimensional but spatially correlated. It is, however, observed that the
dynamics of high-fidelity models often evolve in low-dimensional manifolds.
Furthermore, it is also known that for sufficiently smooth vector fields
defining the nonlinear dynamics, a quadratic model can describe it accurately
in an appropriate coordinate system, conferring to the McCormick relaxation
idea in nonconvex optimization. Here, we aim at finding a low-dimensional
embedding of high-fidelity dynamical data, ensuring a simple quadratic model to
explain its dynamics. To that aim, this work leverages deep learning to
identify low-dimensional quadratic embeddings for high-fidelity dynamical
systems. Precisely, we identify the embedding of data using an autoencoder to
have the desired property of the embedding. We also embed a Runge-Kutta method
to avoid the time-derivative computations, which is often a challenge. We
illustrate the ability of the approach by a couple of examples, arising in
describing flow dynamics and the oscillatory tubular reactor model.

    

### [[2111.12996] Generalizing electrocardiogram delineation: training convolutional neural networks with synthetic data augmentation](http://arxiv.org/abs/2111.12996)


  Obtaining per-beat information is a key task in the analysis of cardiac
electrocardiograms (ECG), as many downstream diagnosis tasks are dependent on
ECG-based measurements. Those measurements, however, are costly to produce,
especially in recordings that change throughout long periods of time. However,
existing annotated databases for ECG delineation are small, being insufficient
in size and in the array of pathological conditions they represent. This
article delves has two main contributions. First, a pseudo-synthetic data
generation algorithm was developed, based in probabilistically composing ECG
traces given "pools" of fundamental segments, as cropped from the original
databases, and a set of rules for their arrangement into coherent synthetic
traces. The generation of conditions is controlled by imposing expert knowledge
on the generated trace, which increases the input variability for training the
model. Second, two novel segmentation-based loss functions have been developed,
which attempt at enforcing the prediction of an exact number of independent
structures and at producing closer segmentation boundaries by focusing on a
reduced number of samples. The best performing model obtained an $F_1$-score of
99.38\% and a delineation error of $2.19 \pm 17.73$ ms and $4.45 \pm 18.32$ ms
for all wave's fiducials (onsets and offsets, respectively), as averaged across
the P, QRS and T waves for three distinct freely available databases. The
excellent results were obtained despite the heterogeneous characteristics of
the tested databases, in terms of lead configurations (Holter, 12-lead),
sampling frequencies ($250$, $500$ and $2,000$ Hz) and represented
pathophysiologies (e.g., different types of arrhythmias, sinus rhythm with
structural heart disease), hinting at its generalization capabilities, while
outperforming current state-of-the-art delineation approaches.

    

### [[2111.13023] Rotation Equivariant 3D Hand Mesh Generation from a Single RGB Image](http://arxiv.org/abs/2111.13023)


  We develop a rotation equivariant model for generating 3D hand meshes from 2D
RGB images. This guarantees that as the input image of a hand is rotated the
generated mesh undergoes a corresponding rotation. Furthermore, this removes
undesirable deformations in the meshes often generated by methods without
rotation equivariance. By building a rotation equivariant model, through
considering symmetries in the problem, we reduce the need for training on very
large datasets to achieve good mesh reconstruction.
The encoder takes images defined on $\mathbb{Z}^{2}$ and maps these to latent
functions defined on the group $C_{8}$. We introduce a novel vector mapping
function to map the function defined on $C_{8}$ to a latent point cloud space
defined on the group $\mathrm{SO}(2)$. Further, we introduce a 3D projection
function that learns a 3D function from the $\mathrm{SO}(2)$ latent space.
Finally, we use an $\mathrm{SO}(3)$ equivariant decoder to ensure rotation
equivariance. Our rotation equivariant model outperforms state-of-the-art
methods on a real-world dataset and we demonstrate that it accurately captures
the shape and pose in the generated meshes under rotation of the input hand.

    

### [[2111.13026] Bandit problems with fidelity rewards](http://arxiv.org/abs/2111.13026)


  The fidelity bandits problem is a variant of the $K$-armed bandit problem in
which the reward of each arm is augmented by a fidelity reward that provides
the player with an additional payoff depending on how 'loyal' the player has
been to that arm in the past. We propose two models for fidelity. In the
loyalty-points model the amount of extra reward depends on the number of times
the arm has previously been played. In the subscription model the additional
reward depends on the current number of consecutive draws of the arm. We
consider both stochastic and adversarial problems. Since single-arm strategies
are not always optimal in stochastic problems, the notion of regret in the
adversarial setting needs careful adjustment. We introduce three possible
notions of regret and investigate which can be bounded sublinearly. We study in
detail the special cases of increasing, decreasing and coupon (where the player
gets an additional reward after every $m$ plays of an arm) fidelity rewards.
For the models which do not necessarily enjoy sublinear regret, we provide a
worst case lower bound. For those models which exhibit sublinear regret, we
provide algorithms and bound their regret.

    

### [[2111.13034] DeepWiVe: Deep-Learning-Aided Wireless Video Transmission](http://arxiv.org/abs/2111.13034)


  We present DeepWiVe, the first-ever end-to-end joint source-channel coding
(JSCC) video transmission scheme that leverages the power of deep neural
networks (DNNs) to directly map video signals to channel symbols, combining
video compression, channel coding, and modulation steps into a single neural
transform. Our DNN decoder predicts residuals without distortion feedback,
which improves video quality by accounting for occlusion/disocclusion and
camera movements. We simultaneously train different bandwidth allocation
networks for the frames to allow variable bandwidth transmission. Then, we
train a bandwidth allocation network using reinforcement learning (RL) that
optimizes the allocation of limited available channel bandwidth among video
frames to maximize overall visual quality. Our results show that DeepWiVe can
overcome the cliff-effect, which is prevalent in conventional separation-based
digital communication schemes, and achieve graceful degradation with the
mismatch between the estimated and actual channel qualities. DeepWiVe
outperforms H.264 video compression followed by low-density parity check (LDPC)
codes in all channel conditions by up to 0.0462 on average in terms of the
multi-scale structural similarity index measure (MS-SSIM), while beating H.265
+ LDPC by up to 0.0058 on average. We also illustrate the importance of
optimizing bandwidth allocation in JSCC video transmission by showing that our
optimal bandwidth allocation policy is superior to the naïve uniform
allocation. We believe this is an important step towards fulfilling the
potential of an end-to-end optimized JSCC wireless video transmission system
that is superior to the current separation-based designs.

    

### [[2111.13037] Learning dynamical systems from data: A simple cross-validation perspective, part III: Irregularly-Sampled Time Series](http://arxiv.org/abs/2111.13037)


  A simple and interpretable way to learn a dynamical system from data is to
interpolate its vector-field with a kernel. In particular, this strategy is
highly efficient (both in terms of accuracy and complexity) when the kernel is
data-adapted using Kernel Flows (KF)~\cite{Owhadi19} (which uses gradient-based
optimization to learn a kernel based on the premise that a kernel is good if
there is no significant loss in accuracy if half of the data is used for
interpolation). Despite its previous successes, this strategy (based on
interpolating the vector field driving the dynamical system) breaks down when
the observed time series is not regularly sampled in time. In this work, we
propose to address this problem by directly approximating the vector field of
the dynamical system by incorporating time differences between observations in
the (KF) data-adapted kernels. We compare our approach with the classical one
over different benchmark dynamical systems and show that it significantly
improves the forecasting accuracy while remaining simple, fast, and robust.

    

### [[2111.13042] DeepJSCC-Q: Channel Input Constrained Deep Joint Source-Channel Coding](http://arxiv.org/abs/2111.13042)


  Recent works have shown that the task of wireless transmission of images can
be learned with the use of machine learning techniques. Very promising results
in end-to-end image quality, superior to popular digital schemes that utilize
source and channel coding separation, have been demonstrated through the
training of an autoencoder, with a non-trainable channel layer in the middle.
However, these methods assume that any complex value can be transmitted over
the channel, which can prevent the application of the algorithm in scenarios
where the hardware or protocol can only admit certain sets of channel inputs,
such as the use of a digital constellation. Herein, we propose DeepJSCC-Q, an
end-to-end optimized joint source-channel coding scheme for wireless image
transmission, which is able to operate with a fixed channel input alphabet. We
show that DeepJSCC-Q can achieve similar performance to models that use
continuous-valued channel input. Importantly, it preserves the graceful
degradation of image quality observed in prior work when channel conditions
worsen, making DeepJSCC-Q much more attractive for deployment in practical
systems.

    

### [[2111.13069] Continual Active Learning Using Pseudo-Domains for Limited Labelling Resources and Changing Acquisition Characteristics](http://arxiv.org/abs/2111.13069)


  Machine learning in medical imaging during clinical routine is impaired by
changes in scanner protocols, hardware, or policies resulting in a
heterogeneous set of acquisition settings. When training a deep learning model
on an initial static training set, model performance and reliability suffer
from changes of acquisition characteristics as data and targets may become
inconsistent. Continual learning can help to adapt models to the changing
environment by training on a continuous data stream. However, continual manual
expert labelling of medical imaging requires substantial effort. Thus, ways to
use labelling resources efficiently on a well chosen sub-set of new examples is
necessary to render this strategy feasible.
Here, we propose a method for continual active learning operating on a stream
of medical images in a multi-scanner setting. The approach automatically
recognizes shifts in image acquisition characteristics - new domains -, selects
optimal examples for labelling and adapts training accordingly. Labelling is
subject to a limited budget, resembling typical real world scenarios. To
demonstrate generalizability, we evaluate the effectiveness of our method on
three tasks: cardiac segmentation, lung nodule detection and brain age
estimation. Results show that the proposed approach outperforms other active
learning methods, while effectively counteracting catastrophic forgetting.

    

### [[2111.13073] Neuronal Learning Analysis using Cycle-Consistent Adversarial Networks](http://arxiv.org/abs/2111.13073)


  Understanding how activity in neural circuits reshapes following task
learning could reveal fundamental mechanisms of learning. Thanks to the recent
advances in neural imaging technologies, high-quality recordings can be
obtained from hundreds of neurons over multiple days or even weeks. However,
the complexity and dimensionality of population responses pose significant
challenges for analysis. Existing methods of studying neuronal adaptation and
learning often impose strong assumptions on the data or model, resulting in
biased descriptions that do not generalize. In this work, we use a variant of
deep generative models called - CycleGAN, to learn the unknown mapping between
pre- and post-learning neural activities recorded $\textit{in vivo}$. We
develop an end-to-end pipeline to preprocess, train and evaluate calcium
fluorescence signals, and a procedure to interpret the resulting deep learning
models. To assess the validity of our method, we first test our framework on a
synthetic dataset with known ground-truth transformation. Subsequently, we
applied our method to neural activities recorded from the primary visual cortex
of behaving mice, where the mice transition from novice to expert-level
performance in a visual-based virtual reality experiment. We evaluate model
performance on generated calcium signals and their inferred spike trains. To
maximize performance, we derive a novel approach to pre-sort neurons such that
convolutional-based networks can take advantage of the spatial information that
exists in neural activities. In addition, we incorporate visual explanation
methods to improve the interpretability of our work and gain insights into the
learning process as manifested in the cellular activities. Together, our
results demonstrate that analyzing neuronal learning processes with data-driven
deep unsupervised methods holds the potential to unravel changes in an unbiased
way.

    

### [[2111.13075] Predicting the success of Gradient Descent for a particular Dataset-Architecture-Initialization (DAI)](http://arxiv.org/abs/2111.13075)


  Despite their massive success, training successful deep neural networks still
largely relies on experimentally choosing an architecture, hyper-parameters,
initialization, and training mechanism. In this work, we focus on determining
the success of standard gradient descent method for training deep neural
networks on a specified dataset, architecture, and initialization (DAI)
combination. Through extensive systematic experiments, we show that the
evolution of singular values of the matrix obtained from the hidden layers of a
DNN can aid in determining the success of gradient descent technique to train a
DAI, even in the absence of validation labels in the supervised learning
paradigm. This phenomenon can facilitate early give-up, stopping the training
of neural networks which are predicted to not generalize well, early in the
training process. Our experimentation across multiple datasets, architectures,
and initializations reveals that the proposed scores can more accurately
predict the success of a DAI than simply relying on the validation accuracy at
earlier epochs to make a judgment.

    

### [[2111.13089] GeomNet: A Neural Network Based on Riemannian Geometries of SPD Matrix Space and Cholesky Space for 3D Skeleton-Based Interaction Recognition](http://arxiv.org/abs/2111.13089)


  In this paper, we propose a novel method for representation and
classification of two-person interactions from 3D skeleton sequences. The key
idea of our approach is to use Gaussian distributions to capture statistics on
R n and those on the space of symmetric positive definite (SPD) matrices. The
main challenge is how to parametrize those distributions. Towards this end, we
develop methods for embedding Gaussian distributions in matrix groups based on
the theory of Lie groups and Riemannian symmetric spaces. Our method relies on
the Riemannian geometry of the underlying manifolds and has the advantage of
encoding high-order statistics from 3D joint positions. We show that the
proposed method achieves competitive results in two-person interaction
recognition on three benchmarks for 3D human activity understanding.

    

### [[2111.13108] Learning Debiased Models with Dynamic Gradient Alignment and Bias-conflicting Sample Mining](http://arxiv.org/abs/2111.13108)


  Deep neural networks notoriously suffer from dataset biases which are
detrimental to model robustness, generalization and fairness. In this work, we
propose a two-stage debiasing scheme to combat against the intractable unknown
biases. Starting by analyzing the factors of the presence of biased models, we
design a novel learning objective which cannot be reached by relying on biases
alone. Specifically, debiased models are achieved with the proposed Gradient
Alignment (GA) which dynamically balances the contributions of bias-aligned and
bias-conflicting samples (refer to samples with/without bias cues respectively)
throughout the whole training process, enforcing models to exploit intrinsic
cues to make fair decisions. While in real-world scenarios, the potential
biases are extremely hard to discover and prohibitively expensive to label
manually. We further propose an automatic bias-conflicting sample mining method
by peer-picking and training ensemble without prior knowledge of bias
information. Experiments conducted on multiple datasets in various settings
demonstrate the effectiveness and robustness of our proposed scheme, which
successfully alleviates the negative impact of unknown biases and achieves
state-of-the-art performance.

    

### [[2111.13110] QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking](http://arxiv.org/abs/2111.13110)


  QNNVerifier is the first open-source tool for verifying implementations of
neural networks that takes into account the finite word-length (i.e.
quantization) of their operands. The novel support for quantization is achieved
by employing state-of-the-art software model checking (SMC) techniques. It
translates the implementation of neural networks to a decidable fragment of
first-order logic based on satisfiability modulo theories (SMT). The effects of
fixed- and floating-point operations are represented through direct
implementations given a hardware-determined precision. Furthermore, QNNVerifier
allows to specify bespoke safety properties and verify the resulting model with
different verification strategies (incremental and k-induction) and SMT
solvers. Finally, QNNVerifier is the first tool that combines invariant
inference via interval analysis and discretization of non-linear activation
functions to speed up the verification of neural networks by orders of
magnitude. A video presentation of QNNVerifier is available at
this https URL


### [[2111.13119] Interesting Object, Curious Agent: Learning Task-Agnostic Exploration](http://arxiv.org/abs/2111.13119)


  Common approaches for task-agnostic exploration learn tabula-rasa --the agent
assumes isolated environments and no prior knowledge or experience. However, in
the real world, agents learn in many environments and always come with prior
experiences as they explore new ones. Exploration is a lifelong process. In
this paper, we propose a paradigm change in the formulation and evaluation of
task-agnostic exploration. In this setup, the agent first learns to explore
across many environments without any extrinsic goal in a task-agnostic manner.
Later on, the agent effectively transfers the learned exploration policy to
better explore new environments when solving tasks. In this context, we
evaluate several baseline exploration strategies and present a simple yet
effective approach to learning task-agnostic exploration policies. Our key idea
is that there are two components of exploration: (1) an agent-centric component
encouraging exploration of unseen parts of the environment based on an agent's
belief; (2) an environment-centric component encouraging exploration of
inherently interesting objects. We show that our formulation is effective and
provides the most consistent exploration across several training-testing
environment pairs. We also introduce benchmarks and metrics for evaluating
task-agnostic exploration strategies. The source code is available at
this https URL.

    

### [[2111.13129] Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models](http://arxiv.org/abs/2111.13129)


  A core challenge for an autonomous agent acting in the real world is to adapt
its repertoire of skills to cope with its noisy perception and dynamics. To
scale learning of skills to long-horizon tasks, robots should be able to learn
and later refine their skills in a structured manner through trajectories
rather than making instantaneous decisions individually at each time step. To
this end, we propose the Soft Actor-Critic Gaussian Mixture Model (SAC-GMM), a
novel hybrid approach that learns robot skills through a dynamical system and
adapts the learned skills in their own trajectory distribution space through
interactions with the environment. Our approach combines classical robotics
techniques of learning from demonstration with the deep reinforcement learning
framework and exploits their complementary nature. We show that our method
utilizes sensors solely available during the execution of preliminarily learned
skills to extract relevant features that lead to faster skill refinement.
Extensive evaluations in both simulation and real-world environments
demonstrate the effectiveness of our method in refining robot skills by
leveraging physical interactions, high-dimensional sensory data, and sparse
task completion rewards. Videos, code, and pre-trained models are available at
\url{this http URL}.

    

### [[2111.13131] Scene Graph Generation with Geometric Context](http://arxiv.org/abs/2111.13131)


  Scene Graph Generation has gained much attention in computer vision research
with the growing demand in image understanding projects like visual question
answering, image captioning, self-driving cars, crowd behavior analysis,
activity recognition, and more. Scene graph, a visually grounded graphical
structure of an image, immensely helps to simplify the image understanding
tasks. In this work, we introduced a post-processing algorithm called Geometric
Context to understand the visual scenes better geometrically. We use this
post-processing algorithm to add and refine the geometric relationships between
object pairs to a prior model. We exploit this context by calculating the
direction and distance between object pairs. We use Knowledge Embedded Routing
Network (KERN) as our baseline model, extend the work with our algorithm, and
show comparable results on the recent state-of-the-art algorithms.

    

### [[2111.13138] TunBERT: Pretrained Contextualized Text Representation for Tunisian Dialect](http://arxiv.org/abs/2111.13138)


  Pretrained contextualized text representation models learn an effective
representation of a natural language to make it machine understandable. After
the breakthrough of the attention mechanism, a new generation of pretrained
models have been proposed achieving good performances since the introduction of
the Transformer. Bidirectional Encoder Representations from Transformers (BERT)
has become the state-of-the-art model for language understanding. Despite their
success, most of the available models have been trained on Indo-European
languages however similar research for under-represented languages and dialects
remains sparse.
In this paper, we investigate the feasibility of training monolingual
Transformer-based language models for under represented languages, with a
specific focus on the Tunisian dialect. We evaluate our language model on
sentiment analysis task, dialect identification task and reading comprehension
question-answering task. We show that the use of noisy web crawled data instead
of structured data (Wikipedia, articles, etc.) is more convenient for such
non-standardized language. Moreover, results indicate that a relatively small
web crawled dataset leads to performances that are as good as those obtained
using larger datasets. Finally, our best performing TunBERT model reaches or
improves the state-of-the-art in all three downstream tasks. We release the
TunBERT pretrained model and the datasets used for fine-tuning.

    

### [[2111.13139] Group equivariant neural posterior estimation](http://arxiv.org/abs/2111.13139)


  Simulation-based inference with conditional neural density estimators is a
powerful approach to solving inverse problems in science. However, these
methods typically treat the underlying forward model as a black box, with no
way to exploit geometric properties such as equivariances. Equivariances are
common in scientific models, however integrating them directly into expressive
inference networks (such as normalizing flows) is not straightforward. We here
describe an alternative method to incorporate equivariances under joint
transformations of parameters and data. Our method -- called group equivariant
neural posterior estimation (GNPE) -- is based on self-consistently
standardizing the "pose" of the data while estimating the posterior over
parameters. It is architecture-independent, and applies both to exact and
approximate equivariances. As a real-world application, we use GNPE for
amortized inference of astrophysical binary black hole systems from
gravitational-wave observations. We show that GNPE achieves state-of-the-art
accuracy while reducing inference times by three orders of magnitude.

    

### [[2111.13142] Ontology-Based Skill Description Learning for Flexible Production Systems](http://arxiv.org/abs/2111.13142)


  The increasing importance of resource-efficient production entails that
manufacturing companies have to create a more dynamic production environment,
with flexible manufacturing machines and processes. To fully utilize this
potential of dynamic manufacturing through automatic production planning,
formal skill descriptions of the machines are essential. However, generating
those skill descriptions in a manual fashion is labor-intensive and requires
extensive domain-knowledge. In this contribution an ontology-based
semi-automatic skill description system that utilizes production logs and
industrial ontologies through inductive logic programming is introduced and
benefits and drawbacks of the proposed solution are evaluated.

    

### [[2111.13152] Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations](http://arxiv.org/abs/2111.13152)


  A classical problem in computer vision is to infer a 3D scene representation
from few images that can be used to render novel views at interactive rates.
Previous work focuses on reconstructing pre-defined 3D representations, e.g.
textured meshes, or implicit representations, e.g. radiance fields, and often
requires input images with precise camera poses and long processing times for
each novel scene.
In this work, we propose the Scene Representation Transformer (SRT), a method
which processes posed or unposed RGB images of a new area, infers a "set-latent
scene representation", and synthesises novel views, all in a single
feed-forward pass. To calculate the scene representation, we propose a
generalization of the Vision Transformer to sets of images, enabling global
information integration, and hence 3D reasoning. An efficient decoder
transformer parameterizes the light field by attending into the scene
representation to render novel views. Learning is supervised end-to-end by
minimizing a novel-view reconstruction error.
We show that this method outperforms recent baselines in terms of PSNR and
speed on synthetic datasets, including a new dataset created for the paper.
Further, we demonstrate that SRT scales to support interactive visualization
and semantic segmentation of real-world outdoor environments using Street View
imagery.

    

### [[2111.13154] Country-wide Retrieval of Forest Structure From Optical and SAR Satellite Imagery With Bayesian Deep Learning](http://arxiv.org/abs/2111.13154)


  Monitoring and managing Earth's forests in an informed manner is an important
requirement for addressing challenges like biodiversity loss and climate
change. While traditional in situ or aerial campaigns for forest assessments
provide accurate data for analysis at regional level, scaling them to entire
countries and beyond with high temporal resolution is hardly possible. In this
work, we propose a Bayesian deep learning approach to densely estimate forest
structure variables at country-scale with 10-meter resolution, using freely
available satellite imagery as input. Our method jointly transforms Sentinel-2
optical images and Sentinel-1 synthetic aperture radar images into maps of five
different forest structure variables: 95th height percentile, mean height,
density, Gini coefficient, and fractional cover. We train and test our model on
reference data from 41 airborne laser scanning missions across Norway and
demonstrate that it is able to generalize to unseen test regions, achieving
normalized mean absolute errors between 11% and 15%, depending on the variable.
Our work is also the first to propose a Bayesian deep learning approach so as
to predict forest structure variables with well-calibrated uncertainty
estimates. These increase the trustworthiness of the model and its suitability
for downstream tasks that require reliable confidence estimates, such as
informed decision making. We present an extensive set of experiments to
validate the accuracy of the predicted maps as well as the quality of the
predicted uncertainties. To demonstrate scalability, we provide Norway-wide
maps for the five forest structure variables.

    

### [[2111.13162] Randomized Stochastic Gradient Descent Ascent](http://arxiv.org/abs/2111.13162)


  An increasing number of machine learning problems, such as robust or
adversarial variants of existing algorithms, require minimizing a loss function
that is itself defined as a maximum. Carrying a loop of stochastic gradient
ascent (SGA) steps on the (inner) maximization problem, followed by an SGD step
on the (outer) minimization, is known as Epoch Stochastic Gradient
\textit{Descent Ascent} (ESGDA). While successful in practice, the theoretical
analysis of ESGDA remains challenging, with no clear guidance on choices for
the inner loop size nor on the interplay between inner/outer step sizes. We
propose RSGDA (Randomized SGDA), a variant of ESGDA with stochastic loop size
with a simpler theoretical analysis. RSGDA comes with the first (among SGDA
algorithms) almost sure convergence rates when used on nonconvex
min/strongly-concave max settings. RSGDA can be parameterized using optimal
loop sizes that guarantee the best convergence rates known to hold for SGDA. We
test RSGDA on toy and larger scale problems, using distributionally robust
optimization and single-cell data matching using optimal transport as a
testbed.

    

### [[2111.13164] Time Series Forecasting with Ensembled Stochastic Differential Equations Driven by Lévy Noise](http://arxiv.org/abs/2111.13164)


  With the fast development of modern deep learning techniques, the study of
dynamic systems and neural networks is increasingly benefiting each other in a
lot of different ways. Since uncertainties often arise in real world
observations, SDEs (stochastic differential equations) come to play an
important role. To be more specific, in this paper, we use a collection of SDEs
equipped with neural networks to predict long-term trend of noisy time series
which has big jump properties and high probability distribution shift. Our
contributions are, first, we use the phase space reconstruction method to
extract intrinsic dimension of the time series data so as to determine the
input structure for our forecasting model. Second, we explore SDEs driven by
$\alpha$-stable Lévy motion to model the time series data and solve the
problem through neural network approximation. Third, we construct the attention
mechanism to achieve multi-time step prediction. Finally, we illustrate our
method by applying it to stock marketing time series prediction and show the
results outperform several baseline deep learning models.

    

### [[2111.13171] Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks](http://arxiv.org/abs/2111.13171)


  Disobeying the classical wisdom of statistical learning theory, modern deep
neural networks generalize well even though they typically contain millions of
parameters. Recently, it has been shown that the trajectories of iterative
optimization algorithms can possess fractal structures, and their
generalization error can be formally linked to the complexity of such fractals.
This complexity is measured by the fractal's intrinsic dimension, a quantity
usually much smaller than the number of parameters in the network. Even though
this perspective provides an explanation for why overparametrized networks
would not overfit, computing the intrinsic dimension (e.g., for monitoring
generalization during training) is a notoriously difficult task, where existing
methods typically fail even in moderate ambient dimensions. In this study, we
consider this problem from the lens of topological data analysis (TDA) and
develop a generic computational tool that is built on rigorous mathematical
foundations. By making a novel connection between learning theory and TDA, we
first illustrate that the generalization error can be equivalently bounded in
terms of a notion called the 'persistent homology dimension' (PHD), where,
compared with prior work, our approach does not require any additional
geometrical or statistical assumptions on the training dynamics. Then, by
utilizing recently established theoretical results and TDA tools, we develop an
efficient algorithm to estimate PHD in the scale of modern deep neural networks
and further provide visualization tools to help understand generalization in
deep learning. Our experiments show that the proposed approach can efficiently
compute a network's intrinsic dimension in a variety of settings, which is
predictive of the generalization error.

    

### [[2111.13176] Computer Vision User Entity Behavior Analytics](http://arxiv.org/abs/2111.13176)


  Insider threats are costly, hard to detect, and unfortunately rising in
occurrence. Seeking to improve detection of such threats, we develop novel
techniques to enable us to extract powerful features, generate high quality
image encodings, and augment attack vectors for greater classification power.
Combined, they form Computer Vision User and Entity Behavior Analytics, a
detection system designed from the ground up to improve upon advancements in
academia and mitigate the issues that prevent the usage of advanced models in
industry. The proposed system beats state-of-art methods used in academia and
as well as in industry.

    

### [[2111.13180] Variational Gibbs inference for statistical model estimation from incomplete data](http://arxiv.org/abs/2111.13180)


  Statistical models are central to machine learning with broad applicability
across a range of downstream tasks. The models are typically controlled by free
parameters that are estimated from data by maximum-likelihood estimation.
However, when faced with real-world datasets many of the models run into a
critical issue: they are formulated in terms of fully-observed data, whereas in
practice the datasets are plagued with missing data. The theory of statistical
model estimation from incomplete data is conceptually similar to the estimation
of latent-variable models, where powerful tools such as variational inference
(VI) exist. However, in contrast to standard latent-variable models, parameter
estimation with incomplete data often requires estimating exponentially-many
conditional distributions of the missing variables, hence making standard VI
methods intractable. We address this gap by introducing variational Gibbs
inference (VGI), a new general-purpose method to estimate the parameters of
statistical models from incomplete data. We validate VGI on a set of synthetic
and real-world estimation tasks, estimating important machine learning models,
VAEs and normalising flows, from incomplete data. The proposed method, whilst
general-purpose, achieves competitive or better performance than existing
model-specific estimation methods.

    

### [[2111.13185] Learning Conditional Invariance through Cycle Consistency](http://arxiv.org/abs/2111.13185)


  Identifying meaningful and independent factors of variation in a dataset is a
challenging learning task frequently addressed by means of deep latent variable
models. This task can be viewed as learning symmetry transformations preserving
the value of a chosen property along latent dimensions. However, existing
approaches exhibit severe drawbacks in enforcing the invariance property in the
latent space. We address these shortcomings with a novel approach to cycle
consistency. Our method involves two separate latent subspaces for the target
property and the remaining input information, respectively. In order to enforce
invariance as well as sparsity in the latent space, we incorporate semantic
knowledge by using cycle consistency constraints relying on property side
information. The proposed method is based on the deep information bottleneck
and, in contrast to other approaches, allows using continuous target properties
and provides inherent model selection capabilities. We demonstrate on synthetic
and molecular data that our approach identifies more meaningful factors which
lead to sparser and more interpretable models with improved invariance
properties.

    

### [[2111.13186] Federated Data Science to Break Down Silos [Vision]](http://arxiv.org/abs/2111.13186)


  Similar to Open Data initiatives, data science as a community has launched
initiatives for sharing not only data but entire pipelines, derivatives,
artifacts, etc. (Open Data Science). However, the few efforts that exist focus
on the technical part on how to facilitate sharing, conversion, etc. This
vision paper goes a step further and proposes KEK, an open federated data
science platform that does not only allow for sharing data science pipelines
and their (meta)data but also provides methods for efficient search and, in the
ideal case, even allows for combining and defining pipelines across platforms
in a federated manner. In doing so, KEK addresses the so far neglected
challenge of actually finding artifacts that are semantically related and that
can be combined to achieve a certain goal.

    

### [[2111.13187] Information Bottleneck-Based Hebbian Learning Rule Naturally Ties Working Memory and Synaptic Updates](http://arxiv.org/abs/2111.13187)


  Artificial neural networks have successfully tackled a large variety of
problems by training extremely deep networks via back-propagation. A direct
application of back-propagation to spiking neural networks contains
biologically implausible components, like the weight transport problem or
separate inference and learning phases. Various methods address different
components individually, but a complete solution remains intangible. Here, we
take an alternate approach that avoids back-propagation and its associated
issues entirely. Recent work in deep learning proposed independently training
each layer of a network via the information bottleneck (IB). Subsequent studies
noted that this layer-wise approach circumvents error propagation across
layers, leading to a biologically plausible paradigm. Unfortunately, the IB is
computed using a batch of samples. The prior work addresses this with a weight
update that only uses two samples (the current and previous sample). Our work
takes a different approach by decomposing the weight update into a local and
global component. The local component is Hebbian and only depends on the
current sample. The global component computes a layer-wise modulatory signal
that depends on a batch of samples. We show that this modulatory signal can be
learned by an auxiliary circuit with working memory (WM) like a reservoir.
Thus, we can use batch sizes greater than two, and the batch size determines
the required capacity of the WM. To the best of our knowledge, our rule is the
first biologically plausible mechanism to directly couple synaptic updates with
a WM of the task. We evaluate our rule on synthetic datasets and image
classification datasets like MNIST, and we explore the effect of the WM
capacity on learning performance. We hope our work is a first-step towards
understanding the mechanistic role of memory in learning.

    

### [[2111.13204] BaLeNAS: Differentiable Architecture Search via the Bayesian Learning Rule](http://arxiv.org/abs/2111.13204)


  Differentiable Architecture Search (DARTS) has received massive attention in
recent years, mainly because it significantly reduces the computational cost
through weight sharing and continuous relaxation. However, more recent works
find that existing differentiable NAS techniques struggle to outperform naive
baselines, yielding deteriorative architectures as the search proceeds. Rather
than directly optimizing the architecture parameters, this paper formulates the
neural architecture search as a distribution learning problem through relaxing
the architecture weights into Gaussian distributions. By leveraging the
natural-gradient variational inference (NGVI), the architecture distribution
can be easily optimized based on existing codebases without incurring more
memory and computational consumption. We demonstrate how the differentiable NAS
benefits from Bayesian principles, enhancing exploration and improving
stability. The experimental results on NAS-Bench-201 and NAS-Bench-1shot1
benchmark datasets confirm the significant improvements the proposed framework
can make. In addition, instead of simply applying the argmax on the learned
parameters, we further leverage the recently-proposed training-free proxies in
NAS to select the optimal architecture from a group architectures drawn from
the optimized distribution, where we achieve state-of-the-art results on the
NAS-Bench-201 and NAS-Bench-1shot1 benchmarks. Our best architecture in the
DARTS search space also obtains competitive test errors with 2.37\%, 15.72\%,
and 24.2\% on CIFAR-10, CIFAR-100, and ImageNet datasets, respectively.

    

### [[2111.13207] Characteristic Neural Ordinary Differential Equations](http://arxiv.org/abs/2111.13207)


  We propose Characteristic Neural Ordinary Differential Equations (C-NODEs), a
framework for extending Neural Ordinary Differential Equations (NODEs) beyond
ODEs. While NODEs model the evolution of the latent state as the solution to an
ODE, the proposed C-NODE models the evolution of the latent state as the
solution of a family of first-order quasi-linear partial differential equations
(PDE) on their characteristics, defined as curves along which the PDEs reduce
to ODEs. The reduction, in turn, allows the application of the standard
frameworks for solving ODEs to PDE settings. Additionally, the proposed
framework can be cast as an extension of existing NODE architectures, thereby
allowing the use of existing black-box ODE solvers. We prove that the C-NODE
framework extends the classical NODE by exhibiting functions that cannot be
represented by NODEs but are representable by C-NODEs. We further investigate
the efficacy of the C-NODE framework by demonstrating its performance in many
synthetic and real data scenarios. Empirical results demonstrate the
improvements provided by the proposed method for CIFAR-10, SVHN, and MNIST
datasets under a similar computational budget as the existing NODE methods.

    

### [[2111.13208] Evaluation of Interpretability for Deep Learning algorithms in EEG Emotion Recognition: A case study in Autism](http://arxiv.org/abs/2111.13208)


  Current models on Explainable Artificial Intelligence (XAI) have shown an
evident and quantified lack of reliability for measuring feature-relevance when
statistically entangled features are proposed for training deep classifiers.
There has been an increase in the application of Deep Learning in clinical
trials to predict early diagnosis of neuro-developmental disorders, such as
Autism Spectrum Disorder (ASD). However, the inclusion of more reliable
saliency-maps to obtain more trustworthy and interpretable metrics using neural
activity features is still insufficiently mature for practical applications in
diagnostics or clinical trials. Moreover, in ASD research the inclusion of deep
classifiers that use neural measures to predict viewed facial emotions is
relatively unexplored. Therefore, in this study we propose the evaluation of a
Convolutional Neural Network (CNN) for electroencephalography (EEG)-based
facial emotion recognition decoding complemented with a novel
RemOve-And-Retrain (ROAR) methodology to recover highly relevant features used
in the classifier. Specifically, we compare well-known relevance maps such as
Layer-Wise Relevance Propagation (LRP), PatternNet, Pattern Attribution, and
Smooth-Grad Squared. This study is the first to consolidate a more transparent
feature-relevance calculation for a successful EEG-based facial emotion
recognition using a within-subject-trained CNN in typically-developed and ASD
individuals.

    

### [[2111.13209] Mitigating Noise-Induced Gradient Vanishing in Variational Quantum Algorithm Training](http://arxiv.org/abs/2111.13209)


  Variational quantum algorithms are expected to demonstrate the advantage of
quantum computing on near-term noisy quantum computers. However, training such
variational quantum algorithms suffers from gradient vanishing as the size of
the algorithm increases. Previous work cannot handle the gradient vanishing
induced by the inevitable noise effects on realistic quantum hardware. In this
paper, we propose a novel training scheme to mitigate such noise-induced
gradient vanishing. We first introduce a new cost function of which the
gradients are significantly augmented by employing traceless observables in
truncated subspace. We then prove that the same minimum can be reached by
optimizing the original cost function with the gradients from the new cost
function. Experiments show that our new training scheme is highly effective for
major variational quantum algorithms of various tasks.

    

### [[2111.13213] OTB-morph: One-Time Biometrics via Morphing applied to Face Templates](http://arxiv.org/abs/2111.13213)


  Cancelable biometrics refers to a group of techniques in which the biometric
inputs are transformed intentionally using a key before processing or storage.
This transformation is repeatable enabling subsequent biometric comparisons.
This paper introduces a new scheme for cancelable biometrics aimed at
protecting the templates against potential attacks, applicable to any
biometric-based recognition system. Our proposed scheme is based on
time-varying keys obtained from morphing random biometric information. An
experimental implementation of the proposed scheme is given for face
biometrics. The results confirm that the proposed approach is able to withstand
against leakage attacks while improving the recognition performance.

    

### [[2111.13219] DP-SEP! Differentially Private Stochastic Expectation Propagation](http://arxiv.org/abs/2111.13219)


  We are interested in privatizing an approximate posterior inference algorithm
called Expectation Propagation (EP). EP approximates the posterior by
iteratively refining approximations to the local likelihoods, and is known to
provide better posterior uncertainties than those by variational inference
(VI). However, using EP for large-scale datasets imposes a challenge in terms
of memory requirements as it needs to maintain each of the local approximates
in memory. To overcome this problem, stochastic expectation propagation (SEP)
was proposed, which only considers a unique local factor that captures the
average effect of each likelihood term to the posterior and refines it in a way
analogous to EP. In terms of privacy, SEP is more tractable than EP because at
each refining step of a factor, the remaining factors are fixed to the same
value and do not depend on other datapoints as in EP, which makes the
sensitivity analysis tractable. We provide a theoretical analysis of the
privacy-accuracy trade-off in the posterior estimates under differentially
private stochastic expectation propagation (DP-SEP). Furthermore, we
demonstrate the performance of our DP-SEP algorithm evaluated on both synthetic
and real-world datasets in terms of the quality of posterior estimates at
different levels of guaranteed privacy.

    

### [[2111.13229] Generalizing Clinical Trials with Convex Hulls](http://arxiv.org/abs/2111.13229)


  Randomized clinical trials eliminate confounding but impose strict exclusion
criteria that limit recruitment to a subset of the population. Observational
datasets are more inclusive but suffer from confounding -- often providing
overly optimistic estimates of treatment effect in practice. We therefore
assume that the true treatment effect lies somewhere in between no treatment
effect and the observational estimate, or in their convex hull. This assumption
allows us to extrapolate results from exclusive trials to the broader
population by analyzing observational and trial data simultaneously using an
algorithm called Optimal Convex Hulls (OCH). OCH represents the treatment
effect either in terms of convex hulls of conditional expectations or convex
hulls (also known as mixtures) of conditional densities. The algorithm first
learns the component expectations or densities using the observational data and
then learns the linear mixing coefficients using trial data in order to
approximate the true treatment effect; theory importantly explains why this
linear combination should hold. OCH estimates the treatment effect in terms
both expectations and densities with state of the art accuracy.

    

### [[2111.13233] Look at here : Utilizing supervision to attend subtle key regions](http://arxiv.org/abs/2111.13233)


  Despite the success of deep learning in computer vision, algorithms to
recognize subtle and small objects (or regions) is still challenging. For
example, recognizing a baseball or a frisbee on a ground scene or a bone
fracture in an X-ray image can easily result in overfitting, unless a huge
amount of training data is available. To mitigate this problem, we need a way
to force a model should identify subtle regions in limited training data. In
this paper, we propose a simple but efficient supervised augmentation method
called Cut\&Remain. It achieved better performance on various medical image
domain (internally sourced- and public dataset) and a natural image domain
(MS-COCO$_s$) than other supervised augmentation and the explicit guidance
methods.
In addition, using the class activation map, we identified that the
Cut\&Remain methods drive a model to focus on relevant subtle and small regions
efficiently. We also show that the performance monotonically increased along
the Cut\&Remain ratio, indicating that a model can be improved even though only
limited amount of Cut\&Remain is applied for, so that it allows low supervising
(annotation) cost for improvement.

    

### [[2111.13236] Joint inference and input optimization in equilibrium networks](http://arxiv.org/abs/2111.13236)


  Many tasks in deep learning involve optimizing over the \emph{inputs} to a
network to minimize or maximize some objective; examples include optimization
over latent spaces in a generative model to match a target image, or
adversarially perturbing an input to worsen classifier performance. Performing
such optimization, however, is traditionally quite costly, as it involves a
complete forward and backward pass through the network for each gradient step.
In a separate line of work, a recent thread of research has developed the deep
equilibrium (DEQ) model, a class of models that foregoes traditional network
depth and instead computes the output of a network by finding the fixed point
of a single nonlinear layer. In this paper, we show that there is a natural
synergy between these two settings. Although, naively using DEQs for these
optimization problems is expensive (owing to the time needed to compute a fixed
point for each gradient step), we can leverage the fact that gradient-based
optimization can \emph{itself} be cast as a fixed point iteration to
substantially improve the overall speed. That is, we \emph{simultaneously} both
solve for the DEQ fixed point \emph{and} optimize over network inputs, all
within a single ``augmented'' DEQ model that jointly encodes both the original
network and the optimization process. Indeed, the procedure is fast enough that
it allows us to efficiently \emph{train} DEQ models for tasks traditionally
relying on an ``inner'' optimization loop. We demonstrate this strategy on
various tasks such as training generative models while optimizing over latent
codes, training models for inverse problems like denoising and inpainting,
adversarial training and gradient based meta-learning.

    

### [[2111.13273] Unsupervised Feature Ranking via Attribute Networks](http://arxiv.org/abs/2111.13273)


  The need for learning from unlabeled data is increasing in contemporary
machine learning. Methods for unsupervised feature ranking, which identify the
most important features in such data are thus gaining attention, and so are
their applications in studying high throughput biological experiments or user
bases for recommender systems. We propose FRANe (Feature Ranking via Attribute
Networks), an unsupervised algorithm capable of finding key features in given
unlabeled data set. FRANe is based on ideas from network reconstruction and
network analysis. FRANe performs better than state-of-the-art competitors, as
we empirically demonstrate on a large collection of benchmarks. Moreover, we
provide the time complexity analysis of FRANe further demonstrating its
scalability. Finally, FRANe offers as the result the interpretable relational
structures used to derive the feature importances.

    

### [[2111.13282] Generative Adversarial Networks and Adversarial Autoencoders: Tutorial and Survey](http://arxiv.org/abs/2111.13282)


  This is a tutorial and survey paper on Generative Adversarial Network (GAN),
adversarial autoencoders, and their variants. We start with explaining
adversarial learning and the vanilla GAN. Then, we explain the conditional GAN
and DCGAN. The mode collapse problem is introduced and various methods,
including minibatch GAN, unrolled GAN, BourGAN, mixture GAN, D2GAN, and
Wasserstein GAN, are introduced for resolving this problem. Then, maximum
likelihood estimation in GAN are explained along with f-GAN, adversarial
variational Bayes, and Bayesian GAN. Then, we cover feature matching in GAN,
InfoGAN, GRAN, LSGAN, energy-based GAN, CatGAN, MMD GAN, LapGAN, progressive
GAN, triple GAN, LAG, GMAN, AdaGAN, CoGAN, inverse GAN, BiGAN, ALI, SAGAN,
Few-shot GAN, SinGAN, and interpolation and evaluation of GAN. Then, we
introduce some applications of GAN such as image-to-image translation
(including PatchGAN, CycleGAN, DeepFaceDrawing, simulated GAN, interactive
GAN), text-to-image translation (including StackGAN), and mixing image
characteristics (including FineGAN and MixNMatch). Finally, we explain the
autoencoders based on adversarial learning including adversarial autoencoder,
PixelGAN, and implicit autoencoder.

    

### [[2111.13293] KNAS: Green Neural Architecture Search](http://arxiv.org/abs/2111.13293)


  Many existing neural architecture search (NAS) solutions rely on downstream
training for architecture evaluation, which takes enormous computations.
Considering that these computations bring a large carbon footprint, this paper
aims to explore a green (namely environmental-friendly) NAS solution that
evaluates architectures without training. Intuitively, gradients, induced by
the architecture itself, directly decide the convergence and generalization
results. It motivates us to propose the gradient kernel hypothesis: Gradients
can be used as a coarse-grained proxy of downstream training to evaluate
random-initialized networks. To support the hypothesis, we conduct a
theoretical analysis and find a practical gradient kernel that has good
correlations with training loss and validation performance. According to this
hypothesis, we propose a new kernel based architecture search approach KNAS.
Experiments show that KNAS achieves competitive results with orders of
magnitude faster than "train-then-test" paradigms on image classification
tasks. Furthermore, the extremely low search cost enables its wide
applications. The searched network also outperforms strong baseline
RoBERTA-large on two text classification tasks. Codes are available at
\url{this https URL} .

    

### [[2111.13296] Approximate Bayesian Computation for Physical Inverse Modeling](http://arxiv.org/abs/2111.13296)


  Semiconductor device models are essential to understand the charge transport
in thin film transistors (TFTs). Using these TFT models to draw inference
involves estimating parameters used to fit to the experimental data. These
experimental data can involve extracted charge carrier mobility or measured
current. Estimating these parameters help us draw inferences about device
performance. Fitting a TFT model for a given experimental data using the model
parameters relies on manual fine tuning of multiple parameters by human
experts. Several of these parameters may have confounding effects on the
experimental data, making their individual effect extraction a non-intuitive
process during manual tuning. To avoid this convoluted process, we propose a
new method for automating the model parameter extraction process resulting in
an accurate model fitting. In this work, model choice based approximate
Bayesian computation (aBc) is used for generating the posterior distribution of
the estimated parameters using observed mobility at various gate voltage
values. Furthermore, it is shown that the extracted parameters can be
accurately predicted from the mobility curves using gradient boosted trees.
This work also provides a comparative analysis of the proposed framework with
fine-tuned neural networks wherein the proposed framework is shown to perform
better.

    

### [[2111.13297] Latent Space based Memory Replay for Continual Learning in Artificial Neural Networks](http://arxiv.org/abs/2111.13297)


  Memory replay may be key to learning in biological brains, which manage to
learn new tasks continually without catastrophically interfering with previous
knowledge. On the other hand, artificial neural networks suffer from
catastrophic forgetting and tend to only perform well on tasks that they were
recently trained on. In this work we explore the application of latent space
based memory replay for classification using artificial neural networks. We are
able to preserve good performance in previous tasks by storing only a small
percentage of the original data in a compressed latent space version.

    

### [[2111.13299] Exploiting full Resolution Feature Context for Liver Tumor and Vessel Segmentation via Fusion Encoder: Application to Liver Tumor and Vessel 3D reconstruction](http://arxiv.org/abs/2111.13299)


  Liver cancer is one of the most common malignant diseases in the world.
Segmentation and labeling of liver tumors and blood vessels in CT images can
provide convenience for doctors in liver tumor diagnosis and surgical
intervention. In the past decades, automatic CT segmentation methods based on
deep learning have received widespread attention in the medical field. Many
state-of-the-art segmentation algorithms appeared during this period. Yet, most
of the existing segmentation methods only care about the local feature context
and have a perception defect in the global relevance of medical images, which
significantly affects the segmentation effect of liver tumors and blood
vessels. We introduce a multi-scale feature context fusion network called
TransFusionNet based on Transformer and SEBottleNet. This network can
accurately detect and identify the details of the region of interest of the
liver vessel, meanwhile it can improve the recognition of morphologic margins
of liver tumors by exploiting the global information of CT images. Experiments
show that TransFusionNet is better than the state-of-the-art method on both the
public dataset LITS and 3Dircadb and our clinical dataset. Finally, we propose
an automatic 3D reconstruction algorithm based on the trained model. The
algorithm can complete the reconstruction quickly and accurately in 1 second.

    

### [[2111.13311] Blaschke Product Neural Networks (BPNN): A Physics-Infused Neural Network for Phase Retrieval of Meromorphic Functions](http://arxiv.org/abs/2111.13311)


  Numerous physical systems are described by ordinary or partial differential
equations whose solutions are given by holomorphic or meromorphic functions in
the complex domain. In many cases, only the magnitude of these functions are
observed on various points on the purely imaginary jw-axis since coherent
measurement of their phases is often expensive. However, it is desirable to
retrieve the lost phases from the magnitudes when possible. To this end, we
propose a physics-infused deep neural network based on the Blaschke products
for phase retrieval. Inspired by the Helson and Sarason Theorem, we recover
coefficients of a rational function of Blaschke products using a Blaschke
Product Neural Network (BPNN), based upon the magnitude observations as input.
The resulting rational function is then used for phase retrieval. We compare
the BPNN to conventional deep neural networks (NNs) on several phase retrieval
problems, comprising both synthetic and contemporary real-world problems (e.g.,
metamaterials for which data collection requires substantial expertise and is
time consuming). On each phase retrieval problem, we compare against a
population of conventional NNs of varying size and hyperparameter settings.
Even without any hyper-parameter search, we find that BPNNs consistently
outperform the population of optimized NNs in scarce data scenarios, and do so
despite being much smaller models. The results can in turn be applied to
calculate the refractive index of metamaterials, which is an important problem
in emerging areas of material science.

    

### [[2111.13314] Amercing: An Intuitive, Elegant and Effective Constraint for Dynamic Time Warping](http://arxiv.org/abs/2111.13314)


  Dynamic Time Warping (DTW), and its constrained (CDTW) and weighted (WDTW)
variants, are time series distances with a wide range of applications. They
minimize the cost of non-linear alignments between series. CDTW and WDTW have
been introduced because DTW is too permissive in its alignments. However, CDTW
uses a crude step function, allowing unconstrained flexibility within the
window, and none beyond it. WDTW's multiplicative weight is relative to the
distances between aligned points along a warped path, rather than being a
direct function of the amount of warping that is introduced. In this paper, we
introduce Amerced Dynamic Time Warping (ADTW), a new, intuitive, DTW variant
that penalizes the act of warping by a fixed additive cost. Like CDTW and WDTW,
ADTW constrains the amount of warping. However, it avoids both abrupt
discontinuities in the amount of warping allowed and the limitations of a
multiplicative penalty. We formally introduce ADTW, prove some of its
properties, and discuss its parameterization. We show on a simple example how
it can be parameterized to achieve an intuitive outcome, and demonstrate its
usefulness on a standard time series classification benchmark. We provide a
demonstration application in C++.

    

### [[2111.13321] Learning source-aware representations of music in a discrete latent space](http://arxiv.org/abs/2111.13321)


  In recent years, neural network based methods have been proposed as a method
that cangenerate representations from music, but they are not human readable
and hardly analyzable oreditable by a human. To address this issue, we propose
a novel method to learn source-awarelatent representations of music through
Vector-Quantized Variational Auto-Encoder(VQ-VAE).We train our VQ-VAE to encode
an input mixture into a tensor of integers in a discrete latentspace, and
design them to have a decomposed structure which allows humans to manipulatethe
latent vector in a source-aware manner. This paper also shows that we can
generate basslines by estimating latent vectors in a discrete space.

    

### [[2111.13322] Random-reshuffled SARAH does not need a full gradient computations](http://arxiv.org/abs/2111.13322)


  The StochAstic Recursive grAdient algoritHm (SARAH) algorithm is a variance
reduced variant of the Stochastic Gradient Descent (SGD) algorithm that needs a
gradient of the objective function from time to time. In this paper, we remove
the necessity of a full gradient computation. This is achieved by using a
randomized reshuffling strategy and aggregating stochastic gradients obtained
in each epoch. The aggregated stochastic gradients serve as an estimate of a
full gradient in the SARAH algorithm. We provide a theoretical analysis of the
proposed approach and conclude the paper with numerical experiments that
demonstrate the efficiency of this approach.

    

### [[2111.13330] ArchRepair: Block-Level Architecture-Oriented Repairing for Deep Neural Networks](http://arxiv.org/abs/2111.13330)


  Over the past few years, deep neural networks (DNNs) have achieved tremendous
success and have been continuously applied in many application domains.
However, during the practical deployment in the industrial tasks, DNNs are
found to be erroneous-prone due to various reasons such as overfitting, lacking
robustness to real-world corruptions during practical usage. To address these
challenges, many recent attempts have been made to repair DNNs for version
updates under practical operational contexts by updating weights (i.e., network
parameters) through retraining, fine-tuning, or direct weight fixing at a
neural level. In this work, as the first attempt, we initiate to repair DNNs by
jointly optimizing the architecture and weights at a higher (i.e., block)
level.
We first perform empirical studies to investigate the limitation of whole
network-level and layer-level repairing, which motivates us to explore a novel
repairing direction for DNN repair at the block level. To this end, we first
propose adversarial-aware spectrum analysis for vulnerable block localization
that considers the neurons' status and weights' gradients in blocks during the
forward and backward processes, which enables more accurate candidate block
localization for repairing even under a few examples. Then, we further propose
the architecture-oriented search-based repairing that relaxes the targeted
block to a continuous repairing search space at higher deep feature levels. By
jointly optimizing the architecture and weights in that space, we can identify
a much better block architecture. We implement our proposed repairing
techniques as a tool, named ArchRepair, and conduct extensive experiments to
validate the proposed method. The results show that our method can not only
repair but also enhance accuracy & robustness, outperforming the
state-of-the-art DNN repair techniques.

    

### [[2111.13331] Implicit Data-Driven Regularization in Deep Neural Networks under SGD](http://arxiv.org/abs/2111.13331)


  Much research effort has been devoted to explaining the success of deep
learning. Random Matrix Theory (RMT) provides an emerging way to this end:
spectral analysis of large random matrices involved in a trained deep neural
network (DNN) such as weight matrices or Hessian matrices with respect to the
stochastic gradient descent algorithm. In this paper, we conduct extensive
experiments on weight matrices in different modules, e.g., layers, networks and
data sets, to analyze the evolution of their spectra. We find that these
spectra can be classified into three main types: Marčenko-Pastur spectrum
(MP), Marčenko-Pastur spectrum with few bleeding outliers (MPB), and Heavy
tailed spectrum (HT). Moreover, these discovered spectra are directly connected
to the degree of regularization in the DNN. We argue that the degree of
regularization depends on the quality of data fed to the DNN, namely
Data-Driven Regularization. These findings are validated in several NNs, using
Gaussian synthetic data and real data sets (MNIST and CIFAR10). Finally, we
propose a spectral criterion and construct an early stopping procedure when the
NN is found highly regularized without test data by using the connection
between the spectra types and the degrees of regularization. Such early stopped
DNNs avoid unnecessary extra training while preserving a much comparable
generalization ability.

    

### [[2111.13332] Testability-Aware Low Power Controller Design with Evolutionary Learning](http://arxiv.org/abs/2111.13332)


  XORNet-based low power controller is a popular technique to reduce circuit
transitions in scan-based testing. However, existing solutions construct the
XORNet evenly for scan chain control, and it may result in sub-optimal
solutions without any design guidance. In this paper, we propose a novel
testability-aware low power controller with evolutionary learning. The XORNet
generated from the proposed genetic algorithm (GA) enables adaptive control for
scan chains according to their usages, thereby significantly improving XORNet
encoding capacity, reducing the number of failure cases with ATPG and
decreasing test data volume. Experimental results indicate that under the same
control bits, our GA-guided XORNet design can improve the fault coverage by up
to 2.11%. The proposed GA-guided XORNets also allows reducing the number of
control bits, and the total testing time decreases by 20.78% on average and up
to 47.09% compared to the existing design without sacrificing test coverage.

    

### [[2111.13346] A multitask transfer learning framework for the prediction of virus-human protein-protein interactions](http://arxiv.org/abs/2111.13346)


  Viral infections are causing significant morbidity and mortality worldwide.
Understanding the interaction patterns between a particular virus and human
proteins plays a crucial role in unveiling the underlying mechanism of viral
infection and pathogenesis. This could further help in the prevention and
treatment of virus-related diseases. However, the task of predicting
protein-protein interactions between a new virus and human cells is extremely
challenging due to scarce data on virus-human interactions and fast mutation
rates of most viruses.
We developed a multitask transfer learning approach that exploits the
information of around 24 million protein sequences and the interaction patterns
from the human interactome to counter the problem of small training datasets.
Instead of using hand-crafted protein features, we utilize statistically rich
protein representations learned by a deep language modeling approach from a
massive source of protein sequences. Additionally, we employ an additional
objective which aims to maximize the probability of observing human
protein-protein interactions. This additional task objective acts as a
regularizer and also allows to incorporate domain knowledge to inform the
virus-human protein-protein interaction prediction model.
Our approach achieved competitive results on 13 benchmark datasets and the
case study for the SAR-CoV-2 virus receptor. Experimental results show that our
proposed model works effectively for both virus-human and bacteria-human
protein-protein interaction prediction tasks. We share our code for
reproducibility and future research at
this https URL.

    

### [[2111.13350] Jointly Learning Agent and Lane Information for Multimodal Trajectory Prediction](http://arxiv.org/abs/2111.13350)


  Predicting the plausible future trajectories of nearby agents is a core
challenge for the safety of Autonomous Vehicles and it mainly depends on two
external cues: the dynamic neighbor agents and static scene context. Recent
approaches have made great progress in characterizing the two cues separately.
However, they ignore the correlation between the two cues and most of them are
difficult to achieve map-adaptive prediction. In this paper, we use lane as
scene data and propose a staged network that Jointly learning Agent and Lane
information for Multimodal Trajectory Prediction (JAL-MTP). JAL-MTP use a
Social to Lane (S2L) module to jointly represent the static lane and the
dynamic motion of the neighboring agents as instance-level lane, a Recurrent
Lane Attention (RLA) mechanism for utilizing the instance-level lanes to
predict the map-adaptive future trajectories and two selectors to identify the
typical and reasonable trajectories. The experiments conducted on the public
Argoverse dataset demonstrate that JAL-MTP significantly outperforms the
existing models in both quantitative and qualitative.

    

### [[2111.13361] Geometric Multimodal Deep Learning with Multi-Scaled Graph Wavelet Convolutional Network](http://arxiv.org/abs/2111.13361)


  Multimodal data provide complementary information of a natural phenomenon by
integrating data from various domains with very different statistical
properties. Capturing the intra-modality and cross-modality information of
multimodal data is the essential capability of multimodal learning methods. The
geometry-aware data analysis approaches provide these capabilities by
implicitly representing data in various modalities based on their geometric
underlying structures. Also, in many applications, data are explicitly defined
on an intrinsic geometric structure. Generalizing deep learning methods to the
non-Euclidean domains is an emerging research field, which has recently been
investigated in many studies. Most of those popular methods are developed for
unimodal data. In this paper, a multimodal multi-scaled graph wavelet
convolutional network (M-GWCN) is proposed as an end-to-end network. M-GWCN
simultaneously finds intra-modality representation by applying the multiscale
graph wavelet transform to provide helpful localization properties in the graph
domain of each modality, and cross-modality representation by learning
permutations that encode correlations among various modalities. M-GWCN is not
limited to either the homogeneous modalities with the same number of data, or
any prior knowledge indicating correspondences between modalities. Several
semi-supervised node classification experiments have been conducted on three
popular unimodal explicit graph-based datasets and five multimodal implicit
ones. The experimental results indicate the superiority and effectiveness of
the proposed methods compared with both spectral graph domain convolutional
neural networks and state-of-the-art multimodal methods.

    

### [[2111.13363] PicArrange -- Visually Sort, Search, and Explore Private Images on a Mac Computer](http://arxiv.org/abs/2111.13363)


  The native macOS application PicArrange integrates state-of-the-art image
sorting and similarity search to enable users to get a better overview of their
images. Many file and image management features have been added to make it a
tool that addresses a full image management workflow. A modification of the
Self Sorting Map algorithm enables a list-like image arrangement without
loosing the visual sorting. Efficient calculation and storage of visual
features as well as the use of many macOS APIs result in an application that is
fluid to use.

    

### [[2111.13394] Non-IID data and Continual Learning processes in Federated Learning: A long road ahead](http://arxiv.org/abs/2111.13394)


  Federated Learning is a novel framework that allows multiple devices or
institutions to train a machine learning model collaboratively while preserving
their data private. This decentralized approach is prone to suffer the
consequences of data statistical heterogeneity, both across the different
entities and over time, which may lead to a lack of convergence. To avoid such
issues, different methods have been proposed in the past few years. However,
data may be heterogeneous in lots of different ways, and current proposals do
not always determine the kind of heterogeneity they are considering. In this
work, we formally classify data statistical heterogeneity and review the most
remarkable learning strategies that are able to face it. At the same time, we
introduce approaches from other machine learning frameworks, such as Continual
Learning, that also deal with data heterogeneity and could be easily adapted to
the Federated Learning settings.

    

### [[2111.13404] Deep Learning for Reaction-Diffusion Glioma Growth Modelling: Towards a Fully Personalised Model?](http://arxiv.org/abs/2111.13404)


  Reaction-diffusion models have been proposed for decades to capture the
growth of gliomas, the most common primary brain tumours. However, severe
limitations regarding the estimation of the initial conditions and parameter
values of such models have restrained their clinical use as a personalised
tool. In this work, we investigate the ability of deep convolutional neural
networks (DCNNs) to address the pitfalls commonly encountered in the field.
Based on 1,200 synthetic tumours grown over real brain geometries derived from
magnetic resonance (MR) data of 6 healthy subjects, we demonstrate the ability
of DCNNs to reconstruct a whole tumour cell density distribution from only two
imaging contours at a single time point. With an additional imaging contour
extracted at a prior time point, we also demonstrate the ability of DCNNs to
accurately estimate the individual diffusivity and proliferation parameters of
the model. From this knowledge, the spatio-temporal evolution of the tumour
cell density distribution at later time points can ultimately be precisely
captured using the model. We finally show the applicability of our approach to
MR data of a real glioblastoma patient. This approach may open the perspective
of a clinical application of reaction-diffusion growth models for tumour
prognosis and treatment planning.

    

### [[2111.13411] TRIP: Refining Image-to-Image Translation via Rival Preferences](http://arxiv.org/abs/2111.13411)


  Relative attribute (RA), referring to the preference over two images on the
strength of a specific attribute, can enable fine-grained image-to-image
translation due to its rich semantic information. Existing work based on RAs
however failed to reconcile the goal for fine-grained translation and the goal
for high-quality generation. We propose a new model TRIP to coordinate these
two goals for high-quality fine-grained translation. In particular, we
simultaneously train two modules: a generator that translates an input image to
the desired image with smooth subtle changes with respect to the interested
attributes; and a ranker that ranks rival preferences consisting of the input
image and the desired image. Rival preferences refer to the adversarial ranking
process: (1) the ranker thinks no difference between the desired image and the
input image in terms of the desired attributes; (2) the generator fools the
ranker to believe that the desired image changes the attributes over the input
image as desired. RAs over pairs of real images are introduced to guide the
ranker to rank image pairs regarding the interested attributes only. With an
effective ranker, the generator would "win" the adversarial game by producing
high-quality images that present desired changes over the attributes compared
to the input image. The experiments on two face image datasets and one shoe
image dataset demonstrate that our TRIP achieves state-of-art results in
generating high-fidelity images which exhibit smooth changes over the
interested attributes.

    

### [[2111.13415] ESCADA: Efficient Safety and Context Aware Dose Allocation for Precision Medicine](http://arxiv.org/abs/2111.13415)


  Finding an optimal individualized treatment regimen is considered one of the
most challenging precision medicine problems. Various patient characteristics
influence the response to the treatment, and hence, there is no
one-size-fits-all regimen. Moreover, the administration of even a single unsafe
dose during the treatment can have catastrophic consequences on patients'
health. Therefore, an individualized treatment model must ensure patient {\em
safety} while {\em efficiently} optimizing the course of therapy. In this work,
we study a prevalent and essential medical problem setting where the treatment
aims to keep a physiological variable in a range, preferably close to a target
level. Such a task is relevant in numerous other domains as well. We propose
ESCADA, a generic algorithm for this problem structure, to make individualized
and context-aware optimal dose recommendations while assuring patient safety.
We derive high probability upper bounds on the regret of ESCADA along with
safety guarantees. Finally, we make extensive simulations on the {\em bolus
insulin dose} allocation problem in type 1 diabetes mellitus disease and
compare ESCADA's performance against Thompson sampling's, rule-based dose
allocators', and clinicians'.

    

### [[2111.13420] Confounder Identification-free Causal Visual Feature Learning](http://arxiv.org/abs/2111.13420)


  Confounders in deep learning are in general detrimental to model's
generalization where they infiltrate feature representations. Therefore,
learning causal features that are free of interference from confounders is
important. Most previous causal learning based approaches employ back-door
criterion to mitigate the adverse effect of certain specific confounder, which
require the explicit identification of confounder. However, in real scenarios,
confounders are typically diverse and difficult to be identified. In this
paper, we propose a novel Confounder Identification-free Causal Visual Feature
Learning (CICF) method, which obviates the need for identifying confounders.
CICF models the interventions among different samples based on front-door
criterion, and then approximates the global-scope intervening effect upon the
instance-level interventions from the perspective of optimization. In this way,
we aim to find a reliable optimization direction, which avoids the intervening
effects of confounders, to learn causal features. Furthermore, we uncover the
relation between CICF and the popular meta-learning strategy MAML, and provide
an interpretation of why MAML works from the theoretical perspective of causal
learning for the first time. Thanks to the effective learning of causal
features, our CICF enables models to have superior generalization capability.
Extensive experiments on domain generalization benchmark datasets demonstrate
the effectiveness of our CICF, which achieves the state-of-the-art performance.

    

### [[2111.13424] ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics](http://arxiv.org/abs/2111.13424)


  High annotation costs are a substantial bottleneck in applying modern deep
learning architectures to clinically relevant medical use cases, substantiating
the need for novel algorithms to learn from unlabeled data. In this work, we
propose ContIG, a self-supervised method that can learn from large datasets of
unlabeled medical images and genetic data. Our approach aligns images and
several genetic modalities in the feature space using a contrastive loss. We
design our method to integrate multiple modalities of each individual person in
the same model end-to-end, even when the available modalities vary across
individuals. Our procedure outperforms state-of-the-art self-supervised methods
on all evaluated downstream benchmark tasks. We also adapt gradient-based
explainability algorithms to better understand the learned cross-modal
associations between the images and genetic modalities. Finally, we perform
genome-wide association studies on the features learned by our models,
uncovering interesting relationships between images and genetic data.

    

### [[2111.13439] Towards Explainable End-to-End Prostate Cancer Relapse Prediction from H&E Images Combining Self-Attention Multiple Instance Learning with a Recurrent Neural Network](http://arxiv.org/abs/2111.13439)


  Clinical decision support for histopathology image data mainly focuses on
strongly supervised annotations, which offers intuitive interpretability, but
is bound by expert performance. Here, we propose an explainable cancer relapse
prediction network (eCaReNet) and show that end-to-end learning without strong
annotations offers state-of-the-art performance while interpretability can be
included through an attention mechanism. On the use case of prostate cancer
survival prediction, using 14,479 images and only relapse times as annotations,
we reach a cumulative dynamic AUC of 0.78 on a validation set, being on par
with an expert pathologist (and an AUC of 0.77 on a separate test set). Our
model is well-calibrated and outputs survival curves as well as a risk score
and group per patient. Making use of the attention weights of a multiple
instance learning layer, we show that malignant patches have a higher influence
on the prediction than benign patches, thus offering an intuitive
interpretation of the prediction. Our code is available at
this http URL.

    

### [[2111.13445] How Well Do Sparse Imagenet Models Transfer?](http://arxiv.org/abs/2111.13445)


  Transfer learning is a classic paradigm by which models pretrained on large
"upstream" datasets are adapted to yield good results on "downstream,"
specialized datasets. Generally, it is understood that more accurate models on
the "upstream" dataset will provide better transfer accuracy "downstream". In
this work, we perform an in-depth investigation of this phenomenon in the
context of convolutional neural networks (CNNs) trained on the ImageNet
dataset, which have been pruned - that is, compressed by sparsifiying their
connections. Specifically, we consider transfer using unstructured pruned
models obtained by applying several state-of-the-art pruning methods, including
magnitude-based, second-order, re-growth and regularization approaches, in the
context of twelve standard transfer tasks. In a nutshell, our study shows that
sparse models can match or even outperform the transfer performance of dense
models, even at high sparsities, and, while doing so, can lead to significant
inference and even training speedups. At the same time, we observe and analyze
significant differences in the behaviour of different pruning methods.

    

### [[2111.13447] A Novel Machine Learning Approach to Data Inconsistency with respect to a Fuzzy Relation](http://arxiv.org/abs/2111.13447)


  Inconsistency in prediction problems occurs when instances that relate in a
certain way on condition attributes, do not follow the same relation on the
decision attribute. For example, in ordinal classification with monotonicity
constraints, it occurs when an instance dominating another instance on
condition attributes has been assigned to a worse decision class. It typically
appears as a result of perturbation in data caused by incomplete knowledge
(missing attributes) or by random effects that occur during data generation
(instability in the assessment of decision attribute values). Inconsistencies
with respect to a crisp preorder relation (expressing either dominance or
indiscernibility between instances) can be handled using symbolic approaches
like rough set theory and by using statistical/machine learning approaches that
involve optimization methods. Fuzzy rough sets can also be seen as a symbolic
approach to inconsistency handling with respect to a fuzzy relation. In this
article, we introduce a new machine learning method for inconsistency handling
with respect to a fuzzy preorder relation. The novel approach is motivated by
the existing machine learning approach used for crisp relations. We provide
statistical foundations for it and develop optimization procedures that can be
used to eliminate inconsistencies. The article also proves important properties
and contains didactic examples of those procedures.

    

### [[2111.13460] Morphology Decoder: A Machine Learning Guided 3D Vision Quantifying Heterogenous Rock Permeability for Planetary Surveillance and Robotic Functions](http://arxiv.org/abs/2111.13460)


  Permeability has a dominant influence on the flow properties of a natural
fluid. Lattice Boltzmann simulator determines permeability from the nano and
micropore network. The simulator holds millions of flow dynamics calculations
with its accumulated errors and high consumption of computing power. To
efficiently and consistently predict permeability, we propose a morphology
decoder, a parallel and serial flow reconstruction of machine learning
segmented heterogeneous Cretaceous texture from 3D micro computerized
tomography and nuclear magnetic resonance images. For 3D vision, we introduce
controllable-measurable-volume as new supervised segmentation, in which a
unique set of voxel intensity corresponds to grain and pore throat sizes. The
morphology decoder demarks and aggregates the morphologies boundaries in a
novel way to produce permeability. Morphology decoder method consists of five
novel processes, which describes in this paper, these novel processes are: (1)
Geometrical 3D Permeability, (2) Machine Learning guided 3D Properties
Recognition of Rock Morphology, (3) 3D Image Properties Integration Model for
Permeability, (4) MRI Permeability Imager, and (5) Morphology Decoder (the
process that integrates the other four novel processes).

    

### [[2111.13461] Measuring Data Quality for Dataset Selection in Offline Reinforcement Learning](http://arxiv.org/abs/2111.13461)


  Recently developed offline reinforcement learning algorithms have made it
possible to learn policies directly from pre-collected datasets, giving rise to
a new dilemma for practitioners: Since the performance the algorithms are able
to deliver depends greatly on the dataset that is presented to them,
practitioners need to pick the right dataset among the available ones. This
problem has so far not been discussed in the corresponding literature. We
discuss ideas how to select promising datasets and propose three very simple
indicators: Estimated relative return improvement (ERI) and estimated action
stochasticity (EAS), as well as a combination of the two (COI), and empirically
show that despite their simplicity they can be very effectively used for
dataset selection.

    

### [[2111.13462] A Taxonomy of Anomalies in Log Data](http://arxiv.org/abs/2111.13462)


  Log data anomaly detection is a core component in the area of artificial
intelligence for IT operations. However, the large amount of existing methods
makes it hard to choose the right approach for a specific system. A better
understanding of different kinds of anomalies, and which algorithms are
suitable for detecting them, would support researchers and IT operators.
Although a common taxonomy for anomalies already exists, it has not yet been
applied specifically to log data, pointing out the characteristics and
peculiarities in this domain.
In this paper, we present a taxonomy for different kinds of log data
anomalies and introduce a method for analyzing such anomalies in labeled
datasets. We applied our taxonomy to the three common benchmark datasets
Thunderbird, Spirit, and BGL, and trained five state-of-the-art unsupervised
anomaly detection algorithms to evaluate their performance in detecting
different kinds of anomalies. Our results show, that the most common anomaly
type is also the easiest to predict. Moreover, deep learning-based approaches
outperform data mining-based approaches in all anomaly types, but especially
when it comes to detecting contextual anomalies.

    

### [[2111.13485] Learning Long-Term Reward Redistribution via Randomized Return Decomposition](http://arxiv.org/abs/2111.13485)


  Many practical applications of reinforcement learning require agents to learn
from sparse and delayed rewards. It challenges the ability of agents to
attribute their actions to future outcomes. In this paper, we consider the
problem formulation of episodic reinforcement learning with trajectory
feedback. It refers to an extreme delay of reward signals, in which the agent
can only obtain one reward signal at the end of each trajectory. A popular
paradigm for this problem setting is learning with a designed auxiliary dense
reward function, namely proxy reward, instead of sparse environmental signals.
Based on this framework, this paper proposes a novel reward redistribution
algorithm, randomized return decomposition (RRD), to learn a proxy reward
function for episodic reinforcement learning. We establish a surrogate problem
by Monte-Carlo sampling that scales up least-squares-based reward
redistribution to long-horizon problems. We analyze our surrogate loss function
by connection with existing methods in the literature, which illustrates the
algorithmic properties of our approach. In experiments, we extensively evaluate
our proposed method on a variety of benchmark tasks with episodic rewards and
demonstrate substantial improvement over baseline algorithms.

    

### [[2111.13486] When Creators Meet the Metaverse: A Survey on Computational Arts](http://arxiv.org/abs/2111.13486)


  The metaverse, enormous virtual-physical cyberspace, has brought
unprecedented opportunities for artists to blend every corner of our physical
surroundings with digital creativity. This article conducts a comprehensive
survey on computational arts, in which seven critical topics are relevant to
the metaverse, describing novel artworks in blended virtual-physical realities.
The topics first cover the building elements for the metaverse, e.g., virtual
scenes and characters, auditory, textual elements. Next, several remarkable
types of novel creations in the expanded horizons of metaverse cyberspace have
been reflected, such as immersive arts, robotic arts, and other user-centric
approaches fuelling contemporary creative outputs. Finally, we propose several
research agendas: democratising computational arts, digital privacy, and safety
for metaverse artists, ownership recognition for digital artworks,
technological challenges, and so on. The survey also serves as introductory
material for artists and metaverse technologists to begin creations in the
realm of surrealistic cyberspace.

    

### [[2111.13489] SurfEmb: Dense and Continuous Correspondence Distributions for Object Pose Estimation with Learnt Surface Embeddings](http://arxiv.org/abs/2111.13489)


  We present an approach to learn dense, continuous 2D-3D correspondence
distributions over the surface of objects from data with no prior knowledge of
visual ambiguities like symmetry. We also present a new method for 6D pose
estimation of rigid objects using the learnt distributions to sample, score and
refine pose hypotheses. The correspondence distributions are learnt with a
contrastive loss, represented in object-specific latent spaces by an
encoder-decoder query model and a small fully connected key model. Our method
is unsupervised with respect to visual ambiguities, yet we show that the query-
and key models learn to represent accurate multi-modal surface distributions.
Our pose estimation method improves the state-of-the-art significantly on the
comprehensive BOP Challenge, trained purely on synthetic data, even compared
with methods trained on real data. The project site is at
this https URL .

    

### [[2111.13507] Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Dependent Mixed Features](http://arxiv.org/abs/2111.13507)


  Shapley values are today extensively used as a model-agnostic explanation
framework to explain complex predictive machine learning models. Shapley values
have desirable theoretical properties and a sound mathematical foundation.
Precise Shapley value estimates for dependent data rely on accurate modeling of
the dependencies between all feature combinations. In this paper, we use a
variational autoencoder with arbitrary conditioning (VAEAC) to model all
feature dependencies simultaneously. We demonstrate through comprehensive
simulation studies that VAEAC outperforms the state-of-the-art methods for a
wide range of settings for both continuous and mixed dependent features.
Finally, we apply VAEAC to the Abalone data set from the UCI Machine Learning
Repository.

    

### [[2111.13526] An Optimization Framework for Federated Edge Learning](http://arxiv.org/abs/2111.13526)


  The optimal design of federated learning (FL) algorithms for solving general
machine learning (ML) problems in practical edge computing systems with
quantized message passing remains an open problem. This paper considers an edge
computing system where the server and workers have possibly different computing
and communication capabilities and employ quantization before transmitting
messages. To explore the full potential of FL in such an edge computing system,
we first present a general FL algorithm, namely GenQSGD, parameterized by the
numbers of global and local iterations, mini-batch size, and step size
sequence. Then, we analyze its convergence for an arbitrary step size sequence
and specify the convergence results under three commonly adopted step size
rules, namely the constant, exponential, and diminishing step size rules. Next,
we optimize the algorithm parameters to minimize the energy cost under the time
constraint and convergence error constraint, with the focus on the overall
implementing process of FL. Specifically, for any given step size sequence
under each considered step size rule, we optimize the numbers of global and
local iterations and mini-batch size to optimally implement FL for applications
with preset step size sequences. We also optimize the step size sequence along
with these algorithm parameters to explore the full potential of FL. The
resulting optimization problems are challenging non-convex problems with
non-differentiable constraint functions. We propose iterative algorithms to
obtain KKT points using general inner approximation (GIA) and tricks for
solving complementary geometric programming (CGP). Finally, we numerically
demonstrate the remarkable gains of GenQSGD with optimized algorithm parameters
over existing FL algorithms and reveal the significance of optimally designing
general FL algorithms.

    

### [[2111.13530] Uncovering the Dark Side of Telegram: Fakes, Clones, Scams, and Conspiracy Movements](http://arxiv.org/abs/2111.13530)


  Telegram is one of the most used instant messaging apps worldwide. Some of
its success lies in providing high privacy protection and social network
features like the channels -- virtual rooms in which only the admins can post
and broadcast messages to all its subscribers. However, these same features
contributed to the emergence of borderline activities and, as is common with
Online Social Networks, the heavy presence of fake accounts. Telegram started
to address these issues by introducing the verified and scam marks for the
channels. Unfortunately, the problem is far from being solved. In this work, we
perform a large-scale analysis of Telegram by collecting 35,382 different
channels and over 130,000,000 messages. We study the channels that Telegram
marks as verified or scam, highlighting analogies and differences. Then, we
move to the unmarked channels. Here, we find some of the infamous activities
also present on privacy-preserving services of the Dark Web, such as carding,
sharing of illegal adult and copyright protected content. In addition, we
identify and analyze two other types of channels: the clones and the fakes.
Clones are channels that publish the exact content of another channel to gain
subscribers and promote services. Instead, fakes are channels that attempt to
impersonate celebrities or well-known services. Fakes are hard to identify even
by the most advanced users. To detect the fake channels automatically, we
propose a machine learning model that is able to identify them with an accuracy
of 86%. Lastly, we study Sabmyk, a conspiracy theory that exploited fakes and
clones to spread quickly on the platform reaching over 1,000,000 users.

    

### [[2111.13537] A model of semantic completion in generative episodic memory](http://arxiv.org/abs/2111.13537)


  Many different studies have suggested that episodic memory is a generative
process, but most computational models adopt a storage view. In this work, we
propose a computational model for generative episodic memory. It is based on
the central hypothesis that the hippocampus stores and retrieves selected
aspects of an episode as a memory trace, which is necessarily incomplete. At
recall, the neocortex reasonably fills in the missing information based on
general semantic information in a process we call semantic completion.
As episodes we use images of digits (MNIST) augmented by different
backgrounds representing context. Our model is based on a VQ-VAE which
generates a compressed latent representation in form of an index matrix, which
still has some spatial resolution. We assume that attention selects some part
of the index matrix while others are discarded, this then represents the gist
of the episode and is stored as a memory trace. At recall the missing parts are
filled in by a PixelCNN, modeling semantic completion, and the completed index
matrix is then decoded into a full image by the VQ-VAE.
The model is able to complete missing parts of a memory trace in a
semantically plausible way up to the point where it can generate plausible
images from scratch. Due to the combinatorics in the index matrix, the model
generalizes well to images not trained on. Compression as well as semantic
completion contribute to a strong reduction in memory requirements and
robustness to noise. Finally we also model an episodic memory experiment and
can reproduce that semantically congruent contexts are always recalled better
than incongruent ones, high attention levels improve memory accuracy in both
cases, and contexts that are not remembered correctly are more often remembered
semantically congruently than completely wrong.

    

### [[2111.13545] $μ$NCA: Texture Generation with Ultra-Compact Neural Cellular Automata](http://arxiv.org/abs/2111.13545)


  We study the problem of example-based procedural texture synthesis using
highly compact models. Given a sample image, we use differentiable programming
to train a generative process, parameterised by a recurrent Neural Cellular
Automata (NCA) rule. Contrary to the common belief that neural networks should
be significantly over-parameterised, we demonstrate that our model architecture
and training procedure allows for representing complex texture patterns using
just a few hundred learned parameters, making their expressivity comparable to
hand-engineered procedural texture generating programs. The smallest models
from the proposed $\mu$NCA family scale down to 68 parameters. When using
quantisation to one byte per parameter, proposed models can be shrunk to a size
range between 588 and 68 bytes. Implementation of a texture generator that uses
these parameters to produce images is possible with just a few lines of GLSL or
C code.

    

### [[2111.13550] Using Fictitious Class Representations to Boost Discriminative Zero-Shot Learners](http://arxiv.org/abs/2111.13550)


  Focusing on discriminative zero-shot learning, in this work we introduce a
novel mechanism that dynamically augments during training the set of seen
classes to produce additional fictitious classes. These fictitious classes
diminish the model's tendency to fixate during training on attribute
correlations that appear in the training set but will not appear in newly
exposed classes. The proposed model is tested within the two formulations of
the zero-shot learning framework; namely, generalized zero-shot learning (GZSL)
and classical zero-shot learning (CZSL). Our model improves the
state-of-the-art performance on the CUB dataset and reaches comparable results
on the other common datasets, AWA2 and SUN. We investigate the strengths and
weaknesses of our method, including the effects of catastrophic forgetting when
training an end-to-end zero-shot model.

    

### [[2111.13557] On Recurrent Neural Networks for learning-based control: recent results and ideas for future developments](http://arxiv.org/abs/2111.13557)


  This paper aims to discuss and analyze the potentialities of Recurrent Neural
Networks (RNN) in control design applications. The main families of RNN are
considered, namely Neural Nonlinear AutoRegressive eXogenous, (NNARX), Echo
State Networks (ESN), Long Short Term Memory (LSTM), and Gated Recurrent Units
(GRU). The goal is twofold. Firstly, to survey recent results concerning the
training of RNN that enjoy Input-to-State Stability (ISS) and Incremental
Input-to-State Stability ({\delta}ISS) guarantees. Secondly, to discuss the
issues that still hinder the widespread use of RNN for control, namely their
robustness, verifiability, and interpretability. The former properties are
related to the so-called generalization capabilities of the networks, i.e.
their consistency with the underlying real plants, even in presence of unseen
or perturbed input trajectories. The latter is instead related to the
possibility of providing a clear formal connection between the RNN model and
the plant. In this context, we illustrate how ISS and {\delta}ISS represent a
significant step towards the robustness and verifiability of the RNN models,
while the requirement of interpretability paves the way to the use of
physics-based networks. The design of model predictive controllers with RNN as
plant's model is also briefly discussed. Lastly, some of the main topics of the
paper are illustrated on a simulated chemical system.

    

### [[2111.13580] Glass-box model representation of seismic failure mode prediction for conventional RC shear walls](http://arxiv.org/abs/2111.13580)


  The recent surge in earthquake engineering is the use of machine learning
methods to develop predictive models for structural behavior. Complex black-box
models are typically used for decision making to achieve high accuracy;
however, as important as high accuracy, it is essential for engineers to
understand how the model makes the decision and verify that the model is
physically meaningful. With this motivation, this study proposes a glass-box
(interpretable) classification model to predict the seismic failure mode of
conventional reinforced concrete shear (structural) walls. Reported
experimental damage information of 176 conventional shear walls tested under
reverse cyclic loading were designated as class-types, whereas key design
properties (e.g. compressive strength of concrete, axial load ratio, and web
reinforcement ratio) of shear walls were used as the basic classification
features. The trade-off between model complexity and model interpretability was
discussed using eight Machine Learning (ML) methods. The results showed that
the Decision Tree method was a more convenient classifier with higher
interpretability with a high classification accuracy than its counterparts.
Also, to enhance the practicality of the model, a feature reduction was
conducted to reduce the complexity of the proposed classifier with higher
classification performance, and the most relevant features were identified,
namely: compressive strength of concrete, wall aspect ratio, transverse
boundary, and web reinforcement ratio. The ability of the final DT model to
predict the failure modes was validated with a classification rate of around
90%. The proposed model aims to provide engineers interpretable, robust, and
rapid prediction in seismic performance assessment.

    

### [[2111.13587] Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers](http://arxiv.org/abs/2111.13587)


  Vision transformers have delivered tremendous success in representation
learning. This is primarily due to effective token mixing through self
attention. However, this scales quadratically with the number of pixels, which
becomes infeasible for high-resolution inputs. To cope with this challenge, we
propose Adaptive Fourier Neural Operator (AFNO) as an efficient token mixer
that learns to mix in the Fourier domain. AFNO is based on a principled
foundation of operator learning which allows us to frame token mixing as a
continuous global convolution without any dependence on the input resolution.
This principle was previously used to design FNO, which solves global
convolution efficiently in the Fourier domain and has shown promise in learning
challenging PDEs. To handle challenges in visual representation learning such
as discontinuities in images and high resolution inputs, we propose principled
architectural modifications to FNO which results in memory and computational
efficiency. This includes imposing a block-diagonal structure on the channel
mixing weights, adaptively sharing weights across tokens, and sparsifying the
frequency modes via soft-thresholding and shrinkage. The resulting model is
highly parallel with a quasi-linear complexity and has linear memory in the
sequence size. AFNO outperforms self-attention mechanisms for few-shot
segmentation in terms of both efficiency and accuracy. For Cityscapes
segmentation with the Segformer-B3 backbone, AFNO can handle a sequence size of
65k and outperforms other efficient self-attention mechanisms.

    

### [[2111.13597] Graph-based Solutions with Residuals for Intrusion Detection: the Modified E-GraphSAGE and E-ResGAT Algorithms](http://arxiv.org/abs/2111.13597)


  The high volume of increasingly sophisticated cyber threats is drawing
growing attention to cybersecurity, where many challenges remain unresolved.
Namely, for intrusion detection, new algorithms that are more robust,
effective, and able to use more information are needed. Moreover, the intrusion
detection task faces a serious challenge associated with the extreme class
imbalance between normal and malicious traffics. Recently, graph-neural network
(GNN) achieved state-of-the-art performance to model the network topology in
cybersecurity tasks. However, only a few works exist using GNNs to tackle the
intrusion detection problem. Besides, other promising avenues such as applying
the attention mechanism are still under-explored. This paper presents two novel
graph-based solutions for intrusion detection, the modified E-GraphSAGE, and
E-ResGATalgorithms, which rely on the established GraphSAGE and graph attention
network (GAT), respectively. The key idea is to integrate residual learning
into the GNN leveraging the available graph information. Residual connections
are added as a strategy to deal with the high-class imbalance, aiming at
retaining the original information and improving the minority classes'
performance. An extensive experimental evaluation of four recent intrusion
detection datasets shows the excellent performance of our approaches,
especially when predicting minority classes.

    

### [[2111.13606] Conditional Image Generation with Score-Based Diffusion Models](http://arxiv.org/abs/2111.13606)


  Score-based diffusion models have emerged as one of the most promising
frameworks for deep generative modelling. In this work we conduct a systematic
comparison and theoretical analysis of different approaches to learning
conditional probability distributions with score-based diffusion models. In
particular, we prove results which provide a theoretical justification for one
of the most successful estimators of the conditional score. Moreover, we
introduce a multi-speed diffusion framework, which leads to a new estimator for
the conditional score, performing on par with previous state-of-the-art
approaches. Our theoretical and experimental findings are accompanied by an
open source library MSDiff which allows for application and further research of
multi-speed diffusion models.

    

### [[2111.13609] A Reinforcement Learning Approach for the Continuous Electricity Market of Germany: Trading from the Perspective of a Wind Park Operator](http://arxiv.org/abs/2111.13609)


  With the rising extension of renewable energies, the intraday electricity
markets have recorded a growing popularity amongst traders as well as electric
utilities to cope with the induced volatility of the energy supply. Through
their short trading horizon and continuous nature, the intraday markets offer
the ability to adjust trading decisions from the day-ahead market or reduce
trading risk in a short-term notice. Producers of renewable energies utilize
the intraday market to lower their forecast risk, by modifying their provided
capacities based on current forecasts. However, the market dynamics are complex
due to the fact that the power grids have to remain stable and electricity is
only partly storable. Consequently, robust and intelligent trading strategies
are required that are capable to operate in the intraday market. In this work,
we propose a novel autonomous trading approach based on Deep Reinforcement
Learning (DRL) algorithms as a possible solution. For this purpose, we model
the intraday trade as a Markov Decision Problem (MDP) and employ the Proximal
Policy Optimization (PPO) algorithm as our DRL approach. A simulation framework
is introduced that enables the trading of the continuous intraday price in a
resolution of one minute steps. We test our framework in a case study from the
perspective of a wind park operator. We include next to general trade
information both price and wind forecasts. On a test scenario of German
intraday trading results from 2018, we are able to outperform multiple
baselines with at least 45.24% improvement, showing the advantage of the DRL
algorithm. However, we also discuss limitations and enhancements of the DRL
agent, in order to increase the performance in future works.

    

### [[2111.13613] The Geometry of Adversarial Training in Binary Classification](http://arxiv.org/abs/2111.13613)


  We establish an equivalence between a family of adversarial training problems
for non-parametric binary classification and a family of regularized risk
minimization problems where the regularizer is a nonlocal perimeter functional.
The resulting regularized risk minimization problems admit exact convex
relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form
frequently studied in image analysis and graph-based learning. A rich geometric
structure is revealed by this reformulation which in turn allows us to
establish a series of properties of optimal solutions of the original problem,
including the existence of minimal and maximal solutions (interpreted in a
suitable sense), and the existence of regular solutions (also interpreted in a
suitable sense). In addition, we highlight how the connection between
adversarial training and perimeter minimization problems provides a novel,
directly interpretable, statistical motivation for a family of regularized risk
minimization problems involving perimeter/total variation. The majority of our
theoretical results are independent of the distance used to define adversarial
attacks.

    

### [[2111.13617] DP-SGD vs PATE: Which Has Less Disparate Impact on GANs?](http://arxiv.org/abs/2111.13617)


  Generative Adversarial Networks (GANs) are among the most popular approaches
to generate synthetic data, especially images, for data sharing purposes. Given
the vital importance of preserving the privacy of the individual data points in
the original data, GANs are trained utilizing frameworks with robust privacy
guarantees such as Differential Privacy (DP). However, these approaches remain
widely unstudied beyond single performance metrics when presented with
imbalanced datasets. To this end, we systematically compare GANs trained with
the two best-known DP frameworks for deep learning, DP-SGD, and PATE, in
different data imbalance settings from two perspectives -- the size of the
classes in the generated synthetic data and their classification performance.
Our analyses show that applying PATE, similarly to DP-SGD, has a disparate
effect on the under/over-represented classes but in a much milder magnitude
making it more robust. Interestingly, our experiments consistently show that
for PATE, unlike DP-SGD, the privacy-utility trade-off is not monotonically
decreasing but is much smoother and inverted U-shaped, meaning that adding a
small degree of privacy actually helps generalization. However, we have also
identified some settings (e.g., large imbalance) where PATE-GAN completely
fails to learn some subparts of the training data.

    

### [[2111.13628] Nonequilibrium Monte Carlo for unfreezing variables in hard combinatorial optimization](http://arxiv.org/abs/2111.13628)


  Optimizing highly complex cost/energy functions over discrete variables is at
the heart of many open problems across different scientific disciplines and
industries. A major obstacle is the emergence of many-body effects among
certain subsets of variables in hard instances leading to critical slowing down
or collective freezing for known stochastic local search strategies. An
exponential computational effort is generally required to unfreeze such
variables and explore other unseen regions of the configuration space. Here, we
introduce a quantum-inspired family of nonlocal Nonequilibrium Monte Carlo
(NMC) algorithms by developing an adaptive gradient-free strategy that can
efficiently learn key instance-wise geometrical features of the cost function.
That information is employed on-the-fly to construct spatially inhomogeneous
thermal fluctuations for collectively unfreezing variables at various length
scales, circumventing costly exploration versus exploitation trade-offs. We
apply our algorithm to two of the most challenging combinatorial optimization
problems: random k-satisfiability (k-SAT) near the computational phase
transitions and Quadratic Assignment Problems (QAP). We observe significant
speedup and robustness over both specialized deterministic solvers and generic
stochastic solvers. In particular, for 90% of random 4-SAT instances we find
solutions that are inaccessible for the best specialized deterministic
algorithm known as Survey Propagation (SP) with an order of magnitude
improvement in the quality of solutions for the hardest 10% instances. We also
demonstrate two orders of magnitude improvement in time-to-solution over the
state-of-the-art generic stochastic solver known as Adaptive Parallel Tempering
(APT).

    

### [[2111.13630] Efficient Multi-Organ Segmentation Using SpatialConfiguration-Net with Low GPU Memory Requirements](http://arxiv.org/abs/2111.13630)


  Even though many semantic segmentation methods exist that are able to perform
well on many medical datasets, often, they are not designed for direct use in
clinical practice. The two main concerns are generalization to unseen data with
a different visual appearance, e.g., images acquired using a different scanner,
and efficiency in terms of computation time and required Graphics Processing
Unit (GPU) memory. In this work, we employ a multi-organ segmentation model
based on the SpatialConfiguration-Net (SCN), which integrates prior knowledge
of the spatial configuration among the labelled organs to resolve spurious
responses in the network outputs. Furthermore, we modified the architecture of
the segmentation model to reduce its memory footprint as much as possible
without drastically impacting the quality of the predictions. Lastly, we
implemented a minimal inference script for which we optimized both, execution
time and required GPU memory.

    

### [[2111.13646] Conditional Manifold Learning](http://arxiv.org/abs/2111.13646)


  This paper addresses a problem called "conditional manifold learning", which
aims to learn a low-dimensional manifold embedding of high-dimensional data,
conditioning on auxiliary manifold information. This auxiliary manifold
information is from controllable or measurable conditions, which are ubiquitous
in many science and engineering applications. A broad class of solutions for
this problem, conditional multidimensional scaling (including a conditional
ISOMAP variant), is proposed. A conditional version of the SMACOF algorithm is
introduced to optimize the objective function of conditional multidimensional
scaling.

    

### [[2111.13650] Latent Space Smoothing for Individually Fair Representations](http://arxiv.org/abs/2111.13650)


  Fair representation learning encodes user data to ensure fairness and
utility, regardless of the downstream application. However, learning
individually fair representations, i.e., guaranteeing that similar individuals
are treated similarly, remains challenging in high-dimensional settings such as
computer vision. In this work, we introduce LASSI, the first representation
learning method for certifying individual fairness of high-dimensional data.
Our key insight is to leverage recent advances in generative modeling to
capture the set of similar individuals in the generative latent space. This
allows learning individually fair representations where similar individuals are
mapped close together, by using adversarial training to minimize the distance
between their representations. Finally, we employ randomized smoothing to
provably map similar individuals close together, in turn ensuring that local
robustness verification of the downstream application results in end-to-end
fairness certification. Our experimental evaluation on challenging real-world
image data demonstrates that our method increases certified individual fairness
by up to 60%, without significantly affecting task utility.

    

### [[2111.13654] Do Language Models Have Beliefs? Methods for Detecting, Updating, and Visualizing Model Beliefs](http://arxiv.org/abs/2111.13654)


  Do language models have beliefs about the world? Dennett (1995) famously
argues that even thermostats have beliefs, on the view that a belief is simply
an informational state decoupled from any motivational state. In this paper, we
discuss approaches to detecting when models have beliefs about the world, and
we improve on methods for updating model beliefs to be more truthful, with a
focus on methods based on learned optimizers or hypernetworks. Our main
contributions include: (1) new metrics for evaluating belief-updating methods
that focus on the logical consistency of beliefs, (2) a training objective for
Sequential, Local, and Generalizing model updates (SLAG) that improves the
performance of learned optimizers, and (3) the introduction of the belief
graph, which is a new form of interface with language models that shows the
interdependencies between model beliefs. Our experiments suggest that models
possess belief-like qualities to only a limited extent, but update methods can
both fix incorrect model beliefs and greatly improve their consistency.
Although off-the-shelf optimizers are surprisingly strong belief-updating
baselines, our learned optimizers can outperform them in more difficult
settings than have been considered in past work. Code is available at
this https URL


### [[2111.13657] Amazon SageMaker Model Monitor: A System for Real-Time Insights into Deployed Machine Learning Models](http://arxiv.org/abs/2111.13657)


  With the increasing adoption of machine learning (ML) models and systems in
high-stakes settings across different industries, guaranteeing a model's
performance after deployment has become crucial. Monitoring models in
production is a critical aspect of ensuring their continued performance and
reliability. We present Amazon SageMaker Model Monitor, a fully managed service
that continuously monitors the quality of machine learning models hosted on
Amazon SageMaker. Our system automatically detects data, concept, bias, and
feature attribution drift in models in real-time and provides alerts so that
model owners can take corrective actions and thereby maintain high quality
models. We describe the key requirements obtained from customers, system design
and architecture, and methodology for detecting different types of drift.
Further, we provide quantitative evaluations followed by use cases, insights,
and lessons learned from more than 1.5 years of production deployment.

    

### [[2111.13663] 3D shape sensing and deep learning-based segmentation of strawberries](http://arxiv.org/abs/2111.13663)


  Automation and robotisation of the agricultural sector are seen as a viable
solution to socio-economic challenges faced by this industry. This technology
often relies on intelligent perception systems providing information about
crops, plants and the entire environment. The challenges faced by traditional
2D vision systems can be addressed by modern 3D vision systems which enable
straightforward localisation of objects, size and shape estimation, or handling
of occlusions. So far, the use of 3D sensing was mainly limited to indoor or
structured environments. In this paper, we evaluate modern sensing technologies
including stereo and time-of-flight cameras for 3D perception of shape in
agriculture and study their usability for segmenting out soft fruit from
background based on their shape. To that end, we propose a novel 3D deep neural
network which exploits the organised nature of information originating from the
camera-based 3D sensors. We demonstrate the superior performance and efficiency
of the proposed architecture compared to the state-of-the-art 3D networks.
Through a simulated study, we also show the potential of the 3D sensing
paradigm for object segmentation in agriculture and provide insights and
analysis of what shape quality is needed and expected for further analysis of
crops. The results of this work should encourage researchers and companies to
develop more accurate and robust 3D sensing technologies to assure their wider
adoption in practical agricultural applications.

    

### [[2111.13666] On the combination of graph data for assessing thin-file borrowers' creditworthiness](http://arxiv.org/abs/2111.13666)


  The thin-file borrowers are customers for whom a creditworthiness assessment
is uncertain due to their lack of credit history; many researchers have used
borrowers' relationships and interactions networks in the form of graphs as an
alternative data source to address this. Incorporating network data is
traditionally made by hand-crafted feature engineering, and lately, the graph
neural network has emerged as an alternative, but it still does not improve
over the traditional method's performance. Here we introduce a framework to
improve credit scoring models by blending several Graph Representation Learning
methods: feature engineering, graph embeddings, and graph neural networks. We
stacked their outputs to produce a single score in this approach. We validated
this framework using a unique multi-source dataset that characterizes the
relationships and credit history for the entire population of a Latin American
country, applying it to credit risk models, application, and behavior,
targeting both individuals and companies.
Our results show that the graph representation learning methods should be
used as complements, and these should not be seen as self-sufficient methods as
is currently done. In terms of AUC and KS, we enhance the statistical
performance, outperforming traditional methods.
In Corporate lending, where the gain is much higher, it confirms that
evaluating an unbanked company cannot solely consider its features. The
business ecosystem where these firms interact with their owners, suppliers,
customers, and other companies provides novel knowledge that enables financial
institutions to enhance their creditworthiness assessment.
Our results let us know when and which group to use graph data and what
effects on performance to expect. They also show the enormous value of graph
data on the unbanked credit scoring problem, principally to help companies'
banking.

    

### [[2111.13667] TMM-Fast: A Transfer Matrix Computation Package for Multilayer Thin-Film Optimization](http://arxiv.org/abs/2111.13667)


  Achieving the desired optical response from a multilayer thin-film structure
over a broad range of wavelengths and angles of incidence can be challenging.
An advanced thin-film structure can consist of multiple materials with
different thicknesses and numerous layers. Design and optimization of complex
thin-film structures with multiple variables is a computationally heavy problem
that is still under active research. To enable fast and easy experimentation
with new optimization techniques, we propose the Python package TMM-Fast which
enables parallelized computation of reflection and transmission of light at
different angles of incidence and wavelengths through the multilayer thin-film.
By decreasing computational time, generating datasets for machine learning
becomes feasible and evolutionary optimization can be used effectively.
Additionally, the sub-package TMM-Torch allows to directly compute analytical
gradients for local optimization by using PyTorch Autograd functionality.
Finally, an OpenAi Gym environment is presented which allows the user to train
reinforcement learning agents on the problem of finding multilayer thin-film
configurations.

    

### [[2111.13674] Neural Fields as Learnable Kernels for 3D Reconstruction](http://arxiv.org/abs/2111.13674)


  We present Neural Kernel Fields: a novel method for reconstructing implicit
3D shapes based on a learned kernel ridge regression. Our technique achieves
state-of-the-art results when reconstructing 3D objects and large scenes from
sparse oriented points, and can reconstruct shape categories outside the
training set with almost no drop in accuracy. The core insight of our approach
is that kernel methods are extremely effective for reconstructing shapes when
the chosen kernel has an appropriate inductive bias. We thus factor the problem
of shape reconstruction into two parts: (1) a backbone neural network which
learns kernel parameters from data, and (2) a kernel ridge regression that fits
the input points on-the-fly by solving a simple positive definite linear system
using the learned kernel. As a result of this factorization, our reconstruction
gains the benefits of data-driven methods under sparse point density while
maintaining interpolatory behavior, which converges to the ground truth shape
as input sampling density increases. Our experiments demonstrate a strong
generalization capability to objects outside the train-set category and scanned
scenes. Source code and pretrained models are available at
this https URL.

    

### [[2111.13681] ManiFest: Manifold Deformation for Few-shot Image Translation](http://arxiv.org/abs/2111.13681)


  Most image-to-image translation methods require a large number of training
images, which restricts their applicability. We instead propose ManiFest: a
framework for few-shot image translation that learns a context-aware
representation of a target domain from a few images only. To enforce feature
consistency, our framework learns a style manifold between source and proxy
anchor domains (assumed to be composed of large numbers of images). The learned
manifold is interpolated and deformed towards the few-shot target domain via
patch-based adversarial and feature statistics alignment losses. All of these
components are trained simultaneously during a single end-to-end loop. In
addition to the general few-shot translation task, our approach can
alternatively be conditioned on a single exemplar image to reproduce its
specific style. Extensive experiments demonstrate the efficacy of ManiFest on
multiple tasks, outperforming the state-of-the-art on all metrics and in both
the general- and exemplar-based scenarios. Our code will be open source.

    

### [[1602.04605] Distributed Information-Theoretic Clustering](http://arxiv.org/abs/1602.04605)


  We study a novel multi-terminal source coding setup motivated by the
biclustering problem. Two separate encoders observe two i.i.d. sequences $X^n$
and $Y^n$, respectively. The goal is to find rate-limited encodings $f(x^n)$
and $g(z^n)$ that maximize the mutual information $I(f(X^n); g(Y^n))/n$. We
discuss connections of this problem with hypothesis testing against
independence, pattern recognition, and the information bottleneck method.
Improving previous cardinality bounds for the inner and outer bounds allows us
to thoroughly study the special case of a binary symmetric source and to
quantify the gap between the inner and the outer bound in this special case.
Furthermore, we investigate a multiple description (MD) extension of the Chief
Operating Officer (CEO) problem with mutual information constraint.
Surprisingly, this MD-CEO problem permits a tight single-letter
characterization of the achievable region.

    

### [[1710.10770] Riemannian Optimization via Frank-Wolfe Methods](http://arxiv.org/abs/1710.10770)


  We study projection-free methods for constrained Riemannian optimization. In
particular, we propose the Riemannian Frank-Wolfe (RFW) method. We analyze
non-asymptotic convergence rates of RFW to an optimum for (geodesically) convex
problems, and to a critical point for nonconvex objectives. We also present a
practical setting under which RFW can attain a linear convergence rate. As a
concrete example, we specialize RFW to the manifold of positive definite
matrices and apply it to two tasks: (i) computing the matrix geometric mean
(Riemannian centroid); and (ii) computing the Bures-Wasserstein barycenter.
Both tasks involve geodesically convex interval constraints, for which we show
that the Riemannian "linear" oracle required by RFW admits a closed-form
solution; this result may be of independent interest. We further specialize RFW
to the special orthogonal group and show that here too, the Riemannian "linear"
oracle can be solved in closed form. Here, we describe an application to the
synchronization of data matrices (Procrustes problem). We complement our
theoretical results with an empirical comparison of RFW against
state-of-the-art Riemannian optimization methods and observe that RFW performs
competitively on the task of computing Riemannian centroids.

    

### [[1906.08101] Pre-Training with Whole Word Masking for Chinese BERT](http://arxiv.org/abs/1906.08101)


  Bidirectional Encoder Representations from Transformers (BERT) has shown
marvelous improvements across various NLP tasks, and its consecutive variants
have been proposed to further improve the performance of the pre-trained
language models. In this paper, we aim to first introduce the whole word
masking (wwm) strategy for Chinese BERT, along with a series of Chinese
pre-trained language models. Then we also propose a simple but effective model
called MacBERT, which improves upon RoBERTa in several ways. Especially, we
propose a new masking strategy called MLM as correction (Mac). To demonstrate
the effectiveness of these models, we create a series of Chinese pre-trained
language models as our baselines, including BERT, RoBERTa, ELECTRA, RBT, etc.
We carried out extensive experiments on ten Chinese NLP tasks to evaluate the
created Chinese pre-trained language models as well as the proposed MacBERT.
Experimental results show that MacBERT could achieve state-of-the-art
performances on many NLP tasks, and we also ablate details with several
findings that may help future research. We open-source our pre-trained language
models for further facilitating our research community. Resources are
available: this https URL


### [[1907.10709] Automatic crack classification by exploiting statistical event descriptors for Deep Learning](http://arxiv.org/abs/1907.10709)


  In modern building infrastructures, the chance to devise adaptive and
unsupervised data-driven health monitoring systems is gaining in popularity due
to the large availability of big data from low-cost sensors with communication
capabilities and advanced modeling tools such as Deep Learning. The main
purpose of this paper is to combine deep neural networks with Bidirectional
Long Short Term Memory and advanced statistical analysis involving
Instantaneous Frequency and Spectral Kurtosis to develop an accurate
classification tool for tensile, shear and mixed modes originated from acoustic
emission events (cracks). We investigated on effective event descriptors to
capture the unique characteristics from the different types of modes. Tests on
experimental results confirm that this method achieves promising classification
among different crack events and can impact on the design of future on
structural health monitoring (SHM) technologies. This approach is effective to
classify incipient damages with 92% of accuracy, which is advantageous to plan
maintenance.

    

### [[2003.08983] A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses](http://arxiv.org/abs/2003.08983)


  Recently, substantial research efforts in Deep Metric Learning (DML) focused
on designing complex pairwise-distance losses, which require convoluted schemes
to ease optimization, such as sample mining or pair weighting. The standard
cross-entropy loss for classification has been largely overlooked in DML. On
the surface, the cross-entropy may seem unrelated and irrelevant to metric
learning as it does not explicitly involve pairwise distances. However, we
provide a theoretical analysis that links the cross-entropy to several
well-known and recent pairwise losses. Our connections are drawn from two
different perspectives: one based on an explicit optimization insight; the
other on discriminative and generative views of the mutual information between
the labels and the learned features. First, we explicitly demonstrate that the
cross-entropy is an upper bound on a new pairwise loss, which has a structure
similar to various pairwise losses: it minimizes intra-class distances while
maximizing inter-class distances. As a result, minimizing the cross-entropy can
be seen as an approximate bound-optimization (or Majorize-Minimize) algorithm
for minimizing this pairwise loss. Second, we show that, more generally,
minimizing the cross-entropy is actually equivalent to maximizing the mutual
information, to which we connect several well-known pairwise losses.
Furthermore, we show that various standard pairwise losses can be explicitly
related to one another via bound relationships. Our findings indicate that the
cross-entropy represents a proxy for maximizing the mutual information -- as
pairwise losses do -- without the need for convoluted sample-mining heuristics.
Our experiments over four standard DML benchmarks strongly support our
findings. We obtain state-of-the-art results, outperforming recent and complex
DML methods.

    

### [[2004.08826] DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks](http://arxiv.org/abs/2004.08826)


  Computational Fluid Dynamics (CFD) simulation by the numerical solution of
the Navier-Stokes equations is an essential tool in a wide range of
applications from engineering design to climate modeling. However, the
computational cost and memory demand required by CFD codes may become very high
for flows of practical interest, such as in aerodynamic shape optimization.
This expense is associated with the complexity of the fluid flow governing
equations, which include non-linear partial derivative terms that are of
difficult solution, leading to long computational times and limiting the number
of hypotheses that can be tested during the process of iterative design.
Therefore, we propose DeepCFD: a convolutional neural network (CNN) based model
that efficiently approximates solutions for the problem of non-uniform steady
laminar flows. The proposed model is able to learn complete solutions of the
Navier-Stokes equations, for both velocity and pressure fields, directly from
ground-truth data generated using a state-of-the-art CFD code. Using DeepCFD,
we found a speedup of up to 3 orders of magnitude compared to the standard CFD
approach at a cost of low error rates.

    

### [[2005.04399] Estimating g-Leakage via Machine Learning](http://arxiv.org/abs/2005.04399)


  This paper considers the problem of estimating the information leakage of a
system in the black-box scenario. It is assumed that the system's internals are
unknown to the learner, or anyway too complicated to analyze, and the only
available information are pairs of input-output data samples, possibly obtained
by submitting queries to the system or provided by a third party. Previous
research has mainly focused on counting the frequencies to estimate the
input-output conditional probabilities (referred to as frequentist approach),
however this method is not accurate when the domain of possible outputs is
large. To overcome this difficulty, the estimation of the Bayes error of the
ideal classifier was recently investigated using Machine Learning (ML) models
and it has been shown to be more accurate thanks to the ability of those models
to learn the input-output correspondence. However, the Bayes vulnerability is
only suitable to describe one-try attacks. A more general and flexible measure
of leakage is the g-vulnerability, which encompasses several different types of
adversaries, with different goals and capabilities. In this paper, we propose a
novel approach to perform black-box estimation of the g-vulnerability using ML.
A feature of our approach is that it does not require to estimate the
conditional probabilities, and that it is suitable for a large class of ML
algorithms. First, we formally show the learnability for all data
distributions. Then, we evaluate the performance via various experiments using
k-Nearest Neighbors and Neural Networks. Our results outperform the frequentist
approach when the observables domain is large.

    

### [[2006.08505] Diversity Policy Gradient for Sample Efficient Quality-Diversity Optimization](http://arxiv.org/abs/2006.08505)


  A fascinating aspect of nature lies in its ability to produce a large and
diverse collection of organisms that are all high-performing in their niche. By
contrast, most AI algorithms focus on finding a single efficient solution to a
given problem. Aiming for diversity in addition to performance is a convenient
way to deal with the exploration-exploitation trade-off that plays a central
role in learning. It also allows for increased robustness when the returned
collection contains several working solutions to the considered problem, making
it well-suited for real applications such as robotics. Quality-Diversity (QD)
methods are evolutionary algorithms designed for this purpose. This paper
proposes a novel algorithm, QD - PG , which combines the strength of Policy
Gradient algorithms and Quality Diversity approaches to produce a collection of
diverse and high-performing neural policies in continuous control environments.
The main contribution of this work is the introduction of a Diversity Policy
Gradient (DPG) that exploits information at the time-step level to thrive
policies towards more diversity in a sample-efficient manner. Specifically, QD
- PG selects neural controllers from a MAP - E lites grid and uses two
gradient-based mutation operators to improve both quality and diversity,
resulting in stable population updates. Our results demonstrate that QD - PG
generates collections of diverse solutions that solve challenging exploration
and control problems while being two orders of magnitude more sample-efficient
than its evolutionary competitors.

    

### [[2008.01883] When is invariance useful in an Out-of-Distribution Generalization problem ?](http://arxiv.org/abs/2008.01883)


  The goal of Out-of-Distribution (OOD) generalization problem is to train a
predictor that generalizes on all environments. Popular approaches in this
field use the hypothesis that such a predictor shall be an \textit{invariant
predictor} that captures the mechanism that remains constant across
environments. While these approaches have been experimentally successful in
various case studies, there is still much room for the theoretical validation
of this hypothesis. This paper presents a new set of theoretical conditions
necessary for an invariant predictor to achieve the OOD optimality. Our theory
not only applies to non-linear cases, but also generalizes the necessary
condition used in \citet{rojas2018invariant}. We also derive Inter Gradient
Alignment algorithm from our theory and demonstrate its competitiveness on
MNIST-derived benchmark datasets as well as on two of the three
\textit{Invariance Unit Tests} proposed by \citet{aubinlinear}.

    

### [[2008.13210] Multiway $p$-spectral graph cuts on Grassmann manifolds](http://arxiv.org/abs/2008.13210)


  Nonlinear reformulations of the spectral clustering method have gained a lot
of recent attention due to their increased numerical benefits and their solid
mathematical background. We present a novel direct multiway spectral clustering
algorithm in the $p$-norm, for $p \in (1, 2]$. The problem of computing
multiple eigenvectors of the graph $p$-Laplacian, a nonlinear generalization of
the standard graph Laplacian, is recasted as an unconstrained minimization
problem on a Grassmann manifold. The value of $p$ is reduced in a
pseudocontinuous manner, promoting sparser solution vectors that correspond to
optimal graph cuts as $p$ approaches one. Monitoring the monotonic decrease of
the balanced graph cuts guarantees that we obtain the best available solution
from the $p$-levels considered. We demonstrate the effectiveness and accuracy
of our algorithm in various artificial test-cases. Our numerical examples and
comparative results with various state-of-the-art clustering methods indicate
that the proposed method obtains high quality clusters both in terms of
balanced graph cut metrics and in terms of the accuracy of the labelling
assignment. Furthermore, we conduct studies for the classification of facial
images and handwritten characters to demonstrate the applicability in
real-world datasets.

    

### [[2009.08093] An early prediction of covid-19 associated hospitalization surge using deep learning approach](http://arxiv.org/abs/2009.08093)


  The global pandemic caused by COVID-19 affects our lives in all aspects. As
of September 11, more than 28 million people have tested positive for COVID-19
infection, and more than 911,000 people have lost their lives in this virus
battle. Some patients can not receive appropriate medical treatment due the
limits of hospitalization volume and shortage of ICU beds. An estimated future
hospitalization is critical so that medical resources can be allocated as
needed. In this study, we propose to use 4 recurrent neural networks to infer
hospitalization change for the following week compared with the current week.
Results show that sequence to sequence model with attention achieves a high
accuracy of 0.938 and AUC of 0.850 in the hospitalization prediction. Our work
has the potential to predict the hospitalization need and send a warning to
medical providers and other stakeholders when a re-surge initializes.

    

### [[2009.09213] Dodging DeepFake Detection via Implicit Spatial-Domain Notch Filtering](http://arxiv.org/abs/2009.09213)


  The current high-fidelity generation and high-precision detection of DeepFake
images are at an arms race. We believe that producing DeepFakes that are highly
realistic and ``detection evasive'' can serve the ultimate goal of improving
future generation DeepFake detection capabilities. In this paper, we propose a
simple yet powerful pipeline to reduce the artifact patterns of fake images
without hurting image quality by performing implicit spatial-domain notch
filtering. We first demonstrate that frequency-domain notch filtering, although
famously shown to be effective in removing periodic noise in the spatial
domain, is infeasible for our task at hand due to manual designs required for
the notch filters. We, therefore, resort to a learning-based approach to
reproduce the notch filtering effects, but solely in the spatial domain. We
adopt a combination of adding overwhelming spatial noise for breaking the
periodic noise pattern and deep image filtering to reconstruct the noise-free
fake images, and we name our method DeepNotch. Deep image filtering provides a
specialized filter for each pixel in the noisy image, producing filtered images
with high fidelity compared to their DeepFake counterparts. Moreover, we also
use the semantic information of the image to generate an adversarial guidance
map to add noise intelligently. Our large-scale evaluation on 3 representative
state-of-the-art DeepFake detection methods (tested on 16 types of DeepFakes)
has demonstrated that our technique significantly reduces the accuracy of these
3 fake image detection methods, 36.79% on average and up to 97.02% in the best
case.

    

### [[2010.11642] The Role of Mutual Information in Variational Classifiers](http://arxiv.org/abs/2010.11642)


  Overfitting data is a well-known phenomenon related with the generation of a
model that mimics too closely (or exactly) a particular instance of data, and
may therefore fail to predict future observations reliably. In practice, this
behaviour is controlled by various--sometimes heuristics--regularization
techniques, which are motivated by developing upper bounds to the
generalization error. In this work, we study the generalization error of
classifiers relying on stochastic encodings trained on the cross-entropy loss,
which is often used in deep learning for classification problems. We derive
bounds to the generalization error showing that there exists a regime where the
generalization error is bounded by the mutual information between input
features and the corresponding representations in the latent space, which are
randomly generated according to the encoding distribution. Our bounds provide
an information-theoretic understanding of generalization in the so-called class
of variational classifiers, which are regularized by a Kullback-Leibler (KL)
divergence term. These results give theoretical grounds for the highly popular
KL term in variational inference methods that was already recognized to act
effectively as a regularization penalty. We further observe connections with
well studied notions such as Variational Autoencoders, Information Dropout,
Information Bottleneck and Boltzmann Machines. Finally, we perform numerical
experiments on MNIST and CIFAR datasets and show that mutual information is
indeed highly representative of the behaviour of the generalization error.

    

### [[2012.04475] Privacy-Preserving Synthetic Smart Meters Data](http://arxiv.org/abs/2012.04475)


  Power consumption data is very useful as it allows to optimize power grids,
detect anomalies and prevent failures, on top of being useful for diverse
research purposes. However, the use of power consumption data raises
significant privacy concerns, as this data usually belongs to clients of a
power company. As a solution, we propose a method to generate synthetic power
consumption samples that faithfully imitate the originals, but are detached
from the clients and their identities. Our method is based on Generative
Adversarial Networks (GANs). Our contribution is twofold. First, we focus on
the quality of the generated data, which is not a trivial task as no standard
evaluation methods are available. Then, we study the privacy guarantees
provided to members of the training set of our neural network. As a minimum
requirement for privacy, we demand our neural network to be robust to
membership inference attacks, as these provide a gateway for further attacks in
addition to presenting a privacy threat on their own. We find that there is a
compromise to be made between the privacy and the performance provided by the
algorithm.

    

### [[2012.04718] Canonical Capsules: Self-Supervised Capsules in Canonical Pose](http://arxiv.org/abs/2012.04718)


  We propose a self-supervised capsule architecture for 3D point clouds. We
compute capsule decompositions of objects through permutation-equivariant
attention, and self-supervise the process by training with pairs of randomly
rotated objects. Our key idea is to aggregate the attention masks into semantic
keypoints, and use these to supervise a decomposition that satisfies the
capsule invariance/equivariance properties. This not only enables the training
of a semantically consistent decomposition, but also allows us to learn a
canonicalization operation that enables object-centric reasoning. To train our
neural network we require neither classification labels nor manually-aligned
training datasets. Yet, by learning an object-centric representation in a
self-supervised manner, our method outperforms the state-of-the-art on 3D point
cloud reconstruction, canonicalization, and unsupervised classification.

    

### [[2101.07755] Quantum Permutation Synchronization](http://arxiv.org/abs/2101.07755)


  We present QuantumSync, the first quantum algorithm for solving a
synchronization problem in the context of computer vision. In particular, we
focus on permutation synchronization which involves solving a non-convex
optimization problem in discrete variables. We start by formulating
synchronization into a quadratic unconstrained binary optimization problem
(QUBO). While such formulation respects the binary nature of the problem,
ensuring that the result is a set of permutations requires extra care. Hence,
we: (I) show how to insert permutation constraints into a QUBO problem and (ii)
solve the constrained QUBO problem on the current generation of the adiabatic
quantum computers D-Wave. Thanks to the quantum annealing, we guarantee global
optimality with high probability while sampling the energy landscape to yield
confidence estimates. Our proof-of-concepts realization on the adiabatic D-Wave
computer demonstrates that quantum machines offer a promising way to solve the
prevalent yet difficult synchronization problems.

    

### [[2101.09571] BF++: a language for general-purpose program synthesis](http://arxiv.org/abs/2101.09571)


  Most state of the art decision systems based on Reinforcement Learning (RL)
are data-driven black-box neural models, where it is often difficult to
incorporate expert knowledge into the models or let experts review and validate
the learned decision mechanisms. Knowledge-insertion and model review are
important requirements in many applications involving human health and safety.
One way to bridge the gap between data and knowledge driven systems is program
synthesis: replacing a neural network that outputs decisions with a symbolic
program generated by a neural network or by means of genetic programming. We
propose a new programming language, BF++, designed specifically for automatic
programming of agents in a Partially Observable Markov Decision Process (POMDP)
setting and apply neural program synthesis to solve standard OpenAI Gym
benchmarks.

    

### [[2102.01852] Organization of a Latent Space structure in VAE/GAN trained by navigation data](http://arxiv.org/abs/2102.01852)


  We present a novel artificial cognitive mapping system using generative deep
neural networks, called variational autoencoder/generative adversarial network
(VAE/GAN), which can map input images to latent vectors and generate temporal
sequences internally. The results show that the distance of the predicted image
is reflected in the distance of the corresponding latent vector after training.
This indicates that the latent space is self-organized to reflect the proximity
structure of the dataset and may provide a mechanism through which many aspects
of cognition are spatially represented. The present study allows the network to
internally generate temporal sequences that are analogous to the hippocampal
replay/pre-play ability, where VAE produces only near-accurate replays of past
experiences, but by introducing GANs, the generated sequences are coupled with
instability and novelty.

    

### [[2102.09718] Permutation-Based SGD: Is Random Optimal?](http://arxiv.org/abs/2102.09718)


  A recent line of ground-breaking results for permutation-based SGD has
corroborated a widely observed phenomenon: random permutations offer faster
convergence than with-replacement sampling. However, is random optimal? We show
that this depends heavily on what functions we are optimizing, and the
convergence gap between optimal and random permutations can vary from
exponential to nonexistent. We first show that for 1-dimensional strongly
convex functions, with smooth second derivatives, there exist permutations that
offer exponentially faster convergence compared to random. However, for general
strongly convex functions, random permutations are optimal. Finally, we show
that for quadratic, strongly-convex functions, there are easy-to-construct
permutations that lead to accelerated convergence compared to random. Our
results suggest that a general convergence characterization of optimal
permutations cannot capture the nuances of individual function classes, and can
mistakenly indicate that one cannot do much better than random.

    

### [[2102.10782] NTopo: Mesh-free Topology Optimization using Implicit Neural Representations](http://arxiv.org/abs/2102.10782)


  Recent advances in implicit neural representations show great promise when it
comes to generating numerical solutions to partial differential equations.
Compared to conventional alternatives, such representations employ
parameterized neural networks to define, in a mesh-free manner, signals that
are highly-detailed, continuous, and fully differentiable. In this work, we
present a novel machine learning approach for topology optimization -- an
important class of inverse problems with high-dimensional parameter spaces and
highly nonlinear objective landscapes. To effectively leverage neural
representations in the context of mesh-free topology optimization, we use
multilayer perceptrons to parameterize both density and displacement fields.
Our experiments indicate that our method is highly competitive for minimizing
structural compliance objectives, and it enables self-supervised learning of
continuous solution spaces for topology optimization problems.

    

### [[2102.12961] On regret bounds for continual single-index learning](http://arxiv.org/abs/2102.12961)


  In this paper, we generalize the problem of single-index model to the context
of continual learning in which a learner is challenged with a sequence of tasks
one by one and the dataset of each task is revealed in an online fashion. We
propose a randomized strategy that is able to learn a common single-index
(meta-parameter) for all tasks and a specific link function for each task. The
common single-index allows to transfer the information gained from the previous
tasks to a new one. We provide a rigorous theoretical analysis of our proposed
strategy by proving some regret bounds under different assumption on the loss
function.

    

### [[2103.01484] Online Orthogonal Dictionary Learning Based on Frank-Wolfe Method](http://arxiv.org/abs/2103.01484)


  Dictionary learning is a widely used unsupervised learning method in signal
processing and machine learning. Most existing works of dictionary learning are
in an offline manner. There are mainly two offline ways for dictionary
learning. One is to do an alternative optimization of both the dictionary and
the sparse code; the other way is to optimize the dictionary by restricting it
over the orthogonal group. The latter one is called orthogonal dictionary
learning which has a lower complexity implementation, hence, it is more
favorable for lowcost devices. However, existing schemes on orthogonal
dictionary learning only work with batch data and can not be implemented
online, which is not applicable for real-time applications. This paper proposes
a novel online orthogonal dictionary scheme to dynamically learn the dictionary
from streaming data without storing the historical data. The proposed scheme
includes a novel problem formulation and an efficient online algorithm design
with convergence analysis. In the problem formulation, we relax the orthogonal
constraint to enable an efficient online algorithm. In the algorithm design, we
propose a new Frank-Wolfe-based online algorithm with a convergence rate of
O(ln t/t^(1/4)). The convergence rate in terms of key system parameters is also
derived. Experiments with synthetic data and real-world sensor readings
demonstrate the effectiveness and efficiency of the proposed online orthogonal
dictionary learning scheme.

    

### [[2103.06501] Density-aware Haze Image Synthesis by Self-Supervised Content-Style Disentanglement](http://arxiv.org/abs/2103.06501)


  The key procedure of haze image translation through adversarial training lies
in the disentanglement between the feature only involved in haze synthesis,
i.e.style feature, and the feature representing the invariant semantic content,
i.e. content feature. Previous methods separate content feature apart by
utilizing it to classify haze image during the training process. However, in
this paper we recognize the incompleteness of the content-style disentanglement
in such technical routine. The flawed style feature entangled with content
information inevitably leads the ill-rendering of the haze images. To address,
we propose a self-supervised style regression via stochastic linear
interpolation to reduce the content information in style feature. The ablative
experiments demonstrate the disentangling completeness and its superiority in
level-aware haze image synthesis. Moreover, the generated haze data are applied
in the testing generalization of vehicle detectors. Further study between
haze-level and detection performance shows that haze has obvious impact on the
generalization of the vehicle detectors and such performance degrading level is
linearly correlated to the haze-level, which, in turn, validates the
effectiveness of the proposed method.

    

### [[2103.16091] Symbolic Music Generation with Diffusion Models](http://arxiv.org/abs/2103.16091)


  Score-based generative models and diffusion probabilistic models have been
successful at generating high-quality samples in continuous domains such as
images and audio. However, due to their Langevin-inspired sampling mechanisms,
their application to discrete and sequential data has been limited. In this
work, we present a technique for training diffusion models on sequential data
by parameterizing the discrete domain in the continuous latent space of a
pre-trained variational autoencoder. Our method is non-autoregressive and
learns to generate sequences of latent embeddings through the reverse process
and offers parallel generation with a constant number of iterative refinement
steps. We apply this technique to modeling symbolic music and show strong
unconditional generation and post-hoc conditional infilling results compared to
autoregressive language models operating over the same continuous embeddings.

    

### [[2104.00530] Gaussian Process Convolutional Dictionary Learning](http://arxiv.org/abs/2104.00530)


  Convolutional dictionary learning (CDL), the problem of estimating
shift-invariant templates from data, is typically conducted in the absence of a
prior/structure on the templates. In data-scarce or low signal-to-noise ratio
(SNR) regimes, learned templates overfit the data and lack smoothness, which
can affect the predictive performance of downstream tasks. To address this
limitation, we propose GPCDL, a convolutional dictionary learning framework
that enforces priors on templates using Gaussian Processes (GPs). With the
focus on smoothness, we show theoretically that imposing a GP prior is
equivalent to Wiener filtering the learned templates, thereby suppressing
high-frequency components and promoting smoothness. We show that the algorithm
is a simple extension of the classical iteratively reweighted least squares
algorithm, independent of the choice of GP kernels. This property allows one to
experiment flexibly with different smoothness assumptions. Through simulation,
we show that GPCDL learns smooth dictionaries with better accuracy than the
unregularized alternative across a range of SNRs. Through an application to
neural spiking data, we show that GPCDL learns a more accurate and
visually-interpretable smooth dictionary, leading to superior predictive
performance compared to non-regularized CDL, as well as parametric
alternatives.

    

### [[2104.04004] ACERAC: Efficient reinforcement learning in fine time discretization](http://arxiv.org/abs/2104.04004)


  We propose a framework for reinforcement learning (RL) in fine time
discretization and a learning algorithm in this framework. One of the main
goals of RL is to provide a way for physical machines to learn optimal behavior
instead of being programmed. However, the machines are usually controlled in
fine time discretization. The most common RL methods apply independent random
elements to each action, which is not suitable in that setting. It is not
feasible because it causes the controlled system to jerk, and does not ensure
sufficient exploration since a single action is not long enough to create a
significant experience that could be translated into policy improvement. In the
RL framework introduced in this paper, policies are considered that produce
actions based on states and random elements autocorrelated in subsequent time
instants. The RL algorithm introduced here approximately optimizes such a
policy. The efficiency of this algorithm is verified against three other RL
methods (PPO, SAC, ACER) in four simulated learning control problems (Ant,
HalfCheetah, Hopper, and Walker2D) in diverse time discretization. The
algorithm introduced here outperforms the competitors in most cases considered.

    

### [[2104.07391] RIANN -- A Robust Neural Network Outperforms Attitude Estimation Filters](http://arxiv.org/abs/2104.07391)


  Inertial-sensor-based attitude estimation is a crucial technology in various
applications, from human motion tracking to autonomous aerial and ground
vehicles. Application scenarios differ in characteristics of the performed
motion, presence of disturbances, and environmental conditions. Since
state-of-the-art attitude estimators do not generalize well over these
characteristics, their parameters must be tuned for the individual motion
characteristics and circumstances. We propose RIANN, a ready-to-use, neural
network-based, parameter-free, real-time-capable inertial attitude estimator,
which generalizes well across different motion dynamics, environments, and
sampling rates, without the need for application-specific adaptations. We
gather six publicly available datasets of which we exploit two datasets for the
method development and the training, and we use four datasets for evaluation of
the trained estimator in three different test scenarios with varying practical
relevance. Results show that RIANN outperforms state-of-the-art attitude
estimation filters in the sense that it generalizes much better across a
variety of motions and conditions in different applications, with different
sensor hardware and different sampling frequencies. This is true even if the
filters are tuned on each individual test dataset, whereas RIANN was trained on
completely separate data and has never seen any of these test datasets. RIANN
can be applied directly without adaptations or training and is therefore
expected to enable plug-and-play solutions in numerous applications, especially
when accuracy is crucial but no ground-truth data is available for tuning or
when motion and disturbance characteristics are uncertain. We made RIANN
publicly available.

    

### [[2104.07620] Collective Iterative Learning Control: Exploiting Diversity in Multi-Agent Systems for Reference Tracking Tasks](http://arxiv.org/abs/2104.07620)


  Multi-agent systems (MASs) can autonomously learn to solve previously unknown
tasks by means of each agent's individual intelligence as well as by
collaborating and exploiting collective intelligence. This article considers a
group of autonomous agents learning to track the same given reference
trajectory in a possibly small number of trials. We propose a novel collective
learning control method that combines iterative learning control (ILC) with a
collective update strategy. We derive conditions for desirable convergence
properties of such systems. We show that the proposed method allows the
collective to combine the advantages of the agents' individual learning
strategies and thereby overcomes trade-offs and limitations of single-agent
ILC. This benefit is achieved by designing a heterogeneous collective, i.e., a
different learning law is assigned to each agent. All theoretical results are
confirmed in simulations and experiments with two-wheeled-inverted-pendulum
robots (TWIPRs) that jointly learn to perform the desired maneuver.

    

### [[2105.04104] AppealNet: An Efficient and Highly-Accurate Edge/Cloud Collaborative Architecture for DNN Inference](http://arxiv.org/abs/2105.04104)


  This paper presents AppealNet, a novel edge/cloud collaborative architecture
that runs deep learning (DL) tasks more efficiently than state-of-the-art
solutions. For a given input, AppealNet accurately predicts on-the-fly whether
it can be successfully processed by the DL model deployed on the
resource-constrained edge device, and if not, appeals to the more powerful DL
model deployed at the cloud. This is achieved by employing a two-head neural
network architecture that explicitly takes inference difficulty into
consideration and optimizes the tradeoff between accuracy and
computation/communication cost of the edge/cloud collaborative architecture.
Experimental results on several image classification datasets show up to more
than 40% energy savings compared to existing techniques without sacrificing
accuracy.

    

### [[2105.10497] Intriguing Properties of Vision Transformers](http://arxiv.org/abs/2105.10497)


  Vision transformers (ViT) have demonstrated impressive performance across
various machine vision problems. These models are based on multi-head
self-attention mechanisms that can flexibly attend to a sequence of image
patches to encode contextual cues. An important question is how such
flexibility in attending image-wide context conditioned on a given patch can
facilitate handling nuisances in natural images e.g., severe occlusions, domain
shifts, spatial permutations, adversarial and natural perturbations. We
systematically study this question via an extensive set of experiments
encompassing three ViT families and comparisons with a high-performing
convolutional neural network (CNN). We show and analyze the following
intriguing properties of ViT: (a) Transformers are highly robust to severe
occlusions, perturbations and domain shifts, e.g., retain as high as 60% top-1
accuracy on ImageNet even after randomly occluding 80% of the image content.
(b) The robust performance to occlusions is not due to a bias towards local
textures, and ViTs are significantly less biased towards textures compared to
CNNs. When properly trained to encode shape-based features, ViTs demonstrate
shape recognition capability comparable to that of human visual system,
previously unmatched in the literature. (c) Using ViTs to encode shape
representation leads to an interesting consequence of accurate semantic
segmentation without pixel-level supervision. (d) Off-the-shelf features from a
single ViT model can be combined to create a feature ensemble, leading to high
accuracy rates across a range of classification datasets in both traditional
and few-shot learning paradigms. We show effective features of ViTs are due to
flexible and dynamic receptive fields possible via the self-attention
mechanism.

    

### [[2105.13623] Enhanced Doubly Robust Learning for Debiasing Post-click Conversion Rate Estimation](http://arxiv.org/abs/2105.13623)


  Post-click conversion, as a strong signal indicating the user preference, is
salutary for building recommender systems. However, accurately estimating the
post-click conversion rate (CVR) is challenging due to the selection bias,
i.e., the observed clicked events usually happen on users' preferred items.
Currently, most existing methods utilize counterfactual learning to debias
recommender systems. Among them, the doubly robust (DR) estimator has achieved
competitive performance by combining the error imputation based (EIB) estimator
and the inverse propensity score (IPS) estimator in a doubly robust way.
However, inaccurate error imputation may result in its higher variance than the
IPS estimator. Worse still, existing methods typically use simple
model-agnostic methods to estimate the imputation error, which are not
sufficient to approximate the dynamically changing model-correlated target
(i.e., the gradient direction of the prediction model). To solve these
problems, we first derive the bias and variance of the DR estimator. Based on
it, a more robust doubly robust (MRDR) estimator has been proposed to further
reduce its variance while retaining its double robustness. Moreover, we propose
a novel double learning approach for the MRDR estimator, which can convert the
error imputation into the general CVR estimation. Besides, we empirically
verify that the proposed learning scheme can further eliminate the high
variance problem of the imputation learning. To evaluate its effectiveness,
extensive experiments are conducted on a semi-synthetic dataset and two
real-world datasets. The results demonstrate the superiority of the proposed
approach over the state-of-the-art methods. The code is available at
this https URL.

    

### [[2106.01138] Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting](http://arxiv.org/abs/2106.01138)


  In molecular dynamics (MD), neural network (NN) potentials trained bottom-up
on quantum mechanical data have seen tremendous success recently. Top-down
approaches that learn NN potentials directly from experimental data have
received less attention, typically facing numerical and computational
challenges when backpropagating through MD simulations. We present the
Differentiable Trajectory Reweighting (DiffTRe) method, which bypasses
differentiation through the MD simulation for time-independent observables.
Leveraging thermodynamic perturbation theory, we avoid exploding gradients and
achieve around 2 orders of magnitude speed-up in gradient computation for
top-down learning. We show effectiveness of DiffTRe in learning NN potentials
for an atomistic model of diamond and a coarse-grained model of water based on
diverse experimental observables including thermodynamic, structural and
mechanical properties. Importantly, DiffTRe also generalizes bottom-up
structural coarse-graining methods such as iterative Boltzmann inversion to
arbitrary potentials. The presented method constitutes an important milestone
towards enriching NN potentials with experimental data, particularly when
accurate bottom-up data is unavailable.

    

### [[2106.05969] Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation](http://arxiv.org/abs/2106.05969)


  We propose a method for object-aware 3D egocentric pose estimation that
tightly integrates kinematics modeling, dynamics modeling, and scene object
information. Unlike prior kinematics or dynamics-based approaches where the two
components are used disjointly, we synergize the two approaches via
dynamics-regulated training. At each timestep, a kinematic model is used to
provide a target pose using video evidence and simulation state. Then, a
prelearned dynamics model attempts to mimic the kinematic pose in a physics
simulator. By comparing the pose instructed by the kinematic model against the
pose generated by the dynamics model, we can use their misalignment to further
improve the kinematic model. By factoring in the 6DoF pose of objects (e.g.,
chairs, boxes) in the scene, we demonstrate for the first time, the ability to
estimate physically-plausible 3D human-object interactions using a single
wearable camera. We evaluate our egocentric pose estimation method in both
controlled laboratory settings and real-world scenarios.

    

### [[2106.06097] Neural Optimization Kernel: Towards Robust Deep Learning](http://arxiv.org/abs/2106.06097)


  Deep neural networks (NN) have achieved great success in many applications.
However, why do deep neural networks obtain good generalization at an
over-parameterization regime is still unclear. To better understand deep NN, we
establish the connection between deep NN and a novel kernel family, i.e.,
Neural Optimization Kernel (NOK). The architecture of structured approximation
of NOK performs monotonic descent updates of implicit regularization problems.
We can implicitly choose the regularization problems by employing different
activation functions, e.g., ReLU, max pooling, and soft-thresholding. We
further establish a new generalization bound of our deep structured
approximated NOK architecture. Our unsupervised structured approximated NOK
block can serve as a simple plug-in of popular backbones for a good
generalization against input noise.

    

### [[2106.08746] Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses](http://arxiv.org/abs/2106.08746)


  Recent work has shown that deep reinforcement learning (DRL) policies are
vulnerable to adversarial perturbations. Adversaries can mislead policies of
DRL agents by perturbing the state of the environment observed by the agents.
Existing attacks are feasible in principle but face challenges in practice, for
example by being too slow to fool DRL policies in real time. We show that using
the Universal Adversarial Perturbation (UAP) method to compute perturbations,
independent of the individual inputs to which they are applied to, can fool DRL
policies effectively and in real time. We describe three such attack variants.
Via an extensive evaluation using three Atari 2600 games, we show that our
attacks are effective, as they fully degrade the performance of three different
DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is
as small as 0.01). It is faster compared to the response time (0.6ms on
average) of different DRL policies, and considerably faster than prior attacks
using adversarial perturbations (1.8ms on average). We also show that our
attack technique is efficient, incurring an online computational cost of
0.027ms on average. Using two further tasks involving robotic movement, we
confirm that our results generalize to more complex DRL tasks. Furthermore, we
demonstrate that the effectiveness of known defenses diminishes against
universal perturbations. We propose an effective technique that detects all
known adversarial perturbations against DRL policies, including all the
universal perturbations presented in this paper.

    

### [[2106.11112] Multivariate Data Explanation by Jumping Emerging Patterns Visualization](http://arxiv.org/abs/2106.11112)


  Visual Analytics (VA) tools and techniques have been instrumental in
supporting users to build better classification models, interpret models'
overall logic, and audit results. In a different direction, VA has recently
been applied to transform classification models into descriptive mechanisms
instead of predictive. The idea is to use such models as surrogates for data
patterns, visualizing the model to understand the phenomenon represented by the
data. Although very useful and inspiring, the few proposed approaches have
opted to use low complex classification models to promote straightforward
interpretation, presenting limitations to capture intricate data patterns. In
this paper, we present VAX (multiVariate dAta eXplanation), a new VA method to
support the identification and visual interpretation of patterns in
multivariate datasets. Unlike the existing similar approaches, VAX uses the
concept of Jumping Emerging Patterns to identify and aggregate several
diversified patterns, producing explanations through logic combinations of data
variables. The potential of VAX to interpret complex multivariate datasets is
demonstrated through use-cases employing two real-world datasets covering
different scenarios.

    

### [[2106.11542] Differentiable Architecture Search Meets Network Pruning at Initialization: A More Reliable, Efficient, and Flexible Framework](http://arxiv.org/abs/2106.11542)


  Although Differentiable ARchiTecture Search (DARTS) has become the mainstream
paradigm in Neural Architecture Search (NAS) due to its simplicity and
efficiency, more recent works found that the performance of the searched
architecture barely increases with the optimization proceeding in DARTS, and
the final magnitudes obtained by DARTS could hardly indicate the importance of
operations. The above observation reveal that the supervision signal in DARTS
may be a poor or unreliable indicator for the architecture search, inspiring an
interesting and promising direction: can we measure the operation importance
without any training under the differentiable paradigm? We provide an
affirmative answer by customizing the NAS as a network pruning at
initialization problem. With leveraging recently-proposed synaptic saliency
criteria in the network pruning at initialization, we seek to score the
importance of candidate operations in differentiable NAS without any training,
and proposed a novel framework called \textit{training free differentiable
architecture search} (FreeDARTS) accordingly. We show that, without any
training, FreeDARTS with different proxy metrics can outperform most NAS
baselines in different search spaces. More importantly, FreeDARTS is extremely
memory-efficient and computational-efficient as it abandons the training in the
architecture search phase, enabling FreeDARTS to perform architecture search on
a more flexible space and eliminate the depth gap between architecture search
and evaluation. We hope our work inspires more attempts in solving NAS from the
perspective of pruning at initialization.

    

### [[2106.14623] Polyconvex anisotropic hyperelasticity with neural networks](http://arxiv.org/abs/2106.14623)


  In the present work, two machine learning based constitutive models for
finite deformations are proposed. Using input convex neural networks, the
models are hyperelastic, anisotropic and fulfill the polyconvexity condition,
which implies ellipticity and thus ensures material stability. The first
constitutive model is based on a set of polyconvex, anisotropic and objective
invariants. The second approach is formulated in terms of the deformation
gradient, its cofactor and determinant, uses group symmetrization to fulfill
the material symmetry condition, and data augmentation to fulfill objectivity
approximately. The extension of the dataset for the data augmentation approach
is based on mechanical considerations and does not require additional
experimental or simulation data. The models are calibrated with highly
challenging simulation data of cubic lattice metamaterials, including finite
deformations and lattice instabilities. A moderate amount of calibration data
is used, based on deformations which are commonly applied in experimental
investigations. While the invariant-based model shows drawbacks for several
deformation modes, the model based on the deformation gradient alone is able to
reproduce and predict the effective material behavior very well and exhibits
excellent generalization capabilities. In addition, the models are calibrated
with transversely isotropic data, generated with an analytical polyconvex
potential. For this case, both models show excellent results, demonstrating the
straightforward applicability of the polyconvex neural network constitutive
models to other symmetry groups.

    

### [[2106.16116] PSD Representations for Effective Probability Models](http://arxiv.org/abs/2106.16116)


  Finding a good way to model probability densities is key to probabilistic
inference. An ideal model should be able to concisely approximate any
probability while being also compatible with two main operations:
multiplications of two models (product rule) and marginalization with respect
to a subset of the random variables (sum rule). In this work, we show that a
recently proposed class of positive semi-definite (PSD) models for non-negative
functions is particularly suited to this end. In particular, we characterize
both approximation and generalization capabilities of PSD models, showing that
they enjoy strong theoretical guarantees. Moreover, we show that we can perform
efficiently both sum and product rule in closed form via matrix operations,
enjoying the same versatility of mixture models. Our results open the way to
applications of PSD models to density estimation, decision theory and
inference.

    

### [[2109.05751] Adversarially Trained Object Detector for Unsupervised Domain Adaptation](http://arxiv.org/abs/2109.05751)


  Unsupervised domain adaptation, which involves transferring knowledge from a
label-rich source domain to an unlabeled target domain, can be used to
substantially reduce annotation costs in the field of object detection. In this
study, we demonstrate that adversarial training in the source domain can be
employed as a new approach for unsupervised domain adaptation. Specifically, we
establish that adversarially trained detectors achieve improved detection
performance in target domains that are significantly shifted from source
domains. This phenomenon is attributed to the fact that adversarially trained
detectors can be used to extract robust features that are in alignment with
human perception and worth transferring across domains while discarding
domain-specific non-robust features. In addition, we propose a method that
combines adversarial training and feature alignment to ensure the improved
alignment of robust features with the target domain. We conduct experiments on
four benchmark datasets and confirm the effectiveness of our proposed approach
on large domain shifts from real to artistic images. Compared to the baseline
models, the adversarially trained detectors improve the mean average precision
by up to 7.7%, and further by up to 11.8% when feature alignments are
incorporated. Although our method degrades performance for small domain shifts,
quantification of the domain shift based on the Frechet distance allows us to
determine whether adversarial training should be conducted.

    

### [[2110.05064] Ab-Initio Potential Energy Surfaces by Pairing GNNs with Neural Wave Functions](http://arxiv.org/abs/2110.05064)


  Solving the Schrödinger equation is key to many quantum mechanical
properties. However, an analytical solution is only tractable for
single-electron systems. Recently, neural networks succeeded at modeling wave
functions of many-electron systems. Together with the variational Monte-Carlo
(VMC) framework, this led to solutions on par with the best known classical
methods. Still, these neural methods require tremendous amounts of
computational resources as one has to train a separate model for each molecular
geometry. In this work, we combine a Graph Neural Network (GNN) with a neural
wave function to simultaneously solve the Schrödinger equation for multiple
geometries via VMC. This enables us to model continuous subsets of the
potential energy surface with a single training pass. Compared to existing
state-of-the-art networks, our Potential Energy Surface Network PESNet speeds
up training for multiple geometries by up to 40 times while matching or
surpassing their accuracy. This may open the path to accurate and orders of
magnitude cheaper quantum mechanical calculations.

    

### [[2110.11328] A Fine-Grained Analysis on Distribution Shift](http://arxiv.org/abs/2110.11328)


  Robustness to distribution shifts is critical for deploying machine learning
models in the real world. Despite this necessity, there has been little work in
defining the underlying mechanisms that cause these shifts and evaluating the
robustness of algorithms across multiple, different distribution shifts. To
this end, we introduce a framework that enables fine-grained analysis of
various distribution shifts. We provide a holistic analysis of current
state-of-the-art methods by evaluating 19 distinct methods grouped into five
categories across both synthetic and real-world datasets. Overall, we train
more than 85K models. Our experimental framework can be easily extended to
include new methods, shifts, and datasets. We find, unlike previous
work~\citep{Gulrajani20}, that progress has been made over a standard ERM
baseline; in particular, pretraining and augmentations (learned or heuristic)
offer large gains in many cases. However, the best methods are not consistent
over different datasets and shifts.

    

### [[2110.12987] Optimization-Based GenQSGD for Federated Edge Learning](http://arxiv.org/abs/2110.12987)


  Optimal algorithm design for federated learning (FL) remains an open problem.
This paper explores the full potential of FL in practical edge computing
systems where workers may have different computation and communication
capabilities, and quantized intermediate model updates are sent between the
server and workers. First, we present a general quantized parallel mini-batch
stochastic gradient descent (SGD) algorithm for FL, namely GenQSGD, which is
parameterized by the number of global iterations, the numbers of local
iterations at all workers, and the mini-batch size. We also analyze its
convergence error for any choice of the algorithm parameters. Then, we optimize
the algorithm parameters to minimize the energy cost under the time constraint
and convergence error constraint. The optimization problem is a challenging
non-convex problem with non-differentiable constraint functions. We propose an
iterative algorithm to obtain a KKT point using advanced optimization
techniques. Numerical results demonstrate the significant gains of GenQSGD over
existing FL algorithms and reveal the importance of optimally designing FL
algorithms.

    

### [[2111.11303] Machine Learning of Thermodynamic Observables in the Presence of Mode Collapse](http://arxiv.org/abs/2111.11303)


  Estimating the free energy, as well as other thermodynamic observables, is a
key task in lattice field theories. Recently, it has been pointed out that deep
generative models can be used in this context [1]. Crucially, these models
allow for the direct estimation of the free energy at a given point in
parameter space. This is in contrast to existing methods based on Markov chains
which generically require integration through parameter space. In this
contribution, we will review this novel machine-learning-based estimation
method. We will in detail discuss the issue of mode collapse and outline
mitigation techniques which are particularly suited for applications at finite
temperature.

    

### [[2104.07699] pLUTo: Enabling Massively Parallel Computation In DRAM via Lookup Tables](http://arxiv.org/abs/2104.07699)


  Data movement between main memory and the processor is a key contributor to
the execution time and energy consumption of memory-intensive applications.
This data movement bottleneck can be alleviated using Processing-in-Memory
(PiM). One category of PiM is Processing-using-Memory (PuM), in which
computation takes place inside the memory array by exploiting intrinsic analog
properties of the memory device. PuM yields high throughput and efficiency, but
supports a limited range of operations. As a result, PuM architectures cannot
efficiently perform some complex operations (e.g., multiplication, division,
exponentiation) without sizeable increases in chip area and design complexity.
To overcome this limitation in DRAM-based PuM architectures, we introduce
pLUTo (processing-using-memory with lookup table [LUT] operations), a
DRAM-based PuM architecture that leverages the high area density of DRAM to
enable the massively parallel storing and querying of lookup tables (LUTs). The
use of LUTs enables pLUTo to efficiently execute complex operations in-memory
via memory reads (i.e., LUT queries) instead of relying on complex extra logic
or performing long sequences of DRAM commands. pLUTo outperforms the optimized
CPU and GPU baselines in performance/energy efficiency by an average of
1960$\times$/307$\times$ and 4.2$\times$/4$\times$ across the evaluated
workloads, and by 33$\times$/8$\times$ and 110$\times$/80$\times$ for the
LeNet-5 quantized neural network. pLUTo outperforms a state-of-the-art PiM
baseline by 50$\times$/342$\times$ in performance/energy efficiency.

    

### [[2111.12785] Notebook-as-a-VRE (NaaVRE): from private notebooks to a collaborative cloud virtual research environment](http://arxiv.org/abs/2111.12785)


  Virtual Research Environments (VREs) provide user-centric support in the
lifecycle of research activities, e.g., discovering and accessing research
assets, or composing and executing application workflows. A typical VRE is
often implemented as an integrated environment, which includes a catalog of
research assets, a workflow management system, a data management framework, and
tools for enabling collaboration among users. Notebook environments, such as
Jupyter, allow researchers to rapidly prototype scientific code and share their
experiments as online accessible notebooks. Jupyter can support several popular
languages that are used by data scientists, such as Python, R, and Julia.
However, such notebook environments do not have seamless support for running
heavy computations on remote infrastructure or finding and accessing software
code inside notebooks. This paper investigates the gap between a notebook
environment and a VRE and proposes an embedded VRE solution for the Jupyter
environment called Notebook-as-a-VRE (NaaVRE). The NaaVRE solution provides
functional components via a component marketplace and allows users to create a
customized VRE on top of the Jupyter environment. From the VRE, a user can
search research assets (data, software, and algorithms), compose workflows,
manage the lifecycle of an experiment, and share the results among users in the
community. We demonstrate how such a solution can enhance a legacy workflow
that uses Light Detection and Ranging (LiDAR) data from country-wide airborne
laser scanning surveys for deriving geospatial data products of ecosystem
structure at high resolution over broad spatial extents. This enables users to
scale out the processing of multi-terabyte LiDAR point clouds for ecological
applications to more data sources in a distributed cloud environment.

    

### [[2111.12885] A Dense Tensor Accelerator with Data Exchange Mesh for DNN and Vision Workloads](http://arxiv.org/abs/2111.12885)


  We propose a dense tensor accelerator called VectorMesh, a scalable,
memory-efficient architecture that can support a wide variety of DNN and
computer vision workloads. Its building block is a tile execution unit~(TEU),
which includes dozens of processing elements~(PEs) and SRAM buffers connected
through a butterfly network. A mesh of FIFOs between the TEUs facilitates data
exchange between tiles and promote local data to global visibility. Our design
performs better according to the roofline model for CNN, GEMM, and spatial
matching algorithms compared to state-of-the-art architectures. It can reduce
global buffer and DRAM fetches by 2-22 times and up to 5 times, respectively.

    

### [[2111.13058] STRETCH: Virtual Shared-Nothing Parallelism for Scalable and Elastic Stream Processing](http://arxiv.org/abs/2111.13058)


  Stream processing applications extract value from raw data through Directed
Acyclic Graphs of data analysis tasks. Shared-nothing (SN) parallelism is the
de-facto standard to scale stream processing applications. Given an
application, SN parallelism instantiates several copies of each analysis task,
making each instance responsible for a dedicated portion of the overall
analysis, and relies on dedicated queues to exchange data among connected
instances. On the one hand, SN parallelism can scale the execution of
applications both up and out since threads can run task instances within and
across processes/nodes. On the other hand, its lack of sharing can cause
unnecessary overheads and hinder the scaling up when threads operate on data
that could be jointly accessed in shared memory. This trade-off motivated us in
studying a way for stream processing applications to leverage shared memory and
boost the scale up (before the scale out) while adhering to the widely-adopted
and SN-based APIs for stream processing applications.
We introduce STRETCH, a framework that maximizes the scale up and offers
instantaneous elastic reconfigurations (without state transfer) for stream
processing applications. We propose the concept of Virtual Shared-Nothing (VSN)
parallelism and elasticity and provide formal definitions and correctness
proofs for the semantics of the analysis tasks supported by STRETCH, showing
they extend the ones found in common Stream Processing Engines. We also provide
a fully implemented prototype and show that STRETCH's performance exceeds that
of state-of-the-art baselines (Apache Flink and ScaleJoin) and offers, to the
best of our knowledge, unprecedented ultra-fast reconfigurations, taking less
than 40 ms even when provisioning tens of new task instances.

    

### [[2111.13291] Influence of atomic FAA on ParallelFor and a cost model for improvements](http://arxiv.org/abs/2111.13291)


  This paper focuses on one of the most frequently visited multithreading
library interfaces - ParallelFor. In this study, it is inferred that
ParallelFor's end-to-end latency performance is noticeably affected by the
frequency with which fetch-add-add (FAA) is called during program execution.
This can be explained by ParallelFor's uniform semantics and the utilization of
atomic FAA. To prove this assumption, a battery of tests was designed and
conducted on diverse platforms. From the collected performance statistics and
overall trends, several conclusions were drawn and a cost model is proposed to
enhance performance by mitigating the influence of FAA.

    

### [[2111.13500] A Trust and Reputation System for IoT Exploiting Distributed Ledger Technology](http://arxiv.org/abs/2111.13500)


  The advent of Bitcoin, and consequently Blockchain, has ushered in a new era
of decentralization. Blockchain enables mutually distrusting entities to work
collaboratively to attain a common objective. However, current Blockchain
technologies lack scalability, which limits their use in Internet of Things
(IoT) applications. Many devices on the Internet have the computational and
communication capabilities to facilitate decision-making. These devices will
soon be a 50 billion node network. Furthermore, new IoT business models such as
Sensor-as-a-Service (SaaS) require a robust Trust and Reputation System (TRS).
In this paper, we introduce an innovative distributed ledger combining Tangle
and Blockchain as a TRS framework for IoT. The combination of Tangle and
Blockchain provides maintainability of the former and scalability of the
latter. The proposed ledger can handle large numbers of IoT device transactions
and facilitates low power nodes joining and contributing. Employing a
distributed ledger mitigates many threats, such as whitewashing attacks. Along
with combining payments and rating protocols, the proposed approach provides
cleaner data to the upper layer reputation algorithm.

    

### [[2004.02841] NVTraverse: In NVRAM Data Structures, the Destination is More Important than the Journey](http://arxiv.org/abs/2004.02841)


  The recent availability of fast, dense, byte-addressable non-volatile memory
has led to increasing interest in the problem of designing and specifying
durable data structures that can recover from system crashes. However,
designing durable concurrent data structures that are efficient and also
satisfy a correctness criterion has proven to be very difficult, leading many
algorithms to be inefficient or incorrect in a concurrent setting. In this
paper, we present a general transformation that takes a lock-free data
structure from a general class called traversal data structure (that we
formally define) and automatically transforms it into an implementation of the
data structure for the NVRAM setting that is provably durably linearizable and
highly efficient. The transformation hinges on the observation that many data
structure operations begin with a traversal phase that does not need to be
persisted, and thus we only begin persisting when the traversal reaches its
destination. We demonstrate the transformation's efficiency through extensive
measurements on a system with Intel's recently released Optane DC persistent
memory, showing that it can outperform competitors on many workloads.

    

### [[2111.12727] Universal Captioner: Long-Tail Vision-and-Language Model Training through Content-Style Separation](http://arxiv.org/abs/2111.12727)


  While captioning models have obtained compelling results in describing
natural images, they still do not cover the entire long-tail distribution of
real-world concepts. In this paper, we address the task of generating
human-like descriptions with in-the-wild concepts by training on web-scale
automatically collected datasets. To this end, we propose a model which can
exploit noisy image-caption pairs while maintaining the descriptive style of
traditional human-annotated datasets like COCO. Our model separates content
from style through the usage of keywords and stylistic tokens, employing a
single objective of prompt language modeling and being simpler than other
recent proposals. Experimentally, our model consistently outperforms existing
methods in terms of caption quality and capability of describing long-tail
concepts, also in zero-shot settings. According to the CIDEr metric, we obtain
a new state of the art on both COCO and nocaps when using external data.

    

### [[2111.12758] Lensless multicore-fiber microendoscope for real-time tailored light field generation with phase encoder neural network (CoreNet)](http://arxiv.org/abs/2111.12758)


  The generation of tailored light with multi-core fiber (MCF) lensless
microendoscopes is widely used in biomedicine. However, the computer-generated
holograms (CGHs) used for such applications are typically generated by
iterative algorithms, which demand high computation effort, limiting advanced
applications like in vivo optogenetic stimulation and fiber-optic cell
manipulation. The random and discrete distribution of the fiber cores induces
strong spatial aliasing to the CGHs, hence, an approach that can rapidly
generate tailored CGHs for MCFs is highly demanded. We demonstrate a novel
phase encoder deep neural network (CoreNet), which can generate accurate
tailored CGHs for MCFs at a near video-rate. Simulations show that CoreNet can
speed up the computation time by two magnitudes and increase the fidelity of
the generated light field compared to the conventional CGH techniques. For the
first time, real-time generated tailored CGHs are on-the-fly loaded to the
phase-only SLM for dynamic light fields generation through the MCF
microendoscope in experiments. This paves the avenue for real-time cell
rotation and several further applications that require real-time high-fidelity
light delivery in biomedicine.

    

### [[2111.12782] Fast mesh denoising with data driven normal filtering using deep variational autoencoders](http://arxiv.org/abs/2111.12782)


  Recent advances in 3D scanning technology have enabled the deployment of 3D
models in various industrial applications like digital twins, remote inspection
and reverse engineering. Despite their evolving performance, 3D scanners, still
introduce noise and artifacts in the acquired dense models. In this work, we
propose a fast and robust denoising method for dense 3D scanned industrial
models. The proposed approach employs conditional variational autoencoders to
effectively filter face normals. Training and inference are performed in a
sliding patch setup reducing the size of the required training data and
execution times. We conducted extensive evaluation studies using 3D scanned and
CAD models. The results verify plausible denoising outcomes, demonstrating
similar or higher reconstruction accuracy, compared to other state-of-the-art
approaches. Specifically, for 3D models with more than 1e4 faces, the presented
pipeline is twice as fast as methods with equivalent reconstruction error.

    

### [[2111.12830] TSO-DSOs Stable Cost Allocation for the Joint Procurement of Flexibility: A Cooperative Game Approach](http://arxiv.org/abs/2111.12830)


  In this paper, a transmission-distribution systems flexibility market is
introduced, in which system operators (SOs) jointly procure flexibility from
different systems to meet their needs (balancing and congestion management)
using a common market. This common market is, then, formulated as a cooperative
game aiming at identifying a stable and efficient split of costs of the jointly
procured flexibility among the participating SOs to incentivize their
cooperation. The non-emptiness of the core of this game is then mathematically
proven, implying the stability of the game and the naturally-arising incentive
for cooperation among the SOs. Several cost allocation mechanisms are then
introduced, while characterizing their mathematical properties. Numerical
results focusing on an interconnected system (composed of the IEEE 14-bus
transmission system and the Matpower 18-bus, 69-bus, and 141-bus distributions
systems) showcase the cooperation-induced reduction in system-wide flexibility
procurement costs, and identifies the varying costs borne by different SOs
under various cost allocations methods.

    

### [[2111.12878] Multiway Non-rigid Point Cloud Registration via Learned Functional Map Synchronization](http://arxiv.org/abs/2111.12878)


  We present SyNoRiM, a novel way to jointly register multiple non-rigid shapes
by synchronizing the maps relating learned functions defined on the point
clouds. Even though the ability to process non-rigid shapes is critical in
various applications ranging from computer animation to 3D digitization, the
literature still lacks a robust and flexible framework to match and align a
collection of real, noisy scans observed under occlusions. Given a set of such
point clouds, our method first computes the pairwise correspondences
parameterized via functional maps. We simultaneously learn potentially
non-orthogonal basis functions to effectively regularize the deformations,
while handling the occlusions in an elegant way. To maximally benefit from the
multi-way information provided by the inferred pairwise deformation fields, we
synchronize the pairwise functional maps into a cycle-consistent whole thanks
to our novel and principled optimization formulation. We demonstrate via
extensive experiments that our method achieves a state-of-the-art performance
in registration accuracy, while being flexible and efficient as we handle both
non-rigid and multi-body cases in a unified framework and avoid the costly
optimization over point-wise permutations by the use of basis function maps.

    

### [[2111.12880] Active Learning at the ImageNet Scale](http://arxiv.org/abs/2111.12880)


  Active learning (AL) algorithms aim to identify an optimal subset of data for
annotation, such that deep neural networks (DNN) can achieve better performance
when trained on this labeled subset. AL is especially impactful in industrial
scale settings where data labeling costs are high and practitioners use every
tool at their disposal to improve model performance. The recent success of
self-supervised pretraining (SSP) highlights the importance of harnessing
abundant unlabeled data to boost model performance. By combining AL with SSP,
we can make use of unlabeled data while simultaneously labeling and training on
particularly informative samples.
In this work, we study a combination of AL and SSP on ImageNet. We find that
performance on small toy datasets -- the typical benchmark setting in the
literature -- is not representative of performance on ImageNet due to the class
imbalanced samples selected by an active learner. Among the existing baselines
we test, popular AL algorithms across a variety of small and large scale
settings fail to outperform random sampling. To remedy the class-imbalance
problem, we propose Balanced Selection (BASE), a simple, scalable AL algorithm
that outperforms random sampling consistently by selecting more balanced
samples for annotation than existing methods. Our code is available at:
this https URL .

    

### [[2111.12888] Effectiveness of Detection-based and Regression-based Approaches for Estimating Mask-Wearing Ratio](http://arxiv.org/abs/2111.12888)


  Estimating the mask-wearing ratio in public places is important as it enables
health authorities to promptly analyze and implement policies. Methods for
estimating the mask-wearing ratio on the basis of image analysis have been
reported. However, there is still a lack of comprehensive research on both
methodologies and datasets. Most recent reports straightforwardly propose
estimating the ratio by applying conventional object detection and
classification methods. It is feasible to use regression-based approaches to
estimate the number of people wearing masks, especially for congested scenes
with tiny and occluded faces, but this has not been well studied. A large-scale
and well-annotated dataset is still in demand. In this paper, we present two
methods for ratio estimation that leverage either a detection-based or
regression-based approach. For the detection-based approach, we improved the
state-of-the-art face detector, RetinaFace, used to estimate the ratio. For the
regression-based approach, we fine-tuned the baseline network, CSRNet, used to
estimate the density maps for masked and unmasked faces. We also present the
first large-scale dataset, the ``NFM dataset,'' which contains 581,108 face
annotations extracted from 18,088 video frames in 17 street-view videos.
Experiments demonstrated that the RetinaFace-based method has higher accuracy
under various situations and that the CSRNet-based method has a shorter
operation time thanks to its compactness.

    

### [[2111.12899] Recommending Multiple Positive Citations for Manuscript via Content-Dependent Modeling and Multi-Positive Triplet](http://arxiv.org/abs/2111.12899)


  Considering the rapidly increasing number of academic papers, searching for
and citing appropriate references has become a non-trial task during the wiring
of papers. Recommending a handful of candidate papers to a manuscript before
publication could ease the burden of the authors, and help the reviewers to
check the completeness of the cited resources. Conventional approaches on
citation recommendation generally consider recommending one ground-truth
citation for a query context from an input manuscript, but lack of
consideration on co-citation recommendations. However, a piece of context often
needs to be supported by two or more co-citation pairs. Here, we propose a
novel scientific paper modeling for citation recommendations, namely
Multi-Positive BERT Model for Citation Recommendation (MP-BERT4CR), complied
with a series of Multi-Positive Triplet objectives to recommend multiple
positive citations for a query context. The proposed approach has the following
advantages: First, the proposed multi-positive objectives are effective to
recommend multiple positive candidates. Second, we adopt noise distributions
which are built based on the historical co-citation frequencies, so that
MP-BERT4CR is not only effective on recommending high-frequent co-citation
pairs; but also the performances on retrieving the low-frequent ones are
significantly improved. Third, we propose a dynamic context sampling strategy
which captures the ``macro-scoped'' citing intents from a manuscript and
empowers the citation embeddings to be content-dependent, which allow the
algorithm to further improve the performances. Single and multiple positive
recommendation experiments testified that MP-BERT4CR delivered significant
improvements. In addition, MP-BERT4CR are also effective in retrieving the full
list of co-citations, and historically low-frequent co-citation pairs compared
with the prior works.

    

### [[2111.12905] CIRCLE: Convolutional Implicit Reconstruction and Completion for Large-scale Indoor Scene](http://arxiv.org/abs/2111.12905)


  We present CIRCLE, a framework for large-scale scene completion and geometric
refinement based on local implicit signed distance functions. It is based on an
end-to-end sparse convolutional network, CircNet, that jointly models local
geometric details and global scene structural contexts, allowing it to preserve
fine-grained object detail while recovering missing regions commonly arising in
traditional 3D scene data. A novel differentiable rendering module enables
test-time refinement for better reconstruction quality. Extensive experiments
on both real-world and synthetic datasets show that our concise framework is
efficient and effective, achieving better reconstruction quality than the
closest competitor while being 10-50x faster.

    

### [[2111.12929] Unbiased Pairwise Learning to Rank in Recommender Systems](http://arxiv.org/abs/2111.12929)


  Nowadays, recommender systems already impact almost every facet of peoples
lives. To provide personalized high quality recommendation results,
conventional systems usually train pointwise rankers to predict the absolute
value of objectives and leverage a distinct shallow tower to estimate and
alleviate the impact of position bias. However, with such a training paradigm,
the optimization target differs a lot from the ranking metrics valuing the
relative order of top ranked items rather than the prediction precision of each
item. Moreover, as the existing system tends to recommend more relevant items
at higher positions, it is difficult for the shallow tower based methods to
precisely attribute the user feedback to the impact of position or relevance.
Therefore, there exists an exciting opportunity for us to get enhanced
performance if we manage to solve the aforementioned issues. Unbiased learning
to rank algorithms, which are verified to model the relative relevance
accurately based on noisy feedback, are appealing candidates and have already
been applied in many applications with single categorical labels, such as user
click signals. Nevertheless, the existing unbiased LTR methods cannot properly
handle multiple feedback incorporating both categorical and continuous labels.
Accordingly, we design a novel unbiased LTR algorithm to tackle the challenges,
which innovatively models position bias in the pairwise fashion and introduces
the pairwise trust bias to separate the position bias, trust bias, and user
relevance explicitly. Experiment results on public benchmark datasets and
internal live traffic show the superior results of the proposed method for both
categorical and continuous labels.

    

### [[2111.12978] Observing Interventions: A logic for thinking about experiments](http://arxiv.org/abs/2111.12978)


  This paper makes a first step towards a logic of learning from experiments.
For this, we investigate formal frameworks for modeling the interaction of
causal and (qualitative) epistemic reasoning. Crucial for our approach is the
idea that the notion of an intervention can be used as a formal expression of a
(real or hypothetical) experiment. In a first step we extend the well-known
causal models with a simple Hintikka-style representation of the epistemic
state of an agent. In the resulting setting, one can talk not only about the
knowledge of an agent about the values of variables and how interventions
affect them, but also about knowledge update. The resulting logic can model
reasoning about thought experiments. However, it is unable to account for
learning from experiments, which is clearly brought out by the fact that it
validates the no learning principle for interventions. Therefore, in a second
step, we implement a more complex notion of knowledge that allows an agent to
observe (measure) certain variables when an experiment is carried out. This
extended system does allow for learning from experiments. For all the proposed
logical systems, we provide a sound and complete axiomatization.

    

### [[2111.13027] Toward an Idiomatic Framework for Cognitive Robotics](http://arxiv.org/abs/2111.13027)


  Inspired by the "Cognitive Hour-glass" model presented in
this https URL, we propose a new framework for
developing cognitive architectures aimed at cognitive robotics. The purpose of
the proposed framework is foremost to ease the development of cognitive
architectures by encouraging and mitigating cooperation and re-use of existing
results. This is done by proposing a framework dividing the development of
cognitive architectures into a series of layers that can be considered partly
in isolation, and some of which directly relate to other research fields.
Finally, we give introductions to and review some topics essential to the
proposed framework.

    

### [[2111.13122] GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval](http://arxiv.org/abs/2111.13122)


  Even though it has extensively been shown that retrieval specific training of
deep neural networks is beneficial for nearest neighbor image search quality,
most of these models are trained and tested in the domain of landmarks images.
However, some applications use images from various other domains and therefore
need a network with good generalization properties - a general-purpose CBIR
model. To the best of our knowledge, no testing protocol has so far been
introduced to benchmark models with respect to general image retrieval quality.
After analyzing popular image retrieval test sets we decided to manually curate
GPR1200, an easy to use and accessible but challenging benchmark dataset with a
broad range of image categories. This benchmark is subsequently used to
evaluate various pretrained models of different architectures on their
generalization qualities. We show that large-scale pretraining significantly
improves retrieval performance and present experiments on how to further
increase these properties by appropriate fine-tuning. With these promising
results, we hope to increase interest in the research topic of general-purpose
CBIR.

    

### [[2111.13136] Monitoring Hybrid Process Specifications with Conflict Management: The Automata-theoretic Approach](http://arxiv.org/abs/2111.13136)


  Business process monitoring approaches have thus far mainly focused on
monitoring the execution of a process with respect to a single process model.
However, in some cases it is necessary to consider multiple process
specifications simultaneously. In addition, these specifications can be
procedural, declarative, or a combination of both. For example, in the medical
domain, a clinical guideline describing the treatment of a specific disease
cannot account for all possible co-factors that can coexist for a specific
patient and therefore additional constraints may need to be considered. In some
cases, these constraints may be incompatible with clinical guidelines,
therefore requiring the violation of either the guidelines or the constraints.
In this paper, we propose a solution for monitoring the interplay of hybrid
process specifications expressed as a combination of (data-aware) Petri nets
and temporal logic rules. During the process execution, if these specifications
are in conflict with each other, it is possible to violate some of them. The
monitoring system is equipped with a violation cost model according to which
the system can recommend the next course of actions in a way that would either
avoid possible violations or minimize the total cost of violations.

    

### [[2111.13144] Learning to Search in Task and Motion Planning with Streams](http://arxiv.org/abs/2111.13144)


  Task and motion planning problems in robotics typically combine symbolic
planning over discrete task variables with motion optimization over continuous
state and action variables, resulting in trajectories that satisfy the logical
constraints imposed on the task variables. Symbolic planning can scale
exponentially with the number of task variables, so recent works such as
PDDLStream have focused on optimistic planning with an incrementally growing
set of objects and facts until a feasible trajectory is found. However, this
set is exhaustively and uniformly expanded in a breadth-first manner,
regardless of the geometric structure of the problem at hand, which makes
long-horizon reasoning with large numbers of objects prohibitively
time-consuming. To address this issue, we propose a geometrically informed
symbolic planner that expands the set of objects and facts in a best-first
manner, prioritized by a Graph Neural Network based score that is learned from
prior search computations. We evaluate our approach on a diverse set of
problems and demonstrate an improved ability to plan in large or difficult
scenarios. We also apply our algorithm on a 7DOF robotic arm in several
block-stacking manipulation tasks.

    

### [[2111.13145] Unravelling multi-agent ranked delegations](http://arxiv.org/abs/2111.13145)


  We introduce a voting model with multi-agent ranked delegations. This model
generalises liquid democracy in two aspects: first, an agent's delegation can
use the votes of multiple other agents to determine their own -- for instance,
an agent's vote may correspond to the majority outcome of the votes of a
trusted group of agents; second, agents can submit a ranking over multiple
delegations, so that a backup delegation can be used when their preferred
delegations are involved in cycles. The main focus of this paper is the study
of unravelling procedures that transform the delegation ballots received from
the agents into a profile of direct votes, from which a winning alternative can
then be determined by using a standard voting rule. We propose and study six
such unravelling procedures, two based on optimisation and four using a greedy
approach. We study both algorithmic and axiomatic properties, as well as
related computational complexity problems of our unravelling procedures for
different restrictions on the types of ballots that the agents can submit.

    

### [[2111.13149] A Comparative Analysis of Machine Learning Techniques for IoT Intrusion Detection](http://arxiv.org/abs/2111.13149)


  The digital transformation faces tremendous security challenges. In
particular, the growing number of cyber-attacks targeting Internet of Things
(IoT) systems restates the need for a reliable detection of malicious network
activity. This paper presents a comparative analysis of supervised,
unsupervised and reinforcement learning techniques on nine malware captures of
the IoT-23 dataset, considering both binary and multi-class classification
scenarios. The developed models consisted of Support Vector Machine (SVM),
Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine
(LightGBM), Isolation Forest (iForest), Local Outlier Factor (LOF) and a Deep
Reinforcement Learning (DRL) model based on a Double Deep Q-Network (DDQN),
adapted to the intrusion detection context. The best performance was achieved
by LightGBM, closely followed by SVM. Nonetheless, iForest displayed good
results against unknown attacks and the DRL model demonstrated the possible
benefits of employing this methodology to continuously improve the detection.
Overall, the obtained results indicate that the analyzed techniques are well
suited for IoT intrusion detection.

    

### [[2111.13157] DA$^{\textbf{2}}$-Net : Diverse & Adaptive Attention Convolutional Neural Network](http://arxiv.org/abs/2111.13157)


  Standard Convolutional Neural Network (CNN) designs rarely focus on the
importance of explicitly capturing diverse features to enhance the network's
performance. Instead, most existing methods follow an indirect approach of
increasing or tuning the networks' depth and width, which in many cases
significantly increases the computational cost. Inspired by a biological visual
system, we propose a Diverse and Adaptive Attention Convolutional Network
(DA$^{2}$-Net), which enables any feed-forward CNNs to explicitly capture
diverse features and adaptively select and emphasize the most informative
features to efficiently boost the network's performance. DA$^{2}$-Net incurs
negligible computational overhead and it is designed to be easily integrated
with any CNN architecture. We extensively evaluated DA$^{2}$-Net on benchmark
datasets, including CIFAR100, SVHN, and ImageNet, with various CNN
architectures. The experimental results show DA$^{2}$-Net provides a
significant performance improvement with very minimal computational overhead.

    

### [[2111.13188] BioLeaF: A Bio-plausible Learning Framework for Training of Spiking Neural Networks](http://arxiv.org/abs/2111.13188)


  Our brain consists of biological neurons encoding information through
accurate spike timing, yet both the architecture and learning rules of our
brain remain largely unknown. Comparing to the recent development of
backpropagation-based (BP-based) methods that are able to train spiking neural
networks (SNNs) with high accuracy, biologically plausible methods are still in
their infancy. In this work, we wish to answer the question of whether it is
possible to attain comparable accuracy of SNNs trained by BP-based rules with
bio-plausible mechanisms. We propose a new bio-plausible learning framework,
consisting of two components: a new architecture, and its supporting learning
rules. With two types of cells and four types of synaptic connections, the
proposed local microcircuit architecture can compute and propagate error
signals through local feedback connections and support training of multi-layers
SNNs with a globally defined spiking error function. Under our microcircuit
architecture, we employ the Spike-Timing-Dependent-Plasticity (STDP) rule
operating in local compartments to update synaptic weights and achieve
supervised learning in a biologically plausible manner. Finally, We interpret
the proposed framework from an optimization point of view and show the
equivalence between it and the BP-based rules under a special circumstance. Our
experiments show that the proposed framework demonstrates learning accuracy
comparable to BP-based rules and may provide new insights on how learning is
orchestrated in biological systems.

    

### [[2111.13230] FedDropoutAvg: Generalizable federated learning for histopathology image classification](http://arxiv.org/abs/2111.13230)


  Federated learning (FL) enables collaborative learning of a deep learning
model without sharing the data of participating sites. FL in medical image
analysis tasks is relatively new and open for enhancements. In this study, we
propose FedDropoutAvg, a new federated learning approach for training a
generalizable model. The proposed method takes advantage of randomness, both in
client selection and also in federated averaging process. We compare
FedDropoutAvg to several algorithms in an FL scenario for real-world multi-site
histopathology image classification task. We show that with FedDropoutAvg, the
final model can achieve performance better than other FL approaches and closer
to a classical deep learning model that requires all data to be shared for
centralized training. We test the trained models on a large dataset consisting
of 1.2 million image tiles from 21 different centers. To evaluate the
generalization ability of the proposed approach, we use held-out test sets from
centers whose data was used in the FL and for unseen data from other
independent centers whose data was not used in the federated training. We show
that the proposed approach is more generalizable than other state-of-the-art
federated training approaches. To the best of our knowledge, ours is the first
study to use a randomized client and local model parameter selection procedure
in a federated setting for a medical image analysis task.

    

### [[2111.13254] Unscented Kalman Filter for Long-Distance Vessel Tracking in Geodetic Coordinates](http://arxiv.org/abs/2111.13254)


  This paper describes a novel tracking filter, designed primarily for use in
collision avoidance systems on autonomous surface vehicles (ASVs). The proposed
methodology leverages real-time kinematic information broadcast via the
Automatic Information System (AIS) messaging protocol, in order to estimate the
position, speed, and heading of nearby cooperative targets. The state of each
target is recursively estimated in geodetic coordinates using an unscented
Kalman filter (UKF) with kinematic equations derived from the spherical law of
cosines. This improves upon previous approaches, many of which employ the
extended Kalman filter (EKF), and thus require the specification of a local
planar coordinate frame, in order to describe the state kinematics in an easily
differentiable form. The proposed geodetic UKF obviates the need for this local
plane. This feature is particularly advantageous for long-range ASVs, which
must otherwise periodically redefine a new local plane to curtail linearization
error. In real-world operations, this recurring redefinition can introduce
error and complicate mission planning. It is shown through both simulation and
field testing that the proposed geodetic UKF performs as well as, or better
than, the traditional plane-Cartesian EKF, both in terms of estimation error
and stability.

    

### [[2111.13259] Identification of Bias Against People with Disabilities in Sentiment Analysis and Toxicity Detection Models](http://arxiv.org/abs/2111.13259)


  Sociodemographic biases are a common problem for natural language processing,
affecting the fairness and integrity of its applications. Within sentiment
analysis, these biases may undermine sentiment predictions for texts that
mention personal attributes that unbiased human readers would consider neutral.
Such discrimination can have great consequences in the applications of
sentiment analysis both in the public and private sectors. For example,
incorrect inferences in applications like online abuse and opinion analysis in
social media platforms can lead to unwanted ramifications, such as wrongful
censoring, towards certain populations. In this paper, we address the
discrimination against people with disabilities, PWD, done by sentiment
analysis and toxicity classification models. We provide an examination of
sentiment and toxicity analysis models to understand in detail how they
discriminate PWD. We present the Bias Identification Test in Sentiments (BITS),
a corpus of 1,126 sentences designed to probe sentiment analysis models for
biases in disability. We use this corpus to demonstrate statistically
significant biases in four widely used sentiment analysis tools (TextBlob,
VADER, Google Cloud Natural Language API and DistilBERT) and two toxicity
analysis models trained to predict toxic comments on Jigsaw challenges (Toxic
comment classification and Unintended Bias in Toxic comments). The results show
that all exhibit strong negative biases on sentences that mention disability.
We publicly release BITS Corpus for others to identify potential biases against
disability in any sentiment analysis tools and also to update the corpus to be
used as a test for other sociodemographic variables as well.

    

### [[2111.13271] Designing a Trusted Data Brokerage Framework in the Aviation Domain](http://arxiv.org/abs/2111.13271)


  In recent years, there is growing interest in the ways the European aviation
industry can leverage the multi-source data fusion towards augmented domain
intelligence. However, privacy, legal and organisational policies together with
technical limitations, hinder data sharing and, thus, its benefits. The current
paper presents the ICARUS data policy and assets brokerage framework, which
aims to (a) formalise the data attributes and qualities that affect how
aviation data assets can be shared and handled subsequently to their
acquisition, including licenses, IPR, characterisation of sensitivity and
privacy risks, and (b) enable the creation of machine-processable data
contracts for the aviation industry. This involves expressing contractual terms
pertaining to data trading agreements into a machine-processable language and
supporting the diverse interactions among stakeholders in aviation data sharing
scenarios through a trusted and robust system based on the Ethereum platform.

    

### [[2111.13285] 3D Pose Estimation and Future Motion Prediction from 2D Images](http://arxiv.org/abs/2111.13285)


  This paper considers to jointly tackle the highly correlated tasks of
estimating 3D human body poses and predicting future 3D motions from RGB image
sequences. Based on Lie algebra pose representation, a novel self-projection
mechanism is proposed that naturally preserves human motion kinematics. This is
further facilitated by a sequence-to-sequence multi-task architecture based on
an encoder-decoder topology, which enables us to tap into the common ground
shared by both tasks. Finally, a global refinement module is proposed to boost
the performance of our framework. The effectiveness of our approach, called
PoseMoNet, is demonstrated by ablation tests and empirical evaluations on
Human3.6M and HumanEva-I benchmark, where competitive performance is obtained
comparing to the state-of-the-arts.

    

### [[2111.13295] Medial Spectral Coordinates for 3D Shape Analysis](http://arxiv.org/abs/2111.13295)


  In recent years there has been a resurgence of interest in our community in
the shape analysis of 3D objects represented by surface meshes, their voxelized
interiors, or surface point clouds. In part, this interest has been stimulated
by the increased availability of RGBD cameras, and by applications of computer
vision to autonomous driving, medical imaging, and robotics. In these settings,
spectral coordinates have shown promise for shape representation due to their
ability to incorporate both local and global shape properties in a manner that
is qualitatively invariant to isometric transformations. Yet, surprisingly,
such coordinates have thus far typically considered only local surface
positional or derivative information. In the present article, we propose to
equip spectral coordinates with medial (object width) information, so as to
enrich them. The key idea is to couple surface points that share a medial ball,
via the weights of the adjacency matrix. We develop a spectral feature using
this idea, and the algorithms to compute it. The incorporation of object width
and medial coupling has direct benefits, as illustrated by our experiments on
object classification, object part segmentation, and surface point
correspondence.

    

### [[2111.13304] A Proposal for Amending Privacy Regulations to Tackle the Challenges Stemming from Combining Data Sets](http://arxiv.org/abs/2111.13304)


  Modern information and communication technology practices present novel
threats to privacy. We focus on some shortcomings in current data protection
regulation's ability to adequately address the ramifications of AI-driven data
processing practices, in particular those of combining data sets. We propose
that privacy regulation relies less on individuals' privacy expectations and
recommend regulatory reform in two directions: (1) abolishing the distinction
between personal and anonymized data for the purposes of triggering the
application of data protection laws and (2) developing methods to prioritize
regulatory intervention based on the level of privacy risk posed by individual
data processing actions. This is an interdisciplinary paper that intends to
build a bridge between the various communities involved in privacy research. We
put special emphasis on linking technical notions with their regulatory
implications and introducing the relevant technical and legal terminology in
use to foster more efficient coordination between the policymaking and
technical communities and enable a timely solution of the problems raised.

    

### [[2111.13309] Data Augmented 3D Semantic Scene Completion with 2D Segmentation Priors](http://arxiv.org/abs/2111.13309)


  Semantic scene completion (SSC) is a challenging Computer Vision task with
many practical applications, from robotics to assistive computing. Its goal is
to infer the 3D geometry in a field of view of a scene and the semantic labels
of voxels, including occluded regions. In this work, we present SPAwN, a novel
lightweight multimodal 3D deep CNN that seamlessly fuses structural data from
the depth component of RGB-D images with semantic priors from a bimodal 2D
segmentation network. A crucial difficulty in this field is the lack of fully
labeled real-world 3D datasets which are large enough to train the current
data-hungry deep 3D CNNs. In 2D computer vision tasks, many data augmentation
strategies have been proposed to improve the generalization ability of CNNs.
However those approaches cannot be directly applied to the RGB-D input and
output volume of SSC solutions. In this paper, we introduce the use of a 3D
data augmentation strategy that can be applied to multimodal SSC networks. We
validate our contributions with a comprehensive and reproducible ablation
study. Our solution consistently surpasses previous works with a similar level
of complexity.

    

### [[2111.13326] Evacuation Shelter Scheduling Problem](http://arxiv.org/abs/2111.13326)


  Evacuation shelters, which are urgently required during natural disasters,
are designed to minimize the burden of evacuation on human survivors. However,
the larger the scale of the disaster, the more costly it becomes to operate
shelters. When the number of evacuees decreases, the operation costs can be
reduced by moving the remaining evacuees to other shelters and closing shelters
as quickly as possible. On the other hand, relocation between shelters imposes
a huge emotional burden on evacuees. In this study, we formulate the
"Evacuation Shelter Scheduling Problem," which allocates evacuees to shelters
in such a way to minimize the movement costs of the evacuees and the operation
costs of the shelters. Since it is difficult to solve this quadratic
programming problem directly, we show its transformation into a 0-1 integer
programming problem. In addition, such a formulation struggles to calculate the
burden of relocating them from historical data because no payments are actually
made. To solve this issue, we propose a method that estimates movement costs
based on the numbers of evacuees and shelters during an actual disaster.
Simulation experiments with records from the Kobe earthquake (Great
Hanshin-Awaji Earthquake) showed that our proposed method reduced operation
costs by 33.7 million dollars: 32%.

    

### [[2111.13365] Machines and Influence](http://arxiv.org/abs/2111.13365)


  Policymakers face a broader challenge of how to view AI capabilities today
and where does society stand in terms of those capabilities. This paper surveys
AI capabilities and tackles this very issue, exploring it in context of
political security in digital societies. We introduce a Matrix of Machine
Influence to frame and navigate the adversarial applications of AI, and further
extend the ideas of Information Management to better understand contemporary AI
systems deployment as part of a complex information system. Providing a
comprehensive review of man-machine interactions in our networked society and
political systems, we suggest that better regulation and management of
information systems can more optimally offset the risks of AI and utilise the
emerging capabilities which these systems have to offer to policymakers and
political institutions across the world. Hopefully this long essay will actuate
further debates and discussions over these ideas, and prove to be a useful
contribution towards governing the future of AI.

    

### [[2111.13463] Soliciting User Preferences in Conversational Recommender Systems via Usage-related Questions](http://arxiv.org/abs/2111.13463)


  A key distinguishing feature of conversational recommender systems over
traditional recommender systems is their ability to elicit user preferences
using natural language. Currently, the predominant approach to preference
elicitation is to ask questions directly about items or item attributes. These
strategies do not perform well in cases where the user does not have sufficient
knowledge of the target domain to answer such questions. Conversely, in a
shopping setting, talking about the planned use of items does not present any
difficulties, even for those that are new to a domain. In this paper, we
propose a novel approach to preference elicitation by asking implicit questions
based on item usage. Our approach consists of two main steps. First, we
identify the sentences from a large review corpus that contain information
about item usage. Then, we generate implicit preference elicitation questions
from those sentences using a neural text-to-text model. The main contributions
of this work also include a multi-stage data annotation protocol using
crowdsourcing for collecting high-quality labeled training data for the neural
model. We show that our approach is effective in selecting review sentences and
transforming them to elicitation questions, even with limited training data.
Additionally, we provide an analysis of patterns where the model does not
perform optimally.

    

### [[2111.13475] QMagFace: Simple and Accurate Quality-Aware Face Recognition](http://arxiv.org/abs/2111.13475)


  Face recognition systems have to deal with large variabilities (such as
different poses, illuminations, and expressions) that might lead to incorrect
matching decisions. These variabilities can be measured in terms of face image
quality which is defined over the utility of a sample for recognition. Previous
works on face recognition either do not employ this valuable information or
make use of non-inherently fit quality estimates. In this work, we propose a
simple and effective face recognition solution (QMagFace) that combines a
quality-aware comparison score with a recognition model based on a
magnitude-aware angular margin loss. The proposed approach includes
model-specific face image qualities in the comparison process to enhance the
recognition performance under unconstrained circumstances. Exploiting the
linearity between the qualities and their comparison scores induced by the
utilized loss, our quality-aware comparison function is simple and highly
generalizable. The experiments conducted on several face recognition databases
and benchmarks demonstrate that the introduced quality-awareness leads to
consistent improvements in the recognition performance. Moreover, the proposed
QMagFace approach performs especially well under challenging circumstances,
such as cross-pose, cross-age, or cross-quality. Consequently, it leads to
state-of-the-art performances on several face recognition benchmarks, such as
98.50% on AgeDB, 83.97% on XQLFQ, and 98.74% on CFP-FP. The code for QMagFace
is publicly available.

    

### [[2111.13562] Asian Giant Hornet Control based on Image Processing and Biological Dispersal](http://arxiv.org/abs/2111.13562)


  The Asian giant hornet (AGH) appeared in Washington State appears to have a
potential danger of bioinvasion. Washington State has collected public photos
and videos of detected insects for verification and further investigation. In
this paper, we analyze AGH using data analysis,statistics, discrete
mathematics, and deep learning techniques to process the data to controlAGH
spreading.First, we visualize the geographical distribution of insects in
Washington State. Then we investigate insect populations to varying months of
the year and different days of a month.Third, we employ wavelet analysis to
examine the periodic spread of AGH. Fourth, we apply ordinary differential
equations to examine AGH numbers at the different natural growthrate and
reaction speed and output the potential propagation coefficient. Next, we
leverage cellular automaton combined with the potential propagation coefficient
to simulate the geographical spread under changing potential propagation. To
update the model, we use delayed differential equations to simulate human
intervention. We use the time difference between detection time and submission
time to determine the unit of time to delay time. After that, we construct a
lightweight CNN called SqueezeNet and assess its classification performance. We
then relate several non-reference image quality metrics, including NIQE, image
gradient, entropy, contrast, and TOPSIS to judge the cause of
misclassification. Furthermore, we build a Random Forest classifier to identify
positive and negative samples based on image qualities only. We also display
the feature importance and conduct an error analysis. Besides, we present
sensitivity analysis to verify the robustness of our models. Finally, we show
the strengths and weaknesses of our model and derives the conclusions.

    

### [[2111.13581] Machine learning-based porosity estimation from spectral decomposed seismic data](http://arxiv.org/abs/2111.13581)


  Estimating porosity models via seismic data is challenging due to the signal
noise and insufficient resolution of seismic data. Although impedance inversion
is often used by combining with well logs, several hurdles remain to retrieve
sub-seismic scale porosity. As an alternative, we propose a machine
learning-based workflow to convert seismic data to porosity models. A ResUNet++
based workflow is designed to take three seismic data in different frequencies
(i.e., decomposed seismic data) and estimate their corresponding porosity
model. The workflow is successfully demonstrated in the 3D channelized
reservoir to estimate the porosity model with more than 0.9 in R2 score for
training and validating data. Moreover, the application is extended for a
stress test by adding signal noise to the seismic data, and the workflow
results show a robust estimation even with 5\% of noise. Another two ResUNet++
are trained to take either the lowest or highest resolution seismic data only
to estimate the porosity model, but they show under- and over-fitting results,
supporting the importance of using decomposed seismic data in porosity
estimation.

    

### [[2111.13585] Evaluating importance of nodes in complex networks with local volume information dimension](http://arxiv.org/abs/2111.13585)


  How to evaluate the importance of nodes is essential in research of complex
network. There are many methods proposed for solving this problem, but they
still have room to be improved. In this paper, a new approach called local
volume information dimension is proposed. In this method, the sum of degree of
nodes within different distances of central node is calculated. The information
within the certain distance is described by the information entropy. Compared
to other methods, the proposed method considers the information of the nodes
from different distances more comprehensively. For the purpose of showing the
effectiveness of the proposed method, experiments on real-world networks are
implemented. Promising results indicate the effectiveness of the proposed
method.

    

### [[2111.13611] Predicting Document Coverage for Relation Extraction](http://arxiv.org/abs/2111.13611)


  This paper presents a new task of predicting the coverage of a text document
for relation extraction (RE): does the document contain many relational tuples
for a given entity? Coverage predictions are useful in selecting the best
documents for knowledge base construction with large input corpora. To study
this problem, we present a dataset of 31,366 diverse documents for 520
entities. We analyze the correlation of document coverage with features like
length, entity mention frequency, Alexa rank, language complexity and
information retrieval scores. Each of these features has only moderate
predictive power. We employ methods combining features with statistical models
like TF-IDF and language models like BERT. The model combining features and
BERT, HERB, achieves an F1 score of up to 46%. We demonstrate the utility of
coverage predictions on two use cases: KB construction and claim refutation.

    

### [[2009.06544] Temporal Answer Set Programming](http://arxiv.org/abs/2009.06544)


  We present an overview on Temporal Logic Programming under the perspective of
its application for Knowledge Representation and declarative problem solving.
Such programs are the result of combining usual rules with temporal modal
operators, as in Linear-time Temporal Logic (LTL). We focus on recent results
of the non-monotonic formalism called Temporal Equilibrium Logic (TEL) that is
defined for the full syntax of LTL, but performs a model selection criterion
based on Equilibrium Logic, a well known logical characterization of Answer Set
Programming (ASP). We obtain a proper extension of the stable models semantics
for the general case of arbitrary temporal formulas. We recall the basic
definitions for TEL and its monotonic basis, the temporal logic of
Here-and-There (THT), and study the differences between infinite and finite
traces. We also provide other useful results, such as the translation into
other formalisms like Quantified Equilibrium Logic or Second-order LTL, and
some techniques for computing temporal stable models based on automata. In a
second part, we focus on practical aspects, defining a syntactic fragment
called temporal logic programs closer to ASP, and explain how this has been
exploited in the construction of the solver TELINGO.

    

### [[2102.10062] A Review of Biomedical Datasets Relating to Drug Discovery: A Knowledge Graph Perspective](http://arxiv.org/abs/2102.10062)


  Drug discovery and development is a complex and costly process. Machine
learning approaches are being investigated to help improve the effectiveness
and speed of multiple stages of the drug discovery pipeline. Of these, those
that use Knowledge Graphs (KG) have promise in many tasks, including drug
repurposing, drug toxicity prediction and target gene-disease prioritisation.
In a drug discovery KG, crucial elements including genes, diseases and drugs
are represented as entities, whilst relationships between them indicate an
interaction. However, to construct high-quality KGs, suitable data is required.
In this review, we detail publicly available sources suitable for use in
constructing drug discovery focused KGs. We aim to help guide machine learning
and KG practitioners who are interested in applying new techniques to the drug
discovery field, but who may be unfamiliar with the relevant data sources. The
datasets are selected via strict criteria, categorised according to the primary
type of information contained within and are considered based upon what
information could be extracted to build a KG. We then present a comparative
analysis of existing public drug discovery KGs and a evaluation of selected
motivating case studies from the literature. Additionally, we raise numerous
and unique challenges and issues associated with the domain and its datasets,
whilst also highlighting key future research directions. We hope this review
will motivate KGs use in solving key and emerging questions in the drug
discovery domain.

    

### [[2104.00351] TrajeVAE: Controllable Human Motion Generation from Trajectories](http://arxiv.org/abs/2104.00351)


  The creation of plausible and controllable 3D human motion animations is a
long-standing problem that requires a manual intervention of skilled artists.
Current machine learning approaches can semi-automate the process, however,
they are limited in a significant way: they can handle only a single trajectory
of the expected motion that precludes fine-grained control over the output. To
mitigate that issue, we reformulate the problem of future pose prediction into
pose completion in space and time where multiple trajectories are represented
as poses with missing joints. We show that such a framework can generalize to
other neural networks designed for future pose prediction. Once trained in this
framework, a model is capable of predicting sequences from any number of
trajectories. We propose a novel transformer-like architecture, TrajeVAE, that
builds on this idea and provides a versatile framework for 3D human animation.
We demonstrate that TrajeVAE offers better accuracy than the trajectory-based
reference approaches and methods that base their predictions on past poses. We
also show that it can predict reasonable future poses even if provided only
with an initial pose.

    

### [[2106.11485] Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis](http://arxiv.org/abs/2106.11485)


  High-resolution satellite imagery has proven useful for a broad range of
tasks, including measurement of global human population, local economic
livelihoods, and biodiversity, among many others. Unfortunately,
high-resolution imagery is both infrequently collected and expensive to
purchase, making it hard to efficiently and effectively scale these downstream
tasks over both time and space. We propose a new conditional pixel synthesis
model that uses abundant, low-cost, low-resolution imagery to generate accurate
high-resolution imagery at locations and times in which it is unavailable. We
show that our model attains photo-realistic sample quality and outperforms
competing baselines on a key downstream task -- object counting -- particularly
in geographic locations where conditions on the ground are changing rapidly.

    

### [[2004.07353] The nucleus of an adjunction and the Street monad on monads](http://arxiv.org/abs/2004.07353)


  An adjunction is a pair of functors related by a pair of natural
transformations, and relating a pair of categories. It displays how a
structure, or a concept, projects from each category to the other, and back.
Adjunctions are the common denominator of Galois connections, representation
theories, spectra, and generalized quantifiers. We call an adjunction nuclear
when its categories determine each other. We show that every adjunction can be
resolved into a nuclear adjunction. This resolution is idempotent in a strong
sense. The nucleus of an adjunction displays its conceptual core, just as the
singular value decomposition of an adjoint pair of linear operators displays
their canonical bases.
The two composites of an adjoint pair of functors induce a monad and a
comonad. Monads and comonads generalize the closure and the interior operators
from topology, or modalities from logic, while providing a saturated view of
algebraic structures and compositions on one side, and of coalgebraic dynamics
and decompositions on the other. They are resolved back into adjunctions over
the induced categories of algebras and of coalgebras. The nucleus of an
adjunction is an adjunction between the induced categories of algebras and
coalgebras. It provides new presentations for both, revealing the meaning of
constructing algebras for a comonad and coalgebras for a monad.
In his seminal early work, Ross Street described an adjunction between monads
and comonads in 2-categories. Lifting the nucleus construction, we show that
the resulting Street monad on monads is strongly idempotent, and extracts the
nucleus of a monad. A dual treatment achieves the same for comonads. Applying a
notable fragment of pure 2-category theory on an acute practical problem of
data analysis thus led to new theoretical result.

    

### [[2111.13040] Sketch-Guided Equality Saturation: Scaling Equality Saturation to Complex Optimizations in Languages with Bindings](http://arxiv.org/abs/2111.13040)


  Equality saturation is a technique for implementing rewrite-driven compiler
optimizations by efficiently representing many equivalent programs in so-called
e-graphs. To improve performance, the set of equivalent programs is grown by
applying rewrites in a purely additive way until a fixed point is reached
(saturation), or the search times out. In practice, two issues limit the
application of equality saturation in programming language compilers. First,
equality saturation is not efficient for the name bindings (variables) that
appear in almost all programming languages. Second, equality saturation does
not scale to complex optimizations with long rewrite sequences such as loop
blocking. This paper addresses both issues, thereby enabling equality
saturation to be applied to more realistic programs and compiler optimizations.
First, we demonstrate how to drastically improve the efficiency of equality
saturation for a functional language based on the typed lambda calculus.
Second, we introduce sketch-guided equality saturation, a semi-automatic
technique that allows programmers to provide sketches guiding rewriting when
performing complex optimizations. We evaluate sketch-guided equality saturation
by performing a series of realistic optimizations of matrix multiplication
expressed in the Rise functional language. The optimizations include loop
blocking, vectorization, and parallelization. We demonstrate that naive
equality saturation does not scale to these optimizations, even with hours of
exploration time. Previous work on orchestrating rewrite sequences shows that
these optimizations can be expressed as rewrites, at the cost of weeks of
programmer effort. Our guided equality saturation combines the advantages of
both techniques: minimal programmer guidance enables complex compiler
optimizations to be applied in seconds.

    

### [[2104.08037] The generalized join the shortest orbit queue system: Stability, exact tail asymptotics and stationary approximations](http://arxiv.org/abs/2104.08037)


  We introduce the \textit{generalized join the shortest queue model with
retrials} and two infinite capacity orbit queues. Three independent Poisson
streams of jobs, namely a \textit{smart}, and two \textit{dedicated} streams,
flow into a single server system, which can hold at most one job. Arriving jobs
that find the server occupied are routed to the orbits as follows: Blocked jobs
from the \textit{smart} stream are routed to the shortest orbit queue, and in
case of a tie, they choose an orbit randomly. Blocked jobs from the
\textit{dedicated} streams are routed directly to their orbits. Orbiting jobs
retry to connect with the server at different retrial rates, i.e.,
heterogeneous orbit queues. Applications of such a system are found in the
modelling of wireless cooperative networks. We are interested in the asymptotic
behaviour of the stationary distribution of this model, provided that the
system is stable. More precisely, we investigate the conditions under which the
tail asymptotic of the minimum orbit queue length is exactly geometric.
Moreover, we apply a heuristic asymptotic approach to obtain approximations of
the steady-state joint orbit queue-length distribution. Useful numerical
examples are presented, and shown that the results obtained through the
asymptotic analysis and the heuristic approach agreed.

    

### [[2111.13384] EOLANG and phi-calculus](http://arxiv.org/abs/2111.13384)


  Object-oriented programming (OOP) is one of the most popular paradigms used
for building software systems. However, despite its industrial and academic
popularity, OOP is still missing a formal apparatus similar to lambda-calculus,
which functional programming is based on. There were a number of attempts to
formalize OOP, but none of them managed to cover all the features available in
modern OO programming languages, such as C++ or Java. We have made yet another
attempt and created phi-calculus. We also created EOLANG (also called EO), an
experimental programming language based on phi-calculus.

    

### [[2111.13662] Modular Information Flow Through Ownership](http://arxiv.org/abs/2111.13662)


  Statically analyzing information flow, or how data influences other data
within a program, is a challenging task in imperative languages. Analyzing
pointers and mutations requires access to a program's complete source. However,
programs often use pre-compiled dependencies where only type signatures are
available. We demonstrate that ownership types can be used to soundly and
precisely analyze information flow through function calls given only their type
signature. From this insight, we built Flowistry, a system for analyzing
information flow in Rust, an ownership-based language. We prove the system's
soundness as a form of noninterference using the Oxide formal model of Rust.
Then we empirically evaluate the precision of Flowistry, showing that modular
flows are identical to whole-program flows in 90% of cases drawn from large
Rust codebases. We illustrate the applicability of Flowistry by implementing
both a program slicer and an IFC checker on top of it.

    

### [[2009.04826] Theory Exploration Powered By Deductive Synthesis](http://arxiv.org/abs/2009.04826)


  Recent years have seen tremendous growth in the amount of verified software.
Proofs for complex properties can now be achieved using higher-order theories
and calculi. Complex properties lead to an ever-growing number of definitions
and associated lemmas, which constitute an integral part of proof construction.
Following this -- whether automatic or semi-automatic -- methods for
computer-aided lemma discovery have emerged. In this work, we introduce a new
symbolic technique for bottom-up lemma discovery, that is, the generation of a
library of lemmas from a base set of inductive data types and recursive
definitions. This is known as the theory exploration problem, and so far,
solutions have been proposed based either on counter-example generation or the
more prevalent random testing combined with first-order solvers. Our new
approach, being purely deductive, eliminates the need for random testing as a
filtering phase and for SMT solvers. Therefore it is amenable compositional
reasoning and for the treatment of user-defined higher-order functions. Our
implementation has shown to find more lemmas than prior art, while avoiding
redundancy.

    

### [[2101.06249] Manifestly Phased Communication via Shared Session Types](http://arxiv.org/abs/2101.06249)


  Session types denote message protocols between concurrent processes, allowing
a type-safe expression of inter-process communication. Although previous work
demonstrate a well-defined notion of subtyping where processes have different
perceptions of the protocol, these formulations were limited to linear session
types where each channel of communication has a unique provider and client. In
this paper, we extend subtyping to shared session types where channels can now
have multiple clients instead of a single client. We demonstrate that this
generalization can statically capture protocol requirements that span multiple
phases of interactions of a client with a shared service provider, something
not possible in prior proposals. Moreover, the phases are manifest in the type
of the client.

    

### [<title>Max_detla_step params does not work - XGBoost</title>](https://discuss.xgboost.ai/t/max-detla-step-params-does-not-work/2566/1)