
## 2021-8-3

### [[2108.00098] Internet of things: a multiprotocol gateway as solution of the interoperability problem](http://arxiv.org/abs/2108.00098)


  One of the main challenges of the Internet of Things is the interoperability
of highly heterogeneous devices, mainly in terms of the communication
capabilities and network protocols used. As consequence, the interconnection
model of the different devices involves an intermediary device, known as
gateway. This gateway is a centralized element for the management of the
devices that make up an IoT application. In addition, it is essential for the
transmission of information to the Internet, especially when many IoT devices
are not IP-based. This chapter describes a proposed model for an IoT gateway
that allows the exchange of data through different wireless technologies and
forwarding of such data to the Internet.
The proposed gateway has important advantages such as: supporting for
multiprotocol interconnectivity, the remote configuration of wireless nodes for
sensor and actuators management; a flexible algorithm totranslate the data
obtained by sensors into a uniform format for transmission to a cloud server;
low energy consumption due to efficient data transfer over MQTT protocol.
In order to demonstrate the usefulness of the developed gateway, a proof of
concept test was implemented. The implemented scenario consists of 2 wireless
nodes responsible for sensing environmental variables and transmitting data to
the gateway node through different communication protocols. The obtained
results show the feasibility for simultaneous data transmission from the remote
wireless nodes to the gateway. Metrics on energy consumption in the devices are
also presented.

    

### [[2108.00149] Eavesdropping with Intelligent Reflective Surfaces: Threats and Defense Strategies](http://arxiv.org/abs/2108.00149)


  Intelligent reflecting surfaces (IRSs) have several prominent advantages,
including improving the level of wireless communications security and privacy.
In this work, we focus on this aspect and envision a strategy to counteract the
presence of passive eavesdroppers overhearing transmissions from a base station
towards legitimate users. Unlike most of the existing works addressing passive
eavesdropping, the strategy we consider has low complexity and is suitable for
scenarios where nodes are equipped with a limited number of antennas. Through
our performance evaluation, we highlight the trade-off between the legitimate
users' data rate and secrecy rate, and how the system parameters affect such a
trade-off.

    

### [[2108.00231] Distributed Learning for Time-varying Networks: A Scalable Design](http://arxiv.org/abs/2108.00231)


  The wireless network is undergoing a trend from "onnection of things" to
"connection of intelligence". With data spread over the communication networks
and computing capability enhanced on the devices, distributed learning becomes
a hot topic in both industrial and academic communities. Many frameworks, such
as federated learning and federated distillation, have been proposed. However,
few of them takes good care of obstacles such as the time-varying topology
resulted by the characteristics of wireless networks. In this paper, we propose
a distributed learning framework based on a scalable deep neural network (DNN)
design. By exploiting the permutation equivalence and invariance properties of
the learning tasks, the DNNs with different scales for different clients can be
built up based on two basic parameter sub-matrices. Further, model aggregation
can also be conducted based on these two sub-matrices to improve the learning
convergence and performance. Finally, simulation results verify the benefits of
the proposed framework by compared with some baselines.

    

### [[2108.00453] A Comparison Study of Cellular Deployments in Chicago and Miami Using Apps on Smartphones](http://arxiv.org/abs/2108.00453)


  Cellular operators have begun deploying 5G New Radio (NR) in all available
bands: low (< 1 GHz), mid (1 - 6 GHz), and high (> 24 GHz) to exploit the
different capabilities of each. At the same time, traditional 4G Long Term
Evolution (LTE) deployments are being enhanced with the addition of bands in
the unlicensed 5 GHz (using License Assisted Access, or LAA) and the 3.5 GHz
Citizens Broadband Radio Service (CBRS) resulting in throughput performance
comparable to 5G in mid-band. We present a detailed study comparing 4G and 5G
deployments, in all bands in Chicago, and focused mmWave measurements and
analysis in Miami. Our methodology, based on commercial and custom apps, is
scalable for crowdsourcing measurements on a large scale and provides detailed
data (throughput, latency, signal strength, etc.) on actual deployments. Our
main conclusions based on the measurements are (i) optimized 4G networks in
mid-band are comparable in both throughput and latency to current deployments
of 5G (both standalone (SA) and non-standalone (NSA)) and (ii) mmWave 5G, even
in NSA mode, can deliver multi-Gbps throughput reliably if the installation is
dense enough, but performance is still brittle due to the propagation
limitations imposed by distance and body loss. Thus, while 5G demonstrates
significant early promise, further work needs to be done to ensure that the
stated goals of 5G are met.

    

### [[2108.00483] Modeling and Analysis of mMTC Traffic in 5G Base Stations](http://arxiv.org/abs/2108.00483)


  Massive Machine Type Communications (mMTC) are one of the three types of
services that should be supported by 5G networks. These are distinguished by
the need to serve a large number of devices which are characterized by
nonintensive traffic and low energy consumption. While the sporadic nature of
the mMTC traffic does not pose an exertion to efficient network operation,
multiplexing the traffic from a large number of these devices within the cell
certainly does. Therefore, planning carefully the network resources for this
traffic is of paramount importance. To do this, the statistics of the traffic
pattern that arrives at the base station should be known. To this end, in this
paper, we derive the distribution of the inter-arrival times of the traffic at
the base station from a general number of mMTC users within the cell, assuming
a generic distribution of the traffic pattern by individual users. We validate
our results on traces. Results show that adding more mMTC users in the cell
increases the variability of the traffic pattern at the base station almost
linearly, which is not the case with increasing the traffic generation rates.

    

### [[2108.00566] Efficient On-Chip Multicast Routing based on Dynamic Partition Merging](http://arxiv.org/abs/2108.00566)


  Networks-on-chips (NoCs) have become the mainstream communication
infrastructure for chip multiprocessors (CMPs) and many-core systems. The
commonly used parallel applications and emerging machine learning-based
applications involve a significant amount of collective communication patterns.
In CMP applications, multicast is widely used in multithreaded programs and
protocols for barrier/clock synchronization and cache coherence. Multicast
routing plays an important role on the system performance of a CMP. Existing
partition-based multicast routing algorithms all use static destination set
partition strategy which lacks the global view of path optimization. In this
paper, we propose an efficient Dynamic Partition Merging (DPM)-based multicast
routing algorithm. The proposed algorithm divides the multicast destination set
into partitions dynamically by comparing the routing cost of different
partition merging options and selecting the merged partitions with lower cost.
The simulation results of synthetic traffic and PARSEC benchmark applications
confirm that the proposed algorithm outperforms the existing path-based routing
algorithms. The proposed algorithm is able to improve up to 23\% in average
packet latency and 14\% in power consumption against the existing multipath
routing algorithm when tested in PARSEC benchmark workloads.

    

### [[2108.00591] Resource Management in Edge and Fog Computing using FogBus2 Framework](http://arxiv.org/abs/2108.00591)


  Edge/Fog computing is a novel computing paradigm that provides
resource-limited Internet of Things (IoT) devices with scalable computing and
storage resources. Compared to cloud computing, edge/fog servers have fewer
resources, but they can be accessed with higher bandwidth and less
communication latency. Thus, integrating edge/fog and cloud infrastructures can
support the execution of diverse latency-sensitive and computation-intensive
IoT applications. Although some frameworks attempt to provide such integration,
there are still several challenges to be addressed, such as dynamic scheduling
of different IoT applications, scalability mechanisms, multi-platform support,
and supporting different interaction models. FogBus2, as a new python-based
framework, offers a lightweight and distributed container-based framework to
overcome these challenges. In this chapter, we highlight key features of the
FogBus2 framework alongside describing its main components. Besides, we provide
a step-by-step guideline to set up an integrated computing environment,
containing multiple cloud service providers (Hybrid-cloud) and edge devices,
which is a prerequisite for any IoT application scenario. To obtain this, a
low-overhead communication network among all computing resources is initiated
by the provided scripts and configuration files. Next, we provide instructions
and corresponding code snippets to install and run the main framework and its
integrated applications. Finally, we demonstrate how to implement and integrate
several new IoT applications and custom scheduling and scalability policies
with the FogBus2 framework.

    

### [[2108.00694] A Novel Internet-of-Drones and Blockchain-based System Architecture for Search and Rescue](http://arxiv.org/abs/2108.00694)


  With the development in information and communications technology (ICT) and
drones such as Internet-of-Things (IoT), edge computing, image processing, and
autonomous drones, solutions supporting search and rescue (SAR) missions can be
developed with more intelligent capabilities. In most of the drone and unmanned
aerial vehicle (UAV) based systems supporting SAR missions, several drones
deployed in different areas acquire images and videos that are sent to a ground
control station (GCS) for processing and detecting a missing person. Although
this offers many advantages, such as easy management and deployment, the
approach still has many limitations. For example, when a connection between a
drone and a GCS has some problems, the quality of service cannot be maintained.
Many drone and UAV-based systems do not support flexibility, transparency,
security, and traceability. In this paper, we propose a novel
Internet-of-Drones (IoD) architecture using blockchain technology. We implement
the proposed architecture with different drones, edge servers, and a
Hyperledger blockchain network. The proof-of-concept design demonstrates that
the proposed architecture can offer high-level services such as prolonging the
operating time of a drone, improving the capability of detecting humans
accurately, and a high level of transparency, traceability, and security.

    

### [[1909.08928] SCDP: Systematic Rateless Coding for Efficient Data Transport in Data Centres (Complete Version)](http://arxiv.org/abs/1909.08928)


  In this paper we propose SCDP, a general-purpose data transport protocol for
data centres that, in contrast to all other protocols proposed to date,
supports efficient one-to-many and many-to-one communication, which is
extremely common in modern data centres. SCDP does so without compromising on
efficiency for short and long unicast flows. SCDP achieves this by integrating
RaptorQ codes with receiver-driven data transport, packet trimming and
Multi-Level Feedback Queuing (MLFQ); (1) RaptorQ codes enable efficient
one-to-many and many-to-one data transport; (2) on top of RaptorQ codes,
receiver-driven flow control, in combination with in-network packet trimming,
enable efficient usage of network resources as well as multi-path transport and
packet spraying for all transport modes. Incast and Outcast are eliminated; (3)
the systematic nature of RaptorQ codes, in combination with MLFQ, enable fast,
decoding-free completion of short flows. We extensively evaluate SCDP in a wide
range of simulated scenarios with realistic data centre workloads. For
one-to-many and many-to-one transport sessions, SCDP performs significantly
better compared to NDP and PIAS. For short and long unicast flows, SCDP
performs equally well or better compared to NDP and PIAS.

    

### [[2006.06131] Sovereign: User-Controlled Smart Homes](http://arxiv.org/abs/2006.06131)


  Recent years have witnessed the rapid deployment of smart homes; most of them
are controlled by remote servers in the cloud. Such designs raise security and
privacy concerns for end users. In this paper, we describe the design of
Sovereign, a home IoT system framework that provides end users complete control
of their home IoT systems. Sovereign lets home IoT devices and applications
communicate via application-named data and secures data directly. This enables
direct, secure, one-to-one and one-to-many device-to-device communication over
wireless broadcast media. Sovereign utilizes semantic names to construct usable
security solutions. We implement Sovereign as a publish-subscribe-based
development platform together with a prototype home IoT controller. Our
preliminary evaluation shows that Sovereign provides a systematic, easy-to-use
solution to user-controlled, self-contained smart homes running on existing IoT
hardware without imposing noticeable overhead.

    

### [[2011.08608] The Potential of Multilayered Hierarchical Nonterrestrial Networks for 6G: A Comparative Analysis Among Networking Architectures](http://arxiv.org/abs/2011.08608)


  6th generation (6G) communication research is currently focusing on
non-terrestrial networks (NTNs) to promote ubiquitous and ultra-high-capacity
global connectivity. Specifically, multi-layered hierarchical networks, i.e.,
the orchestration among different aerial/space platforms, including Low and
High Altitude Platforms (LAPs and HAPs), and satellites co-operating at
different altitudes, currently represent one the most attractive technological
options to solve coverage and latency constraints associated with the NTN
paradigm. However, there are still several issues to be resolved for proper
network design. In this work, we evaluate the performance of different
multi-layered non-terrestrial configurations, and provide guidelines on the
optimal working point(s) for which it is possible to achieve a good compromise
between improved system flexibility and network performance, with respect to a
baseline standalone deployment.

    

### [[2101.01286] A Survey on Integrated Access and Backhaul Networks](http://arxiv.org/abs/2101.01286)


  Benefiting from the usage of the high-frequency band, utilizing part of the
large available bandwidth for wireless backhauling is feasible without
considerable performance sacrifice. In this context, integrated access and
backhaul (IAB) was proposed by 3GPP to reduce the fiber optics deployment cost
of 5G and beyond networks. In this paper, we first give a brief introduction of
IAB based on the 3GPP release. After that, we survey existing research on IAB
networks, the integrations of IAB to cache-enabled network, optical
communication transport network, and the non-terrestrial network. Finally, we
discuss the challenges and opportunities that might arise while developing and
commercializing IAB networks.

    

### [[2108.00002] Bayesian Optimization in Materials Science: A Survey](http://arxiv.org/abs/2108.00002)


  Bayesian optimization is used in many areas of AI for the optimization of
black-box processes and has achieved impressive improvements of the state of
the art for a lot of applications. It intelligently explores large and complex
design spaces while minimizing the number of evaluations of the expensive
underlying process to be optimized. Materials science considers the problem of
optimizing materials' properties given a large design space that defines how to
synthesize or process them, with evaluations requiring expensive experiments or
simulations -- a very similar setting. While Bayesian optimization is also a
popular approach to tackle such problems, there is almost no overlap between
the two communities that are investigating the same concepts. We present a
survey of Bayesian optimization approaches in materials science to increase
cross-fertilization and avoid duplication of work. We highlight common
challenges and opportunities for joint research efforts.

    

### [[2108.00037] Physics-Informed Machine Learning Method for Large-Scale Data Assimilation Problems](http://arxiv.org/abs/2108.00037)


  We develop a physics-informed machine learning approach for large-scale data
assimilation and parameter estimation and apply it for estimating
transmissivity and hydraulic head in the two-dimensional steady-state
subsurface flow model of the Hanford Site given synthetic measurements of said
variables. In our approach, we extend the physics-informed conditional
Karhunen-Loéve expansion (PICKLE) method for modeling subsurface flow with
unknown flux (Neumann) and varying head (Dirichlet) boundary conditions. We
demonstrate that the PICKLE method is comparable in accuracy with the standard
maximum a posteriori (MAP) method, but is significantly faster than MAP for
large-scale problems. Both methods use a mesh to discretize the computational
domain. In MAP, the parameters and states are discretized on the mesh;
therefore, the size of the MAP parameter estimation problem directly depends on
the mesh size. In PICKLE, the mesh is used to evaluate the residuals of the
governing equation, while the parameters and states are approximated by the
truncated conditional Karhunen-Loéve expansions with the number of
parameters controlled by the smoothness of the parameter and state fields, and
not by the mesh size. For a considered example, we demonstrate that the
computational cost of PICKLE increases near linearly (as $N_{FV}^{1.15}$) with
the number of grid points $N_{FV}$, while that of MAP increases much faster as
$N_{FV}^{3.28}$. We demonstrated that once trained for one set of Dirichlet
boundary conditions (i.e., one river stage), the PICKLE method provides
accurate estimates of the hydraulic head for any value of the Dirichlet
boundary conditions (i.e., for any river stage).

    

### [[2108.00043] Toward Robust Autotuning of Noisy Quantum Dot Devices](http://arxiv.org/abs/2108.00043)


  The current autotuning approaches for quantum dot (QD) devices, while showing
some success, lack an assessment of data reliability. This leads to unexpected
failures when noisy data is processed by an autonomous system. In this work, we
propose a framework for robust autotuning of QD devices that combines a machine
learning (ML) state classifier with a data quality control module. The data
quality control module acts as a ``gatekeeper'' system, ensuring that only
reliable data is processed by the state classifier. Lower data quality results
in either device recalibration or termination. To train both ML systems, we
enhance the QD simulation by incorporating synthetic noise typical of QD
experiments. We confirm that the inclusion of synthetic noise in the training
of the state classifier significantly improves the performance, resulting in an
accuracy of 95.1(7) % when tested on experimental data. We then validate the
functionality of the data quality control module by showing the state
classifier performance deteriorates with decreasing data quality, as expected.
Our results establish a robust and flexible ML framework for autonomous tuning
of noisy QD devices.

    

### [[2108.00045] Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning](http://arxiv.org/abs/2108.00045)


  Zero-Shot Learning (ZSL) aims to recognise unseen object classes, which are
not observed during the training phase. The existing body of works on ZSL
mostly relies on pretrained visual features and lacks the explicit attribute
localisation mechanism on images. In this work, we propose an attention-based
model in the problem settings of ZSL to learn attributes useful for unseen
class recognition. Our method uses an attention mechanism adapted from Vision
Transformer to capture and learn discriminative attributes by splitting images
into small patches. We conduct experiments on three popular ZSL benchmarks
(i.e., AWA2, CUB and SUN) and set new state-of-the-art harmonic mean results
{on all the three datasets}, which illustrate the effectiveness of our proposed
method.

    

### [[2108.00049] Object-aware Contrastive Learning for Debiased Scene Representation](http://arxiv.org/abs/2108.00049)


  Contrastive self-supervised learning has shown impressive results in learning
visual representations from unlabeled images by enforcing invariance against
different data augmentations. However, the learned representations are often
contextually biased to the spurious scene correlations of different objects or
object and background, which may harm their generalization on the downstream
tasks. To tackle the issue, we develop a novel object-aware contrastive
learning framework that first (a) localizes objects in a self-supervised manner
and then (b) debias scene correlations via appropriate data augmentations
considering the inferred object locations. For (a), we propose the contrastive
class activation map (ContraCAM), which finds the most discriminative regions
(e.g., objects) in the image compared to the other images using the
contrastively trained models. We further improve the ContraCAM to detect
multiple objects and entire shapes via an iterative refinement procedure. For
(b), we introduce two data augmentations based on ContraCAM, object-aware
random crop and background mixup, which reduce contextual and background biases
during contrastive self-supervised learning, respectively. Our experiments
demonstrate the effectiveness of our representation learning framework,
particularly when trained under multi-object images or evaluated under the
background (and distribution) shifted images.

    

### [[2108.00051] Coordinate descent on the orthogonal group for recurrent neural network training](http://arxiv.org/abs/2108.00051)


  We propose to use stochastic Riemannian coordinate descent on the orthogonal
group for recurrent neural network training. The algorithm rotates successively
two columns of the recurrent matrix, an operation that can be efficiently
implemented as a multiplication by a Givens matrix. In the case when the
coordinate is selected uniformly at random at each iteration, we prove the
convergence of the proposed algorithm under standard assumptions on the loss
function, stepsize and minibatch noise. In addition, we numerically demonstrate
that the Riemannian gradient in recurrent neural network training has an
approximately sparse structure. Leveraging this observation, we propose a
faster variant of the proposed algorithm that relies on the Gauss-Southwell
rule. Experiments on a benchmark recurrent neural network training problem are
presented to demonstrate the effectiveness of the proposed algorithm.

    

### [[2108.00065] Pruning Neural Networks with Interpolative Decompositions](http://arxiv.org/abs/2108.00065)


  We introduce a principled approach to neural network pruning that casts the
problem as a structured low-rank matrix approximation. Our method uses a novel
application of a matrix factorization technique called the interpolative
decomposition to approximate the activation output of a network layer. This
technique selects neurons or channels in the layer and propagates a corrective
interpolation matrix to the next layer, resulting in a dense, pruned network
with minimal degradation before fine tuning. We demonstrate how to prune a
neural network by first building a set of primitives to prune a single fully
connected or convolution layer and then composing these primitives to prune
deep multi-layer networks. Theoretical guarantees are provided for pruning a
single hidden layer fully connected network. Pruning with interpolative
decompositions achieves strong empirical results compared to the
state-of-the-art on multiple applications from one and two hidden layer
networks on Fashion MNIST to VGG and ResNets on CIFAR-10. Notably, we achieve
an accuracy of 93.62 $\pm$ 0.36% using VGG-16 on CIFAR-10, with a 51% FLOPS
reduction. This gains 0.02% from the full-sized model.

    

### [[2108.00069] DySMHO: Data-Driven Discovery of Governing Equations for Dynamical Systems via Moving Horizon Optimization](http://arxiv.org/abs/2108.00069)


  Discovering the governing laws underpinning physical and chemical phenomena
is a key step towards understanding and ultimately controlling systems in
science and engineering. We introduce Discovery of Dynamical Systems via Moving
Horizon Optimization (DySMHO), a scalable machine learning framework for
identifying governing laws in the form of differential equations from
large-scale noisy experimental data sets. DySMHO consists of a novel moving
horizon dynamic optimization strategy that sequentially learns the underlying
governing equations from a large dictionary of basis functions. The sequential
nature of DySMHO allows leveraging statistical arguments for eliminating
irrelevant basis functions, avoiding overfitting to recover accurate and
parsimonious forms of the governing equations. Canonical nonlinear dynamical
system examples are used to demonstrate that DySMHO can accurately recover the
governing laws, is robust to high levels of measurement noise and that it can
handle challenges such as multiple time scale dynamics.

    

### [[2108.00071] Foundations of data imbalance and solutions for a data democracy](http://arxiv.org/abs/2108.00071)


  Dealing with imbalanced data is a prevalent problem while performing
classification on the datasets. Many times, this problem contributes to bias
while making decisions or implementing policies. Thus, it is vital to
understand the factors which cause imbalance in the data (or class imbalance).
Such hidden biases and imbalances can lead to data tyranny and a major
challenge to a data democracy. In this chapter, two essential statistical
elements are resolved: the degree of class imbalance and the complexity of the
concept; solving such issues helps in building the foundations of a data
democracy. Furthermore, statistical measures which are appropriate in these
scenarios are discussed and implemented on a real-life dataset (car insurance
claims). In the end, popular data-level methods such as random oversampling,
random undersampling, synthetic minority oversampling technique, Tomek link,
and others are implemented in Python, and their performance is compared.

    

### [[2108.00079] Zooming Into the Darknet: Characterizing Internet Background Radiation and its Structural Changes](http://arxiv.org/abs/2108.00079)


  Network telescopes or "Darknets" provide a unique window into Internet-wide
malicious activities associated with malware propagation, denial of service
attacks, scanning performed for network reconnaissance, and others. Analyses of
the resulting data can provide actionable insights to security analysts that
can be used to prevent or mitigate cyber-threats. Large Darknets, however,
observe millions of nefarious events on a daily basis which makes the
transformation of the captured information into meaningful insights
challenging. We present a novel framework for characterizing Darknet behavior
and its temporal evolution aiming to address this challenge. The proposed
framework: (i) Extracts a high dimensional representation of Darknet events
composed of features distilled from Darknet data and other external sources;
(ii) Learns, in an unsupervised fashion, an information-preserving
low-dimensional representation of these events (using deep representation
learning) that is amenable to clustering; (iv) Performs clustering of the
scanner data in the resulting representation space and provides interpretable
insights using optimal decision trees; and (v) Utilizes the clustering outcomes
as "signatures" that can be used to detect structural changes in the Darknet
activities. We evaluate the proposed system on a large operational Network
Telescope and demonstrate its ability to detect real-world, high-impact
cybersecurity incidents.

    

### [[2108.00080] A New Semi-supervised Learning Benchmark for Classifying View and Diagnosing Aortic Stenosis from Echocardiograms](http://arxiv.org/abs/2108.00080)


  Semi-supervised image classification has shown substantial progress in
learning from limited labeled data, but recent advances remain largely untested
for clinical applications. Motivated by the urgent need to improve timely
diagnosis of life-threatening heart conditions, especially aortic stenosis, we
develop a benchmark dataset to assess semi-supervised approaches to two tasks
relevant to cardiac ultrasound (echocardiogram) interpretation: view
classification and disease severity classification. We find that a
state-of-the-art method called MixMatch achieves promising gains in heldout
accuracy on both tasks, learning from a large volume of truly unlabeled images
as well as a labeled set collected at great expense to achieve better
performance than is possible with the labeled set alone. We further pursue
patient-level diagnosis prediction, which requires aggregating across hundreds
of images of diverse view types, most of which are irrelevant, to make a
coherent prediction. The best patient-level performance is achieved by new
methods that prioritize diagnosis predictions from images that are predicted to
be clinically-relevant views and transfer knowledge from the view task to the
diagnosis task. We hope our released Tufts Medical Echocardiogram Dataset and
evaluation framework inspire further improvements in multi-task semi-supervised
learning for clinical applications.

    

### [[2108.00089] Tensor-Train Density Estimation](http://arxiv.org/abs/2108.00089)


  Estimation of probability density function from samples is one of the central
problems in statistics and machine learning. Modern neural network-based models
can learn high dimensional distributions but have problems with hyperparameter
selection and are often prone to instabilities during training and inference.
We propose a new efficient tensor train-based model for density estimation
(TTDE). Such density parametrization allows exact sampling, calculation of
cumulative and marginal density functions, and partition function. It also has
very intuitive hyperparameters. We develop an efficient non-adversarial
training procedure for TTDE based on the Riemannian optimization. Experimental
results demonstrate the competitive performance of the proposed method in
density estimation and sampling tasks, while TTDE significantly outperforms
competitors in training speed.

    

### [[2108.00099] A Deep Learning Approach to Predict Blood Pressure from PPG Signals](http://arxiv.org/abs/2108.00099)


  Blood Pressure (BP) is one of the four primary vital signs indicating the
status of the body's vital (life-sustaining) functions. BP is difficult to
continuously monitor using a sphygmomanometer (i.e. a blood pressure cuff),
especially in everyday-setting. However, other health signals which can be
easily and continuously acquired, such as photoplethysmography (PPG), show some
similarities with the Aortic Pressure waveform. Based on these similarities, in
recent years several methods were proposed to predict BP from the PPG signal.
Building on these results, we propose an advanced personalized data-driven
approach that uses a three-layer deep neural network to estimate BP based on
PPG signals. Different from previous work, the proposed model analyzes the PPG
signal in time-domain and automatically extracts the most critical features for
this specific application, then uses a variation of recurrent neural networks
called Long-Short-Term-Memory (LSTM) to map the extracted features to the BP
value associated with that time window. Experimental results on two separate
standard hospital datasets, yielded absolute errors mean and absolute error
standard deviation for systolic and diastolic BP values outperforming prior
works.

    

### [[2108.00103] Extracting Grammars from a Neural Network Parser for Anomaly Detection in Unknown Formats](http://arxiv.org/abs/2108.00103)


  Reinforcement learning has recently shown promise as a technique for training
an artificial neural network to parse sentences in some unknown format. A key
aspect of this approach is that rather than explicitly inferring a grammar that
describes the format, the neural network learns to perform various parsing
actions (such as merging two tokens) over a corpus of sentences, with the goal
of maximizing the total reward, which is roughly based on the estimated
frequency of the resulting parse structures. This can allow the learning
process to more easily explore different action choices, since a given choice
may change the optimality of the parse (as expressed by the total reward), but
will not result in the failure to parse a sentence. However, the approach also
exhibits limitations: first, the neural network does not provide production
rules for the grammar that it uses during parsing; second, because this neural
network can successfully parse any sentence, it cannot be directly used to
identify sentences that deviate from the format of the training sentences,
i.e., that are anomalous. In this paper, we address these limitations by
presenting procedures for extracting production rules from the neural network,
and for using these rules to determine whether a given sentence is nominal or
anomalous, when compared to structures observed within training data. In the
latter case, an attempt is made to identify the location of the anomaly.
Additionally, a two pass mechanism is presented for dealing with formats
containing high-entropy information. We empirically evaluate the approach on
artificial formats, demonstrating effectiveness, but also identifying
limitations. By further improving parser learning, and leveraging rule
extraction and anomaly detection, one might begin to understand common errors,
either benign or malicious, in practical formats.

    

### [[2108.00105] Deep Feature Tracker: A Novel Application for Deep Convolutional Neural Networks](http://arxiv.org/abs/2108.00105)


  Feature tracking is the building block of many applications such as visual
odometry, augmented reality, and target tracking. Unfortunately, the
state-of-the-art vision-based tracking algorithms fail in surgical images due
to the challenges imposed by the nature of such environments. In this paper, we
proposed a novel and unified deep learning-based approach that can learn how to
track features reliably as well as learn how to detect such reliable features
for tracking purposes. The proposed network dubbed as Deep-PT, consists of a
tracker network which is a convolutional neural network simulating
cross-correlation in terms of deep learning and two fully connected networks
that operate on the output of intermediate layers of the tracker to detect
features and predict trackability of the detected points. The ability to detect
features based on the capabilities of the tracker distinguishes the proposed
method from previous algorithms used in this area and improves the robustness
of the algorithms against dynamics of the scene. The network is trained using
multiple datasets due to the lack of specialized dataset for feature tracking
datasets and extensive comparisons are conducted to compare the accuracy of
Deep-PT against recent pixel tracking algorithms. As the experiments suggest,
the proposed deep architecture deliberately learns what to track and how to
track and outperforms the state-of-the-art methods.

    

### [[2108.00106] Soft Calibration Objectives for Neural Networks](http://arxiv.org/abs/2108.00106)


  Optimal decision making requires that classifiers produce uncertainty
estimates consistent with their empirical accuracy. However, deep neural
networks are often under- or over-confident in their predictions. Consequently,
methods have been developed to improve the calibration of their predictive
uncertainty both during training and post-hoc. In this work, we propose
differentiable losses to improve calibration based on a soft (continuous)
version of the binning operation underlying popular calibration-error
estimators. When incorporated into training, these soft calibration losses
achieve state-of-the-art single-model ECE across multiple datasets with less
than 1% decrease in accuracy. For instance, we observe an 82% reduction in ECE
(70% relative to the post-hoc rescaled ECE) in exchange for a 0.7% relative
decrease in accuracy relative to the cross entropy baseline on CIFAR-100. When
incorporated post-training, the soft-binning-based calibration error objective
improves upon temperature scaling, a popular recalibration method. Overall,
experiments across losses and datasets demonstrate that using
calibration-sensitive procedures yield better uncertainty estimates under
dataset shift than the standard practice of using a cross entropy loss and
post-hoc recalibration methods.

    

### [[2108.00109] A Machine-learning Based Initialization for Joint Statistical Iterative Dual-energy CT with Application to Proton Therapy](http://arxiv.org/abs/2108.00109)


  Dual-energy CT (DECT) has been widely investigated to generate more
informative and more accurate images in the past decades. For example,
Dual-Energy Alternating Minimization (DEAM) algorithm achieves sub-percentage
uncertainty in estimating proton stopping-power mappings from experimental 3-mm
collimated phantom data. However, elapsed time of iterative DECT algorithms is
not clinically acceptable, due to their low convergence rate and the tremendous
geometry of modern helical CT scanners. A CNN-based initialization method is
introduced to reduce the computational time of iterative DECT algorithms. DEAM
is used as an example of iterative DECT algorithms in this work. The simulation
results show that our method generates denoised images with greatly improved
estimation accuracy for adipose, tonsils, and muscle tissue. Also, it reduces
elapsed time by approximately 5-fold for DEAM to reach the same objective
function value for both simulated and real data.

    

### [[2108.00128] Physics-informed Dyna-Style Model-Based Deep Reinforcement Learning for Dynamic Control](http://arxiv.org/abs/2108.00128)


  Model-based reinforcement learning (MBRL) is believed to have much higher
sample efficiency compared to model-free algorithms by learning a predictive
model of the environment. However, the performance of MBRL highly relies on the
quality of the learned model, which is usually built in a black-box manner and
may have poor predictive accuracy outside of the data distribution. The
deficiencies of the learned model may prevent the policy from being fully
optimized. Although some uncertainty analysis-based remedies have been proposed
to alleviate this issue, model bias still poses a great challenge for MBRL. In
this work, we propose to leverage the prior knowledge of underlying physics of
the environment, where the governing laws are (partially) known. In particular,
we developed a physics-informed MBRL framework, where governing equations and
physical constraints are utilized to inform the model learning and policy
search. By incorporating the prior information of the environment, the quality
of the learned model can be notably improved, while the required interactions
with the environment are significantly reduced, leading to better sample
efficiency and learning performance. The effectiveness and merit have been
demonstrated over a handful of classic control problems, where the environments
are governed by canonical ordinary/partial differential equations.

    

### [[2108.00131] Simple, Fast, and Flexible Framework for Matrix Completion with Infinite Width Neural Networks](http://arxiv.org/abs/2108.00131)


  Matrix completion problems arise in many applications including
recommendation systems, computer vision, and genomics. Increasingly larger
neural networks have been successful in many of these applications, but at
considerable computational costs. Remarkably, taking the width of a neural
network to infinity allows for improved computational performance. In this
work, we develop an infinite width neural network framework for matrix
completion that is simple, fast, and flexible. Simplicity and speed come from
the connection between the infinite width limit of neural networks and kernels
known as neural tangent kernels (NTK). In particular, we derive the NTK for
fully connected and convolutional neural networks for matrix completion. The
flexibility stems from a feature prior, which allows encoding relationships
between coordinates of the target matrix, akin to semi-supervised learning. The
effectiveness of our framework is demonstrated through competitive results for
virtual drug screening and image inpainting/reconstruction. We also provide an
implementation in Python to make our framework accessible on standard hardware
to a broad audience.

    

### [[2108.00138] Learning to Control Direct Current Motor for Steering in Real Time via Reinforcement Learning](http://arxiv.org/abs/2108.00138)


  Model free techniques have been successful at optimal control of complex
systems at an expense of copious amounts of data and computation. However, it
is often desired to obtain a control policy in a short period of time with
minimal data use and computational burden. To this end, we make use of the NFQ
algorithm for steering position control of a golf cart in both a real hardware
and a simulated environment that was built from real-world interaction. The
controller learns to apply a sequence of voltage signals in the presence of
environmental uncertainties and inherent non-linearities that challenge the the
control task. We were able to increase the rate of successful control under
four minutes in simulation and under 11 minutes in real hardware.

    

### [[2108.00144] Personalized Stress Monitoring using Wearable Sensors in Everyday Settings](http://arxiv.org/abs/2108.00144)


  Since stress contributes to a broad range of mental and physical health
problems, the objective assessment of stress is essential for behavioral and
physiological studies. Although several studies have evaluated stress levels in
controlled settings, objective stress assessment in everyday settings is still
largely under-explored due to challenges arising from confounding contextual
factors and limited adherence for self-reports. In this paper, we explore the
objective prediction of stress levels in everyday settings based on heart rate
(HR) and heart rate variability (HRV) captured via low-cost and easy-to-wear
photoplethysmography (PPG) sensors that are widely available on newer smart
wearable devices. We present a layered system architecture for personalized
stress monitoring that supports a tunable collection of data samples for
labeling, and present a method for selecting informative samples from the
stream of real-time data for labeling. We captured the stress levels of
fourteen volunteers through self-reported questionnaires over periods of
between 1-3 months, and explored binary stress detection based on HR and HRV
using Machine Learning Methods. We observe promising preliminary results given
that the dataset is collected in the challenging environments of everyday
settings. The binary stress detector is fairly accurate and can detect
stressful vs non-stressful samples with a macro-F1 score of up to \%76. Our
study lays the groundwork for more sophisticated labeling strategies that
generate context-aware, personalized models that will empower health
professionals to provide personalized interventions.

    

### [[2108.00151] An Empirical analysis on Transparent Algorithmic Exploration in Recommender Systems](http://arxiv.org/abs/2108.00151)


  All learning algorithms for recommendations face inevitable and critical
trade-off between exploiting partial knowledge of a user's preferences for
short-term satisfaction and exploring additional user preferences for long-term
coverage. Although exploration is indispensable for long success of a
recommender system, the exploration has been considered as the risk to decrease
user satisfaction. The reason for the risk is that items chosen for exploration
frequently mismatch with the user's interests. To mitigate this risk,
recommender systems have mixed items chosen for exploration into a
recommendation list, disguising the items as recommendations to elicit feedback
on the items to discover the user's additional tastes. This mix-in approach has
been widely used in many recommenders, but there is rare research, evaluating
the effectiveness of the mix-in approach or proposing a new approach for
eliciting user feedback without deceiving users. In this work, we aim to
propose a new approach for feedback elicitation without any deception and
compare our approach to the conventional mix-in approach for evaluation. To
this end, we designed a recommender interface that reveals which items are for
exploration and conducted a within-subject study with 94 MTurk workers. Our
results indicated that users left significantly more feedback on items chosen
for exploration with our interface. Besides, users evaluated that our new
interface is better than the conventional mix-in interface in terms of novelty,
diversity, transparency, trust, and satisfaction. Finally, path analysis show
that, in only our new interface, exploration caused to increase user-centric
evaluation metrics. Our work paves the way for how to design an interface,
which utilizes learning algorithm based on users' feedback signals, giving
better user experience and gathering more feedback data.

    

### [[2108.00154] CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](http://arxiv.org/abs/2108.00154)


  Transformers have made much progress in dealing with visual tasks. However,
existing vision transformers still do not possess an ability that is important
to visual input: building the attention among features of different scales. The
reasons for this problem are two-fold: (1) Input embeddings of each layer are
equal-scale without cross-scale features; (2) Some vision transformers
sacrifice the small-scale features of embeddings to lower the cost of the
self-attention module. To make up this defect, we propose Cross-scale Embedding
Layer (CEL) and Long Short Distance Attention (LSDA). In particular, CEL blends
each embedding with multiple patches of different scales, providing the model
with cross-scale embeddings. LSDA splits the self-attention module into a
short-distance and long-distance one, also lowering the cost but keeping both
small-scale and large-scale features in embeddings. Through these two designs,
we achieve cross-scale attention. Besides, we propose dynamic position bias for
vision transformers to make the popular relative position bias apply to
variable-sized images. Based on these proposed modules, we construct our vision
architecture called CrossFormer. Experiments show that CrossFormer outperforms
other transformers on several representative visual tasks, especially object
detection and segmentation. The code has been released:
this https URL.

    

### [[2108.00192] Learning with Noisy Labels via Sparse Regularization](http://arxiv.org/abs/2108.00192)


  Learning with noisy labels is an important and challenging task for training
accurate deep neural networks. Some commonly-used loss functions, such as Cross
Entropy (CE), suffer from severe overfitting to noisy labels. Robust loss
functions that satisfy the symmetric condition were tailored to remedy this
problem, which however encounter the underfitting effect. In this paper, we
theoretically prove that \textbf{any loss can be made robust to noisy labels}
by restricting the network output to the set of permutations over a fixed
vector. When the fixed vector is one-hot, we only need to constrain the output
to be one-hot, which however produces zero gradients almost everywhere and thus
makes gradient-based optimization difficult. In this work, we introduce the
sparse regularization strategy to approximate the one-hot constraint, which is
composed of network output sharpening operation that enforces the output
distribution of a network to be sharp and the $\ell_p$-norm ($p\le 1$)
regularization that promotes the network output to be sparse. This simple
approach guarantees the robustness of arbitrary loss functions while not
hindering the fitting ability. Experimental results demonstrate that our method
can significantly improve the performance of commonly-used loss functions in
the presence of noisy labels and class imbalance, and outperform the
state-of-the-art methods. The code is available at
this https URL.

    

### [[2108.00207] The Separation Capacity of Random Neural Networks](http://arxiv.org/abs/2108.00207)


  Neural networks with random weights appear in a variety of machine learning
applications, most prominently as the initialization of many deep learning
algorithms and as a computationally cheap alternative to fully learned neural
networks. In the present article we enhance the theoretical understanding of
random neural nets by addressing the following data separation problem: under
what conditions can a random neural network make two classes $\mathcal{X}^-,
\mathcal{X}^+ \subset \mathbb{R}^d$ (with positive distance) linearly
separable? We show that a sufficiently large two-layer ReLU-network with
standard Gaussian weights and uniformly distributed biases can solve this
problem with high probability. Crucially, the number of required neurons is
explicitly linked to geometric properties of the underlying sets
$\mathcal{X}^-, \mathcal{X}^+$ and their mutual arrangement. This
instance-specific viewpoint allows us to overcome the usual curse of
dimensionality (exponential width of the layers) in non-pathological situations
where the data carries low-complexity structure. We quantify the relevant
structure of the data in terms of a novel notion of mutual complexity (based on
a localized version of Gaussian mean width), which leads to sound and
informative separation guarantees. We connect our result with related lines of
work on approximation, memorization, and generalization.

    

### [[2108.00214] A Plant Root System Algorithm Based on Swarm Intelligence for One-dimensional Biomedical Signal Feature Engineering](http://arxiv.org/abs/2108.00214)


  To date, very few biomedical signals have transitioned from research
applications to clinical applications. This is largely due to the lack of trust
in the diagnostic ability of non-stationary signals. To reach the level of
clinical diagnostic application, classification using high-quality signal
features is necessary. While there has been considerable progress in machine
learning in recent years, especially deep learning, progress has been quite
limited in the field of feature engineering. This study proposes a feature
extraction algorithm based on group intelligence which we call a Plant Root
System (PRS) algorithm. Importantly, the correlation between features produced
by this PRS algorithm and traditional features is low, and the accuracy of
several widely-used classifiers was found to be substantially improved with the
addition of PRS features. It is expected that more biomedical signals can be
applied to clinical diagnosis using the proposed algorithm.

    

### [[2108.00215] Freezing Sub-Models During Incremental Process Discovery: Extended Version](http://arxiv.org/abs/2108.00215)


  Process discovery aims to learn a process model from observed process
behavior. From a user's perspective, most discovery algorithms work like a
black box. Besides parameter tuning, there is no interaction between the user
and the algorithm. Interactive process discovery allows the user to exploit
domain knowledge and to guide the discovery process. Previously, an incremental
discovery approach has been introduced where a model, considered to be under
construction, gets incrementally extended by user-selected process behavior.
This paper introduces a novel approach that additionally allows the user to
freeze model parts within the model under construction. Frozen sub-models are
not altered by the incremental approach when new behavior is added to the
model. The user can thus steer the discovery algorithm. Our experiments show
that freezing sub-models can lead to higher quality models.

    

### [[2108.00219] Grain: Improving Data Efficiency of Graph Neural Networks via Diversified Influence Maximization](http://arxiv.org/abs/2108.00219)


  Data selection methods, such as active learning and core-set selection, are
useful tools for improving the data efficiency of deep learning models on
large-scale datasets. However, recent deep learning models have moved forward
from independent and identically distributed data to graph-structured data,
such as social networks, e-commerce user-item graphs, and knowledge graphs.
This evolution has led to the emergence of Graph Neural Networks (GNNs) that go
beyond the models existing data selection methods are designed for. Therefore,
we present Grain, an efficient framework that opens up a new perspective
through connecting data selection in GNNs with social influence maximization.
By exploiting the common patterns of GNNs, Grain introduces a novel feature
propagation concept, a diversified influence maximization objective with novel
influence and diversity functions, and a greedy algorithm with an approximation
guarantee into a unified framework. Empirical studies on public datasets
demonstrate that Grain significantly improves both the performance and
efficiency of data selection (including active learning and core-set selection)
for GNNs. To the best of our knowledge, this is the first attempt to bridge two
largely parallel threads of research, data selection, and social influence
maximization, in the setting of GNNs, paving new ways for improving data
efficiency.

    

### [[2108.00230] Pure Exploration and Regret Minimization in Matching Bandits](http://arxiv.org/abs/2108.00230)


  Finding an optimal matching in a weighted graph is a standard combinatorial
problem. We consider its semi-bandit version where either a pair or a full
matching is sampled sequentially. We prove that it is possible to leverage a
rank-1 assumption on the adjacency matrix to reduce the sample complexity and
the regret of off-the-shelf algorithms up to reaching a linear dependency in
the number of vertices (up to poly log terms).

    

### [[2108.00236] Debiasing Samples from Online Learning Using Bootstrap](http://arxiv.org/abs/2108.00236)


  It has been recently shown in the literature that the sample averages from
online learning experiments are biased when used to estimate the mean reward.
To correct the bias, off-policy evaluation methods, including importance
sampling and doubly robust estimators, typically calculate the propensity
score, which is unavailable in this setting due to unknown reward distribution
and the adaptive policy. This paper provides a procedure to debias the samples
using bootstrap, which doesn't require the knowledge of the reward distribution
at all. Numerical experiments demonstrate the effective bias reduction for
samples generated by popular multi-armed bandit algorithms such as
Explore-Then-Commit (ETC), UCB, Thompson sampling and $\epsilon$-greedy. We
also analyze and provide theoretical justifications for the procedure under the
ETC algorithm, including the asymptotic convergence of the bias decay rate in
the real and bootstrap worlds.

    

### [[2108.00241] Diverse Linguistic Features for Assessing Reading Difficulty of Educational Filipino Texts](http://arxiv.org/abs/2108.00241)


  In order to ensure quality and effective learning, fluency, and
comprehension, the proper identification of the difficulty levels of reading
materials should be observed. In this paper, we describe the development of
automatic machine learning-based readability assessment models for educational
Filipino texts using the most diverse set of linguistic features for the
language. Results show that using a Random Forest model obtained a high
performance of 62.7% in terms of accuracy, and 66.1% when using the optimal
combination of feature sets consisting of traditional and syllable
pattern-based predictors.

    

### [[2108.00250] Bayesian analysis of the prevalence bias: learning and predicting from imbalanced data](http://arxiv.org/abs/2108.00250)


  Datasets are rarely a realistic approximation of the target population. Say,
prevalence is misrepresented, image quality is above clinical standards, etc.
This mismatch is known as sampling bias. Sampling biases are a major hindrance
for machine learning models. They cause significant gaps between model
performance in the lab and in the real world. Our work is a solution to
prevalence bias. Prevalence bias is the discrepancy between the prevalence of a
pathology and its sampling rate in the training dataset, introduced upon
collecting data or due to the practioner rebalancing the training batches. This
paper lays the theoretical and computational framework for training models, and
for prediction, in the presence of prevalence bias. Concretely a bias-corrected
loss function, as well as bias-corrected predictive rules, are derived under
the principles of Bayesian risk minimization. The loss exhibits a direct
connection to the information gain. It offers a principled alternative to
heuristic training losses and complements test-time procedures based on
selecting an operating point from summary curves. It integrates seamlessly in
the current paradigm of (deep) learning using stochastic backpropagation and
naturally with Bayesian models.

    

### [[2108.00257] BoA-PTA, A Bayesian Optimization Accelerated Error-Free SPICE Solver](http://arxiv.org/abs/2108.00257)


  One of the greatest challenges in IC design is the repeated executions of
computationally expensive SPICE simulations, particularly when highly complex
chip testing/verification is involved. Recently, pseudo transient analysis
(PTA) has shown to be one of the most promising continuation SPICE solver.
However, the PTA efficiency is highly influenced by the inserted
pseudo-parameters. In this work, we proposed BoA-PTA, a Bayesian optimization
accelerated PTA that can substantially accelerate simulations and improve
convergence performance without introducing extra errors. Furthermore, our
method does not require any pre-computation data or offline training. The
acceleration framework can either be implemented to speed up ongoing repeated
simulations immediately or to improve new simulations of completely different
circuits. BoA-PTA is equipped with cutting-edge machine learning techniques,
e.g., deep learning, Gaussian process, Bayesian optimization, non-stationary
monotonic transformation, and variational inference via parameterization. We
assess BoA-PTA in 43 benchmark circuits against other SOTA SPICE solvers and
demonstrate an average 2.3x (maximum 3.5x) speed-up over the original CEPTA.

    

### [[2108.00259] Provably Efficient Lottery Ticket Discovery](http://arxiv.org/abs/2108.00259)


  The lottery ticket hypothesis (LTH) claims that randomly-initialized, dense
neural networks contain (sparse) subnetworks that, when trained an equal amount
in isolation, can match the dense network's performance. Although LTH is useful
for discovering efficient network architectures, its three-step process --
pre-training, pruning, and re-training -- is computationally expensive, as the
dense model must be fully pre-trained. Luckily, "early-bird" tickets can be
discovered within neural networks that are minimally pre-trained, allowing for
the creation of efficient, LTH-inspired training procedures. Yet, no
theoretical foundation of this phenomenon exists. We derive an analytical bound
for the number of pre-training iterations that must be performed for a winning
ticket to be discovered, thus providing a theoretical understanding of when and
why such early-bird tickets exist. By adopting a greedy forward selection
pruning strategy, we directly connect the pruned network's performance to the
loss of the dense network from which it was derived, revealing a threshold in
the number of pre-training iterations beyond which high-performing subnetworks
are guaranteed to exist. We demonstrate the validity of our theoretical results
across a variety of architectures and datasets, including multi-layer
perceptrons (MLPs) trained on MNIST and several deep convolutional neural
network (CNN) architectures trained on CIFAR10 and ImageNet.

    

### [[2108.00261] ECLARE: Extreme Classification with Label Graph Correlations](http://arxiv.org/abs/2108.00261)


  Deep extreme classification (XC) seeks to train deep architectures that can
tag a data point with its most relevant subset of labels from an extremely
large label set. The core utility of XC comes from predicting labels that are
rarely seen during training. Such rare labels hold the key to personalized
recommendations that can delight and surprise a user. However, the large number
of rare labels and small amount of training data per rare label offer
significant statistical and computational challenges. State-of-the-art deep XC
methods attempt to remedy this by incorporating textual descriptions of labels
but do not adequately address the problem. This paper presents ECLARE, a
scalable deep learning architecture that incorporates not only label text, but
also label correlations, to offer accurate real-time predictions within a few
milliseconds. Core contributions of ECLARE include a frugal architecture and
scalable techniques to train deep models along with label correlation graphs at
the scale of millions of labels. In particular, ECLARE offers predictions that
are 2 to 14% more accurate on both publicly available benchmark datasets as
well as proprietary datasets for a related products recommendation task sourced
from the Bing search engine. Code for ECLARE is available at
this https URL.

    

### [[2108.00262] Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning](http://arxiv.org/abs/2108.00262)


  We present a generative adversarial network to synthesize 3D pose sequences
of co-speech upper-body gestures with appropriate affective expressions. Our
network consists of two components: a generator to synthesize gestures from a
joint embedding space of features encoded from the input speech and the seed
poses, and a discriminator to distinguish between the synthesized pose
sequences and real 3D pose sequences. We leverage the Mel-frequency cepstral
coefficients and the text transcript computed from the input speech in separate
encoders in our generator to learn the desired sentiments and the associated
affective cues. We design an affective encoder using multi-scale
spatial-temporal graph convolutions to transform 3D pose sequences into latent,
pose-based affective features. We use our affective encoder in both our
generator, where it learns affective features from the seed poses to guide the
gesture synthesis, and our discriminator, where it enforces the synthesized
gestures to contain the appropriate affective expressions. We perform extensive
evaluations on two benchmark datasets for gesture synthesis from the speech,
the TED Gesture Dataset and the GENEA Challenge 2020 Dataset. Compared to the
best baselines, we improve the mean absolute joint error by 10--33%, the mean
acceleration difference by 8--58%, and the Fréchet Gesture Distance by
21--34%. We also conduct a user study and observe that compared to the best
current baselines, around 15.28% of participants indicated our synthesized
gestures appear more plausible, and around 16.32% of participants felt the
gestures had more appropriate affective expressions aligned with the speech.

    

### [[2108.00268] RLTutor: Reinforcement Learning Based Adaptive Tutoring System by Modeling Virtual Student with Fewer Interactions](http://arxiv.org/abs/2108.00268)


  A major challenge in the field of education is providing review schedules
that present learned items at appropriate intervals to each student so that
memory is retained over time. In recent years, attempts have been made to
formulate item reviews as sequential decision-making problems to realize
adaptive instruction based on the knowledge state of students. It has been
reported previously that reinforcement learning can help realize mathematical
models of students learning strategies to maintain a high memory rate. However,
optimization using reinforcement learning requires a large number of
interactions, and thus it cannot be applied directly to actual students. In
this study, we propose a framework for optimizing teaching strategies by
constructing a virtual model of the student while minimizing the interaction
with the actual teaching target. In addition, we conducted an experiment
considering actual instructions using the mathematical model and confirmed that
the model performance is comparable to that of conventional teaching methods.
Our framework can directly substitute mathematical models used in experiments
with human students, and our results can serve as a buffer between theoretical
instructional optimization and practical applications in e-learning systems.

    

### [[2108.00290] A Hybrid Ensemble Feature Selection Design for Candidate Biomarkers Discovery from Transcriptome Profiles](http://arxiv.org/abs/2108.00290)


  The discovery of disease biomarkers from gene expression data has been
greatly advanced by feature selection (FS) methods, especially using ensemble
FS (EFS) strategies with perturbation at the data level (i.e., homogeneous,
Hom-EFS) or method level (i.e., heterogeneous, Het-EFS). Here we proposed a
Hybrid EFS (Hyb-EFS) design that explores both types of perturbation to improve
the stability and the predictive power of candidate biomarkers. With this,
Hyb-EFS aims to disrupt associations of good performance with a single dataset,
single algorithm, or a specific combination of both, which is particularly
interesting for better reproducibility of genomic biomarkers. We investigated
the adequacy of our approach for microarray data related to four types of
cancer, carrying out an extensive comparison with other ensemble and single FS
approaches. Five FS methods were used in our experiments: Wx, Symmetrical
Uncertainty (SU), Gain Ratio (GR), Characteristic Direction (GeoDE), and
ReliefF. We observed that the Hyb-EFS and Het-EFS approaches attenuated the
large performance variation observed for most single FS and Hom-EFS across
distinct datasets. Also, the Hyb-EFS improved upon the stability of the Het-EFS
within our domain. Comparing the Hyb-EFS and Het-EFS composed of the
top-performing selectors (Wx, GR, and SU), our hybrid approach surpassed the
equivalent heterogeneous design and the best Hom-EFS (Hom-Wx). Interestingly,
the rankings produced by our Hyb-EFS reached greater biological plausibility,
with a notably high enrichment for cancer-related genes and pathways. Thus, our
experiments suggest the potential of the proposed Hybrid EFS design in
discovering candidate biomarkers from microarray data. Finally, we provide an
open-source framework to support similar analyses in other domains, both as a
user-friendly application and a plain Python package.

    

### [[2108.00293] Inverse Reinforcement Learning for Strategy Identification](http://arxiv.org/abs/2108.00293)


  In adversarial environments, one side could gain an advantage by identifying
the opponent's strategy. For example, in combat games, if an opponents strategy
is identified as overly aggressive, one could lay a trap that exploits the
opponent's aggressive nature. However, an opponent's strategy is not always
apparent and may need to be estimated from observations of their actions. This
paper proposes to use inverse reinforcement learning (IRL) to identify
strategies in adversarial environments. Specifically, the contributions of this
work are 1) the demonstration of this concept on gaming combat data generated
from three pre-defined strategies and 2) the framework for using IRL to achieve
strategy identification. The numerical experiments demonstrate that the
recovered rewards can be identified using a variety of techniques. In this
paper, the recovered reward are visually displayed, clustered using
unsupervised learning, and classified using a supervised learner.

    

### [[2108.00295] Fair Representation Learning using Interpolation Enabled Disentanglement](http://arxiv.org/abs/2108.00295)


  With the growing interest in the machine learning community to solve
real-world problems, it has become crucial to uncover the hidden reasoning
behind their decisions by focusing on the fairness and auditing the predictions
made by these black-box models. In this paper, we propose a novel method to
address two key issues: (a) Can we simultaneously learn fair disentangled
representations while ensuring the utility of the learned representation for
downstream tasks, and (b)Can we provide theoretical insights into when the
proposed approach will be both fair and accurate. To address the former, we
propose the method FRIED, Fair Representation learning using Interpolation
Enabled Disentanglement. In our architecture, by imposing a critic-based
adversarial framework, we enforce the interpolated points in the latent space
to be more realistic. This helps in capturing the data manifold effectively and
enhances the utility of the learned representation for downstream prediction
tasks. We address the latter question by developing a theory on
fairness-accuracy trade-offs using classifier-based conditional mutual
information estimation. We demonstrate the effectiveness of FRIED on datasets
of different modalities - tabular, text, and image datasets. We observe that
the representations learned by FRIED are overall fairer in comparison to
existing baselines and also accurate for downstream prediction tasks.
Additionally, we evaluate FRIED on a real-world healthcare claims dataset where
we conduct an expert aided model auditing study providing useful insights into
opioid ad-diction patterns.

    

### [[2108.00298] Multivariate Time Series Imputation by Graph Neural Networks](http://arxiv.org/abs/2108.00298)


  Dealing with missing values and incomplete time series is a labor-intensive
and time-consuming inevitable task when handling data coming from real-world
applications. Effective spatio-temporal representations would allow imputation
methods to reconstruct missing temporal data by exploiting information coming
from sensors at different locations. However, standard methods fall short in
capturing the nonlinear time and space dependencies existing within networks of
interconnected sensors and do not take full advantage of the available - and
often strong - relational information. Notably, most of state-of-the-art
imputation methods based on deep learning do not explicitly model relational
aspects and, in any case, do not exploit processing frameworks able to
adequately represent structured spatio-temporal data. Conversely, graph neural
networks have recently surged in popularity as both expressive and scalable
tools for processing sequential data with relational inductive biases. In this
work, we present the first assessment of graph neural networks in the context
of multivariate time series imputation. In particular, we introduce a novel
graph neural network architecture, named GRIL, which aims at reconstructing
missing data in the different channels of a multivariate time series by
learning spatial-temporal representations through message passing. Preliminary
empirical results show that our model outperforms state-of-the-art methods in
the imputation task on relevant benchmarks with mean absolute error
improvements often higher than 20%.

    

### [[2108.00302] Conditional Bures Metric for Domain Adaptation](http://arxiv.org/abs/2108.00302)


  As a vital problem in classification-oriented transfer, unsupervised domain
adaptation (UDA) has attracted widespread attention in recent years. Previous
UDA methods assume the marginal distributions of different domains are shifted
while ignoring the discriminant information in the label distributions. This
leads to classification performance degeneration in real applications. In this
work, we focus on the conditional distribution shift problem which is of great
concern to current conditional invariant models. We aim to seek a kernel
covariance embedding for conditional distribution which remains yet unexplored.
Theoretically, we propose the Conditional Kernel Bures (CKB) metric for
characterizing conditional distribution discrepancy, and derive an empirical
estimation for the CKB metric without introducing the implicit kernel feature
map. It provides an interpretable approach to understand the knowledge transfer
mechanism. The established consistency theory of the empirical estimation
provides a theoretical guarantee for convergence. A conditional distribution
matching network is proposed to learn the conditional invariant and
discriminative features for UDA. Extensive experiments and analysis show the
superiority of our proposed model.

    

### [[2108.00316] Chest ImaGenome Dataset for Clinical Reasoning](http://arxiv.org/abs/2108.00316)


  Despite the progress in automatic detection of radiologic findings from chest
X-ray (CXR) images in recent years, a quantitative evaluation of the
explainability of these models is hampered by the lack of locally labeled
datasets for different findings. With the exception of a few expert-labeled
small-scale datasets for specific findings, such as pneumonia and pneumothorax,
most of the CXR deep learning models to date are trained on global "weak"
labels extracted from text reports, or trained via a joint image and
unstructured text learning strategy. Inspired by the Visual Genome effort in
the computer vision community, we constructed the first Chest ImaGenome dataset
with a scene graph data structure to describe $242,072$ images. Local
annotations are automatically produced using a joint rule-based natural
language processing (NLP) and atlas-based bounding box detection pipeline.
Through a radiologist constructed CXR ontology, the annotations for each CXR
are connected as an anatomy-centered scene graph, useful for image-level
reasoning and multimodal fusion applications. Overall, we provide: i) $1,256$
combinations of relation annotations between $29$ CXR anatomical locations
(objects with bounding box coordinates) and their attributes, structured as a
scene graph per image, ii) over $670,000$ localized comparison relations (for
improved, worsened, or no change) between the anatomical locations across
sequential exams, as well as ii) a manually annotated gold standard scene graph
dataset from $500$ unique patients.

    

### [[2108.00318] Statistical learning method for predicting density-matrix based electron dynamics](http://arxiv.org/abs/2108.00318)


  We develop a statistical method to learn a molecular Hamiltonian matrix from
a time-series of electron density matrices. We extend our previous method to
larger molecular systems by incorporating physical properties to reduce
dimensionality, while also exploiting regularization techniques like ridge
regression for addressing multicollinearity. With the learned Hamiltonian we
can solve the Time-Dependent Hartree-Fock (TDHF) equation to propagate the
electron density in time, and predict its dynamics for field-free and field-on
scenarios. We observe close quantitative agreement between the predicted
dynamics and ground truth for both field-off trajectories similar to the
training data, and field-on trajectories outside of the training data.

    

### [[2108.00330] Bilevel Optimization for Machine Learning: Algorithm Design and Convergence Analysis](http://arxiv.org/abs/2108.00330)


  Bilevel optimization has become a powerful framework in various machine
learning applications including meta-learning, hyperparameter optimization, and
network architecture search. There are generally two classes of bilevel
optimization formulations for machine learning: 1) problem-based bilevel
optimization, whose inner-level problem is formulated as finding a minimizer of
a given loss function; and 2) algorithm-based bilevel optimization, whose
inner-level solution is an output of a fixed algorithm. For the first class,
two popular types of gradient-based algorithms have been proposed for
hypergradient estimation via approximate implicit differentiation (AID) and
iterative differentiation (ITD). Algorithms for the second class include the
popular model-agnostic meta-learning (MAML) and almost no inner loop (ANIL).
However, the convergence rate and fundamental limitations of bilevel
optimization algorithms have not been well explored.
This thesis provides a comprehensive convergence rate analysis for bilevel
algorithms in the aforementioned two classes. We further propose principled
algorithm designs for bilevel optimization with higher efficiency and
scalability. For the problem-based formulation, we provide a convergence rate
analysis for AID- and ITD-based bilevel algorithms. We then develop
acceleration bilevel algorithms, for which we provide shaper convergence
analysis with relaxed assumptions. We also provide the first lower bounds for
bilevel optimization, and establish the optimality by providing matching upper
bounds under certain conditions. We finally propose new stochastic bilevel
optimization algorithms with lower complexity and higher efficiency in
practice. For the algorithm-based formulation, we develop a theoretical
convergence for general multi-step MAML and ANIL, and characterize the impact
of parameter selections and loss geometries on the their complexities.

    

### [[2108.00331] Faster Rates of Differentially Private Stochastic Convex Optimization](http://arxiv.org/abs/2108.00331)


  In this paper, we revisit the problem of Differentially Private Stochastic
Convex Optimization (DP-SCO) and provide excess population risks for some
special classes of functions that are faster than the previous results of
general convex and strongly convex functions. In the first part of the paper,
we study the case where the population risk function satisfies the Tysbakov
Noise Condition (TNC) with some parameter $\theta>1$. Specifically, we first
show that under some mild assumptions on the loss functions, there is an
algorithm whose output could achieve an upper bound of
$\tilde{O}((\frac{1}{\sqrt{n}}+\frac{\sqrt{d\log
\frac{1}{\delta}}}{n\epsilon})^\frac{\theta}{\theta-1})$ for $(\epsilon,
\delta)$-DP when $\theta\geq 2$, here $n$ is the sample size and $d$ is the
dimension of the space. Then we address the inefficiency issue, improve the
upper bounds by $\text{Poly}(\log n)$ factors and extend to the case where
$\theta\geq \bar{\theta}>1$ for some known $\bar{\theta}$. Next we show that
the excess population risk of population functions satisfying TNC with
parameter $\theta>1$ is always lower bounded by
$\Omega((\frac{d}{n\epsilon})^\frac{\theta}{\theta-1}) $ and
$\Omega((\frac{\sqrt{d\log
\frac{1}{\delta}}}{n\epsilon})^\frac{\theta}{\theta-1})$ for $\epsilon$-DP and
$(\epsilon, \delta)$-DP, respectively. In the second part, we focus on a
special case where the population risk function is strongly convex. Unlike the
previous studies, here we assume the loss function is {\em non-negative} and
{\em the optimal value of population risk is sufficiently small}. With these
additional assumptions, we propose a new method whose output could achieve an
upper bound of
$O(\frac{d\log\frac{1}{\delta}}{n^2\epsilon^2}+\frac{1}{n^{\tau}})$ for any
$\tau\geq 1$ in $(\epsilon,\delta)$-DP model if the sample size $n$ is
sufficiently large.

    

### [[2108.00352] BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning](http://arxiv.org/abs/2108.00352)


  Self-supervised learning in computer vision aims to pre-train an image
encoder using a large amount of unlabeled images or (image, text) pairs. The
pre-trained image encoder can then be used as a feature extractor to build
downstream classifiers for many downstream tasks with a small amount of or no
labeled training data. In this work, we propose BadEncoder, the first backdoor
attack to self-supervised learning. In particular, our BadEncoder injects
backdoors into a pre-trained image encoder such that the downstream classifiers
built based on the backdoored image encoder for different downstream tasks
simultaneously inherit the backdoor behavior. We formulate our BadEncoder as an
optimization problem and we propose a gradient descent based method to solve
it, which produces a backdoored image encoder from a clean one. Our extensive
empirical evaluation results on multiple datasets show that our BadEncoder
achieves high attack success rates while preserving the accuracy of the
downstream classifiers. We also show the effectiveness of BadEncoder using two
publicly available, real-world image encoders, i.e., Google's image encoder
pre-trained on ImageNet and OpenAI's Contrastive Language-Image Pre-training
(CLIP) image encoder pre-trained on 400 million (image, text) pairs collected
from the Internet. Moreover, we consider defenses including Neural Cleanse and
MNTD (empirical defenses) as well as PatchGuard (a provable defense). Our
results show that these defenses are insufficient to defend against BadEncoder,
highlighting the needs for new defenses against our BadEncoder. Our code is
publicly available at: this https URL.

    

### [[2108.00354] UAV Trajectory Planning in Wireless Sensor Networks for Energy Consumption Minimization by Deep Reinforcement Learning](http://arxiv.org/abs/2108.00354)


  Unmanned aerial vehicles (UAVs) have emerged as a promising candidate
solution for data collection of large-scale wireless sensor networks (WSNs). In
this paper, we investigate a UAV-aided WSN, where cluster heads (CHs) receive
data from their member nodes, and a UAV is dispatched to collect data from CHs
along the planned trajectory. We aim to minimize the total energy consumption
of the UAV-WSN system in a complete round of data collection. Toward this end,
we formulate the energy consumption minimization problem as a constrained
combinatorial optimization problem by jointly selecting CHs from nodes within
clusters and planning the UAV's visiting order to the selected CHs. The
formulated energy consumption minimization problem is NP-hard, and hence, hard
to solve optimally. In order to tackle this challenge, we propose a novel deep
reinforcement learning (DRL) technique, pointer network-A* (Ptr-A*), which can
efficiently learn from experiences the UAV trajectory policy for minimizing the
energy consumption. The UAV's start point and the WSN with a set of
pre-determined clusters are fed into the Ptr-A*, and the Ptr-A* outputs a group
of CHs and the visiting order to these CHs, i.e., the UAV's trajectory. The
parameters of the Ptr-A* are trained on small-scale clusters problem instances
for faster training by using the actor-critic algorithm in an unsupervised
manner. At inference, three search strategies are also proposed to improve the
quality of solutions. Simulation results show that the trained models based on
20-clusters and 40-clusters have a good generalization ability to solve the
UAV's trajectory planning problem in WSNs with different numbers of clusters,
without the need to retrain the models. Furthermore, the results show that our
proposed DRL algorithm outperforms two baseline techniques.

    

### [[2108.00360] IPOF: An Extremely and Excitingly Simple Outlier Detection Booster via Infinite Propagation](http://arxiv.org/abs/2108.00360)


  Outlier detection is one of the most popular and continuously rising topics
in the data mining field due to its crucial academic value and extensive
industrial applications. Among different settings, unsupervised outlier
detection is the most challenging and practical one, which attracts tremendous
efforts from diverse perspectives. In this paper, we consider the score-based
outlier detection category and point out that the performance of current
outlier detection algorithms might be further boosted by score propagation.
Specifically, we propose Infinite Propagation of Outlier Factor (iPOF)
algorithm, an extremely and excitingly simple outlier detection booster via
infinite propagation. By employing score-based outlier detectors for
initialization, iPOF updates each data point's outlier score by averaging the
outlier factors of its nearest common neighbors. Extensive experimental results
on numerous datasets in various domains demonstrate the effectiveness and
efficiency of iPOF significantly over several classical and recent
state-of-the-art methods. We also provide the parameter analysis on the number
of neighbors, the unique parameter in iPOF, and different initial outlier
detectors for general validation. It is worthy to note that iPOF brings in
positive improvements ranging from 2% to 46% on the average level, and in some
cases, iPOF boosts the performance over 3000% over the original outlier
detection algorithm.

    

### [[2108.00365] A Decentralized Federated Learning Framework via Committee Mechanism with Convergence Guarantee](http://arxiv.org/abs/2108.00365)


  Federated learning allows multiple participants to collaboratively train an
efficient model without exposing data privacy. However, this distributed
machine learning training method is prone to attacks from Byzantine clients,
which interfere with the training of the global model by modifying the model or
uploading the false gradient. In this paper, we propose a novel serverless
federated learning framework Committee Mechanism based Federated Learning
(CMFL), which can ensure the robustness of the algorithm with convergence
guarantee. In CMFL, a committee system is set up to screen the uploaded local
gradients. The committee system selects the local gradients rated by the
elected members for the aggregation procedure through the selection strategy,
and replaces the committee member through the election strategy. Based on the
different considerations of model performance and defense, two opposite
selection strategies are designed for the sake of both accuracy and robustness.
Extensive experiments illustrate that CMFL achieves faster convergence and
better accuracy than the typical Federated Learning, in the meanwhile obtaining
better robustness than the traditional Byzantine-tolerant algorithms, in the
manner of a decentralized approach. In addition, we theoretically analyze and
prove the convergence of CMFL under different election and selection
strategies, which coincides with the experimental results.

    

### [[2108.00367] CNN based Channel Estimation using NOMA for mmWave Massive MIMO System](http://arxiv.org/abs/2108.00367)


  Non-Orthogonal Multiple Access (NOMA) schemes are being actively explored to
address some of the major challenges in 5th Generation (5G) Wireless
communications. Channel estimation is exceptionally challenging in scenarios
where NOMA schemes are integrated with millimeter wave (mmWave) massive
multiple-input multiple-output (MIMO) systems. An accurate estimation of the
channel is essential in exploiting the benefits of the pairing of the duo-NOMA
and mmWave. This paper proposes a convolutional neural network (CNN) based
approach to estimate the channel for NOMA based millimeter wave (mmWave)
massive multiple-input multiple-output (MIMO) systems built on a hybrid
architecture. Initially, users are grouped into different clusters based on
their channel gains and beamforming technique is performed to maximize the
signal in the direction of desired cluster. A coarse estimation of the channel
is first made from the received signal and this estimate is given as the input
to CNN to fine estimate the channel coefficients. Numerical illustrations show
that the proposed method outperforms least square (LS) estimate, minimum mean
square error (MMSE) estimate and are close to the Cramer-Rao Bound (CRB).

    

### [[2108.00368] DECAF: Deep Extreme Classification with Label Features](http://arxiv.org/abs/2108.00368)


  Extreme multi-label classification (XML) involves tagging a data point with
its most relevant subset of labels from an extremely large label set, with
several applications such as product-to-product recommendation with millions of
products. Although leading XML algorithms scale to millions of labels, they
largely ignore label meta-data such as textual descriptions of the labels. On
the other hand, classical techniques that can utilize label metadata via
representation learning using deep networks struggle in extreme settings. This
paper develops the DECAF algorithm that addresses these challenges by learning
models enriched by label metadata that jointly learn model parameters and
feature representations using deep networks and offer accurate classification
at the scale of millions of labels. DECAF makes specific contributions to model
architecture design, initialization, and training, enabling it to offer up to
2-6% more accurate prediction than leading extreme classifiers on publicly
available benchmark product-to-product recommendation datasets, such as
LF-AmazonTitles-1.3M. At the same time, DECAF was found to be up to 22x faster
at inference than leading deep extreme classifiers, which makes it suitable for
real-time applications that require predictions within a few milliseconds. The
code for DECAF is available at the following URL
this https URL.

    

### [[2108.00373] SPEAR : Semi-supervised Data Programming in Python](http://arxiv.org/abs/2108.00373)


  We present SPEAR, an open-source python library for data programming with
semi supervision. The package implements several recent data programming
approaches including facility to programmatically label and build training
data. SPEAR facilitates weak supervision in the form of heuristics (or rules)
and association of noisy labels to the training dataset. These noisy labels are
aggregated to assign labels to the unlabeled data for downstream tasks. We have
implemented several label aggregation approaches that aggregate the noisy
labels and then train using the noisily labeled set in a cascaded manner. Our
implementation also includes other approaches that jointly aggregate and train
the model. Thus, in our python package, we integrate several cascade and joint
data-programming approaches while also providing the facility of data
programming by letting the user define labeling functions or rules. The code
and tutorial notebooks are available at
\url{this https URL}.

    

### [[2108.00394] Deep graph matching meets mixed-integer linear programming: Relax at your own risk ?](http://arxiv.org/abs/2108.00394)


  Graph matching is an important problem that has received widespread
attention, especially in the field of computer vision. Recently,
state-of-the-art methods seek to incorporate graph matching with deep learning.
However, there is no research to explain what role the graph matching algorithm
plays in the model. Therefore, we propose an approach integrating a MILP
formulation of the graph matching problem. This formulation is solved to
optimal and it provides inherent baseline. Meanwhile, similar approaches are
derived by releasing the optimal guarantee of the graph matching solver and by
introducing a quality level. This quality level controls the quality of the
solutions provided by the graph matching solver. In addition, several
relaxations of the graph matching problem are put to the test. Our experimental
evaluation gives several theoretical insights and guides the direction of deep
graph matching methods.

    

### [[2108.00401] Threat of Adversarial Attacks on Deep Learning in Computer Vision: Survey II](http://arxiv.org/abs/2108.00401)


  Deep Learning (DL) is the most widely used tool in the contemporary field of
computer vision. Its ability to accurately solve complex problems is employed
in vision research to learn deep neural models for a variety of tasks,
including security critical applications. However, it is now known that DL is
vulnerable to adversarial attacks that can manipulate its predictions by
introducing visually imperceptible perturbations in images and videos. Since
the discovery of this phenomenon in 2013~[1], it has attracted significant
attention of researchers from multiple sub-fields of machine intelligence. In
[2], we reviewed the contributions made by the computer vision community in
adversarial attacks on deep learning (and their defenses) until the advent of
year 2018. Many of those contributions have inspired new directions in this
area, which has matured significantly since witnessing the first generation
methods. Hence, as a legacy sequel of [2], this literature review focuses on
the advances in this area since 2018. To ensure authenticity, we mainly
consider peer-reviewed contributions published in the prestigious sources of
computer vision and machine learning research. Besides a comprehensive
literature review, the article also provides concise definitions of technical
terminologies for non-experts in this domain. Finally, this article discusses
challenges and future outlook of this direction based on the literature
reviewed herein and [2].

    

### [[2108.00404] Masking Neural Networks Using Reachability Graphs to Predict Process Events](http://arxiv.org/abs/2108.00404)


  Decay Replay Mining is a deep learning method that utilizes process model
notations to predict the next event. However, this method does not intertwine
the neural network with the structure of the process model to its full extent.
This paper proposes an approach to further interlock the process model of Decay
Replay Mining with its neural network for next event prediction. The approach
uses a masking layer which is initialized based on the reachability graph of
the process model. Additionally, modifications to the neural network
architecture are proposed to increase the predictive performance. Experimental
results demonstrate the value of the approach and underscore the importance of
discovering precise and generalized process models.

    

### [[2108.00408] CSC-Unet: A Novel Convolutional Sparse Coding Strategy based Neural Network for Semantic Segmentation](http://arxiv.org/abs/2108.00408)


  It is a challenging task to accurately perform semantic segmentation due to
the complexity of real picture scenes. Many semantic segmentation methods based
on traditional deep learning insufficiently captured the semantic and
appearance information of images, which put limit on their generality and
robustness for various application scenes. In this paper, we proposed a novel
strategy that reformulated the popularly-used convolution operation to
multi-layer convolutional sparse coding block to ease the aforementioned
deficiency. This strategy can be possibly used to significantly improve the
segmentation performance of any semantic segmentation model that involves
convolutional operations. To prove the effectiveness of our idea, we chose the
widely-used U-Net model for the demonstration purpose, and we designed CSC-Unet
model series based on U-Net. Through extensive analysis and experiments, we
provided credible evidence showing that the multi-layer convolutional sparse
coding block enables semantic segmentation model to converge faster, can
extract finer semantic and appearance information of images, and improve the
ability to recover spatial detail information. The best CSC-Unet model
significantly outperforms the results of the original U-Net on three public
datasets with different scenarios, i.e., 87.14% vs. 84.71% on DeepCrack
dataset, 68.91% vs. 67.09% on Nuclei dataset, and 53.68% vs. 48.82% on CamVid
dataset, respectively.

    

### [[2108.00413] Data Driven Macroscopic Modeling across Knudsen Numbers for Rarefied Gas Dynamics and Application to Rayleigh Scattering](http://arxiv.org/abs/2108.00413)


  Macroscopic modeling of the gas dynamics across Knudsen numbers from dense
gas region to rarefied gas region remains a great challenge. The reason is
macroscopic models lack accurate constitutive relations valid across different
Knudsen numbers. To address this problem, we proposed a Data-driven, KnUdsen
number Adaptive Linear constitutive relation model named DUAL. The DUAL model
is accurate across a range of Knudsen numbers, from dense to rarefied, through
learning to adapt Knudsen number change from observed data. It is consistent
with the Navier-Stokes equation under the hydrodynamic limit, by utilizing a
constrained neural network. In addition, it naturally satisfies the second law
of thermodynamics and is robust to noisy data. We test the DUAL model on the
calculation of Rayleigh scattering spectra. The DUAL model gives accurate
spectra for various Knudsen numbers and is superior to traditional perturbation
and moment expansion methods.

    

### [[2108.00421] Automated Pest Detection with DNN on the Edge for Precision Agriculture](http://arxiv.org/abs/2108.00421)


  Artificial intelligence has smoothly penetrated several economic activities,
especially monitoring and control applications, including the agriculture
sector. However, research efforts toward low-power sensing devices with fully
functional machine learning (ML) on-board are still fragmented and limited in
smart farming. Biotic stress is one of the primary causes of crop yield
reduction. With the development of deep learning in computer vision technology,
autonomous detection of pest infestation through images has become an important
research direction for timely crop disease diagnosis. This paper presents an
embedded system enhanced with ML functionalities, ensuring continuous detection
of pest infestation inside fruit orchards. The embedded solution is based on a
low-power embedded sensing system along with a Neural Accelerator able to
capture and process images inside common pheromone-based traps. Three different
ML algorithms have been trained and deployed, highlighting the capabilities of
the platform. Moreover, the proposed approach guarantees an extended battery
life thanks to the integration of energy harvesting functionalities. Results
show how it is possible to automate the task of pest infestation for unlimited
time without the farmer's intervention.

    

### [[2108.00439] Transformer-based Map Matching with Model Limited Ground-Truth Data using Transfer-Learning Approach](http://arxiv.org/abs/2108.00439)


  In many trajectory-based applications, it is necessary to map raw GPS
trajectories onto road networks in digital maps, which is commonly referred to
as a map-matching process. While most previous map-matching methods have
focused on using rule-based algorithms to deal with the map-matching problems,
in this paper, we consider the map-matching task from the data perspective,
proposing a deep learning-based map-matching model. We build a
Transformer-based map-matching model with a transfer learning approach. We
generate synthetic trajectory data to pre-train the Transformer model and then
fine-tune the model with a limited number of ground-truth data to minimize the
model development cost and reduce the real-to-virtual gap. Three metrics
(Average Hamming Distance, F-score, and BLEU) at two levels (point and segment
level) are used to evaluate the model performance. The results indicate that
the proposed model outperforms existing models. Furthermore, we use the
attention weights of the Transformer to plot the map-matching process and find
how the model matches the road segments correctly.

    

### [[2108.00462] Explainable Deep Few-shot Anomaly Detection with Deviation Networks](http://arxiv.org/abs/2108.00462)


  Existing anomaly detection paradigms overwhelmingly focus on training
detection models using exclusively normal data or unlabeled data (mostly normal
samples). One notorious issue with these approaches is that they are weak in
discriminating anomalies from normal samples due to the lack of the knowledge
about the anomalies. Here, we study the problem of few-shot anomaly detection,
in which we aim at using a few labeled anomaly examples to train
sample-efficient discriminative detection models. To address this problem, we
introduce a novel weakly-supervised anomaly detection framework to train
detection models without assuming the examples illustrating all possible
classes of anomaly.
Specifically, the proposed approach learns discriminative normality
(regularity) by leveraging the labeled anomalies and a prior probability to
enforce expressive representations of normality and unbounded deviated
representations of abnormality. This is achieved by an end-to-end optimization
of anomaly scores with a neural deviation learning, in which the anomaly scores
of normal samples are imposed to approximate scalar scores drawn from the prior
while that of anomaly examples is enforced to have statistically significant
deviations from these sampled scores in the upper tail. Furthermore, our model
is optimized to learn fine-grained normality and abnormality by top-K
multiple-instance-learning-based feature subspace deviation learning, allowing
more generalized representations. Comprehensive experiments on nine real-world
image anomaly detection benchmarks show that our model is substantially more
sample-efficient and robust, and performs significantly better than
state-of-the-art competing methods in both closed-set and open-set settings.
Our model can also offer explanation capability as a result of its prior-driven
anomaly score learning. Code and datasets are available at:
this https URL.

    

### [[2108.00473] Zeroth-Order Alternating Randomized Gradient Projection Algorithms for General Nonconvex-Concave Minimax Problems](http://arxiv.org/abs/2108.00473)


  In this paper, we study zeroth-order algorithms for nonconvex-concave minimax
problems, which have attracted widely attention in machine learning, signal
processing and many other fields in recent years. We propose a zeroth-order
alternating randomized gradient projection (ZO-AGP) algorithm for smooth
nonconvex-concave minimax problems, and its iteration complexity to obtain an
$\varepsilon$-stationary point is bounded by $\mathcal{O}(\varepsilon^{-4})$,
and the number of function value estimation is bounded by
$\mathcal{O}(d_{x}\varepsilon^{-4}+d_{y}\varepsilon^{-6})$ per iteration.
Moreover, we propose a zeroth-order block alternating randomized proximal
gradient algorithm (ZO-BAPG) for solving block-wise nonsmooth nonconvex-concave
minimax optimization problems, and the iteration complexity to obtain an
$\varepsilon$-stationary point is bounded by $\mathcal{O}(\varepsilon^{-4})$
and the number of function value estimation per iteration is bounded by
$\mathcal{O}(K d_{x}\varepsilon^{-4}+d_{y}\varepsilon^{-6})$. To the best of
our knowledge, this is the first time that zeroth-order algorithms with
iteration complexity gurantee are developed for solving both general smooth and
block-wise nonsmooth nonconvex-concave minimax problems. Numerical results on
data poisoning attack problem validate the efficiency of the proposed
algorithms.

    

### [[2108.00480] Realised Volatility Forecasting: Machine Learning via Financial Word Embedding](http://arxiv.org/abs/2108.00480)


  We develop FinText, a novel, state-of-the-art, financial word embedding from
Dow Jones Newswires Text News Feed Database. Incorporating this word embedding
in a machine learning model produces a substantial increase in volatility
forecasting performance on days with volatility jumps for 23 NASDAQ stocks from
27 July 2007 to 18 November 2016. A simple ensemble model, combining our word
embedding and another machine learning model that uses limit order book data,
provides the best forecasting performance for both normal and jump volatility
days. Finally, we use Integrated Gradients and SHAP (SHapley Additive
exPlanations) to make the results more 'explainable' and the model comparisons
more transparent.

    

### [[2108.00490] A survey of Monte Carlo methods for noisy and costly densities with application to reinforcement learning](http://arxiv.org/abs/2108.00490)


  This survey gives an overview of Monte Carlo methodologies using surrogate
models, for dealing with densities which are intractable, costly, and/or noisy.
This type of problem can be found in numerous real-world scenarios, including
stochastic optimization and reinforcement learning, where each evaluation of a
density function may incur some computationally-expensive or even physical
(real-world activity) cost, likely to give different results each time. The
surrogate model does not incur this cost, but there are important trade-offs
and considerations involved in the choice and design of such methodologies. We
classify the different methodologies into three main classes and describe
specific instances of algorithms under a unified notation. A modular scheme
which encompasses the considered methods is also presented. A range of
application scenarios is discussed, with special attention to the
likelihood-free setting and reinforcement learning. Several numerical
comparisons are also provided.

    

### [[2108.00491] Certified Defense via Latent Space Randomized Smoothing with Orthogonal Encoders](http://arxiv.org/abs/2108.00491)


  Randomized Smoothing (RS), being one of few provable defenses, has been
showing great effectiveness and scalability in terms of defending against
$\ell_2$-norm adversarial perturbations. However, the cost of MC sampling
needed in RS for evaluation is high and computationally expensive. To address
this issue, we investigate the possibility of performing randomized smoothing
and establishing the robust certification in the latent space of a network, so
that the overall dimensionality of tensors involved in computation could be
drastically reduced. To this end, we propose Latent Space Randomized Smoothing.
Another important aspect is that we use orthogonal modules, whose Lipschitz
property is known for free by design, to propagate the certified radius
estimated in the latent space back to the input space, providing valid
certifiable regions for the test samples in the input space. Experiments on
CIFAR10 and ImageNet show that our method achieves competitive certified
robustness but with a significant improvement of efficiency during the test
phase.

    

### [[2108.00505] DeepTrack: Lightweight Deep Learning for Vehicle Path Prediction in Highways](http://arxiv.org/abs/2108.00505)


  Vehicle trajectory prediction is an essential task for enabling many
intelligent transportation systems. While there have been some promising
advances in the field, there is a need for new agile algorithms with smaller
model sizes and lower computational requirements. This article presents
DeepTrack, a novel deep learning algorithm customized for real-time vehicle
trajectory prediction in highways. In contrast to previous methods, the vehicle
dynamics are encoded using Agile Temporal Convolutional Networks (ATCNs) to
provide more robust time prediction with less computation. ATCN also uses
depthwise convolution, which reduces the complexity of models compared to
existing approaches in terms of model size and operations. Overall, our
experimental results demonstrate that DeepTrack achieves comparable accuracy to
state-of-the-art trajectory prediction models but with smaller model sizes and
lower computational complexity, making it more suitable for real-world
deployment.

    

### [[2108.00524] You too Brutus! Trapping Hateful Users in Social Media: Challenges, Solutions & Insights](http://arxiv.org/abs/2108.00524)


  Hate speech is regarded as one of the crucial issues plaguing the online
social media. The current literature on hate speech detection leverages
primarily the textual content to find hateful posts and subsequently identify
hateful users. However, this methodology disregards the social connections
between users. In this paper, we run a detailed exploration of the problem
space and investigate an array of models ranging from purely textual to graph
based to finally semi-supervised techniques using Graph Neural Networks (GNN)
that utilize both textual and graph-based features. We run exhaustive
experiments on two datasets -- Gab, which is loosely moderated and Twitter,
which is strictly moderated. Overall the AGNN model achieves 0.791 macro
F1-score on the Gab dataset and 0.780 macro F1-score on the Twitter dataset
using only 5% of the labeled instances, considerably outperforming all the
other models including the fully supervised ones. We perform detailed error
analysis on the best performing text and graph based models and observe that
hateful users have unique network neighborhood signatures and the AGNN model
benefits by paying attention to these signatures. This property, as we observe,
also allows the model to generalize well across platforms in a zero-shot
setting. Lastly, we utilize the best performing GNN model to analyze the
evolution of hateful users and their targets over time in Gab.

    

### [[2108.00527] Gates are not what you need in RNNs](http://arxiv.org/abs/2108.00527)


  Recurrent neural networks have flourished in many areas. Consequently, we can
see new RNN cells being developed continuously, usually by creating or using
gates in a new, original way. But what if we told you that gates in RNNs are
redundant? In this paper, we propose a new recurrent cell called Residual
Recurrent Unit (RRU) which beats traditional cells and does not employ a single
gate. It is based on the residual shortcut connection together with linear
transformations, ReLU, and normalization. To evaluate our cell's effectiveness,
we compare its performance against the widely-used GRU and LSTM cells and the
recently proposed Mogrifier LSTM on several tasks including, polyphonic music
modeling, language modeling, and sentiment analysis. Our experiments show that
RRU outperforms the traditional gated units on most of these tasks. Also, it
has better robustness to parameter selection, allowing immediate application in
new tasks without much tuning. We have implemented the RRU in TensorFlow, and
the code is made available at this https URL .

    

### [[2108.00548] A Reinforcement Learning Approach for Scheduling in mmWave Networks](http://arxiv.org/abs/2108.00548)


  We consider a source that wishes to communicate with a destination at a
desired rate, over a mmWave network where links are subject to blockage and
nodes to failure (e.g., in a hostile military environment). To achieve
resilience to link and node failures, we here explore a state-of-the-art Soft
Actor-Critic (SAC) deep reinforcement learning algorithm, that adapts the
information flow through the network, without using knowledge of the link
capacities or network topology. Numerical evaluations show that our algorithm
can achieve the desired rate even in dynamic environments and it is robust
against blockage.

    

### [[2108.00559] A Machine-Learning-Based Direction-of-Origin Filter for the Identification of Radio Frequency Interference in the Search for Technosignatures](http://arxiv.org/abs/2108.00559)


  Radio frequency interference (RFI) mitigation remains a major challenge in
the search for radio technosignatures. Typical mitigation strategies include a
direction-of-origin (DoO) filter, where a signal is classified as RFI if it is
detected in multiple directions on the sky. These classifications generally
rely on estimates of signal properties, such as frequency and frequency drift
rate. Convolutional neural networks (CNNs) offer a promising complement to
existing filters because they can be trained to analyze dynamic spectra
directly, instead of relying on inferred signal properties. In this work, we
compiled several data sets consisting of labeled pairs of images of dynamic
spectra, and we designed and trained a CNN that can determine whether or not a
signal detected in one scan is also present in another scan. This CNN-based DoO
filter outperforms both a baseline 2D correlation model as well as existing DoO
filters over a range of metrics, with precision and recall values of 99.15% and
97.81%, respectively. We found that the CNN reduces the number of signals
requiring visual inspection after the application of traditional DoO filters by
a factor of 6-16 in nominal situations.

    

### [[2108.00568] FLASH: Fast Neural Architecture Search with Hardware Optimization](http://arxiv.org/abs/2108.00568)


  Neural architecture search (NAS) is a promising technique to design efficient
and high-performance deep neural networks (DNNs). As the performance
requirements of ML applications grow continuously, the hardware accelerators
start playing a central role in DNN design. This trend makes NAS even more
complicated and time-consuming for most real applications. This paper proposes
FLASH, a very fast NAS methodology that co-optimizes the DNN accuracy and
performance on a real hardware platform. As the main theoretical contribution,
we first propose the NN-Degree, an analytical metric to quantify the
topological characteristics of DNNs with skip connections (e.g., DenseNets,
ResNets, Wide-ResNets, and MobileNets). The newly proposed NN-Degree allows us
to do training-free NAS within one second and build an accuracy predictor by
training as few as 25 samples out of a vast search space with more than 63
billion configurations. Second, by performing inference on the target hardware,
we fine-tune and validate our analytical models to estimate the latency, area,
and energy consumption of various DNN architectures while executing standard ML
datasets. Third, we construct a hierarchical algorithm based on simplicial
homology global optimization (SHGO) to optimize the model-architecture
co-design process, while considering the area, latency, and energy consumption
of the target hardware. We demonstrate that, compared to the state-of-the-art
NAS approaches, our proposed hierarchical SHGO-based algorithm enables more
than four orders of magnitude speedup (specifically, the execution time of the
proposed algorithm is about 0.1 seconds). Finally, our experimental evaluations
show that FLASH is easily transferable to different hardware architectures,
thus enabling us to do NAS on a Raspberry Pi-3B processor in less than 3
seconds.

    

### [[2108.00570] Accelerating Markov Random Field Inference with Uncertainty Quantification](http://arxiv.org/abs/2108.00570)


  Statistical machine learning has widespread application in various domains.
These methods include probabilistic algorithms, such as Markov Chain
Monte-Carlo (MCMC), which rely on generating random numbers from probability
distributions. These algorithms are computationally expensive on conventional
processors, yet their statistical properties, namely interpretability and
uncertainty quantification (UQ) compared to deep learning, make them an
attractive alternative approach. Therefore, hardware specialization can be
adopted to address the shortcomings of conventional processors in running these
applications.
In this paper, we propose a high-throughput accelerator for Markov Random
Field (MRF) inference, a powerful model for representing a wide range of
applications, using MCMC with Gibbs sampling. We propose a tiled architecture
which takes advantage of near-memory computing, and memory optimizations
tailored to the semantics of MRF. Additionally, we propose a novel hybrid
on-chip/off-chip memory system and logging scheme to efficiently support UQ.
This memory system design is not specific to MRF models and is applicable to
applications using probabilistic algorithms. In addition, it dramatically
reduces off-chip memory bandwidth requirements.
We implemented an FPGA prototype of our proposed architecture using
high-level synthesis tools and achieved 146MHz frequency for an accelerator
with 32 function units on an Intel Arria 10 FPGA. Compared to prior work on
FPGA, our accelerator achieves 26X speedup. Furthermore, our proposed memory
system and logging scheme to support UQ reduces off-chip bandwidth by 71% for
two applications. ASIC analysis in 15nm shows our design with 2048 function
units running at 3GHz outperforms GPU implementations of motion estimation and
stereo vision on Nvidia RTX2080Ti by 120X-210X, occupying only 7.7% of the
area.

    

### [[2108.00574] Ab-initio experimental violation of Bell inequalities](http://arxiv.org/abs/2108.00574)


  The violation of a Bell inequality is the paradigmatic example of
device-independent quantum information: the nonclassicality of the data is
certified without the knowledge of the functioning of devices. In practice,
however, all Bell experiments rely on the precise understanding of the
underlying physical mechanisms. Given that, it is natural to ask: Can one
witness nonclassical behaviour in a truly black-box scenario? Here we propose
and implement, computationally and experimentally, a solution to this ab-initio
task. It exploits a robust automated optimization approach based on the
Stochastic Nelder-Mead algorithm. Treating preparation and measurement devices
as black-boxes, and relying on the observed statistics only, our adaptive
protocol approaches the optimal Bell inequality violation after a limited
number of iterations for a variety photonic states, measurement responses and
Bell scenarios. In particular, we exploit it for randomness certification from
unknown states and measurements. Our results demonstrate the power of automated
algorithms, opening a new venue for the experimental implementation of
device-independent quantum technologies.

    

### [[2108.00587] Semi-Supervising Learning, Transfer Learning, and Knowledge Distillation with SimCLR](http://arxiv.org/abs/2108.00587)


  Recent breakthroughs in the field of semi-supervised learning have achieved
results that match state-of-the-art traditional supervised learning methods.
Most successful semi-supervised learning approaches in computer vision focus on
leveraging huge amount of unlabeled data, learning the general representation
via data augmentation and transformation, creating pseudo labels, implementing
different loss functions, and eventually transferring this knowledge to more
task-specific smaller models. In this paper, we aim to conduct our analyses on
three different aspects of SimCLR, the current state-of-the-art semi-supervised
learning framework for computer vision. First, we analyze properties of
contrast learning on fine-tuning, as we understand that contrast learning is
what makes this method so successful. Second, we research knowledge
distillation through teacher-forcing paradigm. We observe that when the teacher
and the student share the same base model, knowledge distillation will achieve
better result. Finally, we study how transfer learning works and its
relationship with the number of classes on different data sets. Our results
indicate that transfer learning performs better when number of classes are
smaller.

    

### [[2108.00597] Exact Pareto Optimal Search for Multi-Task Learning: Touring the Pareto Front](http://arxiv.org/abs/2108.00597)


  Multi-Task Learning (MTL) is a well-established paradigm for training deep
neural network models for multiple correlated tasks. Often the task objectives
conflict, requiring trade-offs between them during model building. In such
cases, MTL models can use gradient-based multi-objective optimization (MOO) to
find one or more Pareto optimal solutions. A common requirement in MTL
applications is to find an {\it Exact} Pareto optimal (EPO) solution, which
satisfies user preferences with respect to task-specific objective functions.
Further, to improve model generalization, various constraints on the weights
may need to be enforced during training. Addressing these requirements is
challenging because it requires a search direction that allows descent not only
towards the Pareto front but also towards the input preference, within the
constraints imposed and in a manner that scales to high-dimensional gradients.
We design and theoretically analyze such search directions and develop the
first scalable algorithm, with theoretical guarantees of convergence, to find
an EPO solution, including when box and equality constraints are imposed. Our
unique method combines multiple gradient descent with carefully controlled
ascent to traverse the Pareto front in a principled manner, making it robust to
initialization. This also facilitates systematic exploration of the Pareto
front, that we utilize to approximate the Pareto front for multi-criteria
decision-making. Empirical results show that our algorithm outperforms
competing methods on benchmark MTL datasets and MOO problems.

    

### [[2108.00599] Synthetic Active Distribution System Generation via Unbalanced Graph Generative Adversarial Network](http://arxiv.org/abs/2108.00599)


  Real active distribution networks with associated smart meter (SM) data are
critical for power researchers. However, it is practically difficult for
researchers to obtain such comprehensive datasets from utilities due to privacy
concerns. To bridge this gap, an implicit generative model with Wasserstein GAN
objectives, namely unbalanced graph generative adversarial network (UG-GAN), is
designed to generate synthetic three-phase unbalanced active distribution
system connectivity. The basic idea is to learn the distribution of random
walks both over a real-world system and across each phase of line segments,
capturing the underlying local properties of an individual real-world
distribution network and generating specific synthetic networks accordingly.
Then, to create a comprehensive synthetic test case, a network correction and
extension process is proposed to obtain time-series nodal demands and standard
distribution grid components with realistic parameters, including distributed
energy resources (DERs) and capacity banks. A Midwest distribution system with
1-year SM data has been utilized to validate the performance of our method.
Case studies with several power applications demonstrate that synthetic active
networks generated by the proposed framework can mimic almost all features of
real-world networks while avoiding the disclosure of confidential information.

    

### [[2108.00605] Bucketed PCA Neural Networks with Neurons Mirroring Signals](http://arxiv.org/abs/2108.00605)


  The bucketed PCA neural network (PCA-NN) with transforms is developed here in
an effort to benchmark deep neural networks (DNN's), for problems on supervised
classification. Most classical PCA models apply PCA to the entire training data
set to establish a reductive representation and then employ non-network tools
such as high-order polynomial classifiers. In contrast, the bucketed PCA-NN
applies PCA to individual buckets which are constructed in two consecutive
phases, as well as retains a genuine architecture of a neural network. This
facilitates a fair apple-to-apple comparison to DNN's, esp. to reveal that a
major chunk of accuracy achieved by many impressive DNN's could possibly be
explained by the bucketed PCA-NN (e.g., 96% out of 98% for the MNIST data set
as an example). Compared with most DNN's, the three building blocks of the
bucketed PCA-NN are easier to comprehend conceptually - PCA, transforms, and
bucketing for error correction. Furthermore, unlike the somewhat quasi-random
neurons ubiquitously observed in DNN's, the PCA neurons resemble or mirror the
input signals and are more straightforward to decipher as a result.

    

### [[2108.00625] Adaptive t-Momentum-based Optimization for Unknown Ratio of Outliers in Amateur Data in Imitation Learning](http://arxiv.org/abs/2108.00625)


  Behavioral cloning (BC) bears a high potential for safe and direct transfer
of human skills to robots. However, demonstrations performed by human operators
often contain noise or imperfect behaviors that can affect the efficiency of
the imitator if left unchecked. In order to allow the imitators to effectively
learn from imperfect demonstrations, we propose to employ the robust t-momentum
optimization algorithm. This algorithm builds on the Student's t-distribution
in order to deal with heavy-tailed data and reduce the effect of outlying
observations. We extend the t-momentum algorithm to allow for an adaptive and
automatic robustness and show empirically how the algorithm can be used to
produce robust BC imitators against datasets with unknown heaviness. Indeed,
the imitators trained with the t-momentum-based Adam optimizers displayed
robustness to imperfect demonstrations on two different manipulation tasks with
different robots and revealed the capability to take advantage of the
additional data while reducing the adverse effect of non-optimal behaviors.

    

### [[2108.00640] Few-shot calibration of low-cost air pollution (PM2.5) sensors using meta-learning](http://arxiv.org/abs/2108.00640)


  Low-cost particulate matter sensors are transforming air quality monitoring
because they have lower costs and greater mobility as compared to reference
monitors. Calibration of these low-cost sensors requires training data from
co-deployed reference monitors. Machine Learning based calibration gives better
performance than conventional techniques, but requires a large amount of
training data from the sensor, to be calibrated, co-deployed with a reference
monitor. In this work, we propose novel transfer learning methods for quick
calibration of sensors with minimal co-deployment with reference monitors.
Transfer learning utilizes a large amount of data from other sensors along with
a limited amount of data from the target sensor. Our extensive experimentation
finds the proposed Model-Agnostic- Meta-Learning (MAML) based transfer learning
method to be the most effective over other competitive baselines.

    

### [[2108.00654] Causal Inference in Educational Systems: A Graphical Modeling Approach](http://arxiv.org/abs/2108.00654)


  Educational systems have traditionally been evaluated using cross-sectional
studies, namely, examining a pretest, posttest, and single intervention.
Although this is a popular approach, it does not model valuable information
such as confounding variables, feedback to students, and other real-world
deviations of studies from ideal conditions. Moreover, learning inherently is a
sequential process and should involve a sequence of interventions. In this
paper, we propose various experimental and quasi-experimental designs for
educational systems and quantify them using the graphical model and directed
acyclic graph (DAG) language. We discuss the applications and limitations of
each method in education. Furthermore, we propose to model the education system
as time-varying treatments, confounders, and time-varying
treatments-confounders feedback. We show that if we control for a sufficient
set of confounders and use appropriate inference techniques such as the inverse
probability of treatment weighting (IPTW) or g-formula, we can close the
backdoor paths and derive the unbiased causal estimate of joint interventions
on the outcome. Finally, we compare the g-formula and IPTW performance and
discuss the pros and cons of using each method.

    

### [[2108.00664] Learning who is in the market from time series: market participant discovery through adversarial calibration of multi-agent simulators](http://arxiv.org/abs/2108.00664)


  In electronic trading markets often only the price or volume time series,
that result from interaction of multiple market participants, are directly
observable. In order to test trading strategies before deploying them to
real-time trading, multi-agent market environments calibrated so that the time
series that result from interaction of simulated agents resemble historical are
often used. To ensure adequate testing, one must test trading strategies in a
variety of market scenarios -- which includes both scenarios that represent
ordinary market days as well as stressed markets (most recently observed due to
the beginning of COVID pandemic). In this paper, we address the problem of
multi-agent simulator parameter calibration to allow simulator capture
characteristics of different market regimes. We propose a novel two-step method
to train a discriminator that is able to distinguish between "real" and "fake"
price and volume time series as a part of GAN with self-attention, and then
utilize it within an optimization framework to tune parameters of a simulator
model with known agent archetypes to represent a market scenario. We conclude
with experimental results that demonstrate effectiveness of our method.

    

### [[2108.00669] Towards Making Deep Learning-based Vulnerability Detectors Robust](http://arxiv.org/abs/2108.00669)


  Automatically detecting software vulnerabilities in source code is an
important problem that has attracted much attention. In particular, deep
learning-based vulnerability detectors, or DL-based detectors, are attractive
because they do not need human experts to define features or patterns of
vulnerabilities. However, such detectors' robustness is unclear. In this paper,
we initiate the study in this aspect by demonstrating that DL-based detectors
are not robust against simple code transformations, dubbed attacks in this
paper, as these transformations may be leveraged for malicious purposes. As a
first step towards making DL-based detectors robust against such attacks, we
propose an innovative framework, dubbed ZigZag, which is centered at (i)
decoupling feature learning and classifier learning and (ii) using a
ZigZag-style strategy to iteratively refine them until they converge to robust
features and robust classifiers. Experimental results show that the ZigZag
framework can substantially improve the robustness of DL-based detectors.

    

### [[2108.00700] Piecewise Linear Units Improve Deep Neural Networks](http://arxiv.org/abs/2108.00700)


  The activation function is at the heart of a deep neural networks
nonlinearity; the choice of the function has great impact on the success of
training. Currently, many practitioners prefer the Rectified Linear Unit (ReLU)
due to its simplicity and reliability, despite its few drawbacks. While most
previous functions proposed to supplant ReLU have been hand-designed, recent
work on learning the function during training has shown promising results. In
this paper we propose an adaptive piecewise linear activation function, the
Piecewise Linear Unit (PiLU), which can be learned independently for each
dimension of the neural network. We demonstrate how PiLU is a generalised
rectifier unit and note its similarities with the Adaptive Piecewise Linear
Units, namely adaptive and piecewise linear. Across a distribution of 30
experiments, we show that for the same model architecture, hyperparameters, and
pre-processing, PiLU significantly outperforms ReLU: reducing classification
error by 18.53% on CIFAR-10 and 13.13% on CIFAR-100, for a minor increase in
the number of neurons. Further work should be dedicated to exploring
generalised piecewise linear units, as well as verifying these results across
other challenging domains and larger problems.

    

### [[2108.00701] Information Stealing in Federated Learning Systems Based on Generative Adversarial Networks](http://arxiv.org/abs/2108.00701)


  An attack on deep learning systems where intelligent machines collaborate to
solve problems could cause a node in the network to make a mistake on a
critical judgment. At the same time, the security and privacy concerns of AI
have galvanized the attention of experts from multiple disciplines. In this
research, we successfully mounted adversarial attacks on a federated learning
(FL) environment using three different datasets. The attacks leveraged
generative adversarial networks (GANs) to affect the learning process and
strive to reconstruct the private data of users by learning hidden features
from shared local model parameters. The attack was target-oriented drawing data
with distinct class distribution from the CIFAR- 10, MNIST, and Fashion-MNIST
respectively. Moreover, by measuring the Euclidean distance between the real
data and the reconstructed adversarial samples, we evaluated the performance of
the adversary in the learning processes in various scenarios. At last, we
successfully reconstructed the real data of the victim from the shared global
model parameters with all the applied datasets.

    

### [[2108.00702] Improving Deep Learning for HAR with shallow LSTMs](http://arxiv.org/abs/2108.00702)


  Recent studies in Human Activity Recognition (HAR) have shown that Deep
Learning methods are able to outperform classical Machine Learning algorithms.
One popular Deep Learning architecture in HAR is the DeepConvLSTM. In this
paper we propose to alter the DeepConvLSTM architecture to employ a 1-layered
instead of a 2-layered LSTM. We validate our architecture change on 5 publicly
available HAR datasets by comparing the predictive performance with and without
the change employing varying hidden units within the LSTM layer(s). Results
show that across all datasets, our architecture consistently improves on the
original one: Recognition performance increases up to 11.7% for the F1-score,
and our architecture significantly decreases the amount of learnable
parameters. This improvement over DeepConvLSTM decreases training time by as
much as 48%. Our results stand in contrast to the belief that one needs at
least a 2-layered LSTM when dealing with sequential data. Based on our results
we argue that said claim might not be applicable to sensor-based HAR.

    

### [[2108.00708] Group Fisher Pruning for Practical Network Compression](http://arxiv.org/abs/2108.00708)


  Network compression has been widely studied since it is able to reduce the
memory and computation cost during inference. However, previous methods seldom
deal with complicated structures like residual connections, group/depth-wise
convolution and feature pyramid network, where channels of multiple layers are
coupled and need to be pruned simultaneously. In this paper, we present a
general channel pruning approach that can be applied to various complicated
structures. Particularly, we propose a layer grouping algorithm to find coupled
channels automatically. Then we derive a unified metric based on Fisher
information to evaluate the importance of a single channel and coupled
channels. Moreover, we find that inference speedup on GPUs is more correlated
with the reduction of memory rather than FLOPs, and thus we employ the memory
reduction of each channel to normalize the importance. Our method can be used
to prune any structures including those with coupled channels. We conduct
extensive experiments on various backbones, including the classic ResNet and
ResNeXt, mobile-friendly MobileNetV2, and the NAS-based RegNet, both on image
classification and object detection which is under-explored. Experimental
results validate that our method can effectively prune sophisticated networks,
boosting inference speed without sacrificing accuracy.

    

### [[2108.00713] Cohort Bias Adaptation in Aggregated Datasets for Lesion Segmentation](http://arxiv.org/abs/2108.00713)


  Many automatic machine learning models developed for focal pathology (e.g.
lesions, tumours) detection and segmentation perform well, but do not
generalize as well to new patient cohorts, impeding their widespread adoption
into real clinical contexts. One strategy to create a more diverse,
generalizable training set is to naively pool datasets from different cohorts.
Surprisingly, training on this \it{big data} does not necessarily increase, and
may even reduce, overall performance and model generalizability, due to the
existence of cohort biases that affect label distributions. In this paper, we
propose a generalized affine conditioning framework to learn and account for
cohort biases across multi-source datasets, which we call Source-Conditioned
Instance Normalization (SCIN). Through extensive experimentation on three
different, large scale, multi-scanner, multi-centre Multiple Sclerosis (MS)
clinical trial MRI datasets, we show that our cohort bias adaptation method (1)
improves performance of the network on pooled datasets relative to naively
pooling datasets and (2) can quickly adapt to a new cohort by fine-tuning the
instance normalization parameters, thus learning the new cohort bias with only
10 labelled samples.

    

### [[2108.00735] Tensor completion using geodesics on Segre manifolds](http://arxiv.org/abs/2108.00735)


  We propose a Riemannian conjugate gradient (CG) optimization method for
finding low rank approximations of incomplete tensors. Our main contribution
consists of an explicit expression of the geodesics on the Segre manifold.
These are exploited in our algorithm to perform the retractions. We apply our
method to movie rating predictions in a recommender system for the MovieLens
dataset, and identification of pure fluorophores via fluorescent spectroscopy
with missing data. In this last application, we recover the tensor
decomposition from less than $10\%$ of the data.

    

### [[2108.00740] Multiplicative updates for symmetric-cone factorizations](http://arxiv.org/abs/2108.00740)


  Given a matrix $X\in \mathbb{R}^{m\times n}_+$ with non-negative entries, the
cone factorization problem over a cone $\mathcal{K}\subseteq \mathbb{R}^k$
concerns computing $\{ a_1,\ldots, a_{m} \} \subseteq \mathcal{K}$ and $\{
b_1,\ldots, b_{n} \} \subseteq~\mathcal{K}^*$ belonging to its dual so that
$X_{ij} = \langle a_i, b_j \rangle$ for all $i\in [m], j\in [n]$. Cone
factorizations are fundamental to mathematical optimization as they allow us to
express convex bodies as feasible regions of linear conic programs. In this
paper, we introduce and analyze the symmetric-cone multiplicative update (SCMU)
algorithm for computing cone factorizations when $\mathcal{K}$ is symmetric;
i.e., it is self-dual and homogeneous. Symmetric cones are of central interest
in mathematical optimization as they provide a common language for studying
linear optimization over the nonnegative orthant (linear programs), over the
second-order cone (second order cone programs), and over the cone of positive
semidefinite matrices (semidefinite programs). The SCMU algorithm is
multiplicative in the sense that the iterates are updated by applying a
meticulously chosen automorphism of the cone computed using a generalization of
the geometric mean to symmetric cones. Using an extension of Lieb's concavity
theorem and von Neumann's trace inequality to symmetric cones, we show that the
squared loss objective is non-decreasing along the trajectories of the SCMU
algorithm. Specialized to the nonnegative orthant, the SCMU algorithm
corresponds to the seminal algorithm by Lee and Seung for computing Nonnegative
Matrix Factorizations.

    

### [[2108.00751] Data-driven model for hydraulic fracturing design optimization. Part II: Inverse problem](http://arxiv.org/abs/2108.00751)


  We describe a stacked model for predicting the cumulative fluid production
for an oil well with a multistage-fracture completion based on a combination of
Ridge Regression and CatBoost algorithms. The model is developed based on an
extended digital field data base of reservoir, well and fracturing design
parameters. The database now includes more than 5000 wells from 23 oilfields of
Western Siberia (Russia), with 6687 fracturing operations in total. Starting
with 387 parameters characterizing each well, including construction, reservoir
properties, fracturing design features and production, we end up with 38 key
parameters used as input features for each well in the model training process.
The model demonstrates physically explainable dependencies plots of the target
on the design parameters (number of stages, proppant mass, average and final
proppant concentrations and fluid rate). We developed a set of methods
including those based on the use of Euclidean distance and clustering
techniques to perform similar (offset) wells search, which is useful for a
field engineer to analyze earlier fracturing treatments on similar wells. These
approaches are also adapted for obtaining the optimization parameters
boundaries for the particular pilot well, as part of the field testing campaign
of the methodology. An inverse problem (selecting an optimum set of fracturing
design parameters to maximize production) is formulated as optimizing a high
dimensional black box approximation function constrained by boundaries and
solved with four different optimization methods: surrogate-based optimization,
sequential least squares programming, particle swarm optimization and
differential evolution. A recommendation system containing all the above
methods is designed to advise a production stimulation engineer on an optimized
fracturing design.

    

### [[2108.00752] Flip Learning: Erase to Segment](http://arxiv.org/abs/2108.00752)


  Nodule segmentation from breast ultrasound images is challenging yet
essential for the diagnosis. Weakly-supervised segmentation (WSS) can help
reduce time-consuming and cumbersome manual annotation. Unlike existing
weakly-supervised approaches, in this study, we propose a novel and general WSS
framework called Flip Learning, which only needs the box annotation.
Specifically, the target in the label box will be erased gradually to flip the
classification tag, and the erased region will be considered as the
segmentation result finally. Our contribution is three-fold. First, our
proposed approach erases on superpixel level using a Multi-agent Reinforcement
Learning framework to exploit the prior boundary knowledge and accelerate the
learning process. Second, we design two rewards: classification score and
intensity distribution reward, to avoid under- and over-segmentation,
respectively. Third, we adopt a coarse-to-fine learning strategy to reduce the
residual errors and improve the segmentation performance. Extensively validated
on a large dataset, our proposed approach achieves competitive performance and
shows great potential to narrow the gap between fully-supervised and
weakly-supervised learning.

    

### [[2108.00774] A Random Matrix Perspective on Random Tensors](http://arxiv.org/abs/2108.00774)


  Tensor models play an increasingly prominent role in many fields, notably in
machine learning. In several applications of such models, such as community
detection, topic modeling and Gaussian mixture learning, one must estimate a
low-rank signal from a noisy tensor. Hence, understanding the fundamental
limits and the attainable performance of estimators of that signal inevitably
calls for the study of random tensors. Substantial progress has been achieved
on this subject thanks to recent efforts, under the assumption that the tensor
dimensions grow large. Yet, some of the most significant among these
results--in particular, a precise characterization of the abrupt phase
transition (in terms of signal-to-noise ratio) that governs the performance of
the maximum likelihood (ML) estimator of a symmetric rank-one model with
Gaussian noise--were derived on the basis of statistical physics ideas, which
are not easily accessible to non-experts.
In this work, we develop a sharply distinct approach, relying instead on
standard but powerful tools brought by years of advances in random matrix
theory. The key idea is to study the spectra of random matrices arising from
contractions of a given random tensor. We show how this gives access to
spectral properties of the random tensor itself. In the specific case of a
symmetric rank-one model with Gaussian noise, our technique yields a hitherto
unknown characterization of the local maximum of the ML problem that is global
above the phase transition threshold. This characterization is in terms of a
fixed-point equation satisfied by a formula that had only been previously
obtained via statistical physics methods. Moreover, our analysis sheds light on
certain properties of the landscape of the ML problem in the large-dimensional
setting. Our approach is versatile and can be extended to other models, such as
asymmetric, non-Gaussian and higher-order ones.

    

### [[2108.00780] Angle Based Feature Learning in GNN for 3D Object Detection using Point Cloud](http://arxiv.org/abs/2108.00780)


  In this paper, we present new feature encoding methods for Detection of 3D
objects in point clouds. We used a graph neural network (GNN) for Detection of
3D objects namely cars, pedestrians, and cyclists. Feature encoding is one of
the important steps in Detection of 3D objects. The dataset used is point cloud
data which is irregular and unstructured and it needs to be encoded in such a
way that ensures better feature encapsulation. Earlier works have used relative
distance as one of the methods to encode the features. These methods are not
resistant to rotation variance problems in Graph Neural Networks. We have
included angular-based measures while performing feature encoding in graph
neural networks. Along with that, we have performed a comparison between other
methods like Absolute, Relative, Euclidean distances, and a combination of the
Angle and Relative methods. The model is trained and evaluated on the subset of
the KITTI object detection benchmark dataset under resource constraints. Our
results demonstrate that a combination of angle measures and relative distance
has performed better than other methods. In comparison to the baseline
method(relative), it achieved better performance. We also performed time
analysis of various feature encoding methods.

    

### [[2108.00781] Generalization Properties of Stochastic Optimizers via Trajectory Analysis](http://arxiv.org/abs/2108.00781)


  Despite the ubiquitous use of stochastic optimization algorithms in machine
learning, the precise impact of these algorithms on generalization performance
in realistic non-convex settings is still poorly understood. In this paper, we
provide an encompassing theoretical framework for investigating the
generalization properties of stochastic optimizers, which is based on their
dynamics. We first prove a generalization bound attributable to the optimizer
dynamics in terms of the celebrated Fernique-Talagrand functional applied to
the trajectory of the optimizer. This data- and algorithm-dependent bound is
shown to be the sharpest possible in the absence of further assumptions. We
then specialize this result by exploiting the Markovian structure of stochastic
optimizers, deriving generalization bounds in terms of the (data-dependent)
transition kernels associated with the optimization algorithms. In line with
recent work that has revealed connections between generalization and
heavy-tailed behavior in stochastic optimization, we link the generalization
error to the local tail behavior of the transition kernels. We illustrate that
the local power-law exponent of the kernel acts as an effective dimension,
which decreases as the transitions become "less Gaussian". We support our
theory with empirical results from a variety of neural networks, and we show
that both the Fernique-Talagrand functional and the local power-law exponent
are predictive of generalization performance.

    

### [[2108.00783] CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms](http://arxiv.org/abs/2108.00783)


  Counterfactual explanations provide means for prescriptive model explanations
by suggesting actionable feature changes (e.g., increase income) that allow
individuals to achieve favorable outcomes in the future (e.g., insurance
approval). Choosing an appropriate method is a crucial aspect for meaningful
counterfactual explanations. As documented in recent reviews, there exists a
quickly growing literature with available methods. Yet, in the absence of
widely available opensource implementations, the decision in favor of certain
models is primarily based on what is readily available. Going forward - to
guarantee meaningful comparisons across explanation methods - we present CARLA
(Counterfactual And Recourse LibrAry), a python library for benchmarking
counterfactual explanation methods across both different data sets and
different machine learning models. In summary, our work provides the following
contributions: (i) an extensive benchmark of 11 popular counterfactual
explanation methods, (ii) a benchmarking framework for research on future
counterfactual explanation methods, and (iii) a standardized set of integrated
evaluation measures and data sets for transparent and extensive comparisons of
these methods. We have open-sourced CARLA and our experimental results on
Github, making them available as competitive baselines. We welcome
contributions from other research groups and practitioners.

    

### [[2108.00784] Towards Robust Object Detection: Bayesian RetinaNet for Homoscedastic Aleatoric Uncertainty Modeling](http://arxiv.org/abs/2108.00784)


  According to recent studies, commonly used computer vision datasets contain
about 4% of label errors. For example, the COCO dataset is known for its high
level of noise in data labels, which limits its use for training robust neural
deep architectures in a real-world scenario. To model such a noise, in this
paper we have proposed the homoscedastic aleatoric uncertainty estimation, and
present a series of novel loss functions to address the problem of image object
detection at scale. Specifically, the proposed functions are based on Bayesian
inference and we have incorporated them into the common community-adopted
object detection deep learning architecture RetinaNet. We have also shown that
modeling of homoscedastic aleatoric uncertainty using our novel functions
allows to increase the model interpretability and to improve the object
detection performance being evaluated on the COCO dataset.

    

### [[2108.00785] Learning to Learn to Demodulate with Uncertainty Quantification via Bayesian Meta-Learning](http://arxiv.org/abs/2108.00785)


  Meta-learning, or learning to learn, offers a principled framework for
few-shot learning. It leverages data from multiple related learning tasks to
infer an inductive bias that enables fast adaptation on a new task. The
application of meta-learning was recently proposed for learning how to
demodulate from few pilots. The idea is to use pilots received and stored for
offline use from multiple devices in order to meta-learn an adaptation
procedure with the aim of speeding up online training on new devices. Standard
frequentist learning, which can yield relatively accurate "hard" classification
decisions, is known to be poorly calibrated, particularly in the small-data
regime. Poor calibration implies that the soft scores output by the demodulator
are inaccurate estimates of the true probability of correct demodulation. In
this work, we introduce the use of Bayesian meta-learning via variational
inference for the purpose of obtaining well-calibrated few-pilot demodulators.
In a Bayesian framework, each neural network weight is represented by a
distribution, capturing epistemic uncertainty. Bayesian meta-learning optimizes
over the prior distribution of the weights. The resulting Bayesian ensembles
offer better calibrated soft decisions, at the computational cost of running
multiple instances of the neural network for demodulation. Numerical results
for single-input single-output Rayleigh fading channels with transmitter's
non-linearities are provided that compare symbol error rate and expected
calibration error for both frequentist and Bayesian meta-learning, illustrating
how the latter is both more accurate and better-calibrated.

    

### [[1906.01437] On the Efficiency of Sinkhorn and Greenkhorn and Their Acceleration for Optimal Transport](http://arxiv.org/abs/1906.01437)


  We present several new complexity results for the algorithms that
approximately solve the optimal transport (OT) problem between two discrete
probability measures with at most $n$ atoms. First, we improve the complexity
bound of a greedy variant of the Sinkhorn algorithm, known as
\textit{Greenkhorn} algorithm, from $\widetilde{O}(n^2\varepsilon^{-3})$ to
$\widetilde{O}(n^2\varepsilon^{-2})$. Notably, this matches the best known
complexity bound of the Sinkhorn algorithm and sheds the light to superior
practical performance of the Greenkhorn algorithm. Second, we generalize an
adaptive primal-dual accelerated gradient descent (APDAGD)
algorithm~\citep{Dvurechensky-2018-Computational} with mirror mapping $\phi$
and prove that the resulting APDAMD algorithm achieves the complexity bound of
$\widetilde{O}(n^2\sqrt{\delta}\varepsilon^{-1})$ where $\delta>0$ refers to
the regularity of $\phi$. We demonstrate that the complexity bound of
$\widetilde{O}(\min\{n^{9/4}\varepsilon^{-1}, n^2\varepsilon^{-2}\})$ is
invalid for the APDAGD algorithm and establish a new complexity bound of
$\widetilde{O}(n^{5/2}\varepsilon^{-1})$. Moreover, we propose a
\textit{deterministic} accelerated Sinkhorn algorithm and prove that it
achieves the complexity bound of $\widetilde{O}(n^{7/3}\varepsilon^{-4/3})$ by
incorporating an estimate sequence. Therefore, the accelerated Sinkhorn
algorithm outperforms the Sinkhorn and Greenkhorn algorithms in terms of
$1/\varepsilon$ and the APDAGD and accelerated alternating
minimization~\citep{Guminov-2021-Combination} algorithms in terms of $n$.
Finally, we conduct experiments on synthetic data and real images with the
proposed algorithms in the paper and demonstrate their efficiency via numerical
results.

    

### [[1907.01651] Selecting the independent coordinates of manifolds with large aspect ratios](http://arxiv.org/abs/1907.01651)


  Many manifold embedding algorithms fail apparently when the data manifold has
a large aspect ratio (such as a long, thin strip). Here, we formulate success
and failure in terms of finding a smooth embedding, showing also that the
problem is pervasive and more complex than previously recognized.
Mathematically, success is possible under very broad conditions, provided that
embedding is done by carefully selected eigenfunctions of the Laplace-Beltrami
operator $\Delta$. Hence, we propose a bicriterial Independent Eigencoordinate
Selection (IES) algorithm that selects smooth embeddings with few eigenvectors.
The algorithm is grounded in theory, has low computational overhead, and is
successful on synthetic and large real data.

    

### [[1909.08610] Sample Efficient Policy Gradient Methods with Recursive Variance Reduction](http://arxiv.org/abs/1909.08610)


  Improving the sample efficiency in reinforcement learning has been a
long-standing research problem. In this work, we aim to reduce the sample
complexity of existing policy gradient methods. We propose a novel policy
gradient algorithm called SRVR-PG, which only requires $O(1/\epsilon^{3/2})$
episodes to find an $\epsilon$-approximate stationary point of the nonconcave
performance function $J(\boldsymbol{\theta})$ (i.e., $\boldsymbol{\theta}$ such
that $\|\nabla J(\boldsymbol{\theta})\|_2^2\leq\epsilon$). This sample
complexity improves the existing result $O(1/\epsilon^{5/3})$ for stochastic
variance reduced policy gradient algorithms by a factor of
$O(1/\epsilon^{1/6})$. In addition, we also propose a variant of SRVR-PG with
parameter exploration, which explores the initial policy parameter from a prior
probability distribution. We conduct numerical experiments on classic control
problems in reinforcement learning to validate the performance of our proposed
algorithms.

    

### [[1909.13203] Learning transport cost from subset correspondence](http://arxiv.org/abs/1909.13203)


  Learning to align multiple datasets is an important problem with many
applications, and it is especially useful when we need to integrate multiple
experiments or correct for confounding. Optimal transport (OT) is a principled
approach to align datasets, but a key challenge in applying OT is that we need
to specify a transport cost function that accurately captures how the two
datasets are related. Reliable cost functions are typically not available and
practitioners often resort to using hand-crafted or Euclidean cost even if it
may not be appropriate. In this work, we investigate how to learn the cost
function using a small amount of side information which is often available. The
side information we consider captures subset correspondence -- i.e. certain
subsets of points in the two data sets are known to be related. For example, we
may have some images labeled as cars in both datasets; or we may have a common
annotated cell type in single-cell data from two batches. We develop an
end-to-end optimizer (OT-SI) that differentiates through the Sinkhorn algorithm
and effectively learns the suitable cost function from side information. On
systematic experiments in images, marriage-matching and single-cell RNA-seq,
our method substantially outperform state-of-the-art benchmarks.

    

### [[1911.08708] Take an Emotion Walk: Perceiving Emotions from Gaits Using Hierarchical Attention Pooling and Affective Mapping](http://arxiv.org/abs/1911.08708)


  We present an autoencoder-based semi-supervised approach to classify
perceived human emotions from walking styles obtained from videos or
motion-captured data and represented as sequences of 3D poses. Given the motion
on each joint in the pose at each time step extracted from 3D pose sequences,
we hierarchically pool these joint motions in a bottom-up manner in the
encoder, following the kinematic chains in the human body. We also constrain
the latent embeddings of the encoder to contain the space of
psychologically-motivated affective features underlying the gaits. We train the
decoder to reconstruct the motions per joint per time step in a top-down manner
from the latent embeddings. For the annotated data, we also train a classifier
to map the latent embeddings to emotion labels. Our semi-supervised approach
achieves a mean average precision of 0.84 on the Emotion-Gait benchmark
dataset, which contains both labeled and unlabeled gaits collected from
multiple sources. We outperform current state-of-art algorithms for both
emotion recognition and action recognition from 3D gaits by 7%--23% on the
absolute. More importantly, we improve the average precision by 10%--50% on the
absolute on classes that each makes up less than 25% of the labeled part of the
Emotion-Gait benchmark dataset.

    

### [[1912.02290] Hierarchical Indian Buffet Neural Networks for Bayesian Continual Learning](http://arxiv.org/abs/1912.02290)


  We place an Indian Buffet process (IBP) prior over the structure of a
Bayesian Neural Network (BNN), thus allowing the complexity of the BNN to
increase and decrease automatically. We further extend this model such that the
prior on the structure of each hidden layer is shared globally across all
layers, using a Hierarchical-IBP (H-IBP). We apply this model to the problem of
resource allocation in Continual Learning (CL) where new tasks occur and the
network requires extra resources. Our model uses online variational inference
with reparameterisation of the Bernoulli and Beta distributions, which
constitute the IBP and H-IBP priors. As we automatically learn the number of
weights in each layer of the BNN, overfitting and underfitting problems are
largely overcome. We show empirically that our approach offers a competitive
edge over existing methods in CL.

    

### [[1912.02620] Learning to synthesise the ageing brain without longitudinal data](http://arxiv.org/abs/1912.02620)


  How will my face look when I get older? Or, for a more challenging question:
How will my brain look when I get older? To answer this question one must
devise (and learn from data) a multivariate auto-regressive function which
given an image and a desired target age generates an output image. While
collecting data for faces may be easier, collecting longitudinal brain data is
not trivial. We propose a deep learning-based method that learns to simulate
subject-specific brain ageing trajectories without relying on longitudinal
data. Our method synthesises images conditioned on two factors: age (a
continuous variable), and status of Alzheimer's Disease (AD, an ordinal
variable). With an adversarial formulation we learn the joint distribution of
brain appearance, age and AD status, and define reconstruction losses to
address the challenging problem of preserving subject identity. We compare with
several benchmarks using two widely used datasets. We evaluate the quality and
realism of synthesised images using ground-truth longitudinal data and a
pre-trained age predictor. We show that, despite the use of cross-sectional
data, our model learns patterns of gray matter atrophy in the middle temporal
gyrus in patients with AD. To demonstrate generalisation ability, we train on
one dataset and evaluate predictions on the other. In conclusion, our model
shows an ability to separate age, disease influence and anatomy using only 2D
cross-sectional data that should should be useful in large studies into
neurodegenerative disease, that aim to combine several data sources. To
facilitate such future studies by the community at large our code is made
available at this https URL.

    

### [[1912.04884] Statistically Robust Neural Network Classification](http://arxiv.org/abs/1912.04884)


  Despite their numerous successes, there are many scenarios where adversarial
risk metrics do not provide an appropriate measure of robustness. For example,
test-time perturbations may occur in a probabilistic manner rather than being
generated by an explicit adversary, while the poor train--test generalization
of adversarial metrics can limit their usage to simple problems. Motivated by
this, we develop a probabilistic robust risk framework, the statistically
robust risk (SRR), which considers pointwise corruption distributions, as
opposed to worst-case adversaries. The SRR provides a distinct and
complementary measure of robust performance, compared to natural and
adversarial risk. We show that the SRR admits estimation and training schemes
which are as simple and efficient as for the natural risk: these simply require
noising the inputs, but with a principled derivation for exactly how and why
this should be done. Furthermore, we demonstrate both theoretically and
experimentally that it can provide superior generalization performance compared
with adversarial risks, enabling application to high-dimensional datasets.

    

### [[2001.01037] Explain and Improve: LRP-Inference Fine-Tuning for Image Captioning Models](http://arxiv.org/abs/2001.01037)


  This paper analyzes the predictions of image captioning models with attention
mechanisms beyond visualizing the attention itself. We develop variants of
layer-wise relevance propagation (LRP) and gradient-based explanation methods,
tailored to image captioning models with attention mechanisms. We compare the
interpretability of attention heatmaps systematically against the explanations
provided by explanation methods such as LRP, Grad-CAM, and Guided Grad-CAM. We
show that explanation methods provide simultaneously pixel-wise image
explanations (supporting and opposing pixels of the input image) and linguistic
explanations (supporting and opposing words of the preceding sequence) for each
word in the predicted captions. We demonstrate with extensive experiments that
explanation methods 1) can reveal additional evidence used by the model to make
decisions compared to attention; 2) correlate to object locations with high
precision; 3) are helpful to "debug" the model, e.g. by analyzing the reasons
for hallucinated object words. With the observed properties of explanations, we
further design an LRP-inference fine-tuning strategy that reduces the issue of
object hallucination in image captioning models, and meanwhile, maintains the
sentence fluency. We conduct experiments with two widely used attention
mechanisms: the adaptive attention mechanism calculated with the additive
attention and the multi-head attention mechanism calculated with the scaled dot
product.

    

### [[2001.10420] OPFython: A Python-Inspired Optimum-Path Forest Classifier](http://arxiv.org/abs/2001.10420)


  Machine learning techniques have been paramount throughout the last years,
being applied in a wide range of tasks, such as classification, object
recognition, person identification, and image segmentation. Nevertheless,
conventional classification algorithms, e.g., Logistic Regression, Decision
Trees, and Bayesian classifiers, might lack complexity and diversity, not
suitable when dealing with real-world data. A recent graph-inspired classifier,
known as the Optimum-Path Forest, has proven to be a state-of-the-art
technique, comparable to Support Vector Machines and even surpassing it in some
tasks. This paper proposes a Python-based Optimum-Path Forest framework,
denoted as OPFython, where all of its functions and classes are based upon the
original C language implementation. Additionally, as OPFython is a Python-based
library, it provides a more friendly environment and a faster prototyping
workspace than the C language.

    

### [[2002.10572] Millimeter Wave Communications with an Intelligent Reflector: Performance Optimization and Distributional Reinforcement Learning](http://arxiv.org/abs/2002.10572)


  In this paper, a novel framework is proposed to optimize the downlink
multi-user communication of a millimeter wave base station, which is assisted
by a reconfigurable intelligent reflector (IR). In particular, a channel
estimation approach is developed to measure the channel state information (CSI)
in real-time. First, for a perfect CSI scenario, the precoding transmission of
the BS and the reflection coefficient of the IR are jointly optimized, via an
iterative approach, so as to maximize the sum of downlink rates towards
multiple users. Next, in the imperfect CSI scenario, a distributional
reinforcement learning (DRL) approach is proposed to learn the optimal IR
reflection and maximize the expectation of downlink capacity. In order to model
the transmission rate's probability distribution, a learning algorithm, based
on quantile regression (QR), is developed, and the proposed QR-DRL method is
proved to converge to a stable distribution of downlink transmission rate.
Simulation results show that, in the error-free CSI scenario, the proposed
approach yields over 30% and 2-fold increase in the downlink sum-rate, compared
with a fixed IR reflection scheme and direct transmission scheme, respectively.
Simulation results also show that by deploying more IR elements, the downlink
sum-rate can be significantly improved. However, as the number of IR components
increases, more time is required for channel estimation, and the slope of
increase in the IR-aided transmission rate will become smaller. Furthermore,
under limited knowledge of CSI, simulation results show that the proposed
QR-DRL method, which learns a full distribution of the downlink rate, yields a
better prediction accuracy and improves the downlink rate by 10% for online
deployments, compared with a Q-learning baseline.

    

### [[2003.00120] End-to-end Robustness for Sensing-Reasoning Machine Learning Pipelines](http://arxiv.org/abs/2003.00120)


  As machine learning (ML) being applied to many mission-critical scenarios,
certifying ML model robustness becomes increasingly important. Many previous
works focuses on the robustness of independent ML and ensemble models, and can
only certify a very small magnitude of the adversarial perturbation. In this
paper, we take a different viewpoint and improve learning robustness by going
beyond independent ML and ensemble models. We aim at promoting the generic
Sensing-Reasoning machine learning pipeline which contains both the sensing
(e.g. deep neural networks) and reasoning (e.g. Markov logic networks (MLN))
components enriched with domain knowledge. Can domain knowledge help improve
learning robustness? Can we formally certify the end-to-end robustness of such
an ML pipeline? We first theoretically analyze the computational complexity of
checking the provable robustness in the reasoning component. We then derive the
provable robustness bound for several concrete reasoning components. We show
that for reasoning components such as MLN and a specific family of Bayesian
networks it is possible to certify the robustness of the whole pipeline even
with a large magnitude of perturbation which cannot be certified by existing
work. Finally, we conduct extensive real-world experiments on large scale
datasets to evaluate the certified robustness for Sensing-Reasoning ML
pipelines.

    

### [[2003.10933] Learn to Forget: Machine Unlearning via Neuron Masking](http://arxiv.org/abs/2003.10933)


  Nowadays, machine learning models, especially neural networks, become
prevalent in many real-world applications.These models are trained based on a
one-way trip from user data: as long as users contribute their data, there is
no way to withdraw; and it is well-known that a neural network memorizes its
training data. This contradicts the "right to be forgotten" clause of GDPR,
potentially leading to law violations. To this end, machine unlearning becomes
a popular research topic, which allows users to eliminate memorization of their
private data from a trained machine learning this http URL this paper, we propose
the first uniform metric called for-getting rate to measure the effectiveness
of a machine unlearning method. It is based on the concept of membership
inference and describes the transformation rate of the eliminated data from
"memorized" to "unknown" after conducting unlearning. We also propose a novel
unlearning method calledForsaken. It is superior to previous work in either
utility or efficiency (when achieving the same forgetting rate). We benchmark
Forsaken with eight standard datasets to evaluate its performance. The
experimental results show that it can achieve more than 90\% forgetting rate on
average and only causeless than 5\% accuracy loss.

    

### [[2005.10743] Tensor Clustering with Planted Structures: Statistical Optimality and Computational Limits](http://arxiv.org/abs/2005.10743)


  This paper studies the statistical and computational limits of high-order
clustering with planted structures. We focus on two clustering models, constant
high-order clustering (CHC) and rank-one higher-order clustering (ROHC), and
study the methods and theory for testing whether a cluster exists (detection)
and identifying the support of cluster (recovery).
Specifically, we identify the sharp boundaries of signal-to-noise ratio for
which CHC and ROHC detection/recovery are statistically possible. We also
develop the tight computational thresholds: when the signal-to-noise ratio is
below these thresholds, we prove that polynomial-time algorithms cannot solve
these problems under the computational hardness conjectures of hypergraphic
planted clique (HPC) detection and hypergraphic planted dense subgraph (HPDS)
recovery. We also propose polynomial-time tensor algorithms that achieve
reliable detection and recovery when the signal-to-noise ratio is above these
thresholds. Both sparsity and tensor structures yield the computational
barriers in high-order tensor clustering. The interplay between them results in
significant differences between high-order tensor clustering and matrix
clustering in literature in aspects of statistical and computational phase
transition diagrams, algorithmic approaches, hardness conjecture, and proof
techniques. To our best knowledge, we are the first to give a thorough
characterization of the statistical and computational trade-off for such a
double computational-barrier problem. Finally, we provide evidence for the
computational hardness conjectures of HPC detection (via low-degree polynomial
and Metropolis methods) and HPDS recovery (via low-degree polynomial method).

    

### [[2006.05630] Distributional Robust Batch Contextual Bandits](http://arxiv.org/abs/2006.05630)


  Policy learning using historical observational data is an important problem
that has found widespread applications. Examples include selecting offers,
prices, advertisements to send to customers, as well as selecting which
medication to prescribe to a patient. However, existing literature rests on the
crucial assumption that the future environment where the learned policy will be
deployed is the same as the past environment that has generated the data--an
assumption that is often false or too coarse an approximation. In this paper,
we lift this assumption and aim to learn a distributional robust policy with
incomplete (bandit) observational data. We propose a novel learning algorithm
that is able to learn a robust policy to adversarial perturbations and unknown
covariate shifts. We first present a policy evaluation procedure in the
ambiguous environment and then give a performance guarantee based on the theory
of uniform convergence. Additionally, we also give a heuristic algorithm to
solve the distributional robust policy learning problems efficiently. Finally,
we demonstrate the robustness of our methods in the synthetic and real-world
datasets.

    

### [[2006.06863] Few-shot Neural Architecture Search](http://arxiv.org/abs/2006.06863)


  Efficient evaluation of a network architecture drawn from a large search
space remains a key challenge in Neural Architecture Search (NAS). Vanilla NAS
evaluates each architecture by training from scratch, which gives the true
performance but is extremely time-consuming. Recently, one-shot NAS
substantially reduces the computation cost by training only one supernetwork,
a.k.a. supernet, to approximate the performance of every architecture in the
search space via weight-sharing. However, the performance estimation can be
very inaccurate due to the co-adaption among operations. In this paper, we
propose few-shot NAS that uses multiple supernetworks, called sub-supernet,
each covering different regions of the search space to alleviate the undesired
co-adaption. Compared to one-shot NAS, few-shot NAS improves the accuracy of
architecture evaluation with a small increase of evaluation cost. With only up
to 7 sub-supernets, few-shot NAS establishes new SoTAs: on ImageNet, it finds
models that reach 80.5% top-1 accuracy at 600 MB FLOPS and 77.5% top-1 accuracy
at 238 MFLOPS; on CIFAR10, it reaches 98.72% top-1 accuracy without using extra
data or transfer learning. In Auto-GAN, few-shot NAS outperforms the previously
published results by up to 20%. Extensive experiments show that few-shot NAS
significantly improves various one-shot methods, including 4 gradient-based and
6 search-based methods on 3 different tasks in NasBench-201 and
NasBench1-shot-1.

    

### [[2006.12958] Can you tell? SSNet -- a Sagittal Stratum-inspired Neural Network Framework for Sentiment Analysis](http://arxiv.org/abs/2006.12958)


  When people try to understand nuanced language they typically process
multiple input sensor modalities to complete this cognitive task. It turns out
the human brain has even a specialized neuron formation, called sagittal
stratum, to help us understand sarcasm. We use this biological formation as the
inspiration for designing a neural network architecture that combines
predictions of different models on the same text to construct robust, accurate
and computationally efficient classifiers for sentiment analysis and study
several different realizations. Among them, we propose a systematic new
approach to combining multiple predictions based on a dedicated neural network
and develop mathematical analysis of it along with state-of-the-art
experimental results. We also propose a heuristic-hybrid technique for
combining models and back it up with experimental results on a representative
benchmark dataset and comparisons to other methods to show the advantages of
the new approaches.

    

### [[2007.04395] Multi-Level Graph Matching Networks for Deep Graph Similarity Learning](http://arxiv.org/abs/2007.04395)


  While the celebrated graph neural networks yield effective representations
for individual nodes of a graph, there has been relatively less success in
extending to the task of graph similarity learning. Recent work on graph
similarity learning has considered either global-level graph-graph interactions
or low-level node-node interactions, however ignoring the rich cross-level
interactions (e.g., between each node of one graph and the other whole graph).
In this paper, we propose a multi-level graph matching network (MGMN) framework
for computing the graph similarity between any pair of graph-structured objects
in an end-to-end fashion. In particular, the proposed MGMN consists of a
node-graph matching network for effectively learning cross-level interactions
between each node of one graph and the other whole graph, and a siamese graph
neural network to learn global-level interactions between two input graphs.
Furthermore, to compensate for the lack of standard benchmark datasets, we have
created and collected a set of datasets for both the graph-graph classification
and graph-graph regression tasks with different sizes in order to evaluate the
effectiveness and robustness of our models. Comprehensive experiments
demonstrate that MGMN consistently outperforms state-of-the-art baseline models
on both the graph-graph classification and graph-graph regression tasks.
Compared with previous work, MGMN also exhibits stronger robustness as the
sizes of the two input graphs increase.

    

### [[2007.09060] Self-Supervised Learning of Context-Aware Pitch Prosody Representations](http://arxiv.org/abs/2007.09060)


  In music and speech, meaning is derived at multiple levels of context.
Affect, for example, can be inferred both by a short sound token and by sonic
patterns over a longer temporal window such as an entire recording. In this
letter, we focus on inferring meaning from this dichotomy of contexts. We show
how contextual representations of short sung vocal lines can be implicitly
learned from fundamental frequency ($F_0$) and thus be used as a meaningful
feature space for downstream Music Information Retrieval (MIR) tasks. We
propose three self-supervised deep learning paradigms which leverage pseudotask
learning of these two levels of context to produce latent representation
spaces. We evaluate the usefulness of these representations by embedding unseen
pitch contours into each space and conducting downstream classification tasks.
Our results show that contextual representation can enhance downstream
classification by as much as 15\% as compared to using traditional statistical
contour features.

    

### [[2007.13086] Anonymizing Machine Learning Models](http://arxiv.org/abs/2007.13086)


  There is a known tension between the need to analyze personal data to drive
business and privacy concerns. Many data protection regulations, including the
EU General Data Protection Regulation (GDPR) and the California Consumer
Protection Act (CCPA), set out strict restrictions and obligations on the
collection and processing of personal data. Moreover, machine learning models
themselves can be used to derive personal information, as demonstrated by
recent membership and attribute inference attacks. Anonymized data, however, is
exempt from the obligations set out in these regulations. It is therefore
desirable to be able to create models that are anonymized, thus also exempting
them from those obligations, in addition to providing better protection against
attacks.
Learning on anonymized data typically results in significant degradation in
accuracy. In this work, we propose a method that is able to achieve better
model accuracy by using the knowledge encoded within the trained model, and
guiding our anonymization process to minimize the impact on the model's
accuracy, a process we call accuracy-guided anonymization. We demonstrate that
by focusing on the model's accuracy rather than generic information loss
measures, our method outperforms state of the art k-anonymity methods in terms
of the achieved utility, in particular with high values of k and large numbers
of quasi-identifiers.
We also demonstrate that our approach has a similar, and sometimes even
better ability to prevent membership inference attacks as approaches based on
differential privacy, while averting some of their drawbacks such as
complexity, performance overhead and model-specific implementations. This makes
model-guided anonymization a legitimate substitute for such methods and a
practical approach to creating privacy-preserving models.

    

### [[2008.04005] Deterministic error bounds for kernel-based learning techniques under bounded noise](http://arxiv.org/abs/2008.04005)


  We consider the problem of reconstructing a function from a finite set of
noise-corrupted samples. Two kernel algorithms are analyzed, namely kernel
ridge regression and $\varepsilon$-support vector regression. By assuming the
ground-truth function belongs to the reproducing kernel Hilbert space of the
chosen kernel, and the measurement noise affecting the dataset is bounded, we
adopt an approximation theory viewpoint to establish \textit{deterministic},
finite-sample error bounds for the two models. Finally, we discuss their
connection with Gaussian processes and two numerical examples are provided. In
establishing our inequalities, we hope to help bring the fields of
non-parametric kernel learning and system identification for robust control
closer to each other.

    

### [[2009.04695] Momentum-based Gradient Methods in Multi-Objective Recommendation](http://arxiv.org/abs/2009.04695)


  Multi-objective gradient methods are becoming the standard for solving
multi-objective problems. Among others, they show promising results in
developing multi-objective recommender systems with both correlated and
conflicting objectives. Classic multi-gradient descent usually relies on the
combination of the gradients, not including the computation of first and second
moments of the gradients. This leads to a brittle behavior and misses important
areas in the solution space. In this work, we create a multi-objective
model-agnostic Adamize method that leverages the benefits of the Adam optimizer
in single-objective problems. This corrects and stabilizes the gradients of
every objective before calculating a common gradient descent vector that
optimizes all the objectives simultaneously. We evaluate the benefits of
multi-objective Adamize on two multi-objective recommender systems and for
three different objective combinations, both correlated or conflicting. We
report significant improvements, measured with three different Pareto front
metrics: hypervolume, coverage, and spacing. Finally, we show that the Adamized
Pareto front strictly dominates the previous one on multiple objective pairs.

    

### [[2009.06606] Adaptive KL-UCB based Bandit Algorithms for Markovian and i.i.d. Settings](http://arxiv.org/abs/2009.06606)


  In the regret-based formulation of multi-armed bandit (MAB) problems, except
in rare instances, much of the literature focuses on arms with i.i.d. rewards.
In this paper, we consider the problem of obtaining regret guarantees for MAB
problems in which the rewards of each arm form a Markov chain which may not
belong to a single parameter exponential family. To achieve logarithmic regret
in such problems is not difficult: a variation of standard KL-UCB does the job.
However, the constants obtained from such an analysis are poor for the
following reason: i.i.d. rewards are a special case of Markov rewards and it is
difficult to design an algorithm that works well independent of whether the
underlying model is truly Markovian or i.i.d. To overcome this issue, we
introduce a novel algorithm that identifies whether the rewards from each arm
are truly Markovian or i.i.d. using a Hellinger distance-based test. Our
algorithm then switches from using a standard KL-UCB to a specialized version
of KL-UCB when it determines that the arm reward is Markovian, thus resulting
in low regret for both i.i.d. and Markovian settings.

    

### [[2010.04007] Filtering in tractography using autoencoders (FINTA)](http://arxiv.org/abs/2010.04007)


  Current brain white matter fiber tracking techniques show a number of
problems, including: generating large proportions of streamlines that do not
accurately describe the underlying anatomy; extracting streamlines that are not
supported by the underlying diffusion signal; and under-representing some fiber
populations, among others. In this paper, we describe a novel autoencoder-based
learning method to filter streamlines from diffusion MRI tractography, and
hence, to obtain more reliable tractograms. Our method, dubbed FINTA (Filtering
in Tractography using Autoencoders) uses raw, unlabeled tractograms to train
the autoencoder, and to learn a robust representation of brain streamlines.
Such an embedding is then used to filter undesired streamline samples using a
nearest neighbor algorithm. Our experiments on both synthetic and in vivo human
brain diffusion MRI tractography data obtain accuracy scores exceeding the 90\%
threshold on the test set. Results reveal that FINTA has a superior filtering
performance compared to conventional, anatomy-based methods, and the
RecoBundles state-of-the-art method. Additionally, we demonstrate that FINTA
can be applied to partial tractograms without requiring changes to the
framework. We also show that the proposed method generalizes well across
different tracking methods and datasets, and shortens significantly the
computation time for large (>1 M streamlines) tractograms. Together, this work
brings forward a new deep learning framework in tractography based on
autoencoders, which offers a flexible and powerful method for white matter
filtering and bundling that could enhance tractometry and connectivity
analyses.

    

### [[2010.09394] Knowledge Graph-based Question Answering with Electronic Health Records](http://arxiv.org/abs/2010.09394)


  Question Answering (QA) is a widely-used framework for developing and
evaluating an intelligent machine. In this light, QA on Electronic Health
Records (EHR), namely EHR QA, can work as a crucial milestone towards
developing an intelligent agent in healthcare. EHR data are typically stored in
a relational database, which can also be converted to a directed acyclic graph,
allowing two approaches for EHR QA: Table-based QA and Knowledge Graph-based
QA. We hypothesize that the graph-based approach is more suitable for EHR QA as
graphs can represent relations between entities and values more naturally
compared to tables, which essentially require JOIN operations. In this paper,
we propose a graph-based EHR QA where natural language queries are converted to
SPARQL instead of SQL. To validate our hypothesis, we create four EHR QA
datasets (graph-based VS table-based, and simplified database schema VS
original database schema), based on a table-based dataset MIMICSQL. We test
both a simple Seq2Seq model and a state-of-the-art EHR QA model on all datasets
where the graph-based datasets facilitated up to 34% higher accuracy than the
table-based dataset without any modification to the model architectures.
Finally, all datasets are open-sourced to encourage further EHR QA research in
both directions.

    

### [[2010.10935] An Eager Splitting Strategy for Online Decision Trees](http://arxiv.org/abs/2010.10935)


  Decision tree ensembles are widely used in practice. In this work, we study
in ensemble settings the effectiveness of replacing the split strategy for the
state-of-the-art online tree learner, Hoeffding Tree, with a rigorous but more
eager splitting strategy that we had previously published as Hoeffding AnyTime
Tree. Hoeffding AnyTime Tree (HATT), uses the Hoeffding Test to determine
whether the current best candidate split is superior to the current split, with
the possibility of revision, while Hoeffding Tree aims to determine whether the
top candidate is better than the second best and if a test is selected, fixes
it for all posterity. HATT converges to the ideal batch tree while Hoeffding
Tree does not. We find that HATT is an efficacious base learner for online
bagging and online boosting ensembles. On UCI and synthetic streams, HATT as a
base learner outperforms HT within a 0.05 significance level for the majority
of tested ensembles on what we believe is the largest and most comprehensive
set of testbenches in the online learning literature. Our results indicate that
HATT is a superior alternative to Hoeffding Tree in a large number of ensemble
settings.

    

### [[2011.02284] Surgical Data Science -- from Concepts toward Clinical Translation](http://arxiv.org/abs/2011.02284)


  Recent developments in data science in general and machine learning in
particular have transformed the way experts envision the future of surgery.
Surgical Data Science (SDS) is a new research field that aims to improve the
quality of interventional healthcare through the capture, organization,
analysis and modeling of data. While an increasing number of data-driven
approaches and clinical applications have been studied in the fields of
radiological and clinical data science, translational success stories are still
lacking in surgery. In this publication, we shed light on the underlying
reasons and provide a roadmap for future advances in the field. Based on an
international workshop involving leading researchers in the field of SDS, we
review current practice, key achievements and initiatives as well as available
standards and tools for a number of topics relevant to the field, namely (1)
infrastructure for data acquisition, storage and access in the presence of
regulatory constraints, (2) data annotation and sharing and (3) data analytics.
We further complement this technical perspective with (4) a review of currently
available SDS products and the translational progress from academia and (5) a
roadmap for faster clinical translation and exploitation of the full potential
of SDS, based on an international multi-round Delphi process.

    

### [[2011.04026] Pathwise Conditioning of Gaussian Processes](http://arxiv.org/abs/2011.04026)


  As Gaussian processes are used to answer increasingly complex questions,
analytic solutions become scarcer and scarcer. Monte Carlo methods act as a
convenient bridge for connecting intractable mathematical expressions with
actionable estimates via sampling. Conventional approaches for simulating
Gaussian process posteriors view samples as draws from marginal distributions
of process values at finite sets of input locations. This distribution-centric
characterization leads to generative strategies that scale cubically in the
size of the desired random vector. These methods are prohibitively expensive in
cases where we would, ideally, like to draw high-dimensional vectors or even
continuous sample paths. In this work, we investigate a different line of
reasoning: rather than focusing on distributions, we articulate Gaussian
conditionals at the level of random variables. We show how this pathwise
interpretation of conditioning gives rise to a general family of approximations
that lend themselves to efficiently sampling Gaussian process posteriors.
Starting from first principles, we derive these methods and analyze the
approximation errors they introduce. We, then, ground these results by
exploring the practical implications of pathwise conditioning in various
applied settings, such as global optimization and reinforcement learning.

    

### [[2011.05260] ATCN: Resource-Efficient Processing of Time Series on Edge](http://arxiv.org/abs/2011.05260)


  This paper presents a scalable deep learning model called Agile Temporal
Convolutional Network (ATCN) for high-accurate fast classification and time
series prediction in resource-constrained embedded systems. ATCN is a family of
compact networks with formalized hyperparameters that enable
application-specific adjustments to be made to the model architecture. It is
primarily designed for embedded edge devices with very limited performance and
memory, such as wearable biomedical devices and real-time reliability
monitoring systems. ATCN makes fundamental improvements over the mainstream
temporal convolutional neural networks, including residual connections as time
attention machines to increase the network depth and accuracy and the
incorporation of separable depth-wise convolution to reduce the computational
complexity of the model. As part of the present work, three ATCN families,
namely T0, T1, and T2, are also presented and evaluated on different ranges of
embedded processors - Cortex-M7 and Cortex-A57 processor. An evaluation of the
ATCN models against the best-in-class InceptionTime shows that ATCN improves
both accuracy and execution time on a broad range of embedded and
cyber-physical applications with demand for real-time processing on the
embedded edge. At the same time, in contrast to existing solutions, ATCN is the
first deep learning-based approach that can be run on embedded microcontrollers
(Cortex-M7) with limited computational performance and memory capacity while
delivering state-of-the-art accuracy.

    

### [[2011.09335] A Tunnel Gaussian Process Model for Learning Interpretable Flight's Landing Parameters](http://arxiv.org/abs/2011.09335)


  Approach and landing accidents have resulted in a significant number of hull
losses worldwide. Technologies (e.g., instrument landing system) and procedures
(e.g., stabilized approach criteria) have been developed to reduce the risks.
In this paper, we propose a data-driven method to learn and interpret flight's
approach and landing parameters to facilitate comprehensible and actionable
insights into flight dynamics. Specifically, we develop two variants of tunnel
Gaussian process (TGP) models to elucidate aircraft's approach and landing
dynamics using advanced surface movement guidance and control system (A-SMGCS)
data, which then indicates the stability of flight. TGP hybridizes the
strengths of sparse variational Gaussian process and polar Gaussian process to
learn from a large amount of data in cylindrical coordinates. We examine TGP
qualitatively and quantitatively by synthesizing three complex trajectory
datasets and compared TGP against existing methods on trajectory learning.
Empirically, TGP demonstrates superior modeling performance. When applied to
operational A-SMGCS data, TGP provides the generative probabilistic description
of landing dynamics and interpretable tunnel views of approach and landing
parameters. These probabilistic tunnel models can facilitate the analysis of
procedure adherence and augment existing aircrew and air traffic controllers'
displays during the approach and landing procedures, enabling necessary
corrective actions.

    

### [[2012.00096] Multi-Modal Detection of Alzheimer's Disease from Speech and Text](http://arxiv.org/abs/2012.00096)


  Reliable detection of the prodromal stages of Alzheimer's disease (AD)
remains difficult even today because, unlike other neurocognitive impairments,
there is no definitive diagnosis of AD in vivo. In this context, existing
research has shown that patients often develop language impairment even in mild
AD conditions. We propose a multimodal deep learning method that utilizes
speech and the corresponding transcript simultaneously to detect AD. For audio
signals, the proposed audio-based network, a convolutional neural network (CNN)
based model, predicts the diagnosis for multiple speech segments, which are
combined for the final prediction. Similarly, we use contextual embedding
extracted from BERT concatenated with a CNN-generated embedding for classifying
the transcript. The individual predictions of the two models are then combined
to make the final classification. We also perform experiments to analyze the
model performance when Automated Speech Recognition (ASR) system generated
transcripts are used instead of manual transcription in the text-based model.
The proposed method achieves 85.3% 10-fold cross-validation accuracy when
trained and evaluated on the Dementiabank Pitt corpus.

    

### [[2012.01959] Towards Human Haptic Gesture Interpretation for Robotic Systems](http://arxiv.org/abs/2012.01959)


  Physical human-robot interactions (pHRI) are less efficient and communicative
than human-human interactions, and a key reason is a lack of informative sense
of touch in robotic systems. Interpreting human touch gestures is a nuanced,
challenging task with extreme gaps between human and robot capability. Among
prior works that demonstrate human touch recognition capability, differences in
sensors, gesture classes, feature sets, and classification algorithms yield a
conglomerate of non-transferable results and a glaring lack of a standard. To
address this gap, this work presents 1) four proposed touch gesture classes
that cover an important subset of the gesture characteristics identified in the
literature, 2) the collection of an extensive force dataset on a common pHRI
robotic arm with only its internal wrist force-torque sensor, and 3) an
exhaustive performance comparison of combinations of feature sets and
classification algorithms on this dataset. We demonstrate high classification
accuracies among our proposed gesture definitions on a test set, emphasizing
that neural net-work classifiers on the raw data outperform other combinations
of feature sets and algorithms. The accompanying video is here:
this https URL


### [[2012.03011] MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements](http://arxiv.org/abs/2012.03011)


  Hyperparameter optimization (HPO) is a fundamental problem in automatic
machine learning (AutoML). However, due to the expensive evaluation cost of
models (e.g., training deep learning models or training models on large
datasets), vanilla Bayesian optimization (BO) is typically computationally
infeasible. To alleviate this issue, Hyperband (HB) utilizes the early stopping
mechanism to speed up configuration evaluations by terminating those
badly-performing configurations in advance. This leads to two kinds of quality
measurements: (1) many low-fidelity measurements for configurations that get
early-stopped, and (2) few high-fidelity measurements for configurations that
are evaluated without being early stopped. The state-of-the-art HB-style
method, BOHB, aims to combine the benefits of both BO and HB. Instead of
sampling configurations randomly in HB, BOHB samples configurations based on a
BO surrogate model, which is constructed with the high-fidelity measurements
only. However, the scarcity of high-fidelity measurements greatly hampers the
efficiency of BO to guide the configuration search. In this paper, we present
MFES-HB, an efficient Hyperband method that is capable of utilizing both the
high-fidelity and low-fidelity measurements to accelerate the convergence of
HPO tasks. Designing MFES-HB is not trivial as the low-fidelity measurements
can be biased yet informative to guide the configuration search. Thus we
propose to build a Multi- Fidelity Ensemble Surrogate (MFES) based on the
generalized Product of Experts framework, which can integrate useful
information from multi-fidelity measurements effectively. The empirical studies
on the real-world AutoML tasks demonstrate that MFES-HB can achieve 3.3-8.9x
speedups over the state-of-the-art approach - BOHB.

    

### [[2012.04201] GPU Accelerated Exhaustive Search for Optimal Ensemble of Black-Box Optimization Algorithms](http://arxiv.org/abs/2012.04201)


  Black-box optimization is essential for tuning complex machine learning
algorithms which are easier to experiment with than to understand. In this
paper, we show that a simple ensemble of black-box optimization algorithms can
outperform any single one of them. However, searching for such an optimal
ensemble requires a large number of experiments. We propose a
Multi-GPU-optimized framework to accelerate a brute force search for the
optimal ensemble of black-box optimization algorithms by running many
experiments in parallel. The lightweight optimizations are performed by CPU
while expensive model training and evaluations are assigned to GPUs. We
evaluate 15 optimizers by training 2.7 million models and running 541,440
optimizations. On a DGX-1, the search time is reduced from more than 10 days on
two 20-core CPUs to less than 24 hours on 8-GPUs. With the optimal ensemble
found by GPU-accelerated exhaustive search, we won the 2nd place of NeurIPS
2020 black-box optimization challenge.

    

### [[2012.05082] Emergent Quantumness in Neural Networks](http://arxiv.org/abs/2012.05082)


  It was recently shown that the Madelung equations, that is, a hydrodynamic
form of the Schrödinger equation, can be derived from a canonical ensemble of
neural networks where the quantum phase was identified with the free energy of
hidden variables. We consider instead a grand canonical ensemble of neural
networks, by allowing an exchange of neurons with an auxiliary subsystem, to
show that the free energy must also be multivalued. By imposing the
multivaluedness condition on the free energy we derive the Schrödinger
equation with "Planck's constant" determined by the chemical potential of
hidden variables. This shows that quantum mechanics provides a correct
statistical description of the dynamics of the grand canonical ensemble of
neural networks at the learning equilibrium. We also discuss implications of
the results for machine learning, fundamental physics and, in a more
speculative way, evolutionary biology.

    

### [[2101.04223] Exploiting Multiple Timescales in Hierarchical Echo State Networks](http://arxiv.org/abs/2101.04223)


  Echo state networks (ESNs) are a powerful form of reservoir computing that
only require training of linear output weights whilst the internal reservoir is
formed of fixed randomly connected neurons. With a correctly scaled
connectivity matrix, the neurons' activity exhibits the echo-state property and
responds to the input dynamics with certain timescales. Tuning the timescales
of the network can be necessary for treating certain tasks, and some
environments require multiple timescales for an efficient representation. Here
we explore the timescales in hierarchical ESNs, where the reservoir is
partitioned into two smaller linked reservoirs with distinct properties. Over
three different tasks (NARMA10, a reconstruction task in a volatile
environment, and psMNIST), we show that by selecting the hyper-parameters of
each partition such that they focus on different timescales, we achieve a
significant performance improvement over a single ESN. Through a linear
analysis, and under the assumption that the timescales of the first partition
are much shorter than the second's (typically corresponding to optimal
operating conditions), we interpret the feedforward coupling of the partitions
in terms of an effective representation of the input signal, provided by the
first partition to the second, whereby the instantaneous input signal is
expanded into a weighted combination of its time derivatives. Furthermore, we
propose a data-driven approach to optimise the hyper-parameters through a
gradient descent optimisation method that is an online approximation of
backpropagation through time. We demonstrate the application of the online
learning rule across all the tasks considered.

    

### [[2101.08398] TDA-Net: Fusion of Persistent Homology and Deep Learning Features for COVID-19 Detection in Chest X-Ray Images](http://arxiv.org/abs/2101.08398)


  Topological Data Analysis (TDA) has emerged recently as a robust tool to
extract and compare the structure of datasets. TDA identifies features in data
such as connected components and holes and assigns a quantitative measure to
these features. Several studies reported that topological features extracted by
TDA tools provide unique information about the data, discover new insights, and
determine which feature is more related to the outcome. On the other hand, the
overwhelming success of deep neural networks in learning patterns and
relationships has been proven on a vast array of data applications, images in
particular. To capture the characteristics of both powerful tools, we propose
\textit{TDA-Net}, a novel ensemble network that fuses topological and deep
features for the purpose of enhancing model generalizability and accuracy. We
apply the proposed \textit{TDA-Net} to a critical application, which is the
automated detection of COVID-19 from CXR images. The experimental results
showed that the proposed network achieved excellent performance and suggests
the applicability of our method in practice.

    

### [[2101.10437] High-fidelity Prediction of Megapixel Longitudinal Phase-space Images of Electron Beams using Encoder-Decoder Neural Networks](http://arxiv.org/abs/2101.10437)


  Modeling of large-scale research facilities is extremely challenging due to
complex physical processes and engineering problems. Here, we adopt a
data-driven approach to model the longitudinal phase-space diagnostic beamline
at the photoinector of the European XFEL with an encoder-decoder neural network
model. A deep convolutional neural network (decoder) is used to build images
measured on the screen from a small feature map generated by another neural
network (encoder). We demonstrate that the model trained only with experimental
data can make high-fidelity predictions of megapixel images for the
longitudinal phase-space measurement without any prior knowledge of
photoinjectors and electron beams. The prediction significantly outperforms
existing methods. We also show the scalability and interpretability of the
model by sharing the same decoder with more than one encoder used for different
setups of the photoinjector, and propose a pragmatic way to model a facility
with various diagnostics and working points. This opens the door to a new way
of accurately modeling a photoinjector using neural networks and experimental
data. The approach can possibly be extended to the whole accelerator and even
other types of scientific facilities.

    

### [[2102.01751] Distributed Conditional Generative Adversarial Networks (GANs) for Data-Driven Millimeter Wave Communications in UAV Networks](http://arxiv.org/abs/2102.01751)


  In this paper, a novel framework is proposed to perform data-driven
air-to-ground (A2G) channel estimation for millimeter wave (mmWave)
communications in an unmanned aerial vehicle (UAV) wireless network. First, an
effective channel estimation approach is developed to collect mmWave channel
information, allowing each UAV to train a stand-alone channel model via a
conditional generative adversarial network (CGAN) along each beamforming
direction. Next, in order to expand the application scenarios of the trained
channel model into a broader spatial-temporal domain, a cooperative framework,
based on a distributed CGAN architecture, is developed, allowing each UAV to
collaboratively learn the mmWave channel distribution in a fully-distributed
manner. To guarantee an efficient learning process, necessary and sufficient
conditions for the optimal UAV network topology that maximizes the learning
rate for cooperative channel modeling are derived, and the optimal CGAN
learning solution per UAV is subsequently characterized, based on the
distributed network structure. Simulation results show that the proposed
distributed CGAN approach is robust to the local training error at each UAV.
Meanwhile, a larger airborne network size requires more communication resources
per UAV to guarantee an efficient learning rate. The results also show that,
compared with a stand-alone CGAN without information sharing and two other
distributed schemes, namely: A multi-discriminator CGAN and a federated CGAN
method, the proposed distributed CGAN approach yields a higher modeling
accuracy while learning the environment, and it achieves a larger average data
rate in the online performance of UAV downlink mmWave communications.

    

### [[2102.03336] Machine Learning Applications on Neuroimaging for Diagnosis and Prognosis of Epilepsy: A Review](http://arxiv.org/abs/2102.03336)


  Machine learning is playing an increasingly important role in medical image
analysis, spawning new advances in the clinical application of neuroimaging.
There have been some reviews of machine learning and epilepsy before, but they
mainly focused on electrophysiological signals such as
electroencephalography(EEG) or stereo electroencephalography(SEEG), while
ignoring the potential of neuroimaging in epilepsy research. Neuroimaging has
its important advantages in confirming the range of epileptic region, which
means a lot in presurgical evaluation and assessment after surgery. However,
EEG is difficult to locate the epilepsy lesion region in the brain. In this
review, we emphasize the interaction between neuroimaging and machine learning
in the context of the epilepsy diagnosis and prognosis. We start with an
overview of typical neuroimaging modalities used in epilepsy clinics, MRI, DTI,
fMRI, and PET. Then, we introduce three approaches for applying machine
learning methods to neuroimaging data: i) the two-step compositional approach
combining feature engineering and machine learning classifiers, ii) the
end-to-end approach, which is usually toward deep learning, and iii) the hybrid
approach using the advantages of the two methods. Subsequently, the application
of machine learning on epilepsy neuroimaging, such as segmentation,
localization and lateralization tasks, as well as tasks directly related to
diagnosis and prognosis are introduced in detail. Finally, we discuss the
current achievements, challenges, and potential future directions in this
field, hoping to pave the way for computer-aided diagnosis and prognosis of
epilepsy.

    

### [[2102.03479] Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2102.03479)


  Many complex multi-robot systems such as robot swarms control and autonomous
vehicle coordination can be modeled as Multi-Agent Reinforcement Learning
(MARL) tasks. QMIX, a widely popular MARL algorithm, has been used as a
baseline for the benchmark environments, e.g., Starcraft Multi-Agent Challenge
(SMAC), Difficulty-Enhanced Predator-Prey (DEPP). Recent variants of QMIX
target relaxing the monotonicity constraint of QMIX, allowing for performance
improvement in SMAC. In this paper, we investigate the code-level optimizations
of these variants and the monotonicity constraint. (1) We find that such
improvements of the variants are significantly affected by various code-level
optimizations. (2) The experiment results show that QMIX with normalized
optimizations outperforms other works in SMAC; (3) beyond the common wisdom
from these works, the monotonicity constraint can improve sample efficiency in
SMAC and DEPP. We also discuss why monotonicity constraints work well in purely
cooperative tasks with a theoretical analysis. We open-source the code at
\url{this https URL}.

    

### [[2102.04594] Rationally Inattentive Utility Maximization for Interpretable Deep Image Classification](http://arxiv.org/abs/2102.04594)


  Are deep convolutional neural networks (CNNs) for image classification
explainable by utility maximization with information acquisition costs? We
demonstrate that deep CNNs behave equivalently (in terms of necessary and
sufficient conditions) to rationally inattentive utility maximizers, a
generative model used extensively in economics for human decision making. Our
claim is based by extensive experiments on 200 deep CNNs from 5 popular
architectures. The parameters of our interpretable model are computed
efficiently via convex feasibility algorithms. As an application, we show that
our economics-based interpretable model can predict the classification
performance of deep CNNs trained with arbitrary parameters with accuracy
exceeding 94% . This eliminates the need to re-train the deep CNNs for image
classification. The theoretical foundation of our approach lies in Bayesian
revealed preference studied in micro-economics. All our results are on GitHub
and completely reproducible.

    

### [[2102.05123] Backdoor Scanning for Deep Neural Networks through K-Arm Optimization](http://arxiv.org/abs/2102.05123)


  Back-door attack poses a severe threat to deep learning systems. It injects
hidden malicious behaviors to a model such that any input stamped with a
special pattern can trigger such behaviors. Detecting back-door is hence of
pressing need. Many existing defense techniques use optimization to generate
the smallest input pattern that forces the model to misclassify a set of benign
inputs injected with the pattern to a target label. However, the complexity is
quadratic to the number of class labels such that they can hardly handle models
with many classes. Inspired by Multi-Arm Bandit in Reinforcement Learning, we
propose a K-Arm optimization method for backdoor detection. By iteratively and
stochastically selecting the most promising labels for optimization with the
guidance of an objective function, we substantially reduce the complexity,
allowing to handle models with many classes. Moreover, by iteratively refining
the selection of labels to optimize, it substantially mitigates the uncertainty
in choosing the right labels, improving detection accuracy. At the time of
submission, the evaluation of our method on over 4000 models in the IARPA
TrojAI competition from round 1 to the latest round 4 achieves top performance
on the leaderboard. Our technique also supersedes three state-of-the-art
techniques in terms of accuracy and the scanning time needed.

    

### [[2102.07365] A Unified Batch Selection Policy for Active Metric Learning](http://arxiv.org/abs/2102.07365)


  Active metric learning is the problem of incrementally selecting high-utility
batches of training data (typically, ordered triplets) to annotate, in order to
progressively improve a learned model of a metric over some input domain as
rapidly as possible. Standard approaches, which independently assess the
informativeness of each triplet in a batch, are susceptible to highly
correlated batches with many redundant triplets and hence low overall utility.
While a recent work \cite{kumari2020batch} proposes batch-decorrelation
strategies for metric learning, they rely on ad hoc heuristics to estimate the
correlation between two triplets at a time. We present a novel batch active
metric learning method that leverages the Maximum Entropy Principle to learn
the least biased estimate of triplet distribution for a given set of prior
constraints. To avoid redundancy between triplets, our method collectively
selects batches with maximum joint entropy, which simultaneously captures both
informativeness and diversity. We take advantage of the submodularity of the
joint entropy function to construct a tractable solution using an efficient
greedy algorithm based on Gram-Schmidt orthogonalization that is provably
$\left( 1 - \frac{1}{e} \right)$-optimal. Our approach is the first batch
active metric learning method to define a unified score that balances
informativeness and diversity for an entire batch of triplets. Experiments with
several real-world datasets demonstrate that our algorithm is robust,
generalizes well to different applications and input modalities, and
consistently outperforms the state-of-the-art.

    

### [[2102.10304] End-to-end neural network approach to 3D reservoir simulation and adaptation](http://arxiv.org/abs/2102.10304)


  Reservoir simulation and adaptation (also known as history matching) are
typically considered as separate problems. While a set of models are aimed at
the solution of the forward simulation problem assuming all initial geological
parameters are known, the other set of models adjust geological parameters
under the fixed forward simulation model to fit production data. This results
in many difficulties for both reservoir engineers and developers of new
efficient computation schemes. We present a unified approach to reservoir
simulation and adaptation problems. A single neural network model allows a
forward pass from initial geological parameters of the 3D reservoir model
through dynamic state variables to well's production rates and backward
gradient propagation to any model inputs and variables. The model fitting and
geological parameters adaptation both become the optimization problem over
specific parts of the same neural network model. Standard gradient-based
optimization schemes can be used to find the optimal solution. Using real-world
oilfield model and historical production rates we demonstrate that the
suggested approach allows reservoir simulation and history matching with a
benefit of several orders of magnitude simulation speed-up. Finally, to
propagate this research we open-source a Python-based framework DeepField that
allows standard processing of reservoir models and reproducing the approach
presented in this paper.

    

### [[2102.11055] Escaping from Zero Gradient: Revisiting Action-Constrained Reinforcement Learning via Frank-Wolfe Policy Optimization](http://arxiv.org/abs/2102.11055)


  Action-constrained reinforcement learning (RL) is a widely-used approach in
various real-world applications, such as scheduling in networked systems with
resource constraints and control of a robot with kinematic constraints. While
the existing projection-based approaches ensure zero constraint violation, they
could suffer from the zero-gradient problem due to the tight coupling of the
policy gradient and the projection, which results in sample-inefficient
training and slow convergence. To tackle this issue, we propose a learning
algorithm that decouples the action constraints from the policy parameter
update by leveraging state-wise Frank-Wolfe and a regression-based policy
update scheme. Moreover, we show that the proposed algorithm enjoys convergence
and policy improvement properties in the tabular case as well as generalizes
the popular DDPG algorithm for action-constrained RL in the general case.
Through experiments, we demonstrate that the proposed algorithm significantly
outperforms the benchmark methods on a variety of control tasks.

    

### [[2103.00501] Exploring the social influence of Kaggle virtual community on the M5 competition](http://arxiv.org/abs/2103.00501)


  One of the most significant differences of M5 over previous forecasting
competitions is that it was held on Kaggle, an online platform of data
scientists and machine learning practitioners. Kaggle provides a gathering
place, or virtual community, for web users who are interested in the M5
competition. Users can share code, models, features, loss functions, etc.
through online notebooks and discussion forums. This paper aims to study the
social influence of virtual community on user behaviors in the M5 competition.
We first research the content of the M5 virtual community by topic modeling and
trend analysis. Further, we perform social media analysis to identify the
potential relationship network of the virtual community. We study the roles and
characteristics of some key participants that promote the diffusion of
information within the M5 virtual community. Overall, this study provides
in-depth insights into the mechanism of the virtual community's influence on
the participants and has potential implications for future online competitions.

    

### [[2103.00683] Decision Making in Monopoly using a Hybrid Deep Reinforcement Learning Approach](http://arxiv.org/abs/2103.00683)


  Learning to adapt and make real-time informed decisions in a dynamic and
complex environment is a challenging problem. Monopoly is a popular strategic
board game that requires players to make multiple decisions during the game.
Decision-making in Monopoly involves many real-world elements such as
strategizing, luck, and modeling of opponent's policies. In this paper, we
present novel representations for the state and action space for the full
version of Monopoly and define an improved reward function. Using these, we
show that our deep reinforcement learning agent can learn winning strategies
for Monopoly against different fixed-policy agents. In Monopoly, players can
take multiple actions even if it is not their turn to roll the dice. Some of
these actions occur more frequently than others, resulting in a skewed
distribution that adversely affects the performance of the learning agent. To
tackle the non-uniform distribution of actions, we propose a hybrid approach
that combines deep reinforcement learning (for frequent but complex decisions)
with a fixed policy approach (for infrequent but straightforward decisions).
Experimental results show that our hybrid agent outperforms a standard deep
reinforcement learning agent by 30% in the number of games won against
fixed-policy agents.

    

### [[2103.02941] Exploring the representativeness of the M5 competition data](http://arxiv.org/abs/2103.02941)


  The main objective of the M5 competition, which focused on forecasting the
hierarchical unit sales of Walmart, was to evaluate the accuracy and
uncertainty of forecasting methods in the field in order to identify best
practices and highlight their practical implications. However, whether the
findings of the M5 competition can be generalized and exploited by retail firms
to better support their decisions and operation depends on the extent to which
the M5 data is sufficiently similar to unit sales data of retailers that
operate in different regions, sell different types of products, and consider
different marketing strategies. To answer this question, we analyze the
characteristics of the M5 time series and compare them with those of two
grocery retailers, namely Corporación Favorita and a major Greek supermarket
chain, using feature spaces. Our results suggest that there are only small
discrepancies between the examined data sets, supporting the representativeness
of the M5 data.

    

### [[2103.03102] Benchmarking Robustness of Deep Learning Classifiers Using Two-Factor Perturbation](http://arxiv.org/abs/2103.03102)


  The accuracy of DL classifiers is unstable in that it often changes
significantly when retested on adversarial images, imperfect images, or
perturbed images. This paper adds to the small but fundamental body of work on
benchmarking the robustness of DL classifiers on defective images. Unlike
existed single-factor digital perturbation work, we provide state-of-the-art
two-factor perturbation that provides two natural perturbations on images
applied in different sequences. The two-factor perturbation includes (1) two
digital perturbations (Salt & pepper noise and Gaussian noise) applied in both
sequences. (2) one digital perturbation (salt & pepper noise) and a geometric
perturbation (rotation) applied in different sequences. To measure robust DL
classifiers, previous scientists provided 15 types of single-factor corruption.
We created 69 benchmarking image sets, including a clean set, sets with single
factor perturbations, and sets with two-factor perturbation conditions. To be
best of our knowledge, this is the first report that two-factor perturbed
images improves both robustness and accuracy of DL classifiers. Previous
research evaluating deep learning (DL) classifiers has often used top-1/top-5
accuracy, so researchers have usually offered tables, line diagrams, and bar
charts to display accuracy of DL classifiers. But these existed approaches
cannot quantitively evaluate robustness of DL classifiers. We innovate a new
two-dimensional, statistical visualization tool, including mean accuracy and
coefficient of variation (CV), to benchmark the robustness of DL classifiers.
All source codes and related image sets are shared on websites
(this http URL or
this https URL ) to
support future academic research and industry projects.

    

### [[2103.04548] Learning to Control an Unstable System with One Minute of Data: Leveraging Gaussian Process Differentiation in Predictive Control](http://arxiv.org/abs/2103.04548)


  We present a straightforward and efficient way to control unstable robotic
systems using an estimated dynamics model. Specifically, we show how to exploit
the differentiability of Gaussian Processes to create a state-dependent
linearized approximation of the true continuous dynamics that can be integrated
with model predictive control. Our approach is compatible with most Gaussian
process approaches for system identification, and can learn an accurate model
using modest amounts of training data. We validate our approach by learning the
dynamics of an unstable system such as a segway with a 7-D state space and 2-D
input space (using only one minute of data), and we show that the resulting
controller is robust to unmodelled dynamics and disturbances, while
state-of-the-art control methods based on nominal models can fail under small
perturbations. Code is open sourced at
this https URL .

    

### [[2103.07492] Continual Learning for Recurrent Neural Networks: an Empirical Evaluation](http://arxiv.org/abs/2103.07492)


  Learning continuously during all model lifetime is fundamental to deploy
machine learning solutions robust to drifts in the data distribution. Advances
in Continual Learning (CL) with recurrent neural networks could pave the way to
a large number of applications where incoming data is non stationary, like
natural language processing and robotics. However, the existing body of work on
the topic is still fragmented, with approaches which are application-specific
and whose assessment is based on heterogeneous learning protocols and datasets.
In this paper, we organize the literature on CL for sequential data processing
by providing a categorization of the contributions and a review of the
benchmarks. We propose two new benchmarks for CL with sequential data based on
existing datasets, whose characteristics resemble real-world applications. We
also provide a broad empirical evaluation of CL and Recurrent Neural Networks
in class-incremental scenario, by testing their ability to mitigate forgetting
with a number of different strategies which are not specific to sequential data
processing. Our results highlight the key role played by the sequence length
and the importance of a clear specification of the CL scenario.

    

### [[2103.07626] Helmholtzian Eigenmap: Topological feature discovery & edge flow learning from point cloud data](http://arxiv.org/abs/2103.07626)


  The manifold Helmholtzian (1-Laplacian) operator $\Delta_1$ elegantly
generalizes the Laplace-Beltrami operator to vector fields on a manifold
$\mathcal M$. In this work, we propose the estimation of the manifold
Helmholtzian from point cloud data by a weighted 1-Laplacian $\mathbf{\mathcal
L}_1$. While higher order Laplacians ave been introduced and studied, this work
is the first to present a graph Helmholtzian constructed from a simplicial
complex as an estimator for the continuous operator in a non-parametric
setting. Equipped with the geometric and topological information about
$\mathcal M$, the Helmholtzian is a useful tool for the analysis of flows and
vector fields on $\mathcal M$ via the Helmholtz-Hodge theorem. In addition, the
$\mathbf{\mathcal L}_1$ allows the smoothing, prediction, and feature
extraction of the flows. We demonstrate these possibilities on substantial sets
of synthetic and real point cloud datasets with non-trivial topological
structures; and provide theoretical results on the limit of $\mathbf{\mathcal
L}_1$ to $\Delta_1$.

    

### [[2103.10368] MSMatch: Semi-Supervised Multispectral Scene Classification with Few Labels](http://arxiv.org/abs/2103.10368)


  Supervised learning techniques are at the center of many tasks in remote
sensing. Unfortunately, these methods, especially recent deep learning methods,
often require large amounts of labeled data for training. Even though
satellites acquire large amounts of data, labeling the data is often tedious,
expensive and requires expert knowledge. Hence, improved methods that require
fewer labeled samples are needed. We present MSMatch, the first semi-supervised
learning approach competitive with supervised methods on scene classification
on the EuroSAT and UC Merced Land Use benchmark datasets. We test both RGB and
multispectral images of EuroSAT and perform various ablation studies to
identify the critical parts of the model. The trained neural network achieves
state-of-the-art results on EuroSAT with an accuracy that is up to 19.76%
better than previous methods depending on the number of labeled training
examples. With just five labeled examples per class, we reach 94.53% and 95.86%
accuracy on the EuroSAT RGB and multispectral datasets, respectively. On the UC
Merced Land Use dataset, we outperform previous works by up to 5.59% and reach
90.71% with five labeled examples. Our results show that MSMatch is capable of
greatly reducing the requirements for labeled data. It translates well to
multispectral data and should enable various applications that are currently
infeasible due to a lack of labeled data. We provide the source code of MSMatch
online to enable easy reproduction and quick adoption.

    

### [[2103.12369] ReCU: Reviving the Dead Weights in Binary Neural Networks](http://arxiv.org/abs/2103.12369)


  Binary neural networks (BNNs) have received increasing attention due to their
superior reductions of computation and memory. Most existing works focus on
either lessening the quantization error by minimizing the gap between the
full-precision weights and their binarization or designing a gradient
approximation to mitigate the gradient mismatch, while leaving the "dead
weights" untouched. This leads to slow convergence when training BNNs. In this
paper, for the first time, we explore the influence of "dead weights" which
refer to a group of weights that are barely updated during the training of
BNNs, and then introduce rectified clamp unit (ReCU) to revive the "dead
weights" for updating. We prove that reviving the "dead weights" by ReCU can
result in a smaller quantization error. Besides, we also take into account the
information entropy of the weights, and then mathematically analyze why the
weight standardization can benefit BNNs. We demonstrate the inherent
contradiction between minimizing the quantization error and maximizing the
information entropy, and then propose an adaptive exponential scheduler to
identify the range of the "dead weights". By considering the "dead weights",
our method offers not only faster BNN training, but also state-of-the-art
performance on CIFAR-10 and ImageNet, compared with recent methods. Code can be
available at this https URL.

    

### [[2103.14158] InversionNet3D: Efficient and Scalable Learning for 3D Full Waveform Inversion](http://arxiv.org/abs/2103.14158)


  Seismic full-waveform inversion (FWI) techniques aim to find a
high-resolution subsurface geophysical model provided with waveform data. Some
recent effort in data-driven FWI has shown some encouraging results in
obtaining 2D velocity maps. However, due to high computational complexity and
large memory consumption, the reconstruction of 3D high-resolution velocity
maps via deep networks is still a great challenge. In this paper, we present
InversionNet3D, an efficient and scalable encoder-decoder network for 3D FWI.
The proposed method employs group convolution in the encoder to establish an
effective hierarchy for learning information from multiple sources while
cutting down unnecessary parameters and operations at the same time. The
introduction of invertible layers further reduces the memory consumption of
intermediate features during training and thus enables the development of
deeper networks with more layers and higher capacity as required by different
application scenarios. Experiments on the 3D Kimberlina dataset demonstrate
that InversionNet3D achieves state-of-the-art reconstruction performance with
lower computational cost and lower memory footprint compared to the baseline.

    

### [[2103.14675] Synthesis of Compositional Animations from Textual Descriptions](http://arxiv.org/abs/2103.14675)


  "How can we animate 3D-characters from a movie script or move robots by
simply telling them what we would like them to do?" "How unstructured and
complex can we make a sentence and still generate plausible movements from it?"
These are questions that need to be answered in the long-run, as the field is
still in its infancy. Inspired by these problems, we present a new technique
for generating compositional actions, which handles complex input sentences.
Our output is a 3D pose sequence depicting the actions in the input sentence.
We propose a hierarchical two-stream sequential model to explore a finer
joint-level mapping between natural language sentences and 3D pose sequences
corresponding to the given motion. We learn two manifold representations of the
motion -- one each for the upper body and the lower body movements. Our model
can generate plausible pose sequences for short sentences describing single
actions as well as long compositional sentences describing multiple sequential
and superimposed actions. We evaluate our proposed model on the publicly
available KIT Motion-Language Dataset containing 3D pose data with
human-annotated sentences. Experimental results show that our model advances
the state-of-the-art on text-based motion synthesis in objective evaluations by
a margin of 50%. Qualitative evaluations based on a user study indicate that
our synthesized motions are perceived to be the closest to the ground-truth
motion captures for both short and compositional sentences.

    

### [[2104.01672] Topological Information Retrieval with Dilation-Invariant Bottleneck Comparative Measures](http://arxiv.org/abs/2104.01672)


  Appropriately representing elements in a database so that queries may be
accurately matched is a central task in information retrieval; recently, this
has been achieved by embedding the graphical structure of the database into a
manifold in a hierarchy-preserving manner using a variety of metrics.
Persistent homology is a tool commonly used in topological data analysis that
is able to rigorously characterize a database in terms of both its hierarchy
and connectivity structure. Computing persistent homology on a variety of
embedded datasets reveals that some commonly used embeddings fail to preserve
the connectivity. We show that those embeddings which successfully retain the
database topology coincide in persistent homology by introducing two
dilation-invariant comparative measures to capture this effect: in particular,
they address the issue of metric distortion on manifolds. We provide an
algorithm for their computation that exhibits greatly reduced time complexity
over existing methods. We use these measures to perform the first instance of
topology-based information retrieval and demonstrate its increased performance
over the standard bottleneck distance for persistent homology. We showcase our
approach on databases of different data varieties including text, videos, and
medical images.

    

### [[2104.05831] Enhancing User' s Income Estimation with Super-App Alternative Data](http://arxiv.org/abs/2104.05831)


  This paper presents the advantages of alternative data from Super-Apps to
enhance user' s income estimation models. It compares the performance of these
alternative data sources with the performance of industry-accepted bureau
income estimators that takes into account only financial system information;
successfully showing that the alternative data manage to capture information
that bureau income estimators do not. By implementing the TreeSHAP method for
Stochastic Gradient Boosting Interpretation, this paper highlights which of the
customer' s behavioral and transactional patterns within a Super-App have a
stronger predictive power when estimating user' s income. Ultimately, this
paper shows the incentive for financial institutions to seek to incorporate
alternative data into constructing their risk profiles.

    

### [[2104.06308] SFE-Net: EEG-based Emotion Recognition with Symmetrical Spatial Feature Extraction](http://arxiv.org/abs/2104.06308)


  Emotion recognition based on EEG (electroencephalography) has been widely
used in human-computer interaction, distance education and health care.
However, the conventional methods ignore the adjacent and symmetrical
characteristics of EEG signals, which also contain salient information related
to emotion. In this paper, a spatial folding ensemble network (SFE-Net) is
presented for EEG feature extraction and emotion recognition. Firstly, for the
undetected area between EEG electrodes, an improved Bicubic-EEG interpolation
algorithm is developed for EEG channels information completion, which allows us
to extract a wider range of adjacent space features. Then, motivated by the
spatial symmetric mechanism of human brain, we fold the input EEG channels data
with five different symmetrical strategies, which enable the proposed network
to extract the information of space features of EEG signals more effectively.
Finally, a 3DCNN-based spatial, temporal extraction, and a multi-voting
strategy of ensemble learning are integrated to model a new neural network.
With this network, the spatial features of different symmetric folding signals
can be extracted simultaneously, which greatly improves the robustness and
accuracy of emotion recognition. The experimental results on DEAP and SEED
datasets show that the proposed algorithm has comparable performance in terms
of recognition accuracy.

    

### [[2104.07963] OpenCSI: An Open-Source Dataset for Indoor Localization Using CSI-Based Fingerprinting](http://arxiv.org/abs/2104.07963)


  Many applications require accurate indoor localization. Fingerprint-based
localization methods propose a solution to this problem, but rely on a radio
map that is effort-intensive to acquire. We automate the radio map acquisition
phase using a software-defined radio (SDR) and a wheeled robot. Furthermore, we
open-source a radio map acquired with our automated tool for a 3GPP Long-Term
Evolution (LTE) wireless link. To the best of our knowledge, this is the first
publicly available radio map containing channel state information (CSI).
Finally, we describe first localization experiments on this radio map using a
convolutional neural network to regress for location coordinates.

    

### [[2104.09943] The principle of weight divergence facilitation for unsupervised pattern recognition in spiking neural networks](http://arxiv.org/abs/2104.09943)


  Parallels between the signal processing tasks and biological neurons lead to
an understanding of the principles of self-organized optimization of input
signal recognition. In the present paper, we discuss such similarities among
biological and technical systems. We propose adding the well-known STDP
synaptic plasticity rule to direct the weight modification towards the state
associated with the maximal difference between background noise and correlated
signals. We use the principle of physically constrained weight growth as a
basis for such weights' modification control. It is proposed that the existence
and production of bio-chemical 'substances' needed for plasticity development
restrict a biological synaptic straight modification. In this paper, the
information about the noise-to-signal ratio controls such a substances'
production and storage and drives the neuron's synaptic pressures towards the
state with the best signal-to-noise ratio. We consider several experiments with
different input signal regimes to understand the functioning of the proposed
approach.

    

### [[2104.13963] Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples](http://arxiv.org/abs/2104.13963)


  This paper proposes a novel method of learning by predicting view assignments
with support samples (PAWS). The method trains a model to minimize a
consistency loss, which ensures that different views of the same unlabeled
instance are assigned similar pseudo-labels. The pseudo-labels are generated
non-parametrically, by comparing the representations of the image views to
those of a set of randomly sampled labeled images. The distance between the
view representations and labeled representations is used to provide a weighting
over class labels, which we interpret as a soft pseudo-label. By
non-parametrically incorporating labeled samples in this way, PAWS extends the
distance-metric loss used in self-supervised methods such as BYOL and SwAV to
the semi-supervised setting. Despite the simplicity of the approach, PAWS
outperforms other semi-supervised methods across architectures, setting a new
state-of-the-art for a ResNet-50 on ImageNet trained with either 10% or 1% of
the labels, reaching 75.5% and 66.5% top-1 respectively. PAWS requires 4x to
12x less training than the previous best methods.

    

### [[2105.00351] Lattice Paths for Persistent Diagrams](http://arxiv.org/abs/2105.00351)


  Persistent homology has undergone significant development in recent years.
However, one outstanding challenge is to build a coherent statistical inference
procedure on persistent diagrams. In this paper, we first present a new lattice
path representation for persistent diagrams. We then develop a new exact
statistical inference procedure for lattice paths via combinatorial
enumerations. The lattice path method is applied to the topological
characterization of the protein structures of the COVID-19 virus. We
demonstrate that there are topological changes during the conformational change
of spike proteins.

    

### [[2105.00594] An End-to-End and Accurate PPG-based Respiratory Rate Estimation Approach Using Cycle Generative Adversarial Networks](http://arxiv.org/abs/2105.00594)


  Respiratory rate (RR) is a clinical sign representing ventilation. An
abnormal change in RR is often the first sign of health deterioration as the
body attempts to maintain oxygen delivery to its tissues. There has been a
growing interest in remotely monitoring of RR in everyday settings which has
made photoplethysmography (PPG) monitoring wearable devices an attractive
choice. PPG signals are useful sources for RR extraction due to the presence of
respiration-induced modulations in them. The existing PPG-based RR estimation
methods mainly rely on hand-crafted rules and manual parameters tuning. An
end-to-end deep learning approach was recently proposed, however, despite its
automatic nature, the performance of this method is not ideal using the real
world data. In this paper, we present an end-to-end and accurate pipeline for
RR estimation using Cycle Generative Adversarial Networks (CycleGAN) to
reconstruct respiratory signals from raw PPG signals. Our results demonstrate a
higher RR estimation accuracy of up to 2$\times$ (mean absolute error of
1.9$\pm$0.3 using five fold cross validation) compared to the state-of-th-art
using a identical publicly available dataset. Our results suggest that CycleGAN
can be a valuable method for RR estimation from raw PPG signals.

    

### [[2105.01480] Neural Weighted A*: Learning Graph Costs and Heuristics with Differentiable Anytime A*](http://arxiv.org/abs/2105.01480)


  Recently, the trend of incorporating differentiable algorithms into deep
learning architectures arose in machine learning research, as the fusion of
neural layers and algorithmic layers has been beneficial for handling
combinatorial data, such as shortest paths on graphs. Recent works related to
data-driven planning aim at learning either cost functions or heuristic
functions, but not both. We propose Neural Weighted A*, a differentiable
anytime planner able to produce improved representations of planar maps as
graph costs and heuristics. Training occurs end-to-end on raw images with
direct supervision on planning examples, thanks to a differentiable A* solver
integrated into the architecture. More importantly, the user can trade off
planning accuracy for efficiency at run-time, using a single, real-valued
parameter. The solution suboptimality is constrained within a linear bound
equal to the optimal path cost multiplied by the tradeoff parameter. We
experimentally show the validity of our claims by testing Neural Weighted A*
against several baselines, introducing a novel, tile-based navigation dataset.
We outperform similar architectures in planning accuracy and efficiency.

    

### [[2105.01688] Height Estimation of Children under Five Years using Depth Images](http://arxiv.org/abs/2105.01688)


  Malnutrition is a global health crisis and is the leading cause of death
among children under five. Detecting malnutrition requires anthropometric
measurements of weight, height, and middle-upper arm circumference. However,
measuring them accurately is a challenge, especially in the global south, due
to limited resources. In this work, we propose a CNN-based approach to estimate
the height of standing children under five years from depth images collected
using a smart-phone. According to the SMART Methodology Manual [5], the
acceptable accuracy for height is less than 1.4 cm. On training our deep
learning model on 87131 depth images, our model achieved an average mean
absolute error of 1.64% on 57064 test images. For 70.3% test images, we
estimated height accurately within the acceptable 1.4 cm range. Thus, our
proposed solution can accurately detect stunting (low height-for-age) in
standing children below five years of age.

    

### [[2105.05842] Kernel Thinning](http://arxiv.org/abs/2105.05842)


  We introduce kernel thinning, a new procedure for compressing a distribution
$\mathbb{P}$ more effectively than i.i.d. sampling or standard thinning. Given
a suitable reproducing kernel $\mathbf{k}$ and $\mathcal{O}(n^2)$ time, kernel
thinning compresses an $n$-point approximation to $\mathbb{P}$ into a
$\sqrt{n}$-point approximation with comparable worst-case integration error
across the associated reproducing kernel Hilbert space. With high probability,
the maximum discrepancy in integration error is
$\mathcal{O}_d(n^{-\frac{1}{2}}\sqrt{\log n})$ for compactly supported
$\mathbb{P}$ and $\mathcal{O}_d(n^{-\frac{1}{2}} \sqrt{(\log n)^{d+1}\log\log
n})$ for sub-exponential $\mathbb{P}$ on $\mathbb{R}^d$. In contrast, an
equal-sized i.i.d. sample from $\mathbb{P}$ suffers $\Omega(n^{-\frac14})$
integration error. Our sub-exponential guarantees resemble the classical
quasi-Monte Carlo error rates for uniform $\mathbb{P}$ on $[0,1]^d$ but apply
to general distributions on $\mathbb{R}^d$ and a wide range of common kernels.
We use our results to derive explicit non-asymptotic maximum mean discrepancy
bounds for Gaussian, Matérn, and B-spline kernels and present two vignettes
illustrating the practical benefits of kernel thinning over i.i.d. sampling and
standard Markov chain Monte Carlo thinning, in dimensions $d=2$ through $100$.

    

### [[2105.06256] Federated Learning with Unreliable Clients: Performance Analysis and Mechanism Design](http://arxiv.org/abs/2105.06256)


  Owing to the low communication costs and privacy-promoting capabilities,
Federated Learning (FL) has become a promising tool for training effective
machine learning models among distributed clients. However, with the
distributed architecture, low quality models could be uploaded to the
aggregator server by unreliable clients, leading to a degradation or even a
collapse of training. In this paper, we model these unreliable behaviors of
clients and propose a defensive mechanism to mitigate such a security risk.
Specifically, we first investigate the impact on the models caused by
unreliable clients by deriving a convergence upper bound on the loss function
based on the gradient descent updates. Our theoretical bounds reveal that with
a fixed amount of total computational resources, there exists an optimal number
of local training iterations in terms of convergence performance. We further
design a novel defensive mechanism, named deep neural network based secure
aggregation (DeepSA). Our experimental results validate our theoretical
analysis. In addition, the effectiveness of DeepSA is verified by comparing
with other state-of-the-art defensive mechanisms.

    

### [[2105.06295] Gait Characterization in Duchenne Muscular Dystrophy (DMD) Using a Single-Sensor Accelerometer: Classical Machine Learning and Deep Learning Approaches](http://arxiv.org/abs/2105.06295)


  Differences in gait patterns of children with Duchenne muscular dystrophy
(DMD) and typically developing (TD) peers are visible to the eye, but
quantification of those differences outside of the gait laboratory has been
elusive. We measured vertical, mediolateral, and anteroposterior acceleration
using a waist-worn iPhone accelerometer during ambulation across a typical
range of velocities. Six TD and six DMD children from 3-15 years of age
underwent seven walking/running tasks, including five 25m walk/run tests at a
slow walk to running speeds, a 6-minute walk test (6MWT), and a
100-meter-run/walk (100MRW). We extracted temporospatial clinical gait features
(CFs) and applied multiple Artificial Intelligence (AI) tools to differentiate
between DMD and TD control children using extracted features and raw data.
Extracted CFs showed reduced step length and a greater mediolateral component
of total power (TP) consistent with shorter strides and Trendelenberg-like gait
commonly observed in DMD. AI methods using CFs and raw data varied
ineffectiveness at differentiating between DMD and TD controls at different
speeds, with an accuracy of some methods exceeding 91%. We demonstrate that by
using AI tools with accelerometer data from a consumer-level smartphone, we can
identify DMD gait disturbance in toddlers to early teens.

    

### [[2106.01382] Undecidability of Learnability](http://arxiv.org/abs/2106.01382)


  Machine learning researchers and practitioners steadily enlarge the multitude
of successful learning models. They achieve this through in-depth theoretical
analyses and experiential heuristics. However, there is no known
general-purpose procedure for rigorously evaluating whether newly proposed
models indeed successfully learn from data. We show that such a procedure
cannot exist. For PAC binary classification, uniform and universal online
learning, and exact learning through teacher-learner interactions, learnability
is in general undecidable, both in the sense of independence of the axioms in a
formal system and in the sense of uncomputability. Our proofs proceed via
computable constructions of function classes that encode the consistency
problem for formal systems and the halting problem for Turing machines into
complexity measures that characterize learnability. Our work shows that
undecidability appears in the theoretical foundations of machine learning:
There is no one-size-fits-all algorithm for deciding whether a machine learning
model can be successful. We cannot in general automatize the process of
assessing new learning models.

    

### [[2106.01969] Global Convergence of Multi-Agent Policy Gradient in Markov Potential Games](http://arxiv.org/abs/2106.01969)


  Potential games are arguably one of the most important and widely studied
classes of normal form games. They define the archetypal setting of multi-agent
coordination as all agent utilities are perfectly aligned with each other via a
common potential function. Can this intuitive framework be transplanted in the
setting of Markov Games? What are the similarities and differences between
multi-agent coordination with and without state dependence? We present a novel
definition of Markov Potential Games (MPG) that generalizes prior attempts at
capturing complex stateful multi-agent coordination. Counter-intuitively,
insights from normal-form potential games do not carry over as MPGs can consist
of settings where state-games can be zero-sum games. In the opposite direction,
Markov games where every state-game is a potential game are not necessarily
MPGs. Nevertheless, MPGs showcase standard desirable properties such as the
existence of deterministic Nash policies. In our main technical result, we
prove fast convergence of independent policy gradient to Nash policies by
adapting recent gradient dominance property arguments developed for single
agent MDPs to multi-agent learning settings.

    

### [[2106.08687] Leveraging Probabilistic Circuits for Nonparametric Multi-Output Regression](http://arxiv.org/abs/2106.08687)


  Inspired by recent advances in the field of expert-based approximations of
Gaussian processes (GPs), we present an expert-based approach to large-scale
multi-output regression using single-output GP experts. Employing a deeply
structured mixture of single-output GPs encoded via a probabilistic circuit
allows us to capture correlations between multiple output dimensions
accurately. By recursively partitioning the covariate space and the output
space, posterior inference in our model reduces to inference on single-output
GP experts, which only need to be conditioned on a small subset of the
observations. We show that inference can be performed exactly and efficiently
in our model, that it can capture correlations between output dimensions and,
hence, often outperforms approaches that do not incorporate inter-output
correlations, as demonstrated on several data sets in terms of the negative log
predictive density.

    

### [[2106.08774] Analysis and Optimisation of Bellman Residual Errors with Neural Function Approximation](http://arxiv.org/abs/2106.08774)


  Recent development of Deep Reinforcement Learning has demonstrated superior
performance of neural networks in solving challenging problems with large or
even continuous state spaces. One specific approach is to deploy neural
networks to approximate value functions by minimising the Mean Squared Bellman
Error function. Despite great successes of Deep Reinforcement Learning,
development of reliable and efficient numerical algorithms to minimise the
Bellman Error is still of great scientific interest and practical demand. Such
a challenge is partially due to the underlying optimisation problem being
highly non-convex or using incorrect gradient information as done in
Semi-Gradient algorithms. In this work, we analyse the Mean Squared Bellman
Error from a smooth optimisation perspective combined with a Residual Gradient
formulation. Our contribution is two-fold.
First, we analyse critical points of the error function and provide technical
insights on the optimisation procure and design choices for neural networks.
When the existence of global minima is assumed and the objective fulfils
certain conditions we can eliminate suboptimal local minima when using
over-parametrised neural networks. We can construct an efficient Approximate
Newton's algorithm based on our analysis and confirm theoretical properties of
this algorithm such as being locally quadratically convergent to a global
minimum numerically.
Second, we demonstrate feasibility and generalisation capabilities of the
proposed algorithm empirically using continuous control problems and provide a
numerical verification of our critical point analysis. We outline the short
coming of Semi-Gradients. To benefit from an approximate Newton's algorithm
complete derivatives of the Mean Squared Bellman error must be considered
during training.

    

### [[2106.08961] Intelligent-Tire-Based Slip Ratio Estimation Using Machine Learning](http://arxiv.org/abs/2106.08961)


  Autonomous vehicles are most concerned about safety control issues, and the
slip ratio is critical to the safety of the vehicle control system. In this
paper, different machine learning algorithms (Neural Networks, Gradient
Boosting Machine, Random Forest, and Support Vector Machine) are used to train
the slip ratio estimation model based on the acceleration signals ($a_x$,
$a_y$, and $a_z$) from the tri-axial Micro-Electro Mechanical System (MEMS)
accelerometer utilized in the intelligent tire system, where the acceleration
signals are divided into four sets ($a_x/a_y/a_z$, $a_x/a_z$, $a_y/a_z$, and
$a_z$) as algorithm inputs. The experimental data used in this study are
collected through the MTS Flat-Trac tire test platform. Performance of
different slip ratio estimation models is compared using the NRMS errors in
10-fold cross-validation (CV). The results indicate that NN and GBM have more
promising accuracy, and the $a_z$ input type has a better performance compared
to other input types, with the best result being the estimation model of the NN
algorithm with $a_z$ as input, which results is 4.88\%. The present study with
the fusion of intelligent tire system and machine learning paves the way for
the accurate estimation of tire slip ratio under different driving conditions,
which will open up a new way of Autonomous vehicles, intelligent tires, and
tire slip ratio estimation.

    

### [[2106.08977] Named Entity Recognition with Small Strongly Labeled and Large Weakly Labeled Data](http://arxiv.org/abs/2106.08977)


  Weak supervision has shown promising results in many natural language
processing tasks, such as Named Entity Recognition (NER). Existing work mainly
focuses on learning deep NER models only with weak supervision, i.e., without
any human annotation, and shows that by merely using weakly labeled data, one
can achieve good performance, though still underperforms fully supervised NER
with manually/strongly labeled data. In this paper, we consider a more
practical scenario, where we have both a small amount of strongly labeled data
and a large amount of weakly labeled data. Unfortunately, we observe that
weakly labeled data does not necessarily improve, or even deteriorate the model
performance (due to the extensive noise in the weak labels) when we train deep
NER models over a simple or weighted combination of the strongly labeled and
weakly labeled data. To address this issue, we propose a new multi-stage
computational framework -- NEEDLE with three essential ingredients: (1) weak
label completion, (2) noise-aware loss function, and (3) final fine-tuning over
the strongly labeled data. Through experiments on E-commerce query NER and
Biomedical NER, we demonstrate that NEEDLE can effectively suppress the noise
of the weak labels and outperforms existing methods. In particular, we achieve
new SOTA F1-scores on 3 Biomedical NER datasets: BC5CDR-chem 93.74,
BC5CDR-disease 90.69, NCBI-disease 92.28.

    

### [[2106.09981] How COVID-19 Has Changed Crowdfunding: Evidence From GoFundMe](http://arxiv.org/abs/2106.09981)


  While the long-term effects of COVID-19 are yet to be determined, its
immediate impact on crowdfunding is nonetheless significant. This study takes a
computational approach to more deeply comprehend this change. Using a unique
data set of all the campaigns published over the past two years on GoFundMe, we
explore the factors that have led to the successful funding of a crowdfunding
project. In particular, we study a corpus of crowdfunded projects, analyzing
cover images and other variables commonly present on crowdfunding sites.
Furthermore, we construct a classifier and a regression model to assess the
significance of features based on XGBoost. In addition, we employ
counterfactual analysis to investigate the causality between features and the
success of crowdfunding. More importantly, sentiment analysis and the paired
sample t-test are performed to examine the differences in crowdfunding
campaigns before and after the COVID-19 outbreak that started in March 2020.
First, we note that there is significant racial disparity in crowdfunding
success. Second, we find that sad emotion expressed through the campaign's
description became significant after the COVID-19 outbreak. Considering all
these factors, our findings shed light on the impact of COVID-19 on
crowdfunding campaigns.

    

### [[2106.10717] Strategies for convex potential games and an application to decision-theoretic online learning](http://arxiv.org/abs/2106.10717)


  The backwards induction method due to Bellman~\cite{bellman1952theory} is a
popular approach to solving problems in optimiztion, optimal control, and many
other areas of applied math. In this paper we analyze the backwords induction
approach, under min/max conditions. We show that if the value function is has
strictly positive derivatives of order 1-4 then the optimal strategy for the
adversary is Brownian motion. Using that fact we analyze different potential
functions and show that the Normal-Hedge potential is optimal.

    

### [[2106.07341] i-Pulse: A NLP based novel approach for employee engagement in logistics organization](http://arxiv.org/abs/2106.07341)


  Although most logistics and freight forwarding organizations, in one way or
another, claim to have core values. The engagement of employees is a vast
structure that affects almost every part of the company's core environmental
values. There is little theoretical knowledge about the relationship between
firms and the engagement of employees. Based on research literature, this paper
aims to provide a novel approach for insight around employee engagement in a
logistics organization by implementing deep natural language processing
concepts. The artificial intelligence-enabled solution named Intelligent Pulse
(I-Pulse) can evaluate hundreds and thousands of pulse survey comments and
provides the actionable insights and gist of employee feedback. I-Pulse allows
the stakeholders to think in new ways in their organization, helping them to
have a powerful influence on employee engagement, retention, and efficiency.
This study is of corresponding interest to researchers and practitioners.

    

### [[2108.00147] Communication-avoiding micro-architecture to compute Xcorr scores for peptide identification](http://arxiv.org/abs/2108.00147)


  Database algorithms play a crucial part in systems biology studies by
identifying proteins from mass spectrometry data. Many of these database search
algorithms incur huge computational costs by computing similarity scores for
each pair of sparse experimental spectrum and candidate theoretical spectrum
vectors. Modern MS instrumentation techniques which are capable of generating
high-resolution spectrometry data require comparison against an enormous search
space, further emphasizing the need of efficient accelerators. Recent research
has shown that the overall cost of scoring, and deducing peptides is dominated
by the communication costs between different hierarchies of memory and
processing units. However, these communication costs are seldom considered in
accelerator-based architectures leading to inefficient DRAM accesses, and poor
data-utilization due to irregular memory access patterns. In this paper, we
propose a novel communication-avoiding micro-architecture to compute
cross-correlation based similarity score by utilizing efficient local cache,
and peptide pre-fetching to minimize DRAM accesses, and a custom-designed
peptide broadcast bus to allow input reuse. An efficient bus arbitration scheme
was designed, and implemented to minimize synchronization cost and exploit
parallelism of processing elements. Our simulation results show that the
proposed micro-architecture performs on average 24x better than a CPU
implementation running on a 3.6 GHz Intel i7-4970 processor with 16GB memory.

    

### [[2108.00444] An efficient reverse-lookup table based strategy for solving the synonym and cache coherence problem in virtually indexed, virtually tagged caches](http://arxiv.org/abs/2108.00444)


  Virtually indexed and virtually tagged (VIVT) caches are an attractive option
for micro-processor level-1 caches, because of their fast response time and
because they are cheaper to implement than more complex caches such as
virtually-indexed physical-tagged (VIPT) caches. The level-1 VIVT cache becomes
even simpler to construct if it is implemented as a direct-mapped cache
(VIVT-DM cache). However, VIVT and VIVT-DM caches have some drawbacks. When the
number of sets in the cache is larger than the smallest page size, there is a
possibility of synonyms (two or more virtual addresses mapped to the same
physical address) existing in the cache. Further, maintenance of cache
coherence across multiple processors requires a physical to virtual translation
mechanism in the hardware. We describe a simple, efficient reverse lookup table
based approach to address the synonym and the coherence problems in VIVT (both
set associative and direct-mapped) caches. In particular, the proposed scheme
does not disturb the critical memory access paths in a typical micro-processor,
and requires a low overhead for its implementation. We have implemented and
validated the scheme in the AJIT 32-bit microprocessor core (an implementation
of the SPARC-V8 ISA) and the implementation uses approximately 2% of the gates
and 5.3% of the memory bits in the processor core.

    

### [[2108.00778] Analysing digital in-memory computing for advanced finFET node](http://arxiv.org/abs/2108.00778)


  Digital In-memory computing improves energy efficiency and throughput of a
data-intensive process, which incur memory thrashing and, resulting multiple
same memory accesses in a von Neumann architecture. Digital in-memory computing
involves accessing multiple SRAM cells simultaneously, which may result in a
bit flip when not timed critically. Therefore we discuss the transient voltage
characteristics of the bitlines during an SRAM compute. To improve the
packaging density and also avoid MOSFET down-scaling issues, we use a 7-nm
predictive PDK which uses a finFET node. The finFET process has discrete fins
and a lower Voltage supply, which makes the design of in-memory compute SRAM
difficult. In this paper, we design a 6T SRAM cell in 7-nm finFET node and
compare its SNMs with a UMC 28nm node implementation. Further, we design and
simulate the rest of the SRAM peripherals, and in-memory computation for an
advanced finFET node.

    

### [[2108.00808] Energy Efficiency Aspects of the AMD Zen 2 Architecture](http://arxiv.org/abs/2108.00808)


  In High Performance Computing, systems are evaluated based on their
computational throughput. However, performance in contemporary server
processors is primarily limited by power and thermal constraints. Ensuring
operation within a given power envelope requires a wide range of sophisticated
control mechanisms. While some of these are handled transparently by hardware
control loops, others are controlled by the operating system. A lack of
publicly disclosed implementation details further complicates this topic.
However, understanding these mechanisms is a prerequisite for any effort to
exploit the full computing capability and to minimize the energy consumption of
today's server systems. This paper highlights the various energy efficiency
aspects of the AMD Zen 2 microarchitecture to facilitate system understanding
and optimization. Key findings include qualitative and quantitative
descriptions regarding core frequency transition delays, workload-based
frequency limitations, effects of I/O die P-states on memory performance as
well as discussion on the built-in power monitoring capabilities and its
limitations. Moreover, we present specifics and caveats of idle states, wakeup
times as well as the impact of idling and inactive hardware threads and cores
on the performance of active resources such as other cores.

    

### [[2108.00026] Private Retrieval, Computing and Learning: Recent Progress and Future Challenges](http://arxiv.org/abs/2108.00026)


  Most of our lives are conducted in the cyberspace. The human notion of
privacy translates into a cyber notion of privacy on many functions that take
place in the cyberspace. This article focuses on three such functions: how to
privately retrieve information from cyberspace (privacy in information
retrieval), how to privately leverage large-scale distributed/parallel
processing (privacy in distributed computing), and how to learn/train machine
learning models from private data spread across multiple users (privacy in
distributed (federated) learning). The article motivates each privacy setting,
describes the problem formulation, summarizes breakthrough results in the
history of each problem, and gives recent results and discusses some of the
major ideas that emerged in each field. In addition, the cross-cutting
techniques and interconnections between the three topics are discussed along
with a set of open problems and challenges.

    

### [[2108.00059] Local certification of graph decompositions and applications to minor-free classes](http://arxiv.org/abs/2108.00059)


  Local certification consists in assigning labels to the nodes of a network to
certify that some given property is satisfied, in such a way that the labels
can be checked locally. In the last few years, certification of graph classes
received a considerable attention. The goal is to certify that a graph $G$
belongs to a given graph class~$\mathcal{G}$. Such certifications with labels
of size $O(\log n)$ (where $n$ is the size of the network) exist for trees,
planar graphs and graphs embedded on surfaces. Feuilloley et al. ask if this
can be extended to any class of graphs defined by a finite set of forbidden
minors. In this work, we develop new decomposition tools for graph
certification, and apply them to show that for every small enough minor $H$,
$H$-minor-free graphs can indeed be certified with labels of size $O(\log n)$.
We also show matching lower bounds with a simple new proof technique.

    

### [[2108.00485] Webots.HPC: A Parallel Robotics Simulation Pipeline for Autonomous Vehicles on High Performance Computing](http://arxiv.org/abs/2108.00485)


  In the rapidly evolving and maturing field of robotics, computer simulation
has become an invaluable tool in the design process. Webots, a state-of-the-art
robotics simulator, is often the software of choice for robotics research. Even
so, Webots simulations are often run on personal and lab computers. For
projects that would benefit from an aggregated output dataset from thousands of
simulation runs, there is no standard recourse; this project sets out to
mitigate this by developing a formalized parallel pipeline for running
sequences of Webots simulations on powerful HPC resources. Such a pipeline
would allow researchers to generate massive datasets from their simulations,
opening the door for potential machine learning applications and decision tool
development. We have developed a pipeline capable of running Webots simulations
both headlessly and in GUI-enabled mode over an SSH X11 server, with simulation
execution occurring remotely on HPC compute nodes. Additionally, simulations
can be run in sequence, with a batch job being distributed across an arbitrary
number of computing nodes and each node having multiple instances running in
parallel. The implemented distribution and parallelization are extremely
effective, with a 100\% simulation completion rate after 12 hours of runs.
Overall, this pipeline is very capable and can be used to extend existing
projects or serve as a platform for new robotics simulation endeavors.

    

### [[2108.00529] BigGraphVis: Leveraging Streaming Algorithms and GPU Acceleration for Visualizing Big Graphs](http://arxiv.org/abs/2108.00529)


  Graph layouts are key to exploring massive graphs. An enormous number of
nodes and edges do not allow network analysis software to produce meaningful
visualization of the pervasive networks. Long computation time, memory and
display limitations encircle the software's ability to explore massive graphs.
This paper introduces BigGraphVis, a new parallel graph visualization method
that uses GPU parallel processing and community detection algorithm to
visualize graph communities. We combine parallelized streaming community
detection algorithm and probabilistic data structure to leverage parallel
processing of Graphics Processing Unit (GPU). To the best of our knowledge,
this is the first attempt to combine the power of streaming algorithms coupled
with GPU computing to tackle big graph visualization challenges. Our method
extracts community information in a few passes on the edge list, and renders
the community structures using the ForceAtlas2 algorithm. Our experiment with
massive real-life graphs indicates that about 70 to 95 percent speedup can be
achieved by visualizing graph communities, and the visualization appears to be
meaningful and reliable. The biggest graph that we examined contains above 3
million nodes and 34 million edges, and the layout computation took about five
minutes. We also observed that the BigGraphVis coloring strategy can be
successfully applied to produce a more informative ForceAtlas2 layout.

    

### [[2108.00554] Experimental Findings on the Sources of Detected Unrecoverable Errors in GPUs](http://arxiv.org/abs/2108.00554)


  We investigate the sources of Detected Unrecoverable Errors (DUEs) in GPUs
exposed to neutron beams. Illegal memory accesses and interface errors are
among the more likely sources of DUEs. ECC increases the launch failure events.
Our test procedure has shown that ECC can reduce the DUEs caused by Illegal
Address access up to 92% for Kepler and 98% for Volta.

    

### [[2108.00730] YASMIN: a Real-time Middleware for COTS Heterogeneous Platforms](http://arxiv.org/abs/2108.00730)


  Commercial-Off-The-Shelf heterogeneous platforms provide immense
computational power, but are difficult to program and to correctly use when
real-time requirements come into play: A sound configuration of the operating
system scheduler is needed, and a suitable mapping of tasks to computing units
must be determined. Flawed designs may lead a sub-optimal system configurations
and thus to wasted resources, or even to deadline misses and failures. We
propose YASMIN, a middleware to schedule end-user applications with real-time
requirements in user space and on behalf of the operating system. YASMIN
provides an easy-to-use programming interface and portability. It treats
heterogeneity on COTS heterogeneous embedded platforms as a first-class
citizen: It supports multiple functionally equivalent task implementations with
distinct extra-functional behaviour. This enables the system designer to
quickly explore different scheduling policies and task-to-core mappings, and
thus, to improve overall system performance. In this paper, we present the
design and implementation of YASMIN and provide an analysis of the scheduling
overhead on an Odroid-XU4 platform. Last but not least, we demonstrate the
merits of YASMIN on an industrial use-case involving a Search & Rescue drone.

    

### [[2004.06436] Broadcast CONGEST Algorithms against Adversarial Edges](http://arxiv.org/abs/2004.06436)


  We consider the corner-stone broadcast task with an adaptive adversary that
controls a fixed number of $t$ edges in the input communication graph. In this
model, the adversary sees the entire communication in the network and the
random coins of the nodes, while maliciously manipulating the messages sent
through a set of $t$ edges (unknown to the nodes). Since the influential work
of [Pease, Shostak and Lamport, JACM'80], broadcast algorithms against
plentiful adversarial models have been studied in both theory and practice for
over more than four decades. Despite this extensive research, there is no round
efficient broadcast algorithm for general graphs in the CONGEST model of
distributed computing. We provide the first round-efficient broadcast
algorithms against adaptive edge adversaries. Our two key results for $n$-node
graphs of diameter $D$ are as follows:
1. For $t=1$, there is a deterministic algorithm that solves the problem
within $\widetilde{O}(D^2)$ rounds, provided that the graph is 3
edge-connected. This round complexity beats the natural barrier of $O(D^3)$
rounds, the existential lower bound on the maximal length of $3$ edge-disjoint
paths between a given pair of nodes in $G$. This algorithm can be extended to a
$\widetilde{O}(D^{O(t)})$-round algorithm against $t$ adversarial edges in
$(2t+1)$ edge-connected graphs.
2. For expander graphs with edge connectivity of $\Omega(t^2\log n)$, there
is an improved broadcast algorithm with $O(t \log ^2 n)$ rounds against $t$
adversarial edges. This algorithm exploits the connectivity and conductance
properties of G-subgraphs obtained by employing the Karger's edge sampling
technique.
Our algorithms mark a new connection between the areas of fault-tolerant
network design and reliable distributed communication.

    

### [[2103.04916] Transparent Checkpointing for OpenGL Applications on GPUs](http://arxiv.org/abs/2103.04916)


  This work presents transparent checkpointing of OpenGL applications, refining
the split-process technique[1] for application in GPU-based 3D graphics. The
split-process technique was earlier applied to checkpointing MPI and CUDA
programs, enabling reinitialization of driver libraries.
The presented design targets practical, checkpoint-package agnostic
checkpointing of OpenGL applications. An early prototype is demonstrated on
Autodesk Maya. Maya is a complex proprietary media-creation software suite used
with large-scale rendering hardware for CGI (Computer-Generated Animation).
Transparent checkpointing of Maya provides critically-needed fault tolerance,
since Maya is prone to crash when artists use some of its bleeding-edge
components. Artists then lose hours of work in re-creating their complex
environment.

    

### [[2105.12912] CuSZ+: Optimizing Error-Bounded Lossy Compression for Scientific Data on GPUs](http://arxiv.org/abs/2105.12912)


  Error-bounded lossy compression is a critical technique for significantly
reducing scientific data volumes. With ever-emerging heterogeneous
high-performance computing (HPC) architecture, GPU-accelerated error-bounded
compressors (such as cuSZ+ and cuZFP) have been developed. However, they suffer
from either low performance or low compression ratios. To this end, we propose
cuSZ+ to target both high compression ratios and throughputs. We identify that
data sparsity and data smoothness are key factors for high compression
throughputs. Our key contributions in this work are fourfold: (1) We propose an
efficient compression workflow to adaptively perform run-length encoding and/or
variable-length encoding. (2) We derive Lorenzo reconstruction in decompression
as multidimensional partial-sum computation and propose a fine-grained Lorenzo
reconstruction algorithm for GPU architectures. (3) We carefully optimize each
of cuSZ+ kernels by leveraging state-of-the-art CUDA parallel primitives. (4)
We evaluate cuSZ+ using seven real-world HPC application datasets on V100 and
A100 GPUs. Experiments show cuSZ+ improves the compression throughputs and
ratios by up to 18.4X and 5.3X, respectively, over cuSZ on the tested datasets.

    

### [[2107.01142] 4C: A Computation, Communication, and Control Co-Design Framework for CAVs](http://arxiv.org/abs/2107.01142)


  Connected and autonomous vehicles (CAVs) are promising due to their potential
safety and efficiency benefits and have attracted massive investment and
interest from government agencies, industry, and academia. With more computing
and communication resources are available, both vehicles and edge servers are
equipped with a set of camera-based vision sensors, also known as Visual IoT
(V-IoT) techniques, for sensing and perception. Tremendous efforts have been
made for achieving programmable communication, computation, and control.
However, they are conducted mainly in the silo mode, limiting the
responsiveness and efficiency of handling challenging scenarios in the real
world. To improve the end-to-end performance, we envision that future CAVs
require the co-design of communication, computation, and control. This paper
presents our vision of the end-to-end design principle for CAVs, called 4C,
which extends the V-IoT system by providing a unified communication,
computation, and control co-design framework. With programmable communications,
fine-grained heterogeneous computation, and efficient vehicle controls in 4C,
CAVs can handle critical scenarios and achieve energy-efficient autonomous
driving. Finally, we present several challenges to achieving the vision of the
4C framework.

    

### [[2108.00003] Secure solutions for Smart City Command Control Centre using AIOT](http://arxiv.org/abs/2108.00003)


  To build a robust secure solution for smart city IOT network from any Cyber
attacks using Artificial Intelligence. In Smart City IOT network, data
collected from different log collectors or direct sources from cloud or edge
should harness the potential of AI. The smart city command and control center
team will leverage these models and deploy it in different city IOT network to
help on intrusion prediction, network packet surge, potential botnet attacks
from external network. Some of the vital use cases considered based on the
users of command-and-control center

    

### [[2108.00056] Procedural Generation of 3D Maps with Snappable Meshes](http://arxiv.org/abs/2108.00056)


  In this paper we present a technique for procedurally generating 3D maps
using a set of premade meshes which snap together based on designer-specified
visual constraints. The proposed approach avoids size and layout limitations,
offering the designer control over the look and feel of the generated maps, as
well as immediate feedback on a given map's navigability. A prototype
implementation of the method, developed in the Unity game engine, is discussed,
and a number of case studies are analyzed. These include a multiplayer game
where the method was used, together with a number of illustrative examples
which highlight various parameterizations and generation methods. We argue that
the technique is designer-friendly and can be used as a map composition method
and/or as a prototyping system in 3D level design, opening the door for quality
map and level creation in a fraction of the time of a fully human-based
approach.

    

### [[2108.00057] WLV-RIT at GermEval 2021: Multitask Learning with Transformers to Detect Toxic, Engaging, and Fact-Claiming Comments](http://arxiv.org/abs/2108.00057)


  This paper addresses the identification of toxic, engaging, and fact-claiming
comments on social media. We used the dataset made available by the organizers
of the GermEval-2021 shared task containing over 3,000 manually annotated
Facebook comments in German. Considering the relatedness of the three tasks, we
approached the problem using large pre-trained transformer models and multitask
learning. Our results indicate that multitask learning achieves performance
superior to the more common single task learning approach in all three tasks.
We submit our best systems to GermEval-2021 under the team name WLV-RIT.

    

### [[2108.00061] MTVR: Multilingual Moment Retrieval in Videos](http://arxiv.org/abs/2108.00061)


  We introduce mTVR, a large-scale multilingual video moment retrieval dataset,
containing 218K English and Chinese queries from 21.8K TV show video clips. The
dataset is collected by extending the popular TVR dataset (in English) with
paired Chinese queries and subtitles. Compared to existing moment retrieval
datasets, mTVR is multilingual, larger, and comes with diverse annotations. We
further propose mXML, a multilingual moment retrieval model that learns and
operates on data from both languages, via encoder parameter sharing and
language neighborhood constraints. We demonstrate the effectiveness of mXML on
the newly collected MTVR dataset, where mXML outperforms strong monolingual
baselines while using fewer parameters. In addition, we also provide detailed
dataset analyses and model ablations. Data and code are publicly available at
this https URL


### [[2108.00082] Towards Continual Entity Learning in Language Models for Conversational Agents](http://arxiv.org/abs/2108.00082)


  Neural language models (LM) trained on diverse corpora are known to work well
on previously seen entities, however, updating these models with dynamically
changing entities such as place names, song titles and shopping items requires
re-training from scratch and collecting full sentences containing these
entities. We aim to address this issue, by introducing entity-aware language
models (EALM), where we integrate entity models trained on catalogues of
entities into the pre-trained LMs. Our combined language model adaptively adds
information from the entity models into the pre-trained LM depending on the
sentence context. Our entity models can be updated independently of the
pre-trained LM, enabling us to influence the distribution of entities output by
the final LM, without any further training of the pre-trained LM. We show
significant perplexity improvements on task-oriented dialogue datasets,
especially on long-tailed utterances, with an ability to continually adapt to
new entities (to an extent).

    

### [[2108.00114] On The State of Data In Computer Vision: Human Annotations Remain Indispensable for Developing Deep Learning Models](http://arxiv.org/abs/2108.00114)


  High-quality labeled datasets play a crucial role in fueling the development
of machine learning (ML), and in particular the development of deep learning
(DL). However, since the emergence of the ImageNet dataset and the AlexNet
model in 2012, the size of new open-source labeled vision datasets has remained
roughly constant. Consequently, only a minority of publications in the computer
vision community tackle supervised learning on datasets that are orders of
magnitude larger than Imagenet. In this paper, we survey computer vision
research domains that study the effects of such large datasets on model
performance across different vision tasks. We summarize the community's current
understanding of those effects, and highlight some open questions related to
training with massive datasets. In particular, we tackle: (a) The largest
datasets currently used in computer vision research and the interesting
takeaways from training on such datasets; (b) The effectiveness of pre-training
on large datasets; (c) Recent advancements and hurdles facing synthetic
datasets; (d) An overview of double descent and sample non-monotonicity
phenomena; and finally, (e) A brief discussion of lifelong/continual learning
and how it fares compared to learning from huge labeled datasets in an offline
setting. Overall, our findings are that research on optimization for deep
learning focuses on perfecting the training routine and thus making DL models
less data hungry, while research on synthetic datasets aims to offset the cost
of data labeling. However, for the time being, acquiring non-synthetic labeled
data remains indispensable to boost performance.

    

### [[2108.00159] Learning Embeddings that Capture Spatial Semantics for Indoor Navigation](http://arxiv.org/abs/2108.00159)


  Incorporating domain-specific priors in search and navigation tasks has shown
promising results in improving generalization and sample complexity over
end-to-end trained policies. In this work, we study how object embeddings that
capture spatial semantic priors can guide search and navigation tasks in a
structured environment. We know that humans can search for an object like a
book, or a plate in an unseen house, based on the spatial semantics of bigger
objects detected. For example, a book is likely to be on a bookshelf or a
table, whereas a plate is likely to be in a cupboard or dishwasher. We propose
a method to incorporate such spatial semantic awareness in robots by leveraging
pre-trained language models and multi-relational knowledge bases as object
embeddings. We demonstrate using these object embeddings to search a query
object in an unseen indoor environment. We measure the performance of these
embeddings in an indoor simulator (AI2Thor). We further evaluate different
pre-trained embedding onSuccess Rate(SR) and success weighted by Path
Length(SPL).

    

### [[2108.00177] Greedy Network Enlarging](http://arxiv.org/abs/2108.00177)


  Recent studies on deep convolutional neural networks present a simple
paradigm of architecture design, i.e., models with more MACs typically achieve
better accuracy, such as EfficientNet and RegNet. These works try to enlarge
all the stages in the model with one unified rule by sampling and statistical
methods. However, we observe that some network architectures have similar MACs
and accuracies, but their allocations on computations for different stages are
quite different. In this paper, we propose to enlarge the capacity of CNN
models by improving their width, depth and resolution on stage level. Under the
assumption that the top-performing smaller CNNs are a proper subcomponent of
the top-performing larger CNNs, we propose an greedy network enlarging method
based on the reallocation of computations. With step-by-step modifying the
computations on different stages, the enlarged network will be equipped with
optimal allocation and utilization of MACs. On EfficientNet, our method
consistently outperforms the performance of the original scaling method. In
particular, with application of our method on GhostNet, we achieve
state-of-the-art 80.9% and 84.3% ImageNet top-1 accuracies under the setting of
600M and 4.4B MACs, respectively.

    

### [[2108.00194] Using Knowledge-Embedded Attention to Augment Pre-trained Language Models for Fine-Grained Emotion Recognition](http://arxiv.org/abs/2108.00194)


  Modern emotion recognition systems are trained to recognize only a small set
of emotions, and hence fail to capture the broad spectrum of emotions people
experience and express in daily life. In order to engage in more empathetic
interactions, future AI has to perform \textit{fine-grained} emotion
recognition, distinguishing between many more varied emotions. Here, we focus
on improving fine-grained emotion recognition by introducing external knowledge
into a pre-trained self-attention model. We propose Knowledge-Embedded
Attention (KEA) to use knowledge from emotion lexicons to augment the
contextual representations from pre-trained ELECTRA and BERT models. Our
results and error analyses outperform previous models on several datasets, and
is better able to differentiate closely-confusable emotions, such as afraid and
terrified.

    

### [[2108.00205] Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding](http://arxiv.org/abs/2108.00205)


  Current one-stage methods for visual grounding encode the language query as
one holistic sentence embedding before fusion with visual feature. Such a
formulation does not treat each word of a query sentence on par when modeling
language to visual attention, therefore prone to neglect words which are less
important for sentence embedding but critical for visual grounding. In this
paper we propose Word2Pix: a one-stage visual grounding network based on
encoder-decoder transformer architecture that enables learning for textual to
visual feature correspondence via word to pixel attention. The embedding of
each word from the query sentence is treated alike by attending to visual
pixels individually instead of single holistic sentence embedding. In this way,
each word is given equivalent opportunity to adjust the language to vision
attention towards the referent target through multiple stacks of transformer
decoder layers. We conduct the experiments on RefCOCO, RefCOCO+ and RefCOCOg
datasets and the proposed Word2Pix outperforms existing one-stage methods by a
notable margin. The results obtained also show that Word2Pix surpasses
two-stage visual grounding models, while at the same time keeping the merits of
one-stage paradigm namely end-to-end training and real-time inference speed
intact.

    

### [[2108.00238] Unlimited Neighborhood Interaction for Heterogeneous Trajectory Prediction](http://arxiv.org/abs/2108.00238)


  Understanding complex social interactions among agents is a key challenge for
trajectory prediction. Most existing methods consider the interactions between
pairwise traffic agents or in a local area, while the nature of interactions is
unlimited, involving an uncertain number of agents and non-local areas
simultaneously. Besides, they only focus on homogeneous trajectory prediction,
namely those among agents of the same category, while neglecting people's
diverse reaction patterns toward traffic agents in different categories. To
address these problems, we propose a simple yet effective Unlimited
Neighborhood Interaction Network (UNIN), which predicts trajectories of
heterogeneous agents in multiply categories. Specifically, the proposed
unlimited neighborhood interaction module generates the fused-features of all
agents involved in an interaction simultaneously, which is adaptive to any
number of agents and any range of interaction area. Meanwhile, a hierarchical
graph attention module is proposed to obtain category-tocategory interaction
and agent-to-agent interaction. Finally, parameters of a Gaussian Mixture Model
are estimated for generating the future trajectories. Extensive experimental
results on benchmark datasets demonstrate a significant performance improvement
of our method over the state-ofthe-art methods.

    

### [[2108.00249] SyDog: A Synthetic Dog Dataset for Improved 2D Pose Estimation](http://arxiv.org/abs/2108.00249)


  Estimating the pose of animals can facilitate the understanding of animal
motion which is fundamental in disciplines such as biomechanics, neuroscience,
ethology, robotics and the entertainment industry. Human pose estimation models
have achieved high performance due to the huge amount of training data
available. Achieving the same results for animal pose estimation is challenging
due to the lack of animal pose datasets. To address this problem we introduce
SyDog: a synthetic dataset of dogs containing ground truth pose and bounding
box coordinates which was generated using the game engine, Unity. We
demonstrate that pose estimation models trained on SyDog achieve better
performance than models trained purely on real data and significantly reduce
the need for the labour intensive labelling of images. We release the SyDog
dataset as a training and evaluation benchmark for research in animal motion.

    

### [[2108.00270] Opinion Prediction with User Fingerprinting](http://arxiv.org/abs/2108.00270)


  Opinion prediction is an emerging research area with diverse real-world
applications, such as market research and situational awareness. We identify
two lines of approaches to the problem of opinion prediction. One uses
topic-based sentiment analysis with time-series modeling, while the other uses
static embedding of text. The latter approaches seek user-specific solutions by
generating user fingerprints. Such approaches are useful in predicting user's
reactions to unseen content. In this work, we propose a novel dynamic
fingerprinting method that leverages contextual embedding of user's comments
conditioned on relevant user's reading history. We integrate BERT variants with
a recurrent neural network to generate predictions. The results show up to 13\%
improvement in micro F1-score compared to previous approaches. Experimental
results show novel insights that were previously unknown such as better
predictions for an increase in dynamic history length, the impact of the nature
of the article on performance, thereby laying the foundation for further
research.

    

### [[2108.00320] StudyMe: A New Mobile App for User-Centric N-of-1 Trials](http://arxiv.org/abs/2108.00320)


  N-of-1 trials are multi-crossover self-experiments that allow individuals to
systematically evaluate the effect of interventions on their personal health
goals. Although several tools for N-of-1 trials exist, none support non-experts
in conducting their own user-centric trials. In this study we present StudyMe,
an open-source mobile application that is freely available from
this https URL and offers users
flexibility and guidance in configuring every component of their trials. We
also present research that informed the development of StudyMe. Through an
initial survey with 272 participants, we learned that individuals are
interested in a variety of personal health aspects and have unique ideas on how
to improve them. In an iterative, user-centered development process with
intermediate user tests we developed StudyMe that also features an educational
part to communicate N-of-1 trial concepts. A final empirical evaluation of
StudyMe showed that all participants were able to create their own trials
successfully using StudyMe and the app achieved a very good usability rating.
Our findings suggest that StudyMe provides a significant step towards enabling
individuals to apply a systematic science-oriented approach to personalize
health-related interventions and behavior modifications in their everyday
lives.

    

### [[2108.00335] Towards Adversarially Robust and Domain Generalizable Stereo Matching by Rethinking DNN Feature Backbones](http://arxiv.org/abs/2108.00335)


  Stereo matching has recently witnessed remarkable progress using Deep Neural
Networks (DNNs). But, how robust are they? Although it has been well-known that
DNNs often suffer from adversarial vulnerability with a catastrophic drop in
performance, the situation is even worse in stereo matching. This paper first
shows that a type of weak white-box attacks can fail state-of-the-art methods.
The attack is learned by a proposed stereo-constrained projected gradient
descent (PGD) method in stereo matching. This observation raises serious
concerns for the deployment of DNN-based stereo matching. Parallel to the
adversarial vulnerability, DNN-based stereo matching is typically trained under
the so-called simulation to reality pipeline, and thus domain generalizability
is an important problem. This paper proposes to rethink the learnable DNN-based
feature backbone towards adversarially-robust and domain generalizable stereo
matching, either by completely removing it or by applying it only to the left
reference image. It computes the matching cost volume using the classic
multi-scale census transform (i.e., local binary pattern) of the raw input
stereo images, followed by a stacked Hourglass head sub-network solving the
matching problem. In experiments, the proposed method is tested in the
SceneFlow dataset and the KITTI2015 benchmark. It significantly improves the
adversarial robustness, while retaining accuracy performance comparable to
state-of-the-art methods. It also shows better generalizability from simulation
(SceneFlow) to real (KITTI) datasets when no fine-tuning is used.

    

### [[2108.00356] Improving Social Meaning Detection with Pragmatic Masking and Surrogate Fine-Tuning](http://arxiv.org/abs/2108.00356)


  Masked language models (MLMs) are pretrained with a denoising objective that,
while useful, is in a mismatch with the objective of downstream fine-tuning. We
propose pragmatic masking and surrogate fine-tuning as two strategies that
exploit social cues to drive pre-trained representations toward a broad set of
concepts useful for a wide class of social meaning tasks. To test our methods,
we introduce a new benchmark of 15 different Twitter datasets for social
meaning detection. Our methods achieve 2.34% F1 over a competitive baseline,
while outperforming other transfer learning methods such as multi-task learning
and domain-specific language models pretrained on large datasets. With only 5%
of training data (severely few-shot), our methods enable an impressive 68.74%
average F1, and we observe promising results in a zero-shot setting involving
six datasets from three different languages.

    

### [[2108.00358] Applications of Artificial Neural Networks in Microorganism Image Analysis: A Comprehensive Review from Conventional Multilayer Perceptron to Popular Convolutional Neural Network and Potential Visual Transformer](http://arxiv.org/abs/2108.00358)


  Microorganisms are widely distributed in the human daily living environment.
They play an essential role in environmental pollution control, disease
prevention and treatment, and food and drug production. The identification,
counting, and detection are the basic steps for making full use of different
microorganisms. However, the conventional analysis methods are expensive,
laborious, and time-consuming. To overcome these limitations, artificial neural
networks are applied for microorganism image analysis. We conduct this review
to understand the development process of microorganism image analysis based on
artificial neural networks. In this review, the background and motivation are
introduced first. Then, the development of artificial neural networks and
representative networks are introduced. After that, the papers related to
microorganism image analysis based on classical and deep neural networks are
reviewed from the perspectives of different tasks. In the end, the methodology
analysis and potential direction are discussed.

    

### [[2108.00362] Neural Free-Viewpoint Performance Rendering under ComplexHuman-object Interactions](http://arxiv.org/abs/2108.00362)


  4D reconstruction of human-object interaction is critical for immersive VR/AR
experience and human activity understanding. Recent advances still fail to
recover fine geometry and texture results from sparse RGB inputs, especially
under challenging human-object interactions scenarios. In this paper, we
propose a neural human performance capture and rendering system to generate
both high-quality geometry and photo-realistic texture of both human and
objects under challenging interaction scenarios in arbitrary novel views, from
only sparse RGB streams. To deal with complex occlusions raised by human-object
interactions, we adopt a layer-wise scene decoupling strategy and perform
volumetric reconstruction and neural rendering of the human and object.
Specifically, for geometry reconstruction, we propose an interaction-aware
human-object capture scheme that jointly considers the human reconstruction and
object reconstruction with their correlations. Occlusion-aware human
reconstruction and robust human-aware object tracking are proposed for
consistent 4D human-object dynamic reconstruction. For neural texture
rendering, we propose a layer-wise human-object rendering scheme, which
combines direction-aware neural blending weight learning and spatial-temporal
texture completion to provide high-resolution and photo-realistic texture
results in the occluded scenarios. Extensive experiments demonstrate the
effectiveness of our approach to achieve high-quality geometry and texture
reconstruction in free viewpoints for challenging human-object interactions.

    

### [[2108.00366] Agent-aware State Estimation in Autonomous Vehicles](http://arxiv.org/abs/2108.00366)


  Autonomous systems often operate in environments where the behavior of
multiple agents is coordinated by a shared global state. Reliable estimation of
the global state is thus critical for successfully operating in a multi-agent
setting. We introduce agent-aware state estimation -- a framework for
calculating indirect estimations of state given observations of the behavior of
other agents in the environment. We also introduce transition-independent
agent-aware state estimation -- a tractable class of agent-aware state
estimation -- and show that it allows the speed of inference to scale linearly
with the number of agents in the environment. As an example, we model traffic
light classification in instances of complete loss of direct observation. By
taking into account observations of vehicular behavior from multiple directions
of traffic, our approach exhibits accuracy higher than that of existing traffic
light-only HMM methods on a real-world autonomous vehicle data set under a
variety of simulated occlusion scenarios.

    

### [[2108.00381] Emerging Methods of Auction Design in Social Networks](http://arxiv.org/abs/2108.00381)


  In recent years, a new branch of auction models called diffusion auction has
extended the traditional auction into social network scenarios. The diffusion
auction models the auction as a networked market whose nodes are potential
customers and whose edges are the relations between these customers. The
diffusion auction mechanism can incentivize buyers to not only submit a
truthful bid, but also further invite their surrounding neighbors to
participate into the auction. It can convene more participants than traditional
auction mechanisms, which leads to better optimizations of different key
aspects, such as social welfare, seller's revenue, amount of redistributed
money and so on. The diffusion auctions have recently attracted a discrete
interest in the algorithmic game theory and market design communities. This
survey summarizes the current progress of diffusion auctions.

    

### [[2108.00385] Transformer-based deep imitation learning for dual-arm robot manipulation](http://arxiv.org/abs/2108.00385)


  Deep imitation learning is promising for solving dexterous manipulation tasks
because it does not require an environment model and pre-programmed robot
behavior. However, its application to dual-arm manipulation tasks remains
challenging. In a dual-arm manipulation setup, the increased number of state
dimensions caused by the additional robot manipulators causes distractions and
results in poor performance of the neural networks. We address this issue using
a self-attention mechanism that computes dependencies between elements in a
sequential input and focuses on important elements. A Transformer, a variant of
self-attention architecture, is applied to deep imitation learning to solve
dual-arm manipulation tasks in the real world. The proposed method has been
tested on dual-arm manipulation tasks using a real robot. The experimental
results demonstrated that the Transformer-based deep imitation learning
architecture can attend to the important features among the sensory inputs,
therefore reducing distractions and improving manipulation performance when
compared with the baseline architecture without the self-attention mechanisms.

    

### [[2108.00400] Transformer-Encoder-GRU (T-E-GRU) for Chinese Sentiment Analysis on Chinese Comment Text](http://arxiv.org/abs/2108.00400)


  Chinese sentiment analysis (CSA) has always been one of the challenges in
natural language processing due to its complexity and uncertainty. Transformer
has succeeded in capturing semantic features, but it uses position encoding to
capture sequence features, which has great shortcomings compared with the
recurrent model. In this paper, we propose T-E-GRU for Chinese sentiment
analysis, which combine transformer encoder and GRU. We conducted experiments
on three Chinese comment datasets. In view of the confusion of punctuation
marks in Chinese comment texts, we selectively retain some punctuation marks
with sentence segmentation ability. The experimental results show that T-E-GRU
outperforms classic recurrent model and recurrent model with attention.

    

### [[2108.00415] Computational Hierarchy of Elementary Cellular Automata](http://arxiv.org/abs/2108.00415)


  The complexity of cellular automata is traditionally measured by their
computational capacity. However, it is difficult to choose a challenging set of
computational tasks suitable for the parallel nature of such systems. We study
the ability of automata to emulate one another, and we use this notion to
define such a set of naturally emerging tasks. We present the results for
elementary cellular automata, although the core ideas can be extended to other
computational systems. We compute a graph showing which elementary cellular
automata can be emulated by which and show that certain chaotic automata are
the only ones that cannot emulate any automata non-trivially. Finally, we use
the emulation notion to suggest a novel definition of chaos that we believe is
suitable for discrete computational systems. We believe our work can help
design parallel computational systems that are Turing-complete and also
computationally efficient.

    

### [[2108.00449] Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization](http://arxiv.org/abs/2108.00449)


  Text style transfer aims to alter the style (e.g., sentiment) of a sentence
while preserving its content. A common approach is to map a given sentence to
content representation that is free of style, and the content representation is
fed to a decoder with a target style. Previous methods in filtering style
completely remove tokens with style at the token level, which incurs the loss
of content information. In this paper, we propose to enhance content
preservation by implicitly removing the style information of each token with
reverse attention, and thereby retain the content. Furthermore, we fuse content
information when building the target style representation, making it dynamic
with respect to the content. Our method creates not only style-independent
content representation, but also content-dependent style representation in
transferring style. Empirical results show that our method outperforms the
state-of-the-art baselines by a large margin in terms of content preservation.
In addition, it is also competitive in terms of style transfer accuracy and
fluency.

    

### [[2108.00516] BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models](http://arxiv.org/abs/2108.00516)


  Tracking the 6D pose of objects in video sequences is important for robot
manipulation. Most prior efforts, however, often assume that the target
object's CAD model, at least at a category-level, is available for offline
training or during online template matching. This work proposes BundleTrack, a
general framework for 6D pose tracking of novel objects, which does not depend
upon 3D models, either at the instance or category-level. It leverages the
complementary attributes of recent advances in deep learning for segmentation
and robust feature extraction, as well as memory-augmented pose graph
optimization for spatiotemporal consistency. This enables long-term, low-drift
tracking under various challenging scenarios, including significant occlusions
and object motions. Comprehensive experiments given two public benchmarks
demonstrate that the proposed approach significantly outperforms state-of-art,
category-level 6D tracking or dynamic SLAM methods. When compared against
state-of-art methods that rely on an object instance CAD model, comparable
performance is achieved, despite the proposed method's reduced information
requirements. An efficient implementation in CUDA provides a real-time
performance of 10Hz for the entire framework. Code is available at:
this https URL


### [[2108.00551] Cybonto: Towards Human Cognitive Digital Twins for Cybersecurity](http://arxiv.org/abs/2108.00551)


  Cyber defense is reactive and slow. On average, the time-to-remedy is
hundreds of times larger than the time-to-compromise. In response to the
expanding ever-more-complex threat landscape, Digital Twins (DTs) and
particularly Human Digital Twins (HDTs) offer the capability of running massive
simulations across multiple knowledge domains. Simulated results may offer
insights into adversaries' behaviors and tactics, resulting in better proactive
cyber-defense strategies. For the first time, this paper solidifies the vision
of DTs and HDTs for cybersecurity via the Cybonto conceptual framework
proposal. The paper also contributes the Cybonto ontology, formally documenting
108 constructs and thousands of cognitive-related paths based on 20 time-tested
psychology theories. Finally, the paper applied 20 network centrality
algorithms in analyzing the 108 constructs. The identified top 10 constructs
call for extensions of current digital cognitive architectures in preparation
for the DT future.

    

### [[2108.00564] Learning Maritime Obstacle Detection from Weak Annotations by Scaffolding](http://arxiv.org/abs/2108.00564)


  Coastal water autonomous boats rely on robust perception methods for obstacle
detection and timely collision avoidance. The current state-of-the-art is based
on deep segmentation networks trained on large datasets. Per-pixel ground truth
labeling of such datasets, however, is labor-intensive and expensive. We
observe that far less information is required for practical obstacle avoidance
- the location of water edge on static obstacles like shore and approximate
location and bounds of dynamic obstacles in the water is sufficient to plan a
reaction. We propose a new scaffolding learning regime (SLR) that allows
training obstacle detection segmentation networks only from such weak
annotations, thus significantly reducing the cost of ground-truth labeling.
Experiments show that maritime obstacle segmentation networks trained using SLR
substantially outperform the same networks trained with dense ground truth
labels. Thus accuracy is not sacrificed for labelling simplicity but is in fact
improved, which is a remarkable result.

    

### [[2108.00573] MuSiQue: Multi-hop Questions via Single-hop Question Composition](http://arxiv.org/abs/2108.00573)


  To build challenging multi-hop question answering datasets, we propose a
bottom-up semi-automatic process of constructing multi-hop question via
composition of single-hop questions. Constructing multi-hop questions as
composition of single-hop questions allows us to exercise greater control over
the quality of the resulting multi-hop questions. This process allows building
a dataset with (i) connected reasoning where each step needs the answer from a
previous step; (ii) minimal train-test leakage by eliminating even partial
overlap of reasoning steps; (iii) variable number of hops and composition
structures; and (iv) contrasting unanswerable questions by modifying the
context. We use this process to construct a new multihop QA dataset:
MuSiQue-Ans with ~25K 2-4 hop questions using seed questions from 5 existing
single-hop datasets. Our experiments demonstrate that MuSique is challenging
for state-of-the-art QA models (e.g., human-machine gap of $~$30 F1 pts),
significantly harder than existing datasets (2x human-machine gap), and
substantially less cheatable (e.g., a single-hop model is worse by 30 F1 pts).
We also build an even more challenging dataset, MuSiQue-Full, consisting of
answerable and unanswerable contrast question pairs, where model performance
drops further by 13+ F1 pts. For data and code, see
\url{this https URL}.

    

### [[2108.00578] Is My Model Using The Right Evidence? Systematic Probes for Examining Evidence-Based Tabular Reasoning](http://arxiv.org/abs/2108.00578)


  While neural models routinely report state-of-the-art performance across NLP
tasks involving reasoning, their outputs are often observed to not properly use
and reason on the evidence presented to them in the inputs. A model that
reasons properly is expected to attend to the right parts of the input, be
self-consistent in its predictions across examples, avoid spurious patterns in
inputs, and to ignore biasing from its underlying pre-trained language model in
a nuanced, context-sensitive fashion (e.g. handling counterfactuals). Do
today's models do so? In this paper, we study this question using the problem
of reasoning on tabular data. The tabular nature of the input is particularly
suited for the study as it admits systematic probes targeting the properties
listed above. Our experiments demonstrate that a BERT-based model
representative of today's state-of-the-art fails to properly reason on the
following counts: it often (a) misses the relevant evidence, (b) suffers from
hypothesis and knowledge biases, and, (c) relies on annotation artifacts and
knowledge from pre-trained language models as primary evidence rather than
relying on reasoning on the premises in the tabular input.

    

### [[2108.00603] TabPert: An Effective Platform for Tabular Perturbation](http://arxiv.org/abs/2108.00603)


  To truly grasp reasoning ability, a Natural Language Inference model should
be evaluated on counterfactual data. TabPert facilitates this by assisting in
the generation of such counterfactual data for assessing model tabular
reasoning issues. TabPert allows a user to update a table, change its
associated hypotheses, change their labels, and highlight rows that are
important for hypothesis classification. TabPert also captures information
about the techniques used to automatically produce the table, as well as the
strategies employed to generate the challenging hypotheses. These
counterfactual tables and hypotheses, as well as the metadata, can then be used
to explore an existing model's shortcomings methodically and quantitatively.

    

### [[2108.00633] Planning with Learned Binarized Neural Networks Benchmarks for MaxSAT Evaluation 2021](http://arxiv.org/abs/2108.00633)


  This document provides a brief introduction to learned automated planning
problem where the state transition function is in the form of a binarized
neural network (BNN), presents a general MaxSAT encoding for this problem, and
describes the four domains, namely: Navigation, Inventory Control, System
Administrator and Cellda, that are submitted as benchmarks for MaxSAT
Evaluation 2021.

    

### [[2108.00710] Multi-Objective Path-Based D* Lite](http://arxiv.org/abs/2108.00710)


  Incremental graph search algorithms, such as D* Lite, reuse previous search
efforts to speed up subsequent similar path planning tasks. These algorithms
have demonstrated their efficiency in comparison with search from scratch, and
have been leveraged in many applications such as navigation in unknown terrain.
On the other hand, path planning typically involves optimizing multiple
conflicting objectives simultaneously, such as travel risk, arrival time, etc.
Multi-objective path planning is challenging as the number of "Pareto-optimal"
solutions can grow exponentially with respect to the size of the graph, which
makes it computationally burdensome to plan from scratch each time when similar
planning tasks needs to be solved. This article presents a new multi-objective
incremental search algorithm called Multi-Objective Path-Based D* Lite (MOPBD*)
which reuses previous search efforts to speed up subsequent planning tasks
while optimizing multiple objectives. Numerical results show that MOPBD* is
more efficient than search from scratch and runs an order of magnitude faster
than existing incremental method for multi-objective path planning.

    

### [[2108.00716] Orientation-Aware Planning for Parallel Task Execution of Omni-Directional Mobile Robot](http://arxiv.org/abs/2108.00716)


  Omni-directional mobile robot (OMR) systems have been very popular in
academia and industry for their superb maneuverability and flexibility. Yet
their potential has not been fully exploited, where the extra degree of freedom
in OMR can potentially enable the robot to carry out extra tasks. For instance,
gimbals or sensors on robots may suffer from a limited field of view or be
constrained by the inherent mechanical design, which will require the chassis
to be orientation-aware and respond in time. To solve this problem and further
develop the OMR systems, in this paper, we categorize the tasks related to OMR
chassis into orientation transition tasks and position transition tasks, where
the two tasks can be carried out at the same time. By integrating the parallel
task goals in a single planning problem, we proposed an orientation-aware
planning architecture for OMR systems to execute the orientation transition and
position transition in a unified and efficient way. A modified trajectory
optimization method called orientation-aware timed-elastic-band (OATEB) is
introduced to generate the trajectory that satisfies the requirements of both
tasks. Experiments in both 2D simulated environments and real scenes are
carried out. A four-wheeled OMR is deployed to conduct the real scene
experiment and the results demonstrate that the proposed method is capable of
simultaneously executing parallel tasks and is applicable to real-life
scenarios.

    

### [[2108.00745] Multi-objective Conflict-based Search Using Safe-interval Path Planning](http://arxiv.org/abs/2108.00745)


  This paper addresses a generalization of the well known multi-agent path
finding (MAPF) problem that optimizes multiple conflicting objectives
simultaneously such as travel time and path risk. This generalization, referred
to as multi-objective MAPF (MOMAPF), arises in several applications ranging
from hazardous material transportation to construction site planning. In this
paper, we present a new multi-objective conflict-based search (MO-CBS) approach
that relies on a novel multi-objective safe interval path planning (MO-SIPP)
algorithm for its low-level search. We first develop the MO-SIPP algorithm,
show its properties and then embed it in MO-CBS. We present extensive numerical
results to show that (1) there is an order of magnitude improvement in the
average low level search time, and (2) a significant improvement in the success
rates of finding the Pareto-optimal front can be obtained using the proposed
approach in comparison with the state of the art. Finally, we also provide a
case study to demonstrate the potential application of the proposed algorithms
for construction site planning.

    

### [[2108.00760] BezierSeg: Parametric Shape Representation for Fast Object Segmentation in Medical Images](http://arxiv.org/abs/2108.00760)


  Delineating the lesion area is an important task in image-based diagnosis.
Pixel-wise classification is a popular approach to segmenting the region of
interest. However, at fuzzy boundaries such methods usually result in glitches,
discontinuity, or disconnection, inconsistent with the fact that lesions are
solid and smooth. To overcome these undesirable artifacts, we propose the
BezierSeg model which outputs bezier curves encompassing the region of
interest. Directly modelling the contour with analytic equations ensures that
the segmentation is connected, continuous, and the boundary is smooth. In
addition, it offers sub-pixel accuracy. Without loss of accuracy, the bezier
contour can be resampled and overlaid with images of any resolution. Moreover,
a doctor can conveniently adjust the curve's control points to refine the
result. Our experiments show that the proposed method runs in real time and
achieves accuracy competitive with pixel-wise segmentation models.

    

### [[2108.00801] LICHEE: Improving Language Model Pre-training with Multi-grained Tokenization](http://arxiv.org/abs/2108.00801)


  Language model pre-training based on large corpora has achieved tremendous
success in terms of constructing enriched contextual representations and has
led to significant performance gains on a diverse range of Natural Language
Understanding (NLU) tasks. Despite the success, most current pre-trained
language models, such as BERT, are trained based on single-grained
tokenization, usually with fine-grained characters or sub-words, making it hard
for them to learn the precise meaning of coarse-grained words and phrases. In
this paper, we propose a simple yet effective pre-training method named LICHEE
to efficiently incorporate multi-grained information of input text. Our method
can be applied to various pre-trained language models and improve their
representation capability. Extensive experiments conducted on CLUE and
SuperGLUE demonstrate that our method achieves comprehensive improvements on a
wide variety of NLU tasks in both Chinese and English with little extra
inference cost incurred, and that our best ensemble model achieves the
state-of-the-art performance on CLUE benchmark competition.

    

### [[2108.00804] Relation Aware Semi-autoregressive Semantic Parsing for NL2SQL](http://arxiv.org/abs/2108.00804)


  Natural language to SQL (NL2SQL) aims to parse a natural language with a
given database into a SQL query, which widely appears in practical Internet
applications. Jointly encode database schema and question utterance is a
difficult but important task in NL2SQL. One solution is to treat the input as a
heterogeneous graph. However, it failed to learn good word representation in
question utterance. Learning better word representation is important for
constructing a well-designed NL2SQL system. To solve the challenging task, we
present a Relation aware Semi-autogressive Semantic Parsing (\MODN) ~framework,
which is more adaptable for NL2SQL. It first learns relation embedding over the
schema entities and question words with predefined schema relations with
ELECTRA and relation aware transformer layer as backbone. Then we decode the
query SQL with a semi-autoregressive parser and predefined SQL syntax. From
empirical results and case study, our model shows its effectiveness in learning
better word representation in NL2SQL.

    

### [[1801.09317] A Cyber Science Based Ontology for Artificial General Intelligence Containment](http://arxiv.org/abs/1801.09317)


  The development of artificial general intelligence is considered by many to
be inevitable. What such intelligence does after becoming aware is not so
certain. To that end, research suggests that the likelihood of artificial
general intelligence becoming hostile to humans is significant enough to
warrant inquiry into methods to limit such potential. Thus, containment of
artificial general intelligence is a timely and meaningful research topic.
While there is limited research exploring possible containment strategies, such
work is bounded by the underlying field the strategies draw upon. Accordingly,
we set out to construct an ontology to describe necessary elements in any
future containment technology. Using existing academic literature, we developed
a single domain ontology containing five levels, 32 codes, and 32 associated
descriptors. Further, we constructed ontology diagrams to demonstrate intended
relationships. We then identified humans, AGI, and the cyber world as novel
agent objects necessary for future containment activities. Collectively, the
work addresses three critical gaps: (a) identifying and arranging fundamental
constructs; (b) situating AGI containment within cyber science; and (c)
developing scientific rigor within the field.

    

### [[1811.03653] Stovepiping and Malicious Software: A Critical Review of AGI Containment](http://arxiv.org/abs/1811.03653)


  Awareness of the possible impacts associated with artificial intelligence has
risen in proportion to progress in the field. While there are tremendous
benefits to society, many argue that there are just as many, if not more,
concerns related to advanced forms of artificial intelligence. Accordingly,
research into methods to develop artificial intelligence safely is increasingly
important. In this paper, we provide an overview of one such safety paradigm:
containment with a critical lens aimed toward generative adversarial networks
and potentially malicious artificial intelligence. Additionally, we illuminate
the potential for a developmental blindspot in the stovepiping of containment
mechanisms.

    

### [[2005.02878] Multi-Resolution POMDP Planning for Multi-Object Search in 3D](http://arxiv.org/abs/2005.02878)


  Robots operating in households must find objects on shelves, under tables,
and in cupboards. In such environments, it is crucial to search efficiently at
3D scale while coping with limited field of view and the complexity of
searching for multiple objects. Principled approaches to object search
frequently use Partially Observable Markov Decision Process (POMDP) as the
underlying framework for computing search strategies, but constrain the search
space in 2D. In this paper, we present a POMDP formulation for multi-object
search in a 3D region with a frustum-shaped field-of-view. To efficiently solve
this POMDP, we propose a multi-resolution planning algorithm based on online
Monte-Carlo tree search. In this approach, we design a novel octree-based
belief representation to capture uncertainty of the target objects at different
resolution levels, then derive abstract POMDPs at lower resolutions with
dramatically smaller state and observation spaces. Evaluation in a simulated 3D
domain shows that our approach finds objects more efficiently and successfully
compared to a set of baselines without resolution hierarchy in larger instances
under the same computational requirement. We demonstrate our approach on a
mobile robot to find objects placed at different heights in two 10m$^2 \times
2$m regions by moving its base and actuating its torso.

    

### [[2006.04167] A tetrachotomy of ontology-mediated queries with a covering axiom](http://arxiv.org/abs/2006.04167)


  Our concern is the problem of efficiently determining the data complexity of
answering queries mediated by description logic ontologies and constructing
their optimal rewritings to standard database queries. Originated in
ontology-based data access and datalog optimisation, this problem is known to
be computationally very complex in general, with no explicit syntactic
characterisations available. In this article, aiming to understand the
fundamental roots of this difficulty, we strip the problem to the bare bones
and focus on Boolean conjunctive queries mediated by a simple covering axiom
stating that one class is covered by the union of two other classes. We show
that, on the one hand, these rudimentary ontology-mediated queries, called
disjunctive sirups (or d-sirups), capture many features and difficulties of the
general case. For example, answering d-sirups is Pi^p_2-complete for combined
complexity and can be in AC0 or LogSpace-, NL-, P-, or coNP-complete for data
complexity (with the problem of recognising FO-rewritability of d-sirups being
2ExpTime-hard); some d-sirups only have exponential-size resolution proofs,
some only double-exponential-size positive existential FO-rewritings and
single-exponential-size nonrecursive datalog rewritings. On the other hand, we
prove a few partial sufficient and necessary conditions of FO- and
(symmetric/linear-) datalog rewritability of d-sirups. Our main technical
result is a complete and transparent syntactic AC0/NL/P/coNP tetrachotomy of
d-sirups with disjoint covering classes and a path-shaped Boolean conjunctive
query. To obtain this tetrachotomy, we develop new techniques for establishing
P- and coNP-hardness of answering non-Horn ontology-mediated queries as well as
showing that they can be answered in NL.

    

### [[2009.07707] DeepC2: AI-powered Covert Botnet Command and Control on OSNs](http://arxiv.org/abs/2009.07707)


  Botnets are one of the major threats to computer security. In previous botnet
command and control (C&C) scenarios using online social networks (OSNs),
methods for addressing (e.g., IDs, links, or DGAs) are hardcoded into bots.
Once a bot is reverse engineered, the botmaster and C&C infrastructure will be
exposed. Additionally, abnormal content from explicit commands may expose
botmasters and raise anomalies on OSNs. To overcome these deficiencies, we
proposed DeepC2, an AI-powered covert C&C method on OSNs. By leveraging neural
networks, bots can find botmasters by avatars, which are converted into feature
vectors and embedded into bots. Adversaries cannot infer botmasters' accounts
from the vectors. Commands are embedded into normal contents (e.g., tweets and
comments) using text data augmentation and hash collision. Experiments on
Twitter show that command-embedded contents can be generated efficiently, and
bots can find botmasters and obtain commands accurately. Security analysis on
different scenarios show that DeepC2 is robust and hard to be shut down. By
demonstrating how AI may help promote covert communication on OSNs, this work
provides a new perspective on botnet detection and confrontation.

    

### [[2010.04351] Connection Pruning for Deep Spiking Neural Networks with On-Chip Learning](http://arxiv.org/abs/2010.04351)


  Long training time hinders the potential of the deep, large-scale Spiking
Neural Network (SNN) with the on-chip learning capability to be realized on the
embedded systems hardware. Our work proposes a novel connection pruning
approach that can be applied during the on-chip Spike Timing Dependent
Plasticity (STDP)-based learning to optimize the learning time and the network
connectivity of the deep SNN. We applied our approach to a deep SNN with the
Time To First Spike (TTFS) coding and has successfully achieved 2.1x speed-up
and 64% energy savings in the on-chip learning and reduced the network
connectivity by 92.83%, without incurring any accuracy loss. Moreover, the
connectivity reduction results in 2.83x speed-up and 78.24% energy savings in
the inference. Evaluation of our proposed approach on the Field Programmable
Gate Array (FPGA) platform revealed 0.56% power overhead was needed to
implement the pruning algorithm.

    

### [[2010.08052] A Learning Approach to Robot-Agnostic Force-Guided High Precision Assembly](http://arxiv.org/abs/2010.08052)


  In this work we propose a learning approach to high-precision robotic
assembly problems. We focus on the contact-rich phase, where the assembly
pieces are in close contact with each other. Unlike many learning-based
approaches that heavily rely on vision or spatial tracking, our approach takes
force/torque in task space as the only observation. Our training environment is
robotless, as the end-effector is not attached to any specific robot. Trained
policies can then be applied to different robotic arms without re-training.
This approach can greatly reduce complexity to perform contact-rich robotic
assembly in the real world, especially in unstructured settings such as in
architectural construction. To achieve it, we have developed a new distributed
RL agent, named Recurrent Distributed DDPG (RD2), which extends Ape-X DDPG with
recurrency and makes two structural improvements on prioritized experience
replay. Our results show that RD2 is able to solve two fundamental
high-precision assembly tasks, lap-joint and peg-in-hole, and outperforms two
state-of-the-art algorithms, Ape-X DDPG and PPO with LSTM. We have successfully
evaluated our robot-agnostic policies on three robotic arms, Kuka KR60, Franka
Panda, and UR10, in simulation. The video presenting our experiments is
available at this https URL


### [[2102.01149] A Tight Bound for Stochastic Submodular Cover](http://arxiv.org/abs/2102.01149)


  We show that the Adaptive Greedy algorithm of Golovin and Krause (2011)
achieves an approximation bound of $(\ln (Q/\eta)+1)$ for Stochastic Submodular
Cover: here $Q$ is the "goal value" and $\eta$ is the smallest non-zero
marginal increase in utility deliverable by an item. (For integer-valued
utility functions, we show a bound of $H(Q)$, where $H(Q)$ is the $Q^{th}$
Harmonic number.) Although this bound was claimed by Golovin and Krause in the
original version of their paper, the proof was later shown to be incorrect by
Nan and Saligrama (2017). The subsequent corrected proof of Golovin and Krause
(2017) gives a quadratic bound of $(\ln(Q/\eta) + 1)^2$. Other previous bounds
for the problem are $56(\ln(Q/\eta) + 1)$, implied by work of Im et al. (2016)
on a related problem, and $k(\ln (Q/\eta)+1)$, due to Deshpande et al. (2016)
and Hellerstein and Kletenik (2018), where $k$ is the number of states. Our
bound generalizes the well-known $(\ln~m + 1)$ approximation bound on the
greedy algorithm for the classical Set Cover problem, where $m$ is the size of
the ground set.

    

### [[2103.04516] Loosely Synchronized Search for Multi-agent Path Finding with Asynchronous Actions](http://arxiv.org/abs/2103.04516)


  Multi-agent path finding (MAPF) determines an ensemble of collision-free
paths for multiple agents between their respective start and goal locations.
Among the available MAPF planners for workspace modeled as a graph, A*-based
approaches have been widely investigated due to their guarantees on
completeness and solution optimality, and have demonstrated their efficiency in
many scenarios. However, almost all of these A*-based methods assume that each
agent executes an action concurrently in that all agents start and stop
together. This article presents a natural generalization of MAPF with
asynchronous actions (MAPF-AA) where agents do not necessarily start and stop
concurrently. The main contribution of the work is a proposed approach called
Loosely Synchronized Search (LSS) that extends A*-based MAPF planners to handle
asynchronous actions. We show LSS is complete and finds an optimal solution if
one exists. We also combine LSS with other existing MAPF methods that aims to
trade-off optimality for computational efficiency. Numerical results are
presented to corroborate the performance of LSS and the applicability of the
proposed method is verified in the Robotarium, a remotely accessible swarm
robotics research platform.

    

### [[2103.07903] Investigating Value of Curriculum Reinforcement Learning in Autonomous Driving Under Diverse Road and Weather Conditions](http://arxiv.org/abs/2103.07903)


  Applications of reinforcement learning (RL) are popular in autonomous driving
tasks. That being said, tuning the performance of an RL agent and guaranteeing
the generalization performance across variety of different driving scenarios is
still largely an open problem. In particular, getting good performance on
complex road and weather conditions require exhaustive tuning and computation
time. Curriculum RL, which focuses on solving simpler automation tasks in order
to transfer knowledge to complex tasks, is attracting attention in RL
community. The main contribution of this paper is a systematic study for
investigating the value of curriculum reinforcement learning in autonomous
driving applications. For this purpose, we setup several different driving
scenarios in a realistic driving simulator, with varying road complexity and
weather conditions. Next, we train and evaluate performance of RL agents on
different sequences of task combinations and curricula. Results show that
curriculum RL can yield significant gains in complex driving tasks, both in
terms of driving performance and sample complexity. Results also demonstrate
that different curricula might enable different benefits, which hints future
research directions for automated curriculum training.

    

### [[2103.08624] Autonomous Drone Racing with Deep Reinforcement Learning](http://arxiv.org/abs/2103.08624)


  In many robotic tasks, such as autonomous drone racing, the goal is to travel
through a set of waypoints as fast as possible. A key challenge for this task
is planning the time-optimal trajectory, which is typically solved by assuming
perfect knowledge of the waypoints to pass in advance. The resulting solution
is either highly specialized for a single-track layout, or suboptimal due to
simplifying assumptions about the platform dynamics. In this work, a new
approach to near-time-optimal trajectory generation for quadrotors is
presented. Leveraging deep reinforcement learning and relative gate
observations, our approach can compute near-time-optimal trajectories and adapt
the trajectory to environment changes. Our method exhibits computational
advantages over approaches based on trajectory optimization for non-trivial
track configurations. The proposed approach is evaluated on a set of race
tracks in simulation and the real world, achieving speeds of up to 60 km/h with
a physical quadrotor.

    

### [[2103.09189] Goal-constrained Sparse Reinforcement Learning for End-to-End Driving](http://arxiv.org/abs/2103.09189)


  Deep reinforcement Learning for end-to-end driving is limited by the need of
complex reward engineering. Sparse rewards can circumvent this challenge but
suffers from long training time and leads to sub-optimal policy. In this work,
we explore full-control driving with only goal-constrained sparse reward and
propose a curriculum learning approach for end-to-end driving using only
navigation view maps that benefit from small virtual-to-real domain gap. To
address the complexity of multiple driving policies, we learn concurrent
individual policies selected at inference by a navigation system. We
demonstrate the ability of our proposal to generalize on unseen road layout,
and to drive significantly longer than in the training.

    

### [[2103.11512] Robust Multi-Modal Policies for Industrial Assembly via Reinforcement Learning and Demonstrations: A Large-Scale Study](http://arxiv.org/abs/2103.11512)


  Over the past several years there has been a considerable research investment
into learning-based approaches to industrial assembly, but despite significant
progress these techniques have yet to be adopted by industry. We argue that it
is the prohibitively large design space for Deep Reinforcement Learning (DRL),
rather than algorithmic limitations per se, that are truly responsible for this
lack of adoption. Pushing these techniques into the industrial mainstream
requires an industry-oriented paradigm which differs significantly from the
academic mindset. In this paper we define criteria for industry-oriented DRL,
and perform a thorough comparison according to these criteria of one family of
learning approaches, DRL from demonstration, against a professional industrial
integrator on the recently established NIST assembly benchmark. We explain the
design choices, representing several years of investigation, which enabled our
DRL system to consistently outperform the integrator baseline in terms of both
speed and reliability. Finally, we conclude with a competition between our DRL
system and a human on a challenge task of insertion into a randomly moving
target. This study suggests that DRL is capable of outperforming not only
established engineered approaches, but the human motor system as well, and that
there remains significant room for improvement. Videos can be found on our
project website: this https URL.

    

### [[2104.00639] HLE-UPC at SemEval-2021 Task 5: Multi-Depth DistilBERT for Toxic Spans Detection](http://arxiv.org/abs/2104.00639)


  This paper presents our submission to SemEval-2021 Task 5: Toxic Spans
Detection. The purpose of this task is to detect the spans that make a text
toxic, which is a complex labour for several reasons. Firstly, because of the
intrinsic subjectivity of toxicity, and secondly, due to toxicity not always
coming from single words like insults or offends, but sometimes from whole
expressions formed by words that may not be toxic individually. Following this
idea of focusing on both single words and multi-word expressions, we study the
impact of using a multi-depth DistilBERT model, which uses embeddings from
different layers to estimate the final per-token toxicity. Our quantitative
results show that using information from multiple depths boosts the performance
of the model. Finally, we also analyze our best model qualitatively.

    

### [[2106.08624] Structured DropConnect for Uncertainty Inference in Image Classification](http://arxiv.org/abs/2106.08624)


  With the complexity of the network structure, uncertainty inference has
become an important task to improve the classification accuracy for artificial
intelligence systems. For image classification tasks, we propose a structured
DropConnect (SDC) framework to model the output of a deep neural network by a
Dirichlet distribution. We introduce a DropConnect strategy on weights in the
fully connected layers during training. In test, we split the network into
several sub-networks, and then model the Dirichlet distribution by match its
moments with the mean and variance of the outputs of these sub-networks. The
entropy of the estimated Dirichlet distribution is finally utilized for
uncertainty inference. In this paper, this framework is implemented on LeNet$5$
and VGG$16$ models for misclassification detection and out-of-distribution
detection on MNIST and CIFAR-$10$ datasets. Experimental results show that the
performance of the proposed SDC can be comparable to other uncertainty
inference methods. Furthermore, the SDC is adapted well to different network
structures with certain generalization capabilities and research prospects.

    

### [[2106.11791] Exemplars-guided Empathetic Response Generation Controlled by the Elements of Human Communication](http://arxiv.org/abs/2106.11791)


  The majority of existing methods for empathetic response generation rely on
the emotion of the context to generate empathetic responses. However, empathy
is much more than generating responses with an appropriate emotion. It also
often entails subtle expressions of understanding and personal resonance with
the situation of the other interlocutor. Unfortunately, such qualities are
difficult to quantify and the datasets lack the relevant annotations. To
address this issue, in this paper we propose an approach that relies on
exemplars to cue the generative model on fine stylistic properties that signal
empathy to the interlocutor. To this end, we employ dense passage retrieval to
extract relevant exemplary responses from the training set. Three elements of
human communication -- emotional presence, interpretation, and exploration, and
sentiment are additionally introduced using synthetic labels to guide the
generation towards empathy. The human evaluation is also extended by these
elements of human communication. We empirically show that these approaches
yield significant improvements in empathetic response quality in terms of both
automated and human-evaluated metrics. The implementation is available at
this https URL.

    

### [[2108.00567] Agile Elicitation of Scalability Requirements for Open Systems: A Case Study](http://arxiv.org/abs/2108.00567)


  Eliciting scalability requirements during agile software development is
complicated and poorly described in previous research. This article presents a
lightweight artifact for eliciting scalability requirements during agile
software development: the ScrumScale model. The ScrumScale model is a simple
spreadsheet. The scalability concepts underlying the ScrumScale model are
clarified in this design science research, which also utilizes coordination
theory. This paper describes the open banking case study, where a legacy
banking system becomes open. This challenges the scalability of this legacy
system. The first step in understanding this challenge is to elicit the new
scalability requirements. In the open banking case study, key stakeholders from
TietoEVRY spent 55 hours eliciting TietoEVRY's open banking project's
scalability requirements. According to TietoEVRY, the ScrumScale model provided
a systematic way of producing scalability requirements. For TietoEVRY, the
scalability concepts behind the ScrumScale model also offered significant
advantages in dialogues with other stakeholders.

    

### [[2108.00225] Solving Constrained Horn Clauses over ADTs by Finite Model Finding](http://arxiv.org/abs/2108.00225)


  First-order logic is a natural way of expressing the properties of
computation, traditionally used in various program logics for expressing the
correctness properties and certificates. Subsequently, modern methods in the
automated inference of program invariants progress towards the construction of
first-order definable invariants. Although the first-order representations are
very expressive for some theories, they fail to express many interesting
properties of algebraic data types (ADTs).
Thus we propose to represent program invariants regularly with tree automata.
We show how to automatically infer such regular invariants of ADT-manipulating
programs using finite model finders. We have implemented our approach and
evaluated it against the state-of-art engines for the invariant inference in
first-order logic for ADT-manipulating programs. Our evaluation shows that
automata-based representation of invariants is more practical than the one
based on first-order logic since invariants are capable of expressing more
complex properties of the computation and their automatic construction is less
expensive.

    

### [[2108.00281] Enhanced Regular Corecursion for Data Streams](http://arxiv.org/abs/2108.00281)


  We propose a simple calculus for processing data streams (infinite flows of
data series), represented by finite sets of equations built on stream
operators. Furthermore, functions defining streams are regularly corecursive,
that is, cyclic calls are detected, avoiding non-termination as happens with
ordinary recursion in the call-by-value evaluation strategy. As we illustrate
by several examples, the combination of such two mechanisms provides a good
compromise between expressive power and decidability. Notably, we provide an
algorithm to check that the stream returned by a function call is represented
by a well-formed set of equations which actually admits a unique solution,
hence access to an arbitrary element of the returned stream will never diverge.

    

### [[2108.00739] Analysis and Transformation of Constrained Horn Clauses for Program Verification](http://arxiv.org/abs/2108.00739)


  This paper surveys recent work on applying analysis and transformation
techniques that originate in the field of constraint logic programming (CLP) to
the problem of verifying software systems. We present specialisation-based
techniques for translating verification problems for different programming
languages, and in general software systems, into satisfiability problems for
constrained Horn clauses (CHCs), a term that has become popular in the
verification field to refer to CLP programs. Then, we describe static analysis
techniques for CHCs that may be used for inferring relevant program properties,
such as loop invariants. We also give an overview of some transformation
techniques based on specialisation and fold/unfold rules, which are useful for
improving the effectiveness of CHC satisfiability tools. Finally, we discuss
future developments in applying these techniques.

    

### [[2010.03608] Type checking extracted methods](http://arxiv.org/abs/2010.03608)


  Many object-oriented dynamic languages allow programmers to extract methods
from objects and treat them as functions. This allows for flexible programming
patterns, but presents challenges for type systems. In particular, a simple
treatment of method extraction would require methods to be contravariant in the
receiver type, making overriding all-but-impossible. We present a detailed
investigation of this problem, as well as an implemented and evaluated
solution. Method extraction is a feature of many dynamically-typed and
gradually-typed languages, ranging from Python and PHP to Flow and TypeScript.
In these languages, the underlying representation of objects as records of
procedures can be accessed, and the procedures that implement methods can be
reified as functions that can be called independently. In many of these
languages, the programmer can then explicitly specify the this value to be used
when the method implementation is called.
Unfortunately, as we show, existing gradual type systems such as TypeScript
and Flow are unsound in the presence of method extraction. The problem for
typing any such system is that the flexibility it allows must be tamed by
requiring a connection between the object the method was extracted from, and
the function value that is later called.
In Racket, where a method extraction-like facility, dubbed "structure type
properties", is fundamental to classes, generic methods, and other APIs, these
same challenges arise, and must be solved to support this feature in Typed
Racket. We show how to combine two existing type system features-existential
types and occurrence typing-to produce a sound approach to typing method
extraction...

    

### [<title>GPU enabled xgboost.dll - XGBoost</title>](https://discuss.xgboost.ai/t/gpu-enabled-xgboost-dll/2408/1)

### [<title>GPU enabled xgboost.dll - XGBoost</title>](https://discuss.xgboost.ai/t/gpu-enabled-xgboost-dll/2408/2)

### [<title>XGBoost 4J spark giving XGBoostError: std::bad_alloc on databricks - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-4j-spark-giving-xgboosterror-std-bad-alloc-on-databricks/2410/1)