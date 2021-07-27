
## 2021-7-27

### [[2107.11439] What Is The Internet? (Considering Partial Connectivity)](http://arxiv.org/abs/2107.11439)


  After 50 years, the Internet is still defined as "a collection of
interconnected networks". Yet desires of countries for "their own internet"
(Internet secession?), country-level firewalling, and persistent peering
disputes all challenge the idea of a single set of "interconnected networks".
We show that the Internet today has peninsulas of persistent, partial
connectivity, and that some outages cause islands where the Internet at the
site is up, but partitioned from the main Internet. We propose a new definition
of the Internet defining a single, global network while helping us to reason
about peninsulas and islands and their relationship to Internet outages. We
provide algorithms to detect peninsulas and islands, find that peninsulas are
more common than outages, with thousands of /24s IPv4 blocks that are part of
peninsulas lasting a month or more. Root causes of most peninsula events (45%)
are transient routing problems. However, a few long-lived peninsulas events
(7%) account for 90% of all peninsula time, and they suggest root causes in
country- or AS-level policy choices. We also show that islands occur. Our
definition shows that no single country can unilaterally claim to be "the
Internet", and helps clarify the spectrum from partial reachability to outages
in prior work.

    

### [[2107.11514] Multi-Perspective Content Delivery Networks Security Framework Using Optimized Unsupervised Anomaly Detection](http://arxiv.org/abs/2107.11514)


  Content delivery networks (CDNs) provide efficient content distribution over
the Internet. CDNs improve the connectivity and efficiency of global
communications, but their caching mechanisms may be breached by
cyber-attackers. Among the security mechanisms, effective anomaly detection
forms an important part of CDN security enhancement. In this work, we propose a
multi-perspective unsupervised learning framework for anomaly detection in
CDNs. In the proposed framework, a multi-perspective feature engineering
approach, an optimized unsupervised anomaly detection model that utilizes an
isolation forest and a Gaussian mixture model, and a multi-perspective
validation method, are developed to detect abnormal behaviors in CDNs mainly
from the client Internet Protocol (IP) and node perspectives, therefore to
identify the denial of service (DoS) and cache pollution attack (CPA) patterns.
Experimental results are presented based on the analytics of eight days of
real-world CDN log data provided by a major CDN operator. Through experiments,
the abnormal contents, compromised nodes, malicious IPs, as well as their
corresponding attack types, are identified effectively by the proposed
framework and validated by multiple cybersecurity experts. This shows the
effectiveness of the proposed method when applied to real-world CDN data.

    

### [[2107.11588] Accelerating Federated Edge Learning via Optimized Probabilistic Device Scheduling](http://arxiv.org/abs/2107.11588)


  The popular federated edge learning (FEEL) framework allows
privacy-preserving collaborative model training via frequent learning-updates
exchange between edge devices and server. Due to the constrained bandwidth,
only a subset of devices can upload their updates at each communication round.
This has led to an active research area in FEEL studying the optimal device
scheduling policy for minimizing communication time. However, owing to the
difficulty in quantifying the exact communication time, prior work in this area
can only tackle the problem partially by considering either the communication
rounds or per-round latency, while the total communication time is determined
by both metrics. To close this gap, we make the first attempt in this paper to
formulate and solve the communication time minimization problem. We first
derive a tight bound to approximate the communication time through
cross-disciplinary effort involving both learning theory for convergence
analysis and communication theory for per-round latency analysis. Building on
the analytical result, an optimized probabilistic scheduling policy is derived
in closed-form by solving the approximate communication time minimization
problem. It is found that the optimized policy gradually turns its priority
from suppressing the remaining communication rounds to reducing per-round
latency as the training process evolves. The effectiveness of the proposed
scheme is demonstrated via a use case on collaborative 3D objective detection
in autonomous driving.

    

### [[2107.11677] Breath to Pair (B2P): Respiration-Based Pairing Protocol for Wearable Devices](http://arxiv.org/abs/2107.11677)


  We propose Breath to Pair (B2P), a protocol for pairing and shared-key
generation for wearable devices that leverages the wearer's respiration
activity to ensure that the devices are part of the same body-area network. We
assume that the devices exploit different types of sensors to extract and
process the respiration signal. We illustrate B2P for the case of two devices
that use respiratory inductance plethysmography (RIP) and accelerometer
sensors, respectively. Allowing for different types of sensors in pairing
allows us to include wearable devices that use a variety of different sensors.
In practice, this form of sensor variety creates a number of challenges that
limit the ability of the shared-key establishment algorithm to generate
matching keys. The two main obstacles are the lack of synchronization across
the devices and the need for correct noise-induced mismatches between the
generated key bit-strings.
B2P addresses the synchronization challenge by utilizing Change Point
Detection (CPD) to detect abrupt changes in the respiration signal and consider
their occurrences as synchronizing points. Any potential mismatches are handled
by optimal quantization and encoding of the respiration signal in order to
maximize the error correction rate and minimize the message overheads.
Extensive evaluation on a dataset collected from 30 volunteers demonstrates
that our protocol can generate a secure 256-bit key every 2.85 seconds (around
one breathing cycle). Particular attention is given to secure B2P against
device impersonation attacks.

    

### [[2107.11685] A Survey of Wearable Devices Pairing Based on Biometric Signals](http://arxiv.org/abs/2107.11685)


  With the growth of wearable devices, which are usually constrained in
computational power and user interface, this pairing has to be autonomous.
Considering devices that do not have prior information about each other, a
secure communication should be established by generating a shared secret key
derived from a common context between the devices. Context-based pairing
solutions increase the usability of wearable device pairing by eliminating any
human involvement in the pairing process. This is possible by utilizing onboard
sensors (with the same sensing modalities) to capture a common physical context
(e.g., body motion, gait, heartbeat, respiration, and EMG signal). A wide range
of approaches has been proposed to address autonomous pairing in wearable
devices. This paper surveys context-based pairing in wearable devices by
focusing on the signals and sensors exploited. We review the steps needed for
generating a common key and provide a survey of existing techniques utilized in
each step.

    

### [[2107.11966] When SRv6 meets 5G Core: Implementation and Deployment of a Network Service Chaining Function in SmartNICs](http://arxiv.org/abs/2107.11966)


  Currently, we have witnessed a myriad of solutions that benefit from
programmable hardware. The 5G Core (5GC) can and should also benefit from such
paradigm to offload certain functions to the dataplane. In this work, we
designed and implemented a P4-based solution for traffic identification and
chaining using the Netronome Agilo SmartNIC. The solution here presented is
deployed in-between the RAN and UPF (User Plane Function) so that traffic
coming from the RAN is identified and chained using SRv6 based on different
rules defined by the control plane. The traffic identification and the
construction of the SRv6 list of segments are done entirely in the SmartNIC. A
minimalist Proof-of-Concept (PoC) was deployed and evaluated to show that this
function is perfectly capable to build service function chainings in a
transparent and efficient way.

    

### [[2107.12068] Virtual Drive-Tests: A Case for Predicting QoE in Adaptive Video Streaming](http://arxiv.org/abs/2107.12068)


  Intelligent and autonomous troubleshooting is a crucial enabler for the
current 5G and future 6G networks. In this work, we develop a flexible
architecture for detecting anomalies in adaptive video streaming comprising
three main components: i) A pattern recognizer that learns a typical pattern
for video quality from the client-side application traces of a specific
reference video, ii) A predictor for mapping Radio Frequency (RF) performance
indicators collected on the network-side using user-based traces to a video
quality measure, iii) An anomaly detector for comparing the predicted video
quality pattern with the typical pattern to identify anomalies. We use real
network traces (i.e., on-device measurements) collected in different
geographical locations and at various times of day to train our machine
learning models. We perform extensive numerical analysis to demonstrate key
parameters impacting correct video quality prediction and anomaly detection. In
particular, we have shown that the video playback time is the most crucial
parameter determining the video quality since buffering continues during the
playback and resulting in better video quality further into the playback.
However, we also reveal that RF performance indicators characterizing the
quality of the cellular connectivity are required to correctly predict QoE in
anomalous cases. Then, we have exhibited that the mean maximum F1-score of our
method is 77%, verifying the efficacy of our models. Our architecture is
flexible and autonomous, so one can apply it to -- and operate with -- other
user applications as long as the relevant user-based traces are available.

    

### [[2107.12193] An Efficient Internet Traffic Classification System Using Deep Learning for IoT](http://arxiv.org/abs/2107.12193)


  Internet of Things (IoT) defines a network of devices connected to the
internet and sharing a massive amount of data between each other and a central
location. These IoT devices are connected to a network therefore prone to
attacks. Various management tasks and network operations such as security,
intrusion detection, Quality-of-Service provisioning, performance monitoring,
resource provisioning, and traffic engineering require traffic classification.
Due to the ineffectiveness of traditional classification schemes, such as
port-based and payload-based methods, researchers proposed machine
learning-based traffic classification systems based on shallow neural networks.
Furthermore, machine learning-based models incline to misclassify internet
traffic due to improper feature selection. In this research, an efficient
multilayer deep learning based classification system is presented to overcome
these challenges that can classify internet traffic. To examine the performance
of the proposed technique, Moore-dataset is used for training the classifier.
The proposed scheme takes the pre-processed data and extracts the flow features
using a deep neural network (DNN). In particular, the maximum entropy
classifier is used to classify the internet traffic. The experimental results
show that the proposed hybrid deep learning algorithm is effective and achieved
high accuracy for internet traffic classification, i.e., 99.23%. Furthermore,
the proposed algorithm achieved the highest accuracy compared to the support
vector machine (SVM) based classification technique and k-nearest neighbours
(KNNs) based classification technique.

    

### [[2009.03575] NC-MOPSO: Network centrality guided multi-objective particle swarm optimization for transport optimization on networks](http://arxiv.org/abs/2009.03575)


  Transport processes are universal in real-world complex networks, such as
communication and transportation networks. As the increase of the traffic in
these complex networks, problems like traffic congestion and transport delay
are becoming more and more serious, which call for a systematic optimization of
these networks. In this paper, we formulate a multi-objective optimization
problem (MOP) to deal with the enhancement of network capacity and efficiency
simultaneously, by appropriately adjusting the weights of edges in networks. To
solve this problem, we provide a multi-objective evolutionary algorithm (MOEA)
based on particle swarm optimization (PSO), namely network centrality guided
multi-objective PSO (NC-MOPSO). Specifically, in the framework of PSO, we
propose a hybrid population initialization mechanism and a local search
strategy by employing the network centrality theory to enhance the quality of
initial solutions and strengthen the exploration of the search space,
respectively. Simulation experiments performed on network models and real
networks show that our algorithm has better performance than four
state-of-the-art alternatives on several most-used metrics.

    

### [[2011.05708] Optimizing AI Service Placement and Resource Allocation in Mobile Edge Intelligence Systems](http://arxiv.org/abs/2011.05708)


  Leveraging recent advances on mobile edge computing (MEC), edge intelligence
has emerged as a promising paradigm to support mobile artificial intelligence
(AI) applications at the network edge. In this paper, we consider the AI
service placement problem in a multi-user MEC system, where the access point
(AP) places the most up-to-date AI program at user devices to enable local
computing/task execution at the user side. To fully utilize the stringent
wireless spectrum and edge computing resources, the AP sends the AI service
program to a user only when enabling local computing at the user yields a
better system performance. We formulate a mixed-integer non-linear programming
(MINLP) problem to minimize the total computation time and energy consumption
of all users by jointly optimizing the service placement (i.e., which users to
receive the program) and resource allocation (on local CPU frequencies, uplink
bandwidth, and edge CPU frequency). To tackle the MINLP problem, we derive
analytical expressions to calculate the optimal resource allocation decisions
with low complexity. This allows us to efficiently obtain the optimal service
placement solution by search-based algorithms such as meta-heuristic or greedy
search algorithms. To enhance the algorithm scalability in large-sized
networks, we further propose an ADMM (alternating direction method of
multipliers) based method to decompose the optimization problem into parallel
tractable MINLP subproblems. The ADMM method eliminates the need of searching
in a high-dimensional space for service placement decisions and thus has a low
computational complexity that grows linearly with the number of users.
Simulation results show that the proposed algorithms perform extremely close to
the optimum and significantly outperform the other representative benchmark
algorithms.

    

### [[2102.10749] CSIT-Free Model Aggregation for Federated Edge Learning via Reconfigurable Intelligent Surface](http://arxiv.org/abs/2102.10749)


  We study over-the-air model aggregation in federated edge learning (FEEL)
systems, where channel state information at the transmitters (CSIT) is assumed
to be unavailable. We leverage the reconfigurable intelligent surface (RIS)
technology to align the cascaded channel coefficients for CSIT-free model
aggregation. To this end, we jointly optimize the RIS and the receiver by
minimizing the aggregation error under the channel alignment constraint. We
then develop a difference-of-convex algorithm for the resulting non-convex
optimization. Numerical experiments on image classification show that the
proposed method is able to achieve a similar learning accuracy as the
state-of-the-art CSIT-based solution, demonstrating the efficiency of our
approach in combating the lack of CSIT.

    

### [[2105.02510] Towards Inference Delivery Networks: Distributing Machine Learning with Optimality Guarantees](http://arxiv.org/abs/2105.02510)


  An increasing number of applications rely on complex inference tasks that are
based on machine learning (ML). Currently, there are two options to run such
tasks: either they are served directly by the end device (e.g., smartphones,
IoT equipment, smart vehicles), or offloaded to a remote cloud. Both options
may be unsatisfactory for many applications: local models may have inadequate
accuracy, while the cloud may fail to meet delay constraints. In this paper, we
present the novel idea of \emph{inference delivery networks} (IDNs), networks
of computing nodes that coordinate to satisfy ML inference requests achieving
the best trade-off between latency and accuracy. IDNs bridge the dichotomy
between device and cloud execution by integrating inference delivery at the
various tiers of the infrastructure continuum (access, edge, regional data
center, cloud). We propose a distributed dynamic policy for ML model allocation
in an IDN by which each node dynamically updates its local set of inference
models based on requests observed during the recent past plus limited
information exchange with its neighboring nodes. Our policy offers strong
performance guarantees in an adversarial setting and shows improvements over
greedy heuristics with similar complexity in realistic scenarios.

    

### [[2105.09389] Stochastic Coordination in Heterogeneous Load Balancing Systems](http://arxiv.org/abs/2105.09389)


  Current-day data centers and high-volume cloud services employ a broad set of
heterogeneous servers. In such settings, client requests typically arrive at
multiple entry points, and dispatching them to servers is an urgent distributed
systems problem. This paper presents an efficient solution to the load
balancing problem in such systems that improves on and overcomes problems of
previous solutions. The load balancing problem is formulated as a stochastic
optimization problem, and an efficient algorithmic solution is obtained based
on a subtle mathematical analysis of the problem. Finally, extensive evaluation
of the solution on simulated data shows that it outperforms previous solutions.
Moreover, the resulting dispatching policy can be computed very efficiently,
making the solution practically viable.

    

### [[2107.11381] TargetNet: Functional microRNA Target Prediction with Deep Neural Networks](http://arxiv.org/abs/2107.11381)


  MicroRNAs (miRNAs) play pivotal roles in gene expression regulation by
binding to target sites of messenger RNAs (mRNAs). While identifying functional
targets of miRNAs is of utmost importance, their prediction remains a great
challenge. Previous computational algorithms have major limitations. They use
conservative candidate target site (CTS) selection criteria mainly focusing on
canonical site types, rely on laborious and time-consuming manual feature
extraction, and do not fully capitalize on the information underlying miRNA-CTS
interactions. In this paper, we introduce TargetNet, a novel deep
learning-based algorithm for functional miRNA target prediction. To address the
limitations of previous approaches, TargetNet has three key components: (1)
relaxed CTS selection criteria accommodating irregularities in the seed region,
(2) a novel miRNA-CTS sequence encoding scheme incorporating extended seed
region alignments, and (3) a deep residual network-based prediction model. The
proposed model was trained with miRNA-CTS pair datasets and evaluated with
miRNA-mRNA pair datasets. TargetNet advances the previous state-of-the-art
algorithms used in functional miRNA target classification. Furthermore, it
demonstrates great potential for distinguishing high-functional miRNA targets.

    

### [[2107.11400] Robust Explainability: A Tutorial on Gradient-Based Attribution Methods for Deep Neural Networks](http://arxiv.org/abs/2107.11400)


  With the rise of deep neural networks, the challenge of explaining the
predictions of these networks has become increasingly recognized. While many
methods for explaining the decisions of deep neural networks exist, there is
currently no consensus on how to evaluate them. On the other hand, robustness
is a popular topic for deep learning research; however, it is hardly talked
about in explainability until very recently. In this tutorial paper, we start
by presenting gradient-based interpretability methods. These techniques use
gradient signals to assign the burden of the decision on the input features.
Later, we discuss how gradient-based methods can be evaluated for their
robustness and the role that adversarial robustness plays in having meaningful
explanations. We also discuss the limitations of gradient-based methods.
Finally, we present the best practices and attributes that should be examined
before choosing an explainability method. We conclude with the future
directions for research in the area at the convergence of robustness and
explainability.

    

### [[2107.11412] Using Deep Learning Techniques and Inferential Speech Statistics for AI Synthesised Speech Recognition](http://arxiv.org/abs/2107.11412)


  The recent developments in technology have re-warded us with amazing audio
synthesis models like TACOTRON and WAVENETS. On the other side, it poses
greater threats such as speech clones and deep fakes, that may go undetected.
To tackle these alarming situations, there is an urgent need to propose models
that can help discriminate a synthesized speech from an actual human speech and
also identify the source of such a synthesis. Here, we propose a model based on
Convolutional Neural Network (CNN) and Bidirectional Recurrent Neural Network
(BiRNN) that helps to achieve both the aforementioned objectives. The temporal
dependencies present in AI synthesized speech are exploited using Bidirectional
RNN and CNN. The model outperforms the state-of-the-art approaches by
classifying the AI synthesized audio from real human speech with an error rate
of 1.9% and detecting the underlying architecture with an accuracy of 97%.

    

### [[2107.11413] A Realistic Simulation Framework for Learning with Label Noise](http://arxiv.org/abs/2107.11413)


  We propose a simulation framework for generating realistic instance-dependent
noisy labels via a pseudo-labeling paradigm. We show that this framework
generates synthetic noisy labels that exhibit important characteristics of the
label noise in practical settings via comparison with the CIFAR10-H dataset.
Equipped with controllable label noise, we study the negative impact of noisy
labels across a few realistic settings to understand when label noise is more
problematic. We also benchmark several existing algorithms for learning with
noisy labels and compare their behavior on our synthetic datasets and on the
datasets with independent random label noise. Additionally, with the
availability of annotator information from our simulation framework, we propose
a new technique, Label Quality Model (LQM), that leverages annotator features
to predict and correct against noisy labels. We show that by adding LQM as a
label correction step before applying existing noisy label techniques, we can
further improve the models' performance.

    

### [[2107.11415] Device Scheduling and Update Aggregation Policies for Asynchronous Federated Learning](http://arxiv.org/abs/2107.11415)


  Federated Learning (FL) is a newly emerged decentralized machine learning
(ML) framework that combines on-device local training with server-based model
synchronization to train a centralized ML model over distributed nodes. In this
paper, we propose an asynchronous FL framework with periodic aggregation to
eliminate the straggler issue in FL systems. For the proposed model, we
investigate several device scheduling and update aggregation policies and
compare their performances when the devices have heterogeneous computation
capabilities and training data distributions. From the simulation results, we
conclude that the scheduling and aggregation design for asynchronous FL can be
rather different from the synchronous case. For example, a norm-based
significance-aware scheduling policy might not be efficient in an asynchronous
FL setting, and an appropriate "age-aware" weighting design for the model
aggregation can greatly improve the learning performance of such systems.

    

### [[2107.11419] Finite-time Analysis of Globally Nonstationary Multi-Armed Bandits](http://arxiv.org/abs/2107.11419)


  We consider nonstationary multi-armed bandit problems where the model
parameters of the arms change over time. We introduce the adaptive resetting
bandit (ADR-bandit), which is a class of bandit algorithms that leverages
adaptive windowing techniques from the data stream community. We first provide
new guarantees on the quality of estimators resulting from adaptive windowing
techniques, which are of independent interest in the data mining community.
Furthermore, we conduct a finite-time analysis of ADR-bandit in two typical
environments: an abrupt environment where changes occur instantaneously and a
gradual environment where changes occur progressively. We demonstrate that
ADR-bandit has nearly optimal performance when the abrupt or global changes
occur in a coordinated manner that we call global changes. We demonstrate that
forced exploration is unnecessary when we restrict the interest to the global
changes. Unlike the existing nonstationary bandit algorithms, ADR-bandit has
optimal performance in stationary environments as well as nonstationary
environments with global changes. Our experiments show that the proposed
algorithms outperform the existing approaches in synthetic and real-world
environments.

    

### [[2107.11433] A general sample complexity analysis of vanilla policy gradient](http://arxiv.org/abs/2107.11433)


  The policy gradient (PG) is one of the most popular methods for solving
reinforcement learning (RL) problems. However, a solid theoretical
understanding of even the "vanilla" PG has remained elusive for long time. In
this paper, we apply recent tools developed for the analysis of SGD in
non-convex optimization to obtain convergence guarantees for both REINFORCE and
GPOMDP under smoothness assumption on the objective function and weak
conditions on the second moment of the norm of the estimated gradient. When
instantiated under common assumptions on the policy space, our general result
immediately recovers existing $\widetilde{\mathcal{O}}(\epsilon^{-4})$ sample
complexity guarantees, but for wider ranges of parameters (e.g., step size and
batch size $m$) with respect to previous literature. Notably, our result
includes the single trajectory case (i.e., $m=1$) and it provides a more
accurate analysis of the dependency on problem-specific parameters by fixing
previous results available in the literature. We believe that the integration
of state-of-the-art tools from non-convex optimization may lead to identify a
much broader range of problems where PG methods enjoy strong theoretical
guarantees.

    

### [[2107.11435] HierMUD: Hierarchical Multi-task Unsupervised Domain Adaptation between Bridges for Drive-by Damage Diagnosis](http://arxiv.org/abs/2107.11435)


  Monitoring bridge health using vibrations of drive-by vehicles has various
benefits, such as no need for directly installing and maintaining sensors on
the bridge. However, many of the existing drive-by monitoring approaches are
based on supervised learning models that require labeled data from every bridge
of interest, which is expensive and time-consuming, if not impossible, to
obtain. To this end, we introduce a new framework that transfers the model
learned from one bridge to diagnose damage in another bridge without any labels
from the target bridge. Our framework trains a hierarchical neural network
model in an adversarial way to extract task-shared and task-specific features
that are informative to multiple diagnostic tasks and invariant across multiple
bridges. We evaluate our framework on experimental data collected from 2
bridges and 3 vehicles. We achieve accuracies of 95% for damage detection, 93%
for localization, and up to 72% for quantification, which are ~2 times
improvements from baseline methods.

    

### [[2107.11442] Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition](http://arxiv.org/abs/2107.11442)


  We present a novel global compression framework for deep neural networks that
automatically analyzes each layer to identify the optimal per-layer compression
ratio, while simultaneously achieving the desired overall compression. Our
algorithm hinges on the idea of compressing each convolutional (or
fully-connected) layer by slicing its channels into multiple groups and
decomposing each group via low-rank decomposition. At the core of our algorithm
is the derivation of layer-wise error bounds from the Eckart Young Mirsky
theorem. We then leverage these bounds to frame the compression problem as an
optimization problem where we wish to minimize the maximum compression error
across layers and propose an efficient algorithm towards a solution. Our
experiments indicate that our method outperforms existing low-rank compression
approaches across a wide range of networks and data sets. We believe that our
results open up new avenues for future research into the global
performance-size trade-offs of modern neural networks. Our code is available at
this https URL.

    

### [[2107.11445] Self-Repairing Neural Networks: Provable Safety for Deep Networks via Dynamic Repair](http://arxiv.org/abs/2107.11445)


  Neural networks are increasingly being deployed in contexts where safety is a
critical concern. In this work, we propose a way to construct neural network
classifiers that dynamically repair violations of non-relational safety
constraints called safe ordering properties. Safe ordering properties relate
requirements on the ordering of a network's output indices to conditions on
their input, and are sufficient to express most useful notions of
non-relational safety for classifiers. Our approach is based on a novel
self-repairing layer, which provably yields safe outputs regardless of the
characteristics of its input. We compose this layer with an existing network to
construct a self-repairing network (SR-Net), and show that in addition to
providing safe outputs, the SR-Net is guaranteed to preserve the accuracy of
the original network. Notably, our approach is independent of the size and
architecture of the network being repaired, depending only on the specified
property and the dimension of the network's output; thus it is scalable to
large state-of-the-art networks. We show that our approach can be implemented
using vectorized computations that execute efficiently on a GPU, introducing
run-time overhead of less than one millisecond on current hardware -- even on
large, widely-used networks containing hundreds of thousands of neurons and
millions of parameters.

    

### [[2107.11453] Automatic Detection Of Noise Events at Shooting Range Using Machine Learning](http://arxiv.org/abs/2107.11453)


  Outdoor shooting ranges are subject to noise regulations from local and
national authorities. Restrictions found in these regulations may include
limits on times of activities, the overall number of noise events, as well as
limits on number of events depending on the class of noise or activity. A noise
monitoring system may be used to track overall sound levels, but rarely provide
the ability to detect activity or count the number of events, required to
compare directly with such regulations. This work investigates the feasibility
and performance of an automatic detection system to count noise events. An
empirical evaluation was done by collecting data at a newly constructed
shooting range and training facility. The data includes tests of multiple
weapon configurations from small firearms to high caliber rifles and
explosives, at multiple source positions, and collected on multiple different
days. Several alternative machine learning models are tested, using as inputs
time-series of standard acoustic indicators such as A-weighted sound levels and
1/3 octave spectrogram, and classifiers such as Logistic Regression and
Convolutional Neural Networks. Performance for the various alternatives are
reported in terms of the False Positive Rate and False Negative Rate. The
detection performance was found to be satisfactory for use in automatic logging
of time-periods with training activity.

    

### [[2107.11460] Non-intrusive reduced order modeling of natural convection in porous media using convolutional autoencoders: comparison with linear subspace techniques](http://arxiv.org/abs/2107.11460)


  Natural convection in porous media is a highly nonlinear multiphysical
problem relevant to many engineering applications (e.g., the process of
$\mathrm{CO_2}$ sequestration). Here, we present a non-intrusive reduced order
model of natural convection in porous media employing deep convolutional
autoencoders for the compression and reconstruction and either radial basis
function (RBF) interpolation or artificial neural networks (ANNs) for mapping
parameters of partial differential equations (PDEs) on the corresponding
nonlinear manifolds. To benchmark our approach, we also describe linear
compression and reconstruction processes relying on proper orthogonal
decomposition (POD) and ANNs. We present comprehensive comparisons among
different models through three benchmark problems. The reduced order models,
linear and nonlinear approaches, are much faster than the finite element model,
obtaining a maximum speed-up of $7 \times 10^{6}$ because our framework is not
bound by the Courant-Friedrichs-Lewy condition; hence, it could deliver
quantities of interest at any given time contrary to the finite element model.
Our model's accuracy still lies within a mean squared error of 0.07 (two-order
of magnitude lower than the maximum value of the finite element results) in the
worst-case scenario. We illustrate that, in specific settings, the nonlinear
approach outperforms its linear counterpart and vice versa. We hypothesize that
a visual comparison between principal component analysis (PCA) or t-Distributed
Stochastic Neighbor Embedding (t-SNE) could indicate which method will perform
better prior to employing any specific compression strategy.

    

### [[2107.11468] Using a Cross-Task Grid of Linear Probes to Interpret CNN Model Predictions On Retinal Images](http://arxiv.org/abs/2107.11468)


  We analyze a dataset of retinal images using linear probes: linear regression
models trained on some "target" task, using embeddings from a deep
convolutional (CNN) model trained on some "source" task as input. We use this
method across all possible pairings of 93 tasks in the UK Biobank dataset of
retinal images, leading to ~164k different models. We analyze the performance
of these linear probes by source and target task and by layer depth. We observe
that representations from the middle layers of the network are more
generalizable. We find that some target tasks are easily predicted irrespective
of the source task, and that some other target tasks are more accurately
predicted from correlated source tasks than from embeddings trained on the same
task.

    

### [[2107.11472] Free Hyperbolic Neural Networks with Limited Radii](http://arxiv.org/abs/2107.11472)


  Non-Euclidean geometry with constant negative curvature, i.e., hyperbolic
space, has attracted sustained attention in the community of machine learning.
Hyperbolic space, owing to its ability to embed hierarchical structures
continuously with low distortion, has been applied for learning data with
tree-like structures. Hyperbolic Neural Networks (HNNs) that operate directly
in hyperbolic space have also been proposed recently to further exploit the
potential of hyperbolic representations. While HNNs have achieved better
performance than Euclidean neural networks (ENNs) on datasets with implicit
hierarchical structure, they still perform poorly on standard classification
benchmarks such as CIFAR and ImageNet. The traditional wisdom is that it is
critical for the data to respect the hyperbolic geometry when applying HNNs. In
this paper, we first conduct an empirical study showing that the inferior
performance of HNNs on standard recognition datasets can be attributed to the
notorious vanishing gradient problem. We further discovered that this problem
stems from the hybrid architecture of HNNs. Our analysis leads to a simple yet
effective solution called Feature Clipping, which regularizes the hyperbolic
embedding whenever its norm exceeding a given threshold. Our thorough
experiments show that the proposed method can successfully avoid the vanishing
gradient problem when training HNNs with backpropagation. The improved HNNs are
able to achieve comparable performance with ENNs on standard image recognition
datasets including MNIST, CIFAR10, CIFAR100 and ImageNet, while demonstrating
more adversarial robustness and stronger out-of-distribution detection
capability.

    

### [[2107.11496] Training multi-objective/multi-task collocation physics-informed neural network with student/teachers transfer learnings](http://arxiv.org/abs/2107.11496)


  This paper presents a PINN training framework that employs (1) pre-training
steps that accelerates and improve the robustness of the training of
physics-informed neural network with auxiliary data stored in point clouds, (2)
a net-to-net knowledge transfer algorithm that improves the weight
initialization of the neural network and (3) a multi-objective optimization
algorithm that may improve the performance of a physical-informed neural
network with competing constraints. We consider the training and transfer and
multi-task learning of physics-informed neural network (PINN) as
multi-objective problems where the physics constraints such as the governing
equation, boundary conditions, thermodynamic inequality, symmetry, and
invariant properties, as well as point cloud used for pre-training can
sometimes lead to conflicts and necessitating the seek of the Pareto optimal
solution. In these situations, weighted norms commonly used to handle multiple
constraints may lead to poor performance, while other multi-objective
algorithms may scale poorly with increasing dimensionality. To overcome this
technical barrier, we adopt the concept of vectorized objective function and
modify a gradient descent approach to handle the issue of conflicting
gradients. Numerical experiments are compared the benchmark boundary value
problems solved via PINN. The performance of the proposed paradigm is compared
against the classical equal-weighted norm approach. Our numerical experiments
indicate that the brittleness and lack of robustness demonstrated in some PINN
implementations can be overcome with the proposed strategy.

    

### [[2107.11500] $μ$DARTS: Model Uncertainty-Aware Differentiable Architecture Search](http://arxiv.org/abs/2107.11500)


  We present a Model Uncertainty-aware Differentiable ARchiTecture Search
($\mu$DARTS) that optimizes neural networks to simultaneously achieve high
accuracy and low uncertainty. We introduce concrete dropout within DARTS cells
and include a Monte-Carlo regularizer within the training loss to optimize the
concrete dropout probabilities. A predictive variance term is introduced in the
validation loss to enable searching for architecture with minimal model
uncertainty. The experiments on CIFAR10, CIFAR100, SVHN, and ImageNet verify
the effectiveness of $\mu$DARTS in improving accuracy and reducing uncertainty
compared to existing DARTS methods. Moreover, the final architecture obtained
from $\mu$DARTS shows higher robustness to noise at the input image and model
parameters compared to the architecture obtained from existing DARTS methods.

    

### [[2107.11508] Imbalanced Big Data Oversampling: Taxonomy, Algorithms, Software, Guidelines and Future Directions](http://arxiv.org/abs/2107.11508)


  Learning from imbalanced data is among the most challenging areas in
contemporary machine learning. This becomes even more difficult when considered
the context of big data that calls for dedicated architectures capable of
high-performance processing. Apache Spark is a highly efficient and popular
architecture, but it poses specific challenges for algorithms to be implemented
for it. While oversampling algorithms are an effective way for handling class
imbalance, they have not been designed for distributed environments. In this
paper, we propose a holistic look on oversampling algorithms for imbalanced big
data. We discuss the taxonomy of oversampling algorithms and their mechanisms
used to handle skewed class distributions. We introduce a Spark library with 14
state-of-the-art oversampling algorithms implemented and evaluate their
efficacy via extensive experimental study. Using binary and multi-class massive
data sets, we analyze the effectiveness of oversampling algorithms and their
relationships with different types of classifiers. We evaluate the trade-off
between accuracy and time complexity of oversampling algorithms, as well as
their scalability when increasing the size of data. This allows us to gain
insight into the usefulness of specific components of oversampling algorithms
for big data, as well as formulate guidelines and recommendations for designing
future resampling approaches for massive imbalanced data. Our library can be
downloaded from this https URL.

    

### [[2107.11526] On the Sample Complexity of Privately Learning Axis-Aligned Rectangles](http://arxiv.org/abs/2107.11526)


  We revisit the fundamental problem of learning Axis-Aligned-Rectangles over a
finite grid $X^d\subseteq{\mathbb{R}}^d$ with differential privacy. Existing
results show that the sample complexity of this problem is at most $\min\left\{
d{\cdot}\log|X| \;,\; d^{1.5}{\cdot}\left(\log^*|X| \right)^{1.5}\right\}$.
That is, existing constructions either require sample complexity that grows
linearly with $\log|X|$, or else it grows super linearly with the dimension
$d$. We present a novel algorithm that reduces the sample complexity to only
$\tilde{O}\left\{d{\cdot}\left(\log^*|X|\right)^{1.5}\right\}$, attaining a
dimensionality optimal dependency without requiring the sample complexity to
grow with $\log|X|$.The technique used in order to attain this improvement
involves the deletion of "exposed" data-points on the go, in a fashion designed
to avoid the cost of the adaptive composition theorems. The core of this
technique may be of individual interest, introducing a new method for
constructing statistically-efficient private algorithms.

    

### [[2107.11533] Combining Online Learning and Offline Learning for Contextual Bandits with Deficient Support](http://arxiv.org/abs/2107.11533)


  We address policy learning with logged data in contextual bandits. Current
offline-policy learning algorithms are mostly based on inverse propensity score
(IPS) weighting requiring the logging policy to have \emph{full support} i.e. a
non-zero probability for any context/action of the evaluation policy. However,
many real-world systems do not guarantee such logging policies, especially when
the action space is large and many actions have poor or missing rewards. With
such \emph{support deficiency}, the offline learning fails to find optimal
policies. We propose a novel approach that uses a hybrid of offline learning
with online exploration. The online exploration is used to explore unsupported
actions in the logged data whilst offline learning is used to exploit supported
actions from the logged data avoiding unnecessary explorations. Our approach
determines an optimal policy with theoretical guarantees using the minimal
number of online explorations. We demonstrate our algorithms' effectiveness
empirically on a diverse collection of datasets.

    

### [[2107.11585] Two Headed Dragons: Multimodal Fusion and Cross Modal Transactions](http://arxiv.org/abs/2107.11585)


  As the field of remote sensing is evolving, we witness the accumulation of
information from several modalities, such as multispectral (MS), hyperspectral
(HSI), LiDAR etc. Each of these modalities possess its own distinct
characteristics and when combined synergistically, perform very well in the
recognition and classification tasks. However, fusing multiple modalities in
remote sensing is cumbersome due to highly disparate domains. Furthermore, the
existing methods do not facilitate cross-modal interactions. To this end, we
propose a novel transformer based fusion method for HSI and LiDAR modalities.
The model is composed of stacked auto encoders that harness the cross key-value
pairs for HSI and LiDAR, thus establishing a communication between the two
modalities, while simultaneously using the CNNs to extract the spectral and
spatial information from HSI and LiDAR. We test our model on Houston (Data
Fusion Contest - 2013) and MUUFL Gulfport datasets and achieve competitive
results.

    

### [[2107.11587] Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?](http://arxiv.org/abs/2107.11587)


  We contribute to micro-data model-based reinforcement learning (MBRL) by
rigorously comparing popular generative models using a fixed (random shooting)
control agent. We find that on an environment that requires multimodal
posterior predictives, mixture density nets outperform all other models by a
large margin. When multimodality is not required, our surprising finding is
that we do not need probabilistic posterior predictives: deterministic models
are on par, in fact they consistently (although non-significantly) outperform
their probabilistic counterparts. We also found that heteroscedasticity at
training time, perhaps acting as a regularizer, improves predictions at longer
horizons. At the methodological side, we design metrics and an experimental
protocol which can be used to evaluate the various models, predicting their
asymptotic performance when using them on the control problem. Using this
framework, we improve the state-of-the-art sample complexity of MBRL on Acrobot
by two to four folds, using an aggressive training schedule which is outside of
the hyperparameter interval usually considered

    

### [[2107.11598] Combining Graph Neural Networks with Expert Knowledge for Smart Contract Vulnerability Detection](http://arxiv.org/abs/2107.11598)


  Smart contract vulnerability detection draws extensive attention in recent
years due to the substantial losses caused by hacker attacks. Existing efforts
for contract security analysis heavily rely on rigid rules defined by experts,
which are labor-intensive and non-scalable. More importantly, expert-defined
rules tend to be error-prone and suffer the inherent risk of being cheated by
crafty attackers. Recent researches focus on the symbolic execution and formal
analysis of smart contracts for vulnerability detection, yet to achieve a
precise and scalable solution. Although several methods have been proposed to
detect vulnerabilities in smart contracts, there is still a lack of effort that
considers combining expert-defined security patterns with deep neural networks.
In this paper, we explore using graph neural networks and expert knowledge for
smart contract vulnerability detection. Specifically, we cast the rich control-
and data- flow semantics of the source code into a contract graph. To highlight
the critical nodes in the graph, we further design a node elimination phase to
normalize the graph. Then, we propose a novel temporal message propagation
network to extract the graph feature from the normalized graph, and combine the
graph feature with designed expert patterns to yield a final detection system.
Extensive experiments are conducted on all the smart contracts that have source
code in Ethereum and VNT Chain platforms. Empirical results show significant
accuracy improvements over the state-of-the-art methods on three types of
vulnerabilities, where the detection accuracy of our method reaches 89.15%,
89.02%, and 83.21% for reentrancy, timestamp dependence, and infinite loop
vulnerabilities, respectively.

    

### [[2107.11609] A Model-Agnostic Algorithm for Bayes Error Determination in Binary Classification](http://arxiv.org/abs/2107.11609)


  This paper presents the intrinsic limit determination algorithm (ILD
Algorithm), a novel technique to determine the best possible performance,
measured in terms of the AUC (area under the ROC curve) and accuracy, that can
be obtained from a specific dataset in a binary classification problem with
categorical features {\sl regardless} of the model used. This limit, namely the
Bayes error, is completely independent of any model used and describes an
intrinsic property of the dataset. The ILD algorithm thus provides important
information regarding the prediction limits of any binary classification
algorithm when applied to the considered dataset. In this paper the algorithm
is described in detail, its entire mathematical framework is presented and the
pseudocode is given to facilitate its implementation. Finally, an example with
a real dataset is given.

    

### [[2107.11621] FedLab: A Flexible Federated Learning Framework](http://arxiv.org/abs/2107.11621)


  Federated learning (FL) is a solution for privacy challenge, which allows
multiparty to train a shared model without violating privacy protection
regulations. Many excellent works of FL have been proposed in recent years. To
help researchers verify their ideas in FL, we designed and developed FedLab, a
flexible and modular FL framework based on PyTorch. In this paper, we will
introduce architecture and features of FedLab. For current popular research
points: optimization and communication compression, FedLab provides functional
interfaces and a series of baseline implementation are available, making
researchers quickly implement ideas. In addition, FedLab is scale-able in both
client simulation and distributed communication.

    

### [[2107.11625] Discrete Denoising Flows](http://arxiv.org/abs/2107.11625)


  Discrete flow-based models are a recently proposed class of generative models
that learn invertible transformations for discrete random variables. Since they
do not require data dequantization and maximize an exact likelihood objective,
they can be used in a straight-forward manner for lossless compression. In this
paper, we introduce a new discrete flow-based model for categorical random
variables: Discrete Denoising Flows (DDFs). In contrast with other discrete
flow-based models, our model can be locally trained without introducing
gradient bias. We show that DDFs outperform Discrete Flows on modeling a toy
example, binary MNIST and Cityscapes segmentation maps, measured in
log-likelihood.

    

### [[2107.11630] Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them](http://arxiv.org/abs/2107.11630)


  Making classifiers robust to adversarial examples is hard. Thus, many
defenses tackle the seemingly easier task of detecting perturbed inputs. We
show a barrier towards this goal. We prove a general hardness reduction between
detection and classification of adversarial examples: given a robust detector
for attacks at distance {\epsilon} (in some metric), we can build a similarly
robust (but inefficient) classifier for attacks at distance {\epsilon}/2. Our
reduction is computationally inefficient, and thus cannot be used to build
practical classifiers. Instead, it is a useful sanity check to test whether
empirical detection results imply something much stronger than the authors
presumably anticipated. To illustrate, we revisit 13 detector defenses. For
11/13 cases, we show that the claimed detection results would imply an
inefficient classifier with robustness far beyond the state-of-the-art.

    

### [[2107.11640] Deep Machine Learning Based Egyptian Vehicle License Plate Recognition Systems](http://arxiv.org/abs/2107.11640)


  Automated Vehicle License Plate (VLP) detection and recognition have ended up
being a significant research issue as of late. VLP localization and recognition
are some of the most essential techniques for managing traffic using digital
techniques. In this paper, four smart systems are developed to recognize
Egyptian vehicles license plates. Two systems are based on character
recognition, which are (System1, Characters Recognition with Classical Machine
Learning) and (System2, Characters Recognition with Deep Machine Learning). The
other two systems are based on the whole plate recognition which are (System3,
Whole License Plate Recognition with Classical Machine Learning) and (System4,
Whole License Plate Recognition with Deep Machine Learning). We use object
detection algorithms, and machine learning based object recognition algorithms.
The performance of the developed systems has been tested on real images, and
the experimental results demonstrate that the best detection accuracy rate for
VLP is provided by using the deep learning method. Where the VLP detection
accuracy rate is better than the classical system by 32%. However, the best
detection accuracy rate for Vehicle License Plate Arabic Character (VLPAC) is
provided by using the classical method. Where VLPAC detection accuracy rate is
better than the deep learning-based system by 6%. Also, the results show that
deep learning is better than the classical technique used in VLP recognition
processes. Where the recognition accuracy rate is better than the classical
system by 8%. Finally, the paper output recommends a robust VLP recognition
system based on both statistical and deep machine learning.

    

### [[2107.11658] Tail of Distribution GAN (TailGAN): Generative- Adversarial-Network-Based Boundary Formation](http://arxiv.org/abs/2107.11658)


  Generative Adversarial Networks (GAN) are a powerful methodology and can be
used for unsupervised anomaly detection, where current techniques have
limitations such as the accurate detection of anomalies near the tail of a
distribution. GANs generally do not guarantee the existence of a probability
density and are susceptible to mode collapse, while few GANs use likelihood to
reduce mode collapse. In this paper, we create a GAN-based tail formation model
for anomaly detection, the Tail of distribution GAN (TailGAN), to generate
samples on the tail of the data distribution and detect anomalies near the
support boundary. Using TailGAN, we leverage GANs for anomaly detection and use
maximum entropy regularization. Using GANs that learn the probability of the
underlying distribution has advantages in improving the anomaly detection
methodology by allowing us to devise a generator for boundary samples, and use
this model to characterize anomalies. TailGAN addresses supports with disjoint
components and achieves competitive performance on images. We evaluate TailGAN
for identifying Out-of-Distribution (OoD) data and its performance evaluated on
MNIST, CIFAR-10, Baggage X-Ray, and OoD data shows competitiveness compared to
methods from the literature.

    

### [[2107.11662] Inference of collective Gaussian hidden Markov models](http://arxiv.org/abs/2107.11662)


  We consider inference problems for a class of continuous state collective
hidden Markov models, where the data is recorded in aggregate (collective) form
generated by a large population of individuals following the same dynamics. We
propose an aggregate inference algorithm called collective Gaussian
forward-backward algorithm, extending recently proposed Sinkhorn belief
propagation algorithm to models characterized by Gaussian densities. Our
algorithm enjoys convergence guarantee. In addition, it reduces to the standard
Kalman filter when the observations are generated by a single individual. The
efficacy of the proposed algorithm is demonstrated through multiple
experiments.

    

### [[2107.11666] Graph Convolutional Network with Generalized Factorized Bilinear Aggregation](http://arxiv.org/abs/2107.11666)


  Although Graph Convolutional Networks (GCNs) have demonstrated their power in
various applications, the graph convolutional layers, as the most important
component of GCN, are still using linear transformations and a simple pooling
step. In this paper, we propose a novel generalization of Factorized Bilinear
(FB) layer to model the feature interactions in GCNs. FB performs two
matrix-vector multiplications, that is, the weight matrix is multiplied with
the outer product of the vector of hidden features from both sides. However,
the FB layer suffers from the quadratic number of coefficients, overfitting and
the spurious correlations due to correlations between channels of hidden
representations that violate the i.i.d. assumption. Thus, we propose a compact
FB layer by defining a family of summarizing operators applied over the
quadratic term. We analyze proposed pooling operators and motivate their use.
Our experimental results on multiple datasets demonstrate that the GFB-GCN is
competitive with other methods for text classification.

    

### [[2107.11671] Adversarial training may be a double-edged sword](http://arxiv.org/abs/2107.11671)


  Adversarial training has been shown as an effective approach to improve the
robustness of image classifiers against white-box attacks. However, its
effectiveness against black-box attacks is more nuanced. In this work, we
demonstrate that some geometric consequences of adversarial training on the
decision boundary of deep networks give an edge to certain types of black-box
attacks. In particular, we define a metric called robustness gain to show that
while adversarial training is an effective method to dramatically improve the
robustness in white-box scenarios, it may not provide such a good robustness
gain against the more realistic decision-based black-box attacks. Moreover, we
show that even the minimal perturbation white-box attacks can converge faster
against adversarially-trained neural networks compared to the regular ones.

    

### [[2107.11676] The Impact of Negative Sampling on Contrastive Structured World Models](http://arxiv.org/abs/2107.11676)


  World models trained by contrastive learning are a compelling alternative to
autoencoder-based world models, which learn by reconstructing pixel states. In
this paper, we describe three cases where small changes in how we sample
negative states in the contrastive loss lead to drastic changes in model
performance. In previously studied Atari datasets, we show that leveraging time
step correlations can double the performance of the Contrastive Structured
World Model. We also collect a full version of the datasets to study
contrastive learning under a more diverse set of experiences.

    

### [[2107.11678] Deep-learning-driven Reliable Single-pixel Imaging with Uncertainty Approximation](http://arxiv.org/abs/2107.11678)


  Single-pixel imaging (SPI) has the advantages of high-speed acquisition over
a broad wavelength range and system compactness, which are difficult to achieve
by conventional imaging sensors. However, a common challenge is low image
quality arising from undersampling. Deep learning (DL) is an emerging and
powerful tool in computational imaging for many applications and researchers
have applied DL in SPI to achieve higher image quality than conventional
reconstruction approaches. One outstanding challenge, however, is that the
accuracy of DL predictions in SPI cannot be assessed in practical applications
where the ground truths are unknown. Here, we propose the use of the Bayesian
convolutional neural network (BCNN) to approximate the uncertainty (coming from
finite training data and network model) of the DL predictions in SPI. Each
pixel in the predicted result from BCNN represents the parameter of a
probability distribution rather than the image intensity value. Then, the
uncertainty can be approximated with BCNN by minimizing a negative
log-likelihood loss function in the training stage and Monte Carlo dropout in
the prediction stage. The results show that the BCNN can reliably approximate
the uncertainty of the DL predictions in SPI with varying compression ratios
and noise levels. The predicted uncertainty from BCNN in SPI reveals that most
of the reconstruction errors in deep-learning-based SPI come from the edges of
the image features. The results show that the proposed BCNN can provide a
reliable tool to approximate the uncertainty of DL predictions in SPI and can
be widely used in many applications of SPI.

    

### [[2107.11707] Boosting Video Captioning with Dynamic Loss Network](http://arxiv.org/abs/2107.11707)


  Video captioning is one of the challenging problems at the intersection of
vision and language, having many real-life applications in video retrieval,
video surveillance, assisting visually challenged people, Human-machine
interface, and many more. Recent deep learning-based methods have shown
promising results but are still on the lower side than other vision tasks (such
as image classification, object detection). A significant drawback with
existing video captioning methods is that they are optimized over cross-entropy
loss function, which is uncorrelated to the de facto evaluation metrics (BLEU,
METEOR, CIDER, ROUGE).In other words, cross-entropy is not a proper surrogate
of the true loss function for video captioning. This paper addresses the
drawback by introducing a dynamic loss network (DLN), which provides an
additional feedback signal that directly reflects the evaluation metrics. Our
results on Microsoft Research Video Description Corpus (MSVD) and MSR-Video to
Text (MSRVTT) datasets outperform previous methods.

    

### [[2107.11712] Efficient inference of interventional distributions](http://arxiv.org/abs/2107.11712)


  We consider the problem of efficiently inferring interventional distributions
in a causal Bayesian network from a finite number of observations. Let
$\mathcal{P}$ be a causal model on a set $\mathbf{V}$ of observable variables
on a given causal graph $G$. For sets $\mathbf{X},\mathbf{Y}\subseteq
\mathbf{V}$, and setting ${\bf x}$ to $\mathbf{X}$, let $P_{\bf x}(\mathbf{Y})$
denote the interventional distribution on $\mathbf{Y}$ with respect to an
intervention ${\bf x}$ to variables ${\bf x}$. Shpitser and Pearl (AAAI 2006),
building on the work of Tian and Pearl (AAAI 2001), gave an exact
characterization of the class of causal graphs for which the interventional
distribution $P_{\bf x}({\mathbf{Y}})$ can be uniquely determined. We give the
first efficient version of the Shpitser-Pearl algorithm. In particular, under
natural assumptions, we give a polynomial-time algorithm that on input a causal
graph $G$ on observable variables $\mathbf{V}$, a setting ${\bf x}$ of a set
$\mathbf{X} \subseteq \mathbf{V}$ of bounded size, outputs succinct
descriptions of both an evaluator and a generator for a distribution $\hat{P}$
that is $\varepsilon$-close (in total variation distance) to $P_{\bf
x}({\mathbf{Y}})$ where $Y=\mathbf{V}\setminus \mathbf{X}$, if $P_{\bf
x}(\mathbf{Y})$ is identifiable. We also show that when $\mathbf{Y}$ is an
arbitrary set, there is no efficient algorithm that outputs an evaluator of a
distribution that is $\varepsilon$-close to $P_{\bf x}({\mathbf{Y}})$ unless
all problems that have statistical zero-knowledge proofs, including the Graph
Isomorphism problem, have efficient randomized algorithms.

    

### [[2107.11717] Invariance-based Multi-Clustering of Latent Space Embeddings for Equivariant Learning](http://arxiv.org/abs/2107.11717)


  Variational Autoencoders (VAEs) have been shown to be remarkably effective in
recovering model latent spaces for several computer vision tasks. However,
currently trained VAEs, for a number of reasons, seem to fall short in learning
invariant and equivariant clusters in latent space. Our work focuses on
providing solutions to this problem and presents an approach to disentangle
equivariance feature maps in a Lie group manifold by enforcing deep,
group-invariant learning. Simultaneously implementing a novel separation of
semantic and equivariant variables of the latent space representation, we
formulate a modified Evidence Lower BOund (ELBO) by using a mixture model pdf
like Gaussian mixtures for invariant cluster embeddings that allows superior
unsupervised variational clustering. Our experiments show that this model
effectively learns to disentangle the invariant and equivariant representations
with significant improvements in the learning rate and an observably superior
image recognition and canonical state reconstruction compared to the currently
best deep learning models.

    

### [[2107.11722] Learning Risk-aware Costmaps for Traversability in Challenging Environments](http://arxiv.org/abs/2107.11722)


  One of the main challenges in autonomous robotic exploration and navigation
in unknown and unstructured environments is determining where the robot can or
cannot safely move. A significant source of difficulty in this determination
arises from stochasticity and uncertainty, coming from localization error,
sensor sparsity and noise, difficult-to-model robot-ground interactions, and
disturbances to the motion of the vehicle. Classical approaches to this problem
rely on geometric analysis of the surrounding terrain, which can be prone to
modeling errors and can be computationally expensive. Moreover, modeling the
distribution of uncertain traversability costs is a difficult task, compounded
by the various error sources mentioned above. In this work, we take a
principled learning approach to this problem. We introduce a neural network
architecture for robustly learning the distribution of traversability costs.
Because we are motivated by preserving the life of the robot, we tackle this
learning problem from the perspective of learning tail-risks, i.e. the
Conditional Value-at-Risk (CVaR). We show that this approach reliably learns
the expected tail risk given a desired probability risk threshold between 0 and
1, producing a traversability costmap which is more robust to outliers, more
accurately captures tail risks, and is more computationally efficient, when
compared against baselines. We validate our method on data collected a legged
robot navigating challenging, unstructured environments including an abandoned
subway, limestone caves, and lava tube caves.

    

### [[2107.11728] Federated Learning with Fair Worker Selection: A Multi-Round Submodular Maximization Approach](http://arxiv.org/abs/2107.11728)


  In this paper, we study the problem of fair worker selection in Federated
Learning systems, where fairness serves as an incentive mechanism that
encourages more workers to participate in the federation. Considering the
achieved training accuracy of the global model as the utility of the selected
workers, which is typically a monotone submodular function, we formulate the
worker selection problem as a new multi-round monotone submodular maximization
problem with cardinality and fairness constraints. The objective is to maximize
the time-average utility over multiple rounds subject to an additional fairness
requirement that each worker must be selected for a certain fraction of time.
While the traditional submodular maximization with a cardinality constraint is
already a well-known NP-Hard problem, the fairness constraint in the
multi-round setting adds an extra layer of difficulty. To address this novel
challenge, we propose three algorithms: Fair Continuous Greedy (FairCG1 and
FairCG2) and Fair Discrete Greedy (FairDG), all of which satisfy the fairness
requirement whenever feasible. Moreover, we prove nontrivial lower bounds on
the achieved time-average utility under FairCG1 and FairCG2. In addition, by
giving a higher priority to fairness, FairDG ensures a stronger short-term
fairness guarantee, which holds in every round. Finally, we perform extensive
simulations to verify the effectiveness of the proposed algorithms in terms of
the time-average utility and fairness satisfaction.

    

### [[2107.11732] Federated Causal Inference in Heterogeneous Observational Data](http://arxiv.org/abs/2107.11732)


  Analyzing observational data from multiple sources can be useful for
increasing statistical power to detect a treatment effect; however, practical
constraints such as privacy considerations may restrict individual-level
information sharing across data sets. This paper develops federated methods
that only utilize summary-level information from heterogeneous data sets. Our
federated methods provide doubly-robust point estimates of treatment effects as
well as variance estimates. We derive the asymptotic distributions of our
federated estimators, which are shown to be asymptotically equivalent to the
corresponding estimators from the combined, individual-level data. We show that
to achieve these properties, federated methods should be adjusted based on
conditions such as whether models are correctly specified and stable across
heterogeneous data sets.

    

### [[2107.11736] WiP Abstract : Robust Out-of-distribution Motion Detection and Localization in Autonomous CPS](http://arxiv.org/abs/2107.11736)


  Highly complex deep learning models are increasingly integrated into modern
cyber-physical systems (CPS), many of which have strict safety requirements.
One problem arising from this is that deep learning lacks interpretability,
operating as a black box. The reliability of deep learning is heavily impacted
by how well the model training data represents runtime test data, especially
when the input space dimension is high as natural images. In response, we
propose a robust out-of-distribution (OOD) detection framework. Our approach
detects unusual movements from driving video in real-time by combining
classical optic flow operation with representation learning via variational
autoencoder (VAE). We also design a method to locate OOD factors in images.
Evaluation on a driving simulation data set shows that our approach is
statistically more robust than related works.

    

### [[2107.11740] Identifying the fragment structure of the organic compounds by deeply learning the original NMR data](http://arxiv.org/abs/2107.11740)


  We preprocess the raw NMR spectrum and extract key characteristic features by
using two different methodologies, called equidistant sampling and peak
sampling for subsequent substructure pattern recognition; meanwhile may provide
the alternative strategy to address the imbalance issue of the NMR dataset
frequently encountered in dataset collection of statistical modeling and
establish two conventional SVM and KNN models to assess the capability of two
feature selection, respectively. Our results in this study show that the models
using the selected features of peak sampling outperform the ones using the
other. Then we build the Recurrent Neural Network (RNN) model trained by Data B
collected from peak sampling. Furthermore, we illustrate the easier
optimization of hyper parameters and the better generalization ability of the
RNN deep learning model by comparison with traditional machine learning SVM and
KNN models in detail.

    

### [[2107.11750] Improving Variational Autoencoder based Out-of-Distribution Detection for Embedded Real-time Applications](http://arxiv.org/abs/2107.11750)


  Uncertainties in machine learning are a significant roadblock for its
application in safety-critical cyber-physical systems (CPS). One source of
uncertainty arises from distribution shifts in the input data between training
and test scenarios. Detecting such distribution shifts in real-time is an
emerging approach to address the challenge. The high dimensional input space in
CPS applications involving imaging adds extra difficulty to the task.
Generative learning models are widely adopted for the task, namely
out-of-distribution (OoD) detection. To improve the state-of-the-art, we
studied existing proposals from both machine learning and CPS fields. In the
latter, safety monitoring in real-time for autonomous driving agents has been a
focus. Exploiting the spatiotemporal correlation of motion in videos, we can
robustly detect hazardous motion around autonomous driving agents. Inspired by
the latest advances in the Variational Autoencoder (VAE) theory and practice,
we tapped into the prior knowledge in data to further boost OoD detection's
robustness. Comparison studies over nuScenes and Synthia data sets show our
methods significantly improve detection capabilities of OoD factors unique to
driving scenarios, 42% better than state-of-the-art approaches. Our model also
generalized near-perfectly, 97% better than the state-of-the-art across the
real-world and simulation driving data sets experimented. Finally, we
customized one proposed method into a twin-encoder model that can be deployed
to resource limited embedded devices for real-time OoD detection. Its execution
time was reduced over four times in low-precision 8-bit integer inference,
while detection capability is comparable to its corresponding floating-point
model.

    

### [[2107.11762] DR2L: Surfacing Corner Cases to Robustify Autonomous Driving via Domain Randomization Reinforcement Learning](http://arxiv.org/abs/2107.11762)


  How to explore corner cases as efficiently and thoroughly as possible has
long been one of the top concerns in the context of deep reinforcement learning
(DeepRL) autonomous driving. Training with simulated data is less costly and
dangerous than utilizing real-world data, but the inconsistency of parameter
distribution and the incorrect system modeling in simulators always lead to an
inevitable Sim2real gap, which probably accounts for the underperformance in
novel, anomalous and risky cases that simulators can hardly generate. Domain
Randomization(DR) is a methodology that can bridge this gap with little or no
real-world data. Consequently, in this research, an adversarial model is put
forward to robustify DeepRL-based autonomous vehicles trained in simulation to
gradually surfacing harder events, so that the models could readily transfer to
the real world.

    

### [[2107.11769] ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation](http://arxiv.org/abs/2107.11769)


  Despite the success of deep learning on supervised point cloud semantic
segmentation, obtaining large-scale point-by-point manual annotations is still
a significant challenge. To reduce the huge annotation burden, we propose a
Region-based and Diversity-aware Active Learning (ReDAL), a general framework
for many deep learning approaches, aiming to automatically select only
informative and diverse sub-scene regions for label acquisition. Observing that
only a small portion of annotated regions are sufficient for 3D scene
understanding with deep learning, we use softmax entropy, color discontinuity,
and structural complexity to measure the information of sub-scene regions. A
diversity-aware selection algorithm is also developed to avoid redundant
annotations resulting from selecting informative but similar regions in a
querying batch. Extensive experiments show that our method highly outperforms
previous active learning strategies, and we achieve the performance of 90%
fully supervised learning, while less than 15% and 5% annotations are required
on S3DIS and SemanticKITTI datasets, respectively.

    

### [[2107.11774] SGD May Never Escape Saddle Points](http://arxiv.org/abs/2107.11774)


  Stochastic gradient descent (SGD) has been deployed to solve highly
non-linear and non-convex machine learning problems such as the training of
deep neural networks. However, previous works on SGD often rely on highly
restrictive and unrealistic assumptions about the nature of noise in SGD. In
this work, we mathematically construct examples that defy previous
understandings of SGD. For example, our constructions show that: (1) SGD may
converge to a local maximum; (2) SGD may escape a saddle point arbitrarily
slowly; (3) SGD may prefer sharp minima over the flat ones; and (4) AMSGrad may
converge to a local maximum. Our result suggests that the noise structure of
SGD might be more important than the loss landscape in neural network training
and that future research should focus on deriving the actual noise structure in
deep learning.

    

### [[2107.11784] Power of human-algorithm collaboration in solving combinatorial optimization problems](http://arxiv.org/abs/2107.11784)


  Many combinatorial optimization problems are often considered intractable to
solve exactly or by approximation. An example of such problem is maximum clique
which -- under standard assumptions in complexity theory -- cannot be solved in
sub-exponential time or be approximated within polynomial factor efficiently.
We show that if a polynomial time algorithm can query informative Gaussian
priors from an expert $poly(n)$ times, then a class of combinatorial
optimization problems can be solved efficiently in expectation up to a
multiplicative factor $\epsilon$ where $\epsilon$ is arbitrary constant. While
our proposed methods are merely theoretical, they cast new light on how to
approach solving these problems that have been usually considered intractable.

    

### [[2107.11789] ROD: Reception-aware Online Distillation for Sparse Graphs](http://arxiv.org/abs/2107.11789)


  Graph neural networks (GNNs) have been widely used in many graph-based tasks
such as node classification, link prediction, and node clustering. However,
GNNs gain their performance benefits mainly from performing the feature
propagation and smoothing across the edges of the graph, thus requiring
sufficient connectivity and label information for effective propagation.
Unfortunately, many real-world networks are sparse in terms of both edges and
labels, leading to sub-optimal performance of GNNs. Recent interest in this
sparse problem has focused on the self-training approach, which expands
supervised signals with pseudo labels. Nevertheless, the self-training approach
inherently cannot realize the full potential of refining the learning
performance on sparse graphs due to the unsatisfactory quality and quantity of
pseudo labels.
In this paper, we propose ROD, a novel reception-aware online knowledge
distillation approach for sparse graph learning. We design three supervision
signals for ROD: multi-scale reception-aware graph knowledge, task-based
supervision, and rich distilled knowledge, allowing online knowledge transfer
in a peer-teaching manner. To extract knowledge concealed in the multi-scale
reception fields, ROD explicitly requires individual student models to preserve
different levels of locality information. For a given task, each student would
predict based on its reception-scale knowledge, while simultaneously a strong
teacher is established on-the-fly by combining multi-scale knowledge. Our
approach has been extensively evaluated on 9 datasets and a variety of
graph-based tasks, including node classification, link prediction, and node
clustering. The result demonstrates that ROD achieves state-of-art performance
and is more robust for the graph sparsity.

    

### [[2107.11795] Character Spotting Using Machine Learning Techniques](http://arxiv.org/abs/2107.11795)


  This work presents a comparison of machine learning algorithms that are
implemented to segment the characters of text presented as an image. The
algorithms are designed to work on degraded documents with text that is not
aligned in an organized fashion. The paper investigates the use of Support
Vector Machines, K-Nearest Neighbor algorithm and an Encoder Network to perform
the operation of character spotting. Character Spotting involves extracting
potential characters from a stream of text by selecting regions bound by white
space.

    

### [[2107.11801] Denoising and Segmentation of Epigraphical Scripts](http://arxiv.org/abs/2107.11801)


  This paper is a presentation of a new method for denoising images using
Haralick features and further segmenting the characters using artificial neural
networks. The image is divided into kernels, each of which is converted to a
GLCM (Gray Level Co-Occurrence Matrix) on which a Haralick Feature generation
function is called, the result of which is an array with fourteen elements
corresponding to fourteen features The Haralick values and the corresponding
noise/text classification form a dictionary, which is then used to de-noise the
image through kernel comparison. Segmentation is the process of extracting
characters from a document and can be used when letters are separated by white
space, which is an explicit boundary marker. Segmentation is the first step in
many Natural Language Processing problems. This paper explores the process of
segmentation using Neural Networks. While there have been numerous methods to
segment characters of a document, this paper is only concerned with the
accuracy of doing so using neural networks. It is imperative that the
characters be segmented correctly, for failing to do so will lead to incorrect
recognition by Natural language processing tools. Artificial Neural Networks
was used to attain accuracy of upto 89%. This method is suitable for languages
where the characters are delimited by white space. However, this method will
fail to provide acceptable results when the language heavily uses connected
letters. An example would be the Devanagari script, which is predominantly used
in northern India.

    

### [[2107.11811] Reinforced Imitation Learning by Free Energy Principle](http://arxiv.org/abs/2107.11811)


  Reinforcement Learning (RL) requires a large amount of exploration especially
in sparse-reward settings. Imitation Learning (IL) can learn from expert
demonstrations without exploration, but it never exceeds the expert's
performance and is also vulnerable to distributional shift between
demonstration and execution. In this paper, we radically unify RL and IL based
on Free Energy Principle (FEP). FEP is a unified Bayesian theory of the brain
that explains perception, action and model learning by a common fundamental
principle. We present a theoretical extension of FEP and derive an algorithm in
which an agent learns the world model that internalizes expert demonstrations
and at the same time uses the model to infer the current and future states and
actions that maximize rewards. The algorithm thus reduces exploration costs by
partially imitating experts as well as maximizing its return in a seamless way,
resulting in a higher performance than the suboptimal expert. Our experimental
results show that this approach is promising in visual control tasks especially
in sparse-reward environments.

    

### [[2107.11817] Go Wider Instead of Deeper](http://arxiv.org/abs/2107.11817)


  The transformer has recently achieved impressive results on various tasks. To
further improve the effectiveness and efficiency of the transformer, there are
two trains of thought among existing works: (1) going wider by scaling to more
trainable parameters; (2) going shallower by parameter sharing or model
compressing along with the depth. However, larger models usually do not scale
well when fewer tokens are available to train, and advanced parallelisms are
required when the model is extremely large. Smaller models usually achieve
inferior performance compared to the original transformer model due to the loss
of representation power. In this paper, to achieve better performance with
fewer trainable parameters, we propose a framework to deploy trainable
parameters efficiently, by going wider instead of deeper. Specially, we scale
along model width by replacing feed-forward network (FFN) with
mixture-of-experts (MoE). We then share the MoE layers across transformer
blocks using individual layer normalization. Such deployment plays the role to
transform various semantic representations, which makes the model more
parameter-efficient and effective. To evaluate our framework, we design WideNet
and evaluate it on ImageNet-1K. Our best model outperforms Vision Transformer
(ViT) by $1.46\%$ with $0.72 \times$ trainable parameters. Using $0.46 \times$
and $0.13 \times$ parameters, our WideNet can still surpass ViT and ViT-MoE by
$0.83\%$ and $2.08\%$, respectively.

    

### [[2107.11822] Distributional Shifts in Automated Diabetic Retinopathy Screening](http://arxiv.org/abs/2107.11822)


  Deep learning-based models are developed to automatically detect if a retina
image is `referable' in diabetic retinopathy (DR) screening. However, their
classification accuracy degrades as the input images distributionally shift
from their training distribution. Further, even if the input is not a retina
image, a standard DR classifier produces a high confident prediction that the
image is `referable'. Our paper presents a Dirichlet Prior Network-based
framework to address this issue. It utilizes an out-of-distribution (OOD)
detector model and a DR classification model to improve generalizability by
identifying OOD images. Experiments on real-world datasets indicate that the
proposed framework can eliminate the unknown non-retina images and identify the
distributionally shifted retina images for human intervention.

    

### [[2107.11843] Deep Learning Explicit Differentiable Predictive Control Laws for Buildings](http://arxiv.org/abs/2107.11843)


  We present a differentiable predictive control (DPC) methodology for learning
constrained control laws for unknown nonlinear systems. DPC poses an
approximate solution to multiparametric programming problems emerging from
explicit nonlinear model predictive control (MPC). Contrary to approximate MPC,
DPC does not require supervision by an expert controller. Instead, a system
dynamics model is learned from the observed system's dynamics, and the neural
control law is optimized offline by leveraging the differentiable closed-loop
system model. The combination of a differentiable closed-loop system and
penalty methods for constraint handling of system outputs and inputs allows us
to optimize the control law's parameters directly by backpropagating economic
MPC loss through the learned system model. The control performance of the
proposed DPC method is demonstrated in simulation using learned model of
multi-zone building thermal dynamics.

    

### [[2107.11844] A binary variant of gravitational search algorithm and its application to windfarm layout optimization problem](http://arxiv.org/abs/2107.11844)


  In the binary search space, GSA framework encounters the shortcomings of
stagnation, diversity loss, premature convergence and high time complexity. To
address these issues, a novel binary variant of GSA called `A novel
neighbourhood archives embedded gravitational constant in GSA for binary search
space (BNAGGSA)' is proposed in this paper. In BNAGGSA, the novel
fitness-distance based social interaction strategy produces a self-adaptive
step size mechanism through which the agent moves towards the optimal direction
with the optimal step size, as per its current search requirement. The
performance of the proposed algorithm is compared with the two binary variants
of GSA over 23 well-known benchmark test problems. The experimental results and
statistical analyses prove the supremacy of BNAGGSA over the compared
algorithms. Furthermore, to check the applicability of the proposed algorithm
in solving real-world applications, a windfarm layout optimization problem is
considered. Two case studies with two different wind data sets of two different
wind sites is considered for experiments.

    

### [[2107.11856] Graph Representation Learning on Tissue-Specific Multi-Omics](http://arxiv.org/abs/2107.11856)


  Combining different modalities of data from human tissues has been critical
in advancing biomedical research and personalised medical care. In this study,
we leverage a graph embedding model (i.e VGAE) to perform link prediction on
tissue-specific Gene-Gene Interaction (GGI) networks. Through ablation
experiments, we prove that the combination of multiple biological modalities
(i.e multi-omics) leads to powerful embeddings and better link prediction
performances. Our evaluation shows that the integration of gene methylation
profiles and RNA-sequencing data significantly improves the link prediction
performance. Overall, the combination of RNA-sequencing and gene methylation
data leads to a link prediction accuracy of 71% on GGI networks. By harnessing
graph representation learning on multi-omics data, our work brings novel
insights to the current literature on multi-omics integration in
bioinformatics.

    

### [[2107.11862] Decision-forest voting scheme for classification of rare classes in network intrusion detection](http://arxiv.org/abs/2107.11862)


  In this paper, Bayesian based aggregation of decision trees in an ensemble
(decision forest) is investigated. The focus is laid on multi-class
classification with number of samples significantly skewed toward one of the
classes. The algorithm leverages out-of-bag datasets to estimate prediction
errors of individual trees, which are then used in accordance with the Bayes
rule to refine the decision of the ensemble. The algorithm takes prevalence of
individual classes into account and does not require setting of any additional
parameters related to class weights or decision-score thresholds. Evaluation is
based on publicly available datasets as well as on an proprietary dataset
comprising network traffic telemetry from hundreds of enterprise networks with
over a million of users overall. The aim is to increase the detection
capabilities of an operating malware detection system. While we were able to
keep precision of the system higher than 94\%, that is only 6 out of 100
detections shown to the network administrator are false alarms, we were able to
achieve increase of approximately 7\% in the number of detections. The
algorithm effectively handles large amounts of data, and can be used in
conjunction with most of the state-of-the-art algorithms used to train decision
forests.

    

### [[2107.11864] Neural Circuit Synthesis from Specification Patterns](http://arxiv.org/abs/2107.11864)


  We train hierarchical Transformers on the task of synthesizing hardware
circuits directly out of high-level logical specifications in linear-time
temporal logic (LTL). The LTL synthesis problem is a well-known algorithmic
challenge with a long history and an annual competition is organized to track
the improvement of algorithms and tooling over time. New approaches using
machine learning might open a lot of possibilities in this area, but suffer
from the lack of sufficient amounts of training data. In this paper, we
consider a method to generate large amounts of additional training data, i.e.,
pairs of specifications and circuits implementing them. We ensure that this
synthetic data is sufficiently close to human-written specifications by mining
common patterns from the specifications used in the synthesis competitions. We
show that hierarchical Transformers trained on this synthetic data solve a
significant portion of problems from the synthesis competitions, and even
out-of-distribution examples from a recent case study.

    

### [[2107.11876] A Study on Speech Enhancement Based on Diffusion Probabilistic Model](http://arxiv.org/abs/2107.11876)


  Diffusion probabilistic models have demonstrated an outstanding capability to
model natural images and raw audio waveforms through a paired diffusion and
reverse processes. The unique property of the reverse process (namely,
eliminating non-target signals from the Gaussian noise and noisy signals) could
be utilized to restore clean signals. Based on this property, we propose a
diffusion probabilistic model-based speech enhancement (DiffuSE) model that
aims to recover clean speech signals from noisy signals. The fundamental
architecture of the proposed DiffuSE model is similar to that of DiffWave--a
high-quality audio waveform generation model that has a relatively low
computational cost and footprint. To attain better enhancement performance, we
designed an advanced reverse process, termed the supportive reverse process,
which adds noisy speech in each time-step to the predicted speech. The
experimental results show that DiffuSE yields performance that is comparable to
related audio generative models on the standardized Voice Bank corpus SE task.
Moreover, relative to the generally suggested full sampling schedule, the
proposed supportive reverse process especially improved the fast sampling,
taking few steps to yield better enhancement results over the conventional full
step inference process.

    

### [[2107.11882] Lung Cancer Risk Estimation with Incomplete Data: A Joint Missing Imputation Perspective](http://arxiv.org/abs/2107.11882)


  Data from multi-modality provide complementary information in clinical
prediction, but missing data in clinical cohorts limits the number of subjects
in multi-modal learning context. Multi-modal missing imputation is challenging
with existing methods when 1) the missing data span across heterogeneous
modalities (e.g., image vs. non-image); or 2) one modality is largely missing.
In this paper, we address imputation of missing data by modeling the joint
distribution of multi-modal data. Motivated by partial bidirectional generative
adversarial net (PBiGAN), we propose a new Conditional PBiGAN (C-PBiGAN) method
that imputes one modality combining the conditional knowledge from another
modality. Specifically, C-PBiGAN introduces a conditional latent space in a
missing imputation framework that jointly encodes the available multi-modal
data, along with a class regularization loss on imputed data to recover
discriminative information. To our knowledge, it is the first generative
adversarial model that addresses multi-modal missing imputation by modeling the
joint distribution of image and non-image data. We validate our model with both
the national lung screening trial (NLST) dataset and an external clinical
validation cohort. The proposed C-PBiGAN achieves significant improvements in
lung cancer risk estimation compared with representative imputation methods
(e.g., AUC values increase in both NLST (+2.9\%) and in-house dataset (+4.3\%)
compared with PBiGAN, p$<$0.05).

    

### [[2107.11886] Logspace Reducibility From Secret Leakage Planted Clique](http://arxiv.org/abs/2107.11886)


  The planted clique problem is well-studied in the context of observing,
explaining, and predicting interesting computational phenomena associated with
statistical problems. When equating computational efficiency with the existence
of polynomial time algorithms, the computational hardness of (some variant of)
the planted clique problem can be used to infer the computational hardness of a
host of other statistical problems.
Is this ability to transfer computational hardness from (some variant of) the
planted clique problem to other statistical problems robust to changing our
notion of computational efficiency to space efficiency?
We answer this question affirmatively for three different statistical
problems, namely Sparse PCA, submatrix detection, and testing almost k-wise
independence. The key challenge is that space efficient randomized reductions
need to repeatedly access the randomness they use. Known reductions to these
problems are all randomized and need polynomially many random bits to
implement. Since we can not store polynomially many random bits in memory, it
is unclear how to implement these existing reductions space efficiently. There
are two ideas involved in circumventing this issue and implementing known
reductions to these problems space efficiently.
1. When solving statistical problems, we can use parts of the input itself as
randomness.
2. Secret leakage variants of the planted clique problem with appropriate
secret leakage can be more useful than the standard planted clique problem when
we want to use parts of the input as randomness.
(abstract shortened due to arxiv constraints)

    

### [[2107.11889] GCExplainer: Human-in-the-Loop Concept-based Explanations for Graph Neural Networks](http://arxiv.org/abs/2107.11889)


  While graph neural networks (GNNs) have been shown to perform well on
graph-based data from a variety of fields, they suffer from a lack of
transparency and accountability, which hinders trust and consequently the
deployment of such models in high-stake and safety-critical scenarios. Even
though recent research has investigated methods for explaining GNNs, these
methods are limited to single-instance explanations, also known as local
explanations. Motivated by the aim of providing global explanations, we adapt
the well-known Automated Concept-based Explanation approach (Ghorbani et al.,
2019) to GNN node and graph classification, and propose GCExplainer.
GCExplainer is an unsupervised approach for post-hoc discovery and extraction
of global concept-based explanations for GNNs, which puts the human in the
loop. We demonstrate the success of our technique on five node classification
datasets and two graph classification datasets, showing that we are able to
discover and extract high-quality concept representations by putting the human
in the loop. We achieve a maximum completeness score of 1 and an average
completeness score of 0.753 across the datasets. Finally, we show that the
concept-based explanations provide an improved insight into the datasets and
GNN models compared to the state-of-the-art explanations produced by
GNNExplainer (Ying et al., 2019).

    

### [[2107.11892] A brief note on understanding neural networks as Gaussian processes](http://arxiv.org/abs/2107.11892)


  As a generalization of the work in [Lee et al., 2017], this note briefly
discusses when the prior of a neural network output follows a Gaussian process,
and how a neural-network-induced Gaussian process is formulated. The posterior
mean functions of such a Gaussian process regression lie in the reproducing
kernel Hilbert space defined by the neural-network-induced kernel. In the case
of two-layer neural networks, the induced Gaussian processes provide an
interpretation of the reproducing kernel Hilbert spaces whose union forms a
Barron space.

    

### [[2107.11906] H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences](http://arxiv.org/abs/2107.11906)


  We describe an efficient hierarchical method to compute attention in the
Transformer architecture. The proposed attention mechanism exploits a matrix
structure similar to the Hierarchical Matrix (H-Matrix) developed by the
numerical analysis community, and has linear run time and memory complexity. We
perform extensive experiments to show that the inductive bias embodied by our
hierarchical attention is effective in capturing the hierarchical structure in
the sequences typical for natural language and vision tasks. Our method is
superior to alternative sub-quadratic proposals by over +6 points on average on
the Long Range Arena benchmark. It also sets a new SOTA test perplexity on
One-Billion Word dataset with 5x fewer model parameters than that of the
previous-best Transformer-based models.

    

### [[2107.11911] Restless Bandits with Many Arms: Beating the Central Limit Theorem](http://arxiv.org/abs/2107.11911)


  We consider finite-horizon restless bandits with multiple pulls per period,
which play an important role in recommender systems, active learning, revenue
management, and many other areas. While an optimal policy can be computed, in
principle, using dynamic programming, the computation required scales
exponentially in the number of arms $N$. Thus, there is substantial value in
understanding the performance of index policies and other policies that can be
computed efficiently for large $N$. We study the growth of the optimality gap,
i.e., the loss in expected performance compared to an optimal policy, for such
policies in a classical asymptotic regime proposed by Whittle in which $N$
grows while holding constant the fraction of arms that can be pulled per
period. Intuition from the Central Limit Theorem and the tightest previous
theoretical bounds suggest that this optimality gap should grow like
$O(\sqrt{N})$. Surprisingly, we show that it is possible to outperform this
bound. We characterize a non-degeneracy condition and a wide class of novel
practically-computable policies, called fluid-priority policies, in which the
optimality gap is $O(1)$. These include most widely-used index policies. When
this non-degeneracy condition does not hold, we show that fluid-priority
policies nevertheless have an optimality gap that is $O(\sqrt{N})$,
significantly generalizing the class of policies for which convergence rates
are known. We demonstrate that fluid-priority policies offer state-of-the-art
performance on a collection of restless bandit problems in numerical
experiments.

    

### [[2107.11913] Measuring Ethics in AI with AI: A Methodology and Dataset Construction](http://arxiv.org/abs/2107.11913)


  Recently, the use of sound measures and metrics in Artificial Intelligence
has become the subject of interest of academia, government, and industry.
Efforts towards measuring different phenomena have gained traction in the AI
community, as illustrated by the publication of several influential field
reports and policy documents. These metrics are designed to help decision
takers to inform themselves about the fast-moving and impacting influences of
key advances in Artificial Intelligence in general and Machine Learning in
particular. In this paper we propose to use such newfound capabilities of AI
technologies to augment our AI measuring capabilities. We do so by training a
model to classify publications related to ethical issues and concerns. In our
methodology we use an expert, manually curated dataset as the training set and
then evaluate a large set of research papers. Finally, we highlight the
implications of AI metrics, in particular their contribution towards developing
trustful and fair AI-based tools and technologies. Keywords: AI Ethics; AI
Fairness; AI Measurement. Ethics in Computer Science.

    

### [[2107.11921] Compensation Learning](http://arxiv.org/abs/2107.11921)


  Weighting strategy prevails in machine learning. For example, a common
approach in robust machine learning is to exert lower weights on samples which
are likely to be noisy or hard. This study reveals another undiscovered
strategy, namely, compensating, that has also been widely used in machine
learning. Learning with compensating is called compensation learning and a
systematic taxonomy is constructed for it in this study. In our taxonomy,
compensation learning is divided on the basis of the compensation targets,
inference manners, and granularity levels. Many existing learning algorithms
including some classical ones can be seen as a special case of compensation
learning or partially leveraging compensating. Furthermore, a family of new
learning algorithms can be obtained by plugging the compensation learning into
existing learning algorithms. Specifically, three concrete new learning
algorithms are proposed for robust machine learning. Extensive experiments on
text sentiment analysis, image classification, and graph classification verify
the effectiveness of the three new algorithms. Compensation learning can also
be used in various learning scenarios, such as imbalance learning, clustering,
regression, and so on.

    

### [[2107.11949] Dissecting FLOPs along input dimensions for GreenAI cost estimations](http://arxiv.org/abs/2107.11949)


  The term GreenAI refers to a novel approach to Deep Learning, that is more
aware of the ecological impact and the computational efficiency of its methods.
The promoters of GreenAI suggested the use of Floating Point Operations (FLOPs)
as a measure of the computational cost of Neural Networks; however, that
measure does not correlate well with the energy consumption of hardware
equipped with massively parallel processing units like GPUs or TPUs. In this
article, we propose a simple refinement of the formula used to compute floating
point operations for convolutional layers, called {\alpha}-FLOPs, explaining
and correcting the traditional discrepancy with respect to different layers,
and closer to reality. The notion of {\alpha}-FLOPs relies on the crucial
insight that, in case of inputs with multiple dimensions, there is no reason to
believe that the speedup offered by parallelism will be uniform along all
different axes.

    

### [[2107.11954] Aggregate or Not? Exploring Where to Privatize in DNN Based Federated Learning Under Different Non-IID Scenes](http://arxiv.org/abs/2107.11954)


  Although federated learning (FL) has recently been proposed for efficient
distributed training and data privacy protection, it still encounters many
obstacles. One of these is the naturally existing statistical heterogeneity
among clients, making local data distributions non independently and
identically distributed (i.e., non-iid), which poses challenges for model
aggregation and personalization. For FL with a deep neural network (DNN),
privatizing some layers is a simple yet effective solution for non-iid
problems. However, which layers should we privatize to facilitate the learning
process? Do different categories of non-iid scenes have preferred privatization
ways? Can we automatically learn the most appropriate privatization way during
FL? In this paper, we answer these questions via abundant experimental studies
on several FL benchmarks. First, we present the detailed statistics of these
benchmarks and categorize them into covariate and label shift non-iid scenes.
Then, we investigate both coarse-grained and fine-grained network splits and
explore whether the preferred privatization ways have any potential relations
to the specific category of a non-iid scene. Our findings are exciting, e.g.,
privatizing the base layers could boost the performances even in label shift
non-iid scenes, which are inconsistent with some natural conjectures. We also
find that none of these privatization ways could improve the performances on
the Shakespeare benchmark, and we guess that Shakespeare may not be a seriously
non-iid scene. Finally, we propose several approaches to automatically learn
where to aggregate via cross-stitch, soft attention, and hard selection. We
advocate the proposed methods could serve as a preliminary try to explore where
to privatize for a novel non-iid scene.

    

### [[2107.11956] Preliminary Steps Towards Federated Sentiment Classification](http://arxiv.org/abs/2107.11956)


  Automatically mining sentiment tendency contained in natural language is a
fundamental research to some artificial intelligent applications, where
solutions alternate with challenges. Transfer learning and multi-task learning
techniques have been leveraged to mitigate the supervision sparsity and
collaborate multiple heterogeneous domains correspondingly. Recent years, the
sensitive nature of users' private data raises another challenge for sentiment
classification, i.e., data privacy protection. In this paper, we resort to
federated learning for multiple domain sentiment classification under the
constraint that the corpora must be stored on decentralized devices. In view of
the heterogeneous semantics across multiple parties and the peculiarities of
word embedding, we pertinently provide corresponding solutions. First, we
propose a Knowledge Transfer Enhanced Private-Shared (KTEPS) framework for
better model aggregation and personalization in federated sentiment
classification. Second, we propose KTEPS$^\star$ with the consideration of the
rich semantic and huge embedding size properties of word vectors, utilizing
Projection-based Dimension Reduction (PDR) methods for privacy protection and
efficient transmission simultaneously. We propose two federated sentiment
classification scenes based on public benchmarks, and verify the superiorities
of our proposed methods with abundant experimental investigations.

    

### [[2107.11972] Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Adaptive Refined Labeling](http://arxiv.org/abs/2107.11972)


  Price movement forecasting aims at predicting the future trends of financial
assets based on the current market conditions and other relevant information.
Recently, machine learning(ML) methods have become increasingly popular and
achieved promising results for price movement forecasting in both academia and
industry. Most existing ML solutions formulate the forecasting problem as a
classification(to predict the direction) or a regression(to predict the return)
problem in the entire set of training data. However, due to the extremely low
signal-to-noise ratio and stochastic nature of financial data, good trading
opportunities are extremely scarce. As a result, without careful selection of
potentially profitable samples, such ML methods are prone to capture the
patterns of noises instead of real signals. To address the above issues, we
propose a novel framework-LARA(Locality-Aware Attention and Adaptive Refined
Labeling), which contains the following three components: 1)Locality-aware
attention automatically extracts the potentially profitable samples by
attending to their label information in order to construct a more accurate
classifier on these selected samples. 2)Adaptive refined labeling further
iteratively refines the labels, alleviating the noise of samples. 3)Equipped
with metric learning techniques, Locality-aware attention enjoys task-specific
distance metrics and distributes attention on potentially profitable samples in
a more effective way. To validate our method, we conduct comprehensive
experiments on three real-world financial markets: ETFs, the China's A-share
stock market, and the cryptocurrency market. LARA achieves superior performance
compared with the time-series analysis methods and a set of machine learning
based competitors on the Qlib platform. Extensive ablation studies and
experiments demonstrate that LARA indeed captures more reliable trading
opportunities.

    

### [[2107.11999] Stable Dynamic Mode Decomposition Algorithm for Noisy Pressure-Sensitive Paint Measurement Data](http://arxiv.org/abs/2107.11999)


  In this study, we proposed the truncated total least squares dynamic mode
decomposition (T-TLS DMD) algorithm, which can perform DMD analysis of noisy
data. By adding truncation regularization to the conventional TLS DMD
algorithm, T-TLS DMD improves the stability of the computation while
maintaining the accuracy of TLS DMD. The effectiveness of the proposed method
was evaluated by the analysis of the wake behind a cylinder and
pressure-sensitive paint (PSP) data for the buffet cell phenomenon. The results
showed the importance of regularization in the DMD algorithm. With respect to
the eigenvalues, T-TLS DMD was less affected by noise, and accurate eigenvalues
could be obtained stably, whereas the eigenvalues of TLS and subspace DMD
varied greatly due to noise. It was also observed that the eigenvalues of the
standard and exact DMD had the problem of shifting to the damping side, as
reported in previous studies. With respect to eigenvectors, T-TLS and exact DMD
captured the characteristic flow patterns clearly even in the presence of
noise, whereas TLS and subspace DMD were not able to capture them clearly due
to noise.

    

### [[2107.12003] Facetron: Multi-speaker Face-to-Speech Model based on Cross-modal Latent Representations](http://arxiv.org/abs/2107.12003)


  In this paper, we propose an effective method to synthesize speaker-specific
speech waveforms by conditioning on videos of an individual's face. Using a
generative adversarial network (GAN) with linguistic and speaker characteristic
features as auxiliary conditions, our method directly converts face images into
speech waveforms under an end-to-end training framework. The linguistic
features are extracted from lip movements using a lip-reading model, and the
speaker characteristic features are predicted from face images using
cross-modal learning with a pre-trained acoustic model. Since these two
features are uncorrelated and controlled independently, we can flexibly
synthesize speech waveforms whose speaker characteristics vary depending on the
input face images. Therefore, our method can be regarded as a multi-speaker
face-to-speech waveform model. We show the superiority of our proposed model
over conventional methods in terms of both objective and subjective evaluation
results. Specifically, we evaluate the performances of the linguistic feature
and the speaker characteristic generation modules by measuring the accuracy of
automatic speech recognition and automatic speaker/gender recognition tasks,
respectively. We also evaluate the naturalness of the synthesized speech
waveforms using a mean opinion score (MOS) test.

    

### [[2107.12009] Weakly Supervised Attention Model for RV StrainClassification from volumetric CTPA Scans](http://arxiv.org/abs/2107.12009)


  Pulmonary embolus (PE) refers to obstruction of pulmonary arteries by blood
clots. PE accounts for approximately 100,000 deaths per year in the United
States alone. The clinical presentation of PE is often nonspecific, making the
diagnosis challenging. Thus, rapid and accurate risk stratification is of
paramount importance. High-risk PE is caused by right ventricular (RV)
dysfunction from acute pressure overload, which in return can help identify
which patients require more aggressive therapy. Reconstructed four-chamber
views of the heart on chest CT can detect right ventricular enlargement. CT
pulmonary angiography (CTPA) is the golden standard in the diagnostic workup of
suspected PE. Therefore, it can link between diagnosis and risk stratification
strategies. We developed a weakly supervised deep learning algorithm, with an
emphasis on a novel attention mechanism, to automatically classify RV strain on
CTPA. Our method is a 3D DenseNet model with integrated 3D residual attention
blocks. We evaluated our model on a dataset of CTPAs of emergency department
(ED) PE patients. This model achieved an area under the receiver operating
characteristic curve (AUC) of 0.88 for classifying RV strain. The model showed
a sensitivity of 87% and specificity of 83.7%. Our solution outperforms
state-of-the-art 3D CNN networks. The proposed design allows for a fully
automated network that can be trained easily in an end-to-end manner without
requiring computationally intensive and time-consuming preprocessing or
strenuous labeling of the data.We infer that unmarked CTPAs can be used for
effective RV strain classification. This could be used as a second reader,
alerting for high-risk PE patients. To the best of our knowledge, there are no
previous deep learning-based studies that attempted to solve this problem.

    

### [[2107.12013] A Shallow Ritz Method for elliptic problems with Singular Sources](http://arxiv.org/abs/2107.12013)


  In this paper, a shallow Ritz-type neural network for solving elliptic
problems with delta function singular sources on an interface is developed.
There are three novel features in the present work; namely, (i) the delta
function singularity is naturally removed, (ii) level set function is
introduced as a feather input, (iii) it is completely shallow consisting of
only one hidden layer. We first introduce the energy functional of the problem
and then transform the contribution of singular sources to a regular surface
integral along the interface. In such a way the delta function singularity can
be naturally removed without the introduction of discrete delta function that
is commonly used in traditional regularization methods such as the well-known
immersed boundary method. The original problem is then reformulated as a
minimization problem. We propose a shallow Ritz-type neural network with one
hidden layer to approximate the global minimizer of the energy functional. As a
result, the network is trained by minimizing the loss function that is a
discrete version of the energy. In addition, we include the level set function
of the interface as a feature input and find that it significantly improves the
training efficiency and accuracy. We perform a series of numerical tests to
demonstrate the accuracy of the present network as well as its capability for
problems in irregular domains and in higher dimensions.

    

### [[2107.12033] Joint Direction and Proximity Classification of Overlapping Sound Events from Binaural Audio](http://arxiv.org/abs/2107.12033)


  Sound source proximity and distance estimation are of great interest in many
practical applications, since they provide significant information for acoustic
scene analysis. As both tasks share complementary qualities, ensuring efficient
interaction between these two is crucial for a complete picture of an aural
environment. In this paper, we aim to investigate several ways of performing
joint proximity and direction estimation from binaural recordings, both defined
as coarse classification problems based on Deep Neural Networks (DNNs).
Considering the limitations of binaural audio, we propose two methods of
splitting the sphere into angular areas in order to obtain a set of directional
classes. For each method we study different model types to acquire information
about the direction-of-arrival (DoA). Finally, we propose various ways of
combining the proximity and direction estimation problems into a joint task
providing temporal information about the onsets and offsets of the appearing
sources. Experiments are performed for a synthetic reverberant binaural dataset
consisting of up to two overlapping sound events.

    

### [[2107.12034] Workpiece Image-based Tool Wear Classification in Blanking Processes Using Deep Convolutional Neural Networks](http://arxiv.org/abs/2107.12034)


  Blanking processes belong to the most widely used manufacturing techniques
due to their economic efficiency. Their economic viability depends to a large
extent on the resulting product quality and the associated customer
satisfaction as well as on possible downtimes. In particular, the occurrence of
increased tool wear reduces the product quality and leads to downtimes, which
is why considerable research has been carried out in recent years with regard
to wear detection. While processes have widely been monitored based on force
and acceleration signals, a new approach is pursued in this paper. Blanked
workpieces manufactured by punches with 16 different wear states are
photographed and then used as inputs for Deep Convolutional Neural Networks to
classify wear states. The results show that wear states can be predicted with
surprisingly high accuracy, opening up new possibilities and research
opportunities for tool wear monitoring of blanking processes.

    

### [[2107.12045] How to Certify Machine Learning Based Safety-critical Systems? A Systematic Literature Review](http://arxiv.org/abs/2107.12045)


  Context: Machine Learning (ML) has been at the heart of many innovations over
the past years. However, including it in so-called 'safety-critical' systems
such as automotive or aeronautic has proven to be very challenging, since the
shift in paradigm that ML brings completely changes traditional certification
approaches.
Objective: This paper aims to elucidate challenges related to the
certification of ML-based safety-critical systems, as well as the solutions
that are proposed in the literature to tackle them, answering the question 'How
to Certify Machine Learning Based Safety-critical Systems?'.
Method: We conduct a Systematic Literature Review (SLR) of research papers
published between 2015 to 2020, covering topics related to the certification of
ML systems. In total, we identified 229 papers covering topics considered to be
the main pillars of ML certification: Robustness, Uncertainty, Explainability,
Verification, Safe Reinforcement Learning, and Direct Certification. We
analyzed the main trends and problems of each sub-field and provided summaries
of the papers extracted.
Results: The SLR results highlighted the enthusiasm of the community for this
subject, as well as the lack of diversity in terms of datasets and type of
models. It also emphasized the need to further develop connections between
academia and industries to deepen the domain study. Finally, it also
illustrated the necessity to build connections between the above mention main
pillars that are for now mainly studied separately.
Conclusion: We highlighted current efforts deployed to enable the
certification of ML based software systems, and discuss some future research
directions.

    

### [[2107.12046] 3D AGSE-VNet: An Automatic Brain Tumor MRI Data Segmentation Framework](http://arxiv.org/abs/2107.12046)


  Background: Glioma is the most common brain malignant tumor, with a high
morbidity rate and a mortality rate of more than three percent, which seriously
endangers human health. The main method of acquiring brain tumors in the clinic
is MRI. Segmentation of brain tumor regions from multi-modal MRI scan images is
helpful for treatment inspection, post-diagnosis monitoring, and effect
evaluation of patients. However, the common operation in clinical brain tumor
segmentation is still manual segmentation, lead to its time-consuming and large
performance difference between different operators, a consistent and accurate
automatic segmentation method is urgently needed. Methods: To meet the above
challenges, we propose an automatic brain tumor MRI data segmentation framework
which is called AGSE-VNet. In our study, the Squeeze and Excite (SE) module is
added to each encoder, the Attention Guide Filter (AG) module is added to each
decoder, using the channel relationship to automatically enhance the useful
information in the channel to suppress the useless information, and use the
attention mechanism to guide the edge information and remove the influence of
irrelevant information such as noise. Results: We used the BraTS2020 challenge
online verification tool to evaluate our approach. The focus of verification is
that the Dice scores of the whole tumor (WT), tumor core (TC) and enhanced
tumor (ET) are 0.68, 0.85 and 0.70, respectively. Conclusion: Although MRI
images have different intensities, AGSE-VNet is not affected by the size of the
tumor, and can more accurately extract the features of the three regions, it
has achieved impressive results and made outstanding contributions to the
clinical diagnosis and treatment of brain tumor patients.

    

### [[2107.12048] Decentralized Federated Learning: Balancing Communication and Computing Costs](http://arxiv.org/abs/2107.12048)


  Decentralized federated learning (DFL) is a powerful framework of distributed
machine learning and decentralized stochastic gradient descent (SGD) is a
driving engine for DFL. The performance of decentralized SGD is jointly
influenced by communication-efficiency and convergence rate. In this paper, we
propose a general decentralized federated learning framework to strike a
balance between communication-efficiency and convergence performance. The
proposed framework performs both multiple local updates and multiple inter-node
communications periodically, unifying traditional decentralized SGD methods. We
establish strong convergence guarantees for the proposed DFL algorithm without
the assumption of convex objective function. The balance of communication and
computation rounds is essential to optimize decentralized federated learning
under constrained communication and computation resources. For further
improving communication-efficiency of DFL, compressed communication is applied
to DFL, named DFL with compressed communication (C-DFL). The proposed C-DFL
exhibits linear convergence for strongly convex objectives. Experiment results
based on MNIST and CIFAR-10 datasets illustrate the superiority of DFL over
traditional decentralized SGD methods and show that C-DFL further enhances
communication-efficiency.

    

### [[2107.12065] Provably Accelerated Decentralized Gradient Method Over Unbalanced Directed Graphs](http://arxiv.org/abs/2107.12065)


  In this work, we consider the decentralized optimization problem in which a
network of $n$ agents, each possessing a smooth and convex objective function,
wish to collaboratively minimize the average of all the objective functions
through peer-to-peer communication in a directed graph. To solve the problem,
we propose two accelerated Push-DIGing methods termed APD and APD-SC for
minimizing non-strongly convex objective functions and strongly convex ones,
respectively. We show that APD and APD-SC respectively converge at the rates
$O\left(\frac{1}{k^2}\right)$ and $O\left(\left(1 -
C\sqrt{\frac{\mu}{L}}\right)^k\right)$ up to constant factors depending only on
the mixing matrix. To the best of our knowledge, APD and APD-SC are the first
decentralized methods to achieve provable acceleration over unbalanced directed
graphs. Numerical experiments demonstrate the effectiveness of both methods.

    

### [[2107.12070] Robust Regularized Locality Preserving Indexing for Fiedler Vector Estimation](http://arxiv.org/abs/2107.12070)


  The Fiedler vector of a connected graph is the eigenvector associated with
the algebraic connectivity of the graph Laplacian and it provides substantial
information to learn the latent structure of a graph. In real-world
applications, however, the data may be subject to heavy-tailed noise and
outliers which results in deteriorations in the structure of the Fiedler vector
estimate. We design a Robust Regularized Locality Preserving Indexing (RRLPI)
method for Fiedler vector estimation that aims to approximate the nonlinear
manifold structure of the Laplace Beltrami operator while minimizing the
negative impact of outliers. First, an analysis of the effects of two
fundamental outlier types on the eigen-decomposition for block affinity
matrices which are essential in cluster analysis is conducted. Then, an error
model is formulated and a robust Fiedler vector estimation algorithm is
developed. An unsupervised penalty parameter selection algorithm is proposed
that leverages the geometric structure of the projection space to perform
robust regularized Fiedler estimation. The performance of RRLPI is benchmarked
against existing competitors in terms of detection probability, partitioning
quality, image segmentation capability, robustness and computation time using a
large variety of synthetic and real data experiments.

    

### [[2107.12078] 6DCNN with roto-translational convolution filters for volumetric data processing](http://arxiv.org/abs/2107.12078)


  In this work, we introduce 6D Convolutional Neural Network (6DCNN) designed
to tackle the problem of detecting relative positions and orientations of local
patterns when processing three-dimensional volumetric data. 6DCNN also includes
SE(3)-equivariant message-passing and nonlinear activation operations
constructed in the Fourier space. Working in the Fourier space allows
significantly reducing the computational complexity of our operations. We
demonstrate the properties of the 6D convolution and its efficiency in the
recognition of spatial patterns. We also assess the 6DCNN model on several
datasets from the recent CASP protein structure prediction challenges. Here,
6DCNN improves over the baseline architecture and also outperforms the state of
the art.

    

### [[2107.12079] An Argumentative Dialogue System for COVID-19 Vaccine Information](http://arxiv.org/abs/2107.12079)


  Dialogue systems are widely used in AI to support timely and interactive
communication with users. We propose a general-purpose dialogue system
architecture that leverages computational argumentation and state-of-the-art
language technologies. We illustrate and evaluate the system using a COVID-19
vaccine information case study.

    

### [[2107.12100] Predicting Influential Higher-Order Patterns in Temporal Network Data](http://arxiv.org/abs/2107.12100)


  Networks are frequently used to model complex systems comprised of
interacting elements. While links capture the topology of direct interactions,
the true complexity of many systems originates from higher-order patterns in
paths by which nodes can indirectly influence each other. Path data,
representing ordered sequences of consecutive direct interactions, can be used
to model these patterns. However, to avoid overfitting, such models should only
consider those higher-order patterns for which the data provide sufficient
statistical evidence. On the other hand, we hypothesise that network models,
which capture only direct interactions, underfit higher-order patterns present
in data. Consequently, both approaches are likely to misidentify influential
nodes in complex networks. We contribute to this issue by proposing eight
centrality measures based on MOGen, a multi-order generative model that
accounts for all paths up to a maximum distance but disregards paths at higher
distances. We compare MOGen-based centralities to equivalent measures for
network models and path data in a prediction experiment where we aim to
identify influential nodes in out-of-sample data. Our results show strong
evidence supporting our hypothesis. MOGen consistently outperforms both the
network model and path-based prediction. We further show that the performance
difference between MOGen and the path-based approach disappears if we have
sufficient observations, confirming that the error is due to overfitting.

    

### [[2107.12110] Combining Maximum-Likelihood with Deep Learning for Event Reconstruction in IceCube](http://arxiv.org/abs/2107.12110)


  The field of deep learning has become increasingly important for particle
physics experiments, yielding a multitude of advances, predominantly in event
classification and reconstruction tasks. Many of these applications have been
adopted from other domains. However, data in the field of physics are unique in
the context of machine learning, insofar as their generation process and the
laws and symmetries they abide by are usually well understood. Most commonly
used deep learning architectures fail at utilizing this available information.
In contrast, more traditional likelihood-based methods are capable of
exploiting domain knowledge, but they are often limited by computational
complexity. In this contribution, a hybrid approach is presented that utilizes
generative neural networks to approximate the likelihood, which may then be
used in a traditional maximum-likelihood setting. Domain knowledge, such as
invariances and detector characteristics, can easily be incorporated in this
approach. The hybrid approach is illustrated by the example of event
reconstruction in IceCube.

    

### [[2107.12137] AA3DNet: Attention Augmented Real Time 3D Object Detection](http://arxiv.org/abs/2107.12137)


  In this work, we address the problem of 3D object detection from point cloud
data in real time. For autonomous vehicles to work, it is very important for
the perception component to detect the real world objects with both high
accuracy and fast inference. We propose a novel neural network architecture
along with the training and optimization details for detecting 3D objects using
point cloud data. We present anchor design along with custom loss functions
used in this work. A combination of spatial and channel wise attention module
is used in this work. We use the Kitti 3D Birds Eye View dataset for
benchmarking and validating our results. Our method surpasses previous state of
the art in this domain both in terms of average precision and speed running at
> 30 FPS. Finally, we present the ablation study to demonstrate that the
performance of our network is generalizable. This makes it a feasible option to
be deployed in real time applications like self driving cars.

    

### [[2107.12156] Brain Inspired Computing Approach for the Optimization of the Thin Film Thickness of Polystyrene on the Glass Substrates](http://arxiv.org/abs/2107.12156)


  Advent in machine learning is leaving a deep impact on various sectors
including the material science domain. The present paper highlights the
application of various supervised machine learning regression algorithms such
as polynomial regression, decision tree regression algorithm, random forest
algorithm, support vector regression algorithm, and artificial neural network
algorithm to determine the thin film thickness of Polystyrene on the glass
substrates. The results showed that the polynomial regression machine learning
algorithm outperforms all other machine learning models by yielding the
coefficient of determination of 0.96 approximately and mean square error of
0.04 respectively.

    

### [[2107.12167] Multimodal Fusion Using Deep Learning Applied to Driver's Referencing of Outside-Vehicle Objects](http://arxiv.org/abs/2107.12167)


  There is a growing interest in more intelligent natural user interaction with
the car. Hand gestures and speech are already being applied for driver-car
interaction. Moreover, multimodal approaches are also showing promise in the
automotive industry. In this paper, we utilize deep learning for a multimodal
fusion network for referencing objects outside the vehicle. We use features
from gaze, head pose and finger pointing simultaneously to precisely predict
the referenced objects in different car poses. We demonstrate the practical
limitations of each modality when used for a natural form of referencing,
specifically inside the car. As evident from our results, we overcome the
modality specific limitations, to a large extent, by the addition of other
modalities. This work highlights the importance of multimodal sensing,
especially when moving towards natural user interaction. Furthermore, our user
based analysis shows noteworthy differences in recognition of user behavior
depending upon the vehicle pose.

    

### [[2107.12173] Membership Inference Attack and Defense for Wireless Signal Classifiers with Deep Learning](http://arxiv.org/abs/2107.12173)


  An over-the-air membership inference attack (MIA) is presented to leak
private information from a wireless signal classifier. Machine learning (ML)
provides powerful means to classify wireless signals, e.g., for PHY-layer
authentication. As an adversarial machine learning attack, the MIA infers
whether a signal of interest has been used in the training data of a target
classifier. This private information incorporates waveform, channel, and device
characteristics, and if leaked, can be exploited by an adversary to identify
vulnerabilities of the underlying ML model (e.g., to infiltrate the PHY-layer
authentication). One challenge for the over-the-air MIA is that the received
signals and consequently the RF fingerprints at the adversary and the intended
receiver differ due to the discrepancy in channel conditions. Therefore, the
adversary first builds a surrogate classifier by observing the spectrum and
then launches the black-box MIA on this classifier. The MIA results show that
the adversary can reliably infer signals (and potentially the radio and channel
information) used to build the target classifier. Therefore, a proactive
defense is developed against the MIA by building a shadow MIA model and fooling
the adversary. This defense can successfully reduce the MIA accuracy and
prevent information leakage from the wireless signal classifier.

    

### [[2107.12183] EGGS: Eigen-Gap Guided Search\\ Making Subspace Clustering Easy](http://arxiv.org/abs/2107.12183)


  The performance of spectral clustering heavily relies on the quality of
affinity matrix. A variety of affinity-matrix-construction methods have been
proposed but they have hyper-parameters to determine beforehand, which requires
strong experience and lead to difficulty in real applications especially when
the inter-cluster similarity is high or/and the dataset is large. On the other
hand, we often have to determine to use a linear model or a nonlinear model,
which still depends on experience. To solve these two problems, in this paper,
we present an eigen-gap guided search method for subspace clustering. The main
idea is to find the most reliable affinity matrix among a set of candidates
constructed by linear and kernel regressions, where the reliability is
quantified by the \textit{relative-eigen-gap} of graph Laplacian defined in
this paper. We show, theoretically and numerically, that the Laplacian matrix
with a larger relative-eigen-gap often yields a higher clustering accuracy and
stability. Our method is able to automatically search the best model and
hyper-parameters in a pre-defined space. The search space is very easy to
determine and can be arbitrarily large, though a relatively compact search
space can reduce the highly unnecessary computation. Our method has high
flexibility and convenience in real applications, and also has low
computational cost because the affinity matrix is not computed by iterative
optimization. We extend the method to large-scale datasets such as MNIST, on
which the time cost is less than 90s and the clustering accuracy is
state-of-the-art. Extensive experiments of natural image clustering show that
our method is more stable, accurate, and efficient than baseline methods.

    

### [[2107.12202] Black-Box Diagnosis and Calibration on GAN Intra-Mode Collapse: A Pilot Study](http://arxiv.org/abs/2107.12202)


  Generative adversarial networks (GANs) nowadays are capable of producing
images of incredible realism. One concern raised is whether the
state-of-the-art GAN's learned distribution still suffers from mode collapse,
and what to do if so. Existing diversity tests of samples from GANs are usually
conducted qualitatively on a small scale, and/or depends on the access to
original training data as well as the trained model parameters. This paper
explores to diagnose GAN intra-mode collapse and calibrate that, in a novel
black-box setting: no access to training data, nor the trained model
parameters, is assumed. The new setting is practically demanded, yet rarely
explored and significantly more challenging. As a first stab, we devise a set
of statistical tools based on sampling, that can visualize, quantify, and
rectify intra-mode collapse. We demonstrate the effectiveness of our proposed
diagnosis and calibration techniques, via extensive simulations and
experiments, on unconditional GAN image generation (e.g., face and vehicle).
Our study reveals that the intra-mode collapse is still a prevailing problem in
state-of-the-art GANs and the mode collapse is diagnosable and calibratable in
black-box settings. Our codes are available at:
this https URL.

    

### [[2107.12211] On The Impact of Client Sampling on Federated Learning Convergence](http://arxiv.org/abs/2107.12211)


  While clients' sampling is a central operation of current state-of-the-art
federated learning (FL) approaches, the impact of this procedure on the
convergence and speed of FL remains to date under-investigated. In this work we
introduce a novel decomposition theorem for the convergence of FL, allowing to
clearly quantify the impact of client sampling on the global model update.
Contrarily to previous convergence analyses, our theorem provides the exact
decomposition of a given convergence step, thus enabling accurate
considerations about the role of client sampling and heterogeneity. First, we
provide a theoretical ground for previously reported results on the
relationship between FL convergence and the variance of the aggregation
weights. Second, we prove for the first time that the quality of FL convergence
is also impacted by the resulting covariance between aggregation weights.
Third, we establish that the sum of the aggregation weights is another source
of slow-down and should be equal to 1 to improve FL convergence speed. Our
theory is general, and is here applied to Multinomial Distribution (MD) and
Uniform sampling, the two default client sampling in FL, and demonstrated
through a series of experiments in non-iid and unbalanced scenarios. Our
results suggest that MD sampling should be used as default sampling scheme, due
to the resilience to the changes in data ratio during the learning process,
while Uniform sampling is superior only in the special case when clients have
the same amount of data.

    

### [[2107.12216] Hindsight Value Function for Variance Reduction in Stochastic Dynamic Environment](http://arxiv.org/abs/2107.12216)


  Policy gradient methods are appealing in deep reinforcement learning but
suffer from high variance of gradient estimate. To reduce the variance, the
state value function is applied commonly. However, the effect of the state
value function becomes limited in stochastic dynamic environments, where the
unexpected state dynamics and rewards will increase the variance. In this
paper, we propose to replace the state value function with a novel hindsight
value function, which leverages the information from the future to reduce the
variance of the gradient estimate for stochastic dynamic environments.
Particularly, to obtain an ideally unbiased gradient estimate, we propose an
information-theoretic approach, which optimizes the embeddings of the future to
be independent of previous actions. In our experiments, we apply the proposed
hindsight value function in stochastic dynamic environments, including
discrete-action environments and continuous-action environments. Compared with
the standard state value function, the proposed hindsight value function
consistently reduces the variance, stabilizes the training, and improves the
eventual policy.

    

### [[2107.12220] Thought Flow Nets: From Single Predictions to Trains of Model Thought](http://arxiv.org/abs/2107.12220)


  When humans solve complex problems, they rarely come up with a decision
right-away. Instead, they start with an intuitive decision, reflect upon it,
spot mistakes, resolve contradictions and jump between different hypotheses.
Thus, they create a sequence of ideas and follow a train of thought that
ultimately reaches a conclusive decision. Contrary to this, today's neural
classification models are mostly trained to map an input to one single and
fixed output. In this paper, we investigate how we can give models the
opportunity of a second, third and $k$-th thought. We take inspiration from
Hegel's dialectics and propose a method that turns an existing classifier's
class prediction (such as the image class forest) into a sequence of
predictions (such as forest $\rightarrow$ tree $\rightarrow$ mushroom).
Concretely, we propose a correction module that is trained to estimate the
model's correctness as well as an iterative prediction update based on the
prediction's gradient. Our approach results in a dynamic system over class
probability distributions $\unicode{x2014}$ the thought flow. We evaluate our
method on diverse datasets and tasks from computer vision and natural language
processing. We observe surprisingly complex but intuitive behavior and
demonstrate that our method (i) can correct misclassifications, (ii)
strengthens model performance, (iii) is robust to high levels of adversarial
attacks, (iv) can increase accuracy up to 4% in a label-distribution-shift
setting and (iv) provides a tool for model interpretability that uncovers model
knowledge which otherwise remains invisible in a single distribution
prediction.

    

### [[2107.12224] Local2Global: Scaling global representation learning on graphs via local training](http://arxiv.org/abs/2107.12224)


  We propose a decentralised "local2global" approach to graph representation
learning, that one can a-priori use to scale any embedding technique. Our
local2global approach proceeds by first dividing the input graph into
overlapping subgraphs (or "patches") and training local representations for
each patch independently. In a second step, we combine the local
representations into a globally consistent representation by estimating the set
of rigid motions that best align the local representations using information
from the patch overlaps, via group synchronization. A key distinguishing
feature of local2global relative to existing work is that patches are trained
independently without the need for the often costly parameter synchronisation
during distributed training. This allows local2global to scale to large-scale
industrial applications, where the input graph may not even fit into memory and
may be stored in a distributed manner. Preliminary results on medium-scale data
sets (up to $\sim$7K nodes and $\sim$200K edges) are promising, with a graph
reconstruction performance for local2global that is comparable to that of
globally trained embeddings. A thorough evaluation of local2global on large
scale data and applications to downstream tasks, such as node classification
and link prediction, constitutes ongoing work.

    

### [[2107.12243] Protein-RNA interaction prediction with deep learning: Structure matters](http://arxiv.org/abs/2107.12243)


  Protein-RNA interactions are of vital importance to a variety of cellular
activities. Both experimental and computational techniques have been developed
to study the interactions. Due to the limitation of the previous database,
especially the lack of protein structure data, most of the existing
computational methods rely heavily on the sequence data, with only a small
portion of the methods utilizing the structural information. Recently,
AlphaFold has revolutionized the entire protein and biology field. Foreseeably,
the protein-RNA interaction prediction will also be promoted significantly in
the upcoming years. In this work, we give a thorough review of this field,
surveying both the binding site and binding preference prediction problems and
covering the commonly used datasets, features, and models. We also point out
the potential challenges and opportunities in this field. This survey
summarizes the development of the RBP-RNA interaction field in the past and
foresees its future development in the post-AlphaFold era.

    

### [[1806.03884] Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](http://arxiv.org/abs/1806.03884)


  Optimization algorithms that leverage gradient covariance information, such
as variants of natural gradient descent (Amari, 1998), offer the prospect of
yielding more effective descent directions. For models with many parameters,
the covariance matrix they are based on becomes gigantic, making them
inapplicable in their original form. This has motivated research into both
simple diagonal approximations and more sophisticated factored approximations
such as KFAC (Heskes, 2000; Martens & Grosse, 2015; Grosse & Martens, 2016). In
the present work we draw inspiration from both to propose a novel approximation
that is provably better than KFAC and amendable to cheap partial updates. It
consists in tracking a diagonal variance, not in parameter coordinates, but in
a Kronecker-factored eigenbasis, in which the diagonal approximation is likely
to be more effective. Experiments show improvements over KFAC in optimization
speed for several deep network architectures.

    

### [[1905.11488] Generalization Bounds in the Predict-then-Optimize Framework](http://arxiv.org/abs/1905.11488)


  The predict-then-optimize framework is fundamental in many practical
settings: predict the unknown parameters of an optimization problem, and then
solve the problem using the predicted values of the parameters. A natural loss
function in this environment is to consider the cost of the decisions induced
by the predicted parameters, in contrast to the prediction error of the
parameters. This loss function was recently introduced in Elmachtoub and Grigas
(2017) and referred to as the Smart Predict-then-Optimize (SPO) loss. In this
work, we seek to provide bounds on how well the performance of a prediction
model fit on training data generalizes out-of-sample, in the context of the SPO
loss. Since the SPO loss is non-convex and non-Lipschitz, standard results for
deriving generalization bounds do not apply.
We first derive bounds based on the Natarajan dimension that, in the case of
a polyhedral feasible region, scale at most logarithmically in the number of
extreme points, but, in the case of a general convex feasible region, have
linear dependence on the decision dimension. By exploiting the structure of the
SPO loss function and a key property of the feasible region, which we denote as
the strength property, we can dramatically improve the dependence on the
decision and feature dimensions. Our approach and analysis rely on placing a
margin around problematic predictions that do not yield unique optimal
solutions, and then providing generalization bounds in the context of a
modified margin SPO loss function that is Lipschitz continuous. Finally, we
characterize the strength property and show that the modified SPO loss can be
computed efficiently for both strongly convex bodies and polytopes with an
explicit extreme point representation.

    

### [[1906.00331] On Gradient Descent Ascent for Nonconvex-Concave Minimax Problems](http://arxiv.org/abs/1906.00331)


  We consider nonconvex-concave minimax problems, $\min_{\mathbf{x}}
\max_{\mathbf{y} \in \mathcal{Y}} f(\mathbf{x}, \mathbf{y})$, where $f$ is
nonconvex in $\mathbf{x}$ but concave in $\mathbf{y}$ and $\mathcal{Y}$ is a
convex and bounded set. One of the most popular algorithms for solving this
problem is the celebrated gradient descent ascent (GDA) algorithm, which has
been widely used in machine learning, control theory and economics. Despite the
extensive convergence results for the convex-concave setting, GDA with equal
stepsize can converge to limit cycles or even diverge in a general setting. In
this paper, we present the complexity results on two-time-scale GDA for solving
nonconvex-concave minimax problems, showing that the algorithm can find a
stationary point of the function $\Phi(\cdot) := \max_{\mathbf{y} \in
\mathcal{Y}} f(\cdot, \mathbf{y})$ efficiently. To the best our knowledge, this
is the first nonasymptotic analysis for two-time-scale GDA in this setting,
shedding light on its superior practical performance in training generative
adversarial networks (GANs) and other real applications.

    

### [[1911.05541] Vehicle-Rear: A New Dataset to Explore Feature Fusion for Vehicle Identification Using Convolutional Neural Networks](http://arxiv.org/abs/1911.05541)


  This work addresses the problem of vehicle identification through
non-overlapping cameras. As our main contribution, we introduce a novel dataset
for vehicle identification, called Vehicle-Rear, that contains more than three
hours of high-resolution videos, with accurate information about the make,
model, color and year of nearly 3,000 vehicles, in addition to the position and
identification of their license plates. To explore our dataset we design a
two-stream CNN that simultaneously uses two of the most distinctive and
persistent features available: the vehicle's appearance and its license plate.
This is an attempt to tackle a major problem: false alarms caused by vehicles
with similar designs or by very close license plate identifiers. In the first
network stream, shape similarities are identified by a Siamese CNN that uses a
pair of low-resolution vehicle patches recorded by two different cameras. In
the second stream, we use a CNN for OCR to extract textual information,
confidence scores, and string similarities from a pair of high-resolution
license plate patches. Then, features from both streams are merged by a
sequence of fully connected layers for decision. In our experiments, we
compared the two-stream network against several well-known CNN architectures
using single or multiple vehicle features. The architectures, trained models,
and dataset are publicly available at this https URL.

    

### [[1911.08756] Hierarchical Multiple-Instance Data Classification with Costly Features](http://arxiv.org/abs/1911.08756)


  We extend the framework of Classification with Costly Features (CwCF) that
works with samples of fixed dimensions to trees of varying depth and breadth
(similar to a JSON/XML file). In this setting, the sample is a tree - sets of
sets of features. Individually for each sample, the task is to sequentially
select informative features that help the classification. Each feature has a
real-valued cost, and the objective is to maximize accuracy while minimizing
the total cost. The process is modeled as an MDP where the states represent the
acquired features, and the actions select unknown features. We present a
specialized neural network architecture trained through deep reinforcement
learning that naturally fits the data and directly selects features in the
tree. We demonstrate our method in seven datasets and compare it to two
baselines.

    

### [[1912.10136] A vector-contraction inequality for Rademacher complexities using $p$-stable variables](http://arxiv.org/abs/1912.10136)


  Andreas Maurer in the paper "A vector-contraction inequality for Rademacher
complexities" extended the contraction inequality for Rademacher averages to
Lipschitz functions with vector-valued domains; He did it replacing the
Rademacher variables in the bounding expression by arbitrary idd symmetric and
sub-gaussian variables. We will see how to extend this work when we replace
sub-gaussian variables by $p$-stable variables for $1<p<2$.

    

### [[2002.02417] Near-Optimal Algorithms for Minimax Optimization](http://arxiv.org/abs/2002.02417)


  This paper resolves a longstanding open question pertaining to the design of
near-optimal first-order algorithms for smooth and
strongly-convex-strongly-concave minimax problems. Current state-of-the-art
first-order algorithms find an approximate Nash equilibrium using
$\tilde{O}(\kappa_{\mathbf x}+\kappa_{\mathbf y})$ or
$\tilde{O}(\min\{\kappa_{\mathbf x}\sqrt{\kappa_{\mathbf y}},
\sqrt{\kappa_{\mathbf x}}\kappa_{\mathbf y}\})$ gradient evaluations, where
$\kappa_{\mathbf x}$ and $\kappa_{\mathbf y}$ are the condition numbers for the
strong-convexity and strong-concavity assumptions. A gap still remains between
these results and the best existing lower bound
$\tilde{\Omega}(\sqrt{\kappa_{\mathbf x}\kappa_{\mathbf y}})$. This paper
presents the first algorithm with $\tilde{O}(\sqrt{\kappa_{\mathbf
x}\kappa_{\mathbf y}})$ gradient complexity, matching the lower bound up to
logarithmic factors. Our algorithm is designed based on an accelerated proximal
point method and an accelerated solver for minimax proximal steps. It can be
easily extended to the settings of strongly-convex-concave, convex-concave,
nonconvex-strongly-concave, and nonconvex-concave functions. This paper also
presents algorithms that match or outperform all existing methods in these
settings in terms of gradient complexity, up to logarithmic factors.

    

### [[2002.08260] On generalization in moment-based domain adaptation](http://arxiv.org/abs/2002.08260)


  Domain adaptation algorithms are designed to minimize the misclassification
risk of a discriminative model for a target domain with little training data by
adapting a model from a source domain with a large amount of training data.
Standard approaches measure the adaptation discrepancy based on distance
measures between the empirical probability distributions in the source and
target domain. In this setting, we address the problem of deriving
generalization bounds under practice-oriented general conditions on the
underlying probability distributions. As a result, we obtain generalization
bounds for domain adaptation based on finitely many moments and smoothness
conditions.

    

### [[2003.12618] Image compression optimized for 3D reconstruction by utilizing deep neural networks](http://arxiv.org/abs/2003.12618)


  Computer vision tasks are often expected to be executed on compressed images.
Classical image compression standards like JPEG 2000 are widely used. However,
they do not account for the specific end-task at hand. Motivated by works on
recurrent neural network (RNN)-based image compression and three-dimensional
(3D) reconstruction, we propose unified network architectures to solve both
tasks jointly. These joint models provide image compression tailored for the
specific task of 3D reconstruction. Images compressed by our proposed models,
yield 3D reconstruction performance superior as compared to using JPEG 2000
compression. Our models significantly extend the range of compression rates for
which 3D reconstruction is possible. We also show that this can be done highly
efficiently at almost no additional cost to obtain compression on top of the
computation already required for performing the 3D reconstruction task.

    

### [[2005.02077] Using Machine Learning to Emulate Agent-Based Simulations](http://arxiv.org/abs/2005.02077)


  In this proof-of-concept work, we evaluate the performance of multiple
machine-learning methods as statistical emulators for use in the analysis of
agent-based models (ABMs). Analysing ABM outputs can be challenging, as the
relationships between input parameters can be non-linear or even chaotic even
in relatively simple models, and each model run can require significant CPU
time. Statistical emulation, in which a statistical model of the ABM is
constructed to facilitate detailed model analyses, has been proposed as an
alternative to computationally costly Monte Carlo methods. Here we compare
multiple machine-learning methods for ABM emulation in order to determine the
approaches best suited to emulating the complex behaviour of ABMs. Our results
suggest that, in most scenarios, artificial neural networks (ANNs) and
gradient-boosted trees outperform Gaussian process emulators, currently the
most commonly used method for the emulation of complex computational models.
ANNs produced the most accurate model replications in scenarios with high
numbers of model runs, although training times were longer than the other
methods. We propose that agent-based modelling would benefit from using
machine-learning methods for emulation, as this can facilitate more robust
sensitivity analyses for the models while also reducing CPU time consumption
when calibrating and analysing the simulation.

    

### [[2006.04730] Picket: Guarding Against Corrupted Data in Tabular Data during Learning and Inference](http://arxiv.org/abs/2006.04730)


  Data corruption is an impediment to modern machine learning deployments.
Corrupted data can severely bias the learned model and can also lead to invalid
inferences. We present, Picket, a simple framework to safeguard against data
corruptions during both training and deployment of machine learning models over
tabular data. For the training stage, Picket identifies and removes corrupted
data points from the training data to avoid obtaining a biased model. For the
deployment stage, Picket flags, in an online manner, corrupted query points to
a trained machine learning model that due to noise will result in incorrect
predictions. To detect corrupted data, Picket uses a self-supervised deep
learning model for mixed-type tabular data, which we call PicketNet. To
minimize the burden of deployment, learning a PicketNet model does not require
any human-labeled data. Picket is designed as a plugin that can increase the
robustness of any machine learning pipeline. We evaluate Picket on a diverse
array of real-world data considering different corruption models that include
systematic and adversarial noise during both training and testing. We show that
Picket consistently safeguards against corrupted data during both training and
deployment of various models ranging from SVMs to neural networks, beating a
diverse array of competing methods that span from data quality validation
models to robust outlier-detection models.

    

### [[2006.05245] A Review of Automated Diagnosis of COVID-19 Based on Scanning Images](http://arxiv.org/abs/2006.05245)


  The pandemic of COVID-19 has caused millions of infections, which has led to
a great loss all over the world, socially and economically. Due to the
false-negative rate and the time-consuming of the conventional Reverse
Transcription Polymerase Chain Reaction (RT-PCR) tests, diagnosing based on
X-ray images and Computed Tomography (CT) images has been widely adopted.
Therefore, researchers of the computer vision area have developed many
automatic diagnosing models based on machine learning or deep learning to
assist the radiologists and improve the diagnosing accuracy. In this paper, we
present a review of these recently emerging automatic diagnosing models. 70
models proposed from February 14, 2020, to July 21, 2020, are involved. We
analyzed the models from the perspective of preprocessing, feature extraction,
classification, and evaluation. Based on the limitation of existing models, we
pointed out that domain adaption in transfer learning and interpretability
promotion would be the possible future directions.

    

### [[2006.07356] Implicit bias of gradient descent for mean squared error regression with wide neural networks](http://arxiv.org/abs/2006.07356)


  We investigate gradient descent training of wide neural networks and the
corresponding implicit bias in function space. For univariate regression, we
show that the solution of training a width-$n$ shallow ReLU network is within
$n^{- 1/2}$ of the function which fits the training data and whose difference
from the initial function has the smallest 2-norm of the second derivative
weighted by a curvature penalty that depends on the probability distribution
that is used to initialize the network parameters. We compute the curvature
penalty function explicitly for various common initialization procedures. For
instance, asymmetric initialization with a uniform distribution yields a
constant curvature penalty, and thence the solution function is the natural
cubic spline interpolation of the training data. We obtain a similar result for
different activation functions. For multivariate regression we show an
analogous result, whereby the second derivative is replaced by the Radon
transform of a fractional Laplacian. For initialization schemes that yield a
constant penalty function, the solutions are polyharmonic splines. Moreover, we
show that the training trajectories are captured by trajectories of smoothing
splines with decreasing regularization strength.

    

### [[2006.08601] Explaining Local, Global, And Higher-Order Interactions In Deep Learning](http://arxiv.org/abs/2006.08601)


  We present a simple yet highly generalizable method for explaining
interacting parts within a neural network's reasoning process. First, we design
an algorithm based on cross derivatives for computing statistical interaction
effects between individual features, which is generalized to both 2-way and
higher-order (3-way or more) interactions. We present results side by side with
a weight-based attribution technique, corroborating that cross derivatives are
a superior metric for both 2-way and higher-order interaction detection.
Moreover, we extend the use of cross derivatives as an explanatory device in
neural networks to the computer vision setting by expanding Grad-CAM, a popular
gradient-based explanatory tool for CNNs, to the higher order. While Grad-CAM
can only explain the importance of individual objects in images, our method,
which we call Taylor-CAM, can explain a neural network's relational reasoning
across multiple objects. We show the success of our explanations both
qualitatively and quantitatively, including with a user study. We will release
all code as a tool package to facilitate explainable deep learning.

    

### [[2006.16241] The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization](http://arxiv.org/abs/2006.16241)


  We introduce four new real-world distribution shift datasets consisting of
changes in image style, image blurriness, geographic location, camera
operation, and more. With our new datasets, we take stock of previously
proposed methods for improving out-of-distribution robustness and put them to
the test. We find that using larger models and artificial data augmentations
can improve robustness on real-world distribution shifts, contrary to claims in
prior work. We find improvements in artificial robustness benchmarks can
transfer to real-world distribution shifts, contrary to claims in prior work.
Motivated by our observation that data augmentations can help with real-world
distribution shifts, we also introduce a new data augmentation method which
advances the state-of-the-art and outperforms models pretrained with 1000 times
more labeled data. Overall we find that some methods consistently help with
distribution shifts in texture and local image statistics, but these methods do
not help with some other distribution shifts like geographic changes. Our
results show that future research must study multiple distribution shifts
simultaneously, as we demonstrate that no evaluated method consistently
improves robustness.

    

### [[2007.03983] Dynamic social learning under graph constraints](http://arxiv.org/abs/2007.03983)


  We introduce a model of graph-constrained dynamic choice with reinforcement
modeled by positively $\alpha$-homogeneous rewards. We show that its empirical
process, which can be written as a stochastic approximation recursion with
Markov noise, has the same probability law as a certain vertex reinforced
random walk. We use this equivalence to show that for $\alpha > 0$, the
asymptotic outcome concentrates around the optimum in a certain limiting sense
when `annealed' by letting $\alpha\uparrow\infty$ slowly.

    

### [[2007.06557] Prediction-Centric Learning of Independent Cascade Dynamics from Partial Observations](http://arxiv.org/abs/2007.06557)


  Spreading processes play an increasingly important role in modeling for
diffusion networks, information propagation, marketing and opinion setting. We
address the problem of learning of a spreading model such that the predictions
generated from this model are accurate and could be subsequently used for the
optimization, and control of diffusion dynamics. We focus on a challenging
setting where full observations of the dynamics are not available, and standard
approaches such as maximum likelihood quickly become intractable for large
network instances. We introduce a computationally efficient algorithm, based on
a scalable dynamic message-passing approach, which is able to learn parameters
of the effective spreading model given only limited information on the
activation times of nodes in the network. The popular Independent Cascade model
is used to illustrate our approach. We show that tractable inference from the
learned model generates a better prediction of marginal probabilities compared
to the original model. We develop a systematic procedure for learning a mixture
of models which further improves the prediction quality.

    

### [[2007.06963] P-KDGAN: Progressive Knowledge Distillation with GANs for One-class Novelty Detection](http://arxiv.org/abs/2007.06963)


  One-class novelty detection is to identify anomalous instances that do not
conform to the expected normal instances. In this paper, the Generative
Adversarial Networks (GANs) based on encoder-decoder-encoder pipeline are used
for detection and achieve state-of-the-art performance. However, deep neural
networks are too over-parameterized to deploy on resource-limited devices.
Therefore, Progressive Knowledge Distillation with GANs (PKDGAN) is proposed to
learn compact and fast novelty detection networks. The P-KDGAN is a novel
attempt to connect two standard GANs by the designed distillation loss for
transferring knowledge from the teacher to the student. The progressive
learning of knowledge distillation is a two-step approach that continuously
improves the performance of the student GAN and achieves better performance
than single step methods. In the first step, the student GAN learns the basic
knowledge totally from the teacher via guiding of the pretrained teacher GAN
with fixed weights. In the second step, joint fine-training is adopted for the
knowledgeable teacher and student GANs to further improve the performance and
stability. The experimental results on CIFAR-10, MNIST, and FMNIST show that
our method improves the performance of the student GAN by 2.44%, 1.77%, and
1.73% when compressing the computation at ratios of 24.45:1, 311.11:1, and
700:1, respectively.

    

### [[2008.02275] Aligning AI With Shared Human Values](http://arxiv.org/abs/2008.02275)


  We show how to assess a language model's knowledge of basic concepts of
morality. We introduce the ETHICS dataset, a new benchmark that spans concepts
in justice, well-being, duties, virtues, and commonsense morality. Models
predict widespread moral judgments about diverse text scenarios. This requires
connecting physical and social world knowledge to value judgements, a
capability that may enable us to steer chatbot outputs or eventually regularize
open-ended reinforcement learning agents. With the ETHICS dataset, we find that
current language models have a promising but incomplete ability to predict
basic human ethical judgements. Our work shows that progress can be made on
machine ethics today, and it provides a steppingstone toward AI that is aligned
with human values.

    

### [[2009.04891] Meta-Learning with Sparse Experience Replay for Lifelong Language Learning](http://arxiv.org/abs/2009.04891)


  Lifelong learning requires models that can continuously learn from sequential
streams of data without suffering catastrophic forgetting due to shifts in data
distributions. Deep learning models have thrived in the non-sequential learning
paradigm; however, when used to learn a sequence of tasks, they fail to retain
past knowledge and learn incrementally. We propose a novel approach to lifelong
learning of language tasks based on meta-learning with sparse experience replay
that directly optimizes to prevent forgetting. We show that under the realistic
setting of performing a single pass on a stream of tasks and without any task
identifiers, our method obtains state-of-the-art results on lifelong text
classification and relation extraction. We analyze the effectiveness of our
approach and further demonstrate its low computational and space complexity.

    

### [[2009.07672] The Dark (and Bright) Side of IoT: Attacks and Countermeasures for Identifying Smart Home Devices and Services](http://arxiv.org/abs/2009.07672)


  We present a new machine learning-based attack that exploits network patterns
to detect the presence of smart IoT devices and running services in the WiFi
radio spectrum. We perform an extensive measurement campaign of data
collection, and we build up a model describing the traffic patterns
characterizing three popular IoT smart home devices, i.e., Google Nest Mini,
Amazon Echo, and Amazon Echo Dot. We prove that it is possible to detect and
identify with overwhelming probability their presence and the services running
by the aforementioned devices in a crowded WiFi scenario. This work proves that
standard encryption techniques alone are not sufficient to protect the privacy
of the end-user, since the network traffic itself exposes the presence of both
the device and the associated service. While more work is required to prevent
non-trusted third parties to detect and identify the user's devices, we
introduce Eclipse, a technique to mitigate these types of attacks, which
reshapes the traffic making the identification of the devices and the
associated services similar to the random classification baseline.

    

### [[2009.10588] Anomalous diffusion dynamics of learning in deep neural networks](http://arxiv.org/abs/2009.10588)


  Learning in deep neural networks (DNNs) is implemented through minimizing a
highly non-convex loss function, typically by a stochastic gradient descent
(SGD) method. This learning process can effectively find good wide minima
without being trapped in poor local ones. We present a novel account of how
such effective deep learning emerges through the interactions of the SGD and
the geometrical structure of the loss landscape. Rather than being a normal
diffusion process (i.e. Brownian motion) as often assumed, we find that the SGD
exhibits rich, complex dynamics when navigating through the loss landscape;
initially, the SGD exhibits anomalous superdiffusion, which attenuates
gradually and changes to subdiffusion at long times when the solution is
reached. Such learning dynamics happen ubiquitously in different DNNs such as
ResNet and VGG-like networks and are insensitive to batch size and learning
rate. The anomalous superdiffusion process during the initial learning phase
indicates that the motion of SGD along the loss landscape possesses
intermittent, big jumps; this non-equilibrium property enables the SGD to
escape from sharp local minima. By adapting the methods developed for studying
energy landscapes in complex physical systems, we find that such superdiffusive
learning dynamics are due to the interactions of the SGD and the fractal-like
structure of the loss landscape. We further develop a simple model to
demonstrate the mechanistic role of the fractal loss landscape in enabling the
SGD to effectively find global minima. Our results thus reveal the
effectiveness of deep learning from a novel perspective and have implications
for designing efficient deep neural networks.

    

### [[2009.12462] Symbolic Relational Deep Reinforcement Learning based on Graph Neural Networks](http://arxiv.org/abs/2009.12462)


  We focus on reinforcement learning (RL) in relational problems that are
naturally defined in terms of objects, their relations, and manipulations.
These problems are characterized by variable state and action spaces, and
finding a fixed-length representation, required by most existing RL methods, is
difficult, if not impossible. We present a deep RL framework based on graph
neural networks and auto-regressive policy decomposition that naturally works
with these problems and is completely domain-independent. We demonstrate the
framework in three very distinct domains and we report the method's competitive
performance and impressive zero-shot generalization over different problem
sizes. In goal-oriented BlockWorld, we demonstrate multi-parameter actions with
pre-conditions. In SysAdmin, we show how to select multiple objects
simultaneously. In the classical planning domain of Sokoban, the method trained
exclusively on 10x10 problems with three boxes solves 89% of 15x15 problems
with five boxes.

    

### [[2009.12920] Privacy-Preserving Dynamic Personalized Pricing with Demand Learning](http://arxiv.org/abs/2009.12920)


  The prevalence of e-commerce has made detailed customers' personal
information readily accessible to retailers, and this information has been
widely used in pricing decisions. When involving personalized information, how
to protect the privacy of such information becomes a critical issue in
practice. In this paper, we consider a dynamic pricing problem over $T$ time
periods with an \emph{unknown} demand function of posted price and personalized
information. At each time $t$, the retailer observes an arriving customer's
personal information and offers a price. The customer then makes the purchase
decision, which will be utilized by the retailer to learn the underlying demand
function. There is potentially a serious privacy concern during this process: a
third party agent might infer the personalized information and purchase
decisions from price changes from the pricing system. Using the fundamental
framework of differential privacy from computer science, we develop a
privacy-preserving dynamic pricing policy, which tries to maximize the retailer
revenue while avoiding information leakage of individual customer's information
and purchasing decisions. To this end, we first introduce a notion of
\emph{anticipating} $(\varepsilon, \delta)$-differential privacy that is
tailored to dynamic pricing problem. Our policy achieves both the privacy
guarantee and the performance guarantee in terms of regret. Roughly speaking,
for $d$-dimensional personalized information, our algorithm achieves the
expected regret at the order of $\tilde{O}(\varepsilon^{-1} \sqrt{d^3 T})$,
when the customers' information is adversarially chosen. For stochastic
personalized information, the regret bound can be further improved to
$\tilde{O}(\sqrt{d^2T} + \varepsilon^{-2} d^2)$

    

### [[2010.02838] A Closer Look at Codistillation for Distributed Training](http://arxiv.org/abs/2010.02838)


  Codistillation has been proposed as a mechanism to share knowledge among
concurrently trained models by encouraging them to represent the same function
through an auxiliary loss. This contrasts with the more commonly used
fully-synchronous data-parallel stochastic gradient descent methods, where
different model replicas average their gradients (or parameters) at every
iteration and thus maintain identical parameters. We investigate codistillation
in a distributed training setup, complementing previous work which focused on
extremely large batch sizes. Surprisingly, we find that even at moderate batch
sizes, models trained with codistillation can perform as well as models trained
with synchronous data-parallel methods, despite using a much weaker
synchronization mechanism. These findings hold across a range of batch sizes
and learning rate schedules, as well as different kinds of models and datasets.
Obtaining this level of accuracy, however, requires properly accounting for the
regularization effect of codistillation, which we highlight through several
empirical observations. Overall, this work contributes to a better
understanding of codistillation and how to best take advantage of it in a
distributed computing environment.

    

### [[2010.03161] Model-Free Non-Stationary RL: Near-Optimal Regret and Applications in Multi-Agent RL and Inventory Control](http://arxiv.org/abs/2010.03161)


  We consider model-free reinforcement learning (RL) in non-stationary Markov
decision processes. Both the reward functions and the state transition
functions are allowed to vary arbitrarily over time as long as their cumulative
variations do not exceed certain variation budgets. We propose Restarted
Q-Learning with Upper Confidence Bounds (RestartQ-UCB), the first model-free
algorithm for non-stationary RL, and show that it outperforms existing
solutions in terms of dynamic regret. Specifically, RestartQ-UCB with
Freedman-type bonus terms achieves a dynamic regret bound of
$\widetilde{O}(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}} H
T^{\frac{2}{3}})$, where $S$ and $A$ are the numbers of states and actions,
respectively, $\Delta>0$ is the variation budget, $H$ is the number of time
steps per episode, and $T$ is the total number of time steps. We further
present a parameter-free algorithm named Double-Restart Q-UCB that does not
require prior knowledge of the variation budget. We show that our algorithms
are \emph{nearly optimal} by establishing an information-theoretical lower
bound of $\Omega(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}}
H^{\frac{2}{3}} T^{\frac{2}{3}})$, the first lower bound in non-stationary RL.
Numerical experiments validate the advantages of RestartQ-UCB in terms of both
cumulative rewards and computational efficiency. We demonstrate the power of
our results in examples of multi-agent RL and inventory control across related
products.

    

### [[2010.06870] FedGroup: Efficient Clustered Federated Learning via Decomposed Data-Driven Measure](http://arxiv.org/abs/2010.06870)


  Federated Learning (FL) enables the multiple participating devices to
collaboratively contribute to a global neural network model while keeping the
training data locally. Unlike the centralized training setting, the non-IID and
imbalanced (statistical heterogeneity) training data of FL is distributed in
the federated network, which will increase the divergences between the local
models and global model, further degrading performance. In this paper, we
propose a novel clustered federated learning (CFL) framework FedGroup, in which
we 1) group the training of clients based on the similarities between the
clients' optimization directions for high training performance; 2) construct a
new data-driven distance measure to improve the efficiency of the client
clustering procedure. 3) implement a newcomer device cold start mechanism based
on the auxiliary global model for framework scalability and practicality.
FedGroup can achieve improvements by dividing joint optimization into groups
of sub-optimization and can be combined with FL optimizer FedProx. The
convergence and complexity are analyzed to demonstrate the efficiency of our
proposed framework. We also evaluate FedGroup and FedGrouProx (combined with
FedProx) on several open datasets and made comparisons with related CFL
frameworks. The results show that FedGroup can significantly improve absolute
test accuracy by +14.1% on FEMNIST compared to FedAvg. +3.4% on Sentiment140
compared to FedProx, +6.9% on MNIST compared to FeSEM.

    

### [[2010.09370] Probabilistic selection of inducing points in sparse Gaussian processes](http://arxiv.org/abs/2010.09370)


  Sparse Gaussian processes and various extensions thereof are enabled through
inducing points, that simultaneously bottleneck the predictive capacity and act
as the main contributor towards model complexity. However, the number of
inducing points is generally not associated with uncertainty which prevents us
from applying the apparatus of Bayesian reasoning for identifying an
appropriate trade-off. In this work we place a point process prior on the
inducing points and approximate the associated posterior through stochastic
variational inference. By letting the prior encourage a moderate number of
inducing points, we enable the model to learn which and how many points to
utilise. We experimentally show that fewer inducing points are preferred by the
model as the points become less informative, and further demonstrate how the
method can be employed in deep Gaussian processes and latent variable
modelling.

    

### [[2010.09559] Multilayer Network Analysis for Improved Credit Risk Prediction](http://arxiv.org/abs/2010.09559)


  We present a multilayer network model for credit risk assessment. Our model
accounts for multiple connections between borrowers (such as their geographic
location and their economic activity) and allows for explicitly modelling the
interaction between connected borrowers. We develop a multilayer personalized
PageRank algorithm that allows quantifying the strength of the default exposure
of any borrower in the network. We test our methodology in an agricultural
lending framework, where it has been suspected for a long time default
correlates between borrowers when they are subject to the same structural
risks. Our results show there are significant predictive gains just by
including centrality multilayer network information in the model, and these
gains are increased by more complex information such as the multilayer PageRank
variables. The results suggest default risk is highest when an individual is
connected to many defaulters, but this risk is mitigated by the size of the
neighbourhood of the individual, showing both default risk and financial
stability propagate throughout the network.

    

### [[2010.11697] A Data Set and a Convolutional Model for Iconography Classification in Paintings](http://arxiv.org/abs/2010.11697)


  Iconography in art is the discipline that studies the visual content of
artworks to determine their motifs and themes andto characterize the way these
are represented. It is a subject of active research for a variety of purposes,
including the interpretation of meaning, the investigation of the origin and
diffusion in time and space of representations, and the study of influences
across artists and art works. With the proliferation of digital archives of art
images, the possibility arises of applying Computer Vision techniques to the
analysis of art images at an unprecedented scale, which may support iconography
research and education. In this paper we introduce a novel paintings data set
for iconography classification and present the quantitativeand qualitative
results of applying a Convolutional Neural Network (CNN) classifier to the
recognition of the iconography of artworks. The proposed classifier achieves
good performances (71.17% Precision, 70.89% Recall, 70.25% F1-Score and 72.73%
Average Precision) in the task of identifying saints in Christian religious
paintings, a task made difficult by the presence of classes with very similar
visual features. Qualitative analysis of the results shows that the CNN focuses
on the traditional iconic motifs that characterize the representation of each
saint and exploits such hints to attain correct identification. The ultimate
goal of our work is to enable the automatic extraction, decomposition, and
comparison of iconography elements to support iconographic studies and
automatic art work annotation.

    

### [[2010.12167] Approximation Theory Based Methods for RKHS Bandits](http://arxiv.org/abs/2010.12167)


  The RKHS bandit problem (also called kernelized multi-armed bandit problem)
is an online optimization problem of non-linear functions with noisy feedback.
Although the problem has been extensively studied, there are unsatisfactory
results for some problems compared to the well-studied linear bandit case.
Specifically, there is no general algorithm for the adversarial RKHS bandit
problem. In addition, high computational complexity of existing algorithms
hinders practical application. We address these issues by considering a novel
amalgamation of approximation theory and the misspecified linear bandit
problem. Using an approximation method, we propose efficient algorithms for the
stochastic RKHS bandit problem and the first general algorithm for the
adversarial RKHS bandit problem. Furthermore, we empirically show that one of
our proposed methods has comparable cumulative regret to IGP-UCB and its
running time is much shorter.

    

### [[2010.12487] An Analysis of LIME for Text Data](http://arxiv.org/abs/2010.12487)


  Text data are increasingly handled in an automated fashion by machine
learning algorithms. But the models handling these data are not always
well-understood due to their complexity and are more and more often referred to
as "black-boxes." Interpretability methods aim to explain how these models
operate. Among them, LIME has become one of the most popular in recent years.
However, it comes without theoretical guarantees: even for simple models, we
are not sure that LIME behaves accurately. In this paper, we provide a first
theoretical analysis of LIME for text data. As a consequence of our theoretical
findings, we show that LIME indeed provides meaningful explanations for simple
models, namely decision trees and linear models.

    

### [[2010.14694] Deep Learning for Individual Heterogeneity: An Automatic Inference Framework](http://arxiv.org/abs/2010.14694)


  We develop methodology for estimation and inference using machine learning to
enrich economic models. Our framework takes a standard economic model and
recasts the parameters as fully flexible nonparametric functions, to capture
the rich heterogeneity based on potentially high dimensional or complex
observable characteristics. These "parameter functions" retain the
interpretability, economic meaning, and discipline of classical parameters.
Deep learning is particularly well-suited to structured modeling of
heterogeneity in economics. We show how to design the network architecture to
match the structure of the economic model, delivering novel methodology that
moves deep learning beyond prediction. We prove convergence rates for the
estimated parameter functions. These functions are the key inputs into the
finite-dimensional parameter of inferential interest. We obtain inference based
on a novel influence function calculation that covers any second-stage
parameter and any machine-learning-enriched model that uses a smooth
per-observation loss function. No additional derivations are required. The
score can be taken directly to data, using automatic differentiation if needed.
The researcher need only define the original model and define the parameter of
interest. A key insight is that we need not write down the influence function
in order to evaluate it on the data. Our framework gives new results for a host
of contexts, covering such diverse examples as price elasticities,
willingness-to-pay, and surplus measures in binary or multinomial choice
models, effects of continuous treatment variables, fractional outcome models,
count data, heterogeneous production functions, and more. We apply our
methodology to a large scale advertising experiment for short-term loans. We
show how economically meaningful estimates and inferences can be made that
would be unavailable without our results.

    

### [[2011.00384] Predictive Monitoring with Logic-Calibrated Uncertainty for Cyber-Physical Systems](http://arxiv.org/abs/2011.00384)


  Predictive monitoring -- making predictions about future states and
monitoring if the predicted states satisfy requirements -- offers a promising
paradigm in supporting the decision making of Cyber-Physical Systems (CPS).
Existing works of predictive monitoring mostly focus on monitoring individual
predictions rather than sequential predictions. We develop a novel approach for
monitoring sequential predictions generated from Bayesian Recurrent Neural
Networks (RNNs) that can capture the inherent uncertainty in CPS, drawing on
insights from our study of real-world CPS datasets. We propose a new logic
named \emph{Signal Temporal Logic with Uncertainty} (STL-U) to monitor a
flowpipe containing an infinite set of uncertain sequences predicted by
Bayesian RNNs. We define STL-U strong and weak satisfaction semantics based on
if all or some sequences contained in a flowpipe satisfy the requirement. We
also develop methods to compute the range of confidence levels under which a
flowpipe is guaranteed to strongly (weakly) satisfy an STL-U formula.
Furthermore, we develop novel criteria that leverage STL-U monitoring results
to calibrate the uncertainty estimation in Bayesian RNNs. Finally, we evaluate
the proposed approach via experiments with real-world datasets and a simulated
smart city case study, which show very encouraging results of STL-U based
predictive monitoring approach outperforming baselines.

    

### [[2011.03622] Settling the Robust Learnability of Mixtures of Gaussians](http://arxiv.org/abs/2011.03622)


  This work represents a natural coalescence of two important lines of work:
learning mixtures of Gaussians and algorithmic robust statistics. In particular
we give the first provably robust algorithm for learning mixtures of any
constant number of Gaussians. We require only mild assumptions on the mixing
weights (bounded fractionality) and that the total variation distance between
components is bounded away from zero. At the heart of our algorithm is a new
method for proving dimension-independent polynomial identifiability through
applying a carefully chosen sequence of differential operations to certain
generating functions that not only encode the parameters we would like to learn
but also the system of polynomial equations we would like to solve. We show how
the symbolic identities we derive can be directly used to analyze a natural
sum-of-squares relaxation.

    

### [[2011.04929] Sim2Sim Evaluation of a Novel Data-Efficient Differentiable Physics Engine for Tensegrity Robots](http://arxiv.org/abs/2011.04929)


  Learning policies in simulation is promising for reducing human effort when
training robot controllers. This is especially true for soft robots that are
more adaptive and safe but also more difficult to accurately model and control.
The sim2real gap is the main barrier to successfully transfer policies from
simulation to a real robot. System identification can be applied to reduce this
gap but traditional identification methods require a lot of manual tuning.
Data-driven alternatives can tune dynamical models directly from data but are
often data hungry, which also incorporates human effort in collecting data.
This work proposes a data-driven, end-to-end differentiable simulator focused
on the exciting but challenging domain of tensegrity robots. To the best of the
authors' knowledge, this is the first differentiable physics engine for
tensegrity robots that supports cable, contact, and actuation modeling. The aim
is to develop a reasonably simplified, data-driven simulation, which can learn
approximate dynamics with limited ground truth data. The dynamics must be
accurate enough to generate policies that can be transferred back to the
ground-truth system. As a first step in this direction, the current work
demonstrates sim2sim transfer, where the unknown physical model of MuJoCo acts
as a ground truth system. Two different tensegrity robots are used for
evaluation and learning of locomotion policies, a 6-bar and a 3-bar tensegrity.
The results indicate that only 0.25\% of ground truth data are needed to train
a policy that works on the ground truth system when the differentiable engine
is used for training against training the policy directly on the ground truth
system.

    

### [[2011.08181] Explaining the Adaptive Generalisation Gap](http://arxiv.org/abs/2011.08181)


  We conjecture that the inherent difference in generalisation between adaptive
and non-adaptive gradient methods stems from the increased estimation noise in
the flattest directions of the true loss surface. We demonstrate that typical
schedules used for adaptive methods (with low numerical stability or damping
constants) serve to bias relative movement towards flat directions relative to
sharp directions, effectively amplifying the noise-to-signal ratio and harming
generalisation. We further demonstrate that the numerical stability/damping
constant used in these methods can be decomposed into a learning rate reduction
and linear shrinkage of the estimated curvature matrix. We then demonstrate
significant generalisation improvements by increasing the shrinkage
coefficient, closing the generalisation gap entirely in both Logistic
Regression and Deep Neural Network experiments. Finally, we show that other
popular modifications to adaptive methods, such as decoupled weight decay and
partial adaptivity can be shown to calibrate parameter updates to make better
use of sharper, more reliable directions.

    

### [[2011.08843] Design Space for Graph Neural Networks](http://arxiv.org/abs/2011.08843)


  The rapid evolution of Graph Neural Networks (GNNs) has led to a growing
number of new architectures as well as novel applications. However, current
research focuses on proposing and evaluating specific architectural designs of
GNNs, as opposed to studying the more general design space of GNNs that
consists of a Cartesian product of different design dimensions, such as the
number of layers or the type of the aggregation function. Additionally, GNN
designs are often specialized to a single task, yet few efforts have been made
to understand how to quickly find the best GNN design for a novel task or a
novel dataset. Here we define and systematically study the architectural design
space for GNNs which consists of 315,000 different designs over 32 different
predictive tasks. Our approach features three key innovations: (1) A general
GNN design space; (2) a GNN task space with a similarity metric, so that for a
given novel task/dataset, we can quickly identify/transfer the best performing
architecture; (3) an efficient and effective design space evaluation method
which allows insights to be distilled from a huge number of model-task
combinations. Our key results include: (1) A comprehensive set of guidelines
for designing well-performing GNNs; (2) while best GNN designs for different
tasks vary significantly, the GNN task space allows for transferring the best
designs across different tasks; (3) models discovered using our design space
achieve state-of-the-art performance. Overall, our work offers a principled and
scalable approach to transition from studying individual GNN designs for
specific tasks, to systematically studying the GNN design space and the task
space. Finally, we release GraphGym, a powerful platform for exploring
different GNN designs and tasks. GraphGym features modularized GNN
implementation, standardized GNN evaluation, and reproducible and scalable
experiment management.

    

### [[2012.12573] Automated Lay Language Summarization of Biomedical Scientific Reviews](http://arxiv.org/abs/2012.12573)


  Health literacy has emerged as a crucial factor in making appropriate health
decisions and ensuring treatment outcomes. However, medical jargon and the
complex structure of professional language in this domain make health
information especially hard to interpret. Thus, there is an urgent unmet need
for automated methods to enhance the accessibility of the biomedical literature
to the general population. This problem can be framed as a type of translation
problem between the language of healthcare professionals, and that of the
general public. In this paper, we introduce the novel task of automated
generation of lay language summaries of biomedical scientific reviews, and
construct a dataset to support the development and evaluation of automated
methods through which to enhance the accessibility of the biomedical
literature. We conduct analyses of the various challenges in solving this task,
including not only summarization of the key points but also explanation of
background knowledge and simplification of professional language. We experiment
with state-of-the-art summarization models as well as several data augmentation
techniques, and evaluate their performance using both automated metrics and
human assessment. Results indicate that automatically generated summaries
produced using contemporary neural architectures can achieve promising quality
and readability as compared with reference summaries developed for the lay
public by experts (best ROUGE-L of 50.24 and Flesch-Kincaid readability score
of 13.30). We also discuss the limitations of the current attempt, providing
insights and directions for future work.

    

### [[2012.14353] DeepHateExplainer: Explainable Hate Speech Detection in Under-resourced Bengali Language](http://arxiv.org/abs/2012.14353)


  The exponential growths of social media and micro-blogging sites not only
provide platforms for empowering freedom of expressions and individual voices,
but also enables people to express anti-social behavior like online harassment,
cyberbullying, and hate speech. Numerous works have been proposed to utilize
textual data for social and anti-social behavior analysis, by predicting the
contexts mostly for highly-resourced languages like English. However, some
languages are under-resourced, e.g., South Asian languages like Bengali, that
lack computational resources for accurate natural language processing (NLP). In
this paper, we propose an explainable approach for hate speech detection from
the under-resourced Bengali language, which we called DeepHateExplainer.
Bengali texts are first comprehensively preprocessed, before classifying them
into political, personal, geopolitical, and religious hates using a neural
ensemble method of transformer-based neural architectures (i.e., monolingual
Bangla BERT-base, multilingual BERT-cased/uncased, and XLM-RoBERTa).
Important~(most and least) terms are then identified using sensitivity analysis
and layer-wise relevance propagation~(LRP), before providing
human-interpretable explanations. Finally, we compute comprehensiveness and
sufficiency scores to measure the quality of explanations w.r.t faithfulness.
Evaluations against machine learning~(linear and tree-based models) and neural
networks (i.e., CNN, Bi-LSTM, and Conv-LSTM with word embeddings) baselines
yield F1-scores of 78%, 91%, 89%, and 84%, for political, personal,
geopolitical, and religious hates, respectively, outperforming both ML and DNN
baselines.

    

### [[2012.15180] Out of Order: How Important Is The Sequential Order of Words in a Sentence in Natural Language Understanding Tasks?](http://arxiv.org/abs/2012.15180)


  Do state-of-the-art natural language understanding models care about word
order - one of the most important characteristics of a sequence? Not always! We
found 75% to 90% of the correct predictions of BERT-based classifiers, trained
on many GLUE tasks, remain constant after input words are randomly shuffled.
Despite BERT embeddings are famously contextual, the contribution of each
individual word to downstream tasks is almost unchanged even after the word's
context is shuffled. BERT-based models are able to exploit superficial cues
(e.g. the sentiment of keywords in sentiment analysis; or the word-wise
similarity between sequence-pair inputs in natural language inference) to make
correct decisions when tokens are arranged in random orders. Encouraging
classifiers to capture word order information improves the performance on most
GLUE tasks, SQuAD 2.0 and out-of-samples. Our work suggests that many GLUE
tasks are not challenging machines to understand the meaning of a sentence.

    

### [[2101.09451] Error Diffusion Halftoning Against Adversarial Examples](http://arxiv.org/abs/2101.09451)


  Adversarial examples contain carefully crafted perturbations that can fool
deep neural networks (DNNs) into making wrong predictions. Enhancing the
adversarial robustness of DNNs has gained considerable interest in recent
years. Although image transformation-based defenses were widely considered at
an earlier time, most of them have been defeated by adaptive attacks. In this
paper, we propose a new image transformation defense based on error diffusion
halftoning, and combine it with adversarial training to defend against
adversarial examples. Error diffusion halftoning projects an image into a 1-bit
space and diffuses quantization error to neighboring pixels. This process can
remove adversarial perturbations from a given image while maintaining
acceptable image quality in the meantime in favor of recognition. Experimental
results demonstrate that the proposed method is able to improve adversarial
robustness even under advanced adaptive attacks, while most of the other image
transformation-based defenses do not. We show that a proper image
transformation can still be an effective defense approach. Code:
this https URL


### [[2101.11517] Investigating Bi-Level Optimization for Learning and Vision from a Unified Perspective: A Survey and Beyond](http://arxiv.org/abs/2101.11517)


  Bi-Level Optimization (BLO) is originated from the area of economic game
theory and then introduced into the optimization community. BLO is able to
handle problems with a hierarchical structure, involving two levels of
optimization tasks, where one task is nested inside the other. In machine
learning and computer vision fields, despite the different motivations and
mechanisms, a lot of complex problems, such as hyper-parameter optimization,
multi-task and meta-learning, neural architecture search, adversarial learning
and deep reinforcement learning, actually all contain a series of closely
related subproblms. In this paper, we first uniformly express these complex
learning and vision problems from the perspective of BLO. Then we construct a
best-response-based single-level reformulation and establish a unified
algorithmic framework to understand and formulate mainstream gradient-based BLO
methodologies, covering aspects ranging from fundamental automatic
differentiation schemes to various accelerations, simplifications, extensions
and their convergence and complexity properties. Last but not least, we discuss
the potentials of our unified BLO framework for designing new algorithms and
point out some promising directions for future research.

    

### [[2101.11589] A Convolutional Neural Network based Cascade Reconstruction for the IceCube Neutrino Observatory](http://arxiv.org/abs/2101.11589)


  Continued improvements on existing reconstruction methods are vital to the
success of high-energy physics experiments, such as the IceCube Neutrino
Observatory. In IceCube, further challenges arise as the detector is situated
at the geographic South Pole where computational resources are limited.
However, to perform real-time analyses and to issue alerts to telescopes around
the world, powerful and fast reconstruction methods are desired. Deep neural
networks can be extremely powerful, and their usage is computationally
inexpensive once the networks are trained. These characteristics make a deep
learning-based approach an excellent candidate for the application in IceCube.
A reconstruction method based on convolutional architectures and hexagonally
shaped kernels is presented. The presented method is robust towards systematic
uncertainties in the simulation and has been tested on experimental data. In
comparison to standard reconstruction methods in IceCube, it can improve upon
the reconstruction accuracy, while reducing the time necessary to run the
reconstruction by two to three orders of magnitude.

    

### [[2102.01336] Probabilistic Trust Intervals for Out of Distribution Detection](http://arxiv.org/abs/2102.01336)


  Building neural network classifiers with an ability to distinguish between in
and out-of distribution inputs is an important step towards faithful deep
learning systems. Some of the successful approaches for this, resort to
architectural novelties, such as ensembles, with increased complexities in
terms of the number of parameters and training procedures. Whereas some other
approaches make use of surrogate samples, which are easy to create and work as
proxies for actual out-of-distribution (OOD) samples, to train the networks for
OOD detection. In this paper, we propose a very simple approach for enhancing
the ability of a pretrained network to detect OOD inputs without even altering
the original parameter values. We define a probabilistic trust interval for
each weight parameter of the network and optimize its size according to the
in-distribution (ID) inputs. It allows the network to sample additional weight
values along with the original values at the time of inference and use the
observed disagreement among the corresponding outputs for OOD detection. In
order to capture the disagreement effectively, we also propose a measure and
establish its suitability using empirical evidence. Our approach outperforms
the existing state-of-the-art methods on various OOD datasets by considerable
margins without using any real or surrogate OOD samples. We also analyze the
performance of our approach on adversarial and corrupted inputs such as
CIFAR-10-C and demonstrate its ability to clearly distinguish such inputs as
well. By using fundamental theorem of calculus on neural networks, we explain
why our technique doesn't need to observe OOD samples during training to
achieve results better than the previous works.

    

### [[2102.05426] BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](http://arxiv.org/abs/2102.05426)


  We study the challenging task of neural network quantization without
end-to-end retraining, called Post-training Quantization (PTQ). PTQ usually
requires a small subset of training data but produces less powerful quantized
models than Quantization-Aware Training (QAT). In this work, we propose a novel
PTQ framework, dubbed BRECQ, which pushes the limits of bitwidth in PTQ down to
INT2 for the first time. BRECQ leverages the basic building blocks in neural
networks and reconstructs them one-by-one. In a comprehensive theoretical study
of the second-order error, we show that BRECQ achieves a good balance between
cross-layer dependency and generalization error. To further employ the power of
quantization, the mixed precision technique is incorporated in our framework by
approximating the inter-layer and intra-layer sensitivity. Extensive
experiments on various handcrafted and searched neural architectures are
conducted for both image classification and object detection tasks. And for the
first time we prove that, without bells and whistles, PTQ can attain 4-bit
ResNet and MobileNetV2 comparable with QAT and enjoy 240 times faster
production of quantized models. Codes are available at
this https URL.

    

### [[2102.06307] What does LIME really see in images?](http://arxiv.org/abs/2102.06307)


  The performance of modern algorithms on certain computer vision tasks such as
object recognition is now close to that of humans. This success was achieved at
the price of complicated architectures depending on millions of parameters and
it has become quite challenging to understand how particular predictions are
made. Interpretability methods propose to give us this understanding. In this
paper, we study LIME, perhaps one of the most popular. On the theoretical side,
we show that when the number of generated examples is large, LIME explanations
are concentrated around a limit explanation for which we give an explicit
expression. We further this study for elementary shape detectors and linear
models. As a consequence of this analysis, we uncover a connection between LIME
and integrated gradients, another explanation method. More precisely, the LIME
explanations are similar to the sum of integrated gradients over the
superpixels used in the preprocessing step of LIME.

    

### [[2102.06777] INSTA-YOLO: Real-Time Instance Segmentation](http://arxiv.org/abs/2102.06777)


  Instance segmentation has gained recently huge attention in various computer
vision applications. It aims at providing different IDs to different objects of
the scene, even if they belong to the same class. Instance segmentation is
usually performed as a two-stage pipeline. First, an object is detected, then
semantic segmentation within the detected box area is performed which involves
costly up-sampling. In this paper, we propose Insta-YOLO, a novel one-stage
end-to-end deep learning model for real-time instance segmentation. Instead of
pixel-wise prediction, our model predicts instances as object contours
represented by 2D points in Cartesian space. We evaluate our model on three
datasets, namely, Carvana,Cityscapes and Airbus. We compare our results to the
state-of-the-art models for instance segmentation. The results show our model
achieves competitive accuracy in terms of mAP at twice the speed on GTX-1080
GPU.

    

### [[2102.06984] Learning low-rank latent mesoscale structures in networks](http://arxiv.org/abs/2102.06984)


  It is common to use networks to encode the architecture of interactions
between entities in complex systems in the physical, biological, social, and
information sciences. Moreover, to study the large-scale behavior of complex
systems, it is important to study mesoscale structures in networks as building
blocks that influence such behavior. In this paper, we present a new approach
for describing low-rank mesoscale structure in networks, and we illustrate our
approach using several synthetic network models and empirical friendship,
collaboration, and protein--protein interaction (PPI) networks. We find that
these networks possess a relatively small number of `latent motifs' that
together can successfully approximate most subnetworks at a fixed mesoscale. We
use an algorithm that we call "network dictionary learning" (NDL), which
combines a network sampling method and nonnegative matrix factorization, to
learn the latent motifs of a given network. The ability to encode a network
using a set of latent motifs has a wide range of applications to
network-analysis tasks, such as comparison, denoising, and edge inference.
Additionally, using our new network denoising and reconstruction (NDR)
algorithm, we demonstrate how to denoise a corrupted network by using only the
latent motifs that one learns directly from the corrupted networks.

    

### [[2102.09604] Privacy-Preserving Graph Convolutional Networks for Text Classification](http://arxiv.org/abs/2102.09604)


  Graph convolutional networks (GCNs) are a powerful architecture for
representation learning on documents that naturally occur as graphs, e.g.,
citation or social networks. However, sensitive personal information, such as
documents with people's profiles or relationships as edges, are prone to
privacy leaks, as the trained model might reveal the original input. Although
differential privacy (DP) offers a well-founded privacy-preserving framework,
GCNs pose theoretical and practical challenges due to their training specifics.
We address these challenges by adapting differentially-private gradient-based
training to GCNs and conduct experiments using two optimizers on five NLP
datasets in two languages. We propose a simple yet efficient method based on
random graph splits that not only improves the baseline privacy bounds by a
factor of 2.7 while retaining competitive F1 scores, but also provides strong
privacy guarantees of epsilon = 1.0. We show that, under certain modeling
choices, privacy-preserving GCNs perform up to 90% of their non-private
variants, while formally guaranteeing strong privacy measures.

    

### [[2102.11329] Action Redundancy in Reinforcement Learning](http://arxiv.org/abs/2102.11329)


  Maximum Entropy (MaxEnt) reinforcement learning is a powerful learning
paradigm which seeks to maximize return under entropy regularization. However,
action entropy does not necessarily coincide with state entropy, e.g., when
multiple actions produce the same transition. Instead, we propose to maximize
the transition entropy, i.e., the entropy of next states. We show that
transition entropy can be described by two terms; namely, model-dependent
transition entropy and action redundancy. Particularly, we explore the latter
in both deterministic and stochastic settings and develop tractable
approximation methods in a near model-free setup. We construct algorithms to
minimize action redundancy and demonstrate their effectiveness on a synthetic
environment with multiple redundant actions as well as contemporary benchmarks
in Atari and Mujoco. Our results suggest that action redundancy is a
fundamental problem in reinforcement learning.

    

### [[2102.12321] AGENT: A Benchmark for Core Psychological Reasoning](http://arxiv.org/abs/2102.12321)


  For machine agents to successfully interact with humans in real-world
settings, they will need to develop an understanding of human mental life.
Intuitive psychology, the ability to reason about hidden mental variables that
drive observable actions, comes naturally to people: even pre-verbal infants
can tell agents from objects, expecting agents to act efficiently to achieve
goals given constraints. Despite recent interest in machine agents that reason
about other agents, it is not clear if such agents learn or hold the core
psychology principles that drive human reasoning. Inspired by cognitive
development studies on intuitive psychology, we present a benchmark consisting
of a large dataset of procedurally generated 3D animations, AGENT (Action,
Goal, Efficiency, coNstraint, uTility), structured around four scenarios (goal
preferences, action efficiency, unobserved constraints, and cost-reward
trade-offs) that probe key concepts of core intuitive psychology. We validate
AGENT with human-ratings, propose an evaluation protocol emphasizing
generalization, and compare two strong baselines built on Bayesian inverse
planning and a Theory of Mind neural network. Our results suggest that to pass
the designed tests of core intuitive psychology at human levels, a model must
acquire or have built-in representations of how agents plan, combining utility
computations and core knowledge of objects and physics.

    

### [[2102.12586] FERMI: Fair Empirical Risk Minimization via Exponential Rényi Mutual Information](http://arxiv.org/abs/2102.12586)


  Despite the success of large-scale empirical risk minimization (ERM) at
achieving high accuracy across a variety of machine learning tasks, fair ERM is
hindered by the incompatibility of fairness constraints with stochastic
optimization. In this paper, we propose the fair empirical risk minimization
via exponential Rényi mutual information (FERMI) framework. FERMI is built on
a stochastic estimator for exponential Rényi mutual information (ERMI), an
information divergence measuring the degree of the dependence of predictions on
sensitive attributes. Theoretically, we show that ERMI upper bounds existing
popular fairness violation metrics, thus controlling ERMI provides guarantees
on other commonly used violations, such as $L_\infty$. We derive an unbiased
estimator for ERMI, which we use to derive the FERMI algorithm. We prove that
FERMI converges for demographic parity, equalized odds, and equal opportunity
notions of fairness in stochastic optimization. Empirically, we show that FERMI
is amenable to large-scale problems with multiple (non-binary) sensitive
attributes and non-binary targets. Extensive experiments show that FERMI
achieves the most favorable tradeoffs between fairness violation and test
accuracy across all tested setups compared with state-of-the-art baselines for
demographic parity, equalized odds, equal opportunity. These benefits are
especially significant for non-binary classification with large sensitive sets
and small batch sizes, showcasing the effectiveness of the FERMI objective and
the developed stochastic algorithm for solving it.

    

### [[2103.02142] Learning to Fly -- a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control](http://arxiv.org/abs/2103.02142)


  Robotic simulators are crucial for academic research and education as well as
the development of safety-critical applications. Reinforcement learning
environments -- simple simulations coupled with a problem specification in the
form of a reward function -- are also important to standardize the development
(and benchmarking) of learning algorithms. Yet, full-scale simulators typically
lack portability and parallelizability. Vice versa, many reinforcement learning
environments trade-off realism for high sample throughputs in toy-like
problems. While public data sets have greatly benefited deep learning and
computer vision, we still lack the software tools to simultaneously develop --
and fairly compare -- control theory and reinforcement learning approaches. In
this paper, we propose an open-source OpenAI Gym-like environment for multiple
quadcopters based on the Bullet physics engine. Its multi-agent and vision
based reinforcement learning interfaces, as well as the support of realistic
collisions and aerodynamic effects, make it, to the best of our knowledge, a
first of its kind. We demonstrate its use through several examples, either for
control (trajectory tracking with PID control, multi-robot flight with
downwash, etc.) or reinforcement learning (single and multi-agent stabilization
tasks), hoping to inspire future research that combines control theory and
machine learning.

    

### [[2103.05102] Self-supervised Multisensor Change Detection](http://arxiv.org/abs/2103.05102)


  Most change detection methods assume that pre-change and post-change images
are acquired by the same sensor. However, in many real-life scenarios, e.g.,
natural disaster, it is more practical to use the latest available images
before and after the occurrence of incidence, which may be acquired using
different sensors. In particular, we are interested in the combination of the
images acquired by optical and Synthetic Aperture Radar (SAR) sensors. SAR
images appear vastly different from the optical images even when capturing the
same scene. Adding to this, change detection methods are often constrained to
use only target image-pair, no labeled data, and no additional unlabeled data.
Such constraints limit the scope of traditional supervised machine learning and
unsupervised generative approaches for multi-sensor change detection. Recent
rapid development of self-supervised learning methods has shown that some of
them can even work with only few images. Motivated by this, in this work we
propose a method for multi-sensor change detection using only the unlabeled
target bi-temporal images that are used for training a network in
self-supervised fashion by using deep clustering and contrastive learning. The
proposed method is evaluated on four multi-modal bi-temporal scenes showing
change and the benefits of our self-supervised approach are demonstrated.

    

### [[2103.14224] Active multi-fidelity Bayesian online changepoint detection](http://arxiv.org/abs/2103.14224)


  Online algorithms for detecting changepoints, or abrupt shifts in the
behavior of a time series, are often deployed with limited resources, e.g., to
edge computing settings such as mobile phones or industrial sensors. In these
scenarios it may be beneficial to trade the cost of collecting an environmental
measurement against the quality or "fidelity" of this measurement and how the
measurement affects changepoint estimation. For instance, one might decide
between inertial measurements or GPS to determine changepoints for motion. A
Bayesian approach to changepoint detection is particularly appealing because we
can represent our posterior uncertainty about changepoints and make active,
cost-sensitive decisions about data fidelity to reduce this posterior
uncertainty. Moreover, the total cost could be dramatically lowered through
active fidelity switching, while remaining robust to changes in data
distribution. We propose a multi-fidelity approach that makes cost-sensitive
decisions about which data fidelity to collect based on maximizing information
gain with respect to changepoints. We evaluate this framework on synthetic,
video, and audio data and show that this information-based approach results in
accurate predictions while reducing total cost.

    

### [[2104.05225] Edgeless-GNN: Unsupervised Inductive Edgeless Network Embedding](http://arxiv.org/abs/2104.05225)


  We study the problem of embedding edgeless nodes such as users who newly
enter the underlying network, while using graph neural networks (GNNs) widely
studied for effective representation learning of graphs thanks to its highly
expressive capability via message passing. Our study is motivated by the fact
that existing GNNs cannot be adopted for our problem since message passing to
such edgeless nodes having no connections is impossible. To tackle this
challenge, we propose Edgeless-GNN, a new framework that enables GNNs to
generate node embeddings even for edgeless nodes through unsupervised inductive
learning. Specifically, we start by constructing a $k$-nearest neighbor graph
($k$NNG) based on the similarity of node attributes to replace the GNN's
computation graph defined by the neighborhood-based aggregation of each node.
As our main contributions, the known network structure is used to train model
parameters, while a new loss function is established using energy-based
learning in such a way that our model learns the network structure. For the
edgeless nodes, we inductively infer embeddings for the edgeless nodes by using
edges via $k$NNG construction as a computation graph. By evaluating the
performance of various downstream machine learning (ML) tasks, we empirically
demonstrate that Edgeless-GNN consistently outperforms state-of-the-art methods
of inductive network embedding. Moreover, our findings corroborate the
effectiveness of Edgeless-GNN in judiciously combining the replaced computation
graph with our newly designed loss. Our framework is GNN-model-agnostic; thus,
GNN models can be appropriately chosen according to ones' needs and ML tasks.

    

### [[2104.07365] D-Cliques: Compensating NonIIDness in Decentralized Federated Learning with Topology](http://arxiv.org/abs/2104.07365)


  The convergence speed of machine learning models trained with Federated
Learning is significantly affected by non-independent and identically
distributed (non-IID) data partitions, even more so in a fully decentralized
setting without a central server. In this paper, we show that the impact of
local class bias, an important type of data non-IIDness, can be significantly
reduced by carefully designing the underlying communication topology. We
present D-Cliques, a novel topology that reduces gradient bias by grouping
nodes in interconnected cliques such that the local joint distribution in a
clique is representative of the global class distribution. We also show how to
adapt the updates of decentralized SGD to obtain unbiased gradients and
implement an effective momentum with D-Cliques. Our empirical evaluation on
MNIST and CIFAR10 demonstrates that our approach provides similar convergence
speed as a fully-connected topology with a significant reduction in the number
of edges and messages. In a 1000-node topology, D-Cliques requires 98% less
edges and 96% less total messages, with further possible gains using a
small-world topology across cliques.

    

### [[2104.09231] Visual analytics of set data for knowledge discovery and member selection support](http://arxiv.org/abs/2104.09231)


  Visual analytics (VA) is a visually assisted exploratory analysis approach in
which knowledge discovery is executed interactively between the user and system
in a human-centered manner. The purpose of this study is to develop a method
for the VA of set data aimed at supporting knowledge discovery and member
selection. A typical target application is a visual support system for team
analysis and member selection, by which users can analyze past teams and
examine candidate lineups for new teams. Because there are several
difficulties, such as the combinatorial explosion problem, developing a VA
system of set data is challenging. In this study, we first define the
requirements that the target system should satisfy and clarify the accompanying
challenges. Then we propose a method for the VA of set data, which satisfies
the requirements. The key idea is to model the generation process of sets and
their outputs using a manifold network model. The proposed method visualizes
the relevant factors as a set of topographic maps on which various information
is visualized. Furthermore, using the topographic maps as a bidirectional
interface, users can indicate their targets of interest in the system on these
maps. We demonstrate the proposed method by applying it to basketball teams,
and compare with a benchmark system for outcome prediction and lineup
reconstruction tasks. Because the method can be adapted to individual
application cases by extending the network structure, it can be a general
method by which practical systems can be built.

    

### [[2104.13471] Discovering nonlinear resonances through physics-informed machine learning](http://arxiv.org/abs/2104.13471)


  For an ensemble of nonlinear systems that model, for instance, molecules or
photonic systems, we propose a method that finds efficiently the configuration
that has prescribed transfer properties. Specifically, we use physics-informed
machine-learning (PIML) techniques to find the parameters for the efficient
transfer of an electron (or photon) to a targeted state in a non-linear dimer.
We create a machine learning model containing two variables, $\chi_D$, and
$\chi_A$, representing the non-linear terms in the donor and acceptor target
system states. We then introduce a data-free physics-informed loss function as
$1.0 - P_j$, where $P_j$ is the probability, the electron being in the targeted
state, $j$. By minimizing the loss function, we maximize the occupation
probability to the targeted state. The method recovers known results in the
Targeted Energy Transfer (TET) model, and it is then applied to a more complex
system with an additional intermediate state. In this trimer configuration, the
PIML approach discovers desired resonant paths from the donor to acceptor
units. The proposed PIML method is general and may be used in the chemical
design of molecular complexes or engineering design of quantum or photonic
systems.

    

### [[2104.14504] An Axiomatic Theory of Provably-Fair Welfare-Centric Machine Learning](http://arxiv.org/abs/2104.14504)


  We address an inherent difficulty in welfare-theoretic fair machine learning
by proposing an equivalently axiomatically-justified alternative and studying
the resulting computational and statistical learning questions. Welfare metrics
quantify overall wellbeing across a population of one or more groups, and
welfare-based objectives and constraints have recently been proposed to
incentivize fair machine learning methods to produce satisfactory solutions
that consider the diverse needs of multiple groups. Unfortunately, many
machine-learning problems are more naturally cast as loss minimization tasks,
rather than utility maximization, which complicates direct application of
welfare-centric methods to fair machine learning. In this work, we define a
complementary measure, termed malfare, measuring overall societal harm (rather
than wellbeing), with axiomatic justification via the standard axioms of
cardinal welfare. We then cast fair machine learning as malfare minimization
over the risk values (expected losses) of each group. Surprisingly, the axioms
of cardinal welfare (malfare) dictate that this is not equivalent to simply
defining utility as negative loss. Building upon these concepts, we define
fair-PAC (FPAC) learning, where an FPAC learner is an algorithm that learns an
$\varepsilon$-$\delta$ malfare-optimal model with bounded sample complexity,
for any data distribution, and for any (axiomatically justified) malfare
concept. Finally, we show broad conditions under which, with appropriate
modifications, standard PAC-learners may be converted to FPAC learners. This
places FPAC learning on firm theoretical ground, as it yields statistical and
computational efficiency guarantees for many well-studied machine-learning
models, and is also practically relevant, as it democratizes fair ML by
providing concrete training algorithms and rigorous generalization guarantees
for these models

    

### [[2104.14659] End-to-End Jet Classification of Boosted Top Quarks with the CMS Open Data](http://arxiv.org/abs/2104.14659)


  We describe a novel application of the end-to-end deep learning technique to
the task of discriminating top quark-initiated jets from those originating from
the hadronization of a light quark or a gluon. The end-to-end deep learning
technique combines deep learning algorithms and low-level detector
representation of the high-energy collision event. In this study, we use
low-level detector information from the simulated CMS Open Data samples to
construct the top jet classifiers. To optimize classifier performance we
progressively add low-level information from the CMS tracking detector,
including pixel detector reconstructed hits and impact parameters, and
demonstrate the value of additional tracking information even when no new
spatial structures are added. Relying only on calorimeter energy deposits and
reconstructed pixel detector hits, the end-to-end classifier achieves an AUC
score of 0.975$\pm$0.002 for the task of classifying boosted top quark jets.
After adding derived track quantities, the classifier AUC score increases to
0.9824$\pm$0.0013, serving as the first performance benchmark for these CMS
Open Data samples. We additionally provide a timing performance comparison of
different processor unit architectures for training the network.

    

### [[2105.02961] UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps](http://arxiv.org/abs/2105.02961)


  Boundary Representations (B-Reps) are the industry standard in 3D Computer
Aided Design/Manufacturing (CAD/CAM) and industrial design due to their
fidelity in representing stylistic details. However, they have been ignored in
the 3D style research. Existing 3D style metrics typically operate on meshes or
pointclouds, and fail to account for end-user subjectivity by adopting fixed
definitions of style, either through crowd-sourcing for style labels or
hand-crafted features. We propose UVStyle-Net, a style similarity measure for
B-Reps that leverages the style signals in the second order statistics of the
activations in a pre-trained (unsupervised) 3D encoder, and learns their
relative importance to a subjective end-user through few-shot learning. Our
approach differs from all existing data-driven 3D style methods since it may be
used in completely unsupervised settings, which is desirable given the lack of
publicly available labelled B-Rep datasets. More importantly, the few-shot
learning accounts for the inherent subjectivity associated with style. We show
quantitatively that our proposed method with B-Reps is able to capture stronger
style signals than alternative methods on meshes and pointclouds despite its
significantly greater computational efficiency. We also show it is able to
generate meaningful style gradients with respect to the input shape, and that
few-shot learning with as few as two positive examples selected by an end-user
is sufficient to significantly improve the style measure. Finally, we
demonstrate its efficacy on a large unlabeled public dataset of CAD models.
Source code and data will be released in the future.

    

### [[2105.04090] MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with One Transformer VAE](http://arxiv.org/abs/2105.04090)


  Transformers and variational autoencoders (VAE) have been extensively
employed for symbolic (e.g., MIDI) domain music generation. While the former
boast an impressive capability in modeling long sequences, the latter allow
users to willingly exert control over different parts (e.g., bars) of the music
to be generated. In this paper, we are interested in bringing the two together
to construct a single model that exhibits both strengths. The task is split
into two steps. First, we equip Transformer decoders with the ability to accept
segment-level, time-varying conditions during sequence generation.
Subsequently, we combine the developed and tested in-attention decoder with a
Transformer encoder, and train the resulting MuseMorphose model with the VAE
objective to achieve style transfer of long musical pieces, in which users can
specify musical attributes including rhythmic intensity and polyphony (i.e.,
harmonic fullness) they desire, down to the bar level. Experiments show that
MuseMorphose outperforms recurrent neural network (RNN) based baselines on
numerous widely-used metrics for style transfer tasks.

    

### [[2105.07179] BubbleNet: Inferring micro-bubble dynamics with semi-physics-informed deep learning](http://arxiv.org/abs/2105.07179)


  Micro-bubbles and bubbly flows are widely observed and applied in chemical
engineering, medicine, involves deformation, rupture, and collision of bubbles,
phase mixture, etc. We study bubble dynamics by setting up two numerical
simulation cases: bubbly flow with a single bubble and multiple bubbles, both
confined in the microchannel, with parameters corresponding to their medical
backgrounds. Both the cases have their medical background applications.
Multiphase flow simulation requires high computation accuracy due to possible
component losses that may be caused by sparse meshing during the computation.
Hence, data-driven methods can be adopted as an useful tool. Based on
physics-informed neural networks (PINNs), we propose a novel deep learning
framework BubbleNet, which entails three main parts: deep neural networks (DNN)
with sub nets for predicting different physics fields; the
semi-physics-informed part, with only the fluid continuum condition and the
pressure Poisson equation $\mathcal{P}$ encoded within; the time discretized
normalizer (TDN), an algorithm to normalize field data per time step before
training. We apply the traditional DNN and our BubbleNet to train the coarsened
simulation data and predict the physics fields of both the two bubbly flow
cases. The BubbleNets are trained for both with and without $\mathcal{P}$, from
which we conclude that the 'physics-informed' part can serve as inner
supervision. Results indicate our framework can predict the physics fields more
accurately, estimating the prediction absolute errors. Our deep learning
predictions outperform traditional numerical methods computed with similar data
density meshing. The proposed network can potentially be applied to many other
engineering fields.

    

### [[2105.07283] Calibrating sufficiently](http://arxiv.org/abs/2105.07283)


  When probabilistic classifiers are trained and calibrated, the so-called
grouping loss component of the calibration loss can easily be overlooked.
Grouping loss refers to the gap between observable information and information
actually exploited in the calibration exercise. We investigate the relation
between grouping loss and the concept of sufficiency, identifying
comonotonicity as a useful criterion for sufficiency. We revisit the probing
reduction approach of Langford & Zadrozny (2005) and find that it produces an
estimator of probabilistic classifiers that reduces grouping loss. Finally, we
discuss Brier curves as tools to support training and 'sufficient' calibration
of probabilistic classifiers.

    

### [[2105.10162] Variational Quantum Classifiers Through the Lens of the Hessian](http://arxiv.org/abs/2105.10162)


  In quantum computing, the variational quantum algorithms (VQAs) are well
suited for finding optimal combinations of things in specific applications
ranging from chemistry all the way to finance. The training of VQAs with
gradient descent optimization algorithm has shown a good convergence. At an
early stage, the simulation of variational quantum circuits on noisy
intermediate-scale quantum (NISQ) devices suffers from noisy outputs. Just like
classical deep learning, it also suffers from vanishing gradient problems. It
is a realistic goal to study the topology of loss landscape, to visualize the
curvature information and trainability of these circuits in the existence of
vanishing gradients. In this paper, we calculated the Hessian and visualized
the loss landscape of variational quantum classifiers at different points in
parameter space. The curvature information of variational quantum classifiers
(VQC) is interpreted and the loss function's convergence is shown. It helps us
better understand the behavior of variational quantum circuits to tackle
optimization problems efficiently. We investigated the variational quantum
classifiers via Hessian on quantum computers, started with a simple 4-bit
parity problem to gain insight into the practical behavior of Hessian, then
thoroughly analyzed the behavior of Hessian's eigenvalues on training the
variational quantum classifier for the Diabetes dataset. Finally, we show that
how the adaptive Hessian learning rate can influence the convergence while
training the variational circuits.

    

### [[2105.10937] Deep Learning Traversability Estimator for Mobile Robots in Unstructured Environments](http://arxiv.org/abs/2105.10937)


  Terrain traversability analysis plays a major role in ensuring safe robotic
navigation in unstructured environments. However, real-time constraints
frequently limit the accuracy of online tests especially in scenarios where
realistic robot-terrain interactions are complex to model. In this context, we
propose a deep learning framework trained in an end-to-end fashion from
elevation maps and trajectories to estimate the occurrence of failure events.
The network is first trained and tested in simulation over synthetic maps
generated by the OpenSimplex algorithm. The prediction performance of the Deep
Learning framework is illustrated by being able to retain over 94% recall of
the original simulator at 30% of the computational time. Finally, the network
is transferred and tested on real elevation maps collected by the SEEKER
consortium during the Martian rover test trial in the Atacama desert in Chile.
We show that transferring and fine-tuning of an application-independent
pre-trained model retains better performance than training uniquely on scarcely
available real data.

    

### [[2105.11990] Optimal Sampling Density for Nonparametric Regression](http://arxiv.org/abs/2105.11990)


  We propose a novel active learning strategy for regression, which is
model-agnostic, robust against model mismatch, and interpretable. Assuming that
a small number of initial samples are available, we derive the optimal training
density that minimizes the generalization error of local polynomial smoothing
(LPS) with its kernel bandwidth tuned locally: We adopt the mean integrated
squared error (MISE) as a generalization criterion, and use the asymptotic
behavior of the MISE as well as the locally optimal bandwidths (LOB) - the
bandwidth function that minimizes MISE in the asymptotic limit. The asymptotic
expression of our objective then reveals the dependence of the MISE on the
training density, enabling analytic minimization. As a result,we obtain the
optimal training density in a closed-form. The almost model-free nature of our
approach thus helps to encode the essential properties of the target problem,
providing a robust and model-agnostic active learning strategy. Furthermore,
the obtained training density factorizes the influence of local function
complexity, noise level and test density in a transparent and interpretable
way. We validate our theory in numerical simulations, and show that the
proposed active learning method outperforms the existing state-of-the-art
model-agnostic approaches.

    

### [[2106.00757] Neural message passing for joint paratope-epitope prediction](http://arxiv.org/abs/2106.00757)


  Antibodies are proteins in the immune system which bind to antigens to detect
and neutralise them. The binding sites in an antibody-antigen interaction are
known as the paratope and epitope, respectively, and the prediction of these
regions is key to vaccine and synthetic antibody development. Contrary to prior
art, we argue that paratope and epitope predictors require asymmetric
treatment, and propose distinct neural message passing architectures that are
geared towards the specific aspects of paratope and epitope prediction,
respectively. We obtain significant improvements on both tasks, setting the new
state-of-the-art and recovering favourable qualitative predictions on antigens
of relevance to COVID-19.

    

### [[2106.02359] How Good Is NLP? A Sober Look at NLP Tasks through the Lens of Social Impact](http://arxiv.org/abs/2106.02359)


  Recent years have seen many breakthroughs in natural language processing
(NLP), transitioning it from a mostly theoretical field to one with many
real-world applications. Noting the rising number of applications of other
machine learning and AI techniques with pervasive societal impact, we
anticipate the rising importance of developing NLP technologies for social
good. Inspired by theories in moral philosophy and global priorities research,
we aim to promote a guideline for social good in the context of NLP. We lay the
foundations via the moral philosophy definition of social good, propose a
framework to evaluate the direct and indirect real-world impact of NLP tasks,
and adopt the methodology of global priorities research to identify priority
causes for NLP research. Finally, we use our theoretical framework to provide
some practical guidelines for future NLP research for social good. Our data and
code are available at this http URL. In
addition, we curate a list of papers and resources on NLP for social good at
this https URL.

    

### [[2106.03637] Deep Canonical Correlation Alignment for Sensor Signals](http://arxiv.org/abs/2106.03637)


  Sensor technologies are becoming increasingly prevalent in the biomedical
field, with applications ranging from telemonitoring of people at risk, to
using sensor derived information as objective endpoints in clinical trials. To
fully utilize sensor information, signals from distinct sensors often have to
be temporally aligned. However, due to imperfect oscillators and significant
noise, commonly encountered with biomedical signals, temporal alignment of raw
signals is an all but trivial problem, with, to-date, no generally applicable
solution. In this work, we present Deep Canonical Correlation Alignment (DCCA),
a novel, generally applicable solution for the temporal alignment of raw
(biomedical) sensor signals. DCCA allows practitioners to directly align raw
signals, from distinct sensors, without requiring deep domain knowledge. On a
selection of artificial and real datasets, we demonstrate the performance and
utility of DCCA under a variety of conditions. We compare the DCCA algorithm to
other warping based methods, DCCA outperforms dynamic time warping and cross
correlation based methods by an order of magnitude in terms of alignment error.
DCCA performs especially well on almost periodic biomedical signals such as
heart-beats and breathing patterns. In comparison to existing approaches, that
are not tailored towards raw sensor data, DCCA is not only fast enough to work
on signals with billions of data points but also provides automatic filtering
and transformation functionalities, allowing it to deal with very noisy and
even morphologically distinct signals.

    

### [[2106.04156] Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss](http://arxiv.org/abs/2106.04156)


  Recent works in self-supervised learning have advanced the state-of-the-art
by relying on the contrastive learning paradigm, which learns representations
by pushing positive pairs, or similar examples from the same class, closer
together while keeping negative pairs far apart. Despite the empirical
successes, theoretical foundations are limited -- prior analyses assume
conditional independence of the positive pairs given the same class label, but
recent empirical applications use heavily correlated positive pairs (i.e., data
augmentations of the same image). Our work analyzes contrastive learning
without assuming conditional independence of positive pairs using a novel
concept of the augmentation graph on data. Edges in this graph connect
augmentations of the same data, and ground-truth classes naturally form
connected sub-graphs. We propose a loss that performs spectral decomposition on
the population augmentation graph and can be succinctly written as a
contrastive learning objective on neural net representations. Minimizing this
objective leads to features with provable accuracy guarantees under linear
probe evaluation. By standard generalization bounds, these accuracy guarantees
also hold when minimizing the training contrastive loss. Empirically, the
features learned by our objective can match or outperform several strong
baselines on benchmark vision datasets. In all, this work provides the first
provable analysis for contrastive learning where guarantees for linear probe
evaluation can apply to realistic empirical settings.

    

### [[2106.04362] DIPS-Plus: The Enhanced Database of Interacting Protein Structures for Interface Prediction](http://arxiv.org/abs/2106.04362)


  How and where proteins interface with one another can ultimately impact the
proteins' functions along with a range of other biological processes. As such,
precise computational methods for protein interface prediction (PIP) come
highly sought after as they could yield significant advances in drug discovery
and design as well as protein function analysis. However, the traditional
benchmark dataset for this task, Docking Benchmark 5 (DB5), contains only a
modest 230 complexes for training, validating, and testing different machine
learning algorithms. In this work, we expand on a dataset recently introduced
for this task, the Database of Interacting Protein Structures (DIPS), to
present DIPS-Plus, an enhanced, feature-rich dataset of 42,112 complexes for
geometric deep learning of protein interfaces. The previous version of DIPS
contains only the Cartesian coordinates and types of the atoms comprising a
given protein complex, whereas DIPS-Plus now includes a plethora of new
residue-level features including protrusion indices, half-sphere amino acid
compositions, and new profile hidden Markov model (HMM)-based sequence features
for each amino acid, giving researchers a large, well-curated feature bank for
training protein interface prediction methods. We demonstrate through rigorous
benchmarks that training an existing state-of-the-art (SOTA) model for PIP on
DIPS-Plus yields SOTA results, surpassing the performance of all other models
trained on residue-level and atom-level encodings of protein complexes to date.

    

### [[2106.10821] Demonstration of Panda: A Weakly Supervised Entity Matching System](http://arxiv.org/abs/2106.10821)


  Entity matching (EM) refers to the problem of identifying tuple pairs in one
or more relations that refer to the same real world entities. Supervised
machine learning (ML) approaches, and deep learning based approaches in
particular, typically achieve state-of-the-art matching results. However, these
approaches require many labeled examples, in the form of matching and
non-matching pairs, which are expensive and time-consuming to label. In this
paper, we introduce Panda, a weakly supervised system specifically designed for
EM. Panda uses the same labeling function abstraction as Snorkel, where
labeling functions (LF) are user-provided programs that can generate large
amounts of (somewhat noisy) labels quickly and cheaply, which can then be
combined via a labeling model to generate accurate final predictions. To
support users developing LFs for EM, Panda provides an integrated development
environment (IDE) that lives in a modern browser architecture. Panda's IDE
facilitates the development, debugging, and life-cycle management of LFs in the
context of EM tasks, similar to how IDEs such as Visual Studio or Eclipse excel
in general-purpose programming. Panda's IDE includes many novel features
purpose-built for EM, such as smart data sampling, a builtin library of EM
utility functions, automatically generated LFs, visual debugging of LFs, and
finally, an EM-specific labeling model. We show in this demo that Panda IDE can
greatly accelerate the development of high-quality EM solutions using weak
supervision.

    

### [[2106.11342] Dive into Deep Learning](http://arxiv.org/abs/2106.11342)


  This open-source book represents our attempt to make deep learning
approachable, teaching readers the concepts, the context, and the code. The
entire book is drafted in Jupyter notebooks, seamlessly integrating exposition
figures, math, and interactive examples with self-contained code. Our goal is
to offer a resource that could (i) be freely available for everyone; (ii) offer
sufficient technical depth to provide a starting point on the path to actually
becoming an applied machine learning scientist; (iii) include runnable code,
showing readers how to solve problems in practice; (iv) allow for rapid
updates, both by us and also by the community at large; (v) be complemented by
a forum for interactive discussion of technical details and to answer
questions.

    

### [[2106.11642] Repulsive Deep Ensembles are Bayesian](http://arxiv.org/abs/2106.11642)


  Deep ensembles have recently gained popularity in the deep learning community
for their conceptual simplicity and efficiency. However, maintaining functional
diversity between ensemble members that are independently trained with gradient
descent is challenging. This can lead to pathologies when adding more ensemble
members, such as a saturation of the ensemble performance, which converges to
the performance of a single model. Moreover, this does not only affect the
quality of its predictions, but even more so the uncertainty estimates of the
ensemble, and thus its performance on out-of-distribution data. We hypothesize
that this limitation can be overcome by discouraging different ensemble members
from collapsing to the same function. To this end, we introduce a kernelized
repulsive term in the update rule of the deep ensembles. We show that this
simple modification not only enforces and maintains diversity among the members
but, even more importantly, transforms the maximum a posteriori inference into
proper Bayesian inference. Namely, we show that the training dynamics of our
proposed repulsive ensembles follow a Wasserstein gradient flow of the KL
divergence with the true posterior. We study repulsive terms in weight and
function space and empirically compare their performance to standard ensembles
and Bayesian baselines on synthetic and real-world prediction tasks.

    

### [[2107.11417] MARS: Middleware for Adaptive Reflective Computer Systems](http://arxiv.org/abs/2107.11417)


  Self-adaptive approaches for runtime resource management of manycore
computing platforms often require a runtime model of the system that represents
the software organization or the architecture of the target platform. The
increasing heterogeneity in a platform's resource types and the interactions
between resources pose challenges for coordinated model-based decision making
in the face of dynamic workloads. Self-awareness properties address these
challenges for emerging heterogeneous manycore processing (HMP) platforms
through reflective resource managers. However, with HMP computing platform
architectures evolving rapidly, porting the self-aware decision logic across
different hardware platforms is challenging, requiring resource managers to
update their models and platform-specific interfaces. We propose MARS
(Middleware for Adaptive and Reflective Systems), a cross-layer and
multi-platform framework that allows users to easily create resource managers
by composing system models and resource management policies in a flexible and
coordinated manner. MARS consists of a generic user-level sensing/actuation
interface that allows for portable policy design, and a reflective system model
used to coordinate multiple policies. We demonstrate MARS' interaction across
multiple layers of the system stack through a dynamic voltage and frequency
scaling (DVFS) policy example which can run on any Linux-based HMP computing
platform.

    

### [[2107.11516] Architecting Optically-Controlled Phase Change Memory](http://arxiv.org/abs/2107.11516)


  Phase Change Memory (PCM) is an attractive candidate for main memory as it
offers non-volatility and zero leakage power, while providing higher cell
densities, longer data retention time, and higher capacity scaling compared to
DRAM. In PCM, data is stored in the crystalline or amorphous state of the phase
change material. The typical electrically-controlled PCM (EPCM), however,
suffers from longer write latency and higher write energy compared to DRAM and
limited multi-level cell (MLC) capacities. These challenges limit the
performance of data-intensive applications running on computing systems with
EPCMs.
Recently, researchers demonstrated optically-controlled PCM (OPCM) cells,
with support for 5 bits/cell in contrast to 2 bits/cell in EPCM. These OPCM
cells can be accessed directly with optical signals that are multiplexed in
high-bandwidth-density silicon-photonic links. The higher MLC capacity in OPCM
and the direct cell access using optical signals enable an increased read/write
throughput and lower energy per access than EPCM. However, due to the direct
cell access using optical signals, OPCM systems cannot be designed using
conventional memory architecture. We need a complete redesign of the memory
architecture that is tailored to the properties of OPCM technology.
This paper presents the design of a unified network and main memory system
called COSMOS that combines OPCM and silicon-photonic links to achieve high
memory throughput. COSMOS is composed of a hierarchical multi-banked OPCM array
with novel read and write access protocols, and uses an
Electrical-Optical-Electrical (E-O-E) control unit to interface with the
processor. Our evaluation of a 2.5D-integrated system containing a processor
and COSMOS demonstrates 2.14x average speedup compared to an EPCM system.
COSMOS consumes 3.8x lower read energy-per-bit and 5.97x lower write
energy-per-bit compared to EPCM.

    

### [[2107.11673] ScaleHLS: Scalable High-Level Synthesis through MLIR](http://arxiv.org/abs/2107.11673)


  High-level Synthesis (HLS) has been widely adopted as it significantly
improves the hardware design productivity and enables efficient design space
exploration (DSE). HLS tools can be used to deliver solutions for many
different kinds of design problems, which are often better solved with
different levels of abstraction. While existing HLS tools are built using
compiler infrastructures largely based on a single-level abstraction (e.g.,
LLVM), we propose ScaleHLS, a next-generation HLS compilation flow, on top of a
multi-level compiler infrastructure called MLIR, for the first time. By using
an intermediate representation (IR) that can be better tuned to particular
algorithms at different representation levels, we are able to build this new
HLS tool that is more scalable and customizable towards various applications
coming with intrinsic structural or functional hierarchies. ScaleHLS is able to
represent and optimize HLS designs at multiple levels of abstraction and
provides an HLS-dedicated transform and analysis library to solve the
optimization problems at the suitable representation levels. On top of the
library, we also build an automated DSE engine to explore the multi-dimensional
design space efficiently. In addition, we develop an HLS C front-end and a
C/C++ emission back-end to translate HLS designs into/from MLIR for enabling
the end-to-end ScaleHLS flow. Experimental results show that, comparing to the
baseline designs only optimized by Xilinx Vivado HLS, ScaleHLS improves the
performances with amazing quality-of-results -- up to 768.1x better on
computation kernel level programs and up to 3825.0x better on neural network
models.

    

### [[2107.11723] A 51.3 TOPS/W, 134.4 GOPS In-memory Binary Image Filtering in 65nm CMOS](http://arxiv.org/abs/2107.11723)


  Neuromorphic vision sensors (NVS) can enable energy savings due to their
event-driven that exploits the temporal redundancy in video streams from a
stationary camera. However, noise-driven events lead to the false triggering of
the object recognition processor. Image denoise operations require
memoryintensive processing leading to a bottleneck in energy and latency. In
this paper, we present in-memory filtering (IMF), a 6TSRAM in-memory computing
based image denoising for eventbased binary image (EBBI) frame from an NVS. We
propose a non-overlap median filter (NOMF) for image denoising. An inmemory
computing framework enables hardware implementation of NOMF leveraging the
inherent read disturb phenomenon of 6T-SRAM. To demonstrate the energy-saving
and effectiveness of the algorithm, we fabricated the proposed architecture in
a 65nm CMOS process. As compared to fully digital implementation, IMF enables >
70x energy savings and a > 3x improvement of processing time when tested with
the video recordings from a DAVIS sensor and achieves a peak throughput of
134.4 GOPS. Furthermore, the peak energy efficiencies of the NOMF is 51.3
TOPS/W, comparable with state of the art inmemory processors. We also show that
the accuracy of the images obtained by NOMF provide comparable accuracy in
tracking and classification applications when compared with images obtained by
conventional median filtering.

    

### [[2107.11814] LightOn Optical Processing Unit: Scaling-up AI and HPC with a Non von Neumann co-processor](http://arxiv.org/abs/2107.11814)


  We introduce LightOn's Optical Processing Unit (OPU), the first photonic AI
accelerator chip available on the market for at-scale Non von Neumann
computations, reaching 1500 TeraOPS. It relies on a combination of free-space
optics with off-the-shelf components, together with a software API allowing a
seamless integration within Python-based processing pipelines. We discuss a
variety of use cases and hybrid network architectures, with the OPU used in
combination of CPU/GPU, and draw a pathway towards "optical advantage".

    

### [[2107.11881] Ultra-Fast, High-Performance 8x8 Approximate Multipliers by a New Multicolumn 3,3:2 Inexact Compressor and its Derivatives](http://arxiv.org/abs/2107.11881)


  Multiplier, which has a key role in many different applications, is a
time-consuming, energy-intensive computation block. Approximate computing is a
practical design paradigm that attempts to improve hardware efficacy while
keeping computation quality satisfactory. A novel multicolumn 3,3:2 inexact
compressor is presented in this paper. It takes three partial products from two
adjacent columns each for rapid partial product reduction. The proposed inexact
compressor and its derivates enable us to design a high-speed approximate
multiplier. Then, another ultra-fast, high-efficient approximate multiplier is
achieved by means a systematic truncation strategy. The proposed multipliers
accumulate partial products in only two stages, one fewer stage than other
approximate multipliers in the literature. Implementation results by Synopsys
Design Compiler and 45 nm technology node demonstrates nearly 11.11% higher
speed for the second proposed design over the fastest existing approximate
multiplier. Furthermore, the new approximate multipliers are applied to the
image processing application of image sharpening. Their performance in this
application is highly satisfactory. It is shown in this paper that the error
pattern of an approximate multiplier, in addition to the mean error distance
and error rate, has a direct effect on the outcomes of the image processing
application.

    

### [[2107.05453] Neat: Low-Complexity, Efficient On-Chip Cache Coherence](http://arxiv.org/abs/2107.05453)


  Cache coherence protocols such as MESI that use writer-initiated invalidation
have high complexity and sometimes have poor performance and energy usage,
especially under false sharing. Such protocols require numerous transient
states, a shared directory, and support for core-to-core communication, while
also suffering under false sharing. An alternative to MESI's writer-initiated
invalidation is self-invalidation, which achieves lower complexity than MESI
but adds high performance costs or relies on programmer annotations or specific
data access patterns.
This paper presents Neat, a low-complexity, efficient cache coherence
protocol. Neat uses self-invalidation, thus avoiding MESI's transient states,
directory, and core-to-core communication requirements. Neat uses novel
mechanisms that effectively avoid many unnecessary self-invalidations. An
evaluation shows that Neat is simple and has lower verification complexity than
the MESI protocol. Neat not only outperforms state-of-the-art self-invalidation
protocols, but its performance and energy consumption are comparable to MESI's,
and it outperforms MESI under false sharing.

    

### [[2107.11513] Distributed stochastic inertial methods with delayed derivatives](http://arxiv.org/abs/2107.11513)


  Stochastic gradient methods (SGMs) are predominant approaches for solving
stochastic optimization. On smooth nonconvex problems, a few acceleration
techniques have been applied to improve the convergence rate of SGMs. However,
little exploration has been made on applying a certain acceleration technique
to a stochastic subgradient method (SsGM) for nonsmooth nonconvex problems. In
addition, few efforts have been made to analyze an (accelerated) SsGM with
delayed derivatives. The information delay naturally happens in a distributed
system, where computing workers do not coordinate with each other.
In this paper, we propose an inertial proximal SsGM for solving nonsmooth
nonconvex stochastic optimization problems. The proposed method can have
guaranteed convergence even with delayed derivative information in a
distributed environment. Convergence rate results are established to three
classes of nonconvex problems: weakly-convex nonsmooth problems with a convex
regularizer, composite nonconvex problems with a nonsmooth convex regularizer,
and smooth nonconvex problems. For each problem class, the convergence rate is
$O(1/K^{\frac{1}{2}})$ in the expected value of the gradient norm square, for
$K$ iterations. In a distributed environment, the convergence rate of the
proposed method will be slowed down by the information delay. Nevertheless, the
slow-down effect will decay with the number of iterations for the latter two
problem classes. We test the proposed method on three applications. The
numerical results clearly demonstrate the advantages of using the
inertial-based acceleration. Furthermore, we observe higher parallelization
speed-up in asynchronous updates over the synchronous counterpart, though the
former uses delayed derivatives.

    

### [[2107.11536] SODA: A Semantics-Aware Optimization Framework for Data-Intensive Applications Using Hybrid Program Analysis](http://arxiv.org/abs/2107.11536)


  In the era of data explosion, a growing number of data-intensive computing
frameworks, such as Apache Hadoop and Spark, have been proposed to handle the
massive volume of unstructured data in parallel. Since programming models
provided by these frameworks allow users to specify complex and diversified
user-defined functions (UDFs) with predefined operations, the grand challenge
of tuning up entire system performance arises if programmers do not fully
understand the semantics of code, data, and runtime systems. In this paper, we
design a holistic semantics-aware optimization for data-intensive applications
using hybrid program analysis} (SODA) to assist programmers to tune performance
issues. SODA is a two-phase framework: the offline phase is a static analysis
that analyzes code and performance profiling data from the online phase of
prior executions to generate a parameterized and instrumented application; the
online phase is a dynamic analysis that keeps track of the application's
execution and collects runtime information of data and system. Extensive
experimental results on four real-world Spark applications show that SODA can
gain up to 60%, 10%, 8%, faster than its original implementation, with the
three proposed optimization strategies, i.e., cache management, operation
reordering, and element pruning, respectively.

    

### [[2107.11540] A Survey of Semantics-Aware Performance Optimization for Data-Intensive Computing](http://arxiv.org/abs/2107.11540)


  We are living in the era of Big Data and witnessing the explosion of data.
Given that the limitation of CPU and I/O in a single computer, the mainstream
approach to scalability is to distribute computations among a large number of
processing nodes in a cluster or cloud. This paradigm gives rise to the term of
data-intensive computing, which denotes a data parallel approach to process
massive volume of data. Through the efforts of different disciplines, several
promising programming models and a few platforms have been proposed for
data-intensive computing, such as MapReduce, Hadoop, Apache Spark and Dyrad.
Even though a large body of research work has being proposed to improve overall
performance of these platforms, there is still a gap between the actual
performance demand and the capability of current commodity systems. This paper
is aimed to provide a comprehensive understanding about current semantics-aware
approaches to improve the performance of data-intensive computing. We first
introduce common characteristics and paradigm shifts in the evolution of
data-intensive computing, as well as contemporary programming models and
technologies. We then propose four kinds of performance defects and survey the
state-of-the-art semantics-aware techniques. Finally, we discuss the research
challenges and opportunities in the field of semantics-aware performance
optimization for data-intensive computing.

    

### [[2107.11541] Performance assessment of CUDA and OpenACC in large scale combustion simulations](http://arxiv.org/abs/2107.11541)


  GPUs have climbed up to the top of supercomputer systems making life harder
to many legacy scientific codes. Nowadays, many recipes are being used in such
code's portability, without any clarity of which is the best option. We present
a comparative analysis of the two most common approaches, CUDA and OpenACC,
into the multi-physics CFD code Alya. Our focus is the combustion problems
which are one of the most computing demanding CFD simulations. The most
computing-intensive parts of the code were analyzed in detail. New data
structures for the matrix assembly step have been created to facilitate a SIMD
execution that benefits vectorization in the CPU and stream processing in the
GPU. As a result, the CPU code has improved its performance by up to 25%. In
GPU execution, CUDA has proven to be up to 2 times faster than OpenACC for the
assembly of the matrix. On the contrary, similar performance has been obtained
in the kernels related to vector operations used in the linear solver, where
there is minimal memory reuse.

    

### [[2107.11592] Blockchain Transaction Processing](http://arxiv.org/abs/2107.11592)


  A blockchain is a linked list of immutable tamper-proof blocks, which is
stored at each participating node. Each block records a set of transactions and
the associated metadata. Blockchain transactions act on the identical ledger
data stored at each node. Blockchain was first perceived by Satoshi Nakamoto,
as a peer-to-peer money exchange system. Nakamoto referred to the transactional
tokens exchanged among clients in his system as Bitcoins.

    

### [[2107.11832] A Holistic Analysis of Datacenter Operations: Resource Usage, Energy, and Workload Characterization -- Extended Technical Report](http://arxiv.org/abs/2107.11832)


  Improving datacenter operations is vital for the digital society. We posit
that doing so requires our community to shift, from operational aspects taken
in isolation to holistic analysis of datacenter resources, energy, and
workloads. In turn, this shift will require new analysis methods, and
open-access, FAIR datasets with fine temporal and spatial granularity. We
leverage in this work one of the (rare) public datasets providing fine-grained
information on datacenter operations. Using it, we show strong evidence that
fine-grained information reveals new operational aspects. We then propose a
method for holistic analysis of datacenter operations, providing statistical
characterization of node, energy, and workload aspects. We demonstrate the
benefits of our holistic analysis method by applying it to the operations of a
datacenter infrastructure with over 300 nodes. Our analysis reveals both
generic and ML-specific aspects, and further details how the operational
behavior of the datacenter changed during the 2020 COVID-19 pandemic. We make
over 30 main observations, providing holistic insight into the long-term
operation of a large-scale, public scientific infrastructure. We suggest such
observations can help immediately with performance engineering tasks such as
predicting future datacenter load, and also long-term with the design of
datacenter infrastructure.

    

### [[2107.11912] Performance vs Programming Effort between Rust and C on Multicore Architectures: Case Study in N-Body](http://arxiv.org/abs/2107.11912)


  Historically, Fortran and C have been the default programming languages in
High-Performance Computing (HPC). In both, programmers have primitives and
functions available that allow manipulating system memory and interacting
directly with the underlying hardware, resulting in efficient code in both
response times and resource use. On the other hand, it is a real challenge to
generate code that is maintainable and scalable over time in these types of
languages. In 2010, Rust emerged as a new programming language designed for
concurrent and secure applications, which adopts features of procedural,
object-oriented and functional languages. Among its design principles, Rust is
aimed at matching C in terms of efficiency, but with increased code security
and productivity. This paper presents a comparative study between C and Rust in
terms of performance and programming effort, selecting as a case study the
simulation of N computational bodies (N-Body), a popular problem in the HPC
community. Based on the experimental work, it was possible to establish that
Rust is a language that reduces programming effort while maintaining acceptable
performance levels, meaning that it is a possible alternative to C for HPC.

    

### [[2107.12016] Cost-effective Land Cover Classification for Remote Sensing Images](http://arxiv.org/abs/2107.12016)


  Land cover maps are of vital importance to various fields such as land use
policy development, ecosystem services, urban planning and agriculture
monitoring, which are mainly generated from remote sensing image classification
techniques. Traditional land cover classification usually needs tremendous
computational resources, which often becomes a huge burden to the remote
sensing community. Undoubtedly cloud computing is one of the best choices for
land cover classification, however, if not managed properly, the computation
cost on the cloud could be surprisingly high. Recently, cutting the unnecessary
computation long tail has become a promising solution for saving the cost in
the cloud. For land cover classification, it is generally not necessary to
achieve the best accuracy and 85% can be regarded as a reliable land cover
classification. Therefore, in this paper, we propose a framework for
cost-effective remote sensing classification. Given the desired accuracy, the
clustering algorithm can stop early for cost-saving whilst achieving sufficient
accuracy for land cover image classification. Experimental results show that
achieving 85%-99.9% accuracy needs only 27.34%-60.83% of the total cloud
computation cost for achieving a 100% accuracy. To put it into perspective, for
the US land cover classification example, the proposed approach can save over
$721,580.46 for the government in each single-use when the desired accuracy is
90%.

    

### [[2107.12053] A Frequency-based Parent Selection for Reducing the Effect of Evaluation Time Bias in Asynchronous Parallel Multi-objective Evolutionary Algorithms](http://arxiv.org/abs/2107.12053)


  This paper proposes a new parent selection method for reducing the effect of
evaluation time bias in asynchronous parallel evolutionary algorithms (APEAs).
APEAs have the advantage of increasing computational efficiency even when the
evaluation times of solutions differ. However, APEAs have a problem that their
search direction is biased toward the search region with a short evaluation
time. The proposed parent selection method considers the search frequency of
solutions to reduce such an adverse influence of APEAs while maintaining their
computational efficiency. We conduct experiments on toy problems that reproduce
the evaluation time bias on multi-objective optimization problems to
investigate the effectiveness of the proposed method. The experiments use
NSGA-III, a well-known multi-objective evolutionary algorithm. In the
experiments, we compare the proposed method with the synchronous and
asynchronous methods. The experimental results reveal that the proposed method
can reduce the effect of the evaluation time bias while reducing the computing
time of the parallel NSGA-III.

    

### [[2107.12147] Federated Action Recognition on Heterogeneous Embedded Devices](http://arxiv.org/abs/2107.12147)


  Federated learning allows a large number of devices to jointly learn a model
without sharing data. In this work, we enable clients with limited computing
power to perform action recognition, a computationally heavy task. We first
perform model compression at the central server through knowledge distillation
on a large dataset. This allows the model to learn complex features and serves
as an initialization for model fine-tuning. The fine-tuning is required because
the limited data present in smaller datasets is not adequate for action
recognition models to learn complex spatio-temporal features. Because the
clients present are often heterogeneous in their computing resources, we use an
asynchronous federated optimization and we further show a convergence bound. We
compare our approach to two baseline approaches: fine-tuning at the central
server (no clients) and fine-tuning using (heterogeneous) clients using
synchronous federated averaging. We empirically show on a testbed of
heterogeneous embedded devices that we can perform action recognition with
comparable accuracy to the two baselines above, while our asynchronous learning
strategy reduces the training time by 40%, relative to synchronous learning.

    

### [[2107.12148] Increasing FPS for single board computers and embedded computers in 2021 (Jetson nano and YOVOv4-tiny). Practice and review](http://arxiv.org/abs/2107.12148)


  This manuscript provides a review of methods for increasing the frame per
second of single-board computers. The main emphasis is on the Jetson family of
single-board computers from Nvidia Company, due to the possibility of using a
graphical interface for calculations. But taking into account the popular
low-cost segment of single-board computers as RaspberryPI family, BananaPI,
OrangePI, etc., we also provided an overview of methods for increasing the
frame per second without using a Graphics Processing Unit. We considered
frameworks, software development kit, and various libraries that can be used in
the process of increasing the frame per second in single-board computers.
Finally, we tested the presented methods for the YOLOv4-tiny model with a
custom dataset on the Jetson nano and presented the results in the table.

    

### [[1911.01195] Controlling a random population](http://arxiv.org/abs/1911.01195)


  Bertrand et al. introduced a model of parameterised systems, where each agent
is represented by a finite state system, and studied the following control
problem: for any number of agents, does there exist a controller able to bring
all agents to a target state? They showed that the problem is decidable and
EXPTIME-complete in the adversarial setting, and posed as an open problem the
stochastic setting, where the agent is represented by a Markov decision
process. In this paper, we show that the stochastic control problem is
decidable. Our solution makes significant uses of well quasi orders, of the
max-flow min-cut theorem, and of the theory of regular cost functions. We
introduce an intermediate problem of independent interest called the sequential
flow problem, and study the complexity of solving it.

    

### [[2011.02600] Upwind summation by parts finite difference methods for large scale elastic wave simulations in 3D complex geometries](http://arxiv.org/abs/2011.02600)


  High-order accurate summation-by-parts (SBP) finite difference (FD) methods
constitute efficient numerical methods for simulating large-scale hyperbolic
wave propagation problems. Traditional SBP FD operators that approximate
first-order spatial derivatives with central-difference stencils often have
spurious unresolved numerical wave-modes in their computed solutions. Recently
derived high order accurate upwind SBP operators based upwind FD stencils have
the potential to suppress these poisonous spurious wave-modes on marginally
resolved computational grids. In this paper, we demonstrate that not all high
order upwind SBP FD operators are applicable. Numerical dispersion relation
analysis shows that odd-order upwind SBP FD operators also support spurious
unresolved high-frequencies on marginally resolved meshes. Meanwhile,
even-order upwind SBP FD operators (of order 2, 4, 6) do not support spurious
unresolved high frequency wave modes and also have better numerical dispersion
properties. We discretise the three space dimensional (3D) elastic wave
equation on boundary-conforming curvilinear meshes. Using the energy method we
prove that the semi-discrete approximation is stable and energy-conserving. We
derive a priori error estimate and prove the convergence of the numerical
error. Numerical experiments for the 3D elastic wave equation in complex
geometries corroborate the theoretical analysis. Numerical simulations of the
3D elastic wave equation in heterogeneous media with complex non-planar free
surface topography are given, including numerical simulations of community
developed seismological benchmark problems. Computational results show that
even-order upwind SBP FD operators are more efficient, robust and less prone to
numerical dispersion errors on marginally resolved meshes when compared to the
odd-order upwind and traditional SBP FD operators.

    

### [[2102.09277] Locally Checkable Problems in Rooted Trees](http://arxiv.org/abs/2102.09277)


  Consider any locally checkable labeling problem $\Pi$ in rooted regular
trees: there is a finite set of labels $\Sigma$, and for each label $x \in
\Sigma$ we specify what are permitted label combinations of the children for an
internal node of label $x$ (the leaf nodes are unconstrained). This formalism
is expressive enough to capture many classic problems studied in distributed
computing, including vertex coloring, edge coloring, and maximal independent
set.
We show that the distributed computational complexity of any such problem
$\Pi$ falls in one of the following classes: it is $O(1)$, $\Theta(\log^* n)$,
$\Theta(\log n)$, or $\Theta(n)$ rounds in trees with $n$ nodes (and all of
these classes are nonempty). We show that the complexity of any given problem
is the same in all four standard models of distributed graph algorithms:
deterministic LOCAL, randomized LOCAL, deterministic CONGEST, and randomized
CONGEST model. In particular, we show that randomness does not help in this
setting, and complexity classes such as $\Theta(\log \log n)$ or
$\Theta(\sqrt{n})$ do not exist (while they do exist in the broader setting of
general trees).
We also show how to systematically determine the distributed computational
complexity of any such problem $\Pi$. We present an algorithm that, given the
description of $\Pi$, outputs the round complexity of $\Pi$ in these models.
While the algorithm may take exponential time in the size of the description of
$\Pi$, it is nevertheless practical: we provide a freely available
implementation of the classifier algorithm, and it is fast enough to classify
many typical problems of interest.

    

### [[2107.11444] Cooperative Exploration for Multi-Agent Deep Reinforcement Learning](http://arxiv.org/abs/2107.11444)


  Exploration is critical for good results in deep reinforcement learning and
has attracted much attention. However, existing multi-agent deep reinforcement
learning algorithms still use mostly noise-based techniques. Very recently,
exploration methods that consider cooperation among multiple agents have been
developed. However, existing methods suffer from a common challenge: agents
struggle to identify states that are worth exploring, and hardly coordinate
exploration efforts toward those states. To address this shortcoming, in this
paper, we propose cooperative multi-agent exploration (CMAE): agents share a
common goal while exploring. The goal is selected from multiple projected state
spaces via a normalized entropy-based technique. Then, agents are trained to
reach this goal in a coordinated manner. We demonstrate that CMAE consistently
outperforms baselines on various tasks, including a sparse-reward version of
the multiple-particle environment (MPE) and the Starcraft multi-agent challenge
(SMAC).

    

### [[2107.11447] Deep Learning Based Cardiac MRI Segmentation: Do We Need Experts?](http://arxiv.org/abs/2107.11447)


  Deep learning methods are the de-facto solutions to a multitude of medical
image analysis tasks. Cardiac MRI segmentation is one such application which,
like many others, requires a large number of annotated data so a trained
network can generalize well. Unfortunately, the process of having a large
number of manually curated images by medical experts is both slow and utterly
expensive. In this paper, we set out to explore whether expert knowledge is a
strict requirement for the creation of annotated datasets that machine learning
can successfully train on. To do so, we gauged the performance of three
segmentation models, namely U-Net, Attention U-Net, and ENet, trained with
different loss functions on expert and non-expert groundtruth for cardiac
cine-MRI segmentation. Evaluation was done with classic segmentation metrics
(Dice index and Hausdorff distance) as well as clinical measurements, such as
the ventricular ejection fractions and the myocardial mass. Results reveal that
generalization performances of a segmentation neural network trained on
non-expert groundtruth data is, to all practical purposes, as good as on expert
groundtruth data, in particular when the non-expert gets a decent level of
training, highlighting an opportunity for the efficient and cheap creation of
annotations for cardiac datasets.

    

### [[2107.11477] Plinko: A Theory-Free Behavioral Measure of Priors for Statistical Learning and Mental Model Updating](http://arxiv.org/abs/2107.11477)


  Probability distributions are central to Bayesian accounts of cognition, but
behavioral assessments do not directly measure them. Posterior distributions
are typically computed from collections of individual participant actions, yet
are used to draw conclusions about the internal structure of participant
beliefs. Also not explicitly measured are the prior distributions that
distinguish Bayesian models from others by representing initial states of
belief. Instead, priors are usually derived from experimenters' intuitions or
model assumptions and applied equally to all participants. Here we present
three experiments using "Plinko", a behavioral task in which participants
estimate distributions of ball drops over all available outcomes and where
distributions are explicitly measured before any observations. In Experiment 1,
we show that participant priors cluster around prototypical probability
distributions (Gaussian, bimodal, etc.), and that prior cluster membership may
indicate learning ability. In Experiment 2, we highlight participants' ability
to update to unannounced changes of presented distributions and how this
ability is affected by environmental manipulation. Finally, in Experiment 3, we
verify that individual participant priors are reliable representations and that
learning is not impeded when faced with a physically implausible ball drop
distribution that is dynamically defined according to individual participant
input. This task will prove useful in more closely examining mechanisms of
statistical learning and mental model updating without requiring many of the
assumptions made by more traditional computational modeling methodologies.

    

### [[2107.11481] Similarity Based Label Smoothing For Dialogue Generation](http://arxiv.org/abs/2107.11481)


  Generative neural conversational systems are generally trained with the
objective of minimizing the entropy loss between the training "hard" targets
and the predicted logits. Often, performance gains and improved generalization
can be achieved by using regularization techniques like label smoothing, which
converts the training "hard" targets to "soft" targets. However, label
smoothing enforces a data independent uniform distribution on the incorrect
training targets, which leads to an incorrect assumption of equi-probable
incorrect targets for each correct target. In this paper we propose and
experiment with incorporating data dependent word similarity based weighing
methods to transforms the uniform distribution of the incorrect target
probabilities in label smoothing, to a more natural distribution based on
semantics. We introduce hyperparameters to control the incorrect target
distribution, and report significant performance gains over networks trained
using standard label smoothing based loss, on two standard open domain dialogue
corpora.

    

### [[2107.11509] Cycled Compositional Learning between Images and Text](http://arxiv.org/abs/2107.11509)


  We present an approach named the Cycled Composition Network that can measure
the semantic distance of the composition of image-text embedding. First, the
Composition Network transit a reference image to target image in an embedding
space using relative caption. Second, the Correction Network calculates a
difference between reference and retrieved target images in the embedding space
and match it with a relative caption. Our goal is to learn a Composition
mapping with the Composition Network. Since this one-way mapping is highly
under-constrained, we couple it with an inverse relation learning with the
Correction Network and introduce a cycled relation for given Image We
participate in Fashion IQ 2020 challenge and have won the first place with the
ensemble of our model.

    

### [[2107.11517] Crosslink-Net: Double-branch Encoder Segmentation Network via Fusing Vertical and Horizontal Convolutions](http://arxiv.org/abs/2107.11517)


  Accurate image segmentation plays a crucial role in medical image analysis,
yet it faces great challenges of various shapes, diverse sizes, and blurry
boundaries. To address these difficulties, square kernel-based encoder-decoder
architecture has been proposed and widely used, but its performance remains
still unsatisfactory. To further cope with these challenges, we present a novel
double-branch encoder architecture. Our architecture is inspired by two
observations: 1) Since the discrimination of features learned via square
convolutional kernels needs to be further improved, we propose to utilize
non-square vertical and horizontal convolutional kernels in the double-branch
encoder, so features learned by the two branches can be expected to complement
each other. 2) Considering that spatial attention can help models to better
focus on the target region in a large-sized image, we develop an attention loss
to further emphasize the segmentation on small-sized targets. Together, the
above two schemes give rise to a novel double-branch encoder segmentation
framework for medical image segmentation, namely Crosslink-Net. The experiments
validate the effectiveness of our model on four datasets. The code is released
at this https URL.

    

### [[2107.11521] Caveats for the use of Web of Science Core Collection in old literature retrieval and historical bibliometric analysis](http://arxiv.org/abs/2107.11521)


  By using publications from Web of Science Core Collection (WoSCC), Fosso
Wamba and his colleagues published an interesting and comprehensive paper in
Technological Forecasting and Social Change to explore the structure and
dynamics of artificial intelligence (AI) scholarship. Data demonstrated in
Fosso Wamba's study implied that the year 1991 seemed to be a "watershed" of AI
research. This research note tried to uncover the 1991 phenomenon from the
perspective of database limitation by probing the limitations of search in
abstract/author keywords/keywords plus fields of WoSCC empirically. The low
availability rates of abstract/author keywords/keywords plus information in
WoSCC found in this study can explain the "watershed" phenomenon of AI
scholarship in 1991 to a large extent. Some other caveats for the use of WoSCC
in old literature retrieval and historical bibliometric analysis were also
mentioned in the discussion section. This research note complements Fosso Wamba
and his colleagues' study and also helps avoid improper interpretation in the
use of WoSCC in old literature retrieval and historical bibliometric analysis.

    

### [[2107.11522] Semantic-guided Pixel Sampling for Cloth-Changing Person Re-identification](http://arxiv.org/abs/2107.11522)


  Cloth-changing person re-identification (re-ID) is a new rising research
topic that aims at retrieving pedestrians whose clothes are changed. This task
is quite challenging and has not been fully studied to date. Current works
mainly focus on body shape or contour sketch, but they are not robust enough
due to view and posture variations. The key to this task is to exploit
cloth-irrelevant cues. This paper proposes a semantic-guided pixel sampling
approach for the cloth-changing person re-ID task. We do not explicitly define
which feature to extract but force the model to automatically learn
cloth-irrelevant cues. Specifically, we first recognize the pedestrian's upper
clothes and pants, then randomly change them by sampling pixels from other
pedestrians. The changed samples retain the identity labels but exchange the
pixels of clothes or pants among different pedestrians. Besides, we adopt a
loss function to constrain the learned features to keep consistent before and
after changes. In this way, the model is forced to learn cues that are
irrelevant to upper clothes and pants. We conduct extensive experiments on the
latest released PRCC dataset. Our method achieved 65.8% on Rank1 accuracy,
which outperforms previous methods with a large margin. The code is available
at this https URL.

    

### [[2107.11572] The USYD-JD Speech Translation System for IWSLT 2021](http://arxiv.org/abs/2107.11572)


  This paper describes the University of Sydney& JD's joint submission of the
IWSLT 2021 low resource speech translation task. We participated in the
Swahili-English direction and got the best scareBLEU (25.3) score among all the
participants. Our constrained system is based on a pipeline framework, i.e. ASR
and NMT. We trained our models with the officially provided ASR and MT
datasets. The ASR system is based on the open-sourced tool Kaldi and this work
mainly explores how to make the most of the NMT models. To reduce the
punctuation errors generated by the ASR model, we employ our previous work
SlotRefine to train a punctuation correction model. To achieve better
translation performance, we explored the most recent effective strategies,
including back translation, knowledge distillation, multi-feature reranking and
transductive finetuning. For model structure, we tried auto-regressive and
non-autoregressive models, respectively. In addition, we proposed two novel
pre-train approaches, i.e. \textit{de-noising training} and
\textit{bidirectional training} to fully exploit the data. Extensive
experiments show that adding the above techniques consistently improves the
BLEU scores, and the final submission system outperforms the baseline
(Transformer ensemble model trained with the original parallel data) by
approximately 10.8 BLEU score, achieving the SOTA performance.

    

### [[2107.11614] Automatic tempered posterior distributions for Bayesian inversion problems](http://arxiv.org/abs/2107.11614)


  We propose a novel adaptive importance sampling scheme for Bayesian inversion
problems where the inference of the variables of interest and the power of the
data noise is split. More specifically, we consider a Bayesian analysis for the
variables of interest (i.e., the parameters of the model to invert), whereas we
employ a maximum likelihood approach for the estimation of the noise power. The
whole technique is implemented by means of an iterative procedure, alternating
sampling and optimization steps. Moreover, the noise power is also used as a
tempered parameter for the posterior distribution of the the variables of
interest. Therefore, a sequence of tempered posterior densities is generated,
where the tempered parameter is automatically selected according to the actual
estimation of the noise power. A complete Bayesian study over the model
parameters and the scale parameter can be also performed. Numerical experiments
show the benefits of the proposed approach.

    

### [[2107.11635] Clustering by Maximizing Mutual Information Across Views](http://arxiv.org/abs/2107.11635)


  We propose a novel framework for image clustering that incorporates joint
representation learning and clustering. Our method consists of two heads that
share the same backbone network - a "representation learning" head and a
"clustering" head. The "representation learning" head captures fine-grained
patterns of objects at the instance level which serve as clues for the
"clustering" head to extract coarse-grain information that separates objects
into clusters. The whole model is trained in an end-to-end manner by minimizing
the weighted sum of two sample-oriented contrastive losses applied to the
outputs of the two heads. To ensure that the contrastive loss corresponding to
the "clustering" head is optimal, we introduce a novel critic function called
"log-of-dot-product". Extensive experimental results demonstrate that our
method significantly outperforms state-of-the-art single-stage clustering
methods across a variety of image datasets, improving over the best baseline by
about 5-7% in accuracy on CIFAR10/20, STL10, and ImageNet-Dogs. Further, the
"two-stage" variant of our method also achieves better results than baselines
on three challenging ImageNet subsets.

    

### [[2107.11652] Stress Test Evaluation of Biomedical Word Embeddings](http://arxiv.org/abs/2107.11652)


  The success of pretrained word embeddings has motivated their use in the
biomedical domain, with contextualized embeddings yielding remarkable results
in several biomedical NLP tasks. However, there is a lack of research on
quantifying their behavior under severe "stress" scenarios. In this work, we
systematically evaluate three language models with adversarial examples --
automatically constructed tests that allow us to examine how robust the models
are. We propose two types of stress scenarios focused on the biomedical named
entity recognition (NER) task, one inspired by spelling errors and another
based on the use of synonyms for medical terms. Our experiments with three
benchmarks show that the performance of the original models decreases
considerably, in addition to revealing their weaknesses and strengths. Finally,
we show that adversarial training causes the models to improve their robustness
and even to exceed the original performance in some cases.

    

### [[2107.11695] Efficient QUBO transformation for Higher Degree Pseudo Boolean Functions](http://arxiv.org/abs/2107.11695)


  Quadratic Unconstrained Binary Optimization (QUBO) is recognized as a
unifying framework for modeling a wide range of problems. Problems can be
solved with commercial solvers customized for solving QUBO and since QUBO have
degree two, it is useful to have a method for transforming higher degree
pseudo-Boolean problems to QUBO format. The standard transformation approach
requires additional auxiliary variables supported by penalty terms for each
higher degree term. This paper improves on the existing cubic-to-quadratic
transformation approach by minimizing the number of additional variables as
well as penalty coefficient. Extensive experimental testing on Max 3-SAT
modeled as QUBO shows a near 100% reduction in the subproblem size used for
minimization of the number of auxiliary variables.

    

### [[2107.11768] A Joint and Domain-Adaptive Approach to Spoken Language Understanding](http://arxiv.org/abs/2107.11768)


  Spoken Language Understanding (SLU) is composed of two subtasks: intent
detection (ID) and slot filling (SF). There are two lines of research on SLU.
One jointly tackles these two subtasks to improve their prediction accuracy,
and the other focuses on the domain-adaptation ability of one of the subtasks.
In this paper, we attempt to bridge these two lines of research and propose a
joint and domain adaptive approach to SLU. We formulate SLU as a constrained
generation task and utilize a dynamic vocabulary based on domain-specific
ontology. We conduct experiments on the ASMixed and MTOD datasets and achieve
competitive performance with previous state-of-the-art joint models. Besides,
results show that our joint model can be effectively adapted to a new domain.

    

### [[2107.11778] Learn to Focus: Hierarchical Dynamic Copy Network for Dialogue State Tracking](http://arxiv.org/abs/2107.11778)


  Recently, researchers have explored using the encoder-decoder framework to
tackle dialogue state tracking (DST), which is a key component of task-oriented
dialogue systems. However, they regard a multi-turn dialogue as a flat
sequence, failing to focus on useful information when the sequence is long. In
this paper, we propose a Hierarchical Dynamic Copy Network (HDCN) to facilitate
focusing on the most informative turn, making it easier to extract slot values
from the dialogue context. Based on the encoder-decoder framework, we adopt a
hierarchical copy approach that calculates two levels of attention at the word-
and turn-level, which are then renormalized to obtain the final copy
distribution. A focus loss term is employed to encourage the model to assign
the highest turn-level attention weight to the most informative turn.
Experimental results show that our model achieves 46.76% joint accuracy on the
MultiWOZ 2.1 dataset.

    

### [[2107.11785] Sensitivity and robustness analysis in Bayesian networks with the bnmonitor R package](http://arxiv.org/abs/2107.11785)


  Bayesian networks are a class of models that are widely used for risk
assessment of complex operational systems. There are now multiple approaches,
as well as implemented software, that guide their construction via data
learning or expert elicitation. However, a constructed Bayesian network needs
to be validated before it can be used for practical risk assessment. Here, we
illustrate the usage of the bnmonitor R package: the first comprehensive
software for the validation of a Bayesian network. An applied data analysis
using bnmonitor is carried out over a medical dataset to illustrate the use of
its wide array of functions.

    

### [[2107.11818] Bangla sign language recognition using concatenated BdSL network](http://arxiv.org/abs/2107.11818)


  Sign language is the only medium of communication for the hearing impaired
and the deaf and dumb community. Communication with the general mass is thus
always a challenge for this minority group. Especially in Bangla sign language
(BdSL), there are 38 alphabets with some having nearly identical symbols. As a
result, in BdSL recognition, the posture of hand is an important factor in
addition to visual features extracted from traditional Convolutional Neural
Network (CNN). In this paper, a novel architecture "Concatenated BdSL Network"
is proposed which consists of a CNN based image network and a pose estimation
network. While the image network gets the visual features, the relative
positions of hand keypoints are taken by the pose estimation network to obtain
the additional features to deal with the complexity of the BdSL symbols. A
score of 91.51% was achieved by this novel approach in test set and the
effectiveness of the additional pose estimation network is suggested by the
experimental results.

    

### [[2107.11820] A Survey of Monte Carlo Methods for Parameter Estimation](http://arxiv.org/abs/2107.11820)


  Statistical signal processing applications usually require the estimation of
some parameters of interest given a set of observed data. These estimates are
typically obtained either by solving a multi-variate optimization problem, as
in the maximum likelihood (ML) or maximum a posteriori (MAP) estimators, or by
performing a multi-dimensional integration, as in the minimum mean squared
error (MMSE) estimators. Unfortunately, analytical expressions for these
estimators cannot be found in most real-world applications, and the Monte Carlo
(MC) methodology is one feasible approach. MC methods proceed by drawing random
samples, either from the desired distribution or from a simpler one, and using
them to compute consistent estimators. The most important families of MC
algorithms are Markov chain MC (MCMC) and importance sampling (IS). On the one
hand, MCMC methods draw samples from a proposal density, building then an
ergodic Markov chain whose stationary distribution is the desired distribution
by accepting or rejecting those candidate samples as the new state of the
chain. On the other hand, IS techniques draw samples from a simple proposal
density, and then assign them suitable weights that measure their quality in
some appropriate way. In this paper, we perform a thorough review of MC methods
for the estimation of static parameters in signal processing applications. A
historical note on the development of MC schemes is also provided, followed by
the basic MC method and a brief description of the rejection sampling (RS)
algorithm, as well as three sections describing many of the most relevant MCMC
and IS algorithms, and their combined use.

    

### [[2107.11838] New Algebraic Normative Theories for Ethical and Legal Reasoning in the LogiKEy Framework](http://arxiv.org/abs/2107.11838)


  To design and engineer ethical and legal reasoners and responsible systems,
Benzmüller, Parent and van der Torre introduce LogiKEy methodology based on
the semantical embedding of deontic logics into classic higher-order logic. In
this paper, we considerably extend the LogiKEy deontic logics and dataset using
an algebraic approach. We develop theory of input/output operations for
normative reasoning on top of Boolean algebras.

    

### [[2107.11845] On-Device Content Moderation](http://arxiv.org/abs/2107.11845)


  With the advent of internet, not safe for work(NSFW) content moderation is a
major problem today. Since,smartphones are now part of daily life of billions
of people,it becomes even more important to have a solution which coulddetect
and suggest user about potential NSFW content present ontheir phone. In this
paper we present a novel on-device solutionfor detecting NSFW images. In
addition to conventional porno-graphic content moderation, we have also
included semi-nudecontent moderation as it is still NSFW in a large
demography.We have curated a dataset comprising of three major
categories,namely nude, semi-nude and safe images. We have created anensemble
of object detector and classifier for filtering of nudeand semi-nude contents.
The solution provides unsafe body partannotations along with identification of
semi-nude images. Weextensively tested our proposed solution on several public
datasetand also on our custom dataset. The model achieves F1 scoreof 0.91 with
95% precision and 88% recall on our customNSFW16k dataset and 0.92 MAP on NPDI
dataset. Moreover itachieves average 0.002 false positive rate on a collection
of safeimage open datasets.

    

### [[2107.11879] Hybrid Autoregressive Solver for Scalable Abductive Natural Language Inference](http://arxiv.org/abs/2107.11879)


  Regenerating natural language explanations for science questions is a
challenging task for evaluating complex multi-hop and abductive inference
capabilities. In this setting, Transformers trained on human-annotated
explanations achieve state-of-the-art performance when adopted as cross-encoder
architectures. However, while much attention has been devoted to the quality of
the constructed explanations, the problem of performing abductive inference at
scale is still under-studied. As intrinsically not scalable, the cross-encoder
architectural paradigm is not suitable for efficient multi-hop inference on
massive facts banks. To maximise both accuracy and inference time, we propose a
hybrid abductive solver that autoregressively combines a dense bi-encoder with
a sparse model of explanatory power, computed leveraging explicit patterns in
the explanations. Our experiments demonstrate that the proposed framework can
achieve performance comparable with the state-of-the-art cross-encoder while
being $\approx 50$ times faster and scalable to corpora of millions of facts.
Moreover, we study the impact of the hybridisation on semantic drift and
science question answering without additional training, showing that it boosts
the quality of the explanations and contributes to improved downstream
inference performance.

    

### [[2107.11904] Transferable Dialogue Systems and User Simulators](http://arxiv.org/abs/2107.11904)


  One of the difficulties in training dialogue systems is the lack of training
data. We explore the possibility of creating dialogue data through the
interaction between a dialogue system and a user simulator. Our goal is to
develop a modelling framework that can incorporate new dialogue scenarios
through self-play between the two agents. In this framework, we first pre-train
the two agents on a collection of source domain dialogues, which equips the
agents to converse with each other via natural language. With further
fine-tuning on a small amount of target domain data, the agents continue to
interact with the aim of improving their behaviors using reinforcement learning
with structured reward functions. In experiments on the MultiWOZ dataset, two
practical transfer learning problems are investigated: 1) domain adaptation and
2) single-to-multiple domain transfer. We demonstrate that the proposed
framework is highly effective in bootstrapping the performance of the two
agents in transfer learning. We also show that our method leads to improvements
in dialogue system performance on complete datasets.

    

### [[2107.11927] On Blame Attribution for Accountable Multi-Agent Sequential Decision Making](http://arxiv.org/abs/2107.11927)


  Blame attribution is one of the key aspects of accountable decision making,
as it provides means to quantify the responsibility of an agent for a decision
making outcome. In this paper, we study blame attribution in the context of
cooperative multi-agent sequential decision making. As a particular setting of
interest, we focus on cooperative decision making formalized by Multi-Agent
Markov Decision Processes (MMDP), and we analyze different blame attribution
methods derived from or inspired by existing concepts in cooperative game
theory. We formalize desirable properties of blame attribution in the setting
of interest, and we analyze the relationship between these properties and the
studied blame attribution methods. Interestingly, we show that some of the well
known blame attribution methods, such as Shapley value, are not
performance-incentivizing, while others, such as Banzhaf index, may over-blame
agents. To mitigate these value misalignment and fairness issues, we introduce
a novel blame attribution method, unique in the set of properties it satisfies,
which trade-offs explanatory power (by under-blaming agents) for the
aforementioned properties. We further show how to account for uncertainty about
agents' decision making policies, and we experimentally: a) validate the
qualitative properties of the studied blame attribution methods, and b) analyze
their robustness to uncertainty.

    

### [[2107.11934] Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection](http://arxiv.org/abs/2107.11934)


  Detecting rumors on social media is a very critical task with significant
implications to the economy, public health, etc. Previous works generally
capture effective features from texts and the propagation structure. However,
the uncertainty caused by unreliable relations in the propagation structure is
common and inevitable due to wily rumor producers and the limited collection of
spread data. Most approaches neglect it and may seriously limit the learning of
features. Towards this issue, this paper makes the first attempt to explore
propagation uncertainty for rumor detection. Specifically, we propose a novel
Edge-enhanced Bayesian Graph Convolutional Network (EBGCN) to capture robust
structural features. The model adaptively rethinks the reliability of latent
relations by adopting a Bayesian approach. Besides, we design a new edge-wise
consistency training framework to optimize the model by enforcing consistency
on relations. Experiments on three public benchmark datasets demonstrate that
the proposed model achieves better performance than baseline methods on both
rumor detection and early rumor detection tasks.

    

### [[2107.11965] Playtesting: What is Beyond Personas](http://arxiv.org/abs/2107.11965)


  Playtesting is an essential step in the game design process. Game designers
use the feedback from playtests to refine their design. Game designers may
employ procedural personas to automate the playtesting process. In this paper,
we present two approaches to improve automated playtesting. First, we propose a
goal-based persona model, which we call developing persona -- developing
persona proposes a dynamic persona model, whereas the current persona models
are static. Game designers can use the developing persona to model the changes
that a player undergoes while playing a game. Additionally, a human playtester
knows which paths she has tested before, and during the consequent tests, she
may test different paths. However, RL agents disregard the previously generated
trajectories. We propose a novel methodology that helps Reinforcement Learning
(RL) agents to generate distinct trajectories than the previous trajectories.
We refer to this methodology as Alternative Path Finder (APF). We present a
generic APF framework that can be applied to all RL agents. APF is trained with
the previous trajectories, and APF distinguishes the novel states from similar
states. We use the General Video Game Artificial Intelligence (GVG-AI) and
VizDoom frameworks to test our proposed methodologies. We use Proximal Policy
Optimization (PPO) RL agent during experiments. First, we show that the
playtest data generated by the developing persona cannot be generated using the
procedural personas. Second, we present the alternative paths found using APF.
We show that the APF penalizes the previous paths and rewards the distinct
paths.

    

### [[2107.11986] Benign Adversarial Attack: Tricking Algorithm for Goodness](http://arxiv.org/abs/2107.11986)


  In spite of the successful application in many fields, machine learning
algorithms today suffer from notorious problems like vulnerability to
adversarial examples. Beyond falling into the cat-and-mouse game between
adversarial attack and defense, this paper provides alternative perspective to
consider adversarial example and explore whether we can exploit it in benign
applications. We first propose a novel taxonomy of visual information along
task-relevance and semantic-orientation. The emergence of adversarial example
is attributed to algorithm's utilization of task-relevant non-semantic
information. While largely ignored in classical machine learning mechanisms,
task-relevant non-semantic information enjoys three interesting characteristics
as (1) exclusive to algorithm, (2) reflecting common weakness, and (3)
utilizable as features. Inspired by this, we present brave new idea called
benign adversarial attack to exploit adversarial examples for goodness in three
directions: (1) adversarial Turing test, (2) rejecting malicious algorithm, and
(3) adversarial data augmentation. Each direction is positioned with motivation
elaboration, justification analysis and prototype applications to showcase its
potential.

    

### [[2107.12024] Leaf-FM: A Learnable Feature Generation Factorization Machine for Click-Through Rate Prediction](http://arxiv.org/abs/2107.12024)


  Click-through rate (CTR) prediction plays important role in personalized
advertising and recommender systems. Though many models have been proposed such
as FM, FFM and DeepFM in recent years, feature engineering is still a very
important way to improve the model performance in many applications because
using raw features can rarely lead to optimal results. For example, the
continuous features are usually transformed to the power forms by adding a new
feature to allow it to easily form non-linear functions of the feature.
However, this kind of feature engineering heavily relies on peoples experience
and it is both time consuming and labor consuming. On the other side, concise
CTR model with both fast online serving speed and good model performance is
critical for many real life applications. In this paper, we propose LeafFM
model based on FM to generate new features from the original feature embedding
by learning the transformation functions automatically. We also design three
concrete Leaf-FM models according to the different strategies of combing the
original and the generated features. Extensive experiments are conducted on
three real-world datasets and the results show Leaf-FM model outperforms
standard FMs by a large margin. Compared with FFMs, Leaf-FM can achieve
significantly better performance with much less parameters. In Avazu and
Malware dataset, add version Leaf-FM achieves comparable performance with some
deep learning based models such as DNN and AutoInt. As an improved FM model,
Leaf-FM has the same computation complexity with FM in online serving phase and
it means Leaf-FM is applicable in many industry applications because of its
better performance and high computation efficiency.

    

### [[2107.12025] ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding](http://arxiv.org/abs/2107.12025)


  Click-through rate (CTR) estimation is a fundamental task in personalized
advertising and recommender systems and it's important for ranking models to
effectively capture complex high-order features.Inspired by the success of ELMO
and Bert in NLP field, which dynamically refine word embedding according to the
context sentence information where the word appears, we think it's also
important to dynamically refine each feature's embedding layer by layer
according to the context information contained in input instance in CTR
estimation tasks. We can effectively capture the useful feature interactions
for each feature in this way. In this paper, We propose a novel CTR Framework
named ContextNet that implicitly models high-order feature interactions by
dynamically refining each feature's embedding according to the input context.
Specifically, ContextNet consists of two key components: contextual embedding
module and ContextNet block. Contextual embedding module aggregates contextual
information for each feature from input instance and ContextNet block maintains
each feature's embedding layer by layer and dynamically refines its
representation by merging contextual high-order interaction information into
feature embedding. To make the framework specific, we also propose two
models(ContextNet-PFFN and ContextNet-SFFN) under this framework by introducing
linear contextual embedding network and two non-linear mapping sub-network in
ContextNet block. We conduct extensive experiments on four real-world datasets
and the experiment results demonstrate that our proposed ContextNet-PFFN and
ContextNet-SFFN model outperform state-of-the-art models such as DeepFM and
xDeepFM significantly.

    

### [[2107.12049] SVEva Fair: A Framework for Evaluating Fairness in Speaker Verification](http://arxiv.org/abs/2107.12049)


  Despite the success of deep neural networks (DNNs) in enabling on-device
voice assistants, increasing evidence of bias and discrimination in machine
learning is raising the urgency of investigating the fairness of these systems.
Speaker verification is a form of biometric identification that gives access to
voice assistants. Due to a lack of fairness metrics and evaluation frameworks
that are appropriate for testing the fairness of speaker verification
components, little is known about how model performance varies across
subgroups, and what factors influence performance variation. To tackle this
emerging challenge, we design and develop SVEva Fair, an accessible, actionable
and model-agnostic framework for evaluating the fairness of speaker
verification components. The framework provides evaluation measures and
visualisations to interrogate model performance across speaker subgroups and
compare fairness between models. We demonstrate SVEva Fair in a case study with
end-to-end DNNs trained on the VoxCeleb datasets to reveal potential bias in
existing embedded speech recognition systems based on the demographic
attributes of speakers. Our evaluation shows that publicly accessible benchmark
models are not fair and consistently produce worse predictions for some
nationalities, and for female speakers of most nationalities. To pave the way
for fair and reliable embedded speaker verification, SVEva Fair has been
implemented as an open-source python library and can be integrated into the
embedded ML development pipeline to facilitate developers and researchers in
troubleshooting unreliable speaker verification performance, and selecting high
impact approaches for mitigating fairness challenges

    

### [[2107.12051] Adaptation of Tacotron2-based Text-To-Speech for Articulatory-to-Acoustic Mapping using Ultrasound Tongue Imaging](http://arxiv.org/abs/2107.12051)


  For articulatory-to-acoustic mapping, typically only limited parallel
training data is available, making it impossible to apply fully end-to-end
solutions like Tacotron2. In this paper, we experimented with transfer learning
and adaptation of a Tacotron2 text-to-speech model to improve the final
synthesis quality of ultrasound-based articulatory-to-acoustic mapping with a
limited database. We use a multi-speaker pre-trained Tacotron2 TTS model and a
pre-trained WaveGlow neural vocoder. The articulatory-to-acoustic conversion
contains three steps: 1) from a sequence of ultrasound tongue image recordings,
a 3D convolutional neural network predicts the inputs of the pre-trained
Tacotron2 model, 2) the Tacotron2 model converts this intermediate
representation to an 80-dimensional mel-spectrogram, and 3) the WaveGlow model
is applied for final inference. This generated speech contains the timing of
the original articulatory data from the ultrasound recording, but the F0
contour and the spectral information is predicted by the Tacotron2 model. The
F0 values are independent of the original ultrasound images, but represent the
target speaker, as they are inferred from the pre-trained Tacotron2 model. In
our experiments, we demonstrated that the synthesized speech quality is more
natural with the proposed solutions than with our earlier model.

    

### [[2107.12061] Predicting Game Engagement and Difficulty Using AI Players](http://arxiv.org/abs/2107.12061)


  This paper presents a novel approach to automated playtesting for the
prediction of human player behavior and experience. It has previously been
demonstrated that Deep Reinforcement Learning (DRL) game-playing agents can
predict both game difficulty and player engagement, operationalized as average
pass and churn rates. We improve this approach by enhancing DRL with Monte
Carlo Tree Search (MCTS). We also motivate an enhanced selection strategy for
predictor features, based on the observation that an AI agent's best-case
performance can yield stronger correlations with human data than the agent's
average performance. Both additions consistently improve the prediction
accuracy, and the DRL-enhanced MCTS outperforms both DRL and vanilla MCTS in
the hardest levels. We conclude that player modelling via automated playtesting
can benefit from combining DRL and MCTS. Moreover, it can be worthwhile to
investigate a subset of repeated best AI agent runs, if AI gameplay does not
yield good predictions on average.

    

### [[2107.12064] How Knowledge Graph and Attention Help? A Quantitative Analysis into Bag-level Relation Extraction](http://arxiv.org/abs/2107.12064)


  Knowledge Graph (KG) and attention mechanism have been demonstrated effective
in introducing and selecting useful information for weakly supervised methods.
However, only qualitative analysis and ablation study are provided as evidence.
In this paper, we contribute a dataset and propose a paradigm to quantitatively
evaluate the effect of attention and KG on bag-level relation extraction (RE).
We find that (1) higher attention accuracy may lead to worse performance as it
may harm the model's ability to extract entity mention features; (2) the
performance of attention is largely influenced by various noise distribution
patterns, which is closely related to real-world datasets; (3) KG-enhanced
attention indeed improves RE performance, while not through enhanced attention
but by incorporating entity prior; and (4) attention mechanism may exacerbate
the issue of insufficient training data. Based on these findings, we show that
a straightforward variant of RE model can achieve significant improvements (6%
AUC on average) on two real-world datasets as compared with three
state-of-the-art baselines. Our codes and datasets are available at
this https URL.

    

### [[2107.12085] Learning to Adversarially Blur Visual Object Tracking](http://arxiv.org/abs/2107.12085)


  Motion blur caused by the moving of the object or camera during the exposure
can be a key challenge for visual object tracking, affecting tracking accuracy
significantly. In this work, we explore the robustness of visual object
trackers against motion blur from a new angle, i.e., adversarial blur attack
(ABA). Our main objective is to online transfer input frames to their natural
motion-blurred counterparts while misleading the state-of-the-art trackers
during the tracking process. To this end, we first design the motion blur
synthesizing method for visual tracking based on the generation principle of
motion blur, considering the motion information and the light accumulation
process. With this synthetic method, we propose \textit{optimization-based ABA
(OP-ABA)} by iteratively optimizing an adversarial objective function against
the tracking w.r.t. the motion and light accumulation parameters. The OP-ABA is
able to produce natural adversarial examples but the iteration can cause heavy
time cost, making it unsuitable for attacking real-time trackers. To alleviate
this issue, we further propose \textit{one-step ABA (OS-ABA)} where we design
and train a joint adversarial motion and accumulation predictive network
(JAMANet) with the guidance of OP-ABA, which is able to efficiently estimate
the adversarial motion and accumulation parameters in a one-step way. The
experiments on four popular datasets (\eg, OTB100, VOT2018, UAV123, and LaSOT)
demonstrate that our methods are able to cause significant accuracy drops on
four state-of-the-art trackers with high transferability. Please find the
source code at this https URL


### [[2107.12130] Structural Learning of Probabilistic Sentential Decision Diagrams under Partial Closed-World Assumption](http://arxiv.org/abs/2107.12130)


  Probabilistic sentential decision diagrams are a class of
structured-decomposable probabilistic circuits especially designed to embed
logical constraints. To adapt the classical LearnSPN scheme to learn the
structure of these models, we propose a new scheme based on a partial
closed-world assumption: data implicitly provide the logical base of the
circuit. Sum nodes are thus learned by recursively clustering batches in the
initial data base, while the partitioning of the variables obeys a given input
vtree. Preliminary experiments show that the proposed approach might properly
fit training data, and generalize well to test data, provided that these remain
consistent with the underlying logical base, that is a relaxation of the
training data base.

    

### [[2107.12135] Fine-Grained Emotion Prediction by Modeling Emotion Definitions](http://arxiv.org/abs/2107.12135)


  In this paper, we propose a new framework for fine-grained emotion prediction
in the text through emotion definition modeling. Our approach involves a
multi-task learning framework that models definitions of emotions as an
auxiliary task while being trained on the primary task of emotion prediction.
We model definitions using masked language modeling and class definition
prediction tasks. Our models outperform existing state-of-the-art for
fine-grained emotion dataset GoEmotions. We further show that this trained
model can be used for transfer learning on other benchmark datasets in emotion
prediction with varying emotion label sets, domains, and sizes. The proposed
models outperform the baselines on transfer learning experiments demonstrating
the generalization capability of the models.

    

### [[2107.12143] Perceptually Validated Precise Local Editing for Facial Action Units with StyleGAN](http://arxiv.org/abs/2107.12143)


  The ability to edit facial expressions has a wide range of applications in
computer graphics. The ideal facial expression editing algorithm needs to
satisfy two important criteria. First, it should allow precise and targeted
editing of individual facial actions. Second, it should generate high fidelity
outputs without artifacts. We build a solution based on StyleGAN, which has
been used extensively for semantic manipulation of faces. As we do so, we add
to our understanding of how various semantic attributes are encoded in
StyleGAN. In particular, we show that a naive strategy to perform editing in
the latent space results in undesired coupling between certain action units,
even if they are conceptually distinct. For example, although brow lowerer and
lip tightener are distinct action units, they appear correlated in the training
data. Hence, StyleGAN has difficulty in disentangling them. We allow
disentangled editing of such action units by computing detached regions of
influence for each action unit, and restrict editing to these regions. We
validate the effectiveness of our local editing method through perception
experiments conducted with 23 subjects. The results show that our method
provides higher control over local editing and produces images with superior
fidelity compared to the state-of-the-art methods.

    

### [[2107.12178] Novel Span Measure, Spanning Sets and Applications](http://arxiv.org/abs/2107.12178)


  Rough Set based Spanning Sets were recently proposed to deal with
uncertainties arising in the problem in domain of natural language processing
problems. This paper presents a novel span measure using upper approximations.
The key contribution of this paper is to propose another uncertainty measure of
span and spanning sets. Firstly, this paper proposes a new definition of
computing span which use upper approximation instead of boundary regions. This
is useful in situations where computing upper approximations are much more
convenient that computing boundary region. Secondly, properties of novel span
and relation with earlier span measure are discussed. Thirdly, the paper
presents application areas where the proposed span measure can be utilized.

    

### [[2107.12189] An Efficient Insect Pest Classification Using Multiple Convolutional Neural Network Based Models](http://arxiv.org/abs/2107.12189)


  Accurate insect pest recognition is significant to protect the crop or take
the early treatment on the infected yield, and it helps reduce the loss for the
agriculture economy. Design an automatic pest recognition system is necessary
because manual recognition is slow, time-consuming, and expensive. The
Image-based pest classifier using the traditional computer vision method is not
efficient due to the complexity. Insect pest classification is a difficult task
because of various kinds, scales, shapes, complex backgrounds in the field, and
high appearance similarity among insect species. With the rapid development of
deep learning technology, the CNN-based method is the best way to develop a
fast and accurate insect pest classifier. We present different convolutional
neural network-based models in this work, including attention, feature pyramid,
and fine-grained models. We evaluate our methods on two public datasets: the
large-scale insect pest dataset, the IP102 benchmark dataset, and a smaller
dataset, namely D0 in terms of the macro-average precision (MPre), the
macro-average recall (MRec), the macro-average F1- score (MF1), the accuracy
(Acc), and the geometric mean (GM). The experimental results show that
combining these convolutional neural network-based models can better perform
than the state-of-the-art methods on these two datasets. For instance, the
highest accuracy we obtained on IP102 and D0 is $74.13\%$ and $99.78\%$,
respectively, bypassing the corresponding state-of-the-art accuracy: $67.1\%$
(IP102) and $98.8\%$ (D0). We also publish our codes for contributing to the
current research related to the insect pest classification problem.

    

### [[2107.12226] DYPLODOC: Dynamic Plots for Document Classification](http://arxiv.org/abs/2107.12226)


  Narrative generation and analysis are still on the fringe of modern natural
language processing yet are crucial in a variety of applications. This paper
proposes a feature extraction method for plot dynamics. We present a dataset
that consists of the plot descriptions for thirteen thousand TV shows alongside
meta-information on their genres and dynamic plots extracted from them. We
validate the proposed tool for plot dynamics extraction and discuss possible
applications of this method to the tasks of narrative analysis and generation.

    

### [[2107.12230] Belief Propagation as Diffusion](http://arxiv.org/abs/2107.12230)


  We introduce novel belief propagation algorithms to estimate the marginals of
a high dimensional probability distribution. They involve natural
(co)homological constructions relevant for a localised description of
statistical systems.

    

### [[2010.05418] Achilles Heels for AGI/ASI via Decision Theoretic Adversaries](http://arxiv.org/abs/2010.05418)


  As progress in AI continues to advance, it is crucial to know how advanced
systems will make choices and in what ways they may fail. Machines can already
outsmart humans in some domains, and understanding how to safely build ones
which may have capabilities at or above the human level is of particular
concern. One might suspect that artificially generally intelligent (AGI) and
artificially superintelligent (ASI) systems should be modeled as as something
which humans, by definition, can't reliably outsmart. As a challenge to this
assumption, this paper presents the Achilles Heel hypothesis which states that
even a potentially superintelligent system may nonetheless have stable
decision-theoretic delusions which cause them to make obviously irrational
decisions in adversarial settings. In a survey of relevant dilemmas and
paradoxes from the decision theory literature, a number of these potential
Achilles Heels are discussed in context of this hypothesis. Several novel
contributions are made toward understanding the ways in which these weaknesses
might be implanted into a system.

    

### [[2010.07038] OnRAMP for Regulating AI in Medical Products](http://arxiv.org/abs/2010.07038)


  Medical Artificial Intelligence (AI) involves the application of machine
learning algorithms to biomedical datasets in order to improve medical
practices. Products incorporating medical AI require certification before
deployment in most jurisdictions. To date, clear pathways for regulating
medical AI are still under development. Below the level of formal pathways lies
the actual practice of developing a medical AI solution. This Perspective
proposes best practice guidelines for development compatible with the
production of a regulatory package which, regardless of the formal regulatory
path, will form a core component of a certification process. The approach is
predicated on a statistical risk perspective, typical of medical device
regulators, and a deep understanding of machine learning methodologies. These
guidelines will allow all parties to communicate more clearly in the
development of a common Good Machine Learning Practice (GMLP), and thus lead to
the enhanced development of both medical AI products and regulations.

    

### [[2012.08911] Communicative Message Passing for Inductive Relation Reasoning](http://arxiv.org/abs/2012.08911)


  Relation prediction for knowledge graphs aims at predicting missing
relationships between entities. Despite the importance of inductive relation
prediction, most previous works are limited to a transductive setting and
cannot process previously unseen entities. The recent proposed subgraph-based
relation reasoning models provided alternatives to predict links from the
subgraph structure surrounding a candidate triplet inductively. However, we
observe that these methods often neglect the directed nature of the extracted
subgraph and weaken the role of relation information in the subgraph modeling.
As a result, they fail to effectively handle the asymmetric/anti-symmetric
triplets and produce insufficient embeddings for the target triplets. To this
end, we introduce a \textbf{C}\textbf{o}mmunicative \textbf{M}essage
\textbf{P}assing neural network for \textbf{I}nductive re\textbf{L}ation
r\textbf{E}asoning, \textbf{CoMPILE}, that reasons over local directed subgraph
structures and has a vigorous inductive bias to process entity-independent
semantic relations. In contrast to existing models, CoMPILE strengthens the
message interactions between edges and entitles through a communicative kernel
and enables a sufficient flow of relation information. Moreover, we demonstrate
that CoMPILE can naturally handle asymmetric/anti-symmetric relations without
the need for explosively increasing the number of model parameters by
extracting the directed enclosing subgraphs. Extensive experiments show
substantial performance gains in comparison to state-of-the-art methods on
commonly used benchmark datasets with variant inductive settings.

    

### [[2103.05266] BASAR:Black-box Attack on Skeletal Action Recognition](http://arxiv.org/abs/2103.05266)


  Skeletal motion plays a vital role in human activity recognition as either an
independent data source or a complement. The robustness of skeleton-based
activity recognizers has been questioned recently, which shows that they are
vulnerable to adversarial attacks when the full-knowledge of the recognizer is
accessible to the attacker. However, this white-box requirement is overly
restrictive in most scenarios and the attack is not truly threatening. In this
paper, we show that such threats do exist under black-box settings too. To this
end, we propose the first black-box adversarial attack method BASAR. Through
BASAR, we show that adversarial attack is not only truly a threat but also can
be extremely deceitful, because on-manifold adversarial samples are rather
common in skeletal motions, in contrast to the common belief that adversarial
samples only exist off-manifold. Through exhaustive evaluation and comparison,
we show that BASAR can deliver successful attacks across models, data, and
attack modes. Through harsh perceptual studies, we show that it achieves
effective yet imperceptible attacks. By analyzing the attack on different
activity recognizers, BASAR helps identify the potential causes of their
vulnerability and provides insights on what classifiers are likely to be more
robust against attack. Code is available at
this https URL.

    

### [[2104.06832] Image Manipulation Detection by Multi-View Multi-Scale Supervision](http://arxiv.org/abs/2104.06832)


  The key challenge of image manipulation detection is how to learn
generalizable features that are sensitive to manipulations in novel data,
whilst specific to prevent false alarms on authentic images. Current research
emphasizes the sensitivity, with the specificity overlooked. In this paper we
address both aspects by multi-view feature learning and multi-scale
supervision. By exploiting noise distribution and boundary artifact surrounding
tampered regions, the former aims to learn semantic-agnostic and thus more
generalizable features. The latter allows us to learn from authentic images
which are nontrivial to be taken into account by current semantic segmentation
network based methods. Our thoughts are realized by a new network which we term
MVSS-Net. Extensive experiments on five benchmark sets justify the viability of
MVSS-Net for both pixel-level and image-level manipulation detection.

    

### [[2104.10719] A Fully Spiking Hybrid Neural Network for Energy-Efficient Object Detection](http://arxiv.org/abs/2104.10719)


  This paper proposes a Fully Spiking Hybrid Neural Network (FSHNN) for
energy-efficient and robust object detection in resource-constrained platforms.
The network architecture is based on Convolutional SNN using
leaky-integrate-fire neuron models. The model combines unsupervised Spike
Time-Dependent Plasticity (STDP) learning with back-propagation (STBP) learning
methods and also uses Monte Carlo Dropout to get an estimate of the uncertainty
error. FSHNN provides better accuracy compared to DNN based object detectors
while being 150X energy-efficient. It also outperforms these object detectors,
when subjected to noisy input data and less labeled training data with a lower
uncertainty error.

    

### [[2104.10857] Attribute-Modulated Generative Meta Learning for Zero-Shot Classification](http://arxiv.org/abs/2104.10857)


  Zero-shot learning (ZSL) aims to transfer knowledge from seen classes to
semantically related unseen classes, which are absent during training. The
promising strategies for ZSL are to synthesize visual features of unseen
classes conditioned on semantic side information and to incorporate
meta-learning to eliminate the model's inherent bias towards seen classes.
While existing meta generative approaches pursue a common model shared across
task distributions, we aim to construct a generative network adaptive to task
characteristics. To this end, we propose an Attribute-Modulated generAtive
meta-model for Zero-shot learning (AMAZ). Our model consists of an
attribute-aware modulation network, an attribute-augmented generative network,
and an attribute-weighted classifier. Given unseen classes, the modulation
network adaptively modulates the generator by applying task-specific
transformations so that the generative network can adapt to highly diverse
tasks. The weighted classifier utilizes the data quality to enhance the
training procedure, further improving the model performance. Our empirical
evaluations on four widely-used benchmarks show that AMAZ outperforms
state-of-the-art methods by 3.8% and 3.1% in ZSL and generalized ZSL settings,
respectively, demonstrating the superiority of our method. Our experiments on a
zero-shot image retrieval task show AMAZ's ability to synthesize instances that
portray real visual characteristics.

    

### [[2105.01882] DeepPlastic: A Novel Approach to Detecting Epipelagic Bound Plastic Using Deep Visual Models](http://arxiv.org/abs/2105.01882)


  The quantification of positively buoyant marine plastic debris is critical to
understanding how concentrations of trash from across the world's ocean and
identifying high concentration garbage hotspots in dire need of trash removal.
Currently, the most common monitoring method to quantify floating plastic
requires the use of a manta trawl. Techniques requiring manta trawls (or
similar surface collection devices) utilize physical removal of marine plastic
debris as the first step and then analyze collected samples as a second step.
The need for physical removal before analysis incurs high costs and requires
intensive labor preventing scalable deployment of a real-time marine plastic
monitoring service across the entirety of Earth's ocean bodies. Without better
monitoring and sampling methods, the total impact of plastic pollution on the
environment as a whole, and details of impact within specific oceanic regions,
will remain unknown. This study presents a highly scalable workflow that
utilizes images captured within the epipelagic layer of the ocean as an input.
It produces real-time quantification of marine plastic debris for accurate
quantification and physical removal. The workflow includes creating and
preprocessing a domain-specific dataset, building an object detection model
utilizing a deep neural network, and evaluating the model's performance.
YOLOv5-S was the best performing model, which operates at a Mean Average
Precision (mAP) of 0.851 and an F1-Score of 0.89 while maintaining
near-real-time speed.

    

### [[2105.05424] Transitioning to human interaction with AI systems: New challenges and opportunities for HCI professionals to enable human-centered AI](http://arxiv.org/abs/2105.05424)


  While AI has benefited humans, it may also harm humans if not appropriately
developed. The focus of HCI work is transiting from conventional human
interaction with non-AI computing systems to interaction with AI systems. We
conducted a high-level literature review and a holistic analysis of current
work in developing AI systems from an HCI perspective. Our review and analysis
highlight the new changes introduced by AI technology and the new challenges
that HCI professionals face when applying the human-centered AI (HCAI) approach
in the development of AI systems. We also identified seven main issues in human
interaction with AI systems, which HCI professionals did not encounter when
developing non-AI computing systems. To further enable the implementation of
the HCAI approach, we identified new HCI opportunities tied to specific
HCAI-driven design goals to guide HCI professionals in addressing these new
issues. Finally, our assessment of current HCI methods shows the limitations of
these methods in support of developing AI systems. We propose alternative
methods that can help overcome these limitations and effectively help HCI
professionals apply the HCAI approach to the development of AI systems. We also
offer strategic recommendations for HCI professionals to effectively influence
the development of AI systems with the HCAI approach, eventually developing
HCAI systems.

    

### [[2106.11008] Wheelchair automation by a hybrid BCI system using SSVEP and eye blinks](http://arxiv.org/abs/2106.11008)


  This work proposes a hybrid Brain Computer Interface system for the
automation of a wheelchair for the disabled. Herein a working prototype of a
BCI-based wheelchair is detailed that can navigate inside a typical home
environment with minimum structural modification and without any visual
obstruction and discomfort to the user. The prototype is based on a combined
mechanism of steady-state visually evoked potential and eye blinks. To elicit
SSVEP, LEDs flickering at 13Hz and 15Hz were used to select the left and right
direction, respectively, and EEG data was recorded. In addition, the occurrence
of three continuous blinks was used as an indicator for stopping an ongoing
action. The wavelet packet denoising method was applied, followed by feature
extraction methods such as Wavelet Packet Decomposition and Canonical
Correlation Analysis over narrowband reconstructed EEG signals. Bayesian
optimization was used to obtain 5 fold cross-validations to optimize the
hyperparameters of the Support Vector Machine. The resulting new model was
tested and the average cross-validation accuracy 89.65% + 6.6% (SD) and testing
accuracy 83.53% + 8.59% (SD) were obtained. The wheelchair was controlled by
RaspberryPi through WiFi. The developed prototype demonstrated an average of
86.97% success rate for all trials with 4.015s for each command execution. The
prototype can be used efficiently in a home environment without causing any
discomfort to the user.

    

### [[2107.11674] Case Studies in Formal Reasoning About Lambda-Calculus: Semantics, Church-Rosser, Standardization and HOAS](http://arxiv.org/abs/2107.11674)


  We have previously published the Isabelle/HOL formalization of a general
theory of syntax with bindings. In this companion paper, we instantiate the
general theory to the syntax of lambda-calculus and formalize the development
leading to several fundamental constructions and results: sound semantic
interpretation, the Church-Rosser and standardization theorems, and
higher-order abstract syntax (HOAS) encoding. For Church-Rosser and
standardization, our work covers both the call-by-name and call-by-value
versions of the calculus, following classic papers by Takahashi and Plotkin.
During the formalization, we were able to stay focused on the high-level ideas
of the development -- thanks to the arsenal provided by our general theory: a
wealth of basic facts about the substitution, swapping and freshness operators,
as well as recursive-definition and reasoning principles, including a
specialization to semantic interpretation of syntax.

    

### [[2107.11679] Reasoning about Recursive Quantum Programs](http://arxiv.org/abs/2107.11679)


  Most modern (classical) programming languages support recursion. Recursion
has also been successfully applied to the design of several quantum algorithms
and introduced in a couple of quantum programming languages. So, it can be
expected that recursion will become one of the fundamental paradigms of quantum
programming. Several program logics have been developed for verification of
quantum while-programs. However, there are as yet no general methods for
reasoning about (mutual) recursive procedures and ancilla quantum data
structure in quantum computing (with measurement). We fill the gap in this
paper by proposing a parameterized quantum assertion logic and, based on which,
designing a quantum Hoare logic for verifying parameterized recursive quantum
programs with ancilla data and probabilistic control. The quantum Hoare logic
can be used to prove partial, total, and even probabilistic correctness (by
reducing to total correctness) of those quantum programs. In particular, two
counterexamples for illustrating incompleteness of non-parameterized assertions
in verifying recursive procedures, and, one counterexample for showing the
failure of reasoning with exact probabilities based on partial correctness, are
constructed. The effectiveness of our logic is shown by three main examples --
recursive quantum Markov chain (with probabilistic control), fixed-point
Grover's search, and recursive quantum Fourier sampling.

    

### [[2107.12136] The Role of Functional Programming in Management and Orchestration of Virtualized Network Resources Part I. System structure for Complex Systems and Design Principles](http://arxiv.org/abs/2107.12136)


  This is part I of the follow-up lecture notes of the lectures given by the
authors at the Three \CO" (Composability, Comprehensibility, Correctness)
Winter School held in Košice, Slovakia, in January 2018, and Summer School
held in Budapest, Hungary, in June 2019. In this part we explain the role of
functional programming paradigm in the management of complex software systems,
and how the functional programming concepts play important role in the
designing such systems. Key prerequisite for implementing functional
programming concepts is properly designed system structure following well
defined design principles and rules. That is the main goal of this lecture to
introduce students with proper system modeling. Furthermore, we also explain
how new emerging technologies are designed in such a way that they enforce the
development of systems that comply to the design rules inspired by the
functional programming. This is extremely important in view of the current
network evolution and virtualization concepts, which will require many
functional programming concepts in the network services and functions, as will
be discussed in part II of these lecture notes. These notes provide an
introduction to the subject, with the goal of explaining the problems and the
principles, methods and techniques used for their solution. The worked examples
and exercises serve students as the teaching material, from which they can
learn how to use design principles to model effective system structures. Here
we focus on students understanding of importance of effective system structures
for coordination of development and management processes that are driven by
business goals and further evolution.

    

### [[2107.12144] Quantum Information Effects](http://arxiv.org/abs/2107.12144)


  We study the two dual quantum information effects to manipulate the amount of
information in quantum computation: hiding and allocation. The resulting
type-and-effect system is fully expressive for irreversible quantum computing,
including measurement. We provide universal categorical constructions that
semantically interpret this arrow metalanguage with choice, starting with any
rig groupoid interpreting the reversible base language. Several properties of
quantum measurement follow in general, and we translate quantum flow charts
into our language. The semantic constructions turn the category of unitaries
between Hilbert spaces into the category of completely positive
trace-preserving maps, and they turn the category of bijections between finite
sets into the category of functions with chosen garbage. Thus they capture the
fundamental theorems of classical and quantum reversible computing of Toffoli
and Stinespring.

    

### [[2007.14381] BUSTLE: Bottom-Up Program Synthesis Through Learning-Guided Exploration](http://arxiv.org/abs/2007.14381)


  Program synthesis is challenging largely because of the difficulty of search
in a large space of programs. Human programmers routinely tackle the task of
writing complex programs by writing sub-programs and then analyzing their
intermediate results to compose them in appropriate ways. Motivated by this
intuition, we present a new synthesis approach that leverages learning to guide
a bottom-up search over programs. In particular, we train a model to prioritize
compositions of intermediate values during search conditioned on a given set of
input-output examples. This is a powerful combination because of several
emergent properties. First, in bottom-up search, intermediate programs can be
executed, providing semantic information to the neural network. Second, given
the concrete values from those executions, we can exploit rich features based
on recent work on property signatures. Finally, bottom-up search allows the
system substantial flexibility in what order to generate the solution, allowing
the synthesizer to build up a program from multiple smaller sub-programs.
Overall, our empirical evaluation finds that the combination of learning and
bottom-up search is remarkably effective, even with simple supervised learning
approaches. We demonstrate the effectiveness of our technique on two datasets,
one from the SyGuS competition and one of our own creation.

    

### [<title>Re: XGBoost 1.5 Ubuntu 20.04 - XGBoost</title>](https://discuss.xgboost.ai/t/re-xgboost-1-5-ubuntu-20-04/2387/5)

### [<title>Re: XGBoost 1.5 Ubuntu 20.04 - XGBoost</title>](https://discuss.xgboost.ai/t/re-xgboost-1-5-ubuntu-20-04/2387/4)

### [<title>Re: XGBoost 1.5 Ubuntu 20.04 - XGBoost</title>](https://discuss.xgboost.ai/t/re-xgboost-1-5-ubuntu-20-04/2387/3)

### [<title>Re: XGBoost 1.5 Ubuntu 20.04 - XGBoost</title>](https://discuss.xgboost.ai/t/re-xgboost-1-5-ubuntu-20-04/2387/2)

### [<title>Re: XGBoost 1.5 Ubuntu 20.04 - XGBoost</title>](https://discuss.xgboost.ai/t/re-xgboost-1-5-ubuntu-20-04/2387/1)

### [<title>Specifying Grouping in XGBoost R package - XGBoost</title>](https://discuss.xgboost.ai/t/specifying-grouping-in-xgboost-r-package/2389/3)

### [<title>CPP API XGBoosterPredict cost too much time! - XGBoost</title>](https://discuss.xgboost.ai/t/cpp-api-xgboosterpredict-cost-too-much-time/97/9)

### [<title>Scaling xgboost on multiple-GPUs - XGBoost</title>](https://discuss.xgboost.ai/t/scaling-xgboost-on-multiple-gpus/2383/2)

### [<title>Specifying Grouping in XGBoost R package - XGBoost</title>](https://discuss.xgboost.ai/t/specifying-grouping-in-xgboost-r-package/2389/2)

### [<title>Specifying Grouping in XGBoost R package - XGBoost</title>](https://discuss.xgboost.ai/t/specifying-grouping-in-xgboost-r-package/2389/1)