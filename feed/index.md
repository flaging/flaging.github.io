
## 2021-8-6

### [[2108.02253] Two-Chains: High Performance Framework for Function Injection and Execution](http://arxiv.org/abs/2108.02253)


  Some important problems, such as semantic graph analysis, require large-scale
irregular applications composed of many coordinating tasks that operate on a
shared data set so big it has to be stored on many physical devices. In these
cases, it may be more efficient to dynamically choose where code runs as the
applications progresses. Many programming environments provide task migration
or remote function calls, but they have sharp trade-offs between flexible
composition, portability, performance, and code complexity.
We developed Two-Chains, a high performance framework inspired by active
message communication semantics. We use the GNU Binutils, the ELF binary
format, and the RDMA network protocol to provide ultra-low granularity
distributed function composition at runtime in user space at HPC performance
levels using C libraries. Our framework allows the direct injection of function
binaries and data to a remote machine cache using the RDMA network. It
interoperates seamlessly with existing C libraries using standard dynamic
linking and load symbol resolution. We analyze function delivery and execution
on cache stashing-enabled hardware and show that stashing decreases latency,
increases message rates, and improves noise tolerance. This demonstrates one
way this method is suited to increasingly network-oriented hardware
architectures.

    

### [[2108.02270] Implementation of a Multi-Beam MAC Protocol for Multi-Hop Wireless Networks in Riverbed Modeler](http://arxiv.org/abs/2108.02270)


  Recent advances in antenna technology have made the design of multi-beam
antennas (MBA) feasible. Compared to an omni-directional or a single beam
directional antenna, an MBA equipped node can achieve a throughput of up to m
times, by simultaneously communicating on its m non-interfering beams. As a
result, a few multi-beam directional medium access control (MAC) schemes have
been proposed in the literature recently, which are implemented mostly on the
in-house simulation setups in Matlab or C/C++. These implementations make many
assumptions to simplify their design, without a thorough implementation of
other network layers. However, the implementation of a multi-beam MAC scheme on
the well-known discrete event network simulator platforms (such as the Riverbed
Modeler, NS3, QualNet) is challenging as it requires extensive changes and
additions to various source code modules. In fact, the network protocols in
these simulator packages have been mainly designed for omni-directional
communication, and very few implementations of directional MAC and other
network protocols exist in the literature.
This paper presents a framework to implement a multi-beam directional MAC
scheme in multi-hop wireless networks, by using the Wireless Suite of Riverbed
Modeler. The detailed implementation procedures are described for multi-beam
antenna module, multi-beam node model, concurrent packet transmission and
reception, scheduling, collision avoidance, retransmission, and local node
synchronization. These MAC modules and methodology can be very helpful to the
researchers and developers for implementing the single-beam as well as
multi-beam directional MAC and routing protocols in Riverbed Modeler.

    

### [[2108.02281] Context-Aware Environment Monitoring to Support LPWAN-based Battlefield Applications](http://arxiv.org/abs/2108.02281)


  The use of IoT-related technologies is growing in several areas. Applications
of environmental monitoring, logistics, smart cities are examples of
applications that benefit from advances in IoT. In the military context, IoT
applications can support the decision-making process by delivering information
collected directly from the battlefield to Command, Control, Communications,
Computers, Intelligence, Surveillance and Reconnaissance (C4ISR) systems.
Taking the benefit of the installed IoT network in the battlefield, the use of
the data collected by the IoT nodes is a way to improve resiliency and increase
the survivability of networks, as well as to optimize the use of available
resources. Towards improving the communication network present on the
battlefield, this work presents a context-aware environmental monitoring system
that uses real-time battlefield information to increase military networks'
resilience and survivability. The proposed approach is validated by a
proof-of-concept experiment. The obtained results show that the implementation
of this system can improve the communication process even when the network is
exposed to unfavorable climatic factors.

    

### [[2108.02293] IoT Notary: Attestable Sensor Data Capture in IoT Environments](http://arxiv.org/abs/2108.02293)


  Contemporary IoT environments, such as smart buildings, require end-users to
trust data-capturing rules published by the systems. There are several reasons
why such a trust is misplaced -- IoT systems may violate the rules deliberately
or IoT devices may transfer user data to a malicious third-party due to
cyberattacks, leading to the loss of individuals' privacy or service integrity.
To address such concerns, we propose IoT Notary, a framework to ensure trust in
IoT systems and applications. IoT Notary provides secure log sealing on live
sensor data to produce a verifiable `proof-of-integrity,' based on which a
verifier can attest that captured sensor data adheres to the published
data-capturing rules. IoT Notary is an integral part of TIPPERS, a smart space
system that has been deployed at the University of California Irvine to provide
various real-time location-based services on the campus. We present extensive
experiments over realtime WiFi connectivity data to evaluate IoT Notary, and
the results show that IoT Notary imposes nominal overheads. The secure logs
only take 21% more storage, while users can verify their one day's data in less
than two seconds even using a resource-limited device.

    

### [[2108.02458] Proof of convergence of LoRaWAN model](http://arxiv.org/abs/2108.02458)


  In this document, we prove the convergence of the model proposed in [1],
which aims at estimating the LoRaWAN network performance in a single-gateway
scenario. First, we provide an analytical proof of the existence of a fixed
point solution for such a system. Then, we report experimental results, showing
that the system of the two inter-dependent equations provided by the model can
be solved through fixed-point iterations, and that a limited number of
iterations is enough to reach convergence.

    

### [[2108.02495] DRL-based Slice Placement Under Non-Stationary Conditions](http://arxiv.org/abs/2108.02495)


  We consider online learning for optimal network slice placement under the
assumption that slice requests arrive according to a non-stationary Poisson
process. We propose a framework based on Deep Reinforcement Learning (DRL)
combined with a heuristic to design algorithms. We specifically design two
pure-DRL algorithms and two families of hybrid DRL-heuristic algorithms. To
validate their performance, we perform extensive simulations in the context of
a large-scale operator infrastructure. The evaluation results show that the
proposed hybrid DRL-heuristic algorithms require three orders of magnitude of
learning episodes less than pure-DRL to achieve convergence. This result
indicates that the proposed hybrid DRL-heuristic approach is more reliable than
pure-DRL in a real non-stationary network scenario.

    

### [[2108.02505] On the Robustness of Controlled Deep Reinforcement Learning for Slice Placement](http://arxiv.org/abs/2108.02505)


  The evaluation of the impact of using Machine Learning in the management of
softwarized networks is considered in multiple research works. Beyond that, we
propose to evaluate the robustness of online learning for optimal network slice
placement. A major assumption to this study is to consider that slice request
arrivals are non-stationary. In this context, we simulate unpredictable network
load variations and compare two Deep Reinforcement Learning (DRL) algorithms: a
pure DRL-based algorithm and a heuristically controlled DRL as a hybrid
DRL-heuristic algorithm, to assess the impact of these unpredictable changes of
traffic load on the algorithms performance. We conduct extensive simulations of
a large-scale operator infrastructure. The evaluation results show that the
proposed hybrid DRL-heuristic approach is more robust and reliable in case of
unpredictable network load changes than pure DRL as it reduces the performance
degradation. These results are follow-ups for a series of recent research we
have performed showing that the proposed hybrid DRL-heuristic approach is
efficient and more adapted to real network scenarios than pure DRL.

    

### [[2108.02532] Routing with Face Traversal and Auctions Algorithms for Task Allocation in WSRN](http://arxiv.org/abs/2108.02532)


  Four new algorithms (RFTA1, RFTA2, GFGF2A, and RFTA2GE) handling the event in
wireless sensor and robot networks based on the Greedy-Face-Greedy (GFG)
routing extended with auctions are proposed in this paper. In this paper we
assume that all robots are mobile and after the event is found (reported by
sensors), the goal is to allocate the task to the most suitable robot to act
upon the event, using either distance or the robots' remaining energy as
metrics. The proposed algorithms consist of two phases. The first phase of
algorithms is based on face routing and we introduced the parameter called
search radius (SR) at the end of this first phase. Routing is considered
successful if the found robot is inside SR. After that, the second phase, based
on auctions, is initiated by the robot found in SR trying to find a more
suitable one. In the simulations, network lifetime and communication costs are
measured and used for comparison. We compare our algorithms with two similar
algorithms from the literature (k-SAAP and BFS) used for the task assignment.
RFTA2 and RFTA2GE feature up to 7 times longer network lifetime with
significant communication overhead reduction compared to k-SAAP and BFS. Among
our algorithms, RFTA2GE features the best robot energy utilization.

    

### [[2108.02712] On Addressing Heterogeneity in Federated Learning for Autonomous Vehicles Connected to a Drone Orchestrator](http://arxiv.org/abs/2108.02712)


  In this paper we envision a federated learning (FL) scenario in service of
amending the performance of autonomous road vehicles, through a drone traffic
monitor (DTM), that also acts as an orchestrator. Expecting non-IID data
distribution, we focus on the issue of accelerating the learning of a
particular class of critical object (CO), that may harm the nominal operation
of an autonomous vehicle. This can be done through proper allocation of the
wireless resources for addressing learner and data heterogeneity. Thus, we
propose a reactive method for the allocation of wireless resources, that
happens dynamically each FL round, and is based on each learner's contribution
to the general model. In addition to this, we explore the use of static methods
that remain constant across all rounds. Since we expect partial work from each
learner, we use the FedProx FL algorithm, in the task of computer vision. For
testing, we construct a non-IID data distribution of the MNIST and FMNIST
datasets among four types of learners, in scenarios that represent the quickly
changing environment. The results show that proactive measures are effective
and versatile at improving system accuracy, and quickly learning the CO class
when underrepresented in the network. Furthermore, the experiments show a
tradeoff between FedProx intensity and resource allocation efforts.
Nonetheless, a well adjusted FedProx local optimizer allows for an even better
overall accuracy, particularly when using deeper neural network (NN)
implementations.

    

### [[2108.02716] Link Quality-Guaranteed Minimum-Cost Millimeter-Wave Base Station Deployment](http://arxiv.org/abs/2108.02716)


  Today's growth in the volume of wireless devices coupled with the promise of
supporting data-intensive 5G-&-beyond use cases is driving the industry to
deploy more millimeter-wave (mmWave) base stations (BSs). Although mmWave
cellular systems can carry a larger volume of traffic, dense deployment, in
turn, increases the BS installation and maintenance cost, which has been
largely ignored in their utilization. In this paper, we present an approach to
the problem of mmWave BS deployment in urban environments by minimizing BS
deployment cost subject to BS association and user equipment (UE) outage
constraints. By exploiting the macro diversity, which enables each UE to be
associated with multiple BSs, we derive an expression for UE outage that
integrates physical blockage, UE access-limited blockage, and
signal-to-interference-plus-noise-ratio (SINR) outage into its expression. The
minimum-cost BS deployment problem is then formulated as integer non-linear
programming (INP). The combinatorial nature of the problem motivates the
pursuit of the optimal solution by decomposing the original problem into the
two separable subproblems, i.e., cell coverage optimization and minimum subset
selection subproblems. We provide the optimal solution and theoretical
justifications for each subproblem. The simulation results demonstrating UE
outage guarantees of the proposed method are presented. Interestingly, the
proposed method produces a unique distribution of the macro-diversity orders
over the network that is distinct from other benchmarks.

    

### [[1909.08074] MER-SDN: Machine Learning Framework for Traffic Aware Energy Efficient Routing in SDN](http://arxiv.org/abs/1909.08074)


  Software Defined Networking (SDN) achieves programmability of a network
through separation of the control and data planes. It enables flexibility in
network management and control. Energy efficiency is one of the challenging
global problems which has both economic and environmental impact. A massive
amount of information is generated in the controller of an SDN based network.
Machine learning gives the ability to computers to progressively learn from
data without having to write specific instructions. In this work, we propose
MER-SDN: a machine learning framework for traffic-aware energy efficient
routing in SDN. Feature extraction, training, and testing are the three main
stages of the learning machine. Experiments are conducted on Mininet and POX
controller using real-world network topology and dynamic traffic traces from
SNDlib. Results show that our approach achieves more than 65\% feature size
reduction, more than 70% accuracy in parameter prediction of an energy
efficient heuristics algorithm, also our prediction refine heuristics converges
the predicted value to the optimal parameters values with up to 25X speedup as
compared to the brute force method.

    

### [[2006.00944] Health Risks Associated with 5G Exposure: A View from the Communications Engineering Perspective](http://arxiv.org/abs/2006.00944)


  The deployment of 5G wireless communication services requires the
installation of 5G next-generation Node-B Base Stations (gNBs) over the
territory and the wide adoption of 5G User Equipment (UE). In this context, the
population is concerned about the potential health risks associated with the
Radio Frequency (RF) emissions from 5G equipment, with several communities
actively working toward stopping the 5G deployment. To face these concerns, in
this work, we analyze the health risks associated with 5G exposure by adopting
a new and comprehensive viewpoint, based on the communications engineering
perspective. By exploiting our background, we analyze the alleged health
effects of 5G exposure and critically review the latest works that are often
referenced to support the health concerns from 5G. We then precisely examine
the up-to-date metrics, regulations, and assessment of compliance procedures
for 5G exposure, by evaluating the latest guidelines from IEEE, ICNIRP, ITU,
IEC, and FCC, as well as the national regulations in more than 220 countries.
We also thoroughly analyze the main health risks that are frequently associated
with specific 5G features (e.g., MIMO, beamforming, cell densification,
adoption of millimeter waves, and connection of millions of devices). Finally,
we examine the risk mitigation techniques based on communications engineering
that can be implemented to reduce the exposure from 5G gNB and UE. Overall, we
argue that the widely perceived health risks that are attributed to 5G are not
supported by scientific evidence from communications engineering. In addition,
we explain how the solutions to minimize the health risks from 5G are already
mature and ready to be implemented. Finally, future works, e.g., aimed at
evaluating long-term impacts of 5G exposure, as well as innovative solutions to
further reduce the RF emissions, are suggested.

    

### [[2106.06933] Active Learning for Network Traffic Classification: A Technical Study](http://arxiv.org/abs/2106.06933)


  Network Traffic Classification (NTC) has become an important feature in
various network management operations, e.g., Quality of Service (QoS)
provisioning and security services. Machine Learning (ML) algorithms as a
popular approach for NTC can promise reasonable accuracy in classification and
deal with encrypted traffic. However, ML-based NTC techniques suffer from the
shortage of labeled traffic data which is the case in many real-world
applications. This study investigates the applicability of an active form of
ML, called Active Learning (AL), in NTC. AL reduces the need for a large number
of labeled examples by actively choosing the instances that should be labeled.
The study first provides an overview of NTC and its fundamental challenges
along with surveying the literature on ML-based NTC methods. Then, it
introduces the concepts of AL, discusses it in the context of NTC, and review
the literature in this field. Further, challenges and open issues in AL-based
classification of network traffic are discussed. Moreover, as a technical
survey, some experiments are conducted to show the broad applicability of AL in
NTC. The simulation results show that AL can achieve high accuracy with a small
amount of data.

    

### [[2108.02221] Deep multi-task mining Calabi-Yau four-folds](http://arxiv.org/abs/2108.02221)


  We continue earlier efforts in computing the dimensions of tangent space
cohomologies of Calabi-Yau manifolds using deep learning. In this paper, we
consider the dataset of all Calabi-Yau four-folds constructed as complete
intersections in products of projective spaces. Employing neural networks
inspired by state-of-the-art computer vision architectures, we improve earlier
benchmarks and demonstrate that all four non-trivial Hodge numbers can be
learned at the same time using a multi-task architecture. With 30% (80%)
training ratio, we reach an accuracy of 100% for $h^{(1,1)}$ and 97% for
$h^{(2,1)}$ (100% for both), 81% (96%) for $h^{(3,1)}$, and 49% (83%) for
$h^{(2,2)}$. Assuming that the Euler number is known, as it is easy to compute,
and taking into account the linear constraint arising from index computations,
we get 100% total accuracy.

    

### [[2108.02223] Adversarial learning of cancer tissue representations](http://arxiv.org/abs/2108.02223)


  Deep learning based analysis of histopathology images shows promise in
advancing the understanding of tumor progression, tumor micro-environment, and
their underpinning biological processes. So far, these approaches have focused
on extracting information associated with annotations. In this work, we ask how
much information can be learned from the tissue architecture itself.
We present an adversarial learning model to extract feature representations
of cancer tissue, without the need for manual annotations. We show that these
representations are able to identify a variety of morphological characteristics
across three cancer types: Breast, colon, and lung. This is supported by 1) the
separation of morphologic characteristics in the latent space; 2) the ability
to classify tissue type with logistic regression using latent representations,
with an AUC of 0.97 and 85% accuracy, comparable to supervised deep models; 3)
the ability to predict the presence of tumor in Whole Slide Images (WSIs) using
multiple instance learning (MIL), achieving an AUC of 0.98 and 94% accuracy.
Our results show that our model captures distinct phenotypic characteristics
of real tissue samples, paving the way for further understanding of tumor
progression and tumor micro-environment, and ultimately refining
histopathological classification for diagnosis and treatment. The code and
pretrained models are available at:
this https URL


### [[2108.02231] Growing an architecture for a neural network](http://arxiv.org/abs/2108.02231)


  We propose a new kind of automatic architecture search algorithm. The
algorithm alternates pruning connections and adding neurons, and it is not
restricted to layered architectures only. Here architecture is an arbitrary
oriented graph with some weights (along with some biases and an activation
function), so there may be no layered structure in such a network. The
algorithm minimizes the complexity of staying within a given error. We
demonstrate our algorithm on the brightness prediction problem of the next
point through the previous points on an image. Our second test problem is the
approximation of the bivariate function defining the brightness of a black and
white image. Our optimized networks significantly outperform the standard
solution for neural network architectures in both cases.

    

### [[2108.02233] Unsupervised Detection of Lung Nodules in Chest Radiography Using Generative Adversarial Networks](http://arxiv.org/abs/2108.02233)


  Lung nodules are commonly missed in chest radiographs. We propose and
evaluate P-AnoGAN, an unsupervised anomaly detection approach for lung nodules
in radiographs. P-AnoGAN modifies the fast anomaly detection generative
adversarial network (f-AnoGAN) by utilizing a progressive GAN and a
convolutional encoder-decoder-encoder pipeline. Model training uses only
unlabelled healthy lung patches extracted from the Indiana University Chest
X-Ray Collection. External validation and testing are performed using healthy
and unhealthy patches extracted from the ChestX-ray14 and Japanese Society for
Radiological Technology datasets, respectively. Our model robustly identifies
patches containing lung nodules in external validation and test data with
ROC-AUC of 91.17% and 87.89%, respectively. These results show unsupervised
methods may be useful in challenging tasks such as lung nodule detection in
radiographs.

    

### [[2108.02235] Dynamic Relevance Learning for Few-Shot Object Detection](http://arxiv.org/abs/2108.02235)


  Expensive bounding-box annotations have limited the development of object
detection task. Thus, it is necessary to focus on more challenging task of
few-shot object detection. It requires the detector to recognize objects of
novel classes with only a few training samples. Nowadays, many existing popular
methods based on meta-learning have achieved promising performance, such as
Meta R-CNN series. However, only a single category of support data is used as
the attention to guide the detecting of query images each time. Their relevance
to each other remains unexploited. Moreover, a lot of recent works treat the
support data and query images as independent branch without considering the
relationship between them. To address this issue, we propose a dynamic
relevance learning model, which utilizes the relationship between all support
images and Region of Interest (RoI) on the query images to construct a dynamic
graph convolutional network (GCN). By adjusting the prediction distribution of
the base detector using the output of this GCN, the proposed model can guide
the detector to improve the class representation implicitly. Comprehensive
experiments have been conducted on Pascal VOC and MS-COCO dataset. The proposed
model achieves the best overall performance, which shows its effectiveness of
learning more generalized features. Our code is available at
this https URL.

    

### [[2108.02241] Attentive Cross-modal Connections for Deep Multimodal Wearable-based Emotion Recognition](http://arxiv.org/abs/2108.02241)


  Classification of human emotions can play an essential role in the design and
improvement of human-machine systems. While individual biological signals such
as Electrocardiogram (ECG) and Electrodermal Activity (EDA) have been widely
used for emotion recognition with machine learning methods, multimodal
approaches generally fuse extracted features or final classification/regression
results to boost performance. To enhance multimodal learning, we present a
novel attentive cross-modal connection to share information between
convolutional neural networks responsible for learning individual modalities.
Specifically, these connections improve emotion classification by sharing
intermediate representations among EDA and ECG and apply attention weights to
the shared information, thus learning more effective multimodal embeddings. We
perform experiments on the WESAD dataset to identify the best configuration of
the proposed method for emotion classification. Our experiments show that the
proposed approach is capable of learning strong multimodal representations and
outperforms a number of baselines methods.

    

### [[2108.02283] Machine Learning Classification Methods and Portfolio Allocation: An Examination of Market Efficiency](http://arxiv.org/abs/2108.02283)


  We design a novel framework to examine market efficiency through
out-of-sample (OOS) predictability. We frame the asset pricing problem as a
machine learning classification problem and construct classification models to
predict return states. The prediction-based portfolios beat the market with
significant OOS economic gains. We measure prediction accuracies directly. For
each model, we introduce a novel application of binomial test to test the
accuracy of 3.34 million return state predictions. The tests show that our
models can extract useful contents from historical information to predict
future return states. We provide unique economic insights about OOS
predictability and machine learning models.

    

### [[2108.02289] High dimensional Bayesian Optimization Algorithm for Complex System in Time Series](http://arxiv.org/abs/2108.02289)


  At present, high-dimensional global optimization problems with time-series
models have received much attention from engineering fields. Since it was
proposed, Bayesian optimization has quickly become a popular and promising
approach for solving global optimization problems. However, the standard
Bayesian optimization algorithm is insufficient to solving the global optimal
solution when the model is high-dimensional. Hence, this paper presents a novel
high dimensional Bayesian optimization algorithm by considering dimension
reduction and different dimension fill-in strategies. Most existing literature
about Bayesian optimization algorithms did not discuss the sampling strategies
to optimize the acquisition function. This study proposed a new sampling method
based on both the multi-armed bandit and random search methods while optimizing
the acquisition function. Besides, based on the time-dependent or
dimension-dependent characteristics of the model, the proposed algorithm can
reduce the dimension evenly. Then, five different dimension fill-in strategies
were discussed and compared in this study. Finally, to increase the final
accuracy of the optimal solution, the proposed algorithm adds a local search
based on a series of Adam-based steps at the final stage. Our computational
experiments demonstrated that the proposed Bayesian optimization algorithm
could achieve reasonable solutions with excellent performances for high
dimensional global optimization problems with a time-series optimal control
model.

    

### [[2108.02297] Spartus: A 9.4 TOp/s FPGA-based LSTM Accelerator Exploiting Spatio-temporal Sparsity](http://arxiv.org/abs/2108.02297)


  Long Short-Term Memory (LSTM) recurrent networks are frequently used for
tasks involving time sequential data such as speech recognition. However, it is
difficult to deploy these networks on hardware to achieve high throughput and
low latency because the fully-connected structure makes LSTM networks a
memory-bounded algorithm. Previous work in LSTM accelerators either exploited
weight spatial sparsity or temporal sparsity. In this paper, we present a new
accelerator called "Spartus" that exploits spatio-temporal sparsity to achieve
ultra-low latency inference. The spatial sparsity was induced using our
proposed pruning method called Column-Balanced Targeted Dropout (CBTD) that
leads to structured sparse weight matrices benefiting workload balance. It
achieved up to 96% weight sparsity with negligible accuracy difference for an
LSTM network trained on a TIMIT phone recognition task. To induce temporal
sparsity in LSTM, we create the DeltaLSTM by extending the previous DeltaGRU
method to the LSTM network. This combined sparsity saves on weight memory
access and associated arithmetic operations simultaneously. Spartus was
implemented on a Xilinx Zynq-7100 FPGA. The per-sample latency for a single
DeltaLSTM layer of 1024 neurons running on Spartus is 1 us. Spartus achieved
9.4 TOp/s effective batch-1 throughput and 1.1 TOp/J energy efficiency, which
are respectively 4X and 7X higher than the previous state-of-the-art.

    

### [[2108.02307] Regret Analysis of Learning-Based MPC with Partially-Unknown Cost Function](http://arxiv.org/abs/2108.02307)


  The exploration/exploitation trade-off is an inherent challenge in
data-driven and adaptive control. Though this trade-off has been studied for
multi-armed bandits, reinforcement learning (RL) for finite Markov chains, and
RL for linear control systems; it is less well-studied for learning-based
control of nonlinear control systems. A significant theoretical challenge in
the nonlinear setting is that, unlike the linear case, there is no explicit
characterization of an optimal controller for a given set of cost and system
parameters. We propose in this paper the use of a finite-horizon oracle
controller with perfect knowledge of all system parameters as a reference for
optimal control actions. First, this allows us to propose a new regret notion
with respect to this oracle finite-horizon controller. Second, this allows us
to develop learning-based policies that we prove achieve low regret (i.e.,
square-root regret up to a log-squared factor) with respect to this oracle
finite-horizon controller. This policy is developed in the context of
learning-based model predictive control (LBMPC). We conduct a statistical
analysis to prove finite sample concentration bounds for the estimation step of
our policy, and then we perform a control-theoretic analysis using techniques
from MPC- and optimization-theory to show this policy ensures closed-loop
stability and achieves low regret. We conclude with numerical experiments on a
model of heating, ventilation, and air-conditioning (HVAC) systems that show
the low regret of our policy in a setting where the cost function is
partially-unknown to the controller.

    

### [[2108.02313] BEANNA: A Binary-Enabled Architecture for Neural Network Acceleration](http://arxiv.org/abs/2108.02313)


  Modern hardware design trends have shifted towards specialized hardware
acceleration for computationally intensive tasks like machine learning and
computer vision. While these complex workloads can be accelerated by commercial
GPUs, domain-specific hardware is far more optimal when needing to meet the
stringent memory, throughput, and power constraints of mobile and embedded
devices. This paper proposes and evaluates a Binary-Enabled Architecture for
Neural Network Acceleration (BEANNA), a neural network hardware accelerator
capable of processing both floating point and binary network layers. Through
the use of a novel 16x16 systolic array based matrix multiplier with processing
elements that compute both floating point and binary multiply-adds, BEANNA
seamlessly switches between high precision floating point and binary neural
network layers. Running at a clock speed of 100MHz, BEANNA achieves a peak
throughput of 52.8 GigaOps/second when operating in high precision mode, and
820 GigaOps/second when operating in binary mode. Evaluation of BEANNA was
performed by comparing a hybrid network with floating point outer layers and
binary hidden layers to a network with only floating point layers. The hybrid
network accelerated using BEANNA achieved a 194% throughput increase, a 68%
memory usage decrease, and a 66% energy consumption decrease per inference, all
this at the cost of a mere 0.23% classification accuracy decrease on the MNIST
dataset.

    

### [[2108.02316] Deep Stable neural networks: large-width asymptotics and convergence rates](http://arxiv.org/abs/2108.02316)


  In modern deep learning, there is a recent and growing literature on the
interplay between large-width asymptotics for deep Gaussian neural networks
(NNs), i.e. deep NNs with Gaussian-distributed weights, and classes of Gaussian
stochastic processes (SPs). Such an interplay has proved to be critical in
several contexts of practical interest, e.g. Bayesian inference under Gaussian
SP priors, kernel regression for infinite-wide deep NNs trained via gradient
descent, and information propagation within infinite-wide NNs. Motivated by
empirical analysis, showing the potential of replacing Gaussian distributions
with Stable distributions for the NN's weights, in this paper we investigate
large-width asymptotics for (fully connected) feed-forward deep Stable NNs,
i.e. deep NNs with Stable-distributed weights. First, we show that as the width
goes to infinity jointly over the NN's layers, a suitable rescaled deep Stable
NN converges weakly to a Stable SP whose distribution is characterized
recursively through the NN's layers. Because of the non-triangular NN's
structure, this is a non-standard asymptotic problem, to which we propose a
novel and self-contained inductive approach, which may be of independent
interest. Then, we establish sup-norm convergence rates of a deep Stable NN to
a Stable SP, quantifying the critical difference between the settings of
``joint growth" and ``sequential growth" of the width over the NN's layers. Our
work extends recent results on infinite-wide limits for deep Gaussian NNs to
the more general deep Stable NNs, providing the first result on convergence
rates for infinite-wide deep NNs.

    

### [[2108.02318] Forecasting the outcome of spintronic experiments with Neural Ordinary Differential Equations](http://arxiv.org/abs/2108.02318)


  Deep learning has an increasing impact to assist research, allowing, for
example, the discovery of novel materials. Until now, however, these artificial
intelligence techniques have fallen short of discovering the full differential
equation of an experimental physical system. Here we show that a dynamical
neural network, trained on a minimal amount of data, can predict the behavior
of spintronic devices with high accuracy and an extremely efficient simulation
time, compared to the micromagnetic simulations that are usually employed to
model them. For this purpose, we re-frame the formalism of Neural Ordinary
Differential Equations (ODEs) to the constraints of spintronics: few measured
outputs, multiple inputs and internal parameters. We demonstrate with
Spin-Neural ODEs an acceleration factor over 200 compared to micromagnetic
simulations for a complex problem -- the simulation of a reservoir computer
made of magnetic skyrmions (20 minutes compared to three days). In a second
realization, we show that we can predict the noisy response of experimental
spintronic nano-oscillators to varying inputs after training Spin-Neural ODEs
on five milliseconds of their measured response to different excitations.
Spin-Neural ODE is a disruptive tool for developing spintronic applications in
complement to micromagnetic simulations, which are time-consuming and cannot
fit experiments when noise or imperfections are present. Spin-Neural ODE can
also be generalized to other electronic devices involving dynamics.

    

### [[2108.02319] Generalization in Multimodal Language Learning from Simulation](http://arxiv.org/abs/2108.02319)


  Neural networks can be powerful function approximators, which are able to
model high-dimensional feature distributions from a subset of examples drawn
from the target distribution. Naturally, they perform well at generalizing
within the limits of their target function, but they often fail to generalize
outside of the explicitly learned feature space. It is therefore an open
research topic whether and how neural network-based architectures can be
deployed for systematic reasoning. Many studies have shown evidence for poor
generalization, but they often work with abstract data or are limited to
single-channel input. Humans, however, learn and interact through a combination
of multiple sensory modalities, and rarely rely on just one. To investigate
compositional generalization in a multimodal setting, we generate an extensible
dataset with multimodal input sequences from simulation. We investigate the
influence of the underlying training data distribution on compostional
generalization in a minimal LSTM-based network trained in a supervised, time
continuous setting. We find compositional generalization to fail in simple
setups while improving with the number of objects, actions, and particularly
with a lot of color overlaps between objects. Furthermore, multimodality
strongly improves compositional generalization in settings where a pure vision
model struggles to generalize.

    

### [[2108.02323] Active Reinforcement Learning over MDPs](http://arxiv.org/abs/2108.02323)


  The past decade has seen the rapid development of Reinforcement Learning,
which acquires impressive performance with numerous training resources.
However, one of the greatest challenges in RL is generalization efficiency
(i.e., generalization performance in a unit time). This paper proposes a
framework of Active Reinforcement Learning (ARL) over MDPs to improve
generalization efficiency in a limited resource by instance selection. Given a
number of instances, the algorithm chooses out valuable instances as training
sets while training the policy, thereby costing fewer resources. Unlike
existing approaches, we attempt to actively select and use training data rather
than train on all the given data, thereby costing fewer resources. Furthermore,
we introduce a general instance evaluation metrics and selection mechanism into
the framework. Experiments results reveal that the proposed framework with
Proximal Policy Optimization as policy optimizer can effectively improve
generalization efficiency than unselect-ed and unbiased selected methods.

    

### [[2108.02327] PI3NN: Prediction intervals from three independently trained neural networks](http://arxiv.org/abs/2108.02327)


  We propose a novel prediction interval method to learn prediction mean
values, lower and upper bounds of prediction intervals from three independently
trained neural networks only using the standard mean squared error (MSE) loss,
for uncertainty quantification in regression tasks. Our method requires no
distributional assumption on data, does not introduce unusual hyperparameters
to either the neural network models or the loss function. Moreover, our method
can effectively identify out-of-distribution samples and reasonably quantify
their uncertainty. Numerical experiments on benchmark regression problems show
that our method outperforms the state-of-the-art methods with respect to
predictive uncertainty quality, robustness, and identification of
out-of-distribution samples.

    

### [[2108.02335] Advances in Trajectory Optimization for Space Vehicle Control](http://arxiv.org/abs/2108.02335)


  Space mission design places a premium on cost and operational efficiency. The
search for new science and life beyond Earth calls for spacecraft that can
deliver scientific payloads to geologically rich yet hazardous landing sites.
At the same time, the last four decades of optimization research have put a
suite of powerful optimization tools at the fingertips of the controls
engineer. As we enter the new decade, optimization theory, algorithms, and
software tooling have reached a critical mass to start seeing serious
application in space vehicle guidance and control systems. This survey paper
provides a detailed overview of recent advances, successes, and promising
directions for optimization-based space vehicle control. The considered
applications include planetary landing, rendezvous and proximity operations,
small body landing, constrained reorientation, endo-atmospheric flight
including ascent and re-entry, and orbit transfer and injection. The primary
focus is on the last ten years of progress, which have seen a veritable rise in
the number of applications using three core technologies: lossless
convexification, sequential convex programming, and model predictive control.
The reader will come away with a well-rounded understanding of the
state-of-the-art in each space vehicle control application, and will be well
positioned to tackle important current open problems using convex optimization
as a core technology.

    

### [[2108.02347] FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention](http://arxiv.org/abs/2108.02347)


  We propose FMMformers, a class of efficient and flexible transformers
inspired by the celebrated fast multipole method (FMM) for accelerating
interacting particle simulation. FMM decomposes particle-particle interaction
into near-field and far-field components and then performs direct and
coarse-grained computation, respectively. Similarly, FMMformers decompose the
attention into near-field and far-field attention, modeling the near-field
attention by a banded matrix and the far-field attention by a low-rank matrix.
Computing the attention matrix for FMMformers requires linear complexity in
computational time and memory footprint with respect to the sequence length. In
contrast, standard transformers suffer from quadratic complexity. We analyze
and validate the advantage of FMMformers over the standard transformer on the
Long Range Arena and language modeling benchmarks. FMMformers can even
outperform the standard transformer in terms of accuracy by a significant
margin. For instance, FMMformers achieve an average classification accuracy of
$60.74\%$ over the five Long Range Arena tasks, which is significantly better
than the standard transformer's average accuracy of $58.70\%$.

    

### [[2108.02370] Spotify Danceability and Popularity Analysis using SAP](http://arxiv.org/abs/2108.02370)


  Our analysis reviews and visualizes the audio features and popularity of
songs streamed on Spotify*. Our dataset, downloaded from Kaggle and originally
sourced from Spotify API, consists of multiple Excel files containing
information relevant to our visualization and regression analysis. The exercise
seeks to determine the connection between the popularity of the songs and the
danceability. Insights to be included and factored as part of our analysis
include song energy, valence, BPM, release date, and year.

    

### [[2108.02390] Fuzzy Logic based Logical Query Answering on Knowledge Graph](http://arxiv.org/abs/2108.02390)


  Answering complex First-Order Logical (FOL) queries on large-scale incomplete
knowledge graphs (KGs) is an important yet challenging task. Recent advances
embed logical queries and KG entities in the vector space and conduct query
answering via dense similarity search. However, most of the designed logical
operators in existing works do not satisfy the axiomatic system of classical
logic. Moreover, these logical operators are parameterized so that they require
a large number of complex FOL queries as training data, which are often arduous
or even inaccessible to collect in most real-world KGs. In this paper, we
present FuzzQE, a fuzzy logic based query embedding framework for answering FOL
queries over KGs. FuzzQE follows fuzzy logic to define logical operators in a
principled and learning free manner. Extensive experiments on two benchmark
datasets demonstrate that FuzzQE achieves significantly better performance in
answering FOL queries compared to the state-of-the-art methods. In addition,
FuzzQE trained with only KG link prediction without any complex queries can
achieve comparable performance with the systems trained with all FOL queries.

    

### [[2108.02391] Adapting to Function Difficulty and Growth Conditions in Private Optimization](http://arxiv.org/abs/2108.02391)


  We develop algorithms for private stochastic convex optimization that adapt
to the hardness of the specific function we wish to optimize. While previous
work provide worst-case bounds for arbitrary convex functions, it is often the
case that the function at hand belongs to a smaller class that enjoys faster
rates. Concretely, we show that for functions exhibiting $\kappa$-growth around
the optimum, i.e., $f(x) \ge f(x^*) + \lambda \kappa^{-1} \|x-x^*\|_2^\kappa$
for $\kappa > 1$, our algorithms improve upon the standard
${\sqrt{d}}/{n\varepsilon}$ privacy rate to the faster
$({\sqrt{d}}/{n\varepsilon})^{\tfrac{\kappa}{\kappa - 1}}$. Crucially, they
achieve these rates without knowledge of the growth constant $\kappa$ of the
function. Our algorithms build upon the inverse sensitivity mechanism, which
adapts to instance difficulty (Asi & Duchi, 2020), and recent localization
techniques in private optimization (Feldman et al., 2020). We complement our
algorithms with matching lower bounds for these function classes and
demonstrate that our adaptive algorithm is \emph{simultaneously} (minimax)
optimal over all $\kappa \ge 1+c$ whenever $c = \Theta(1)$.

    

### [[2108.02393] Online Model-Free Reinforcement Learning for the Automatic Control of a Flexible Wing Aircraft](http://arxiv.org/abs/2108.02393)


  The control problem of the flexible wing aircraft is challenging due to the
prevailing and high nonlinear deformations in the flexible wing system. This
urged for new control mechanisms that are robust to the real-time variations in
the wing's aerodynamics. An online control mechanism based on a value iteration
reinforcement learning process is developed for flexible wing aerial
structures. It employs a model-free control policy framework and a guaranteed
convergent adaptive learning architecture to solve the system's Bellman
optimality equation. A Riccati equation is derived and shown to be equivalent
to solving the underlying Bellman equation. The online reinforcement learning
solution is implemented using means of an adaptive-critic mechanism. The
controller is proven to be asymptotically stable in the Lyapunov sense. It is
assessed through computer simulations and its superior performance is
demonstrated on two scenarios under different operating conditions.

    

### [[2108.02416] Aspis: A Robust Detection System for Distributed Learning](http://arxiv.org/abs/2108.02416)


  State of the art machine learning models are routinely trained on large scale
distributed clusters. Crucially, such systems can be compromised when some of
the computing devices exhibit abnormal (Byzantine) behavior and return
arbitrary results to the parameter server (PS). This behavior may be attributed
to a plethora of reasons including system failures and orchestrated attacks.
Existing work suggests robust aggregation and/or computational redundancy to
alleviate the effect of distorted gradients. However, most of these schemes are
ineffective when an adversary knows the task assignment and can judiciously
choose the attacked workers to induce maximal damage. Our proposed method Aspis
assigns gradient computations to worker nodes using a subset-based assignment
which allows for multiple consistency checks on the behavior of a worker node.
Examination of the calculated gradients and post-processing (clique-finding in
an appropriately constructed graph) by the central node allows for efficient
detection and subsequent exclusion of adversaries from the training process. We
prove the Byzantine resilience and detection guarantees of Aspis under weak and
strong attacks and extensively evaluate the system on various large-scale
training scenarios. The main metric for our experiments is the test accuracy
for which we demonstrate significant improvement of about 30% compared to many
state-of-the-art approaches on the CIFAR-10 dataset. The corresponding
reduction of the fraction of corrupted gradients ranges from 16% to 98%.

    

### [[2108.02424] PSTN: Periodic Spatial-temporal Deep Neural Network for Traffic Condition Prediction](http://arxiv.org/abs/2108.02424)


  Accurate forecasting of traffic conditions is critical for improving safety,
stability, and efficiency of a city transportation system. In reality, it is
challenging to produce accurate traffic forecasts due to the complex and
dynamic spatiotemporal correlations. Most existing works only consider partial
characteristics and features of traffic data, and result in unsatisfactory
performances on modeling and forecasting. In this paper, we propose a periodic
spatial-temporal deep neural network (PSTN) with three pivotal modules to
improve the forecasting performance of traffic conditions through a novel
integration of three types of information. First, the historical traffic
information is folded and fed into a module consisting of a graph convolutional
network and a temporal convolutional network. Second, the recent traffic
information together with the historical output passes through the second
module consisting of a graph convolutional network and a gated recurrent unit
framework. Finally, a multi-layer perceptron is applied to process the
auxiliary road attributes and output the final predictions. Experimental
results on two publicly accessible real-world urban traffic data sets show that
the proposed PSTN outperforms the state-of-the-art benchmarks by significant
margins for short-term traffic conditions forecasting

    

### [[2108.02428] A Method for Medical Data Analysis Using the LogNNet for Clinical Decision Support Systems and Edge Computing in Healthcare](http://arxiv.org/abs/2108.02428)


  The study presents a new method for analyzing medical data based on the
LogNNet neural network, which uses chaotic mappings to transform input
information. The technique calculates risk factors for the presence of a
disease in a patient according to a set of medical health indicators. The
LogNNet architecture allows the implementation of artificial intelligence on
medical pe-ripherals of the Internet of Things with low RAM resources, and the
development of edge computing in healthcare. The efficiency of LogNNet in
assessing perinatal risk is illustrated on cardiotocogram data of 2126 pregnant
women, obtained from the UC Irvine machine learning repository. The
classification accuracy reaches ~ 91%, with the ~ 3-10 kB of RAM used on the
Arduino microcontroller. In addition, examples for diagnosing COVID-19 are
provided, using LogNNet trained on a publicly available database from the
Israeli Ministry of Health. The service concept has been developed, which uses
the data of the express test for COVID-19 and reaches the classification
accuracy of ~ 95% with the ~ 0.6 kB of RAM used on Arduino microcontrollers. In
all examples, the model is tested using standard classification quality
metrics: Precision, Recall, and F1-measure. The study results can be used in
clinical decision support systems.

    

### [[2108.02430] Deep Neural Networks and PIDE discretizations](http://arxiv.org/abs/2108.02430)


  In this paper, we propose neural networks that tackle the problems of
stability and field-of-view of a Convolutional Neural Network (CNN). As an
alternative to increasing the network's depth or width to improve performance,
we propose integral-based spatially nonlocal operators which are related to
global weighted Laplacian, fractional Laplacian and inverse fractional
Laplacian operators that arise in several problems in the physical sciences.
The forward propagation of such networks is inspired by partial
integro-differential equations (PIDEs). We test the effectiveness of the
proposed neural architectures on benchmark image classification datasets and
semantic segmentation tasks in autonomous driving. Moreover, we investigate the
extra computational costs of these dense operators and the stability of forward
propagation of the proposed neural networks.

    

### [[2108.02431] AutoLL: Automatic Linear Layout of Graphs based on Deep Neural Network](http://arxiv.org/abs/2108.02431)


  Linear layouts are a graph visualization method that can be used to capture
an entry pattern in an adjacency matrix of a given graph. By reordering the
node indices of the original adjacency matrix, linear layouts provide knowledge
of latent graph structures. Conventional linear layout methods commonly aim to
find an optimal reordering solution based on predefined features of a given
matrix and loss function. However, prior knowledge of the appropriate features
to use or structural patterns in a given adjacency matrix is not always
available. In such a case, performing the reordering based on data-driven
feature extraction without assuming a specific structure in an adjacency matrix
is preferable. Recently, a neural-network-based matrix reordering method called
DeepTMR has been proposed to perform this function. However, it is limited to a
two-mode reordering (i.e., the rows and columns are reordered separately) and
it cannot be applied in the one-mode setting (i.e., the same node order is used
for reordering both rows and columns), owing to the characteristics of its
model architecture. In this study, we extend DeepTMR and propose a new one-mode
linear layout method referred to as AutoLL. We developed two types of neural
network models, AutoLL-D and AutoLL-U, for reordering directed and undirected
networks, respectively. To perform one-mode reordering, these AutoLL models
have specific encoder architectures, which extract node features from an
observed adjacency matrix. We conducted both qualitative and quantitative
evaluations of the proposed approach, and the experimental results demonstrate
its effectiveness.

    

### [[2108.02479] HyperJump: Accelerating HyperBand via Risk Modelling](http://arxiv.org/abs/2108.02479)


  In the literature on hyper-parameter tuning, a number of recent solutions
rely on low-fidelity observations (e.g., training with sub-sampled datasets or
for short periods of time) to extrapolate good configurations to use when
performing full training. Among these, HyperBand is arguably one of the most
popular solutions, due to its efficiency and theoretically provable robustness.
In this work, we introduce HyperJump, a new approach that builds on HyperBand's
robust search strategy and complements it with novel model-based risk analysis
techniques that accelerate the search by jumping the evaluation of low risk
configurations, i.e., configurations that are likely to be discarded by
HyperBand. We evaluate HyperJump on a suite of hyper-parameter optimization
problems and show that it provides over one-order of magnitude speed-ups on a
variety of deep-learning and kernel-based learning problems when compared to
HyperBand as well as to a number of state of the art optimizers.

    

### [[2108.02497] How to avoid machine learning pitfalls: a guide for academic researchers](http://arxiv.org/abs/2108.02497)


  This document gives a concise outline of some of the common mistakes that
occur when using machine learning techniques, and what can be done to avoid
them. It is intended primarily as a guide for research students, and focuses on
issues that are of particular concern within academic research, such as the
need to do rigorous comparisons and reach valid conclusions. It covers five
stages of the machine learning process: what to do before model building, how
to reliably build models, how to robustly evaluate models, how to compare
models fairly, and how to report results.

    

### [[2108.02501] Locally Interpretable One-Class Anomaly Detection for Credit Card Fraud Detection](http://arxiv.org/abs/2108.02501)


  For the highly imbalanced credit card fraud detection problem, most existing
methods either use data augmentation methods or conventional machine learning
models, while neural network-based anomaly detection approaches are lacking.
Furthermore, few studies have employed AI interpretability tools to investigate
the feature importance of transaction data, which is crucial for the black-box
fraud detection module. Considering these two points together, we propose a
novel anomaly detection framework for credit card fraud detection as well as a
model-explaining module responsible for prediction explanations. The fraud
detection model is composed of two deep neural networks, which are trained in
an unsupervised and adversarial manner. Precisely, the generator is an
AutoEncoder aiming to reconstruct genuine transaction data, while the
discriminator is a fully-connected network for fraud detection. The explanation
module has three white-box explainers in charge of interpretations of the
AutoEncoder, discriminator, and the whole detection model, respectively.
Experimental results show the state-of-the-art performances of our fraud
detection model on the benchmark dataset compared with baselines. In addition,
prediction analyses by three explainers are presented, offering a clear
perspective on how each feature of an instance of interest contributes to the
final model output.

    

### [[2108.02507] Shape Modeling with Spline Partitions](http://arxiv.org/abs/2108.02507)


  Shape modelling (with methods that output shapes) is a new and important task
in Bayesian nonparametrics and bioinformatics. In this work, we focus on
Bayesian nonparametric methods for capturing shapes by partitioning a space
using curves. In related work, the classical Mondrian process is used to
partition spaces recursively with axis-aligned cuts, and is widely applied in
multi-dimensional and relational data. The Mondrian process outputs
hyper-rectangles. Recently, the random tessellation process was introduced as a
generalization of the Mondrian process, partitioning a domain with non-axis
aligned cuts in an arbitrary dimensional space, and outputting polytopes.
Motivated by these processes, in this work, we propose a novel parallelized
Bayesian nonparametric approach to partition a domain with curves, enabling
complex data-shapes to be acquired. We apply our method to HIV-1-infected human
macrophage image dataset, and also simulated datasets sets to illustrate our
approach. We compare to support vector machines, random forests and
state-of-the-art computer vision methods such as simple linear iterative
clustering super pixel image segmentation. We develop an R package that is
available at
\url{this https URL}.

    

### [[2108.02517] Multi-task Federated Edge Learning (MtFEEL) in Wireless Networks](http://arxiv.org/abs/2108.02517)


  Federated Learning (FL) has evolved as a promising technique to handle
distributed machine learning across edge devices. A single neural network (NN)
that optimises a global objective is generally learned in most work in FL,
which could be suboptimal for edge devices. Although works finding a NN
personalised for edge device specific tasks exist, they lack generalisation
and/or convergence guarantees. In this paper, a novel communication efficient
FL algorithm for personalised learning in a wireless setting with guarantees is
presented. The algorithm relies on finding a ``better`` empirical estimate of
losses at each device, using a weighted average of the losses across different
devices. It is devised from a Probably Approximately Correct (PAC) bound on the
true loss in terms of the proposed empirical loss and is bounded by (i) the
Rademacher complexity, (ii) the discrepancy, (iii) and a penalty term. Using a
signed gradient feedback to find a personalised NN at each device, it is also
proven to converge in a Rayleigh flat fading (in the uplink) channel, at a rate
of the order max{1/SNR,1/sqrt(T)} Experimental results show that the proposed
algorithm outperforms locally trained devices as well as the conventionally
used FedAvg and FedSGD algorithms under practical SNR regimes.

    

### [[2108.02537] Redatuming physical systems using symmetric autoencoders](http://arxiv.org/abs/2108.02537)


  This paper considers physical systems described by hidden states and
indirectly observed through repeated measurements corrupted by unmodeled
nuisance parameters. A network-based representation learns to disentangle the
coherent information (relative to the state) from the incoherent nuisance
information (relative to the sensing). Instead of physical models, the
representation uses symmetry and stochastic regularization to inform an
autoencoder architecture called SymAE. It enables redatuming, i.e., creating
virtual data instances where the nuisances are uniformized across measurements.

    

### [[2108.02550] VBridge: Connecting the Dots Between Features, Explanations, and Data for Healthcare Models](http://arxiv.org/abs/2108.02550)


  Machine learning (ML) is increasingly applied to Electronic Health Records
(EHRs) to solve clinical prediction tasks. Although many ML models perform
promisingly, issues with model transparency and interpretability limit their
adoption in clinical practice. Directly using existing explainable ML
techniques in clinical settings can be challenging. Through literature surveys
and collaborations with six clinicians with an average of 17 years of clinical
experience, we identified three key challenges, including clinicians'
unfamiliarity with ML features, lack of contextual information, and the need
for cohort-level evidence. Following an iterative design process, we further
designed and developed VBridge, a visual analytics tool that seamlessly
incorporates ML explanations into clinicians' decision-making workflow. The
system includes a novel hierarchical display of contribution-based feature
explanations and enriched interactions that connect the dots between ML
features, explanations, and data. We demonstrated the effectiveness of VBridge
through two case studies and expert interviews with four clinicians, showing
that visually associating model explanations with patients' situational records
can help clinicians better interpret and use model predictions when making
clinician decisions. We further derived a list of design implications for
developing future explainable ML tools to support clinical decision-making.

    

### [[2108.02551] Ensemble Consensus-based Representation Deep Reinforcement Learning for Hybrid FSO/RF Communication Systems](http://arxiv.org/abs/2108.02551)


  Hybrid FSO/RF system requires an efficient FSO and RF link switching
mechanism to improve the system capacity by realizing the complementary
benefits of both the links. The dynamics of network conditions, such as fog,
dust, and sand storms compound the link switching problem and control
complexity. To address this problem, we initiate the study of deep
reinforcement learning (DRL) for link switching of hybrid FSO/RF systems.
Specifically, in this work, we focus on actor-critic called Actor/Critic-FSO/RF
and Deep-Q network (DQN) called DQN-FSO/RF for FSO/RF link switching under
atmospheric turbulences. To formulate the problem, we define the state, action,
and reward function of a hybrid FSO/RF system. DQN-FSO/RF frequently updates
the deployed policy that interacts with the environment in a hybrid FSO/RF
system, resulting in high switching costs. To overcome this, we lift this
problem to ensemble consensus-based representation learning for deep
reinforcement called DQNEnsemble-FSO/RF. The proposed novel DQNEnsemble-FSO/RF
DRL approach uses consensus learned features representations based on an
ensemble of asynchronous threads to update the deployed policy. Experimental
results corroborate that the proposed DQNEnsemble-FSO/RF's consensus-learned
features switching achieves better performance than Actor/Critic-FSO/RF,
DQN-FSO/RF, and MyOpic for FSO/RF link switching while keeping the switching
cost significantly low.

    

### [[2108.02555] DeepScanner: a Robotic System for Automated 2D Object Dataset Collection with Annotations](http://arxiv.org/abs/2108.02555)


  In the proposed study, we describe the possibility of automated dataset
collection using an articulated robot. The proposed technology reduces the
number of pixel errors on a polygonal dataset and the time spent on manual
labeling of 2D objects. The paper describes a novel automatic dataset
collection and annotation system, and compares the results of automated and
manual dataset labeling. Our approach increases the speed of data labeling
240-fold, and improves the accuracy compared to manual labeling 13-fold. We
also present a comparison of metrics for training a neural network on a
manually annotated and an automatically collected dataset.

    

### [[2108.02562] Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models](http://arxiv.org/abs/2108.02562)


  Systems that can find correspondences between multiple modalities, such as
between speech and images, have great potential to solve different recognition
and data analysis tasks in an unsupervised manner. This work studies multimodal
learning in the context of visually grounded speech (VGS) models, and focuses
on their recently demonstrated capability to extract spatiotemporal alignments
between spoken words and the corresponding visual objects without ever been
explicitly trained for object localization or word recognition. As the main
contributions, we formalize the alignment problem in terms of an audiovisual
alignment tensor that is based on earlier VGS work, introduce systematic
metrics for evaluating model performance in aligning visual objects and spoken
words, and propose a new VGS model variant for the alignment task utilizing
cross-modal attention layer. We test our model and a previously proposed model
in the alignment task using SPEECH-COCO captions coupled with MSCOCO images. We
compare the alignment performance using our proposed evaluation metrics to the
semantic retrieval task commonly used to evaluate VGS models. We show that
cross-modal attention layer not only helps the model to achieve higher semantic
cross-modal retrieval performance, but also leads to substantial improvements
in the alignment performance between image object and spoken words.

    

### [[2108.02563] GuavaNet: A deep neural network architecture for automatic sensory evaluation to predict degree of acceptability for Guava by a consumer](http://arxiv.org/abs/2108.02563)


  This thesis is divided into two parts:Part I: Analysis of Fruits, Vegetables,
Cheese and Fish based on Image Processing using Computer Vision and Deep
Learning: A Review. It consists of a comprehensive review of image processing,
computer vision and deep learning techniques applied to carry out analysis of
fruits, vegetables, cheese and fish.This part also serves as a literature
review for Part II.Part II: GuavaNet: A deep neural network architecture for
automatic sensory evaluation to predict degree of acceptability for Guava by a
consumer. This part introduces to an end-to-end deep neural network
architecture that can predict the degree of acceptability by the consumer for a
guava based on sensory evaluation.

    

### [[2108.02565] Dependable Neural Networks Through Redundancy, A Comparison of Redundant Architectures](http://arxiv.org/abs/2108.02565)


  With edge-AI finding an increasing number of real-world applications,
especially in industry, the question of functionally safe applications using AI
has begun to be asked. In this body of work, we explore the issue of achieving
dependable operation of neural networks. We discuss the issue of dependability
in general implementation terms before examining lockstep solutions. We intuit
that it is not necessarily a given that two similar neural networks generate
results at precisely the same time and that synchronization between the
platforms will be required. We perform some preliminary measurements that may
support this intuition and introduce some work in implementing lockstep neural
network engines.

    

### [[2108.02566] Missingness Augmentation: A General Approach for Improving Generative Imputation Models](http://arxiv.org/abs/2108.02566)


  Despite tremendous progress in missing data imputation task, designing new
imputation models has become more and more cumbersome but the corresponding
gains are relatively small. Is there any simple but general approach that can
exploit the existing models to further improve the quality of the imputation?
In this article, we aim to respond to this concern and propose a novel general
data augmentation method called Missingness Augmentation (MA), which can be
applied in many existing generative imputation frameworks to further improve
the performance of these models. For MA, before each training epoch, we use the
outputs of the generator to expand the incomplete samples on the fly, and then
determine a special reconstruction loss for these augmented samples. This
reconstruction loss plus the original loss constitutes the final optimization
objective of the model. It is noteworthy that MA is very efficient and does not
need to change the structure of the original model. Experimental results
demonstrate that MA can significantly improve the performance of many recently
developed generative imputation models on a variety of datasets. Our code is
available at this https URL.

    

### [[2108.02567] Improving the Performance of a NoC-based CNN Accelerator with Gather Support](http://arxiv.org/abs/2108.02567)


  The increasing application of deep learning technology drives the need for an
efficient parallel computing architecture for Convolutional Neural Networks
(CNNs). A significant challenge faced when designing a many-core CNN
accelerator is to handle the data movement between the processing elements. The
CNN workload introduces many-to-one traffic in addition to one-to-one and
one-to-many traffic. As the de-facto standard for on-chip communication,
Network-on-Chip (NoC) can support various unicast and multicast traffic. For
many-to-one traffic, repetitive unicast is employed which is not an efficient
way. In this paper, we propose to use the gather packet on mesh-based NoCs
employing output stationary systolic array in support of many-to-one traffic.
The gather packet will collect the data from the intermediate nodes eventually
leading to the destination efficiently. This method is evaluated using the
traffic traces generated from the convolution layer of AlexNet and VGG-16 with
improvement in the latency and power than the repetitive unicast method.

    

### [[2108.02569] Data Streaming and Traffic Gathering in Mesh-based NoC for Deep Neural Network Acceleration](http://arxiv.org/abs/2108.02569)


  The increasing popularity of deep neural network (DNN) applications demands
high computing power and efficient hardware accelerator architecture. DNN
accelerators use a large number of processing elements (PEs) and on-chip memory
for storing weights and other parameters. As the communication backbone of a
DNN accelerator, networks-on-chip (NoC) play an important role in supporting
various dataflow patterns and enabling processing with communication
parallelism in a DNN accelerator. However, the widely used mesh-based NoC
architectures inherently cannot support the efficient one-to-many and
many-to-one traffic largely existing in DNN workloads. In this paper, we
propose a modified mesh architecture with a one-way/two-way streaming bus to
speedup one-to-many (multicast) traffic, and the use of gather packets to
support many-to-one (gather) traffic. The analysis of the runtime latency of a
convolutional layer shows that the two-way streaming architecture achieves
better improvement than the one-way streaming architecture for an Output
Stationary (OS) dataflow architecture. The simulation results demonstrate that
the gather packets can help to reduce the runtime latency up to 1.8 times and
network power consumption up to 1.7 times, compared with the repetitive unicast
method on modified mesh architectures supporting two-way streaming.

    

### [[2108.02570] Predicting Post-Concussion Syndrome Outcomes with Machine Learning](http://arxiv.org/abs/2108.02570)


  In this paper, machine learning models are used to predict outcomes for
patients with persistent post-concussion syndrome (PCS). Patients had sustained
a concussion at an average of two to three months before the study. By
utilizing assessed data, the machine learning models aimed to predict whether
or not a patient would continue to have PCS after four to five months. The
random forest classifier achieved the highest performance with an 85% accuracy
and an area under the receiver operating characteristic curve (AUC) of 0.94.
Factors found to be predictive of PCS outcome were Post-Traumatic Stress
Disorder (PTSD), perceived injustice, self-rated prognosis, and symptom
severity post-injury. The results of this study demonstrate that machine
learning models can predict PCS outcomes with high accuracy. With further
research, machine learning models may be implemented in healthcare settings to
help patients with persistent PCS.

    

### [[2108.02571] Learning Linearized Assignment Flows for Image Labeling](http://arxiv.org/abs/2108.02571)


  We introduce a novel algorithm for estimating optimal parameters of
linearized assignment flows for image labeling. An exact formula is derived for
the parameter gradient of any loss function that is constrained by the linear
system of ODEs determining the linearized assignment flow. We show how to
efficiently evaluate this formula using a Krylov subspace and a low-rank
approximation. This enables us to perform parameter learning by Riemannian
gradient descent in the parameter space, without the need to backpropagate
errors or to solve an adjoint equation, in less than 10 seconds for a
$512\times 512$ image using just about $0.5$ GB memory. Experiments demonstrate
that our method performs as good as highly-tuned machine learning software
using automatic differentiation. Unlike methods employing automatic
differentiation, our approach yields a low-dimensional representation of
internal parameters and their dynamics which helps to understand how networks
work and perform that realize assignment flows and generalizations thereof.

    

### [[2108.02572] SINGA-Easy: An Easy-to-Use Framework for MultiModal Analysis](http://arxiv.org/abs/2108.02572)


  Deep learning has achieved great success in a wide spectrum of multimedia
applications such as image classification, natural language processing and
multimodal data analysis. Recent years have seen the development of many deep
learning frameworks that provide a high-level programming interface for users
to design models, conduct training and deploy inference. However, it remains
challenging to build an efficient end-to-end multimedia application with most
existing frameworks. Specifically, in terms of usability, it is demanding for
non-experts to implement deep learning models, obtain the right settings for
the entire machine learning pipeline, manage models and datasets, and exploit
external data sources all together. Further, in terms of adaptability, elastic
computation solutions are much needed as the actual serving workload fluctuates
constantly, and scaling the hardware resources to handle the fluctuating
workload is typically infeasible. To address these challenges, we introduce
SINGA-Easy, a new deep learning framework that provides distributed
hyper-parameter tuning at the training stage, dynamic computational cost
control at the inference stage, and intuitive user interactions with multimedia
contents facilitated by model explanation. Our experiments on the training and
deployment of multi-modality data analysis applications show that the framework
is both usable and adaptable to dynamic inference loads. We implement
SINGA-Easy on top of Apache SINGA and demonstrate our system with the entire
machine learning life cycle.

    

### [[2108.02574] Optimal Transport for Unsupervised Restoration Learning](http://arxiv.org/abs/2108.02574)


  Recently, much progress has been made in unsupervised restoration learning.
However, existing methods more or less rely on some assumptions on the signal
and/or degradation model, which limits their practical performance. How to
construct an optimal criterion for unsupervised restoration learning without
any prior knowledge on the degradation model is still an open question. Toward
answering this question, this work proposes a criterion for unsupervised
restoration learning based on the optimal transport theory. This criterion has
favorable properties, e.g., approximately maximal preservation of the
information of the signal, whilst achieving perceptual reconstruction.
Furthermore, though a relaxed unconstrained formulation is used in practical
implementation, we show that the relaxed formulation in theory has the same
solution as the original constrained formulation. Experiments on synthetic and
real-world data, including realistic photographic, microscopy, depth, and raw
depth images, demonstrate that the proposed method even compares favorably with
supervised methods, e.g., approaching the PSNR of supervised methods while
having better perceptual quality. Particularly, for spatially correlated noise
and realistic microscopy images, the proposed method not only achieves better
perceptual quality but also has higher PSNR than supervised methods. Besides,
it shows remarkable superiority in harsh practical conditions with complex
noise, e.g., raw depth images.

    

### [[2108.02576] Performer Identification From Symbolic Representation of Music Using Statistical Models](http://arxiv.org/abs/2108.02576)


  Music Performers have their own idiosyncratic way of interpreting a musical
piece. A group of skilled performers playing the same piece of music would
likely to inject their unique artistic styles in their performances. The
variations of the tempo, timing, dynamics, articulation etc. from the actual
notated music are what make the performers unique in their performances. This
study presents a dataset consisting of four movements of Schubert's ``Sonata in
B-flat major, D.960" performed by nine virtuoso pianists individually. We
proposed and extracted a set of expressive features that are able to capture
the characteristics of an individual performer's style. We then present a
performer identification method based on the similarity of feature
distribution, given a set of piano performances. The identification is done
considering each feature individually as well as a fusion of the features.
Results show that the proposed method achieved a precision of 0.903 using
fusion features. Moreover, the onset time deviation feature shows promising
result when considered individually.

    

### [[2108.02594] A variational Bayesian spatial interaction model for estimating revenue and demand at business facilities](http://arxiv.org/abs/2108.02594)


  We study the problem of estimating potential revenue or demand at business
facilities and understanding its generating mechanism. This problem arises in
different fields such as operation research or urban science, and more
generally, it is crucial for businesses' planning and decision making. We
develop a Bayesian spatial interaction model, henceforth BSIM, which provides
probabilistic predictions about revenues generated by a particular business
location provided their features and the potential customers' characteristics
in a given region. BSIM explicitly accounts for the competition among the
competitive facilities through a probability value determined by evaluating a
store-specific Gaussian distribution at a given customer location. We propose a
scalable variational inference framework that, while being significantly faster
than competing Markov Chain Monte Carlo inference schemes, exhibits comparable
performances in terms of parameters identification and uncertainty
quantification. We demonstrate the benefits of BSIM in various synthetic
settings characterised by an increasing number of stores and customers.
Finally, we construct a real-world, large spatial dataset for pub activities in
London, UK, which includes over 1,500 pubs and 150,000 customer regions. We
demonstrate how BSIM outperforms competing approaches on this large dataset in
terms of prediction performances while providing results that are both
interpretable and consistent with related indicators observed for the London
region.

    

### [[2108.02606] Self-supervised optimization of random material microstructures in the small-data regime](http://arxiv.org/abs/2108.02606)


  While the forward and backward modeling of the process-structure-property
chain has received a lot of attention from the materials community, fewer
efforts have taken into consideration uncertainties. Those arise from a
multitude of sources and their quantification and integration in the inversion
process are essential in meeting the materials design objectives. The first
contribution of this paper is a flexible, fully probabilistic formulation of
such optimization problems that accounts for the uncertainty in the
process-structure and structure-property linkages and enables the
identification of optimal, high-dimensional, process parameters. We employ a
probabilistic, data-driven surrogate for the structure-property link which
expedites computations and enables handling of non-differential objectives. We
couple this with a novel active learning strategy, i.e. a self-supervised
collection of data, which significantly improves accuracy while requiring small
amounts of training data. We demonstrate its efficacy in optimizing the
mechanical and thermal properties of two-phase, random media but envision its
applicability encompasses a wide variety of microstructure-sensitive design
problems.

    

### [[2108.02628] A New State-of-the-Art Transformers-Based Load Forecaster on the Smart Grid Domain](http://arxiv.org/abs/2108.02628)


  Meter-level load forecasting is crucial for efficient energy management and
power system planning for Smart Grids (SGs), in tasks associated with
regulation, dispatching, scheduling, and unit commitment of power grids.
Although a variety of algorithms have been proposed and applied on the field,
more accurate and robust models are still required: the overall utility cost of
operations in SGs increases 10 million currency units if the load forecasting
error increases 1%, and the mean absolute percentage error (MAPE) in
forecasting is still much higher than 1%. Transformers have become the new
state-of-the-art in a variety of tasks, including the ones in computer vision,
natural language processing and time series forecasting, surpassing alternative
neural models such as convolutional and recurrent neural networks. In this
letter, we present a new state-of-the-art Transformer-based algorithm for the
meter-level load forecasting task, which has surpassed the former
state-of-the-art, LSTM, and the traditional benchmark, vanilla RNN, in all
experiments by a margin of at least 13% in MAPE.

    

### [[2108.02644] Parallel Capsule Networks for Classification of White Blood Cells](http://arxiv.org/abs/2108.02644)


  Capsule Networks (CapsNets) is a machine learning architecture proposed to
overcome some of the shortcomings of convolutional neural networks (CNNs).
However, CapsNets have mainly outperformed CNNs in datasets where images are
small and/or the objects to identify have minimal background noise. In this
work, we present a new architecture, parallel CapsNets, which exploits the
concept of branching the network to isolate certain capsules, allowing each
branch to identify different entities. We applied our concept to the two
current types of CapsNet architectures, studying the performance for networks
with different layers of capsules. We tested our design in a public, highly
unbalanced dataset of acute myeloid leukaemia images (15 classes). Our
experiments showed that conventional CapsNets show similar performance than our
baseline CNN (ResNeXt-50) but depict instability problems. In contrast,
parallel CapsNets can outperform ResNeXt-50, is more stable, and shows better
rotational invariance than both, conventional CapsNets and ResNeXt-50.

    

### [[2108.02646] A Hypothesis for the Aesthetic Appreciation in Neural Networks](http://arxiv.org/abs/2108.02646)


  This paper proposes a hypothesis for the aesthetic appreciation that
aesthetic images make a neural network strengthen salient concepts and discard
inessential concepts. In order to verify this hypothesis, we use multi-variate
interactions to represent salient concepts and inessential concepts contained
in images. Furthermore, we design a set of operations to revise images towards
more beautiful ones. In experiments, we find that the revised images are more
aesthetic than the original ones to some extent.

    

### [[2108.02658] Sparse Communication via Mixed Distributions](http://arxiv.org/abs/2108.02658)


  Neural networks and other machine learning models compute continuous
representations, while humans communicate mostly through discrete symbols.
Reconciling these two forms of communication is desirable for generating
human-readable interpretations or learning discrete latent variable models,
while maintaining end-to-end differentiability. Some existing approaches (such
as the Gumbel-Softmax transformation) build continuous relaxations that are
discrete approximations in the zero-temperature limit, while others (such as
sparsemax transformations and the Hard Concrete distribution) produce
discrete/continuous hybrids. In this paper, we build rigorous theoretical
foundations for these hybrids, which we call "mixed random variables." Our
starting point is a new "direct sum" base measure defined on the face lattice
of the probability simplex. From this measure, we introduce new entropy and
Kullback-Leibler divergence functions that subsume the discrete and
differential cases and have interpretations in terms of code optimality. Our
framework suggests two strategies for representing and sampling mixed random
variables, an extrinsic ("sample-and-project") and an intrinsic one (based on
face stratification). We experiment with both approaches on an emergent
communication benchmark and on modeling MNIST and Fashion-MNIST data with
variational auto-encoders with mixed latent variables.

    

### [[2108.02662] Reducing Unintended Bias of ML Models on Tabular and Textual Data](http://arxiv.org/abs/2108.02662)


  Unintended biases in machine learning (ML) models are among the major
concerns that must be addressed to maintain public trust in ML. In this paper,
we address process fairness of ML models that consists in reducing the
dependence of models on sensitive features, without compromising their
performance. We revisit the framework FixOut that is inspired in the approach
"fairness through unawareness" to build fairer models. We introduce several
improvements such as automating the choice of FixOut's parameters. Also, FixOut
was originally proposed to improve fairness of ML models on tabular data. We
also demonstrate the feasibility of FixOut's workflow for models on textual
data. We present several experimental results that illustrate the fact that
FixOut improves process fairness on different classification settings.

    

### [[2108.02665] Deep Reinforcement Learning for Continuous Docking Control of Autonomous Underwater Vehicles: A Benchmarking Study](http://arxiv.org/abs/2108.02665)


  Docking control of an autonomous underwater vehicle (AUV) is a task that is
integral to achieving persistent long term autonomy. This work explores the
application of state-of-the-art model-free deep reinforcement learning (DRL)
approaches to the task of AUV docking in the continuous domain. We provide a
detailed formulation of the reward function, utilized to successfully dock the
AUV onto a fixed docking platform. A major contribution that distinguishes our
work from the previous approaches is the usage of a physics simulator to define
and simulate the underwater environment as well as the DeepLeng AUV. We propose
a new reward function formulation for the docking task, incorporating several
components, that outperforms previous reward formulations. We evaluate proximal
policy optimization (PPO), twin delayed deep deterministic policy gradients
(TD3) and soft actor-critic (SAC) in combination with our reward function. Our
evaluation yielded results that conclusively show the TD3 agent to be most
efficient and consistent in terms of docking the AUV, over multiple evaluation
runs it achieved a 100% success rate and episode return of 10667.1 +- 688.8. We
also show how our reward function formulation improves over the state of the
art.

    

### [[2108.02671] Visual Domain Adaptation for Monocular Depth Estimation on Resource-Constrained Hardware](http://arxiv.org/abs/2108.02671)


  Real-world perception systems in many cases build on hardware with limited
resources to adhere to cost and power limitations of their carrying system.
Deploying deep neural networks on resource-constrained hardware became possible
with model compression techniques, as well as efficient and hardware-aware
architecture design. However, model adaptation is additionally required due to
the diverse operation environments. In this work, we address the problem of
training deep neural networks on resource-constrained hardware in the context
of visual domain adaptation. We select the task of monocular depth estimation
where our goal is to transform a pre-trained model to the target's domain data.
While the source domain includes labels, we assume an unlabelled target domain,
as it happens in real-world applications. Then, we present an adversarial
learning approach that is adapted for training on the device with limited
resources. Since visual domain adaptation, i.e. neural network training, has
not been previously explored for resource-constrained hardware, we present the
first feasibility study for image-based depth estimation. Our experiments show
that visual domain adaptation is relevant only for efficient network
architectures and training sets at the order of a few hundred samples. Models
and code are publicly available.

    

### [[2108.02676] Redesigning Fully Convolutional DenseUNets for Large Histopathology Images](http://arxiv.org/abs/2108.02676)


  The automated segmentation of cancer tissue in histopathology images can help
clinicians to detect, diagnose, and analyze such disease. Different from other
natural images used in many convolutional networks for benchmark,
histopathology images can be extremely large, and the cancerous patterns can
reach beyond 1000 pixels. Therefore, the well-known networks in the literature
were never conceived to handle these peculiarities. In this work, we propose a
Fully Convolutional DenseUNet that is particularly designed to solve
histopathology problems. We evaluated our network in two public pathology
datasets published as challenges in the recent MICCAI 2019: binary segmentation
in colon cancer images (DigestPath2019), and multi-class segmentation in
prostate cancer images (Gleason2019), achieving similar and better results than
the winners of the challenges, respectively. Furthermore, we discussed some
good practices in the training setup to yield the best performance and the main
challenges in these histopathology datasets.

    

### [[2108.02694] Using Metamorphic Relations to Verify and Enhance Artcode Classification](http://arxiv.org/abs/2108.02694)


  Software testing is often hindered where it is impossible or impractical to
determine the correctness of the behaviour or output of the software under test
(SUT), a situation known as the oracle problem. An example of an area facing
the oracle problem is automatic image classification, using machine learning to
classify an input image as one of a set of predefined classes. An approach to
software testing that alleviates the oracle problem is metamorphic testing
(MT). While traditional software testing examines the correctness of individual
test cases, MT instead examines the relations amongst multiple executions of
test cases and their outputs. These relations are called metamorphic relations
(MRs): if an MR is found to be violated, then a fault must exist in the SUT.
This paper examines the problem of classifying images containing visually
hidden markers called Artcodes, and applies MT to verify and enhance the
trained classifiers. This paper further examines two MRs, Separation and
Occlusion, and reports on their capability in verifying the image
classification using one-way analysis of variance (ANOVA) in conjunction with
three other statistical analysis methods: t-test (for unequal variances),
Kruskal-Wallis test, and Dunnett's test. In addition to our previously-studied
classifier, that used Random Forests, we introduce a new classifier that uses a
support vector machine, and present its MR-augmented version. Experimental
evaluations across a number of performance metrics show that the augmented
classifiers can achieve better performance than non-augmented classifiers. This
paper also analyses how the enhanced performance is obtained.

    

### [[2108.02696] A Low Rank Promoting Prior for Unsupervised Contrastive Learning](http://arxiv.org/abs/2108.02696)


  Unsupervised learning is just at a tipping point where it could really take
off. Among these approaches, contrastive learning has seen tremendous progress
and led to state-of-the-art performance. In this paper, we construct a novel
probabilistic graphical model that effectively incorporates the low rank
promoting prior into the framework of contrastive learning, referred to as
LORAC. In contrast to the existing conventional self-supervised approaches that
only considers independent learning, our hypothesis explicitly requires that
all the samples belonging to the same instance class lie on the same subspace
with small dimension. This heuristic poses particular joint learning
constraints to reduce the degree of freedom of the problem during the search of
the optimal network parameterization. Most importantly, we argue that the low
rank prior employed here is not unique, and many different priors can be
invoked in a similar probabilistic way, corresponding to different hypotheses
about underlying truth behind the contrastive features. Empirical evidences
show that the proposed algorithm clearly surpasses the state-of-the-art
approaches on multiple benchmarks, including image classification, object
detection, instance segmentation and keypoint detection.

    

### [[2108.02701] Lyapunov Robust Constrained-MDPs: Soft-Constrained Robustly Stable Policy Optimization under Model Uncertainty](http://arxiv.org/abs/2108.02701)


  Safety and robustness are two desired properties for any reinforcement
learning algorithm. CMDPs can handle additional safety constraints and RMDPs
can perform well under model uncertainties. In this paper, we propose to unite
these two frameworks resulting in robust constrained MDPs (RCMDPs). The
motivation is to develop a framework that can satisfy safety constraints while
also simultaneously offer robustness to model uncertainties. We develop the
RCMDP objective, derive gradient update formula to optimize this objective and
then propose policy gradient based algorithms. We also independently propose
Lyapunov based reward shaping for RCMDPs, yielding better stability and
convergence properties.

    

### [[2108.02704] Rotaflip: A New CNN Layer for Regularization and Rotational Invariance in Medical Images](http://arxiv.org/abs/2108.02704)


  Regularization in convolutional neural networks (CNNs) is usually addressed
with dropout layers. However, dropout is sometimes detrimental in the
convolutional part of a CNN as it simply sets to zero a percentage of pixels in
the feature maps, adding unrepresentative examples during training. Here, we
propose a CNN layer that performs regularization by applying random rotations
of reflections to a small percentage of feature maps after every convolutional
layer. We prove how this concept is beneficial for images with orientational
symmetries, such as in medical images, as it provides a certain degree of
rotational invariance. We tested this method in two datasets, a patch-based set
of histopathology images (PatchCamelyon) to perform classification using a
generic DenseNet, and a set of specular microscopy images of the corneal
endothelium to perform segmentation using a tailored U-net, improving the
performance in both cases.

    

### [[2108.02713] Role-based lateral movement detection with unsupervised learning](http://arxiv.org/abs/2108.02713)


  Adversarial lateral movement via compromised accounts remains difficult to
discover via traditional rule-based defenses because it generally lacks
explicit indicators of compromise. We propose a behavior-based, unsupervised
framework comprising two methods of lateral movement detection on enterprise
networks: one aimed at generic lateral movement via either exploit or
authenticated connections, and one targeting the specific techniques of process
injection and hijacking. The first method is based on the premise that the role
of a system---the functions it performs on the network---determines the roles
of the systems it should make connections with. The adversary meanwhile might
move between any systems whatever, possibly seeking out systems with unusual
roles that facilitate certain accesses. We use unsupervised learning to cluster
systems according to role and identify connections to systems with novel roles
as potentially malicious. The second method is based on the premise that the
temporal patterns of inter-system processes that facilitate these connections
depend on the roles of the systems involved. If a process is compromised by an
attacker, these normal patterns might be disrupted in discernible ways. We
apply frequent-itemset mining to process sequences to establish regular
patterns of communication between systems based on role, and identify rare
process sequences as signalling potentially malicious connections.

    

### [[2108.02717] Beyond No Regret: Instance-Dependent PAC Reinforcement Learning](http://arxiv.org/abs/2108.02717)


  The theory of reinforcement learning has focused on two fundamental problems:
achieving low regret, and identifying $\epsilon$-optimal policies. While a
simple reduction allows one to apply a low-regret algorithm to obtain an
$\epsilon$-optimal policy and achieve the worst-case optimal rate, it is
unknown whether low-regret algorithms can obtain the instance-optimal rate for
policy identification. We show that this is not possible -- there exists a
fundamental tradeoff between achieving low regret and identifying an
$\epsilon$-optimal policy at the instance-optimal rate.
Motivated by our negative finding, we propose a new measure of
instance-dependent sample complexity for PAC tabular reinforcement learning
which explicitly accounts for the attainable state visitation distributions in
the underlying MDP. We then propose and analyze a novel, planning-based
algorithm which attains this sample complexity -- yielding a complexity which
scales with the suboptimality gaps and the ``reachability'' of a state. We show
that our algorithm is nearly minimax optimal, and on several examples that our
instance-dependent sample complexity offers significant improvements over
worst-case bounds.

    

### [[2108.02722] Video Contrastive Learning with Global Context](http://arxiv.org/abs/2108.02722)


  Contrastive learning has revolutionized self-supervised image representation
learning field, and recently been adapted to video domain. One of the greatest
advantages of contrastive learning is that it allows us to flexibly define
powerful loss objectives as long as we can find a reasonable way to formulate
positive and negative samples to contrast. However, existing approaches rely
heavily on the short-range spatiotemporal salience to form clip-level
contrastive signals, thus limit themselves from using global context. In this
paper, we propose a new video-level contrastive learning method based on
segments to formulate positive pairs. Our formulation is able to capture global
context in a video, thus robust to temporal content change. We also incorporate
a temporal order regularization term to enforce the inherent sequential
structure of videos. Extensive experiments show that our video-level
contrastive learning framework (VCLR) is able to outperform previous
state-of-the-arts on five video datasets for downstream action classification,
action localization and video retrieval. Code is available at
this https URL.

    

### [[2108.02731] Mean-Field Multi-Agent Reinforcement Learning: A Decentralized Network Approach](http://arxiv.org/abs/2108.02731)


  One of the challenges for multi-agent reinforcement learning (MARL) is
designing efficient learning algorithms for a large system in which each agent
has only limited or partial information of the entire system. In this system,
it is desirable to learn policies of a decentralized type. A recent and
promising paradigm to analyze such decentralized MARL is to take network
structures into consideration. While exciting progress has been made to analyze
decentralized MARL with the network of agents, often found in social networks
and team video games, little is known theoretically for decentralized MARL with
the network of states, frequently used for modeling self-driving vehicles,
ride-sharing, and data and traffic routing.
This paper proposes a framework called localized training and decentralized
execution to study MARL with network of states, with homogeneous (a.k.a.
mean-field type) agents. Localized training means that agents only need to
collect local information in their neighboring states during the training
phase; decentralized execution implies that, after the training stage, agents
can execute the learned decentralized policies, which only requires knowledge
of the agents' current states. The key idea is to utilize the homogeneity of
agents and regroup them according to their states, thus the formulation of a
networked Markov decision process with teams of agents, enabling the update of
the Q-function in a localized fashion. In order to design an efficient and
scalable reinforcement learning algorithm under such a framework, we adopt the
actor-critic approach with over-parameterized neural networks, and establish
the convergence and sample complexity for our algorithm, shown to be scalable
with respect to the size of both agents and states.

    

### [[2108.02741] GIFAIR-FL: An Approach for Group and Individual Fairness in Federated Learning](http://arxiv.org/abs/2108.02741)


  In this paper we propose \texttt{GIFAIR-FL}: an approach that imposes group
and individual fairness to federated learning settings. By adding a
regularization term, our algorithm penalizes the spread in the loss of client
groups to drive the optimizer to fair solutions. Theoretically, we show
convergence in non-convex and strongly convex settings. Our convergence
guarantees hold for both $i.i.d.$ and non-$i.i.d.$ data. To demonstrate the
empirical performance of our algorithm, we apply our method on image
classification and text prediction tasks. Compared to existing algorithms, our
method shows improved fairness results while retaining superior or similar
prediction accuracy.

    

### [[2108.02743] Semi- and Self-Supervised Multi-View Fusion of 3D Microscopy Images using Generative Adversarial Networks](http://arxiv.org/abs/2108.02743)


  Recent developments in fluorescence microscopy allow capturing
high-resolution 3D images over time for living model organisms. To be able to
image even large specimens, techniques like multi-view light-sheet imaging
record different orientations at each time point that can then be fused into a
single high-quality volume. Based on measured point spread functions (PSF),
deconvolution and content fusion are able to largely revert the inevitable
degradation occurring during the imaging process. Classical multi-view
deconvolution and fusion methods mainly use iterative procedures and
content-based averaging. Lately, Convolutional Neural Networks (CNNs) have been
deployed to approach 3D single-view deconvolution microscopy, but the
multi-view case waits to be studied. We investigated the efficacy of CNN-based
multi-view deconvolution and fusion with two synthetic data sets that mimic
developing embryos and involve either two or four complementary 3D views.
Compared with classical state-of-the-art methods, the proposed semi- and
self-supervised models achieve competitive and superior deconvolution and
fusion quality in the two-view and quad-view cases, respectively.

    

### [[2108.02744] Deep learning for inverse problems with unknown operator](http://arxiv.org/abs/2108.02744)


  We consider ill-posed inverse problems where the forward operator $T$ is
unknown, and instead we have access to training data consisting of functions
$f_i$ and their noisy images $Tf_i$. This is a practically relevant and
challenging problem which current methods are able to solve only under strong
assumptions on the training set. Here we propose a new method that requires
minimal assumptions on the data, and prove reconstruction rates that depend on
the number of training points and the noise level. We show that, in the regime
of "many" training data, the method is minimax optimal. The proposed method
employs a type of convolutional neural networks (U-nets) and empirical risk
minimization in order to "fit" the unknown operator. In a nutshell, our
approach is based on two ideas: the first is to relate U-nets to multiscale
decompositions such as wavelets, thereby linking them to the existing theory,
and the second is to use the hierarchical structure of U-nets and the low
number of parameters of convolutional neural nets to prove entropy bounds that
are practically useful. A significant difference with the existing works on
neural networks in nonparametric statistics is that we use them to approximate
operators and not functions, which we argue is mathematically more natural and
technically more convenient.

    

### [[2108.02755] The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning](http://arxiv.org/abs/2108.02755)


  AI and reinforcement learning (RL) have improved many areas, but are not yet
widely adopted in economic policy design, mechanism design, or economics at
large. At the same time, current economic methodology is limited by a lack of
counterfactual data, simplistic behavioral models, and limited opportunities to
experiment with policies and evaluate behavioral responses. Here we show that
machine-learning-based economic simulation is a powerful policy and mechanism
design framework to overcome these limitations. The AI Economist is a
two-level, deep RL framework that trains both agents and a social planner who
co-adapt, providing a tractable solution to the highly unstable and novel
two-level RL challenge. From a simple specification of an economy, we learn
rational agent behaviors that adapt to learned planner policies and vice versa.
We demonstrate the efficacy of the AI Economist on the problem of optimal
taxation. In simple one-step economies, the AI Economist recovers the optimal
tax policy of economic theory. In complex, dynamic economies, the AI Economist
substantially improves both utilitarian social welfare and the trade-off
between equality and productivity over baselines. It does so despite emergent
tax-gaming strategies, while accounting for agent interactions and behavioral
change more accurately than economic theory. These results demonstrate for the
first time that two-level, deep RL can be used for understanding and as a
complement to theory for economic design, unlocking a new computational
learning-based approach to understanding economic policy.

    

### [[2108.02756] BOSS: Bidirectional One-Shot Synthesis of Adversarial Examples](http://arxiv.org/abs/2108.02756)


  The design of additive imperceptible perturbations to the inputs of deep
classifiers to maximize their misclassification rates is a central focus of
adversarial machine learning. An alternative approach is to synthesize
adversarial examples from scratch using GAN-like structures, albeit with the
use of large amounts of training data. By contrast, this paper considers
one-shot synthesis of adversarial examples; the inputs are synthesized from
scratch to induce arbitrary soft predictions at the output of pre-trained
models, while simultaneously maintaining high similarity to specified inputs.
To this end, we present a problem that encodes objectives on the distance
between the desired and output distributions of the trained model and the
similarity between such inputs and the synthesized examples. We prove that the
formulated problem is NP-complete. Then, we advance a generative approach to
the solution in which the adversarial examples are obtained as the output of a
generative network whose parameters are iteratively updated by optimizing
surrogate loss functions for the dual-objective. We demonstrate the generality
and versatility of the framework and approach proposed through applications to
the design of targeted adversarial attacks, generation of decision boundary
samples, and synthesis of low confidence classification inputs. The approach is
further extended to an ensemble of models with different soft output
specifications. The experimental results verify that the targeted and
confidence reduction attack methods developed perform on par with
state-of-the-art algorithms.

    

### [[2108.02768] Learning to Elect](http://arxiv.org/abs/2108.02768)


  Voting systems have a wide range of applications including recommender
systems, web search, product design and elections. Limited by the lack of
general-purpose analytical tools, it is difficult to hand-engineer desirable
voting rules for each use case. For this reason, it is appealing to
automatically discover voting rules geared towards each scenario. In this
paper, we show that set-input neural network architectures such as Set
Transformers, fully-connected graph networks and DeepSets are both
theoretically and empirically well-suited for learning voting rules. In
particular, we show that these network models can not only mimic a number of
existing voting rules to compelling accuracy --- both position-based (such as
Plurality and Borda) and comparison-based (such as Kemeny, Copeland and
Maximin) --- but also discover near-optimal voting rules that maximize
different social welfare functions. Furthermore, the learned voting rules
generalize well to different voter utility distributions and election sizes
unseen during training.

    

### [[2108.02774] Sketch Your Own GAN](http://arxiv.org/abs/2108.02774)


  Can a user create a deep generative model by sketching a single example?
Traditionally, creating a GAN model has required the collection of a
large-scale dataset of exemplars and specialized knowledge in deep learning. In
contrast, sketching is possibly the most universally accessible way to convey a
visual concept. In this work, we present a method, GAN Sketching, for rewriting
GANs with one or more sketches, to make GANs training easier for novice users.
In particular, we change the weights of an original GAN model according to user
sketches. We encourage the model's output to match the user sketches through a
cross-domain adversarial loss. Furthermore, we explore different regularization
methods to preserve the original model's diversity and image quality.
Experiments have shown that our method can mold GANs to match shapes and poses
specified by sketches while maintaining realism and diversity. Finally, we
demonstrate a few applications of the resulting GAN, including latent space
interpolation and image editing.

    

### [[2108.02776] Sinsy: A Deep Neural Network-Based Singing Voice Synthesis System](http://arxiv.org/abs/2108.02776)


  This paper presents Sinsy, a deep neural network (DNN)-based singing voice
synthesis (SVS) system. In recent years, DNNs have been utilized in statistical
parametric SVS systems, and DNN-based SVS systems have demonstrated better
performance than conventional hidden Markov model-based ones. SVS systems are
required to synthesize a singing voice with pitch and timing that strictly
follow a given musical score. Additionally, singing expressions that are not
described on the musical score, such as vibrato and timing fluctuations, should
be reproduced. The proposed system is composed of four modules: a time-lag
model, a duration model, an acoustic model, and a vocoder, and singing voices
can be synthesized taking these characteristics of singing voices into account.
To better model a singing voice, the proposed system incorporates improved
approaches to modeling pitch and vibrato and better training criteria into the
acoustic model. In addition, we incorporated PeriodNet, a non-autoregressive
neural vocoder with robustness for the pitch, into our systems to generate a
high-fidelity singing voice waveform. Moreover, we propose automatic pitch
correction techniques for DNN-based SVS to synthesize singing voices with
correct pitch even if the training data has out-of-tune phrases. Experimental
results show our system can synthesize a singing voice with better timing, more
natural vibrato, and correct pitch, and it can achieve better mean opinion
scores in subjective evaluation tests.

    

### [[1910.00964] Benchmarking machine learning models on multi-centre eICU critical care dataset](http://arxiv.org/abs/1910.00964)


  Progress of machine learning in critical care has been difficult to track, in
part due to absence of public benchmarks. Other fields of research (such as
computer vision and natural language processing) have established various
competitions and public benchmarks. Recent availability of large clinical
datasets has enabled the possibility of establishing public benchmarks. Taking
advantage of this opportunity, we propose a public benchmark suite to address
four areas of critical care, namely mortality prediction, estimation of length
of stay, patient phenotyping and risk of decompensation. We define each task
and compare the performance of both clinical models as well as baseline and
deep learning models using eICU critical care dataset of around 73,000
patients. This is the first public benchmark on a multi-centre critical care
dataset, comparing the performance of clinical gold standard with our
predictive model. We also investigate the impact of numerical variables as well
as handling of categorical variables on each of the defined tasks. The source
code, detailing our methods and experiments is publicly available such that
anyone can replicate our results and build upon our work.

    

### [[2001.02309] VC-dimensions of nondeterministic finite automata for words of equal length](http://arxiv.org/abs/2001.02309)


  Let $NFA_b(q)$ denote the set of languages accepted by nondeterministic
finite automata with $q$ states over an alphabet with $b$ letters. Let $B_n$
denote the set of words of length $n$. We give a quadratic lower bound on the
VC dimension of \[
NFA_2(q)\cap B_n = \{L\cap B_n \mid L \in NFA_2(q)\} \] as a function of $q$.
Next, the work of Gruber and Holzer (2007) gives an upper bound for the
nondeterministic state complexity of finite languages contained in $B_n$, which
we strengthen using our methods.
Finally, we give some theoretical and experimental results on the dependence
on $n$ of the VC dimension and testing dimension of $NFA_2(q)\cap B_n$.

    

### [[2003.04696] TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning](http://arxiv.org/abs/2003.04696)


  Processing of medical images such as MRI or CT presents unique challenges
compared to RGB images typically used in computer vision. These include a lack
of labels for large datasets, high computational costs, and metadata to
describe the physical properties of voxels. Data augmentation is used to
artificially increase the size of the training datasets. Training with image
patches decreases the need for computational power. Spatial metadata needs to
be carefully taken into account in order to ensure a correct alignment of
volumes.
We present TorchIO, an open-source Python library to enable efficient
loading, preprocessing, augmentation and patch-based sampling of medical images
for deep learning. TorchIO follows the style of PyTorch and integrates standard
medical image processing libraries to efficiently process images during
training of neural networks. TorchIO transforms can be composed, reproduced,
traced and extended. We provide multiple generic preprocessing and augmentation
operations as well as simulation of MRI-specific artifacts.
Source code, comprehensive tutorials and extensive documentation for TorchIO
can be found at this https URL. The package can be installed from the
Python Package Index running 'pip install torchio'. It includes a command-line
interface which allows users to apply transforms to image files without using
Python. Additionally, we provide a graphical interface within a TorchIO
extension in 3D Slicer to visualize the effects of transforms.
TorchIO was developed to help researchers standardize medical image
processing pipelines and allow them to focus on the deep learning experiments.
It encourages open science, as it supports reproducibility and is version
controlled so that the software can be cited precisely. Due to its modularity,
the library is compatible with other frameworks for deep learning with medical
images.

    

### [[2006.04005] Entropic Out-of-Distribution Detection: Seamless Detection of Unknown Examples](http://arxiv.org/abs/2006.04005)


  In this paper, we argue that the unsatisfactory out-of-distribution (OOD)
detection performance of neural networks is mainly due to the SoftMax loss
anisotropy and propensity to produce low entropy probability distributions in
disagreement with the principle of maximum entropy. Current out-of-distribution
(OOD) detection approaches usually do not directly fix the SoftMax loss
drawbacks, but rather build techniques to circumvent it. Unfortunately, those
methods usually produce undesired side effects (e.g., classification accuracy
drop, additional hyperparameters, slower inferences, and collecting extra
data). In the opposite direction, we propose replacing SoftMax loss with a
novel loss function that does not suffer from the mentioned weaknesses. The
proposed IsoMax loss is isotropic (exclusively distance-based) and provides
high entropy posterior probability distributions. Replacing the SoftMax loss by
IsoMax loss requires no model or training changes. Additionally, the models
trained with IsoMax loss produce as fast and energy-efficient inferences as
those trained using SoftMax loss. Moreover, no classification accuracy drop is
observed. The proposed method does not rely on outlier/background data,
hyperparameter tuning, temperature calibration, feature extraction, metric
learning, adversarial training, ensemble procedures, or generative models. Our
experiments showed that IsoMax loss works as a seamless SoftMax loss drop-in
replacement that significantly improves neural networks' OOD detection
performance. Hence, it may be used as a baseline OOD detection approach to be
combined with current or future OOD detection techniques to achieve even higher
results.

    

### [[2006.14621] Understanding collections of related datasets using dependent MMD coresets](http://arxiv.org/abs/2006.14621)


  Understanding how two datasets differ can help us determine whether one
dataset under-represents certain sub-populations, and provides insights into
how well models will generalize across datasets. Representative points selected
by a maximum mean discrepency (MMD) coreset can provide interpretable summaries
of a single dataset, but are not easily compared across datasets. In this paper
we introduce dependent MMD coresets, a data summarization method for
collections of datasets that facilitates comparison of distributions. We show
that dependent MMD coresets are useful for understanding multiple related
datasets and understanding model generalization between such datasets.

    

### [[2010.04767] Robust Behavioral Cloning for Autonomous Vehicles using End-to-End Imitation Learning](http://arxiv.org/abs/2010.04767)


  In this work, we present a lightweight pipeline for robust behavioral cloning
of a human driver using end-to-end imitation learning. The proposed pipeline
was employed to train and deploy three distinct driving behavior models onto a
simulated vehicle. The training phase comprised of data collection, balancing,
augmentation, preprocessing and training a neural network, following which, the
trained model was deployed onto the ego vehicle to predict steering commands
based on the feed from an onboard camera. A novel coupled control law was
formulated to generate longitudinal control commands on-the-go based on the
predicted steering angle and other parameters such as actual speed of the ego
vehicle and the prescribed constraints for speed and steering. We analyzed
computational efficiency of the pipeline and evaluated robustness of the
trained models through exhaustive experimentation during the deployment phase.
We also compared our approach against state-of-the-art implementation in order
to comment on its validity.

    

### [[2010.06610] Training independent subnetworks for robust prediction](http://arxiv.org/abs/2010.06610)


  Recent approaches to efficiently ensemble neural networks have shown that
strong robustness and uncertainty performance can be achieved with a negligible
gain in parameters over the original network. However, these methods still
require multiple forward passes for prediction, leading to a significant
computational cost. In this work, we show a surprising result: the benefits of
using multiple predictions can be achieved `for free' under a single model's
forward pass. In particular, we show that, using a multi-input multi-output
(MIMO) configuration, one can utilize a single model's capacity to train
multiple subnetworks that independently learn the task at hand. By ensembling
the predictions made by the subnetworks, we improve model robustness without
increasing compute. We observe a significant improvement in negative
log-likelihood, accuracy, and calibration error on CIFAR10, CIFAR100, ImageNet,
and their out-of-distribution variants compared to previous methods.

    

### [[2011.11503] Metric Transforms and Low Rank Matrices via Representation Theory of the Real Hyperrectangle](http://arxiv.org/abs/2011.11503)


  In this paper, we develop a new technique which we call representation theory
of the real hyperrectangle, which describes how to compute the eigenvectors and
eigenvalues of certain matrices arising from hyperrectangles. We show that
these matrices arise naturally when analyzing a number of different algorithmic
tasks such as kernel methods, neural network training, natural language
processing, and the design of algorithms using the polynomial method. We then
use our new technique along with these connections to prove several new
structural results in these areas, including:
$\bullet$ A function is a positive definite Manhattan kernel if and only if
it is a completely monotone function. These kernels are widely used across
machine learning; one example is the Laplace kernel which is widely used in
machine learning for chemistry.
$\bullet$ A function transforms Manhattan distances to Manhattan distances if
and only if it is a Bernstein function. This completes the theory of Manhattan
to Manhattan metric transforms initiated by Assouad in 1980.
$\bullet$ A function applied entry-wise to any square matrix of rank $r$
always results in a matrix of rank $< 2^{r-1}$ if and only if it is a
polynomial of sufficiently low degree. This gives a converse to a key lemma
used by the polynomial method in algorithm design.
Our work includes a sophisticated combination of techniques from different
fields, including metric embeddings, the polynomial method, and group
representation theory.

    

### [[2012.10861] Memory AMP](http://arxiv.org/abs/2012.10861)


  Approximate message passing (AMP) is a low-cost iterative
parameter-estimation technique for certain high-dimensional linear systems with
non-Gaussian distributions. However, AMP only applies to independent
identically distributed (IID) transform matrices, but may become unreliable
(e.g. perform poorly or even diverge) for other matrix ensembles, especially
for ill-conditioned ones. Orthogonal/vector AMP (OAMP/VAMP) was proposed for
general right-unitarily-invariant matrices to handle this difficulty. However,
the Bayes-optimal OAMP/VAMP requires a high-complexity linear minimum mean
square error (MMSE) estimator. This limits the application of OAMP/VAMP to
large-scale systems.
To solve the disadvantages of AMP and OAMP/VAMP, this paper proposes a memory
AMP (MAMP) framework under an orthogonality principle, which guarantees the
asymptotic IID Gaussianity of estimation errors in MAMP. We present an
orthogonalization procedure for the local memory estimators to realize the
required orthogonality for MAMP. Furthermore, we propose a Bayes-optimal MAMP
(BO-MAMP), in which a long-memory matched filter is proposed for interference
suppression. The complexity of BO-MAMP is comparable to AMP. A state evolution
is derived to asymptotically characterize the performance of BO-MAMP. Based on
state evolution, the relaxation parameters and damping vector in BO-MAMP are
optimized. For all right-unitarily-invariant matrices, the optimized BO-MAMP
converges to the high-complexity OAMP/VAMP, and thus is Bayes-optimal if it has
a unique fixed point. Finally, simulations are provided to verify the validity
and accuracy of the theoretical results.

    

### [[2012.13838] Inserting Information Bottlenecks for Attribution in Transformers](http://arxiv.org/abs/2012.13838)


  Pretrained transformers achieve the state of the art across tasks in natural
language processing, motivating researchers to investigate their inner
mechanisms. One common direction is to understand what features are important
for prediction. In this paper, we apply information bottlenecks to analyze the
attribution of each feature for prediction on a black-box model. We use BERT as
the example and evaluate our approach both quantitatively and qualitatively. We
show the effectiveness of our method in terms of attribution and the ability to
provide insight into how information flows through layers. We demonstrate that
our technique outperforms two competitive methods in degradation tests on four
datasets. Code is available at this https URL.

    

### [[2012.15194] Test Score Algorithms for Budgeted Stochastic Utility Maximization](http://arxiv.org/abs/2012.15194)


  Motivated by recent developments in designing algorithms based on individual
item scores for solving utility maximization problems, we study the framework
of using test scores, defined as a statistic of observed individual item
performance data, for solving the budgeted stochastic utility maximization
problem. We extend an existing scoring mechanism, namely the replication test
scores, to incorporate heterogeneous item costs as well as item values. We show
that a natural greedy algorithm that selects items solely based on their
replication test scores outputs solutions within a constant factor of the
optimum for a broad class of utility functions. Our algorithms and
approximation guarantees assume that test scores are noisy estimates of certain
expected values with respect to marginal distributions of individual item
values, thus making our algorithms practical and extending previous work that
assumes noiseless estimates. Moreover, we show how our algorithm can be adapted
to the setting where items arrive in a streaming fashion while maintaining the
same approximation guarantee. We present numerical results, using synthetic
data and data sets from the Academia.StackExchange Q&A forum, which show that
our test score algorithm can achieve competitiveness, and in some cases better
performance than a benchmark algorithm that requires access to a value oracle
to evaluate function values.

    

### [[2101.02703] Distribution-Free, Risk-Controlling Prediction Sets](http://arxiv.org/abs/2101.02703)


  While improving prediction accuracy has been the focus of machine learning in
recent years, this alone does not suffice for reliable decision-making.
Deploying learning systems in consequential settings also requires calibrating
and communicating the uncertainty of predictions. To convey instance-wise
uncertainty for prediction tasks, we show how to generate set-valued
predictions from a black-box predictor that control the expected loss on future
test points at a user-specified level. Our approach provides explicit
finite-sample guarantees for any dataset by using a holdout set to calibrate
the size of the prediction sets. This framework enables simple,
distribution-free, rigorous error control for many tasks, and we demonstrate it
in five large-scale machine learning problems: (1) classification problems
where some mistakes are more costly than others; (2) multi-label
classification, where each observation has multiple associated labels; (3)
classification problems where the labels have a hierarchical structure; (4)
image segmentation, where we wish to predict a set of pixels containing an
object of interest; and (5) protein structure prediction. Lastly, we discuss
extensions to uncertainty quantification for ranking, metric learning and
distributionally robust learning.

    

### [[2101.05231] Robust CUR Decomposition: Theory and Imaging Applications](http://arxiv.org/abs/2101.05231)


  This paper considers the use of Robust PCA in a CUR decomposition framework
and applications thereof. Our main algorithms produce a robust version of
column-row factorizations of matrices $\mathbf{D}=\mathbf{L}+\mathbf{S}$ where
$\mathbf{L}$ is low-rank and $\mathbf{S}$ contains sparse outliers. These
methods yield interpretable factorizations at low computational cost, and
provide new CUR decompositions that are robust to sparse outliers, in contrast
to previous methods. We consider two key imaging applications of Robust PCA:
video foreground-background separation and face modeling. This paper examines
the qualitative behavior of our Robust CUR decompositions on the benchmark
videos and face datasets, and find that our method works as well as standard
Robust PCA while being significantly faster. Additionally, we consider hybrid
randomized and deterministic sampling methods which produce a compact CUR
decomposition of a given matrix, and apply this to video sequences to produce
canonical frames thereof.

    

### [[2101.08176] Introduction to Normalizing Flows for Lattice Field Theory](http://arxiv.org/abs/2101.08176)


  This notebook tutorial demonstrates a method for sampling Boltzmann
distributions of lattice field theories using a class of machine learning
models known as normalizing flows. The ideas and approaches proposed in
arXiv:1904.12072, arXiv:2002.02428, and arXiv:2003.06413 are reviewed and a
concrete implementation of the framework is presented. We apply this framework
to a lattice scalar field theory and to U(1) gauge theory, explicitly encoding
gauge symmetries in the flow-based approach to the latter. This presentation is
intended to be interactive and working with the attached Jupyter notebook is
recommended.

    

### [[2102.00968] CRPS Learning](http://arxiv.org/abs/2102.00968)


  Combination and aggregation techniques can significantly improve forecast
accuracy. This also holds for probabilistic forecasting methods where
predictive distributions are combined. There are several time-varying and
adaptive weighting schemes such as Bayesian model averaging (BMA). However, the
quality of different forecasts may vary not only over time but also within the
distribution. For example, some distribution forecasts may be more accurate in
the center of the distributions, while others are better at predicting the
tails. Therefore, we introduce a new weighting method that considers the
differences in performance over time and within the distribution. We discuss
pointwise combination algorithms based on aggregation across quantiles that
optimize with respect to the continuous ranked probability score (CRPS). After
analyzing the theoretical properties of pointwise CRPS learning, we discuss B-
and P-Spline-based estimation techniques for batch and online learning, based
on quantile regression and prediction with expert advice. We prove that the
proposed fully adaptive Bernstein online aggregation (BOA) method for pointwise
CRPS online learning has optimal convergence properties. They are confirmed in
simulations and a probabilistic forecasting study for European emission
allowance (EUA) prices.

    

### [[2102.01583] The Min-Max Complexity of Distributed Stochastic Convex Optimization with Intermittent Communication](http://arxiv.org/abs/2102.01583)


  We resolve the min-max complexity of distributed stochastic convex
optimization (up to a log factor) in the intermittent communication setting,
where $M$ machines work in parallel over the course of $R$ rounds of
communication to optimize the objective, and during each round of
communication, each machine may sequentially compute $K$ stochastic gradient
estimates. We present a novel lower bound with a matching upper bound that
establishes an optimal algorithm.

    

### [[2103.00111] Graph Self-Supervised Learning: A Survey](http://arxiv.org/abs/2103.00111)


  Deep learning on graphs has attracted significant interests recently.
However, most of the works have focused on (semi-) supervised learning,
resulting in shortcomings including heavy label reliance, poor generalization,
and weak robustness. To address these issues, self-supervised learning (SSL),
which extracts informative knowledge through well-designed pretext tasks
without relying on manual labels, has become a promising and trending learning
paradigm for graph data. Different from SSL on other domains like computer
vision and natural language processing, SSL on graphs has an exclusive
background, design ideas, and taxonomies. Under the umbrella of graph
self-supervised learning, we present a timely and comprehensive review of the
existing approaches which employ SSL techniques for graph data. We construct a
unified framework that mathematically formalizes the paradigm of graph SSL.
According to the objectives of pretext tasks, we divide these approaches into
four categories: generation-based, auxiliary property-based, contrast-based,
and hybrid approaches. We further conclude the applications of graph SSL across
various research fields and summarize the commonly used datasets, evaluation
benchmark, performance comparison and open-source codes of graph SSL. Finally,
we discuss the remaining challenges and potential future directions in this
research field.

    

### [[2103.04000] Off-Belief Learning](http://arxiv.org/abs/2103.04000)


  The standard problem setting in Dec-POMDPs is self-play, where the goal is to
find a set of policies that play optimally together. Policies learned through
self-play may adopt arbitrary conventions and implicitly rely on multi-step
reasoning based on fragile assumptions about other agents' actions and thus
fail when paired with humans or independently trained agents at test time. To
address this, we present off-belief learning (OBL). At each timestep OBL agents
follow a policy $\pi_1$ that is optimized assuming past actions were taken by a
given, fixed policy ($\pi_0$), but assuming that future actions will be taken
by $\pi_1$. When $\pi_0$ is uniform random, OBL converges to an optimal policy
that does not rely on inferences based on other agents' behavior (an optimal
grounded policy). OBL can be iterated in a hierarchy, where the optimal policy
from one level becomes the input to the next, thereby introducing multi-level
cognitive reasoning in a controlled manner. Unlike existing approaches, which
may converge to any equilibrium policy, OBL converges to a unique policy,
making it suitable for zero-shot coordination (ZSC). OBL can be scaled to
high-dimensional settings with a fictitious transition mechanism and shows
strong performance in both a toy-setting and the benchmark human-AI & ZSC
problem Hanabi.

    

### [[2103.15758] Compositional Abstraction Error and a Category of Causal Models](http://arxiv.org/abs/2103.15758)


  Interventional causal models describe several joint distributions over some
variables used to describe a system, one for each intervention setting. They
provide a formal recipe for how to move between the different joint
distributions and make predictions about the variables upon intervening on the
system. Yet, it is difficult to formalise how we may change the underlying
variables used to describe the system, say moving from fine-grained to
coarse-grained variables. Here, we argue that compositionality is a desideratum
for such model transformations and the associated errors: When abstracting a
reference model M iteratively, first obtaining M' and then further simplifying
that to obtain M'', we expect the composite transformation from M to M'' to
exist and its error to be bounded by the errors incurred by each individual
transformation step. Category theory, the study of mathematical objects via
compositional transformations between them, offers a natural language to
develop our framework for model transformations and abstractions. We introduce
a category of finite interventional causal models and, leveraging theory of
enriched categories, prove the desired compositionality properties for our
framework.

    

### [[2104.00531] Extending Neural P-frame Codecs for B-frame Coding](http://arxiv.org/abs/2104.00531)


  While most neural video codecs address P-frame coding (predicting each frame
from past ones), in this paper we address B-frame compression (predicting
frames using both past and future reference frames). Our B-frame solution is
based on the existing P-frame methods. As a result, B-frame coding capability
can easily be added to an existing neural codec. The basic idea of our B-frame
coding method is to interpolate the two reference frames to generate a single
reference frame and then use it together with an existing P-frame codec to
encode the input B-frame. Our studies show that the interpolated frame is a
much better reference for the P-frame codec compared to using the previous
frame as is usually done. Our results show that using the proposed method with
an existing P-frame codec can lead to 28.5%saving in bit-rate on the UVG
dataset compared to the P-frame codec while generating the same video quality.

    

### [[2104.10972] ImageNet-21K Pretraining for the Masses](http://arxiv.org/abs/2104.10972)


  ImageNet-1K serves as the primary dataset for pretraining deep learning
models for computer vision tasks. ImageNet-21K dataset, which is bigger and
more diverse, is used less frequently for pretraining, mainly due to its
complexity, low accessibility, and underestimation of its added value. This
paper aims to close this gap, and make high-quality efficient pretraining on
ImageNet-21K available for everyone. Via a dedicated preprocessing stage,
utilization of WordNet hierarchical structure, and a novel training scheme
called semantic softmax, we show that various models significantly benefit from
ImageNet-21K pretraining on numerous datasets and tasks, including small
mobile-oriented models. We also show that we outperform previous ImageNet-21K
pretraining schemes for prominent new models like ViT and Mixer. Our proposed
pretraining pipeline is efficient, accessible, and leads to SoTA reproducible
results, from a publicly available dataset. The training code and pretrained
models are available at: this https URL


### [[2104.14744] Human strategic decision making in parametrized games](http://arxiv.org/abs/2104.14744)


  Many real-world games contain parameters which can affect payoffs, action
spaces, and information states. For fixed values of the parameters, the game
can be solved using standard algorithms. However, in many settings agents must
act without knowing the values of the parameters that will be encountered in
advance. Often the decisions must be made by a human under time and resource
constraints, and it is unrealistic to assume that a human can solve the game in
real time. We present a new framework that enables human decision makers to
make fast decisions without the aid of real-time solvers. We demonstrate
applicability to a variety of situations including settings with multiple
players and imperfect information.

    

### [[2105.06337] Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](http://arxiv.org/abs/2105.06337)


  Recently, denoising diffusion probabilistic models and generative score
matching have shown high potential in modelling complex data distributions
while stochastic calculus has provided a unified point of view on these
techniques allowing for flexible inference schemes. In this paper we introduce
Grad-TTS, a novel text-to-speech model with score-based decoder producing
mel-spectrograms by gradually transforming noise predicted by encoder and
aligned with text input by means of Monotonic Alignment Search. The framework
of stochastic differential equations helps us to generalize conventional
diffusion probabilistic models to the case of reconstructing data from noise
with different parameters and allows to make this reconstruction flexible by
explicitly controlling trade-off between sound quality and inference speed.
Subjective human evaluation shows that Grad-TTS is competitive with
state-of-the-art text-to-speech approaches in terms of Mean Opinion Score. We
will make the code publicly available shortly.

    

### [[2105.11804] Bridging Few-Shot Learning and Adaptation: New Challenges of Support-Query Shift](http://arxiv.org/abs/2105.11804)


  Few-Shot Learning (FSL) algorithms have made substantial progress in learning
novel concepts with just a handful of labelled data. To classify query
instances from novel classes encountered at test-time, they only require a
support set composed of a few labelled samples. FSL benchmarks commonly assume
that those queries come from the same distribution as instances in the support
set. However, in a realistic set-ting, data distribution is plausibly subject
to change, a situation referred to as Distribution Shift (DS). The present work
addresses the new and challenging problem of Few-Shot Learning under
Support/Query Shift (FSQS) i.e., when support and query instances are sampled
from related but different distributions. Our contributions are the following.
First, we release a testbed for FSQS, including datasets, relevant baselines
and a protocol for a rigorous and reproducible evaluation. Second, we observe
that well-established FSL algorithms unsurprisingly suffer from a considerable
drop in accuracy when facing FSQS, stressing the significance of our study.
Finally, we show that transductive algorithms can limit the inopportune effect
of DS. In particular, we study both the role of Batch-Normalization and Optimal
Transport (OT) in aligning distributions, bridging Unsupervised Domain
Adaptation with FSL. This results in a new method that efficiently combines OT
with the celebrated Prototypical Networks. We bring compelling experiments
demonstrating the advantage of our method. Our work opens an exciting line of
research by providing a testbed and strong baselines. Our code is available at
this https URL.

    

### [[2106.00001] Privately Learning Subspaces](http://arxiv.org/abs/2106.00001)


  Private data analysis suffers a costly curse of dimensionality. However, the
data often has an underlying low-dimensional structure. For example, when
optimizing via gradient descent, the gradients often lie in or near a
low-dimensional subspace. If that low-dimensional structure can be identified,
then we can avoid paying (in terms of privacy or accuracy) for the high ambient
dimension.
We present differentially private algorithms that take input data sampled
from a low-dimensional linear subspace (possibly with a small amount of error)
and output that subspace (or an approximation to it). These algorithms can
serve as a pre-processing step for other procedures.

    

### [[2106.08365] Predicting Unreliable Predictions by Shattering a Neural Network](http://arxiv.org/abs/2106.08365)


  Piecewise linear neural networks can be split into subfunctions, each with
its own activation pattern, domain, and empirical error. Empirical error for
the full network can be written as an expectation over empirical error of
subfunctions. Constructing a generalization bound on subfunction empirical
error indicates that the more densely a subfunction is surrounded by training
samples in representation space, the more reliable its predictions are.
Further, it suggests that models with fewer activation regions generalize
better, and models that abstract knowledge to a greater degree generalize
better, all else equal. We propose not only a theoretical framework to reason
about subfunction error bounds but also a pragmatic way of approximately
evaluating it, which we apply to predicting which samples the network will not
successfully generalize to. We test our method on detection of
misclassification and out-of-distribution samples, finding that it performs
competitively in both cases. In short, some network activation patterns are
associated with higher reliability than others, and these can be identified
using subfunction error bounds.

    

### [[2106.12699] Distilling the Knowledge from Conditional Normalizing Flows](http://arxiv.org/abs/2106.12699)


  Normalizing flows are a powerful class of generative models demonstrating
strong performance in several speech and vision problems. In contrast to other
generative models, normalizing flows are latent variable models with tractable
likelihoods and allow for stable training. However, they have to be carefully
designed to represent invertible functions with efficient Jacobian determinant
calculation. In practice, these requirements lead to overparameterized and
sophisticated architectures that are inferior to alternative feed-forward
models in terms of inference time and memory consumption. In this work, we
investigate whether one can distill flow-based models into more efficient
alternatives. We provide a positive answer to this question by proposing a
simple distillation approach and demonstrating its effectiveness on
state-of-the-art conditional flow-based models for image super-resolution and
speech synthesis.

    

### [[2107.00116] Robust Generative Adversarial Imitation Learning via Local Lipschitzness](http://arxiv.org/abs/2107.00116)


  We explore methodologies to improve the robustness of generative adversarial
imitation learning (GAIL) algorithms to observation noise. Towards this
objective, we study the effect of local Lipschitzness of the discriminator and
the generator on the robustness of policies learned by GAIL. In many robotics
applications, the learned policies by GAIL typically suffer from a degraded
performance at test time since the observations from the environment might be
corrupted by noise. Hence, robustifying the learned policies against the
observation noise is of critical importance. To this end, we propose a
regularization method to induce local Lipschitzness in the generator and the
discriminator of adversarial imitation learning methods. We show that the
modified objective leads to learning significantly more robust policies.
Moreover, we demonstrate --- both theoretically and experimentally --- that
training a locally Lipschitz discriminator leads to a locally Lipschitz
generator, thereby improving the robustness of the resultant policy. We perform
extensive experiments on simulated robot locomotion environments from the
MuJoCo suite that demonstrate the proposed method learns policies that
significantly outperform the state-of-the-art generative adversarial imitation
learning algorithm when applied to test scenarios with noise-corrupted
observations.

    

### [[2007.02156] On spectral algorithms for community detection in stochastic blockmodel graphs with vertex covariates](http://arxiv.org/abs/2007.02156)


  In network inference applications, it is often desirable to detect community
structure, namely to cluster vertices into groups, or blocks, according to some
measure of similarity. Beyond mere adjacency matrices, many real networks also
involve vertex covariates that carry key information about underlying block
structure in graphs. To assess the effects of such covariates on block
recovery, we present a comparative analysis of two model-based spectral
algorithms for clustering vertices in stochastic blockmodel graphs with vertex
covariates. The first algorithm uses only the adjacency matrix, and directly
estimates the block assignments. The second algorithm incorporates both the
adjacency matrix and the vertex covariates into the estimation of block
assignments, and moreover quantifies the explicit impact of the vertex
covariates on the resulting estimate of the block assignments. We employ
Chernoff information to analytically compare the algorithms' performance and
derive the information-theoretic Chernoff ratio for certain models of interest.
Analytic results and simulations suggest that the second algorithm is often
preferred: we can often better estimate the induced block assignments by first
estimating the effect of vertex covariates. In addition, real data examples
also indicate that the second algorithm has the advantages of revealing
underlying block structure and taking observed vertex heterogeneity into
account in real applications. Our findings emphasize the importance of
distinguishing between observed and unobserved factors that can affect block
structure in graphs.

    

### [[2108.01403] Nonperturbative renormalization for the neural network-QFT correspondence](http://arxiv.org/abs/2108.01403)


  In a recent work arXiv:2008.08601, Halverson, Maiti and Stoner proposed a
description of neural networks in terms of a Wilsonian effective field theory.
The infinite-width limit is mapped to a free field theory, while finite $N$
corrections are taken into account by interactions (non-Gaussian terms in the
action). In this paper, we study two related aspects of this correspondence.
First, we comment on the concepts of locality and power-counting in this
context. Indeed, these usual space-time notions may not hold for neural
networks (since inputs can be arbitrary), however, the renormalization group
provides natural notions of locality and scaling. Moreover, we comment on
several subtleties, for example, that data components may not have a
permutation symmetry: in that case, we argue that random tensor field theories
could provide a natural generalization. Second, we improve the perturbative
Wilsonian renormalization from arXiv:2008.08601 by providing an analysis in
terms of the nonperturbative renormalization group using the Wetterich-Morris
equation. An important difference with usual nonperturbative RG analysis is
that only the effective (IR) 2-point function is known, which requires setting
the problem with care. Our aim is to provide a useful formalism to investigate
neural networks behavior beyond the large-width limit (i.e.~far from Gaussian
limit) in a nonperturbative fashion. A major result of our analysis is that
changing the standard deviation of the neural network weight distribution can
be interpreted as a renormalization flow in the space of networks. We focus on
translations invariant kernels and provide preliminary numerical results.

    

### [[2108.02328] A Distributed Application Placement and Migration Management Techniques for Edge and Fog Computing Environments](http://arxiv.org/abs/2108.02328)


  Fog/Edge computing model allows harnessing of resources in the proximity of
the Internet of Things (IoT) devices to support various types of real-time IoT
applications. However, due to the mobility of users and a wide range of IoT
applications with different requirements, it is a challenging issue to satisfy
these applications' requirements. The execution of IoT applications exclusively
on one fog/edge server may not be always feasible due to limited resources,
while execution of IoT applications on different servers needs further
collaboration among servers. Also, considering user mobility, some modules of
each IoT application may require migration to other servers for execution,
leading to service interruption and extra execution costs. In this article, we
propose a new weighted cost model for hierarchical fog computing environments,
in terms of the response time of IoT applications and energy consumption of IoT
devices, to minimize the cost of running IoT applications and potential
migrations. Besides, a distributed clustering technique is proposed to enable
the collaborative execution of tasks, emitted from application modules, among
servers. Also, we propose an application placement technique to minimize the
overall cost of executing IoT applications on multiple servers in a distributed
manner. Furthermore, a distributed migration management technique is proposed
for the potential migration of applications' modules to other remote servers as
the users move along their path. Besides, failure recovery methods are embedded
in the clustering, application placement, and migration management techniques
to recover from unpredicted failures. The performance results show that our
technique significantly improves its counterparts in terms of placement
deployment time, average execution cost of tasks, total number of migrations,
total number of interrupted tasks, and cumulative migration cost.

    

### [[2108.02558] JITA4DS: Disaggregated execution of Data Science Pipelines between the Edge and the Data Centre](http://arxiv.org/abs/2108.02558)


  This paper targets the execution of data science (DS) pipelines supported by
data processing, transmission and sharing across several resources executing
greedy processes. Current data science pipelines environments provide various
infrastructure services with computing resources such as general-purpose
processors (GPP), Graphics Processing Units (GPUs), Field Programmable Gate
Arrays (FPGAs) and Tensor Processing Unit (TPU) coupled with platform and
software services to design, run and maintain DS pipelines. These one-fits-all
solutions impose the complete externalization of data pipeline tasks. However,
some tasks can be executed in the edge, and the backend can provide just in
time resources to ensure ad-hoc and elastic execution environments.
This paper introduces an innovative composable "Just in Time Architecture"
for configuring DCs for Data Science Pipelines (JITA-4DS) and associated
resource management techniques. JITA-4DS is a cross-layer management system
that is aware of both the application characteristics and the underlying
infrastructures to break the barriers between applications,
middleware/operating system, and hardware layers. Vertical integration of these
layers is needed for building a customizable Virtual Data Center (VDC) to meet
the dynamically changing data science pipelines' requirements such as
performance, availability, and energy consumption. Accordingly, the paper shows
an experimental simulation devoted to run data science workloads and determine
the best strategies for scheduling the allocation of resources implemented by
JITA-4DS.

    

### [[2108.02582] An Abstract View of Big Data Processing Programs](http://arxiv.org/abs/2108.02582)


  This paper proposes a model for specifying data flow based parallel data
processing programs agnostic of target Big Data processing frameworks. The
paper focuses on the formal abstract specification of non-iterative and
iterative programs, generalizing the strategies adopted by data flow Big Data
processing frameworks. The proposed model relies on monoid AlgebraandPetri
Netstoabstract Big Data processing programs in two levels: a high level
representing the program data flow and a lower level representing data
transformation operations (e.g., filtering, aggregation, join). We extend the
model for data processing programs proposed in [1], to enable the use of
iterative programs. The general specification of iterative data processing
programs implemented by data flow-based parallel programming models is
essential given the democratization of iterative and greedy Big Data analytics
algorithms. Indeed, these algorithms call for revisiting parallel programming
models to express iterations. The paper gives a comparative analysis of the
iteration strategies proposed byApache Spark, DryadLINQ, Apache Beam and Apache
Flink. It discusses how the model achieves to generalize these strategies.

    

### [[2108.02589] TRANSMUT-SPARK: Transformation Mutation for Apache Spark](http://arxiv.org/abs/2108.02589)


  We propose TRANSMUT-Spark, a tool that automates the mutation testing process
of Big Data processing code within Spark programs. Apache Spark is an engine
for Big Data Processing. It hides the complexity inherent to Big Data parallel
and distributed programming and processing through built-in functions,
underlying parallel processes, and data management strategies. Nonetheless,
programmers must cleverly combine these functions within programs and guide the
engine to use the right data management strategies to exploit the large number
of computational resources required by Big Data processing and avoid
substantial production losses. Many programming details in data processing code
within Spark programs are prone to false statements that need to be correctly
and automatically tested. This paper explores the application of mutation
testing in Spark programs, a fault-based testing technique that relies on fault
simulation to evaluate and design test sets. The paper introduces the
TRANSMUT-Spark solution for testing Spark programs. TRANSMUT-Spark automates
the most laborious steps of the process and fully executes the mutation testing
process. The paper describes how the tool automates the mutants generation,
test execution, and adequacy analysis phases of mutation testing with
TRANSMUT-Spark. It also discusses the results of experiments that were carried
out to validate the tool to argue its scope and limitations.

    

### [[2108.02638] Efficient CONGEST Algorithms for the Lovasz Local Lemma](http://arxiv.org/abs/2108.02638)


  We present a poly $\log \log n$ time randomized CONGEST algorithm for a
natural class of Lovasz Local Lemma (LLL) instances on constant degree graphs.
This implies, among other things, that there are no LCL problems with
randomized complexity between $\log n$ and poly $\log \log n$. Furthermore, we
provide extensions to the network decomposition algorithms given in the recent
breakthrough by Rozhon and Ghaffari [STOC2020] and the follow up by Ghaffari,
Grunau, and Rozhon [SODA2021]. In particular, we show how to obtain a large
distance separated weak network decomposition with a negligible dependency on
the range of unique identifiers.

    

### [[2108.02655] Sinkless orientation is hard also in the supported LOCAL model](http://arxiv.org/abs/2108.02655)


  We show that any algorithm that solves the sinkless orientation problem in
the supported LOCAL model requires $\Omega(\log n)$ rounds, and this is tight.
The supported LOCAL is at least as strong as the usual LOCAL model, and as a
corollary this also gives a new, short and elementary proof that shows that the
round complexity of the sinkless orientation problem in the deterministic LOCAL
model is $\Omega(\log n)$.

    

### [[2108.02692] Accelerating XOR-based Erasure Coding using Program Optimization Techniques](http://arxiv.org/abs/2108.02692)


  Erasure coding (EC) affords data redundancy for large-scale systems.
XOR-based EC is an easy-to-implement method for optimizing EC. This paper
addresses a significant performance gap between the state-of-the-art XOR-based
EC approach (with 4.9 GB/s coding throughput) and Intel's high-performance EC
library based on another approach (with 6.7 GB/s). We propose a novel approach
based on our observation that XOR-based EC virtually generates programs of a
Domain Specific Language for XORing byte arrays. We formalize such programs as
straight-line programs (SLPs) of compiler construction and optimize SLPs using
various optimization techniques. Our optimization flow is three-fold: 1)
reducing operations using grammar compression algorithms; 2) reducing memory
accesses using deforestation, a functional program optimization method; and 3)
reducing cache misses using the (red-blue) pebble game of program analysis. We
provide an experimental library, which outperforms Intel's library with 8.92
GB/s throughput.

    

### [[2108.02697] A tight local algorithm for the minimum dominating set problem in outerplanar graphs](http://arxiv.org/abs/2108.02697)


  We show that there is a deterministic local algorithm (constant-time
distributed graph algorithm) that finds a 5-approximation of a minimum
dominating set on outerplanar graphs. We show there is no such algorithm that
finds a $(5-\varepsilon)$-approximation, for any $\varepsilon>0$. Our algorithm
only requires knowledge of the degree of a vertex and of its neighbors, so that
large messages and unique identifiers are not needed.

    

### [[2108.02763] Crystalline: Fast and Memory Efficient Wait-Free Reclamation](http://arxiv.org/abs/2108.02763)


  Historically, memory management based on lock-free reference counting was
very inefficient, especially for read-dominated workloads. Thus, approaches
such as epoch-based reclamation (EBR), hazard pointers (HP), or a combination
thereof have received significant attention. EBR exhibits excellent performance
but is blocking due to potentially unbounded memory usage. In contrast, HP are
non-blocking and achieve good memory efficiency but are much slower. Moreover,
HP are only lock-free in the general case. Recently, several new memory
reclamation approaches such as WFE and Hyaline have been proposed. WFE achieves
wait-freedom, but is less memory efficient and suffers from suboptimal
performance in oversubscribed scenarios; Hyaline achieves higher performance
and memory efficiency, but lacks wait-freedom.
We present a new wait-free memory reclamation scheme, Crystalline, that
simultaneously addresses the challenges of high performance, high memory
efficiency, and wait-freedom. Crystalline guarantees complete wait-freedom even
when threads are dynamically recycled, asynchronously reclaims memory in the
sense that any thread can reclaim memory retired by any other thread, and
ensures (an almost) balanced reclamation workload across all threads. The
latter two properties result in Crystalline's high performance and high memory
efficiency. Simultaneously ensuring all three properties require overcoming
unique challenges which we discuss in the paper.
Crystalline's implementation relies on specialized instructions which are
widely available on commodity hardware such as x86-64 or ARM64. Our
experimental evaluations show that Crystalline exhibits outstanding scalability
and memory efficiency, and achieves superior throughput than typical
reclamation schemes such as EBR as the number of threads grows.

    

### [[2108.02775] Space and Time Bounded Multiversion Garbage Collection](http://arxiv.org/abs/2108.02775)


  We present a general technique for garbage collecting old versions for
multi-version concurrency control that simultaneously achieves good time and
space complexity. Our technique takes only $O(1)$ time on average for each new
version and maintains only a constant factor more than then number of needed
versions. Our technique is designed for multi-version schemes using version
lists, which are the most common.
Our approach uses two components that are of independent interest. First, we
define a novel range-tracking data structure which stores a set of old versions
and efficiently finds those that are no longer needed. We provide a wait-free
implementation in which all operations take amortized constant time. Second, we
represent version lists using a new lock-free doubly-linked list algorithm that
supports efficient (amortized constant time) removals given a pointer to any
node in the list. These two components naturally fit together to solve the
multiversion garbage collection problem--the range-tracker identifies which
versions to remove and the list algorithm splices them out. We apply our
garbage collection technique to generate end-to-end time and space bounds for
the multiversioning system of Wei et al. (PPoPP 2021).

    

### [[2002.06993] In Search for an Optimal Authenticated Byzantine Agreement](http://arxiv.org/abs/2002.06993)


  In this paper, we challenge the conventional approach of state machine
replication systems to design deterministic agreement protocols in the
eventually synchronous communication model. We first prove that no such
protocol can guarantee bounded communication cost before the global
stabilization time and propose a different approach that hopes for the best
(synchrony) but prepares for the worst (asynchrony). Accordingly, we design an
optimistic byzantine agreement protocol that first tries an efficient
deterministic algorithm that relies on synchrony for termination only, and
then, only if an agreement was not reached due to asynchrony, the protocol uses
a randomized asynchronous protocol for fallback that guarantees termination
with probability 1.
We formally prove that our protocol achieves optimal communication complexity
under all network conditions and failure scenarios. We first prove a lower
bound of $\Omega(ft+ t)$ for synchronous deterministic byzantine agreement
protocols, where $t$ is the failure threshold, and $f$ is the actual number of
failures. Then, we present a tight upper bound and use it for the synchronous
part of the optimistic protocol. Finally, for the asynchronous fallback, we use
a variant of the (optimal) VABA protocol, which we reconstruct to safely
combine it with the synchronous part.
We believe that our adaptive to failures synchronous byzantine agreement
protocol has an independent interest since it is the first protocol we are
aware of which communication complexity optimally depends on the actual number
of failures.

    

### [[2003.11859] Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience](http://arxiv.org/abs/2003.11859)


  The brain exhibits capabilities of fast incremental learning from few noisy
examples, as well as the ability to associate similar memories in
autonomously-created categories and to combine contextual hints with sensory
perceptions. Together with sleep, these mechanisms are thought to be key
components of many high-level cognitive functions. Yet, little is known about
the underlying processes and the specific roles of different brain states. In
this work, we exploited the combination of context and perception in a
thalamo-cortical model based on a soft winner-take-all circuit of excitatory
and inhibitory spiking neurons. After calibrating this model to express awake
and deep-sleep states with features comparable with biological measures, we
demonstrate the model capability of fast incremental learning from few
examples, its resilience when proposed with noisy perceptions and contextual
signals, and an improvement in visual classification after sleep due to induced
synaptic homeostasis and association of similar memories.

    

### [[2108.02214] A FAIR and AI-ready Higgs Boson Decay Dataset](http://arxiv.org/abs/2108.02214)


  To enable the reusability of massive scientific datasets by humans and
machines, researchers aim to create scientific datasets that adhere to the
principles of findability, accessibility, interoperability, and reusability
(FAIR) for data and artificial intelligence (AI) models. This article provides
a domain-agnostic, step-by-step assessment guide to evaluate whether or not a
given dataset meets each FAIR principle. We then demonstrate how to use this
guide to evaluate the FAIRness of an open simulated dataset produced by the CMS
Collaboration at the CERN Large Hadron Collider. This dataset consists of Higgs
boson decays and quark and gluon background, and is available through the CERN
Open Data Portal. We also use other available tools to assess the FAIRness of
this dataset, and incorporate feedback from members of the FAIR community to
validate our results. This article is accompanied by a Jupyter notebook to
facilitate an understanding and exploration of the dataset, including
visualization of its elements. This study marks the first in a planned series
of articles that will guide scientists in the creation and quantification of
FAIRness in high energy particle physics datasets and AI models.

    

### [[2108.02257] Indoor Localization Under Limited Measurements: A Cross-Environment Joint Semi-Supervised and Transfer Learning Approach](http://arxiv.org/abs/2108.02257)


  The development of highly accurate deep learning methods for indoor
localization is often hindered by the unavailability of sufficient data
measurements in the desired environment to perform model training. To overcome
the challenge of collecting costly measurements, this paper proposes a
cross-environment approach that compensates for insufficient labelled
measurements via a joint semi-supervised and transfer learning technique to
transfer, in an appropriate manner, the model obtained from a rich-data
environment to the desired environment for which data is limited. This is
achieved via a sequence of operations that exploit the similarity across
environments to enhance unlabelled data model training of the desired
environment. Numerical experiments demonstrate that the proposed
cross-environment approach outperforms the conventional method, convolutional
neural network (CNN), with a significant increase in localization accuracy, up
to 43%. Moreover, with only 40% data measurements, the proposed
cross-environment approach compensates for data inadequacy and replicates the
localization accuracy of the conventional method, CNN, which uses 75% data
measurements.

    

### [[2108.02278] Pan-Cancer Integrative Histology-Genomic Analysis via Interpretable Multimodal Deep Learning](http://arxiv.org/abs/2108.02278)


  The rapidly emerging field of deep learning-based computational pathology has
demonstrated promise in developing objective prognostic models from histology
whole slide images. However, most prognostic models are either based on
histology or genomics alone and do not address how histology and genomics can
be integrated to develop joint image-omic prognostic models. Additionally
identifying explainable morphological and molecular descriptors from these
models that govern such prognosis is of interest. We used multimodal deep
learning to integrate gigapixel whole slide pathology images, RNA-seq
abundance, copy number variation, and mutation data from 5,720 patients across
14 major cancer types. Our interpretable, weakly-supervised, multimodal deep
learning algorithm is able to fuse these heterogeneous modalities for
predicting outcomes and discover prognostic features from these modalities that
corroborate with poor and favorable outcomes via multimodal interpretability.
We compared our model with unimodal deep learning models trained on histology
slides and molecular profiles alone, and demonstrate performance increase in
risk stratification on 9 out of 14 cancers. In addition, we analyze morphologic
and molecular markers responsible for prognostic predictions across all cancer
types. All analyzed data, including morphological and molecular correlates of
patient prognosis across the 14 cancer types at a disease and patient level are
presented in an interactive open-access database
(this http URL) to allow for further exploration and
prognostic biomarker discovery. To validate that these model explanations are
prognostic, we further analyzed high attention morphological regions in WSIs,
which indicates that tumor-infiltrating lymphocyte presence corroborates with
favorable cancer prognosis on 9 out of 14 cancer types studied.

    

### [[2108.02352] Understand me, if you refer to Aspect Knowledge: Knowledge-aware Gated Recurrent Memory Network](http://arxiv.org/abs/2108.02352)


  Aspect-level sentiment classification (ASC) aims to predict the fine-grained
sentiment polarity towards a given aspect mentioned in a review. Despite recent
advances in ASC, enabling machines to preciously infer aspect sentiments is
still challenging. This paper tackles two challenges in ASC: (1) due to lack of
aspect knowledge, aspect representation derived in prior works is inadequate to
represent aspect's exact meaning and property information; (2) prior works only
capture either local syntactic information or global relational information,
thus missing either one of them leads to insufficient syntactic information. To
tackle these challenges, we propose a novel ASC model which not only end-to-end
embeds and leverages aspect knowledge but also marries the two kinds of
syntactic information and lets them compensate for each other. Our model
includes three key components: (1) a knowledge-aware gated recurrent memory
network recurrently integrates dynamically summarized aspect knowledge; (2) a
dual syntax graph network combines both kinds of syntactic information to
comprehensively capture sufficient syntactic information; (3) a knowledge
integrating gate re-enhances the final representation with further needed
aspect knowledge; (4) an aspect-to-context attention mechanism aggregates the
aspect-related semantics from all hidden states into the final representation.
Experimental results on several benchmark datasets demonstrate the
effectiveness of our model, which overpass previous state-of-the-art models by
large margins in terms of both Accuracy and Macro-F1.

    

### [[2108.02356] Video Abnormal Event Detection by Learning to Complete Visual Cloze Tests](http://arxiv.org/abs/2108.02356)


  Video abnormal event detection (VAD) is a vital semi-supervised task that
requires learning with only roughly labeled normal videos, as anomalies are
often practically unavailable. Although deep neural networks (DNNs) enable
great progress in VAD, existing solutions typically suffer from two issues: (1)
The precise and comprehensive localization of video events is ignored. (2) The
video semantics and temporal context are under-explored. To address those
issues, we are motivated by the prevalent cloze test in education and propose a
novel approach named visual cloze completion (VCC), which performs VAD by
learning to complete "visual cloze tests" (VCTs). Specifically, VCC first
localizes each video event and encloses it into a spatio-temporal cube (STC).
To achieve both precise and comprehensive localization, appearance and motion
are used as mutually complementary cues to mark the object region associated
with each video event. For each marked region, a normalized patch sequence is
extracted from temporally adjacent frames and stacked into the STC. By
comparing each patch and the patch sequence of a STC to a visual "word" and
"sentence" respectively, we can deliberately erase a certain "word" (patch) to
yield a VCT. DNNs are then trained to infer the erased patch by video
semantics, so as to complete the VCT. To fully exploit the temporal context,
each patch in STC is alternatively erased to create multiple VCTs, and the
erased patch's optical flow is also inferred to integrate richer motion clues.
Meanwhile, a new DNN architecture is designed as a model-level solution to
utilize video semantics and temporal context. Extensive experiments demonstrate
that VCC achieves state-of-the-art VAD performance. Our codes and results are
open at \url{this https URL}

    

### [[2108.02388] TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding](http://arxiv.org/abs/2108.02388)


  Recently proposed fine-grained 3D visual grounding is an essential and
challenging task, whose goal is to identify the 3D object referred by a natural
language sentence from other distractive objects of the same category. Existing
works usually adopt dynamic graph networks to indirectly model the
intra/inter-modal interactions, making the model difficult to distinguish the
referred object from distractors due to the monolithic representations of
visual and linguistic contents. In this work, we exploit Transformer for its
natural suitability on permutation-invariant 3D point clouds data and propose a
TransRefer3D network to extract entity-and-relation aware multimodal context
among objects for more discriminative feature learning. Concretely, we devise
an Entity-aware Attention (EA) module and a Relation-aware Attention (RA)
module to conduct fine-grained cross-modal feature matching. Facilitated by
co-attention operation, our EA module matches visual entity features with
linguistic entity features while RA module matches pair-wise visual relation
features with linguistic relation features, respectively. We further integrate
EA and RA modules into an Entity-and-Relation aware Contextual Block (ERCB) and
stack several ERCBs to form our TransRefer3D for hierarchical multimodal
context modeling. Extensive experiments on both Nr3D and Sr3D datasets
demonstrate that our proposed model significantly outperforms existing
approaches by up to 10.6% and claims the new state-of-the-art. To the best of
our knowledge, this is the first work investigating Transformer architecture
for fine-grained 3D visual grounding task.

    

### [[2108.02401] WeChat Neural Machine Translation Systems for WMT21](http://arxiv.org/abs/2108.02401)


  This paper introduces WeChat AI's participation in WMT 2021 shared news
translation task on English->Chinese, English->Japanese, Japanese->English and
English->German. Our systems are based on the Transformer (Vaswani et al.,
2017) with several novel and effective variants. In our experiments, we employ
data filtering, large-scale synthetic data generation (i.e., back-translation,
knowledge distillation, forward-translation, iterative in-domain knowledge
transfer), advanced finetuning approaches, and boosted Self-BLEU based model
ensemble. Our constrained systems achieve 36.9, 46.9, 27.8 and 31.3
case-sensitive BLEU scores on English->Chinese, English->Japanese,
Japanese->English and English->German, respectively. The BLEU scores of
English->Chinese, English->Japanese and Japanese->English are the highest among
all submissions, and that of English->German is the highest among all
constrained submissions.

    

### [[2108.02446] Finetuning Pretrained Transformers into Variational Autoencoders](http://arxiv.org/abs/2108.02446)


  Text variational autoencoders (VAEs) are notorious for posterior collapse, a
phenomenon where the model's decoder learns to ignore signals from the encoder.
Because posterior collapse is known to be exacerbated by expressive decoders,
Transformers have seen limited adoption as components of text VAEs. Existing
studies that incorporate Transformers into text VAEs (Li et al., 2020; Fang et
al., 2021) mitigate posterior collapse using massive pretraining, a technique
unavailable to most of the research community without extensive computing
resources. We present a simple two-phase training scheme to convert a
sequence-to-sequence Transformer into a VAE with just finetuning. The resulting
language model is competitive with massively pretrained Transformer-based VAEs
in some internal metrics while falling short on others. To facilitate training
we comprehensively explore the impact of common posterior collapse alleviation
techniques in the literature. We release our code for reproducability.

    

### [[2108.02448] MFuseNet: Robust Depth Estimation with Learned Multiscopic Fusion](http://arxiv.org/abs/2108.02448)


  We design a multiscopic vision system that utilizes a low-cost monocular RGB
camera to acquire accurate depth estimation. Unlike multi-view stereo with
images captured at unconstrained camera poses, the proposed system controls the
motion of a camera to capture a sequence of images in horizontally or
vertically aligned positions with the same parallax. In this system, we propose
a new heuristic method and a robust learning-based method to fuse multiple cost
volumes between the reference image and its surrounding images. To obtain
training data, we build a synthetic dataset with multiscopic images. The
experiments on the real-world Middlebury dataset and real robot demonstration
show that our multiscopic vision system outperforms traditional two-frame
stereo matching methods in depth estimation. Our code and dataset are available
at \url{this https URL


### [[2108.02451] Unifying Nonlocal Blocks for Neural Networks](http://arxiv.org/abs/2108.02451)


  The nonlocal-based blocks are designed for capturing long-range
spatial-temporal dependencies in computer vision tasks. Although having shown
excellent performance, they still lack the mechanism to encode the rich,
structured information among elements in an image or video. In this paper, to
theoretically analyze the property of these nonlocal-based blocks, we provide a
new perspective to interpret them, where we view them as a set of graph filters
generated on a fully-connected graph. Specifically, when choosing the Chebyshev
graph filter, a unified formulation can be derived for explaining and analyzing
the existing nonlocal-based blocks (e.g., nonlocal block, nonlocal stage,
double attention block). Furthermore, by concerning the property of spectral,
we propose an efficient and robust spectral nonlocal block, which can be more
robust and flexible to catch long-range dependencies when inserted into deep
neural networks than the existing nonlocal blocks. Experimental results
demonstrate the clear-cut improvements and practical applicabilities of our
method on image classification, action recognition, semantic segmentation, and
person re-identification tasks.

    

### [[2108.02455] LSENet: Location and Seasonality Enhanced Network for Multi-Class Ocean Front Detection](http://arxiv.org/abs/2108.02455)


  Ocean fronts can cause the accumulation of nutrients and affect the
propagation of underwater sound, so high-precision ocean front detection is of
great significance to the marine fishery and national defense fields. However,
the current ocean front detection methods either have low detection accuracy or
most can only detect the occurrence of ocean front by binary classification,
rarely considering the differences of the characteristics of multiple ocean
fronts in different sea areas. In order to solve the above problems, we propose
a semantic segmentation network called location and seasonality enhanced
network (LSENet) for multi-class ocean fronts detection at pixel level. In this
network, we first design a channel supervision unit structure, which integrates
the seasonal characteristics of the ocean front itself and the contextual
information to improve the detection accuracy. We also introduce a location
attention mechanism to adaptively assign attention weights to the fronts
according to their frequently occurred sea area, which can further improve the
accuracy of multi-class ocean front detection. Compared with other semantic
segmentation methods and current representative ocean front detection method,
the experimental results demonstrate convincingly that our method is more
effective.

    

### [[2108.02502] Imperceptible Adversarial Examples by Spatial Chroma-Shift](http://arxiv.org/abs/2108.02502)


  Deep Neural Networks have been shown to be vulnerable to various kinds of
adversarial perturbations. In addition to widely studied additive noise based
perturbations, adversarial examples can also be created by applying a per pixel
spatial drift on input images. While spatial transformation based adversarial
examples look more natural to human observers due to absence of additive noise,
they still possess visible distortions caused by spatial transformations. Since
the human vision is more sensitive to the distortions in the luminance compared
to those in chrominance channels, which is one of the main ideas behind the
lossy visual multimedia compression standards, we propose a spatial
transformation based perturbation method to create adversarial examples by only
modifying the color components of an input image. While having competitive
fooling rates on CIFAR-10 and NIPS2017 Adversarial Learning Challenge datasets,
examples created with the proposed method have better scores with regards to
various perceptual quality metrics. Human visual perception studies validate
that the examples are more natural looking and often indistinguishable from
their original counterparts.

    

### [[2108.02510] Improved Speech Emotion Recognition using Transfer Learning and Spectrogram Augmentation](http://arxiv.org/abs/2108.02510)


  Automatic speech emotion recognition (SER) is a challenging task that plays a
crucial role in natural human-computer interaction. One of the main challenges
in SER is data scarcity, i.e., insufficient amounts of carefully labeled data
to build and fully explore complex deep learning models for emotion
classification. This paper aims to address this challenge using a transfer
learning strategy combined with spectrogram augmentation. Specifically, we
propose a transfer learning approach that leverages a pre-trained residual
network (ResNet) model including a statistics pooling layer from speaker
recognition trained using large amounts of speaker-labeled data. The statistics
pooling layer enables the model to efficiently process variable-length input,
thereby eliminating the need for sequence truncation which is commonly used in
SER systems. In addition, we adopt a spectrogram augmentation technique to
generate additional training data samples by applying random time-frequency
masks to log-mel spectrograms to mitigate overfitting and improve the
generalization of emotion recognition models. We evaluate the effectiveness of
our proposed approach on the interactive emotional dyadic motion capture
(IEMOCAP) dataset. Experimental results indicate that the transfer learning and
spectrogram augmentation approaches improve the SER performance, and when
combined achieve state-of-the-art results.

    

### [[2108.02524] Bambara Language Dataset for Sentiment Analysis](http://arxiv.org/abs/2108.02524)


  For easier communication, posting, or commenting on each others posts, people
use their dialects. In Africa, various languages and dialects exist. However,
they are still underrepresented and not fully exploited for analytical studies
and research purposes. In order to perform approaches like Machine Learning and
Deep Learning, datasets are required. One of the African languages is Bambara,
used by citizens in different countries. However, no previous work on datasets
for this language was performed for Sentiment Analysis. In this paper, we
present the first common-crawl-based Bambara dialectal dataset dedicated for
Sentiment Analysis, available freely for Natural Language Processing research
purposes.

    

### [[2108.02547] Fairer Chess: A Reversal of Two Opening Moves in Chess Creates Balance Between White and Black](http://arxiv.org/abs/2108.02547)


  Unlike tic-tac-toe or checkers, in which optimal play leads to a draw, it is
not known whether optimal play in chess ends in a win for White, a win for
Black, or a draw. But after White moves first in chess, if Black has a double
move followed by a double move of White and then alternating play, play is more
balanced because White does not always tie or lead in moves. Symbolically,
Balanced Alternation gives the following move sequence: After White's (W)
initial move, first Black (B) and then White each have two moves in a row
(BBWW), followed by the alternating sequence, beginning with W, which
altogether can be written as WB/BW/WB/WB/WB... (the slashes separate
alternating pairs of moves). Except for reversal of the 3rd and 4th moves from
WB to BW, this is the standard chess sequence. Because Balanced Alternation
lies between the standard sequence, which favors White, and a comparable
sequence that favors Black, it is highly likely to produce a draw with optimal
play, rendering chess fairer. This conclusion is supported by a computer
analysis of chess openings and how they would play out under Balanced
Alternation.

    

### [[2108.02605] EENLP: Cross-lingual Eastern European NLP Index](http://arxiv.org/abs/2108.02605)


  This report presents the results of the EENLP project, done as a part of EEML
2021 summer school.
It presents a broad index of NLP resources for Eastern European languages,
which, we hope, could be helpful for the NLP community; several new
hand-crafted cross-lingual datasets focused on Eastern European languages, and
a sketch evaluation of cross-lingual transfer learning abilities of several
modern multilingual Transformer-based models.

    

### [[2108.02613] Planning with Learned Dynamic Model for Unsupervised Point Cloud Registration](http://arxiv.org/abs/2108.02613)


  Point cloud registration is a fundamental problem in 3D computer vision. In
this paper, we cast point cloud registration into a planning problem in
reinforcement learning, which can seek the transformation between the source
and target point clouds through trial and error. By modeling the point cloud
registration process as a Markov decision process (MDP), we develop a latent
dynamic model of point clouds, consisting of a transformation network and
evaluation network. The transformation network aims to predict the new
transformed feature of the point cloud after performing a rigid transformation
(i.e., action) on it while the evaluation network aims to predict the alignment
precision between the transformed source point cloud and target point cloud as
the reward signal. Once the dynamic model of the point cloud is trained, we
employ the cross-entropy method (CEM) to iteratively update the planning policy
by maximizing the rewards in the point cloud registration process. Thus, the
optimal policy, i.e., the transformation between the source and target point
clouds, can be obtained via gradually narrowing the search space of the
transformation. Experimental results on ModelNet40 and 7Scene benchmark
datasets demonstrate that our method can yield good registration performance in
an unsupervised manner.

    

### [[2108.02618] Using a Collated Cybersecurity Dataset for Machine Learning and Artificial Intelligence](http://arxiv.org/abs/2108.02618)


  Artificial Intelligence (AI) and Machine Learning (ML) algorithms can support
the span of indicator-level, e.g. anomaly detection, to behavioral level cyber
security modeling and inference. This contribution is based on a dataset named
BRON which is amalgamated from public threat and vulnerability behavioral
sources. We demonstrate how BRON can support prediction of related threat
techniques and attack patterns. We also discuss other AI and ML uses of BRON to
exploit its behavioral knowledge.

    

### [[2108.02637] An ASP-based Solution to the Chemotherapy Treatment Scheduling problem](http://arxiv.org/abs/2108.02637)


  The problem of scheduling chemotherapy treatments in oncology clinics is a
complex problem, given that the solution has to satisfy (as much as possible)
several requirements such as the cyclic nature of chemotherapy treatment plans,
maintaining a constant number of patients, and the availability of resources,
e.g., treatment time, nurses, and drugs. At the same time, realizing a
satisfying schedule is of upmost importance for obtaining the best health
outcomes. In this paper we first consider a specific instance of the problem
which is employed in the San Martino Hospital in Genova, Italy, and present a
solution to the problem based on Answer Set Programming (ASP). Then, we enrich
the problem and the related ASP encoding considering further features often
employed in other hospitals, desirable also in S. Martino, and/or considered in
related papers. Results of an experimental analysis, conducted on the real data
provided by the San Martino Hospital, show that ASP is an effective solving
methodology also for this important scheduling problem. Under consideration for
acceptance in TPLP.

    

### [[2108.02656] A Computer-Aided Diagnosis System for Breast Pathology: A Deep Learning Approach with Model Interpretability from Pathological Perspective](http://arxiv.org/abs/2108.02656)


  Objective: We develop a computer-aided diagnosis (CAD) system using deep
learning approaches for lesion detection and classification on whole-slide
images (WSIs) with breast cancer. The deep features being distinguishing in
classification from the convolutional neural networks (CNN) are demonstrated in
this study to provide comprehensive interpretability for the proposed CAD
system using pathological knowledge. Methods: In the experiment, a total of 186
slides of WSIs were collected and classified into three categories:
Non-Carcinoma, Ductal Carcinoma in Situ (DCIS), and Invasive Ductal Carcinoma
(IDC). Instead of conducting pixel-wise classification into three classes
directly, we designed a hierarchical framework with the multi-view scheme that
performs lesion detection for region proposal at higher magnification first and
then conducts lesion classification at lower magnification for each detected
lesion. Results: The slide-level accuracy rate for three-category
classification reaches 90.8% (99/109) through 5-fold cross-validation and
achieves 94.8% (73/77) on the testing set. The experimental results show that
the morphological characteristics and co-occurrence properties learned by the
deep learning models for lesion classification are accordant with the clinical
rules in diagnosis. Conclusion: The pathological interpretability of the deep
features not only enhances the reliability of the proposed CAD system to gain
acceptance from medical specialists, but also facilitates the development of
deep learning frameworks for various tasks in pathology. Significance: This
paper presents a CAD system for pathological image analysis, which fills the
clinical requirements and can be accepted by medical specialists with providing
its interpretability from the pathological perspective.

    

### [[2108.02707] Fairness Properties of Face Recognition and Obfuscation Systems](http://arxiv.org/abs/2108.02707)


  The proliferation of automated facial recognition in various commercial and
government sectors has caused significant privacy concerns for individuals. A
recent and popular approach to address these privacy concerns is to employ
evasion attacks against the metric embedding networks powering facial
recognition systems. Face obfuscation systems generate imperceptible
perturbations, when added to an image, cause the facial recognition system to
misidentify the user. The key to these approaches is the generation of
perturbations using a pre-trained metric embedding network followed by their
application to an online system, whose model might be proprietary. This
dependence of face obfuscation on metric embedding networks, which are known to
be unfair in the context of facial recognition, surfaces the question of
demographic fairness -- \textit{are there demographic disparities in the
performance of face obfuscation systems?} To address this question, we perform
an analytical and empirical exploration of the performance of recent face
obfuscation systems that rely on deep embedding networks. We find that metric
embedding networks are demographically aware; they cluster faces in the
embedding space based on their demographic attributes. We observe that this
effect carries through to the face obfuscation systems: faces belonging to
minority groups incur reduced utility compared to those from majority groups.
For example, the disparity in average obfuscation success rate on the online
Face++ API can reach up to 20 percentage points. Further, for some demographic
groups, the average perturbation size increases by up to 17\% when choosing a
target identity belonging to a different demographic group versus the same
demographic group. Finally, we present a simple analytical model to provide
insights into these phenomena.

    

### [[2009.05774] Sequential Composition of Propositional Horn Theories](http://arxiv.org/abs/2009.05774)


  Rule-based reasoning is an essential part of human intelligence prominently
formalized in artificial intelligence research via Horn theories. Describing
complex objects as the composition of elementary ones is a common strategy in
computer science and science in general. Recently, the author introduced the
sequential composition of Horn logic programs for syntactic program composition
and decomposition in the context of logic-based analogical reasoning and
learning. This paper contributes to the foundations of logic programming,
knowledge representation, and database theory by studying the sequential
composition of propositional Horn theories. Specifically, we show that the
notion of composition gives rise to a family of finite magmas and algebras,
baptized {\em Horn magmas} and {\em Horn algebras} in this paper. On the
semantic side, we show that the van Emden-Kowalski immediate consequence
operator of a theory can be represented via composition, which allows us to
compute its least model semantics without any explicit reference to operators.
This bridges the conceptual gap between the syntax and semantics of a
propositional Horn theory in a mathematically satisfactory way. Moreover, it
gives rise to an algebraic meta-calculus for propositional Horn theories. In a
broader sense, this paper is a first step towards an algebra of rule-based
logical theories and in the future we plan to adapt and generalize the methods
of this paper to wider classes of theories, most importantly to first-, and
higher-order logic programs, and non-monotonic logic programs under the stable
model or answer set semantics and extensions thereof.

    

### [[2010.07079] Recipes for Safety in Open-domain Chatbots](http://arxiv.org/abs/2010.07079)


  Models trained on large unlabeled corpora of human interactions will learn
patterns and mimic behaviors therein, which include offensive or otherwise
toxic behavior and unwanted biases. We investigate a variety of methods to
mitigate these issues in the context of open-domain generative dialogue models.
We introduce a new human-and-model-in-the-loop framework for both training
safer models and for evaluating them, as well as a novel method to distill
safety considerations inside generative models without the use of an external
classifier at deployment time. We conduct experiments comparing these methods
and find our new techniques are (i) safer than existing models as measured by
automatic and human evaluations while (ii) maintaining usability metrics such
as engagingness relative to the state of the art. We then discuss the
limitations of this work by analyzing failure cases of our models.

    

### [[2011.01306] Pairwise Relations Discriminator for Unsupervised Raven's Progressive Matrices](http://arxiv.org/abs/2011.01306)


  The ability to hypothesise, develop abstract concepts based on concrete
observations and apply these hypotheses to justify future actions has been
paramount in human development. An existing line of research in outfitting
intelligent machines with abstract reasoning capabilities revolves around the
Raven's Progressive Matrices (RPM). There have been many breakthroughs in
supervised approaches to solving RPM in recent years. However, this process
requires external assistance, and thus it cannot be claimed that machines have
achieved reasoning ability comparable to humans. Namely, humans can solve RPM
problems without supervision or prior experience once the RPM rule that
relations can only exist row/column-wise is properly introduced. In this paper,
we introduce a pairwise relations discriminator (PRD), a technique to develop
unsupervised models with sufficient reasoning abilities to tackle an RPM
problem. PRD reframes the RPM problem into a relation comparison task, which we
can solve without requiring the labelling of the RPM problem. We can identify
the optimal candidate by adapting the application of PRD to the RPM problem.
Our approach, the PRD, establishes a new state-of-the-art unsupervised learning
benchmark with an accuracy of 55.9% on the I-RAVEN, presenting a significant
improvement and a step forward in equipping machines with abstract reasoning.

    

### [[2012.10171] Which Heroes to Pick? Learning to Draft in MOBA Games with Neural Networks and Tree Search](http://arxiv.org/abs/2012.10171)


  Hero drafting is essential in MOBA game playing as it builds the team of each
side and directly affects the match outcome. State-of-the-art drafting methods
fail to consider: 1) drafting efficiency when the hero pool is expanded; 2) the
multi-round nature of a MOBA 5v5 match series, i.e., two teams play best-of-N
and the same hero is only allowed to be drafted once throughout the series. In
this paper, we formulate the drafting process as a multi-round combinatorial
game and propose a novel drafting algorithm based on neural networks and
Monte-Carlo tree search, named JueWuDraft. Specifically, we design a long-term
value estimation mechanism to handle the best-of-N drafting case. Taking Honor
of Kings, one of the most popular MOBA games at present, as a running case, we
demonstrate the practicality and effectiveness of JueWuDraft when compared to
state-of-the-art drafting methods.

    

### [[2101.00737] Coreference Resolution: Are the eliminated spans totally worthless?](http://arxiv.org/abs/2101.00737)


  Various neural-based methods have been proposed so far for joint mention
detection and coreference resolution. However, existing works on coreference
resolution are mainly dependent on filtered mention representation, while other
spans are largely neglected. In this paper, we aim at increasing the
utilization rate of data and investigating whether those eliminated spans are
totally useless, or to what extent they can improve the performance of
coreference resolution. To achieve this, we propose a mention representation
refining strategy where spans highly related to mentions are well leveraged
using a pointer network for representation enhancing. Notably, we utilize an
additional loss term in this work to encourage the diversity between entity
clusters. Experimental results on the document-level CoNLL-2012 Shared Task
English dataset show that eliminated spans are indeed much effective and our
approach can achieve competitive results when compared with previous
state-of-the-art in coreference resolution.

    

### [[2101.10861] A Review on Deep Learning in UAV Remote Sensing](http://arxiv.org/abs/2101.10861)


  Deep Neural Networks (DNNs) learn representation from data with an impressive
capability, and brought important breakthroughs for processing images,
time-series, natural language, audio, video, and many others. In the remote
sensing field, surveys and literature revisions specifically involving DNNs
algorithms' applications have been conducted in an attempt to summarize the
amount of information produced in its subfields. Recently, Unmanned Aerial
Vehicles (UAV) based applications have dominated aerial sensing research.
However, a literature revision that combines both "deep learning" and "UAV
remote sensing" thematics has not yet been conducted. The motivation for our
work was to present a comprehensive review of the fundamentals of Deep Learning
(DL) applied in UAV-based imagery. We focused mainly on describing
classification and regression techniques used in recent applications with
UAV-acquired data. For that, a total of 232 papers published in international
scientific journal databases was examined. We gathered the published material
and evaluated their characteristics regarding application, sensor, and
technique used. We relate how DL presents promising results and has the
potential for processing tasks associated with UAV-based image data. Lastly, we
project future perspectives, commentating on prominent DL paths to be explored
in the UAV remote sensing field. Our revision consists of a friendly-approach
to introduce, commentate, and summarize the state-of-the-art in UAV-based image
applications with DNNs algorithms in diverse subfields of remote sensing,
grouping it in the environmental, urban, and agricultural contexts.

    

### [[2108.02290] Relational E-Matching](http://arxiv.org/abs/2108.02290)


  We present a new approach to e-matching based on relational join; in
particular, we apply recent database query execution techniques to guarantee
worst-case optimal run time. Compared to the conventional backtracking approach
that always searches the e-graph "top down", our new relational e-matching
approach can better exploit pattern structure by searching the e-graph
according to an optimized query plan. We also establish the first data
complexity result for e-matching, bounding run time as a function of the
e-graph size and output size. We prototyped and evaluated our technique in the
state-of-the-art egg e-graph framework. Compared to a conventional baseline,
relational e-matching is simpler to implement and orders of magnitude faster in
practice.

    

### [[2108.02369] Proceedings of the 6th Workshop on Formal Integrated Development Environment](http://arxiv.org/abs/2108.02369)


  This volume contains the proceedings of F-IDE 2021, the sixth international
workshop on Formal Integrated Development Environment, which was held online on
May 24-25, 2021, as part of NFM'21, the 13th NASA Formal Methods Symposium.
High levels of safety, security and privacy standards require the use of formal
methods to specify and develop compliant software (sub)systems. Any standard
comes with an assessment process, which requires a complete documentation of
the application in order to ease the justification of design choices and the
review of code and proofs. Thus tools are needed for handling specifications,
program constructs and verification artifacts. The aim of the F-IDE workshop is
to provide a forum for presenting and discussing research efforts as well as
experience returns on design, development and usage of formal IDE aiming at
making formal methods more accessible for both specialists and non-specialists.

    

### [[2108.02490] HIPPODROME: Data Race Repair using Static Analysis Summaries](http://arxiv.org/abs/2108.02490)


  Implementing bug-free concurrent programs is a challenging task in modern
software development. State-of-the-art static analyses find hundreds of
concurrency bugs in production code, scaling to large codebases. Yet, fixing
these bugs in constantly changing codebases represents a daunting effort for
programmers, particularly because a fix in the concurrent code can introduce
other bugs in a subtle way.
In this work, we show how to harness compositional static analysis for
concurrency bug detection, to enable a new Automated Program Repair (APR)
technique for data races in large concurrent Java codebases. The key innovation
of our work is an algorithm that translates procedure summaries inferred by the
analysis tool for the purpose of bug reporting, into small local patches that
fix concurrency bugs (without introducing new ones). This synergy makes it
possible to extend the virtues of compositional static concurrency analysis to
APR, making our approach effective (it can detect and fix many more bugs than
existing tools for data race repair), scalable (it takes seconds to analyse and
suggest fixes for sizeable codebases), and usable (generally, it does not
require annotations from the users and can perform continuous automated
repair). Our study conducted on popular open-source projects has confirmed that
our tool automatically produces concurrency fixes similar to those proposed by
the developers in the past.

    

### [[2108.02672] Protocol-based Smart Contract Generation](http://arxiv.org/abs/2108.02672)


  The popularity of smart contracts is on the rise, yet breaches in reliability
and security linger. Among the many facets of smart contract reliability, we
concentrate on faults rooted in out-of-order interactions with contract
endpoints. We propose SmartScribble, a protocol language to describe valid
patterns of interaction between users and endpoints. SmartScribble not only
ensures correct interactive behaviour but also simplifies smart contract
coding. From a protocol description, our compiler generates a smart contract
that can then be completed by the programmer with the relevant business logic.
The generated contracts rely on finite state machines to control endpoint
invocations. As a proof of concept, we target Plutus, the contract programming
language for the Cardano blockchain. Preliminary evaluation points to a 75%
decrease in the size of the code that developers must write, coupled with an
increase of reliability by enforcing the specified patterns of interaction.

    