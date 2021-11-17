
## 2021-11-17

### [<title>Objective function definition - XGBoost</title>](https://discuss.xgboost.ai/t/objective-function-definition/2540/1)

### [<title>Objective definition - XGBoost</title>](https://discuss.xgboost.ai/t/objective-definition/2539/1)

### [<title>GXBoost Error after reinstall. Please Help - XGBoost</title>](https://discuss.xgboost.ai/t/gxboost-error-after-reinstall-please-help/2538/1)

### [[2111.08051] Common Language for Goal-Oriented Semantic Communications: A Curriculum Learning Framework](http://arxiv.org/abs/2111.08051)


  Semantic communications will play a critical role in enabling goal-oriented
services over next-generation wireless systems. However, most prior art in this
domain is restricted to specific applications (e.g., text or image), and it
does not enable goal-oriented communications in which the effectiveness of the
transmitted information must be considered along with the semantics so as to
execute a certain task. In this paper, a comprehensive semantic communications
framework is proposed for enabling goal-oriented task execution. To capture the
semantics between a speaker and a listener, a common language is defined using
the concept of beliefs to enable the speaker to describe the environment
observations to the listener. Then, an optimization problem is posed to choose
the minimum set of beliefs that perfectly describes the observation while
minimizing the task execution time and transmission cost. A novel top-down
framework that combines curriculum learning (CL) and reinforcement learning
(RL) is proposed to solve this problem. Simulation results show that the
proposed CL method outperforms traditional RL in terms of convergence time,
task execution time, and transmission cost during training.

    

### [[2111.08073] Learning Robust Scheduling with Search and Attention](http://arxiv.org/abs/2111.08073)


  Allocating physical layer resources to users based on channel quality, buffer
size, requirements and constraints represents one of the central optimization
problems in the management of radio resources. The solution space grows
combinatorially with the cardinality of each dimension making it hard to find
optimal solutions using an exhaustive search or even classical optimization
algorithms given the stringent time requirements. This problem is even more
pronounced in MU-MIMO scheduling where the scheduler can assign multiple users
to the same time-frequency physical resources. Traditional approaches thus
resort to designing heuristics that trade optimality in favor of feasibility of
execution. In this work we treat the MU-MIMO scheduling problem as a
tree-structured combinatorial problem and, borrowing from the recent successes
of AlphaGo Zero, we investigate the feasibility of searching for the best
performing solutions using a combination of Monte Carlo Tree Search and
Reinforcement Learning. To cater to the nature of the problem at hand, like the
lack of an intrinsic ordering of the users as well as the importance of
dependencies between combinations of users, we make fundamental modifications
to the neural network architecture by introducing the self-attention mechanism.
We then demonstrate that the resulting approach is not only feasible but vastly
outperforms state-of-the-art heuristic-based scheduling approaches in the
presence of measurement uncertainties and finite buffers.

    

### [[2111.08105] El impacto del buffer en la calidad de servicio](http://arxiv.org/abs/2111.08105)


  The response in data flows transmission in real time is analyzed, for access
network scenarios, in which said flows converge on an outgoing link, competing
to achieve a certain level of quality of service. The concurrence of these
types of flows can generate bursts of packets, which in certain circumstances
can compromise the capacity of the buffers to absorb packets in congestion
periods. In addition, an analysis of the characteristics of buffers in access
devices is presented, especially their size and packet loss. In particular, it
describes how these characteristics can affect the quality of multimedia
applications when bursty traffic is generated, it also presents possible
effects on the traffic of other applications that share a common link.

    

### [[2111.08193] HyperNAT: Scaling Up Network AddressTranslation with SmartNICs for Clouds](http://arxiv.org/abs/2111.08193)


  Network address translation (NAT) is a basic functionality in cloud gateways.
With the increasing traffic volume and number of flows introduced by the cloud
tenants, the NAT gateway needs to be implemented on a cluster of servers. We
propose to scale up the gateway servers, which could reduce the number of
servers so as to reduce the capital expense and operation expense. We design
HyperNAT, which leverages smartNICs to improve the server's processing
capacity. In HyperNAT, the NAT functionality is distributed on multiple NICs,
and the flow space is divided and assigned accordingly. HyperNAT overcomes the
challenge that the packets in two directions of one connection need to be
processed by the same NAT rule (named two-direction consistency, TDC) by
cloning the rule to both data paths of the two directions. Our implementation
and evaluation of HyperNAT show that HyperNAT could scale up cloud gateway
effectively with low overhead.

    

### [[2111.08326] Moving the Network to the Cloud: the Cloud Central Office Revolution and its Implications for the Optical Layer](http://arxiv.org/abs/2111.08326)


  SDN and NFV have recently changed the way we operate networks. By decoupling
control and data plane operations and virtualising their components, they have
opened up new frontiers towards reducing network ownership costs and improving
usability and efficiency. Recently, their applicability has moved towards
public telecommunications networks, with concepts such as the cloud-CO that
have pioneered its use in access and metro networks: an idea that has quickly
attracted the interest of network operators. By merging mobile, residential and
enterprise services into a common framework, built around commoditised data
centre types of architectures, future embodiments of this CO virtualisation
concept could achieve significant capital and operational cost savings, while
providing customised network experience to high-capacity and low-latency future
applications.
This tutorial provides an overview of the various frameworks and
architectures outlining current network disaggregation trends that are leading
to the virtualisation/cloudification of central offices. It also provides
insight on the virtualisation of the access-metro network, showcasing new
software functionalities like the virtual \ac{DBA} mechanisms for \acp{PON}. In
addition, we explore how it can bring together different network technologies
to enable convergence of mobile and optical access networks and pave the way
for the integration of disaggregated ROADM networks. Finally, this paper
discusses some of the open challenges towards the realisation of networks
capable of delivering guaranteed performance, while sharing resources across
multiple operators and services.

    

### [[2111.08328] On The Complexity of Maximizing Temporal Reachability via Trip Temporalisation](http://arxiv.org/abs/2111.08328)


  We consider the problem of assigning appearing times to the edges of a
digraph in order to maximize the (average) temporal reachability between pairs
of nodes. Motivated by the application to public transit networks, where edges
cannot be scheduled independently one of another, we consider the setting where
the edges are grouped into certain walks (called trips) in the digraph and
where assigning the appearing time to the first edge of a trip forces the
appearing times of the subsequent edges. In this setting, we show that, quite
surprisingly, it is NP-complete to decide whether there exists an assignment of
times connecting a given pair of nodes. This result allows us to prove that the
problem of maximising the temporal reachability cannot be approximated within a
factor better than some polynomial term in the size of the graph. We thus focus
on the case where, for each pair of nodes, there exists an assignment of times
such that one node is reachable from the other. We call this property strong
temporalisability. It is a very natural assumption for the application to
public transit networks. On the negative side, the problem of maximising the
temporal reachability remains hard to approximate within a factor $\sqrt$ n/12
in that setting. Moreover, we show the existence of collections of trips that
are strongly temporalisable but for which any assignment of starting times to
the trips connects at most an O(1/ $\sqrt$ n) fraction of all pairs of nodes.
On the positive side, we show that there must exist an assignment of times that
connects a constant fraction of all pairs in the strongly temporalisable and
symmetric case, that is, when the set of trips to be scheduled is such that,
for each trip, there is a symmetric trip visiting the same nodes in reverse
order. Keywords:edge labeling edge scheduled network network optimisation
temporal graph temporal path temporal reachability time assignment

    

### [[2111.08397] CLARA: A Constrained Reinforcement Learning Based Resource Allocation Framework for Network Slicing](http://arxiv.org/abs/2111.08397)


  As mobile networks proliferate, we are experiencing a strong diversification
of services, which requires greater flexibility from the existing network.
Network slicing is proposed as a promising solution for resource utilization in
5G and future networks to address this dire need. In network slicing, dynamic
resource orchestration and network slice management are crucial for maximizing
resource utilization. Unfortunately, this process is too complex for
traditional approaches to be effective due to a lack of accurate models and
dynamic hidden structures. We formulate the problem as a Constrained Markov
Decision Process (CMDP) without knowing models and hidden structures.
Additionally, we propose to solve the problem using CLARA, a Constrained
reinforcement LeArning based Resource Allocation algorithm. In particular, we
analyze cumulative and instantaneous constraints using adaptive interior-point
policy optimization and projection layer, respectively. Evaluations show that
CLARA clearly outperforms baselines in resource allocation with service demand
guarantees.

    

### [[2111.08520] Hyperbolicity Computation through Dominating Sets *](http://arxiv.org/abs/2111.08520)


  Hyperbolicity is a graph parameter related to how much a graph resembles a
tree with respect to distances. Its computation is challenging as the main
approaches consist in scanning all quadruples of the graph or using fast matrix
multiplication as building block, both are not practical for large graphs. In
this paper, we propose and evaluate an approach that uses a hierarchy of
distance-k dominating sets to reduce the search space. This technique, compared
to the previous best practical algorithms, enables us to compute the
hyperbolicity of graphs with unprecedented size (up to a million nodes) and
speeds up the computation of previously attainable graphs by up to 3 orders of
magnitude while reducing the memory consumption by up to more than a factor of
23.

    

### [[2111.08572] Saath: Speeding up CoFlows by Exploiting the Spatial Dimension](http://arxiv.org/abs/2111.08572)


  Coflow scheduling improves data-intensive application performance by
improving their networking performance. State-of-the-art Coflow schedulers in
essence approximate the classic online Shortest-Job-First (SJF) scheduling,
designed for a single CPU, in a distributed setting, with no coordination among
how the flows of a Coflow at individual ports are scheduled, and as a result
suffer two performance drawbacks: (1) The flows of a Coflow may suffer the
out-of-sync problem -- they may be scheduled at different times and become
drifting apart, negatively affecting the Coflow completion time (CCT); (2) FIFO
scheduling of flows at each port bears no notion of SJF, leading to suboptimal
CCT. We propose SAATH, an online Coflow scheduler that overcomes the above
drawbacks by explicitly exploiting the spatial dimension of Coflows. In SAATH,
the global scheduler schedules the flows of a Coflow using an all-or-none
policy which mitigates the out-of-sync problem. To order the Coflows within
each queue, SAATH resorts to a Least-Contention-First (LCoF) policy which we
show extends the gist of SJF to the spatial dimension, complemented with
starvation freedom. Our evaluation using an Azure testbed and simulations of
two production cluster traces show that compared to Aalo, SAATH reduces the CCT
in median (P90) cases by 1.53x (4.5x) and 1.42x (37x), respectively.

    

### [[2111.08629] Communication by means of Modulated Johnson Noise](http://arxiv.org/abs/2111.08629)


  We present the design of a new passive communication method that does not
rely on ambient or generated RF sources. Instead, we exploit the Johnson
(thermal) noise generated by a resistor to transmit information bits
wirelessly. By switching the load connected to an antenna between a resistor
and open circuit, we can achieve data rates of up to 26bps and distances of up
to 7.3 meters. This communication method is orders of magnitude less power
consuming than conventional communication schemes and presents the opportunity
to enable wireless communication in areas with a complete lack of connectivity.

    

### [[2111.08663] Engineering Edge-Cloud Offloading of Big Data for Channel Modelling in THz-range Communications](http://arxiv.org/abs/2111.08663)


  Channel estimation in mmWave and THz-range wireless communications (producing
Gb/Tb-range of data) is critical to configuring system parameters related to
transmission signal quality, and yet it remains a daunting challenge both in
software and hardware. Current methods of channel estimations, be it modeling-
or data-based (machine learning (ML)), - use and create big data. This in turn
requires a large amount of computational resources, read operations to prove if
there is some predefined channel configurations, e.g., QoS requirements, in the
database, as well as write operations to store the new combinations of QoS
parameters in the database. Especially the ML-based approach requires high
computational and storage resources, low latency and a higher hardware
flexibility. In this paper, we engineer and study the offloading of the above
operations to edge and cloud computing systems to understand the suitability of
edge and cloud computing to provide rapid response with channel and link
configuration parameters on the example of THz channel modeling. We evaluate
the performance of the engineered system when the computational and storage
resources are orchestrated based on: 1) monolithic architecture, 2)
microservices architectures, both in edge-cloud based approach. For
microservices approach, we engineer both Docker Swarm and Kubernetes systems.
The measurements show a great promise of edge computing and microservices that
can quickly respond to properly configure parameters and improve transmission
distance and signal quality with ultra-high speed wireless communications.

    

### [[2101.02615] Benchmarking Buffer Size in IoT Devices Deploying REST HTTP](http://arxiv.org/abs/2101.02615)


  A few potential IoT communication protocols at the application layer have
been proposed, including MQTT, CoAP and REST HTTP, with the latter being the
protocol of choice for software developers due to its compatibility with the
existing systems. We present a theoretical model of the expected buffer size on
the REST HTTP client buffer in IoT devices under lossy wireless conditions, and
validate the study experimentally. The results show that increasing the buffer
size in IoT devices does not always improve performance in lossy environments,
hence demonstrating the importance of benchmarking the buffer size in IoT
systems deploying REST HTTP.

    

### [[2101.08681] Streaming from the Air: Enabling Drone-sourced Video Streaming Applications on 5G Open-RAN Architectures](http://arxiv.org/abs/2101.08681)


  Enabling high data-rate uplink cellular connectivity for drones is a
challenging problem, since a flying drone has a higher likelihood of having
line-of-sight propagation to base stations that terrestrial UEs normally do not
have line-of-sight to. This may result in uplink inter-cell interference and
uplink performance degradation for the neighboring ground UEs when drones
transmit at high data-rates (e.g., video streaming). We address this problem
from a cellular operator's standpoint to support drone-sourced video streaming
of a point of interest. We propose a low-complexity, closed-loop control system
for Open-RAN architectures that jointly optimizes the drone's location in space
and its transmission directionality to support video streaming and minimize its
uplink interference impact on the network. We prototype and experimentally
evaluate the proposed control system on a dedicated outdoor multi-cell RAN
testbed, which is the first measurement campaign of its kind. Furthermore, we
perform a large-scale simulation assessment of the proposed control system
using the actual cell deployment topologies and cell load profiles of a major
US cellular carrier. The proposed Open-RAN control scheme achieves an average
19% network capacity gain over traditional BS-constrained control solutions and
satisfies the application data-rate requirements of the drone (e.g., to stream
an HD video).

    

### [[2104.06188] Sectors, Beams and Environmental Impact on the Performance of Commercial 5G mmWave Cells: an Empirical Study](http://arxiv.org/abs/2104.06188)


  MmWave communication is one of the cornerstones of future generations of
mobile networks. It is recognized by its multi-gigabit transmission capacity,
high propagation loss, and sensitivity to blockage. The high rate provided by
mmWave links can potentially unlock emerging applications such as online
ultra-high-definition video streaming, augmented and virtual reality, and
instant file sharing for 5G networks. While the performance of mmWave links has
been thoroughly investigated either by simulations or testbeds, the behavior of
this technology in real-world commercial setups has not yet been thoroughly
documented. In this paper, we address this gap and present the results of an
empirical study to determine the actual performance of a commercial 5G mmWave
cell through on-field measurements. We evaluate the signal and beam coverage
map of an operational network as well as the end-to-end communication
performance of a 5G mmWave connection, considering various scenarios, including
human body blockage effects, foliage-caused and rain-induced attenuation, and
water surface effects. To the best of our knowledge, this paper is the first to
report on a commercial deployment while not treating the radio as a black box.
This measurement analysis provides valuable information for researchers and 5G
verticals to fully understand how a 5G mmWave access network operates in
real-world and operational conditions, with buildings, humans, trees, water
surfaces, etc.

    

### [[2104.08122] Benchmarking Machine Learning Techniques for THz Channel Estimation Problems](http://arxiv.org/abs/2104.08122)


  Terahertz communication is one of the most promising wireless communication
technologies for 6G generation and beyond. For THz systems to be practically
adopted, channel estimation is one of the key issues. We consider the problem
of channel modeling and estimation with deterministic channel propagation and
the related physical characteristics of THz bands, and benchmark various
machine learning algorithms to estimate THz channel, including neural networks
(NN), logistic regression (LR), and projected gradient ascent (PGA). Numerical
results show that PGA algorithm yields the most promising performance at SNR=0
dB with NMSE of -12.8 dB.

    

### [[2111.08001] Metagenome2Vec: Building Contextualized Representations for Scalable Metagenome Analysis](http://arxiv.org/abs/2111.08001)


  Advances in next-generation metagenome sequencing have the potential to
revolutionize the point-of-care diagnosis of novel pathogen infections, which
could help prevent potential widespread transmission of diseases. Given the
high volume of metagenome sequences, there is a need for scalable frameworks to
analyze and segment metagenome sequences from clinical samples, which can be
highly imbalanced. There is an increased need for learning robust
representations from metagenome reads since pathogens within a family can have
highly similar genome structures (some more than 90%) and hence enable the
segmentation and identification of novel pathogen sequences with limited
labeled data. In this work, we propose Metagenome2Vec - a contextualized
representation that captures the global structural properties inherent in
metagenome data and local contextualized properties through self-supervised
representation learning. We show that the learned representations can help
detect six (6) related pathogens from clinical samples with less than 100
labeled sequences. Extensive experiments on simulated and clinical metagenome
data show that the proposed representation encodes compositional properties
that can generalize beyond annotations to segment novel pathogens in an
unsupervised setting.

    

### [[2111.08002] Natural Gradient Variational Inference with Gaussian Mixture Models](http://arxiv.org/abs/2111.08002)


  Bayesian methods estimate a measure of uncertainty by using the posterior
distribution. One source of difficulty in these methods is the computation of
the normalizing constant. Calculating exact posterior is generally intractable
and we usually approximate it. Variational Inference (VI) methods approximate
the posterior with a distribution usually chosen from a simple family using
optimization. The main contribution of this work is described is a set of
update rules for natural gradient variational inference with mixture of
Gaussians, which can be run independently for each of the mixture components,
potentially in parallel.

    

### [[2111.08005] Solving Inverse Problems in Medical Imaging with Score-Based Generative Models](http://arxiv.org/abs/2111.08005)


  Reconstructing medical images from partial measurements is an important
inverse problem in Computed Tomography (CT) and Magnetic Resonance Imaging
(MRI). Existing solutions based on machine learning typically train a model to
directly map measurements to medical images, leveraging a training dataset of
paired images and measurements. These measurements are typically synthesized
from images using a fixed physical model of the measurement process, which
hinders the generalization capability of models to unknown measurement
processes. To address this issue, we propose a fully unsupervised technique for
inverse problem solving, leveraging the recently introduced score-based
generative models. Specifically, we first train a score-based generative model
on medical images to capture their prior distribution. Given measurements and a
physical model of the measurement process at test time, we introduce a sampling
method to reconstruct an image consistent with both the prior and the observed
measurements. Our method does not assume a fixed measurement process during
training, and can thus be flexibly adapted to different measurement processes
at test time. Empirically, we observe comparable or better performance to
supervised learning techniques in several medical imaging tasks in CT and MRI,
while demonstrating significantly better generalization to unknown measurement
processes.

    

### [[2111.08006] Disparities in Dermatology AI: Assessments Using Diverse Clinical Images](http://arxiv.org/abs/2111.08006)


  More than 3 billion people lack access to care for skin disease. AI
diagnostic tools may aid in early skin cancer detection; however most models
have not been assessed on images of diverse skin tones or uncommon diseases. To
address this, we curated the Diverse Dermatology Images (DDI) dataset - the
first publicly available, pathologically confirmed images featuring diverse
skin tones. We show that state-of-the-art dermatology AI models perform
substantially worse on DDI, with ROC-AUC dropping 29-40 percent compared to the
models' original results. We find that dark skin tones and uncommon diseases,
which are well represented in the DDI dataset, lead to performance drop-offs.
Additionally, we show that state-of-the-art robust training methods cannot
correct for these biases without diverse training data. Our findings identify
important weaknesses and biases in dermatology AI that need to be addressed to
ensure reliable application to diverse patients and across all disease.

    

### [[2111.08008] SPLDExtraTrees: Robust machine learning approach for predicting kinase inhibitor resistance](http://arxiv.org/abs/2111.08008)


  Drug resistance is a major threat to the global health and a significant
concern throughout the clinical treatment of diseases and drug development. The
mutation in proteins that is related to drug binding is a common cause for
adaptive drug resistance. Therefore, quantitative estimations of how mutations
would affect the interaction between a drug and the target protein would be of
vital significance for the drug development and the clinical practice.
Computational methods that rely on molecular dynamics simulations, Rosetta
protocols, as well as machine learning methods have been proven to be capable
of predicting ligand affinity changes upon protein mutation. However, the
severely limited sample size and heavy noise induced overfitting and
generalization issues have impeded wide adoption of machine learning for
studying drug resistance. In this paper, we propose a robust machine learning
method, termed SPLDExtraTrees, which can accurately predict ligand binding
affinity changes upon protein mutation and identify resistance-causing
mutations. Especially, the proposed method ranks training data following a
specific scheme that starts with easy-to-learn samples and gradually
incorporates harder and diverse samples into the training, and then iterates
between sample weight recalculations and model updates. In addition, we
calculate additional physics-based structural features to provide the machine
learning model with the valuable domain knowledge on proteins for this
data-limited predictive tasks. The experiments substantiate the capability of
the proposed method for predicting kinase inhibitor resistance under three
scenarios, and achieves predictive accuracy comparable to that of molecular
dynamics and Rosetta methods with much less computational costs.

    

### [[2111.08010] Modular Networks Prevent Catastrophic Interference in Model-Based Multi-Task Reinforcement Learning](http://arxiv.org/abs/2111.08010)


  In a multi-task reinforcement learning setting, the learner commonly benefits
from training on multiple related tasks by exploiting similarities among them.
At the same time, the trained agent is able to solve a wider range of different
problems. While this effect is well documented for model-free multi-task
methods, we demonstrate a detrimental effect when using a single learned
dynamics model for multiple tasks. Thus, we address the fundamental question of
whether model-based multi-task reinforcement learning benefits from shared
dynamics models in a similar way model-free methods do from shared policy
networks. Using a single dynamics model, we see clear evidence of task
confusion and reduced performance. As a remedy, enforcing an internal structure
for the learned dynamics model by training isolated sub-networks for each task
notably improves performance while using the same amount of parameters. We
illustrate our findings by comparing both methods on a simple gridworld and a
more complex vizdoom multi-task experiment.

    

### [[2111.08011] Advantage of Machine Learning over Maximum Likelihood in Limited-Angle Low-Photon X-Ray Tomography](http://arxiv.org/abs/2111.08011)


  Limited-angle X-ray tomography reconstruction is an ill-conditioned inverse
problem in general. Especially when the projection angles are limited and the
measurements are taken in a photon-limited condition, reconstructions from
classical algorithms such as filtered backprojection may lose fidelity and
acquire artifacts due to the missing-cone problem. To obtain satisfactory
reconstruction results, prior assumptions, such as total variation minimization
and nonlocal image similarity, are usually incorporated within the
reconstruction algorithm. In this work, we introduce deep neural networks to
determine and apply a prior distribution in the reconstruction process. Our
neural networks learn the prior directly from synthetic training samples. The
neural nets thus obtain a prior distribution that is specific to the class of
objects we are interested in reconstructing. In particular, we used deep
generative models with 3D convolutional layers and 3D attention layers which
are trained on 3D synthetic integrated circuit (IC) data from a model dubbed
CircuitFaker. We demonstrate that, when the projection angles and photon
budgets are limited, the priors from our deep generative models can
dramatically improve the IC reconstruction quality on synthetic data compared
with maximum likelihood estimation. Training the deep generative models with
synthetic IC data from CircuitFaker illustrates the capabilities of the learned
prior from machine learning. We expect that if the process were reproduced with
experimental data, the advantage of the machine learning would persist. The
advantages of machine learning in limited angle X-ray tomography may further
enable applications in low-photon nanoscale imaging.

    

### [[2111.08014] Tensor network to learn the wavefunction of data](http://arxiv.org/abs/2111.08014)


  How many different ways are there to handwrite digit 3? To quantify this
question imagine extending a dataset of handwritten digits MNIST by sampling
additional images until they start repeating. We call the collection of all
resulting images of digit 3 the "full set." To study the properties of the full
set we introduce a tensor network architecture which simultaneously
accomplishes both classification (discrimination) and sampling tasks.
Qualitatively, our trained network represents the indicator function of the
full set. It therefore can be used to characterize the data itself. We
illustrate that by studying the full sets associated with the digits of MNIST.
Using quantum mechanical interpretation of our network we characterize the
full set by calculating its entanglement entropy. We also study its geometric
properties such as mean Hamming distance, effective dimension, and size. The
latter answers the question above -- the total number of black and white threes
written MNIST style is $2^{72}$.

    

### [[2111.08030] Fast and Credible Likelihood-Free Cosmology with Truncated Marginal Neural Ratio Estimation](http://arxiv.org/abs/2111.08030)


  Sampling-based inference techniques are central to modern cosmological data
analysis; these methods, however, scale poorly with dimensionality and
typically require approximate or intractable likelihoods. In this paper we
describe how Truncated Marginal Neural Ratio Estimation (TMNRE) (a new approach
in so-called simulation-based inference) naturally evades these issues,
improving the $(i)$ efficiency, $(ii)$ scalability, and $(iii)$ trustworthiness
of the inferred posteriors. Using measurements of the Cosmic Microwave
Background (CMB), we show that TMNRE can achieve converged posteriors using
orders of magnitude fewer simulator calls than conventional Markov Chain Monte
Carlo (MCMC) methods. Remarkably, the required number of samples is effectively
independent of the number of nuisance parameters. In addition, a property
called \emph{local amortization} allows the performance of rigorous statistical
consistency checks that are not accessible to sampling-based methods. TMNRE
promises to become a powerful tool for cosmological data analysis, particularly
in the context of extended cosmologies, where the timescale required for
conventional sampling-based inference methods to converge can greatly exceed
that of simple cosmological models such as $\Lambda$CDM. To perform these
computations, we use an implementation of TMNRE via the open-source code
\texttt{swyft}.

    

### [[2111.08057] Margin-Independent Online Multiclass Learning via Convex Geometry](http://arxiv.org/abs/2111.08057)


  We consider the problem of multi-class classification, where a stream of
adversarially chosen queries arrive and must be assigned a label online. Unlike
traditional bounds which seek to minimize the misclassification rate, we
minimize the total distance from each query to the region corresponding to its
correct label. When the true labels are determined via a nearest neighbor
partition -- i.e. the label of a point is given by which of $k$ centers it is
closest to in Euclidean distance -- we show that one can achieve a loss that is
independent of the total number of queries. We complement this result by
showing that learning general convex sets requires an almost linear loss per
query. Our results build off of regret guarantees for the geometric problem of
contextual search. In addition, we develop a novel reduction technique from
multiclass classification to binary classification which may be of independent
interest.

    

### [[2111.08066] Exploiting Action Impact Regularity and Partially Known Models for Offline Reinforcement Learning](http://arxiv.org/abs/2111.08066)


  Offline reinforcement learning-learning a policy from a batch of data-is
known to be hard: without making strong assumptions, it is easy to construct
counterexamples such that existing algorithms fail. In this work, we instead
consider a property of certain real world problems where offline reinforcement
learning should be effective: those where actions only have limited impact for
a part of the state. We formalize and introduce this Action Impact Regularity
(AIR) property. We further propose an algorithm that assumes and exploits the
AIR property, and bound the suboptimality of the output policy when the MDP
satisfies AIR. Finally, we demonstrate that our algorithm outperforms existing
offline reinforcement learning algorithms across different data collection
policies in two simulated environments where the regularity holds.

    

### [[2111.08067] ModelLight: Model-Based Meta-Reinforcement Learning for Traffic Signal Control](http://arxiv.org/abs/2111.08067)


  Traffic signal control is of critical importance for the effective use of
transportation infrastructures. The rapid increase of vehicle traffic and
changes in traffic patterns make traffic signal control more and more
challenging. Reinforcement Learning (RL)-based algorithms have demonstrated
their potential in dealing with traffic signal control. However, most existing
solutions require a large amount of training data, which is unacceptable for
many real-world scenarios. This paper proposes a novel model-based
meta-reinforcement learning framework (ModelLight) for traffic signal control.
Within ModelLight, an ensemble of models for road intersections and the
optimization-based meta-learning method are used to improve the data efficiency
of an RL-based traffic light control method. Experiments on real-world datasets
demonstrate that ModelLight can outperform state-of-the-art traffic light
control algorithms while substantially reducing the number of required
interactions with the real-world environment.

    

### [[2111.08082] Learning Graph Neural Networks for Multivariate Time Series Anomaly Detection](http://arxiv.org/abs/2111.08082)


  In this work, we propose GLUE (Graph Deviation Network with Local Uncertainty
Estimation), building on the recently proposed Graph Deviation Network (GDN).
GLUE not only automatically learns complex dependencies between variables and
uses them to better identify anomalous behavior, but also quantifies its
predictive uncertainty, allowing us to account for the variation in the data as
well to have more interpretable anomaly detection thresholds. Results on two
real world datasets tell us that optimizing the negative Gaussian log
likelihood is reasonable because GLUE's forecasting results are at par with GDN
and in fact better than the vector autoregressor baseline, which is significant
given that GDN directly optimizes the MSE loss. In summary, our experiments
demonstrate that GLUE is competitive with GDN at anomaly detection, with the
added benefit of uncertainty estimations. We also show that GLUE learns
meaningful sensor embeddings which clusters similar sensors together.

    

### [[2111.08094] LIMEcraft: Handcrafted superpixel selection and inspection for Visual eXplanations](http://arxiv.org/abs/2111.08094)


  The increased interest in deep learning applications, and their
hard-to-detect biases result in the need to validate and explain complex
models. However, current explanation methods are limited as far as both the
explanation of the reasoning process and prediction results are concerned. They
usually only show the location in the image that was important for model
prediction. The lack of possibility to interact with explanations makes it
difficult to verify and understand exactly how the model works. This creates a
significant risk when using the model. It is compounded by the fact that
explanations do not take into account the semantic meaning of the explained
objects. To escape from the trap of static explanations, we propose an approach
called LIMEcraft that allows a user to interactively select semantically
consistent areas and thoroughly examine the prediction for the image instance
in case of many image features. Experiments on several models showed that our
method improves model safety by inspecting model fairness for image pieces that
may indicate model bias. The code is available at:
this http URL


### [[2111.08095] TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation](http://arxiv.org/abs/2111.08095)


  Recent work in synthetic data generation in the time-series domain has
focused on the use of Generative Adversarial Networks. We propose a novel
architecture for synthetically generating time-series data with the use of
Variational Auto-Encoders (VAEs). The proposed architecture has several
distinct properties: interpretability, ability to encode domain knowledge, and
reduced training times. We evaluate data generation quality by similarity and
predictability against four multivariate datasets. We experiment with varying
sizes of training data to measure the impact of data availability on generation
quality for our VAE method as well as several state-of-the-art data generation
methods. Our results on similarity tests show that the VAE approach is able to
accurately represent the temporal attributes of the original data. On next-step
prediction tasks using generated data, the proposed VAE architecture
consistently meets or exceeds performance of state-of-the-art data generation
methods. While noise reduction may cause the generated data to deviate from
original data, we demonstrate the resulting de-noised data can significantly
improve performance for next-step prediction using generated data. Finally, the
proposed architecture can incorporate domain-specific time-patterns such as
polynomial trends and seasonalities to provide interpretable outputs. Such
interpretability can be highly advantageous in applications requiring
transparency of model outputs or where users desire to inject prior knowledge
of time-series patterns into the generative model.

    

### [[2111.08096] VisualEnv: visual Gym environments with Blender](http://arxiv.org/abs/2111.08096)


  In this paper VisualEnv, a new tool for creating visual environment for
reinforcement learning is introduced. It is the product of an integration of an
open-source modelling and rendering software, Blender, and a python module used
to generate environment model for simulation, OpenAI Gym. VisualEnv allows the
user to create custom environments with photorealistic rendering capabilities
and full integration with python. The framework is described and tested on a
series of example problems that showcase its features for training
reinforcement learning agents.

    

### [[2111.08117] Neural networks with linear threshold activations: structure and algorithms](http://arxiv.org/abs/2111.08117)


  In this article we present new results on neural networks with linear
threshold activation functions. We precisely characterize the class of
functions that are representable by such neural networks and show that 2 hidden
layers are necessary and sufficient to represent any function representable in
the class. This is a surprising result in the light of recent exact
representability investigations for neural networks using other popular
activation functions like rectified linear units (ReLU). We also give precise
bounds on the sizes of the neural networks required to represent any function
in the class. Finally, we design an algorithm to solve the empirical risk
minimization (ERM) problem to global optimality for these neural networks with
a fixed architecture. The algorithm's running time is polynomial in the size of
the data sample, if the input dimension and the size of the network
architecture are considered fixed constants. The algorithm is unique in the
sense that it works for any architecture with any number of layers, whereas
previous polynomial time globally optimal algorithms work only for very
restricted classes of architectures.

    

### [[2111.08133] Exploring Story Generation with Multi-task Objectives in Variational Autoencoders](http://arxiv.org/abs/2111.08133)


  GPT-2 has been frequently adapted in story generation models as it provides
powerful generative capability. However, it still fails to generate consistent
stories and lacks diversity. Current story generation models leverage
additional information such as plots or commonsense into GPT-2 to guide the
generation process. These approaches focus on improving generation quality of
stories while our work look at both quality and diversity. We explore combining
BERT and GPT-2 to build a variational autoencoder (VAE), and extend it by
adding additional objectives to learn global features such as story topic and
discourse relations. Our evaluations show our enhanced VAE can provide better
quality and diversity trade off, generate less repetitive story content and
learn a more informative latent variable.

    

### [[2111.08137] Joint Unsupervised and Supervised Training for Multilingual ASR](http://arxiv.org/abs/2111.08137)


  Self-supervised training has shown promising gains in pretraining models and
facilitating the downstream finetuning for speech recognition, like
multilingual ASR. Most existing methods adopt a 2-stage scheme where the
self-supervised loss is optimized in the first pretraining stage, and the
standard supervised finetuning resumes in the second stage. In this paper, we
propose an end-to-end (E2E) Joint Unsupervised and Supervised Training (JUST)
method to combine the supervised RNN-T loss and the self-supervised contrastive
and masked language modeling (MLM) losses. We validate its performance on the
public dataset Multilingual LibriSpeech (MLS), which includes 8 languages and
is extremely imbalanced. On MLS, we explore (1) JUST trained from scratch, and
(2) JUST finetuned from a pretrained checkpoint. Experiments show that JUST can
consistently outperform other existing state-of-the-art methods, and beat the
monolingual baseline by a significant margin, demonstrating JUST's capability
of handling low-resource languages in multilingual ASR. Our average WER of all
languages outperforms average monolingual baseline by 33.3%, and the
state-of-the-art 2-stage XLSR by 32%. On low-resource languages like Polish,
our WER is less than half of the monolingual baseline and even beats the
supervised transfer learning method which uses external supervision.

    

### [[2111.08140] Bayesian inference of the climbing grade scale](http://arxiv.org/abs/2111.08140)


  Climbing grades are used to classify a climbing route based on its perceived
difficulty, and have come to play a central role in the sport of rock climbing.
Recently, the first statistically rigorous method for estimating climbing
grades from whole-history ascent data was described, based on the dynamic
Bradley-Terry model for games between players of time-varying ability. In this
paper, we implement inference under the whole-history rating model using Markov
chain Monte Carlo and apply the method to a curated data set made up of
climbers who climb regularly. We use these data to get an estimate of the
model's fundamental scale parameter m, which defines the proportional increase
in difficulty associated with an increment of grade. We show that the data
conform to assumptions that the climbing grade scale is a logarithmic scale of
difficulty, like decibels or stellar magnitude. We estimate that an increment
in Ewbank, French and UIAA climbing grade systems corresponds to 2.1, 2.09 and
2.13 times increase in difficulty respectively, assuming a logistic model of
probability of success as a function of grade. Whereas we find that the Vermin
scale for bouldering (V-grade scale) corresponds to a 3.17 increase in
difficulty per grade increment. In addition, we highlight potential connections
between the logarithmic properties of climbing grade scales and the
psychophysical laws of Weber and Fechner.

    

### [[2111.08154] On the utility of power spectral techniques with feature selection techniques for effective mental task classification in noninvasive BCI](http://arxiv.org/abs/2111.08154)


  In this paper classification of mental task-root Brain-Computer Interfaces
(BCI) is being investigated, as those are a dominant area of investigations in
BCI and are of utmost interest as these systems can be augmented life of people
having severe disabilities. The BCI model's performance is primarily dependent
on the size of the feature vector, which is obtained through multiple channels.
In the case of mental task classification, the availability of training samples
to features are minimal. Very often, feature selection is used to increase the
ratio for the mental task classification by getting rid of irrelevant and
superfluous features. This paper proposes an approach to select relevant and
non-redundant spectral features for the mental task classification. This can be
done by using four very known multivariate feature selection methods viz,
Bhattacharya's Distance, Ratio of Scatter Matrices, Linear Regression and
Minimum Redundancy & Maximum Relevance. This work also deals with a comparative
analysis of multivariate and univariate feature selection for mental task
classification. After applying the above-stated method, the findings
demonstrate substantial improvements in the performance of the learning model
for mental task classification. Moreover, the efficacy of the proposed approach
is endorsed by carrying out a robust ranking algorithm and Friedman's
statistical test for finding the best combinations and comparing different
combinations of power spectral density and feature selection methods.

    

### [[2111.08161] Sparse Graph Learning Under Laplacian-Related Constraints](http://arxiv.org/abs/2111.08161)


  We consider the problem of learning a sparse undirected graph underlying a
given set of multivariate data. We focus on graph Laplacian-related constraints
on the sparse precision matrix that encodes conditional dependence between the
random variables associated with the graph nodes. Under these constraints the
off-diagonal elements of the precision matrix are non-positive (total
positivity), and the precision matrix may not be full-rank. We investigate
modifications to widely used penalized log-likelihood approaches to enforce
total positivity but not the Laplacian structure. The graph Laplacian can then
be extracted from the off-diagonal precision matrix. An alternating direction
method of multipliers (ADMM) algorithm is presented and analyzed for
constrained optimization under Laplacian-related constraints and lasso as well
as adaptive lasso penalties. Numerical results based on synthetic data show
that the proposed constrained adaptive lasso approach significantly outperforms
existing Laplacian-based approaches. We also evaluate our approach on real
financial data.

    

### [[2111.08162] On Bock's Conjecture Regarding the Adam Optimizer](http://arxiv.org/abs/2111.08162)


  In 2014, Kingma and Ba published their Adam optimizer algorithm, together
with a mathematical argument that was meant to help justify it. In 2018, Bock
and colleagues reported that a key piece was missing from that argument $-$ an
unproven lemma which we will call Bock's conjecture. Here we show that this
conjecture is false, but a modified version of it does hold, and fills the gap
in Bock's proof of convergence for Adam.

    

### [[2111.08163] An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks](http://arxiv.org/abs/2111.08163)


  Modern convolutional neural networks (CNNs) are known to be overconfident in
terms of their calibration on unseen input data. That is to say, they are more
confident than they are accurate. This is undesirable if the probabilities
predicted are to be used for downstream decision making. When considering
accuracy, CNNs are also surprisingly robust to compression techniques, such as
quantization, which aim to reduce computational and memory costs. We show that
this robustness can be partially explained by the calibration behavior of
modern CNNs, and may be improved with overconfidence. This is due to an
intuitive result: low confidence predictions are more likely to change
post-quantization, whilst being less accurate. High confidence predictions will
be more accurate, but more difficult to change. Thus, a minimal drop in
post-quantization accuracy is incurred. This presents a potential conflict in
neural network design: worse calibration from overconfidence may lead to better
robustness to quantization. We perform experiments applying post-training
quantization to a variety of CNNs, on the CIFAR-100 and ImageNet datasets.

    

### [[2111.08164] A Survey on Neural-symbolic Systems](http://arxiv.org/abs/2111.08164)


  In recent years, neural systems have demonstrated superior perceptual
intelligence through highly effective learning, but their reasoning
capabilities remain poor. In contrast, symbolic systems have exceptional
cognitive intelligence through efficient reasoning, but their learning
capabilities are poor. In this case, an ideal intelligent system--a
neural-symbolic system--with high perceptual and cognitive intelligence through
powerful learning and reasoning capabilities gains a growing interest in the
research community. Combining the fast computation ability of neural systems
and the powerful expression ability of symbolic systems, neural-symbolic
systems can perform effective learning and reasoning in multi-domain tasks,
demonstrating concurrent perception and cognition capabilities in intelligent
systems. This paper surveys the latest research in neural-symbolic systems
along four dimensions: the necessity of combination, technical challenges,
methods, and applications. This paper aims to help advance this emerging area
of research by providing researchers with a holistic and comprehensive view,
highlighting the state of art and identifying the opportunities.

    

### [[2111.08165] RapidRead: Global Deployment of State-of-the-art Radiology AI for a Large Veterinary Teleradiology Practice](http://arxiv.org/abs/2111.08165)


  This work describes the development and real-world deployment of a deep
learning-based AI system for evaluating canine and feline radiographs across a
broad range of findings and abnormalities. We describe a new semi-supervised
learning approach that combines NLP-derived labels with self-supervised
training leveraging more than 2.5 million x-ray images. Finally we describe the
clinical deployment of the model including system architecture, real-time
performance evaluation and data drift detection.

    

### [[2111.08168] Explaining medical AI performance disparities across sites with confounder Shapley value analysis](http://arxiv.org/abs/2111.08168)


  Medical AI algorithms can often experience degraded performance when
evaluated on previously unseen sites. Addressing cross-site performance
disparities is key to ensuring that AI is equitable and effective when deployed
on diverse patient populations. Multi-site evaluations are key to diagnosing
such disparities as they can test algorithms across a broader range of
potential biases such as patient demographics, equipment types, and technical
parameters. However, such tests do not explain why the model performs worse.
Our framework provides a method for quantifying the marginal and cumulative
effect of each type of bias on the overall performance difference when a model
is evaluated on external data. We demonstrate its usefulness in a case study of
a deep learning model trained to detect the presence of pneumothorax, where our
framework can help explain up to 60% of the discrepancy in performance across
different sites with known biases like disease comorbidities and imaging
parameters.

    

### [[2111.08169] A Supervised Feature Selection Method For Mixed-Type Data using Density-based Feature Clustering](http://arxiv.org/abs/2111.08169)


  Feature selection methods are widely used to address the high computational
overheads and curse of dimensionality in classifying high-dimensional data.
Most conventional feature selection methods focus on handling homogeneous
features, while real-world datasets usually have a mixture of continuous and
discrete features. Some recent mixed-type feature selection studies only select
features with high relevance to class labels and ignore the redundancy among
features. The determination of an appropriate feature subset is also a
challenge. In this paper, a supervised feature selection method using
density-based feature clustering (SFSDFC) is proposed to obtain an appropriate
final feature subset for mixed-type data. SFSDFC decomposes the feature space
into a set of disjoint feature clusters using a novel density-based clustering
method. Then, an effective feature selection strategy is employed to obtain a
subset of important features with minimal redundancy from those feature
clusters. Extensive experiments as well as comparison studies with five
state-of-the-art methods are conducted on SFSDFC using thirteen real-world
benchmark datasets and results justify the efficacy of the SFSDFC method.

    

### [[2111.08171] Solving Linear Algebra by Program Synthesis](http://arxiv.org/abs/2111.08171)


  We solve MIT's Linear Algebra 18.06 course and Columbia University's
Computational Linear Algebra COMS3251 courses with perfect accuracy by
interactive program synthesis. This surprisingly strong result is achieved by
turning the course questions into programming tasks and then running the
programs to produce the correct answers. We use OpenAI Codex with zero-shot
learning, without providing any examples in the prompts, to synthesize code
from questions. We quantify the difference between the original question text
and the transformed question text that yields a correct answer. Since all
COMS3251 questions are not available online the model is not overfitting. We go
beyond just generating code for questions with numerical answers by
interactively generating code that also results visually pleasing plots as
output. Finally, we automatically generate new questions given a few sample
questions which may be used as new course content. This work is a significant
step forward in solving quantitative math problems and opens the door for
solving many university level STEM courses by machine.

    

### [[2111.08172] Off-Policy Actor-Critic with Emphatic Weightings](http://arxiv.org/abs/2111.08172)


  A variety of theoretically-sound policy gradient algorithms exist for the
on-policy setting due to the policy gradient theorem, which provides a
simplified form for the gradient. The off-policy setting, however, has been
less clear due to the existence of multiple objectives and the lack of an
explicit off-policy policy gradient theorem. In this work, we unify these
objectives into one off-policy objective, and provide a policy gradient theorem
for this unified objective. The derivation involves emphatic weightings and
interest functions. We show multiple strategies to approximate the gradients,
in an algorithm called Actor Critic with Emphatic weightings (ACE). We prove in
a counterexample that previous (semi-gradient) off-policy actor-critic
methods--particularly OffPAC and DPG--converge to the wrong solution whereas
ACE finds the optimal solution. We also highlight why these semi-gradient
approaches can still perform well in practice, suggesting strategies for
variance reduction in ACE. We empirically study several variants of ACE on two
classic control environments and an image-based environment designed to
illustrate the tradeoffs made by each gradient approximation. We find that by
approximating the emphatic weightings directly, ACE performs as well as or
better than OffPAC in all settings tested.

    

### [[2111.08174] ShapeY: Measuring Shape Recognition Capacity Using Nearest Neighbor Matching](http://arxiv.org/abs/2111.08174)


  Object recognition in humans depends primarily on shape cues. We have
developed a new approach to measuring the shape recognition performance of a
vision system based on nearest neighbor view matching within the system's
embedding space. Our performance benchmark, ShapeY, allows for precise control
of task difficulty, by enforcing that view matching span a specified degree of
3D viewpoint change and/or appearance change. As a first test case we measured
the performance of ResNet50 pre-trained on ImageNet. Matching error rates were
high. For example, a 27 degree change in object pitch led ResNet50 to match the
incorrect object 45% of the time. Appearance changes were also highly
disruptive. Examination of false matches indicates that ResNet50's embedding
space is severely "tangled". These findings suggest ShapeY can be a useful tool
for charting the progress of artificial vision systems towards human-level
shape recognition capabilities.

    

### [[2111.08175] Inverse-Weighted Survival Games](http://arxiv.org/abs/2111.08175)


  Deep models trained through maximum likelihood have achieved state-of-the-art
results for survival analysis. Despite this training scheme, practitioners
evaluate models under other criteria, such as binary classification losses at a
chosen set of time horizons, e.g. Brier score (BS) and Bernoulli log likelihood
(BLL). Models trained with maximum likelihood may have poor BS or BLL since
maximum likelihood does not directly optimize these criteria. Directly
optimizing criteria like BS requires inverse-weighting by the censoring
distribution, estimation of which itself also requires inverse-weighted by the
failure distribution. But neither are known. To resolve this dilemma, we
introduce Inverse-Weighted Survival Games to train both failure and censoring
models with respect to criteria such as BS or BLL. In these games, objectives
for each model are built from re-weighted estimates featuring the other model,
where the re-weighting model is held fixed during training. When the loss is
proper, we show that the games always have the true failure and censoring
distributions as a stationary point. This means models in the game do not leave
the correct distributions once reached. We construct one case where this
stationary point is unique. We show that these games optimize BS on simulations
and then apply these principles on real world cancer and critically-ill patient
data.

    

### [[2111.08176] Coarse-to-fine Animal Pose and Shape Estimation](http://arxiv.org/abs/2111.08176)


  Most existing animal pose and shape estimation approaches reconstruct animal
meshes with a parametric SMAL model. This is because the low-dimensional pose
and shape parameters of the SMAL model makes it easier for deep networks to
learn the high-dimensional animal meshes. However, the SMAL model is learned
from scans of toy animals with limited pose and shape variations, and thus may
not be able to represent highly varying real animals well. This may result in
poor fittings of the estimated meshes to the 2D evidences, e.g. 2D keypoints or
silhouettes. To mitigate this problem, we propose a coarse-to-fine approach to
reconstruct 3D animal mesh from a single image. The coarse estimation stage
first estimates the pose, shape and translation parameters of the SMAL model.
The estimated meshes are then used as a starting point by a graph convolutional
network (GCN) to predict a per-vertex deformation in the refinement stage. This
combination of SMAL-based and vertex-based representations benefits from both
parametric and non-parametric representations. We design our mesh refinement
GCN (MRGCN) as an encoder-decoder structure with hierarchical feature
representations to overcome the limited receptive field of traditional GCNs.
Moreover, we observe that the global image feature used by existing animal mesh
reconstruction works is unable to capture detailed shape information for mesh
refinement. We thus introduce a local feature extractor to retrieve a
vertex-level feature and use it together with the global feature as the input
of the MRGCN. We test our approach on the StanfordExtra dataset and achieve
state-of-the-art results. Furthermore, we test the generalization capacity of
our approach on the Animal Pose and BADJA datasets. Our code is available at
the project website.

    

### [[2111.08177] Deep Diffusion Models for Robust Channel Estimation](http://arxiv.org/abs/2111.08177)


  Channel estimation is a critical task in digital communications that greatly
impacts end-to-end system performance. In this work, we introduce a novel
approach for multiple-input multiple-output (MIMO) channel estimation using
deep diffusion models. Our method uses a deep neural network that is trained to
estimate the gradient of the log-likelihood of wireless channels at any point
in high-dimensional space, and leverages this model to solve channel estimation
via posterior sampling. We train a deep diffusion model on channel realizations
from the CDL-D model for two antenna spacings and show that the approach leads
to competitive in- and out-of-distribution performance when compared to
generative adversarial network (GAN) and compressed sensing (CS) methods. When
tested on CDL-C channels which are never seen during training or fine-tuned on,
our approach leads to end-to-end coded performance gains of up to $3$ dB
compared to CS methods and losses of only $0.5$ dB compared to ideal channel
knowledge. To encourage open and reproducible research, our source code is
available at this https URL .

    

### [[2111.08185] Graph neural network-based fault diagnosis: a review](http://arxiv.org/abs/2111.08185)


  Graph neural network (GNN)-based fault diagnosis (FD) has received increasing
attention in recent years, due to the fact that data coming from several
application domains can be advantageously represented as graphs. Indeed, this
particular representation form has led to superior performance compared to
traditional FD approaches. In this review, an easy introduction to GNN,
potential applications to the field of fault diagnosis, and future perspectives
are given. First, the paper reviews neural network-based FD methods by focusing
on their data representations, namely, time-series, images, and graphs. Second,
basic principles and principal architectures of GNN are introduced, with
attention to graph convolutional networks, graph attention networks, graph
sample and aggregate, graph auto-encoder, and spatial-temporal graph
convolutional networks. Third, the most relevant fault diagnosis methods based
on GNN are validated through the detailed experiments, and conclusions are made
that the GNN-based methods can achieve good fault diagnosis performance.
Finally, discussions and future challenges are provided.

    

### [[2111.08190] Learning Augmentation Distributions using Transformed Risk Minimization](http://arxiv.org/abs/2111.08190)


  Adapting to the structure of data distributions (such as symmetry and
transformation invariances) is an important challenge in machine learning.
Invariances can be built into the learning process by architecture design, or
by augmenting the dataset. Both require a priori knowledge about the exact
nature of the symmetries. Absent this knowledge, practitioners resort to
expensive and time-consuming tuning. To address this problem, we propose a new
approach to learn distributions of augmentation transforms, in a new
\emph{Transformed Risk Minimization} (TRM) framework. In addition to predictive
models, we also optimize over transformations chosen from a hypothesis space.
As an algorithmic framework, our TRM method is (1) efficient (jointly learns
augmentations and models in a \emph{single training loop}), (2) modular (works
with \emph{any} training algorithm), and (3) general (handles \emph{both
discrete and continuous} augmentations). We theoretically compare TRM with
standard risk minimization, and give a PAC-Bayes upper bound on its
generalization error. We propose to optimize this bound over a rich
augmentation space via a new parametrization over compositions of blocks,
leading to the new \emph{Stochastic Compositional Augmentation Learning}
(SCALE) algorithm. We compare SCALE experimentally with prior methods (Fast
AutoAugment and Augerino) on CIFAR10/100, SVHN . Additionally, we show that
SCALE can correctly learn certain symmetries in the data distribution
(recovering rotations on rotated MNIST) and can also improve calibration of the
learned model.

    

### [[2111.08195] MoRe-Fi: Motion-robust and Fine-grained Respiration Monitoring via Deep-Learning UWB Radar](http://arxiv.org/abs/2111.08195)


  Crucial for healthcare and biomedical applications, respiration monitoring
often employs wearable sensors in practice, causing inconvenience due to their
direct contact with human bodies. Therefore, researchers have been constantly
searching for contact-free alternatives. Nonetheless, existing contact-free
designs mostly require human subjects to remain static, largely confining their
adoptions in everyday environments where body movements are inevitable.
Fortunately, radio-frequency (RF) enabled contact-free sensing, though
suffering motion interference inseparable by conventional filtering, may offer
a potential to distill respiratory waveform with the help of deep learning. To
realize this potential, we introduce MoRe-Fi to conduct fine-grained
respiration monitoring under body movements. MoRe-Fi leverages an IR-UWB radar
to achieve contact-free sensing, and it fully exploits the complex radar signal
for data augmentation. The core of MoRe-Fi is a novel variational
encoder-decoder network; it aims to single out the respiratory waveforms that
are modulated by body movements in a non-linear manner. Our experiments with 12
subjects and 66-hour data demonstrate that MoRe-Fi accurately recovers
respiratory waveform despite the interference caused by body movements. We also
discuss potential applications of MoRe-Fi for pulmonary disease diagnoses.

    

### [[2111.08202] Learn Locally, Correct Globally: A Distributed Algorithm for Training Graph Neural Networks](http://arxiv.org/abs/2111.08202)


  Despite the recent success of Graph Neural Networks (GNNs), training GNNs on
large graphs remains challenging. The limited resource capacities of the
existing servers, the dependency between nodes in a graph, and the privacy
concern due to the centralized storage and model learning have spurred the need
to design an effective distributed algorithm for GNN training. However,
existing distributed GNN training methods impose either excessive communication
costs or large memory overheads that hinders their scalability. To overcome
these issues, we propose a communication-efficient distributed GNN training
technique named $\text{Learn Locally, Correct Globally}$ (LLCG). To reduce
the communication and memory overhead, each local machine in LLCG first trains
a GNN on its local data by ignoring the dependency between nodes among
different machines, then sends the locally trained model to the server for
periodic model averaging. However, ignoring node dependency could result in
significant performance degradation. To solve the performance degradation, we
propose to apply $\text{Global Server Corrections}$ on the server to refine
the locally learned models. We rigorously analyze the convergence of
distributed methods with periodic model averaging for training GNNs and show
that naively applying periodic model averaging but ignoring the dependency
between nodes will suffer from an irreducible residual error. However, this
residual error can be eliminated by utilizing the proposed global corrections
to entail fast convergence rate. Extensive experiments on real-world datasets
show that LLCG can significantly improve the efficiency without hurting the
performance.

    

### [[2111.08211] FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning](http://arxiv.org/abs/2111.08211)


  Federated learning (FL) aims to protect data privacy by enabling clients to
collaboratively build machine learning models without sharing their private
data. However, recent works demonstrate that FL is vulnerable to gradient-based
data recovery attacks. Varieties of privacy-preserving technologies have been
leveraged to further enhance the privacy of FL. Nonetheless, they either are
computational or communication expensive (e.g., homomorphic encryption) or
suffer from precision loss (e.g., differential privacy). In this work, we
propose \textsc{FedCG}, a novel \underline{fed}erated learning method that
leverages \underline{c}onditional \underline{g}enerative adversarial networks
to achieve high-level privacy protection while still maintaining competitive
model performance. More specifically, \textsc{FedCG} decomposes each client's
local network into a private extractor and a public classifier and keeps the
extractor local to protect privacy. Instead of exposing extractors which is the
culprit of privacy leakage, \textsc{FedCG} shares clients' generators with the
server for aggregating common knowledge aiming to enhance the performance of
clients' local networks. Extensive experiments demonstrate that \textsc{FedCG}
can achieve competitive model performance compared with baseline FL methods,
and numerical privacy analysis shows that \textsc{FedCG} has high-level
privacy-preserving capability.

    

### [[2111.08221] Fairness-aware Online Price Discrimination with Nonparametric Demand Models](http://arxiv.org/abs/2111.08221)


  Price discrimination, which refers to the strategy of setting different
prices for different customer groups, has been widely used in online retailing.
Although it helps boost the collected revenue for online retailers, it might
create serious concern in fairness, which even violates the regulation and law.
This paper studies the problem of dynamic discriminatory pricing under fairness
constraints. In particular, we consider a finite selling horizon of length $T$
for a single product with two groups of customers. Each group of customers has
its unknown demand function that needs to be learned. For each selling period,
the seller determines the price for each group and observes their purchase
behavior. While existing literature mainly focuses on maximizing revenue,
ensuring fairness among different customers has not been fully explored in the
dynamic pricing literature. In this work, we adopt the fairness notion from
(Cohen et al. 2021a). For price fairness, we propose an optimal dynamic pricing
policy in terms of regret, which enforces the strict price fairness constraint.
In contrast to the standard $\sqrt{T}$-type regret in online learning, we show
that the optimal regret in our case is $\tilde{\Theta}(T^{4/5})$. We further
extend our algorithm to a more general notion of fairness, which includes
demand fairness as a special case. To handle this general class, we propose a
soft fairness constraint and develop the dynamic pricing policy that achieves
$\tilde{O}(T^{4/5})$ regret.

    

### [[2111.08226] Comparative Analysis of Machine Learning Models for Predicting Travel Time](http://arxiv.org/abs/2111.08226)


  In this paper, five different deep learning models are being compared for
predicting travel time. These models are autoregressive integrated moving
average (ARIMA) model, recurrent neural network (RNN) model, autoregressive
(AR) model, Long-short term memory (LSTM) model, and gated recurrent units
(GRU) model. The aim of this study is to investigate the performance of each
developed model for forecasting travel time. The dataset used in this paper
consists of travel time and travel speed information from the state of
Missouri. The learning rate used for building each model was varied from
0.0001-0.01. The best learning rate was found to be 0.001. The study concluded
that the ARIMA model was the best model architecture for travel time prediction
and forecasting.

    

### [[2111.08227] Phase function estimation from a diffuse optical image via deep learning](http://arxiv.org/abs/2111.08227)


  The phase function is a key element of a light propagation model for Monte
Carlo (MC) simulation, which is usually fitted with an analytic function with
associated parameters. In recent years, machine learning methods were reported
to estimate the parameters of the phase function of a particular form such as
the Henyey-Greenstein phase function but, to our knowledge, no studies have
been performed to determine the form of the phase function. Here we design a
convolutional neural network to estimate the phase function from a diffuse
optical image without any explicit assumption on the form of the phase
function. Specifically, we use a Gaussian mixture model as an example to
represent the phase function generally and learn the model parameters
accurately. The Gaussian mixture model is selected because it provides the
analytic expression of phase function to facilitate deflection angle sampling
in MC simulation, and does not significantly increase the number of free
parameters. Our proposed method is validated on MC-simulated reflectance images
of typical biological tissues using the Henyey-Greenstein phase function with
different anisotropy factors. The effects of field of view (FOV) and spatial
resolution on the errors are analyzed to optimize the estimation method. The
mean squared error of the phase function is 0.01 and the relative error of the
anisotropy factor is 3.28%.

    

### [[2111.08228] SStaGCN: Simplified stacking based graph convolutional networks](http://arxiv.org/abs/2111.08228)


  Graph convolutional network (GCN) is a powerful model studied broadly in
various graph structural data learning tasks. However, to mitigate the
over-smoothing phenomenon, and deal with heterogeneous graph structural data,
the design of GCN model remains a crucial issue to be investigated. In this
paper, we propose a novel GCN called SStaGCN (Simplified stacking based GCN) by
utilizing the ideas of stacking and aggregation, which is an adaptive general
framework for tackling heterogeneous graph data. Specifically, we first use the
base models of stacking to extract the node features of a graph. Subsequently,
aggregation methods such as mean, attention and voting techniques are employed
to further enhance the ability of node features extraction. Thereafter, the
node features are considered as inputs and fed into vanilla GCN model.
Furthermore, theoretical generalization bound analysis of the proposed model is
explicitly given. Extensive experiments on $3$ public citation networks and
another $3$ heterogeneous tabular data demonstrate the effectiveness and
efficiency of the proposed approach over state-of-the-art GCNs. Notably, the
proposed SStaGCN can efficiently mitigate the over-smoothing problem of GCN.

    

### [[2111.08230] Selective Ensembles for Consistent Predictions](http://arxiv.org/abs/2111.08230)


  Recent work has shown that models trained to the same objective, and which
achieve similar measures of accuracy on consistent test data, may nonetheless
behave very differently on individual predictions. This inconsistency is
undesirable in high-stakes contexts, such as medical diagnosis and finance. We
show that this inconsistent behavior extends beyond predictions to feature
attributions, which may likewise have negative implications for the
intelligibility of a model, and one's ability to find recourse for subjects. We
then introduce selective ensembles to mitigate such inconsistencies by applying
hypothesis testing to the predictions of a set of models trained using
randomly-selected starting conditions; importantly, selective ensembles can
abstain in cases where a consistent outcome cannot be achieved up to a
specified confidence level. We prove that that prediction disagreement between
selective ensembles is bounded, and empirically demonstrate that selective
ensembles achieve consistent predictions and feature attributions while
maintaining low abstention rates. On several benchmark datasets, selective
ensembles reach zero inconsistently predicted points, with abstention rates as
low 1.5%.

    

### [[2111.08234] Covariate Shift in High-Dimensional Random Feature Regression](http://arxiv.org/abs/2111.08234)


  A significant obstacle in the development of robust machine learning models
is covariate shift, a form of distribution shift that occurs when the input
distributions of the training and test sets differ while the conditional label
distributions remain the same. Despite the prevalence of covariate shift in
real-world applications, a theoretical understanding in the context of modern
machine learning has remained lacking. In this work, we examine the exact
high-dimensional asymptotics of random feature regression under covariate shift
and present a precise characterization of the limiting test error, bias, and
variance in this setting. Our results motivate a natural partial order over
covariate shifts that provides a sufficient condition for determining when the
shift will harm (or even help) test performance. We find that overparameterized
models exhibit enhanced robustness to covariate shift, providing one of the
first theoretical explanations for this intriguing phenomenon. Additionally,
our analysis reveals an exact linear relationship between in-distribution and
out-of-distribution generalization performance, offering an explanation for
this surprising recent empirical observation.

    

### [[2111.08239] Assessing Deep Neural Networks as Probability Estimators](http://arxiv.org/abs/2111.08239)


  Deep Neural Networks (DNNs) have performed admirably in classification tasks.
However, the characterization of their classification uncertainties, required
for certain applications, has been lacking. In this work, we investigate the
issue by assessing DNNs' ability to estimate conditional probabilities and
propose a framework for systematic uncertainty characterization. Denoting the
input sample as x and the category as y, the classification task of assigning a
category y to a given input x can be reduced to the task of estimating the
conditional probabilities p(y|x), as approximated by the DNN at its last layer
using the softmax function. Since softmax yields a vector whose elements all
fall in the interval (0, 1) and sum to 1, it suggests a probabilistic
interpretation to the DNN's outcome. Using synthetic and real-world datasets,
we look into the impact of various factors, e.g., probability density f(x) and
inter-categorical sparsity, on the precision of DNNs' estimations of p(y|x),
and find that the likelihood probability density and the inter-categorical
sparsity have greater impacts than the prior probability to DNNs'
classification uncertainty.

    

### [[2111.08249] Bengali Handwritten Grapheme Classification: Deep Learning Approach](http://arxiv.org/abs/2111.08249)


  Despite being one of the most spoken languages in the world ($6^{th}$ based
on population), research regarding Bengali handwritten grapheme (smallest
functional unit of a writing system) classification has not been explored
widely compared to other prominent languages. Moreover, the large number of
combinations of graphemes in the Bengali language makes this classification
task very challenging. With an effort to contribute to this research problem,
we participate in a Kaggle competition \cite{kaggle_link} where the challenge
is to separately classify three constituent elements of a Bengali grapheme in
the image: grapheme root, vowel diacritics, and consonant diacritics. We
explore the performances of some existing neural network models such as
Multi-Layer Perceptron (MLP) and state of the art ResNet50. To further improve
the performance we propose our own convolution neural network (CNN) model for
Bengali grapheme classification with validation root accuracy 95.32\%, vowel
accuracy 98.61\%, and consonant accuracy 98.76\%. We also explore Region
Proposal Network (RPN) using VGGNet with a limited setting that can be a
potential future direction to improve the performance.

    

### [[2111.08251] Enabling equivariance for arbitrary Lie groups](http://arxiv.org/abs/2111.08251)


  Although provably robust to translational perturbations, convolutional neural
networks (CNNs) are known to suffer from extreme performance degradation when
presented at test time with more general geometric transformations of inputs.
Recently, this limitation has motivated a shift in focus from CNNs to Capsule
Networks (CapsNets). However, CapsNets suffer from admitting relatively few
theoretical guarantees of invariance. We introduce a rigourous mathematical
framework to permit invariance to any Lie group of warps, exclusively using
convolutions (over Lie groups), without the need for capsules. Previous work on
group convolutions has been hampered by strong assumptions about the group,
which precludes the application of such techniques to common warps in computer
vision such as affine and homographic. Our framework enables the implementation
of group convolutions over \emph{any} finite-dimensional Lie group. We
empirically validate our approach on the benchmark affine-invariant
classification task, where we achieve $\sim$30\% improvement in accuracy
against conventional CNNs while outperforming the state-of-the-art CapsNet. As
further illustration of the generality of our framework, we train a
homography-convolutional model which achieves superior robustness on a
homography-perturbed dataset, where CapsNet results degrade.

    

### [[2111.08255] A Unified and Fast Interpretable Model for Predictive Analytics](http://arxiv.org/abs/2111.08255)


  In this paper, we propose FXAM (Fast and eXplainable Additive Model), a
unified and fast interpretable model for predictive analytics. FXAM extends
GAM's (Generalized Additive Model) modeling capability with a unified additive
model for numerical, categorical, and temporal features. FXAM conducts a novel
training procedure called Three-Stage Iteration (TSI). The three stages
correspond to learning over numerical, categorical and temporal features
respectively. Each stage learns a local optimum by fixing parameters of other
stages. We design joint learning over categorical features and partial learning
over temporal features to achieve high accuracy and training efficiency. We
prove that TSI is guaranteed to converge to global optimum. We further propose
a set of optimization techniques to speed up FXAM's training algorithm to meet
the needs of interactive analysis. Evaluations verify that FXAM significantly
outperforms existing GAMs in terms of training speed and modeling categorical
and temporal features.

    

### [[2111.08267] Solving Probability and Statistics Problems by Program Synthesis](http://arxiv.org/abs/2111.08267)


  We solve university level probability and statistics questions by program
synthesis using OpenAI's Codex, a Transformer trained on text and fine-tuned on
code. We transform course problems from MIT's 18.05 Introduction to Probability
and Statistics and Harvard's STAT110 Probability into programming tasks. We
then execute the generated code to get a solution. Since these course questions
are grounded in probability, we often aim to have Codex generate probabilistic
programs that simulate a large number of probabilistic dependencies to compute
its solution. Our approach requires prompt engineering to transform the
question from its original form to an explicit, tractable form that results in
a correct program and solution. To estimate the amount of work needed to
translate an original question into its tractable form, we measure the
similarity between original and transformed questions. Our work is the first to
introduce a new dataset of university-level probability and statistics problems
and solve these problems in a scalable fashion using the program synthesis
capabilities of large language models.

    

### [[2111.08268] Pre-training Graph Neural Network for Cross Domain Recommendation](http://arxiv.org/abs/2111.08268)


  A recommender system predicts users' potential interests in items, where the
core is to learn user/item embeddings. Nevertheless, it suffers from the
data-sparsity issue, which the cross-domain recommendation can alleviate.
However, most prior works either jointly learn the source domain and target
domain models, or require side-features. However, jointly training and side
features would affect the prediction on the target domain as the learned
embedding is dominated by the source domain containing bias information.
Inspired by the contemporary arts in pre-training from graph representation
learning, we propose a pre-training and fine-tuning diagram for cross-domain
recommendation. We devise a novel Pre-training Graph Neural Network for
Cross-Domain Recommendation (PCRec), which adopts the contrastive
self-supervised pre-training of a graph encoder. Then, we transfer the
pre-trained graph encoder to initialize the node embeddings on the target
domain, which benefits the fine-tuning of the single domain recommender system
on the target domain. The experimental results demonstrate the superiority of
PCRec. Detailed analyses verify the superiority of PCRec in transferring
information while avoiding biases from source domains.

    

### [[2111.08274] HADFL: Heterogeneity-aware Decentralized Federated Learning Framework](http://arxiv.org/abs/2111.08274)


  Federated learning (FL) supports training models on geographically
distributed devices. However, traditional FL systems adopt a centralized
synchronous strategy, putting high communication pressure and model
generalization challenge. Existing optimizations on FL either fail to speedup
training on heterogeneous devices or suffer from poor communication efficiency.
In this paper, we propose HADFL, a framework that supports decentralized
asynchronous training on heterogeneous devices. The devices train model locally
with heterogeneity-aware local steps using local data. In each aggregation
cycle, they are selected based on probability to perform model synchronization
and aggregation. Compared with the traditional FL system, HADFL can relieve the
central server's communication pressure, efficiently utilize heterogeneous
computing power, and can achieve a maximum speedup of 3.15x than
decentralized-FedAvg and 4.68x than Pytorch distributed training scheme,
respectively, with almost no loss of convergence accuracy.

    

### [[2111.08275] Deep Distilling: automated code generation using explainable deep learning](http://arxiv.org/abs/2111.08275)


  Human reasoning can distill principles from observed patterns and generalize
them to explain and solve novel problems. The most powerful artificial
intelligence systems lack explainability and symbolic reasoning ability, and
have therefore not achieved supremacy in domains requiring human understanding,
such as science or common sense reasoning. Here we introduce deep distilling, a
machine learning method that learns patterns from data using explainable deep
learning and then condenses it into concise, executable computer code. The
code, which can contain loops, nested logical statements, and useful
intermediate variables, is equivalent to the neural network but is generally
orders of magnitude more compact and human-comprehensible. On a diverse set of
problems involving arithmetic, computer vision, and optimization, we show that
deep distilling generates concise code that generalizes out-of-distribution to
solve problems orders-of-magnitude larger and more complex than the training
data. For problems with a known ground-truth rule set, deep distilling
discovers the rule set exactly with scalable guarantees. For problems that are
ambiguous or computationally intractable, the distilled rules are similar to
existing human-derived algorithms and perform at par or better. Our approach
demonstrates that unassisted machine intelligence can build generalizable and
intuitive rules explaining patterns in large datasets that would otherwise
overwhelm human reasoning.

    

### [[2111.08277] Wyner-Ziv Gradient Compression for Federated Learning](http://arxiv.org/abs/2111.08277)


  Due to limited communication resources at the client and a massive number of
model parameters, large-scale distributed learning tasks suffer from
communication bottleneck. Gradient compression is an effective method to reduce
communication load by transmitting compressed gradients. Motivated by the fact
that in the scenario of stochastic gradients descent, gradients between
adjacent rounds may have a high correlation since they wish to learn the same
model, this paper proposes a practical gradient compression scheme for
federated learning, which uses historical gradients to compress gradients and
is based on Wyner-Ziv coding but without any probabilistic assumption. We also
implement our gradient quantization method on the real dataset, and the
performance of our method is better than the previous schemes.

    

### [[2111.08291] Switching Recurrent Kalman Networks](http://arxiv.org/abs/2111.08291)


  Forecasting driving behavior or other sensor measurements is an essential
component of autonomous driving systems. Often real-world multivariate time
series data is hard to model because the underlying dynamics are nonlinear and
the observations are noisy. In addition, driving data can often be multimodal
in distribution, meaning that there are distinct predictions that are likely,
but averaging can hurt model performance. To address this, we propose the
Switching Recurrent Kalman Network (SRKN) for efficient inference and
prediction on nonlinear and multi-modal time-series data. The model switches
among several Kalman filters that model different aspects of the dynamics in a
factorized latent state. We empirically test the resulting scalable and
interpretable deep state-space model on toy data sets and real driving data
from taxis in Porto. In all cases, the model can capture the multimodal nature
of the dynamics in the data.

    

### [[2111.08295] Machine Learning-Based Assessment of Energy Behavior of RC Shear Walls](http://arxiv.org/abs/2111.08295)


  Current seismic design codes primarily rely on the strength and displacement
capacity of structural members and do not account for the influence of the
ground motion duration or the hysteretic behavior characteristics. The
energy-based approach serves as a supplemental index to response quantities and
includes the effect of repeated loads in seismic performance. The design
philosophy suggests that the seismic demands are met by the energy dissipation
capacity of the structural members. Therefore, the energy dissipation behavior
of the structural members should be well understood to achieve an effective
energy-based design approach. This study focuses on the energy dissipation
capacity of reinforced concrete (RC) shear walls that are widely used in high
seismic regions as they provide significant stiffness and strength to resist
lateral forces. A machine learning (Gaussian Process Regression (GPR))-based
predictive model for energy dissipation capacity of shear walls is developed as
a function of wall design parameters. Eighteen design parameters are shown to
influence energy dissipation, whereas the most important ones are determined by
applying sequential backward elimination and by using feature selection methods
to reduce the complexity of the predictive model. The ability of the proposed
model to make robust and accurate predictions is validated based on novel data
with a prediction accuracy (the ratio of predicted/actual values) of around
1.00 and a coefficient of determination (R2) of 0.93. The outcomes of this
study are believed to contribute to the energy-based approach by (i) defining
the most influential wall properties on the seismic energy dissipation capacity
of shear walls and (ii) providing predictive models that can enable comparisons
of different wall design configurations to achieve higher energy dissipation
capacity.

    

### [[2111.08308] Learning with convolution and pooling operations in kernel methods](http://arxiv.org/abs/2111.08308)


  Recent empirical work has shown that hierarchical convolutional kernels
inspired by convolutional neural networks (CNNs) significantly improve the
performance of kernel methods in image classification tasks. A widely accepted
explanation for the success of these architectures is that they encode
hypothesis classes that are suitable for natural images. However, understanding
the precise interplay between approximation and generalization in convolutional
architectures remains a challenge. In this paper, we consider the stylized
setting of covariates (image pixels) uniformly distributed on the hypercube,
and fully characterize the RKHS of kernels composed of single layers of
convolution, pooling, and downsampling operations. We then study the gain in
sample efficiency of kernel methods using these kernels over standard
inner-product kernels. In particular, we show that 1) the convolution layer
breaks the curse of dimensionality by restricting the RKHS to `local'
functions; 2) local pooling biases learning towards low-frequency functions,
which are stable by small translations; 3) downsampling may modify the
high-frequency eigenspaces but leaves the low-frequency part approximately
unchanged. Notably, our results quantify how choosing an architecture adapted
to the target function leads to a large improvement in the sample complexity.

    

### [[2111.08330] Bayesian Optimization for Cascade-type Multi-stage Processes](http://arxiv.org/abs/2111.08330)


  Complex processes in science and engineering are often formulated as
multi-stage decision-making problems. In this paper, we consider a type of
multi-stage decision-making process called a cascade process. A cascade process
is a multi-stage process in which the output of one stage is used as an input
for the next stage. When the cost of each stage is expensive, it is difficult
to search for the optimal controllable parameters for each stage exhaustively.
To address this problem, we formulate the optimization of the cascade process
as an extension of Bayesian optimization framework and propose two types of
acquisition functions (AFs) based on credible intervals and expected
improvement. We investigate the theoretical properties of the proposed AFs and
demonstrate their effectiveness through numerical experiments. In addition, we
consider an extension called suspension setting in which we are allowed to
suspend the cascade process at the middle of the multi-stage decision-making
process that often arises in practical problems. We apply the proposed method
in the optimization problem of the solar cell simulator, which was the
motivation for this study.

    

### [[2111.08333] Reshaping Smart Energy Transition: An analysis of human-building interactions in Qatar Using Machine Learning Techniques](http://arxiv.org/abs/2111.08333)


  Policy Planning have the potential to contribute to the strategic development
and economic diversification of developing countries even without considerable
structural changes. In this study, we analyzed a set of human-oriented
dimensions aimed at improving energy policies related to the building sector in
Qatar. Considering the high percentage of expatriate and migrant communities
with different financial and cultural backgrounds and behavioral patterns
compared with local communities in the GCC Union, it is required to investigate
human dimensions to propose adequate energy policies. This study explored the
correlations of socioeconomic, behavioral, and demographic dimensions to
determine the main factors behind discrepancies in energy use,
responsibilities, motivations, habits, and overall well-being. The sample
included 2,200 people in Qatar, and it was clustered into two consumer
categories: high and low. In particular, the study focused on exploring human
indoor comfort perception dependencies with building features. Financial
drivers, such as demand programs and energy subsidies, were explored in
relation to behavioral patterns. Subsequently, the data analysis resulted in
implications for energy policies regarding interventions, social well-being,
and awareness. Machine learning methods were used to perform a feature
importance analysis to determine the main factors of human behavior. The
findings of this study demonstrated how human factors impact comfort perception
in residential and work environments, norms, habits, self-responsibility,
consequence awareness, and consumption. The study has important implications
for developing targeted strategies aimed at improving the efficacy of energy
policies and sustainability performance indicators.

    

### [[2111.08344] Mathematical Models for Local Sensing Hashes](http://arxiv.org/abs/2111.08344)


  As data volumes continue to grow, searches in data are becoming increasingly
time-consuming. Classical index structures for neighbor search are no longer
sustainable due to the "curse of dimensionality". Instead, approximated index
structures offer a good opportunity to significantly accelerate the neighbor
search for clustering and outlier detection and to have the lowest possible
error rate in the results of the algorithms. Local sensing hashes is one of
those. We indicate directions to mathematically model the properties of it.

    

### [[2111.08356] Inference-Time Personalized Federated Learning](http://arxiv.org/abs/2111.08356)


  In Federated learning (FL), multiple clients collaborate to learn a model
through a central server but keep the data decentralized. Personalized
federated learning (PFL) further extends FL to handle data heterogeneity
between clients by learning personalized models. In both FL and PFL, all
clients participate in the training process and their labeled data is used for
training. However, in reality, novel clients may wish to join a prediction
service after it has been deployed, obtaining predictions for their own
unlabeled data.
Here, we defined a new learning setup, Inference-Time PFL (IT-PFL), where a
model trained on a set of clients, needs to be later evaluated on novel
unlabeled clients at inference time. We propose a novel approach to this
problem IT-PFL-HN, based on a hypernetwork module and an encoder module.
Specifically, we train an encoder network that learns a representation for a
client given its unlabeled data. That client representation is fed to a
hypernetwork that generates a personalized model for that client. Evaluated on
four benchmark datasets, we find that IT-PFL-HN generalizes better than current
FL and PFL methods, especially when the novel client has a large domain shift.
We also analyzed the generalization error for the novel client, showing how it
can be bounded using results from multi-task learning and domain adaptation.
Finally, since novel clients do not contribute their data to training, they can
potentially have better control over their data privacy; indeed, we showed
analytically and experimentally how novel clients can apply differential
privacy to their data.

    

### [[2111.08386] Towards Generating Real-World Time Series Data](http://arxiv.org/abs/2111.08386)


  Time series data generation has drawn increasing attention in recent years.
Several generative adversarial network (GAN) based methods have been proposed
to tackle the problem usually with the assumption that the targeted time series
data are well-formatted and complete. However, real-world time series (RTS)
data are far away from this utopia, e.g., long sequences with variable lengths
and informative missing data raise intractable challenges for designing
powerful generation algorithms. In this paper, we propose a novel generative
framework for RTS data - RTSGAN to tackle the aforementioned challenges. RTSGAN
first learns an encoder-decoder module which provides a mapping between a time
series instance and a fixed-dimension latent vector and then learns a
generation module to generate vectors in the same latent space. By combining
the generator and the decoder, RTSGAN is able to generate RTS which respect the
original feature distributions and the temporal dynamics. To generate time
series with missing values, we further equip RTSGAN with an observation
embedding layer and a decide-and-generate decoder to better utilize the
informative missing patterns. Experiments on the four RTS datasets show that
the proposed framework outperforms the previous generation methods in terms of
synthetic data utility for downstream classification and prediction tasks.

    

### [[2111.08409] Grounding Psychological Shape Space in Convolutional Neural Networks](http://arxiv.org/abs/2111.08409)


  Shape information is crucial for human perception and cognition, and should
therefore also play a role in cognitive AI systems. We employ the
interdisciplinary framework of conceptual spaces, which proposes a geometric
representation of conceptual knowledge through low-dimensional interpretable
similarity spaces. These similarity spaces are often based on psychological
dissimilarity ratings for a small set of stimuli, which are then transformed
into a spatial representation by a technique called multidimensional scaling.
Unfortunately, this approach is incapable of generalizing to novel stimuli. In
this paper, we use convolutional neural networks to learn a generalizable
mapping between perceptual inputs (pixels of grayscale line drawings) and a
recently proposed psychological similarity space for the shape domain. We
investigate different network architectures (classification network vs.
autoencoder) and different training regimes (transfer learning vs. multi-task
learning). Our results indicate that a classification-based multi-task learning
scenario yields the best results, but that its performance is relatively
sensitive to the dimensionality of the similarity space.

    

### [[2111.08410] Thoughts on the Consistency between Ricci Flow and Neural Network Behavior](http://arxiv.org/abs/2111.08410)


  The Ricci flow is a partial differential equation for evolving the metric in
a Riemannian manifold to make it more regular. However, in most cases, the
Ricci flow tends to develop singularities and lead to divergence of the
solution. In this paper, we propose the linearly nearly Euclidean metric to
assist manifold micro-surgery, which means that we prove the dynamical
stability and convergence of the metrics close to the linearly nearly Euclidean
metric under the Ricci-DeTurck flow. In practice, from the information geometry
and mirror descent points of view, we give the steepest descent gradient flow
for neural networks on the linearly nearly Euclidean manifold. During the
training process of the neural network, we observe that its metric will also
regularly converge to the linearly nearly Euclidean metric, which is consistent
with the convergent behavior of linearly nearly Euclidean manifolds under
Ricci-DeTurck flow.

    

### [[2111.08415] Causal policy ranking](http://arxiv.org/abs/2111.08415)


  Policies trained via reinforcement learning (RL) are often very complex even
for simple tasks. In an episode with $n$ time steps, a policy will make $n$
decisions on actions to take, many of which may appear non-intuitive to the
observer. Moreover, it is not clear which of these decisions directly
contribute towards achieving the reward and how significant is their
contribution. Given a trained policy, we propose a black-box method based on
counterfactual reasoning that estimates the causal effect that these decisions
have on reward attainment and ranks the decisions according to this estimate.
In this preliminary work, we compare our measure against an alternative,
non-causal, ranking procedure, highlight the benefits of causality-based policy
ranking, and discuss potential future work integrating causal algorithms into
the interpretation of RL agent policies.

    

### [[2111.08435] Free Will Belief as a consequence of Model-based Reinforcement Learning](http://arxiv.org/abs/2111.08435)


  The debate on whether or not humans have free will has been raging for
centuries. Although there are good arguments based on our current understanding
of the laws of nature for the view that it is not possible for humans to have
free will, most people believe they do. This discrepancy begs for an
explanation. If we accept that we do not have free will, we are faced with two
problems: (1) while freedom is a very commonly used concept that everyone
intuitively understands, what are we actually referring to when we say that an
action or choice is "free" or not? And, (2) why is the belief in free will so
common? Where does this belief come from, and what is its purpose, if any? In
this paper, we examine these questions from the perspective of reinforcement
learning (RL). RL is a framework originally developed for training artificial
intelligence agents. However, it can also be used as a computational model of
human decision making and learning, and by doing so, we propose that the first
problem can be answered by observing that people's common sense understanding
of freedom is closely related to the information entropy of an RL agent's
normalized action values, while the second can be explained by the necessity
for agents to model themselves as if they could have taken decisions other than
those they actually took, when dealing with the temporal credit assignment
problem. Put simply, we suggest that by applying the RL framework as a model
for human learning it becomes evident that in order for us to learn efficiently
and be intelligent we need to view ourselves as if we have free will.

    

### [[2111.08438] Fourier Neural Networks for Function Approximation](http://arxiv.org/abs/2111.08438)


  The success of Neural networks in providing miraculous results when applied
to a wide variety of tasks is astonishing. Insight in the working can be
obtained by studying the universal approximation property of neural networks.
It is proved extensively that neural networks are universal approximators.
Further it is proved that deep Neural networks are better approximators. It is
specifically proved that for a narrow neural network to approximate a function
which is otherwise implemented by a deep Neural network, the network take
exponentially large number of neurons. In this work, we have implemented
existing methodologies for a variety of synthetic functions and identified
their deficiencies. Further, we examined that Fourier neural network is able to
perform fairly good with only two layers in the neural network. A modified
Fourier Neural network which has sinusoidal activation and two hidden layer is
proposed and the results are tabulated.

    

### [[2111.08440] On the Importance of Difficulty Calibration in Membership Inference Attacks](http://arxiv.org/abs/2111.08440)


  The vulnerability of machine learning models to membership inference attacks
has received much attention in recent years. However, existing attacks mostly
remain impractical due to having high false positive rates, where non-member
samples are often erroneously predicted as members. This type of error makes
the predicted membership signal unreliable, especially since most samples are
non-members in real world applications. In this work, we argue that membership
inference attacks can benefit drastically from \emph{difficulty calibration},
where an attack's predicted membership score is adjusted to the difficulty of
correctly classifying the target sample. We show that difficulty calibration
can significantly reduce the false positive rate of a variety of existing
attacks without a loss in accuracy.

    

### [[2111.08441] Use of machine learning in geriatric clinical care for chronic diseases: a systematic literature review](http://arxiv.org/abs/2111.08441)


  Objectives-Geriatric clinical care is a multidisciplinary assessment designed
to evaluate older patients (age 65 years and above) functional ability,
physical health, and cognitive wellbeing. The majority of these patients suffer
from multiple chronic conditions and require special attention. Recently,
hospitals utilize various artificial intelligence (AI) systems to improve care
for elderly patients. The purpose of this systematic literature review is to
understand the current use of AI systems, particularly machine learning (ML),
in geriatric clinical care for chronic diseases. Materials and Methods-We
restricted our search to eight databases, namely PubMed, WorldCat, MEDLINE,
ProQuest, ScienceDirect, SpringerLink, Wiley, and ERIC, to analyze research
articles published in English between January 2010 and June 2019. We focused on
studies that used ML algorithms in the care of geriatrics patients with chronic
conditions. Results-We identified 35 eligible studies and classified in three
groups-psychological disorder (n=22), eye diseases (n=6), and others (n=7).
This review identified the lack of standardized ML evaluation metrics and the
need for data governance specific to health care applications. Conclusion- More
studies and ML standardization tailored to health care applications are
required to confirm whether ML could aid in improving geriatric clinical care.

    

### [[2111.08446] Automatic Sleep Staging: Recent Development, Challenges, and Future Directions](http://arxiv.org/abs/2111.08446)


  Modern deep learning holds a great potential to transform clinical practice
on human sleep. Teaching a machine to carry out routine tasks would be a
tremendous reduction in workload for clinicians. Sleep staging, a fundamental
step in sleep practice, is a suitable task for this and will be the focus in
this article. Recently, automatic sleep staging systems have been trained to
mimic manual scoring, leading to similar performance to human sleep experts, at
least on scoring of healthy subjects. Despite tremendous progress, we have not
seen automatic sleep scoring adopted widely in clinical environments. This
review aims to give a shared view of the authors on the most recent
state-of-the-art development in automatic sleep staging, the challenges that
still need to be addressed, and the future directions for automatic sleep
scoring to achieve clinical value.

    

### [[2111.08449] Complementary Ensemble Learning](http://arxiv.org/abs/2111.08449)


  To achieve high performance of a machine learning (ML) task, a deep
learning-based model must implicitly capture the entire distribution from data.
Thus, it requires a huge amount of training samples, and data are expected to
fully present the real distribution, especially for high dimensional data,
e.g., images, videos. In practice, however, data are usually collected with a
diversity of styles, and several of them have insufficient number of
representatives. This might lead to uncertainty in models' prediction, and
significantly reduce ML task performance.
In this paper, we provide a comprehensive study on this problem by looking at
model uncertainty. From this, we derive a simple but efficient technique to
improve performance of state-of-the-art deep learning models. Specifically, we
train auxiliary models which are able to complement state-of-the-art model
uncertainty. As a result, by assembling these models, we can significantly
improve the ML task performance for types of data mentioned earlier. While
slightly improving ML classification accuracy on benchmark datasets (e.g., 0.2%
on MNIST), our proposed method significantly improves on limited data (i.e.,
1.3% on Eardrum and 3.5% on ChestXray).

    

### [[2111.08450] A Spatial-temporal Graph Deep Learning Model for Urban Flood Nowcasting Leveraging Heterogeneous Community Features](http://arxiv.org/abs/2111.08450)


  The objective of this study is to develop and test a novel structured
deep-learning modeling framework for urban flood nowcasting by integrating
physics-based and human-sensed features. We present a new computational
modeling framework including an attention-based spatial-temporal graph
convolution network (ASTGCN) model and different streams of data that are
collected in real-time, preprocessed, and fed into the model to consider
spatial and temporal information and dependencies that improve flood
nowcasting. The novelty of the computational modeling framework is threefold;
first, the model is capable of considering spatial and temporal dependencies in
inundation propagation thanks to the spatial and temporal graph convolutional
modules; second, it enables capturing the influence of heterogeneous temporal
data streams that can signal flooding status, including physics-based features
such as rainfall intensity and water elevation, and human-sensed data such as
flood reports and fluctuations of human activity. Third, its attention
mechanism enables the model to direct its focus on the most influential
features that vary dynamically. We show the application of the modeling
framework in the context of Harris County, Texas, as the case study and
Hurricane Harvey as the flood event. Results indicate that the model provides
superior performance for the nowcasting of urban flood inundation at the census
tract level, with a precision of 0.808 and a recall of 0.891, which shows the
model performs better compared with some other novel models. Moreover, ASTGCN
model performance improves when heterogeneous dynamic features are added into
the model that solely relies on physics-based features, which demonstrates the
promise of using heterogenous human-sensed data for flood nowcasting,

    

### [[2111.08451] Which is Making the Contribution: Modulating Unimodal and Cross-modal Dynamics for Multimodal Sentiment Analysis](http://arxiv.org/abs/2111.08451)


  Multimodal sentiment analysis (MSA) draws increasing attention with the
availability of multimodal data. The boost in performance of MSA models is
mainly hindered by two problems. On the one hand, recent MSA works mostly focus
on learning cross-modal dynamics, but neglect to explore an optimal solution
for unimodal networks, which determines the lower limit of MSA models. On the
other hand, noisy information hidden in each modality interferes the learning
of correct cross-modal dynamics. To address the above-mentioned problems, we
propose a novel MSA framework \textbf{M}odulation \textbf{M}odel for
\textbf{M}ultimodal \textbf{S}entiment \textbf{A}nalysis ({$ M^3SA $}) to
identify the contribution of modalities and reduce the impact of noisy
information, so as to better learn unimodal and cross-modal dynamics.
Specifically, modulation loss is designed to modulate the loss contribution
based on the confidence of individual modalities in each utterance, so as to
explore an optimal update solution for each unimodal network. Besides, contrary
to most existing works which fail to explicitly filter out noisy information,
we devise a modality filter module to identify and filter out modality noise
for the learning of correct cross-modal embedding. Extensive experiments on
publicly datasets demonstrate that our approach achieves state-of-the-art
performance.

    

### [[2111.08452] On minimizers and convolutional filters: a partial justification for the unreasonable effectiveness of CNNs in categorical sequence analysis](http://arxiv.org/abs/2111.08452)


  Minimizers and convolutional neural networks (CNNs) are two quite distinct
popular techniques that have both been employed to analyze biological
sequences. At face value, the methods seem entirely dissimilar. Minimizers use
min-wise hashing on a rolling window to extract a single important k-mer
feature per window. CNNs start with a wide array of randomly initialized
convolutional filters, paired with a pooling operation, and then multiple
additional neural layers to learn both the filters themselves and how those
filters can be used to classify the sequence. In this manuscript, I demonstrate
through a careful mathematical analysis of hash function properties that there
are deep theoretical connections between minimizers and convolutional filters
-- in short, for sequences over a categorical alphabet, random Gaussian
initialization of convolutional filters with max-pooling is equivalent to
choosing minimizers from a random hash function biased towards more distinct
k-mers. This provides a partial explanation for the unreasonable effectiveness
of CNNs in categorical sequence analysis.

    

### [[2111.08453] Entropy optimized semi-supervised decomposed vector-quantized variational autoencoder model based on transfer learning for multiclass text classification and generation](http://arxiv.org/abs/2111.08453)


  Semisupervised text classification has become a major focus of research over
the past few years. Hitherto, most of the research has been based on supervised
learning, but its main drawback is the unavailability of labeled data samples
in practical applications. It is still a key challenge to train the deep
generative models and learn comprehensive representations without supervision.
Even though continuous latent variables are employed primarily in deep latent
variable models, discrete latent variables, with their enhanced
understandability and better compressed representations, are effectively used
by researchers. In this paper, we propose a semisupervised discrete latent
variable model for multi-class text classification and text generation. The
proposed model employs the concept of transfer learning for training a
quantized transformer model, which is able to learn competently using fewer
labeled instances. The model applies decomposed vector quantization technique
to overcome problems like posterior collapse and index collapse. Shannon
entropy is used for the decomposed sub-encoders, on which a variable
DropConnect is applied, to retain maximum information. Moreover, gradients of
the Loss function are adaptively modified during backpropagation from decoder
to encoder to enhance the performance of the model. Three conventional datasets
of diversified range have been used for validating the proposed model on a
variable number of labeled instances. Experimental results indicate that the
proposed model has surpassed the state-of-the-art models remarkably.

    

### [[2111.08456] Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions](http://arxiv.org/abs/2111.08456)


  Multimodal regression is a fundamental task, which integrates the information
from different sources to improve the performance of follow-up applications.
However, existing methods mainly focus on improving the performance and often
ignore the confidence of prediction for diverse situations. In this study, we
are devoted to trustworthy multimodal regression which is critical in
cost-sensitive domains. To this end, we introduce a novel Mixture of
Normal-Inverse Gamma distributions (MoNIG) algorithm, which efficiently
estimates uncertainty in principle for adaptive integration of different
modalities and produces a trustworthy regression result. Our model can be
dynamically aware of uncertainty for each modality, and also robust for
corrupted modalities. Furthermore, the proposed MoNIG ensures explicitly
representation of (modality-specific/global) epistemic and aleatoric
uncertainties, respectively. Experimental results on both synthetic and
different real-world data demonstrate the effectiveness and trustworthiness of
our method on various multimodal regression tasks (e.g., temperature prediction
for superconductivity, relative location prediction for CT slices, and
multimodal sentiment analysis).

    

### [[2111.08457] A Novel TSK Fuzzy System Incorporating Multi-view Collaborative Transfer Learning for Personalized Epileptic EEG Detection](http://arxiv.org/abs/2111.08457)


  In clinical practice, electroencephalography (EEG) plays an important role in
the diagnosis of epilepsy. EEG-based computer-aided diagnosis of epilepsy can
greatly improve the ac-curacy of epilepsy detection while reducing the workload
of physicians. However, there are many challenges in practical applications for
personalized epileptic EEG detection (i.e., training of detection model for a
specific person), including the difficulty in extracting effective features
from one single view, the undesirable but common scenario of lacking sufficient
training data in practice, and the no guarantee of identically distributed
training and test data. To solve these problems, we propose a TSK fuzzy
system-based epilepsy detection algorithm that integrates multi-view
collaborative transfer learning. To address the challenge due to the limitation
of single-view features, multi-view learning ensures the diversity of features
by extracting them from different views. The lack of training data for building
a personalized detection model is tackled by leveraging the knowledge from the
source domain (reference scene) to enhance the performance of the target domain
(current scene of interest), where mismatch of data distributions between the
two domains is resolved with adaption technique based on maximum mean
discrepancy. Notably, the transfer learning and multi-view feature extraction
are performed at the same time. Furthermore, the fuzzy rules of the TSK fuzzy
system equip the model with strong fuzzy logic inference capability. Hence, the
proposed method has the potential to detect epileptic EEG signals effectively,
which is demonstrated with the positive results from a large number of
experiments on the CHB-MIT dataset.

    

### [[2111.08458] Lifelong Learning from Event-based Data](http://arxiv.org/abs/2111.08458)


  Lifelong learning is a long-standing aim for artificial agents that act in
dynamic environments, in which an agent needs to accumulate knowledge
incrementally without forgetting previously learned representations. We
investigate methods for learning from data produced by event cameras and
compare techniques to mitigate forgetting while learning incrementally. We
propose a model that is composed of both, feature extraction and continuous
learning. Furthermore, we introduce a habituation-based method to mitigate
forgetting. Our experimental results show that the combination of different
techniques can help to avoid catastrophic forgetting while learning
incrementally from the features provided by the extraction module.

    

### [[2111.08462] Towards Lightweight Controllable Audio Synthesis with Conditional Implicit Neural Representations](http://arxiv.org/abs/2111.08462)


  The high temporal resolution of audio and our perceptual sensitivity to small
irregularities in waveforms make synthesizing at high sampling rates a complex
and computationally intensive task, prohibiting real-time, controllable
synthesis within many approaches. In this work we aim to shed light on the
potential of Conditional Implicit Neural Representations (CINRs) as lightweight
backbones in generative frameworks for audio synthesis.
Implicit neural representations (INRs) are neural networks used to
approximate low-dimensional functions, trained to represent a single geometric
object by mapping input coordinates to structural information at input
locations. In contrast with other neural methods for representing geometric
objects, the memory required to parameterize the object is independent of
resolution, and only scales with its complexity. A corollary of this is that
INRs have infinite resolution, as they can be sampled at arbitrary resolutions.
To apply the concept of INRs in the generative domain we frame generative
modelling as learning a distribution of continuous functions. This can be
achieved by introducing conditioning methods to INRs.
Our experiments show that Periodic Conditional INRs (PCINRs) learn faster and
generally produce quantitatively better audio reconstructions than Transposed
Convolutional Neural Networks with equal parameter counts. However, their
performance is very sensitive to activation scaling hyperparameters. When
learning to represent more uniform sets, PCINRs tend to introduce artificial
high-frequency components in reconstructions. We validate this noise can be
minimized by applying standard weight regularization during training or
decreasing the compositional depth of PCINRs, and suggest directions for future
research.

    

### [[2111.08463] Multi-Centroid Hyperdimensional Computing Approach for Epileptic Seizure Detection](http://arxiv.org/abs/2111.08463)


  Long-term monitoring of patients with epilepsy presents a challenging problem
from the engineering perspective of real-time detection and wearable devices
design. It requires new solutions that allow continuous unobstructed monitoring
and reliable detection and prediction of seizures. A high variability in the
electroencephalogram (EEG) patterns exists among people, brain states, and time
instances during seizures, but also during non-seizure periods. This makes
epileptic seizure detection very challenging, especially if data is grouped
under only seizure and non-seizure labels.
Hyperdimensional (HD) computing, a novel machine learning approach, comes in
as a promising tool. However, it has certain limitations when the data shows a
high intra-class variability. Therefore, in this work, we propose a novel
semi-supervised learning approach based on a multi-centroid HD computing. The
multi-centroid approach allows to have several prototype vectors representing
seizure and non-seizure states, which leads to significantly improved
performance when compared to a simple 2-class HD model.
Further, real-life data imbalance poses an additional challenge and the
performance reported on balanced subsets of data is likely to be overestimated.
Thus, we test our multi-centroid approach with three different dataset
balancing scenarios, showing that performance improvement is higher for the
less balanced dataset. More specifically, up to 14% improvement is achieved on
an unbalanced test set with 10 times more non-seizure than seizure data. At the
same time, the total number of sub-classes is not significantly increased
compared to the balanced dataset. Thus, the proposed multi-centroid approach
can be an important element in achieving a high performance of epilepsy
detection with real-life data balance or during online learning, where seizures
are infrequent.

    

### [[2111.08466] Interpretable and Fair Boolean Rule Sets via Column Generation](http://arxiv.org/abs/2111.08466)


  This paper considers the learning of Boolean rules in either disjunctive
normal form (DNF, OR-of-ANDs, equivalent to decision rule sets) or conjunctive
normal form (CNF, AND-of-ORs) as an interpretable model for classification. An
integer program is formulated to optimally trade classification accuracy for
rule simplicity. We also consider the fairness setting and extend the
formulation to include explicit constraints on two different measures of
classification parity: equality of opportunity and equalized odds. Column
generation (CG) is used to efficiently search over an exponential number of
candidate clauses (conjunctions or disjunctions) without the need for heuristic
rule mining. This approach also bounds the gap between the selected rule set
and the best possible rule set on the training data. To handle large datasets,
we propose an approximate CG algorithm using randomization. Compared to three
recently proposed alternatives, the CG algorithm dominates the
accuracy-simplicity trade-off in 8 out of 16 datasets. When maximized for
accuracy, CG is competitive with rule learners designed for this purpose,
sometimes finding significantly simpler solutions that are no less accurate.
Compared to other fair and interpretable classifiers, our method is able to
find rule sets that meet stricter notions of fairness with a modest trade-off
in accuracy.

    

### [[2111.08472] An Energy Consumption Model for Electrical Vehicle Networks via Extended Federated-learning](http://arxiv.org/abs/2111.08472)


  Electrical vehicle (EV) raises to promote an eco-sustainable society.
Nevertheless, the ``range anxiety'' of EV hinders its wider acceptance among
customers. This paper proposes a novel solution to range anxiety based on a
federated-learning model, which is capable of estimating battery consumption
and providing energy-efficient route planning for vehicle networks.
Specifically, the new approach extends the federated-learning structure with
two components: anomaly detection and sharing policy. The first component
identifies preventing factors in model learning, while the second component
offers guidelines for information sharing amongst vehicle networks when the
sharing is necessary to preserve learning efficiency. The two components
collaborate to enhance learning robustness against data heterogeneities in
networks. Numerical experiments are conducted, and the results show that
compared with considered solutions, the proposed approach could provide higher
accuracy of battery-consumption estimation for vehicles under heterogeneous
data distributions, without increasing the time complexity or transmitting raw
data among vehicle networks.

    

### [[2111.08478] Spatial machine-learning model diagnostics: a model-agnostic distance-based approach](http://arxiv.org/abs/2111.08478)


  While significant progress has been made towards explaining black-box
machine-learning (ML) models, there is still a distinct lack of diagnostic
tools that elucidate the spatial behaviour of ML models in terms of predictive
skill and variable importance. This contribution proposes spatial prediction
error profiles (SPEPs) and spatial variable importance profiles (SVIPs) as
novel model-agnostic assessment and interpretation tools for spatial prediction
models with a focus on prediction distance. Their suitability is demonstrated
in two case studies representing a regionalization task in an
environmental-science context, and a classification task from remotely-sensed
land cover classification. In these case studies, the SPEPs and SVIPs of
geostatistical methods, linear models, random forest, and hybrid algorithms
show striking differences but also relevant similarities. Limitations of
related cross-validation techniques are outlined, and the case is made that
modelers should focus their model assessment and interpretation on the intended
spatial prediction horizon. The range of autocorrelation, in contrast, is not a
suitable criterion for defining spatial cross-validation test sets. The novel
diagnostic tools enrich the toolkit of spatial data science, and may improve ML
model interpretation, selection, and design.

    

### [[2111.08480] A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals](http://arxiv.org/abs/2111.08480)


  Cardiovascular diseases are the most common causes of death around the world.
To detect and treat heart-related diseases, continuous Blood Pressure (BP)
monitoring along with many other parameters are required. Several invasive and
non-invasive methods have been developed for this purpose. Most existing
methods used in the hospitals for continuous monitoring of BP are invasive. On
the contrary, cuff-based BP monitoring methods, which can predict Systolic
Blood Pressure (SBP) and Diastolic Blood Pressure (DBP), cannot be used for
continuous monitoring. Several studies attempted to predict BP from
non-invasively collectible signals such as Photoplethysmogram (PPG) and
Electrocardiogram (ECG), which can be used for continuous monitoring. In this
study, we explored the applicability of autoencoders in predicting BP from PPG
and ECG signals. The investigation was carried out on 12,000 instances of 942
patients of the MIMIC-II dataset and it was found that a very shallow,
one-dimensional autoencoder can extract the relevant features to predict the
SBP and DBP with the state-of-the-art performance on a very large dataset.
Independent test set from a portion of the MIMIC-II dataset provides an MAE of
2.333 and 0.713 for SBP and DBP, respectively. On an external dataset of forty
subjects, the model trained on the MIMIC-II dataset, provides an MAE of 2.728
and 1.166 for SBP and DBP, respectively. For both the cases, the results met
British Hypertension Society (BHS) Grade A and surpassed the studies from the
current literature.

    

### [[2111.08481] PySINDy: A comprehensive Python package for robust sparse system identification](http://arxiv.org/abs/2111.08481)


  Automated data-driven modeling, the process of directly discovering the
governing equations of a system from data, is increasingly being used across
the scientific community. PySINDy is a Python package that provides tools for
applying the sparse identification of nonlinear dynamics (SINDy) approach to
data-driven model discovery. In this major update to PySINDy, we implement
several advanced features that enable the discovery of more general
differential equations from noisy and limited data. The library of candidate
terms is extended for the identification of actuated systems, partial
differential equations (PDEs), and implicit differential equations. Robust
formulations, including the integral form of SINDy and ensembling techniques,
are also implemented to improve performance for real-world data. Finally, we
provide a range of new optimization algorithms, including several sparse
regression techniques and algorithms to enforce and promote inequality
constraints and stability. Together, these updates enable entirely new SINDy
model discovery capabilities that have not been reported in the literature,
such as constrained PDE identification and ensembling with different sparse
regression optimizers.

    

### [[2111.08489] Generative Pre-Trained Transformer for Design Concept Generation: An Exploration](http://arxiv.org/abs/2111.08489)


  Novel concepts are essential for design innovation and can be generated with
the aid of data stimuli and computers. However, current generative design
algorithms focus on diagrammatic or spatial concepts that are either too
abstract to understand or too detailed for early phase design exploration. This
paper explores the uses of generative pre-trained transformers (GPT) for
natural language design concept generation. Our experiments involve the use of
GPT-2 and GPT-3 for different creative reasonings in design tasks. Both show
reasonably good performance for verbal design concept generation.

    

### [[2111.08493] An adaptive dimension reduction algorithm for latent variables of variational autoencoder](http://arxiv.org/abs/2111.08493)


  Constructed by the neural network, variational autoencoder has the
overfitting problem caused by setting too many neural units, we develop an
adaptive dimension reduction algorithm that can automatically learn the
dimension of latent variable vector, moreover, the dimension of every hidden
layer. This approach not only apply to the variational autoencoder but also
other variants like Conditional VAE(CVAE), and we show the empirical results on
six data sets which presents the universality and efficiency of this algorithm.
The key advantages of this algorithm is that it can converge the dimension of
latent variable vector which approximates the dimension reaches minimum loss of
variational autoencoder(VAE), also speeds up the generating and computing speed
by reducing the neural units.

    

### [[2111.08498] Reducing the Long Tail Losses in Scientific Emulations with Active Learning](http://arxiv.org/abs/2111.08498)


  Deep-learning-based models are increasingly used to emulate scientific
simulations to accelerate scientific research. However, accurate, supervised
deep learning models require huge amount of labelled data, and that often
becomes the bottleneck in employing neural networks. In this work, we leveraged
an active learning approach called core-set selection to actively select data,
per a pre-defined budget, to be labelled for training. To further improve the
model performance and reduce the training costs, we also warm started the
training using a shrink-and-perturb trick. We tested on two case studies in
different fields, namely galaxy halo occupation distribution modelling in
astrophysics and x-ray emission spectroscopy in plasma physics, and the results
are promising: we achieved competitive overall performance compared to using a
random sampling baseline, and more importantly, successfully reduced the larger
absolute losses, i.e. the long tail in the loss distribution, at virtually no
overhead costs.

    

### [[2111.08502] Human-error-potential Estimation based on Wearable Biometric Sensors](http://arxiv.org/abs/2111.08502)


  This study tackles on a new problem of estimating human-error potential on a
shop floor on the basis of wearable sensors. Unlike existing studies that
utilize biometric sensing technology to estimate people's internal state such
as fatigue and mental stress, we attempt to estimate the human-error potential
in a situation where a target person does not stay calm, which is much more
difficult as sensor noise significantly increases. We propose a novel
formulation, in which the human-error-potential estimation problem is reduced
to a classification problem, and introduce a new method that can be used for
solving the classification problem even with noisy sensing data. The key ideas
are to model the process of calculating biometric indices probabilistically so
that the prior knowledge on the biometric indices can be integrated, and to
utilize the features that represent the movement of target persons in
combination with biometric features. The experimental analysis showed that our
method effectively estimates the human-error potential.

    

### [[2111.08507] Machine Learning for Genomic Data](http://arxiv.org/abs/2111.08507)


  This report explores the application of machine learning techniques on short
timeseries gene expression data. Although standard machine learning algorithms
work well on longer time-series', they often fail to find meaningful insights
from fewer timepoints. In this report, we explore model-based clustering
techniques. We combine popular unsupervised learning techniques like K-Means,
Gaussian Mixture Models, Bayesian Networks, Hidden Markov Models with the
well-known Expectation Maximization algorithm. K-Means and Gaussian Mixture
Models are fairly standard, while Hidden Markov Model and Bayesian Networks
clustering are more novel ideas that suit time-series gene expression data.

    

### [[2111.08510] CVSS-BERT: Explainable Natural Language Processing to Determine the Severity of a Computer Security Vulnerability from its Description](http://arxiv.org/abs/2111.08510)


  When a new computer security vulnerability is publicly disclosed, only a
textual description of it is available. Cybersecurity experts later provide an
analysis of the severity of the vulnerability using the Common Vulnerability
Scoring System (CVSS). Specifically, the different characteristics of the
vulnerability are summarized into a vector (consisting of a set of metrics),
from which a severity score is computed. However, because of the high number of
vulnerabilities disclosed everyday this process requires lot of manpower, and
several days may pass before a vulnerability is analyzed. We propose to
leverage recent advances in the field of Natural Language Processing (NLP) to
determine the CVSS vector and the associated severity score of a vulnerability
from its textual description in an explainable manner. To this purpose, we
trained multiple BERT classifiers, one for each metric composing the CVSS
vector. Experimental results show that our trained classifiers are able to
determine the value of the metrics of the CVSS vector with high accuracy. The
severity score computed from the predicted CVSS vector is also very close to
the real severity score attributed by a human expert. For explainability
purpose, gradient-based input saliency method was used to determine the most
relevant input words for a given prediction made by our classifiers. Often, the
top relevant words include terms in agreement with the rationales of a human
cybersecurity expert, making the explanation comprehensible for end-users.

    

### [[2111.08513] Comparing Cross Correlation-Based Similarities](http://arxiv.org/abs/2111.08513)


  The common product between two multisets or functions can be understood as
being analogue to the inner product in real vector or function spaces in spite
of its non-linear nature. In addition to other interesting features, it also
allows respective correlations to be derived which, in addition to their
conceptual and computational simplicity, have been verified to be able to
provide enhanced results in tasks such as template matching. The multiset-based
correlations based on the real-valued multiset Jaccard and coincidence indices
are compared in this work, with encouraging results which have immediate
implications not only in pattern recognition and deep learning, but also in
scientific modeling in general. As expected, the multiset correlation methods,
and especially the coincidence index, presented remarkable performance
characterized by sharper and narrower peaks while secondary peaks were
attenuated, even in presence of noise. In particular, the two methods derived
from the coincidence index led to the sharpest and narrowest peaks, as well as
intense attenuation of the secondary peaks. The cross correlation, however,
presented the best robustness to symmetric additive noise, which suggested a
combination of the considered approaches. After a preliminary investigation of
the performance of the multiset approaches, as well as the classic
cross-correlation, a systematic comparison framework is proposed and applied
for the study of the aforementioned methods. Several interesting results are
reported, including the confirmation, at least for the considered type of data,
of the coincidence correlation as providing enhanced performance regarding
detection of narrow, sharp peaks while secondary matches are duly attenuated.
The combined method also confirmed its good performance for signals in presence
of intense additive noise.

    

### [[2111.08514] Multiset Signal Processing and Electronics](http://arxiv.org/abs/2111.08514)


  Multisets are an intuitive extension of the traditional concept of sets that
allow repetition of elements, with the number of times each element appears
being understood as the respective multiplicity. Recent generalizations of
multisets to real-valued functions, accounting for possibly negative values,
have paved the way to a number of interesting implications and applications,
including respective implementations as electronic systems. The basic multiset
operations include the set complementation (sign change), intersection (minimum
between two values), union (maximum between two values), difference and sum
(identical to the algebraic counterparts). When applied to functions or
signals, the sign and conjoint sign functions are also required. Given that
signals are functions, it becomes possible to effectively translate the
multiset and multifunction operations to analog electronics, which is the
objective of the present work. It is proposed that effective multiset
operations capable of high performance self and cross-correlation can be
obtained with relative simplicity in either discrete or integrated circuits.
The problem of switching noise is also briefly discussed. The present results
have great potential for applications and related developments in analog and
digital electronics, as well as for pattern recognition, signal processing, and
deep learning.

    

### [[2111.08516] Common Product Neurons](http://arxiv.org/abs/2111.08516)


  The present work develops a comparative performance of artificial neurons
obtained in terms of the recently introduced real-valued Jaccard and
coincidence indices and respective functionals. The interiority index and
classic cross-correlation are also included in our study. After presenting the
basic concepts related to multisets and the adopted similarity metrics,
including new results about the generalization of the family of real-valued
Jaccard and conicidence indices to higher orders, we proceed to studying the
response of a single neuron, not taking into account the output non-linearity
(e.g.~sigmoid), respectively to the detection of a gaussian stimulus in
presence of displacement, magnification, intensity variation, noise and
interference from additional patterns. It is shown that the real-valued Jaccard
and coincidence approaches are substantially more robust and effective than the
interiority index and the classic cross-correlation. The coincidence based
neurons are shown to have the best overall performance for the considered type
of data and perturbations. The reported concepts, methods, and results, have
substantial implications not only for patter recognition and deep learning, but
also regarding neurobiology and neuroscience.

    

### [[2111.08524] Non-separable Spatio-temporal Graph Kernels via SPDEs](http://arxiv.org/abs/2111.08524)


  Gaussian processes (GPs) provide a principled and direct approach for
inference and learning on graphs. However, the lack of justified graph kernels
for spatio-temporal modelling has held back their use in graph problems. We
leverage an explicit link between stochastic partial differential equations
(SPDEs) and GPs on graphs, and derive non-separable spatio-temporal graph
kernels that capture interaction across space and time. We formulate the graph
kernels for the stochastic heat equation and wave equation. We show that by
providing novel tools for spatio-temporal GP modelling on graphs, we outperform
pre-existing graph kernels in real-world applications that feature diffusion,
oscillation, and other complicated interactions.

    

### [[2111.08535] Sequential Community Mode Estimation](http://arxiv.org/abs/2111.08535)


  We consider a population, partitioned into a set of communities, and study
the problem of identifying the largest community within the population via
sequential, random sampling of individuals. There are multiple sampling
domains, referred to as \emph{boxes}, which also partition the population. Each
box may consist of individuals of different communities, and each community may
in turn be spread across multiple boxes. The learning agent can, at any time,
sample (with replacement) a random individual from any chosen box; when this is
done, the agent learns the community the sampled individual belongs to, and
also whether or not this individual has been sampled before. The goal of the
agent is to minimize the probability of mis-identifying the largest community
in a \emph{fixed budget} setting, by optimizing both the sampling strategy as
well as the decision rule. We propose and analyse novel algorithms for this
problem, and also establish information theoretic lower bounds on the
probability of error under any algorithm. In several cases of interest, the
exponential decay rates of the probability of error under our algorithms are
shown to be optimal up to constant factors. The proposed algorithms are further
validated via simulations on real-world datasets.

    

### [[2111.08536] HiRID-ICU-Benchmark -- A Comprehensive Machine Learning Benchmark on High-resolution ICU Data](http://arxiv.org/abs/2111.08536)


  The recent success of machine learning methods applied to time series
collected from Intensive Care Units (ICU) exposes the lack of standardized
machine learning benchmarks for developing and comparing such methods. While
raw datasets, such as MIMIC-IV or eICU, can be freely accessed on Physionet,
the choice of tasks and pre-processing is often chosen ad-hoc for each
publication, limiting comparability across publications. In this work, we aim
to improve this situation by providing a benchmark covering a large spectrum of
ICU-related tasks. Using the HiRID dataset, we define multiple clinically
relevant tasks developed in collaboration with clinicians. In addition, we
provide a reproducible end-to-end pipeline to construct both data and labels.
Finally, we provide an in-depth analysis of current state-of-the-art sequence
modeling methods, highlighting some limitations of deep learning approaches for
this type of data. With this benchmark, we hope to give the research community
the possibility of a fair comparison of their work.

    

### [[2111.08538] Utilizing Textual Reviews in Latent Factor Models for Recommender Systems](http://arxiv.org/abs/2111.08538)


  Most of the existing recommender systems are based only on the rating data,
and they ignore other sources of information that might increase the quality of
recommendations, such as textual reviews, or user and item characteristics.
Moreover, the majority of those systems are applicable only on small datasets
(with thousands of observations) and are unable to handle large datasets (with
millions of observations). We propose a recommender algorithm that combines a
rating modelling technique (i.e., Latent Factor Model) with a topic modelling
method based on textual reviews (i.e., Latent Dirichlet Allocation), and we
extend the algorithm such that it allows adding extra user- and item-specific
information to the system. We evaluate the performance of the algorithm using
this http URL datasets with different sizes, corresponding to 23 product
categories. After comparing the built model to four other models we found that
combining textual reviews with ratings leads to better recommendations.
Moreover, we found that adding extra user and item features to the model
increases its prediction accuracy, which is especially true for medium and
large datasets.

    

### [[2111.08543] WikiContradiction: Detecting Self-Contradiction Articles on Wikipedia](http://arxiv.org/abs/2111.08543)


  While Wikipedia has been utilized for fact-checking and claim verification to
debunk misinformation and disinformation, it is essential to either improve
article quality and rule out noisy articles. Self-contradiction is one of the
low-quality article types in Wikipedia. In this work, we propose a task of
detecting self-contradiction articles in Wikipedia. Based on the
"self-contradictory" template, we create a novel dataset for the
self-contradiction detection task. Conventional contradiction detection focuses
on comparing pairs of sentences or claims, but self-contradiction detection
needs to further reason the semantics of an article and simultaneously learn
the contradiction-aware comparison from all pairs of sentences. Therefore, we
present the first model, Pairwise Contradiction Neural Network (PCNN), to not
only effectively identify self-contradiction articles, but also highlight the
most contradiction pairs of contradiction sentences. The main idea of PCNN is
two-fold. First, to mitigate the effect of data scarcity on self-contradiction
articles, we pre-train the module of pairwise contradiction learning using SNLI
and MNLI benchmarks. Second, we select top-K sentence pairs with the highest
contradiction probability values and model their correlation to determine
whether the corresponding article belongs to self-contradiction. Experiments
conducted on the proposed WikiContradiction dataset exhibit that PCNN can
generate promising performance and comprehensively highlight the sentence pairs
the contradiction locates.

    

### [[2111.08546] Interpreting Language Models Through Knowledge Graph Extraction](http://arxiv.org/abs/2111.08546)


  Transformer-based language models trained on large text corpora have enjoyed
immense popularity in the natural language processing community and are
commonly used as a starting point for downstream tasks. While these models are
undeniably useful, it is a challenge to quantify their performance beyond
traditional accuracy metrics. In this paper, we compare BERT-based language
models through snapshots of acquired knowledge at sequential stages of the
training process. Structured relationships from training corpora may be
uncovered through querying a masked language model with probing tasks. We
present a methodology to unveil a knowledge acquisition timeline by generating
knowledge graph extracts from cloze "fill-in-the-blank" statements at various
stages of RoBERTa's early training. We extend this analysis to a comparison of
pretrained variations of BERT models (DistilBERT, BERT-base, RoBERTa). This
work proposes a quantitative framework to compare language models through
knowledge graph extraction (GED, Graph2Vec) and showcases a part-of-speech
analysis (POSOR) to identify the linguistic strengths of each model variant.
Using these metrics, machine learning practitioners can compare models,
diagnose their models' behavioral strengths and weaknesses, and identify new
targeted datasets to improve model performance.

    

### [[2111.08550] On Effective Scheduling of Model-based Reinforcement Learning](http://arxiv.org/abs/2111.08550)


  Model-based reinforcement learning has attracted wide attention due to its
superior sample efficiency. Despite its impressive success so far, it is still
unclear how to appropriately schedule the important hyperparameters to achieve
adequate performance, such as the real data ratio for policy optimization in
Dyna-style model-based algorithms. In this paper, we first theoretically
analyze the role of real data in policy training, which suggests that gradually
increasing the ratio of real data yields better performance. Inspired by the
analysis, we propose a framework named AutoMBPO to automatically schedule the
real data ratio as well as other hyperparameters in training model-based policy
optimization (MBPO) algorithm, a representative running case of model-based
methods. On several continuous control tasks, the MBPO instance trained with
hyperparameters scheduled by AutoMBPO can significantly surpass the original
one, and the real data ratio schedule found by AutoMBPO shows consistency with
our theoretical analysis.

    

### [[2111.08563] Rank-Regret Minimization](http://arxiv.org/abs/2111.08563)


  Multi-criteria decision-making often requires finding a small representative
subset from the database. A recently proposed method is the regret minimization
set (RMS) query. RMS returns a fixed size subset S of dataset D that minimizes
the regret ratio of S (the difference between the score of top1 in S and the
score of top-1 in D, for any possible utility function). Existing work showed
that the regret-ratio is not able to accurately quantify the regret level of a
user. Further, relative to the regret-ratio, users do understand the notion of
rank. Consequently, it considered the problem of finding a minimal set S with
at most k rank-regret (the minimal rank of tuples of S in the sorted list of
D).
Corresponding to RMS, we focus on the dual version of the above problem,
defined as the rank-regret minimization (RRM) problem, which seeks to find a
fixed size set S that minimizes the maximum rank-regret for all possible
utility functions. Further, we generalize RRM and propose the restricted
rank-regret minimization (RRRM) problem to minimize the rank-regret of S for
functions in a restricted space. The solution for RRRM usually has a lower
regret level and can better serve the specific preferences of some users. In 2D
space, we design a dynamic programming algorithm 2DRRM to find the optimal
solution for RRM. In HD space, we propose an algorithm HDRRM for RRM that
bounds the output size and introduces a double approximation guarantee for
rank-regret. Both 2DRRM and HDRRM can be generalized to the RRRM problem.
Extensive experiments are performed on the synthetic and real datasets to
verify the efficiency and effectiveness of our algorithms.

    

### [[2111.08565] Polymatrix Competitive Gradient Descent](http://arxiv.org/abs/2111.08565)


  Many economic games and machine learning approaches can be cast as
competitive optimization problems where multiple agents are minimizing their
respective objective function, which depends on all agents' actions. While
gradient descent is a reliable basic workhorse for single-agent optimization,
it often leads to oscillation in competitive optimization. In this work we
propose polymatrix competitive gradient descent (PCGD) as a method for solving
general sum competitive optimization involving arbitrary numbers of agents. The
updates of our method are obtained as the Nash equilibria of a local polymatrix
approximation with a quadratic regularization, and can be computed efficiently
by solving a linear system of equations. We prove local convergence of PCGD to
stable fixed points for $n$-player general-sum games, and show that it does not
require adapting the step size to the strength of the player-interactions. We
use PCGD to optimize policies in multi-agent reinforcement learning and
demonstrate its advantages in Snake, Markov soccer and an electricity market
game. Agents trained by PCGD outperform agents trained with simultaneous
gradient descent, symplectic gradient adjustment, and extragradient in Snake
and Markov soccer games and on the electricity market game, PCGD trains faster
than both simultaneous gradient descent and the extragradient method.

    

### [[2111.08566] SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](http://arxiv.org/abs/2111.08566)


  The in-memory algorithms for approximate nearest neighbor search (ANNS) have
achieved great success for fast high-recall search, but are extremely expensive
when handling very large scale database. Thus, there is an increasing request
for the hybrid ANNS solutions with small memory and inexpensive solid-state
drive (SSD). In this paper, we present a simple but efficient memory-disk
hybrid indexing and search system, named SPANN, that follows the inverted index
methodology. It stores the centroid points of the posting lists in the memory
and the large posting lists in the disk. We guarantee both disk-access
efficiency (low latency) and high recall by effectively reducing the
disk-access number and retrieving high-quality posting lists. In the
index-building stage, we adopt a hierarchical balanced clustering algorithm to
balance the length of posting lists and augment the posting list by adding the
points in the closure of the corresponding clusters. In the search stage, we
use a query-aware scheme to dynamically prune the access of unnecessary posting
lists. Experiment results demonstrate that SPANN is 2$\times$ faster than the
state-of-the-art ANNS solution DiskANN to reach the same recall quality $90\%$
with same memory cost in three billion-scale datasets. It can reach $90\%$
recall@1 and recall@10 in just around one millisecond with only 32GB memory
cost. Code is available at:
{\footnotesize\color{blue}{\url{this https URL}}}.

    

### [[2111.08568] Robust recovery for stochastic block models](http://arxiv.org/abs/2111.08568)


  We develop an efficient algorithm for weak recovery in a robust version of
the stochastic block model. The algorithm matches the statistical guarantees of
the best known algorithms for the vanilla version of the stochastic block
model. In this sense, our results show that there is no price of robustness in
the stochastic block model. Our work is heavily inspired by recent work of
Banks, Mohanty, and Raghavendra (SODA 2021) that provided an efficient
algorithm for the corresponding distinguishing problem. Our algorithm and its
analysis significantly depart from previous ones for robust recovery. A key
challenge is the peculiar optimization landscape underlying our algorithm: The
planted partition may be far from optimal in the sense that completely
unrelated solutions could achieve the same objective value. This phenomenon is
related to the push-out effect at the BBP phase transition for PCA. To the best
of our knowledge, our algorithm is the first to achieve robust recovery in the
presence of such a push-out effect in a non-asymptotic setting. Our algorithm
is an instantiation of a framework based on convex optimization (related to but
distinct from sum-of-squares), which may be useful for other robust matrix
estimation problems. A by-product of our analysis is a general technique that
boosts the probability of success (over the randomness of the input) of an
arbitrary robust weak-recovery algorithm from constant (or slowly vanishing)
probability to exponentially high probability.

    

### [[2111.08577] Neuron-based Pruning of Deep Neural Networks with Better Generalization using Kronecker Factored Curvature Approximation](http://arxiv.org/abs/2111.08577)


  Existing methods of pruning deep neural networks focus on removing
unnecessary parameters of the trained network and fine tuning the model
afterwards to find a good solution that recovers the initial performance of the
trained model. Unlike other works, our method pays special attention to the
quality of the solution in the compressed model and inference computation time
by pruning neurons. The proposed algorithm directs the parameters of the
compressed model toward a flatter solution by exploring the spectral radius of
Hessian which results in better generalization on unseen data. Moreover, the
method does not work with a pre-trained network and performs training and
pruning simultaneously. Our result shows that it improves the state-of-the-art
results on neuron compression. The method is able to achieve very small
networks with small accuracy degradation across different neural network
models.

    

### [[2111.08585] CEHR-BERT: Incorporating temporal information from structured EHR data to improve prediction tasks](http://arxiv.org/abs/2111.08585)


  Embedding algorithms are increasingly used to represent clinical concepts in
healthcare for improving machine learning tasks such as clinical phenotyping
and disease prediction. Recent studies have adapted state-of-the-art
bidirectional encoder representations from transformers (BERT) architecture to
structured electronic health records (EHR) data for the generation of
contextualized concept embeddings, yet do not fully incorporate temporal data
across multiple clinical domains. Therefore we developed a new BERT adaptation,
CEHR-BERT, to incorporate temporal information using a hybrid approach by
augmenting the input to BERT using artificial time tokens, incorporating time,
age, and concept embeddings, and introducing a new second learning objective
for visit type. CEHR-BERT was trained on a subset of Columbia University Irving
Medical Center-York Presbyterian Hospital's clinical data, which includes 2.4M
patients, spanning over three decades, and tested using 4-fold cross-validation
on the following prediction tasks: hospitalization, death, new heart failure
(HF) diagnosis, and HF readmission. Our experiments show that CEHR-BERT
outperformed existing state-of-the-art clinical BERT adaptations and baseline
models across all 4 prediction tasks in both ROC-AUC and PR-AUC. CEHR-BERT also
demonstrated strong transfer learning capability, as our model trained on only
5% of data outperformed comparison models trained on the entire data set.
Ablation studies to better understand the contribution of each time component
showed incremental gains with every element, suggesting that CEHR-BERT's
incorporation of artificial time tokens, time and age embeddings with concept
embeddings, and the addition of the second learning objective represents a
promising approach for future BERT-based clinical embeddings.

    

### [[2111.08591] Robustness of Bayesian Neural Networks to White-Box Adversarial Attacks](http://arxiv.org/abs/2111.08591)


  Bayesian Neural Networks (BNNs), unlike Traditional Neural Networks (TNNs)
are robust and adept at handling adversarial attacks by incorporating
randomness. This randomness improves the estimation of uncertainty, a feature
lacking in TNNs. Thus, we investigate the robustness of BNNs to white-box
attacks using multiple Bayesian neural architectures. Furthermore, we create
our BNN model, called BNN-DenseNet, by fusing Bayesian inference (i.e.,
variational Bayes) to the DenseNet architecture, and BDAV, by combining this
intervention with adversarial training. Experiments are conducted on the
CIFAR-10 and FGVC-Aircraft datasets. We attack our models with strong white-box
attacks ($l_\infty$-FGSM, $l_\infty$-PGD, $l_2$-PGD, EOT $l_\infty$-FGSM, and
EOT $l_\infty$-PGD). In all experiments, at least one BNN outperforms
traditional neural networks during adversarial attack scenarios. An
adversarially-trained BNN outperforms its non-Bayesian, adversarially-trained
counterpart in most experiments, and often by significant margins. Lastly, we
investigate network calibration and find that BNNs do not make overconfident
predictions, providing evidence that BNNs are also better at measuring
uncertainty.

    

### [[2111.08596] Reinforcement Learning with Feedback from Multiple Humans with Diverse Skills](http://arxiv.org/abs/2111.08596)


  A promising approach to improve the robustness and exploration in
Reinforcement Learning is collecting human feedback and that way incorporating
prior knowledge of the target environment. It is, however, often too expensive
to obtain enough feedback of good quality. To mitigate the issue, we aim to
rely on a group of multiple experts (and non-experts) with different skill
levels to generate enough feedback. Such feedback can therefore be inconsistent
and infrequent. In this paper, we build upon prior work -- Advise, a Bayesian
approach attempting to maximise the information gained from human feedback --
extending the algorithm to accept feedback from this larger group of humans,
the trainers, while also estimating each trainer's reliability. We show how
aggregating feedback from multiple trainers improves the total feedback's
accuracy and make the collection process easier in two ways. Firstly, this
approach addresses the case of some of the trainers being adversarial.
Secondly, having access to the information about each trainer reliability
provides a second layer of robustness and offers valuable information for
people managing the whole system to improve the overall trust in the system. It
offers an actionable tool for improving the feedback collection process or
modifying the reward function design if needed. We empirically show that our
approach can accurately learn the reliability of each trainer correctly and use
it to maximise the information gained from the multiple trainers' feedback,
even if some of the sources are adversarial.

    

### [[2111.08611] Stochastic Extragradient: General Analysis and Improved Rates](http://arxiv.org/abs/2111.08611)


  The Stochastic Extragradient (SEG) method is one of the most popular
algorithms for solving min-max optimization and variational inequalities
problems (VIP) appearing in various machine learning tasks. However, several
important questions regarding the convergence properties of SEG are still open,
including the sampling of stochastic gradients, mini-batching, convergence
guarantees for the monotone finite-sum variational inequalities with possibly
non-monotone terms, and others. To address these questions, in this paper, we
develop a novel theoretical framework that allows us to analyze several
variants of SEG in a unified manner. Besides standard setups, like Same-Sample
SEG under Lipschitzness and monotonicity or Independent-Samples SEG under
uniformly bounded variance, our approach allows us to analyze variants of SEG
that were never explicitly considered in the literature before. Notably, we
analyze SEG with arbitrary sampling which includes importance sampling and
various mini-batching strategies as special cases. Our rates for the new
variants of SEG outperform the current state-of-the-art convergence guarantees
and rely on less restrictive assumptions.

    

### [[2111.08617] Project CGX: Scalable Deep Learning on Commodity GPUs](http://arxiv.org/abs/2111.08617)


  The ability to scale out training workloads has been one of the key
performance enablers of deep learning. The main scaling approach is
data-parallel GPU-based training, which has been boosted by hardware and
software support for highly efficient inter-GPU communication, in particular
via bandwidth overprovisioning. This support comes at a price: there is an
order of magnitude cost difference between "cloud-grade" servers with such
support, relative to their "consumer-grade" counterparts, although server-grade
and consumer-grade GPUs can have similar computational envelopes. In this
paper, we investigate whether the expensive hardware overprovisioning approach
can be supplanted via algorithmic and system design, and propose a framework
called CGX, which provides efficient software support for communication
compression. We show that this framework is able to remove communication
bottlenecks from consumer-grade multi-GPU systems, in the absence of hardware
support: when training modern models and tasks to full accuracy, our framework
enables self-speedups of 2-3X on a commodity system using 8 consumer-grade
NVIDIA RTX 3090 GPUs, and enables it to surpass the throughput of an NVIDIA
DGX-1 server, which has similar peak FLOPS but benefits from bandwidth
overprovisioning.

    

### [[2111.08626] Adjoint-Matching Neural Network Surrogates for Fast 4D-Var Data Assimilation](http://arxiv.org/abs/2111.08626)


  The data assimilation procedures used in many operational numerical weather
forecasting systems are based around variants of the 4D-Var algorithm. The cost
of solving the 4D-Var problem is dominated by the cost of forward and adjoint
evaluations of the physical model. This motivates their substitution by fast,
approximate surrogate models. Neural networks offer a promising approach for
the data-driven creation of surrogate models. The accuracy of the surrogate
4D-Var problem's solution has been shown to depend explicitly on accurate
modeling of the forward and adjoint for other surrogate modeling approaches and
in the general nonlinear setting. We formulate and analyze several approaches
to incorporating derivative information into the construction of neural network
surrogates. The resulting networks are tested on out of training set data and
in a sequential data assimilation setting on the Lorenz-63 system. Two methods
demonstrate superior performance when compared with a surrogate network trained
without adjoint information, showing the benefit of incorporating adjoint
information into the training process.

    

### [[2111.08634] NVIDIA NeMo Neural Machine Translation Systems for English-German and English-Russian News and Biomedical Tasks at WMT21](http://arxiv.org/abs/2111.08634)


  This paper provides an overview of NVIDIA NeMo's neural machine translation
systems for the constrained data track of the WMT21 News and Biomedical Shared
Translation Tasks. Our news task submissions for English-German (En-De) and
English-Russian (En-Ru) are built on top of a baseline transformer-based
sequence-to-sequence model. Specifically, we use a combination of 1) checkpoint
averaging 2) model scaling 3) data augmentation with backtranslation and
knowledge distillation from right-to-left factorized models 4) finetuning on
test sets from previous years 5) model ensembling 6) shallow fusion decoding
with transformer language models and 7) noisy channel re-ranking. Additionally,
our biomedical task submission for English-Russian uses a biomedically biased
vocabulary and is trained from scratch on news task data, medically relevant
text curated from the news task dataset, and biomedical data provided by the
shared task. Our news system achieves a sacreBLEU score of 39.5 on the WMT'20
En-De test set outperforming the best submission from last year's task of 38.8.
Our biomedical task Ru-En and En-Ru systems reach BLEU scores of 43.8 and 40.3
respectively on the WMT'20 Biomedical Task Test set, outperforming the previous
year's best submissions.

    

### [[2111.08635] Single-channel speech separation using Soft-minimum Permutation Invariant Training](http://arxiv.org/abs/2111.08635)


  The goal of speech separation is to extract multiple speech sources from a
single microphone recording. Recently, with the advancement of deep learning
and availability of large datasets, speech separation has been formulated as a
supervised learning problem. These approaches aim to learn discriminative
patterns of speech, speakers, and background noise using a supervised learning
algorithm, typically a deep neural network. A long-lasting problem in
supervised speech separation is finding the correct label for each separated
speech signal, referred to as label permutation ambiguity. Permutation
ambiguity refers to the problem of determining the output-label assignment
between the separated sources and the available single-speaker speech labels.
Finding the best output-label assignment is required for calculation of
separation error, which is later used for updating parameters of the model.
Recently, Permutation Invariant Training (PIT) has been shown to be a promising
solution in handling the label ambiguity problem. However, the overconfident
choice of the output-label assignment by PIT results in a sub-optimal trained
model. In this work, we propose a probabilistic optimization framework to
address the inefficiency of PIT in finding the best output-label assignment.
Our proposed method entitled trainable Soft-minimum PIT is then employed on the
same Long-Short Term Memory (LSTM) architecture used in Permutation Invariant
Training (PIT) speech separation method. The results of our experiments show
that the proposed method outperforms conventional PIT speech separation
significantly (p-value $ < 0.01$) by +1dB in Signal to Distortion Ratio (SDR)
and +1.5dB in Signal to Interference Ratio (SIR).

    

### [[2111.08644] UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection](http://arxiv.org/abs/2111.08644)


  Detecting abnormal events in video is commonly framed as a one-class
classification task, where training videos contain only normal events, while
test videos encompass both normal and abnormal events. In this scenario,
anomaly detection is an open-set problem. However, some studies assimilate
anomaly detection to action recognition. This is a closed-set scenario that
fails to test the capability of systems at detecting new anomaly types. To this
end, we propose UBnormal, a new supervised open-set benchmark composed of
multiple virtual scenes for video anomaly detection. Unlike existing data sets,
we introduce abnormal events annotated at the pixel level at training time, for
the first time enabling the use of fully-supervised learning methods for
abnormal event detection. To preserve the typical open-set formulation, we make
sure to include disjoint sets of anomaly types in our training and test
collections of videos. To our knowledge, UBnormal is the first video anomaly
detection benchmark to allow a fair head-to-head comparison between one-class
open-set models and supervised closed-set models, as shown in our experiments.
Moreover, we provide empirical evidence showing that UBnormal can enhance the
performance of a state-of-the-art anomaly detection framework on two prominent
data sets, Avenue and ShanghaiTech.

    

### [[2111.08645] Machine Learning-Assisted Analysis of Small Angle X-ray Scattering](http://arxiv.org/abs/2111.08645)


  Small angle X-ray scattering (SAXS) is extensively used in materials science
as a way of examining nanostructures. The analysis of experimental SAXS data
involves mapping a rather simple data format to a vast amount of structural
models. Despite various scientific computing tools to assist the model
selection, the activity heavily relies on the SAXS analysts' experience, which
is recognized as an efficiency bottleneck by the community. To cope with this
decision-making problem, we develop and evaluate the open-source, Machine
Learning-based tool SCAN (SCattering Ai aNalysis) to provide recommendations on
model selection. SCAN exploits multiple machine learning algorithms and uses
models and a simulation tool implemented in the SasView package for generating
a well defined set of datasets. Our evaluation shows that SCAN delivers an
overall accuracy of 95%-97%. The XGBoost Classifier has been identified as the
most accurate method with a good balance between accuracy and training time.
With eleven predefined structural models for common nanostructures and an easy
draw-drop function to expand the number and types training models, SCAN can
accelerate the SAXS data analysis workflow.

    

### [[2111.08647] DataCLUE: A Benchmark Suite for Data-centric NLP](http://arxiv.org/abs/2111.08647)


  Data-centric AI has recently proven to be more effective and
high-performance, while traditional model-centric AI delivers fewer and fewer
benefits. It emphasizes improving the quality of datasets to achieve better
model performance. This field has significant potential because of its great
practicability and getting more and more attention. However, we have not seen
significant research progress in this field, especially in NLP. We propose
DataCLUE, which is the first Data-Centric benchmark applied in NLP field. We
also provide three simple but effective baselines to foster research in this
field (improve Macro-F1 up to 5.7% point). In addition, we conduct
comprehensive experiments with human annotators and show the hardness of
DataCLUE. We also try an advanced method: the forgetting informed bootstrapping
label correction method. All the resources related to DataCLUE, including
dataset, toolkit, leaderboard, and baselines, is available online at
this https URL


### [[2111.08649] FedCostWAvg: A new averaging for better Federated Learning](http://arxiv.org/abs/2111.08649)


  We propose a simple new aggregation strategy for federated learning that won
the MICCAI Federated Tumor Segmentation Challenge 2021 (FETS), the first ever
challenge on Federated Learning in the Machine Learning community. Our method
addresses the problem of how to aggregate multiple models that were trained on
different data sets. Conceptually, we propose a new way to choose the weights
when averaging the different models, thereby extending the current state of the
art (FedAvg). Empirical validation demonstrates that our approach reaches a
notable improvement in segmentation performance compared to FedAvg.

    

### [[2111.08656] Causal Effect Variational Autoencoder with Uniform Treatment](http://arxiv.org/abs/2111.08656)


  Causal effect variational autoencoder (CEVAE) are trained to predict the
outcome given observational treatment data, while uniform treatment variational
autoencoders (UTVAE) are trained with uniform treatment distribution using
importance sampling. In this paper, we show that using uniform treatment over
observational treatment distribution leads to better causal inference by
mitigating the distribution shift that occurs from training to test time. We
also explore the combination of uniform and observational treatment
distributions with inference and generative network training objectives to find
a better training procedure for inferring treatment effect. Experimentally, we
find that the proposed UTVAE yields better absolute average treatment effect
error and precision in estimation of heterogeneous effect error than the CEVAE
on synthetic and IHDP datasets.

    

### [[2111.08658] A Comparative Study on Transfer Learning and Distance Metrics in Semantic Clustering over the COVID-19 Tweets](http://arxiv.org/abs/2111.08658)


  This paper is a comparison study in the context of Topic Detection on
COVID-19 data. There are various approaches for Topic Detection, among which
the Clustering approach is selected in this paper. Clustering requires distance
and calculating distance needs embedding. The aim of this research is to
simultaneously study the three factors of embedding methods, distance metrics
and clustering methods and their interaction. A dataset including one-month
tweets collected with COVID-19-related hashtags is used for this study. Five
methods, from earlier to new methods, are selected among the embedding methods:
Word2Vec, fastText, GloVe, BERT and T5. Five clustering methods are
investigated in this paper that are: k-means, DBSCAN, OPTICS, spectral and
Jarvis-Patrick. Euclidian distance and Cosine distance as the most important
distance metrics in this field are also examined. First, more than 7,500 tests
are performed to tune the parameters. Then, all the different combinations of
embedding methods with distance metrics and clustering methods are investigated
by silhouette metric. The number of these combinations is 50 cases. First, the
results of these 50 tests are examined. Then, the rank of each method is taken
into account in all the tests of that method. Finally, the major variables of
the research (embedding methods, distance metrics and clustering methods) are
studied separately. Averaging is performed over the control variables to
neutralize their effect. The experimental results show that T5 strongly
outperforms other embedding methods in terms of silhouette metric. In terms of
distance metrics, cosine distance is weakly better. DBSCAN is also superior to
other methods in terms of clustering methods.

    

### [[2111.08667] Machine Learning and Ensemble Approach Onto Predicting Heart Disease](http://arxiv.org/abs/2111.08667)


  The four essential chambers of one's heart that lie in the thoracic cavity
are crucial for one's survival, yet ironically prove to be the most vulnerable.
Cardiovascular disease (CVD) also commonly referred to as heart disease has
steadily grown to the leading cause of death amongst humans over the past few
decades. Taking this concerning statistic into consideration, it is evident
that patients suffering from CVDs need a quick and correct diagnosis in order
to facilitate early treatment to lessen the chances of fatality. This paper
attempts to utilize the data provided to train classification models such as
Logistic Regression, K Nearest Neighbors, Support Vector Machine, Decision
Tree, Gaussian Naive Bayes, Random Forest, and Multi-Layer Perceptron
(Artificial Neural Network) and eventually using a soft voting ensemble
technique in order to attain as many correct diagnoses as possible.

    

### [[2111.08674] Multiclass Optimal Classification Trees with SVM-splits](http://arxiv.org/abs/2111.08674)


  In this paper we present a novel mathematical optimization-based methodology
to construct tree-shaped classification rules for multiclass instances. Our
approach consists of building Classification Trees in which, except for the
leaf nodes, the labels are temporarily left out and grouped into two classes by
means of a SVM separating hyperplane. We provide a Mixed Integer Non Linear
Programming formulation for the problem and report the results of an extended
battery of computational experiments to assess the performance of our proposal
with respect to other benchmarking classification methods.

    

### [[2111.08679] Automatically detecting anomalous exoplanet transits](http://arxiv.org/abs/2111.08679)


  Raw light curve data from exoplanet transits is too complex to naively apply
traditional outlier detection methods. We propose an architecture which
estimates a latent representation of both the main transit and residual
deviations with a pair of variational autoencoders. We show, using two
fabricated datasets, that our latent representations of anomalous transit
residuals are significantly more amenable to outlier detection than raw data or
the latent representation of a traditional variational autoencoder. We then
apply our method to real exoplanet transit data. Our study is the first which
automatically identifies anomalous exoplanet transit light curves. We
additionally release three first-of-their-kind datasets to enable further
research.

    

### [[2111.08683] Inferring halo masses with Graph Neural Networks](http://arxiv.org/abs/2111.08683)


  Understanding the halo-galaxy connection is fundamental in order to improve
our knowledge on the nature and properties of dark matter. In this work we
build a model that infers the mass of a halo given the positions, velocities,
stellar masses, and radii of the galaxies it hosts. In order to capture
information from correlations among galaxy properties and their phase-space, we
use Graph Neural Networks (GNNs), that are designed to work with irregular and
sparse data. We train our models on galaxies from more than 2,000
state-of-the-art simulations from the Cosmology and Astrophysics with MachinE
Learning Simulations (CAMELS) project. Our model, that accounts for
cosmological and astrophysical uncertainties, is able to constrain the masses
of the halos with a $\sim$0.2 dex accuracy. Furthermore, a GNN trained on a
suite of simulations is able to preserve part of its accuracy when tested on
simulations run with a different code that utilizes a distinct subgrid physics
model, showing the robustness of our method. The PyTorch Geometric
implementation of the GNN is publicly available on Github at
this https URL


### [[2111.08687] INTERN: A New Learning Paradigm Towards General Vision](http://arxiv.org/abs/2111.08687)


  Enormous waves of technological innovations over the past several years,
marked by the advances in AI technologies, are profoundly reshaping the
industry and the society. However, down the road, a key challenge awaits us,
that is, our capability of meeting rapidly-growing scenario-specific demands is
severely limited by the cost of acquiring a commensurate amount of training
data. This difficult situation is in essence due to limitations of the
mainstream learning paradigm: we need to train a new model for each new
scenario, based on a large quantity of well-annotated data and commonly from
scratch. In tackling this fundamental problem, we move beyond and develop a new
learning paradigm named INTERN. By learning with supervisory signals from
multiple sources in multiple stages, the model being trained will develop
strong generalizability. We evaluate our model on 26 well-known datasets that
cover four categories of tasks in computer vision. In most cases, our models,
adapted with only 10% of the training data in the target domain, outperform the
counterparts trained with the full set of data, often by a significant margin.
This is an important step towards a promising prospect where such a model with
general vision capability can dramatically reduce our reliance on data, thus
expediting the adoption of AI technologies. Furthermore, revolving around our
new paradigm, we also introduce a new data system, a new architecture, and a
new benchmark, which, together, form a general vision ecosystem to support its
future development in an open and inclusive manner.

    

### [[2111.08691] Uncertainty quantification and inverse modeling for subsurface flow in 3D heterogeneous formations using a theory-guided convolutional encoder-decoder network](http://arxiv.org/abs/2111.08691)


  We build surrogate models for dynamic 3D subsurface single-phase flow
problems with multiple vertical producing wells. The surrogate model provides
efficient pressure estimation of the entire formation at any timestep given a
stochastic permeability field, arbitrary well locations and penetration
lengths, and a timestep matrix as inputs. The well production rate or bottom
hole pressure can then be determined based on Peaceman's formula. The original
surrogate modeling task is transformed into an image-to-image regression
problem using a convolutional encoder-decoder neural network architecture. The
residual of the governing flow equation in its discretized form is incorporated
into the loss function to impose theoretical guidance on the model training
process. As a result, the accuracy and generalization ability of the trained
surrogate models are significantly improved compared to fully data-driven
models. They are also shown to have flexible extrapolation ability to
permeability fields with different statistics. The surrogate models are used to
conduct uncertainty quantification considering a stochastic permeability field,
as well as to infer unknown permeability information based on limited well
production data and observation data of formation properties. Results are shown
to be in good agreement with traditional numerical simulation tools, but
computational efficiency is dramatically improved.

    

### [[2111.08693] Inverting brain grey matter models with likelihood-free inference: a tool for trustable cytoarchitecture measurements](http://arxiv.org/abs/2111.08693)


  Effective characterisation of the brain grey matter cytoarchitecture with
quantitative sensitivity to soma density and volume remains an unsolved
challenge in diffusion MRI (dMRI). Solving the problem of relating the dMRI
signal with cytoarchitectural characteristics calls for the definition of a
mathematical model that describes brain tissue via a handful of
physiologically-relevant parameters and an algorithm for inverting the model.
To address this issue, we propose a new forward model, specifically a new
system of equations, requiring a few relatively sparse b-shells. We then apply
modern tools from Bayesian analysis known as likelihood-free inference (LFI) to
invert our proposed model. As opposed to other approaches from the literature,
our algorithm yields not only an estimation of the parameter vector $\theta$
that best describes a given observed data point $x_0$, but also a full
posterior distribution $p(\theta|x_0)$ over the parameter space. This enables a
richer description of the model inversion, providing indicators such as
credible intervals for the estimated parameters and a complete characterization
of the parameter regions where the model may present indeterminacies. We
approximate the posterior distribution using deep neural density estimators,
known as normalizing flows, and fit them using a set of repeated simulations
from the forward model. We validate our approach on simulations using dmipy and
then apply the whole pipeline on two publicly available datasets.

    

### [[1605.07725] Adversarial Training Methods for Semi-Supervised Text Classification](http://arxiv.org/abs/1605.07725)


  Adversarial training provides a means of regularizing supervised learning
algorithms while virtual adversarial training is able to extend supervised
learning algorithms to the semi-supervised setting. However, both methods
require making small perturbations to numerous entries of the input vector,
which is inappropriate for sparse high-dimensional inputs such as one-hot word
representations. We extend adversarial and virtual adversarial training to the
text domain by applying perturbations to the word embeddings in a recurrent
neural network rather than to the original input itself. The proposed method
achieves state of the art results on multiple benchmark semi-supervised and
purely supervised tasks. We provide visualizations and analysis showing that
the learned word embeddings have improved in quality and that while training,
the model is less prone to overfitting. Code is available at
this https URL.

    

### [[1709.05506] A statistical interpretation of spectral embedding: the generalised random dot product graph](http://arxiv.org/abs/1709.05506)


  Spectral embedding is a procedure which can be used to obtain vector
representations of the nodes of a graph. This paper proposes a generalisation
of the latent position network model known as the random dot product graph, to
allow interpretation of those vector representations as latent position
estimates. The generalisation is needed to model heterophilic connectivity
(e.g., `opposites attract') and to cope with negative eigenvalues more
generally. We show that, whether the adjacency or normalised Laplacian matrix
is used, spectral embedding produces uniformly consistent latent position
estimates with asymptotically Gaussian error (up to identifiability). The
standard and mixed membership stochastic block models are special cases in
which the latent positions take only $K$ distinct vector values, representing
communities, or live in the $(K-1)$-simplex with those vertices, respectively.
Under the stochastic block model, our theory suggests spectral clustering using
a Gaussian mixture model (rather than $K$-means) and, under mixed membership,
fitting the minimum volume enclosing simplex, existing recommendations
previously only supported under non-negative-definite assumptions. Empirical
improvements in link prediction (over the random dot product graph), and the
potential to uncover richer latent structure (than posited under the standard
or mixed membership stochastic block models) are demonstrated in a
cyber-security example.

    

### [[1809.03461] Physics-Information-Aided Kriging: Constructing Covariance Functions using Stochastic Simulation Models](http://arxiv.org/abs/1809.03461)


  In this work, we propose a new Gaussian process regression (GPR) method:
physics information aided Kriging (PhIK). In the standard data-driven Kriging,
the unknown function of interest is usually treated as a Gaussian process with
assumed stationary covariance with hyperparameters estimated from data. In
PhIK, we compute the mean and covariance function from realizations of
available stochastic models, e.g., from realizations of governing stochastic
partial differential equations solutions. Such constructed Gaussian process
generally is non-stationary, and does not assume a specific form of the
covariance function. Our approach avoids the optimization step in data-driven
GPR methods to identify the hyperparameters. More importantly, we prove that
the physical constraints in the form of a deterministic linear operator are
guaranteed in the resulting prediction. We also provide an error estimate in
preserving the physical constraints when errors are included in the stochastic
model realizations. To reduce the computational cost of obtaining stochastic
model realizations, we propose a multilevel Monte Carlo estimate of the mean
and covariance functions. Further, we present an active learning algorithm that
guides the selection of additional observation locations. The efficiency and
accuracy of PhIK are demonstrated for reconstructing a partially known modified
Branin function, studying a three-dimensional heat transfer problem and
learning a conservative tracer distribution from sparse concentration
measurements.

    

### [[1809.09582] Contextual Bandits with Cross-learning](http://arxiv.org/abs/1809.09582)


  In the classical contextual bandits problem, in each round $t$, a learner
observes some context $c$, chooses some action $i$ to perform, and receives
some reward $r_{i,t}(c)$. We consider the variant of this problem where in
addition to receiving the reward $r_{i,t}(c)$, the learner also learns the
values of $r_{i,t}(c')$ for some other contexts $c'$ in set $\mathcal{O}_i(c)$;
i.e., the rewards that would have been achieved by performing that action under
different contexts $c'\in \mathcal{O}_i(c)$. This variant arises in several
strategic settings, such as learning how to bid in non-truthful repeated
auctions, which has gained a lot of attention lately as many platforms have
switched to running first-price auctions. We call this problem the contextual
bandits problem with cross-learning. The best algorithms for the classical
contextual bandits problem achieve $\tilde{O}(\sqrt{CKT})$ regret against all
stationary policies, where $C$ is the number of contexts, $K$ the number of
actions, and $T$ the number of rounds. We design and analyze new algorithms for
the contextual bandits problem with cross-learning and show that their regret
has better dependence on the number of contexts. Under complete cross-learning
where the rewards for all contexts are learned when choosing an action, i.e.,
set $\mathcal{O}_i(c)$ contains all contexts, we show that our algorithms
achieve regret $\tilde{O}(\sqrt{KT})$, removing the dependence on $C$. For any
other cases, i.e., under partial cross-learning where $|\mathcal{O}_i(c)|< C$
for some context-action pair of $(i,c)$, the regret bounds depend on how the
sets $\mathcal O_i(c)$ impact the degree to which cross-learning between
contexts is possible. We simulate our algorithms on real auction data from an
ad exchange running first-price auctions and show that they outperform
traditional contextual bandit algorithms.

    

### [[1902.00194] Sharp Analysis of Expectation-Maximization for Weakly Identifiable Models](http://arxiv.org/abs/1902.00194)


  We study a class of weakly identifiable location-scale mixture models for
which the maximum likelihood estimates based on $n$ i.i.d. samples are known to
have lower accuracy than the classical $n^{- \frac{1}{2}}$ error. We
investigate whether the Expectation-Maximization (EM) algorithm also converges
slowly for these models. We provide a rigorous characterization of EM for
fitting a weakly identifiable Gaussian mixture in a univariate setting where we
prove that the EM algorithm converges in order $n^{\frac{3}{4}}$ steps and
returns estimates that are at a Euclidean distance of order ${ n^{-
\frac{1}{8}}}$ and ${ n^{-\frac{1} {4}}}$ from the true location and scale
parameter respectively. Establishing the slow rates in the univariate setting
requires a novel localization argument with two stages, with each stage
involving an epoch-based argument applied to a different surrogate EM operator
at the population level. We demonstrate several multivariate ($d \geq 2$)
examples that exhibit the same slow rates as the univariate case. We also prove
slow statistical rates in higher dimensions in a special case, when the fitted
covariance is constrained to be a multiple of the identity.

    

### [[1905.10651] Asymptotic Distributions and Rates of Convergence for Random Forests via Generalized U-statistics](http://arxiv.org/abs/1905.10651)


  Random forests remain among the most popular off-the-shelf supervised
learning algorithms. Despite their well-documented empirical success, however,
until recently, few theoretical results were available to describe their
performance and behavior. In this work we push beyond recent work on
consistency and asymptotic normality by establishing rates of convergence for
random forests and other supervised learning ensembles. We develop the notion
of generalized U-statistics and show that within this framework, random forest
predictions can potentially remain asymptotically normal for larger subsample
sizes than previously established. We also provide Berry-Esseen bounds in order
to quantify the rate at which this convergence occurs, making explicit the
roles of the subsample size and the number of trees in determining the
distribution of random forest predictions.

    

### [[2004.08730] Predicting MMSE Score from Finger-Tapping Measurement](http://arxiv.org/abs/2004.08730)


  Dementia is a leading cause of diseases for the elderly. Early diagnosis is
very important for the elderly living with dementias. In this paper, we propose
a method for dementia diagnosis by predicting MMSE score from finger-tapping
measurement with machine learning pipeline. Based on measurement of finger
tapping movement, the pipeline is first to select finger-tapping attributes
with copula entropy and then to predict MMSE score from the selected attributes
with predictive models. Experiments on real world data show that the predictive
models such developed present good prediction performance. As a byproduct, the
associations between certain finger-tapping attributes ('Number of taps',
'Average of intervals', and 'Frequency of taps' of both hands of bimanual
in-phase task) and MMSE score are discovered with copula entropy, which may be
interpreted as the biological relationship between cognitive ability and motor
ability and therefore makes the predictive models explainable. The selected
finger-tapping attributes can be considered as dementia biomarkers.

    

### [[2006.07897] Entropic gradient descent algorithms and wide flat minima](http://arxiv.org/abs/2006.07897)


  The properties of flat minima in the empirical risk landscape of neural
networks have been debated for some time. Increasing evidence suggests they
possess better generalization capabilities with respect to sharp ones. First,
we discuss Gaussian mixture classification models and show analytically that
there exist Bayes optimal pointwise estimators which correspond to minimizers
belonging to wide flat regions. These estimators can be found by applying
maximum flatness algorithms either directly on the classifier (which is norm
independent) or on the differentiable loss function used in learning. Next, we
extend the analysis to the deep learning scenario by extensive numerical
validations. Using two algorithms, Entropy-SGD and Replicated-SGD, that
explicitly include in the optimization objective a non-local flatness measure
known as local entropy, we consistently improve the generalization error for
common architectures (e.g. ResNet, EfficientNet). An easy to compute flatness
measure shows a clear correlation with test accuracy.

    

### [[2006.16142] $k$FW: A Frank-Wolfe style algorithm with stronger subproblem oracles](http://arxiv.org/abs/2006.16142)


  This paper proposes a new variant of Frank-Wolfe (FW), called $k$FW. Standard
FW suffers from slow convergence: iterates often zig-zag as update directions
oscillate around extreme points of the constraint set. The new variant, $k$FW,
overcomes this problem by using two stronger subproblem oracles in each
iteration. The first is a $k$ linear optimization oracle ($k$LOO) that computes
the $k$ best update directions (rather than just one). The second is a $k$
direction search ($k$DS) that minimizes the objective over a constraint set
represented by the $k$ best update directions and the previous iterate. When
the problem solution admits a sparse representation, both oracles are easy to
compute, and $k$FW converges quickly for smooth convex objectives and several
interesting constraint sets: $k$FW achieves finite
$\frac{4L_f^3D^4}{\gamma\delta^2}$ convergence on polytopes and group norm
balls, and linear convergence on spectrahedra and nuclear norm balls. Numerical
experiments validate the effectiveness of $k$FW and demonstrate an
order-of-magnitude speedup over existing approaches.

    

### [[2007.07449] Downsampling for Testing and Learning in Product Distributions](http://arxiv.org/abs/2007.07449)


  We study distribution-free property testing and learning problems where the
unknown probability distribution is a product distribution over $\mathbb{R}^d$.
For many important classes of functions, such as intersections of halfspaces,
polynomial threshold functions, convex sets, and $k$-alternating functions, the
known algorithms either have complexity that depends on the support size of the
distribution, or are proven to work only for specific examples of product
distributions. We introduce a general method, which we call downsampling, that
resolves these issues. Downsampling uses a notion of "rectilinear isoperimetry"
for product distributions, which further strengthens the connection between
isoperimetry, testing, and learning. Using this technique, we attain new
efficient distribution-free algorithms under product distributions on
$\mathbb{R}^d$:
1. A simpler proof for non-adaptive, one-sided monotonicity testing of
functions $[n]^d \to \{0,1\}$, and improved sample complexity for testing
monotonicity over unknown product distributions, from $O(d^7)$ [Black,
Chakrabarty, & Seshadhri, SODA 2020] to $\widetilde O(d^3)$.
2. Polynomial-time agnostic learning algorithms for functions of a constant
number of halfspaces, and constant-degree polynomial threshold functions.
3. An $\exp(O(d \log(dk)))$-time agnostic learning algorithm, and an
$\exp(O(d \log(dk)))$-sample tolerant tester, for functions of $k$ convex sets;
and a $2^{\widetilde O(d)}$ sample-based one-sided tester for convex sets.
4. An $\exp(\widetilde O(k \sqrt d))$-time agnostic learning algorithm for
$k$-alternating functions, and a sample-based tolerant tester with the same
complexity.

    

### [[2007.13695] Adaptive Height Optimisation for Cellular-Connected UAVs using Reinforcement Learning](http://arxiv.org/abs/2007.13695)


  Providing reliable connectivity to cellular-connected UAVs can be very
challenging; their performance highly depends on the nature of the surrounding
environment, such as density and heights of the ground BSs. On the other hand,
tall buildings might block undesired interference signals from ground BSs,
thereby improving the connectivity between the UAVs and their serving BSs. To
address the connectivity of UAVs in such environments, this paper proposes a RL
algorithm to dynamically optimise the height of a UAV as it moves through the
environment, with the goal of increasing the throughput that it experiences.
The proposed solution is evaluated using experimentally-obtained measurements
from two different locations in Dublin city centre, Ireland. In the first
scenario, the UAV is connected to a macro-cell, while in the second scenario,
the UAVs associates to different small cells in a two-tier mobile network.
Results show that the proposed solution increases 6 to 41% in throughput,
compared to baseline approaches.

    

### [[2008.00113] Multi-officer Routing for Patrolling High Risk Areas Jointly Learned from Check-ins, Crime and Incident Response Data](http://arxiv.org/abs/2008.00113)


  A well-crafted police patrol route design is vital in providing community
safety and security in the society. Previous works have largely focused on
predicting crime events with historical crime data. The usage of large-scale
mobility data collected from Location-Based Social Network, or check-ins, and
Point of Interests (POI) data for designing an effective police patrol is
largely understudied. Given that there are multiple police officers being on
duty in a real-life situation, this makes the problem more complex to solve. In
this paper, we formulate the dynamic crime patrol planning problem for multiple
police officers using check-ins, crime, incident response data, and POI
information. We propose a joint learning and non-random optimisation method for
the representation of possible solutions where multiple police officers patrol
the high crime risk areas simultaneously first rather than the low crime risk
areas. Later, meta-heuristic Genetic Algorithm (GA) and Cuckoo Search (CS) are
implemented to find the optimal routes. The performance of the proposed
solution is verified and compared with several state-of-art methods using
real-world datasets.

    

### [[2008.13777] Low-rank matrix recovery with non-quadratic loss: projected gradient method and regularity projection oracle](http://arxiv.org/abs/2008.13777)


  Existing results for low-rank matrix recovery largely focus on quadratic
loss, which enjoys favorable properties such as restricted strong
convexity/smoothness (RSC/RSM) and well conditioning over all low rank
matrices. However, many interesting problems involve more general,
non-quadratic losses, which do not satisfy such properties. For these problems,
standard nonconvex approaches such as rank-constrained projected gradient
descent (a.k.a. iterative hard thresholding) and Burer-Monteiro factorization
could have poor empirical performance, and there is no satisfactory theory
guaranteeing global and fast convergence for these algorithms.
In this paper, we show that a critical component in provable low-rank
recovery with non-quadratic loss is a regularity projection oracle. This oracle
restricts iterates to low-rank matrices within an appropriate bounded set, over
which the loss function is well behaved and satisfies a set of approximate
RSC/RSM conditions. Accordingly, we analyze an (averaged) projected gradient
method equipped with such an oracle, and prove that it converges globally and
linearly. Our results apply to a wide range of non-quadratic low-rank
estimation problems including one bit matrix sensing/completion, individualized
rank aggregation, and more broadly generalized linear models with rank
constraints.

    

### [[2009.06472] Estimating Individual Treatment Effects using Non-Parametric Regression Models: a Review](http://arxiv.org/abs/2009.06472)


  Large observational data are increasingly available in disciplines such as
health, economic and social sciences, where researchers are interested in
causal questions rather than prediction. In this paper, we examine the problem
of estimating heterogeneous treatment effects using non-parametric
regression-based methods, starting from an empirical study aimed at
investigating the effect of participation in school meal programs on health
indicators. Firstly, we introduce the setup and the issues related to
conducting causal inference with observational or non-fully randomized data,
and how these issues can be tackled with the help of statistical learning
tools. Then, we review and develop a unifying taxonomy of the existing
state-of-the-art frameworks that allow for individual treatment effects
estimation via non-parametric regression models. After presenting a brief
overview on the problem of model selection, we illustrate the performance of
some of the methods on three different simulated studies. We conclude by
demonstrating the use of some of the methods on an empirical analysis of the
school meal program data.

    

### [[2009.10931] Drug Repurposing for COVID-19 using Graph Neural Network with Genetic, Mechanistic, and Epidemiological Validation](http://arxiv.org/abs/2009.10931)


  Amid the pandemic of 2019 novel coronavirus disease (COVID-19) infected by
SARS-CoV-2, a vast amount of drug research for prevention and treatment has
been quickly conducted, but these efforts have been unsuccessful thus far. Our
objective is to prioritize repurposable drugs using a drug repurposing pipeline
that systematically integrates multiple SARS-CoV-2 and drug interactions, deep
graph neural networks, and in-vitro/population-based validations. We first
collected all the available drugs (n= 3,635) involved in COVID-19 patient
treatment through CTDbase. We built a SARS-CoV-2 knowledge graph based on the
interactions among virus baits, host genes, pathways, drugs, and phenotypes. A
deep graph neural network approach was used to derive the candidate
representation based on the biological interactions. We prioritized the
candidate drugs using clinical trial history, and then validated them with
their genetic profiles, in vitro experimental efficacy, and electronic health
records. We highlight the top 22 drugs including Azithromycin, Atorvastatin,
Aspirin, Acetaminophen, and Albuterol. We further pinpointed drug combinations
that may synergistically target COVID-19. In summary, we demonstrated that the
integration of extensive interactions, deep neural networks, and rigorous
validation can facilitate the rapid identification of candidate drugs for
COVID-19 treatment. This is a post-peer-review, pre-copyedit version of an
article published in Scientific Reports The final authenticated version is
available online at: this https URL


### [[2011.08174] Policy design in experiments with unknown interference](http://arxiv.org/abs/2011.08174)


  This paper proposes an experimental design for estimation and inference on
welfare-maximizing policies in the presence of spillover effects. I consider a
setting where units are organized into a finite number of large clusters and
interact in unobserved ways within each cluster. As a first contribution, I
introduce a single-wave experiment to estimate the marginal effect of a change
in the treatment probabilities taking spillovers into account and test for
policy optimality. The design randomizes treatments independently within
clusters and induces local perturbations to treatment probabilities within
pairs of clusters. Using the estimated marginal effect, I construct a practical
test for whether a given treatment allocation rule maximizes welfare, and I
characterize its asymptotic properties. The idea is that researchers should
report estimates of the marginal effect and test for welfare-maximizing
policies: the marginal effect indicates the direction for a welfare
improvement, and the test provides evidence on whether it is worth conducting
additional experiments to estimate a welfare-improving treatment allocation. As
a second contribution, I design a multiple-wave experiment to estimate
treatment assignment rules and maximize welfare. I derive small-sample
guarantees on the difference between the maximum attainable welfare and the
welfare evaluated at the estimated policy (regret). A corollary of such
guarantees is that the regret converges to zero linearly in the number of
iterations and clusters. Simulations calibrated to existing experiments on
information diffusion and cash-transfer programs show that the method leads to
significant welfare improvements.

    

### [[2011.09052] Visual Time Series Forecasting: An Image-driven Approach](http://arxiv.org/abs/2011.09052)


  In this work, we address time-series forecasting as a computer vision task.
We capture input data as an image and train a model to produce the subsequent
image. This approach results in predicting distributions as opposed to
pointwise values. To assess the robustness and quality of our approach, we
examine various datasets and multiple evaluation metrics. Our experiments show
that our forecasting tool is effective for cyclic data but somewhat less for
irregular data such as stock prices. Importantly, when using image-based
evaluation metrics, we find our method to outperform various baselines,
including ARIMA, and a numerical variation of our deep learning approach.

    

### [[2012.13773] Deep reinforcement learning for portfolio management](http://arxiv.org/abs/2012.13773)


  In our paper, we apply deep reinforcement learning approach to optimize
investment decisions in portfolio management. We make several innovations, such
as adding short mechanism and designing an arbitrage mechanism, and applied our
model to make decision optimization for several randomly selected portfolios.
The experimental results show that our model is able to optimize investment
decisions and has the ability to obtain excess return in stock market, and the
optimized agent maintains the asset weights at fixed value throughout the
trading periods and trades at a very low transaction cost rate. In addition, we
redesigned the formula for calculating portfolio asset weights in continuous
trading process which can make leverage trading, that fills the theoretical gap
in the calculation of portfolio weights when shorting.

    

### [[2102.06573] Shrinkage Bayesian Causal Forests for Heterogeneous Treatment Effects Estimation](http://arxiv.org/abs/2102.06573)


  This paper develops a sparsity-inducing version of Bayesian Causal Forests, a
recently proposed nonparametric causal regression model that employs Bayesian
Additive Regression Trees and is specifically designed to estimate
heterogeneous treatment effects using observational data. The sparsity-inducing
component we introduce is motivated by empirical studies where not all the
available covariates are relevant, leading to different degrees of sparsity
underlying the surfaces of interest in the estimation of individual treatment
effects. The extended version presented in this work, which we name Shrinkage
Bayesian Causal Forest, is equipped with an additional pair of priors allowing
the model to adjust the weight of each covariate through the corresponding
number of splits in the tree ensemble. These priors improve the model's
adaptability to sparse data generating processes and allow to perform fully
Bayesian feature shrinkage in a framework for treatment effects estimation, and
thus to uncover the moderating factors driving heterogeneity. In addition, the
method allows prior knowledge about the relevant confounding covariates and the
relative magnitude of their impact on the outcome to be incorporated in the
model. We illustrate the performance of our method in simulated studies, in
comparison to Bayesian Causal Forest and other state-of-the-art models, to
demonstrate how it scales up with an increasing number of covariates and how it
handles strongly confounded scenarios. Finally, we also provide an example of
application using real-world data.

    

### [[2102.13042] Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](http://arxiv.org/abs/2102.13042)


  With a better understanding of the loss surfaces for multilayer networks, we
can build more robust and accurate training procedures. Recently it was
discovered that independently trained SGD solutions can be connected along
one-dimensional paths of near-constant training loss. In this paper, we show
that there are mode-connecting simplicial complexes that form multi-dimensional
manifolds of low loss, connecting many independently trained models. Inspired
by this discovery, we show how to efficiently build simplicial complexes for
fast ensembling, outperforming independently trained deep ensembles in
accuracy, calibration, and robustness to dataset shift. Notably, our approach
only requires a few training epochs to discover a low-loss simplex, starting
from a pre-trained solution. Code is available at
this https URL.

    

### [[2103.00845] A Brief Summary of Interactions Between Meta-Learning and Self-Supervised Learning](http://arxiv.org/abs/2103.00845)


  This paper briefly reviews the connections between meta-learning and
self-supervised learning. Meta-learning can be applied to improve model
generalization capability and to construct general AI algorithms.
Self-supervised learning utilizes self-supervision from original data and
extracts higher-level generalizable features through unsupervised pre-training
or optimization of contrastive loss objectives. In self-supervised learning,
data augmentation techniques are widely applied and data labels are not
required since pseudo labels can be estimated from trained models on similar
tasks. Meta-learning aims to adapt trained deep models to solve diverse tasks
and to develop general AI algorithms. We review the associations of
meta-learning with both generative and contrastive self-supervised learning
models. Unlabeled data from multiple sources can be jointly considered even
when data sources are vastly different. We show that an integration of
meta-learning and self-supervised learning models can best contribute to the
improvement of model generalization capability. Self-supervised learning guided
by meta-learner and general meta-learning algorithms under self-supervision are
both examples of possible combinations.

    

### [[2103.01640] Double Coverage with Machine-Learned Advice](http://arxiv.org/abs/2103.01640)


  We study the fundamental online k-server problem in a learning-augmented
setting. While in the traditional online model, an algorithm has no information
about the request sequence, we assume that there is given some advice (e.g.
machine-learned predictions) on an algorithm's decision. There is, however, no
guarantee on the quality of the prediction and it might be far from being
correct.
Our main result is a learning-augmented variation of the well-known Double
Coverage algorithm for k-server on the line (Chrobak et al., SIDMA 1991) in
which we integrate predictions as well as our trust into their quality. We give
an error-dependent competitive ratio, which is a function of a user-defined
confidence parameter, and which interpolates smoothly between an optimal
consistency, the performance in case that all predictions are correct, and the
best-possible robustness regardless of the prediction quality. When given good
predictions, we improve upon known lower bounds for online algorithms without
advice. We further show that our algorithm achieves for any k an almost optimal
consistency-robustness tradeoff, within a class of deterministic algorithms
respecting local and memoryless properties.
Our algorithm outperforms a previously proposed (more general)
learning-augmented algorithm. It is remarkable that the previous algorithm
crucially exploits memory, whereas our algorithm is memoryless. Finally, we
demonstrate in experiments the practicability and the superior performance of
our algorithm on real-world data.

    

### [[2105.05622] On risk-based active learning for structural health monitoring](http://arxiv.org/abs/2105.05622)


  A primary motivation for the development and implementation of structural
health monitoring systems, is the prospect of gaining the ability to make
informed decisions regarding the operation and maintenance of structures and
infrastructure. Unfortunately, descriptive labels for measured data
corresponding to health-state information for the structure of interest are
seldom available prior to the implementation of a monitoring system. This issue
limits the applicability of the traditional supervised and unsupervised
approaches to machine learning in the development of statistical classifiers
for decision-supporting SHM systems.
The current paper presents a risk-based formulation of active learning, in
which the querying of class-label information is guided by the expected value
of said information for each incipient data point. When applied to structural
health monitoring, the querying of class labels can be mapped onto the
inspection of a structure of interest in order to determine its health state.
In the current paper, the risk-based active learning process is explained and
visualised via a representative numerical example and subsequently applied to
the Z24 Bridge benchmark. The results of the case studies indicate that a
decision-maker's performance can be improved via the risk-based active learning
of a statistical classifier, such that the decision process itself is taken
into account.

    

### [[2106.12894] InFlow: Robust outlier detection utilizing Normalizing Flows](http://arxiv.org/abs/2106.12894)


  Normalizing flows are prominent deep generative models that provide tractable
probability distributions and efficient density estimation. However, they are
well known to fail while detecting Out-of-Distribution (OOD) inputs as they
directly encode the local features of the input representations in their latent
space. In this paper, we solve this overconfidence issue of normalizing flows
by demonstrating that flows, if extended by an attention mechanism, can
reliably detect outliers including adversarial attacks. Our approach does not
require outlier data for training and we showcase the efficiency of our method
for OOD detection by reporting state-of-the-art performance in diverse
experimental settings. Code available at
this https URL .

    

### [[2108.01034] An Efficient Image-to-Image Translation HourGlass-based Architecture for Object Pushing Policy Learning](http://arxiv.org/abs/2108.01034)


  Humans effortlessly solve pushing tasks in everyday life but unlocking these
capabilities remains a challenge in robotics because physics models of these
tasks are often inaccurate or unattainable. State-of-the-art data-driven
approaches learn to compensate for these inaccuracies or replace the
approximated physics models altogether. Nevertheless, approaches like Deep
Q-Networks (DQNs) suffer from local optima in large state-action spaces.
Furthermore, they rely on well-chosen deep learning architectures and learning
paradigms. In this paper, we propose to frame the learning of pushing policies
(where to push and how) by DQNs as an image-to-image translation problem and
exploit an Hourglass-based architecture. We present an architecture combining a
predictor of which pushes lead to changes in the environment with a
state-action value predictor dedicated to the pushing task. Moreover, we
investigate positional information encoding to learn position-dependent policy
behaviors. We demonstrate in simulation experiments with a UR5 robot arm that
our overall architecture helps the DQN learn faster and achieve higher
performance in a pushing task involving objects with unknown dynamics.

    

### [[2109.02411] Data-Driven Wind Turbine Wake Modeling via Probabilistic Machine Learning](http://arxiv.org/abs/2109.02411)


  Wind farm design primarily depends on the variability of the wind turbine
wake flows to the atmospheric wind conditions, and the interaction between
wakes. Physics-based models that capture the wake flow-field with high-fidelity
are computationally very expensive to perform layout optimization of wind
farms, and, thus, data-driven reduced order models can represent an efficient
alternative for simulating wind farms. In this work, we use real-world light
detection and ranging (LiDAR) measurements of wind-turbine wakes to construct
predictive surrogate models using machine learning. Specifically, we first
demonstrate the use of deep autoencoders to find a low-dimensional
\emph{latent} space that gives a computationally tractable approximation of the
wake LiDAR measurements. Then, we learn the mapping between the parameter space
and the (latent space) wake flow-fields using a deep neural network.
Additionally, we also demonstrate the use of a probabilistic machine learning
technique, namely, Gaussian process modeling, to learn the
parameter-space-latent-space mapping in addition to the epistemic and aleatoric
uncertainty in the data. Finally, to cope with training large datasets, we
demonstrate the use of variational Gaussian process models that provide a
tractable alternative to the conventional Gaussian process models for large
datasets. Furthermore, we introduce the use of active learning to adaptively
build and improve a conventional Gaussian process model predictive capability.
Overall, we find that our approach provides accurate approximations of the
wind-turbine wake flow field that can be queried at an orders-of-magnitude
cheaper cost than those generated with high-fidelity physics-based simulations.

    

### [[2111.00901] Click-Based Student Performance Prediction: A Clustering Guided Meta-Learning Approach](http://arxiv.org/abs/2111.00901)


  We study the problem of predicting student knowledge acquisition in online
courses from clickstream behavior. Motivated by the proliferation of eLearning
lecture delivery, we specifically focus on student in-video activity in
lectures videos, which consist of content and in-video quizzes. Our methodology
for predicting in-video quiz performance is based on three key ideas we
develop. First, we model students' clicking behavior via time-series learning
architectures operating on raw event data, rather than defining hand-crafted
features as in existing approaches that may lose important information embedded
within the click sequences. Second, we develop a self-supervised clickstream
pre-training to learn informative representations of clickstream events that
can initialize the prediction model effectively. Third, we propose a clustering
guided meta-learning-based training that optimizes the prediction model to
exploit clusters of frequent patterns in student clickstream sequences. Through
experiments on three real-world datasets, we demonstrate that our method
obtains substantial improvements over two baseline models in predicting
students' in-video quiz performance. Further, we validate the importance of the
pre-training and meta-learning components of our framework through ablation
studies. Finally, we show how our methodology reveals insights on
video-watching behavior associated with knowledge acquisition for useful
learning analytics.

    

### [[2111.08134] A fixed latency ORBGRAND decoder architecture with LUT-aided error-pattern scheduling](http://arxiv.org/abs/2111.08134)


  Guessing Random Additive Noise Decoding (GRAND) is a universal decoding
algorithm that has been recently proposed as a practical way to perform maximum
likelihood decoding. It generates a sequence of possible error patterns and
applies them to the received vector, checking if the result is a valid
codeword. Ordered reliability bits GRAND (ORBGRAND) improves on GRAND by
considering soft information received from the channel. Both GRAND and ORBGRAND
have been implemented in hardware, focusing on average performance, sacrificing
worst case throughput and latency. In this work, an improved pattern schedule
for ORBGRAND is proposed. It provides $>0.5$dB gain over the standard schedule
at a block error rate $\le 10^{-5}$, and outperforms more complex GRAND flavors
with a fraction of the complexity. The proposed schedule is used within a novel
code-agnositic decoder architecture: the decoder guarantees fixed high
throughput and low latency, making it attractive for latency-constrained
applications. It outperforms the worst-case performance of decoders by orders
of magnitude, and outperforms many best-case figures. Decoding a code of length
128, it achieves a throughput of $79.21$Gb/s with $58.49$ns latency, and of
$69.61$Gb/s with $40.58$ns latency, yielding better energy efficiency and
comparable area efficiency with respect to the state of the art.

    

### [[2111.08142] Quo Vadis MPI RMA? Towards a More Efficient Use of MPI One-Sided Communication](http://arxiv.org/abs/2111.08142)


  The MPI standard has long included one-sided communication abstractions
through the MPI Remote Memory Access (RMA) interface. Unfortunately, the MPI
RMA chapter in the 4.0 version of the MPI standard still contains both
well-known and lesser known short-comings for both implementations and users,
which lead to potentially non-optimal usage patterns. In this paper, we
identify a set of issues and propose ways for applications to better express
anticipated usage of RMA routines, allowing the MPI implementation to better
adapt to the application's needs. In order to increase the flexibility of the
RMA interface, we add the capability to duplicate windows, allowing access to
the same resources encapsulated by a window using different configurations. In
the same vein, we introduce the concept of MPI memory handles, meant to provide
life-time guarantees on memory attached to dynamic windows, removing the
overhead currently present in using dynamically exposed memory. We will show
that our extensions provide improved accumulate latencies, reduced overheads
for multi-threaded flushes, and allow for zero overhead dynamic memory window
usage.

    

### [[2111.08203] Is CADP an Applicable Formal Method?](http://arxiv.org/abs/2111.08203)


  CADP is a comprehensive toolbox implementing results of concurrency theory.
This paper addresses the question, whether CADP qualifies as an applicable
formal method, based on the experience of the authors and feedback reported by
users.

    

### [[2111.08206] JMSNAS: Joint Model Split and Neural Architecture Search for Learning over Mobile Edge Networks](http://arxiv.org/abs/2111.08206)


  The main challenge to deploy deep neural network (DNN) over a mobile edge
network is how to split the DNN model so as to match the network architecture
as well as all the nodes' computation and communication capacity. This
essentially involves two highly coupled procedures: model generating and model
splitting. In this paper, a joint model split and neural architecture search
(JMSNAS) framework is proposed to automatically generate and deploy a DNN model
over a mobile edge network. Considering both the computing and communication
resource constraints, a computational graph search problem is formulated to
find the multi-split points of the DNN model, and then the model is trained to
meet some accuracy requirements. Moreover, the trade-off between model accuracy
and completion latency is achieved through the proper design of the objective
function. The experiment results confirm the superiority of the proposed
framework over the state-of-the-art split machine learning design methods.

    

### [[2111.08232] Online Self-Evolving Anomaly Detection in Cloud Computing Environments](http://arxiv.org/abs/2111.08232)


  Modern cloud computing systems contain hundreds to thousands of computing and
storage servers. Such a scale, combined with ever-growing system complexity, is
causing a key challenge to failure and resource management for dependable cloud
computing. Autonomic failure detection is a crucial technique for understanding
emergent, cloud-wide phenomena and self-managing cloud resources for
system-level dependability assurance. To detect failures, we need to monitor
the cloud execution and collect runtime performance data. These data are
usually unlabeled, and thus a prior failure history is not always available in
production clouds. In this paper, we present a \emph{self-evolving anomaly
detection} (SEAD) framework for cloud dependability assurance. Our framework
self-evolves by recursively exploring newly verified anomaly records and
continuously updating the anomaly detector online. As a distinct advantage of
our framework, cloud system administrators only need to check a small number of
detected anomalies, and their decisions are leveraged to update the detector.
Thus, the detector evolves following the upgrade of system hardware, update of
the software stack, and change of user workloads. Moreover, we design two types
of detectors, one for general anomaly detection and the other for type-specific
anomaly detection. With the help of self-evolving techniques, our detectors can
achieve 88.94\% in sensitivity and 94.60\% in specificity on average, which
makes them suitable for real-world deployment.

    

### [[2111.08272] Task allocation for decentralized training in heterogeneous environment](http://arxiv.org/abs/2111.08272)


  The demand for large-scale deep learning is increasing, and distributed
training is the current mainstream solution. Ring AllReduce is widely used as a
data parallel decentralized algorithm. However, in a heterogeneous environment,
each worker calculates the same amount of data, so that there is a lot of
waiting time loss among different workers, which makes the algorithm unable to
adapt well to heterogeneous clusters. Resources are not used as they should be.
In this paper, we design an implementation of static allocation algorithm. The
dataset is artificially allocated to each worker, and samples are drawn
proportionally for training, thereby speeding up the training speed of the
network in a heterogeneous environment. We verify the convergence and influence
on training speed of the network model under this algorithm on one machine with
multi-card and multi-machine with multi-card. On this basis of feasibility, we
propose a self-adaptive allocation algorithm that allows each machine to find
the data it needs to adapt to the current environment. The self-adaptive
allocation algorithm can reduce the training time by nearly one-third to half
compared to the same proportional this http URL order to better show the
applicability of the algorithm in heterogeneous clusters, We replace a poorly
performing worker with a good performing worker or add a poorly performing
worker to the heterogeneous cluster. Experimental results show that training
time will decrease as the overall performance improves. Therefore, it means
that resources are fully used. Further, this algorithm is not only suitable for
straggler problems, but also for most heterogeneous situations. It can be used
as a plug-in for AllReduce and its variant algorithms.

    

### [[2111.08300] Highly Efficient Indexing Scheme for k-Dominant Skyline Processing over Uncertain Data Streams](http://arxiv.org/abs/2111.08300)


  Skyline is widely used in reality to solve multi-criteria problems, such as
environmental monitoring and business decision-making. When a data is not worse
than another data on all criteria and is better than another data at least one
criterion, the data is said to dominate another data. When a data item is not
dominated by any other data item, this data is said to be a member of the
skyline. However, as the number of criteria increases, the possibility that a
data dominates another data decreases, resulting in too many members of the
skyline set. To solve this kind of problem, the concept of the k-dominant
skyline was proposed, which reduces the number of skyline members by relaxing
the limit. The uncertainty of the data makes each data have a probability of
appearing, so each data has the probability of becoming a member of the
k-dominant skyline. When a new data item is added, the probability of other
data becoming members of the k-dominant skyline may change. How to quickly
update the k-dominant skyline for real-time applications is a serious problem.
This paper proposes an effective method, Middle Indexing (MI), which filters
out a large amount of irrelevant data in the uncertain data stream by sorting
data specifically, so as to improve the efficiency of updating the k-dominant
skyline. Experiments show that the proposed MI outperforms the existing method
by approximately 13% in terms of computation time.

    

### [[2111.08339] Global Sensitivity Analysis of Four Chamber Heart Hemodynamics Using Surrogate Models](http://arxiv.org/abs/2111.08339)


  Computational Fluid Dynamics (CFD) is used to assist in designing artificial
valves and planning procedures, focusing on local flow features. However,
assessing the impact on overall cardiovascular function or predicting
longer-term outcomes may require more comprehensive whole heart CFD models.
Fitting such models to patient data requires numerous computationally expensive
simulations, and depends on specific clinical measurements to constrain model
parameters, hampering clinical adoption. Surrogate models can help to
accelerate the fitting process while accounting for the added uncertainty. We
create a validated patient-specific four-chamber heart CFD model based on the
Navier-Stokes-Brinkman (NSB) equations and test Gaussian Process Emulators
(GPEs) as a surrogate model for performing a variance-based global sensitivity
analysis (GSA). GSA identified preload as the dominant driver of flow in both
the right and left side of the heart, respectively. Left-right differences were
seen in terms of vascular outflow resistances, with pulmonary artery resistance
having a much larger impact on flow than aortic resistance. Our results suggest
that GPEs can be used to identify parameters in personalized whole heart CFD
models, and highlight the importance of accurate preload measurements.

    

### [[2111.08348] Self-Stabilization and Byzantine Tolerance for Maximal Independent Set](http://arxiv.org/abs/2111.08348)


  We analyze the impact of transient and Byzantine faults on the construction
of a maximal independent set in a general network. We adapt the
self-stabilizing algorithm presented by Turau \cite{turau2007linear} for
computing such a vertex set. Our algorithm is self-stabilizing and also works
under the more difficult context of arbitrary Byzantine faults.
Byzantine nodes can prevent nodes close to them from taking part in the
independent set for an arbitrarily long time. We give boundaries to their
impact by focusing on the set of all nodes excluding nodes at distance 1 or
less of Byzantine nodes, and excluding some of the nodes at distance 2. As far
as we know, we present the first algorithm tolerating both transient and
Byzantine faults under the fair distributed daemon. We prove that this
algorithm converges in $ \mathcal O(\Delta n)$ rounds w.h.p., where $n$ and
$\Delta$ are the size and the maximum degree of the network, resp.
Additionally, we present a modified version of this algorithm for anonymous
systems under the adversarial distributed daemon that converges in
$ \mathcal O(n^{2})$ expected number of steps.

    

### [[2012.04705] Structured Index Coding Problem and Multi-access Coded Caching](http://arxiv.org/abs/2012.04705)


  Index coding and coded caching are two active research topics in information
theory with strong ties to each other. Motivated by the multi-access coded
caching problem, we study a new class of structured index coding problems
(ICPs) which are formed by the union of several symmetric ICPs. We derive upper
and lower bounds on the optimal server transmission rate for this class of ICPs
and demonstrate that they differ by at most a factor of two. Finally, we apply
these results to the multi-access coded caching problem to derive better bounds
than the state of the art.

    

### [[2111.08054] Revisiting C.S.Peirce's Experiment: 150 Years Later](http://arxiv.org/abs/2111.08054)


  An iconoclastic philosopher and polymath, Charles Sanders Peirce (1837-1914)
is among the greatest of American minds. In 1872, Peirce conducted a series of
experiments to determine the distribution of response times to an auditory
stimulus, which is widely regarded as one of the most significant statistical
investigations in the history of nineteenth-century American mathematical
research (Stigler, 1978). On the 150th anniversary of this historic experiment,
we look back at Peirce's view on empirical modeling through a modern
statistical lens.

    

### [[2111.08102] The Partially Observable History Process](http://arxiv.org/abs/2111.08102)


  We introduce the partially observable history process (POHP) formalism for
reinforcement learning. POHP centers around the actions and observations of a
single agent and abstracts away the presence of other players without reducing
them to stochastic processes. Our formalism provides a streamlined interface
for designing algorithms that defy categorization as exclusively single or
multi-agent, and for developing theory that applies across these domains. We
show how the POHP formalism unifies traditional models including the Markov
decision process, the Markov game, the extensive-form game, and their partially
observable extensions, without introducing burdensome technical machinery or
violating the philosophical underpinnings of reinforcement learning. We
illustrate the utility of our formalism by concisely exploring observable
sequential rationality, re-deriving the extensive-form regret minimization
(EFR) algorithm, and examining EFR's theoretical properties in greater
generality.

    

### [[2111.08108] Learning Optimal Control with Stochastic Models of Hamiltonian Dynamics](http://arxiv.org/abs/2111.08108)


  Optimal control problems can be solved by first applying the Pontryagin
maximum principle, followed by computing a solution of the corresponding
unconstrained Hamiltonian dynamical system. In this paper, and to achieve a
balance between robustness and efficiency, we learn a reduced Hamiltonian of
the unconstrained Hamiltonian. This reduced Hamiltonian is learned by going
backward in time and by minimizing the loss function resulting from application
of the Pontryagin maximum principle conditions. The robustness of our learning
process is then further improved by progressively learning a posterior
distribution of reduced Hamiltonians. This leads to a more efficient sampling
of the generalized coordinates (position, velocity) of our phase space. Our
solution framework applies to not only optimal control problems with
finite-dimensional phase (state) spaces but also the infinite dimensional case.

    

### [[2111.08156] Improving Learning from Demonstrations by Learning from Experience](http://arxiv.org/abs/2111.08156)


  How to make imitation learning more general when demonstrations are
relatively limited has been a persistent problem in reinforcement learning
(RL). Poor demonstrations lead to narrow and biased date distribution,
non-Markovian human expert demonstration makes it difficult for the agent to
learn, and over-reliance on sub-optimal trajectories can make it hard for the
agent to improve its performance. To solve these problems we propose a new
algorithm named TD3fG that can smoothly transition from learning from experts
to learning from experience. Our algorithm achieves good performance in the
MUJOCO environment with limited and sub-optimal demonstrations. We use behavior
cloning to train the network as a reference action generator and utilize it in
terms of both loss function and exploration noise. This innovation can help
agents extract a priori knowledge from demonstrations while reducing the
detrimental effects of the poor Markovian properties of the demonstrations. It
has a better performance compared to the BC+ fine-tuning and DDPGfD approach,
especially when the demonstrations are relatively limited. We call our method
TD3fG meaning TD3 from a generator.

    

### [[2111.08222] Will We Trust What We Don't Understand? Impact of Model Interpretability and Outcome Feedback on Trust in AI](http://arxiv.org/abs/2111.08222)


  Despite AI's superhuman performance in a variety of domains, humans are often
unwilling to adopt AI systems. The lack of interpretability inherent in many
modern AI techniques is believed to be hurting their adoption, as users may not
trust systems whose decision processes they do not understand. We investigate
this proposition with a novel experiment in which we use an interactive
prediction task to analyze the impact of interpretability and outcome feedback
on trust in AI and on human performance in AI-assisted prediction tasks. We
find that interpretability led to no robust improvements in trust, while
outcome feedback had a significantly greater and more reliable effect. However,
both factors had modest effects on participants' task performance. Our findings
suggest that (1) factors receiving significant attention, such as
interpretability, may be less effective at increasing trust than factors like
outcome feedback, and (2) augmenting human performance via AI systems may not
be a simple matter of increasing trust in AI, as increased trust is not always
associated with equally sizable improvements in performance. These findings
invite the research community to focus not only on methods for generating
interpretations but also on techniques for ensuring that interpretations impact
trust and performance in practice.

    

### [[2111.08246] Self-encoding Barnacle Mating Optimizer Algorithm for Manpower Scheduling in Flow Shop](http://arxiv.org/abs/2111.08246)


  Flow Shop Scheduling (FSS) has been widely researched due to its application
in many types of fields, while the human participant brings great challenges to
this problem. Manpower scheduling captures attention for assigning workers with
diverse proficiency to the appropriate stages, which is of great significance
to production efficiency.
In this paper, we present a novel algorithm called Self-encoding Barnacle
Mating Optimizer (SBMO), which solves the FSS problem considering worker
proficiency, defined as a new problem, Flow Shop Manpower Scheduling Problem
(FSMSP). The highlight of the SBMO algorithm is the combination with the
encoding method, crossover and mutation operators. Moreover, in order to solve
the local optimum problem, we design a neighborhood search scheme. Finally, the
extensive comparison simulations are conducted to demonstrate the superiority
of the proposed SBMO. The results indicate the effectiveness of SBMO in
approximate ratio, powerful stability, and execution time, compared with the
classic and popular counterparts.

    

### [[2111.08259] Pose Recognition in the Wild: Animal pose estimation using Agglomerative Clustering and Contrastive Learning](http://arxiv.org/abs/2111.08259)


  Animal pose estimation has recently come into the limelight due to its
application in biology, zoology, and aquaculture. Deep learning methods have
effectively been applied to human pose estimation. However, the major
bottleneck to the application of these methods to animal pose estimation is the
unavailability of sufficient quantities of labeled data. Though there are ample
quantities of unlabelled data publicly available, it is economically
impractical to label large quantities of data for each animal. In addition, due
to the wide variety of body shapes in the animal kingdom, the transfer of
knowledge across domains is ineffective. Given the fact that the human brain is
able to recognize animal pose without requiring large amounts of labeled data,
it is only reasonable that we exploit unsupervised learning to tackle the
problem of animal pose recognition from the available, unlabelled data. In this
paper, we introduce a novel architecture that is able to recognize the pose of
multiple animals fromunlabelled data. We do this by (1) removing background
information from each image and employing an edge detection algorithm on the
body of the animal, (2) Tracking motion of the edge pixels and performing
agglomerative clustering to segment body parts, (3) employing contrastive
learning to discourage grouping of distant body parts together. Hence we are
able to distinguish between body parts of the animal, based on their visual
behavior, instead of the underlying anatomy. Thus, we are able to achieve a
more effective classification of the data than their human-labeled
counterparts. We test our model on the TigDog and WLD (WildLife Documentary)
datasets, where we outperform state-of-the-art approaches by a significant
margin. We also study the performance of our model on other public data to
demonstrate the generalization ability of our model.

    

### [[2111.08299] Accounting for Gaussian Process Imprecision in Bayesian Optimization](http://arxiv.org/abs/2111.08299)


  Bayesian optimization (BO) with Gaussian processes (GP) as surrogate models
is widely used to optimize analytically unknown and expensive-to-evaluate
functions. In this paper, we propose Prior-mean-RObust Bayesian Optimization
(PROBO) that outperforms classical BO on specific problems. First, we study the
effect of the Gaussian processes' prior specifications on classical BO's
convergence. We find the prior's mean parameters to have the highest influence
on convergence among all prior components. In response to this result, we
introduce PROBO as a generalization of BO that aims at rendering the method
more robust towards prior mean parameter misspecification. This is achieved by
explicitly accounting for GP imprecision via a prior near-ignorance model. At
the heart of this is a novel acquisition function, the generalized lower
confidence bound (GLCB). We test our approach against classical BO on a
real-world problem from material science and observe PROBO to converge faster.
Further experiments on multimodal and wiggly target functions confirm the
superiority of our method.

    

### [[2111.08322] An Empirical Study of Finding Similar Exercises](http://arxiv.org/abs/2111.08322)


  Education artificial intelligence aims to profit tasks in the education
domain such as intelligent test paper generation and consolidation exercises
where the main technique behind is how to match the exercises, known as the
finding similar exercises(FSE) problem. Most of these approaches emphasized
their model abilities to represent the exercise, unfortunately there are still
many challenges such as the scarcity of data, insufficient understanding of
exercises and high label noises. We release a Chinese education pre-trained
language model BERT$_{Edu}$ for the label-scarce dataset and introduce the
exercise normalization to overcome the diversity of mathematical formulas and
terms in exercise. We discover new auxiliary tasks in an innovative way depends
on problem-solving ideas and propose a very effective MoE enhanced multi-task
model for FSE task to attain better understanding of exercises. In addition,
confidence learning was utilized to prune train-set and overcome high noises in
labeling data. Experiments show that these methods proposed in this paper are
very effective.

    

### [[2111.08357] A first approach to closeness distributions](http://arxiv.org/abs/2111.08357)


  Probabilistic graphical models allow us to encode a large probability
distribution as a composition of smaller ones. It is oftentimes the case that
we are interested in incorporating in the model the idea that some of these
smaller distributions are likely to be similar to one another. In this paper we
provide an information geometric approach on how to incorporate this
information, and see that it allows us to reinterpret some already existing
models.

    

### [[2111.08361] From Convolutions towards Spikes: The Environmental Metric that the Community currently Misses](http://arxiv.org/abs/2111.08361)


  Today, the AI community is obsessed with 'state-of-the-art' scores (80%
papers in NeurIPS) as the major performance metrics, due to which an important
parameter, i.e., the environmental metric, remains unreported. Computational
capabilities were a limiting factor a decade ago; however, in foreseeable
future circumstances, the challenge will be to develop environment-friendly and
power-efficient algorithms. The human brain, which has been optimizing itself
for almost a million years, consumes the same amount of power as a typical
laptop. Therefore, developing nature-inspired algorithms is one solution to it.
In this study, we show that currently used ANNs are not what we find in nature,
and why, although having lower performance, spiking neural networks, which
mirror the mammalian visual cortex, have attracted much interest. We further
highlight the hardware gaps restricting the researchers from using spike-based
computation for developing neuromorphic energy-efficient microchips on a large
scale. Using neuromorphic processors instead of traditional GPUs might be more
environment friendly and efficient. These processors will turn SNNs into an
ideal solution for the problem. This paper presents in-depth attention
highlighting the current gaps, the lack of comparative research, while
proposing new research directions at the intersection of two fields --
neuroscience and deep learning. Further, we define a new evaluation metric
'NATURE' for reporting the carbon footprint of AI models.

    

### [[2111.08374] Literature-Augmented Clinical Outcome Prediction](http://arxiv.org/abs/2111.08374)


  Predictive models for medical outcomes hold great promise for enhancing
clinical decision-making. These models are trained on rich patient data such as
clinical notes, aggregating many patient signals into an outcome prediction.
However, AI-based clinical models have typically been developed in isolation
from the prominent paradigm of Evidence Based Medicine (EBM), in which medical
decisions are based on explicit evidence from existing literature. In this
work, we introduce techniques to help bridge this gap between EBM and AI-based
clinical models, and show that these methods can improve predictive accuracy.
We propose a novel system that automatically retrieves patient-specific
literature based on intensive care (ICU) patient information, aggregates
relevant papers and fuses them with internal admission notes to form outcome
predictions. Our model is able to substantially boost predictive accuracy on
three challenging tasks in comparison to strong recent baselines; for
in-hospital mortality, we are able to boost top-10% precision by a large margin
of over 25%.

    

### [[2111.08408] STAMP 4 NLP -- An Agile Framework for Rapid Quality-Driven NLP Applications Development](http://arxiv.org/abs/2111.08408)


  The progress in natural language processing (NLP) research over the last
years, offers novel business opportunities for companies, as automated user
interaction or improved data analysis. Building sophisticated NLP applications
requires dealing with modern machine learning (ML) technologies, which impedes
enterprises from establishing successful NLP projects. Our experience in
applied NLP research projects shows that the continuous integration of research
prototypes in production-like environments with quality assurance builds trust
in the software and shows convenience and usefulness regarding the business
goal. We introduce STAMP 4 NLP as an iterative and incremental process model
for developing NLP applications. With STAMP 4 NLP, we merge software
engineering principles with best practices from data science. Instantiating our
process model allows efficiently creating prototypes by utilizing templates,
conventions, and implementations, enabling developers and data scientists to
focus on the business goals. Due to our iterative-incremental approach,
businesses can deploy an enhanced version of the prototype to their software
environment after every iteration, maximizing potential business value and
trust early and avoiding the cost of successful yet never deployed experiments.

    

### [[2111.08468] Point detection through multi-instance deep heatmap regression for sutures in endoscopy](http://arxiv.org/abs/2111.08468)


  Purpose: Mitral valve repair is a complex minimally invasive surgery of the
heart valve. In this context, suture detection from endoscopic images is a
highly relevant task that provides quantitative information to analyse suturing
patterns, assess prosthetic configurations and produce augmented reality
visualisations. Facial or anatomical landmark detection tasks typically contain
a fixed number of landmarks, and use regression or fixed heatmap-based
approaches to localize the landmarks. However in endoscopy, there are a varying
number of sutures in every image, and the sutures may occur at any location in
the annulus, as they are not semantically unique. Method: In this work, we
formulate the suture detection task as a multi-instance deep heatmap regression
problem, to identify entry and exit points of sutures. We extend our previous
work, and introduce the novel use of a 2D Gaussian layer followed by a
differentiable 2D spatial Soft-Argmax layer to function as a local non-maximum
suppression. Results: We present extensive experiments with multiple heatmap
distribution functions and two variants of the proposed model. In the
intra-operative domain, Variant 1 showed a mean F1 of +0.0422 over the
baseline. Similarly, in the simulator domain, Variant 1 showed a mean F1 of
+0.0865 over the baseline. Conclusion: The proposed model shows an improvement
over the baseline in the intra-operative and the simulator domains. The data is
made publicly available within the scope of the MICCAI AdaptOR2021 Challenge
this https URL, and the code at
this https URL.
DOI:10.1007/s11548-021-02523-w. The link to the open access article can be
found here: this https URL


### [[2111.08486] Neural Class Expression Synthesis](http://arxiv.org/abs/2111.08486)


  Class expression learning is a branch of explainable supervised machine
learning of increasing importance. Most existing approaches for class
expression learning in description logics are search algorithms or
hard-rule-based. In particular, approaches based on refinement operators suffer
from scalability issues as they rely on heuristic functions to explore a large
search space for each learning problem. We propose a new family of approaches,
which we dub synthesis approaches. Instances of this family compute class
expressions directly from the examples provided. Consequently, they are not
subject to the runtime limitations of search-based approaches nor the lack of
flexibility of hard-rule-based approaches. We study three instances of this
novel family of approaches that use lightweight neural network architectures to
synthesize class expressions from sets of positive examples. The results of
their evaluation on four benchmark datasets suggest that they can effectively
synthesize high-quality class expressions with respect to the input examples in
under a second on average. Moreover, a comparison with the state-of-the-art
approaches CELOE and ELTL suggests that we achieve significantly better
F-measures on large ontologies. For reproducibility purposes, we provide our
implementation as well as pre-trained models in the public GitHub repository at
this https URL


### [[2111.08500] Patent Data for Engineering Design: A Review](http://arxiv.org/abs/2111.08500)


  Patent data have been utilized for engineering design research for long
because it contains massive amount of design information. Recent advances in
artificial intelligence and data science present unprecedented opportunities to
mine, analyse and make sense of patent data to develop design theory and
methodology. Herein, we survey the patent-for-design literature by their
contributions to design theories, methods, tools, and strategies, as well as
different forms of patent data and various methods. Our review sheds light on
promising future research directions for the field.

    

### [[2111.08529] Improving the robustness and accuracy of biomedical language models through adversarial training](http://arxiv.org/abs/2111.08529)


  Deep transformer neural network models have improved the predictive accuracy
of intelligent text processing systems in the biomedical domain. They have
obtained state-of-the-art performance scores on a wide variety of biomedical
and clinical Natural Language Processing (NLP) benchmarks. However, the
robustness and reliability of these models has been less explored so far.
Neural NLP models can be easily fooled by adversarial samples, i.e. minor
changes to input that preserve the meaning and understandability of the text
but force the NLP system to make erroneous decisions. This raises serious
concerns about the security and trust-worthiness of biomedical NLP systems,
especially when they are intended to be deployed in real-world use cases. We
investigated the robustness of several transformer neural language models, i.e.
BioBERT, SciBERT, BioMed-RoBERTa, and Bio-ClinicalBERT, on a wide range of
biomedical and clinical text processing tasks. We implemented various
adversarial attack methods to test the NLP systems in different attack
scenarios. Experimental results showed that the biomedical NLP models are
sensitive to adversarial samples; their performance dropped in average by 21
and 18.9 absolute percent on character-level and word-level adversarial noise,
respectively. Conducting extensive adversarial training experiments, we
fine-tuned the NLP models on a mixture of clean samples and adversarial inputs.
Results showed that adversarial training is an effective defense mechanism
against adversarial noise; the models robustness improved in average by 11.3
absolute percent. In addition, the models performance on clean data increased
in average by 2.4 absolute present, demonstrating that adversarial training can
boost generalization abilities of biomedical NLP systems.

    

### [[2111.08531] Language bias in Visual Question Answering: A Survey and Taxonomy](http://arxiv.org/abs/2111.08531)


  Visual question answering (VQA) is a challenging task, which has attracted
more and more attention in the field of computer vision and natural language
processing. However, the current visual question answering has the problem of
language bias, which reduces the robustness of the model and has an adverse
impact on the practical application of visual question answering. In this
paper, we conduct a comprehensive review and analysis of this field for the
first time, and classify the existing methods according to three categories,
including enhancing visual information, weakening language priors, data
enhancement and training strategies. At the same time, the relevant
representative methods are introduced, summarized and analyzed in turn. The
causes of language bias are revealed and classified. Secondly, this paper
introduces the datasets mainly used for testing, and reports the experimental
results of various existing methods. Finally, we discuss the possible future
research directions in this field.

    

### [[2111.08557] Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation](http://arxiv.org/abs/2111.08557)


  In keypoint estimation tasks such as human pose estimation, heatmap-based
regression is the dominant approach despite possessing notable drawbacks:
heatmaps intrinsically suffer from quantization error and require excessive
computation to generate and post-process. Motivated to find a more efficient
solution, we propose a new heatmap-free keypoint estimation method in which
individual keypoints and sets of spatially related keypoints (i.e., poses) are
modeled as objects within a dense single-stage anchor-based detection
framework. Hence, we call our method KAPAO (pronounced "Ka-Pow!") for Keypoints
And Poses As Objects. We apply KAPAO to the problem of single-stage
multi-person human pose estimation by simultaneously detecting human pose
objects and keypoint objects and fusing the detections to exploit the strengths
of both object representations. In experiments, we observe that KAPAO is
significantly faster and more accurate than previous methods, which suffer
greatly from heatmap post-processing. Moreover, the accuracy-speed trade-off is
especially favourable in the practical setting when not using test-time
augmentation. Our large model, KAPAO-L, achieves an AP of 70.6 on the Microsoft
COCO Keypoints validation set without test-time augmentation, which is 2.5x
faster and 4.0 AP more accurate than the next best single-stage model.
Furthermore, KAPAO excels in the presence of heavy occlusion. On the CrowdPose
test set, KAPAO-L achieves new state-of-the-art accuracy for a single-stage
method with an AP of 68.9.

    

### [[2111.08564] Doxastic Extensions of Łukasiewicz Logic](http://arxiv.org/abs/2111.08564)


  We propose two new doxastic extensions of fuzzy Łukasiewicz logic in which
their semantics are Kripke-based with both fuzzy atomic propositions and fuzzy
accessibility relations. A class of these extensions is equipped with
uninformed belief operator, and the other class is based on a new notion of
skeptical belief. We model a fuzzy version of muddy children problem and a
CPA-security experiment using uniformed belief and skeptical belief,
respectively. Moreover, we prove soundness and completeness for both of these
belief extensions.

    

### [[2111.08581] Words of Wisdom: Representational Harms in Learning From AI Communication](http://arxiv.org/abs/2111.08581)


  Many educational technologies use artificial intelligence (AI) that presents
generated or produced language to the learner. We contend that all language,
including all AI communication, encodes information about the identity of the
human or humans who contributed to crafting the language. With AI
communication, however, the user may index identity information that does not
match the source. This can lead to representational harms if language
associated with one cultural group is presented as "standard" or "neutral", if
the language advantages one group over another, or if the language reinforces
negative stereotypes. In this work, we discuss a case study using a Visual
Question Generation (VQG) task involving gathering crowdsourced data from
targeted demographic groups. Generated questions will be presented to human
evaluators to understand how they index the identity behind the language,
whether and how they perceive any representational harms, and how they would
ideally address any such harms caused by AI communication. We reflect on the
educational applications of this work as well as the implications for equality,
diversity, and inclusion (EDI).

    

### [[2111.08587] Offline Contextual Bandits for Wireless Network Optimization](http://arxiv.org/abs/2111.08587)


  The explosion in mobile data traffic together with the ever-increasing
expectations for higher quality of service call for the development of AI
algorithms for wireless network optimization. In this paper, we investigate how
to learn policies that can automatically adjust the configuration parameters of
every cell in the network in response to the changes in the user demand. Our
solution combines existent methods for offline learning and adapts them in a
principled way to overcome crucial challenges arising in this context.
Empirical results suggest that our proposed method will achieve important
performance gains when deployed in the real network while satisfying practical
constrains on computational efficiency.

    

### [[2111.08597] A layer-stress learning framework universally augments deep neural network tasks](http://arxiv.org/abs/2111.08597)


  Deep neural networks (DNN) such as Multi-Layer Perception (MLP) and
Convolutional Neural Networks (CNN) represent one of the most established deep
learning algorithms. Given the tremendous effects of the number of hidden
layers on network architecture and performance, it is very important to choose
the number of hidden layers but still a serious challenge. More importantly,
the current network architectures can only process the information from the
last layer of the feature extractor, which greatly limited us to further
improve its performance. Here we presented a layer-stress deep learning
framework (x-NN) which implemented automatic and wise depth decision on shallow
or deep feature map in a deep network through firstly designing enough number
of layers and then trading off them by Multi-Head Attention Block. The x-NN can
make use of features from various depth layers through attention allocation and
then help to make final decision as well. As a result, x-NN showed outstanding
prediction ability in the Alzheimer's Disease Classification Technique
Challenge PRCV 2021, in which it won the top laurel and outperformed all other
AI models. Moreover, the performance of x-NN was verified by one more AD
neuroimaging dataset and other AI tasks.

    

### [[2111.08625] Uncertainty-Aware Multiple Instance Learning fromLarge-Scale Long Time Series Data](http://arxiv.org/abs/2111.08625)


  We propose a novel framework to classify large-scale time series data with
long duration. Long time seriesclassification (L-TSC) is a challenging problem
because the dataoften contains a large amount of irrelevant information to
theclassification target. The irrelevant period degrades the classifica-tion
performance while the relevance is unknown to the system.This paper proposes an
uncertainty-aware multiple instancelearning (MIL) framework to identify the
most relevant periodautomatically. The predictive uncertainty enables designing
anattention mechanism that forces the MIL model to learn from thepossibly
discriminant period. Moreover, the predicted uncertaintyyields a principled
estimator to identify whether a prediction istrustworthy or not. We further
incorporate another modality toaccommodate unreliable predictions by training a
separate modelbased on its availability and conduct uncertainty aware fusion
toproduce the final prediction. Systematic evaluation is conductedon the
Automatic Identification System (AIS) data, which is col-lected to identify and
track real-world vessels. Empirical resultsdemonstrate that the proposed method
can effectively detect thetypes of vessels based on the trajectory and the
uncertainty-awarefusion with other available data modality
(Synthetic-ApertureRadar or SAR imagery is used in our experiments) can
furtherimprove the detection accuracy.

    

### [[2010.01909] Deliberative Acting, Online Planning and Learning with Hierarchical Operational Models](http://arxiv.org/abs/2010.01909)


  In AI research, synthesizing a plan of action has typically used descriptive
models of the actions that abstractly specify what might happen as a result of
an action, and are tailored for efficiently computing state transitions.
However, executing the planned actions has needed operational models, in which
rich computational control structures and closed-loop online decision-making
are used to specify how to perform an action in a nondeterministic execution
context, react to events and adapt to an unfolding situation. Deliberative
actors, which integrate acting and planning, have typically needed to use both
of these models together -- which causes problems when attempting to develop
the different models, verify their consistency, and smoothly interleave acting
and planning.
As an alternative, we define and implement an integrated acting and planning
system in which both planning and acting use the same operational models. These
rely on hierarchical task-oriented refinement methods offering rich control
structures. The acting component, called Reactive Acting Engine (RAE), is
inspired by the well-known PRS system. At each decision step, RAE can get
advice from a planner for a near-optimal choice with respect to a utility
function. The anytime planner uses a UCT-like Monte Carlo Tree Search
procedure, called UPOM, whose rollouts are simulations of the actor's
operational models. We also present learning strategies for use with RAE and
UPOM that acquire, from online acting experiences and/or simulated planning
results, a mapping from decision contexts to method instances as well as a
heuristic function to guide UPOM. We demonstrate the asymptotic convergence of
UPOM towards optimal methods in static domains, and show experimentally that
UPOM and the learning strategies significantly improve the acting efficiency
and robustness.

    

### [[2011.11311] Uncovering the Bias in Facial Expressions](http://arxiv.org/abs/2011.11311)


  Over the past decades the machine and deep learning community has celebrated
great achievements in challenging tasks such as image classification. The deep
architecture of artificial neural networks together with the plenitude of
available data makes it possible to describe highly complex relations. Yet, it
is still impossible to fully capture what the deep learning model has learned
and to verify that it operates fairly and without creating bias, especially in
critical tasks, for instance those arising in the medical field. One example
for such a task is the detection of distinct facial expressions, called Action
Units, in facial images. Considering this specific task, our research aims to
provide transparency regarding bias, specifically in relation to gender and
skin color. We train a neural network for Action Unit classification and
analyze its performance quantitatively based on its accuracy and qualitatively
based on heatmaps. A structured review of our results indicates that we are
able to detect bias. Even though we cannot conclude from our results that lower
classification performance emerged solely from gender and skin color bias,
these biases must be addressed, which is why we end by giving suggestions on
how the detected bias can be avoided.

    

### [[2012.06157] Fairness in Rating Prediction by Awareness of Verbal and Gesture Quality of Public Speeches](http://arxiv.org/abs/2012.06157)


  The role of verbal and non-verbal cues towards great public speaking has been
a topic of exploration for many decades. We identify a commonality across
present theories, the element of "variety or heterogeneity" in channels or
modes of communication (e.g. resorting to stories, scientific facts, emotional
connections, facial expressions etc.) which is essential for effectively
communicating information. We use this observation to formalize a novel
HEterogeneity Metric, HEM, that quantifies the quality of a talk both in the
verbal and non-verbal domain (transcript and facial gestures). We use TED talks
as an input repository of public speeches because it consists of speakers from
a diverse community besides having a wide outreach. We show that there is an
interesting relationship between HEM and the ratings of TED talks given to
speakers by viewers. It emphasizes that HEM inherently and successfully
represents the quality of a talk based on "variety or heterogeneity". Further,
we also discover that HEM successfully captures the prevalent bias in ratings
with respect to race and gender, that we call sensitive attributes (because
prediction based on these might result in unfair outcome). We incorporate the
HEM metric into the loss function of a neural network with the goal to reduce
unfairness in rating predictions with respect to race and gender. Our results
show that the modified loss function improves fairness in prediction without
considerably affecting prediction accuracy of the neural network. Our work ties
together a novel metric for public speeches in both verbal and non-verbal
domain with the computational power of a neural network to design a fair
prediction system for speakers.

    

### [[2101.11952] Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss](http://arxiv.org/abs/2101.11952)


  Boundary discontinuity and its inconsistency to the final detection metric
have been the bottleneck for rotating detection regression loss design. In this
paper, we propose a novel regression loss based on Gaussian Wasserstein
distance as a fundamental approach to solve the problem. Specifically, the
rotated bounding box is converted to a 2-D Gaussian distribution, which enables
to approximate the indifferentiable rotational IoU induced loss by the Gaussian
Wasserstein distance (GWD) which can be learned efficiently by gradient
back-propagation. GWD can still be informative for learning even there is no
overlapping between two rotating bounding boxes which is often the case for
small object detection. Thanks to its three unique properties, GWD can also
elegantly solve the boundary discontinuity and square-like problem regardless
how the bounding box is defined. Experiments on five datasets using different
detectors show the effectiveness of our approach. Codes are available at
this https URL.

    

### [[2103.02362] Video Sentiment Analysis with Bimodal Information-augmented Multi-Head Attention](http://arxiv.org/abs/2103.02362)


  Humans express feelings or emotions via different channels. Take language as
an example, it entails different sentiments under different visual-acoustic
contexts. To precisely understand human intentions as well as reduce the
misunderstandings caused by ambiguity and sarcasm, we should consider
multimodal signals including textual, visual and acoustic signals. The crucial
challenge is to fuse different modalities of features for sentiment analysis.
To effectively fuse the information carried by different modalities and better
predict the sentiments, we design a novel multi-head attention based fusion
network, which is inspired by the observations that the interactions between
any two pair-wise modalities are different and they do not equally contribute
to the final sentiment prediction. By assigning the acoustic-visual,
acoustic-textual and visual-textual features with reasonable attention and
exploiting a residual structure, we attend to attain the significant features.
We conduct extensive experiments on four public multimodal datasets including
one in Chinese and three in English. The results show that our approach
outperforms the existing methods and can explain the contributions of bimodal
interaction in multiple modalities.

    

### [[2105.03641] Diversifying Neural Text Generation with Part-of-Speech Guided Softmax and Sampling](http://arxiv.org/abs/2105.03641)


  Neural text generation models are likely to suffer from the low-diversity
problem. Various decoding strategies and training-based methods have been
proposed to promote diversity only by exploiting contextual features, but
rarely do they consider incorporating syntactic structure clues. In this work,
we propose using linguistic annotation, i.e., part-of-speech (POS), to guide
the text generation. In detail, we introduce POS Guided Softmax to explicitly
model two posterior probabilities: (i) next-POS, and (ii) next-token from the
vocabulary of the target POS. A POS Guided Sampling strategy is further
proposed to address the low-diversity problem by enriching the diversity of
POS. Extensive experiments and human evaluations demonstrate that, compared
with existing state-of-the-art methods, our POS Guided Softmax and Sampling
(POSG) can generate more diverse text while maintaining comparable quality.

    

### [[2111.08099] Moebius: Metaprogramming using Contextual Types -- The stage where System F can pattern match on itself (Long Version)](http://arxiv.org/abs/2111.08099)


  We describe the foundation of the metaprogramming language, Moebius, which
supports the generation of polymorphic code and, more importantly the analysis
of polymorphic code via pattern matching.
Moebius has two main ingredients: 1) we exploit contextual modal types to
describe open code together with the context in which it is meaningful. In
Moebius, open code can depend on type and term variables (level 0) whose values
are supplied at a later stage, as well as code variables (level 1) that stand
for code templates supplied at a later stage. This leads to a multi-level modal
lambda-calculus that supports System-F style polymorphism and forms the basis
for polymorphic code generation. 2) we extend the multi-level modal
lambda-calculus to support pattern matching on code. As pattern matching on
polymorphic code may refine polymorphic type variables, we extend our
type-theoretic foundation to generate and track typing constraints that arise.
We also give an operational semantics and prove type preservation.
Our multi-level modal foundation for Moebius provides the appropriate
abstractions for both generating and pattern matching on open code without
committing to a concrete representation of variable binding and contexts.
Hence, our work is a step towards building a general type-theoretic foundation
for multi-staged metaprogramming that, on the one hand, enforces strong type
guarantees and, on the other hand, makes it easy to generate and manipulate
code. This will allow us to exploit the full potential of metaprogramming
without sacrificing the reliability of and trust in the code we are producing
and running.

    

### [[2111.08692] Unicode at Gigabytes per Second](http://arxiv.org/abs/2111.08692)


  We often represent text using Unicode formats (UTF-8 and UTF-16). The UTF-8
format is increasingly popular, especially on the web (XML, HTML, JSON, Rust,
Go, Swift, Ruby). The UTF-16 format is most common in Java, .NET, and inside
operating systems such as Windows.
Software systems frequently have to convert text from one Unicode format to
the other. While recent disks have bandwidths of 5 GiB/s or more, conventional
approaches transcode non-ASCII text at a fraction of a gigabyte per second.
We show that we can validate and transcode Unicode text at gigabytes per
second on current systems (x64 and ARM) without sacrificing safety. Our
open-source library can be ten times faster than the popular ICU library on
non-ASCII strings and even faster on ASCII strings.

    

### [<title>Prometheus中创建记录规则 - DockOne.io</title>](http://dockone.io/question/1593251)

### [<title>Debugging a weird 'file not found' error</title>](https://jvns.ca/blog/2021/11/17/debugging-a-weird--file-not-found--error/)