
## 2021-9-20

### [[2109.08343] Acila: Attaching Identities of Workloads for Efficient Packet Classification in a Cloud Data Center Network](http://arxiv.org/abs/2109.08343)


  IP addresses and port numbers (network based identifiers hereafter) in
packets are two major identifiers for network devices to identify systems and
roles of hosts sending and receiving packets for access control lists, priority
control, etc. However, in modern system design on cloud, such as microservices
architecture, network based identifiers are inefficient for network devices to
identify systems and roles of hosts. This is because, due to autoscaling and
automatic deployment of new software, many VMs and containers consisting of the
system (workload hereafter) are frequently created and deleted on servers whose
resources are available, and network based identifiers are assigned based on
servers where containers and VMs are running. In this paper, we propose a new
system, Acila, to classify packets based on the identity of a workload at
network devices, by marking packets with the necessary information extracted
from the identity that usually stored in orchestrators or controllers. We then
implement Acila and show that packet filtering and priority control can be
implemented with Acila, and entries for them with Acila is more efficient than
conventional network based identifiers approach, with little overhead on
performance

    

### [[2109.08389] Coordinated Random Access for Industrial IoT With Correlated Traffic By Reinforcement-Learning](http://arxiv.org/abs/2109.08389)


  We propose a coordinated random access scheme for industrial
internet-of-things (IIoT) scenarios, with machine-type devices (MTDs)
generating sporadic correlated traffic. This occurs, e.g., when external events
trigger data generation at multiple MTDs simultaneously. Time is divided into
frames, each split into slots and each MTD randomly selects one slot for
(re)transmission, with probability density functions (PDFs) specific of both
the MTD and the number of the current retransmission. PDFs are locally
optimized to minimize the probability of packet collision. The optimization
problem is modeled as a repeated Markov game with incomplete information, and
the linear reward-inaction algorithm is used at each MTD, which provably
converges to a deterministic (suboptimal) slot assignment. We compare our
solution with both the slotted ALOHA and the min-max pairwise correlation
random access schemes, showing that our approach achieves a higher network
throughput with moderate traffic intensity.

    

### [[2109.08429] OTFS-superimposed PRACH-aided Localization for UAV Safety Applications](http://arxiv.org/abs/2109.08429)


  The adoption of Unmanned Aerial Vehicles (UAVs) for public safety
applications has skyrocketed in the last years. Leveraging on Physical Random
Access Channel (PRACH) preambles, in this paper we pioneer a novel localization
technique for UAVs equipped with cellular base stations used in emergency
scenarios. We exploit the new concept of Orthogonal Time Frequency Space (OTFS)
modulation (tolerant to channel Doppler spread caused by UAVs motion) to build
a fully standards-compliant OTFS-modulated PRACH transmission and reception
scheme able to perform time-of-arrival (ToA) measurements. First, we analyze
such novel ToA ranging technique, both analytically and numerically, to
accurately and iteratively derive the distance between localized users and the
points traversed by the UAV along its trajectory. Then, we determine the
optimal UAV speed as a trade-off between the accuracy of the ranging technique
and the power needed by the UAV to reach and keep its speed during emergency
operations. Finally, we demonstrate that our solution outperforms standard
PRACH-based localization techniques in terms of Root Mean Square Error (RMSE)
by about 20% in quasi-static conditions and up to 80% in high-mobility
conditions.

    

### [[2109.08446] Heterogeneous download times in bandwidth-homogeneous BitTorrent swarms](http://arxiv.org/abs/2109.08446)


  Modeling and understanding BitTorrent (BT) dynamics is a recurrent research
topic mainly due to its high complexity and tremendous practical efficiency.
Over the years, different models have uncovered various phenomena exhibited by
the system, many of which have direct impact on its performance. In this paper
we identify and characterize a phenomenon that has not been previously
observed: homogeneous peers (with respect to their upload capacities)
experience heterogeneous download times. This behavior has direct impact on
peer and system performance, such as high variability of download times,
unfairness with respect to peer arrival order, bursty departures and content
synchronization. Detailed packet-level simulations and prototype-based
experiments on the Internet were performed to characterize this phenomenon. We
also develop a mathematical model that accurately predicts the heterogeneous
download rates of the homogeneous peers as a function of their content. In
addition, we apply the model to calculate lower and upper bounds to the number
of departures that occur in a burst. The heterogeneous download rates are more
prevalent in unpopular swarms (very few peers). Although few works have
addressed this kind of swarm, these by far represent the most common type of
swarm in BT.

    

### [[2109.08482] An Optimization-based Approach for Flow Table Capacity Bottleneck Mitigation in Software-Defined Networks](http://arxiv.org/abs/2109.08482)


  Flow delegation is a flexible technique to mitigate flow table capacity
bottlenecks in Software-defined Networks (SDN). Such bottlenecks occur when SDN
switches provide insufficient flow table capacity which leads to performance
degradation and network failures. Flow delegation addresses this problem by
automatically relocating flow rules from a bottlenecked switch to neighboring
switches with spare capacity. This paper introduces a new algorithm to
efficiently perform flow delegation based on a novel delegation template
abstraction and multi-period multi-objective optimization. Different from
existing work, our approach can include estimated knowledge about future
network situations and deal with different optimization criteria such as link
and control overhead. We discuss the problem decomposition for the new
algorithm and introduce an efficient two-step heuristic. Results show, that our
approach performs significantly better than the simple greedy algorithm used in
earlier work and is capable of handling flow delegation for networks with
hundreds of switches.

    

### [[2109.08532] PAPIR: Practical RIS-aided Localization via Statistical User Information](http://arxiv.org/abs/2109.08532)


  The integration of advanced localization techniques in the upcoming next
generation networks (B5G/6G) is becoming increasingly important for many use
cases comprising contact tracing, natural disasters, terrorist attacks, etc.
Therefore, emerging lightweight and passive technologies that allow accurately
controlling the propagation environment, such as reconfigurable intelligent
surfaces (RISs), may help to develop advance positioning solutions relying on
channel statistics and beamforming. In this paper, we devise PAPIR, a practical
localization system leveraging on RISs by designing a two-stage solution
building upon prior statistical information on the target user equipment (UE)
position. PAPIR aims at finely estimating the UE position by performing
statistical beamforming, direction-of-arrival (DoA) and time-of-arrival (ToA)
estimation on a given three-dimensional search space, which is iteratively
updated by exploiting the likelihood of the UE position.

    

### [[2109.08651] A Tutorial on Mathematical Modeling of Millimeter Wave and Terahertz Cellular Systems](http://arxiv.org/abs/2109.08651)


  Millimeter wave (mmWave) and terahertz (THz) radio access technologies (RAT)
are expected to become a critical part of the future cellular ecosystem
providing an abundant amount of bandwidth in areas with high traffic demands.
However, extremely directional antenna radiation patterns that need to be
utilized at both transmit and receive sides of a link to overcome severe path
losses, dynamic blockage of propagation paths by large static and small dynamic
objects, macro- and micromobility of user equipment (UE) makes provisioning of
reliable service over THz/mmWave RATs an extremely complex task. This challenge
is further complicated by the type of applications envisioned for these systems
inherently requiring guaranteed bitrates at the air interface. This tutorial
aims to introduce a versatile mathematical methodology for assessing
performance reliability improvement algorithms for mmWave and THz systems. Our
methodology accounts for both radio interface specifics as well as service
process of sessions at mmWave/THz base stations (BS) and is capable of
evaluating the performance of systems with multiconnectivity operation,
resource reservation mechanisms, priorities between multiple traffic types
having different service requirements. The framework is logically separated
into two parts: (i) parameterization part that abstracts the specifics of
deployment and radio mechanisms, and (ii) queuing part, accounting for details
of the service process at mmWave/THz BSs. The modular decoupled structure of
the framework allows for further extensions to advanced service mechanisms in
prospective mmWave/THz cellular deployments while keeping the complexity
manageable and thus making it attractive for system analysts.

    

### [[2109.08669] Version Age of Information in Clustered Gossip Networks](http://arxiv.org/abs/2109.08669)


  We consider a network consisting of a single source and $n$ receiver nodes
that are grouped into equal-sized clusters. We use cluster heads in each
cluster to facilitate communication between the source and the nodes within
that cluster. Inside clusters, nodes are connected to each other according to a
given network topology. Based on the connectivity among the nodes, each node
relays its current stored version of the source update to its neighboring nodes
by $local$ $gossiping$. We use the $version$ $age$ metric to assess information
freshness at the nodes. We consider disconnected, ring, and fully connected
network topologies for each cluster. For each network topology, we characterize
the average version age at each node and find the average version age scaling
as a function of the network size $n$. Our results indicate that per node
average version age scalings of $O(\sqrt{n})$, $O(n^{\frac{1}{3}})$, and
$O(\log n)$ are achievable in disconnected, ring, and fully connected cluster
models, respectively. Next, we increase connectivity in the network and allow
gossiping among the cluster heads to improve version age at the nodes. With
that, we show that when the cluster heads form a ring network among themselves,
we obtain per node average version age scalings of $O(n^{\frac{1}{3}})$,
$O(n^{\frac{1}{4}})$, and $O(\log n)$ in disconnected, ring, and fully
connected cluster models, respectively. Next, focusing on a ring network
topology in each cluster, we introduce hierarchy to the considered clustered
gossip network model and show that when we employ two levels of hierarchy, we
can achieve the same $O(n^{\frac{1}{4}})$ scaling without using dedicated
cluster heads. We generalize this result for $h$ levels of hierarchy and show
that per user average version age scaling of $O(n^{\frac{1}{2h}})$ is
achievable in the case of a ring network in each cluster across all hierarchy
levels.

    

### [[2101.05462] Leader Confirmation Replication for Millisecond Consensus in Private Chains](http://arxiv.org/abs/2101.05462)


  The private chain-based Internet of Things (IoT) system ensures the security
of cross-organizational data sharing. As a widely used consensus model in
private chains, the leader-based state-machine replication (SMR) model meets
the performance bottleneck in IoT blockchain applications, where
nontransactional sensor data are generated on a scale. We analyzed IoT private
chain systems and found that the leader maintains too many connections due to
high latency and client request frequency, which results in lower consensus
performance and efficiency.
To meet this challenge, we propose a novel solution for maintaining low
request latency and high transactions per second (TPS): replicate
nontransactional data by followers and confirm by the leader to achieve
nonconfliction SMR, rather than all by the leader. Our solution, named Leader
Confirmation Replication (LCR), uses the newly proposed future log and
confirmation signal to achieve nontransactional data replication on the
followers, thereby reducing the leader's network traffic and the request
latency of transactional data. In addition, the generation replication strategy
is designed to ensure the reliability and consistency of LCR when meeting
membership changes. We evaluated LCR with various cluster sizes and network
latencies. Experimental results show that in ms-network latency (2-30)
environments, the TPS of LCR is 1.4X-1.9X higher than Raft, the transactional
data response time is reduced by 40%-60%, and the network traffic is reduced by
20%-30% with acceptable network traffic and CPU cost on the followers. In
addition, LCR shows high portability and availability since it does not change
the number of leaders or the election process.

    

### [[2101.12691] Isolation mechanisms for high-speed packet-processing pipelines](http://arxiv.org/abs/2101.12691)


  Data-plane programmability is now mainstream. As we find more use cases,
deployments need to be able to run multiple packet-processing modules in a
single device. These are likely to be developed by independent teams, either
within the same organization or from multiple organizations. Therefore, we need
isolation mechanisms to ensure that modules on the same device do not interfere
with each other. This paper presents Menshen, an extension of the RMT/PISA
pipeline that enforces isolation between different packet-processing modules.
Menshen is comprised of a set of lightweight hardware primitives and an
extension to the open source P4-16 reference compiler that act in conjunction
to meet this goal. We have prototyped Menshen on two FPGA platforms (NetFPGA
and Corundum). We show that our design provides isolation, and allows new
modules to be loaded without impacting the ones already running. Finally, we
demonstrate the feasibility of implementing our design on ASICs by using the
FreePDK45nm technology library and the Synopsys DC synthesis software, showing
that our design meets timing at a 1 GHz clock frequency and needs approximately
6% additional chip area. We have open sourced the code for Menshen's hardware
and software at this https URL.

    

### [[2109.08159] A Cautionary Tale of Decorrelating Theory Uncertainties](http://arxiv.org/abs/2109.08159)


  A variety of techniques have been proposed to train machine learning
classifiers that are independent of a given feature. While this can be an
essential technique for enabling background estimation, it may also be useful
for reducing uncertainties. We carefully examine theory uncertainties, which
typically do not have a statistical origin. We will provide explicit examples
of two-point (fragmentation modeling) and continuous (higher-order corrections)
uncertainties where decorrelating significantly reduces the apparent
uncertainty while the actual uncertainty is much larger. These results suggest
that caution should be taken when using decorrelation for these types of
uncertainties as long as we do not have a complete decomposition into
statistically meaningful components.

    

### [[2109.08180] Interpretable Local Tree Surrogate Policies](http://arxiv.org/abs/2109.08180)


  High-dimensional policies, such as those represented by neural networks,
cannot be reasonably interpreted by humans. This lack of interpretability
reduces the trust users have in policy behavior, limiting their use to
low-impact tasks such as video games. Unfortunately, many methods rely on
neural network representations for effective learning. In this work, we propose
a method to build predictable policy trees as surrogates for policies such as
neural networks. The policy trees are easily human interpretable and provide
quantitative predictions of future behavior. We demonstrate the performance of
this approach on several simulated tasks.

    

### [[2109.08184] Sparse Factorization of Large Square Matrices](http://arxiv.org/abs/2109.08184)


  Square matrices appear in many machine learning problems and models.
Optimization over a large square matrix is expensive in memory and in time.
Therefore an economic approximation is needed. Conventional approximation
approaches factorize the square matrix into a number matrices of much lower
ranks. However, the low-rank constraint is a performance bottleneck if the
approximated matrix is intrinsically high-rank or close to full rank. In this
paper, we propose to approximate a large square matrix with a product of sparse
full-rank matrices. In the approximation, our method needs only $N(\log N)^2$
non-zero numbers for an $N\times N$ full matrix. We present both non-parametric
and parametric ways to find the factorization. In the former, we learn the
factorizing matrices directly, and in the latter, we train neural networks to
map input data to the non-zero matrix entries. The sparse factorization method
is tested for a variety of synthetic and real-world square matrices. The
experimental results demonstrate that our method gives a better approximation
when the approximated matrix is sparse and high-rank. Based on this finding, we
use our parametric method as a scalable attention architecture that performs
strongly in learning tasks for long sequential data and defeats Transformer and
its several variants.

    

### [[2109.08191] KATANA: Simple Post-Training Robustness Using Test Time Augmentations](http://arxiv.org/abs/2109.08191)


  Although Deep Neural Networks (DNNs) achieve excellent performance on many
real-world tasks, they are highly vulnerable to adversarial attacks. A leading
defense against such attacks is adversarial training, a technique in which a
DNN is trained to be robust to adversarial attacks by introducing adversarial
noise to its input. This procedure is effective but must be done during the
training phase. In this work, we propose a new simple and easy-to-use
technique, KATANA, for robustifying an existing pretrained DNN without
modifying its weights. For every image, we generate N randomized Test Time
Augmentations (TTAs) by applying diverse color, blur, noise, and geometric
transforms. Next, we utilize the DNN's logits output to train a simple random
forest classifier to predict the real class label. Our strategy achieves
state-of-the-art adversarial robustness on diverse attacks with minimal
compromise on the natural images' classification. We test KATANA also against
two adaptive white-box attacks and it shows excellent results when combined
with adversarial training. Code is available in
this https URL.

    

### [[2109.08213] Improving Regression Uncertainty Estimation Under Statistical Change](http://arxiv.org/abs/2109.08213)


  While deep neural networks are highly performant and successful in a wide
range of real-world problems, estimating their predictive uncertainty remains a
challenging task. To address this challenge, we propose and implement a loss
function for regression uncertainty estimation based on the Bayesian Validation
Metric (BVM) framework while using ensemble learning. A series of experiments
on in-distribution data show that the proposed method is competitive with
existing state-of-the-art methods. In addition, experiments on
out-of-distribution data show that the proposed method is robust to statistical
change and exhibits superior predictive capability.

    

### [[2109.08215] Automatic prior selection for meta Bayesian optimization with a case study on tuning deep neural network optimizers](http://arxiv.org/abs/2109.08215)


  The performance of deep neural networks can be highly sensitive to the choice
of a variety of meta-parameters, such as optimizer parameters and model
hyperparameters. Tuning these well, however, often requires extensive and
costly experimentation. Bayesian optimization (BO) is a principled approach to
solve such expensive hyperparameter tuning problems efficiently. Key to the
performance of BO is specifying and refining a distribution over functions,
which is used to reason about the optima of the underlying function being
optimized. In this work, we consider the scenario where we have data from
similar functions that allows us to specify a tighter distribution a priori.
Specifically, we focus on the common but potentially costly task of tuning
optimizer parameters for training neural networks. Building on the meta BO
method from Wang et al. (2018), we develop practical improvements that (a)
boost its performance by leveraging tuning results on multiple tasks without
requiring observations for the same meta-parameter points across all tasks, and
(b) retain its regret bound for a special case of our method. As a result, we
provide a coherent BO solution for iterative optimization of continuous
optimizer parameters. To verify our approach in realistic model training
setups, we collected a large multi-task hyperparameter tuning dataset by
training tens of thousands of configurations of near-state-of-the-art models on
popular image and text datasets, as well as a protein sequence dataset. Our
results show that on average, our method is able to locate good hyperparameters
at least 3 times more efficiently than the best competing methods.

    

### [[2109.08216] Beyond Average Performance -- exploring regions of deviating performance for black box classification models](http://arxiv.org/abs/2109.08216)


  Machine learning models are becoming increasingly popular in different types
of settings. This is mainly caused by their ability to achieve a level of
predictive performance that is hard to match by human experts in this new era
of big data. With this usage growth comes an increase of the requirements for
accountability and understanding of the models' predictions. However, the
degree of sophistication of the most successful models (e.g. ensembles, deep
learning) is becoming a large obstacle to this endeavour as these models are
essentially black boxes. In this paper we describe two general approaches that
can be used to provide interpretable descriptions of the expected performance
of any black box classification model. These approaches are of high practical
relevance as they provide means to uncover and describe in an interpretable way
situations where the models are expected to have a performance that deviates
significantly from their average behaviour. This may be of critical relevance
for applications where costly decisions are driven by the predictions of the
models, as it can be used to warn end users against the usage of the models in
some specific cases.

    

### [[2109.08218] SLAW: Scaled Loss Approximate Weighting for Efficient Multi-Task Learning](http://arxiv.org/abs/2109.08218)


  Multi-task learning (MTL) is a subfield of machine learning with important
applications, but the multi-objective nature of optimization in MTL leads to
difficulties in balancing training between tasks. The best MTL optimization
methods require individually computing the gradient of each task's loss
function, which impedes scalability to a large number of tasks. In this paper,
we propose Scaled Loss Approximate Weighting (SLAW), a method for multi-task
optimization that matches the performance of the best existing methods while
being much more efficient. SLAW balances learning between tasks by estimating
the magnitudes of each task's gradient without performing any extra backward
passes. We provide theoretical and empirical justification for SLAW's
estimation of gradient magnitudes. Experimental results on non-linear
regression, multi-task computer vision, and virtual screening for drug
discovery demonstrate that SLAW is significantly more efficient than strong
baselines without sacrificing performance and applicable to a diverse range of
domains.

    

### [[2109.08227] Stereo Video Reconstruction Without Explicit Depth Maps for Endoscopic Surgery](http://arxiv.org/abs/2109.08227)


  We introduce the task of stereo video reconstruction or, equivalently,
2D-to-3D video conversion for minimally invasive surgical video. We design and
implement a series of end-to-end U-Net-based solutions for this task by varying
the input (single frame vs. multiple consecutive frames), loss function (MSE,
MAE, or perceptual losses), and network architecture. We evaluate these
solutions by surveying ten experts - surgeons who routinely perform endoscopic
surgery. We run two separate reader studies: one evaluating individual frames
and the other evaluating fully reconstructed 3D video played on a VR headset.
In the first reader study, a variant of the U-Net that takes as input multiple
consecutive video frames and outputs the missing view performs best. We draw
two conclusions from this outcome. First, motion information coming from
multiple past frames is crucial in recreating stereo vision. Second, the
proposed U-Net variant can indeed exploit such motion information for solving
this task. The result from the second study further confirms the effectiveness
of the proposed U-Net variant. The surgeons reported that they could
successfully perceive depth from the reconstructed 3D video clips. They also
expressed a clear preference for the reconstructed 3D video over the original
2D video. These two reader studies strongly support the usefulness of the
proposed task of stereo reconstruction for minimally invasive surgical video
and indicate that deep learning is a promising approach to this task. Finally,
we identify two automatic metrics, LPIPS and DISTS, that are strongly
correlated with expert judgement and that could serve as proxies for the latter
in future studies.

    

### [[2109.08229] Policy Choice and Best Arm Identification: Comments on "Adaptive Treatment Assignment in Experiments for Policy Choice"](http://arxiv.org/abs/2109.08229)


  The purpose of this paper is to connect the "policy choice" problem, proposed
in Kasy and Sautmann (2021), to the frontiers of the bandit literature in
machine learning. We discuss how the policy choice problem can be framed in a
way such that it is identical to what is called the "best arm identification"
(BAI) problem. By connecting the literature, we identify that the asymptotic
optimality of policy choice algorithms tackled in Kasy and Sautmann (2021) is a
long-standing open question in the literature. Unfortunately, this connection
highlights several major issues with the main theorem. In particular, we show
that Theorem 1 in Kasy and Sautmann (2021) is false. We find that the proofs of
statements (1) and (2) of Theorem 1 are incorrect, though the statements
themselves may be true, though non-trivial to fix. Statement (3), and its
proof, on the other hand, is false, which we show by utilizing existing
theoretical results in the bandit literature. As this question is critically
important, garnering much interest in the last decade within the bandit
community, we provide a review of recent developments in the BAI literature. We
hope this serves to highlight the relevance to economic problems and stimulate
methodological and theoretical developments in the econometric community.

    

### [[2109.08231] RAPID-RL: A Reconfigurable Architecture with Preemptive-Exits for Efficient Deep-Reinforcement Learning](http://arxiv.org/abs/2109.08231)


  Present-day Deep Reinforcement Learning (RL) systems show great promise
towards building intelligent agents surpassing human-level performance.
However, the computational complexity associated with the underlying deep
neural networks (DNNs) leads to power-hungry implementations. This makes deep
RL systems unsuitable for deployment on resource-constrained edge devices. To
address this challenge, we propose a reconfigurable architecture with
preemptive exits for efficient deep RL (RAPID-RL). RAPID-RL enables conditional
activation of DNN layers based on the difficulty level of inputs. This allows
to dynamically adjust the compute effort during inference while maintaining
competitive performance. We achieve this by augmenting a deep Q-network (DQN)
with side-branches capable of generating intermediate predictions along with an
associated confidence score. We also propose a novel training methodology for
learning the actions and branch confidence scores in a dynamic RL setting. Our
experiments evaluate the proposed framework for Atari 2600 gaming tasks and a
realistic Drone navigation task on an open-source drone simulator (PEDRA). We
show that RAPID-RL incurs 0.34x (0.25x) number of operations (OPS) while
maintaining performance above 0.88x (0.91x) on Atari (Drone navigation) tasks,
compared to a baseline-DQN without any side-branches. The reduction in OPS
leads to fast and efficient inference, proving to be highly beneficial for the
resource-constrained edge where making quick decisions with minimal compute is
essential.

    

### [[2109.08234] Deep Spiking Neural Networks with Resonate-and-Fire Neurons](http://arxiv.org/abs/2109.08234)


  In this work, we explore a new Spiking Neural Network (SNN) formulation with
Resonate-and-Fire (RAF) neurons (Izhikevich, 2001) trained with gradient
descent via back-propagation. The RAF-SNN, while more biologically plausible,
achieves performance comparable to or higher than conventional models in the
Machine Learning literature across different network configurations, using
similar or fewer parameters. Strikingly, the RAF-SNN proves robust against
noise induced at testing/training time, under both static and dynamic
conditions. Against CNN on MNIST, we show 25% higher absolute accuracy with
N(0, 0.2) induced noise at testing time. Against LSTM on N-MNIST, we show 70%
higher absolute accuracy with 20% induced noise at training time.

    

### [[2109.08236] Reinforcement Learning on Encrypted Data](http://arxiv.org/abs/2109.08236)


  The growing number of applications of Reinforcement Learning (RL) in
real-world domains has led to the development of privacy-preserving techniques
due to the inherently sensitive nature of data. Most existing works focus on
differential privacy, in which information is revealed in the clear to an agent
whose learned model should be robust against information leakage to malicious
third parties. Motivated by use cases in which only encrypted data might be
shared, such as information from sensitive sites, in this work we consider
scenarios in which the inputs themselves are sensitive and cannot be revealed.
We develop a simple extension to the MDP framework which provides for the
encryption of states. We present a preliminary, experimental study of how a DQN
agent trained on encrypted states performs in environments with discrete and
continuous state spaces. Our results highlight that the agent is still capable
of learning in small state spaces even in presence of non-deterministic
encryption, but performance collapses in more complex environments.

    

### [[2109.08237] Subtle Inverse Crimes: Naïvely training machine learning algorithms could lead to overly-optimistic results](http://arxiv.org/abs/2109.08237)


  While open databases are an important resource in the Deep Learning (DL) era,
they are sometimes used "off-label": data published for one task are used for
training algorithms for a different one. This work aims to highlight that in
some cases, this common practice may lead to biased, overly-optimistic results.
We demonstrate this phenomenon for inverse problem solvers and show how their
biased performance stems from hidden data preprocessing pipelines. We describe
two preprocessing pipelines typical of open-access databases and study their
effects on three well-established algorithms developed for Magnetic Resonance
Imaging (MRI) reconstruction: Compressed Sensing (CS), Dictionary Learning
(DictL), and DL. In this large-scale study we performed extensive computations.
Our results demonstrate that the CS, DictL and DL algorithms yield
systematically biased results when naïvely trained on seemingly-appropriate
data: the Normalized Root Mean Square Error (NRMSE) improves consistently with
the preprocessing extent, showing an artificial increase of 25%-48% in some
cases. Since this phenomenon is generally unknown, biased results are sometimes
published as state-of-the-art; we refer to that as subtle inverse crimes. This
work hence raises a red flag regarding naïve off-label usage of Big Data and
reveals the vulnerability of modern inverse problem solvers to the resulting
bias.

    

### [[2109.08240] Strategic Ranking](http://arxiv.org/abs/2109.08240)


  Strategic classification studies the design of a classifier robust to the
manipulation of input by strategic individuals. However, the existing
literature does not consider the effect of competition among individuals as
induced by the algorithm design. Motivated by constrained allocation settings
such as college admissions, we introduce strategic ranking, in which the
(designed) individual reward depends on an applicant's post-effort rank in a
measurement of interest. Our results illustrate how competition among
applicants affects the resulting equilibria and model insights. We analyze how
various ranking reward designs trade off applicant, school, and societal
utility and in particular how ranking design can counter inequities arising
from disparate access to resources to improve one's measured score: We find
that randomization in the ranking reward design can mitigate two measures of
disparate impact, welfare gap and access, whereas non-randomization may induce
a high level of competition that systematically excludes a disadvantaged group.

    

### [[2109.08247] Towards agricultural autonomy: crop row detection under varying field conditions using deep learning](http://arxiv.org/abs/2109.08247)


  This paper presents a novel metric to evaluate the robustness of deep
learning based semantic segmentation approaches for crop row detection under
different field conditions encountered by a field robot. A dataset with ten
main categories encountered under various field conditions was used for
testing. The effect on these conditions on the angular accuracy of crop row
detection was compared. A deep convolutional encoder decoder network is
implemented to predict crop row masks using RGB input images. The predicted
mask is then sent to a post processing algorithm to extract the crop rows. The
deep learning model was found to be robust against shadows and growth stages of
the crop while the performance was reduced under direct sunlight, increasing
weed density, tramlines and discontinuities in crop rows when evaluated with
the novel metric.

    

### [[2109.08248] Assessments of model-form uncertainty using Gaussian stochastic weight averaging for fluid-flow regression](http://arxiv.org/abs/2109.08248)


  We use Gaussian stochastic weight averaging (SWAG) to assess the model-form
uncertainty associated with neural-network-based function approximation
relevant to fluid flows. SWAG approximates a posterior Gaussian distribution of
each weight, given training data, and a constant learning rate. Having access
to this distribution, it is able to create multiple models with various
combinations of sampled weights, which can be used to obtain ensemble
predictions. The average of such an ensemble can be regarded as the `mean
estimation', whereas its standard deviation can be used to construct
`confidence intervals', which enable us to perform uncertainty quantification
(UQ) with regard to the training process of neural networks. We utilize
representative neural-network-based function approximation tasks for the
following cases: (i) a two-dimensional circular-cylinder wake; (ii) the DayMET
dataset (maximum daily temperature in North America); (iii) a three-dimensional
square-cylinder wake; and (iv) urban flow, to assess the generalizability of
the present idea for a wide range of complex datasets. SWAG-based UQ can be
applied regardless of the network architecture, and therefore, we demonstrate
the applicability of the method for two types of neural networks: (i) global
field reconstruction from sparse sensors by combining convolutional neural
network (CNN) and multi-layer perceptron (MLP); and (ii) far-field state
estimation from sectional data with two-dimensional CNN. We find that SWAG can
obtain physically-interpretable confidence-interval estimates from the
perspective of model-form uncertainty. This capability supports its use for a
wide range of problems in science and engineering.

    

### [[2109.08266] Hard to Forget: Poisoning Attacks on Certified Machine Unlearning](http://arxiv.org/abs/2109.08266)


  The right to erasure requires removal of a user's information from data held
by organizations, with rigorous interpretations extending to downstream
products such as learned models. Retraining from scratch with the particular
user's data omitted fully removes its influence on the resulting model, but
comes with a high computational cost. Machine "unlearning" mitigates the cost
incurred by full retraining: instead, models are updated incrementally,
possibly only requiring retraining when approximation errors accumulate. Rapid
progress has been made towards privacy guarantees on the indistinguishability
of unlearned and retrained models, but current formalisms do not place
practical bounds on computation. In this paper we demonstrate how an attacker
can exploit this oversight, highlighting a novel attack surface introduced by
machine unlearning. We consider an attacker aiming to increase the
computational cost of data removal. We derive and empirically investigate a
poisoning attack on certified machine unlearning where strategically designed
training data triggers complete retraining when removed.

    

### [[2109.08267] CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research](http://arxiv.org/abs/2109.08267)


  Interest in applying Artificial Intelligence (AI) techniques to compiler
optimizations is increasing rapidly, but compiler research has a high entry
barrier. Unlike in other domains, compiler and AI researchers do not have
access to the datasets and frameworks that enable fast iteration and
development of ideas, and getting started requires a significant engineering
investment. What is needed is an easy, reusable experimental infrastructure for
real world compiler optimization tasks that can serve as a common benchmark for
comparing techniques, and as a platform to accelerate progress in the field.
We introduce CompilerGym, a set of environments for real world compiler
optimization tasks, and a toolkit for exposing new optimization tasks to
compiler researchers. CompilerGym enables anyone to experiment on production
compiler optimization problems through an easy-to-use package, regardless of
their experience with compilers. We build upon the popular OpenAI Gym interface
enabling researchers to interact with compilers using Python and a familiar
API.
We describe the CompilerGym architecture and implementation, characterize the
optimization spaces and computational efficiencies of three included compiler
environments, and provide extensive empirical evaluations. Compared to prior
works, CompilerGym offers larger datasets and optimization spaces, is 27x more
computationally efficient, is fault-tolerant, and capable of detecting
reproducibility bugs in the underlying compilers.
In making it easy for anyone to experiment with compilers - irrespective of
their background - we aim to accelerate progress in the AI and compiler
research domains.

    

### [[2109.08275] Multi-Level Visual Similarity Based Personalized Tourist Attraction Recommendation Using Geo-Tagged Photos](http://arxiv.org/abs/2109.08275)


  Geo-tagged photo based tourist attraction recommendation can discover users'
travel preferences from their taken photos, so as to recommend suitable tourist
attractions to them. However, existing visual content based methods cannot
fully exploit the user and tourist attraction information of photos to extract
visual features, and do not differentiate the significances of different
photos. In this paper, we propose multi-level visual similarity based
personalized tourist attraction recommendation using geo-tagged photos (MEAL).
MEAL utilizes the visual contents of photos and interaction behavior data to
obtain the final embeddings of users and tourist attractions, which are then
used to predict the visit probabilities. Specifically, by crossing the user and
tourist attraction information of photos, we define four visual similarity
levels and introduce a corresponding quintuplet loss to embed the visual
contents of photos. In addition, to capture the significances of different
photos, we exploit the self-attention mechanism to obtain the visual
representations of users and tourist attractions. We conducted experiments on a
dataset crawled from Flickr, and the experimental results proved the advantage
of this method.

    

### [[2109.08282] AdaLoss: A computationally-efficient and provably convergent adaptive gradient method](http://arxiv.org/abs/2109.08282)


  We propose a computationally-friendly adaptive learning rate schedule,
"AdaLoss", which directly uses the information of the loss function to adjust
the stepsize in gradient descent methods. We prove that this schedule enjoys
linear convergence in linear regression. Moreover, we provide a linear
convergence guarantee over the non-convex regime, in the context of two-layer
over-parameterized neural networks. If the width of the first-hidden layer in
the two-layer networks is sufficiently large (polynomially), then AdaLoss
converges robustly \emph{to the global minimum} in polynomial time. We
numerically verify the theoretical results and extend the scope of the
numerical experiments by considering applications in LSTM models for text
clarification and policy gradients for control problems.

    

### [[2109.08291] Natlog: a Lightweight Logic Programming Language with a Neuro-symbolic Touch](http://arxiv.org/abs/2109.08291)


  We introduce Natlog, a lightweight Logic Programming language, sharing
Prolog's unification-driven execution model, but with a simplified syntax and
semantics. Our proof-of-concept Natlog implementation is tightly embedded in
the Python-based deep-learning ecosystem with focus on content-driven indexing
of ground term datasets. As an overriding of our symbolic indexing algorithm,
the same function can be delegated to a neural network, serving ground facts to
Natlog's resolution engine. Our open-source implementation is available as a
Python package at this https URL .

    

### [[2109.08294] A Logic-based Multi-agent System for Ethical Monitoring and Evaluation of Dialogues](http://arxiv.org/abs/2109.08294)


  Dialogue Systems are tools designed for various practical purposes concerning
human-machine interaction. These systems should be built on ethical foundations
because their behavior may heavily influence a user (think especially about
children). The primary objective of this paper is to present the architecture
and prototype implementation of a Multi Agent System (MAS) designed for ethical
monitoring and evaluation of a dialogue system. A prototype application, for
monitoring and evaluation of chatting agents' (human/artificial) ethical
behavior in an online customer service chat point w.r.t their
institution/company's codes of ethics and conduct, is developed and presented.
Future work and open issues with this research are discussed.

    

### [[2109.08311] Adaptive Hierarchical Dual Consistency for Semi-Supervised Left Atrium Segmentation on Cross-Domain Data](http://arxiv.org/abs/2109.08311)


  Semi-supervised learning provides great significance in left atrium (LA)
segmentation model learning with insufficient labelled data. Generalising
semi-supervised learning to cross-domain data is of high importance to further
improve model robustness. However, the widely existing distribution difference
and sample mismatch between different data domains hinder the generalisation of
semi-supervised learning. In this study, we alleviate these problems by
proposing an Adaptive Hierarchical Dual Consistency (AHDC) for the
semi-supervised LA segmentation on cross-domain data. The AHDC mainly consists
of a Bidirectional Adversarial Inference module (BAI) and a Hierarchical Dual
Consistency learning module (HDC). The BAI overcomes the difference of
distributions and the sample mismatch between two different domains. It mainly
learns two mapping networks adversarially to obtain two matched domains through
mutual adaptation. The HDC investigates a hierarchical dual learning paradigm
for cross-domain semi-supervised segmentation based on the obtained matched
domains. It mainly builds two dual-modelling networks for mining the
complementary information in both intra-domain and inter-domain. For the
intra-domain learning, a consistency constraint is applied to the
dual-modelling targets to exploit the complementary modelling information. For
the inter-domain learning, a consistency constraint is applied to the LAs
modelled by two dual-modelling networks to exploit the complementary knowledge
among different data domains. We demonstrated the performance of our proposed
AHDC on four 3D late gadolinium enhancement cardiac MR (LGE-CMR) datasets from
different centres and a 3D CT dataset. Compared to other state-of-the-art
methods, our proposed AHDC achieved higher segmentation accuracy, which
indicated its capability in the cross-domain semi-supervised LA segmentation.

    

### [[2109.08325] Decision Tree Learning with Spatial Modal Logics](http://arxiv.org/abs/2109.08325)


  Symbolic learning represents the most straightforward approach to
interpretable modeling, but its applications have been hampered by a single
structural design choice: the adoption of propositional logic as the underlying
language. Recently, more-than-propositional symbolic learning methods have
started to appear, in particular for time-dependent data. These methods exploit
the expressive power of modal temporal logics in powerful learning algorithms,
such as temporal decision trees, whose classification capabilities are
comparable with the best non-symbolic ones, while producing models with
explicit knowledge representation.
With the intent of following the same approach in the case of spatial data,
in this paper we: i) present a theory of spatial decision tree learning; ii)
describe a prototypical implementation of a spatial decision tree learning
algorithm based, and strictly extending, the classical C4.5 algorithm; and iii)
perform a series of experiments in which we compare the predicting power of
spatial decision trees with that of classical propositional decision trees in
several versions, for a multi-class image classification problem, on publicly
available datasets. Our results are encouraging, showing clear improvements in
the performances from the propositional to the spatial models, which in turn
show higher levels of interpretability.

    

### [[2109.08331] Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation](http://arxiv.org/abs/2109.08331)


  In recommender systems (RecSys) and real-time bidding (RTB) for online
advertisements, we often try to optimize sequential decision making using
bandit and reinforcement learning (RL) techniques. In these applications,
offline reinforcement learning (offline RL) and off-policy evaluation (OPE) are
beneficial because they enable safe policy optimization using only logged data
without any risky online interaction. In this position paper, we explore the
potential of using simulation to accelerate practical research of offline RL
and OPE, particularly in RecSys and RTB. Specifically, we discuss how
simulation can help us conduct empirical research of offline RL and OPE. We
take a position to argue that we should effectively use simulations in the
empirical research of offline RL and OPE. To refute the counterclaim that
experiments using only real-world data are preferable, we first point out the
underlying risks and reproducibility issue in real-world experiments. Then, we
describe how these issues can be addressed by using simulations. Moreover, we
show how to incorporate the benefits of both real-world and simulation-based
experiments to defend our position. Finally, we also present an open challenge
to further facilitate practical research of offline RL and OPE in RecSys and
RTB, with respect to public simulation platforms. As a possible solution for
the issue, we show our ongoing open source project and its potential use case.
We believe that building and utilizing simulation-based evaluation platforms
for offline RL and OPE will be of great interest and relevance for the RecSys
and RTB community.

    

### [[2109.08342] Dropout's Dream Land: Generalization from Learned Simulators to Reality](http://arxiv.org/abs/2109.08342)


  A World Model is a generative model used to simulate an environment. World
Models have proven capable of learning spatial and temporal representations of
Reinforcement Learning environments. In some cases, a World Model offers an
agent the opportunity to learn entirely inside of its own dream environment. In
this work we explore improving the generalization capabilities from dream
environments to real environments (Dream2Real). We present a general approach
to improve a controller's ability to transfer from a neural network dream
environment to reality at little additional cost. These improvements are gained
by drawing on inspiration from Domain Randomization, where the basic idea is to
randomize as much of a simulator as possible without fundamentally changing the
task at hand. Generally, Domain Randomization assumes access to a pre-built
simulator with configurable parameters but oftentimes this is not available. By
training the World Model using dropout, the dream environment is capable of
creating a nearly infinite number of different dream environments. Previous use
cases of dropout either do not use dropout at inference time or averages the
predictions generated by multiple sampled masks (Monte-Carlo Dropout).
Dropout's Dream Land leverages each unique mask to create a diverse set of
dream environments. Our experimental results show that Dropout's Dream Land is
an effective technique to bridge the reality gap between dream environments and
reality. Furthermore, we additionally perform an extensive set of ablation
studies.

    

### [[2109.08344] Achieving Model Fairness in Vertical Federated Learning](http://arxiv.org/abs/2109.08344)


  Vertical federated learning (VFL), which enables multiple enterprises
possessing non-overlapped features to strengthen their machine learning models
without disclosing their private data and model parameters, has received
increasing attention lately. Similar to other machine learning algorithms, VFL
suffers from fairness issues, i.e., the learned model may be unfairly
discriminatory over the group with sensitive attributes. To tackle this
problem, we propose a fair VFL framework in this work. First, we systematically
formulate the problem of training fair models in VFL, where the learning task
is modeled as a constrained optimization problem. To solve it in a federated
manner, we consider its equivalent dual form and develop an asynchronous
gradient coordinate-descent ascent algorithm, where each data party performs
multiple parallelized local updates per communication round to effectively
reduce the number of communication rounds. We prove that the algorithm finds a
$\delta$-stationary point of the dual objective in $\mathcal{O}(\delta^{-4})$
communication rounds under mild conditions. Finally, extensive experiments on
three benchmark datasets demonstrate the superior performance of our method in
training fair models.

    

### [[2109.08345] Learning Enhanced Optimisation for Routing Problems](http://arxiv.org/abs/2109.08345)


  Deep learning approaches have shown promising results in solving routing
problems. However, there is still a substantial gap in solution quality between
machine learning and operations research algorithms. Recently, another line of
research has been introduced that fuses the strengths of machine learning and
operational research algorithms. In particular, search perturbation operators
have been used to improve the solution. Nevertheless, using the perturbation
may not guarantee a quality solution. This paper presents "Learning to Guide
Local Search" (L2GLS), a learning-based approach for routing problems that uses
a penalty term and reinforcement learning to adaptively adjust search efforts.
L2GLS combines local search (LS) operators' strengths with penalty terms to
escape local optimals. Routing problems have many practical applications, often
presetting larger instances that are still challenging for many existing
algorithms introduced in the learning to optimise field. We show that L2GLS
achieves the new state-of-the-art results on larger TSP and CVRP over other
machine learning methods.

    

### [[2109.08346] Comfetch: Federated Learning of Large Networks on Memory-Constrained Clients via Sketching](http://arxiv.org/abs/2109.08346)


  A popular application of federated learning is using many clients to train a
deep neural network, the parameters of which are maintained on a central
server. While recent efforts have focused on reducing communication complexity,
existing algorithms assume that each participating client is able to download
the current and full set of parameters, which may not be a practical assumption
depending on the memory constraints of clients such as mobile devices. In this
work, we propose a novel algorithm Comfetch, which allows clients to train
large networks using compressed versions of the global architecture via Count
Sketch, thereby reducing communication and local memory costs. We provide a
theoretical convergence guarantee and experimentally demonstrate that it is
possible to learn large networks, such as a deep convolutional network and an
LSTM, through federated agents training on their sketched counterparts. The
resulting global models exhibit competitive test accuracy when compared against
the state-of-the-art FetchSGD and the classical FedAvg, both of which require
clients to download the full architecture.

    

### [[2109.08356] Accurate, Interpretable, and Fast Animation: AnIterative, Sparse, and Nonconvex Approach](http://arxiv.org/abs/2109.08356)


  Digital human animation relies on high-quality 3D models of the human face:
rigs. A face rig must be accurate and, at the same time, fast to compute. One
of the most common rigging models is the blendshape model. We propose a novel
algorithm for solving the nonconvex inverse rig problem in facial animation.
Our approach is model-based, but in contrast with previous model-based
approaches, we use a quadratic instead of the linear approximation to the
higher order rig model. This increases the accuracy of the solution by 8
percent on average and, confirmed by the empirical results, increases the
sparsity of the resulting parameter vector -- an important feature for
interpretability by animation artists. The proposed solution is based on a
Levenberg-Marquardt (LM) algorithm, applied to a nonconvex constrained problem
with sparsity regularization. In order to reduce the complexity of the
iterates, a paradigm of Majorization Minimization (MM) is further invoked,
which leads to an easy to solve problem that is separable in the parameters at
each algorithm iteration. The algorithm is evaluated on a number of animation
datasets, proprietary and open-source, and the results indicate the superiority
of our method compared to the standard approach based on the linear rig
approximation. Although our algorithm targets the specific problem, it might
have additional signal processing applications.

    

### [[2109.08357] Dynamic Spatiotemporal Graph Convolutional Neural Networks for Traffic Data Imputation with Complex Missing Patterns](http://arxiv.org/abs/2109.08357)


  Missing data is an inevitable and ubiquitous problem for traffic data
collection in intelligent transportation systems. Despite extensive research
regarding traffic data imputation, there still exist two limitations to be
addressed: first, existing approaches fail to capture the complex
spatiotemporal dependencies in traffic data, especially the dynamic spatial
dependencies evolving with time; second, prior studies mainly focus on randomly
missing patterns while other more complex missing scenarios are less discussed.
To fill these research gaps, we propose a novel deep learning framework called
Dynamic Spatiotemporal Graph Convolutional Neural Networks (DSTGCN) to impute
missing traffic data. The model combines the recurrent architecture with
graph-based convolutions to model the spatiotemporal dependencies. Moreover, we
introduce a graph structure estimation technique to model the dynamic spatial
dependencies from real-time traffic information and road network structure.
Extensive experiments based on two public traffic speed datasets are conducted
to compare our proposed model with state-of-the-art deep learning approaches in
four types of missing patterns. The results show that our proposed model
outperforms existing deep learning models in all kinds of missing scenarios and
the graph structure estimation technique contributes to the model performance.
We further compare our proposed model with a tensor factorization model and
find distinct behaviors across different model families under different
training schemes and data availability.

    

### [[2109.08359] Distilling Linguistic Context for Language Model Compression](http://arxiv.org/abs/2109.08359)


  A computationally expensive and memory intensive neural network lies behind
the recent success of language representation learning. Knowledge distillation,
a major technique for deploying such a vast language model in resource-scarce
environments, transfers the knowledge on individual word representations
learned without restrictions. In this paper, inspired by the recent
observations that language representations are relatively positioned and have
more semantic knowledge as a whole, we present a new knowledge distillation
objective for language representation learning that transfers the contextual
knowledge via two types of relationships across representations: Word Relation
and Layer Transforming Relation. Unlike other recent distillation techniques
for the language models, our contextual distillation does not have any
restrictions on architectural changes between teacher and student. We validate
the effectiveness of our method on challenging benchmarks of language
understanding tasks, not only in architectures of various sizes, but also in
combination with DynaBERT, the recently proposed adaptive size pruning method.

    

### [[2109.08360] An Interpretable Framework for Drug-Target Interaction with Gated Cross Attention](http://arxiv.org/abs/2109.08360)


  In silico prediction of drug-target interactions (DTI) is significant for
drug discovery because it can largely reduce timelines and costs in the drug
development process. Specifically, deep learning-based DTI approaches have been
shown promising results in terms of accuracy and low cost for the prediction.
However, they pay little attention to the interpretability of their prediction
results and feature-level interactions between a drug and a target. In this
study, we propose a novel interpretable framework that can provide reasonable
cues for the interaction sites. To this end, we elaborately design a gated
cross-attention mechanism that crossly attends drug and target features by
constructing explicit interactions between these features. The gating function
in the method enables neural models to focus on salient regions over entire
sequences of drugs and proteins, and the byproduct from the function, which is
the attention map, could serve as interpretable factors. The experimental
results show the efficacy of the proposed method in two DTI datasets.
Additionally, we show that gated cross-attention can sensitively react to the
mutation, and this result could provide insights into the identification of
novel drugs targeting mutant proteins.

    

### [[2109.08381] From Known to Unknown: Knowledge-guided Transformer for Time-Series Sales Forecasting in Alibaba](http://arxiv.org/abs/2109.08381)


  Time series forecasting (TSF) is fundamentally required in many real-world
applications, such as electricity consumption planning and sales forecasting.
In e-commerce, accurate time-series sales forecasting (TSSF) can significantly
increase economic benefits. TSSF in e-commerce aims to predict future sales of
millions of products. The trend and seasonality of products vary a lot, and the
promotion activity heavily influences sales. Besides the above difficulties, we
can know some future knowledge in advance except for the historical statistics.
Such future knowledge may reflect the influence of the future promotion
activity on current sales and help achieve better accuracy. However, most
existing TSF methods only predict the future based on historical information.
In this work, we make up for the omissions of future knowledge. Except for
introducing future knowledge for prediction, we propose Aliformer based on the
bidirectional Transformer, which can utilize the historical information,
current factor, and future knowledge to predict future sales. Specifically, we
design a knowledge-guided self-attention layer that uses known knowledge's
consistency to guide the transmission of timing information. And the
future-emphasized training strategy is proposed to make the model focus more on
the utilization of future knowledge. Extensive experiments on four public
benchmark datasets and one proposed large-scale industrial dataset from Tmall
demonstrate that Aliformer can perform much better than state-of-the-art TSF
methods. Aliformer has been deployed for goods selection on Tmall Industry
Tablework, and the dataset will be released upon approval.

    

### [[2109.08438] TS-MULE: Local Interpretable Model-Agnostic Explanations for Time Series Forecast Models](http://arxiv.org/abs/2109.08438)


  Time series forecasting is a demanding task ranging from weather to failure
forecasting with black-box models achieving state-of-the-art performances.
However, understanding and debugging are not guaranteed. We propose TS-MULE, a
local surrogate model explanation method specialized for time series extending
the LIME approach. Our extended LIME works with various ways to segment and
perturb the time series data. In our extension, we present six sampling
segmentation approaches for time series to improve the quality of surrogate
attributions and demonstrate their performances on three deep learning model
architectures and three common multivariate time series datasets.

    

### [[2109.08449] New Students on Sesame Street: What Order-Aware Matrix Embeddings Can Learn from BERT](http://arxiv.org/abs/2109.08449)


  Large-scale pretrained language models (PreLMs) are revolutionizing natural
language processing across all benchmarks. However, their sheer size is
prohibitive in low-resource or large-scale applications. While common
approaches reduce the size of PreLMs via same-architecture distillation or
pruning, we explore distilling PreLMs into more efficient order-aware embedding
models. Our results on the GLUE benchmark show that embedding-centric students,
which have learned from BERT, yield scores comparable to DistilBERT on QQP and
RTE, often match or exceed the scores of ELMo, and only fall behind on
detecting linguistic acceptability.

    

### [[2109.08467] Online Learning of Network Bottlenecks via Minimax Paths](http://arxiv.org/abs/2109.08467)


  In this paper, we study bottleneck identification in networks via extracting
minimax paths. Many real-world networks have stochastic weights for which full
knowledge is not available in advance. Therefore, we model this task as a
combinatorial semi-bandit problem to which we apply a combinatorial version of
Thompson Sampling and establish an upper bound on the corresponding Bayesian
regret. Due to the computational intractability of the problem, we then devise
an alternative problem formulation which approximates the original objective.
Finally, we experimentally evaluate the performance of Thompson Sampling with
the approximate formulation on real-world directed and undirected networks.

    

### [[2109.08490] Integrating Deep Reinforcement and Supervised Learning to Expedite Indoor Mapping](http://arxiv.org/abs/2109.08490)


  The challenge of mapping indoor environments is addressed. Typical heuristic
algorithms for solving the motion planning problem are frontier-based methods,
that are especially effective when the environment is completely unknown.
However, in cases where prior statistical data on the environment's
architectonic features is available, such algorithms can be far from optimal.
Furthermore, their calculation time may increase substantially as more areas
are exposed. In this paper we propose two means by which to overcome these
shortcomings. One is the use of deep reinforcement learning to train the motion
planner. The second is the inclusion of a pre-trained generative deep neural
network, acting as a map predictor. Each one helps to improve the decision
making through use of the learned structural statistics of the environment, and
both, being realized as neural networks, ensure a constant calculation time. We
show that combining the two methods can shorten the mapping time, compared to
frontier-based motion planning, by up to 75%.

    

### [[2109.08495] Micro-architectural Analysis of a Learned Index](http://arxiv.org/abs/2109.08495)


  Since the publication of The Case for Learned Index Structures in 2018, there
has been a rise in research that focuses on learned indexes for different
domains and with different functionalities. While the effectiveness of learned
indexes as an alternative to traditional index structures such as B+Trees have
already been demonstrated by several studies, previous work tend to focus on
higher-level performance metrics such as throughput and index size. In this
paper, our goal is to dig deeper and investigate how learned indexes behave at
a micro-architectural level compared to traditional indexes.
More specifically, we focus on previously proposed learned index structure
ALEX, which is a tree-based in-memory index structure that consists of a
hierarchy of machine learned models. Unlike the original proposal for learned
indexes, ALEX is designed from the ground up to allow updates and inserts.
Therefore, it enables more dynamic workloads using learned indexes. In this
work, we perform a micro-architectural analysis of ALEX and compare its
behavior to the tree-based index structures that are not based on learned
models, i.e., ART and B+Tree.
Our results show that ALEX is bound by memory stalls, mainly stalls due to
data misses from the last-level cache. Compared to ART and B+Tree, ALEX
exhibits fewer stalls and a lower cycles-per-instruction value across different
workloads. On the other hand, the amount of instructions required to handle
out-of-bound inserts in ALEX can increase the instructions needed per request
significantly (10X) for write-heavy workloads. However, the micro-architectural
behavior shows that this increase in the instruction footprint exhibit high
instruction-level parallelism, and, therefore, does not negatively impact the
overall execution time.

    

### [[2109.08501] SaCoFa: Semantics-aware Control-flow Anonymization for Process Mining](http://arxiv.org/abs/2109.08501)


  Privacy-preserving process mining enables the analysis of business processes
using event logs, while giving guarantees on the protection of sensitive
information on process stakeholders. To this end, existing approaches add noise
to the results of queries that extract properties of an event log, such as the
frequency distribution of trace variants, for analysis.Noise insertion neglects
the semantics of the process, though, and may generate traces not present in
the original log. This is problematic. It lowers the utility of the published
data and makes noise easily identifiable, as some traces will violate
well-known semantic this http URL this paper, we therefore argue for privacy
preservation that incorporates a process semantics. For common trace-variant
queries, we show how, based on the exponential mechanism, semantic constraints
are incorporated to ensure differential privacy of the query result.
Experiments demonstrate that our semantics-aware anonymization yields event
logs of significantly higher utility than existing approaches.

    

### [[2109.08512] Soft Actor-Critic With Integer Actions](http://arxiv.org/abs/2109.08512)


  Reinforcement learning is well-studied under discrete actions. Integer
actions setting is popular in the industry yet still challenging due to its
high dimensionality. To this end, we study reinforcement learning under integer
actions by incorporating the Soft Actor-Critic (SAC) algorithm with an integer
reparameterization. Our key observation for integer actions is that their
discrete structure can be simplified using their comparability property. Hence,
the proposed integer reparameterization does not need one-hot encoding and is
of low dimensionality. Experiments show that the proposed SAC under integer
actions is as good as the continuous action version on robot control tasks and
outperforms Proximal Policy Optimization on power distribution systems control
tasks.

    

### [[2109.08518] Knowledge is reward: Learning optimal exploration by predictive reward cashing](http://arxiv.org/abs/2109.08518)


  There is a strong link between the general concept of intelligence and the
ability to collect and use information. The theory of Bayes-adaptive
exploration offers an attractive optimality framework for training machines to
perform complex information gathering tasks. However, the computational
complexity of the resulting optimal control problem has limited the diffusion
of the theory to mainstream deep AI research. In this paper we exploit the
inherent mathematical structure of Bayes-adaptive problems in order to
dramatically simplify the problem by making the reward structure denser while
simultaneously decoupling the learning of exploitation and exploration
policies. The key to this simplification comes from the novel concept of
cross-value (i.e. the value of being in an environment while acting optimally
according to another), which we use to quantify the value of currently
available information. This results in a new denser reward structure that
"cashes in" all future rewards that can be predicted from the current
information state. In a set of experiments we show that the approach makes it
possible to learn challenging information gathering tasks without the use of
shaping and heuristic bonuses in situations where the standard RL algorithms
fail.

    

### [[2109.08522] Dynamics-Aware Quality-Diversity for Efficient Learning of Skill Repertoires](http://arxiv.org/abs/2109.08522)


  Quality-Diversity (QD) algorithms are powerful exploration algorithms that
allow robots to discover large repertoires of diverse and high-performing
skills. However, QD algorithms are sample inefficient and require millions of
evaluations. In this paper, we propose Dynamics-Aware Quality-Diversity
(DA-QD), a framework to improve the sample efficiency of QD algorithms through
the use of dynamics models. We also show how DA-QD can then be used for
continual acquisition of new skill repertoires. To do so, we incrementally
train a deep dynamics model from experience obtained when performing skill
discovery using QD. We can then perform QD exploration in imagination with an
imagined skill repertoire. We evaluate our approach on three robotic
experiments. First, our experiments show DA-QD is 20 times more sample
efficient than existing QD approaches for skill discovery. Second, we
demonstrate learning an entirely new skill repertoire in imagination to perform
zero-shot learning. Finally, we show how DA-QD is useful and effective for
solving a long horizon navigation task and for damage adaptation in the real
world. Videos and source code are available at:
this https URL.

    

### [[2109.08527] An open GPS trajectory dataset and benchmark for travel mode detection](http://arxiv.org/abs/2109.08527)


  Travel mode detection has been a hot topic in the field of GPS
trajectory-related processing. Former scholars have developed many mathematical
methods to improve the accuracy of detection. Among these studies, almost all
of the methods require ground truth dataset for training. A large amount of the
studies choose to collect the GPS trajectory dataset for training by their
customized ways. Currently, there is no open GPS dataset marked with travel
mode. If there exists one, it will not only save a lot of efforts in model
developing, but also help compare the performance of models. In this study, we
propose and open GPS trajectory dataset marked with travel mode and benchmark
for the travel mode detection. The dataset is collected by 7 independent
volunteers in Japan and covers the time period of a complete month. The travel
mode ranges from walking to railway. A part of routines are traveled repeatedly
in different time slots to experience different road and travel conditions. We
also provide a case study to distinguish the walking and bike trips in a
massive GPS trajectory dataset.

    

### [[2109.08536] Decentralized Global Connectivity Maintenance for Multi-Robot Navigation: A Reinforcement Learning Approach](http://arxiv.org/abs/2109.08536)


  The problem of multi-robot navigation of connectivity maintenance is
challenging in multi-robot applications. This work investigates how to navigate
a multi-robot team in unknown environments while maintaining connectivity. We
propose a reinforcement learning (RL) approach to develop a decentralized
policy, which is shared among multiple robots. Given range sensor measurements
and the positions of other robots, the policy aims to generate control commands
for navigation and preserve the global connectivity of the robot team. We
incorporate connectivity concerns into the RL framework as constraints and
introduce behavior cloning to reduce the exploration complexity of policy
optimization. The policy is optimized with all transition data collected by
multiple robots in random simulated scenarios. We validate the effectiveness of
the proposed approach by comparing different combinations of connectivity
constraints and behavior cloning. We also show that our policy can generalize
to unseen scenarios in both simulation and holonomic robots experiments.

    

### [[2109.08544] Conversational Multi-Hop Reasoning with Neural Commonsense Knowledge and Symbolic Logic Rules](http://arxiv.org/abs/2109.08544)


  One of the challenges faced by conversational agents is their inability to
identify unstated presumptions of their users' commands, a task trivial for
humans due to their common sense. In this paper, we propose a zero-shot
commonsense reasoning system for conversational agents in an attempt to achieve
this. Our reasoner uncovers unstated presumptions from user commands satisfying
a general template of if-(state), then-(action), because-(goal). Our reasoner
uses a state-of-the-art transformer-based generative commonsense knowledge base
(KB) as its source of background knowledge for reasoning. We propose a novel
and iterative knowledge query mechanism to extract multi-hop reasoning chains
from the neural KB which uses symbolic logic rules to significantly reduce the
search space. Similar to any KBs gathered to date, our commonsense KB is prone
to missing knowledge. Therefore, we propose to conversationally elicit the
missing knowledge from human users with our novel dynamic question generation
strategy, which generates and presents contextualized queries to human users.
We evaluate the model with a user study with human users that achieves a 35%
higher success rate compared to SOTA.

    

### [[2109.08548] Scheduling in Parallel Finite Buffer Systems: Optimal Decisions under Delayed Feedback](http://arxiv.org/abs/2109.08548)


  Scheduling decisions in parallel queuing systems arise as a fundamental
problem, underlying the dimensioning and operation of many computing and
communication systems, such as job routing in data center clusters, multipath
communication, and Big Data systems. In essence, the scheduler maps each
arriving job to one of the possibly heterogeneous servers while aiming at an
optimization goal such as load balancing, low average delay or low loss rate.
One main difficulty in finding optimal scheduling decisions here is that the
scheduler only partially observes the impact of its decisions, e.g., through
the delayed acknowledgements of the served jobs. In this paper, we provide a
partially observable (PO) model that captures the scheduling decisions in
parallel queuing systems under limited information of delayed acknowledgements.
We present a simulation model for this PO system to find a near-optimal
scheduling policy in real-time using a scalable Monte Carlo tree search
algorithm. We numerically show that the resulting policy outperforms other
limited information scheduling strategies such as variants of
Join-the-Most-Observations and has comparable performance to full information
strategies like: Join-the-Shortest-Queue, Join-the- Shortest-Queue(d) and
Shortest-Expected-Delay. Finally, we show how our approach can optimise the
real-time parallel processing by using network data provided by Kaggle.

    

### [[2109.08549] Measuring Fairness under Unawareness via Quantification](http://arxiv.org/abs/2109.08549)


  Models trained by means of supervised learning are increasingly deployed in
high-stakes domains, and, when their predictions inform decisions about people,
they inevitably impact (positively or negatively) on their lives. As a
consequence, those in charge of developing these models must carefully evaluate
their impact on different groups of people and ensure that sensitive
demographic attributes, such as race or sex, do not result in unfair treatment
for members of specific groups. For doing this, awareness of demographic
attributes on the part of those evaluating model impacts is fundamental.
Unfortunately, the collection of these attributes is often in conflict with
industry practices and legislation on data minimization and privacy. For this
reason, it may be hard to measure the group fairness of trained models, even
from within the companies developing them. In this work, we tackle the problem
of measuring group fairness under unawareness of sensitive attributes, by using
techniques from quantification, a supervised learning task concerned with
directly providing group-level prevalence estimates (rather than
individual-level class labels). We identify five important factors that
complicate the estimation of fairness under unawareness and formalize them into
five different experimental protocols under which we assess the effectiveness
of different estimators of group fairness. We also consider the problem of
potential model misuse to infer sensitive attributes at an individual level,
and demonstrate that quantification approaches are suitable for decoupling the
(desirable) objective of measuring group fairness from the (undesirable)
objective of inferring sensitive attributes of individuals.

    

### [[2109.08561] Context-aware Retail Product Recommendation with Regularized Gradient Boosting](http://arxiv.org/abs/2109.08561)


  In the FARFETCH Fashion Recommendation challenge, the participants needed to
predict the order in which various products would be shown to a user in a
recommendation impression. The data was provided in two phases - a validation
phase and a test phase. The validation phase had a labelled training set that
contained a binary column indicating whether a product has been clicked or not.
The dataset comprises over 5,000,000 recommendation events, 450,000 products
and 230,000 unique users. It represents real, unbiased, but anonymised,
interactions of actual users of the FARFETCH platform. The final evaluation was
done according to the performance in the second phase. A total of 167
participants participated in the challenge, and we secured the 6th rank during
the final evaluation with an MRR of 0.4658 on the test set. We have designed a
unique context-aware system that takes the similarity of a product to the user
context into account to rank products more effectively. Post evaluation, we
have been able to fine-tune our approach with an MRR of 0.4784 on the test set,
which would have placed us at the 3rd position.

    

### [[2109.08564] Slot Filling for Biomedical Information Extraction](http://arxiv.org/abs/2109.08564)


  Information Extraction (IE) from text refers to the task of extracting
structured knowledge from unstructured text. The task typically consists of a
series of sub-tasks such as Named Entity Recognition and Relation Extraction.
Sourcing entity and relation type specific training data is a major bottleneck
in the above this http URL this work we present a slot filling approach to the
task of biomedical IE, effectively replacing the need for entity and
relation-specific training data, allowing to deal with zero-shot settings. We
follow the recently proposed paradigm of coupling a Tranformer-based
bi-encoder, Dense Passage Retrieval, with a Transformer-based reader model to
extract relations from biomedical text. We assemble a biomedical slot filling
dataset for both retrieval and reading comprehension and conduct a series of
experiments demonstrating that our approach outperforms a number of simpler
baselines. We also evaluate our approach end-to-end for standard as well as
zero-shot settings. Our work provides a fresh perspective on how to solve
biomedical IE tasks, in the absence of relevant training data. Our code, models
and pretrained data are available at
this https URL.

    

### [[2109.08580] Self-Supervised Neural Architecture Search for Imbalanced Datasets](http://arxiv.org/abs/2109.08580)


  Neural Architecture Search (NAS) provides state-of-the-art results when
trained on well-curated datasets with annotated labels. However, annotating
data or even having balanced number of samples can be a luxury for
practitioners from different scientific fields, e.g., in the medical domain. To
that end, we propose a NAS-based framework that bears the threefold
contributions: (a) we focus on the self-supervised scenario, i.e., where no
labels are required to determine the architecture, and (b) we assume the
datasets are imbalanced, (c) we design each component to be able to run on a
resource constrained setup, i.e., on a single GPU (e.g. Google Colab). Our
components build on top of recent developments in self-supervised
learning~\citep{zbontar2021barlow}, self-supervised NAS~\citep{kaplan2020self}
and extend them for the case of imbalanced datasets. We conduct experiments on
an (artificially) imbalanced version of CIFAR-10 and we demonstrate our
proposed method outperforms standard neural networks, while using $27\times$
less parameters. To validate our assumption on a naturally imbalanced dataset,
we also conduct experiments on ChestMNIST and COVID-19 X-ray. The results
demonstrate how the proposed method can be used in imbalanced datasets, while
it can be fully run on a single GPU. Code is available
\href{this https URL}{here}.

    

### [[2109.08597] Boosting Transformers for Job Expression Extraction and Classification in a Low-Resource Setting](http://arxiv.org/abs/2109.08597)


  In this paper, we explore possible improvements of transformer models in a
low-resource setting. In particular, we present our approaches to tackle the
first two of three subtasks of the MEDDOPROF competition, i.e., the extraction
and classification of job expressions in Spanish clinical texts. As neither
language nor domain experts, we experiment with the multilingual XLM-R
transformer model and tackle these low-resource information extraction tasks as
sequence-labeling problems. We explore domain- and language-adaptive
pretraining, transfer learning and strategic datasplits to boost the
transformer model. Our results show strong improvements using these methods by
up to 5.3 F1 points compared to a fine-tuned XLM-R model. Our best models
achieve 83.2 and 79.3 F1 for the first two tasks, respectively.

    

### [[2109.08603] Is Curiosity All You Need? On the Utility of Emergent Behaviours from Curious Exploration](http://arxiv.org/abs/2109.08603)


  Curiosity-based reward schemes can present powerful exploration mechanisms
which facilitate the discovery of solutions for complex, sparse or long-horizon
tasks. However, as the agent learns to reach previously unexplored spaces and
the objective adapts to reward new areas, many behaviours emerge only to
disappear due to being overwritten by the constantly shifting objective. We
argue that merely using curiosity for fast environment exploration or as a
bonus reward for a specific task does not harness the full potential of this
technique and misses useful skills. Instead, we propose to shift the focus
towards retaining the behaviours which emerge during curiosity-based learning.
We posit that these self-discovered behaviours serve as valuable skills in an
agent's repertoire to solve related tasks. Our experiments demonstrate the
continuous shift in behaviour throughout training and the benefits of a simple
policy snapshot method to reuse discovered behaviour for transfer tasks.

    

### [[2109.08604] Enforcing fairness in private federated learning via the modified method of differential multipliers](http://arxiv.org/abs/2109.08604)


  Federated learning with differential privacy, or private federated learning,
provides a strategy to train machine learning models while respecting users'
privacy. However, differential privacy can disproportionately degrade the
performance of the models on under-represented groups, as these parts of the
distribution are difficult to learn in the presence of noise. Existing
approaches for enforcing fairness in machine learning models have considered
the centralized setting, in which the algorithm has access to the users' data.
This paper introduces an algorithm to enforce group fairness in private
federated learning, where users' data does not leave their devices. First, the
paper extends the modified method of differential multipliers to empirical risk
minimization with fairness constraints, thus providing an algorithm to enforce
fairness in the central setting. Then, this algorithm is extended to the
private federated learning setting. The proposed algorithm, FPFL, is tested on
a federated version of the Adult dataset and an "unfair" version of the FEMNIST
dataset. The experiments on these datasets show how private federated learning
accentuates unfairness in the trained models, and how FPFL is able to mitigate
such unfairness.

    

### [[2109.08618] A review of deep learning methods for MRI reconstruction](http://arxiv.org/abs/2109.08618)


  Following the success of deep learning in a wide range of applications,
neural network-based machine-learning techniques have received significant
interest for accelerating magnetic resonance imaging (MRI) acquisition and
reconstruction strategies. A number of ideas inspired by deep learning
techniques for computer vision and image processing have been successfully
applied to nonlinear image reconstruction in the spirit of compressed sensing
for accelerated MRI. Given the rapidly growing nature of the field, it is
imperative to consolidate and summarize the large number of deep learning
methods that have been reported in the literature, to obtain a better
understanding of the field in general. This article provides an overview of the
recent developments in neural-network based approaches that have been proposed
specifically for improving parallel imaging. A general background and
introduction to parallel MRI is also given from a classical view of k-space
based reconstruction methods. Image domain based techniques that introduce
improved regularizers are covered along with k-space based methods which focus
on better interpolation strategies using neural networks. While the field is
rapidly evolving with thousands of papers published each year, in this review,
we attempt to cover broad categories of methods that have shown good
performance on publicly available data sets. Limitations and open problems are
also discussed and recent efforts for producing open data sets and benchmarks
for the community are examined.

    

### [[2109.08628] Autonomous Vision-based UAV Landing with Collision Avoidance using Deep Learning](http://arxiv.org/abs/2109.08628)


  There is a risk of collision when multiple UAVs land simultaneously without
communication on the same platform. This work accomplishes vision-based
autonomous landing and uses a deep-learning-based method to realize collision
avoidance during the landing process.

    

### [[2109.08630] A Fairness Analysis on Private Aggregation of Teacher Ensembles](http://arxiv.org/abs/2109.08630)


  The Private Aggregation of Teacher Ensembles (PATE) is an important private
machine learning framework. It combines multiple learning models used as
teachers for a student model that learns to predict an output chosen by noisy
voting among the teachers. The resulting model satisfies differential privacy
and has been shown effective in learning high-quality private models in
semisupervised settings or when one wishes to protect the data labels.
This paper asks whether this privacy-preserving framework introduces or
exacerbates bias and unfairness and shows that PATE can introduce accuracy
disparity among individuals and groups of individuals. The paper analyzes which
algorithmic and data properties are responsible for the disproportionate
impacts, why these aspects are affecting different groups disproportionately,
and proposes guidelines to mitigate these effects. The proposed approach is
evaluated on several datasets and settings.

    

### [[2109.08632] Graph Learning for Cognitive Digital Twins in Manufacturing Systems](http://arxiv.org/abs/2109.08632)


  Future manufacturing requires complex systems that connect simulation
platforms and virtualization with physical data from industrial processes.
Digital twins incorporate a physical twin, a digital twin, and the connection
between the two. Benefits of using digital twins, especially in manufacturing,
are abundant as they can increase efficiency across an entire manufacturing
life-cycle. The digital twin concept has become increasingly sophisticated and
capable over time, enabled by rises in many technologies. In this paper, we
detail the cognitive digital twin as the next stage of advancement of a digital
twin that will help realize the vision of Industry 4.0. Cognitive digital twins
will allow enterprises to creatively, effectively, and efficiently exploit
implicit knowledge drawn from the experience of existing manufacturing systems.
They also enable more autonomous decisions and control, while improving the
performance across the enterprise (at scale). This paper presents graph
learning as one potential pathway towards enabling cognitive functionalities in
manufacturing digital twins. A novel approach to realize cognitive digital
twins in the product design stage of manufacturing that utilizes graph learning
is presented.

    

### [[2109.08666] Learning Sparse Graph with Minimax Concave Penalty under Gaussian Markov Random Fields](http://arxiv.org/abs/2109.08666)


  This paper presents a convex-analytic framework to learn sparse graphs from
data. While our problem formulation is inspired by an extension of the
graphical lasso using the so-called combinatorial graph Laplacian framework, a
key difference is the use of a nonconvex alternative to the $\ell_1$ norm to
attain graphs with better interpretability. Specifically, we use the
weakly-convex minimax concave penalty (the difference between the $\ell_1$ norm
and the Huber function) which is known to yield sparse solutions with lower
estimation bias than $\ell_1$ for regression problems. In our framework, the
graph Laplacian is replaced in the optimization by a linear transform of the
vector corresponding to its upper triangular part. Via a reformulation relying
on Moreau's decomposition, we show that overall convexity is guaranteed by
introducing a quadratic function to our cost function. The problem can be
solved efficiently by the primal-dual splitting method, of which the admissible
conditions for provable convergence are presented. Numerical examples show that
the proposed method significantly outperforms the existing graph learning
methods with reasonable CPU time.

    

### [[2109.08668] Primer: Searching for Efficient Transformers for Language Modeling](http://arxiv.org/abs/2109.08668)


  Large Transformer models have been central to recent advances in natural
language processing. The training and inference costs of these models, however,
have grown rapidly and become prohibitively expensive. Here we aim to reduce
the costs of Transformers by searching for a more efficient variant. Compared
to previous approaches, our search is performed at a lower level, over the
primitives that define a Transformer TensorFlow program. We identify an
architecture, named Primer, that has a smaller training cost than the original
Transformer and other variants for auto-regressive language modeling. Primer's
improvements can be mostly attributed to two simple modifications: squaring
ReLU activations and adding a depthwise convolution layer after each Q, K, and
V projection in self-attention.
Experiments show Primer's gains over Transformer increase as compute scale
grows and follow a power law with respect to quality at optimal model sizes. We
also verify empirically that Primer can be dropped into different codebases to
significantly speed up training without additional tuning. For example, at a
500M parameter size, Primer improves the original T5 architecture on C4
auto-regressive language modeling, reducing the training cost by 4X.
Furthermore, the reduced training cost means Primer needs much less compute to
reach a target one-shot performance. For instance, in a 1.9B parameter
configuration similar to GPT-3 XL, Primer uses 1/3 of the training compute to
achieve the same one-shot performance as Transformer. We open source our models
and several comparisons in T5 to help with reproducibility.

    

### [[2109.08675] Discriminative Similarity for Data Clustering](http://arxiv.org/abs/2109.08675)


  Similarity-based clustering methods separate data into clusters according to
the pairwise similarity between the data, and the pairwise similarity is
crucial for their performance. In this paper, we propose Clustering by
Discriminative Similarity (CDS), a novel method which learns discriminative
similarity for data clustering. CDS learns an unsupervised similarity-based
classifier from each data partition, and searches for the optimal partition of
the data by minimizing the generalization error of the learnt classifiers
associated with the data partitions. By generalization analysis via Rademacher
complexity, the generalization error bound for the unsupervised
similarity-based classifier is expressed as the sum of discriminative
similarity between the data from different classes. It is proved that the
derived discriminative similarity can also be induced by the integrated squared
error bound for kernel density classification. In order to evaluate the
performance of the proposed discriminative similarity, we propose a new
clustering method using a kernel as the similarity function, CDS via
unsupervised kernel classification (CDSK), with its effectiveness demonstrated
by experimental results.

    

### [[2109.08677] Realistic PointGoal Navigation via Auxiliary Losses and Information Bottleneck](http://arxiv.org/abs/2109.08677)


  We propose a novel architecture and training paradigm for training realistic
PointGoal Navigation -- navigating to a target coordinate in an unseen
environment under actuation and sensor noise without access to ground-truth
localization. Specifically, we find that the primary challenge under this
setting is learning localization -- when stripped of idealized localization,
agents fail to stop precisely at the goal despite reliably making progress
towards it. To address this we introduce a set of auxiliary losses to help the
agent learn localization. Further, we explore the idea of treating the precise
location of the agent as privileged information -- it is unavailable during
test time, however, it is available during training time in simulation. We
grant the agent restricted access to ground-truth localization readings during
training via an information bottleneck. Under this setting, the agent incurs a
penalty for using this privileged information, encouraging the agent to only
leverage this information when it is crucial to learning. This enables the
agent to first learn navigation and then learn localization instead of
conflating these two objectives in training. We evaluate our proposed method
both in a semi-idealized (noiseless simulation without Compass+GPS) and
realistic (addition of noisy simulation) settings. Specifically, our method
outperforms existing baselines on the semi-idealized setting by 18\%/21\%
SPL/Success and by 15\%/20\% SPL in the realistic setting. Our improved Success
and SPL metrics indicate our agent's improved ability to accurately
self-localize while maintaining a strong navigation policy. Our implementation
can be found at this https URL.

    

### [[1912.03716] VideoDG: Generalizing Temporal Relations in Videos to Novel Domains](http://arxiv.org/abs/1912.03716)


  This paper introduces video domain generalization where most video
classification networks degenerate due to the lack of exposure to the target
domains of divergent distributions. We observe that the global temporal
features are less generalizable, due to the temporal domain shift that videos
from other unseen domains may have an unexpected absence or misalignment of the
temporal relations. This finding has motivated us to solve video domain
generalization by effectively learning the local-relation features of different
timescales that are more generalizable, and exploiting them along with the
global-relation features to maintain the discriminability. This paper presents
the VideoDG framework with two technical contributions. The first is a new deep
architecture named the Adversarial Pyramid Network, which improves the
generalizability of video features by capturing the local-relation,
global-relation, and cross-relation features progressively. On the basis of
pyramid features, the second contribution is a new and robust approach of
adversarial data augmentation that can bridge different video domains by
improving the diversity and quality of augmented data. We construct three video
domain generalization benchmarks in which domains are divided according to
different datasets, different consequences of actions, or different camera
views, respectively. VideoDG consistently outperforms the combinations of
previous video classification models and existing domain generalization methods
on all benchmarks.

    

### [[1912.05625] Pre-Training of Deep Bidirectional Protein Sequence Representations with Structural Information](http://arxiv.org/abs/1912.05625)


  Bridging the exponentially growing gap between the numbers of unlabeled and
labeled protein sequences, several studies adopted semi-supervised learning for
protein sequence modeling. In these studies, models were pre-trained with a
substantial amount of unlabeled data, and the representations were transferred
to various downstream tasks. Most pre-training methods solely rely on language
modeling and often exhibit limited performance. In this paper, we introduce a
novel pre-training scheme called PLUS, which stands for Protein sequence
representations Learned Using Structural information. PLUS consists of masked
language modeling and a complementary protein-specific pre-training task,
namely same-family prediction. PLUS can be used to pre-train various model
architectures. In this work, we use PLUS to pre-train a bidirectional recurrent
neural network and refer to the resulting model as PLUS-RNN. Our experiment
results demonstrate that PLUS-RNN outperforms other models of similar size
solely pre-trained with the language modeling in six out of seven widely used
protein biology tasks. Furthermore, we present the results from our qualitative
interpretation analyses to illustrate the strengths of PLUS-RNN. PLUS provides
a novel way to exploit evolutionary relationships among unlabeled proteins and
is broadly applicable across a variety of protein biology tasks. We expect that
the gap between the numbers of unlabeled and labeled proteins will continue to
grow exponentially, and the proposed pre-training method will play a larger
role.

    

### [[2002.12398] TSS: Transformation-Specific Smoothing for Robustness Certification](http://arxiv.org/abs/2002.12398)


  As machine learning (ML) systems become pervasive, safeguarding their
security is critical. However, recently it has been demonstrated that motivated
adversaries are able to mislead ML systems by perturbing test data using
semantic transformations. While there exists a rich body of research providing
provable robustness guarantees for ML models against $\ell_p$ norm bounded
adversarial perturbations, guarantees against semantic perturbations remain
largely underexplored. In this paper, we provide TSS -- a unified framework for
certifying ML robustness against general adversarial semantic transformations.
First, depending on the properties of each transformation, we divide common
transformations into two categories, namely resolvable (e.g., Gaussian blur)
and differentially resolvable (e.g., rotation) transformations. For the former,
we propose transformation-specific randomized smoothing strategies and obtain
strong robustness certification. The latter category covers transformations
that involve interpolation errors, and we propose a novel approach based on
stratified sampling to certify the robustness. Our framework TSS leverages
these certification strategies and combines with consistency-enhanced training
to provide rigorous certification of robustness. We conduct extensive
experiments on over ten types of challenging semantic transformations and show
that TSS significantly outperforms the state of the art. Moreover, to the best
of our knowledge, TSS is the first approach that achieves nontrivial certified
robustness on the large-scale ImageNet dataset. For instance, our framework
achieves 30.4% certified robust accuracy against rotation attack (within $\pm
30^\circ$) on ImageNet. Moreover, to consider a broader range of
transformations, we show TSS is also robust against adaptive attacks and
unforeseen image corruptions such as CIFAR-10-C and ImageNet-C.

    

### [[2006.13256] Rescaling Egocentric Vision](http://arxiv.org/abs/2006.13256)


  This paper introduces the pipeline to extend the largest dataset in
egocentric vision, EPIC-KITCHENS. The effort culminates in EPIC-KITCHENS-100, a
collection of 100 hours, 20M frames, 90K actions in 700 variable-length videos,
capturing long-term unscripted activities in 45 environments, using
head-mounted cameras. Compared to its previous version, EPIC-KITCHENS-100 has
been annotated using a novel pipeline that allows denser (54% more actions per
minute) and more complete annotations of fine-grained actions (+128% more
action segments). This collection enables new challenges such as action
detection and evaluating the "test of time" - i.e. whether models trained on
data collected in 2018 can generalise to new footage collected two years later.
The dataset is aligned with 6 challenges: action recognition (full and weak
supervision), action detection, action anticipation, cross-modal retrieval
(from captions), as well as unsupervised domain adaptation for action
recognition. For each challenge, we define the task, provide baselines and
evaluation metrics

    

### [[2006.16789] Causality Learning: A New Perspective for Interpretable Machine Learning](http://arxiv.org/abs/2006.16789)


  Recent years have witnessed the rapid growth of machine learning in a wide
range of fields such as image recognition, text classification, credit scoring
prediction, recommendation system, etc. In spite of their great performance in
different sectors, researchers still concern about the mechanism under any
machine learning (ML) techniques that are inherently black-box and becoming
more complex to achieve higher accuracy. Therefore, interpreting machine
learning model is currently a mainstream topic in the research community.
However, the traditional interpretable machine learning focuses on the
association instead of the causality. This paper provides an overview of causal
analysis with the fundamental background and key concepts, and then summarizes
most recent causal approaches for interpretable machine learning. The
evaluation techniques for assessing method quality, and open problems in causal
interpretability are also discussed in this paper.

    

### [[2007.15528] Membership Leakage in Label-Only Exposures](http://arxiv.org/abs/2007.15528)


  Machine learning (ML) has been widely adopted in various privacy-critical
applications, e.g., face recognition and medical image analysis. However,
recent research has shown that ML models are vulnerable to attacks against
their training data. Membership inference is one major attack in this domain:
Given a data sample and model, an adversary aims to determine whether the
sample is part of the model's training set. Existing membership inference
attacks leverage the confidence scores returned by the model as their inputs
(score-based attacks). However, these attacks can be easily mitigated if the
model only exposes the predicted label, i.e., the final model decision.
In this paper, we propose decision-based membership inference attacks and
demonstrate that label-only exposures are also vulnerable to membership
leakage. In particular, we develop two types of decision-based attacks, namely
transfer attack, and boundary attack. Empirical evaluation shows that our
decision-based attacks can achieve remarkable performance, and even outperform
the previous score-based attacks in some cases. We further present new insights
on the success of membership inference based on quantitative and qualitative
analysis, i.e., member samples of a model are more distant to the model's
decision boundary than non-member samples. Finally, we evaluate multiple
defense mechanisms against our decision-based attacks and show that our two
types of attacks can bypass most of these defenses.

    

### [[2007.15779] Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](http://arxiv.org/abs/2007.15779)


  Pretraining large neural language models, such as BERT, has led to impressive
gains on many natural language processing (NLP) tasks. However, most
pretraining efforts focus on general domain corpora, such as newswire and Web.
A prevailing assumption is that even domain-specific pretraining can benefit by
starting from general-domain language models. In this paper, we challenge this
assumption by showing that for domains with abundant unlabeled text, such as
biomedicine, pretraining language models from scratch results in substantial
gains over continual pretraining of general-domain language models. To
facilitate this investigation, we compile a comprehensive biomedical NLP
benchmark from publicly-available datasets. Our experiments show that
domain-specific pretraining serves as a solid foundation for a wide range of
biomedical NLP tasks, leading to new state-of-the-art results across the board.
Further, in conducting a thorough evaluation of modeling choices, both for
pretraining and task-specific fine-tuning, we discover that some common
practices are unnecessary with BERT models, such as using complex tagging
schemes in named entity recognition (NER). To help accelerate research in
biomedical NLP, we have released our state-of-the-art pretrained and
task-specific models for the community, and created a leaderboard featuring our
BLURB benchmark (short for Biomedical Language Understanding & Reasoning
Benchmark) at this https URL.

    

### [[2008.06006] Textual Echo Cancellation](http://arxiv.org/abs/2008.06006)


  In this paper, we propose Textual Echo Cancellation (TEC) - a framework for
cancelling the text-to-speech (TTS) playback echo from overlapping speech
recordings. Such a system can largely improve speech recognition performance
and user experience for intelligent devices such as smart speakers, as the user
can talk to the device while the device is still playing the TTS signal
responding to the previous query. We implement this system by using a novel
sequence-to-sequence model with multi-source attention that takes both the
microphone mixture signal and source text of the TTS playback as inputs, and
predicts the enhanced audio. Experiments show that the textual information of
the TTS playback is critical to enhancement performance. Besides, the text
sequence is much smaller in size compared with the raw acoustic signal of the
TTS playback, and can be immediately transmitted to the device or ASR server
even before the playback is synthesized. Therefore, our proposed approach
effectively reduces Internet communication and latency compared with
alternative approaches such as acoustic echo cancellation (AEC).

    

### [[2008.11600] Estimating Example Difficulty Using Variance of Gradients](http://arxiv.org/abs/2008.11600)


  In machine learning, a question of great interest is understanding what
examples are challenging for a model to classify. Identifying atypical examples
ensures the safe deployment of models, isolates samples that require further
human inspection and provides interpretability into model behavior. In this
work, we propose Variance of Gradients (VoG) as a valuable and efficient metric
to rank data by difficulty and to surface a tractable subset of the most
challenging examples for human-in-the-loop auditing. We show that data points
with high VoG scores are far more difficult for the model to learn and
over-index on corrupted or memorized examples. Further, restricting the
evaluation to the test set instances with the lowest VoG improves the model's
generalization performance. Finally, we show that VoG is a valuable and
efficient ranking for out-of-distribution detection.

    

### [[2010.09469] A Point-Cloud Deep Learning Framework for Prediction of Fluid Flow Fields on Irregular Geometries](http://arxiv.org/abs/2010.09469)


  We present a novel deep learning framework for flow field predictions in
irregular domains when the solution is a function of the geometry of either the
domain or objects inside the domain. Grid vertices in a computational fluid
dynamics (CFD) domain are viewed as point clouds and used as inputs to a neural
network based on the PointNet architecture, which learns an end-to-end mapping
between spatial positions and CFD quantities. Using our approach, (i) the
network inherits desirable features of unstructured meshes (e.g., fine and
coarse point spacing near the object surface and in the far field,
respectively), which minimizes network training cost; (ii) object geometry is
accurately represented through vertices located on object boundaries, which
maintains boundary smoothness and allows the network to detect small changes
between geometries; and (iii) no data interpolation is utilized for creating
training data; thus accuracy of the CFD data is preserved. None of these
features are achievable by extant methods based on projecting scattered CFD
data into Cartesian grids and then using regular convolutional neural networks.
Incompressible laminar steady flow past a cylinder with various shapes for its
cross section is considered. The mass and momentum of predicted fields are
conserved. We test the generalizability of our network by predicting the flow
around multiple objects as well as an airfoil, even though only single objects
and no airfoils are observed during training. The network predicts the flow
fields hundreds of times faster than our conventional CFD solver, while
maintaining excellent to reasonable accuracy.

    

### [[2010.15417] ProCAN: Progressive Growing Channel Attentive Non-Local Network for Lung Nodule Classification](http://arxiv.org/abs/2010.15417)


  Lung cancer classification in screening computed tomography (CT) scans is one
of the most crucial tasks for early detection of this disease. Many lives can
be saved if we are able to accurately classify malignant/cancerous lung
nodules. Consequently, several deep learning based models have been proposed
recently to classify lung nodules as malignant or benign. Nevertheless, the
large variation in the size and heterogeneous appearance of the nodules makes
this task an extremely challenging one. We propose a new Progressive Growing
Channel Attentive Non-Local (ProCAN) network for lung nodule classification.
The proposed method addresses this challenge from three different aspects.
First, we enrich the Non-Local network by adding channel-wise attention
capability to it. Second, we apply Curriculum Learning principles, whereby we
first train our model on easy examples before hard ones. Third, as the
classification task gets harder during the Curriculum learning, our model is
progressively grown to increase its capability of handling the task at hand. We
examined our proposed method on two different public datasets and compared its
performance with state-of-the-art methods in the literature. The results show
that the ProCAN model outperforms state-of-the-art methods and achieves an AUC
of 98.05% and an accuracy of 95.28% on the LIDC-IDRI dataset. Moreover, we
conducted extensive ablation studies to analyze the contribution and effects of
each new component of our proposed method.

    

### [[2011.06923] LEAN: graph-based pruning for convolutional neural networks by extracting longest chains](http://arxiv.org/abs/2011.06923)


  Convolutional neural networks (CNNs) have proven to be highly successful at a
range of image-to-image tasks. CNNs can be prohibitively computationally
expensive for real time use, which can limit their applicability in practice.
Model pruning can improve computational efficiency by sparsifying trained
networks. Common methods for pruning CNNs determine what convolutional filters
to remove by ranking filters on an individual basis. However, filters are not
independent, as CNNs consist of chains of convolutions, which can result in
sub-optimal filter selection.
We propose a novel pruning method, LongEst-chAiN (LEAN) pruning, which takes
the interdependency between the convolution operations into account. We propose
to prune CNNs by using graph-based algorithms to select relevant chains of
convolutions. A CNN is interpreted as a graph, with the operator norm of each
operator as distance metric for the edges. LEAN pruning iteratively extracts
the highest value path from the graph to keep. In our experiments, we test LEAN
pruning for several image-to-image tasks, including the well-known CamVid
dataset, and a real-world dynamic X-ray CT dataset. When pruning CNNs with
LEAN, we achieve a higher accuracy than pruning filters individually, and
different pruned substructures emerge.

    

### [[2012.03628] K-means for Evolving Data Streams](http://arxiv.org/abs/2012.03628)


  Currently the amount of data produced worldwide is increasing beyond measure,
thus a high volume of unsupervised data must be processed continuously. One of
the main unsupervised data analysis is clustering. In streaming data scenarios,
the data is composed by an increasing sequence of batches of samples where the
concept drift phenomenon may happen. In this paper, we formally define the
Streaming $K$-means(S$K$M) problem, which implies a restart of the error
function when a concept drift occurs. We propose a surrogate error function
that does not rely on concept drift detection. We proof that the surrogate is a
good approximation of the S$K$M error. Hence, we suggest an algorithm which
minimizes this alternative error each time a new batch arrives. We present some
initialization techniques for streaming data scenarios as well. Besides
providing theoretical results, experiments demonstrate an improvement of the
converged error for the non-trivial initialization methods.

    

### [[2101.01300] A Linearly Convergent Algorithm for Distributed Principal Component Analysis](http://arxiv.org/abs/2101.01300)


  Principal Component Analysis (PCA) is the workhorse tool for dimensionality
reduction in this era of big data. While often overlooked, the purpose of PCA
is not only to reduce data dimensionality, but also to yield features that are
uncorrelated. Furthermore, the ever-increasing volume of data in the modern
world often requires storage of data samples across multiple machines, which
precludes the use of centralized PCA algorithms. This paper focuses on the dual
objective of PCA, namely, dimensionality reduction and decorrelation of
features, but in a distributed setting. This requires estimating the
eigenvectors of the data covariance matrix, as opposed to only estimating the
subspace spanned by the eigenvectors, when data is distributed across a network
of machines. Although a few distributed solutions to the PCA problem have been
proposed recently, convergence guarantees and/or communications overhead of
these solutions remain a concern. With an eye towards communications
efficiency, this paper introduces a feedforward neural network-based one
time-scale distributed PCA algorithm termed Distributed Sanger's Algorithm
(DSA) that estimates the eigenvectors of the data covariance matrix when data
is distributed across an undirected and arbitrarily connected network of
machines. Furthermore, the proposed algorithm is shown to converge linearly to
a neighborhood of the true solution. Numerical results are also provided to
demonstrate the efficacy of the proposed solution.

    

### [[2101.03170] BDNNSurv: Bayesian deep neural networks for survival analysis using pseudo values](http://arxiv.org/abs/2101.03170)


  There has been increasing interest in modeling survival data using deep
learning methods in medical research. In this paper, we proposed a Bayesian
hierarchical deep neural networks model for modeling and prediction of survival
data. Compared with previously studied methods, the new proposal can provide
not only point estimate of survival probability but also quantification of the
corresponding uncertainty, which can be of crucial importance in predictive
modeling and subsequent decision making. The favorable statistical properties
of point and uncertainty estimates were demonstrated by simulation studies and
real data analysis. The Python code implementing the proposed approach was
provided.

    

### [[2102.05297] Using hardware performance counters to speed up autotuning convergence on GPUs](http://arxiv.org/abs/2102.05297)


  Nowadays, GPU accelerators are commonly used to speed up general-purpose
computing tasks on a variety of hardware. However, due to the diversity of GPU
architectures and processed data, optimization of codes for a particular type
of hardware and specific data characteristics can be extremely challenging. The
autotuning of performance-relevant source-code parameters allows for automatic
optimization of applications and keeps their performance portable. Although the
autotuning process typically results in code speed-up, searching the tuning
space can bring unacceptable overhead if (i) the tuning space is vast and full
of poorly-performing implementations, or (ii) the autotuning process has to be
repeated frequently because of changes in processed data or migration to
different hardware.
In this paper, we introduce a novel method for searching tuning spaces. The
method takes advantage of collecting hardware performance counters (also known
as profiling counters) during empirical tuning. Those counters are used to
navigate the searching process towards faster implementations. The method
requires the tuning space to be sampled on any GPU. It builds a
problem-specific model, which can be used during autotuning on various, even
previously unseen inputs or GPUs. Using a set of five benchmarks, we
experimentally demonstrate that our method can speed up autotuning when an
application needs to be ported to different hardware or when it needs to
process data with different characteristics. We also compared our method to
state of the art and show that our method is superior in terms of the number of
searching steps and typically outperforms other searches in terms of
convergence time.

    

### [[2102.06808] Planning and Learning Using Adaptive Entropy Tree Search](http://arxiv.org/abs/2102.06808)


  We present the Adaptive Entropy Tree Search (ANTS) algorithm, a planning
method based on the Principle of Maximum Entropy. Importantly, we design ANTS
so that it is a practical component of a planning-learning loop, outperforming
state-of-the-art methods on the Atari benchmark. The key algorithmic novelty is
entropy parameterization, which mitigates sensitivity to the temperature
parameter - a bottleneck of the prior maximum entropy planning methods. To
confirm our design choices, we perform a comprehensive suite of ablations in
isolation from learning. Moreover, we theoretically show that ANTS enjoys
exponential convergence in the softmax bandit setting.

    

### [[2102.08245] Classification of multivariate weakly-labelled time-series with attention](http://arxiv.org/abs/2102.08245)


  This research identifies a gap in weakly-labelled multivariate time-series
classification (TSC), where state-of-the-art TSC models do not per-form well.
Weakly labelled time-series are time-series containing noise and significant
redundancies. In response to this gap, this paper proposes an approach of
exploiting context relevance of subsequences from previous subsequences to
improve classification accuracy. To achieve this, state-of-the-art Attention
algorithms are experimented in combination with the top CNN models for TSC (FCN
and ResNet), in an CNN-LSTM architecture. Attention is a popular strategy for
context extraction with exceptional performance in modern sequence-to-sequence
tasks. This paper shows how attention algorithms can be used for improved
weakly labelledTSC by evaluating models on a multivariate EEG time-series
dataset obtained using a commercial Emotiv headsets from participants
performing various activities while driving. These time-series are segmented
into sub-sequences and labelled to allow supervised TSC.

    

### [[2103.03099] ILoSA: Interactive Learning of Stiffness and Attractors](http://arxiv.org/abs/2103.03099)


  Teaching robots how to apply forces according to our preferences is still an
open challenge that has to be tackled from multiple engineering perspectives.
This paper studies how to learn variable impedance policies where both the
Cartesian stiffness and the attractor can be learned from human demonstrations
and corrections with a user-friendly interface. The presented framework, named
ILoSA, uses Gaussian Processes for policy learning, identifying regions of
uncertainty and allowing interactive corrections, stiffness modulation and
active disturbance rejection. The experimental evaluation of the framework is
carried out on a Franka-Emika Panda in four separate cases with unique force
interaction properties: 1) pulling a plug wherein a sudden force discontinuity
occurs upon successful removal of the plug, 2) pushing a box where a sustained
force is required to keep the robot in motion, 3) wiping a whiteboard in which
the force is applied perpendicular to the direction of movement, and 4)
inserting a plug to verify the usability for precision-critical tasks in an
experimental validation performed with non-expert users.

    

### [[2103.13885] Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier in Online Class-Incremental Continual Learning](http://arxiv.org/abs/2103.13885)


  Online class-incremental continual learning (CL) studies the problem of
learning new classes continually from an online non-stationary data stream,
intending to adapt to new data while mitigating catastrophic forgetting. While
memory replay has shown promising results, the recency bias in online learning
caused by the commonly used Softmax classifier remains an unsolved challenge.
Although the Nearest-Class-Mean (NCM) classifier is significantly undervalued
in the CL community, we demonstrate that it is a simple yet effective
substitute for the Softmax classifier. It addresses the recency bias and avoids
structural changes in the fully-connected layer for new classes. Moreover, we
observe considerable and consistent performance gains when replacing the
Softmax classifier with the NCM classifier for several state-of-the-art replay
methods. To leverage the NCM classifier more effectively, data embeddings
belonging to the same class should be clustered and well-separated from those
with a different class label. To this end, we contribute Supervised Contrastive
Replay (SCR), which explicitly encourages samples from the same class to
cluster tightly in embedding space while pushing those of different classes
further apart during replay-based training. Overall, we observe that our
proposed SCR substantially reduces catastrophic forgetting and outperforms
state-of-the-art CL methods by a significant margin on a variety of datasets.

    

### [[2104.11843] One-Round Active Learning](http://arxiv.org/abs/2104.11843)


  In this work, we initiate the study of one-round active learning, which aims
to select a subset of unlabeled data points that achieve the highest model
performance after being labeled with only the information from initially
labeled data points. The challenge of directly applying existing data selection
criteria to the one-round setting is that they are not indicative of model
performance when available labeled data is limited. We address the challenge by
explicitly modeling the dependence of model performance on the dataset.
Specifically, we propose DULO, a data-driven framework for one-round active
learning, wherein we learn a model to predict the model performance for a given
dataset and then leverage this model to guide the selection of unlabeled data.
Our results demonstrate that DULO leads to the state-of-the-art performance on
various active learning benchmarks in the one-round setting.

    

### [[2104.14527] Online certification of preference-based fairness for personalized recommender systems](http://arxiv.org/abs/2104.14527)


  Recommender systems are facing scrutiny because of their growing impact on
the opportunities we have access to. Current audits for fairness are limited to
coarse-grained parity assessments at the level of sensitive groups. We propose
to audit for envy-freeness, a more granular criterion aligned with individual
preferences: every user should prefer their recommendations to those of other
users. Since auditing for envy requires to estimate the preferences of users
beyond their existing recommendations, we cast the audit as a new pure
exploration problem in multi-armed bandits. We propose a sample-efficient
algorithm with theoretical guarantees that it does not deteriorate user
experience. We also study the trade-offs achieved on real-world recommendation
datasets.

    

### [[2104.14929] On In-network learning. A Comparative Study with Federated and Split Learning](http://arxiv.org/abs/2104.14929)


  In this paper, we consider a problem in which distributively extracted
features are used for performing inference in wireless networks. We elaborate
on our proposed architecture, which we herein refer to as "in-network
learning", provide a suitable loss function and discuss its optimization using
neural networks. We compare its performance with both Federated- and Split
learning; and show that this architecture offers both better accuracy and
bandwidth savings.

    

### [[2105.01016] AI-assisted super-resolution cosmological simulations II: Halo substructures, velocities and higher order statistics](http://arxiv.org/abs/2105.01016)


  In this work, we expand and test the capabilities of our recently developed
super-resolution (SR) model to generate high-resolution (HR) realizations of
the full phase-space matter distribution, including both displacement and
velocity, from computationally cheap low-resolution (LR) cosmological N-body
simulations. The SR model enhances the simulation resolution by generating 512
times more tracer particles, extending into the deeply non-linear regime where
complex structure formation processes take place. We validate the SR model by
deploying the model in 10 test simulations of box size 100 Mpc/h, and examine
the matter power spectra, bispectra and 2D power spectra in redshift space. We
find the generated SR field matches the true HR result at percent level down to
scales of k ~ 10 h/Mpc. We also identify and inspect dark matter halos and
their substructures. Our SR model generate visually authentic small-scale
structures, that cannot be resolved by the LR input, and are in good
statistical agreement with the real HR results. The SR model performs
satisfactorily on the halo occupation distribution, halo correlations in both
real and redshift space, and the pairwise velocity distribution, matching the
HR results with comparable scatter, thus demonstrating its potential in making
mock halo catalogs. The SR technique can be a powerful and promising tool for
modelling small-scale galaxy formation physics in large cosmological volumes.

    

### [[2105.02551] Structured Ensembles: an Approach to Reduce the Memory Footprint of Ensemble Methods](http://arxiv.org/abs/2105.02551)


  In this paper, we propose a novel ensembling technique for deep neural
networks, which is able to drastically reduce the required memory compared to
alternative approaches. In particular, we propose to extract multiple
sub-networks from a single, untrained neural network by solving an end-to-end
optimization task combining differentiable scaling over the original
architecture, with multiple regularization terms favouring the diversity of the
ensemble. Since our proposal aims to detect and extract sub-structures, we call
it Structured Ensemble. On a large experimental evaluation, we show that our
method can achieve higher or comparable accuracy to competing methods while
requiring significantly less storage. In addition, we evaluate our ensembles in
terms of predictive calibration and uncertainty, showing they compare
favourably with the state-of-the-art. Finally, we draw a link with the
continual learning literature, and we propose a modification of our framework
to handle continuous streams of tasks with a sub-linear memory cost. We compare
with a number of alternative strategies to mitigate catastrophic forgetting,
highlighting advantages in terms of average accuracy and memory.

    

### [[2106.00116] Effect of Pre-Training Scale on Intra- and Inter-Domain Full and Few-Shot Transfer Learning for Natural and Medical X-Ray Chest Images](http://arxiv.org/abs/2106.00116)


  Transfer learning aims to exploit pre-trained models for more efficient
follow-up training on wide range of downstream tasks and datasets, enabling
successful training also on small data. Recently, strong improvement was shown
for transfer learning and model generalization when increasing model, data and
compute budget scale in the pre-training. To compare effect of scale both in
intra- and inter-domain full and few-shot transfer, in this study we combine
for the first time large openly available medical X-Ray chest imaging datasets
to reach a dataset scale comparable to ImageNet-1k. We then conduct
pre-training and transfer to different natural or medical targets while varying
network size and source data scale and domain, being either large natural
(ImageNet-1k/21k) or large medical chest X-Ray datasets. We observe strong
improvement due to larger pre-training scale for intra-domain natural-natural
and medical-medical transfer. For inter-domain natural-medical transfer, we
find improvements due to larger pre-training scale on larger X-Ray targets in
full shot regime, while for smaller targets and for few-shot regime the
improvement is not visible. Remarkably, large networks pre-trained on very
large natural ImageNet-21k are as good or better than networks pre-trained on
largest available medical X-Ray data when performing transfer to large X-Ray
targets. We conclude that high quality models for inter-domain transfer can be
also obtained by substantially increasing scale of model and generic natural
source data, removing necessity for large domain-specific medical source data
in the pre-training. Code is available at:
\url{this https URL}}

    

### [[2106.02100] Double Descent Optimization Pattern and Aliasing: Caveats of Noisy Labels](http://arxiv.org/abs/2106.02100)


  Optimization plays a key role in the training of deep neural networks.
Deciding when to stop training can have a substantial impact on the performance
of the network during inference. Under certain conditions, the generalization
error can display a double descent pattern during training: the learning curve
is non-monotonic and seemingly diverges before converging again after
additional epochs. This optimization pattern can lead to early stopping
procedures to stop training before the second convergence and consequently
select a suboptimal set of parameters for the network, with worse performance
during inference. In this work, in addition to confirming that double descent
occurs with small datasets and noisy labels as evidenced by others, we show
that noisy labels must be present both in the training and generalization sets
to observe a double descent pattern. We also show that the learning rate has an
influence on double descent, and study how different optimizers and optimizer
parameters influence the apparition of double descent. Finally, we show that
increasing the learning rate can create an aliasing effect that masks the
double descent pattern without suppressing it. We study this phenomenon through
extensive experiments on variants of CIFAR-10 and show that they translate to a
real world application: the forecast of seizure events in epileptic patients
from continuous electroencephalographic recordings.

    

### [[2106.04484] Are VQA Systems RAD? Measuring Robustness to Augmented Data with Focused Interventions](http://arxiv.org/abs/2106.04484)


  Deep learning algorithms have shown promising results in visual question
answering (VQA) tasks, but a more careful look reveals that they often do not
understand the rich signal they are being fed with. To understand and better
measure the generalization capabilities of VQA systems, we look at their
robustness to counterfactually augmented data. Our proposed augmentations are
designed to make a focused intervention on a specific property of the question
such that the answer changes. Using these augmentations, we propose a new
robustness measure, Robustness to Augmented Data (RAD), which measures the
consistency of model predictions between original and augmented examples.
Through extensive experimentation, we show that RAD, unlike classical accuracy
measures, can quantify when state-of-the-art systems are not robust to
counterfactuals. We find substantial failure cases which reveal that current
VQA systems are still brittle. Finally, we connect between robustness and
generalization, demonstrating the predictive power of RAD for performance on
unseen augmentations.

    

### [[2106.11175] NetTraj: A Network-based Vehicle Trajectory Prediction Model with Directional Representation and Spatiotemporal Attention Mechanisms](http://arxiv.org/abs/2106.11175)


  Trajectory prediction of vehicles in city-scale road networks is of great
importance to various location-based applications such as vehicle navigation,
traffic management, and location-based recommendations. Existing methods
typically represent a trajectory as a sequence of grid cells, road segments or
intention sets. None of them is ideal, as the cell-based representation ignores
the road network structures and the other two are less efficient in analyzing
city-scale road networks. Moreover, previous models barely leverage spatial
dependencies or only consider them at the grid cell level, ignoring the
non-Euclidean spatial structure shaped by irregular road networks. To address
these problems, we propose a network-based vehicle trajectory prediction model
named NetTraj, which represents each trajectory as a sequence of intersections
and associated movement directions, and then feeds them into a LSTM
encoder-decoder network for future trajectory generation. Furthermore, we
introduce a local graph attention mechanism to capture network-level spatial
dependencies of trajectories, and a temporal attention mechanism with a sliding
context window to capture both short- and long-term temporal dependencies in
trajectory data. Extensive experiments based on two real-world large-scale taxi
trajectory datasets show that NetTraj outperforms the existing state-of-the-art
methods for vehicle trajectory prediction, validating the effectiveness of the
proposed trajectory representation method and spatiotemporal attention
mechanisms.

    

### [[2106.16233] Long Short-term Cognitive Networks](http://arxiv.org/abs/2106.16233)


  In this paper, we present a recurrent neural system named Long Short-term
Cognitive Networks (LSTCNs) as a generalization of the Short-term Cognitive
Network (STCN) model. Such a generalization is motivated by the difficulty of
forecasting very long time series efficiently. The LSTCN model can be defined
as a collection of STCN blocks, each processing a specific time patch of the
(multivariate) time series being modeled. In this neural ensemble, each block
passes information to the subsequent one in the form of weight matrices
representing the prior knowledge. As a second contribution, we propose a
deterministic learning algorithm to compute the learnable weights while
preserving the prior knowledge resulting from previous learning processes. As a
third contribution, we introduce a feature influence score as a proxy to
explain the forecasting process in multivariate time series. The simulations
using three case studies show that our neural system reports small forecasting
errors while being significantly faster than state-of-the-art recurrent models.

    

### [[2107.00195] Non-parametric Semi-Supervised Learning in Many-body Hilbert Space with Rescaled Logarithmic Fidelity](http://arxiv.org/abs/2107.00195)


  In quantum and quantum-inspired machine learning, the very first step is to
embed the data in quantum space known as Hilbert space. Developing quantum
kernel function (QKF), which defines the distances among the samples in the
Hilbert space, belongs to the fundamental topics for machine learning. In this
work, we propose the rescaled logarithmic fidelity (RLF) and non-parametric
semi-supervised learning in the quantum space, which we name as RLF-NSSL. The
rescaling takes advantage of the non-linearity of the kernel to tune the mutual
distances of samples in the Hilbert space, and meanwhile avoids the
exponentially-small fidelities between quantum many-qubit states. Being
non-parametric excludes the possible effects from the variational parameters,
and evidently demonstrates the advantages from the space itself. We compare
RLF-NSSL with several well-known non-parametric algorithms including naive
Bayes classifiers, k-nearest neighbors, and spectral clustering. Our method
exhibits better accuracy particularly for the unsupervised case with no labeled
samples and the few-shot cases with small numbers of labeled samples. With the
visualizations by t-stochastic neighbor embedding, our results imply that the
machine learning in the Hilbert space complies with the principles of maximal
coding rate reduction, where the low-dimensional data exhibit within-class
compressibility, between-class discrimination, and overall diversity. Our
proposals can be applied to other quantum and quantum-inspired machine
learning, including the methods using the parametric models such as tensor
networks, quantum circuits, and quantum neural networks.

    

### [[2107.00425] Online learning of windmill time series using Long Short-term Cognitive Networks](http://arxiv.org/abs/2107.00425)


  Forecasting windmill time series is often the basis of other processes such
as anomaly detection, health monitoring, or maintenance scheduling. The amount
of data generated on windmill farms makes online learning the most viable
strategy to follow. Such settings require retraining the model each time a new
batch of data is available. However, update the model with the new information
is often very expensive to perform using traditional Recurrent Neural Networks
(RNNs). In this paper, we use Long Short-term Cognitive Networks (LSTCNs) to
forecast windmill time series in online settings. These recently introduced
neural systems consist of chained Short-term Cognitive Network blocks, each
processing a temporal data chunk. The learning algorithm of these blocks is
based on a very fast, deterministic learning rule that makes LSTCNs suitable
for online learning tasks. The numerical simulations using a case study with
four windmills showed that our approach reported the lowest forecasting errors
with respect to a simple RNN, a Long Short-term Memory, a Gated Recurrent Unit,
and a Hidden Markov Model. What is perhaps more important is that the LSTCN
approach is significantly faster than these state-of-the-art models.

    

### [[2109.08225] The Accuracy and Efficiency of Posit Arithmetic](http://arxiv.org/abs/2109.08225)


  Motivated by the increasing interest in the posit numeric format, in this
paper we evaluate the accuracy and efficiency of posit arithmetic in contrast
to the traditional IEEE 754 32-bit floating-point (FP32) arithmetic. We first
design and implement a Posit Arithmetic Unit (PAU), called POSAR, with flexible
bit-sized arithmetic suitable for applications that can trade accuracy for
savings in chip area. Next, we analyze the accuracy and efficiency of POSAR
with a series of benchmarks including mathematical computations, ML kernels,
NAS Parallel Benchmarks (NPB), and Cifar-10 CNN. This analysis is done on our
implementation of POSAR integrated into a RISC-V Rocket Chip core in comparison
with the IEEE 754-based Floting Point Unit (FPU) of Rocket Chip. Our analysis
shows that POSAR can outperform the FPU, but the results are not spectacular.
For NPB, 32-bit posit achieves better accuracy than FP32 and improves the
execution by up to 2%. However, POSAR with 32-bit posit needs 30% more FPGA
resources compared to the FPU. For classic ML algorithms, we find that 8-bit
posits are not suitable to replace FP32 because they exhibit low accuracy
leading to wrong results. Instead, 16-bit posit offers the best option in terms
of accuracy and efficiency. For example, 16-bit posit achieves the same Top-1
accuracy as FP32 on a Cifar-10 CNN with a speedup of 18%.

    

### [[2012.11473] From micro-OPs to abstract resources: constructing a simpler CPU performance model through microbenchmarking](http://arxiv.org/abs/2012.11473)


  In a super-scalar architecture, the scheduler dynamically assigns
micro-operations ({\mu}ops) to execution ports. The port mapping of an
architecture describes how an instruction decomposes into {\mu}ops and lists
for each {\mu}ops the set of ports it can be mapped to. It is used by compilers
and performance debugging tools to characterize the performance throughput of a
sequence of instructions repeatedly executed as the core component of a loop.
This paper introduces a dual equivalent representation: The resource mapping
of an architecture is an abstract model where, to be executed, an instruction
must use a set of abstract resources, themselves representing combinations of
execution ports. For a given architecture, finding a port mapping is an
important but difficult problem. Building a resource mapping is a more
tractable problem and provides a simpler and equivalent model. This paper
describes PALMED, a tool that automatically builds a resource mapping for
pipelined, super-scalar, out-of-order CPU architectures. PALMED does not
require hardware performance counters, and relies solely on runtime
measurements.
We evaluate the pertinence of our dual representation for throughput modeling
by extracting a representative set of basic-blocks from the compiled binaries
of the SPEC CPU 2017 benchmarks~\cite{SPECCPU2017}. We compared the throughput
predicted by existing machine models to that produced by \tool, and found
comparable accuracy to state-of-the art tools, achieving sub-10 \% mean square
error rate on this workload on Intel's Skylake microarchitecture.

    

### [[2109.08192] BuDDI: A Declarative Bloom Language for CALM Programming](http://arxiv.org/abs/2109.08192)


  In this paper, we show through examples that Bloom is a suitable language
beyond classical distributed computing. Based on these examples, we refine the
current Bloom implementation, called Bud, and introduce a new language called
BuDDI: Bud to enhance Distributed Data Independence. BuDDI hides the challenges
of distributed computing from the programmer without compromising performance
through logically global tables.

    

### [[2109.08219] Dr. Top-k: Delegate-Centric Top-k on GPUs](http://arxiv.org/abs/2109.08219)


  Recent top-$k$ computation efforts explore the possibility of revising
various sorting algorithms to answer top-$k$ queries on GPUs. These endeavors,
unfortunately, perform significantly more work than needed. This paper
introduces Dr. Top-k, a Delegate-centric top-$k$ system on GPUs that can reduce
the top-$k$ workloads significantly. Particularly, it contains three major
contributions: First, we introduce a comprehensive design of the
delegate-centric concept, including maximum delegate, delegate-based filtering,
and $\beta$ delegate mechanisms to help reduce the workload for top-$k$ up to
more than 99%. Second, due to the difficulty and importance of deriving a
proper subrange size, we perform a rigorous theoretical analysis, coupled with
thorough experimental validations to identify the desirable subrange size.
Third, we introduce four key system optimizations to enable fast multi-GPU
top-$k$ computation. Taken together, this work constantly outperforms the
state-of-the-art.

    

### [[2109.08241] The enhanced derived-vector-space approach to domain decomposition methods](http://arxiv.org/abs/2109.08241)


  Standard approaches to domain decomposition methods (DDM) are uncapable of
producing block-diagonal system matrices. The derived-vector-space (DVS),
approach to DDM, introduced in 2013, overcomes this limitation. However, the
DVS approach in its original form was applicable to a relatively narrow class
of problems because it required building a special matrix, whose construction
is frequently impossible. In this paper, an enhanced formulation of DVS is
presented, which does not require the construction of a special matrix and is
applicable to any linear problem. Keywords: domain decomposition, parallel
processing, DVS, FETI, BDDC

    

### [[2109.08288] DMAPF: A Decentralized and Distributed Solver for Multi-Agent Path Finding Problem with Obstacles](http://arxiv.org/abs/2109.08288)


  Multi-Agent Path Finding (MAPF) is a problem of finding a sequence of
movements for agents to reach their assigned location without collision.
Centralized algorithms usually give optimal solutions, but have difficulties to
scale without employing various techniques - usually with a sacrifice of
optimality; but solving MAPF problems with the number of agents greater than a
thousand remains a challenge nevertheless. To tackle the scalability issue, we
present DMAPF - a decentralized and distributed MAPF solver, which is a
continuation of our recently published work, ros-dmapf. We address the issues
of ros-dmapf where it (i) only works in maps without obstacles; and (ii) has a
low success rate with dense maps. Given a MAPF problem, both ros-dmapf and
DMAPF divide the map spatially into subproblems, but the latter further divides
each subproblem into disconnected regions called areas. Each subproblem is
assigned to a distributed solver, which then individually creates an abstract
plan - a sequence of areas that an agent needs to visit - for each agent in it,
and interleaves agent migration with movement planning. Answer Set Programming,
which is known for its performance in small but complex problems, is used in
many parts including problem division, abstract planning, border assignment for
the migration, and movement planning. Robot Operating System is used to
facilitate communication between the solvers and to enable the opportunity to
integrate with robotic systems. DMAPF introduces a new interaction protocol
between the solvers, and mechanisms that together result in a higher success
rate and better solution quality without sacrificing much of the performance.
We implement and experimentally validate DMAPF by comparing it with other
state-of-the-art MAPF solvers and the results show that our system achieves
better scalability.

    

### [[2109.08315] Reconfigurable Broadcast Networks and Asynchronous Shared-Memory Systems are Equivalent](http://arxiv.org/abs/2109.08315)


  We show the equivalence of two distributed computing models, namely
reconfigurable broadcast networks (RBN) and asynchronous shared-memory systems
(ASMS), that were introduced independently. Both RBN and ASMS are systems in
which a collection of anonymous, finite-state processes run the same protocol.
In RBN, the processes communicate by selective broadcast: a process can
broadcast a message which is received by all of its neighbors, and the set of
neighbors of a process can change arbitrarily over time. In ASMS, the processes
communicate by shared memory: a process can either write to or read from a
shared register. Our main result is that RBN and ASMS can simulate each other,
i.e. they are equivalent with respect to parameterized reachability, where we
are given two (possibly infinite) sets of configurations C and C' defined by
upper and lower bounds on the number of processes in each state and we would
like to decide if some configuration in C can reach some configuration in C'.
Using this simulation equivalence, we transfer results of RBN to ASMS and vice
versa. Finally, we show that RBN and ASMS can simulate a third distributed
model called immediate observation (IO) nets. Moreover, for a slightly stronger
notion of simulation (which is satisfied by all the simulations given in this
paper), we show that IO nets cannot simulate RBN.

    

### [[2109.08329] Cross-layer Visualization and Profiling of Network and I/O Communication for HPC Clusters](http://arxiv.org/abs/2109.08329)


  Understanding and visualizing the full-stack performance trade-offs and
interplay between HPC applications, MPI libraries, the communication fabric,
and the file system is a challenging endeavor. Designing a holistic profiling
and visualization method for HPC communication networks is challenging since
different levels of communication coexist and interact with each other on the
communication fabric. A breakdown of traffic is essential to understand the
interplay of different layers along with the application's communication
behavior without losing a general view of network traffic. Unfortunately,
existing profiling tools are disjoint and either focus on only profiling and
visualizing a few levels of the HPC stack, which limits the insights they can
provide, or they provide extremely detailed information which necessitates a
steep learning curve to understand. We target our profiling tool visualization
to provide holistic and real-time insights into HPC communication stacks.
In this paper, we propose and implement our visualization methods to enable
holistic insight for representing the cross-stack metrics. Moreover, we propose
and implement a low-overhead I/O profiling inside the communication library,
collect and store the profiling information, and then study the correlation and
evaluation of I/O traffic with MPI communication using a cross-stack approach
by INAM. Through experimental evaluations and use cases, we demonstrate novel
benefits of our cross-stack communication analysis in real-time to detect
bottlenecks and understand communication performance.

    

### [[2109.08358] Security Analysis of Distributed Ledgers and Blockchains through Agent-based Simulation](http://arxiv.org/abs/2109.08358)


  In this paper we describe LUNES-Blockchain, an agent-based simulator of
blockchains that relies on Parallel and Distributed Simulation (PADS)
techniques to obtain high scalability. The software is organized as a
multi-level simulator that permits to simulate a virtual environment, made of
many nodes running the protocol of a specific Distributed Ledger Technology
(DLT), such as the Bitcoin or the Ethereum blockchains. This virtual
environment is executed on top of a lower-level Peer-to-Peer (P2P) network
overlay, which can be structured based on different topologies and with a given
number of nodes and edges. Functionalities at different levels of abstraction
are managed separately, by different software modules and with different time
granularity. This allows for accurate simulations, where (and when) it is
needed, and enhances the simulation performance. Using LUNES-Blockchain, it is
possible to simulate different types of attacks on the DLT. In this paper, we
specifically focus on the P2P layer, considering the selfish mining, the 51%
attack and the Sybil attack. For which concerns selfish mining and the 51%
attack, our aim is to understand how much the hash-rate (i.e. a general measure
of the processing power in the blockchain network) of the attacker can
influence the outcome of the misbehaviour. On the other hand, in the filtering
denial of service (i.e. Sybil Attack), we investigate which dissemination
protocol in the underlying P2P network makes the system more resilient to a
varying number of nodes that drop the messages. The results confirm the
viability of the simulation-based techniques for the investigation of security
aspects of DLTs.

    

### [[2109.08566] GLASS: Towards Secure and Decentralized eGovernance Services using IPFS](http://arxiv.org/abs/2109.08566)


  The continuously advancing digitization has provided answers to the
bureaucratic problems faced by eGovernance services. This innovation led them
to an era of automation it has broadened the attack surface and made them a
popular target for cyber attacks. eGovernance services utilize internet, which
is currently a location addressed system where whoever controls the location
controls not only the content itself, but the integrity of that content, and
the access to that content. We propose GLASS, a decentralised solution which
combines the InterPlanetary File System (IPFS) with Distributed Ledger
technology and Smart Contracts to secure EGovernance services. We also create a
testbed environment where we measure the IPFS performance.

    

### [[2109.08611] Relaxed Reliable Broadcast for Decentralized Trust](http://arxiv.org/abs/2109.08611)


  Reliable broadcast is a fundamental primitive, widely used as a building
block for data replication in distributed systems. Informally, it ensures that
system members deliver the same values, even in the presence of equivocating
Byzantine participants. Classical broadcast protocols are based on centralized
(globally known) trust assumptions defined via sets of participants (quorums)
that are likely not to fail in system executions. In this paper, we consider
the reliable broadcast abstraction in decentralized trust settings, where every
system participant chooses its quorums locally. We introduce a class of relaxed
reliable broadcast abstractions that perfectly match these settings. We then
describe a broadcast protocol that achieves optimal consistency, measured as
the maximal number of different values from the same source that the system
members may deliver. In particular, we establish how this optimal consistency
is related to parameters of a graph representation of decentralized trust
assumptions.

    

### [[2109.08149] Karpov's Queen Sacrifices and AI](http://arxiv.org/abs/2109.08149)


  Anatoly Karpov's Queen sacrifices are analyzed. Stockfish 14 NNUE -- an AI
chess engine -- evaluates how efficient Karpov's sacrifices are. For
comparative purposes, we provide a dataset on Karpov's Rook and Knight
sacrifices to test whether Karpov achieves a similar level of accuracy. Our
study has implications for human-AI interaction and how humans can better
understand the strategies employed by black-box AI algorithms. Finally, we
conclude with implications for human study in. chess with computer engines.

    

### [[2109.08207] Numerical reasoning in machine reading comprehension tasks: are we there yet?](http://arxiv.org/abs/2109.08207)


  Numerical reasoning based machine reading comprehension is a task that
involves reading comprehension along with using arithmetic operations such as
addition, subtraction, sorting, and counting. The DROP benchmark (Dua et al.,
2019) is a recent dataset that has inspired the design of NLP models aimed at
solving this task. The current standings of these models in the DROP
leaderboard, over standard metrics, suggest that the models have achieved
near-human performance. However, does this mean that these models have learned
to reason? In this paper, we present a controlled study on some of the
top-performing model architectures for the task of numerical reasoning. Our
observations suggest that the standard metrics are incapable of measuring
progress towards such tasks.

    

### [[2109.08214] Hierarchical Control of Situated Agents through Natural Language](http://arxiv.org/abs/2109.08214)


  When humans conceive how to perform a particular task, they do so
hierarchically: splitting higher-level tasks into smaller sub-tasks. However,
in the literature on natural language (NL) command of situated agents, most
works have treated the procedures to be executed as flat sequences of simple
actions, or any hierarchies of procedures have been shallow at best. In this
paper, we propose a formalism of procedures as programs, a powerful yet
intuitive method of representing hierarchical procedural knowledge for agent
command and control. We further propose a modeling paradigm of hierarchical
modular networks, which consist of a planner and reactors that convert NL
intents to predictions of executable programs and probe the environment for
information necessary to complete the program execution. We instantiate this
framework on the IQA and ALFRED datasets for NL instruction following. Our
model outperforms reactive baselines by a large margin on both datasets. We
also demonstrate that our framework is more data-efficient, and that it allows
for fast iterative development.

    

### [[2109.08238] Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI](http://arxiv.org/abs/2109.08238)


  We present the Habitat-Matterport 3D (HM3D) dataset. HM3D is a large-scale
dataset of 1,000 building-scale 3D reconstructions from a diverse set of
real-world locations. Each scene in the dataset consists of a textured 3D mesh
reconstruction of interiors such as multi-floor residences, stores, and other
private indoor spaces.
HM3D surpasses existing datasets available for academic research in terms of
physical scale, completeness of the reconstruction, and visual fidelity. HM3D
contains 112.5k m^2 of navigable space, which is 1.4 - 3.7x larger than other
building-scale datasets such as MP3D and Gibson. When compared to existing
photorealistic 3D datasets such as Replica, MP3D, Gibson, and ScanNet, images
rendered from HM3D have 20 - 85% higher visual fidelity w.r.t. counterpart
images captured with real cameras, and HM3D meshes have 34 - 91% fewer
artifacts due to incomplete surface reconstruction.
The increased scale, fidelity, and diversity of HM3D directly impacts the
performance of embodied AI agents trained using it. In fact, we find that HM3D
is `pareto optimal' in the following sense -- agents trained to perform
PointGoal navigation on HM3D achieve the highest performance regardless of
whether they are evaluated on HM3D, Gibson, or MP3D. No similar claim can be
made about training on other datasets. HM3D-trained PointNav agents achieve
100% performance on Gibson-test dataset, suggesting that it might be time to
retire that episode dataset.

    

### [[2109.08256] Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis](http://arxiv.org/abs/2109.08256)


  The importance and pervasiveness of emotions in our lives makes affective
computing a tremendously important and vibrant line of work. Systems for
automatic emotion recognition (AER) and sentiment analysis can be facilitators
of enormous progress (e.g., in improving public health and commerce) but also
enablers of great harm (e.g., for suppressing dissidents and manipulating
voters). Thus, it is imperative that the affective computing community actively
engage with the ethical ramifications of their creations. In this paper, I have
synthesized and organized information from AI Ethics and Emotion Recognition
literature to present fifty ethical considerations relevant to AER. Notably,
the sheet fleshes out assumptions hidden in how AER is commonly framed, and in
the choices often made regarding the data, method, and evaluation. Special
attention is paid to the implications of AER on privacy and social groups. The
objective of the sheet is to facilitate and encourage more thoughtfulness on
why to automate, how to automate, and how to judge success well before the
building of AER systems. Additionally, the sheet acts as a useful introductory
document on emotion recognition (complementing survey articles).

    

### [[2109.08270] Language Models as a Knowledge Source for Cognitive Agents](http://arxiv.org/abs/2109.08270)


  Language models (LMs) are sentence-completion engines trained on massive
corpora. LMs have emerged as a significant breakthrough in natural-language
processing, providing capabilities that go far beyond sentence completion
including question answering, summarization, and natural-language inference.
While many of these capabilities have potential application to cognitive
systems, exploiting language models as a source of task knowledge, especially
for task learning, offers significant, near-term benefits. We introduce
language models and the various tasks to which they have been applied and then
review methods of knowledge extraction from language models. The resulting
analysis outlines both the challenges and opportunities for using language
models as a new knowledge source for cognitive systems. It also identifies
possible ways to improve knowledge extraction from language models using the
capabilities provided by cognitive systems. Central to success will be the
ability of a cognitive agent to itself learn an abstract model of the knowledge
implicit in the LM as well as methods to extract high-quality knowledge
effectively and efficiently. To illustrate, we introduce a hypothetical robot
agent and describe how language models could extend its task knowledge and
improve its performance and the kinds of knowledge and methods the agent can
use to exploit the knowledge within a language model.

    

### [[2109.08273] ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning](http://arxiv.org/abs/2109.08273)


  Effective robot learning often requires online human feedback and
interventions that can cost significant human time, giving rise to the central
challenge in interactive imitation learning: is it possible to control the
timing and length of interventions to both facilitate learning and limit burden
on the human supervisor? This paper presents ThriftyDAgger, an algorithm for
actively querying a human supervisor given a desired budget of human
interventions. ThriftyDAgger uses a learned switching policy to solicit
interventions only at states that are sufficiently (1) novel, where the robot
policy has no reference behavior to imitate, or (2) risky, where the robot has
low confidence in task completion. To detect the latter, we introduce a novel
metric for estimating risk under the current robot policy. Experiments in
simulation and on a physical cable routing experiment suggest that
ThriftyDAgger's intervention criteria balances task performance and supervisor
burden more effectively than prior algorithms. ThriftyDAgger can also be
applied at execution time, where it achieves a 100% success rate on both the
simulation and physical tasks. A user study (N=10) in which users control a
three-robot fleet while also performing a concentration task suggests that
ThriftyDAgger increases human and robot performance by 58% and 80% respectively
compared to the next best algorithm while reducing supervisor burden.

    

### [[2109.08279] Automata Techniques for Temporal Answer Set Programming](http://arxiv.org/abs/2109.08279)


  Temporal and dynamic extensions of Answer Set Programming (ASP) have played
an important role in addressing dynamic problems, as they allow the use of
temporal operators to reason with dynamic scenarios in a very effective way. In
my Ph.D. research, I intend to exploit the relationship between automata theory
and dynamic logic to add automata-based techniques to the ASP solver CLINGO
helping us to deal with theses type of problems.

    

### [[2109.08281] Quantitative and Stream Extensions of Answer Set Programming](http://arxiv.org/abs/2109.08281)


  Answer Set Programming has separately been extended with constraints, to the
streaming domain, and with capabilities to reason over the quantities
associated with answer sets. We propose the introduction and analysis of a
general framework that incorporates all three directions of extension by
exploiting the strengths of Here-and-There Logic and Weighted Logic.

    

### [[2109.08283] Syntactic Requirements for Well-defined Hybrid Probabilistic Logic Programs](http://arxiv.org/abs/2109.08283)


  Hybrid probabilistic logic programs can represent several scenarios thanks to
the expressivity of Logic Programming extended with facts representing discrete
and continuous distributions. The semantics for this type of programs is
crucial since it ensures that a probability can be assigned to every query.
Here, following one recent semantics proposal, we illustrate a concrete syntax,
and we analyse the syntactic requirements needed to preserve the
well-definedness.

    

### [[2109.08284] How to Split a Logic Program](http://arxiv.org/abs/2109.08284)


  Answer Set Programming (ASP) is a successful method for solving a range of
real-world applications. Despite the availability of fast ASP solvers,
computing answer sets demands a very large computational power, since the
problem tackled is in the second level of the polynomial hierarchy. A speed-up
in answer set computation may be attained, if the program can be split into two
disjoint parts, bottom and top. Thus, the bottom part is evaluated
independently of the top part, and the results of the bottom part evaluation
are used to simplify the top part. Lifschitz and Turner have introduced the
concept of a splitting set, i.e., a set of atoms that defines the splitting.
In this paper, We show that the problem of computing a splitting set with
some desirable properties can be reduced to a classic Search Problem and solved
in polynomial time. This allows us to conduct experiments on the size of the
splitting set in various programs and lead to an interesting discovery of a
source of complication in stable model computation. We also show that for
Head-Cycle-Free programs, the definition of splitting sets can be adjusted to
allow splitting of a broader class of programs.

    

### [[2109.08285] Fixpoint Semantics for Recursive SHACL](http://arxiv.org/abs/2109.08285)


  SHACL is a W3C-proposed language for expressing structural constraints on RDF
graphs. The recommendation only specifies semantics for non-recursive SHACL;
recently, some efforts have been made to allow recursive SHACL schemas. In this
paper, we argue that for defining and studying semantics of recursive SHACL,
lessons can be learned from years of research in non-monotonic reasoning. We
show that from a SHACL schema, a three-valued semantic operator can directly be
obtained. Building on Approximation Fixpoint Theory (AFT), this operator
immediately induces a wide variety of semantics, including a supported, stable,
and well-founded semantics, related in the expected ways. By building on AFT, a
rich body of theoretical results becomes directly available for SHACL. As such,
the main contribution of this short paper is providing theoretical foundations
for the study of recursive SHACL, which can later enable an informed decision
for an extension of the W3C recommendation.

    

### [[2109.08286] Weighted Conditional EL{^}bot Knowledge Bases with Integer Weights: an ASP Approach](http://arxiv.org/abs/2109.08286)


  Weighted knowledge bases for description logics with typicality have been
recently considered under a "concept-wise" multipreference semantics (in both
the two-valued and fuzzy case), as the basis of a logical semantics of
Multilayer Perceptrons. In this paper we consider weighted conditional EL^bot
knowledge bases in the two-valued case, and exploit ASP and asprin for encoding
concept-wise multipreference entailment for weighted KBs with integer weights.

    

### [[2109.08289] Refining the Semantics of Epistemic Specifications](http://arxiv.org/abs/2109.08289)


  Answer set programming (ASP) is an efficient problem-solving approach, which
has been strongly supported both scientifically and technologically by several
solvers, ongoing active research, and implementations in many different fields.
However, although researchers acknowledged long ago the necessity of epistemic
operators in the language of ASP for better introspective reasoning, this
research venue did not attract much attention until recently. Moreover, the
existing epistemic extensions of ASP in the literature are not widely approved
either, due to the fact that some propose unintended results even for some
simple acyclic epistemic programs, new unexpected results may possibly be
found, and more importantly, researchers have different reasonings for some
critical programs. To that end, Cabalar et al. have recently identified some
structural properties of epistemic programs to formally support a possible
semantics proposal of such programs and standardise their results. Nonetheless,
the soundness of these properties is still under debate, and they are not
widely accepted either by the ASP community. Thus, it seems that there is still
time to really understand the paradigm, have a mature formalism, and determine
the principles providing formal justification of their understandable models.
In this paper, we mainly focus on the existing semantics approaches, the
criteria that a satisfactory semantics is supposed to satisfy, and the ways to
improve them. We also extend some well-known propositions of here-and-there
logic (HT) into epistemic HT so as to reveal the real behaviour of programs.
Finally, we propose a slightly novel semantics for epistemic ASP, which can be
considered as a reflexive extension of Cabalar et al.'s recent formalism called
autoepistemic ASP.

    

### [[2109.08290] Generating Explainable Rule Sets from Tree-Ensemble Learning Methods by Answer Set Programming](http://arxiv.org/abs/2109.08290)


  We propose a method for generating explainable rule sets from tree-ensemble
learners using Answer Set Programming (ASP). To this end, we adopt a
decompositional approach where the split structures of the base decision trees
are exploited in the construction of rules, which in turn are assessed using
pattern mining methods encoded in ASP to extract interesting rules. We show how
user-defined constraints and preferences can be represented declaratively in
ASP to allow for transparent and flexible rule set generation, and how rules
can be used as explanations to help the user better understand the models.
Experimental evaluation with real-world datasets and popular tree-ensemble
algorithms demonstrates that our approach is applicable to a wide range of
classification tasks.

    

### [[2109.08292] exp(ASPc) : Explaining ASP Programs with Choice Atoms and Constraint Rules](http://arxiv.org/abs/2109.08292)


  We present an enhancement of exp(ASP), a system that generates explanation
graphs for a literal l - an atom a or its default negation ~a - given an answer
set A of a normal logic program P, which explain why l is true (or false) given
A and P. The new system, exp(ASPc), differs from exp(ASP) in that it supports
choice rules and utilizes constraint rules to provide explanation graphs that
include information about choices and constraints.

    

### [[2109.08293] Modeling and Solving Graph Synthesis Problems Using SAT-Encoded Reachability Constraints in Picat](http://arxiv.org/abs/2109.08293)


  Many constraint satisfaction problems involve synthesizing subgraphs that
satisfy certain reachability constraints. This paper presents programs in Picat
for four problems selected from the recent LP/CP programming competitions. The
programs demonstrate the modeling capabilities of the Picat language and the
solving efficiency of the cutting-edge SAT solvers empowered with effective
encodings.

    

### [[2109.08297] DiscASP: A Graph-based ASP System for Finding Relevant Consistent Concepts with Applications to Conversational Socialbots](http://arxiv.org/abs/2109.08297)


  We consider the problem of finding relevant consistent concepts in a
conversational AI system, particularly, for realizing a conversational
socialbot. Commonsense knowledge about various topics can be represented as an
answer set program. However, to advance the conversation, we need to solve the
problem of finding relevant consistent concepts, i.e., find consistent
knowledge in the "neighborhood" of the current topic being discussed that can
be used to advance the conversation. Traditional ASP solvers will generate the
whole answer set which is stripped of all the associations between the various
atoms (concepts) and thus cannot be used to find relevant consistent concepts.
Similarly, goal-directed implementations of ASP will only find concepts
directly relevant to a query. We present the DiscASP system that will find the
partial consistent model that is relevant to a given topic in a manner similar
to how a human will find it. DiscASP is based on a novel graph-based algorithm
for finding stable models of an answer set program. We present the DiscASP
algorithm, its implementation, and its application to developing a
conversational socialbot.

    

### [[2109.08298] Generating Concurrent Programs From Sequential Data Structure Knowledge Using Answer Set Programming](http://arxiv.org/abs/2109.08298)


  We tackle the problem of automatically designing concurrent data structure
operations given a sequential data structure specification and knowledge about
concurrent behavior. Designing concurrent code is a non-trivial task even in
simplest of cases. Humans often design concurrent data structure operations by
transforming sequential versions into their respective concurrent versions.
This requires an understanding of the data structure, its sequential behavior,
thread interactions during concurrent execution and shared memory
synchronization primitives. We mechanize this design process using automated
commonsense reasoning. We assume that the data structure description is
provided as axioms alongside the sequential code of its algebraic operations.
This information is used to automatically derive concurrent code for that data
structure, such as dictionary operations for linked lists and binary search
trees. Knowledge in our case is expressed using Answer Set Programming (ASP),
and we employ deduction and abduction -- just as humans do -- in the reasoning
involved. ASP allows for succinct modeling of first order theories of pointer
data structures, run-time thread interactions and shared memory
synchronization. Our reasoner can systematically make the same judgments as a
human reasoner, while constructing provably safe concurrent code. We present
several reasoning challenges involved in transforming the sequential data
structure into its equivalent concurrent version. All the reasoning tasks are
encoded in ASP and our reasoner can make sound judgments to transform
sequential code into concurrent code. To the best of our knowledge, our work is
the first one to use commonsense reasoning to automatically transform
sequential programs into concurrent code. We also have developed a tool that we
describe that relies on state-of-the-art ASP solvers and performs the reasoning
tasks involved to generate concurrent code.

    

### [[2109.08299] Flexible and Explainable Solutions for Multi-Agent Path Finding Problems](http://arxiv.org/abs/2109.08299)


  The multi-agent path finding (MAPF) problem is a combinatorial search problem
that aims at finding paths for multiple agents (e.g., robots) in an environment
(e.g., an autonomous warehouse) such that no two agents collide with each
other, and subject to some constraints on the lengths of paths. The real-world
applications of MAPF require flexibility (e.g., solving variations of MAPF) as
well as explainability. In this study, both of these challenges are addressed
and some flexible and explainable solutions for MAPF and its variants are
introduced.

    

### [[2109.08301] Comprehensive Multi-Agent Epistemic Planning](http://arxiv.org/abs/2109.08301)


  Over the last few years, the concept of Artificial Intelligence has become
central in different tasks concerning both our daily life and several working
scenarios. Among these tasks automated planning has always been central in the
AI research community. In particular, this manuscript is focused on a
specialized kind of planning known as Multi-agent Epistemic Planning (MEP).
Epistemic Planning (EP) refers to an automated planning setting where the agent
reasons in the space of knowledge/beliefs states and tries to find a plan to
reach a desirable state from a starting one. Its general form, the MEP problem,
involves multiple agents who need to reason about both the state of the world
and the information flows between agents. To tackle the MEP problem several
tools have been developed and, while the diversity of approaches has led to a
deeper understanding of the problem space, each proposed tool lacks some
abilities and does not allow for a comprehensive investigation of the
information flows. That is why, the objective of our work is to formalize an
environment where a complete characterization of the agents' knowledge/beliefs
interaction and update is possible. In particular, we aim to achieve such goal
by defining a new action-based language for multi-agent epistemic planning and
to implement an epistemic planner based on it. This solver should provide a
tool flexible enough to reason on different domains, e.g., economy, security,
justice and politics, where considering others' knowledge/beliefs could lead to
winning strategies.

    

### [[2109.08304] Product Configuration in Answer Set Programming](http://arxiv.org/abs/2109.08304)


  This is a preliminary work on configuration knowledge representation which
serves as a foundation for building interactive configuration systems in Answer
Set programming (ASP). The major concepts of the product configuration problem
are identified and discussed with a bike configuration example. A fact format
is developed for expressing product knowledge that is domain-specific and can
be mapped from other systems. Finally, a domain-independent ASP encoding is
provided that represents the concepts in the configuration problem.

    

### [[2109.08305] Formalisation of Action with Durations in Answer Set Programming](http://arxiv.org/abs/2109.08305)


  In this paper, I will discuss the work I am currently doing as a Ph.D.
student at the University of Potsdam, under the tutoring of T. Schaub. I'm
currently looking into action description in ASP. More precisely, my goal is to
explore how to represent actions with durations in ASP, in different contexts.
Right now, I'm focused on Multi-Agent Path Finding (MAPF), looking at how to
represent speeds for different agents and contexts.
Before tackling duration, I wanted to explore and compare different
representations of action taking in ASP. For this, I started comparing
different simple encodings tackling the MAPF problem. Even in simple code,
choices and assumptions have been made in their creations. The objective of my
work is to present the consequences of those design decisions in terms of
performance and knowledge representation. As far as I know, there is no current
research on this topic.
Besides that, I'm also exploring different ways to represent duration and to
solve related problems. I planed to compare them the same way I described
before. I also want this to help me find innovative and effective ways to solve
problems with duration.

    

### [[2109.08306] SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis](http://arxiv.org/abs/2109.08306)


  Aspect-based sentiment analysis (ABSA) is an emerging fine-grained sentiment
analysis task that aims to extract aspects, classify corresponding sentiment
polarities and find opinions as the causes of sentiment. The latest research
tends to solve the ABSA task in a unified way with end-to-end frameworks. Yet,
these frameworks get fine-tuned from downstream tasks without any task-adaptive
modification. Specifically, they do not use task-related knowledge well or
explicitly model relations between aspect and opinion terms, hindering them
from better performance. In this paper, we propose SentiPrompt to use sentiment
knowledge enhanced prompts to tune the language model in the unified framework.
We inject sentiment knowledge regarding aspects, opinions, and polarities into
prompt and explicitly model term relations via constructing consistency and
polarity judgment templates from the ground truth triplets. Experimental
results demonstrate that our approach can outperform strong baselines on
Triplet Extraction, Pair Extraction, and Aspect Term Extraction with Sentiment
Classification by a notable margin.

    

### [[2109.08307] Sinoledge: A Knowledge Engine based on Logical Reasoning and Distributed Micro Services](http://arxiv.org/abs/2109.08307)


  We propose a knowledge engine called Sinoledge mainly for doctors,
physicians, and researchers in medical field to organize thoughts, manage
reasoning process, test and deploy to production environments effortlessly. Our
proposal can be related to rule engine usually used in business or medical
fields. More importantly, our proposal provides a user-friendly interface, an
easy-maintain way of organizing knowledge, an understandable testing
functionality and a highly available and efficient back-end architecture.

    

### [[2109.08330] Mass Segmentation in Automated 3-D Breast Ultrasound Using Dual-Path U-net](http://arxiv.org/abs/2109.08330)


  Automated 3-D breast ultrasound (ABUS) is a newfound system for breast
screening that has been proposed as a supplementary modality to mammography for
breast cancer detection. While ABUS has better performance in dense breasts,
reading ABUS images is exhausting and time-consuming. So, a computer-aided
detection system is necessary for interpretation of these images. Mass
segmentation plays a vital role in the computer-aided detection systems and it
affects the overall performance. Mass segmentation is a challenging task
because of the large variety in size, shape, and texture of masses. Moreover,
an imbalanced dataset makes segmentation harder. A novel mass segmentation
approach based on deep learning is introduced in this paper. The deep network
that is used in this study for image segmentation is inspired by U-net, which
has been used broadly for dense segmentation in recent years. The system's
performance was determined using a dataset of 50 masses including 38 malign and
12 benign lesions. The proposed segmentation method attained a mean Dice of
0.82 which outperformed a two-stage supervised edge-based method with a mean
Dice of 0.74 and an adaptive region growing method with a mean Dice of 0.65.

    

### [[2109.08379] PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering](http://arxiv.org/abs/2109.08379)


  Generating portrait images by controlling the motions of existing faces is an
important task of great consequence to social media industries. For easy use
and intuitive control, semantically meaningful and fully disentangled
parameters should be used as modifications. However, many existing techniques
do not provide such fine-grained controls or use indirect editing methods i.e.
mimic motions of other individuals. In this paper, a Portrait Image Neural
Renderer (PIRenderer) is proposed to control the face motions with the
parameters of three-dimensional morphable face models (3DMMs). The proposed
model can generate photo-realistic portrait images with accurate movements
according to intuitive modifications. Experiments on both direct and indirect
editing tasks demonstrate the superiority of this model. Meanwhile, we further
extend this model to tackle the audio-driven facial reenactment task by
extracting sequential motions from audio inputs. We show that our model can
generate coherent videos with convincing movements from only a single reference
image and a driving audio stream. Our source code is available at
this https URL.

    

### [[2109.08411] Cross Modification Attention Based Deliberation Model for Image Captioning](http://arxiv.org/abs/2109.08411)


  The conventional encoder-decoder framework for image captioning generally
adopts a single-pass decoding process, which predicts the target descriptive
sentence word by word in temporal order. Despite the great success of this
framework, it still suffers from two serious disadvantages. Firstly, it is
unable to correct the mistakes in the predicted words, which may mislead the
subsequent prediction and result in error accumulation problem. Secondly, such
a framework can only leverage the already generated words but not the possible
future words, and thus lacks the ability of global planning on linguistic
information. To overcome these limitations, we explore a universal two-pass
decoding framework, where a single-pass decoding based model serving as the
Drafting Model first generates a draft caption according to an input image, and
a Deliberation Model then performs the polishing process to refine the draft
caption to a better image description. Furthermore, inspired from the
complementarity between different modalities, we propose a novel Cross
Modification Attention (CMA) module to enhance the semantic expression of the
image features and filter out error information from the draft captions. We
integrate CMA with the decoder of our Deliberation Model and name it as Cross
Modification Attention based Deliberation Model (CMA-DM). We train our proposed
framework by jointly optimizing all trainable components from scratch with a
trade-off coefficient. Experiments on MS COCO dataset demonstrate that our
approach obtains significant improvements over single-pass decoding baselines
and achieves competitive performances compared with other state-of-the-art
two-pass decoding based methods.

    

### [[2109.08425] Repurposing of Resources: from Everyday Problem Solving through to Crisis Management](http://arxiv.org/abs/2109.08425)


  The human ability to repurpose objects and processes is universal, but it is
not a well-understood aspect of human intelligence. Repurposing arises in
everyday situations such as finding substitutes for missing ingredients when
cooking, or for unavailable tools when doing DIY. It also arises in critical,
unprecedented situations needing crisis management. After natural disasters and
during wartime, people must repurpose the materials and processes available to
make shelter, distribute food, etc. Repurposing is equally important in
professional life (e.g. clinicians often repurpose medicines off-license) and
in addressing societal challenges (e.g. finding new roles for waste products,).
Despite the importance of repurposing, the topic has received little academic
attention. By considering examples from a variety of domains such as every-day
activities, drug repurposing and natural disasters, we identify some principle
characteristics of the process and describe some technical challenges that
would be involved in modelling and simulating it. We consider cases of both
substitution, i.e. finding an alternative for a missing resource, and
exploitation, i.e. identifying a new role for an existing resource. We argue
that these ideas could be developed into general formal theory of repurposing,
and that this could then lead to the development of AI methods based on
commonsense reasoning, argumentation, ontological reasoning, and various
machine learning methods, to develop tools to support repurposing in practice.

    

### [[2109.08473] Carl-Lead: Lidar-based End-to-End Autonomous Driving with Contrastive Deep Reinforcement Learning](http://arxiv.org/abs/2109.08473)


  Autonomous driving in urban crowds at unregulated intersections is
challenging, where dynamic occlusions and uncertain behaviors of other vehicles
should be carefully considered. Traditional methods are heuristic and based on
hand-engineered rules and parameters, but scale poorly in new situations.
Therefore, they require high labor cost to design and maintain rules in all
foreseeable scenarios. Recently, deep reinforcement learning (DRL) has shown
promising results in urban driving scenarios. However, DRL is known to be
sample inefficient, and most previous works assume perfect observations such as
ground-truth locations and motions of vehicles without considering noises and
occlusions, which might be a too strong assumption for policy deployment. In
this work, we use DRL to train lidar-based end-to-end driving policies that
naturally consider imperfect partial observations. We further use unsupervised
contrastive representation learning as an auxiliary task to improve the sample
efficiency. The comparative evaluation results reveal that our method achieves
higher success rates than the state-of-the-art (SOTA) lidar-based end-to-end
driving network, better trades off safety and efficiency than the carefully
tuned rule-based method, and generalizes better to new scenarios than the
baselines. Demo videos are available at this https URL.

    

### [[2109.08521] Focus on Impact: Indoor Exploration with Intrinsic Motivation](http://arxiv.org/abs/2109.08521)


  Exploration of indoor environments has recently experienced a significant
interest, also thanks to the introduction of deep neural agents built in a
hierarchical fashion and trained with Deep Reinforcement Learning (DRL) on
simulated environments. Current state-of-the-art methods employ a dense
extrinsic reward that requires the complete a priori knowledge of the layout of
the training environment to learn an effective exploration policy. However,
such information is expensive to gather in terms of time and resources. In this
work, we propose to train the model with a purely intrinsic reward signal to
guide exploration, which is based on the impact of the robot's actions on the
environment. So far, impact-based rewards have been employed for simple tasks
and in procedurally generated synthetic environments with countable states.
Since the number of states observable by the agent in realistic indoor
environments is non-countable, we include a neural-based density model and
replace the traditional count-based regularization with an estimated
pseudo-count of previously visited states. The proposed exploration approach
outperforms DRL-based competitors relying on intrinsic rewards and surpasses
the agents trained with a dense extrinsic reward computed with the environment
layouts. We also show that a robot equipped with the proposed approach
seamlessly adapts to point-goal navigation and real-world deployment.

    

### [[2109.08523] A Computable Piece of Uncomputable Art whose Expansion May Explain the Universe in Software Space](http://arxiv.org/abs/2109.08523)


  At the intersection of what I call uncomputable art and computational
epistemology, a form of experimental philosophy, we find an exciting and
promising area of science related to causation with an alternative, possibly
best possible, solution to the challenge of the inverse problem. That is the
problem of finding the possible causes, mechanistic origins, first principles,
and generative models of a piece of data from a physical phenomenon. Here we
explain how generating and exploring software space following the framework of
Algorithmic Information Dynamics, it is possible to find small models and learn
to navigate a sci-fi-looking space that can advance the field of scientific
discovery with complementary tools to offer an opportunity to advance science
itself.

    

### [[2109.08585] Hierarchy-Aware T5 with Path-Adaptive Mask Mechanism for Hierarchical Text Classification](http://arxiv.org/abs/2109.08585)


  Hierarchical Text Classification (HTC), which aims to predict text labels
organized in hierarchical space, is a significant task lacking in investigation
in natural language processing. Existing methods usually encode the entire
hierarchical structure and fail to construct a robust label-dependent model,
making it hard to make accurate predictions on sparse lower-level labels and
achieving low Macro-F1. In this paper, we propose a novel PAMM-HiA-T5 model for
HTC: a hierarchy-aware T5 model with path-adaptive mask mechanism that not only
builds the knowledge of upper-level labels into low-level ones but also
introduces path dependency information in label prediction. Specifically, we
generate a multi-level sequential label structure to exploit hierarchical
dependency across different levels with Breadth-First Search (BFS) and T5
model. To further improve label dependency prediction within each path, we then
propose an original path-adaptive mask mechanism (PAMM) to identify the label's
path information, eliminating sources of noises from other paths. Comprehensive
experiments on three benchmark datasets show that our novel PAMM-HiA-T5 model
greatly outperforms all state-of-the-art HTC approaches especially in Macro-F1.
The ablation studies show that the improvements mainly come from our innovative
approach instead of T5.

    

### [[2109.08621] Data-Driven Off-Policy Estimator Selection: An Application in User Marketing on An Online Content Delivery Service](http://arxiv.org/abs/2109.08621)


  Off-policy evaluation (OPE) is the method that attempts to estimate the
performance of decision making policies using historical data generated by
different policies without conducting costly online A/B tests. Accurate OPE is
essential in domains such as healthcare, marketing or recommender systems to
avoid deploying poor performing policies, as such policies may hart human lives
or destroy the user experience. Thus, many OPE methods with theoretical
backgrounds have been proposed. One emerging challenge with this trend is that
a suitable estimator can be different for each application setting. It is often
unknown for practitioners which estimator to use for their specific
applications and purposes. To find out a suitable estimator among many
candidates, we use a data-driven estimator selection procedure for off-policy
policy performance estimators as a practical solution. As proof of concept, we
use our procedure to select the best estimator to evaluate coupon treatment
policies on a real-world online content delivery service. In the experiment, we
first observe that a suitable estimator might change with different definitions
of the outcome variable, and thus the accurate estimator selection is critical
in real-world applications of OPE. Then, we demonstrate that, by utilizing the
estimator selection procedure, we can easily find out suitable estimators for
each purpose.

    

### [[2109.08634] Grounding Natural Language Instructions: Can Large Language Models Capture Spatial Information?](http://arxiv.org/abs/2109.08634)


  Models designed for intelligent process automation are required to be capable
of grounding user interface elements. This task of interface element grounding
is centred on linking instructions in natural language to their target
referents. Even though BERT and similar pre-trained language models have
excelled in several NLP tasks, their use has not been widely explored for the
UI grounding domain. This work concentrates on testing and probing the
grounding abilities of three different transformer-based models: BERT, RoBERTa
and LayoutLM. Our primary focus is on these models' spatial reasoning skills,
given their importance in this domain. We observe that LayoutLM has a promising
advantage for applications in this domain, even though it was created for a
different original purpose (representing scanned documents): the learned
spatial features appear to be transferable to the UI grounding setting,
especially as they demonstrate the ability to discriminate between target
directions in natural language instructions.

    

### [[2109.08642] Efficient State Representation Learning for Dynamic Robotic Scenarios](http://arxiv.org/abs/2109.08642)


  While the rapid progress of deep learning fuels end-to-end reinforcement
learning (RL), direct application, especially in high-dimensional space like
robotic scenarios still suffers from high sample efficiency. Therefore State
Representation Learning (SRL) is proposed to specifically learn to encode
task-relevant features from complex sensory data into low-dimensional states.
However, the pervasive implementation of SRL is usually conducted by a
decoupling strategy in which the observation-state mapping is learned
separately, which is prone to over-fit. To handle such problem, we present a
new algorithm called Policy Optimization via Abstract Representation which
integrates SRL into the original RL scale. Firstly, We engage RL loss to assist
in updating SRL model so that the states can evolve to meet the demand of
reinforcement learning and maintain a good physical interpretation. Secondly,
we introduce a dynamic parameter adjustment mechanism so that both models can
efficiently adapt to each other. Thirdly, we introduce a new prior called
domain resemblance to leverage expert demonstration to train the SRL model.
Finally, we provide a real-time access by state graph to monitor the course of
learning. Results show that our algorithm outperforms the PPO baselines and
decoupling strategies in terms of sample efficiency and final rewards. Thus our
model can efficiently deal with tasks in high dimensions and facilitate
training real-life robots directly from scratch.

    

### [[2109.08644] Allocating Indivisible Goods to Strategic Agents: Pure Nash Equilibria and Fairness](http://arxiv.org/abs/2109.08644)


  We consider the problem of fairly allocating a set of indivisible goods to a
set of strategic agents with additive valuation functions. We assume no
monetary transfers and, therefore, a mechanism in our setting is an algorithm
that takes as input the reported -- rather than the true -- values of the
agents. Our main goal is to explore whether there exist mechanisms that have
pure Nash equilibria for every instance and, at the same time, provide fairness
guarantees for the allocations that correspond to these equilibria. We focus on
two relaxations of envy-freeness, namely envy-freeness up to one good (EF1),
and envy-freeness up to any good (EFX), and we positively answer the above
question. In particular, we study two algorithms that are known to produce such
allocations in the non-strategic setting: Round-Robin (EF1 allocations for any
number of agents) and a cut-and-choose algorithm of Plaut and Roughgarden [SIAM
Journal of Discrete Mathematics, 2020] (EFX allocations for two agents). For
Round-Robin we show that all of its pure Nash equilibria induce allocations
that are EF1 with respect to the underlying true values, while for the
algorithm of Plaut and Roughgarden we show that the corresponding allocations
not only are EFX but also satisfy maximin share fairness, something that is not
true for this algorithm in the non-strategic setting! Further, we show that a
weaker version of the latter result holds for any mechanism for two agents that
always has pure Nash equilibria which all induce EFX allocations.

    

### [[2109.08662] Aggregate Semantics for Propositional Answer Set Programs](http://arxiv.org/abs/2109.08662)


  Answer Set Programming (ASP) emerged in the late 1990ies as a paradigm for
Knowledge Representation and Reasoning. The attractiveness of ASP builds on an
expressive high-level modeling language along with the availability of powerful
off-the-shelf solving systems. While the utility of incorporating aggregate
expressions in the modeling language has been realized almost simultaneously
with the inception of the first ASP solving systems, a general semantics of
aggregates and its efficient implementation have been long-standing challenges.
Aggregates have been proposed and widely used in database systems, and also in
the deductive database language Datalog, which is one of the main precursors of
ASP. The use of aggregates was, however, still restricted in Datalog (by either
disallowing recursion or only allowing monotone aggregates), while several ways
to integrate unrestricted aggregates evolved in the context of ASP. In this
survey, we pick up at this point of development by presenting and comparing the
main aggregate semantics that have been proposed for propositional ASP
programs. We highlight crucial properties such as computational complexity and
expressive power, and outline the capabilities and limitations of different
approaches by illustrative examples.

    

### [[2011.08523] Self-supervised Document Clustering Based on BERT with Data Augment](http://arxiv.org/abs/2011.08523)


  Contrastive learning is a promising approach to unsupervised learning, as it
inherits the advantages of well-studied deep models without a dedicated and
complex model design. In this paper, based on bidirectional encoder
representations from transformers, we propose self-supervised contrastive
learning (SCL) as well as few-shot contrastive learning (FCL) with unsupervised
data augmentation (UDA) for text clustering. SCL outperforms state-of-the-art
unsupervised clustering approaches for short texts and those for long texts in
terms of several clustering evaluation measures. FCL achieves performance close
to supervised learning, and FCL with UDA further improves the performance for
short texts.

    

### [[2109.08278] A Note on Occur-Check](http://arxiv.org/abs/2109.08278)


  Most known results on avoiding the occur-check are based on the notion of
"not subject to occur-check" (NSTO). It means that unification is performed
only on such pairs of atoms for which the occur-check never succeeds in any run
of a nondeterministic unification algorithm. Here we show that this requirement
is too strong. We show how to weaken it, and present some related sufficient
conditions under which the occur-check may be safely omitted. We show examples
for which the proposed approach provides more general results than the
approaches based on well-moded and nicely moded programs (this includes cases
to which the latter approaches are inapplicable).

    

### [[2012.13129] Rast: A Language for Resource-Aware Session Types](http://arxiv.org/abs/2012.13129)


  Traditional session types prescribe bidirectional communication protocols for
concurrent computations, where well-typed programs are guaranteed to adhere to
the protocols. However, simple session types cannot capture properties beyond
the basic type of the exchanged messages. In response, recent work has extended
session types with refinements from linear arithmetic, capturing intrinsic
attributes of processes and data. These refinements then play a central role in
describing sequential and parallel complexity bounds on session-typed programs.
The Rast language provides an open-source implementation of session-typed
concurrent programs extended with arithmetic refinements as well as ergometric
and temporal types to capture work and span of program execution. To further
support generic programming, Rast also enhances arithmetically refined session
types with recently developed nested parametric polymorphism. Type checking
relies on Cooper's algorithm for quantifier elimination in Presburger
arithmetic with a few significant optimizations, and a heuristic extension to
nonlinear constraints. Rast furthermore includes a reconstruction engine so
that most program constructs pertaining the layers of refinements and resources
are inserted automatically. We provide a variety of examples to demonstrate the
expressivity of the language.

    