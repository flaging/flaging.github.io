
## 2021-10-5

### [[2110.00631] The Programmable Data Plane: Abstractions, Architectures, Algorithms, and Applications](http://arxiv.org/abs/2110.00631)


  Programmable data plane technology enables the systematic reconfiguration of
the low-level processing steps applied to network packets and is a key driver
in realizing the next generation of network services and applications. This
survey presents recent trends and issues in the design and implementation of
programmable network devices, focusing on prominent architectures,
abstractions, algorithms, and applications proposed, debated, and realized over
the past years. We elaborate on the trends that led to the emergence of this
technology and highlight the most important pointers from the literature,
casting different taxonomies for the field and identifying avenues for future
research.

    

### [[2110.00772] Network Friendly Recommendations: Optimizing for Long Viewing Sessions](http://arxiv.org/abs/2110.00772)


  Caching algorithms try to predict content popularity, and place the content
closer to the users. Additionally, nowadays requests are increasingly driven by
recommendation systems (RS). These important trends, point to the following:
\emph{make RSs favor locally cached content}, this way operators reduce network
costs, and users get better streaming rates. Nevertheless, this process should
preserve the quality of the recommendations (QoR). In this work, we propose a
Markov Chain model for a stochastic, recommendation-driven \emph{sequence} of
requests, and formulate the problem of selecting high quality recommendations
that minimize the network cost \emph{in the long run}. While the original
optimization problem is non-convex, it can be convexified through a series of
transformations. Moreover, we extend our framework for users who show
preference in some positions of the recommendations' list. To our best
knowledge, this is the first work to provide an optimal polynomial-time
algorithm for these problems. Finally, testing our algorithms on real datasets
suggests significant potential, e.g., $2\times$ improvement compared to
baseline recommendations, and 80\% compared to a greedy network-friendly-RS
(which optimizes the cost for I.I.D. requests), while preserving at least 90\%
of the original QoR. Finally, we show that taking position preference into
account leads to additional performance gains.

    

### [[2110.00781] Terahertz Wireless Communications in Space](http://arxiv.org/abs/2110.00781)


  The New Space Era has increased communication traffic in space by new space
missions led by public space agencies and private companies. Mars colonization
is also targeted by crewed missions in the near future. Due to increasing space
traffic near Earth and Mars, the bandwidth is getting congested. Moreover, the
downlink performance of the current missions is not satisfactory in terms of
delay and data rate. Therefore, to meet the increasing demand in space links,
Terahertz band (0.1-10 THz) wireless communications are proposed in this study.
In line with this, we discuss the major challenges that the realization of THz
band space links pose and possible solutions. Moreover, we simulate Mars-space
THz links for the case of a clear Mars atmosphere, and a heavy dust storm to
show that even in the worst conditions, a large bandwidth is available for Mars
communication traffic.

    

### [[2110.00817] Age of Changed Information: Content-Aware Status Updating in the Internet of Things](http://arxiv.org/abs/2110.00817)


  In Internet of Things (IoT), the freshness of status updates is crucial for
mission-critical applications. In this regard, it is suggested to quantify the
freshness of updates by using Age of Information (AoI) from the receiver's
perspective. Specifically, the AoI measures the freshness over time. However,
the freshness in the content is neglected. In this paper, we introduce an
age-based utility, named as Age of Changed Information (AoCI), which captures
both the passage of time and the change of information content. By modeling the
underlying physical process as a discrete time Markov chain, we investigate the
AoCI in a time-slotted status update system, where a sensor samples the
physical process and transmits the update packets to the destination. With the
aim of minimizing the weighted sum of the AoCI and the update cost, we
formulate an infinite horizon average cost Markov Decision Process. We show
that the optimal updating policy has a special structure with respect to the
AoCI and identify the condition under which the special structure exists. By
exploiting the special structure, we provide a low complexity relative policy
iteration algorithm that finds the optimal updating policy. We further
investigate the optimal policy for two special cases. In the first case where
the state of the physical process transits with equiprobability, we show that
optimal policy is of threshold type and derive the closed-form of the optimal
threshold. We then study a more generalized periodic Markov model of the
physical process in the second case. Lastly, simulation results are laid out to
exhibit the performance of the optimal updating policy and its superiority over
the zero-wait baseline policy.

    

### [[2110.00851] Optimized Graph Based Routing Algorithm for the Angara Interconnect](http://arxiv.org/abs/2110.00851)


  JSC NICEVT has developed the Angara high-speed interconnect with 4D torus
topology. The Angara interconnect router implements deterministic routing based
on the bubble flow control, a direction order routing (DOR) and direction bits
rules. The router chip also supports non standard First Step / Last Step for
bypassing failed nodes and links, these steps can violate the DOR rule. In the
previous work we have proposed an algorithm for generation and analysis of
routing tables that guarantees no deadlocks in the Angara interconnect. It is
based on a breadth-first search algorithm in a graph and it practically does
not take into consideration communication channel load. Also we have never
evaluated the influence of routing table generation algorithm on the
performance of a real-world Angara based cluster. In this paper we present a
routing graph notation that provides a possibility to build routes in the torus
topology of the Angara interconnect. We propose a deadlock-free routing
algorithm based on a fast single-source shortest path algorithm for the
deterministic Angara routing with a single virtual channel. We evaluated the
considered routing algorithms on a 32-node Desmos cluster system and
benchmarked the proposed algorithm performance improvement of 11.1% for the
Alltoall communication pattern and of more than 5% for the FT and IS
application kernels.

    

### [[2110.01080] Fieldable Cross-Layer Optimized Embedded Software Defined Radio is Finally Here!](http://arxiv.org/abs/2110.01080)


  The concept of cross-layer optimization has been around for several years
now. The primary goal of the cross-layer approach was to liberate the strict
boundary between the layers of the traditional OSI protocol stack. This is to
enable information flow between layers which then can be leveraged to optimize
the network's performance across the layers. This concept has been of keen
interest for tactical application as there is an overwhelming requirement to
operate in a challenging and dynamic environment. The advent of software
defined radios (SDR) accelerated the growth of this domain due to the added
flexibility provided by SDRs. Even with the immense interest and progress in
this area of research, there has been a gaping abyss between solutions designed
in theory and ones deployed in practice. To the best of our knowledge, this is
the first time in literature, an embedded SDR has been leveraged to
successfully design a cross-layer optimized transceiver that provides high
throughput and high reliability in a ruggedized, weatherized, and fieldable
form-factor. The design ethos focuses on efficiency and flexibility such that
optimization objectives, cross-layer interactions can be reconfigured rapidly.
To demonstrate our claims, we provide results from extensive outdoor
over-the-air evaluation in various settings with up to 10-node network
typologies. The results demonstrate high reliability, throughput, and dynamic
routing capability achieving high technology readiness level (TRL) for tactical
applications.

    

### [[2110.01168] BLEnD: Improving NDN Performance Over Wireless Links Using Interest Bundling](http://arxiv.org/abs/2110.01168)


  Named Data Networking (NDN) employs small-sized Interest packets to retrieve
large-sized Data packets. Given the half-duplex nature of wireless links,
Interest packets frequently contend for the channel with Data packets, leading
to throughput degradation over wireless links. In this work, we present a novel
idea called BLEnD, an Interest-bundling technique that encodes multiple
Interests into one at the sender and decodes at the receiver. The major design
challenges are to reduce the number of Interest transmissions without impacting
the one-Interest one-Data principle embedded everywhere in NDN architecture and
implementation, and support flow/congestion control mechanisms that usually use
Interest packets as signals. BLEnD achieves these by bundling/unbundling
Interests at the link adaptation layer, keeping all NDN components unaware and
unaffected. Over a one-hop WiFi link, BLEnD improves application throughput by
30%. It may also be used over multiple hops and be improved in a number of
ways.

    

### [[1911.05164] A Reproducibility Study of "IP Spoofing Detection in Inter-Domain Traffic"](http://arxiv.org/abs/1911.05164)


  IP spoofing enables reflection and amplification attacks, which cause major
threats to the current Internet infrastructure. Detecting IP packets with
incorrect source addresses would help to improve the situation. This is easy at
the attacker's network, but very challenging at Internet eXchange Points (IXPs)
or in transit networks. In this reproducibility study, we revisit the paper
\textit{Detection, Classification, and Analysis of Inter-Domain Traffic with
Spoofed Source IP Addresses} published at ACM IMC 2017. Using data from a
different IXP and from a different time, we were not able to reproduce the
results. Unfortunately, our further analysis reveals structural problems of the
state of the art methodology, which are not easy to overcome.

    

### [[2012.14219] Enabling End-to-End Simulation for Host Networking Evaluation using SimBricks](http://arxiv.org/abs/2012.14219)


  Full system "end-to-end" measurements in physical testbeds are the gold
standard for evaluation of network systems but are fraught with challenges.
Adequate testbeds are often not available, as projects target next generation
devices, propose new hardware, or require larger scale. Further, evaluations in
testbeds limit what we can observe without affecting system behavior, are
frequently hard to reproduce, and are only available to groups with sufficient
funding. Yet, we lack an accepted alternative, leaving us with ad-hoc
non-end-to-end evaluations that do not form a solid basis for future work.
We argue that full system simulations enable comparable end-to-end evaluation
and are the next best alternative when a physical testbed is not available. To
this end, we present SimBricks, a modular full system simulation framework for
network systems. SimBricks combines multiple existing simulators for individual
components, including processor and memory, NIC, and network, into full virtual
testbeds running unmodified software. The architecture combines well-defined
component interfaces for extensibility with new simulators, efficient
communication channels for local and distributed simulators, and a novel
efficient synchronization protocol for accurate timing across different
simulators. We demonstrate that SimBricks simulations reproduce key findings
from prior work in congestion control, NIC architecture, and in-network
computing, and show scalability to 1000 simulated hosts.

    

### [[2103.02964] R-Learning Based Admission Control for Service Federation in Multi-domain 5G Networks](http://arxiv.org/abs/2103.02964)


  Service federation in 5G/B5G networks enables service providers to
orchestrate network services across multiple domains where admission control is
a key issue. For each demand, without knowing the future ones, the admission
controller either determines the domain to deploy the demand or rejects it in
order to maximize the long-term average profit. In this paper, at first, under
the assumption of knowing the arrival and departure rates of demands, we obtain
the optimal admission control policy by formulating the problem as a Markov
decision process that is solved by the policy iteration method. As a practical
solution, where the rates are not known, we apply the Q-Learning and R-Learning
algorithms to approximate the optimal policy. The extensive simulation results
show the learning approaches outperform the greedy policy, and while the
performance of Q-Learning depends on the discount factor, the optimality gap of
the R-Learning algorithm is at most 3-5% independent of the system
configuration.

    

### [[2105.07560] Dynamic Routing and Spectrum Assignment based on the Availability of Consecutive Sub-channels in Flexi-grid Optical Networks](http://arxiv.org/abs/2105.07560)


  Using Optical Orthogonal Frequency Multiplexing (O-OFDM), variable bandwidth
channels can be created in Elastic Optical Networks (EON). This allows the use
of spectrum more efficiently by allocating integral multiple of basic bandwidth
slots to the lightpath requests. Consequently, such networks are also called
flexible grid optical networks. It also adds a constraint of keeping all the
allocated slots together when deciding the routes for the requests. This
constraint called the contiguity constraint makes the routing and spectrum
algorithms more challenging. In any network, the lightpath requests will arrive
and depart dynamically and will invariably lead to spectrum fragmentation, and
hence network will have a reduction in maximum possible utilization due to
increased blocking probability. In this paper, we have presented an improvised
RSA algorithm that leads to lesser fragmentation. It is evident from the
results that the presented RSA algorithm uses adaptive parameters to reduce the
blocking probability and fragmentation compared to other algorithms reported in
the recent past.

    

### [[2106.09518] Multi-Layered Blockchain Governance Game](http://arxiv.org/abs/2106.09518)


  This paper deals with design of an integrated secure Blockchain network
framework to prevent damages from attackers. The multi-layer concept which
could handle multiple number of networks is adapted on the top of Blockchain
Governance Game frameworks. This new integrated theoretical model is designed
to find the best strategies toward preparation for preventing whole network
systems malfunction from attackers and it is developed based on the combination
of the Blockchain Governance Game and the Strategic Alliance for Blockchain
Governance Game. Analytically tractable results for executing a safety mode are
fully obtained and simulated results are demonstrated to obtain the optimal
values of hyper parameters of a Blockchain based security network. This
research helps those whom are constructing a multiple layer network by
enhancing security features through multi-layer framework in a decentralized
network.

    

### [[2106.13306] CEAZ: Accelerating Parallel I/O Via Hardware-Algorithm Co-Designed Adaptive Lossy Compression](http://arxiv.org/abs/2106.13306)


  As parallel computers continue to grow to exascale, the amount of data that
needs to be saved or transmitted is exploding. To this end, many previous works
have studied using error-bounded lossy compressors to reduce the data size and
improve the I/O performance. However, little work has been done for effectively
offloading lossy compression onto FPGA-based SmartNICs to reduce the
compression overhead. In this paper, we propose a hardware-algorithm co-design
for an efficient and adaptive lossy compressor for scientific data on FPGAs
(called CEAZ), which is the first lossy compressor that can achieve high
compression ratios and throughputs simultaneously. Specifically, we propose an
efficient Huffman coding approach that can adaptively update Huffman codewords
online based on codewords generated offline, from a variety of representative
scientific datasets. Moreover, we derive a theoretical analysis to support a
precise control of compression ratio under an error-bounded compression mode,
enabling accurate offline Huffman codewords generation. This also helps us
create a fixed-ratio compression mode for consistent throughput. In addition,
we develop an efficient compression pipeline by adopting cuSZ's
dual-quantization algorithm to our hardware use cases. Finally, we evaluate
CEAZ on five real-world datasets with both a single FPGA board and 128 nodes
(to accelerate parallel I/O). Experiments show that CEAZ outperforms the
second-best FPGA-based lossy compressor by 2X of throughput and 9.6X of ratio.
It also improves MPI_File_write and MPI_Gather throughputs by up to 28.1X and
36.9X, respectively.

    

### [[2107.01028] Universal Transceivers: Opportunities and Future Directions for the Internet of Everything (IoE)](http://arxiv.org/abs/2107.01028)


  The Internet of Everything (IoE) is a recently introduced information and
communication technology (ICT) framework promising for extending the human
connectivity to the entire universe, which itself can be regarded as a natural
IoE, an interconnected network of everything we perceive. The countless number
of opportunities that can be enabled by IoE through a blend of heterogeneous
ICT technologies across different scales and environments and a seamless
interface with the natural IoE impose several fundamental challenges, such as
interoperability, ubiquitous connectivity, energy efficiency, and
miniaturization. The key to address these challenges is to advance our
communication technology to match the multi-scale, multi-modal, and dynamic
features of the natural IoE. To this end, we introduce a new communication
device concept, namely the universal IoE transceiver, that encompasses
transceiver architectures that are characterized by multi-modality in
communication (with modalities such as molecular, RF/THz, optical and acoustic)
and in energy harvesting (with modalities such as mechanical, solar,
biochemical), modularity, tunability, and scalability. Focusing on these
fundamental traits, we provide an overview of the opportunities that can be
opened up by micro/nanoscale universal transceiver architectures towards
realizing the IoE applications. We also discuss the most pressing challenges in
implementing such transceivers and briefly review the open research directions.
Our discussion is particularly focused on the opportunities and challenges
pertaining to the IoE physical layer, which can enable the efficient and
effective design of higher-level techniques. We believe that such universal
transceivers can pave the way for seamless connection and communication with
the universe at a deeper level and pioneer the construction of the forthcoming
IoE landscape.

    

### [[2108.13176] Maximum Expected Delay: A New Metric to Analyse the Performance of Asynchronous Quorum-based Protocols in Wireless Sensor Networks](http://arxiv.org/abs/2108.13176)


  Energy management is a crucial challenge in wireless sensor networks. To
date, many techniques have been proposed to reduce energy consumption. Duty
cycle methods reduce the energy consumption of wireless sensor networks since
energy consumption declines in the sleep mode. Using quorum-based methods,
sensors can stay in the sleep mode and be awaken periodically to send and
receive data from adjacent nodes. In this paper, we review a subset of these
methods called asynchronous quorum-based methods, independent of
synchronization between nodes, and investigate their performances in different
metrics. Then, we propose a new metric to investigate the latency of adjacent
nodes in wireless sensor networks. Next, we study the performances of all
discussed methods using the proposed metric. Finally, we introduce the best and
worst methods based on different metrics.

    

### [[2110.00581] Classification of Time-Series Data Using Boosted Decision Trees](http://arxiv.org/abs/2110.00581)


  Time-series data classification is central to the analysis and control of
autonomous systems, such as robots and self-driving cars. Temporal logic-based
learning algorithms have been proposed recently as classifiers of such data.
However, current frameworks are either inaccurate for real-world applications,
such as autonomous driving, or they generate long and complicated formulae that
lack interpretability. To address these limitations, we introduce a novel
learning method, called Boosted Concise Decision Trees (BCDTs), to generate
binary classifiers that are represented as Signal Temporal Logic (STL)
formulae. Our algorithm leverages an ensemble of Concise Decision Trees (CDTs)
to improve the classification performance, where each CDT is a decision tree
that is empowered by a set of techniques to generate simpler formulae and
improve interpretability. The effectiveness and classification performance of
our algorithm are evaluated on naval surveillance and urban-driving case
studies.

    

### [[2110.00594] STRONG: Synchronous and asynchronous RObust Network localization, under Non-Gaussian noise](http://arxiv.org/abs/2110.00594)


  Real-world network applications must cope with failing nodes, malicious
attacks, or nodes facing corrupted data - data classified as outliers. Our work
addresses these concerns in the scope of the sensor network localization
problem where, despite the abundance of technical literature, prior research
seldom considered outlier data. We propose robust, fast, and distributed
network localization algorithms, resilient to high-power noise, but also
precise under regular Gaussian noise. We use a Huber M-estimator, thus
obtaining a robust (but nonconvex) optimization problem. We convexify and
change the problem representation, to allow for distributed robust localization
algorithms: a synchronous distributed method that has optimal convergence rate
and an asynchronous one with proven convergence guarantees. A major highlight
of our contribution lies on the fact that we pay no price for provable
distributed computation neither in accuracy, nor in communication cost or
convergence speed. Simulations showcase the superior performance of our
algorithms, both in the presence of outliers and under regular Gaussian noise:
our method exceeds the accuracy of alternative approaches, distributed and
centralized, even under heavy additive and multiplicative outlier noise.

    

### [[2110.00603] Algorithm Fairness in AI for Medicine and Healthcare](http://arxiv.org/abs/2110.00603)


  In the current development and deployment of many artificial intelligence
(AI) systems in healthcare, algorithm fairness is a challenging problem in
delivering equitable care. Recent evaluation of AI models stratified across
race sub-populations have revealed enormous inequalities in how patients are
diagnosed, given treatments, and billed for healthcare costs. In this
perspective article, we summarize the intersectional field of fairness in
machine learning through the context of current issues in healthcare, outline
how algorithmic biases (e.g. - image acquisition, genetic variation,
intra-observer labeling variability) arise in current clinical workflows and
their resulting healthcare disparities. Lastly, we also review emerging
strategies for mitigating bias via decentralized learning, disentanglement, and
model explainability.

    

### [[2110.00604] Bilevel stochastic methods for optimization and machine learning: Bilevel stochastic descent and DARTS](http://arxiv.org/abs/2110.00604)


  Two-level stochastic optimization formulations have become instrumental in a
number of machine learning contexts such as neural architecture search,
continual learning, adversarial learning, and hyperparameter tuning. Practical
stochastic bilevel optimization problems become challenging in optimization or
learning scenarios where the number of variables is high or there are
constraints.
The goal of this paper is twofold. First, we aim at promoting the use of
bilevel optimization in large-scale learning and we introduce a practical
bilevel stochastic gradient method (BSG-1) that requires neither lower level
second-order derivatives nor system solves (and dismisses any matrix-vector
products). Our BSG-1 method is close to first-order principles, which allows it
to achieve a performance better than those that are not, such as DARTS. Second,
we develop bilevel stochastic gradient descent for bilevel problems with lower
level constraints, and we introduce a convergence theory that covers the
unconstrained and constrained cases and abstracts as much as possible from the
specifics of the bilevel gradient calculation.

    

### [[2110.00610] Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions](http://arxiv.org/abs/2110.00610)


  The efficiency of Hamiltonian Monte Carlo (HMC) can suffer when sampling a
distribution with a wide range of length scales, because the small step sizes
needed for stability in high-curvature regions are inefficient elsewhere. To
address this we present a delayed rejection variant: if an initial HMC
trajectory is rejected, we make one or more subsequent proposals each using a
step size geometrically smaller than the last. We extend the standard delayed
rejection framework by allowing the probability of a retry to depend on the
probability of accepting the previous proposal. We test the scheme in several
sampling tasks, including multiscale model distributions such as Neal's funnel,
and statistical applications. Delayed rejection enables up to five-fold
performance gains over optimally-tuned HMC, as measured by effective sample
size per gradient evaluation. Even for simpler distributions, delayed rejection
provides increased robustness to step size misspecification. Along the way, we
provide an accessible but rigorous review of detailed balance for HMC.

    

### [[2110.00615] Predicting erectile dysfunction after treatment for localized prostate cancer](http://arxiv.org/abs/2110.00615)


  While the 10-year survival rate for localized prostate cancer patients is
very good (>98%), side effects of treatment may limit quality of life
significantly. Erectile dysfunction (ED) is a common burden associated with
increasing age as well as prostate cancer treatment. Although many studies have
investigated the factors affecting erectile dysfunction (ED) after prostate
cancer treatment, only limited studies have investigated whether ED can be
predicted before the start of treatment. The advent of machine learning (ML)
based prediction tools in oncology offers a promising approach to improve
accuracy of prediction and quality of care. Predicting ED may help aid shared
decision making by making the advantages and disadvantages of certain
treatments clear, so that a tailored treatment for an individual patient can be
chosen. This study aimed to predict ED at 1-year and 2-year post-diagnosis
based on patient demographics, clinical data and patient-reported outcomes
(PROMs) measured at diagnosis.

    

### [[2110.00623] Calibrated Adversarial Training](http://arxiv.org/abs/2110.00623)


  Adversarial training is an approach of increasing the robustness of models to
adversarial attacks by including adversarial examples in the training set. One
major challenge of producing adversarial examples is to contain sufficient
perturbation in the example to flip the model's output while not making severe
changes in the example's semantical content. Exuberant change in the semantical
content could also change the true label of the example. Adding such examples
to the training set results in adverse effects. In this paper, we present the
Calibrated Adversarial Training, a method that reduces the adverse effects of
semantic perturbations in adversarial training. The method produces pixel-level
adaptations to the perturbations based on novel calibrated robust error. We
provide theoretical analysis on the calibrated robust error and derive an upper
bound for it. Our empirical results show a superior performance of the
Calibrated Adversarial Training over a number of public datasets.

    

### [[2110.00625] Accelerate Distributed Stochastic Descent for Nonconvex Optimization with Momentum](http://arxiv.org/abs/2110.00625)


  Momentum method has been used extensively in optimizers for deep learning.
Recent studies show that distributed training through K-step averaging has many
nice properties. We propose a momentum method for such model averaging
approaches. At each individual learner level traditional stochastic gradient is
applied. At the meta-level (global learner level), one momentum term is applied
and we call it block momentum. We analyze the convergence and scaling
properties of such momentum methods. Our experimental results show that block
momentum not only accelerates training, but also achieves better results.

    

### [[2110.00627] On the complexity of the optimal transport problem with graph-structured cost](http://arxiv.org/abs/2110.00627)


  Multi-marginal optimal transport (MOT) is a generalization of optimal
transport to multiple marginals. Optimal transport has evolved into an
important tool in many machine learning applications, and its multi-marginal
extension opens up for addressing new challenges in the field of machine
learning. However, the usage of MOT has been largely impeded by its
computational complexity which scales exponentially in the number of marginals.
Fortunately, in many applications, such as barycenter or interpolation
problems, the cost function adheres to structures, which has recently been
exploited for developing efficient computational methods. In this work we
derive computational bounds for these methods. With $m$ marginal distributions
supported on $n$ points, we provide a $ \mathcal{\tilde O}(d(G)m
n^2\epsilon^{-2})$ bound for a $\epsilon$-accuracy when the problem is
associated with a tree with diameter $d(G)$. For the special case of the
Wasserstein barycenter problem, which corresponds to a star-shaped tree, our
bound is in alignment with the existing complexity bound for it.

    

### [[2110.00629] Factored couplings in multi-marginal optimal transport via difference of convex programming](http://arxiv.org/abs/2110.00629)


  Optimal transport (OT) theory underlies many emerging machine learning (ML)
methods nowadays solving a wide range of tasks such as generative modeling,
transfer learning and information retrieval. These latter works, however,
usually build upon a traditional OT setup with two distributions, while leaving
a more general multi-marginal OT formulation somewhat unexplored. In this
paper, we study the multi-marginal OT (MMOT) problem and unify several popular
OT methods under its umbrella by promoting structural information on the
coupling. We show that incorporating such structural information into MMOT
results in an instance of a different of convex (DC) programming problem
allowing us to solve it numerically. Despite high computational cost of the
latter procedure, the solutions provided by DC optimization are usually as
qualitative as those obtained using currently employed optimization schemes.

    

### [[2110.00635] ALBU: An approximate Loopy Belief message passing algorithm for LDA to improve performance on small data sets](http://arxiv.org/abs/2110.00635)


  Variational Bayes (VB) applied to latent Dirichlet allocation (LDA) has
become the most popular algorithm for aspect modeling. While sufficiently
successful in text topic extraction from large corpora, VB is less successful
in identifying aspects in the presence of limited data. We present a novel
variational message passing algorithm as applied to Latent Dirichlet Allocation
(LDA) and compare it with the gold standard VB and collapsed Gibbs sampling. In
situations where marginalisation leads to non-conjugate messages, we use ideas
from sampling to derive approximate update equations. In cases where conjugacy
holds, Loopy Belief update (LBU) (also known as Lauritzen-Spiegelhalter) is
used. Our algorithm, ALBU (approximate LBU), has strong similarities with
Variational Message Passing (VMP) (which is the message passing variant of VB).
To compare the performance of the algorithms in the presence of limited data,
we use data sets consisting of tweets and news groups. Additionally, to perform
more fine grained evaluations and comparisons, we use simulations that enable
comparisons with the ground truth via Kullback-Leibler divergence (KLD). Using
coherence measures for the text corpora and KLD with the simulations we show
that ALBU learns latent distributions more accurately than does VB, especially
for smaller data sets.

    

### [[2110.00637] ML4C: Seeing Causality Through Latent Vicinity](http://arxiv.org/abs/2110.00637)


  Supervised Causal Learning (SCL) aims to learn causal relations from
observational data by accessing previously seen datasets associated with ground
truth causal relations. This paper presents a first attempt at addressing a
fundamental question: What are the benefits from supervision and how does it
benefit? Starting from seeing that SCL is not better than random guessing if
the learning target is non-identifiable a priori, we propose a two-phase
paradigm for SCL by explicitly considering structure identifiability. Following
this paradigm, we tackle the problem of SCL on discrete data and propose ML4C.
The core of ML4C is a binary classifier with a novel learning target: it
classifies whether an Unshielded Triple (UT) is a v-structure or not. Starting
from an input dataset with the corresponding skeleton provided, ML4C orients
each UT once it is classified as a v-structure. These v-structures are together
used to construct the final output. To address the fundamental question of SCL,
we propose a principled method for ML4C featurization: we exploit the vicinity
of a given UT (i.e., the neighbors of UT in skeleton), and derive features by
considering the conditional dependencies and structural entanglement within the
vicinity. We further prove that ML4C is asymptotically perfect. Last but
foremost, thorough experiments conducted on benchmark datasets demonstrate that
ML4C remarkably outperforms other state-of-the-art algorithms in terms of
accuracy, robustness, tolerance and transferability. In summary, ML4C shows
promising results on validating the effectiveness of supervision for causal
learning.

    

### [[2110.00640] Motion Planning for Autonomous Vehicles in the Presence of Uncertainty Using Reinforcement Learning](http://arxiv.org/abs/2110.00640)


  Motion planning under uncertainty is one of the main challenges in developing
autonomous driving vehicles. In this work, we focus on the uncertainty in
sensing and perception, resulted from a limited field of view, occlusions, and
sensing range. This problem is often tackled by considering hypothetical hidden
objects in occluded areas or beyond the sensing range to guarantee passive
safety. However, this may result in conservative planning and expensive
computation, particularly when numerous hypothetical objects need to be
considered. We propose a reinforcement learning (RL) based solution to manage
uncertainty by optimizing for the worst case outcome. This approach is in
contrast to traditional RL, where the agents try to maximize the average
expected reward. The proposed approach is built on top of the Distributional RL
with its policy optimization maximizing the stochastic outcomes' lower bound.
This modification can be applied to a range of RL algorithms. As a
proof-of-concept, the approach is applied to two different RL algorithms, Soft
Actor-Critic and DQN. The approach is evaluated against two challenging
scenarios of pedestrians crossing with occlusion and curved roads with a
limited field of view. The algorithm is trained and evaluated using the SUMO
traffic simulator. The proposed approach yields much better motion planning
behavior compared to conventional RL algorithms and behaves comparably to
humans driving style.

    

### [[2110.00641] Batch size-invariance for policy optimization](http://arxiv.org/abs/2110.00641)


  We say an algorithm is batch size-invariant if changes to the batch size can
largely be compensated for by changes to other hyperparameters. Stochastic
gradient descent is well-known to have this property at small batch sizes, via
the learning rate. However, some policy optimization algorithms (such as PPO)
do not have this property, because of how they control the size of policy
updates. In this work we show how to make these algorithms batch
size-invariant. Our key insight is to decouple the proximal policy (used for
controlling policy updates) from the behavior policy (used for off-policy
corrections). Our experiments help explain why these algorithms work, and
additionally show how they can make more efficient use of stale data.

    

### [[2110.00645] How To Not Drive: Learning Driving Constraints from Demonstration](http://arxiv.org/abs/2110.00645)


  We propose a new scheme to learn motion planning constraints from human
driving trajectories. Behavioral and motion planning are the key components in
an autonomous driving system. The behavioral planning is responsible for
high-level decision making required to follow traffic rules and interact with
other road participants. The motion planner role is to generate feasible, safe
trajectories for a self-driving vehicle to follow. The trajectories are
generated through an optimization scheme to optimize a cost function based on
metrics related to smoothness, movability, and comfort, and subject to a set of
constraints derived from the planned behavior, safety considerations, and
feasibility. A common practice is to manually design the cost function and
constraints. Recent work has investigated learning the cost function from human
driving demonstrations. While effective, the practical application of such
approaches is still questionable in autonomous driving. In contrast, this paper
focuses on learning driving constraints, which can be used as an add-on module
to existing autonomous driving solutions. To learn the constraint, the planning
problem is formulated as a constrained Markov Decision Process, whose elements
are assumed to be known except the constraints. The constraints are then
learned by learning the distribution of expert trajectories and estimating the
probability of optimal trajectories belonging to the learned distribution. The
proposed scheme is evaluated using NGSIM dataset, yielding less than 1\%
collision rate and out of road maneuvers when the learned constraints is used
in an optimization-based motion planner.

    

### [[2110.00650] Multi-lane Cruising Using Hierarchical Planning and Reinforcement Learning](http://arxiv.org/abs/2110.00650)


  Competent multi-lane cruising requires using lane changes and within-lane
maneuvers to achieve good speed and maintain safety. This paper proposes a
design for autonomous multi-lane cruising by combining a hierarchical
reinforcement learning framework with a novel state-action space abstraction.
While the proposed solution follows the classical hierarchy of behavior
decision, motion planning and control, it introduces a key intermediate
abstraction within the motion planner to discretize the state-action space
according to high level behavioral decisions. We argue that this design allows
principled modular extension of motion planning, in contrast to using either
monolithic behavior cloning or a large set of hand-written rules. Moreover, we
demonstrate that our state-action space abstraction allows transferring of the
trained models without retraining from a simulated environment with virtually
no dynamics to one with significantly more realistic dynamics. Together, these
results suggest that our proposed hierarchical architecture is a promising way
to allow reinforcement learning to be applied to complex multi-lane cruising in
the real world.

    

### [[2110.00653] Sparse Deep Learning: A New Framework Immune to Local Traps and Miscalibration](http://arxiv.org/abs/2110.00653)


  Deep learning has powered recent successes of artificial intelligence (AI).
However, the deep neural network, as the basic model of deep learning, has
suffered from issues such as local traps and miscalibration. In this paper, we
provide a new framework for sparse deep learning, which has the above issues
addressed in a coherent way. In particular, we lay down a theoretical
foundation for sparse deep learning and propose prior annealing algorithms for
learning sparse neural networks. The former has successfully tamed the sparse
deep neural network into the framework of statistical modeling, enabling
prediction uncertainty correctly quantified. The latter can be asymptotically
guaranteed to converge to the global optimum, enabling the validity of the
down-stream statistical inference. Numerical result indicates the superiority
of the proposed method compared to the existing ones.

    

### [[2110.00660] Online Obstructive Sleep Apnea Detection Based on Hybrid Machine Learning And Classifier Combination For Home-based Applications](http://arxiv.org/abs/2110.00660)


  Automatic detection of obstructive sleep apnea (OSA) is in great demand. OSA
is one of the most prevalent diseases of the current century and established
comorbidity to Covid-19. OSA is characterized by complete or relative breathing
pauses during sleep. According to medical observations, if OSA remained
unrecognized and un-treated, it may lead to physical and mental complications.
The gold standard of scoring OSA severity is the time-consuming and expensive
method of polysomnography (PSG). The idea of online home-based surveillance of
OSA is welcome. It serves as an effective way for spurred detection and
reference of patients to sleep clinics. In addition, it can perform automatic
control of the therapeutic/assistive devices. In this paper, several
configurations for online OSA detection are proposed. The best configuration
uses both ECG and SpO2 signals for feature extraction and MI analysis for
feature reduction. Various methods of supervised machine learning are exploited
for classification. Finally, to reach the best result, the most successful
classifiers in sensitivity and specificity are combined in groups of three
members with four different combination methods. The proposed method has
advantages like limited use of biological signals, automatic detection, online
working scheme, and uniform and acceptable performance (over 85%) in all the
employed databases. These advantages have not been integrated in previous
published methods.

    

### [[2110.00672] Low Frequency Names Exhibit Bias and Overfitting in Contextualizing Language Models](http://arxiv.org/abs/2110.00672)


  We use a dataset of U.S. first names with labels based on predominant gender
and racial group to examine the effect of training corpus frequency on
tokenization, contextualization, similarity to initial representation, and bias
in BERT, GPT-2, T5, and XLNet. We show that predominantly female and non-white
names are less frequent in the training corpora of these four language models.
We find that infrequent names are more self-similar across contexts, with
Spearman's r between frequency and self-similarity as low as -.763. Infrequent
names are also less similar to initial representation, with Spearman's r
between frequency and linear centered kernel alignment (CKA) similarity to
initial representation as high as .702. Moreover, we find Spearman's r between
racial bias and name frequency in BERT of .492, indicating that lower-frequency
minority group names are more associated with unpleasantness. Representations
of infrequent names undergo more processing, but are more self-similar,
indicating that models rely on less context-informed representations of
uncommon and minority names which are overfit to a lower number of observed
contexts.

    

### [[2110.00673] Multi-Agent Algorithmic Recourse](http://arxiv.org/abs/2110.00673)


  The recent adoption of machine learning as a tool in real world decision
making has spurred interest in understanding how these decisions are being
made. Counterfactual Explanations are a popular interpretable machine learning
technique that aims to understand how a machine learning model would behave if
given alternative inputs. Many explanations attempt to go further and recommend
actions an individual could take to obtain a more desirable output from the
model. These recommendations are known as algorithmic recourse. Past work has
largely focused on the effect algorithmic recourse has on a single agent. In
this work, we show that when the assumption of a single agent environment is
relaxed, current approaches to algorithmic recourse fail to guarantee certain
ethically desirable properties. Instead, we propose a new game theory inspired
framework for providing algorithmic recourse in a multi-agent environment that
does guarantee these properties.

    

### [[2110.00675] Contraction Theory for Nonlinear Stability Analysis and Learning-based Control: A Tutorial Overview](http://arxiv.org/abs/2110.00675)


  Contraction theory is an analytical tool to study differential dynamics of a
non-autonomous (i.e., time-varying) nonlinear system under a contraction metric
defined with a uniformly positive definite matrix, the existence of which
results in a necessary and sufficient characterization of incremental
exponential stability of multiple solution trajectories with respect to each
other. By using a squared differential length as a Lyapunov-like function, its
nonlinear stability analysis boils down to finding a suitable contraction
metric that satisfies a stability condition expressed as a linear matrix
inequality, indicating that many parallels can be drawn between well-known
linear systems theory and contraction theory for nonlinear systems.
Furthermore, contraction theory takes advantage of a superior robustness
property of exponential stability used in conjunction with the comparison
lemma. This yields much-needed safety and stability guarantees for neural
network-based control and estimation schemes, without resorting to a more
involved method of using uniform asymptotic stability for input-to-state
stability. Such distinctive features permit systematic construction of a
contraction metric via convex optimization, thereby obtaining an explicit
exponential bound on the distance between a time-varying target trajectory and
solution trajectories perturbed externally due to disturbances and learning
errors. The objective of this paper is therefore to present a tutorial overview
of contraction theory and its advantages in nonlinear stability analysis of
deterministic and stochastic systems, with an emphasis on deriving formal
robustness and stability guarantees for various learning-based and data-driven
automatic control methods. In particular, we provide a detailed review of
techniques for finding contraction metrics and associated control and
estimation laws using deep neural networks.

    

### [[2110.00678] Speech Technology for Everyone: Automatic Speech Recognition for Non-Native English with Transfer Learning](http://arxiv.org/abs/2110.00678)


  To address the performance gap of English ASR models on L2 English speakers,
we evaluate fine-tuning of pretrained wav2vec 2.0 models (Baevski et al., 2020;
Xu et al., 2021) on L2-ARCTIC, a non-native English speech corpus (Zhao et al.,
2018) under different training settings. We compare \textbf{(a)} models trained
with a combination of diverse accents to ones trained with only specific
accents and \textbf{(b)} results from different single-accent models. Our
experiments demonstrate the promise of developing ASR models for non-native
English speakers, even with small amounts of L2 training data and even without
a language model. Our models also excel in the zero-shot setting where we train
on multiple L2 datasets and test on a blind L2 test set.

    

### [[2110.00681] A systematic evaluation of methods for cell phenotype classification using single-cell RNA sequencing data](http://arxiv.org/abs/2110.00681)


  Background: Single-cell RNA sequencing (scRNA-seq) yields valuable insights
about gene expression and gives critical information about complex tissue
cellular composition. In the analysis of single-cell RNA sequencing, the
annotations of cell subtypes are often done manually, which is time-consuming
and irreproducible. Garnett is a cell-type annotation software based the on
elastic net method. Besides cell-type annotation, supervised machine learning
methods can also be applied to predict other cell phenotypes from genomic data.
Despite the popularity of such applications, there is no existing study to
systematically investigate the performance of those supervised algorithms in
various sizes of scRNA-seq data sets.
Methods and Results: This study evaluates 13 popular supervised machine
learning algorithms to classify cell phenotypes, using published real and
simulated data sets with diverse cell sizes. The benchmark contained two parts.
In the first part, we used real data sets to assess the popular supervised
algorithms' computing speed and cell phenotype classification performance. The
classification performances were evaluated using AUC statistics, F1-score,
precision, recall, and false-positive rate. In the second part, we evaluated
gene selection performance using published simulated data sets with a known
list of real genes.
Conclusion: The study outcomes showed that ElasticNet with interactions
performed best in small and medium data sets. NB was another appropriate method
for medium data sets. In large data sets, XGB works excellent. Ensemble
algorithms were not significantly superior to individual machine learning
methods. Adding interactions to ElasticNet can help, and the improvement was
significant in small data sets.

    

### [[2110.00683] Learning through atypical ''phase transitions'' in overparameterized neural networks](http://arxiv.org/abs/2110.00683)


  Current deep neural networks are highly overparameterized (up to billions of
connection weights) and nonlinear. Yet they can fit data almost perfectly
through variants of gradient descent algorithms and achieve unexpected levels
of prediction accuracy without overfitting. These are formidable results that
escape the bias-variance predictions of statistical learning and pose
conceptual challenges for non-convex optimization. In this paper, we use
methods from statistical physics of disordered systems to analytically study
the computational fallout of overparameterization in nonconvex neural network
models. As the number of connection weights increases, we follow the changes of
the geometrical structure of different minima of the error loss function and
relate them to learning and generalisation performance. We find that there
exist a gap between the SAT/UNSAT interpolation transition where solutions
begin to exist and the point where algorithms start to find solutions, i.e.
where accessible solutions appear. This second phase transition coincides with
the discontinuous appearance of atypical solutions that are locally extremely
entropic, i.e., flat regions of the weight space that are particularly
solution-dense and have good generalization properties. Although exponentially
rare compared to typical solutions (which are narrower and extremely difficult
to sample), entropic solutions are accessible to the algorithms used in
learning. We can characterize the generalization error of different solutions
and optimize the Bayesian prediction, for data generated from a structurally
different network. Numerical tests on observables suggested by the theory
confirm that the scenario extends to realistic deep networks.

    

### [[2110.00684] Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)](http://arxiv.org/abs/2110.00684)


  A central goal in deep learning is to learn compact representations of
features at every layer of a neural network, which is useful for both
unsupervised representation learning and structured network pruning. While
there is a growing body of work in structured pruning, current state-of-the-art
methods suffer from two key limitations: (i) instability during training, and
(ii) need for an additional step of fine-tuning, which is resource-intensive.
At the core of these limitations is the lack of a systematic approach that
jointly prunes and refines weights during training in a single stage, and does
not require any fine-tuning upon convergence to achieve state-of-the-art
performance. We present a novel single-stage structured pruning method termed
DiscriminAtive Masking (DAM). The key intuition behind DAM is to
discriminatively prefer some of the neurons to be refined during the training
process, while gradually masking out other neurons. We show that our proposed
DAM approach has remarkably good performance over various applications,
including dimensionality reduction, recommendation system, graph representation
learning, and structured pruning for image classification. We also
theoretically show that the learning objective of DAM is directly related to
minimizing the L0 norm of the masking layer.

    

### [[2110.00685] Fast Multi-Resolution Transformer Fine-tuning for Extreme Multi-label Text Classification](http://arxiv.org/abs/2110.00685)


  Extreme multi-label text classification (XMC) seeks to find relevant labels
from an extreme large label collection for a given text input. Many real-world
applications can be formulated as XMC problems, such as recommendation systems,
document tagging and semantic search. Recently, transformer based XMC methods,
such as X-Transformer and LightXML, have shown significant improvement over
other XMC methods. Despite leveraging pre-trained transformer models for text
representation, the fine-tuning procedure of transformer models on large label
space still has lengthy computational time even with powerful GPUs. In this
paper, we propose a novel recursive approach, XR-Transformer to accelerate the
procedure through recursively fine-tuning transformer models on a series of
multi-resolution objectives related to the original XMC objective function.
Empirical results show that XR-Transformer takes significantly less training
time compared to other transformer-based XMC models while yielding better
state-of-the-art results. In particular, on the public Amazon-3M dataset with 3
million labels, XR-Transformer is not only 20x faster than X-Transformer but
also improves the Precision@1 from 51% to 54%.

    

### [[2110.00693] A Theoretical Overview of Neural Contraction Metrics for Learning-based Control with Guaranteed Stability](http://arxiv.org/abs/2110.00693)


  This paper presents a theoretical overview of a Neural Contraction Metric
(NCM): a neural network model of an optimal contraction metric and
corresponding differential Lyapunov function, the existence of which is a
necessary and sufficient condition for incremental exponential stability of
non-autonomous nonlinear system trajectories. Its innovation lies in providing
formal robustness guarantees for learning-based control frameworks, utilizing
contraction theory as an analytical tool to study the nonlinear stability of
learned systems via convex optimization. In particular, we rigorously show in
this paper that, by regarding modeling errors of the learning schemes as
external disturbances, the NCM control is capable of obtaining an explicit
bound on the distance between a time-varying target trajectory and perturbed
solution trajectories, which exponentially decreases with time even under the
presence of deterministic and stochastic perturbation. These useful features
permit simultaneous synthesis of a contraction metric and associated control
law by a neural network, thereby enabling real-time computable and probably
robust learning-based control for general control-affine nonlinear systems.

    

### [[2110.00695] Deep Learning for Rain Fade Prediction in Satellite Communications](http://arxiv.org/abs/2110.00695)


  Line of sight satellite systems, unmanned aerial vehicles, high-altitude
platforms, and microwave links that operate on frequency bands such as Ka-band
or higher are extremely susceptible to rain. Thus, rain fade forecasting for
these systems is critical because it allows the system to switch between ground
gateways proactively before a rain fade event to maintain seamless service.
Although empirical, statistical, and fade slope models can predict rain fade to
some extent, they typically require statistical measurements of rain
characteristics in a given area and cannot be generalized to a large scale
system. Furthermore, such models typically predict near-future rain fade events
but are incapable of forecasting far into the future, making proactive resource
management more difficult. In this paper, a deep learning (DL)-based
architecture is proposed that forecasts future rain fade using satellite and
radar imagery data as well as link power measurements. Furthermore, the data
preprocessing and architectural design have been thoroughly explained and
multiple experiments have been conducted. Experiments show that the proposed DL
architecture outperforms current state-of-the-art machine learning-based
algorithms in rain fade forecasting in the near and long term. Moreover, the
results indicate that radar data with weather condition information is more
effective for short-term prediction, while satellite data with cloud movement
information is more effective for long-term predictions.

    

### [[2110.00704] OSCAR: Data-Driven Operational Space Control for Adaptive and Robust Robot Manipulation](http://arxiv.org/abs/2110.00704)


  Learning performant robot manipulation policies can be challenging due to
high-dimensional continuous actions and complex physics-based dynamics. This
can be alleviated through intelligent choice of action space. Operational Space
Control (OSC) has been used as an effective task-space controller for
manipulation. Nonetheless, its strength depends on the underlying modeling
fidelity, and is prone to failure when there are modeling errors. In this work,
we propose OSC for Adaptation and Robustness (OSCAR), a data-driven variant of
OSC that compensates for modeling errors by inferring relevant dynamics
parameters from online trajectories. OSCAR decomposes dynamics learning into
task-agnostic and task-specific phases, decoupling the dynamics dependencies of
the robot and the extrinsics due to its environment. This structure enables
robust zero-shot performance under out-of-distribution and rapid adaptation to
significant domain shifts through additional finetuning. We evaluate our method
on a variety of simulated manipulation problems, and find substantial
improvements over an array of controller baselines. For more results and
information, please visit this https URL.

    

### [[2110.00708] Universal Adversarial Spoofing Attacks against Face Recognition](http://arxiv.org/abs/2110.00708)


  We assess the vulnerabilities of deep face recognition systems for images
that falsify/spoof multiple identities simultaneously. We demonstrate that, by
manipulating the deep feature representation extracted from a face image via
imperceptibly small perturbations added at the pixel level using our proposed
Universal Adversarial Spoofing Examples (UAXs), one can fool a face
verification system into recognizing that the face image belongs to multiple
different identities with a high success rate. One characteristic of the UAXs
crafted with our method is that they are universal (identity-agnostic); they
are successful even against identities not known in advance. For a certain deep
neural network, we show that we are able to spoof almost all tested identities
(99\%), including those not known beforehand (not included in training). Our
results indicate that a multiple-identity attack is a real threat and should be
taken into account when deploying face recognition systems.

    

### [[2110.00719] One-Bit Matrix Completion with Differential Privacy](http://arxiv.org/abs/2110.00719)


  Matrix completion is a prevailing collaborative filtering method for
recommendation systems that requires the data offered by users to provide
personalized service. However, due to insidious attacks and unexpected
inference, the release of user data often raises serious privacy concerns. Most
of the existing solutions focus on improving the privacy guarantee for general
matrix completion. As a special case, in recommendation systems where the
observations are binary, one-bit matrix completion covers a broad range of
real-life situations. In this paper, we propose a novel framework for one-bit
matrix completion under the differential privacy constraint. In this framework,
we develop several perturbation mechanisms and analyze the privacy-accuracy
trade-off offered by each mechanism. The experiments conducted on both
synthetic and real-world datasets demonstrate that our proposed approaches can
maintain high-level privacy with little loss of completion accuracy.

    

### [[2110.00720] Is There More Pattern in Knowledge Graph? Exploring Proximity Pattern for Knowledge Graph Embedding](http://arxiv.org/abs/2110.00720)


  Modeling of relation pattern is the core focus of previous Knowledge Graph
Embedding works, which represents how one entity is related to another
semantically by some explicit relation. However, there is a more natural and
intuitive relevancy among entities being always ignored, which is that how one
entity is close to another semantically, without the consideration of any
explicit relation. We name such semantic phenomenon in knowledge graph as
proximity pattern. In this work, we explore the problem of how to define and
represent proximity pattern, and how it can be utilized to help knowledge graph
embedding. Firstly, we define the proximity of any two entities according to
their statistically shared queries, then we construct a derived graph structure
and represent the proximity pattern from global view. Moreover, with the
original knowledge graph, we design a Chained couPle-GNN (CP-GNN) architecture
to deeply merge the two patterns (graphs) together, which can encode a more
comprehensive knowledge embedding. Being evaluated on FB15k-237 and WN18RR
datasets, CP-GNN achieves state-of-the-art results for Knowledge Graph
Completion task, and can especially boost the modeling capacity for complex
queries that contain multiple answer entities, proving the effectiveness of
introduced proximity pattern.

    

### [[2110.00724] Complex Spin Hamiltonian Represented by Artificial Neural Network](http://arxiv.org/abs/2110.00724)


  The effective spin Hamiltonian method is widely adopted to simulate and
understand the behavior of magnetism. However, the magnetic interactions of
some systems, such as itinerant magnets, are too complex to be described by any
explicit function, which prevents an accurate description of magnetism in such
systems. Here, we put forward a machine learning (ML) approach, applying an
artificial neural network (ANN) and a local spin descriptor to develop
effective spin potentials for any form of interaction. The constructed
Hamiltonians include an explicit Heisenberg part and an implicit non-linear ANN
part. Such a method successfully reproduces artificially constructed models and
also sufficiently describe the itinerant magnetism of bulk Fe3GeTe2. Our work
paves a new way for investigating complex magnetic phenomena (e.g., skyrmions)
of magnetic materials.

    

### [[2110.00728] Implementation of MPPT Technique of Solar Module with Supervised Machine Learning](http://arxiv.org/abs/2110.00728)


  In this paper, we proposed a method using supervised ML in solar PV system
for MPPT analysis. For this purpose, an overall schematic diagram of a PV
system is designed and simulated to create a dataset in MATLAB/ Simulink. Thus,
by analyzing the output characteristics of a solar cell, an improved MPPT
algorithm on the basis of neural network (NN) method is put forward to track
the maximum power point (MPP) of solar cell modules. To perform the task,
Bayesian Regularization method was chosen as the training algorithm as it works
best even for smaller data supporting the wide range of the train data set. The
theoretical results show that the improved NN MPPT algorithm has higher
efficiency compared with the Perturb and Observe method in the same
environment, and the PV system can keep working at MPP without oscillation and
probability of any kind of misjudgment. So it can not only reduce misjudgment,
but also avoid power loss around the MPP. Moreover, we implemented the
algorithm in a hardware set-up and verified the theoretical result comparing it
with the empirical data.

    

### [[2110.00744] Random Subgraph Detection Using Queries](http://arxiv.org/abs/2110.00744)


  The planted densest subgraph detection problem refers to the task of testing
whether in a given (random) graph there is a subgraph that is unusually dense.
Specifically, we observe an undirected and unweighted graph on $n$ nodes. Under
the null hypothesis, the graph is a realization of an Erdős-Rényi graph
with edge probability (or, density) $q$. Under the alternative, there is a
subgraph on $k$ vertices with edge probability $p>q$. The statistical as well
as the computational barriers of this problem are well-understood for a wide
range of the edge parameters $p$ and $q$. In this paper, we consider a natural
variant of the above problem, where one can only observe a small part of the
graph using adaptive edge queries.
For this model, we determine the number of queries necessary and sufficient
for detecting the presence of the planted subgraph. Specifically, we show that
any (possibly randomized) algorithm must make $\mathsf{Q} =
\Omega(\frac{n^2}{k^2\chi^4(p||q)}\log^2n)$ adaptive queries (on expectation)
to the adjacency matrix of the graph to detect the planted subgraph with
probability more than $1/2$, where $\chi^2(p||q)$ is the Chi-Square distance.
On the other hand, we devise a quasi-polynomial-time algorithm that finds the
planted subgraph with high probability by making $\mathsf{Q} =
O(\frac{n^2}{k^2\chi^4(p||q)}\log^2n)$ adaptive queries. We then propose a
polynomial-time algorithm which is able to detect the planted subgraph using
$\mathsf{Q} = O(\frac{n^4}{k^4\chi^2(p||q)}\log n)$ queries. We conjecture that
in the leftover regime, where $\frac{n^2}{k^2}\ll\mathsf{Q}\ll
\frac{n^4}{k^4}$, no polynomial-time algorithms exist; we give an evidence for
this hypothesis using the planted clique conjecture. Our results resolve three
questions posed in \cite{racz2020finding}, where the special case of adaptive
detection and recovery of a planted clique was considered.

    

### [[2110.00745] End-to-End Complex-Valued Multidilated Convolutional Neural Network for Joint Acoustic Echo Cancellation and Noise Suppression](http://arxiv.org/abs/2110.00745)


  Echo and noise suppression is an integral part of a full-duplex communication
system. Many recent acoustic echo cancellation (AEC) systems rely on a separate
adaptive filtering module for linear echo suppression and a neural module for
residual echo suppression. However, not only do adaptive filtering modules
require convergence and remain susceptible to changes in acoustic environments,
but this two-stage framework also often introduces unnecessary delays to the
AEC system when neural modules are already capable of both linear and nonlinear
echo suppression. In this paper, we exploit the offset-compensating ability of
complex time-frequency masks and propose an end-to-end complex-valued neural
network architecture. The building block of the proposed model is a
pseudocomplex extension based on the densely-connected multidilated DenseNet
(D3Net) building block, resulting in a very small network of only 354K
parameters. The architecture utilized the multi-resolution nature of the D3Net
building blocks to eliminate the need for pooling, allowing the network to
extract features using large receptive fields without any loss of output
resolution. We also propose a dual-mask technique for joint echo and noise
suppression with simultaneous speech enhancement. Evaluation on both synthetic
and real test sets demonstrated promising results across multiple energy-based
metrics and perceptual proxies.

    

### [[2110.00751] Partner-Aware Algorithms in Decentralized Cooperative Bandit Teams](http://arxiv.org/abs/2110.00751)


  When humans collaborate with each other, they often make decisions by
observing others and considering the consequences that their actions may have
on the entire team, instead of greedily doing what is best for just themselves.
We would like our AI agents to effectively collaborate in a similar way by
capturing a model of their partners. In this work, we propose and analyze a
decentralized Multi-Armed Bandit (MAB) problem with coupled rewards as an
abstraction of more general multi-agent collaboration. We demonstrate that
naïve extensions of single-agent optimal MAB algorithms fail when applied for
decentralized bandit teams. Instead, we propose a Partner-Aware strategy for
joint sequential decision-making that extends the well-known single-agent Upper
Confidence Bound algorithm. We analytically show that our proposed strategy
achieves logarithmic regret, and provide extensive experiments involving
human-AI and human-robot collaboration to validate our theoretical findings.
Our results show that the proposed partner-aware strategy outperforms other
known methods, and our human subject studies suggest humans prefer to
collaborate with AI agents implementing our partner-aware strategy.

    

### [[2110.00755] Explainable Event Recognition](http://arxiv.org/abs/2110.00755)


  The literature shows outstanding capabilities for CNNs in event recognition
in images. However, fewer attempts are made to analyze the potential causes
behind the decisions of the models and exploring whether the predictions are
based on event-salient objects or regions? To explore this important aspect of
event recognition, in this work, we propose an explainable event recognition
framework relying on Grad-CAM and an Xception architecture-based CNN model.
Experiments are conducted on three large-scale datasets covering a diversified
set of natural disasters, social, and sports events. Overall, the model showed
outstanding generalization capabilities obtaining overall F1-scores of 0.91,
0.94, and 0.97 on natural disasters, social, and sports events, respectively.
Moreover, for subjective analysis of activation maps generated through Grad-CAM
for the predicted samples of the model, a crowdsourcing study is conducted to
analyze whether the model's predictions are based on event-related
objects/regions or not? The results of the study indicate that 78%, 84%, and
78% of the model decisions on natural disasters, sports, and social events
datasets, respectively, are based onevent-related objects or regions.

    

### [[2110.00784] Seeking Visual Discomfort: Curiosity-driven Representations for Reinforcement Learning](http://arxiv.org/abs/2110.00784)


  Vision-based reinforcement learning (RL) is a promising approach to solve
control tasks involving images as the main observation. State-of-the-art RL
algorithms still struggle in terms of sample efficiency, especially when using
image observations. This has led to increased attention on integrating state
representation learning (SRL) techniques into the RL pipeline. Work in this
field demonstrates a substantial improvement in sample efficiency among other
benefits. However, to take full advantage of this paradigm, the quality of
samples used for training plays a crucial role. More importantly, the diversity
of these samples could affect the sample efficiency of vision-based RL, but
also its generalization capability. In this work, we present an approach to
improve sample diversity for state representation learning. Our method enhances
the exploration capability of RL algorithms, by taking advantage of the SRL
setup. Our experiments show that our proposed approach boosts the visitation of
problematic states, improves the learned state representation, and outperforms
the baselines for all tested environments. These results are most apparent for
environments where the baseline methods struggle. Even in simple environments,
our method stabilizes the training, reduces the reward variance, and promotes
sample efficiency.

    

### [[2110.00788] Inference-InfoGAN: Inference Independence via Embedding Orthogonal Basis Expansion](http://arxiv.org/abs/2110.00788)


  Disentanglement learning aims to construct independent and interpretable
latent variables in which generative models are a popular strategy. InfoGAN is
a classic method via maximizing Mutual Information (MI) to obtain interpretable
latent variables mapped to the target space. However, it did not emphasize
independent characteristic. To explicitly infer latent variables with
inter-independence, we propose a novel GAN-based disentanglement framework via
embedding Orthogonal Basis Expansion (OBE) into InfoGAN network
(Inference-InfoGAN) in an unsupervised way. Under the OBE module, one set of
orthogonal basis can be adaptively found to expand arbitrary data with
independence property. To ensure the target-wise interpretable representation,
we add a consistence constraint between the expansion coefficients and latent
variables on the base of MI maximization. Additionally, we design an
alternating optimization step on the consistence constraint and orthogonal
requirement updating, so that the training of Inference-InfoGAN can be more
convenient. Finally, experiments validate that our proposed OBE module obtains
adaptive orthogonal basis, which can express better independent characteristics
than fixed basis expression of Discrete Cosine Transform (DCT). To depict the
performance in downstream tasks, we compared with the state-of-the-art
GAN-based and even VAE-based approaches on different datasets. Our
Inference-InfoGAN achieves higher disentanglement score in terms of FactorVAE,
Separated Attribute Predictability (SAP), Mutual Information Gap (MIG) and
Variation Predictability (VP) metrics without model fine-tuning. All the
experimental results illustrate that our method has inter-independence
inference ability because of the OBE module, and provides a good trade-off
between it and target-wise interpretability of latent variables via jointing
the alternating optimization.

    

### [[2110.00792] Learning Models as Functionals of Signed-Distance Fields for Manipulation Planning](http://arxiv.org/abs/2110.00792)


  This work proposes an optimization-based manipulation planning framework
where the objectives are learned functionals of signed-distance fields that
represent objects in the scene. Most manipulation planning approaches rely on
analytical models and carefully chosen abstractions/state-spaces to be
effective. A central question is how models can be obtained from data that are
not primarily accurate in their predictions, but, more importantly, enable
efficient reasoning within a planning framework, while at the same time being
closely coupled to perception spaces. We show that representing objects as
signed-distance fields not only enables to learn and represent a variety of
models with higher accuracy compared to point-cloud and occupancy measure
representations, but also that SDF-based models are suitable for
optimization-based planning. To demonstrate the versatility of our approach, we
learn both kinematic and dynamic models to solve tasks that involve hanging
mugs on hooks and pushing objects on a table. We can unify these quite
different tasks within one framework, since SDFs are the common object
representation. Video: this https URL


### [[2110.00804] ProTo: Program-Guided Transformer for Program-Guided Tasks](http://arxiv.org/abs/2110.00804)


  Programs, consisting of semantic and structural information, play an
important role in the communication between humans and agents. Towards learning
general program executors to unify perception, reasoning, and decision making,
we formulate program-guided tasks which require learning to execute a given
program on the observed task specification. Furthermore, we propose the
Program-guided Transformer (ProTo), which integrates both semantic and
structural guidance of a program by leveraging cross-attention and masked
self-attention to pass messages between the specification and routines in the
program. ProTo executes a program in a learned latent space and enjoys stronger
representation ability than previous neural-symbolic approaches. We demonstrate
that ProTo significantly outperforms the previous state-of-the-art methods on
GQA visual reasoning and 2D Minecraft policy learning datasets. Additionally,
ProTo demonstrates better generalization to unseen, complex, and human-written
programs.

    

### [[2110.00808] Cycle-Consistent World Models for Domain Independent Latent Imagination](http://arxiv.org/abs/2110.00808)


  End-to-end autonomous driving seeks to solve the perception, decision, and
control problems in an integrated way, which can be easier to generalize at
scale and be more adapting to new scenarios. However, high costs and risks make
it very hard to train autonomous cars in the real world. Simulations can
therefore be a powerful tool to enable training. Due to slightly different
observations, agents trained and evaluated solely in simulation often perform
well there but have difficulties in real-world environments. To tackle this
problem, we propose a novel model-based reinforcement learning approach called
Cycleconsistent World Models. Contrary to related approaches, our model can
embed two modalities in a shared latent space and thereby learn from samples in
one modality (e.g., simulated data) and be used for inference in different
domain (e.g., real-world data). Our experiments using different modalities in
the CARLA simulator showed that this enables CCWM to outperform
state-of-the-art domain adaptation approaches. Furthermore, we show that CCWM
can decode a given latent representation into semantically coherent
observations in both modalities.

    

### [[2110.00809] Classifying COVID-19 Spike Sequences from Geographic Location Using Deep Learning](http://arxiv.org/abs/2110.00809)


  With the rapid spread of COVID-19 worldwide, viral genomic data is available
in the order of millions of sequences on public databases such as GISAID. This
\emph{Big Data} creates a unique opportunity for analysis towards the research
of effective vaccine development for current pandemics, and avoiding or
mitigating future pandemics. One piece of information that comes with every
such viral sequence is the geographical location where it was collected -- the
patterns found between viral variants and geographic location surely being an
important part of this analysis. One major challenge that researchers face is
processing such huge, highly dimensional data to get useful insights as quickly
as possible. Most of the existing methods face scalability issues when dealing
with the magnitude of such data. In this paper, we propose an algorithm that
first computes a numerical representation of the spike protein sequence of
SARS-CoV-2 using $k$-mers substrings) and then uses a deep learning-based model
to classify the sequences in terms of geographical location. We show that our
proposed model significantly outperforms the baselines. We also show the
importance of different amino acids in the spike sequences by computing the
information gain corresponding to the true class labels.

    

### [[2110.00813] Consider the Alternatives: Navigating Fairness-Accuracy Tradeoffs via Disqualification](http://arxiv.org/abs/2110.00813)


  In many machine learning settings there is an inherent tension between
fairness and accuracy desiderata. How should one proceed in light of such
trade-offs? In this work we introduce and study $\gamma$-disqualification, a
new framework for reasoning about fairness-accuracy tradeoffs w.r.t a benchmark
class $H$ in the context of supervised learning. Our requirement stipulates
that a classifier should be disqualified if it is possible to improve its
fairness by switching to another classifier from $H$ without paying "too much"
in accuracy. The notion of "too much" is quantified via a parameter $\gamma$
that serves as a vehicle for specifying acceptable tradeoffs between accuracy
and fairness, in a way that is independent from the specific metrics used to
quantify fairness and accuracy in a given task. Towards this objective, we
establish principled translations between units of accuracy and units of
(un)fairness for different accuracy measures. We show $\gamma$-disqualification
can be used to easily compare different learning strategies in terms of how
they trade-off fairness and accuracy, and we give an efficient reduction from
the problem of finding the optimal classifier that satisfies our requirement to
the problem of approximating the Pareto frontier of $H$.

    

### [[2110.00816] Calibrated Multiple-Output Quantile Regression with Representation Learning](http://arxiv.org/abs/2110.00816)


  We develop a method to generate predictive regions that cover a multivariate
response variable with a user-specified probability. Our work is composed of
two components. First, we use a deep generative model to learn a representation
of the response that has a unimodal distribution. Existing multiple-output
quantile regression approaches are effective in such cases, so we apply them on
the learned representation, and then transform the solution to the original
space of the response. This process results in a flexible and informative
region that can have an arbitrary shape, a property that existing methods lack.
Second, we propose an extension of conformal prediction to the multivariate
response setting that modifies any method to return sets with a pre-specified
coverage level. The desired coverage is theoretically guaranteed in the
finite-sample case for any distribution. Experiments conducted on both real and
synthetic data show that our method constructs regions that are significantly
smaller (sometimes by a factor of 100) compared to existing techniques.

    

### [[2110.00836] AI Back-End as a Service for Learning Switching of Mobile Apps between the Fog and the Cloud](http://arxiv.org/abs/2110.00836)


  Given that cloud servers are usually remotely located from the devices of
mobile apps, the end-users of the apps can face delays. The Fog has been
introduced to augment the apps with machines located at the network edge close
to the end-users. However, edge machines are usually resource constrained.
Thus, the execution of online data-analytics on edge machines may not be
feasible if the time complexity of the data-analytics algorithm is high. To
overcome this, multiple instances of the back-end should be deployed on edge
and remote machines. In this case, the research question is how the switching
of the app among the instances of the back-end can be dynamically decided based
on the response time of the service instances. To answer this, we contribute an
AI approach that trains machine-learning models of the response time of service
instances. Our approach extends a back-end as a service into an AI
self-back-end as a service that self-decides at runtime the right edge/remote
instance that achieves the lowest response-time. We evaluate the accuracy and
the efficiency of our approach by using real-word machine-learning datasets on
an existing auction app.

    

### [[2110.00841] Transfer Learning Approaches for Knowledge Discovery in Grid-based Geo-Spatiotemporal Data](http://arxiv.org/abs/2110.00841)


  Extracting and meticulously analyzing geo-spatiotemporal features is crucial
to recognize intricate underlying causes of natural events, such as floods.
Limited evidence about hidden factors leading to climate change makes it
challenging to predict regional water discharge accurately. In addition, the
explosive growth in complex geo-spatiotemporal environment data that requires
repeated learning by the state-of-the-art neural networks for every new region
emphasizes the need for new computationally efficient methods, advanced
computational resources, and extensive training on a massive amount of
available monitored data. We, therefore, propose HydroDeep, an effectively
reusable pretrained model to address this problem of transferring knowledge
from one region to another by effectively capturing their intrinsic
geo-spatiotemporal variance. Further, we present four transfer learning
approaches on HydroDeep for spatiotemporal interpretability that improve
Nash-Sutcliffe efficiency by 9% to 108% in new regions with a 95% reduction in
time.

    

### [[2110.00843] SHARP: Shielding-Aware Robust Planning for Safe and Efficient Human-Robot Interaction](http://arxiv.org/abs/2110.00843)


  Jointly achieving safety and efficiency in human-robot interaction (HRI)
settings is a challenging problem, as the robot's planning objectives may be at
odds with the human's own intent and expectations. Recent approaches ensure
safe robot operation in uncertain environments through a supervisory control
scheme, sometimes called "shielding", which overrides the robot's nominal plan
with a safety fallback strategy when a safety-critical event is imminent. These
reactive "last-resort" strategies (typically in the form of aggressive
emergency maneuvers) focus on preserving safety without efficiency
considerations; when the nominal planner is unaware of possible safety
overrides, shielding can be activated more frequently than necessary, leading
to degraded performance. In this work, we propose a new shielding-based
planning approach that allows the robot to plan efficiently by explicitly
accounting for possible future shielding events. Leveraging recent work on
Bayesian human motion prediction, the resulting robot policy proactively
balances nominal performance with the risk of high-cost emergency maneuvers
triggered by low-probability human behaviors. We formalize Shielding-Aware
Robust Planning (SHARP) as a stochastic optimal control problem and propose a
computationally efficient framework for finding tractable approximate solutions
at runtime. Our method outperforms the shielding-agnostic motion planning
baseline (equipped with the same human intent inference scheme) on simulated
driving examples with human trajectories taken from the recently released Waymo
Open Motion Dataset.

    

### [[2110.00844] A Robust Alternative for Graph Convolutional Neural Networks via Graph Neighborhood Filters](http://arxiv.org/abs/2110.00844)


  Graph convolutional neural networks (GCNNs) are popular deep learning
architectures that, upon replacing regular convolutions with graph filters
(GFs), generalize CNNs to irregular domains. However, classical GFs are prone
to numerical errors since they consist of high-order polynomials. This problem
is aggravated when several filters are applied in cascade, limiting the
practical depth of GCNNs. To tackle this issue, we present the neighborhood
graph filters (NGFs), a family of GFs that replaces the powers of the graph
shift operator with $k$-hop neighborhood adjacency matrices. NGFs help to
alleviate the numerical issues of traditional GFs, allow for the design of
deeper GCNNs, and enhance the robustness to errors in the topology of the
graph. To illustrate the advantage over traditional GFs in practical
applications, we use NGFs in the design of deep neighborhood GCNNs to solve
graph signal denoising and node classification problems over both synthetic and
real-world data.

    

### [[2110.00852] Learning Networked Linear Dynamical Systems under Non-white Excitation from a Single Trajectory](http://arxiv.org/abs/2110.00852)


  We consider a networked linear dynamical system with $p$ agents/nodes. We
study the problem of learning the underlying graph of interactions/dependencies
from observations of the nodal trajectories over a time-interval $T$. We
present a regularized non-casual consistent estimator for this problem and
analyze its sample complexity over two regimes: (a) where the interval $T$
consists of $n$ i.i.d. observation windows of length $T/n$ (restart and
record), and (b) where $T$ is one continuous observation window (consecutive).
Using the theory of $M$-estimators, we show that the estimator recovers the
underlying interactions, in either regime, in a time-interval that is
logarithmic in the system size $p$. To the best of our knowledge, this is the
first work to analyze the sample complexity of learning linear dynamical
systems driven by unobserved not-white wide-sense stationary (WSS) inputs.

    

### [[2110.00855] SurvTRACE: Transformers for Survival Analysis with Competing Events](http://arxiv.org/abs/2110.00855)


  In medicine, survival analysis studies the time duration to events of
interest such as mortality. One major challenge is how to deal with multiple
competing events (e.g., multiple disease diagnoses). In this work, we propose a
transformer-based model that does not make the assumption for the underlying
survival distribution and is capable of handling competing events, namely
SurvTRACE. We account for the implicit \emph{confounders} in the observational
setting in multi-events scenarios, which causes selection bias as the predicted
survival probability is influenced by irrelevant factors. To sufficiently
utilize the survival data to train transformers from scratch, multiple
auxiliary tasks are designed for multi-task learning. The model hence learns a
strong shared representation from all these tasks and in turn serves for better
survival analysis. We further demonstrate how to inspect the covariate
relevance and importance through interpretable attention mechanisms of
SurvTRACE, which suffices to great potential in enhancing clinical trial design
and new treatment development. Experiments on METABRIC, SUPPORT, and SEER data
with 470k patients validate the all-around superiority of our method.

    

### [[2110.00857] FairFed: Enabling Group Fairness in Federated Learning](http://arxiv.org/abs/2110.00857)


  As machine learning becomes increasingly incorporated in crucial
decision-making scenarios such as healthcare, recruitment, and loan assessment,
there have been increasing concerns about the privacy and fairness of such
systems. Federated learning has been viewed as a promising solution for
collaboratively learning machine learning models among multiple parties while
maintaining the privacy of their local data. However, federated learning also
poses new challenges in mitigating the potential bias against certain
populations (e.g., demographic groups), which typically requires centralized
access to the sensitive information (e.g., race, gender) of each data point.
Motivated by the importance and challenges of group fairness in federated
learning, in this work, we propose FairFed, a novel algorithm to enhance group
fairness via a fairness-aware aggregation method, aiming to provide fair model
performance across different sensitive groups (e.g., racial, gender groups)
while maintaining high utility. The formulation can potentially provide more
flexibility in the customized local debiasing strategies for each client. When
running federated training on two widely investigated fairness datasets, Adult
and COMPAS, our proposed method outperforms the state-of-the-art fair federated
learning frameworks under a high heterogeneous sensitive attribute
distribution.

    

### [[2110.00859] A Comparative Study of Sentiment Analysis Using NLP and Different Machine Learning Techniques on US Airline Twitter Data](http://arxiv.org/abs/2110.00859)


  Today's business ecosystem has become very competitive. Customer satisfaction
has become a major focus for business growth. Business organizations are
spending a lot of money and human resources on various strategies to understand
and fulfill their customer's needs. But, because of defective manual analysis
on multifarious needs of customers, many organizations are failing to achieve
customer satisfaction. As a result, they are losing customer's loyalty and
spending extra money on marketing. We can solve the problems by implementing
Sentiment Analysis. It is a combined technique of Natural Language Processing
(NLP) and Machine Learning (ML). Sentiment Analysis is broadly used to extract
insights from wider public opinion behind certain topics, products, and
services. We can do it from any online available data. In this paper, we have
introduced two NLP techniques (Bag-of-Words and TF-IDF) and various ML
classification algorithms (Support Vector Machine, Logistic Regression,
Multinomial Naive Bayes, Random Forest) to find an effective approach for
Sentiment Analysis on a large, imbalanced, and multi-classed dataset. Our best
approaches provide 77% accuracy using Support Vector Machine and Logistic
Regression with Bag-of-Words technique.

    

### [[2110.00871] Feel-Good Thompson Sampling for Contextual Bandits and Reinforcement Learning](http://arxiv.org/abs/2110.00871)


  Thompson Sampling has been widely used for contextual bandit problems due to
the flexibility of its modeling power. However, a general theory for this class
of methods in the frequentist setting is still lacking. In this paper, we
present a theoretical analysis of Thompson Sampling, with a focus on
frequentist regret bounds. In this setting, we show that the standard Thompson
Sampling is not aggressive enough in exploring new actions, leading to
suboptimality in some pessimistic situations. A simple modification called
Feel-Good Thompson Sampling, which favors high reward models more aggressively
than the standard Thompson Sampling, is proposed to remedy this problem. We
show that the theoretical framework can be used to derive Bayesian regret
bounds for standard Thompson Sampling, and frequentist regret bounds for
Feel-Good Thompson Sampling. It is shown that in both cases, we can reduce the
bandit regret problem to online least squares regression estimation. For the
frequentist analysis, the online least squares regression bound can be directly
obtained using online aggregation techniques which have been well studied. The
resulting bandit regret bound matches the minimax lower bound in the finite
action case. Moreover, the analysis can be generalized to handle a class of
linearly embeddable contextual bandit problems (which generalizes the popular
linear contextual bandit model). The obtained result again matches the minimax
lower bound. Finally we illustrate that the analysis can be extended to handle
some MDP problems.

    

### [[2110.00874] Fast Line Search for Multi-Task Learning](http://arxiv.org/abs/2110.00874)


  Multi-task learning is a powerful method for solving several tasks jointly by
learning robust representation. Optimization of the multi-task learning model
is a more complex task than a single-task due to task conflict. Based on
theoretical results, convergence to the optimal point is guaranteed when step
size is chosen through line search. But, usually, line search for the step size
is not the best choice due to the large computational time overhead. We propose
a novel idea for line search algorithms in multi-task learning. The idea is to
use latent representation space instead of parameter space for finding step
size. We examined this idea with backtracking line search. We compare this fast
backtracking algorithm with classical backtracking and gradient methods with a
constant learning rate on MNIST, CIFAR-10, Cityscapes tasks. The systematic
empirical study showed that the proposed method leads to more accurate and fast
solution, than the traditional backtracking approach and keep competitive
computational time and performance compared to the constant learning rate
method.

    

### [[2110.00876] Online Incremental Non-Gaussian Inference for SLAM Using Normalizing Flows](http://arxiv.org/abs/2110.00876)


  This paper presents a novel non-Gaussian inference algorithm, Normalizing
Flow iSAM (NF-iSAM), for solving SLAM problems with non-Gaussian factors and/or
nonlinear measurement models. NF-iSAM exploits the expressive power of neural
networks to model normalizing flows that can accurately approximate the joint
posterior of highly nonlinear and non-Gaussian factor graphs. By leveraging the
Bayes tree, NF-iSAM is able to exploit the sparsity structure of SLAM, thus
enabling efficient incremental updates similar to iSAM2, although in the more
challenging non-Gaussian setting. We demonstrate the performance of NF-iSAM and
compare it against state-of-the-art algorithms such as iSAM2 (Gaussian) and
mm-iSAM (non-Gaussian) in synthetic and real range-only SLAM datasets with data
association ambiguity.

    

### [[2110.00881] Weakly Supervised Attention-based Models Using Activation Maps for Citrus Mite and Insect Pest Classification](http://arxiv.org/abs/2110.00881)


  Citrus juices and fruits are commodities with great economic potential in the
international market, but productivity losses caused by mites and other pests
are still far from being a good mark. Despite the integrated pest mechanical
aspect, only a few works on automatic classification have handled images with
orange mite characteristics, which means tiny and noisy regions of interest. On
the computational side, attention-based models have gained prominence in deep
learning research, and, along with weakly supervised learning algorithms, they
have improved tasks performed with some label restrictions. In agronomic
research of pests and diseases, these techniques can improve classification
performance while pointing out the location of mites and insects without
specific labels, reducing deep learning development costs related to generating
bounding boxes. In this context, this work proposes an attention-based
activation map approach developed to improve the classification of tiny regions
called Two-Weighted Activation Mapping, which also produces locations using
feature map scores learned from class labels. We apply our method in a
two-stage network process called Attention-based Multiple Instance Learning
Guided by Saliency Maps. We analyze the proposed approach in two challenging
datasets, the Citrus Pest Benchmark, which was captured directly in the field
using magnifying glasses, and the Insect Pest, a large pest image benchmark. In
addition, we evaluate and compare our models with weakly supervised methods,
such as Attention-based Deep MIL and WILDCAT. The results show that our
classifier is superior to literature methods that use tiny regions in their
classification tasks, surpassing them in all scenarios by at least 16
percentage points. Moreover, our approach infers bounding box locations for
salient insects, even training without any location labels.

    

### [[2110.00894] BRAC+: Improved Behavior Regularized Actor Critic for Offline Reinforcement Learning](http://arxiv.org/abs/2110.00894)


  Online interactions with the environment to collect data samples for training
a Reinforcement Learning (RL) agent is not always feasible due to economic and
safety concerns. The goal of Offline Reinforcement Learning is to address this
problem by learning effective policies using previously collected datasets.
Standard off-policy RL algorithms are prone to overestimations of the values of
out-of-distribution (less explored) actions and are hence unsuitable for
Offline RL. Behavior regularization, which constraints the learned policy
within the support set of the dataset, has been proposed to tackle the
limitations of standard off-policy algorithms. In this paper, we improve the
behavior regularized offline reinforcement learning and propose BRAC+. First,
we propose quantification of the out-of-distribution actions and conduct
comparisons between using Kullback-Leibler divergence versus using Maximum Mean
Discrepancy as the regularization protocol. We propose an analytical upper
bound on the KL divergence as the behavior regularizer to reduce variance
associated with sample based estimations. Second, we mathematically show that
the learned Q values can diverge even using behavior regularized policy update
under mild assumptions. This leads to large overestimations of the Q values and
performance deterioration of the learned policy. To mitigate this issue, we add
a gradient penalty term to the policy evaluation objective. By doing so, the Q
values are guaranteed to converge. On challenging offline RL benchmarks, BRAC+
outperforms the baseline behavior regularized approaches by 40%~87% and the
state-of-the-art approach by 6%.

    

### [[2110.00908] GROWN: GRow Only When Necessary for Continual Learning](http://arxiv.org/abs/2110.00908)


  Catastrophic forgetting is a notorious issue in deep learning, referring to
the fact that Deep Neural Networks (DNN) could forget the knowledge about
earlier tasks when learning new tasks. To address this issue, continual
learning has been developed to learn new tasks sequentially and perform
knowledge transfer from the old tasks to the new ones without forgetting. While
recent structure-based learning methods show the capability of alleviating the
forgetting problem, these methods start from a redundant full-size network and
require a complex learning process to gradually grow-and-prune or search the
network structure for each task, which is inefficient. To address this problem
and enable efficient network expansion for new tasks, we first develop a
learnable sparse growth method eliminating the additional pruning/searching
step in previous structure-based methods. Building on this learnable sparse
growth method, we then propose GROWN, a novel end-to-end continual learning
framework to dynamically grow the model only when necessary. Different from all
previous structure-based methods, GROWN starts from a small seed network,
instead of a full-sized one. We validate GROWN on multiple datasets against
state-of-the-art methods, which shows superior performance in both accuracy and
model size. For example, we achieve 1.0\% accuracy gain on average compared to
the current SOTA results on CIFAR-100 Superclass 20 tasks setting.

    

### [[2110.00911] Enhancing Model Robustness and Fairness with Causality: A Regularization Approach](http://arxiv.org/abs/2110.00911)


  Recent work has raised concerns on the risk of spurious correlations and
unintended biases in statistical machine learning models that threaten model
robustness and fairness. In this paper, we propose a simple and intuitive
regularization approach to integrate causal knowledge during model training and
build a robust and fair model by emphasizing causal features and de-emphasizing
spurious features. Specifically, we first manually identify causal and spurious
features with principles inspired from the counterfactual framework of causal
inference. Then, we propose a regularization approach to penalize causal and
spurious features separately. By adjusting the strength of the penalty for each
type of feature, we build a predictive model that relies more on causal
features and less on non-causal features. We conduct experiments to evaluate
model robustness and fairness on three datasets with multiple metrics.
Empirical results show that the new models built with causal awareness
significantly improve model robustness with respect to counterfactual texts and
model fairness with respect to sensitive attributes.

    

### [[2110.00916] Progressive Transmission and Inference of Deep Learning Models](http://arxiv.org/abs/2110.00916)


  Modern image files are usually progressively transmitted and provide a
preview before downloading the entire image for improved user experience to
cope with a slow network connection. In this paper, with a similar goal, we
propose a progressive transmission framework for deep learning models,
especially to deal with the scenario where pre-trained deep learning models are
transmitted from servers and executed at user devices (e.g., web browser or
mobile). Our progressive transmission allows inferring approximate models in
the middle of file delivery, and quickly provide an acceptable intermediate
outputs. On the server-side, a deep learning model is divided and progressively
transmitted to the user devices. Then, the divided pieces are progressively
concatenated to construct approximate models on user devices. Experiments show
that our method is computationally efficient without increasing the model size
and total transmission time while preserving the model accuracy. We further
demonstrate that our method can improve the user experience by providing the
approximate models especially in a slow connection.

    

### [[2110.00918] Does deep learning model calibration improve performance in class-imbalanced medical image classification?](http://arxiv.org/abs/2110.00918)


  In medical image classification tasks, it is common to find that the number
of normal samples far exceeds the number of abnormal samples. In such
class-imbalanced situations, reliable training of deep neural networks
continues to be a major challenge. Under these circumstances, the predicted
class confidence may be biased toward the majority class. Calibration has been
suggested to alleviate some of these effects. However, there is insufficient
analysis explaining when and whether calibrating a model would be beneficial in
improving performance. In this study, we perform a systematic analysis of the
effect of model calibration on its performance on two medical image modalities,
namely, chest X-rays (CXRs) and fundus images, using various deep learning
classifier backbones. For this, we study the following variations: (i) the
degree of imbalances in the dataset used for training; (ii) calibration
methods; and, (iii) two classification thresholds, namely, default decision
threshold of 0.5, and optimal threshold from precision-recall (PR) curves. Our
results indicate that at the default operating threshold of 0.5, the
performance achieved through calibration is significantly superior (p < 0.05)
to an uncalibrated model. However, at the PR-guided threshold, these gains were
not significantly different (p > 0.05). This finding holds for both image
modalities and at varying degrees of imbalance.

    

### [[2110.00921] Hierarchical Gaussian Process Models for Regression Discontinuity/Kink under Sharp and Fuzzy Designs](http://arxiv.org/abs/2110.00921)


  We propose nonparametric Bayesian estimators for causal inference exploiting
Regression Discontinuity/Kink (RD/RK) under sharp and fuzzy designs. Our
estimators are based on Gaussian Process (GP) regression and classification.
The GP methods are powerful probabilistic modeling approaches that are
advantageous in terms of derivative estimation and uncertainty qualification,
facilitating RK estimation and inference of RD/RK models. These estimators are
extended to hierarchical GP models with an intermediate Bayesian neural network
layer and can be characterized as hybrid deep learning models. Monte Carlo
simulations show that our estimators perform similarly and often better than
competing estimators in terms of precision, coverage and interval length. The
hierarchical GP models improve upon one-layer GP models substantially. An
empirical application of the proposed estimators is provided.

    

### [[2110.00925] Deep Neural Matching Models for Graph Retrieval](http://arxiv.org/abs/2110.00925)


  Graph Retrieval has witnessed continued interest and progress in the past few
years. In thisreport, we focus on neural network based approaches for Graph
matching and retrieving similargraphs from a corpus of graphs. We explore
methods which can soft predict the similaritybetween two graphs. Later, we
gauge the power of a particular baseline (Shortest Path Kernel)and try to model
it in our product graph random walks setting while making it more generalised.

    

### [[2110.00926] Information-Theoretic Generalization Bounds for Iterative Semi-Supervised Learning](http://arxiv.org/abs/2110.00926)


  We consider iterative semi-supervised learning (SSL) algorithms that
iteratively generate pseudo-labels for a large amount unlabelled data to
progressively refine the model parameters. In particular, we seek to understand
the behaviour of the {\em generalization error} of iterative SSL algorithms
using information-theoretic principles. To obtain bounds that are amenable to
numerical evaluation, we first work with a simple model -- namely, the binary
Gaussian mixture model. Our theoretical results suggest that when the class
conditional variances are not too large, the upper bound on the generalization
error decreases monotonically with the number of iterations, but quickly
saturates. The theoretical results on the simple model are corroborated by
extensive experiments on several benchmark datasets such as the MNIST and CIFAR
datasets in which we notice that the generalization error improves after
several pseudo-labelling iterations, but saturates afterwards.

    

### [[2110.00929] Scheduling Optimization Techniques for Neural Network Training](http://arxiv.org/abs/2110.00929)


  Neural network training requires a large amount of computation and thus GPUs
are often used for the acceleration. While they improve the performance, GPUs
are underutilized during the training.This paper proposes out-of-order (ooo)
backprop, an effective scheduling technique for neural network training. By
exploiting the dependencies of gradient computations, ooo backprop enables to
reorder their executions to make the most of the GPU resources. We show that
the GPU utilization in single-GPU, data-parallel, and pipeline-parallel
training can be commonly improve by applying ooo back-prop and prioritizing
critical operations. We propose three scheduling algorithms based on ooo
backprop. For single-GPU training, we schedule with multi-stream out-of-order
computation to mask the kernel launch overhead. In data-parallel training, we
reorder the gradient computations to maximize the overlapping of computation
and parameter communication; in pipeline-parallel training, we prioritize
critical gradient computations to reduce the pipeline stalls.We evaluate our
optimizations with twelve neural networks including a light-weight computer
vision model (MobileNet) and largeNLP models (BERT and GPT-3) with up to forty
eight V100 GPUs.Our scheduling algorithms effectively improve the performance
of single-GPU training as well as data- and pipeline-parallel training.Compared
to the respective state of the art training systems, the throughput is
substantially improved for single-GPU, data-parallel, and pipeline-parallel
training.

    

### [[2110.00942] Artificial Intelligence For Breast Cancer Detection: Trends & Directions](http://arxiv.org/abs/2110.00942)


  In the last decade, researchers working in the domain of computer vision and
Artificial Intelligence (AI) have beefed up their efforts to come up with the
automated framework that not only detects but also identifies stage of breast
cancer. The reason for this surge in research activities in this direction are
mainly due to advent of robust AI algorithms (deep learning), availability of
hardware that can train those robust and complex AI algorithms and
accessibility of large enough dataset required for training AI algorithms.
Different imaging modalities that have been exploited by researchers to
automate the task of breast cancer detection are mammograms, ultrasound,
magnetic resonance imaging, histopathological images or any combination of
them. This article analyzes these imaging modalities and presents their
strengths, limitations and enlists resources from where their datasets can be
accessed for research purpose. This article then summarizes AI and computer
vision based state-of-the-art methods proposed in the last decade, to detect
breast cancer using various imaging modalities. Generally, in this article we
have focused on to review frameworks that have reported results using
mammograms as it is most widely used breast imaging modality that serves as
first test that medical practitioners usually prescribe for the detection of
breast cancer. Second reason of focusing on mammogram imaging modalities is the
availability of its labeled datasets. Datasets availability is one of the most
important aspect for the development of AI based frameworks as such algorithms
are data hungry and generally quality of dataset affects performance of AI
based algorithms. In a nutshell, this research article will act as a primary
resource for the research community working in the field of automated breast
imaging analysis.

    

### [[2110.00944] Kalman Bayesian Neural Networks for Closed-form Online Learning](http://arxiv.org/abs/2110.00944)


  Compared to point estimates calculated by standard neural networks, Bayesian
neural networks (BNN) provide probability distributions over the output
predictions and model parameters, i.e., the weights. Training the weight
distribution of a BNN, however, is more involved due to the intractability of
the underlying Bayesian inference problem and thus, requires efficient
approximations. In this paper, we propose a novel approach for BNN learning via
closed-form Bayesian inference. For this purpose, the calculation of the
predictive distribution of the output and the update of the weight distribution
are treated as Bayesian filtering and smoothing problems, where the weights are
modeled as Gaussian random variables. This allows closed-form expressions for
training the network's parameters in a sequential/online fashion without
gradient descent. We demonstrate our method on several UCI datasets and compare
it to the state of the art.

    

### [[2110.00950] An Unsupervised Video Game Playstyle Metric via State Discretization](http://arxiv.org/abs/2110.00950)


  On playing video games, different players usually have their own playstyles.
Recently, there have been great improvements for the video game AIs on the
playing strength. However, past researches for analyzing the behaviors of
players still used heuristic rules or the behavior features with the
game-environment support, thus being exhausted for the developers to define the
features of discriminating various playstyles. In this paper, we propose the
first metric for video game playstyles directly from the game observations and
actions, without any prior specification on the playstyle in the target game.
Our proposed method is built upon a novel scheme of learning discrete
representations that can map game observations into latent discrete states,
such that playstyles can be exhibited from these discrete states. Namely, we
measure the playstyle distance based on game observations aligned to the same
states. We demonstrate high playstyle accuracy of our metric in experiments on
some video game platforms, including TORCS, RGSK, and seven Atari games, and
for different agents including rule-based AI bots, learning-based AI bots, and
human players.

    

### [[2110.00952] Information Elicitation Meets Clustering](http://arxiv.org/abs/2110.00952)


  In the setting where we want to aggregate people's subjective evaluations,
plurality vote may be meaningless when a large amount of low-effort people
always report "good" regardless of the true quality. "Surprisingly popular"
method, picking the most surprising answer compared to the prior, handle this
issue to some extent. However, it is still not fully robust to people's
strategies. Here in the setting where a large number of people are asked to
answer a small number of multi-choice questions (multi-task, large group), we
propose an information aggregation method that is robust to people's
strategies. Interestingly, this method can be seen as a rotated "surprisingly
popular". It is based on a new clustering method, Determinant MaxImization
(DMI)-clustering, and a key conceptual idea that information elicitation
without ground-truth can be seen as a clustering problem. Of independent
interest, DMI-clustering is a general clustering method that aims to maximize
the volume of the simplex consisting of each cluster's mean multiplying the
product of the cluster sizes. We show that DMI-clustering is invariant to any
non-degenerate affine transformation for all data points. When the data point's
dimension is a constant, DMI-clustering can be solved in polynomial time. In
general, we present a simple heuristic for DMI-clustering which is very similar
to Lloyd's algorithm for k-means. Additionally, we also apply the clustering
idea in the single-task setting and use the spectral method to propose a new
aggregation method that utilizes the second-moment information elicited from
the crowds.

    

### [[2110.00959] Boost Neural Networks by Checkpoints](http://arxiv.org/abs/2110.00959)


  Training multiple deep neural networks (DNNs) and averaging their outputs is
a simple way to improve the predictive performance. Nevertheless, the
multiplied training cost prevents this ensemble method to be practical and
efficient. Several recent works attempt to save and ensemble the checkpoints of
DNNs, which only requires the same computational cost as training a single
network. However, these methods suffer from either marginal accuracy
improvements due to the low diversity of checkpoints or high risk of divergence
due to the cyclical learning rates they adopted. In this paper, we propose a
novel method to ensemble the checkpoints, where a boosting scheme is utilized
to accelerate model convergence and maximize the checkpoint diversity. We
theoretically prove that it converges by reducing exponential loss. The
empirical evaluation also indicates our proposed ensemble outperforms single
model and existing ensembles in terms of accuracy and efficiency. With the same
training budget, our method achieves 4.16% lower error on Cifar-100 and 6.96%
on Tiny-ImageNet with ResNet-110 architecture. Moreover, the adaptive sample
weights in our method make it an effective solution to address the imbalanced
class distribution. In the experiments, it yields up to 5.02% higher accuracy
over single EfficientNet-B0 on the imbalanced datasets.

    

### [[2110.00972] A Robust Scheme for 3D Point Cloud Copy Detection](http://arxiv.org/abs/2110.00972)


  Most existing 3D geometry copy detection research focused on 3D watermarking,
which first embeds ``watermarks'' and then detects the added watermarks.
However, this kind of methods is non-straightforward and may be less robust to
attacks such as cropping and noise. In this paper, we focus on a fundamental
and practical research problem: judging whether a point cloud is plagiarized or
copied to another point cloud in the presence of several manipulations (e.g.,
similarity transformation, smoothing). We propose a novel method to address
this critical problem. Our key idea is first to align the two point clouds and
then calculate their similarity distance. We design three different measures to
compute the similarity. We also introduce two strategies to speed up our
method. Comprehensive experiments and comparisons demonstrate the effectiveness
and robustness of our method in estimating the similarity of two given 3D point
clouds.

    

### [[2110.00973] Graph Pointer Neural Networks](http://arxiv.org/abs/2110.00973)


  Graph Neural Networks (GNNs) have shown advantages in various graph-based
applications. Most existing GNNs assume strong homophily of graph structure and
apply permutation-invariant local aggregation of neighbors to learn a
representation for each node. However, they fail to generalize to heterophilic
graphs, where most neighboring nodes have different labels or features, and the
relevant nodes are distant. Few recent studies attempt to address this problem
by combining multiple hops of hidden representations of central nodes (i.e.,
multi-hop-based approaches) or sorting the neighboring nodes based on attention
scores (i.e., ranking-based approaches). As a result, these approaches have
some apparent limitations. On the one hand, multi-hop-based approaches do not
explicitly distinguish relevant nodes from a large number of multi-hop
neighborhoods, leading to a severe over-smoothing problem. On the other hand,
ranking-based models do not joint-optimize node ranking with end tasks and
result in sub-optimal solutions. In this work, we present Graph Pointer Neural
Networks (GPNN) to tackle the challenges mentioned above. We leverage a pointer
network to select the most relevant nodes from a large amount of multi-hop
neighborhoods, which constructs an ordered sequence according to the
relationship with the central node. 1D convolution is then applied to extract
high-level features from the node sequence. The pointer-network-based ranker in
GPNN is joint-optimized with other parts in an end-to-end manner. Extensive
experiments are conducted on six public node classification datasets with
heterophilic graphs. The results show that GPNN significantly improves the
classification performance of state-of-the-art methods. In addition, analyses
also reveal the privilege of the proposed GPNN in filtering out irrelevant
neighbors and reducing over-smoothing.

    

### [[2110.00981] SecFL: Confidential Federated Learning using TEEs](http://arxiv.org/abs/2110.00981)


  Federated Learning (FL) is an emerging machine learning paradigm that enables
multiple clients to jointly train a model to take benefits from diverse
datasets from the clients without sharing their local training datasets. FL
helps reduce data privacy risks. Unfortunately, FL still exist several issues
regarding privacy and security. First, it is possible to leak sensitive
information from the shared training parameters. Second, malicious clients can
collude with each other to steal data, models from regular clients or corrupt
the global training model. To tackle these challenges, we propose SecFL - a
confidential federated learning framework that leverages Trusted Execution
Environments (TEEs). SecFL performs the global and local training inside TEE
enclaves to ensure the confidentiality and integrity of the computations
against powerful adversaries with privileged access. SecFL provides a
transparent remote attestation mechanism, relying on the remote attestation
provided by TEEs, to allow clients to attest the global training computation as
well as the local training computation of each other. Thus, all malicious
clients can be detected using the remote attestation mechanisms.

    

### [[2110.00987] Motif-based Graph Self-Supervised Learning forMolecular Property Prediction](http://arxiv.org/abs/2110.00987)


  Predicting molecular properties with data-driven methods has drawn much
attention in recent years. Particularly, Graph Neural Networks (GNNs) have
demonstrated remarkable success in various molecular generation and prediction
tasks. In cases where labeled data is scarce, GNNs can be pre-trained on
unlabeled molecular data to first learn the general semantic and structural
information before being fine-tuned for specific tasks. However, most existing
self-supervised pre-training frameworks for GNNs only focus on node-level or
graph-level tasks. These approaches cannot capture the rich information in
subgraphs or graph motifs. For example, functional groups (frequently-occurred
subgraphs in molecular graphs) often carry indicative information about the
molecular properties. To bridge this gap, we propose Motif-based Graph
Self-supervised Learning (MGSSL) by introducing a novel self-supervised motif
generation framework for GNNs. First, for motif extraction from molecular
graphs, we design a molecule fragmentation method that leverages a
retrosynthesis-based algorithm BRICS and additional rules for controlling the
size of motif vocabulary. Second, we design a general motif-based generative
pre-training framework in which GNNs are asked to make topological and label
predictions. This generative framework can be implemented in two different
ways, i.e., breadth-first or depth-first. Finally, to take the multi-scale
information in molecular graphs into consideration, we introduce a multi-level
self-supervised pre-training. Extensive experiments on various downstream
benchmark tasks show that our methods outperform all state-of-the-art
baselines.

    

### [[2110.00998] Simple Recurrent Neural Networks is all we need for clinical events predictions using EHR data](http://arxiv.org/abs/2110.00998)


  Recently, there is great interest to investigate the application of deep
learning models for the prediction of clinical events using electronic health
records (EHR) data. In EHR data, a patient's history is often represented as a
sequence of visits, and each visit contains multiple events. As a result, deep
learning models developed for sequence modeling, like recurrent neural networks
(RNNs) are common architecture for EHR-based clinical events predictive models.
While a large variety of RNN models were proposed in the literature, it is
unclear if complex architecture innovations will offer superior predictive
performance. In order to move this field forward, a rigorous evaluation of
various methods is needed. In this study, we conducted a thorough benchmark of
RNN architectures in modeling EHR data. We used two prediction tasks: the risk
for developing heart failure and the risk of early readmission for inpatient
hospitalization. We found that simple gated RNN models, including GRUs and
LSTMs, often offer competitive results when properly tuned with Bayesian
Optimization, which is in line with similar to findings in the natural language
processing (NLP) domain. For reproducibility, Our codebase is shared at
this https URL.

    

### [[2110.01015] Spatio-Temporal Video Representation Learning for AI Based Video Playback Style Prediction](http://arxiv.org/abs/2110.01015)


  Ever-increasing smartphone-generated video content demands intelligent
techniques to edit and enhance videos on power-constrained devices. Most of the
best performing algorithms for video understanding tasks like action
recognition, localization, etc., rely heavily on rich spatio-temporal
representations to make accurate predictions. For effective learning of the
spatio-temporal representation, it is crucial to understand the underlying
object motion patterns present in the video. In this paper, we propose a novel
approach for understanding object motions via motion type classification. The
proposed motion type classifier predicts a motion type for the video based on
the trajectories of the objects present. Our classifier assigns a motion type
for the given video from the following five primitive motion classes: linear,
projectile, oscillatory, local and random. We demonstrate that the
representations learned from the motion type classification generalizes well
for the challenging downstream task of video retrieval. Further, we proposed a
recommendation system for video playback style based on the motion type
classifier predictions.

    

### [[2110.01035] RAP-Net: Region Attention Predictive Network for Precipitation Nowcasting](http://arxiv.org/abs/2110.01035)


  Natural disasters caused by heavy rainfall often cost huge loss of life and
property. To avoid it, the task of precipitation nowcasting is imminent. To
solve the problem, increasingly deep learning methods are proposed to forecast
future radar echo images and then the predicted maps have converted the
distribution of rainfall. The prevailing spatiotemporal sequence prediction
methods apply ConvRNN structure which combines the Convolution and Recurrent
neural network. Although improvements based on ConvRNN achieve remarkable
success, these methods ignore capturing both local and global spatial features
simultaneously, which degrades the nowcasting in the region of heavy rainfall.
To address this issue, we proposed the Region Attention Block (RAB) and embed
it into ConvRNN to enhance the forecast in the area with strong rainfall.
Besides, the ConvRNN models are hard to memory longer history representations
with limited parameters. Considering it, we propose Recall Attention Mechanism
(RAM) to improve the prediction. By preserving longer temporal information, RAM
contributes to the forecasting, especially in the middle rainfall intensity.
The experiments show that the proposed model Region Attention Predictive
Network (RAP-Net) has outperformed the state-of-art method.

    

### [[2110.01050] Marginally calibrated response distributions for end-to-end learning in autonomous driving](http://arxiv.org/abs/2110.01050)


  End-to-end learners for autonomous driving are deep neural networks that
predict the instantaneous steering angle directly from images of the
ahead-lying street. These learners must provide reliable uncertainty estimates
for their predictions in order to meet safety requirements and initiate a
switch to manual control in areas of high uncertainty. Yet end-to-end learners
typically only deliver point predictions, since distributional predictions are
associated with large increases in training time or additional computational
resources during prediction. To address this shortcoming we investigate
efficient and scalable approximate inference for the implicit copula neural
linear model of Klein, Nott and Smith (2021) in order to quantify uncertainty
for the predictions of end-to-end learners. The result are densities for the
steering angle that are marginally calibrated, i.e.~the average of the
estimated densities equals the empirical distribution of steering angles. To
ensure the scalability to large $n$ regimes, we develop efficient estimation
based on variational inference as a fast alternative to computationally
intensive, exact inference via Hamiltonian Monte Carlo. We demonstrate the
accuracy and speed of the variational approach in comparison to Hamiltonian
Monte Carlo on two end-to-end learners trained for highway driving using the
comma2k19 data set. The implicit copula neural linear model delivers accurate
calibration, high-quality prediction intervals and allows to identify
overconfident learners. Our approach also contributes to the explainability of
black-box end-to-end learners, since predictive densities can be used to
understand which steering actions the end-to-end learner sees as valid.

    

### [[2110.01052] Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](http://arxiv.org/abs/2110.01052)


  We introduce Learn then Test (LTT), a framework for calibrating machine
learning models so that their predictions satisfy explicit, finite-sample
statistical guarantees regardless of the underlying model and (unknown)
data-generating distribution. The framework addresses, among other examples,
false discovery rate control in multi-label classification,
intersection-over-union control in instance segmentation, and the simultaneous
control of the type-1 error of outlier detection and confidence set coverage in
classification or regression. To accomplish this, we solve a key technical
challenge: the control of arbitrary risks that are not necessarily monotonic.
Our main insight is to reframe the risk-control problem as multiple hypothesis
testing, enabling techniques and mathematical arguments different from those in
the previous literature. We use our framework to provide new calibration
methods for several core machine learning tasks with detailed worked examples
in computer vision.

    

### [[2110.01053] Treeging](http://arxiv.org/abs/2110.01053)


  Treeging combines the flexible mean structure of regression trees with the
covariance-based prediction strategy of kriging into the base learner of an
ensemble prediction algorithm. In so doing, it combines the strengths of the
two primary types of spatial and space-time prediction models: (1) models with
flexible mean structures (often machine learning algorithms) that assume
independently distributed data, and (2) kriging or Gaussian Process (GP)
prediction models with rich covariance structures but simple mean structures.
We investigate the predictive accuracy of treeging across a thorough and widely
varied battery of spatial and space-time simulation scenarios, comparing it to
ordinary kriging, random forest and ensembles of ordinary kriging base
learners. Treeging performs well across the board, whereas kriging suffers when
dependence is weak or in the presence of spurious covariates, and random forest
suffers when the covariates are less informative. Treeging also outperforms
these competitors in predicting atmospheric pollutants (ozone and PM$_{2.5}$)
in several case studies. We examine sensitivity to tuning parameters (number of
base learners and training data sampling proportion), finding they follow the
familiar intuition of their random forest counterparts. We include a discussion
of scaleability, noting that any covariance approximation techniques that
expedite kriging (GP) may be similarly applied to expedite treeging.

    

### [[2110.01072] Active Learning for Contextual Search with Binary Feedbacks](http://arxiv.org/abs/2110.01072)


  In this paper, we study the learning problem in contextual search, which is
motivated by applications such as first-price auction, personalized medicine
experiments, and feature-based pricing experiments. In particular, for a
sequence of arriving context vectors, with each context associated with an
underlying value, the decision-maker either makes a query at a certain point or
skips the context. The decision-maker will only observe the binary feedback on
the relationship between the query point and the value associated with the
context. We study a PAC learning setting, where the goal is to learn the
underlying mean value function in context with a minimum number of queries. To
address this challenge, we propose a tri-section search approach combined with
a margin-based active learning method. We show that the algorithm only needs to
make $O(1/\varepsilon^2)$ queries to achieve an $\epsilon$-estimation accuracy.
This sample complexity significantly reduces the required sample complexity in
the passive setting, at least $\Omega(1/\varepsilon^4)$.

    

### [[2110.01101] Parallel Actors and Learners: A Framework for Generating Scalable RL Implementations](http://arxiv.org/abs/2110.01101)


  Reinforcement Learning (RL) has achieved significant success in application
domains such as robotics, games, health care and others. However, training RL
agents is very time consuming. Current implementations exhibit poor performance
due to challenges such as irregular memory accesses and synchronization
overheads.
In this work, we propose a framework for generating scalable reinforcement
learning implementations on multicore systems. Replay Buffer is a key component
of RL algorithms which facilitates storage of samples obtained from
environmental interactions and their sampling for the learning process. We
define a new data structure for prioritized replay buffer based on $K$-ary sum
tree that supports asynchronous parallel insertions, sampling, and priority
updates. To address the challenge of irregular memory accesses, we propose a
novel data layout to store the nodes of the sum tree that reduces the number of
cache misses. Additionally, we propose \textit{lazy writing} mechanism to
reduce synchronization overheads of the replay buffer. Our framework employs
parallel actors to concurrently collect data via environmental interactions,
and parallel learners to perform stochastic gradient descent using the
collected data. Our framework supports a wide range of reinforcement learning
algorithms including DQN, DDPG, TD3, SAC, etc. We demonstrate the effectiveness
of our framework in accelerating RL algorithms by performing experiments on CPU
+ GPU platform using OpenAI benchmarks. Our results show that the performance
of our approach scales linearly with the number of cores. Compared with the
baseline approaches, we reduce the convergence time by 3.1x$\sim$10.8x. By
plugging our replay buffer implementation into existing open source
reinforcement learning frameworks, we achieve 1.1x$\sim$2.1x speedup for
sequential executions.

    

### [[2110.01107] TinyFedTL: Federated Transfer Learning on Tiny Devices](http://arxiv.org/abs/2110.01107)


  TinyML has rose to popularity in an era where data is everywhere. However,
the data that is in most demand is subject to strict privacy and security
guarantees. In addition, the deployment of TinyML hardware in the real world
has significant memory and communication constraints that traditional ML fails
to address. In light of these challenges, we present TinyFedTL, the first
implementation of federated transfer learning on a resource-constrained
microcontroller.

    

### [[2110.01108] Human-Centered AI for Data Science: A Systematic Approach](http://arxiv.org/abs/2110.01108)


  Human-Centered AI (HCAI) refers to the research effort that aims to design
and implement AI techniques to support various human tasks, while taking human
needs into consideration and preserving human control. In this short position
paper, we illustrate how we approach HCAI using a series of research projects
around Data Science (DS) works as a case study. The AI techniques built for
supporting DS works are collectively referred to as AutoML systems, and their
goals are to automate some parts of the DS workflow. We illustrate a three-step
systematical research approach(i.e., explore, build, and integrate) and four
practical ways of implementation for HCAI systems. We argue that our work is a
cornerstone towards the ultimate future of Human-AI Collaboration for DS and
beyond, where AI and humans can take complementary and indispensable roles to
achieve a better outcome and experience.

    

### [[2110.01109] xFAIR: Better Fairness via Model-based Rebalancing of Protected Attributes](http://arxiv.org/abs/2110.01109)


  Machine learning software can generate models that inappropriately
discriminate against specific protected social groups (e.g., groups based on
gender, ethnicity, etc). Motivated by those results, software engineering
researchers have proposed many methods for mitigating those discriminatory
effects. While those methods are effective in mitigating bias, few of them can
provide explanations on what is the cause of bias. Here we propose xFAIR, a
model-based extrapolation method, that is capable of both mitigating bias and
explaining the cause. In our xFAIR approach, protected attributes are
represented by models learned from the other independent variables (and these
models offer extrapolations over the space between existing examples). We then
use the extrapolation models to relabel protected attributes, which aims to
offset the biased predictions of the classification model via rebalancing the
distribution of protected attributes. The experiments of this paper show that,
without compromising(original) model performance,xFAIRcan achieve significantly
better group and individual fairness (as measured in different metrics)than
benchmark methods. Moreover, when compared to another instance-based
rebalancing method, our model-based approach shows faster runtime and thus
better scalability

    

### [[2110.01110] Safe Control with Neural Network Dynamic Models](http://arxiv.org/abs/2110.01110)


  Safety is critical in autonomous robotic systems. A safe control law ensures
forward invariance of a safe set (a subset in the state space). It has been
extensively studied regarding how to derive a safe control law with a
control-affine analytical dynamic model. However, in complex environments and
tasks, it is challenging and time-consuming to obtain a principled analytical
model of the system. In these situations, data-driven learning is extensively
used and the learned models are encoded in neural networks. How to formally
derive a safe control law with Neural Network Dynamic Models (NNDM) remains
unclear due to the lack of computationally tractable methods to deal with these
black-box functions. In fact, even finding the control that minimizes an
objective for NNDM without any safety constraint is still challenging. In this
work, we propose MIND-SIS (Mixed Integer for Neural network Dynamic model with
Safety Index Synthesis), the first method to derive safe control laws for NNDM.
The method includes two parts: 1) SIS: an algorithm for the offline synthesis
of the safety index (also called as barrier function), which uses evolutionary
methods and 2) MIND: an algorithm for online computation of the optimal and
safe control signal, which solves a constrained optimization using a
computationally efficient encoding of neural networks. It has been
theoretically proved that MIND-SIS guarantees forward invariance and finite
convergence. And it has been numerically validated that MIND-SIS achieves safe
and optimal control of NNDM. From our experiments, the optimality gap is less
than $10^{-8}$, and the safety constraint violation is $0$.

    

### [[2110.01127] Deep Learning for Principal-Agent Mean Field Games](http://arxiv.org/abs/2110.01127)


  Here, we develop a deep learning algorithm for solving Principal-Agent (PA)
mean field games with market-clearing conditions -- a class of problems that
have thus far not been studied and one that poses difficulties for standard
numerical methods. We use an actor-critic approach to optimization, where the
agents form a Nash equilibria according to the principal's penalty function,
and the principal evaluates the resulting equilibria. The inner problem's Nash
equilibria is obtained using a variant of the deep backward stochastic
differential equation (BSDE) method modified for McKean-Vlasov forward-backward
SDEs that includes dependence on the distribution over both the forward and
backward processes. The outer problem's loss is further approximated by a
neural net by sampling over the space of penalty functions. We apply our
approach to a stylized PA problem arising in Renewable Energy Certificate (REC)
markets, where agents may rent clean energy production capacity, trade RECs,
and expand their long-term capacity to navigate the market at maximum profit.
Our numerical results illustrate the efficacy of the algorithm and lead to
interesting insights into the nature of optimal PA interactions in the
mean-field limit of these markets.

    

### [[2110.01154] An Analysis of Super-Net Heuristics in Weight-Sharing NAS](http://arxiv.org/abs/2110.01154)


  Weight sharing promises to make neural architecture search (NAS) tractable
even on commodity hardware. Existing methods in this space rely on a diverse
set of heuristics to design and train the shared-weight backbone network,
a.k.a. the super-net. Since heuristics substantially vary across different
methods and have not been carefully studied, it is unclear to which extent they
impact super-net training and hence the weight-sharing NAS algorithms. In this
paper, we disentangle super-net training from the search algorithm, isolate 14
frequently-used training heuristics, and evaluate them over three benchmark
search spaces. Our analysis uncovers that several commonly-used heuristics
negatively impact the correlation between super-net and stand-alone
performance, whereas simple, but often overlooked factors, such as proper
hyper-parameter settings, are key to achieve strong performance. Equipped with
this knowledge, we show that simple random search achieves competitive
performance to complex state-of-the-art NAS algorithms when the super-net is
properly trained.

    

### [[2110.01160] Beyond Topics: Discovering Latent Healthcare Objectives from Event Sequences](http://arxiv.org/abs/2110.01160)


  A meaningful understanding of clinical protocols and patient pathways helps
improve healthcare outcomes. Electronic health records (EHR) reflect real-world
treatment behaviours that are used to enhance healthcare management but present
challenges; protocols and pathways are often loosely defined and with elements
frequently not recorded in EHRs, complicating the enhancement. To solve this
challenge, healthcare objectives associated with healthcare management
activities can be indirectly observed in EHRs as latent topics. Topic models,
such as Latent Dirichlet Allocation (LDA), are used to identify latent patterns
in EHR data. However, they do not examine the ordered nature of EHR sequences,
nor do they appraise individual events in isolation. Our novel approach, the
Categorical Sequence Encoder (CaSE) addresses these shortcomings. The
sequential nature of EHRs is captured by CaSE's event-level representations,
revealing latent healthcare objectives. In synthetic EHR sequences, CaSE
outperforms LDA by up to 37% at identifying healthcare objectives. In the
real-world MIMIC-III dataset, CaSE identifies meaningful representations that
could critically enhance protocol and pathway development.

    

### [[2110.01164] Decoupling Speaker-Independent Emotions for Voice Conversion Via Source-Filter Networks](http://arxiv.org/abs/2110.01164)


  Emotional voice conversion (VC) aims to convert a neutral voice to an
emotional (e.g. happy) one while retaining the linguistic information and
speaker identity. We note that the decoupling of emotional features from other
speech information (such as speaker, content, etc.) is the key to achieving
remarkable performance. Some recent attempts about speech representation
decoupling on the neutral speech can not work well on the emotional speech, due
to the more complex acoustic properties involved in the latter. To address this
problem, here we propose a novel Source-Filter-based Emotional VC model (SFEVC)
to achieve proper filtering of speaker-independent emotion features from both
the timbre and pitch features. Our SFEVC model consists of multi-channel
encoders, emotion separate encoders, and one decoder. Note that all encoder
modules adopt a designed information bottlenecks auto-encoder. Additionally, to
further improve the conversion quality for various emotions, a novel two-stage
training strategy based on the 2D Valence-Arousal (VA) space was proposed.
Experimental results show that the proposed SFEVC along with a two-stage
training strategy outperforms all baselines and achieves the state-of-the-art
performance in speaker-independent emotional VC with nonparallel data.

    

### [[2110.01165] DESTRESS: Computation-Optimal and Communication-Efficient Decentralized Nonconvex Finite-Sum Optimization](http://arxiv.org/abs/2110.01165)


  Emerging applications in multi-agent environments such as internet-of-things,
networked sensing, autonomous systems and federated learning, call for
decentralized algorithms for finite-sum optimizations that are
resource-efficient in terms of both computation and communication. In this
paper, we consider the prototypical setting where the agents work
collaboratively to minimize the sum of local loss functions by only
communicating with their neighbors over a predetermined network topology. We
develop a new algorithm, called DEcentralized STochastic REcurSive gradient
methodS (DESTRESS) for nonconvex finite-sum optimization, which matches the
optimal incremental first-order oracle (IFO) complexity of centralized
algorithms for finding first-order stationary points, while maintaining
communication efficiency. Detailed theoretical and numerical comparisons
corroborate that the resource efficiencies of DESTRESS improve upon prior
decentralized algorithms over a wide range of parameter regimes. DESTRESS
leverages several key algorithm design ideas including stochastic recursive
gradient updates with mini-batches for local computation, gradient tracking
with extra mixing (i.e., multiple gossiping rounds) for per-iteration
communication, together with careful choices of hyper-parameters and new
analysis frameworks to provably achieve a desirable computation-communication
trade-off.

    

### [[2110.01167] Trustworthy AI: From Principles to Practices](http://arxiv.org/abs/2110.01167)


  Fast developing artificial intelligence (AI) technology has enabled various
applied systems deployed in the real world, impacting people's everyday lives.
However, many current AI systems were found vulnerable to imperceptible
attacks, biased against underrepresented groups, lacking in user privacy
protection, etc., which not only degrades user experience but erodes the
society's trust in all AI systems. In this review, we strive to provide AI
practitioners a comprehensive guide towards building trustworthy AI systems. We
first introduce the theoretical framework of important aspects of AI
trustworthiness, including robustness, generalization, explainability,
transparency, reproducibility, fairness, privacy preservation, alignment with
human values, and accountability. We then survey leading approaches in these
aspects in the industry. To unify the current fragmented approaches towards
trustworthy AI, we propose a systematic approach that considers the entire
lifecycle of AI systems, ranging from data acquisition to model development, to
development and deployment, finally to continuous monitoring and governance. In
this framework, we offer concrete action items to practitioners and societal
stakeholders (e.g., researchers and regulators) to improve AI trustworthiness.
Finally, we identify key opportunities and challenges in the future development
of trustworthy AI systems, where we identify the need for paradigm shift
towards comprehensive trustworthy AI systems.

    

### [[2110.01171] Deep Fraud Detection on Non-attributed Graph](http://arxiv.org/abs/2110.01171)


  Fraud detection problems are usually formulated as a machine learning problem
on a graph. Recently, Graph Neural Networks (GNNs) have shown solid performance
on fraud detection. The successes of most previous methods heavily rely on rich
node features and high-fidelity labels. However, labeled data is scarce in
large-scale industrial problems, especially for fraud detection where new
patterns emerge from time to time. Meanwhile, node features are also limited
due to privacy and other constraints. In this paper, two improvements are
proposed: 1) We design a graph transformation method capturing the structural
information to facilitate GNNs on non-attributed fraud graphs. 2) We propose a
novel graph pre-training strategy to leverage more unlabeled data via
contrastive learning. Experiments on a large-scale industrial dataset
demonstrate the effectiveness of the proposed framework for fraud detection.

    

### [[2110.01185] Adding Quaternion Representations to Attention Networks for Classification](http://arxiv.org/abs/2110.01185)


  This paper introduces a novel modification to axial-attention networks to
improve their image classification accuracy. The modification involves
supplementing axial-attention modules with quaternion input representations to
improve image classification accuracy. We chose axial-attention networks
because they factor 2D attention operations into two consecutive 1D operations
(similar to separable convolution) and are thus less resource intensive than
non-axial attention networks. We chose a quaternion encoder because of they
share weights across four real-valued input channels and the weight-sharing has
been shown to produce a more interlinked/interwoven output representation. We
hypothesize that an attention module can be more effective using these
interlinked representations as input. Our experiments support this hypothesis
as reflected in the improved classification accuracy compared to standard
axial-attention networks. We think this happens because the attention modules
have better input representations to work with.

    

### [[2110.01191] 3D-Transformer: Molecular Representation with Transformer in 3D Space](http://arxiv.org/abs/2110.01191)


  Spatial structures in the 3D space are important to determine molecular
properties. Recent papers use geometric deep learning to represent molecules
and predict properties. These papers, however, are computationally expensive in
capturing long-range dependencies of input atoms; and have not considered the
non-uniformity of interatomic distances, thus failing to learn
context-dependent representations at different scales. To deal with such
issues, we introduce 3D-Transformer, a variant of the Transformer for molecular
representations that incorporates 3D spatial information. 3D-Transformer
operates on a fully-connected graph with direct connections between atoms. To
cope with the non-uniformity of interatomic distances, we develop a multi-scale
self-attention module that exploits local fine-grained patterns with increasing
contextual scales. As molecules of different sizes rely on different kinds of
spatial features, we design an adaptive position encoding module that adopts
different position encoding methods for small and large molecules. Finally, to
attain the molecular representation from atom embeddings, we propose an
attentive farthest point sampling algorithm that selects a portion of atoms
with the assistance of attention scores, overcoming handicaps of the virtual
node and previous distance-dominant downsampling methods. We validate
3D-Transformer across three important scientific domains: quantum chemistry,
material science, and proteomics. Our experiments show significant improvements
over state-of-the-art models on the crystal property prediction task and the
protein-ligand binding affinity prediction task, and show better or competitive
performance in quantum chemistry molecular datasets. This work provides clear
evidence that biochemical tasks can gain consistent benefits from 3D molecular
representations and different tasks require different position encoding
methods.

    

### [[2110.01200] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks](http://arxiv.org/abs/2110.01200)


  Artefacts that differentiate spoofed from bona-fide utterances can reside in
spectral or temporal domains. Their reliable detection usually depends upon
computationally demanding ensemble systems where each subsystem is tuned to
some specific artefacts. We seek to develop an efficient, single system that
can detect a broad range of different spoofing attacks without score-level
ensembles. We propose a novel heterogeneous stacking graph attention layer
which models artefacts spanning heterogeneous temporal and spectral domains
with a heterogeneous attention mechanism and a stack node. With a new max graph
operation that involves a competitive mechanism and an extended readout scheme,
our approach, named AASIST, outperforms the current state-of-the-art by 20%
relative. Even a lightweight variant, AASIST-L, with only 85K parameters,
outperforms all competing systems.

    

### [[2110.01207] Row-clustering of a Point Process-valued Matrix](http://arxiv.org/abs/2110.01207)


  Structured point process data harvested from various platforms poses new
challenges to the machine learning community. By imposing a matrix structure to
repeatedly observed marked point processes, we propose a novel mixture model of
multi-level marked point processes for identifying potential heterogeneity in
the observed data. Specifically, we study a matrix whose entries are marked
log-Gaussian Cox processes and cluster rows of such a matrix. An efficient
semi-parametric Expectation-Solution (ES) algorithm combined with functional
principal component analysis (FPCA) of point processes is proposed for model
estimation. The effectiveness of the proposed framework is demonstrated through
simulation studies and a real data analysis.

    

### [[2110.01212] Inducing Equilibria via Incentives: Simultaneous Design-and-Play Finds Global Optima](http://arxiv.org/abs/2110.01212)


  To induce a desired equilibrium in a social system comprised of
self-interested agents, economic incentives (e.g., taxes, tolls, and subsidies)
are often required to correct an inefficient outcome. Such an incentive design
problem naturally possesses a bi-level structure, in which an upper-level
"designer" revises the payoffs of the agents with incentives while anticipating
the response of the agents, who play a non-cooperative game at the lower level.
The existing bi-level optimization algorithms developed in machine learning
raise a dilemma when applied to this problem: anticipating how incentives
affect the agents at equilibrium requires solving the equilibrium problem
repeatedly, which is computationally inefficient; bypassing the time-consuming
step of equilibrium-finding can reduce the computational cost, but may lead to
a sub-optimal solution. Therefore, we propose an efficient method that tackles
the designer's and agents' problems simultaneously in a single loop. At each
iteration, both the designer and the agents only move one step based on the
first-order information. In the proposed scheme, although the designer does not
solve the equilibrium problem repeatedly, it can anticipate the overall
influence of the incentives on the agents, which guarantees optimality. We
prove that the algorithm converges to the global optima at a sublinear rate for
a broad class of games.

    

### [[2110.01219] Hit and Lead Discovery with Explorative RL and Fragment-based Molecule Generation](http://arxiv.org/abs/2110.01219)


  Recently, utilizing reinforcement learning (RL) to generate molecules with
desired properties has been highlighted as a promising strategy for drug
design. A molecular docking program - a physical simulation that estimates
protein-small molecule binding affinity - can be an ideal reward scoring
function for RL, as it is a straightforward proxy of the therapeutic potential.
Still, two imminent challenges exist for this task. First, the models often
fail to generate chemically realistic and pharmacochemically acceptable
molecules. Second, the docking score optimization is a difficult exploration
problem that involves many local optima and less smooth surfaces with respect
to molecular structure. To tackle these challenges, we propose a novel RL
framework that generates pharmacochemically acceptable molecules with large
docking scores. Our method - Fragment-based generative RL with Explorative
Experience replay for Drug design (FREED) - constrains the generated molecules
to a realistic and qualified chemical space and effectively explores the space
to find drugs by coupling our fragment-based generation method and a novel
error-prioritized experience replay (PER). We also show that our model performs
well on both de novo and scaffold-based schemes. Our model produces molecules
of higher quality compared to existing methods while achieving state-of-the-art
performance on two of three targets in terms of the docking scores of the
generated molecules. We further show with ablation studies that our method,
predictive error-PER (FREED(PE)), significantly improves the model performance.

    

### [[2110.01221] DenDrift: A Drift-Aware Algorithm for Host Profiling](http://arxiv.org/abs/2110.01221)


  Detecting and reacting to unauthorized actions is an essential task in
security monitoring. What make this task challenging are the large number and
various categories of hosts and processes to monitor. To these we should add
the lack of an exact definition of normal behavior for each category. Host
profiling using stream clustering algorithms is an effective means of analyzing
hosts' behaviors, categorizing them, and identifying atypical ones. However,
unforeseen changes in behavioral data (i.e. concept drift) make the obtained
profiles unreliable. DenStream is a well-known stream clustering algorithm,
which can be effectively used for host profiling. This algorithm is an
incremental extension of DBSCAN which is a non-parametric algorithm widely used
in real-world clustering applications. Recent experimental studies indicate
that DenStream is not robust against concept drift. In this paper, we present
DenDrift as a drift-aware host profiling algorithm based on DenStream. DenDrift
relies on non-negative matrix factorization for dimensionality reduction and
Page-Hinckley test for drift detection. We have done experiments on both
synthetic and industrial datasets and the results affirm the robustness of
DenDrift against abrupt, gradual and incremental drifts.

    

### [[2110.01230] Identifiability in Exact Multilayer Sparse Matrix Factorization](http://arxiv.org/abs/2110.01230)


  Many well-known matrices Z are associated to fast transforms corresponding to
factorizations of the form Z = X^(L). .. X^(1) , where each factor X^(l) is
sparse. Based on general result for the case with two factors, established in a
companion paper, we investigate essential uniqueness of such factorizations. We
show some identifiability results for the sparse factorization into two factors
of the discrete Fourier Transform, discrete cosine transform or discrete sine
transform matrices of size N = 2^L , when enforcing N/2-sparsity by column on
the left factor, and 2-sparsity by row on the right factor. We also show that
the analysis with two factors can be extended to the multilayer case, based on
a hierarchical factorization method. We prove that any matrix which is the
product of L factors whose supports are exactly the so-called butterfly
supports, admits a unique sparse factorization into L factors. This applies in
particular to the Hadamard or the discrete Fourier transform matrix of size 2^L .

    

### [[1803.06510] Hidden Integrality and Semi-random Robustness of SDP Relaxation for Sub-Gaussian Mixture Model](http://arxiv.org/abs/1803.06510)


  We consider the problem of estimating the discrete clustering structures
under the Sub-Gaussian Mixture Model. Our main results establish a hidden
integrality property of a semidefinite programming (SDP) relaxation for this
problem: while the optimal solution to the SDP is not integer-valued in
general, its estimation error can be upper bounded by that of an idealized
integer program. The error of the integer program, and hence that of the SDP,
are further shown to decay exponentially in the signal-to-noise ratio. In
addition, we show that the SDP relaxation is robust under the semi-random
setting in which an adversary can modify the data generated from the mixture
model. In particular, we generalize the hidden integrality property to the
semi-random model and thereby show that SDP achieves the optimal error bound in
this setting. These results together highlight the "global-to-local" mechanism
that drives the performance of the SDP relaxation.
To the best of our knowledge, our result is the first exponentially decaying
error bound for convex relaxations of mixture models. A corollary of our
results shows that in certain regimes the SDP solutions are in fact integral
and exact. More generally, our results establish sufficient conditions for the
SDP to correctly recover the cluster memberships of $(1-\delta)$ fraction of
the points for any $\delta\in(0,1)$. As a special case, we show that under the
$d$-dimensional Stochastic Ball Model, SDP achieves non-trivial (sometimes
exact) recovery when the center separation is as small as $\sqrt{1/d}$, which
improves upon previous exact recovery results that require constant separation.

    

### [[1808.01132] Multitask Gaussian Process with Hierarchical Latent Interactions](http://arxiv.org/abs/1808.01132)


  Multitask Gaussian process (MTGP) is powerful for joint learning of multiple
tasks with complicated correlation patterns. However, due to the assembling of
additive independent latent functions, all current MTGPs including the salient
linear model of coregionalization (LMC) and convolution frameworks cannot
effectively represent and learn the hierarchical latent interactions between
its latent functions. In this paper, we further investigate the interactions in
LMC of MTGP and then propose a novel kernel representation of the hierarchical
interactions, which ameliorates both the expressiveness and the
interpretability of MTGP. Specifically, we express the interaction as a product
of function interaction and coefficient interaction. The function interaction
is modeled by using cross convolution of latent functions. The coefficient
interaction between the LMCs is described as a cross coregionalization term. We
validate that considering the interactions can promote knowledge transferring
in MTGP and compare our approach with some state-of-the-art MTGPs on both
synthetic- and real-world datasets.

    

### [[1903.08519] Topology-based Representative Datasets to Reduce Neural Network Training Resources](http://arxiv.org/abs/1903.08519)


  One of the main drawbacks of the practical use of neural networks is the long
time required in the training process. Such a training process consists of an
iterative change of parameters trying to minimize a loss function. These
changes are driven by a dataset, which can be seen as a set of labelled points
in an n-dimensional space. In this paper, we explore the concept of are
representative dataset which is a dataset smaller than the original one,
satisfying a nearness condition independent of isometric transformations.
Representativeness is measured using persistence diagrams (a computational
topology tool) due to its computational efficiency. We prove that the accuracy
of the learning process of a neural network on a representative dataset is
"similar" to the accuracy on the original dataset when the neural network
architecture is a perceptron and the loss function is the mean squared error.
These theoretical results accompanied by experimentation open a door to
reducing the size of the dataset to gain time in the training process of any
neural network.

    

### [[1906.05205] Warping Resilient Scalable Anomaly Detection in Time Series](http://arxiv.org/abs/1906.05205)


  Time series data is ubiquitous in the real-world problems across various
domains including healthcare, social media, and crime surveillance. Detecting
anomalies, or irregular and rare events, in time series data, can enable us to
find abnormal events in any natural phenomena, which may require special
treatment. Moreover, labeled instances of anomaly are hard to get in time
series data. On the other hand, time series data, due to its nature, often
exhibits localized expansions and compressions in the time dimension which is
called warping. These two challenges make it hard to detect anomalies in time
series as often such warpings could get detected as anomalies erroneously. Our
objective is to build an anomaly detection model that is robust to such warping
variations. In this paper, we propose a novel unsupervised time series anomaly
detection method, WaRTEm-AD, that operates in two stages. Within the key stage
of representation learning, we employ data augmentation through bespoke time
series operators which are passed through a twin autoencoder architecture to
learn warping-robust representations for time series data. Second, adaptations
of state-of-the-art anomaly detection methods are employed on the learnt
representations to identify anomalies. We will illustrate that WaRTEm-AD is
designed to detect two types of time series anomalies: point and sequence
anomalies. We compare WaRTEm-AD with the state-of-the-art baselines and
establish the effectiveness of our method both in terms of anomaly detection
performance and computational efficiency.

    

### [[1909.05006] Boltzmann machine learning and regularization methods for inferring evolutionary fields and couplings from a multiple sequence alignment](http://arxiv.org/abs/1909.05006)


  The inverse Potts problem to infer a Boltzmann distribution for homologous
protein sequences from their single-site and pairwise amino acid frequencies
recently attracts a great deal of attention in the studies of protein structure
and evolution. We study regularization and learning methods and how to tune
regularization parameters to correctly infer interactions in Boltzmann machine
learning. Using $L_2$ regularization for fields, group $L_1$ for couplings is
shown to be very effective for sparse couplings in comparison with $L_2$ and
$L_1$. Two regularization parameters are tuned to yield equal values for both
the sample and ensemble averages of evolutionary energy. Both averages smoothly
change and converge, but their learning profiles are very different between
learning methods. The Adam method is modified to make stepsize proportional to
the gradient for sparse couplings and to use a soft-thresholding function for
group $L_1$. It is shown by first inferring interactions from protein sequences
and then from Monte Carlo samples that the fields and couplings can be well
recovered, but that recovering the pairwise correlations in the resolution of a
total energy is harder for the natural proteins than for the protein-like
sequences. Selective temperature for folding/structural constrains in protein
evolution is also estimated.

    

### [[1910.08701] Robust Distributed Accelerated Stochastic Gradient Methods for Multi-Agent Networks](http://arxiv.org/abs/1910.08701)


  We study distributed stochastic gradient (D-SG) method and its accelerated
variant (D-ASG) for solving decentralized strongly convex stochastic
optimization problems where the objective function is distributed over several
computational units, lying on a fixed but arbitrary connected communication
graph, subject to local communication constraints where noisy estimates of the
gradients are available. We develop a framework which allows to choose the
stepsize and the momentum parameters of these algorithms in a way to optimize
performance by systematically trading off the bias, variance, robustness to
gradient noise and dependence to network effects. When gradients do not contain
noise, we also prove that distributed accelerated methods can \emph{achieve
acceleration}, requiring $\mathcal{O}(\kappa \log(1/\varepsilon))$ gradient
evaluations and $\mathcal{O}(\kappa \log(1/\varepsilon))$ communications to
converge to the same fixed point with the non-accelerated variant where
$\kappa$ is the condition number and $\varepsilon$ is the target accuracy. To
our knowledge, this is the first acceleration result where the iteration
complexity scales with the square root of the condition number in the context
of \emph{primal} distributed inexact first-order methods. For quadratic
functions, we also provide finer performance bounds that are tight with respect
to bias and variance terms. Finally, we study a multistage version of D-ASG
with parameters carefully varied over stages to ensure exact
$\mathcal{O}(-k/\sqrt{\kappa})$ linear decay in the bias term as well as
optimal $\mathcal{O}(\sigma^2/k)$ in the variance term. We illustrate through
numerical experiments that our approach results in practical algorithms that
are robust to gradient noise and that can outperform existing methods.

    

### [[1911.04278] Adversarial Attacks on Time-Series Intrusion Detection for Industrial Control Systems](http://arxiv.org/abs/1911.04278)


  Neural networks are increasingly used for intrusion detection on industrial
control systems (ICS). With neural networks being vulnerable to adversarial
examples, attackers who wish to cause damage to an ICS can attempt to hide
their attacks from detection by using adversarial example techniques. In this
work we address the domain specific challenges of constructing such attacks
against autoregressive based intrusion detection systems (IDS) in an ICS
setting.
We model an attacker that can compromise a subset of sensors in a ICS which
has a LSTM based IDS. The attacker manipulates the data sent to the IDS, and
seeks to hide the presence of real cyber-physical attacks occurring in the ICS.
We evaluate our adversarial attack methodology on the Secure Water Treatment
system when examining solely continuous data, and on data containing a mixture
of discrete and continuous variables. In the continuous data domain our attack
successfully hides the cyber-physical attacks requiring 2.87 out of 12
monitored sensors to be compromised on average. With both discrete and
continuous data our attack required, on average, 3.74 out of 26 monitored
sensors to be compromised.

    

### [[2001.09328] On the Fairness of Randomized Trials for Recommendation with Heterogeneous Demographics and Beyond](http://arxiv.org/abs/2001.09328)


  Observed events in recommendation are consequence of the decisions made by a
policy, thus they are usually selectively labeled, namely the data are Missing
Not At Random (MNAR), which often causes large bias to the estimate of true
outcomes risk. A general approach to correct MNAR bias is performing small
Randomized Controlled Trials (RCTs), where an additional uniform policy is
employed to randomly assign items to each user. In this work, we concentrate on
the fairness of RCTs under both homogeneous and heterogeneous demographics,
especially analyzing the bias for the least favorable group on the latter
setting. Considering RCTs' limitations, we propose a novel Counterfactual
Robust Risk Minimization (CRRM) framework, which is totally free of expensive
RCTs, and derive its theoretical generalization error bound. At last, empirical
experiments are performed on synthetic tasks and real-world data sets,
substantiating our method's superiority both in fairness and generalization.

    

### [[2002.02826] Conditional Deep Gaussian Processes: multi-fidelity kernel learning](http://arxiv.org/abs/2002.02826)


  Deep Gaussian Processes (DGPs) were proposed as an expressive Bayesian model
capable of a mathematically grounded estimation of uncertainty. The
expressivity of DPGs results from not only the compositional character but the
distribution propagation within the hierarchy. Recently, [1] pointed out that
the hierarchical structure of DGP well suited modeling the multi-fidelity
regression, in which one is provided sparse observations with high precision
and plenty of low fidelity observations. We propose the conditional DGP model
in which the latent GPs are directly supported by the fixed lower fidelity
data. Then the moment matching method in [2] is applied to approximate the
marginal prior of conditional DGP with a GP. The obtained effective kernels are
implicit functions of the lower-fidelity data, manifesting the expressivity
contributed by distribution propagation within the hierarchy. The
hyperparameters are learned via optimizing the approximate marginal likelihood.
Experiments with synthetic and high dimensional data show comparable
performance against other multi-fidelity regression methods, variational
inference, and multi-output GP. We conclude that, with the low fidelity data
and the hierarchical DGP structure, the effective kernel encodes the inductive
bias for true function allowing the compositional freedom discussed in [3,4].

    

### [[2002.10645] Multivariate time-series modeling with generative neural networks](http://arxiv.org/abs/2002.10645)


  Generative moment matching networks (GMMNs) are introduced as dependence
models for the joint innovation distribution of multivariate time series (MTS).
Following the popular copula-GARCH approach for modeling dependent MTS data, a
framework based on a GMMN-GARCH approach is presented. First, ARMA-GARCH models
are utilized to capture the serial dependence within each univariate marginal
time series. Second, if the number of marginal time series is large, principal
component analysis (PCA) is used as a dimension-reduction step. Last, the
remaining cross-sectional dependence is modeled via a GMMN, the main
contribution of this work. GMMNs are highly flexible and easy to simulate from,
which is a major advantage over the copula-GARCH approach. Applications
involving yield curve modeling and the analysis of foreign exchange-rate
returns demonstrate the utility of the GMMN-GARCH approach, especially in terms
of producing better empirical predictive distributions and making better
probabilistic forecasts.

    

### [[2003.06511] Optimal Change-Point Detection with Training Sequences in the Large and Moderate Deviations Regimes](http://arxiv.org/abs/2003.06511)


  This paper investigates a novel offline change-point detection problem from
an information-theoretic perspective. In contrast to most related works, we
assume that the knowledge of the underlying pre- and post-change distributions
are not known and can only be learned from the training sequences which are
available. We further require the probability of the \emph{estimation error} to
decay either exponentially or sub-exponentially fast (corresponding
respectively to the large and moderate deviations regimes in information theory
parlance). Based on the training sequences as well as the test sequence
consisting of a single change-point, we design a change-point estimator and
further show that this estimator is optimal by establishing matching (strong)
converses. This leads to a full characterization of the optimal confidence
width (i.e., half the width of the confidence interval within which the true
change-point is located at with high probability) as a function of the
undetected error, under both the large and moderate deviations regimes.

    

### [[2005.09110] Two-View Fine-grained Classification of Plant Species](http://arxiv.org/abs/2005.09110)


  Automatic plant classification is a challenging problem due to the wide
biodiversity of the existing plant species in a fine-grained scenario. Powerful
deep learning architectures have been used to improve the classification
performance in such a fine-grained problem, but usually building models that
are highly dependent on a large training dataset and which are not scalable. In
this paper, we propose a novel method based on a two-view leaf image
representation and a hierarchical classification strategy for fine-grained
recognition of plant species. It uses the botanical taxonomy as a basis for a
coarse-to-fine strategy applied to identify the plant genus and species. The
two-view representation provides complementary global and local features of
leaf images. A deep metric based on Siamese convolutional neural networks is
used to reduce the dependence on a large number of training samples and make
the method scalable to new plant species. The experimental results on two
challenging fine-grained datasets of leaf images (i.e. LifeCLEF 2015 and
LeafSnap) have shown the effectiveness of the proposed method, which achieved
recognition accuracy of 0.87 and 0.96 respectively.

    

### [[2006.03654] DeBERTa: Decoding-enhanced BERT with Disentangled Attention](http://arxiv.org/abs/2006.03654)


  Recent progress in pre-trained neural language models has significantly
improved the performance of many natural language processing (NLP) tasks. In
this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT
with disentangled attention) that improves the BERT and RoBERTa models using
two novel techniques. The first is the disentangled attention mechanism, where
each word is represented using two vectors that encode its content and
position, respectively, and the attention weights among words are computed
using disentangled matrices on their contents and relative positions,
respectively. Second, an enhanced mask decoder is used to incorporate absolute
positions in the decoding layer to predict the masked tokens in model
pre-training. In addition, a new virtual adversarial training method is used
for fine-tuning to improve models' generalization. We show that these
techniques significantly improve the efficiency of model pre-training and the
performance of both natural language understanding (NLU) and natural langauge
generation (NLG) downstream tasks. Compared to RoBERTa-Large, a DeBERTa model
trained on half of the training data performs consistently better on a wide
range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%),
on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%).
Notably, we scale up DeBERTa by training a larger version that consists of 48
Transform layers with 1.5 billion parameters. The significant performance boost
makes the single DeBERTa model surpass the human performance on the SuperGLUE
benchmark (Wang et al., 2019a) for the first time in terms of macro-average
score (89.9 versus 89.8), and the ensemble DeBERTa model sits atop the
SuperGLUE leaderboard as of January 6, 2021, out performing the human baseline
by a decent margin (90.3 versus 89.8).

    

### [[2006.07988] Adaptive Universal Generalized PageRank Graph Neural Network](http://arxiv.org/abs/2006.07988)


  In many important graph data processing applications the acquired information
includes both node features and observations of the graph topology. Graph
neural networks (GNNs) are designed to exploit both sources of evidence but
they do not optimally trade-off their utility and integrate them in a manner
that is also universal. Here, universality refers to independence on homophily
or heterophily graph assumptions. We address these issues by introducing a new
Generalized PageRank (GPR) GNN architecture that adaptively learns the GPR
weights so as to jointly optimize node feature and topological information
extraction, regardless of the extent to which the node labels are homophilic or
heterophilic. Learned GPR weights automatically adjust to the node label
pattern, irrelevant on the type of initialization, and thereby guarantee
excellent learning performance for label patterns that are usually hard to
handle. Furthermore, they allow one to avoid feature over-smoothing, a process
which renders feature information nondiscriminative, without requiring the
network to be shallow. Our accompanying theoretical analysis of the GPR-GNN
method is facilitated by novel synthetic benchmark datasets generated by the
so-called contextual stochastic block model. We also compare the performance of
our GNN architecture with that of several state-of-the-art GNNs on the problem
of node-classification, using well-known benchmark homophilic and heterophilic
datasets. The results demonstrate that GPR-GNN offers significant performance
improvement compared to existing techniques on both synthetic and benchmark
data.

    

### [[2007.02203] Accuracy-Efficiency Trade-Offs and Accountability in Distributed ML Systems](http://arxiv.org/abs/2007.02203)


  Trade-offs between accuracy and efficiency pervade law, public health, and
other non-computing domains, which have developed policies to guide how to
balance the two in conditions of uncertainty. While computer science also
commonly studies accuracy-efficiency trade-offs, their policy implications
remain poorly examined. Drawing on risk assessment practices in the US, we
argue that, since examining these trade-offs has been useful for guiding
governance in other domains, we need to similarly reckon with these trade-offs
in governing computer systems. We focus our analysis on distributed machine
learning systems. Understanding the policy implications in this area is
particularly urgent because such systems, which include autonomous vehicles,
tend to be high-stakes and safety-critical. We 1) describe how the trade-off
takes shape for these systems, 2) highlight gaps between existing US risk
assessment standards and what these systems require to be properly assessed,
and 3) make specific calls to action to facilitate accountability when
hypothetical risks concerning the accuracy-efficiency trade-off become realized
as accidents in the real world. We close by discussing how such accountability
mechanisms encourage more just, transparent governance aligned with public
values.

    

### [[2008.01062] QPLEX: Duplex Dueling Multi-Agent Q-Learning](http://arxiv.org/abs/2008.01062)


  We explore value-based multi-agent reinforcement learning (MARL) in the
popular paradigm of centralized training with decentralized execution (CTDE).
CTDE has an important concept, Individual-Global-Max (IGM) principle, which
requires the consistency between joint and local action selections to support
efficient local decision-making. However, in order to achieve scalability,
existing MARL methods either limit representation expressiveness of their value
function classes or relax the IGM consistency, which may suffer from
instability risk or may not perform well in complex domains. This paper
presents a novel MARL approach, called duPLEX dueling multi-agent Q-learning
(QPLEX), which takes a duplex dueling network architecture to factorize the
joint value function. This duplex dueling structure encodes the IGM principle
into the neural network architecture and thus enables efficient value function
learning. Theoretical analysis shows that QPLEX achieves a complete IGM
function class. Empirical experiments on StarCraft II micromanagement tasks
demonstrate that QPLEX significantly outperforms state-of-the-art baselines in
both online and offline data collection settings, and also reveal that QPLEX
achieves high sample efficiency and can benefit from offline datasets without
additional online exploration.

    

### [[2008.13607] Ranking Policy Decisions](http://arxiv.org/abs/2008.13607)


  Policies trained via Reinforcement Learning (RL) are often needlessly
complex, making them difficult to analyse and interpret. In a run with $n$ time
steps, a policy will make $n$ decisions on actions to take; we conjecture that
only a small subset of these decisions delivers value over selecting a simple
default action. Given a trained policy, we propose a novel black-box method
based on statistical fault localisation that ranks the states of the
environment according to the importance of decisions made in those states. We
argue that among other things, the ranked list of states can help explain and
understand the policy. As the ranking method is statistical, a direct
evaluation of its quality is hard. As a proxy for quality, we use the ranking
to create new, simpler policies from the original ones by pruning decisions
identified as unimportant (that is, replacing them by default actions) and
measuring the impact on performance. Our experiments on a diverse set of
standard benchmarks demonstrate that pruned policies can perform on a level
comparable to the original policies. Conversely, we show that naive approaches
for ranking policy decisions, e.g., ranking based on the frequency of visiting
a state, do not result in high-performing pruned policies.

    

### [[2009.13716] Grow-Push-Prune: aligning deep discriminants for effective structural network compression](http://arxiv.org/abs/2009.13716)


  Most of today's popular deep architectures are hand-engineered to be
generalists. However, this design procedure usually leads to massive redundant,
useless, or even harmful features for specific tasks. Unnecessarily high
complexities render deep nets impractical for many real-world applications,
especially those without powerful GPU support. In this paper, we attempt to
derive task-dependent compact models from a deep discriminant analysis
perspective. We propose an iterative and proactive approach for classification
tasks which alternates between (1) a pushing step, with an objective to
simultaneously maximize class separation, penalize co-variances, and push deep
discriminants into alignment with a compact set of neurons, and (2) a pruning
step, which discards less useful or even interfering neurons. Deconvolution is
adopted to reverse 'unimportant' filters' effects and recover useful
contributing sources. A simple network growing strategy based on the basic
Inception module is proposed for challenging tasks requiring larger capacity
than what the base net can offer. Experiments on the MNIST, CIFAR10, and
ImageNet datasets demonstrate our approach's efficacy. On ImageNet, by pushing
and pruning our grown Inception-88 model, we achieve more accurate models than
Inception nets generated during growing, residual nets, and popular compact
nets at similar sizes. We also show that our grown Inception nets (without
hard-coded dimension alignment) clearly outperform residual nets of similar
complexities.

    

### [[2010.03655] Reinforcement Learning for Many-Body Ground-State Preparation Inspired by Counterdiabatic Driving](http://arxiv.org/abs/2010.03655)


  The quantum alternating operator ansatz (QAOA) is a prominent example of
variational quantum algorithms. We propose a generalized QAOA called CD-QAOA,
which is inspired by the counterdiabatic driving procedure, designed for
quantum many-body systems and optimized using a reinforcement learning (RL)
approach. The resulting hybrid control algorithm proves versatile in preparing
the ground state of quantum-chaotic many-body spin chains by minimizing the
energy. We show that using terms occurring in the adiabatic gauge potential as
generators of additional control unitaries, it is possible to achieve fast
high-fidelity many-body control away from the adiabatic regime. While each
unitary retains the conventional QAOA-intrinsic continuous control degree of
freedom such as the time duration, we consider the order of the multiple
available unitaries appearing in the control sequence as an additional discrete
optimization problem. Endowing the policy gradient algorithm with an
autoregressive deep learning architecture to capture causality, we train the RL
agent to construct optimal sequences of unitaries. The algorithm has no access
to the quantum state, and we find that the protocol learned on small systems
may generalize to larger systems. By scanning a range of protocol durations, we
present numerical evidence for a finite quantum speed limit in the
nonintegrable mixed-field spin-1/2 Ising and Lipkin-Meshkov-Glick models, and
for the suitability to prepare ground states of the spin-1 Heisenberg chain in
the long-range and topologically ordered parameter regimes. This work paves the
way to incorporate recent success from deep learning for the purpose of quantum
many-body control.

    

### [[2010.06647] Video Action Understanding](http://arxiv.org/abs/2010.06647)


  Many believe that the successes of deep learning on image understanding
problems can be replicated in the realm of video understanding. However, due to
the scale and temporal nature of video, the span of video understanding
problems and the set of proposed deep learning solutions is arguably wider and
more diverse than those of their 2D image siblings. Finding, identifying, and
predicting actions are a few of the most salient tasks in this emerging and
rapidly evolving field. With a pedagogical emphasis, this tutorial introduces
and systematizes fundamental topics, basic concepts, and notable examples in
supervised video action understanding. Specifically, we clarify a taxonomy of
action problems, catalog and highlight video datasets, describe common video
data preparation methods, present the building blocks of state-of-the art deep
learning model architectures, and formalize domain-specific metrics to baseline
proposed solutions. This tutorial is intended to be accessible to a general
computer science audience and assumes a conceptual understanding of supervised
learning.

    

### [[2010.08689] Towards Compact Neural Networks via End-to-End Training: A Bayesian Tensor Approach with Automatic Rank Determination](http://arxiv.org/abs/2010.08689)


  While post-training model compression can greatly reduce the inference cost
of a deep neural network, uncompressed training still consumes a huge amount of
hardware resources, run-time and energy. It is highly desirable to directly
train a compact neural network from scratch with low memory and low
computational cost. Low-rank tensor decomposition is one of the most effective
approaches to reduce the memory and computing requirements of large-size neural
networks. However, directly training a low-rank tensorized neural network is a
very challenging task because it is hard to determine a proper tensor rank {\it
a priori}, which controls the model complexity and compression ratio in the
training process. This paper presents a novel end-to-end framework for low-rank
tensorized training of neural networks. We first develop a flexible Bayesian
model that can handle various low-rank tensor formats (e.g., CP, Tucker, tensor
train and tensor-train matrix) that compress neural network parameters in
training. This model can automatically determine the tensor ranks inside a
nonlinear forward model, which is beyond the capability of existing Bayesian
tensor methods. We further develop a scalable stochastic variational inference
solver to estimate the posterior density of large-scale problems in training.
Our work provides the first general-purpose rank-adaptive framework for
end-to-end tensorized training. Our numerical results on various neural network
architectures show orders-of-magnitude parameter reduction and little accuracy
loss (or even better accuracy) in the training process. Specifically, on a very
large deep learning recommendation system with over $4.2\times 10^9$ model
parameters, our method can reduce the variables to only $1.6\times 10^5$
automatically in the training process (i.e., by $2.6\times 10^4$ times) while
achieving almost the same accuracy.

    

### [[2010.15673] Interpretable Data-Driven Demand Modelling for On-Demand Transit Services](http://arxiv.org/abs/2010.15673)


  In recent years, with the advancements in information and communication
technology, different emerging on-demand shared mobility services have been
introduced as innovative solutions in the low-density areas, including
on-demand transit (ODT), mobility on-demand (MOD) transit, and crowdsourced
mobility services. However, due to their infancy, there is a strong need to
understand and model the demand for these services. In this study, we developed
trip production and distribution models for ODT services at Dissemination areas
(DA) level using four machine learning algorithms: Random Forest (RF), Bagging,
Artificial Neural Network (ANN) and Deep Neural Network (DNN). The data used in
the modelling process were acquired from Belleville's ODT operational data and
2016 census data. Bayesian optimalization approach was used to find the optimal
architecture of the adopted algorithms. Moreover, post-hoc model was employed
to interpret the predictions and examine the importance of the explanatory
variables. The results showed that the land-use type was the most important
variable in the trip production model. On the other hand, the demographic
characteristics of the trip destination were the most important variables in
the trip distribution model. Moreover, the results revealed that higher trip
distribution levels are expected between dissemination areas with
commercial/industrial land-use type and dissemination areas with high-density
residential land-use. Our findings suggest that the performance of ODT services
can be further enhanced by (a) locating idle vehicles in the neighbourhoods
with commercial/industrial land-use and (b) using the spatio-temporal demand
models obtained in this work to continuously update the operating fleet size.

    

### [[2010.16132] Multiview Variational Graph Autoencoders for Canonical Correlation Analysis](http://arxiv.org/abs/2010.16132)


  We present a novel multiview canonical correlation analysis model based on a
variational approach. This is the first nonlinear model that takes into account
the available graph-based geometric constraints while being scalable for
processing large scale datasets with multiple views. It is based on an
autoencoder architecture with graph convolutional neural network layers. We
experiment with our approach on classification, clustering, and recommendation
tasks on real datasets. The algorithm is competitive with state-of-the-art
multiview representation learning techniques.

    

### [[2011.01457] Blockchain based Attack Detection on Machine Learning Algorithms for IoT based E-Health Applications](http://arxiv.org/abs/2011.01457)


  The application of machine learning (ML) algorithms are massively scaling-up
due to rapid digitization and emergence of new tecnologies like Internet of
Things (IoT). In today's digital era, we can find ML algorithms being applied
in the areas of healthcare, IoT, engineering, finance and so on. However, all
these algorithms need to be trained in order to predict/solve a particular
problem. There is high possibility of tampering the training datasets and
produce biased results. Hence, in this article, we have proposed blockchain
based solution to secure the datasets generated from IoT devices for E-Health
applications. The proposed blockchain based solution uses using private cloud
to tackle the aforementioned issue. For evaluation, we have developed a system
that can be used by dataset owners to secure their data.

    

### [[2011.07142] Sparse Representations of Positive Functions via First and Second-Order Pseudo-Mirror Descent](http://arxiv.org/abs/2011.07142)


  We consider expected risk minimization when the range of the estimator is
required to be nonnegative, motivated by the settings of maximum likelihood
estimation (MLE) and trajectory optimization. To facilitate nonlinear
interpolation, we hypothesize that search is conducted over a Reproducing
Kernel Hilbert Space (RKHS). To solve it, we develop first and second-order
variants of stochastic mirror descent employing (i) pseudo-gradients and (ii)
complexity-reducing projections. Compressive projection in first-order scheme
is executed via kernel orthogonal matching pursuit (KOMP), and overcome the
fact that the vanilla RKHS parameterization grows unbounded with time.
Moreover, pseudo-gradients are needed when stochastic estimates of the gradient
of the expected cost are only computable up to some numerical errors, which
arise in, e.g., integral approximations. The second-order scheme develops a
Hessian inverse approximation via recursively averaged pseudo-gradient outer
products. For the first-order scheme, we establish tradeoffs between accuracy
of convergence in mean and the projection budget parameter under constant
step-size and compression budget are established, as well as non-asymptotic
bounds on the model complexity. Analogous convergence results are established
for the second-order scheme under an additional eigenvalue decay condition on
the Hessian of the optimal RKHS element. Experiments demonstrate favorable
performance on inhomogeneous Poisson Process intensity estimation in practice.

    

### [[2011.12468] Nudge: Accelerating Overdue Pull Requests Towards Completion](http://arxiv.org/abs/2011.12468)


  Pull requests are a key part of the collaborative software development and
code review process today. However, pull requests can also slow down the
software development process when the reviewer(s) or the author do not actively
engage with the pull request. In this work, we design an end-to-end service,
Nudge, for accelerating overdue pull requests towards completion by reminding
the author or the reviewer(s) to engage with their overdue pull requests.
First, we use models based on effort estimation and machine learning to predict
the completion time for a given pull request. Second, we use activity detection
to reduce false positives. Lastly, we use dependency determination to
understand the blocker of the pull request and nudge the appropriate
actor(author or reviewer(s)). We also do a correlation analysis to understand
the statistical relationship between the pull request completion times and
various pull request and developer related attributes. Nudge has been deployed
on 147 repositories at Microsoft since 2019. We do a large scale evaluation
based on the implicit and explicit feedback we received from sending the Nudge
notifications on 8,500 pull requests. We observe significant reduction in
completion time, by over 60%, for pull requests which were nudged thus
increasing the efficiency of the code review process and accelerating the pull
request progression.

    

### [[2011.14230] CROCS: Clustering and Retrieval of Cardiac Signals Based on Patient Disease Class, Sex, and Age](http://arxiv.org/abs/2011.14230)


  The process of manually searching for relevant instances in, and extracting
information from, clinical databases underpin a multitude of clinical tasks.
Such tasks include disease diagnosis, clinical trial recruitment, and
continuing medical education. This manual search-and-extract process, however,
has been hampered by the growth of large-scale clinical databases and the
increased prevalence of unlabelled instances. To address this challenge, we
propose a supervised contrastive learning framework, CROCS, where
representations of cardiac signals associated with a set of patient-specific
attributes (e.g., disease class, sex, age) are attracted to learnable
embeddings entitled clinical prototypes. We exploit such prototypes for both
the clustering and retrieval of unlabelled cardiac signals based on multiple
patient attributes. We show that CROCS outperforms the state-of-the-art method,
DTC, when clustering and also retrieves relevant cardiac signals from a large
database. We also show that clinical prototypes adopt a semantically meaningful
arrangement based on patient attributes and thus confer a high degree of
interpretability.

    

### [[2012.03214] TornadoAggregate: Accurate and Scalable Federated Learning via the Ring-Based Architecture](http://arxiv.org/abs/2012.03214)


  Federated learning has emerged as a new paradigm of collaborative machine
learning; however, many prior studies have used global aggregation along a star
topology without much consideration of the communication scalability or the
diurnal property relied on clients' local time variety. In contrast, ring
architecture can resolve the scalability issue and even satisfy the diurnal
property by iterating nodes without an aggregation. Nevertheless, such
ring-based algorithms can inherently suffer from the high-variance problem. To
this end, we propose a novel algorithm called TornadoAggregate that improves
both accuracy and scalability by facilitating the ring architecture. In
particular, to improve the accuracy, we reformulate the loss minimization into
a variance reduction problem and establish three principles to reduce variance:
Ring-Aware Grouping, Small Ring, and Ring Chaining. Experimental results show
that TornadoAggregate improved the test accuracy by up to 26.7% and achieved
near-linear scalability.

    

### [[2012.09265] Variational Quantum Algorithms](http://arxiv.org/abs/2012.09265)


  Applications such as simulating complicated quantum systems or solving
large-scale linear algebra problems are very challenging for classical
computers due to the extremely high computational cost. Quantum computers
promise a solution, although fault-tolerant quantum computers will likely not
be available in the near future. Current quantum devices have serious
constraints, including limited numbers of qubits and noise processes that limit
circuit depth. Variational Quantum Algorithms (VQAs), which use a classical
optimizer to train a parametrized quantum circuit, have emerged as a leading
strategy to address these constraints. VQAs have now been proposed for
essentially all applications that researchers have envisioned for quantum
computers, and they appear to the best hope for obtaining quantum advantage.
Nevertheless, challenges remain including the trainability, accuracy, and
efficiency of VQAs. Here we overview the field of VQAs, discuss strategies to
overcome their challenges, and highlight the exciting prospects for using them
to obtain quantum advantage.

    

### [[2012.11926] Few-Shot Text Generation with Pattern-Exploiting Training](http://arxiv.org/abs/2012.11926)


  Providing pretrained language models with simple task descriptions in natural
language enables them to solve some tasks in a fully unsupervised fashion.
Moreover, when combined with regular learning from examples, this idea yields
impressive few-shot results for a wide range of text classification tasks. It
is also a promising direction to improve data efficiency in generative
settings, but there are several challenges to using a combination of task
descriptions and example-based learning for text generation. In particular, it
is crucial to find task descriptions that are easy to understand for the
pretrained model and to ensure that it actually makes good use of them;
furthermore, effective measures against overfitting have to be implemented. In
this paper, we show how these challenges can be tackled: We introduce GenPET, a
method for text generation that is based on pattern-exploiting training, a
recent approach for combining textual instructions with supervised learning
that only works for classification tasks. On several summarization and headline
generation datasets, GenPET gives consistent improvements over strong baselines
in few-shot settings.

    

### [[2012.14738] With False Friends Like These, Who Can Notice Mistakes?](http://arxiv.org/abs/2012.14738)


  Adversarial examples crafted by an explicit adversary have attracted
significant attention in machine learning. However, the security risk posed by
a potential false friend has been largely overlooked. In this paper, we unveil
the threat of hypocritical examples -- inputs that are originally misclassified
yet perturbed by a false friend to force correct predictions. While such
perturbed examples seem harmless, we point out for the first time that they
could be maliciously used to conceal the mistakes of a substandard (i.e., not
as good as required) model during an evaluation. Once a deployer trusts the
hypocritical performance and applies the "well-performed" model in real-world
applications, unexpected failures may happen even in benign environments. More
seriously, this security risk seems to be pervasive: we find that many types of
substandard models are vulnerable to hypocritical examples across multiple
datasets. Furthermore, we provide the first attempt to characterize the threat
with a metric called hypocritical risk and try to circumvent it via several
countermeasures. Results demonstrate the effectiveness of the countermeasures,
while the risk remains non-negligible even after adaptive robust training.

    

### [[2101.01835] Risk markers by sex for in-hospital mortality in patients with acute coronary syndrome based on machine learning](http://arxiv.org/abs/2101.01835)


  Background: Several studies have highlighted the importance of considering
sex differences in the diagnosis and treatment of Acute Coronary Syndrome
(ACS). However, the identification of sex-specific risk markers in ACS
sub-populations has been scarcely studied. The goal of this paper is to
identify in-hospital mortality markers for women and men in ACS sub-populations
from a public database of electronic health records (EHR) using machine
learning methods. Methods: From the MIMIC-III database, we extracted 1,299
patients with ST-elevation myocardial infarction and 2,820 patients with
Non-ST-elevation myocardial infarction. We trained and validated mortality
prediction models and used an interpretability technique based on Shapley
values to identify sex-specific markers for each sub-population. Results: The
models based on eXtreme Gradient Boosting achieved the highest performance:
AUC=0.94 (95\% CI:0.84-0.96) for STEMI and AUC=0.94 (95\% CI:0.80-0.90) for
NSTEMI. For STEMI, the top markers in women are chronic kidney failure, high
heart rate, and age over 70 years, while for men are acute kidney failure, high
troponin T levels, and age over 75 years. In contrast, for NSTEMI, the top
markers in women are low troponin levels, high urea level, and age over 80
years, and for men are high heart rate and creatinine levels, and age over 70
years. Conclusions: Our results show that it is possible to find significant
and coherent sex-specific risk markers of different ACS sub-populations by
interpreting machine learning mortality models trained on EHRs. Differences are
observed in the identified risk markers between women and men, which highlight
the importance of considering sex-specific markers to have more appropriate
treatment strategies and better clinical outcomes.

    

### [[2101.08539] Orthogonal Least Squares Based Fast Feature Selection for Linear Classification](http://arxiv.org/abs/2101.08539)


  An Orthogonal Least Squares (OLS) based feature selection method is proposed
for both binomial and multinomial classification. The novel Squared Orthogonal
Correlation Coefficient (SOCC) is defined based on Error Reduction Ratio (ERR)
in OLS and used as the feature ranking criterion. The equivalence between the
canonical correlation coefficient, Fisher's criterion, and the sum of the SOCCs
is revealed, which unveils the statistical implication of ERR in OLS for the
first time. It is also shown that the OLS based feature selection method has
speed advantages when applied for greedy search. The proposed method is
comprehensively compared with the mutual information based feature selection
methods and the embedded methods using both synthetic and real world datasets.
The results show that the proposed method is always in the top 5 among the 12
candidate methods. Besides, the proposed method can be directly applied to
continuous features without discretisation, which is another significant
advantage over mutual information based methods.

    

### [[2102.05738] Refinement of polygonal grids using Convolutional Neural Networks with applications to polygonal Discontinuous Galerkin and Virtual Element methods](http://arxiv.org/abs/2102.05738)


  We propose new strategies to handle polygonal grids refinement based on
Convolutional Neural Networks (CNNs). We show that CNNs can be successfully
employed to identify correctly the "shape" of a polygonal element so as to
design suitable refinement criteria to be possibly employed within adaptive
refinement strategies. We propose two refinement strategies that exploit the
use of CNNs to classify elements' shape, at a low computational cost. We test
the proposed idea considering two families of finite element methods that
support arbitrarily shaped polygonal elements, namely Polygonal Discontinuous
Galerkin (PolyDG) methods and Virtual Element Methods (VEMs). We demonstrate
that the proposed algorithms can greatly improve the performance of the
discretization schemes both in terms of accuracy and quality of the underlying
grids. Moreover, since the training phase is performed off-line and is
independent of the differential model the overall computational costs are kept
low.

    

### [[2102.12668] Learning-based Robust Motion Planning with Guaranteed Stability: A Contraction Theory Approach](http://arxiv.org/abs/2102.12668)


  This paper presents Learning-based Autonomous Guidance with RObustness and
Stability guarantees (LAG-ROS), which provides machine learning-based nonlinear
motion planners with formal robustness and stability guarantees, by designing a
differential Lyapunov function using contraction theory. LAG-ROS utilizes a
neural network to model a robust tracking controller independently of a target
trajectory, for which we show that the Euclidean distance between the target
and controlled trajectories is exponentially bounded linearly in the learning
error, even under the existence of bounded external disturbances. We also
present a convex optimization approach that minimizes the steady-state bound of
the tracking error to construct the robust control law for neural network
training. In numerical simulations, it is demonstrated that the proposed method
indeed possesses superior properties of robustness and nonlinear stability
resulting from contraction theory, whilst retaining the computational
efficiency of existing learning-based motion planners.

    

### [[2103.01638] Learning disentangled representations via product manifold projection](http://arxiv.org/abs/2103.01638)


  We propose a novel approach to disentangle the generative factors of
variation underlying a given set of observations. Our method builds upon the
idea that the (unknown) low-dimensional manifold underlying the data space can
be explicitly modeled as a product of submanifolds. This definition of
disentanglement gives rise to a novel weakly-supervised algorithm for
recovering the unknown explanatory factors behind the data. At training time,
our algorithm only requires pairs of non i.i.d. data samples whose elements
share at least one, possibly multidimensional, generative factor of variation.
We require no knowledge on the nature of these transformations, and do not make
any limiting assumption on the properties of each subspace. Our approach is
easy to implement, and can be successfully applied to different kinds of data
(from images to 3D surfaces) undergoing arbitrary transformations. In addition
to standard synthetic benchmarks, we showcase our method in challenging
real-world applications, where we compare favorably with the state of the art.

    

### [[2103.02987] Learning-based Adaptive Control using Contraction Theory](http://arxiv.org/abs/2103.02987)


  Adaptive control is subject to stability and performance issues when a
learned model is used to enhance its performance. This paper thus presents a
deep learning-based adaptive control framework for nonlinear systems with
multiplicatively-separable parametrization, called adaptive Neural Contraction
Metric (aNCM). The aNCM approximates real-time optimization for computing a
differential Lyapunov function and a corresponding stabilizing adaptive control
law by using a Deep Neural Network (DNN). The use of DNNs permits real-time
implementation of the control law and broad applicability to a variety of
nonlinear systems with parametric and nonparametric uncertainties. We show
using contraction theory that the aNCM ensures exponential boundedness of the
distance between the target and controlled trajectories in the presence of
parametric uncertainties of the model, learning errors caused by aNCM
approximation, and external disturbances. Its superiority to the existing
robust and adaptive control methods is demonstrated using a cart-pole balancing
model.

    

### [[2103.09396] Pros and Cons of GAN Evaluation Measures: New Developments](http://arxiv.org/abs/2103.09396)


  This work is an update of a previous paper on the same topic published a few
years ago. With the dramatic progress in generative modeling, a suite of new
quantitative and qualitative techniques to evaluate models has emerged.
Although some measures such as Inception Score, Frechet Inception Distance,
Precision-Recall, and Perceptual Path Length are relatively more popular, GAN
evaluation is not a settled issue and there is still room for improvement.
Here, I describe new dimensions that are becoming important in assessing models
(e.g. bias and fairness) and discuss the connection between GAN evaluation and
deepfakes. These are important areas of concern in the machine learning
community today and progress in GAN evaluation can help mitigate them.

    

### [[2103.12158] Convergence of Finite Memory Q-Learning for POMDPs and Near Optimality of Learned Policies under Filter Stability](http://arxiv.org/abs/2103.12158)


  In this paper, for POMDPs, we provide the convergence of a Q learning
algorithm for control policies using a finite history of past observations and
control actions, and, consequentially, we establish near optimality of such
limit Q functions under explicit filter stability conditions. We present
explicit error bounds relating the approximation error to the length of the
finite history window. We establish the convergence of such Q-learning
iterations under mild ergodicity assumptions on the state process during the
exploration phase. We further show that the limit fixed point equation gives an
optimal solution for an approximate belief-MDP. We then provide bounds on the
performance of the policy obtained using the limit Q values compared to the
performance of the optimal policy for the POMDP, where we also present explicit
conditions using recent results on filter stability in controlled POMDPs. While
there exist many experimental results, (i) the rigorous asymptotic convergence
(to an approximate MDP value function) for such finite-memory Q-learning
algorithms, and (ii) the near optimality with an explicit rate of convergence
(in the memory size) are results that are new to the literature, to our
knowledge.

    

### [[2103.13466] Asymptotic Freeness of Layerwise Jacobians Caused by Invariance of Multilayer Perceptron: The Haar Orthogonal Case](http://arxiv.org/abs/2103.13466)


  Free Probability Theory (FPT) provides rich knowledge for handling
mathematical difficulties caused by random matrices that appear in research
related to deep neural networks (DNNs), such as the dynamical isometry, Fisher
information matrix, and training dynamics. FPT suits these researches because
the DNN's parameter-Jacobian and input-Jacobian are polynomials of layerwise
Jacobians. However, the critical assumption of asymptotic freenss of the
layerwise Jacobian has not been proven completely so far. The asymptotic
freeness assumption plays a fundamental role when propagating spectral
distributions through the layers. Haar distributed orthogonal matrices are
essential for achieving dynamical isometry. In this work, we prove asymptotic
freeness of layerwise Jacobians of multilayer perceptron (MLP) in this case. A
key of the proof is an invariance of the MLP. Considering the orthogonal
matrices that fix the hidden units in each layer, we replace each layer's
parameter matrix with itself multiplied by the orthogonal matrix, and then the
MLP does not change. Furthermore, if the original weights are Haar orthogonal,
the Jacobian is also unchanged by this replacement. Lastly, we can replace each
weight with a Haar orthogonal random matrix independent of the Jacobian of the
activation function using this key fact.

    

### [[2103.14430] Combining distribution-based neural networks to predict weather forecast probabilities](http://arxiv.org/abs/2103.14430)


  The success of deep learning techniques over the last decades has opened up a
new avenue of research for weather forecasting. Here, we take the novel
approach of using a neural network to predict full probability density
functions at each point in space and time rather than a single output value,
thus producing a probabilistic weather forecast. This enables the calculation
of both uncertainty and skill metrics for the neural network predictions, and
overcomes the common difficulty of inferring uncertainty from these
predictions.
This approach is data-driven and the neural network is trained on the
WeatherBench dataset (processed ERA5 data) to forecast geopotential and
temperature 3 and 5 days ahead. Data exploration leads to the identification of
the most important input variables, which are also found to agree with physical
reasoning, thereby validating our approach. In order to increase computational
efficiency further, each neural network is trained on a small subset of these
variables. The outputs are then combined through a stacked neural network, the
first time such a technique has been applied to weather data. Our approach is
found to be more accurate than some numerical weather prediction models and as
accurate as more complex alternative neural networks, with the added benefit of
providing key probabilistic information necessary for making informed weather
forecasts.

    

### [[2104.02705] deepregression: a Flexible Neural Network Framework for Semi-Structured Deep Distributional Regression](http://arxiv.org/abs/2104.02705)


  In this paper we describe the implementation of semi-structured deep
distributional regression, a flexible framework to learn conditional
distributions based on the combination of additive regression models and deep
networks. Our implementation encompasses (1) a modular neural network building
system based on the deep learning library TensorFlow for the fusion of various
statistical and deep learning approaches, (2) an orthogonalization cell to
allow for an interpretable combination of different subnetworks, as well as (3)
pre-processing steps necessary to set up such models. The software package
allows to define models in a user-friendly manner via a formula interface that
is inspired by classical statistical model frameworks such as mgcv. The
packages' modular design and functionality provides a unique resource for both
scalable estimation of complex statistical models and the combination of
approaches from deep learning and statistics. This allows for state-of-the-art
predictive performance while simultaneously retaining the indispensable
interpretability of classical statistical models.

    

### [[2104.07540] Generating Datasets with Pretrained Language Models](http://arxiv.org/abs/2104.07540)


  To obtain high-quality sentence embeddings from pretrained language models
(PLMs), they must either be augmented with additional pretraining objectives or
finetuned on a large set of labeled text pairs. While the latter approach
typically outperforms the former, it requires great human effort to generate
suitable datasets of sufficient size. In this paper, we show how PLMs can be
leveraged to obtain high-quality sentence embeddings without the need for
labeled data, finetuning or modifications to the pretraining objective: We
utilize the generative abilities of large and high-performing PLMs to generate
entire datasets of labeled text pairs from scratch, which we then use for
finetuning much smaller and more efficient models. Our fully unsupervised
approach outperforms strong baselines on several semantic textual similarity
datasets.

    

### [[2104.08815] FedNLP: Benchmarking Federated Learning Methods for Natural Language Processing Tasks](http://arxiv.org/abs/2104.08815)


  Increasing concerns and regulations about data privacy and sparsity
necessitate the study of privacy-preserving, decentralized learning methods for
natural language processing (NLP) tasks. Federated learning (FL) provides
promising approaches for a large number of clients (e.g., personal devices or
organizations) to collaboratively learn a shared global model to benefit all
clients while allowing users to keep their data locally. Despite interest in
studying FL methods for NLP tasks, a systematic comparison and analysis is
lacking in the literature. Herein, we present the FedNLP, a benchmarking
framework for evaluating federated learning methods on four different task
formulations: text classification, sequence tagging, question answering, and
seq2seq. We propose a universal interface between Transformer-based language
models (e.g., BERT, BART) and FL methods (e.g., FedAvg, FedOPT, etc.) under
various non-IID partitioning strategies. Our extensive experiments with FedNLP
provide empirical comparisons between FL methods and helps us better understand
the inherent challenges of this direction. The comprehensive analysis points to
intriguing and exciting future research aimed at developing FL methods for NLP
tasks.

    

### [[2104.12909] Algorithm is Experiment: Machine Learning, Market Design, and Policy Eligibility Rules](http://arxiv.org/abs/2104.12909)


  Algorithms produce a growing portion of decisions and recommendations both in
policy and business. Such algorithmic decisions are natural experiments
(conditionally quasi-randomly assigned instruments) since the algorithms make
decisions based only on observable input variables. We use this observation to
develop a treatment-effect estimator for a class of stochastic and
deterministic decision-making algorithms. Our estimator is shown to be
consistent and asymptotically normal for well-defined causal effects. A key
special case of our estimator is a multidimensional regression discontinuity
design. We apply our estimator to evaluate the effect of the Coronavirus Aid,
Relief, and Economic Security (CARES) Act, where hundreds of billions of
dollars worth of relief funding is allocated to hospitals via an algorithmic
rule. Our estimates suggest that the relief funding has little effect on
COVID-19-related hospital activity levels. Naive OLS and IV estimates exhibit
substantial selection bias.

    

### [[2105.06479] Advances in Machine and Deep Learning for Modeling and Real-time Detection of Multi-Messenger Sources](http://arxiv.org/abs/2105.06479)


  We live in momentous times. The science community is empowered with an
arsenal of cosmic messengers to study the Universe in unprecedented detail.
Gravitational waves, electromagnetic waves, neutrinos and cosmic rays cover a
wide range of wavelengths and time scales. Combining and processing these
datasets that vary in volume, speed and dimensionality requires new modes of
instrument coordination, funding and international collaboration with a
specialized human and technological infrastructure. In tandem with the advent
of large-scale scientific facilities, the last decade has experienced an
unprecedented transformation in computing and signal processing algorithms. The
combination of graphics processing units, deep learning, and the availability
of open source, high-quality datasets, have powered the rise of artificial
intelligence. This digital revolution now powers a multi-billion dollar
industry, with far-reaching implications in technology and society. In this
chapter we describe pioneering efforts to adapt artificial intelligence
algorithms to address computational grand challenges in Multi-Messenger
Astrophysics. We review the rapid evolution of these disruptive algorithms,
from the first class of algorithms introduced in early 2017, to the
sophisticated algorithms that now incorporate domain expertise in their
architectural design and optimization schemes. We discuss the importance of
scientific visualization and extreme-scale computing in reducing
time-to-insight and obtaining new knowledge from the interplay between models
and data.

    

### [[2106.00394] Improving Conditional Coverage via Orthogonal Quantile Regression](http://arxiv.org/abs/2106.00394)


  We develop a method to generate prediction intervals that have a
user-specified coverage level across all regions of feature-space, a property
called conditional coverage. A typical approach to this task is to estimate the
conditional quantiles with quantile regression -- it is well-known that this
leads to correct coverage in the large-sample limit, although it may not be
accurate in finite samples. We find in experiments that traditional quantile
regression can have poor conditional coverage. To remedy this, we modify the
loss function to promote independence between the size of the intervals and the
indicator of a miscoverage event. For the true conditional quantiles, these two
quantities are independent (orthogonal), so the modified loss function
continues to be valid. Moreover, we empirically show that the modified loss
function leads to improved conditional coverage, as evaluated by several
metrics. We also introduce two new metrics that check conditional coverage by
looking at the strength of the dependence between the interval size and the
indicator of miscoverage.

    

### [[2106.01908] You Never Cluster Alone](http://arxiv.org/abs/2106.01908)


  Recent advances in self-supervised learning with instance-level contrastive
objectives facilitate unsupervised clustering. However, a standalone datum is
not perceiving the context of the holistic cluster, and may undergo sub-optimal
assignment. In this paper, we extend the mainstream contrastive learning
paradigm to a cluster-level scheme, where all the data subjected to the same
cluster contribute to a unified representation that encodes the context of each
data group. Contrastive learning with this representation then rewards the
assignment of each datum. To implement this vision, we propose twin-contrast
clustering (TCC). We define a set of categorical variables as clustering
assignment confidence, which links the instance-level learning track with the
cluster-level one. On one hand, with the corresponding assignment variables
being the weight, a weighted aggregation along the data points implements the
set representation of a cluster. We further propose heuristic cluster
augmentation equivalents to enable cluster-level contrastive learning. On the
other hand, we derive the evidence lower-bound of the instance-level
contrastive objective with the assignments. By reparametrizing the assignment
variables, TCC is trained end-to-end, requiring no alternating steps. Extensive
experiments show that TCC outperforms the state-of-the-art on challenging
benchmarks.

    

### [[2106.02073] Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path](http://arxiv.org/abs/2106.02073)


  The recently discovered Neural Collapse (NC) phenomenon occurs pervasively in
today's deep net training paradigm of driving cross-entropy (CE) loss towards
zero. During NC, last-layer features collapse to their class-means, both
classifiers and class-means collapse to the same Simplex Equiangular Tight
Frame, and classifier behavior collapses to the nearest-class-mean decision
rule. Recent works demonstrated that deep nets trained with mean squared error
(MSE) loss perform comparably to those trained with CE. We empirically
establish that NC emerges in such MSE-trained deep nets as well through
experiments on three canonical networks and five benchmark datasets. We
provide, in a Google Colab notebook, PyTorch code for reproducing MSE-NC and
CE-NC:
this https URL.
The analytically-tractable MSE loss offers more mathematical opportunities than
the hard-to-analyze CE loss, inspiring us to leverage MSE loss towards the
theoretical investigation of NC. We develop three main contributions: (I) We
show a new decomposition of the MSE loss into (A) terms directly interpretable
through the lens of NC and which assume the last-layer classifier is exactly
the least-squares classifier; and (B) a term capturing the deviation from this
least-squares classifier. (II) We exhibit experiments on canonical datasets and
networks demonstrating that term-(B) is negligible during training. This
motivates us to introduce a new theoretical construct: the central path, where
the linear classifier stays MSE-optimal for feature activations throughout the
dynamics. (III) By studying renormalized gradient flow along the central path,
we derive exact dynamics that predict NC.

    

### [[2106.04627] Densely connected normalizing flows](http://arxiv.org/abs/2106.04627)


  Normalizing flows are bijective mappings between inputs and latent
representations with a fully factorized distribution. They are very attractive
due to exact likelihood evaluation and efficient sampling. However, their
effective capacity is often insufficient since the bijectivity constraint
limits the model width. We address this issue by incrementally padding
intermediate representations with noise. We precondition the noise in
accordance with previous invertible units, which we describe as cross-unit
coupling. Our invertible glow-like modules express intra-unit affine coupling
as a fusion of a densely connected block and Nyström self-attention. We refer
to our architecture as DenseFlow since both cross-unit and intra-unit couplings
rely on dense connectivity. Experiments show significant improvements due to
the proposed contributions, and reveal state-of-the-art density estimation
among all generative models under moderate computing budgets.

    

### [[2106.07108] Pointwise Feasibility of Gaussian Process-based Safety-Critical Control under Model Uncertainty](http://arxiv.org/abs/2106.07108)


  Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs) are
popular tools for enforcing safety and stability of a controlled system,
respectively. They are commonly utilized to build constraints that can be
incorporated in a min-norm quadratic program (CBF-CLF-QP) which solves for a
safety-critical control input. However, since these constraints rely on a model
of the system, when this model is inaccurate the guarantees of safety and
stability can be easily lost. In this paper, we present a Gaussian Process
(GP)-based approach to tackle the problem of model uncertainty in
safety-critical controllers that use CBFs and CLFs. The considered model
uncertainty is affected by both state and control input. We derive
probabilistic bounds on the effects that such model uncertainty has on the
dynamics of the CBF and CLF. We then use these bounds to build safety and
stability chance constraints that can be incorporated in a min-norm convex
optimization-based controller, called GP-CBF-CLF-SOCP. As the main theoretical
result of the paper, we present necessary and sufficient conditions for
pointwise feasibility of the proposed optimization problem. We believe that
these conditions could serve as a starting point towards understanding what are
the minimal requirements on the distribution of data collected from the real
system in order to guarantee safety. Finally, we validate the proposed
framework with numerical simulations of an adaptive cruise controller for an
automotive system.

    

### [[2106.10156] Predicting Gender by First Name Using Character-level Machine Learning](http://arxiv.org/abs/2106.10156)


  Predicting gender by the first name is not a simple task. In many
applications, especially in the natural language processing (NLP) field, this
task may be necessary, mainly when considering foreign names. In this paper, we
examined and implemented several machine learning algorithms, such as extra
trees, KNN, Naive Bayes, SVM, random forest, gradient boosting, light GBM,
logistic regression, ridge classifier, and deep neural network models, such as
MLP, RNN, GRU, CNN, and BiLSTM, to classify gender through the first name. A
dataset of Brazilian names is used to train and evaluate the models. We
analyzed the accuracy, recall, precision, f1 score, and confusion matrix to
measure the models' performances. The results indicate that the gender
prediction can be performed from the feature extraction strategy looking at the
names as a set of strings. Some models accurately predict gender in more than
95% of the cases. The recurrent models overcome the feedforward models in this
binary classification problem.

    

### [[2106.14269] Deep Learning for Technical Document Classification](http://arxiv.org/abs/2106.14269)


  In large technology companies, the requirements for managing and organizing
technical documents created by engineers and managers in supporting relevant
decision making have increased dramatically in recent years, which has led to a
higher demand for more scalable, accurate, and automated document
classification. Prior studies have only focused on processing text for
classification, whereas technical documents often contain multimodal
information. This paper presents a novel multimodal deep learning architecture,
TechDoc, for technical document classification, which utilizes three types of
information, including natural language texts and descriptive images within
documents and the associations among the documents. The architecture
synthesizes the convolutional neural network, recurrent neural network, and
graph neural network through an integrated multimodal training process. We
applied the architecture to a large multimodal technical document database and
trained the model for classifying documents based on the hierarchical
International Patent Classification system. Our results show that TechDoc
presents a greater classification accuracy than the unimodal methods and other
state-of-the-art methods.

    

### [[2109.13076] Using neural networks to solve the 2D Poisson equation for electric field computation in plasma fluid simulations](http://arxiv.org/abs/2109.13076)


  The Poisson equation is critical to get a self-consistent solution in plasma
fluid simulations used for Hall effect thrusters and streamers discharges.
Solving the 2D Poisson equation with zero Dirichlet boundary conditions using a
deep neural network is investigated using multiple-scale architectures, defined
in terms of number of branches, depth and receptive field. The latter is found
critical to correctly capture large topological structures of the field. The
investigation of multiple architectures, losses, and hyperparameters provides
an optimum network to solve accurately the steady Poisson problem.
Generalization to new resolutions and domain sizes is then proposed using a
proper scaling of the network. Finally, found neural network solver, called
PlasmaNet, is coupled with an unsteady Euler plasma fluid equations solver. The
test case corresponds to electron plasma oscillations which is used to assess
the accuracy of the neural network solution in a time-dependent simulation. In
this time-evolving problem, a physical loss is necessary to produce a stable
simulation. PlasmaNet is then benchmarked on meshes with increasing number of
nodes, and compared with an existing solver based on a standard linear system
algorithm for the Poisson equation. It outperforms the classical plasma solver,
up to speedups 700 times faster on large meshes. PlasmaNet is finally tested on
a more complex case of discharge propagation involving chemistry and advection.
The guidelines established in previous sections are applied to build the CNN to
solve the same Poisson equation but in cylindrical coordinates. Results reveal
good CNN predictions with significant speedup. These results pave the way to
new computational strategies to predict unsteady problems involving a Poisson
equation, including configurations with coupled multiphysics interactions such
as in plasma flows.

    

### [[2106.00720] Fair-Net: A Network Architecture For Reducing Performance Disparity Between Identifiable Sub-Populations](http://arxiv.org/abs/2106.00720)


  In real world datasets, particular groups are under-represented, much rarer
than others, and machine learning classifiers will often preform worse on
under-represented populations. This problem is aggravated across many domains
where datasets are class imbalanced, with a minority class far rarer than the
majority class. Naive approaches to handle under-representation and class
imbalance include training sub-population specific classifiers that handle
class imbalance or training a global classifier that overlooks sub-population
disparities and aims to achieve high overall accuracy by handling class
imbalance. In this study, we find that these approaches are vulnerable in class
imbalanced datasets with minority sub-populations. We introduced Fair-Net, a
branched multitask neural network architecture that improves both
classification accuracy and probability calibration across identifiable
sub-populations in class imbalanced datasets. Fair-Nets is a straightforward
extension to the output layer and error function of a network, so can be
incorporated in far more complex architectures. Empirical studies with three
real world benchmark datasets demonstrate that Fair-Net improves classification
and calibration performance, substantially reducing performance disparity
between gender and racial sub-populations.

    

### [[2106.05423] A New Notion of Individually Fair Clustering: $α$-Equitable $k$-Center](http://arxiv.org/abs/2106.05423)


  Clustering is a fundamental problem in unsupervised machine learning, and
fair variants of it have recently received significant attention due to its
societal implications. In this work we introduce a novel definition of
individual fairness for clustering problems. Specifically, in our model, each
point $j$ has a set of other points $\mathcal{S}_j$ that it perceives as
similar to itself, and it feels that it is fairly treated if the quality of
service it receives in the solution is $\alpha$-close (in a multiplicative
sense, for a given $\alpha \geq 1$) to that of the points in $\mathcal{S}_j$.
We begin our study by answering questions regarding the structure of the
problem, namely for what values of $\alpha$ the problem is well-defined, and
what the behavior of the \emph{Price of Fairness (PoF)} for it is. For the
well-defined region of $\alpha$, we provide efficient and easily-implementable
approximation algorithms for the $k$-center objective, which in certain cases
enjoy bounded-PoF guarantees. We finally complement our analysis by an
extensive suite of experiments that validates the effectiveness of our
theoretical results.

    

### [[2110.00777] Automated Seed Quality Testing System using GAN & Active Learning](http://arxiv.org/abs/2110.00777)


  Quality assessment of agricultural produce is a crucial step in minimizing
food stock wastage. However, this is currently done manually and often requires
expert supervision, especially in smaller seeds like corn. We propose a novel
computer vision-based system for automating this process. We build a novel seed
image acquisition setup, which captures both the top and bottom views. Dataset
collection for this problem has challenges of data annotation costs/time and
class imbalance. We address these challenges by i.) using a Conditional
Generative Adversarial Network (CGAN) to generate real-looking images for the
classes with lesser images and ii.) annotate a large dataset with minimal
expert human intervention by using a Batch Active Learning (BAL) based
annotation tool. We benchmark different image classification models on the
dataset obtained. We are able to get accuracies of up to 91.6% for testing the
physical purity of seed samples.

    

### [[2110.01103] Heterogeneous Dual-Core Overlay Processor for Light-Weight CNNs](http://arxiv.org/abs/2110.01103)


  Light-weight convolutional neural networks (CNNs) have small complexity and
are good candidates for low-power, high-throughput inference. Such networks are
heterogeneous in terms of computation-to-communication (CTC) ratios and
computation patterns between layers, especially for different layer types. Yet,
existing AI processors either use homogeneous processing elements (PEs),
resulting in low runtime PE efficiency, or run different layers on
heterogeneous PEs in sequential, introducing resource redundancy. This paper
proposes a heterogeneous dual-core architecture (dual-OPU), where one core is
optimized for regular convolution layers and the other for depthwise
convolution layers. PEs are homogeneous with each core. To make full use of
dual-core parallelism, we develop a scheduling algorithm to concurrently
execute layers for different input images on dual-core and balance parallel
workload. Meanwhile, we automatically tune the PE number for a core and tune
the input size for each PE to maximize throughput. Compared with a single-core
processor with the same area for a single network, heterogeneous dual-OPU on
average improves runtime PE efficiency and throughput by 11% and 31%,
respectively. For a workload of multiple networks, dual-OPU improves average
throughput by 11% compared with the state-of-the-art processors scaled to the
same area. To the best of our knowledge, it is the first in-depth study on the
heterogeneous dual-core processor for light-weight CNNs.

    

### [[2110.01202] Leaked-Web: Accurate and Efficient Machine Learning-Based Website Fingerprinting Attack through Hardware Performance Counters](http://arxiv.org/abs/2110.01202)


  Users' website browsing history contains sensitive information, like health
conditions, political interests, financial situations, etc. Some recent studies
have demonstrated the possibility of inferring website fingerprints based on
important usage information such as traffic, cache usage, memory usage, CPU
activity, power consumption, and hardware performance counters information.
However, existing website fingerprinting attacks demand a high sampling rate
which causes high performance overheads and large network traffic, and/or they
require launching an additional malicious website by the user, which is not
guaranteed. As a result, such drawbacks make the existing attacks more
noticeable to users and corresponding fingerprinting detection mechanisms. In
response, in this work, we propose Leaked-Web, a novel accurate and efficient
machine learning-based website fingerprinting attack through processor's
Hardware Performance Counters (HPCs). Leaked-Web efficiently collects hardware
performance counters in users' computer systems at a significantly low
granularity monitoring rate and sends the samples to the remote attack's server
for further classification. Leaked-Web examines the web browsers'
microarchitectural features using various advanced machine learning algorithms
ranging from classical, boosting, deep learning, and time-series models. Our
experimental results indicate that Leaked-Web based on a LogitBoost ML
classifier using only the top 4 HPC features achieves 91% classification
accuracy outperforming the state-of-the-art attacks by nearly 5%. Furthermore,
our proposed attack obtains a negligible performance overhead (only <1%),
around 12% lower than the existing hardware-assisted website fingerprinting
attacks.

    

### [[2110.01208] HyGain: High Performance, Energy-Efficient Hybrid Gain Cell based Cache Hierarchy](http://arxiv.org/abs/2110.01208)


  In this paper, we propose a 'full-stack' solution to designing high capacity
and low latency on-chip cache hierarchies by starting at the circuit level of
the hardware design stack. First, we propose a novel Gain Cell (GC) design
using FDSOI. The GC has several desirable characteristics, including ~50%
higher storage density and ~50% lower dynamic energy as compared to the
traditional 6T SRAM, even after accounting for peripheral circuit overheads. We
also exploit back-gate bias to increase retention time to 1.12 ms (~60x of
eDRAM) which, combined with optimizations like staggered refresh, makes it an
ideal candidate to architect all levels of on-chip caches. We show that
compared to 6T SRAM, for a given area budget, GC based caches, on average,
provide 29% and 36% increase in IPC for single- and multi-programmed workloads,
respectively on contemporary workloads including SPEC CPU2017. We also observe
dynamic energy savings of 42% and 34% for single- and multi-programmed
workloads, respectively.
We utilize the inherent properties of the proposed GC, including decoupled
read and write bitlines to devise optimizations to save precharge energy and
architect GC caches with better energy and performance characteristics.
Finally, in a quest to utilize the best of all worlds, we combine GC with
STT-RAM to create hybrid hierarchies. We show that a hybrid hierarchy with GC
caches at L1 and L2, and an LLC split between GC and STT-RAM, with asymmetric
write optimization enabled, is able to provide a 54% benefit in energy-delay
product (EDP) as compared to an all-SRAM design, and 13% as compared to an
all-GC cache hierarchy, averaged across multi-programmed workloads.

    

### [[2106.07449] Isadora: Automated Information Flow Property Generation for Hardware Designs](http://arxiv.org/abs/2106.07449)


  Isadora is a methodology for creating information flow specifications of
hardware designs. The methodology combines information flow tracking and
specification mining to produce a set of information flow properties that are
suitable for use during the security validation process, and which support a
better understanding of the security posture of the design. Isadora is fully
automated; the user provides only the design under consideration and a
testbench and need not supply a threat model nor security specifications. We
evaluate Isadora on a RISC-V processor plus two designs related to SoC access
control. Isadora generates security properties that align with those suggested
by the Common Weakness Enumerations (CWEs), and in the case of the SoC designs,
align with the properties written manually by security experts.

    

### [[2110.00617] Cuttlefish: Library for Achieving Energy Efficiency in Multicore Parallel Programs](http://arxiv.org/abs/2110.00617)


  A low-cap power budget is challenging for exascale computing. Dynamic Voltage
and Frequency Scaling (DVFS) and Uncore Frequency Scaling (UFS) are the two
widely used techniques for limiting the HPC application's energy footprint.
However, existing approaches fail to provide a unified solution that can work
with different types of parallel programming models and applications.
This paper proposes Cuttlefish, a programming model oblivious C/C++ library
for achieving energy efficiency in multicore parallel programs running over
Intel processors. An online profiler periodically profiles model-specific
registers to discover a running application's memory access pattern. Using a
combination of DVFS and UFS, Cuttlefish then dynamically adapts the processor's
core and uncore frequencies, thereby improving its energy efficiency. The
evaluation on a 20-core Intel Xeon processor using a set of widely used OpenMP
benchmarks, consisting of several irregular-tasking and work-sharing pragmas,
achieves geometric mean energy savings of 19.4% with a 3.6% slowdown.

    

### [[2110.00643] Distributed $Δ$-Coloring Plays Hide-and-Seek](http://arxiv.org/abs/2110.00643)


  We prove several new tight distributed lower bounds for classic symmetry
breaking graph problems. As a basic tool, we first provide a new insightful
proof that any deterministic distributed algorithm that computes a
$\Delta$-coloring on $\Delta$-regular trees requires $\Omega(\log_\Delta n)$
rounds and any randomized algorithm requires $\Omega(\log_\Delta\log n)$
rounds. We prove this result by showing that a natural relaxation of the
$\Delta$-coloring problem is a fixed point in the round elimination framework.
As a first application, we show that our $\Delta$-coloring lower bound proof
directly extends to arbdefective colorings. We exactly characterize which
variants of the arbdefective coloring problem are "easy", and which of them
instead are "hard".
As a second application, we use the structure of the fixed point as a
building block to prove lower bounds as a function of $\Delta$ for a large
class of distributed symmetry breaking problems. For example, we obtain a tight
linear-in-$\Delta$ lower bound for computing a maximal independent set in
$\Delta$-regular trees. For the case where an initial $O(\Delta)$-coloring is
given, we obtain a tight lower bound for computing a $(2,\beta)$-ruling set.
Our lower bounds even apply to a much more general family of problems, such as
variants of ruling sets where nodes in the set do not need to satisfy the
independence requirement, but they only need to satisfy the requirements of
some arbdefective coloring.
Our lower bounds as a function of $\Delta$ also imply lower bounds as a
function of $n$. We obtain, for example, that maximal independent set, on
trees, requires $\Omega(\log n / \log \log n)$ rounds for deterministic
algorithms, which is tight.

    

### [[2110.00741] Lower Bounds for Induced Cycle Detection in Distributed Computing](http://arxiv.org/abs/2110.00741)


  The distributed subgraph detection asks, for a fixed graph $H$, whether the
$n$-node input graph contains $H$ as a subgraph or not. In the standard CONGEST
model of distributed computing, the complexity of clique/cycle detection and
listing has received a lot of attention recently.
In this paper we consider the induced variant of subgraph detection, where
the goal is to decide whether the $n$-node input graph contains $H$ as an
\emph{induced} subgraph or not. We first show a $\tilde{\Omega}(n)$ lower bound
for detecting the existence of an induced $k$-cycle for any $k\geq 4$ in the
CONGEST model. This lower bound is tight for $k=4$, and shows that the induced
variant of $k$-cycle detection is much harder than the non-induced version.
This lower bound is proved via a reduction from two-party communication
complexity. We complement this result by showing that for $5\leq k\leq 7$, this
$\tilde{\Omega}(n)$ lower bound cannot be improved via the two-party
communication framework.
We then show how to prove stronger lower bounds for larger values of $k$.
More precisely, we show that detecting an induced $k$-cycle for any $k\geq 8$
requires $\tilde{\Omega}(n^{2-\Theta{(1/k)}})$ rounds in the CONGEST model,
nearly matching the known upper bound $\tilde{O}(n^{2-\Theta{(1/k)}})$ of the
general $k$-node subgraph detection (which also applies to the induced version)
by Eden, Fiat, Fischer, Kuhn, and Oshman~[DISC 2019].
Finally, we investigate the case where $H$ is the diamond (the diamond is
obtained by adding an edge to a 4-cycle, or equivalently removing an edge from
a 4-clique), and show non-trivial upper and lower bounds on the complexity of
the induced version of diamond detecting and listing.

    

### [[2110.00819] Multi-Feasibility Variable Selection](http://arxiv.org/abs/2110.00819)


  This paper is the report of the problem proposed for the !Optimizer 2021
competition, and the solutions of the gold medalist team, i.e., the Panda team.
The competition was held in two stages, the research and development stage and
a two-week contest stage, consisting of five rounds, and seven teams succeeded
in finishing both stages to the end. In this joint report of the winner team
Panda and the problem design committee coordinated by Mojtaba Tefagh, we first
explain each of the five rounds and then provide the solutions proposed by our
team (Panda) to fulfill the required tasks in the fastest and most accurate
way. Afterward, some preprocessing and data manipulating ideas used to enhance
the algorithms would be presented. All codes are written in the Julia language,
which showed a better performance than Python on optimization problems in our
comparisons during the R&D stage, and are publicly available in the Github
repository: this https URL


### [[2110.00846] Repttack: Exploiting Cloud Schedulers to Guide Co-Location Attacks](http://arxiv.org/abs/2110.00846)


  Cloud computing paradigms have emerged as a major facility to store and
process the massive data produced by various business units, public
organizations, Internet-of-Things, and cyber-physical systems. To meet users'
performance requirements while maximizing resource utilization to achieve
cost-efficiency, cloud administrators leverage schedulers to orchestrate tasks
to different physical nodes and allow applications from different users to
share the same physical node. On the other hand, micro-architectural attacks
can exploit the shared resources to compromise the confidentiality/integrity of
a co-located victim application. Since co-location is an essential requirement
for micro-architectural attacks, in this work, we investigate whether attackers
can exploit the cloud schedulers to satisfy the co-location requirement. Our
analysis shows that for cloud schedulers that allow users to submit application
requirements, an attacker can carefully select the attacker's application
requirements to influence the scheduler to co-locate it with a targeted victim
application. We call such attack Replication Attack (Repttack). Our
experimental results, in both a simulated cluster environment and a real
cluster, show similar trends; a single attack instance can reach up to 50%
co-location rate and with only 5 instances the co-location rate can reach up to
80%. Furthermore, we propose and evaluate a mitigation strategy that can help
defend against Repttack. We believe that our results highlight the fact that
schedulers in multi-user clusters need to be more carefully designed with
security in mind, and the process of making scheduling decisions should involve
as little user-defined information as possible.

    

### [[2110.00886] Spindle: Techniques for Optimizing Atomic Multicast on RDMA](http://arxiv.org/abs/2110.00886)


  Leveraging one-sided RDMA for applications that replicate small data objects
can be surprisingly difficult: such uses amplify any protocol overheads.
Spindle is a set of optimization techniques for systematically tackling this
class of challenges for atomic multicast over RDMA. These include memory
polling optimizations using novel sender and receiver batching techniques,
null-message send logic, and improved multi-thread synchronization. We applied
Spindle to Derecho, an open-source C++ library for atomic multicast, and
obtained significant performance improvements both for the library itself and
for an OMG-compliant avionics DDS built over Derecho. Derecho's multicast
bandwidth utilization for 10KB messages rose from 1GB/s to 9.7GB/s on a
12.5GB/s network, and it became more robust to delays. Interestingly, although
some of our techniques employ batching, latency dropped by nearly two orders of
magnitude. Spindle optimizations should also be of value in other RDMA
applications limited by the speed of coordination.

    

### [[2110.00960] Be Aware of Your Leaders](http://arxiv.org/abs/2110.00960)


  Advances in blockchains have influenced the State-Machine-Replication (SMR)
world and many state-of-the-art blockchain-SMR solutions are based on two
pillars: Chaining and Leader-rotation. A predetermined round-robin mechanism
used for Leader-rotation, however, has an undesirable behavior: crashed parties
become designated leaders infinitely often, slowing down overall system
performance. In this paper, we provide a new Leader-Aware SMR framework that,
among other desirable properties, formalizes a Leader-utilization requirement
that bounds the number of rounds whose leaders are faulty in crash-only
executions. We introduce Carousel, a novel, reputation-based Leader-rotation
solution to achieve Leader-Aware SMR. The challenge in adaptive Leader-rotation
is that it cannot rely on consensus to determine a leader, since consensus
itself needs a leader. Carousel uses the available on-chain information to
determine a leader locally and achieves Liveness despite this difficulty. A
HotStuff implementation fitted with Carousel demonstrates drastic performance
improvements: it increases throughput over 2x in faultless settings and
provided a 20x throughput increase and 5x latency reduction in the presence of
faults.

    

### [[2110.01162] Controlling Resource Allocation using Blockchain-Based Delegation](http://arxiv.org/abs/2110.01162)


  Allocation of resources and their control over multiple organisations is
challenging. This is especially true for a large-scale and dynamic system like
the Internet of Things (IoT). One of the core issues in such a system is the
provision of secure access control. In particular, transfer of access rights
from one entity to another in a secure, flexible and fine-grained manner. In
this paper, we present a multi-organisational delegation framework using
blockchain. Our framework takes advantage of blockchain smart contracts to
define the interactions and resource allocation between the consortium of
organisations. We show the feasibility of our solution in a real-world scenario
using the allocation of transportation credits in a multi-level organisational
setting as a use-case. We provide proof of implementation of the proposed
framework using the Hyperledger Fabric blockchain platform. Our results
indicate that the proposed framework is efficient and can be used for city-wide
transport, potentially even scale country-wide with a shared blockchain with
complex access control rules. It also bestows better transparency to the
delegation of access rights and control over the employees' transportation
access for the organisations.

    

### [[2110.01172] A New Acceleration Paradigm for Discrete CosineTransform and Other Fourier-Related Transforms](http://arxiv.org/abs/2110.01172)


  Discrete cosine transform (DCT) and other Fourier-related transforms have
broad applications in scientific computing. However, off-the-shelf
high-performance multi-dimensional DCT (MD DCT) libraries are not readily
available in parallel computing systems. Public MD DCT implementations leverage
a straightforward method that decomposes the computation into multiple 1D DCTs
along every single dimension, which inevitably has non-optimal performance due
to low computational efficiency, parallelism, and locality. In this paper, we
propose a new acceleration paradigm for MD DCT. A three-stage procedure is
proposed to factorize MD DCT into MD FFT and highly-optimized
preprocessing/postprocessing with efficient computation and high arithmetic
intensity. Our paradigm can be easily extended to other Fourier-related
transforms and other parallel computing systems. Experimental results show that
our 2D DCT/IDCT CUDA implementation has a stable, FFT-comparable execution
time, which is $2\times$ faster than the previous row-column method. Several
case studies demonstrate that a promising efficiency improvement can be
achieved with our paradigm. The implementations are available at
this https URL.

    

### [[2110.01229] AsymML: An Asymmetric Decomposition Framework for Privacy-Preserving DNN Training and Inference](http://arxiv.org/abs/2110.01229)


  Leveraging parallel hardware (e.g. GPUs) to conduct deep neural network (DNN)
training/inference, though significantly speeds up the computations, raises
several data privacy concerns. Trusted execution environments (TEEs) have
emerged as a promising solution to enable privacy-preserving inference and
training. TEEs, however, have limited memory and computation resources which
renders it not comparable to untrusted parallel hardware in performance. To
mitigate the trade-off between privacy and computing performance, we propose an
asymmetric model decomposition framework, AsymML, to (1) accelerate
training/inference using parallel hardware; and (2) preserve privacy using
TEEs. By exploiting the low-rank characteristics in data and intermediate
features, AsymML asymmetrically splits a DNN model into trusted and untrusted
parts: the trusted part features privacy-sensitive data but incurs small
compute/memory costs; while the untrusted part is computationally-intensive but
not privacy-sensitive. Computing performance and privacy are guaranteed by
respectively delegating the trusted and untrusted part to TEEs and GPUs.
Furthermore, we present a theoretical rank bound analysis showing that low-rank
characteristics are still preserved in intermediate features, which guarantees
efficiency of AsymML. Extensive evaluations on DNN models shows that AsymML
delivers $11.2\times$ speedup in inference, $7.6\times$ in training compared to
the TEE-only executions.

    

### [[1808.06705] Graph connectivity in log steps using label propagation](http://arxiv.org/abs/1808.06705)


  The fastest deterministic algorithms for connected components take
logarithmic time and perform superlinear work on a Parallel Random Access
Machine (PRAM). These algorithms maintain a spanning forest by merging and
compressing trees, which requires pointer-chasing operations that increase
memory access latency and are limited to shared-memory systems. Many of these
PRAM algorithms are also very complicated to implement. Another popular method
is "leader-contraction" where the challenge is to select a constant fraction of
leaders that are adjacent to a constant fraction of non-leaders with high
probability. Instead we investigate label propagation because it is
deterministic and does not rely on pointer-chasing. Label propagation exchanges
representative labels within a component using simple graph traversal, but it
is inherently difficult to complete in a sublinear number of steps. We are able
to solve the problems with label propagation for graph connectivity.
We introduce a surprisingly simple framework for deterministic graph
connectivity using label propagation that is easily adaptable to many
computational models. It propagates directed edges and alternates edge
direction to achieve linear edge count each step and sublinear convergence. We
present new algorithms in PRAM, Stream, and MapReduce for a simple, undirected
graph $G=(V,E)$ with $n=|V|$ vertices, $m=|E|$ edges. Our approach takes $O(m)$
work each step, but we can only prove logarithmic convergence on a path graph.
It was conjectured by Liu and Tarjan (2019) to take $O(\log n)$ steps or
possibly $O(\log^2 n)$ steps. We leave the proof of convergence as an open
problem.

    

### [[2006.01866] ALADIN-$α$ -- An open-source MATLAB toolbox for distributed non-convex optimization](http://arxiv.org/abs/2006.01866)


  This paper introduces an open-source software for distributed and
decentralized non-convex optimization named ALADIN-$\alpha$. ALADIN-$\alpha$ is
a MATLAB implementation of tailored variants of the Augmented Lagrangian
Alternating Direction Inexact Newton (ALADIN) algorithm. Its user interface is
convenient for rapid prototyping of non-convex distributed optimization
algorithms. An improved version of the recently proposed bi-level variant of
ALADIN is included enabling decentralized non-convex optimization with reduced
information exchange. A collection of examples from different applications
fields including chemical engineering, robotics, and power systems underpins
the potential of ALADIN-$\alpha$.

    

### [[2010.11454] Fast-HotStuff: A Fast and Resilient HotStuff Protocol](http://arxiv.org/abs/2010.11454)


  The HotStuff protocol is a breakthrough in Byzantine Fault Tolerant (BFT)
consensus that enjoys both responsiveness and linear view change. It creatively
adds an additional round to classic BFT protocols (like PBFT) using two rounds.
This brings us to an interesting question: Is this additional round really
necessary in practice? In this paper, we answer this question by designing a
new two-round BFT protocol called Fast-HotStuff, which enjoys responsiveness
and efficient view change that is comparable to linear view change in terms of
performance. Compared to (three-round) HotStuff, Fast-HotStuff has lower
latency and is more robust against performance attacks that HotStuff is
susceptible to.

    

### [[2110.00712] Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages](http://arxiv.org/abs/2110.00712)


  Although the multilingual Neural Machine Translation(NMT), which extends
Google's multilingual NMT, has ability to perform zero-shot translation and the
iterative self-learning algorithm can improve the quality of zero-shot
translation, it confronts with two problems: the multilingual NMT model is
prone to generate wrong target language when implementing zero-shot
translation; the self-learning algorithm, which uses beam search to generate
synthetic parallel data, demolishes the diversity of the generated source
language and amplifies the impact of the same noise during the iterative
learning process. In this paper, we propose the tagged-multilingual NMT model
and improve the self-learning algorithm to handle these two problems. Firstly,
we extend the Google's multilingual NMT model and add target tokens to the
target languages, which associates the start tag with the target language to
ensure that the source language can be translated to the required target
language. Secondly, we improve the self-learning algorithm by replacing beam
search with random sample to increases the diversity of the generated data and
makes it properly cover the true data distribution. Experimental results on
IWSLT show that the adjusted tagged-multilingual NMT separately obtains 9.41
and 7.85 BLEU scores over the multilingual NMT on 2010 and 2017
Romanian-Italian test sets. Similarly, it obtains 9.08 and 7.99 BLEU scores on
Italian-Romanian zero-shot translation. Furthermore, the improved self-learning
algorithm shows its superiorities over the conventional self-learning algorithm
on zero-shot translations.

    

### [[2110.00740] FICGAN: Facial Identity Controllable GAN for De-identification](http://arxiv.org/abs/2110.00740)


  In this work, we present Facial Identity Controllable GAN (FICGAN) for not
only generating high-quality de-identified face images with ensured privacy
protection, but also detailed controllability on attribute preservation for
enhanced data utility. We tackle the less-explored yet desired functionality in
face de-identification based on the two factors. First, we focus on the
challenging issue to obtain a high level of privacy protection in the
de-identification task while uncompromising the image quality. Second, we
analyze the facial attributes related to identity and non-identity and explore
the trade-off between the degree of face de-identification and preservation of
the source attributes for enhanced data utility. Based on the analysis, we
develop Facial Identity Controllable GAN (FICGAN), an autoencoder-based
conditional generative model that learns to disentangle the identity attributes
from non-identity attributes on a face image. By applying the manifold k-same
algorithm to satisfy k-anonymity for strengthened security, our method achieves
enhanced privacy protection in de-identified face images. Numerous experiments
demonstrate that our model outperforms others in various scenarios of face
de-identification.

    

### [[2110.00758] Making Things Explainable vs Explaining: Requirements and Challenges under the GDPR](http://arxiv.org/abs/2110.00758)


  The European Union (EU) through the High-Level Expert Group on Artificial
Intelligence (AI-HLEG) and the General Data Protection Regulation (GDPR) has
recently posed an interesting challenge to the eXplainable AI (XAI) community,
by demanding a more user-centred approach to explain Automated Decision-Making
systems (ADMs). Looking at the relevant literature, XAI is currently focused on
producing explainable software and explanations that generally follow an
approach we could term One-Size-Fits-All, that is unable to meet a requirement
of centring on user needs. One of the causes of this limit is the belief that
making things explainable alone is enough to have pragmatic explanations. Thus,
insisting on a clear separation between explainabilty (something that can be
explained) and explanations, we point to explanatorY AI (YAI) as an alternative
and more powerful approach to win the AI-HLEG challenge. YAI builds over XAI
with the goal to collect and organize explainable information, articulating it
into something we called user-centred explanatory discourses. Through the use
of explanatory discourses/narratives we represent the problem of generating
explanations for Automated Decision-Making systems (ADMs) into the
identification of an appropriate path over an explanatory space, allowing
explainees to interactively explore it and produce the explanation best suited
to their needs.

    

### [[2110.00762] Generating User-Centred Explanations via Illocutionary Question Answering: From Philosophy to Interfaces](http://arxiv.org/abs/2110.00762)


  We propose a new method for generating explanations with Artificial
Intelligence (AI) and a tool to test its expressive power within a user
interface. In order to bridge the gap between philosophy and human-computer
interfaces, we show a new approach for the generation of interactive
explanations based on a sophisticated pipeline of AI algorithms for structuring
natural language documents into knowledge graphs, answering questions
effectively and satisfactorily. With this work we aim to prove that the
philosophical theory of explanations presented by Achinstein can be actually
adapted for being implemented into a concrete software application, as an
interactive and illocutionary process of answering questions. Specifically, our
contribution is an approach to frame illocution in a computer-friendly way, to
achieve user-centrality with statistical question answering. In fact, we frame
illocution, in an explanatory process, as that mechanism responsible for
anticipating the needs of the explainee in the form of unposed, implicit,
archetypal questions, hence improving the user-centrality of the underlying
explanatory process. More precisely, we hypothesise that given an arbitrary
explanatory process, increasing its goal-orientedness and degree of illocution
results in the generation of more usable (as per ISO 9241-210) explanations. We
tested our hypotheses with a user-study involving more than 60 participants, on
two XAI-based systems, one for credit approval (finance) and one for heart
disease prediction (healthcare). The results showed that our proposed solution
produced a statistically significant improvement (hence with a p-value lower
than 0.05) on effectiveness. This, combined with a visible alignment between
the increments in effectiveness and satisfaction, suggests that our
understanding of illocution can be correct, giving evidence in favour of our
theory.

    

### [[2110.00791] Optimizing Neural Network for Computer Vision task in Edge Device](http://arxiv.org/abs/2110.00791)


  The field of computer vision has grown very rapidly in the past few years due
to networks like convolution neural networks and their variants. The memory
required to store the model and computational expense are very high for such a
network limiting it to deploy on the edge device. Many times, applications rely
on the cloud but that makes it hard for working in real-time due to round-trip
delays. We overcome these problems by deploying the neural network on the edge
device itself. The computational expense for edge devices is reduced by
reducing the floating-point precision of the parameters in the model. After
this the memory required for the model decreases and the speed of the
computation increases where the performance of the model is least affected.
This makes an edge device to predict from the neural network all by itself.

    

### [[2110.00828] Artificial intelligence for Sustainable Energy: A Contextual Topic Modeling and Content Analysis](http://arxiv.org/abs/2110.00828)


  Parallel to the rising debates over sustainable energy and artificial
intelligence solutions, the world is currently discussing the ethics of
artificial intelligence and its possible negative effects on society and the
environment. In these arguments, sustainable AI is proposed, which aims at
advancing the pathway toward sustainability, such as sustainable energy. In
this paper, we offered a novel contextual topic modeling combining LDA, BERT,
and Clustering. We then combined these computational analyses with content
analysis of related scientific publications to identify the main scholarly
topics, sub-themes, and cross-topic themes within scientific research on
sustainable AI in energy. Our research identified eight dominant topics
including sustainable buildings, AI-based DSSs for urban water management,
climate artificial intelligence, Agriculture 4, the convergence of AI with IoT,
AI-based evaluation of renewable technologies, smart campus and engineering
education, and AI-based optimization. We then recommended 14 potential future
research strands based on the observed theoretical gaps. Theoretically, this
analysis contributes to the existing literature on sustainable AI and
sustainable energy, and practically, it intends to act as a general guide for
energy engineers and scientists, AI scientists, and social scientists to widen
their knowledge of sustainability in AI and energy convergence research.

    

### [[2110.00840] Induction, Popper, and machine learning](http://arxiv.org/abs/2110.00840)


  Francis Bacon popularized the idea that science is based on a process of
induction by which repeated observations are, in some unspecified way,
generalized to theories based on the assumption that the future resembles the
past. This idea was criticized by Hume and others as untenable leading to the
famous problem of induction. It wasn't until the work of Karl Popper that this
problem was solved, by demonstrating that induction is not the basis for
science and that the development of scientific knowledge is instead based on
the same principles as biological evolution. Today, machine learning is also
taught as being rooted in induction from big data. Solomonoff induction
implemented in an idealized Bayesian agent (Hutter's AIXI) is widely discussed
and touted as a framework for understanding AI algorithms, even though
real-world attempts to implement something like AIXI immediately encounter
fatal problems. In this paper, we contrast frameworks based on induction with
Donald T. Campbell's universal Darwinism. We show that most AI algorithms in
use today can be understood as using an evolutionary trial and error process
searching over a solution space. In this work we argue that a universal
Darwinian framework provides a better foundation for understanding AI systems.
Moreover, at a more meta level the process of development of all AI algorithms
can be understood under the framework of universal Darwinism.

    

### [[2110.00866] A Case Study to Reveal if an Area of Interest has a Trend in Ongoing Tweets Using Word and Sentence Embeddings](http://arxiv.org/abs/2110.00866)


  In the field of Natural Language Processing, information extraction from
texts has been the objective of many researchers for years. Many different
techniques have been applied in order to reveal the opinion that a tweet might
have, thus understanding the sentiment of the small writing up to 280
characters. Other than figuring out the sentiment of a tweet, a study can also
focus on finding the correlation of the tweets with a certain area of interest,
which constitutes the purpose of this study. In order to reveal if an area of
interest has a trend in ongoing tweets, we have proposed an easily applicable
automated methodology in which the Daily Mean Similarity Scores that show the
similarity between the daily tweet corpus and the target words representing our
area of interest is calculated by using a naïve correlation-based technique
without training any Machine Learning Model. The Daily Mean Similarity Scores
have mainly based on cosine similarity and word/sentence embeddings computed by
Multilanguage Universal Sentence Encoder and showed main opinion stream of the
tweets with respect to a certain area of interest, which proves that an ongoing
trend of a specific subject on Twitter can easily be captured in almost real
time by using the proposed methodology in this study. We have also compared the
effectiveness of using word versus sentence embeddings while applying our
methodology and realized that both give almost the same results, whereas using
word embeddings requires less computational time than sentence embeddings, thus
being more effective. This paper will start with an introduction followed by
the background information about the basics, then continue with the explanation
of the proposed methodology and later on finish by interpreting the results and
concluding the findings.

    

### [[2110.00898] A Novel Automated Curriculum Strategy to Solve Hard Sokoban Planning Instances](http://arxiv.org/abs/2110.00898)


  In recent years, we have witnessed tremendous progress in deep reinforcement
learning (RL) for tasks such as Go, Chess, video games, and robot control.
Nevertheless, other combinatorial domains, such as AI planning, still pose
considerable challenges for RL approaches. The key difficulty in those domains
is that a positive reward signal becomes {\em exponentially rare} as the
minimal solution length increases. So, an RL approach loses its training
signal. There has been promising recent progress by using a curriculum-driven
learning approach that is designed to solve a single hard instance. We present
a novel {\em automated} curriculum approach that dynamically selects from a
pool of unlabeled training instances of varying task complexity guided by our
{\em difficulty quantum momentum} strategy. We show how the smoothness of the
task hardness impacts the final learning results. In particular, as the size of
the instance pool increases, the ``hardness gap'' decreases, which facilitates
a smoother automated curriculum based learning process. Our automated
curriculum approach dramatically improves upon the previous approaches. We show
our results on Sokoban, which is a traditional PSPACE-complete planning problem
and presents a great challenge even for specialized solvers. Our RL agent can
solve hard instances that are far out of reach for any previous
state-of-the-art Sokoban solver. In particular, our approach can uncover plans
that require hundreds of steps, while the best previous search methods would
take many years of computing time to solve such instances. In addition, we show
that we can further boost the RL performance with an intricate coupling of our
automated curriculum approach with a curiosity-driven search strategy and a
graph neural net representation.

    

### [[2110.00931] Exploration of AI-Oriented Power System Transient Stability Simulations](http://arxiv.org/abs/2110.00931)


  Artificial Intelligence (AI) has made significant progress in the past 5
years and is playing a more and more important role in power system analysis
and control. It is foreseeable that the future power system transient stability
simulations will be deeply integrated with AI. However, the existing power
system dynamic simulation tools are not AI-friendly enough. In this paper, a
general design of an AI-oriented power system transient stability simulator is
proposed. It is a parallel simulator with a flexible application programming
interface so that the simulator has rapid simulation speed, neural network
supportability, and network topology accessibility. A prototype of this design
is implemented and made public based on our previously realized simulator.
Tests of this AI-oriented simulator are carried out under multiple scenarios,
which proves that the design and implementation of the simulator are
reasonable, AI-friendly, and highly efficient.

    

### [[2110.00934] Bounding Box Tightness Prior for Weakly Supervised Image Segmentation](http://arxiv.org/abs/2110.00934)


  This paper presents a weakly supervised image segmentation method that adopts
tight bounding box annotations. It proposes generalized multiple instance
learning (MIL) and smooth maximum approximation to integrate the bounding box
tightness prior into the deep neural network in an end-to-end manner. In
generalized MIL, positive bags are defined by parallel crossing lines with a
set of different angles, and negative bags are defined as individual pixels
outside of any bounding boxes. Two variants of smooth maximum approximation,
i.e., $\alpha$-softmax function and $\alpha$-quasimax function, are exploited
to conquer the numeral instability introduced by maximum function of bag
prediction. The proposed approach was evaluated on two pubic medical datasets
using Dice coefficient. The results demonstrate that it outperforms the
state-of-the-art methods. The codes are available at
\url{this https URL}.

    

### [[2110.00940] PL-EESR: Perceptual Loss Based END-TO-END Robust Speaker Representation Extraction](http://arxiv.org/abs/2110.00940)


  Speech enhancement aims to improve the perceptual quality of the speech
signal by suppression of the background noise. However, excessive suppression
may lead to speech distortion and speaker information loss, which degrades the
performance of speaker embedding extraction. To alleviate this problem, we
propose an end-to-end deep learning framework, dubbed PL-EESR, for robust
speaker representation extraction. This framework is optimized based on the
feedback of the speaker identification task and the high-level perceptual
deviation between the raw speech signal and its noisy version. We conducted
speaker verification tasks in both noisy and clean environment respectively to
evaluate our system. Compared to the baseline, our method shows better
performance in both clean and noisy environments, which means our method can
not only enhance the speaker relative information but also avoid adding
distortions.

    

### [[2110.00943] Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography](http://arxiv.org/abs/2110.00943)


  The cup-to-disc ratio (CDR) is one of the most significant indicator for
glaucoma diagnosis. Different from the use of costly fully supervised learning
formulation with pixel-wise annotations in the literature, this study
investigates the feasibility of accurate CDR measurement in fundus images using
only tight bounding box supervision. For this purpose, we develop a two-task
network for accurate CDR measurement, one for weakly supervised image
segmentation, and the other for bounding-box regression. The weakly supervised
image segmentation task is implemented based on generalized multiple instance
learning formulation and smooth maximum approximation, and the bounding-box
regression task outputs class-specific bounding box prediction in a single
scale at the original image resolution. To get accurate bounding box
prediction, a class-specific bounding-box normalizer and an expected
intersection-over-union are proposed. In the experiments, the proposed approach
was evaluated by a testing set with 1200 images using CDR error and F1 score
for CDR measurement and dice coefficient for image segmentation. A grader study
was conducted to compare the performance of the proposed approach with those of
individual graders. The results demonstrate that the proposed approach
outperforms the state-of-the-art performance obtained from the fully supervised
image segmentation (FSIS) approach using pixel-wise annotation for CDR
measurement, which is also better than those of individual graders. It also
gets performance close to the state-of-the-art obtained from FSIS for optic cup
and disc segmentation, similar to those of individual graders. The codes are
available at \url{this https URL}.

    

### [[2110.00992] Precise Object Placement with Pose Distance Estimations for Different Objects and Grippers](http://arxiv.org/abs/2110.00992)


  This paper introduces a novel approach for the grasping and precise placement
of various known rigid objects using multiple grippers within highly cluttered
scenes. Using a single depth image of the scene, our method estimates multiple
6D object poses together with an object class, a pose distance for object pose
estimation, and a pose distance from a target pose for object placement for
each automatically obtained grasp pose with a single forward pass of a neural
network. By incorporating model knowledge into the system, our approach has
higher success rates for grasping than state-of-the-art model-free approaches.
Furthermore, our method chooses grasps that result in significantly more
precise object placements than prior model-based work.

    

### [[2110.01013] Counterfactual Samples Synthesizing and Training for Robust Visual Question Answering](http://arxiv.org/abs/2110.01013)


  Today's VQA models still tend to capture superficial linguistic correlations
in the training set and fail to generalize to the test set with different QA
distributions. To reduce these language biases, recent VQA works introduce an
auxiliary question-only model to regularize the training of targeted VQA model,
and achieve dominating performance on diagnostic benchmarks for
out-of-distribution testing. However, due to complex model design, these
ensemble-based methods are unable to equip themselves with two indispensable
characteristics of an ideal VQA model: 1) Visual-explainable: The model should
rely on the right visual regions when making decisions. 2) Question-sensitive:
The model should be sensitive to the linguistic variations in questions. To
this end, we propose a novel model-agnostic Counterfactual Samples Synthesizing
and Training (CSST) strategy. After training with CSST, VQA models are forced
to focus on all critical objects and words, which significantly improves both
visual-explainable and question-sensitive abilities. Specifically, CSST is
composed of two parts: Counterfactual Samples Synthesizing (CSS) and
Counterfactual Samples Training (CST). CSS generates counterfactual samples by
carefully masking critical objects in images or words in questions and
assigning pseudo ground-truth answers. CST not only trains the VQA models with
both complementary samples to predict respective ground-truth answers, but also
urges the VQA models to further distinguish the original samples and
superficially similar counterfactual ones. To facilitate the CST training, we
propose two variants of supervised contrastive loss for VQA, and design an
effective positive and negative sample selection mechanism based on CSS.
Extensive experiments have shown the effectiveness of CSST. Particularly, by
building on top of model LMH+SAR, we achieve record-breaking performance on all
OOD benchmarks.

    

### [[2110.01056] Dr.Aid: Supporting Data-governance Rule Compliance for Decentralized Collaboration in an Automated Way](http://arxiv.org/abs/2110.01056)


  Collaboration across institutional boundaries is widespread and increasing
today. It depends on federations sharing data that often have governance rules
or external regulations restricting their use. However, the handling of data
governance rules (aka. data-use policies) remains manual, time-consuming and
error-prone, limiting the rate at which collaborations can form and respond to
challenges and opportunities, inhibiting citizen science and reducing data
providers' trust in compliance. Using an automated system to facilitate
compliance handling reduces substantially the time needed for such non-mission
work, thereby accelerating collaboration and improving productivity. We present
a framework, Dr.Aid, that helps individuals, organisations and federations
comply with data rules, using automation to track which rules are applicable as
data is passed between processes and as derived data is generated. It encodes
data-governance rules using a formal language and performs reasoning on
multi-input-multi-output data-flow graphs in decentralised contexts. We test
its power and utility by working with users performing cyclone tracking and
earthquake modelling to support mitigation and emergency response. We query
standard provenance traces to detach Dr.Aid from details of the tools and
systems they are using, as these inevitably vary across members of a federation
and through time. We evaluate the model in three aspects by encoding real-life
data-use policies from diverse fields, showing its capability for real-world
usage and its advantages compared with traditional frameworks. We argue that
this approach will lead to more agile, more productive and more trustworthy
collaborations and show that the approach can be adopted incrementally. This,
in-turn, will allow more appropriate data policies to emerge opening up new
forms of collaboration.

    

### [[2110.01152] Efficiency, Fairness, and Stability in Non-Commercial Peer-to-Peer Ridesharing](http://arxiv.org/abs/2110.01152)


  Unlike commercial ridesharing, non-commercial peer-to-peer (P2P) ridesharing
has been subject to limited research -- although it can promote viable
solutions in non-urban communities. This paper focuses on the core problem in
P2P ridesharing: the matching of riders and drivers. We elevate users'
preferences as a first-order concern and introduce novel notions of fairness
and stability in P2P ridesharing. We propose algorithms for efficient matching
while considering user-centric factors, including users' preferred departure
time, fairness, and stability. Results suggest that fair and stable solutions
can be obtained in reasonable computational times and can improve baseline
outcomes based on system-wide efficiency exclusively.

    

### [[2110.01186] The state-of-the-art in text-based automatic personality prediction](http://arxiv.org/abs/2110.01186)


  Personality detection is an old topic in psychology and Automatic Personality
Prediction (or Perception) (APP) is the automated (computationally) forecasting
of the personality on different types of human generated/exchanged contents
(such as text, speech, image, video). The principal objective of this study is
to offer a shallow (overall) review of natural language processing approaches
on APP since 2010. With the advent of deep learning and following it
transfer-learning and pre-trained model in NLP, APP research area has been a
hot topic, so in this review, methods are categorized into three; pre-trained
independent, pre-trained model based, multimodal approaches. Also, to achieve a
comprehensive comparison, reported results are informed by datasets.

    

### [[2110.01188] LawSum: A weakly supervised approach for Indian Legal Document Summarization](http://arxiv.org/abs/2110.01188)


  Unlike the courts in western countries, public records of Indian judiciary
are completely unstructured and noisy. No large scale publicly available
annotated datasets of Indian legal documents exist till date. This limits the
scope for legal analytics research. In this work, we propose a new dataset
consisting of over 10,000 judgements delivered by the supreme court of India
and their corresponding hand written summaries. The proposed dataset is
pre-processed by normalising common legal abbreviations, handling spelling
variations in named entities, handling bad punctuations and accurate sentence
tokenization. Each sentence is tagged with their rhetorical roles. We also
annotate each judgement with several attributes like date, names of the
plaintiffs, defendants and the people representing them, judges who delivered
the judgement, acts/statutes that are cited and the most common citations used
to refer the judgement. Further, we propose an automatic labelling technique
for identifying sentences which have summary worthy information. We demonstrate
that this auto labeled data can be used effectively to train a weakly
supervised sentence extractor with high accuracy. Some possible applications of
this dataset besides legal document summarization can be in retrieval, citation
analysis and prediction of decisions by a particular judge.

    

### [[2110.01232] Benchmarking Safety Monitors for Image Classifiers with Machine Learning](http://arxiv.org/abs/2110.01232)


  High-accurate machine learning (ML) image classifiers cannot guarantee that
they will not fail at operation. Thus, their deployment in safety-critical
applications such as autonomous vehicles is still an open issue. The use of
fault tolerance mechanisms such as safety monitors is a promising direction to
keep the system in a safe state despite errors of the ML classifier. As the
prediction from the ML is the core information directly impacting safety, many
works are focusing on monitoring the ML model itself. Checking the efficiency
of such monitors in the context of safety-critical applications is thus a
significant challenge. Therefore, this paper aims at establishing a baseline
framework for benchmarking monitors for ML image classifiers. Furthermore, we
propose a framework covering the entire pipeline, from data generation to
evaluation. Our approach measures monitor performance with a broader set of
metrics than usually proposed in the literature. Moreover, we benchmark three
different monitor approaches in 79 benchmark datasets containing five
categories of out-of-distribution data for image classifiers: class novelty,
noise, anomalies, distributional shifts, and adversarial attacks. Our results
indicate that these monitors are no more accurate than a random monitor. We
also release the code of all experiments for reproducibility.

    

### [[2108.13343] A Mathematical Walkthrough and Discussion of the Free Energy Principle](http://arxiv.org/abs/2108.13343)


  The Free-Energy-Principle (FEP) is an influential and controversial theory
which postulates a deep and powerful connection between the stochastic
thermodynamics of self-organization and learning through variational inference.
Specifically, it claims that any self-organizing system which can be
statistically separated from its environment, and which maintains itself at a
non-equilibrium steady state, can be construed as minimizing an
information-theoretic functional -- the variational free energy -- and thus
performing variational Bayesian inference to infer the hidden state of its
environment. This principle has also been applied extensively in neuroscience,
and is beginning to make inroads in machine learning by spurring the
construction of novel and powerful algorithms by which action, perception, and
learning can all be unified under a single objective. While its expansive and
often grandiose claims have spurred significant debates in both philosophy and
theoretical neuroscience, the mathematical depth and lack of accessible
introductions and tutorials for the core claims of the theory have often
precluded a deep understanding within the literature. Here, we aim to provide a
mathematically detailed, yet intuitive walk-through of the formulation and
central claims of the FEP while also providing a discussion of the assumptions
necessary and potential limitations of the theory. Additionally, since the FEP
is a still a living theory, subject to internal controversy, change, and
revision, we also present a detailed appendix highlighting and condensing
current perspectives as well as controversies about the nature, applicability,
and the mathematical assumptions and formalisms underlying the FEP.

    

### [[2110.00633] Uniform Bounds for Scheduling with Job Size Estimates](http://arxiv.org/abs/2110.00633)


  We consider the problem of scheduling to minimize mean response time in M/G/1
queues where only estimated job sizes (processing times) are known to the
scheduler, where a job of true size $s$ has estimated size in the interval
$[\beta s, \alpha s]$ for some $\alpha \geq \beta > 0$. We evaluate each
scheduling policy by its approximation ratio, which we define to be the ratio
between its mean response time and that of Shortest Remaining Processing Time
(SRPT), the optimal policy when true sizes are known. Our question: is there a
scheduling policy that (a) has approximation ratio near 1 when $\alpha$ and
$\beta$ are near 1, (b) has approximation ratio bounded by some function of
$\alpha$ and $\beta$ even when they are far from 1, and (c) can be implemented
without knowledge of $\alpha$ and $\beta$?
We first show that naively running SRPT using estimated sizes in place of
true sizes is not such a policy: its approximation ratio can be arbitrarily
large for any fixed $\beta < 1$. We then provide a simple variant of SRPT for
estimated sizes that satisfies criteria (a), (b), and (c). In particular, we
prove its approximation ratio approaches 1 uniformly as $\alpha$ and $\beta$
approach 1. This is the first result showing this type of convergence for M/G/1
scheduling.
We also study the Preemptive Shortest Job First (PSJF) policy, a cousin of
SRPT. We show that, unlike SRPT, naively running PSJF using estimated sizes in
place of true sizes satisfies criteria (b) and (c), as well as a weaker version
of (a).

    

### [[1910.11184] Cubic Metric Reduction for Repetitive CAZAC Sequences in frequency domain](http://arxiv.org/abs/1910.11184)


  In NR-based Access to Unlicensed Spectrum (NR-U) of 5G system, to satisfy the
rules of Occupied Channel Bandwidth (OCB) of unlicensed spectrum, the channels
of PRACH and PUCCH have to use some sequence repetition mechanisms in frequency
domain. These repetition mechanisms will cause serious cubic metric(CM)
problems for these channels, although these two types of channels are composed
of Constant Amplitude Zero Auto-correlation(CAZAC) sequences.. Based on the
characteristics of CAZAC sequences, which are used for PRACH and PUCCH (refer
to PUCCH format 0 and format 1) in 5G NR, in this paper, we propose some new
mechanisms of CM reduction for these two types of channels considering the
design principles to ensure the sequence performance of the auto-correlation
and cross-correlation. Then the proposed CM schemes are evaluated and the
optimized parameters are further provided considering CM performance and the
complexity.

    

### [[2110.00677] SolType: Refinement Types for Solidity](http://arxiv.org/abs/2110.00677)


  As smart contracts gain adoption in financial transactions, it becomes
increasingly important to ensure that they are free of bugs and security
vulnerabilities. Of particular relevance in this context are arithmetic
overflow bugs, as integers are often used to represent financial assets like
account balances. Motivated by this observation, this paper presents SolType, a
refinement type system for Solidity that can be used to prevent arithmetic
over- and under-flows in smart contracts. SolType allows developers to add
refinement type annotations and uses them to prove that arithmetic operations
do not lead to over- and under-flows. SolType incorporates a rich vocabulary of
refinement terms that allow expressing relationships between integer values and
aggregate properties of complex data structures. Furthermore, our
implementation, called Solid, incorporates a type inference engine and can
automatically infer useful type annotations, including non-trivial contract
invariants.
To evaluate the usefulness of our type system, we use Solid to prove
arithmetic safety of a total of 120 smart contracts. When used in its fully
automated mode (i.e., using Solid's type inference capabilities), Solid is able
to eliminate 86.3% of redundant runtime checks used to guard against overflows.
We also compare Solid against a state-of-the-art arithmetic safety verifier
called VeriSmart and show that Solid has a significantly lower false positive
rate, while being significantly faster in terms of verification time.

    

### [[2110.00776] Minimizing LR(1) State Machines is NP-Hard](http://arxiv.org/abs/2110.00776)


  LR(1) parsing was a focus of extensive research in the past 50 years. Though
most fundamental mysteries have been resolved, a few remain hidden in the dark
corners. The one we bumped into is the minimization of the LR(1) state
machines, which we prove is NP-hard. It is the node-coloring problem that is
reduced to the minimization puzzle. The reduction makes use of two technique:
indirect reduction and incremental construction. Indirect reduction means the
graph to be colored is not reduced to an LR(1) state machine directly. Instead,
it is reduced to a context-free grammar from which an LR(1) state machine is
derived. Furthermore, by considering the nodes in the graph to be colored one
at a time, the context-free grammar is incrementally extended from a template
context-free grammar that is for a two-node graph. The extension is done by
adding new grammar symbols and rules. A minimized LR(1) machine can be used to
recover a minimum coloring of the original graph.

    

### [[2110.01098] Does the Bronze Garbage Collector Make Rust Easier to Use? A Controlled Experiment](http://arxiv.org/abs/2110.01098)


  Rust is a general-purpose programming language that is both type- and
memory-safe. Rust does not use a garbage collector, but rather achieves these
properties through a sophisticated, but complex, type system. Doing so makes
Rust very efficient, but makes Rust relatively hard to learn and use. We
designed Bronze, an optional, library-based garbage collector for Rust. To see
whether Bronze could make Rust more usable, we conducted a randomized
controlled trial with volunteers from a 633-person class, collecting data from
428 students in total. We found that for a task that required managing complex
aliasing, Bronze users were more likely to complete the task in the time
available, and those who did so required only about a third as much time (4
hours vs. 12 hours). We found no significant difference in total time, even
though Bronze users re-did the task without Bronze afterward. Surveys indicated
that ownership, borrowing, and lifetimes were primary causes of the challenges
that users faced when using Rust.

    

### [<title>When will GPU pre-built be available for R on Win 64? - XGBoost</title>](https://discuss.xgboost.ai/t/when-will-gpu-pre-built-be-available-for-r-on-win-64/2463/5)