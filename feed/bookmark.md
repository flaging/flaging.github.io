
## 2021-9-1

### [[2108.12971] HELMHOLTZ: A Verifier for Tezos Smart Contracts Based on Refinement Types](http://arxiv.org/abs/2108.12971)


  A smart contract is a program executed on a blockchain, based on which many
cryptocurrencies are implemented, and is being used for automating
transactions. Due to the large amount of money that smart contracts deal with,
there is a surging demand for a method that can statically and formally verify
them.
This article describes our type-based static verification tool HELMHOLTZ for
Michelson, which is a statically typed stack-based language for writing smart
contracts that are executed on the blockchain platform Tezos. HELMHOLTZ is
designed on top of our extension of Michelson's type system with refinement
types. HELMHOLTZ takes a Michelson program annotated with a user-defined
specification written in the form of a refinement type as input; it then
typechecks the program against the specification based on the refinement type
system, discharging the generated verification conditions with the SMT solver
Z3. We briefly introduce our refinement type system for the core calculus
Mini-Michelson of Michelson, which incorporates the characteristic features
such as compound datatypes (e.g., lists and pairs), higher-order functions, and
invocation of another contract. \HELMHOLTZ{} successfully verifies several
practical Michelson programs, including one that transfers money to an account
and that checks a digital signature.

    

### [[2108.13448] Machine Learning Methods for Management UAV Flocks -- a Survey](http://arxiv.org/abs/2108.13448)


  The development of unmanned aerial vehicles (UAVs) has been gaining momentum
in recent years owing to technological advances and a significant reduction in
their cost. UAV technology can be used in a wide range of domains, including
communication, agriculture, security, and transportation. It may be useful to
group the UAVs into clusters/flocks in certain domains, and various challenges
associated with UAV usage can be alleviated by clustering. Several
computational challenges arise in UAV flock management, which can be solved by
using machine learning (ML) methods. In this survey, we describe the basic
terms relating to UAVS and modern ML methods, and we provide an overview of
related tutorials and surveys. We subsequently consider the different
challenges that appear in UAV flocks. For each issue, we survey several machine
learning-based methods that have been suggested in the literature to handle the
associated challenges. Thereafter, we describe various open issues in which ML
can be applied to solve the different challenges of flocks, and we suggest
means of using ML methods for this purpose. This comprehensive review may be
useful for both researchers and developers in providing a wide view of various
aspects of state-of-the-art ML technologies that are applicable to flock
management.

    

### [[2108.13560] CWmin Estimation and Collision Identification in Wi-Fi Systems](http://arxiv.org/abs/2108.13560)


  Wi-Fi networks are susceptible to aggressive behavior caused by selfish or
malicious devices that reduce their minimum contention window size (CWmin) to
below the standard CWmin. In this paper, we propose a scheme called Minimum
Contention Window Estimation (CWE) to detect aggressive stations with low
CWmin's, where the AP estimates the CWmin value of all stations transmitting
uplink by monitoring their backoff values over a period of time and keeping
track of the idle time each station spends during backoff. To correctly
estimate each backoff value, we present a cross-correlation-based technique
that uses the frequency offset between the AP and each station to identify
stations involved in uplink collisions. The AP constructs empirical
distributions for the monitored backoff values and compares them with a set of
nominal PMF's, created via Markov analysis of the DCF protocol to estimate
CWmin of various stations. After detecting the aggressive stations, the AP can
choose to stop serving those stations. Simulation results show that the
accuracy of our collision detection technique is 96%, 94%, and 88% when there
are 3, 6, and 9 stations in the WLAN, respectively. For the former WLAN
settings, the estimation accuracy of CWE scheme is 100%, 98.81%, and 96.3%,
respectively.

    

### [[2108.13760] A Hierarchical Stitching Algorithm for Coded Compressed Sensing](http://arxiv.org/abs/2108.13760)


  Recently, a novel coded compressed sensing (CCS) approach was proposed in [1]
for dealing with the scalability problem for large sensing matrices in massive
machine-type communications. The approach is to divide the compressed sensing
(CS) problem into smaller CS sub-problems. However, such an approach requires
stitching the results from the sub-problems to recover the result in the
original CS problem. For this stitching problem, we propose a hierarchical
stitching algorithm that is easier to implement in hardware for parallelization
than the tree coding algorithm in [1]. For our algorithm, we also derive an
upper bound on the probability of recovery errors.

    

### [[2108.13801] Optimal Latency-Oriented Scheduling in Parallel Queuing Systems](http://arxiv.org/abs/2108.13801)


  Today, more and more interactive applications, such as augmented/virtual
reality, haptic Internet, and Industrial Internet of Things, require
communication services with guaranteed end-to-end latency limits, which are
difficult to provide over shared communication networks, particularly in the
presence of wireless links. Robustness against disturbances affecting
individual links can be obtained by coding the information flow in multiple
streams to be forwarded across parallel transmission links. This approach,
however, requires coding and scheduling algorithms that can adapt to the state
of links to take full advantage of path diversity and avoid self-induced
congestion on some links. To gain some fundamental insights on this challenging
problem, in this paper we resort to Markov Decision Process (MDP) theory and
abstract the parallel paths as independent queuing systems, whose arrival
processes are managed by a common controller that determines the amount of
redundancy to be applied to the source messages and the number of (coded)
packets to be sent to each queue. The objective is to find the joint coding and
scheduling policy that maximizes a certain utility function, e.g., the fraction
of source blocks delivered to the destination within a predetermined deadline,
despite the variability of the individual connections. We find the optimal
redundancy and scheduling strategies by using policy iteration methods. We then
analyze the optimal policy in a series of scenarios, highlighting its most
important aspects and analyzing ways to improve existing heuristics from the
literature.

    

### [[2108.13909] More WiFi for Everyone: Increasing Spectral Efficiency in WiFi6 Networks using OBSS/PD Mechanism](http://arxiv.org/abs/2108.13909)


  This study aims to enhance spatial reuse by using the new features of IEEE
802.11ax WLANs. Since the wireless medium is a shared medium and there may be
multiple basic service sets (BSS) in the same vicinity, BSSs may overlap, and
interference occurs. In this situation, BSSs cannot transmit simultaneously due
to the exposed node problem. The IEEE 802.11ax standard has a couple of
mechanisms to resolve these spectral efficiency problems. One of the most
effective mechanisms that address these problems is the overlapping BSS
preamble detection (OBSS/PD) mechanism. OBSS/PD mechanism uses the color
mechanism to distinguish OBSS signals. By using a signal threshold, the
mechanism can ignore some of the signals, which cause interference. In this
paper, we propose a rate-adaptive dynamic OBSS/PD threshold algorithm that
tracks the changes in transmission rate and dynamically adjusts the threshold
step by step considering the changes.

    

### [[2108.13923] TrackerSift: Untangling Mixed Tracking and Functional Web Resources](http://arxiv.org/abs/2108.13923)


  Trackers typically circumvent filter lists used by privacy-enhancing content
blocking tools by changing the domains or URLs of their resources. Filter list
maintainers painstakingly attempt to keep up in the ensuing arms race by
frequently updating the filter lists. Trackers have recently started to mix
tracking and functional resources, putting content blockers in a bind: risk
breaking legitimate functionality if they act and risk missing privacy-invasive
advertising and tracking if they do not. In this paper, we conduct a
large-scale measurement study of such mixed (i.e., both tracking and
functional) resources on 100K websites. We propose TRACKERSIFT, an approach
that progressively classifies and untangles mixed web resources at multiple
granularities of analysis (domain, hostname, script, and method). Using
TRACKERSIFT, we find that 83% of the domains can be separated as tracking or
functional, and the remaining 17% (11.8K) domains are classified as mixed. For
the mixed domains, 52% of the hostnames can be separated, and the remaining 48%
(12.3K) hostnames are classified as mixed. For the mixed hostnames, 94% of the
javascript snippets can be separated, and the remaining 6%(21.1K) scripts are
classified as mixed. For the mixed scripts,91% of the JavaScript methods can be
separated, and the remaining 9% (5.5K) methods are classified as mixed.
Overall, TRACKERSIFT is able to attribute 98% of all requests to tracking or
functional resources at the finest level of granularity. Our analysis shows
that mixed resources at different granularities are typically served from CDNs
or as inlined and bundled scripts. Our results highlight opportunities for
fine-grained content blocking to remove mixed resources without breaking
legitimate functionality.

    

### [[2108.13949] Latency-Redundancy Tradeoff in Distributed Read-Write Systems](http://arxiv.org/abs/2108.13949)


  Data is replicated and stored redundantly over multiple servers for
availability in distributed databases. We focus on databases with frequent
reads and writes, where both read and write latencies are important. This is in
contrast to databases designed primarily for either read or write applications.
Redundancy has contrasting effects on read and write latency. Read latency can
be reduced by potential parallel access from multiple servers, whereas write
latency increases as a larger number of replicas have to be updated. We
quantify this tradeoff between read and write latency as a function of
redundancy, and provide a closed-form approximation when the request arrival is
Poisson and the service is memoryless. We empirically show that this
approximation is tight across all ranges of system parameters. Thus, we provide
guidelines for redundancy selection in distributed databases.

    

### [[2102.06254] Securing RPL using Network Coding: The Chained Secure Mode (CSM)](http://arxiv.org/abs/2102.06254)


  As the de facto routing protocol for many Internet of Things (IoT) networks
nowadays, and to assure the confidentiality and integrity of its control
messages, the Routing Protocol for Low Power and Lossy Networks (RPL)
incorporates three modes of security: the Unsecured Mode (UM), Preinstalled
Secure Mode (PSM), and the Authenticated Secure Mode (ASM). While the PSM and
ASM are intended to protect against external routing attacks and some replay
attacks (through an optional replay protection mechanism), recent research
showed that RPL in PSM is still vulnerable to many routing attacks, both
internal and external. In this paper, we propose a novel secure mode for RPL,
the Chained Secure Mode (CSM), based on the concept of intraflow Network Coding
(NC). The CSM is designed to enhance RPL resilience and mitigation capability
against replay attacks while allowing the integration with external security
measures such as Intrusion Detection Systems (IDSs). The security and
performance of the proposed CSM were evaluated and compared against RPL in UM
and PSM (with and without the optional replay protection) under several routing
attacks: the Neighbor attack (NA), Wormhole (WH), and CloneID attack (CA),
using average packet delivery rate (PDR), End-to-End (E2E) latency, and power
consumption as metrics. It showed that CSM has better performance and more
enhanced security than both the UM and PSM with the replay protection, while
mitigating both the NA and WH attacks and significantly reducing the effect of
the CA in the investigated scenarios.

    

### [[2104.13184] Efficient channel charting via phase-insensitive distance computation](http://arxiv.org/abs/2104.13184)


  Channel charting is an unsupervised learning task whose objective is to
encode channels so that the obtained representation reflects the relative
spatial locations of the corresponding users. It has many potential
applications, ranging from user scheduling to proactive handover. In this
paper, a channel charting method is proposed, based on a distance measure
specifically designed to reduce the effect of small scale fading, which is an
irrelevant phenomenon with respect to the channel charting task. A nonlinear
dimensionality reduction technique aimed at preserving local distances (Isomap)
is then applied to actually get the channel representation. The approach is
empirically validated on realistic synthetic \new{multipath} MIMO channels,
achieving better results than previously proposed approaches, at a lower cost.

    

### [[2106.09754] Towards Real-Time Routing Optimization with Deep Reinforcement Learning: Open Challenges](http://arxiv.org/abs/2106.09754)


  The digital transformation is pushing the existing network technologies
towards new horizons, enabling new applications (e.g., vehicular networks). As
a result, the networking community has seen a noticeable increase in the
requirements of emerging network applications. One main open challenge is the
need to accommodate control systems to highly dynamic network scenarios.
Nowadays, existing network optimization technologies do not meet the needed
requirements to effectively operate in real time. Some of them are based on
hand-crafted heuristics with limited performance and adaptability, while some
technologies use optimizers which are often too time-consuming. Recent advances
in Deep Reinforcement Learning (DRL) have shown a dramatic improvement in
decision-making and automated control problems. Consequently, DRL represents a
promising technique to efficiently solve a variety of relevant network
optimization problems, such as online routing. In this paper, we explore the
use of state-of-the-art DRL technologies for real-time routing optimization and
outline some relevant open challenges to achieve production-ready DRL-based
solutions.

    

### [[2106.12805] Retrospective Interference Regeneration Schemes for Relay-Aided K-user MIMO Downlink Networks](http://arxiv.org/abs/2106.12805)


  To accommodate the explosive growth of the Internet-of-Things (IoT),
incorporating interference alignment (IA) into existing multiple access (MA)
schemes is under investigation. However, when it is applied in MIMO networks to
improve the system compacity, the incoming problem regarding information delay
arises which does not meet the requirement of low-latency. Therefore, in this
paper, we first propose a new metric, degree of delay (DoD), to quantify the
issue of information delay, and characterize DoD for three typical transmission
schemes, i.e., TDMA, beamforming based TDMA (BD-TDMA), and retrospective
interference alignment (RIA). By analyzing DoD in these schemes, its value
mainly depends on three factors, i.e., delay sensitive factor, size of data
set, and queueing delay slot. The first two reflect the relationship between
quality of service (QoS) and information delay sensitivity, and normalize time
cost for each symbol, respectively. These two factors are independent of the
transmission schemes, and thus we aim to reduce the queueing delay slot to
improve DoD. Herein, three novel joint IA schemes are proposed for MIMO
downlink networks with different number of users. That is, hybrid antenna array
based partial interference elimination and retrospective interference
regeneration scheme (HAA-PIE-RIR), HAA based improved PIE and RIR scheme
(HAA-IPIE-RIR), and HAA based cyclic interference elimination and RIR scheme
(HAA-CIE-RIR). Based on the first scheme, the second scheme extends the
application scenarios from $2$-user to $K$-user while causing heavy
computational burden. The third scheme relieves such computational burden,
though it has certain degree of freedom (DoF) loss due to insufficient
utilization of space resources.

    

### [[2108.13421] Recent advances for quantum classifiers](http://arxiv.org/abs/2108.13421)


  Machine learning has achieved dramatic success in a broad spectrum of
applications. Its interplay with quantum physics may lead to unprecedented
perspectives for both fundamental research and commercial applications, giving
rise to an emergent research frontier of quantum machine learning. Along this
line, quantum classifiers, which are quantum devices that aim to solve
classification problems in machine learning, have attracted tremendous
attention recently. In this review, we give a relatively comprehensive overview
for the studies of quantum classifiers, with a focus on recent advances. First,
we will review a number of quantum classification algorithms, including quantum
support vector machine, quantum kernel methods, quantum decision tree, and
quantum nearest neighbor algorithm. Then, we move on to introduce the
variational quantum classifiers, which are essentially variational quantum
circuits for classifications. We will review different architectures for
constructing variational quantum classifiers and introduce the barren plateau
problem, where the training of quantum classifiers might be hindered by the
exponentially vanishing gradient. In addition, the vulnerability aspect of
quantum classifiers in the setting of adversarial learning and the recent
experimental progress on different quantum classifiers will also be discussed.

    

### [[2108.13446] Benchmarking the Accuracy and Robustness of Feedback Alignment Algorithms](http://arxiv.org/abs/2108.13446)


  Backpropagation is the default algorithm for training deep neural networks
due to its simplicity, efficiency and high convergence rate. However, its
requirements make it impossible to be implemented in a human brain. In recent
years, more biologically plausible learning methods have been proposed. Some of
these methods can match backpropagation accuracy, and simultaneously provide
other extra benefits such as faster training on specialized hardware (e.g.,
ASICs) or higher robustness against adversarial attacks. While the interest in
the field is growing, there is a necessity for open-source libraries and
toolkits to foster research and benchmark algorithms. In this paper, we present
BioTorch, a software framework to create, train, and benchmark biologically
motivated neural networks. In addition, we investigate the performance of
several feedback alignment methods proposed in the literature, thereby
unveiling the importance of the forward and backward weight initialization and
optimizer choice. Finally, we provide a novel robustness study of these methods
against state-of-the-art white and black-box adversarial attacks.

    

### [[2108.13461] Time Series Prediction using Deep Learning Methods in Healthcare](http://arxiv.org/abs/2108.13461)


  Traditional machine learning methods face two main challenges in dealing with
healthcare predictive analytics tasks. First, the high-dimensional nature of
healthcare data needs labor-intensive and time-consuming processes to select an
appropriate set of features for each new task. Secondly, these methods depend
on feature engineering to capture the sequential nature of patient data, which
may not adequately leverage the temporal patterns of the medical events and
their dependencies. Recent deep learning methods have shown promising
performance for various healthcare prediction tasks by addressing the
high-dimensional and temporal challenges of medical data. These methods can
learn useful representations of key factors (e.g., medical concepts or
patients) and their interactions from high-dimensional raw (or
minimally-processed) healthcare data. In this paper we systemically reviewed
studies focused on using deep learning as the prediction model to leverage
patient time series data for a healthcare prediction task from methodological
perspective. To identify relevant studies, MEDLINE, IEEE, Scopus and ACM
digital library were searched for studies published up to February 7th 2021. We
found that researchers have contributed to deep time series prediction
literature in ten research streams: deep learning models, missing value
handling, irregularity handling, patient representation, static data inclusion,
attention mechanisms, interpretation, incorporating medical ontologies,
learning strategies, and scalability. This study summarizes research insights
from these literature streams, identifies several critical research gaps, and
suggests future research opportunities for deep learning in patient time series
data.

    

### [[2108.13465] Full-Cycle Energy Consumption Benchmark for Low-Carbon Computer Vision](http://arxiv.org/abs/2108.13465)


  The energy consumption of deep learning models is increasing at a
breathtaking rate, which raises concerns due to potential negative effects on
carbon neutrality in the context of global warming and climate change. With the
progress of efficient deep learning techniques, e.g., model compression,
researchers can obtain efficient models with fewer parameters and smaller
latency. However, most of the existing efficient deep learning methods do not
explicitly consider energy consumption as a key performance indicator.
Furthermore, existing methods mostly focus on the inference costs of the
resulting efficient models, but neglect the notable energy consumption
throughout the entire life cycle of the algorithm. In this paper, we present
the first large-scale energy consumption benchmark for efficient computer
vision models, where a new metric is proposed to explicitly evaluate the
full-cycle energy consumption under different model usage intensity. The
benchmark can provide insights for low carbon emission when selecting efficient
deep learning algorithms in different model usage scenarios.

    

### [[2108.13475] An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction](http://arxiv.org/abs/2108.13475)


  Industrial recommender systems are frequently tasked with approximating
probabilities for multiple, often closely related, user actions. For example,
predicting if a user will click on an advertisement and if they will then
purchase the advertised product. The conceptual similarity between these tasks
has promoted the use of multi-task learning: a class of algorithms that aim to
bring positive inductive transfer from related tasks. Here, we empirically
evaluate multi-task learning approaches with neural networks for an online
advertising task. Specifically, we consider approximating the probability of
post-click conversion events (installs) (CVR) for mobile app advertising on a
large-scale advertising platform, using the related click events (CTR) as an
auxiliary task. We use an ablation approach to systematically study recent
approaches that incorporate both multitask learning and "entire space modeling"
which train the CVR on all logged examples rather than learning a conditional
likelihood of conversion given clicked. Based on these results we show that
several different approaches result in similar levels of positive transfer from
the data-abundant CTR task to the CVR task and offer some insight into how the
multi-task design choices address the two primary problems affecting the CVR
task: data sparsity and data bias. Our findings add to the growing body of
evidence suggesting that standard multi-task learning is a sensible approach to
modelling related events in real-world large-scale applications and suggest the
specific multitask approach can be guided by ease of implementation in an
existing system.

    

### [[2108.13493] Semi-Supervised Exaggeration Detection of Health Science Press Releases](http://arxiv.org/abs/2108.13493)


  Public trust in science depends on honest and factual communication of
scientific papers. However, recent studies have demonstrated a tendency of news
media to misrepresent scientific papers by exaggerating their findings. Given
this, we present a formalization of and study into the problem of exaggeration
detection in science communication. While there are an abundance of scientific
papers and popular media articles written about them, very rarely do the
articles include a direct link to the original paper, making data collection
challenging. We address this by curating a set of labeled press
release/abstract pairs from existing expert annotated studies on exaggeration
in press releases of scientific papers suitable for benchmarking the
performance of machine learning models on the task. Using limited data from
this and previous studies on exaggeration detection in science, we introduce
MT-PET, a multi-task version of Pattern Exploiting Training (PET), which
leverages knowledge from complementary cloze-style QA tasks to improve few-shot
learning. We demonstrate that MT-PET outperforms PET and supervised learning
both when data is limited, as well as when there is an abundance of data for
the main task.

    

### [[2108.13509] An FEA surrogate model with Boundary Oriented Graph Embedding approach](http://arxiv.org/abs/2108.13509)


  In this work, we present a Boundary Oriented Graph Embedding (BOGE) approach
for the Graph Neural Network (GNN) to serve as a general surrogate model for
regressing physical fields and solving boundary value problems. Providing
shortcuts for both boundary elements and local neighbor elements, the BOGE
approach can embed structured mesh elements into the graph and performs an
efficient regression on large-scale triangular-mesh-based FEA results, which
cannot be realized by other machine-learning-based surrogate methods. Focusing
on the cantilever beam problem, our BOGE approach cannot only fit the
distribution of stress fields but also regresses the topological optimization
results, which show its potential of realizing abstract decision-making design
process. The BOGE approach with 3-layer DeepGCN model \textcolor{blue}{achieves
the regression with MSE of 0.011706 (2.41\% MAPE) for stress field prediction
and 0.002735 MSE (with 1.58\% elements having error larger than 0.01) for
topological optimization.} The overall concept of the BOGE approach paves the
way for a general and efficient deep-learning-based FEA simulator that will
benefit both industry and design-related areas.

    

### [[2108.13517] A Convolutional Neural Network-based Approach to Field Reconstruction](http://arxiv.org/abs/2108.13517)


  This work has been submitted to the IEEE for possible publication. Copyright
may be transferred without notice, after which this version may no longer be
accessible.
In many applications, the spatial distribution of a field needs to be
carefully monitored to detect spikes, discontinuities or dangerous
heterogeneities, but invasive monitoring approaches cannot be used. Besides,
technical specifications about the process might not be available by preventing
the adoption of an accurate model of the system. In this work, a
physics-informed, data-driven algorithm that allows addressing these
requirements is presented. The approach is based on the implementation of a
boundary element method (BEM)-scheme within a convolutional neural network.
Thanks to the capability of representing any continuous mathematical function
with a reduced number of parameters, the network allows predicting the field
value in any point of the domain, given the boundary conditions and few
measurements within the domain. The proposed approach was applied to
reconstruct a field described by the Helmholtz equation over a
three-dimensional domain. A sensitivity analysis was also performed by
investigating different physical conditions and different network
configurations. Since the only assumption is the applicability of BEM, the
current approach can be applied to the monitoring of a wide range of processes,
from the localization of the source of pollutant within a water reservoir to
the monitoring of the neutron flux in a nuclear reactor.

    

### [[2108.13518] DoWhy: Addressing Challenges in Expressing and Validating Causal Assumptions](http://arxiv.org/abs/2108.13518)


  Estimation of causal effects involves crucial assumptions about the
data-generating process, such as directionality of effect, presence of
instrumental variables or mediators, and whether all relevant confounders are
observed. Violation of any of these assumptions leads to significant error in
the effect estimate. However, unlike cross-validation for predictive models,
there is no global validator method for a causal estimate. As a result,
expressing different causal assumptions formally and validating them (to the
extent possible) becomes critical for any analysis. We present DoWhy, a
framework that allows explicit declaration of assumptions through a causal
graph and provides multiple validation tests to check a subset of these
assumptions. Our experience with DoWhy highlights a number of open questions
for future research: developing new ways beyond causal graphs to express
assumptions, the role of causal discovery in learning relevant parts of the
graph, and developing validation tests that can better detect errors, both for
average and conditional treatment effects. DoWhy is available at
this https URL.

    

### [[2108.13525] Identifying optimal cycles in quantum thermal machines with reinforcement-learning](http://arxiv.org/abs/2108.13525)


  The optimal control of open quantum systems is a challenging task but has a
key role in improving existing quantum information processing technologies. We
introduce a general framework based on Reinforcement Learning to discover
optimal thermodynamic cycles that maximize the power of out-of-equilibrium
quantum heat engines and refrigerators. We apply our method, based on the soft
actor-critic algorithm, to three systems: a benchmark two-level system heat
engine, where we find the optimal known cycle; an experimentally realistic
refrigerator based on a superconducting qubit that generates coherence, where
we find a non-intuitive control sequence that outperform previous cycles
proposed in literature; a heat engine based on a quantum harmonic oscillator,
where we find a cycle with an elaborate structure that outperforms the
optimized Otto cycle. We then evaluate the corresponding efficiency at maximum
power.

    

### [[2108.13551] Regularizing (Stabilizing) Deep Learning Based Reconstruction Algorithms](http://arxiv.org/abs/2108.13551)


  It's well-known that inverse problems are ill-posed and to solve them
meaningfully one has to employ regularization methods. Traditionally, popular
regularization methods have been the penalized Variational approaches. In
recent years, the classical regularized-reconstruction approaches have been
outclassed by the (deep-learning-based) learned reconstruction algorithms.
However, unlike the traditional regularization methods, the theoretical
underpinnings, such as stability and regularization, have been insufficient for
such learned reconstruction algorithms. Hence, the results obtained from such
algorithms, though empirically outstanding, can't always be completely trusted,
as they may contain certain instabilities or (hallucinated) features arising
from the learned process. In fact, it has been shown that such learning
algorithms are very susceptible to small (adversarial) noises in the data and
can lead to severe instabilities in the recovered solution, which can be quite
different than the inherent instabilities of the ill-posed (inverse) problem.
Whereas, the classical regularization methods can handle such (adversarial)
noises very well and can produce stable recovery. Here, we try to present
certain regularization methods to stabilize such (unstable) learned
reconstruction methods and recover a regularized solution, even in the presence
of adversarial noises. For this, we need to extend the classical notion of
regularization and incorporate it in the learned reconstruction algorithms. We
also present some regularization techniques to regularize two of the most
popular learning reconstruction algorithms, the Learned Post-Processing
Reconstruction and the Learned Unrolling Reconstruction.

    

### [[2108.13555] Adaptive Label Smoothing To Regularize Large-Scale Graph Training](http://arxiv.org/abs/2108.13555)


  Graph neural networks (GNNs), which learn the node representations by
recursively aggregating information from its neighbors, have become a
predominant computational tool in many domains. To handle large-scale graphs,
most of the existing methods partition the input graph into multiple sub-graphs
(e.g., through node clustering) and apply batch training to save memory cost.
However, such batch training will lead to label bias within each batch, and
then result in over-confidence in model predictions. Since the connected nodes
with positively related labels tend to be assigned together, the traditional
cross-entropy minimization process will attend on the predictions of biased
classes in the batch, and may intensify the overfitting issue. To overcome the
label bias problem, we propose the adaptive label smoothing (ALS) method to
replace the one-hot hard labels with smoothed ones, which learns to allocate
label confidences from the biased classes to the others. Specifically, ALS
propagates node labels to aggregate the neighborhood label distribution in a
pre-processing step, and then updates the optimal smoothed labels online to
adapt to specific graph structure. Experiments on the real-world datasets
demonstrate that ALS can be generally applied to the main scalable learning
frameworks to calibrate the biased labels and improve generalization
performances.

    

### [[2108.13570] Fast Multi-label Learning](http://arxiv.org/abs/2108.13570)


  Embedding approaches have become one of the most pervasive techniques for
multi-label classification. However, the training process of embedding methods
usually involves a complex quadratic or semidefinite programming problem, or
the model may even involve an NP-hard problem. Thus, such methods are
prohibitive on large-scale applications. More importantly, much of the
literature has already shown that the binary relevance (BR) method is usually
good enough for some applications. Unfortunately, BR runs slowly due to its
linear dependence on the size of the input data. The goal of this paper is to
provide a simple method, yet with provable guarantees, which can achieve
competitive performance without a complex training process. To achieve our
goal, we provide a simple stochastic sketch strategy for multi-label
classification and present theoretical results from both algorithmic and
statistical learning perspectives. Our comprehensive empirical studies
corroborate our theoretical findings and demonstrate the superiority of the
proposed methods.

    

### [[2108.13577] Rapidly and accurately estimating brain strain and strain rate across head impact types with transfer learning and data fusion](http://arxiv.org/abs/2108.13577)


  Brain strain and strain rate are effective in predicting traumatic brain
injury (TBI) caused by head impacts. However, state-of-the-art finite element
modeling (FEM) demands considerable computational time in the computation,
limiting its application in real-time TBI risk monitoring. To accelerate,
machine learning head models (MLHMs) were developed, and the model accuracy was
found to decrease when the training/test datasets were from different head
impacts types. However, the size of dataset for specific impact types may not
be enough for model training. To address the computational cost of FEM, the
limited strain rate prediction, and the generalizability of MLHMs to on-field
datasets, we propose data fusion and transfer learning to develop a series of
MLHMs to predict the maximum principal strain (MPS) and maximum principal
strain rate (MPSR). We trained and tested the MLHMs on 13,623 head impacts from
simulations, American football, mixed martial arts, car crash, and compared
against the models trained on only simulations or only on-field impacts. The
MLHMs developed with transfer learning are significantly more accurate in
estimating MPS and MPSR than other models, with a mean absolute error (MAE)
smaller than 0.03 in predicting MPS and smaller than 7 (1/s) in predicting MPSR
on all impact datasets. The MLHMs can be applied to various head impact types
for rapidly and accurately calculating brain strain and strain rate. Besides
the clinical applications in real-time brain strain and strain rate monitoring,
this model helps researchers estimate the brain strain and strain rate caused
by head impacts more efficiently than FEM.

    

### [[2108.13581] DoGR: Disaggregated Gaussian Regression for Reproducible Analysis of Heterogeneous Data](http://arxiv.org/abs/2108.13581)


  Quantitative analysis of large-scale data is often complicated by the
presence of diverse subgroups, which reduce the accuracy of inferences they
make on held-out data. To address the challenge of heterogeneous data analysis,
we introduce DoGR, a method that discovers latent confounders by simultaneously
partitioning the data into overlapping clusters (disaggregation) and modeling
the behavior within them (regression). When applied to real-world data, our
method discovers meaningful clusters and their characteristic behaviors, thus
giving insight into group differences and their impact on the outcome of
interest. By accounting for latent confounders, our framework facilitates
exploratory analysis of noisy, heterogeneous data and can be used to learn
predictive models that better generalize to new data. We provide the code to
enable others to use DoGR within their data analytic workflows.

    

### [[2108.13583] A New Approach to Multilinear Dynamical Systems and Control](http://arxiv.org/abs/2108.13583)


  The current paper presents a new approach to multilinear dynamical systems
analysis and control. The approach is based upon recent developments in tensor
decompositions and a newly defined algebra of circulants. In particular, it is
shown that under the right tensor multiplication operator, a third order tensor
can be written as a product of third order tensors that is analogous to a
traditional matrix eigenvalue decomposition where the "eigenvectors" become
eigenmatrices and the "eigenvalues" become eigen-tuples. This new development
allows for a proper tensor eigenvalue decomposition to be defined and has
natural extension to linear systems theory through a
\textit{tensor-exponential}. Through this framework we extend many of
traditional techniques used in linear system theory to their multilinear
counterpart.

    

### [[2108.13597] Self-balanced Learning For Domain Generalization](http://arxiv.org/abs/2108.13597)


  Domain generalization aims to learn a prediction model on multi-domain source
data such that the model can generalize to a target domain with unknown
statistics. Most existing approaches have been developed under the assumption
that the source data is well-balanced in terms of both domain and class.
However, real-world training data collected with different composition biases
often exhibits severe distribution gaps for domain and class, leading to
substantial performance degradation. In this paper, we propose a self-balanced
domain generalization framework that adaptively learns the weights of losses to
alleviate the bias caused by different distributions of the multi-domain source
data. The self-balanced scheme is based on an auxiliary reweighting network
that iteratively updates the weight of loss conditioned on the domain and class
information by leveraging balanced meta data. Experimental results demonstrate
the effectiveness of our method overwhelming state-of-the-art works for domain
generalization.

    

### [[2108.13602] How Does Adversarial Fine-Tuning Benefit BERT?](http://arxiv.org/abs/2108.13602)


  Adversarial training (AT) is one of the most reliable methods for defending
against adversarial attacks in machine learning. Variants of this method have
been used as regularization mechanisms to achieve SOTA results on NLP
benchmarks, and they have been found to be useful for transfer learning and
continual learning. We search for the reasons for the effectiveness of AT by
contrasting vanilla and adversarially fine-tuned BERT models. We identify
partial preservation of BERT's syntactic abilities during fine-tuning as the
key to the success of AT. We observe that adversarially fine-tuned models
remain more faithful to BERT's language modeling behavior and are more
sensitive to the word order. As concrete examples of syntactic abilities, an
adversarially fine-tuned model could have an advantage of up to 38% on anaphora
agreement and up to 11% on dependency parsing. Our analysis demonstrates that
vanilla fine-tuning oversimplifies the sentence representation by focusing
heavily on one or a few label-indicative words. AT, however, moderates the
effect of these influential words and encourages representational diversity.
This allows for a more hierarchical representation of a sentence and leads to
the mitigation of BERT's loss of syntactic abilities.

    

### [[2108.13617] Segmentation Fault: A Cheap Defense Against Adversarial Machine Learning](http://arxiv.org/abs/2108.13617)


  Recently published attacks against deep neural networks (DNNs) have stressed
the importance of methodologies and tools to assess the security risks of using
this technology in critical systems. Efficient techniques for detecting
adversarial machine learning helps establishing trust and boost the adoption of
deep learning in sensitive and security systems. In this paper, we propose a
new technique for defending deep neural network classifiers, and convolutional
ones in particular. Our defense is cheap in the sense that it requires less
computation power despite a small cost to pay in terms of detection accuracy.
The work refers to a recently published technique called ML-LOO. We replace the
costly pixel by pixel leave-one-out approach of ML-LOO by adopting
coarse-grained leave-one-out. We evaluate and compare the efficiency of
different segmentation algorithms for this task. Our results show that a large
gain in efficiency is possible, even though penalized by a marginal decrease in
detection accuracy.

    

### [[2108.13620] Cross-Lingual Text Classification of Transliterated Hindi and Malayalam](http://arxiv.org/abs/2108.13620)


  Transliteration is very common on social media, but transliterated text is
not adequately handled by modern neural models for various NLP tasks. In this
work, we combine data augmentation approaches with a Teacher-Student training
scheme to address this issue in a cross-lingual transfer setting for
fine-tuning state-of-the-art pre-trained multilingual language models such as
mBERT and XLM-R. We evaluate our method on transliterated Hindi and Malayalam,
also introducing new datasets for benchmarking on real-world scenarios: one on
sentiment classification in transliterated Malayalam, and another on crisis
tweet classification in transliterated Hindi and Malayalam (related to the 2013
North India and 2018 Kerala floods). Our method yielded an average improvement
of +5.6% on mBERT and +4.7% on XLM-R in F1 scores over their strong baselines.

    

### [[2108.13624] Towards Out-Of-Distribution Generalization: A Survey](http://arxiv.org/abs/2108.13624)


  Classic machine learning methods are built on the $i.i.d.$ assumption that
training and testing data are independent and identically distributed. However,
in real scenarios, the $i.i.d.$ assumption can hardly be satisfied, rendering
the sharp drop of classic machine learning algorithms' performances under
distributional shifts, which indicates the significance of investigating the
Out-of-Distribution generalization problem. Out-of-Distribution (OOD)
generalization problem addresses the challenging setting where the testing
distribution is unknown and different from the training. This paper serves as
the first effort to systematically and comprehensively discuss the OOD
generalization problem, from the definition, methodology, evaluation to the
implications and future directions. Firstly, we provide the formal definition
of the OOD generalization problem. Secondly, existing methods are categorized
into three parts based on their positions in the whole learning pipeline,
namely unsupervised representation learning, supervised model learning and
optimization, and typical methods for each category are discussed in detail. We
then demonstrate the theoretical connections of different categories, and
introduce the commonly used datasets and evaluation metrics. Finally, we
summarize the whole literature and raise some future directions for OOD
generalization problem. The summary of OOD generalization methods reviewed in
this survey can be found at this http URL.

    

### [[2108.13628] Learning Optimal Prescriptive Trees from Observational Data](http://arxiv.org/abs/2108.13628)


  We consider the problem of learning an optimal prescriptive tree (i.e., a
personalized treatment assignment policy in the form of a binary tree) of
moderate depth, from observational data. This problem arises in numerous
socially important domains such as public health and personalized medicine,
where interpretable and data-driven interventions are sought based on data
gathered in deployment, through passive collection of data, rather than from
randomized trials. We propose a method for learning optimal prescriptive trees
using mixed-integer optimization (MIO) technology. We show that under mild
conditions our method is asymptotically exact in the sense that it converges to
an optimal out-of-sample treatment assignment policy as the number of
historical data samples tends to infinity. This sets us apart from existing
literature on the topic which either requires data to be randomized or imposes
stringent assumptions on the trees. Based on extensive computational
experiments on both synthetic and real data, we demonstrate that our asymptotic
guarantees translate to significant out-of-sample performance improvements even
in finite samples.

    

### [[2108.13637] When are Deep Networks really better than Random Forests at small sample sizes?](http://arxiv.org/abs/2108.13637)


  Random forests (RF) and deep networks (DN) are two of the most popular
machine learning methods in the current scientific literature and yield
differing levels of performance on different data modalities. We wish to
further explore and establish the conditions and domains in which each approach
excels, particularly in the context of sample size and feature dimension. To
address these issues, we tested the performance of these approaches across
tabular, image, and audio settings using varying model parameters and
architectures. Our focus is on datasets with at most 10,000 samples, which
represent a large fraction of scientific and biomedical datasets. In general,
we found RF to excel at tabular and structured data (image and audio) with
small sample sizes, whereas DN performed better on structured data with larger
sample sizes. Although we plan to continue updating this technical report in
the coming months, we believe the current preliminary results may be of
interest to others.

    

### [[2108.13643] Learning to Synthesize Programs as Interpretable and Generalizable Policies](http://arxiv.org/abs/2108.13643)


  Recently, deep reinforcement learning (DRL) methods have achieved impressive
performance on tasks in a variety of domains. However, neural network policies
produced with DRL methods are not human-interpretable and often have difficulty
generalizing to novel scenarios. To address these issues, prior works explore
learning programmatic policies that are more interpretable and structured for
generalization. Yet, these works either employ limited policy representations
(e.g. decision trees, state machines, or predefined program templates) or
require stronger supervision (e.g. input/output state pairs or expert
demonstrations). We present a framework that instead learns to synthesize a
program, which details the procedure to solve a task in a flexible and
expressive manner, solely from reward signals. To alleviate the difficulty of
learning to compose programs to induce the desired agent behavior from scratch,
we propose to first learn a program embedding space that continuously
parameterizes diverse behaviors in an unsupervised manner and then search over
the learned program embedding space to yield a program that maximizes the
return for a given task. Experimental results demonstrate that the proposed
framework not only learns to reliably synthesize task-solving programs but also
outperforms DRL and program synthesis baselines while producing interpretable
and more generalizable policies. We also justify the necessity of the proposed
two-stage learning scheme as well as analyze various methods for learning the
program embedding.

    

### [[2108.13650] Heterogeneous Graph Neural Network with Multi-view Representation Learning](http://arxiv.org/abs/2108.13650)


  Graph neural networks for heterogeneous graph embedding is to project nodes
into a low-dimensional space by exploring the heterogeneity and semantics of
the heterogeneous graph. However, on the one hand, most of existing
heterogeneous graph embedding methods either insufficiently model the local
structure under specific semantic, or neglect the heterogeneity when
aggregating information from it. On the other hand, representations from
multiple semantics are not comprehensively integrated to obtain versatile node
embeddings. To address the problem, we propose a Heterogeneous Graph Neural
Network with Multi-View Representation Learning (named MV-HetGNN) for
heterogeneous graph embedding by introducing the idea of multi-view
representation learning. The proposed model consists of node feature
transformation, view-specific ego graph encoding and auto multi-view fusion to
thoroughly learn complex structural and semantic information for generating
comprehensive node representations. Extensive experiments on three real-world
heterogeneous graph datasets show that the proposed MV-HetGNN model
consistently outperforms all the state-of-the-art GNN baselines in various
downstream tasks, e.g., node classification, node clustering, and link
prediction.

    

### [[2108.13669] Unit-Modulus Wireless Federated Learning Via Penalty Alternating Minimization](http://arxiv.org/abs/2108.13669)


  Wireless federated learning (FL) is an emerging machine learning paradigm
that trains a global parametric model from distributed datasets via wireless
communications. This paper proposes a unit-modulus wireless FL (UMWFL)
framework, which simultaneously uploads local model parameters and computes
global model parameters via optimized phase shifting. The proposed framework
avoids sophisticated baseband signal processing, leading to both low
communication delays and implementation costs. A training loss bound is derived
and a penalty alternating minimization (PAM) algorithm is proposed to minimize
the nonconvex nonsmooth loss bound. Experimental results in the Car Learning to
Act (CARLA) platform show that the proposed UMWFL framework with PAM algorithm
achieves smaller training losses and testing errors than those of the benchmark
scheme.

    

### [[2108.13672] Medical SANSformers: Training self-supervised transformers without attention for Electronic Medical Records](http://arxiv.org/abs/2108.13672)


  We leverage deep sequential models to tackle the problem of predicting
healthcare utilization for patients, which could help governments to better
allocate resources for future healthcare use. Specifically, we study the
problem of \textit{divergent subgroups}, wherein the outcome distribution in a
smaller subset of the population considerably deviates from that of the general
population. The traditional approach for building specialized models for
divergent subgroups could be problematic if the size of the subgroup is very
small (for example, rare diseases). To address this challenge, we first develop
a novel attention-free sequential model, SANSformers, instilled with inductive
biases suited for modeling clinical codes in electronic medical records. We
then design a task-specific self-supervision objective and demonstrate its
effectiveness, particularly in scarce data settings, by pre-training each model
on the entire health registry (with close to one million patients) before
fine-tuning for downstream tasks on the divergent subgroups. We compare the
novel SANSformer architecture with the LSTM and Transformer models using two
data sources and a multi-task learning objective that aids healthcare
utilization prediction. Empirically, the attention-free SANSformer models
perform consistently well across experiments, outperforming the baselines in
most cases by at least $\sim 10$\%. Furthermore, the self-supervised
pre-training boosts performance significantly throughout, for example by over
$\sim 50$\% (and as high as $800$\%) on $R^2$ score when predicting the number
of hospital visits.

    

### [[2108.13680] Learning Practically Feasible Policies for Online 3D Bin Packing](http://arxiv.org/abs/2108.13680)


  We tackle the Online 3D Bin Packing Problem, a challenging yet practically
useful variant of the classical Bin Packing Problem. In this problem, the items
are delivered to the agent without informing the full sequence information.
Agent must directly pack these items into the target bin stably without
changing their arrival order, and no further adjustment is permitted. Online
3D-BPP can be naturally formulated as Markov Decision Process (MDP). We adopt
deep reinforcement learning, in particular, the on-policy actor-critic
framework, to solve this MDP with constrained action space. To learn a
practically feasible packing policy, we propose three critical designs. First,
we propose an online analysis of packing stability based on a novel stacking
tree. It attains a high analysis accuracy while reducing the computational
complexity from $O(N^2)$ to $O(N \log N)$, making it especially suited for RL
training. Second, we propose a decoupled packing policy learning for different
dimensions of placement which enables high-resolution spatial discretization
and hence high packing precision. Third, we introduce a reward function that
dictates the robot to place items in a far-to-near order and therefore
simplifies the collision avoidance in movement planning of the robotic arm.
Furthermore, we provide a comprehensive discussion on several key implemental
issues. The extensive evaluation demonstrates that our learned policy
outperforms the state-of-the-art methods significantly and is practically
usable for real-world applications.

    

### [[2108.13696] Phy-Q: A Benchmark for Physical Reasoning](http://arxiv.org/abs/2108.13696)


  Humans are well-versed in reasoning about the behaviors of physical objects
when choosing actions to accomplish tasks, while it remains a major challenge
for AI. To facilitate research addressing this problem, we propose a new
benchmark that requires an agent to reason about physical scenarios and take an
action accordingly. Inspired by the physical knowledge acquired in infancy and
the capabilities required for robots to operate in real-world environments, we
identify 15 essential physical scenarios. For each scenario, we create a wide
variety of distinct task templates, and we ensure all the task templates within
the same scenario can be solved by using one specific physical rule. By having
such a design, we evaluate two distinct levels of generalization, namely the
local generalization and the broad generalization. We conduct an extensive
evaluation with human players, learning agents with varying input types and
architectures, and heuristic agents with different strategies. The benchmark
gives a Phy-Q (physical reasoning quotient) score that reflects the physical
reasoning ability of the agents. Our evaluation shows that 1) all agents fail
to reach human performance, and 2) learning agents, even with good local
generalization ability, struggle to learn the underlying physical reasoning
rules and fail to generalize broadly. We encourage the development of
intelligent agents with broad generalization abilities in physical domains.

    

### [[2108.13697] Attention-based Multi-Reference Learning for Image Super-Resolution](http://arxiv.org/abs/2108.13697)


  This paper proposes a novel Attention-based Multi-Reference Super-resolution
network (AMRSR) that, given a low-resolution image, learns to adaptively
transfer the most similar texture from multiple reference images to the
super-resolution output whilst maintaining spatial coherence. The use of
multiple reference images together with attention-based sampling is
demonstrated to achieve significantly improved performance over
state-of-the-art reference super-resolution approaches on multiple benchmark
datasets. Reference super-resolution approaches have recently been proposed to
overcome the ill-posed problem of image super-resolution by providing
additional information from a high-resolution reference image. Multi-reference
super-resolution extends this approach by providing a more diverse pool of
image features to overcome the inherent information deficit whilst maintaining
memory efficiency. A novel hierarchical attention-based sampling approach is
introduced to learn the similarity between low-resolution image features and
multiple reference images based on a perceptual loss. Ablation demonstrates the
contribution of both multi-reference and hierarchical attention-based sampling
to overall performance. Perceptual and quantitative ground-truth evaluation
demonstrates significant improvement in performance even when the reference
images deviate significantly from the target image. The project website can be
found at this https URL


### [[2108.13702] SemIE: Semantically-aware Image Extrapolation](http://arxiv.org/abs/2108.13702)


  We propose a semantically-aware novel paradigm to perform image extrapolation
that enables the addition of new object instances. All previous methods are
limited in their capability of extrapolation to merely extending the already
existing objects in the image. However, our proposed approach focuses not only
on (i) extending the already present objects but also on (ii) adding new
objects in the extended region based on the context. To this end, for a given
image, we first obtain an object segmentation map using a state-of-the-art
semantic segmentation method. The, thus, obtained segmentation map is fed into
a network to compute the extrapolated semantic segmentation and the
corresponding panoptic segmentation maps. The input image and the obtained
segmentation maps are further utilized to generate the final extrapolated
image. We conduct experiments on Cityscapes and ADE20K-bedroom datasets and
show that our method outperforms all baselines in terms of FID, and similarity
in object co-occurrence statistics.

    

### [[2108.13703] Evaluating the Robustness of Off-Policy Evaluation](http://arxiv.org/abs/2108.13703)


  Off-policy Evaluation (OPE), or offline evaluation in general, evaluates the
performance of hypothetical policies leveraging only offline log data. It is
particularly useful in applications where the online interaction involves high
stakes and expensive setting such as precision medicine and recommender
systems. Since many OPE estimators have been proposed and some of them have
hyperparameters to be tuned, there is an emerging challenge for practitioners
to select and tune OPE estimators for their specific application.
Unfortunately, identifying a reliable estimator from results reported in
research papers is often difficult because the current experimental procedure
evaluates and compares the estimators' performance on a narrow set of
hyperparameters and evaluation policies. Therefore, it is difficult to know
which estimator is safe and reliable to use. In this work, we develop
Interpretable Evaluation for Offline Evaluation (IEOE), an experimental
procedure to evaluate OPE estimators' robustness to changes in hyperparameters
and/or evaluation policies in an interpretable manner. Then, using the IEOE
procedure, we perform extensive evaluation of a wide variety of existing
estimators on Open Bandit Dataset, a large-scale public real-world dataset for
OPE. We demonstrate that our procedure can evaluate the estimators' robustness
to the hyperparamter choice, helping us avoid using unsafe estimators. Finally,
we apply IEOE to real-world e-commerce platform data and demonstrate how to use
our protocol in practice.

    

### [[2108.13732] Deep Learning on Edge TPUs](http://arxiv.org/abs/2108.13732)


  Computing at the edge is important in remote settings, however, conventional
hardware is not optimized for utilizing deep neural networks. The Google Edge
TPU is an emerging hardware accelerator that is cost, power and speed
efficient, and is available for prototyping and production purposes. Here, I
review the Edge TPU platform, the tasks that have been accomplished using the
Edge TPU, and which steps are necessary to deploy a model to the Edge TPU
hardware. The Edge TPU is not only capable of tackling common computer vision
tasks, but also surpasses other hardware accelerators, especially when the
entire model can be deployed to the Edge TPU. Co-embedding the Edge TPU in
cameras allows a seamless analysis of primary data. In summary, the Edge TPU is
a maturing system that has proven its usability across multiple tasks.

    

### [[2108.13739] Super-Resolution Appearance Transfer for 4D Human Performances](http://arxiv.org/abs/2108.13739)


  A common problem in the 4D reconstruction of people from multi-view video is
the quality of the captured dynamic texture appearance which depends on both
the camera resolution and capture volume. Typically the requirement to frame
cameras to capture the volume of a dynamic performance ($>50m^3$) results in
the person occupying only a small proportion $<$ 10% of the field of view. Even
with ultra high-definition 4k video acquisition this results in sampling the
person at less-than standard definition 0.5k video resolution resulting in
low-quality rendering. In this paper we propose a solution to this problem
through super-resolution appearance transfer from a static high-resolution
appearance capture rig using digital stills cameras ($> 8k$) to capture the
person in a small volume ($<8m^3$). A pipeline is proposed for super-resolution
appearance transfer from high-resolution static capture to dynamic video
performance capture to produce super-resolution dynamic textures. This
addresses two key problems: colour mapping between different camera systems;
and dynamic texture map super-resolution using a learnt model. Comparative
evaluation demonstrates a significant qualitative and quantitative improvement
in rendering the 4D performance capture with super-resolution dynamic texture
appearance. The proposed approach reproduces the high-resolution detail of the
static capture whilst maintaining the appearance dynamics of the captured
video.

    

### [[2108.13753] Disentanglement Analysis with Partial Information Decomposition](http://arxiv.org/abs/2108.13753)


  Given data generated from multiple factors of variation that cooperatively
transform their appearance, disentangled representations aim at reversing the
process by mapping data to multiple random variables that individually capture
distinct generative factors. As the concept is intuitive but abstract, one
needs to quantify it with disentanglement metrics to evaluate and compare the
quality of disentangled representations between different models. Current
disentanglement metrics are designed to measure the concentration, e.g.,
absolute deviation, variance, or entropy, of each variable conditioned by each
generative factor, optionally offset by the concentration of its marginal
distribution, and compare it among different variables. When representations
consist of more than two variables, such metrics may fail to detect the
interplay between them as they only measure pairwise interactions. In this
work, we use the Partial Information Decomposition framework to evaluate
information sharing between more than two variables, and build a framework,
including a new disentanglement metric, for analyzing how the representations
encode the generative factors distinctly, redundantly, and cooperatively. We
establish an experimental protocol to assess how each metric evaluates
increasingly entangled representations and confirm through artificial and
realistic settings that the proposed metric correctly responds to entanglement.
Our results are expected to promote information theoretic understanding of
disentanglement and lead to further development of metrics as well as learning
methods.

    

### [[2108.13797] Sample Efficient Detection and Classification of Adversarial Attacks via Self-Supervised Embeddings](http://arxiv.org/abs/2108.13797)


  Adversarial robustness of deep models is pivotal in ensuring safe deployment
in real world settings, but most modern defenses have narrow scope and
expensive costs. In this paper, we propose a self-supervised method to detect
adversarial attacks and classify them to their respective threat models, based
on a linear model operating on the embeddings from a pre-trained
self-supervised encoder. We use a SimCLR encoder in our experiments, since we
show the SimCLR embedding distance is a good proxy for human perceptibility,
enabling it to encapsulate many threat models at once. We call our method
SimCat since it uses SimCLR encoder to catch and categorize various types of
adversarial attacks, including L_p and non-L_p evasion attacks, as well as data
poisonings. The simple nature of a linear classifier makes our method efficient
in both time and sample complexity. For example, on SVHN, using only five pairs
of clean and adversarial examples computed with a PGD-L_inf attack, SimCat's
detection accuracy is over 85%. Moreover, on ImageNet, using only 25 examples
from each threat model, SimCat can classify eight different attack types such
as PGD-L_2, PGD-L_inf, CW-L_2, PPGD, LPA, StAdv, ReColor, and JPEG-L_inf, with
over 40% accuracy. On STL10 data, we apply SimCat as a defense against
poisoning attacks, such as BP, CP, FC, CLBD, HTBD, halving the success rate
while using only twenty total poisons for training. We find that the detectors
generalize well to unseen threat models. Lastly, we investigate the performance
of our detection method under adaptive attacks and further boost its robustness
against such attacks via adversarial training.

    

### [[2108.13807] Identifying Ransomware Actors in the Bitcoin Network](http://arxiv.org/abs/2108.13807)


  Due to the pseudo-anonymity of the Bitcoin network, users can hide behind
their bitcoin addresses that can be generated in unlimited quantity, on the
fly, without any formal links between them. Thus, it is being used for payment
transfer by the actors involved in ransomware and other illegal activities. The
other activity we consider is related to gambling since gambling is often used
for transferring illegal funds. The question addressed here is that given
temporally limited graphs of Bitcoin transactions, to what extent can one
identify common patterns associated with these fraudulent activities and apply
them to find other ransomware actors. The problem is rather complex, given that
thousands of addresses can belong to the same actor without any obvious links
between them and any common pattern of behavior. The main contribution of this
paper is to introduce and apply new algorithms for local clustering and
supervised graph machine learning for identifying malicious actors. We show
that very local subgraphs of the known such actors are sufficient to
differentiate between ransomware, random and gambling actors with 85%
prediction accuracy on the test data set.

    

### [[2108.13810] Max-Utility Based Arm Selection Strategy For Sequential Query Recommendations](http://arxiv.org/abs/2108.13810)


  We consider the query recommendation problem in closed loop interactive
learning settings like online information gathering and exploratory analytics.
The problem can be naturally modelled using the Multi-Armed Bandits (MAB)
framework with countably many arms. The standard MAB algorithms for countably
many arms begin with selecting a random set of candidate arms and then applying
standard MAB algorithms, e.g., UCB, on this candidate set downstream. We show
that such a selection strategy often results in higher cumulative regret and to
this end, we propose a selection strategy based on the maximum utility of the
arms. We show that in tasks like online information gathering, where sequential
query recommendations are employed, the sequences of queries are correlated and
the number of potentially optimal queries can be reduced to a manageable size
by selecting queries with maximum utility with respect to the currently
executing query. Our experimental results using a recent real online literature
discovery service log file demonstrate that the proposed arm selection strategy
improves the cumulative regret substantially with respect to the
state-of-the-art baseline algorithms. % and commonly used random selection
strategy for a variety of contextual multi-armed bandit algorithms. Our data
model and source code are available at
~\url{https://anonymous.4open.science/r/0e5ad6b7-ac02-4577-9212-c9d505d3dbdb/}.

    

### [[2108.13822] Chi-square Loss for Softmax: an Echo of Neural Network Structure](http://arxiv.org/abs/2108.13822)


  Softmax working with cross-entropy is widely used in classification, which
evaluates the similarity between two discrete distribution columns (predictions
and true labels). Inspired by chi-square test, we designed a new loss function
called chi-square loss, which is also works for Softmax. Chi-square loss has a
statistical background. We proved that it is unbiased in optimization, and
clarified its using conditions (its formula determines that it must work with
label smoothing). In addition, we studied the sample distribution of this loss
function by visualization and found that the distribution is related to the
neural network structure, which is distinct compared to cross-entropy. In the
past, the influence of structure was often ignored when visualizing. Chi-square
loss can notice changes in neural network structure because it is very strict,
and we explained the reason for this strictness. We also studied the influence
of label smoothing and discussed the relationship between label smoothing and
training accuracy and stability. Since the chi-square loss is very strict, the
performance will degrade when dealing samples of very many classes.

    

### [[2108.13823] Temporal Deep Learning Architecture for Prediction of COVID-19 Cases in India](http://arxiv.org/abs/2108.13823)


  To combat the recent coronavirus disease 2019 (COVID-19), academician and
clinician are in search of new approaches to predict the COVID-19 outbreak
dynamic trends that may slow down or stop the pandemic. Epidemiological models
like Susceptible-Infected-Recovered (SIR) and its variants are helpful to
understand the dynamics trend of pandemic that may be used in decision making
to optimize possible controls from the infectious disease. But these
epidemiological models based on mathematical assumptions may not predict the
real pandemic situation. Recently the new machine learning approaches are being
used to understand the dynamic trend of COVID-19 spread. In this paper, we
designed the recurrent and convolutional neural network models: vanilla LSTM,
stacked LSTM, ED-LSTM, Bi-LSTM, CNN, and hybrid CNN+LSTM model to capture the
complex trend of COVID-19 outbreak and perform the forecasting of COVID-19
daily confirmed cases of 7, 14, 21 days for India and its four most affected
states (Maharashtra, Kerala, Karnataka, and Tamil Nadu). The root mean square
error (RMSE) and mean absolute percentage error (MAPE) evaluation metric are
computed on the testing data to demonstrate the relative performance of these
models. The results show that the stacked LSTM and hybrid CNN+LSTM models
perform best relative to other models.

    

### [[2108.13824] Aligning Hotel Embeddings using Domain Adaptation for Next-Item Recommendation](http://arxiv.org/abs/2108.13824)


  In online platforms it is often the case to have multiple brands under the
same group which may target different customer profiles, or have different
domains. For example, in the hospitality domain, Expedia Group has multiple
brands like Brand Expedia, this http URL and Wotif which have either different
traveler profiles or are more relevant in a local context.
In this context, learning embeddings for hotels that can be leveraged in
recommendation tasks in multiple brands requires to have a common embedding
that can be induced using alignment approaches. In the same time, one needs to
ensure that this common embedding space does not degrade the performance in any
of the brands.
In this work we build upon the hotel2vec model and propose a simple
regularization approach for aligning hotel embeddings of different brands via
domain adaptation. We also explore alignment methods previously used in
cross-lingual embeddings to align spaces of different languages. We present
results on the task of next-hotel prediction using click sessions from two
brands. The results show that the proposed approach can align the two embedding
spaces while achieving good performance in both brands. Additionally, with
respect to single-brand training we show that the proposed approach can
significantly reduce training time and improve the predictive performance.

    

### [[2108.13831] Deep Learning of Transferable MIMO Channel Modes for 6G V2X Communications](http://arxiv.org/abs/2108.13831)


  In the emerging high mobility Vehicle-to-Everything (V2X) communications
using millimeter Wave (mmWave) and sub-THz, Multiple-Input Multiple-Output
(MIMO) channel estimation is an extremely challenging task. At mmWaves/sub-THz
frequencies, MIMO channels exhibit few leading paths in the space-time domain
(i.e., directions or arrival/departure and delays). Algebraic Low-rank (LR)
channel estimation exploits space-time channel sparsity through the computation
of position-dependent MIMO channel eigenmodes leveraging recurrent training
vehicle passages in the coverage cell. LR requires vehicles' geographical
positions and tens to hundreds of training vehicles' passages for each
position, leading to significant complexity and control signalling overhead.
Here we design a DL-based LR channel estimation method to infer MIMO channel
eigenmodes in V2X urban settings, starting from a single LS channel estimate
and without needing vehicle's position information. Numerical results show that
the proposed method attains comparable Mean Squared Error (MSE) performance as
the position-based LR. Moreover, we show that the proposed model can be trained
on a reference scenario and be effectively transferred to urban contexts with
different space-time channel features, providing comparable MSE performance
without an explicit transfer learning procedure. This result eases the
deployment in arbitrary dense urban scenarios.

    

### [[2108.13836] Explainable AI for engineering design: A unified approach of systems engineering and component-based deep learning](http://arxiv.org/abs/2108.13836)


  Data-driven models created by machine learning gain in importance in all
fields of design and engineering. They have high potential to assists
decision-makers in creating novel artefacts with a better performance and
sustainability. However, limited generalization and the black-box nature of
these models induce limited explainability and reusability. These drawbacks
provide significant barriers retarding adoption in engineering design. To
overcome this situation, we propose a component-based approach to create
partial component models by machine learning (ML). This component-based
approach aligns deep learning to systems engineering (SE). By means of the
example of energy efficient building design, we first demonstrate
generalization of the component-based method by accurately predicting the
performance of designs with random structure different from training data.
Second, we illustrate explainability by local sampling, sensitivity information
and rules derived from low-depth decision trees and by evaluating this
information from an engineering design perspective. The key for explainability
is that activations at interfaces between the components are interpretable
engineering quantities. In this way, the hierarchical component system forms a
deep neural network (DNN) that directly integrates information for engineering
explainability. The large range of possible configurations in composing
components allows the examination of novel unseen design cases with
understandable data-driven models. The matching of parameter ranges of
components by similar probability distribution produces reusable,
well-generalizing, and trustworthy models. The approach adapts the model
structure to engineering methods of systems engineering and domain knowledge.

    

### [[2108.13837] Towards a Common Testing Terminology for Software Engineering and Artificial Intelligence Experts](http://arxiv.org/abs/2108.13837)


  Analytical quality assurance, especially testing, is an integral part of
software-intensive system development. With the increased usage of Artificial
Intelligence (AI) and Machine Learning (ML) as part of such systems, this
becomes more difficult as well-understood software testing approaches cannot be
applied directly to the AI-enabled parts of the system. The required adaptation
of classical testing approaches and development of new concepts for AI would
benefit from a deeper understanding and exchange between AI and software
engineering experts. A major obstacle on this way, we see in the different
terminologies used in the two communities. As we consider a mutual
understanding of the testing terminology as a key, this paper contributes a
mapping between the most important concepts from classical software testing and
AI testing. In the mapping, we highlight differences in relevance and naming of
the mapped concepts.

    

### [[2108.13858] GRP-FED: Addressing Client Imbalance in Federated Learning via Global-Regularized Personalization](http://arxiv.org/abs/2108.13858)


  Since data is presented long-tailed in reality, it is challenging for
Federated Learning (FL) to train across decentralized clients as practical
applications. We present Global-Regularized Personalization (GRP-FED) to tackle
the data imbalanced issue by considering a single global model and multiple
local models for each client. With adaptive aggregation, the global model
treats multiple clients fairly and mitigates the global long-tailed issue. Each
local model is learned from the local data and aligns with its distribution for
customization. To prevent the local model from just overfitting, GRP-FED
applies an adversarial discriminator to regularize between the learned
global-local features. Extensive results show that our GRP-FED improves under
both global and local scenarios on real-world MIT-BIH and synthesis CIFAR-10
datasets, achieving comparable performance and addressing client imbalance.

    

### [[2108.13865] InSeGAN: A Generative Approach to Segmenting Identical Instances in Depth Images](http://arxiv.org/abs/2108.13865)


  In this paper, we present InSeGAN, an unsupervised 3D generative adversarial
network (GAN) for segmenting (nearly) identical instances of rigid objects in
depth images. Using an analysis-by-synthesis approach, we design a novel GAN
architecture to synthesize a multiple-instance depth image with independent
control over each instance. InSeGAN takes in a set of code vectors (e.g.,
random noise vectors), each encoding the 3D pose of an object that is
represented by a learned implicit object template. The generator has two
distinct modules. The first module, the instance feature generator, uses each
encoded pose to transform the implicit template into a feature map
representation of each object instance. The second module, the depth image
renderer, aggregates all of the single-instance feature maps output by the
first module and generates a multiple-instance depth image. A discriminator
distinguishes the generated multiple-instance depth images from the
distribution of true depth images. To use our model for instance segmentation,
we propose an instance pose encoder that learns to take in a generated depth
image and reproduce the pose code vectors for all of the object instances. To
evaluate our approach, we introduce a new synthetic dataset, "Insta-10",
consisting of 100,000 depth images, each with 5 instances of an object from one
of 10 classes. Our experiments on Insta-10, as well as on real-world noisy
depth images, show that InSeGAN achieves state-of-the-art performance, often
outperforming prior methods by large margins.

    

### [[2108.13872] Reinforcement Learning Based Sparse Black-box Adversarial Attack on Video Recognition Models](http://arxiv.org/abs/2108.13872)


  We explore the black-box adversarial attack on video recognition models.
Attacks are only performed on selected key regions and key frames to reduce the
high computation cost of searching adversarial perturbations on a video due to
its high dimensionality. To select key frames, one way is to use heuristic
algorithms to evaluate the importance of each frame and choose the essential
ones. However, it is time inefficient on sorting and searching. In order to
speed up the attack process, we propose a reinforcement learning based frame
selection strategy. Specifically, the agent explores the difference between the
original class and the target class of videos to make selection decisions. It
receives rewards from threat models which indicate the quality of the
decisions. Besides, we also use saliency detection to select key regions and
only estimate the sign of gradient instead of the gradient itself in zeroth
order optimization to further boost the attack process. We can use the trained
model directly in the untargeted attack or with little fine-tune in the
targeted attack, which saves computation time. A range of empirical results on
real datasets demonstrate the effectiveness and efficiency of the proposed
method.

    

### [[2108.13873] Beyond Model Extraction: Imitation Attack for Black-Box NLP APIs](http://arxiv.org/abs/2108.13873)


  Machine-learning-as-a-service (MLaaS) has attracted millions of users to
their outperforming sophisticated models. Although published as black-box APIs,
the valuable models behind these services are still vulnerable to imitation
attacks. Recently, a series of works have demonstrated that attackers manage to
steal or extract the victim models. Nonetheless, none of the previous stolen
models can outperform the original black-box APIs. In this work, we take the
first step of showing that attackers could potentially surpass victims via
unsupervised domain adaptation and multi-victim ensemble. Extensive experiments
on benchmark datasets and real-world APIs validate that the imitators can
succeed in outperforming the original black-box models. We consider this as a
milestone in the research of imitation attack, especially on NLP APIs, as the
superior performance could influence the defense or even publishing strategy of
API providers.

    

### [[2108.13880] Using a one dimensional parabolic model of the full-batch loss to estimate learning rates during training](http://arxiv.org/abs/2108.13880)


  A fundamental challenge in Deep Learning is to find optimal step sizes for
stochastic gradient descent. In traditional optimization, line searches are a
commonly used method to determine step sizes. One problem in Deep Learning is
that finding appropriate step sizes on the full-batch loss is unfeasible
expensive. Therefore, classical line search approaches, designed for losses
without inherent noise, are usually not applicable. Recent empirical findings
suggest that the full-batch loss behaves locally parabolically in the direction
of noisy update step directions. Furthermore, the trend of the optimal update
step size is changing slowly. By exploiting these findings, this work
introduces a line-search method that approximates the full-batch loss with a
parabola estimated over several mini-batches. Learning rates are derived from
such parabolas during training. In the experiments conducted, our approach
mostly outperforms SGD tuned with a piece-wise constant learning rate schedule
and other line search approaches for Deep Learning across models, datasets, and
batch sizes on validation and test accuracy.

    

### [[2108.13886] Structure-Aware Hard Negative Mining for Heterogeneous Graph Contrastive Learning](http://arxiv.org/abs/2108.13886)


  Recently, heterogeneous Graph Neural Networks (GNNs) have become a de facto
model for analyzing HGs, while most of them rely on a relative large number of
labeled data. In this work, we investigate Contrastive Learning (CL), a key
component in self-supervised approaches, on HGs to alleviate the label scarcity
problem. We first generate multiple semantic views according to metapaths and
network schemas. Then, by pushing node embeddings corresponding to different
semantic views close to each other (positives) and pulling other embeddings
apart (negatives), one can obtain informative representations without human
annotations. However, this CL approach ignores the relative hardness of
negative samples, which may lead to suboptimal performance. Considering the
complex graph structure and the smoothing nature of GNNs, we propose a
structure-aware hard negative mining scheme that measures hardness by
structural characteristics for HGs. By synthesizing more negative nodes, we
give larger weights to harder negatives with limited computational overhead to
further boost the performance. Empirical studies on three real-world datasets
show the effectiveness of our proposed method. The proposed method consistently
outperforms existing state-of-the-art methods and notably, even surpasses
several supervised counterparts.

    

### [[2108.13902] Estimation of Air Pollution with Remote Sensing Data: Revealing Greenhouse Gas Emissions from Space](http://arxiv.org/abs/2108.13902)


  Air pollution is a major driver of climate change. Anthropogenic emissions
from the burning of fossil fuels for transportation and power generation emit
large amounts of problematic air pollutants, including Greenhouse Gases (GHGs).
Despite the importance of limiting GHG emissions to mitigate climate change,
detailed information about the spatial and temporal distribution of GHG and
other air pollutants is difficult to obtain. Existing models for surface-level
air pollution rely on extensive land-use datasets which are often locally
restricted and temporally static. This work proposes a deep learning approach
for the prediction of ambient air pollution that only relies on remote sensing
data that is globally available and frequently updated. Combining optical
satellite imagery with satellite-based atmospheric column density air pollution
measurements enables the scaling of air pollution estimates (in this case
NO$_2$) to high spatial resolution (up to $\sim$10m) at arbitrary locations and
adds a temporal component to these estimates. The proposed model performs with
high accuracy when evaluated against air quality measurements from ground
stations (mean absolute error $<$6$~\mu g/m^3$). Our results enable the
identification and temporal monitoring of major sources of air pollution and
GHGs.

    

### [[2108.13908] Modeling the effect of the vaccination campaign on the Covid-19 pandemic](http://arxiv.org/abs/2108.13908)


  Population-wide vaccination is critical for containing the SARS-CoV-2
(Covid-19) pandemic when combined with restrictive and prevention measures. In
this study, we introduce SAIVR, a mathematical model able to forecast the
Covid-19 epidemic evolution during the vaccination campaign. SAIVR extends the
widely used Susceptible-Infectious-Removed (SIR) model by considering the
Asymptomatic (A) and Vaccinated (V) compartments. The model contains several
parameters and initial conditions that are estimated by employing a
semi-supervised machine learning procedure. After training an unsupervised
neural network to solve the SAIVR differential equations, a supervised
framework then estimates the optimal conditions and parameters that best fit
recent infectious curves of 27 countries. Instructed by these results, we
performed an extensive study on the temporal evolution of the pandemic under
varying values of roll-out daily rates, vaccine efficacy, and a broad range of
societal vaccine hesitancy/denial levels. The concept of herd immunity is
questioned by studying future scenarios which involve different vaccination
efforts and more infectious Covid-19 variants.

    

### [[2108.13910] A manifold learning perspective on representation learning: Learning decoder and representations without an encoder](http://arxiv.org/abs/2108.13910)


  Autoencoders are commonly used in representation learning. They consist of an
encoder and a decoder, which provide a straightforward way to map
$n$-dimensional data in input space to a lower $m$-dimensional representation
space and back. The decoder itself defines an $m$-dimensional manifold in input
space. Inspired by manifold learning, we show that the decoder can be trained
on its own by learning the representations of the training samples along with
the decoder weights using gradient descent. A sum-of-squares loss then
corresponds to optimizing the manifold to have the smallest Euclidean distance
to the training samples, and similarly for other loss functions. We derive
expressions for the number of samples needed to specify the encoder and decoder
and show that the decoder generally requires much less training samples to be
well-specified compared to the encoder. We discuss training of autoencoders in
this perspective and relate to previous work in the field that use noisy
training examples and other types of regularization. On the natural image data
sets MNIST and CIFAR10, we demonstrate that the decoder is much better suited
to learn a low-dimensional representation, especially when trained on small
data sets. Using simulated gene regulatory data, we further show that the
decoder alone leads to better generalization and meaningful representations.
Our approach of training the decoder alone facilitates representation learning
even on small data sets and can lead to improved training of autoencoders. We
hope that the simple analyses presented will also contribute to an improved
conceptual understanding of representation learning.

    

### [[2108.13914] On the interpretation of black-box default prediction models: an Italian Small and Medium Enterprises case](http://arxiv.org/abs/2108.13914)


  Academic research and the financial industry have recently paid great
attention to Machine Learning algorithms due to their power to solve complex
learning tasks. In the field of firms' default prediction, however, the lack of
interpretability has prevented the extensive adoption of the black-box type of
models. To overcome this drawback and maintain the high performances of
black-boxes, this paper relies on a model-agnostic approach. Accumulated Local
Effects and Shapley values are used to shape the predictors' impact on the
likelihood of default and rank them according to their contribution to the
model outcome. Prediction is achieved by two Machine Learning algorithms
(eXtreme Gradient Boosting and FeedForward Neural Network) compared with three
standard discriminant models. Results show that our analysis of the Italian
Small and Medium Enterprises manufacturing industry benefits from the overall
highest classification power by the eXtreme Gradient Boosting algorithm without
giving up a rich interpretation framework.

    

### [[2108.13930] EG-Booster: Explanation-Guided Booster of ML Evasion Attacks](http://arxiv.org/abs/2108.13930)


  The widespread usage of machine learning (ML) in a myriad of domains has
raised questions about its trustworthiness in security-critical environments.
Part of the quest for trustworthy ML is robustness evaluation of ML models to
test-time adversarial examples. Inline with the trustworthy ML goal, a useful
input to potentially aid robustness evaluation is feature-based explanations of
model predictions. In this paper, we present a novel approach called EG-Booster
that leverages techniques from explainable ML to guide adversarial example
crafting for improved robustness evaluation of ML models before deploying them
in security-critical settings. The key insight in EG-Booster is the use of
feature-based explanations of model predictions to guide adversarial example
crafting by adding consequential perturbations likely to result in model
evasion and avoiding non-consequential ones unlikely to contribute to evasion.
EG-Booster is agnostic to model architecture, threat model, and supports
diverse distance metrics used previously in the literature. We evaluate
EG-Booster using image classification benchmark datasets, MNIST and CIFAR10.
Our findings suggest that EG-Booster significantly improves evasion rate of
state-of-the-art attacks while performing less number of perturbations. Through
extensive experiments that covers four white-box and three black-box attacks,
we demonstrate the effectiveness of EG-Booster against two undefended neural
networks trained on MNIST and CIFAR10, and another adversarially-trained ResNet
model trained on CIFAR10. Furthermore, we introduce a stability assessment
metric and evaluate the reliability of our explanation-based approach by
observing the similarity between the model's classification outputs across
multiple runs of EG-Booster.

    

### [[2108.13941] Bubblewrap: Online tiling and real-time flow prediction on neural manifolds](http://arxiv.org/abs/2108.13941)


  While most classic studies of function in experimental neuroscience have
focused on the coding properties of individual neurons, recent developments in
recording technologies have resulted in an increasing emphasis on the dynamics
of neural populations. This has given rise to a wide variety of models for
analyzing population activity in relation to experimental variables, but direct
testing of many neural population hypotheses requires intervening in the system
based on current neural state, necessitating models capable of inferring neural
state online. Existing approaches, primarily based on dynamical systems,
require strong parametric assumptions that are easily violated in the
noise-dominated regime and do not scale well to the thousands of data channels
in modern experiments. To address this problem, we propose a method that
combines fast, stable dimensionality reduction with a soft tiling of the
resulting neural manifold, allowing dynamics to be approximated as a
probability flow between tiles. This method can be fit efficiently using online
expectation maximization, scales to tens of thousands of tiles, and outperforms
existing methods when dynamics are noise-dominated or feature multi-modal
transition probabilities. The resulting model can be trained at kiloHertz data
rates, produces accurate approximations of neural dynamics within minutes, and
generates predictions on submillisecond time scales. It retains predictive
performance throughout many time steps into the future and is fast enough to
serve as a component of closed-loop causal experiments.

    

### [[2108.13947] Decision Tree-Based Predictive Models for Academic Achievement Using College Students' Support Networks](http://arxiv.org/abs/2108.13947)


  In this study, we examine a set of primary data collected from 484 students
enrolled in a large public university in the Mid-Atlantic United States region
during the early stages of the COVID-19 pandemic. The data, called Ties data,
included students' demographic and support network information. The support
network data comprised of information that highlighted the type of support,
(i.e. emotional or educational; routine or intense). Using this data set,
models for predicting students' academic achievement, quantified by their
self-reported GPA, were created using Chi-Square Automatic Interaction
Detection (CHAID), a decision tree algorithm, and cforest, a random forest
algorithm that uses conditional inference trees. We compare the methods'
accuracy and variation in the set of important variables suggested by each
algorithm. Each algorithm found different variables important for different
student demographics with some overlap. For White students, different types of
educational support were important in predicting academic achievement, while
for non-White students, different types of emotional support were important in
predicting academic achievement. The presence of differing types of routine
support were important in predicting academic achievement for cisgender women,
while differing types of intense support were important in predicting academic
achievement for cisgender men.

    

### [[2108.13952] Morphence: Moving Target Defense Against Adversarial Examples](http://arxiv.org/abs/2108.13952)


  Robustness to adversarial examples of machine learning models remains an open
topic of research. Attacks often succeed by repeatedly probing a fixed target
model with adversarial examples purposely crafted to fool it. In this paper, we
introduce Morphence, an approach that shifts the defense landscape by making a
model a moving target against adversarial examples. By regularly moving the
decision function of a model, Morphence makes it significantly challenging for
repeated or correlated attacks to succeed. Morphence deploys a pool of models
generated from a base model in a manner that introduces sufficient randomness
when it responds to prediction queries. To ensure repeated or correlated
attacks fail, the deployed pool of models automatically expires after a query
budget is reached and the model pool is seamlessly replaced by a new model pool
generated in advance. We evaluate Morphence on two benchmark image
classification datasets (MNIST and CIFAR10) against five reference attacks (2
white-box and 3 black-box). In all cases, Morphence consistently outperforms
the thus-far effective defense, adversarial training, even in the face of
strong white-box attacks, while preserving accuracy on clean data.

    

### [[2108.13956] APS: Active Pretraining with Successor Features](http://arxiv.org/abs/2108.13956)


  We introduce a new unsupervised pretraining objective for reinforcement
learning. During the unsupervised reward-free pretraining phase, the agent
maximizes mutual information between tasks and states induced by the policy.
Our key contribution is a novel lower bound of this intractable quantity. We
show that by reinterpreting and combining variational successor
features~\citep{Hansen2020Fast} with nonparametric entropy
maximization~\citep{liu2021behavior}, the intractable mutual information can be
efficiently optimized. The proposed method Active Pretraining with Successor
Feature (APS) explores the environment via nonparametric entropy maximization,
and the explored data can be efficiently leveraged to learn behavior by
variational successor features. APS addresses the limitations of existing
mutual information maximization based and entropy maximization based
unsupervised RL, and combines the best of both worlds. When evaluated on the
Atari 100k data-efficiency benchmark, our approach significantly outperforms
previous methods combining unsupervised pretraining with task-specific
finetuning.

    

### [[2108.13963] Clustering of Pain Dynamics in Sickle Cell Disease from Sparse, Uneven Samples](http://arxiv.org/abs/2108.13963)


  Irregularly sampled time series data are common in a variety of fields. Many
typical methods for drawing insight from data fail in this case. Here we
attempt to generalize methods for clustering trajectories to irregularly and
sparsely sampled data. We first construct synthetic data sets, then propose and
assess four methods of data alignment to allow for application of spectral
clustering. We also repeat the same process for real data drawn from medical
records of patients with sickle cell disease -- patients whose subjective
experiences of pain were tracked for several months via a mobile app.
We find that different methods for aligning irregularly sampled sparse data
sets can lead to different optimal numbers of clusters, even for synthetic data
with known properties. For the case of sickle cell disease, we find that three
clusters is a reasonable choice, and these appear to correspond to (1) a low
pain group with occasionally acute pain, (2) a group which experiences moderate
mean pain that fluctuates often from low to high, and (3) a group that
experiences persistent high levels of pain.
Our results may help physicians and patients better understand and manage
patients' pain levels over time, and we expect that the methods we develop will
apply to a wide range of other data sources in medicine and beyond.

    

### [[2108.13965] Approximation Methods for Partially Observed Markov Decision Processes (POMDPs)](http://arxiv.org/abs/2108.13965)


  POMDPs are useful models for systems where the true underlying state is not
known completely to an outside observer; the outside observer incompletely
knows the true state of the system, and observes a noisy version of the true
system state. When the number of system states is large in a POMDP that often
necessitates the use of approximation methods to obtain near optimal solutions
for control. This survey is centered around the origins, theory, and
approximations of finite-state POMDPs. In order to understand POMDPs, it is
required to have an understanding of finite-state Markov Decision Processes
(MDPs) in \autoref{mdp} and Hidden Markov Models (HMMs) in \autoref{hmm}. For
this background theory, I provide only essential details on MDPs and HMMs and
leave longer expositions to textbook treatments before diving into the main
topics of POMDPs. Once the required background is covered, the POMDP is
introduced in \autoref{pomdp}. The origins of the POMDP are explained in the
classical papers section \autoref{classical}. Once the high computational
requirements are understood from the exact methodological point of view, the
main approximation methods are surveyed in \autoref{approximations}. Then, I
end the survey with some new research directions in \autoref{conclusion}.

    

### [[2108.13969] S4-Crowd: Semi-Supervised Learning with Self-Supervised Regularisation for Crowd Counting](http://arxiv.org/abs/2108.13969)


  Crowd counting has drawn more attention because of its wide application in
smart cities. Recent works achieved promising performance but relied on the
supervised paradigm with expensive crowd annotations. To alleviate annotation
cost, in this work we proposed a semi-supervised learning framework S4-Crowd,
which can leverage both unlabeled/labeled data for robust crowd modelling. In
the unsupervised pathway, two self-supervised losses were proposed to simulate
the crowd variations such as scale, illumination, etc., based on which and the
supervised information pseudo labels were generated and gradually refined. We
also proposed a crowd-driven recurrent unit Gated-Crowd-Recurrent-Unit (GCRU),
which can preserve discriminant crowd information by extracting second-order
statistics, yielding pseudo labels with improved quality. A joint loss
including both unsupervised/supervised information was proposed, and a dynamic
weighting strategy was employed to balance the importance of the unsupervised
loss and supervised loss at different training stages. We conducted extensive
experiments on four popular crowd counting datasets in semi-supervised
settings. Experimental results suggested the effectiveness of each proposed
component in our S4-Crowd framework. Our method also outperformed other
state-of-the-art semi-supervised learning approaches on these crowd datasets.

    

### [[2108.13976] WarpDrive: Extremely Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU](http://arxiv.org/abs/2108.13976)


  Deep reinforcement learning (RL) is a powerful framework to train
decision-making models in complex dynamical environments. However, RL can be
slow as it learns through repeated interaction with a simulation of the
environment. Accelerating RL requires both algorithmic and engineering
innovations. In particular, there are key systems engineering bottlenecks when
using RL in complex environments that feature multiple agents or
high-dimensional state, observation, or action spaces, for example. We present
WarpDrive, a flexible, lightweight, and easy-to-use open-source RL framework
that implements end-to-end multi-agent RL on a single GPU (Graphics Processing
Unit), building on PyCUDA and PyTorch. Using the extreme parallelization
capability of GPUs, WarpDrive enables orders-of-magnitude faster RL compared to
common implementations that blend CPU simulations and GPU models. Our design
runs simulations and the agents in each simulation in parallel. It eliminates
data copying between CPU and GPU. It also uses a single simulation data store
on the GPU that is safely updated in-place. Together, this allows the user to
run thousands of concurrent multi-agent simulations and train on extremely
large batches of experience. For example, WarpDrive yields 2.9 million
environment steps/second with 2000 environments and 1000 agents (at least 100x
higher throughput compared to a CPU implementation) in a benchmark Tag
simulation. WarpDrive provides a lightweight Python interface and environment
wrappers to simplify usage and promote flexibility and extensions. As such,
WarpDrive provides a framework for building high-throughput RL systems.

    

### [[2108.13984] A Subsampling Based Method for Causal Discovery on Discrete Data](http://arxiv.org/abs/2108.13984)


  Inferring causal directions on discrete and categorical data is an important
yet challenging problem. Even though the additive noise models (ANMs) approach
can be adapted to the discrete data, the functional structure assumptions make
it not applicable on categorical data. Inspired by the principle that the cause
and mechanism are independent, various methods have been developed, leveraging
independence tests such as the distance correlation measure. In this work, we
take an alternative perspective and propose a subsampling-based method to test
the independence between the generating schemes of the cause and that of the
mechanism. Our methodology works for both discrete and categorical data and
does not imply any functional model on the data, making it a more flexible
approach. To demonstrate the efficacy of our methodology, we compare it with
existing baselines over various synthetic data and real data experiments.

    

### [[2108.13990] Effective Sequence-to-Sequence Dialogue State Tracking](http://arxiv.org/abs/2108.13990)


  Sequence-to-sequence models have been applied to a wide variety of NLP tasks,
but how to properly use them for dialogue state tracking has not been
systematically investigated. In this paper, we study this problem from the
perspectives of pre-training objectives as well as the formats of context
representations. We demonstrate that the choice of pre-training objective makes
a significant difference to the state tracking quality. In particular, we find
that masked span prediction is more effective than auto-regressive language
modeling. We also explore using Pegasus, a span prediction-based pre-training
objective for text summarization, for the state tracking model. We found that
pre-training for the seemingly distant summarization task works surprisingly
well for dialogue state tracking. In addition, we found that while recurrent
state context representation works also reasonably well, the model may have a
hard time recovering from earlier mistakes. We conducted experiments on the
MultiWOZ 2.1-2.4 data sets with consistent observations.

    

### [[2108.13992] Bayesian learning of forest and tree graphical models](http://arxiv.org/abs/2108.13992)


  In Bayesian learning of Gaussian graphical model structure, it is common to
restrict attention to certain classes of graphs and approximate the posterior
distribution by repeatedly moving from one graph to another, using MCMC or
methods such as stochastic shotgun search (SSS). I give two corrected versions
of an algorithm for non-decomposable graphs and discuss random graph
distributions, in particular as prior distributions. The main topic of the
thesis is Bayesian structure-learning with forests or trees. Restricting
attention to these graphs can be justified using theorems on random graphs. I
describe how to use the Chow$\unicode{x2013}$Liu algorithm and the Matrix Tree
Theorem to find the MAP forest and certain quantities in the posterior
distribution on trees. I give adapted versions of MCMC and SSS for
approximating the posterior distribution for forests and trees, and systems for
storing these graphs so that it is easy to choose moves to neighbouring graphs.
Experiments show that SSS with trees does well when the true graph is a tree or
sparse graph. SSS with trees or forests does better than SSS with decomposable
graphs in certain cases. Graph priors improve detection of hubs but need large
ranges of probabilities. MCMC on forests fails to mix well and MCMC on trees is
slower than SSS. (For a longer abstract see the thesis.)

    

### [[2108.13993] Designing Rotationally Invariant Neural Networks from PDEs and Variational Methods](http://arxiv.org/abs/2108.13993)


  Partial differential equation (PDE) models and their associated variational
energy formulations are often rotationally invariant by design. This ensures
that a rotation of the input results in a corresponding rotation of the output,
which is desirable in applications such as image analysis. Convolutional neural
networks (CNNs) do not share this property, and existing remedies are often
complex. The goal of our paper is to investigate how diffusion and variational
models achieve rotation invariance and transfer these ideas to neural networks.
As a core novelty we propose activation functions which couple network channels
by combining information from several oriented filters. This guarantees
rotation invariance within the basic building blocks of the networks while
still allowing for directional filtering. The resulting neural architectures
are inherently rotationally invariant. With only a few small filters, they can
achieve the same invariance as existing techniques which require a fine-grained
sampling of orientations. Our findings help to translate diffusion and
variational models into mathematically well-founded network architectures, and
provide novel concepts for model-based CNN design.

    

### [[2108.13996] Quantization of Generative Adversarial Networks for Efficient Inference: a Methodological Study](http://arxiv.org/abs/2108.13996)


  Generative adversarial networks (GANs) have an enormous potential impact on
digital content creation, e.g., photo-realistic digital avatars, semantic
content editing, and quality enhancement of speech and images. However, the
performance of modern GANs comes together with massive amounts of computations
performed during the inference and high energy consumption. That complicates,
or even makes impossible, their deployment on edge devices. The problem can be
reduced with quantization -- a neural network compression technique that
facilitates hardware-friendly inference by replacing floating-point
computations with low-bit integer ones. While quantization is well established
for discriminative models, the performance of modern quantization techniques in
application to GANs remains unclear. GANs generate content of a more complex
structure than discriminative models, and thus quantization of GANs is
significantly more challenging. To tackle this problem, we perform an extensive
experimental study of state-of-art quantization techniques on three diverse GAN
architectures, namely StyleGAN, Self-Attention GAN, and CycleGAN. As a result,
we discovered practical recipes that allowed us to successfully quantize these
models for inference with 4/8-bit weights and 8-bit activations while
preserving the quality of the original full-precision models.

    

### [[1505.02213] Measuring dependence powerfully and equitably](http://arxiv.org/abs/1505.02213)


  Given a high-dimensional data set we often wish to find the strongest
relationships within it. A common strategy is to evaluate a measure of
dependence on every variable pair and retain the highest-scoring pairs for
follow-up. This strategy works well if the statistic used is equitable [Reshef
et al. 2015a], i.e., if, for some measure of noise, it assigns similar scores
to equally noisy relationships regardless of relationship type (e.g., linear,
exponential, periodic).
In this paper, we introduce and characterize a population measure of
dependence called MIC*. We show three ways that MIC* can be viewed: as the
population value of MIC, a highly equitable statistic from [Reshef et al.
2011], as a canonical "smoothing" of mutual information, and as the supremum of
an infinite sequence defined in terms of optimal one-dimensional partitions of
the marginals of the joint distribution. Based on this theory, we introduce an
efficient approach for computing MIC* from the density of a pair of random
variables, and we define a new consistent estimator MICe for MIC* that is
efficiently computable. In contrast, there is no known polynomial-time
algorithm for computing the original equitable statistic MIC. We show through
simulations that MICe has better bias-variance properties than MIC. We then
introduce and prove the consistency of a second statistic, TICe, that is a
trivial side-product of the computation of MICe and whose goal is powerful
independence testing rather than equitability.
We show in simulations that MICe and TICe have good equitability and power
against independence respectively. The analyses here complement a more in-depth
empirical evaluation of several leading measures of dependence [Reshef et al.
2015b] that shows state-of-the-art performance for MICe and TICe.

    

### [[1912.03437] Early Prediction for Merged vs Abandoned Code Changes in Modern Code Reviews](http://arxiv.org/abs/1912.03437)


  The modern code review process is an integral part of the current software
development practice. Considerable effort is given here to inspect code
changes, find defects, suggest an improvement, and address the suggestions of
the reviewers. In a code review process, usually, several iterations take place
where an author submits code changes and a reviewer gives feedback until is
happy to accept the change. In around 12% cases, the changes are abandoned,
eventually wasting all the efforts. In this research, our objective is to
design a tool that can predict whether a code change would be merged or
abandoned at an early stage to reduce the waste of efforts of all stakeholders
(e.g., program author, reviewer, project management, etc.) involved. The
real-world demand for such a tool was formally identified by a study by Fan et
al. [1]. We have mined 146,612 code changes from the code reviews of three
large and popular open-source software and trained and tested a suite of
supervised machine learning classifiers, both shallow and deep learning based.
We consider a total of 25 features in each code change during the training and
testing of the models. The best performing model named PredCR (Predicting Code
Review), a LightGBM-based classifier achieves around 85% AUC score on average
and relatively improves the state-of-the-art [1] by 14-23%. In our empirical
study on the 146,612 code changes from the three software projects, we find
that (1) The new features like reviewer dimensions that are introduced in
PredCR are the most informative. (2) Compared to the baseline, PredCR is more
effective towards reducing bias against new developers. (3) PredCR uses
historical data in the code review repository and as such the performance of
PredCR improves as a software system evolves with new and more data.

    

### [[1912.10340] Bandit Multiclass Linear Classification for the Group Linear Separable Case](http://arxiv.org/abs/1912.10340)


  We consider the online multiclass linear classification under the bandit
feedback setting. Beygelzimer, Pál, Szörényi, Thiruvenkatachari,
Wei, and Zhang [ICML'19] considered two notions of linear separability, weak
and strong linear separability. When examples are strongly linearly separable
with margin $\gamma$, they presented an algorithm based on Multiclass
Perceptron with mistake bound $O(K/\gamma^2)$, where $K$ is the number of
classes. They employed rational kernel to deal with examples under the weakly
linearly separable condition, and obtained the mistake bound of $\min(K\cdot
2^{\tilde{O}(K\log^2(1/\gamma))},K\cdot 2^{\tilde{O}(\sqrt{1/\gamma}\log K)})$.
In this paper, we refine the notion of weak linear separability to support the
notion of class grouping, called group weak linear separable condition. This
situation may arise from the fact that class structures contain inherent
grouping. We show that under this condition, we can also use the rational
kernel and obtain the mistake bound of $K\cdot 2^{\tilde{O}(\sqrt{1/\gamma}\log
L)})$, where $L\leq K$ represents the number of groups.

    

### [[2002.11219] Convex Geometry and Duality of Over-parameterized Neural Networks](http://arxiv.org/abs/2002.11219)


  We develop a convex analytic approach to analyze finite width two-layer ReLU
networks. We first prove that an optimal solution to the regularized training
problem can be characterized as extreme points of a convex set, where simple
solutions are encouraged via its convex geometrical properties. We then
leverage this characterization to show that an optimal set of parameters yield
linear spline interpolation for regression problems involving one dimensional
or rank-one data. We also characterize the classification decision regions in
terms of a kernel matrix and minimum $\ell_1$-norm solutions. This is in
contrast to Neural Tangent Kernel which is unable to explain predictions of
finite width networks. Our convex geometric characterization also provides
intuitive explanations of hidden neurons as auto-encoders. In higher
dimensions, we show that the training problem can be cast as a finite
dimensional convex problem with infinitely many constraints. Then, we apply
certain convex relaxations and introduce a cutting-plane algorithm to globally
optimize the network. We further analyze the exactness of the relaxations to
provide conditions for the convergence to a global optimum. Our analysis also
shows that optimal network parameters can be also characterized as
interpretable closed-form formulas in some practically relevant special cases.

    

### [[2004.06243] Physics-Incorporated Convolutional Recurrent Neural Networks for Source Identification and Forecasting of Dynamical Systems](http://arxiv.org/abs/2004.06243)


  Spatio-temporal dynamics of physical processes are generally modeled using
partial differential equations (PDEs). Though the core dynamics follows some
principles of physics, real-world physical processes are often driven by
unknown external sources. In such cases, developing a purely analytical model
becomes very difficult and data-driven modeling can be of assistance. In this
paper, we present a hybrid framework combining physics-based numerical models
with deep learning for source identification and forecasting of spatio-temporal
dynamical systems with unobservable time-varying external sources. We formulate
our model PhICNet as a convolutional recurrent neural network (RNN) which is
end-to-end trainable for spatio-temporal evolution prediction of dynamical
systems and learns the source behavior as an internal state of the RNN.
Experimental results show that the proposed model can forecast the dynamics for
a relatively long time and identify the sources as well.

    

### [[2004.06698] Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer](http://arxiv.org/abs/2004.06698)


  Visual dialog is a task of answering a sequence of questions grounded in an
image using the previous dialog history as context. In this paper, we study how
to address two fundamental challenges for this task: (1) reasoning over
underlying semantic structures among dialog rounds and (2) identifying several
appropriate answers to the given question. To address these challenges, we
propose a Sparse Graph Learning (SGL) method to formulate visual dialog as a
graph structure learning task. SGL infers inherently sparse dialog structures
by incorporating binary and score edges and leveraging a new structural loss
function. Next, we introduce a Knowledge Transfer (KT) method that extracts the
answer predictions from the teacher model and uses them as pseudo labels. We
propose KT to remedy the shortcomings of single ground-truth labels, which
severely limit the ability of a model to obtain multiple reasonable answers. As
a result, our proposed model significantly improves reasoning capability
compared to baseline methods and outperforms the state-of-the-art approaches on
the VisDial v1.0 dataset. The source code is available at
this https URL.

    

### [[2005.11671] Arms Race in Adversarial Malware Detection: A Survey](http://arxiv.org/abs/2005.11671)


  Malicious software (malware) is a major cyber threat that has to be tackled
with Machine Learning (ML) techniques because millions of new malware examples
are injected into cyberspace on a daily basis. However, ML is vulnerable to
attacks known as adversarial examples. In this paper, we survey and systematize
the field of Adversarial Malware Detection (AMD) through the lens of a unified
conceptual framework of assumptions, attacks, defenses, and security
properties. This not only leads us to map attacks and defenses to partial order
structures, but also allows us to clearly describe the attack-defense arms race
in the AMD context. We draw a number of insights, including: knowing the
defender's feature set is critical to the success of transfer attacks; the
effectiveness of practical evasion attacks largely depends on the attacker's
freedom in conducting manipulations in the problem space; knowing the
attacker's manipulation set is critical to the defender's success; the
effectiveness of adversarial training depends on the defender's capability in
identifying the most powerful attack. We also discuss a number of future
research directions.

    

### [[2007.01002] DeepOPF: A Feasibility-Optimized Deep Neural Network Approach for AC Optimal Power Flow Problems](http://arxiv.org/abs/2007.01002)


  We develop a Deep Neural Network (DNN) approach, named DeepOPF, for solving
alternative current optimal power flow (AC-OPF) problems. A key difficulty for
applying machine learning techniques for solving AC-OPF problems lies in
ensuring that the obtained solutions respect the equality and inequality
physical and operational constraints. Generalized a 2-stage procedure proposed
in [1], [2], DeepOPF first trains a DNN model to predict a set of independent
operating variables and then directly compute the remaining dependable ones by
solving the AC power flow equations. Such an approach not only preserves the
power-flow balance equality constraints, but also reduces the number of
variables to predict by the DNN, subsequently cutting down the number of
neurons and training data needed. DeepOPF then employs a penalty approach with
a zero-order gradient estimation technique in the training process to preserve
the remaining inequality constraints. As another contribution, we drive a
condition for tuning the size of the DNN according to the desired approximation
accuracy, which measures the DNN generalization capability. It provides
theoretical justification for using DNN to solve AC-OPF problem. Simulation
results of IEEE 30/118/300-bus and a synthetic 2000-bus test cases show that
DeepOPF can speed up the computing time by up to two orders of magnitude as
compared to a state-of-the-art solver, at the expense of $<$0.1% cost
difference.

    

### [[2007.03502] srMO-BO-3GP: A sequential regularized multi-objective constrained Bayesian optimization for design applications](http://arxiv.org/abs/2007.03502)


  Bayesian optimization (BO) is an efficient and flexible global optimization
framework that is applicable to a very wide range of engineering applications.
To leverage the capability of the classical BO, many extensions, including
multi-objective, multi-fidelity, parallelization, latent-variable model, have
been proposed to improve the limitation of the classical BO framework. In this
work, we propose a novel multi-objective (MO) extension, called srMO-BO-3GP, to
solve the MO optimization problems in a sequential setting. Three different
Gaussian processes (GPs) are stacked together, where each of the GP is assigned
with a different task: the first GP is used to approximate the single-objective
function, the second GP is used to learn the unknown constraints, and the third
GP is used to learn the uncertain Pareto frontier. At each iteration, a MO
augmented Tchebycheff function converting MO to single-objective is adopted and
extended with a regularized ridge term, where the regularization is introduced
to smoothen the single-objective function. Finally, we couple the third GP
along with the classical BO framework to promote the richness and diversity of
the Pareto frontier by the exploitation and exploration acquisition function.
The proposed framework is demonstrated using several numerical benchmark
functions, as well as a thermomechanical finite element model for flip-chip
package design optimization.

    

### [[2008.07191] Deep Variational Generative Models for Audio-visual Speech Separation](http://arxiv.org/abs/2008.07191)


  In this paper, we are interested in audio-visual speech separation given a
single-channel audio recording as well as visual information (lips movements)
associated with each speaker. We propose an unsupervised technique based on
audio-visual generative modeling of clean speech. More specifically, during
training, a latent variable generative model is learned from clean speech
spectrograms using a variational auto-encoder (VAE). To better utilize the
visual information, the posteriors of the latent variables are inferred from
mixed speech (instead of clean speech) as well as the visual data. The visual
modality also serves as a prior for latent variables, through a visual network.
At test time, the learned generative model (both for speaker-independent and
speaker-dependent scenarios) is combined with an unsupervised non-negative
matrix factorization (NMF) variance model for background noise. All the latent
variables and noise parameters are then estimated by a Monte Carlo
expectation-maximization algorithm. Our experiments show that the proposed
unsupervised VAE-based method yields better separation performance than
NMF-based approaches as well as a supervised deep learning-based technique.

    

### [[2008.13336] Shape Defense Against Adversarial Attacks](http://arxiv.org/abs/2008.13336)


  Humans rely heavily on shape information to recognize objects. Conversely,
convolutional neural networks (CNNs) are biased more towards texture. This is
perhaps the main reason why CNNs are vulnerable to adversarial examples. Here,
we explore how shape bias can be incorporated into CNNs to improve their
robustness. Two algorithms are proposed, based on the observation that edges
are invariant to moderate imperceptible perturbations. In the first one, a
classifier is adversarially trained on images with the edge map as an
additional channel. At inference time, the edge map is recomputed and
concatenated to the image. In the second algorithm, a conditional GAN is
trained to translate the edge maps, from clean and/or perturbed images, into
clean images. Inference is done over the generated image corresponding to the
input's edge map. Extensive experiments over 10 datasets demonstrate the
effectiveness of the proposed algorithms against FGSM and $\ell_\infty$ PGD-40
attacks. Further, we show that a) edge information can also benefit other
adversarial training methods, and b) CNNs trained on edge-augmented inputs are
more robust against natural image corruptions such as motion blur, impulse
noise and JPEG compression, than CNNs trained solely on RGB images. From a
broader perspective, our study suggests that CNNs do not adequately account for
image structures that are crucial for robustness. Code is available
at:~\url{this https URL}.

    

### [[2009.01454] Learning Fair Graph Neural Networks with Limited and Private Sensitive Attribute Information](http://arxiv.org/abs/2009.01454)


  Graph neural networks (GNNs) have shown great power in modeling graph
structured data. However, similar to other machine learning models, GNNs may
make biased predictions w.r.t protected sensitive attributes, e.g., skin color
and gender. This is because the training data often contains historical bias
towards sensitive attributes. In addition, we empirically show that the
discrimination in GNNs can be magnified by graph structures and the
message-passing mechanism of GNNs. As a result, the applications of GNNs in
high-stake domains such as crime rate prediction would be largely limited.
Though extensive studies of fair classification have been conducted on i.i.d
data, methods to address the problem of discrimination on non-i.i.d data are
rather limited. Generally, learning fair models require abundant sensitive
attributes to regularize the model. However, for many graphs such as social
networks, users are reluctant to share sensitive attributes. Thus, only limited
sensitive attributes are available for fair GNN training in practice. Moreover,
directly collecting and applying the sensitive attributes in fair model
training may cause privacy issues, because the sensitive information can be
leaked in data breach or attacks on the trained model. Therefore, we study a
novel and crucial problem of learning fair GNNs with limited and private
sensitive attribute information. In an attempt to address these problems,
FairGNN is proposed to eliminate the bias of GNNs whilst maintaining high
accuracy by leveraging graph structures and limited sensitive information. We
further extend FairGNN to NT-FairGNN which can achieve both fairness and
privacy on sensitive attributes by using limited and private sensitive
attributes. Theoretical analysis and extensive experiments on real-world
datasets demonstrate the effectiveness of FairGNN and NT-FairGNN in achieving
fair and high-accurate classification.

    

### [[2011.03030] Fast Rates for Contextual Linear Optimization](http://arxiv.org/abs/2011.03030)


  Incorporating side observations in decision making can reduce uncertainty and
boost performance, but it also requires we tackle a potentially complex
predictive relationship. While one may use off-the-shelf machine learning
methods to separately learn a predictive model and plug it in, a variety of
recent methods instead integrate estimation and optimization by fitting the
model to directly optimize downstream decision performance. Surprisingly, in
the case of contextual linear optimization, we show that the naive plug-in
approach actually achieves regret convergence rates that are significantly
faster than methods that directly optimize downstream decision performance. We
show this by leveraging the fact that specific problem instances do not have
arbitrarily bad near-dual-degeneracy. While there are other pros and cons to
consider as we discuss and illustrate numerically, our results highlight a
nuanced landscape for the enterprise to integrate estimation and optimization.
Our results are overall positive for practice: predictive models are easy and
fast to train using existing tools, simple to interpret, and, as we show, lead
to decisions that perform very well.

    

### [[2011.03334] Occlusion-Aware Search for Object Retrieval in Clutter](http://arxiv.org/abs/2011.03334)


  We address the manipulation task of retrieving a target object from a
cluttered shelf. When the target object is hidden, the robot must search
through the clutter for retrieving it. Solving this task requires reasoning
over the likely locations of the target object. It also requires physics
reasoning over multi-object interactions and future occlusions. In this work,
we present a data-driven hybrid planner for generating occlusion-aware actions
in closed-loop. The hybrid planner explores likely locations of the occluded
target object as predicted by a learned distribution from the observation
stream. The search is guided by a heuristic trained with reinforcement learning
to act on observations with occlusions. We evaluate our approach in different
simulation and real-world settings (video available on
this https URL). The results validate that our approach can
search and retrieve a target object in near real time in the real world while
only being trained in simulation.

    

### [[2012.09359] Speech Enhancement with Zero-Shot Model Selection](http://arxiv.org/abs/2012.09359)


  Recent research on speech enhancement (SE) has seen the emergence of
deep-learning-based methods. It is still a challenging task to determine the
effective ways to increase the generalizability of SE under diverse test
conditions. In this study, we combine zero-shot learning and ensemble learning
to propose a zero-shot model selection (ZMOS) approach to increase the
generalization of SE performance. The proposed approach is realized in the
offline and online phases. The offline phase clusters the entire set of
training data into multiple subsets and trains a specialized SE model (termed
component SE model) with each subset. The online phase selects the most
suitable component SE model to perform the enhancement. Furthermore, two
selection strategies were developed: selection based on the quality score (QS)
and selection based on the quality embedding (QE). Both QS and QE were obtained
using a Quality-Net, a non-intrusive quality assessment network. Experimental
results confirmed that the proposed ZMOS approach can achieve better
performance in both seen and unseen noise types compared to the baseline
systems and other model selection systems, which indicates the effectiveness of
the proposed approach in providing robust SE performance.

    

### [[2101.08387] A Survey on Ensemble Learning under the Era of Deep Learning](http://arxiv.org/abs/2101.08387)


  Due to the dominant position of deep learning (mostly deep neural networks)
in various artificial intelligence applications, recently, ensemble learning
based on deep neural networks (ensemble deep learning) has shown significant
performances in improving the generalization of learning system. However, since
modern deep neural networks usually have millions to billions of parameters,
the time and space overheads for training multiple base deep learners and
testing with the ensemble deep learner are far greater than that of traditional
ensemble learning. Though several algorithms of fast ensemble deep learning
have been proposed to promote the deployment of ensemble deep learning in some
applications, further advances still need to be made for many applications in
specific fields, where the developing time and computing resources are usually
restricted or the data to be processed is of large dimensionality. An urgent
problem needs to be solved is how to take the significant advantages of
ensemble deep learning while reduce the required time and space overheads so
that many more applications in specific fields can benefit from it. For the
alleviation of this problem, it is essential to know about how ensemble
learning has developed under the era of deep learning. Thus, in this article,
we present discussions focusing on data analyses of published works,
methodologies, recent advances and unattainability of traditional ensemble
learning and ensemble deep learning. We hope this article will be helpful to
realize the technical challenges faced by future developments of ensemble
learning under the era of deep learning.

    

### [[2102.01496] Gaussian Experts Selection using Graphical Models](http://arxiv.org/abs/2102.01496)


  Local approximations are popular methods to scale Gaussian processes (GPs) to
big data. Local approximations reduce time complexity by dividing the original
dataset into subsets and training a local expert on each subset. Aggregating
the experts' prediction is done assuming either conditional dependence or
independence between the experts. Imposing the \emph{conditional independence
assumption} (CI) between the experts renders the aggregation of different
expert predictions time efficient at the cost of poor uncertainty
quantification. On the other hand, modeling dependent experts can provide
precise predictions and uncertainty quantification at the expense of
impractically high computational costs. By eliminating weak experts via a
theory-guided expert selection step, we substantially reduce the computational
cost of aggregating dependent experts while ensuring calibrated uncertainty
quantification. We leverage techniques from the literature on undirected
graphical models, using sparse precision matrices that encode conditional
dependencies between experts to select the most important experts. Moreov

    

### [[2102.06408] Supervised training of spiking neural networks for robust deployment on mixed-signal neuromorphic processors](http://arxiv.org/abs/2102.06408)


  Mixed-signal analog/digital circuits emulate spiking neurons and synapses
with extremely high energy efficiency, an approach known as "neuromorphic
engineering". However, analog circuits are sensitive to process-induced
variation among transistors in a chip ("device mismatch"). For neuromorphic
implementation of Spiking Neural Networks (SNNs), mismatch causes parameter
variation between identically-configured neurons and synapses. Each chip
exhibits a different distribution of neural parameters, causing deployed
networks to respond differently between chips. Current solutions to mitigate
mismatch based on per-chip calibration or on-chip learning entail increased
design complexity, area and cost, making deployment of neuromorphic devices
expensive and difficult. Here we present a supervised learning approach that
produces SNNs with high robustness to mismatch and other common sources of
noise. Our method trains SNNs to perform temporal classification tasks by
mimicking a pre-trained dynamical system, using a local learning rule from
non-linear control theory. We demonstrate our method on two tasks requiring
memory, and measure the robustness of our approach to several forms of noise
and mismatch. We show that our approach is more robust than common alternatives
for training SNNs. Our method provides robust deployment of pre-trained
networks on mixed-signal neuromorphic hardware, without requiring per-device
training or calibration.

    

### [[2102.06810] Understanding self-supervised Learning Dynamics without Contrastive Pairs](http://arxiv.org/abs/2102.06810)


  While contrastive approaches of self-supervised learning (SSL) learn
representations by minimizing the distance between two augmented views of the
same data point (positive pairs) and maximizing views from different data
points (negative pairs), recent \emph{non-contrastive} SSL (e.g., BYOL and
SimSiam) show remarkable performance {\it without} negative pairs, with an
extra learnable predictor and a stop-gradient operation. A fundamental question
arises: why do these methods not collapse into trivial representations? We
answer this question via a simple theoretical study and propose a novel
approach, DirectPred, that \emph{directly} sets the linear predictor based on
the statistics of its inputs, without gradient training. On ImageNet, it
performs comparably with more complex two-layer non-linear predictors that
employ BatchNorm and outperforms a linear predictor by $2.5\%$ in 300-epoch
training (and $5\%$ in 60-epoch). DirectPred is motivated by our theoretical
study of the nonlinear learning dynamics of non-contrastive SSL in simple
linear networks. Our study yields conceptual insights into how non-contrastive
SSL methods learn, how they avoid representational collapse, and how multiple
factors, like predictor networks, stop-gradients, exponential moving averages,
and weight decay all come into play. Our simple theory recapitulates the
results of real-world ablation studies in both STL-10 and ImageNet. Code is
released this https URL.

    

### [[2102.07053] Linear Convergence in Federated Learning: Tackling Client Heterogeneity and Sparse Gradients](http://arxiv.org/abs/2102.07053)


  We consider a standard federated learning (FL) architecture where a group of
clients periodically coordinate with a central server to train a statistical
model. We develop a general algorithmic framework called FedLin to tackle some
of the key challenges intrinsic to FL, namely objective heterogeneity, systems
heterogeneity, and infrequent and imprecise communication. Our framework is
motivated by the observation that under these challenges, various existing FL
algorithms suffer from a fundamental speed-accuracy conflict: they either
guarantee linear convergence but to an incorrect point, or convergence to the
global minimum but at a sub-linear rate, i.e., fast convergence comes at the
expense of accuracy. In contrast, when the clients' local loss functions are
smooth and strongly convex, we show that FedLin guarantees linear convergence
to the global minimum, despite arbitrary objective and systems heterogeneity.
We then establish matching upper and lower bounds on the convergence rate of
FedLin that highlight the effects of intermittent communication. Finally, we
show that FedLin preserves linear convergence rates under aggressive gradient
sparsification, and quantify the effect of the compression level on the
convergence rate. Our work is the first to provide tight linear convergence
rate guarantees, and constitutes the first comprehensive analysis of gradient
sparsification in FL.

    

### [[2102.11872] Dont Just Divide; Polarize and Conquer!](http://arxiv.org/abs/2102.11872)


  In data containing heterogeneous subpopulations, classification performance
benefits from incorporating the knowledge of cluster structure in the
classifier. Previous methods for such combined clustering and classification
are either 1) classifier-specific and not generic, or 2) independently perform
clustering and classifier training, which may not form clusters that can
potentially benefit classifier performance. The question of how to perform
clustering to improve the performance of classifiers trained on the clusters
has received scant attention in previous literature, despite its importance in
several real-world applications. In this paper, we design a simple and
efficient classification algorithm called Clustering Aware Classification
(CAC), to find clusters that are well suited for being used as training
datasets by classifiers for each underlying subpopulation. Our experiments on
synthetic and real benchmark datasets demonstrate the efficacy of CAC over
previous methods for combined clustering and classification.

    

### [[2104.08821] SimCSE: Simple Contrastive Learning of Sentence Embeddings](http://arxiv.org/abs/2104.08821)


  This paper presents SimCSE, a simple contrastive learning framework that
greatly advances the state-of-the-art sentence embeddings. We first describe an
unsupervised approach, which takes an input sentence and predicts itself in a
contrastive objective, with only standard dropout used as noise. This simple
method works surprisingly well, performing on par with previous supervised
counterparts. We hypothesize that dropout acts as minimal data augmentation and
removing it leads to a representation collapse. Then, we incorporate annotated
pairs from natural language inference datasets into our contrastive learning
framework, by using "entailment" pairs as positives and "contradiction" pairs
as hard negatives. We evaluate SimCSE on standard semantic textual similarity
(STS) tasks, and our unsupervised and supervised models using BERT-base achieve
an average of 76.3% and 81.6% Spearman's correlation respectively, a 4.2 and
2.2 points improvement compared to previous best results. We also show -- both
theoretically and empirically -- that contrastive learning objective
regularizes pre-trained embeddings' anisotropic space to be more uniform, and
it better aligns positive pairs when supervised signals are available.

    

### [[2104.10201] Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020](http://arxiv.org/abs/2104.10201)


  This paper presents the results and insights from the black-box optimization
(BBO) challenge at NeurIPS 2020 which ran from July-October, 2020. The
challenge emphasized the importance of evaluating derivative-free optimizers
for tuning the hyperparameters of machine learning models. This was the first
black-box optimization challenge with a machine learning emphasis. It was based
on tuning (validation set) performance of standard machine learning models on
real datasets. This competition has widespread impact as black-box optimization
(e.g., Bayesian optimization) is relevant for hyperparameter tuning in almost
every machine learning project as well as many applications outside of machine
learning. The final leaderboard was determined using the optimization
performance on held-out (hidden) objective functions, where the optimizers ran
without human intervention. Baselines were set using the default settings of
several open-source black-box optimization packages as well as random search.

    

### [[2104.12546] The Effects of Air Quality on the Spread of the COVID-19 Pandemic in Italy: An Artificial Intelligence Approach](http://arxiv.org/abs/2104.12546)


  The COVID-19 pandemic considerably affects public health systems around the
world. The lack of knowledge about the virus, the extension of this phenomenon,
and the speed of the evolution of the infection are all factors that highlight
the necessity of employing new approaches to study these events. Artificial
intelligence techniques may be useful in analyzing data related to areas
affected by the virus. The aim of this work is to investigate any possible
relationships between air quality and confirmed cases of COVID-19 in Italian
districts. Specifically, we report an analysis of the correlation between daily
COVID-19 cases and environmental factors, such as temperature, relative
humidity, and atmospheric pollutants. Our analysis confirms a significant
association of some environmental parameters with the spread of the virus. This
suggests that machine learning models trained on the environmental parameters
to predict the number of future infected cases may be accurate. Predictive
models may be useful for helping institutions in making decisions for
protecting the population and contrasting the pandemic.

    

### [[2104.13626] Self-Bounding Majority Vote Learning Algorithms by the Direct Minimization of a Tight PAC-Bayesian C-Bound](http://arxiv.org/abs/2104.13626)


  In the PAC-Bayesian literature, the C-Bound refers to an insightful relation
between the risk of a majority vote classifier (under the zero-one loss) and
the first two moments of its margin (i.e., the expected margin and the voters'
diversity). Until now, learning algorithms developed in this framework minimize
the empirical version of the C-Bound, instead of explicit PAC-Bayesian
generalization bounds. In this paper, by directly optimizing PAC-Bayesian
guarantees on the C-Bound, we derive self-bounding majority vote learning
algorithms. Moreover, our algorithms based on gradient descent are scalable and
lead to accurate predictors paired with non-vacuous guarantees.

    

### [[2105.01060] Curious Representation Learning for Embodied Intelligence](http://arxiv.org/abs/2105.01060)


  Self-supervised representation learning has achieved remarkable success in
recent years. By subverting the need for supervised labels, such approaches are
able to utilize the numerous unlabeled images that exist on the Internet and in
photographic datasets. Yet to build truly intelligent agents, we must construct
representation learning algorithms that can learn not only from datasets but
also learn from environments. An agent in a natural environment will not
typically be fed curated data. Instead, it must explore its environment to
acquire the data it will learn from. We propose a framework, curious
representation learning (CRL), which jointly learns a reinforcement learning
policy and a visual representation model. The policy is trained to maximize the
error of the representation learner, and in doing so is incentivized to explore
its environment. At the same time, the learned representation becomes stronger
and stronger as the policy feeds it ever harder data to learn from. Our learned
representations enable promising transfer to downstream navigation tasks,
performing better than or comparably to ImageNet pretraining without using any
supervision at all. In addition, despite being trained in simulation, our
learned representations can obtain interpretable results on real images. Code
is available at this https URL.

    

### [[2105.03178] Graph Entropy Guided Node Embedding Dimension Selection for Graph Neural Networks](http://arxiv.org/abs/2105.03178)


  Graph representation learning has achieved great success in many areas,
including e-commerce, chemistry, biology, etc. However, the fundamental problem
of choosing the appropriate dimension of node embedding for a given graph still
remains unsolved. The commonly used strategies for Node Embedding Dimension
Selection (NEDS) based on grid search or empirical knowledge suffer from heavy
computation and poor model performance. In this paper, we revisit NEDS from the
perspective of minimum entropy principle. Subsequently, we propose a novel
Minimum Graph Entropy (MinGE) algorithm for NEDS with graph data. To be
specific, MinGE considers both feature entropy and structure entropy on graphs,
which are carefully designed according to the characteristics of the rich
information in them. The feature entropy, which assumes the embeddings of
adjacent nodes to be more similar, connects node features and link topology on
graphs. The structure entropy takes the normalized degree as basic unit to
further measure the higher-order structure of graphs. Based on them, we design
MinGE to directly calculate the ideal node embedding dimension for any graph.
Finally, comprehensive experiments with popular Graph Neural Networks (GNNs) on
benchmark datasets demonstrate the effectiveness and generalizability of our
proposed MinGE.

    

### [[2105.03491] Uniform Convergence, Adversarial Spheres and a Simple Remedy](http://arxiv.org/abs/2105.03491)


  Previous work has cast doubt on the general framework of uniform convergence
and its ability to explain generalization in neural networks. By considering a
specific dataset, it was observed that a neural network completely
misclassifies a projection of the training data (adversarial set), rendering
any existing generalization bound based on uniform convergence vacuous. We
provide an extensive theoretical investigation of the previously studied data
setting through the lens of infinitely-wide models. We prove that the Neural
Tangent Kernel (NTK) also suffers from the same phenomenon and we uncover its
origin. We highlight the important role of the output bias and show
theoretically as well as empirically how a sensible choice completely mitigates
the problem. We identify sharp phase transitions in the accuracy on the
adversarial set and study its dependency on the training sample size. As a
result, we are able to characterize critical sample sizes beyond which the
effect disappears. Moreover, we study decompositions of a neural network into a
clean and noisy part by considering its canonical decomposition into its
different eigenfunctions and show empirically that for too small bias the
adversarial phenomenon still persists.

    

### [[2105.10766] Embedding Information onto a Dynamical System](http://arxiv.org/abs/2105.10766)


  The celebrated Takens' embedding theorem concerns embedding an attractor of a
dynamical system in a Euclidean space of appropriate dimension through a
generic delay-observation map. The embedding also establishes a topological
conjugacy. In this paper, we show how an arbitrary sequence can be mapped into
another space as an attractive solution of a nonautonomous dynamical system.
Such mapping also entails a topological conjugacy and an embedding between the
sequence and the attractive solution spaces. This result is not a
generalization of Takens embedding theorem but helps us understand what exactly
is required by discrete-time state space models widely used in applications to
embed an external stimulus onto its solution space. Our results settle another
basic problem concerning the perturbation of an autonomous dynamical system. We
describe what exactly happens to the dynamics when exogenous noise perturbs
continuously a local irreducible attracting set (such as a stable fixed point)
of a discrete-time autonomous dynamical system.

    

### [[2011.03293] On the Stability Properties and the Optimization Landscape of Training Problems with Squared Loss for Neural Networks and General Nonlinear Conic Approximation Schemes](http://arxiv.org/abs/2011.03293)


  We study the optimization landscape and the stability properties of training
problems with squared loss for neural networks and general nonlinear conic
approximation schemes. It is demonstrated that, if a nonlinear conic
approximation scheme is considered that is (in an appropriately defined sense)
more expressive than a classical linear approximation approach and if there
exist unrealizable label vectors, then a training problem with squared loss is
necessarily unstable in the sense that its solution set depends discontinuously
on the label vector in the training data. We further prove that the same
effects that are responsible for these instability properties are also the
reason for the emergence of saddle points and spurious local minima, which may
be arbitrarily far away from global solutions, and that neither the instability
of the training problem nor the existence of spurious local minima can, in
general, be overcome by adding a regularization term to the objective function
that penalizes the size of the parameters in the approximation scheme. The
latter results are shown to be true regardless of whether the assumption of
realizability is satisfied or not. We demonstrate that our analysis in
particular applies to training problems for free-knot interpolation schemes and
deep and shallow neural networks with variable widths that involve an arbitrary
mixture of various activation functions (e.g., binary, sigmoid, tanh, arctan,
soft-sign, ISRU, soft-clip, SQNL, ReLU, leaky ReLU, soft-plus, bent identity,
SILU, ISRLU, and ELU). In summary, the findings of this paper illustrate that
the improved approximation properties of neural networks and general nonlinear
conic approximation instruments are linked in a direct and quantifiable way to
undesirable properties of the optimization problems that have to be solved in
order to train them.

    

### [[2108.13342] DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](http://arxiv.org/abs/2108.13342)


  Deep Neural Networks (DNNs) have emerged as the core enabler of many major
applications on mobile devices. To achieve high accuracy, DNN models have
become increasingly deep with hundreds or even thousands of operator layers,
leading to high memory and computational requirements for inference. Operator
fusion (or kernel/layer fusion) is key optimization in many state-of-the-art
DNN execution frameworks, such as TensorFlow, TVM, and MNN. However, these
frameworks usually adopt fusion approaches based on certain patterns that are
too restrictive to cover the diversity of operators and layer connections.
Polyhedral-based loop fusion techniques, on the other hand, work on a low-level
view of the computation without operator-level information, and can also miss
potential fusion opportunities. To address this challenge, this paper proposes
a novel and extensive loop fusion framework called DNNFusion. The basic idea of
this work is to work at an operator view of DNNs, but expand fusion
opportunities by developing a classification of both individual operators and
their combinations. In addition, DNNFusion includes 1) a novel
mathematical-property-based graph rewriting framework to reduce evaluation
costs and facilitate subsequent operator fusion, 2) an integrated fusion plan
generation that leverages the high-level analysis and accurate light-weight
profiling, and 3) additional optimizations during fusion code generation.
DNNFusion is extensively evaluated on 15 DNN models with varied types of tasks,
model sizes, and layer counts. The evaluation results demonstrate that
DNNFusion finds up to 8.8x higher fusion opportunities, outperforms four
state-of-the-art DNN execution frameworks with 9.3x speedup. The memory
requirement reduction and speedups can enable the execution of many of the
target models on mobile devices and even make them part of a real-time
application.

    

### [[2108.13922] Stockade: Hardware Hardening for Distributed Trusted Sandboxes](http://arxiv.org/abs/2108.13922)


  The widening availability of hardware-based trusted execution environments
(TEEs) has been accelerating the adaptation of new applications using TEEs.
Recent studies showed that a cloud application consists of multiple distributed
software modules provided by mutually distrustful parties. The applications use
multiple TEEs (enclaves) communicating through software-encrypted memory
channels. Such execution model requires bi-directional protection: protecting
the rest of the system from the enclave module with sandboxing and protecting
the enclave module from a third-part module and operating systems. However, the
current TEE model, such as Intel SGX, cannot efficiently represent such
distributed sandbox applications. To overcome the lack of hardware supports for
sandboxed TEEs, this paper proposes an extended enclave model called Stockade,
which supports distributed sandboxes hardened by hardware. Stockade proposes
new three key techniques. First, it extends the hardware-based memory isolation
in SGX to confine a user software module only within its enclave. Second, it
proposes a trusted monitor enclave that filters and validates systems calls
from enclaves. Finally, it allows hardware-protected memory sharing between a
pair of enclaves for efficient protected communication without software-based
encryption. Using an emulated SGX platform with the proposed extensions, this
paper shows that distributed sandbox applications can be effectively supported
with small changes of SGX hardware.

    

### [[2009.14381] AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators](http://arxiv.org/abs/2009.14381)


  Adopting FPGA as an accelerator in datacenters is becoming mainstream for
customized computing, but the fact that FPGAs are hard to program creates a
steep learning curve for software programmers. Even with the help of high-level
synthesis (HLS), accelerator designers still have to manually perform code
reconstruction and cumbersome parameter tuning to achieve the optimal
performance. While many learning models have been leveraged by existing work to
automate the design of efficient accelerators, the unpredictability of modern
HLS tools becomes a major obstacle for them to maintain high accuracy. To
address this problem, we propose an automated DSE framework-AutoDSE- that
leverages a bottleneck-guided coordinate optimizer to systematically find a
better design point. AutoDSE detects the bottleneck of the design in each step
and focuses on high-impact parameters to overcome it. The experimental results
show that AutoDSE is able to identify the design point that achieves, on the
geometric mean, 19.9x speedup over one CPU core for Machsuite and Rodinia
benchmarks. Compared to the manually optimized HLS vision kernels in Xilinx
Vitis libraries, AutoDSE can reduce their optimization pragmas by 26.38x while
achieving similar performance. With less than one optimization pragma per
design on average, we are making progress towards democratizing customizable
computing by enabling software programmers to design efficient FPGA
accelerators.

    

### [[2108.13449] Population Protocols: Beyond Runtime Analysis](http://arxiv.org/abs/2108.13449)


  I survey our recent work on the verification of population protocols and
their state complexity.

    

### [[2108.13502] Generalizing Weighted Trees: A Bridge from Bitcoin to GHOST](http://arxiv.org/abs/2108.13502)


  Despite the tremendous interest in cryptocurrencies like Bitcoin and Ethereum
today, many aspects of the underlying consensus protocols are poorly
understood. Therefore, the search for protocols that improve either throughput
or security (or both) continues. Bitcoin always selects the longest chain
(i.e., the one with most work). Forks may occur when two miners extend the same
block simultaneously, and the frequency of forks depends on how fast blocks are
propagated in the network. In the GHOST protocol, used by Ethereum, all blocks
involved in the fork contribute to the security. However, the greedy chain
selection rule of GHOST does not consider the full information available in the
block tree, which has led to some concerns about its security.
This paper introduces a new family of protocols, called Medium, which takes
the structure of the whole block tree into account, by weighting blocks
differently according to their depths. Bitcoin and GHOST result as special
cases. This protocol leads to new insights about the security of Bitcoin and
GHOST and paves the way for developing network- and application-specific
protocols, in which the influence of forks on the chain-selection process can
be controlled. It is shown that almost all protocols in this family achieve
strictly greater throughput than Bitcoin (at the same security level) and
resist attacks that can be mounted against GHOST.

    

### [[2108.13521] ExaWorks: Workflows for Exascale](http://arxiv.org/abs/2108.13521)


  Exascale computers will offer transformative capabilities to combine
data-driven and learning-based approaches with traditional simulation
applications to accelerate scientific discovery and insight. These software
combinations and integrations, however, are difficult to achieve due to
challenges of coordination and deployment of heterogeneous software components
on diverse and massive platforms. We present the ExaWorks project, which can
address many of these challenges: ExaWorks is leading a co-design process to
create a workflow software development Toolkit (SDK) consisting of a wide range
of workflow management tools that can be composed and interoperate through
common interfaces. We describe the initial set of tools and interfaces
supported by the SDK, efforts to make them easier to apply to complex science
challenges, and examples of their application to exemplar cases. Furthermore,
we discuss how our project is working with the workflows community, large
computing facilities as well as HPC platform vendors to sustainably address the
requirements of workflows at the exascale.

    

### [[2108.13716] A log-linear $(2+5/6)$-approximation algorithm for parallel machine scheduling with a single orthogonal resource](http://arxiv.org/abs/2108.13716)


  As the gap between compute and I/O performance tends to grow, modern
High-Performance Computing (HPC) architectures include a new resource type: an
intermediate persistent fast memory layer, called burst buffers. This is just
one of many kinds of renewable resources which are orthogonal to the processors
themselves, such as network bandwidth or software licenses. Ignoring orthogonal
resources while making scheduling decisions just for processors may lead to
unplanned delays of jobs of which resource requirements cannot be immediately
satisfied. We focus on a classic problem of makespan minimization for
parallel-machine scheduling of independent sequential jobs with additional
requirements on the amount of a single renewable orthogonal resource. We
present an easily-implementable log-linear algorithm that we prove is
$2\frac56$-approximation. In simulation experiments, we compare our algorithm
to standard greedy list-scheduling heuristics and show that, compared to LPT,
resource-based algorithms generate significantly shorter schedules.

    

### [[2108.13871] Building Time-Triggered Schedules for typed-DAG Tasks with alternative implementations](http://arxiv.org/abs/2108.13871)


  Hard real-time systems like image processing, autonomous driving, etc.
require an increasing need of computational power that classical multi-core
platforms can not provide, to fulfill with their timing constraints.
Heterogeneous Instruction Set Architecture (ISA) platforms allow accelerating
real-time workloads on application-specific cores (e.g. GPU, DSP, ASICs) etc.
and are suitable for these applications. In addition, these platforms provide
larger design choices as a given functionnality can be implemented onto several
types of compute elements. HPC-DAG (Heterogeneous Parallel Directed Acyclic
Graph) task model has been recently proposed to capture real-time workload
execution on heterogeneous platforms. It expresses the ISA heterogeneity, and
some specific characteristics of hardware accelerators, as the absence of
preemption or costly preemption, alternative implementations and on-line
conditional execution. In this paper, we propose a time-table scheduling
approach to allocate and schedule a set of HPC-DAG tasks onto a set of
heterogeneous cores, by the mean Integer Linear Programming (ILP). Our design
allows to handle heterogeniety of resources, on-line execution costs, and a
faster solving time, by exploring gradually the design space

    

### [[2102.09166] Latency Modeling of Hyperledger Fabric for Blockchain-enabled IoT Networks](http://arxiv.org/abs/2102.09166)


  Hyperledger Fabric (HLF), one of the most popular private blockchain
platforms, has recently received attention for blockchain-enabled Internet of
things (BC-IoT) networks. However, for IoT devices handling latency-critical
tasks, the additional time spent in HLF has emerged as a new challenge in
BC-IoT networks. In this paper, therefore, we develop an HLF latency model
using the probability distribution fitting method for HLF-based IoT networks.
We first explain the architecture and the transaction flow in HLF, and
structure of an HLF-based IoT network. After implementing real HLF, we capture
the latencies that each transaction experiences for various HLF environments,
and then show that the total latency of HLF can be modeled as a Gamma
distribution. Our HLF latency model is also validated by conducting a
goodness-of-fit test, i.e., KS test. Furthermore, we explore the impacts of
three HLF parameters including transaction generation rate, block size, and
block-generation timeout on the HLF latency. As a result, some HLF design
insights on minimizing the latency are provided for HLF-based IoT networks.

    

### [[2103.02182] Distributed statistical inference with pyhf enabled through funcX](http://arxiv.org/abs/2103.02182)


  In High Energy Physics facilities that provide High Performance Computing
environments provide an opportunity to efficiently perform the statistical
inference required for analysis of data from the Large Hadron Collider, but can
pose problems with orchestration and efficient scheduling. The compute
architectures at these facilities do not easily support the Python compute
model, and the configuration scheduling of batch jobs for physics often
requires expertise in multiple job scheduling services. The combination of the
pure-Python libraries pyhf and funcX reduces the common problem in HEP analyses
of performing statistical inference with binned models, that would
traditionally take multiple hours and bespoke scheduling, to an on-demand
(fitting) "function as a service" that can scalably execute across workers in
just a few minutes, offering reduced time to insight and inference. We
demonstrate execution of a scalable workflow using funcX to simultaneously fit
125 signal hypotheses from a published ATLAS search for new physics using pyhf
with a wall time of under 3 minutes. We additionally show performance
comparisons for other physics analyses with openly published probability models
and argue for a blueprint of fitting as a service systems at HPC centers.

    

### [[2106.00083] Composing Networks of Automated Market Makers](http://arxiv.org/abs/2106.00083)


  Automated market makers (AMMs) are automata that trade electronic assets at
rates set by mathematical formulas. AMMs are usually implemented by smart
contracts on blockchains. In practice, AMMs are often composed: and outputs
from AMMs can be directed into other compatible AMMs. This paper proposes a
mathematical model for AMM composition. We define sequential and parallel
composition operators for AMMs in a way that ensures that AMMs are closed under
composition, in a way that works for "higher-dimensional" AMMs that manage more
than two asset classes, and so the composition of AMMs in "stable" states
remains stable.

    

### [[2108.13414] Astrocytes mediate analogous memory in a multi-layer neuron-astrocytic network](http://arxiv.org/abs/2108.13414)


  Modeling the neuronal processes underlying short-term working memory remains
the focus of many theoretical studies in neuroscience. Here we propose a
mathematical model of spiking neuron network (SNN) demonstrating how a piece of
information can be maintained as a robust activity pattern for several seconds
then completely disappear if no other stimuli come. Such short-term memory
traces are preserved due to the activation of astrocytes accompanying the SNN.
The astrocytes exhibit calcium transients at a time scale of seconds. These
transients further modulate the efficiency of synaptic transmission and, hence,
the firing rate of neighboring neurons at diverse timescales through
gliotransmitter release. We show how such transients continuously encode
frequencies of neuronal discharges and provide robust short-term storage of
analogous information. This kind of short-term memory can keep operative
information for seconds, then completely forget it to avoid overlapping with
forthcoming patterns. The SNN is inter-connected with the astrocytic layer by
local inter-cellular diffusive connections. The astrocytes are activated only
when the neighboring neurons fire quite synchronously, e.g. when an information
pattern is loaded. For illustration, we took greyscale photos of people's faces
where the grey level encoded the level of applied current stimulating the
neurons. The astrocyte feedback modulates (facilitates) synaptic transmission
by varying the frequency of neuronal firing. We show how arbitrary patterns can
be loaded, then stored for a certain interval of time, and retrieved if the
appropriate clue pattern is applied to the input.

    

### [[2108.13454] Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback](http://arxiv.org/abs/2108.13454)


  Dense retrieval systems conduct first-stage retrieval using embedded
representations and simple similarity metrics to match a query to documents.
Its effectiveness depends on encoded embeddings to capture the semantics of
queries and documents, a challenging task due to the shortness and ambiguity of
search queries. This paper proposes ANCE-PRF, a new query encoder that uses
pseudo relevance feedback (PRF) to improve query representations for dense
retrieval. ANCE-PRF uses a BERT encoder that consumes the query and the top
retrieved documents from a dense retrieval model, ANCE, and it learns to
produce better query embeddings directly from relevance labels. It also keeps
the document index unchanged to reduce overhead. ANCE-PRF significantly
outperforms ANCE and other recent dense retrieval systems on several datasets.
Analysis shows that the PRF encoder effectively captures the relevant and
complementary information from PRF documents, while ignoring the noise with its
learned attention mechanism.

    

### [[2108.13487] Want To Reduce Labeling Cost? GPT-3 Can Help](http://arxiv.org/abs/2108.13487)


  Data annotation is a time-consuming and labor-intensive process for many NLP
tasks. Although there exist various methods to produce pseudo data labels, they
are often task-specific and require a decent amount of labeled data to start
with. Recently, the immense language model GPT-3 with 175 billion parameters
has achieved tremendous improvement across many few-shot learning tasks. In
this paper, we explore ways to leverage GPT-3 as a low-cost data labeler to
train other models. We find that, to make the downstream model achieve the same
performance on a variety of NLU and NLG tasks, it costs 50% to 96% less to use
labels from GPT-3 than using labels from humans. Furthermore, we propose a
novel framework of combining pseudo labels from GPT-3 with human labels, which
leads to even better performance with limited labeling budget. These results
present a cost-effective data labeling methodology that is generalizable to
many practical applications.

    

### [[2108.13592] Zero Shot on the Cold-Start Problem: Model-Agnostic Interest Learning for Recommender Systems](http://arxiv.org/abs/2108.13592)


  User behavior has been validated to be effective in revealing personalized
preferences for commercial recommendations. However, few user-item interactions
can be collected for new users, which results in a null space for their
interests, i.e., the cold-start dilemma. In this paper, a two-tower framework,
namely, the model-agnostic interest learning (MAIL) framework, is proposed to
address the cold-start recommendation (CSR) problem for recommender systems. In
MAIL, one unique tower is constructed to tackle the CSR from a zero-shot view,
and the other tower focuses on the general ranking task. Specifically, the
zero-shot tower first performs cross-modal reconstruction with dual
auto-encoders to obtain virtual behavior data from highly aligned hidden
features for new users; and the ranking tower can then output recommendations
for users based on the completed data by the zero-shot tower. Practically, the
ranking tower in MAIL is model-agnostic and can be implemented with any
embedding-based deep models. Based on the co-training of the two towers, the
MAIL presents an end-to-end method for recommender systems that shows an
incremental performance improvement. The proposed method has been successfully
deployed on the live recommendation system of NetEase Cloud Music to achieve a
click-through rate improvement of 13% to 15% for millions of users. Offline
experiments on real-world datasets also show its superior performance in CSR.
Our code is available.

    

### [[2108.13621] Spike time displacement based error backpropagation in convolutional spiking neural networks](http://arxiv.org/abs/2108.13621)


  We recently proposed the STiDi-BP algorithm, which avoids backward recursive
gradient computation, for training multi-layer spiking neural networks (SNNs)
with single-spike-based temporal coding. The algorithm employs a linear
approximation to compute the derivative of the spike latency with respect to
the membrane potential and it uses spiking neurons with piecewise linear
postsynaptic potential to reduce the computational cost and the complexity of
neural processing. In this paper, we extend the STiDi-BP algorithm to employ it
in deeper and convolutional architectures. The evaluation results on the image
classification task based on two popular benchmarks, MNIST and Fashion-MNIST
datasets with the accuracies of respectively 99.2% and 92.8%, confirm that this
algorithm has been applicable in deep SNNs. Another issue we consider is the
reduction of memory storage and computational cost. To do so, we consider a
convolutional SNN (CSNN) with two sets of weights: real-valued weights that are
updated in the backward pass and their signs, binary weights, that are employed
in the feedforward process. We evaluate the binary CSNN on two datasets of
MNIST and Fashion-MNIST and obtain acceptable performance with a negligible
accuracy drop with respect to real-valued weights (about $0.6%$ and $0.8%$
drops, respectively).

    

### [[2108.13653] Explaining Classes through Word Attribution](http://arxiv.org/abs/2108.13653)


  In recent years, several methods have been proposed for explaining individual
predictions of deep learning models, yet there has been little study of how to
aggregate these predictions to explain how such models view classes as a whole
in text classification tasks. In this work, we propose a method for explaining
classes using deep learning models and the Integrated Gradients feature
attribution technique by aggregating explanations of individual examples in
text classification to general descriptions of the classes. We demonstrate the
approach on Web register (genre) classification using the XML-R model and the
Corpus of Online Registers of English (CORE), finding that the method
identifies plausible and discriminative keywords characterizing all but the
smallest class.

    

### [[2108.13679] Task-Oriented Dialogue System as Natural Language Generation](http://arxiv.org/abs/2108.13679)


  In this paper, we propose to formulate the task-oriented dialogue system as
the purely natural language generation task, so as to fully leverage the
large-scale pre-trained models like GPT-2 and simplify complicated
delexicalization prepossessing. However, directly applying this method heavily
suffers from the dialogue entity inconsistency caused by the removal of
delexicalized tokens, as well as the catastrophic forgetting problem of the
pre-trained model during fine-tuning, leading to unsatisfactory performance. To
alleviate these problems, we design a novel GPT-Adapter-CopyNet network, which
incorporates the lightweight adapter and CopyNet modules into GPT-2 to achieve
better performance on transfer learning and dialogue entity generation.
Experimental results conducted on the DSTC8 Track 1 benchmark and MultiWOZ
dataset demonstrate that our proposed approach significantly outperforms
baseline models with a remarkable performance on automatic and human
evaluations.

    

### [[2108.13700] TNNT: The Named Entity Recognition Toolkit](http://arxiv.org/abs/2108.13700)


  Extraction of categorised named entities from text is a complex task given
the availability of a variety of Named Entity Recognition (NER) models and the
unstructured information encoded in different source document formats.
Processing the documents to extract text, identifying suitable NER models for a
task, and obtaining statistical information is important in data analysis to
make informed decisions. This paper presents TNNT, a toolkit that automates the
extraction of categorised named entities from unstructured information encoded
in source documents, using diverse state-of-the-art Natural Language Processing
(NLP) tools and NER models. TNNT integrates 21 different NER models as part of
a Knowledge Graph Construction Pipeline (KGCP) that takes a document set as
input and processes it based on the defined settings, applying the selected
blocks of NER models to output the results. The toolkit generates all results
with an integrated summary of the extracted entities, enabling enhanced data
analysis to support the KGCP, and also, to aid further NLP tasks.

    

### [[2108.13741] Monolingual versus Multilingual BERTology for Vietnamese Extractive Multi-Document Summarization](http://arxiv.org/abs/2108.13741)


  Recent researches have demonstrated that BERT shows potential in a wide range
of natural language processing tasks. It is adopted as an encoder for many
state-of-the-art automatic summarizing systems, which achieve excellent
performance. However, so far, there is not much work done for Vietnamese. In
this paper, we showcase how BERT can be implemented for extractive text
summarization in Vietnamese. We introduce a novel comparison between different
multilingual and monolingual BERT models. The experiment results indicate that
monolingual models produce promising results compared to other multilingual
models and previous text summarizing models for Vietnamese.

    

### [[2108.13744] The Horn Non-Clausal Class and its Polynomiality](http://arxiv.org/abs/2108.13744)


  The expressiveness of propositional non-clausal (NC) formulas is
exponentially richer than that of clausal formulas. Yet, clausal efficiency
outperforms non-clausal one. Indeed, a major weakness of the latter is that,
while Horn clausal formulas, along with Horn algorithms, are crucial for the
high efficiency of clausal reasoning, no Horn-like formulas in non-clausal form
had been proposed. To overcome such weakness, we define the hybrid class
$\mathbb{H_{NC}}$ of Horn Non-Clausal (Horn-NC) formulas, by adequately lifting
the Horn pattern to NC form, and argue that $\mathbb{H_{NC}}$, along with
future Horn-NC algorithms, shall increase non-clausal efficiency just as the
Horn class has increased clausal efficiency. Secondly, we: (i) give the
compact, inductive definition of $\mathbb{H_{NC}}$; (ii) prove that
syntactically $\mathbb{H_{NC}}$ subsumes the Horn class but semantically both
classes are equivalent, and (iii) characterize the non-clausal formulas
belonging to $\mathbb{H_{NC}}$. Thirdly, we define the Non-Clausal
Unit-Resolution calculus, $UR_{NC}$, and prove that it checks the
satisfiability of $\mathbb{H_{NC}}$ in polynomial time. This fact, to our
knowledge, makes $\mathbb{H_{NC}}$ the first characterized polynomial class in
NC reasoning. Finally, we prove that $\mathbb{H_{NC}}$ is linearly
recognizable, and also that it is both strictly succincter and exponentially
richer than the Horn class. We discuss that in NC automated reasoning, e.g.
satisfiability solving, theorem proving, logic programming, etc., can directly
benefit from $\mathbb{H_{NC}}$ and $UR_{NC}$ and that, as a by-product of its
proved properties, $\mathbb{H_{NC}}$ arises as a new alternative to analyze
Horn functions and implication systems.

    

### [[2108.13766] The five Is: Key principles for interpretable and safe conversational AI](http://arxiv.org/abs/2108.13766)


  In this position paper, we present five key principles, namely
interpretability, inherent capability to explain, independent data, interactive
learning, and inquisitiveness, for the development of conversational AI that,
unlike the currently popular black box approaches, is transparent and
accountable. At present, there is a growing concern with the use of black box
statistical language models: While displaying impressive average performance,
such systems are also prone to occasional spectacular failures, for which there
is no clear remedy. In an effort to initiate a discussion on possible
alternatives, we outline and exemplify how our five principles enable the
development of conversational AI systems that are transparent and thus safer
for use. We also present some of the challenges inherent in the implementation
of those principles.

    

### [[2108.13772] Artificial Intelligence Algorithms for Natural Language Processing and the Semantic Web Ontology Learning](http://arxiv.org/abs/2108.13772)


  Evolutionary clustering algorithms have considered as the most popular and
widely used evolutionary algorithms for minimising optimisation and practical
problems in nearly all fields. In this thesis, a new evolutionary clustering
algorithm star (ECA*) is proposed. Additionally, a number of experiments were
conducted to evaluate ECA* against five state-of-the-art approaches. For this,
32 heterogeneous and multi-featured datasets were used to examine their
performance using internal and external clustering measures, and to measure the
sensitivity of their performance towards dataset features in the form of
operational framework. The results indicate that ECA* overcomes its competitive
techniques in terms of the ability to find the right clusters. Based on its
superior performance, exploiting and adapting ECA* on the ontology learning had
a vital possibility. In the process of deriving concept hierarchies from
corpora, generating formal context may lead to a time-consuming process.
Therefore, formal context size reduction results in removing uninterested and
erroneous pairs, taking less time to extract the concept lattice and concept
hierarchies accordingly. In this premise, this work aims to propose a framework
to reduce the ambiguity of the formal context of the existing framework using
an adaptive version of ECA*. In turn, an experiment was conducted by applying
385 sample corpora from Wikipedia on the two frameworks to examine the
reduction of formal context size, which leads to yield concept lattice and
concept hierarchy. The resulting lattice of formal context was evaluated to the
original one using concept lattice-invariants. Accordingly, the homomorphic
between the two lattices preserves the quality of resulting concept hierarchies
by 89% in contrast to the basic ones, and the reduced concept lattice inherits
the structural relation of the original one.

    

### [[2108.13796] Addressing the IEEE AV Test Challenge with Scenic and VerifAI](http://arxiv.org/abs/2108.13796)


  This paper summarizes our formal approach to testing autonomous vehicles
(AVs) in simulation for the IEEE AV Test Challenge. We demonstrate a systematic
testing framework leveraging our previous work on formally-driven simulation
for intelligent cyber-physical systems. First, to model and generate
interactive scenarios involving multiple agents, we used Scenic, a
probabilistic programming language for specifying scenarios. A Scenic program
defines an abstract scenario as a distribution over configurations of physical
objects and their behaviors over time. Sampling from an abstract scenario
yields many different concrete scenarios which can be run as test cases for the
AV. Starting from a Scenic program encoding an abstract driving scenario, we
can use the VerifAI toolkit to search within the scenario for failure cases
with respect to multiple AV evaluation metrics. We demonstrate the
effectiveness of our testing framework by identifying concrete failure
scenarios for an open-source autopilot, Apollo, starting from a variety of
realistic traffic scenarios.

    

### [[2108.13828] PACE: Posthoc Architecture-Agnostic Concept Extractor for Explaining CNNs](http://arxiv.org/abs/2108.13828)


  Deep CNNs, though have achieved the state of the art performance in image
classification tasks, remain a black-box to a human using them. There is a
growing interest in explaining the working of these deep models to improve
their trustworthiness. In this paper, we introduce a Posthoc
Architecture-agnostic Concept Extractor (PACE) that automatically extracts
smaller sub-regions of the image called concepts relevant to the black-box
prediction. PACE tightly integrates the faithfulness of the explanatory
framework to the black-box model. To the best of our knowledge, this is the
first work that extracts class-specific discriminative concepts in a posthoc
manner automatically. The PACE framework is used to generate explanations for
two different CNN architectures trained for classifying the AWA2 and
Imagenet-Birds datasets. Extensive human subject experiments are conducted to
validate the human interpretability and consistency of the explanations
extracted by PACE. The results from these experiments suggest that over 72% of
the concepts extracted by PACE are human interpretable.

    

### [[2108.13844] Fiducial marker recovery and detection from severely truncated data in navigation assisted spine surgery](http://arxiv.org/abs/2108.13844)


  Fiducial markers are commonly used in navigation assisted minimally invasive
spine surgery (MISS) and they help transfer image coordinates into real world
coordinates. In practice, these markers might be located outside the
field-of-view (FOV), due to the limited detector sizes of C-arm cone-beam
computed tomography (CBCT) systems used in intraoperative surgeries. As a
consequence, reconstructed markers in CBCT volumes suffer from artifacts and
have distorted shapes, which sets an obstacle for navigation. In this work, we
propose two fiducial marker detection methods: direct detection from distorted
markers (direct method) and detection after marker recovery (recovery method).
For direct detection from distorted markers in reconstructed volumes, an
efficient automatic marker detection method using two neural networks and a
conventional circle detection algorithm is proposed. For marker recovery, a
task-specific learning strategy is proposed to recover markers from severely
truncated data. Afterwards, a conventional marker detection algorithm is
applied for position detection. The two methods are evaluated on simulated data
and real data, both achieving a marker registration error smaller than 0.2 mm.
Our experiments demonstrate that the direct method is capable of detecting
distorted markers accurately and the recovery method with task-specific
learning has high robustness and generalizability on various data sets. In
addition, the task-specific learning is able to reconstruct other structures of
interest accurately, e.g. ribs for image-guided needle biopsy, from severely
truncated data, which empowers CBCT systems with new potential applications.

    

### [[2108.13854] Contrastive Domain Adaptation for Question Answering using Limited Text Corpora](http://arxiv.org/abs/2108.13854)


  Question generation has recently shown impressive results in customizing
question answering (QA) systems to new domains. These approaches circumvent the
need for manually annotated training data from the new domain and, instead,
generate synthetic question-answer pairs that are used for training. However,
existing methods for question generation rely on large amounts of synthetically
generated datasets and costly computational resources, which render these
techniques widely inaccessible when the text corpora is of limited size. This
is problematic as many niche domains rely on small text corpora, which
naturally restricts the amount of synthetic data that can be generated. In this
paper, we propose a novel framework for domain adaptation called contrastive
domain adaptation for QA (CAQA). Specifically, CAQA combines techniques from
question generation and domain-invariant learning to answer out-of-domain
questions in settings with limited text corpora. Here, we train a QA system on
both source data and generated data from the target domain with a contrastive
adaptation loss that is incorporated in the training objective. By combining
techniques from question generation and domain-invariant learning, our model
achieved considerable improvements compared to state-of-the-art baselines.

    

### [[2108.13875] When Retriever-Reader Meets Scenario-Based Multiple-Choice Questions](http://arxiv.org/abs/2108.13875)


  Scenario-based question answering (SQA) requires retrieving and reading
paragraphs from a large corpus to answer a question which is contextualized by
a long scenario description. Since a scenario contains both keyphrases for
retrieval and much noise, retrieval for SQA is extremely difficult. Moreover,
it can hardly be supervised due to the lack of relevance labels of paragraphs
for SQA. To meet the challenge, in this paper we propose a joint
retriever-reader model called JEEVES where the retriever is implicitly
supervised only using QA labels via a novel word weighting mechanism. JEEVES
significantly outperforms a variety of strong baselines on multiple-choice
questions in three SQA datasets.

    

### [[2108.13897] mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset](http://arxiv.org/abs/2108.13897)


  The MS MARCO ranking dataset has been widely used for training deep learning
models for IR tasks, achieving considerable effectiveness on diverse zero-shot
scenarios. However, this type of resource is scarce in other languages than
English. In this work we present mMARCO, a multilingual version of the MS MARCO
passage ranking dataset comprising 8 languages that was created using machine
translation. We evaluated mMARCO by fine-tuning mono and multilingual
re-ranking models on it. Experimental results demonstrate that multilingual
models fine-tuned on our translated dataset achieve superior effectiveness than
models fine-tuned on the original English version alone. Also, our distilled
multilingual re-ranker is competitive with non-distilled models while having
5.4 times fewer parameters. The translated datasets as well as fine-tuned
models are available at this https URL.

    

### [[2108.13934] Robust Retrieval Augmented Generation for Zero-shot Slot Filling](http://arxiv.org/abs/2108.13934)


  Automatically inducing high quality knowledge graphs from a given collection
of documents still remains a challenging problem in AI. One way to make headway
for this problem is through advancements in a related task known as slot
filling. In this task, given an entity query in form of [Entity, Slot, ?], a
system is asked to fill the slot by generating or extracting the missing value
exploiting evidence extracted from relevant passage(s) in the given document
collection. The recent works in the field try to solve this task in an
end-to-end fashion using retrieval-based language models. In this paper, we
present a novel approach to zero-shot slot filling that extends dense passage
retrieval with hard negatives and robust training procedures for retrieval
augmented generation models. Our model reports large improvements on both T-REx
and zsRE slot filling datasets, improving both passage retrieval and slot value
generation, and ranking at the top-1 position in the KILT leaderboard.
Moreover, we demonstrate the robustness of our system showing its domain
adaptation capability on a new variant of the TACRED dataset for slot filling,
through a combination of zero/few-shot learning. We release the source code and
pre-trained models.

    

### [[2108.13979] Toward AI-enhanced online-characterization and shaping of ultrashort X-ray free-electron laser pulses](http://arxiv.org/abs/2108.13979)


  X-ray free-electron lasers (XFELs) as the world`s most brilliant light
sources provide ultrashort X-ray pulses with durations typically on the order
of femtoseconds. Recently, they have approached and entered the attosecond
regime, which holds new promises for single-molecule imaging and studying
nonlinear and ultrafast phenomena like localized electron dynamics. The
technological evolution of XFELs toward well-controllable light sources for
precise metrology of ultrafast processes was, however, hampered by the
diagnostic capabilities for characterizing X-ray pulses at the attosecond
frontier. In this regard, the spectroscopic technique of photoelectron angular
streaking has successfully proven how to non-destructively retrieve the exact
time-energy structure of XFEL pulses on a single-shot basis. By using
artificial intelligence algorithms, in particular convolutional neural
networks, we here show how this technique can be leveraged from its
proof-of-principle stage toward routine diagnostics at XFELs, thus enhancing
and refining their scientific access in all related disciplines.

    

### [[2108.13980] Incorporating Deception into CyberBattleSim for Autonomous Defense](http://arxiv.org/abs/2108.13980)


  Deceptive elements, including honeypots and decoys, were incorporated into
the Microsoft CyberBattleSim experimentation and research platform. The
defensive capabilities of the deceptive elements were tested using
reinforcement learning based attackers in the provided capture the flag
environment. The attacker's progress was found to be dependent on the number
and location of the deceptive elements. This is a promising step toward
reproducibly testing attack and defense algorithms in a simulated enterprise
network with deceptive defensive elements.

    

### [[2108.13983] Detecting Mitosis against Domain Shift using a Fused Detector and Deep Ensemble Classification Model for MIDOG Challenge](http://arxiv.org/abs/2108.13983)


  Mitotic figure count is an important marker of tumor proliferation and has
been shown to be associated with patients' prognosis. Deep learning based
mitotic figure detection methods have been utilized to automatically locate the
cell in mitosis using hematoxylin \& eosin (H\&E) stained images. However, the
model performance deteriorates due to the large variation of color tone and
intensity in H\&E images. In this work, we proposed a two stage mitotic figure
detection framework by fusing a detector and a deep ensemble classification
model. To alleviate the impact of color variation in H\&E images, we utilize
both stain normalization and data augmentation, aiding model to learn color
irrelevant features. The proposed model obtains an F1 score of 0.7550 on the
preliminary testing set released by the MIDOG challenge.

    

### [[2108.13989] DeepTaskAPT: Insider APT detection using Task-tree based Deep Learning](http://arxiv.org/abs/2108.13989)


  APT, known as Advanced Persistent Threat, is a difficult challenge for cyber
defence. These threats make many traditional defences ineffective as the
vulnerabilities exploited by these threats are insiders who have access to and
are within the network. This paper proposes DeepTaskAPT, a heterogeneous
task-tree based deep learning method to construct a baseline model based on
sequences of tasks using a Long Short-Term Memory (LSTM) neural network that
can be applied across different users to identify anomalous behaviour. Rather
than applying the model to sequential log entries directly, as most current
approaches do, DeepTaskAPT applies a process tree based task generation method
to generate sequential log entries for the deep learning model. To assess the
performance of DeepTaskAPT, we use a recently released synthetic dataset, DARPA
Operationally Transparent Computing (OpTC) dataset and a real-world dataset,
Los Alamos National Laboratory (LANL) dataset. Both of them are composed of
host-based data collected from sensors. Our results show that DeepTaskAPT
outperforms similar approaches e.g. DeepLog and the DeepTaskAPT baseline model
demonstrate its capability to detect malicious traces in various attack
scenarios while having high accuracy and low false-positive rates. To the best
of knowledge this is the very first attempt of using recently introduced OpTC
dataset for cyber threat detection.

    

### [[2105.07443] How Can Robots Trust Each Other For Better Cooperation? A Relative Needs Entropy Based Robot-Robot Trust Assessment Model](http://arxiv.org/abs/2105.07443)


  Cooperation in multi-agent and multi-robot systems can help agents build
various formations, shapes, and patterns presenting corresponding functions and
purposes adapting to different situations. Relationships between agents such as
their spatial proximity and functional similarities could play a crucial role
in cooperation between agents. Trust level between agents is an essential
factor in evaluating their relationships' reliability and stability, much as
people do. This paper proposes a new model called Relative Needs Entropy (RNE)
to assess trust between robotic agents. RNE measures the distance of needs
distribution between individual agents or groups of agents. To exemplify its
utility, we implement and demonstrate our trust model through experiments
simulating a heterogeneous multi-robot grouping task in a persistent urban
search and rescue mission consisting of tasks at two levels of difficulty. The
results suggest that RNE trust-Based grouping of robots can achieve better
performance and adaptability for diverse task execution compared to the
state-of-the-art energy-based or distance-based grouping models.

    

### [[2108.13783] Synbit: Synthesizing Bidirectional Programs using Unidirectional Sketches](http://arxiv.org/abs/2108.13783)


  We propose a technique for synthesizing bidirectional programs from the
corresponding unidirectional code plus a few input/output examples. The core
ideas are: (1) constructing a sketch using the given unidirectional program as
a specification, and (2) filling the sketch in a modular fashion by exploiting
the properties of bidirectional programs. These ideas are enabled by our choice
of programming language, HOBiT, which is specifically designed to maintain the
unidirectional program structure in bidirectional programming, and keep the
parts that control bidirectional behavior modular. To evaluate our approach, we
implemented it in a tool called Synbit and used it to generate bidirectional
programs for intricate microbenchmarks, as well as for a few larger, more
realistic problems. We also compared Synbit to a state-of-the-art
unidirectional synthesis tool on the task of synthesizing backward
computations.

    

### [[2108.13818] Cats vs. Spectre: An Axiomatic Approach to Modeling Speculative Execution Attacks](http://arxiv.org/abs/2108.13818)


  The Spectre family of speculative execution attacks have required a
rethinking of formal methods for security. Approaches based on operational
speculative semantics have made initial inroads towards finding vulnerable code
and validating defenses. However, with each new attack grows the amount of
microarchitectural detail that has to be integrated into the underlying
semantics. We propose an alternative, light-weight and axiomatic approach to
specifying speculative semantics that relies on insights from memory models for
concurrency. We use the CAT modeling language for memory consistency to specify
execution models that capture speculative control flow, store-to-load
forwarding, predictive store forwarding, and memory ordering machine clears. We
present a bounded model checking framework parametrized by our speculative CAT
models and evaluate its implementation against the state of the art. Due to the
axiomatic approach, our models can be rapidly extended to allow our framework
to detect new types of attacks and validate defenses against them.

    