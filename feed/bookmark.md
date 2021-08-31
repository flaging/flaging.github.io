
## 2021-8-31

### [[2108.12509] A Comprehensive Study of Virtual Machine and Container Based Core Network Components Migration in OpenROADM SDN-Enabled Network](http://arxiv.org/abs/2108.12509)


  With the increasing demand for openness, flexibility, and monetization the
Network Function Virtualization (NFV) of mobile network functions has become
the embracing factor for most mobile network operators. Early reported field
deployments of virtualized Evolved Packet Core (EPC) - the core network
component of 4G LTE and 5G non-standalone mobile networks - reflect this
growing trend. To best meet the requirements of power management, load
balancing, and fault tolerance in the cloud environment, the need for live
migration for these virtualized components cannot be shunned. Virtualization
platforms of interest include both Virtual Machines (VMs) and Containers, with
the latter option offering more lightweight characteristics. The first
contribution of this paper is the implementation of a number of custom
functions that enable migration of Containers supporting virtualized EPC
components. The current CRIU-based migration of Docker Container does not fully
support the mobile network protocol stack. CRIU extensions to support the
mobile network protocol stack are therefore required and described in the
paper. The second contribution is an experimental-based comprehensive analysis
of live migration in two backhaul network settings and two virtualization
technologies. The two backhaul network settings are the one provided by
CloudLab and one based on a programmable optical network testbed that makes use
of OpenROADM dense wavelength division multiplexing (DWDM) equipment. The paper
compares the migration performance of the proposed implementation of
OpenAirInterface (OAI) based containerized EPC components with the one
utilizing VMs, running in OpenStack. The presented experimental comparison
accounts for a number of system parameters and configurations, image size of
the virtualized EPC components, network characteristics, and signal propagation
time across the OpenROADM backhaul network.

    

### [[2108.12573] A Pluralist Approach to Democratizing Online Discourse](http://arxiv.org/abs/2108.12573)


  Online discourse takes place in corporate-controlled spaces thought by users
to be public realms. These platforms in name enable free speech but in practice
implement varying degrees of censorship either by government edict or by uneven
and unseen corporate policy. This kind of censorship has no countervailing
accountability mechanism, and as such platform owners, moderators, and
algorithms shape public discourse without recourse or transparency.
Systems research has explored approaches to decentralizing or democratizing
Internet infrastructure for decades. In parallel, the Internet censorship
literature is replete with efforts to measure and overcome online censorship.
However, in the course of designing specialized open-source platforms and
tools, projects generally neglect the needs of supportive but uninvolved
`average' users. In this paper, we propose a pluralistic approach to
democratizing online discourse that considers both the systems-related and
user-facing issues as first-order design goals.

    

### [[2108.12592] Simulation of Hybrid Edge Computing Architectures](http://arxiv.org/abs/2108.12592)


  Dealing with a growing amount of data is a crucial challenge for the future
of information and communication technologies. More and more devices are
expected to transfer data through the Internet, therefore new solutions have to
be designed in order to guarantee low latency and efficient traffic management.
In this paper, we propose a solution that combines the edge computing paradigm
with a decentralized communication approach based on Peer-to-Peer (P2P).
According to the proposed scheme, participants to the system are employed to
relay messages of other devices, so as to reach a destination (usually a server
at the edge of the network) even in absence of an Internet connection. This
approach can be useful in dynamic and crowded environments, allowing the system
to outsource part of the traffic management from the Cloud servers to
end-devices. To evaluate our proposal, we carry out some experiments with the
help of LUNES, an open source discrete events simulator specifically designed
for distributed environments. In our simulations, we tested several system
configurations in order to understand the impact of the algorithms involved in
the data dissemination and some possible network arrangements.

    

### [[2108.12720] Towards Retina-Quality VR Video Streaming: 15ms Could Save You 80% of Your Bandwidth](http://arxiv.org/abs/2108.12720)


  Virtual reality systems today cannot yet stream immersive, retina-quality
virtual reality video over a network. One of the greatest challenges to this
goal is the sheer data rates required to transmit retina-quality video frames
at high resolutions and frame rates. Recent work has leveraged the decay of
visual acuity in human perception in novel gaze-contingent video compression
techniques. In this paper, we show that reducing the motion-to-photon latency
of a system itself is a key method for improving the compression ratio of
gaze-contingent compression. Our key finding is that a client and streaming
server system with sub-15ms latency can achieve 5x better compression than
traditional techniques while also using simpler software algorithms than
previous work.

    

### [[2108.12722] Feature Extraction for Machine Learning-based Intrusion Detection in IoT Networks](http://arxiv.org/abs/2108.12722)


  The tremendous numbers of network security breaches that have occurred in IoT
networks have demonstrated the unreliability of current Network Intrusion
Detection Systems (NIDSs). Consequently, network interruptions and loss of
sensitive data have occurred which led to an active research area for improving
NIDS technologies. During an analysis of related works, it was observed that
most researchers aimed to obtain better classification results by using a set
of untried combinations of Feature Reduction (FR) and Machine Learning (ML)
techniques on NIDS datasets. However, these datasets are different in feature
sets, attack types, and network design. Therefore, this paper aims to discover
whether these techniques can be generalised across various datasets. Six ML
models are utilised: a Deep Feed Forward, Convolutional Neural Network,
Recurrent Neural Network, Decision Tree, Logistic Regression, and Naive Bayes.
The detection accuracy of three Feature Extraction (FE) algorithms; Principal
Component Analysis (PCA), Auto-encoder (AE), and Linear Discriminant Analysis
(LDA) is evaluated using three benchmark datasets; UNSW-NB15, ToN-IoT and
CSE-CIC-IDS2018. Although PCA and AE algorithms have been widely used,
determining their optimal number of extracted dimensions has been overlooked.
The results obtained indicate that there is no clear FE method or ML model that
can achieve the best scores for all datasets. The optimal number of extracted
dimensions has been identified for each dataset and LDA decreases the
performance of the ML models on two datasets. The variance is used to analyse
the extracted dimensions of LDA and PCA. Finally, this paper concludes that the
choice of datasets significantly alters the performance of the applied
techniques and we argue for the need for a universal (benchmark) feature set to
facilitate further advancement and progress in this field of research.

    

### [[2108.12726] Characterizing Malicious URL Campaigns](http://arxiv.org/abs/2108.12726)


  URLs are central to a myriad of cyber-security threats, from phishing to the
distribution of malware. Their inherent ease of use and familiarity is
continuously abused by attackers to evade defences and deceive end-users.
Seemingly dissimilar URLs are being used in an organized way to perform
phishing attacks and distribute malware. We refer to such behaviours as
campaigns, with the hypothesis being that attacks are often coordinated to
maximize success rates and develop evasion tactics. The aim is to gain better
insights into campaigns, bolster our grasp of their characteristics, and thus
aid the community devise more robust solutions. To this end, we performed
extensive research and analysis into 311M records containing 77M unique
real-world URLs that were submitted to VirusTotal from Dec 2019 to Jan 2020.
From this dataset, 2.6M suspicious campaigns were identified based on their
attached metadata, of which 77,810 were doubly verified as malicious. Using the
38.1M records and 9.9M URLs within these malicious campaigns, we provide varied
insights such as their targeted victim brands as well as URL sizes and
heterogeneity. Some surprising findings were observed, such as detection rates
falling to just 13.27% for campaigns that employ more than 100 unique URLs. The
paper concludes with several case-studies that illustrate the common malicious
techniques employed by attackers to imperil users and circumvent defences.

    

### [[2108.12732] Feature Analysis for ML-based IIoT Intrusion Detection](http://arxiv.org/abs/2108.12732)


  Industrial Internet of Things (IIoT) networks have become an increasingly
attractive target of cyberattacks. Powerful Machine Learning (ML) models have
recently been adopted to implement Network Intrusion Detection Systems (NIDSs),
which can protect IIoT networks. For the successful training of such ML models,
it is important to select the right set of data features, which maximise the
detection accuracy as well as computational efficiency. This paper provides an
extensive analysis of the optimal feature sets in terms of the importance and
predictive power of network attacks. Three feature selection algorithms;
chi-square, information gain and correlation have been utilised to identify and
rank data features. The features are fed into two ML classifiers; deep
feed-forward and random forest, to measure their attack detection accuracy. The
experimental evaluation considered three NIDS datasets: UNSW-NB15,
CSE-CIC-IDS2018, and ToN-IoT in their proprietary flow format. In addition, the
respective variants in NetFlow format were also considered, i.e., NF-UNSW-NB15,
NF-CSE-CIC-IDS2018, and NF-ToN-IoT. The experimental evaluation explored the
marginal benefit of adding features one-by-one. Our results show that the
accuracy initially increases rapidly with the addition of features, but
converges quickly to the maximum achievable detection accuracy. Our results
demonstrate a significant potential of reducing the computational and storage
cost of NIDS while maintaining near-optimal detection accuracy. This has
particular relevance in IIoT systems, with typically limited computational and
storage resource.

    

### [[2108.12825] Modeling and Simulation of Reconfigurable Intelligent Surfaces for Hybrid Aerial and Ground-based Vehicular Communications](http://arxiv.org/abs/2108.12825)


  The requirements of vehicular communications grow with increasing level of
automated driving and future applications of intelligent transportation systems
(ITS). Beside the ever-increasing need for high capacity radio links,
reliability and latency constraints challenge the mobile network supply. While
for example the millimeter-wave spectrum and THz-bands offer a vast amount of
radio resources, their applicability is limited due to delicate radio channel
conditions and signal propagation characteristics. Reconfigurable intelligent
surfaces (RISs) as part of smart radio environments (SREs) of future ITS
infrastructure promise improved radio link qualities by means of purposeful
cultivation of passive reflections. With this, obstructed mmWave or THz beams
can be guided around obstacles through RIS reflection paths to improve the
otherwise limited coverage. In this article, application use cases of
RIS-enhanced vehicular communications are proposed. Beside static deployments
of RISs at exterior walls of buildings, unmanned aerial vehicles (UAV) could
provide reflection capabilities on demand, while future vehicles could - in a
visionary approach - consist of meta-material allowing for their opportunistic
utilization within an enriched SRE. Results of a case study based on our
multi-scale mobility and network simulation model clearly highlight the
potential of RIS deployment for hybrid vehicular communication scenarios. Path
loss and outage percentages can be reduced considerably.

    

### [[2108.12835] Performance Evaluation of Ad Hoc Multicast Routing Protocols to Facilitate Video Streaming in VANETS](http://arxiv.org/abs/2108.12835)


  Vehicular Ad Hoc Network (VANET) is a type of mobile ad hoc network (MANET)
that facilitates communication among vehicles. VANET provides inter-vehicular
communications to serve for the application like road traffic safety and
traffic efficiency. Infotainment service has been an anticipating trend in
VANETs, and video streaming has a high potential in VANET. Although, this
emerging technology is trending, there are still some issues like QoS
provisions, decentralized medium access control, node coverage area, and
finding and maintaining routes due to highly dynamic topology. These issues
make multicast communication difficult in VANETs. Numerous routing protocols
and routing strategies have been projected to cope with these issues. Lots of
work has taken place to assess and measure the performances of these protocols
in VANETs but these protocols are rarely analyzed for performance under stress
of real time video multicast. In this study two different multicast routing
protocols viz. Multicast Ad hoc On Demand Distance Vector (MAODV) and Protocol
for Unified Multicasting through Announcements (PUMA) are evaluated for
facilitating video streaming in VANETS. The protocols are examined against the
QoS parameters such as Network Throughput, Packet Delivery Ratio (PDR), Average
end to end Delay, and Normalized Routing Load (NRL). Variable Bit Rate (VBR)
traffic is used to evaluate the performances of protocol. PUMA, at the end,
showed better performance against different QoS provisions in different
scenarios

    

### [[2108.12849] ACE: an Accurate and Cost-Effective Measurement System in SDN](http://arxiv.org/abs/2108.12849)


  Packet-level traffic measurement is essential in applications like QoS,
traffic engineering, or anomaly detection. Software-Defined Networking (SDN)
enables efficient and dynamic network configuration that we can deploy for
fine-grained network state measurement. As the state changes dynamically, the
sampling frequency must cope with that for accurate measurement. At the same
time, we must consider the measurement cost, e.g., nodes' resource utilization.
Existing works fall short in offering an optimal model to balance both the
measurement accuracy and cost. We fill that gap by proposing ACE, an accurate
and cost-effective measurement system that builds on a multi-objective
optimization problem. As the optimization problem is NP-hard, we develop a
heuristic. We solve the model using CPLEX; then implement a prototype of ACE in
Mininet over real-world network topologies. The results confirm that ACE
outperforms its counterparts in balancing both accuracy and cost.

    

### [[2108.13012] Arctic connectivity: A frugal approach to infrastructural development](http://arxiv.org/abs/2108.13012)


  As the Arctic is heating up, so are efforts to strengthen connectivity within
the region, but also to enhance the connections from remote settlements to the
global networks of trade as well as sociality. With global interest in the
Arctic on the rise, it becomes increasingly relevant to ensure that investments
in arctic infrastructure actually serve the people of the Arctic, while
promoting industrial and commercial innovation in the region through widespread
access to broadband and Internet of Things (IoT) services. This calls for
interdisciplinary research strategies that are able to connect and integrate
technological and societal approaches, which are commonly applied separately
and in isolation from one another. In this article, we propose an
interdisciplinary collaborative research agenda for Arctic connectivity.
Drawing on examples from Greenland, we stress the need for localized knowledge
to design valuable and cost-effective connectivity solutions that cover the
needs for everyday life and may also provide a new set of collaborative
connectivity tools for innovation at an international level. Such solutions,
termed 'frugal connectivity', are vital for the development of connected Arctic
communities.

    

### [[2108.13148] Probabilistic Verification for Reliability of a Two-by-Two Network-on-Chip System](http://arxiv.org/abs/2108.13148)


  Modern network-on-chip (NoC) systems face reliability issues due to process
and environmental variations. The power supply noise (PSN) in the power
delivery network of a NoC plays a key role in determining reliability. PSN
leads to voltage droop, which can cause timing errors in the NoC. This paper
makes a novel contribution towards formally analyzing PSN in NoC systems. We
present a probabilistic model checking approach to observe the PSN in a generic
2x2 mesh NoC with a uniform random traffic load. Key features of PSN are
measured at the behavioral level. To tackle state explosion, we apply
incremental abstraction techniques, including a novel probabilistic choice
abstraction, based on observations of NoC behavior. The Modest Toolset is used
for probabilistic modeling and verification. Results are obtained for several
flit injection patterns to reveal their impacts on PSN. Our analysis finds an
optimal flit pattern generation with zero probability of PSN events and
suggests spreading flits rather than releasing them in consecutive cycles in
order to minimize PSN.

    

### [[2108.13149] An Optimization of Fractal Microstrip Patch Antenna with Partial Ground using Genetic Algorithm Method](http://arxiv.org/abs/2108.13149)


  Ultra-wideband is increasingly advancing as a high data rate wireless
technology after the Federal Communication Commission announced the bandwidth
of 7.5 GHz (from 3.1 GHz to 10.6 GHz) for ultra-wideband applications.
Furthermore, designing a UWB antenna faces more difficulties than designing a
narrow band antenna. A suitable UWB antenna should be able to work over the
Federal Communication Commission of ultra-wide bandwidth allocation.
Furthermore, good radiation properties across the entire frequency spectrum are
needed. This paper outlines an optimization of fractal square microstrip patch
antenna with the partial ground using a genetic algorithm at 3.5 GHz and 6 GHz.
The optimized antenna design shows improved results compared to the
non-optimized design. This design is optimized using a genetic algorithm and
simulated using CST simulation software. The size of the optimized design is
reduced by cutting the edges and the center of the patch. The optimized results
reported, and concentrated on the rerun loss, VSWR and gain. The results
indicate a significant enhancement as is illustrated in Table II. Thus, the
optimized design is suitable for S-band and C-band applications.

    

### [[2108.13150] Transient Analysis for Resonant Beam Charging and Communication](http://arxiv.org/abs/2108.13150)


  High communication speed and sufficient energy supply are the directions of
technological development. Energy and information available anywhere and
anytime has always been people's good wishes. On this basis, resonant beam
system (RBS) has demonstrated its unique superiority in meeting the needs for
energy and communication. The previous work has mostly focused on the analysis
of charging performance of RBS and its steady-state characteristics. In order
to analyze the communication performance of RBS more thoroughly, we propose a
resonant beam charging and communication (RBCC) system and use the equivalent
circuit analysis method to conduct transient analysis on it. The equivalent
circuit reveals the dynamic establishment process of the resonant beam from
scratch, which facilitates the analysis of the relaxation oscillation process
and a deeper understanding of the energy transmission and communication
performance. In addition, we explore the energy transmission and communication
performance of the RBCC under different energy allocation strategies.

    

### [[2108.13154] Towards Secure Wireless Mesh Networks for UAV Swarm Connectivity: Current Threats, Research, and Opportunities](http://arxiv.org/abs/2108.13154)


  UAVs are increasingly appearing in swarms or formations to leverage
cooperative behavior, forming flying ad hoc networks. These UAV-enabled
networks can meet several complex mission requirements and are seen as a
potential enabler for many of the emerging use-cases in future communication
networks. Such networks, however, are characterized by a highly dynamic and
mobile environment with no guarantee of a central network infrastructure which
can cause both connectivity and security issues. While wireless mesh networks
are envisioned as a solution for such scenarios, these networks come with their
own challenges and security vulnerabilities. In this paper, we analyze the key
security and resilience issues resulting from the application of wireless mesh
networks within UAV swarms. Specifically, we highlight the main challenges of
applying current mesh technologies within the domain of UAV swarms and expose
existing vulnerabilities across the communication stack. Based on this
analysis, we present a security-focused architecture for UAV mesh
communications. Finally, from the identification of these vulnerabilities, we
discuss research opportunities posed by the unique challenges of UAV swarm
connectivity.

    

### [[2108.13156] MonTrees: Automated Detection and Classification of Networking Anomalies in Cellular Networks](http://arxiv.org/abs/2108.13156)


  The active growth and dynamic nature of cellular networks makes network
troubleshooting challenging. Identification of network problems leveraging on
machine learning has gained a lot of visibility in the past few years,
resulting in dramatically improved cellular network services. In this paper, we
present a novel methodology to automate the fault identification process in a
cellular network and to classify network anomalies, which combines supervised
and unsupervised machine learning algorithms. Our experiments using real data
from operational commercial mobile networks obtained through drive-test
measurements as well as via the MONROE platform show that our method can
automatically identify and classify networking anomalies, thus enabling timely
and precise troubleshooting actions.

    

### [[2108.13157] DQLEL: Deep Q-Learning for Energy-Optimized LoS/NLoS UWB Node Selection](http://arxiv.org/abs/2108.13157)


  Recent advancements in Internet of Things (IoTs) have brought about a surge
of interest in indoor positioning for the purpose of providing reliable,
accurate, and energy-efficient indoor navigation/localization systems. Ultra
Wide Band (UWB) technology has been emerged as a potential candidate to satisfy
the aforementioned requirements. Although UWB technology can enhance the
accuracy of indoor positioning due to the use of a wide-frequency spectrum,
there are key challenges ahead for its efficient implementation. On the one
hand, achieving high precision in positioning relies on the
identification/mitigation Non Line of Sight (NLoS) links, leading to a
significant increase in the complexity of the localization framework. On the
other hand, UWB beacons have a limited battery life, which is especially
problematic in practical circumstances with certain beacons located in
strategic positions. To address these challenges, we introduce an efficient
node selection framework to enhance the location accuracy without using complex
NLoS mitigation methods, while maintaining a balance between the remaining
battery life of UWB beacons. Referred to as the Deep Q-Learning
Energy-optimized LoS/NLoS (DQLEL) UWB node selection framework, the mobile user
is autonomously trained to determine the optimal pair of UWB beacons to be
localized based on the 2-D Time Difference of Arrival (TDoA) framework. The
effectiveness of the proposed DQLEL framework is evaluated in terms of the link
condition, the deviation of the remaining battery life of UWB beacons, location
error, and cumulative rewards. Based on the simulation results, the proposed
DQLEL framework significantly outperformed its counterparts across the
aforementioned aspects.

    

### [[2108.13158] Exploring Channel Probing to Determine Coherent Optical Transponder Configurations in a Long-Haul Network](http://arxiv.org/abs/2108.13158)


  We use channel probing to determine the best transponder configurations for
spectral services in a long-haul production network. An estimation accuracy
better than +/- 0,7dB in GSNR margin is obtained for lightpaths up to 5738km.

    

### [[2108.13159] A Dynamic Game Approach to Designing Secure Interdependent IoT-Enabled Infrastructure Network](http://arxiv.org/abs/2108.13159)


  The emerging Internet of Things (IoT) applications that leverage ubiquitous
connectivity and big data are facilitating the realization of smart everything
initiatives. IoT-enabled infrastructures have naturally a multi-layer system
architecture with an overlaid or underlaid device network and its coexisting
infrastructure network. The connectivity between different components in these
two heterogeneous interdependent networks plays an important role in delivering
real-time information and ensuring a high-level situational awareness. However,
IoT-enabled infrastructures face cyber threats due to the wireless nature of
communications. Therefore, maintaining network connectivity in the presence of
adversaries is a critical task for infrastructure network operators. In this
paper, we establish a three-player three-stage dynamic game-theoretic framework
including two network operators and one attacker to capture the secure design
of multi-layer interdependent infrastructure networks by allocating limited
resources. We use subgame perfect Nash equilibrium (SPE) to characterize the
strategies of players with sequential moves. In addition, we assess the
efficiency of the equilibrium network by comparing with its team optimal
solution counterparts in which two network operators can coordinate. We further
design a scalable algorithm to guide the construction of the equilibrium
IoT-enabled infrastructure networks. Finally, we use case studies on the
emerging paradigm of the Internet of Battlefield Things (IoBT) to corroborate
the obtained results.

    

### [[2108.13160] NOMA Assisted Multi-MEC Offloading for IoVT Networks](http://arxiv.org/abs/2108.13160)


  Nowadays, Internet of Video Things (IoVT) grows rapidly in terms of quantity
and computation demands. In spite of the higher local computation capability on
visual processing compared with conventional Internet of Things devices, IoVT
devices need to offload partial visual processing tasks to the mobile edge
computing (MEC) server wirelessly due to its larger computation demands.
However, visual processing task offloading is limited by uplink throughput and
computation capability of the MEC server. To break through these limitations, a
novel non-orthogonal multiple access (NOMA) assisted IoVT framework with
multiple MEC servers is proposed, where NOMA is exploited to improve uplink
throughput and MEC servers are co-located with base stations to provide enough
computation capability for offloading. In the proposed framework, the
association strategy, uplink visual data transmission assisted by NOMA and
division of the visual processing tasks as well as computation resource
allocation at the MEC servers are jointly optimized to minimize the total delay
of all visual processing tasks, while meeting the delay requirements of all
IoVT devices. Simulation results demonstrate that significant performance gains
can be achieved by proposed joint optimization with NOMA transmission and
multi-MEC offloading in the heterogeneous IoVT network.

    

### [[2108.13164] Applications and challenges of Reconfigurable Intelligent Surface for 6G networks (Original published in Chinese)](http://arxiv.org/abs/2108.13164)


  Reconfigurable intelligent surface has attracted the attention of academia
and industry as soon as it appears because it can flexibly manipulate the
electromagnetic characteristics of wireless channel. Especially in the past one
or two years, RIS has been developing rapidly in academic research and industry
promotion and is one of the key candidate technologies for 5G-Advanced and 6G
networks. RIS can build a smart radio environment through its ability to
regulate radio wave transmission in a flexible way. The introduction of RIS may
create a new network paradigm, which brings new possibilities to the future
network, but also leads to many new challenges in the technological and
engineering applications. This paper first introduces the main aspects of RIS
enabled wireless communication network from a new perspective, and then focuses
on the key challenges faced by the introduction of RIS. This paper briefly
summarizes the main engineering application challenges faced by RIS networks,
and further analyzes and discusses several key technical challenges among of
them in depth, such as channel degradation, network coexistence, network
coexistence and network deployment, and proposes possible solutions.

    

### [[2108.13165] Deep Learning Based Power Allocation Schemes in NOMA Systems: A Review](http://arxiv.org/abs/2108.13165)


  Achieving significant performance gains both in terms of system throughput
and massive connectivity, non-orthogonal multiple access (NOMA) has been
considered as a very promising candidate for future wireless communications
technologies. It has already received serious consideration for implementation
in the fifth generation (5G) and beyond wireless communication systems. This is
mainly due to NOMA allowing more than one user to utilise one transmission
resource simultaneously at the transmitter side and successive interference
cancellation (SIC) at the receiver side. However, in order to take advantage of
the benefits, NOMA provides in an optimal manner, power allocation needs to be
considered to maximise the system throughput. This problem is non-deterministic
polynomial-time (NP)-hard which is mainly why the use of deep learning
techniques for power allocation is required. In this paper, a state-of-the-art
review on cutting-edge solutions to the power allocation optimisation problem
using deep learning is provided. It is shown that the use of deep learning
techniques to obtain effective solutions to the power allocation problem in
NOMA is paramount for the future of NOMA-based wireless communication systems.
Furthermore, several possible research directions based on the use of deep
learning in NOMA systems are presented.

    

### [[2108.13167] Transportation Polytope and its Applications in Parallel Server Systems](http://arxiv.org/abs/2108.13167)


  Parallel server system is a stochastic processing network widely studied in
the context of manufacturing, supply chain, ride-hailing, call centers, etc.
Heterogeneous customers arrive into the system and only a subset of servers can
serve any given customer type depending on the flexibility graph. As the
flexibility can be overlapping, scheduling decisions must be made to minimize
the delay experienced by the customers. Exact analysis of delay is not possible
and so, we consider the heavy traffic asymptotic regime, wherein the arrival
rate is loaded up to approach the service rate. We consider the general case
when the so called complete resource pooling (CRP) is not satisfied. Recent
work established that when the MaxWeight scheduling algorithm is used, the
state space collapses (SSC) into a lower dimensional sub-space. Building upon
this result, the goal of our paper is to design, analyze and improve the
flexibility graph such that the dimension of SSC is minimized. First, we
characterize the SSC and thus, the mean delay performance in terms of a given
flexibility graph. Using this result, we next study the problem of designing
the sparsest flexibility graph that leads to a target SSC dimension. We
establish a necessary and sufficient condition on the number of edges required,
and provide an algorithm to construct such a graph. Finally, we consider the
question of how to improve a given flexibility graph if one is allowed to add a
single additional edge. The above results are obtained by identifying a
connection to the transportation polytope, and adding to a long line of
literature, we develop new theoretical results for it. These results are
therefore of independent interest. In particular, we obtain new results on the
extreme points and the so-called support graphs of the transportation polytope.

    

### [[2010.05958] FedAT: A High-Performance and Communication-Efficient Federated Learning System with Asynchronous Tiers](http://arxiv.org/abs/2010.05958)


  Federated learning (FL) involves training a model over massive distributed
devices, while keeping the training data localized. This form of collaborative
learning exposes new tradeoffs among model convergence speed, model accuracy,
balance across clients, and communication cost, with new challenges including:
(1) straggler problem, where the clients lag due to data or (computing and
network) resource heterogeneity, and (2) communication bottleneck, where a
large number of clients communicate their local updates to a central server and
bottleneck the server. Many existing FL methods focus on optimizing along only
one dimension of the tradeoff space. Existing solutions use asynchronous model
updating or tiering-based synchronous mechanisms to tackle the straggler
problem. However, the asynchronous methods can easily create a network
communication bottleneck, while tiering may introduce biases as tiering favors
faster tiers with shorter response latencies. To address these issues, we
present FedAT, a novel Federated learning method with Asynchronous Tiers under
Non-i.i.d. data. FedAT synergistically combines synchronous intra-tier training
and asynchronous cross-tier training. By bridging the synchronous and
asynchronous training through tiering, FedAT minimizes the straggler effect
with improved convergence speed and test accuracy. FedAT uses a
straggler-aware, weighted aggregation heuristic to steer and balance the
training for further accuracy improvement. FedAT compresses the uplink and
downlink communications using an efficient, polyline-encoding-based compression
algorithm, therefore minimizing the communication cost. Results show that FedAT
improves the prediction performance by up to 21.09%, and reduces the
communication cost by up to 8.5x, compared to state-of-the-art FL methods.

    

### [[2104.07183] Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based Network Intrusion Detection](http://arxiv.org/abs/2104.07183)


  Machine Learning (ML)-based network intrusion detection systems bring many
benefits for enhancing the cybersecurity posture of an organisation. Many
systems have been designed and developed in the research community, often
achieving a close to perfect detection rate when evaluated using synthetic
datasets. However, the high number of academic research has not often
translated into practical deployments. There are several causes contributing
towards the wide gap between research and production, such as the limited
ability of comprehensive evaluation of ML models and lack of understanding of
internal ML operations. This paper tightens the gap by evaluating the
generalisability of a common feature set to different network environments and
attack scenarios. Therefore, two feature sets (NetFlow and CICFlowMeter) have
been evaluated in terms of detection accuracy across three key datasets, i.e.,
CSE-CIC-IDS2018, BoT-IoT, and ToN-IoT. The results show the superiority of the
NetFlow feature set in enhancing the ML models detection accuracy of various
network attacks. In addition, due to the complexity of the learning models,
SHapley Additive exPlanations (SHAP), an explainable AI methodology, has been
adopted to explain and interpret the classification decisions of ML models. The
Shapley values of two common feature sets have been analysed across multiple
datasets to determine the influence contributed by each feature towards the
final ML prediction.

    

### [[2108.12445] Multimodal Data Fusion in High-Dimensional Heterogeneous Datasets via Generative Models](http://arxiv.org/abs/2108.12445)


  The commonly used latent space embedding techniques, such as Principal
Component Analysis, Factor Analysis, and manifold learning techniques, are
typically used for learning effective representations of homogeneous data.
However, they do not readily extend to heterogeneous data that are a
combination of numerical and categorical variables, e.g., arising from linked
GPS and text data. In this paper, we are interested in learning probabilistic
generative models from high-dimensional heterogeneous data in an unsupervised
fashion. The learned generative model provides latent unified representations
that capture the factors common to the multiple dimensions of the data, and
thus enable fusing multimodal data for various machine learning tasks.
Following a Bayesian approach, we propose a general framework that combines
disparate data types through the natural parameterization of the exponential
family of distributions. To scale the model inference to millions of instances
with thousands of features, we use the Laplace-Bernstein approximation for
posterior computations involving nonlinear link functions. The proposed
algorithm is presented in detail for the commonly encountered heterogeneous
datasets with real-valued (Gaussian) and categorical (multinomial) features.
Experiments on two high-dimensional and heterogeneous datasets (NYC Taxi and
MovieLens-10M) demonstrate the scalability and competitive performance of the
proposed algorithm on different machine learning tasks such as anomaly
detection, data imputation, and recommender systems.

    

### [[2108.12453] Convolutional Autoencoders for Reduced-Order Modeling](http://arxiv.org/abs/2108.12453)


  In the construction of reduced-order models for dynamical systems, linear
projection methods, such as proper orthogonal decompositions, are commonly
employed. However, for many dynamical systems, the lower dimensional
representation of the state space can most accurately be described by a
\textit{nonlinear} manifold. Previous research has shown that deep learning can
provide an efficient method for performing nonlinear dimension reduction,
though they are dependent on the availability of training data and are often
problem-specific \citep[see][]{carlberg_ca}. Here, we utilize randomized
training data to create and train convolutional autoencoders that perform
nonlinear dimension reduction for the wave and Kuramoto-Shivasinsky equations.
Moreover, we present training methods that are independent of full-order model
samples and use the manifold least-squares Petrov-Galerkin projection method to
define a reduced-order model for the heat, wave, and Kuramoto-Shivasinsky
equations using the same autoencoder.

    

### [[2108.12461] Approximate Bayesian Optimisation for Neural Networks](http://arxiv.org/abs/2108.12461)


  A body of work has been done to automate machine learning algorithm to
highlight the importance of model choice. Automating the process of choosing
the best forecasting model and its corresponding parameters can result to
improve a wide range of real-world applications. Bayesian optimisation (BO)
uses a blackbox optimisation methods to propose solutions according to an
exploration-exploitation trade-off criterion through acquisition functions. BO
framework imposes two key ingredients: a probabilistic surrogate model that
consist of prior belief of the unknown objective function(data-dependant) and
an objective function that describes how optimal is the model-fit. Choosing the
best model and its associated hyperparameters can be very expensive, and is
typically fit using Gaussian processes (GPs) and at some extends applying
approximate inference due its intractability. However, since GPs scale
cubically with the number of observations, it has been challenging to handle
objectives whose optimization requires many evaluations. In addition, most
real-dataset are non-stationary which make idealistic assumptions on surrogate
models. The necessity to solve the analytical tractability and the
computational feasibility in a stochastic fashion enables to ensure the
efficiency and the applicability of Bayesian optimisation. In this paper we
explore the use of neural networks as an alternative to GPs to model
distributions over functions, we provide a link between density-ratio
estimation and class probability estimation based on approximate inference,
this reformulation provides algorithm efficiency and tractability.

    

### [[2108.12468] Learning Inner-Group Relations on Point Clouds](http://arxiv.org/abs/2108.12468)


  The prevalence of relation networks in computer vision is in stark contrast
to underexplored point-based methods. In this paper, we explore the
possibilities of local relation operators and survey their feasibility. We
propose a scalable and efficient module, called group relation aggregator. The
module computes a feature of a group based on the aggregation of the features
of the inner-group points weighted by geometric relations and semantic
relations. We adopt this module to design our RPNet. We further verify the
expandability of RPNet, in terms of both depth and width, on the tasks of
classification and segmentation. Surprisingly, empirical results show that
wider RPNet fits for classification, while deeper RPNet works better on
segmentation. RPNet achieves state-of-the-art for classification and
segmentation on challenging benchmarks. We also compare our local aggregator
with PointNet++, with around 30% parameters and 50% computation saving.
Finally, we conduct experiments to reveal the robustness of RPNet with regard
to rigid transformation and noises.

    

### [[2108.12471] Machine learning on DNA-encoded library count data using an uncertainty-aware probabilistic loss function](http://arxiv.org/abs/2108.12471)


  DNA-encoded library (DEL) screening and quantitative structure-activity
relationship (QSAR) modeling are two techniques used in drug discovery to find
small molecules that bind a protein target. Applying QSAR modeling to DEL data
can facilitate the selection of compounds for off-DNA synthesis and evaluation.
Such a combined approach has been shown recently by training binary classifiers
to learn DEL enrichments of aggregated "disynthons" to accommodate the sparse
and noisy nature of DEL data. However, a binary classifier cannot distinguish
between different levels of enrichment, and information is potentially lost
during disynthon aggregation. Here, we demonstrate a regression approach to
learning DEL enrichments of individual molecules using a custom negative
log-likelihood loss function that effectively denoises DEL data and introduces
opportunities for visualization of learned structure-activity relationships
(SAR). Our approach explicitly models the Poisson statistics of the sequencing
process used in the DEL experimental workflow under a frequentist view. We
illustrate this approach on a dataset of 108k compounds screened against CAIX,
and a dataset of 5.7M compounds screened against sEH and SIRT2. Due to the
treatment of uncertainty in the data through the negative log-likelihood loss
function, the models can ignore low-confidence outliers. While our approach
does not demonstrate a benefit for extrapolation to novel structures, we expect
our denoising and visualization pipeline to be useful in identifying SAR trends
and enriched pharmacophores in DEL data. Further, this approach to
uncertainty-aware regression is applicable to other sparse or noisy datasets
where the nature of stochasticity is known or can be modeled; in particular,
the Poisson enrichment ratio metric we use can apply to other settings that
compare sequencing count data between two experimental conditions.

    

### [[2108.12472] ReGen: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models](http://arxiv.org/abs/2108.12472)


  Automatic construction of relevant Knowledge Bases (KBs) from text, and
generation of semantically meaningful text from KBs are both long-standing
goals in Machine Learning. In this paper, we present ReGen, a bidirectional
generation of text and graph leveraging Reinforcement Learning (RL) to improve
performance. Graph linearization enables us to re-frame both tasks as a
sequence to sequence generation problem regardless of the generative direction,
which in turn allows the use of Reinforcement Learning for sequence training
where the model itself is employed as its own critic leading to Self-Critical
Sequence Training (SCST). We present an extensive investigation demonstrating
that the use of RL via SCST benefits graph and text generation on WebNLG+ 2020
and TekGen datasets. Our system provides state-of-the-art results on WebNLG+
2020 by significantly improving upon published results from the WebNLG 2020+
Challenge for both text-to-graph and graph-to-text generation tasks.

    

### [[2108.12473] Mal2GCN: A Robust Malware Detection Approach Using Deep Graph Convolutional Networks With Non-Negative Weights](http://arxiv.org/abs/2108.12473)


  With the growing pace of using machine learning to solve various problems,
securing these models against adversaries has become one of the main concerns
of researchers. Recent studies have shown that in an adversarial environment,
machine learning models are vulnerable to adversarial examples, and adversaries
can create carefully crafted inputs to fool the models. With the advent of deep
neural networks, many researchers have used deep neural networks for various
tasks, and have achieved impressive results. These models must become robust
against attacks before being deployed safely, especially in security-related
fields such as malware detection. In this paper, we first present a black-box
source code-based adversarial malware generation approach that can be used to
evaluate the robustness of malware detection models against real-world
adversaries. The proposed approach injects adversarial codes into the various
locations of malware source codes to evade malware detection models. We then
propose Mal2GCN, a robust malware detection model. Mal2GCN uses the
representation power of graph convolutional networks combined with the
non-negative weights training method to create a malware detection model with
high detection accuracy, which is also robust against adversarial attacks that
add benign features to the input.

    

### [[2108.12489] Using Graph Neural Networks to model the performance of Deep Neural Networks](http://arxiv.org/abs/2108.12489)


  With the unprecedented proliferation of machine learning software, there is
an ever-increasing need to generate efficient code for such applications.
State-of-the-art deep-learning compilers like TVM and Halide incorporate a
learning-based performance model to search the space of valid implementations
of a given deep learning algorithm. For a given application, the model
generates a performance metric such as the run time without executing the
application on hardware. Such models speed up the compilation process by
obviating the need to benchmark an enormous number of candidate
implementations, referred to as schedules, on hardware. Existing performance
models employ feed-forward networks, recurrent networks, or decision tree
ensembles to estimate the performance of different implementations of a neural
network. Graphs present a natural and intuitive way to model deep-learning
networks where each node represents a computational stage or operation.
Incorporating the inherent graph structure of these workloads in the
performance model can enable a better representation and learning of
inter-stage interactions. The accuracy of a performance model has direct
implications on the efficiency of the search strategy, making it a crucial
component of this class of deep-learning compilers. In this work, we develop a
novel performance model that adopts a graph representation. In our model, each
stage of computation represents a node characterized by features that capture
the operations performed by the stage. The interaction between nodes is
achieved using graph convolutions. Experimental evaluation shows a 7:75x and
12x reduction in prediction error compared to the Halide and TVM models,
respectively.

    

### [[2108.12492] Disrupting Adversarial Transferability in Deep Neural Networks](http://arxiv.org/abs/2108.12492)


  Adversarial attack transferability is a well-recognized phenomenon in deep
learning. Prior work has partially explained transferability by recognizing
common adversarial subspaces and correlations between decision boundaries, but
we have found little explanation in the literature beyond this. In this paper,
we propose that transferability between seemingly different models is due to a
high linear correlation between features that different deep neural networks
extract. In other words, two models trained on the same task that are seemingly
distant in the parameter space likely extract features in the same fashion,
just with trivial shifts and rotations between the latent spaces. Furthermore,
we show how applying a feature correlation loss, which decorrelates the
extracted features in a latent space, can drastically reduce the
transferability of adversarial attacks between models, suggesting that the
models complete tasks in semantically different ways. Finally, we propose a
Dual Neck Autoencoder (DNA), which leverages this feature correlation loss to
create two meaningfully different encodings of input information with reduced
transferability.

    

### [[2108.12493] Variational embedding of protein folding simulations using gaussian mixture variational autoencoders](http://arxiv.org/abs/2108.12493)


  Conformational sampling of biomolecules using molecular dynamics simulations
often produces large amount of high dimensional data that makes it difficult to
interpret using conventional analysis techniques. Dimensionality reduction
methods are thus required to extract useful and relevant information. Here we
devise a machine learning method, Gaussian mixture variational autoencoder
(GMVAE) that can simultaneously perform dimensionality reduction and clustering
of biomolecular conformations in an unsupervised way. We show that GMVAE can
learn a reduced representation of the free energy landscape of protein folding
with highly separated clusters that correspond to the metastable states during
folding. Since GMVAE uses a mixture of Gaussians as the prior, it can directly
acknowledge the multi-basin nature of protein folding free-energy landscape. To
make the model end-to-end differentialble, we use a Gumbel-softmax
distribution. We test the model on three long-timescale protein folding
trajectories and show that GMVAE embedding resembles the folding funnel with
folded states down the funnel and unfolded states outer in the funnel path.
Additionally, we show that the latent space of GMVAE can be used for kinetic
analysis and Markov state models built on this embedding produce folding and
unfolding timescales that are in close agreement with other rigorous dynamical
embeddings such as time independent component analysis (TICA).

    

### [[2108.12502] StressNAS: Affect State and Stress Detection Using Neural Architecture Search](http://arxiv.org/abs/2108.12502)


  Smartwatches have rapidly evolved towards capabilities to accurately capture
physiological signals. As an appealing application, stress detection attracts
many studies due to its potential benefits to human health. It is propitious to
investigate the applicability of deep neural networks (DNN) to enhance human
decision-making through physiological signals. However, manually engineering
DNN proves a tedious task especially in stress detection due to the complex
nature of this phenomenon. To this end, we propose an optimized deep neural
network training scheme using neural architecture search merely using
wrist-worn data from WESAD. Experiments show that our approach outperforms
traditional ML methods by 8.22% and 6.02% in the three-state and two-state
classifiers, respectively, using the combination of WESAD wrist signals.
Moreover, the proposed method can minimize the need for human-design DNN while
improving performance by 4.39% (three-state) and 8.99% (binary).

    

### [[2108.12505] On the impact of using X-ray energy response imagery for object detection via Convolutional Neural Networks](http://arxiv.org/abs/2108.12505)


  Automatic detection of prohibited items within complex and cluttered X-ray
security imagery is essential to maintaining transport security, where prior
work on automatic prohibited item detection focus primarily on pseudo-colour
(rgb}) X-ray imagery. In this work we study the impact of variant X-ray
imagery, i.e., X-ray energy response (high, low}) and effective-z compared to
rgb, via the use of deep Convolutional Neural Networks (CNN) for the joint
object detection and segmentation task posed within X-ray baggage security
screening. We evaluate state-of-the-art CNN architectures (Mask R-CNN, YOLACT,
CARAFE and Cascade Mask R-CNN) to explore the transferability of models trained
with such 'raw' variant imagery between the varying X-ray security scanners
that exhibits differing imaging geometries, image resolutions and material
colour profiles. Overall, we observe maximal detection performance using
CARAFE, attributable to training using combination of rgb, high, low, and
effective-z X-ray imagery, obtaining 0.7 mean Average Precision (mAP) for a six
class object detection problem. Our results also exhibit a remarkable degree of
generalisation capability in terms of cross-scanner transferability (AP:
0.835/0.611) for a one class object detection problem by combining rgb, high,
low, and effective-z imagery.

    

### [[2108.12508] Robustness Disparities in Commercial Face Detection](http://arxiv.org/abs/2108.12508)


  Facial detection and analysis systems have been deployed by large companies
and critiqued by scholars and activists for the past decade. Critiques that
focus on system performance analyze disparity of the system's output, i.e., how
frequently is a face detected for different Fitzpatrick skin types or perceived
genders. However, we focus on the robustness of these system outputs under
noisy natural perturbations. We present the first of its kind detailed
benchmark of the robustness of three such systems: Amazon Rekognition,
Microsoft Azure, and Google Cloud Platform. We use both standard and recently
released academic facial datasets to quantitatively analyze trends in
robustness for each. Across all the datasets and systems, we generally find
that photos of individuals who are older, masculine presenting, of darker skin
type, or have dim lighting are more susceptible to errors than their
counterparts in other identities.

    

### [[2108.12510] Pulling Up by the Causal Bootstraps: Causal Data Augmentation for Pre-training Debiasing](http://arxiv.org/abs/2108.12510)


  Machine learning models achieve state-of-the-art performance on many
supervised learning tasks. However, prior evidence suggests that these models
may learn to rely on shortcut biases or spurious correlations (intuitively,
correlations that do not hold in the test as they hold in train) for good
predictive performance. Such models cannot be trusted in deployment
environments to provide accurate predictions. While viewing the problem from a
causal lens is known to be useful, the seamless integration of causation
techniques into machine learning pipelines remains cumbersome and expensive. In
this work, we study and extend a causal pre-training debiasing technique called
causal bootstrapping (CB) under five practical confounded-data
generation-acquisition scenarios (with known and unknown confounding). Under
these settings, we systematically investigate the effect of confounding bias on
deep learning model performance, demonstrating their propensity to rely on
shortcut biases when these biases are not properly accounted for. We
demonstrate that such a causal pre-training technique can significantly
outperform existing base practices to mitigate confounding bias on real-world
domain generalization benchmarking tasks. This systematic investigation
underlines the importance of accounting for the underlying data-generating
mechanisms and fortifying data-preprocessing pipelines with a causal framework
to develop methods robust to confounding biases.

    

### [[2108.12515] Convergence Rates for Learning Linear Operators from Noisy Data](http://arxiv.org/abs/2108.12515)


  We study the Bayesian inverse problem of learning a linear operator on a
Hilbert space from its noisy pointwise evaluations on random input data. Our
framework assumes that this target operator is self-adjoint and diagonal in a
basis shared with the Gaussian prior and noise covariance operators arising
from the imposed statistical model and is able to handle target operators that
are compact, bounded, or even unbounded. We establish posterior contraction
rates with respect to a family of Bochner norms as the number of data tend to
infinity and derive related lower bounds on the estimation error. In the large
data limit, we also provide asymptotic convergence rates of suitably defined
excess risk and generalization gap functionals associated with the posterior
mean point estimator. In doing so, we connect the posterior consistency results
to nonparametric learning theory. Furthermore, these convergence rates
highlight and quantify the difficulty of learning unbounded linear operators in
comparison with the learning of bounded or compact ones. Numerical experiments
confirm the theory and demonstrate that similar conclusions may be expected in
more general problem settings.

    

### [[2108.12519] Predicting the Factuality of Reporting of News Media Using Observations About User Attention in Their YouTube Channels](http://arxiv.org/abs/2108.12519)


  We propose a novel framework for predicting the factuality of reporting of
news media outlets by studying the user attention cycles in their YouTube
channels. In particular, we design a rich set of features derived from the
temporal evolution of the number of views, likes, dislikes, and comments for a
video, which we then aggregate to the channel level. We develop and release a
dataset for the task, containing observations of user attention on YouTube
channels for 489 news media. Our experiments demonstrate both complementarity
and sizable improvements over state-of-the-art textual representations.

    

### [[2108.12522] Learning Energy-Based Approximate Inference Networks for Structured Applications in NLP](http://arxiv.org/abs/2108.12522)


  Structured prediction in natural language processing (NLP) has a long
history. The complex models of structured application come at the difficulty of
learning and inference. These difficulties lead researchers to focus more on
models with simple structure components (e.g., local classifier). Deep
representation learning has become increasingly popular in recent years. The
structure components of their method, on the other hand, are usually relatively
simple. We concentrate on complex structured models in this dissertation. We
provide a learning framework for complicated structured models as well as an
inference method with a better speed/accuracy/search error trade-off. The
dissertation begins with a general introduction to energy-based models. In NLP
and other applications, an energy function is comparable to the concept of a
scoring function. In this dissertation, we discuss the concept of the energy
function and structured models with different energy functions. Then, we
propose a method in which we train a neural network to do argmax inference
under a structured energy function, referring to the trained networks as
"inference networks" or "energy-based inference networks". We then develop ways
of jointly learning energy functions and inference networks using an
adversarial learning framework. Despite the inference and learning difficulties
of energy-based models, we present approaches in this thesis that enable
energy-based models more easily to be applied in structured NLP applications.

    

### [[2108.12530] Combining chest X-rays and EHR data using machine learning to diagnose acute respiratory failure](http://arxiv.org/abs/2108.12530)


  When patients develop acute respiratory failure, accurately identifying the
underlying etiology is essential for determining the best treatment, but it can
be challenging to differentiate between common diagnoses in clinical practice.
Machine learning models could improve medical diagnosis by augmenting clinical
decision making and play a role in the diagnostic evaluation of patients with
acute respiratory failure. While machine learning models have been developed to
identify common findings on chest radiographs (e.g. pneumonia), augmenting
these approaches by also analyzing clinically relevant data from the electronic
health record (EHR) could aid in the diagnosis of acute respiratory failure.
Machine learning models were trained to predict the cause of acute respiratory
failure (pneumonia, heart failure, and/or COPD) using chest radiographs and EHR
data from patients within an internal cohort using diagnoses based on physician
chart review. Models were also tested on patients in an external cohort using
discharge diagnosis codes. A model combining chest radiographs and EHR data
outperformed models based on each modality alone for pneumonia and COPD. For
pneumonia, the combined model AUROC was 0.79 (0.78-0.79), image model AUROC was
0.73 (0.72-0.75), and EHR model AUROC was 0.73 (0.70-0.76); for COPD, combined:
0.89 (0.83-0.91), image: 0.85 (0.77-0.89), and EHR: 0.80 (0.76-0.84); for heart
failure, combined: 0.80 (0.77-0.84), image: 0.77 (0.71-0.81), and EHR: 0.80
(0.75-0.82). In the external cohort, performance was consistent for heart
failure and COPD, but declined slightly for pneumonia. Overall, machine
learning models combing chest radiographs and EHR data can accurately
differentiate between common causes of acute respiratory failure. Further work
is needed to determine whether these models could aid clinicians in the
diagnosis of acute respiratory failure in clinical settings.

    

### [[2108.12531] Speech Representations and Phoneme Classification for Preserving the Endangered Language of Ladin](http://arxiv.org/abs/2108.12531)


  A vast majority of the world's 7,000 spoken languages are predicted to become
extinct within this century, including the endangered language of Ladin from
the Italian Alps. Linguists who work to preserve a language's phonetic and
phonological structure can spend hours transcribing each minute of speech from
native speakers. To address this problem in the context of Ladin, our paper
presents the first analysis of speech representations and machine learning
models for classifying 32 phonemes of Ladin. We experimented with a novel
dataset of the Fascian dialect of Ladin, collected from native speakers in
Italy. We created frame-level and segment-level speech feature extraction
approaches and conducted extensive experiments with 8 different classifiers
trained on 9 different speech representations. Our speech representations
ranged from traditional features (MFCC, LPC) to features learned with deep
neural network models (autoencoders, LSTM autoencoders, and WaveNet). Our
highest-performing classifier, trained on MFCC representations of speech
signals, achieved an 86% average accuracy across all Ladin phonemes. We also
obtained average accuracies above 77% for all Ladin phoneme subgroups examined.
Our findings contribute insights for learning discriminative Ladin phoneme
representations and demonstrate the potential for leveraging machine learning
and speech signal processing to preserve Ladin and other endangered languages.

    

### [[2108.12547] Self-fulfilling Bandits: Endogeneity Spillover and Dynamic Selection in Algorithmic Decision-making](http://arxiv.org/abs/2108.12547)


  In this paper, we study endogeneity problems in algorithmic decision-making
where data and actions are interdependent. When there are endogenous covariates
in a contextual multi-armed bandit model, a novel bias (self-fulfilling bias)
arises because the endogeneity of the covariates spills over to the actions. We
propose a class of algorithms to correct for the bias by incorporating
instrumental variables into leading online learning algorithms. These
algorithms also attain regret levels that match the best known lower bound for
the cases without endogeneity. To establish the theoretical properties, we
develop a general technique that untangles the interdependence between data and
actions.

    

### [[2108.12579] Power-Based Attacks on Spatial DNN Accelerators](http://arxiv.org/abs/2108.12579)


  With proliferation of DNN-based applications, the confidentiality of DNN
model is an important commercial goal. Spatial accelerators, that parallelize
matrix/vector operations, are utilized for enhancing energy efficiency of DNN
computation. Recently, model extraction attacks on simple accelerators, either
with a single processing element or running a binarized network, were
demonstrated using the methodology derived from differential power analysis
(DPA) attack on cryptographic devices. This paper investigates the
vulnerability of realistic spatial accelerators using general, 8-bit, number
representation.
We investigate two systolic array architectures with weight-stationary
dataflow: (1) a 3 $\times$ 1 array for a dot-product operation, and (2) a 3
$\times$ 3 array for matrix-vector multiplication. Both are implemented on the
SAKURA-G FPGA board. We show that both architectures are ultimately vulnerable.
A conventional DPA succeeds fully on the 1D array, requiring 20K power
measurements. However, the 2D array exhibits higher security even with 460K
traces. We show that this is because the 2D array intrinsically entails
multiple MACs simultaneously dependent on the same input. However, we find that
a novel template-based DPA with multiple profiling phases is able to fully
break the 2D array with only 40K traces. Corresponding countermeasures need to
be investigated for spatial DNN accelerators.

    

### [[2108.12581] Influence-based Reinforcement Learning for Intrinsically-motivated Agents](http://arxiv.org/abs/2108.12581)


  The reinforcement learning (RL) research area is very active, with several
important applications. However, certain challenges still need to be addressed,
amongst which one can mention the ability to find policies that achieve
sufficient exploration and coordination while solving a given task. In this
work, we present an algorithmic framework of two RL agents each with a
different objective. We introduce a novel function approximation approach to
assess the influence $F$ of a certain policy on others. While optimizing $F$ as
a regularizer of $\pi$'s objective, agents learn to coordinate team behavior
while exploiting high-reward regions of the solution space. Additionally, both
agents use prediction error as intrinsic motivation to learn policies that
behave as differently as possible, thus achieving the exploration criterion.
Our method was evaluated on the suite of OpenAI gym tasks as well as
cooperative and mixed scenarios, where agent populations are able to discover
various physical and informational coordination strategies, showing
state-of-the-art performance when compared to famous baselines.

    

### [[2108.12594] Layer-wise Model Pruning based on Mutual Information](http://arxiv.org/abs/2108.12594)


  The proposed pruning strategy offers merits over weight-based pruning
techniques: (1) it avoids irregular memory access since representations and
matrices can be squeezed into their smaller but dense counterparts, leading to
greater speedup; (2) in a manner of top-down pruning, the proposed method
operates from a more global perspective based on training signals in the top
layer, and prunes each layer by propagating the effect of global signals
through layers, leading to better performances at the same sparsity level.
Extensive experiments show that at the same sparsity level, the proposed
strategy offers both greater speedup and higher performances than weight-based
pruning methods (e.g., magnitude pruning, movement pruning).

    

### [[2108.12596] Representation Memorization for Fast Learning New Knowledge without Forgetting](http://arxiv.org/abs/2108.12596)


  The ability to quickly learn new knowledge (e.g. new classes or data
distributions) is a big step towards human-level intelligence. In this paper,
we consider scenarios that require learning new classes or data distributions
quickly and incrementally over time, as it often occurs in real-world dynamic
environments. We propose "Memory-based Hebbian Parameter Adaptation" (Hebb) to
tackle the two major challenges (i.e., catastrophic forgetting and sample
efficiency) towards this goal in a unified framework. To mitigate catastrophic
forgetting, Hebb augments a regular neural classifier with a continuously
updated memory module to store representations of previous data. To improve
sample efficiency, we propose a parameter adaptation method based on the
well-known Hebbian theory, which directly "wires" the output network's
parameters with similar representations retrieved from the memory. We
empirically verify the superior performance of Hebb through extensive
experiments on a wide range of learning tasks (image classification, language
model) and learning scenarios (continual, incremental, online). We demonstrate
that Hebb effectively mitigates catastrophic forgetting, and it indeed learns
new knowledge better and faster than the current state-of-the-art.

    

### [[2108.12601] Mitigation of Diachronic Bias in Fake News Detection Dataset](http://arxiv.org/abs/2108.12601)


  Fake news causes significant damage to this http URL deal with these fake news,
several studies on building detection models and arranging datasets have been
conducted. Most of the fake news datasets depend on a specific time period.
Consequently, the detection models trained on such a dataset have difficulty
detecting novel fake news generated by political changes and social changes;
they may possibly result in biased output from the input, including specific
person names and organizational names. We refer to this problem as
\textbf{Diachronic Bias} because it is caused by the creation date of news in
each dataset. In this study, we confirm the bias, especially proper nouns
including person names, from the deviation of phrase appearances in each
dataset. Based on these findings, we propose masking methods using Wikidata to
mitigate the influence of person names and validate whether they make fake news
detection models robust through experiments with in-domain and out-of-domain
data.

    

### [[2108.12603] WALNUT: A Benchmark on Weakly Supervised Learning for Natural Language Understanding](http://arxiv.org/abs/2108.12603)


  Building quality machine learning models for natural language understanding
(NLU) tasks relies heavily on labeled data. Weak supervision has been shown to
provide valuable supervision when large amount of labeled data is unavailable
or expensive to obtain. Existing works studying weak supervision for NLU either
mostly focus on a specific task or simulate weak supervision signals from
ground-truth labels. To date a benchmark for NLU with real world weak
supervision signals for a collection of NLU tasks is still not available. In
this paper, we propose such a benchmark, named WALNUT, to advocate and
facilitate research on weak supervision for NLU. WALNUT consists of NLU tasks
with different types, including both document-level prediction tasks and
token-level prediction tasks and for each task contains weak labels generated
by multiple real-world weak sources. We conduct baseline evaluations on the
benchmark to systematically test the value of weak supervision for NLU tasks,
with various weak supervision methods and model architectures. We demonstrate
the benefits of weak supervision for low-resource NLU tasks and expect WALNUT
to stimulate further research on methodologies to best leverage weak
supervision. The benchmark and code for baselines will be publicly available at
this http URL.

    

### [[2108.12626] HeadlineCause: A Dataset of News Headlines for Detecting Casualties](http://arxiv.org/abs/2108.12626)


  Detecting implicit causal relations in texts is a task that requires both
common sense and world knowledge. Existing datasets are focused either on
commonsense causal reasoning or explicit causal relations. In this work, we
present HeadlineCause, a dataset for detecting implicit causal relations
between pairs of news headlines. The dataset includes over 5000 headline pairs
from English news and over 9000 headline pairs from Russian news labeled
through crowdsourcing. The pairs vary from totally unrelated or belonging to
the same general topic to the ones including causation and refutation
relations. We also present a set of models and experiments that demonstrates
the dataset validity, including a multilingual XLM-RoBERTa based model for
causality detection and a GPT-2 based model for possible effects prediction.

    

### [[2108.12627] Generalized Huber Loss for Robust Learning and its Efficient Minimization for a Robust Statistics](http://arxiv.org/abs/2108.12627)


  We propose a generalized formulation of the Huber loss. We show that with a
suitable function of choice, specifically the log-exp transform; we can achieve
a loss function which combines the desirable properties of both the absolute
and the quadratic loss. We provide an algorithm to find the minimizer of such
loss functions and show that finding a centralizing metric is not that much
harder than the traditional mean and median.

    

### [[2108.12641] Prototypes-Guided Memory Replay for Continual Learning](http://arxiv.org/abs/2108.12641)


  Continual learning (CL) refers to a machine learning paradigm that using only
a small account of training samples and previously learned knowledge to enhance
learning performance. CL models learn tasks from various domains in a
sequential manner. The major difficulty in CL is catastrophic forgetting of
previously learned tasks, caused by shifts in data distributions. The existing
CL models often employ a replay-based approach to diminish catastrophic
forgetting. Most CL models stochastically select previously seen samples to
retain learned knowledge. However, occupied memory size keeps enlarging along
with accumulating learned tasks. Hereby, we propose a memory-efficient CL
method. We devise a dynamic prototypes-guided memory replay module,
incorporating it into an online meta-learning model. We conduct extensive
experiments on text classification and additionally investigate the effect of
training set orders on CL model performance. The experimental results testify
the superiority of our method in alleviating catastrophic forgetting and
enabling efficient knowledge transfer.

    

### [[2108.12643] Master memory function for delay-based reservoir computers with single-variable dynamics](http://arxiv.org/abs/2108.12643)


  We show that many delay-based reservoir computers considered in the
literature can be characterized by a universal master memory function (MMF).
Once computed for two independent parameters, this function provides linear
memory capacity for any delay-based single-variable reservoir with small
inputs. Moreover, we propose an analytical description of the MMF that enables
its efficient and fast computation.
Our approach can be applied not only to reservoirs governed by known
dynamical rules such as Mackey-Glass or Ikeda-like systems but also to
reservoirs whose dynamical model is not available. We also present results
comparing the performance of the reservoir computer and the memory capacity
given by the MMF.

    

### [[2108.12657] Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models](http://arxiv.org/abs/2108.12657)


  Fast inference of numerical model parameters from data is an important
prerequisite to generate predictive models for a wide range of applications.
Use of sampling-based approaches such as Markov chain Monte Carlo may become
intractable when each likelihood evaluation is computationally expensive. New
approaches combining variational inference with normalizing flow are
characterized by a computational cost that grows only linearly with the
dimensionality of the latent variable space, and rely on gradient-based
optimization instead of sampling, providing a more efficient approach for
Bayesian inference about the model parameters. Moreover, the cost of frequently
evaluating an expensive likelihood can be mitigated by replacing the true model
with an offline trained surrogate model, such as neural networks. However, this
approach might generate significant bias when the surrogate is insufficiently
accurate around the posterior modes. To reduce the computational cost without
sacrificing inferential accuracy, we propose Normalizing Flow with Adaptive
Surrogate (NoFAS), an optimization strategy that alternatively updates the
normalizing flow parameters and the weights of a neural network surrogate
model. We also propose an efficient sample weighting scheme for surrogate model
training that ensures some global accuracy of the surrogate while capturing the
likely regions of the parameters that yield the observed data. We demonstrate
the inferential and computational superiority of NoFAS against various
benchmarks, including cases where the underlying model lacks identifiability.
The source code and numerical experiments used for this study are available at
this https URL.

    

### [[2108.12659] DKM: Differentiable K-Means Clustering Layer for Neural Network Compression](http://arxiv.org/abs/2108.12659)


  Deep neural network (DNN) model compression for efficient on-device inference
is becoming increasingly important to reduce memory requirements and keep user
data on-device. To this end, we propose a novel differentiable k-means
clustering layer (DKM) and its application to train-time weight
clustering-based DNN model compression. DKM casts k-means clustering as an
attention problem and enables joint optimization of the parameters and
clustering centroids. Unlike prior works that rely on additional regularizers
and parameters, DKM-based compression keeps the original loss function and
model architecture fixed. We evaluated DKM-based compression on various DNN
models for computer vision and natural language processing (NLP) tasks. Our
results demonstrate that DMK delivers superior compression and accuracy
trade-off on ImageNet1k and GLUE benchmarks. For example, DKM-based compression
can offer 74.5% top-1 ImageNet1k accuracy on ResNet50 DNN model with 3.3MB
model size (29.4x model compression factor). For MobileNet-v1, which is a
challenging DNN to compress, DKM delivers 62.8% top-1 ImageNet1k accuracy with
0.74 MB model size (22.4x model compression factor). This result is 6.8% higher
top-1 accuracy and 33% relatively smaller model size than the current
state-of-the-art DNN compression algorithms. Additionally, DKM enables
compression of DistilBERT model by 11.8x with minimal (1.1%) accuracy loss on
GLUE NLP benchmarks.

    

### [[2108.12680] Avoiding unwanted results in locally linear embedding: A new understanding of regularization](http://arxiv.org/abs/2108.12680)


  We demonstrate that locally linear embedding (LLE) inherently admits some
unwanted results when no regularization is used, even for cases in which
regularization is not supposed to be needed in the original algorithm. The
existence of one special type of result, which we call ``projection pattern'',
is mathematically proved in the situation that an exact local linear relation
is achieved in each neighborhood of the data. These special patterns as well as
some other bizarre results that may occur in more general situations are shown
by numerical examples on the Swiss roll with a hole embedded in a high
dimensional space. It is observed that all these bad results can be effectively
prevented by using regularization.

    

### [[2108.12704] Compact representations of convolutional neural networks via weight pruning and quantization](http://arxiv.org/abs/2108.12704)


  The state-of-the-art performance for several real-world problems is currently
reached by convolutional neural networks (CNN). Such learning models exploit
recent results in the field of deep learning, typically leading to highly
performing, yet very large neural networks with (at least) millions of
parameters. As a result, the deployment of such models is not possible when
only small amounts of RAM are available, or in general within resource-limited
platforms, and strategies to compress CNNs became thus of paramount importance.
In this paper we propose a novel lossless storage format for CNNs based on
source coding and leveraging both weight pruning and quantization. We
theoretically derive the space upper bounds for the proposed structures,
showing their relationship with both sparsity and quantization levels of the
weight matrices. Both compression rates and excution times have been tested
against reference methods for matrix compression, and an empirical evaluation
of state-of-the-art quantization schemes based on weight sharing is also
discussed, to assess their impact on the performance when applied to both
convolutional and fully connected layers. On four benchmarks for classification
and regression problems and comparing to the baseline pre-trained uncompressed
network, we achieved a reduction of space occupancy up to 0.6% on fully
connected layers and 5.44% on the whole network, while performing at least as
competitive as the baseline.

    

### [[2108.12717] Harvesting Idle Resources in Serverless Computing via Reinforcement Learning](http://arxiv.org/abs/2108.12717)


  Serverless computing has become a new cloud computing paradigm that promises
to deliver high cost-efficiency and simplified cloud deployment with automated
resource scaling at a fine granularity. Users decouple a cloud application into
chained functions and preset each serverless function's memory and CPU demands
at megabyte-level and core-level, respectively. Serverless platforms then
automatically scale the number of functions to accommodate the workloads.
However, the complexities of chained functions make it non-trivial to
accurately determine the resource demands of each function for users, leading
to either resource over-provision or under-provision for individual functions.
This paper presents FaaSRM, a new resource manager (RM) for serverless
platforms that maximizes resource efficiency by dynamically harvesting idle
resources from functions over-supplied to functions under-supplied. FaaSRM
monitors each function's resource utilization in real-time, detects
over-provisioning and under-provisioning, and applies deep reinforcement
learning to harvest idle resources safely using a safeguard mechanism and
accelerate functions efficiently. We have implemented and deployed a FaaSRM
prototype in a 13-node Apache OpenWhisk cluster. Experimental results on the
OpenWhisk cluster show that FaaSRM reduces the execution time of 98% of
function invocations by 35.81% compared to the baseline RMs by harvesting idle
resources from 38.8% of the invocations and accelerating 39.2% of the
invocations.

    

### [[2108.12719] A Dual Adversarial Calibration Framework for Automatic Fetal Brain Biometry](http://arxiv.org/abs/2108.12719)


  This paper presents a novel approach to automatic fetal brain biometry
motivated by needs in low- and medium- income countries. Specifically, we
leverage high-end (HE) ultrasound images to build a biometry solution for
low-cost (LC) point-of-care ultrasound images. We propose a novel unsupervised
domain adaptation approach to train deep models to be invariant to significant
image distribution shift between the image types. Our proposed method, which
employs a Dual Adversarial Calibration (DAC) framework, consists of adversarial
pathways which enforce model invariance to; i) adversarial perturbations in the
feature space derived from LC images, and ii) appearance domain discrepancy.
Our Dual Adversarial Calibration method estimates transcerebellar diameter and
head circumference on images from low-cost ultrasound devices with a mean
absolute error (MAE) of 2.43mm and 1.65mm, compared with 7.28 mm and 5.65 mm
respectively for SOTA.

    

### [[2108.12734] Deep Dive into Semi-Supervised ELBO for Improving Classification Performance](http://arxiv.org/abs/2108.12734)


  Decomposition of the evidence lower bound (ELBO) objective of VAE used for
density estimation revealed the deficiency of VAE for representation learning
and suggested ways to improve the model. In this paper, we investigate whether
we can get similar insights by decomposing the ELBO for semi-supervised
classification using VAE model. Specifically, we show that mutual information
between input and class labels decreases during maximization of ELBO objective.
We propose a method to address this issue. We also enforce cluster assumption
to aid in classification. Experiments on a diverse datasets verify that our
method can be used to improve the classification performance of existing VAE
based semi-supervised models. Experiments also show that, this can be achieved
without sacrificing the generative power of the model.

    

### [[2108.12746] Certifying One-Phase Technology-Assisted Reviews](http://arxiv.org/abs/2108.12746)


  Technology-assisted review (TAR) workflows based on iterative active learning
are widely used in document review applications. Most stopping rules for
one-phase TAR workflows lack valid statistical guarantees, which has
discouraged their use in some legal contexts. Drawing on the theory of quantile
estimation, we provide the first broadly applicable and statistically valid
sample-based stopping rules for one-phase TAR. We further show theoretically
and empirically that overshooting a recall target, which has been treated as
innocuous or desirable in past evaluations of stopping rules, is a major source
of excess cost in one-phase TAR workflows. Counterintuitively, incurring a
larger sampling cost to reduce excess recall leads to lower total cost in
almost all scenarios.

    

### [[2108.12768] CrossedWires: A Dataset of Syntactically Equivalent but Semantically Disparate Deep Learning Models](http://arxiv.org/abs/2108.12768)


  The training of neural networks using different deep learning frameworks may
lead to drastically differing accuracy levels despite the use of the same
neural network architecture and identical training hyperparameters such as
learning rate and choice of optimization algorithms. Currently, our ability to
build standardized deep learning models is limited by the availability of a
suite of neural network and corresponding training hyperparameter benchmarks
that expose differences between existing deep learning frameworks. In this
paper, we present a living dataset of models and hyperparameters, called
CrossedWires, that exposes semantic differences between two popular deep
learning frameworks: PyTorch and Tensorflow. The CrossedWires dataset currently
consists of models trained on CIFAR10 images using three different computer
vision architectures: VGG16, ResNet50 and DenseNet121 across a large
hyperparameter space. Using hyperparameter optimization, each of the three
models was trained on 400 sets of hyperparameters suggested by the HyperSpace
search algorithm. The CrossedWires dataset includes PyTorch and Tensforflow
models with test accuracies as different as 0.681 on syntactically equivalent
models and identical hyperparameter choices. The 340 GB dataset and benchmarks
presented here include the performance statistics, training curves, and model
weights for all 1200 hyperparameter choices, resulting in 2400 total models.
The CrossedWires dataset provides an opportunity to study semantic differences
between syntactically equivalent models across popular deep learning
frameworks. Further, the insights obtained from this study can enable the
development of algorithms and tools that improve reliability and
reproducibility of deep learning frameworks. The dataset is freely available at
this https URL through a Python API and direct
download link.

    

### [[2108.12784] TCCT: Tightly-Coupled Convolutional Transformer on Time Series Forecasting](http://arxiv.org/abs/2108.12784)


  Time series forecasting is essential for a wide range of real-world
applications. Recent studies have shown the superiority of Transformer in
dealing with such problems, especially long sequence time series input(LSTI)
and long sequence time series forecasting(LSTF) problems. To improve the
efficiency and enhance the locality of Transformer, these studies combine
Transformer with CNN in varying degrees. However, their combinations are
loosely-coupled and do not make full use of CNN. To address this issue, we
propose the concept of tightly-coupled convolutional Transformer(TCCT) and
three TCCT architectures which apply transformed CNN architectures into
Transformer: (1) CSPAttention: through fusing CSPNet with self-attention
mechanism, the computation cost of self-attention mechanism is reduced by 30%
and the memory usage is reduced by 50% while achieving equivalent or beyond
prediction accuracy. (2) Dilated causal convolution: this method is to modify
the distilling operation proposed by Informer through replacing canonical
convolutional layers with dilated causal convolutional layers to gain
exponentially receptive field growth. (3) Passthrough mechanism: the
application of passthrough mechanism to stack of self-attention blocks helps
Transformer-like models get more fine-grained information with negligible extra
computation costs. Our experiments on real-world datasets show that our TCCT
architectures could greatly improve the performance of existing state-of-art
Transformer models on time series forecasting with much lower computation and
memory costs, including canonical Transformer, LogTrans and Informer.

    

### [[2108.12788] Attempt to Predict Failure Case Classification in a Failure Database by using Neural Network Models](http://arxiv.org/abs/2108.12788)


  With the recent progress of information technology, the use of networked
information systems has rapidly expanded. Electronic commerce and electronic
payments between banks and companies, and online shopping and social networking
services used by the general public are examples of such systems. Therefore, in
order to maintain and improve the dependability of these systems, we are
constructing a failure database from past failure cases. When importing new
failure cases to the database, it is necessary to classify these cases
according to failure type. The problems are the accuracy and efficiency of the
classification. Especially when working with multiple individuals, unification
of classification is required. Therefore, we are attempting to automate
classification using machine learning. As evaluation models, we selected the
multilayer perceptron (MLP), the convolutional neural network (CNN), and the
recurrent neural network (RNN), which are models that use neural networks. As a
result, the optimal model in terms of accuracy is first the MLP followed by the
CNN, and the processing time of the classification is practical.

    

### [[2108.12801] Markov Switching Model for Driver Behavior Prediction: Use cases on Smartphones](http://arxiv.org/abs/2108.12801)


  Several intelligent transportation systems focus on studying the various
driver behaviors for numerous objectives. This includes the ability to analyze
driver actions, sensitivity, distraction, and response time. As the data
collection is one of the major concerns for learning and validating different
driving situations, we present a driver behavior switching model validated by a
low-cost data collection solution using smartphones. The proposed model is
validated using a real dataset to predict the driver behavior in short duration
periods. A literature survey on motion detection (specifically driving behavior
detection using smartphones) is presented. Multiple Markov Switching Variable
Auto-Regression (MSVAR) models are implemented to achieve a sophisticated
fitting with the collected driver behavior data. This yields more accurate
predictions not only for driver behavior but also for the entire driving
situation. The performance of the presented models together with a suitable
model selection criteria is also presented. The proposed driver behavior
prediction framework can potentially be used in accident prediction and driver
safety systems.

    

### [[2108.12802] Interpretable Propaganda Detection in News Articles](http://arxiv.org/abs/2108.12802)


  Online users today are exposed to misleading and propagandistic news articles
and media posts on a daily basis. To counter thus, a number of approaches have
been designed aiming to achieve a healthier and safer online news and media
consumption. Automatic systems are able to support humans in detecting such
content; yet, a major impediment to their broad adoption is that besides being
accurate, the decisions of such systems need also to be interpretable in order
to be trusted and widely adopted by users. Since misleading and propagandistic
content influences readers through the use of a number of deception techniques,
we propose to detect and to show the use of such techniques as a way to offer
interpretability. In particular, we define qualitatively descriptive features
and we analyze their suitability for detecting deception techniques. We further
show that our interpretable features can be easily combined with pre-trained
language models, yielding state-of-the-art results.

    

### [[2108.12805] DropAttack: A Masked Weight Adversarial Training Method to Improve Generalization of Neural Networks](http://arxiv.org/abs/2108.12805)


  Adversarial training has been proven to be a powerful regularization method
to improve the generalization of models. However, current adversarial training
methods only attack the original input sample or the embedding vectors, and
their attacks lack coverage and diversity. To further enhance the breadth and
depth of attack, we propose a novel masked weight adversarial training method
called DropAttack, which enhances generalization of model by adding
intentionally worst-case adversarial perturbations to both the input and hidden
layers in different dimensions and minimize the adversarial risks generated by
each layer. DropAttack is a general technique and can be adopt to a wide
variety of neural networks with different architectures. To validate the
effectiveness of the proposed method, we used five public datasets in the
fields of natural language processing (NLP) and computer vision (CV) for
experimental evaluating. We compare the proposed method with other adversarial
training methods and regularization methods, and our method achieves
state-of-the-art on all datasets. In addition, Dropattack can achieve the same
performance when it use only a half training data compared to other standard
training method. Theoretical analysis reveals that DropAttack can perform
gradient regularization at random on some of the input and wight parameters of
the model. Further visualization experiments show that DropAttack can push the
minimum risk of the model to a lower and flatter loss landscapes. Our source
code is publicly available on this https URL.

    

### [[2108.12816] Privacy-preserving Machine Learning for Medical Image Classification](http://arxiv.org/abs/2108.12816)


  With the rising use of Machine Learning (ML) and Deep Learning (DL) in
various industries, the medical industry is also not far behind. A very simple
yet extremely important use case of ML in this industry is for image
classification. This is important for doctors to help them detect certain
diseases timely, thereby acting as an aid to reduce chances of human judgement
error. However, when using automated systems like these, there is a privacy
concern as well. Attackers should not be able to get access to the medical
records and images of the patients. It is also required that the model be
secure, and that the data that is sent to the model and the predictions that
are received both should not be revealed to the model in clear text.
In this study, we aim to solve these problems in the context of a medical
image classification problem of detection of pneumonia by examining chest x-ray
images.

    

### [[2108.12821] Analyzing and Mitigating Interference in Neural Architecture Search](http://arxiv.org/abs/2108.12821)


  Weight sharing has become the \textit{de facto} approach to reduce the
training cost of neural architecture search (NAS) by reusing the weights of
shared operators from previously trained child models. However, the estimated
accuracy of those child models has a low rank correlation with the ground truth
accuracy due to the interference among different child models caused by weight
sharing. In this paper, we investigate the interference issue by sampling
different child models and calculating the gradient similarity of shared
operators, and observe that: 1) the interference on a shared operator between
two child models is positively correlated to the number of different operators
between them; 2) the interference is smaller when the inputs and outputs of the
shared operator are more similar. Inspired by these two observations, we
propose two approaches to mitigate the interference: 1) rather than randomly
sampling child models for optimization, we propose a gradual modification
scheme by modifying one operator between adjacent optimization steps to
minimize the interference on the shared operators; 2) forcing the inputs and
outputs of the operator across all child models to be similar to reduce the
interference. Experiments on a BERT search space verify that mitigating
interference via each of our proposed methods improves the rank correlation of
super-pet and combining both methods can achieve better results. Our searched
architecture outperforms RoBERTa$_{\rm base}$ by 1.1 and 0.6 scores and
ELECTRA$_{\rm base}$ by 1.6 and 1.1 scores on the dev and test set of GLUE
benchmark. Extensive results on the BERT compression task, SQuAD datasets and
other search spaces also demonstrate the effectiveness and generality of our
proposed methods.

    

### [[2108.12828] MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification](http://arxiv.org/abs/2108.12828)


  Recent research in disaster informatics demonstrates a practical and
important use case of artificial intelligence to save human lives and
sufferings during post-natural disasters based on social media contents (text
and images). While notable progress has been made using texts, research on
exploiting the images remains relatively under-explored. To advance the
image-based approach, we propose MEDIC (available at:
this https URL), which is the largest social media
image classification dataset for humanitarian response consisting of 71,198
images to address four different tasks in a multi-task learning setup. This is
the first dataset of its kind: social media image, disaster response, and
multi-task learning research. An important property of this dataset is its high
potential to contribute research on multi-task learning, which recently
receives much interest from the machine learning community and has shown
remarkable results in terms of memory, inference speed, performance, and
generalization capability. Therefore, the proposed dataset is an important
resource for advancing image-based disaster management and multi-task machine
learning research.

    

### [[2108.12857] Uncertainty quantification for multiclass data description](http://arxiv.org/abs/2108.12857)


  In this manuscript, we propose a multiclass data description model based on
kernel Mahalanobis distance (MDD-KM) with self-adapting hyperparameter setting.
MDD-KM provides uncertainty quantification and can be deployed to build
classification systems for the realistic scenario where out-of-distribution
(OOD) samples are present among the test data. Given a test signal, a quantity
related to empirical kernel Mahalanobis distance between the signal and each of
the training classes is computed. Since these quantities correspond to the same
reproducing kernel Hilbert space, they are commensurable and hence can be
readily treated as classification scores without further application of fusion
techniques. To set kernel parameters, we exploit the fact that predictive
variance according to a Gaussian process (GP) is empirical kernel Mahalanobis
distance when a centralized kernel is used, and propose to use GP's negative
likelihood function as the cost function. We conduct experiments on the real
problem of avian note classification. We report a prototypical classification
system based on a hierarchical linear dynamical system with MDD-KM as a
component. Our classification system does not require sound event detection as
a preprocessing step, and is able to find instances of training avian notes
with varying length among OOD samples (corresponding to unknown notes of
disinterest) in the test audio clip. Domain knowledge is leveraged to make
crisp decisions from raw classification scores. We demonstrate the superior
performance of MDD-KM over possibilistic K-nearest neighbor.

    

### [[2108.12858] Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator](http://arxiv.org/abs/2108.12858)


  As one of the basic tasks of computer vision, object detection has been
widely used in many intelligent applications. However, object detection
algorithms are usually heavyweight in computation, hindering their
implementations on resource-constrained edge devices. Current edge-cloud
collaboration methods, such as CNN partition over Edge-cloud devices, are not
suitable for object detection since the huge data size of the intermediate
results will introduce extravagant communication costs. To address this
challenge, we propose a small-big model framework that deploys a big model in
the cloud and a small model on the edge devices. Upon receiving data, the edge
device operates a difficult-case discriminator to classify the images into easy
cases and difficult cases according to the specific semantics of the images.
The easy cases will be processed locally at the edge, and the difficult cases
will be uploaded to the cloud. Experimental results on the VOC, COCO, HELMET
datasets using two different object detection algorithms demonstrate that the
small-big model system can detect 94.01%-97.84% of objects with only about 50%
images uploaded to the cloud when using SSD. In addition, the small-big model
averagely reaches 91.22%- 92.52% end-to-end mAP of the scheme that uploading
all images to the cloud.

    

### [[2108.12862] Neural Network Gaussian Processes by Increasing Depth](http://arxiv.org/abs/2108.12862)


  Recent years have witnessed an increasing interest in the correspondence
between infinitely wide networks and Gaussian processes. Despite the
effectiveness and elegance of the current neural network Gaussian process
theory, to the best of our knowledge, all the neural network Gaussian processes
are essentially induced by increasing width. However, in the era of deep
learning, what concerns us more regarding a neural network is its depth as well
as how depth impacts the behaviors of a network. Inspired by a width-depth
symmetry consideration, we use a shortcut network to show that increasing the
depth of a neural network can also give rise to a Gaussian process, which is a
valuable addition to the existing theory and contributes to revealing the true
picture of deep learning. Beyond the proposed Gaussian process by depth, we
theoretically characterize its uniform tightness property and the smallest
eigenvalue of its associated kernel. These characterizations can not only
enhance our understanding of the proposed depth-induced Gaussian processes, but
also pave the way for future applications. Lastly, we examine the performance
of the proposed Gaussian process by regression experiments on two real-world
data sets.

    

### [[2108.12867] Partial Domain Adaptation without Domain Alignment](http://arxiv.org/abs/2108.12867)


  Unsupervised domain adaptation (UDA) aims to transfer knowledge from a
well-labeled source domain to a different but related unlabeled target domain
with identical label space. Currently, the main workhorse for solving UDA is
domain alignment, which has proven successful. However, it is often difficult
to find an appropriate source domain with identical label space. A more
practical scenario is so-called partial domain adaptation (PDA) in which the
source label set or space subsumes the target one. Unfortunately, in PDA, due
to the existence of the irrelevant categories in the source domain, it is quite
hard to obtain a perfect alignment, thus resulting in mode collapse and
negative transfer. Although several efforts have been made by down-weighting
the irrelevant source categories, the strategies used tend to be burdensome and
risky since exactly which irrelevant categories are unknown. These challenges
motivate us to find a relatively simpler alternative to solve PDA. To achieve
this, we first provide a thorough theoretical analysis, which illustrates that
the target risk is bounded by both model smoothness and between-domain
discrepancy. Considering the difficulty of perfect alignment in solving PDA, we
turn to focus on the model smoothness while discard the riskier domain
alignment to enhance the adaptability of the model. Specifically, we
instantiate the model smoothness as a quite simple intra-domain structure
preserving (IDSP). To our best knowledge, this is the first naive attempt to
address the PDA without domain alignment. Finally, our empirical results on
multiple benchmark datasets demonstrate that IDSP is not only superior to the
PDA SOTAs by a significant margin on some benchmarks (e.g., +10% on Cl->Rw and
+8% on Ar->Rw ), but also complementary to domain alignment in the standard UDA

    

### [[2108.12883] A Closed Loop Gradient Descent Algorithm applied to Rosenbrock's function](http://arxiv.org/abs/2108.12883)


  We introduce a novel adaptive damping technique for an inertial gradient
system which finds application as a gradient descent algorithm for
unconstrained optimisation. In an example using the non-convex Rosenbrock's
function, we show an improvement on existing momentum-based gradient
optimisation methods. Also using Lyapunov stability analysis, we demonstrate
the performance of the continuous-time version of the algorithm. Using
numerical simulations, we consider the performance of its discrete-time
counterpart obtained by using the symplectic Euler method of discretisation.

    

### [[2108.12898] Generating Answer Candidates for Quizzes and Answer-Aware Question Generators](http://arxiv.org/abs/2108.12898)


  In education, open-ended quiz questions have become an important tool for
assessing the knowledge of students. Yet, manually preparing such questions is
a tedious task, and thus automatic question generation has been proposed as a
possible alternative. So far, the vast majority of research has focused on
generating the question text, relying on question answering datasets with
readily picked answers, and the problem of how to come up with answer
candidates in the first place has been largely ignored. Here, we aim to bridge
this gap. In particular, we propose a model that can generate a specified
number of answer candidates for a given passage of text, which can then be used
by instructors to write questions manually or can be passed as an input to
automatic answer-aware question generators. Our experiments show that our
proposed answer candidate generation model outperforms several baselines.

    

### [[2108.12899] Fine-Grained Chemical Entity Typing with Multimodal Knowledge Representation](http://arxiv.org/abs/2108.12899)


  Automated knowledge discovery from trending chemical literature is essential
for more efficient biomedical research. How to extract detailed knowledge about
chemical reactions from the core chemistry literature is a new emerging
challenge that has not been well studied. In this paper, we study the new
problem of fine-grained chemical entity typing, which poses interesting new
challenges especially because of the complex name mentions frequently occurring
in chemistry literature and graphic representation of entities. We introduce a
new benchmark data set (CHEMET) to facilitate the study of the new task and
propose a novel multi-modal representation learning framework to solve the
problem of fine-grained chemical entity typing by leveraging external resources
with chemical structures and using cross-modal attention to learn effective
representation of text in the chemistry domain. Experiment results show that
the proposed framework outperforms multiple state-of-the-art methods.

    

### [[2108.12905] Lipschitz Continuity Guided Knowledge Distillation](http://arxiv.org/abs/2108.12905)


  Knowledge distillation has become one of the most important model compression
techniques by distilling knowledge from larger teacher networks to smaller
student ones. Although great success has been achieved by prior distillation
methods via delicately designing various types of knowledge, they overlook the
functional properties of neural networks, which makes the process of applying
those techniques to new tasks unreliable and non-trivial. To alleviate such
problem, in this paper, we initially leverage Lipschitz continuity to better
represent the functional characteristic of neural networks and guide the
knowledge distillation process. In particular, we propose a novel Lipschitz
Continuity Guided Knowledge Distillation framework to faithfully distill
knowledge by minimizing the distance between two neural networks' Lipschitz
constants, which enables teacher networks to better regularize student networks
and improve the corresponding performance. We derive an explainable
approximation algorithm with an explicit theoretical derivation to address the
NP-hard problem of calculating the Lipschitz constant. Experimental results
have shown that our method outperforms other benchmarks over several knowledge
distillation tasks (e.g., classification, segmentation and object detection) on
CIFAR-100, ImageNet, and PASCAL VOC datasets.

    

### [[2108.12914] Leveraging Transprecision Computing for Machine Vision Applications at the Edge](http://arxiv.org/abs/2108.12914)


  Machine vision tasks present challenges for resource constrained edge
devices, particularly as they execute multiple tasks with variable workloads. A
robust approach that can dynamically adapt in runtime while maintaining the
maximum quality of service (QoS) within resource constraints, is needed. The
paper presents a lightweight approach that monitors the runtime workload
constraint and leverages accuracy-throughput trade-off. Optimisation techniques
are included which find the configurations for each task for optimal accuracy,
energy and memory and manages transparent switching between configurations. For
an accuracy drop of 1%, we show a 1.6x higher achieved frame processing rate
with further improvements possible at lower accuracy.

    

### [[2108.12916] A Policy Efficient Reduction Approach to Convex Constrained Deep Reinforcement Learning](http://arxiv.org/abs/2108.12916)


  Although well-established in general reinforcement learning (RL), value-based
methods are rarely explored in constrained RL (CRL) for their incapability of
finding policies that can randomize among multiple actions. To apply
value-based methods to CRL, a recent groundbreaking line of game-theoretic
approaches uses the mixed policy that randomizes among a set of carefully
generated policies to converge to the desired constraint-satisfying policy.
However, these approaches require storing a large set of policies, which is not
policy efficient, and may incur prohibitive memory costs in constrained deep
RL. To address this problem, we propose an alternative approach. Our approach
first reformulates the CRL to an equivalent distance optimization problem. With
a specially designed linear optimization oracle, we derive a meta-algorithm
that solves it using any off-the-shelf RL algorithm and any conditional
gradient (CG) type algorithm as subroutines. We then propose a new variant of
the CG-type algorithm, which generalizes the minimum norm point (MNP) method.
The proposed method matches the convergence rate of the existing game-theoretic
approaches and achieves the worst-case optimal policy efficiency. The
experiments on a navigation task show that our method reduces the memory costs
by an order of magnitude, and meanwhile achieves better performance,
demonstrating both its effectiveness and efficiency.

    

### [[2108.12926] Photonic Quantum Policy Learning in OpenAI Gym](http://arxiv.org/abs/2108.12926)


  In recent years, near-term noisy intermediate scale quantum (NISQ) computing
devices have become available. One of the most promising application areas to
leverage such NISQ quantum computer prototypes is quantum machine learning.
While quantum neural networks are widely studied for supervised learning,
quantum reinforcement learning is still just an emerging field of this area. To
solve a classical continuous control problem, we use a continuous-variable
quantum machine learning approach. We introduce proximal policy optimization
for photonic variational quantum agents and also study the effect of the data
re-uploading. We present performance assessment via empirical study using
Strawberry Fields, a photonic simulator Fock backend and a hybrid training
framework connected to an OpenAI Gym environment and TensorFlow. For the
restricted CartPole problem, the two variations of the photonic policy learning
achieve comparable performance levels and a faster convergence than the
baseline classical neural network of same number of trainable parameters.

    

### [[2108.12929] Convolutional versus Dense Neural Networks: Comparing the Two Neural Networks Performance in Predicting Building Operational Energy Use Based on the Building Shape](http://arxiv.org/abs/2108.12929)


  A building self-shading shape impacts substantially on the amount of direct
sunlight received by the building and contributes significantly to building
operational energy use, in addition to other major contributing variables, such
as materials and window-to-wall ratios. Deep Learning has the potential to
assist designers and engineers by efficiently predicting building energy
performance. This paper assesses the applicability of two different neural
networks structures, Dense Neural Network (DNN) and Convolutional Neural
Network (CNN), for predicting building operational energy use with respect to
building shape. The comparison between the two neural networks shows that the
DNN model surpasses the CNN model in performance, simplicity, and computation
time. However, image-based CNN has the benefit of utilizing architectural
graphics that facilitates design communication.

    

### [[2108.12943] Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks](http://arxiv.org/abs/2108.12943)


  Convolution neural networks have been successful in solving many socially
important and economically significant problems. Their ability to learn complex
high-dimensional functions hierarchically can be attributed to the use of
nonlinear activation functions. A key discovery that made training deep
networks feasible was the adoption of the Rectified Linear Unit (ReLU)
activation function to alleviate the vanishing gradient problem caused by using
saturating activation functions. Since then many improved variants of the ReLU
activation have been proposed. However a majority of activation functions used
today are non-oscillatory and monotonically increasing due to their biological
plausibility. This paper demonstrates that oscillatory activation functions can
improve gradient flow and reduce network size. It is shown that oscillatory
activation functions allow neurons to switch classification (sign of output)
within the interior of neuronal hyperplane positive and negative half-spaces
allowing complex decisions with fewer neurons. A new oscillatory activation
function C(z) = z cos z that outperforms Sigmoids, Swish, Mish and ReLU on a
variety of architectures and benchmarks is presented. This new activation
function allows even single neurons to exhibit nonlinear decision boundaries.
This paper presents a single neuron solution to the famous XOR problem.
Experimental results indicate that replacing the activation function in the
convolutional layers with C(z) significantly improves performance on CIFAR-10,
CIFAR-100 and Imagenette.

    

### [[2108.12947] Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](http://arxiv.org/abs/2108.12947)


  Detecting and localizing image manipulation are necessary to counter
malicious use of image editing techniques. Accordingly, it is essential to
distinguish between authentic and tampered regions by analyzing intrinsic
statistics in an image. We focus on JPEG compression artifacts left during
image acquisition and editing. We propose a convolutional neural network (CNN)
that uses discrete cosine transform (DCT) coefficients, where compression
artifacts remain, to localize image manipulation. Standard CNNs cannot learn
the distribution of DCT coefficients because the convolution throws away the
spatial coordinates, which are essential for DCT coefficients. We illustrate
how to design and train a neural network that can learn the distribution of DCT
coefficients. Furthermore, we introduce Compression Artifact Tracing Network
(CAT-Net) that jointly uses image acquisition artifacts and compression
artifacts. It significantly outperforms traditional and deep neural
network-based methods in detecting and localizing tampered regions.

    

### [[2108.12953] Multi-Channel Transformer Transducer for Speech Recognition](http://arxiv.org/abs/2108.12953)


  Multi-channel inputs offer several advantages over single-channel, to improve
the robustness of on-device speech recognition systems. Recent work on
multi-channel transformer, has proposed a way to incorporate such inputs into
end-to-end ASR for improved accuracy. However, this approach is characterized
by a high computational complexity, which prevents it from being deployed in
on-device systems. In this paper, we present a novel speech recognition model,
Multi-Channel Transformer Transducer (MCTT), which features end-to-end
multi-channel training, low computation cost, and low latency so that it is
suitable for streaming decoding in on-device speech recognition. In a far-field
in-house dataset, our MCTT outperforms stagewise multi-channel models with
transformer-transducer up to 6.01% relative WER improvement (WERR). In
addition, MCTT outperforms the multi-channel transformer up to 11.62% WERR, and
is 15.8 times faster in terms of inference speed. We further show that we can
improve the computational cost of MCTT by constraining the future and previous
context in attention computations.

    

### [[2108.12955] Unsupervised Learning of Deep Features for Music Segmentation](http://arxiv.org/abs/2108.12955)


  Music segmentation refers to the dual problem of identifying boundaries
between, and labeling, distinct music segments, e.g., the chorus, verse, bridge
etc. in popular music. The performance of a range of music segmentation
algorithms has been shown to be dependent on the audio features chosen to
represent the audio. Some approaches have proposed learning feature
transformations from music segment annotation data, although, such data is time
consuming or expensive to create and as such these approaches are likely
limited by the size of their datasets. While annotated music segmentation data
is a scarce resource, the amount of available music audio is much greater. In
the neighboring field of semantic audio unsupervised deep learning has shown
promise in improving the performance of solutions to the query-by-example and
sound classification tasks. In this work, unsupervised training of deep feature
embeddings using convolutional neural networks (CNNs) is explored for music
segmentation. The proposed techniques exploit only the time proximity of audio
features that is implicit in any audio timeline. Employing these embeddings in
a classic music segmentation algorithm is shown not only to significantly
improve the performance of this algorithm, but obtain state of the art
performance in unsupervised music segmentation.

    

### [[2108.12956] Normalizing Field Flows: Solving forward and inverse stochastic differential equations using Physics-Informed flow model](http://arxiv.org/abs/2108.12956)


  We introduce in this work the normalizing field flows (NFF) for learning
random fields from scattered measurements. More precisely, we construct a
bijective transformation (a normalizing flow characterizing by neural networks)
between a reference random field (say, a Gaussian random field with the
Karhunen-Loève expansion structure) and the target stochastic field, where
the KL expansion coefficients and the invertible networks are trained by
maximizing the sum of the log-likelihood on scattered measurements. This NFF
model can be used to solve data-driven forward, inverse, and mixed
forward/inverse stochastic partial differential equations in a unified
framework. We demonstrate the capability of the proposed NFF model for learning
Non Gaussian processes, mixed Gaussian processes, and forward & inverse
stochastic partial differential equations.

    

### [[2108.12976] Approximating Pandora's Box with Correlations](http://arxiv.org/abs/2108.12976)


  The Pandora's Box problem asks to find a search strategy over $n$
alternatives given stochastic information about their values, aiming to
minimize the sum of the search cost and the value of the chosen alternative.
Even though the case of independently distributed values is well understood,
our algorithmic understanding of the problem is very limited once the
independence assumption is dropped.
Our work aims to characterize the complexity of approximating the Pandora's
Box problem under correlated value distributions. To that end, we present a
general reduction to a simpler version of Pandora's Box, that only asks to find
a value below a certain threshold, and eliminates the need to reason about
future values that will arise during the search. Using this general tool, we
study two cases of correlation; the case of explicitly given distributions of
support $m$ and the case of mixtures of $m$ product distributions.
$\bullet$ In the first case, we connect Pandora's Box to the well studied
problem of Optimal Decision Tree, obtaining an $O(\log m)$ approximation but
also showing that the problem is strictly easier as it is equivalent (up to
constant factors) to the Uniform Decision Tree problem.
$\bullet$ In the case of mixtures of product distributions, the problem is
again related to the noisy variant of Optimal Decision Tree which is
significantly more challenging. We give a constant-factor approximation that
runs in time $n^{ \tilde O( m^2/\varepsilon^2 ) }$ for $m$ mixture components
whose marginals on every alternative are either identical or separated in TV
distance by $\varepsilon$.

    

### [[2108.12978] Private Multi-Task Learning: Formulation and Applications to Federated Learning](http://arxiv.org/abs/2108.12978)


  Many problems in machine learning rely on multi-task learning (MTL), in which
the goal is to solve multiple related machine learning tasks simultaneously.
MTL is particularly relevant for privacy-sensitive applications in areas such
as healthcare, finance, and IoT computing, where sensitive data from multiple,
varied sources are shared for the purpose of learning. In this work, we
formalize notions of task-level privacy for MTL via joint differential
privacy(JDP), a relaxation of differential privacy for mechanism design and
distributed optimization. We then propose an algorithm for mean-regularized
MTL, an objective commonly used for applications in personalized federated
learning, subject to JDP. We analyze our objective and solver, providing
certifiable guarantees on both privacy and utility. Empirically, we find that
our method allows for improved privacy/utility trade-offs relative to global
baselines across common federated learning benchmarks.

    

### [[2108.12982] Adversarial Stein Training for Graph Energy Models](http://arxiv.org/abs/2108.12982)


  Learning distributions over graph-structured data is a challenging task with
many applications in biology and chemistry. In this work we use an energy-based
model (EBM) based on multi-channel graph neural networks (GNN) to learn
permutation invariant unnormalized density functions on graphs. Unlike standard
EBM training methods our approach is to learn the model via minimizing
adversarial stein discrepancy. Samples from the model can be obtained via
Langevin dynamics based MCMC. We find that this approach achieves competitive
results on graph generation compared to benchmark models.

    

### [[2108.12988] Learning Meta Representations for Agents in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2108.12988)


  In multi-agent reinforcement learning, the behaviors that agents learn in a
single Markov Game (MG) are typically confined to the given agent number (i.e.,
population size). Every single MG induced by varying population sizes may
possess distinct optimal joint strategies and game-specific knowledge, which
are modeled independently in modern multi-agent algorithms. In this work, we
focus on creating agents that generalize across population-varying MGs. Instead
of learning a unimodal policy, each agent learns a policy set that is formed by
effective strategies across a variety of games. We propose Meta Representations
for Agents (MRA) that explicitly models the game-common and game-specific
strategic knowledge. By representing the policy sets with multi-modal latent
policies, the common strategic knowledge and diverse strategic modes are
discovered with an iterative optimization procedure. We prove that as an
approximation to a constrained mutual information maximization objective, the
learned policies can reach Nash Equilibrium in every evaluation MG under the
assumption of Lipschitz game on a sufficiently large latent space. When
deploying it at practical latent models with limited size, fast adaptation can
be achieved by leveraging the first-order gradient information. Extensive
experiments show the effectiveness of MRA on both training performance and
generalization ability in hard and unseen games.

    

### [[2108.12992] SHIFT15M: Multiobjective Large-Scale Fashion Dataset with Distributional Shifts](http://arxiv.org/abs/2108.12992)


  Many machine learning algorithms assume that the training data and the test
data follow the same distribution. However, such assumptions are often violated
in real-world machine learning problems. In this paper, we propose SHIFT15M, a
dataset that can be used to properly evaluate models in situations where the
distribution of data changes between training and testing. The SHIFT15M dataset
has several good properties: (i) Multiobjective. Each instance in the dataset
has several numerical values that can be used as target variables. (ii)
Large-scale. The SHIFT15M dataset consists of 15million fashion images. (iii)
Coverage of types of dataset shifts. SHIFT15M contains multiple dataset shift
problem settings (e.g., covariate shift or target shift). SHIFT15M also enables
the performance evaluation of the model under various magnitudes of dataset
shifts by switching the magnitude. In addition, we provide software to handle
SHIFT15M in a very simple way: this https URL.

    

### [[2108.13009] Communication-Computation Efficient Device-Edge Co-Inference via AutoML](http://arxiv.org/abs/2108.13009)


  Device-edge co-inference, which partitions a deep neural network between a
resource-constrained mobile device and an edge server, recently emerges as a
promising paradigm to support intelligent mobile applications. To accelerate
the inference process, on-device model sparsification and intermediate feature
compression are regarded as two prominent techniques. However, as the on-device
model sparsity level and intermediate feature compression ratio have direct
impacts on computation workload and communication overhead respectively, and
both of them affect the inference accuracy, finding the optimal values of these
hyper-parameters brings a major challenge due to the large search space. In
this paper, we endeavor to develop an efficient algorithm to determine these
hyper-parameters. By selecting a suitable model split point and a pair of
encoder/decoder for the intermediate feature vector, this problem is casted as
a sequential decision problem, for which, a novel automated machine learning
(AutoML) framework is proposed based on deep reinforcement learning (DRL).
Experiment results on an image classification task demonstrate the
effectiveness of the proposed framework in achieving a better
communication-computation trade-off and significant inference speedup against
various baseline schemes.

    

### [[2108.13032] Shatter: An Efficient Transformer Encoder with Single-Headed Self-Attention and Relative Sequence Partitioning](http://arxiv.org/abs/2108.13032)


  The highly popular Transformer architecture, based on self-attention, is the
foundation of large pretrained models such as BERT, that have become an
enduring paradigm in NLP. While powerful, the computational resources and time
required to pretrain such models can be prohibitive. In this work, we present
an alternative self-attention architecture, Shatter, that more efficiently
encodes sequence information by softly partitioning the space of relative
positions and applying different value matrices to different parts of the
sequence. This mechanism further allows us to simplify the multi-headed
attention in Transformer to single-headed. We conduct extensive experiments
showing that Shatter achieves better performance than BERT, with pretraining
being faster per step (15% on TPU), converging in fewer steps, and offering
considerable memory savings (>50%). Put together, Shatter can be pretrained on
8 V100 GPUs in 7 days, and match the performance of BERT_Base -- making the
cost of pretraining much more affordable.

    

### [[2108.13034] Evaluating Bayes Error Estimators on Read-World Datasets with FeeBee](http://arxiv.org/abs/2108.13034)


  The Bayes error rate (BER) is a fundamental concept in machine learning that
quantifies the best possible accuracy any classifier can achieve on a fixed
probability distribution. Despite years of research on building estimators of
lower and upper bounds for the BER, these were usually compared only on
synthetic datasets with known probability distributions, leaving two key
questions unanswered: (1) How well do they perform on real-world datasets?, and
(2) How practical are they? Answering these is not trivial. Apart from the
obvious challenge of an unknown BER for real-world datasets, there are two main
aspects any BER estimator needs to overcome in order to be applicable in
real-world settings: (1) the computational and sample complexity, and (2) the
sensitivity and selection of hyper-parameters. In this work, we propose FeeBee,
the first principled framework for analyzing and comparing BER estimators on
any modern real-world dataset with unknown probability distribution. We achieve
this by injecting a controlled amount of label noise and performing multiple
evaluations on a series of different noise levels, supported by a theoretical
result which allows drawing conclusions about the evolution of the BER. By
implementing and analyzing 7 multi-class BER estimators on 6 commonly used
datasets of the computer vision and NLP domains, FeeBee allows a thorough study
of these estimators, clearly identifying strengths and weaknesses of each,
whilst being easily deployable on any future BER estimator.

    

### [[2108.13038] Integrated Decision and Control at Multi-Lane Intersections with Mixed Traffic Flow](http://arxiv.org/abs/2108.13038)


  Autonomous driving at intersections is one of the most complicated and
accident-prone traffic scenarios, especially with mixed traffic participants
such as vehicles, bicycles and pedestrians. The driving policy should make safe
decisions to handle the dynamic traffic conditions and meet the requirements of
on-board computation. However, most of the current researches focuses on
simplified intersections considering only the surrounding vehicles and
idealized traffic lights. This paper improves the integrated decision and
control framework and develops a learning-based algorithm to deal with complex
intersections with mixed traffic flows, which can not only take account of
realistic characteristics of traffic lights, but also learn a safe policy under
different safety constraints. We first consider different velocity models for
green and red lights in the training process and use a finite state machine to
handle different modes of light transformation. Then we design different types
of distance constraints for vehicles, traffic lights, pedestrians, bicycles
respectively and formulize the constrained optimal control problems (OCPs) to
be optimized. Finally, reinforcement learning (RL) with value and policy
networks is adopted to solve the series of OCPs. In order to verify the safety
and efficiency of the proposed method, we design a multi-lane intersection with
the existence of large-scale mixed traffic participants and set practical
traffic light phases. The simulation results indicate that the trained decision
and control policy can well balance safety and tracking performance. Compared
with model predictive control (MPC), the computational time is three orders of
magnitude lower.

    

### [[2108.13039] An Interpretable Web-based Glioblastoma Multiforme Prognosis Prediction Tool using Random Forest Model](http://arxiv.org/abs/2108.13039)


  We propose predictive models that estimate GBM patients' health status of
one-year after treatments (Classification task), predict the long-term
prognosis of GBM patients at an individual level (Survival task). We used total
of 467 GBM patients' clinical profile consists of 13 features and two follow-up
dates. For baseline models of random forest classifier(RFC) and random survival
forest model (RSF), we introduced generalized linear model (GLM), support
vector machine (SVM) and Cox proportional hazardous model (COX), accelerated
failure time model (AFT) respectively. After preprocessing and prefixing
stratified 5-fold data set, we generated best performing models for model types
using recursive feature elimination process. Total 10, 4, and 13 features were
extracted for best performing one-year survival/progression status RFC models
and RSF model via the recursive feature elimination process. In classification
task, AUROC of best performing RFC recorded 0.6990 (for one-year survival
status classification) and 0.7076 (for one-year progression classification)
while that of second best baseline models (GLM in both cases) recorded 0.6691
and 0.6997 respectively. About survival task, the highest C-index of 0.7157 and
the lowest IBS of 0.1038 came from the best performing RSF model while that of
second best baseline models were 0.6556 and 0.1139 respectively. A simplified
linear correlation (extracted from LIME and virtual patient group analysis)
between each feature and prognosis of GBM patient were consistent with proven
medical knowledge. Our machine learning models suggest that the top three
prognostic factors for GBM patient survival were MGMT gene promoter, the extent
of resection, and age. To the best of our knowledge, this study is the very
first study introducing a interpretable and medical knowledge consistent GBM
prognosis predictive models.

    

### [[2108.13041] Auto-Split: A General Framework of Collaborative Edge-Cloud AI](http://arxiv.org/abs/2108.13041)


  In many industry scale applications, large and resource consuming machine
learning models reside in powerful cloud servers. At the same time, large
amounts of input data are collected at the edge of cloud. The inference results
are also communicated to users or passed to downstream tasks at the edge. The
edge often consists of a large number of low-power devices. It is a big
challenge to design industry products to support sophisticated deep model
deployment and conduct model inference in an efficient manner so that the model
accuracy remains high and the end-to-end latency is kept low. This paper
describes the techniques and engineering practice behind Auto-Split, an
edge-cloud collaborative prototype of Huawei Cloud. This patented technology is
already validated on selected applications, is on its way for broader
systematic edge-cloud application integration, and is being made available for
public use as an automated pipeline service for end-to-end cloud-edge
collaborative intelligence deployment. To the best of our knowledge, there is
no existing industry product that provides the capability of Deep Neural
Network (DNN) splitting.

    

### [[2108.13046] Data-driven Small-signal Modeling for Converter-based Power Systems](http://arxiv.org/abs/2108.13046)


  This article details a complete procedure to derive a data-driven
small-signal-based model useful to perform converter-based power system related
studies. To compute the model, Decision Tree (DT) regression, both using single
DT and ensemble DT, and Spline regression have been employed and their
performances have been compared, in terms of accuracy, training and computing
time. The methodology includes a comprehensive step-by-step procedure to
develop the model: data generation by conventional simulation and mathematical
models, databases (DBs) arrangement, regression training and testing, realizing
prediction for new instances. The methodology has been developed using an
essential network and then tested on a more complex system, to show the
validity and usefulness of the suggested approach. Both power systems test
cases have the essential characteristics of converter-based power systems,
simulating high penetration of converter interfaced generation and the presence
of HVDC links. Moreover, it is proposed how to represent in a visual manner the
results of the small-signal stability analysis for a wide range of system
operating conditions, exploiting DT regressions. Finally, the possible
applications of the model are discussed, highlighting the potential of the
developed model in further power system small-signal related studies.

    

### [[2108.13049] Single Node Injection Attack against Graph Neural Networks](http://arxiv.org/abs/2108.13049)


  Node injection attack on Graph Neural Networks (GNNs) is an emerging and
practical attack scenario that the attacker injects malicious nodes rather than
modifying original nodes or edges to affect the performance of GNNs. However,
existing node injection attacks ignore extremely limited scenarios, namely the
injected nodes might be excessive such that they may be perceptible to the
target GNN. In this paper, we focus on an extremely limited scenario of single
node injection evasion attack, i.e., the attacker is only allowed to inject one
single node during the test phase to hurt GNN's performance. The discreteness
of network structure and the coupling effect between network structure and node
features bring great challenges to this extremely limited scenario. We first
propose an optimization-based method to explore the performance upper bound of
single node injection evasion attack. Experimental results show that 100%,
98.60%, and 94.98% nodes on three public datasets are successfully attacked
even when only injecting one node with one edge, confirming the feasibility of
single node injection evasion attack. However, such an optimization-based
method needs to be re-optimized for each attack, which is computationally
unbearable. To solve the dilemma, we further propose a Generalizable Node
Injection Attack model, namely G-NIA, to improve the attack efficiency while
ensuring the attack performance. Experiments are conducted across three
well-known GNNs. Our proposed G-NIA significantly outperforms state-of-the-art
baselines and is 500 times faster than the optimization-based method when
inferring.

    

### [[2108.13051] Demystifying Drug Repurposing Domain Comprehension with Knowledge Graph Embedding](http://arxiv.org/abs/2108.13051)


  Drug repurposing is more relevant than ever due to drug development's rising
costs and the need to respond to emerging diseases quickly. Knowledge graph
embedding enables drug repurposing using heterogeneous data sources combined
with state-of-the-art machine learning models to predict new drug-disease links
in the knowledge graph. As in many machine learning applications, significant
work is still required to understand the predictive models' behavior. We
propose a structured methodology to understand better machine learning models'
results for drug repurposing, suggesting key elements of the knowledge graph to
improve predictions while saving computational resources. We reduce the
training set of 11.05% and the embedding space by 31.87%, with only a 2%
accuracy reduction, and increase accuracy by 60% on the open ogbl-biokg graph
adding only 1.53% new triples.

    

### [[2108.13054] Wasserstein Generative Adversarial Uncertainty Quantification in Physics-Informed Neural Networks](http://arxiv.org/abs/2108.13054)


  In this paper, we study a physics-informed algorithm for Wasserstein
Generative Adversarial Networks (WGANs) for uncertainty quantification in
solutions of partial differential equations. By using groupsort activation
functions in adversarial network discriminators, network generators are
utilized to learn the uncertainty in solutions of partial differential
equations observed from the initial/boundary data. Under mild assumptions, we
show that the generalization error of the computed generator converges to the
approximation error of the network with high probability, when the number of
samples are sufficiently taken. According to our established error bound, we
also find that our physics-informed WGANs have higher requirement for the
capacity of discriminators than that of generators. Numerical results on
synthetic examples of partial differential equations are reported to validate
our theoretical results and demonstrate how uncertainty quantification can be
obtained for solutions of partial differential equations and the distributions
of initial/boundary data.

    

### [[2108.13066] To tune or not to tune? An Approach for Recommending Important Hyperparameters](http://arxiv.org/abs/2108.13066)


  Novel technologies in automated machine learning ease the complexity of
algorithm selection and hyperparameter optimization. Hyperparameters are
important for machine learning models as they significantly influence the
performance of machine learning models. Many optimization techniques have
achieved notable success in hyperparameter tuning and surpassed the performance
of human experts. However, depending on such techniques as blackbox algorithms
can leave machine learning practitioners without insight into the relative
importance of different hyperparameters. In this paper, we consider building
the relationship between the performance of the machine learning models and
their hyperparameters to discover the trend and gain insights, with empirical
results based on six classifiers and 200 datasets. Our results enable users to
decide whether it is worth conducting a possibly time-consuming tuning
strategy, to focus on the most important hyperparameters, and to choose
adequate hyperparameter spaces for tuning. The results of our experiments show
that gradient boosting and Adaboost outperform other classifiers across 200
problems. However, they need tuning to boost their performance. Overall, the
results obtained from this study provide a quantitative basis to focus efforts
toward guided automated hyperparameter optimization and contribute toward the
development of better-automated machine learning frameworks.

    

### [[2108.13083] An Introduction to Variational Inference](http://arxiv.org/abs/2108.13083)


  Approximating complex probability densities is a core problem in modern
statistics. In this paper, we introduce the concept of Variational Inference
(VI), a popular method in machine learning that uses optimization techniques to
estimate complex probability densities. This property allows VI to converge
faster than classical methods, such as, Markov Chain Monte Carlo sampling.
Conceptually, VI works by choosing a family of probability density functions
and then finding the one closest to the actual probability density -- often
using the Kullback-Leibler (KL) divergence as the optimization metric. We
introduce the Evidence Lower Bound to tractably compute the approximated
probability density and we review the ideas behind mean-field variational
inference. Finally, we discuss the applications of VI to variational
auto-encoders (VAE) and VAE-Generative Adversarial Network (VAE-GAN). With this
paper, we aim to explain the concept of VI and assist in future research with
this approach.

    

### [[2108.13092] GeoVectors: A Linked Open Corpus of OpenStreetMap Embeddings on World Scale](http://arxiv.org/abs/2108.13092)


  OpenStreetMap (OSM) is currently the richest publicly available information
source on geographic entities (e.g., buildings and roads) worldwide. However,
using OSM entities in machine learning models and other applications is
challenging due to the large scale of OSM, the extreme heterogeneity of entity
annotations, and a lack of a well-defined ontology to describe entity semantics
and properties. This paper presents GeoVectors - a unique, comprehensive
world-scale linked open corpus of OSM entity embeddings covering the entire OSM
dataset and providing latent representations of over 980 million geographic
entities in 180 countries. The GeoVectors corpus captures semantic and
geographic dimensions of OSM entities and makes these entities directly
accessible to machine learning algorithms and semantic applications. We create
a semantic description of the GeoVectors corpus, including identity links to
the Wikidata and DBpedia knowledge graphs to supply context information.
Furthermore, we provide a SPARQL endpoint - a semantic interface that offers
direct access to the semantic and latent representations of geographic entities
in OSM.

    

### [[2108.13093] Investigating Vulnerabilities of Deep Neural Policies](http://arxiv.org/abs/2108.13093)


  Reinforcement learning policies based on deep neural networks are vulnerable
to imperceptible adversarial perturbations to their inputs, in much the same
way as neural network image classifiers. Recent work has proposed several
methods to improve the robustness of deep reinforcement learning agents to
adversarial perturbations based on training in the presence of these
imperceptible perturbations (i.e. adversarial training). In this paper, we
study the effects of adversarial training on the neural policy learned by the
agent. In particular, we follow two distinct parallel approaches to investigate
the outcomes of adversarial training on deep neural policies based on
worst-case distributional shift and feature sensitivity. For the first
approach, we compare the Fourier spectrum of minimal perturbations computed for
both adversarially trained and vanilla trained neural policies. Via experiments
in the OpenAI Atari environments we show that minimal perturbations computed
for adversarially trained policies are more focused on lower frequencies in the
Fourier domain, indicating a higher sensitivity of these policies to low
frequency perturbations. For the second approach, we propose a novel method to
measure the feature sensitivities of deep neural policies and we compare these
feature sensitivity differences in state-of-the-art adversarially trained deep
neural policies and vanilla trained deep neural policies. We believe our
results can be an initial step towards understanding the relationship between
adversarial training and different notions of robustness for neural policies.

    

### [[2108.13097] A fast point solver for deep nonlinear function approximators](http://arxiv.org/abs/2108.13097)


  Deep kernel processes (DKPs) generalise Bayesian neural networks, but do not
require us to represent either features or weights. Instead, at each hidden
layer they represent and optimize a flexible kernel. Here, we develop a
Newton-like method for DKPs that converges in around 10 steps, exploiting
matrix solvers initially developed in the control theory literature. These are
many times faster the usual gradient descent approach. We generalise to
arbitrary DKP architectures, by developing "kernel backprop", and algorithms
for "kernel autodiff". While these methods currently are not Bayesian as they
give point estimates and scale poorly as they are cubic in the number of
datapoints, we hope they will form the basis of a new class of much more
efficient approaches to optimizing deep nonlinear function approximators.

    

### [[2108.13099] Open Set RF Fingerprinting using Generative Outlier Augmentation](http://arxiv.org/abs/2108.13099)


  RF devices can be identified by unique imperfections embedded in the signals
they transmit called RF fingerprints. The closed set classification of such
devices, where the identification must be made among an authorized set of
transmitters, has been well explored. However, the much more difficult problem
of open set classification, where the classifier needs to reject unauthorized
transmitters while recognizing authorized transmitters, has only been recently
visited. So far, efforts at open set classification have largely relied on the
utilization of signal samples captured from a known set of unauthorized
transmitters to aid the classifier learn unauthorized transmitter fingerprints.
Since acquiring new transmitters to use as known transmitters is highly
expensive, we propose to use generative deep learning methods to emulate
unauthorized signal samples for the augmentation of training datasets. We
develop two different data augmentation techniques, one that exploits a limited
number of known unauthorized transmitters and the other that does not require
any unauthorized transmitters. Experiments conducted on a dataset captured from
a WiFi testbed indicate that data augmentation allows for significant increases
in open set classification accuracy, especially when the authorized set is
small.

    

### [[2108.13122] Tune It or Don't Use It: Benchmarking Data-Efficient Image Classification](http://arxiv.org/abs/2108.13122)


  Data-efficient image classification using deep neural networks in settings,
where only small amounts of labeled data are available, has been an active
research area in the recent past. However, an objective comparison between
published methods is difficult, since existing works use different datasets for
evaluation and often compare against untuned baselines with default
hyper-parameters. We design a benchmark for data-efficient image classification
consisting of six diverse datasets spanning various domains (e.g., natural
images, medical imagery, satellite data) and data types (RGB, grayscale,
multispectral). Using this benchmark, we re-evaluate the standard cross-entropy
baseline and eight methods for data-efficient deep learning published between
2017 and 2021 at renowned venues. For a fair and realistic comparison, we
carefully tune the hyper-parameters of all methods on each dataset.
Surprisingly, we find that tuning learning rate, weight decay, and batch size
on a separate validation split results in a highly competitive baseline, which
outperforms all but one specialized method and performs competitively to the
remaining one.

    

### [[2108.13137] Thermodynamics-based Artificial Neural Networks (TANN) for multiscale modeling of materials with inelastic microstructure](http://arxiv.org/abs/2108.13137)


  The mechanical behavior of inelastic materials with microstructure is very
complex and hard to grasp with heuristic, empirical constitutive models. For
this purpose, multiscale, homogenization approaches are often used for
performing reliable, accurate predictions of the macroscopic mechanical
behavior of microstructured solids. Nevertheless, the calculation cost of such
approaches is extremely high and prohibitive for real-scale applications
involving inelastic materials. Recently, data-driven approaches based on deep
learning have risen as a promising alternative to replace ad-hoc constitutive
laws and speed-up multiscale numerical methods. However, such approaches lack a
rigorous frame based on the laws of physics. As a result, their application to
model materials with complex microstructure in inelasticity is not yet
established. Here, we propose Thermodynamics-based Artificial Neural Networks
(TANN) for the constitutive modeling of materials with inelastic and complex
microstructure. Our approach integrates thermodynamics-aware dimensionality
reduction techniques and deep neural networks to identify the constitutive laws
and the internal state variables of complex inelastic materials. The ability of
TANN in delivering high-fidelity, physically consistent predictions is
demonstrated through several examples both at the microscopic and macroscopic
scale. In particular, we show the efficiency and accuracy of TANN in predicting
the average and local stress-strain response, the internal energy and the
dissipation of both regular and perturbed lattice microstructures in
inelasticity. Finally, a double-scale homogenization scheme is used to solve a
large scale boundary value problem. The high performance of the homogenized
model using TANN is illustrated through detailed comparisons. An excellent
agreement is shown for a variety of monotonous and cyclic stress-strain paths.

    

### [[2108.13171] Functional Nanomaterials Design in the Workflow of Building Machine-Learning Models](http://arxiv.org/abs/2108.13171)


  Machine-learning (ML) techniques have revolutionized a host of research
fields of chemical and materials science with accelerated, high-efficiency
discoveries in design, synthesis, manufacturing, characterization and
application of novel functional materials, especially at the nanometre scale.
The reason is the time efficiency, prediction accuracy and good generalization
abilities, which gradually replaces the traditional experimental or
computational work. With enormous potentiality to tackle more real-world
problems, ML provides a more comprehensive insight into combinations with
molecules/materials under the fundamental procedures for constructing ML
models, like predicting properties or functionalities from given parameters,
nanoarchitecture design and generating specific models for other purposes. The
key to the advances in nanomaterials discovery is how input fingerprints and
output values can be linked quantitatively. Finally, some great opportunities
and technical challenges are concluded in this fantastic field.

    

### [[1802.10489] As you like it: Localization via paired comparisons](http://arxiv.org/abs/1802.10489)


  Suppose that we wish to estimate a vector $\mathbf{x}$ from a set of binary
paired comparisons of the form "$\mathbf{x}$ is closer to $\mathbf{p}$ than to
$\mathbf{q}$" for various choices of vectors $\mathbf{p}$ and $\mathbf{q}$. The
problem of estimating $\mathbf{x}$ from this type of observation arises in a
variety of contexts, including nonmetric multidimensional scaling, "unfolding,"
and ranking problems, often because it provides a powerful and flexible model
of preference. We describe theoretical bounds for how well we can expect to
estimate $\mathbf{x}$ under a randomized model for $\mathbf{p}$ and
$\mathbf{q}$. We also present results for the case where the comparisons are
noisy and subject to some degree of error. Additionally, we show that under a
randomized model for $\mathbf{p}$ and $\mathbf{q}$, a suitable number of binary
paired comparisons yield a stable embedding of the space of target vectors.
Finally, we also show that we can achieve significant gains by adaptively
changing the distribution for choosing $\mathbf{p}$ and $\mathbf{q}$.

    

### [[1810.05640] Inventory Balancing with Online Learning](http://arxiv.org/abs/1810.05640)


  We study a general problem of allocating limited resources to heterogeneous
customers over time under model uncertainty. Each type of customer can be
serviced using different actions, each of which stochastically consumes some
combination of resources, and returns different rewards for the resources
consumed. We consider a general model where the resource consumption
distribution associated with each (customer type, action)-combination is not
known, but is consistent and can be learned over time. In addition, the
sequence of customer types to arrive over time is arbitrary and completely
unknown.
We overcome both the challenges of model uncertainty and customer
heterogeneity by judiciously synthesizing two algorithmic frameworks from the
literature: inventory balancing, which "reserves" a portion of each resource
for high-reward customer types which could later arrive, and online learning,
which shows how to "explore" the resource consumption distributions of each
customer type under different actions. We define an auxiliary problem, which
allows for existing competitive ratio and regret bounds to be seamlessly
integrated. Furthermore, we show that the performance guarantee generated by
our framework is tight, that is, we provide an information-theoretic lower
bound which shows that both the loss from competitive ratio and the loss for
regret are relevant in the combined problem.
Finally, we demonstrate the efficacy of our algorithms on a publicly
available hotel data set. Our framework is highly practical in that it requires
no historical data (no fitted customer choice models, nor forecasting of
customer arrival patterns) and can be used to initialize allocation strategies
in fast-changing environments.

    

### [[1810.07368] Learning the Compositional Spaces for Generalized Zero-shot Learning](http://arxiv.org/abs/1810.07368)


  This paper studies the problem of Generalized Zero-shot Learning (G-ZSL),
whose goal is to classify instances belonging to both seen and unseen classes
at the test time. We propose a novel space decomposition method to solve G-ZSL.
Some previous models with space decomposition operations only calibrate the
confident prediction of source classes (W-SVM [46]) or take target-class
instances as outliers [49]. In contrast, we propose to directly estimate and
fine-tune the decision boundary between the source and the target classes.
Specifically, we put forward a framework that enables to learn compositional
spaces by splitting the instances into Source, Target, and Uncertain spaces and
perform recognition in each space, where the uncertain space contains instances
whose labels cannot be confidently predicted. We use two statistical tools,
namely, bootstrapping and Kolmogorov-Smirnov (K-S) Test, to learn the
compositional spaces for G-ZSL. We validate our method extensively on multiple
G-ZSL benchmarks, on which it achieves state-of-the-art performances.

    

### [[1901.01585] Solving L1-regularized SVMs and related linear programs: Revisiting the effectiveness of Column and Constraint Generation](http://arxiv.org/abs/1901.01585)


  The linear Support Vector Machine (SVM) is a classic classification technique
in machine learning. Motivated by applications in modern high dimensional
statistics, we consider penalized SVM problems involving the minimization of a
hinge-loss function with a convex sparsity-inducing regularizer such as: the
L1-norm on the coefficients, its grouped generalization and the sorted
L1-penalty (aka Slope). Each problem can be expressed as a Linear Program (LP)
and is computationally challenging when the number of features and/or samples
is large -- the current state of algorithms for these problems is rather
nascent when compared to the usual L2-regularized linear SVM. To this end, we
propose new computational algorithms for these LPs by bringing together
techniques from (a) classical column (and constraint) generation methods and
(b) first order methods for non-smooth convex optimization -- techniques that
are rarely used together for solving large scale LPs. These components have
their respective strengths; and while they are found to be useful as separate
entities, they have not been used together in the context of solving large
scale LPs such as the ones studied herein. Our approach complements the
strengths of (a) and (b) -- leading to a scheme that seems to significantly
outperform commercial solvers as well as specialized implementations for these
problems. We present numerical results on a series of real and synthetic
datasets demonstrating the surprising effectiveness of classic
column/constraint generation methods in the context of challenging LP-based
machine learning tasks.

    

### [[1902.03964] Deep Node Ranking for Neuro-symbolic Structural Node Embedding and Classification](http://arxiv.org/abs/1902.03964)


  Network node embedding is an active research subfield of complex network
analysis. This paper contributes a novel approach to learning network node
embeddings and direct node classification using a node ranking scheme coupled
with an autoencoder-based neural network architecture. The main advantages of
the proposed Deep Node Ranking (DNR) algorithm are competitive or better
classification performance, significantly higher learning speed and lower space
requirements when compared to state-of-the-art approaches on 15 real-life node
classification benchmarks. Furthermore, it enables exploration of the
relationship between symbolic and the derived sub-symbolic node
representations, offering insights into the learned node space structure. To
avoid the space complexity bottleneck in a direct node classification setting,
DNR computes stationary distributions of personalized random walks from given
nodes in mini-batches, scaling seamlessly to larger networks. The scaling laws
associated with DNR were also investigated on 1488 synthetic Erdős-Rényi
networks, demonstrating its scalability to tens of millions of links.

    

### [[1905.10617] Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation?](http://arxiv.org/abs/1905.10617)


  Exposure bias has been regarded as a central problem for auto-regressive
language models (LM). It claims that teacher forcing would cause the test-time
generation to be incrementally distorted due to the training-generation
discrepancy. Although a lot of algorithms have been proposed to avoid teacher
forcing and therefore alleviate exposure bias, there is little work showing how
serious the exposure bias problem actually is. In this work, we focus on the
task of open-ended language generation, propose metrics to quantify the impact
of exposure bias in the aspects of quality, diversity, and consistency. Our key
intuition is that if we feed ground-truth data prefixes (instead of prefixes
generated by the model itself) into the model and ask it to continue the
generation, the performance should become much better because the
training-generation discrepancy in the prefix is removed. Both automatic and
human evaluations are conducted in our experiments. On the contrary to the
popular belief in exposure bias, we find that the the distortion induced by the
prefix discrepancy is limited, and does not seem to be incremental during the
generation. Moreover, our analysis reveals an interesting self-recovery ability
of the LM, which we hypothesize to be countering the harmful effects from
exposure bias.

    

### [[2003.03813] Keeping it simple: Implementation and performance of the proto-principle of adaptation and learning in the language sciences](http://arxiv.org/abs/2003.03813)


  In this paper we present the Widrow-Hoff rule and its applications to
language data. After contextualizing the rule historically and placing it in
the chain of neurally inspired artificial learning models, we explain its
rationale and implementational considerations. Using a number of case studies
we illustrate how the Widrow-Hoff rule offers unexpected opportunities for the
computational simulation of a range of language phenomena that make it possible
to approach old problems from a novel perspective.

    

### [[2004.09281] Tractable Approximate Gaussian Inference for Bayesian Neural Networks](http://arxiv.org/abs/2004.09281)


  In this paper, we propose an analytical method for performing tractable
approximate Gaussian inference (TAGI) in Bayesian neural networks. The method
enables the analytical Gaussian inference of the posterior mean vector and
diagonal covariance matrix for weights and biases. The method proposed has a
computational complexity of $\mathcal{O}(n)$ with respect to the number of
parameters $n$, and the tests performed on regression and classification
benchmarks confirm that, for a same network architecture, it matches the
performance of existing methods relying on gradient backpropagation.

    

### [[2006.10713] Zero-Shot Learning with Common Sense Knowledge Graphs](http://arxiv.org/abs/2006.10713)


  Zero-shot learning relies on semantic class representations such as
hand-engineered attributes or learned embeddings to predict classes without any
labeled examples. We propose to learn class representations by embedding nodes
from common sense knowledge graphs in a vector space. Common sense knowledge
graphs are an untapped source of explicit high-level knowledge that requires
little human effort to apply to a range of tasks. To capture the knowledge in
the graph, we introduce ZSL-KG, a general-purpose framework with a novel
transformer graph convolutional network (TrGCN) for generating class
representations. Our proposed TrGCN architecture computes non-linear
combinations of node neighbourhoods. Our results show that ZSL-KG improves over
existing WordNet-based methods on five out of six zero-shot benchmark datasets
in language and vision.

    

### [[2007.01524] Domain Adaptation without Source Data](http://arxiv.org/abs/2007.01524)


  Domain adaptation assumes that samples from source and target domains are
freely accessible during a training phase. However, such an assumption is
rarely plausible in the real-world and possibly causes data-privacy issues,
especially when the label of the source domain can be a sensitive attribute as
an identifier. To avoid accessing source data that may contain sensitive
information, we introduce Source data-Free Domain Adaptation (SFDA). Our key
idea is to leverage a pre-trained model from the source domain and
progressively update the target model in a self-learning manner. We observe
that target samples with lower self-entropy measured by the pre-trained source
model are more likely to be classified correctly. From this, we select the
reliable samples with the self-entropy criterion and define these as class
prototypes. We then assign pseudo labels for every target sample based on the
similarity score with class prototypes. Furthermore, to reduce the uncertainty
from the pseudo labeling process, we propose set-to-set distance-based
filtering which does not require any tunable hyperparameters. Finally, we train
the target model with the filtered pseudo labels with regularization from the
pre-trained source model. Surprisingly, without direct usage of labeled source
samples, our PrDA outperforms conventional domain adaptation methods on
benchmark datasets. Our code is publicly available at
this https URL


### [[2007.06968] Deep composition of tensor-trains using squared inverse Rosenblatt transports](http://arxiv.org/abs/2007.06968)


  Characterising intractable high-dimensional random variables is one of the
fundamental challenges in stochastic computation. The recent surge of transport
maps offers a mathematical foundation and new insights for tackling this
challenge by coupling intractable random variables with tractable reference
random variables. This paper generalises the functional tensor-train
approximation of the inverse Rosenblatt transport recently developed by Dolgov
et al. (Stat Comput 30:603--625, 2020) to a wide class of high-dimensional
non-negative functions, such as unnormalised probability density functions.
First, we extend the inverse Rosenblatt transform to enable the transport to
general reference measures other than the uniform measure. We develop an
efficient procedure to compute this transport from a squared tensor-train
decomposition which preserves the monotonicity. More crucially, we integrate
the proposed order-preserving functional tensor-train transport into a nested
variable transformation framework inspired by the layered structure of deep
neural networks. The resulting deep inverse Rosenblatt transport significantly
expands the capability of tensor approximations and transport maps to random
variables with complicated nonlinear interactions and concentrated density
functions. We demonstrate the efficiency of the proposed approach on a range of
applications in statistical learning and uncertainty quantification, including
parameter estimation for dynamical systems and inverse problems constrained by
partial differential equations.

    

### [[2007.08128] Detecting Out-of-distribution Samples via Variational Auto-encoder with Reliable Uncertainty Estimation](http://arxiv.org/abs/2007.08128)


  Variational autoencoders (VAEs) are influential generative models with rich
representation capabilities from the deep neural network architecture and
Bayesian method. However, VAE models have a weakness that assign a higher
likelihood to out-of-distribution (OOD) inputs than in-distribution (ID)
inputs. To address this problem, a reliable uncertainty estimation is
considered to be critical for in-depth understanding of OOD inputs. In this
study, we propose an improved noise contrastive prior (INCP) to be able to
integrate into the encoder of VAEs, called INCPVAE. INCP is scalable, trainable
and compatible with VAEs, and it also adopts the merits from the INCP for
uncertainty estimation. Experiments on various datasets demonstrate that
compared to the standard VAEs, our model is superior in uncertainty estimation
for the OOD data and is robust in anomaly detection tasks. The INCPVAE model
obtains reliable uncertainty estimation for OOD inputs and solves the OOD
problem in VAE models.

    

### [[2009.00298] Universal Approximation Property of Quantum Machine Learning Models in Quantum-Enhanced Feature Spaces](http://arxiv.org/abs/2009.00298)


  Encoding classical data into quantum states is considered a quantum feature
map to map classical data into a quantum Hilbert space. This feature map
provides opportunities to incorporate quantum advantages into machine learning
algorithms to be performed on near-term intermediate-scale quantum computers.
The crucial idea is using the quantum Hilbert space as a quantum-enhanced
feature space in machine learning models. While the quantum feature map has
demonstrated its capability when combined with linear classification models in
some specific applications, its expressive power from the theoretical
perspective remains unknown. We prove that the machine learning models induced
from the quantum-enhanced feature space are universal approximators of
continuous functions under typical quantum feature maps. We also study the
capability of quantum feature maps in the classification of disjoint regions.
Our work enables an important theoretical analysis to ensure that machine
learning algorithms based on quantum feature maps can handle a broad class of
machine learning tasks. In light of this, one can design a quantum machine
learning model with more powerful expressivity.

    

### [[2009.05160] Rank over Class: The Untapped Potential of Ranking in Natural Language Processing](http://arxiv.org/abs/2009.05160)


  Text classification has long been a staple within Natural Language Processing
(NLP) with applications spanning across diverse areas such as sentiment
analysis, recommender systems and spam detection. With such a powerful
solution, it is often tempting to use it as the go-to tool for all NLP problems
since when you are holding a hammer, everything looks like a nail. However, we
argue here that many tasks which are currently addressed using classification
are in fact being shoehorned into a classification mould and that if we instead
address them as a ranking problem, we not only improve the model, but we
achieve better performance. We propose a novel end- to-end ranking approach
consisting of a Transformer network responsible for producing representations
for a pair of text sequences, which are in turn passed into a context
aggregating network outputting ranking scores used to determine an ordering to
the sequences based on some notion of relevance. We perform numerous
experiments on publicly-available datasets and investigate the applications of
ranking in problems often solved using classification. In an experiment on a
heavily-skewed sentiment analysis dataset, converting ranking results to
classification labels yields an approximately 22% improvement over
state-of-the-art text classification, demonstrating the efficacy of text
ranking over text classification in certain scenarios.

    

### [[2009.08161] Byzantine-Robust Variance-Reduced Federated Learning over Distributed Non-i.i.d. Data](http://arxiv.org/abs/2009.08161)


  We consider the federated learning problem where data on workers are not
independent and identically distributed (i.i.d.). During the learning process,
an unknown number of Byzantine workers may send malicious messages to the
central node, leading to remarkable learning error. Most of the
Byzantine-robust methods address this issue by using robust aggregation rules
to aggregate the received messages, but rely on the assumption that all the
regular workers have i.i.d. data, which is not the case in many federated
learning applications. In light of the significance of reducing stochastic
gradient noise for mitigating the effect of Byzantine attacks, we use a
resampling strategy to reduce the impact of both inner variation (that
describes the sample heterogeneity on every regular worker) and outer variation
(that describes the sample heterogeneity among the regular workers), along with
a stochastic average gradient algorithm to gradually eliminate the inner
variation. The variance-reduced messages are then aggregated with a robust
geometric median operator. We prove that the proposed method reaches a
neighborhood of the optimal solution at a linear convergence rate and the
learning error is determined by the number of Byzantine workers. Numerical
experiments corroborate the theoretical results and show that the proposed
method outperforms the state-of-the-arts in the non-i.i.d. setting.

    

### [[2009.12981] Parametric UMAP embeddings for representation and semi-supervised learning](http://arxiv.org/abs/2009.12981)


  UMAP is a non-parametric graph-based dimensionality reduction algorithm using
applied Riemannian geometry and algebraic topology to find low-dimensional
embeddings of structured data. The UMAP algorithm consists of two steps: (1)
Compute a graphical representation of a dataset (fuzzy simplicial complex), and
(2) Through stochastic gradient descent, optimize a low-dimensional embedding
of the graph. Here, we extend the second step of UMAP to a parametric
optimization over neural network weights, learning a parametric relationship
between data and embedding. We first demonstrate that Parametric UMAP performs
comparably to its non-parametric counterpart while conferring the benefit of a
learned parametric mapping (e.g. fast online embeddings for new data). We then
explore UMAP as a regularization, constraining the latent distribution of
autoencoders, parametrically varying global structure preservation, and
improving classifier accuracy for semi-supervised learning by capturing
structure in unlabeled data. Google Colab walkthrough:
this https URL


### [[2010.06993] Weight Squeezing: Reparameterization for Knowledge Transfer and Model Compression](http://arxiv.org/abs/2010.06993)


  In this work, we present a novel approach for simultaneous knowledge transfer
and model compression called Weight Squeezing. With this method, we perform
knowledge transfer from a teacher model by learning the mapping from its
weights to smaller student model weights.
We applied Weight Squeezing to a pre-trained text classification model based
on BERT-Medium model and compared our method to various other knowledge
transfer and model compression methods on GLUE multitask benchmark. We observed
that our approach produces better results while being significantly faster than
other methods for training student models.
We also proposed a variant of Weight Squeezing called Gated Weight Squeezing,
for which we combined fine-tuning of BERT-Medium model and learning mapping
from BERT-Base weights. We showed that fine-tuning with Gated Weight Squeezing
outperforms plain fine-tuning of BERT-Medium model as well as other concurrent
SoTA approaches while much being easier to implement.

    

### [[2010.10016] Action Sequence Augmentation for Early Graph-based Anomaly Detection](http://arxiv.org/abs/2010.10016)


  The proliferation of web platforms has created incentives for online abuse.
Many graph-based anomaly detection techniques are proposed to identify the
suspicious accounts and behaviors. However, most of them detect the anomalies
once the users have performed many such behaviors. Their performance is
substantially hindered when the users' observed data is limited at an early
stage, which needs to be improved to minimize financial loss. In this work, we
propose Eland, a novel framework that uses action sequence augmentation for
early anomaly detection. Eland utilizes a sequence predictor to predict next
actions of every user and exploits the mutual enhancement between action
sequence augmentation and user-action graph anomaly detection. Experiments on
three real-world datasets show that Eland improves the performance of a variety
of graph-based anomaly detection methods. With Eland, anomaly detection
performance at an earlier stage is better than non-augmented methods that need
significantly more observed data by up to 15% on the Area under the ROC curve.

    

### [[2011.00288] Optimal Sample Complexity of Subgradient Descent for Amplitude Flow via Non-Lipschitz Matrix Concentration](http://arxiv.org/abs/2011.00288)


  We consider the problem of recovering a real-valued $n$-dimensional signal
from $m$ phaseless, linear measurements and analyze the amplitude-based
non-smooth least squares objective. We establish local convergence of
subgradient descent with optimal sample complexity based on the uniform
concentration of a random, discontinuous matrix-valued operator arising from
the objective's gradient dynamics. While common techniques to establish uniform
concentration of random functions exploit Lipschitz continuity, we prove that
the discontinuous matrix-valued operator satisfies a uniform matrix
concentration inequality when the measurement vectors are Gaussian as soon as
$m = \Omega(n)$ with high probability. We then show that satisfaction of this
inequality is sufficient for subgradient descent with proper initialization to
converge linearly to the true solution up to the global sign ambiguity. As a
consequence, this guarantees local convergence for Gaussian measurements at
optimal sample complexity. The concentration methods in the present work have
previously been used to establish recovery guarantees for a variety of inverse
problems under generative neural network priors. This paper demonstrates the
applicability of these techniques to more traditional inverse problems and
serves as a pedagogical introduction to those results.

    

### [[2011.02852] Neural networks for classification of strokes in electrical impedance tomography on a 3D head model](http://arxiv.org/abs/2011.02852)


  We consider the problem of the detection of brain hemorrhages from three
dimensional (3D) electrical impedance tomography (EIT) measurements. This is a
condition requiring urgent treatment for which EIT might provide a portable and
quick diagnosis. We employ two neural network architectures -- a fully
connected and a convolutional one -- for the classification of hemorrhagic and
ischemic strokes. The networks are trained on a dataset with $40\,000$ samples
of synthetic electrode measurements generated with the complete electrode model
on realistic heads with a 3-layer structure. We consider changes in head
anatomy and layers, electrode position, measurement noise and conductivity
values. We then test the networks on several datasets of unseen EIT data, with
more complex stroke modeling (different shapes and volumes), higher levels of
noise and different amounts of electrode misplacement. On most test datasets we
achieve $\geq 90\%$ average accuracy with fully connected neural networks,
while the convolutional ones display an average accuracy $\geq 80\%$. Despite
the use of simple neural network architectures, the results obtained are very
promising and motivate the applications of EIT-based classification methods on
real phantoms and ultimately on human patients.

    

### [[2011.07466] Continuous Conditional Generative Adversarial Networks: Novel Empirical Losses and Label Input Mechanisms](http://arxiv.org/abs/2011.07466)


  This work proposes the continuous conditional generative adversarial network
(CcGAN), the first generative model for image generation conditional on
continuous, scalar conditions (termed regression labels). Existing conditional
GANs (cGANs) are mainly designed for categorical conditions (eg, class labels);
conditioning on regression labels is mathematically distinct and raises two
fundamental problems:(P1) Since there may be very few (even zero) real images
for some regression labels, minimizing existing empirical versions of cGAN
losses (aka empirical cGAN losses) often fails in practice;(P2) Since
regression labels are scalar and infinitely many, conventional label input
methods are not applicable. The proposed CcGAN solves the above problems,
respectively, by (S1) reformulating existing empirical cGAN losses to be
appropriate for the continuous scenario; and (S2) proposing a naive label input
(NLI) method and an improved label input (ILI) method to incorporate regression
labels into the generator and the discriminator. The reformulation in (S1)
leads to two novel empirical discriminator losses, termed the hard vicinal
discriminator loss (HVDL) and the soft vicinal discriminator loss (SVDL)
respectively, and a novel empirical generator loss. The error bounds of a
discriminator trained with HVDL and SVDL are derived under mild assumptions in
this work. Two new benchmark datasets (RC-49 and Cell-200) and a novel
evaluation metric (Sliding Fréchet Inception Distance) are also proposed for
this continuous scenario. Our experiments on the Circular 2-D Gaussians, RC-49,
UTKFace, Cell-200, and Steering Angle datasets show that CcGAN is able to
generate diverse, high-quality samples from the image distribution conditional
on a given regression label. Moreover, in these experiments, CcGAN
substantially outperforms cGAN both visually and quantitatively.

    

### [[2012.01338] Siamese Basis Function Network for Data Efficient Defect Classification in Technical Domains](http://arxiv.org/abs/2012.01338)


  Training Deep Learning Models in technical domains often brings the
challenges that although the task is clear, insufficient data for training is
available. In this work we propose a novel approach based on the combination of
Siamese-Networks and Radial-Basis- Function-Networks to perform data-efficient
classification without pre-Training by measuring the distance between images in
semantic space in a data efficient manner. We develop the models using three
technical datasets, the NEU dataset the BSD dataset as well as the TEX dataset.
Additional to the technical domain show the general applicability to classical
datasets (cifar10 and MNIST) as well. The approach is tested against state of
the art models (Resnet50 and Resnet101) by stepwise reducing the number of
samples available for training. The authors show that the proposed approach
outperforms the state of the art models in the low data regime.

    

### [[2012.03063] FairOD: Fairness-aware Outlier Detection](http://arxiv.org/abs/2012.03063)


  Fairness and Outlier Detection (OD) are closely related, as it is exactly the
goal of OD to spot rare, minority samples in a given population. However, when
being a minority (as defined by protected variables, such as
race/ethnicity/sex/age) does not reflect positive-class membership (such as
criminal/fraud), OD produces unjust outcomes. Surprisingly, fairness-aware OD
has been almost untouched in prior work, as fair machine learning literature
mainly focuses on supervised settings. Our work aims to bridge this gap.
Specifically, we develop desiderata capturing well-motivated fairness criteria
for OD, and systematically formalize the fair OD problem. Further, guided by
our desiderata, we propose FairOD, a fairness-aware outlier detector that has
the following desirable properties: FairOD (1) exhibits treatment parity at
test time, (2) aims to flag equal proportions of samples from all groups (i.e.
obtain group fairness, via statistical parity), and (3) strives to flag truly
high-risk samples within each group. Extensive experiments on a diverse set of
synthetic and real world datasets show that FairOD produces outcomes that are
fair with respect to protected variables, while performing comparable to (and
in some cases, even better than) fairness-agnostic detectors in terms of
detection performance.

    

### [[2012.03236] Cross-Layer Distillation with Semantic Calibration](http://arxiv.org/abs/2012.03236)


  Knowledge distillation is a technique to enhance the generalization ability
of a student model by exploiting outputs from a teacher model. Recently,
feature-map based variants explore knowledge transfer between manually assigned
teacher-student pairs in intermediate layers for further improvement. However,
layer semantics may vary in different neural networks and semantic mismatch in
manual layer associations will lead to performance degeneration due to negative
regularization. To address this issue, we propose Semantic Calibration for
cross-layer Knowledge Distillation (SemCKD), which automatically assigns proper
target layers of the teacher model for each student layer with an attention
mechanism. With a learned attention distribution, each student layer distills
knowledge contained in multiple teacher layers rather than a specific
intermediate layer for appropriate cross-layer supervision. We further provide
theoretical analysis of the association weights and conduct extensive
experiments to demonstrate the effectiveness of our approach. Code is avaliable
at \url{this https URL}.

    

### [[2012.05661] Effect of the regularization hyperparameter on deep learning-based segmentation in LGE-MRI](http://arxiv.org/abs/2012.05661)


  The extent to which the arbitrarily selected L2 regularization hyperparameter
value affects the outcome of semantic segmentation with deep learning is
demonstrated. Demonstrations rely on training U-net on small LGE-MRI datasets
using the arbitrarily selected L2 regularization values. The remaining
hyperparameters are to be manually adjusted or tuned only when 10 % of all
epochs are reached before the training validation accuracy reaches 90%.
Semantic segmentation with deep learning outcomes are objectively and
subjectively evaluated against the manual ground truth segmentation.

    

### [[2012.06757] Query-free Black-box Adversarial Attacks on Graphs](http://arxiv.org/abs/2012.06757)


  Adversarial attacks on graphs have attracted considerable research interests.
Existing works assume the attacker is either (partly) aware of the victim
model, or able to send queries to it. These assumptions are, however,
unrealistic. To bridge the gap between theoretical graph attacks and real-world
scenarios, in this work, we propose a novel and more realistic setting: strict
black-box graph attack, in which the attacker has no knowledge about the victim
model at all and is not allowed to send any queries. To design such an attack
strategy, we first propose a generic graph filter to unify different families
of graph-based models. The strength of attacks can then be quantified by the
change in the graph filter before and after attack. By maximizing this change,
we are able to find an effective attack strategy, regardless of the underlying
model. To solve this optimization problem, we also propose a relaxation
technique and approximation theories to reduce the difficulty as well as the
computational expense. Experiments demonstrate that, even with no exposure to
the model, the Macro-F1 drops 6.4% in node classification and 29.5% in graph
classification, which is a significant result compared with existent works.

    

### [[2012.09070] Evaluation of deep learning-based myocardial infarction quantification using Segment CMR software](http://arxiv.org/abs/2012.09070)


  This work evaluates deep learning-based myocardial infarction (MI)
quantification using Segment cardiovascular magnetic resonance (CMR) software.
Segment CMR software incorporates the expectation-maximization, weighted
intensity, a priori information (EWA) algorithm used to generate the infarct
scar volume, infarct scar percentage, and microvascular obstruction percentage.
Also, Segment CMR software segmentation algorithm is updated with accurate
semantic segmentation with U-net for fully automated or deep learning-based MI
quantification. The direct observation of graphs and the number of infarcted
and contoured myocardium are two options used to estimate the relationship
between deep learning-based MI quantification and medical expert-based results.

    

### [[2012.13760] Variance Reduction on General Adaptive Stochastic Mirror Descent](http://arxiv.org/abs/2012.13760)


  In this work, we investigate the idea of variance reduction by studying its
properties with general adaptive mirror descent algorithms in nonsmooth
nonconvex finite-sum optimization problems. We propose a simple yet generalized
framework for variance reduced adaptive mirror descent algorithms named SVRAMD
and provide its convergence analysis in both the nonsmooth nonconvex problem
and the P-L conditioned problem. We prove that variance reduction reduces the
SFO complexity of adaptive mirror descent algorithms and thus accelerates their
convergence. In particular, our general theory implies that variance reduction
can be applied to algorithms using time-varying step sizes and self-adaptive
algorithms such as AdaGrad and RMSProp. Moreover, the convergence rates of
SVRAMD recover the best existing rates of non-adaptive variance reduced mirror
descent algorithms without complicated algorithmic components. Extensive
experiments in deep learning validate our theoretical findings.

    

### [[2101.02420] Towards Optimally Efficient Search with Deep Learning for Large-Scale MIMO Systems](http://arxiv.org/abs/2101.02420)


  This paper investigates the optimal signal detection problem with a
particular interest in large-scale multiple-input multiple-output (MIMO)
systems. The problem is NP-hard and can be solved optimally by searching the
shortest path on the decision tree. Unfortunately, the existing optimal search
algorithms often involve prohibitively high complexities, which indicates that
they are infeasible in large-scale MIMO systems. To address this issue, we
propose a general heuristic search algorithm, namely, hyperaccelerated tree
search (HATS) algorithm. The proposed algorithm employs a deep neural network
(DNN) to estimate the optimal heuristic, and then use the estimated heuristic
to speed up the underlying memory-bounded search algorithm. This idea is
inspired by the fact that the underlying heuristic search algorithm reaches the
optimal efficiency with the optimal heuristic function. Simulation results show
that the proposed algorithm reaches almost the optimal bit error rate (BER)
performance in large-scale systems, while the memory size can be bounded. In
the meanwhile, it visits nearly the fewest tree nodes. This indicates that the
proposed algorithm reaches almost the optimal efficiency in practical
scenarios, and thereby it is applicable for large-scale systems. Besides, the
code for this paper is available at this https URL.

    

### [[2101.07235] Reducing bias and increasing utility by federated generative modeling of medical images using a centralized adversary](http://arxiv.org/abs/2101.07235)


  We introduce FELICIA (FEderated LearnIng with a CentralIzed Adversary) a
generative mechanism enabling collaborative learning. In particular, we show
how a data owner with limited and biased data could benefit from other data
owners while keeping data from all the sources private. This is a common
scenario in medical image analysis where privacy legislation prevents data from
being shared outside local premises. FELICIA works for a large family of
Generative Adversarial Networks (GAN) architectures including vanilla and
conditional GANs as demonstrated in this work. We show that by using the
FELICIA mechanism, a data owner with limited image samples can generate
high-quality synthetic images with high utility while neither data owners has
to provide access to its data. The sharing happens solely through a central
discriminator that has access limited to synthetic data. Here, utility is
defined as classification performance on a real test set. We demonstrate these
benefits on several realistic healthcare scenarions using benchmark image
datasets (MNIST, CIFAR-10) as well as on medical images for the task of skin
lesion classification. With multiple experiments, we show that even in the
worst cases, combining FELICIA with real data gracefully achieves performance
on par with real data while most results significantly improves the utility.

    

### [[2102.01307] Human-Machine Collaborative Video Coding Through Cuboidal Partitioning](http://arxiv.org/abs/2102.01307)


  Video coding algorithms encode and decode an entire video frame while feature
coding techniques only preserve and communicate the most critical information
needed for a given application. This is because video coding targets human
perception, while feature coding aims for machine vision tasks. Recently,
attempts are being made to bridge the gap between these two domains. In this
work, we propose a video coding framework by leveraging on to the commonality
that exists between human vision and machine vision applications using cuboids.
This is because cuboids, estimated rectangular regions over a video frame, are
computationally efficient, has a compact representation and object centric.
Such properties are already shown to add value to traditional video coding
systems. Herein cuboidal feature descriptors are extracted from the current
frame and then employed for accomplishing a machine vision task in the form of
object detection. Experimental results show that a trained classifier yields
superior average precision when equipped with cuboidal features oriented
representation of the current test frame. Additionally, this representation
costs $7\%$ less in bit rate if the captured frames are need be communicated to
a receiver.

    

### [[2102.02390] A Universal Framework for Featurization of Atomistic Systems](http://arxiv.org/abs/2102.02390)


  Molecular dynamics simulations are an invaluable tool in numerous scientific
fields. However, the ubiquitous classical force fields cannot describe reactive
systems, and quantum molecular dynamics are too computationally demanding to
treat large systems or long timescales. Reactive force fields based on physics
or machine learning can be used to bridge the gap in time and length scales,
but these force fields require substantial effort to construct and are highly
specific to a given chemical composition and application. A significant
limitation of machine learning models is the use of element-specific features,
leading to models that scale poorly with the number of elements. This work
introduces the Gaussian multipole (GMP) featurization scheme that utilizes
physically-relevant multipole expansions of the electron density around atoms
to yield feature vectors that interpolate between element types and have a
fixed dimension regardless of the number of elements present. We combine GMP
with neural networks to directly compare it to the widely used Behler-Parinello
symmetry functions for the MD17 dataset, revealing that it exhibits improved
accuracy and computational efficiency. Further, we demonstrate that GMP-based
models can achieve chemical accuracy for the QM9 dataset, and their accuracy
remains reasonable even when extrapolating to new elements. Finally, we test
GMP-based models for the Open Catalysis Project (OCP) dataset, revealing
comparable performance to graph convolutional deep learning models. The results
indicate that this featurization scheme fills a critical gap in the
construction of efficient and transferable machine-learned force fields.

    

### [[2102.04822] Learning How to Search: Generating Effective Test Cases Through Adaptive Fitness Function Selection](http://arxiv.org/abs/2102.04822)


  Search-based test generation is guided by feedback from one or more fitness
functions - scoring functions that judge solution optimality. Choosing
informative fitness functions is crucial to meeting the goals of a tester.
Unfortunately, many goals - such as forcing the class-under-test to throw
exceptions, increasing test suite diversity, and attaining Strong Mutation
Coverage - do not have effective fitness function formulations. We propose that
meeting such goals requires treating fitness function identification as a
secondary optimization step. An adaptive algorithm that can vary the selection
of fitness functions could adjust its selection throughout the generation
process to maximize goal attainment, based on the current population of test
suites. To test this hypothesis, we have implemented two reinforcement learning
algorithms in the EvoSuite unit test generation framework, and used these
algorithms to dynamically set the fitness functions used during generation for
the three goals identified above.
We have evaluated our framework, EvoSuiteFIT, on a set of Java case examples.
EvoSuiteFIT techniques attain significant improvements for two of the three
goals, and show limited improvements on the third when the number of
generations of evolution is fixed. Additionally, for two of the three goals,
EvoSuiteFIT detects faults missed by the other techniques. The ability to
adjust fitness functions allows strategic choices that efficiently produce more
effective test suites, and examining these choices offers insight into how to
attain our testing goals. We find that adaptive fitness function selection is a
powerful technique to apply when an effective fitness function does not already
exist for achieving a testing goal.

    

### [[2102.06622] MetaGrad: Adaptation using Multiple Learning Rates in Online Learning](http://arxiv.org/abs/2102.06622)


  We provide a new adaptive method for online convex optimization, MetaGrad,
that is robust to general convex losses but achieves faster rates for a broad
class of special functions, including exp-concave and strongly convex
functions, but also various types of stochastic and non-stochastic functions
without any curvature. We prove this by drawing a connection to the Bernstein
condition, which is known to imply fast rates in offline statistical learning.
MetaGrad further adapts automatically to the size of the gradients. Its main
feature is that it simultaneously considers multiple learning rates, which are
weighted directly proportional to their empirical performance on the data using
a new meta-algorithm. We provide three versions of MetaGrad. The full matrix
version maintains a full covariance matrix and is applicable to learning tasks
for which we can afford update time quadratic in the dimension. The other two
versions provide speed-ups for high-dimensional learning tasks with an update
time that is linear in the dimension: one is based on sketching, the other on
running a separate copy of the basic algorithm per coordinate. We evaluate all
versions of MetaGrad on benchmark online classification and regression tasks,
on which they consistently outperform both online gradient descent and AdaGrad.

    

### [[2102.09548] Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development](http://arxiv.org/abs/2102.09548)


  Therapeutics machine learning is an emerging field with incredible
opportunities for innovatiaon and impact. However, advancement in this field
requires formulation of meaningful learning tasks and careful curation of
datasets. Here, we introduce Therapeutics Data Commons (TDC), the first
unifying platform to systematically access and evaluate machine learning across
the entire range of therapeutics. To date, TDC includes 66 AI-ready datasets
spread across 22 learning tasks and spanning the discovery and development of
safe and effective medicines. TDC also provides an ecosystem of tools and
community resources, including 33 data functions and types of meaningful data
splits, 23 strategies for systematic model evaluation, 17 molecule generation
oracles, and 29 public leaderboards. All resources are integrated and
accessible via an open Python library. We carry out extensive experiments on
selected datasets, demonstrating that even the strongest algorithms fall short
of solving key therapeutics challenges, including real dataset distributional
shifts, multi-scale modeling of heterogeneous data, and robust generalization
to novel data points. We envision that TDC can facilitate algorithmic and
scientific advances and considerably accelerate machine-learning model
development, validation and transition into biomedical and clinical
implementation. TDC is an open-science initiative available at
this https URL.

    

### [[2102.10326] Automated identification of transiting exoplanet candidates in NASA Transiting Exoplanets Survey Satellite (TESS) data with machine learning methods](http://arxiv.org/abs/2102.10326)


  A novel artificial intelligence (AI) technique that uses machine learning
(ML) methodologies combines several algorithms, which were developed by
ThetaRay, Inc., is applied to NASA's Transiting Exoplanets Survey Satellite
(TESS) dataset to identify exoplanetary candidates. The AI/ML ThetaRay system
is trained initially with Kepler exoplanetary data and validated with confirmed
exoplanets before its application to TESS data. Existing and new features of
the data, based on various observational parameters, are constructed and used
in the AI/ML analysis by employing semi-supervised and unsupervised machine
learning techniques. By the application of ThetaRay system to 10,803 light
curves of threshold crossing events (TCEs) produced by the TESS mission,
obtained from the Mikulski Archive for Space Telescopes, the algorithm yields
about 50 targets for further analysis, and we uncover three new exoplanetary
candidates by further manual vetting. This study demonstrates for the first
time the successful application of the particular combined multiple AI/ML-based
methodologies to a large astrophysical dataset for rapid automated
classification of TCEs.

    

### [[2103.01006] GaNDLF: A Generally Nuanced Deep Learning Framework for Scalable End-to-End Clinical Workflows in Medical Imaging](http://arxiv.org/abs/2103.01006)


  Deep Learning (DL) has greatly highlighted the potential impact of optimized
machine learning in both the scientific and clinical communities. The advent of
open-source DL libraries from major industrial entities, such as TensorFlow
(Google), PyTorch (Facebook), and MXNet (Apache), further contributes to DL
promises on the democratization of computational analytics. However, increased
technical and specialized background is required to develop DL algorithms, and
the variability of implementation details hinders their reproducibility.
Towards lowering the barrier and making the mechanism of DL development,
training, and inference more stable, reproducible, and scalable, without
requiring an extensive technical background, this manuscript proposes the
Generally Nuanced Deep Learning Framework (GaNDLF). With built-in support for
$k$-fold cross-validation, data augmentation, multiple modalities and output
classes, and multi-GPU training, as well as the ability to work with both
radiographic and histologic imaging, GaNDLF aims to provide an end-to-end
solution for all DL-related tasks, to tackle problems in medical imaging and
provide a robust application framework for deployment in clinical workflows.

    

### [[2103.01422] Adaptive Transmission Scheduling in Wireless Networks for Asynchronous Federated Learning](http://arxiv.org/abs/2103.01422)


  In this paper, we study asynchronous federated learning (FL) in a wireless
distributed learning network (WDLN). To allow each edge device to use its local
data more efficiently via asynchronous FL, transmission scheduling in the WDLN
for asynchronous FL should be carefully determined considering system
uncertainties, such as time-varying channel and stochastic data arrivals, and
the scarce radio resources in the WDLN. To address this, we propose a metric,
called an effectivity score, which represents the amount of learning from
asynchronous FL. We then formulate an Asynchronous Learning-aware transmission
Scheduling (ALS) problem to maximize the effectivity score and develop three
ALS algorithms, called ALSA-PI, BALSA, and BALSA-PO, to solve it. If the
statistical information about the uncertainties is known, the problem can be
optimally and efficiently solved by ALSA-PI. Even if not, it can be still
optimally solved by BALSA that learns the uncertainties based on a Bayesian
approach using the state information reported from devices. BALSA-PO
suboptimally solves the problem, but it addresses a more restrictive WDLN in
practice, where the AP can observe a limited state information compared with
the information used in BALSA. We show via simulations that the models trained
by our ALS algorithms achieve performances close to that by an ideal benchmark
and outperform those by other state-of-the-art baseline scheduling algorithms
in terms of model accuracy, training loss, learning speed, and robustness of
learning. These results demonstrate that the adaptive scheduling strategy in
our ALS algorithms is effective to asynchronous FL.

    

### [[2103.09354] Digital Peter: Dataset, Competition and Handwriting Recognition Methods](http://arxiv.org/abs/2103.09354)


  This paper presents a new dataset of Peter the Great's manuscripts and
describes a segmentation procedure that converts initial images of documents
into the lines. The new dataset may be useful for researchers to train
handwriting text recognition models as a benchmark for comparing different
models. It consists of 9 694 images and text files corresponding to lines in
historical documents. The open machine learning competition Digital Peter was
held based on the considered dataset. The baseline solution for this
competition as well as more advanced methods on handwritten text recognition
are described in the article. Full dataset and all code are publicly available.

    

### [[2103.09402] In-air Knotting of Rope using Dual-Arm Robot based on Deep Learning](http://arxiv.org/abs/2103.09402)


  In this study, we report the successful execution of in-air knotting of rope
using a dual-arm two-finger robot based on deep learning. Owing to its
flexibility, the state of the rope was in constant flux during the operation of
the robot. This required the robot control system to dynamically correspond to
the state of the object at all times. However, a manual description of
appropriate robot motions corresponding to all object states is difficult to be
prepared in advance. To resolve this issue, we constructed a model that
instructed the robot to perform bowknots and overhand knots based on two deep
neural networks trained using the data gathered from its sensorimotor,
including visual and proximity sensors. The resultant model was verified to be
capable of predicting the appropriate robot motions based on the sensory
information available online. In addition, we designed certain task motions
based on the Ian knot method using the dual-arm two-fingers robot. The designed
knotting motions do not require a dedicated workbench or robot hand, thereby
enhancing the versatility of the proposed method. Finally, experiments were
performed to estimate the knotting performance of the real robot while
executing overhand knots and bowknots on rope and its success rate. The
experimental results established the effectiveness and high performance of the
proposed method.

    

### [[2103.09696] Generating Annotated Training Data for 6D Object Pose Estimation in Operational Environments with Minimal User Interaction](http://arxiv.org/abs/2103.09696)


  Recently developed deep neural networks achieved state-of-the-art results in
the subject of 6D object pose estimation for robot manipulation. However, those
supervised deep learning methods require expensive annotated training data.
Current methods for reducing those costs frequently use synthetic data from
simulations, but rely on expert knowledge and suffer from the "domain gap" when
shifting to the real world. Here, we present a proof of concept for a novel
approach of autonomously generating annotated training data for 6D object pose
estimation. This approach is designed for learning new objects in operational
environments while requiring little interaction and no expertise on the part of
the user. We evaluate our autonomous data generation approach in two grasping
experiments, where we archive a similar grasping success rate as related work
on a non autonomously generated data set.

    

### [[2103.10000] Human-Inspired Multi-Agent Navigation using Knowledge Distillation](http://arxiv.org/abs/2103.10000)


  Despite significant advancements in the field of multi-agent navigation,
agents still lack the sophistication and intelligence that humans exhibit in
multi-agent settings. In this paper, we propose a framework for learning a
human-like general collision avoidance policy for agent-agent interactions in
fully decentralized, multi-agent environments. Our approach uses knowledge
distillation with reinforcement learning to shape the reward function based on
expert policies extracted from human trajectory demonstrations through behavior
cloning. We show that agents trained with our approach can take human-like
trajectories in collision avoidance and goal-directed steering tasks not
provided by the demonstrations, outperforming the experts as well as
learning-based agents trained without knowledge distillation.

    

### [[2103.14539] FeatureEnVi: Visual Analytics for Feature Engineering Using Stepwise Selection and Semi-Automatic Extraction Approaches](http://arxiv.org/abs/2103.14539)


  The machine learning (ML) life cycle involves a series of iterative steps,
from the effective gathering and preparation of the data, including complex
feature engineering processes, to the presentation and improvement of results,
with various algorithms to choose from in every step. Feature engineering in
particular can be very beneficial for ML, leading to numerous improvements such
as boosting the predictive results, decreasing computational times, reducing
excessive noise, and increasing the transparency behind the decisions taken
during the training. Despite that, while several visual analytics tools exist
to monitor and control the different stages of the ML life cycle (especially
those related to data and algorithms), feature engineering support remains
inadequate. In this paper, we present FeatureEnVi, a visual analytics system
specifically designed to assist with the feature engineering process. Our
proposed system helps users to choose the most important feature, to transform
the original features into powerful alternatives, and to experiment with
different feature generation combinations. Additionally, data space slicing
allows users to explore the impact of features on both local and global scales.
FeatureEnVi utilizes multiple automatic feature selection techniques;
furthermore, it visually guides users with statistical evidence about the
influence of each feature (or subsets of features). The final outcome is the
extraction of heavily engineered features, evaluated by multiple validation
metrics. The usefulness and applicability of FeatureEnVi are demonstrated with
two use cases and a case study. We also report feedback from interviews with
two ML experts and a visualization researcher who assessed the effectiveness of
our system.

    

### [[2104.12840] Graph Neural Networks with Adaptive Frequency Response Filter](http://arxiv.org/abs/2104.12840)


  Graph Neural Networks have recently become a prevailing paradigm for various
high-impact graph learning tasks. Existing efforts can be mainly categorized as
spectral-based and spatial-based methods. The major challenge for the former is
to find an appropriate graph filter to distill discriminative information from
input signals for learning. Recently, attempts such as Graph Convolutional
Network (GCN) leverage Chebyshev polynomial truncation to seek an approximation
of graph filters and bridge these two families of methods. It has been shown in
recent studies that GCN and its variants are essentially employing fixed
low-pass filters to perform information denoising. Thus their learning
capability is rather limited and may over-smooth node representations at deeper
layers. To tackle these problems, we develop a novel graph neural network
framework AdaGNN with a well-designed adaptive frequency response filter. At
its core, AdaGNN leverages a simple but elegant trainable filter that spans
across multiple layers to capture the varying importance of different frequency
components for node representation learning. The inherent differences among
different feature channels are also well captured by the filter. As such, it
empowers AdaGNN with stronger expressiveness and naturally alleviates the
over-smoothing problem. We empirically validate the effectiveness of the
proposed framework on various benchmark datasets. Theoretical analysis is also
provided to show the superiority of the proposed AdaGNN. The implementation of
AdaGNN is available at \url{this https URL}.

    

### [[2105.04046] A likelihood approach to nonparametric estimation of a singular distribution using deep generative models](http://arxiv.org/abs/2105.04046)


  We investigate statistical properties of a likelihood approach to
nonparametric estimation of a singular distribution using deep generative
models. More specifically, a deep generative model is used to model
high-dimensional data that are assumed to concentrate around some
low-dimensional structure. Estimating the distribution supported on this
low-dimensional structure such as a low-dimensional manifold is challenging due
to its singularity with respect to the Lebesgue measure in the ambient space.
In the considered model, a usual likelihood approach can fail to estimate the
target distribution consistently due to the singularity. We prove that a novel
and effective solution exists by perturbing the data with an instance noise
which leads to consistent estimation of the underlying distribution with
desirable convergence rates. We also characterize the class of distributions
that can be efficiently estimated via deep generative models. This class is
sufficiently general to contain various structured distributions such as
product distributions, classically smooth distributions and distributions
supported on a low-dimensional manifold. Our analysis provides some insights on
how deep generative models can avoid the curse of dimensionality for
nonparametric distribution estimation. We conduct thorough simulation study and
real data analysis to empirically demonstrate that the proposed data
perturbation technique improves the estimation performance significantly.

    

### [[2105.07066] Node Selection Toward Faster Convergence for Federated Learning on Non-IID Data](http://arxiv.org/abs/2105.07066)


  Federated Learning (FL) is a distributed learning paradigm that enables a
large number of resource-limited nodes to collaboratively train a model without
data sharing. The non-independent-and-identically-distributed (non-i.i.d.) data
samples invoke discrepancy between global and local objectives, making the FL
model slow to converge. In this paper, we proposed Optimal Aggregation
algorithm for better aggregation, which finds out the optimal subset of local
updates of participating nodes in each global round, by identifying and
excluding the adverse local updates via checking the relationship between the
local gradient and the global gradient. Then, we proposed a Probabilistic Node
Selection framework (FedPNS) to dynamically change the probability for each
node to be selected based on the output of Optimal Aggregation. FedPNS can
preferentially select nodes that propel faster model convergence. The
unbiasedness of the proposed FedPNS design is illustrated and the convergence
rate improvement of FedPNS over the commonly adopted Federated Averaging
(FedAvg) algorithm is analyzed theoretically. Experimental results demonstrate
the effectiveness of FedPNS in accelerating the FL convergence rate, as
compared to FedAvg with random node selection.

    

### [[2105.08053] Algorithm-Agnostic Explainability for Unsupervised Clustering](http://arxiv.org/abs/2105.08053)


  Supervised machine learning explainability has developed rapidly in recent
years. However, clustering explainability has lagged behind. Here, we
demonstrate the first adaptation of model-agnostic explainability methods to
explain unsupervised clustering. We present two novel "algorithm-agnostic"
explainability methods - global permutation percent change (G2PC) and local
perturbation percent change (L2PC) - that identify feature importance globally
to a clustering algorithm and locally to the clustering of individual samples.
The methods are (1) easy to implement and (2) broadly applicable across
clustering algorithms, which could make them highly impactful. We demonstrate
the utility of the methods for explaining five popular clustering methods on
low-dimensional synthetic datasets and on high-dimensional functional network
connectivity data extracted from a resting-state functional magnetic resonance
imaging dataset of 151 individuals with schizophrenia and 160 controls. Our
results are consistent with existing literature while also shedding new light
on how changes in brain connectivity may lead to schizophrenia symptoms. We
further compare the explanations from our methods to an interpretable
classifier and find them to be highly similar. Our proposed methods robustly
explain multiple clustering algorithms and could facilitate new insights into
many applications. We hope this study will greatly accelerate the development
of the field of clustering explainability.

    

### [[2105.10066] A GAN-Like Approach for Physics-Based Imitation Learning and Interactive Character Control](http://arxiv.org/abs/2105.10066)


  We present a simple and intuitive approach for interactive control of
physically simulated characters. Our work builds upon generative adversarial
networks (GAN) and reinforcement learning, and introduces an imitation learning
framework where an ensemble of classifiers and an imitation policy are trained
in tandem given pre-processed reference clips. The classifiers are trained to
discriminate the reference motion from the motion generated by the imitation
policy, while the policy is rewarded for fooling the discriminators. Using our
GAN-based approach, multiple motor control policies can be trained separately
to imitate different behaviors. In runtime, our system can respond to external
control signal provided by the user and interactively switch between different
policies. Compared to existing methods, our proposed approach has the following
attractive properties: 1) achieves state-of-the-art imitation performance
without manually designing and fine tuning a reward function; 2) directly
controls the character without having to track any target reference pose
explicitly or implicitly through a phase state; and 3) supports interactive
policy switching without requiring any motion generation or motion matching
mechanism. We highlight the applicability of our approach in a range of
imitation and interactive control tasks, while also demonstrating its ability
to withstand external perturbations as well as to recover balance. Overall, our
approach generates high-fidelity motion, has low runtime cost, and can be
easily integrated into interactive applications and games.

    

### [[2105.10478] Spatial-Temporal Conv-sequence Learning with Accident Encoding for Traffic Flow Prediction](http://arxiv.org/abs/2105.10478)


  In an intelligent transportation system, the key problem of traffic
forecasting is how to extract the periodic temporal dependencies and complex
spatial correlation. Current state-of-the-art methods for traffic flow
forecasting are based on graph architectures and sequence learning models, but
they do not fully exploit spatial-temporal dynamic information in the traffic
system. Specifically, the temporal dependence of the short-range is diluted by
recurrent neural networks, and the existing sequence model ignores local
spatial information because the convolution operation uses global average
pooling. Besides, there will be some traffic accidents during the transitions
of objects causing congestion in the real world that trigger increased
prediction deviation. To overcome these challenges, we propose the
Spatial-Temporal Conv-sequence Learning (STCL), in which a focused temporal
block uses unidirectional convolution to effectively capture short-term
periodic temporal dependence, and a spatial-temporal fusion module is able to
extract the dependencies of both interactions and decrease the feature
dimensions. Moreover, the accidents features impact on local traffic
congestion, and position encoding is employed to detect anomalies in complex
traffic situations. We conduct a large number of experiments on real-world
tasks and verify the effectiveness of our proposed method.

    

### [[2105.14250] Cherry-Picking Gradients: Learning Low-Rank Embeddings of Visual Data via Differentiable Cross-Approximation](http://arxiv.org/abs/2105.14250)


  We propose an end-to-end trainable framework that processes large-scale
visual data tensors by looking at a fraction of their entries only. Our method
combines a neural network encoder with a tensor train decomposition to learn a
low-rank latent encoding, coupled with cross-approximation (CA) to learn the
representation through a subset of the original samples. CA is an adaptive
sampling algorithm that is native to tensor decompositions and avoids working
with the full high-resolution data explicitly. Instead, it actively selects
local representative samples that we fetch out-of-core and on-demand. The
required number of samples grows only logarithmically with the size of the
input. Our implicit representation of the tensor in the network enables
processing large grids that could not be otherwise tractable in their
uncompressed form. The proposed approach is particularly useful for large-scale
multidimensional grid data (e.g., 3D tomography), and for tasks that require
context over a large receptive field (e.g., predicting the medical condition of
entire organs). The code is available at this https URL.

    

### [[2106.01958] Multiplierless MP-Kernel Machine For Energy-efficient Edge Devices](http://arxiv.org/abs/2106.01958)


  We present a novel framework for designing multiplierless kernel machines
that can be used on resource-constrained platforms like intelligent edge
devices. The framework uses a piecewise linear (PWL) approximation based on a
margin propagation (MP) technique and uses only addition/subtraction, shift,
comparison, and register underflow/overflow operations. We propose a
hardware-friendly MP-based inference and online training algorithm that has
been optimized for a Field Programmable Gate Array (FPGA) platform. Our FPGA
implementation eliminates the need for DSP units and reduces the number of
LUTs. By reusing the same hardware for inference and training, we show that the
platform can overcome classification errors and local minima artifacts that
result from the MP approximation. Using the FPGA platform, we also show that
the proposed multiplierless MP-kernel machine demonstrates superior performance
in terms of power, performance, and area compared to other comparable
implementations.

    

### [[2106.03032] Deep Particulate Matter Forecasting Model Using Correntropy-Induced Loss](http://arxiv.org/abs/2106.03032)


  Forecasting the particulate matter (PM) concentration in South Korea has
become urgently necessary owing to its strong negative impact on human life. In
most statistical or machine learning methods, independent and identically
distributed data, for example, a Gaussian distribution, are assumed; however,
time series such as air pollution and weather data do not meet this assumption.
In this study, the maximum correntropy criterion for regression (MCCR) loss is
used in an analysis of the statistical characteristics of air pollution and
weather data. Rigorous seasonality adjustment of the air pollution and weather
data was performed because of their complex seasonality patterns and the
heavy-tailed distribution of data even after deseasonalization. The MCCR loss
was applied to multiple models including conventional statistical models and
state-of-the-art machine learning models. The results show that the MCCR loss
is more appropriate than the conventional mean squared error loss for
forecasting extreme values.

    

### [[2106.07297] Node Classification Meets Link Prediction on Knowledge Graphs](http://arxiv.org/abs/2106.07297)


  Node classification and link prediction are widely studied in graph
representation learning. While both transductive node classification and link
prediction operate over a single input graph, they have so far been studied
separately. Node classification models take an input graph with node features
and incomplete node labels, and implicitly assume that the graph is
relationally complete, i.e., no edges are missing. By contrast, link prediction
models are solely motivated by relational incompleteness of the input graphs,
and do not typically leverage node features or classes. We propose a unifying
perspective and study the problems of (i) transductive node classification over
incomplete graphs and (ii) link prediction over graphs with node features,
introduce a new dataset for this setting, WikiAlumni, and conduct an extensive
benchmarking study.

    

### [[2106.07302] Quantum diffusion map for nonlinear dimensionality reduction](http://arxiv.org/abs/2106.07302)


  Inspired by random walk on graphs, diffusion map (DM) is a class of
unsupervised machine learning that offers automatic identification of
low-dimensional data structure hidden in a high-dimensional dataset. In recent
years, among its many applications, DM has been successfully applied to
discover relevant order parameters in many-body systems, enabling automatic
classification of quantum phases of matter. However, classical DM algorithm is
computationally prohibitive for a large dataset, and any reduction of the time
complexity would be desirable. With a quantum computational speedup in mind, we
propose a quantum algorithm for DM, termed quantum diffusion map (qDM). Our qDM
takes as an input $N$ classical data vectors, performs an eigen-decomposition
of the Markov transition matrix in time $O(\log^3 N)$, and classically
constructs the diffusion map via the readout (tomography) of the eigenvectors,
giving a total expected runtime proportional to $N^2 \text{polylog}\, N$.
Lastly, quantum subroutines in qDM for constructing a Markov transition matrix,
and for analyzing its spectral properties can also be useful for other random
walk-based algorithms.

    

### [[2106.09692] Hi-Phy: A Benchmark for Hierarchical Physical Reasoning](http://arxiv.org/abs/2106.09692)


  Reasoning about the behaviour of physical objects is a key capability of
agents operating in physical worlds. Humans are very experienced in physical
reasoning while it remains a major challenge for AI. To facilitate research
addressing this problem, several benchmarks have been proposed recently.
However, these benchmarks do not enable us to measure an agent's granular
physical reasoning capabilities when solving a complex reasoning task. In this
paper, we propose a new benchmark for physical reasoning that allows us to test
individual physical reasoning capabilities. Inspired by how humans acquire
these capabilities, we propose a general hierarchy of physical reasoning
capabilities with increasing complexity. Our benchmark tests capabilities
according to this hierarchy through generated physical reasoning tasks in the
video game Angry Birds. This benchmark enables us to conduct a comprehensive
agent evaluation by measuring the agent's granular physical reasoning
capabilities. We conduct an evaluation with human players, learning agents, and
heuristic agents and determine their capabilities. Our evaluation shows that
learning agents, with good local generalization ability, still struggle to
learn the underlying physical reasoning capabilities and perform worse than
current state-of-the-art heuristic agents and humans. We believe that this
benchmark will encourage researchers to develop intelligent agents with
advanced, human-like physical reasoning capabilities. URL:
this https URL


### [[2106.15058] Improving Transferability of Adversarial Patches on Face Recognition with Generative Models](http://arxiv.org/abs/2106.15058)


  Face recognition is greatly improved by deep convolutional neural networks
(CNNs). Recently, these face recognition models have been used for identity
authentication in security sensitive applications. However, deep CNNs are
vulnerable to adversarial patches, which are physically realizable and
stealthy, raising new security concerns on the real-world applications of these
models. In this paper, we evaluate the robustness of face recognition models
using adversarial patches based on transferability, where the attacker has
limited accessibility to the target models. First, we extend the existing
transfer-based attack techniques to generate transferable adversarial patches.
However, we observe that the transferability is sensitive to initialization and
degrades when the perturbation magnitude is large, indicating the overfitting
to the substitute models. Second, we propose to regularize the adversarial
patches on the low dimensional data manifold. The manifold is represented by
generative models pre-trained on legitimate human face images. Using face-like
features as adversarial perturbations through optimization on the manifold, we
show that the gaps between the responses of substitute models and the target
models dramatically decrease, exhibiting a better transferability. Extensive
digital world experiments are conducted to demonstrate the superiority of the
proposed method in the black-box setting. We apply the proposed method in the
physical world as well.

    

### [[1206.5224] Stock prices assessment: proposal of a new index based on volume weighted historical prices through the use of computer modeling](http://arxiv.org/abs/1206.5224)


  The importance of considering the volumes to analyze stock prices movements
can be considered as a well-accepted practice in the financial area. However,
when we look at the scientific production in this field, we still cannot find a
unified model that includes volume and price variations for stock assessment
purposes. In this paper we present a computer model that could fulfill this
gap, proposing a new index to evaluate stock prices based on their historical
prices and volumes traded. Besides the model can be considered mathematically
very simple, it was able to improve significantly the performance of agents
operating with real financial data. Based on the results obtained, and also on
the very intuitive logic of our model, we believe that the index proposed here
can be very useful to help investors on the activity of determining ideal price
ranges for buying and selling stocks in the financial market.

    

### [[2101.06905] Blockchain Assisted Decentralized Federated Learning (BLADE-FL): Performance Analysis and Resource Allocation](http://arxiv.org/abs/2101.06905)


  Federated learning (FL), as a distributed machine learning paradigm, promotes
personal privacy by local data processing at each client. However, relying on a
centralized server for model aggregation, standard FL is vulnerable to server
malfunctions, untrustworthy server, and external attacks. To address this
issue, we propose a decentralized FL framework by integrating blockchain into
FL, namely, blockchain assisted decentralized federated learning (BLADE-FL). In
a round of the proposed BLADE-FL, each client broadcasts the trained model to
other clients, aggregates its own model with received ones, and then competes
to generate a block before its local training of the next round. We evaluate
the learning performance of BLADE-FL, and develop an upper bound on the global
loss function. Then we verify that this bound is convex with respect to the
number of overall aggregation rounds K, and optimize the computing resource
allocation for minimizing the upper bound. We also note that there is a
critical problem of training deficiency, caused by lazy clients who plagiarize
others' trained models and add artificial noises to disguise their cheating
behaviors. Focusing on this problem, we explore the impact of lazy clients on
the learning performance of BLADE-FL, and characterize the relationship among
the optimal K, the learning parameters, and the proportion of lazy clients.
Based on MNIST and Fashion-MNIST datasets, we show that the experimental
results are consistent with the analytical ones. To be specific, the gap
between the developed upper bound and experimental results is lower than 5%,
and the optimized K based on the upper bound can effectively minimize the loss
function.

    

### [[2106.07502] Training like Playing: A Reinforcement Learning And Knowledge Graph-based framework for building Automatic Consultation System in Medical Field](http://arxiv.org/abs/2106.07502)


  We introduce a framework for AI-based medical consultation system with
knowledge graph embedding and reinforcement learning components and its
implement. Our implement of this framework leverages knowledge organized as a
graph to have diagnosis according to evidence collected from patients
recurrently and dynamically. According to experiment we designed for evaluating
its performance, it archives a good result. More importantly, for getting
better performance, researchers can implement it on this framework based on
their innovative ideas, well designed experiments and even clinical trials.

    

### [[2108.12043] A Tutorial on Learning Disentangled Representations in the Imaging Domain](http://arxiv.org/abs/2108.12043)


  Disentangled representation learning has been proposed as an approach to
learning general representations. This can be done in the absence of, or with
limited, annotations. A good general representation can be readily fine-tuned
for new target tasks using modest amounts of data, or even be used directly in
unseen domains achieving remarkable performance in the corresponding task. This
alleviation of the data and annotation requirements offers tantalising
prospects for tractable and affordable applications in computer vision and
healthcare. Finally, disentangled representations can offer model
explainability and can help us understand the underlying causal relations of
the factors of variation, increasing their suitability for real-world
deployment. In this tutorial paper, we will offer an overview of the
disentangled representation learning, its building blocks and criteria, and
discuss applications in computer vision and medical imaging. We conclude our
tutorial by presenting the identified opportunities for the integration of
recent machine learning advances into disentanglement, as well as the remaining
challenges.

    

### [[2108.12444] A Design Flow for Mapping Spiking Neural Networks to Many-Core Neuromorphic Hardware](http://arxiv.org/abs/2108.12444)


  The design of many-core neuromorphic hardware is getting more and more
complex as these systems are expected to execute large machine learning models.
To deal with the design complexity, a predictable design flow is needed to
guarantee real-time performance such as latency and throughput without
significantly increasing the buffer requirement of computing cores. Synchronous
Data Flow Graphs (SDFGs) are used for predictable mapping of streaming
applications to multiprocessor systems. We propose an SDFG-based design flow
for mapping spiking neural networks (SNNs) to many-core neuromorphic hardware
with the objective of exploring the tradeoff between throughput and buffer
size. The proposed design flow integrates an iterative partitioning approach,
based on Kernighan-Lin graph partitioning heuristic, creating SNN clusters such
that each cluster can be mapped to a core of the hardware. The partitioning
approach minimizes the inter-cluster spike communication, which improves
latency on the shared interconnect of the hardware. Next, the design flow maps
clusters to cores using an instance of the Particle Swarm Optimization (PSO),
an evolutionary algorithm, exploring the design space of throughput and buffer
size. Pareto optimal mappings are retained from the design flow, allowing
system designers to select a Pareto mapping that satisfies throughput and
buffer size requirements of the design. We evaluated the design flow using five
large-scale convolutional neural network (CNN) models. Results demonstrate 63%
higher maximum throughput and 10% lower buffer size requirement compared to
state-of-the-art dataflow-based mapping solutions.

    

### [[2108.12897] ACTreS: Analog Clock Tree Synthesis](http://arxiv.org/abs/2108.12897)


  This paper describes a graph-theoretic formalism and a flow that, to a great
extent, automate the design of clock trees in Sampled-Data Analog Circuits
(SDACs). The current practice for clock tree design of SDACs is a manual
process, which is time-consuming and error-prone. Clock tree design in digital
domain, however, is fully automated and is carried out by Clock Tree Synthesis
(CTS) software. In spite of critical differences, SDAC clock tree design
problem has fundamental similarities with its digital counterpart. We exploited
these similarities and built a design flow and tool set, which uses commercial
digital CTS software as an intermediate step. We will explain our flow using a
0.18 micron 10-bit 60 MHz 2-stage pipelined differential-input flash
analog-to-digital converter as a test circuit.

    

### [[2108.12770] Risk Assessment, Prediction, and Avoidance of Collision in Autonomous Drones](http://arxiv.org/abs/2108.12770)


  Unmanned Aerial Vehicles (UAVs), in particular Drones, have gained
significant importance in diverse sectors, mainly military uses. Recently, we
can see a growth in acceptance of autonomous UAVs in civilian spaces as well.
However, there is still a long way to go before drones are capable enough to be
safely used without human surveillance. A lot of subsystems and components are
involved in taking care of position estimation, route planning, software/data
security, and collision avoidance to have autonomous drones that fly in
civilian spaces without being harmful to themselves, other UAVs, environment,
or humans. The ultimate goal of this research is to advance collision avoidance
and mitigation techniques through quantitative safety risk assessment. To this
end, it is required to identify the most relevant faults/failures/threats that
can happen during a drone's flight/mission. The analysis of historical data is
also a relevant instrument to help to characterize the most frequent and
relevant issues in UAV systems, which may cause safety hazards. Then we need to
estimate their impact quantitatively, by using fault injection techniques.
Knowing the growing interests in UAVs and their huge potential for future
commercial applications, the expected outcome of this work will be helpful to
researchers for future related research studies. Furthermore, we envisage the
utilization of expected results by companies to develop safer drone
applications, and by air traffic controllers for building failure prediction
and collision avoidance solutions.

    

### [[2108.12771] Towards Reference Architectures for Trustworthy Collaborative Cyber-Physical Systems: Reference Architectures as Boundary Objects](http://arxiv.org/abs/2108.12771)


  This paper presents our work-in-progress study on reference architectures as
boundary objects for realizing trustworthy collaborative Cyber-Physical Systems
(CPS). Furthermore, the preliminary results from interviews with systems
engineering experts from industry and academia are also discussed. The
interview results reveal challenges in using reference architectures during the
system development process. Furthermore, exactly which trustworthiness
attributes (security, availability, reliability, etc.) should be addressed to
realize trustworthy collaborative CPS is identified as an open question, which
we will address in our future work.

    

### [[2108.12773] Towards formally analyzed Cyber-Physical Systems](http://arxiv.org/abs/2108.12773)


  Cyber-physical systems (CPS) can be found everywhere: smart homes, autonomous
vehicles, aircrafts, healthcare, agriculture and industrial production lines.
CPSs are often critical, as system failure can cause serious damage to property
and human lives. Today's cyber-physical systems are extremely complex,
heterogeneous systems: to be able to manage their complexity in a unified way,
we need an infrastructure that ensures that our systems operate with the high
reliability as intended. In addition to the infrastructure, we need to provide
engineers a method to ensure system reliability at design time. The paradigm of
model-driven design provides a toolkit supporting the design and analysis and
by choosing the proper formalisms, the model-driven design approach allows us
to validate our system at design time.

    

### [[2108.12781] Outlier Detection in Smart Grid Communication](http://arxiv.org/abs/2108.12781)


  Industrial Control System (ICS) networks transmit control and monitoring data
in critical environments such as smart grid. Cyber attacks on smart grid
communication may cause fatal consequences on energy production, distribution,
and eventually the lives of people. Since the attacks can be initiated from
both the inside and outside of the network, traditional smart grid security
tools like firewalls or Intrusion Detection Systems (IDS), which are typically
deployed on the edge of the network, are not able to detect internal threats.
For this reason, we also need to analyze behavior of internal ICS
communication.

    

### [[2108.12792] Making Honey Files Sweeter: SentryFS -- A Service-Oriented Smart Ransomware Solution](http://arxiv.org/abs/2108.12792)


  The spread of ransomware continues to cause devastation and is a major
concern for the security community. An often-used technique against this threat
is the use of honey (or canary) files, which serve as ``trip wires'' to detect
ransomware in its early stages. However, in our analysis of ransomware samples
from the wild, we discovered that attackers are well-aware of these traps, and
newer variants use several evasive strategies to bypass traditional honey
files. Hence, we present the design of SentryFS - a specialized file system
that strategically ``sprays'' specially-crafted honey files across the file
system. The canaries are generated using Natural Language Processing (NLP) and
the content and the metadata is constantly updated to make the canaries appear
more attractive for smarter ransomware that is selective in choosing victim
files. Furthermore, to assist with the management of the honey files, SentryFS
connects with an anti-ransomware web service to download the latest
intelligence on novel ransomware strategies to update the canaries. Finally, as
a contingency, SentryFS also leverages file clones to prevent processes from
writing to files directly in the event a highly stealthy ransomware goes
undetected. In this case, the ransomware encrypts the clones rather than the
actual files, leaving users' data unmodified. An AI agent then assigns a
suspicion score to the write activity so that users can approve/discard the
changes. As an early-warning system, the proposed design might help mitigate
the problem of ransomware.

    

### [[2108.12831] A Survey and Comparative Study on Multi-Cloud Architectures: Emerging Issues And Challenges For Cloud Federation](http://arxiv.org/abs/2108.12831)


  Multi-cloud concept has broaden the world of cloud computing and has become a
buzzword today. The word Multi-cloud envisions utilization of services from
multiple heterogeneous cloud providers via a single architecture at customer
premises. Though cloud computing has many issues and offers open research
challenges, still the academics and industrial research has paved a pathway for
multi-cloud environment. The concept of multi-cloud is in maturing phase, and
many research projects are in progress to provide a multi-cloud architecture
which is successfully enabled in all the respects like easy configuration,
security, management etc. In this paper, concepts, challenges, requirement and
future directions for multi-cloud environment are discussed. A survey of
existing approaches and solutions provided by different multi-cloud
architectures is entailed along with analysis of the pros and cons of different
architectures while comparing the same.

    

### [[2108.13162] Parallel Sub-Structuring Methods for solving Sparse Linear Systems on a cluster of GPU](http://arxiv.org/abs/2108.13162)


  The main objective of this work consists in analyzing sub-structuring method
for the parallel solution of sparse linear systems with matrices arising from
the discretization of partial differential equations such as finite element,
finite volume and finite difference. With the success encountered by the
general-purpose processing on graphics processing units (GPGPU), we develop an
hybrid multiGPUs and CPUs sub-structuring algorithm. GPU computing, with CUDA,
is used to accelerate the operations performed on each processor. Numerical
experiments have been performed on a set of matrices arising from engineering
problems. We compare C+MPI implementation on classical CPU cluster with
C+MPI+CUDA on a cluster of GPU. The performance comparison shows a speed-up for
the sub-structuring method up to 19 times in double precision by using CUDA.

    

### [[2108.13169] Enterprise Architecture Model Transformation Engine](http://arxiv.org/abs/2108.13169)


  With increasing linkage within value chains, the IT systems of different
companies are also being connected with each other. This enables the
integration of services within the movement of Industry 4.0 in order to improve
the quality and performance of the processes. Enterprise architecture models
form the basis for this with a better buisness IT-alignment. However, the
heterogeneity of the modeling frameworks and description languages makes a
concatenation considerably difficult, especially differences in syntax,
semantic and relations. Therefore, this paper presents a transformation engine
to convert enterprise architecture models between several languages. We
developed the first generic translation approach that is free of specific
meta-modeling, which is flexible adaptable to arbitrary modeling languages. The
transformation process is defined by various pattern matching techniques using
a rule-based description language. It uses set theory and first-order logic for
an intuitive description as a basis. The concept is practical evaluated using
an example in the area of a large German IT-service provider. Anyhow, the
approach is applicable between a wide range of enterprise architecture
frameworks.

    

### [[2105.06614] Impossibility of Strongly-Linearizable Message-Passing Objects via Simulation by Single-Writer Registers](http://arxiv.org/abs/2105.06614)


  A key way to construct complex distributed systems is through modular
composition of linearizable concurrent objects. A prominent example is shared
registers, which have crash-tolerant implementations on top of message-passing
systems, allowing the advantages of shared memory to carry over to
message-passing. Yet linearizable registers do not always behave properly when
used inside randomized programs. A strengthening of linearizability, called
strong linearizability, has been shown to preserve probabilistic behavior, as
well as other hypersafety properties. In order to exploit composition and
abstraction in message-passing systems, it is crucial to know whether there
exist strongly-linearizable implementations of registers in message-passing.
This paper answers the question in the negative: there are no
strongly-linearizable fault-tolerant message-passing implementations of
multi-writer registers, max-registers, snapshots or counters. This result is
proved by reduction from the corresponding result by Helmi et al. The reduction
is a novel extension of the BG simulation that connects shared-memory and
message-passing, supports long-lived objects, and preserves strong
linearizability. The main technical challenge arises from the discrepancy
between the potentially minuscule fraction of failures to be tolerated in the
simulated message-passing algorithm and the large fraction of failures that can
afflict the simulating shared-memory system. The reduction is general and can
be viewed as the inverse of the ABD simulation of shared memory in
message-passing.

    

### [[2108.12427] Why and How Governments Should Monitor AI Development](http://arxiv.org/abs/2108.12427)


  In this paper we outline a proposal for improving the governance of
artificial intelligence (AI) by investing in government capacity to
systematically measure and monitor the capabilities and impacts of AI systems.
If adopted, this would give governments greater information about the AI
ecosystem, equipping them to more effectively direct AI development and
deployment in the most societally and economically beneficial directions. It
would also create infrastructure that could rapidly identify potential threats
or harms that could occur as a consequence of changes in the AI ecosystem, such
as the emergence of strategically transformative capabilities, or the
deployment of harmful systems.
We begin by outlining the problem which motivates this proposal: in brief,
traditional governance approaches struggle to keep pace with the speed of
progress in AI. We then present our proposal for addressing this problem:
governments must invest in measurement and monitoring infrastructure. We
discuss this proposal in detail, outlining what specific things governments
could focus on measuring and monitoring, and the kinds of benefits this would
generate for policymaking. Finally, we outline some potential pilot projects
and some considerations for implementing this in practice.

    

### [[2108.12463] Automatic Text Evaluation through the Lens of Wasserstein Barycenters](http://arxiv.org/abs/2108.12463)


  A new metric \texttt{BaryScore} to evaluate text generation based on deep
contextualized embeddings (\textit{e.g.}, BERT, Roberta, ELMo) is introduced.
This metric is motivated by a new framework relying on optimal transport tools,
\textit{i.e.}, Wasserstein distance and barycenter. By modelling the layer
output of deep contextualized embeddings as a probability distribution rather
than by a vector embedding; this framework provides a natural way to aggregate
the different outputs through the Wasserstein space topology. In addition, it
provides theoretical grounds to our metric and offers an alternative to
available solutions (\textit{e.g.}, MoverScore and BertScore). Numerical
evaluation is performed on four different tasks: machine translation,
summarization, data2text generation and image captioning. Our results show that
\texttt{BaryScore} outperforms other BERT based metrics and exhibits more
consistent behaviour in particular for text summarization.

    

### [[2108.12465] Code-switched inspired losses for generic spoken dialog representations](http://arxiv.org/abs/2108.12465)


  Spoken dialog systems need to be able to handle both multiple languages and
multilinguality inside a conversation (\textit{e.g} in case of code-switching).
In this work, we introduce new pretraining losses tailored to learn
multilingual spoken dialog representations. The goal of these losses is to
expose the model to code-switched language. To scale up training, we
automatically build a pretraining corpus composed of multilingual conversations
in five different languages (French, Italian, English, German and Spanish) from
\texttt{OpenSubtitles}, a huge multilingual corpus composed of 24.3G tokens. We
test the generic representations on \texttt{MIAM}, a new benchmark composed of
five dialog act corpora on the same aforementioned languages as well as on two
novel multilingual downstream tasks (\textit{i.e} multilingual mask utterance
retrieval and multilingual inconsistency identification). Our experiments show
that our new code switched-inspired losses achieve a better performance in both
monolingual and multilingual settings.

    

### [[2108.12537] Anytime Stochastic Task and Motion Policies](http://arxiv.org/abs/2108.12537)


  In order to solve complex, long-horizon tasks, intelligent robots need to
carry out high-level, abstract planning and reasoning in conjunction with
motion planning. However, abstract models are typically lossy and plans or
policies computed using them can be inexecutable. These problems are
exacerbated in stochastic situations where the robot needs to reason about and
plan for multiple contingencies. We present a new approach for integrated task
and motion planning in stochastic settings. In contrast to prior work in this
direction, we show that our approach can effectively compute integrated task
and motion policies whose branching structures encode agent behaviors that
handle multiple execution-time contingencies. We prove that our algorithm is
probabilistically complete and can compute feasible solution policies in an
anytime fashion so that the probability of encountering an unresolved
contingency decreases over time. Empirical results on a set of challenging
problems show the utility and scope of our method.

    

### [[2108.12565] AMMASurv: Asymmetrical Multi-Modal Attention for Accurate Survival Analysis with Whole Slide Images and Gene Expression Data](http://arxiv.org/abs/2108.12565)


  The use of multi-modal data such as the combination of whole slide images
(WSIs) and gene expression data for survival analysis can lead to more accurate
survival predictions. Previous multi-modal survival models are not able to
efficiently excavate the intrinsic information within each modality. Moreover,
despite experimental results show that WSIs provide more effective information
than gene expression data, previous methods regard the information from
different modalities as similarly important so they cannot flexibly utilize the
potential connection between the modalities. To address the above problems, we
propose a new asymmetrical multi-modal method, termed as AMMASurv.
Specifically, we design an asymmetrical multi-modal attention mechanism (AMMA)
in Transformer encoder for multi-modal data to enable a more flexible
multi-modal information fusion for survival prediction. Different from previous
works, AMMASurv can effectively utilize the intrinsic information within every
modality and flexibly adapts to the modalities of different importance.
Extensive experiments are conducted to validate the effectiveness of the
proposed model. Encouraging results demonstrate the superiority of our method
over other state-of-the-art methods.

    

### [[2108.12582] Distilling the Knowledge of Large-scale Generative Models into Retrieval Models for Efficient Open-domain Conversation](http://arxiv.org/abs/2108.12582)


  Despite the remarkable performance of large-scale generative models in
open-domain conversation, they are known to be less practical for building
real-time conversation systems due to high latency. On the other hand,
retrieval models could return responses with much lower latency but show
inferior performance to the large-scale generative models since the
conversation quality is bounded by the pre-defined response set. To take
advantage of both approaches, we propose a new training method called G2R
(Generative-to-Retrieval distillation) that preserves the efficiency of a
retrieval model while leveraging the conversational ability of a large-scale
generative model by infusing the knowledge of the generative model into the
retrieval model. G2R consists of two distinct techniques of distillation: the
data-level G2R augments the dialogue dataset with additional responses
generated by the large-scale generative model, and the model-level G2R
transfers the response quality score assessed by the generative model to the
score of the retrieval model by the knowledge distillation loss. Through
extensive experiments including human evaluation, we demonstrate that our
retrieval-based conversation system trained with G2R shows a substantially
improved performance compared to the baseline retrieval model while showing
significantly lower inference latency than the large-scale generative models.

    

### [[2108.12599] Smoothing Dialogue States for Open Conversational Machine Reading](http://arxiv.org/abs/2108.12599)


  Conversational machine reading (CMR) requires machines to communicate with
humans through multi-turn interactions between two salient dialogue states of
decision making and question generation processes. In open CMR settings, as the
more realistic scenario, the retrieved background knowledge would be noisy,
which results in severe challenges in the information transmission. Existing
studies commonly train independent or pipeline systems for the two subtasks.
However, those methods are trivial by using hard-label decisions to activate
question generation, which eventually hinders the model performance. In this
work, we propose an effective gating strategy by smoothing the two dialogue
states in only one decoder and bridge decision making and question generation
to provide a richer dialogue state reference. Experiments on the OR-ShARC
dataset show the effectiveness of our method, which achieves new
state-of-the-art results.

    

### [[2108.12705] CHAINGE: A Blockchain Solution to Automate Payment Detail Updates to Subscription Services](http://arxiv.org/abs/2108.12705)


  The rise of the subscription-based business model has led to a corresponding
increase in the number of subscriptions where a customer needs to manage their
payments. This management of payments for multiple subscriptions has become a
very complicated and insecure task for customers, especially when it comes to
renewing payment details when the card is lost, stolen, or expires. In
addition, this, mostly manual, process is vulnerable to human error, digital
frauds, and data breaches, according to security reports. Thus, in this paper,
we propose a novel approach to automate, manage and simplify the Financial
Supply Chain involved in the process of updating and managing payments to user
subscriptions. This is done by utilising the Hyperledger Sawtooth blockchain
framework, that allows a consumer to enter their payment card details in a
central digital wallet and link their subscriptions to their cards. The card
being updated triggers an event on the blockchain, which allow for the payment
details to be updated on subscription systems automatically. The verification
tests performed on the prototype of the proposed system shows that its current
implementation has been securely achieved.

    

### [[2108.12724] Event Extraction as Natural Language Generation](http://arxiv.org/abs/2108.12724)


  Event extraction (EE), the task that identifies event triggers and their
arguments in text, is usually formulated as a classification or structured
prediction problem. Such models usually reduce labels to numeric identifiers,
making them unable to take advantage of label semantics (e.g. an event type
named Arrest is related to words like arrest, detain, or apprehend). This
prevents the generalization to new event types. In this work, we formulate EE
as a natural language generation task and propose GenEE, a model that not only
captures complex dependencies within an event but also generalizes well to
unseen or rare event types. Given a passage and an event type, GenEE is trained
to generate a natural sentence following a predefined template for that event
type. The generated output is then decoded into trigger and argument
predictions. The autoregressive generation process naturally models the
dependencies among the predictions -- each new word predicted depends on those
already generated in the output sentence. Using carefully designed input
prompts during generation, GenEE is able to capture label semantics, which
enables the generalization to new event types. Empirical results show that our
model achieves strong performance on event extraction tasks under all
zero-shot, few-shot, and high-resource scenarios. Especially, in the
high-resource setting, GenEE outperforms the state-of-the-art model on argument
extraction and gets competitive results with the current best on end-to-end EE
tasks.

    

### [[2108.12739] Risk-Aware Fine-Grained Access Control in Cyber-Physical Contexts](http://arxiv.org/abs/2108.12739)


  Access to resources by users may need to be granted only upon certain
conditions and contexts, perhaps particularly in cyber-physical settings.
Unfortunately, creating and modifying context-sensitive access control
solutions in dynamic environments creates ongoing challenges to manage the
authorization contexts. This paper proposes RASA, a context-sensitive access
authorization approach and mechanism leveraging unsupervised machine learning
to automatically infer risk-based authorization decision boundaries. We explore
RASA in a healthcare usage environment, wherein cyber and physical conditions
create context-specific risks for protecting private health information. The
risk levels are associated with access control decisions recommended by a
security policy. A coupling method is introduced to track coexistence of the
objects within context using frequency and duration of coexistence, and these
are clustered to reveal sets of actions with common risk levels; these are used
to create authorization decision boundaries. In addition, we propose a method
for assessing the risk level and labelling the clusters with respect to their
corresponding risk levels. We evaluate the promise of RASA-generated policies
against a heuristic rule-based policy. By employing three different coupling
features (frequency-based, duration-based, and combined features), the
decisions of the unsupervised method and that of the policy are more than 99%
consistent.

    

### [[2108.12820] A Hybrid Rule-Based and Data-Driven Approach to Driver Modeling through Particle Filtering](http://arxiv.org/abs/2108.12820)


  Autonomous vehicles need to model the behavior of surrounding human driven
vehicles to be safe and efficient traffic participants. Existing approaches to
modeling human driving behavior have relied on both data-driven and rule-based
methods. While data-driven models are more expressive, rule-based models are
interpretable, which is an important requirement for safety-critical domains
like driving. However, rule-based models are not sufficiently representative of
data, and data-driven models are yet unable to generate realistic traffic
simulation due to unrealistic driving behavior such as collisions. In this
paper, we propose a methodology that combines rule-based modeling with
data-driven learning. While the rules are governed by interpretable parameters
of the driver model, these parameters are learned online from driving
demonstration data using particle filtering. We perform driver modeling
experiments on the task of highway driving and merging using data from three
real-world driving demonstration datasets. Our results show that driver models
based on our hybrid rule-based and data-driven approach can accurately capture
real-world driving behavior. Further, we assess the realism of the driving
behavior generated by our model by having humans perform a driving Turing test,
where they are asked to distinguish between videos of real driving and those
generated using our driver models.

    

### [[2108.12845] Flow-Guided Video Inpainting with Scene Templates](http://arxiv.org/abs/2108.12845)


  We consider the problem of filling in missing spatio-temporal regions of a
video. We provide a novel flow-based solution by introducing a generative model
of images in relation to the scene (without missing regions) and mappings from
the scene to images. We use the model to jointly infer the scene template, a 2D
representation of the scene, and the mappings. This ensures consistency of the
frame-to-frame flows generated to the underlying scene, reducing geometric
distortions in flow based inpainting. The template is mapped to the missing
regions in the video by a new L2-L1 interpolation scheme, creating crisp
inpaintings and reducing common blur and distortion artifacts. We show on two
benchmark datasets that our approach out-performs state-of-the-art
quantitatively and in user studies.

    

### [[2108.12920] KO codes: Inventing Nonlinear Encoding and Decoding for Reliable Wireless Communication via Deep-learning](http://arxiv.org/abs/2108.12920)


  Landmark codes underpin reliable physical layer communication, e.g.,
Reed-Muller, BCH, Convolution, Turbo, LDPC and Polar codes: each is a linear
code and represents a mathematical breakthrough. The impact on humanity is
huge: each of these codes has been used in global wireless communication
standards (satellite, WiFi, cellular). Reliability of communication over the
classical additive white Gaussian noise (AWGN) channel enables benchmarking and
ranking of the different codes. In this paper, we construct KO codes, a
computationaly efficient family of deep-learning driven (encoder, decoder)
pairs that outperform the state-of-the-art reliability performance on the
standardized AWGN channel. KO codes beat state-of-the-art Reed-Muller and Polar
codes, under the low-complexity successive cancellation decoding, in the
challenging short-to-medium block length regime on the AWGN channel. We show
that the gains of KO codes are primarily due to the nonlinear mapping of
information bits directly to transmit real symbols (bypassing modulation) and
yet possess an efficient, high performance decoder. The key technical
innovation that renders this possible is design of a novel family of neural
architectures inspired by the computation tree of the {\bf K}ronecker {\bf
O}peration (KO) central to Reed-Muller and Polar codes. These architectures
pave way for the discovery of a much richer class of hitherto unexplored
nonlinear algebraic structures. The code is available at
\href{this https URL}{this https URL}

    

### [[2108.12934] Distributed Swarm Collision Avoidance Based on Angular Calculations](http://arxiv.org/abs/2108.12934)


  Collision avoidance is one of the most important topics in the robotics
field. The goal is to move the robots from initial locations to target
locations such that they follow shortest non-colliding paths in the shortest
time and with the least amount of energy. In this paper, a distributed and
real-time algorithm for dense and complex 2D and 3D environments is proposed.
This algorithm uses angular calculations to select the optimal direction for
the movement of each robot and it has been shown that these separate
calculations lead to a form of cooperative behavior among agents. We evaluated
the proposed approach on various simulation and experimental scenarios and
compared the results with FMP and ORCA, two important algorithms in this field.
The results show that the proposed approach is at least 25% faster than ORCA
and at least 7% faster than FMP and also more reliable than both methods. The
proposed method is shown to enable fully autonomous navigation of a swarm of
crazyflies.

    

### [[2108.12941] RetroGAN: A Cyclic Post-Specialization System for Improving Out-of-Knowledge and Rare Word Representations](http://arxiv.org/abs/2108.12941)


  Retrofitting is a technique used to move word vectors closer together or
further apart in their space to reflect their relationships in a Knowledge Base
(KB). However, retrofitting only works on concepts that are present in that KB.
RetroGAN uses a pair of Generative Adversarial Networks (GANs) to learn a
one-to-one mapping between concepts and their retrofitted counterparts. It
applies that mapping (post-specializes) to handle concepts that do not appear
in the original KB in a manner similar to how some natural language systems
handle out-of-vocabulary entries. We test our system on three word-similarity
benchmarks and a downstream sentence simplification task and achieve the state
of the art (CARD-660). Altogether, our results demonstrate our system's
effectiveness for out-of-knowledge and rare word generalization.

    

### [[2108.12957] Searching for Two-Stream Models in Multivariate Space for Video Recognition](http://arxiv.org/abs/2108.12957)


  Conventional video models rely on a single stream to capture the complex
spatial-temporal features. Recent work on two-stream video models, such as
SlowFast network and AssembleNet, prescribe separate streams to learn
complementary features, and achieve stronger performance. However, manually
designing both streams as well as the in-between fusion blocks is a daunting
task, requiring to explore a tremendously large design space. Such manual
exploration is time-consuming and often ends up with sub-optimal architectures
when computational resources are limited and the exploration is insufficient.
In this work, we present a pragmatic neural architecture search approach, which
is able to search for two-stream video models in giant spaces efficiently. We
design a multivariate search space, including 6 search variables to capture a
wide variety of choices in designing two-stream models. Furthermore, we propose
a progressive search procedure, by searching for the architecture of individual
streams, fusion blocks, and attention blocks one after the other. We
demonstrate two-stream models with significantly better performance can be
automatically discovered in our design space. Our searched two-stream models,
namely Auto-TSNet, consistently outperform other models on standard benchmarks.
On Kinetics, compared with the SlowFast model, our Auto-TSNet-L model reduces
FLOPS by nearly 11 times while achieving the same accuracy 78.9%. On
Something-Something-V2, Auto-TSNet-M improves the accuracy by at least 2% over
other methods which use less than 50 GFLOPS per video.

    

### [[2108.12958] 3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations](http://arxiv.org/abs/2108.12958)


  We propose a method to create plausible geometric and texture style
variations of 3D objects in the quest to democratize 3D content creation. Given
a pair of textured source and target objects, our method predicts a part-aware
affine transformation field that naturally warps the source shape to imitate
the overall geometric style of the target. In addition, the texture style of
the target is transferred to the warped source object with the help of a
multi-view differentiable renderer. Our model, 3DStyleNet, is composed of two
sub-networks trained in two stages. First, the geometric style network is
trained on a large set of untextured 3D shapes. Second, we jointly optimize our
geometric style network and a pre-trained image style transfer network with
losses defined over both the geometry and the rendering of the result. Given a
small set of high-quality textured objects, our method can create many novel
stylized shapes, resulting in effortless 3D content creation and style-ware
data augmentation. We showcase our approach qualitatively on 3D content
stylization, and provide user studies to validate the quality of our results.
In addition, our method can serve as a valuable tool to create 3D data
augmentations for computer vision tasks. Extensive quantitative analysis shows
that 3DStyleNet outperforms alternative data augmentation techniques for the
downstream task of single-image 3D reconstruction.

    

### [[2108.13004] X2Teeth: 3D Teeth Reconstruction from a Single Panoramic Radiograph](http://arxiv.org/abs/2108.13004)


  3D teeth reconstruction from X-ray is important for dental diagnosis and many
clinical operations. However, no existing work has explored the reconstruction
of teeth for a whole cavity from a single panoramic radiograph. Different from
single object reconstruction from photos, this task has the unique challenge of
constructing multiple objects at high resolutions. To conquer this task, we
develop a novel ConvNet X2Teeth that decomposes the task into teeth
localization and single-shape estimation. We also introduce a patch-based
training strategy, such that X2Teeth can be end-to-end trained for optimal
performance. Extensive experiments show that our method can successfully
estimate the 3D structure of the cavity and reflect the details for each tooth.
Moreover, X2Teeth achieves a reconstruction IoU of 0.681, which significantly
outperforms the encoder-decoder method by $1.71X and the retrieval-based method
by $1.52X. Our method can also be promising for other multi-anatomy 3D
reconstruction tasks.

    

### [[2108.13024] A Temporal Knowledge Graph Completion Method Based on Balanced Timestamp Distribution](http://arxiv.org/abs/2108.13024)


  Completion through the embedding representation of the knowledge graph (KGE)
has been a research hotspot in recent years. Realistic knowledge graphs are
mostly related to time, while most of the existing KGE algorithms ignore the
time information. A few existing methods directly or indirectly encode the time
information, ignoring the balance of timestamp distribution, which greatly
limits the performance of temporal knowledge graph completion (KGC). In this
paper, a temporal KGC method is proposed based on the direct encoding time
information framework, and a given time slice is treated as the finest
granularity for balanced timestamp distribution. A large number of experiments
on temporal knowledge graph datasets extracted from the real world demonstrate
the effectiveness of our method.

    

### [[2108.13025] Transport-based Counterfactual Models](http://arxiv.org/abs/2108.13025)


  Counterfactual frameworks have grown popular in explainable and fair machine
learning, as they offer a natural notion of causation. However,
state-of-the-art models to compute counterfactuals are either unrealistic or
unfeasible. In particular, while Pearl's causal inference provides appealing
rules to calculate counterfactuals, it relies on a model that is unknown and
hard to discover in practice. We address the problem of designing realistic and
feasible counterfactuals in the absence of a causal model. We define
transport-based counterfactual models as collections of joint probability
distributions between observable distributions, and show their connection to
causal counterfactuals. More specifically, we argue that optimal transport
theory defines relevant transport-based counterfactual models, as they are
numerically feasible, statistically-faithful, and can even coincide with causal
counterfactual models. We illustrate the practicality of these models by
defining sharper fairness criteria than typical group fairness conditions.

    

### [[2108.13035] SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning](http://arxiv.org/abs/2108.13035)


  Autonomous surgical execution relieves tedious routines and surgeon's
fatigue. Recent learning-based methods, especially reinforcement learning (RL)
based methods, achieve promising performance for dexterous manipulation, which
usually requires the simulation to collect data efficiently and reduce the
hardware cost. The existing learning-based simulation platforms for medical
robots suffer from limited scenarios and simplified physical interactions,
which degrades the real-world performance of learned policies. In this work, we
designed SurRoL, an RL-centered simulation platform for surgical robot learning
compatible with the da Vinci Research Kit (dVRK). The designed SurRoL
integrates a user-friendly RL library for algorithm development and a real-time
physics engine, which is able to support more PSM/ECM scenarios and more
realistic physical interactions. Ten learning-based surgical tasks are built in
the platform, which are common in the real autonomous surgical execution. We
evaluate SurRoL using RL algorithms in simulation, provide in-depth analysis,
deploy the trained policies on the real dVRK, and show that our SurRoL achieves
better transferability in the real world.

    

### [[2108.13036] Aleatoric Description Logic for Probailistic Reasoning (Long Version)](http://arxiv.org/abs/2108.13036)


  Description logics are a powerful tool for describing ontological knowledge
bases. That is, they give a factual account of the world in terms of
individuals, concepts and relations. In the presence of uncertainty, such
factual accounts are not feasible, and a subjective or epistemic approach is
required. Aleatoric description logic models uncertainty in the world as
aleatoric events, by the roll of the dice, where an agent has subjective
beliefs about the bias of these dice. This provides a subjective Bayesian
description logic, where propositions and relations are assigned probabilities
according to what a rational agent would bet, given a configuration of possible
individuals and dice. Aleatoric description logic is shown to generalise the
description logic ALC, and can be seen to describe a probability space of
interpretations of a restriction of ALC where all roles are functions. Several
computational problems are considered and model-checking and consistency
checking algorithms are presented. Finally, aleatoric description logic is
shown to be able to model learning, where agents are able to condition their
beliefs on the bias of dice according to observations.

    

### [[2108.13063] Satisfiability and Containment of Recursive SHACL](http://arxiv.org/abs/2108.13063)


  The Shapes Constraint Language (SHACL) is the recent W3C recommendation
language for validating RDF data, by verifying certain shapes on graphs.
Previous work has largely focused on the validation problem and the standard
decision problems of satisfiability and containment, crucial for design and
optimisation purposes, have only been investigated for simplified versions of
SHACL. Moreover, the SHACL specification does not define the semantics of
recursively-defined constraints, which led to several alternative recursive
semantics being proposed in the literature. The interaction between these
different semantics and important decision problems has not been investigated
yet. In this article we provide a comprehensive study of the different features
of SHACL, by providing a translation to a new first-order language, called SCL,
that precisely captures the semantics of SHACL. We also present MSCL, a
second-order extension of SCL, which allows us to define, in a single formal
logic framework, the main recursive semantics of SHACL. Within this language we
also provide an effective treatment of filter constraints which are often
neglected in the related literature. Using this logic we provide a detailed map
of (un)decidability and complexity results for the satisfiability and
containment decision problems for different SHACL fragments. Notably, we prove
that both problems are undecidable for the full language, but we present
decidable combinations of interesting features, even in the face of recursion.

    

### [[2108.13140] A Sentiment Analysis Dataset for Trustworthiness Evaluation](http://arxiv.org/abs/2108.13140)


  While deep learning models have greatly improved the performance of most
artificial intelligence tasks, they are often criticized to be untrustworthy
due to the black-box problem. Consequently, many works have been proposed to
study the trustworthiness of deep learning. However, as most open datasets are
designed for evaluating the accuracy of model outputs, there is still a lack of
appropriate datasets for evaluating the inner workings of neural networks. The
lack of datasets obviously hinders the development of trustworthiness research.
Therefore, in order to systematically evaluate the factors for building
trustworthy systems, we propose a novel and well-annotated sentiment analysis
dataset to evaluate robustness and interpretability. To evaluate these factors,
our dataset contains diverse annotations about the challenging distribution of
instances, manual adversarial instances and sentiment explanations. Several
evaluation metrics are further proposed for interpretability and robustness.
Based on the dataset and metrics, we conduct comprehensive comparisons for the
trustworthiness of three typical models, and also study the relations between
accuracy, robustness and interpretability. We release this trustworthiness
evaluation dataset at \url{https://github/xyz} and hope our work can facilitate
the progress on building more trustworthy systems for real-world applications.

    

### [[1710.00310] Personalized Recommender System for Children's Book Recommendation with A Realtime Interactive Robot](http://arxiv.org/abs/1710.00310)


  In this paper we study the personalized book recommender system in a
child-robot interactive environment. Firstly, we propose a novel text search
algorithm using an inverse filtering mechanism that improves the efficiency.
Secondly, we propose a user interest prediction method based on the Bayesian
network and a novel feedback mechanism. According to children's fuzzy language
input, the proposed method gives the predicted interests. Thirdly, the domain
specific synonym association is proposed based on word vectorization, in order
to improve the understanding of user intention. Experimental results show that
the proposed recommender system has an improved performance and it can operate
on embedded consumer devices with limited computational resources.

    

### [[1908.00409] Deduction Theorem: The Problematic Nature of Common Practice in Game Theory](http://arxiv.org/abs/1908.00409)


  We consider the Deduction Theorem used in the literature of game theory to
run a purported proof by contradiction. In the context of game theory, it is
stated that if we have a proof of $\phi \vdash \varphi$, then we also have a
proof of $\phi \Rightarrow \varphi$. Hence, the proof of $\phi \Rightarrow
\varphi$ is deduced from a previously known statement. However, we argue that
one has to manage to establish that a proof exists for the clauses $\phi$ and
$\varphi$, i.e., they are known true statements in order to show that $\phi
\vdash \varphi$ is provable, and that therefore $\phi \Rightarrow \varphi$ is
provable as well. Thus, we are not allowed to assume that the clause $\phi$ or
$\varphi$ is a true statement. This leads immediately to a wrong conclusion.
Apart from this, we stress to other facts why the Deduction Theorem is not
applicable to run a proof by contradiction. Finally, we present an example from
industrial cooperation where the Deduction Theorem is not correctly applied
with the consequence that the obtained result contradicts the well-known
aggregation issue.

    

### [[2004.12919] First return, then explore](http://arxiv.org/abs/2004.12919)


  The promise of reinforcement learning is to solve complex sequential decision
problems autonomously by specifying a high-level reward function only. However,
reinforcement learning algorithms struggle when, as is often the case, simple
and intuitive rewards provide sparse and deceptive feedback. Avoiding these
pitfalls requires thoroughly exploring the environment, but creating algorithms
that can do so remains one of the central challenges of the field. We
hypothesise that the main impediment to effective exploration originates from
algorithms forgetting how to reach previously visited states ("detachment") and
from failing to first return to a state before exploring from it
("derailment"). We introduce Go-Explore, a family of algorithms that addresses
these two challenges directly through the simple principles of explicitly
remembering promising states and first returning to such states before
intentionally exploring. Go-Explore solves all heretofore unsolved Atari games
and surpasses the state of the art on all hard-exploration games, with orders
of magnitude improvements on the grand challenges Montezuma's Revenge and
Pitfall. We also demonstrate the practical potential of Go-Explore on a
sparse-reward pick-and-place robotics task. Additionally, we show that adding a
goal-conditioned policy can further improve Go-Explore's exploration efficiency
and enable it to handle stochasticity throughout training. The substantial
performance gains from Go-Explore suggest that the simple principles of
remembering states, returning to them, and exploring from them are a powerful
and general approach to exploration, an insight that may prove critical to the
creation of truly intelligent learning agents.

    

### [[2010.08925] Implementing Agent-Based Systems via Computability Logic CL2](http://arxiv.org/abs/2010.08925)


  Computability logic(CoL) is a powerful computational model. In this paper, we
show that CoL naturally supports multi-agent programming models where resources
(coffee for example) are involved. To be specific, we discuss an implementation
of the Starbucks based on CoL (CL2 to be exact).

    

### [[2102.07594] Fast End-to-End Speech Recognition via Non-Autoregressive Models and Cross-Modal Knowledge Transferring from BERT](http://arxiv.org/abs/2102.07594)


  Attention-based encoder-decoder (AED) models have achieved promising
performance in speech recognition. However, because the decoder predicts text
tokens (such as characters or words) in an autoregressive manner, it is
difficult for an AED model to predict all tokens in parallel. This makes the
inference speed relatively slow. We believe that because the encoder already
captures the whole speech utterance, which has the token-level relationship
implicitly, we can predict a token without explicitly autoregressive language
modeling. When the prediction of a token does not rely on other tokens, the
parallel prediction of all tokens in the sequence is realizable. Based on this
idea, we propose a non-autoregressive speech recognition model called LASO
(Listen Attentively, and Spell Once). The model consists of an encoder, a
decoder, and a position dependent summarizer (PDS). The three modules are based
on basic attention blocks. The encoder extracts high-level representations from
the speech. The PDS uses positional encodings corresponding to tokens to
convert the acoustic representations into token-level representations. The
decoder further captures token-level relationships with the self-attention
mechanism. At last, the probability distribution on the vocabulary is computed
for each token position. Therefore, speech recognition is re-formulated as a
position-wise classification problem. Further, we propose a cross-modal
transfer learning method to refine semantics from a large-scale pre-trained
language model BERT for improving the performance.

    

### [[2103.02137] Leading or Following? Dyadic Robot Imitative Interaction Using the Active Inference Framework](http://arxiv.org/abs/2103.02137)


  This study investigated how social interaction among robotic agents changes
dynamically depending on the individual belief of action intention. In a set of
simulation studies, we examine dyadic imitative interactions of robots using a
variational recurrent neural network model. The model is based on the free
energy principle such that a pair of interacting robots find themselves in a
loop, attempting to predict and infer each other's actions using active
inference. We examined how regulating the complexity term to minimize free
energy determines the dynamic characteristics of networks and interactions.
When one robot trained with tighter regulation and another trained with looser
regulation interact, the latter tends to lead the interaction by exerting
stronger action intention, while the former tends to follow by adapting to its
observations. The study confirms that the dyadic imitative interaction becomes
successful by achieving a high synchronization rate when a leader and a
follower are determined by developing action intentions with strong belief and
weak belief, respectively.

    

### [[2105.09647] Survey and Perspective on Social Emotions in Robotics](http://arxiv.org/abs/2105.09647)


  This study reviews research on social emotions in robotics. In robotics, the
study of emotions has been pursued for a long time, including the study of
their recognition, expression, and computational modeling of the basic
mechanisms which underlie them. Research has advanced according to well-known
psychological findings, such as category and dimension theories. Many studies
have been based on these basic theories, addressing only basic emotions.
However, social emotions, also referred to as higher-level emotions, have been
studied in psychology. We believe that these higher-level emotions are worth
pursuing in robotics for next-generation, socially aware robots. In this review
paper, we summarize the findings on social emotions in psychology and
neuroscience, along with a survey of the studies on social emotions in robotics
that have been conducted to date. Thereafter, research directions toward the
implementation of social emotions in robots are discussed.

    

### [[2105.12655] CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks](http://arxiv.org/abs/2105.12655)


  Over the last several decades, software has been woven into the fabric of
every aspect of our society. As software development surges and code
infrastructure of enterprise applications ages, it is now more critical than
ever to increase software development productivity and modernize legacy
applications. Advances in deep learning and machine learning algorithms have
enabled numerous breakthroughs, motivating researchers to leverage AI
techniques to improve software development efficiency. Thus, the fast-emerging
research area of AI for Code has garnered new interest and gathered momentum.
In this paper, we present a large-scale dataset CodeNet, consisting of over 14
million code samples and about 500 million lines of code in 55 different
programming languages, which is aimed at teaching AI to code. In addition to
its large scale, CodeNet has a rich set of high-quality annotations to
benchmark and help accelerate research in AI techniques for a variety of
critical coding tasks, including code similarity and classification, code
translation between a large variety of programming languages, and code
performance (runtime and memory) improvement techniques. Additionally, CodeNet
provides sample input and output test sets for 98.5% of the code samples, which
can be used as an oracle for determining code correctness and potentially guide
reinforcement learning for code quality improvements. As a usability feature,
we provide several pre-processing tools in CodeNet to transform source code
into representations that can be readily used as inputs into machine learning
models. Results of code classification and code similarity experiments using
the CodeNet dataset are provided as a reference. We hope that the scale,
diversity and rich, high-quality annotations of CodeNet will offer
unprecedented research opportunities at the intersection of AI and Software
Engineering.

    

### [[2105.13801] A Probabilistic Forecast-Driven Strategy for a Risk-Aware Participation in the Capacity Firming Market: extended version](http://arxiv.org/abs/2105.13801)


  This paper addresses the energy management of a grid-connected renewable
generation plant coupled with a battery energy storage device in the capacity
firming market, designed to promote renewable power generation facilities in
small non-interconnected grids. The core contribution is to propose a
probabilistic forecast-driven strategy, modeled as a min-max-min robust
optimization problem with recourse. It is solved using a Benders-dual cutting
plane algorithm and a column and constraints generation algorithm in a
tractable manner. A dynamic risk-averse parameters selection strategy based on
the quantile forecasts distribution is proposed to improve the results. A
secondary contribution is to use a recently developed deep learning model known
as normalizing flows to generate quantile forecasts of renewable generation for
the robust optimization problem. This technique provides a general mechanism
for defining expressive probability distributions, only requiring the
specification of a base distribution and a series of bijective transformations.
Overall, the robust approach improves the results over a deterministic approach
with nominal point forecasts by finding a trade-off between conservative and
risk-seeking policies. The case study uses the photovoltaic generation
monitored on-site at the University of Liège (ULiège), Belgium.

    

### [[2107.00456] Crowdsourcing Evaluation of Saliency-based XAI Methods](http://arxiv.org/abs/2107.00456)


  Understanding the reasons behind the predictions made by deep neural networks
is critical for gaining human trust in many important applications, which is
reflected in the increasing demand for explainability in AI (XAI) in recent
years. Saliency-based feature attribution methods, which highlight important
parts of images that contribute to decisions by classifiers, are often used as
XAI methods, especially in the field of computer vision. In order to compare
various saliency-based XAI methods quantitatively, several approaches for
automated evaluation schemes have been proposed; however, there is no guarantee
that such automated evaluation metrics correctly evaluate explainability, and a
high rating by an automated evaluation scheme does not necessarily mean a high
explainability for humans. In this study, instead of the automated evaluation,
we propose a new human-based evaluation scheme using crowdsourcing to evaluate
XAI methods. Our method is inspired by a human computation game, "Peek-a-boom",
and can efficiently compare different XAI methods by exploiting the power of
crowds. We evaluate the saliency maps of various XAI methods on two datasets
with automated and crowd-based evaluation schemes. Our experiments show that
the result of our crowd-based evaluation scheme is different from those of
automated evaluation schemes. In addition, we regard the crowd-based evaluation
results as ground truths and provide a quantitative performance measure to
compare different automated evaluation schemes. We also discuss the impact of
crowd workers on the results and show that the varying ability of crowd workers
does not significantly impact the results.

    

### [[2001.00946] Block-Structured Double-Ended Queues and Bilateral QBD Processes](http://arxiv.org/abs/2001.00946)


  It is interesting and challenging to study double-ended queues with
First-Come-First-Match discipline under customers' impatient behavior and
non-Poisson inputs. Note that the system stability can be guaranteed by the
customers' impatient behavior, but the existence of impatient customers makes
analysis of such double-ended queues more difficult or even impossible to find
any explicitly analytic solution due to having to deal with more complicated
level-dependent Markov processes. Thus it becomes more and more important to
develop effective algorithms in a variety of practical matching problems. This
paper studies a block-structured double-ended queue, whose block structure
comes from two independent Markovian arrival processes (MAPs), which are
non-Poisson inputs. We first show that such a queue can be expressed as a new
bilateral quasi birth-and-death (QBD) process which has its own interest. Based
on this, we provide a detailed analysis for the bilateral QBD process and the
double-ended queue, including the system stability, the stationary queue length
distributions, the average stationary queue lengths, and the sojourn time of
any arriving customers. Then we develop three effective algorithms for
computing the performance measures (e.g., the stationary queue length
distributions, the average stationary queue lengths, and the average sojourn
time) of the double-ended queue. Finally, we use some numerical examples in
tabular and graphical to illustrate how the performance measures are influenced
by some key system parameters. We believe that the methodology and results
given in this paper can be applicable to deal with more general double-ended
queues in practice, and develop effective algorithms for the purpose of many
actual uses.

    

### [[2108.12469] LaForge: Always-Correct and Fast Incremental Builds from Simple Specifications](http://arxiv.org/abs/2108.12469)


  Developers rely on build systems to generate software from code. At a
minimum, a build system should produce build targets from a clean copy of the
code. However, developers rarely work from clean checkouts. Instead, they
rebuild software repeatedly, sometimes hundreds of times a day. To keep
rebuilds fast, build systems run incrementally, executing commands only when
built state cannot be reused. Existing tools like make present users with a
tradeoff. Simple build specifications are easy to write, but limit incremental
work. More complex build specifications produce faster incremental builds, but
writing them is labor-intensive and error-prone. This work shows that no such
tradeoff is necessary; build specifications can be both simple and fast.
We introduce LaForge, a novel build tool that eliminates the need to specify
dependencies or incremental build steps. LaForge builds are easy to specify;
developers write a simple script that runs a full build. Even a single command
like gcc src/*.c will suffice. LaForge traces the execution of the build and
generates a transcript in the TraceIR language. On later builds, LaForge
evaluates the TraceIR transcript to detect changes and perform an efficient
incremental rebuild that automatically captures all build dependencies.
We evaluate LaForge by building 14 software packages, including LLVM and
memcached. Our results show that LaForge automatically generates efficient
builds from simple build specifications. Full builds with LaForge have a median
overhead of 16.1% compared to a project's default full build. LaForge's
incremental builds consistently run fewer commands, and most take less than
3.08s longer than manually-specified incremental builds. Finally, LaForge is
always correct.

    

### [[2108.13114] Embedded Pattern Matching](http://arxiv.org/abs/2108.13114)


  Haskell is a popular choice for hosting deeply embedded languages. A
recurring challenge for these embeddings is how to seamlessly integrate user
defined algebraic data types. In particular, one important, convenient, and
expressive feature for creating and inspecting data -- pattern matching -- is
not directly available on embedded terms. In this paper, we present a novel
technique, embedded pattern matching, which enables a natural and user friendly
embedding of user defined algebraic data types into the embedded language. Our
technique enables users to pattern match on terms in the embedded language in
much the same way they would in the host language.

    

### [[1908.01909] Circular Proofs as Session-Typed Processes: A Local Validity Condition](http://arxiv.org/abs/1908.01909)


  Proof theory provides a foundation for studying and reasoning about
programming languages, most directly based on the well-known Curry-Howard
isomorphism between intuitionistic logic and the typed lambda-calculus. More
recently, a correspondence between intuitionistic linear logic and the
session-typed pi-calculus has been discovered. In this paper, we establish an
extension of the latter correspondence for a fragment of substructural logic
with least and greatest fixed points. We describe the computational
interpretation of the resulting infinitary proof system as session-typed
processes, and provide an effectively decidable local criterion to recognize
mutually recursive processes corresponding to valid circular proofs as
introduced by Fortier and Santocanale. We show that our algorithm imposes a
stricter requirement than Fortier and Santocanale's guard condition, but is
local and compositional and therefore more suitable as the basis for a
programming language.

    

### [[2006.06077] S-semantics -- an example](http://arxiv.org/abs/2006.06077)


  The s-semantics makes it possible to explicitly deal with variables in
program answers. So it seems suitable for programs using nonground data
structures, like open lists. However it is difficult to find published examples
of using the s-semantics to reason about particular programs.
Here we apply s-semantics to prove correctness and completeness of
Frühwirth's $n$ queens program. This is compared with a proof, published
elsewhere, based on the standard semantics and Herbrand interpretations.

    

### [[2103.04880] Iterative Program Synthesis for Adaptable Social Navigation](http://arxiv.org/abs/2103.04880)


  Robot social navigation is influenced by human preferences and
environment-specific scenarios such as elevators and doors, thus necessitating
end-user adaptability. State-of-the-art approaches to social navigation fall
into two categories: model-based social constraints and learning-based
approaches. While effective, these approaches have fundamental limitations --
model-based approaches require constraint and parameter tuning to adapt to
preferences and new scenarios, while learning-based approaches require reward
functions, significant training data, and are hard to adapt to new social
scenarios or new domains with limited demonstrations. In this work, we propose
Iterative Dimension Informed Program Synthesis (IDIPS) to address these
limitations by learning and adapting social navigation in the form of
human-readable symbolic programs. IDIPS works by combining program synthesis,
parameter optimization, predicate repair, and iterative human demonstration to
learn and adapt model-free action selection policies from orders of magnitude
less data than learning-based approaches. We introduce a novel predicate repair
technique that can accommodate previously unseen social scenarios or
preferences by growing existing policies. We present experimental results
showing that IDIPS: 1) synthesizes effective policies that model user
preference, 2) can adapt existing policies to changing preferences, 3) can
extend policies to handle novel social scenarios such as locked doors, and 4)
generates policies that can be transferred from simulation to real-world robots
with minimal effort.

    

### [[2105.05159] Proving LTL Properties of Bitvector Programs and Decompiled Binaries (Extended)](http://arxiv.org/abs/2105.05159)


  There is increasing interest in applying verification tools to programs that
have bitvector operations (eg., binaries). SMT solvers, which serve as a
foundation for these tools, have thus increased support for bitvector reasoning
through bit-blasting and linear arithmetic approximations. In this paper we
show that similar linear arithmetic approximation of bitvector operations can
be done at the source level through transformations. Specifically, we introduce
new paths that over-approximate bitvector operations with linear
conditions/constraints, increasing branching but allowing us to better exploit
the well-developed integer reasoning and interpolation of verification tools.
We show that, for reachability of bitvector programs, increased branching
incurs negligible overhead yet, when combined with integer interpolation
optimizations, enables more programs to be verified. We further show this
exploitation of integer interpolation in the common case also enables
competitive termination verification of bitvector programs and leads to the
first effective technique for LTL verification of bitvector programs. Finally,
we provide an in-depth case study of decompiled ("lifted") binary programs,
which emulate X86 execution through frequent use of bitvector operations. We
present a new tool DarkSea, the first tool capable of verifying reachability,
termination, and LTL of lifted binaries.

    

### [<title>Will old XGB GPU build work for new RTX 3080? - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/will-old-xgb-gpu-build-work-for-new-rtx-3080/2452/1)