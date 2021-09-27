
## 2021-9-27

### [[2109.11607] Proceedings of the 8th OMNeT++ Community Summit, Virtual Summit, September 8-10, 2021](http://arxiv.org/abs/2109.11607)


  These are the Proceedings of the 8th OMNeT++ Community Summit, which was held
virtually on September 8-10, 2021.

    

### [[2109.11624] Opportunistic Spectrum Access: Does Maximizing Throughput Minimize File Transfer Time?](http://arxiv.org/abs/2109.11624)


  The Opportunistic Spectrum Access (OSA) model has been developed for the
secondary users (SUs) to exploit the stochastic dynamics of licensed channels
for file transfer in an opportunistic manner. Common approaches to design
channel sensing strategies for throughput-oriented applications tend to
maximize the long-term throughput, with the hope that it provides reduced file
transfer time as well. In this paper, we show that this is not correct in
general, especially for small files. Unlike prior delay-related works that
seldom consider the heterogeneous channel rate and bursty incoming packets, our
work explicitly considers minimizing the file transfer time of a single file
consisting of multiple packets in a set of heterogeneous channels. We formulate
a mathematical framework for the static policy, and extend to dynamic policy by
mapping our file transfer problem to the stochastic shortest path problem. We
analyze the performance of our proposed static optimal and dynamic optimal
policies over the policy that maximizes long-term throughput. We then propose a
heuristic policy that takes into account the performance-complexity tradeoff
and an extension to online implementation with unknown channel parameters, and
also present the regret bound for our online algorithm. We also present
numerical simulations that reflect our analytical results.

    

### [[2109.11642] Bayesian Inference of a Social Graph with Trace Feasibility Guarantees](http://arxiv.org/abs/2109.11642)


  Network inference is the process of deciding what is the true unknown graph
underlying a set of interactions between nodes. There is a vast literature on
the subject, but most known methods have an important drawback: the inferred
graph is not guaranteed to explain every interaction from the input trace. We
consider this an important issue since such inferred graph cannot be used as
input for applications that require a reliable estimate of the true graph. On
the other hand, a graph having trace feasibility guarantees can help us better
understand the true (hidden) interactions that may have taken place between
nodes of interest. The inference of such graph is the goal of this paper.
Firstly, given an activity log from a social network, we introduce a set of
constraints that take into consideration all the hidden paths that are possible
between the nodes of the trace, given their timestamps of interaction. Then, we
develop a nontrivial modification of the Expectation-Maximization algorithm by
Newman [1], that we call Constrained-EM, which incorporates the constraints and
a set of auxiliary variables into the inference process to guide it towards the
feasibility of the trace. Experimental results on real-world data from Twitter
confirm that Constrained-EM generates a posterior distribution of graphs that
explains all the events observed in the trace while presenting the desired
properties of a scale-free, small-world graph. Our method also outperforms
established methods in terms of feasibility and quality of the inferred graph.

    

### [[2109.11665] WebFlow: Scalable and Decentralized Routing for Payment Channel Networks with High Resource Utilization](http://arxiv.org/abs/2109.11665)


  Payment channel networks (PCNs) have been designed and utilized to address
the scalability challenge and throughput limitation of blockchains. Routing is
a core problem of PCNs. An ideal PCN routing method needs to achieve 1) high
scalability that can maintain low per-node memory and communication cost for
large PCNs, 2) high resource utilization of payment channels, and 3) the
privacy of users. However, none of the existing PCN systems consider all these
requirements. In this work, we propose WebFlow, a distributed routing solution
for PCNs, which only requires each user to maintain localized information and
can be used for massive-scale networks with high resource utilization. We make
use of two distributed data structures: multi-hop Delaunay triangulation (MDT)
originally proposed for wireless networks and our innovation called distributed
Voronoi diagram. We propose new protocols to generate a virtual Euclidean space
in order to apply MDT to PCNs and use the distributed Voronoi diagram to
enhance routing privacy. We conduct extensive simulations and prototype
implementation to further evaluate WebFlow. The results using real and
synthetic PCN topologies and transaction traces show that WebFlow can achieve
extremely low per-node overhead and a high success rate compared to existing
methods.

    

### [[2109.11693] Updating the Theory of Buffer Sizing](http://arxiv.org/abs/2109.11693)


  Routers have packet buffers to reduce packet drops during times of
congestion. It is important to correctly size the buffer: make it too small,
and packets are dropped unnecessarily and the link may be underutilized; make
it too big, and packets may wait for a long time, and the router itself may be
more expensive to build. Despite its importance, there are few guidelines for
picking the buffer size. The two most well-known rules only apply to long-lived
TCP Reno flows; either for a network carrying a single TCP Reno flow (the
buffer size should equal the bandwidth-delay product, or $BDP$) or for a
network carrying $n$ TCP Reno flows (the buffer size should equal
$BDP/\sqrt{n}$). Since these rules were introduced, TCP Reno has been replaced
by newer algorithms as the default congestion control algorithm in all major
operating systems, yet little has been written about how the rules need to
change. This paper revisits both rules. For the single flow case, we generalize
the $BDP$ rule to account for changes to TCP, such as Proportional Rate
Reduction (PRR), and the introduction of new algorithms including Cubic and
BBR. We find that buffers can be made 60-75% smaller for newer algorithms. For
the multiple flow case, we show that the square root of $n$ rule holds under a
broader set of assumptions than previously known, including for these new
congestion control algorithms. We also demonstrate situations where the square
root of $n$ rule does not hold, including for unfair flows and certain settings
with ECN. We validate our results by precisely measuring the time series of
buffer occupancy in a real network, and comparing it to the per-packet window
size.

    

### [[2109.11770] Next generation IEEE 802.11 Wireless Local Area Networks: Current status, future directions and open challenges](http://arxiv.org/abs/2109.11770)


  new generation of Wireless Local Area Networks (WLANs) will make its
appearance in the market in the forthcoming years based on the amendments to
the IEEE 802.11 standards that have recently been approved or are under
development. Examples of the most expected ones are IEEE 802.11aa (Robust Audio
Video Transport Streaming), IEEE 802.11ac (Very-high throughput at < 6 GHz),
IEEE 802.11af (TV White Spaces) and IEEE 802.11ah (Machine-to-Machine
communications) specifications. The aim of this survey is to provide a
comprehensive overview of these novel technical features and the related open
technical challenges that will drive the future WLAN evolution. In contrast to
other IEEE 802.11 surveys, this is a use case oriented study. Specifically, we
first describe the three key scenarios in which next-generation WLANs will have
to operate. We then review the most relevant amendments for each of these use
cases focusing on the additional functionalities and the new technologies they
include, such as multi-user MIMO techniques, groupcast communications, dynamic
channel bonding, spectrum databases and channel sensing, enhanced power saving
mechanisms and efficient small data transmissions. We also discuss the related
work to highlight the key issues that must still be addressed. Finally, we
review emerging trends that can influence the design of future WLANs, with
special focus on software-defined MACs and the internet-working with cellular
systems.

    

### [[2109.11777] Radiation-constrained algorithms for Wireless Energy Transfer in Ad hoc Networks](http://arxiv.org/abs/2109.11777)


  We study the problem of efficiently charging a set of rechargeable nodes
using a set of wireless chargers, under safety constraints on the
electromagnetic radiation incurred. In particular, we define a new charging
model that greatly differs from existing models in that it takes into account
real technology restrictions of the chargers and nodes of the network, mainly
regarding energy limitations. Our model also introduces non-linear constraints
(in the time domain), that radically change the nature of the computational
problems we consider. In this charging model, we present and study the Low
Radiation Efficient Charging Problem (LREC), in which we wish to optimize the
amount of "useful" energy transferred from chargers to nodes (under constraints
on the maximum level of imposed radiation). We present several fundamental
properties of this problem and provide indications of its hardness. Finally, we
propose an iterative local improvement heuristic for LREC, which runs in
polynomial time and we evaluate its performance via simulation. Our algorithm
decouples the computation of the objective function from the computation of the
maximum radiation and also does not depend on the exact formula used for the
computation of the electromagnetic radiation in each point of the network,
achieving good trade-offs between charging efficiency and radiation control; it
also exhibits good energy balance properties. We provide extensive simulation
results supporting our claims and theoretical results.

    

### [[2109.11787] Wireless charging for weighted energy balance in populations of mobile peers](http://arxiv.org/abs/2109.11787)


  Wireless energy transfer is an emerging technology that is used in networks
of battery-powered devices in order to deliver energy and keep the network
functional. Existing state-of-the-art studies have mainly focused on applying
this technology on networks of relatively strong computational and
communicational capabilities (wireless sensor networks, ad-hoc networks); also
they assume energy transfer from special chargers to regular network nodes.
Different from these works, we study how to efficiently transfer energy
wirelessly in populations of battery-limited devices, towards prolonging their
lifetime. In contrast to the state-of-the-art, we assume a much weaker
population of distributed devices which are exchanging energy in a "peer to
peer" manner with each other, without any special charger nodes. We address a
quite general case of diverse energy levels and priorities in the network and
study the problem of how the system can efficiently reach a weighted energy
balance state distributively, under both loss-less and lossy power transfer
assumptions. Three protocols are designed, analyzed and evaluated, achieving
different performance trade-offs between energy balance quality, convergence
time and energy efficiency.

    

### [[2109.11791] An algorithmic study in the vector model for Wireless Power Transfer maximization](http://arxiv.org/abs/2109.11791)


  Rapid technological advances in the domain of Wireless Power Transfer (WPT)
pave the way for novel methods for power management in systems of wireless
devices and recent research works have already started considering algorithmic
solutions for tackling emerging problems. However, many of those works are
limited by the system modelling, and more specifically the one-dimensional
abstraction suggested by Friis formula for the power received by one antenna
under idealized conditions given another antenna some distance away. Different
to those works, we use a model which arises naturally from fundamental
properties of the superposition of energy fields. This model has been shown to
be more realistic than other one-dimensional models that have been used in the
past and can capture superadditive and cancellation effects. Under this model,
we define two new interesting problems for configuring the wireless power
transmitters so as to maximize the total power in the system and we prove that
the first problem can be solved in polynomial time. We present a distributed
solution that runs in pseudo-polynomial time and uses various knowledge levels
and we provide theoretical performance guarantees. Finally, we design three
heuristics for the second problem and evaluate them via simulations.

    

### [[2109.11910] Privacy-Preserving Social Distancing Bracelet](http://arxiv.org/abs/2109.11910)


  This demo presents a functional Proof-of-Concept prototype of a smart
bracelet that utilizes IoT and ML to help in the effort to contain pandemics
such as COVID-19. The designed smart bracelet aids people to navigate life
safely by monitoring health signs; and detecting and alerting people when they
violate social distancing regulations. In addition, the bracelet communicates
with similar bracelets to keep track of recent contacts. Using RFID technology,
the bracelet helps in automating access control to premises such as workplaces.
All this is achieved while preserving the privacy of the users.

    

### [[2109.11970] Making opportunistic networks in IoT environments CCN-ready: A performance evaluation of the MobCCN protocol](http://arxiv.org/abs/2109.11970)


  In future IoT environments it is expected that the role of personal devices
of mobile users in the physical area where IoT devices are deployed will become
more and more important. In particular, due to the push towards
decentralisation of services towards the edge, it is likely that a significant
share of data generated by IoT devices will be needed by other (mobile) nodes
nearby, while global Internet access will be limited only to a small fraction
of data. In this context, opportunistic networking schemes can be adopted to
build efficient content-centric protocols, through which data generated by IoT
devices (or by mobile nodes themselves) can be accessed by the other nodes
nearby. In this paper, we propose MobCCN, which is an ICN-compliant protocol
for this heterogeneous environment. MobCCN is designed to implement the routing
and forwarding mechanisms of the main ICN realisations, such as CCN. The
original aspect of MobCCN is to implement an efficient opportunistic networking
routing scheme to populate the Forwarding Interest Base (FIB) tables of the
nodes, in order to guide the propagation of Interest packets towards nodes that
store the required data. Specifically, MobCCN defines the utility of each node
as a forwarder of Interest packets for a certain type of content, such that
Interest packets can be propagated along a positive utility gradient, until
reaching some node storing the data. We evaluate MobCCN against protocols
representing two possible endpoints of the spectrum, respectively in terms of
minimising the data delivery delay and the resource consumption. Performance
results show that MobCCN is very effective and efficient, as it guarantees very
high delivery rates and low delays, while keeping the total generated traffic
at a reasonable level and also saving local resources.

    

### [[2109.12018] Coupling Microscopic Mobility and Mobile Network Emulation for Pedestrian Communication Applications](http://arxiv.org/abs/2109.12018)


  Network emulation is a well-established method for demonstrating and testing
real devices and mobile apps in a controlled scenario. This paper reports
preliminary results for an open-source extension of the CrowNet pedestrian
communication framework. It enables the interaction between simulated and real
devices using the emulation feature of OMNeT++. The interaction is handled by
several OMNeT++ modules that can be combined to match different use-cases.
Initial timing measurements have been conducted for an example application
which creates decentralized pedestrian density maps based on pedestrian
communication. The results indicate that the approach is feasible for scenarios
with a limited number of pedestrians. This limitation is mainly due to the
real-time simulation requirements in coupled emulation.

    

### [[2109.12046] Developing and experimenting with LEO satellite constellations in OMNeT++](http://arxiv.org/abs/2109.12046)


  In this paper, we present our work in designing and implementing a LEO
satellite constellation simulation model1 within OMNeT++ and INET, which is
validated by comparing the results with existing work. Our model builds upon
the fundamentals of the Open Source Satellite Simulator (OS$^3$), which was
ported to INET 4.3. We describe how the model was integrated on top of the INET
and ported OS$^3$ Framework. We then experiment with the simulation model to
demonstrate its viability in simulating LEO satellite constellations. This
involved simulating both outdated and more recent satellite constellations,
using FCC filing information, to validate latency results.

    

### [[2109.12047] Intermittent Opportunistic Routing Components for the INET Framework](http://arxiv.org/abs/2109.12047)


  Intermittently-powered wireless sensor networks (WSNs) use energy harvesting
and small energy storage to remove the need for battery replacement and to
extend the operational lifetime. However, an intermittently-powered forwarder
regularly turns on or off, which requires alternative networking solutions.
Opportunistic routing (OR) is a potential cross-layer solution for this novel
application, but due to the interaction with the energy storage, the operation
of these protocols is highly dynamic. To compare protocols and components in
like-for-like scenarios we propose module interfaces for MAC, routing and
discovery protocols, that enable clear separation of concerns and good
interchangeability. We also suggest some candidates for each of the protocols
based on our own implementation and research.

    

### [[2109.12048] Deployment and configuration of MEC apps with Simu5G](http://arxiv.org/abs/2109.12048)


  Multi-access Edge Computing (MEC) is expected to act as the enabler for the
integration of 5G (and future 6G) communication technologies with
cloud-computing-based capabilities at the edge of the network. This will enable
low-latency and context-aware applications for users of such mobile networks.
In this paper we describe the implementation of a MEC model for the Simu5G
simulator and illustrate how to configure the environment to evaluate MEC
applications in both simulation and real-time emulation modes.

    

### [[1703.09521] Active Link Obfuscation to Thwart Link-flooding Attacks for Internet of Things](http://arxiv.org/abs/1703.09521)


  The DDoS attack is a serious threat to Internet of Things (IoT). As a new
class of DDoS attack, Link-flooding attack (LFA) disrupts connectivity between
legitimate IoT devices and target servers by flooding only a small number of
links. In this paper, we propose an active LFA mitigation mechanism, called
Linkbait, that is a proactive and preventive defense to throttle LFA for IoT.
We propose a link obfuscation algorithm in Linkbait that selectively reroutes
probing flows to hide target links from adversaries and mislead them to
identify bait links as target links. To block attack traffic and further reduce
the impact in IoT, we propose a compromised IoT devices detection algorithm
that extracts unique traffic patterns of LFA for IoT and leverages support
vector machine (SVM) to identify attack traffic. We evaluate the performance of
Linkbait by using both real-world experiments and large-scale simulations. The
experimental results demonstrate the effectiveness of Linkbait.

    

### [[2105.09553] A New Dynamic Optimal M2M RF interfaces Setting in Relay Selection Algorithm (DORSA) for IoT Applications with the Same Requests](http://arxiv.org/abs/2105.09553)


  Machine-to-Machine (M2M) communication is one of the main communications in
the Internet of Things (IoT). How to send data in these high-density
communications using relay selection can help better performance of this type
of communications in various applications. In addition, the possibility of
simultaneous use of different Radio Frequency (RF) interfaces helps to make
better use of the network radio frequencies. Therefore, in this work, we try to
further use of machine communication equipment and improve the average data
rate of networks in different applications such as the Internet of Things,
which have different bandwidth requirements, by providing an optimization
algorithm for relay selection as well as the simultaneous and dynamic multiple
M2M RF interfaces setting that called Dynamic Optimal Relay Selection and RF
interfaces Setting Algorithm (DORSA). The simulation results show that the
average DORSA\_W-B-Z data rate is improved by 0.8 to 10% compared to the
studied algorithms such as direct transmission as well as relay selection
algorithms with static RF interface setting.

    

### [[2109.11541] CSAGN: Conversational Structure Aware Graph Network for Conversational Semantic Role Labeling](http://arxiv.org/abs/2109.11541)


  Conversational semantic role labeling (CSRL) is believed to be a crucial step
towards dialogue understanding. However, it remains a major challenge for
existing CSRL parser to handle conversational structural information. In this
paper, we present a simple and effective architecture for CSRL which aims to
address this problem. Our model is based on a conversational structure-aware
graph network which explicitly encodes the speaker dependent information. We
also propose a multi-task learning method to further improve the model.
Experimental results on benchmark datasets show that our model with our
proposed training objectives significantly outperforms previous baselines.

    

### [[2109.11542] ADVERSARIALuscator: An Adversarial-DRL Based Obfuscator and Metamorphic Malware SwarmGenerator](http://arxiv.org/abs/2109.11542)


  Advanced metamorphic malware and ransomware, by using obfuscation, could
alter their internal structure with every attack. If such malware could intrude
even into any of the IoT networks, then even if the original malware instance
gets detected, by that time it can still infect the entire network. It is
challenging to obtain training data for such evasive malware. Therefore, in
this paper, we present ADVERSARIALuscator, a novel system that uses specialized
Adversarial-DRL to obfuscate malware at the opcode level and create multiple
metamorphic instances of the same. To the best of our knowledge,
ADVERSARIALuscator is the first-ever system that adopts the Markov Decision
Process-based approach to convert and find a solution to the problem of
creating individual obfuscations at the opcode level. This is important as the
machine language level is the least at which functionality could be preserved
so as to mimic an actual attack effectively. ADVERSARIALuscator is also the
first-ever system to use efficient continuous action control capable of deep
reinforcement learning agents like the Proximal Policy Optimization in the area
of cyber security. Experimental results indicate that ADVERSARIALuscator could
raise the metamorphic probability of a corpus of malware by >0.45.
Additionally, more than 33% of metamorphic instances generated by
ADVERSARIALuscator were able to evade the most potent IDS. If such malware
could intrude even into any of the IoT networks, then even if the original
malware instance gets detected, by that time it can still infect the entire
network. Hence ADVERSARIALuscator could be used to generate data representative
of a swarm of very potent and coordinated AI-based metamorphic malware attacks.
The so generated data and simulations could be used to bolster the defenses of
an IDS against an actual AI-based metamorphic attack from advanced malware and
ransomware.

    

### [[2109.11544] Lifelong 3D Object Recognition and Grasp Synthesis Using Dual Memory Recurrent Self-Organization Networks](http://arxiv.org/abs/2109.11544)


  Humans learn to recognize and manipulate new objects in lifelong settings
without forgetting the previously gained knowledge under non-stationary and
sequential conditions. In autonomous systems, the agents also need to mitigate
similar behavior to continually learn the new object categories and adapt to
new environments. In most conventional deep neural networks, this is not
possible due to the problem of catastrophic forgetting, where the newly gained
knowledge overwrites existing representations. Furthermore, most
state-of-the-art models excel either in recognizing the objects or in grasp
prediction, while both tasks use visual input. The combined architecture to
tackle both tasks is very limited. In this paper, we proposed a hybrid model
architecture consists of a dynamically growing dual-memory recurrent neural
network (GDM) and an autoencoder to tackle object recognition and grasping
simultaneously. The autoencoder network is responsible to extract a compact
representation for a given object, which serves as input for the GDM learning,
and is responsible to predict pixel-wise antipodal grasp configurations. The
GDM part is designed to recognize the object in both instances and categories
levels. We address the problem of catastrophic forgetting using the intrinsic
memory replay, where the episodic memory periodically replays the neural
activation trajectories in the absence of external sensory information. To
extensively evaluate the proposed model in a lifelong setting, we generate a
synthetic dataset due to lack of sequential 3D objects dataset. Experiment
results demonstrated that the proposed model can learn both object
representation and grasping simultaneously in continual learning scenarios.

    

### [[2109.11547] Bridging the Last Mile in Sim-to-Real Robot Perception via Bayesian Active Learning](http://arxiv.org/abs/2109.11547)


  Learning from synthetic data is popular in avariety of robotic vision tasks
such as object detection, becauselarge amount of data can be generated without
annotationsby humans. However, when relying only on synthetic data,we encounter
the well-known problem of the simulation-to-reality (Sim-to-Real) gap, which is
hard to resolve completelyin practice. For such cases, real human-annotated
data isnecessary to bridge this gap, and in our work we focus on howto acquire
this data efficiently. Therefore, we propose a Sim-to-Real pipeline that relies
on deep Bayesian active learningand aims to minimize the manual annotation
efforts. We devisea learning paradigm that autonomously selects the data thatis
considered useful for the human expert to annotate. Toachieve this, a Bayesian
Neural Network (BNN) object detectorproviding reliable uncertain estimates is
adapted to infer theinformativeness of the unlabeled data, in order to
performactive learning. In our experiments on two object detectiondata sets, we
show that the labeling effort required to bridge thereality gap can be reduced
to a small amount. Furthermore, wedemonstrate the practical effectiveness of
this idea in a graspingtask on an assistive robot.

    

### [[2109.11576] Efficient, Interpretable Atomistic Graph Neural Network Representation for Angle-dependent Properties and its Application to Optical Spectroscopy Prediction](http://arxiv.org/abs/2109.11576)


  Graph neural networks (GNNs) are attractive for learning properties of atomic
structures thanks to their intuitive, physically informed graph encoding of
atoms and bonds. However, conventional GNN encodings do not account for angular
information, which is critical for describing complex atomic arrangements in
disordered materials, interfaces, and molecular distortions. In this work, we
extend the recently proposed ALIGNN encoding, which incorporates bond angles,
to also include dihedral angles (ALIGNN-d), and we apply the model to capture
the structures of aqua copper complexes for spectroscopy prediction. This
simple extension is shown to lead to a memory-efficient graph representation
capable of capturing the full geometric information of atomic structures.
Specifically, the ALIGNN-d encoding is a sparse yet equally expressive
representation compared to the dense, maximally-connected graph, in which all
bonds are encoded. We also explore model interpretability based on ALIGNN-d by
elucidating the relative contributions of individual structural components to
the optical response of the copper complexes. Lastly, we briefly discuss future
developments to validate the computational efficiency and to extend the
interpretability of ALIGNN-d.

    

### [[2109.11577] Text Ranking and Classification using Data Compression](http://arxiv.org/abs/2109.11577)


  A well-known but rarely used approach to text categorization uses conditional
entropy estimates computed using data compression tools. Text affinity scores
derived from compressed sizes can be used for classification and ranking tasks,
but their success depends on the compression tools used. We use the Zstandard
compressor and strengthen these ideas in several ways, calling the resulting
language-agnostic technique Zest. In applications, this approach simplifies
configuration, avoiding careful feature extraction and large ML models. Our
ablation studies confirm the value of individual enhancements we introduce. We
show that Zest complements and can compete with language-specific
multidimensional content embeddings in production, but cannot outperform other
counting methods on public datasets.

    

### [[2109.11579] Remaining useful life prediction with uncertainty quantification: development of a highly accurate model for rotating machinery](http://arxiv.org/abs/2109.11579)


  Rotating machinery is essential to modern life, from power generation to
transportation and a host of other industrial applications. Since such
equipment generally operates under challenging working conditions, which can
lead to untimely failures, accurate remaining useful life (RUL) prediction is
essential for maintenance planning and to prevent catastrophic failures. In
this work, we address current challenges in data-driven RUL prediction for
rotating machinery. The challenges revolve around the accuracy and uncertainty
quantification of the prediction, and the non-stationarity of the system
degradation and RUL estimation given sensor data. We devise a novel
architecture and RUL prediction model with uncertainty quantification, termed
VisPro, which integrates time-frequency analysis, deep learning image
recognition, and nonstationary Gaussian process regression. We analyze and
benchmark the results obtained with our model against those of other advanced
data-driven RUL prediction models for rotating machinery using the PHM12
bearing vibration dataset. The computational experiments show that (1) the
VisPro predictions are highly accurate and provide significant improvements
over existing prediction models (three times more accurate than the second-best
model), and (2) the RUL uncertainty bounds are valid and informative. We
identify and discuss the architectural and modeling choices made that explain
this excellent predictive performance of VisPro.

    

### [[2109.11592] Evaluating Attacker Risk Behavior in an Internet of Things Ecosystem](http://arxiv.org/abs/2109.11592)


  In cybersecurity, attackers range from brash, unsophisticated script kiddies
and cybercriminals to stealthy, patient advanced persistent threats. When
modeling these attackers, we can observe that they demonstrate different
risk-seeking and risk-averse behaviors. This work explores how an attacker's
risk seeking or risk averse behavior affects their operations against
detection-optimizing defenders in an Internet of Things ecosystem. Using an
evaluation framework which uses real, parametrizable malware, we develop a game
that is played by a defender against attackers with a suite of malware that is
parameterized to be more aggressive and more stealthy. These results are
evaluated under a framework of exponential utility according to their
willingness to accept risk. We find that against a defender who must choose a
single strategy up front, risk-seeking attackers gain more actual utility than
risk-averse attackers, particularly in cases where the defender is better
equipped than the two attackers anticipate. Additionally, we empirically
confirm that high-risk, high-reward scenarios are more beneficial to
risk-seeking attackers like cybercriminals, while low-risk, low-reward
scenarios are more beneficial to risk-averse attackers like advanced persistent
threats.

    

### [[2109.11602] Chess AI: Competing Paradigms for Machine Intelligence](http://arxiv.org/abs/2109.11602)


  Endgame studies have long served as a tool for testing human creativity and
intelligence. We find that they can serve as a tool for testing machine ability
as well. Two of the leading chess engines, Stockfish and Leela Chess Zero
(LCZero), employ significantly different methods during play. We use Plaskett's
Puzzle, a famous endgame study from the late 1970s, to compare the two engines.
Our experiments show that Stockfish outperforms LCZero on the puzzle. We
examine the algorithmic differences between the engines and use our
observations as a basis for carefully interpreting the test results. Drawing
inspiration from how humans solve chess problems, we ask whether machines can
possess a form of imagination. On the theoretical side, we describe how
Bellman's equation may be applied to optimize the probability of winning. To
conclude, we discuss the implications of our work on artificial intelligence
(AI) and artificial general intelligence (AGI), suggesting possible avenues for
future research.

    

### [[2109.11603] Document Automation Architectures and Technologies: A Survey](http://arxiv.org/abs/2109.11603)


  This paper surveys the current state of the art in document automation (DA).
The objective of DA is to reduce the manual effort during the generation of
documents by automatically integrating input from different sources and
assembling documents conforming to defined templates. There have been reviews
of commercial solutions of DA, particularly in the legal domain, but to date
there has been no comprehensive review of the academic research on DA
architectures and technologies. The current survey of DA reviews the academic
literature and provides a clearer definition and characterization of DA and its
features, identifies state-of-the-art DA architectures and technologies in
academic research, and provides ideas that can lead to new research
opportunities within the DA field in light of recent advances in artificial
intelligence and deep neural networks.

    

### [[2109.11612] Regret Lower Bound and Optimal Algorithm for High-Dimensional Contextual Linear Bandit](http://arxiv.org/abs/2109.11612)


  In this paper, we consider the multi-armed bandit problem with
high-dimensional features. First, we prove a minimax lower bound,
$\mathcal{O}\big((\log d)^{\frac{\alpha+1}{2}}T^{\frac{1-\alpha}{2}}+\log
T\big)$, for the cumulative regret, in terms of horizon $T$, dimension $d$ and
a margin parameter $\alpha\in[0,1]$, which controls the separation between the
optimal and the sub-optimal arms. This new lower bound unifies existing regret
bound results that have different dependencies on T due to the use of different
values of margin parameter $\alpha$ explicitly implied by their assumptions.
Second, we propose a simple and computationally efficient algorithm inspired by
the general Upper Confidence Bound (UCB) strategy that achieves a regret upper
bound matching the lower bound. The proposed algorithm uses a properly centered
$\ell_1$-ball as the confidence set in contrast to the commonly used ellipsoid
confidence set. In addition, the algorithm does not require any forced sampling
step and is thereby adaptive to the practically unknown margin parameter.
Simulations and a real data analysis are conducted to compare the proposed
method with existing ones in the literature.

    

### [[2109.11629] Recurrent Neural Networks for Partially Observed Dynamical Systems](http://arxiv.org/abs/2109.11629)


  Complex nonlinear dynamics are ubiquitous in many fields. Moreover, we rarely
have access to all of the relevant state variables governing the dynamics.
Delay embedding allows us, in principle, to account for unobserved state
variables. Here we provide an algebraic approach to delay embedding that
permits explicit approximation of error. We also provide the asymptotic
dependence of the first order approximation error on the system size. More
importantly, this formulation of delay embedding can be directly implemented
using a Recurrent Neural Network (RNN). This observation expands the
interpretability of both delay embedding and RNN and facilitates principled
incorporation of structure and other constraints into these approaches.

    

### [[2109.11637] Learning Generative Deception Strategies in Combinatorial Masking Games](http://arxiv.org/abs/2109.11637)


  Deception is a crucial tool in the cyberdefence repertoire, enabling
defenders to leverage their informational advantage to reduce the likelihood of
successful attacks. One way deception can be employed is through obscuring, or
masking, some of the information about how systems are configured, increasing
attacker's uncertainty about their targets. We present a novel game-theoretic
model of the resulting defender-attacker interaction, where the defender
chooses a subset of attributes to mask, while the attacker responds by choosing
an exploit to execute. The strategies of both players have combinatorial
structure with complex informational dependencies, and therefore even
representing these strategies is not trivial. First, we show that the problem
of computing an equilibrium of the resulting zero-sum defender-attacker game
can be represented as a linear program with a combinatorial number of system
configuration variables and constraints, and develop a constraint generation
approach for solving this problem. Next, we present a novel highly scalable
approach for approximately solving such games by representing the strategies of
both players as neural networks. The key idea is to represent the defender's
mixed strategy using a deep neural network generator, and then using
alternating gradient-descent-ascent algorithm, analogous to the training of
Generative Adversarial Networks. Our experiments, as well as a case study,
demonstrate the efficacy of the proposed approach.

    

### [[2109.11641] Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection](http://arxiv.org/abs/2109.11641)


  In this paper, we present a novel speaker diarization system for streaming
on-device applications. In this system, we use a transformer transducer to
detect the speaker turns, represent each speaker turn by a speaker embedding,
then cluster these embeddings with constraints from the detected speaker turns.
Compared with conventional clustering-based diarization systems, our system
largely reduces the computational cost of clustering due to the sparsity of
speaker turns. Unlike other supervised speaker diarization systems which
require annotations of time-stamped speaker labels for training, our system
only requires including speaker turn tokens during the transcribing process,
which largely reduces the human efforts involved in data collection.

    

### [[2109.11644] A Learned Stereo Depth System for Robotic Manipulation in Homes](http://arxiv.org/abs/2109.11644)


  We present a passive stereo depth system that produces dense and accurate
point clouds optimized for human environments, including dark, textureless,
thin, reflective and specular surfaces and objects, at 2560x2048 resolution,
with 384 disparities, in 30 ms. The system consists of an algorithm combining
learned stereo matching with engineered filtering, a training and data-mixing
methodology, and a sensor hardware design. Our architecture is 15x faster than
approaches that perform similarly on the Middlebury and Flying Things Stereo
Benchmarks. To effectively supervise the training of this model, we combine
real data labelled using off-the-shelf depth sensors, as well as a number of
different rendered, simulated labeled datasets. We demonstrate the efficacy of
our system by presenting a large number of qualitative results in the form of
depth maps and point-clouds, experiments validating the metric accuracy of our
system and comparisons to other sensors on challenging objects and scenes. We
also show the competitiveness of our algorithm compared to state-of-the-art
learned models using the Middlebury and FlyingThings datasets.

    

### [[2109.11649] Deep Learning with Kernel Flow Regularization for Time Series Forecasting](http://arxiv.org/abs/2109.11649)


  Long Short-Term Memory (LSTM) neural networks have been widely used for time
series forecasting problems. However, LSTMs are prone to overfitting and
performance reduction during test phases. Several different regularization
techniques have been shown in literature to prevent overfitting problems in
neural networks. In this paper, first, we introduce application of kernel flow
methods for time series forecasting in general. Afterward, we examine the
effectiveness of applying kernel flow regularization on LSTM layers to avoid
overfitting problems. We describe a regularization method by applying kernel
flow loss function on LSTM layers. In experimental results, we show that kernel
flow outperforms baseline models on time series forecasting benchmarks. We also
compare the effect of dropout and kernel flow regularization techniques on
LSTMs. The experimental results illustrate that kernel flow achieves similar
regularization effect to dropout. It also shows that the best results is
obtained using both kernel flow and dropout regularizations with early stopping
on LSTM layers on some time series datasets (e.g. power-load demand forecasts).

    

### [[2109.11661] Learning-Based Path Planning for Long-Range Autonomous Valet Parking](http://arxiv.org/abs/2109.11661)


  In this paper, to reduce the congestion rate at the city center and increase
the quality of experience (QoE) of each user, the framework of long-range
autonomous valet parking (LAVP) is presented, where an Electric Autonomous
Vehicle (EAV) is deployed in the city, which can pick up, drop off users at
their required spots, and then drive to the car park out of city center
autonomously. In this framework, we aim to minimize the overall distance of the
EAV, while guarantee all users are served, i.e., picking up, and dropping off
users at their required spots through optimizing the path planning of the EAV
and number of serving time slots. To this end, we first propose a learning
based algorithm, which is named as Double-Layer Ant Colony Optimization
(DL-ACO) algorithm to solve the above problem in an iterative way. Then, to
make the real-time decision, while consider the dynamic environment (i.e., the
EAV may pick up and drop off users from different locations), we further
present a deep reinforcement learning (DRL) based algorithm, which is known as
deep Q network (DQN). The experimental results show that the DL-ACO and
DQN-based algorithms both achieve the considerable performance.

    

### [[2109.11672] A Multi-Agent Deep Reinforcement Learning Coordination Framework for Connected and Automated Vehicles at Merging Roadways](http://arxiv.org/abs/2109.11672)


  The steady increase in the number of vehicles operating on the highways
continues to exacerbate congestion, accidents, energy consumption, and
greenhouse gas emissions. Emerging mobility systems, e.g., connected and
automated vehicles (CAVs), have the potential to directly address these issues
and improve transportation network efficiency and safety. In this paper, we
consider a highway merging scenario and propose a framework for coordinating
CAVs such that stop-and-go driving is eliminated. We use a decentralized form
of the actor-critic approach to deep reinforcement learning$-$multi-agent deep
deterministic policy gradient. We demonstrate the coordination of CAVs through
numerical simulations and show that a smooth traffic flow is achieved by
eliminating stop-and-go driving. Videos and plots of the simulation results can
be found at this supplemental
$\href{this https URL}{site}$.

    

### [[2109.11676] Theory of overparametrization in quantum neural networks](http://arxiv.org/abs/2109.11676)


  The prospect of achieving quantum advantage with Quantum Neural Networks
(QNNs) is exciting. Understanding how QNN properties (e.g., the number of
parameters $M$) affect the loss landscape is crucial to the design of scalable
QNN architectures. Here, we rigorously analyze the overparametrization
phenomenon in QNNs with periodic structure. We define overparametrization as
the regime where the QNN has more than a critical number of parameters $M_c$
that allows it to explore all relevant directions in state space. Our main
results show that the dimension of the Lie algebra obtained from the generators
of the QNN is an upper bound for $M_c$, and for the maximal rank that the
quantum Fisher information and Hessian matrices can reach. Underparametrized
QNNs have spurious local minima in the loss landscape that start disappearing
when $M\geq M_c$. Thus, the overparametrization onset corresponds to a
computational phase transition where the QNN trainability is greatly improved
by a more favorable landscape. We then connect the notion of
overparametrization to the QNN capacity, so that when a QNN is
overparametrized, its capacity achieves its maximum possible value. We run
numerical simulations for eigensolver, compilation, and autoencoding
applications to showcase the overparametrization computational phase
transition. We note that our results also apply to variational quantum
algorithms and quantum optimal control.

    

### [[2109.11678] Optimization Strategies in Multi-Task Learning: Averaged or Separated Losses?](http://arxiv.org/abs/2109.11678)


  In Multi-Task Learning (MTL), it is a common practice to train multi-task
networks by optimizing an objective function, which is a weighted average of
the task-specific objective functions. Although the computational advantages of
this strategy are clear, the complexity of the resulting loss landscape has not
been studied in the literature. Arguably, its optimization may be more
difficult than a separate optimization of the constituting task-specific
objectives. In this work, we investigate the benefits of such an alternative,
by alternating independent gradient descent steps on the different
task-specific objective functions and we formulate a novel way to combine this
approach with state-of-the-art optimizers. As the separation of task-specific
objectives comes at the cost of increased computational time, we propose a
random task grouping as a trade-off between better optimization and
computational efficiency. Experimental results over three well-known visual MTL
datasets show better overall absolute performance on losses and standard
metrics compared to an averaged objective function and other state-of-the-art
MTL methods. In particular, our method shows the most benefits when dealing
with tasks of different nature and it enables a wider exploration of the shared
parameter space. We also show that our random grouping strategy allows to
trade-off between these benefits and computational efficiency.

    

### [[2109.11679] Safe Policy Learning through Extrapolation: Application to Pre-trial Risk Assessment](http://arxiv.org/abs/2109.11679)


  Algorithmic recommendations and decisions have become ubiquitous in today's
society. Many of these and other data-driven policies are based on known,
deterministic rules to ensure their transparency and interpretability. This is
especially true when such policies are used for public policy decision-making.
For example, algorithmic pre-trial risk assessments, which serve as our
motivating application, provide relatively simple, deterministic classification
scores and recommendations to help judges make release decisions.
Unfortunately, existing methods for policy learning are not applicable because
they require existing policies to be stochastic rather than deterministic. We
develop a robust optimization approach that partially identifies the expected
utility of a policy, and then finds an optimal policy by minimizing the
worst-case regret. The resulting policy is conservative but has a statistical
safety guarantee, allowing the policy-maker to limit the probability of
producing a worse outcome than the existing policy. We extend this approach to
common and important settings where humans make decisions with the aid of
algorithmic recommendations. Lastly, we apply the proposed methodology to a
unique field experiment on pre-trial risk assessments. We derive new
classification and recommendation rules that retain the transparency and
interpretability of the existing risk assessment instrument while potentially
leading to better overall outcomes at a lower cost.

    

### [[2109.11680] Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](http://arxiv.org/abs/2109.11680)


  Recent progress in self-training, self-supervised pretraining and
unsupervised learning enabled well performing speech recognition systems
without any labeled data. However, in many cases there is labeled data
available for related languages which is not utilized by these methods. This
paper extends previous work on zero-shot cross-lingual transfer learning by
fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen
languages. This is done by mapping phonemes of the training languages to the
target language using articulatory features. Experiments show that this simple
method significantly outperforms prior work which introduced task-specific
architectures and used only part of a monolingually pretrained model.

    

### [[2109.11683] Optimal Decision Making in High-Throughput Virtual Screening Pipelines](http://arxiv.org/abs/2109.11683)


  Effective selection of the potential candidates that meet certain conditions
in a tremendously large search space has been one of the major concerns in many
real-world applications. In addition to the nearly infinitely large search
space, rigorous evaluation of a sample based on the reliable experimental or
computational platform is often prohibitively expensive, making the screening
problem more challenging. In such a case, constructing a high-throughput
screening (HTS) pipeline that pre-sifts the samples expected to be potential
candidates through the efficient earlier stages, results in a significant
amount of savings in resources. However, to the best of our knowledge, despite
many successful applications, no one has studied optimal pipeline design or
optimal pipeline operations. In this study, we propose two optimization
frameworks, applying to most (if not all) screening campaigns involving
experimental or/and computational evaluations, for optimally determining the
screening thresholds of an HTS pipeline. We validate the proposed frameworks on
both analytic and practical scenarios. In particular, we consider the optimal
computational campaign for the long non-coding RNA (lncRNA) classification as a
practical example. To accomplish this, we built the high-throughput virtual
screening (HTVS) pipeline for classifying the lncRNA. The simulation results
demonstrate that the proposed frameworks significantly reduce the effective
selection cost per potential candidate and make the HTS pipelines less
sensitive to their structural variations. In addition to the validation, we
provide insights on constructing a better HTS pipeline based on the simulation
results.

    

### [[2109.11690] Discovering and Validating AI Errors With Crowdsourced Failure Reports](http://arxiv.org/abs/2109.11690)


  AI systems can fail to learn important behaviors, leading to real-world
issues like safety concerns and biases. Discovering these systematic failures
often requires significant developer attention, from hypothesizing potential
edge cases to collecting evidence and validating patterns. To scale and
streamline this process, we introduce crowdsourced failure reports, end-user
descriptions of how or why a model failed, and show how developers can use them
to detect AI errors. We also design and implement Deblinder, a visual analytics
system for synthesizing failure reports that developers can use to discover and
validate systematic failures. In semi-structured interviews and think-aloud
studies with 10 AI practitioners, we explore the affordances of the Deblinder
system and the applicability of failure reports in real-world settings. Lastly,
we show how collecting additional data from the groups identified by developers
can improve model performance.

    

### [[2109.11692] Dimension-Free Rates for Natural Policy Gradient in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2109.11692)


  Cooperative multi-agent reinforcement learning is a decentralized paradigm in
sequential decision making where agents distributed over a network iteratively
collaborate with neighbors to maximize global (network-wide) notions of
rewards. Exact computations typically involve a complexity that scales
exponentially with the number of agents. To address this curse of
dimensionality, we design a scalable algorithm based on the Natural Policy
Gradient framework that uses local information and only requires agents to
communicate with neighbors within a certain range. Under standard assumptions
on the spatial decay of correlations for the transition dynamics of the
underlying Markov process and the localized learning policy, we show that our
algorithm converges to the globally optimal policy with a dimension-free
statistical and computational complexity, incurring a localization error that
does not depend on the number of agents and converges to zero exponentially
fast as a function of the range of communication.

    

### [[2109.11700] Untrained Graph Neural Networks for Denoising](http://arxiv.org/abs/2109.11700)


  A fundamental problem in signal processing is to denoise a signal. While
there are many well-performing methods for denoising signals defined on regular
supports, such as images defined on two-dimensional grids of pixels, many
important classes of signals are defined over irregular domains such as graphs.
This paper introduces two untrained graph neural network architectures for
graph signal denoising, provides theoretical guarantees for their denoising
capabilities in a simple setup, and numerically validates the theoretical
results in more general scenarios. The two architectures differ on how they
incorporate the information encoded in the graph, with one relying on graph
convolutions and the other employing graph upsampling operators based on
hierarchical clustering. Each architecture implements a different prior over
the targeted signals. To numerically illustrate the validity of the theoretical
results and to compare the performance of the proposed architectures with other
denoising alternatives, we present several experimental results with real and
synthetic datasets.

    

### [[2109.11703] An Improved Frequent Directions Algorithm for Low-Rank Approximation via Block Krylov Iteration](http://arxiv.org/abs/2109.11703)


  Frequent Directions, as a deterministic matrix sketching technique, has been
proposed for tackling low-rank approximation problems. This method has a high
degree of accuracy and practicality, but experiences a lot of computational
cost for large-scale data. Several recent works on the randomized version of
Frequent Directions greatly improve the computational efficiency, but
unfortunately sacrifice some precision. To remedy such issue, this paper aims
to find a more accurate projection subspace to further improve the efficiency
and effectiveness of the existing Frequent Directions techniques. Specifically,
by utilizing the power of Block Krylov Iteration and random projection
technique, this paper presents a fast and accurate Frequent Directions
algorithm named as r-BKIFD. The rigorous theoretical analysis shows that the
proposed r-BKIFD has a comparable error bound with original Frequent
Directions, and the approximation error can be arbitrarily small when the
number of iterations is chosen appropriately. Extensive experimental results on
both synthetic and real data further demonstrate the superiority of r-BKIFD
over several popular Frequent Directions algorithms, both in terms of
computational efficiency and accuracy.

    

### [[2109.11720] Online Adaptation of Parameters using GRU-based Neural Network with BO for Accurate Driving Model](http://arxiv.org/abs/2109.11720)


  Testing self-driving cars in different areas requires surrounding cars with
accordingly different driving styles such as aggressive or conservative styles.
A method of numerically measuring and differentiating human driving styles to
create a virtual driver with a certain driving style is in demand. However,
most methods for measuring human driving styles require thresholds or labels to
classify the driving styles, and some require additional questionnaires for
drivers about their driving attitude. These limitations are not suitable for
creating a large virtual testing environment. Driving models (DMs) simulate
human driving styles. Calibrating a DM makes the simulated driving behavior
closer to human-driving behavior, and enable the simulation of human-driving
cars. Conventional DM-calibrating methods do not take into account that the
parameters in a DM vary while driving. These "fixed" calibrating methods cannot
reflect an actual interactive driving scenario. In this paper, we propose a
DM-calibration method for measuring human driving styles to reproduce real
car-following behavior more accurately. The method includes 1) an objective
entropy weight method for measuring and clustering human driving styles, and 2)
online adaption of DM parameters based on deep learning by combining Bayesian
optimization (BO) and a gated recurrent unit neural network. We conducted
experiments to evaluate the proposed method, and the results indicate that it
can be easily used to measure human driver styles. The experiments also showed
that we can calibrate a corresponding DM in a virtual testing environment with
up to 26% more accuracy than with fixed calibration methods.

    

### [[2109.11723] Distributed Deep Reinforcement Learning for Adaptive Medium Access and Modulation in Shared Spectrum](http://arxiv.org/abs/2109.11723)


  Spectrum scarcity has led to growth in the use of unlicensed spectrum for
cellular systems. This motivates intelligent adaptive approaches to spectrum
access for both WiFi and 5G that improve upon traditional carrier sensing and
listen-before-talk methods. We study decentralized contention-based medium
access for base stations (BSs) of a single Radio Access Technology (RAT)
operating on unlicensed shared spectrum. We devise a learning-based algorithm
for both contention and adaptive modulation that attempts to maximize a
network-wide downlink throughput objective. We formulate and develop novel
distributed implementations of two deep reinforcement learning approaches -
Deep Q Networks and Proximal Policy Optimization - modelled on a two stage
Markov decision process. Empirically, we find the (proportional fairness)
reward accumulated by the policy gradient approach to be significantly higher
than even a genie-aided adaptive energy detection threshold. Our approaches are
further validated by improved sum and peak throughput. The scalability of our
approach to large networks is demonstrated via an improved cumulative reward
earned on both indoor and outdoor layouts with a large number of BSs.

    

### [[2109.11726] MORSE-STF: A Privacy Preserving Computation System](http://arxiv.org/abs/2109.11726)


  Privacy-preserving machine learning has become a popular area of research due
to the increasing concern over data privacy. One way to achieve
privacy-preserving machine learning is to use secure multi-party computation,
where multiple distrusting parties can perform computations on data without
revealing the data itself. We present Secure-TF, a privacy-preserving machine
learning framework based on MPC. Our framework is able to support widely-used
machine learning models such as logistic regression, fully-connected neural
network, and convolutional neural network. We propose novel cryptographic
protocols that has lower round complexity and less communication for computing
sigmoid, ReLU, conv2D and there derivatives. All are central building blocks
for modern machine learning models. With our more efficient protocols, our
system is able to outperform previous state-of-the-art privacy-preserving
machine learning framework in the WAN setting.

    

### [[2109.11730] GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction](http://arxiv.org/abs/2109.11730)


  Recently many efforts have been devoted to applying graph neural networks
(GNNs) to molecular property prediction which is a fundamental task for
computational drug and material discovery. One of major obstacles to hinder the
successful prediction of molecule property by GNNs is the scarcity of labeled
data. Though graph contrastive learning (GCL) methods have achieved
extraordinary performance with insufficient labeled data, most focused on
designing data augmentation schemes for general graphs. However, the
fundamental property of a molecule could be altered with the augmentation
method (like random perturbation) on molecular graphs. Whereas, the critical
geometric information of molecules remains rarely explored under the current
GNN and GCL architectures. To this end, we propose a novel graph contrastive
learning method utilizing the geometry of the molecule across 2D and 3D views,
which is named GeomGCL. Specifically, we first devise a dual-view geometric
message passing network (GeomMPNN) to adaptively leverage the rich information
of both 2D and 3D graphs of a molecule. The incorporation of geometric
properties at different levels can greatly facilitate the molecular
representation learning. Then a novel geometric graph contrastive scheme is
designed to make both geometric views collaboratively supervise each other to
improve the generalization ability of GeomMPNN. We evaluate GeomGCL on various
downstream property prediction tasks via a finetune process. Experimental
results on seven real-life molecular datasets demonstrate the effectiveness of
our proposed GeomGCL against state-of-the-art baselines.

    

### [[2109.11731] Adversarial Neural Trip Recommendation](http://arxiv.org/abs/2109.11731)


  Trip recommender system, which targets at recommending a trip consisting of
several ordered Points of Interest (POIs), has long been treated as an
important application for many location-based services. Currently, most prior
arts generate trips following pre-defined objectives based on constraint
programming, which may fail to reflect the complex latent patterns hidden in
the human mobility data. And most of these methods are usually difficult to
respond in real time when the number of POIs is large. To that end, we propose
an Adversarial Neural Trip Recommendation (ANT) framework to tackle the above
challenges. First of all, we devise a novel attention-based encoder-decoder
trip generator that can learn the correlations among POIs and generate
well-designed trips under given constraints. Another novelty of ANT relies on
an adversarial learning strategy integrating with reinforcement learning to
guide the trip generator to produce high-quality trips. For this purpose, we
introduce a discriminator, which distinguishes the generated trips from
real-life trips taken by users, to provide reward signals to optimize the
generator. Moreover, we devise a novel pre-train schema based on learning from
demonstration, which speeds up the convergence to achieve a
sufficient-and-efficient training process. Extensive experiments on four
real-world datasets validate the effectiveness and efficiency of our proposed
ANT framework, which demonstrates that ANT could remarkably outperform the
state-of-the-art baselines with short response time.

    

### [[2109.11732] Holistic Semi-Supervised Approaches for EEG Representation Learning](http://arxiv.org/abs/2109.11732)


  Recently, supervised methods, which often require substantial amounts of
class labels, have achieved promising results for EEG representation learning.
However, labeling EEG data is a challenging task. More recently, holistic
semi-supervised learning approaches, which only require few output labels, have
shown promising results in the field of computer vision. These methods,
however, have not yet been adapted for EEG learning. In this paper, we adapt
three state-of-the-art holistic semi-supervised approaches, namely MixMatch,
FixMatch, and AdaMatch, as well as five classical semi-supervised methods for
EEG learning. We perform rigorous experiments with all 8 methods on two public
EEG-based emotion recognition datasets, namely SEED and SEED-IV. The
experiments with different amounts of limited labeled samples show that the
holistic approaches achieve strong results even when only 1 labeled sample is
used per class. Further experiments show that in most cases, AdaMatch is the
most effective method, followed by MixMatch and FixMatch.

    

### [[2109.11737] Estimating Rényi's $α$-Cross-Entropies in a Matrix-Based Way](http://arxiv.org/abs/2109.11737)


  Conventional information-theoretic quantities assume access to probability
distributions. Estimating such distributions is not trivial. Here, we consider
function-based formulations of cross entropy that sidesteps this a priori
estimation requirement. We propose three measures of Rényi's
$\alpha$-cross-entropies in the setting of reproducing-kernel Hilbert spaces.
Each measure has its appeals. We prove that we can estimate these measures in
an unbiased, non-parametric, and minimax-optimal way. We do this via
sample-constructed Gram matrices. This yields matrix-based estimators of
Rényi's $\alpha$-cross-entropies. These estimators satisfy all of the axioms
that Rényi established for divergences. Our cross-entropies can thus be used
for assessing distributional differences. They are also appropriate for
handling high-dimensional distributions, since the convergence rate of our
estimator is independent of the sample dimensionality.
Python code for implementing these measures can be found at
this https URL


### [[2109.11750] Indoor Localization Using Smartphone Magnetic with Multi-Scale TCN and LSTM](http://arxiv.org/abs/2109.11750)


  A novel multi-scale temporal convolutional network (TCN) and long short-term
memory network (LSTM) based magnetic localization approach is proposed. To
enhance the discernibility of geomagnetic signals, the time-series
preprocessing approach is constructed at first. Next, the TCN is invoked to
expand the feature dimensions on the basis of keeping the time-series
characteristics of LSTM model. Then, a multi-scale time-series layer is
constructed with multiple TCNs of different dilation factors to address the
problem of inconsistent time-series speed between localization model and mobile
users. A stacking framework of multi-scale TCN and LSTM is eventually proposed
for indoor magnetic localization. Experiment results demonstrate the
effectiveness of the proposed algorithm in indoor localization.

    

### [[2109.11762] Exploring Multi-dimensional Hierarchical Network Topologies for Efficient Distributed Training of Trillion Parameter DL Models](http://arxiv.org/abs/2109.11762)


  Deep Neural Networks have gained significant attraction due to their wide
applicability in different domains. DNN sizes and training samples are
constantly growing, making training of such workloads more challenging.
Distributed training is a solution to reduce the training time.
High-performance distributed training platforms should leverage
multi-dimensional hierarchical networks, which interconnect accelerators
through different levels of the network, to dramatically reduce expensive NICs
required for the scale-out network. However, it comes at the expense of
communication overhead between distributed accelerators to exchange gradients
or input/output activation. In order to allow for further scaling of the
workloads, communication overhead needs to be minimized. In this paper, we
motivate the fact that in training platforms, adding more intermediate network
dimensions is beneficial for efficiently mitigating the excessive use of
expensive NIC resources. Further, we address different challenges of the DNN
training on hierarchical networks. We discuss when designing the interconnect,
how to distribute network bandwidth resources across different dimensions in
order to (i) maximize BW utilization of all dimensions, and (ii) minimizing the
overall training time for the target workload. We then implement a framework
that, for a given workload, determines the best network configuration that
maximizes performance, or performance-per-cost.

    

### [[2109.11765] Dimension Reduction for Data with Heterogeneous Missingness](http://arxiv.org/abs/2109.11765)


  Dimension reduction plays a pivotal role in analysing high-dimensional data.
However, observations with missing values present serious difficulties in
directly applying standard dimension reduction techniques. As a large number of
dimension reduction approaches are based on the Gram matrix, we first
investigate the effects of missingness on dimension reduction by studying the
statistical properties of the Gram matrix with or without missingness, and then
we present a bias-corrected Gram matrix with nice statistical properties under
heterogeneous missingness. Extensive empirical results, on both simulated and
publicly available real datasets, show that the proposed unbiased Gram matrix
can significantly improve a broad spectrum of representative dimension
reduction approaches.

    

### [[2109.11767] Improved Soft Actor-Critic: Mixing Prioritized Off-Policy Samples with On-Policy Experience](http://arxiv.org/abs/2109.11767)


  Soft Actor-Critic (SAC) is an off-policy actor-critic reinforcement learning
algorithm, essentially based on entropy regularization. SAC trains a policy by
maximizing the trade-off between expected return and entropy (randomness in the
policy). It has achieved state-of-the-art performance on a range of
continuous-control benchmark tasks, outperforming prior on-policy and
off-policy methods. SAC works in an off-policy fashion where data are sampled
uniformly from past experiences (stored in a buffer) using which parameters of
the policy and value function networks are updated. We propose certain crucial
modifications for boosting the performance of SAC and make it more sample
efficient. In our proposed improved SAC, we firstly introduce a new
prioritization scheme for selecting better samples from the experience replay
buffer. Secondly we use a mixture of the prioritized off-policy data with the
latest on-policy data for training the policy and the value function networks.
We compare our approach with the vanilla SAC and some recent variants of SAC
and show that our approach outperforms the said algorithmic benchmarks. It is
comparatively more stable and sample efficient when tested on a number of
continuous control tasks in MuJoCo environments.

    

### [[2109.11769] Non-Euclidean Self-Organizing Maps](http://arxiv.org/abs/2109.11769)


  Self-Organizing Maps (SOMs, Kohonen networks) belong to neural network models
of the unsupervised class. In this paper, we present the generalized setup for
non-Euclidean SOMs. Most data analysts take it for granted to use some
subregions of a flat space as their data model; however, by the assumption that
the underlying geometry is non-Euclidean we obtain a new degree of freedom for
the techniques that translate the similarities into spatial neighborhood
relationships. We improve the traditional SOM algorithm by introducing
topology-related extensions. Our proposition can be successfully applied to
dimension reduction, clustering or finding similarities in big data (both
hierarchical and non-hierarchical).

    

### [[2109.11788] Parameter-Free Deterministic Reduction of the Estimation Bias in Continuous Control](http://arxiv.org/abs/2109.11788)


  Approximation of the value functions in value-based deep reinforcement
learning systems induces overestimation bias, resulting in suboptimal policies.
We show that when the reinforcement signals received by the agents have a high
variance, deep actor-critic approaches that overcome the overestimation bias
lead to a substantial underestimation bias. We introduce a parameter-free,
novel deep Q-learning variant to reduce this underestimation bias for
continuous control. By obtaining fixed weights in computing the critic
objective as a linear combination of the approximate critic functions, our
Q-value update rule integrates the concepts of Clipped Double Q-learning and
Maxmin Q-learning. We test the performance of our improvement on a set of
MuJoCo and Box2D continuous control tasks and find that it improves the
state-of-the-art and outperforms the baseline algorithms in the majority of the
environments.

    

### [[2109.11792] Regularization Guarantees Generalization in Bayesian Reinforcement Learning through Algorithmic Stability](http://arxiv.org/abs/2109.11792)


  In the Bayesian reinforcement learning (RL) setting, a prior distribution
over the unknown problem parameters -- the rewards and transitions -- is
assumed, and a policy that optimizes the (posterior) expected return is sought.
A common approximation, which has been recently popularized as meta-RL, is to
train the agent on a sample of $N$ problem instances from the prior, with the
hope that for large enough $N$, good generalization behavior to an unseen test
instance will be obtained. In this work, we study generalization in Bayesian RL
under the probably approximately correct (PAC) framework, using the method of
algorithmic stability. Our main contribution is showing that by adding
regularization, the optimal policy becomes stable in an appropriate sense. Most
stability results in the literature build on strong convexity of the
regularized loss -- an approach that is not suitable for RL as Markov decision
processes (MDPs) are not convex. Instead, building on recent results of fast
convergence rates for mirror descent in regularized MDPs, we show that
regularized MDPs satisfy a certain quadratic growth criterion, which is
sufficient to establish stability. This result, which may be of independent
interest, allows us to study the effect of regularization on generalization in
the Bayesian RL setting.

    

### [[2109.11796] Edge but not Least: Cross-View Graph Pooling](http://arxiv.org/abs/2109.11796)


  Graph neural networks have emerged as a powerful model for graph
representation learning to undertake graph-level prediction tasks. Various
graph pooling methods have been developed to coarsen an input graph into a
succinct graph-level representation through aggregating node embeddings
obtained via graph convolution. However, most graph pooling methods are heavily
node-centric and are unable to fully leverage the crucial information contained
in global graph structure. This paper presents a cross-view graph pooling
(Co-Pooling) method to better exploit crucial graph structure information. The
proposed Co-Pooling fuses pooled representations learnt from both node view and
edge view. Through cross-view interaction, edge-view pooling and node-view
pooling seamlessly reinforce each other to learn more informative graph-level
representations. Co-Pooling has the advantage of handling various graphs with
different types of node attributes. Extensive experiments on a total of 15
graph benchmark datasets validate the effectiveness of our proposed method,
demonstrating its superior performance over state-of-the-art pooling methods on
both graph classification and graph regression tasks.

    

### [[2109.11800] How Does Knowledge Graph Embedding Extrapolate to Unseen Data: a Semantic Evidence View](http://arxiv.org/abs/2109.11800)


  Knowledge Graph Embedding (KGE) aims to learn representations for entities
and relations. Most KGE models have gained great success, especially on
extrapolation scenarios. Specifically, given an unseen triple (h, r, t), a
trained model can still correctly predict t from (h, r, ?), or h from (?, r,
t), such extrapolation ability is impressive. However, most existing KGE works
focus on the design of delicate triple modeling function, which mainly tell us
how to measure the plausibility of observed triples, but we have limited
understanding of why the methods can extrapolate to unseen data, and what are
the important factors to help KGE extrapolate. Therefore in this work, we
attempt to, from a data relevant view, study KGE extrapolation of two problems:
1. How does KGE extrapolate to unseen data? 2. How to design the KGE model with
better extrapolation ability? For the problem 1, we first discuss the impact
factors for extrapolation and from relation, entity and triple level
respectively, propose three Semantic Evidences (SEs), which can be observed
from training set and provide important semantic information for extrapolation
to unseen data. Then we verify the effectiveness of SEs through extensive
experiments on several typical KGE methods, and demonstrate that SEs serve as
an important role for understanding the extrapolation ability of KGE. For the
problem 2, to make better use of the SE information for more extrapolative
knowledge representation, we propose a novel GNN-based KGE model, called
Semantic Evidence aware Graph Neural Network (SE-GNN). Finally, through
extensive experiments on FB15k-237 and WN18RR datasets, we show that SE-GNN
achieves state-of-the-art performance on Knowledge Graph Completion task and
perform a better extrapolation ability.

    

### [[2109.11801] SIM2REALVIZ: Visualizing the Sim2Real Gap in Robot Ego-Pose Estimation](http://arxiv.org/abs/2109.11801)


  The Robotics community has started to heavily rely on increasingly realistic
3D simulators for large-scale training of robots on massive amounts of data.
But once robots are deployed in the real world, the simulation gap, as well as
changes in the real world (e.g. lights, objects displacements) lead to errors.
In this paper, we introduce Sim2RealViz, a visual analytics tool to assist
experts in understanding and reducing this gap for robot ego-pose estimation
tasks, i.e. the estimation of a robot's position using trained models.
Sim2RealViz displays details of a given model and the performance of its
instances in both simulation and real-world. Experts can identify environment
differences that impact model predictions at a given location and explore
through direct interactions with the model hypothesis to fix it. We detail the
design of the tool, and case studies related to the exploit of the regression
to the mean bias and how it can be addressed, and how models are perturbed by
the vanish of landmarks such as bikes.

    

### [[2109.11803] Local Intrinsic Dimensionality Signals Adversarial Perturbations](http://arxiv.org/abs/2109.11803)


  The vulnerability of machine learning models to adversarial perturbations has
motivated a significant amount of research under the broad umbrella of
adversarial machine learning. Sophisticated attacks may cause learning
algorithms to learn decision functions or make decisions with poor predictive
performance. In this context, there is a growing body of literature that uses
local intrinsic dimensionality (LID), a local metric that describes the minimum
number of latent variables required to describe each data point, for detecting
adversarial samples and subsequently mitigating their effects. The research to
date has tended to focus on using LID as a practical defence method often
without fully explaining why LID can detect adversarial samples. In this paper,
we derive a lower-bound and an upper-bound for the LID value of a perturbed
data point and demonstrate that the bounds, in particular the lower-bound, has
a positive correlation with the magnitude of the perturbation. Hence, we
demonstrate that data points that are perturbed by a large amount would have
large LID values compared to unperturbed samples, thus justifying its use in
the prior literature. Furthermore, our empirical validation demonstrates the
validity of the bounds on benchmark datasets.

    

### [[2109.11806] Few-shot Learning Based on Multi-stage Transfer and Class-Balanced Loss for Diabetic Retinopathy Grading](http://arxiv.org/abs/2109.11806)


  Diabetic retinopathy (DR) is one of the major blindness-causing diseases
current-ly known. Automatic grading of DR using deep learning methods not only
speeds up the diagnosis of the disease but also reduces the rate of
misdiagnosis. However, problems such as insufficient samples and imbalanced
class distribu-tion in DR datasets have constrained the improvement of grading
performance. In this paper, we introduce the idea of multi-stage transfer into
the grading task of DR. The new transfer learning technique leverages multiple
datasets with differ-ent scales to enable the model to learn more feature
representation information. Meanwhile, to cope with imbalanced DR datasets, we
present a class-balanced loss function that performs well in natural image
classification tasks, and adopt a simple and easy-to-implement training method
for it. The experimental results show that the application of multi-stage
transfer and class-balanced loss function can effectively improve the grading
performance metrics such as accuracy and quadratic weighted kappa. In fact, our
method has outperformed two state-of-the-art methods and achieved the best
result on the DR grading task of IDRiD Sub-Challenge 2.

    

### [[2109.11808] A dynamic programming algorithm for informative measurements and near-optimal path-planning](http://arxiv.org/abs/2109.11808)


  An informative measurement is the most efficient way to gain information
about an unknown state. We give a first-principles derivation of a
general-purpose dynamic programming algorithm that returns a sequence of
informative measurements by sequentially maximizing the entropy of possible
measurement outcomes. This algorithm can be used by an autonomous agent or
robot to decide where best to measure next, planning a path corresponding to an
optimal sequence of informative measurements. This algorithm is applicable to
states and controls that are continuous or discrete, and agent dynamics that is
either stochastic or deterministic; including Markov decision processes. Recent
results from approximate dynamic programming and reinforcement learning,
including on-line approximations such as rollout and Monte Carlo tree search,
allow an agent or robot to solve the measurement task in real-time. The
resulting near-optimal solutions include non-myopic paths and measurement
sequences that can generally outperform, sometimes substantially, commonly-used
greedy heuristics such as maximizing the entropy of each measurement outcome.
This is demonstrated for a global search problem, where on-line planning with
an extended local search is found to reduce the number of measurements in the
search by half.

    

### [[2109.11812] Predicting pigging operations in oil pipelines](http://arxiv.org/abs/2109.11812)


  This paper presents an innovative machine learning methodology that leverages
on long-term vibroacoustic measurements to perform automated predictions of the
needed pigging operations in crude oil trunklines. Historical pressure signals
have been collected by Eni (e-vpms monitoring system) for two years on discrete
points at a relative distance of 30-35 km along an oil pipeline (100 km length,
16 inch diameter pipes) located in Northern Italy. In order to speed up the
activity and to check the operation logs, a tool has been implemented to
automatically highlight the historical pig operations performed on the line.
Such a tool is capable of detecting, in the observed pressure measurements, the
acoustic noise generated by the travelling pig. All the data sets have been
reanalyzed and exploited by using field data validations to guide a decision
tree regressor (DTR). Several statistical indicators, computed from pressure
head loss between line segments, are fed to the DTR, which automatically
outputs probability values indicating the possible need for pigging the
pipeline. The procedure is applied to the vibroacoustic signals of each pair of
consecutive monitoring stations, such that the proposed predictive maintenance
strategy is capable of tracking the conditions of individual pipeline sections,
thus determining which portion of the conduit is subject to the highest
occlusion levels in order to optimize the clean-up operations. Prediction
accuracy is assessed by evaluating the typical metrics used in statistical
analysis of regression problems, such as the Root Mean Squared Error (RMSE).

    

### [[2109.11817] Unbiased Gradient Estimation with Balanced Assignments for Mixtures of Experts](http://arxiv.org/abs/2109.11817)


  Training large-scale mixture of experts models efficiently on modern hardware
requires assigning datapoints in a batch to different experts, each with a
limited capacity. Recently proposed assignment procedures lack a probabilistic
interpretation and use biased estimators for training. As an alternative, we
propose two unbiased estimators based on principled stochastic assignment
procedures: one that skips datapoints which exceed expert capacity, and one
that samples perfectly balanced assignments using an extension of the
Gumbel-Matching distribution [29]. Both estimators are unbiased, as they
correct for the used sampling procedure. On a toy experiment, we find the
`skip'-estimator is more effective than the balanced sampling one, and both are
more robust in solving the task than biased alternatives.

    

### [[2109.11830] The More, the Better? A Study on Collaborative Machine Learning for DGA Detection](http://arxiv.org/abs/2109.11830)


  Domain generation algorithms (DGAs) prevent the connection between a botnet
and its master from being blocked by generating a large number of domain names.
Promising single-data-source approaches have been proposed for separating
benign from DGA-generated domains. Collaborative machine learning (ML) can be
used in order to enhance a classifier's detection rate, reduce its false
positive rate (FPR), and to improve the classifier's generalization capability
to different networks. In this paper, we complement the research area of DGA
detection by conducting a comprehensive collaborative learning study, including
a total of 13,440 evaluation runs. In two real-world scenarios we evaluate a
total of eleven different variations of collaborative learning using three
different state-of-the-art classifiers. We show that collaborative ML can lead
to a reduction in FPR by up to 51.7%. However, while collaborative ML is
beneficial for DGA detection, not all approaches and classifier types profit
equally. We round up our comprehensive study with a thorough discussion of the
privacy threats implicated by the different collaborative ML approaches.

    

### [[2109.11851] Approximate Latent Force Model Inference](http://arxiv.org/abs/2109.11851)


  Physically-inspired latent force models offer an interpretable alternative to
purely data driven tools for inference in dynamical systems. They carry the
structure of differential equations and the flexibility of Gaussian processes,
yielding interpretable parameters and dynamics-imposed latent functions.
However, the existing inference techniques associated with these models rely on
the exact computation of posterior kernel terms which are seldom available in
analytical form. Most applications relevant to practitioners, such as Hill
equations or diffusion equations, are hence intractable. In this paper, we
overcome these computational problems by proposing a variational solution to a
general class of non-linear and parabolic partial differential equation latent
force models. Further, we show that a neural operator approach can scale our
model to thousands of instances, enabling fast, distributed computation. We
demonstrate the efficacy and flexibility of our framework by achieving
competitive performance on several tasks where the kernels are of varying
degrees of tractability.

    

### [[2109.11861] Training dataset generation for bridge game registration](http://arxiv.org/abs/2109.11861)


  This paper presents a method for automatic generation of a training dataset
for a deep convolutional neural network used for playing card detection. The
solution allows to skip the time-consuming processes of manual image collecting
and labelling recognised objects. The YOLOv4 network trained on the generated
dataset achieved an efficiency of 99.8% in the cards detection task. The
proposed method is a part of a project that aims to automate the process of
broadcasting duplicate bridge competitions using a vision system and neural
networks.

    

### [[2109.11863] Learning Multi-Layered GBDT Via Back Propagation](http://arxiv.org/abs/2109.11863)


  Deep neural networks are able to learn multi-layered representation via back
propagation (BP). Although the gradient boosting decision tree (GBDT) is
effective for modeling tabular data, it is non-differentiable with respect to
its input, thus suffering from learning multi-layered representation. In this
paper, we propose a framework of learning multi-layered GBDT via BP. We
approximate the gradient of GBDT based on linear regression. Specifically, we
use linear regression to replace the constant value at each leaf ignoring the
contribution of individual samples to the tree structure. In this way, we
estimate the gradient for intermediate representations, which facilitates BP
for multi-layered GBDT. Experiments show the effectiveness of the proposed
method in terms of performance and representation ability. To the best of our
knowledge, this is the first work of optimizing multi-layered GBDT via BP. This
work provides a new possibility of exploring deep tree based learning and
combining GBDT with neural networks.

    

### [[2109.11867] Combing Policy Evaluation and Policy Improvement in a Unified f-Divergence Framework](http://arxiv.org/abs/2109.11867)


  The framework of deep reinforcement learning (DRL) provides a powerful and
widely applicable mathematical formalization for sequential decision-making. In
this paper, we start from studying the f-divergence between learning policy and
sampling policy and derive a novel DRL framework, termed f-Divergence
Reinforcement Learning (FRL). We highlight that the policy evaluation and
policy improvement phases are induced by minimizing f-divergence between
learning policy and sampling policy, which is distinct from the conventional
DRL algorithm objective that maximizes the expected cumulative rewards.
Besides, we convert this framework to a saddle-point optimization problem with
a specific f function through Fenchel conjugate, which consists of policy
evaluation and policy improvement. Then we derive new policy evaluation and
policy improvement methods in FRL. Our framework may give new insights for
analyzing DRL algorithms. The FRL framework achieves two advantages: (1) policy
evaluation and policy improvement processes are derived simultaneously by
f-divergence; (2) overestimation issue of value function are alleviated. To
evaluate the effectiveness of the FRL framework, we conduct experiments on
Atari 2600 video games, which show that our framework matches or surpasses the
DRL algorithms we tested.

    

### [[2109.11871] Discovering Novel Customer Features with Recurrent Neural Networks for Personality Based Financial Services](http://arxiv.org/abs/2109.11871)


  The micro-segmentation of customers in the finance sector is a non-trivial
task and has been an atypical omission from recent scientific literature. Where
traditional segmentation classifies customers based on coarse features such as
demographics, micro-segmentation depicts more nuanced differences between
individuals, bringing forth several advantages including the potential for
improved personalization in financial services. AI and representation learning
offer a unique opportunity to solve the problem of micro-segmentation. Although
ubiquitous in many industries, the proliferation of AI in sensitive industries
such as finance has become contingent on the imperatives of responsible AI. We
had previously solved the micro-segmentation problem by extracting temporal
features from the state space of a recurrent neural network (RNN). However, due
to the inherent opacity of RNNs our solution lacked an explanation - one of the
imperatives of responsible AI. In this study, we address this issue by
extracting an explanation for and providing an interpretation of our temporal
features. We investigate the state space of our RNN and through a linear
regression model reconstruct the trajectories in the state space with high
fidelity. We show that our linear regression coefficients have not only learned
the rules used to create the RNN's output data but have also learned the
relationships that were not directly evident in the raw data.

    

### [[2109.11877] Learning-based Noise Component Map Estimation for Image Denoising](http://arxiv.org/abs/2109.11877)


  A problem of image denoising when images are corrupted by a non-stationary
noise is considered in this paper. Since in practice no a priori information on
noise is available, noise statistics should be pre-estimated for image
denoising. In this paper, deep convolutional neural network (CNN) based method
for estimation of a map of local, patch-wise, standard deviations of noise
(so-called sigma-map) is proposed. It achieves the state-of-the-art performance
in accuracy of estimation of sigma-map for the case of non-stationary noise, as
well as estimation of noise variance for the case of additive white Gaussian
noise. Extensive experiments on image denoising using estimated sigma-maps
demonstrate that our method outperforms recent CNN-based blind image denoising
methods by up to 6 dB in PSNR, as well as other state-of-the-art methods based
on sigma-map estimation by up to 0.5 dB, providing same time better usage
flexibility. Comparison with the ideal case, when denoising is applied using
ground-truth sigma-map, shows that a difference of corresponding PSNR values
for most of noise levels is within 0.1-0.2 dB and does not exceeds 0.6 dB.

    

### [[2109.11897] Adaptive Clustering-based Reduced-Order Modeling Framework: Fast and accurate modeling of localized history-dependent phenomena](http://arxiv.org/abs/2109.11897)


  This paper proposes a novel Adaptive Clustering-based Reduced-Order Modeling
(ACROM) framework to significantly improve and extend the recent family of
clustering-based reduced-order models (CROMs). This adaptive framework enables
the clustering-based domain decomposition to evolve dynamically throughout the
problem solution, ensuring optimum refinement in regions where the relevant
fields present steeper gradients. It offers a new route to fast and accurate
material modeling of history-dependent nonlinear problems involving highly
localized plasticity and damage phenomena. The overall approach is composed of
three main building blocks: target clusters selection criterion, adaptive
cluster analysis, and computation of cluster interaction tensors. In addition,
an adaptive clustering solution rewinding procedure and a dynamic adaptivity
split factor strategy are suggested to further enhance the adaptive process.
The coined Adaptive Self-Consistent Clustering Analysis (ASCA) is shown to
perform better than its static counterpart when capturing the multi-scale
elasto-plastic behavior of a particle-matrix composite and predicting the
associated fracture and toughness. Given the encouraging results shown in this
paper, the ACROM framework sets the stage and opens new avenues to explore
adaptivity in the context of CROMs.

    

### [[2109.11909] Learning to maximize global influence from local observations](http://arxiv.org/abs/2109.11909)


  We study a family online influence maximization problems where in a sequence
of rounds $t=1,\ldots,T$, a decision maker selects one from a large number of
agents with the goal of maximizing influence. Upon choosing an agent, the
decision maker shares a piece of information with the agent, which information
then spreads in an unobserved network over which the agents communicate. The
goal of the decision maker is to select the sequence of agents in a way that
the total number of influenced nodes in the network. In this work, we consider
a scenario where the networks are generated independently for each $t$
according to some fixed but unknown distribution, so that the set of influenced
nodes corresponds to the connected component of the random graph containing the
vertex corresponding to the selected agent. Furthermore, we assume that the
decision maker only has access to limited feedback: instead of making the
unrealistic assumption that the entire network is observable, we suppose that
the available feedback is generated based on a small neighborhood of the
selected vertex. Our results show that such partial local observations can be
sufficient for maximizing global influence. We model the underlying random
graph as a sparse inhomogeneous Erdős--Rényi graph, and study three
specific families of random graph models in detail: stochastic block models,
Chung--Lu models and Kronecker random graphs. We show that in these cases one
may learn to maximize influence by merely observing the degree of the selected
vertex in the generated random graph. We propose sequential learning algorithms
that aim at maximizing influence, and provide their theoretical analysis in
both the subcritical and supercritical regimes of all considered models.

    

### [[2109.11920] A Bayesian Optimization Approach for Attenuation Correction in SPECT Brain Imaging](http://arxiv.org/abs/2109.11920)


  Photon attenuation and scatter are the two main physical factors affecting
the diagnostic quality of SPECT in its applications in brain imaging. In this
work, we present a novel Bayesian Optimization approach for Attenuation
Correction (BOAC) in SPECT brain imaging. BOAC utilizes a prior model
parametrizing the head geometry and exploits High Performance Computing (HPC)
to reconstruct attenuation corrected images without requiring prior anatomical
information from complementary CT scans. BOAC is demonstrated in SPECT brain
imaging using noisy and attenuated sinograms, simulated from numerical
phantoms. The quality of the tomographic images obtained with the proposed
method are compared to those obtained without attenuation correction by
employing the appropriate image quality metrics. The quantitative results show
the capacity of BOAC to provide images exhibiting higher contrast and less
background artifacts as compared to the non-attenuation corrected MLEM images.

    

### [[2109.11926] Sinkhorn Distributionally Robust Optimization](http://arxiv.org/abs/2109.11926)


  We study distributionally robust optimization with Sinkorn distance -- a
variant of Wasserstein distance based on entropic regularization. We derive
convex programming dual reformulations when the nominal distribution is an
empirical distribution and a general distribution, respectively. Compared with
Wasserstein DRO, it is computationally tractable for a larger class of loss
functions, and its worst-case distribution is more reasonable. To solve the
dual reformulation, we propose an efficient batch gradient descent with a
bisection search algorithm. Finally, we provide various numerical examples
using both synthetic and real data to demonstrate its competitive performance.

    

### [[2109.11928] Is the Number of Trainable Parameters All That Actually Matters?](http://arxiv.org/abs/2109.11928)


  Recent work has identified simple empirical scaling laws for language models,
linking compute budget, dataset size, model size, and autoregressive modeling
loss. The validity of these simple power laws across orders of magnitude in
model scale provides compelling evidence that larger models are also more
capable models. However, scaling up models under the constraints of hardware
and infrastructure is no easy feat, and rapidly becomes a hard and expensive
engineering problem. We investigate ways to tentatively cheat scaling laws, and
train larger models for cheaper. We emulate an increase in effective
parameters, using efficient approximations: either by doping the models with
frozen random parameters, or by using fast structured transforms in place of
dense linear layers. We find that the scaling relationship between test loss
and compute depends only on the actual number of trainable parameters; scaling
laws cannot be deceived by spurious parameters.

    

### [[2109.11929] Deep Bayesian Estimation for Dynamic Treatment Regimes with a Long Follow-up Time](http://arxiv.org/abs/2109.11929)


  Causal effect estimation for dynamic treatment regimes (DTRs) contributes to
sequential decision making. However, censoring and time-dependent confounding
under DTRs are challenging as the amount of observational data declines over
time due to a reducing sample size but the feature dimension increases over
time. Long-term follow-up compounds these challenges. Another challenge is the
highly complex relationships between confounders, treatments, and outcomes,
which causes the traditional and commonly used linear methods to fail. We
combine outcome regression models with treatment models for high dimensional
features using uncensored subjects that are small in sample size and we fit
deep Bayesian models for outcome regression models to reveal the complex
relationships between confounders, treatments, and outcomes. Also, the
developed deep Bayesian models can model uncertainty and output the prediction
variance which is essential for the safety-aware applications, such as
self-driving cars and medical treatment design. The experimental results on
medical simulations of HIV treatment show the ability of the proposed method to
obtain stable and accurate dynamic causal effect estimation from observational
data, especially with long-term follow-up. Our technique provides practical
guidance for sequential decision making, and policy-making.

    

### [[2109.11939] Discovering PDEs from Multiple Experiments](http://arxiv.org/abs/2109.11939)


  Automated model discovery of partial differential equations (PDEs) usually
considers a single experiment or dataset to infer the underlying governing
equations. In practice, experiments have inherent natural variability in
parameters, initial and boundary conditions that cannot be simply averaged out.
We introduce a randomised adaptive group Lasso sparsity estimator to promote
grouped sparsity and implement it in a deep learning based PDE discovery
framework. It allows to create a learning bias that implies the a priori
assumption that all experiments can be explained by the same underlying PDE
terms with potentially different coefficients. Our experimental results show
more generalizable PDEs can be found from multiple highly noisy datasets, by
this grouped sparsity promotion rather than simply performing independent model
discoveries.

    

### [[2109.11955] Visual Scene Graphs for Audio Source Separation](http://arxiv.org/abs/2109.11955)


  State-of-the-art approaches for visually-guided audio source separation
typically assume sources that have characteristic sounds, such as musical
instruments. These approaches often ignore the visual context of these sound
sources or avoid modeling object interactions that may be useful to better
characterize the sources, especially when the same object class may produce
varied sounds from distinct interactions. To address this challenging problem,
we propose Audio Visual Scene Graph Segmenter (AVSGS), a novel deep learning
model that embeds the visual structure of the scene as a graph and segments
this graph into subgraphs, each subgraph being associated with a unique sound
obtained by co-segmenting the audio spectrogram. At its core, AVSGS uses a
recursive neural network that emits mutually-orthogonal sub-graph embeddings of
the visual graph using multi-head attention. These embeddings are used for
conditioning an audio encoder-decoder towards source separation. Our pipeline
is trained end-to-end via a self-supervised task consisting of separating audio
sources using the visual graph from artificially mixed sounds. In this paper,
we also introduce an "in the wild'' video dataset for sound source separation
that contains multiple non-musical sources, which we call Audio Separation in
the Wild (ASIW). This dataset is adapted from the AudioCaps dataset, and
provides a challenging, natural, and daily-life setting for source separation.
Thorough experiments on the proposed ASIW and the standard MUSIC datasets
demonstrate state-of-the-art sound separation performance of our method against
recent prior approaches.

    

### [[2109.11978] Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](http://arxiv.org/abs/2109.11978)


  In this work, we present and study a training set-up that achieves fast
policy generation for real-world robotic tasks by using massive parallelism on
a single workstation GPU. We analyze and discuss the impact of different
training algorithm components in the massively parallel regime on the final
policy performance and training times. In addition, we present a novel
game-inspired curriculum that is well suited for training with thousands of
simulated robots in parallel. We evaluate the approach by training the
quadrupedal robot ANYmal to walk on challenging terrain. The parallel approach
allows training policies for flat terrain in under four minutes, and in twenty
minutes for uneven terrain. This represents a speedup of multiple orders of
magnitude compared to previous work. Finally, we transfer the policies to the
real robot to validate the approach. We open-source our training code to help
accelerate further research in the field of learned legged locomotion.

    

### [[2109.11990] Optimization-based Causal Estimation from Heterogenous Environments](http://arxiv.org/abs/2109.11990)


  This paper presents a new optimization approach to causal estimation. Given
data that contains covariates and an outcome, which covariates are causes of
the outcome, and what is the strength of the causality? In classical machine
learning (ML), the goal of optimization is to maximize predictive accuracy.
However, some covariates might exhibit a non-causal association to the outcome.
Such spurious associations provide predictive power for classical ML, but they
prevent us from causally interpreting the result. This paper proposes CoCo, an
optimization algorithm that bridges the gap between pure prediction and causal
inference. CoCo leverages the recently-proposed idea of environments, datasets
of covariates/response where the causal relationships remain invariant but
where the distribution of the covariates changes from environment to
environment. Given datasets from multiple environments -- and ones that exhibit
sufficient heterogeneity -- CoCo maximizes an objective for which the only
solution is the causal solution. We describe the theoretical foundations of
this approach and demonstrate its effectiveness on simulated and real datasets.
Compared to classical ML and existing methods, CoCo provides more accurate
estimates of the causal model.

    

### [[2109.12001] Optimisation of MCTS Player for The Lord of the Rings: The Card Game](http://arxiv.org/abs/2109.12001)


  The article presents research on the use of Monte-Carlo Tree Search (MCTS)
methods to create an artificial player for the popular card game "The Lord of
the Rings". The game is characterized by complicated rules, multi-stage round
construction, and a high level of randomness. The described study found that
the best probability of a win is received for a strategy combining expert
knowledge-based agents with MCTS agents at different decision stages. It is
also beneficial to replace random playouts with playouts using expert
knowledge. The results of the final experiments indicate that the relative
effectiveness of the developed solution grows as the difficulty of the game
increases.

    

### [[2109.12002] Optimal policy evaluation using kernel-based temporal difference methods](http://arxiv.org/abs/2109.12002)


  We study methods based on reproducing kernel Hilbert spaces for estimating
the value function of an infinite-horizon discounted Markov reward process
(MRP). We study a regularized form of the kernel least-squares temporal
difference (LSTD) estimate; in the population limit of infinite data, it
corresponds to the fixed point of a projected Bellman operator defined by the
associated reproducing kernel Hilbert space. The estimator itself is obtained
by computing the projected fixed point induced by a regularized version of the
empirical operator; due to the underlying kernel structure, this reduces to
solving a linear system involving kernel matrices. We analyze the error of this
estimate in the $L^2(\mu)$-norm, where $\mu$ denotes the stationary
distribution of the underlying Markov chain. Our analysis imposes no
assumptions on the transition operator of the Markov chain, but rather only
conditions on the reward function and population-level kernel LSTD solutions.
We use empirical process theory techniques to derive a non-asymptotic upper
bound on the error with explicit dependence on the eigenvalues of the
associated kernel operator, as well as the instance-dependent variance of the
Bellman residual error. In addition, we prove minimax lower bounds over
sub-classes of MRPs, which shows that our rate is optimal in terms of the
sample size $n$ and the effective horizon $H = (1 - \gamma)^{-1}$. Whereas
existing worst-case theory predicts cubic scaling ($H^3$) in the effective
horizon, our theory reveals that there is in fact a much wider range of
scalings, depending on the kernel, the stationary distribution, and the
variance of the Bellman residual error. Notably, it is only parametric and
near-parametric problems that can ever achieve the worst-case cubic scaling.

    

### [[2109.12008] Separating Retention from Extraction in the Evaluation of End-to-end Relation Extraction](http://arxiv.org/abs/2109.12008)


  State-of-the-art NLP models can adopt shallow heuristics that limit their
generalization capability (McCoy et al., 2019). Such heuristics include lexical
overlap with the training set in Named-Entity Recognition (Taillé et al.,
2020) and Event or Type heuristics in Relation Extraction (Rosenman et al.,
2020). In the more realistic end-to-end RE setting, we can expect yet another
heuristic: the mere retention of training relation triples. In this paper, we
propose several experiments confirming that retention of known facts is a key
factor of performance on standard benchmarks. Furthermore, one experiment
suggests that a pipeline model able to use intermediate type representations is
less prone to over-rely on retention.

    

### [[2109.12013] Learning Relative Interactions through Imitation](http://arxiv.org/abs/2109.12013)


  In this project we trained a neural network to perform specific interactions
between a robot and objects in the environment, through imitation learning. In
particular, we tackle the task of moving the robot to a fixed pose with respect
to a certain object and later extend our method to handle any arbitrary pose
around this object. We show that a simple network, with relatively little
training data, is able to reach very good performance on the fixed-pose task,
while more work is needed to perform the arbitrary-pose task satisfactorily. We
also explore the effect of ambiguities in the sensor readings, in particular
caused by symmetries in the target object, on the behaviour of the learned
controller.

    

### [[2109.12014] A data acquisition setup for data driven acoustic design](http://arxiv.org/abs/2109.12014)


  In this paper, we present a novel interdisciplinary approach to study the
relationship between diffusive surface structures and their acoustic
performance. Using computational design, surface structures are iteratively
generated and 3D printed at 1:10 model scale. They originate from different
fabrication typologies and are designed to have acoustic diffusion and
absorption effects. An automated robotic process measures the impulse responses
of these surfaces by positioning a microphone and a speaker at multiple
locations. The collected data serves two purposes: first, as an exploratory
catalogue of different spatio-temporal-acoustic scenarios and second, as data
set for predicting the acoustic response of digitally designed surface
geometries using machine learning. In this paper, we present the automated data
acquisition setup, the data processing and the computational generation of
diffusive surface structures. We describe first results of comparative studies
of measured surface panels and conclude with steps of future research.

    

### [[2109.12020] Distributed Estimation of Sparse Inverse Covariance Matrices](http://arxiv.org/abs/2109.12020)


  Learning the relationships between various entities from time-series data is
essential in many applications. Gaussian graphical models have been studied to
infer these relationships. However, existing algorithms process data in a batch
at a central location, limiting their applications in scenarios where data is
gathered by different agents. In this paper, we propose a distributed sparse
inverse covariance algorithm to learn the network structure (i.e., dependencies
among observed entities) in real-time from data collected by distributed
agents. Our approach is built on an online graphical alternating minimization
algorithm, augmented with a consensus term that allows agents to learn the
desired structure cooperatively. We allow the system designer to select the
number of communication rounds and optimization steps per data point. We
characterize the rate of convergence of our algorithm and provide simulations
on synthetic datasets.

    

### [[2109.12021] Pythia: A Customizable Hardware Prefetching Framework Using Online Reinforcement Learning](http://arxiv.org/abs/2109.12021)


  Past research has proposed numerous hardware prefetching techniques, most of
which rely on exploiting one specific type of program context information
(e.g., program counter, cacheline address) to predict future memory accesses.
These techniques either completely neglect a prefetcher's undesirable effects
(e.g., memory bandwidth usage) on the overall system, or incorporate
system-level feedback as an afterthought to a system-unaware prefetch
algorithm. We show that prior prefetchers often lose their performance benefit
over a wide range of workloads and system configurations due to their inherent
inability to take multiple different types of program context and system-level
feedback information into account while prefetching. In this paper, we make a
case for designing a holistic prefetch algorithm that learns to prefetch using
multiple different types of program context and system-level feedback
information inherent to its design.
To this end, we propose Pythia, which formulates the prefetcher as a
reinforcement learning agent. For every demand request, Pythia observes
multiple different types of program context information to make a prefetch
decision. For every prefetch decision, Pythia receives a numerical reward that
evaluates prefetch quality under the current memory bandwidth usage. Pythia
uses this reward to reinforce the correlation between program context
information and prefetch decision to generate highly accurate, timely, and
system-aware prefetch requests in the future. Our extensive evaluations using
simulation and hardware synthesis show that Pythia outperforms multiple
state-of-the-art prefetchers over a wide range of workloads and system
configurations, while incurring only 1.03% area overhead over a desktop-class
processor and no software changes in workloads. The source code of Pythia can
be freely downloaded from this https URL.

    

### [[2109.12026] Description-based Label Attention Classifier for Explainable ICD-9 Classification](http://arxiv.org/abs/2109.12026)


  ICD-9 coding is a relevant clinical billing task, where unstructured texts
with information about a patient's diagnosis and treatments are annotated with
multiple ICD-9 codes. Automated ICD-9 coding is an active research field, where
CNN- and RNN-based model architectures represent the state-of-the-art
approaches. In this work, we propose a description-based label attention
classifier to improve the model explainability when dealing with noisy texts
like clinical notes. We evaluate our proposed method with different
transformer-based encoders on the MIMIC-III-50 dataset. Our method achieves
strong results together with augmented explainablilty.

    

### [[2109.12029] Identifying Distributional Differences in Convective Evolution Prior to Rapid Intensification in Tropical Cyclones](http://arxiv.org/abs/2109.12029)


  Tropical cyclone (TC) intensity forecasts are issued by human forecasters who
evaluate spatio-temporal observations (e.g., satellite imagery) and model
output (e.g., numerical weather prediction, statistical models) to produce
forecasts every 6 hours. Within these time constraints, it can be challenging
to draw insight from such data. While high-capacity machine learning methods
are well suited for prediction problems with complex sequence data, extracting
interpretable scientific information with such methods is difficult. Here we
leverage powerful AI prediction algorithms and classical statistical inference
to identify patterns in the evolution of TC convective structure leading up to
the rapid intensification of a storm, hence providing forecasters and
scientists with key insight into TC behavior.

    

### [[2109.12040] From images in the wild to video-informed image classification](http://arxiv.org/abs/2109.12040)


  Image classifiers work effectively when applied on structured images, yet
they often fail when applied on images with very high visual complexity. This
paper describes experiments applying state-of-the-art object classifiers toward
a unique set of images in the wild with high visual complexity collected on the
island of Bali. The text describes differences between actual images in the
wild and images from Imagenet, and then discusses a novel approach combining
informational cues particular to video with an ensemble of imperfect
classifiers in order to improve classification results on video sourced images
of plants in the wild.

    

### [[2109.12042] Combining Discrete Choice Models and Neural Networks through Embeddings: Formulation, Interpretability and Performance](http://arxiv.org/abs/2109.12042)


  This study proposes a novel approach that combines theory and data-driven
choice models using Artificial Neural Networks (ANNs). In particular, we use
continuous vector representations, called embeddings, for encoding categorical
or discrete explanatory variables with a special focus on interpretability and
model transparency. Although embedding representations within the logit
framework have been conceptualized by Camara (2019), their dimensions do not
have an absolute definitive meaning, hence offering limited behavioral
insights. The novelty of our work lies in enforcing interpretability to the
embedding vectors by formally associating each of their dimensions to a choice
alternative. Thus, our approach brings benefits much beyond a simple
parsimonious representation improvement over dummy encoding, as it provides
behaviorally meaningful outputs that can be used in travel demand analysis and
policy decisions. Additionally, in contrast to previously suggested ANN-based
Discrete Choice Models (DCMs) that either sacrifice interpretability for
performance or are only partially interpretable, our models preserve
interpretability of the utility coefficients for all the input variables
despite being based on ANN principles. The proposed models were tested on two
real world datasets and evaluated against benchmark and baseline models that
use dummy-encoding. The results of the experiments indicate that our models
deliver state-of-the-art predictive performance, outperforming existing
ANN-based models while drastically reducing the number of required network
parameters.

    

### [[2109.12043] Sample Efficient Model Evaluation](http://arxiv.org/abs/2109.12043)


  Labelling data is a major practical bottleneck in training and testing
classifiers. Given a collection of unlabelled data points, we address how to
select which subset to label to best estimate test metrics such as accuracy,
$F_1$ score or micro/macro $F_1$. We consider two sampling based approaches,
namely the well-known Importance Sampling and we introduce a novel application
of Poisson Sampling. For both approaches we derive the minimal error sampling
distributions and how to approximate and use them to form estimators and
confidence intervals. We show that Poisson Sampling outperforms Importance
Sampling both theoretically and experimentally.

    

### [[2109.12062] A Generative Federated Learning Framework for Differential Privacy](http://arxiv.org/abs/2109.12062)


  In machine learning, differential privacy and federated learning concepts are
gaining more and more importance in an increasingly interconnected world. While
the former refers to the sharing of private data characterized by strict
security rules to protect individual privacy, the latter refers to distributed
learning techniques in which a central server exchanges information with
different clients for machine learning purposes. In recent years, many studies
have shown the possibility of bypassing the privacy shields of these systems
and exploiting the vulnerabilities of machine learning models, making them leak
the information with which they have been trained. In this work, we present the
3DGL framework, an alternative to the current federated learning paradigms. Its
goal is to share generative models with high levels of
$\varepsilon$-differential privacy. In addition, we propose DDP-$\beta$VAE, a
deep generative model capable of generating synthetic data with high levels of
utility and safety for the individual. We evaluate the 3DGL framework based on
DDP-$\beta$VAE, showing how the overall system is resilient to the principal
attacks in federated learning and improves the performance of distributed
learning algorithms.

    

### [[2109.12063] Reduced-Lead ECG Classifier Model Trained with DivideMix and Model Ensemble](http://arxiv.org/abs/2109.12063)


  Automatic diagnosis of multiple cardiac abnormalities from reduced-lead
electrocardiogram (ECG) data is challenging. One of the reasons for this is the
difficulty of defining labels from standard 12-lead data. Reduced-lead ECG data
usually do not have identical characteristics of cardiac abnormalities because
of the noisy label problem. Thus, there is an inconsistency in the annotated
labels between the reduced-lead and 12-lead ECG data. To solve this, we propose
deep neural network (DNN)-based ECG classifier models that incorporate
DivideMix and stochastic weight averaging (SWA). DivideMix was used to refine
the noisy label by using two separate models. Besides DivideMix, we used a
model ensemble technique, SWA, which also focuses on the noisy label problem,
to enhance the effect of the models generated by DivideMix. Our classifiers
(ami_kagoshima) received scores of 0.49, 0.47, 0.48, 0.47, and 0.47 (ranked
9th, 10th, 10th, 11th, and 10th, respectively, out of 39 teams) for the
12-lead, 6-lead, 4-lead, 3-lead, and 2-lead versions, respectively, of the
hidden test set with the challenge evaluation metric. We obtained the scores of
0.701, 0.686, 0.693, 0.693, and 0.685 on the 10-fold cross validation, and
0.623, 0.593, 0.606, 0.612, and 0.601 on the hidden validation set for each
lead combination.

    

### [[2109.12073] A Graph Policy Network Approach for Volt-Var Control in Power Distribution Systems](http://arxiv.org/abs/2109.12073)


  Volt-var control (VVC) is the problem of operating power distribution systems
within healthy regimes by controlling actuators in power systems. Existing
works have mostly adopted the conventional routine of representing the power
systems (a graph with tree topology) as vectors to train deep reinforcement
learning (RL) policies. We propose a framework that combines RL with graph
neural networks and study the benefits and limitations of graph-based policy in
the VVC setting. Our results show that graph-based policies converge to the
same rewards asymptotically however at a slower rate when compared to vector
representation counterpart. We conduct further analysis on the impact of both
observations and actions: on the observation end, we examine the robustness of
graph-based policy on two typical data acquisition errors in power systems,
namely sensor communication failure and measurement misalignment. On the action
end, we show that actuators have various impacts on the system, thus using a
graph representation induced by power systems topology may not be the optimal
choice. In the end, we conduct a case study to demonstrate that the choice of
readout function architecture and graph augmentation can further improve
training performance and robustness.

    

### [[2109.12075] Towards A Measure Of General Machine Intelligence](http://arxiv.org/abs/2109.12075)


  To build increasingly general-purpose artificial intelligence systems that
can deal with unknown variables across unknown domains, we need benchmarks that
measure precisely how well these systems perform on tasks they have never seen
before. A prerequisite for this is a measure of a task's generalization
difficulty, or how dissimilar it is from the system's prior knowledge and
experience. If the skill of an intelligence system in a particular domain is
defined as it's ability to consistently generate a set of instructions (or
programs) to solve tasks in that domain, current benchmarks do not
quantitatively measure the efficiency of acquiring new skills, making it
possible to brute-force skill acquisition by training with unlimited amounts of
data and compute power. With this in mind, we first propose a common language
of instruction, i.e. a programming language that allows the expression of
programs in the form of directed acyclic graphs across a wide variety of
real-world domains and computing platforms. Using programs generated in this
language, we demonstrate a match-based method to both score performance and
calculate the generalization difficulty of any given set of tasks. We use these
to define a numeric benchmark called the g-index to measure and compare the
skill-acquisition efficiency of any intelligence system on a set of real-world
tasks. Finally, we evaluate the suitability of some well-known models as
general intelligence systems by calculating their g-index scores.

    

### [[2109.12077] The Mirror Langevin Algorithm Converges with Vanishing Bias](http://arxiv.org/abs/2109.12077)


  The technique of modifying the geometry of a problem from Euclidean to
Hessian metric has proved to be quite effective in optimization, and has been
the subject of study for sampling. The Mirror Langevin Diffusion (MLD) is a
sampling analogue of mirror flow in continuous time, and it has nice
convergence properties under log-Sobolev or Poincare inequalities relative to
the Hessian metric, as shown by Chewi et al. (2020). In discrete time, a simple
discretization of MLD is the Mirror Langevin Algorithm (MLA) studied by Zhang
et al. (2020), who showed a biased convergence bound with a non-vanishing bias
term (does not go to zero as step size goes to zero). This raised the question
of whether we need a better analysis or a better discretization to achieve a
vanishing bias. Here we study the basic Mirror Langevin Algorithm and show it
indeed has a vanishing bias. We apply mean-square analysis based on Li et al.
(2019) and Li et al. (2021) to show the mixing time bound for MLA under the
modified self-concordance condition introduced by Zhang et al. (2020).

    

### [[2109.12081] Deep Social Force](http://arxiv.org/abs/2109.12081)


  The Social Force model introduced by Helbing and Molnar in 1995 is a
cornerstone of pedestrian simulation. This paper introduces a differentiable
simulation of the Social Force model where the assumptions on the shapes of
interaction potentials are relaxed with the use of universal function
approximators in the form of neural networks. Classical force-based pedestrian
simulations suffer from unnatural locking behavior on head-on collision paths.
In addition, they cannot model the bias of pedestrians to avoid each other on
the right or left depending on the geographic region. My experiments with more
general interaction potentials show that potentials with a sharp tip in the
front avoid locking. In addition, asymmetric interaction potentials lead to a
left or right bias when pedestrians avoid each other.

    

### [[2109.12094] A spatiotemporal machine learning approach to forecasting COVID-19 incidence at the county level in the United States](http://arxiv.org/abs/2109.12094)


  With COVID-19 affecting every country globally and changing everyday life,
the ability to forecast the spread of the disease is more important than any
previous epidemic. The conventional methods of disease-spread modeling,
compartmental models, are based on the assumption of spatiotemporal homogeneity
of the spread of the virus, which may cause forecasting to underperform,
especially at high spatial resolutions. In this paper we approach the
forecasting task with an alternative technique -- spatiotemporal machine
learning. We present COVID-LSTM, a data-driven model based on a Long Short-term
Memory deep learning architecture for forecasting COVID-19 incidence at the
county-level in the US. We use the weekly number of new positive cases as
temporal input, and hand-engineered spatial features from Facebook movement and
connectedness datasets to capture the spread of the disease in time and space.
COVID-LSTM outperforms the COVID-19 Forecast Hub's Ensemble model
(COVIDhub-ensemble) on our 17-week evaluation period, making it the first model
to be more accurate than the COVIDhub-ensemble over one or more forecast
periods. Over the 4-week forecast horizon, our model is on average 50 cases per
county more accurate than the COVIDhub-ensemble. We highlight that the
underutilization of data-driven forecasting of disease spread prior to COVID-19
is likely due to the lack of sufficient data available for previous diseases,
in addition to the recent advances in machine learning methods for
spatiotemporal forecasting. We discuss the impediments to the wider uptake of
data-driven forecasting, and whether it is likely that more deep learning-based
models will be used in the future.

    

### [[2109.12098] CLIPort: What and Where Pathways for Robotic Manipulation](http://arxiv.org/abs/2109.12098)


  How can we imbue robots with the ability to manipulate objects precisely but
also to reason about them in terms of abstract concepts? Recent works in
manipulation have shown that end-to-end networks can learn dexterous skills
that require precise spatial reasoning, but these methods often fail to
generalize to new goals or quickly learn transferable concepts across tasks. In
parallel, there has been great progress in learning generalizable semantic
representations for vision and language by training on large-scale internet
data, however these representations lack the spatial understanding necessary
for fine-grained manipulation. To this end, we propose a framework that
combines the best of both worlds: a two-stream architecture with semantic and
spatial pathways for vision-based manipulation. Specifically, we present
CLIPort, a language-conditioned imitation-learning agent that combines the
broad semantic understanding (what) of CLIP [1] with the spatial precision
(where) of Transporter [2]. Our end-to-end framework is capable of solving a
variety of language-specified tabletop tasks from packing unseen objects to
folding cloths, all without any explicit representations of object poses,
instance segmentations, memory, symbolic states, or syntactic structures.
Experiments in simulated and real-world settings show that our approach is data
efficient in few-shot settings and generalizes effectively to seen and unseen
semantic concepts. We even learn one multi-task policy for 10 simulated and 9
real-world tasks that is better or comparable to single-task policies.

    

### [[2109.12100] MLIMC: Machine learning-based implicit-solvent Monte Carlo](http://arxiv.org/abs/2109.12100)


  Monte Carlo (MC) methods are important computational tools for molecular
structure optimizations and predictions. When solvent effects are explicitly
considered, MC methods become very expensive due to the large degree of freedom
associated with the water molecules and mobile ions. Alternatively
implicit-solvent MC can largely reduce the computational cost by applying a
mean field approximation to solvent effects and meanwhile maintains the atomic
detail of the target molecule. The two most popular implicit-solvent models are
the Poisson-Boltzmann (PB) model and the Generalized Born (GB) model in a way
such that the GB model is an approximation to the PB model but is much faster
in simulation time. In this work, we develop a machine learning-based
implicit-solvent Monte Carlo (MLIMC) method by combining the advantages of both
implicit solvent models in accuracy and efficiency. Specifically, the MLIMC
method uses a fast and accurate PB-based machine learning (PBML) scheme to
compute the electrostatic solvation free energy at each step. We validate our
MLIMC method by using a benzene-water system and a protein-water system. We
show that the proposed MLIMC method has great advantages in speed and accuracy
for molecular structure optimization and prediction.

    

### [[2109.12104] GERNERMED -- An Open German Medical NER Model](http://arxiv.org/abs/2109.12104)


  The current state of adoption of well-structured electronic health records
and integration of digital methods for storing medical patient data in
structured formats can often considered as inferior compared to the use of
traditional, unstructured text based patient data documentation. Data mining in
the field of medical data analysis often needs to rely solely on processing of
unstructured data to retrieve relevant data. In natural language processing
(NLP), statistical models have been shown successful in various tasks like
part-of-speech tagging, relation extraction (RE) and named entity recognition
(NER). In this work, we present GERNERMED, the first open, neural NLP model for
NER tasks dedicated to detect medical entity types in German text data. Here,
we avoid the conflicting goals of protection of sensitive patient data from
training data extraction and the publication of the statistical model weights
by training our model on a custom dataset that was translated from publicly
available datasets in foreign language by a pretrained neural machine
translation model. The sample code and the statistical model is available at:
this https URL


### [[1811.11881] Adversarial Bandits with Knapsacks](http://arxiv.org/abs/1811.11881)


  We consider Bandits with Knapsacks (henceforth, BwK), a general model for
multi-armed bandits under supply/budget constraints. In particular, a bandit
algorithm needs to solve a well-known knapsack problem: find an optimal packing
of items into a limited-size knapsack. The BwK problem is a common
generalization of numerous motivating examples, which range from dynamic
pricing to repeated auctions to dynamic ad allocation to network routing and
scheduling. While the prior work on BwK focused on the stochastic version, we
pioneer the other extreme in which the outcomes can be chosen adversarially.
This is a considerably harder problem, compared to both the stochastic version
and the "classic" adversarial bandits, in that regret minimization is no longer
feasible. Instead, the objective is to minimize the competitive ratio: the
ratio of the benchmark reward to the algorithm's reward.
We design an algorithm with competitive ratio O(log T) relative to the best
fixed distribution over actions, where T is the time horizon; we also prove a
matching lower bound. The key conceptual contribution is a new perspective on
the stochastic version of the problem. We suggest a new algorithm for the
stochastic version, which builds on the framework of regret minimization in
repeated games and admits a substantially simpler analysis compared to prior
work. We then analyze this algorithm for the adversarial version and use it as
a subroutine to solve the latter.

    

### [[1907.00787] CNN-based synthesis of realistic high-resolution LiDAR data](http://arxiv.org/abs/1907.00787)


  This paper presents a novel CNN-based approach for synthesizing
high-resolution LiDAR point cloud data. Our approach generates semantically and
perceptually realistic results with guidance from specialized loss-functions.
First, we utilize a modified per-point loss that addresses missing LiDAR point
measurements. Second, we align the quality of our generated output with
real-world sensor data by applying a perceptual loss. In large-scale
experiments on real-world datasets, we evaluate both the geometric accuracy and
semantic segmentation performance using our generated data vs. ground truth. In
a mean opinion score testing we further assess the perceptual quality of our
generated point clouds. Our results demonstrate a significant quantitative and
qualitative improvement in both geometry and semantics over traditional non
CNN-based up-sampling methods.

    

### [[1910.01523] ReNAS:Relativistic Evaluation of Neural Architecture Search](http://arxiv.org/abs/1910.01523)


  An effective and efficient architecture performance evaluation scheme is
essential for the success of Neural Architecture Search (NAS). To save
computational cost, most of existing NAS algorithms often train and evaluate
intermediate neural architectures on a small proxy dataset with limited
training epochs. But it is difficult to expect an accurate performance
estimation of an architecture in such a coarse evaluation way. This paper
advocates a new neural architecture evaluation scheme, which aims to determine
which architecture would perform better instead of accurately predict the
absolute architecture performance. Therefore, we propose a
\textbf{relativistic} architecture performance predictor in NAS (ReNAS). We
encode neural architectures into feature tensors, and further refining the
representations with the predictor. The proposed relativistic performance
predictor can be deployed in discrete searching methods to search for the
desired architectures without additional evaluation. Experimental results on
NAS-Bench-101 dataset suggests that, sampling 424 ($0.1\%$ of the entire search
space) neural architectures and their corresponding validation performance is
already enough for learning an accurate architecture performance predictor. The
accuracies of our searched neural architectures on NAS-Bench-101 and
NAS-Bench-201 datasets are higher than that of the state-of-the-art methods and
show the priority of the proposed method.

    

### [[1910.02497] mfEGRA: Multifidelity Efficient Global Reliability Analysis through Active Learning for Failure Boundary Location](http://arxiv.org/abs/1910.02497)


  This paper develops mfEGRA, a multifidelity active learning method using
data-driven adaptively refined surrogates for failure boundary location in
reliability analysis. This work addresses the issue of prohibitive cost of
reliability analysis using Monte Carlo sampling for expensive-to-evaluate
high-fidelity models by using cheaper-to-evaluate approximations of the
high-fidelity model. The method builds on the Efficient Global Reliability
Analysis (EGRA) method, which is a surrogate-based method that uses adaptive
sampling for refining Gaussian process surrogates for failure boundary location
using a single-fidelity model. Our method introduces a two-stage adaptive
sampling criterion that uses a multifidelity Gaussian process surrogate to
leverage multiple information sources with different fidelities. The method
combines expected feasibility criterion from EGRA with one-step lookahead
information gain to refine the surrogate around the failure boundary. The
computational savings from mfEGRA depends on the discrepancy between the
different models, and the relative cost of evaluating the different models as
compared to the high-fidelity model. We show that accurate estimation of
reliability using mfEGRA leads to computational savings of $\sim$46% for an
analytic multimodal test problem and 24% for a three-dimensional acoustic horn
problem, when compared to single-fidelity EGRA. We also show the effect of
using a priori drawn Monte Carlo samples in the implementation for the acoustic
horn problem, where mfEGRA leads to computational savings of 45% for the
three-dimensional case and 48% for a rarer event four-dimensional case as
compared to single-fidelity EGRA.

    

### [[2001.00378] Deep Representation Learning in Speech Processing: Challenges, Recent Advances, and Future Trends](http://arxiv.org/abs/2001.00378)


  Research on speech processing has traditionally considered the task of
designing hand-engineered acoustic features (feature engineering) as a separate
distinct problem from the task of designing efficient machine learning (ML)
models to make prediction and classification decisions. There are two main
drawbacks to this approach: firstly, the feature engineering being manual is
cumbersome and requires human knowledge; and secondly, the designed features
might not be best for the objective at hand. This has motivated the adoption of
a recent trend in speech community towards utilisation of representation
learning techniques, which can learn an intermediate representation of the
input signal automatically that better suits the task at hand and hence lead to
improved performance. The significance of representation learning has increased
with advances in deep learning (DL), where the representations are more useful
and less dependent on human knowledge, making it very conducive for tasks like
classification, prediction, etc. The main contribution of this paper is to
present an up-to-date and comprehensive survey on different techniques of
speech representation learning by bringing together the scattered research
across three distinct research areas including Automatic Speech Recognition
(ASR), Speaker Recognition (SR), and Speaker Emotion Recognition (SER). Recent
reviews in speech have been conducted for ASR, SR, and SER, however, none of
these has focused on the representation learning from speech -- a gap that our
survey aims to bridge.

    

### [[2002.06442] Monotonic Cardinality Estimation of Similarity Selection: A Deep Learning Approach](http://arxiv.org/abs/2002.06442)


  Due to the outstanding capability of capturing underlying data distributions,
deep learning techniques have been recently utilized for a series of
traditional database problems. In this paper, we investigate the possibilities
of utilizing deep learning for cardinality estimation of similarity selection.
Answering this problem accurately and efficiently is essential to many data
management applications, especially for query optimization. Moreover, in some
applications the estimated cardinality is supposed to be consistent and
interpretable. Hence a monotonic estimation w.r.t. the query threshold is
preferred. We propose a novel and generic method that can be applied to any
data type and distance function. Our method consists of a feature extraction
model and a regression model. The feature extraction model transforms original
data and threshold to a Hamming space, in which a deep learning-based
regression model is utilized to exploit the incremental property of cardinality
w.r.t. the threshold for both accuracy and monotonicity. We develop a training
strategy tailored to our model as well as techniques for fast estimation. We
also discuss how to handle updates. We demonstrate the accuracy and the
efficiency of our method through experiments, and show how it improves the
performance of a query optimizer.

    

### [[2003.08854] Goal-Conditioned End-to-End Visuomotor Control for Versatile Skill Primitives](http://arxiv.org/abs/2003.08854)


  Visuomotor control (VMC) is an effective means of achieving basic
manipulation tasks such as pushing or pick-and-place from raw images.
Conditioning VMC on desired goal states is a promising way of achieving
versatile skill primitives. However, common conditioning schemes either rely on
task-specific fine tuning - e.g. using one-shot imitation learning (IL) - or on
sampling approaches using a forward model of scene dynamics i.e.
model-predictive control (MPC), leaving deployability and planning horizon
severely limited. In this paper we propose a conditioning scheme which avoids
these pitfalls by learning the controller and its conditioning in an end-to-end
manner. Our model predicts complex action sequences based directly on a dynamic
image representation of the robot motion and the distance to a given target
observation. In contrast to related works, this enables our approach to
efficiently perform complex manipulation tasks from raw image observations
without predefined control primitives or test time demonstrations. We report
significant improvements in task success over representative MPC and IL
baselines. We also demonstrate our model's generalisation capabilities in
challenging, unseen tasks featuring visual noise, cluttered scenes and unseen
object geometries.

    

### [[2004.11803] Scan-based Semantic Segmentation of LiDAR Point Clouds: An Experimental Study](http://arxiv.org/abs/2004.11803)


  Autonomous vehicles need to have a semantic understanding of the
three-dimensional world around them in order to reason about their environment.
State of the art methods use deep neural networks to predict semantic classes
for each point in a LiDAR scan. A powerful and efficient way to process LiDAR
measurements is to use two-dimensional, image-like projections. In this work,
we perform a comprehensive experimental study of image-based semantic
segmentation architectures for LiDAR point clouds. We demonstrate various
techniques to boost the performance and to improve runtime as well as memory
constraints.
First, we examine the effect of network size and suggest that much faster
inference times can be achieved at a very low cost to accuracy. Next, we
introduce an improved point cloud projection technique that does not suffer
from systematic occlusions. We use a cyclic padding mechanism that provides
context at the horizontal field-of-view boundaries. In a third part, we perform
experiments with a soft Dice loss function that directly optimizes for the
intersection-over-union metric. Finally, we propose a new kind of convolution
layer with a reduced amount of weight-sharing along one of the two spatial
dimensions, addressing the large difference in appearance along the vertical
axis of a LiDAR scan. We propose a final set of the above methods with which
the model achieves an increase of 3.2% in mIoU segmentation performance over
the baseline while requiring only 42% of the original inference time.

    

### [[2006.03950] ValNorm Quantifies Semantics to Reveal Consistent Valence Biases Across Languages and Over Centuries](http://arxiv.org/abs/2006.03950)


  Word embeddings learn implicit biases from linguistic regularities captured
by word co-occurrence statistics. By extending methods that quantify human-like
biases in word embeddings, we introduceValNorm, a novel intrinsic evaluation
task and method to quantify the valence dimension of affect in human-rated word
sets from social psychology. We apply ValNorm on static word embeddings from
seven languages (Chinese, English, German, Polish, Portuguese, Spanish, and
Turkish) and from historical English text spanning 200 years. ValNorm achieves
consistently high accuracy in quantifying the valence of non-discriminatory,
non-social group word sets. Specifically, ValNorm achieves a Pearson
correlation of r=0.88 for human judgment scores of valence for 399 words
collected to establish pleasantness norms in English. In contrast, we measure
gender stereotypes using the same set of word embeddings and find that social
biases vary across languages. Our results indicate that valence associations of
non-discriminatory, non-social group words represent widely-shared
associations, in seven languages and over 200 years.

    

### [[2007.14384] Noise-Induced Barren Plateaus in Variational Quantum Algorithms](http://arxiv.org/abs/2007.14384)


  Variational Quantum Algorithms (VQAs) may be a path to quantum advantage on
Noisy Intermediate-Scale Quantum (NISQ) computers. A natural question is
whether noise on NISQ devices places fundamental limitations on VQA
performance. We rigorously prove a serious limitation for noisy VQAs, in that
the noise causes the training landscape to have a barren plateau (i.e.,
vanishing gradient). Specifically, for the local Pauli noise considered, we
prove that the gradient vanishes exponentially in the number of qubits $n$ if
the depth of the ansatz grows linearly with $n$. These noise-induced barren
plateaus (NIBPs) are conceptually different from noise-free barren plateaus,
which are linked to random parameter initialization. Our result is formulated
for a generic ansatz that includes as special cases the Quantum Alternating
Operator Ansatz and the Unitary Coupled Cluster Ansatz, among others. For the
former, our numerical heuristics demonstrate the NIBP phenomenon for a
realistic hardware noise model.

    

### [[2009.05277] AFP-SRC: Identification of Antifreeze Proteins Using Sparse Representation Classifier](http://arxiv.org/abs/2009.05277)


  Species living in the extreme cold environment fight against the harsh
conditions using antifreeze proteins (AFPs), that manipulates the freezing
mechanism of water in more than one way. This amazing nature of AFP turns out
to be extremely useful in several industrial and medical applications. The lack
of similarity in their structure and sequence makes their prediction an arduous
task and identifying them experimentally in the wet-lab is time-consuming and
expensive. In this research, we propose a computational framework for the
prediction of AFPs which is essentially based on a sample-specific
classification method using the sparse reconstruction. A linear model and an
over-complete dictionary matrix of known AFPs are used to predict a sparse
class-label vector that provides a sample-association score. Delta-rule is
applied for the reconstruction of two pseudo-samples using lower and upper
parts of the sample-association vector and based on the minimum recovery score,
class labels are assigned. We compare our approach with contemporary methods on
a standard dataset and the proposed method is found to outperform in terms of
Balanced accuracy and Youden's index. The MATLAB implementation of the proposed
method is available at the author's GitHub page
(\{this https URL}{this https URL}).

    

### [[2009.05474] A black-box adversarial attack for poisoning clustering](http://arxiv.org/abs/2009.05474)


  Clustering algorithms play a fundamental role as tools in decision-making and
sensible automation processes. Due to the widespread use of these applications,
a robustness analysis of this family of algorithms against adversarial noise
has become imperative. To the best of our knowledge, however, only a few works
have currently addressed this problem. In an attempt to fill this gap, in this
work, we propose a black-box adversarial attack for crafting adversarial
samples to test the robustness of clustering algorithms. We formulate the
problem as a constrained minimization program, general in its structure and
customizable by the attacker according to her capability constraints. We do not
assume any information about the internal structure of the victim clustering
algorithm, and we allow the attacker to query it as a service only. In the
absence of any derivative information, we perform the optimization with a
custom approach inspired by the Abstract Genetic Algorithm (AGA). In the
experimental part, we demonstrate the sensibility of different single and
ensemble clustering algorithms against our crafted adversarial samples on
different scenarios. Furthermore, we perform a comparison of our algorithm with
a state-of-the-art approach showing that we are able to reach or even
outperform its performance. Finally, to highlight the general nature of the
generated noise, we show that our attacks are transferable even against
supervised algorithms such as SVMs, random forests, and neural networks.

    

### [[2009.06520] A Systematic Literature Review on the Use of Deep Learning in Software Engineering Research](http://arxiv.org/abs/2009.06520)


  An increasingly popular set of techniques adopted by software engineering
(SE) researchers to automate development tasks are those rooted in the concept
of Deep Learning (DL). The popularity of such techniques largely stems from
their automated feature engineering capabilities, which aid in modeling
software artifacts. However, due to the rapid pace at which DL techniques have
been adopted, it is difficult to distill the current successes, failures, and
opportunities of the current research landscape. In an effort to bring clarity
to this crosscutting area of work, from its modern inception to the present,
this paper presents a systematic literature review of research at the
intersection of SE & DL. The review canvases work appearing in the most
prominent SE and DL conferences and journals and spans 128 papers across 23
unique SE tasks. We center our analysis around the components of learning, a
set of principles that govern the application of machine learning techniques
(ML) to a given problem domain, discussing several aspects of the surveyed work
at a granular level. The end result of our analysis is a research roadmap that
both delineates the foundations of DL techniques applied to SE research, and
highlights likely areas of fertile exploration for the future.

    

### [[2010.09990] The Open Catalyst 2020 (OC20) Dataset and Community Challenges](http://arxiv.org/abs/2010.09990)


  Catalyst discovery and optimization is key to solving many societal and
energy challenges including solar fuels synthesis, long-term energy storage,
and renewable fertilizer production. Despite considerable effort by the
catalysis community to apply machine learning models to the computational
catalyst discovery process, it remains an open challenge to build models that
can generalize across both elemental compositions of surfaces and adsorbate
identity/configurations, perhaps because datasets have been smaller in
catalysis than related fields. To address this we developed the OC20 dataset,
consisting of 1,281,040 Density Functional Theory (DFT) relaxations
(~264,890,000 single point evaluations) across a wide swath of materials,
surfaces, and adsorbates (nitrogen, carbon, and oxygen chemistries). We
supplemented this dataset with randomly perturbed structures, short timescale
molecular dynamics, and electronic structure analyses. The dataset comprises
three central tasks indicative of day-to-day catalyst modeling and comes with
pre-defined train/validation/test splits to facilitate direct comparisons with
future model development efforts. We applied three state-of-the-art graph
neural network models (CGCNN, SchNet, Dimenet++) to each of these tasks as
baseline demonstrations for the community to build on. In almost every task, no
upper limit on model size was identified, suggesting that even larger models
are likely to improve on initial results. The dataset and baseline models are
both provided as open resources, as well as a public leader board to encourage
community contributions to solve these important tasks.

    

### [[2011.12690] DeepKoCo: Efficient latent planning with a task-relevant Koopman representation](http://arxiv.org/abs/2011.12690)


  This paper presents DeepKoCo, a novel model-based agent that learns a latent
Koopman representation from images. This representation allows DeepKoCo to plan
efficiently using linear control methods, such as linear model predictive
control. Compared to traditional agents, DeepKoCo learns task-relevant
dynamics, thanks to the use of a tailored lossy autoencoder network that allows
DeepKoCo to learn latent dynamics that reconstruct and predict only observed
costs, rather than all observed dynamics. As our results show, DeepKoCo
achieves similar final performance as traditional model-free methods on complex
control tasks while being considerably more robust to distractor dynamics,
making the proposed agent more amenable for real-life applications.

    

### [[2012.03826] An Empirical Study of Assumptions in Bayesian Optimisation](http://arxiv.org/abs/2012.03826)


  In this work we rigorously analyse assumptions inherent to black-box
optimisation hyper-parameter tuning tasks. Our results on the Bayesmark
benchmark indicate that heteroscedasticity and non-stationarity pose
significant challenges for black-box optimisers. Based on these findings, we
propose a Heteroscedastic and Evolutionary Bayesian Optimisation solver (HEBO).
HEBO performs non-linear input and output warping, admits exact marginal
log-likelihood optimisation and is robust to the values of lea\rned parameters.
We demonstrate HEBO's empirical efficacy on the NeurIPS 2020 Black-Box
Optimisation challenge, where HEBO placed first. Upon further analysis, we
observe that HEBO significantly outperforms existing black-box optimisers on
108 machine learning hyperparameter tuning tasks comprising the Bayesmark
benchmark. Our findings indicate that the majority of hyper-parameter tuning
tasks exhibit heteroscedasticity and non-stationarity, multi-objective
acquisition ensembles with Pareto front solutions improve queried
configurations, and robust acquisition maximisers afford empirical advantages
relative to their non-robust counterparts. We hope these findings may serve as
guiding principles for practitioners of Bayesian optimisation. All code is made
available at this https URL.

    

### [[2012.05973] Clustering multivariate functional data using unsupervised binary trees](http://arxiv.org/abs/2012.05973)


  We propose a model-based clustering algorithm for a general class of
functional data for which the components could be curves or images. The random
functional data realizations could be measured with error at discrete, and
possibly random, points in the definition domain. The idea is to build a set of
binary trees by recursive splitting of the observations. The number of groups
are determined in a data-driven way. The new algorithm provides easily
interpretable results and fast predictions for online data sets. Results on
simulated datasets reveal good performance in various complex settings. The
methodology is applied to the analysis of vehicle trajectories on a German
roundabout.

    

### [[2101.01494] Weight-of-evidence 2.0 with shrinkage and spline-binning](http://arxiv.org/abs/2101.01494)


  In many practical applications, such as fraud detection, credit risk modeling
or medical decision making, classification models for assigning instances to a
predefined set of classes are required to be both precise as well as
interpretable. Linear modeling methods such as logistic regression are often
adopted, since they offer an acceptable balance between precision and
interpretability. Linear methods, however, are not well equipped to handle
categorical predictors with high-cardinality or to exploit non-linear relations
in the data. As a solution, data preprocessing methods such as
weight-of-evidence are typically used for transforming the predictors. The
binning procedure that underlies the weight-of-evidence approach, however, has
been little researched and typically relies on ad-hoc or expert driven
procedures. The objective in this paper, therefore, is to propose a formalized,
data-driven and powerful method.
To this end, we explore the discretization of continuous variables through
the binning of spline functions, which allows for capturing non-linear effects
in the predictor variables and yields highly interpretable predictors taking
only a small number of discrete values. Moreover, we extend upon the
weight-of-evidence approach and propose to estimate the proportions using
shrinkage estimators. Together, this offers an improved ability to exploit both
non-linear and categorical predictors for achieving increased classification
precision, while maintaining interpretability of the resulting model and
decreasing the risk of overfitting.
We present the results of a series of experiments in a fraud detection
setting, which illustrate the effectiveness of the presented approach. We
facilitate reproduction of the presented results and adoption of the proposed
approaches by providing both the dataset and the code for implementing the
experiments and the presented approach.

    

### [[2101.11055] LDLE: Low Distortion Local Eigenmaps](http://arxiv.org/abs/2101.11055)


  We present Low Distortion Local Eigenmaps (LDLE), a manifold learning
technique which constructs a set of low distortion local views of a dataset in
lower dimension and registers them to obtain a global embedding. The local
views are constructed using the global eigenvectors of the graph Laplacian and
are registered using Procrustes analysis. The choice of these eigenvectors may
vary across the regions. In contrast to existing techniques, LDLE can embed
closed and non-orientable manifolds into their intrinsic dimension by tearing
them apart. It also provides gluing instruction on the boundary of the torn
embedding to help identify the topology of the original manifold. Our
experimental results will show that LDLE largely preserved distances up to a
constant scale while other techniques produced higher distortion. We also
demonstrate that LDLE produces high quality embeddings even when the data is
noisy or sparse.

    

### [[2102.13653] On the Generalization of Stochastic Gradient Descent with Momentum](http://arxiv.org/abs/2102.13653)


  While momentum-based methods, in conjunction with stochastic gradient descent
(SGD), are widely used when training machine learning models, there is little
theoretical understanding on the generalization error of such methods. In this
work, we first show that there exists a convex loss function for which
algorithmic stability fails to establish generalization guarantees when SGD
with standard heavy-ball momentum (SGDM) is run for multiple epochs. Then, for
smooth Lipschitz loss functions, we analyze a modified momentum-based update
rule, i.e., SGD with early momentum (SGDEM), and show that it admits an
upper-bound on the generalization error. Thus, our results show that machine
learning models can be trained for multiple epochs of SGDEM with a guarantee
for generalization. Finally, for the special case of strongly convex loss
functions, we find a range of momentum such that multiple epochs of standard
SGDM, as a special form of SGDEM, also generalizes. Extending our results on
generalization, we also develop an upper-bound on the expected true risk, in
terms of the number of training steps, the size of the training set, and the
momentum parameter. Experimental evaluations verify the consistency between the
numerical results and our theoretical bounds and the effectiveness of SGDEM for
smooth Lipschitz loss functions.

    

### [[2103.03561] Nishimori meets Bethe: a spectral method for node classification in sparse weighted graphs](http://arxiv.org/abs/2103.03561)


  This article unveils a new relation between the Nishimori temperature
parametrizing a distribution P and the Bethe free energy on random Erdos-Renyi
graphs with edge weights distributed according to P. Estimating the Nishimori
temperature being a task of major importance in Bayesian inference problems, as
a practical corollary of this new relation, a numerical method is proposed to
accurately estimate the Nishimori temperature from the eigenvalues of the Bethe
Hessian matrix of the weighted graph. The algorithm, in turn, is used to
propose a new spectral method for node classification in weighted (possibly
sparse) graphs. The superiority of the method over competing state-of-the-art
approaches is demonstrated both through theoretical arguments and real-world
data experiments.

    

### [[2103.13751] Data Generation in Low Sample Size Setting Using Manifold Sampling and a Geometry-Aware VAE](http://arxiv.org/abs/2103.13751)


  We propose a new efficient way to sample from a Variational Autoencoder in
the challenging low sample size setting. This method reveals particularly well
suited to perform data augmentation in such a low data regime and is validated
across various standard and real-life data sets. In particular, this scheme
allows to greatly improve classification results on the OASIS database where
balanced accuracy jumps from 80.7% for a classifier trained with the raw data
to 88.6% when trained only with the synthetic data generated by our method.
Such results were also observed on 3 standard data sets and with other
classifiers. A code is available at
this https URL.

    

### [[2104.01747] Fast Design Space Exploration of Nonlinear Systems: Part I](http://arxiv.org/abs/2104.01747)


  System design tools are often only available as input-output blackboxes: for
a given design as input they compute an output representing system behavior.
Blackboxes are intended to be run in the forward direction. This paper presents
a new method of solving the inverse design problem namely, given requirements
or constraints on output, find an input that also optimizes an objective
function. This problem is challenging for several reasons. First, blackboxes
are not designed to be run in reverse. Second, inputs and outputs can be
discrete and continuous. Third, finding designs concurrently satisfying a set
of requirements is hard because designs satisfying individual requirements may
conflict with each other. Fourth, blackbox evaluations can be expensive.
Finally, blackboxes can sometimes fail to produce an output. This paper
presents CNMA, a new method of solving the inverse problem that overcomes these
challenges. CNMA tries to sample only the part of the design space relevant to
solving the problem, leveraging the power of neural networks, Mixed Integer
Linear Programs, and a new learning-from-failure feedback loop. The paper also
presents a parallel version of CNMA that improves the efficiency and quality of
solutions over the sequential version, and tries to steer it away from local
optima. CNMA's performance is evaluated against conventional optimization
methods for seven nonlinear design problems of 8 (two problems), 10, 15, 36 and
60 real-valued dimensions and one with 186 binary dimensions. Conventional
methods evaluated are off-the-shelf implementations of Bayesian Optimization
with Gaussian Processes, Nelder Mead and Random Search. The first two do not
solve problems that are high-dimensional, have discrete and continuous
variables or whose blackboxes can fail to return values. CNMA solves all
problems, and surpasses the performance of conventional methods by 1%-87%.

    

### [[2104.12384] Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations](http://arxiv.org/abs/2104.12384)


  We present a framework that allows for the non-asymptotic study of the
$2$-Wasserstein distance between the invariant distribution of an ergodic
stochastic differential equation and the distribution of its numerical
approximation in the strongly log-concave case. This allows us to study in a
unified way a number of different integrators proposed in the literature for
the overdamped and underdamped Langevin dynamics. In addition, we analyse a
novel splitting method for the underdamped Langevin dynamics which only
requires one gradient evaluation per time step. Under an additional smoothness
assumption on a $d$--dimensional strongly log-concave distribution with
condition number $\kappa$, the algorithm is shown to produce with an
$\mathcal{O}\big(\kappa^{5/4} d^{1/4}\epsilon^{-1/2} \big)$ complexity samples
from a distribution that, in Wasserstein distance, is at most $\epsilon>0$ away
from the target distribution.

    

### [[2106.11851] Stochastic Polyak Stepsize with a Moving Target](http://arxiv.org/abs/2106.11851)


  We propose a new stochastic gradient method called MOTAPS (Moving Targetted
Polyak Stepsize) that uses recorded past loss values to compute adaptive
stepsizes. MOTAPS can be seen as a variant of the Stochastic Polyak (SP) which
is also a method that also uses loss values to adjust the stepsize. The
downside to the SP method is that it only converges when the interpolation
condition holds. MOTAPS is an extension of SP that does not rely on the
interpolation condition. The MOTAPS method uses $n$ auxiliary variables, one
for each data point, that track the loss value for each data point. We provide
a global convergence theory for SP, an intermediary method TAPS, and MOTAPS by
showing that they all can be interpreted as a special variant of online SGD. We
also perform several numerical experiments on convex learning problems, and
deep learning models for image classification and language translation. In all
of our tasks we show that MOTAPS is competitive with the relevant baseline
method.

    

### [[2106.12782] Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control](http://arxiv.org/abs/2106.12782)


  Accurate models of robot dynamics are critical for safe and stable control
and generalization to novel operational conditions. Hand-designed models,
however, may be insufficiently accurate, even after careful parameter tuning.
This motivates the use of machine learning techniques to approximate the robot
dynamics over a training set of state-control trajectories. The dynamics of
many robots, including ground, aerial, and underwater vehicles, are described
in terms of their SE(3) pose and generalized velocity, and satisfy conservation
of energy principles. This paper proposes a Hamiltonian formulation over the
SE(3) manifold of the structure of a neural ordinary differential equation
(ODE) network to approximate the dynamics of a rigid body. In contrast to a
black-box ODE network, our formulation guarantees total energy conservation by
construction. We develop energy shaping and damping injection control for the
learned, potentially under-actuated SE(3) Hamiltonian dynamics to enable a
unified approach for stabilization and trajectory tracking with various
platforms, including pendulum, rigid-body, and quadrotor systems.

    

### [[2109.10960] Quantitative analysis of phase transitions in two-dimensional XY models using persistent homology](http://arxiv.org/abs/2109.10960)


  We use persistent homology and persistence images as an observable of three
different variants of the two-dimensional XY model in order to identify and
study their phase transitions. We examine models with the classical XY action,
a topological lattice action, and an action with an additional nematic term. In
particular, we introduce a new way of computing the persistent homology of
lattice spin model configurations and, by considering the fluctuations in the
output of logistic regression and k-nearest neighbours models trained on
persistence images, we develop a methodology to extract estimates of the
critical temperature and the critical exponent of the correlation length. We
put particular emphasis on finite-size scaling behaviour and producing
estimates with quantifiable error. For each model we successfully identify its
phase transition(s) and are able to get an accurate determination of the
critical temperatures and critical exponents of the correlation length.

    

### [[2103.07977] Understanding the Design Space of Sparse/Dense Multiphase Dataflows for Mapping Graph Neural Networks on Spatial Accelerators](http://arxiv.org/abs/2103.07977)


  Graph Neural Networks (GNNs) have garnered a lot of recent interest because
of their success in learning representations from graph-structured data across
several critical applications in cloud and HPC. Owing to their unique compute
and memory characteristics that come from an interplay between dense and sparse
phases of computations, the emergence of reconfigurable dataflow (aka spatial)
accelerators offers promise for acceleration by mapping optimized dataflows
(i.e., computation order and parallelism) for both phases. The goal of this
work is to characterize and understand the design-space of dataflow choices for
running GNNs on spatial accelerators in order for the compilers to optimize the
dataflow based on the workload. Specifically, we propose a taxonomy to describe
all possible choices for mapping the dense and sparse phases of GNNs spatially
and temporally over a spatial accelerator, capturing both the intra-phase
dataflow and the inter-phase (pipelined) dataflow. Using this taxonomy, we do
deep-dives into the cost and benefits of several dataflows and perform case
studies on implications of hardware parameters for dataflows and value of
flexibility to support pipelined execution.

    

### [[2109.11709] User-Defined Functions for HDF5](http://arxiv.org/abs/2109.11709)


  Scientific datasets are known for their challenging storage demands and the
associated processing pipelines that transform their information. Some of those
processing tasks include filtering, cleansing, aggregation, normalization, and
data format translation -- all of which generate even more data. In this paper,
we present an infrastructure for the HDF5 file format that enables dataset
values to be populated on the fly: task-related scripts can be attached into
HDF5 files and only execute when the dataset is read by an application. We
provide details on the software architecture that supports user-defined
functions (UDFs) and how it integrates with hardware accelerators and
computational storage. Moreover, we describe the built-in security model that
limits the system resources a UDF can access. Last, we present several use
cases that show how UDFs can be used to extend scientific datasets in ways that
go beyond the original scope of this work.

    

### [[2109.11774] Paving the Way for Distributed ArtificialIntelligence over the Air](http://arxiv.org/abs/2109.11774)


  Distributed Artificial Intelligence (DAI) is regarded as one of the most
promising techniques to provide intelligent services under strict privacy
protection regulations for multiple clients. By applying DAI, training on raw
data is carried out locally, while the trained outputs, e.g., model parameters,
from multiple local clients, are sent back to a central server for aggregation.
Recently, for achieving better practicality, DAI is studied in conjunction with
wireless communication networks, incorporating various random effects brought
by wireless channels. However, because of the complex and case-dependent nature
of wireless channels, a generic simulator for applying DAI in wireless
communication networks is still lacking. To accelerate the development of DAI
applied in wireless communication networks, we propose a generic system design
in this paper as well as an associated simulator that can be set according to
wireless channels and system-level configurations. Details of the system design
and analysis of the impacts of wireless environments are provided to facilitate
further implementations and updates. We employ a series of experiments to
verify the effectiveness and efficiency of the proposed system design and
reveal its superior scalability.

    

### [[2109.11856] The Max-Line-Formation Problem](http://arxiv.org/abs/2109.11856)


  We consider n robots with limited visibility: each robot can observe other
robots only up to a constant distance denoted as the viewing range. The robots
operate in discrete rounds that are either fully synchronous (FSync) or
semi-synchronized (SSync). Most previously studied formation problems in this
setting seek to bring the robots closer together (e.g., Gathering or
Chain-Formation). In this work, we introduce the Max-Line-Formation problem,
which has a contrary goal: to arrange the robots on a straight line of maximal
length. First, we prove that the problem is impossible to solve by robots with
a constant sized circular viewing range. The impossibility holds under
comparably strong assumptions: robots that agree on both axes of their local
coordinate systems in FSync. On the positive side, we show that the problem is
solvable by robots with a constant square viewing range, i.e., the robots can
observe other robots that lie within a constant-sized square centered at their
position. In this case, the robots need to agree on only one axis of their
local coordinate systems. We derive two algorithms: the first algorithm
considers oblivious robots and converges to the optimal configuration in time
$\mathcal{O}(n^2 \cdot \log (n/\varepsilon))$ under the SSync scheduler. The
other algorithm makes use of locally visible lights (LUMI). It is designed for
the FSync scheduler and can solve the problem exactly in optimal time
$\Theta(n)$. Afterward, we show that both the algorithmic and the analysis
techniques can also be applied to the Gathering and Chain-Formation problem: we
introduce an algorithm with a reduced viewing range for Gathering and give new
and improved runtime bounds for the Chain-Formation problem.

    

### [[2109.11987] Formal Verification of a Distributed Dynamic Reconfiguration Protocol](http://arxiv.org/abs/2109.11987)


  We present a formal, machine checked TLA+ safety proof of MongoRaftReconfig,
a distributed dynamic reconfiguration protocol. MongoRaftReconfig was designed
for and implemented in MongoDB, a distributed database whose replication
protocol is derived from the Raft consensus algorithm. We present an inductive
invariant for MongoRaftReconfig that is formalized in TLA+ and formally proved
using the TLA+ proof system (TLAPS). We also present a formal TLAPS proof of
two key safety properties of MongoRaftReconfig, LeaderCompleteness and
StateMachineSafety. To our knowledge, these are the first machine checked
inductive invariant and safety proof of a dynamic reconfiguration protocol for
a Raft based replication system.

    

### [[2109.12060] Extreme Scale Survey Simulation with Python Workflows](http://arxiv.org/abs/2109.12060)


  The Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) will
soon carry out an unprecedented wide, fast, and deep survey of the sky in
multiple optical bands. The data from LSST will open up a new discovery space
in astronomy and cosmology, simultaneously providing clues toward addressing
burning issues of the day, such as the origin of dark energy and and the nature
of dark matter, while at the same time yielding data that will, in turn, pose
fresh new questions. To prepare for the imminent arrival of this remarkable
data set, it is crucial that the associated scientific communities be able to
develop the software needed to analyze it. Computational power now available
allows us to generate synthetic data sets that can be used as a realistic
training ground for such an effort. This effort raises its own challenges --
the need to generate very large simulations of the night sky, scaling up
simulation campaigns to large numbers of compute nodes across multiple
computing centers with different architectures, and optimizing the complex
workload around memory requirements and widely varying wall clock times. We
describe here a large-scale workflow that melds together Python code to steer
the workflow, Parsl to manage the large-scale distributed execution of workflow
components, and containers to carry out the image simulation campaign across
multiple sites. Taking advantage of these tools, we developed an extreme-scale
computational framework and used it to simulate five years of observations for
300 square degrees of sky area. We describe our experiences and lessons learned
in developing this workflow capability, and highlight how the scalability and
portability of our approach enabled us to efficiently execute it on up to 4000
compute nodes on two supercomputers.

    

### [[2012.06065] Coded sparse matrix computation schemes that leverage partial stragglers](http://arxiv.org/abs/2012.06065)


  Distributed matrix computations over large clusters can suffer from the
problem of slow or failed worker nodes (called stragglers) which can dominate
the overall job execution time. Coded computation utilizes concepts from
erasure coding to mitigate the effect of stragglers by running 'coded' copies
of tasks comprising a job; stragglers are typically treated as erasures. While
this is useful, there are issues with applying, e.g., MDS codes in a
straightforward manner. Several practical matrix computation scenarios involve
sparse matrices. MDS codes typically require dense linear combinations of
submatrices of the original matrices which destroy their inherent sparsity.
This is problematic as it results in significantly higher worker computation
times. Moreover, treating slow nodes as erasures ignores the potentially useful
partial computations performed by them. Furthermore, some MDS techniques also
suffer from significant numerical stability issues. In this work we present
schemes that allow us to leverage partial computation by stragglers while
imposing constraints on the level of coding that is required in generating the
encoded submatrices. This significantly reduces the worker computation time as
compared to previous approaches and results in improved numerical stability in
the decoding process. Exhaustive numerical experiments on Amazon Web Services
(AWS) clusters support our findings.

    

### [[2012.12868] A Flat-Combining-Based Persistent Stack for Non-Volatile Memory](http://arxiv.org/abs/2012.12868)


  Flat combining (FC) is a synchronization paradigm in which a single thread,
holding a global lock, collects requests by multiple threads for accessing a
concurrent data structure and applies their combined requests to it. Although
FC is sequential, it significantly reduces synchronization overheads and cache
invalidations and thus often provides better performance than that of lock-free
implementations. The recent emergence of non-volatile memory (NVM) technologies
increases the interest in the development of persistent concurrent objects.
These are objects that are able to recover from system failures and ensure
consistency by retaining their state in NVM and fixing it, if required, upon
recovery. Of particular interest are detectable objects that, in addition to
ensuring consistency, allow recovery code to infer if a failed operation took
effect before the crash and, if it did, obtain its response. In this work, we
present the first FC-based persistent object implementations. Specifically, we
introduce a detectable FC-based implementation of a concurrent LIFO stack, a
concurrent FIFO queue, and a double-ended queue. Our empirical evaluation
establishes that due to flat combining, the novel implementations require a
much smaller number of costly persistence instructions than competing
algorithms and are therefore able to significantly outperform them.

    

### [[1906.00219] Probabilistic Top-k Dominating Query Monitoring over Multiple Uncertain IoT Data Streams in Edge Computing Environments](http://arxiv.org/abs/1906.00219)


  Extracting the valuable features and information in Big Data has become one
of the important research issues in Data Science. In most Internet of Things
(IoT) applications, the collected data are uncertain and imprecise due to
sensor device variations or transmission errors. In addition, the sensing data
may change as time evolves. We refer an uncertain data stream as a dataset that
has velocity, veracity, and volume properties simultaneously. This paper
employs the parallelism in edge computing environments to facilitate the top-k
dominating query process over multiple uncertain IoT data streams. The
challenges of this problem include how to quickly update the result for
processing uncertainty and reduce the computation cost as well as provide
highly accurate results. By referring to the related existing papers for
certain data, we provide an effective probabilistic top-k dominating query
process on uncertain data streams, which can be parallelized easily. After
discussing the properties of the proposed approach, we validate our methods
through the complexity analysis and extensive simulated experiments. In
comparison with the existing works, the experimental results indicate that our
method can improve almost 60% computation time, reduce nearly 20% communication
cost between servers, and provide highly accurate results in most scenarios.

    

### [[2109.11595] Adaptive Sampling using POMDPs with Domain-Specific Considerations](http://arxiv.org/abs/2109.11595)


  We investigate improving Monte Carlo Tree Search based solvers for Partially
Observable Markov Decision Processes (POMDPs), when applied to adaptive
sampling problems. We propose improvements in rollout allocation, the action
exploration algorithm, and plan commitment. The first allocates a different
number of rollouts depending on how many actions the agent has taken in an
episode. We find that rollouts are more valuable after some initial information
is gained about the environment. Thus, a linear increase in the number of
rollouts, i.e. allocating a fixed number at each step, is not appropriate for
adaptive sampling tasks. The second alters which actions the agent chooses to
explore when building the planning tree. We find that by using knowledge of the
number of rollouts allocated, the agent can more effectively choose actions to
explore. The third improvement is in determining how many actions the agent
should take from one plan. Typically, an agent will plan to take the first
action from the planning tree and then call the planner again from the new
state. Using statistical techniques, we show that it is possible to greatly
reduce the number of rollouts by increasing the number of actions taken from a
single planning tree without affecting the agent's final reward. Finally, we
demonstrate experimentally, on simulated and real aquatic data from an
underwater robot, that these improvements can be combined, leading to better
adaptive sampling. The code for this work is available at
this https URL


### [[2109.11654] Modeling Dynamic Attributes for Next Basket Recommendation](http://arxiv.org/abs/2109.11654)


  Traditional approaches to next item and next basket recommendation typically
extract users' interests based on their past interactions and associated static
contextual information (e.g. a user id or item category). However, extracted
interests can be inaccurate and become obsolete. Dynamic attributes, such as
user income changes, item price changes (etc.), change over time. Such dynamics
can intrinsically reflect the evolution of users' interests. We argue that
modeling such dynamic attributes can boost recommendation performance. However,
properly integrating them into user interest models is challenging since
attribute dynamics can be diverse such as time-interval aware, periodic
patterns (etc.), and they represent users' behaviors from different
perspectives, which can happen asynchronously with interactions. Besides
dynamic attributes, items in each basket contain complex interdependencies
which might be beneficial but nontrivial to effectively capture. To address
these challenges, we propose a novel Attentive network to model Dynamic
attributes (named AnDa). AnDa separately encodes dynamic attributes and basket
item sequences. We design a periodic aware encoder to allow the model to
capture various temporal patterns from dynamic attributes. To effectively learn
useful item relationships, intra-basket attention module is proposed.
Experimental results on three real-world datasets demonstrate that our method
consistently outperforms the state-of-the-art.

    

### [[2109.11668] Exact Learning of Qualitative Constraint Networks from Membership Queries](http://arxiv.org/abs/2109.11668)


  A Qualitative Constraint Network (QCN) is a constraint graph for representing
problems under qualitative temporal and spatial relations, among others. More
formally, a QCN includes a set of entities, and a list of qualitative
constraints defining the possible scenarios between these entities. These
latter constraints are expressed as disjunctions of binary relations capturing
the (incomplete) knowledge between the involved entities. QCNs are very
effective in representing a wide variety of real-world applications, including
scheduling and planning, configuration and Geographic Information Systems
(GIS). It is however challenging to elicit, from the user, the QCN representing
a given problem. To overcome this difficulty in practice, we propose a new
algorithm for learning, through membership queries, a QCN from a non expert. In
this paper, membership queries are asked in order to elicit temporal or spatial
relationships between pairs of temporal or spatial entities. In order to
improve the time performance of our learning algorithm in practice, constraint
propagation, through transitive closure, as well as ordering heuristics, are
enforced. The goal here is to reduce the number of membership queries needed to
reach the target QCN. In order to assess the practical effect of constraint
propagation and ordering heuristics, we conducted several experiments on
randomly generated temporal and spatial constraint network instances. The
results of the experiments are very encouraging and promising.

    

### [[2109.11682] Paint4Poem: A Dataset for Artistic Visualization of Classical Chinese Poems](http://arxiv.org/abs/2109.11682)


  In this work we propose a new task: artistic visualization of classical
Chinese poems, where the goal is to generatepaintings of a certain artistic
style for classical Chinese poems. For this purpose, we construct a new dataset
called Paint4Poem. Thefirst part of Paint4Poem consists of 301 high-quality
poem-painting pairs collected manually from an influential modern Chinese
artistFeng Zikai. As its small scale poses challenges for effectively training
poem-to-painting generation models, we introduce the secondpart of Paint4Poem,
which consists of 3,648 caption-painting pairs collected manually from Feng
Zikai's paintings and 89,204 poem-painting pairs collected automatically from
the web. We expect the former to help learning the artist painting style as it
containshis most paintings, and the latter to help learning the semantic
relevance between poems and paintings. Further, we analyze Paint4Poem regarding
poem diversity, painting style, and the semantic relevance between poems and
paintings. We create abenchmark for Paint4Poem: we train two representative
text-to-image generation models: AttnGAN and MirrorGAN, and evaluate
theirperformance regarding painting pictorial quality, painting stylistic
relevance, and semantic relevance between poems and paintings.The results
indicate that the models are able to generate paintings that have good
pictorial quality and mimic Feng Zikai's style, but thereflection of poem
semantics is limited. The dataset also poses many interesting research
directions on this task, including transferlearning, few-shot learning,
text-to-image generation for low-resource data etc. The dataset is publicly
available.(this https URL)

    

### [[2109.11728] AES Are Both Overstable And Oversensitive: Explaining Why And Proposing Defenses](http://arxiv.org/abs/2109.11728)


  Deep-learning based Automatic Essay Scoring (AES) systems are being actively
used by states and language testing agencies alike to evaluate millions of
candidates for life-changing decisions ranging from college applications to
visa approvals. However, little research has been put to understand and
interpret the black-box nature of deep-learning based scoring algorithms.
Previous studies indicate that scoring models can be easily fooled. In this
paper, we explore the reason behind their surprising adversarial brittleness.
We utilize recent advances in interpretability to find the extent to which
features such as coherence, content, vocabulary, and relevance are important
for automated scoring mechanisms. We use this to investigate the
oversensitivity i.e., large change in output score with a little change in
input essay content) and overstability i.e., little change in output scores
with large changes in input essay content) of AES. Our results indicate that
autoscoring models, despite getting trained as "end-to-end" models with rich
contextual embeddings such as BERT, behave like bag-of-words models. A few
words determine the essay score without the requirement of any context making
the model largely overstable. This is in stark contrast to recent probing
studies on pre-trained representation learning models, which show that rich
linguistic features such as parts-of-speech and morphology are encoded by them.
Further, we also find that the models have learnt dataset biases, making them
oversensitive. To deal with these issues, we propose detection-based protection
models that can detect oversensitivity and overstability causing samples with
high accuracies. We find that our proposed models are able to detect unusual
attribution patterns and flag adversarial samples successfully.

    

### [[2109.11745] DACT-BERT: Differentiable Adaptive Computation Time for an Efficient BERT Inference](http://arxiv.org/abs/2109.11745)


  Large-scale pre-trained language models have shown remarkable results in
diverse NLP applications. Unfortunately, these performance gains have been
accompanied by a significant increase in computation time and model size,
stressing the need to develop new or complementary strategies to increase the
efficiency of these models. In this paper we propose DACT-BERT, a
differentiable adaptive computation time strategy for BERT-like models.
DACT-BERT adds an adaptive computational mechanism to BERT's regular processing
pipeline, which controls the number of Transformer blocks that need to be
executed at inference time. By doing this, the model learns to combine the most
appropriate intermediate representations for the task at hand. Our experiments
demonstrate that our approach, when compared to the baselines, excels on a
reduced computational regime and is competitive in other less restrictive ones.

    

### [[2109.11747] Multi-View Video-Based 3D Hand Pose Estimation](http://arxiv.org/abs/2109.11747)


  Hand pose estimation (HPE) can be used for a variety of human-computer
interaction applications such as gesture-based control for physical or
virtual/augmented reality devices. Recent works have shown that videos or
multi-view images carry rich information regarding the hand, allowing for the
development of more robust HPE systems. In this paper, we present the
Multi-View Video-Based 3D Hand (MuViHand) dataset, consisting of multi-view
videos of the hand along with ground-truth 3D pose labels. Our dataset includes
more than 402,000 synthetic hand images available in 4,560 videos. The videos
have been simultaneously captured from six different angles with complex
backgrounds and random levels of dynamic lighting. The data has been captured
from 10 distinct animated subjects using 12 cameras in a semi-circle topology
where six tracking cameras only focus on the hand and the other six fixed
cameras capture the entire body. Next, we implement MuViHandNet, a neural
pipeline consisting of image encoders for obtaining visual embeddings of the
hand, recurrent learners to learn both temporal and angular sequential
information, and graph networks with U-Net architectures to estimate the final
3D pose information. We perform extensive experiments and show the challenging
nature of this new dataset as well as the effectiveness of our proposed method.
Ablation studies show the added value of each component in MuViHandNet, as well
as the benefit of having temporal and sequential information in the dataset.

    

### [[2109.11763] Lacking the embedding of a word? Look it up into a traditional dictionary](http://arxiv.org/abs/2109.11763)


  Word embeddings are powerful dictionaries, which may easily capture language
variations. However, these dictionaries fail to give sense to rare words, which
are surprisingly often covered by traditional dictionaries. In this paper, we
propose to use definitions retrieved in traditional dictionaries to produce
word embeddings for rare words. For this purpose, we introduce two methods:
Definition Neural Network (DefiNNet) and Define BERT (DefBERT). In our
experiments, DefiNNet and DefBERT significantly outperform state-of-the-art as
well as baseline methods devised for producing embeddings of unknown words. In
fact, DefiNNet significantly outperforms FastText, which implements a method
for the same task-based on n-grams, and DefBERT significantly outperforms the
BERT method for OOV words. Then, definitions in traditional dictionaries are
useful to build word embeddings for rare words.

    

### [[2109.11790] Learning Dual Dynamic Representations on Time-Sliced User-Item Interaction Graphs for Sequential Recommendation](http://arxiv.org/abs/2109.11790)


  Sequential Recommendation aims to recommend items that a target user will
interact with in the near future based on the historically interacted items.
While modeling temporal dynamics is crucial for sequential recommendation, most
of the existing studies concentrate solely on the user side while overlooking
the sequential patterns existing in the counterpart, i.e., the item side.
Although a few studies investigate the dynamics involved in the dual sides, the
complex user-item interactions are not fully exploited from a global
perspective to derive dynamic user and item representations. In this paper, we
devise a novel Dynamic Representation Learning model for Sequential
Recommendation (DRL-SRe). To better model the user-item interactions for
characterizing the dynamics from both sides, the proposed model builds a global
user-item interaction graph for each time slice and exploits time-sliced graph
neural networks to learn user and item representations. Moreover, to enable the
model to capture fine-grained temporal information, we propose an auxiliary
temporal prediction task over consecutive time slices based on temporal point
process. Comprehensive experiments on three public real-world datasets
demonstrate DRL-SRe outperforms the state-of-the-art sequential recommendation
models with a large margin.

    

### [[2109.11849] Explanation Strategies as an Empirical-Analytical Lens for Socio-Technical Contextualization of Machine Learning Interpretability](http://arxiv.org/abs/2109.11849)


  During a research project in which we developed a machine learning (ML)
driven visualization system for non-ML experts, we reflected on
interpretability research in ML, computer-supported collaborative work and
human-computer interaction. We found that while there are manifold technical
approaches, these often focus on ML experts and are evaluated in
decontextualized empirical studies. We hypothesized that participatory design
research may support the understanding of stakeholders' situated sense-making
in our project, yet, found guidance regarding ML interpretability inexhaustive.
Building on philosophy of technology, we formulated explanation strategies as
an empirical-analytical lens explicating how technical explanations mediate the
contextual preferences concerning people's interpretations. In this paper, we
contribute a report of our proof-of-concept use of explanation strategies to
analyze a co-design workshop with non-ML experts, methodological implications
for participatory design research, design implications for explanations for
non-ML experts and suggest further investigation of technological mediation
theories in the ML interpretability space.

    

### [[2109.11938] Meta-brain Models: biologically-inspired cognitive agents](http://arxiv.org/abs/2109.11938)


  Artificial Intelligence (AI) systems based solely on neural networks or
symbolic computation present a representational complexity challenge. While
minimal representations can produce behavioral outputs like locomotion or
simple decision-making, more elaborate internal representations might offer a
richer variety of behaviors. We propose that these issues can be addressed with
a computational approach we call meta-brain models. Meta-brain models are
embodied hybrid models that include layered components featuring varying
degrees of representational complexity. We will propose combinations of layers
composed using specialized types of models. Rather than using a generic black
box approach to unify each component, this relationship mimics systems like the
neocortical-thalamic system relationship of the Mammalian brain, which utilizes
both feedforward and feedback connectivity to facilitate functional
communication. Importantly, the relationship between layers can be made
anatomically explicit. This allows for structural specificity that can be
incorporated into the model's function in interesting ways. We will propose
several types of layers that might be functionally integrated into agents that
perform unique types of tasks, from agents that simultaneously perform
morphogenesis and perception, to agents that undergo morphogenesis and the
acquisition of conceptual representations simultaneously. Our approach to
meta-brain models involves creating models with different degrees of
representational complexity, creating a layered meta-architecture that mimics
the structural and functional heterogeneity of biological brains, and an
input/output methodology flexible enough to accommodate cognitive functions,
social interactions, and adaptive behaviors more generally. We will conclude by
proposing next steps in the development of this flexible and open-source
approach.

    

### [[2109.11969] Rethinking Crowd Sourcing for Semantic Similarity](http://arxiv.org/abs/2109.11969)


  Estimation of semantic similarity is crucial for a variety of natural
language processing (NLP) tasks. In the absence of a general theory of semantic
information, many papers rely on human annotators as the source of ground truth
for semantic similarity estimation. This paper investigates the ambiguities
inherent in crowd-sourced semantic labeling. It shows that annotators that
treat semantic similarity as a binary category (two sentences are either
similar or not similar and there is no middle ground) play the most important
role in the labeling. The paper offers heuristics to filter out unreliable
annotators and stimulates further discussions on human perception of semantic
similarity.

    

### [[2109.12056] Parameterized Channel Normalization for Far-field Deep Speaker Verification](http://arxiv.org/abs/2109.12056)


  We address far-field speaker verification with deep neural network (DNN)
based speaker embedding extractor, where mismatch between enrollment and test
data often comes from convolutive effects (e.g. room reverberation) and noise.
To mitigate these effects, we focus on two parametric normalization methods:
per-channel energy normalization (PCEN) and parameterized cepstral mean
normalization (PCMN). Both methods contain differentiable parameters and thus
can be conveniently integrated to, and jointly optimized with the DNN using
automatic differentiation methods. We consider both fixed and trainable
(data-driven) variants of each method. We evaluate the performance on Hi-MIA, a
recent large-scale far-field speech corpus, with varied microphone and
positional settings. Our methods outperform conventional mel filterbank
features, with maximum of 33.5% and 39.5% relative improvement on equal error
rate under matched microphone and mismatched microphone conditions,
respectively.

    

### [[2109.12058] Optimized Power Normalized Cepstral Coefficients towards Robust Deep Speaker Verification](http://arxiv.org/abs/2109.12058)


  After their introduction to robust speech recognition, power normalized
cepstral coefficient (PNCC) features were successfully adopted to other tasks,
including speaker verification. However, as a feature extractor with long-term
operations on the power spectrogram, its temporal processing and amplitude
scaling steps dedicated on environmental compensation may be redundant.
Further, they might suppress intrinsic speaker variations that are useful for
speaker verification based on deep neural networks (DNN). Therefore, in this
study, we revisit and optimize PNCCs by ablating its medium-time processor and
by introducing channel energy normalization. Experimental results with a
DNN-based speaker verification system indicate substantial improvement over
baseline PNCCs on both in-domain and cross-domain scenarios, reflected by
relatively 5.8% and 61.2% maximum lower equal error rate on VoxCeleb1 and
VoxMovies, respectively.

    

### [[2109.12065] DeepStroke: An Efficient Stroke Screening Framework for Emergency Rooms with Multimodal Adversarial Deep Learning](http://arxiv.org/abs/2109.12065)


  In an emergency room (ER) setting, the diagnosis of stroke is a common
challenge. Due to excessive execution time and cost, an MRI scan is usually not
available in the ER. Clinical tests are commonly referred to in stroke
screening, but neurologists may not be immediately available. We propose a
novel multimodal deep learning framework, DeepStroke, to achieve computer-aided
stroke presence assessment by recognizing the patterns of facial motion
incoordination and speech inability for patients with suspicion of stroke in an
acute setting. Our proposed DeepStroke takes video data for local facial
paralysis detection and audio data for global speech disorder analysis. It
further leverages a multi-modal lateral fusion to combine the low- and
high-level features and provides mutual regularization for joint training. A
novel adversarial training loss is also introduced to obtain
identity-independent and stroke-discriminative features. Experiments on our
video-audio dataset with actual ER patients show that the proposed approach
outperforms state-of-the-art models and achieves better performance than ER
doctors, attaining a 6.60% higher sensitivity and maintaining 4.62% higher
accuracy when specificity is aligned. Meanwhile, each assessment can be
completed in less than 6 minutes, demonstrating the framework's great potential
for clinical implementation.

    

### [[2109.12093] SAIS: Supervising and Augmenting Intermediate Steps for Document-Level Relation Extraction](http://arxiv.org/abs/2109.12093)


  Stepping from sentence-level to document-level relation extraction, the
research community confronts increasing text length and more complicated entity
interactions. Consequently, it is more challenging to encode the key sources of
information--relevant contexts and entity types. However, existing methods only
implicitly learn to model these critical information sources while being
trained for relation extraction. As a result, they suffer the problems of
ineffective supervision and uninterpretable model predictions. In contrast, we
propose to explicitly teach the model to capture relevant contexts and entity
types by supervising and augmenting intermediate steps (SAIS) for relation
extraction. Based on a broad spectrum of carefully designed tasks, our proposed
SAIS method not only extracts relations of better quality due to more effective
supervision, but also retrieves the corresponding supporting evidence more
accurately so as to enhance interpretability. By assessing model uncertainty,
SAIS further boosts the performance via evidence-based data augmentation and
ensemble inference while reducing the computational cost. Eventually, SAIS
delivers state-of-the-art relation extraction results on three benchmarks
(DocRED, CDR, and GDA) and achieves 5.04% relative gains in F1 score compared
to the runner-up in evidence retrieval on DocRED.

    

### [[1910.01208] Distributed Attack-Robust Submodular Maximization for Multi-Robot Planning](http://arxiv.org/abs/1910.01208)


  In this paper, we design algorithms to protect swarm-robotics applications
against sensor denial-of-service (DoS) attacks on robots. We focus on
applications requiring the robots to jointly select actions, e.g., which
trajectory to follow, among a set of available ones. Such applications are
central in large-scale robotic applications, such as multi-robot motion
planning for target tracking. But the current attack-robust algorithms are
centralized. In this paper, we propose a general-purpose distributed algorithm
towards robust optimization at scale, with local communications only. We name
it Distributed Robust Maximization (DRM). DRM proposes a divide-and-conquer
approach that distributively partitions the problem among cliques of robots.
Then, the cliques optimize in parallel, independently of each other. We prove
DRM achieves a close-to-optimal performance. We demonstrate DRM's performance
in both Gazebo and MATLAB simulations, in scenarios of active target tracking
with swarms of robots. In the simulations, DRM achieves computational
speed-ups, being 1-2 orders faster than the centralized algorithms; yet, it
nearly matches the tracking performance of the centralized counterparts. Since,
DRM overestimates the number of attacks in each clique, in this paper we also
introduce an Improved Distributed Robust Maximization (IDRM) algorithm. IDRM
infers the number of attacks in each clique less conservatively than DRM by
leveraging 3-hop neighboring communications. We verify IDRM improves DRM's
performance in simulations.

    

### [[2007.06796] Calling Out Bluff: Evaluation Toolkit For Robustness Testing Of Automatic Essay Scoring Systems](http://arxiv.org/abs/2007.06796)


  Automatic scoring engines have been used for scoring approximately fifteen
million test-takers in just the last three years. This number is increasing
further due to COVID-19 and the associated automation of education and testing.
Despite such wide usage, the AI-based testing literature of these "intelligent"
models is highly lacking. Most of the papers proposing new models rely only on
quadratic weighted kappa (QWK) based agreement with human raters for showing
model efficacy. However, this effectively ignores the highly multi-feature
nature of essay scoring. Essay scoring depends on features like coherence,
grammar, relevance, sufficiency and, vocabulary. To date, there has been no
study testing Automated Essay Scoring: AES systems holistically on all these
features. With this motivation, we propose a model agnostic adversarial
evaluation scheme and associated metrics for AES systems to test their natural
language understanding capabilities and overall robustness. We evaluate the
current state-of-the-art AES models using the proposed scheme and report the
results on five recent models. These models range from
feature-engineering-based approaches to the latest deep learning algorithms. We
find that AES models are highly overstable. Even heavy modifications(as much as
25%) with content unrelated to the topic of the questions do not decrease the
score produced by the models. On the other hand, irrelevant content, on
average, increases the scores, thus showing that the model evaluation strategy
and rubrics should be reconsidered. We also ask 200 human raters to score both
an original and adversarial response to seeing if humans can detect differences
between the two and whether they agree with the scores assigned by auto scores.

    

### [[2009.09191] OpenAttack: An Open-source Textual Adversarial Attack Toolkit](http://arxiv.org/abs/2009.09191)


  Textual adversarial attacking has received wide and increasing attention in
recent years. Various attack models have been proposed, which are enormously
distinct and implemented with different programming frameworks and settings.
These facts hinder quick utilization and fair comparison of attack models. In
this paper, we present an open-source textual adversarial attack toolkit named
OpenAttack to solve these issues. Compared with existing other textual
adversarial attack toolkits, OpenAttack has its unique strengths in support for
all attack types, multilinguality, and parallel processing. Currently,
OpenAttack includes 15 typical attack models that cover all attack types. Its
highly inclusive modular design not only supports quick utilization of existing
attack models, but also enables great flexibility and extensibility. OpenAttack
has broad uses including comparing and evaluating attack models, measuring
robustness of a model, assisting in developing new attack models, and
adversarial training. Source code and documentation can be obtained at
this https URL.

    

### [[2010.09325] Body models in humans, animals, and robots: mechanisms and plasticity](http://arxiv.org/abs/2010.09325)


  Humans and animals excel in combining information from multiple sensory
modalities, controlling their complex bodies, adapting to growth, failures, or
using tools. These capabilities are also highly desirable in robots. They are
displayed by machines to some extent - yet, as is so often the case, the
artificial creatures are lagging behind. The key foundation is an internal
representation of the body that the agent - human, animal, or robot - has
developed. In the biological realm, evidence has been accumulated by diverse
disciplines giving rise to the concepts of body image, body schema, and others.
In robotics, a model of the robot is an indispensable component that enables to
control the machine. In this article I compare the character of body
representations in biology with their robotic counterparts and relate that to
the differences in performance that we observe. I put forth a number of axes
regarding the nature of such body models: fixed vs. plastic, amodal vs. modal,
explicit vs. implicit, serial vs. parallel, modular vs. holistic, and
centralized vs. distributed. An interesting trend emerges: on many of the axes,
there is a sequence from robot body models, over body image, body schema, to
the body representation in lower animals like the octopus. In some sense,
robots have a lot in common with Ian Waterman - "the man who lost his body" -
in that they rely on an explicit, veridical body model (body image taken to the
extreme) and lack any implicit, multimodal representation (like the body
schema) of their bodies. I will then detail how robots can inform the
biological sciences dealing with body representations and finally, I will study
which of the features of the "body in the brain" should be transferred to
robots, giving rise to more adaptive and resilient, self-calibrating machines.

    

### [[2101.06848] Faster Convergence in Deep-Predictive-Coding Networks to Learn Deeper Representations](http://arxiv.org/abs/2101.06848)


  Deep-predictive-coding networks (DPCNs) are hierarchical, generative models.
They rely on feed-forward and feed-back connections to modulate latent feature
representations of stimuli in a dynamic and context-sensitive manner. A crucial
element of DPCNs is a forward-backward inference procedure to uncover sparse,
invariant features. However, this inference is a major computational
bottleneck. It severely limits the network depth due to learning stagnation.
Here, we prove why this bottleneck occurs. We then propose a new
forward-inference strategy based on accelerated proximal gradients. This
strategy has faster theoretical convergence guarantees than the one used for
DPCNs. It overcomes learning stagnation. We also demonstrate that it permits
constructing deep and wide predictive-coding networks. Such convolutional
networks implement receptive fields that capture well the entire classes of
objects on which the networks are trained. This improves the feature
representations compared with our lab's previous non-convolutional and
convolutional DPCNs. It yields unsupervised object recognition that surpass
convolutional autoencoders and are on par with convolutional networks trained
in a supervised manner.

    

### [[2104.03616] Arena-Rosnav: Towards Deployment of Deep-Reinforcement-Learning-Based Obstacle Avoidance into Conventional Autonomous Navigation Systems](http://arxiv.org/abs/2104.03616)


  Recently, mobile robots have become important tools in various industries,
especially in logistics. Deep reinforcement learning emerged as an alternative
planning method to replace overly conservative approaches and promises more
efficient and flexible navigation. However, deep reinforcement learning
approaches are not suitable for long-range navigation due to their proneness to
local minima and lack of long term memory, which hinders its widespread
integration into industrial applications of mobile robotics. In this paper, we
propose a navigation system incorporating deep-reinforcement-learning-based
local planners into conventional navigation stacks for long-range navigation.
Therefore, a framework for training and testing the deep reinforcement learning
algorithms along with classic approaches is presented. We evaluated our
deep-reinforcement-learning-enhanced navigation system against various
conventional planners and found that our system outperforms them in terms of
safety, efficiency and robustness.

    

### [[2104.09757] Imaginative Walks: Generative Random Walk Deviation Loss for Improved Unseen Learning Representation](http://arxiv.org/abs/2104.09757)


  We propose a novel loss for generative models, dubbed as GRaWD (Generative
Random Walk Deviation), to improve learning representations of unexplored
visual spaces. Quality learning representation of unseen classes (or styles) is
critical to facilitate novel image generation and better generative
understanding of unseen visual classes, i.e., zero-shot learning (ZSL). By
generating representations of unseen classes based on their semantic
descriptions, e.g., attributes or text, generative ZSL attempts to
differentiate unseen from seen categories. The proposed GRaWD loss is defined
by constructing a dynamic graph that includes the seen class/style centers and
generated samples in the current minibatch. Our loss initiates a random walk
probability from each center through visual generations produced from
hallucinated unseen classes. As a deviation signal, we encourage the random
walk to eventually land after t steps in a feature representation that is
difficult to classify as any of the seen classes. We demonstrate that the
proposed loss can improve unseen class representation quality inductively on
text-based ZSL benchmarks on CUB and NABirds datasets and attribute-based ZSL
benchmarks on AWA2, SUN, and aPY datasets. In addition, we investigate the
ability of the proposed loss to generate meaningful novel visual art on the
WikiArt dataset. The results of experiments and human evaluations demonstrate
that the proposed GRaWD loss can improve StyleGAN1 and StyleGAN2 generation
quality and create novel art that is significantly more preferable. Our code is
made publicly available at this https URL.

    

### [[2109.11666] SLO beyond the Hardware Isolation Limits](http://arxiv.org/abs/2109.11666)


  Performance isolation is a keystone for SLO guarantees with shared resources
in cloud and datacenter environments. To meet SLO requirements, the state of
the art relies on hardware QoS support (e.g., Intel RDT) to allocate shared
resources such as last-level caches and memory bandwidth for co-located
latency-critical applications. As a result, the number of latency-critical
applications that can be deployed on a physical machine is bounded by the
hardware allocation capability. Unfortunately, such hardware capability is very
limited. For example, Intel Xeon E5 v3 processors support at most four
partitions for last-level caches, i.e., at most four applications can have
dedicated resource allocation. This paper discusses the feasibility and
unexplored challenges of providing SLO guarantees beyond the limits of hardware
capability. We present CoCo to show the feasibility and the benefits. CoCo
schedules applications to time-share interference-free partitions as a
transparent software layer. Our evaluation shows that CoCo outperforms
non-partitioned and round-robin approaches by up to 9x and 1.2x.

    

### [[2109.11802] Automated Modular Verification for Race-Free Channels with Implicit and Explicit Synchronization](http://arxiv.org/abs/2109.11802)


  Ensuring the correctness of software for communication centric programs is
important but challenging. Previous approaches, based on session types, have
been intensively investigated over the past decade. They provide a concise way
to express protocol specifications and a lightweight approach for checking
their implementation. Current solutions are based on only implicit
synchronization, and are based on the less precise types rather than logical
formulae. In this paper, we propose a more expressive session logic to capture
multiparty protocols. By using two kinds of ordering constraints, namely
"happens-before" <HB and "communicates-before" <CB, we show how to ensure from
first principle race-freedom over common channels. Our approach refines each
specification with both assumptions and proof obligations to ensure compliance
to some global protocol. Each specification is then projected for each party
and then each channel, to allow cooperative proving through localized automated
verification. Our primary goal in automated verification is to ensure
race-freedom and communication-safety, but the approach is extensible for
deadlock-freedom as well. We shall also describe how modular protocols can be
captured and handled by our approach.

    

### [<title>ValueError: Input contains NaN, infinity or a value too large for dtype('float32') - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtype-float32/1095/6)