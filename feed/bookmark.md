
## 2021-12-22

### [[2112.10885] PRONTO: Preamble Overhead Reduction with Neural Networks for Coarse Synchronization](http://arxiv.org/abs/2112.10885)


  In IEEE 802.11 WiFi-based waveforms, the receiver performs coarse time and
frequency synchronization using the first field of the preamble known as the
legacy short training field (L-STF). The L-STF occupies upto 40% of the
preamble length and takes upto 32 us of airtime. With the goal of reducing
communication overhead, we propose a modified waveform, where the preamble
length is reduced by eliminating the L-STF. To decode this modified waveform,
we propose a machine learning (ML)-based scheme called PRONTO that performs
coarse time and frequency estimations using other preamble fields, specifically
the legacy long training field (L-LTF). Our contributions are threefold: (i) We
present PRONTO featuring customized convolutional neural networks (CNNs) for
packet detection and coarse CFO estimation, along with data augmentation steps
for robust training. (ii) We propose a generalized decision flow that makes
PRONTO compatible with legacy waveforms that include the standard L-STF. (iii)
We validate the outcomes on an over-the-air WiFi dataset from a testbed of
software defined radios (SDRs). Our evaluations show that PRONTO can perform
packet detection with 100% accuracy, and coarse CFO estimation with errors as
small as 3%. We demonstrate that PRONTO provides upto 40% preamble length
reduction with no bit error rate (BER) degradation. Finally, we experimentally
show the speedup achieved by PRONTO through GPU parallelization over the
corresponding CPU-only implementations.

    

### [[2112.10931] Hiding Signal Strength Interference from Outside Adversaries](http://arxiv.org/abs/2112.10931)


  The presence of people can be detected by passively observing the signal
strength of Wifi and related forms of communication. This paper tackles the
question of how and when can this be prevented by adjustments to the
transmitted signal strength, and other similar measures. The main contribution
of this paper is a formal framework to analyze this problem, and the
identification of several scenarios and corresponding protocols which can
prevent or limit the inference from passive signal strength snooping.

    

### [[2112.11002] Efficient Quantum Network Communication using Optimized Entanglement-Swapping Trees](http://arxiv.org/abs/2112.11002)


  Quantum network communication is challenging, as the No-cloning theorem in
quantum regime makes many classical techniques inapplicable. For long-distance
communication, the only viable communication approach is teleportation of
quantum states, which requires a prior distribution of entangled pairs (EPs) of
qubits. Establishment of EPs across remote nodes can incur significant latency
due to the low probability of success of the underlying physical processes.
The focus of our work is to develop efficient techniques that minimize EP
generation latency. Prior works have focused on selecting entanglement paths;
in contrast, we select entanglement swapping trees--a more accurate
representation of the entanglement generation structure. We develop a dynamic
programming algorithm to select an optimal swapping-tree for a single pair of
nodes, under the given capacity and fidelity constraints. For the general
setting, we develop an efficient iterative algorithm to compute a set of
swapping trees. We present simulation results which show that our solutions
outperform the prior approaches by an order of magnitude and are viable for
long-distance entanglement generation.

    

### [[2112.11090] Aerial Base Station Positioning and Power Control for Securing Communications: A Deep Q-Network Approach](http://arxiv.org/abs/2112.11090)


  The unmanned aerial vehicle (UAV) is one of the technological breakthroughs
that supports a variety of services, including communications. UAV will play a
critical role in enhancing the physical layer security of wireless networks.
This paper defines the problem of eavesdropping on the link between the ground
user and the UAV, which serves as an aerial base station (ABS). The
reinforcement learning algorithms Q-learning and deep Q-network (DQN) are
proposed for optimizing the position of the ABS and the transmission power to
enhance the data rate of the ground user. This increases the secrecy capacity
without the system knowing the location of the eavesdropper. Simulation results
show fast convergence and the highest secrecy capacity of the proposed DQN
compared to Q-learning and baseline approaches.

    

### [[2112.11109] Network Anomaly Detection in Cars: A Case for Time-Sensitive Stream Filtering and Policing](http://arxiv.org/abs/2112.11109)


  Connected cars are vulnerable to cyber attacks. Security challenges arise
from vehicular management uplinks, from signaling with roadside units or nearby
cars, as well as from common Internet services. Major threats arrive from bogus
traffic that enters the in-car backbone, which will comprise of Ethernet
technologies in the near future. Various security techniques from different
areas and layers are under discussion to protect future vehicles. In this
paper, we show how Per-Stream Filtering and Policing of IEEE Time-Sensitive
Networking (TSN) can be used as a core technology for identifying misbehaving
traffic flows in cars, and thereby serve as network anomaly detectors. TSN is
the leading candidate for implementing quality of service in vehicular Ethernet
backbones. We classify the impact of network attacks on traffic flows and
benchmark the detection performance in each individual class. Based on a
backbone topology derived from a real car and its traffic definition, we
evaluate the detection system in realistic scenarios with real attack traces.
Our results show that the detection accuracy depends on the precision of the
in-vehicle communication specification, the traffic type, the corruption layer,
and the attack impact on the link layer. Most notably, the anomaly indicators
of our approach remain free of false positive alarms, which is an important
foundation for implementing automated countermeasures in future vehicles.

    

### [[2112.11119] Trying an IP Over NDN Packet Gateway](http://arxiv.org/abs/2112.11119)


  Even though the TCP/IP architecture has served the Internet quite
satisfactorily during its more than forty years of lifespan, there are doubts
about whether this host-centric paradigm is well suited for the communication
patterns of modern applications. These applications mainly need to get
information pieces without being bothered about where they are located in the
network. Additionally, the lack of both in-built security mechanisms and proper
global multicast support may also be a symptom of fundamental problems with the
classic architecture. Proponents of the novel Information Centric Networking
(ICN) paradigm assert that the replacement of the TCP/IP architecture with one
based on the information itself could be the ideal solution to all these
problems. However, if such a replacement ever takes place, it will need the
help of some transition mechanisms, for instance, for the transmission of
legacy IP traffic on top of the new ICN based networks. In this paper we design
and optimize such an open source IP over ICN transition application using the
Named Data Networking (NDN) proposal as our target ICN network. We performed
several tests that confirm that our prototype is able to transparently
transport IP traffic across a pure NDN network with negligible packets losses,
low jitter, and good performance.

    

### [[2112.11165] Scalable High-Rate Twin-Field Quantum Key Distribution Networks without Constraint of Probability and Intensity](http://arxiv.org/abs/2112.11165)


  There have been several recent advancements in the field of long-distance
point-to-point twin-field quantum key distribution (TFQKD) protocols, with an
ultimate objective to build a large scalable quantum network for numerous
users. Currently, fundamental limitations still exist for the implementation of
a practical TFQKD network, including the strict constraint regarding intensity
and probability for sending-or-not-sending type protocols and the low tolerance
of large interference errors for phase-matching type protocols. Here, we
propose a two-photon TFQKD protocol to overcome these issues simultaneously and
introduce a cost-effective solution to construct a real TFQKD network, under
which each node with fixed system parameters can dynamically switch different
attenuation links while achieving good performance in long-distance
transmission. For a four-user network, simulation results indicate that the key
rates of our protocol for all six links can either exceed or approach the
secret key capacity; however, four of them could not extract the key rate if
using sending-or-not-sending type protocols. We anticipate that our proposed
method can facilitate new practical and efficient TFQKD networks in the future.

    

### [[2112.11191] Developing a Trusted Human-AI Network for Humanitarian Benefit](http://arxiv.org/abs/2112.11191)


  Humans and artificial intelligences (AI) will increasingly participate
digitally and physically in conflicts, yet there is a lack of trusted
communications across agents and platforms. For example, humans in disasters
and conflict already use messaging and social media to share information,
however, international humanitarian relief organisations treat this information
as unverifiable and untrustworthy. AI may reduce the 'fog-of-war' and improve
outcomes, however AI implementations are often brittle, have a narrow scope of
application and wide ethical risks. Meanwhile, human error causes significant
civilian harms even by combatants committed to complying with international
humanitarian law. AI offers an opportunity to help reduce the tragedy of war
and deliver humanitarian aid to those who need it. In this paper we consider
the integration of a communications protocol (the 'Whiteflag protocol'),
distributed ledger technology, and information fusion with artificial
intelligence (AI), to improve conflict communications called 'Protected
Assurance Understanding Situation and Entities' (PAUSE). Such a trusted
human-AI communication network could provide accountable information exchange
regarding protected entities, critical infrastructure; humanitarian signals and
status updates for humans and machines in conflicts.

    

### [[2112.11256] Tackling System and Statistical Heterogeneity for Federated Learning with Adaptive Client Sampling](http://arxiv.org/abs/2112.11256)


  Federated learning (FL) algorithms usually sample a fraction of clients in
each round (partial participation) when the number of participants is large and
the server's communication bandwidth is limited. Recent works on the
convergence analysis of FL have focused on unbiased client sampling, e.g.,
sampling uniformly at random, which suffers from slow wall-clock time for
convergence due to high degrees of system heterogeneity and statistical
heterogeneity. This paper aims to design an adaptive client sampling algorithm
that tackles both system and statistical heterogeneity to minimize the
wall-clock convergence time. We obtain a new tractable convergence bound for FL
algorithms with arbitrary client sampling probabilities. Based on the bound, we
analytically establish the relationship between the total learning time and
sampling probabilities, which results in a non-convex optimization problem for
training time minimization. We design an efficient algorithm for learning the
unknown parameters in the convergence bound and develop a low-complexity
algorithm to approximately solve the non-convex problem. Experimental results
from both hardware prototype and simulation demonstrate that our proposed
sampling scheme significantly reduces the convergence time compared to several
baseline sampling schemes. Notably, our scheme in hardware prototype spends 73%
less time than the uniform sampling baseline for reaching the same target loss.

    

### [[2112.11324] Satellite-Based Communications Security: A Survey on Threats, Solutions, and Research Challenges](http://arxiv.org/abs/2112.11324)


  Satellite-based Communication (SATCOM) systems are gaining renewed momentum
in Industry and Academia, thanks to innovative services introduced by leading
tech companies and the promising impact they can deliver towards the global
connectivity objective tackled by early 6G initiatives. On the one hand, the
emergence of new manufacturing processes and radio technologies promises to
reduce service costs while guaranteeing outstanding communication latency,
available bandwidth, flexibility, and coverage range. On the other hand,
cybersecurity techniques and solutions applied in SATCOM links should be
updated to reflect the enormous advancements in attacker capabilities
characterizing the last two decades. However, business urgency and
opportunities are leading operators towards challenging system trade-offs,
leading to an increased attack surface and a general relaxation of the
available security services.
This paper tackles the cited problem and presents a comprehensive survey on
the security threats, solutions, and challenges faced when deploying and
operating SATCOM systems. Specifically, we classify the literature on security
for SATCOM systems into two main branches, i.e., physical-layer security and
cryptography schemes. Then, we further identify specific research domains for
each of the identified branches, focusing on dedicated security issues,
including, e.g., physical-layer confidentiality, anti-jamming schemes,
anti-spoofing strategies, and quantum-based key distribution schemes. For each
of the above domains, we highlight the most essential techniques,
peculiarities, advantages, disadvantages, lessons learned, and future
directions. Finally, we also identify emerging research topics whose additional
investigation by Academia and Industry could further attract researchers and
investors, ultimately unleashing the potential behind ubiquitous satellite
communications.

    

### [[2112.11414] Covert Communications via Adversarial Machine Learning and Reconfigurable Intelligent Surfaces](http://arxiv.org/abs/2112.11414)


  By moving from massive antennas to antenna surfaces for software-defined
wireless systems, the reconfigurable intelligent surfaces (RISs) rely on arrays
of unit cells to control the scattering and reflection profiles of signals,
mitigating the propagation loss and multipath attenuation, and thereby
improving the coverage and spectral efficiency. In this paper, covert
communication is considered in the presence of the RIS. While there is an
ongoing transmission boosted by the RIS, both the intended receiver and an
eavesdropper individually try to detect this transmission using their own deep
neural network (DNN) classifiers. The RIS interaction vector is designed by
balancing two (potentially conflicting) objectives of focusing the transmitted
signal to the receiver and keeping the transmitted signal away from the
eavesdropper. To boost covert communications, adversarial perturbations are
added to signals at the transmitter to fool the eavesdropper's classifier while
keeping the effect on the receiver low. Results from different network
topologies show that adversarial perturbation and RIS interaction vector can be
jointly designed to effectively increase the signal detection accuracy at the
receiver while reducing the detection accuracy at the eavesdropper to enable
covert communications.

    

### [[2005.05321] Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers](http://arxiv.org/abs/2005.05321)


  This paper presents channel-aware adversarial attacks against deep
learning-based wireless signal classifiers. There is a transmitter that
transmits signals with different modulation types. A deep neural network is
used at each receiver to classify its over-the-air received signals to
modulation types. In the meantime, an adversary transmits an adversarial
perturbation (subject to a power budget) to fool receivers into making errors
in classifying signals that are received as superpositions of transmitted
signals and adversarial perturbations. First, these evasion attacks are shown
to fail when channels are not considered in designing adversarial
perturbations. Then, realistic attacks are presented by considering channel
effects from the adversary to each receiver. After showing that a channel-aware
attack is selective (i.e., it affects only the receiver whose channel is
considered in the perturbation design), a broadcast adversarial attack is
presented by crafting a common adversarial perturbation to simultaneously fool
classifiers at different receivers. The major vulnerability of modulation
classifiers to over-the-air adversarial attacks is shown by accounting for
different levels of information available about the channel, the transmitter
input, and the classifier model. Finally, a certified defense based on
randomized smoothing that augments training data with noise is introduced to
make the modulation classifier robust to adversarial perturbations.

    

### [[2112.10765] A Grid-Structured Model of Tubular Reactors](http://arxiv.org/abs/2112.10765)


  We propose a grid-like computational model of tubular reactors. The
architecture is inspired by the computations performed by solvers of partial
differential equations which describe the dynamics of the chemical process
inside a tubular reactor. The proposed model may be entirely based on the known
form of the partial differential equations or it may contain generic machine
learning components such as multi-layer perceptrons. We show that the proposed
model can be trained using limited amounts of data to describe the state of a
fixed-bed catalytic reactor. The trained model can reconstruct unmeasured
states such as the catalyst activity using the measurements of inlet
concentrations and temperatures along the reactor.

    

### [[2112.10767] GCN-Geo: A Graph Convolution Network-based Fine-grained IP Geolocation Framework](http://arxiv.org/abs/2112.10767)


  Classical fine-grained measurement-based IP geolocation algorithms often rely
on some specific linear delay-distance rules. This could cause unreliable
geolocation results in actual network environments where the delay-distance
relationship is non-linear. Recently, researchers begin to pay attention to
learning-based IP geolocation algorithms. These data-driven algorithms leverage
multi-layer perceptron (MLP) to model the network environments. They do not
need strong pre-assumptions about the linear delay-distance rule and are
capable to learn non-linear relationships. In theory, they should improve the
generalization ability of IP geolocation in different networks. However,
networks are fundamentally represented as graphs. MLP is not well suited to
model information structured as graphs. MLP-based IP geolocation methods treat
target IP addresses as isolated data instances and ignore the connection
information between targets. This would lead to suboptimal representations and
limit the geolocation performance.
Graph convolutional network (GCN) is an emerging deep learning method for
graph data presentation. In this work, we research how to model computer
networks for fine-grained IP geolocation with GCN. First, we formulate the IP
geolocation task as an attributed graph node regression problem. Then, a
GCN-based IP geolocation framework named GCN-Geo is proposed to predict the
location of each IP address. Finally, the experimental results in three
real-world datasets (New York State, Hong Kong, and Shanghai) show that the
proposed GCN-Geo framework clearly outperforms the state-of-art rule-based and
learning-based baselines on average error distance, median error distance and
max error distance. This verifies the potential of GCN in fine-grained IP
geolocation.

    

### [[2112.10768] Improving Learning-to-Defer Algorithms Through Fine-Tuning](http://arxiv.org/abs/2112.10768)


  The ubiquity of AI leads to situations where humans and AI work together,
creating the need for learning-to-defer algorithms that determine how to
partition tasks between AI and humans. We work to improve learning-to-defer
algorithms when paired with specific individuals by incorporating two
fine-tuning algorithms and testing their efficacy using both synthetic and
image datasets. We find that fine-tuning can pick up on simple human skill
patterns, but struggles with nuance, and we suggest future work that uses
robust semi-supervised to improve learning.

    

### [[2112.10769] Logarithmic Unbiased Quantization: Practical 4-bit Training in Deep Learning](http://arxiv.org/abs/2112.10769)


  Quantization of the weights and activations is one of the main methods to
reduce the computational footprint of Deep Neural Networks (DNNs) training.
Current methods enable 4-bit quantization of the forward phase. However, this
constitutes only a third of the training process. Reducing the computational
footprint of the entire training process requires the quantization of the
neural gradients, i.e., the loss gradients with respect to the outputs of
intermediate neural layers. In this work, we examine the importance of having
unbiased quantization in quantized neural network training, where to maintain
it, and how. Based on this, we suggest a $\textit{logarithmic unbiased
quantization}$ (LUQ) method to quantize both the forward and backward phase to
4-bit, achieving state-of-the-art results in 4-bit training without overhead.
For example, in ResNet50 on ImageNet, we achieved a degradation of 1.18%. We
further improve this to degradation of only 0.64% after a single epoch of high
precision fine-tuning combined with a variance reduction method -- both add
overhead comparable to previously suggested methods. Finally, we suggest a
method that uses the low precision format to avoid multiplications during
two-thirds of the training process, thus reducing by 5x the area used by the
multiplier.

    

### [[2112.10771] Efficient Tensor Robust PCA under Hybrid Model of Tucker and Tensor Train](http://arxiv.org/abs/2112.10771)


  Tensor robust principal component analysis (TRPCA) is a fundamental model in
machine learning and computer vision. Recently, tensor train (TT) decomposition
has been verified effective to capture the global low-rank correlation for
tensor recovery tasks. However, due to the large-scale tensor data in
real-world applications, previous TRPCA models often suffer from high
computational complexity. In this letter, we propose an efficient TRPCA under
hybrid model of Tucker and TT. Specifically, in theory we reveal that TT
nuclear norm (TTNN) of the original big tensor can be equivalently converted to
that of a much smaller tensor via a Tucker compression format, thereby
significantly reducing the computational cost of singular value decomposition
(SVD). Numerical experiments on both synthetic and real-world tensor data
verify the superiority of the proposed model.

    

### [[2112.10774] TFDPM: Attack detection for cyber-physical systems with diffusion probabilistic models](http://arxiv.org/abs/2112.10774)


  With the development of AIoT, data-driven attack detection methods for
cyber-physical systems (CPSs) have attracted lots of attention. However,
existing methods usually adopt tractable distributions to approximate data
distributions, which are not suitable for complex systems. Besides, the
correlation of the data in different channels does not attract sufficient
attention. To address these issues, we use energy-based generative models,
which are less restrictive on functional forms of the data distribution. In
addition, graph neural networks are used to explicitly model the correlation of
the data in different channels. In the end, we propose TFDPM, a general
framework for attack detection tasks in CPSs. It simultaneously extracts
temporal pattern and feature pattern given the historical data. Then extract
features are sent to a conditional diffusion probabilistic model. Predicted
values can be obtained with the conditional generative network and attacks are
detected based on the difference between predicted values and observed values.
In addition, to realize real-time detection, a conditional noise scheduling
network is proposed to accelerate the prediction process. Experimental results
show that TFDPM outperforms existing state-of-the-art attack detection methods.
The noise scheduling network increases the detection speed by three times.

    

### [[2112.10775] HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images](http://arxiv.org/abs/2112.10775)


  Multiple medical institutions collaboratively training a model using
federated learning (FL) has become a promising solution for maximizing the
potential of data-driven models, yet the non-independent and identically
distributed (non-iid) data in medical images is still an outstanding challenge
in real-world practice. The feature heterogeneity caused by diverse scanners or
protocols introduces a drift in the learning process, in both local (client)
and global (server) optimizations, which harms the convergence as well as model
performance. Many previous works have attempted to address the non-iid issue by
tackling the drift locally or globally, but how to jointly solve the two
essentially coupled drifts is still unclear. In this work, we concentrate on
handling both local and global drifts and introduce a new harmonizing framework
called HarmoFL. First, we propose to mitigate the local update drift by
normalizing amplitudes of images transformed into the frequency domain to mimic
a unified imaging setting, in order to generate a harmonized feature space
across local clients. Second, based on harmonized features, we design a client
weight perturbation guiding each local model to reach a flat optimum, where a
neighborhood area of the local optimal solution has a uniformly low loss.
Without any extra communication cost, the perturbation assists the global model
to optimize towards a converged optimal solution by aggregating several local
flat optima. We have theoretically analyzed the proposed method and empirically
conducted extensive experiments on three medical image classification and
segmentation tasks, showing that HarmoFL outperforms a set of recent
state-of-the-art methods with promising convergence behavior.

    

### [[2112.10789] Machine learning discovery of new phases in programmable quantum simulator snapshots](http://arxiv.org/abs/2112.10789)


  Machine learning has recently emerged as a promising approach for studying
complex phenomena characterized by rich datasets. In particular, data-centric
approaches lend to the possibility of automatically discovering structures in
experimental datasets that manual inspection may miss. Here, we introduce an
interpretable unsupervised-supervised hybrid machine learning approach, the
hybrid-correlation convolutional neural network (Hybrid-CCNN), and apply it to
experimental data generated using a programmable quantum simulator based on
Rydberg atom arrays. Specifically, we apply Hybrid-CCNN to analyze new quantum
phases on square lattices with programmable interactions. The initial
unsupervised dimensionality reduction and clustering stage first reveals five
distinct quantum phase regions. In a second supervised stage, we refine these
phase boundaries and characterize each phase by training fully interpretable
CCNNs and extracting the relevant correlations for each phase. The
characteristic spatial weightings and snippets of correlations specifically
recognized in each phase capture quantum fluctuations in the striated phase and
identify two previously undetected phases, the rhombic and boundary-ordered
phases. These observations demonstrate that a combination of programmable
quantum simulators with machine learning can be used as a powerful tool for
detailed exploration of correlated quantum states of matter.

    

### [[2112.10821] Natural language processing to identify lupus nephritis phenotype in electronic health records](http://arxiv.org/abs/2112.10821)


  Systemic lupus erythematosus (SLE) is a rare autoimmune disorder
characterized by an unpredictable course of flares and remission with diverse
manifestations. Lupus nephritis, one of the major disease manifestations of SLE
for organ damage and mortality, is a key component of lupus classification
criteria. Accurately identifying lupus nephritis in electronic health records
(EHRs) would therefore benefit large cohort observational studies and clinical
trials where characterization of the patient population is critical for
recruitment, study design, and analysis. Lupus nephritis can be recognized
through procedure codes and structured data, such as laboratory tests. However,
other critical information documenting lupus nephritis, such as histologic
reports from kidney biopsies and prior medical history narratives, require
sophisticated text processing to mine information from pathology reports and
clinical notes. In this study, we developed algorithms to identify lupus
nephritis with and without natural language processing (NLP) using EHR data. We
developed four algorithms: a rule-based algorithm using only structured data
(baseline algorithm) and three algorithms using different NLP models. The three
NLP models are based on regularized logistic regression and use different sets
of features including positive mention of concept unique identifiers (CUIs),
number of appearances of CUIs, and a mixture of three components respectively.
The baseline algorithm and the best performed NLP algorithm were external
validated on a dataset from Vanderbilt University Medical Center (VUMC). Our
best performing NLP model incorporating features from both structured data,
regular expression concepts, and mapped CUIs improved F measure in both the
NMEDW (0.41 vs 0.79) and VUMC (0.62 vs 0.96) datasets compared to the baseline
lupus nephritis algorithm.

    

### [[2112.10852] The effective noise of Stochastic Gradient Descent](http://arxiv.org/abs/2112.10852)


  Stochastic Gradient Descent (SGD) is the workhorse algorithm of deep learning
technology. At each step of the training phase, a mini batch of samples is
drawn from the training dataset and the weights of the neural network are
adjusted according to the performance on this specific subset of examples. The
mini-batch sampling procedure introduces a stochastic dynamics to the gradient
descent, with a non-trivial state-dependent noise. We characterize the
stochasticity of SGD and a recently-introduced variant, persistent SGD, in a
prototypical neural network model. In the under-parametrized regime, where the
final training error is positive, the SGD dynamics reaches a stationary state
and we define an effective temperature from the fluctuation-dissipation
theorem, computed from dynamical mean-field theory. We use the effective
temperature to quantify the magnitude of the SGD noise as a function of the
problem parameters. In the over-parametrized regime, where the training error
vanishes, we measure the noise magnitude of SGD by computing the average
distance between two replicas of the system with the same initialization and
two different realizations of SGD noise. We find that the two noise measures
behave similarly as a function of the problem parameters. Moreover, we observe
that noisier algorithms lead to wider decision boundaries of the corresponding
constraint satisfaction problem.

    

### [[2112.10877] AGPNet -- Autonomous Grading Policy Network](http://arxiv.org/abs/2112.10877)


  In this work, we establish heuristics and learning strategies for the
autonomous control of a dozer grading an uneven area studded with sand piles.
We formalize the problem as a Markov Decision Process, design a simulation
which demonstrates agent-environment interactions and finally compare our
simulator to a real dozer prototype. We use methods from reinforcement
learning, behavior cloning and contrastive learning to train a hybrid policy.
Our trained agent, AGPNet, reaches human-level performance and outperforms
current state-of-the-art machine learning methods for the autonomous grading
task. In addition, our agent is capable of generalizing from random scenarios
to unseen real world problems.

    

### [[2112.10878] Enabling NAS with Automated Super-Network Generation](http://arxiv.org/abs/2112.10878)


  Recent Neural Architecture Search (NAS) solutions have produced impressive
results training super-networks and then deriving subnetworks, a.k.a. child
models that outperform expert-crafted models from a pre-defined search space.
Efficient and robust subnetworks can be selected for resource-constrained edge
devices, allowing them to perform well in the wild. However, constructing
super-networks for arbitrary architectures is still a challenge that often
prevents the adoption of these approaches. To address this challenge, we
present BootstrapNAS, a software framework for automatic generation of
super-networks for NAS. BootstrapNAS takes a pre-trained model from a popular
architecture, e.g., ResNet- 50, or from a valid custom design, and
automatically creates a super-network out of it, then uses state-of-the-art NAS
techniques to train the super-network, resulting in subnetworks that
significantly outperform the given pre-trained model. We demonstrate the
solution by generating super-networks from arbitrary model repositories and
make available the resulting super-networks for reproducibility of the results.

    

### [[2112.10884] Learning Bayesian Networks in the Presence of Structural Side Information](http://arxiv.org/abs/2112.10884)


  We study the problem of learning a Bayesian network (BN) of a set of
variables when structural side information about the system is available. It is
well known that learning the structure of a general BN is both computationally
and statistically challenging. However, often in many applications, side
information about the underlying structure can potentially reduce the learning
complexity. In this paper, we develop a recursive constraint-based algorithm
that efficiently incorporates such knowledge (i.e., side information) into the
learning process. In particular, we study two types of structural side
information about the underlying BN: (I) an upper bound on its clique number is
known, or (II) it is diamond-free. We provide theoretical guarantees for the
learning algorithms, including the worst-case number of tests required in each
scenario. As a consequence of our work, we show that bounded treewidth BNs can
be learned with polynomial complexity. Furthermore, we evaluate the performance
and the scalability of our algorithms in both synthetic and real-world
structures and show that they outperform the state-of-the-art structure
learning algorithms.

    

### [[2112.10889] Surrogate Model for Shallow Water Equations Solvers with Deep Learning](http://arxiv.org/abs/2112.10889)


  Shallow water equations are the foundation of most models for flooding and
river hydraulics analysis. These physics-based models are usually expensive and
slow to run, thus not suitable for real-time prediction or parameter inversion.
An attractive alternative is surrogate model. This work introduces an
efficient, accurate, and flexible surrogate model, NN-p2p, based on deep
learning and it can make point-to-point predictions on unstructured or
irregular meshes. The new method was evaluated and compared against existing
methods based on convolutional neural networks (CNNs), which can only make
image-to-image predictions on structured or regular meshes. In NN-p2p, the
input includes both spatial coordinates and boundary features that can describe
the geometry of hydraulic structures, such as bridge piers. All surrogate
models perform well in predicting flow around different types of piers in the
training domain. However, only NN-p2p works well when spatial extrapolation is
performed. The limitations of CNN-based methods are rooted in their
raster-image nature which cannot capture boundary geometry and flow features
exactly, which are of paramount importance to fluid dynamics. NN-p2p also has
good performance in predicting flow around piers unseen by the neural network.
The NN-p2p model also respects conservation laws more strictly. The application
of the proposed surrogate model was demonstrated by calculating the drag
coefficient $C_D$ for piers and a new linear relationship between $C_D$ and the
logarithmic transformation of pier's length/width ratio was discovered.

    

### [[2112.10893] VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements](http://arxiv.org/abs/2112.10893)


  Automatically locating vulnerable statements in source code is crucial to
assure software security and alleviate developers' debugging efforts. This
becomes even more important in today's software ecosystem, where vulnerable
code can flow easily and unwittingly within and across software repositories
like GitHub. Across such millions of lines of code, traditional static and
dynamic approaches struggle to scale. Although existing machine-learning-based
approaches look promising in such a setting, most work detects vulnerable code
at a higher granularity -- at the method or file level. Thus, developers still
need to inspect a significant amount of code to locate the vulnerable
statement(s) that need to be fixed.
This paper presents VELVET, a novel ensemble learning approach to locate
vulnerable statements. Our model combines graph-based and sequence-based neural
networks to successfully capture the local and global context of a program
graph and effectively understand code semantics and vulnerable patterns. To
study VELVET's effectiveness, we use an off-the-shelf synthetic dataset and a
recently published real-world dataset. In the static analysis setting, where
vulnerable functions are not detected in advance, VELVET achieves 4.5x better
performance than the baseline static analyzers on the real-world data. For the
isolated vulnerability localization task, where we assume the vulnerability of
a function is known while the specific vulnerable statement is unknown, we
compare VELVET with several neural networks that also attend to local and
global context of code. VELVET achieves 99.6% and 43.6% top-1 accuracy over
synthetic data and real-world data, respectively, outperforming the baseline
deep-learning models by 5.3-29.0%.

    

### [[2112.10894] Subject-Independent Drowsiness Recognition from Single-Channel EEG with an Interpretable CNN-LSTM model](http://arxiv.org/abs/2112.10894)


  For EEG-based drowsiness recognition, it is desirable to use
subject-independent recognition since conducting calibration on each subject is
time-consuming. In this paper, we propose a novel Convolutional Neural Network
(CNN)-Long Short-Term Memory (LSTM) model for subject-independent drowsiness
recognition from single-channel EEG signals. Different from existing deep
learning models that are mostly treated as black-box classifiers, the proposed
model can explain its decisions for each input sample by revealing which parts
of the sample contain important features identified by the model for
classification. This is achieved by a visualization technique by taking
advantage of the hidden states output by the LSTM layer. Results show that the
model achieves an average accuracy of 72.97% on 11 subjects for leave-one-out
subject-independent drowsiness recognition on a public dataset, which is higher
than the conventional baseline methods of 55.42%-69.27%, and state-of-the-art
deep learning methods. Visualization results show that the model has discovered
meaningful patterns of EEG signals related to different mental states across
different subjects.

    

### [[2112.10898] Load-balanced Gather-scatter Patterns for Sparse Deep Neural Networks](http://arxiv.org/abs/2112.10898)


  Deep neural networks (DNNs) have been proven to be effective in solving many
real-life problems, but its high computation cost prohibits those models from
being deployed to edge devices. Pruning, as a method to introduce zeros to
model weights, has shown to be an effective method to provide good trade-offs
between model accuracy and computation efficiency, and is a widely-used method
to generate compressed models. However, the granularity of pruning makes
important trade-offs. At the same sparsity level, a coarse-grained structured
sparse pattern is more efficient on conventional hardware but results in worse
accuracy, while a fine-grained unstructured sparse pattern can achieve better
accuracy but is inefficient on existing hardware.
On the other hand, some modern processors are equipped with fast on-chip
scratchpad memories and gather/scatter engines that perform indirect load and
store operations on such memories. In this work, we propose a set of novel
sparse patterns, named gather-scatter (GS) patterns, to utilize the scratchpad
memories and gather/scatter engines to speed up neural network inferences.
Correspondingly, we present a compact sparse format. The proposed set of sparse
patterns, along with a novel pruning methodology, address the load imbalance
issue and result in models with quality close to unstructured sparse models and
computation efficiency close to structured sparse models. Our experiments show
that GS patterns consistently make better trade-offs between accuracy and
computation efficiency compared to conventional structured sparse patterns. GS
patterns can reduce the runtime of the DNN components by two to three times at
the same accuracy levels. This is confirmed on three different deep learning
tasks and popular models, namely, GNMT for machine translation, ResNet50 for
image recognition, and Japser for acoustic speech recognition.

    

### [[2112.10912] Common Misconceptions about Population Data](http://arxiv.org/abs/2112.10912)


  Databases covering all individuals of a population are increasingly used for
research studies in domains ranging from public health to the social sciences.
There is also growing interest by governments and businesses to use population
data to support data-driven decision making. The massive size of such databases
is often mistaken as a guarantee for valid inferences on the population of
interest. However, population data have characteristics that make them
challenging to use, including various assumptions being made how such data were
collected and what types of processing have been applied to them. Furthermore,
the full potential of population data can often only be unlocked when such data
are linked to other databases, a process that adds fresh challenges. This
article discusses a diverse range of misconceptions about population data that
we believe anybody who works with such data needs to be aware of. Many of these
misconceptions are not well documented in scientific publications but only
discussed anecdotally among researchers and practitioners. We conclude with a
set of recommendations for inference when using population data.

    

### [[2112.10919] Provable Hierarchical Lifelong Learning with a Sketch-based Modular Architecture](http://arxiv.org/abs/2112.10919)


  We propose a modular architecture for the lifelong learning of hierarchically
structured tasks. Specifically, we prove that our architecture is theoretically
able to learn tasks that can be solved by functions that are learnable given
access to functions for other, previously learned tasks as subroutines. We
empirically show that some tasks that we can learn in this way are not learned
by standard training methods in practice; indeed, prior work suggests that some
such tasks cannot be learned by any efficient method without the aid of the
simpler tasks. We also consider methods for identifying the tasks
automatically, without relying on explicitly given indicators.

    

### [[2112.10930] Compact Multi-level Sparse Neural Networks with Input Independent Dynamic Rerouting](http://arxiv.org/abs/2112.10930)


  Deep neural networks (DNNs) have shown to provide superb performance in many
real life applications, but their large computation cost and storage
requirement have prevented them from being deployed to many edge and
internet-of-things (IoT) devices. Sparse deep neural networks, whose majority
weight parameters are zeros, can substantially reduce the computation
complexity and memory consumption of the models. In real-use scenarios, devices
may suffer from large fluctuations of the available computation and memory
resources under different environment, and the quality of service (QoS) is
difficult to maintain due to the long tail inferences with large latency.
Facing the real-life challenges, we propose to train a sparse model that
supports multiple sparse levels. That is, a hierarchical structure of weights
are satisfied such that the locations and the values of the non-zero parameters
of the more-sparse sub-model area subset of the less-sparse sub-model. In this
way, one can dynamically select the appropriate sparsity level during
inference, while the storage cost is capped by the least sparse sub-model. We
have verified our methodologies on a variety of DNN models and tasks, including
the ResNet-50, PointNet++, GNMT, and graph attention networks. We obtain sparse
sub-models with an average of 13.38% weights and 14.97% FLOPs, while the
accuracies are as good as their dense counterparts. More-sparse sub-models with
5.38% weights and 4.47% of FLOPs, which are subsets of the less-sparse ones,
can be obtained with only 3.25% relative accuracy loss.

    

### [[2112.10935] Nearly Optimal Policy Optimization with Stable at Any Time Guarantee](http://arxiv.org/abs/2112.10935)


  Policy optimization methods are one of the most widely used classes of
Reinforcement Learning (RL) algorithms. However, theoretical understanding of
these methods remains insufficient. Even in the episodic (time-inhomogeneous)
tabular setting, the state-of-the-art theoretical result of policy-based method
in \citet{shani2020optimistic} is only $\tilde{O}(\sqrt{S^2AH^4K})$ where $S$
is the number of states, $A$ is the number of actions, $H$ is the horizon, and
$K$ is the number of episodes, and there is a $\sqrt{SH}$ gap compared with the
information theoretic lower bound $\tilde{\Omega}(\sqrt{SAH^3K})$. To bridge
such a gap, we propose a novel algorithm Reference-based Policy Optimization
with Stable at Any Time guarantee (\algnameacro), which features the property
"Stable at Any Time". We prove that our algorithm achieves
$\tilde{O}(\sqrt{SAH^3K} + \sqrt{AH^4})$ regret. When $S > H$, our algorithm is
minimax optimal when ignoring logarithmic factors. To our best knowledge,
RPO-SAT is the first computationally efficient, nearly minimax optimal
policy-based algorithm for tabular RL.

    

### [[2112.10944] Reinforcement Learning based Sequential Batch-sampling for Bayesian Optimal Experimental Design](http://arxiv.org/abs/2112.10944)


  Engineering problems that are modeled using sophisticated mathematical
methods or are characterized by expensive-to-conduct tests or experiments, are
encumbered with limited budget or finite computational resources. Moreover,
practical scenarios in the industry, impose restrictions, based on logistics
and preference, on the manner in which the experiments can be conducted. For
example, material supply may enable only a handful of experiments in a
single-shot or in the case of computational models one may face significant
wait-time based on shared computational resources. In such scenarios, one
usually resorts to performing experiments in a manner that allows for
maximizing one's state-of-knowledge while satisfying the above mentioned
practical constraints. Sequential design of experiments (SDOE) is a popular
suite of methods, that has yielded promising results in recent years across
different engineering and practical problems. A common strategy, that leverages
Bayesian formalism is the Bayesian SDOE, which usually works best in the
one-step-ahead or myopic scenario of selecting a single experiment at each step
of a sequence of experiments. In this work, we aim to extend the SDOE strategy,
to query the experiment or computer code at a batch of inputs. To this end, we
leverage deep reinforcement learning (RL) based policy gradient methods, to
propose batches of queries that are selected taking into account entire budget
in hand. The algorithm retains the sequential nature, inherent in the SDOE,
while incorporating elements of reward based on task from the domain of deep
RL. A unique capability of the proposed methodology is its ability to be
applied to multiple tasks, for example optimization of a function, once its
trained. We demonstrate the performance of the proposed algorithm on a
synthetic problem, and a challenging high-dimensional engineering problem.

    

### [[2112.10947] The entropic barrier is $n$-self-concordant](http://arxiv.org/abs/2112.10947)


  For any convex body $K \subseteq \mathbb R^n$, S. Bubeck and R. Eldan
introduced the entropic barrier on $K$ and showed that it is a $(1+o(1)) \,
n$-self-concordant barrier. In this note, we observe that the optimal bound of
$n$ on the self-concordance parameter holds as a consequence of the dimensional
Brascamp-Lieb inequality.

    

### [[2112.10950] Augmented Contrastive Self-Supervised Learning for Audio Invariant Representations](http://arxiv.org/abs/2112.10950)


  Improving generalization is a major challenge in audio classification due to
labeled data scarcity. Self-supervised learning (SSL) methods tackle this by
leveraging unlabeled data to learn useful features for downstream
classification tasks. In this work, we propose an augmented contrastive SSL
framework to learn invariant representations from unlabeled data. Our method
applies various perturbations to the unlabeled input data and utilizes
contrastive learning to learn representations robust to such perturbations.
Experimental results on the Audioset and DESED datasets show that our framework
significantly outperforms state-of-the-art SSL and supervised learning methods
on sound/event classification tasks.

    

### [[2112.10955] Joint Learning of Linear Time-Invariant Dynamical Systems](http://arxiv.org/abs/2112.10955)


  Learning the parameters of a linear time-invariant dynamical system (LTIDS)
is a problem of current interest. In many applications, one is interested in
jointly learning the parameters of multiple related LTIDS, which remains
unexplored to date. To that end, we develop a joint estimator for learning the
transition matrices of LTIDS that share common basis matrices. Further, we
establish finite-time error bounds that depend on the underlying sample size,
dimension, number of tasks, and spectral properties of the transition matrices.
The results are obtained under mild regularity assumptions and showcase the
gains from pooling information across LTIDS, in comparison to learning each
system separately. We also study the impact of misspecifying the joint
structure of the transition matrices and show that the established results are
robust in the presence of moderate misspecifications.

    

### [[2112.10961] Nonlinear Transform Source-Channel Coding for Semantic Communications](http://arxiv.org/abs/2112.10961)


  In this paper, we propose a new class of high-efficient deep joint
source-channel coding methods that can closely adapt to the source distribution
under the nonlinear transform, it can be collected under the name nonlinear
transform source-channel coding (NTSCC). In the considered model, the
transmitter first learns a nonlinear analysis transform to map the source data
into latent space, then transmits the latent representation to the receiver via
deep joint source-channel coding. Our model incorporates the nonlinear
transform as a strong prior to effectively extract the source semantic features
and provide side information for source-channel coding. Unlike existing
conventional deep joint source-channel coding methods, the proposed NTSCC
essentially learns both the source latent representation and an entropy model
as the prior on the latent representation. Accordingly, novel adaptive rate
transmission and hyperprior-aided codec refinement mechanisms are developed to
upgrade deep joint source-channel coding. The whole system design is formulated
as an optimization problem whose goal is to minimize the end-to-end
transmission rate-distortion performance under established perceptual quality
metrics. Across simple example sources and test image sources, we find that the
proposed NTSCC transmission method generally outperforms both the analog
transmission using the standard deep joint source-channel coding and the
classical separation-based digital transmission. Notably, the proposed NTSCC
method can potentially support future semantic communications due to its
vigorous content-aware ability.

    

### [[2112.10971] Differentiated uniformization: A new method for inferring Markov chains on combinatorial state spaces including stochastic epidemic models](http://arxiv.org/abs/2112.10971)


  Motivation: We consider continuous-time Markov chains that describe the
stochastic evolution of a dynamical system by a transition-rate matrix $Q$
which depends on a parameter $\theta$. Computing the probability distribution
over states at time $t$ requires the matrix exponential $\exp(tQ)$, and
inferring $\theta$ from data requires its derivative
$\partial\exp\!(tQ)/\partial\theta$. Both are challenging to compute when the
state space and hence the size of $Q$ is huge. This can happen when the state
space consists of all combinations of the values of several interacting
discrete variables. Often it is even impossible to store $Q$. However, when $Q$
can be written as a sum of tensor products, computing $\exp(tQ)$ becomes
feasible by the uniformization method, which does not require explicit storage
of $Q$.
Results: Here we provide an analogous algorithm for computing
$\partial\exp\!(tQ)/\partial\theta$, the differentiated uniformization method.
We demonstrate our algorithm for the stochastic SIR model of epidemic spread,
for which we show that $Q$ can be written as a sum of tensor products. We
estimate monthly infection and recovery rates during the first wave of the
COVID-19 pandemic in Austria and quantify their uncertainty in a full Bayesian
analysis.
Availability: Implementation and data are available at
this https URL.

    

### [[2112.10974] What are Attackers after on IoT Devices? An approach based on a multi-phased multi-faceted IoT honeypot ecosystem and data clustering](http://arxiv.org/abs/2112.10974)


  The growing number of Internet of Things (IoT) devices makes it imperative to
be aware of the real-world threats they face in terms of cybersecurity. While
honeypots have been historically used as decoy devices to help
researchers/organizations gain a better understanding of the dynamic of threats
on a network and their impact, IoT devices pose a unique challenge for this
purpose due to the variety of devices and their physical connections. In this
work, by observing real-world attackers' behavior in a low-interaction honeypot
ecosystem, we (1) presented a new approach to creating a multi-phased,
multi-faceted honeypot ecosystem, which gradually increases the sophistication
of honeypots' interactions with adversaries, (2) designed and developed a
low-interaction honeypot for cameras that allowed researchers to gain a deeper
understanding of what attackers are targeting, and (3) devised an innovative
data analytics method to identify the goals of adversaries. Our honeypots have
been active for over three years. We were able to collect increasingly
sophisticated attack data in each phase. Furthermore, our data analytics points
to the fact that the vast majority of attack activities captured in the
honeypots share significant similarity, and can be clustered and grouped to
better understand the goals, patterns, and trends of IoT attacks in the wild.

    

### [[2112.10977] ACGNet: Action Complement Graph Network for Weakly-supervised Temporal Action Localization](http://arxiv.org/abs/2112.10977)


  Weakly-supervised temporal action localization (WTAL) in untrimmed videos has
emerged as a practical but challenging task since only video-level labels are
available. Existing approaches typically leverage off-the-shelf segment-level
features, which suffer from spatial incompleteness and temporal incoherence,
thus limiting their performance. In this paper, we tackle this problem from a
new perspective by enhancing segment-level representations with a simple yet
effective graph convolutional network, namely action complement graph network
(ACGNet). It facilitates the current video segment to perceive spatial-temporal
dependencies from others that potentially convey complementary clues,
implicitly mitigating the negative effects caused by the two issues above. By
this means, the segment-level features are more discriminative and robust to
spatial-temporal variations, contributing to higher localization accuracies.
More importantly, the proposed ACGNet works as a universal module that can be
flexibly plugged into different WTAL frameworks, while maintaining the
end-to-end training fashion. Extensive experiments are conducted on the
THUMOS'14 and ActivityNet1.2 benchmarks, where the state-of-the-art results
clearly demonstrate the superiority of the proposed approach.

    

### [[2112.10982] Generalized Few-Shot Semantic Segmentation: All You Need is Fine-Tuning](http://arxiv.org/abs/2112.10982)


  Generalized few-shot semantic segmentation was introduced to move beyond only
evaluating few-shot segmentation models on novel classes to include testing
their ability to remember base classes. While all approaches currently are
based on meta-learning, they perform poorly and saturate in learning after
observing only a few shots. We propose the first fine-tuning solution, and
demonstrate that it addresses the saturation problem while achieving
state-of-art results on two datasets, PASCAL-$5^i$ and COCO-$20^i$. We also
show it outperforms existing methods whether fine-tuning multiple final layers
or only the final layer. Finally, we present a triplet loss regularization that
shows how to redistribute the balance of performance between novel and base
categories so that there is a smaller gap between them.

    

### [[2112.10985] Learned ISTA with Error-based Thresholding for Adaptive Sparse Coding](http://arxiv.org/abs/2112.10985)


  The learned iterative shrinkage thresholding algorithm (LISTA) introduces
deep unfolding models with learnable thresholds in some shrinkage functions for
sparse coding. Drawing on some theoretical insights, we advocate an error-based
thresholding (EBT) mechanism for LISTA, which leverages a function of the
layer-wise reconstruction error to suggest an appropriate threshold value for
each observation on each layer. We show that the EBT mechanism well
disentangles the learnable parameters in the shrinkage functions from the
reconstruction errors, making them more adaptive to the various observations.
With rigorous theoretical analyses, we show that the proposed EBT can lead to a
faster convergence on the basis of LISTA and its variants, in addition to its
higher adaptivity. Extensive experimental results confirm our theoretical
analyses and verify the effectiveness of our methods.

    

### [[2112.10988] Mapping industrial poultry operations at scale with deep learning and aerial imagery](http://arxiv.org/abs/2112.10988)


  Concentrated Animal Feeding Operations (CAFOs) pose serious risks to air,
water, and public health, but have proven to be challenging to regulate. The
U.S. Government Accountability Office notes that a basic challenge is the lack
of comprehensive location information on CAFOs. We use the USDA's National
Agricultural Imagery Program (NAIP) 1m/pixel aerial imagery to detect poultry
CAFOs across the continental United States. We train convolutional neural
network (CNN) models to identify individual poultry barns and apply the best
performing model to over 42 TB of imagery to create the first national,
open-source dataset of poultry CAFOs. We validate the model predictions against
held-out validation set on poultry CAFO facility locations from 10 hand-labeled
counties in California and demonstrate that this approach has significant
potential to fill gaps in environmental monitoring.

    

### [[2112.11018] A Theoretical View of Linear Backpropagation and Its Convergence](http://arxiv.org/abs/2112.11018)


  Backpropagation is widely used for calculating gradients in deep neural
networks (DNNs). Applied often along with stochastic gradient descent (SGD) or
its variants, backpropagation is considered as a de-facto choice in a variety
of machine learning tasks including DNN training and adversarial
attack/defense. Recently, a linear variant of BP named LinBP was introduced for
generating more transferable adversarial examples for black-box adversarial
attacks, by Guo et al. Yet, it has not been theoretically studied and the
convergence analysis of such a method is lacking. This paper serves as a
complement and somewhat an extension to Guo et al.'s paper, by providing
theoretical analyses on LinBP in neural-network-involved learning tasks
including adversarial attack and model training. We demonstrate that, somewhat
surprisingly, LinBP can lead to faster convergence in these tasks in the same
hyper-parameter settings, compared to BP. We confirm our theoretical results
with extensive experiments.

    

### [[2112.11019] Mining Drifting Data Streams on a Budget: Combining Active Learning with Self-Labeling](http://arxiv.org/abs/2112.11019)


  Mining data streams poses a number of challenges, including the continuous
and non-stationary nature of data, the massive volume of information to be
processed and constraints put on the computational resources. While there is a
number of supervised solutions proposed for this problem in the literature,
most of them assume that access to the ground truth (in form of class labels)
is unlimited and such information can be instantly utilized when updating the
learning system. This is far from being realistic, as one must consider the
underlying cost of acquiring labels. Therefore, solutions that can reduce the
requirements for ground truth in streaming scenarios are required. In this
paper, we propose a novel framework for mining drifting data streams on a
budget, by combining information coming from active learning and self-labeling.
We introduce several strategies that can take advantage of both intelligent
instance selection and semi-supervised procedures, while taking into account
the potential presence of concept drift. Such a hybrid approach allows for
efficient exploration and exploitation of streaming data structures within
realistic labeling budgets. Since our framework works as a wrapper, it may be
applied with different learning algorithms. Experimental study, carried out on
a diverse set of real-world data streams with various types of concept drift,
proves the usefulness of the proposed strategies when dealing with highly
limited access to class labels. The presented hybrid approach is especially
feasible when one cannot increase a budget for labeling or replace an
inefficient classifier. We deliver a set of recommendations regarding areas of
applicability for our strategies.

    

### [[2112.11022] Synthetic Data and Simulators for Recommendation Systems: Current State and Future Directions](http://arxiv.org/abs/2112.11022)


  Synthetic data and simulators have the potential to markedly improve the
performance and robustness of recommendation systems. These approaches have
already had a beneficial impact in other machine-learning driven fields. We
identify and discuss a key trade-off between data fidelity and privacy in the
past work on synthetic data and simulators for recommendation systems. For the
important use case of predicting algorithm rankings on real data from synthetic
data, we provide motivation and current successes versus limitations. Finally
we outline a number of exciting future directions for recommendation systems
that we believe deserve further attention and work, including mixing real and
synthetic data, feedback in dataset generation, robust simulations, and
privacy-preserving methods.

    

### [[2112.11027] More is Less: Inducing Sparsity via Overparameterization](http://arxiv.org/abs/2112.11027)


  In deep learning it is common to overparameterize the neural networks, that
is, to use more parameters than training samples. Quite surprisingly training
the neural network via (stochastic) gradient descent leads to models that
generalize very well, while classical statistics would suggest overfitting. In
order to gain understanding of this implicit bias phenomenon we study the
special case of sparse recovery (compressive sensing) which is of interest on
its own. More precisely, in order to reconstruct a vector from underdetermined
linear measurements, we introduce a corresponding overparameterized square loss
functional, where the vector to be reconstructed is deeply factorized into
several vectors. We show that, under a very mild assumption on the measurement
matrix, vanilla gradient flow for the overparameterized loss functional
converges to a solution of minimal $\ell_1$-norm. The latter is well-known to
promote sparse solutions. As a by-product, our results significantly improve
the sample complexity for compressive sensing in previous works. The theory
accurately predicts the recovery rate in numerical experiments. For the proofs,
we introduce the concept of {\textit{solution entropy}}, which bypasses the
obstacles caused by non-convexity and should be of independent interest.

    

### [[2112.11032] ANUBIS: A Provenance Graph-Based Framework for Advanced Persistent Threat Detection](http://arxiv.org/abs/2112.11032)


  We present ANUBIS, a highly effective machine learning-based APT detection
system. Our design philosophy for ANUBIS involves two principal components.
Firstly, we intend ANUBIS to be effectively utilized by cyber-response teams.
Therefore, prediction explainability is one of the main focuses of ANUBIS
design. Secondly, ANUBIS uses system provenance graphs to capture causality and
thereby achieve high detection performance. At the core of the predictive
capability of ANUBIS, there is a Bayesian Neural Network that can tell how
confident it is in its predictions. We evaluate ANUBIS against a recent APT
dataset (DARPA OpTC) and show that ANUBIS can detect malicious activity akin to
APT campaigns with high accuracy. Moreover, ANUBIS learns about high-level
patterns that allow it to explain its predictions to threat analysts. The high
predictive performance with explainable attack story reconstruction makes
ANUBIS an effective tool to use for enterprise cyber defense.

    

### [[2112.11040] Distributed Machine Learning and the Semblance of Trust](http://arxiv.org/abs/2112.11040)


  The utilisation of large and diverse datasets for machine learning (ML) at
scale is required to promote scientific insight into many meaningful problems.
However, due to data governance regulations such as GDPR as well as ethical
concerns, the aggregation of personal and sensitive data is problematic, which
prompted the development of alternative strategies such as distributed ML
(DML). Techniques such as Federated Learning (FL) allow the data owner to
maintain data governance and perform model training locally without having to
share their data. FL and related techniques are often described as
privacy-preserving. We explain why this term is not appropriate and outline the
risks associated with over-reliance on protocols that were not designed with
formal definitions of privacy in mind. We further provide recommendations and
examples on how such algorithms can be augmented to provide guarantees of
governance, security, privacy and verifiability for a general ML audience
without prior exposure to formal privacy techniques.

    

### [[2112.11041] Geometry-Aware Unsupervised Domain Adaptation](http://arxiv.org/abs/2112.11041)


  Unsupervised Domain Adaptation (UDA) aims to transfer the knowledge from the
labeled source domain to the unlabeled target domain in the presence of dataset
shift. Most existing methods cannot address the domain alignment and class
discrimination well, which may distort the intrinsic data structure for
downstream tasks (e.g., classification). To this end, we propose a novel
geometry-aware model to learn the transferability and discriminability
simultaneously via nuclear norm optimization. We introduce the domain coherence
and class orthogonality for UDA from the perspective of subspace geometry. The
domain coherence will ensure the model has a larger capacity for learning
separable representations, and class orthogonality will minimize the
correlation between clusters to alleviate the misalignment. So, they are
consistent and can benefit from each other. Besides, we provide a theoretical
insight into the norm-based learning literature in UDA, which ensures the
interpretability of our model. We show that the norms of domains and clusters
are expected to be larger and smaller to enhance the transferability and
discriminability, respectively. Extensive experimental results on standard UDA
datasets demonstrate the effectiveness of our theory and model.

    

### [[2112.11055] A Scalable Deep Reinforcement Learning Model for Online Scheduling Coflows of Multi-Stage Jobs for High Performance Computing](http://arxiv.org/abs/2112.11055)


  Coflow is a recently proposed networking abstraction to help improve the
communication performance of data-parallel computing jobs. In multi-stage jobs,
each job consists of multiple coflows and is represented by a Directed Acyclic
Graph (DAG). Efficiently scheduling coflows is critical to improve the
data-parallel computing performance in data centers. Compared with hand-tuned
scheduling heuristics, existing work DeepWeave [1] utilizes Reinforcement
Learning (RL) framework to generate highly-efficient coflow scheduling policies
automatically. It employs a graph neural network (GNN) to encode the job
information in a set of embedding vectors, and feeds a flat embedding vector
containing the whole job information to the policy network. However, this
method has poor scalability as it is unable to cope with jobs represented by
DAGs of arbitrary sizes and shapes, which requires a large policy network for
processing a high-dimensional embedding vector that is difficult to train. In
this paper, we first utilize a directed acyclic graph neural network (DAGNN) to
process the input and propose a novel Pipelined-DAGNN, which can effectively
speed up the feature extraction process of the DAGNN. Next, we feed the
embedding sequence composed of schedulable coflows instead of a flat embedding
of all coflows to the policy network, and output a priority sequence, which
makes the size of the policy network depend on only the dimension of features
instead of the product of dimension and number of nodes in the job's
DAG.Furthermore, to improve the accuracy of the priority scheduling policy, we
incorporate the Self-Attention Mechanism into a deep RL model to capture the
interaction between different parts of the embedding sequence to make the
output priority scores relevant. Based on this model, we then develop a coflow
scheduling algorithm for online multi-stage jobs.

    

### [[2112.11070] An Inference Approach To Question Answering Over Knowledge Graphs](http://arxiv.org/abs/2112.11070)


  Knowledge Graphs (KG) act as a great tool for holding distilled information
from large natural language text corpora. The problem of natural language
querying over knowledge graphs is essential for the human consumption of this
information. This problem is typically addressed by converting the natural
language query to a structured query and then firing the structured query on
the KG. Direct answering models over knowledge graphs in literature are very
few. The query conversion models and direct models both require specific
training data pertaining to the domain of the knowledge graph. In this work, we
convert the problem of natural language querying over knowledge graphs to an
inference problem over premise-hypothesis pairs. Using trained deep learning
models for the converted proxy inferencing problem, we provide the solution for
the original natural language querying problem. Our method achieves over 90%
accuracy on MetaQA dataset, beating the existing state-of-the-art. We also
propose a model for inferencing called Hierarchical Recurrent Path
Encoder(HRPE). The inferencing models can be fine-tuned to be used across
domains with less training data. Our approach does not require large
domain-specific training data for querying on new knowledge graphs from
different domains.

    

### [[2112.11071] Explanation of Machine Learning Models Using Shapley Additive Explanation and Application for Real Data in Hospital](http://arxiv.org/abs/2112.11071)


  When using machine learning techniques in decision-making processes, the
interpretability of the models is important. In the present paper, we adopted
the Shapley additive explanation (SHAP), which is based on fair profit
allocation among many stakeholders depending on their contribution, for
interpreting a gradient-boosting decision tree model using hospital data. For
better interpretability, we propose two novel techniques as follows: (1) a new
metric of feature importance using SHAP and (2) a technique termed feature
packing, which packs multiple similar features into one grouped feature to
allow an easier understanding of the model without reconstruction of the model.
We then compared the explanation results between the SHAP framework and
existing methods. In addition, we showed how the A/G ratio works as an
important prognostic factor for cerebral infarction using our hospital data and
proposed techniques.

    

### [[2112.11081] RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](http://arxiv.org/abs/2112.11081)


  Compared to convolutional layers, fully-connected (FC) layers are better at
modeling the long-range dependencies but worse at capturing the local patterns,
hence usually less favored for image recognition. In this paper, we propose a
methodology, Locality Injection, to incorporate local priors into an FC layer
via merging the trained parameters of a parallel conv kernel into the FC
kernel. Locality Injection can be viewed as a novel Structural
Re-parameterization method since it equivalently converts the structures via
transforming the parameters. Based on that, we propose a multi-layer-perceptron
(MLP) block named RepMLP Block, which uses three FC layers to extract features,
and a novel architecture named RepMLPNet. The hierarchical design distinguishes
RepMLPNet from the other concurrently proposed vision MLPs. As it produces
feature maps of different levels, it qualifies as a backbone model for
downstream tasks like semantic segmentation. Our results reveal that 1)
Locality Injection is a general methodology for MLP models; 2) RepMLPNet has
favorable accuracy-efficiency trade-off compared to the other MLPs; 3)
RepMLPNet is the first MLP that seamlessly transfer to Cityscapes semantic
segmentation. The code and models are available at
this https URL.

    

### [[2112.11085] Can We Use Neural Regularization to Solve Depth Super-Resolution?](http://arxiv.org/abs/2112.11085)


  Depth maps captured with commodity sensors often require super-resolution to
be used in applications. In this work we study a super-resolution approach
based on a variational problem statement with Tikhonov regularization where the
regularizer is parametrized with a deep neural network. This approach was
previously applied successfully in photoacoustic tomography. We experimentally
show that its application to depth map super-resolution is difficult, and
provide suggestions about the reasons for that.

    

### [[2112.11099] High pressure hydrogen by machine learning and quantum Monte Carlo](http://arxiv.org/abs/2112.11099)


  We have developed a technique combining the accuracy of quantum Monte Carlo
in describing the electron correlation with the efficiency of a machine
learning potential (MLP). We use kernel linear regression in combination with
SOAP (Smooth Overlap Atomic Position) approach, implemented here in a very
efficient way. The key ingredients are: i) a sparsification technique, based on
farthest point sampling, ensuring generality and transferability of our MLPs
and ii) the so called $\Delta$-learning, allowing a small training data set, a
fundamental property for highly accurate but computationally demanding
calculations, such as the ones based on quantum Monte Carlo. As a first
application we present a benchmark study of the liquid-liquid transition of
high-pressure hydrogen and show the quality of our MLP, by emphasizing the
importance of high accuracy for this very debated subject, where experiments
are difficult in the lab, and theory is still far from being conclusive.

    

### [[2112.11111] Developing and Validating Semi-Markov Occupancy Generative Models: A Technical Report](http://arxiv.org/abs/2112.11111)


  This report documents recent technical work on developing and validating
stochastic occupancy models in commercial buildings, performed by the Pacific
Northwest National Laboratory (PNNL) as part of the Sensor Impact Evaluation
and Verification project under the U.S. Department of Energy (DOE) Building
Technologies Office (BTO). In this report, we present our work on developing
and validating inhomogeneous semi-Markov chain models for generating sequences
of zone-level occupancy presence and occupancy counts in a commercial building.
Real datasets are used to learn and validate the generative occupancy models.
Relevant metrics such as normalized Jensen-Shannon distance (NJSD) are used to
demonstrate the ability of the models to express realistic occupancy behavioral
patterns.

    

### [[2112.11115] Soft Actor-Critic with Cross-Entropy Policy Optimization](http://arxiv.org/abs/2112.11115)


  Soft Actor-Critic (SAC) is one of the state-of-the-art off-policy
reinforcement learning (RL) algorithms that is within the maximum entropy based
RL framework. SAC is demonstrated to perform very well in a list of continous
control tasks with good stability and robustness. SAC learns a stochastic
Gaussian policy that can maximize a trade-off between total expected reward and
the policy entropy. To update the policy, SAC minimizes the KL-Divergence
between the current policy density and the soft value function density.
Reparameterization trick is then used to obtain the approximate gradient of
this divergence. In this paper, we propose Soft Actor-Critic with Cross-Entropy
Policy Optimization (SAC-CEPO), which uses Cross-Entropy Method (CEM) to
optimize the policy network of SAC. The initial idea is to use CEM to
iteratively sample the closest distribution towards the soft value function
density and uses the resultant distribution as a target to update the policy
network. For the purpose of reducing the computational complexity, we also
introduce a decoupled policy structure that decouples the Gaussian policy into
one policy that learns the mean and one other policy that learns the deviation
such that only the mean policy is trained by CEM. We show that this decoupled
policy structure does converge to a optimal and we also demonstrate by
experiments that SAC-CEPO achieves competitive performance against the original
SAC.

    

### [[2112.11134] FedPOIRec: Privacy Preserving Federated POI Recommendation with Social Influence](http://arxiv.org/abs/2112.11134)


  With the growing number of Location-Based Social Networks, privacy preserving
location prediction has become a primary task for helping users discover new
points-of-interest (POIs). Traditional systems consider a centralized approach
that requires the transmission and collection of users' private data. In this
work, we present FedPOIRec, a privacy preserving federated learning approach
enhanced with features from users' social circles for top-$N$ POI
recommendations. First, the FedPOIRec framework is built on the principle that
local data never leave the owner's device, while the local updates are blindly
aggregated by a parameter server. Second, the local recommenders get
personalized by allowing users to exchange their learned parameters, enabling
knowledge transfer among friends. To this end, we propose a privacy preserving
protocol for integrating the preferences of a user's friends after the
federated computation, by exploiting the properties of the CKKS fully
homomorphic encryption scheme. To evaluate FedPOIRec, we apply our approach
into five real-world datasets using two recommendation models. Extensive
experiments demonstrate that FedPOIRec achieves comparable recommendation
quality to centralized approaches, while the social integration protocol incurs
low computation and communication overhead on the user side.

    

### [[2112.11136] Adversarial Gradient Driven Exploration for Deep Click-Through Rate Prediction](http://arxiv.org/abs/2112.11136)


  Nowadays, data-driven deep neural models have already shown remarkable
progress on Click-through Rate (CTR) prediction. Unfortunately, the
effectiveness of such models may fail when there are insufficient data. To
handle this issue, researchers often adopt exploration strategies to examine
items based on the estimated reward, e.g., UCB or Thompson Sampling. In the
context of Exploitation-and-Exploration for CTR prediction, recent studies have
attempted to utilize the prediction uncertainty along with model prediction as
the reward score. However, we argue that such an approach may make the final
ranking score deviate from the original distribution, and thereby affect model
performance in the online system. In this paper, we propose a novel exploration
method called \textbf{A}dversarial \textbf{G}radient Driven
\textbf{E}xploration (AGE). Specifically, we propose a Pseudo-Exploration
Module to simulate the gradient updating process, which can approximate the
influence of the samples of to-be-explored items for the model. In addition,
for better exploration efficiency, we propose an Dynamic Threshold Unit to
eliminate the effects of those samples with low potential CTR. The
effectiveness of our approach was demonstrated on an open-access academic
dataset. Meanwhile, AGE has also been deployed in a real-world display
advertising platform and all online metrics have been significantly improved.

    

### [[2112.11153] PONet: Robust 3D Human Pose Estimation via Learning Orientations Only](http://arxiv.org/abs/2112.11153)


  Conventional 3D human pose estimation relies on first detecting 2D body
keypoints and then solving the 2D to 3D correspondence problem.Despite the
promising results, this learning paradigm is highly dependent on the quality of
the 2D keypoint detector, which is inevitably fragile to occlusions and
out-of-image this http URL this paper,we propose a novel Pose Orientation Net
(PONet) that is able to robustly estimate 3D pose by learning orientations
only, hence bypassing the error-prone keypoint detector in the absence of image
evidence. For images with partially invisible limbs, PONet estimates the 3D
orientation of these limbs by taking advantage of the local image evidence to
recover the 3D pose.Moreover, PONet is competent to infer full 3D poses even
from images with completely invisible limbs, by exploiting the orientation
correlation between visible limbs to complement the estimated poses,further
improving the robustness of 3D pose estimation.We evaluate our method on
multiple datasets, including Human3.6M, MPII, MPI-INF-3DHP, and 3DPW. Our
method achieves results on par with state-of-the-art techniques in ideal
settings, yet significantly eliminates the dependency on keypoint detectors and
the corresponding computation burden. In highly challenging scenarios, such as
truncation and erasing, our method performs very robustly and yields much
superior results as compared to state of the art,demonstrating its potential
for real-world applications.

    

### [[2112.11157] Integral representations of shallow neural network with Rectified Power Unit activation function](http://arxiv.org/abs/2112.11157)


  In this effort, we derive a formula for the integral representation of a
shallow neural network with the Rectified Power Unit activation function.
Mainly, our first result deals with the univariate case of representation
capability of RePU shallow networks. The multidimensional result in this paper
characterizes the set of functions that can be represented with bounded norm
and possibly unbounded width.

    

### [[2112.11161] Manifold learning via quantum dynamics](http://arxiv.org/abs/2112.11161)


  We introduce an algorithm for computing geodesics on sampled manifolds that
relies on simulation of quantum dynamics on a graph embedding of the sampled
data. Our approach exploits classic results in semiclassical analysis and the
quantum-classical correspondence, and forms a basis for techniques to learn the
manifold from which a dataset is sampled, and subsequently for nonlinear
dimensionality reduction of high-dimensional datasets. We illustrate the new
algorithm with data sampled from model manifolds and also by a clustering
demonstration based on COVID-19 mobility data. Finally, our method reveals
interesting connections between the discretization provided by data sampling
and quantization.

    

### [[2112.11167] Hierarchical Over-the-Air Federated Edge Learning](http://arxiv.org/abs/2112.11167)


  Federated learning (FL) over wireless communication channels, specifically,
over-the-air (OTA) model aggregation framework is considered. In OTA wireless
setups, the adverse channel effects can be alleviated by increasing the number
of receive antennas at the parameter server (PS), which performs model
aggregation. However, the performance of OTA FL is limited by the presence of
mobile users (MUs) located far away from the PS. In this paper, to mitigate
this limitation, we propose hierarchical over-the-air federated learning
(HOTAFL), which utilizes intermediary servers (IS) to form clusters near MUs.
We provide a convergence analysis for the proposed setup, and demonstrate
through theoretical and experimental results that local aggregation in each
cluster before global aggregation leads to a better performance and faster
convergence than OTA FL.

    

### [[2112.11172] Dynamically Stable Poincaré Embeddings for Neural Manifolds](http://arxiv.org/abs/2112.11172)


  In a Riemannian manifold, the Ricci flow is a partial differential equation
for evolving the metric to become more regular. We hope that topological
structures from such metrics may be used to assist in the tasks of machine
learning. However, this part of the work is still missing. In this paper, we
bridge this gap between the Ricci flow and deep neural networks by dynamically
stable Poincaré embeddings for neural manifolds. As a result, we prove that,
if initial metrics have an $L^2$-norm perturbation which deviates from the
Hyperbolic metric on the Poincaré ball, the scaled Ricci-DeTurck flow of such
metrics smoothly and exponentially converges to the Hyperbolic metric.
Specifically, the role of the Ricci flow is to serve as naturally evolving to
the stable Poincaré ball that will then be mapped back to the Euclidean
space. For such dynamically stable neural manifolds under the Ricci flow, the
convergence of neural networks embedded with such manifolds is not susceptible
to perturbations. And we show that such Ricci flow assisted neural networks
outperform with their all Euclidean versions on image classification tasks
(CIFAR datasets).

    

### [[2112.11174] AutoCTS: Automated Correlated Time Series Forecasting -- Extended Version](http://arxiv.org/abs/2112.11174)


  Correlated time series (CTS) forecasting plays an essential role in many
cyber-physical systems, where multiple sensors emit time series that capture
interconnected processes. Solutions based on deep learning that deliver
state-of-the-art CTS forecasting performance employ a variety of
spatio-temporal (ST) blocks that are able to model temporal dependencies and
spatial correlations among time series. However, two challenges remain. First,
ST-blocks are designed manually, which is time consuming and costly. Second,
existing forecasting models simply stack the same ST-blocks multiple times,
which limits the model potential. To address these challenges, we propose
AutoCTS that is able to automatically identify highly competitive ST-blocks as
well as forecasting models with heterogeneous ST-blocks connected using diverse
topologies, as opposed to the same ST-blocks connected using simple stacking.
Specifically, we design both a micro and a macro search space to model possible
architectures of ST-blocks and the connections among heterogeneous ST-blocks,
and we provide a search strategy that is able to jointly explore the search
spaces to identify optimal forecasting models. Extensive experiments on eight
commonly used CTS forecasting benchmark datasets justify our design choices and
demonstrate that AutoCTS is capable of automatically discovering forecasting
models that outperform state-of-the-art human-designed models. This is an
extended version of ``AutoCTS: Automated Correlated Time Series Forecasting'',
to appear in PVLDB 2022.

    

### [[2112.11187] Predicting infections in the Covid-19 pandemic -- lessons learned](http://arxiv.org/abs/2112.11187)


  Throughout the Covid-19 pandemic, a significant amount of effort had been put
into developing techniques that predict the number of infections under various
assumptions about the public policy and non-pharmaceutical interventions. While
both the available data and the sophistication of the AI models and available
computing power exceed what was available in previous years, the overall
success of prediction approaches was very limited. In this paper, we start from
prediction algorithms proposed for XPrize Pandemic Response Challenge and
consider several directions that might allow their improvement. Then, we
investigate their performance over medium-term predictions extending over
several months. We find that augmenting the algorithms with additional
information about the culture of the modeled region, incorporating traditional
compartmental models and up-to-date deep learning architectures can improve the
performance for short term predictions, the accuracy of medium-term predictions
is still very low and a significant amount of future research is needed to make
such models a reliable component of a public policy toolbox.

    

### [[2112.11188] Diagnostic Assessment Generation via Combinatorial Search](http://arxiv.org/abs/2112.11188)


  Initial assessment tests are crucial in capturing learner knowledge states in
a consistent manner. Aside from crafting questions itself, putting together
relevant problems to form a question sheet is also a time-consuming process. In
this work, we present a generic formulation of question assembly and a genetic
algorithm based method that can generate assessment tests from raw
problem-solving history. First, we estimate the learner-question knowledge
matrix (snapshot). Each matrix element stands for the probability that a
learner correctly answers a specific question. We formulate the task as a
combinatorial search over this snapshot. To ensure representative and
discriminative diagnostic tests, questions are selected (1) that has a low root
mean squared error against the whole question pool and (2) high standard
deviation among learner performances. Experimental results show that the
proposed method outperforms greedy and random baseline by a large margin in one
private dataset and four public datasets. We also performed qualitative
analysis on the generated assessment test for 9th graders, which enjoys good
problem scatterness across the whole 9th grader curriculum and decent
difficulty level distribution.

    

### [[2112.11207] How are cities pledging net zero? A computational approach to analyzing subnational climate strategies](http://arxiv.org/abs/2112.11207)


  Cities have become primary actors on climate change and are increasingly
setting goals aimed at net-zero emissions. The rapid proliferation of
subnational governments "racing to zero" emissions and articulating their own
climate mitigation plans warrants closer examination to understand how these
actors intend to meet these goals. The scattered, incomplete and heterogeneous
nature of city climate policy documents, however, has made their systemic
analysis challenging. We analyze 318 climate action documents from cities that
have pledged net-zero targets or joined a transnational climate initiative with
this goal using machine learning-based natural language processing (NLP)
techniques. We use these approaches to accomplish two primary goals: 1)
determine text patterns that predict "ambitious" net-zero targets, where we
define an ambitious target as one that encompasses a subnational government's
economy-wide emissions; and 2) perform a sectoral analysis to identify patterns
and trade-offs in climate action themes (i.e., land-use, industry, buildings,
etc.). We find that cities that have defined ambitious climate actions tend to
emphasize quantitative metrics and specific high-emitting sectors in their
plans, supported by mentions of governance and citizen participation. Cities
predominantly emphasize energy-related actions in their plans, particularly in
the buildings, transport and heating sectors, but often at the expense of other
sectors, including land-use and climate impacts. The method presented in this
paper provides a replicable, scalable approach to analyzing climate action
plans and a first step towards facilitating cross-city learning.

    

### [[2112.11209] Interpretable Knowledge Tracing: Simple and Efficient Student Modeling with Causal Relations](http://arxiv.org/abs/2112.11209)


  Intelligent Tutoring Systems have become critically important in future
learning environments. Knowledge Tracing (KT) is a crucial part of that system.
It is about inferring the skill mastery of students and predicting their
performance to adjust the curriculum accordingly. Deep Learning-based KT models
have shown significant predictive performance compared with traditional models.
However, it is difficult to extract psychologically meaningful explanations
from the tens of thousands of parameters in neural networks, that would relate
to cognitive theory. There are several ways to achieve high accuracy in student
performance prediction but diagnostic and prognostic reasoning is more critical
in learning sciences. Since KT problem has few observable features (problem ID
and student's correctness at each practice), we extract meaningful latent
features from students' response data by using machine learning and data mining
techniques. In this work, we present Interpretable Knowledge Tracing (IKT), a
simple model that relies on three meaningful latent features: individual skill
mastery, ability profile (learning transfer across skills), and problem
difficulty. IKT's prediction of future student performance is made using a
Tree-Augmented Naive Bayes Classifier (TAN), therefore its predictions are
easier to explain than deep learning-based student models. IKT also shows
better student performance prediction than deep learning-based student models
without requiring a huge amount of parameters. We conduct ablation studies on
each feature to examine their contribution to student performance prediction.
Thus, IKT has great potential for providing adaptive and personalized
instructions with causal reasoning in real-world educational systems.

    

### [[2112.11210] Discrete fully probabilistic design: a tool to design control policies from examples](http://arxiv.org/abs/2112.11210)


  We present a discretized design that expounds an algorithm recently
introduced in Gagliardi and Russo (2021) to synthesize control policies from
examples for constrained, possibly stochastic and nonlinear, systems. The
constraints do not need to be fulfilled in the possibly noisy example data,
which in turn might be collected from a system that is different from the one
under control. For this discretized design, we discuss a number of properties
and give a design pipeline. The design, which we term as discrete fully
probabilistic design, is benchmarked numerically on an example that involves
controlling an inverted pendulum with actuation constraints starting from data
collected from a physically different pendulum that does not satisfy the
system-specific actuation constraints.

    

### [[2112.11212] Predicting Defects in Laser Powder Bed Fusion using in-situ Thermal Imaging Data and Machine Learning](http://arxiv.org/abs/2112.11212)


  Variation in the local thermal history during the laser powder bed fusion
(LPBF) process in additive manufacturing (AM) can cause microporosity defects.
in-situ sensing has been proposed to monitor the AM process to minimize
defects, but the success requires establishing a quantitative relationship
between the sensing data and the porosity, which is especially challenging for
a large number of variables and computationally costly. In this work, we
develop machine learning (ML) models that can use in-situ thermographic data to
predict the microporosity of LPBF stainless steel materials. This work
considers two identified key features from the thermal histories: the time
above the apparent melting threshold (/tau) and the maximum radiance (T_{max}).
These features are computed, stored for each voxel in the built material, are
used as inputs. The binary state of each voxel, either defective or normal, is
the output. Different ML models are trained and tested for the binary
classification task. In addition to using the thermal features of each voxel to
predict its own state, the thermal features of neighboring voxels are also
included as inputs. This is shown to improve the prediction accuracy, which is
consistent with thermal transport physics around each voxel contributing to its
final state. Among the models trained, the F1 scores on test sets reach above
0.96 for random forests. Feature importance analysis based on the ML models
shows that T_{max}is more important to the voxel state than /tau. The analysis
also finds that the thermal history of the voxels above the present voxel is
more influential than those beneath it.

    

### [[2112.11216] Value Activation for Bias Alleviation: Generalized-activated Deep Double Deterministic Policy Gradients](http://arxiv.org/abs/2112.11216)


  It is vital to accurately estimate the value function in Deep Reinforcement
Learning (DRL) such that the agent could execute proper actions instead of
suboptimal ones. However, existing actor-critic methods suffer more or less
from underestimation bias or overestimation bias, which negatively affect their
performance. In this paper, we reveal a simple but effective principle: proper
value correction benefits bias alleviation, where we propose the
generalized-activated weighting operator that uses any non-decreasing function,
namely activation function, as weights for better value estimation.
Particularly, we integrate the generalized-activated weighting operator into
value estimation and introduce a novel algorithm, Generalized-activated Deep
Double Deterministic Policy Gradients (GD3). We theoretically show that GD3 is
capable of alleviating the potential estimation bias. We interestingly find
that simple activation functions lead to satisfying performance with no
additional tricks, and could contribute to faster convergence. Experimental
results on numerous challenging continuous control tasks show that GD3 with
task-specific activation outperforms the common baseline methods. We also
uncover a fact that fine-tuning the polynomial activation function achieves
superior results on most of the tasks.

    

### [[2112.11217] Model-Based Safe Reinforcement Learning with Time-Varying State and Control Constraints: An Application to Intelligent Vehicles](http://arxiv.org/abs/2112.11217)


  Recently, barrier function-based safe reinforcement learning (RL) with the
actor-critic structure for continuous control tasks has received increasing
attention. It is still challenging to learn a near-optimal control policy with
safety and convergence guarantees. Also, few works have addressed the safe RL
algorithm design under time-varying safety constraints. This paper proposes a
model-based safe RL algorithm for optimal control of nonlinear systems with
time-varying state and control constraints. In the proposed approach, we
construct a novel barrier-based control policy structure that can guarantee
control safety. A multi-step policy evaluation mechanism is proposed to predict
the policy's safety risk under time-varying safety constraints and guide the
policy to update safely. Theoretical results on stability and robustness are
proven. Also, the convergence of the actor-critic learning algorithm is
analyzed. The performance of the proposed algorithm outperforms several
state-of-the-art RL algorithms in the simulated Safety Gym environment.
Furthermore, the approach is applied to the integrated path following and
collision avoidance problem for two real-world intelligent vehicles. A
differential-drive vehicle and an Ackermann-drive one are used to verify the
offline deployment performance and the online learning performance,
respectively. Our approach shows an impressive sim-to-real transfer capability
and a satisfactory online control performance in the experiment.

    

### [[2112.11218] Multiple Time Series Fusion Based on LSTM An Application to CAP A Phase Classification Using EEG](http://arxiv.org/abs/2112.11218)


  Biomedical decision making involves multiple signal processing, either from
different sensors or from different channels. In both cases, information fusion
plays a significant role. A deep learning based electroencephalogram channels'
feature level fusion is carried out in this work for the electroencephalogram
cyclic alternating pattern A phase classification. Channel selection, fusion,
and classification procedures were optimized by two optimization algorithms,
namely, Genetic Algorithm and Particle Swarm Optimization. The developed
methodologies were evaluated by fusing the information from multiple
electroencephalogram channels for patients with nocturnal frontal lobe epilepsy
and patients without any neurological disorder, which was significantly more
challenging when compared to other state of the art works. Results showed that
both optimization algorithms selected a comparable structure with similar
feature level fusion, consisting of three electroencephalogram channels, which
is in line with the CAP protocol to ensure multiple channels' arousals for CAP
detection. Moreover, the two optimized models reached an area under the
receiver operating characteristic curve of 0.82, with average accuracy ranging
from 77% to 79%, a result which is in the upper range of the specialist
agreement. The proposed approach is still in the upper range of the best state
of the art works despite a difficult dataset, and has the advantage of
providing a fully automatic analysis without requiring any manual procedure.
Ultimately, the models revealed to be noise resistant and resilient to multiple
channel loss.

    

### [[2112.11222] Jamming Pattern Recognition over Multi-Channel Networks: A Deep Learning Approach](http://arxiv.org/abs/2112.11222)


  With the advent of intelligent jammers, jamming attacks have become a more
severe threat to the performance of wireless systems. An intelligent jammer is
able to change its policy to minimize the probability of being traced by
legitimate nodes. Thus, an anti-jamming mechanism capable of constantly
adjusting to the jamming policy is required to combat such a jammer.
Remarkably, existing anti-jamming methods are not applicable here because they
mainly focus on mitigating jamming attacks with an invariant jamming policy,
and they rarely consider an intelligent jammer as an adversary. Therefore, in
this paper, to employ a jamming type recognition technique working alongside an
anti-jamming technique is proposed. The proposed recognition method employs a
recurrent neural network that takes the jammer's occupied channels as inputs
and outputs the jammer type. Under this scheme, the real-time jammer policy is
first identified, and, then, the most appropriate countermeasure is chosen.
Consequently, any changes to the jammer policy can be instantly detected with
the proposed recognition technique allowing for a rapid switch to a new
anti-jamming method fitted to the new jamming policy. To evaluate the
performance of the proposed recognition method, the accuracy of the detection
is derived as a function of the jammer policy switching time. Simulation
results show the detection accuracy for all the considered users numbers is
greater than 70% when the jammer switches its policy every 5 time slots and the
accuracy raises to 90% when the jammer policy switching time is 45.

    

### [[2112.11225] RetroComposer: Discovering Novel Reactions by Composing Templates for Retrosynthesis Prediction](http://arxiv.org/abs/2112.11225)


  The main target of retrosynthesis is to recursively decompose desired
molecules into available building blocks. Existing template-based
retrosynthesis methods follow a template selection stereotype and suffer from
the limited training templates, which prevents them from discovering novel
reactions. To overcome the limitation, we propose an innovative retrosynthesis
prediction framework that can compose novel templates beyond training
templates. So far as we know, this is the first method that can find novel
templates for retrosynthesis prediction. Besides, we propose an effective
reactant candidates scoring model that can capture atom-level transformation
information, and it helps our method outperform existing methods by a large
margin. Experimental results show that our method can produce novel templates
for 328 test reactions in the USPTO-50K dataset, including 21 test reactions
that are not covered by the training templates.

    

### [[2112.11226] Energy-bounded Learning for Robust Models of Code](http://arxiv.org/abs/2112.11226)


  In programming, learning code representations has a variety of applications,
including code classification, code search, comment generation, bug prediction,
and so on. Various representations of code in terms of tokens, syntax trees,
dependency graphs, code navigation paths, or a combination of their variants
have been proposed, however, existing vanilla learning techniques have a major
limitation in robustness, i.e., it is easy for the models to make incorrect
predictions when the inputs are altered in a subtle way. To enhance the
robustness, existing approaches focus on recognizing adversarial samples rather
than on the valid samples that fall outside a given distribution, which we
refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is
the novel problem investigated in this paper. To this end, we propose to first
augment the in=distribution datasets with out-of-distribution samples such
that, when trained together, they will enhance the model's robustness. We
propose the use of an energy-bounded learning objective function to assign a
higher score to in-distribution samples and a lower score to
out-of-distribution samples in order to incorporate such out-of-distribution
samples into the training process of source code models. In terms of OOD
detection and adversarial samples detection, our evaluation results demonstrate
a greater robustness for existing source code models to become more accurate at
recognizing OOD data while being more resistant to adversarial attacks at the
same time. Furthermore, the proposed energy-bounded score outperforms all
existing OOD detection scores by a large margin, including the softmax
confidence score, the Mahalanobis score, and ODIN.

    

### [[2112.11230] Interpretable Preference-based Reinforcement Learning with Tree-Structured Reward Functions](http://arxiv.org/abs/2112.11230)


  The potential of reinforcement learning (RL) to deliver aligned and
performant agents is partially bottlenecked by the reward engineering problem.
One alternative to heuristic trial-and-error is preference-based RL (PbRL),
where a reward function is inferred from sparse human feedback. However, prior
PbRL methods lack interpretability of the learned reward structure, which
hampers the ability to assess robustness and alignment. We propose an online,
active preference learning algorithm that constructs reward functions with the
intrinsically interpretable, compositional structure of a tree. Using both
synthetic and human-provided feedback, we demonstrate sample-efficient learning
of tree-structured reward functions in several environments, then harness the
enhanced interpretability to explore and debug for alignment.

    

### [[2112.11231] Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time](http://arxiv.org/abs/2112.11231)


  The event-driven and sparse nature of communication between spiking neurons
in the brain holds great promise for flexible and energy-efficient AI. Recent
advances in learning algorithms have demonstrated that recurrent networks of
spiking neurons can be effectively trained to achieve competitive performance
compared to standard recurrent neural networks. Still, as these learning
algorithms use error-backpropagation through time (BPTT), they suffer from high
memory requirements, are slow to train, and are incompatible with online
learning. This limits the application of these learning algorithms to
relatively small networks and to limited temporal sequence lengths. Online
approximations to BPTT with lower computational and memory complexity have been
proposed (e-prop, OSTL), but in practice also suffer from memory limitations
and, as approximations, do not outperform standard BPTT training. Here, we show
how a recently developed alternative to BPTT, Forward Propagation Through Time
(FPTT) can be applied in spiking neural networks. Different from BPTT, FPTT
attempts to minimize an ongoing dynamically regularized risk on the loss. As a
result, FPTT can be computed in an online fashion and has fixed complexity with
respect to the sequence length. When combined with a novel dynamic spiking
neuron model, the Liquid-Time-Constant neuron, we show that SNNs trained with
FPTT outperform online BPTT approximations, and approach or exceed offline BPTT
accuracy on temporal classification tasks. This approach thus makes it feasible
to train SNNs in a memory-friendly online fashion on long sequences and scale
up SNNs to novel and complex neural architectures.

    

### [[2112.11239] Preserving gauge invariance in neural networks](http://arxiv.org/abs/2112.11239)


  In these proceedings we present lattice gauge equivariant convolutional
neural networks (L-CNNs) which are able to process data from lattice gauge
theory simulations while exactly preserving gauge symmetry. We review aspects
of the architecture and show how L-CNNs can represent a large class of gauge
invariant and equivariant functions on the lattice. We compare the performance
of L-CNNs and non-equivariant networks using a non-linear regression problem
and demonstrate how gauge invariance is broken for non-equivariant models.

    

### [[2112.11242] Unsupervised deep learning techniques for powdery mildew recognition based on multispectral imaging](http://arxiv.org/abs/2112.11242)


  Objectives. Sustainable management of plant diseases is an open challenge
which has relevant economic and environmental impact. Optimal strategies rely
on human expertise for field scouting under favourable conditions to assess the
current presence and extent of disease symptoms. This labor-intensive task is
complicated by the large field area to be scouted, combined with the
millimeter-scale size of the early symptoms to be detected. In view of this,
image-based detection of early disease symptoms is an attractive approach to
automate this process, enabling a potential high throughput monitoring at
sustainable costs.
Methods. Deep learning has been successfully applied in various domains to
obtain an automatic selection of the relevant image features by learning
filters via a training procedure. Deep learning has recently entered also the
domain of plant disease detection: following this idea, in this work we present
a deep learning approach to automatically recognize powdery mildew on cucumber
leaves. We focus on unsupervised deep learning techniques applied to
multispectral imaging data and we propose the use of autoencoder architectures
to investigate two strategies for disease detection: i) clusterization of
features in a compressed space; ii) anomaly detection.
Results. The two proposed approaches have been assessed by quantitative
indices. The clusterization approach is not fully capable by itself to provide
accurate predictions but it does cater relevant information. Anomaly detection
has instead a significant potential of resolution which could be further
exploited as a prior for supervised architectures with a very limited number of
labeled samples.

    

### [[2112.11279] A Pilot Study on Detecting Unfairness in Human Decisions With Machine Learning Algorithmic Bias Detection](http://arxiv.org/abs/2112.11279)


  Fairness in decision-making has been a long-standing issue in our society.
Despite the increasing number of research activities on unfairness mitigation
in machine learning models, there is little research focusing on mitigating
unfairness in human decisions. Fairness in human decisions is as important as,
if not more important than, fairness in machine learning models since there are
processes where humans make the final decisions and machine learning models can
inherit bias from the human decisions they were trained on. As a result, this
work aims to detect unfairness in human decisions, the very first step of
solving the unfair human decision problem.
This paper proposes to utilize the existing machine learning fairness
detection mechanisms to detect unfairness in human decisions. The rationale
behind this is, while it is difficult to directly test whether a human makes
unfair decisions, with current research on machine learning fairness, it is now
easy to test, on a large scale at a low cost, whether a machine learning model
is unfair. By synthesizing unfair labels on four general machine learning
fairness datasets and one image processing dataset, this paper shows that the
proposed approach is able to detect (1) whether or not unfair labels exist in
the training data and (2) the degree and direction of the unfairness. We
believe that this work demonstrates the potential of utilizing machine learning
fairness to detect human decision fairness. Following this work, research can
be conducted on (1) preventing future unfair decisions, (2) fixing prior unfair
decisions, and (3) training a fairer machine learning model.

    

### [[2112.11282] VW-SDK: Efficient Convolutional Weight Mapping Using Variable Windows for Processing-In-Memory Architectures](http://arxiv.org/abs/2112.11282)


  With their high energy efficiency, processing-in-memory (PIM) arrays are
increasingly used for convolutional neural network (CNN) inference. In
PIM-based CNN inference, the computational latency and energy are dependent on
how the CNN weights are mapped to the PIM array. A recent study proposed
shifted and duplicated kernel (SDK) mapping that reuses the input feature maps
with a unit of a parallel window, which is convolved with duplicated kernels to
obtain multiple output elements in parallel. However, the existing SDK-based
mapping algorithm does not always result in the minimum computing cycles
because it only maps a square-shaped parallel window with the entire channels.
In this paper, we introduce a novel mapping algorithm called variable-window
SDK (VW-SDK), which adaptively determines the shape of the parallel window that
leads to the minimum computing cycles for a given convolutional layer and PIM
array. By allowing rectangular-shaped windows with partial channels, VW-SDK
utilizes the PIM array more efficiently, thereby further reduces the number of
computing cycles. The simulation with a 512x512 PIM array and Resnet-18 shows
that VW-SDK improves the inference speed by 1.69x compared to the existing
SDK-based algorithm.

    

### [[2112.11294] Extending CLIP for Category-to-image Retrieval in E-commerce](http://arxiv.org/abs/2112.11294)


  E-commerce provides rich multimodal data that is barely leveraged in
practice. One aspect of this data is a category tree that is being used in
search and recommendation. However, in practice, during a user's session there
is often a mismatch between a textual and a visual representation of a given
category. Motivated by the problem, we introduce the task of category-to-image
retrieval in e-commerce and propose a model for the task, CLIP-ITA. The model
leverages information from multiple modalities (textual, visual, and attribute
modality) to create product representations. We explore how adding information
from multiple modalities (textual, visual, and attribute modality) impacts the
model's performance. In particular, we observe that CLIP-ITA significantly
outperforms a comparable model that leverages only the visual modality and a
comparable model that leverages the visual and attribute modality.

    

### [[2112.11312] Implicit Neural Video Compression](http://arxiv.org/abs/2112.11312)


  We propose a method to compress full-resolution video sequences with implicit
neural representations. Each frame is represented as a neural network that maps
coordinate positions to pixel values. We use a separate implicit network to
modulate the coordinate inputs, which enables efficient motion compensation
between frames. Together with a small residual network, this allows us to
efficiently compress P-frames relative to the previous frame. We further lower
the bitrate by storing the network weights with learned integer quantization.
Our method, which we call implicit pixel flow (IPF), offers several
simplifications over established neural video codecs: it does not require the
receiver to have access to a pretrained neural network, does not use expensive
interpolation-based warping operations, and does not require a separate
training dataset. We demonstrate the feasibility of neural implicit compression
on image and video data.

    

### [[2112.11313] On the Adversarial Robustness of Causal Algorithmic Recourse](http://arxiv.org/abs/2112.11313)


  Algorithmic recourse seeks to provide actionable recommendations for
individuals to overcome unfavorable outcomes made by automated decision-making
systems. Recourse recommendations should ideally be robust to reasonably small
uncertainty in the features of the individual seeking recourse. In this work,
we formulate the adversarially robust recourse problem and show that recourse
methods offering minimally costly recourse fail to be robust. We then present
methods for generating adversarially robust recourse in the linear and in the
differentiable case. To ensure that recourse is robust, individuals are asked
to make more effort than they would have otherwise had to. In order to shift
part of the burden of robustness from the decision-subject to the
decision-maker, we propose a model regularizer that encourages the additional
cost of seeking robust recourse to be low. We show that classifiers trained
with our proposed model regularizer, which penalizes relying on unactionable
features for prediction, offer potentially less effortful recourse.

    

### [[2112.11317] Deep Learning Based Cloud Cover Parameterization for ICON](http://arxiv.org/abs/2112.11317)


  A promising approach to improve cloud parameterizations within climate models
and thus climate projections is to use deep learning in combination with
training data from storm-resolving model (SRM) simulations. The Icosahedral
Non-Hydrostatic (ICON) modeling framework permits simulations ranging from
numerical weather prediction to climate projections, making it an ideal target
to develop neural network (NN) based parameterizations for sub-grid scale
processes. Within the ICON framework, we train NN based cloud cover
parameterizations with coarse-grained data based on realistic regional and
global ICON SRM simulations. We set up three different types of NNs that differ
in the degree of vertical locality they assume for diagnosing cloud cover from
coarse-grained atmospheric state variables. The NNs accurately estimate
sub-grid scale cloud cover from coarse-grained data that has similar
geographical characteristics as their training data. Additionally, globally
trained NNs can reproduce sub-grid scale cloud cover of the regional SRM
simulation. Using the game-theory based interpretability library SHapley
Additive exPlanations, we identify an overemphasis on specific humidity and
cloud ice as the reason why our column-based NN cannot perfectly generalize
from the global to the regional coarse-grained SRM data. The interpretability
tool also helps visualize similarities and differences in feature importance
between regionally and globally trained column-based NNs, and reveals a local
relationship between their cloud cover predictions and the thermodynamic
environment. Our results show the potential of deep learning to derive accurate
yet interpretable cloud cover parameterizations from global SRMs, and suggest
that neighborhood-based models may be a good compromise between accuracy and
generalizability.

    

### [[2112.11323] Physics-informed neural network method for modelling beam-wall interactions](http://arxiv.org/abs/2112.11323)


  A mesh-free approach for modelling beam-wall interactions in particle
accelerators is proposed. The key idea of our method is to use a deep neural
network as a surrogate for the solution to a set of partial differential
equations involving the particle beam, and the surface impedance concept. The
proposed approach is applied to the coupling impedance of an accelerator vacuum
chamber with thin conductive coating, and also verified in comparison with the
existing analytical formula.

    

### [[2112.11330] PrimSeq: a deep learning-based pipeline to quantitate rehabilitation training](http://arxiv.org/abs/2112.11330)


  Stroke rehabilitation seeks to increase neuroplasticity through the repeated
practice of functional motions, but may have minimal impact on recovery because
of insufficient repetitions. The optimal training content and quantity are
currently unknown because no practical tools exist to measure them. Here, we
present PrimSeq, a pipeline to classify and count functional motions trained in
stroke rehabilitation. Our approach integrates wearable sensors to capture
upper-body motion, a deep learning model to predict motion sequences, and an
algorithm to tally motions. The trained model accurately decomposes
rehabilitation activities into component functional motions, outperforming
competitive machine learning methods. PrimSeq furthermore quantifies these
motions at a fraction of the time and labor costs of human experts. We
demonstrate the capabilities of PrimSeq in previously unseen stroke patients
with a range of upper extremity motor impairment. We expect that these advances
will support the rigorous measurement required for quantitative dosing trials
in stroke rehabilitation.

    

### [[2112.11335] Deep Learning Based 3D Point Cloud Regression for Estimating Forest Biomass](http://arxiv.org/abs/2112.11335)


  Knowledge of forest biomass stocks and their development is important for
implementing effective climate change mitigation measures. It is needed for
studying the processes driving af-, re-, and deforestation and is a
prerequisite for carbon-accounting. Remote sensing using airborne LiDAR can be
used to measure vegetation biomass at large scale. We present deep learning
systems for predicting wood volume, above-ground biomass (AGB), and
subsequently carbon directly from 3D LiDAR point cloud data. We devise
different neural network architectures for point cloud regression and evaluate
them on remote sensing data of areas for which AGB estimates have been obtained
from field measurements in a national forest inventory. Our adaptation of
Minkowski convolutional neural networks for regression gave the best results.
The deep neural networks produced significantly more accurate wood volume, AGB,
and carbon estimates compared to state-of-the-art approaches operating on basic
statistics of the point clouds, and we expect this finding to have a strong
impact on LiDAR-based analyses of terrestrial ecosystem dynamics.

    

### [[2112.11360] Neural network guided adjoint computations in dual weighted residual error estimation](http://arxiv.org/abs/2112.11360)


  Deep learning has shown successful application in visual recognition and
certain artificial intelligence tasks. Deep learning is also considered as a
powerful tool with high flexibility to approximate functions. In the present
work, functions with desired properties are devised to approximate the
solutions of PDEs. Our approach is based on a posteriori error estimation in
which the adjoint problem is solved for the error localization to formulate an
error estimator within the framework of neural network. An efficient and easy
to implement algorithm is developed to obtain a posteriori error estimate for
multiple goal functionals by employing the dual-weighted residual approach,
which is followed by the computation of both primal and adjoint solutions using
the neural network. The present study shows that such a data-driven model based
learning has superior approximation of quantities of interest even with
relatively less training data. The novel algorithmic developments are
substantiated with numerical test examples. The advantages of using deep neural
network over the shallow neural network are demonstrated and the convergence
enhancing techniques are also presented

    

### [[2112.11367] Deep Learning and Earth Observation to Support the Sustainable Development Goals](http://arxiv.org/abs/2112.11367)


  The synergistic combination of deep learning models and Earth observation
promises significant advances to support the sustainable development goals
(SDGs). New developments and a plethora of applications are already changing
the way humanity will face the living planet challenges. This paper reviews
current deep learning approaches for Earth observation data, along with their
application towards monitoring and achieving the SDGs most impacted by the
rapid development of deep learning in Earth observation. We systematically
review case studies to 1) achieve zero hunger, 2) sustainable cities, 3)
deliver tenure security, 4) mitigate and adapt to climate change, and 5)
preserve biodiversity. Important societal, economic and environmental
implications are concerned. Exciting times ahead are coming where algorithms
and Earth data can help in our endeavor to address the climate crisis and
support more sustainable development.

    

### [[2112.11384] Sports Video: Fine-Grained Action Detection and Classification of Table Tennis Strokes from Videos for MediaEval 2021](http://arxiv.org/abs/2112.11384)


  Sports video analysis is a prevalent research topic due to the variety of
application areas, ranging from multimedia intelligent devices with
user-tailored digests up to analysis of athletes' performance. The Sports Video
task is part of the MediaEval 2021 benchmark. This task tackles fine-grained
action detection and classification from videos. The focus is on recordings of
table tennis games. Running since 2019, the task has offered a classification
challenge from untrimmed video recorded in natural conditions with known
temporal boundaries for each stroke. This year, the dataset is extended and
offers, in addition, a detection challenge from untrimmed videos without
annotations. This work aims at creating tools for sports coaches and players in
order to analyze sports performance. Movement analysis and player profiling may
be built upon such technology to enrich the training experience of athletes and
improve their performance.

    

### [[2112.11389] Supervised Graph Contrastive Pretraining for Text Classification](http://arxiv.org/abs/2112.11389)


  Contrastive pretraining techniques for text classification has been largely
studied in an unsupervised setting. However, oftentimes labeled data from
related tasks which share label semantics with current task is available. We
hypothesize that using this labeled data effectively can lead to better
generalization on current task. In this paper, we propose a novel way to
effectively utilize labeled data from related tasks with a graph based
supervised contrastive learning approach. We formulate a token-graph by
extrapolating the supervised information from examples to tokens. Our
formulation results in an embedding space where tokens with high/low
probability of belonging to same class are near/further-away from one another.
We also develop detailed theoretical insights which serve as a motivation for
our method. In our experiments with $13$ datasets, we show our method
outperforms pretraining schemes by $2.5\%$ and also example-level contrastive
learning based formulation by $1.8\%$ on average. In addition, we show
cross-domain effectiveness of our method in a zero-shot setting by $3.91\%$ on
average. Lastly, we also demonstrate our method can be used as a noisy teacher
in a knowledge distillation setting to significantly improve performance of
transformer based models in low labeled data regime by $4.57\%$ on average.

    

### [[2112.11391] Voice Quality and Pitch Features in Transformer-Based Speech Recognition](http://arxiv.org/abs/2112.11391)


  Jitter and shimmer measurements have shown to be carriers of voice quality
and prosodic information which enhance the performance of tasks like speaker
recognition, diarization or automatic speech recognition (ASR). However, such
features have been seldom used in the context of neural-based ASR, where
spectral features often prevail. In this work, we study the effects of
incorporating voice quality and pitch features altogether and separately to a
Transformer-based ASR model, with the intuition that the attention mechanisms
might exploit latent prosodic traits. For doing so, we propose separated
convolutional front-ends for prosodic and spectral features, showing that this
architectural choice yields better results than simple concatenation of such
pitch and voice quality features to mel-spectrogram filterbanks. Furthermore,
we find mean Word Error Rate relative reductions of up to 5.6% with the
LibriSpeech benchmark. Such findings motivate further research on the
application of prosody knowledge for increasing the robustness of
Transformer-based ASR.

    

### [[2112.11397] NN2Poly: A polynomial representation for deep feed-forward artificial neural networks](http://arxiv.org/abs/2112.11397)


  Interpretability of neural networks and their underlying theoretical
behaviour remain being an open field of study, even after the great success of
their practical applications, particularly with the emergence of deep learning.
In this work, NN2Poly is proposed: a theoretical approach that allows to obtain
polynomials that provide an alternative representation of an already trained
deep neural network. This extends the previous idea proposed in
arXiv:2102.03865, which was limited to single hidden layer neural networks, to
work with arbitrarily deep feed-forward neural networks in both regression and
classification tasks. The objective of this paper is achieved by using a Taylor
expansion on the activation function, at each layer, and then using several
combinatorial properties that allow to identify the coefficients of the desired
polynomials. The main computational limitations when implementing this
theoretical method are discussed and it is presented an example of the
constraints on the neural network weights that are necessary for NN2Poly to
work. Finally, some simulations are presented were it is concluded that using
NN2Poly it is possible to obtain a representation for the given neural network
with low error between the obtained predictions.

    

### [[2112.11407] Toward Explainable AI for Regression Models](http://arxiv.org/abs/2112.11407)


  In addition to the impressive predictive power of machine learning (ML)
models, more recently, explanation methods have emerged that enable an
interpretation of complex non-linear learning models such as deep neural
networks. Gaining a better understanding is especially important e.g. for
safety-critical ML applications or medical diagnostics etc. While such
Explainable AI (XAI) techniques have reached significant popularity for
classifiers, so far little attention has been devoted to XAI for regression
models (XAIR). In this review, we clarify the fundamental conceptual
differences of XAI for regression and classification tasks, establish novel
theoretical insights and analysis for XAIR, provide demonstrations of XAIR on
genuine practical regression problems, and finally discuss the challenges
remaining for the field.

    

### [[2112.11413] Offloading Algorithms for Maximizing Inference Accuracy on Edge Device Under a Time Constraint](http://arxiv.org/abs/2112.11413)


  With the emergence of edge computing, the problem of offloading jobs between
an Edge Device (ED) and an Edge Server (ES) received significant attention in
the past. Motivated by the fact that an increasing number of applications are
using Machine Learning (ML) inference, we study the problem of offloading
inference jobs by considering the following novel aspects: 1) in contrast to a
typical computational job, the processing time of an inference job depends on
the size of the ML model, and 2) recently proposed Deep Neural Networks (DNNs)
for resource-constrained devices provide the choice of scaling the model size.
We formulate an assignment problem with the aim of maximizing the total
inference accuracy of n data samples available at the ED, subject to a time
constraint T on the makespan. We propose an approximation algorithm AMR2, and
prove that it results in a makespan at most 2T, and achieves a total accuracy
that is lower by a small constant from optimal total accuracy. As proof of
concept, we implemented AMR2 on a Raspberry Pi, equipped with MobileNet, and is
connected to a server equipped with ResNet, and studied the total accuracy and
makespan performance of AMR2 for image classification application.

    

### [[2112.11429] Machine Learning Emulation of Urban Land Surface Processes](http://arxiv.org/abs/2112.11429)


  Can we improve the modeling of urban land surface processes with machine
learning (ML)? A prior comparison of urban land surface models (ULSMs) found
that no single model is 'best' at predicting all common surface fluxes. Here,
we develop an urban neural network (UNN) trained on the mean predicted fluxes
from 22 ULSMs at one site. The UNN emulates the mean output of ULSMs
accurately. When compared to a reference ULSM (Town Energy Balance; TEB), the
UNN has greater accuracy relative to flux observations, less computational
cost, and requires fewer input parameters. When coupled to the Weather Research
Forecasting (WRF) model using TensorFlow bindings, WRF-UNN is stable and more
accurate than the reference WRF-TEB. Although the application is currently
constrained by the training data (1 site), we show a novel approach to improve
the modeling of surface fluxes by combining the strengths of several ULSMs into
one using ML.

    

### [[2112.11439] Automated Drug-Related Information Extraction from French Clinical Documents: ReLyfe Approach](http://arxiv.org/abs/2112.11439)


  Structuring medical data in France remains a challenge mainly because of the
lack of medical data due to privacy concerns and the lack of methods and
approaches on processing the French language. One of these challenges is
structuring drug-related information in French clinical documents. To our
knowledge, over the last decade, there are less than five relevant papers that
study French prescriptions. This paper proposes a new approach for extracting
drug-related information from French clinical scanned documents while
preserving patients' privacy. In addition, we deployed our method in a health
data management platform where it is used to structure drug medical data and
help patients organize their drug schedules. It can be implemented on any web
or mobile platform. This work closes the gap between theoretical and practical
work by creating an application adapted to real production problems. It is a
combination of a rule-based phase and a Deep Learning approach. Finally,
numerical results show the outperformance and relevance of the proposed
methodology.

    

### [[2112.11442] Deliberation of Streaming RNN-Transducer by Non-autoregressive Decoding](http://arxiv.org/abs/2112.11442)


  We propose to deliberate the hypothesis alignment of a streaming RNN-T model
with the previously proposed Align-Refine non-autoregressive decoding method
and its improved versions. The method performs a few refinement steps, where
each step shares a transformer decoder that attends to both text features
(extracted from alignments) and audio features, and outputs complete updated
alignments. The transformer decoder is trained with the CTC loss which
facilitates parallel greedy decoding, and performs full-context attention to
capture label dependencies. We improve Align-Refine by introducing cascaded
encoder that captures more audio context before refinement, and alignment
augmentation which enforces learning label dependency. We show that,
conditioned on hypothesis alignments of a streaming RNN-T model, our method
obtains significantly more accurate recognition results than the first-pass
RNN-T, with only small amount of model parameters.

    

### [[2112.11445] Controversy Detection: a Text and Graph Neural Network Based Approach](http://arxiv.org/abs/2112.11445)


  Controversial content refers to any content that attracts both positive and
negative feedback. Its automatic identification, especially on social media, is
a challenging task as it should be done on a large number of continuously
evolving posts, covering a large variety of topics. Most of the existing
approaches rely on the graph structure of a topic-discussion and/or the content
of messages. This paper proposes a controversy detection approach based on both
graph structure of a discussion and text features. Our proposed approach relies
on Graph Neural Network (gnn) to encode the graph representation (including its
texts) in an embedding vector before performing a graph classification task.
The latter will classify the post as controversial or not. Two controversy
detection strategies are proposed. The first one is based on a hierarchical
graph representation learning. Graph user nodes are embedded hierarchically and
iteratively to compute the whole graph embedding vector. The second one is
based on the attention mechanism, which allows each user node to give more or
less importance to its neighbors when computing node embeddings. We conduct
experiments to evaluate our approach using different real-world datasets.
Conducted experiments show the positive impact of combining textual features
and structural information in terms of performance.

    

### [[2112.11449] Doubly-Valid/Doubly-Sharp Sensitivity Analysis for Causal Inference with Unmeasured Confounding](http://arxiv.org/abs/2112.11449)


  We study the problem of constructing bounds on the average treatment effect
in the presence of unobserved confounding under the marginal sensitivity model
of Tan (2006). Combining an existing characterization involving adversarial
propensity scores with a new distributionally robust characterization of the
problem, we propose novel estimators of these bounds that we call
"doubly-valid/doubly-sharp" (DVDS) estimators. Double sharpness corresponds to
the fact that DVDS estimators consistently estimate the tightest possible
(i.e., sharp) bounds implied by the sensitivity model even when one of two
nuisance parameters is misspecified and achieve semiparametric efficiency when
all nuisance parameters are suitably consistent. Double validity is an entirely
new property for partial identification: DVDS estimators still provide valid,
though not sharp, bounds even when most nuisance parameters are misspecified.
In fact, even in cases when DVDS point estimates fail to be asymptotically
normal, standard Wald confidence intervals may remain valid. In the case of
binary outcomes, the DVDS estimators are particularly convenient and possesses
a closed-form expression in terms of the outcome regression and propensity
score. We demonstrate the DVDS estimators in a simulation study as well as a
case study of right heart catheterization.

    

### [[2112.11450] Max-Margin Contrastive Learning](http://arxiv.org/abs/2112.11450)


  Standard contrastive learning approaches usually require a large number of
negatives for effective unsupervised learning and often exhibit slow
convergence. We suspect this behavior is due to the suboptimal selection of
negatives used for offering contrast to the positives. We counter this
difficulty by taking inspiration from support vector machines (SVMs) to present
max-margin contrastive learning (MMCL). Our approach selects negatives as the
sparse support vectors obtained via a quadratic optimization problem, and
contrastiveness is enforced by maximizing the decision margin. As SVM
optimization can be computationally demanding, especially in an end-to-end
setting, we present simplifications that alleviate the computational burden. We
validate our approach on standard vision benchmark datasets, demonstrating
better performance in unsupervised representation learning over
state-of-the-art, while having better empirical convergence properties.

    

### [[1902.00947] Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions](http://arxiv.org/abs/1902.00947)


  We provide a novel computer-assisted technique for systematically analyzing
first-order methods for optimization. In contrast with previous works, the
approach is particularly suited for handling sublinear convergence rates and
stochastic oracles. The technique relies on semidefinite programming and
potential functions. It allows simultaneously obtaining worst-case guarantees
on the behavior of those algorithms, and assisting in choosing appropriate
parameters for tuning their worst-case performances. The technique also
benefits from comfortable tightness guarantees, meaning that unsatisfactory
results can be improved only by changing the setting. We use the approach for
analyzing deterministic and stochastic first-order methods under different
assumptions on the nature of the stochastic noise. Among others, we treat
unstructured noise with bounded variance, different noise models arising in
over-parametrized expectation minimization problems, and randomized
block-coordinate descent schemes.

    

### [[2003.05033] Generalized Energy Based Models](http://arxiv.org/abs/2003.05033)


  We introduce the Generalized Energy Based Model (GEBM) for generative
modelling. These models combine two trained components: a base distribution
(generally an implicit model), which can learn the support of data with low
intrinsic dimension in a high dimensional space; and an energy function, to
refine the probability mass on the learned support. Both the energy function
and base jointly constitute the final model, unlike GANs, which retain only the
base distribution (the "generator"). GEBMs are trained by alternating between
learning the energy and the base. We show that both training stages are
well-defined: the energy is learned by maximising a generalized likelihood, and
the resulting energy-based loss provides informative gradients for learning the
base. Samples from the posterior on the latent space of the trained model can
be obtained via MCMC, thus finding regions in this space that produce better
quality samples. Empirically, the GEBM samples on image-generation tasks are of
much better quality than those from the learned generator alone, indicating
that all else being equal, the GEBM will outperform a GAN of the same
complexity. When using normalizing flows as base measures, GEBMs succeed on
density modelling tasks, returning comparable performance to direct maximum
likelihood of the same networks.

    

### [[2003.05402] FuDGE: A Method to Estimate a Functional Differential Graph in a High-Dimensional Setting](http://arxiv.org/abs/2003.05402)


  We consider the problem of estimating the difference between two functional
undirected graphical models with shared structures. In many applications, data
are naturally regarded as a vector of random functions rather than a vector of
scalars. For example, electroencephalography (EEG) data are more appropriately
treated as functions of time. In such a problem, not only can the number of
functions measured per sample be large, but each function is itself an infinite
dimensional object, making estimation of model parameters challenging. This is
further complicated by the fact that the curves are usually only observed at
discrete time points. We first define a functional differential graph that
captures the differences between two functional graphical models and formally
characterize when the functional differential graph is well defined. We then
propose a method, FuDGE, that directly estimates the functional differential
graph without first estimating each individual graph. This is particularly
beneficial in settings where the individual graphs are dense, but the
differential graph is sparse. We show that FuDGE consistently estimates the
functional differential graph even in a high-dimensional setting for both fully
observed and discretely observed function paths. We illustrate the finite
sample properties of our method through simulation studies. We also propose a
competing method, the Joint Functional Graphical Lasso, which generalizes the
Joint Graphical Lasso to the functional setting. Finally, we apply our method
to EEG data to uncover differences in functional brain connectivity between a
group of individuals with alcohol use disorder and a control group.

    

### [[2010.04116] Interlocking Backpropagation: Improving depthwise model-parallelism](http://arxiv.org/abs/2010.04116)


  The number of parameters in state of the art neural networks has drastically
increased in recent years. This surge of interest in large scale neural
networks has motivated the development of new distributed training strategies
enabling such models. One such strategy is model-parallel distributed training.
Unfortunately, model-parallelism suffers from poor resource utilisation, which
leads to wasted resources. In this work, we improve upon recent developments in
an idealised model-parallel optimisation setting: local learning. Motivated by
poor resource utilisation, we introduce a class of intermediary strategies
between local and global learning referred to as interlocking backpropagation.
These strategies preserve many of the compute-efficiency advantages of local
optimisation, while recovering much of the task performance achieved by global
optimisation. We assess our strategies on both image classification ResNets and
Transformer language models, finding that our strategy consistently
out-performs local learning in terms of task performance, and out-performs
global learning in training efficiency.

    

### [[2010.12112] Investigating Membership Inference Attacks under Data Dependencies](http://arxiv.org/abs/2010.12112)


  Training machine learning models on privacy-sensitive data has become a
popular practice, driving innovation in ever-expanding fields. This has opened
the door to new attacks that can have serious privacy implications. One such
attack, the Membership Inference Attack (MIA), exposes whether or not a
particular data point was used to train a model. A growing body of literature
uses Differentially Private (DP) training algorithms as a defence against such
attacks. However, these works evaluate the defence under the restrictive
assumption that all members of the training set, as well as non-members, are
independent and identically distributed. This assumption does not hold for many
real-world use cases in the literature. Motivated by this, we evaluate
membership inference with statistical dependencies among samples and explain
why DP does not provide meaningful protection (the privacy parameter $\epsilon$
scales with the training set size $n$) in this more general case. We conduct a
series of empirical evaluations with off-the-shelf MIAs using training sets
built from real-world data showing different types of dependencies among
samples. Our results reveal that training set dependencies can severely
increase the performance of MIAs, and therefore assuming that data samples are
statistically independent can significantly underestimate the performance of
MIAs.

    

### [[2012.07257] A Visual Mining Approach to Improved Multiple-Instance Learning](http://arxiv.org/abs/2012.07257)


  Multiple-instance learning (MIL) is a paradigm of machine learning that aims
to classify a set (bag) of objects (instances), assigning labels only to the
bags. This problem is often addressed by selecting an instance to represent
each bag, transforming a MIL problem into standard supervised learning.
Visualization can be a useful tool to assess learning scenarios by
incorporating the users' knowledge into the classification process. Considering
that multiple-instance learning is a paradigm that cannot be handled by current
visualization techniques, we propose a multiscale tree-based visualization
called MILTree to support MIL problems. The first level of the tree represents
the bags, and the second level represents the instances belonging to each bag,
allowing users to understand the MIL datasets in an intuitive way. In addition,
we propose two new instance selection methods for MIL, which help users improve
the model even further. Our methods can handle both binary and multiclass
scenarios. In our experiments, SVM was used to build the classifiers. With
support of the MILTree layout, the initial classification model was updated by
changing the training set, which is composed of the prototype instances.
Experimental results validate the effectiveness of our approach, showing that
visual mining by MILTree can support exploring and improving models in MIL
scenarios and that our instance selection methods outperform the currently
available alternatives in most cases.

    

### [[2012.13329] Vector-output ReLU Neural Network Problems are Copositive Programs: Convex Analysis of Two Layer Networks and Polynomial-time Algorithms](http://arxiv.org/abs/2012.13329)


  We describe the convex semi-infinite dual of the two-layer vector-output ReLU
neural network training problem. This semi-infinite dual admits a finite
dimensional representation, but its support is over a convex set which is
difficult to characterize. In particular, we demonstrate that the non-convex
neural network training problem is equivalent to a finite-dimensional convex
copositive program. Our work is the first to identify this strong connection
between the global optima of neural networks and those of copositive programs.
We thus demonstrate how neural networks implicitly attempt to solve copositive
programs via semi-nonnegative matrix factorization, and draw key insights from
this formulation. We describe the first algorithms for provably finding the
global minimum of the vector output neural network training problem, which are
polynomial in the number of samples for a fixed data rank, yet exponential in
the dimension. However, in the case of convolutional architectures, the
computational complexity is exponential in only the filter size and polynomial
in all other parameters. We describe the circumstances in which we can find the
global optimum of this neural network training problem exactly with
soft-thresholded SVD, and provide a copositive relaxation which is guaranteed
to be exact for certain classes of problems, and which corresponds with the
solution of Stochastic Gradient Descent in practice.

    

### [[2101.07361] Through the Data Management Lens: Experimental Analysis and Evaluation of Fair Classification](http://arxiv.org/abs/2101.07361)


  Classification, a heavily-studied data-driven machine learning task, drives
an increasing number of prediction systems involving critical human decisions
such as loan approval and criminal risk assessment. However, classifiers often
demonstrate discriminatory behavior, especially when presented with biased
data. Consequently, fairness in classification has emerged as a high-priority
research area. Data management research is showing an increasing presence and
interest in topics related to data and algorithmic fairness, including the
topic of fair classification. The interdisciplinary efforts in fair
classification, with machine learning research having the largest presence,
have resulted in a large number of fairness notions and a wide range of
approaches that have not been systematically evaluated and compared. In this
paper, we contribute a broad analysis of 13 fair classification approaches and
additional variants, over their correctness, fairness, efficiency, scalability,
robustness to data errors, sensitivity to underlying ML model, data efficiency,
and stability using a variety of metrics and real-world datasets. Our analysis
highlights novel insights on the impact of different metrics and high-level
approach characteristics on different aspects of performance. We also discuss
general principles for choosing approaches suitable for different practical
settings, and identify areas where data-management-centric solutions are likely
to have the most impact.

    

### [[2101.09545] Acceleration Methods](http://arxiv.org/abs/2101.09545)


  This monograph covers some recent advances in a range of acceleration
techniques frequently used in convex optimization. We first use quadratic
optimization problems to introduce two key families of methods, namely momentum
and nested optimization schemes. They coincide in the quadratic case to form
the Chebyshev method. We discuss momentum methods in detail, starting with the
seminal work of Nesterov and structure convergence proofs using a few master
templates, such as that for optimized gradient methods, which provide the key
benefit of showing how momentum methods optimize convergence guarantees. We
further cover proximal acceleration, at the heart of the Catalyst and
Accelerated Hybrid Proximal Extragradient frameworks, using similar algorithmic
patterns. Common acceleration techniques rely directly on the knowledge of some
of the regularity parameters in the problem at hand. We conclude by discussing
restart schemes, a set of simple techniques for reaching nearly optimal
convergence rates while adapting to unobserved regularity parameters.

    

### [[2101.11410] Reproducing kernel Hilbert C*-module and kernel mean embeddings](http://arxiv.org/abs/2101.11410)


  Kernel methods have been among the most popular techniques in machine
learning, where learning tasks are solved using the property of reproducing
kernel Hilbert space (RKHS). In this paper, we propose a novel data analysis
framework with reproducing kernel Hilbert $C^*$-module (RKHM) and kernel mean
embedding (KME) in RKHM. Since RKHM contains richer information than RKHS or
vector-valued RKHS (vvRKHS), analysis with RKHM enables us to capture and
extract structural properties in such as functional data. We show a branch of
theories for RKHM to apply to data analysis, including the representer theorem,
and the injectivity and universality of the proposed KME. We also show RKHM
generalizes RKHS and vvRKHS. Then, we provide concrete procedures for employing
RKHM and the proposed KME to data analysis.

    

### [[2104.08166] Automatic Termination for Hyperparameter Optimization](http://arxiv.org/abs/2104.08166)


  Bayesian optimization (BO) is a widely popular approach for the
hyperparameter optimization (HPO) of machine learning algorithms. At its core,
BO iteratively evaluates promising configurations until a user-defined budget,
such as wall-clock time or number of iterations, is exhausted. While the final
performance after tuning heavily depends on the provided budget, it is hard to
pre-specify an optimal value in advance. In this work, we propose an effective
and intuitive termination criterion for BO that automatically stops the
procedure if it is sufficiently close to the global optima. Across an extensive
range of real-world HPO problems, we show that our termination criterion
achieves better test performance compared to existing baselines from the
literature, such as stopping when the probability of improvement drops below a
fixed threshold. We also provide evidence that these baselines are, compared to
our method, highly sensitive to the choices of their own hyperparameters.
Additionally, we find that overfitting might occur in the context of HPO, which
is arguably an overlooked problem in the literature, and show that our
termination criterion mitigates this phenomenon on both small and large
datasets.

    

### [[2105.08509] Conjunction Data Messages behave as a Poisson Process](http://arxiv.org/abs/2105.08509)


  Space debris is a major problem in space exploration. International bodies
continuously monitor a large database of orbiting objects and emit warnings in
the form of conjunction data messages. An important question for satellite
operators is to estimate when fresh information will arrive so that they can
react timely but sparingly with satellite maneuvers. We propose a statistical
learning model of the message arrival process, allowing us to answer two
important questions: (1) Will there be any new message in the next specified
time interval? (2) When exactly and with what uncertainty will the next message
arrive? The average prediction error for question (2) of our Bayesian Poisson
process model is smaller than the baseline in more than 4 hours in a test set
of 50k close encounter events.

    

### [[2107.00464] On the Convergence of Stochastic Extragradient for Bilinear Games with Restarted Iteration Averaging](http://arxiv.org/abs/2107.00464)


  We study the stochastic bilinear minimax optimization problem, presenting an
analysis of the same-sample Stochastic ExtraGradient (SEG) method with constant
step size, and presenting variations of the method that yield favorable
convergence. In sharp contrasts with the basic SEG method whose last iterate
only contracts to a fixed neighborhood of the Nash equilibrium, SEG augmented
with iteration averaging provably converges to the Nash equilibrium under the
same standard settings, and such a rate is further improved by incorporating a
scheduled restarting procedure. In the interpolation setting where noise
vanishes at the Nash equilibrium, we achieve an optimal convergence rate up to
tight constants. We present numerical experiments that validate our theoretical
findings and demonstrate the effectiveness of the SEG method when equipped with
iteration averaging and restarting.

    

### [[2109.06050] Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training](http://arxiv.org/abs/2109.06050)


  The goal of stance detection is to determine the viewpoint expressed in a
piece of text towards a target. These viewpoints or contexts are often
expressed in many different languages depending on the user and the platform,
which can be a local news outlet, a social media platform, a news forum, etc.
Most research in stance detection, however, has been limited to working with a
single language and on a few limited targets, with little work on cross-lingual
stance detection. Moreover, non-English sources of labelled data are often
scarce and present additional challenges. Recently, large multilingual language
models have substantially improved the performance on many non-English tasks,
especially such with limited numbers of examples. This highlights the
importance of model pre-training and its ability to learn from few examples. In
this paper, we present the most comprehensive study of cross-lingual stance
detection to date: we experiment with 15 diverse datasets in 12 languages from
6 language families, and with 6 low-resource evaluation settings each. For our
experiments, we build on pattern-exploiting training, proposing the addition of
a novel label encoder to simplify the verbalisation procedure. We further
propose sentiment-based generation of stance data for pre-training, which shows
sizeable improvement of more than 6% F1 absolute in low-shot settings compared
to several strong baselines.

    

### [[2112.10629] Turbo-Sim: a generalised generative model with a physical latent space](http://arxiv.org/abs/2112.10629)


  We present Turbo-Sim, a generalised autoencoder framework derived from
principles of information theory that can be used as a generative model. By
maximising the mutual information between the input and the output of both the
encoder and the decoder, we are able to rediscover the loss terms usually found
in adversarial autoencoders and generative adversarial networks, as well as
various more sophisticated related models. Our generalised framework makes
these models mathematically interpretable and allows for a diversity of new
ones by setting the weight of each loss term separately. The framework is also
independent of the intrinsic architecture of the encoder and the decoder thus
leaving a wide choice for the building blocks of the whole network. We apply
Turbo-Sim to a collider physics generation problem: the transformation of the
properties of several particles from a theory space, right after the collision,
to an observation space, right after the detection in an experiment.

    

### [[2112.09490] Visual Microfossil Identificationvia Deep Metric Learning](http://arxiv.org/abs/2112.09490)


  We apply deep metric learning for the first time to the prob-lem of
classifying planktic foraminifer shells on microscopic images. This species
recognition task is an important information source and scientific pillar for
reconstructing past climates. All foraminifer CNN recognition pipelines in the
literature produce black-box classifiers that lack visualisation options for
human experts and cannot be applied to open set problems. Here, we benchmark
metric learning against these pipelines, produce the first scientific
visualisation of the phenotypic planktic foraminifer morphology space, and
demonstrate that metric learning can be used to cluster species unseen during
training. We show that metric learning out-performs all published CNN-based
state-of-the-art benchmarks in this domain. We evaluate our approach on the
34,640 expert-annotated images of the Endless Forams public library of 35
modern planktic foraminifera species. Our results on this data show leading 92%
accuracy (at 0.84 F1-score) in reproducing expert labels on withheld test data,
and 66.5% accuracy (at 0.70 F1-score) when clustering species never encountered
in training. We conclude that metric learning is highly effective for this
domain and serves as an important tool towards expert-in-the-loop automation of
microfossil identification. Key code, network weights, and data splits are
published with this paper for full reproducibility.

    

### [[2112.10814] Checkpoint-Restart Libraries Must Become More Fault Tolerant](http://arxiv.org/abs/2112.10814)


  Production MPI codes need checkpoint-restart (CPR) support. Clearly,
checkpoint-restart libraries must be fault tolerant lest they open up a window
of vulnerability for failures with byzantine outcomes. But, certain popular
libraries that leverage MPI are evidently not fault tolerant. Nowadays, fault
detection with automatic recovery without batch requeueing is a strong
requirement for production environments. Thus, allowing deadlock and setting
long timeouts are suboptimal for fault detection even when paired with
conservative recovery from the penultimate checkpoint.
When MPI is used as a communication mechanism within a CPR library, such
libraries must offer fault-tolerant extensions with minimal detection,
isolation, mitigation, and potential recovery semantics to aid the CPR's
library fail-backward. Communication between MPI and the checkpoint library
regarding system health may be valuable. For fault-tolerant MPI programs (e.g.,
using APIs like FA-MPI, Stages/Reinit, or ULFM), the checkpoint library must
cooperate with the extended model or else invalidate fault-tolerant operation.

    

### [[2112.10913] Accelerating Clique Counting in Sparse Real-World Graphs via Communication-Reducing Optimizations](http://arxiv.org/abs/2112.10913)


  Counting instances of specific subgraphs in a larger graph is an important
problem in graph mining. Finding cliques of size k (k-cliques) is one example
of this NP-hard problem. Different algorithms for clique counting avoid
counting the same clique multiple times by pivoting or ordering the graph.
Ordering-based algorithms include an ordering step to direct the edges in the
input graph, and a counting step, which is dominated by building node or
edge-induced subgraphs. Of the ordering-based algorithms, kClist is the
state-of-the art algorithm designed to work on sparse real-world graphs.
Despite its leading overall performance, kClist's vertex-parallel
implementation does not scale well in practice on graphs with a few million
vertices.
We present CITRON (Clique counting with Traffic Reducing Optimizations) to
improve the parallel scalability and thus overall performance of clique
counting. We accelerate the ordering phase by abandoning kClist's sequential
core ordering and using a parallelized degree ordering. We accelerate the
counting phase with our reorganized subgraph data structures that reduce memory
traffic to improve scaling bottlenecks. Our sorted, compact neighbor lists
improve locality and communication efficiency which results in near-linear
parallel scaling. CITRON significantly outperforms kClist while counting
moderately sized cliques, and thus increases the size of graph practical for
clique counting.
We have recently become aware of ArbCount (arXiv:2002.10047), which often
outperforms us. However, we believe that the analysis included in this paper
will be helpful for anyone who wishes to understand the performance
characteristics of k-clique counting.

    

### [[2112.11024] Reputation-based PoS for the Restriction of Illicit Activities on Blockchain: Algorand Usecase](http://arxiv.org/abs/2112.11024)


  In cryptocurrency-based permissionless blockchain networks, the decentralized
structure enables any user to join and operate across different regions. The
criminal entities exploit it by using cryptocurrency transactions on the
blockchain to facilitate activities such as money laundering, gambling, and
ransomware attacks. In recent times, different machine learning-based
techniques can detect such criminal elements based on blockchain transaction
data. However, there is no provision within the blockchain to deal with such
elements. We propose a reputation-based methodology for response to the users
detected carrying out the aforementioned illicit activities. We select Algorand
blockchain to implement our methodology by incorporating it within the
consensus protocol. The theoretical results obtained prove the restriction and
exclusion of criminal elements through block proposal rejection and attenuation
of the voting power as a validator for such entities. Further, we analyze the
efficacy of our method and show that it puts no additional strain on the
communication resources.

    

### [[2112.11030] Maxwell: a hardware and software highly integrated compute-storage system](http://arxiv.org/abs/2112.11030)


  The compute-storage framework is responsible for data storage and processing,
and acts as the digital chassis of all upper-level businesses. The performance
of the framework affects the business's processing throughput, latency, jitter,
and etc., and also determines the theoretical performance upper bound that the
business can achieve. In financial applications, the compute-storage framework
must have high reliability and high throughput, but with low latency as well as
low jitter characteristics. For some scenarios such as hot-spot account update,
the performance of the compute-storage framework even surfaces to become a
server performance bottleneck of the whole business system. In this paper, we
study the hot-spot account issue faced by Alipay and present our exciting
solution to this problem by developing a new compute-storage system, called
Maxwell. Maxwell is a distributed compute-storage system with integrated
hardware and software optimizations. Maxwell does not rely on any specific
hardware (e.g. GPUs or FPGAs). Instead, it takes deep advantage of computer
components' characteristics, such as disk, network, operating system and CPU,
and aims to emit the ultimate performance of both hardware and software. In
comparison with the existing hot-spot account updating solutions deployed
online, Maxwell achieves three orders of magnitude performance improvement for
end-to-end evaluation. Meanwhile, Maxwell also demonstrates remarkable
performance gains in other related businesses of Ant Group.

    

### [[2112.11072] BlockReduce -- Scaling Proof-of-Work Blockchains](http://arxiv.org/abs/2112.11072)


  This paper introduces BlockReduce, a Proof-of-Work (PoW) based blockchain
system which achieves high transaction throughput through a hierarchy of merged
mined blockchains, each operating in parallel on a partition the overall
application state. Most notably, the full PoW available within the network is
applied to all blockchains in BlockReduce, and cross-blockchain state
transitions are enabled seamlessly within the core protocol. This paper shows
that, given a hierarchy of blockchains and its associated security model, the
protocol scales superlinearly in transaction throughput with the number of
blockchains operated by the protocol.

    

### [[2112.11337] Byzantine Fault Tolerant Causal Ordering](http://arxiv.org/abs/2112.11337)


  Causal ordering in an asynchronous system has many applications in
distributed computing, including in replicated databases and real-time
collaborative software. Previous work in the area focused on ordering
point-to-point messages in a fault-free setting, and on ordering broadcasts
under various fault models. To the best of our knowledge, Byzantine
fault-tolerant causal ordering has not been attempted for point-to-point
communication in an asynchronous setting. In this paper, we first show that
existing algorithms for causal ordering of point-to-point communication fail
under Byzantine faults. We then prove that it is impossible to causally order
messages under point-to-point communication in an asynchronous system with one
or more Byzantine failures. We then present two algorithms that can causally
order messages under Byzantine failures, where the network provides an upper
bound on the message transmission time. The proofs of correctness for these
algorithms show that it is possible to achieve causal ordering for
point-to-point communication under a stronger asynchrony model where the
network provides an upper bound on message transmission time. We also give
extensions of our two algorithms for Byzantine fault-tolerant causal ordering
of multicasts.

    

### [[2005.10103] BeepTrace: Blockchain-enabled Privacy-preserving Contact Tracing for COVID-19 Pandemic and Beyond](http://arxiv.org/abs/2005.10103)


  The outbreak of COVID-19 pandemic has exposed an urgent need for effective
contact tracing solutions through mobile phone applications to prevent the
infection from spreading further. However, due to the nature of contact
tracing, public concern on privacy issues has been a bottleneck to the existing
solutions, which is significantly affecting the uptake of contact tracing
applications across the globe. In this paper, we present a blockchain-enabled
privacy-preserving contact tracing scheme: BeepTrace, where we propose to adopt
blockchain bridging the user/patient and the authorized solvers to desensitize
the user ID and location information. Compared with recently proposed contract
tracing solutions, our approach shows higher security and privacy with the
additional advantages of being battery friendly and globally accessible.
Results show viability in terms of the required resource at both server and
mobile phone perspectives. Through breaking the privacy concerns of the public,
the proposed BeepTrace solution can provide a timely framework for authorities,
companies, software developers and researchers to fast develop and deploy
effective digital contact tracing applications, to conquer COVID-19 pandemic
soon. Meanwhile, the open initiative of BeepTrace allows worldwide
collaborations, integrate existing tracing and positioning solutions with the
help of blockchain technology.

    

### [[2102.00096] A Categorical Semantics for Hierarchical Petri Nets](http://arxiv.org/abs/2102.00096)


  We show how a particular variety of hierarchical nets, where the firing of a
transition in the parent net must correspond to an execution in some child net,
can be modelled utilizing a functorial semantics from a free category --
representing the parent net -- to the category of sets and spans between them.
This semantics can be internalized via Grothendieck construction, resulting in
the category of executions of a Petri net representing the semantics of the
overall hierarchical net. We conclude the paper by giving an
engineering-oriented overview of how our model of hierarchical nets can be
implemented in a transaction-based smart contract environment.

    

### [[2112.10807] Demonstration Informed Specification Search](http://arxiv.org/abs/2112.10807)


  This paper considers the problem of learning history dependent task
specifications, e.g. automata and temporal logic, from expert demonstrations.
Unfortunately, the (countably infinite) number of tasks under consideration
combined with an a-priori ignorance of what historical features are needed to
encode the demonstrated task makes existing approaches to learning tasks from
demonstrations inapplicable. To address this deficit, we propose Demonstration
Informed Specification Search (DISS): a family of algorithms parameterized by
black box access to (i) a maximum entropy planner and (ii) an algorithm for
identifying concepts, e.g., automata, from labeled examples. DISS works by
alternating between (i) conjecturing labeled examples to make the
demonstrations less surprising and (ii) sampling concepts consistent with the
current labeled examples. In the context of tasks described by deterministic
finite automata, we provide a concrete implementation of DISS that efficiently
combines partial knowledge of the task and a single expert demonstration to
identify the full task specification.

    

### [[2112.10890] Fast Algorithms for Poker Require Modelling it as a Sequential Bayesian Game](http://arxiv.org/abs/2112.10890)


  Many recent results in imperfect information games were only formulated for,
or evaluated on, poker and poker-like games such as liar's dice. We argue that
sequential Bayesian games constitute a natural class of games for generalizing
these results. In particular, this model allows for an elegant formulation of
the counterfactual regret minimization algorithm, called public-state CFR
(PS-CFR), which naturally lends itself to an efficient implementation.
Empirically, solving a poker subgame with 10^7 states by public-state CFR takes
3 minutes and 700 MB while a comparable version of vanilla CFR takes 5.5 hours
and 20 GB. Additionally, the public-state formulation of CFR opens up the
possibility for exploiting domain-specific assumptions, leading to a quadratic
reduction in asymptotic complexity (and a further empirical speedup) over
vanilla CFR in poker and other domains. Overall, this suggests that the ability
to represent poker as a sequential Bayesian game played a key role in the
success of CFR-based methods. Finally, we extend public-state CFR to general
extensive-form games, arguing that this extension enjoys some - but not all -
of the benefits of the version for sequential Bayesian games.

    

### [[2112.10892] A Constraint Programming Approach to Weighted Isomorphic Mapping of Fragment-based Shape Signatures](http://arxiv.org/abs/2112.10892)


  Fragment-based shape signature techniques have proven to be powerful tools
for computer-aided drug design. They allow scientists to search for target
molecules with some similarity to a known active compound. They do not require
reference to the full underlying chemical structure, which is essential to deal
with chemical databases containing millions of compounds. However, finding the
optimal match of a part of the fragmented compound can be time-consuming. In
this paper, we use constraint programming to solve this specific problem. It
involves finding a weighted assignment of fragments subject to connectivity
constraints. Our experiments demonstrate the practical relevance of our
approach and open new perspectives, including generating multiple, diverse
solutions. Our approach constitutes an original use of a constraint solver in a
real time setting, where propagation allows to avoid an enumeration of weighted
paths. The model must remain robust to the addition of additional constraints
making some instances not tractable. This particular context requires the use
of unusual criteria for the choice of the model: lightweight, standard
propagation algorithms, data structures without prohibitive constant cost while
reducing the search space. The objective is not to design new, complex
algorithms to solve difficult instances.

    

### [[2112.10925] DB-BERT: a Database Tuning Tool that "Reads the Manual"](http://arxiv.org/abs/2112.10925)


  DB-BERT is a database tuning tool that exploits information gained via
natural language analysis of manuals and other relevant text documents. It uses
text to identify database system parameters to tune as well as recommended
parameter values. DB-BERT applies large, pre-trained language models
(specifically, the BERT model) for text analysis. During an initial training
phase, it fine-tunes model weights in order to translate natural language hints
into recommended settings. At run time, DB-BERT learns to aggregate, adapt, and
prioritize hints to achieve optimal performance for a specific database system
and benchmark. Both phases are iterative and use reinforcement learning to
guide the selection of tuning settings to evaluate (penalizing settings that
the database system rejects while rewarding settings that improve performance).
In our experiments, we leverage hundreds of text documents about database
tuning as input for DB-BERT. We compare DB-BERT against various baselines,
considering different benchmarks (TPC-C and TPC-H), metrics (throughput and run
time), as well as database systems (Postgres and MySQL). In all cases, DB-BERT
finds the best parameter settings among all compared methods. The code of
DB-BERT is available online at this https URL.

    

### [[2112.10936] Watch Those Words: Video Falsification Detection Using Word-Conditioned Facial Motion](http://arxiv.org/abs/2112.10936)


  In today's era of digital misinformation, we are increasingly faced with new
threats posed by video falsification techniques. Such falsifications range from
cheapfakes (e.g., lookalikes or audio dubbing) to deepfakes (e.g.,
sophisticated AI media synthesis methods), which are becoming perceptually
indistinguishable from real videos. To tackle this challenge, we propose a
multi-modal semantic forensic approach to discover clues that go beyond
detecting discrepancies in visual quality, thereby handling both simpler
cheapfakes and visually persuasive deepfakes. In this work, our goal is to
verify that the purported person seen in the video is indeed themselves by
detecting anomalous correspondences between their facial movements and the
words they are saying. We leverage the idea of attribution to learn
person-specific biometric patterns that distinguish a given speaker from
others. We use interpretable Action Units (AUs) to capture a persons' face and
head movement as opposed to deep CNN visual features, and we are the first to
use word-conditioned facial motion analysis. Unlike existing person-specific
approaches, our method is also effective against attacks that focus on lip
manipulation. We further demonstrate our method's effectiveness on a range of
fakes not seen in training including those without video manipulation, that
were not addressed in prior work.

    

### [[2112.10960] Continuous-Time Video Generation via Learning Motion Dynamics with Neural ODE](http://arxiv.org/abs/2112.10960)


  In order to perform unconditional video generation, we must learn the
distribution of the real-world videos. In an effort to synthesize high-quality
videos, various studies attempted to learn a mapping function between noise and
videos, including recent efforts to separate motion distribution and appearance
distribution. Previous methods, however, learn motion dynamics in discretized,
fixed-interval timesteps, which is contrary to the continuous nature of motion
of a physical body. In this paper, we propose a novel video generation approach
that learns separate distributions for motion and appearance, the former
modeled by neural ODE to learn natural motion dynamics. Specifically, we employ
a two-stage approach where the first stage converts a noise vector to a
sequence of keypoints in arbitrary frame rates, and the second stage
synthesizes videos based on the given keypoints sequence and the appearance
noise vector. Our model not only quantitatively outperforms recent baselines
for video generation, but also demonstrates versatile functionality such as
dynamic frame rate manipulation and motion transfer between two datasets, thus
opening new doors to diverse video generation applications.

    

### [[2112.11023] Robust Recommendation with Implicit Feedback via Eliminating the Effects of Unexpected Behaviors](http://arxiv.org/abs/2112.11023)


  In the implicit feedback recommendation, incorporating short-term preference
into recommender systems has attracted increasing attention in recent years.
However, unexpected behaviors in historical interactions like clicking some
items by accident don't well reflect users' inherent preferences. Existing
studies fail to model the effects of unexpected behaviors, thus achieve
inferior recommendation performance. In this paper, we propose a
Multi-Preferences Model (MPM) to eliminate the effects of unexpected behaviors.
MPM first extracts the users' instant preferences from their recent historical
interactions by a fine-grained preference module. Then an unexpected-behaviors
detector is trained to judge whether these instant preferences are biased by
unexpected behaviors. We also integrate user's general preference in MPM.
Finally, an output module is performed to eliminate the effects of unexpected
behaviors and integrates all the information to make a final recommendation. We
conduct extensive experiments on two datasets of a movie and an e-retailing,
demonstrating significant improvements in our model over the state-of-the-art
methods. The experimental results show that MPM gets a massive improvement in
HR@10 and NDCG@10, which relatively increased by 3.643% and 4.107% compare with
AttRec model on average. We publish our code at
this https URL.

    

### [[2112.11176] Task-oriented Dialogue Systems: performance vs. quality-optima, a review](http://arxiv.org/abs/2112.11176)


  Task-oriented dialogue systems (TODS) are continuing to rise in popularity as
various industries find ways to effectively harness their capabilities, saving
both time and money. However, even state-of-the-art TODS are not yet reaching
their full potential. TODS typically have a primary design focus on completing
the task at hand, so the metric of task-resolution should take priority. Other
conversational quality attributes that may point to the success, or otherwise,
of the dialogue, may be ignored. This can cause interactions between human and
dialogue system that leave the user dissatisfied or frustrated. This paper
explores the literature on evaluative frameworks of dialogue systems and the
role of conversational quality attributes in dialogue systems, looking at if,
how, and where they are utilised, and examining their correlation with the
performance of the dialogue system.

    

### [[2112.11193] There is an elephant in the room: Towards a critique on the use of fairness in biometrics](http://arxiv.org/abs/2112.11193)


  In 2019, the UK's Immigration and Asylum Chamber of the Upper Tribunal
dismissed an asylum appeal basing the decision on the output of a biometric
system, alongside other discrepancies. The fingerprints of the asylum seeker
were found in a biometric database which contradicted the appellant's account.
The Tribunal found this evidence unequivocal and denied the asylum claim.
Nowadays, the proliferation of biometric systems is shaping public debates
around its political, social and ethical implications. Yet whilst concerns
towards the racialised use of this technology for migration control have been
on the rise, investment in the biometrics industry and innovation is increasing
considerably. Moreover, fairness has also been recently adopted by biometrics
to mitigate bias and discrimination on biometrics. However, algorithmic
fairness cannot distribute justice in scenarios which are broken or intended
purpose is to discriminate, such as biometrics deployed at the border.
In this paper, we offer a critical reading of recent debates about biometric
fairness and show its limitations drawing on research in fairness in machine
learning and critical border studies. Building on previous fairness
demonstrations, we prove that biometric fairness criteria are mathematically
mutually exclusive. Then, the paper moves on illustrating empirically that a
fair biometric system is not possible by reproducing experiments from previous
works. Finally, we discuss the politics of fairness in biometrics by situating
the debate at the border. We claim that bias and error rates have different
impact on citizens and asylum seekers. Fairness has overshadowed the elephant
in the room of biometrics, focusing on the demographic biases and ethical
discourses of algorithms rather than examine how these systems reproduce
historical and political injustices.

    

### [[2112.11195] Building a Decision Support System for Automated Mobile Asthma Monitoring in Remote Areas](http://arxiv.org/abs/2112.11195)


  Advances in mobile computing have paved the way for the development of
several health applications using smartphone as a platform for data
acquisition, analysis and presentation. Such areas where mhealth systems have
been extensively deployed include monitoring of long term health conditions
like Cardio Vascular Diseases and pulmonary disorders, as well as detection of
changes from baseline measurements of such conditions. Asthma is one of the
respiratory conditions with growing concern across the globe due to the
economic, social and emotional burden associated with the ailment. The
management and control of asthma can be improved by consistent monitoring of
the condition in realtime since attack could occur anytime and anywhere. This
paper proposes the use of smartphone equipped with embedded sensors, to capture
and analyze early symptoms of asthma triggered by exercise. The system design
is based on Decision Support System techniques for measuring and analyzing the
level and type of patients physical activity as well as weather conditions that
predispose asthma attack. Preliminary results show that smartphones can be used
to monitor and detect asthma symptoms without other networked devices. This
would enhance the usability of the health system while ensuring users data
privacy, and reducing the overall cost of system deployment. Further, the
proposed system can serve as a handy tool for a quick medical response for
asthmatics in low income countries where there are limited access to
specialized medical devices and shortages of health professionals. Development
of such monitoring systems signals a positive response to lessen the global
burden of asthma.

    

### [[2112.11208] Artificial Intelligence Ethics and Safety: practical tools for creating "good" models](http://arxiv.org/abs/2112.11208)


  The AI Robotics Ethics Society (AIRES) is a non-profit organization founded
in 2018 by Aaron Hui to promote awareness and the importance of ethical
implementation and regulation of AI. AIRES is now an organization with chapters
at universities such as UCLA (Los Angeles), USC (University of Southern
California), Caltech (California Institute of Technology), Stanford University,
Cornell University, Brown University, and the Pontifical Catholic University of
Rio Grande do Sul (Brazil). AIRES at PUCRS is the first international chapter
of AIRES, and as such, we are committed to promoting and enhancing the AIRES
Mission. Our mission is to focus on educating the AI leaders of tomorrow in
ethical principles to ensure that AI is created ethically and responsibly. As
there are still few proposals for how we should implement ethical principles
and normative guidelines in the practice of AI system development, the goal of
this work is to try to bridge this gap between discourse and praxis. Between
abstract principles and technical implementation. In this work, we seek to
introduce the reader to the topic of AI Ethics and Safety. At the same time, we
present several tools to help developers of intelligent systems develop "good"
models. This work is a developing guide published in English and Portuguese.
Contributions and suggestions are welcome.

    

### [[2112.11233] A next-generation platform for Cyber Range-as-a-Service](http://arxiv.org/abs/2112.11233)


  In the last years, Cyber Ranges have become a widespread solution to train
professionals for responding to cyber threats and attacks. Cloud computing
plays a key role in this context since it enables the creation of virtual
infrastructures on which Cyber Ranges are based. However, the setup and
management of Cyber Ranges are expensive and time-consuming activities. In this
paper, we highlight the novel features for the next-generation Cyber Range
platforms. In particular, these features include the creation of a virtual
clone for an actual corporate infrastructure, relieving the security managers
from the setup of the training scenarios and sessions, the automatic monitoring
of the participants' activities, and the emulation of their behavior.

    

### [[2112.11244] Hateful Memes Challenge: An Enhanced Multimodal Framework](http://arxiv.org/abs/2112.11244)


  Hateful Meme Challenge proposed by Facebook AI has attracted contestants
around the world. The challenge focuses on detecting hateful speech in
multimodal memes. Various state-of-the-art deep learning models have been
applied to this problem and the performance on challenge's leaderboard has also
been constantly improved. In this paper, we enhance the hateful detection
framework, including utilizing Detectron for feature extraction, exploring
different setups of VisualBERT and UNITER models with different loss functions,
researching the association between the hateful memes and the sensitive text
features, and finally building ensemble method to boost model performance. The
AUROC of our fine-tuned VisualBERT, UNITER, and ensemble method achieves 0.765,
0.790, and 0.803 on the challenge's test set, respectively, which beats the
baseline models. Our code is available at
this https URL


### [[2112.11255] Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems](http://arxiv.org/abs/2112.11255)


  Safe deployment of self-driving cars (SDC) necessitates thorough simulated
and in-field testing. Most testing techniques consider virtualized SDCs within
a simulation environment, whereas less effort has been directed towards
assessing whether such techniques transfer to and are effective with a physical
real-world vehicle. In this paper, we leverage the Donkey Car open-source
framework to empirically compare testing of SDCs when deployed on a physical
small-scale vehicle vs its virtual simulated counterpart. In our empirical
study, we investigate the transferability of behavior and failure exposure
between virtual and real-world environments on a vast set of corrupted and
adversarial settings. While a large number of testing results do transfer
between virtual and physical environments, we also identified critical
shortcomings that contribute to the reality gap between the virtual and
physical world, threatening the potential of existing testing solutions when
applied to physical SDCs.

    

### [[2112.11362] Reasoning About Causal Models With Infinitely Many Variables](http://arxiv.org/abs/2112.11362)


  Generalized structural equations models (GSEMs) [Peters and Halpern 2021],
are, as the name suggests, a generalization of structural equations models
(SEMs). They can deal with (among other things) infinitely many variables with
infinite ranges, which is critical for capturing dynamical systems. We provide
a sound and complete axiomatization of causal reasoning in GSEMs that is an
extension of the sound and complete axiomatization provided by Halpern [2000]
for SEMs. Considering GSEMs helps clarify what properties Halpern's axioms
capture.

    

### [[2112.11444] ESAN: Efficient Sentiment Analysis Network of A-Shares Research Reports for Stock Price Prediction](http://arxiv.org/abs/2112.11444)


  In this paper, we are going to develop a natural language processing model to
help us to predict stocks in the long term. The whole network includes two
modules. The first module is a natural language processing model which seeks
out reliable factors from input reports. While the other is a time-series
forecasting model which takes the factors as input and aims to predict stocks
earnings yield. To indicate the efficiency of our model to combine the
sentiment analysis module and the time-series forecasting module, we name our
method ESAN.

    

### [[2112.11446] Scaling Language Models: Methods, Analysis & Insights from Training Gopher](http://arxiv.org/abs/2112.11446)


  Language modelling provides a step towards intelligent communication systems
by harnessing large repositories of written human knowledge to better predict
and understand the world. In this paper, we present an analysis of
Transformer-based language model performance across a wide range of model
scales -- from models with tens of millions of parameters up to a 280 billion
parameter model called Gopher. These models are evaluated on 152 diverse tasks,
achieving state-of-the-art performance across the majority. Gains from scale
are largest in areas such as reading comprehension, fact-checking, and the
identification of toxic language, but logical and mathematical reasoning see
less benefit. We provide a holistic analysis of the training dataset and
model's behaviour, covering the intersection of model scale with bias and
toxicity. Finally we discuss the application of language models to AI safety
and the mitigation of downstream harms.

    

### [[2112.11447] Multi-Modality Distillation via Learning the teacher's modality-level Gram Matrix](http://arxiv.org/abs/2112.11447)


  In the context of multi-modality knowledge distillation research, the
existing methods was mainly focus on the problem of only learning teacher final
output. Thus, there are still deep differences between the teacher network and
the student network. It is necessary to force the student network to learn the
modality relationship information of the teacher network. To effectively
exploit transfering knowledge from teachers to students, a novel modality
relation distillation paradigm by modeling the relationship information among
different modality are adopted, that is learning the teacher modality-level
Gram Matrix.

    

### [[2106.10197] A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidents](http://arxiv.org/abs/2106.10197)


  The rapid advancement of sensor technologies and artificial intelligence are
creating new opportunities for traffic safety enhancement. Dashboard cameras
(dashcams) have been widely deployed on both human driving vehicles and
automated driving vehicles. A computational intelligence model that can
accurately and promptly predict accidents from the dashcam video will enhance
the preparedness for accident prevention. The spatial-temporal interaction of
traffic agents is complex. Visual cues for predicting a future accident are
embedded deeply in dashcam video data. Therefore, the early anticipation of
traffic accidents remains a challenge. Inspired by the attention behavior of
humans in visually perceiving accident risks, this paper proposes a Dynamic
Spatial-Temporal Attention (DSTA) network for the early accident anticipation
from dashcam videos. The DSTA-network learns to select discriminative temporal
segments of a video sequence with a Dynamic Temporal Attention (DTA) module. It
also learns to focus on the informative spatial regions of frames with a
Dynamic Spatial Attention (DSA) module. A Gated Recurrent Unit (GRU) is trained
jointly with the attention modules to predict the probability of a future
accident. The evaluation of the DSTA-network on two benchmark datasets confirms
that it has exceeded the state-of-the-art performance. A thorough ablation
study that assesses the DSTA-network at the component level reveals how the
network achieves such performance. Furthermore, this paper proposes a method to
fuse the prediction scores from two complementary models and verifies its
effectiveness in further boosting the performance of early accident
anticipation.

    

### [[2112.10869] Cell-Free Massive MIMO Meets OTFS Modulation](http://arxiv.org/abs/2112.10869)


  We provide the first-ever performance evaluation of orthogonal time frequency
space (OTFS) modulation in cell-free massive multiple-input multiple-output
(MIMO) systems. To investigate trade-off between performance and overhead, we
apply embedded pilot-aided and superimposed pilot-based channel estimation
methods. We then derive a closed-form expression for the individual user
downlink and uplink spectral efficiencies as a function of the numbers of APs,
users and delay-Doppler domain channel estimate parameters. Based on these
analytical results, we also present new scaling laws that the AP's and user's
transmit power should satisfy, to sustain a desirable quality of service. It is
found that when the number of APs, $M_a$, grows without bound, we can reduce
the transmit power of each user and AP proportionally to $1/M_a$ and $1/M_a^2$,
respectively, during the uplink and downlink phases. We compare the OTFS
performance with that of orthogonal frequency division multiplexing (OFDM) at
high-mobility conditions. Our findings reveal that with shadowing correlation,
OTFS modulation with embedded pilot-based channel estimation provides
$30$-folds gain over the OFDM counterpart in terms of $95\%$-likely per-user
downlink rate. Finally, with superimposed pilot-based channel estimation, the
increase in the per-user throughput is more pronounced at the median rates over
the correlated shadowing channels.

    

### [[2112.11270] Adding semantics to measurements: Ontology-guided, systematic performance analysis](http://arxiv.org/abs/2112.11270)


  The design and operation of modern software systems exhibit a shift towards
virtualization, containerization and service-based orchestration. Performance
capacity engineering and resource utilization tuning become priority
requirements in such environments.
Measurement-based performance evaluation is the cornerstone of capacity
engineering and designing for performance. Moreover, the increasing complexity
of systems necessitates rigorous performance analysis approaches. However,
empirical performance analysis lacks sophisticated model-based support similar
to the functional design of the system.
The paper proposes an ontology-based approach for facilitating and guiding
the empirical evaluation throughout its various steps. Hyperledger Fabric
(HLF), an open-source blockchain platform by the Linux Foundation, is modelled
and evaluated as a pilot example of the approach, using the standard TPC-C
performance benchmark workload.

    

### [[2112.11277] Porting a benchmark with a classic workload to blockchain: TPC-C on Hyperledger Fabric](http://arxiv.org/abs/2112.11277)


  Many cross-organization cooperation applications of blockchain-based
distributed ledger technologies (DLT) do not aim at innovation at the
cooperation pattern level: essentially the same ''business'' is conducted by
the parties, but this time without a central party to be trusted with
bookkeeping. The migration to DLT is expected to have a negative performance
impact, but some DLTs, such as Hyperledger Fabric, are accepted to be much
better suited performance-wise to such use cases than others. However, with the
somewhat surprising, but ongoing absence of application-level performance
benchmarks for DLTs, cross-DLT comparison for "classic" workloads and the
evaluation of the performance impact of "blockchainification" is still
ill-supported. We present the design and Hyperledger Caliper-based open
implementation of a full port of the classic TPC-C benchmark to Hyperledger
Fabric, complete with a structured approach for transforming the original
database schema to a smart contract data model. Initial measurements about the
workload characteristics that will affect the design of large-scale performance
evaluations are also included.

    

### [[1911.07434] Ultra-Fast Accurate AoA Estimation via Automotive Massive-MIMO Radar](http://arxiv.org/abs/1911.07434)


  Massive multiple-input multiple-output (MIMO) radar, enabled by
millimeter-wave virtual MIMO techniques, provides great promises to the
high-resolution automotive sensing and target detection in unmanned
ground/aerial vehicles (UGA/UAV). As a long-established problem, however,
existing subspace methods suffer from either high complexity or low accuracy.
In this work, we propose two efficient methods, to accomplish fast subspace
computation and accurate angle of arrival (AoA) acquisition. By leveraging
randomized low-rank approximation, our fast multiple signal classification
(MUSIC) methods, relying on random sampling and projection techniques,
substantially accelerate the subspace estimation by orders of magnitude.
Moreover, we establish the theoretical bounds of our proposed methods, which
ensure the accuracy of the approximated pseudo-spectrum. As demonstrated, the
pseudo-spectrum acquired by our fast-MUSIC would be highly precise; and the
estimated AoA is almost as accurate as standard MUSIC. In contrast, our new
methods are tremendously faster than standard MUSIC. Thus, our fast-MUSIC
enables the high-resolution real-time environmental sensing with massive MIMO
radars, which has great potential in the emerging unmanned systems.

    

### [[2112.11077] A Small-Step Operational Semantics for GP 2](http://arxiv.org/abs/2112.11077)


  The operational semantics of a programming language is said to be small-step
if each transition step is an atomic computation step in the language. A
semantics with this property faithfully corresponds to the implementation of
the language. The previous semantics of the graph programming language GP 2 is
not fully small-step because the loop and branching commands are defined in
big-step style. In this paper, we present a truly small-step operational
semantics for GP 2 which, in particular, accurately models diverging
computations. To obtain small-step definitions of all commands, we equip the
transition relation with a stack of host graphs and associated operations. We
prove that the new semantics is non-blocking in that every computation either
diverges or eventually produces a result graph or the failure state. We also
show the finite nondeterminism property, viz. that each configuration has only
a finite number of direct successors. The previous semantics of GP 2 is neither
non-blocking nor does it have the finite nondeterminism property. We also show
that, for a program and a graph that terminate, both semantics are equivalent,
and that the old semantics can be simulated with the new one.

    

### [[2112.11101] Chat2Code: Towards conversational concrete syntax for model specification and code generation, the case of smart contracts](http://arxiv.org/abs/2112.11101)


  The revolutionary potential of automatic code generation tools based on
Model-Driven Engineering (MDE) frameworks has yet to be realized. Beyond their
ability to help software professionals write more accurate, reusable code, they
could make programming accessible for a whole new class of non-technical users.
However, non-technical users have been slow to embrace these tools. This may be
because their concrete syntax is often patterned after the operations of
textual or graphical interfaces. The interfaces are common, but users would
need more extensive, precise and detailed knowledge of them than they can be
assumed to have, to use them as concrete syntax.
Conversational interfaces (chatbots) offer a much more accessible way for
non-technical users to generate code. In this paper, we discuss the basic
challenge of integrating conversational agents within Model-Driven Engineering
(MDE) frameworks, then turn to look at a specific application: the
auto-generation of smart contract code in multiple languages by non-technical
users, based on conversational syntax. We demonstrate how this can be done, and
evaluate our approach by conducting user experience survey to assess the
usability and functionality of the chatbot framework.

    

### [[2005.11023] Symbolic Reasoning about Quantum Circuits in Coq](http://arxiv.org/abs/2005.11023)


  A quantum circuit is a computational unit that transforms an input quantum
state to an output one. A natural way to reason about its behavior is to
compute explicitly the unitary matrix implemented by it. However, when the
number of qubits increases, the matrix dimension grows exponentially and the
computation becomes intractable.
In this paper, we propose a symbolic approach to reasoning about quantum
circuits. It is based on a small set of laws involving some basic manipulations
on vectors and matrices. This symbolic reasoning scales better than the
explicit one and is well suited to be automated in Coq, as demonstrated with
some typical examples.

    