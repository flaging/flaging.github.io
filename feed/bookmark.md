
## 2021-11-19

### [<title>Error when using dataframe - XGBoost</title>](https://discuss.xgboost.ai/t/error-when-using-dataframe/2545/1)

### [[2111.09410] EdgeML: Towards Network-Accelerated Federated Learning over Wireless Edge](http://arxiv.org/abs/2111.09410)


  Federated learning (FL) is a distributed machine learning technology for
next-generation AI systems that allows a number of workers, i.e., edge devices,
collaboratively learn a shared global model while keeping their data locally to
prevent privacy leakage. Enabling FL over wireless multi-hop networks can
democratize AI and make it accessible in a cost-effective manner. However, the
noisy bandwidth-limited multi-hop wireless connections can lead to delayed and
nomadic model updates, which significantly slows down the FL convergence speed.
To address such challenges, this paper aims to accelerate FL convergence over
wireless edge by optimizing the multi-hop federated networking performance. In
particular, the FL convergence optimization problem is formulated as a Markov
decision process (MDP). To solve such MDP, multi-agent reinforcement learning
(MA-RL) algorithms along with domain-specific action space refining schemes are
developed, which online learn the delay-minimum forwarding paths to minimize
the model exchange latency between the edge devices (i.e., workers) and the
remote server. To validate the proposed solutions, FedEdge is developed and
implemented, which is the first experimental framework in the literature for FL
over multi-hop wireless edge computing networks. FedEdge allows us to fast
prototype, deploy, and evaluate novel FL algorithms along with RL-based system
optimization methods in real wireless devices. Moreover, a physical
experimental testbed is implemented by customizing the widely adopted Linux
wireless routers and ML computing nodes.Finally, our experimentation results on
the testbed show that the proposed network-accelerated FL system can
practically and significantly improve FL convergence speed, compared to the FL
system empowered by the production-grade commercially available wireless
networking protocol, BATMAN-Adv.

    

### [[2111.09411] Multi-sided Matching for the Association of Space-Air-Ground Integrated Systems](http://arxiv.org/abs/2111.09411)


  Space-air-ground integrated networks (SAGINs) will play a key role in 6G
communication systems. They are considered a promising technology to enhance
the network capacity in highly dense agglomerations and to provide connectivity
in rural areas. The multi-layer and heterogeneous nature of SAGINs necessitates
an innovative design of their multi-tier associations. We propose a modeling of
the SAGINs association problem using multi-sided matching theory. Our aim is to
provide a reliable, asynchronous and fully distributed approach that associates
nodes across the layers so that the total end-to-end rate of the assigned
agents is maximized. To this end, our problem is modeled as a multi-sided
many-to-one matching game. A randomized matching algorithm with low information
exchange is proposed. The algorithm is shown to reach an efficient and stable
association between nodes in adjacent layers. Our simulation results show that
the proposed approach achieves significant gain compared to the greedy and
distance-based algorithms.

    

### [[2111.09412] Meta-Reinforcement Learning via Buffering Graph Signatures for Live Video Streaming Events](http://arxiv.org/abs/2111.09412)


  In this study, we present a meta-learning model to adapt the predictions of
the network's capacity between viewers who participate in a live video
streaming event. We propose the MELANIE model, where an event is formulated as
a Markov Decision Process, performing meta-learning on reinforcement learning
tasks. By considering a new event as a task, we design an actor-critic learning
scheme to compute the optimal policy on estimating the viewers' high-bandwidth
connections. To ensure fast adaptation to new connections or changes among
viewers during an event, we implement a prioritized replay memory buffer based
on the Kullback-Leibler divergence of the reward/throughput of the viewers'
connections. Moreover, we adopt a model-agnostic meta-learning framework to
generate a global model from past events. As viewers scarcely participate in
several events, the challenge resides on how to account for the low structural
similarity of different events. To combat this issue, we design a graph
signature buffer to calculate the structural similarities of several streaming
events and adjust the training of the global model accordingly. We evaluate the
proposed model on the link weight prediction task on three real-world datasets
of live video streaming events. Our experiments demonstrate the effectiveness
of our proposed model, with an average relative gain of 25% against
state-of-the-art strategies. For reproduction purposes, our evaluation datasets
and implementation are publicly available at
this https URL


### [[2111.09413] Mixed Dual-Hop IRS-Assisted FSO-RF Communication System with H-ARQ Protocols](http://arxiv.org/abs/2111.09413)


  Intelligent reflecting surface (IRS) is an emerging key technology for the
fifth-generation (5G) and beyond wireless communication systems to provide more
robust and reliable communication links. In this paper, we propose a mixed
dual-hop free-space optical (FSO)-radio frequency (RF) communication system
that serves the end user via a decode-and-forward (DF) relay employing hybrid
automatic repeat request (HARQ) protocols on both hops. Novel closed-form
expressions of the probability density function (PDF) and cumulative density
function (CDF) of the equivalent end-to-end signal-to-noise ratio (SNR) are
computed for the considered system. Utilizing the obtained statistics
functions, we derive the outage probability (OP) and packet error rate (PER) of
the proposed system by considering generalized detection techniques on the
source-to-relay (S-R) link with H-ARQ protocol and IRS having phase error. We
obtain useful insights into the system performance through the asymptotic
analysis which aids to compute the diversity gain. The derived analytical
results are validated using Monte Carlo simulation.

    

### [[2111.09416] Highly Accurate and Reliable Wireless Network Slicing in 5th Generation Networks: A Hybrid Deep Learning Approach](http://arxiv.org/abs/2111.09416)


  In the current era, the next-generation networks like 5th generation (5G) and
6th generation (6G) networks require high security, low latency with a high
reliable standards and capacity. In these networks, reconfigurable wireless
network slicing is considered as one of the key elements for 5G and 6G
networks. A reconfigurable slicing allows the operators to run various
instances of the network using a single infrastructure for a better quality of
services (QoS). The QoS can be achieved by reconfiguring and optimizing these
networks using Artificial intelligence and machine learning algorithms. To
develop a smart decision-making mechanism for network management and
restricting network slice failures, machine learning-enabled reconfigurable
wireless network solutions are required. In this paper, we propose a hybrid
deep learning model that consists of a convolution neural network (CNN) and
long short term memory (LSTM). The CNN performs resource allocation, network
reconfiguration, and slice selection while the LSTM is used for statistical
information (load balancing, error rate etc.) regarding network slices. The
applicability of the proposed model is validated by using multiple unknown
devices, slice failure, and overloading conditions. The overall accuracy of
95.17% is achieved by the proposed model that reflects its applicability.

    

### [[2111.09417] Blind Calibration of Air Quality Wireless Sensor Networks Using Deep Neural Networks](http://arxiv.org/abs/2111.09417)


  Temporal drift of low-cost sensors is crucial for the applicability of
wireless sensor networks (WSN) to measure highly local phenomenon such as air
quality. The emergence of wireless sensor networks in locations without
available reference data makes calibrating such networks without the aid of
true values a key area of research. While deep learning (DL) has proved
successful on numerous other tasks, it is under-researched in the context of
blind WSN calibration, particularly in scenarios with networks that mix static
and mobile sensors. In this paper we investigate the use of DL architectures
for such scenarios, including the effects of weather in both drifting and
sensor measurement. New models are proposed and compared against a baseline,
based on a previous proposed model and extended to include mobile sensors and
weather data. Also, a procedure for generating simulated air quality data is
presented, including the emission, dispersion and measurement of the two most
common particulate matter pollutants: PM 2.5 and PM 10 . Results show that our
models reduce the calibration error with an order of magnitude compared to the
baseline, showing that DL is a suitable method for WSN calibration and that
these networks can be remotely calibrated with minimal cost for the deployer.

    

### [[2111.09418] Impact of Weather Conditions on 5G Communication Channel under Connected Vehicles Framework](http://arxiv.org/abs/2111.09418)


  Recent research focused on improving the vehicle-to-vehicle communication
(V2V) based on the 5G technology. The V2V application is important because it
will reduce the risk of accidents up to 70%-80%, improve traffic management,
reduce congestion, and improve fuel consumption. Autonomous vehicles
applications require a high bandwidth transmission channel where the 5G
communication channel would be a reliable solution to support this disruptive
technology. The dedicated short-range communications (DSRC), which is
characterized with a frequency bandwidth of 5.9 gigahertz (GHz) (4G spectrum),
was used as vehicular connectivity with a bandwidth of up to 200 megabytes per
second (mb/s) and limited capacity. The 5G band can support connected multiple
autonomous vehicles with high data rates and large bandwidth. In this study,
the 5G communication channel is considered as vehicular connectivity with high
bandwidth in the millimeter waves spectrum range. The quality of 5G wireless
communication channels between connected vehicles possibly be affected by
weather conditions such as rain, snow, fog, dust, and sand. In this paper, we
estimate the effect of dust and sand on the propagation of millimeter waves.
The Mie model is used to investigate the effect of dust and sand storms on the
propagating mm-waves. The effect of dust and sand on the communication path
loss of DSRC and 5G frequency band is investigated in the case of urban freeway
and rural highway settings. Results show that the attenuation of dust and sand
is changed when the particle size of sand, frequency of propagating wave, and
concentration of dust are changed. Finally, the new model of link margin is
created to estimate the effect of dust and sand on DSCR (5.9 GHz) and 5G (28
GHz) communication path loss.

    

### [[2111.09420] Distributed Proximal Policy Optimization for Contention-Based Spectrum Access](http://arxiv.org/abs/2111.09420)


  The increasing number of wireless devices operating in unlicensed spectrum
motivates the development of intelligent adaptive approaches to spectrum access
that go beyond traditional carrier sensing. We develop a novel distributed
implementation of a policy gradient method known as Proximal Policy
Optimization modelled on a two stage Markov decision process that enables such
an intelligent approach, and still achieves decentralized contention-based
medium access. In each time slot, a base station (BS) uses information from
spectrum sensing and reception quality to autonomously decide whether or not to
transmit on a given resource, with the goal of maximizing proportional fairness
network-wide. Empirically, we find the proportional fairness reward accumulated
by the policy gradient approach to be significantly higher than even a
genie-aided adaptive energy detection threshold. This is further validated by
the improved sum and maximum user throughputs achieved by our approach.

    

### [[2111.09421] Low-to-Zero-Overhead IRS Reconfiguration: Decoupling Illumination and Channel Estimation](http://arxiv.org/abs/2111.09421)


  Most algorithms developed so far for the optimization of Intelligent
Reflecting Surfaces (IRSs) require knowledge of full Channel State Information
(CSI). However, the resulting acquisition overhead constitutes a major
bottleneck for the realization of IRS-assisted wireless systems in practice. In
contrast, in this paper, focusing on downlink transmissions from a Base Station
(BS) to a Mobile User (MU) that is located in a blockage region, we propose to
optimize the IRS for illumination of the area centered around the MU. Hence,
the proposed design requires the estimation of the MU's position and not the
full CSI. For a given IRS phase-shift configuration, the end-to-end BS-IRS-MU
channel can then be estimated using conventional channel estimation techniques.
The IRS reconfiguration overhead for the proposed scheme depends on the MU
mobility as well as how wide the coverage of the IRS illumination is.
Therefore, we develop a general IRS phase-shift design, which is valid for both
the near- and far-field regimes and features a parameter for tuning the size of
the illumination area. Moreover, we study a special case where the IRS
illuminates the entire blockage area, which implies that the IRS phase shifts
do not change over time leading to zero overhead for IRS reconfiguration.

    

### [[2111.09422] ORPHEUS: Living Labs for End-to-End Data Infrastructures for Digital Agriculture](http://arxiv.org/abs/2111.09422)


  IoT networks are being used to collect, analyze, and utilize sensor data.
There are still some key requirements to leverage IoT networks in digital
agriculture, e.g., design and deployment of energy saving and ruggedized sensor
nodes (SN), reliable and long-range wireless network connectivity, end-to-end
data collection pipelines for batch and streaming data. Thus, we introduce our
living lab ORPHEUS and its design and implementation trajectory to showcase our
orchestrated testbed of IoT sensors, data connectivity, database orchestration,
and visualization dashboard. We deploy light-weight energy saving SNs in the
field to collect data, using LoRa (Long Range wireless) to transmit data from
the SNs to the Gateway node, upload all the data to the database server, and
finally visualize the data. For future exploration, we also built a testbed of
embedded devices using four different variants of NVIDIA Jetson development
modules (Nano, TX2, Xavier NX, AGX Xavier) to benchmark the potential upgrade
choices for SNs in ORPHEUS. Based on our deployment in multiple farms in a
3-county region around Purdue University, and on the Purdue University campus,
we present analyses from our living lab deployment and additional components of
the next-generation IoT farm.

    

### [[2111.09424] Design of a Scalable 4G Portable Network Using Low Cost SDR And Raspberry Pi](http://arxiv.org/abs/2111.09424)


  Of late, Software Defined Radio (SDR) approach has become an effective means
to design high data rate wireless systems for a range of applications. There
are methods with which low cost SDR based 4th generation (4G) or long term
evolution (LTE) systems can be designed. Using low cost Raspberry Pi systems,
the SDR aided 4G systems can be designed for high data rate communication. The
work is related to the design of a 4G wireless system using low cost SDR
solutions and integrated to a programmable controller based on a Raspberry Pi.
Experimental results show that the system is effective in a range of
conditions.

    

### [[2111.09425] Quality-Aware Deep Reinforcement Learning for Streaming in Infrastructure-Assisted Connected Vehicles](http://arxiv.org/abs/2111.09425)


  This paper proposes a deep reinforcement learning-based video streaming
scheme for mobility-aware vehicular networks, e.g., vehicles on the highway. We
consider infrastructure-assisted and mmWave-based scenarios in which the macro
base station (MBS) cannot directly provide the streaming service to vehicles
due to the short range of mmWave beams so that small mmWave base stations
(mBSs) along the road deliver the desired videos to users. For a smoother
streaming service, the MBS proactively pushes video chunks to mBSs. This is
done to support vehicles that are currently covered and/or will be by each mBS.
We formulate the dynamic video delivery scheme that adaptively determines 1)
which content, 2) what quality and 3) how many chunks to be proactively
delivered from the MBS to mBSs using Markov decision process (MDP). Since it is
difficult for the MBS to track all the channel conditions and the network
states have extensive dimensions, we adopt the deep deterministic policy
gradient (DDPG) algorithm for the DRL-based video delivery scheme. This paper
finally shows that the DRL agent learns a streaming policy that pursues high
average quality while limiting packet drops, avoiding playback stalls, reducing
quality fluctuations and saving backhaul usage.

    

### [[2111.09716] Development of NavIC synchronized fully automated inter-building QKD framework and demonstration of quantum secured video calling](http://arxiv.org/abs/2111.09716)


  Quantum key distribution (QKD) is a revolutionary communication technology
that promises ultimate security assurance by exploiting the fundamental
principles of quantum mechanics. In this work, we report design and development
of a fully automated inter-building QKD framework for generation and
distribution of cryptographic keys, securely and seamlessly, by executing weak
coherent pulse based BB84 protocol. This framework is experimentally validated
by establishing a quantum communication link between two buildings separated by
~300m of free-space atmospheric channel. A novel synchronization technique
enabled with indigenous NavIC (IRNSS) constellation is developed and
implemented. This QKD system demonstrates generation of secure key rate as high
as 300 Kbps with QBER< 3% for mean photon no. per pulse (${\mu}$) of 0.15. The
intercept-resend eavesdropping attack has been emulated within the system and
evaluated during experiment. A novel quantum secured end-to-end encrypted video
calling app (QuViC) is also developed and integrated with QKD framework to
demonstrate unconditionally secure two-way communication over Ethernet,
functioning alongside with quantum communication.

    

### [[2105.03503] Photonic Network Coding and Partial Protection for Optical Core Networks: Two for a Tango](http://arxiv.org/abs/2105.03503)


  The digital transformation is creating basically a digital version of our
physical world and the currency in that digital space is data. Massive amount
of data has been generated ranging from wearable devices monitoring our
physical health every single millisecond to autonomous vehicles generating
roughly 5Tb hourly to even astronomical activities producing an order of
Exabytes on daily basis and then ultra-broadband Internet comes into play,
moving such data to the cloud. Internet traffic therefore has been experiencing
explosive growth and in this context, optical transport networks forming the
backbone of the Internet are pushed for transformation in system capacity.
While the intuitive solution of deploying multiple fibers can address the
pressing demand for increased capacity, doing so does not bring improvement in
economic of scales in terms of cost, power consumption and spectral efficiency.
This necessitates for a different approach so that the fiber capacity could be
utilized in a more efficient manner. In this paper, we focus on innovative
techniques, that is, photonic network coding and partial protection, to reduce
the effective traffic load in order to achieve greater capacity efficiency for
optical transport networks. Specifically, the application of network coding is
examined by upgrading the functionalities of intermediate nodes with
all-optical processing (i.e., encoding and decoding) capabilities. Besides,
partial protection relying on the premise of providing just enough bandwidth in
case of failure events is investigated for saving the redundant protection
capacity. That it takes two to tango, combining photonic network coding and
partial protection therefore bring to light new opportunities and challenges.
In mining such new avenue, we present insights on how to derive compounding
gains to maximize spectral efficiency via a case study.

    

### [[2111.09308] Transformation of Node to Knowledge Graph Embeddings for Faster Link Prediction in Social Networks](http://arxiv.org/abs/2111.09308)


  Recent advances in neural networks have solved common graph problems such as
link prediction, node classification, node clustering, node recommendation by
developing embeddings of entities and relations into vector spaces. Graph
embeddings encode the structural information present in a graph. The encoded
embeddings then can be used to predict the missing links in a graph. However,
obtaining the optimal embeddings for a graph can be a computationally
challenging task specially in an embedded system. Two techniques which we focus
on in this work are 1) node embeddings from random walk based methods and 2)
knowledge graph embeddings. Random walk based embeddings are computationally
inexpensive to obtain but are sub-optimal whereas knowledge graph embeddings
perform better but are computationally expensive. In this work, we investigate
a transformation model which converts node embeddings obtained from random walk
based methods to embeddings obtained from knowledge graph methods directly
without an increase in the computational cost. Extensive experimentation shows
that the proposed transformation model can be used for solving link prediction
in real-time.

    

### [[2111.09314] GAETS: A Graph Autoencoder Time Series Approach Towards Battery Parameter Estimation](http://arxiv.org/abs/2111.09314)


  Lithium-ion batteries are powering the ongoing transportation electrification
revolution. Lithium-ion batteries possess higher energy density and favourable
electrochemical properties which make it a preferable energy source for
electric vehicles. Precise estimation of battery parameters (Charge capacity,
voltage etc) is vital to estimate the available range in an electric vehicle.
Graph-based estimation techniques enable us to understand the variable
dependencies underpinning them to improve estimates. In this paper we employ
Graph Neural Networks for battery parameter estimation, we introduce a unique
graph autoencoder time series estimation approach. Variables in battery
measurements are known to have an underlying relationship with each other in a
certain correlation within variables of interest. We use graph autoencoder
based on a non-linear version of NOTEARS as this allowed us to perform
gradient-descent in learning the structure (instead of treating it as a
combinatorial optimisation problem). The proposed architecture outperforms the
state-of-the-art Graph Time Series (GTS) architecture for battery parameter
estimation. We call our method GAETS (Graph AutoEncoder Time Series).

    

### [[2111.09344] The People's Speech: A Large-Scale Diverse English Speech Recognition Dataset for Commercial Usage](http://arxiv.org/abs/2111.09344)


  The People's Speech is a free-to-download 30,000-hour and growing supervised
conversational English speech recognition dataset licensed for academic and
commercial usage under CC-BY-SA (with a CC-BY subset). The data is collected
via searching the Internet for appropriately licensed audio data with existing
transcriptions. We describe our data collection methodology and release our
data collection system under the Apache 2.0 license. We show that a model
trained on this dataset achieves a 9.98% word error rate on Librispeech's
test-clean test set.Finally, we discuss the legal and ethical issues
surrounding the creation of a sizable machine learning corpora and plans for
continued maintenance of the project under MLCommons's sponsorship.

    

### [[2111.09360] Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360)


  Federated learning allows clients to collaboratively learn statistical models
while keeping their data local. Federated learning was originally used to train
a unique global model to be served to all clients, but this approach might be
sub-optimal when clients' local data distributions are heterogeneous. In order
to tackle this limitation, recent personalized federated learning methods train
a separate model for each client while still leveraging the knowledge available
at other clients. In this work, we exploit the ability of deep neural networks
to extract high quality vectorial representations (embeddings) from non-tabular
data, e.g., images and text, to propose a personalization mechanism based on
local memorization. Personalization is obtained interpolating a pre-trained
global model with a $k$-nearest neighbors (kNN) model based on the shared
representation provided by the global model. We provide generalization bounds
for the proposed approach and we show on a suite of federated datasets that
this approach achieves significantly higher accuracy and fairness than
state-of-the-art methods.

    

### [[2111.09372] BLOOM-Net: Blockwise Optimization for Masking Networks Toward Scalable and Efficient Speech Enhancement](http://arxiv.org/abs/2111.09372)


  In this paper, we present a blockwise optimization method for masking-based
networks (BLOOM-Net) for training scalable speech enhancement networks. Here,
we design our network with a residual learning scheme and train the internal
separator blocks sequentially to obtain a scalable masking-based deep neural
network for speech enhancement. Its scalability lets it adjust the run-time
complexity based on the test-time resource constraints: once deployed, the
model can alter its complexity dynamically depending on the test time
environment. To this end, we modularize our models in that they can flexibly
accommodate varying needs for enhancement performance and constraints on the
resources, incurring minimal memory or training overhead due to the added
scalability. Our experiments on speech enhancement demonstrate that the
proposed blockwise optimization method achieves the desired scalability with
only a slight performance degradation compared to corresponding models trained
end-to-end.

    

### [[2111.09378] MPF6D: Masked Pyramid Fusion 6D Pose Estimation](http://arxiv.org/abs/2111.09378)


  Object pose estimation has multiple important applications, such as robotic
grasping and augmented reality. We present a new method to estimate the 6D pose
of objects that improves upon the accuracy of current proposals and can still
be used in real-time. Our method uses RGB-D data as input to segment objects
and estimate their pose. It uses a neural network with multiple heads, one head
estimates the object classification and generates the mask, the second
estimates the values of the translation vector and the last head estimates the
values of the quaternion that represents the rotation of the object. These
heads leverage a pyramid architecture used during feature extraction and
feature fusion. Our method can be used in real-time with its low inference time
of 0.12 seconds and has high accuracy. With this combination of fast inference
and good accuracy it is possible to use our method in robotic pick and place
tasks and/or augmented reality applications.

    

### [[2111.09381] MEDCOD: A Medically-Accurate, Emotive, Diverse, and Controllable Dialog System](http://arxiv.org/abs/2111.09381)


  We present MEDCOD, a Medically-Accurate, Emotive, Diverse, and Controllable
Dialog system with a unique approach to the natural language generator module.
MEDCOD has been developed and evaluated specifically for the history taking
task. It integrates the advantage of a traditional modular approach to
incorporate (medical) domain knowledge with modern deep learning techniques to
generate flexible, human-like natural language expressions. Two key aspects of
MEDCOD's natural language output are described in detail. First, the generated
sentences are emotive and empathetic, similar to how a doctor would communicate
to the patient. Second, the generated sentence structures and phrasings are
varied and diverse while maintaining medical consistency with the desired
medical concept (provided by the dialogue manager module of MEDCOD).
Experimental results demonstrate the effectiveness of our approach in creating
a human-like medical dialogue system. Relevant code is available at
this https URL


### [[2111.09388] Minimum Bayes Risk Decoding with Neural Metrics of Translation Quality](http://arxiv.org/abs/2111.09388)


  This work applies Minimum Bayes Risk (MBR) decoding to optimize diverse
automated metrics of translation quality. Automatic metrics in machine
translation have made tremendous progress recently. In particular, neural
metrics, fine-tuned on human ratings (e.g. BLEURT, or COMET) are outperforming
surface metrics in terms of correlations to human judgements. Our experiments
show that the combination of a neural translation model with a neural
reference-based metric, BLEURT, results in significant improvement in automatic
and human evaluations. This improvement is obtained with translations different
from classical beam-search output: these translations have much lower
likelihood and are less favored by surface metrics like BLEU.

    

### [[2111.09389] Low Precision Decentralized Distributed Training with Heterogeneous Data](http://arxiv.org/abs/2111.09389)


  Decentralized distributed learning is the key to enabling large-scale machine
learning (training) on the edge devices utilizing private user-generated local
data, without relying on the cloud. However, practical realization of such
on-device training is limited by the communication bottleneck, computation
complexity of training deep models and significant data distribution skew
across devices. Many feedback-based compression techniques have been proposed
in the literature to reduce the communication cost and a few works propose
algorithmic changes to aid the performance in the presence of skewed data
distribution by improving convergence rate. To the best of our knowledge, there
is no work in the literature that applies and shows compute efficient training
techniques such quantization, pruning etc., for peer-to-peer decentralized
learning setups. In this paper, we analyze and show the convergence of low
precision decentralized training that aims to reduce the computational
complexity of training and inference. Further, We study the effect of degree of
skew and communication compression on the low precision decentralized training
over various computer vision and Natural Language Processing (NLP) tasks. Our
experiments indicate that 8-bit decentralized training has minimal accuracy
loss compared to its full precision counterpart even with heterogeneous data.
However, when low precision training is accompanied by communication
compression through sparsification we observe 1-2% drop in accuracy. The
proposed low precision decentralized training decreases computational
complexity, memory usage, and communication cost by ~4x while trading off less
than a 1% accuracy for both IID and non-IID data. In particular, with higher
skew values, we observe an increase in accuracy (by ~0.5%) with low precision
training, indicating the regularization effect of the quantization.

    

### [[2111.09395] FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance](http://arxiv.org/abs/2111.09395)


  Deep reinforcement learning (DRL) has been envisioned to have a competitive
edge in quantitative finance. However, there is a steep development curve for
quantitative traders to obtain an agent that automatically positions to win in
the market, namely \textit{to decide where to trade, at what price} and
\textit{what quantity}, due to the error-prone programming and arduous
debugging. In this paper, we present the first open-source framework
\textit{FinRL} as a full pipeline to help quantitative traders overcome the
steep learning curve. FinRL is featured with simplicity, applicability and
extensibility under the key principles, \textit{full-stack framework,
customization, reproducibility} and \textit{hands-on tutoring}.
Embodied as a three-layer architecture with modular structures, FinRL
implements fine-tuned state-of-the-art DRL algorithms and common reward
functions, while alleviating the debugging workloads. Thus, we help users
pipeline the strategy design at a high turnover rate. At multiple levels of
time granularity, FinRL simulates various markets as training environments
using historical data and live trading APIs. Being highly extensible, FinRL
reserves a set of user-import interfaces and incorporates trading constraints
such as market friction, market liquidity and investor's risk-aversion.
Moreover, serving as practitioners' stepping stones, typical trading tasks are
provided as step-by-step tutorials, e.g., stock trading, portfolio allocation,
cryptocurrency trading, etc.

    

### [[2111.09415] Automated PII Extraction from Social Media for Raising Privacy Awareness: A Deep Transfer Learning Approach](http://arxiv.org/abs/2111.09415)


  Internet users have been exposing an increasing amount of Personally
Identifiable Information (PII) on social media. Such exposed PII can cause
severe losses to the users, and informing users of their PII exposure is
crucial to raise their privacy awareness and encourage them to take protective
measures. To this end, advanced automatic techniques are needed. While
Information Extraction (IE) techniques can be used to extract the PII
automatically, Deep Learning (DL)-based IE models alleviate the need for
feature engineering and further improve the efficiency. However, DL-based IE
models often require large-scale labeled data for training, but PII-labeled
social media posts are difficult to obtain due to privacy concerns. Also, these
models rely heavily on pre-trained word embeddings, while PII in social media
often varies in forms and thus has no fixed representations in pre-trained word
embeddings. In this study, we propose the Deep Transfer Learning for PII
Extraction (DTL-PIIE) framework to address these two limitations. DTL-PIIE
transfers knowledge learned from publicly available PII data to social media to
address the problem of rare PII-labeled data. Moreover, our framework leverages
Graph Convolutional Networks (GCNs) to incorporate syntactic patterns to guide
PIIE without relying on pre-trained word embeddings. Evaluation against
benchmark IE models indicates that our approach outperforms state-of-the-art
DL-based IE models. Our framework can facilitate various applications, such as
PII misuse prediction and privacy risk assessment, protecting the privacy of
internet users.

    

### [[2111.09434] On the Effectiveness of Iterative Learning Control](http://arxiv.org/abs/2111.09434)


  Iterative learning control (ILC) is a powerful technique for high performance
tracking in the presence of modeling errors for optimal control applications.
There is extensive prior work showing its empirical effectiveness in
applications such as chemical reactors, industrial robots and quadcopters.
However, there is little prior theoretical work that explains the effectiveness
of ILC even in the presence of large modeling errors, where optimal control
methods using the misspecified model (MM) often perform poorly. Our work
presents such a theoretical study of the performance of both ILC and MM on
Linear Quadratic Regulator (LQR) problems with unknown transition dynamics. We
show that the suboptimality gap, as measured with respect to the optimal LQR
controller, for ILC is lower than that for MM by higher order terms that become
significant in the regime of high modeling errors. A key part of our analysis
is the perturbation bounds for the discrete Ricatti equation in the finite
horizon setting, where the solution is not a fixed point and requires tracking
the error using recursive bounds. We back our theoretical findings with
empirical experiments on a toy linear dynamical system with an approximate
model, a nonlinear inverted pendulum system with misspecified mass, and a
nonlinear planar quadrotor system in the presence of wind. Experiments show
that ILC outperforms MM significantly, in terms of the cost of computed
trajectories, when modeling errors are high.

    

### [[2111.09437] Sustainable Artificial Intelligence through Continual Learning](http://arxiv.org/abs/2111.09437)


  The increasing attention on Artificial Intelligence (AI) regulation has led
to the definition of a set of ethical principles grouped into the Sustainable
AI framework. In this article, we identify Continual Learning, an active area
of AI research, as a promising approach towards the design of systems compliant
with the Sustainable AI principles. While Sustainable AI outlines general
desiderata for ethical applications, Continual Learning provides means to put
such desiderata into practice.

    

### [[2111.09445] FLSys: Toward an Open Ecosystem for FederatedLearning Mobile Apps](http://arxiv.org/abs/2111.09445)


  This paper presents the design, implementation, and evaluation of FLSys, a
mobile-cloud federated learning (FL) system that supports deep learning models
for mobile apps. FLSys is a key component toward creating an open ecosystem of
FL models and apps that use these models. FLSys is designed to work with mobile
sensing data collected on smart phones, balance model performance with resource
consumption on the phones, tolerate phone communication failures, and achieve
scalability in the cloud. In FLSys, different DL models with different FL
aggregation methods in the cloud can be trained and accessed concurrently by
different apps. Furthermore, FLSys provides a common API for third-party app
developers to train FL models. FLSys is implemented in Android and AWS cloud.
We co-designed FLSys with a human activity recognition (HAR) in the wild FL
model. HAR sensing data was collected in two areas from the phones of 100+
college students during a five-month period. We implemented HAR-Wild, a CNN
model tailored to mobile devices, with a data augmentation mechanism to
mitigate the problem of non-Independent and Identically Distributed (non-IID)
data that affects FL model training in the wild. A sentiment analysis (SA)
model is used to demonstrate how FLSys effectively supports concurrent models,
and it uses a dataset with 46,000+ tweets from 436 users. We conducted
extensive experiments on Android phones and emulators showing that FLSys
achieves good model utility and practical system performance.

    

### [[2111.09446] L4-Norm Weight Adjustments for Converted Spiking Neural Networks](http://arxiv.org/abs/2111.09446)


  Spiking Neural Networks (SNNs) are being explored for their potential energy
efficiency benefits due to sparse, event-driven computation. Non-spiking
artificial neural networks are typically trained with stochastic gradient
descent using backpropagation. The calculation of true gradients for
backpropagation in spiking neural networks is impeded by the non-differentiable
firing events of spiking neurons. On the other hand, using approximate
gradients is effective, but computationally expensive over many time steps. One
common technique, then, for training a spiking neural network is to train a
topologically-equivalent non-spiking network, and then convert it to an spiking
network, replacing real-valued inputs with proportionally rate-encoded Poisson
spike trains. Converted SNNs function sufficiently well because the mean
pre-firing membrane potential of a spiking neuron is proportional to the dot
product of the input rate vector and the neuron weight vector, similar to the
functionality of a non-spiking network. However, this conversion only considers
the mean and not the temporal variance of the membrane potential. As the
standard deviation of the pre-firing membrane potential is proportional to the
L4-norm of the neuron weight vector, we propose a weight adjustment based on
the L4-norm during the conversion process in order to improve classification
accuracy of the converted network.

    

### [[2111.09463] Self-Attending Task Generative Adversarial Network for Realistic Satellite Image Creation](http://arxiv.org/abs/2111.09463)


  We introduce a self-attending task generative adversarial network (SATGAN)
and apply it to the problem of augmenting synthetic high contrast scientific
imagery of resident space objects with realistic noise patterns and sensor
characteristics learned from collected data. Augmenting these synthetic data is
challenging due to the highly localized nature of semantic content in the data
that must be preserved. Real collected images are used to train a network what
a given class of sensor's images should look like. The trained network then
acts as a filter on noiseless context images and outputs realistic-looking
fakes with semantic content unaltered. The architecture is inspired by
conditional GANs but is modified to include a task network that preserves
semantic information through augmentation. Additionally, the architecture is
shown to reduce instances of hallucinatory objects or obfuscation of semantic
content in context images representing space observation scenes.

    

### [[2111.09467] Contrastive Multiview Coding for Enzyme-Substrate Interaction Prediction](http://arxiv.org/abs/2111.09467)


  Characterizing Enzyme function is an important requirement for predicting
Enzyme-Substrate interactions. In this paper, we present a novel approach of
applying Contrastive Multiview Coding to this problem to improve the
performance of prediction. We present a method to leverage auxiliary data from
an Enzymatic database like KEGG to learn the mutual information present in
multiple views of enzyme-substrate reactions. We show that congruency in the
multiple views of the reaction data can be used to improve prediction
performance.

    

### [[2111.09487] A Novel Optimized Asynchronous Federated Learning Framework](http://arxiv.org/abs/2111.09487)


  Federated Learning (FL) since proposed has been applied in many fields, such
as credit assessment, medical, etc. Because of the difference in the network or
computing resource, the clients may not update their gradients at the same time
that may take a lot of time to wait or idle. That's why Asynchronous Federated
Learning (AFL) method is needed. The main bottleneck in AFL is communication.
How to find a balance between the model performance and the communication cost
is a challenge in AFL. This paper proposed a novel AFL framework VAFL. And we
verified the performance of the algorithm through sufficient experiments. The
experiments show that VAFL can reduce the communication times about 51.02\%
with 48.23\% average communication compression rate and allow the model to be
converged faster. The code is available at
\url{this https URL}

    

### [[2111.09488] Attacking Deep Learning AI Hardware with Universal Adversarial Perturbation](http://arxiv.org/abs/2111.09488)


  Universal Adversarial Perturbations are image-agnostic and model-independent
noise that when added with any image can mislead the trained Deep Convolutional
Neural Networks into the wrong prediction. Since these Universal Adversarial
Perturbations can seriously jeopardize the security and integrity of practical
Deep Learning applications, existing techniques use additional neural networks
to detect the existence of these noises at the input image source. In this
paper, we demonstrate an attack strategy that when activated by rogue means
(e.g., malware, trojan) can bypass these existing countermeasures by augmenting
the adversarial noise at the AI hardware accelerator stage. We demonstrate the
accelerator-level universal adversarial noise attack on several deep Learning
models using co-simulation of the software kernel of Conv2D function and the
Verilog RTL model of the hardware under the FuseSoC environment.

    

### [[2111.09489] Data-driven discovery of Bäcklund transforms and soliton evolution equations via deep neural network learning schemes](http://arxiv.org/abs/2111.09489)


  We introduce a deep neural network learning scheme to learn the Bäcklund
transforms (BTs) of soliton evolution equations and an enhanced deep learning
scheme for data-driven soliton equation discovery based on the known BTs,
respectively. The first scheme takes advantage of some solution (or soliton
equation) information to study the data-driven BT of sine-Gordon equation, and
complex and real Miura transforms between the defocusing (focusing) mKdV
equation and KdV equation, as well as the data-driven mKdV equation discovery
via the Miura transforms. The second deep learning scheme uses the
explicit/implicit BTs generating the higher-order solitons to train the
data-driven discovery of mKdV and sine-Gordon equations, in which the
high-order solution informations are more powerful for the enhanced leaning
soliton equations with higher accurates.

    

### [[2111.09499] Dynamically pruning segformer for efficient semantic segmentation](http://arxiv.org/abs/2111.09499)


  As one of the successful Transformer-based models in computer vision tasks,
SegFormer demonstrates superior performance in semantic segmentation.
Nevertheless, the high computational cost greatly challenges the deployment of
SegFormer on edge devices. In this paper, we seek to design a lightweight
SegFormer for efficient semantic segmentation. Based on the observation that
neurons in SegFormer layers exhibit large variances across different images, we
propose a dynamic gated linear layer, which prunes the most uninformative set
of neurons based on the input instance. To improve the dynamically pruned
SegFormer, we also introduce two-stage knowledge distillation to transfer the
knowledge within the original teacher to the pruned student network.
Experimental results show that our method can significantly reduce the
computation overhead of SegFormer without an apparent performance drop. For
instance, we can achieve 36.9% mIoU with only 3.3G FLOPs on ADE20K, saving more
than 60% computation with the drop of only 0.5% in mIoU

    

### [[2111.09502] Docking-based Virtual Screening with Multi-Task Learning](http://arxiv.org/abs/2111.09502)


  Machine learning shows great potential in virtual screening for drug
discovery. Current efforts on accelerating docking-based virtual screening do
not consider using existing data of other previously developed targets. To make
use of the knowledge of the other targets and take advantage of the existing
data, in this work, we apply multi-task learning to the problem of
docking-based virtual screening. With two large docking datasets, the results
of extensive experiments show that multi-task learning can achieve better
performances on docking score prediction. By learning knowledge across multiple
targets, the model trained by multi-task learning shows a better ability to
adapt to a new target. Additional empirical study shows that other problems in
drug discovery, such as the experimental drug-target affinity prediction, may
also benefit from multi-task learning. Our results demonstrate that multi-task
learning is a promising machine learning approach for docking-based virtual
screening and accelerating the process of drug discovery.

    

### [[2111.09507] Assessing Social Determinants-Related Performance Bias of Machine Learning Models: A case of Hyperchloremia Prediction in ICU Population](http://arxiv.org/abs/2111.09507)


  Machine learning in medicine leverages the wealth of healthcare data to
extract knowledge, facilitate clinical decision-making, and ultimately improve
care delivery. However, ML models trained on datasets that lack demographic
diversity could yield suboptimal performance when applied to the
underrepresented populations (e.g. ethnic minorities, lower social-economic
status), thus perpetuating health disparity. In this study, we evaluated four
classifiers built to predict Hyperchloremia - a condition that often results
from aggressive fluids administration in the ICU population - and compared
their performance in racial, gender, and insurance subgroups. We observed that
adding social determinants features in addition to the lab-based ones improved
model performance on all patients. The subgroup testing yielded significantly
different AUC scores in 40 out of the 44 model-subgroup, suggesting disparities
when applying ML models to social determinants subgroups. We urge future
researchers to design models that proactively adjust for potential biases and
include subgroup reporting in their studies.

    

### [[2111.09533] DeepGuard: A Framework for Safeguarding Autonomous Driving Systems from Inconsistent Behavior](http://arxiv.org/abs/2111.09533)


  The deep neural networks (DNNs)based autonomous driving systems (ADSs) are
expected to reduce road accidents and improve safety in the transportation
domain as it removes the factor of human error from driving tasks. The DNN
based ADS sometimes may exhibit erroneous or unexpected behaviors due to
unexpected driving conditions which may cause accidents. It is not possible to
generalize the DNN model performance for all driving conditions. Therefore, the
driving conditions that were not considered during the training of the ADS may
lead to unpredictable consequences for the safety of autonomous vehicles. This
study proposes an autoencoder and time series analysis based anomaly detection
system to prevent the safety critical inconsistent behavior of autonomous
vehicles at runtime. Our approach called DeepGuard consists of two components.
The first component, the inconsistent behavior predictor, is based on an
autoencoder and time series analysis to reconstruct the driving scenarios.
Based on reconstruction error and threshold it determines the normal and
unexpected driving scenarios and predicts potential inconsistent behavior. The
second component provides on the fly safety guards, that is, it automatically
activates healing strategies to prevent inconsistencies in the behavior. We
evaluated the performance of DeepGuard in predicting the injected anomalous
driving scenarios using already available open sourced DNN based ADSs in the
Udacity simulator. Our simulation results show that the best variant of
DeepGuard can predict up to 93 percent on the CHAUFFEUR ADS, 83 percent on
DAVE2 ADS, and 80 percent of inconsistent behavior on the EPOCH ADS model,
outperforming SELFORACLE and DeepRoad. Overall, DeepGuard can prevent up to 89
percent of all predicted inconsistent behaviors of ADS by executing predefined
safety guards.

    

### [[2111.09537] The Prominence of Artificial Intelligence in COVID-19](http://arxiv.org/abs/2111.09537)


  In December 2019, a novel virus called COVID-19 had caused an enormous number
of causalities to date. The battle with the novel Coronavirus is baffling and
horrifying after the Spanish Flu 2019. While the front-line doctors and medical
researchers have made significant progress in controlling the spread of the
highly contiguous virus, technology has also proved its significance in the
battle. Moreover, Artificial Intelligence has been adopted in many medical
applications to diagnose many diseases, even baffling experienced doctors.
Therefore, this survey paper explores the methodologies proposed that can aid
doctors and researchers in early and inexpensive methods of diagnosis of the
disease. Most developing countries have difficulties carrying out tests using
the conventional manner, but a significant way can be adopted with Machine and
Deep Learning. On the other hand, the access to different types of medical
images has motivated the researchers. As a result, a mammoth number of
techniques are proposed. This paper first details the background knowledge of
the conventional methods in the Artificial Intelligence domain. Following that,
we gather the commonly used datasets and their use cases to date. In addition,
we also show the percentage of researchers adopting Machine Learning over Deep
Learning. Thus we provide a thorough analysis of this scenario. Lastly, in the
research challenges, we elaborate on the problems faced in COVID-19 research,
and we address the issues with our understanding to build a bright and healthy
environment.

    

### [[2111.09543] DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](http://arxiv.org/abs/2111.09543)


  This paper presents a new pre-trained language model, DeBERTaV3, which
improves the original DeBERTa model by replacing mask language modeling (MLM)
with replaced token detection (RTD), a more sample-efficient pre-training task.
Our analysis shows that vanilla embedding sharing in ELECTRA hurts training
efficiency and model performance. This is because the training losses of the
discriminator and the generator pull token embeddings in different directions,
creating the "tug-of-war" dynamics. We thus propose a new gradient-disentangled
embedding sharing method that avoids the tug-of-war dynamics, improving both
training efficiency and the quality of the pre-trained model. We have
pre-trained DeBERTaV3 using the same settings as DeBERTa to demonstrate its
exceptional performance on a wide range of downstream natural language
understanding (NLU) tasks. Taking the GLUE benchmark with eight tasks as an
example, the DeBERTaV3 Large model achieves a 91.37% average score, which is
1.37% over DeBERTa and 1.91% over ELECTRA, setting a new state-of-the-art
(SOTA) among the models with a similar structure. Furthermore, we have
pre-trained a multi-lingual model mDeBERTa and observed a larger improvement
over strong baselines compared to English models. For example, the mDeBERTa
Base achieves a 79.8% zero-shot cross-lingual accuracy on XNLI and a 3.6%
improvement over XLM-R Base, creating a new SOTA on this benchmark. We have
made our pre-trained models and inference code publicly available at
this https URL.

    

### [[2111.09544] C-OPH: Improving the Accuracy of One Permutation Hashing (OPH) with Circulant Permutations](http://arxiv.org/abs/2111.09544)


  Minwise hashing (MinHash) is a classical method for efficiently estimating
the Jaccrad similarity in massive binary (0/1) data. To generate $K$ hash
values for each data vector, the standard theory of MinHash requires $K$
independent permutations. Interestingly, the recent work on "circulant MinHash"
(C-MinHash) has shown that merely two permutations are needed. The first
permutation breaks the structure of the data and the second permutation is
re-used $K$ time in a circulant manner. Surprisingly, the estimation accuracy
of C-MinHash is proved to be strictly smaller than that of the original
MinHash. The more recent work further demonstrates that practically only one
permutation is needed. Note that C-MinHash is different from the well-known
work on "One Permutation Hashing (OPH)" published in NIPS'12. OPH and its
variants using different "densification" schemes are popular alternatives to
the standard MinHash. The densification step is necessary in order to deal with
empty bins which exist in One Permutation Hashing.
In this paper, we propose to incorporate the essential ideas of C-MinHash to
improve the accuracy of One Permutation Hashing. Basically, we develop a new
densification method for OPH, which achieves the smallest estimation variance
compared to all existing densification schemes for OPH. Our proposed method is
named C-OPH (Circulant OPH). After the initial permutation (which breaks the
existing structure of the data), C-OPH only needs a "shorter" permutation of
length $D/K$ (instead of $D$), where $D$ is the original data dimension and $K$
is the total number of bins in OPH. This short permutation is re-used in $K$
bins in a circulant shifting manner. It can be shown that the estimation
variance of the Jaccard similarity is strictly smaller than that of the
existing (densified) OPH methods.

    

### [[2111.09564] LAnoBERT : System Log Anomaly Detection based on BERT Masked Language Model](http://arxiv.org/abs/2111.09564)


  The system log generated in a computer system refers to large-scale data that
are collected simultaneously and used as the basic data for determining simple
errors and detecting external adversarial intrusion or the abnormal behaviors
of insiders. The aim of system log anomaly detection is to promptly identify
anomalies while minimizing human intervention, which is a critical problem in
the industry. Previous studies performed anomaly detection through algorithms
after converting various forms of log data into a standardized template using a
parser. These methods involved generating a template for refining the log key.
Particularly, a template corresponding to a specific event should be defined in
advance for all the log data using which the information within the log key may
get this http URL this study, we propose LAnoBERT, a parser free system log anomaly
detection method that uses the BERT model, exhibiting excellent natural
language processing performance. The proposed method, LAnoBERT, learns the
model through masked language modeling, which is a BERT-based pre-training
method, and proceeds with unsupervised learning-based anomaly detection using
the masked language modeling loss function per log key word during the
inference process. LAnoBERT achieved better performance compared to previous
methodology in an experiment conducted using benchmark log datasets, HDFS, and
BGL, and also compared to certain supervised learning-based models.

    

### [[2111.09612] How Emotionally Stable is ALBERT? Testing Robustness with Stochastic Weight Averaging on a Sentiment Analysis Task](http://arxiv.org/abs/2111.09612)


  Despite their success, modern language models are fragile. Even small changes
in their training pipeline can lead to unexpected results. We study this
phenomenon by examining the robustness of ALBERT (arXiv:1909.11942) in
combination with Stochastic Weight Averaging (SWA) (arXiv:1803.05407) -- a
cheap way of ensembling -- on a sentiment analysis task (SST-2). In particular,
we analyze SWA's stability via CheckList criteria (arXiv:2005.04118), examining
the agreement on errors made by models differing only in their random seed. We
hypothesize that SWA is more stable because it ensembles model snapshots taken
along the gradient descent trajectory. We quantify stability by comparing the
models' mistakes with Fleiss' Kappa (Fleiss, 1971) and overlap ratio scores. We
find that SWA reduces error rates in general; yet the models still suffer from
their own distinct biases (according to CheckList).

    

### [[2111.09613] Improving Transferability of Representations via Augmentation-Aware Self-Supervision](http://arxiv.org/abs/2111.09613)


  Recent unsupervised representation learning methods have shown to be
effective in a range of vision tasks by learning representations invariant to
data augmentations such as random cropping and color jittering. However, such
invariance could be harmful to downstream tasks if they rely on the
characteristics of the data augmentations, e.g., location- or color-sensitive.
This is not an issue just for unsupervised learning; we found that this occurs
even in supervised learning because it also learns to predict the same label
for all augmented samples of an instance. To avoid such failures and obtain
more generalizable representations, we suggest to optimize an auxiliary
self-supervised loss, coined AugSelf, that learns the difference of
augmentation parameters (e.g., cropping positions, color adjustment
intensities) between two randomly augmented samples. Our intuition is that
AugSelf encourages to preserve augmentation-aware information in learned
representations, which could be beneficial for their transferability.
Furthermore, AugSelf can easily be incorporated into recent state-of-the-art
representation learning methods with a negligible additional training cost.
Extensive experiments demonstrate that our simple idea consistently improves
the transferability of representations learned by supervised and unsupervised
methods in various transfer learning scenarios. The code is available at
this https URL.

    

### [[2111.09637] A Modular 1D-CNN Architecture for Real-time Digital Pre-distortion](http://arxiv.org/abs/2111.09637)


  This study reports a novel hardware-friendly modular architecture for
implementing one dimensional convolutional neural network (1D-CNN) digital
predistortion (DPD) technique to linearize RF power amplifier (PA)
real-time.The modular nature of our design enables DPD system adaptation for
variable resource and timing constraints.Our work also presents a co-simulation
architecture to verify the DPD performance with an actual power amplifier
hardware-in-the-loop.The experimental results with 100 MHz signals show that
the proposed 1D-CNN obtains superior performance compared with other neural
network architectures for real-time DPD application.

    

### [[2111.09642] Towards Intelligibility-Oriented Audio-Visual Speech Enhancement](http://arxiv.org/abs/2111.09642)


  Existing deep learning (DL) based speech enhancement approaches are generally
optimised to minimise the distance between clean and enhanced speech features.
These often result in improved speech quality however they suffer from a lack
of generalisation and may not deliver the required speech intelligibility in
real noisy situations. In an attempt to address these challenges, researchers
have explored intelligibility-oriented (I-O) loss functions and integration of
audio-visual (AV) information for more robust speech enhancement (SE). In this
paper, we introduce DL based I-O SE algorithms exploiting AV information, which
is a novel and previously unexplored research direction. Specifically, we
present a fully convolutional AV SE model that uses a modified short-time
objective intelligibility (STOI) metric as a training cost function. To the
best of our knowledge, this is the first work that exploits the integration of
AV modalities with an I-O based loss function for SE. Comparative experimental
results demonstrate that our proposed I-O AV SE framework outperforms
audio-only (AO) and AV models trained with conventional distance-based loss
functions, in terms of standard objective evaluation measures when dealing with
unseen speakers and noises.

    

### [[2111.09645] Dynamic-TinyBERT: Boost TinyBERT's Inference Efficiency by Dynamic Sequence Length](http://arxiv.org/abs/2111.09645)


  Limited computational budgets often prevent transformers from being used in
production and from having their high accuracy utilized. TinyBERT addresses the
computational efficiency by self-distilling BERT into a smaller transformer
representation having fewer layers and smaller internal embedding. However,
TinyBERT's performance drops when we reduce the number of layers by 50%, and
drops even more abruptly when we reduce the number of layers by 75% for
advanced NLP tasks such as span question answering. Additionally, a separate
model must be trained for each inference scenario with its distinct
computational budget. In this work we present Dynamic-TinyBERT, a TinyBERT
model that utilizes sequence-length reduction and Hyperparameter Optimization
for enhanced inference efficiency per any computational budget.
Dynamic-TinyBERT is trained only once, performing on-par with BERT and
achieving an accuracy-speedup trade-off superior to any other efficient
approaches (up to 3.3x with <1% loss-drop). Upon publication, the code to
reproduce our work will be open-sourced.

    

### [[2111.09656] CLMB: deep contrastive learning for robust metagenomic binning](http://arxiv.org/abs/2111.09656)


  The reconstruction of microbial genomes from large metagenomic datasets is a
critical procedure for finding uncultivated microbial populations and defining
their microbial functional roles. To achieve that, we need to perform
metagenomic binning, clustering the assembled contigs into draft genomes.
Despite the existing computational tools, most of them neglect one important
property of the metagenomic data, that is, the noise. To further improve the
metagenomic binning step and reconstruct better metagenomes, we propose a deep
Contrastive Learning framework for Metagenome Binning (CLMB), which can
efficiently eliminate the disturbance of noise and produce more stable and
robust results. Essentially, instead of denoising the data explicitly, we add
simulated noise to the training data and force the deep learning model to
produce similar and stable representations for both the noise-free data and the
distorted data. Consequently, the trained model will be robust to noise and
handle it implicitly during usage. CLMB outperforms the previous
state-of-the-art binning methods significantly, recovering the most
near-complete genomes on almost all the benchmarking datasets (up to 17\% more
reconstructed genomes compared to the second-best method). It also improves the
performance of bin refinement, reconstructing 8-22 more high-quality genomes
and 15-32 more middle-quality genomes than the second-best result.
Impressively, in addition to being compatible with the binning refiner, single
CLMB even recovers on average 15 more HQ genomes than the refiner of VAMB and
Maxbin on the benchmarking datasets. CLMB is open-source and available at
this https URL.

    

### [[2111.09666] CCSL: A Causal Structure Learning Method from Multiple Unknown Environments](http://arxiv.org/abs/2111.09666)


  Most existing causal structure learning methods require data to be
independent and identically distributed (i.i.d.), which often cannot be
guaranteed when the data come from different environments. Some previous
efforts try to tackle this problem in two independent stages, i.e., first
discovering i.i.d. clusters from non-i.i.d. samples, then learning the causal
structures from different groups. This straightforward solution ignores the
intrinsic connections between the two stages, that is both the clustering stage
and the learning stage should be guided by the same causal mechanism. Towards
this end, we propose a unified Causal Cluster Structures Learning (named CCSL)
method for causal discovery from non-i.i.d. data. This method simultaneously
integrates the following two tasks: 1) clustering subjects with the same causal
mechanism; 2) learning causal structures from the samples of subjects.
Specifically, for the former, we provide a Causality-related Chinese Restaurant
Process to cluster samples based on the similarity of the causal structure; for
the latter, we introduce a variational-inference-based approach to learn the
causal structures. Theoretical results provide identification of the causal
model and the clustering model under the linear non-Gaussian assumption.
Experimental results on both simulated and real-world data further validate the
correctness and effectiveness of the proposed method.

    

### [[2111.09679] Enhanced Membership Inference Attacks against Machine Learning Models](http://arxiv.org/abs/2111.09679)


  How much does a given trained model leak about each individual data record in
its training set? Membership inference attacks are used as an auditing tool to
quantify the private information that a model leaks about the individual data
points in its training set. Membership inference attacks are influenced by
different uncertainties that an attacker has to resolve about training data,
the training algorithm, and the underlying data distribution. Thus attack
success rates, of many attacks in the literature, do not precisely capture the
information leakage of models about their data, as they also reflect other
uncertainties that the attack algorithm has. In this paper, we explain the
implicit assumptions and also the simplifications made in prior work using the
framework of hypothesis testing. We also derive new attack algorithms from the
framework that can achieve a high AUC score while also highlighting the
different factors that affect their performance. Our algorithms capture a very
precise approximation of privacy loss in models, and can be used as a tool to
perform an accurate and informed estimation of privacy risk in machine learning
models. We provide a thorough empirical evaluation of our attack strategies on
various machine learning tasks and benchmark datasets.

    

### [[2111.09695] Features selection in NBA outcome prediction through Deep Learning](http://arxiv.org/abs/2111.09695)


  This manuscript is focused on features' definition for the outcome prediction
of matches of NBA basketball championship. It is shown how models based on one
a single feature (Elo rating or the relative victory frequency) have a quality
of fit better than models using box-score predictors (e.g. the Four Factors).
Features have been ex ante calculated for a dataset containing data of 16 NBA
regular seasons, paying particular attention to home court factor. Models have
been produced via Deep Learning, using cross validation.

    

### [[2111.09705] Learning Free-Surface Flow with Physics-Informed Neural Networks](http://arxiv.org/abs/2111.09705)


  The interface between data-driven learning methods and classical simulation
poses an interesting field offering a multitude of new applications. In this
work, we build on the notion of physics-informed neural networks (PINNs) and
employ them in the area of shallow-water equation (SWE) models. These models
play an important role in modeling and simulating free-surface flow scenarios
such as in flood-wave propagation or tsunami waves. Different formulations of
the PINN residual are compared to each other and multiple optimizations are
being evaluated to speed up the convergence rate. We test these with different
1-D and 2-D experiments and finally demonstrate that regarding a SWE scenario
with varying bathymetry, the method is able to produce competitive results in
comparison to the direct numerical simulation with a total relative $L_2$ error
of $8.9e-3$.

    

### [[2111.09708] A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration](http://arxiv.org/abs/2111.09708)


  Hyperspectral imaging offers new perspectives for diverse applications,
ranging from the monitoring of the environment using airborne or satellite
remote sensing, precision farming, food safety, planetary exploration, or
astrophysics. Unfortunately, the spectral diversity of information comes at the
expense of various sources of degradation, and the lack of accurate
ground-truth "clean" hyperspectral signals acquired on the spot makes
restoration tasks challenging. In particular, training deep neural networks for
restoration is difficult, in contrast to traditional RGB imaging problems where
deep models tend to shine. In this paper, we advocate instead for a hybrid
approach based on sparse coding principles that retains the interpretability of
classical techniques encoding domain knowledge with handcrafted image priors,
while allowing to train model parameters end-to-end without massive amounts of
data. We show on various denoising benchmarks that our method is
computationally efficient and significantly outperforms the state of the art.

    

### [[2111.09714] You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](http://arxiv.org/abs/2111.09714)


  Transformer-based models are widely used in natural language processing
(NLP). Central to the transformer model is the self-attention mechanism, which
captures the interactions of token pairs in the input sequences and depends
quadratically on the sequence length. Training such models on longer sequences
is expensive. In this paper, we show that a Bernoulli sampling attention
mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic
complexity of such models to linear. We bypass the quadratic cost by
considering self-attention as a sum of individual tokens associated with
Bernoulli random variables that can, in principle, be sampled at once by a
single hash (although in practice, this number may be a small constant). This
leads to an efficient sampling scheme to estimate self-attention which relies
on specific modifications of LSH (to enable deployment on GPU architectures).
We evaluate our algorithm on the GLUE benchmark with standard 512 sequence
length where we see favorable performance relative to a standard pretrained
Transformer. On the Long Range Arena (LRA) benchmark, for evaluating
performance on long sequences, our method achieves results consistent with
softmax self-attention but with sizable speed-ups and memory savings and often
outperforms other efficient self-attention methods. Our code is available at
this https URL


### [[2111.09724] From Optimality to Robustness: Dirichlet Sampling Strategies in Stochastic Bandits](http://arxiv.org/abs/2111.09724)


  The stochastic multi-arm bandit problem has been extensively studied under
standard assumptions on the arm's distribution (e.g bounded with known support,
exponential family, etc). These assumptions are suitable for many real-world
problems but sometimes they require knowledge (on tails for instance) that may
not be precisely accessible to the practitioner, raising the question of the
robustness of bandit algorithms to model misspecification. In this paper we
study a generic Dirichlet Sampling (DS) algorithm, based on pairwise
comparisons of empirical indices computed with re-sampling of the arms'
observations and a data-dependent exploration bonus. We show that different
variants of this strategy achieve provably optimal regret guarantees when the
distributions are bounded and logarithmic regret for semi-bounded distributions
with a mild quantile condition. We also show that a simple tuning achieve
robustness with respect to a large class of unbounded distributions, at the
cost of slightly worse than logarithmic asymptotic regret. We finally provide
numerical experiments showing the merits of DS in a decision-making problem on
synthetic agriculture data.

    

### [[2111.09741] Patent Sentiment Analysis to Highlight Patent Paragraphs](http://arxiv.org/abs/2111.09741)


  Given a patent document, identifying distinct semantic annotations is an
interesting research aspect. Text annotation helps the patent practitioners
such as examiners and patent attorneys to quickly identify the key arguments of
any invention, successively providing a timely marking of a patent text. In the
process of manual patent analysis, to attain better readability, recognising
the semantic information by marking paragraphs is in practice. This semantic
annotation process is laborious and time-consuming. To alleviate such a
problem, we proposed a novel dataset to train Machine Learning algorithms to
automate the highlighting process. The contributions of this work are: i) we
developed a multi-class, novel dataset of size 150k samples by traversing USPTO
patents over a decade, ii) articulated statistics and distributions of data
using imperative exploratory data analysis, iii) baseline Machine Learning
models are developed to utilize the dataset to address patent paragraph
highlighting task, iv) dataset and codes relating to this task are open-sourced
through a dedicated GIT web page:
this https URL and v) future path to
extend this work using Deep Learning and domain specific pre-trained language
models to develop a tool to highlight is provided. This work assist patent
practitioners in highlighting semantic information automatically and aid to
create a sustainable and efficient patent analysis using the aptitude of
Machine Learning.

    

### [[2111.09744] Covered Information Disentanglement: Model Transparency via Unbiased Permutation Importance](http://arxiv.org/abs/2111.09744)


  Model transparency is a prerequisite in many domains and an increasingly
popular area in machine learning research. In the medical domain, for instance,
unveiling the mechanisms behind a disease often has higher priority than the
diagnostic itself since it might dictate or guide potential treatments and
research directions. One of the most popular approaches to explain model global
predictions is the permutation importance where the performance on permuted
data is benchmarked against the baseline. However, this method and other
related approaches will undervalue the importance of a feature in the presence
of covariates since these cover part of its provided information. To address
this issue, we propose Covered Information Disentanglement (CID), a method that
considers all feature information overlap to correct the values provided by
permutation importance. We further show how to compute CID efficiently when
coupled with Markov random fields. We demonstrate its efficacy in adjusting
permutation importance first on a controlled toy dataset and discuss its effect
on real-world medical data.

    

### [[2111.09758] CSI Clustering with Variational Autoencoding](http://arxiv.org/abs/2111.09758)


  The model order of a wireless channel plays an important role for a variety
of applications in communications engineering, e.g., it represents the number
of resolvable incident wavefronts with non-negligible power incident from a
transmitter to a receiver. Areas such as direction of arrival estimation
leverage the model order to analyze the multipath components of channel state
information. In this work, we propose to use a variational autoencoder to group
unlabeled channel state information with respect to the model order in the
variational autoencoder latent space in an unsupervised manner. We validate our
approach with simulated 3GPP channel data. Our results suggest that, in order
to learn an appropriate clustering, it is crucial to use a more flexible
likelihood model for the variational autoencoder decoder than it is usually the
case in standard applications.

    

### [[2111.09768] Complex Terrain Navigation via Model Error Prediction](http://arxiv.org/abs/2111.09768)


  Robot navigation traditionally relies on building an explicit map that is
used to plan collision-free trajectories to a desired target. In deformable,
complex terrain, using geometric-based approaches can fail to find a path due
to mischaracterizing deformable objects as rigid and impassable. Instead, we
learn to predict an estimate of traversability of terrain regions and to prefer
regions that are easier to navigate (e.g., short grass over small shrubs).
Rather than predicting collisions, we instead regress on realized error
compared to a canonical dynamics model. We train with an on-policy approach,
resulting in successful navigation policies using as little as 50 minutes of
training data split across simulation and real world. Our learning-based
navigation system is a sample efficient short-term planner that we demonstrate
on a Clearpath Husky navigating through a variety of terrain including
grassland and forest

    

### [[2111.09779] Wiggling Weights to Improve the Robustness of Classifiers](http://arxiv.org/abs/2111.09779)


  Robustness against unwanted perturbations is an important aspect of deploying
neural network classifiers in the real world. Common natural perturbations
include noise, saturation, occlusion, viewpoint changes, and blur deformations.
All of them can be modelled by the newly proposed transform-augmented
convolutional networks. While many approaches for robustness train the network
by providing augmented data to the network, we aim to integrate perturbations
in the network architecture to achieve improved and more general robustness. To
demonstrate that wiggling the weights consistently improves classification, we
choose a standard network and modify it to a transform-augmented network. On
perturbed CIFAR-10 images, the modified network delivers a better performance
than the original network. For the much smaller STL-10 dataset, in addition to
delivering better general robustness, wiggling even improves the classification
of unperturbed, clean images substantially. We conclude that wiggled
transform-augmented networks acquire good robustness even for perturbations not
seen during training.

    

### [[2111.09785] DIVA: Dataset Derivative of a Learning Task](http://arxiv.org/abs/2111.09785)


  We present a method to compute the derivative of a learning task with respect
to a dataset. A learning task is a function from a training set to the
validation error, which can be represented by a trained deep neural network
(DNN). The "dataset derivative" is a linear operator, computed around the
trained model, that informs how perturbations of the weight of each training
sample affect the validation error, usually computed on a separate validation
dataset. Our method, DIVA (Differentiable Validation) hinges on a closed-form
differentiable expression of the leave-one-out cross-validation error around a
pre-trained DNN. Such expression constitutes the dataset derivative. DIVA could
be used for dataset auto-curation, for example removing samples with faulty
annotations, augmenting a dataset with additional relevant samples, or
rebalancing. More generally, DIVA can be used to optimize the dataset, along
with the parameters of the model, as part of the training process without the
need for a separate validation dataset, unlike bi-level optimization methods
customary in AutoML. To illustrate the flexibility of DIVA, we report
experiments on sample auto-curation tasks such as outlier rejection, dataset
extension, and automatic aggregation of multi-modal data.

    

### [[2111.09790] MCCE: Monte Carlo sampling of realistic counterfactual explanations](http://arxiv.org/abs/2111.09790)


  In this paper we introduce MCCE: Monte Carlo sampling of realistic
Counterfactual Explanations, a model-based method that generates counterfactual
explanations by producing a set of feasible examples using conditional
inference trees. Unlike algorithmic-based counterfactual methods that have to
solve complex optimization problems or other model based methods that model the
data distribution using heavy machine learning models, MCCE is made up of only
two light-weight steps (generation and post-processing). MCCE is also
straightforward for the end user to understand and implement, handles any type
of predictive model and type of feature, takes into account actionability
constraints when generating the counterfactual explanations, and generates as
many counterfactual explanations as needed. In this paper we introduce MCCE and
give a comprehensive list of performance metrics that can be used to compare
counterfactual explanations. We also compare MCCE with a range of
state-of-the-art methods and a new baseline method on benchmark data sets. MCCE
outperforms all model-based methods and most algorithmic-based methods when
also taking into account validity (i.e., a correctly changed prediction) and
actionability constraints. Finally, we show that MCCE has the strength of
performing almost as well when given just a small subset of the training data.

    

### [[2111.09791] Supporting Undotted Arabic with Pre-trained Language Models](http://arxiv.org/abs/2111.09791)


  We observe a recent behaviour on social media, in which users intentionally
remove consonantal dots from Arabic letters, in order to bypass
content-classification algorithms. Content classification is typically done by
fine-tuning pre-trained language models, which have been recently employed by
many natural-language-processing applications. In this work we study the effect
of applying pre-trained Arabic language models on "undotted" Arabic texts. We
suggest several ways of supporting undotted texts with pre-trained models,
without additional training, and measure their performance on two Arabic
natural-language-processing downstream tasks. The results are encouraging; in
one of the tasks our method shows nearly perfect performance.

    

### [[2111.09794] A Survey of Generalisation in Deep Reinforcement Learning](http://arxiv.org/abs/2111.09794)


  The study of generalisation in deep Reinforcement Learning (RL) aims to
produce RL algorithms whose policies generalise well to novel unseen situations
at deployment time, avoiding overfitting to their training environments.
Tackling this is vital if we are to deploy reinforcement learning algorithms in
real world scenarios, where the environment will be diverse, dynamic and
unpredictable. This survey is an overview of this nascent field. We provide a
unifying formalism and terminology for discussing different generalisation
problems, building upon previous works. We go on to categorise existing
benchmarks for generalisation, as well as current methods for tackling the
generalisation problem. Finally, we provide a critical discussion of the
current state of the field, including recommendations for future work. Among
other conclusions, we argue that taking a purely procedural content generation
approach to benchmark design is not conducive to progress in generalisation, we
suggest fast online adaptation and tackling RL-specific problems as some areas
for future work on methods for generalisation, and we recommend building
benchmarks in underexplored problem settings such as offline RL generalisation
and reward-function variation.

    

### [[2111.09800] Reinforcement Learning on Human Decision Models for Uniquely Collaborative AI Teammates](http://arxiv.org/abs/2111.09800)


  In 2021 the Johns Hopkins University Applied Physics Laboratory held an
internal challenge to develop artificially intelligent (AI) agents that could
excel at the collaborative card game Hanabi. Agents were evaluated on their
ability to play with human players whom the agents had never previously
encountered. This study details the development of the agent that won the
challenge by achieving a human-play average score of 16.5, outperforming the
current state-of-the-art for human-bot Hanabi scores. The winning agent's
development consisted of observing and accurately modeling the author's
decision making in Hanabi, then training with a behavioral clone of the author.
Notably, the agent discovered a human-complementary play style by first
mimicking human decision making, then exploring variations to the human-like
strategy that led to higher simulated human-bot scores. This work examines in
detail the design and implementation of this human compatible Hanabi teammate,
as well as the existence and implications of human-complementary strategies and
how they may be explored for more successful applications of AI in human
machine teams.

    

### [[2111.09805] On the Effectiveness of Sparsification for Detecting the Deep Unknowns](http://arxiv.org/abs/2111.09805)


  Detecting out-of-distribution (OOD) inputs is a central challenge for safely
deploying machine learning models in the real world. Previous methods commonly
rely on an OOD score derived from the overparameterized weight space, while
largely overlooking the role of sparsification. In this paper, we reveal
important insights that reliance on unimportant weights and units can directly
attribute to the brittleness of OOD detection. To mitigate the issue, we
propose a sparsification-based OOD detection framework termed DICE. Our key
idea is to rank weights based on a measure of contribution, and selectively use
the most salient weights to derive the output for OOD detection. We provide
both empirical and theoretical insights, characterizing and explaining the
mechanism by which DICE improves OOD detection. By pruning away noisy signals,
DICE provably reduces the output variance for OOD data, resulting in a sharper
output distribution and stronger separability from ID data. DICE establishes
superior performance, reducing the FPR95 by up to 24.69% compared to the
previous best method.

    

### [[2111.09808] Exploring the Limits of Epistemic Uncertainty Quantification in Low-Shot Settings](http://arxiv.org/abs/2111.09808)


  Uncertainty quantification in neural network promises to increase safety of
AI systems, but it is not clear how performance might vary with the training
set size. In this paper we evaluate seven uncertainty methods on Fashion MNIST
and CIFAR10, as we sub-sample and produce varied training set sizes. We find
that calibration error and out of distribution detection performance strongly
depend on the training set size, with most methods being miscalibrated on the
test set with small training sets. Gradient-based methods seem to poorly
estimate epistemic uncertainty and are the most affected by training set size.
We expect our results can guide future research into uncertainty quantification
and help practitioners select methods based on their particular available data.

    

### [[2111.09824] Machine Learning Assisted Approach for Security-Constrained Unit Commitment](http://arxiv.org/abs/2111.09824)


  Security-constrained unit commitment (SCUC) which is used in the power system
day-ahead generation scheduling is a mixed-integer linear programming problem
that is computationally intensive. A good warm-start solution or a reduced-SCUC
model can bring significant time savings. In this work, a novel approach is
proposed to effectively utilize machine learning (ML) to provide a good
starting solution and/or reduce the problem size of SCUC. An ML model using a
logistic regression algorithm is proposed and trained using historical nodal
demand profiles and the respective commitment schedules. The ML outputs are
processed and analyzed to assist SCUC. The proposed approach is validated on
several standard test systems namely, IEEE 24-bus system, IEEE 73-bus system,
IEEE 118-bus system, synthetic South Carolina 500-bus system, and Polish
2383-bus system. Simulation results demonstrate that the prediction from the
proposed machine learning model can provide a good warm-start solution and/or
reduce the number of variables and constraints in SCUC with minimal loss in
solution quality while substantially reducing the computing time.

    

### [[2111.09831] Causal Forecasting:Generalization Bounds for Autoregressive Models](http://arxiv.org/abs/2111.09831)


  Despite the increasing relevance of forecasting methods, the causal
implications of these algorithms remain largely unexplored. This is concerning
considering that, even under simplifying assumptions such as causal
sufficiency, the statistical risk of a model can differ significantly from its
\textit{causal risk}. Here, we study the problem of *causal generalization* --
generalizing from the observational to interventional distributions -- in
forecasting. Our goal is to find answers to the question: How does the efficacy
of an autoregressive (VAR) model in predicting statistical associations compare
with its ability to predict under interventions?
To this end, we introduce the framework of *causal learning theory* for
forecasting. Using this framework, we obtain a characterization of the
difference between statistical and causal risks, which helps identify sources
of divergence between them. Under causal sufficiency, the problem of causal
generalization amounts to learning under covariate shifts albeit with
additional structure (restriction to interventional distributions). This
structure allows us to obtain uniform convergence bounds on causal
generalizability for the class of VAR models. To the best of our knowledge,
this is the first work that provides theoretical guarantees for causal
generalization in the time-series setting.

    

### [[2111.09832] Merging Models with Fisher-Weighted Averaging](http://arxiv.org/abs/2111.09832)


  Transfer learning provides a way of leveraging knowledge from one task when
learning another task. Performing transfer learning typically involves
iteratively updating a model's parameters through gradient descent on a
training dataset. In this paper, we introduce a fundamentally different method
for transferring knowledge across models that amounts to "merging" multiple
models into one. Our approach effectively involves computing a weighted average
of the models' parameters. We show that this averaging is equivalent to
approximately sampling from the posteriors of the model weights. While using an
isotropic Gaussian approximation works well in some cases, we also demonstrate
benefits by approximating the precision matrix via the Fisher information. In
sum, our approach makes it possible to combine the "knowledge" in multiple
models at an extremely low computational cost compared to standard
gradient-based training. We demonstrate that model merging achieves comparable
performance to gradient descent-based transfer learning on intermediate-task
training and domain adaptation problems. We also show that our merging
procedure makes it possible to combine models in previously unexplored ways. To
measure the robustness of our approach, we perform an extensive ablation on the
design of our algorithm.

    

### [[2111.09838] On Efficient Uncertainty Estimation for Resource-Constrained Mobile Applications](http://arxiv.org/abs/2111.09838)


  Deep neural networks have shown great success in prediction quality while
reliable and robust uncertainty estimation remains a challenge. Predictive
uncertainty supplements model predictions and enables improved functionality of
downstream tasks including embedded and mobile applications, such as virtual
reality, augmented reality, sensor fusion, and perception. These applications
often require a compromise in complexity to obtain uncertainty estimates due to
very limited memory and compute resources. We tackle this problem by building
upon Monte Carlo Dropout (MCDO) models using the Axolotl framework;
specifically, we diversify sampled subnetworks, leverage dropout patterns, and
use a branching technique to improve predictive performance while maintaining
fast computations. We conduct experiments on (1) a multi-class classification
task using the CIFAR10 dataset, and (2) a more complex human body segmentation
task. Our results show the effectiveness of our approach by reaching close to
Deep Ensemble prediction quality and uncertainty estimation, while still
achieving faster inference on resource-limited mobile platforms.

    

### [[2111.09839] Training Neural Networks with Fixed Sparse Masks](http://arxiv.org/abs/2111.09839)


  During typical gradient-based training of deep neural networks, all of the
model's parameters are updated at each iteration. Recent work has shown that it
is possible to update only a small subset of the model's parameters during
training, which can alleviate storage and communication requirements. In this
paper, we show that it is possible to induce a fixed sparse mask on the model's
parameters that selects a subset to update over many iterations. Our method
constructs the mask out of the $k$ parameters with the largest Fisher
information as a simple approximation as to which parameters are most important
for the task at hand. In experiments on parameter-efficient transfer learning
and distributed training, we show that our approach matches or exceeds the
performance of other methods for training with sparse updates while being more
efficient in terms of memory usage and communication costs. We release our code
publicly to promote further applications of our approach.

    

### [[2111.09847] Edge-preserving Domain Adaptation for semantic segmentation of Medical Images](http://arxiv.org/abs/2111.09847)


  Domain Adaptation is a technique to address the lack of massive amounts of
labeled data in unseen environments. Unsupervised domain adaptation is proposed
to adapt a model to new modalities using solely labeled source data and
unlabeled target domain data. Though many image-spaces domain adaptation
methods have been proposed to capture pixel-level domain-shift, such techniques
may fail to maintain high-level semantic information for the segmentation task.
For the case of biomedical images, fine details such as blood vessels can be
lost during the image transformation operations between domains. In this work,
we propose a model that adapts between domains using cycle-consistent loss
while maintaining edge details of the original images by enforcing an
edge-based loss during the adaptation process. We demonstrate the effectiveness
of our algorithm by comparing it to other approaches on two eye fundus vessels
segmentation datasets. We achieve 1.1 to 9.2 increment in DICE score compared
to the SOTA and ~5.2 increments compared to a vanilla CycleGAN implementation.

    

### [[2111.09858] Successor Feature Landmarks for Long-Horizon Goal-Conditioned Reinforcement Learning](http://arxiv.org/abs/2111.09858)


  Operating in the real-world often requires agents to learn about a complex
environment and apply this understanding to achieve a breadth of goals. This
problem, known as goal-conditioned reinforcement learning (GCRL), becomes
especially challenging for long-horizon goals. Current methods have tackled
this problem by augmenting goal-conditioned policies with graph-based planning
algorithms. However, they struggle to scale to large, high-dimensional state
spaces and assume access to exploration mechanisms for efficiently collecting
training data. In this work, we introduce Successor Feature Landmarks (SFL), a
framework for exploring large, high-dimensional environments so as to obtain a
policy that is proficient for any goal. SFL leverages the ability of successor
features (SF) to capture transition dynamics, using it to drive exploration by
estimating state-novelty and to enable high-level planning by abstracting the
state-space as a non-parametric landmark-based graph. We further exploit SF to
directly compute a goal-conditioned policy for inter-landmark traversal, which
we use to execute plans to "frontier" landmarks at the edge of the explored
state space. We show in our experiments on MiniGrid and ViZDoom that SFL
enables efficient exploration of large, high-dimensional state spaces and
outperforms state-of-the-art baselines on long-horizon GCRL tasks.

    

### [[2111.09863] A Secure Experimentation Sandbox for the design and execution of trusted and secure analytics in the aviation domain](http://arxiv.org/abs/2111.09863)


  The aviation industry as well as the industries that benefit and are linked
to it are ripe for innovation in the form of Big Data analytics. The number of
available big data technologies is constantly growing, while at the same time
the existing ones are rapidly evolving and empowered with new features.
However, the Big Data era imposes the crucial challenge of how to effectively
handle information security while managing massive and rapidly evolving data
from heterogeneous data sources. While multiple technologies have emerged,
there is a need to find a balance between multiple security requirements,
privacy obligations, system performance and rapid dynamic analysis on large
datasets. The current paper aims to introduce the ICARUS Secure Experimentation
Sandbox of the ICARUS platform. The ICARUS platform aims to provide a big
data-enabled platform that aspires to become an 'one-stop shop' for aviation
data and intelligence marketplace that provides a trusted and secure
'sandboxed' analytics workspace, allowing the exploration, integration and deep
analysis of original and derivative data in a trusted and fair manner. Towards
this end, a Secure Experimentation Sandbox has been designed and integrated in
the ICARUS platform offering, that enables the provisioning of a sophisticated
environment that can completely guarantee the safety and confidentiality of
data, allowing to any interested party to utilise the platform to conduct
analytical experiments in closed-lab conditions.

    

### [[2111.09872] A big data intelligence marketplace and secure analytics experimentation platform for the aviation industry](http://arxiv.org/abs/2111.09872)


  The unprecedented volume, diversity and richness of aviation data that can be
acquired, generated, stored, and managed provides unique capabilities for the
aviation-related industries and pertains value that remains to be unlocked with
the adoption of the innovative Big Data Analytics technologies. Despite the
large efforts and investments on research and innovation, the Big Data
technologies introduce a number of challenges to its adopters. Besides the
effective storage and access to the underlying big data, efficient data
integration and data interoperability should be considered, while at the same
time multiple data sources should be effectively combined by performing data
exchange and data sharing between the different stakeholders. However, this
reveals additional challenges for the crucial preservation of the information
security of the collected data, the trusted and secure data exchange and data
sharing, as well as the robust data access control. The current paper aims to
introduce the ICARUS big data-enabled platform that aims provide a multi-sided
platform that offers a novel aviation data and intelligence marketplace
accompanied by a trusted and secure analytics workspace. It holistically
handles the complete big data lifecycle from the data collection, data curation
and data exploration to the data integration and data analysis of data
originating from heterogeneous data sources with different velocity, variety
and volume in a trusted and secure manner.

    

### [[2111.09884] Assisted Robust Reward Design](http://arxiv.org/abs/2111.09884)


  Real-world robotic tasks require complex reward functions. When we define the
problem the robot needs to solve, we pretend that a designer specifies this
complex reward exactly, and it is set in stone from then on. In practice,
however, reward design is an iterative process: the designer chooses a reward,
eventually encounters an "edge-case" environment where the reward incentivizes
the wrong behavior, revises the reward, and repeats. What would it mean to
rethink robotics problems to formally account for this iterative nature of
reward design? We propose that the robot not take the specified reward for
granted, but rather have uncertainty about it, and account for the future
design iterations as future evidence. We contribute an Assisted Reward Design
method that speeds up the design process by anticipating and influencing this
future evidence: rather than letting the designer eventually encounter failure
cases and revise the reward then, the method actively exposes the designer to
such environments during the development phase. We test this method in a
simplified autonomous driving task and find that it more quickly improves the
car's behavior in held-out environments by proposing environments that are
"edge cases" for the current reward.

    

### [[2111.09885] Optimal Simple Regret in Bayesian Best Arm Identification](http://arxiv.org/abs/2111.09885)


  We consider Bayesian best arm identification in the multi-armed bandit
problem. Assuming certain continuity conditions of the prior, we characterize
the rate of the Bayesian simple regret. Differing from Bayesian regret
minimization (Lai, 1987), the leading factor in Bayesian simple regret derives
from the region where the gap between optimal and sub-optimal arms is smaller
than $\sqrt{\frac{\log T}{T}}$. We propose a simple and easy-to-compute
algorithm with its leading factor matches with the lower bound up to a constant
factor; simulation results support our theoretical findings.

    

### [[2111.09887] PyTorchVideo: A Deep Learning Library for Video Understanding](http://arxiv.org/abs/2111.09887)


  We introduce PyTorchVideo, an open-source deep-learning library that provides
a rich set of modular, efficient, and reproducible components for a variety of
video understanding tasks, including classification, detection, self-supervised
learning, and low-level processing. The library covers a full stack of video
understanding tools including multimodal data loading, transformations, and
models that reproduce state-of-the-art performance. PyTorchVideo further
supports hardware acceleration that enables real-time inference on mobile
devices. The library is based on PyTorch and can be used by any training
framework; for example, PyTorchLightning, PySlowFast, or Classy Vision.
PyTorchVideo is available at this https URL


### [[1912.02368] Inter-Level Cooperation in Hierarchical Reinforcement Learning](http://arxiv.org/abs/1912.02368)


  Hierarchies of temporally decoupled policies present a promising approach for
enabling structured exploration in complex long-term planning problems. To
fully achieve this approach an end-to-end training paradigm is needed. However,
training these multi-level policies has had limited success due to challenges
arising from interactions between the goal-assigning and goal-achieving levels
within a hierarchy. In this article, we consider the policy optimization
process as a multi-agent process. This allows us to draw on connections between
communication and cooperation in multi-agent RL, and demonstrate the benefits
of increased cooperation between sub-policies on the training performance of
the overall policy. We introduce a simple yet effective technique for inducing
inter-level cooperation by modifying the objective function and subsequent
gradients of higher-level policies. Experimental results on a wide variety of
simulated robotics and traffic control tasks demonstrate that inducing
cooperation results in stronger performing policies and increased sample
efficiency on a set of difficult long time horizon tasks. We also find that
goal-conditioned policies trained using our method display better transfer to
new tasks, highlighting the benefits of our method in learning task-agnostic
lower-level behaviors. Videos and code are available at:
this https URL.

    

### [[2002.08538] Non-asymptotic and Accurate Learning of Nonlinear Dynamical Systems](http://arxiv.org/abs/2002.08538)


  We consider the problem of learning stabilizable systems governed by
nonlinear state equation $h_{t+1}=\phi(h_t,u_t;\theta)+w_t$. Here $\theta$ is
the unknown system dynamics, $h_t $ is the state, $u_t$ is the input and $w_t$
is the additive noise vector. We study gradient based algorithms to learn the
system dynamics $\theta$ from samples obtained from a single finite trajectory.
If the system is run by a stabilizing input policy, we show that
temporally-dependent samples can be approximated by i.i.d. samples via a
truncation argument by using mixing-time arguments. We then develop new
guarantees for the uniform convergence of the gradients of empirical loss.
Unlike existing work, our bounds are noise sensitive which allows for learning
ground-truth dynamics with high accuracy and small sample complexity. Together,
our results facilitate efficient learning of the general nonlinear system under
stabilizing policy. We specialize our guarantees to entry-wise nonlinear
activations and verify our theory in various numerical experiments

    

### [[2003.04675] Towards Interpretable ANNs: An Exact Transformation to Multi-Class Multivariate Decision Trees](http://arxiv.org/abs/2003.04675)


  On the one hand, artificial neural networks (ANNs) are commonly labelled as
black-boxes, lacking interpretability; an issue that hinders human
understanding of ANNs' behaviors. A need exists to generate a meaningful
sequential logic of the ANN for interpreting a production process of a specific
output. On the other hand, decision trees exhibit better interpretability and
expressive power due to their representation language and the existence of
efficient algorithms to transform the trees into rules. However, growing a
decision tree based on the available data could produce larger than necessary
trees or trees that do not generalise well. In this paper, we introduce two
novel multivariate decision tree (MDT) algorithms for rule extraction from
ANNs: an Exact-Convertible Decision Tree (EC-DT) and an Extended C-Net
algorithm. They both transform a neural network with Rectified Linear Unit
activation functions into a representative tree, which can further be used to
extract multivariate rules for reasoning. While the EC-DT translates an ANN in
a layer-wise manner to represent exactly the decision boundaries implicitly
learned by the hidden layers of the network, the Extended C-Net combines the
decompositional approach from EC-DT with a C5 tree learning algorithm to form
decision rules. The results suggest that while EC-DT is superior in preserving
the structure and the fidelity of ANN, Extended C-Net generates the most
compact and highly effective trees from ANN. Both proposed MDT algorithms
generate rules including combinations of multiple attributes for precise
interpretations for decision-making.

    

### [[2003.05425] Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs](http://arxiv.org/abs/2003.05425)


  A common approach to define convolutions on meshes is to interpret them as a
graph and apply graph convolutional networks (GCNs). Such GCNs utilize
isotropic kernels and are therefore insensitive to the relative orientation of
vertices and thus to the geometry of the mesh as a whole. We propose Gauge
Equivariant Mesh CNNs which generalize GCNs to apply anisotropic gauge
equivariant kernels. Since the resulting features carry orientation
information, we introduce a geometric message passing scheme defined by
parallel transporting features over mesh edges. Our experiments validate the
significantly improved expressivity of the proposed model over conventional
GCNs and other methods.

    

### [[2006.05911] Continuous Action Reinforcement Learning from a Mixture of Interpretable Experts](http://arxiv.org/abs/2006.05911)


  Reinforcement learning (RL) has demonstrated its ability to solve high
dimensional tasks by leveraging non-linear function approximators. However,
these successes are mostly achieved by 'black-box' policies in simulated
domains. When deploying RL to the real world, several concerns regarding the
use of a 'black-box' policy might be raised. In order to make the learned
policies more transparent, we propose in this paper a policy iteration scheme
that retains a complex function approximator for its internal value predictions
but constrains the policy to have a concise, hierarchical, and human-readable
structure, based on a mixture of interpretable experts. Each expert selects a
primitive action according to a distance to a prototypical state. A key design
decision to keep such experts interpretable is to select the prototypical
states from trajectory data. The main technical contribution of the paper is to
address the challenges introduced by this non-differentiable prototypical state
selection procedure. Experimentally, we show that our proposed algorithm can
learn compelling policies on continuous action deep RL benchmarks, matching the
performance of neural network based policies, but returning policies that are
more amenable to human inspection than neural network or linear-in-feature
policies.

    

### [[2007.04433] Error Estimation and Correction from within Neural Network Differential Equation Solvers](http://arxiv.org/abs/2007.04433)


  Neural Network Differential Equation (NN DE) solvers have surged in
popularity due to a combination of factors: computational advances making their
optimization more tractable, their capacity to handle high dimensional
problems, easy interpret-ability of their models, etc. However, almost all NN
DE solvers suffer from a fundamental limitation: they are trained using loss
functions that depend only implicitly on the error associated with the
estimate. As such, validation and error analysis of solution estimates requires
knowledge of the true solution. Indeed, if the true solution is unknown, we are
often reduced to simply hoping that a "low enough" loss implies "small enough"
errors, since explicit relationships between the two are not available/well
defined. In this work, we describe a general strategy for efficiently
constructing error estimates and corrections for Neural Network Differential
Equation solvers. Our methods do not require advance knowledge of the true
solutions and obtain explicit relationships between loss functions and the
error associated with solution estimates. In turn, these explicit relationships
directly allow us to estimate and correct for the errors.

    

### [[2007.06437] A Provably Efficient Sample Collection Strategy for Reinforcement Learning](http://arxiv.org/abs/2007.06437)


  One of the challenges in online reinforcement learning (RL) is that the agent
needs to trade off the exploration of the environment and the exploitation of
the samples to optimize its behavior. Whether we optimize for regret, sample
complexity, state-space coverage or model estimation, we need to strike a
different exploration-exploitation trade-off. In this paper, we propose to
tackle the exploration-exploitation problem following a decoupled approach
composed of: 1) An "objective-specific" algorithm that (adaptively) prescribes
how many samples to collect at which states, as if it has access to a
generative model (i.e., a simulator of the environment); 2) An
"objective-agnostic" sample collection exploration strategy responsible for
generating the prescribed samples as fast as possible. Building on recent
methods for exploration in the stochastic shortest path problem, we first
provide an algorithm that, given as input the number of samples $b(s,a)$ needed
in each state-action pair, requires $\tilde{O}(B D + D^{3/2} S^2 A)$ time steps
to collect the $B=\sum_{s,a} b(s,a)$ desired samples, in any unknown
communicating MDP with $S$ states, $A$ actions and diameter $D$. Then we show
how this general-purpose exploration algorithm can be paired with
"objective-specific" strategies that prescribe the sample requirements to
tackle a variety of settings -- e.g., model estimation, sparse reward
discovery, goal-free cost-free exploration in communicating MDPs -- for which
we obtain improved or novel sample complexity guarantees.

    

### [[2010.08755] Variational Dynamic for Self-Supervised Exploration in Deep Reinforcement Learning](http://arxiv.org/abs/2010.08755)


  Efficient exploration remains a challenging problem in reinforcement
learning, especially for tasks where extrinsic rewards from environments are
sparse or even totally disregarded. Significant advances based on intrinsic
motivation show promising results in simple environments but often get stuck in
environments with multimodal and stochastic dynamics. In this work, we propose
a variational dynamic model based on the conditional variational inference to
model the multimodality and stochasticity. We consider the environmental
state-action transition as a conditional generative process by generating the
next-state prediction under the condition of the current state, action, and
latent variable, which provides a better understanding of the dynamics and
leads a better performance in exploration. We derive an upper bound of the
negative log-likelihood of the environmental transition and use such an upper
bound as the intrinsic reward for exploration, which allows the agent to learn
skills by self-supervised exploration without observing extrinsic rewards. We
evaluate the proposed method on several image-based simulation tasks and a real
robotic manipulating task. Our method outperforms several state-of-the-art
environment model-based exploration approaches.

    

### [[2011.07607] Deep Ordinal Regression using Optimal Transport Loss and Unimodal Output Probabilities](http://arxiv.org/abs/2011.07607)


  It is often desired that ordinal regression models yield unimodal
predictions. However, in many recent works this characteristic is either
absent, or implemented using soft targets, which do not guarantee unimodal
outputs at inference. In addition, we argue that the standard maximum
likelihood objective is not suitable for ordinal regression problems, and that
optimal transport is better suited for this task, as it naturally captures the
order of the classes. In this work, we propose a framework for deep ordinal
regression, based on unimodal output distribution and optimal transport loss.
Inspired by the well-known Proportional Odds model, we propose to modify its
design by using an architectural mechanism which guarantees that the model
output distribution will be unimodal. We empirically analyze the different
components of our proposed approach and demonstrate their contribution to the
performance of the model. Experimental results on eight real-world datasets
demonstrate that our proposed approach consistently performs on par with and
often better than several recently proposed deep learning approaches for deep
ordinal regression with unimodal output probabilities, while having guarantee
on the output unimodality. In addition, we demonstrate that proposed approach
is less overconfident than current baselines.

    

### [[2012.03414] Vehicular Cooperative Perception Through Action Branching and Federated Reinforcement Learning](http://arxiv.org/abs/2012.03414)


  Cooperative perception plays a vital role in extending a vehicle's sensing
range beyond its line-of-sight. However, exchanging raw sensory data under
limited communication resources is infeasible. Towards enabling an efficient
cooperative perception, vehicles need to address the following fundamental
question: What sensory data needs to be shared?, at which resolution?, and with
which vehicles? To answer this question, in this paper, a novel framework is
proposed to allow reinforcement learning (RL)-based vehicular association,
resource block (RB) allocation, and content selection of cooperative perception
messages (CPMs) by utilizing a quadtree-based point cloud compression
mechanism. Furthermore, a federated RL approach is introduced in order to speed
up the training process across vehicles. Simulation results show the ability of
the RL agents to efficiently learn the vehicles' association, RB allocation,
and message content selection while maximizing vehicles' satisfaction in terms
of the received sensory information. The results also show that federated RL
improves the training process, where better policies can be achieved within the
same amount of time compared to the non-federated approach.

    

### [[2012.12291] Learning a Group-Aware Policy for Robot Navigation](http://arxiv.org/abs/2012.12291)


  Human-aware robot navigation promises a range of applications in which mobile
robots bring versatile assistance to people in common human environments. While
prior research has mostly focused on modeling pedestrians as independent,
intentional individuals, people move in groups; consequently, it is imperative
for mobile robots to respect human groups when navigating around people. This
paper explores learning group-aware navigation policies based on dynamic group
formation using deep reinforcement learning. Through simulation experiments, we
show that group-aware policies, compared to baseline policies that neglect
human groups, achieve greater robot navigation performance (e.g., fewer
collisions), minimize violation of social norms and discomfort, and reduce the
robot's movement impact on pedestrians. Our results contribute to the
development of social navigation and the integration of mobile robots into
human environments.

    

### [[2102.02979] The Fourier Discrepancy Function](http://arxiv.org/abs/2102.02979)


  In this paper, we propose the Fourier Discrepancy Function, a new discrepancy
to compare discrete probability measures. We show that this discrepancy takes
into account the geometry of the underlying space. We prove that the Fourier
Discrepancy is convex, twice differentiable, and that its gradient has an
explicit formula. We also provide a compelling statistical interpretation.
Finally, we study the lower and upper tight bounds for the Fourier Discrepancy
in terms of the Total Variation distance.

    

### [[2102.09479] Make Sure You're Unsure: A Framework for Verifying Probabilistic Specifications](http://arxiv.org/abs/2102.09479)


  Most real world applications require dealing with stochasticity like sensor
noise or predictive uncertainty, where formal specifications of desired
behavior are inherently probabilistic. Despite the promise of formal
verification in ensuring the reliability of neural networks, progress in the
direction of probabilistic specifications has been limited. In this direction,
we first introduce a general formulation of probabilistic specifications for
neural networks, which captures both probabilistic networks (e.g., Bayesian
neural networks, MC-Dropout networks) and uncertain inputs (distributions over
inputs arising from sensor noise or other perturbations). We then propose a
general technique to verify such specifications by generalizing the notion of
Lagrangian duality, replacing standard Lagrangian multipliers with "functional
multipliers" that can be arbitrary functions of the activations at a given
layer. We show that an optimal choice of functional multipliers leads to exact
verification (i.e., sound and complete verification), and for specific forms of
multipliers, we develop tractable practical verification algorithms.
We empirically validate our algorithms by applying them to Bayesian Neural
Networks (BNNs) and MC Dropout Networks, and certifying properties such as
adversarial robustness and robust detection of out-of-distribution (OOD) data.
On these tasks we are able to provide significantly stronger guarantees when
compared to prior work -- for instance, for a VGG-64 MC-Dropout CNN trained on
CIFAR-10, we improve the certified AUC (a verified lower bound on the true AUC)
for robust OOD detection (on CIFAR-100) from $0\% \rightarrow 29\%$. Similarly,
for a BNN trained on MNIST, we improve on the robust accuracy from $60.2\%
\rightarrow 74.6\%$. Further, on a novel specification -- distributionally
robust OOD detection -- we improve the certified AUC from $5\% \rightarrow
23\%$.

    

### [[2102.13128] An Online Learning Approach to Interpolation and Extrapolation in Domain Generalization](http://arxiv.org/abs/2102.13128)


  A popular assumption for out-of-distribution generalization is that the
training data comprises sub-datasets, each drawn from a distinct distribution;
the goal is then to "interpolate" these distributions and "extrapolate" beyond
them -- this objective is broadly known as domain generalization. A common
belief is that ERM can interpolate but not extrapolate and that the latter is
considerably more difficult, but these claims are vague and lack formal
justification. In this work, we recast generalization over sub-groups as an
online game between a player minimizing risk and an adversary presenting new
test distributions. Under an existing notion of inter- and extrapolation based
on reweighting of sub-group likelihoods, we rigorously demonstrate that
extrapolation is computationally much harder than interpolation, though their
statistical complexity is not significantly different. Furthermore, we show
that ERM -- or a noisy variant -- is provably minimax-optimal for both tasks.
Our framework presents a new avenue for the formal analysis of domain
generalization algorithms which may be of independent interest.

    

### [[2103.00255] Expert Decision Support System for aeroacoustic source type identification using clustering](http://arxiv.org/abs/2103.00255)


  This paper presents an Expert Decision Support System for the identification
of time-invariant, aeroacoustic source types. The system comprises two steps:
first, acoustic properties are calculated based on spectral and spatial
information. Second, clustering is performed based on these properties. The
clustering aims at helping and guiding an expert for quick identification of
different source types, providing an understanding of how sources differ. This
supports the expert in determining similar or atypical behavior. A variety of
features are proposed for capturing the characteristics of the sources. These
features represent aeroacoustic properties that can be interpreted by both the
machine and by experts. The features are independent of the absolute Mach
number which enables the proposed method to cluster data measured at different
flow configurations. The method is evaluated on deconvolved beamforming data
from two scaled airframe half-model measurements. For this exemplary data, the
proposed support system method results in clusters that mostly correspond to
the source types identified by the authors. The clustering also provides the
mean feature values and the cluster hierarchy for each cluster and for each
cluster member a clustering confidence. This additional information makes the
results transparent and allows the expert to understand the clustering choices.

    

### [[2103.12024] Stability and Deviation Optimal Risk Bounds with Convergence Rate $O(1/n)$](http://arxiv.org/abs/2103.12024)


  The sharpest known high probability generalization bounds for uniformly
stable algorithms (Feldman, Vondrák, 2018, 2019), (Bousquet, Klochkov,
Zhivotovskiy, 2020) contain a generally inevitable sampling error term of order
$\Theta(1/\sqrt{n})$. When applied to excess risk bounds, this leads to
suboptimal results in several standard stochastic convex optimization problems.
We show that if the so-called Bernstein condition is satisfied, the term
$\Theta(1/\sqrt{n})$ can be avoided, and high probability excess risk bounds of
order up to $O(1/n)$ are possible via uniform stability. Using this result, we
show a high probability excess risk bound with the rate $O(\log n/n)$ for
strongly convex and Lipschitz losses valid for \emph{any} empirical risk
minimization method. This resolves a question of Shalev-Shwartz, Shamir,
Srebro, and Sridharan (2009). We discuss how $O(\log n/n)$ high probability
excess risk bounds are possible for projected gradient descent in the case of
strongly convex and Lipschitz losses without the usual smoothness assumption.

    

### [[2104.02710] The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions](http://arxiv.org/abs/2104.02710)


  Multi-agent behavior modeling aims to understand the interactions that occur
between agents. We present a multi-agent dataset from behavioral neuroscience,
the Caltech Mouse Social Interactions (CalMS21) Dataset. Our dataset consists
of trajectory data of social interactions, recorded from videos of freely
behaving mice in a standard resident-intruder assay. To help accelerate
behavioral studies, the CalMS21 dataset provides benchmarks to evaluate the
performance of automated behavior classification methods in three settings: (1)
for training on large behavioral datasets all annotated by a single annotator,
(2) for style transfer to learn inter-annotator differences in behavior
definitions, and (3) for learning of new behaviors of interest given limited
training data. The dataset consists of 6 million frames of unlabeled tracked
poses of interacting mice, as well as over 1 million frames with tracked poses
and corresponding frame-level behavior annotations. The challenge of our
dataset is to be able to classify behaviors accurately using both labeled and
unlabeled tracking data, as well as being able to generalize to new settings.

    

### [[2104.07737] Random Persistence Diagram Generation](http://arxiv.org/abs/2104.07737)


  Topological data analysis (TDA) studies the shape patterns of data.
Persistent homology (PH) is a widely used method in TDA that summarizes
homological features of data at multiple scales and stores them in persistence
diagrams (PDs). In this paper, we propose a random persistence diagram
generation (RPDG) method that generates a sequence of random PDs from the ones
produced by the data. RPDG is underpinned by (i) a model based on pairwise
interacting point processes for inference of persistence diagrams, and (ii) by
a reversible jump Markov chain Monte Carlo (RJ-MCMC) algorithm for generating
samples of PDs. A first example, which is based on a synthetic dataset,
demonstrates the efficacy of RPDG and provides a detailed comparison with other
existing methods for sampling PDs. A second example demonstrates the utility of
RPDG to solve a materials science problem given a real dataset of small sample
size.

    

### [[2105.03061] Deep reinforcement learning-designed radiofrequency waveform in MRI](http://arxiv.org/abs/2105.03061)


  Carefully engineered radiofrequency (RF) pulses play a key role in a number
of systems such as mobile phone, radar, and magnetic resonance imaging. The
design of an RF waveform, however, is often posed as an inverse problem with no
general solution. As a result, various design methods each with a specific
purpose have been developed based on the intuition of human experts. In this
work, we propose an artificial intelligence (AI)-powered RF pulse design
framework, DeepRF, which utilizes the self-learning characteristics of deep
reinforcement learning to generate a novel RF pulse. The effectiveness of
DeepRF is demonstrated using four types of RF pulses that are commonly used.
The DeepRF-designed pulses successfully satisfy the design criteria while
reporting reduced energy. Analyses demonstrate the pulses utilize new
mechanisms of magnetization manipulation, suggesting the potentials of DeepRF
in discovering unseen design dimensions beyond human intuition. This work may
lay the foundation for an emerging field of AI-driven RF waveform design.

    

### [[2105.13283] Deep Ensembles from a Bayesian Perspective](http://arxiv.org/abs/2105.13283)


  Deep ensembles can be considered as the current state-of-the-art for
uncertainty quantification in deep learning. While the approach was originally
proposed as a non-Bayesian technique, arguments supporting its Bayesian footing
have been put forward as well. We show that deep ensembles can be viewed as an
approximate Bayesian method by specifying the corresponding assumptions. Our
findings lead to an improved approximation which results in an enlarged
epistemic part of the uncertainty. Numerical examples suggest that the improved
approximation can lead to more reliable uncertainties. Analytical derivations
ensure easy calculation of results.

    

### [[2106.00009] Deep-Learning Discovers Macroscopic Governing Equations for Viscous Gravity Currents from Microscopic Simulation Data](http://arxiv.org/abs/2106.00009)


  Although deep-learning has been successfully applied in a variety of science
and engineering problems owing to its strong high-dimensional nonlinear mapping
capability, it is of limited use in scientific knowledge discovery. In this
work, we propose a deep-learning based framework to discover the macroscopic
governing equation of viscous gravity current based on high-resolution
microscopic simulation data without the need for prior knowledge of underlying
terms. For two typical scenarios with different viscosity ratios, the
deep-learning based equations exactly capture the same dominated terms as the
theoretically derived equations for describing long-term asymptotic behaviors,
which validates the proposed framework. Unknown macroscopic equations are then
obtained for describing short-term behaviors, and additional deep-learned
compensation terms are eventually discovered. Comparison of posterior tests
shows that the deep-learning based PDEs actually perform better than the
theoretically derived PDEs in predicting evolving viscous gravity currents for
both long-term and short-term regimes. Moreover, the proposed framework is
proven to be very robust against non-biased data noise for training, which is
up to 20%. Consequently, the presented deep-learning framework shows
considerable potential for discovering unrevealed intrinsic laws in scientific
semantic space from raw experimental or simulation results in data space.

    

### [[2106.00719] Stochastic Collapsed Variational Inference for Structured Gaussian Process Regression Network](http://arxiv.org/abs/2106.00719)


  This paper presents an efficient variational inference framework for deriving
a family of structured gaussian process regression network (SGPRN) models. The
key idea is to incorporate auxiliary inducing variables in latent functions and
jointly treats both the distributions of the inducing variables and
hyper-parameters as variational parameters. Then we propose structured variable
distributions and marginalize latent variables, which enables the
decomposability of a tractable variational lower bound and leads to stochastic
optimization. Our inference approach is able to model data in which outputs do
not share a common input set with a computational complexity independent of the
size of the inputs and outputs and thus easily handle datasets with missing
values. We illustrate the performance of our method on synthetic data and real
datasets and show that our model generally provides better imputation results
on missing data than the state-of-the-art. We also provide a visualization
approach for time-varying correlation across outputs in electrocoticography
data and those estimates provide insight to understand the neural population
dynamics.

    

### [[2106.01834] Continual Learning in Deep Networks: an Analysis of the Last Layer](http://arxiv.org/abs/2106.01834)


  We study how different output layers in a deep neural network learn and
forget in continual learning settings. The following three factors can affect
catastrophic forgetting in the output layer: (1) weights modifications, (2)
interference, and (3) projection drift. In this paper, our goal is to provide
more insights into how changing the output layers may address (1) and (2). Some
potential solutions to those issues are proposed and evaluated here in several
continual learning scenarios. We show that the best-performing type of the
output layer depends on the data distribution drifts and/or the amount of data
available. In particular, in some cases where a standard linear layer would
fail, it turns out that changing parameterization is sufficient in order to
achieve a significantly better performance, whithout introducing a
continual-learning algorithm and instead using the standard SGD to train a
model. Our analysis and results shed light on the dynamics of the output layer
in continual learning scenarios, and suggest a way of selecting the best type
of output layer for a given scenario.

    

### [[2106.05410] DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection](http://arxiv.org/abs/2106.05410)


  Semi-supervised anomaly detection aims to detect anomalies from normal
samples using a model that is trained on normal data. With recent advancements
in deep learning, researchers have designed efficient deep anomaly detection
methods. Existing works commonly use neural networks to map the data into a
more informative representation and then apply an anomaly detection algorithm.
In this paper, we propose a method, DASVDD, that jointly learns the parameters
of an autoencoder while minimizing the volume of an enclosing hyper-sphere on
its latent representation. We propose an anomaly score which is a combination
of autoencoder's reconstruction error and the distance from the center of the
enclosing hypersphere in the latent representation. Minimizing this anomaly
score aids us in learning the underlying distribution of the normal class
during training. Including the reconstruction error in the anomaly score
ensures that DASVDD does not suffer from the common hypersphere collapse issue
since the DASVDD model does not converge to the trivial solution of mapping all
inputs to a constant point in the latent representation. Experimental
evaluations on several benchmark datasets show that the proposed method
outperforms the commonly used state-of-the-art anomaly detection algorithms
while maintaining robust performance across different anomaly classes.

    

### [[2106.08849] How memory architecture affects learning in a simple POMDP: the two-hypothesis testing problem](http://arxiv.org/abs/2106.08849)


  Reinforcement learning is generally difficult for partially observable Markov
decision processes (POMDPs), which occurs when the agent's observation is
partial or noisy. To seek good performance in POMDPs, one strategy is to endow
the agent with a finite memory, whose update is governed by the policy.
However, policy optimization is non-convex in that case and can lead to poor
training performance for random initialization. The performance can be
empirically improved by constraining the memory architecture, then sacrificing
optimality to facilitate training. Here we study this trade-off in a
two-hypothesis testing problem, akin to the two-arm bandit problem. We compare
two extreme cases: (i) the random access memory where any transitions between
$M$ memory states are allowed and (ii) a fixed memory where the agent can
access its last $m$ actions and rewards. For (i), the probability $q$ to play
the worst arm is known to be exponentially small in $M$ for the optimal policy.
Our main result is to show that similar performance can be reached for (ii) as
well, despite the simplicity of the memory architecture: using a conjecture on
Gray-ordered binary necklaces, we find policies for which $q$ is exponentially
small in $2^m$, i.e. $q\sim\alpha^{2^m}$ with $\alpha < 1$. In addition, we
observe empirically that training from random initialization leads to very poor
results for (i), and significantly better results for (ii) thanks to the
constraints on the memory architecture.

    

### [[2106.15230] Convolutional Hypercomplex Embeddings for Link Prediction](http://arxiv.org/abs/2106.15230)


  Knowledge graph embedding research has mainly focused on the two smallest
normed division algebras, $\mathbb{R}$ and $\mathbb{C}$. Recent results suggest
that trilinear products of quaternion-valued embeddings can be a more effective
means to tackle link prediction. In addition, models based on convolutions on
real-valued embeddings often yield state-of-the-art results for link
prediction. In this paper, we investigate a composition of convolution
operations with hypercomplex multiplications. We propose the four approaches
QMult, OMult, ConvQ and ConvO to tackle the link prediction problem. QMult and
OMult can be considered as quaternion and octonion extensions of previous
state-of-the-art approaches, including DistMult and ComplEx. ConvQ and ConvO
build upon QMult and OMult by including convolution operations in a way
inspired by the residual learning framework. We evaluated our approaches on
seven link prediction datasets including WN18RR, FB15K-237 and YAGO3-10.
Experimental results suggest that the benefits of learning hypercomplex-valued
vector representations become more apparent as the size and complexity of the
knowledge graph grows. ConvO outperforms state-of-the-art approaches on
FB15K-237 in MRR, Hit@1 and Hit@3, while QMult, OMult, ConvQ and ConvO
outperform state-of-the-approaches on YAGO3-10 in all metrics. Results also
suggest that link prediction performances can be further improved via
prediction averaging. To foster reproducible research, we provide an
open-source implementation of approaches, including training and evaluation
scripts as well as pretrained models.

    

### [[2107.00451] VideoLightFormer: Lightweight Action Recognition using Transformers](http://arxiv.org/abs/2107.00451)


  Efficient video action recognition remains a challenging problem. One large
model after another takes the place of the state-of-the-art on the Kinetics
dataset, but real-world efficiency evaluations are often lacking. In this work,
we fill this gap and investigate the use of transformers for efficient action
recognition. We propose a novel, lightweight action recognition architecture,
VideoLightFormer. In a factorized fashion, we carefully extend the 2D
convolutional Temporal Segment Network with transformers, while maintaining
spatial and temporal video structure throughout the entire model. Existing
methods often resort to one of the two extremes, where they either apply huge
transformers to video features, or minimal transformers on highly pooled video
features. Our method differs from them by keeping the transformer models small,
but leveraging full spatiotemporal feature structure. We evaluate
VideoLightFormer in a high-efficiency setting on the temporally-demanding
EPIC-KITCHENS-100 and Something-Something-V2 (SSV2) datasets and find that it
achieves a better mix of efficiency and accuracy than existing state-of-the-art
models, apart from the Temporal Shift Module on SSV2.

    

### [[2110.10906] Single-Modal Entropy based Active Learning for Visual Question Answering](http://arxiv.org/abs/2110.10906)


  Constructing a large-scale labeled dataset in the real world, especially for
high-level tasks (eg, Visual Question Answering), can be expensive and
time-consuming. In addition, with the ever-growing amounts of data and
architecture complexity, Active Learning has become an important aspect of
computer vision research. In this work, we address Active Learning in the
multi-modal setting of Visual Question Answering (VQA). In light of the
multi-modal inputs, image and question, we propose a novel method for effective
sample acquisition through the use of ad hoc single-modal branches for each
input to leverage its information. Our mutual information based sample
acquisition strategy Single-Modal Entropic Measure (SMEM) in addition to our
self-distillation technique enables the sample acquisitor to exploit all
present modalities and find the most informative samples. Our novel idea is
simple to implement, cost-efficient, and readily adaptable to other multi-modal
tasks. We confirm our findings on various VQA datasets through state-of-the-art
performance by comparing to existing Active Learning baselines.

    

### [[2111.01080] ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack](http://arxiv.org/abs/2111.01080)


  In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA)
that precisely destroys deep neural networks (DNNs) by synthesizing its own
attack datasets. Many prior works on adversarial weight attack require not only
the weight parameters, but also the training or test dataset in searching
vulnerable bits to be attacked. We propose to synthesize the attack dataset,
named distilled target data, by utilizing the statistics of batch normalization
layers in the victim DNN model. Equipped with the distilled target data, our
ZeBRA algorithm can search vulnerable bits in the model without accessing
training or test dataset. Thus, our approach makes the adversarial weight
attack more fatal to the security of DNNs. Our experimental results show that
2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on
average to destroy DNNs compared to the previous attack method. Our code is
available at https://github. com/pdh930105/ZeBRA.

    

### [[2111.04474] Weapon Engagement Zone Maximum Launch Range Estimation Using a Deep Neural Network](http://arxiv.org/abs/2111.04474)


  This work investigates the use of a Deep Neural Network (DNN) to perform an
estimation of the Weapon Engagement Zone (WEZ) maximum launch range. The WEZ
allows the pilot to identify an airspace in which the available missile has a
more significant probability of successfully engaging a particular target,
i.e., a hypothetical area surrounding an aircraft in which an adversary is
vulnerable to a shot. We propose an approach to determine the WEZ of a given
missile using 50,000 simulated launches in variate conditions. These
simulations are used to train a DNN that can predict the WEZ when the aircraft
finds itself on different firing conditions, with a coefficient of
determination of 0.99. It provides another procedure concerning preceding
research since it employs a non-discretized model, i.e., it considers all
directions of the WEZ at once, which has not been done previously.
Additionally, the proposed method uses an experimental design that allows for
fewer simulation runs, providing faster model training.

    

### [[2012.08580] PANTHER: Pathway Augmented Nonnegative Tensor factorization for HighER-order feature learning](http://arxiv.org/abs/2012.08580)


  Genetic pathways usually encode molecular mechanisms that can inform targeted
interventions. It is often challenging for existing machine learning approaches
to jointly model genetic pathways (higher-order features) and variants (atomic
features), and present to clinicians interpretable models. In order to build
more accurate and better interpretable machine learning models for genetic
medicine, we introduce Pathway Augmented Nonnegative Tensor factorization for
HighER-order feature learning (PANTHER). PANTHER selects informative genetic
pathways that directly encode molecular mechanisms. We apply genetically
motivated constrained tensor factorization to group pathways in a way that
reflects molecular mechanism interactions. We then train a softmax classifier
for disease types using the identified pathway groups. We evaluated PANTHER
against multiple state-of-the-art constrained tensor/matrix factorization
models, as well as group guided and Bayesian hierarchical models. PANTHER
outperforms all state-of-the-art comparison models significantly (p<0.05). Our
experiments on large scale Next Generation Sequencing (NGS) and whole-genome
genotyping datasets also demonstrated wide applicability of PANTHER. We
performed feature analysis in predicting disease types, which suggested
insights and benefits of the identified pathway groups.

    

### [[2111.09747] Hamming Distance Tolerant Content-Addressable Memory (HD-CAM) for Approximate Matching Applications](http://arxiv.org/abs/2111.09747)


  We propose a novel Hamming distance tolerant content-addressable memory
(HD-CAM) for energy-efficient in memory approximate matching applications.
HD-CAM implements approximate search using matchline charge redistribution
rather than its rise or fall time, frequently employed in state of-the-art
solutions. HD-CAM was designed in a 65 nm 1.2 V CMOS technology and evaluated
through extensive Monte Carlo simulations. Our analysis shows that HD-CAM
supports robust operation under significant process variations and changes in
the design parameters, enabling a wide range of mismatch threshold (tolerable
Hamming distance) levels and pattern lengths. HD-CAM was functionally evaluated
for virus DNA classification, which makes HD-CAM suitable for hardware
acceleration of genomic surveillance of viral outbreaks such as Covid-19
pandemics.

    

### [[2111.09353] Case study of SARS-CoV-2 transmission risk assessment in indoor environments using cloud computing resources](http://arxiv.org/abs/2111.09353)


  Complex flow simulations are conventionally performed on HPC clusters.
However, the limited availability of HPC resources and steep learning curve of
executing on traditional supercomputer infrastructure has drawn attention
towards deploying flow simulation software on the cloud. We showcase how a
complex computational framework -- that can evaluate COVID-19 transmission risk
in various indoor classroom scenarios -- can be abstracted and deployed on
cloud services. The availability of such cloud-based personalized planning
tools can enable educational institutions, medical institutions, public sector
workers (courthouses, police stations, airports, etc.), and other entities to
comprehensively evaluate various in-person interaction scenarios for
transmission risk. We deploy the simulation framework on the Azure cloud
framework, utilizing the Dendro-kT mesh generation tool and PETSc solvers. The
cloud abstraction is provided by RocketML cloud infrastructure. We compare the
performance of the cloud machines with state-of-the-art HPC machine TACC
Frontera. Our results suggest that cloud-based HPC resources are a viable
strategy for a diverse array of end-users to rapidly and efficiently deploy
simulation software.

    

### [[2111.09449] Local Mutual Exclusion for Dynamic, Anonymous, Bounded Memory Message Passing Systems](http://arxiv.org/abs/2111.09449)


  Mutual exclusion is a classical problem in distributed computing that
provides isolation among concurrent action executions that may require access
to the same shared resources. Inspired by algorithmic research on distributed
systems of weakly capable entities whose connections change over time, we
address the local mutual exclusion problem that tasks each node with acquiring
exclusive locks for itself and the maximal subset of its "persistent" neighbors
that remain connected to it over the time interval of the lock request. Using
the established time-varying graphs model to capture adversarial topological
changes, we propose and rigorously analyze a local mutual exclusion algorithm
for nodes that are anonymous and communicate via asynchronous message passing.
The algorithm satisfies mutual exclusion (non-intersecting lock sets) and
lockout freedom (eventual success) under both semi-synchronous and asynchronous
concurrency. It requires $\mathcal{O}(\Delta\log\Delta)$ memory per node and
messages of size $\mathcal{O}(\log\Delta)$, where $\Delta$ is the maximum
number of connections per node. For systems of weak entities, $\Delta$ is often
a small constant, reducing the memory and message size requirements to
$\mathcal{O}(1)$. We conclude by describing how our algorithm can be used to
implement the schedulers assumed by population protocols and the concurrency
control operations assumed by the canonical amoebot model, demonstrating its
utility in both passively and actively dynamic distributed systems.

    

### [[2111.09547] QGTC: Accelerating Quantized GNN via GPU Tensor Core](http://arxiv.org/abs/2111.09547)


  Over the most recent years, quantized graph neural network (QGNN) attracts
lots of research and industry attention due to its high robustness and low
computation and memory overhead. Unfortunately, the performance gains of QGNN
have never been realized on modern GPU platforms. To this end, we propose the
first Tensor Core (TC) based computing framework, QGTC, to support any-bitwidth
computation for QGNNs on GPUs. We introduce a novel quantized low-bit
arithmetic design based on the low-bit data representation and bit-decomposed
computation. We craft a novel TC-tailored CUDA kernel design by incorporating
3D-stacked bit compression, zero-tile jumping, and non-zero tile reuse
technique to improve the performance systematically. We incorporate an
effective bandwidth-optimized subgraph packing strategy to maximize the
transferring efficiency between CPU host and GPU device. We integrate QGTC with
Pytorch for better programmability and extensibility. Extensive experiments
demonstrate that QGTC achieves an average of 3.17x speedup compared with the
state-of-the-art Deep Graph Library framework across diverse settings.

    

### [[2111.09562] COMET: A Novel Memory-Efficient Deep Learning Training Framework by Using Error-Bounded Lossy Compression](http://arxiv.org/abs/2111.09562)


  Training wide and deep neural networks (DNNs) require large amounts of
storage resources such as memory because the intermediate activation data must
be saved in the memory during forward propagation and then restored for
backward propagation. However, state-of-the-art accelerators such as GPUs are
only equipped with very limited memory capacities due to hardware design
constraints, which significantly limits the maximum batch size and hence
performance speedup when training large-scale DNNs. Traditional memory saving
techniques either suffer from performance overhead or are constrained by
limited interconnect bandwidth or specific interconnect technology. In this
paper, we propose a novel memory-efficient CNN training framework (called
COMET) that leverages error-bounded lossy compression to significantly reduce
the memory requirement for training, to allow training larger models or to
accelerate training. Different from the state-of-the-art solutions that adopt
image-based lossy compressors (such as JPEG) to compress the activation data,
our framework purposely adopts error-bounded lossy compression with a strict
error-controlling mechanism. Specifically, we perform a theoretical analysis on
the compression error propagation from the altered activation data to the
gradients, and empirically investigate the impact of altered gradients over the
training process. Based on these analyses, we optimize the error-bounded lossy
compression and propose an adaptive error-bound control scheme for activation
data compression. We evaluate our design against state-of-the-art solutions
with five widely-adopted CNNs and ImageNet dataset. Experiments demonstrate
that our proposed framework can significantly reduce the training memory
consumption by up to 13.5X over the baseline training and 1.8X over another
state-of-the-art compression-based framework, respectively, with little or no
accuracy loss.

    

### [[2111.09815] Improving Prediction-Based Lossy Compression Dramatically Via Ratio-Quality Modeling](http://arxiv.org/abs/2111.09815)


  Error-bounded lossy compression is one of the most effective techniques for
scientific data reduction. However, the traditional trial-and-error approach
used to configure lossy compressors for finding the optimal trade-off between
reconstructed data quality and compression ratio is prohibitively expensive. To
resolve this issue, we develop a general-purpose analytical ratio-quality model
based on the prediction-based lossy compression framework, which can
effectively foresee the reduced data quality and compression ratio, as well as
the impact of the lossy compressed data on post-hoc analysis quality. Our
analytical model significantly improves the prediction-based lossy compression
in three use-cases: (1) optimization of predictor by selecting the best-fit
predictor; (2) memory compression with a target ratio; and (3) in-situ
compression optimization by fine-grained error-bound tuning of various data
partitions. We evaluate our analytical model on 10 scientific datasets,
demonstrating its high accuracy (93.47% accuracy on average) and low
computational cost (up to 18.7X lower than the trial-and-error approach) for
estimating the compression ratio and the impact of lossy compression on
post-hoc analysis quality. We also verified the high efficiency of our
ratio-quality model using different applications across the three use-cases. In
addition, the experiment demonstrates that our modeling based approach reduces
the time to store the 3D Reverse Time Migration data by up to 3.4X over the
traditional solution using 128 CPU cores from 8 compute nodes.

    

### [[2111.09461] Advancing COVID-19 Diagnosis with Privacy-Preserving Collaboration in Artificial Intelligence](http://arxiv.org/abs/2111.09461)


  Artificial intelligence (AI) provides a promising substitution for
streamlining COVID-19 diagnoses. However, concerns surrounding security and
trustworthiness impede the collection of large-scale representative medical
data, posing a considerable challenge for training a well-generalised model in
clinical practices. To address this, we launch the Unified CT-COVID AI
Diagnostic Initiative (UCADI), where the AI model can be distributedly trained
and independently executed at each host institution under a federated learning
framework (FL) without data sharing. Here we show that our FL model
outperformed all the local models by a large yield (test sensitivity
/specificity in China: 0.973/0.951, in the UK: 0.730/0.942), achieving
comparable performance with a panel of professional radiologists. We further
evaluated the model on the hold-out (collected from another two hospitals
leaving out the FL) and heterogeneous (acquired with contrast materials) data,
provided visual explanations for decisions made by the model, and analysed the
trade-offs between the model performance and the communication costs in the
federated training process. Our study is based on 9,573 chest computed
tomography scans (CTs) from 3,336 patients collected from 23 hospitals located
in China and the UK. Collectively, our work advanced the prospects of utilising
federated learning for privacy-preserving AI in digital health.

    

### [[2111.09475] Lifelong Reinforcement Learning with Temporal Logic Formulas and Reward Machines](http://arxiv.org/abs/2111.09475)


  Continuously learning new tasks using high-level ideas or knowledge is a key
capability of humans. In this paper, we propose Lifelong reinforcement learning
with Sequential linear temporal logic formulas and Reward Machines (LSRM),
which enables an agent to leverage previously learned knowledge to fasten
learning of logically specified tasks. For the sake of more flexible
specification of tasks, we first introduce Sequential Linear Temporal Logic
(SLTL), which is a supplement to the existing Linear Temporal Logic (LTL)
formal language. We then utilize Reward Machines (RM) to exploit structural
reward functions for tasks encoded with high-level events, and propose
automatic extension of RM and efficient knowledge transfer over tasks for
continuous learning in lifetime. Experimental results show that LSRM
outperforms the methods that learn the target tasks from scratch by taking
advantage of the task decomposition using SLTL and knowledge transfer over RM
during the lifelong learning process.

    

### [[2111.09478] Software Engineering for Responsible AI: An Empirical Study and Operationalised Patterns](http://arxiv.org/abs/2111.09478)


  Although artificial intelligence (AI) is solving real-world challenges and
transforming industries, there are serious concerns about its ability to behave
and make decisions in a responsible way. Many AI ethics principles and
guidelines for responsible AI have been recently issued by governments,
organisations, and enterprises. However, these AI ethics principles and
guidelines are typically high-level and do not provide concrete guidance on how
to design and develop responsible AI systems. To address this shortcoming, we
first present an empirical study where we interviewed 21 scientists and
engineers to understand the practitioners' perceptions on AI ethics principles
and their implementation. We then propose a template that enables AI ethics
principles to be operationalised in the form of concrete patterns and suggest a
list of patterns using the newly created template. These patterns provide
concrete, operationalised guidance that facilitate the development of
responsible AI systems.

    

### [[2111.09618] To Augment or Not to Augment? A Comparative Study on Text Augmentation Techniques for Low-Resource NLP](http://arxiv.org/abs/2111.09618)


  Data-hungry deep neural networks have established themselves as the standard
for many NLP tasks including the traditional sequence tagging ones. Despite
their state-of-the-art performance on high-resource languages, they still fall
behind of their statistical counter-parts in low-resource scenarios. One
methodology to counter attack this problem is text augmentation, i.e.,
generating new synthetic training data points from existing data. Although NLP
has recently witnessed a load of textual augmentation techniques, the field
still lacks a systematic performance analysis on a diverse set of languages and
sequence tagging tasks. To fill this gap, we investigate three categories of
text augmentation methodologies which perform changes on the syntax (e.g.,
cropping sub-sentences), token (e.g., random word insertion) and character
(e.g., character swapping) levels. We systematically compare them on
part-of-speech tagging, dependency parsing and semantic role labeling for a
diverse set of language families using various models including the
architectures that rely on pretrained multilingual contextualized language
models such as mBERT. Augmentation most significantly improves dependency
parsing, followed by part-of-speech tagging and semantic role labeling. We find
the experimented techniques to be effective on morphologically rich languages
in general rather than analytic languages such as Vietnamese. Our results
suggest that the augmentation techniques can further improve over strong
baselines based on mBERT. We identify the character-level methods as the most
consistent performers, while synonym replacement and syntactic augmenters
provide inconsistent improvements. Finally, we discuss that the results most
heavily depend on the task, language pair, and the model type.

    

### [[2111.09701] Visual design intuition: Predicting dynamic properties of beams from raw cross-section images](http://arxiv.org/abs/2111.09701)


  In this work we aim to mimic the human ability to acquire the intuition to
estimate the performance of a design from visual inspection and experience
alone. We study the ability of convolutional neural networks to predict static
and dynamic properties of cantilever beams directly from their raw
cross-section images. Using pixels as the only input, the resulting models
learn to predict beam properties such as volume maximum deflection and
eigenfrequencies with 4.54% and 1.43% Mean Average Percentage Error (MAPE)
respectively, compared to the Finite Element Analysis (FEA) approach. Training
these models doesn't require prior knowledge of theory or relevant geometric
properties, but rather relies solely on simulated or empirical data, thereby
making predictions based on "experience" as opposed to theoretical knowledge.
Since this approach is over 1000 times faster than FEA, it can be adopted to
create surrogate models that could speed up the preliminary optimization
studies where numerous consecutive evaluations of similar geometries are
required. We suggest that this modeling approach would aid in addressing
challenging optimization problems involving complex structures and physical
phenomena for which theoretical models are unavailable.

    

### [[2111.09739] Learning Ultrasound Scanning Skills from Human Demonstrations](http://arxiv.org/abs/2111.09739)


  Recently, the robotic ultrasound system has become an emerging topic owing to
the widespread use of medical ultrasound. However, it is still a challenging
task to model and to transfer the ultrasound skill from an ultrasound
physician. In this paper, we propose a learning-based framework to acquire
ultrasound scanning skills from human demonstrations. First, the ultrasound
scanning skills are encapsulated into a high-dimensional multi-modal model in
terms of interactions among ultrasound images, the probe pose and the contact
force. The parameters of the model are learned using the data collected from
skilled sonographers' demonstrations. Second, a sampling-based strategy is
proposed with the learned model to adjust the extracorporeal ultrasound
scanning process to guide a newbie sonographer or a robot arm. Finally, the
robustness of the proposed framework is validated with the experiments on real
data from sonographers.

    

### [[2111.09762] Hybrid Super Intelligence and Polymetric Analysis](http://arxiv.org/abs/2111.09762)


  The problem of possible applications Polymetric Analysis for the resolution
problems of artificial Intelligence is discussed. As example the hybrid super
intelligence system by N. Moiseev type was selected. The bond between
polymetric analysis and hybrid super intelligence system was shown. In
operational sense polymetric analysis is more general system. Therefore main
principles of Moiseev concept may be unify with the help of polymetric
analysis. Main peculiarities of this unification are analyzed.

    

### [[2111.09851] The Effects of Learning in Morphologically Evolving Robot Systems](http://arxiv.org/abs/2111.09851)


  Simultaneously evolving morphologies (bodies) and controllers (brains) of
robots can cause a mismatch between the inherited body and brain in the
offspring. To mitigate this problem, the addition of an infant learning period
by the so-called Triangle of Life framework has been proposed relatively long
ago. However, an empirical assessment is still lacking to-date. In this paper
we investigate the effects of such a learning mechanism from different
perspectives. Using extensive simulations we show that learning can greatly
increase task performance and reduce the number of generations required to
reach a certain fitness level compared to the purely evolutionary approach.
Furthermore, although learning only directly affects the controllers, we
demonstrate that the evolved morphologies will be also different. This provides
a quantitative demonstration that changes in the brain can induce changes in
the body. Finally, we examine the concept of morphological intelligence
quantified by the ability of a given body to learn. We observe that the
learning delta, the performance difference between the inherited and the
learned brain, is growing throughout the evolutionary process. This shows that
evolution is producing robots with an increasing plasticity, that is,
consecutive generations are becoming better and better learners which in turn
makes them better and better at the given task. All in all, our results
demonstrate that the Triangle of Life is not only a concept of theoretical
interest, but a system architecture with practical benefits.

    

### [[1808.08079] Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information](http://arxiv.org/abs/1808.08079)


  How do neural language models keep track of number agreement between subject
and verb? We show that `diagnostic classifiers', trained to predict number from
the internal states of a language model, provide a detailed understanding of
how, when, and where this information is represented. Moreover, they give us
insight into when and where number information is corrupted in cases where the
language model ends up making agreement errors. To demonstrate the causal role
played by the representations we find, we then use agreement information to
influence the course of the LSTM during the processing of difficult sentences.
Results from such an intervention reveal a large increase in the language
model's accuracy. Together, these results show that diagnostic classifiers give
us an unrivalled detailed look into the representation of linguistic
information in neural models, and demonstrate that this knowledge can be used
to improve their performance.

    

### [[2012.09542] Weakly-Supervised Action Localization and Action Recognition using Global-Local Attention of 3D CNN](http://arxiv.org/abs/2012.09542)


  3D Convolutional Neural Network (3D CNN) captures spatial and temporal
information on 3D data such as video sequences. However, due to the convolution
and pooling mechanism, the information loss seems unavoidable. To improve the
visual explanations and classification in 3D CNN, we propose two approaches; i)
aggregate layer-wise global to local (global-local) discrete gradients using
trained 3DResNext network, and ii) implement attention gating network to
improve the accuracy of the action recognition. The proposed approach intends
to show the usefulness of every layer termed as global-local attention in 3D
CNN via visual attribution, weakly-supervised action localization, and action
recognition. Firstly, the 3DResNext is trained and applied for action
classification using backpropagation concerning the maximum predicted class.
The gradients and activations of every layer are then up-sampled. Later,
aggregation is used to produce more nuanced attention, which points out the
most critical part of the predicted class's input videos. We use contour
thresholding of final attention for final localization. We evaluate spatial and
temporal action localization in trimmed videos using fine-grained visual
explanation via 3DCam. Experimental results show that the proposed approach
produces informative visual explanations and discriminative attention.
Furthermore, the action recognition via attention gating on each layer produces
better classification results than the baseline model.

    

### [[2104.08826] GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation](http://arxiv.org/abs/2104.08826)


  Large-scale language models such as GPT-3 are excellent few-shot learners,
allowing them to be controlled via natural text prompts. Recent studies report
that prompt-based direct classification eliminates the need for fine-tuning but
lacks data and inference scalability. This paper proposes a novel data
augmentation technique that leverages large-scale language models to generate
realistic text samples from a mixture of real samples. We also propose
utilizing soft-labels predicted by the language models, effectively distilling
knowledge from the large-scale language models and creating textual
perturbations simultaneously. We perform data augmentation experiments on
diverse classification tasks and show that our method hugely outperforms
existing text augmentation methods. Ablation studies and a qualitative analysis
provide more insights into our approach.

    

### [[2111.09826] Analysis and Design of Distributed MIMO Networks with a Wireless Fronthaul](http://arxiv.org/abs/2111.09826)


  We consider the analysis and design of distributed wireless networks wherein
remote radio heads (RRHs) coordinate transmissions to serve multiple users on
the same resource block (RB). Specifically, we analyze two possible
multiple-input multiple-output wireless fronthaul solutions: multicast and zero
forcing (ZF) beamforming. We develop a statistical model for the fronthaul rate
and, coupled with an analysis of the user access rate, we optimize the
placement of the RRHs. This model allows us to formulate the location
optimization problem with a statistical constraint on fronthaul outage. Our
results are cautionary, showing that the fronthaul requires considerable
bandwidth to enable joint service to users. This requirement can be relaxed by
serving a low number of users on the same RB. Additionally, we show that, with
a fixed number of antennas, for the multicast fronthaul, it is prudent to
concentrate these antennas on a few RRHs. However, for the ZF beamforming
fronthaul, it is better to distribute the antennas on more RRHs. For the
parameters chosen, using a ZF beamforming fronthaul improves the typical access
rate by approximately 8% compared to multicast. Crucially, our work quantifies
the effect of these fronthaul solutions and provides an effective tool for the
design of distributed networks.

    

### [[2111.09728] Measuring source code conciseness across programming languages using compression](http://arxiv.org/abs/2111.09728)


  It is well-known, and often a topic of heated debates, that programs in some
programming languages are more concise than in others. This is a relevant
factor when comparing or aggregating volume-impacted metrics on source code
written in a combination of programming languages. In this paper, we present a
model for measuring the conciseness of programming languages in a consistent,
objective and evidence-based way. We present the approach, explain how it is
founded on information theoretical principles, present detailed analysis steps
and show the quantitative results of applying this model to a large benchmark
of diverse commercial software applications. We demonstrate that our metric for
language conciseness is strongly correlated with both an alternative analytical
approach, and with a large scale developer survey, and show how its results can
be applied to improve software metrics for multi-language applications.

    

### [[2111.09823] NetQASM -- A low-level instruction set architecture for hybrid quantum-classical programs in a quantum internet](http://arxiv.org/abs/2111.09823)


  We introduce NetQASM, a low-level instruction set architecture for quantum
internet applications. NetQASM is a universal, platform-independent and
extendable instruction set with support for local quantum gates, powerful
classical logic and quantum networking operations for remote entanglement
generation. Furthermore, NetQASM allows for close integration of classical
logic and communication at the application layer with quantum operations at the
physical layer. We implement NetQASM in a series of tools to write, parse,
encode and run NetQASM code, which are available online. Our tools include a
higher-level SDK in Python, which allows an easy way of programming
applications for a quantum internet. Our SDK can be used at home by making use
of our existing quantum simulators, NetSquid and SimulaQron, and will also
provide a public interface to hardware released on a future iteration of
Quantum Network Explorer.

    

### [[2012.12143] Introducing CPL](http://arxiv.org/abs/2012.12143)


  CPL here stands for a computer programming language conceived and developed
by the author since 1993, but published for the first time in 2020. It was born
as a Compiled Programming Language, designed together with its compiler and
therefore suitable for computationally intensive numerical applications,
although some years later an interpreter was also provided for interactive
usage. CPL's distinctive features are Concealed Pointer Lookup, the ability to
implicitly dereference pointers based on the type of operands involved,
Consistent Procedure Linkage, the enforcement of function prototypes without
dedicated header or interface files, and Coactive Parameter Lists, the ability
to overload function names which are then distinguished by the type of their
parameters and/or parameter separators. Perhaps even more distinctly, CPL's
syntax can be extended on the fly by the program being compiled; library
modules tap this feature to seamlessly add real and complex matrix operations,
graphics, parallel-computing extensions, and symbolic differentiation. The CPL
coding software is available for free download at this http URL .

    