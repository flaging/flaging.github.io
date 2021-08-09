
## 2021-8-9

### [[2108.02935] Implementasi dan Analisis Performa Bonding Interface Mode 802.3ad sebagai Link Redundancy pada Router Mikrotik](http://arxiv.org/abs/2108.02935)


  Network stability and reliability is an absolute requirement for
telecommunications networks. Bonding interface is a technique that can
facilitate the network to serve services that require network stability and
reliability. This interface bonding system combines two interfaces into a
virtual link that is characterized by the use of one IP address. If a frame is
sent from the sender to the receiver, but in the process of sending one of the
links there is an interruption, then the link that is still connected will be
able to maintain the connection so that frame transmission continues, this
working system is called link redundancy. To monitor the condition of the link
whether it is connected or broken, a mechanism called link-monitoring is used.
The monitoring link used in this final project is Media Independent Interface
(MII). The test step is to implement a Bonding Interface network for the use of
video streaming, VoIP, and file transfer. The analysis carried out is a
failover test that uses a connection switching time parameter and a QoS test
with packet loss, delay, jitter, and throughput parameters. The results will
then be compared with a network that does not use a Bonding Interface, which is
a network that is only connected using one link. From the results of the
research conducted is that the Bonding Interface system is able to perform a
redundancy mechanism when one of the links is broken/down to an
active/connected link and the services that are run are not interrupted.
Bonding Interface is also able to provide network stability compared to
networks that do not use a Bonding Interface, this is characterized by a
smaller jitter value than the use of a network with one link.

    

### [[2108.03035] Performance trade-offs in cyber-physical control applications with multi-connectivity](http://arxiv.org/abs/2108.03035)


  Modern communication devices are often equipped with multiple wireless
communication interfaces with diverse characteristics. This enables exploiting
a form of multi-connectivity known as interface diversity to provide path
diversity with multiple communication interfaces. Interface diversity helps to
combat the problems suffered by single-interface systems due to error bursts in
the link, which are a consequence of temporal correlation in the wireless
channel. The length of an error burst is an essential performance indicator for
cyber-physical control applications with periodic traffic, as these define the
period in which the control link is unavailable. However, the available
interfaces must be correctly orchestrated to achieve an adequate trade-off
between latency, reliability, and energy consumption. This work investigates
how the packet error statistics from different interfaces impacts the overall
latency-reliability characteristics and explores mechanisms to derive adequate
interface diversity policies. For this, we model the optimization problem as a
partially observable Markov Decision Process (POMDP), where the state of each
interface is determined by a Gilbert-Elliott model whose parameters are
estimated based on experimental measurement traces from LTE and Wi-Fi. Our
results show that the POMDP approach provides an all-round adaptable solution,
whose performance is only 0.1% below the absolute upper bound, dictated by the
optimal policy under the impractical assumption of full observability.

    

### [[2108.03122] Computation and Communication Co-Design for Real-Time Monitoring and Control in Multi-Agent Systems](http://arxiv.org/abs/2108.03122)


  We investigate the problem of co-designing computation and communication in a
multi-agent system (e.g. a sensor network or a multi-robot team). We consider
the realistic setting where each agent acquires sensor data and is capable of
local processing before sending updates to a base station, which is in charge
of making decisions or monitoring phenomena of interest in real time. Longer
processing at an agent leads to more informative updates but also larger
delays, giving rise to a delay-accuracy-tradeoff in choosing the right amount
of local processing at each agent. We assume that the available communication
resources are limited due to interference, bandwidth, and power constraints.
Thus, a scheduling policy needs to be designed to suitably share the
communication channel among the agents. To that end, we develop a general
formulation to jointly optimize the local processing at the agents and the
scheduling of transmissions. Our novel formulation leverages the notion of Age
of Information to quantify the freshness of data and capture the delays caused
by computation and communication. We develop efficient resource allocation
algorithms using the Whittle index approach and demonstrate our proposed
algorithms in two practical applications: multi-agent occupancy grid mapping in
time-varying environments, and ride sharing in autonomous vehicle networks. Our
experiments show that the proposed co-design approach leads to a substantial
performance improvement (18-82% in our tests).

    

### [[2108.03221] FloMore: Meeting bandwidth requirements of flows](http://arxiv.org/abs/2108.03221)


  Wide-area cloud provider networks must support the bandwidth requirements of
diverse services (e.g., applications, product groups, customers) despite
failures. Existing traffic engineering (TE) schemes operate at much coarser
granularity than services, which we show necessitates unduly conservative
decisions. To tackle this, we present FloMore, which directly considers the
bandwidth needs of individual services and ensures they are met a desired
percentage of time. Rather than meet the requirements for all services over the
same set of failure states, FloMore exploits a key opportunity that each
service could meet its bandwidth requirements over a different set of failure
states. FloMore consists of an offline phase that identifies the critical
failure states of each service, and on failure allocates traffic in a manner
that prioritizes those services for which that failure state is critical. We
present a novel decomposition scheme to handle FloMore's offline phase in a
tractable manner. Our evaluations show that FloMore outperforms
state-of-the-art TE schemes including SMORE and Teavar, and also out-performs
extensions of these schemes that we devise. The results also show FloMore's
decomposition approach allows it to scale well to larger network topologies.

    

### [[1808.00794] Analysis of the Threshold for Energy Consumption in Displacement of Random Sensors](http://arxiv.org/abs/1808.00794)


  The fundamental problem of energy-efficient reallocation of mobile random
sensors to provide full coverage without interference is addressed in this
paper. We consider $n$ mobile sensors with identical sensing range placed
randomly on the unit interval and on the unit square. The main contribution is
summarized as follows:
If the sensors are placed on the unit interval we explain the sharp increase
around the sensing radius equal to $\frac{1}{2n}$ and the interference distance
equal to $\frac{1}{n}$ for the expected minimal $a$-total displacement,
If the sensors are placed on the unit square we explain the sharp increase
around the square sensing radius equal to $\frac{1}{2 \sqrt{n}}$ and the
interference distance equal to $\frac{1}{\sqrt{n}}$ for the expected minimal
$a$-total displacement.

    

### [[1912.01285] A Configurable Mathematical Model for Single-Gateway LoRaWAN Performance Analysis](http://arxiv.org/abs/1912.01285)


  LoRaWAN is a Low Power Wide Area Network technology featuring long
transmission ranges and a simple MAC layer, which can support sensor data
collection, control applications and reliable services thanks to the
flexibility offered by a large set of configurable system parameters. However,
the impact of such parameters settings on the system's performance is often
difficult to predict, depending on several factors. To ease this task, in this
paper, we provide a mathematical model to estimate the performance of a LoRaWAN
gateway serving a set of devices that may or may not employ confirmed traffic.
The model features a set of parameters that can be adjusted to investigate
different gateway and end-device configurations, making it possible to carry
out a systematic analysis of various trade-offs. The results given by the
proposed model are validated through realistic ns-3 simulations that confirm
the ability of the model to predict the system performance with high accuracy,
and assess the impact of the assumptions made in the model for tractability.

    

### [[2108.02786] Quantum Continual Learning Overcoming Catastrophic Forgetting](http://arxiv.org/abs/2108.02786)


  Catastrophic forgetting describes the fact that machine learning models will
likely forget the knowledge of previously learned tasks after the learning
process of a new one. It is a vital problem in the continual learning scenario
and recently has attracted tremendous concern across different communities. In
this paper, we explore the catastrophic forgetting phenomena in the context of
quantum machine learning. We find that, similar to those classical learning
models based on neural networks, quantum learning systems likewise suffer from
such forgetting problem in classification tasks emerging from various
application scenes. We show that based on the local geometrical information in
the loss function landscape of the trained model, a uniform strategy can be
adapted to overcome the forgetting problem in the incremental learning setting.
Our results uncover the catastrophic forgetting phenomena in quantum machine
learning and offer a practical method to overcome this problem, which opens a
new avenue for exploring potential quantum advantages towards continual
learning.

    

### [[2108.02798] Self-Supervised Learning from Unlabeled Fundus Photographs Improves Segmentation of the Retina](http://arxiv.org/abs/2108.02798)


  Fundus photography is the primary method for retinal imaging and essential
for diabetic retinopathy prevention. Automated segmentation of fundus
photographs would improve the quality, capacity, and cost-effectiveness of eye
care screening programs. However, current segmentation methods are not robust
towards the diversity in imaging conditions and pathologies typical for
real-world clinical applications. To overcome these limitations, we utilized
contrastive self-supervised learning to exploit the large variety of unlabeled
fundus images in the publicly available EyePACS dataset. We pre-trained an
encoder of a U-Net, which we later fine-tuned on several retinal vessel and
lesion segmentation datasets. We demonstrate for the first time that by using
contrastive self-supervised learning, the pre-trained network can recognize
blood vessels, optic disc, fovea, and various lesions without being provided
any labels. Furthermore, when fine-tuned on a downstream blood vessel
segmentation task, such pre-trained networks achieve state-of-the-art
performance on images from different datasets. Additionally, the pre-training
also leads to shorter training times and an improved few-shot performance on
both blood vessel and lesion segmentation tasks. Altogether, our results
showcase the benefits of contrastive self-supervised pre-training which can
play a crucial role in real-world clinical applications requiring robust models
able to adapt to new devices with only a few annotated samples.

    

### [[2108.02799] Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends](http://arxiv.org/abs/2108.02799)


  League of Legends (LoL) is the most widely played multiplayer online battle
arena (MOBA) game in the world. An important aspect of LoL is competitive
ranked play, which utilizes a skill-based matchmaking system to form fair
teams. However, players' skill levels vary widely depending on which champion,
or hero, that they choose to play as. In this paper, we propose a method for
predicting game outcomes in ranked LoL games based on players' experience with
their selected champion. Using a deep neural network, we found that game
outcomes can be predicted with 75.1% accuracy after all players have selected
champions, which occurs before gameplay begins. Our results have important
implications for playing LoL and matchmaking. Firstly, individual champion
skill plays a significant role in the outcome of a match, regardless of team
composition. Secondly, even after the skill-based matchmaking, there is still a
wide variance in team skill before gameplay begins. Finally, players should
only play champions that they have mastered, if they want to win games.

    

### [[2108.02811] Quantum Topological Data Analysis with Linear Depth and Exponential Speedup](http://arxiv.org/abs/2108.02811)


  Quantum computing offers the potential of exponential speedups for certain
classical computations. Over the last decade, many quantum machine learning
(QML) algorithms have been proposed as candidates for such exponential
improvements. However, two issues unravel the hope of exponential speedup for
some of these QML algorithms: the data-loading problem and, more recently, the
stunning dequantization results of Tang et al. A third issue, namely the
fault-tolerance requirements of most QML algorithms, has further hindered their
practical realization. The quantum topological data analysis (QTDA) algorithm
of Lloyd, Garnerone and Zanardi was one of the first QML algorithms that
convincingly offered an expected exponential speedup. From the outset, it did
not suffer from the data-loading problem. A recent result has also shown that
the generalized problem solved by this algorithm is likely classically
intractable, and would therefore be immune to any dequantization efforts.
However, the QTDA algorithm of Lloyd et~al. has a time complexity of
$O(n^4/(\epsilon^2 \delta))$ (where $n$ is the number of data points,
$\epsilon$ is the error tolerance, and $\delta$ is the smallest nonzero
eigenvalue of the restricted Laplacian) and requires fault-tolerant quantum
computing, which has not yet been achieved. In this paper, we completely
overhaul the QTDA algorithm to achieve an improved exponential speedup and
depth complexity of $O(n\log(1/(\delta\epsilon)))$. Our approach includes three
key innovations: (a) an efficient realization of the combinatorial Laplacian as
a sum of Pauli operators; (b) a quantum rejection sampling approach to restrict
the superposition to the simplices in the complex; and (c) a stochastic rank
estimation method to estimate the Betti numbers. We present a theoretical error
analysis, and the circuit and computational time and depth complexities for
Betti number estimation.

    

### [[2108.02814] Potential Applications of Artificial Intelligence and Machine Learning in Radiochemistry and Radiochemical Engineering](http://arxiv.org/abs/2108.02814)


  Artificial intelligence and machine learning are poised to disrupt PET
imaging from bench to clinic. In this perspective we offer insights into how
the technology could be applied to improve the design and synthesis of new
radiopharmaceuticals for PET imaging, including identification of an optimal
labeling approach as well as strategies for radiolabeling reaction
optimization.

    

### [[2108.02817] THALIS: Human-Machine Analysis of Longitudinal Symptoms in Cancer Therapy](http://arxiv.org/abs/2108.02817)


  Although cancer patients survive years after oncologic therapy, they are
plagued with long-lasting or permanent residual symptoms, whose severity, rate
of development, and resolution after treatment vary largely between survivors.
The analysis and interpretation of symptoms is complicated by their partial
co-occurrence, variability across populations and across time, and, in the case
of cancers that use radiotherapy, by further symptom dependency on the tumor
location and prescribed treatment. We describe THALIS, an environment for
visual analysis and knowledge discovery from cancer therapy symptom data,
developed in close collaboration with oncology experts. Our approach leverages
unsupervised machine learning methodology over cohorts of patients, and, in
conjunction with custom visual encodings and interactions, provides context for
new patients based on patients with similar diagnostic features and symptom
evolution. We evaluate this approach on data collected from a cohort of head
and neck cancer patients. Feedback from our clinician collaborators indicates
that THALIS supports knowledge discovery beyond the limits of machines or
humans alone, and that it serves as a valuable tool in both the clinic and
symptom research.

    

### [[2108.02827] An Elementary Proof that Q-learning Converges Almost Surely](http://arxiv.org/abs/2108.02827)


  Watkins' and Dayan's Q-learning is a model-free reinforcement learning
algorithm that iteratively refines an estimate for the optimal action-value
function of an MDP by stochastically "visiting" many state-ation pairs [Watkins
and Dayan, 1992]. Variants of the algorithm lie at the heart of numerous recent
state-of-the-art achievements in reinforcement learning, including the
superhuman Atari-playing deep Q-network [Mnih et al., 2015]. The goal of this
paper is to reproduce a precise and (nearly) self-contained proof that
Q-learning converges. Much of the available literature leverages powerful
theory to obtain highly generalizable results in this vein. However, this
approach requires the reader to be familiar with and make many deep connections
to different research areas. A student seeking to deepen their understand of
Q-learning risks becoming caught in a vicious cycle of "RL-learning Hell". For
this reason, we give a complete proof from start to finish using only one
external result from the field of stochastic approximation, despite the fact
that this minimal dependence on other results comes at the expense of some
"shininess".

    

### [[2108.02830] Hate Speech Detection in Roman Urdu](http://arxiv.org/abs/2108.02830)


  Hate speech is a specific type of controversial content that is widely
legislated as a crime that must be identified and blocked. However, due to the
sheer volume and velocity of the Twitter data stream, hate speech detection
cannot be performed manually. To address this issue, several studies have been
conducted for hate speech detection in European languages, whereas little
attention has been paid to low-resource South Asian languages, making the
social media vulnerable for millions of users. In particular, to the best of
our knowledge, no study has been conducted for hate speech detection in Roman
Urdu text, which is widely used in the sub-continent. In this study, we have
scrapped more than 90,000 tweets and manually parsed them to identify 5,000
Roman Urdu tweets. Subsequently, we have employed an iterative approach to
develop guidelines and used them for generating the Hate Speech Roman Urdu 2020
corpus. The tweets in the this corpus are classified at three levels:
Neutral-Hostile, Simple-Complex, and Offensive-Hate speech. As another
contribution, we have used five supervised learning techniques, including a
deep learning technique, to evaluate and compare their effectiveness for hate
speech detection. The results show that Logistic Regression outperformed all
other techniques, including deep learning techniques for the two levels of
classification, by achieved an F1 score of 0.906 for distinguishing between
Neutral-Hostile tweets, and 0.756 for distinguishing between Offensive-Hate
speech tweets.

    

### [[2108.02831] Differentially Private n-gram Extraction](http://arxiv.org/abs/2108.02831)


  We revisit the problem of $n$-gram extraction in the differential privacy
setting. In this problem, given a corpus of private text data, the goal is to
release as many $n$-grams as possible while preserving user level privacy.
Extracting $n$-grams is a fundamental subroutine in many NLP applications such
as sentence completion, response generation for emails etc. The problem also
arises in other applications such as sequence mining, and is a generalization
of recently studied differentially private set union (DPSU). In this paper, we
develop a new differentially private algorithm for this problem which, in our
experiments, significantly outperforms the state-of-the-art. Our improvements
stem from combining recent advances in DPSU, privacy accounting, and new
heuristics for pruning in the tree-based approach initiated by Chen et al.
(2012).

    

### [[2108.02834] Efficient recurrent neural network methods for anomalously diffusing single particle short and noisy trajectories](http://arxiv.org/abs/2108.02834)


  Anomalous diffusion occurs at very different scales in nature, from atomic
systems to motions in cell organelles, biological tissues or ecology, and also
in artificial materials, such as cement. Being able to accurately measure the
anomalous exponent associated with a given particle trajectory, thus
determining whether the particle subdiffuses, superdiffuses or performs normal
diffusion is of key importance to understand the diffusion process. Also, it is
often important to trustingly identify the model behind the trajectory, as this
gives a large amount of information on the system dynamics. Both aspects are
particularly difficult when the input data are short and noisy trajectories. It
is even more difficult if one cannot guarantee that the trajectories output in
experiments is homogeneous, hindering the statistical methods based on
ensembles of trajectories. We present a data-driven method able to infer the
anomalous exponent and to identify the type of anomalous diffusion process
behind single, noisy and short trajectories, with good accuracy. This model was
used in our participation in the Anomalous Diffusion (AnDi) Challenge. A
combination of convolutional and recurrent neural networks were used to achieve
state-of-the-art results when compared to methods participating in the AnDi
Challenge, ranking top 4 in both classification and diffusion exponent
regression.

    

### [[2108.02837] Lossless Multi-Scale Constitutive Elastic Relations with Artificial Intelligence](http://arxiv.org/abs/2108.02837)


  The elastic properties of materials derive from their electronic and atomic
nature. However, simulating bulk materials fully at these scales is not
feasible, so that typically homogenized continuum descriptions are used
instead. A seamless and lossless transition of the constitutive description of
the elastic response of materials between these two scales has been so far
elusive. Here we show how this problem can be overcome by using Artificial
Intelligence (AI). A Convolutional Neural Network (CNN) model is trained, by
taking the structure image of a nanoporous material as input and the
corresponding elasticity tensor, calculated from Molecular Statics (MS), as
output. Trained with the atomistic data, the CNN model captures the size- and
pore-dependency of the material's elastic properties which, on the physics
side, can stem from surfaces and non-local effects. Such effects are often
ignored in upscaling from atomistic to classical continuum theory. To
demonstrate the accuracy and the efficiency of the trained CNN model, a Finite
Element Method (FEM) based result of an elastically deformed nanoporous beam
equipped with the CNN as constitutive law is compared with that by a full
atomistic simulation. The good agreement between the atomistic simulations and
the FEM-AI combination for a system with size and surface effects establishes a
new lossless scale bridging approach to such problems. The trained CNN model
deviates from the atomistic result by 9.6\% for porosity scenarios of up to
90\% but it is about 230 times faster than the MS calculation and does not
require to change simulation methods between different scales. The efficiency
of the CNN evaluation together with the preservation of important atomistic
effects makes the trained model an effective atomistically-informed
constitutive model for macroscopic simulations of nanoporous materials and
solving of inverse problems.

    

### [[2108.02838] Two-Stage Sector Rotation Methodology Using Machine Learning and Deep Learning Techniques](http://arxiv.org/abs/2108.02838)


  Market indicators such as CPI and GDP have been widely used over decades to
identify the stage of business cycles and also investment attractiveness of
sectors given market conditions. In this paper, we propose a two-stage
methodology that consists of predicting ETF prices for each sector using market
indicators and ranking sectors based on their predicted rate of returns. We
initially start with choosing sector specific macroeconomic indicators and
implement Recursive Feature Elimination algorithm to select the most important
features for each sector. Using our prediction tool, we implement different
Recurrent Neural Networks models to predict the future ETF prices for each
sector. We then rank the sectors based on their predicted rate of returns. We
select the best performing model by evaluating the annualized return,
annualized Sharpe ratio, and Calmar ratio of the portfolios that includes the
top four ranked sectors chosen by the model. We also test the robustness of the
model performance with respect to lookback windows and look ahead windows. Our
empirical results show that our methodology beats the equally weighted
portfolio performance even in the long run. We also find that Echo State
Networks exhibits an outstanding performance compared to other models yet it is
faster to implement compared to other RNN models.

    

### [[2108.02842] Multimodal Meta-Learning for Time Series Regression](http://arxiv.org/abs/2108.02842)


  Recent work has shown the efficiency of deep learning models such as Fully
Convolutional Networks (FCN) or Recurrent Neural Networks (RNN) to deal with
Time Series Regression (TSR) problems. These models sometimes need a lot of
data to be able to generalize, yet the time series are sometimes not long
enough to be able to learn patterns. Therefore, it is important to make use of
information across time series to improve learning. In this paper, we will
explore the idea of using meta-learning for quickly adapting model parameters
to new short-history time series by modifying the original idea of Model
Agnostic Meta-Learning (MAML) \cite{finn2017model}. Moreover, based on prior
work on multimodal MAML \cite{vuorio2019multimodal}, we propose a method for
conditioning parameters of the model through an auxiliary network that encodes
global information of the time series to extract meta-features. Finally, we
apply the data to time series of different domains, such as pollution
measurements, heart-rate sensors, and electrical battery data. We show
empirically that our proposed meta-learning method learns TSR with few data
fast and outperforms the baselines in 9 of 12 experiments.

    

### [[2108.02846] Communicative Learning with Natural Gestures for Embodied Navigation Agents with Human-in-the-Scene](http://arxiv.org/abs/2108.02846)


  Human-robot collaboration is an essential research topic in artificial
intelligence (AI), enabling researchers to devise cognitive AI systems and
affords an intuitive means for users to interact with the robot. Of note,
communication plays a central role. To date, prior studies in embodied agent
navigation have only demonstrated that human languages facilitate communication
by instructions in natural languages. Nevertheless, a plethora of other forms
of communication is left unexplored. In fact, human communication originated in
gestures and oftentimes is delivered through multimodal cues, e.g. "go there"
with a pointing gesture. To bridge the gap and fill in the missing dimension of
communication in embodied agent navigation, we propose investigating the
effects of using gestures as the communicative interface instead of verbal
cues. Specifically, we develop a VR-based 3D simulation environment, named
Ges-THOR, based on AI2-THOR platform. In this virtual environment, a human
player is placed in the same virtual scene and shepherds the artificial agent
using only gestures. The agent is tasked to solve the navigation problem guided
by natural gestures with unknown semantics; we do not use any predefined
gestures due to the diversity and versatile nature of human gestures. We argue
that learning the semantics of natural gestures is mutually beneficial to
learning the navigation task--learn to communicate and communicate to learn. In
a series of experiments, we demonstrate that human gesture cues, even without
predefined semantics, improve the object-goal navigation for an embodied agent,
outperforming various state-of-the-art methods.

    

### [[2108.02850] Unsupervised Domain Adaptation in Speech Recognition using Phonetic Features](http://arxiv.org/abs/2108.02850)


  Automatic speech recognition is a difficult problem in pattern recognition
because several sources of variability exist in the speech input like the
channel variations, the input might be clean or noisy, the speakers may have
different accent and variations in the gender, etc. As a result, domain
adaptation is important in speech recognition where we train the model for a
particular source domain and test it on a different target domain. In this
paper, we propose a technique to perform unsupervised gender-based domain
adaptation in speech recognition using phonetic features. The experiments are
performed on the TIMIT dataset and there is a considerable decrease in the
phoneme error rate using the proposed approach.

    

### [[2108.02853] Supervised Neural Networks for Illiquid Alternative Asset Cash Flow Forecasting](http://arxiv.org/abs/2108.02853)


  Institutional investors have been increasing the allocation of the illiquid
alternative assets such as private equity funds in their portfolios, yet there
exists a very limited literature on cash flow forecasting of illiquid
alternative assets. The net cash flow of private equity funds typically follow
a J-curve pattern, however the timing and the size of the contributions and
distributions depend on the investment opportunities. In this paper, we develop
a benchmark model and present two novel approaches (direct vs. indirect) to
predict the cash flows of private equity funds. We introduce a sliding window
approach to apply on our cash flow data because different vintage year funds
contain different lengths of cash flow information. We then pass the data to an
LSTM/ GRU model to predict the future cash flows either directly or indirectly
(based on the benchmark model). We further integrate macroeconomic indicators
into our data, which allows us to consider the impact of market environment on
cash flows and to apply stress testing. Our results indicate that the direct
model is easier to implement compared to the benchmark model and the indirect
model, but still the predicted cash flows align better with the actual cash
flows. We also show that macroeconomic variables improve the performance of the
direct model whereas the impact is not obvious on the indirect model.

    

### [[2108.02867] Enterprise Analytics using Graph Database and Graph-based Deep Learning](http://arxiv.org/abs/2108.02867)


  In a business-to-business (B2B) customer relationship management (CRM) use
case, each client is a potential business organization/company with a solid
business strategy and focused and rational decisions. This paper introduces a
graph-based analytics approach to improve CRM within a B2B environment. In our
approach, in the first instance, we have designed a graph database using the
Neo4j platform. Secondly, the graph database has been investigated by using
data mining and exploratory analysis coupled with cypher graph query language.
Specifically, we have applied the graph convolution network (GCN) to enable CRM
analytics to forecast sales. This is the first step towards a GCN-based binary
classification based on graph databases in the domain of B2B CRM. We evaluate
the performance of the proposed GCN model on graph databases and compare it
with Random Forest (RF), Convolutional Neural Network (CNN), and Artificial
Neural Network (ANN). The proposed GCN approach is further augmented with the
shortest path and eigenvector centrality attribute to significantly improve the
accuracy of sales prediction. Experimental results reveal that the proposed
graph-based deep learning approach outperforms the Random Forests (RsF) and two
deep learning models, i.e., CNN and ANN under different combinations of graph
features.

    

### [[2108.02870] A Data Augmented Approach to Transfer Learning for Covid-19 Detection](http://arxiv.org/abs/2108.02870)


  Covid-19 detection at an early stage can aid in an effective treatment and
isolation plan to prevent its spread. Recently, transfer learning has been used
for Covid-19 detection using X-ray, ultrasound, and CT scans. One of the major
limitations inherent to these proposed methods is limited labeled dataset size
that affects the reliability of Covid-19 diagnosis and disease progression. In
this work, we demonstrate that how we can augment limited X-ray images data by
using Contrast limited adaptive histogram equalization (CLAHE) to train the
last layer of the pre-trained deep learning models to mitigate the bias of
transfer learning for Covid-19 detection. We transfer learned various
pre-trained deep learning models including AlexNet, ZFNet, VGG-16, ResNet-18,
and GoogLeNet, and fine-tune the last layer by using CLAHE-augmented dataset.
The experiment results reveal that the CLAHE-based augmentation to various
pre-trained deep learning models significantly improves the model efficiency.
The pre-trained VCG-16 model with CLAHEbased augmented images achieves a
sensitivity of 95% using 15 epochs. AlexNet works show good sensitivity when
trained on non-augmented data. Other models demonstrate a value of less than
60% when trained on non-augmented data. Our results reveal that the sample bias
can negatively impact the performance of transfer learning which is
significantly improved by using CLAHE-based augmentation.

    

### [[2108.02872] Understanding Human Innate Immune System Dependencies using Graph Neural Networks](http://arxiv.org/abs/2108.02872)


  Since the rapid outbreak of Covid-19 and with no approved vaccines to date,
profound research interest has emerged to understand the innate immune response
to viruses. This understanding can help to inhibit virus replication, prolong
adaptive immune response, accelerated virus clearance, and tissue recovery, a
key milestone to propose a vaccine to combat coronaviruses (CoVs), e.g.,
Covid-19. Although an innate immune system triggers inflammatory responses
against CoVs upon recognition of viruses, however, a vaccine is the ultimate
protection against CoV spread. The development of this vaccine is
time-consuming and requires a deep understanding of the innate immune response
system. In this work, we propose a graph neural network-based model that
exploits the interactions between pattern recognition receptors (PRRs), i.e.,
the human immune response system. These interactions can help to recognize
pathogen-associated molecular patterns (PAMPs) to predict the activation
requirements of each PRR. The immune response information of each PRR is
derived from combining its historical PAMPs activation coupled with the modeled
effect on the same from PRRs in its neighborhood. On one hand, this work can
help to understand how long Covid-19 can confer immunity where a strong immune
response means people already been infected can safely return to work. On the
other hand, this GNN-based understanding can also abode well for vaccine
development efforts. Our proposal has been evaluated using CoVs immune response
dataset, with results showing an average IFNs activation prediction accuracy of
90%, compared to 85% using feed-forward neural networks.

    

### [[2108.02883] Interpolation can hurt robust generalization even when there is no noise](http://arxiv.org/abs/2108.02883)


  Numerous recent works show that overparameterization implicitly reduces
variance for min-norm interpolators and max-margin classifiers. These findings
suggest that ridge regularization has vanishing benefits in high dimensions. We
challenge this narrative by showing that, even in the absence of noise,
avoiding interpolation through ridge regularization can significantly improve
generalization. We prove this phenomenon for the robust risk of both linear
regression and classification and hence provide the first theoretical result on
robust overfitting.

    

### [[2108.02889] RIS-assisted UAV Communications for IoT with Wireless Power Transfer Using Deep Reinforcement Learning](http://arxiv.org/abs/2108.02889)


  Many of the devices used in Internet-of-Things (IoT) applications are
energy-limited, and thus supplying energy while maintaining seamless
connectivity for IoT devices is of considerable importance. In this context, we
propose a simultaneous wireless power transfer and information transmission
scheme for IoT devices with support from reconfigurable intelligent surface
(RIS)-aided unmanned aerial vehicle (UAV) communications. In particular, in a
first phase, IoT devices harvest energy from the UAV through wireless power
transfer; and then in a second phase, the UAV collects data from the IoT
devices through information transmission. To characterise the agility of the
UAV, we consider two scenarios: a hovering UAV and a mobile UAV. Aiming at
maximizing the total network sum-rate, we jointly optimize the trajectory of
the UAV, the energy harvesting scheduling of IoT devices, and the phaseshift
matrix of the RIS. We formulate a Markov decision process and propose two deep
reinforcement learning algorithms to solve the optimization problem of
maximizing the total network sum-rate. Numerical results illustrate the
effectiveness of the UAV's flying path optimization and the network's
throughput of our proposed techniques compared with other benchmark schemes.
Given the strict requirements of the RIS and UAV, the significant improvement
in processing time and throughput performance demonstrates that our proposed
scheme is well applicable for practical IoT applications.

    

### [[2108.02891] User Scheduling for Federated Learning Through Over-the-Air Computation](http://arxiv.org/abs/2108.02891)


  A new machine learning (ML) technique termed as federated learning (FL) aims
to preserve data at the edge devices and to only exchange ML model parameters
in the learning process. FL not only reduces the communication needs but also
helps to protect the local privacy. Although FL has these advantages, it can
still experience large communication latency when there are massive edge
devices connected to the central parameter server (PS) and/or millions of model
parameters involved in the learning process. Over-the-air computation (AirComp)
is capable of computing while transmitting data by allowing multiple devices to
send data simultaneously by using analog modulation. To achieve good
performance in FL through AirComp, user scheduling plays a critical role. In
this paper, we investigate and compare different user scheduling policies,
which are based on various criteria such as wireless channel conditions and the
significance of model updates. Receiver beamforming is applied to minimize the
mean-square-error (MSE) of the distortion of function aggregation result via
AirComp. Simulation results show that scheduling based on the significance of
model updates has smaller fluctuations in the training process while scheduling
based on channel condition has the advantage on energy efficiency.

    

### [[2108.02892] Deep Reinforcement Learning for Intelligent Reflecting Surface-assisted D2D Communications](http://arxiv.org/abs/2108.02892)


  In this paper, we propose a deep reinforcement learning (DRL) approach for
solving the optimisation problem of the network's sum-rate in device-to-device
(D2D) communications supported by an intelligent reflecting surface (IRS). The
IRS is deployed to mitigate the interference and enhance the signal between the
D2D transmitter and the associated D2D receiver. Our objective is to jointly
optimise the transmit power at the D2D transmitter and the phase shift matrix
at the IRS to maximise the network sum-rate. We formulate a Markov decision
process and then propose the proximal policy optimisation for solving the
maximisation game. Simulation results show impressive performance in terms of
the achievable rate and processing time.

    

### [[2108.02899] Lights, Camera, Action! A Framework to Improve NLP Accuracy over OCR documents](http://arxiv.org/abs/2108.02899)


  Document digitization is essential for the digital transformation of our
societies, yet a crucial step in the process, Optical Character Recognition
(OCR), is still not perfect. Even commercial OCR systems can produce
questionable output depending on the fidelity of the scanned documents. In this
paper, we demonstrate an effective framework for mitigating OCR errors for any
downstream NLP task, using Named Entity Recognition (NER) as an example. We
first address the data scarcity problem for model training by constructing a
document synthesis pipeline, generating realistic but degraded data with NER
labels. We measure the NER accuracy drop at various degradation levels and show
that a text restoration model, trained on the degraded data, significantly
closes the NER accuracy gaps caused by OCR errors, including on an
out-of-domain dataset. For the benefit of the community, we have made the
document synthesis pipeline available as an open-source project.

    

### [[2108.02904] Building a Foundation for Data-Driven, Interpretable, and Robust Policy Design using the AI Economist](http://arxiv.org/abs/2108.02904)


  Optimizing economic and public policy is critical to address socioeconomic
issues and trade-offs, e.g., improving equality, productivity, or wellness, and
poses a complex mechanism design problem. A policy designer needs to consider
multiple objectives, policy levers, and behavioral responses from strategic
actors who optimize for their individual objectives. Moreover, real-world
policies should be explainable and robust to simulation-to-reality gaps, e.g.,
due to calibration issues. Existing approaches are often limited to a narrow
set of policy levers or objectives that are hard to measure, do not yield
explicit optimal policies, or do not consider strategic behavior, for example.
Hence, it remains challenging to optimize policy in real-world scenarios. Here
we show that the AI Economist framework enables effective, flexible, and
interpretable policy design using two-level reinforcement learning (RL) and
data-driven simulations. We validate our framework on optimizing the stringency
of US state policies and Federal subsidies during a pandemic, e.g., COVID-19,
using a simulation fitted to real data. We find that log-linear policies
trained using RL significantly improve social welfare, based on both public
health and economic outcomes, compared to past outcomes. Their behavior can be
explained, e.g., well-performing policies respond strongly to changes in
recovery and vaccination rates. They are also robust to calibration errors,
e.g., infection rates that are over or underestimated. As of yet, real-world
policymaking has not seen adoption of machine learning methods at large,
including RL and AI-driven simulations. Our results show the potential of AI to
guide policy design and improve social welfare amidst the complexity of the
real world.

    

### [[2108.02922] Mitigating dataset harms requires stewardship: Lessons from 1000 papers](http://arxiv.org/abs/2108.02922)


  Concerns about privacy, bias, and harmful applications have shone a light on
the ethics of machine learning datasets, even leading to the retraction of
prominent datasets including DukeMTMC, MS-Celeb-1M, TinyImages, and VGGFace2.
In response, the machine learning community has called for higher ethical
standards, transparency efforts, and technical fixes in the dataset creation
process. The premise of our work is that these efforts can be more effective if
informed by an understanding of how datasets are used in practice in the
research community. We study three influential face and person recognition
datasets - DukeMTMC, MS-Celeb-1M, and Labeled Faces in the Wild (LFW) - by
analyzing nearly 1000 papers that cite them. We found that the creation of
derivative datasets and models, broader technological and social change, the
lack of clarity of licenses, and dataset management practices can introduce a
wide range of ethical concerns. We conclude by suggesting a distributed
approach that can mitigate these harms, making recommendations to dataset
creators, conference program committees, dataset users, and the broader
research community.

    

### [[2108.02932] Incremental Feature Learning For Infinite Data](http://arxiv.org/abs/2108.02932)


  This study addresses the actual behavior of the credit-card fraud detection
environment where financial transactions containing sensitive data must not be
amassed in an enormous amount to conduct learning. We introduce a new adaptive
learning approach that adjusts frequently and efficiently to new transaction
chunks; each chunk is discarded after each incremental training step. Our
approach combines transfer learning and incremental feature learning. The
former improves the feature relevancy for subsequent chunks, and the latter, a
new paradigm, increases accuracy during training by determining the optimal
network architecture dynamically for each new chunk. The architectures of past
incremental approaches are fixed; thus, the accuracy may not improve with new
chunks. We show the effectiveness and superiority of our approach
experimentally on an actual fraud dataset.

    

### [[2108.02941] Is it Fake? News Disinformation Detection on South African News Websites](http://arxiv.org/abs/2108.02941)


  Disinformation through fake news is an ongoing problem in our society and has
become easily spread through social media. The most cost and time effective way
to filter these large amounts of data is to use a combination of human and
technical interventions to identify it. From a technical perspective, Natural
Language Processing (NLP) is widely used in detecting fake news. Social media
companies use NLP techniques to identify the fake news and warn their users,
but fake news may still slip through undetected. It is especially a problem in
more localised contexts (outside the United States of America). How do we
adjust fake news detection systems to work better for local contexts such as in
South Africa. In this work we investigate fake news detection on South African
websites. We curate a dataset of South African fake news and then train
detection models. We contrast this with using widely available fake news
datasets (from mostly USA website). We also explore making the datasets more
diverse by combining them and observe the differences in behaviour in writing
between nations' fake news using interpretable machine learning.

    

### [[2108.02943] Unsupervised Learning of Debiased Representations with Pseudo-Attributes](http://arxiv.org/abs/2108.02943)


  Dataset bias is a critical challenge in machine learning, and its negative
impact is aggravated when models capture unintended decision rules with
spurious correlations. Although existing works often handle this issue using
human supervision, the availability of the proper annotations is impractical
and even unrealistic. To better tackle this challenge, we propose a simple but
effective debiasing technique in an unsupervised manner. Specifically, we
perform clustering on the feature embedding space and identify pseudoattributes
by taking advantage of the clustering results even without an explicit
attribute supervision. Then, we employ a novel cluster-based reweighting scheme
for learning debiased representation; this prevents minority groups from being
discounted for minimizing the overall loss, which is desirable for worst-case
generalization. The extensive experiments demonstrate the outstanding
performance of our approach on multiple standard benchmarks, which is even as
competitive as the supervised counterpart.

    

### [[2108.02949] Auxiliary Class Based Multiple Choice Learning](http://arxiv.org/abs/2108.02949)


  The merit of ensemble learning lies in having different outputs from many
individual models on a single input, i.e., the diversity of the base models.
The high quality of diversity can be achieved when each model is specialized to
different subsets of the whole dataset. Moreover, when each model explicitly
knows to which subsets it is specialized, more opportunities arise to improve
diversity. In this paper, we propose an advanced ensemble method, called
Auxiliary class based Multiple Choice Learning (AMCL), to ultimately specialize
each model under the framework of multiple choice learning (MCL). The
advancement of AMCL is originated from three novel techniques which control the
framework from different directions: 1) the concept of auxiliary class to
provide more distinct information through the labels, 2) the strategy, named
memory-based assignment, to determine the association between the inputs and
the models, and 3) the feature fusion module to achieve generalized features.
To demonstrate the performance of our method compared to all variants of MCL
methods, we conduct extensive experiments on the image classification and
segmentation tasks. Overall, the performance of AMCL exceeds all others in most
of the public datasets trained with various networks as members of the
ensembles.

    

### [[2108.02998] AI-based Aortic Vessel Tree Segmentation for Cardiovascular Diseases Treatment: Status Quo](http://arxiv.org/abs/2108.02998)


  The aortic vessel tree is composed of the aorta and its branching arteries,
and plays a key role in supplying the whole body with blood. Aortic diseases,
like aneurysms or dissections, can lead to an aortic rupture, whose treatment
with open surgery is highly risky. Therefore, patients commonly undergo drug
treatment under constant monitoring, which requires regular inspections of the
vessels through imaging. The standard imaging modality for diagnosis and
monitoring is computed tomography (CT), which can provide a detailed picture of
the aorta and its branching vessels if combined with a contrast agent,
resulting in a CT angiography (CTA). Optimally, the whole aortic vessel tree
geometry from consecutive CTAs, are overlaid and compared. This allows to not
only detect changes in the aorta, but also more peripheral vessel tree changes,
caused by the primary pathology or newly developed. When performed manually,
this reconstruction requires slice by slice contouring, which could easily take
a whole day for a single aortic vessel tree and, hence, is not feasible in
clinical practice. Automatic or semi-automatic vessel tree segmentation
algorithms, on the other hand, can complete this task in a fraction of the
manual execution time and run in parallel to the clinical routine of the
clinicians. In this paper, we systematically review computing techniques for
the automatic and semi-automatic segmentation of the aortic vessel tree. The
review concludes with an in-depth discussion on how close these
state-of-the-art approaches are to an application in clinical practice and how
active this research field is, taking into account the number of publications,
datasets and challenges.

    

### [[2108.03002] Fast and Accurate Low-Rank Tensor Completion Methods Based on QR Decomposition and $L_{2,1}$ Norm Minimization](http://arxiv.org/abs/2108.03002)


  More recently, an Approximate SVD Based on Qatar Riyal (QR) Decomposition
(CSVD-QR) method for matrix complete problem is presented, whose computational
complexity is $O(r^2(m+n))$, which is mainly due to that $r$ is far less than
$\min\{m,n\}$, where $r$ represents the largest number of singular values of
matrix $X$. What is particularly interesting is that after replacing the
nuclear norm with the $L_{2,1}$ norm proposed based on this decomposition, as
the upper bound of the nuclear norm, when the intermediate matrix $D$ in its
decomposition is close to the diagonal matrix, it will converge to the nuclear
norm, and is exactly equal, when the $D$ matrix is equal to the diagonal
matrix, to the nuclear norm, which ingeniously avoids the calculation of the
singular value of the matrix. To the best of our knowledge, there is no
literature to generalize and apply it to solve tensor complete problems.
Inspired by this, in this paper we propose a class of tensor minimization model
based on $L_{2,1}$ norm and CSVD-QR method for the tensor complete problem,
which is convex and therefore has a global minimum solution.

    

### [[2108.03011] Inspecting the Process of Bank Credit Rating via Visual Analytics](http://arxiv.org/abs/2108.03011)


  Bank credit rating classifies banks into different levels based on publicly
disclosed and internal information, serving as an important input in financial
risk management. However, domain experts have a vague idea of exploring and
comparing different bank credit rating schemes. A loose connection between
subjective and quantitative analysis and difficulties in determining
appropriate indicator weights obscure understanding of bank credit ratings.
Furthermore, existing models fail to consider bank types by just applying a
unified indicator weight set to all banks. We propose RatingVis to assist
experts in exploring and comparing different bank credit rating schemes. It
supports interactively inferring indicator weights for banks by involving
domain knowledge and considers bank types in the analysis loop. We conduct a
case study with real-world bank data to verify the efficacy of RatingVis.
Expert feedback suggests that our approach helps them better understand
different rating schemes.

    

### [[2108.03013] Interpretable Summaries of Black Box Incident Triaging with Subgroup Discovery](http://arxiv.org/abs/2108.03013)


  The need of predictive maintenance comes with an increasing number of
incidents reported by monitoring systems and equipment/software users. In the
front line, on-call engineers (OCEs) have to quickly assess the degree of
severity of an incident and decide which service to contact for corrective
actions. To automate these decisions, several predictive models have been
proposed, but the most efficient models are opaque (say, black box), strongly
limiting their adoption. In this paper, we propose an efficient black box model
based on 170K incidents reported to our company over the last 7 years and
emphasize on the need of automating triage when incidents are massively
reported on thousands of servers running our product, an ERP. Recent
developments in eXplainable Artificial Intelligence (XAI) help in providing
global explanations to the model, but also, and most importantly, with local
explanations for each model prediction/outcome. Sadly, providing a human with
an explanation for each outcome is not conceivable when dealing with an
important number of daily predictions. To address this problem, we propose an
original data-mining method rooted in Subgroup Discovery, a pattern mining
technique with the natural ability to group objects that share similar
explanations of their black box predictions and provide a description for each
group. We evaluate this approach and present our preliminary results which give
us good hope towards an effective OCE's adoption. We believe that this approach
provides a new way to address the problem of model agnostic outcome
explanation.

    

### [[2108.03039] Identifiable Energy-based Representations: An Application to Estimating Heterogeneous Causal Effects](http://arxiv.org/abs/2108.03039)


  Conditional average treatment effects (CATEs) allow us to understand the
effect heterogeneity across a large population of individuals. However, typical
CATE learners assume all confounding variables are measured in order for the
CATE to be identifiable. Often, this requirement is satisfied by simply
collecting many variables, at the expense of increased sample complexity for
estimating CATEs. To combat this, we propose an energy-based model (EBM) that
learns a low-dimensional representation of the variables by employing a noise
contrastive loss function. With our EBM we introduce a preprocessing step that
alleviates the dimensionality curse for any existing model and learner
developed for estimating CATE. We prove that our EBM keeps the representations
partially identifiable up to some universal constant, as well as having
universal approximation capability to avoid excessive information loss from
model misspecification; these properties combined with our loss function,
enable the representations to converge and keep the CATE estimation consistent.
Experiments demonstrate the convergence of the representations, as well as show
that estimating CATEs on our representations performs better than on the
variables or the representations obtained via various benchmark dimensionality
reduction methods.

    

### [[2108.03064] Spatiotemporal Contrastive Learning of Facial Expressions in Videos](http://arxiv.org/abs/2108.03064)


  We propose a self-supervised contrastive learning approach for facial
expression recognition (FER) in videos. We propose a novel temporal
sampling-based augmentation scheme to be utilized in addition to standard
spatial augmentations used for contrastive learning. Our proposed temporal
augmentation scheme randomly picks from one of three temporal sampling
techniques: (1) pure random sampling, (2) uniform sampling, and (3) sequential
sampling. This is followed by a combination of up to three standard spatial
augmentations. We then use a deep R(2+1)D network for FER, which we train in a
self-supervised fashion based on the augmentations and subsequently fine-tune.
Experiments are performed on the Oulu-CASIA dataset and the performance is
compared to other works in FER. The results indicate that our method achieves
an accuracy of 89.4%, setting a new state-of-the-art by outperforming other
works. Additional experiments and analysis confirm the considerable
contribution of the proposed temporal augmentation versus the existing spatial
ones.

    

### [[2108.03067] Deriving Disinformation Insights from Geolocalized Twitter Callouts](http://arxiv.org/abs/2108.03067)


  This paper demonstrates a two-stage method for deriving insights from social
media data relating to disinformation by applying a combination of geospatial
classification and embedding-based language modelling across multiple
languages. In particular, the analysis in centered on Twitter and
disinformation for three European languages: English, French and Spanish.
Firstly, Twitter data is classified into European and non-European sets using
BERT. Secondly, Word2vec is applied to the classified texts resulting in
Eurocentric, non-Eurocentric and global representations of the data for the
three target languages. This comparative analysis demonstrates not only the
efficacy of the classification method but also highlights geographic, temporal
and linguistic differences in the disinformation-related media. Thus, the
contributions of the work are threefold: (i) a novel language-independent
transformer-based geolocation method; (ii) an analytical approach that exploits
lexical specificity and word embeddings to interrogate user-generated content;
and (iii) a dataset of 36 million disinformation related tweets in English,
French and Spanish.

    

### [[2108.03068] Machine learning for surface prediction in ACTS](http://arxiv.org/abs/2108.03068)


  We present an ongoing R&D activity for machine-learning-assisted navigation
through detectors to be used for track reconstruction. We investigate different
approaches of training neural networks for surface prediction and compare their
results. This work is carried out in the context of the ACTS tracking toolkit.

    

### [[2108.03081] Rectified Euler k-means and Beyond](http://arxiv.org/abs/2108.03081)


  Euler k-means (EulerK) first maps data onto the unit hyper-sphere surface of
equi-dimensional space via a complex mapping which induces the robust Euler
kernel and next employs the popular $k$-means. Consequently, besides enjoying
the virtues of k-means such as simplicity and scalability to large data sets,
EulerK is also robust to noises and outliers. Although so, the centroids
captured by EulerK deviate from the unit hyper-sphere surface and thus in
strict distributional sense, actually are outliers. This weird phenomenon also
occurs in some generic kernel clustering methods. Intuitively, using such
outlier-like centroids should not be quite reasonable but it is still seldom
attended. To eliminate the deviation, we propose two Rectified Euler k-means
methods, i.e., REK1 and REK2, which retain the merits of EulerK while acquire
real centroids residing on the mapped space to better characterize the data
structures. Specifically, REK1 rectifies EulerK by imposing the constraint on
the centroids while REK2 views each centroid as the mapped image from a
pre-image in the original space and optimizes these pre-images in Euler kernel
induced space. Undoubtedly, our proposed REKs can methodologically be extended
to solve problems of such a category. Finally, the experiments validate the
effectiveness of REK1 and REK2.

    

### [[2108.03084] Transferring Knowledge Distillation for Multilingual Social Event Detection](http://arxiv.org/abs/2108.03084)


  Recently published graph neural networks (GNNs) show promising performance at
social event detection tasks. However, most studies are oriented toward
monolingual data in languages with abundant training samples. This has left the
more common multilingual settings and lesser-spoken languages relatively
unexplored. Thus, we present a GNN that incorporates cross-lingual word
embeddings for detecting events in multilingual data streams. The first exploit
is to make the GNN work with multilingual data. For this, we outline a
construction strategy that aligns messages in different languages at both the
node and semantic levels. Relationships between messages are established by
merging entities that are the same but are referred to in different languages.
Non-English message representations are converted into English semantic space
via the cross-lingual word embeddings. The resulting message graph is then
uniformly encoded by a GNN model. In special cases where a lesser-spoken
language needs to be detected, a novel cross-lingual knowledge distillation
framework, called CLKD, exploits prior knowledge learned from similar threads
in English to make up for the paucity of annotated data. Experiments on both
synthetic and real-world datasets show the framework to be highly effective at
detection in both multilingual data and in languages where training samples are
scarce.

    

### [[2108.03087] Detecting Requirements Smells With Deep Learning: Experiences, Challenges and Future Work](http://arxiv.org/abs/2108.03087)


  Requirements Engineering (RE) is the initial step towards building a software
system. The success or failure of a software project is firmly tied to this
phase, based on communication among stakeholders using natural language. The
problem with natural language is that it can easily lead to different
understandings if it is not expressed precisely by the stakeholders involved,
which results in building a product different from the expected one. Previous
work proposed to enhance the quality of the software requirements detecting
language errors based on ISO 29148 requirements language criteria. The existing
solutions apply classical Natural Language Processing (NLP) to detect them. NLP
has some limitations, such as domain dependability which results in poor
generalization capability. Therefore, this work aims to improve the previous
work by creating a manually labeled dataset and using ensemble learning, Deep
Learning (DL), and techniques such as word embeddings and transfer learning to
overcome the generalization problem that is tied with classical NLP and improve
precision and recall metrics using a manually labeled dataset. The current
findings show that the dataset is unbalanced and which class examples should be
added more. It is tempting to train algorithms even if the dataset is not
considerably representative. Whence, the results show that models are
overfitting; in Machine Learning this issue is solved by adding more instances
to the dataset, improving label quality, removing noise, and reducing the
learning algorithms complexity, which is planned for this research.

    

### [[2108.03090] Path classification by stochastic linear recurrent neural networks](http://arxiv.org/abs/2108.03090)


  We investigate the functioning of a classifying biological neural network
from the perspective of statistical learning theory, modelled, in a simplified
setting, as a continuous-time stochastic recurrent neural network (RNN) with
identity activation function. In the purely stochastic (robust) regime, we give
a generalisation error bound that holds with high probability, thus showing
that the empirical risk minimiser is the best-in-class hypothesis. We show that
RNNs retain a partial signature of the paths they are fed as the unique
information exploited for training and classification tasks. We argue that
these RNNs are easy to train and robust and back these observations with
numerical experiments on both synthetic and real data. We also exhibit a
trade-off phenomenon between accuracy and robustness.

    

### [[2108.03117] Uncertainty-Based Dynamic Graph Neighborhoods For Medical Segmentation](http://arxiv.org/abs/2108.03117)


  In recent years, deep learning based methods have shown success in essential
medical image analysis tasks such as segmentation. Post-processing and refining
the results of segmentation is a common practice to decrease the
misclassifications originating from the segmentation network. In addition to
widely used methods like Conditional Random Fields (CRFs) which focus on the
structure of the segmented volume/area, a graph-based recent approach makes use
of certain and uncertain points in a graph and refines the segmentation
according to a small graph convolutional network (GCN). However, there are two
drawbacks of the approach: most of the edges in the graph are assigned randomly
and the GCN is trained independently from the segmentation network. To address
these issues, we define a new neighbor-selection mechanism according to feature
distances and combine the two networks in the training procedure. According to
the experimental results on pancreas segmentation from Computed Tomography (CT)
images, we demonstrate improvement in the quantitative measures. Also,
examining the dynamic neighbors created by our method, edges between
semantically similar image parts are observed. The proposed method also shows
qualitative enhancements in the segmentation maps, as demonstrated in the
visual results.

    

### [[2108.03120] Stochastic Deep Model Reference Adaptive Control](http://arxiv.org/abs/2108.03120)


  In this paper, we present a Stochastic Deep Neural Network-based Model
Reference Adaptive Control. Building on our work "Deep Model Reference Adaptive
Control", we extend the controller capability by using Bayesian deep neural
networks (DNN) to represent uncertainties and model non-linearities. Stochastic
Deep Model Reference Adaptive Control uses a Lyapunov-based method to adapt the
output-layer weights of the DNN model in real-time, while a data-driven
supervised learning algorithm is used to update the inner-layers parameters.
This asynchronous network update ensures boundedness and guaranteed tracking
performance with a learning-based real-time feedback controller. A Bayesian
approach to DNN learning helped avoid over-fitting the data and provide
confidence intervals over the predictions. The controller's stochastic nature
also ensured "Induced Persistency of excitation," leading to convergence of the
overall system signal.

    

### [[2108.03131] COVID-Net US: A Tailored, Highly Efficient, Self-Attention Deep Convolutional Neural Network Design for Detection of COVID-19 Patient Cases from Point-of-care Ultrasound Imaging](http://arxiv.org/abs/2108.03131)


  The Coronavirus Disease 2019 (COVID-19) pandemic has impacted many aspects of
life globally, and a critical factor in mitigating its effects is screening
individuals for infections, thereby allowing for both proper treatment for
those individuals as well as action to be taken to prevent further spread of
the virus. Point-of-care ultrasound (POCUS) imaging has been proposed as a
screening tool as it is a much cheaper and easier to apply imaging modality
than others that are traditionally used for pulmonary examinations, namely
chest x-ray and computed tomography. Given the scarcity of expert radiologists
for interpreting POCUS examinations in many highly affected regions around the
world, low-cost deep learning-driven clinical decision support solutions can
have a large impact during the on-going pandemic. Motivated by this, we
introduce COVID-Net US, a highly efficient, self-attention deep convolutional
neural network design tailored for COVID-19 screening from lung POCUS images.
Experimental results show that the proposed COVID-Net US can achieve an AUC of
over 0.98 while achieving 353X lower architectural complexity, 62X lower
computational complexity, and 14.3X faster inference times on a Raspberry Pi.
Clinical validation was also conducted, where select cases were reviewed and
reported on by a practicing clinician (20 years of clinical practice)
specializing in intensive care (ICU) and 15 years of expertise in POCUS
interpretation. To advocate affordable healthcare and artificial intelligence
for resource-constrained environments, we have made COVID-Net US open source
and publicly available as part of the COVID-Net open source initiative.

    

### [[2108.03132] RockGPT: Reconstructing three-dimensional digital rocks from single two-dimensional slice from the perspective of video generation](http://arxiv.org/abs/2108.03132)


  Random reconstruction of three-dimensional (3D) digital rocks from
two-dimensional (2D) slices is crucial for elucidating the microstructure of
rocks and its effects on pore-scale flow in terms of numerical modeling, since
massive samples are usually required to handle intrinsic uncertainties. Despite
remarkable advances achieved by traditional process-based methods, statistical
approaches and recently famous deep learning-based models, few works have
focused on producing several kinds of rocks with one trained model and allowing
the reconstructed samples to satisfy certain given properties, such as
porosity. To fill this gap, we propose a new framework, named RockGPT, which is
composed of VQ-VAE and conditional GPT, to synthesize 3D samples based on a
single 2D slice from the perspective of video generation. The VQ-VAE is
utilized to compress high-dimensional input video, i.e., the sequence of
continuous rock slices, to discrete latent codes and reconstruct them. In order
to obtain diverse reconstructions, the discrete latent codes are modeled using
conditional GPT in an autoregressive manner, while incorporating conditional
information from a given slice, rock type, and porosity. We conduct two
experiments on five kinds of rocks, and the results demonstrate that RockGPT
can produce different kinds of rocks with the same model, and the reconstructed
samples can successfully meet certain specified porosities. In a broader sense,
through leveraging the proposed conditioning scheme, RockGPT constitutes an
effective way to build a general model to produce multiple kinds of rocks
simultaneously that also satisfy user-defined properties.

    

### [[2108.03140] SELM: Siamese Extreme Learning Machine with Application to Face Biometrics](http://arxiv.org/abs/2108.03140)


  Extreme Learning Machine is a powerful classification method very competitive
existing classification methods. It is extremely fast at training.
Nevertheless, it cannot perform face verification tasks properly because face
verification tasks require comparison of facial images of two individuals at
the same time and decide whether the two faces identify the same person. The
structure of Extreme Leaning Machine was not designed to feed two input data
streams simultaneously, thus, in 2-input scenarios Extreme Learning Machine
methods are normally applied using concatenated inputs. However, this setup
consumes two times more computational resources and it is not optimized for
recognition tasks where learning a separable distance metric is critical. For
these reasons, we propose and develop a Siamese Extreme Learning Machine
(SELM). SELM was designed to be fed with two data streams in parallel
simultaneously. It utilizes a dual-stream Siamese condition in the extra
Siamese layer to transform the data before passing it along to the hidden
layer. Moreover, we propose a Gender-Ethnicity-Dependent triplet feature
exclusively trained on a variety of specific demographic groups. This feature
enables learning and extracting of useful facial features of each group.
Experiments were conducted to evaluate and compare the performances of SELM,
Extreme Learning Machine, and DCNN. The experimental results showed that the
proposed feature was able to perform correct classification at 97.87% accuracy
and 99.45% AUC. They also showed that using SELM in conjunction with the
proposed feature provided 98.31% accuracy and 99.72% AUC. They outperformed the
well-known DCNN and Extreme Leaning Machine methods by a wide margin.

    

### [[2108.03150] Attainment Regions in Feature-Parameter Space for High-Level Debugging in Autonomous Robots](http://arxiv.org/abs/2108.03150)


  Understanding a controller's performance in different scenarios is crucial
for robots that are going to be deployed in safety-critical tasks. If we do not
have a model of the dynamics of the world, which is often the case in complex
domains, we may need to approximate a performance function of the robot based
on its interaction with the environment. Such a performance function gives us
insights into the behaviour of the robot, allowing us to fine-tune the
controller with manual interventions. In high-dimensionality systems, where the
actionstate space is large, fine-tuning a controller is non-trivial. To
overcome this problem, we propose a performance function whose domain is
defined by external features and parameters of the controller. Attainment
regions are defined over such a domain defined by feature-parameter pairs, and
serve the purpose of enabling prediction of successful execution of the task.
The use of the feature-parameter space -in contrast to the action-state space-
allows us to adapt, explain and finetune the controller over a simpler (i.e.,
lower dimensional space). When the robot successfully executes the task, we use
the attainment regions to gain insights into the limits of the controller, and
its robustness. When the robot fails to execute the task, we use the regions to
debug the controller and find adaptive and counterfactual changes to the
solutions. Another advantage of this approach is that we can generalise through
the use of Gaussian processes regression of the performance function in the
high-dimensional space. To test our approach, we demonstrate learning an
approximation to the performance function in simulation, with a mobile robot
traversing different terrain conditions. Then, with a sample-efficient method,
we propagate the attainment regions to a physical robot in a similar
environment.

    

### [[2108.03166] Feature Augmented Hybrid CNN for Stress Recognition Using Wrist-based Photoplethysmography Sensor](http://arxiv.org/abs/2108.03166)


  Stress is a physiological state that hampers mental health and has serious
consequences to physical health. Moreover, the COVID-19 pandemic has increased
stress levels among people across the globe. Therefore, continuous monitoring
and detection of stress are necessary. The recent advances in wearable devices
have allowed the monitoring of several physiological signals related to stress.
Among them, wrist-worn wearable devices like smartwatches are most popular due
to their convenient usage. And the photoplethysmography (PPG) sensor is the
most prevalent sensor in almost all consumer-grade wrist-worn smartwatches.
Therefore, this paper focuses on using a wrist-based PPG sensor that collects
Blood Volume Pulse (BVP) signals to detect stress which may be applicable for
consumer-grade wristwatches. Moreover, state-of-the-art works have used either
classical machine learning algorithms to detect stress using hand-crafted
features or have used deep learning algorithms like Convolutional Neural
Network (CNN) which automatically extracts features. This paper proposes a
novel hybrid CNN (H-CNN) classifier that uses both the hand-crafted features
and the automatically extracted features by CNN to detect stress using the BVP
signal. Evaluation on the benchmark WESAD dataset shows that, for 3-class
classification (Baseline vs. Stress vs. Amusement), our proposed H-CNN
outperforms traditional classifiers and normal CNN by 5% and 7% accuracy, and
10% and 7% macro F1 score, respectively. Also for 2-class classification
(Stress vs. Non-stress), our proposed H-CNN outperforms traditional classifiers
and normal CNN by 3% and ~5% accuracy, and ~3% and ~7% macro F1 score,
respectively.

    

### [[2108.03167] Generalized Tensor Summation Compressive Sensing Network (GTSNET): An Easy to Learn Compressive Sensing Operation](http://arxiv.org/abs/2108.03167)


  In CS literature, the efforts can be divided into two groups: finding a
measurement matrix that preserves the compressed information at the maximum
level, and finding a reconstruction algorithm for the compressed information.
In the traditional CS setup, the measurement matrices are selected as random
matrices, and optimization-based iterative solutions are used to recover the
signals. However, when we handle large signals, using random matrices become
cumbersome especially when it comes to iterative optimization-based solutions.
Even though recent deep learning-based solutions boost the reconstruction
accuracy performance while speeding up the recovery, still jointly learning the
whole measurement matrix is a difficult process. In this work, we introduce a
separable multi-linear learning of the CS matrix by representing it as the
summation of arbitrary number of tensors. For a special case where the CS
operation is set as a single tensor multiplication, the model is reduced to the
learning-based separable CS; while a dense CS matrix can be approximated and
learned as the summation of multiple tensors. Both cases can be used in CS of
two or multi-dimensional signals e.g., images, multi-spectral images, videos,
etc. Structural CS matrices can also be easily approximated and learned in our
multi-linear separable learning setup with structural tensor sum
representation. Hence, our learnable generalized tensor summation CS operation
encapsulates most CS setups including separable CS, non-separable CS
(traditional vector-matrix multiplication), structural CS, and CS of the
multi-dimensional signals. For both gray-scale and RGB images, the proposed
scheme surpasses most state-of-the-art solutions, especially in lower
measurement rates. Although the performance gain remains limited from tensor to
the sum of tensor representation for gray-scale images, it becomes significant
in the RGB case.

    

### [[2108.03169] Responding to Illegal Activities Along the Canadian Coastlines Using Reinforcement Learning](http://arxiv.org/abs/2108.03169)


  This article elaborates on how machine learning (ML) can leverage the
solution of a contemporary problem related to the security of maritime domains.
The worldwide ``Illegal, Unreported, and Unregulated'' (IUU) fishing incidents
have led to serious environmental and economic consequences which involve
drastic changes in our ecosystems in addition to financial losses caused by the
depletion of natural resources. The Fisheries and Aquatic Department (FAD) of
the United Nation's Food and Agriculture Organization (FAO) issued a report
which indicated that the annual losses due to IUU fishing reached $25 Billion.
This imposes negative impacts on the future-biodiversity of the marine
ecosystem and domestic Gross National Product (GNP). Hence, robust interception
mechanisms are increasingly needed for detecting and pursuing the unrelenting
illegal fishing incidents in maritime territories. This article addresses the
problem of coordinating the motion of a fleet of marine vessels (pursuers) to
catch an IUU vessel while still in local waters. The problem is formulated as a
pursuer-evader problem that is tackled within an ML framework. One or more
pursuers, such as law enforcement vessels, intercept an evader (i.e., the
illegal fishing ship) using an online reinforcement learning mechanism that is
based on a value iteration process. It employs real-time navigation
measurements of the evader ship as well as those of the pursuing vessels and
returns back model-free interception strategies.

    

### [[2108.03177] Shift-invariant waveform learning on epileptic ECoG](http://arxiv.org/abs/2108.03177)


  Seizure detection algorithms must discriminate abnormal neuronal activity
associated with a seizure from normal neural activity in a variety of
conditions. Our approach is to seek spatiotemporal waveforms with distinct
morphology in electrocorticographic (ECoG) recordings of epileptic patients
that are indicative of a subsequent seizure (preictal) versus non-seizure
segments (interictal). To find these waveforms we apply a shift-invariant
k-means algorithm to segments of spatially filtered signals to learn codebooks
of prototypical waveforms. The frequency of the cluster labels from the
codebooks is then used to train a binary classifier that predicts the class
(preictal or interictal) of a test ECoG segment. We use the Matthews
correlation coefficient to evaluate the performance of the classifier and the
quality of the codebooks. We found that our method finds recurrent
non-sinusoidal waveforms that could be used to build interpretable features for
seizure prediction and that are also physiologically meaningful.

    

### [[2108.03213] Temporally Abstract Partial Models](http://arxiv.org/abs/2108.03213)


  Humans and animals have the ability to reason and make predictions about
different courses of action at many time scales. In reinforcement learning,
option models (Sutton, Precup \& Singh, 1999; Precup, 2000) provide the
framework for this kind of temporally abstract prediction and reasoning.
Natural intelligent agents are also able to focus their attention on courses of
action that are relevant or feasible in a given situation, sometimes termed
affordable actions. In this paper, we define a notion of affordances for
options, and develop temporally abstract partial option models, that take into
account the fact that an option might be affordable only in certain situations.
We analyze the trade-offs between estimation and approximation error in
planning and learning when using such models, and identify some interesting
special cases. Additionally, we demonstrate empirically the potential impact of
partial option models on the efficiency of planning.

    

### [[2108.03214] Simple Modifications to Improve Tabular Neural Networks](http://arxiv.org/abs/2108.03214)


  There is growing interest in neural network architectures for tabular data.
Many general-purpose tabular deep learning models have been introduced
recently, with performance sometimes rivaling gradient boosted decision trees
(GBDTs). These recent models draw inspiration from various sources, including
GBDTs, factorization machines, and neural networks from other application
domains. Previous tabular neural networks are also drawn upon, but are possibly
under-considered, especially models associated with specific tabular problems.
This paper focuses on several such models, and proposes modifications for
improving their performance. When modified, these models are shown to be
competitive with leading general-purpose tabular models, including GBDTs.

    

### [[2108.03217] Analysis of Driving Scenario Trajectories with Active Learning](http://arxiv.org/abs/2108.03217)


  Annotating the driving scenario trajectories based only on explicit rules
(i.e., knowledge-based methods) can be subject to errors, such as false
positive/negative classification of scenarios that lie on the border of two
scenario classes, missing unknown scenario classes, and also anomalies. On the
other side, verifying the labels by the annotators is not cost-efficient. For
this purpose, active learning (AL) could potentially improve the annotation
procedure by inclusion of an annotator/expert in an efficient way. In this
study, we develop an active learning framework to annotate driving trajectory
time-series data. At the first step, we compute an embedding of the time-series
trajectories into a latent space in order to extract the temporal nature. For
this purpose, we study three different latent space representations:
multivariate Time Series t-Distributed Stochastic Neighbor Embedding (mTSNE),
Recurrent Auto-Encoder (RAE) and Variational Recurrent Auto-Encoder (VRAE). We
then apply different active learning paradigms with different classification
models to the embedded data. In particular, we study the two classifiers Neural
Network (NN) and Support Vector Machines (SVM), with three active learning
query strategies (i.e., entropy, margin and random). In the following, we
explore the possibilities of the framework to discover unknown classes and
demonstrate how it can be used to identify the out-of-class trajectories.

    

### [[2108.03222] A Study on Dense and Sparse (Visual) Rewards in Robot Policy Learning](http://arxiv.org/abs/2108.03222)


  Deep Reinforcement Learning (DRL) is a promising approach for teaching robots
new behaviour. However, one of its main limitations is the need for carefully
hand-coded reward signals by an expert. We argue that it is crucial to automate
the reward learning process so that new skills can be taught to robots by their
users. To address such automation, we consider task success classifiers using
visual observations to estimate the rewards in terms of task success. In this
work, we study the performance of multiple state-of-the-art deep reinforcement
learning algorithms under different types of reward: Dense, Sparse, Visual
Dense, and Visual Sparse rewards. Our experiments in various simulation tasks
(Pendulum, Reacher, Pusher, and Fetch Reach) show that while DRL agents can
learn successful behaviours using visual rewards when the goal targets are
distinguishable, their performance may decrease if the task goal is not clearly
visible. Our results also show that visual dense rewards are more successful
than visual sparse rewards and that there is no single best algorithm for all
tasks.

    

### [[1911.03809] Meta Label Correction for Noisy Label Learning](http://arxiv.org/abs/1911.03809)


  Leveraging weak or noisy supervision for building effective machine learning
models has long been an important research problem. Its importance has further
increased recently due to the growing need for large-scale datasets to train
deep learning models. Weak or noisy supervision could originate from multiple
sources including non-expert annotators or automatic labeling based on
heuristics or user interaction signals. There is an extensive amount of
previous work focusing on leveraging noisy labels. Most notably, recent work
has shown impressive gains by using a meta-learned instance re-weighting
approach where a meta-learning framework is used to assign instance weights to
noisy labels. In this paper, we extend this approach via posing the problem as
label correction problem within a meta-learning framework. We view the label
correction procedure as a meta-process and propose a new meta-learning based
framework termed MLC (Meta Label Correction) for learning with noisy labels.
Specifically, a label correction network is adopted as a meta-model to produce
corrected labels for noisy labels while the main model is trained to leverage
the corrected labeled. Both models are jointly trained by solving a bi-level
optimization problem. We run extensive experiments with different label noise
levels and types on both image recognition and text classification tasks. We
compare the reweighing and correction approaches showing that the correction
framing addresses some of the limitation of reweighting. We also show that the
proposed MLC approach achieves large improvements over previous methods in many
settings.

    

### [[1912.07942] Analyzing Information Leakage of Updates to Natural Language Models](http://arxiv.org/abs/1912.07942)


  To continuously improve quality and reflect changes in data, machine learning
applications have to regularly retrain and update their core models. We show
that a differential analysis of language model snapshots before and after an
update can reveal a surprising amount of detailed information about changes in
the training data. We propose two new metrics---\emph{differential score} and
\emph{differential rank}---for analyzing the leakage due to updates of natural
language models. We perform leakage analysis using these metrics across models
trained on several different datasets using different methods and
configurations. We discuss the privacy implications of our findings, propose
mitigation strategies and evaluate their effect.

    

### [[2006.09638] Approximate Gradient Coding with Optimal Decoding](http://arxiv.org/abs/2006.09638)


  In distributed optimization problems, a technique called gradient coding,
which involves replicating data points, has been used to mitigate the effect of
straggling machines. Recent work has studied approximate gradient coding, which
concerns coding schemes where the replication factor of the data is too low to
recover the full gradient exactly. Our work is motivated by the challenge of
creating approximate gradient coding schemes that simultaneously work well in
both the adversarial and stochastic models. To that end, we introduce novel
approximate gradient codes based on expander graphs, in which each machine
receives exactly two blocks of data points. We analyze the decoding error both
in the random and adversarial straggler setting, when optimal decoding
coefficients are used. We show that in the random setting, our schemes achieve
an error to the gradient that decays exponentially in the replication factor.
In the adversarial setting, the error is nearly a factor of two smaller than
any existing code with similar performance in the random setting. We show
convergence bounds both in the random and adversarial setting for gradient
descent under standard assumptions using our codes. In the random setting, our
convergence rate improves upon block-box bounds. In the adversarial setting, we
show that gradient descent can converge down to a noise floor that scales
linearly with the adversarial error to the gradient. We demonstrate empirically
that our schemes achieve near-optimal error in the random setting and converge
faster than algorithms which do not use the optimal decoding coefficients.

    

### [[2007.08428] On Adversarial Robustness: A Neural Architecture Search perspective](http://arxiv.org/abs/2007.08428)


  Adversarial robustness of deep learning models has gained much traction in
the last few years. Various attacks and defenses are proposed to improve the
adversarial robustness of modern-day deep learning architectures. While all
these approaches help improve the robustness, one promising direction for
improving adversarial robustness is un-explored, i.e., the complex topology of
the neural network architecture. In this work, we answer the following
question: "Can the complex topology of a neural network give adversarial
robustness without any form of adversarial training?" empirically by
experimenting with different hand-crafted and NAS based architectures. Our
findings show that, for small-scale attacks, NAS-based architectures are more
robust for small-scale datasets and simple tasks than hand-crafted
architectures. However, as the dataset's size or the task's complexity
increase, hand-crafted architectures are more robust than NAS-based
architectures. We perform the first large scale study to understand adversarial
robustness purely from an architectural perspective. Our results show that
random sampling in the search space of DARTS (a popular NAS method) with simple
ensembling can improve the robustness to PGD attack by nearly ~12\%. We show
that NAS, which is popular for SoTA accuracy, can provide adversarial accuracy
as a free add-on without any form of adversarial training. Our results show
that leveraging the power of neural network topology with methods like
ensembles can be an excellent way to achieve adversarial robustness without any
form of adversarial training. We also introduce a metric that can be used to
calculate the trade-off between clean accuracy and adversarial robustness.

    

### [[2007.08637] COV-ELM classifier: An Extreme Learning Machine based identification of COVID-19 using Chest X-Ray Images](http://arxiv.org/abs/2007.08637)


  Coronaviruses constitute a family of viruses that gives rise to respiratory
diseases. As COVID-19 is highly contagious, early diagnosis of COVID-19 is
crucial for an effective treatment strategy. However, the RT-PCR test which is
considered to be a gold standard in the diagnosis of COVID-19 suffers from a
high false-negative rate. Chest X-ray (CXR) image analysis has emerged as a
feasible and effective diagnostic technique towards this objective. In this
work, we propose the COVID-19 classification problem as a three-class
classification problem to distinguish between COVID-19, normal, and pneumonia
classes. We propose a three-stage framework, named COV-ELM. Stage one deals
with preprocessing and transformation while stage two deals with feature
extraction. These extracted features are passed as an input to the ELM at the
third stage, resulting in the identification of COVID-19. The choice of ELM in
this work has been motivated by its faster convergence, better generalization
capability, and shorter training time in comparison to the conventional
gradient-based learning algorithms. As bigger and diverse datasets become
available, ELM can be quickly retrained as compared to its gradient-based
competitor models. The proposed model achieved a macro average F1-score of 0.95
and the overall sensitivity of ${0.94 \pm 0.02} at a 95% confidence interval.
When compared to state-of-the-art machine learning algorithms, the COV-ELM is
found to outperform its competitors in this three-class classification
scenario. Further, LIME has been integrated with the proposed COV-ELM model to
generate annotated CXR images. The annotations are based on the superpixels
that have contributed to distinguish between the different classes. It was
observed that the superpixels correspond to the regions of the human lungs that
are clinically observed in COVID-19 and Pneumonia cases.

    

### [[2011.06175] Optimizing Large-Scale Fleet Management on a Road Network using Multi-Agent Deep Reinforcement Learning with Graph Neural Network](http://arxiv.org/abs/2011.06175)


  We propose a novel approach to optimize fleet management by combining
multi-agent reinforcement learning with graph neural network. To provide
ride-hailing service, one needs to optimize dynamic resources and demands over
spatial domain. While the spatial structure was previously approximated with a
regular grid, our approach represents the road network with a graph, which
better reflects the underlying geometric structure. Dynamic resource allocation
is formulated as multi-agent reinforcement learning, whose action-value
function (Q function) is approximated with graph neural networks. We use
stochastic policy update rule over the graph with deep Q-networks (DQN), and
achieve superior results over the greedy policy update. We design a realistic
simulator that emulates the empirical taxi call data, and confirm the
effectiveness of the proposed model under various conditions.

    

### [[2011.10367] PSD2 Explainable AI Model for Credit Scoring](http://arxiv.org/abs/2011.10367)


  The aim of this project is to develop and test advanced analytical methods to
improve the prediction accuracy of Credit Risk Models, preserving at the same
time the model interpretability. In particular, the project focuses on applying
an explainable machine learning model to bank-related databases. The input data
were obtained from open data. Over the total proven models, CatBoost has shown
the highest performance. The algorithm implementation produces a GINI of 0.68
after tuning the hyper-parameters. SHAP package is used to provide a global and
local interpretation of the model predictions to formulate a
human-comprehensive approach to understanding the decision-maker algorithm. The
20 most important features are selected using the Shapley values to present a
full human-understandable model that reveals how the attributes of an
individual are related to its model prediction.

    

### [[2012.00377] Latent Programmer: Discrete Latent Codes for Program Synthesis](http://arxiv.org/abs/2012.00377)


  In many sequence learning tasks, such as program synthesis and document
summarization, a key problem is searching over a large space of possible output
sequences. We propose to learn representations of the outputs that are
specifically meant for search: rich enough to specify the desired output but
compact enough to make search more efficient. Discrete latent codes are
appealing for this purpose, as they naturally allow sophisticated combinatorial
search strategies. The latent codes are learned using a self-supervised
learning principle, in which first a discrete autoencoder is trained on the
output sequences, and then the resulting latent codes are used as intermediate
targets for the end-to-end sequence prediction task. Based on these insights,
we introduce the \emph{Latent Programmer}, a program synthesis method that
first predicts a discrete latent code from input/output examples, and then
generates the program in the target language. We evaluate the Latent Programmer
on two domains: synthesis of string transformation programs, and generation of
programs from natural language descriptions. We demonstrate that the discrete
latent representation significantly improves synthesis accuracy.

    

### [[2012.06341] Beyond Occam's Razor in System Identification: Double-Descent when Modeling Dynamics](http://arxiv.org/abs/2012.06341)


  System identification aims to build models of dynamical systems from data.
Traditionally, choosing the model requires the designer to balance between two
goals of conflicting nature; the model must be rich enough to capture the
system dynamics, but not so flexible that it learns spurious random effects
from the dataset. It is typically observed that the model validation
performance follows a U-shaped curve as the model complexity increases. Recent
developments in machine learning and statistics, however, have observed
situations where a "double-descent" curve subsumes this U-shaped
model-performance curve. With a second decrease in performance occurring beyond
the point where the model has reached the capacity of interpolating - i.e.,
(near) perfectly fitting - the training data. To the best of our knowledge,
such phenomena have not been studied within the context of dynamic systems. The
present paper aims to answer the question: "Can such a phenomenon also be
observed when estimating parameters of dynamic systems?" We show that the
answer is yes, verifying such behavior experimentally both for artificially
generated and real-world datasets.

    

### [[2101.08862] Breaking the Deadly Triad with a Target Network](http://arxiv.org/abs/2101.08862)


  The deadly triad refers to the instability of a reinforcement learning
algorithm when it employs off-policy learning, function approximation, and
bootstrapping simultaneously. In this paper, we investigate the target network
as a tool for breaking the deadly triad, providing theoretical support for the
conventional wisdom that a target network stabilizes training. We first propose
and analyze a novel target network update rule which augments the commonly used
Polyak-averaging style update with two projections. We then apply the target
network and ridge regularization in several divergent algorithms and show their
convergence to regularized TD fixed points. Those algorithms are off-policy
with linear function approximation and bootstrapping, spanning both policy
evaluation and control, as well as both discounted and average-reward settings.
In particular, we provide the first convergent linear $Q$-learning algorithms
under nonrestrictive and changing behavior policies without bi-level
optimization.

    

### [[2101.11713] Neural networks for Anatomical Therapeutic Chemical (ATC) classification](http://arxiv.org/abs/2101.11713)


  Motivation: Automatic Anatomical Therapeutic Chemical (ATC) classification is
a critical and highly competitive area of research in bioinformatics because of
its potential for expediting drug develop-ment and research. Predicting an
unknown compound's therapeutic and chemical characteristics ac-cording to how
these characteristics affect multiple organs/systems makes automatic ATC
classifica-tion a challenging multi-label problem. Results: In this work, we
propose combining multiple multi-label classifiers trained on distinct sets of
features, including sets extracted from a Bidirectional Long Short-Term Memory
Network (BiLSTM). Experiments demonstrate the power of this approach, which is
shown to outperform the best methods reported in the literature, including the
state-of-the-art developed by the this http URL research group. Availability: All
source code developed for this study is available at
this https URL. Contact: loris.nanni@unipd.it

    

### [[2102.02504] Meta-strategy for Learning Tuning Parameters with Guarantees](http://arxiv.org/abs/2102.02504)


  Online learning methods, like the online gradient algorithm (OGA) and
exponentially weighted aggregation (EWA), often depend on tuning parameters
that are difficult to set in practice. We consider an online meta-learning
scenario, and we propose a meta-strategy to learn these parameters from past
tasks. Our strategy is based on the minimization of a regret bound. It allows
to learn the initialization and the step size in OGA with guarantees. It also
allows to learn the prior or the learning rate in EWA. We provide a regret
analysis of the strategy. It allows to identify settings where meta-learning
indeed improves on learning each task in isolation.

    

### [[2102.03877] Noise Reduction in X-ray Photon Correlation Spectroscopy with Convolutional Neural Networks Encoder-Decoder Models](http://arxiv.org/abs/2102.03877)


  Like other experimental techniques, X-ray Photon Correlation Spectroscopy is
subject to various kinds of noise. Random and correlated fluctuations and
heterogeneities can be present in a two-time correlation function and obscure
the information about the intrinsic dynamics of a sample. Simultaneously
addressing the disparate origins of noise in the experimental data is
challenging. We propose a computational approach for improving the
signal-to-noise ratio in two-time correlation functions that is based on
Convolutional Neural Network Encoder-Decoder (CNN-ED) models. Such models
extract features from an image via convolutional layers, project them to a low
dimensional space and then reconstruct a clean image from this reduced
representation via transposed convolutional layers. Not only are ED models a
general tool for random noise removal, but their application to low
signal-to-noise data can enhance the data quantitative usage since they are
able to learn the functional form of the signal. We demonstrate that the CNN-ED
models trained on real-world experimental data help to effectively extract
equilibrium dynamics parameters from two-time correlation functions, containing
statistical noise and dynamic heterogeneities. Strategies for optimizing the
models performance and their applicability limits are discussed.

    

### [[2102.05257] Robust Federated Learning with Attack-Adaptive Aggregation](http://arxiv.org/abs/2102.05257)


  Federated learning is vulnerable to various attacks, such as model poisoning
and backdoor attacks, even if some existing defense strategies are used. To
address this challenge, we propose an attack-adaptive aggregation strategy to
defend against various attacks for robust federated learning. The proposed
approach is based on training a neural network with an attention mechanism that
learns the vulnerability of federated learning models from a set of plausible
attacks. To the best of our knowledge, our aggregation strategy is the first
one that can be adapted to defend against various attacks in a data-driven
fashion. Our approach has achieved competitive performance in defending model
poisoning and backdoor attacks in federated learning tasks on image and text
datasets.

    

### [[2102.08991] Generalization in Quantum Machine Learning: a Quantum Information Perspective](http://arxiv.org/abs/2102.08991)


  Quantum classification and hypothesis testing are two tightly related
subjects, the main difference being that the former is data driven: how to
assign to quantum states $\rho(x)$ the corresponding class $c$ (or hypothesis)
is learnt from examples during training, where $x$ can be either tunable
experimental parameters or classical data "embedded" into quantum states. Does
the model generalize? This is the main question in any data-driven strategy,
namely the ability to predict the correct class even of previously unseen
states. Here we establish a link between quantum machine learning
classification and quantum hypothesis testing (state and channel
discrimination) and then show that the accuracy and generalization capability
of quantum classifiers depend on the (Rényi) mutual informations $I(C{:}Q)$
and $I_2(X{:}Q)$ between the quantum state space $Q$ and the classical
parameter space $X$ or class space $C$. Based on the above characterization, we
then show how different properties of $Q$ affect classification accuracy and
generalization, such as the dimension of the Hilbert space, the amount of
noise, and the amount of neglected information from $X$ via, e.g., pooling
layers. Moreover, we introduce a quantum version of the Information Bottleneck
principle that allows us to explore the various tradeoffs between accuracy and
generalization. Finally, in order to check our theoretical predictions, we
study the classification of the quantum phases of an Ising spin chain, and we
propose the Variational Quantum Information Bottleneck (VQIB) method to
optimize quantum embeddings of classical data to favor generalization.

    

### [[2102.11777] Federated Learning for Physical Layer Design](http://arxiv.org/abs/2102.11777)


  Model-free techniques, such as machine learning (ML), have recently attracted
much interest towards the physical layer design, e.g., symbol detection,
channel estimation, and beamforming. Most of these ML techniques employ
centralized learning (CL) schemes and assume the availability of datasets at a
parameter server (PS), demanding the transmission of data from edge devices,
such as mobile phones, to the PS. Exploiting the data generated at the edge,
federated learning (FL) has been proposed recently as a distributed learning
scheme, in which each device computes the model parameters and sends them to
the PS for model aggregation while the datasets are kept intact at the edge.
Thus, FL is more communication-efficient and privacy-preserving than CL and
applicable to the wireless communication scenarios, wherein the data are
generated at the edge devices. This article presents the recent advances in
FL-based training for physical layer design problems. Compared to CL, the
effectiveness of FL is presented in terms of communication overhead with a
slight performance loss in the learning accuracy. The design challenges, such
as model, data, and hardware complexity, are also discussed in detail along
with possible solutions.

    

### [[2103.01391] Sample Complexity and Overparameterization Bounds for Temporal Difference Learning with Neural Network Approximation](http://arxiv.org/abs/2103.01391)


  In this paper, we study the dynamics of temporal difference learning with
neural network-based value function approximation over a general state space,
namely, \emph{Neural TD learning}. We consider two practically used algorithms,
projection-free and max-norm regularized Neural TD learning, and establish the
first convergence bounds for these algorithms. An interesting observation from
our results is that max-norm regularization can dramatically improve the
performance of TD learning algorithms, both in terms of sample complexity and
overparameterization. In particular, we prove that max-norm regularization
improves state-of-the-art sample complexity and overparameterization bounds.
The results in this work rely on a novel Lyapunov drift analysis of the network
parameters as a stopped and controlled random process.

    

### [[2105.04240] A rigorous introduction for linear models](http://arxiv.org/abs/2105.04240)


  This survey is meant to provide an introduction to linear models and the
theories behind them. Our goal is to give a rigorous introduction to the
readers with prior exposure to ordinary least squares. In machine learning, the
output is usually a nonlinear function of the input. Deep learning even aims to
find a nonlinear dependence with many layers which require a large amount of
computation. However, most of these algorithms build upon simple linear models.
We then describe linear models from different views and find the properties and
theories behind the models. The linear model is the main technique in
regression problems and the primary tool for it is the least squares
approximation which minimizes a sum of squared errors. This is a natural choice
when we're interested in finding the regression function which minimizes the
corresponding expected squared error. This survey is primarily a summary of
purpose, significance of important theories behind linear models, e.g.,
distribution theory, minimum variance estimator. We first describe ordinary
least squares from three different points of view upon which we disturb the
model with random noise and Gaussian noise. By Gaussian noise, the model gives
rise to the likelihood so that we introduce a maximum likelihood estimator. It
also develops some distribution theories via this Gaussian disturbance. The
distribution theory of least squares will help us answer various questions and
introduce related applications. We then prove least squares is the best
unbiased linear model in the sense of mean squared error and most importantly,
it actually approaches the theoretical limit. We end up with linear models with
the Bayesian approach and beyond.

    

### [[2105.06073] Good and Bad Optimization Models: Insights from Rockafellians](http://arxiv.org/abs/2105.06073)


  A basic requirement for a mathematical model is often that its solution
(output) shouldn't change much if the model's parameters (input) are perturbed.
This is important because the exact values of parameters may not be known and
one would like to avoid being mislead by an output obtained using incorrect
values. Thus, it's rarely enough to address an application by formulating a
model, solving the resulting optimization problem and presenting the solution
as the answer. One would need to confirm that the model is suitable, i.e.,
"good," and this can, at least in part, be achieved by considering a family of
optimization problems constructed by perturbing parameters of concern. The
resulting sensitivity analysis uncovers troubling situations with unstable
solutions, which we referred to as "bad" models, and indicates better model
formulations. Embedding an actual problem of interest within a family of
problems is also a primary path to optimality conditions as well as
computationally attractive, alternative problems, which under ideal
circumstances, and when properly tuned, may even furnish the minimum value of
the actual problem. The tuning of these alternative problems turns out to be
intimately tied to finding multipliers in optimality conditions and thus
emerges as a main component of several optimization algorithms. In fact, the
tuning amounts to solving certain dual optimization problems. In this tutorial,
we'll discuss the opportunities and insights afforded by this broad
perspective.

    

### [[2106.04306] Residual Feedback Learning for Contact-Rich Manipulation Tasks with Uncertainty](http://arxiv.org/abs/2106.04306)


  While classic control theory offers state of the art solutions in many
problem scenarios, it is often desired to improve beyond the structure of such
solutions and surpass their limitations. To this end, residual policy learning
(RPL) offers a formulation to improve existing controllers with reinforcement
learning (RL) by learning an additive "residual" to the output of a given
controller. However, the applicability of such an approach highly depends on
the structure of the controller. Often, internal feedback signals of the
controller limit an RL algorithm to adequately change the policy and, hence,
learn the task. We propose a new formulation that addresses these limitations
by also modifying the feedback signals to the controller with an RL policy and
show superior performance of our approach on a contact-rich peg-insertion task
under position and orientation uncertainty. In addition, we use a recent
Cartesian impedance control architecture as the control framework which can be
available to us as a black-box while assuming no knowledge about its
input/output structure, and show the difficulties of standard RPL. Furthermore,
we introduce an adaptive curriculum for the given task to gradually increase
the task difficulty in terms of position and orientation uncertainty. A video
showing the results can be found at this https URL .

    

### [[2106.05476] Learning Based Proximity Matrix Factorization for Node Embedding](http://arxiv.org/abs/2106.05476)


  Node embedding learns a low-dimensional representation for each node in the
graph. Recent progress on node embedding shows that proximity matrix
factorization methods gain superb performance and scale to large graphs with
millions of nodes. Existing approaches first define a proximity matrix and then
learn the embeddings that fit the proximity by matrix factorization. Most
existing matrix factorization methods adopt the same proximity for different
tasks, while it is observed that different tasks and datasets may require
different proximity, limiting their representation power.
Motivated by this, we propose {\em Lemane}, a framework with trainable
proximity measures, which can be learned to best suit the datasets and tasks at
hand automatically. Our method is end-to-end, which incorporates differentiable
SVD in the pipeline so that the parameters can be trained via backpropagation.
However, this learning process is still expensive on large graphs. To improve
the scalability, we train proximity measures only on carefully subsampled
graphs, and then apply standard proximity matrix factorization on the original
graph using the learned proximity. Note that, computing the learned proximities
for each pair is still expensive for large graphs, and existing techniques for
computing proximities are not applicable to the learned proximities. Thus, we
present generalized push techniques to make our solution scalable to large
graphs with millions of nodes. Extensive experiments show that our proposed
solution outperforms existing solutions on both link prediction and node
classification tasks on almost all datasets.

    

### [[2106.08334] Quantum-inspired event reconstruction with Tensor Networks: Matrix Product States](http://arxiv.org/abs/2106.08334)


  Tensor Networks are non-trivial representations of high-dimensional tensors,
originally designed to describe quantum many-body systems. We show that Tensor
Networks are ideal vehicles to connect quantum mechanical concepts to machine
learning techniques, thereby facilitating an improved interpretability of
neural networks. This study presents the discrimination of top quark signal
over QCD background processes using a Matrix Product State classifier. We show
that entanglement entropy can be used to interpret what a network learns, which
can be used to reduce the complexity of the network and feature space without
loss of generality or performance. For the optimisation of the network, we
compare the Density Matrix Renormalization Group (DMRG) algorithm to stochastic
gradient descent (SGD) and propose a joined training algorithm to harness the
explainability of DMRG with the efficiency of SGD.

    

### [[2106.12543] Synthetic Benchmarks for Scientific Research in Explainable Machine Learning](http://arxiv.org/abs/2106.12543)


  As machine learning models grow more complex and their applications become
more high-stakes, tools for explaining model predictions have become
increasingly important. This has spurred a flurry of research in model
explainability and has given rise to feature attribution methods such as LIME
and SHAP. Despite their widespread use, evaluating and comparing different
feature attribution methods remains challenging: evaluations ideally require
human studies, and empirical evaluation metrics are often data-intensive or
computationally prohibitive on real-world datasets. In this work, we address
this issue by releasing XAI-Bench: a suite of synthetic datasets along with a
library for benchmarking feature attribution algorithms. Unlike real-world
datasets, synthetic datasets allow the efficient computation of conditional
expected values that are needed to evaluate ground-truth Shapley values and
other metrics. The synthetic datasets we release offer a wide variety of
parameters that can be configured to simulate real-world data. We demonstrate
the power of our library by benchmarking popular explainability techniques
across several evaluation metrics and across a variety of settings. The
versatility and efficiency of our library will help researchers bring their
explainability methods from development to deployment. Our code is available at
this https URL.

    

### [[2106.14144] Learning-based Framework for Sensor Fault-Tolerant Building HVAC Control with Model-assisted Learning](http://arxiv.org/abs/2106.14144)


  As people spend up to 87% of their time indoors, intelligent Heating,
Ventilation, and Air Conditioning (HVAC) systems in buildings are essential for
maintaining occupant comfort and reducing energy consumption. These HVAC
systems in smart buildings rely on real-time sensor readings, which in practice
often suffer from various faults and could also be vulnerable to malicious
attacks. Such faulty sensor inputs may lead to the violation of indoor
environment requirements (e.g., temperature, humidity, etc.) and the increase
of energy consumption. While many model-based approaches have been proposed in
the literature for building HVAC control, it is costly to develop accurate
physical models for ensuring their performance and even more challenging to
address the impact of sensor faults. In this work, we present a novel
learning-based framework for sensor fault-tolerant HVAC control, which includes
three deep learning based components for 1) generating temperature proposals
with the consideration of possible sensor faults, 2) selecting one of the
proposals based on the assessment of their accuracy, and 3) applying
reinforcement learning with the selected temperature proposal. Moreover, to
address the challenge of training data insufficiency in building-related tasks,
we propose a model-assisted learning method leveraging an abstract model of
building physical dynamics. Through extensive experiments, we demonstrate that
the proposed fault-tolerant HVAC control framework can significantly reduce
building temperature violations under a variety of sensor fault patterns while
maintaining energy efficiency.

    

### [[1806.00984] DNN-HMM based Speaker Adaptive Emotion Recognition using Proposed Epoch and MFCC Features](http://arxiv.org/abs/1806.00984)


  Speech is produced when time varying vocal tract system is excited with time
varying excitation source. Therefore, the information present in a speech such
as message, emotion, language, speaker is due to the combined effect of both
excitation source and vocal tract system. However, there is very less
utilization of excitation source features to recognize emotion. In our earlier
work, we have proposed a novel method to extract glottal closure instants
(GCIs) known as epochs. In this paper, we have explored epoch features namely
instantaneous pitch, phase and strength of epochs for discriminating emotions.
We have combined the excitation source features and the well known
Male-frequency cepstral coefficient (MFCC) features to develop an emotion
recognition system with improved performance. DNN-HMM speaker adaptive models
have been developed using MFCC, epoch and combined features. IEMOCAP emotional
database has been used to evaluate the models. The average accuracy for emotion
recognition system when using MFCC and epoch features separately is 59.25% and
54.52% respectively. The recognition performance improves to 64.2% when MFCC
and epoch features are combined.

    

### [[2108.02802] Lower Bounds for Shared-Memory Leader Election under Bounded Write Contention](http://arxiv.org/abs/2108.02802)


  This paper gives tight logarithmic lower bounds on the solo step complexity
of leader election in an asynchronous shared-memory model with single-writer
multi-reader (SWMR) registers, for both deterministic and randomized
obstruction-free algorithms.
The approach extends to lower bounds for deterministic and randomized
obstruction-free algorithms using multi-writer registers under bounded write
concurrency, showing a trade-off between the solo step complexity of a leader
election algorithm, and its worst-case write contention.

    

### [[2108.02898] Scalable Analysis for Covid-19 and Vaccine Data](http://arxiv.org/abs/2108.02898)


  This paper explains the scalable methods used for extracting and analyzing
the Covid-19 vaccine data. Using Big Data such as Hadoop and Hive, we collect
and analyze the massive data set of the confirmed, the fatality, and the
vaccination data set of Covid-19. The data size is about 3.2 Giga-Byte. We show
that it is possible to store and process massive data with Big Data. The paper
proceeds tempo-spatial analysis, and visual maps, charts, and pie charts
visualize the result of the investigation. We illustrate that the more
vaccinated, the fewer the confirmed cases.

    

### [[2108.02917] Toward Efficient Online Scheduling for Distributed Machine Learning Systems](http://arxiv.org/abs/2108.02917)


  Recent years have witnessed a rapid growth of distributed machine learning
(ML) frameworks, which exploit the massive parallelism of computing clusters to
expedite ML training. However, the proliferation of distributed ML frameworks
also introduces many unique technical challenges in computing system design and
optimization. In a networked computing cluster that supports a large number of
training jobs, a key question is how to design efficient scheduling algorithms
to allocate workers and parameter servers across different machines to minimize
the overall training time. Toward this end, in this paper, we develop an online
scheduling algorithm that jointly optimizes resource allocation and locality
decisions. Our main contributions are three-fold: i) We develop a new
analytical model that considers both resource allocation and locality; ii)
Based on an equivalent reformulation and observations on the worker-parameter
server locality configurations, we transform the problem into a mixed packing
and covering integer program, which enables approximation algorithm design;
iii) We propose a meticulously designed approximation algorithm based on
randomized rounding and rigorously analyze its performance. Collectively, our
results contribute to the state of the art of distributed ML system
optimization and algorithm design.

    

### [[2108.02991] Optimizing full 3D SPARKLING trajectories for high-resolution T2*-weighted Magnetic Resonance Imaging](http://arxiv.org/abs/2108.02991)


  The Spreading Projection Algorithm for Rapid K-space samplING, or SPARKLING,
is an optimization-driven method that has been recently introduced for
accelerated 2D T2*-w MRI using compressed sensing. It has then been extended to
address 3D imaging using either stacks of 2D sampling patterns or a local 3D
strategy that optimizes a single sampling trajectory at a time. 2D SPARKLING
actually performs variable density sampling (VDS) along a prescribed target
density while maximizing sampling efficiency and meeting the gradient-based
hardware constraints. However, 3D SPARKLING has remained limited in terms of
acceleration factors along the third dimension if one wants to preserve a peaky
point spread function (PSF) and thus good image this http URL this paper, in order
to achieve higher acceleration factors in 3D imaging while preserving image
quality, we propose a new efficient algorithm that performs optimization on
full 3D SPARKLING. The proposed implementation based on fast multipole methods
(FMM) allows us to design sampling patterns with up to 10^7 k-space samples,
thus opening the door to 3D VDS. We compare multi-CPU and GPU implementations
and demonstrate that the latter is optimal for 3D imaging in the
high-resolution acquisition regime (600$\mu$m isotropic). Finally, we show that
this novel optimization for full 3D SPARKLING outperforms stacking strategies
or 3D twisted projection imaging through retrospective and prospective studies
on NIST phantom and in vivo brain scans at 3 Tesla. Overall the proposed method
allows for 2.5-3.75x shorter scan times compared to GRAPPA-4 parallel imaging
acquisition at 3 Tesla without compromising image quality.

    

### [[2008.12712] Coffea -- Columnar Object Framework For Effective Analysis](http://arxiv.org/abs/2008.12712)


  The coffea framework provides a new approach to High-Energy Physics analysis,
via columnar operations, that improves time-to-insight, scalability,
portability, and reproducibility of analysis. It is implemented with the Python
programming language, the scientific python package ecosystem, and commodity
big data technologies. To achieve this suite of improvements across many use
cases, coffea takes a factorized approach, separating the analysis
implementation and data delivery scheme. All analysis operations are
implemented using the NumPy or awkward-array packages which are wrapped to
yield user code whose purpose is quickly intuited. Various data delivery
schemes are wrapped into a common front-end which accepts user inputs and code,
and returns user defined outputs. We will discuss our experience in
implementing analysis of CMS data using the coffea framework along with a
discussion of the user experience and future directions.

    

### [[2102.04322] Distributed Storage Allocations for Optimal Service Rates](http://arxiv.org/abs/2102.04322)


  Redundant storage maintains the performance of distributed systems under
various forms of uncertainty. This paper considers the uncertainty in node
access and download service. We consider two access models under two download
service models. In one access model, a user can access each node with a fixed
probability, and in the other, a user can access a random fixed-size subset of
nodes. We consider two download service models. In the first (small file)
model, the randomness associated with the file size is negligible. In the
second (large file) model, randomness is associated with both the file size and
the system's operations. We focus on the service rate of the system. For a
fixed redundancy level, the systems' service rate is determined by the
allocation of coded chunks over the storage nodes. We consider quasi-uniform
allocations, where coded content is uniformly spread among a subset of nodes.
The question we address asks what the size of this subset (spreading) should
be. We show that in the small file model, concentrating the coded content to a
minimum-size subset is universally optimal. For the large file model, the
optimal spreading depends on the system parameters. These conclusions hold for
both access models.

    

### [[2102.11660] Massively Parallel Correlation Clustering in Bounded Arboricity Graphs](http://arxiv.org/abs/2102.11660)


  Identifying clusters of similar elements in a set is a common task in data
analysis. With the immense growth of data and physical limitations on single
processor speed, it is necessary to find efficient parallel algorithms for
clustering tasks. In this paper, we study the problem of correlation clustering
in bounded arboricity graphs with respect to the Massively Parallel Computation
(MPC) model. More specifically, we are given a complete graph where the edges
are either positive or negative, indicating whether pairs of vertices are
similar or dissimilar. The task is to partition the vertices into clusters with
as few disagreements as possible. That is, we want to minimize the number of
positive inter-cluster edges and negative intra-cluster edges.
Consider an input graph $G$ on $n$ vertices such that the positive edges
induce a $\lambda$-arboric graph. Our main result is a 3-approximation
($\textit{in expectation}$) algorithm to correlation clustering that runs in
$\mathcal{O}(\log \lambda \cdot \textrm{poly}(\log \log n))$ MPC rounds in the
$\textit{strongly sublinear memory regime}$. This is obtained by combining
structural properties of correlation clustering on bounded arboricity graphs
with the insights of Fischer and Noever (SODA '18) on randomized greedy MIS and
the $\texttt{PIVOT}$ algorithm of Ailon, Charikar, and Newman (STOC '05).
Combined with known graph matching algorithms, our structural property also
implies an exact algorithm and algorithms with $\textit{worst case}$
$(1+\epsilon)$-approximation guarantees in the special case of forests, where
$\lambda=1$.

    

### [[2104.05313] Voting-based probabilistic consensuses and their applications in distributed ledgers](http://arxiv.org/abs/2104.05313)


  We review probabilistic models known as majority dynamics (also known as
threshold Voter Models) and discuss their possible applications for achieving
consensus in cryptocurrency systems. In particular, we show that using this
approach straightforwardly for practical consensus in Byzantine setting can be
problematic and requires extensive further research. We then discuss the FPC
consensus protocol which circumvents the problems mentioned above by using
external randomness.

    

### [[2105.04966] Permissionless and Asynchronous Asset Transfer [Technical Report]](http://arxiv.org/abs/2105.04966)


  Most modern asset transfer systems use consensus to maintain a totally
ordered chain of transactions. It was recently shown that consensus is not
always necessary for implementing asset transfer. More efficient, asynchronous
solutions can be built using reliable broadcast instead of consensus. This
approach has been originally used in the closed (permissioned) setting. In this
paper, we extend it to the open (permissionless) environment. We present
Pastro, a permissionless and asynchronous asset-transfer implementation, in
which quorum systems, traditionally used in reliable broadcast, are replaced
with a weighted Proof-of-Stake mechanism. Pastro tolerates a dynamic adversary
that is able to adaptively corrupt participants based on the assets owned by
them.

    

### [[2108.02797] I-DLV-sr: A Stream Reasoning System based on I-DLV](http://arxiv.org/abs/2108.02797)


  We introduce a novel logic-based system for reasoning over data streams,
which relies on a framework enabling a tight, fine-tuned interaction between
Apache Flink and the I^2-DLV system. The architecture allows to take advantage
from both the powerful distributed stream processing capabilities of Flink and
the incremental reasoning capabilities of I^2-DLV based on overgrounding
techniques. Besides the system architecture, we illustrate the supported input
language and its modeling capabilities, and discuss the results of an
experimental activity aimed at assessing the viability of the approach. This
paper is under consideration in Theory and Practice of Logic Programming
(TPLP).

    

### [[2108.02816] ProcessCO v1.3's Terms, Properties, Relationships and Axioms - A Core Ontology for Processes](http://arxiv.org/abs/2108.02816)


  The present preprint specifies and defines all Terms, Properties,
Relationships and Axioms of ProcessCO (Process Core Ontology). ProcessCO is an
ontology devoted mainly for Work Entities and related terms, which is placed at
the core level in the context of a multilayer ontological architecture called
FCD-OntoArch (Foundational, Core, and Domain Ontological Architecture for
Sciences). This is a five-layered ontological architecture, which considers
Foundational, Core, Domain and Instance levels, where the domain level is split
down in two sub-levels, namely: Top-domain and Low-domain. Ontologies at the
same level can be related to each other, except for the foundational level
where only ThingFO (Thing Foundational Ontology) is found. In addition,
ontologies' terms and relationships at lower levels can be semantically
enriched by ontologies' terms and relationships from the higher levels. Note
that both ThingFO and ontologies at the core level such as ProcessCO,
SituationCO, among others, are domain independent with respect to their terms.
Stereotypes are the mechanism used for enriching ProcessCO terms mainly from
the ThingFO ontology. Note that in the end of this document, we address the
ProcessCO vs. ThingFO non-taxonomic relationship verification matrix.
Additionally, note that annotations of updates from the previous version
(ProcessCO v1.2) to the current one (v1.3) can be found in Appendix A. For
instance, 6 axioms were added.

    

### [[2108.02818] Evaluating CLIP: Towards Characterization of Broader Capabilities and Downstream Implications](http://arxiv.org/abs/2108.02818)


  Recently, there have been breakthroughs in computer vision ("CV") models that
are more generalizable with the advent of models such as CLIP and ALIGN. In
this paper, we analyze CLIP and highlight some of the challenges such models
pose. CLIP reduces the need for task specific training data, potentially
opening up many niche tasks to automation. CLIP also allows its users to
flexibly specify image classification classes in natural language, which we
find can shift how biases manifest. Additionally, through some preliminary
probes we find that CLIP can inherit biases found in prior computer vision
systems. Given the wide and unpredictable domain of uses for such models, this
raises questions regarding what sufficiently safe behaviour for such systems
may look like. These results add evidence to the growing body of work calling
for a change in the notion of a 'better' model--to move beyond simply looking
at higher accuracy at task-oriented capability evaluations, and towards a
broader 'better' that takes into account deployment-critical features such as
different use contexts, and people who interact with the model when thinking
about model deployment.

    

### [[2108.02854] GENder-IT: An Annotated English-Italian Parallel Challenge Set for Cross-Linguistic Natural Gender Phenomena](http://arxiv.org/abs/2108.02854)


  Languages differ in terms of the absence or presence of gender features, the
number of gender classes and whether and where gender features are explicitly
marked. These cross-linguistic differences can lead to ambiguities that are
difficult to resolve, especially for sentence-level MT systems. The
identification of ambiguity and its subsequent resolution is a challenging task
for which currently there aren't any specific resources or challenge sets
available. In this paper, we introduce gENder-IT, an English--Italian challenge
set focusing on the resolution of natural gender phenomena by providing
word-level gender tags on the English source side and multiple gender
alternative translations, where needed, on the Italian target side.

    

### [[2108.02858] 3DRIMR: 3D Reconstruction and Imaging via mmWave Radar based on Deep Learning](http://arxiv.org/abs/2108.02858)


  mmWave radar has been shown as an effective sensing technique in low
visibility, smoke, dusty, and dense fog environment. However tapping the
potential of radar sensing to reconstruct 3D object shapes remains a great
challenge, due to the characteristics of radar data such as sparsity, low
resolution, specularity, high noise, and multi-path induced shadow reflections
and artifacts. In this paper we propose 3D Reconstruction and Imaging via
mmWave Radar (3DRIMR), a deep learning based architecture that reconstructs 3D
shape of an object in dense detailed point cloud format, based on sparse raw
mmWave radar intensity data. The architecture consists of two back-to-back
conditional GAN deep neural networks: the first generator network generates 2D
depth images based on raw radar intensity data, and the second generator
network outputs 3D point clouds based on the results of the first generator.
The architecture exploits both convolutional neural network's convolutional
operation (that extracts local structure neighborhood information) and the
efficiency and detailed geometry capture capability of point clouds (other than
costly voxelization of 3D space or distance fields). Our experiments have
demonstrated 3DRIMR's effectiveness in reconstructing 3D objects, and its
performance improvement over standard techniques.

    

### [[2108.02924] Interpretable Visual Understanding with Cognitive Attention Network](http://arxiv.org/abs/2108.02924)


  While image understanding on recognition-level has achieved remarkable
advancements, reliable visual scene understanding requires comprehensive image
understanding on recognition-level but also cognition-level, which calls for
exploiting the multi-source information as well as learning different levels of
understanding and extensive commonsense knowledge. In this paper, we propose a
novel Cognitive Attention Network (CAN) for visual commonsense reasoning to
achieve interpretable visual understanding. Specifically, we first introduce an
image-text fusion module to fuse information from images and text collectively.
Second, a novel inference module is designed to encode commonsense among image,
query and response. Extensive experiments on large-scale Visual Commonsense
Reasoning (VCR) benchmark dataset demonstrate the effectiveness of our
approach. The implementation is publicly available at
this https URL


### [[2108.03001] AceNAS: Learning to Rank Ace Neural Architectures with Weak Supervision of Weight Sharing](http://arxiv.org/abs/2108.03001)


  Architecture performance predictors have been widely used in neural
architecture search (NAS). Although they are shown to be simple and effective,
the optimization objectives in previous arts (e.g., precise accuracy estimation
or perfect ranking of all architectures in the space) did not capture the
ranking nature of NAS. In addition, a large number of ground-truth
architecture-accuracy pairs are usually required to build a reliable predictor,
making the process too computationally expensive. To overcome these, in this
paper, we look at NAS from a novel point of view and introduce Learning to Rank
(LTR) methods to select the best (ace) architectures from a space.
Specifically, we propose to use Normalized Discounted Cumulative Gain (NDCG) as
the target metric and LambdaRank as the training algorithm. We also propose to
leverage weak supervision from weight sharing by pretraining architecture
representation on weak labels obtained from the super-net and then finetuning
the ranking model using a small number of architectures trained from scratch.
Extensive experiments on NAS benchmarks and large-scale search spaces
demonstrate that our approach outperforms SOTA with a significantly reduced
search cost.

    

### [[2108.03008] An Empirical Study on End-to-End Singing Voice Synthesis with Encoder-Decoder Architectures](http://arxiv.org/abs/2108.03008)


  With the rapid development of neural network architectures and speech
processing models, singing voice synthesis with neural networks is becoming the
cutting-edge technique of digital music production. In this work, in order to
explore how to improve the quality and efficiency of singing voice synthesis,
in this work, we use encoder-decoder neural models and a number of vocoders to
achieve singing voice synthesis. We conduct experiments to demonstrate that the
models can be trained using voice data with pitch information, lyrics and beat
information, and the trained models can produce smooth, clear and natural
singing voice that is close to real human voice. As the models work in the
end-to-end manner, they allow users who are not domain experts to directly
produce singing voice by arranging pitches, lyrics and beats.

    

### [[2108.03022] Utilizing Treewidth for Quantitative Reasoning on Epistemic Logic Programs](http://arxiv.org/abs/2108.03022)


  Extending the popular Answer Set Programming (ASP) paradigm by introspective
reasoning capacities has received increasing interest within the last years.
Particular attention is given to the formalism of epistemic logic programs
(ELPs) where standard rules are equipped with modal operators which allow to
express conditions on literals for being known or possible, i.e., contained in
all or some answer sets, respectively. ELPs thus deliver multiple collections
of answer sets, known as world views. Employing ELPs for reasoning problems so
far has mainly been restricted to standard decision problems (complexity
analysis) and enumeration (development of systems) of world views. In this
paper, we take a next step and contribute to epistemic logic programming in two
ways: First, we establish quantitative reasoning for ELPs, where the acceptance
of a certain set of literals depends on the number (proportion) of world views
that are compatible with the set. Second, we present a novel system that is
capable of efficiently solving the underlying counting problems required to
answer such quantitative reasoning problems. Our system exploits the
graph-based measure treewidth and works by iteratively finding and refining
(graph) abstractions of an ELP program. On top of these abstractions, we apply
dynamic programming that is combined with utilizing existing search-based
solvers like (e)clingo for hard combinatorial subproblems that appear during
solving. It turns out that our approach is competitive with existing systems
that were introduced recently. This work is under consideration for acceptance
in TPLP.

    

### [[2108.03033] Non-ground Abductive Logic Programming with Probabilistic Integrity Constraints](http://arxiv.org/abs/2108.03033)


  Uncertain information is being taken into account in an increasing number of
application fields. In the meantime, abduction has been proved a powerful tool
for handling hypothetical reasoning and incomplete knowledge. Probabilistic
logical models are a suitable framework to handle uncertain information, and in
the last decade many probabilistic logical languages have been proposed, as
well as inference and learning systems for them. In the realm of Abductive
Logic Programming (ALP), a variety of proof procedures have been defined as
well. In this paper, we consider a richer logic language, coping with
probabilistic abduction with variables. In particular, we consider an ALP
program enriched with integrity constraints `a la IFF, possibly annotated with
a probability value. We first present the overall abductive language, and its
semantics according to the Distribution Semantics. We then introduce a proof
procedure, obtained by extending one previously presented, and prove its
soundness and completeness.

    

### [[2108.03061] Towards a Semantics for Hybrid ASP systems](http://arxiv.org/abs/2108.03061)


  Over the last decades the development of ASP has brought about an expressive
modeling language powered by highly performant systems. At the same time, it
gets more and more difficult to provide semantic underpinnings capturing the
resulting constructs and inferences. This is even more severe when it comes to
hybrid ASP languages and systems that are often needed to handle real-world
applications. We address this challenge and introduce the concept of abstract
and structured theories that allow us to formally elaborate upon their
integration with ASP. We then use this concept to make precise the semantic
characterization of CLINGO's theory-reasoning framework and establish its
correspondence to the logic of Here-and-there with constraints. This provides
us with a formal framework in which we can elaborate formal properties of
existing hybridizations of CLINGO such as CLINGCON, CLINGOM[DL], and
CLINGO[LP].

    

### [[2108.03100] Reasoning on Multi-Relational Contextual Hierarchies via Answer Set Programming with Algebraic Measures](http://arxiv.org/abs/2108.03100)


  Dealing with context dependent knowledge has led to different formalizations
of the notion of context. Among them is the Contextualized Knowledge Repository
(CKR) framework, which is rooted in description logics but links on the
reasoning side strongly to logic programs and Answer Set Programming (ASP) in
particular. The CKR framework caters for reasoning with defeasible axioms and
exceptions in contexts, which was extended to knowledge inheritance across
contexts in a coverage (specificity) hierarchy. However, the approach supports
only this single type of contextual relation and the reasoning procedures work
only for restricted hierarchies, due to non-trivial issues with model
preference under exceptions. In this paper, we overcome these limitations and
present a generalization of CKR hierarchies to multiple contextual relations,
along with their interpretation of defeasible axioms and preference. To support
reasoning, we use ASP with algebraic measures, which is a recent extension of
ASP with weighted formulas over semirings that allows one to associate
quantities with interpretations depending on the truth values of propositional
atoms. Notably, we show that for a relevant fragment of CKR hierarchies with
multiple contextual relations, query answering can be realized with the popular
asprin framework. The algebraic measures approach is more powerful and enables
e.g. reasoning with epistemic queries over CKRs, which opens interesting
perspectives for the use of quantitative ASP extensions in other applications.
Under consideration for acceptance in Theory and Practice of Logic Programming
(TPLP).

    

### [[2108.03173] Incremental learning of LSTM framework for sensor fusion in attitude estimation](http://arxiv.org/abs/2108.03173)


  This paper presents a novel method for attitude estimation of an object in 3D
space by incremental learning of the Long-Short Term Memory (LSTM) network.
Gyroscope, accelerometer, and magnetometer are few widely used sensors in
attitude estimation applications. Traditionally, multi-sensor fusion methods
such as the Extended Kalman Filter and Complementary Filter are employed to
fuse the measurements from these sensors. However, these methods exhibit
limitations in accounting for the uncertainty, unpredictability, and dynamic
nature of the motion in real-world situations. In this paper, the inertial
sensors data are fed to the LSTM network which are then updated incrementally
to incorporate the dynamic changes in motion occurring in the run time. The
robustness and efficiency of the proposed framework is demonstrated on the
dataset collected from a commercially available inertial measurement unit. The
proposed framework offers a significant improvement in the results compared to
the traditional method, even in the case of a highly dynamic environment. The
LSTM framework-based attitude estimation approach can be deployed on a standard
AI-supported processing module for real-time applications.

    

### [[2102.04130] Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models](http://arxiv.org/abs/2102.04130)


  The capabilities of natural language models trained on large-scale data have
increased immensely over the past few years. Open source libraries such as
HuggingFace have made these models easily available and accessible. While prior
research has identified biases in large language models, this paper considers
biases contained in the most popular versions of these models when applied
`out-of-the-box' for downstream tasks. We focus on generative language models
as they are well-suited for extracting biases inherited from training data.
Specifically, we conduct an in-depth analysis of GPT-2, which is the most
downloaded text generation model on HuggingFace, with over half a million
downloads in the past month alone. We assess biases related to occupational
associations for different protected categories by intersecting gender with
religion, sexuality, ethnicity, political affiliation, and continental name
origin. Using a template-based data collection pipeline, we collect 396K
sentence completions made by GPT-2 and find: (i) The machine-predicted jobs are
less diverse and more stereotypical for women than for men, especially for
intersections; (ii) Intersectional interactions are highly relevant for
occupational associations, which we quantify by fitting 262 logistic models;
(iii) For most occupations, GPT-2 reflects the skewed gender and ethnicity
distribution found in US Labour Bureau data, and even pulls the
societally-skewed distribution towards gender parity in cases where its
predictions deviate from real labor market observations. This raises the
normative question of what language models _should_ learn - whether they should
reflect or correct for existing inequalities.

    

### [[2103.00334] BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection](http://arxiv.org/abs/2103.00334)


  Salient object detection (SOD) is viewed as a pixel-wise saliency modeling
task by traditional deep learning-based methods. A limitation of current SOD
models is insufficient utilization of inter-pixel information, which usually
results in imperfect segmentation near edge regions and low spatial coherence.
As we demonstrate, using a saliency mask as the only label is suboptimal. To
address this limitation, we propose a connectivity-based approach called
bilateral connectivity network (BiconNet), which uses connectivity masks
together with saliency masks as labels for effective modeling of inter-pixel
relationships and object saliency. Moreover, we propose a bilateral voting
module to enhance the output connectivity map, and a novel edge feature
enhancement method that efficiently utilizes edge-specific features. Through
comprehensive experiments on five benchmark datasets, we demonstrate that our
proposed method can be plugged into any existing state-of-the-art
saliency-based SOD framework to improve its performance with negligible
parameter increase.

    

### [[2104.10824] Colonoscopy Polyp Detection and Classification: Dataset Creation and Comparative Evaluations](http://arxiv.org/abs/2104.10824)


  Colorectal cancer (CRC) is one of the most common types of cancer with a high
mortality rate. Colonoscopy is the preferred procedure for CRC screening and
has proven to be effective in reducing CRC mortality. Thus, a reliable
computer-aided polyp detection and classification system can significantly
increase the effectiveness of colonoscopy. In this paper, we create an
endoscopic dataset collected from various sources and annotate the ground truth
of polyp location and classification results with the help of experienced
gastroenterologists. The dataset can serve as a benchmark platform to train and
evaluate the machine learning models for polyp classification. We have also
compared the performance of eight state-of-the-art deep learning-based object
detection models. The results demonstrate that deep CNN models are promising in
CRC screening. This work can serve as a baseline for future research in polyp
detection and classification.

    

### [[2104.12156] Sequential composition of answer set programs](http://arxiv.org/abs/2104.12156)


  Non-monotonic reasoning is an essential part of human intelligence
prominently formalized in artificial intelligence research via answer set
programming. Describing complex objects as the composition of elementary ones
is a common strategy in computer science and science in general. Recently, the
author introduced the sequential composition of Horn logic programs for
syntactic program composition and decomposition in the context of logic-based
analogical reasoning and learning. This paper contributes to the foundations of
answer set programming and artificial intelligence by generalizing the
construction of composition from Horn to (propositional) answer set programs
containing negation as failure. This task turns out to be non-trivial due to
the intricate algebraic properties of composing negation as failure occurring
in rule bodies. Specifically, we show that the notion of composition gives rise
to a family of finite magmas and algebras, baptized {\em ASP magmas} and {\em
ASP algebras} in this paper. On the semantic side, we show that the van
Emden-Kowalski immediate consequence operator of a program can be represented
via composition, which allows us to compute the least model semantics of Horn
programs without any explicit reference to operators. As a result, we can
characterize answer sets algebraically, which bridges the conceptual gap
between the syntax and semantics of an answer set program in a mathematically
satisfactory way, and which provides an algebraic characterization of strong
and uniform equivalence. In a broader sense, this paper is a further step
towards an algebra of rule-based logical theories with applications to
logic-based analogical reasoning and learning, and in the future we plan to
adapt and generalize the methods of this paper to wider classes of formalisms,
most importantly to higher-order and disjunctive logic programs and extensions
thereof.

    

### [[2106.09018] End-to-End Semi-Supervised Object Detection with Soft Teacher](http://arxiv.org/abs/2106.09018)


  This paper presents an end-to-end semi-supervised object detection approach,
in contrast to previous more complex multi-stage methods. The end-to-end
training gradually improves pseudo label qualities during the curriculum, and
the more and more accurate pseudo labels in turn benefit object detection
training. We also propose two simple yet effective techniques within this
framework: a soft teacher mechanism where the classification loss of each
unlabeled bounding box is weighed by the classification score produced by the
teacher network; a box jittering approach to select reliable pseudo boxes for
the learning of box regression. On the COCO benchmark, the proposed approach
outperforms previous methods by a large margin under various labeling ratios,
i.e. 1\%, 5\% and 10\%. Moreover, our approach proves to perform also well when
the amount of labeled data is relatively large. For example, it can improve a
40.9 mAP baseline detector trained using the full COCO training set by +3.6
mAP, reaching 44.5 mAP, by leveraging the 123K unlabeled images of COCO. On the
state-of-the-art Swin Transformer based object detector (58.9 mAP on test-dev),
it can still significantly improve the detection accuracy by +1.5 mAP, reaching
60.4 mAP, and improve the instance segmentation accuracy by +1.2 mAP, reaching
52.4 mAP. Further incorporating with the Object365 pre-trained model, the
detection accuracy reaches 61.3 mAP and the instance segmentation accuracy
reaches 53.0 mAP, pushing the new state-of-the-art.

    

### [[2108.02110] Recursive Fusion and Deformable Spatiotemporal Attention for Video Compression Artifact Reduction](http://arxiv.org/abs/2108.02110)


  A number of deep learning based algorithms have been proposed to recover
high-quality videos from low-quality compressed ones. Among them, some restore
the missing details of each frame via exploring the spatiotemporal information
of neighboring frames. However, these methods usually suffer from a narrow
temporal scope, thus may miss some useful details from some frames outside the
neighboring ones. In this paper, to boost artifact removal, on the one hand, we
propose a Recursive Fusion (RF) module to model the temporal dependency within
a long temporal range. Specifically, RF utilizes both the current reference
frames and the preceding hidden state to conduct better spatiotemporal
compensation. On the other hand, we design an efficient and effective
Deformable Spatiotemporal Attention (DSTA) module such that the model can pay
more effort on restoring the artifact-rich areas like the boundary area of a
moving object. Extensive experiments show that our method outperforms the
existing ones on the MFQE 2.0 dataset in terms of both fidelity and perceptual
effect. Code is available at this https URL.

    

### [[2108.02852] Two Basic Queueing Models of Service Platforms in Digital Sharing Economy](http://arxiv.org/abs/2108.02852)


  This paper describes two basic queueing models of service platforms in
digital sharing economy by means of two different policies of platform matching
information. We show that the two queueing models of service platforms can be
expressed as the level-independent quasi birth-and-death (QBD) processes. Using
the proposed QBD processes, we provide a detailed analysis for the two queueing
models of service platforms, including the system stability, the average
stationary numbers of seekers and of idle owners, the expected sojourn time of
an arriving seeker, and the expected profits for both the service platform and
each owner. Finally, numerical examples are employed to verify our theoretical
results, and demonstrate how the performance measures of service platforms are
influenced by some key system parameters. We believe that the methodology and
results developed in this paper not only can be applied to develop a broad
class of queuing models of service platforms, but also will open a series of
promising innovative research on performance evaluation, optimal control and
queueing-game of service platforms and digital sharing economy.

    

### [[2108.02962] Dezyne: Paving the Way to Practical Formal Software Engineering](http://arxiv.org/abs/2108.02962)


  Designing software that controls industrial equipment is challenging,
especially due to its inherent concurrent nature. Testing this kind of event
driven control software is difficult and, due to the large number of possible
execution scenarios only a low dynamic test coverage is achieved in practice.
This in turn is undesirable due to the high cost of software failure for this
type of equipment.
In this paper we describe the Dezyne language and tooling; Dezyne is a
programming language aimed at software engineers designing large industrial
control software. We discuss its underlying two layered and compositional
approach that enables reaping the benefits of Formal Methods, hereby strongly
supporting guiding principles of software engineering. The core of Dezyne uses
the mCRL2 language and model-checker (Jan Friso Groote et al.) to verify the
correctness and completeness of all possible execution scenarios.
The IDE of Dezyne is based on the Language Server Protocol allowing a smooth
integration with e.g., Visual Studio Code, and Emacs, extended with several
automatically generated interactive graphical views. We report on the
introduction of Dezyne and its predecessor at several large high-tech equipment
manufacturers resulting in a decrease of software developing time and a major
decrease of reported field defects.

    

### [[2108.02967] Explaining Counterexamples with Giant-Step Assertion Checking](http://arxiv.org/abs/2108.02967)


  Identifying the cause of a proof failure during deductive verification of
programs is hard: it may be due to an incorrectness in the program, an
incompleteness in the program annotations, or an incompleteness of the prover.
The changes needed to resolve a proof failure depend on its category, but the
prover cannot provide any help on the categorisation. When using an SMT solver
to discharge a proof obligation, that solver can propose a model from a failed
attempt, from which a possible counterexample can be derived. But the
counterexample may be invalid, in which case it may add more confusion than
help. To check the validity of a counterexample and to categorise the proof
failure, we propose the comparison between the run-time assertion-checking
(RAC) executions under two different semantics, using the counterexample as an
oracle. The first RAC execution follows the normal program semantics, and a
violation of a program annotation indicates an incorrectness in the program.
The second RAC execution follows a novel "giant-step" semantics that does not
execute loops nor function calls but instead retrieves return values and values
of modified variables from the oracle. A violation of the program annotations
only observed under giant-step execution characterises an incompleteness of the
program annotations. We implemented this approach in the Why3 platform for
deductive program verification and evaluated it using examples from prior
literature.

    

### [[2108.02972] Disjunctive Delimited Control](http://arxiv.org/abs/2108.02972)


  Delimited control is a powerful mechanism for programming language extension
which has been recently proposed for Prolog (and implemented in SWI-Prolog). By
manipulating the control flow of a program from inside the language, it enables
the implementation of powerful features, such as tabling, without modifying the
internals of the Prolog engine. However, its current formulation is inadequate:
it does not capture Prolog's unique non-deterministic nature which allows
multiple ways to satisfy a goal.
This paper fully embraces Prolog's non-determinism with a novel interface for
disjunctive delimited control, which gives the programmer not only control over
the sequential (conjunctive) control flow, but also over the non-deterministic
control flow. We provide a meta-interpreter that conservatively extends Prolog
with delimited control and show that it enables a range of typical Prolog
features and extensions, now at the library level: findall, cut,
branch-and-bound optimisation, probabilistic programming,...

    

### [[2108.02995] Extracting functional programs from Coq, in Coq](http://arxiv.org/abs/2108.02995)


  We implement extraction of Coq programs to functional languages based on
MetaCoq's certified erasure. We extend the MetaCoq erasure output language with
typing information and use it as an intermediate representation, which we call
$\lambda^T_\square$. We complement the extraction functionality with a full
pipeline that includes several standard transformations (eta-expansion,
inlining, etc) implemented in a proof-generating manner along with a verified
optimisation pass removing unused arguments. We prove the pass correct wrt. a
conventional call-by-value operational semantics of functional languages. From
the optimised $\lambda^T_\square$ representation, we obtain code in two
functional smart contract languages (Liquidity and CameLIGO), the functional
language Elm, and a subset of the multi-paradigm language for systems
programming Rust. Rust is currently gaining popularity as a language for smart
contracts, and we demonstrate how our extraction can be used to extract smart
contract code for the Concordium network. The development is done in the
context of the ConCert framework that enables smart contract verification. We
contribute with two verified real-world smart contracts (boardroom voting and
escrow), which we use, among other examples, to exemplify the applicability of
the pipeline. In addition, we develop a verified web application and extract it
to fully functional Elm code. In total, this gives us a way to write
dependently typed programs in Coq, verify, and then extract them to several
target languages while retaining a small trusted computing base of only MetaCoq
and the pretty-printers into these languages.

    

### [[2108.03076] Certified Compilation of Financial Contracts](http://arxiv.org/abs/2108.03076)


  We present an extension to a certified financial contract management system
that allows for templated declarative financial contracts and for integration
with financial stochastic models through verified compilation into so-called
payoff-expressions. Such expressions readily allow for determining the value of
a contract in a given evaluation context, such as contexts created for
stochastic simulations. The templating mechanism is useful both at the contract
specification level, for writing generic reusable contracts, and for reuse of
code that, without the templating mechanism, needs to be recompiled for
different evaluation contexts. We report on the effect of using the certified
system in the context of a GPGPU-based Monte Carlo simulation engine for
pricing various over-the-counter (OTC) financial contracts. The full
contract-management system, including the payoff-language compilation, is
verified in the Coq proof assistant and certified Haskell code is extracted
from our Coq development along with Futhark code for use in a data-parallel
pricing engine.

    

### [[2108.03178] Transformation-Enabled Precondition Inference](http://arxiv.org/abs/2108.03178)


  Precondition inference is a non-trivial problem with important applications
in program analysis and verification. We present a novel iterative method for
automatically deriving preconditions for the safety and unsafety of programs.
Each iteration maintains over-approximations of the set of safe and unsafe
initial states; which are used to partition the program's initial states into
those known to be safe, known to be unsafe and unknown. We then construct
revised programs with those unknown initial states and iterate the procedure
until the approximations are disjoint or some termination criteria are met. An
experimental evaluation of the method on a set of software verification
benchmarks shows that it can infer precise preconditions (sometimes optimal)
that are not possible using previous methods.

    

### [[2008.06295] Proving Almost-Sure Termination of Probabilistic Programs via Incremental Pruning](http://arxiv.org/abs/2008.06295)


  The extension of classical imperative programs with real-valued random
variables and random branching gives rise to probabilistic programs. The
termination problem is one of the most fundamental liveness properties for such
programs. The qualitative (aka almost-sure) termination problem asks whether a
given program terminates with probability 1. Ranking functions provide a sound
and complete approach for termination of non-probabilistic programs, and their
extension to probabilistic programs is achieved via ranking supermartingales
(RSMs). RSMs have been extended to lexicographic RSMs to handle programs with
involved control-flow structure, as well as for compositional approach. There
are two key limitations of the existing RSM-based approaches: First, the
lexicographic RSM-based approach requires a strong nonnegativity assumption,
which need not always be satisfied. The second key limitation of the existing
RSM-based algorithmic approaches is that they rely on pre-computed invariants.
The main drawback of relying on pre-computed invariants is the
insufficiency-inefficiency trade-off: weak invariants might be insufficient for
RSMs to prove termination, while using strong invariants leads to inefficiency
in computing them. Our contributions are twofold: First, we show how to relax
the strong nonnegativity condition and still provide soundness guarantee for
almost-sure termination. Second, we present an incremental approach where the
process of computing lexicographic RSMs proceeds by iterative pruning of parts
of the program that were already shown to be terminating, in cooperation with a
safety prover. In particular, our technique does not rely on strong
pre-computed invariants. We present experimental results to show the
applicability of our approach to examples of probabilistic programs from the
literature.

    