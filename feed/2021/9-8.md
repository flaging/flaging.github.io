
## 2021-9-8

### [[2109.02757] Analysis of Dampers in Time-Sensitive Networks with Non-ideal Clocks](http://arxiv.org/abs/2109.02757)


  Dampers are devices that reduce delay jitter in the context of time-sensitive
networks, by delaying packets for the amount written in packet headers. Jitter
reduction is required by some real-time applications; beyond this, dampers have
the potential to solve the burstiness cascade problem of deterministic networks
in a scalable way, as they can be stateless. Dampers exist in several variants:
some apply only to earliest-deadline-first schedulers, whereas others can be
associated with any packet schedulers; some enforce FIFO ordering whereas some
others do not. Existing analyses of dampers are specific to some
implementations and some network configurations; also, they assume ideal,
non-realistic clocks. In this paper, we provide a taxonomy of all existing
dampers in general network settings and analyze their timing properties in
presence of non-ideal clocks. In particular, we give formulas for computing
residual jitter bounds of networks with dampers of any kind. We show that
non-FIFO dampers may cause reordering due to clock non-idealities and that the
combination of FIFO dampers with non-FIFO network elements may very negatively
affect the performance bounds. Our results can be used to analyze timing
properties and burstiness increase in any time-sensitive network, as we
illustrate on an industrial case-study.

    

### [[2109.02760] Tools for Network Traffic Generation -- A Quantitative Comparison](http://arxiv.org/abs/2109.02760)


  Network traffic generators are invaluable tools that allow for applied
experimentation to evaluate the performance of networks, infrastructure, and
security controls, by modelling and simulating the communication packets and
payloads that would be produced by machines and devices on the network.
Specifically for security applications, these tools can be used to consistently
simulate malicious activity on the network and test the components designed to
detect and mitigate malicious activities, in a highly reliable and customisable
way. However, despite the promising features, most of these tools have some
problems that can undermine the correctness of experiments. The accuracy of the
simulation results depends strongly on the performance and reliability of the
used generator. Thus, in this paper, we investigate the performance and
accuracy of three of the most reviewed network traffic generators in
literature, namely Cisco TRex, Ostinato and Genesids. Mainly, the comparative
experiments examine the strengths and limitations of these tools, which can
help the research community to choose the most suitable one to assess the
performance of their networks and security controls

    

### [[2109.02834] P3FA: Unified Unicast/Multicast Forwarding with Low Egress Diversities](http://arxiv.org/abs/2109.02834)


  Multicast is an efficient way to realize one-to-many group communications in
large-scale networks such as the Internet. However, the deployment of IP
multicast services over the Internet has not been as rapid as expected and
needed. Excepting the fatal defects in designing IPv4 address structure.
Another main reason that contributes to this slow deployment is the lack of
carrier-grade multicast-enabled switches and routers that can be as to scale as
their unicast counterparts. Implementing a high-performance switch/router
relies on a polynomial-time group membership query algorithm within the Packet
Forwarding Engines (PFEs) to determine whether or not a packet is forwarded
through an egress. Among these, Bloom filter (BF)-based and Residue Number
System (RNS)-based are being considered as two representations of the
membership query algorithms. However, both approaches suffer from some fatal
weaknesses such as space and time inefficiencies, especially for a
carrier-grade PFE with high port-density features. According to similar
properties of the prime theorem, we propose a simplified forwarding scheme in
this paper, named Per-Port Prime Filter Array (P3FA). The simulation results
indicate that the P3FA can significantly improve space efficiencies under
specific lower egress-diversities conditions. Under the same space constraints,
compared with the SVRF, the multicast time efficiencies, the unicast time
efficiency of the P3FA are respectively increased by 12x-17234x and 19x-2038x
at a range of port-densities 16-1024, but at the expense of hardware cost,
which increased by \r{ho}/2x. A PFE designer that attempts to adopt P3FA should
trade-off between required performance and cost.

    

### [[2109.02886] Bayesian Multidimensional Scaling for Location Awareness in Hybrid-Internet of Underwater Things](http://arxiv.org/abs/2109.02886)


  Localization of sensor nodes in the Internet of Underwater Things (IoUT) is
of considerable significance due to its various applications, such as
navigation, data tagging, and detection of underwater objects. Therefore, in
this paper, we propose a hybrid Bayesian multidimensional scaling (BMDS) based
localization technique that can work on a fully hybrid IoUT network where the
nodes can communicate using either optical, magnetic induction, and acoustic
technologies. These technologies are already used for communication in the
underwater environment; however, lacking localization solutions. Optical and
magnetic induction communication achieves higher data rates for short
communication. On the contrary, acoustic waves provide a low data rate for
long-range underwater communication. The proposed method collectively uses
optical, magnetic induction, and acoustic communication-based ranging to
estimate the underwater sensor nodes' final locations. Moreover, we also
analyze the proposed scheme by deriving the hybrid Cramer Rao lower bound
(HCRLB). Simulation results provide a complete comparative analysis of the
proposed method with the literature.

    

### [[2109.02907] A P2P Network Topology for Optimizing Partition Tolerance to Reach the CAP Guarantee Bound in Consortium Blockchains](http://arxiv.org/abs/2109.02907)


  Decentralized cryptocurrency systems, known as blockchains, have shown
promise as infrastructure for mutually distrustful parties to agree on
transactions safely. However, Bitcoin-derived blockchains and their variants
suffer from the limitations of the CAP Trilemma, which is difficult to be
solved through optimizing consensus protocols. Moreover, the P2P network of
blockchains have problems in efficiency and reliability without considering the
matching of physical and logical topologies. For the CAP Trilemma in consortium
blockchains, we propose a physical topology based on the multi-dimensional
hypercube with excellent partition tolerance in probability. On the other hand,
the general hypercube has advantages in solving the mismatch problems in P2P
networks. The general topology is further extended to a hierarchical recursive
topology with more medium links or short links to balance the reliability
requirements and the costs of physical network. We prove that the hypercube
topology has better partition tolerance than the regular rooted tree topology
and the ring lattice topology, and effectively fits the upper-layer protocols.
As a result, blockchains constructed by the proposed topology can reach the CAP
guarantee bound through adopting the proper transmission and the consensus
protocols protocols that have strong consistency and availability.

    

### [[2109.02967] Wide Area Network Autoscaling for Cloud Applications](http://arxiv.org/abs/2109.02967)


  Modern cloud orchestrators like Kubernetes provide a versatile and robust way
to host applications at scale. One of their key features is autoscaling, which
automatically adjusts cloud resources (compute, memory, storage) in order to
adapt to the demands of applications. However, the scope of cloud autoscaling
is limited to the datacenter hosting the cloud and it doesn't apply uniformly
to the allocation of network resources. In I/O-constrained or data-in-motion
use cases this can lead to severe performance degradation for the application.
For example, when the load on a cloud service increases and the Wide Area
Network (WAN) connecting the datacenter to the Internet becomes saturated, the
application flows experience an increase in delay and loss. In many cases this
is dealt with overprovisioning network capacity, which introduces additional
costs and inefficiencies.
On the other hand, thanks to the concept of "Network as Code", the WAN
exposes a set of APIs that can be used to dynamically allocate and de-allocate
capacity on-demand. In this paper we propose extending the concept of cloud
autoscaling into the network to address this limitation. This way, applications
running in the cloud can communicate their networking requirements, like
bandwidth or traffic profile, to a Software-Defined Networking (SDN) controller
or Network as a Service (NaaS) platform. Moreover, we aim to define the
concepts of vertical and horizontal autoscaling applied to networking. We
present a prototype that automatically allocates bandwidth to the underlay
network, according to the requirements of the applications hosted in
Kubernetes. Finally, we discuss open research challenges.

    

### [[2109.03011] Understanding Model Drift in a Large Cellular Network](http://arxiv.org/abs/2109.03011)


  Operational networks are increasingly using machine learning models for a
variety of tasks, including detecting anomalies, inferring application
performance, and forecasting demand. Accurate models are important, yet
accuracy can degrade over time due to concept drift, whereby either the
characteristics of the data change over time (data drift) or the relationship
between the features and the target predictor change over time (model drift).
Drift is important to detect because changes in properties of the underlying
data or relationships to the target prediction can require model retraining,
which can be time-consuming and expensive. Concept drift occurs in operational
networks for a variety of reasons, ranging from software upgrades to
seasonality to changes in user behavior. Yet, despite the prevalence of drift
in networks, its extent and effects on prediction accuracy have not been
extensively studied. This paper presents an initial exploration into concept
drift in a large cellular network in the United States for a major metropolitan
area in the context of demand forecasting. We find that concept drift arises
largely due to data drift, and it appears across different key performance
indicators (KPIs), models, training set sizes, and time intervals. We identify
the sources of concept drift for the particular problem of forecasting downlink
volume. Weekly and seasonal patterns introduce both high and low-frequency
model drift, while disasters and upgrades result in sudden drift due to
exogenous shocks. Regions with high population density, lower traffic volumes,
and higher speeds also tend to correlate with more concept drift. The features
that contribute most significantly to concept drift are User Equipment (UE)
downlink packets, UE uplink packets, and Real-time Transport Protocol (RTP)
total received packets.

    

### [[2109.03032] A Just-In-Time Networking Framework for Minimizing Request-Response Latency of Wireless Time-Sensitive Applications](http://arxiv.org/abs/2109.03032)


  This paper puts forth a networking paradigm, referred to as just-in-time
(JIT) communication, to support client-server applications with stringent
request-response latency requirement. Of interest is not just the round-trip
delay of the network, but the actual request-response latency experienced by
the application. The JIT framework contains two salient features. At the client
side, the communication layer will 'pull' a request from the client just when
there is an upcoming transmission opportunity from the network. This ensures
that the request contains information that is as fresh as possible (e.g., a
sensor reading obtained just before the transmission opportunity). At the
server side, the network ascertains that the server, after receiving and
processing the request to generate a response (e.g., a control command to be
sent to the client), will have a transmission opportunity at just this time. We
realize the JIT system, including the protocol stack, over a
Time-Division-Multiple-Access (TDMA) network implemented on a System-on-Chip
(SoC) platform. We prove that a TDMA network with a power-of-2 time slots per
superframe is optimal for realizing the server-side JIT function. Our
experimental results validate that JIT networks can yield significantly lower
request-response latency than networks without JIT support can.

    

### [[2109.03180] First Responders Got Wings: UAVs to the Rescue of Localization Operations in Beyond 5G Systems](http://arxiv.org/abs/2109.03180)


  Natural and human-made disasters have dramatically increased during the last
decades. Given the strong relationship between first responders localization
time and the final number of deaths, the modernization of search-and-rescue
operations has become imperative. In this context, Unmanned Aerial Vehicles
(UAVs)-based solutions are the most promising candidates to take up on the
localization challenge by leveraging on emerging technologies such as:
Artificial Intelligence (AI), Reconfigurable Intelligent Surfaces (RIS) and
Orthogonal Time Frequency Space (OTFS) modulations. In this paper, we
capitalize on such recently available techniques by shedding light on the main
challenges and future opportunities to boost the localization performance of
state-of-the-art techniques to give birth to unprecedentedly effective missing
victims localization solutions.

    

### [[2109.03210] Routing Emergency Vehicles in Arterial Road Networks using Real-time Mixed Criticality Systems*](http://arxiv.org/abs/2109.03210)


  Reducing the response time of Emergency Vehicles (EVs) has an undoubted
advantage in saving life and property. Implementing pre-emption can aid in
achieving it. EVs get unobstructed movement via pre-emption, usually by
altering traffic signals and giving a green wave throughout the route. This
approach of absolute pre-emption effects adversely on regular traffic by
imposing unnecessary waiting. In this paper, we propose a novel emergency
vehicle pre-emption (EVP) algorithm implemented in the Vehicular Ad-hoc Network
(VANET) that can reduce the imposed undesirable waiting time, but still
ascertains EVs meet target response time. We introduce mixed-criticality
real-time system scheduling concept where different level of emergencies is
mapped with different criticality levels and assign certain success assurance
level to respective criticality. We implemented the EVP algorithm for an
arterial traffic network and leveraged the use of valuable information that
gets transmitted via VANET to make critical decisions. The proposed algorithm
can significantly reduce the average waiting time of regular traffic. It also
ascertains all EVs with different level of criticality meet target response
time respective to their assurance level.

    

### [[2109.02639] On the Out-of-distribution Generalization of Probabilistic Image Modelling](http://arxiv.org/abs/2109.02639)


  Out-of-distribution (OOD) detection and lossless compression constitute two
problems that can be solved by the training of probabilistic models on a first
dataset with subsequent likelihood evaluation on a second dataset, where data
distributions differ. By defining the generalization of probabilistic models in
terms of likelihood we show that, in the case of image models, the OOD
generalization ability is dominated by local features. This motivates our
proposal of a Local Autoregressive model that exclusively models local image
features towards improving OOD performance. We apply the proposed model to OOD
detection tasks and achieve state-of-the-art unsupervised OOD detection
performance without the introduction of additional data. Additionally, we
employ our model to build a new lossless image compressor: NeLLoC (Neural Local
Lossless Compressor) and report state-of-the-art compression rates and model
size.

    

### [[2109.02645] Backpropagation and fuzzy algorithm Modelling to Resolve Blood Supply Chain Issues in the Covid-19 Pandemic](http://arxiv.org/abs/2109.02645)


  Bloodstock shortages and its uncertain demand has become a major problem for
all countries worldwide. Therefore, this study aims to provide solution to the
issues of blood distribution during the Covid-19 Pandemic at Bengkulu,
Indonesia. The Backpropagation algorithm was used to improve the possibility of
discovering available and potential donors. Furthermore, the distances, age,
and length of donation were measured to obtain the right person to donate blood
when it needed. The Backpropagation uses three input layers to classify
eligible donors, namely age, body, weight, and bias. In addition, the system
through its query automatically counts the variables via the Fuzzy Tahani and
simultaneously access the vast database.

    

### [[2109.02691] SS-BERT: Mitigating Identity Terms Bias in Toxic Comment Classification by Utilising the Notion of "Subjectivity" and "Identity Terms"](http://arxiv.org/abs/2109.02691)


  Toxic comment classification models are often found biased toward identity
terms which are terms characterizing a specific group of people such as
"Muslim" and "black". Such bias is commonly reflected in false-positive
predictions, i.e. non-toxic comments with identity terms. In this work, we
propose a novel approach to tackle such bias in toxic comment classification,
leveraging the notion of subjectivity level of a comment and the presence of
identity terms. We hypothesize that when a comment is made about a group of
people that is characterized by an identity term, the likelihood of that
comment being toxic is associated with the subjectivity level of the comment,
i.e. the extent to which the comment conveys personal feelings and opinions.
Building upon the BERT model, we propose a new structure that is able to
leverage these features, and thoroughly evaluate our model on 4 datasets of
varying sizes and representing different social media platforms. The results
show that our model can consistently outperform BERT and a SOTA model devised
to address identity term bias in a different way, with a maximum improvement in
F1 of 2.43% and 1.91% respectively.

    

### [[2109.02692] Machine Learning: Challenges, Limitations, and Compatibility for Audio Restoration Processes](http://arxiv.org/abs/2109.02692)


  In this paper machine learning networks are explored for their use in
restoring degraded and compressed speech audio. The project intent is to build
a new trained model from voice data to learn features of compression
artifacting distortion introduced by data loss from lossy compression and
resolution loss with an existing algorithm presented in SEGAN: Speech
Enhancement Generative Adversarial Network. The resulting generator from the
model was then to be used to restore degraded speech audio. This paper details
an examination of the subsequent compatibility and operational issues presented
by working with deprecated code, which obstructed the trained model from
successfully being developed. This paper further serves as an examination of
the challenges, limitations, and compatibility in the current state of machine
learning.

    

### [[2109.02703] Large-Scale System Identification Using a Randomized SVD](http://arxiv.org/abs/2109.02703)


  Learning a dynamical system from input/output data is a fundamental task in
the control design pipeline. In the partially observed setting there are two
components to identification: parameter estimation to learn the Markov
parameters, and system realization to obtain a state space model. In both
sub-problems it is implicitly assumed that standard numerical algorithms such
as the singular value decomposition (SVD) can be easily and reliably computed.
When trying to fit a high-dimensional model to data, for example in the
cyber-physical system setting, even computing an SVD is intractable. In this
work we show that an approximate matrix factorization obtained using randomized
methods can replace the standard SVD in the realization algorithm while
maintaining the non-asymptotic (in data-set size) performance and robustness
guarantees of classical methods. Numerical examples illustrate that for large
system models, this is the only method capable of producing a model.

    

### [[2109.02704] gen2Out: Detecting and Ranking Generalized Anomalies](http://arxiv.org/abs/2109.02704)


  In a cloud of m-dimensional data points, how would we spot, as well as rank,
both single-point- as well as group- anomalies? We are the first to generalize
anomaly detection in two dimensions: The first dimension is that we handle both
point-anomalies, as well as group-anomalies, under a unified view -- we shall
refer to them as generalized anomalies. The second dimension is that gen2Out
not only detects, but also ranks, anomalies in suspiciousness order. Detection,
and ranking, of anomalies has numerous applications: For example, in EEG
recordings of an epileptic patient, an anomaly may indicate a seizure; in
computer network traffic data, it may signify a power failure, or a DoS/DDoS
attack. We start by setting some reasonable axioms; surprisingly, none of the
earlier methods pass all the axioms. Our main contribution is the gen2Out
algorithm, that has the following desirable properties: (a) Principled and
Sound anomaly scoring that obeys the axioms for detectors, (b) Doubly-general
in that it detects, as well as ranks generalized anomaly -- both point- and
group-anomalies, (c) Scalable, it is fast and scalable, linear on input size.
(d) Effective, experiments on real-world epileptic recordings (200GB)
demonstrate effectiveness of gen2Out as confirmed by clinicians. Experiments on
27 real-world benchmark datasets show that gen2Out detects ground truth groups,
matches or outperforms point-anomaly baseline algorithms on accuracy, with no
competition for group-anomalies and requires about 2 minutes for 1 million data
points on a stock machine.

    

### [[2109.02711] Graph Attention Layer Evolves Semantic Segmentation for Road Pothole Detection: A Benchmark and Algorithms](http://arxiv.org/abs/2109.02711)


  Existing road pothole detection approaches can be classified as computer
vision-based or machine learning-based. The former approaches typically employ
2-D image analysis/understanding or 3-D point cloud modeling and segmentation
algorithms to detect road potholes from vision sensor data. The latter
approaches generally address road pothole detection using convolutional neural
networks (CNNs) in an end-to-end manner. However, road potholes are not
necessarily ubiquitous and it is challenging to prepare a large well-annotated
dataset for CNN training. In this regard, while computer vision-based methods
were the mainstream research trend in the past decade, machine learning-based
methods were merely discussed. Recently, we published the first stereo
vision-based road pothole detection dataset and a novel disparity
transformation algorithm, whereby the damaged and undamaged road areas can be
highly distinguished. However, there are no benchmarks currently available for
state-of-the-art (SoTA) CNNs trained using either disparity images or
transformed disparity images. Therefore, in this paper, we first discuss the
SoTA CNNs designed for semantic segmentation and evaluate their performance for
road pothole detection with extensive experiments. Additionally, inspired by
graph neural network (GNN), we propose a novel CNN layer, referred to as graph
attention layer (GAL), which can be easily deployed in any existing CNN to
optimize image feature representations for semantic segmentation. Our
experiments compare GAL-DeepLabv3+, our best-performing implementation, with
nine SoTA CNNs on three modalities of training data: RGB images, disparity
images, and transformed disparity images. The experimental results suggest that
our proposed GAL-DeepLabv3+ achieves the best overall pothole detection
accuracy on all training data modalities.

    

### [[2109.02715] Individual Mobility Prediction via Attentive Marked Temporal Point Processes](http://arxiv.org/abs/2109.02715)


  Individual mobility prediction is an essential task for transportation demand
management and traffic system operation. There exist a large body of works on
modeling location sequence and predicting the next location of users; however,
little attention is paid to the prediction of the next trip, which is governed
by the strong spatiotemporal dependencies between diverse attributes, including
trip start time $t$, origin $o$, and destination $d$. To fill this gap, in this
paper we propose a novel point process-based model -- Attentive Marked temporal
point processes (AMTPP) -- to model human mobility and predict the whole trip
$(t,o,d)$ in a joint manner. To encode the influence of history trips, AMTPP
employs the self-attention mechanism with a carefully designed positional
embedding to capture the daily/weekly periodicity and regularity in individual
travel behavior. Given the unique peaked nature of inter-event time in human
behavior, we use an asymmetric log-Laplace mixture distribution to precisely
model the distribution of trip start time $t$. Furthermore, an
origin-destination (OD) matrix learning block is developed to model the
relationship between every origin and destination pair. Experimental results on
two large metro trip datasets demonstrate the superior performance of AMTPP.

    

### [[2109.02717] Iterative Pseudo-Labeling with Deep Feature Annotation and Confidence-Based Sampling](http://arxiv.org/abs/2109.02717)


  Training deep neural networks is challenging when large and annotated
datasets are unavailable. Extensive manual annotation of data samples is
time-consuming, expensive, and error-prone, notably when it needs to be done by
experts. To address this issue, increased attention has been devoted to
techniques that propagate uncertain labels (also called pseudo labels) to large
amounts of unsupervised samples and use them for training the model. However,
these techniques still need hundreds of supervised samples per class in the
training set and a validation set with extra supervised samples to tune the
model. We improve a recent iterative pseudo-labeling technique, Deep Feature
Annotation (DeepFA), by selecting the most confident unsupervised samples to
iteratively train a deep neural network. Our confidence-based sampling strategy
relies on only dozens of annotated training samples per class with no
validation set, considerably reducing user effort in data annotation. We first
ascertain the best configuration for the baseline -- a self-trained deep neural
network -- and then evaluate our confidence DeepFA for different confidence
thresholds. Experiments on six datasets show that DeepFA already outperforms
the self-trained baseline, but confidence DeepFA can considerably outperform
the original DeepFA and the baseline.

    

### [[2109.02723] OKSP: A Novel Deep Learning Automatic Event Detection Pipeline for Seismic Monitoringin Costa Rica](http://arxiv.org/abs/2109.02723)


  Small magnitude earthquakes are the most abundant but the most difficult to
locate robustly and well due to their low amplitudes and high frequencies
usually obscured by heterogeneous noise sources. They highlight crucial
information about the stress state and the spatio-temporal behavior of fault
systems during the earthquake cycle, therefore, its full characterization is
then crucial for improving earthquake hazard assessment. Modern DL algorithms
along with the increasing computational power are exploiting the continuously
growing seismological databases, allowing scientists to improve the
completeness for earthquake catalogs, systematically detecting smaller
magnitude earthquakes and reducing the errors introduced mainly by human
intervention. In this work, we introduce OKSP, a novel automatic earthquake
detection pipeline for seismic monitoring in Costa Rica. Using Kabre
supercomputer from the Costa Rica High Technology Center, we applied OKSP to
the day before and the first 5 days following the Puerto Armuelles, M6.5,
earthquake that occurred on 26 June, 2019, along the Costa Rica-Panama border
and found 1100 more earthquakes previously unidentified by the Volcanological
and Seismological Observatory of Costa Rica. From these events, a total of 23
earthquakes with magnitudes below 1.0 occurred a day to hours prior to the
mainshock, shedding light about the rupture initiation and earthquake
interaction leading to the occurrence of this productive seismic sequence. Our
observations show that for the study period, the model was 100% exhaustive and
82% precise, resulting in an F1 score of 0.90. This effort represents the very
first attempt for automatically detecting earthquakes in Costa Rica using deep
learning methods and demonstrates that, in the near future, earthquake
monitoring routines will be carried out entirely by AI algorithms.

    

### [[2109.02724] Bringing a Ruler Into the Black Box: Uncovering Feature Impact from Individual Conditional Expectation Plots](http://arxiv.org/abs/2109.02724)


  As machine learning systems become more ubiquitous, methods for understanding
and interpreting these models become increasingly important. In particular,
practitioners are often interested both in what features the model relies on
and how the model relies on them--the feature's impact on model predictions.
Prior work on feature impact including partial dependence plots (PDPs) and
Individual Conditional Expectation (ICE) plots has focused on a visual
interpretation of feature impact. We propose a natural extension to ICE plots
with ICE feature impact, a model-agnostic, performance-agnostic feature impact
metric drawn out from ICE plots that can be interpreted as a close analogy to
linear regression coefficients. Additionally, we introduce an in-distribution
variant of ICE feature impact to vary the influence of out-of-distribution
points as well as heterogeneity and non-linearity measures to characterize
feature impact. Lastly, we demonstrate ICE feature impact's utility in several
tasks using real-world data.

    

### [[2109.02748] Zero-Shot Open Set Detection by Extending CLIP](http://arxiv.org/abs/2109.02748)


  In a regular open set detection problem, samples of known classes (also
called closed set classes) are used to train a special classifier. In testing,
the classifier can (1) classify the test samples of known classes to their
respective classes and (2) also detect samples that do not belong to any of the
known classes (we say they belong to some unknown or open set classes). This
paper studies the problem of zero-shot open-set detection, which still performs
the same two tasks in testing but has no training except using the given known
class names. This paper proposes a novel and yet simple method (called ZO-CLIP)
to solve the problem. ZO-CLIP builds on top of the recent advances in zero-shot
classification through multi-modal representation learning. It first extends
the pre-trained multi-modal model CLIP by training a text-based image
description generator on top of CLIP. In testing, it uses the extended model to
generate some candidate unknown class names for each test sample and computes a
confidence score based on both the known class names and candidate unknown
class names for zero-shot open set detection. Experimental results on 5
benchmark datasets for open set detection confirm that ZO-CLIP outperforms the
baselines by a large margin.

    

### [[2109.02749] Pano3D: A Holistic Benchmark and a Solid Baseline for $360^o$ Depth Estimation](http://arxiv.org/abs/2109.02749)


  Pano3D is a new benchmark for depth estimation from spherical panoramas. It
aims to assess performance across all depth estimation traits, the primary
direct depth estimation performance targeting precision and accuracy, and also
the secondary traits, boundary preservation, and smoothness. Moreover, Pano3D
moves beyond typical intra-dataset evaluation to inter-dataset performance
assessment. By disentangling the capacity to generalize to unseen data into
different test splits, Pano3D represents a holistic benchmark for $360^o$ depth
estimation. We use it as a basis for an extended analysis seeking to offer
insights into classical choices for depth estimation. This results in a solid
baseline for panoramic depth that follow-up works can build upon to steer
future progress.

    

### [[2109.02752] Training Deep Networks from Zero to Hero: avoiding pitfalls and going beyond](http://arxiv.org/abs/2109.02752)


  Training deep neural networks may be challenging in real world data. Using
models as black-boxes, even with transfer learning, can result in poor
generalization or inconclusive results when it comes to small datasets or
specific applications. This tutorial covers the basic steps as well as more
recent options to improve models, in particular, but not restricted to,
supervised learning. It can be particularly useful in datasets that are not as
well-prepared as those in challenges, and also under scarce annotation and/or
small data. We describe basic procedures: as data preparation, optimization and
transfer learning, but also recent architectural choices such as use of
transformer modules, alternative convolutional layers, activation functions,
wide and deep networks, as well as training procedures including as curriculum,
contrastive and self-supervised learning.

    

### [[2109.02755] Motion Artifact Reduction In Photoplethysmography For Reliable Signal Selection](http://arxiv.org/abs/2109.02755)


  Photoplethysmography (PPG) is a non-invasive and economical technique to
extract vital signs of the human body. Although it has been widely used in
consumer and research grade wrist devices to track a user's physiology, the PPG
signal is very sensitive to motion which can corrupt the signal's quality.
Existing Motion Artifact (MA) reduction techniques have been developed and
evaluated using either synthetic noisy signals or signals collected during
high-intensity activities - both of which are difficult to generalize for
real-life scenarios. Therefore, it is valuable to collect realistic PPG signals
while performing Activities of Daily Living (ADL) to develop practical signal
denoising and analysis methods. In this work, we propose an automatic pseudo
clean PPG generation process for reliable PPG signal selection. For each noisy
PPG segment, the corresponding pseudo clean PPG reduces the MAs and contains
rich temporal details depicting cardiac features. Our experimental results show
that 71% of the pseudo clean PPG collected from ADL can be considered as high
quality segment where the derived MAE of heart rate and respiration rate are
1.46 BPM and 3.93 BrPM, respectively. Therefore, our proposed method can
determine the reliability of the raw noisy PPG by considering quality of the
corresponding pseudo clean PPG signal.

    

### [[2109.02765] Robustness and Generalization via Generative Adversarial Training](http://arxiv.org/abs/2109.02765)


  While deep neural networks have achieved remarkable success in various
computer vision tasks, they often fail to generalize to new domains and subtle
variations of input images. Several defenses have been proposed to improve the
robustness against these variations. However, current defenses can only
withstand the specific attack used in training, and the models often remain
vulnerable to other input variations. Moreover, these methods often degrade
performance of the model on clean images and do not generalize to out-of-domain
samples. In this paper we present Generative Adversarial Training, an approach
to simultaneously improve the model's generalization to the test set and
out-of-domain samples as well as its robustness to unseen adversarial attacks.
Instead of altering a low-level pre-defined aspect of images, we generate a
spectrum of low-level, mid-level and high-level changes using generative models
with a disentangled latent space. Adversarial training with these examples
enable the model to withstand a wide range of attacks by observing a variety of
input alterations during training. We show that our approach not only improves
performance of the model on clean images and out-of-domain samples but also
makes it robust against unforeseen attacks and outperforms prior work. We
validate effectiveness of our method by demonstrating results on various tasks
such as classification, segmentation and object detection.

    

### [[2109.02773] Complementing Handcrafted Features with Raw Waveform Using a Light-weight Auxiliary Model](http://arxiv.org/abs/2109.02773)


  An emerging trend in audio processing is capturing low-level speech
representations from raw waveforms. These representations have shown promising
results on a variety of tasks, such as speech recognition and speech
separation. Compared to handcrafted features, learning speech features via
backpropagation provides the model greater flexibility in how it represents
data for different tasks theoretically. However, results from empirical study
shows that, in some tasks, such as voice spoof detection, handcrafted features
are more competitive than learned features. Instead of evaluating handcrafted
features and raw waveforms independently, this paper proposes an Auxiliary
Rawnet model to complement handcrafted features with features learned from raw
waveforms. A key benefit of the approach is that it can improve accuracy at a
relatively low computational cost. The proposed Auxiliary Rawnet model is
tested using the ASVspoof 2019 dataset and the results from this dataset
indicate that a light-weight waveform encoder can potentially boost the
performance of handcrafted-features-based encoders in exchange for a small
amount of additional computational work.

    

### [[2109.02774] FastAudio: A Learnable Audio Front-End for Spoof Speech Detection](http://arxiv.org/abs/2109.02774)


  Voice assistants, such as smart speakers, have exploded in popularity. It is
currently estimated that the smart speaker adoption rate has exceeded 35% in
the US adult population. Manufacturers have integrated speaker identification
technology, which attempts to determine the identity of the person speaking, to
provide personalized services to different members of the same family. Speaker
identification can also play an important role in controlling how the smart
speaker is used. For example, it is not critical to correctly identify the user
when playing music. However, when reading the user's email out loud, it is
critical to correctly verify the speaker that making the request is the
authorized user. Speaker verification systems, which authenticate the speaker
identity, are therefore needed as a gatekeeper to protect against various
spoofing attacks that aim to impersonate the enrolled user. This paper compares
popular learnable front-ends which learn the representations of audio by joint
training with downstream tasks (End-to-End). We categorize the front-ends by
defining two generic architectures and then analyze the filtering stages of
both types in terms of learning constraints. We propose replacing fixed
filterbanks with a learnable layer that can better adapt to anti-spoofing
tasks. The proposed FastAudio front-end is then tested with two popular
back-ends to measure the performance on the LA track of the ASVspoof 2019
dataset. The FastAudio front-end achieves a relative improvement of 27% when
compared with fixed front-ends, outperforming all other learnable front-ends on
this task.

    

### [[2109.02785] Analysis of MRI Biomarkers for Brain Cancer Survival Prediction](http://arxiv.org/abs/2109.02785)


  Prediction of Overall Survival (OS) of brain cancer patients from multi-modal
MRI is a challenging field of research. Most of the existing literature on
survival prediction is based on Radiomic features, which does not consider
either non-biological factors or the functional neurological status of the
patient(s). Besides, the selection of an appropriate cut-off for survival and
the presence of censored data create further problems. Application of deep
learning models for OS prediction is also limited due to the lack of large
annotated publicly available datasets. In this scenario we analyse the
potential of two novel neuroimaging feature families, extracted from brain
parcellation atlases and spatial habitats, along with classical radiomic and
geometric features; to study their combined predictive power for analysing
overall survival. A cross validation strategy with grid search is proposed to
simultaneously select and evaluate the most predictive feature subset based on
its predictive power. A Cox Proportional Hazard (CoxPH) model is employed for
univariate feature selection, followed by the prediction of patient-specific
survival functions by three multivariate parsimonious models viz. Coxnet,
Random survival forests (RSF) and Survival SVM (SSVM). The brain cancer MRI
data used for this research was taken from two open-access collections TCGA-GBM
and TCGA-LGG available from The Cancer Imaging Archive (TCIA). Corresponding
survival data for each patient was downloaded from The Cancer Genome Atlas
(TCGA). A high cross validation $C-index$ score of $0.82\pm.10$ was achieved
using RSF with the best $24$ selected features. Age was found to be the most
important biological predictor. There were $9$, $6$, $6$ and $2$ features
selected from the parcellation, habitat, radiomic and region-based feature
groups respectively.

    

### [[2109.02791] Safe-Critical Modular Deep Reinforcement Learning with Temporal Logic through Gaussian Processes and Control Barrier Functions](http://arxiv.org/abs/2109.02791)


  Reinforcement learning (RL) is a promising approach and has limited success
towards real-world applications, because ensuring safe exploration or
facilitating adequate exploitation is a challenges for controlling robotic
systems with unknown models and measurement uncertainties. Such a learning
problem becomes even more intractable for complex tasks over continuous space
(state-space and action-space). In this paper, we propose a learning-based
control framework consisting of several aspects: (1) linear temporal logic
(LTL) is leveraged to facilitate complex tasks over an infinite horizons which
can be translated to a novel automaton structure; (2) we propose an innovative
reward scheme for RL-agent with the formal guarantee such that global optimal
policies maximize the probability of satisfying the LTL specifications; (3)
based on a reward shaping technique, we develop a modular policy-gradient
architecture utilizing the benefits of automaton structures to decompose
overall tasks and facilitate the performance of learned controllers; (4) by
incorporating Gaussian Processes (GPs) to estimate the uncertain dynamic
systems, we synthesize a model-based safeguard using Exponential Control
Barrier Functions (ECBFs) to address problems with high-order relative degrees.
In addition, we utilize the properties of LTL automatons and ECBFs to construct
a guiding process to further improve the efficiency of exploration. Finally, we
demonstrate the effectiveness of the framework via several robotic
environments. And we show such an ECBF-based modular deep RL algorithm achieves
near-perfect success rates and guard safety with a high probability confidence
during training.

    

### [[2109.02797] Puzzle Solving without Search or Human Knowledge: An Unnatural Language Approach](http://arxiv.org/abs/2109.02797)


  The application of Generative Pre-trained Transformer (GPT-2) to learn
text-archived game notation provides a model environment for exploring sparse
reward gameplay. The transformer architecture proves amenable to training on
solved text archives describing mazes, Rubik's Cube, and Sudoku solvers. The
method benefits from fine-tuning the transformer architecture to visualize
plausible strategies derived outside any guidance from human heuristics or
domain expertise. The large search space ($>10^{19}$) for the games provides a
puzzle environment in which the solution has few intermediate rewards and a
final move that solves the challenge.

    

### [[2109.02801] ArGoT: A Glossary of Terms extracted from the arXiv](http://arxiv.org/abs/2109.02801)


  We introduce ArGoT, a data set of mathematical terms extracted from the
articles hosted on the arXiv website. A term is any mathematical concept
defined in an article. Using labels in the article's source code and examples
from other popular math websites, we mine all the terms in the arXiv data and
compile a comprehensive vocabulary of mathematical terms. Each term can be then
organized in a dependency graph by using the term's definitions and the arXiv's
metadata. Using both hyperbolic and standard word embeddings, we demonstrate
how this structure is reflected in the text's vector representation and how
they capture relations of entailment in mathematical concepts. This data set is
part of an ongoing effort to align natural mathematical text with existing
Interactive Theorem Prover Libraries (ITPs) of formally verified statements.

    

### [[2109.02808] A Scalable AI Approach for Clinical Trial Cohort Optimization](http://arxiv.org/abs/2109.02808)


  FDA has been promoting enrollment practices that could enhance the diversity
of clinical trial populations, through broadening eligibility criteria.
However, how to broaden eligibility remains a significant challenge. We propose
an AI approach to Cohort Optimization (AICO) through transformer-based natural
language processing of the eligibility criteria and evaluation of the criteria
using real-world data. The method can extract common eligibility criteria
variables from a large set of relevant trials and measure the generalizability
of trial designs to real-world patients. It overcomes the scalability limits of
existing manual methods and enables rapid simulation of eligibility criteria
design for a disease of interest. A case study on breast cancer trial design
demonstrates the utility of the method in improving trial generalizability.

    

### [[2109.02820] Few-shot Learning via Dependency Maximization and Instance Discriminant Analysis](http://arxiv.org/abs/2109.02820)


  We study the few-shot learning (FSL) problem, where a model learns to
recognize new objects with extremely few labeled training data per category.
Most of previous FSL approaches resort to the meta-learning paradigm, where the
model accumulates inductive bias through learning many training tasks so as to
solve a new unseen few-shot task. In contrast, we propose a simple approach to
exploit unlabeled data accompanying the few-shot task for improving few-shot
performance. Firstly, we propose a Dependency Maximization method based on the
Hilbert-Schmidt norm of the cross-covariance operator, which maximizes the
statistical dependency between the embedded feature of those unlabeled data and
their label predictions, together with the supervised loss over the support
set. We then use the obtained model to infer the pseudo-labels for those
unlabeled data. Furthermore, we propose anInstance Discriminant Analysis to
evaluate the credibility of each pseudo-labeled example and select the most
faithful ones into an augmented support set to retrain the model as in the
first step. We iterate the above process until the pseudo-labels for the
unlabeled data becomes stable. Following the standard transductive and
semi-supervised FSL setting, our experiments show that the proposed method
out-performs previous state-of-the-art methods on four widely used benchmarks,
including mini-ImageNet, tiered-ImageNet, CUB, and CIFARFS.

    

### [[2109.02832] Besov Function Approximation and Binary Classification on Low-Dimensional Manifolds Using Convolutional Residual Networks](http://arxiv.org/abs/2109.02832)


  Most of existing statistical theories on deep neural networks have sample
complexities cursed by the data dimension and therefore cannot well explain the
empirical success of deep learning on high-dimensional data. To bridge this
gap, we propose to exploit low-dimensional geometric structures of the real
world data sets. We establish theoretical guarantees of convolutional residual
networks (ConvResNet) in terms of function approximation and statistical
estimation for binary classification. Specifically, given the data lying on a
$d$-dimensional manifold isometrically embedded in $\mathbb{R}^D$, we prove
that if the network architecture is properly chosen, ConvResNets can (1)
approximate Besov functions on manifolds with arbitrary accuracy, and (2) learn
a classifier by minimizing the empirical logistic risk, which gives an excess
risk in the order of $n^{-\frac{s}{2s+2(s\vee d)}}$, where $s$ is a smoothness
parameter. This implies that the sample complexity depends on the intrinsic
dimension $d$, instead of the data dimension $D$. Our results demonstrate that
ConvResNets are adaptive to low-dimensional structures of data sets.

    

### [[2109.02836] Trojan Signatures in DNN Weights](http://arxiv.org/abs/2109.02836)


  Deep neural networks have been shown to be vulnerable to backdoor, or trojan,
attacks where an adversary has embedded a trigger in the network at training
time such that the model correctly classifies all standard inputs, but
generates a targeted, incorrect classification on any input which contains the
trigger. In this paper, we present the first ultra light-weight and highly
effective trojan detection method that does not require access to the
training/test data, does not involve any expensive computations, and makes no
assumptions on the nature of the trojan trigger. Our approach focuses on
analysis of the weights of the final, linear layer of the network. We
empirically demonstrate several characteristics of these weights that occur
frequently in trojaned networks, but not in benign networks. In particular, we
show that the distribution of the weights associated with the trojan target
class is clearly distinguishable from the weights associated with other
classes. Using this, we demonstrate the effectiveness of our proposed detection
method against state-of-the-art attacks across a variety of architectures,
datasets, and trigger types.

    

### [[2109.02839] Self-adaptive deep neural network: Numerical approximation to functions and PDEs](http://arxiv.org/abs/2109.02839)


  Designing an optimal deep neural network for a given task is important and
challenging in many machine learning applications. To address this issue, we
introduce a self-adaptive algorithm: the adaptive network enhancement (ANE)
method, written as loops of the form train, estimate and enhance. Starting with
a small two-layer neural network (NN), the step train is to solve the
optimization problem at the current NN; the step estimate is to compute a
posteriori estimator/indicators using the solution at the current NN; the step
enhance is to add new neurons to the current NN.
Novel network enhancement strategies based on the computed
estimator/indicators are developed in this paper to determine how many new
neurons and when a new layer should be added to the current NN. The ANE method
provides a natural process for obtaining a good initialization in training the
current NN; in addition, we introduce an advanced procedure on how to
initialize newly added neurons for a better approximation. We demonstrate that
the ANE method can automatically design a nearly minimal NN for learning
functions exhibiting sharp transitional layers as well as discontinuous
solutions of hyperbolic partial differential equations.

    

### [[2109.02859] Hyper Meta-Path Contrastive Learning for Multi-Behavior Recommendation](http://arxiv.org/abs/2109.02859)


  User purchasing prediction with multi-behavior information remains a
challenging problem for current recommendation systems. Various methods have
been proposed to address it via leveraging the advantages of graph neural
networks (GNNs) or multi-task learning. However, most existing works do not
take the complex dependencies among different behaviors of users into
consideration. They utilize simple and fixed schemes, like neighborhood
information aggregation or mathematical calculation of vectors, to fuse the
embeddings of different user behaviors to obtain a unified embedding to
represent a user's behavioral patterns which will be used in downstream
recommendation tasks. To tackle the challenge, in this paper, we first propose
the concept of hyper meta-path to construct hyper meta-paths or hyper
meta-graphs to explicitly illustrate the dependencies among different behaviors
of a user. How to obtain a unified embedding for a user from hyper meta-paths
and avoid the previously mentioned limitations simultaneously is critical.
Thanks to the recent success of graph contrastive learning, we leverage it to
learn embeddings of user behavior patterns adaptively instead of assigning a
fixed scheme to understand the dependencies among different behaviors. A new
graph contrastive learning based framework is proposed by coupling with hyper
meta-paths, namely HMG-CR, which consistently and significantly outperforms all
baselines in extensive comparison experiments.

    

### [[2109.02862] ICCAD Special Session Paper: Quantum-Classical Hybrid Machine Learning for Image Classification](http://arxiv.org/abs/2109.02862)


  Image classification is a major application domain for conventional deep
learning (DL). Quantum machine learning (QML) has the potential to
revolutionize image classification. In any typical DL-based image
classification, we use convolutional neural network (CNN) to extract features
from the image and multi-layer perceptron network (MLP) to create the actual
decision boundaries. On one hand, QML models can be useful in both of these
tasks. Convolution with parameterized quantum circuits (Quanvolution) can
extract rich features from the images. On the other hand, quantum neural
network (QNN) models can create complex decision boundaries. Therefore,
Quanvolution and QNN can be used to create an end-to-end QML model for image
classification. Alternatively, we can extract image features separately using
classical dimension reduction techniques such as, Principal Components Analysis
(PCA) or Convolutional Autoencoder (CAE) and use the extracted features to
train a QNN. We review two proposals on quantum-classical hybrid ML models for
image classification namely, Quanvolutional Neural Network and dimension
reduction using a classical algorithm followed by QNN. Particularly, we make a
case for trainable filters in Quanvolution and CAE-based feature extraction for
image datasets (instead of dimension reduction using linear transformations
such as, PCA). We discuss various design choices, potential opportunities, and
drawbacks of these models. We also release a Python-based framework to create
and explore these hybrid models with a variety of design choices.

    

### [[2109.02863] Refinement of Hottopixx and its Postprocessing](http://arxiv.org/abs/2109.02863)


  Hottopixx, proposed by Bittorf et al. at NIPS 2012, is an algorithm for
solving nonnegative matrix factorization (NMF) problems under the separability
assumption. Separable NMFs have important applications, such as topic
extraction from documents and unmixing of hyperspectral images. In such
applications, the robustness of the algorithm to noise is the key to the
success. Hottopixx has been shown to be robust to noise, and its robustness can
be further enhanced through postprocessing. However, there is a drawback.
Hottopixx and its postprocessing require us to estimate the noise level
involved in the matrix we want to factorize before running, since they use it
as part of the input data. The noise-level estimation is not an easy task. In
this paper, we overcome this drawback. We present a refinement of Hottopixx and
its postprocessing that runs without prior knowledge of the noise level. We
show that the refinement has almost the same robustness to noise as the
original algorithm.

    

### [[2109.02868] HMSG: Heterogeneous Graph Neural Network based on Metapath Subgraph Learning](http://arxiv.org/abs/2109.02868)


  Many real-world data can be represented as heterogeneous graphs with
different types of nodes and connections. Heterogeneous graph neural network
model aims to embed nodes or subgraphs into low-dimensional vector space for
various downstream tasks such as node classification, link prediction, etc.
Although several models were proposed recently, they either only aggregate
information from the same type of neighbors, or just indiscriminately treat
homogeneous and heterogeneous neighbors in the same way. Based on these
observations, we propose a new heterogeneous graph neural network model named
HMSG to comprehensively capture structural, semantic and attribute information
from both homogeneous and heterogeneous neighbors. Specifically, we first
decompose the heterogeneous graph into multiple metapath-based homogeneous and
heterogeneous subgraphs, and each subgraph associates specific semantic and
structural information. Then message aggregation methods are applied to each
subgraph independently, so that information can be learned in a more targeted
and efficient manner. Through a type-specific attribute transformation, node
attributes can also be transferred among different types of nodes. Finally, we
fuse information from subgraphs together to get the complete representation.
Extensive experiments on several datasets for node classification, node
clustering and link prediction tasks show that HMSG achieves the best
performance in all evaluation metrics than state-of-the-art baselines.

    

### [[2109.02889] Adversarial Parameter Defense by Multi-Step Risk Minimization](http://arxiv.org/abs/2109.02889)


  Previous studies demonstrate DNNs' vulnerability to adversarial examples and
adversarial training can establish a defense to adversarial examples. In
addition, recent studies show that deep neural networks also exhibit
vulnerability to parameter corruptions. The vulnerability of model parameters
is of crucial value to the study of model robustness and generalization. In
this work, we introduce the concept of parameter corruption and propose to
leverage the loss change indicators for measuring the flatness of the loss
basin and the parameter robustness of neural network parameters. On such basis,
we analyze parameter corruptions and propose the multi-step adversarial
corruption algorithm. To enhance neural networks, we propose the adversarial
parameter defense algorithm that minimizes the average risk of multiple
adversarial parameter corruptions. Experimental results show that the proposed
algorithm can improve both the parameter robustness and accuracy of neural
networks.

    

### [[2109.02890] Using Satellite Imagery and Machine Learning to Estimate the Livelihood Impact of Electricity Access](http://arxiv.org/abs/2109.02890)


  In many regions of the world, sparse data on key economic outcomes inhibits
the development, targeting, and evaluation of public policy. We demonstrate how
advancements in satellite imagery and machine learning can help ameliorate
these data and inference challenges. In the context of an expansion of the
electrical grid across Uganda, we show how a combination of satellite imagery
and computer vision can be used to develop local-level livelihood measurements
appropriate for inferring the causal impact of electricity access on
livelihoods. We then show how ML-based inference techniques deliver more
reliable estimates of the causal impact of electrification than traditional
alternatives when applied to these data. We estimate that grid access improves
village-level asset wealth in rural Uganda by 0.17 standard deviations, more
than doubling the growth rate over our study period relative to untreated
areas. Our results provide country-scale evidence on the impact of a key
infrastructure investment, and provide a low-cost, generalizable approach to
future policy evaluation in data sparse environments.

    

### [[2109.02894] Prescriptive Process Monitoring Under Resource Constraints: A Causal Inference Approach](http://arxiv.org/abs/2109.02894)


  Prescriptive process monitoring is a family of techniques to optimize the
performance of a business process by triggering interventions at runtime.
Existing prescriptive process monitoring techniques assume that the number of
interventions that may be triggered is unbounded. In practice, though, specific
interventions consume resources with finite capacity. For example, in a loan
origination process, an intervention may consist of preparing an alternative
loan offer to increase the applicant's chances of taking a loan. This
intervention requires a certain amount of time from a credit officer, and thus,
it is not possible to trigger this intervention in all cases. This paper
proposes a prescriptive process monitoring technique that triggers
interventions to optimize a cost function under fixed resource constraints. The
proposed technique relies on predictive modeling to identify cases that are
likely to lead to a negative outcome, in combination with causal inference to
estimate the effect of an intervention on the outcome of the case. These
outputs are then used to allocate resources to interventions to maximize a cost
function. A preliminary empirical evaluation suggests that the proposed
approach produces a higher net gain than a purely predictive (non-causal)
baseline.

    

### [[2109.02909] BioNetExplorer: Architecture-Space Exploration of Bio-Signal Processing Deep Neural Networks for Wearables](http://arxiv.org/abs/2109.02909)


  In this work, we propose the BioNetExplorer framework to systematically
generate and explore multiple DNN architectures for bio-signal processing in
wearables. Our framework adapts key neural architecture parameters to search
for an embedded DNN with a low hardware overhead, which can be deployed in
wearable edge devices to analyse the bio-signal data and to extract the
relevant information, such as arrhythmia and seizure. Our framework also
enables hardware-aware DNN architecture search using genetic algorithms by
imposing user requirements and hardware constraints (storage, FLOPs, etc.)
during the exploration stage, thereby limiting the number of networks explored.
Moreover, BioNetExplorer can also be used to search for DNNs based on the
user-required output classes; for instance, a user might require a specific
output class due to genetic predisposition or a pre-existing heart condition.
The use of genetic algorithms reduces the exploration time, on average, by 9x,
compared to exhaustive exploration. We are successful in identifying
Pareto-optimal designs, which can reduce the storage overhead of the DNN by
~30MB for a quality loss of less than 0.5%. To enable low-cost embedded DNNs,
BioNetExplorer also employs different model compression techniques to further
reduce the storage overhead of the network by up to 53x for a quality loss of
<0.2%.

    

### [[2109.02914] Scale-invariant representation of machine learning](http://arxiv.org/abs/2109.02914)


  The success of machine learning stems from its structured data
representation. Similar data have close representation as compressed codes for
classification or emerged labels for clustering. We observe that the frequency
of the internal representation follows power laws in both supervised and
unsupervised learning. The scale-invariant distribution implies that machine
learning largely compresses frequent typical data, and at the same time,
differentiates many atypical data as outliers. In this study, we derive how the
power laws can naturally arise in machine learning. In terms of information
theory, the scale-invariant representation corresponds to a maximally uncertain
data grouping among possible representations that guarantee pre-specified
learning accuracy.

    

### [[2109.02915] Few-shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network with Adaptive Sample Pair Formation](http://arxiv.org/abs/2109.02915)


  Speech-based machine learning (ML) has been heralded as a promising solution
for tracking prosodic and spectrotemporal patterns in real-life that are
indicative of emotional changes, providing a valuable window into one's
cognitive and mental state. Yet, the scarcity of labelled data in ambulatory
studies prevents the reliable training of ML models, which usually rely on
"data-hungry" distribution-based learning. Leveraging the abundance of labelled
speech data from acted emotions, this paper proposes a few-shot learning
approach for automatically recognizing emotion in spontaneous speech from a
small number of labelled samples. Few-shot learning is implemented via a metric
learning approach through a siamese neural network, which models the relative
distance between samples rather than relying on learning absolute patterns of
the corresponding distributions of each emotion. Results indicate the
feasibility of the proposed metric learning in recognizing emotions from
spontaneous speech in four datasets, even with a small amount of labelled
samples. They further demonstrate superior performance of the proposed metric
learning compared to commonly used adaptation methods, including network
fine-tuning and adversarial learning. Findings from this work provide a
foundation for the ambulatory tracking of human emotion in spontaneous speech
contributing to the real-life assessment of mental health degradation.

    

### [[2109.02929] Brand Label Albedo Extraction of eCommerce Products using Generative Adversarial Network](http://arxiv.org/abs/2109.02929)


  In this paper we present our solution to extract albedo of branded labels for
e-commerce products. To this end, we generate a large-scale photo-realistic
synthetic data set for albedo extraction followed by training a generative
model to translate images with diverse lighting conditions to albedo. We
performed an extensive evaluation to test the generalisation of our method to
in-the-wild images. From the experimental results, we observe that our solution
generalises well compared to the existing method both in the unseen rendered
images as well as in the wild image.

    

### [[2109.02934] Fishr: Invariant Gradient Variances for Out-of-distribution Generalization](http://arxiv.org/abs/2109.02934)


  Learning robust models that generalize well under changes in the data
distribution is critical for real-world applications. To this end, there has
been a growing surge of interest to learn simultaneously from multiple training
domains - while enforcing different types of invariance across those domains.
Yet, all existing approaches fail to show systematic benefits under fair
evaluation protocols. In this paper, we propose a new learning scheme to
enforce domain invariance in the space of the gradients of the loss function:
specifically, we introduce a regularization term that matches the domain-level
variances of gradients across training domains. Critically, our strategy, named
Fishr, exhibits close relations with the Fisher Information and the Hessian of
the loss. We show that forcing domain-level gradient covariances to be similar
during the learning procedure eventually aligns the domain-level loss
landscapes locally around the final weights. Extensive experiments demonstrate
the effectiveness of Fishr for out-of-distribution generalization. In
particular, Fishr improves the state of the art on the DomainBed benchmark and
performs significantly better than Empirical Risk Minimization. The code is
released at this https URL.

    

### [[2109.02941] Countering Online Hate Speech: An NLP Perspective](http://arxiv.org/abs/2109.02941)


  Online hate speech has caught everyone's attention from the news related to
the COVID-19 pandemic, US elections, and worldwide protests. Online toxicity -
an umbrella term for online hateful behavior, manifests itself in forms such as
online hate speech. Hate speech is a deliberate attack directed towards an
individual or a group motivated by the targeted entity's identity or opinions.
The rising mass communication through social media further exacerbates the
harmful consequences of online hate speech. While there has been significant
research on hate-speech identification using Natural Language Processing (NLP),
the work on utilizing NLP for prevention and intervention of online hate speech
lacks relatively. This paper presents a holistic conceptual framework on
hate-speech NLP countering methods along with a thorough survey on the current
progress of NLP for countering online hate speech. It classifies the countering
techniques based on their time of action, and identifies potential future
research areas on this topic.

    

### [[2109.02969] Efficient ADMM-based Algorithms for Convolutional Sparse Coding](http://arxiv.org/abs/2109.02969)


  Convolutional sparse coding improves on the standard sparse approximation by
incorporating a global shift-invariant model. The most efficient convolutional
sparse coding methods are based on the alternating direction method of
multipliers and the convolution theorem. The only major difference between
these methods is how they approach a convolutional least-squares fitting
subproblem. This letter presents a solution to this subproblem, which improves
the efficiency of the state-of-the-art algorithms. We also use the same
approach for developing an efficient convolutional dictionary learning method.
Furthermore, we propose a novel algorithm for convolutional sparse coding with
a constraint on the approximation error.

    

### [[2109.02975] BERT based classification system for detecting rumours on Twitter](http://arxiv.org/abs/2109.02975)


  The role of social media in opinion formation has far-reaching implications
in all spheres of society. Though social media provide platforms for expressing
news and views, it is hard to control the quality of posts due to the sheer
volumes of posts on platforms like Twitter and Facebook. Misinformation and
rumours have lasting effects on society, as they tend to influence people's
opinions and also may motivate people to act irrationally. It is therefore very
important to detect and remove rumours from these platforms. The only way to
prevent the spread of rumours is through automatic detection and classification
of social media posts. Our focus in this paper is the Twitter social medium, as
it is relatively easy to collect data from Twitter. The majority of previous
studies used supervised learning approaches to classify rumours on Twitter.
These approaches rely on feature extraction to obtain both content and context
features from the text of tweets to distinguish rumours and non-rumours.
Manually extracting features however is time-consuming considering the volume
of tweets. We propose a novel approach to deal with this problem by utilising
sentence embedding using BERT to identify rumours on Twitter, rather than the
usual feature extraction techniques. We use sentence embedding using BERT to
represent each tweet's sentences into a vector according to the contextual
meaning of the tweet. We classify those vectors into rumours or non-rumours by
using various supervised learning techniques. Our BERT based models improved
the accuracy by approximately 10% as compared to previous methods.

    

### [[2109.02986] Instance-dependent Label-noise Learning under a Structural Causal Model](http://arxiv.org/abs/2109.02986)


  Label noise will degenerate the performance of deep learning algorithms
because deep neural networks easily overfit label errors. Let X and Y denote
the instance and clean label, respectively. When Y is a cause of X, according
to which many datasets have been constructed, e.g., SVHN and CIFAR, the
distributions of P(X) and P(Y|X) are entangled. This means that the
unsupervised instances are helpful to learn the classifier and thus reduce the
side effect of label noise. However, it remains elusive on how to exploit the
causal information to handle the label noise problem. In this paper, by
leveraging a structural causal model, we propose a novel generative approach
for instance-dependent label-noise learning. In particular, we show that
properly modeling the instances will contribute to the identifiability of the
label noise transition matrix and thus lead to a better classifier.
Empirically, our method outperforms all state-of-the-art methods on both
synthetic and real-world label-noise datasets.

    

### [[2109.03008] Semiparametric Bayesian Networks](http://arxiv.org/abs/2109.03008)


  We introduce semiparametric Bayesian networks that combine parametric and
nonparametric conditional probability distributions. Their aim is to
incorporate the advantages of both components: the bounded complexity of
parametric models and the flexibility of nonparametric ones. We demonstrate
that semiparametric Bayesian networks generalize two well-known types of
Bayesian networks: Gaussian Bayesian networks and kernel density estimation
Bayesian networks. For this purpose, we consider two different conditional
probability distributions required in a semiparametric Bayesian network. In
addition, we present modifications of two well-known algorithms (greedy
hill-climbing and PC) to learn the structure of a semiparametric Bayesian
network from data. To realize this, we employ a score function based on
cross-validation. In addition, using a validation dataset, we apply an
early-stopping criterion to avoid overfitting. To evaluate the applicability of
the proposed algorithm, we conduct an exhaustive experiment on synthetic data
sampled by mixing linear and nonlinear functions, multivariate normal data
sampled from Gaussian Bayesian networks, real data from the UCI repository, and
bearings degradation data. As a result of this experiment, we conclude that the
proposed algorithm accurately learns the combination of parametric and
nonparametric components, while achieving a performance comparable with those
provided by state-of-the-art methods.

    

### [[2109.03020] Deep Convolutional Neural Networks Predict Elasticity Tensors and their Bounds in Homogenization](http://arxiv.org/abs/2109.03020)


  In the present work, 3D convolutional neural networks (CNNs) are trained to
link random heterogeneous, two-phase materials of arbitrary phase fractions to
their elastic macroscale stiffness thus replacing explicit homogenization
simulations. In order to reduce the uncertainty of the true stiffness of the
synthetic composites due to unknown boundary conditions (BCs), the CNNs predict
beyond the stiffness for periodic BC the upper bound through kinematically
uniform BC, and the lower bound through stress uniform BC. This work describes
the workflow of the homogenization-CNN, from microstructure generation over the
CNN design, the operations of convolution, nonlinear activation and pooling as
well as training and validation along with backpropagation up to performance
measurements in tests. Therein the CNNs demonstrate the predictive accuracy not
only for the standard test set but also for samples of the real, two-phase
microstructure of a diamond-based coating. The CNN that covers all three
boundary types is virtually as accurate as the separate treatment in three
different nets. The CNNs of this contribution provide through stiffness bounds
an indicator of the proper RVE size for individual snapshot samples. Moreover,
they enable statistical analyses for the effective elastic stiffness on
ensembles of synthetical microstructures without costly simulations.

    

### [[2109.03029] Predicting Mood Disorder Symptoms with Remotely Collected Videos Using an Interpretable Multimodal Dynamic Attention Fusion Network](http://arxiv.org/abs/2109.03029)


  We developed a novel, interpretable multimodal classification method to
identify symptoms of mood disorders viz. depression, anxiety and anhedonia
using audio, video and text collected from a smartphone application. We used
CNN-based unimodal encoders to learn dynamic embeddings for each modality and
then combined these through a transformer encoder. We applied these methods to
a novel dataset - collected by a smartphone application - on 3002 participants
across up to three recording sessions. Our method demonstrated better
multimodal classification performance compared to existing methods that
employed static embeddings. Lastly, we used SHapley Additive exPlanations
(SHAP) to prioritize important features in our model that could serve as
potential digital markers.

    

### [[2109.03040] Reconfigurable co-processor architecture with limited numerical precision to accelerate deep convolutional neural networks](http://arxiv.org/abs/2109.03040)


  Convolutional Neural Networks (CNNs) are widely used in deep learning
applications, e.g. visual systems, robotics etc. However, existing software
solutions are not efficient. Therefore, many hardware accelerators have been
proposed optimizing performance, power and resource utilization of the
implementation. Amongst existing solutions, Field Programmable Gate Array
(FPGA) based architecture provides better cost-energy-performance trade-offs as
well as scalability and minimizing development time. In this paper, we present
a model-independent reconfigurable co-processing architecture to accelerate
CNNs. Our architecture consists of parallel Multiply and Accumulate (MAC) units
with caching techniques and interconnection networks to exploit maximum data
parallelism. In contrast to existing solutions, we introduce limited precision
32 bit Q-format fixed point quantization for arithmetic representations and
operations. As a result, our architecture achieved significant reduction in
resource utilization with competitive accuracy. Furthermore, we developed an
assembly-type microinstructions to access the co-processing fabric to manage
layer-wise parallelism, thereby making re-use of limited resources. Finally, we
have tested our architecture up to 9x9 kernel size on Xilinx Virtex 7 FPGA,
achieving a throughput of up to 226.2 GOp/S for 3x3 kernel size.

    

### [[2109.03048] Early ICU Mortality Prediction and Survival Analysis for Respiratory Failure](http://arxiv.org/abs/2109.03048)


  Respiratory failure is the one of major causes of death in critical care
unit. During the outbreak of COVID-19, critical care units experienced an
extreme shortage of mechanical ventilation because of respiratory failure
related syndromes. To help this, the early mortality risk prediction in
patients who suffer respiratory failure can provide timely support for clinical
treatment and resource management. In the study, we propose a dynamic modeling
approach for early mortality risk prediction of the respiratory failure
patients based on the first 24 hours ICU physiological data. Our proposed model
is validated on the eICU collaborate database. We achieved a high AUROC
performance (80-83%) and significantly improved AUCPR 4% on Day 5 since ICU
admission, compared to the state-of-art prediction models. In addition, we
illustrated that the survival curve includes the time-varying information for
the early ICU admission survival analysis.

    

### [[2109.03069] Sequential Diagnosis Prediction with Transformer and Ontological Representation](http://arxiv.org/abs/2109.03069)


  Sequential diagnosis prediction on the Electronic Health Record (EHR) has
been proven crucial for predictive analytics in the medical domain. EHR data,
sequential records of a patient's interactions with healthcare systems, has
numerous inherent characteristics of temporality, irregularity and data
insufficiency. Some recent works train healthcare predictive models by making
use of sequential information in EHR data, but they are vulnerable to
irregular, temporal EHR data with the states of admission/discharge from
hospital, and insufficient data. To mitigate this, we propose an end-to-end
robust transformer-based model called SETOR, which exploits neural ordinary
differential equation to handle both irregular intervals between a patient's
visits with admitted timestamps and length of stay in each visit, to alleviate
the limitation of insufficient data by integrating medical ontology, and to
capture the dependencies between the patient's visits by employing multi-layer
transformer blocks. Experiments conducted on two real-world healthcare datasets
show that, our sequential diagnoses prediction model SETOR not only achieves
better predictive results than previous state-of-the-art approaches,
irrespective of sufficient or insufficient training data, but also derives more
interpretable embeddings of medical codes. The experimental codes are available
at the GitHub repository (this https URL).

    

### [[2109.03091] OdoNet: Untethered Speed Aiding for Vehicle Navigation Without Hardware Wheeled Odometer](http://arxiv.org/abs/2109.03091)


  Odometer has been proven to significantly improve the accuracy of the Global
Navigation Satellite System / Inertial Navigation System (GNSS/INS) integrated
vehicle navigation in GNSS-challenged environments. However, the odometer is
inaccessible in many applications, especially for aftermarket devices. To apply
forward speed aiding without hardware wheeled odometer, we propose OdoNet, an
untethered one-dimensional Convolution Neural Network (CNN)-based
pseudo-odometer model learning from a single Inertial Measurement Unit (IMU),
which can act as an alternative to the wheeled odometer. Dedicated experiments
have been conducted to verify the feasibility and robustness of the OdoNet. The
results indicate that the IMU individuality, the vehicle loads, and the road
conditions have little impact on the robustness and precision of the OdoNet,
while the IMU biases and the mounting angles may notably ruin the OdoNet. Thus,
a data-cleaning procedure is added to effectively mitigate the impacts of the
IMU biases and the mounting angles. Compared to the process using only
non-holonomic constraint (NHC), after employing the pseudo-odometer, the
positioning error is reduced by around 68%, while the percentage is around 74%
for the hardware wheeled odometer. In conclusion, the proposed OdoNet can be
employed as an untethered pseudo-odometer for vehicle navigation, which can
efficiently improve the accuracy and reliability of the positioning in
GNSS-denied environments.

    

### [[2109.03099] Optimizing model-agnostic Random Subspace ensembles](http://arxiv.org/abs/2109.03099)


  This paper presents a model-agnostic ensemble approach for supervised
learning. The proposed approach alternates between (1) learning an ensemble of
models using a parametric version of the Random Subspace approach, in which
feature subsets are sampled according to Bernoulli distributions, and (2)
identifying the parameters of the Bernoulli distributions that minimize the
generalization error of the ensemble model. Parameter optimization is rendered
tractable by using an importance sampling approach able to estimate the
expected model output for any given parameter set, without the need to learn
new models. While the degree of randomization is controlled by a
hyper-parameter in standard Random Subspace, it has the advantage to be
automatically tuned in our parametric version. Furthermore, model-agnostic
feature importance scores can be easily derived from the trained ensemble
model. We show the good performance of the proposed approach, both in terms of
prediction and feature ranking, on simulated and real-world datasets. We also
show that our approach can be successfully used for the reconstruction of gene
regulatory networks.

    

### [[2109.03115] Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity](http://arxiv.org/abs/2109.03115)


  The study of functional brain connectivity (FC) is important for
understanding the underlying mechanisms of many psychiatric disorders. Many
recent analyses adopt graph convolutional networks, to study non-linear
interactions between functionally-correlated states. However, although patterns
of brain activation are known to be hierarchically organised in both space and
time, many methods have failed to extract powerful spatio-temporal features. To
overcome those challenges, and improve understanding of long-range functional
dynamics, we translate an approach, from the domain of skeleton-based action
recognition, designed to model interactions across space and time. We evaluate
this approach using the Human Connectome Project (HCP) dataset on sex
classification and fluid intelligence prediction. To account for subject
topographic variability of functional organisation, we modelled functional
connectomes using multi-resolution dual-regressed (subject-specific) ICA nodes.
Results show a prediction accuracy of 94.4% for sex classification (an increase
of 6.2% compared to other methods), and an improvement of correlation with
fluid intelligence of 0.325 vs 0.144, relative to a baseline model that encodes
space and time separately. Results suggest that explicit encoding of
spatio-temporal dynamics of brain functional activity may improve the precision
with which behavioural and cognitive phenotypes may be predicted in the future.

    

### [[2109.03124] GANSER: A Self-supervised Data Augmentation Framework for EEG-based Emotion Recognition](http://arxiv.org/abs/2109.03124)


  The data scarcity problem in Electroencephalography (EEG) based affective
computing results into difficulty in building an effective model with high
accuracy and stability using machine learning algorithms especially deep
learning models. Data augmentation has recently achieved considerable
performance improvement for deep learning models: increased accuracy,
stability, and reduced over-fitting. In this paper, we propose a novel data
augmentation framework, namely Generative Adversarial Network-based
Self-supervised Data Augmentation (GANSER). As the first to combine adversarial
training with self-supervised learning for EEG-based emotion recognition, the
proposed framework can generate high-quality and high-diversity simulated EEG
samples. In particular, we utilize adversarial training to learn an EEG
generator and force the generated EEG signals to approximate the distribution
of real samples, ensuring the quality of augmented samples. A transformation
function is employed to mask parts of EEG signals and force the generator to
synthesize potential EEG signals based on the remaining parts, to produce a
wide variety of samples. The masking possibility during transformation is
introduced as prior knowledge to guide to extract distinguishable features for
simulated EEG signals and generalize the classifier to the augmented sample
space. Finally, extensive experiments demonstrate our proposed method can help
emotion recognition for performance gain and achieve state-of-the-art results.

    

### [[2109.03137] NumGPT: Improving Numeracy Ability of Generative Pre-trained Models](http://arxiv.org/abs/2109.03137)


  Existing generative pre-trained language models (e.g., GPT) focus on modeling
the language structure and semantics of general texts. However, those models do
not consider the numerical properties of numbers and cannot perform robustly on
numerical reasoning tasks (e.g., math word problems and measurement
estimation). In this paper, we propose NumGPT, a generative pre-trained model
that explicitly models the numerical properties of numbers in texts.
Specifically, it leverages a prototype-based numeral embedding to encode the
mantissa of the number and an individual embedding to encode the exponent of
the number. A numeral-aware loss function is designed to integrate numerals
into the pre-training objective of NumGPT. We conduct extensive experiments on
four different datasets to evaluate the numeracy ability of NumGPT. The
experiment results show that NumGPT outperforms baseline models (e.g., GPT and
GPT with DICE) on a range of numerical reasoning tasks such as measurement
estimation, number comparison, math word problems, and magnitude
classification. Ablation studies are also conducted to evaluate the impact of
pre-training and model hyperparameters on the performance.

    

### [[2109.03150] Recommendation Fairness: From Static to Dynamic](http://arxiv.org/abs/2109.03150)


  Driven by the need to capture users' evolving interests and optimize their
long-term experiences, more and more recommender systems have started to model
recommendation as a Markov decision process and employ reinforcement learning
to address the problem. Shouldn't research on the fairness of recommender
systems follow the same trend from static evaluation and one-shot intervention
to dynamic monitoring and non-stop control? In this paper, we portray the
recent developments in recommender systems first and then discuss how fairness
could be baked into the reinforcement learning techniques for recommendation.
Moreover, we argue that in order to make further progress in recommendation
fairness, we may want to consider multi-agent (game-theoretic) optimization,
multi-objective (Pareto) optimization, and simulation-based optimization, in
the general framework of stochastic games.

    

### [[2109.03154] PEEK: A Large Dataset of Learner Engagement with Educational Videos](http://arxiv.org/abs/2109.03154)


  Educational recommenders have received much less attention in comparison to
e-commerce and entertainment-related recommenders, even though efficient
intelligent tutors have great potential to improve learning gains. One of the
main challenges in advancing this research direction is the scarcity of large,
publicly available datasets. In this work, we release a large, novel dataset of
learners engaging with educational videos in-the-wild. The dataset, named
Personalised Educational Engagement with Knowledge Topics PEEK, is the first
publicly available dataset of this nature. The video lectures have been
associated with Wikipedia concepts related to the material of the lecture, thus
providing a humanly intuitive taxonomy. We believe that granular learner
engagement signals in unison with rich content representations will pave the
way to building powerful personalization algorithms that will revolutionise
educational and informational recommendation systems. Towards this goal, we 1)
construct a novel dataset from a popular video lecture repository, 2) identify
a set of benchmark algorithms to model engagement, and 3) run extensive
experimentation on the PEEK dataset to demonstrate its value. Our experiments
with the dataset show promise in building powerful informational recommender
systems. The dataset and the support code is available publicly.

    

### [[2109.03155] PAUSE: Positive and Annealed Unlabeled Sentence Embedding](http://arxiv.org/abs/2109.03155)


  Sentence embedding refers to a set of effective and versatile techniques for
converting raw text into numerical vector representations that can be used in a
wide range of natural language processing (NLP) applications. The majority of
these techniques are either supervised or unsupervised. Compared to the
unsupervised methods, the supervised ones make less assumptions about
optimization objectives and usually achieve better results. However, the
training requires a large amount of labeled sentence pairs, which is not
available in many industrial scenarios. To that end, we propose a generic and
end-to-end approach -- PAUSE (Positive and Annealed Unlabeled Sentence
Embedding), capable of learning high-quality sentence embeddings from a
partially labeled dataset. We experimentally show that PAUSE achieves, and
sometimes surpasses, state-of-the-art results using only a small fraction of
labeled sentence pairs on various benchmark tasks. When applied to a real
industrial use case where labeled samples are scarce, PAUSE encourages us to
extend our dataset without the liability of extensive manual annotation work.

    

### [[2109.03159] Regularized Learning in Banach Spaces](http://arxiv.org/abs/2109.03159)


  This article presents a different way to study the theory of regularized
learning for generalized data including representer theorems and convergence
theorems. The generalized data are composed of linear functionals and real
scalars to represent the discrete information of the local models. By the
extension of the classical machine learning, the empirical risks are computed
by the generalized data and the loss functions. According to the techniques of
regularization, the global solutions are approximated by minimizing the
regularized empirical risks over the Banach spaces. The Banach spaces are
adaptively chosen to endow the generalized input data with compactness such
that the existence and convergence of the approximate solutions are guaranteed
by the weak* topology.

    

### [[2109.03173] Learning to Bid in Contextual First Price Auctions](http://arxiv.org/abs/2109.03173)


  In this paper, we investigate the problem about how to bid in repeated
contextual first price auctions. We consider a single bidder (learner) who
repeatedly bids in the first price auctions: at each time $t$, the learner
observes a context $x_t\in \mathbb{R}^d$ and decides the bid based on
historical information and $x_t$. We assume a structured linear model of the
maximum bid of all the others $m_t = \alpha_0\cdot x_t + z_t$, where
$\alpha_0\in \mathbb{R}^d$ is unknown to the learner and $z_t$ is randomly
sampled from a noise distribution $\mathcal{F}$ with log-concave density
function $f$. We consider both \emph{binary feedback} (the learner can only
observe whether she wins or not) and \emph{full information feedback} (the
learner can observe $m_t$) at the end of each time $t$. For binary feedback,
when the noise distribution $\mathcal{F}$ is known, we propose a bidding
algorithm, by using maximum likelihood estimation (MLE) method to achieve at
most $\widetilde{O}(\sqrt{\log(d) T})$ regret. Moreover, we generalize this
algorithm to the setting with binary feedback and the noise distribution is
unknown but belongs to a parametrized family of distributions. For the full
information feedback with \emph{unknown} noise distribution, we provide an
algorithm that achieves regret at most $\widetilde{O}(\sqrt{dT})$. Our approach
combines an estimator for log-concave density functions and then MLE method to
learn the noise distribution $\mathcal{F}$ and linear weight $\alpha_0$
simultaneously. We also provide a lower bound result such that any bidding
policy in a broad class must achieve regret at least $\Omega(\sqrt{T})$, even
when the learner receives the full information feedback and $\mathcal{F}$ is
known.

    

### [[2109.03188] Optimizing Quantum Variational Circuits with Deep Reinforcement Learning](http://arxiv.org/abs/2109.03188)


  Quantum Machine Learning (QML) is considered to be one of the most promising
applications of near term quantum devices. However, the optimization of quantum
machine learning models presents numerous challenges arising from the
imperfections of hardware and the fundamental obstacles in navigating an
exponentially scaling Hilbert space. In this work, we evaluate the potential of
contemporary methods in deep reinforcement learning to augment gradient based
optimization routines in quantum variational circuits. We find that
reinforcement learning augmented optimizers consistently outperform gradient
descent in noisy environments. All code and pretrained weights are available to
replicate the results or deploy the models at
this https URL.

    

### [[2109.03194] On the Convergence of Decentralized Adaptive Gradient Methods](http://arxiv.org/abs/2109.03194)


  Adaptive gradient methods including Adam, AdaGrad, and their variants have
been very successful for training deep learning models, such as neural
networks. Meanwhile, given the need for distributed computing, distributed
optimization algorithms are rapidly becoming a focal point. With the growth of
computing power and the need for using machine learning models on mobile
devices, the communication cost of distributed training algorithms needs
careful consideration. In this paper, we introduce novel convergent
decentralized adaptive gradient methods and rigorously incorporate adaptive
gradient methods into decentralized training procedures. Specifically, we
propose a general algorithmic framework that can convert existing adaptive
gradient methods to their decentralized counterparts. In addition, we
thoroughly analyze the convergence behavior of the proposed algorithmic
framework and show that if a given adaptive gradient method converges, under
some specific conditions, then its decentralized counterpart is also
convergent. We illustrate the benefit of our generic decentralized framework on
a prototype method, i.e., AMSGrad, both theoretically and numerically.

    

### [[2109.03200] ExCode-Mixed: Explainable Approaches towards Sentiment Analysis on Code-Mixed Data using BERT models](http://arxiv.org/abs/2109.03200)


  The increasing use of social media sites in countries like India has given
rise to large volumes of code-mixed data. Sentiment analysis of this data can
provide integral insights into people's perspectives and opinions. Developing
robust explainability techniques which explain why models make their
predictions becomes essential. In this paper, we propose an adequate
methodology to integrate explainable approaches into code-mixed sentiment
analysis.

    

### [[2109.03207] COCO Denoiser: Using Co-Coercivity for Variance Reduction in Stochastic Convex Optimization](http://arxiv.org/abs/2109.03207)


  First-order methods for stochastic optimization have undeniable relevance, in
part due to their pivotal role in machine learning. Variance reduction for
these algorithms has become an important research topic. In contrast to common
approaches, which rarely leverage global models of the objective function, we
exploit convexity and L-smoothness to improve the noisy estimates outputted by
the stochastic gradient oracle. Our method, named COCO denoiser, is the joint
maximum likelihood estimator of multiple function gradients from their noisy
observations, subject to co-coercivity constraints between them. The resulting
estimate is the solution of a convex Quadratically Constrained Quadratic
Problem. Although this problem is expensive to solve by interior point methods,
we exploit its structure to apply an accelerated first-order algorithm, the
Fast Dual Proximal Gradient method. Besides analytically characterizing the
proposed estimator, we show empirically that increasing the number and
proximity of the queried points leads to better gradient estimates. We also
apply COCO in stochastic settings by plugging it in existing algorithms, such
as SGD, Adam or STRSAGA, outperforming their vanilla versions, even in
scenarios where our modelling assumptions are mismatched.

    

### [[2109.03214] Robust Predictable Control](http://arxiv.org/abs/2109.03214)


  Many of the challenges facing today's reinforcement learning (RL) algorithms,
such as robustness, generalization, transfer, and computational efficiency are
closely related to compression. Prior work has convincingly argued why
minimizing information is useful in the supervised learning setting, but
standard RL algorithms lack an explicit mechanism for compression. The RL
setting is unique because (1) its sequential nature allows an agent to use past
information to avoid looking at future observations and (2) the agent can
optimize its behavior to prefer states where decision making requires few bits.
We take advantage of these properties to propose a method (RPC) for learning
simple policies. This method brings together ideas from information
bottlenecks, model-based RL, and bits-back coding into a simple and
theoretically-justified algorithm. Our method jointly optimizes a latent-space
model and policy to be self-consistent, such that the policy avoids states
where the model is inaccurate. We demonstrate that our method achieves much
tighter compression than prior methods, achieving up to 5x higher reward than a
standard information bottleneck. We also demonstrate that our method learns
policies that are more robust and generalize better to new tasks.

    

### [[2109.03216] Learning Fast Sample Re-weighting Without Reward Data](http://arxiv.org/abs/2109.03216)


  Training sample re-weighting is an effective approach for tackling data
biases such as imbalanced and corrupted labels. Recent methods develop
learning-based algorithms to learn sample re-weighting strategies jointly with
model training based on the frameworks of reinforcement learning and meta
learning. However, depending on additional unbiased reward data is limiting
their general applicability. Furthermore, existing learning-based sample
re-weighting methods require nested optimizations of models and weighting
parameters, which requires expensive second-order computation. This paper
addresses these two problems and presents a novel learning-based fast sample
re-weighting (FSR) method that does not require additional reward data. The
method is based on two key ideas: learning from history to build proxy reward
data and feature sharing to reduce the optimization cost. Our experiments show
the proposed method achieves competitive results compared to state of the arts
on label noise robustness and long-tailed recognition, and does so while
achieving significantly improved training efficiency. The source code is
publicly available at
this https URL.

    

### [[2109.03219] Fruit-CoV: An Efficient Vision-based Framework for Speedy Detection and Diagnosis of SARS-CoV-2 Infections Through Recorded Cough Sounds](http://arxiv.org/abs/2109.03219)


  SARS-CoV-2 is colloquially known as COVID-19 that had an initial outbreak in
December 2019. The deadly virus has spread across the world, taking part in the
global pandemic disease since March 2020. In addition, a recent variant of
SARS-CoV-2 named Delta is intractably contagious and responsible for more than
four million deaths over the world. Therefore, it is vital to possess a
self-testing service of SARS-CoV-2 at home. In this study, we introduce
Fruit-CoV, a two-stage vision framework, which is capable of detecting
SARS-CoV-2 infections through recorded cough sounds. Specifically, we convert
sounds into Log-Mel Spectrograms and use the EfficientNet-V2 network to extract
its visual features in the first stage. In the second stage, we use 14
convolutional layers extracted from the large-scale Pretrained Audio Neural
Networks for audio pattern recognition (PANNs) and the Wavegram-Log-Mel-CNN to
aggregate feature representations of the Log-Mel Spectrograms. Finally, we use
the combined features to train a binary classifier. In this study, we use a
dataset provided by the AICovidVN 115M Challenge, which includes a total of
7371 recorded cough sounds collected throughout Vietnam, India, and
Switzerland. Experimental results show that our proposed model achieves an AUC
score of 92.8% and ranks the 1st place on the leaderboard of the AICovidVN
Challenge. More importantly, our proposed framework can be integrated into a
call center or a VoIP system to speed up detecting SARS-CoV-2 infections
through online/recorded cough sounds.

    

### [[2109.03220] Revisiting Recursive Least Squares for Training Deep Neural Networks](http://arxiv.org/abs/2109.03220)


  Recursive least squares (RLS) algorithms were once widely used for training
small-scale neural networks, due to their fast convergence. However, previous
RLS algorithms are unsuitable for training deep neural networks (DNNs), since
they have high computational complexity and too many preconditions. In this
paper, to overcome these drawbacks, we propose three novel RLS optimization
algorithms for training feedforward neural networks, convolutional neural
networks and recurrent neural networks (including long short-term memory
networks), by using the error backpropagation and our average-approximation RLS
method, together with the equivalent gradients of the linear least squares loss
function with respect to the linear outputs of hidden layers. Compared with
previous RLS optimization algorithms, our algorithms are simple and elegant.
They can be viewed as an improved stochastic gradient descent (SGD) algorithm,
which uses the inverse autocorrelation matrix of each layer as the adaptive
learning rate. Their time and space complexities are only several times those
of SGD. They only require the loss function to be the mean squared error and
the activation function of the output layer to be invertible. In fact, our
algorithms can be also used in combination with other first-order optimization
algorithms without requiring these two preconditions. In addition, we present
two improved methods for our algorithms. Finally, we demonstrate their
effectiveness compared to the Adam algorithm on MNIST, CIFAR-10 and IMDB
datasets, and investigate the influences of their hyperparameters
experimentally.

    

### [[2109.03228] Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression](http://arxiv.org/abs/2109.03228)


  Recent studies on compression of pretrained language models (e.g., BERT)
usually use preserved accuracy as the metric for evaluation. In this paper, we
propose two new metrics, label loyalty and probability loyalty that measure how
closely a compressed model (i.e., student) mimics the original model (i.e.,
teacher). We also explore the effect of compression with regard to robustness
under adversarial attacks. We benchmark quantization, pruning, knowledge
distillation and progressive module replacing with loyalty and robustness. By
combining multiple compression techniques, we provide a practical strategy to
achieve better accuracy, loyalty and robustness.

    

### [[1809.03048] Distance preserving model order reduction of graph-Laplacians and cluster analysis](http://arxiv.org/abs/1809.03048)


  Graph-Laplacians and their spectral embeddings play an important role in
multiple areas of machine learning. This paper is focused on graph-Laplacian
dimension reduction for the spectral clustering of data as a primary
application. Spectral embedding provides a low-dimensional parametrization of
the data manifold which makes the subsequent task (e.g., clustering) much
easier. However, despite reducing the dimensionality of data, the overall
computational cost may still be prohibitive for large data sets due to two
factors. First, computing the partial eigendecomposition of the graph-Laplacian
typically requires a large Krylov subspace. Second, after the spectral
embedding is complete, one still has to operate with the same number of data
points. For example, clustering of the embedded data is typically performed
with various relaxations of k-means which computational cost scales poorly with
respect to the size of data set. In this work, we switch the focus from the
entire data set to a subset of graph vertices (target subset). We develop two
novel algorithms for such low-dimensional representation of the original graph
that preserves important global distances between the nodes of the target
subset. In particular, it allows to ensure that target subset clustering is
consistent with the spectral clustering of the full data set if one would
perform such. That is achieved by a properly parametrized reduced-order model
(ROM) of the graph-Laplacian that approximates accurately the diffusion
transfer function of the original graph for inputs and outputs restricted to
the target subset. Working with a small target subset reduces greatly the
required dimension of Krylov subspace and allows to exploit the conventional
algorithms (like approximations of k-means) in the regimes when they are most
robust and efficient.

    

### [[1911.12990] Semi-Relaxed Quantization with DropBits: Training Low-Bit Neural Networks via Bit-wise Regularization](http://arxiv.org/abs/1911.12990)


  Network quantization, which aims to reduce the bit-lengths of the network
weights and activations, has emerged as one of the key ingredients to reduce
the size of neural networks for their deployments to resource-limited devices.
In order to overcome the nature of transforming continuous activations and
weights to discrete ones, recent study called Relaxed Quantization (RQ)
[Louizos et al. 2019] successfully employ the popular Gumbel-Softmax that
allows this transformation with efficient gradient-based optimization. However,
RQ with this Gumbel-Softmax relaxation still suffers from bias-variance
trade-off depending on the temperature parameter of Gumbel-Softmax. To resolve
the issue, we propose a novel method, Semi-Relaxed Quantization (SRQ) that uses
multi-class straight-through estimator to effectively reduce the bias and
variance, along with a new regularization technique, DropBits that replaces
dropout regularization to randomly drop the bits instead of neurons to further
reduce the bias of the multi-class straight-through estimator in SRQ. As a
natural extension of DropBits, we further introduce the way of learning
heterogeneous quantization levels to find proper bit-length for each layer
using DropBits. We experimentally validate our method on various benchmark
datasets and network architectures, and also support the quantized lottery
ticket hypothesis: learning heterogeneous quantization levels outperforms the
case using the same but fixed quantization levels from scratch.

    

### [[2004.08597] Robust Density Estimation under Besov IPM Losses](http://arxiv.org/abs/2004.08597)


  We study minimax convergence rates of nonparametric density estimation in the
Huber contamination model, in which a proportion of the data comes from an
unknown outlier distribution. We provide the first results for this problem
under a large family of losses, called Besov integral probability metrics
(IPMs), that includes $\mathcal{L}^p$, Wasserstein, Kolmogorov-Smirnov, and
other common distances between probability distributions. Specifically, under a
range of smoothness assumptions on the population and outlier distributions, we
show that a re-scaled thresholding wavelet series estimator achieves minimax
optimal convergence rates under a wide variety of losses. Finally, based on
connections that have recently been shown between nonparametric density
estimation under IPM losses and generative adversarial networks (GANs), we show
that certain GAN architectures also achieve these minimax rates.

    

### [[2004.12835] Intuitive Contrasting Map for Antonym Embeddings](http://arxiv.org/abs/2004.12835)


  This paper shows that, modern word embeddings contain information that
distinguishes synonyms and antonyms despite small cosine similarities between
corresponding vectors. This information is encoded in the geometry of the
embeddings and could be extracted with a straight-forward and intuitive
manifold learning procedure or a contrasting map. Such a map is trained on a
small labeled subset of the data and can produce new embeddings that explicitly
highlight specific semantic attributes of the word. The new embeddings produced
by the map are shown to improve the performance on downstream tasks.

    

### [[2005.01026] Multi-Center Federated Learning](http://arxiv.org/abs/2005.01026)


  Federated learning has received great attention for its capability to train a
large-scale model in a decentralized manner without needing to access user data
directly. It helps protect the users' private data from centralized collecting.
Unlike distributed machine learning, federated learning aims to tackle non-IID
data from heterogeneous sources in various real-world applications, such as
those on smartphones. Existing federated learning approaches usually adopt a
single global model to capture the shared knowledge of all users by aggregating
their gradients, regardless of the discrepancy between their data
distributions. However, due to the diverse nature of user behaviors, assigning
users' gradients to different global models (i.e., centers) can better capture
the heterogeneity of data distributions across users. Our paper proposes a
novel multi-center aggregation mechanism for federated learning, which learns
multiple global models from the non-IID user data and simultaneously derives
the optimal matching between users and centers. We formulate the problem as a
joint optimization that can be efficiently solved by a stochastic expectation
maximization (EM) algorithm. Our experimental results on benchmark datasets
show that our method outperforms several popular federated learning methods.

    

### [[2005.02921] Restricted maximum-likelihood method for learning latent variance components in gene expression data with known and unknown confounders](http://arxiv.org/abs/2005.02921)


  Random effect models are popular statistical models for detecting and
correcting spurious sample correlations due to hidden confounders in
genome-wide gene expression data. In applications where some confounding
factors are known, estimating simultaneously the contribution of known and
latent variance components in random effect models is a challenge that has so
far relied on numerical gradient-based optimizers to maximize the likelihood
function. This is unsatisfactory because the resulting solution is poorly
characterized and the efficiency of the method may be suboptimal. Here we prove
analytically that maximum-likelihood latent variables can always be chosen
orthogonal to the known confounding factors, in other words, that
maximum-likelihood latent variables explain sample covariances not already
explained by known factors. Based on this result we propose a restricted
maximum-likelihood method which estimates the latent variables by maximizing
the likelihood on the restricted subspace orthogonal to the known confounding
factors, and show that this reduces to probabilistic PCA on that subspace. The
method then estimates the variance-covariance parameters by maximizing the
remaining terms in the likelihood function given the latent variables, using a
newly derived analytic solution for this problem. Compared to gradient-based
optimizers, our method attains greater or equal likelihood values, can be
computed using standard matrix operations, results in latent factors that don't
overlap with any known factors, and has a runtime reduced by several orders of
magnitude. Hence the restricted maximum-likelihood method facilitates the
application of random effect modelling strategies for learning latent variance
components to much larger gene expression datasets than possible with current
methods.

    

### [[2006.14580] Backdoor Attacks Against Deep Learning Systems in the Physical World](http://arxiv.org/abs/2006.14580)


  Backdoor attacks embed hidden malicious behaviors into deep learning models,
which only activate and cause misclassifications on model inputs containing a
specific trigger. Existing works on backdoor attacks and defenses, however,
mostly focus on digital attacks that use digitally generated patterns as
triggers. A critical question remains unanswered: can backdoor attacks succeed
using physical objects as triggers, thus making them a credible threat against
deep learning systems in the real world? We conduct a detailed empirical study
to explore this question for facial recognition, a critical deep learning task.
Using seven physical objects as triggers, we collect a custom dataset of 3205
images of ten volunteers and use it to study the feasibility of physical
backdoor attacks under a variety of real-world conditions. Our study reveals
two key findings. First, physical backdoor attacks can be highly successful if
they are carefully configured to overcome the constraints imposed by physical
objects. In particular, the placement of successful triggers is largely
constrained by the target model's dependence on key facial features. Second,
four of today's state-of-the-art defenses against (digital) backdoors are
ineffective against physical backdoors, because the use of physical objects
breaks core assumptions used to construct these defenses. Our study confirms
that (physical) backdoor attacks are not a hypothetical phenomenon but rather
pose a serious real-world threat to critical classification tasks. We need new
and more robust defenses against backdoors in the physical world.

    

### [[2007.00596] A New Basis for Sparse Principal Component Analysis](http://arxiv.org/abs/2007.00596)


  Previous versions of sparse principal component analysis (PCA) have presumed
that the eigen-basis (a $p \times k$ matrix) is approximately sparse. We
propose a method that presumes the $p \times k$ matrix becomes approximately
sparse after a $k \times k$ rotation. The simplest version of the algorithm
initializes with the leading $k$ principal components. Then, the principal
components are rotated with an $k \times k$ orthogonal rotation to make them
approximately sparse. Finally, soft-thresholding is applied to the rotated
principal components. This approach differs from prior approaches because it
uses an orthogonal rotation to approximate a sparse basis. One consequence is
that a sparse component need not to be a leading eigenvector, but rather a
mixture of them. In this way, we propose a new (rotated) basis for sparse PCA.
In addition, our approach avoids "deflation" and multiple tuning parameters
required for that. Our sparse PCA framework is versatile; for example, it
extends naturally to a two-way analysis of a data matrix for simultaneous
dimensionality reduction of rows and columns. We provide evidence showing that
for the same level of sparsity, the proposed sparse PCA method is more stable
and can explain more variance compared to alternative methods. Through three
applications -- sparse coding of images, analysis of transcriptome sequencing
data, and large-scale clustering of social networks, we demonstrate the modern
usefulness of sparse PCA in exploring multivariate data.

    

### [[2009.03714] Dual-constrained Deep Semi-Supervised Coupled Factorization Network with Enriched Prior](http://arxiv.org/abs/2009.03714)


  Nonnegative matrix factorization is usually powerful for learning the
"shallow" parts-based representation, but it clearly fails to discover deep
hierarchical information within both the basis and representation spaces. In
this paper, we technically propose a new enriched prior based Dual-constrained
Deep Semi-Supervised Coupled Factorization Network, called DS2CF-Net, for
learning the hierarchical coupled representations. To ex-tract hidden deep
features, DS2CF-Net is modeled as a deep-structure and geometrical
structure-constrained neural network. Specifically, DS2CF-Net designs a deep
coupled factorization architecture using multi-layers of linear
transformations, which coupled updates the bases and new representations in
each layer. To improve the discriminating ability of learned deep
representations and deep coefficients, our network clearly considers enriching
the supervised prior by the joint deep coefficients-regularized label
prediction, and incorporates enriched prior information as additional label and
structure constraints. The label constraint can enable the samples of the same
label to have the same coordinate in the new feature space, while the structure
constraint forces the coefficient matrices in each layer to be block-diagonal
so that the enhanced prior using the self-expressive label propagation are more
accurate. Our network also integrates the adaptive dual-graph learning to
retain the local manifold structures of both the data manifold and feature
manifold by minimizing the reconstruction errors in each layer. Extensive
experiments on several real databases demonstrate that our DS2CF-Net can obtain
state-of-the-art performance for representation learning and clustering.

    

### [[2009.03831] Refined approachability algorithms and application to regret minimization with global costs](http://arxiv.org/abs/2009.03831)


  Blackwell's approachability is a framework where two players, the Decision
Maker and the Environment, play a repeated game with vector-valued payoffs. The
goal of the Decision Maker is to make the average payoff converge to a given
set called the target. When this is indeed possible, simple algorithms which
guarantee the convergence are known. This abstract tool was successfully used
for the construction of optimal strategies in various repeated games, but also
found several applications in online learning. By extending an approach
proposed by (Abernethy et al., 2011), we construct and analyze a class of
Follow the Regularized Leader algorithms (FTRL) for Blackwell's approachability
which are able to minimize not only the Euclidean distance to the target set
(as it is often the case in the context of Blackwell's approachability) but a
wide range of distance-like quantities. This flexibility enables us to apply
these algorithms to closely minimize the quantity of interest in various online
learning problems. In particular, for regret minimization with $\ell_p$ global
costs, we obtain the first bounds with explicit dependence in $p$ and the
dimension $d$.

    

### [[2009.13401] Injecting Entity Types into Entity-Guided Text Generation](http://arxiv.org/abs/2009.13401)


  Recent successes in deep generative modeling have led to significant advances
in natural language generation (NLG). Incorporating entities into neural
generation models has demonstrated great improvements by assisting to infer the
summary topic and to generate coherent content. To enhance the role of entity
in NLG, in this paper, we aim to model the entity type in the decoding phase to
generate contextual words accurately. We develop a novel NLG model to produce a
target sequence based on a given list of entities. Our model has a multi-step
decoder that injects the entity types into the process of entity mention
generation. Experiments on two public news datasets demonstrate type injection
performs better than existing type embedding concatenation baselines.

    

### [[2010.06201] Experimental Quantum Generative Adversarial Networks for Image Generation](http://arxiv.org/abs/2010.06201)


  Quantum machine learning is expected to be one of the first practical
applications of near-term quantum devices. Pioneer theoretical works suggest
that quantum generative adversarial networks (GANs) may exhibit a potential
exponential advantage over classical GANs, thus attracting widespread
attention. However, it remains elusive whether quantum GANs implemented on
near-term quantum devices can actually solve real-world learning tasks. Here,
we devise a flexible quantum GAN scheme to narrow this knowledge gap, which
could accomplish image generation with arbitrarily high-dimensional features,
and could also take advantage of quantum superposition to train multiple
examples in parallel. For the first time, we experimentally achieve the
learning and generation of real-world hand-written digit images on a
superconducting quantum processor. Moreover, we utilize a gray-scale bar
dataset to exhibit the competitive performance between quantum GANs and the
classical GANs based on multilayer perceptron and convolutional neural network
architectures, respectively, benchmarked by the Fréchet Distance score. Our
work provides guidance for developing advanced quantum generative models on
near-term quantum devices and opens up an avenue for exploring quantum
advantages in various GAN-related learning tasks.

    

### [[2010.07858] What you need to know to train recurrent neural networks to make Flip Flops memories and more](http://arxiv.org/abs/2010.07858)


  Training neural networks to perform different tasks is relevant across
various disciplines that go beyond Machine Learning. In particular, Recurrent
Neural Networks (RNN) are of great interest to different scientific
communities, for example, Computational Neuroscience research and Dynamical
Systems among others. Open-source frameworks dedicated to Machine Learning such
as Tensorflow and Keras has produced significant changes in the development of
technologies that we currently use. One relevant problem that can be approached
is how to build the models for the study of dynamical systems, and how to
extract the relevant information to be able to answer the scientific questions
of interest. The purpose of the present work is to contribute to this aim by
using a temporal processing task, in this case, a 3-bit Flip Flop memory, to
show the modeling procedure in every step: from equations to the software code
using Tensorflow and Keras. The obtained networks are analyzed to describe the
dynamics and to show different visualization and analysis tools. The code
developed in this work is provided to be used as a base for model other
systems.

    

### [[2011.13045] Learning to Infer Shape Programs Using Self Training](http://arxiv.org/abs/2011.13045)


  Inferring programs which generate 2D and 3D shapes is important for reverse
engineering, editing, and more. Training such inference models is challenging
due to the lack of paired (shape, program) data in most domains. A popular
approach is to pre-train a model on synthetic data and then fine-tune on real
shapes using slow, unstable reinforcement learning. In this paper, we argue
that self-training is a viable alternative for fine-tuning such models.
Self-training is a semi-supervised learning paradigm where a model assigns
pseudo-labels to unlabeled data, and then retrains with (data, pseudo-label)
pairs as the new ground truth. We show that for constructive solid geometry and
assembly-based modeling, self-training outperforms state-of-the-art
reinforcement learning approaches. Additionally, shape program inference has a
unique property that circumvents a potential downside of self-training
(incorrect pseudo-label assignment): inferred programs are executable. For a
given shape from our distribution of interest $\mathbf{x}^*$ and its predicted
program $\mathbf{z}$, one can execute $\mathbf{z}$ to obtain a shape
$\mathbf{x}$ and train on $(\mathbf{z}, \mathbf{x})$ pairs, rather than
$(\mathbf{z}, \mathbf{x}^*)$ pairs. We term this procedure latent execution
self training (LEST). We demonstrate that self training infers shape programs
with higher shape reconstruction accuracy and converges significantly faster
than reinforcement learning approaches, and in some domains, LEST can further
improve this performance.

    

### [[2012.05825] Novelty detection using ensembles with regularized disagreement](http://arxiv.org/abs/2012.05825)


  Despite their excellent performance on in-distribution (ID) data, machine
learning-based prediction systems often predict out-of-distribution (OOD)
samples incorrectly while indicating high confidence. Instead, they should flag
samples that are not similar to the training data, for example, when new
classes emerge over time. Even though current OOD detection algorithms can
successfully distinguish completely different data sets, they fail to reliably
identify samples from novel classes. We develop a new ensemble-based procedure
that promotes model diversity and exploits regularization to limit disagreement
to only OOD samples, using a batch containing an unknown mixture of ID and OOD
data. We show that our procedure significantly outperforms state-of-the-art
methods, including those that have access, during training, to data that is
known to be OOD. We run extensive comparisons of our approach on a variety of
novel-class detection scenarios, on standard image data sets such as
SVHN/CIFAR-10/CIFAR-100, as well as on new disease detection on medical image
data sets.

    

### [[2101.00234] Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers](http://arxiv.org/abs/2101.00234)


  Transformers have shown improved performance when compared to previous
architectures for sequence processing such as RNNs. Despite their sizeable
performance gains, as recently suggested, the model is computationally
expensive to train and with a high parameter budget. In light of this, we
explore parameter-sharing methods in Transformers with a specific focus on
generative models. We perform an analysis of different parameter
sharing/reduction methods and develop the Subformer. Our model combines
sandwich-style parameter sharing, which overcomes naive cross-layer parameter
sharing in generative models, and self-attentive embedding factorization
(SAFE). Experiments on machine translation, abstractive summarization and
language modeling show that the Subformer can outperform the Transformer even
when using significantly fewer parameters.

    

### [[2101.06986] Interactive slice visualization for exploring machine learning models](http://arxiv.org/abs/2101.06986)


  Machine learning models fit complex algorithms to arbitrarily large datasets.
These algorithms are well-known to be high on performance and low on
interpretability. We use interactive visualization of slices of predictor space
to address the interpretability deficit; in effect opening up the black-box of
machine learning algorithms, for the purpose of interrogating, explaining,
validating and comparing model fits. Slices are specified directly through
interaction, or using various touring algorithms designed to visit
high-occupancy sections or regions where the model fits have interesting
properties. The methods presented here are implemented in the R package
\pkg{condvis2}.

    

### [[2102.00632] ConvNets for Counting: Object Detection of Transient Phenomena in Steelpan Drums](http://arxiv.org/abs/2102.00632)


  We train an object detector built from convolutional neural networks to count
interference fringes in elliptical antinode regions in frames of high-speed
video recordings of transient oscillations in Caribbean steelpan drums
illuminated by electronic speckle pattern interferometry (ESPI). The
annotations provided by our model aim to contribute to the understanding of
time-dependent behavior in such drums by tracking the development of
sympathetic vibration modes. The system is trained on a dataset of crowdsourced
human-annotated images obtained from the Zooniverse Steelpan Vibrations
Project. Due to the small number of human-annotated images and the ambiguity of
the annotation task, we also evaluate the model on a large corpus of synthetic
images whose properties have been matched to the real images by style transfer
using a Generative Adversarial Network. Applying the model to thousands of
unlabeled video frames, we measure oscillations consistent with audio
recordings of these drum strikes. One unanticipated result is that sympathetic
oscillations of higher-octave notes significantly precede the rise in sound
intensity of the corresponding second harmonic tones; the mechanism responsible
for this remains unidentified. This paper primarily concerns the development of
the predictive model; further exploration of the steelpan images and deeper
physical insights await its further application.

    

### [[2102.07818] Certified Robustness to Programmable Transformations in LSTMs](http://arxiv.org/abs/2102.07818)


  Deep neural networks for natural language processing are fragile in the face
of adversarial examples -- small input perturbations, like synonym substitution
or word duplication, which cause a neural network to change its prediction. We
present an approach to certifying the robustness of LSTMs (and extensions of
LSTMs) and training models that can be efficiently certified. Our approach can
certify robustness to intractably large perturbation spaces defined
programmatically in a language of string transformations. Our evaluation shows
that (1) our approach can train models that are more robust to combinations of
string transformations than those produced using existing techniques; (2) our
approach can show high certification accuracy of the resulting models.

    

### [[2102.09262] PLAM: a Posit Logarithm-Approximate Multiplier](http://arxiv.org/abs/2102.09262)


  The Posit Number System was introduced in 2017 as a replacement for
floating-point numbers. Since then, the community has explored its application
in Neural Network related tasks and produced some unit designs which are still
far from being competitive with their floating-point counterparts. This paper
proposes a Posit Logarithm-Approximate Multiplication (PLAM) scheme to
significantly reduce the complexity of posit multipliers, the most power-hungry
units within Deep Neural Network architectures. When comparing with
state-of-the-art posit multipliers, experiments show that the proposed
technique reduces the area, power, and delay of hardware multipliers up to
72.86%, 81.79%, and 17.01%, respectively, without accuracy degradation.

    

### [[2103.03622] Explanations for Occluded Images](http://arxiv.org/abs/2103.03622)


  Existing algorithms for explaining the output of image classifiers perform
poorly on inputs where the object of interest is partially occluded. We present
a novel, black-box algorithm for computing explanations that uses a principled
approach based on causal theory. We have implemented the method in the
DEEPCOVER tool. We obtain explanations that are much more accurate than those
generated by the existing explanation tools on images with occlusions and
observe a level of performance comparable to the state of the art when
explaining images without occlusions.

    

### [[2103.08889] Quick Learning Mechanism with Cross-Domain Adaptation for Intelligent Fault Diagnosis](http://arxiv.org/abs/2103.08889)


  The fault diagnostic model trained for a laboratory case machine fails to
perform well on the industrial machines running under variable operating
conditions. For every new operating condition of such machines, a new
diagnostic model has to be trained which is a time-consuming and uneconomical
process. Therefore, we propose a quick learning mechanism that can transform
the existing diagnostic model into a new model suitable for industrial machines
operating in different conditions. The proposed method uses the Net2Net
transformation followed by a fine-tuning to cancel/minimize the maximum mean
discrepancy between the new data and the previous one. The fine-tuning of the
model requires a very less amount of labelled target samples and very few
iterations of training. Therefore, the proposed method is capable of learning
the new target data pattern quickly. The effectiveness of the proposed fault
diagnosis method has been demonstrated on the Case Western Reserve University
dataset, Intelligent Maintenance Systems bearing dataset, and Paderborn
university dataset under the wide variations of the operating conditions. It
has been validated that the diagnostic model trained on artificially damaged
fault datasets can be used to quickly train another model for a real damage
dataset.

    

### [[2103.12551] Deep Learning for Exotic Option Valuation](http://arxiv.org/abs/2103.12551)


  A common approach to valuing exotic options involves choosing a model and
then determining its parameters to fit the volatility surface as closely as
possible. We refer to this as the model calibration approach (MCA). A
disadvantage of MCA is that some information in the volatility surface is lost
during the calibration process and the prices of exotic options will not in
general be consistent with those of plain vanilla options. We consider an
alternative approach where the structure of the user's preferred model is
preserved but points on the volatility are features input to a neural network.
We refer to this as the volatility feature approach (VFA) model. We conduct
experiments showing that VFA can be expected to outperform MCA for the
volatility surfaces encountered in practice. Once the upfront computational
time has been invested in developing the neural network, the valuation of
exotic options using VFA is very fast.

    

### [[2103.12954] Convergence Analysis of Nonconvex Distributed Stochastic Zeroth-order Coordinate Method](http://arxiv.org/abs/2103.12954)


  This paper investigates the stochastic distributed nonconvex optimization
problem of minimizing a global cost function formed by the summation of $n$
local cost functions. We solve such a problem by involving zeroth-order (ZO)
information exchange. In this paper, we propose a ZO distributed primal-dual
coordinate method (ZODIAC) to solve the stochastic optimization problem. Agents
approximate their own local stochastic ZO oracle along with coordinates with an
adaptive smoothing parameter. We show that the proposed algorithm achieves the
convergence rate of $\mathcal{O}(\sqrt{p}/\sqrt{T})$ for general nonconvex cost
functions. We demonstrate the efficiency of proposed algorithms through a
numerical example in comparison with the existing state-of-the-art centralized
and distributed ZO algorithms.

    

### [[2105.03155] Diff-ResNets for Few-shot Learning -- an ODE Perspective](http://arxiv.org/abs/2105.03155)


  Interpreting deep neural networks from the ordinary differential equations
(ODEs) perspective has inspired many efficient and robust network
architectures. However, existing ODE based approaches ignore the relationship
among data points, which is a critical component in many problems including
few-shot learning and semi-supervised learning. In this paper, inspired by the
diffusive ODEs, we propose a novel diffusion residual network (Diff-ResNet) to
strengthen the interactions among data points. Under the structured data
assumption, it is proved that the diffusion mechanism can decrease the
distance-diameter ratio that improves the separability of inter-class points
and reduces the distance among local intra-class points. This property can be
easily adopted by the residual networks for constructing the separable
hyperplanes. The synthetic binary classification experiments demonstrate the
effectiveness of the proposed diffusion mechanism. Moreover, extensive
experiments of few-shot image classification and semi-supervised graph node
classification in various datasets validate the advantages of the proposed
Diff-ResNet over existing few-shot learning methods.

    

### [[2105.03534] SimJEB: Simulated Jet Engine Bracket Dataset](http://arxiv.org/abs/2105.03534)


  This paper introduces the Simulated Jet Engine Bracket Dataset (SimJEB): a
new, public collection of crowdsourced mechanical brackets and accompanying
structural simulations. SimJEB is applicable to a wide range of geometry
processing tasks; the complexity of the shapes in SimJEB offer a challenge to
automated geometry cleaning and meshing, while categorical labels and
structural simulations facilitate classification and regression (i.e.
engineering surrogate modeling). In contrast to existing shape collections,
SimJEB's models are all designed for the same engineering function and thus
have consistent structural loads and support conditions. On the other hand,
SimJEB models are more complex, diverse, and realistic than the synthetically
generated datasets commonly used in parametric surrogate model evaluation. The
designs in SimJEB were derived from submissions to the GrabCAD Jet Engine
Bracket Challenge: an open engineering design competition with over 700
hand-designed CAD entries from 320 designers representing 56 countries. Each
model has been cleaned, categorized, meshed, and simulated with finite element
analysis according to the original competition specifications. The result is a
collection of 381 diverse, high-quality and application-focused designs for
advancing geometric deep learning, engineering surrogate modeling, automated
cleaning and related geometry processing tasks.

    

### [[2105.11627] Least-Squares ReLU Neural Network (LSNN) Method For Scalar Nonlinear Hyperbolic Conservation Law](http://arxiv.org/abs/2105.11627)


  We introduced the least-squares ReLU neural network (LSNN) method for solving
the linear advection-reaction problem with discontinuous solution and showed
that the method outperforms mesh-based numerical methods in terms of the number
of degrees of freedom. This paper studies the LSNN method for scalar nonlinear
hyperbolic conservation law. The method is a discretization of an equivalent
least-squares (LS) formulation in the set of neural network functions with the
ReLU activation function. Evaluation of the LS functional is done by using
numerical integration and conservative finite volume scheme. Numerical results
of some test problems show that the method is capable of approximating the
discontinuous interface of the underlying problem automatically through the
free breaking lines of the ReLU neural network. Moreover, the method does not
exhibit the common Gibbs phenomena along the discontinuous interface.

    

### [[2106.00884] Deep Personalized Glucose Level Forecasting Using Attention-based Recurrent Neural Networks](http://arxiv.org/abs/2106.00884)


  In this paper, we study the problem of blood glucose forecasting and provide
a deep personalized solution. Predicting blood glucose level in people with
diabetes has significant value because health complications of abnormal glucose
level are serious, sometimes even leading to death. Therefore, having a model
that can accurately and quickly warn patients of potential problems is
essential. To develop a better deep model for blood glucose forecasting, we
analyze the data and detect important patterns. These observations helped us to
propose a method that has several key advantages over existing methods: 1- it
learns a personalized model for each patient as well as a global model; 2- it
uses an attention mechanism and extracted time features to better learn
long-term dependencies in the data; 3- it introduces a new, robust training
procedure for time series data. We empirically show the efficacy of our model
on a real dataset.

    

### [[2106.03932] How to Design a Three-Stage Architecture for Audio-Visual Active Speaker Detection in the Wild](http://arxiv.org/abs/2106.03932)


  Successful active speaker detection requires a three-stage pipeline: (i)
audio-visual encoding for all speakers in the clip, (ii) inter-speaker relation
modeling between a reference speaker and the background speakers within each
frame, and (iii) temporal modeling for the reference speaker. Each stage of
this pipeline plays an important role for the final performance of the created
architecture. Based on a series of controlled experiments, this work presents
several practical guidelines for audio-visual active speaker detection.
Correspondingly, we present a new architecture called ASDNet, which achieves a
new state-of-the-art on the AVA-ActiveSpeaker dataset with a mAP of 93.5%
outperforming the second best with a large margin of 4.7%. Our code and
pretrained models are publicly available.

    

### [[2106.10796] CD-SGD: Distributed Stochastic Gradient Descent with Compression and Delay Compensation](http://arxiv.org/abs/2106.10796)


  Communication overhead is the key challenge for distributed training.
Gradient compression is a widely used approach to reduce communication traffic.
When combining with parallel communication mechanism method like pipeline,
gradient compression technique can greatly alleviate the impact of
communication overhead. However, there exists two problems of gradient
compression technique to be solved. Firstly, gradient compression brings in
extra computation cost, which will delay the next training iteration. Secondly,
gradient compression usually leads to the decrease of convergence accuracy.

    

### [[2109.02517] Error Controlled Actor-Critic](http://arxiv.org/abs/2109.02517)


  On error of value function inevitably causes an overestimation phenomenon and
has a negative impact on the convergence of the algorithms. To mitigate the
negative effects of the approximation error, we propose Error Controlled
Actor-critic which ensures confining the approximation error in value function.
We present an analysis of how the approximation error can hinder the
optimization process of actor-critic methods.Then, we derive an upper boundary
of the approximation error of Q function approximator and find that the error
can be lowered by restricting on the KL-divergence between every two
consecutive policies when training the policy. The results of experiments on a
range of continuous control tasks demonstrate that the proposed actor-critic
algorithm apparently reduces the approximation error and significantly
outperforms other model-free RL algorithms.

    

### [[2109.03021] Limited Associativity Makes Concurrent Software Caches a Breeze](http://arxiv.org/abs/2109.03021)


  Software caches optimize the performance of diverse storage systems,
databases and other software systems. Existing works on software caches
automatically resort to fully associative cache designs. Our work shows that
limited associativity caches are a promising direction for concurrent software
caches. Specifically, we demonstrate that limited associativity enables simple
yet efficient realizations of multiple cache management schemes that can be
trivially parallelized. We show that the obtained hit ratio is usually similar
to fully associative caches of the same management policy, but the throughput
is improved by up to X5 compared to production-grade caching libraries,
especially in multi-threaded executions.

    

### [[2109.03022] Augmented Memory Computing: Dynamically Augmented SRAM Storage for Data Intensive Applications](http://arxiv.org/abs/2109.03022)


  In this paper, we propose a novel memory-centric scheme based on CMOS SRAM
for acceleration of data intensive applications. Our proposal aims at
dynamically increasing the on-chip memory storage capacity of SRAM arrays
on-demand. The proposed scheme called - Augmented Memory Computing allows an
SRAM cell to operate in two different modes 1) the Normal mode and 2) the
Augmented mode. In the Normal mode of operation, the SRAM cell functions like a
standard 6 transistor (6T) SRAM cell, storing one bit of data in static format.
While in the Augmented mode, each SRAM cell can store >1 bit of data (in a
dynamic fashion). Specifically, we propose two novel SRAM cells - an 8
transistor (8T) dual bit storage augmented cell and a 7 transistor (7T) ternary
bit storage augmented cell. The proposed 8T dual bit SRAM cell when operated in
the Augmented mode, can store a static bit of data while also, simultaneously,
storing another bit in a dynamic form. Thus, when operated in Augmented mode,
the 8T SRAM cell can store two bits of data - one SRAM-like data and one
DRAM-like data, thereby increasing or augmenting the memory storage capacity.
On the other hand, the proposed 7T ternary bit storage augmented cell can
either store a single SRAM data in Normal mode or can be configured to operate
in Augmented mode, wherein it can store ternary data (3 levels (0,0), (0,1),
(1,0)) in a dynamic manner. Thus, based on the mode of operation, the proposed
augmented memory bit-cells can either store one static bit of data or >1 bit of
data in a dynamic format. We show the feasibility of our proposed bit-cells
through extensive simulations at Globalfoundries 22nm FDX node. It is worth
mentioning, the novel scheme of augmented memory bit-cells can be seamlessly
combined with existing in-memory computing approaches for added energy and
throughput benefits.

    

### [[2109.03024] Versa: A Dataflow-Centric Multiprocessor with 36 Systolic ARM Cortex-M4F Cores and a Reconfigurable Crossbar-Memory Hierarchy in 28nm](http://arxiv.org/abs/2109.03024)


  We present Versa, an energy-efficient processor with 36 systolic ARM
Cortex-M4F cores and a runtime-reconfigurable memory hierarchy. Versa exploits
algorithm-specific characteristics in order to optimize bandwidth, access
latency, and data reuse. Measured on a set of kernels with diverse data access,
control, and synchronization characteristics, reconfiguration between different
Versa modes yields median energy-efficiency improvements of 11.6x and 37.2x
over mobile CPU and GPU baselines, respectively.

    

### [[2109.03026] High-Resolution Waveform Capture Device on a Cyclone-V FPGA](http://arxiv.org/abs/2109.03026)


  We introduce the waveform capture device (WCD), a flexible measurement system
capable of recording complex digital signals on trillionth-of-a-second (ps)
time scales. The WCD is implemented via modular code on an off-the-shelf
field-programmable gate-array (FPGA, Intel/Altera Cyclone V), and incorporates
both time-to-digital converter (TDC) and digital storage oscilloscope (DSO)
functionality. The device captures a waveform by taking snapshots of a signal
as it propagates down an ultra-fast transmission line known as a carry chain
(CC). It is calibrated via a novel dynamic phase-shifting (DPS) method that
requires substantially less data and resources than the state-of-the-art. Using
DPS, we find the measurement resolution - or mean propagation delay from one CC
element to the next - to be 4.91 +/- 0.04 ps (4.54 +/- 0.02 ps) for a pulse of
logic high (low). Similarly, we find the single-shot precision - or mean error
on the timing of the waveform - to be 29.52 ps (27.14 ps) for pulses of logic
high (low). We verify these findings by reproducing commercial oscilloscope
measurements of asynchronous ring-oscillators on FPGAs, finding the mean pulse
width to be 0.240 +/- 0.002 ns per inverter gate. Finally, we present a careful
analysis of design constraints, introduce a novel error correction algorithm,
and sketch a simple extension to the analog domain. We also provide the Verilog
code instantiating the our design on an FPGA in an Appendix, and make our
methods available as an open-source Python library at
this https URL.

    

### [[2109.03041] Only Six Passive Circuit Elements Exist](http://arxiv.org/abs/2109.03041)


  We found that a second-order ideal memristor degenerates into a negative
nonlinear resistor. This phenomenon is quite similar to what are observed in
chemistry: a chemical element with a higher atomic number is unstable and may
decay radioactively into another chemical element with a lower atomic number.
After extending the above local activity to other higher-order circuit
elements, we concluded that all higher-order passive memory circuit elements do
not exist in nature and that the periodic table of the two-terminal passive
circuit elements can be dramatically reduced to a six-pointed star comprising
only six passive elements. Such a bounded table may mark the end of the hunt
for missing higher-order passive circuit elements predicted nearly 40 years
ago.

    

### [[2109.03112] Efficient Instruction Scheduling using Real-time Load Delay Tracking](http://arxiv.org/abs/2109.03112)


  Many hardware structures in today's high-performance out-of-order processors
do not scale in an efficient way. To address this, different solutions have
been proposed that build execution schedules in an energy-efficient manner.
Issue time prediction processors are one such solution that use data-flow
dependencies and predefined instruction latencies to predict issue times of
repeated instructions. In this work, we aim to improve their accuracy, and
consequently their performance, in an energy efficient way. We accomplish this
by taking advantage of two key observations. First, memory accesses often take
additional time to arrive than the static, predefined access latency that is
used to describe these systems. Second, we find that these memory access delays
often repeat across iterations of the same code. This, in turn, allows us to
predict the arrival time of these accesses.
In this work, we introduce a new processor microarchitecture, that replaces a
complex reservation-station-based scheduler with an efficient, scalable
alternative. Our proposed scheduling technique tracks real-time delays of loads
to accurately predict instruction issue times, and uses a reordering mechanism
to prioritize instructions based on that prediction, achieving
close-to-out-of-order processor performance. To accomplish this in an
energy-efficient manner we introduce: (1) an instruction delay learning
mechanism that monitors repeated load instructions and learns their latest
delay, (2) an issue time predictor that uses learned delays and data-flow
dependencies to predict instruction issue times and (3) priority queues that
reorder instructions based on their issue time prediction. Together, our
processor achieves 86.2% of the performance of a traditional out-of-order
processor, higher than previous efficient scheduler proposals, while still
consuming 30% less power.

    

### [[2109.02754] In-situ visualization of natural hazards with Galaxy and Material Point Method](http://arxiv.org/abs/2109.02754)


  Visualizing regional-scale landslides is the key to conveying the threat of
natural hazards to stakeholders and policymakers. Traditional visualization
techniques are restricted to post-processing a limited subset of simulation
data and are not scalable to rendering exascale models with billions of
particles. In-situ visualization is a technique of rendering simulation data in
real-time, i.e., rendering visuals in tandem while the simulation is running.
In this study, we develop a scalable N:M interface architecture to visualize
regional-scale landslides. We demonstrate the scalability of the architecture
by simulating the long runout of the 2014 Oso landslide using the Material
Point Method coupled with the Galaxy ray tracing engine rendering 4.2 million
material points as spheres. In-situ visualization has an amortized runtime
increase of 2% compared to non-visualized simulations. The developed approach
can achieve in-situ visualization of regional-scale landslides with billions of
particles with minimal impact on the simulation process.

    

### [[2109.02922] Memory at Your Service: Fast Memory Allocation for Latency-critical Services](http://arxiv.org/abs/2109.02922)


  Co-location and memory sharing between latency-critical services, such as
key-value store and web search, and best-effort batch jobs is an appealing
approach to improving memory utilization in multi-tenant datacenter systems.
However, we find that the very diverse goals of job co-location and the
GNU/Linux system stack can lead to severe performance degradation of
latency-critical services under memory pressure in a multi-tenant system. We
address memory pressure for latency-critical services via fast memory
allocation and proactive reclamation. We find that memory allocation latency
dominates the overall query latency, especially under memory pressure. We
analyze the default memory management mechanism provided by GNU/Linux system
stack and identify the reasons why it is inefficient for latency-critical
services in a multi-tenant system. We present Hermes, a fast memory allocation
mechanism in user space that adaptively reserves memory for latency-critical
services. It advises Linux OS to proactively reclaim memory of batch jobs. We
implement Hermes in GNU C Library. Experimental result shows that Hermes
reduces the average and the $99^{th}$ percentile memory allocation latency by
up to 54.4% and 62.4% for a micro benchmark, respectively. For two real-world
latency-critical services, Hermes reduces both the average and the $99^{th}$
percentile tail query latency by up to 40.3%. Compared to the default Glibc,
jemalloc and TCMalloc, Hermes reduces Service Level Objective violation by up
to 84.3% under memory pressure.

    

### [[2002.07672] Structural Invariants for the Verification of Systems with Parameterized Architectures](http://arxiv.org/abs/2002.07672)


  We consider parameterized concurrent systems consisting of a finite but
unknown number of components, obtained by replicating a given set of finite
state automata. Components communicate by executing atomic interactions whose
participants update their states simultaneously. We introduce an interaction
logic to specify both the type of interactions (e.g.\ rendez-vous, broadcast)
and the topology of the system (e.g.\ pipeline, ring). The logic can be easily
embedded in monadic second order logic of finitely many successors, and is
therefore decidable.
Proving safety properties of such a parameterized system, like deadlock
freedom or mutual exclusion, requires to infer an inductive invariant that
contains all reachable states of all system instances, and no unsafe state. We
present a method to automatically synthesize inductive invariants directly from
the formula describing the interactions, without costly fixed point iterations.
We experimentally prove that this invariant is strong enough to verify safety
properties of a large number of systems including textbook examples (dining
philosophers, synchronization schemes), classical mutual exclusion algorithms,
cache-coherence protocols and self-stabilization algorithms, for an arbitrary
number of components.

    

### [[2004.10908] Taskflow: A Lightweight Parallel and Heterogeneous Task Graph Computing System](http://arxiv.org/abs/2004.10908)


  Taskflow aims to streamline the building of parallel and heterogeneous
applications using a lightweight task graph-based approach. Taskflow introduces
an expressive task graph programming model to assist developers in the
implementation of parallel and heterogeneous decomposition strategies on a
heterogeneous computing platform. Our programming model distinguishes itself as
a very general class of task graph parallelism with in-graph control flow to
enable end-to-end parallel optimization. To support our model with high
performance, we design an efficient system runtime that solves many of the new
scheduling challenges arising out of our models and optimizes the performance
across latency, energy efficiency, and throughput. We have demonstrated the
promising performance of Taskflow in real-world applications. As an example,
Taskflow solves a large-scale machine learning workload up to 29% faster, 1.5x
less memory, and 1.9x higher throughput than the industrial system, oneTBB, on
a machine of 40 CPUs and 4 GPUs. We have opened the source of Taskflow and
deployed it to large numbers of users in the open-source community.

    

### [[2109.02772] An Empirical Study on Few-shot Knowledge Probing for Pretrained Language Models](http://arxiv.org/abs/2109.02772)


  Prompt-based knowledge probing for 1-hop relations has been used to measure
how much world knowledge is stored in pretrained language models. Existing work
uses considerable amounts of data to tune the prompts for better performance.
In this work, we compare a variety of approaches under a few-shot knowledge
probing setting, where only a small number (e.g., 10 or 20) of example triples
are available. In addition, we create a new dataset named TREx-2p, which
contains 2-hop relations. We report that few-shot examples can strongly boost
the probing performance for both 1-hop and 2-hop relations. In particular, we
find that a simple-yet-effective approach of finetuning the bias vectors in the
model outperforms existing prompt-engineering methods. Our dataset and code are
available at \url{this https URL}.

    

### [[2109.02806] Symbolic Computation in Software Science: My Personal View](http://arxiv.org/abs/2109.02806)


  In this note, I develop my personal view on the scope and relevance of
symbolic computation in software science. For this, I discuss the interaction
and differences between symbolic computation, software science, automatic
programming, mathematical knowledge management, artificial intelligence,
algorithmic intelligence, numerical computation, and machine learning. In the
discussion of these notions, I allow myself to refer also to papers (1982,
1985, 2001, 2003, 2013) of mine in which I expressed my views on these areas at
early stages of some of these fields.

    

### [[2109.02823] Robot Sound Interpretation: Learning Visual-Audio Representations for Voice-Controlled Robots](http://arxiv.org/abs/2109.02823)


  Inspired by sensorimotor theory, we propose a novel pipeline for
voice-controlled robots. Previous work relies on explicit labels of sounds and
images as well as extrinsic reward functions. Not only do such approaches have
little resemblance to human sensorimotor development, but also require
hand-tuning rewards and extensive human labor. To address these problems, we
learn a representation that associates images and sound commands with minimal
supervision. Using this representation, we generate an intrinsic reward
function to learn robotic tasks with reinforcement learning. We demonstrate our
approach on three robot platforms, a TurtleBot3, a Kuka-IIWA arm, and a Kinova
Gen3 robot, which hear a command word, identify the associated target object,
and perform precise control to approach the target. We show that our method
outperforms previous work across various sound types and robotic tasks
empirically. We successfully deploy the policy learned in simulator to a
real-world Kinova Gen3.

    

### [[2109.02843] A new neighborhood structure for job shop scheduling problems](http://arxiv.org/abs/2109.02843)


  Job shop scheduling problem (JSP) is a widely studied NP-complete
combinatorial optimization problem. Neighborhood structures play a critical
role in solving JSP. At present, there are three state-of-the-art neighborhood
structures, i.e., N5, N6, and N7. Improving the upper bounds of some famous
benchmarks is inseparable from the role of these neighborhood structures.
However, these existing neighborhood structures only consider the movement of
critical operations within a critical block. According to our experiments, it
is also possible to improve the makespan of a scheduling scheme by moving a
critical operation outside its critical block. According to the above finding,
this paper proposes a new N8 neighborhood structure considering the movement of
critical operations within a critical block and the movement of critical
operations outside the critical block. Besides, a neighborhood clipping method
is designed to avoid invalid movement, reducing the computational time. Tabu
search (TS) is a commonly used algorithm framework combined with neighborhood
structures. This paper uses this framework to compare the N8 neighborhood
structure with N5, N6, and N7 neighborhood structures on four famous
benchmarks. The experimental results verify that the N8 neighborhood structure
is more effective and efficient in solving JSP than the other state-of-the-art
neighborhood structures.

    

### [[2109.02860] GCsT: Graph Convolutional Skeleton Transformer for Action Recognition](http://arxiv.org/abs/2109.02860)


  Graph convolutional networks (GCNs) achieve promising performance for
skeleton-based action recognition. However, in most GCN-based methods, the
spatial-temporal graph convolution is strictly restricted by the graph topology
while only captures the short-term temporal context, thus lacking the
flexibility of feature extraction. In this work, we present a novel
architecture, named Graph Convolutional skeleton Transformer (GCsT), which
addresses limitations in GCNs by introducing Transformer. Our GCsT employs all
the benefits of Transformer (i.e. dynamical attention and global context) while
keeps the advantages of GCNs (i.e. hierarchy and local topology structure). In
GCsT, the spatial-temporal GCN forces the capture of local dependencies while
Transformer dynamically extracts global spatial-temporal relationships.
Furthermore, the proposed GCsT shows stronger expressive capability by adding
additional information present in skeleton sequences. Incorporating the
Transformer allows that information to be introduced into the model almost
effortlessly. We validate the proposed GCsT by conducting extensive
experiments, which achieves the state-of-the-art performance on NTU RGB+D, NTU
RGB+D 120 and Northwestern-UCLA datasets.

    

### [[2109.02866] Readying Medical Students for Medical AI: The Need to Embed AI Ethics Education](http://arxiv.org/abs/2109.02866)


  Medical students will almost inevitably encounter powerful medical AI systems
early in their careers. Yet, contemporary medical education does not adequately
equip students with the basic clinical proficiency in medical AI needed to use
these tools safely and effectively. Education reform is urgently needed, but
not easily implemented, largely due to an already jam-packed medical curricula.
In this article, we propose an education reform framework as an effective and
efficient solution, which we call the Embedded AI Ethics Education Framework.
Unlike other calls for education reform to accommodate AI teaching that are
more radical in scope, our framework is modest and incremental. It leverages
existing bioethics or medical ethics curricula to develop and deliver content
on the ethical issues associated with medical AI, especially the harms of
technology misuse, disuse, and abuse that affect the risk-benefit analyses at
the heart of healthcare. In doing so, the framework provides a simple tool for
going beyond the "What?" and the "Why?" of medical AI ethics education, to
answer the "How?", giving universities, course directors, and/or professors a
broad road-map for equipping their students with the necessary clinical
proficiency in medical AI.

    

### [[2109.02899] Blockchains through ontologies: the case study of the Ethereum ERC721 standard in \ONT{} (Extended Version)](http://arxiv.org/abs/2109.02899)


  Blockchains are gaining momentum due to the interest of industries and people
in \emph{decentralized applications} (Dapps), particularly in those for trading
assets through digital certificates secured on blockchain, called tokens. As a
consequence, providing a clear unambiguous description of any activities
carried out on blockchains has become crucial, and we feel the urgency to
achieve that description at least for trading. This paper reports on how to
leverage the \emph{Ontology for Agents, Systems, and Integration of Services}
("\ONT{}") as a general means for the semantic representation of smart
contracts stored on blockchain as software agents. Special attention is paid to
non-fungible tokens (NFTs), whose management through the ERC721 standard is
presented as a case study.

    

### [[2109.02903] IndicBART: A Pre-trained Model for Natural Language Generation of Indic Languages](http://arxiv.org/abs/2109.02903)


  In this paper we present IndicBART, a multilingual, sequence-to-sequence
pre-trained model focusing on 11 Indic languages and English. Different from
existing pre-trained models, IndicBART utilizes the orthographic similarity
between Indic scripts to improve transfer learning between similar Indic
languages. We evaluate IndicBART on two NLG tasks: Neural Machine Translation
(NMT) and extreme summarization. Our experiments on NMT for 12 language pairs
and extreme summarization for 7 languages using multilingual fine-tuning show
that IndicBART is competitive with or better than mBART50 despite containing
significantly fewer parameters. Our analyses focus on identifying the impact of
script unification (to Devanagari), corpora size as well as multilingualism on
the final performance. The IndicBART model is available under the MIT license
at this https URL .

    

### [[2109.02938] Naturalness Evaluation of Natural Language Generation in Task-oriented Dialogues using BERT](http://arxiv.org/abs/2109.02938)


  This paper presents an automatic method to evaluate the naturalness of
natural language generation in dialogue systems. While this task was previously
rendered through expensive and time-consuming human labor, we present this
novel task of automatic naturalness evaluation of generated language. By
fine-tuning the BERT model, our proposed naturalness evaluation method shows
robust results and outperforms the baselines: support vector machines,
bi-directional LSTMs, and BLEURT. In addition, the training speed and
evaluation performance of naturalness model are improved by transfer learning
from quality and informativeness linguistic knowledge.

    

### [[2109.02944] Dutch Comfort: The limits of AI governance through municipal registers](http://arxiv.org/abs/2109.02944)


  In this commentary, we respond to a recent editorial letter by Professor
Luciano Floridi entitled 'AI as a public service: Learning from Amsterdam and
Helsinki'. Here, Floridi considers the positive impact of these municipal AI
registers, which collect a limited number of algorithmic systems used by the
city of Amsterdam and Helsinki. There are a number of assumptions about AI
registers as a governance model for automated systems that we seek to question.
Starting with recent attempts to normalize AI by decontextualizing and
depoliticizing it, which is a fraught political project that encourages what we
call 'ethics theater' given the proven dangers of using these systems in the
context of the digital welfare state. We agree with Floridi that much can be
learned from these registers about the role of AI systems in municipal city
management. Yet, the lessons we draw, on the basis of our extensive
ethnographic engagement with digital well-fare states are distinctly less
optimistic.

    

### [[2109.02956] Smart Automotive Technology Adherence to the Law: (De)Constructing Road Rules for Autonomous System Development, Verification and Safety](http://arxiv.org/abs/2109.02956)


  Driving is an intuitive task that requires skills, constant alertness and
vigilance for unexpected events. The driving task also requires long
concentration spans focusing on the entire task for prolonged periods, and
sophisticated negotiation skills with other road users, including wild animals.
These requirements are particularly important when approaching intersections,
overtaking, giving way, merging, turning and while adhering to the vast body of
road rules. Modern motor vehicles now include an array of smart assistive and
autonomous driving systems capable of subsuming some, most, or in limited
cases, all of the driving task. The UK Department of Transport's response to
the Safe Use of Automated Lane Keeping System consultation proposes that these
systems are tested for compliance with relevant traffic rules. Building these
smart automotive systems requires software developers with highly technical
software engineering skills, and now a lawyer's in-depth knowledge of traffic
legislation as well. These skills are required to ensure the systems are able
to safely perform their tasks while being observant of the law. This paper
presents an approach for deconstructing the complicated legalese of traffic law
and representing its requirements and flow. The approach (de)constructs road
rules in legal terminology and specifies them in structured English logic that
is expressed as Boolean logic for automation and Lawmaps for visualisation. We
demonstrate an example using these tools leading to the construction and
validation of a Bayesian Network model. We strongly believe these tools to be
approachable by programmers and the general public, and capable of use in
developing Artificial Intelligence to underpin motor vehicle smart systems, and
in validation to ensure these systems are considerate of the law when making
decisions.

    

### [[2109.03004] Empathetic Dialogue Generation with Pre-trained RoBERTa-GPT2 and External Knowledge](http://arxiv.org/abs/2109.03004)


  One challenge for dialogue agents is to recognize feelings of the
conversation partner and respond accordingly. In this work, RoBERTa-GPT2 is
proposed for empathetic dialogue generation, where the pre-trained
auto-encoding RoBERTa is utilised as encoder and the pre-trained
auto-regressive GPT-2 as decoder. With the combination of the pre-trained
RoBERTa and GPT-2, our model realizes a new state-of-the-art emotion accuracy.
To enable the empathetic ability of RoBERTa-GPT2 model, we propose a
commonsense knowledge and emotional concepts extractor, in which the
commonsensible and emotional concepts of dialogue context are extracted for the
GPT-2 decoder. The experiment results demonstrate that the empathetic dialogue
generation benefits from both pre-trained encoder-decoder architecture and
external knowledge.

    

### [[2109.03009] Sequential Attention Module for Natural Language Processing](http://arxiv.org/abs/2109.03009)


  Recently, large pre-trained neural language models have attained remarkable
performance on many downstream natural language processing (NLP) applications
via fine-tuning. In this paper, we target at how to further improve the token
representations on the language models. We, therefore, propose a simple yet
effective plug-and-play module, Sequential Attention Module (SAM), on the token
embeddings learned from a pre-trained language model. Our proposed SAM consists
of two main attention modules deployed sequentially: Feature-wise Attention
Module (FAM) and Token-wise Attention Module (TAM). More specifically, FAM can
effectively identify the importance of features at each dimension and promote
the effect via dot-product on the original token embeddings for downstream NLP
applications. Meanwhile, TAM can further re-weight the features at the
token-wise level. Moreover, we propose an adaptive filter on FAM to prevent
noise impact and increase information absorption. Finally, we conduct extensive
experiments to demonstrate the advantages and properties of our proposed SAM.
We first show how SAM plays a primary role in the champion solution of two
subtasks of SemEval'21 Task 7. After that, we apply SAM on sentiment analysis
and three popular NLP tasks and demonstrate that SAM consistently outperforms
the state-of-the-art baselines.

    

### [[2109.03034] Generate & Rank: A Multi-task Framework for Math Word Problems](http://arxiv.org/abs/2109.03034)


  Math word problem (MWP) is a challenging and critical task in natural
language processing. Many recent studies formalize MWP as a generation task and
have adopted sequence-to-sequence models to transform problem descriptions to
mathematical expressions. However, mathematical expressions are prone to minor
mistakes while the generation objective does not explicitly handle such
mistakes. To address this limitation, we devise a new ranking task for MWP and
propose Generate & Rank, a multi-task framework based on a generative
pre-trained language model. By joint training with generation and ranking, the
model learns from its own mistakes and is able to distinguish between correct
and incorrect expressions. Meanwhile, we perform tree-based disturbance
specially designed for MWP and an online update to boost the ranker. We
demonstrate the effectiveness of our proposed method on the benchmark and the
results show that our method consistently outperforms baselines in all
datasets. Particularly, in the classical Math23k, our method is 7% (78.4%
$\rightarrow$ 85.4%) higher than the state-of-the-art.

    

### [[2109.03039] POSSCORE: A Simple Yet Effective Evaluation of Conversational Search with Part of Speech Labelling](http://arxiv.org/abs/2109.03039)


  Conversational search systems, such as Google Assistant and Microsoft
Cortana, provide a new search paradigm where users are allowed, via natural
language dialogues, to communicate with search systems. Evaluating such systems
is very challenging since search results are presented in the format of natural
language sentences. Given the unlimited number of possible responses,
collecting relevance assessments for all the possible responses is infeasible.
In this paper, we propose POSSCORE, a simple yet effective automatic evaluation
method for conversational search. The proposed embedding-based metric takes the
influence of part of speech (POS) of the terms in the response into account. To
the best knowledge, our work is the first to systematically demonstrate the
importance of incorporating syntactic information, such as POS labels, for
conversational search evaluation. Experimental results demonstrate that our
metrics can correlate with human preference, achieving significant improvements
over state-of-the-art baseline metrics.

    

### [[2109.03084] Learning grounded word meaning representations on similarity graphs](http://arxiv.org/abs/2109.03084)


  This paper introduces a novel approach to learn visually grounded meaning
representations of words as low-dimensional node embeddings on an underlying
graph hierarchy. The lower level of the hierarchy models modality-specific word
representations through dedicated but communicating graphs, while the higher
level puts these representations together on a single graph to learn a
representation jointly from both modalities. The topology of each graph models
similarity relations among words, and is estimated jointly with the graph
embedding. The assumption underlying this model is that words sharing similar
meaning correspond to communities in an underlying similarity graph in a
low-dimensional space. We named this model Hierarchical Multi-Modal Similarity
Graph Embedding (HM-SGE). Experimental results validate the ability of HM-SGE
to simulate human similarity judgements and concept categorization,
outperforming the state of the art.

    

### [[2109.03089] Distributed Allocation and Scheduling of Tasks with Cross-Schedule Dependencies for Heterogeneous Multi-Robot Teams](http://arxiv.org/abs/2109.03089)


  To enable safe and efficient use of multi-robot systems in everyday life, a
robust and fast method for coordinating their actions must be developed. In
this paper, we present a distributed task allocation and scheduling algorithm
for missions where the tasks of different robots are tightly coupled with
temporal and precedence constraints. The approach is based on representing the
problem as a variant of the vehicle routing problem, and the solution is found
using a distributed metaheuristic algorithm based on evolutionary computation
(CBM-pop). Such an approach allows a fast and near-optimal allocation and can
therefore be used for online replanning in case of task changes. Simulation
results show that the approach has better computational speed and scalability
without loss of optimality compared to the state-of-the-art distributed
methods. An application of the planning procedure to a practical use case of a
greenhouse maintained by a multi-robot system is given.

    

### [[2109.03106] Fudge: A light-weight solver for abstract argumentation based on SAT reductions](http://arxiv.org/abs/2109.03106)


  We present Fudge, an abstract argumentation solver that tightly integrates
satisfiability solving technology to solve a series of abstract argumentation
problems. While most of the encodings used by Fudge derive from standard
translation approaches, Fudge makes use of completely novel encodings to solve
the skeptical reasoning problem wrt. preferred semantics and problems wrt.
ideal semantics.

    

### [[2109.03162] The pyglaf argumentation reasoner (ICCMA2021)](http://arxiv.org/abs/2109.03162)


  The pyglaf reasoner takes advantage of circumscription to solve computational
problems of abstract argumentation frameworks. In fact, many of these problems
are reduced to circumscription by means of linear encodings, and a few others
are solved by means of a sequence of calls to an oracle for circumscription.
Within pyglaf, Python is used to build the encodings and to control the
execution of the external circumscription solver, which extends the SAT solver
glucose and implements algorithms taking advantage of unsatisfiable core
analysis and incremental computation.

    

### [[2109.03166] Aspartix-V21](http://arxiv.org/abs/2109.03166)


  In this solver description we present ASPARTIX-V, in its 2021 edition, which
participates in the International Competition on Computational Models of
Argumentation (ICCMA) 2021. ASPARTIX-V is capable of solving all classical
(static) reasoning tasks part of ICCMA'21 and extends the ASPARTIX system suite
by incorporation of recent ASP language constructs (e.g. conditional literals),
domain heuristics within ASP, and multi-shot methods. In this light ASPARTIX-V
deviates from the traditional focus of ASPARTIX on monolithic approaches (i.e.,
one-shot solving via a single ASP encoding) to further enhance performance.

    

### [[2109.03181] IEEE BigData 2021 Cup: Soft Sensing at Scale](http://arxiv.org/abs/2109.03181)


  IEEE BigData 2021 Cup: Soft Sensing at Scale is a data mining competition
organized by Seagate Technology, in association with the IEEE BigData 2021
conference. The scope of this challenge is to tackle the task of classifying
soft sensing data with machine learning techniques. In this paper we go into
the details of the challenge and describe the data set provided to
participants. We define the metrics of interest, baseline models, and describe
approaches we found meaningful which may be a good starting point for further
analysis. We discuss the results obtained with our approaches and give insights
on what potential challenges participants may run into. Students, researchers,
and anyone interested in working on a major industrial problem are welcome to
participate in the challenge!

    

### [[2109.03202] On the impact of MDP design for Reinforcement Learning agents in Resource Management](http://arxiv.org/abs/2109.03202)


  The recent progress in Reinforcement Learning applications to Resource
Management presents MDPs without a deeper analysis of the impacts of design
decisions on agent performance. In this paper, we compare and contrast four
different MDP variations, discussing their computational requirements and
impacts on agent performance by means of an empirical analysis. We conclude by
showing that, in our experiments, when using Multi-Layer Perceptrons as
approximation function, a compact state representation allows transfer of
agents between environments, and that transferred agents have good performance
and outperform specialized agents in 80\% of the tested scenarios, even without
retraining.

    

### [[1901.09127] Strong Equivalence and Program Structure in Arguing Essential Equivalence between Logic Programs](http://arxiv.org/abs/1901.09127)


  Answer set programming is a prominent declarative programming paradigm used
in formulating combinatorial search problems and implementing different
knowledge representation formalisms. Frequently, several related and yet
substantially different answer set programs exist for a given problem.
Sometimes these encodings may display significantly different performance.
Uncovering precise formal links between these programs is often important and
yet far from trivial. This paper presents formal results carefully relating a
number of interesting program rewritings. It also provides the proof of
correctness of system Projector concerned with automatic program rewritings for
the sake of efficiency. Under consideration in Theory and Practice of Logic
Programming (TPLP).

    

### [[1903.10559] The Mode of Computing](http://arxiv.org/abs/1903.10559)


  The Turing Machine is the paradigmatic case of computing machines, but there
are others such as Artificial Neural Networks, quantum computing, holography,
and diverse forms of analogical computing, each based on a particular intuition
of the phenomenon of computing. This variety can be captured in terms of system
levels, re-interpreting and generalizing Newell's hierarchy, which includes the
knowledge level at the top and the symbol level immediately below it. In this
re-interpretation the knowledge level consists of human knowledge and the
symbol level is generalized into a new level that here is called The Mode of
Computing. Natural computing performed by brains of humans and non-human
animals with a developed enough neural system should be understood in terms of
a hierarchy of system levels too. By analogy from standard computing machinery
there must be a system level above the neural circuitry and directly below the
knowledge level that is named here The mode of Natural Computing. A central
question for Cognition is the characterization of this mode. The Mode of
Computing provides a novel perspective on the phenomena of computing,
interpreting, the representational and non-representational views of cognition,
and consciousness.

    

### [[2010.12619] Learning Implicitly with Noisy Data in Linear Arithmetic](http://arxiv.org/abs/2010.12619)


  Robust learning in expressive languages with real-world data continues to be
a challenging task. Numerous conventional methods appeal to heuristics without
any assurances of robustness. While probably approximately correct (PAC)
Semantics offers strong guarantees, learning explicit representations is not
tractable, even in propositional logic. However, recent work on so-called
"implicit" learning has shown tremendous promise in terms of obtaining
polynomial-time results for fragments of first-order logic. In this work, we
extend implicit learning in PAC-Semantics to handle noisy data in the form of
intervals and threshold uncertainty in the language of linear arithmetic. We
prove that our extended framework keeps the existing polynomial-time complexity
guarantees. Furthermore, we provide the first empirical investigation of this
hitherto purely theoretical framework. Using benchmark problems, we show that
our implicit approach to learning optimal linear programming objective
constraints significantly outperforms an explicit approach in practice.

    

### [[2103.03598] WordBias: An Interactive Visual Tool for Discovering Intersectional Biases Encoded in Word Embeddings](http://arxiv.org/abs/2103.03598)


  Intersectional bias is a bias caused by an overlap of multiple social factors
like gender, sexuality, race, disability, religion, etc. A recent study has
shown that word embedding models can be laden with biases against
intersectional groups like African American females, etc. The first step
towards tackling such intersectional biases is to identify them. However,
discovering biases against different intersectional groups remains a
challenging task. In this work, we present WordBias, an interactive visual tool
designed to explore biases against intersectional groups encoded in static word
embeddings. Given a pretrained static word embedding, WordBias computes the
association of each word along different groups based on race, age, etc. and
then visualizes them using a novel interactive interface. Using a case study,
we demonstrate how WordBias can help uncover biases against intersectional
groups like Black Muslim Males, Poor Females, etc. encoded in word embedding.
In addition, we also evaluate our tool using qualitative feedback from expert
interviews. The source code for this tool can be publicly accessed for
reproducibility at this http URL.

    

### [[2104.06172] On the Computational Intelligibility of Boolean Classifiers](http://arxiv.org/abs/2104.06172)


  In this paper, we investigate the computational intelligibility of Boolean
classifiers, characterized by their ability to answer XAI queries in polynomial
time. The classifiers under consideration are decision trees, DNF formulae,
decision lists, decision rules, tree ensembles, and Boolean neural nets. Using
9 XAI queries, including both explanation queries and verification queries, we
show the existence of large intelligibility gap between the families of
classifiers. On the one hand, all the 9 XAI queries are tractable for decision
trees. On the other hand, none of them is tractable for DNF formulae, decision
lists, random forests, boosted decision trees, Boolean multilayer perceptrons,
and binarized neural networks.

    

### [[2104.07228] Sentence-Permuted Paragraph Generation](http://arxiv.org/abs/2104.07228)


  Generating paragraphs of diverse contents is important in many applications.
Existing generation models produce similar contents from homogenized contexts
due to the fixed left-to-right sentence order. Our idea is permuting the
sentence orders to improve the content diversity of multi-sentence paragraph.
We propose a novel framework PermGen whose objective is to maximize the
expected log-likelihood of output paragraph distributions with respect to all
possible sentence orders. PermGen uses hierarchical positional embedding and
designs new procedures for training, decoding, and candidate ranking in the
sentence-permuted generation. Experiments on three paragraph generation
benchmarks demonstrate PermGen generates more diverse outputs with a higher
quality than existing models.

    

### [[2105.14452] A logic for binary classifiers and their explanation](http://arxiv.org/abs/2105.14452)


  Recent years have witnessed a renewed interest in Boolean function in
explaining binary classifiers in the field of explainable AI (XAI). The
standard approach of Boolean function is propositional logic. We study a family
of classifier models, axiomatize it and show completeness of our axiomatics.
Moreover, we prove that satisfiability checking for our modal language relative
to such a class of models is NP-complete. We leverage the language to formalize
counterfactual conditional as well as a variety of notions of explanation
including abductive, contrastive and counterfactual explanations, and biases.
Finally, we present two extensions of our language: a dynamic extension by the
notion of assignment enabling classifier change and an epistemic extension in
which the classifier's uncertainty about the actual input can be represented.

    

### [[2109.02529] ViSTA: a Framework for Virtual Scenario-based Testing of Autonomous Vehicles](http://arxiv.org/abs/2109.02529)


  In this paper, we present ViSTA, a framework for Virtual Scenario-based
Testing of Autonomous Vehicles (AV), developed as part of the 2021 IEEE
Autonomous Test Driving AI Test Challenge. Scenario-based virtual testing aims
to construct specific challenges posed for the AV to overcome, albeit in
virtual test environments that may not necessarily resemble the real world.
This approach is aimed at identifying specific issues that arise safety
concerns before an actual deployment of the AV on the road. In this paper, we
describe a comprehensive test case generation approach that facilitates the
design of special-purpose scenarios with meaningful parameters to form test
cases, both in automated and manual ways, leveraging the strength and
weaknesses of either. Furthermore, we describe how to automate the execution of
test cases, and analyze the performance of the AV under these test cases.

    

### [[2109.02810] An Inversion Tool for Conditional Term Rewriting Systems -- A Case Study of Ackermann Inversion](http://arxiv.org/abs/2109.02810)


  We report on an inversion tool for a class of oriented conditional
constructor term rewriting systems. Four well-behaved rule inverters ranging
from trivial to full, partial and semi-inverters are included. Conditional term
rewriting systems are theoretically well founded and can model functional and
non-functional rewrite relations. We illustrate the inversion by experiments
with full and partial inversions of the Ackermann function. The case study
demonstrates, among others, that polyvariant inversion and input-output set
propagation can reduce the search space of the generated inverse systems.

    

### [[2109.02812] Program Specialization as a Tool for Solving Word Equations](http://arxiv.org/abs/2109.02812)


  The paper focuses on the automatic generating of the witnesses for the word
equation satisfiability problem by means of specializing an interpreter which
tests whether a composition of variable substitutions of a given word equation
system produces its solution. We specialize such an interpreter w.r.t. the
equation system, while the substitutions are unknown. We show that several
variants of such interpreters, when specialized using the basic unfold/fold
specialization methods, are able to construct the whole solution sets for some
classes of the word equations whose left- and right-hand sides share variables.
We prove that the specialization process wrt the constructed interpreters gives
a simple syntactic criterion of the satisfiability of the equations considered,
and show that the suggested approach can solve some equations not solvable by
Z3str3 and CVC4, the widely-used SMT-solvers.

    

### [[2109.02813] Improving Dynamic Code Analysis by Code Abstraction](http://arxiv.org/abs/2109.02813)


  In this paper, our aim is to propose a model for code abstraction, based on
abstract interpretation, allowing us to improve the precision of a recently
proposed static analysis by abstract interpretation of dynamic languages. The
problem we tackle here is that the analysis may add some spurious code to the
string-to-execute abstract value and this code may need some abstract
representations in order to make it analyzable. This is precisely what we
propose here, where we drive the code abstraction by the analysis we have to
perform.

    

### [[2109.02814] An Empirical Study of Partial Deduction for miniKanren](http://arxiv.org/abs/2109.02814)


  We study conjunctive partial deduction, an advanced specialization technique
aimed at improving the performance of logic programs, in the context of
relational programming language miniKanren. We identify a number of issues,
caused by miniKanren peculiarities, and describe a novel approach to
specialization based on partial deduction and supercompilation. The results of
the evaluation demonstrate successful specialization of relational
interpreters. Although the project is at an early stage, we consider it as the
first step towards an efficient optimization framework for miniKanren.

    

### [[2109.02958] Multi-Level Quickening: Ten Years Later](http://arxiv.org/abs/2109.02958)


  This paper presents important performance improvements for interpreters,
exemplified by speedups of up to 5.5$\times$ for CPython. Although the original
version of this papers was rejected multiple times, the reported speedups have
not been achieved by any other interpreter optimization technique since. In
addition, the paper uses a sound evaluation methodology based on a corollary on
Amdahl's law to quantify the speedup potential of benchmarks, which also has
not been used in any other paper since.
This paper documents my best efforts, and includes all of the reviews the
paper received, plus some more commentary on my side on what has changed since
and what purpose the archived document could serve.

    

### [[2109.02991] Abstraction Logic: The Marriage of Contextual Refinement and Separation Logic](http://arxiv.org/abs/2109.02991)


  Contextual refinement and separation logics are successful verification
techniques that are very different in nature. First, the former guarantees
behavioral refinement between a concrete program and an abstract program while
the latter guarantees safety of a concrete program under certain conditions
(expressed in terms of pre and post conditions). Second, the former does not
allow any assumption about the context when locally reasoning about a module
while the latter allows rich assumptions. In this paper, we present a new
verification technique, called abstraction logic (AL), that inherently combines
contextual refinement and separation logics such as Iris and VST, thereby
taking the advantages of both. Specifically, AL allows us to locally verify a
concrete module against an abstract module under separation-logic-style pre and
post conditions about external modules. AL are fully formalized in Coq and
provides a proof mode that supports a combination of simulation-style reasoning
using our own tactics and SL-style reasoning using IPM (Iris Proof Mode). Using
the proof mode, we verified various examples to demonstrate reasoning about
ownership (based on partial commutative monoids) and purity ($i.e.$,
termination with no system call), cyclic and higher-order reasoning about
mutual recursion and function pointers, and reusable and gradual verification
via intermediate abstractions. Also, the verification results are combined with
CompCert, so that we formally establish behavioral refinement from top-level
abstract programs, all the way down to their assembly code.

    

### [[2109.03139] An Executable Structural Operational Formal Semantics for Python](http://arxiv.org/abs/2109.03139)


  Python is a popular high-level general-purpose programming language also
heavily used by the scientific community. It supports a variety of different
programming paradigms and is preferred by many for its ease of use. With the
vision of harvesting static analysis techniques like abstract interpretation
for Python, we develop a formal semantics for Python. A formal semantics is an
important cornerstone for any sound static analysis technique. We base our
efforts on the general framework of structural operational semantics yielding a
small-step semantics in principle allowing for concurrency and interaction with
an environment. The main contributions of this thesis are twofold: first, we
develop a meta-theoretic framework for the formalization of structural
operational semantics in tandem with the necessary tool support for the
automated derivation of interpreters from such formal semantics, and, second,
we validate the suitability of this approach for the formalization of modern
programming languages developing a semantics for Python.

    