
## 2021-9-9

### [<title>XGBoost4J - Scala dataframe to sparse dmatrix - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost4j-scala-dataframe-to-sparse-dmatrix/2457/3)

### [[2109.03284] Smart On-Chip Electromagnetic Environment](http://arxiv.org/abs/2109.03284)


  We introduce the concept of smart radio environments, currently extensively
studied for wireless communication in metasurface-programmable meter-scaled
environments (e.g., inside rooms), on the chip scale. Wired intra-chip
communication for information exchange between cores increasingly becomes a
computation-speed-bottleneck for modern multi-core chips. Wireless intra-chip
links with millimeter waves are a candidate technology to address this
challenge, but they currently face their own problems: the on-chip propagation
environment can be highly reverberant due to the metallic chip enclosure but
transceiver modules must be kept simple (on/off keying) such that long channel
impulse responses (CIRs) slow down the communication rate. Here, we overcome
this problem by endowing the on-chip propagation environment with in situ
programmability, allowing us to shape the CIR at will, and to impose, for
instance, a pulse-like CIR despite the strong multi-path environment. Using
full-wave simulations, we design a programmable metasurface suitable for
integration in the on-chip environment ("on-chip reconfigurable intelligent
surface"), and we demonstrate that the spatial control offered by the
metasurface allows us to shape the CIR profile. We envision (i) dynamic
multi-channel CIR shaping adapted to on-chip traffic patterns, (ii) analog
wave-based over-the-air computing inside the chip enclosure, and (iii) the
application of the explored concepts to off-chip communication inside racks,
inside the chassis of personal computers, etc.

    

### [[2109.03312] Self-adaptive Architectures in IoT Systems: A Systematic Literature Review](http://arxiv.org/abs/2109.03312)


  Over the past few years, the relevance of the Internet of Things (IoT) has
grown significantly and is now a key component of many industrial processes and
even a transparent participant in various activities performed in our daily
life. IoT systems are subjected to changes in the dynamic environments they
operate in. These changes (e.g. variations in the bandith consumption or new
devices joining/leaving) may impact the Quality of Service (QoS) of the IoT
system. A number of self-adaptation strategies for IoT architectures to better
deal with these changes have been proposed in the literature. Nevertheless,
they focus on isolated types of changes. We lack a comprehensive view of the
trade-offs of each proposal and how they could be combined to cope with dynamic
situations involving simultaneous types of events.
In this paper, we identify, analyze, and interpret relevant studies related
to IoT adaptation and develop a comprehensive and holistic view of the
interplay of different dynamic events, their consequences on the architecture
QoS, and the alternatives for the adaptation. To do so, we have conducted a
systematic literature review of existing scientific proposals and defined a
research agenda for the near future based on the findings and weaknesses
identified in the literature.

    

### [[2109.03328] Predicting Process Name from Network Data](http://arxiv.org/abs/2109.03328)


  The ability to identify applications based on the network data they generate
could be a valuable tool for cyber defense. We report on a machine learning
technique capable of using netflow-like features to predict the application
that generated the traffic. In our experiments, we used ground-truth labels
obtained from host-based sensors deployed in a large enterprise environment; we
applied random forests and multilayer perceptrons to the tasks of browser vs.
non-browser identification, browser fingerprinting, and process name
prediction. For each of these tasks, we demonstrate how machine learning models
can achieve high classification accuracy using only netflow-like features as
the basis for classification.

    

### [[2109.03395] From Cloud to Edge: A First Look at Public Edge Platforms](http://arxiv.org/abs/2109.03395)


  Public edge platforms have drawn increasing attention from both academia and
industry. In this study, we perform a first-of-its-kind measurement study on a
leading public edge platform that has been densely deployed in China. Based on
this measurement, we quantitatively answer two critical yet unexplored
questions. First, from end users' perspective, what is the performance of
commodity edge platforms compared to cloud, in terms of the end-to-end network
delay, throughput, and the application QoE. Second, from the edge service
provider's perspective, how are the edge workloads different from cloud, in
terms of their VM subscription, monetary cost, and resource usage. Our study
quantitatively reveals the status quo of today's public edge platforms, and
provides crucial insights towards developing and operating future edge
services.

    

### [[2109.03488] Partial Symbol Recovery for Interference Resilience in Low-Power Wide Area Networks](http://arxiv.org/abs/2109.03488)


  Recent years have witnessed the proliferation of Low-power Wide Area Networks
(LPWANs) in the unlicensed band for various Internet-of-Things (IoT)
applications. Due to the ultra-low transmission power and long transmission
duration, LPWAN devices inevitably suffer from high power Cross Technology
Interference (CTI), such as interference from Wi-Fi, coexisting in the same
spectrum. To alleviate this issue, this paper introduces the Partial Symbol
Recovery (PSR) scheme for improving the CTI resilience of LPWAN. We verify our
idea on LoRa, a widely adopted LPWAN technique, as a proof of concept. At the
PHY layer, although CTI has much higher power, its duration is relatively
shorter compared with LoRa symbols, leaving part of a LoRa symbol uncorrupted.
Moreover, due to its high redundancy, LoRa chips within a symbol are highly
correlated. This opens the possibility of detecting a LoRa symbol with only
part of the chips. By examining the unique frequency patterns in LoRa symbols
with time-frequency analysis, our design effectively detects the clean LoRa
chips that are free of CTI. This enables PSR to only rely on clean LoRa chips
for successfully recovering from communication failures. We evaluate our PSR
design with real-world testbeds, including SX1280 LoRa chips and USRP B210,
under Wi-Fi interference in various scenarios. Extensive experiments
demonstrate that our design offers reliable packet recovery performance,
successfully boosting the LoRa packet reception ratio from 45.2% to 82.2% with
a performance gain of 1.8 times.

    

### [[1912.01473] Age of Information in Random Access Channels](http://arxiv.org/abs/1912.01473)


  In applications of remote sensing, estimation, and control, timely
communication is not always ensured by high-rate communication. This work
proposes distributed age-efficient transmission policies for random access
channels with $M$ transmitters. In the first part of this work, we analyze the
age performance of stationary randomized policies by relating the problem of
finding age to the absorption time of a related Markov chain. In the second
part of this work, we propose the notion of \emph{age-gain} of a packet to
quantify how much the packet will reduce the instantaneous age of information
at the receiver side upon successful delivery. We then utilize this notion to
propose a transmission policy in which transmitters act in a distributed manner
based on the age-gain of their available packets. In particular, each
transmitter sends its latest packet only if its corresponding age-gain is
beyond a certain threshold which could be computed adaptively using the
collision feedback or found as a fixed value analytically in advance. Both
methods improve age of information significantly compared to the state of the
art. In the limit of large $M$, we prove that when the arrival rate is small
(below $\frac{1}{eM}$), slotted ALOHA-type algorithms are asymptotically
optimal. As the arrival rate increases beyond $\frac{1}{eM}$, while age
increases under slotted ALOHA, it decreases significantly under the proposed
age-based policies. For arrival rates $\theta$, $\theta=\frac{1}{o(M)}$, the
proposed algorithms provide a multiplicative factor of at least two compared to
the minimum age under slotted ALOHA (minimum over all arrival rates). We
conclude that, as opposed to the common practice, it is beneficial to increase
the sampling rate (and hence the arrival rate) and transmit packets selectively
based on their age-gain.

    

### [[2005.07301] Joint Planning of Network Slicing and Mobile Edge Computing: Models and Algorithms](http://arxiv.org/abs/2005.07301)


  Multi-access Edge Computing (MEC) facilitates the deployment of critical
applications with stringent QoS requirements, latency in particular. This paper
considers the problem of jointly planning the availability of computational
resources at the edge, the slicing of mobile network and edge computation
resources, and the routing of heterogeneous traffic types to the various
slices. These aspects are intertwined and must be addressed together to provide
the desired QoS to all mobile users and traffic types still keeping costs under
control. We formulate our problem as a mixed-integer nonlinear program (MINLP)
and we define a heuristic, named Neighbor Exploration and Sequential Fixing
(NESF), to facilitate the solution of the problem. The approach allows network
operators to fine tune the network operation cost and the total latency
experienced by users. We evaluate the performance of the proposed model and
heuristic against two natural greedy approaches. We show the impact of the
variation of all the considered parameters (viz., different types of traffic,
tolerable latency, network topology and bandwidth, computation and link
capacity) on the defined model. Numerical results demonstrate that NESF is very
effective, achieving near-optimal planning and resource allocation solutions in
a very short computing time even for large-scale network scenarios.

    

### [[2007.03652] Real-time Sampling and Estimation on Random Access Channels: Age of Information and Beyond](http://arxiv.org/abs/2007.03652)


  Efficient sampling and remote estimation are critical for a plethora of
wireless-empowered applications in the Internet of Things and cyber-physical
systems. Motivated by such applications, this work proposes decentralized
policies for the real-time monitoring and estimation of autoregressive
processes over random access channels. Two classes of policies are
investigated: (i) oblivious schemes in which sampling and transmission policies
are independent of the processes that are monitored, and (ii) non-oblivious
schemes in which transmitters causally observe their corresponding processes
for decision making. In the class of oblivious policies, we show that
minimizing the expected time-average estimation error is equivalent to
minimizing the expected age of information. Consequently, we prove lower and
upper bounds on the minimum achievable estimation error in this class. Next, we
consider non-oblivious policies and design a threshold policy, called
error-based thinning, in which each source node becomes active if its
instantaneous error has crossed a fixed threshold (which we optimize). Active
nodes then transmit stochastically following a slotted ALOHA policy. A
closed-form, approximately optimal, solution is found for the threshold as well
as the resulting estimation error. It is shown that non-oblivious policies
offer a multiplicative gain close to $3$ compared to oblivious policies.
Moreover, it is shown that oblivious policies that use the age of information
for decision making improve the state-of-the-art at least by the multiplicative
factor $2$. The performance of all discussed policies is compared using
simulations. The numerical comparison shows that the performance of the
proposed decentralized policy is very close to that of centralized greedy
scheduling.

    

### [[2106.06291] DRLD-SP: A Deep Reinforcement Learning-based Dynamic Service Placement in Edge-Enabled Internet of Vehicles](http://arxiv.org/abs/2106.06291)


  The growth of 5G and edge computing has enabled the emergence of Internet of
Vehicles. It supports different types of services with different resource and
service requirements. However, limited resources at the edge, high mobility of
vehicles, increasing demand, and dynamicity in service request-types have made
service placement a challenging task. A typical static placement solution is
not effective as it does not consider the traffic mobility and service
dynamics. Handling dynamics in IoV for service placement is an important and
challenging problem which is the primary focus of our work in this paper. We
propose a Deep Reinforcement Learning-based Dynamic Service Placement (DRLD-SP)
framework with the objective of minimizing the maximum edge resource usage and
service delay while considering the vehicle's mobility, varying demand, and
dynamics in the requests for different types of services. We use SUMO and
MATLAB to carry out simulation experiments. The experimental results show that
the proposed DRLD-SP approach is effective and outperforms other static and
dynamic placement approaches.

    

### [[2109.03264] Text-Free Prosody-Aware Generative Spoken Language Modeling](http://arxiv.org/abs/2109.03264)


  Speech pre-training has primarily demonstrated efficacy on classification
tasks, while its capability of generating novel speech, similar to how GPT-2
can generate coherent paragraphs, has barely been explored. Generative Spoken
Language Modeling (GSLM) (Lakhotia et al., 2021) is the only prior work
addressing the generative aspects of speech pre-training, which replaces text
with discovered phone-like units for language modeling and shows the ability to
generate meaningful novel sentences. Unfortunately, despite eliminating the
need of text, the units used in GSLM discard most of the prosodic information.
Hence, GSLM fails to leverage prosody for better comprehension, and does not
generate expressive speech. In this work, we present a prosody-aware generative
spoken language model (pGSLM). It is composed of a multi-stream transformer
language model (MS-TLM) of speech, represented as discovered unit and prosodic
feature streams, and an adapted HiFi-GAN model converting MS-TLM outputs to
waveforms. We devise a series of metrics for prosody modeling and generation,
and re-use metrics from GSLM for content modeling. Experimental results show
that the pGSLM can utilize prosody to improve both prosody and content
modeling, and also generate natural, meaningful, and coherent speech given a
spoken prompt. Audio samples can be found at this https URL.

    

### [[2109.03275] A New Non-Negative Matrix Co-Factorisation Approach for Noisy Neonatal Chest Sound Separation](http://arxiv.org/abs/2109.03275)


  Obtaining high-quality heart and lung sounds enables clinicians to accurately
assess a newborn's cardio-respiratory health and provide timely care. However,
noisy chest sound recordings are common, hindering timely and accurate
assessment. A new Non-negative Matrix Co-Factorisation-based approach is
proposed to separate noisy chest sound recordings into heart, lung, and noise
components to address this problem. This method is achieved through training
with 20 high-quality heart and lung sounds, in parallel with separating the
sounds of the noisy recording. The method was tested on 68 10-second noisy
recordings containing both heart and lung sounds and compared to the current
state of the art Non-negative Matrix Factorisation methods. Results show
significant improvements in heart and lung sound quality scores respectively,
and improved accuracy of 3.6bpm and 1.2bpm in heart and breathing rate
estimation respectively, when compared to existing methods.

    

### [[2109.03285] Amazon SageMaker Clarify: Machine Learning Bias Detection and Explainability in the Cloud](http://arxiv.org/abs/2109.03285)


  Understanding the predictions made by machine learning (ML) models and their
potential biases remains a challenging and labor-intensive task that depends on
the application, the dataset, and the specific model. We present Amazon
SageMaker Clarify, an explainability feature for Amazon SageMaker that launched
in December 2020, providing insights into data and ML models by identifying
biases and explaining predictions. It is deeply integrated into Amazon
SageMaker, a fully managed service that enables data scientists and developers
to build, train, and deploy ML models at any scale. Clarify supports bias
detection and feature importance computation across the ML lifecycle, during
data preparation, model evaluation, and post-deployment monitoring. We outline
the desiderata derived from customer input, the modular architecture, and the
methodology for bias and explanation computations. Further, we describe the
technical challenges encountered and the tradeoffs we had to make. For
illustration, we discuss two customer use cases. We present our deployment
results including qualitative customer feedback and a quantitative evaluation.
Finally, we summarize lessons learned, and discuss best practices for the
successful adoption of fairness and explanation tools in practice.

    

### [[2109.03292] Simple Video Generation using Neural ODEs](http://arxiv.org/abs/2109.03292)


  Despite having been studied to a great extent, the task of conditional
generation of sequences of frames, or videos, remains extremely challenging. It
is a common belief that a key step towards solving this task resides in
modelling accurately both spatial and temporal information in video signals. A
promising direction to do so has been to learn latent variable models that
predict the future in latent space and project back to pixels, as suggested in
recent literature. Following this line of work and building on top of a family
of models introduced in prior work, Neural ODE, we investigate an approach that
models time-continuous dynamics over a continuous latent space with a
differential equation with respect to time. The intuition behind this approach
is that these trajectories in latent space could then be extrapolated to
generate video frames beyond the time steps for which the model is trained. We
show that our approach yields promising results in the task of future frame
prediction on the Moving MNIST dataset with 1 and 2 digits.

    

### [[2109.03309] CRNNTL: convolutional recurrent neural network and transfer learning for QSAR modelling](http://arxiv.org/abs/2109.03309)


  In this study, we propose the convolutional recurrent neural network and
transfer learning (CRNNTL) for QSAR modelling. The method was inspired by the
applications of polyphonic sound detection and electrocardiogram
classification. Our strategy takes advantages of both convolutional and
recurrent neural networks for feature extraction, as well as the data
augmentation method. Herein, CRNNTL is evaluated on 20 benchmark datasets in
comparison with baseline methods. In addition, one isomers based dataset is
used to elucidate its ability for both local and global feature extraction.
Then, knowledge transfer performance of CRNNTL is tested, especially for small
biological activity datasets. Finally, different latent representations from
other type of AEs were used for versatility study of our model. The results
show the effectiveness of CRNNTL using different latent representation.
Moreover, efficient knowledge transfer is achieved to overcome data scarcity
considering binding site similarity between different targets.

    

### [[2109.03323] Effective and interpretable dispatching rules for dynamic job shops via guided empirical learning](http://arxiv.org/abs/2109.03323)


  The emergence of Industry 4.0 is making production systems more flexible and
also more dynamic. In these settings, schedules often need to be adapted in
real-time by dispatching rules. Although substantial progress was made until
the '90s, the performance of these rules is still rather limited. The machine
learning literature is developing a variety of methods to improve them, but the
resulting rules are difficult to interpret and do not generalise well for a
wide range of settings. This paper is the first major attempt at combining
machine learning with domain problem reasoning for scheduling. The idea
consists of using the insights obtained with the latter to guide the empirical
search of the former. Our hypothesis is that this guided empirical learning
process should result in dispatching rules that are effective and interpretable
and which generalise well to different instance classes. We test our approach
in the classical dynamic job shop scheduling problem minimising tardiness,
which is one of the most well-studied scheduling problems. Nonetheless, results
suggest that our approach was able to find new state-of-the-art rules, which
significantly outperform the existing literature in the vast majority of
settings, from loose to tight due dates and from low utilisation conditions to
congested shops. Overall, the average improvement is 19%. Moreover, the rules
are compact, interpretable, and generalise well to extreme, unseen scenarios.

    

### [[2109.03326] DexRay: A Simple, yet Effective Deep Learning Approach to Android Malware Detection based on Image Representation of Bytecode](http://arxiv.org/abs/2109.03326)


  Computer vision has witnessed several advances in recent years, with
unprecedented performance provided by deep representation learning research.
Image formats thus appear attractive to other fields such as malware detection,
where deep learning on images alleviates the need for comprehensively
hand-crafted features generalising to different malware variants. We postulate
that this research direction could become the next frontier in Android malware
detection, and therefore requires a clear roadmap to ensure that new approaches
indeed bring novel contributions. We contribute with a first building block by
developing and assessing a baseline pipeline for image-based malware detection
with straightforward steps. We propose DexRay, which converts the bytecode of
the app DEX files into grey-scale "vector" images and feeds them to a
1-dimensional Convolutional Neural Network model. We view DexRay as
foundational due to the exceedingly basic nature of the design choices,
allowing to infer what could be a minimal performance that can be obtained with
image-based learning in malware detection. The performance of DexRay evaluated
on over 158k apps demonstrates that, while simple, our approach is effective
with a high detection rate(F1-score= 0.96). Finally, we investigate the impact
of time decay and image-resizing on the performance of DexRay and assess its
resilience to obfuscation. This work-in-progress paper contributes to the
domain of Deep Learning based Malware detection by providing a sound, simple,
yet effective approach (with available artefacts) that can be the basis to
scope the many profound questions that will need to be investigated to fully
develop this domain.

    

### [[2109.03327] Reconstructing High-resolution Turbulent Flows Using Physics-Guided Neural Networks](http://arxiv.org/abs/2109.03327)


  Direct numerical simulation (DNS) of turbulent flows is computationally
expensive and cannot be applied to flows with large Reynolds numbers. Large
eddy simulation (LES) is an alternative that is computationally less demanding,
but is unable to capture all of the scales of turbulent transport accurately.
Our goal in this work is to build a new data-driven methodology based on
super-resolution techniques to reconstruct DNS data from LES predictions. We
leverage the underlying physical relationships to regularize the relationships
amongst different physical variables. We also introduce a hierarchical
generative process and a reverse degradation process to fully explore the
correspondence between DNS and LES data. We demonstrate the effectiveness of
our method through a single-snapshot experiment and a cross-time experiment.
The results confirm that our method can better reconstruct high-resolution DNS
data over space and over time in terms of pixel-wise reconstruction error and
structural similarity. Visual comparisons show that our method performs much
better in capturing fine-level flow dynamics.

    

### [[2109.03329] Real-World Adversarial Examples involving Makeup Application](http://arxiv.org/abs/2109.03329)


  Deep neural networks have developed rapidly and have achieved outstanding
performance in several tasks, such as image classification and natural language
processing. However, recent studies have indicated that both digital and
physical adversarial examples can fool neural networks. Face-recognition
systems are used in various applications that involve security threats from
physical adversarial examples. Herein, we propose a physical adversarial attack
with the use of full-face makeup. The presence of makeup on the human face is a
reasonable possibility, which possibly increases the imperceptibility of
attacks. In our attack framework, we combine the cycle-adversarial generative
network (cycle-GAN) and a victimized classifier. The Cycle-GAN is used to
generate adversarial makeup, and the architecture of the victimized classifier
is VGG 16. Our experimental results show that our attack can effectively
overcome manual errors in makeup application, such as color and
position-related errors. We also demonstrate that the approaches used to train
the models can influence physical attacks; the adversarial perturbations
crafted from the pre-trained model are affected by the corresponding training
data.

    

### [[2109.03331] CyGIL: A Cyber Gym for Training Autonomous Agents over Emulated Network Systems](http://arxiv.org/abs/2109.03331)


  Given the success of reinforcement learning (RL) in various domains, it is
promising to explore the application of its methods to the development of
intelligent and autonomous cyber agents. Enabling this development requires a
representative RL training environment. To that end, this work presents CyGIL:
an experimental testbed of an emulated RL training environment for network
cyber operations. CyGIL uses a stateless environment architecture and
incorporates the MITRE ATT&CK framework to establish a high fidelity training
environment, while presenting a sufficiently abstracted interface to enable RL
training. Its comprehensive action space and flexible game design allow the
agent training to focus on particular advanced persistent threat (APT)
profiles, and to incorporate a broad range of potential threats and
vulnerabilities. By striking a balance between fidelity and simplicity, it aims
to leverage state of the art RL algorithms for application to real-world cyber
defence.

    

### [[2109.03337] C-MinHash: Rigorously Reducing $K$ Permutations to Two](http://arxiv.org/abs/2109.03337)


  Minwise hashing (MinHash) is an important and practical algorithm for
generating random hashes to approximate the Jaccard (resemblance) similarity in
massive binary (0/1) data. The basic theory of MinHash requires applying
hundreds or even thousands of independent random permutations to each data
vector in the dataset, in order to obtain reliable results for (e.g.,) building
large-scale learning models or approximate near neighbor search in massive
data. In this paper, we propose {\bf Circulant MinHash (C-MinHash)} and provide
the surprising theoretical results that we just need \textbf{two} independent
random permutations. For C-MinHash, we first conduct an initial permutation on
the data vector, then we use a second permutation to generate hash values.
Basically, the second permutation is re-used $K$ times via circulant shifting
to produce $K$ hashes. Unlike classical MinHash, these $K$ hashes are obviously
correlated, but we are able to provide rigorous proofs that we still obtain an
unbiased estimate of the Jaccard similarity and the theoretical variance is
uniformly smaller than that of the classical MinHash with $K$ independent
permutations. The theoretical proofs of C-MinHash require some non-trivial
efforts. Numerical experiments are conducted to justify the theory and
demonstrate the effectiveness of C-MinHash.

    

### [[2109.03350] Federated Learning Beyond the Star: Local D2D Model Consensus with Global Cluster Sampling](http://arxiv.org/abs/2109.03350)


  Federated learning has emerged as a popular technique for distributing model
training across the network edge. Its learning architecture is conventionally a
star topology between the devices and a central server. In this paper, we
propose two timescale hybrid federated learning (TT-HF), which migrates to a
more distributed topology via device-to-device (D2D) communications. In TT-HF,
local model training occurs at devices via successive gradient iterations, and
the synchronization process occurs at two timescales: (i) macro-scale, where
global aggregations are carried out via device-server interactions, and (ii)
micro-scale, where local aggregations are carried out via D2D cooperative
consensus formation in different device clusters. Our theoretical analysis
reveals how device, cluster, and network-level parameters affect the
convergence of TT-HF, and leads to a set of conditions under which a
convergence rate of O(1/t) is guaranteed. Experimental results demonstrate the
improvements in convergence and utilization that can be obtained by TT-HF over
state-of-the-art federated learning baselines.

    

### [[2109.03362] On the space of coefficients of a Feed Forward Neural Network](http://arxiv.org/abs/2109.03362)


  We define and establish the conditions for `equivalent neural networks' -
neural networks with different weights, biases, and threshold functions that
result in the same associated function. We prove that given a neural network
$\mathcal{N}$ with piece-wise linear activation, the space of coefficients
describing all equivalent neural networks is given by a semialgebraic set. This
result is obtained by studying different representations of a given piece-wise
linear function using the Tarski-Seidenberg theorem.

    

### [[2109.03366] Forward and Inverse models in HCI:Physical simulation and deep learning for inferring 3D finger pose](http://arxiv.org/abs/2109.03366)


  We outline the role of forward and inverse modelling approaches in the design
of human--computer interaction systems. Causal, forward models tend to be
easier to specify and simulate, but HCI requires solutions of the inverse
problem. We infer finger 3D position $(x,y,z)$ and pose (pitch and yaw) on a
mobile device using capacitive sensors which can sense the finger up to 5cm
above the screen. We use machine learning to develop data-driven models to
infer position, pose and sensor readings, based on training data from: 1. data
generated by robots, 2. data from electrostatic simulators 3. human-generated
data. Machine learned emulation is used to accelerate the electrostatic
simulation performance by a factor of millions. We combine a Conditional
Variational Autoencoder with domain expertise/models experimentally collected
data. We compare forward and inverse model approaches to direct inference of
finger pose. The combination gives the most accurate reported results on
inferring 3D position and pose with a capacitive sensor on a mobile device.

    

### [[2109.03378] AWGAN: Empowering High-Dimensional Discriminator Output for Generative Adversarial Networks](http://arxiv.org/abs/2109.03378)


  Empirically multidimensional discriminator (critic) output can be
advantageous, while a solid explanation for it has not been discussed. In this
paper, (i) we rigorously prove that high-dimensional critic output has
advantage on distinguishing real and fake distributions; (ii) we also introduce
an square-root velocity transformation (SRVT) block which further magnifies
this advantage. The proof is based on our proposed maximal p-centrality
discrepancy which is bounded above by p-Wasserstein distance and perfectly fits
the Wasserstein GAN framework with high-dimensional critic output n. We have
also showed when n = 1, the proposed discrepancy is equivalent to 1-Wasserstein
distance. The SRVT block is applied to break the symmetric structure of
high-dimensional critic output and improve the generalization capability of the
discriminator network. In terms of implementation, the proposed framework does
not require additional hyper-parameter tuning, which largely facilitates its
usage. Experiments on image generation tasks show performance improvement on
benchmark datasets.

    

### [[2109.03381] Self-supervised Contrastive Cross-Modality Representation Learning for Spoken Question Answering](http://arxiv.org/abs/2109.03381)


  Spoken question answering (SQA) requires fine-grained understanding of both
spoken documents and questions for the optimal answer prediction. In this
paper, we propose novel training schemes for spoken question answering with a
self-supervised training stage and a contrastive representation learning stage.
In the self-supervised stage, we propose three auxiliary self-supervised tasks,
including utterance restoration, utterance insertion, and question
discrimination, and jointly train the model to capture consistency and
coherence among speech documents without any additional data or annotations. We
then propose to learn noise-invariant utterance representations in a
contrastive objective by adopting multiple augmentation strategies, including
span deletion and span substitution. Besides, we design a Temporal-Alignment
attention to semantically align the speech-text clues in the learned common
space and benefit the SQA tasks. By this means, the training schemes can more
effectively guide the generation model to predict more proper answers.
Experimental results show that our model achieves state-of-the-art results on
three SQA benchmarks.

    

### [[2109.03385] RoadAtlas: Intelligent Platform for Automated Road Defect Detection and Asset Management](http://arxiv.org/abs/2109.03385)


  With the rapid development of intelligent detection algorithms based on deep
learning, much progress has been made in automatic road defect recognition and
road marking parsing. This can effectively address the issue of an expensive
and time-consuming process for professional inspectors to review the street
manually. Towards this goal, we present RoadAtlas, a novel end-to-end
integrated system that can support 1) road defect detection, 2) road marking
parsing, 3) a web-based dashboard for presenting and inputting data by users,
and 4) a backend containing a well-structured database and developed APIs.

    

### [[2109.03386] On the Fundamental Trade-offs in Learning Invariant Representations](http://arxiv.org/abs/2109.03386)


  Many applications of representation learning, such as privacy-preservation,
algorithmic fairness and domain adaptation, desire explicit control over
semantic information being discarded. This goal is often formulated as
satisfying two potentially competing objectives: maximizing utility for
predicting a target attribute while simultaneously being independent or
invariant with respect to a known semantic attribute. In this paper, we
\emph{identify and determine} two fundamental trade-offs between utility and
semantic dependence induced by the statistical dependencies between the data
and its corresponding target and semantic attributes. We derive closed-form
solutions for the global optima of the underlying optimization problems under
mild assumptions, which in turn yields closed formulae for the exact
trade-offs. We also derive empirical estimates of the trade-offs and show their
convergence to the corresponding population counterparts. Finally, we
numerically quantify the trade-offs on representative problems and compare to
the solutions achieved by baseline representation learning algorithms.

    

### [[2109.03396] Learning Zero-sum Stochastic Games with Posterior Sampling](http://arxiv.org/abs/2109.03396)


  In this paper, we propose Posterior Sampling Reinforcement Learning for
Zero-sum Stochastic Games (PSRL-ZSG), the first online learning algorithm that
achieves Bayesian regret bound of $O(HS\sqrt{AT})$ in the infinite-horizon
zero-sum stochastic games with average-reward criterion. Here $H$ is an upper
bound on the span of the bias function, $S$ is the number of states, $A$ is the
number of joint actions and $T$ is the horizon. We consider the online setting
where the opponent can not be controlled and can take any arbitrary
time-adaptive history-dependent strategy. This improves the best existing
regret bound of $O(\sqrt[3]{DS^2AT^2})$ by Wei et. al., 2017 under the same
assumption and matches the theoretical lower bound in $A$ and $T$.

    

### [[2109.03400] Entangled Datasets for Quantum Machine Learning](http://arxiv.org/abs/2109.03400)


  High-quality, large-scale datasets have played a crucial role in the
development and success of classical machine learning. Quantum Machine Learning
(QML) is a new field that aims to use quantum computers for data analysis, with
the hope of obtaining a quantum advantage of some sort. While most proposed QML
architectures are benchmarked using classical datasets, there is still doubt
whether QML on classical datasets will achieve such an advantage. In this work,
we argue that one should instead employ quantum datasets composed of quantum
states. For this purpose, we introduce the NTangled dataset composed of quantum
states with different amounts and types of multipartite entanglement. We first
show how a quantum neural network can be trained to generate the states in the
NTangled dataset. Then, we use the NTangled dataset to benchmark QML models for
supervised learning classification tasks. We also consider an alternative
entanglement-based dataset, which is scalable and is composed of states
prepared by quantum circuits with different depths. As a byproduct of our
results, we introduce a novel method for generating multipartite entangled
states, providing a use-case of quantum neural networks for quantum
entanglement theory.

    

### [[2109.03429] Computing on Functions Using Randomized Vector Representations](http://arxiv.org/abs/2109.03429)


  Vector space models for symbolic processing that encode symbols by random
vectors have been proposed in cognitive science and connectionist communities
under the names Vector Symbolic Architecture (VSA), and, synonymously,
Hyperdimensional (HD) computing. In this paper, we generalize VSAs to function
spaces by mapping continuous-valued data into a vector space such that the
inner product between the representations of any two data points represents a
similarity kernel. By analogy to VSA, we call this new function encoding and
computing framework Vector Function Architecture (VFA). In VFAs, vectors can
represent individual data points as well as elements of a function space (a
reproducing kernel Hilbert space). The algebraic vector operations, inherited
from VSA, correspond to well-defined operations in function space. Furthermore,
we study a previously proposed method for encoding continuous data, fractional
power encoding (FPE), which uses exponentiation of a random base vector to
produce randomized representations of data points and fulfills the kernel
properties for inducing a VFA. We show that the distribution from which
elements of the base vector are sampled determines the shape of the FPE kernel,
which in turn induces a VFA for computing with band-limited functions. In
particular, VFAs provide an algebraic framework for implementing large-scale
kernel machines with random features, extending Rahimi and Recht, 2007.
Finally, we demonstrate several applications of VFA models to problems in image
recognition, density estimation and nonlinear regression. Our analyses and
results suggest that VFAs constitute a powerful new framework for representing
and manipulating functions in distributed neural systems, with myriad
applications in artificial intelligence.

    

### [[2109.03430] Can Noise on Qubits Be Learned in Quantum Neural Network? A Case Study on QuantumFlow](http://arxiv.org/abs/2109.03430)


  In the noisy intermediate-scale quantum (NISQ) era, one of the key questions
is how to deal with the high noise level existing in physical quantum bits
(qubits). Quantum error correction is promising but requires an extensive
number (e.g., over 1,000) of physical qubits to create one "perfect" qubit,
exceeding the capacity of the existing quantum computers. This paper aims to
tackle the noise issue from another angle: instead of creating perfect qubits
for general quantum algorithms, we investigate the potential to mitigate the
noise issue for dedicate algorithms. Specifically, this paper targets quantum
neural network (QNN), and proposes to learn the errors in the training phase,
so that the identified QNN model can be resilient to noise. As a result, the
implementation of QNN needs no or a small number of additional physical qubits,
which is more realistic for the near-term quantum computers. To achieve this
goal, an application-specific compiler is essential: on the one hand, the error
cannot be learned if the mapping from logical qubits to physical qubits exists
randomness; on the other hand, the compiler needs to be efficient so that the
lengthy training procedure can be completed in a reasonable time. In this
paper, we utilize the recent QNN framework, QuantumFlow, as a case study.
Experimental results show that the proposed approach can optimize QNN models
for different errors in qubits, achieving up to 28% accuracy improvement
compared with the model obtained by the error-agnostic training.

    

### [[2109.03431] Fixed Support Tree-Sliced Wasserstein Barycenter](http://arxiv.org/abs/2109.03431)


  The Wasserstein barycenter has been widely studied in various fields,
including natural language processing, and computer vision. However, it
requires a high computational cost to solve the Wasserstein barycenter problem
because the computation of the Wasserstein distance requires a quadratic time
with respect to the number of supports. By contrast, the Wasserstein distance
on a tree, called the tree-Wasserstein distance, can be computed in linear time
and allows for the fast comparison of a large number of distributions. In this
study, we propose a barycenter under the tree-Wasserstein distance, called the
fixed support tree-Wasserstein barycenter (FS-TWB) and its extension, called
the fixed support tree-sliced Wasserstein barycenter (FS-TSWB). More
specifically, we first show that the FS-TWB and FS-TSWB problems are convex
optimization problems and can be solved by using the projected subgradient
descent. Moreover, we propose a more efficient algorithm to compute the
subgradient and objective function value by using the properties of
tree-Wasserstein barycenter problems. Through real-world experiments, we show
that, by using the proposed algorithm, the FS-TWB and FS-TSWB can be solved two
orders of magnitude faster than the original Wasserstein barycenter.

    

### [[2109.03433] A Clustering-aided Ensemble Method for Predicting Ridesourcing Demand in Chicago](http://arxiv.org/abs/2109.03433)


  Accurately forecasting ridesourcing demand is important for effective
transportation planning and policy-making. With the rise of Artificial
Intelligence (AI), researchers have started to utilize machine learning models
to forecast travel demand, which, in many cases, can produce higher prediction
accuracy than statistical models. However, most existing machine-learning
studies used a global model to predict the demand and ignored the influence of
spatial heterogeneity (i.e., the spatial variations in the impacts of
explanatory variables). Spatial heterogeneity can drive the parameter
estimations varying over space; failing to consider the spatial variations may
limit the model's prediction performance. To account for spatial heterogeneity,
this study proposes a Clustering-aided Ensemble Method (CEM) to forecast the
zone-to-zone (census-tract-to-census-tract) travel demand for ridesourcing
services. Specifically, we develop a clustering framework to split the
origin-destination pairs into different clusters and ensemble the
cluster-specific machine learning models for prediction. We implement and test
the proposed methodology by using the ridesourcing-trip data in Chicago. The
results show that, with a more transparent and flexible model structure, the
CEM significantly improves the prediction accuracy than the benchmark models
(i.e., global machine-learning and statistical models directly trained on all
observations). This study offers transportation researchers and practitioners a
new methodology of travel demand forecasting, especially for new travel modes
like ridesourcing and micromobility.

    

### [[2109.03443] ADER:Adapting between Exploration and Robustness for Actor-Critic Methods](http://arxiv.org/abs/2109.03443)


  Combining off-policy reinforcement learning methods with function
approximators such as neural networks has been found to lead to overestimation
of the value function and sub-optimal solutions. Improvement such as TD3 has
been proposed to address this issue. However, we surprisingly find that its
performance lags behind the vanilla actor-critic methods (such as DDPG) in some
primitive environments. In this paper, we show that the failure of some cases
can be attributed to insufficient exploration. We reveal the culprit of
insufficient exploration in TD3, and propose a novel algorithm toward this
problem that ADapts between Exploration and Robustness, namely ADER. To enhance
the exploration ability while eliminating the overestimation bias, we introduce
a dynamic penalty term in value estimation calculated from estimated
uncertainty, which takes into account different compositions of the uncertainty
in different learning stages. Experiments in several challenging environments
demonstrate the supremacy of the proposed method in continuous control tasks.

    

### [[2109.03445] Convergence of Batch Asynchronous Stochastic Approximation With Applications to Reinforcement Learning](http://arxiv.org/abs/2109.03445)


  The stochastic approximation (SA) algorithm is a widely used probabilistic
method for finding a solution to an equation of the form
$\mathbf{f}(\boldsymbol{\theta}) = \mathbf{0}$ where $\mathbf{f} : \mathbb{R}^d
\rightarrow \mathbb{R}^d$, when only noisy measurements of $\mathbf{f}(\cdot)$
are available. In the literature to date, one can make a distinction between
"synchronous" updating, whereby the entire vector of the current guess
$\boldsymbol{\theta}_t$ is updated at each time, and "asynchronous" updating,
whereby ony one component of $\boldsymbol{\theta}_t$ is updated. In convex and
nonconvex optimization, there is also the notion of "batch" updating, whereby
some but not all components of $\boldsymbol{\theta}_t$ are updated at each time
$t$. In addition, there is also a distinction between using a "local" clock
versus a "global" clock. In the literature to date, convergence proofs when a
local clock is used make the assumption that the measurement noise is an i.i.d\
sequence, an assumption that does not hold in Reinforcement Learning (RL).
In this note, we provide a general theory of convergence for batch
asymchronous stochastic approximation (BASA), that works whether the updates
use a local clock or a global clock, for the case where the measurement noises
form a martingale difference sequence. This is the most general result to date
and encompasses all others.

    

### [[2109.03454] Signal-domain representation of symbolic music for learning embedding spaces](http://arxiv.org/abs/2109.03454)


  A key aspect of machine learning models lies in their ability to learn
efficient intermediate features. However, the input representation plays a
crucial role in this process, and polyphonic musical scores remain a
particularly complex type of information. In this paper, we introduce a novel
representation of symbolic music data, which transforms a polyphonic score into
a continuous signal. We evaluate the ability to learn meaningful features from
this representation from a musical point of view. Hence, we introduce an
evaluation method relying on principled generation of synthetic data. Finally,
to test our proposed representation we conduct an extensive benchmark against
recent polyphonic symbolic representations. We show that our signal-like
representation leads to better reconstruction and disentangled features. This
improvement is reflected in the metric properties and in the generation ability
of the space learned from our signal-like representation according to music
theory properties.

    

### [[2109.03457] Uncertainty Quantification and Experimental Design for large-scale linear Inverse Problems under Gaussian Process Priors](http://arxiv.org/abs/2109.03457)


  We consider the use of Gaussian process (GP) priors for solving inverse
problems in a Bayesian framework. As is well known, the computational
complexity of GPs scales cubically in the number of datapoints. We here show
that in the context of inverse problems involving integral operators, one faces
additional difficulties that hinder inversion on large grids. Furthermore, in
that context, covariance matrices can become too large to be stored. By
leveraging results about sequential disintegrations of Gaussian measures, we
are able to introduce an implicit representation of posterior covariance
matrices that reduces the memory footprint by only storing low rank
intermediate matrices, while allowing individual elements to be accessed
on-the-fly without needing to build full posterior covariance matrices.
Moreover, it allows for fast sequential inclusion of new observations. These
features are crucial when considering sequential experimental design tasks. We
demonstrate our approach by computing sequential data collection plans for
excursion set recovery for a gravimetric inverse problem, where the goal is to
provide fine resolution estimates of high density regions inside the Stromboli
volcano, Italy. Sequential data collection plans are computed by extending the
weighted integrated variance reduction (wIVR) criterion to inverse problems.
Our results show that this criterion is able to significantly reduce the
uncertainty on the excursion volume, reaching close to minimal levels of
residual uncertainty. Overall, our techniques allow the advantages of
probabilistic models to be brought to bear on large-scale inverse problems
arising in the natural sciences.

    

### [[2109.03459] Dual Correction Strategy for Ranking Distillation in Top-N Recommender System](http://arxiv.org/abs/2109.03459)


  Knowledge Distillation (KD), which transfers the knowledge of a well-trained
large model (teacher) to a small model (student), has become an important area
of research for practical deployment of recommender systems. Recently, Relaxed
Ranking Distillation (RRD) has shown that distilling the ranking information in
the recommendation list significantly improves the performance. However, the
method still has limitations in that 1) it does not fully utilize the
prediction errors of the student model, which makes the training not fully
efficient, and 2) it only distills the user-side ranking information, which
provides an insufficient view under the sparse implicit feedback. This paper
presents Dual Correction strategy for Distillation (DCD), which transfers the
ranking information from the teacher model to the student model in a more
efficient manner. Most importantly, DCD uses the discrepancy between the
teacher model and the student model predictions to decide which knowledge to be
distilled. By doing so, DCD essentially provides the learning guidance tailored
to "correcting" what the student model has failed to accurately predict. This
process is applied for transferring the ranking information from the user-side
as well as the item-side to address sparse implicit user feedback. Our
experiments show that the proposed method outperforms the state-of-the-art
baselines, and ablation studies validate the effectiveness of each component.

    

### [[2109.03465] A Review of Sound Source Localization with Deep Learning Methods](http://arxiv.org/abs/2109.03465)


  This article is a review on deep learning methods for single and multiple
sound source localization. We are particularly interested in sound source
localization in indoor/domestic environment, where reverberation and diffuse
noise are present. We provide an exhaustive topography of the neural-based
localization literature in this context, organized according to several
aspects: the neural network architecture, the type of input features, the
output strategy (classification or regression), the types of data used for
model training and evaluation, and the model training strategy. This way, an
interested reader can easily comprehend the vast panorama of the deep
learning-based sound source localization methods. Tables summarizing the
literature review are provided at the end of the review for a quick search of
methods with a given set of target characteristics.

    

### [[2109.03467] A Deep Reinforcement Learning Approach for Constrained Online Logistics Route Assignment](http://arxiv.org/abs/2109.03467)


  As online shopping prevails and e-commerce platforms emerge, there is a
tremendous number of parcels being transported every day. Thus, it is crucial
for the logistics industry on how to assign a candidate logistics route for
each shipping parcel properly as it leaves a significant impact on the total
logistics cost optimization and business constraints satisfaction such as
transit hub capacity and delivery proportion of delivery providers. This online
route-assignment problem can be viewed as a constrained online decision-making
problem. Notably, the large amount (beyond ${10^5}$) of daily parcels, the
variability and non-Markovian characteristics of parcel information impose
difficulties on attaining (near-) optimal solution without violating
constraints excessively. In this paper, we develop a model-free DRL approach
named PPO-RA, in which Proximal Policy Optimization (PPO) is improved with
dedicated techniques to address the challenges for route assignment (RA). The
actor and critic networks use attention mechanism and parameter sharing to
accommodate each incoming parcel with varying numbers and identities of
candidate routes, without modeling non-Markovian parcel arriving dynamics since
we make assumption of i.i.d. parcel arrival. We use recorded delivery parcel
data to evaluate the performance of PPO-RA by comparing it with widely-used
baselines via simulation. The results show the capability of the proposed
approach to achieve considerable cost savings while satisfying most
constraints.

    

### [[2109.03468] Preprocessing and Modeling of Radial Fan Data for Health State Prediction](http://arxiv.org/abs/2109.03468)


  Monitoring critical components of systems is a crucial step towards failure
safety. Affordable sensors are available and the industry is in the process of
introducing and extending monitoring solutions to improve product quality.
Often, no expertise of how much data is required for a certain task (e.g.
monitoring) exists. Especially in vital machinery, a trend to exaggerated
sensors may be noticed, both in quality and in quantity. This often results in
an excessive generation of data, which should be transferred, processed and
stored nonetheless. In a previous case study, several sensors have been mounted
on a healthy radial fan, which was later artificially damaged. The gathered
data was used for modeling (and therefore monitoring) a healthy state. The
models were evaluated on a dataset created by using a faulty impeller. This
paper focuses on the reduction of this data through downsampling and binning.
Different models are created with linear regression and random forest
regression and the resulting difference in quality is discussed.

    

### [[2109.03469] Understanding and Preparing Data of Industrial Processes for Machine Learning Applications](http://arxiv.org/abs/2109.03469)


  Industrial applications of machine learning face unique challenges due to the
nature of raw industry data. Preprocessing and preparing raw industrial data
for machine learning applications is a demanding task that often takes more
time and work than the actual modeling process itself and poses additional
challenges. This paper addresses one of those challenges, specifically, the
challenge of missing values due to sensor unavailability at different
production units of nonlinear production lines. In cases where only a small
proportion of the data is missing, those missing values can often be imputed.
In cases of large proportions of missing data, imputing is often not feasible,
and removing observations containing missing values is often the only option.
This paper presents a technique, that allows to utilize all of the available
data without the need of removing large amounts of observations where data is
only partially available. We do not only discuss the principal idea of the
presented method, but also show different possible implementations that can be
applied depending on the data at hand. Finally, we demonstrate the application
of the presented method with data from a steel production plant.

    

### [[2109.03475] A Bottom-up method Towards the Automatic and Objective Monitoring of Smoking Behavior In-the-wild using Wrist-mounted Inertial Sensors](http://arxiv.org/abs/2109.03475)


  The consumption of tobacco has reached global epidemic proportions and is
characterized as the leading cause of death and illness. Among the different
ways of consuming tobacco (e.g., smokeless, cigars), smoking cigarettes is the
most widespread. In this paper, we present a two-step, bottom-up algorithm
towards the automatic and objective monitoring of cigarette-based, smoking
behavior during the day, using the 3D acceleration and orientation velocity
measurements from a commercial smartwatch. In the first step, our algorithm
performs the detection of individual smoking gestures (i.e., puffs) using an
artificial neural network with both convolutional and recurrent layers. In the
second step, we make use of the detected puff density to achieve the temporal
localization of smoking sessions that occur throughout the day. In the
experimental section we provide extended evaluation regarding each step of the
proposed algorithm, using our publicly available, realistic Smoking Event
Detection (SED) and Free-living Smoking Event Detection (SED-FL) datasets
recorded under semi-controlled and free-living conditions, respectively. In
particular, leave-one-subject-out (LOSO) experiments reveal an F1-score of
0.863 for the detection of puffs and an F1-score/Jaccard index equal to
0.878/0.604 towards the temporal localization of smoking sessions during the
day. Finally, to gain further insight, we also compare the puff detection part
of our algorithm with a similar approach found in the recent literature.

    

### [[2109.03480] Estimating Expected Calibration Errors](http://arxiv.org/abs/2109.03480)


  Uncertainty in probabilistic classifiers predictions is a key concern when
models are used to support human decision making, in broader probabilistic
pipelines or when sensitive automatic decisions have to be taken. Studies have
shown that most models are not intrinsically well calibrated, meaning that
their decision scores are not consistent with posterior probabilities. Hence
being able to calibrate these models, or enforce calibration while learning
them, has regained interest in recent literature. In this context, properly
assessing calibration is paramount to quantify new contributions tackling
calibration. However, there is room for improvement for commonly used metrics
and evaluation of calibration could benefit from deeper analyses. Thus this
paper focuses on the empirical evaluation of calibration metrics in the context
of classification. More specifically it evaluates different estimators of the
Expected Calibration Error ($ECE$), amongst which legacy estimators and some
novel ones, proposed in this paper. We build an empirical procedure to quantify
the quality of these $ECE$ estimators, and use it to decide which estimator
should be used in practice for different settings.

    

### [[2109.03484] Shuffled Patch-Wise Supervision for Presentation Attack Detection](http://arxiv.org/abs/2109.03484)


  Face anti-spoofing is essential to prevent false facial verification by using
a photo, video, mask, or a different substitute for an authorized person's
face. Most of the state-of-the-art presentation attack detection (PAD) systems
suffer from overfitting, where they achieve near-perfect scores on a single
dataset but fail on a different dataset with more realistic data. This problem
drives researchers to develop models that perform well under real-world
conditions. This is an especially challenging problem for frame-based
presentation attack detection systems that use convolutional neural networks
(CNN). To this end, we propose a new PAD approach, which combines pixel-wise
binary supervision with patch-based CNN. We believe that training a CNN with
face patches allows the model to distinguish spoofs without learning background
or dataset-specific traces. We tested the proposed method both on the standard
benchmark datasets -- Replay-Mobile, OULU-NPU -- and on a real-world dataset.
The proposed approach shows its superiority on challenging experimental setups.
Namely, it achieves higher performance on OULU-NPU protocol 3, 4 and on
inter-dataset real-world experiments.

    

### [[2109.03501] How do I update my model? On the resilience of Predictive Process Monitoring models to change](http://arxiv.org/abs/2109.03501)


  Existing well investigated Predictive Process Monitoring techniques typically
construct a predictive model based on past process executions, and then use it
to predict the future of new ongoing cases, without the possibility of updating
it with new cases when they complete their execution. This can make Predictive
Process Monitoring too rigid to deal with the variability of processes working
in real environments that continuously evolve and/or exhibit new variant
behaviours over time. As a solution to this problem, we evaluate the use of
three different strategies that allow the periodic rediscovery or incremental
construction of the predictive model so as to exploit new available data. The
evaluation focuses on the performance of the new learned predictive models, in
terms of accuracy and time, against the original one, and uses a number of real
and synthetic datasets with and without explicit Concept Drift. The results
provide an evidence of the potential of incremental learning algorithms for
predicting process monitoring in real environments.

    

### [[2109.03502] R2-D2: A Modular Baseline for Open-Domain Question Answering](http://arxiv.org/abs/2109.03502)


  This work presents a novel four-stage open-domain QA pipeline R2-D2 (Rank
twice, reaD twice). The pipeline is composed of a retriever, passage reranker,
extractive reader, generative reader and a mechanism that aggregates the final
prediction from all system's components. We demonstrate its strength across
three open-domain QA datasets: NaturalQuestions, TriviaQA and EfficientQA,
surpassing state-of-the-art on the first two. Our analysis demonstrates that:
(i) combining extractive and generative reader yields absolute improvements up
to 5 exact match and it is at least twice as effective as the posterior
averaging ensemble of the same models with different parameters, (ii) the
extractive reader with fewer parameters can match the performance of the
generative reader on extractive QA datasets.

    

### [[2109.03508] RepNAS: Searching for Efficient Re-parameterizing Blocks](http://arxiv.org/abs/2109.03508)


  In the past years, significant improvements in the field of neural
architecture search(NAS) have been made. However, it is still challenging to
search for efficient networks due to the gap between the searched constraint
and real inference time exists. To search for a high-performance network with
low inference time, several previous works set a computational complexity
constraint for the search algorithm. However, many factors affect the speed of
inference(e.g., FLOPs, MACs). The correlation between a single indicator and
the latency is not strong. Currently, some re-parameterization(Rep) techniques
are proposed to convert multi-branch to single-path architecture which is
inference-friendly. Nevertheless, multi-branch architectures are still
human-defined and inefficient. In this work, we propose a new search space that
is suitable for structural re-parameterization techniques. RepNAS, a one-stage
NAS approach, is present to efficiently search the optimal diverse branch
block(ODBB) for each layer under the branch number constraint. Our experimental
results show the searched ODBB can easily surpass the manual diverse branch
block(DBB) with efficient training. Code and models will be available sooner.

    

### [[2109.03535] DeepAltTrip: Top-k Alternative Itineraries for Trip Recommendation](http://arxiv.org/abs/2109.03535)


  Trip itinerary recommendation finds an ordered sequence of Points-of-Interest
(POIs) from a large number of candidate POIs in a city. In this paper, we
propose a deep learning-based framework, called DeepAltTrip, that learns to
recommend top-k alternative itineraries for given source and destination POIs.
These alternative itineraries would be not only popular given the historical
routes adopted by past users but also dissimilar (or diverse) to each other.
The DeepAltTrip consists of two major components: (i) Itinerary Net (ITRNet)
which estimates the likelihood of POIs on an itinerary by using graph
autoencoders and two (forward and backward) LSTMs; and (ii) a route generation
procedure to generate k diverse itineraries passing through relevant POIs
obtained using ITRNet. For the route generation step, we propose a novel
sampling algorithm that can seamlessly handle a wide variety of user-defined
constraints. To the best of our knowledge, this is the first work that learns
from historical trips to provide a set of alternative itineraries to the users.
Extensive experiments conducted on eight popular real-world datasets show the
effectiveness and efficacy of our approach over state-of-the-art methods.

    

### [[2109.03552] Cross-lingual Offensive Language Identification for Low Resource Languages: The Case of Marathi](http://arxiv.org/abs/2109.03552)


  The widespread presence of offensive language on social media motivated the
development of systems capable of recognizing such content automatically. Apart
from a few notable exceptions, most research on automatic offensive language
identification has dealt with English. To address this shortcoming, we
introduce MOLD, the Marathi Offensive Language Dataset. MOLD is the first
dataset of its kind compiled for Marathi, thus opening a new domain for
research in low-resource Indo-Aryan languages. We present results from several
machine learning experiments on this dataset, including zero-short and other
transfer learning experiments on state-of-the-art cross-lingual transformers
from existing data in Bengali, English, and Hindi.

    

### [[2109.03560] Graph-MVP: Multi-View Prototypical Contrastive Learning for Multiplex Graphs](http://arxiv.org/abs/2109.03560)


  Contrastive Learning (CL) is one of the most popular self-supervised learning
frameworks for graph representation learning, which trains a Graph Neural
Network (GNN) by discriminating positive and negative node pairs. However,
there are two challenges for CL on graphs. On the one hand, traditional CL
methods will unavoidably introduce semantic errors since they will treat some
semantically similar nodes as negative pairs. On the other hand, most of the
existing CL methods ignore the multiplexity nature of the real-world graphs,
where nodes are connected by various relations and each relation represents a
view of the graph. To address these challenges, we propose a novel Graph
Multi-View Prototypical (Graph-MVP) framework to extract node embeddings on
multiplex graphs. Firstly, we introduce a Graph Prototypical Contrastive
Learning (Graph-PCL) framework to capture both node-level and semantic-level
information for each view of multiplex graphs. Graph-PCL captures the
node-level information by a simple yet effective data transformation technique.
It captures the semantic-level information by an Expectation-Maximization (EM)
algorithm, which alternatively performs clustering over node embeddings and
parameter updating for GNN. Next, we introduce Graph-MVP based on Graph-PCL to
jointly model different views of the multiplex graphs. Our key insight behind
Graph-MVP is that different view-specific embeddings of the same node should
have similar underlying semantic, based on which we propose two versions of
Graph-MVP: Graph-MVP_hard and Graph-MVP_soft to align embeddings across views.
Finally, we evaluate the proposed Graph-PCL and Graph-MVP on a variety of
real-world datasets and downstream tasks. The experimental results demonstrate
the effectiveness of the proposed Graph-PCL and Graph-MVP frameworks.

    

### [[2109.03575] Deriving Explanation of Deep Visual Saliency Models](http://arxiv.org/abs/2109.03575)


  Deep neural networks have shown their profound impact on achieving human
level performance in visual saliency prediction. However, it is still unclear
how they learn the task and what it means in terms of understanding human
visual system. In this work, we develop a technique to derive explainable
saliency models from their corresponding deep neural architecture based
saliency models by applying human perception theories and the conventional
concepts of saliency. This technique helps us understand the learning pattern
of the deep network at its intermediate layers through their activation maps.
Initially, we consider two state-of-the-art deep saliency models, namely UNISAL
and MSI-Net for our interpretation. We use a set of biologically plausible
log-gabor filters for identifying and reconstructing the activation maps of
them using our explainable saliency model. The final saliency map is generated
using these reconstructed activation maps. We also build our own deep saliency
model named cross-concatenated multi-scale residual block based network
(CMRNet) for saliency prediction. Then, we evaluate and compare the performance
of the explainable models derived from UNISAL, MSI-Net and CMRNet on three
benchmark datasets with other state-of-the-art methods. Hence, we propose that
this approach of explainability can be applied to any deep visual saliency
model for interpretation which makes it a generic one.

    

### [[2109.03582] Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes](http://arxiv.org/abs/2109.03582)


  Stochastic processes are random variables with values in some space of paths.
However, reducing a stochastic process to a path-valued random variable ignores
its filtration, i.e. the flow of information carried by the process through
time. By conditioning the process on its filtration, we introduce a family of
higher order kernel mean embeddings (KMEs) that generalizes the notion of KME
and captures additional information related to the filtration. We derive
empirical estimators for the associated higher order maximum mean discrepancies
(MMDs) and prove consistency. We then construct a filtration-sensitive kernel
two-sample test able to pick up information that gets missed by the standard
MMD test. In addition, leveraging our higher order MMDs we construct a family
of universal kernels on stochastic processes that allows to solve real-world
calibration and optimal stopping problems in quantitative finance (such as the
pricing of American options) via classical kernel-based regression methods.
Finally, adapting existing tests for conditional independence to the case of
stochastic processes, we design a causal-discovery algorithm to recover the
causal graph of structural dependencies among interacting bodies solely from
observations of their multidimensional trajectories.

    

### [[2109.03596] AgreementLearning: An End-to-End Framework for Learning with Multiple Annotators without Groundtruth](http://arxiv.org/abs/2109.03596)


  The annotation of domain experts is important for some medical applications
where the objective groundtruth is ambiguous to define, e.g., the
rehabilitation for some chronic diseases, and the prescreening of some
musculoskeletal abnormalities without further medical examinations. However,
improper uses of the annotations may hinder developing reliable models. On one
hand, forcing the use of a single groundtruth generated from multiple
annotations is less informative for the modeling. On the other hand, feeding
the model with all the annotations without proper regularization is noisy given
existing disagreements. For such issues, we propose a novel agreement learning
framework to tackle the challenge of learning from multiple annotators without
objective groundtruth. The framework has two streams, with one stream fitting
with the multiple annotators and the other stream learning agreement
information between the annotators. In particular, the agreement learning
stream produces regularization information to the classifier stream, tuning its
decision to be better in line with the agreement between the annotators. The
proposed method can be easily plugged to existing backbones developed with
majority-voted groundtruth or multiple annotations. Thereon, experiments on two
medical datasets demonstrate improved agreement levels with annotators.

    

### [[2109.03604] Power to the Relational Inductive Bias: Graph Neural Networks in Electrical Power Grids](http://arxiv.org/abs/2109.03604)


  The application of graph neural networks (GNNs) to the domain of electrical
power grids has high potential impact on smart grid monitoring. Even though
there is a natural correspondence of power flow to message-passing in GNNs,
their performance on power grids is not well-understood. We argue that there is
a gap between GNN research driven by benchmarks which contain graphs that
differ from power grids in several important aspects. Additionally, inductive
learning of GNNs across multiple power grid topologies has not been explored
with real-world data. We address this gap by means of (i) defining power grid
graph datasets in inductive settings, (ii) an exploratory analysis of graph
properties, and (iii) an empirical study of the concrete learning task of state
estimation on real-world power grids. Our results show that GNNs are more
robust to noise with up to 400% lower error compared to baselines. Furthermore,
due to the unique properties of electrical grids, we do not observe the well
known over-smoothing phenomenon of GNNs and find the best performing models to
be exceptionally deep with up to 13 layers. This is in stark contrast to
existing benchmark datasets where the consensus is that 2 to 3 layer GNNs
perform best. Our results demonstrate that a key challenge in this domain is to
effectively handle long-range dependence.

    

### [[2109.03615] Tactile Image-to-Image Disentanglement of Contact Geometry from Motion-Induced Shear](http://arxiv.org/abs/2109.03615)


  Robotic touch, particularly when using soft optical tactile sensors, suffers
from distortion caused by motion-dependent shear. The manner in which the
sensor contacts a stimulus is entangled with the tactile information about the
geometry of the stimulus. In this work, we propose a supervised convolutional
deep neural network model that learns to disentangle, in the latent space, the
components of sensor deformations caused by contact geometry from those due to
sliding-induced shear. The approach is validated by reconstructing unsheared
tactile images from sheared images and showing they match unsheared tactile
images collected with no sliding motion. In addition, the unsheared tactile
images give a faithful reconstruction of the contact geometry that is not
possible from the sheared data, and robust estimation of the contact pose that
can be used for servo control sliding around various 2D shapes. Finally, the
contact geometry reconstruction in conjunction with servo control sliding were
used for faithful full object reconstruction of various 2D shapes. The methods
have broad applicability to deep learning models for robots with a
shear-sensitive sense of touch.

    

### [[2109.03616] Deep Learning for Multi-View Ultrasonic Image Fusion](http://arxiv.org/abs/2109.03616)


  Ultrasonic imaging is being used to obtain information about the acoustic
properties of a medium by emitting waves into it and recording their
interaction using ultrasonic transducer arrays. The Delay-And-Sum (DAS)
algorithm forms images using the main path on which reflected signals travel
back to the transducers. In some applications, different insonification paths
can be considered, for instance by placing the transducers at different
locations or if strong reflectors inside the medium are known a-priori. These
different modes give rise to multiple DAS images reflecting different geometric
information about the scatterers and the challenge is to either fuse them into
one image or to directly extract higher-level information regarding the
materials of the medium, e.g., a segmentation map. Traditional image fusion
techniques typically use ad-hoc combinations of pre-defined image transforms,
pooling operations and thresholding. In this work, we propose a deep neural
network (DNN) architecture that directly maps all available data to a
segmentation map while explicitly incorporating the DAS image formation for the
different insonification paths as network layers. This enables information flow
between data pre-processing and image post-processing DNNs, trained end-to-end.
We compare our proposed method to a traditional image fusion technique using
simulated data experiments, mimicking a non-destructive testing application
with four image modes, i.e., two transducer locations and two internal
reflection boundaries. Using our approach, it is possible to obtain much more
accurate segmentation of defects.

    

### [[2109.03624] FaBiAN: A Fetal Brain magnetic resonance Acquisition Numerical phantom](http://arxiv.org/abs/2109.03624)


  Accurate characterization of in utero human brain maturation is critical as
it involves complex and interconnected structural and functional processes that
may influence health later in life. Magnetic resonance imaging is a powerful
tool to investigate equivocal neurological patterns during fetal development.
However, the number of acquisitions of satisfactory quality available in this
cohort of sensitive subjects remains scarce, thus hindering the validation of
advanced image processing techniques. Numerical phantoms can mitigate these
limitations by providing a controlled environment with a known ground truth. In
this work, we present FaBiAN, an open-source Fetal Brain magnetic resonance
Acquisition Numerical phantom that simulates clinical T2-weighted fast spin
echo sequences of the fetal brain. This unique tool is based on a general,
flexible and realistic setup that includes stochastic fetal movements, thus
providing images of the fetal brain throughout maturation comparable to
clinical acquisitions. We demonstrate its value to evaluate the robustness and
optimize the accuracy of an algorithm for super-resolution fetal brain magnetic
resonance imaging from simulated motion-corrupted 2D low-resolution series as
compared to a synthetic high-resolution reference volume. We also show that the
images generated can complement clinical datasets to support data-intensive
deep learning methods for fetal brain tissue segmentation.

    

### [[2109.03655] On Event-Driven Knowledge Graph Completion in Digital Factories](http://arxiv.org/abs/2109.03655)


  Smart factories are equipped with machines that can sense their manufacturing
environments, interact with each other, and control production processes.
Smooth operation of such factories requires that the machines and engineering
personnel that conduct their monitoring and diagnostics share a detailed common
industrial knowledge about the factory, e.g., in the form of knowledge graphs.
Creation and maintenance of such knowledge is expensive and requires
automation. In this work we show how machine learning that is specifically
tailored towards industrial applications can help in knowledge graph
completion. In particular, we show how knowledge completion can benefit from
event logs that are common in smart factories. We evaluate this on the
knowledge graph from a real world-inspired smart factory with encouraging
results.

    

### [[2109.03661] Single Plane-Wave Imaging using Physics-Based Deep Learning](http://arxiv.org/abs/2109.03661)


  In plane-wave imaging, multiple unfocused ultrasound waves are transmitted
into a medium of interest from different angles and an image is formed from the
recorded reflections. The number of plane waves used leads to a trade-off
between frame-rate and image quality, with single-plane-wave (SPW) imaging
being the fastest possible modality with the worst image quality. Recently,
deep learning methods have been proposed to improve ultrasound imaging. One
approach is to use image-to-image networks that work on the formed image and
another is to directly learn a mapping from data to an image. Both approaches
utilize purely data-driven models and require deep, expressive network
architectures, combined with large numbers of training samples to obtain good
results. Here, we propose a data-to-image architecture that incorporates a
wave-physics-based image formation algorithm in-between deep convolutional
neural networks. To achieve this, we implement the Fourier (FK) migration
method as network layers and train the whole network end-to-end. We compare our
proposed data-to-image network with an image-to-image network in simulated data
experiments, mimicking a medical ultrasound application. Experiments show that
it is possible to obtain high-quality SPW images, almost similar to an image
formed using 75 plane waves over an angular range of $\pm$16$^\circ$. This
illustrates the great potential of combining deep neural networks with
physics-based image formation algorithms for SPW imaging.

    

### [[2109.03670] YAHPO Gym -- Design Criteria and a new Multifidelity Benchmark for Hyperparameter Optimization](http://arxiv.org/abs/2109.03670)


  When developing and analyzing new hyperparameter optimization (HPO) methods,
it is vital to empirically evaluate and compare them on well-curated benchmark
suites. In this work, we list desirable properties and requirements for such
benchmarks and propose a new set of challenging and relevant multifidelity HPO
benchmark problems motivated by these requirements. For this, we revisit the
concept of surrogate-based benchmarks and empirically compare them to more
widely-used tabular benchmarks, showing that the latter ones may induce bias in
performance estimation and ranking of HPO methods. We present a new
surrogate-based benchmark suite for multifidelity HPO methods consisting of 9
benchmark collections that constitute over 700 multifidelity HPO problems in
total. All our benchmarks also allow for querying of multiple optimization
targets, enabling the benchmarking of multi-objective HPO. We examine and
compare our benchmark suite with respect to the defined requirements and show
that our benchmarks provide viable additions to existing suites.

    

### [[2109.03675] EMA: Auditing Data Removal from Trained Models](http://arxiv.org/abs/2109.03675)


  Data auditing is a process to verify whether certain data have been removed
from a trained model. A recently proposed method (Liu et al. 20) uses
Kolmogorov-Smirnov (KS) distance for such data auditing. However, it fails
under certain practical conditions. In this paper, we propose a new method
called Ensembled Membership Auditing (EMA) for auditing data removal to
overcome these limitations. We compare both methods using benchmark datasets
(MNIST and SVHN) and Chest X-ray datasets with multi-layer perceptrons (MLP)
and convolutional neural networks (CNN). Our experiments show that EMA is
robust under various conditions, including the failure cases of the previously
proposed method. Our code is available at: this https URL.

    

### [[2109.03676] Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization](http://arxiv.org/abs/2109.03676)


  Given multiple source domains, domain generalization aims at learning a
universal model that performs well on any unseen but related target domain. In
this work, we focus on the domain generalization scenario where domain shifts
occur among class-conditional distributions of different domains. Existing
approaches are not sufficiently robust when the variation of conditional
distributions given the same class is large. In this work, we extend the
concept of distributional robust optimization to solve the class-conditional
domain generalization problem. Our approach optimizes the worst-case
performance of a classifier over class-conditional distributions within a
Wasserstein ball centered around the barycenter of the source conditional
distributions. We also propose an iterative algorithm for learning the optimal
radius of the Wasserstein balls automatically. Experiments show that the
proposed framework has better performance on unseen target domain than
approaches without domain generalization.

    

### [[2109.03685] Open Aspect Target Sentiment Classification with Natural Language Prompts](http://arxiv.org/abs/2109.03685)


  For many business applications, we often seek to analyze sentiments
associated with any arbitrary aspects of commercial products, despite having a
very limited amount of labels or even without any labels at all. However,
existing aspect target sentiment classification (ATSC) models are not trainable
if annotated datasets are not available. Even with labeled data, they fall
short of reaching satisfactory performance. To address this, we propose simple
approaches that better solve ATSC with natural language prompts, enabling the
task under zero-shot cases and enhancing supervised settings, especially for
few-shot cases. Under the few-shot setting for SemEval 2014 Task 4 laptop
domain, our method of reformulating ATSC as an NLI task outperforms supervised
SOTA approaches by up to 24.13 accuracy points and 33.14 macro F1 points.
Moreover, we demonstrate that our prompts could handle implicitly stated
aspects as well: our models reach about 77% accuracy on detecting sentiments
for aspect categories (e.g., food), which do not necessarily appear within the
text, even though we trained the models only with explicitly mentioned aspect
terms (e.g., fajitas) from just 16 reviews - while the accuracy of the
no-prompt baseline is only around 65%.

    

### [[2109.03697] U-FNO -- an enhanced Fourier neural operator based-deep learning model for multiphase flow](http://arxiv.org/abs/2109.03697)


  Numerical simulation of multiphase flow in porous media is essential for many
geoscience applications. However, due to the multi-physics, non-linear, and
multi-scale problem nature, these simulations are very expensive at desirable
grid resolutions, and the computational cost often impedes rigorous engineering
decision-making. Machine learning methods provide faster alternatives to
traditional simulators by training neural network models with numerical
simulation data mappings. Traditional convolutional neural network (CNN)-based
models are accurate yet data-intensive and are prone to overfitting. Here we
present a new architecture, U-FNO, an enhanced Fourier neural operator for
solving the multiphase flow problem. The U-FNO is designed based on the Fourier
neural operator (FNO) that learns an integral kernel in the Fourier space.
Through a systematic comparison among a CNN benchmark and three types of FNO
variations on a CO2-water multiphase problem in the context of CO2 geological
storage, we show that the U-FNO architecture has the advantages of both
traditional CNN and original FNO, providing significantly more accurate and
efficient performance than previous architectures. The trained U-FNO provides
gas saturation and pressure buildup predictions with a 10,000 times speedup
compared to traditional numerical simulators while maintaining similar
accuracy.

    

### [[2109.03699] Sample and Communication-Efficient Decentralized Actor-Critic Algorithms with Finite-Time Analysis](http://arxiv.org/abs/2109.03699)


  Actor-critic (AC) algorithms have been widely adopted in decentralized
multi-agent systems to learn the optimal joint control policy. However,
existing decentralized AC algorithms either do not preserve the privacy of
agents or are not sample and communication-efficient. In this work, we develop
two decentralized AC and natural AC (NAC) algorithms that are private, and
sample and communication-efficient. In both algorithms, agents share noisy
information to preserve privacy and adopt mini-batch updates to improve sample
and communication efficiency. Particularly for decentralized NAC, we develop a
decentralized Markovian SGD algorithm with an adaptive mini-batch size to
efficiently compute the natural policy gradient. Under Markovian sampling and
linear function approximation, we prove the proposed decentralized AC and NAC
algorithms achieve the state-of-the-art sample complexities
$\mathcal{O}\big(\epsilon^{-2}\ln(\epsilon^{-1})\big)$ and
$\mathcal{O}\big(\epsilon^{-3}\ln(\epsilon^{-1})\big)$, respectively, and the
same small communication complexity
$\mathcal{O}\big(\epsilon^{-1}\ln(\epsilon^{-1})\big)$. Numerical experiments
demonstrate that the proposed algorithms achieve lower sample and communication
complexities than the existing decentralized AC algorithm.

    

### [[2109.03708] Self-explaining variational posterior distributions for Gaussian Process models](http://arxiv.org/abs/2109.03708)


  Bayesian methods have become a popular way to incorporate prior knowledge and
a notion of uncertainty into machine learning models. At the same time, the
complexity of modern machine learning makes it challenging to comprehend a
model's reasoning process, let alone express specific prior assumptions in a
rigorous manner. While primarily interested in the former issue, recent
developments intransparent machine learning could also broaden the range of
prior information that we can provide to complex Bayesian models. Inspired by
the idea of self-explaining models, we introduce a corresponding concept for
variational GaussianProcesses. On the one hand, our contribution improves
transparency for these types of models. More importantly though, our proposed
self-explaining variational posterior distribution allows to incorporate both
general prior knowledge about a target function as a whole and prior knowledge
about the contribution of individual features.

    

### [[2109.03709] Priming PCA with EigenGame](http://arxiv.org/abs/2109.03709)


  We introduce primed-PCA (pPCA), an extension of the recently proposed
EigenGame algorithm for computing principal components in a large-scale setup.
Our algorithm first runs EigenGame to get an approximation of the principal
components, and then applies an exact PCA in the subspace they span. Since this
subspace is of small dimension in any practical use of EigenGame, this second
step is extremely cheap computationally. Nonetheless, it improves accuracy
significantly for a given computational budget across datasets. In this setup,
the purpose of EigenGame is to narrow down the search space, and prepare the
data for the second step, an exact calculation.
We show formally that pPCA improves upon EigenGame under very mild
conditions, and we provide experimental validation on both synthetic and real
large-scale datasets showing that it systematically translates to improved
performance. In our experiments we achieve improvements in convergence speed by
factors of 5-25 on the datasets of the original EigenGame paper.

    

### [[2109.03718] Multiscale Laplacian Learning](http://arxiv.org/abs/2109.03718)


  Machine learning methods have greatly changed science, engineering, finance,
business, and other fields. Despite the tremendous accomplishments of machine
learning and deep learning methods, many challenges still remain. In
particular, the performance of machine learning methods is often severely
affected in case of diverse data, usually associated with smaller data sets or
data related to areas of study where the size of the data sets is constrained
by the complexity and/or high cost of experiments. Moreover, data with limited
labeled samples is a challenge to most learning approaches. In this paper, the
aforementioned challenges are addressed by integrating graph-based frameworks,
multiscale structure, modified and adapted optimization procedures and
semi-supervised techniques. This results in two innovative multiscale Laplacian
learning (MLL) approaches for machine learning tasks, such as data
classification, and for tackling diverse data, data with limited samples and
smaller data sets. The first approach, called multikernel manifold learning
(MML), integrates manifold learning with multikernel information and solves a
regularization problem consisting of a loss function and a warped kernel
regularizer using multiscale graph Laplacians. The second approach, called the
multiscale MBO (MMBO) method, introduces multiscale Laplacians to a
modification of the famous classical Merriman-Bence-Osher (MBO) scheme, and
makes use of fast solvers for finding the approximations to the extremal
eigenvectors of the graph Laplacian. We demonstrate the performance of our
methods experimentally on a variety of data sets, such as biological, text and
image data, and compare them favorably to existing approaches.

    

### [[2109.03723] Disentangling Alzheimer's disease neurodegeneration from typical brain aging using machine learning](http://arxiv.org/abs/2109.03723)


  Neuroimaging biomarkers that distinguish between typical brain aging and
Alzheimer's disease (AD) are valuable for determining how much each contributes
to cognitive decline. Machine learning models can derive multi-variate brain
change patterns related to the two processes, including the SPARE-AD (Spatial
Patterns of Atrophy for Recognition of Alzheimer's Disease) and SPARE-BA (of
Brain Aging) investigated herein. However, substantial overlap between brain
regions affected in the two processes confounds measuring them independently.
We present a methodology toward disentangling the two. T1-weighted MRI images
of 4,054 participants (48-95 years) with AD, mild cognitive impairment (MCI),
or cognitively normal (CN) diagnoses from the iSTAGING (Imaging-based
coordinate SysTem for AGIng and NeurodeGenerative diseases) consortium were
analyzed. First, a subset of AD patients and CN adults were selected based
purely on clinical diagnoses to train SPARE-BA1 (regression of age using CN
individuals) and SPARE-AD1 (classification of CN versus AD). Second, analogous
groups were selected based on clinical and molecular markers to train SPARE-BA2
and SPARE-AD2: amyloid-positive (A+) AD continuum group (consisting of A+AD,
A+MCI, and A+ and tau-positive CN individuals) and amyloid-negative (A-) CN
group. Finally, the combined group of the AD continuum and A-/CN individuals
was used to train SPARE-BA3, with the intention to estimate brain age
regardless of AD-related brain changes. Disentangled SPARE models derived brain
patterns that were more specific to the two types of the brain changes.
Correlation between the SPARE-BA and SPARE-AD was significantly reduced.
Correlation of disentangled SPARE-AD was non-inferior to the molecular
measurements and to the number of APOE4 alleles, but was less to AD-related
psychometric test scores, suggesting contribution of advanced brain aging to
these scores.

    

### [[2109.03747] Conservative Policy Construction Using Variational Autoencoders for Logged Data with Missing Values](http://arxiv.org/abs/2109.03747)


  In high-stakes applications of data-driven decision making like healthcare,
it is of paramount importance to learn a policy that maximizes the reward while
avoiding potentially dangerous actions when there is uncertainty. There are two
main challenges usually associated with this problem. Firstly, learning through
online exploration is not possible due to the critical nature of such
applications. Therefore, we need to resort to observational datasets with no
counterfactuals. Secondly, such datasets are usually imperfect, additionally
cursed with missing values in the attributes of features. In this paper, we
consider the problem of constructing personalized policies using logged data
when there are missing values in the attributes of features in both training
and test data. The goal is to recommend an action (treatment) when $\Xt$, a
degraded version of $\Xb$ with missing values, is observed. We consider three
strategies for dealing with missingness. In particular, we introduce the
\textit{conservative strategy} where the policy is designed to safely handle
the uncertainty due to missingness. In order to implement this strategy we need
to estimate posterior distribution $p(\Xb|\Xt)$, we use variational autoencoder
to achieve this. In particular, our method is based on partial variational
autoencoders (PVAE) which are designed to capture the underlying structure of
features with missing values.

    

### [[2109.03748] A robust approach for deep neural networks in presence of label noise: relabelling and filtering instances during training](http://arxiv.org/abs/2109.03748)


  Deep learning has outperformed other machine learning algorithms in a variety
of tasks, and as a result, it has become more and more popular and used.
However, as other machine learning algorithms, deep learning, and convolutional
neural networks (CNNs) in particular, perform worse when the data sets present
label noise. Therefore, it is important to develop algorithms that help the
training of deep networks and their generalization to noise-free test sets. In
this paper, we propose a robust training strategy against label noise, called
RAFNI, that can be used with any CNN. This algorithm filters and relabels
instances of the training set based on the predictions and their probabilities
made by the backbone neural network during the training process. That way, this
algorithm improves the generalization ability of the CNN on its own. RAFNI
consists of three mechanisms: two mechanisms that filter instances and one
mechanism that relabels instances. In addition, it does not suppose that the
noise rate is known nor does it need to be estimated. We evaluated our
algorithm using different data sets of several sizes and characteristics. We
also compared it with state-of-the-art models using the CIFAR10 and CIFAR100
benchmarks under different types and rates of label noise and found that RAFNI
achieves better results in most cases.

    

### [[2109.03756] Diagnostics-Guided Explanation Generation](http://arxiv.org/abs/2109.03756)


  Explanations shed light on a machine learning model's rationales and can aid
in identifying deficiencies in its reasoning process. Explanation generation
models are typically trained in a supervised way given human explanations. When
such annotations are not available, explanations are often selected as those
portions of the input that maximise a downstream task's performance, which
corresponds to optimising an explanation's Faithfulness to a given model.
Faithfulness is one of several so-called diagnostic properties, which prior
work has identified as useful for gauging the quality of an explanation without
requiring annotations. Other diagnostic properties are Data Consistency, which
measures how similar explanations are for similar input instances, and
Confidence Indication, which shows whether the explanation reflects the
confidence of the model. In this work, we show how to directly optimise for
these diagnostic properties when training a model to generate sentence-level
explanations, which markedly improves explanation quality, agreement with human
rationales, and downstream task performance on three complex reasoning tasks.

    

### [[2109.03764] Active Learning by Acquiring Contrastive Examples](http://arxiv.org/abs/2109.03764)


  Common acquisition functions for active learning use either uncertainty or
diversity sampling, aiming to select difficult and diverse data points from the
pool of unlabeled data, respectively. In this work, leveraging the best of both
worlds, we propose an acquisition function that opts for selecting
\textit{contrastive examples}, i.e. data points that are similar in the model
feature space and yet the model outputs maximally different predictive
likelihoods. We compare our approach, CAL (Contrastive Active Learning), with a
diverse set of acquisition functions in four natural language understanding
tasks and seven datasets. Our experiments show that CAL performs consistently
better or equal than the best performing baseline across all tasks, on both
in-domain and out-of-domain data. We also conduct an extensive ablation study
of our method and we further analyze all actively acquired datasets showing
that CAL achieves a better trade-off between uncertainty and diversity compared
to other strategies.

    

### [[2109.03769] Training Algorithm Matters for the Performance of Neural Network Potential](http://arxiv.org/abs/2109.03769)


  One hidden yet important issue for developing neural network potentials
(NNPs) is the choice of training algorithm. Here we compare the performance of
two popular training algorithms, the adaptive moment estimation algorithm
(Adam) and the extended Kalman filter algorithm (EKF), using the
Behler-Parrinello neural network (BPNN) and two publicly accessible datasets of
liquid water. It is found that NNPs trained with EKF are more transferable and
less sensitive to the value of the learning rate, as compared to Adam. In both
cases, error metrics of the test set do not always serve as a good indicator
for the actual performance of NNPs. Instead, we show that their performance
correlates well with a Fisher information based similarity measure.

    

### [[2109.03775] FedZKT: Zero-Shot Knowledge Transfer towards Heterogeneous On-Device Models in Federated Learning](http://arxiv.org/abs/2109.03775)


  Federated learning enables distributed devices to collaboratively learn a
shared prediction model without centralizing on-device training data. Most of
the current algorithms require comparable individual efforts to train on-device
models with the same structure and size, impeding participation from
resource-constrained devices. Given the widespread yet heterogeneous devices
nowadays, this paper proposes a new framework supporting federated learning
across heterogeneous on-device models via Zero-shot Knowledge Transfer, named
by FedZKT. Specifically, FedZKT allows participating devices to independently
determine their on-device models. To transfer knowledge across on-device
models, FedZKT develops a zero-shot distillation approach contrary to certain
prior research based on a public dataset or a pre-trained data generator. To
utmostly reduce on-device workload, the resource-intensive distillation task is
assigned to the server, which constructs a generator to adversarially train
with the ensemble of the received heterogeneous on-device models. The distilled
central knowledge will then be sent back in the form of the corresponding
on-device model parameters, which can be easily absorbed at the device side.
Experimental studies demonstrate the effectiveness and the robustness of FedZKT
towards heterogeneous on-device models and challenging federated learning
scenarios, such as non-iid data distribution and straggler effects.

    

### [[2109.03777] Forget me not: A Gentle Reminder to Mind the Simple Multi-Layer Perceptron Baseline for Text Classification](http://arxiv.org/abs/2109.03777)


  Graph neural networks have triggered a resurgence of graph-based text
classification. We show that already a simple MLP baseline achieves comparable
performance on benchmark datasets, questioning the importance of synthetic
graph structures. When considering an inductive scenario, i. e., when adding
new documents to a corpus, a simple MLP even outperforms most graph-based
models. We further fine-tune DistilBERT for comparison and find that it
outperforms all state-of-the-art models. We suggest that future studies use at
least an MLP baseline to contextualize the results. We provide recommendations
for the design and training of such a baseline.

    

### [[2109.03778] Axial multi-layer perceptron architecture for automatic segmentation of choroid plexus in multiple sclerosis](http://arxiv.org/abs/2109.03778)


  Choroid plexuses (CP) are structures of the ventricles of the brain which
produce most of the cerebrospinal fluid (CSF). Several postmortem and in vivo
studies have pointed towards their role in the inflammatory process in multiple
sclerosis (MS). Automatic segmentation of CP from MRI thus has high value for
studying their characteristics in large cohorts of patients. To the best of our
knowledge, the only freely available tool for CP segmentation is FreeSurfer but
its accuracy for this specific structure is poor. In this paper, we propose to
automatically segment CP from non-contrast enhanced T1-weighted MRI. To that
end, we introduce a new model called "Axial-MLP" based on an assembly of Axial
multi-layer perceptrons (MLPs). This is inspired by recent works which showed
that the self-attention layers of Transformers can be replaced with MLPs. This
approach is systematically compared with a standard 3D U-Net, nnU-Net,
Freesurfer and FastSurfer. For our experiments, we make use of a dataset of 141
subjects (44 controls and 97 patients with MS). We show that all the tested
deep learning (DL) methods outperform FreeSurfer (Dice around 0.7 for DL vs
0.33 for FreeSurfer). Axial-MLP is competitive with U-Nets even though it is
slightly less accurate. The conclusions of our paper are two-fold: 1) the
studied deep learning methods could be useful tools to study CP in large
cohorts of MS patients; 2)~Axial-MLP is a potentially viable alternative to
convolutional neural networks for such tasks, although it could benefit from
further improvements.

    

### [[2109.03781] Highly Scalable and Provably Accurate Classification in Poincare Balls](http://arxiv.org/abs/2109.03781)


  Many high-dimensional and large-volume data sets of practical relevance have
hierarchical structures induced by trees, graphs or time series. Such data sets
are hard to process in Euclidean spaces and one often seeks low-dimensional
embeddings in other space forms to perform required learning tasks. For
hierarchical data, the space of choice is a hyperbolic space since it
guarantees low-distortion embeddings for tree-like structures. Unfortunately,
the geometry of hyperbolic spaces has properties not encountered in Euclidean
spaces that pose challenges when trying to rigorously analyze algorithmic
solutions. Here, for the first time, we establish a unified framework for
learning scalable and simple hyperbolic linear classifiers with provable
performance guarantees. The gist of our approach is to focus on Poincaré ball
models and formulate the classification problems using tangent space
formalisms. Our results include a new hyperbolic and second-order perceptron
algorithm as well as an efficient and highly accurate convex optimization setup
for hyperbolic support vector machine classifiers. All algorithms provably
converge and are highly scalable as they have complexities comparable to those
of their Euclidean counterparts. Their performance accuracies on synthetic data
sets comprising millions of points, as well as on complex real-world data sets
such as single-cell RNA-seq expression measurements, CIFAR10, Fashion-MNIST and
mini-ImageNet.

    

### [[2109.03784] A Survey on Machine Learning Techniques for Auto Labeling of Video, Audio, and Text Data](http://arxiv.org/abs/2109.03784)


  Machine learning has been utilized to perform tasks in many different domains
such as classification, object detection, image segmentation and natural
language analysis. Data labeling has always been one of the most important
tasks in machine learning. However, labeling large amounts of data increases
the monetary cost in machine learning. As a result, researchers started to
focus on reducing data annotation and labeling costs. Transfer learning was
designed and widely used as an efficient approach that can reasonably reduce
the negative impact of limited data, which in turn, reduces the data
preparation cost. Even transferring previous knowledge from a source domain
reduces the amount of data needed in a target domain. However, large amounts of
annotated data are still demanded to build robust models and improve the
prediction accuracy of the model. Therefore, researchers started to pay more
attention on auto annotation and labeling. In this survey paper, we provide a
review of previous techniques that focuses on optimized data annotation and
labeling for video, audio, and text data.

    

### [[2109.03795] Desiderata for Representation Learning: A Causal Perspective](http://arxiv.org/abs/2109.03795)


  Representation learning constructs low-dimensional representations to
summarize essential features of high-dimensional data. This learning problem is
often approached by describing various desiderata associated with learned
representations; e.g., that they be non-spurious, efficient, or disentangled.
It can be challenging, however, to turn these intuitive desiderata into formal
criteria that can be measured and enhanced based on observed data. In this
paper, we take a causal perspective on representation learning, formalizing
non-spuriousness and efficiency (in supervised representation learning) and
disentanglement (in unsupervised representation learning) using counterfactual
quantities and observable consequences of causal assertions. This yields
computable metrics that can be used to assess the degree to which
representations satisfy the desiderata of interest and learn non-spurious and
disentangled representations from single observational datasets.

    

### [[2109.03798] AppQ: Warm-starting App Recommendation Based on View Graphs](http://arxiv.org/abs/2109.03798)


  Current app ranking and recommendation systems are mainly based on
user-generated information, e.g., number of downloads and ratings. However, new
apps often have few (or even no) user feedback, suffering from the classic
cold-start problem. How to quickly identify and then recommend new apps of high
quality is a challenging issue. Here, a fundamental requirement is the
capability to accurately measure an app's quality based on its inborn features,
rather than user-generated features. Since users obtain first-hand experience
of an app by interacting with its views, we speculate that the inborn features
are largely related to the visual quality of individual views in an app and the
ways the views switch to one another. In this work, we propose AppQ, a novel
app quality grading and recommendation system that extracts inborn features of
apps based on app source code. In particular, AppQ works in parallel to perform
code analysis to extract app-level features as well as dynamic analysis to
capture view-level layout hierarchy and the switching among views. Each app is
then expressed as an attributed view graph, which is converted into a vector
and fed to classifiers for recognizing its quality classes. Our evaluation with
an app dataset from Google Play reports that AppQ achieves the best performance
with accuracy of 85.0\%. This shows a lot of promise to warm-start app grading
and recommendation systems with AppQ.

    

### [[2109.03805] Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking](http://arxiv.org/abs/2109.03805)


  Panoptic scene understanding and tracking of dynamic agents are essential for
robots and automated vehicles to navigate in urban environments. As LiDARs
provide accurate illumination-independent geometric depictions of the scene,
performing these tasks using LiDAR point clouds provides reliable predictions.
However, existing datasets lack diversity in the type of urban scenes and have
a limited number of dynamic object instances which hinders both learning of
these tasks as well as credible benchmarking of the developed methods. In this
paper, we introduce the large-scale Panoptic nuScenes benchmark dataset that
extends our popular nuScenes dataset with point-wise groundtruth annotations
for semantic segmentation, panoptic segmentation, and panoptic tracking tasks.
To facilitate comparison, we provide several strong baselines for each of these
tasks on our proposed dataset. Moreover, we analyze the drawbacks of the
existing metrics for the panoptic tracking problem and propose a novel
instance-centric metric that addresses the concerns. We present extensive
experiments that demonstrate the utility of Panoptic nuScenes compared to
existing datasets and make the online evaluation server available at
\url{this http URL}. We believe that this extension will accelerate the research
of novel methods for scene understanding of dynamic urban environments.

    

### [[2109.03812] fastMRI+: Clinical Pathology Annotations for Knee and Brain Fully Sampled Multi-Coil MRI Data](http://arxiv.org/abs/2109.03812)


  Improving speed and image quality of Magnetic Resonance Imaging (MRI) via
novel reconstruction approaches remains one of the highest impact applications
for deep learning in medical imaging. The fastMRI dataset, unique in that it
contains large volumes of raw MRI data, has enabled significant advances in
accelerating MRI using deep learning-based reconstruction methods. While the
impact of the fastMRI dataset on the field of medical imaging is unquestioned,
the dataset currently lacks clinical expert pathology annotations, critical to
addressing clinically relevant reconstruction frameworks and exploring
important questions regarding rendering of specific pathology using such novel
approaches. This work introduces fastMRI+, which consists of 16154
subspecialist expert bounding box annotations and 13 study-level labels for 22
different pathology categories on the fastMRI knee dataset, and 7570
subspecialist expert bounding box annotations and 643 study-level labels for 30
different pathology categories for the fastMRI brain dataset. The fastMRI+
dataset is open access and aims to support further research and advancement of
medical imaging in MRI reconstruction and beyond.

    

### [[1811.03179] How Well Generative Adversarial Networks Learn Distributions](http://arxiv.org/abs/1811.03179)


  This paper studies the rates of convergence for learning distributions
implicitly with the adversarial framework and Generative Adversarial Networks
(GANs), which subsume Wasserstein, Sobolev, MMD GAN, and Generalized/Simulated
Method of Moments (GMM/SMM) as special cases. We study a wide range of
parametric and nonparametric target distributions under a host of objective
evaluation metrics. We investigate how to obtain valid statistical guarantees
for GANs through the lens of regularization. On the nonparametric end, we
derive the optimal minimax rates for distribution estimation under the
adversarial framework. On the parametric end, we establish a theory for general
neural network classes (including deep leaky ReLU networks) that characterizes
the interplay on the choice of generator and discriminator pair. We discover
and isolate a new notion of regularization, called the
generator-discriminator-pair regularization, that sheds light on the advantage
of GANs compared to classical parametric and nonparametric approaches for
explicit distribution estimation. We develop novel oracle inequalities as the
main technical tools for analyzing GANs, which are of independent interest.

    

### [[1908.11435] Improving Adversarial Robustness via Attention and Adversarial Logit Pairing](http://arxiv.org/abs/1908.11435)


  Though deep neural networks have achieved the state of the art performance in
visual classification, recent studies have shown that they are all vulnerable
to the attack of adversarial examples. In this paper, we develop improved
techniques for defending against adversarial examples. First, we propose an
enhanced defense technique denoted Attention and Adversarial Logit
Pairing(AT+ALP), which encourages both attention map and logit for the pairs of
examples to be similar. When being applied to clean examples and their
adversarial counterparts, AT+ALP improves accuracy on adversarial examples over
adversarial training. We show that AT+ALP can effectively increase the average
activations of adversarial examples in the key area and demonstrate that it
focuses on discriminate features to improve the robustness of the model.
Finally, we conduct extensive experiments using a wide range of datasets and
the experiment results show that our AT+ALP achieves the state of the art
defense performance. For example, on 17 Flower Category Database, under strong
200-iteration PGD gray-box and black-box attacks where prior art has 34% and
39% accuracy, our method achieves 50% and 51%. Compared with previous work, our
work is evaluated under highly challenging PGD attack: the maximum perturbation
$\epsilon \in \{0.25,0.5\}$ i.e. $L_\infty \in \{0.25,0.5\}$ with 10 to 200
attack iterations. To the best of our knowledge, such a strong attack has not
been previously explored on a wide range of datasets.

    

### [[2001.09684] Challenges and Countermeasures for Adversarial Attacks on Deep Reinforcement Learning](http://arxiv.org/abs/2001.09684)


  Deep Reinforcement Learning (DRL) has numerous applications in the real world
thanks to its outstanding ability in quickly adapting to the surrounding
environments. Despite its great advantages, DRL is susceptible to adversarial
attacks, which precludes its use in real-life critical systems and applications
(e.g., smart grids, traffic controls, and autonomous vehicles) unless its
vulnerabilities are addressed and mitigated. Thus, this paper provides a
comprehensive survey that discusses emerging attacks in DRL-based systems and
the potential countermeasures to defend against these attacks. We first cover
some fundamental backgrounds about DRL and present emerging adversarial attacks
on machine learning techniques. We then investigate more details of the
vulnerabilities that the adversary can exploit to attack DRL along with the
state-of-the-art countermeasures to prevent such attacks. Finally, we highlight
open issues and research challenges for developing solutions to deal with
attacks for DRL-based intelligent systems.

    

### [[2003.07040] Discrete-Valued Latent Preference Matrix Estimation with Graph Side Information](http://arxiv.org/abs/2003.07040)


  Incorporating graph side information into recommender systems has been widely
used to better predict ratings, but relatively few works have focused on
theoretical guarantees. Ahn et al. (2018) firstly characterized the optimal
sample complexity in the presence of graph side information, but the results
are limited due to strict, unrealistic assumptions made on the unknown latent
preference matrix and the structure of user clusters. In this work, we propose
a new model in which 1) the unknown latent preference matrix can have any
discrete values, and 2) users can be clustered into multiple clusters, thereby
relaxing the assumptions made in prior work. Under this new model, we fully
characterize the optimal sample complexity and develop a
computationally-efficient algorithm that matches the optimal sample complexity.
Our algorithm is robust to model errors and outperforms the existing algorithms
in terms of prediction performance on both synthetic and real data.

    

### [[2004.13805] Multilingual Chart-based Constituency Parse Extraction from Pre-trained Language Models](http://arxiv.org/abs/2004.13805)


  As it has been unveiled that pre-trained language models (PLMs) are to some
extent capable of recognizing syntactic concepts in natural language, much
effort has been made to develop a method for extracting complete (binary)
parses from PLMs without training separate parsers. We improve upon this
paradigm by proposing a novel chart-based method and an effective top-K
ensemble technique. Moreover, we demonstrate that we can broaden the scope of
application of the approach into multilingual settings. Specifically, we show
that by applying our method on multilingual PLMs, it becomes possible to induce
non-trivial parses for sentences from nine languages in an integrated and
language-agnostic manner, attaining performance superior or comparable to that
of unsupervised PCFGs. We also verify that our approach is robust to
cross-lingual transfer. Finally, we provide analyses on the inner workings of
our method. For instance, we discover universal attention heads which are
consistently sensitive to syntactic information irrespective of the input
language.

    

### [[2004.13847] Word Equations: Inherently Interpretable Sparse Word Embeddingsthrough Sparse Coding](http://arxiv.org/abs/2004.13847)


  Word embeddings are a powerful natural lan-guage processing technique, but
they are ex-tremely difficult to interpret. To enable inter-pretable NLP
models, we create vectors whereeach dimension isinherently interpretable.
Byinherently interpretable, we mean a systemwhere each dimension is associated
with somehuman-understandablehintthat can describethe meaning of that
dimension. In order tocreate more interpretable word embeddings,we transform
pretrained dense word embed-dings into sparse embeddings. These new em-beddings
are inherently interpretable: each oftheir dimensions is created from and
repre-sents a natural language word or specific gram-matical concept. We
construct these embed-dings through sparse coding, where each vec-tor in the
basis set is itself a word embedding.Therefore, each dimension of our sparse
vec-tors corresponds to a natural language word.We also show that models
trained using thesesparse embeddings can achieve good perfor-mance and are more
interpretable in practice,including through human evaluations.

    

### [[2005.13183] Interpretable and Efficient Heterogeneous Graph Convolutional Network](http://arxiv.org/abs/2005.13183)


  Graph Convolutional Network (GCN) has achieved extraordinary success in
learning effective task-specific representations of nodes in graphs. However,
regarding Heterogeneous Information Network (HIN), existing HIN-oriented GCN
methods still suffer from two deficiencies: (1) they cannot flexibly explore
all possible meta-paths and extract the most useful ones for a target object,
which hinders both effectiveness and interpretability; (2) they often need to
generate intermediate meta-path based dense graphs, which leads to high
computational complexity. To address the above issues, we propose an
interpretable and efficient Heterogeneous Graph Convolutional Network (ie-HGCN)
to learn the representations of objects in HINs. It is designed as a
hierarchical aggregation architecture, i.e., object-level aggregation first,
followed by type-level aggregation. The novel architecture can automatically
extract useful meta-paths for each object from all possible meta-paths (within
a length limit), which brings good model interpretability. It can also reduce
the computational cost by avoiding intermediate HIN transformation and
neighborhood attention. We provide theoretical analysis about the proposed
ie-HGCN in terms of evaluating the usefulness of all possible meta-paths, its
connection to the spectral graph convolution on HINs, and its quasi-linear time
complexity. Extensive experiments on three real network datasets demonstrate
the superiority of ie-HGCN over the state-of-the-art methods.

    

### [[2006.06664] Quasi-Dense Similarity Learning for Multiple Object Tracking](http://arxiv.org/abs/2006.06664)


  Similarity learning has been recognized as a crucial step for object
tracking. However, existing multiple object tracking methods only use sparse
ground truth matching as the training objective, while ignoring the majority of
the informative regions on the images. In this paper, we present Quasi-Dense
Similarity Learning, which densely samples hundreds of region proposals on a
pair of images for contrastive learning. We can directly combine this
similarity learning with existing detection methods to build Quasi-Dense
Tracking (QDTrack) without turning to displacement regression or motion priors.
We also find that the resulting distinctive feature space admits a simple
nearest neighbor search at the inference time. Despite its simplicity, QDTrack
outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking
benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external
training data. Compared to methods with similar detectors, it boosts almost 10
points of MOTA and significantly decreases the number of ID switches on BDD100K
and Waymo datasets. Our code and trained models are available at
http://vis.xyz/pub/qdtrack.

    

### [[2006.07200] Learning to Communicate Using Counterfactual Reasoning](http://arxiv.org/abs/2006.07200)


  Learning to communicate in order to share state information is an active
problem in the area of multi-agent reinforcement learning (MARL). The credit
assignment problem, the non-stationarity of the communication environment and
the creation of influenceable agents are major challenges within this research
field which need to be overcome in order to learn a valid communication
protocol. This paper introduces the novel multi-agent counterfactual
communication learning (MACC) method which adapts counterfactual reasoning in
order to overcome the credit assignment problem for communicating agents.
Secondly, the non-stationarity of the communication environment while learning
the communication Q-function is overcome by creating the communication
Q-function using the action policy of the other agents and the Q-function of
the action environment. Additionally, a social loss function is introduced in
order to create influenceable agents which is required to learn a valid
communication protocol. Our experiments show that MACC is able to outperform
the state-of-the-art baselines in four different scenarios in the Particle
environment.

    

### [[2006.10820] A New One-Point Residual-Feedback Oracle For Black-Box Learning and Control](http://arxiv.org/abs/2006.10820)


  Zeroth-order optimization (ZO) algorithms have been recently used to solve
black-box or simulation-based learning and control problems, where the gradient
of the objective function cannot be easily computed but can be approximated
using the objective function values. Many existing ZO algorithms adopt
two-point feedback schemes due to their fast convergence rate compared to
one-point feedback schemes. However, two-point schemes require two evaluations
of the objective function at each iteration, which can be impractical in
applications where the data are not all available a priori, e.g., in online
optimization. In this paper, we propose a novel one-point feedback scheme that
queries the function value once at each iteration and estimates the gradient
using the residual between two consecutive points. When optimizing a
deterministic Lipschitz function, we show that the query complexity of ZO with
the proposed one-point residual feedback matches that of ZO with the existing
two-point schemes. Moreover, the query complexity of the proposed algorithm can
be improved when the objective function has Lipschitz gradient. Then, for
stochastic bandit optimization problems where only noisy objective function
values are given, we show that ZO with one-point residual feedback achieves the
same convergence rate as that of two-point scheme with uncontrollable data
samples. We demonstrate the effectiveness of the proposed one-point residual
feedback via extensive numerical experiments.

    

### [[2007.15710] Privacy Enhancing Machine Learning via Removal of Unwanted Dependencies](http://arxiv.org/abs/2007.15710)


  The rapid rise of IoT and Big Data has facilitated copious data driven
applications to enhance our quality of life. However, the omnipresent and
all-encompassing nature of the data collection can generate privacy concerns.
Hence, there is a strong need to develop techniques that ensure the data serve
only the intended purposes, giving users control over the information they
share. To this end, this paper studies new variants of supervised and
adversarial learning methods, which remove the sensitive information in the
data before they are sent out for a particular application. The explored
methods optimize privacy preserving feature mappings and predictive models
simultaneously in an end-to-end fashion. Additionally, the models are built
with an emphasis on placing little computational burden on the user side so
that the data can be desensitized on device in a cheap manner. Experimental
results on mobile sensing and face datasets demonstrate that our models can
successfully maintain the utility performances of predictive models while
causing sensitive predictions to perform poorly.

    

### [[2010.04678] Concurrent Alternating Least Squares for multiple simultaneous Canonical Polyadic Decompositions](http://arxiv.org/abs/2010.04678)


  Tensor decompositions, such as CANDECOMP/PARAFAC (CP), are widely used in a
variety of applications, such as chemometrics, signal processing, and machine
learning. A broadly used method for computing such decompositions relies on the
Alternating Least Squares (ALS) algorithm. When the number of components is
small, regardless of its implementation, ALS exhibits low arithmetic intensity,
which severely hinders its performance and makes GPU offloading ineffective. We
observe that, in practice, experts often have to compute multiple
decompositions of the same tensor, each with a small number of components
(typically fewer than 20), to ultimately find the best ones to use for the
application at hand. In this paper, we illustrate how multiple decompositions
of the same tensor can be fused together at the algorithmic level to increase
the arithmetic intensity. Therefore, it becomes possible to make efficient use
of GPUs for further speedups; at the same time the technique is compatible with
many enhancements typically used in ALS, such as line search, extrapolation,
and non-negativity constraints. We introduce the Concurrent ALS algorithm and
library, which offers an interface to Matlab, and a mechanism to effectively
deal with the issue that decompositions complete at different times.
Experimental results on artificial and real datasets demonstrate a shorter time
to completion due to increased arithmetic intensity.

    

### [[2010.05689] Continuous Safety Verification of Neural Networks](http://arxiv.org/abs/2010.05689)


  Deploying deep neural networks (DNNs) as core functions in autonomous driving
creates unique verification and validation challenges. In particular, the
continuous engineering paradigm of gradually perfecting a DNN-based perception
can make the previously established result of safety verification no longer
valid. This can occur either due to the newly encountered examples (i.e., input
domain enlargement) inside the Operational Design Domain or due to the
subsequent parameter fine-tuning activities of a DNN. This paper considers
approaches to transfer results established in the previous DNN safety
verification problem to the modified problem setting. By considering the reuse
of state abstractions, network abstractions, and Lipschitz constants, we
develop several sufficient conditions that only require formally analyzing a
small part of the DNN in the new problem. The overall concept is evaluated in a
$1/10$-scaled vehicle that equips a DNN controller to determine the visual
waypoint from the perceived image.

    

### [[2010.09313] BERTnesia: Investigating the capture and forgetting of knowledge in BERT](http://arxiv.org/abs/2010.09313)


  Probing complex language models has recently revealed several insights into
linguistic and semantic patterns found in the learned representations. In this
paper, we probe BERT specifically to understand and measure the relational
knowledge it captures. We utilize knowledge base completion tasks to probe
every layer of pre-trained as well as fine-tuned BERT (ranking, question
answering, NER). Our findings show that knowledge is not just contained in
BERT's final layers. Intermediate layers contribute a significant amount
(17-60%) to the total knowledge found. Probing intermediate layers also reveals
how different types of knowledge emerge at varying rates. When BERT is
fine-tuned, relational knowledge is forgotten but the extent of forgetting is
impacted by the fine-tuning objective but not the size of the dataset. We found
that ranking models forget the least and retain more knowledge in their final
layer. We release our code on github to repeat the experiments.

    

### [[2010.13933] Memorizing without overfitting: Bias, variance, and interpolation in over-parameterized models](http://arxiv.org/abs/2010.13933)


  The bias-variance trade-off is a central concept in supervised learning. In
classical statistics, increasing the complexity of a model (e.g., number of
parameters) reduces bias but also increases variance. Until recently, it was
commonly believed that optimal performance is achieved at intermediate model
complexities which strike a balance between bias and variance. Modern Deep
Learning methods flout this dogma, achieving state-of-the-art performance using
"over-parameterized models" where the number of fit parameters is large enough
to perfectly fit the training data. As a result, understanding bias and
variance in over-parameterized models has emerged as a fundamental problem in
machine learning. Here, we use methods from statistical physics to derive
analytic expressions for bias and variance in two minimal models of
over-parameterization (linear regression and two-layer neural networks with
nonlinear data distributions), allowing us to disentangle properties stemming
from the model architecture and random sampling of data. In both models,
increasing the number of fit parameters leads to a phase transition where the
training error goes to zero and the test error diverges as a result of the
variance (while the bias remains finite). Beyond this threshold in the
interpolation regime, the training error remains zero while the test error
decreases. We also show that in contrast with classical intuition,
over-parameterized models can overfit even in the absence of noise and exhibit
bias even if the student and teacher models match. We synthesize these results
to construct a holistic understanding of generalization error and the
bias-variance trade-off in over-parameterized models and relate our results to
random matrix theory.

    

### [[2011.07119] tvopt: A Python Framework for Time-Varying Optimization](http://arxiv.org/abs/2011.07119)


  This paper introduces tvopt, a Python framework for prototyping and
benchmarking time-varying (or online) optimization algorithms. The paper first
describes the theoretical approach that informed the development of tvopt. Then
it discusses the different components of the framework and their use for
modeling and solving time-varying optimization problems. In particular, tvopt
provides functionalities for defining both centralized and distributed online
problems, and a collection of built-in algorithms to solve them, for example
gradient-based methods, ADMM and other splitting methods. Moreover, the
framework implements prediction strategies to improve the accuracy of the
online solvers. The paper then proposes some numerical results on a benchmark
problem and discusses their implementation using tvopt. The code for tvopt is
available at this https URL.

    

### [[2011.14306] Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection Based on Reconstructability of Colors](http://arxiv.org/abs/2011.14306)


  This paper proposes an unsupervised anomaly detection technique for
image-based plant disease diagnosis. The construction of large and publicly
available datasets containing labeled images of healthy and diseased crop
plants led to growing interest in computer vision techniques for automatic
plant disease diagnosis. Although supervised image classifiers based on deep
learning can be a powerful tool for plant disease diagnosis, they require a
huge amount of labeled data. The data mining technique of anomaly detection
includes unsupervised approaches that do not require rare samples for training
classifiers. We propose an unsupervised anomaly detection technique for
image-based plant disease diagnosis that is based on the reconstructability of
colors; a deep encoder-decoder network trained to reconstruct the colors of
\textit{healthy} plant images should fail to reconstruct colors of symptomatic
regions. Our proposed method includes a new image-based framework for plant
disease detection that utilizes a conditional adversarial network called
pix2pix and a new anomaly score based on CIEDE2000 color difference.
Experiments with PlantVillage dataset demonstrated the superiority of our
proposed method compared to an existing anomaly detector at identifying
diseased crop images in terms of accuracy, interpretability and computational
efficiency.

    

### [[2012.03173] Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification](http://arxiv.org/abs/2012.03173)


  Deep AUC Maximization (DAM) is a new paradigm for learning a deep neural
network by maximizing the AUC score of the model on a dataset. Most previous
works of AUC maximization focus on the perspective of optimization by designing
efficient stochastic algorithms, and studies on generalization performance of
large-scale DAM on difficult tasks are missing. In this work, we aim to make
DAM more practical for interesting real-world applications (e.g., medical image
classification). First, we propose a new margin-based min-max surrogate loss
function for the AUC score (named as AUC min-max-margin loss or simply AUC
margin loss for short). It is more robust than the commonly used AUC square
loss, while enjoying the same advantage in terms of large-scale stochastic
optimization. Second, we conduct extensive empirical studies of our DAM method
on four difficult medical image classification tasks, namely (i) classification
of chest x-ray images for identifying many threatening diseases, (ii)
classification of images of skin lesions for identifying melanoma, (iii)
classification of mammogram for breast cancer screening, and (iv)
classification of microscopic images for identifying tumor tissue. Our studies
demonstrate that the proposed DAM method improves the performance of optimizing
cross-entropy loss by a large margin, and also achieves better performance than
optimizing the existing AUC square loss on these medical image classification
tasks. Specifically, our DAM method has achieved the 1st place on Stanford
CheXpert competition on Aug. 31, 2020. To the best of our knowledge, this is
the first work that makes DAM succeed on large-scale medical image datasets. We
also conduct extensive ablation studies to demonstrate the advantages of the
new AUC margin loss over the AUC square loss on benchmark datasets. The
proposed method is implemented in our open-sourced library LibAUC
(this http URL).

    

### [[2012.12803] Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling](http://arxiv.org/abs/2012.12803)


  Recent work of Erlingsson, Feldman, Mironov, Raghunathan, Talwar, and
Thakurta [EFMRTT19] demonstrates that random shuffling amplifies differential
privacy guarantees of locally randomized data. Such amplification implies
substantially stronger privacy guarantees for systems in which data is
contributed anonymously [BEMMRLRKTS17] and has lead to significant interest in
the shuffle model of privacy [CSUZZ19; EFMRTT19].
We show that random shuffling of $n$ data records that are input to
$\varepsilon_0$-differentially private local randomizers results in an
$(O((1-e^{-\varepsilon_0})\sqrt{\frac{e^{\varepsilon_0}\log(1/\delta)}{n}}),
\delta)$-differentially private algorithm. This significantly improves over
previous work and achieves the asymptotically optimal dependence in
$\varepsilon_0$. Our result is based on a new approach that is simpler than
previous work and extends to approximate differential privacy with nearly the
same guarantees. Importantly, our work also yields an algorithm for deriving
tighter bounds on the resulting $\varepsilon$ and $\delta$ as well as Rényi
differential privacy guarantees. We show numerically that our algorithm gets to
within a small constant factor of the optimal bound. As a direct corollary of
our analysis we derive a simple and nearly optimal algorithm for frequency
estimation in the shuffle model of privacy. We also observe that our result
implies the first asymptotically optimal privacy analysis of noisy stochastic
gradient descent that applies to sampling without replacement.

    

### [[2101.01169] Transformers in Vision: A Survey](http://arxiv.org/abs/2101.01169)


  Astounding results from Transformer models on natural language tasks have
intrigued the vision community to study their application to computer vision
problems. Among their salient benefits, Transformers enable modeling long
dependencies between input sequence elements and support parallel processing of
sequence as compared to recurrent networks e.g., Long short-term memory (LSTM).
Different from convolutional networks, Transformers require minimal inductive
biases for their design and are naturally suited as set-functions. Furthermore,
the straightforward design of Transformers allows processing multiple
modalities (e.g., images, videos, text and speech) using similar processing
blocks and demonstrates excellent scalability to very large capacity networks
and huge datasets. These strengths have led to exciting progress on a number of
vision tasks using Transformer networks. This survey aims to provide a
comprehensive overview of the Transformer models in the computer vision
discipline. We start with an introduction to fundamental concepts behind the
success of Transformers i.e., self-attention, large-scale pre-training, and
bidirectional encoding. We then cover extensive applications of transformers in
vision including popular recognition tasks (e.g., image classification, object
detection, action recognition, and segmentation), generative modeling,
multi-modal tasks (e.g., visual-question answering, visual reasoning, and
visual grounding), video processing (e.g., activity recognition, video
forecasting), low-level vision (e.g., image super-resolution, image
enhancement, and colorization) and 3D analysis (e.g., point cloud
classification and segmentation). We compare the respective advantages and
limitations of popular techniques both in terms of architectural design and
their experimental value. Finally, we provide an analysis on open research
directions and possible future works.

    

### [[2102.01203] Emergent Unfairness in Algorithmic Fairness-Accuracy Trade-Off Research](http://arxiv.org/abs/2102.01203)


  Across machine learning (ML) sub-disciplines, researchers make explicit
mathematical assumptions in order to facilitate proof-writing. We note that,
specifically in the area of fairness-accuracy trade-off optimization
scholarship, similar attention is not paid to the normative assumptions that
ground this approach. Such assumptions presume that 1) accuracy and fairness
are in inherent opposition to one another, 2) strict notions of mathematical
equality can adequately model fairness, 3) it is possible to measure the
accuracy and fairness of decisions independent from historical context, and 4)
collecting more data on marginalized individuals is a reasonable solution to
mitigate the effects of the trade-off. We argue that such assumptions, which
are often left implicit and unexamined, lead to inconsistent conclusions: While
the intended goal of this work may be to improve the fairness of machine
learning models, these unexamined, implicit assumptions can in fact result in
emergent unfairness. We conclude by suggesting a concrete path forward toward a
potential resolution.

    

### [[2102.02336] On the Approximation Power of Two-Layer Networks of Random ReLUs](http://arxiv.org/abs/2102.02336)


  This paper considers the following question: how well can depth-two ReLU
networks with randomly initialized bottom-level weights represent smooth
functions? We give near-matching upper- and lower-bounds for
$L_2$-approximation in terms of the Lipschitz constant, the desired accuracy,
and the dimension of the problem, as well as similar results in terms of
Sobolev norms. Our positive results employ tools from harmonic analysis and
ridgelet representation theory, while our lower-bounds are based on (robust
versions of) dimensionality arguments.

    

### [[2102.04635] Federated Deep AUC Maximization for Heterogeneous Data with a Constant Communication Complexity](http://arxiv.org/abs/2102.04635)


  Deep AUC (area under the ROC curve) Maximization (DAM) has attracted much
attention recently due to its great potential for imbalanced data
classification. However, the research on Federated Deep AUC Maximization (FDAM)
is still limited. Compared with standard federated learning (FL) approaches
that focus on decomposable minimization objectives, FDAM is more complicated
due to its minimization objective is non-decomposable over individual examples.
In this paper, we propose improved FDAM algorithms for heterogeneous data by
solving the popular non-convex strongly-concave min-max formulation of DAM in a
distributed fashion, which can also be applied to a class of non-convex
strongly-concave min-max problems. A striking result of this paper is that the
communication complexity of the proposed algorithm is a constant independent of
the number of machines and also independent of the accuracy level, which
improves an existing result by orders of magnitude. The experiments have
demonstrated the effectiveness of our FDAM algorithm on benchmark datasets, and
on medical chest X-ray images from different organizations. Our experiment
shows that the performance of FDAM using data from multiple hospitals can
improve the AUC score on testing data from a single hospital for detecting
life-threatening diseases based on chest radiographs.

    

### [[2102.13604] Federated Edge Learning with Misaligned Over-The-Air Computation](http://arxiv.org/abs/2102.13604)


  Over-the-air computation (OAC) is a promising technique to realize fast model
aggregation in the uplink of federated edge learning. OAC, however, hinges on
accurate channel-gain precoding and strict synchronization among the edge
devices, which are challenging in practice. As such, how to design the maximum
likelihood (ML) estimator in the presence of residual channel-gain mismatch and
asynchronies is an open problem. To fill this gap, this paper formulates the
problem of misaligned OAC for federated edge learning and puts forth a whitened
matched filtering and sampling scheme to obtain oversampled, but independent,
samples from the misaligned and overlapped signals. Given the whitened samples,
a sum-product ML estimator and an aligned-sample estimator are devised to
estimate the arithmetic sum of the transmitted symbols. In particular, the
computational complexity of our sum-product ML estimator is linear in the
packet length and hence is significantly lower than the conventional ML
estimator. Extensive simulations on the test accuracy versus the average
received energy per symbol to noise power spectral density ratio (EsN0) yield
two main results: 1) In the low EsN0 regime, the aligned-sample estimator can
achieve superior test accuracy provided that the phase misalignment is
non-severe. In contrast, the ML estimator does not work well due to the error
propagation and noise enhancement in the estimation process. 2) In the high
EsN0 regime, the ML estimator attains the optimal learning performance
regardless of the severity of phase misalignment. On the other hand, the
aligned-sample estimator suffers from a test-accuracy loss caused by phase
misalignment.

    

### [[2103.08160] DMN4: Few-shot Learning via Discriminative Mutual Nearest Neighbor Neural Network](http://arxiv.org/abs/2103.08160)


  Few-shot learning (FSL) aims to classify images under low-data regimes, where
the conventional pooled global feature is likely to lose useful local
characteristics. Recent work has achieved promising performances by using deep
descriptors. They generally take all deep descriptors from neural networks into
consideration while ignoring that some of them are useless in classification
due to their limited receptive field, e.g., task-irrelevant descriptors could
be misleading and multiple aggregative descriptors from background clutter
could even overwhelm the object's presence. In this paper, we argue that a
Mutual Nearest Neighbor (MNN) relation should be established to explicitly
select the query descriptors that are most relevant to each task and discard
less relevant ones from aggregative clutters in FSL. Specifically, we propose
Discriminative Mutual Nearest Neighbor Neural Network (DMN4) for FSL. Extensive
experiments demonstrate that our method outperforms the existing
state-of-the-arts on both fine-grained and generalized datasets.

    

### [[2103.14017] Scaling-up Disentanglement for Image Translation](http://arxiv.org/abs/2103.14017)


  Image translation methods typically aim to manipulate a set of labeled
attributes (given as supervision at training time e.g. domain label) while
leaving the unlabeled attributes intact. Current methods achieve either: (i)
disentanglement, which exhibits low visual fidelity and can only be satisfied
where the attributes are perfectly uncorrelated. (ii) visually-plausible
translations, which are clearly not disentangled. In this work, we propose
OverLORD, a single framework for disentangling labeled and unlabeled attributes
as well as synthesizing high-fidelity images, which is composed of two stages;
(i) Disentanglement: Learning disentangled representations with latent
optimization. Differently from previous approaches, we do not rely on
adversarial training or any architectural biases. (ii) Synthesis: Training
feed-forward encoders for inferring the learned attributes and tuning the
generator in an adversarial manner to increase the perceptual quality. When the
labeled and unlabeled attributes are correlated, we model an additional
representation that accounts for the correlated attributes and improves
disentanglement. We highlight that our flexible framework covers multiple
settings as disentangling labeled attributes, pose and appearance, localized
concepts, and shape and texture. We present significantly better
disentanglement with higher translation quality and greater output diversity
than state-of-the-art methods.

    

### [[2104.00606] Model Selection's Disparate Impact in Real-World Deep Learning Applications](http://arxiv.org/abs/2104.00606)


  Algorithmic fairness has emphasized the role of biased data in automated
decision outcomes. Recently, there has been a shift in attention to sources of
bias that implicate fairness in other stages in the ML pipeline. We contend
that one source of such bias, human preferences in model selection, remains
under-explored in terms of its role in disparate impact across demographic
groups. Using a deep learning model trained on real-world medical imaging data,
we verify our claim empirically and argue that choice of metric for model
comparison, especially those that do not take variability into account, can
significantly bias model selection outcomes.

    

### [[2104.00631] Residual Model Learning for Microrobot Control](http://arxiv.org/abs/2104.00631)


  A majority of microrobots are constructed using compliant materials that are
difficult to model analytically, limiting the utility of traditional
model-based controllers. Challenges in data collection on microrobots and large
errors between simulated models and real robots make current model-based
learning and sim-to-real transfer methods difficult to apply. We propose a
novel framework residual model learning (RML) that leverages approximate models
to substantially reduce the sample complexity associated with learning an
accurate robot model. We show that using RML, we can learn a model of the
Harvard Ambulatory MicroRobot (HAMR) using just 12 seconds of passively
collected interaction data. The learned model is accurate enough to be
leveraged as "proxy-simulator" for learning walking and turning behaviors using
model-free reinforcement learning algorithms. RML provides a general framework
for learning from extremely small amounts of interaction data, and our
experiments with HAMR clearly demonstrate that RML substantially outperforms
existing techniques.

    

### [[2104.01027] Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training](http://arxiv.org/abs/2104.01027)


  Self-supervised learning of speech representations has been a very active
research area but most work is focused on a single domain such as read audio
books for which there exist large quantities of labeled and unlabeled data. In
this paper, we explore more general setups where the domain of the unlabeled
data for pre-training data differs from the domain of the labeled data for
fine-tuning, which in turn may differ from the test data domain. Our
experiments show that using target domain data during pre-training leads to
large performance improvements across a variety of setups. On a large-scale
competitive setup, we show that pre-training on unlabeled in-domain data
reduces the gap between models trained on in-domain and out-of-domain labeled
data by 66%-73%. This has obvious practical implications since it is much
easier to obtain unlabeled target domain data than labeled data. Moreover, we
find that pre-training on multiple domains improves generalization performance
on domains not seen during training. Code and models will be made available at
this https URL.

    

### [[2104.01527] Distributed Reinforcement Learning for Age of Information Minimization in Real-Time IoT Systems](http://arxiv.org/abs/2104.01527)


  In this paper, the problem of minimizing the weighted sum of age of
information (AoI) and total energy consumption of Internet of Things (IoT)
devices is studied. In the considered model, each IoT device monitors a
physical process that follows nonlinear dynamics. As the dynamics of the
physical process vary over time, each device must find an optimal sampling
frequency to sample the real-time dynamics of the physical system and send
sampled information to a base station (BS). Due to limited wireless resources,
the BS can only select a subset of devices to transmit their sampled
information. Thus, edge devices must cooperatively sample their monitored
dynamics based on the local observations and the BS must collect the sampled
information from the devices immediately, hence avoiding the additional time
and energy used for sampling and information transmission. To this end, it is
necessary to jointly optimize the sampling policy of each device and the device
selection scheme of the BS so as to accurately monitor the dynamics of the
physical process using minimum energy. This problem is formulated as an
optimization problem whose goal is to minimize the weighted sum of AoI cost and
energy consumption. To solve this problem, we propose a novel distributed
reinforcement learning (RL) approach for the sampling policy optimization. The
proposed algorithm enables edge devices to cooperatively find the global
optimal sampling policy using their own local observations. Given the sampling
policy, the device selection scheme can be optimized thus minimizing the
weighted sum of AoI and energy consumption of all devices. Simulations with
real data of PM 2.5 pollution show that the proposed algorithm can reduce the
sum of AoI by up to 17.8% and 33.9% and the total energy consumption by up to
13.2% and 35.1%, compared to a conventional deep Q network method and a uniform
sampling policy.

    

### [[2104.07145] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks](http://arxiv.org/abs/2104.07145)


  Graph Neural Network (GNN) research is rapidly growing thanks to the capacity
of GNNs in learning distributed representations from graph-structured data.
However, centralizing a massive amount of real-world graph data for GNN
training is prohibitive due to privacy concerns, regulation restrictions, and
commercial competitions. Federated learning (FL), a trending distributed
learning paradigm, provides possibilities to solve this challenge while
preserving data privacy. Despite recent advances in vision and language
domains, there is no suitable platform for the FL of GNNs. To this end, we
introduce FedGraphNN, an open FL benchmark system that can facilitate research
on federated GNNs. FedGraphNN is built on a unified formulation of graph FL and
contains a wide range of datasets from different domains, popular GNN models,
and FL algorithms, with secure and efficient system support. Particularly for
the datasets, we collect, preprocess, and partition 36 datasets from 7 domains,
including both publicly available ones and specifically obtained ones such as
hERG and Tencent. Our empirical analysis showcases the utility of our benchmark
system, while exposing significant challenges in graph FL: federated GNNs
perform worse in most datasets with a non-IID split than centralized GNNs; the
GNN model that attains the best result in the centralized setting may not
maintain its advantage in the FL setting. These results imply that more
research efforts are needed to unravel the mystery behind federated GNNs.
Moreover, our system performance analysis demonstrates that the FedGraphNN
system is computationally efficient and secure to large-scale graphs datasets.
We maintain the source code at this https URL.

    

### [[2104.14756] Predicting Intraoperative Hypoxemia with Joint Sequence Autoencoder Networks](http://arxiv.org/abs/2104.14756)


  We present an end-to-end model using streaming physiological time series to
accurately predict near-term risk for hypoxemia, a rare, but life-threatening
condition known to cause serious patient harm during surgery. Our proposed
model makes inference on both hypoxemia outcomes and future input sequences,
enabled by a joint sequence autoencoder that simultaneously optimizes a
discriminative decoder for label prediction, and two auxiliary decoders trained
for data reconstruction and forecast, which seamlessly learns future-indicative
latent representation. All decoders share a memory-based encoder that helps
capture the global dynamics of patient data. In a large surgical cohort of
73,536 surgeries at a major academic medical center, our model outperforms all
baselines and gives a large performance gain over the state-of-the-art
hypoxemia prediction system. With a high sensitivity cutoff at 80%, it presents
99.36% precision in predicting hypoxemia and 86.81% precision in predicting the
much more severe and rare hypoxemic condition, persistent hypoxemia. With
exceptionally low rate of false alarms, our proposed model is promising in
improving clinical decision making and easing burden on the health system.

    

### [[2105.10277] Maximum and Leaky Maximum Propagation](http://arxiv.org/abs/2105.10277)


  In this work, we present an alternative to conventional residual connections,
which is inspired by maxout nets. This means that instead of the addition in
residual connections, our approach only propagates the maximum value or, in the
leaky formulation, propagates a percentage of both. In our evaluation, we show
on different public data sets that the presented approaches are comparable to
the residual connections and have other interesting properties, such as better
generalization with a constant batch normalization, faster learning, and also
the possibility to generalize without additional activation functions. In
addition, the proposed approaches work very well if ensembles together with
residual networks are formed.
this https URL


### [[2105.14074] Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning](http://arxiv.org/abs/2105.14074)


  In robotic domains, learning and planning are complicated by continuous state
spaces, continuous action spaces, and long task horizons. In this work, we
address these challenges with Neuro-Symbolic Relational Transition Models
(NSRTs), a novel class of models that are data-efficient to learn, compatible
with powerful robotic planning methods, and generalizable over objects. NSRTs
have both symbolic and neural components, enabling a bilevel planning scheme
where symbolic AI planning in an outer loop guides continuous planning with
neural models in an inner loop. Experiments in four robotic planning domains
show that NSRTs can be learned after only tens or hundreds of training
episodes, and then used for fast planning in new tasks that require up to 60
actions and involve many more objects than were seen during training. Video:
this https URL


### [[2105.14933] The use of Generative Adversarial Networks to characterise new physics in multi-lepton final states at the LHC](http://arxiv.org/abs/2105.14933)


  Semi-supervision in Machine Learning can be used in searches for new physics
where the signal plus background regions are not labelled. This strongly
reduces model dependency in the search for signals Beyond the Standard Model.
This approach displays the drawback in that over-fitting can give rise to fake
signals. Tossing toy Monte Carlo (MC) events can be used to estimate the
corresponding trials factor through a frequentist inference. However, MC events
that are based on full detector simulations are resource intensive. Generative
Adversarial Networks (GANs) can be used to mimic MC generators. GANs are
powerful generative models, but often suffer from training instability. We
henceforth show a review of GANs. We advocate the use of Wasserstein GAN (WGAN)
with weight clipping and WGAN with gradient penalty (WGAN-GP) where the norm of
gradient of the critic is penalized with respect to its input. Following the
emergence of multi-lepton anomalies at the LHC, we apply GANs for the
generation of di-leptons final states in association with b-quarks at the LHC.
A good agreement between the MC events and the WGAN-GP events is found for the
observables selected in the study.

    

### [[2106.00952] End-to-End Information Extraction by Character-Level Embedding and Multi-Stage Attentional U-Net](http://arxiv.org/abs/2106.00952)


  Information extraction from document images has received a lot of attention
recently, due to the need for digitizing a large volume of unstructured
documents such as invoices, receipts, bank transfers, etc. In this paper, we
propose a novel deep learning architecture for end-to-end information
extraction on the 2D character-grid embedding of the document, namely the
\textit{Multi-Stage Attentional U-Net}. To effectively capture the textual and
spatial relations between 2D elements, our model leverages a specialized
multi-stage encoder-decoders design, in conjunction with efficient uses of the
self-attention mechanism and the box convolution. Experimental results on
different datasets show that our model outperforms the baseline U-Net
architecture by a large margin while using 40\% fewer parameters. Moreover, it
also significantly improved the baseline in erroneous OCR and limited training
data scenario, thus becomes practical for real-world applications.

    

### [[2106.02810] An Attribute-Aligned Strategy for Learning Speech Representation](http://arxiv.org/abs/2106.02810)


  Advancement in speech technology has brought convenience to our life.
However, the concern is on the rise as speech signal contains multiple personal
attributes, which would lead to either sensitive information leakage or bias
toward decision. In this work, we propose an attribute-aligned learning
strategy to derive speech representation that can flexibly address these issues
by attribute-selection mechanism. Specifically, we propose a
layered-representation variational autoencoder (LR-VAE), which factorizes
speech representation into attribute-sensitive nodes, to derive an
identity-free representation for speech emotion recognition (SER), and an
emotionless representation for speaker verification (SV). Our proposed method
achieves competitive performances on identity-free SER and a better performance
on emotionless SV, comparing to the current state-of-the-art method of using
adversarial learning applied on a large emotion corpora, the MSP-Podcast. Also,
our proposed learning strategy reduces the model and training process needed to
achieve multiple privacy-preserving tasks.

    

### [[2106.07258] GitTables: A Large-Scale Corpus of Relational Tables](http://arxiv.org/abs/2106.07258)


  The practical success of deep learning has sparked interest in improving
relational table tasks, like data search, with models trained on large table
corpora. Existing corpora primarily contain tables extracted from HTML pages,
limiting the capability to represent offline database tables. To train and
evaluate high-capacity models for applications beyond the Web, we need
additional resources with tables that resemble relational database tables.
Here we introduce GitTables, a corpus of currently 1.7M relational tables
extracted from GitHub. Our continuing curation aims at growing the corpus to at
least 20M tables. We annotate table columns in GitTables with more than 2K
different semantic types from this http URL and DBpedia. Our column annotations
consist of semantic types, hierarchical relations, range types and
descriptions.
The corpus is available at this https URL. Our analysis of
GitTables shows that its structure, content, and topical coverage differ
significantly from existing table corpora. We evaluate our annotation pipeline
on hand-labeled tables from the T2Dv2 benchmark and find that our approach
provides results on par with human annotations. We demonstrate a use case of
GitTables by training a semantic type detection model on it and obtain high
prediction accuracy. We also show that the same model trained on tables from
theWeb generalizes poorly.

    

### [[2106.10698] Plant Disease Detection Using Image Processing and Machine Learning](http://arxiv.org/abs/2106.10698)


  One of the important and tedious task in agricultural practices is the
detection of the disease on crops. It requires huge time as well as skilled
labor. This paper proposes a smart and efficient technique for detection of
crop disease which uses computer vision and machine learning techniques. The
proposed system is able to detect 20 different diseases of 5 common plants with
93% accuracy.

    

### [[2109.02377] PermuteFormer: Efficient Relative Position Encoding for Long Sequences](http://arxiv.org/abs/2109.02377)


  A recent variation of Transformer, Performer, scales Transformer to longer
sequences with a linear attention mechanism. However, it is not compatible with
relative position encoding, which has advantages over absolute position
encoding. In this paper, we discuss possible ways to add relative position
encoding to Performer. Based on the analysis, we propose PermuteFormer, a
Performer-based model with relative position encoding that scales linearly on
long sequences. PermuteFormer applies position-dependent transformation on
queries and keys to encode positional information into the attention module.
This transformation is carefully crafted so that the final output of
self-attention is not affected by absolute positions of tokens. PermuteFormer
introduces negligible computational overhead by design that it runs as fast as
Performer. We evaluate PermuteFormer on Long-Range Arena, a dataset for long
sequences, as well as WikiText-103, a language modeling dataset. The
experiments show that PermuteFormer uniformly improves the performance of
Performer with almost no computational overhead and outperforms vanilla
Transformer on most of the tasks.

    

### [[2109.02442] Parkinson's Disease Diagnosis based on Gait Cycle Analysis Through an Interpretable Interval Type-2 Neuro-Fuzzy System](http://arxiv.org/abs/2109.02442)


  In this paper, an interpretable classifier using an interval type-2 fuzzy
neural network for detecting patients suffering from Parkinson's Disease (PD)
based on analyzing the gait cycle is presented. The proposed method utilizes
clinical features extracted from the vertical Ground Reaction Force (vGRF),
measured by 16 wearable sensors placed in the soles of subjects' shoes and
learns interpretable fuzzy rules. Therefore, experts can verify the decision
made by the proposed method based on investigating the firing strength of
interpretable fuzzy rules. Moreover, experts can utilize the extracted fuzzy
rules for patient diagnosing or adjust them based on their knowledge. To
improve the robustness of the proposed method against uncertainty and noisy
sensor measurements, Interval Type-2 Fuzzy Logic is applied. To learn fuzzy
rules, two paradigms are proposed: 1- A batch learning approach based on
clustering available samples is applied to extract initial fuzzy rules, 2- A
complementary online learning is proposed to improve the rule base encountering
new labeled samples. The performance of the method is evaluated for classifying
patients and healthy subjects in different conditions including the presence of
noise or observing new instances. Moreover, the performance of the model is
compared to some previous supervised and unsupervised machine learning
approaches. The final Accuracy, Precision, Recall, and F1 Score of the proposed
method are 88.74%, 89.41%, 95.10%, and 92.16%. Finally, the extracted fuzzy
sets for each feature are reported.

    

### [[2109.03276] Adaptive Computing in Robotics, Leveraging ROS 2 to Enable Software-Defined Hardware for FPGAs](http://arxiv.org/abs/2109.03276)


  Traditional software development in robotics is about programming
functionality in the CPU of a given robot with a pre-defined architecture and
constraints. With adaptive computing, instead, building a robotic behavior is
about programming an architecture. By leveraging adaptive computing,
roboticists can adapt one or more of the properties of its computing systems
(e.g. its determinism, power consumption, security posture, or throughput) at
run time. Roboticists are not, however, hardware engineers, and embedded
expertise is scarce among them. This white paper adopts a ROS 2
roboticist-centric view for adaptive computing and proposes an architecture to
include FPGAs as a first-class participant of the ROS 2 ecosystem. The
architecture proposed is platform- and technology-agnostic, and is easily
portable. The core components of the architecture are disclosed under an Apache
2.0 license, paving the way for roboticists to leverage adaptive computing and
create software-defined hardware.

    

### [[2109.03373] IceClave: A Trusted Execution Environment for In-Storage Computing](http://arxiv.org/abs/2109.03373)


  In-storage computing with modern solid-state drives (SSDs) enables developers
to offload programs from the host to the SSD. It has been proven to be an
effective approach to alleviate the I/O bottleneck. To facilitate in-storage
computing, many frameworks have been proposed. However, few of them treat the
in-storage security as the first citizen. Specifically, since modern SSD
controllers do not have a trusted execution environment, an offloaded
(malicious) program could steal, modify, and even destroy the data stored in
the SSD. In this paper, we first investigate the attacks that could be
conducted by offloaded in-storage programs. To defend against these attacks, we
build a lightweight trusted execution environment, named IceClave for
in-storage computing. IceClave enables security isolation between in-storage
programs and flash management functions that include flash address translation,
data access control, and garbage collection, with TrustZone extensions.
IceClave also achieves security isolation between in-storage programs by
enforcing memory integrity verification of in-storage DRAM with low overhead.
To protect data loaded from flash chips, IceClave develops a lightweight data
encryption/decryption mechanism in flash controllers. We develop IceClave with
a full system simulator. We evaluate IceClave with a variety of data-intensive
applications such as databases. Compared to state-of-the-art in-storage
computing approaches, IceClave introduces only 7.6% performance overhead, while
enforcing security isolation in the SSD controller with minimal hardware cost.
IceClave still keeps the performance benefit of in-storage computing by
delivering up to 2.31$\times$ better performance than the conventional
host-based trusted computing approach.

    

### [[2109.03389] An Optimal Resource Allocator of Elastic Training for Deep Learning Jobs on Cloud](http://arxiv.org/abs/2109.03389)


  Cloud training platforms, such as Amazon Web Services and Huawei Cloud
provide users with computational resources to train their deep learning jobs.
Elastic training is a service embedded in cloud training platforms that
dynamically scales up or down the resources allocated to a job. The core
technique of an elastic training system is to best allocate limited resources
among heterogeneous jobs in terms of shorter queueing delay and higher training
efficiency. This paper presents an optimal resource allocator for elastic
training system that leverages a mixed-integer programming (MIP) model to
maximize the training progress of deep learning jobs. We take advantage of the
real-world job data obtained from ModelArts, the deep learning training
platform of Huawei Cloud and conduct simulation experiments to compare the
optimal resource allocator with a greedy one as benchmark. Numerical results
show that the proposed allocator can reduce queuing time by up to 32% and
accelerate training efficiency by up to 24% relative to the greedy resource
allocator, thereby greatly improving user experience with Huawei ModelArts and
potentially enabling the realization of higher profits for the product. Also,
the optimal resource allocator is fast in decision-making, taking merely 0.4
seconds on average.

    

### [[2109.03592] Strong Scaling of OpenACC enabled Nek5000 on several GPU based HPC systems](http://arxiv.org/abs/2109.03592)


  We present new results on the strong parallel scaling for the
OpenACC-accelerated implementation of the high-order spectral element fluid
dynamics solver Nek5000. The test case considered consists of a direct
numerical simulation of fully-developed turbulent flow in a straight pipe, at
two different Reynolds numbers $Re_\tau=360$ and $Re_\tau=550$, based on
friction velocity and pipe radius. The strong scaling is tested on several
GPU-enabled HPC systems, including the Swiss Piz Daint system, TACC's Longhorn,
Jülich's JUWELS Booster, and Berzelius in Sweden. The performance results
show that speed-up between 3-5 can be achieved using the GPU accelerated
version compared with the CPU version on these different systems. The run-time
for 20 timesteps reduces from 43.5 to 13.2 seconds with increasing the number
of GPUs from 64 to 512 for $Re_\tau=550$ case on JUWELS Booster system. This
illustrates the GPU accelerated version the potential for high throughput. At
the same time, the strong scaling limit is significantly larger for GPUs, at
about $2000-5000$ elements per rank; compared to about $50-100$ for a CPU-rank.

    

### [[2109.03667] Energy Footprint of Blockchain Consensus Mechanisms Beyond Proof-of-Work](http://arxiv.org/abs/2109.03667)


  Second generation consensus mechanisms, such as Proof-of-Stake, promise to
provide more favourable energy consumption characteristics than those of their
predecessors, such as Proof-of-Work. In this paper, we quantify and compare the
energy demand of four archetypal modalities of second-generation systems:
Algorand, Ethereum 2.0, Hedera Hashgraph, and Polkadot. While numerous studies
that analyse the energy demands of individual distributed ledger systems have
been undertaken previously, little work has been done to compare different
systems that operate based on distinct technological assumptions. We approach
this research question by formalising a basic mathematical consumption model
for validatorbased Sybil attack resistance schemes. This model allows
quantifying the energy consumption per transaction based on common input
variables, such as the number of validators and the throughput characteristics
of the system analysed. We find that, when applying contemporary throughput and
validator counts, Hedera Hashgraph, by operating as a permissioned system, has
the most favourable energy consumption characteristics with 20.95 mW h/tx. This
stands in contrast to the permissionless systems Algorand with 4.427 W h/tx,
and Polkadot with 115.6 W h/tx. A very broad projection for Ethereum 2.0
suggests an energy consumption of 2.862 W h/tx to 557.5 W h/tx. The present
findings support the intuition that the complexity of Sybil attack resistance
mechanisms, and therefore the energy needs of the overarching consensus
protocols, is largely dependent on the number of active validators.
Consequently, a permissioned setting in which a can control the number of
validators can be beneficial to minimise energy consumption.

    

### [[2109.03739] A Dynamic, Hierarchical Resource Model for Converged Computing](http://arxiv.org/abs/2109.03739)


  Extreme dynamic heterogeneity in high performance computing systems and the
convergence of traditional HPC with new simulation, analysis, and data science
approaches impose increasingly more complex requirements on resource and job
management software (RJMS). However, there is a paucity of RJMS techniques that
can solve key technical challenges associated with those new requirements,
particularly when they are coupled. In this paper, we propose a novel dynamic
and multi-level resource model approach to address three key well-known
challenges individually and in combination: i.e., 1) RJMS dynamism to
facilitate job and workflow adaptability, 2) integration of specialized
external resources (e.g. user-centric cloud bursting), and 3) scheduling cloud
orchestration framework tasks. The core idea is to combine a dynamic directed
graph resource model with fully hierarchical scheduling to provide a unified
solution to all three key challenges. Our empirical and analytical evaluations
of the solution using our prototype extension to Fluxion, a production
hierarchical graph-based scheduler, suggest that our unified solution can
significantly improve flexibility, performance and scalability across all three
problems in comparison to limited traditional approaches.

    

### [[2104.10013] Parallel Physics-Informed Neural Networks via Domain Decomposition](http://arxiv.org/abs/2104.10013)


  We develop a distributed framework for the physics-informed neural networks
(PINNs) based on two recent extensions, namely conservative PINNs (cPINNs) and
extended PINNs (XPINNs), which employ domain decomposition in space and in
time-space, respectively. This domain decomposition endows cPINNs and XPINNs
with several advantages over the vanilla PINNs, such as parallelization
capacity, large representation capacity, efficient hyperparameter tuning, and
is particularly effective for multi-scale and multi-physics problems. Here, we
present a parallel algorithm for cPINNs and XPINNs constructed with a hybrid
programming model described by MPI $+$ X, where X $\in
\{\text{CPUs},~\text{GPUs}\}$. The main advantage of cPINN and XPINN over the
more classical data and model parallel approaches is the flexibility of
optimizing all hyperparameters of each neural network separately in each
subdomain. We compare the performance of distributed cPINNs and XPINNs for
various forward problems, using both weak and strong scalings. Our results
indicate that for space domain decomposition, cPINNs are more efficient in
terms of communication cost but XPINNs provide greater flexibility as they can
also handle time-domain decomposition for any differential equations, and can
deal with any arbitrarily shaped complex subdomains. To this end, we also
present an application of the parallel XPINN method for solving an inverse
diffusion problem with variable conductivity on the United States map, using
ten regions as subdomains.

    

### [[2109.03283] Have a break from making decisions, have a MARS: The Multi-valued Action Reasoning System](http://arxiv.org/abs/2109.03283)


  The Multi-valued Action Reasoning System (MARS) is an automated value-based
ethical decision-making model for artificial agents (AI). Given a set of
available actions and an underlying moral paradigm, by employing MARS one can
identify the ethically preferred action. It can be used to implement and model
different ethical theories, different moral paradigms, as well as combinations
of such, in the context of automated practical reasoning and normative decision
analysis. It can also be used to model moral dilemmas and discover the moral
paradigms that result in the desired outcomes therein. In this paper, we give a
condensed description of MARS, explain its uses, and comparatively place it in
the existing literature.

    

### [[2109.03310] Melatect: A Machine Learning Model Approach For Identifying Malignant Melanoma in Skin Growths](http://arxiv.org/abs/2109.03310)


  Malignant melanoma is a common skin cancer that is mostly curable before
metastasis, where melanoma growths spawn in organs away from the original site.
Melanoma is the most dangerous type of skin cancer if left untreated due to the
high chance of metastasis. This paper presents Melatect, a machine learning
model that identifies potential malignant melanoma. A recursive computer image
analysis algorithm was used to create a machine learning model which is capable
of detecting likely melanoma. The comparison is performed using 20,000 raw
images of benign and malignant lesions from the International Skin Imaging
Collaboration (ISIC) archive that were augmented to 60,000 images. Tests of the
algorithm using subsets of the ISIC images suggest it accurately classifies
lesions as malignant or benign over 95% of the time with no apparent bias or
overfitting. The Melatect iOS app was later created (unpublished), in which the
machine learning model was embedded. With the app, users have the ability to
take pictures of skin lesions (moles) using the app, which are then processed
through the machine learning model, and users are notified whether their lesion
could be abnormal or not. Melatect provides a convenient way to get free advice
on lesions and track these lesions over time.

    

### [[2109.03334] On the Challenges of Evaluating Compositional Explanations in Multi-Hop Inference: Relevance, Completeness, and Expert Ratings](http://arxiv.org/abs/2109.03334)


  Building compositional explanations requires models to combine two or more
facts that, together, describe why the answer to a question is correct.
Typically, these "multi-hop" explanations are evaluated relative to one (or a
small number of) gold explanations. In this work, we show these evaluations
substantially underestimate model performance, both in terms of the relevance
of included facts, as well as the completeness of model-generated explanations,
because models regularly discover and produce valid explanations that are
different than gold explanations. To address this, we construct a large corpus
of 126k domain-expert (science teacher) relevance ratings that augment a corpus
of explanations to standardized science exam questions, discovering 80k
additional relevant facts not rated as gold. We build three strong models based
on different methodologies (generation, ranking, and schemas), and empirically
show that while expert-augmented ratings provide better estimates of
explanation quality, both original (gold) and expert-augmented automatic
evaluations still substantially underestimate performance by up to 36% when
compared with full manual expert judgements, with different models being
disproportionately affected. This poses a significant methodological challenge
to accurately evaluating explanations produced by compositional reasoning
models.

    

### [[2109.03341] Software Vulnerability Detection via Deep Learning over Disaggregated Code Graph Representation](http://arxiv.org/abs/2109.03341)


  Identifying vulnerable code is a precautionary measure to counter software
security breaches. Tedious expert effort has been spent to build static
analyzers, yet insecure patterns are barely fully enumerated. This work
explores a deep learning approach to automatically learn the insecure patterns
from code corpora. Because code naturally admits graph structures with parsing,
we develop a novel graph neural network (GNN) to exploit both the semantic
context and structural regularity of a program, in order to improve prediction
performance. Compared with a generic GNN, our enhancements include a synthesis
of multiple representations learned from the several parsed graphs of a
program, and a new training loss metric that leverages the fine granularity of
labeling. Our model outperforms multiple text, image and graph-based
approaches, across two real-world datasets.

    

### [[2109.03372] Identifying Influential Nodes in Two-mode Data Networks using Formal Concept Analysis](http://arxiv.org/abs/2109.03372)


  Identifying important actors (or nodes) in a two-mode network often remains a
crucial challenge in mining, analyzing, and interpreting real-world networks.
While traditional bipartite centrality indices are often used to recognize key
nodes that influence the network information flow, they frequently produce poor
results in intricate situations such as massive networks with complex local
structures or a lack of complete knowledge about the network topology and
certain properties. In this paper, we introduce Bi-face (BF), a new bipartite
centrality measurement for identifying important nodes in two-mode networks.
Using the powerful mathematical formalism of Formal Concept Analysis, the BF
measure exploits the faces of concept intents to identify nodes that have
influential bicliques connectivity and are not located in irrelevant bridges.
Unlike off-the shelf centrality indices, it quantifies how a node has a
cohesive-substructure influence on its neighbour nodes via bicliques while not
being in network core-peripheral ones through its absence from non-influential
bridges. Our experiments on several real-world and synthetic networks show the
efficiency of BF over existing prominent bipartite centrality measures such as
betweenness, closeness, eigenvector, and vote-rank among others.

    

### [[2109.03375] Malware Squid: A Novel IoT Malware Traffic Analysis Framework using Convolutional Neural Network and Binary Visualisation](http://arxiv.org/abs/2109.03375)


  Internet of Things devices have seen a rapid growth and popularity in recent
years with many more ordinary devices gaining network capability and becoming
part of the ever growing IoT network. With this exponential growth and the
limitation of resources, it is becoming increasingly harder to protect against
security threats such as malware due to its evolving faster than the defence
mechanisms can handle with. The traditional security systems are not able to
detect unknown malware as they use signature-based methods. In this paper, we
aim to address this issue by introducing a novel IoT malware traffic analysis
approach using neural network and binary visualisation. The prime motivation of
the proposed approach is to faster detect and classify new malware (zero-day
malware). The experiment results show that our method can satisfy the accuracy
requirement of practical application.

    

### [[2109.03383] DeepZensols: Deep Natural Language Processing Framework](http://arxiv.org/abs/2109.03383)


  Reproducing results in publications by distributing publicly available source
code is becoming ever more popular. Given the difficulty of reproducing machine
learning (ML) experiments, there have been significant efforts in reducing the
variance of these results. As in any science, the ability to consistently
reproduce results effectively strengthens the underlying hypothesis of the
work, and thus, should be regarded as important as the novel aspect of the
research itself. The contribution of this work is a framework that is able to
reproduce consistent results and provides a means of easily creating, training,
and evaluating natural language processing (NLP) deep learning (DL) models.

    

### [[2109.03391] Visual Sensation and Perception Computational Models for Deep Learning: State of the art, Challenges and Prospects](http://arxiv.org/abs/2109.03391)


  Visual sensation and perception refers to the process of sensing, organizing,
identifying, and interpreting visual information in environmental awareness and
understanding. Computational models inspired by visual perception have the
characteristics of complexity and diversity, as they come from many subjects
such as cognition science, information science, and artificial intelligence. In
this paper, visual perception computational models oriented deep learning are
investigated from the biological visual mechanism and computational vision
theory systematically. Then, some points of view about the prospects of the
visual perception computational models are presented. Finally, this paper also
summarizes the current challenges of visual perception and predicts its future
development trends. Through this survey, it will provide a comprehensive
reference for research in this direction.

    

### [[2109.03423] It is AI's Turn to Ask Human a Question: Question and Answer Pair Generation for Children Storybooks in FairytaleQA Dataset](http://arxiv.org/abs/2109.03423)


  Existing question answering (QA) datasets are created mainly for the
application of having AI to be able to answer questions asked by humans. But in
educational applications, teachers and parents sometimes may not know what
questions they should ask a child that can maximize their language learning
results. With a newly released book QA dataset (FairytaleQA), which educational
experts labeled on 46 fairytale storybooks for early childhood readers, we
developed an automated QA generation model architecture for this novel
application. Our model (1) extracts candidate answers from a given storybook
passage through carefully designed heuristics based on a pedagogical framework;
(2) generates appropriate questions corresponding to each extracted answer
using a language model; and, (3) uses another QA model to rank top QA-pairs.
Automatic and human evaluations show that our model outperforms baselines. We
also demonstrate that our method can help with the scarcity issue of the
children's book QA dataset via data augmentation on 200 unlabeled storybooks.

    

### [[2109.03438] ArchivalQA: A Large-scale Benchmark Dataset for Open Domain Question Answering over Archival News Collections](http://arxiv.org/abs/2109.03438)


  In the last few years, open-domain question answering (ODQA) has advanced
rapidly due to the development of deep learning techniques and the availability
of large-scale QA datasets. However, the current datasets are essentially
designed for synchronic document collections (e.g., Wikipedia). Temporal news
collections such as long-term news archives spanning several decades, are
rarely used in training the models despite they are quite valuable for our
society. In order to foster the research in the field of ODQA on such
historical collections, we present ArchivalQA, a large question answering
dataset consisting of 1,067,056 question-answer pairs which is designed for
temporal news QA. In addition, we create four subparts of our dataset based on
the question difficulty levels and the containment of temporal expressions,
which we believe could be useful for training or testing ODQA systems
characterized by different strengths and abilities. The novel QA
dataset-constructing framework that we introduce can be also applied to create
datasets over other types of collections.

    

### [[2109.03540] A Survey of Deep Reinforcement Learning in Recommender Systems: A Systematic Review and Future Directions](http://arxiv.org/abs/2109.03540)


  In light of the emergence of deep reinforcement learning (DRL) in recommender
systems research and several fruitful results in recent years, this survey aims
to provide a timely and comprehensive overview of the recent trends of deep
reinforcement learning in recommender systems. We start with the motivation of
applying DRL in recommender systems. Then, we provide a taxonomy of current
DRL-based recommender systems and a summary of existing methods. We discuss
emerging topics and open issues, and provide our perspective on advancing the
domain. This survey serves as introductory material for readers from academia
and industry into the topic and identifies notable opportunities for further
research.

    

### [[2109.03554] Do What Nature Did To Us: Evolving Plastic Recurrent Neural Networks For Task Generalization](http://arxiv.org/abs/2109.03554)


  While artificial neural networks (ANNs) have been widely adopted in machine
learning, researchers are increasingly obsessed by the gaps between ANNs and
biological neural networks (BNNs). In this paper, we propose a framework named
as Evolutionary Plastic Recurrent Neural Networks} (EPRNN). Inspired by BNN,
EPRNN composes Evolution Strategies, Plasticity Rules, and Recursion-based
Learning all in one meta learning framework for generalization to different
tasks. More specifically, EPRNN incorporates with nested loops for meta
learning -- an outer loop searches for optimal initial parameters of the neural
network and learning rules; an inner loop adapts to specific tasks. In the
inner loop of EPRNN, we effectively attain both long term memory and short term
memory by forging plasticity with recursion-based learning mechanisms, both of
which are believed to be responsible for memristance in BNNs. The inner-loop
setting closely simulate that of BNNs, which neither query from any gradient
oracle for optimization nor require the exact forms of learning objectives. To
evaluate the performance of EPRNN, we carry out extensive experiments in two
groups of tasks: Sequence Predicting, and Wheeled Robot Navigating. The
experiment results demonstrate the unique advantage of EPRNN compared to
state-of-the-arts based on plasticity and recursion while yielding comparably
good performance against deep learning based approaches in the tasks. The
experiment results suggest the potential of EPRNN to generalize to variety of
tasks and encourage more efforts in plasticity and recursion based learning
mechanisms.

    

### [[2109.03564] NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task--Next Sentence Prediction](http://arxiv.org/abs/2109.03564)


  Using prompts to utilize language models to perform various downstream tasks,
also known as prompt-based learning or prompt-learning, has lately gained
significant success in comparison to the pre-train and fine-tune paradigm.
Nonetheless, virtually all prompt-based methods are token-level, meaning they
all utilize GPT's left-to-right language model or BERT's masked language model
to perform cloze-style tasks. In this paper, we attempt to accomplish several
NLP tasks in the zero-shot scenario using a BERT original pre-training task
abandoned by RoBERTa and other models--Next Sentence Prediction (NSP). Unlike
token-level techniques, our sentence-level prompt-based method NSP-BERT does
not need to fix the length of the prompt or the position to be predicted,
allowing it to handle tasks such as entity linking with ease. Based on the
characteristics of NSP-BERT, we offer several quick building templates for
various downstream tasks. We suggest a two-stage prompt method for word sense
disambiguation tasks in particular. Our strategies for mapping the labels
significantly enhance the model's performance on sentence pair tasks. On the
FewCLUE benchmark, our NSP-BERT outperforms other zero-shot methods on most of
these tasks and comes close to the few-shot methods.

    

### [[2109.03569] LiDARTouch: Monocular metric depth estimation with a few-beam LiDAR](http://arxiv.org/abs/2109.03569)


  Vision-based depth estimation is a key feature in autonomous systems, which
often relies on a single camera or several independent ones. In such a
monocular setup, dense depth is obtained with either additional input from one
or several expensive LiDARs, e.g., with 64 beams, or camera-only methods, which
suffer from scale-ambiguity and infinite-depth problems. In this paper, we
propose a new alternative of densely estimating metric depth by combining a
monocular camera with a light-weight LiDAR, e.g., with 4 beams, typical of
today's automotive-grade mass-produced laser scanners. Inspired by recent
self-supervised methods, we introduce a novel framework, called LiDARTouch, to
estimate dense depth maps from monocular images with the help of ``touches'' of
LiDAR, i.e., without the need for dense ground-truth depth. In our setup, the
minimal LiDAR input contributes on three different levels: as an additional
model's input, in a self-supervised LiDAR reconstruction objective function,
and to estimate changes of pose (a key component of self-supervised depth
estimation architectures). Our LiDARTouch framework achieves new state of the
art in self-supervised depth estimation on the KITTI dataset, thus supporting
our choices of integrating the very sparse LiDAR signal with other visual
features. Moreover, we show that the use of a few-beam LiDAR alleviates scale
ambiguity and infinite-depth issues that camera-only methods suffer from. We
also demonstrate that methods from the fully-supervised depth-completion
literature can be adapted to a self-supervised regime with a minimal LiDAR
signal.

    

### [[2109.03695] Continuous Entailment Patterns for Lexical Inference in Context](http://arxiv.org/abs/2109.03695)


  Combining a pretrained language model (PLM) with textual patterns has been
shown to help in both zero- and few-shot settings. For zero-shot performance,
it makes sense to design patterns that closely resemble the text seen during
self-supervised pretraining because the model has never seen anything else.
Supervised training allows for more flexibility. If we allow for tokens outside
the PLM's vocabulary, patterns can be adapted more flexibly to a PLM's
idiosyncrasies. Contrasting patterns where a "token" can be any continuous
vector vs. those where a discrete choice between vocabulary elements has to be
made, we call our method CONtinuous pAtterNs (CONAN). We evaluate CONAN on two
established benchmarks for lexical inference in context (LIiC) a.k.a. predicate
entailment, a challenging natural language understanding task with relatively
small training sets. In a direct comparison with discrete patterns, CONAN
consistently leads to improved performance, setting a new state of the art. Our
experiments give valuable insights into the kind of pattern that enhances a
PLM's performance on LIiC and raise important questions regarding our
understanding of PLMs using text patterns.

    

### [[2109.03710] BotSpot: Deep Learning Classification of Bot Accounts within Twitter](http://arxiv.org/abs/2109.03710)


  The openness feature of Twitter allows programs to generate and control
Twitter accounts automatically via the Twitter API. These accounts, which are
known as bots, can automatically perform actions such as tweeting, re-tweeting,
following, unfollowing, or direct messaging other accounts, just like real
people. They can also conduct malicious tasks such as spreading of fake news,
spams, malicious software and other cyber-crimes. In this paper, we introduce a
novel bot detection approach using deep learning, with the Multi-layer
Perceptron Neural Networks and nine features of a bot account. A web crawler is
developed to automatically collect data from public Twitter accounts and build
the testing and training datasets, with 860 samples of human and bot accounts.
After the initial training is done, the Multilayer Perceptron Neural Networks
achieved an overall accuracy rate of 92%, which proves the performance of the
proposed approach.

    

### [[2109.03721] Conjectures, Tests and Proofs: An Overview of Theory Exploration](http://arxiv.org/abs/2109.03721)


  A key component of mathematical reasoning is the ability to formulate
interesting conjectures about a problem domain at hand. In this paper, we give
a brief overview of a theory exploration system called QuickSpec, which is able
to automatically discover interesting conjectures about a given set of
functions. QuickSpec works by interleaving term generation with random testing
to form candidate conjectures. This is made tractable by starting from small
sizes and ensuring that only terms that are irreducible with respect to already
discovered conjectures are considered. QuickSpec has been successfully applied
to generate lemmas for automated inductive theorem proving as well as to
generate specifications of functional programs. We give an overview of typical
use-cases of QuickSpec, as well as demonstrating how to easily connect it to a
theorem prover of the user's choice.

    

### [[2109.03754] Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories](http://arxiv.org/abs/2109.03754)


  Measuring event salience is essential in the understanding of stories. This
paper takes a recent unsupervised method for salience detection derived from
Barthes Cardinal Functions and theories of surprise and applies it to longer
narrative forms. We improve the standard transformer language model by
incorporating an external knowledgebase (derived from Retrieval Augmented
Generation) and adding a memory mechanism to enhance performance on longer
works. We use a novel approach to derive salience annotation using
chapter-aligned summaries from the Shmoop corpus for classic literary works.
Our evaluation against this data demonstrates that our salience detection model
improves performance over and above a non-knowledgebase and memory augmented
language model, both of which are crucial to this improvement.

    

### [[2109.03792] Highly Parallel Autoregressive Entity Linking with Discriminative Correction](http://arxiv.org/abs/2109.03792)


  Generative approaches have been recently shown to be effective for both
Entity Disambiguation and Entity Linking (i.e., joint mention detection and
disambiguation). However, the previously proposed autoregressive formulation
for EL suffers from i) high computational cost due to a complex (deep) decoder,
ii) non-parallelizable decoding that scales with the source sequence length,
and iii) the need for training on a large amount of data. In this work, we
propose a very efficient approach that parallelizes autoregressive linking
across all potential mentions and relies on a shallow and efficient decoder.
Moreover, we augment the generative objective with an extra discriminative
component, i.e., a correction term which lets us directly optimize the
generator's ranking. When taken together, these techniques tackle all the above
issues: our model is >70 times faster and more accurate than the previous
generative method, outperforming state-of-the-art approaches on the standard
English dataset AIDA-CoNLL. Source code available at
this https URL


### [[2109.03806] Exploration of Quantum Neural Architecture by Mixing Quantum Neuron Designs](http://arxiv.org/abs/2109.03806)


  With the constant increase of the number of quantum bits (qubits) in the
actual quantum computers, implementing and accelerating the prevalent deep
learning on quantum computers are becoming possible. Along with this trend,
there emerge quantum neural architectures based on different designs of quantum
neurons. A fundamental question in quantum deep learning arises: what is the
best quantum neural architecture? Inspired by the design of neural
architectures for classical computing which typically employs multiple types of
neurons, this paper makes the very first attempt to mix quantum neuron designs
to build quantum neural architectures. We observe that the existing quantum
neuron designs may be quite different but complementary, such as neurons from
variation quantum circuits (VQC) and Quantumflow. More specifically, VQC can
apply real-valued weights but suffer from being extended to multiple layers,
while QuantumFlow can build a multi-layer network efficiently, but is limited
to use binary weights. To take their respective advantages, we propose to mix
them together and figure out a way to connect them seamlessly without
additional costly measurement. We further investigate the design principles to
mix quantum neurons, which can provide guidance for quantum neural architecture
exploration in the future. Experimental results demonstrate that the identified
quantum neural architectures with mixed quantum neurons can achieve 90.62% of
accuracy on the MNIST dataset, compared with 52.77% and 69.92% on the VQC and
QuantumFlow, respectively.

    

### [[2109.03813] Video2Skill: Adapting Events in Demonstration Videos to Skills in an Environment using Cyclic MDP Homomorphisms](http://arxiv.org/abs/2109.03813)


  Humans excel at learning long-horizon tasks from demonstrations augmented
with textual commentary, as evidenced by the burgeoning popularity of tutorial
videos online. Intuitively, this capability can be separated into 2 distinct
subtasks - first, dividing a long-horizon demonstration sequence into
semantically meaningful events; second, adapting such events into meaningful
behaviors in one's own environment. Here, we present Video2Skill (V2S), which
attempts to extend this capability to artificial agents by allowing a robot arm
to learn from human cooking videos. We first use sequence-to-sequence
Auto-Encoder style architectures to learn a temporal latent space for events in
long-horizon demonstrations. We then transfer these representations to the
robotic target domain, using a small amount of offline and unrelated
interaction data (sequences of state-action pairs of the robot arm controlled
by an expert) to adapt these events into actionable representations, i.e.,
skills. Through experiments, we demonstrate that our approach results in
self-supervised analogy learning, where the agent learns to draw analogies
between motions in human demonstration data and behaviors in the robotic
environment. We also demonstrate the efficacy of our approach on model learning
- demonstrating how Video2Skill utilizes prior knowledge from human
demonstration to outperform traditional model learning of long-horizon
dynamics. Finally, we demonstrate the utility of our approach for non-tabula
rasa decision-making, i.e, utilizing video demonstration for zero-shot skill
generation.

    

### [[1909.06673] Propagation complete encodings of smooth DNNF theories](http://arxiv.org/abs/1909.06673)


  We investigate conjunctive normal form (CNF) encodings of a function
represented with a decomposable negation normal form (DNNF). Several encodings
of DNNFs and decision diagrams were considered by (Abio et al. 2016). The
authors differentiate between encodings which implement consistency or domain
consistency by unit propagation from encodings which are unit refutation
complete or propagation complete. The difference is that in the former case we
do not care about propagation strength of the encoding with respect to the
auxiliary variables while in the latter case we treat all variables (the main
and the auxiliary ones) in the same way. The currently known encodings of DNNF
theories implement domain consistency. Building on these encodings we
generalize the result of (Abio et al. 2016) on a propagation complete encoding
of decision diagrams and present a propagation complete encoding of a DNNF and
its generalization for variables with finite domains.

    

### [[2008.03641] NMR Assignment through Linear Programming](http://arxiv.org/abs/2008.03641)


  Nuclear Magnetic Resonance (NMR) Spectroscopy is the second most used
technique (after X-ray crystallography) for structural determination of
proteins. A computational challenge in this technique involves solving a
discrete optimization problem that assigns the resonance frequency to each atom
in the protein. This paper introduces LIAN (LInear programming Assignment for
NMR), a novel linear programming formulation of the problem which yields
state-of-the-art results in simulated and experimental datasets.

    

### [[2104.08663] BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](http://arxiv.org/abs/2104.08663)


  Existing neural information retrieval (IR) models have often been studied in
homogeneous and narrow settings, which has considerably limited insights into
their out-of-distribution (OOD) generalization capabilities. To address this,
and to facilitate researchers to broadly evaluate the effectiveness of their
models, we introduce Benchmarking-IR (BEIR), a robust and heterogeneous
evaluation benchmark for information retrieval. We leverage a careful selection
of 18 publicly available datasets from diverse text retrieval tasks and domains
and evaluate 10 state-of-the-art retrieval systems including lexical, sparse,
dense, late-interaction and re-ranking architectures on the BEIR benchmark. Our
results show BM25 is a robust baseline and re-ranking and
late-interaction-based models on average achieve the best zero-shot
performances, however, at high computational costs. In contrast, dense and
sparse-retrieval models are computationally more efficient but often
underperform other approaches, highlighting the considerable room for
improvement in their generalization capabilities. We hope this framework allows
us to better evaluate and understand existing retrieval systems, and
contributes to accelerating progress towards better robust and generalizable
systems in the future. BEIR is publicly available at
this https URL.

    

### [[2109.02614] The Animation Transformer: Visual Correspondence via Segment Matching](http://arxiv.org/abs/2109.02614)


  Visual correspondence is a fundamental building block on the way to building
assistive tools for hand-drawn animation. However, while a large body of work
has focused on learning visual correspondences at the pixel-level, few
approaches have emerged to learn correspondence at the level of line enclosures
(segments) that naturally occur in hand-drawn animation. Exploiting this
structure in animation has numerous benefits: it avoids the intractable memory
complexity of attending to individual pixels in high resolution images and
enables the use of real-world animation datasets that contain correspondence
information at the level of per-segment colors. To that end, we propose the
Animation Transformer (AnT) which uses a transformer-based architecture to
learn the spatial and visual relationships between segments across a sequence
of images. AnT enables practical ML-assisted colorization for professional
animation workflows and is publicly accessible as a creative tool in Cadmium.

    

### [[2109.01163] Efficient conformer: Progressive downsampling and grouped attention for automatic speech recognition](http://arxiv.org/abs/2109.01163)


  The recently proposed Conformer architecture has shown state-of-the-art
performances in Automatic Speech Recognition by combining convolution with
attention to model both local and global dependencies. In this paper, we study
how to reduce the Conformer architecture complexity with a limited computing
budget, leading to a more efficient architecture design that we call Efficient
Conformer. We introduce progressive downsampling to the Conformer encoder and
propose a novel attention mechanism named grouped attention, allowing us to
reduce attention complexity from $O(n^{2}d)$ to $O(n^{2}d / g)$ for sequence
length $n$, hidden dimension $d$ and group size parameter $g$. We also
experiment the use of strided multi-head self-attention as a global
downsampling operation. Our experiments are performed on the LibriSpeech
dataset with CTC and RNN-Transducer losses. We show that within the same
computing budget, the proposed architecture achieves better performances with
faster training and decoding compared to the Conformer. Our 13M parameters CTC
model achieves competitive WERs of 3.6%/9.0% without using a language model and
2.7%/6.7% with an external n-gram language model on the test-clean/test-other
sets while being 29% faster than our CTC Conformer baseline at inference and
36% faster to train.

    

### [[2109.03338] $\mathcal{N}$IPM-MPC: An Efficient Null-Space Method Based Interior-Point Method for Model Predictive Control](http://arxiv.org/abs/2109.03338)


  Linear Model Predictive Control (MPC) is a widely used method to control
systems with linear dynamics. Efficient interior-point methods have been
proposed which leverage the block diagonal structure of the quadratic program
(QP) resulting from the receding horizon control formulation. However, they
require two matrix factorizations per interior-point method iteration, one each
for the computation of the dual and the primal. Recently though an interior
point method based on the null-space method has been proposed which requires
only a single decomposition per iteration. While the then used null-space basis
leads to dense null-space projections, in this work we propose a sparse
null-space basis which preserves the block diagonal structure of the MPC
matrices. Since it is based on the inverse of the transfer matrix we introduce
the notion of so-called virtual controls which enables just that invertibility.
A combination of the reduced number of factorizations and omission of the
evaluation of the dual lets our solver outperform others in terms of
computational speed by an increasing margin dependent on the number of state
and control variables.

    

### [[2109.03602] SecRSL: Security Separation Logic for C11 Release-Acquire Concurrency (Extended version with technical appendices)](http://arxiv.org/abs/2109.03602)


  We present Security Relaxed Separation Logic (SecRSL), a separation logic for
proving information-flow security of C11 programs in the Release-Acquire
fragment with relaxed accesses. SecRSL is the first security logic that (1)
supports weak-memory reasoning about programs in a high-level language; (2)
inherits separation logic's virtues of compositional, local reasoning about (3)
expressive security policies like value-dependent classification.
SecRSL is also, to our knowledge, the first security logic developed over an
axiomatic memory model. Thus we also present the first definitions of
information-flow security for an axiomatic weak memory model, against which we
prove SecRSL sound. SecRSL ensures that programs satisfy a constant-time
security guarantee, while being free of undefined behaviour.
We apply SecRSL to implement and verify the functional correctness and
constant-time security of a range of concurrency primitives, including a
spinlock module, a mixed-sensitivity mutex, and multiple synchronous channel
implementations. Empirical performance evaluations of the latter demonstrate
SecRSL's power to support the development of secure and performant concurrent C
programs.

    

### [[2104.05558] A meta-theory for big-step semantics](http://arxiv.org/abs/2104.05558)


  It is well-known that big-step semantics is not able to distinguish stuck and
non-terminating computations. This is a strong limitation as it makes very
difficult to reason about properties involving infinite computations, such as
type soundness, which cannot even be expressed. We show that this issue is only
apparent: the distinction between stuck and diverging computations is implicit
in any big-step semantics and it just needs to be uncovered. To achieve this
goal, we develop a systematic study of big-step semantics: we introduce an
abstract definition of what a big-step semantics is, we define a notion of
computation by formalising the evaluation algorithm implicitly associated with
any big-step semantics, and we show how to canonically extend a big-step
semantics to characterise stuck and diverging computations. Building on these
notions, we describe a general proof technique to show that a predicate is
sound, that is, it prevents stuck computation, with respect to a big-step
semantics. One needs to check three properties relating the predicate and the
semantics and, if they hold, the predicate is sound. The extended semantics are
essential to establish this meta-logical result, but are of no concerns to the
user, who only needs to prove the three properties of the initial big-step
semantics. Finally, we illustrate the technique by several examples, showing
that it is applicable also in cases where subject reduction does not hold,
hence the standard technique for small-step semantics cannot be used.

    