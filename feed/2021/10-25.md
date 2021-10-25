
## 2021-10-25

### [[2110.11426] Native versus Overlay-based NDN over Wi-Fi 6 for the Internet of Vehicles](http://arxiv.org/abs/2110.11426)


  Internet of Vehicles (IoV) is a cornerstone building block of smart cities to
provide better traffic safety and mobile infotainment. Recently, improved
efficiency in WLAN-based dense scenarios has become widespread through Wi-Fi 6,
a license-free spectrum technology that can complement the cellular-based
infrastructure for IoV. In addition, Named Data Networking (NDN) is a promising
Internet architecture to accomplish content distribution in dynamic IoV
scenarios. However, NDN deployments, i.e., native (clean-slate) and overlay
(running on top of IP stack), require further investigation of their
performance over wireless networks, particularly regarding the IoV scenario.
This paper performs a comparative simulation-based study of these NDN
deployments over Wi-Fi 6 for IoV using real vehicular traces. To the best of
our knowledge, this is the first effort that extends ndnSIM 2 with an
overlay-based NDN implementation and that compares it with the native approach.
Results show that the overlay-based NDN consistently outperforms the native
one, reaching around 99% of requests satisfied, against only 42.35% in the best
case of native deployment.

    

### [[2110.11488] Certificate Root Stores: An Area of Unity or Disparity?](http://arxiv.org/abs/2110.11488)


  Organizations like Apple, Microsoft, Mozilla and Google maintain certificate
root stores, which are used as trust anchors by their software platforms. Is
there sufficient consensus on their root-store inclusion and trust policies?
Disparities appear astounding, including in the government-owned certificates
that they trust. Such a status-quo is alarming.

    

### [[2110.11518] Challenges and Opportunities in Integrated Space-Terrestrial Internet of Things](http://arxiv.org/abs/2110.11518)


  Large geographical regions and communities remain uncovered by terrestrial
network connections. To enable access equality, near-Earth orbit satellites
will play a defining role in providing Internet of Things (IoT) connectivity on
a world-wide scale in a flexible and cost-effective manner. This paper presents
the opportunities arising from global IoT solutions based on space assets, as
well as the key research challenges to be addressed. In particular, we discuss
existing space and terrestrial IoT technologies and protocols, and the
requirements that they need to meet to successfully materialize satellite IoT
services. We also propose a novel network architecture to be used by NB-IoT and
LoRaWAN technologies for implementing future space-terrestrial integrated IoT
networks.

    

### [[2110.11766] Semantic Identifiers and DNS Names for IoT](http://arxiv.org/abs/2110.11766)


  In this paper, we propose a scheme for representing semantic metadata of IoT
devices in compact identifiers and DNS names to enable simple discovery and
search with standard DNS servers. Our scheme defines a binary identifier as a
sequence of bits: a Context to use and several bits of fields corresponding to
semantic properties specific to the Context. The bit string is then encoded as
base32 characters and registered in DNS. Furthermore, we use the compact
semantic DNS names to offer support for search and discovery. We propose to
take advantage of the DNS system as the basic functionality for querying and
discovery of semantic properties related to IoT devices. We have defined three
specific Contexts for hierarchical semantic properties as well as logical and
geographical locations. For this last part, we have developed two prototypes
for managing geo-identifiers in LoRa networks, one based on Node and the Redis
in-memory database, the other one based on the CoreDNS server.

    

### [[2110.11865] Multipoint-to-point data aggregation using a single receiver and frequency-multiplexed intensity-modulated ONUs](http://arxiv.org/abs/2110.11865)


  We demonstrate 2.5-GHz-spacing frequency multiplexing capable of aggregating
64 intensity-modulated end-users using low-speed electronic and optoelectronic
components. All optical network units (ONUs) achieved high per-user capacity
with dedicated optical bands, enabling future large-bandwidth and low latency
applications.

    

### [[2110.11868] Proposition d'approches de déploiement des unités de bord de route dans les réseaux véhiculaires](http://arxiv.org/abs/2110.11868)


  Road Side Units (RSUs) have a crucial role in maintaining Vehicular Ad-hoc
Networks (VANETs) connectivity and coverage, especially, for applications
gathering or disseminating non-safety information. In big cities with complex
road network topology, a huge number of costly RSUs must be deployed to collect
data gathered by all moving vehicles. In this respect, several research works
focusing on RSUs deployment have been proposed. The thriving challenge would be
to (i) reduce the deployment cost by minimizing as far as possible the number
of used RSUs; and (ii) to maximize the coverage ratio. In this thesis, we
introduce a spatio-temporal RSU deployment framework including three methods
namely SPaCov/SPaCov+, HeSPic and MIP. SPaCov starts by mining frequent
mobility patterns of moving vehicles from their trajectories then it computes
the best RSU locations that cover the extracted patterns. Nonetheless, SPaCov+
extracts the frequent mobility patterns as well as the rare ones to enhance the
coverage ratio. HeSiC is a budget-constrained spatio-temporal coverage method
that aims to maximize the coverage ratio subject to a budget constraint, which
is defined in terms of RSUs number. MIP is a spatio-temporal coverage method
that aims to finding representative transactions from the sequential database
and computing coverage. Performed simulations highlight the efficiency and the
effectiveness of the proposed RSU deployment framework in terms of coverage
ratio, deployment cost, network latency and overhead.

    

### [[2012.15081] Fairness-Oriented User Scheduling for Bursty Downlink Transmission Using Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2012.15081)


  In this work, we develop practical user scheduling algorithms for downlink
bursty traffic with emphasis on user fairness. In contrast to the conventional
scheduling algorithms that either equally divides the transmission time slots
among users or maximizing some ratios without physcial meanings, we propose to
use the 5%-tile user data rate (5TUDR) as the metric to evaluate user fairness.
Since it is difficult to directly optimize 5TUDR, we first cast the problem
into the stochastic game framework and subsequently propose a Multi-Agent
Reinforcement Learning (MARL)-based algorithm to perform distributed
optimization on the resource block group (RBG) allocation. Furthermore, each
MARL agent is designed to take information measured by network counters from
multiple network layers (e.g. Channel Quality Indicator, Buffer size) as the
input states while the RBG allocation as action with a proposed reward function
designed to maximize 5TUDR. Extensive simulation is performed to show that the
proposed MARL-based scheduler can achieve fair scheduling while maintaining
good average network throughput as compared to conventional schedulers.

    

### [[2110.11337] Learning Universal User Representations via Self-Supervised Lifelong Behaviors Modeling](http://arxiv.org/abs/2110.11337)


  Universal user representation is an important research topic in industry, and
is widely used in diverse downstream user analysis tasks, such as user
profiling and user preference prediction. With the rapid development of
Internet service platforms, extremely long user behavior sequences have been
accumulated. However, existing researches have little ability to model
universal user representation based on lifelong sequences of user behavior
since registration. In this study, we propose a novel framework called Lifelong
User Representation Model (LURM) to tackle this challenge. Specifically, LURM
consists of two cascaded sub-models: (i) Bag of Interests (BoI) encodes user
behaviors in any time period into a sparse vector with super-high dimension
(e.g.,105); (ii) Self-supervised Multi-anchor EncoderNetwork (SMEN) maps
sequences of BoI features to multiple low-dimensional user representations by
contrastive learning. SMEN achieves almost lossless dimensionality reduction,
benefiting from a novel multi-anchor module which can learn different aspects
of user preferences. Experiments on several benchmark datasets show that our
approach outperforms state-of-the-art unsupervised representation methods in
downstream tasks

    

### [[2110.11339] Unsupervised cross-user adaptation in taste sensationrecognition based on surface electromyography withconformal prediction and domain regularizedcomponent analysis](http://arxiv.org/abs/2110.11339)


  Human taste sensation can be qualitatively described with surface
electromyography. However, the pattern recognition models trained on one
subject (the source domain) do not generalize well on other subjects (the
target domain). To improve the generalizability and transferability of taste
sensation models developed with sEMG data, two methods were innovatively
applied in this study: domain regularized component analysis (DRCA) and
conformal prediction with shrunken centroids (CPSC). The effectiveness of these
two methods was investigated independently in an unlabeled data augmentation
process with the unlabeled data from the target domain, and the same cross-user
adaptation pipeline were conducted on six subjects. The results show that DRCA
improved the classification accuracy on six subjects (p < 0.05), compared with
the baseline models trained only with the source domain data;, while CPSC did
not guarantee the accuracy improvement. Furthermore, the combination of DRCA
and CPSC presented statistically significant improvement (p < 0.05) in
classification accuracy on six subjects. The proposed strategy combining DRCA
and CPSC showed its effectiveness in addressing the cross-user data
distribution drift in sEMG-based taste sensation recognition application. It
also shows the potential in more cross-user adaptation applications.

    

### [[2110.11342] ESOD:Edge-based Task Scheduling for Object Detection](http://arxiv.org/abs/2110.11342)


  Object Detection on the mobile system is a challenge in terms of everything.
Nowadays, many object detection models have been designed, and most of them
concentrate on precision. However, the computation burden of those models on
mobile systems is unacceptable. Researchers have designed some lightweight
networks for mobiles by sacrificing precision. We present a novel edge-based
task scheduling framework for object detection (termed as ESOD). In detail, we
train a DNN model (termed as pre-model) to predict which object detection model
to use for the coming task and offloads to which edge servers by physical
characteristics of the image task (e.g., brightness, saturation). The results
show that ESOD can reduce latency and energy consumption by an average of
22.13% and 29.60% and improve the mAP to 45.8(with 0.9 mAP better),
respectively, compared with the SOTA DETR model.

    

### [[2110.11346] Data-Driven Offline Optimization For Architecting Hardware Accelerators](http://arxiv.org/abs/2110.11346)


  Industry has gradually moved towards application-specific hardware
accelerators in order to attain higher efficiency. While such a paradigm shift
is already starting to show promising results, designers need to spend
considerable manual effort and perform a large number of time-consuming
simulations to find accelerators that can accelerate multiple target
applications while obeying design constraints. Moreover, such a
"simulation-driven" approach must be re-run from scratch every time the set of
target applications or design constraints change. An alternative paradigm is to
use a "data-driven", offline approach that utilizes logged simulation data, to
architect hardware accelerators, without needing any form of simulations. Such
an approach not only alleviates the need to run time-consuming simulation, but
also enables data reuse and applies even when set of target applications
changes. In this paper, we develop such a data-driven offline optimization
method for designing hardware accelerators, dubbed PRIME, that enjoys all of
these properties. Our approach learns a conservative, robust estimate of the
desired cost function, utilizes infeasible points, and optimizes the design
against this estimate without any additional simulator queries during
optimization. PRIME architects accelerators -- tailored towards both single and
multiple applications -- improving performance upon state-of-the-art
simulation-driven methods by about 1.54x and 1.20x, while considerably reducing
the required total simulation time by 93% and 99%, respectively. In addition,
PRIME also architects effective accelerators for unseen applications in a
zero-shot setting, outperforming simulation-based methods by 1.26x.

    

### [[2110.11347] Multidimensional representations in late-life depression: convergence in neuroimaging, cognition, clinical symptomatology and genetics](http://arxiv.org/abs/2110.11347)


  Late-life depression (LLD) is characterized by considerable heterogeneity in
clinical manifestation. Unraveling such heterogeneity would aid in elucidating
etiological mechanisms and pave the road to precision and individualized
medicine. We sought to delineate, cross-sectionally and longitudinally,
disease-related heterogeneity in LLD linked to neuroanatomy, cognitive
functioning, clinical symptomatology, and genetic profiles. Multimodal data
from a multicentre sample (N=996) were analyzed. A semi-supervised clustering
method (HYDRA) was applied to regional grey matter (GM) brain volumes to derive
dimensional representations. Two dimensions were identified, which accounted
for the LLD-related heterogeneity in voxel-wise GM maps, white matter (WM)
fractional anisotropy (FA), neurocognitive functioning, clinical phenotype, and
genetics. Dimension one (Dim1) demonstrated relatively preserved brain anatomy
without WM disruptions relative to healthy controls. In contrast, dimension two
(Dim2) showed widespread brain atrophy and WM integrity disruptions, along with
cognitive impairment and higher depression severity. Moreover, one de novo
independent genetic variant (rs13120336) was significantly associated with Dim
1 but not with Dim 2. Notably, the two dimensions demonstrated significant
SNP-based heritability of 18-27% within the general population (N=12,518 in
UKBB). Lastly, in a subset of individuals having longitudinal measurements,
Dim2 demonstrated a more rapid longitudinal decrease in GM and brain age, and
was more likely to progress to Alzheimers disease, compared to Dim1 (N=1,413
participants and 7,225 scans from ADNI, BLSA, and BIOCARD datasets).

    

### [[2110.11377] CaloFlow II: Even Faster and Still Accurate Generation of Calorimeter Showers with Normalizing Flows](http://arxiv.org/abs/2110.11377)


  Recently, we introduced CaloFlow, a high-fidelity generative model for GEANT4
calorimeter shower emulation based on normalizing flows. Here, we present
CaloFlow v2, an improvement on our original framework that speeds up shower
generation by a further factor of 500 relative to the original. The improvement
is based on a technique called Probability Density Distillation, originally
developed for speech synthesis in the ML literature, and which we develop
further by introducing a set of powerful new loss terms. We demonstrate that
CaloFlow v2 preserves the same high fidelity of the original using qualitative
(average images, histograms of high level features) and quantitative
(classifier metric between GEANT4 and generated samples) measures. The result
is a generative model for calorimeter showers that matches the state-of-the-art
in speed (a factor of $10^4$ faster than GEANT4) and greatly surpasses the
previous state-of-the-art in fidelity.

    

### [[2110.11382] Efficient and Robust Mixed-Integer Optimization Methods for Training Binarized Deep Neural Networks](http://arxiv.org/abs/2110.11382)


  Compared to classical deep neural networks its binarized versions can be
useful for applications on resource-limited devices due to their reduction in
memory consumption and computational demands. In this work we study deep neural
networks with binary activation functions and continuous or integer weights
(BDNN). We show that the BDNN can be reformulated as a mixed-integer linear
program with bounded weight space which can be solved to global optimality by
classical mixed-integer programming solvers. Additionally, a local search
heuristic is presented to calculate locally optimal networks. Furthermore to
improve efficiency we present an iterative data-splitting heuristic which
iteratively splits the training set into smaller subsets by using the k-mean
method. Afterwards all data points in a given subset are forced to follow the
same activation pattern, which leads to a much smaller number of integer
variables in the mixed-integer programming formulation and therefore to
computational improvements. Finally for the first time a robust model is
presented which enforces robustness of the BDNN during training. All methods
are tested on random and real datasets and our results indicate that all models
can often compete with or even outperform classical DNNs on small network
architectures confirming the viability for applications having restricted
memory or computing power.

    

### [[2110.11383] Finite-Time Complexity of Online Primal-Dual Natural Actor-Critic Algorithm for Constrained Markov Decision Processes](http://arxiv.org/abs/2110.11383)


  We consider a discounted cost constrained Markov decision process (CMDP)
policy optimization problem, in which an agent seeks to maximize a discounted
cumulative reward subject to a number of constraints on discounted cumulative
utilities. To solve this constrained optimization program, we study an online
actor-critic variant of a classic primal-dual method where the gradients of
both the primal and dual functions are estimated using samples from a single
trajectory generated by the underlying time-varying Markov processes. This
online primal-dual natural actor-critic algorithm maintains and iteratively
updates three variables: a dual variable (or Lagrangian multiplier), a primal
variable (or actor), and a critic variable used to estimate the gradients of
both primal and dual variables. These variables are updated simultaneously but
on different time scales (using different step sizes) and they are all
intertwined with each other. Our main contribution is to derive a finite-time
analysis for the convergence of this algorithm to the global optimum of a CMDP
problem. Specifically, we show that with a proper choice of step sizes the
optimality gap and constraint violation converge to zero in expectation at a
rate $\mathcal{O}(1/K^{1/6})$, where K is the number of iterations. To our
knowledge, this paper is the first to study the finite-time complexity of an
online primal-dual actor-critic method for solving a CMDP problem. We also
validate the effectiveness of this algorithm through numerical simulations.

    

### [[2110.11385] Self-Initiated Open World Learning for Autonomous AI Agents](http://arxiv.org/abs/2110.11385)


  As more and more AI agents are used in practice, it is time to think about
how to make these agents fully autonomous so that they can learn by themselves
in a self-motivated and self-supervised manner rather than being retrained
periodically on the initiation of human engineers using expanded training data.
As the real-world is an open environment with unknowns or novelties, detecting
novelties or unknowns, gathering ground-truth training data, and incrementally
learning the unknowns make the agent more and more knowledgeable and powerful
over time. The key challenge is how to automate the process so that it is
carried out on the agent's own initiative and through its own interactions with
humans and the environment. Since an AI agent usually has a performance task,
characterizing each novelty becomes necessary so that the agent can formulate
an appropriate response to adapt its behavior to cope with the novelty and to
learn from it to improve its future responses and task performance. This paper
proposes a theoretic framework for this learning paradigm to promote the
research of building self-initiated open world learning agents.

    

### [[2110.11390] An Adaptive Digital Autopilot for Fixed-Wing Aircraft with Actuator Faults](http://arxiv.org/abs/2110.11390)


  This paper develops an adaptive digital autopilot for a fixed-wing aircraft
and compares its performance with a fixed-gain autopilot. The adaptive digital
autopilot is constructed by augmenting the autopilot architecture implemented
in PX4 flight stack with adaptive digital control laws that are updated using
the retrospective cost adaptive control algorithm. In order to investigate the
performance of the adaptive digital autopilot, the default gains of the
fixed-gain autopilot are scaled down to degrade its performance. This scenario
provides a venue for determining the ability of the adaptive digital autopilot
to compensate for the detuned fixed-gain autopilot. Next, the performance of
the adaptive autopilot is examined under failure conditions by simulating a
scenario where one of the control surfaces is assumed to be stuck at an unknown
angular position. The adaptive digital autopilot is tested in simulation, and
the resulting performance improvements are examined.

    

### [[2110.11395] SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](http://arxiv.org/abs/2110.11395)


  Pruning neural networks reduces inference time and memory costs. On standard
hardware, these benefits will be especially prominent if coarse-grained
structures, like feature maps, are pruned. We devise two novel saliency-based
methods for second-order structured pruning (SOSP) which include correlations
among all structures and layers. Our main method SOSP-H employs an innovative
second-order approximation, which enables saliency evaluations by fast
Hessian-vector products. SOSP-H thereby scales like a first-order method
despite taking into account the full Hessian. We validate SOSP-H by comparing
it to our second method SOSP-I that uses a well-established Hessian
approximation, and to numerous state-of-the-art methods. While SOSP-H performs
on par or better in terms of accuracy, it has clear advantages in terms of
scalability and efficiency. This allowed us to scale SOSP-H to large-scale
vision tasks, even though it captures correlations across all layers of the
network. To underscore the global nature of our pruning methods, we evaluate
their performance not only by removing structures from a pretrained network,
but also by detecting architectural bottlenecks. We show that our algorithms
allow to systematically reveal architectural bottlenecks, which we then remove
to further increase the accuracy of the networks.

    

### [[2110.11396] A Data-Driven Reconstruction Technique based on Newton's Method for Emission Tomography](http://arxiv.org/abs/2110.11396)


  In this work, we present the Deep Newton Reconstruction Network (DNR-Net), a
hybrid data-driven reconstruction technique for emission tomography inspired by
Newton's method, a well-known iterative optimization algorithm. The DNR-Net
employs prior information about the tomographic problem provided by the
projection operator while utilizing deep learning approaches to a) imitate
Newton's method by approximating the Newton descent direction and b) provide
data-driven regularisation. We demonstrate that DNR-Net is capable of providing
high-quality image reconstructions using data from SPECT phantom simulations by
applying it to reconstruct images from noisy sinograms, each one containing 24
projections. The Structural Similarity Index (SSIM) and the Contrast-to-Noise
ratio (CNR) were used to quantify the image quality. We also compare our
results to those obtained by the OSEM method. According to the quantitative
results, the DNR-Net produces reconstructions comparable to the ones produced
by OSEM while featuring higher contrast and less noise.

    

### [[2110.11400] Channel redundancy and overlap in convolutional neural networks with channel-wise NNK graphs](http://arxiv.org/abs/2110.11400)


  Feature spaces in the deep layers of convolutional neural networks (CNNs) are
often very high-dimensional and difficult to interpret. However, convolutional
layers consist of multiple channels that are activated by different types of
inputs, which suggests that more insights may be gained by studying the
channels and how they relate to each other. In this paper, we first analyze
theoretically channel-wise non-negative kernel (CW-NNK) regression graphs,
which allow us to quantify the overlap between channels and, indirectly, the
intrinsic dimension of the data representation manifold. We find that
redundancy between channels is significant and varies with the layer depth and
the level of regularization during training. Additionally, we observe that
there is a correlation between channel overlap in the last convolutional layer
and generalization performance. Our experimental results demonstrate that these
techniques can lead to a better understanding of deep representations.

    

### [[2110.11401] Trajectory Prediction using Generative Adversarial Network in Multi-Class Scenarios](http://arxiv.org/abs/2110.11401)


  Predicting traffic agents' trajectories is an important task for
auto-piloting. Most previous work on trajectory prediction only considers a
single class of road agents. We use a sequence-to-sequence model to predict
future paths from observed paths and we incorporate class information into the
model by concatenating extracted label representations with traditional
location inputs. We experiment with both LSTM and transformer encoders and we
use generative adversarial network as introduced in Social GAN to learn the
multi-modal behavior of traffic agents. We train our model on Stanford Drone
dataset which includes 6 classes of road agents and evaluate the impact of
different model components on the prediction performance in multi-class scenes.

    

### [[2110.11402] On the Regularization of Autoencoders](http://arxiv.org/abs/2110.11402)


  While much work has been devoted to understanding the implicit (and explicit)
regularization of deep nonlinear networks in the supervised setting, this paper
focuses on unsupervised learning, i.e., autoencoders are trained with the
objective of reproducing the output from the input. We extend recent results
[Jin et al. 2021] on unconstrained linear models and apply them to (1)
nonlinear autoencoders and (2) constrained linear autoencoders, obtaining the
following two results: first, we show that the unsupervised setting by itself
induces strong additional regularization, i.e., a severe reduction in the
model-capacity of the learned autoencoder: we derive that a deep nonlinear
autoencoder cannot fit the training data more accurately than a linear
autoencoder does if both models have the same dimensionality in their last
hidden layer (and under a few additional assumptions). Our second contribution
is concerned with the low-rank EDLAE model [Steck 2020], which is a linear
autoencoder with a constraint on the diagonal of the learned low-rank
parameter-matrix for improved generalization: we derive a closed-form
approximation to the optimum of its non-convex training-objective, and
empirically demonstrate that it is an accurate approximation across all
model-ranks in our experiments on three well-known data sets.

    

### [[2110.11403] SCENIC: A JAX Library for Computer Vision Research and Beyond](http://arxiv.org/abs/2110.11403)


  Scenic is an open-source JAX library with a focus on Transformer-based models
for computer vision research and beyond. The goal of this toolkit is to
facilitate rapid experimentation, prototyping, and research of new vision
architectures and models. Scenic supports a diverse range of vision tasks
(e.g., classification, segmentation, detection)and facilitates working on
multi-modal problems, along with GPU/TPU support for multi-host, multi-device
large-scale training. Scenic also offers optimized implementations of
state-of-the-art research models spanning a wide range of modalities. Scenic
has been successfully used for numerous projects and published papers and
continues serving as the library of choice for quick prototyping and
publication of new research ideas.

    

### [[2110.11404] Statistical discrimination in learning agents](http://arxiv.org/abs/2110.11404)


  Undesired bias afflicts both human and algorithmic decision making, and may
be especially prevalent when information processing trade-offs incentivize the
use of heuristics. One primary example is \textit{statistical discrimination}
-- selecting social partners based not on their underlying attributes, but on
readily perceptible characteristics that covary with their suitability for the
task at hand. We present a theoretical model to examine how information
processing influences statistical discrimination and test its predictions using
multi-agent reinforcement learning with various agent architectures in a
partner choice-based social dilemma. As predicted, statistical discrimination
emerges in agent policies as a function of both the bias in the training
population and of agent architecture. All agents showed substantial statistical
discrimination, defaulting to using the readily available correlates instead of
the outcome relevant features. We show that less discrimination emerges with
agents that use recurrent neural networks, and when their training environment
has less bias. However, all agent algorithms we tried still exhibited
substantial bias after learning in biased training populations.

    

### [[2110.11405] Illiterate DALL$\cdot$E Learns to Compose](http://arxiv.org/abs/2110.11405)


  Although DALL$\cdot$E has shown an impressive ability of composition-based
systematic generalization in image generation, it requires the dataset of
text-image pairs and the compositionality is provided by the text. In contrast,
object-centric representation models like the Slot Attention model learn
composable representations without the text prompt. However, unlike
DALL$\cdot$E its ability to systematically generalize for zero-shot generation
is significantly limited. In this paper, we propose a simple but novel
slot-based autoencoding architecture, called SLATE, for combining the best of
both worlds: learning object-centric representations that allows systematic
generalization in zero-shot image generation without text. As such, this model
can also be seen as an illiterate DALL$\cdot$E model. Unlike the pixel-mixture
decoders of existing object-centric representation models, we propose to use
the Image GPT decoder conditioned on the slots for capturing complex
interactions among the slots and pixels. In experiments, we show that this
simple and easy-to-implement architecture not requiring a text prompt achieves
significant improvement in in-distribution and out-of-distribution (zero-shot)
image generation and qualitatively comparable or better slot-attention
structure than the models based on mixture decoders.

    

### [[2110.11407] Video-Data Pipelines for Machine Learning Applications](http://arxiv.org/abs/2110.11407)


  Data pipelines are an essential component for end-to-end solutions that take
machine learning algorithms to production. Engineering data pipelines for
video-sequences poses several challenges including isolation of key-frames from
video sequences that are high quality and represent significant variations in
the scene. Manual isolation of such quality key-frames can take hours of
sifting through hours worth of video data. In this work, we present a data
pipeline framework that can automate this process of manual frame sifting in
video sequences by controlling the fraction of frames that can be removed based
on image quality and content type. Additionally, the frames that are retained
can be automatically tagged per sequence, thereby simplifying the process of
automated data retrieval for future ML model deployments. We analyze the
performance of the proposed video-data pipeline for versioned deployment and
monitoring for object detection algorithms that are trained on outdoor
autonomous driving video sequences. The proposed video-data pipeline can retain
anywhere between 0.1-20% of the all input frames that are representative of
high image quality and high variations in content. This frame selection,
automated scene tagging followed by model verification can be completed in
under 30 seconds for 22 video-sequences under analysis in this work. Thus, the
proposed framework can be scaled to additional video-sequence data sets for
automating ML versioned deployments.

    

### [[2110.11424] Analysis of memory consumption by neural networks based on hyperparameters](http://arxiv.org/abs/2110.11424)


  Deep learning models are trained and deployed in multiple domains. Increasing
usage of deep learning models alarms the usage of memory consumed while
computation by deep learning models. Existing approaches for reducing memory
consumption like model compression, hardware changes are specific. We propose a
generic analysis of memory consumption while training deep learning models in
comparison with hyperparameters used for training. Hyperparameters which
includes the learning rate, batchsize, number of hidden layers and depth of
layers decide the model performance, accuracy of the model. We assume the
optimizers and type of hidden layers as a known values. The change in
hyperparamaters and the number of hidden layers are the variables considered in
this proposed approach. For better understanding of the computation cost, this
proposed analysis studies the change in memory consumption with respect to
hyperparameters as main focus. This results in general analysis of memory
consumption changes during training when set of hyperparameters are altered.

    

### [[2110.11425] A Machine Learning Framework Towards Transparency in Experts' Decision Quality](http://arxiv.org/abs/2110.11425)


  Expert workers make non-trivial decisions with significant implications.
Experts' decision accuracy is thus a fundamental aspect of their judgment
quality, key to both management and consumers of experts' services. Yet, in
many important settings, transparency in experts' decision quality is rarely
possible because ground truth data for evaluating the experts' decisions is
costly and available only for a limited set of decisions. Furthermore,
different experts typically handle exclusive sets of decisions, and thus prior
solutions that rely on the aggregation of multiple experts' decisions for the
same instance are inapplicable. We first formulate the problem of estimating
experts' decision accuracy in this setting and then develop a
machine-learning-based framework to address it. Our method effectively
leverages both abundant historical data on workers' past decisions, and scarce
decision instances with ground truth information. We conduct extensive
empirical evaluations of our method's performance relative to alternatives
using both semi-synthetic data based on publicly available datasets, and
purposefully compiled dataset on real workers' decisions. The results show that
our approach is superior to existing alternatives across diverse settings,
including different data domains, experts' qualities, and the amount of ground
truth data. To our knowledge, this paper is the first to posit and address the
problem of estimating experts' decision accuracies from historical data with
scarcely available ground truth, and it is the first to offer comprehensive
results for this problem setting, establishing the performances that can be
achieved across settings, as well as the state-of-the-art performance on which
future work can build.

    

### [[2110.11430] How can classical multidimensional scaling go wrong?](http://arxiv.org/abs/2110.11430)


  Given a matrix $D$ describing the pairwise dissimilarities of a data set, a
common task is to embed the data points into Euclidean space. The classical
multidimensional scaling (cMDS) algorithm is a widespread method to do this.
However, theoretical analysis of the robustness of the algorithm and an
in-depth analysis of its performance on non-Euclidean metrics is lacking.
In this paper, we derive a formula, based on the eigenvalues of a matrix
obtained from $D$, for the Frobenius norm of the difference between $D$ and the
metric $D_{\text{cmds}}$ returned by cMDS. This error analysis leads us to the
conclusion that when the derived matrix has a significant number of negative
eigenvalues, then $\|D-D_{\text{cmds}}\|_F$, after initially decreasing, will
eventually increase as we increase the dimension. Hence, counterintuitively,
the quality of the embedding degrades as we increase the dimension. We
empirically verify that the Frobenius norm increases as we increase the
dimension for a variety of non-Euclidean metrics. We also show on several
benchmark datasets that this degradation in the embedding results in the
classification accuracy of both simple (e.g., 1-nearest neighbor) and complex
(e.g., multi-layer neural nets) classifiers decreasing as we increase the
embedding dimension.
Finally, our analysis leads us to a new efficiently computable algorithm that
returns a matrix $D_l$ that is at least as close to the original distances as
$D_t$ (the Euclidean metric closest in $\ell_2$ distance). While $D_l$ is not
metric, when given as input to cMDS instead of $D$, it empirically results in
solutions whose distance to $D$ does not increase when we increase the
dimension and the classification accuracy degrades less than the cMDS solution.

    

### [[2110.11431] Categorizing Items with Short and Noisy Descriptions using Ensembled Transferred Embeddings](http://arxiv.org/abs/2110.11431)


  Item categorization is a machine learning task which aims at classifying
e-commerce items, typically represented by textual attributes, to their most
suitable category from a predefined set of categories. An accurate item
categorization system is essential for improving both the user experience and
the operational processes of the company. In this work, we focus on item
categorization settings in which the textual attributes representing items are
noisy and short, and labels (i.e., accurate classification of items into
categories) are not available. In order to cope with such settings, we propose
a novel learning framework, Ensembled Transferred Embeddings (ETE), which
relies on two key ideas: 1) labeling a relatively small sample of the target
dataset, in a semi-automatic process, and 2) leveraging other datasets from
related domains or related tasks that are large-scale and labeled, to extract
"transferable embeddings". Evaluation of ETE on a large-scale real-world
dataset provided to us by PayPal, shows that it significantly outperforms
traditional as well as state-of-the-art item categorization methods.

    

### [[2110.11435] Generating Multivariate Load States Using a Conditional Variational Autoencoder](http://arxiv.org/abs/2110.11435)


  For planning of power systems and for the calibration of operational tools,
it is essential to analyse system performance in a large range of
representative scenarios. When the available historical data is limited,
generative models are a promising solution, but modelling high-dimensional
dependencies is challenging. In this paper, a multivariate load state
generating model on the basis of a conditional variational autoencoder (CVAE)
neural network is proposed. Going beyond common CVAE implementations, the model
includes stochastic variation of output samples under given latent vectors and
co-optimizes the parameters for this output variability. It is shown that this
improves statistical properties of the generated data. The quality of generated
multivariate loads is evaluated using univariate and multivariate performance
metrics. A generation adequacy case study on the European network is used to
illustrate model's ability to generate realistic tail distributions. The
experiments demonstrate that the proposed generator outperforms other data
generating mechanisms.

    

### [[2110.11439] Online Bipartite Matching with Predicted Degrees](http://arxiv.org/abs/2110.11439)


  We propose a model for online graph problems where algorithms are given
access to an oracle that predicts the degrees of nodes in the graph (e.g.,
based on past data). Within this model, we study the classic problem of online
bipartite matching. An extensive empirical evaluation shows that a greedy
algorithm called MinPredictedDegree compares favorably to state-of-the-art
online algorithms for this problem. We also initiate the theoretical study of
MinPredictedDegree on a natural random graph model with power law degree
distribution and show that it produces matchings almost as large as the maximum
matching on such graphs.

    

### [[2110.11442] Towards Noise-adaptive, Problem-adaptive Stochastic Gradient Descent](http://arxiv.org/abs/2110.11442)


  We design step-size schemes that make stochastic gradient descent (SGD)
adaptive to (i) the noise $\sigma^2$ in the stochastic gradients and (ii)
problem-dependent constants. When minimizing smooth, strongly-convex functions
with condition number $\kappa$, we first prove that $T$ iterations of SGD with
Nesterov acceleration and exponentially decreasing step-sizes can achieve a
near-optimal $\tilde{O}(\exp(-T/\sqrt{\kappa}) + \sigma^2/T)$ convergence rate.
Under a relaxed assumption on the noise, with the same step-size scheme and
knowledge of the smoothness, we prove that SGD can achieve an
$\tilde{O}(\exp(-T/\kappa) + \sigma^2/T)$ rate. In order to be adaptive to the
smoothness, we use a stochastic line-search (SLS) and show (via upper and
lower-bounds) that SGD converges at the desired rate, but only to a
neighbourhood of the solution. Next, we use SGD with an offline estimate of the
smoothness and prove convergence to the minimizer. However, its convergence is
slowed down proportional to the estimation error and we prove a lower-bound
justifying this slowdown. Compared to other step-size schemes, we empirically
demonstrate the effectiveness of exponential step-sizes coupled with a novel
variant of SLS.

    

### [[2110.11443] Off-Dynamics Inverse Reinforcement Learning from Hetero-Domain](http://arxiv.org/abs/2110.11443)


  We propose an approach for inverse reinforcement learning from hetero-domain
which learns a reward function in the simulator, drawing on the demonstrations
from the real world. The intuition behind the method is that the reward
function should not only be oriented to imitate the experts, but should
encourage actions adjusted for the dynamics difference between the simulator
and the real world. To achieve this, the widely used GAN-inspired IRL method is
adopted, and its discriminator, recognizing policy-generating trajectories, is
modified with the quantification of dynamics difference. The training process
of the discriminator can yield the transferable reward function suitable for
simulator dynamics, which can be guaranteed by derivation. Effectively, our
method assigns higher rewards for demonstration trajectories which do not
exploit discrepancies between the two domains. With extensive experiments on
continuous control tasks, our method shows its effectiveness and demonstrates
its scalability to high-dimensional tasks.

    

### [[2110.11446] ML with HE: Privacy Preserving Machine Learning Inferences for Genome Studies](http://arxiv.org/abs/2110.11446)


  Preserving the privacy and security of big data in the context of cloud
computing, while maintaining a certain level of efficiency of its processing
remains to be a subject, open for improvement. One of the most popular
applications epitomizing said concerns is found to be useful in genome
analysis. This work proposes a secure multi-label tumor classification method
using homomorphic encryption, whereby two different machine learning
algorithms, SVM and XGBoost, are used to classify the encrypted genome data of
different tumor types.

    

### [[2110.11451] An EMD-based Method for the Detection of Power Transformer Faults with a Hierarchical Ensemble Classifier](http://arxiv.org/abs/2110.11451)


  In this paper, an Empirical Mode Decomposition-based method is proposed for
the detection of transformer faults from Dissolve gas analysis (DGA) data.
Ratio-based DGA parameters are ranked using their skewness. Optimal sets of
intrinsic mode function coefficients are obtained from the ranked DGA
parameters. A Hierarchical classification scheme employing XGBoost is presented
for classifying the features to identify six different categories of
transformer faults. Performance of the Proposed Method is studied for publicly
available DGA data of 377 transformers. It is shown that the proposed method
can yield more than 90% sensitivity and accuracy in the detection of
transformer faults, a superior performance as compared to conventional methods
as well as several existing machine learning-based techniques.

    

### [[2110.11464] FDGATII : Fast Dynamic Graph Attention with Initial Residual and Identity Mapping](http://arxiv.org/abs/2110.11464)


  While Graph Neural Networks have gained popularity in multiple domains,
graph-structured input remains a major challenge due to (a) over-smoothing, (b)
noisy neighbours (heterophily), and (c) the suspended animation problem. To
address all these problems simultaneously, we propose a novel graph neural
network FDGATII, inspired by attention mechanism's ability to focus on
selective information supplemented with two feature preserving mechanisms.
FDGATII combines Initial Residuals and Identity Mapping with the more
expressive dynamic self-attention to handle noise prevalent from the
neighbourhoods in heterophilic data sets. By using sparse dynamic attention,
FDGATII is inherently parallelizable in design, whist efficient in operation;
thus theoretically able to scale to arbitrary graphs with ease. Our approach
has been extensively evaluated on 7 datasets. We show that FDGATII outperforms
GAT and GCN based benchmarks in accuracy and performance on fully supervised
tasks, obtaining state-of-the-art results on Chameleon and Cornell datasets
with zero domain-specific graph pre-processing, and demonstrate its versatility
and fairness.

    

### [[2110.11466] MLPerfTM HPC: A Holistic Benchmark Suite for Scientific Machine Learning on HPC Systems](http://arxiv.org/abs/2110.11466)


  Scientific communities are increasingly adopting machine learning and deep
learning models in their applications to accelerate scientific insights. High
performance computing systems are pushing the frontiers of performance with a
rich diversity of hardware resources and massive scale-out capabilities. There
is a critical need to understand fair and effective benchmarking of machine
learning applications that are representative of real-world scientific use
cases. MLPerfTM is a community-driven standard to benchmark machine learning
workloads, focusing on end-to-end performance metrics. In this paper, we
introduce MLPerf HPC, a benchmark suite of largescale scientific machine
learning training applications, driven by the MLCommonsTM Association. We
present the results from the first submission round including a diverse set of
some of the world's largest HPC systems. We develop a systematic framework for
their joint analysis and compare them in terms of data staging, algorithmic
convergence, and compute performance. As a result, we gain a quantitative
understanding of optimizations on different subsystems such as staging and
on-node loading of data, compute-unit utilization, and communication scheduling
enabling overall > 10x (end-to-end) performance improvements through system
scaling. Notably, our analysis shows a scale-dependent interplay between the
dataset size, a system's memory hierarchy, and training convergence that
underlines the importance of near compute storage. To overcome the
data-parallel scalability challenge at large batch sizes, we discuss specific
learning techniques and hybrid data-and-model parallelism that are effective on
large systems. We conclude by characterizing each benchmark with respect to
low-level memory, I/O, and network behavior to parameterize extended roofline
performance models in future rounds.

    

### [[2110.11467] Power Transformer Fault Diagnosis with Intrinsic Time-scale Decomposition and XGBoost Classifier](http://arxiv.org/abs/2110.11467)


  An intrinsic time-scale decomposition (ITD) based method for power
transformer fault diagnosis is proposed. Dissolved gas analysis (DGA)
parameters are ranked according to their skewness, and then ITD based features
extraction is performed. An optimal set of PRC features are determined by an
XGBoost classifier. For classification purpose, an XGBoost classifier is used
to the optimal PRC features set. The proposed method's performance in
classification is studied using publicly available DGA data of 376 power
transformers and employing an XGBoost classifier. The Proposed method achieves
more than 95% accuracy and high sensitivity and F1-score, better than
conventional methods and some recent machine learning-based fault diagnosis
approaches. Moreover, it gives better Cohen Kappa and F1-score as compared to
the recently introduced EMD-based hierarchical technique for fault diagnosis in
power transformers.

    

### [[2110.11477] Conditioning of Random Feature Matrices: Double Descent and Generalization Error](http://arxiv.org/abs/2110.11477)


  We provide (high probability) bounds on the condition number of random
feature matrices. In particular, we show that if the complexity ratio
$\frac{N}{m}$ where $N$ is the number of neurons and $m$ is the number of data
samples scales like $\log^{-3}(N)$ or $\log^{3}(m)$, then the random feature
matrix is well-conditioned. This result holds without the need of
regularization and relies on establishing a bound on the restricted isometry
constant of the random feature matrix. In addition, we prove that the risk
associated with regression problems using a random feature matrix exhibits the
double descent phenomenon and that this is an effect of the double descent
behavior of the condition number. The risk bounds include the
underparameterized setting using the least squares problem and the
overparameterized setting where using either the minimum norm interpolation
problem or a sparse regression problem. For the least squares or sparse
regression cases, we show that the risk decreases as $m$ and $N$ increase, even
in the presence of bounded or random noise. The risk bound matches the optimal
scaling in the literature and the constants in our results are explicit and
independent of the dimension of the data.

    

### [[2110.11479] Synt++: Utilizing Imperfect Synthetic Data to Improve Speech Recognition](http://arxiv.org/abs/2110.11479)


  With recent advances in speech synthesis, synthetic data is becoming a viable
alternative to real data for training speech recognition models. However,
machine learning with synthetic data is not trivial due to the gap between the
synthetic and the real data distributions. Synthetic datasets may contain
artifacts that do not exist in real data such as structured noise, content
errors, or unrealistic speaking styles. Moreover, the synthesis process may
introduce a bias due to uneven sampling of the data manifold. We propose two
novel techniques during training to mitigate the problems due to the
distribution gap: (i) a rejection sampling algorithm and (ii) using separate
batch normalization statistics for the real and the synthetic samples. We show
that these methods significantly improve the training of speech recognition
models using synthetic data. We evaluate the proposed approach on keyword
detection and Automatic Speech Recognition (ASR) tasks, and observe up to 18%
and 13% relative error reduction, respectively, compared to naively using the
synthetic data.

    

### [[2110.11486] Guess what? You can boost Federated Learning for free](http://arxiv.org/abs/2110.11486)


  Federated Learning (FL) exploits the computation power of edge devices,
typically mobile phones, while addressing privacy by letting data stay where it
is produced. FL has been used by major service providers to improve item
recommendations, virtual keyboards and text auto-completion services. While
appealing, FL performance is hampered by multiple factors: i) differing
capabilities of participating clients (e.g., computing power, memory and
network connectivity); ii) strict training constraints where devices must be
idle, plugged-in and connected to an unmetered WiFi; and iii) data
heterogeneity (a.k.a non-IIDness). Together, these lead to uneven
participation, straggling, dropout and consequently slow down convergence,
challenging the practicality of FL for many applications.
In this paper, we present GeL, the Guess and Learn algorithm, that
significantly speeds up convergence by guessing model updates for each client.
The power of GeL is to effectively perform ''free'' learning steps without any
additional gradient computations. GeL provides these guesses through clever use
of moments in the Adam optimizer in combination with the last computed gradient
on clients. Our extensive experimental study involving five standard FL
benchmarks shows that GeL speeds up the convergence up to 1.64x in
heterogeneous systems in the presence of data non-IIDness, saving tens of
thousands of gradient computations.

    

### [[2110.11489] Supporting Massive DLRM Inference Through Software Defined Memory](http://arxiv.org/abs/2110.11489)


  Deep Learning Recommendation Models (DLRM) are widespread, account for a
considerable data center footprint, and grow by more than 1.5x per year. With
model size soon to be in terabytes range, leveraging Storage ClassMemory (SCM)
for inference enables lower power consumption and cost. This paper evaluates
the major challenges in extending the memory hierarchy to SCM for DLRM, and
presents different techniques to improve performance through a Software Defined
Memory. We show how underlying technologies such as Nand Flash and3DXP
differentiate, and relate to real world scenarios, enabling from 5% to 29%
power savings.

    

### [[2110.11499] Wav2CLIP: Learning Robust Audio Representations From CLIP](http://arxiv.org/abs/2110.11499)


  We propose Wav2CLIP, a robust audio representation learning method by
distilling from Contrastive Language-Image Pre-training (CLIP). We
systematically evaluate Wav2CLIP on a variety of audio tasks including
classification, retrieval, and generation, and show that Wav2CLIP can
outperform several publicly available pre-trained audio representation
algorithms. Wav2CLIP projects audio into a shared embedding space with images
and text, which enables multimodal applications such as zero-shot
classification, and cross-modal retrieval. Furthermore, Wav2CLIP needs just
~10% of the data to achieve competitive performance on downstream tasks
compared with fully supervised models, and is more efficient to pre-train than
competing methods as it does not require learning a visual model in concert
with an auditory model. Finally, we demonstrate image generation from Wav2CLIP
as qualitative assessment of the shared embedding space. Our code and model
weights are open sourced and made available for further applications.

    

### [[2110.11501] Cortico-cerebellar networks as decoupling neural interfaces](http://arxiv.org/abs/2110.11501)


  The brain solves the credit assignment problem remarkably well. For credit to
be assigned across neural networks they must, in principle, wait for specific
neural computations to finish. How the brain deals with this inherent locking
problem has remained unclear. Deep learning methods suffer from similar locking
constraints both on the forward and feedback phase. Recently, decoupled neural
interfaces (DNIs) were introduced as a solution to the forward and feedback
locking problems in deep networks. Here we propose that a specialised brain
region, the cerebellum, helps the cerebral cortex solve similar locking
problems akin to DNIs. To demonstrate the potential of this framework we
introduce a systems-level model in which a recurrent cortical network receives
online temporal feedback predictions from a cerebellar module. We test this
cortico-cerebellar recurrent neural network (ccRNN) model on a number of
sensorimotor (line and digit drawing) and cognitive tasks (pattern recognition
and caption generation) that have been shown to be cerebellar-dependent. In all
tasks, we observe that ccRNNs facilitates learning while reducing ataxia-like
behaviours, consistent with classical experimental observations. Moreover, our
model also explains recent behavioural and neuronal observations while making
several testable predictions across multiple levels. Overall, our work offers a
novel perspective on the cerebellum as a brain-wide decoupling machine for
efficient credit assignment and opens a new avenue between deep learning and
neuroscience.

    

### [[2110.11524] Sequential Decision-Making for Active Object Detection from Hand](http://arxiv.org/abs/2110.11524)


  A key component of understanding hand-object interactions is the ability to
identify the active object -- the object that is being manipulated by the human
hand -- despite the occlusion induced by hand-object interactions. Based on the
observation that hand appearance is a strong indicator of the location and size
of the active object, we set up our active object detection method as a
sequential decision-making process that is conditioned on the location and
appearance of the hands. The key innovation of our approach is the design of
the active object detection policy that uses an internal representation called
the Relational Box Field, which allows for every pixel to regress an improved
location of an active object bounding box, essentially giving every pixel the
ability to vote for a better bounding box location. The policy is trained using
a hybrid imitation learning and reinforcement learning approach, and at test
time, the policy is used repeatedly to refine the bounding box location of the
active object. We perform experiments on two large-scale datasets: 100DOH and
MECCANO, improving AP50 performance by 8% and 30%, respectively, over the state
of the art.

    

### [[2110.11525] Digital and Physical-World Attacks on Remote Pulse Detection](http://arxiv.org/abs/2110.11525)


  Remote photoplethysmography (rPPG) is a technique for estimating blood volume
changes from reflected light without the need for a contact sensor. We present
the first examples of presentation attacks in the digital and physical domains
on rPPG from face video. Digital attacks are easily performed by adding
imperceptible periodic noise to the input videos. Physical attacks are
performed with illumination from visible spectrum LEDs placed in close
proximity to the face, while still being difficult to perceive with the human
eye. We also show that our attacks extend beyond medical applications, since
the method can effectively generate a strong periodic pulse on 3D-printed face
masks, which presents difficulties for pulse-based face presentation attack
detection (PAD). The paper concludes with ideas for using this work to improve
robustness of rPPG methods and pulse-based face PAD.

    

### [[2110.11526] Wide Neural Networks Forget Less Catastrophically](http://arxiv.org/abs/2110.11526)


  A growing body of research in continual learning is devoted to overcoming the
"Catastrophic Forgetting" of neural networks by designing new algorithms that
are more robust to the distribution shifts. While the recent progress in
continual learning literature is encouraging, our understanding of what
properties of neural networks contribute to catastrophic forgetting is still
limited. To address this, instead of focusing on continual learning algorithms,
in this work, we focus on the model itself and study the impact of "width" of
the neural network architecture on catastrophic forgetting, and show that width
has a surprisingly significant effect on forgetting. To explain this effect, we
study the learning dynamics of the network from various perspectives such as
gradient norm and sparsity, orthogonalization, and lazy training regime. We
provide potential explanations that are consistent with the empirical results
across different architectures and continual learning benchmarks.

    

### [[2110.11536] Neural-guided, Bidirectional Program Search for Abstraction and Reasoning](http://arxiv.org/abs/2110.11536)


  One of the challenges facing artificial intelligence research today is
designing systems capable of utilizing systematic reasoning to generalize to
new tasks. The Abstraction and Reasoning Corpus (ARC) measures such a
capability through a set of visual reasoning tasks. In this paper we report
incremental progress on ARC and lay the foundations for two approaches to
abstraction and reasoning not based in brute-force search. We first apply an
existing program synthesis system called DreamCoder to create symbolic
abstractions out of tasks solved so far, and show how it enables solving of
progressively more challenging ARC tasks. Second, we design a reasoning
algorithm motivated by the way humans approach ARC. Our algorithm constructs a
search graph and reasons over this graph structure to discover task solutions.
More specifically, we extend existing execution-guided program synthesis
approaches with deductive reasoning based on function inverse semantics to
enable a neural-guided bidirectional search algorithm. We demonstrate the
effectiveness of the algorithm on three domains: ARC, 24-Game tasks, and a
'double-and-add' arithmetic puzzle.

    

### [[2110.11538] Computing the Invariant Distribution of Randomly Perturbed Dynamical Systems Using Deep Learning](http://arxiv.org/abs/2110.11538)


  The invariant distribution, which is characterized by the stationary
Fokker-Planck equation, is an important object in the study of randomly
perturbed dynamical systems. Traditional numerical methods for computing the
invariant distribution based on the Fokker-Planck equation, such as finite
difference or finite element methods, are limited to low-dimensional systems
due to the curse of dimensionality. In this work, we propose a deep learning
based method to compute the generalized potential, i.e. the negative logarithm
of the invariant distribution multiplied by the noise. The idea of the method
is to learn a decomposition of the force field, as specified by the
Fokker-Planck equation, from the trajectory data. The potential component of
the decomposition gives the generalized potential. The method can deal with
high-dimensional systems, possibly with partially known dynamics. Using the
generalized potential also allows us to deal with systems at low temperatures,
where the invariant distribution becomes singular around the metastable states.
These advantages make it an efficient method to analyze invariant distributions
for practical dynamical systems. The effectiveness of the proposed method is
demonstrated by numerical examples.

    

### [[2110.11542] Adverse Media Mining for KYC and ESG Compliance](http://arxiv.org/abs/2110.11542)


  In recent years, institutions operating in the global market economy face
growing risks stemming from non-financial risk factors such as cyber,
third-party, and reputational outweighing traditional risks of credit and
liquidity. Adverse media or negative news screening is crucial for the
identification of such non-financial risks. Typical tools for screening are not
real-time, involve manual searches, require labor-intensive monitoring of
information sources. Moreover, they are costly processes to maintain up-to-date
with complex regulatory requirements and the institution's evolving risk
appetite.
In this extended abstract, we present an automated system to conduct both
real-time and batch search of adverse media for users' queries (person or
organization entities) using news and other open-source, unstructured sources
of information. Our scalable, machine-learning driven approach to
high-precision, adverse news filtering is based on four perspectives -
relevance to risk domains, search query (entity) relevance, adverse sentiment
analysis, and risk encoding. With the help of model evaluations and case
studies, we summarize the performance of our deployed application.

    

### [[2110.11552] GCNScheduler: Scheduling Distributed Computing Applications using Graph Convolutional Networks](http://arxiv.org/abs/2110.11552)


  We consider the classical problem of scheduling task graphs corresponding to
complex applications on distributed computing systems. A number of heuristics
have been previously proposed to optimize task scheduling with respect to
metrics such as makespan and throughput. However, they tend to be slow to run,
particularly for larger problem instances, limiting their applicability in more
dynamic systems. Motivated by the goal of solving these problems more rapidly,
we propose, for the first time, a graph convolutional network-based scheduler
(GCNScheduler). By carefully integrating an inter-task data dependency
structure with network settings into an input graph and feeding it to an
appropriate GCN, the GCNScheduler can efficiently schedule tasks of complex
applications for a given objective. We evaluate our scheme with baselines
through simulations. We show that not only can our scheme quickly and
efficiently learn from existing scheduling schemes, but also it can easily be
applied to large-scale settings where current scheduling schemes fail to
handle. We show that it achieves better makespan than the classic HEFT
algorithm, and almost the same throughput as throughput-oriented HEFT
(TP-HEFT), while providing several orders of magnitude faster scheduling times
in both cases. For example, for makespan minimization, GCNScheduler schedules
50-node task graphs in about 4 milliseconds while HEFT takes more than 1500
seconds; and for throughput maximization, GCNScheduler schedules 100-node task
graphs in about 3.3 milliseconds, compared to about 6.9 seconds for TP-HEFT.

    

### [[2110.11561] Merging Two Cultures: Deep and Statistical Learning](http://arxiv.org/abs/2110.11561)


  Merging the two cultures of deep and statistical learning provides insights
into structured high-dimensional data. Traditional statistical modeling is
still a dominant strategy for structured tabular data. Deep learning can be
viewed through the lens of generalized linear models (GLMs) with composite link
functions. Sufficient dimensionality reduction (SDR) and sparsity performs
nonlinear feature engineering. We show that prediction, interpolation and
uncertainty quantification can be achieved using probabilistic methods at the
output layer of the model. Thus a general framework for machine learning arises
that first generates nonlinear features (a.k.a factors) via sparse
regularization and stochastic gradient optimisation and second uses a
stochastic output layer for predictive uncertainty. Rather than using shallow
additive architectures as in many statistical models, deep learning uses layers
of semi affine input transformations to provide a predictive rule. Applying
these layers of transformations leads to a set of attributes (a.k.a features)
to which predictive statistical methods can be applied. Thus we achieve the
best of both worlds: scalability and fast predictive rule construction together
with uncertainty quantification. Sparse regularisation with un-supervised or
supervised learning finds the features. We clarify the duality between shallow
and wide models such as PCA, PPR, RRR and deep but skinny architectures such as
autoencoders, MLPs, CNN, and LSTM. The connection with data transformations is
of practical importance for finding good network architectures. By
incorporating probabilistic components at the output level we allow for
predictive uncertainty. For interpolation we use deep Gaussian process and ReLU
trees for classification. We provide applications to regression, classification
and interpolation. Finally, we conclude with directions for future research.

    

### [[2110.11571] Anti-Backdoor Learning: Training Clean Models on Poisoned Data](http://arxiv.org/abs/2110.11571)


  Backdoor attack has emerged as a major security threat to deep neural
networks (DNNs). While existing defense methods have demonstrated promising
results on detecting and erasing backdoor triggers, it is still not clear if
measures can be taken to avoid the triggers from being learned into the model
in the first place. In this paper, we introduce the concept of
\emph{anti-backdoor learning}, of which the aim is to train clean models on
backdoor-poisoned data. We frame the overall learning process as a dual-task of
learning the clean portion of data and learning the backdoor portion of data.
From this view, we identify two inherent characteristics of backdoor attacks as
their weaknesses: 1) the models learn backdoored data at a much faster rate
than learning clean data, and the stronger the attack the faster the model
converges on backdoored data; and 2) the backdoor task is tied to a specific
class (the backdoor target class). Based on these two weaknesses, we propose a
general learning scheme, Anti-Backdoor Learning (ABL), to automatically prevent
backdoor attacks during training. ABL introduces a two-stage \emph{gradient
ascent} mechanism into standard training to 1) help isolate backdoor examples
at an early training stage, and 2) break the correlation between backdoor
examples and the target class at a later training stage. Through extensive
experiments on multiple benchmark datasets against 10 state-of-the-art attacks,
we empirically show that ABL-trained models on backdoor-poisoned data achieve
the same performance as they were trained on purely clean data. Code is
available at \underline{this https URL}.

    

### [[2110.11578] PRECAD: Privacy-Preserving and Robust Federated Learning via Crypto-Aided Differential Privacy](http://arxiv.org/abs/2110.11578)


  Federated Learning (FL) allows multiple participating clients to train
machine learning models collaboratively by keeping their datasets local and
only exchanging model updates. Existing FL protocol designs have been shown to
be vulnerable to attacks that aim to compromise data privacy and/or model
robustness. Recently proposed defenses focused on ensuring either privacy or
robustness, but not both. In this paper, we develop a framework called PRECAD,
which simultaneously achieves differential privacy (DP) and enhances robustness
against model poisoning attacks with the help of cryptography. Using secure
multi-party computation (MPC) techniques (e.g., secret sharing), noise is added
to the model updates by the honest-but-curious server(s) (instead of each
client) without revealing clients' inputs, which achieves the benefit of
centralized DP in terms of providing a better privacy-utility tradeoff than
local DP based solutions. Meanwhile, a crypto-aided secure validation protocol
is designed to verify that the contribution of model update from each client is
bounded without leaking privacy. We show analytically that the noise added to
ensure DP also provides enhanced robustness against malicious model
submissions. We experimentally demonstrate that our PRECAD framework achieves
higher privacy-utility tradeoff and enhances robustness for the trained models.

    

### [[2110.11597] ProtoShotXAI: Using Prototypical Few-Shot Architecture for Explainable AI](http://arxiv.org/abs/2110.11597)


  Unexplainable black-box models create scenarios where anomalies cause
deleterious responses, thus creating unacceptable risks. These risks have
motivated the field of eXplainable Artificial Intelligence (XAI) to improve
trust by evaluating local interpretability in black-box neural networks.
Unfortunately, the ground truth is unavailable for the model's decision, so
evaluation is limited to qualitative assessment. Further, interpretability may
lead to inaccurate conclusions about the model or a false sense of trust. We
propose to improve XAI from the vantage point of the user's trust by exploring
a black-box model's latent feature space. We present an approach, ProtoShotXAI,
that uses a Prototypical few-shot network to explore the contrastive manifold
between nonlinear features of different classes. A user explores the manifold
by perturbing the input features of a query sample and recording the response
for a subset of exemplars from any class. Our approach is the first locally
interpretable XAI model that can be extended to, and demonstrated on, few-shot
networks. We compare ProtoShotXAI to the state-of-the-art XAI approaches on
MNIST, Omniglot, and ImageNet to demonstrate, both quantitatively and
qualitatively, that ProtoShotXAI provides more flexibility for model
exploration. Finally, ProtoShotXAI also demonstrates novel explainabilty and
detectabilty on adversarial samples.

    

### [[2110.11599] High Fidelity 3D Reconstructions with Limited Physical Views](http://arxiv.org/abs/2110.11599)


  Multi-view triangulation is the gold standard for 3D reconstruction from 2D
correspondences given known calibration and sufficient views. However in
practice, expensive multi-view setups -- involving tens sometimes hundreds of
cameras -- are required in order to obtain the high fidelity 3D reconstructions
necessary for many modern applications. In this paper we present a novel
approach that leverages recent advances in 2D-3D lifting using neural shape
priors while also enforcing multi-view equivariance. We show how our method can
achieve comparable fidelity to expensive calibrated multi-view rigs using a
limited (2-3) number of uncalibrated camera views.

    

### [[2110.11611] Error-Correcting Neural Networks for Semi-Lagrangian Advection in the Level-Set Method](http://arxiv.org/abs/2110.11611)


  We present a machine learning framework that blends image super-resolution
technologies with scalar transport in the level-set method. Here, we
investigate whether we can compute on-the-fly data-driven corrections to
minimize numerical viscosity in the coarse-mesh evolution of an interface. The
proposed system's starting point is the semi-Lagrangian formulation. And, to
reduce numerical dissipation, we introduce an error-quantifying multilayer
perceptron. The role of this neural network is to improve the numerically
estimated surface trajectory. To do so, it processes localized level-set,
velocity, and positional data in a single time frame for select vertices near
the moving front. Our main contribution is thus a novel
machine-learning-augmented transport algorithm that operates alongside
selective redistancing and alternates with conventional advection to keep the
adjusted interface trajectory smooth. Consequently, our procedure is more
efficient than full-scan convolutional-based applications because it
concentrates computational effort only around the free boundary. Also, we show
through various tests that our strategy is effective at counteracting both
numerical diffusion and mass loss. In passive advection problems, for example,
our method can achieve the same precision as the baseline scheme at twice the
resolution but at a fraction of the cost. Similarly, our hybrid technique can
produce feasible solidification fronts for crystallization processes. On the
other hand, highly deforming or lengthy simulations can precipitate bias
artifacts and inference deterioration. Likewise, stringent design velocity
constraints can impose certain limitations, especially for problems involving
rapid interface changes. In the latter cases, we have identified several
opportunity avenues to enhance robustness without forgoing our approach's basic
concept.

    

### [[2110.11619] DistFL: Distribution-aware Federated Learning for Mobile Scenarios](http://arxiv.org/abs/2110.11619)


  Federated learning (FL) has emerged as an effective solution to decentralized
and privacy-preserving machine learning for mobile clients. While traditional
FL has demonstrated its superiority, it ignores the non-iid (independently
identically distributed) situation, which widely exists in mobile scenarios.
Failing to handle non-iid situations could cause problems such as performance
decreasing and possible attacks. Previous studies focus on the "symptoms"
directly, as they try to improve the accuracy or detect possible attacks by
adding extra steps to conventional FL models. However, previous techniques
overlook the root causes for the "symptoms": blindly aggregating models with
the non-iid distributions. In this paper, we try to fundamentally address the
issue by decomposing the overall non-iid situation into several iid clusters
and conducting aggregation in each cluster. Specifically, we propose
\textbf{DistFL}, a novel framework to achieve automated and accurate
\textbf{Dist}ribution-aware \textbf{F}ederated \textbf{L}earning in a
cost-efficient way. DistFL achieves clustering via extracting and comparing the
\textit{distribution knowledge} from the uploaded models. With this framework,
we are able to generate multiple personalized models with distinctive
distributions and assign them to the corresponding clients. Extensive
experiments on mobile scenarios with popular model architectures have
demonstrated the effectiveness of DistFL.

    

### [[2110.11633] Explainable Landscape-Aware Optimization Performance Prediction](http://arxiv.org/abs/2110.11633)


  Efficient solving of an unseen optimization problem is related to appropriate
selection of an optimization algorithm and its hyper-parameters. For this
purpose, automated algorithm performance prediction should be performed that in
most commonly-applied practices involves training a supervised ML algorithm
using a set of problem landscape features. However, the main issue of training
such models is their limited explainability since they only provide information
about the joint impact of the set of landscape features to the end prediction
results. In this study, we are investigating explainable landscape-aware
regression models where the contribution of each landscape feature to the
prediction of the optimization algorithm performance is estimated on a global
and local level. The global level provides information about the impact of the
feature across all benchmark problems' instances, while the local level
provides information about the impact on a specific problem instance. The
experimental results are obtained using the COCO benchmark problems and three
differently configured modular CMA-ESs. The results show a proof of concept
that different set of features are important for different problem instances,
which indicates that further personalization of the landscape space is required
when training an automated algorithm performance prediction model.

    

### [[2110.11665] Diversified Sampling for Batched Bayesian Optimization with Determinantal Point Processes](http://arxiv.org/abs/2110.11665)


  In Bayesian Optimization (BO) we study black-box function optimization with
noisy point evaluations and Bayesian priors. Convergence of BO can be greatly
sped up by batching, where multiple evaluations of the black-box function are
performed in a single round. The main difficulty in this setting is to propose
at the same time diverse and informative batches of evaluation points. In this
work, we introduce DPP-Batch Bayesian Optimization (DPP-BBO), a universal
framework for inducing batch diversity in sampling based BO by leveraging the
repulsive properties of Determinantal Point Processes (DPP) to naturally
diversify the batch sampling procedure. We illustrate this framework by
formulating DPP-Thompson Sampling (DPP-TS) as a variant of the popular Thompson
Sampling (TS) algorithm and introducing a Markov Chain Monte Carlo procedure to
sample from it. We then prove novel Bayesian simple regret bounds for both
classical batched TS as well as our counterpart DPP-TS, with the latter bound
being tighter. Our real-world, as well as synthetic, experiments demonstrate
improved performance of DPP-BBO over classical batching methods with Gaussian
process and Cox process models.

    

### [[2110.11678] DQC: a Python program package for Differentiable Quantum Chemistry](http://arxiv.org/abs/2110.11678)


  Automatic differentiation represents a paradigm shift in scientific
programming, where evaluating both functions and their derivatives is required
for most applications. By removing the need to explicitly derive expressions
for gradients, development times can be be shortened, and calculations
simplified. For these reasons, automatic differentiation has fueled the rapid
growth of a variety of sophisticated machine learning techniques over the past
decade, but is now also increasingly showing its value to support {\it ab
initio} simulations of quantum systems, and enhance computational quantum
chemistry. Here we present an open-source differentiable quantum chemistry
simulation code, DQC, and explore applications facilitated by automatic
differentiation: (1) calculating molecular perturbation properties; (2)
reoptimizing a basis set for hydrocarbons; (3) checking the stability of
self-consistent field wave functions; and (4) predicting molecular properties
via alchemical perturbations.

    

### [[2110.11685] Adaptive Fusion Affinity Graph with Noise-free Online Low-rank Representation for Natural Image Segmentation](http://arxiv.org/abs/2110.11685)


  Affinity graph-based segmentation methods have become a major trend in
computer vision. The performance of these methods relies on the constructed
affinity graph, with particular emphasis on the neighborhood topology and
pairwise affinities among superpixels. Due to the advantages of assimilating
different graphs, a multi-scale fusion graph has a better performance than a
single graph with single-scale. However, these methods ignore the noise from
images which influences the accuracy of pairwise similarities. Multi-scale
combinatorial grouping and graph fusion also generate a higher computational
complexity. In this paper, we propose an adaptive fusion affinity graph
(AFA-graph) with noise-free low-rank representation in an online manner for
natural image segmentation. An input image is first over-segmented into
superpixels at different scales and then filtered by the proposed improved
kernel density estimation method. Moreover, we select global nodes of these
superpixels on the basis of their subspace-preserving presentation, which
reveals the feature distribution of superpixels exactly. To reduce time
complexity while improving performance, a sparse representation of global nodes
based on noise-free online low-rank representation is used to obtain a global
graph at each scale. The global graph is finally used to update a local graph
which is built upon all superpixels at each scale. Experimental results on the
BSD300, BSD500, MSRC, SBD, and PASCAL VOC show the effectiveness of AFA-graph
in comparison with state-of-the-art approaches.

    

### [[2110.11688] Differentially Private Coordinate Descent for Composite Empirical Risk Minimization](http://arxiv.org/abs/2110.11688)


  Machine learning models can leak information about the data used to train
them. Differentially Private (DP) variants of optimization algorithms like
Stochastic Gradient Descent (DP-SGD) have been designed to mitigate this,
inducing a trade-off between privacy and utility. In this paper, we propose a
new method for composite Differentially Private Empirical Risk Minimization
(DP-ERM): Differentially Private proximal Coordinate Descent (DP-CD). We
analyze its utility through a novel theoretical analysis of inexact coordinate
descent, and highlight some regimes where DP-CD outperforms DP-SGD, thanks to
the possibility of using larger step sizes. We also prove new lower bounds for
composite DP-ERM under coordinate-wise regularity assumptions, that are, in
some settings, nearly matched by our algorithm. In practical implementations,
the coordinate-wise nature of DP-CD updates demands special care in choosing
the clipping thresholds used to bound individual contributions to the
gradients. A natural parameterization of these thresholds emerges from our
theory, limiting the addition of unnecessarily large noise without requiring
coordinate-wise hyperparameter tuning or extra computational cost.

    

### [[2110.11707] Variational Wasserstein Barycenters with c-Cyclical Monotonicity](http://arxiv.org/abs/2110.11707)


  Wasserstein barycenter, built on the theory of optimal transport, provides a
powerful framework to aggregate probability distributions, and it has
increasingly attracted great attention within the machine learning community.
However, it suffers from severe computational burden, especially for high
dimensional and continuous settings. To this end, we develop a novel continuous
approximation method for the Wasserstein barycenters problem given sample
access to the input distributions. The basic idea is to introduce a variational
distribution as the approximation of the true continuous barycenter, so as to
frame the barycenters computation problem as an optimization problem, where
parameters of the variational distribution adjust the proxy distribution to be
similar to the barycenter. Leveraging the variational distribution, we
construct a tractable dual formulation for the regularized Wasserstein
barycenter problem with c-cyclical monotonicity, which can be efficiently
solved by stochastic optimization. We provide theoretical analysis on
convergence and demonstrate the practical effectiveness of our method on real
applications of subset posterior aggregation and synthetic data.

    

### [[2110.11713] Mechanistic Interpretation of Machine Learning Inference: A Fuzzy Feature Importance Fusion Approach](http://arxiv.org/abs/2110.11713)


  With the widespread use of machine learning to support decision-making, it is
increasingly important to verify and understand the reasons why a particular
output is produced. Although post-training feature importance approaches assist
this interpretation, there is an overall lack of consensus regarding how
feature importance should be quantified, making explanations of model
predictions unreliable. In addition, many of these explanations depend on the
specific machine learning approach employed and on the subset of data used when
calculating feature importance. A possible solution to improve the reliability
of explanations is to combine results from multiple feature importance
quantifiers from different machine learning approaches coupled with
re-sampling. Current state-of-the-art ensemble feature importance fusion uses
crisp techniques to fuse results from different approaches. There is, however,
significant loss of information as these approaches are not context-aware and
reduce several quantifiers to a single crisp output. More importantly, their
representation of 'importance' as coefficients is misleading and
incomprehensible to end-users and decision makers. Here we show how the use of
fuzzy data fusion methods can overcome some of the important limitations of
crisp fusion methods.

    

### [[2110.11721] Projection-Free Algorithm for Stochastic Bi-level Optimization](http://arxiv.org/abs/2110.11721)


  This work presents the first projection-free algorithm to solve stochastic
bi-level optimization problems, where the objective function depends on the
solution of another stochastic optimization problem. The proposed
$\textbf{S}$tochastic $\textbf{Bi}$-level $\textbf{F}$rank-$\textbf{W}$olfe
($\textbf{SBFW}$) algorithm can be applied to streaming settings and does not
make use of large batches or checkpoints. The sample complexity of SBFW is
shown to be $\mathcal{O}(\epsilon^{-3})$ for convex objectives and
$\mathcal{O}(\epsilon^{-4})$ for non-convex objectives. Improved rates are
derived for the stochastic compositional problem, which is a special case of
the bi-level problem, and entails minimizing the composition of two
expected-value functions. The proposed $\textbf{S}$tochastic
$\textbf{C}$ompositional $\textbf{F}$rank-$\textbf{W}$olfe ($\textbf{SCFW}$) is
shown to achieve a sample complexity of $\mathcal{O}(\epsilon^{-2})$ for convex
objectives and $\mathcal{O}(\epsilon^{-3})$ for non-convex objectives, at par
with the state-of-the-art sample complexities for projection-free algorithms
solving single-level problems. We demonstrate the advantage of the proposed
methods by solving the problem of matrix completion with denoising and the
problem of policy value evaluation in reinforcement learning.

    

### [[2110.11736] MANDERA: Malicious Node Detection in Federated Learning via Ranking](http://arxiv.org/abs/2110.11736)


  Federated learning is a distributed learning paradigm which seeks to preserve
the privacy of each participating node's data. However, federated learning is
vulnerable to attacks, specifically to our interest, model integrity attacks.
In this paper, we propose a novel method for malicious node detection called
MANDERA. By transferring the original message matrix into a ranking matrix
whose column shows the relative rankings of all local nodes along different
parameter dimensions, our approach seeks to distinguish the malicious nodes
from the benign ones with high efficiency based on key characteristics of the
rank domain. We have proved, under mild conditions, that MANDERA is guaranteed
to detect all malicious nodes under typical Byzantine attacks with no prior
knowledge or history about the participating nodes. The effectiveness of the
proposed approach is further confirmed by experiments on two classic datasets,
CIFAR-10 and MNIST. Compared to the state-of-art methods in the literature for
defending Byzantine attacks, MANDERA is unique in its way to identify the
malicious nodes by ranking and its robustness to effectively defense a wide
range of attacks.

    

### [[2110.11738] A Fast and Accurate Splitting Method for Optimal Transport: Analysis and Implementation](http://arxiv.org/abs/2110.11738)


  We develop a fast and reliable method for solving large-scale optimal
transport (OT) problems at an unprecedented combination of speed and accuracy.
Built on the celebrated Douglas-Rachford splitting technique, our method
tackles the original OT problem directly instead of solving an approximate
regularized problem, as many state-of-the-art techniques do. This allows us to
provide sparse transport plans and avoid numerical issues of methods that use
entropic regularization. The algorithm has the same cost per iteration as the
popular Sinkhorn method, and each iteration can be executed efficiently, in
parallel. The proposed method enjoys an iteration complexity $O(1/\epsilon)$
compared to the best-known $O(1/\epsilon^2)$ of the Sinkhorn method. In
addition, we establish a linear convergence rate for our formulation of the OT
problem. We detail an efficient GPU implementation of the proposed method that
maintains a primal-dual stopping criterion at no extra cost. Substantial
experiments demonstrate the effectiveness of our method, both in terms of
computation times and robustness.

    

### [[2110.11749] The Equilibrium Hypothesis: Rethinking implicit regularization in Deep Neural Networks](http://arxiv.org/abs/2110.11749)


  Modern Deep Neural Networks (DNNs) exhibit impressive generalization
properties on a variety of tasks without explicit regularization, suggesting
the existence of hidden regularization effects. Recent work by Baratin et al.
(2021) sheds light on an intriguing implicit regularization effect, showing
that some layers are much more aligned with data labels than other layers. This
suggests that as the network grows in depth and width, an implicit layer
selection phenomenon occurs during training. In this work, we provide the first
explanation for this alignment hierarchy. We introduce and empirically validate
the Equilibrium Hypothesis which states that the layers that achieve some
balance between forward and backward information loss are the ones with the
highest alignment to data labels. Our experiments demonstrate an excellent
match with the theoretical predictions.

    

### [[2110.11751] Forecasting Financial Market Structure from Network Features using Machine Learning](http://arxiv.org/abs/2110.11751)


  We propose a model that forecasts market correlation structure from link- and
node-based financial network features using machine learning. For such, market
structure is modeled as a dynamic asset network by quantifying time-dependent
co-movement of asset price returns across company constituents of major global
market indices. We provide empirical evidence using three different network
filtering methods to estimate market structure, namely Dynamic Asset Graph
(DAG), Dynamic Minimal Spanning Tree (DMST) and Dynamic Threshold Networks
(DTN). Experimental results show that the proposed model can forecast market
structure with high predictive performance with up to $40\%$ improvement over a
time-invariant correlation-based benchmark. Non-pair-wise correlation features
showed to be important compared to traditionally used pair-wise correlation
measures for all markets studied, particularly in the long-term forecasting of
stock market structure. Evidence is provided for stock constituents of the
DAX30, EUROSTOXX50, FTSE100, HANGSENG50, NASDAQ100 and NIFTY50 market indices.
Findings can be useful to improve portfolio selection and risk management
methods, which commonly rely on a backward-looking covariance matrix to
estimate portfolio risk.

    

### [[2110.11763] Game Redesign in No-regret Game Playing](http://arxiv.org/abs/2110.11763)


  We study the game redesign problem in which an external designer has the
ability to change the payoff function in each round, but incurs a design cost
for deviating from the original game. The players apply no-regret learning
algorithms to repeatedly play the changed games with limited feedback. The
goals of the designer are to (i) incentivize all players to take a specific
target action profile frequently; and (ii) incur small cumulative design cost.
We present game redesign algorithms with the guarantee that the target action
profile is played in T-o(T) rounds while incurring only o(T) cumulative design
cost. Game redesign describes both positive and negative applications: a
benevolent designer who incentivizes players to take a target action profile
with better social welfare compared to the solution of the original game, or a
malicious attacker whose target action profile benefits themselves but not the
players. Simulations on four classic games confirm the effectiveness of our
proposed redesign algorithms.

    

### [[2110.11769] Clustering of Bank Customers using LSTM-based encoder-decoder and Dynamic Time Warping](http://arxiv.org/abs/2110.11769)


  Clustering is an unsupervised data mining technique that can be employed to
segment customers. The efficient clustering of customers enables banks to
design and make offers based on the features of the target customers. The
present study uses a real-world financial dataset (Berka, 2000) to cluster bank
customers by an encoder-decoder network and the dynamic time warping (DTW)
method. The customer features required for clustering are obtained in four
ways: Dynamic Time Warping (DTW), Recency Frequency and Monetary (RFM), LSTM
encoder-decoder network, and our proposed hybrid method. Once the LSTM model
was trained by customer transaction data, a feature vector of each customer was
automatically extracted by the encoder.Moreover, the distance between pairs of
sequences of transaction amounts was obtained using DTW. Another vector feature
was calculated for customers by RFM scoring. In the hybrid method, the feature
vectors are combined from the encoder-decoder output, the DTW distance, and the
demographic data (e.g., age and gender). Finally, feature vectors were
introduced as input to the k-means clustering algorithm, and we compared
clustering results with Silhouette and Davies-Bouldin index. As a result, the
clusters obtained from the hybrid approach are more accurate and meaningful
than those derived from individual clustering techniques. In addition, the type
of neural network layers had a substantial effect on the clusters, and high
network error does not necessarily worsen clustering performance.

    

### [[2110.11773] Sinkformers: Transformers with Doubly Stochastic Attention](http://arxiv.org/abs/2110.11773)


  Attention based models such as Transformers involve pairwise interactions
between data points, modeled with a learnable attention matrix. Importantly,
this attention matrix is normalized with the SoftMax operator, which makes it
row-wise stochastic. In this paper, we propose instead to use Sinkhorn's
algorithm to make attention matrices doubly stochastic. We call the resulting
model a Sinkformer. We show that the row-wise stochastic attention matrices in
classical Transformers get close to doubly stochastic matrices as the number of
epochs increases, justifying the use of Sinkhorn normalization as an
informative prior. On the theoretical side, we show that, unlike the SoftMax
operation, this normalization makes it possible to understand the iterations of
self-attention modules as a discretized gradient-flow for the Wasserstein
metric. We also show in the infinite number of samples limit that, when
rescaling both attention matrices and depth, Sinkformers operate a heat
diffusion. On the experimental side, we show that Sinkformers enhance model
accuracy in vision and natural language processing tasks. In particular, on 3D
shapes classification, Sinkformers lead to a significant improvement.

    

### [[2110.11774] Learning Stable Vector Fields on Lie Groups](http://arxiv.org/abs/2110.11774)


  Learning robot motions from demonstration requires having models that are
able to represent vector fields for the full robot pose when the task is
defined in operational space. Recent advances in reactive motion generation
have shown that it is possible to learn adaptive, reactive, smooth, and stable
vector fields. However, these approaches define a vector field on a flat
Euclidean manifold, while representing vector fields for orientations required
to model the dynamics in non-Euclidean manifolds, such as Lie Groups. In this
paper, we present a novel vector field model that can guarantee most of the
properties of previous approaches i.e., stability, smoothness, and reactivity
beyond the Euclidean space. In the experimental evaluation, we show the
performance of our proposed vector field model to learn stable vector fields
for full robot poses as SE(2) and SE(3) in both simulated and real robotics
tasks.

    

### [[2110.11775] Federated Learning over Wireless IoT Networks with Optimized Communication and Resources](http://arxiv.org/abs/2110.11775)


  To leverage massive distributed data and computation resources, machine
learning in the network edge is considered to be a promising technique
especially for large-scale model training. Federated learning (FL), as a
paradigm of collaborative learning techniques, has obtained increasing research
attention with the benefits of communication efficiency and improved data
privacy. Due to the lossy communication channels and limited communication
resources (e.g., bandwidth and power), it is of interest to investigate fast
responding and accurate FL schemes over wireless systems. Hence, we investigate
the problem of jointly optimized communication efficiency and resources for FL
over wireless Internet of things (IoT) networks. To reduce complexity, we
divide the overall optimization problem into two sub-problems, i.e., the client
scheduling problem and the resource allocation problem. To reduce the
communication costs for FL in wireless IoT networks, a new client scheduling
policy is proposed by reusing stale local model parameters. To maximize
successful information exchange over networks, a Lagrange multiplier method is
first leveraged by decoupling variables including power variables, bandwidth
variables and transmission indicators. Then a linear-search based power and
bandwidth allocation method is developed. Given appropriate hyper-parameters,
we show that the proposed communication-efficient federated learning (CEFL)
framework converges at a strong linear rate. Through extensive experiments, it
is revealed that the proposed CEFL framework substantially boosts both the
communication efficiency and learning performance of both training loss and
test accuracy for FL over wireless IoT networks compared to a basic FL approach
with uniform resource allocation.

    

### [[2110.11780] Reconstruction of Sentinel-2 Time Series Using Robust Gaussian Mixture Models -- Application to the Detection of Anomalous Crop Development in wheat and rapeseed crops](http://arxiv.org/abs/2110.11780)


  Missing data is a recurrent problem in remote sensing, mainly due to cloud
coverage for multispectral images and acquisition problems. This can be a
critical issue for crop monitoring, especially for applications relying on
machine learning techniques, which generally assume that the feature matrix
does not have missing values. This paper proposes a Gaussian Mixture Model
(GMM) for the reconstruction of parcel-level features extracted from
multispectral images. A robust version of the GMM is also investigated, since
datasets can be contaminated by inaccurate samples or features (e.g., wrong
crop type reported, inaccurate boundaries, undetected clouds, etc). Additional
features extracted from Synthetic Aperture Radar (SAR) images using Sentinel-1
data are also used to provide complementary information and improve the
imputations. The robust GMM investigated in this work assigns reduced weights
to the outliers during the estimation of the GMM parameters, which improves the
final reconstruction. These weights are computed at each step of an
Expectation-Maximization (EM) algorithm by using outlier scores provided by the
isolation forest algorithm. Experimental validation is conducted on rapeseed
and wheat parcels located in the Beauce region (France). Overall, we show that
the GMM imputation method outperforms other reconstruction strategies. A mean
absolute error (MAE) of 0.013 (resp. 0.019) is obtained for the imputation of
the median Normalized Difference Index (NDVI) of the rapeseed (resp. wheat)
parcels. Other indicators (e.g., Normalized Difference Water Index) and
statistics (for instance the interquartile range, which captures heterogeneity
among the parcel indicator) are reconstructed at the same time with good
accuracy. In a dataset contaminated by irrelevant samples, using the robust GMM
is recommended since the standard GMM imputation can lead to inaccurate imputed
values.

    

### [[2110.11784] Safe rules for the identification of zeros in the solutions of the SLOPE problem](http://arxiv.org/abs/2110.11784)


  In this paper we propose a methodology to accelerate the resolution of the
so-called ``Sorted L-One Penalized Estimation'' (SLOPE) problem. Our method
leverages the concept of ``safe screening'', well-studied in the literature for
\textit{group-separable} sparsity-inducing norms, and aims at identifying the
zeros in the solution of SLOPE. More specifically, we introduce a family of
\(n!\) safe screening rules for this problem, where \(n\) is the dimension of
the primal variable, and propose a tractable procedure to verify if one of
these tests is passed. Our procedure has a complexity \(\mathcal{O}(n\log n +
LT)\) where \(T\leq n\) is a problem-dependent constant and \(L\) is the number
of zeros identified by the tests. We assess the performance of our proposed
method on a numerical benchmark and emphasize that it leads to significant
computational savings in many setups.

    

### [[2110.11794] Federated Unlearning via Class-Discriminative Pruning](http://arxiv.org/abs/2110.11794)


  We explore the problem of selectively forgetting categories from trained CNN
classification models in the federated learning (FL). Given that the data used
for training cannot be accessed globally in FL, our insights probe deep into
the internal influence of each channel. Through the visualization of feature
maps activated by different channels, we observe that different channels have a
varying contribution to different categories in image classification. Inspired
by this, we propose a method for scrubbing the model clean of information about
particular categories. The method does not require retraining from scratch, nor
global access to the data used for training. Instead, we introduce the concept
of Term Frequency Inverse Document Frequency (TF-IDF) to quantize the class
discrimination of channels. Channels with high TF-IDF scores have more
discrimination on the target categories and thus need to be pruned to unlearn.
The channel pruning is followed by a fine-tuning process to recover the
performance of the pruned model. Evaluated on CIFAR10 dataset, our method
accelerates the speed of unlearning by 8.9x for the ResNet model, and 7.9x for
the VGG model under no degradation in accuracy, compared to retraining from
scratch. For CIFAR100 dataset, the speedups are 9.9x and 8.4x, respectively. We
envision this work as a complementary block for FL towards compliance with
legal and ethical criteria.

    

### [[2110.11802] Deep Convolutional Autoencoders as Generic Feature Extractors in Seismological Applications](http://arxiv.org/abs/2110.11802)


  The idea of using a deep autoencoder to encode seismic waveform features and
then use them in different seismological applications is appealing. In this
paper, we designed tests to evaluate this idea of using autoencoders as feature
extractors for different seismological applications, such as event
discrimination (i.e., earthquake vs. noise waveforms, earthquake vs. explosion
waveforms, and phase picking). These tests involve training an autoencoder,
either undercomplete or overcomplete, on a large amount of earthquake
waveforms, and then using the trained encoder as a feature extractor with
subsequent application layers (either a fully connected layer, or a
convolutional layer plus a fully connected layer) to make the decision. By
comparing the performance of these newly designed models against the baseline
models trained from scratch, we conclude that the autoencoder feature extractor
approach may only perform well under certain conditions such as when the target
problems require features to be similar to the autoencoder encoded features,
when a relatively small amount of training data is available, and when certain
model structures and training strategies are utilized. The model structure that
works best in all these tests is an overcomplete autoencoder with a
convolutional layer and a fully connected layer to make the estimation.

    

### [[2110.11804] Probabilistic fine-tuning of pruning masks and PAC-Bayes self-bounded learning](http://arxiv.org/abs/2110.11804)


  We study an approach to learning pruning masks by optimizing the expected
loss of stochastic pruning masks, i.e., masks which zero out each weight
independently with some weight-specific probability. We analyze the training
dynamics of the induced stochastic predictor in the setting of linear
regression, and observe a data-adaptive L1 regularization term, in contrast to
the dataadaptive L2 regularization term known to underlie dropout in linear
regression. We also observe a preference to prune weights that are less
well-aligned with the data labels. We evaluate probabilistic fine-tuning for
optimizing stochastic pruning masks for neural networks, starting from masks
produced by several baselines. In each case, we see improvements in test error
over baselines, even after we threshold fine-tuned stochastic pruning masks.
Finally, since a stochastic pruning mask induces a stochastic neural network,
we consider training the weights and/or pruning probabilities simultaneously to
minimize a PAC-Bayes bound on generalization error. Using data-dependent
priors, we obtain a selfbounded learning algorithm with strong performance and
numerically tight bounds. In the linear model, we show that a PAC-Bayes
generalization error bound is controlled by the magnitude of the change in
feature alignment between the 'prior' and 'posterior' data.

    

### [[2110.11805] Model, sample, and epoch-wise descents: exact solution of gradient flow in the random feature model](http://arxiv.org/abs/2110.11805)


  Recent evidence has shown the existence of a so-called double-descent and
even triple-descent behavior for the generalization error of deep-learning
models. This important phenomenon commonly appears in implemented neural
network architectures, and also seems to emerge in epoch-wise curves during the
training process. A recent line of research has highlighted that random matrix
tools can be used to obtain precise analytical asymptotics of the
generalization (and training) errors of the random feature model. In this
contribution, we analyze the whole temporal behavior of the generalization and
training errors under gradient flow for the random feature model. We show that
in the asymptotic limit of large system size the full time-evolution path of
both errors can be calculated analytically. This allows us to observe how the
double and triple descents develop over time, if and when early stopping is an
option, and also observe time-wise descent structures. Our techniques are based
on Cauchy complex integral representations of the errors together with recent
random matrix methods based on linear pencils.

    

### [[2110.11812] Probabilistic ODE Solutions in Millions of Dimensions](http://arxiv.org/abs/2110.11812)


  Probabilistic solvers for ordinary differential equations (ODEs) have emerged
as an efficient framework for uncertainty quantification and inference on
dynamical systems. In this work, we explain the mathematical assumptions and
detailed implementation schemes behind solving {high-dimensional} ODEs with a
probabilistic numerical algorithm. This has not been possible before due to
matrix-matrix operations in each solver step, but is crucial for scientifically
relevant problems -- most importantly, the solution of discretised {partial}
differential equations. In a nutshell, efficient high-dimensional probabilistic
ODE solutions build either on independence assumptions or on Kronecker
structure in the prior model. We evaluate the resulting efficiency on a range
of problems, including the probabilistic numerical simulation of a differential
equation with millions of dimensions.

    

### [[2110.11819] Break your Bandit Routine with LSD Rewards: a Last Switch Dependent Analysis of Satiation and Seasonality](http://arxiv.org/abs/2110.11819)


  Motivated by the fact that humans like some level of unpredictability or
novelty, and might therefore get quickly bored when interacting with a
stationary policy, we introduce a novel non-stationary bandit problem, where
the expected reward of an arm is fully determined by the time elapsed since the
arm last took part in a switch of actions. Our model generalizes previous
notions of delay-dependent rewards, and also relaxes most assumptions on the
reward function. This enables the modeling of phenomena such as progressive
satiation and periodic behaviours. Building upon the Combinatorial Semi-Bandits
(CSB) framework, we design an algorithm and prove a bound on its regret with
respect to the optimal non-stationary policy (which is NP-hard to compute).
Similarly to previous works, our regret analysis is based on defining and
solving an appropriate trade-off between approximation and estimation.
Preliminary experiments confirm the superiority of our algorithm over both the
oracle greedy approach and a vanilla CSB solver.

    

### [[2110.11826] Predictive machine learning for prescriptive applications: a coupled training-validating approach](http://arxiv.org/abs/2110.11826)


  In this research we propose a new method for training predictive machine
learning models for prescriptive applications. This approach, which we refer to
as coupled validation, is based on tweaking the validation step in the standard
training-validating-testing scheme. Specifically, the coupled method considers
the prescription loss as the objective for hyper-parameter calibration. This
method allows for intelligent introduction of bias in the prediction stage to
improve decision making at the prescriptive stage, and is generally applicable
to most machine learning methods, including recently proposed hybrid
prediction-stochastic-optimization techniques, and can be easily implemented
without model-specific mathematical modeling. Several experiments with
synthetic and real data demonstrate promising results in reducing the
prescription costs in both deterministic and stochastic models.

    

### [[2110.11842] Multi-view Contrastive Graph Clustering](http://arxiv.org/abs/2110.11842)


  With the explosive growth of information technology, multi-view graph data
have become increasingly prevalent and valuable. Most existing multi-view
clustering techniques either focus on the scenario of multiple graphs or
multi-view attributes. In this paper, we propose a generic framework to cluster
multi-view attributed graph data. Specifically, inspired by the success of
contrastive learning, we propose multi-view contrastive graph clustering (MCGC)
method to learn a consensus graph since the original graph could be noisy or
incomplete and is not directly applicable. Our method composes of two key
steps: we first filter out the undesirable high-frequency noise while
preserving the graph geometric features via graph filtering and obtain a smooth
representation of nodes; we then learn a consensus graph regularized by graph
contrastive loss. Results on several benchmark datasets show the superiority of
our method with respect to state-of-the-art approaches. In particular, our
simple approach outperforms existing deep learning-based methods.

    

### [[2110.11848] Clustering Market Regimes using the Wasserstein Distance](http://arxiv.org/abs/2110.11848)


  The problem of rapid and automated detection of distinct market regimes is a
topic of great interest to financial mathematicians and practitioners alike. In
this paper, we outline an unsupervised learning algorithm for clustering
financial time-series into a suitable number of temporal segments (market
regimes). As a special case of the above, we develop a robust algorithm that
automates the process of classifying market regimes. The method is robust in
the sense that it does not depend on modelling assumptions of the underlying
time series as our experiments with real datasets show. This method -- dubbed
the Wasserstein $k$-means algorithm -- frames such a problem as one on the
space of probability measures with finite $p^\text{th}$ moment, in terms of the
$p$-Wasserstein distance between (empirical) distributions. We compare our
WK-means approach with a more traditional clustering algorithms by studying the
so-called maximum mean discrepancy scores between, and within clusters. In both
cases it is shown that the WK-means algorithm vastly outperforms all considered
competitor approaches. We demonstrate the performance of all approaches both in
a controlled environment on synthetic data, and on real data.

    

### [[2110.11854] Using scientific machine learning for experimental bifurcation analysis of dynamic systems](http://arxiv.org/abs/2110.11854)


  Augmenting mechanistic ordinary differential equation (ODE) models with
machine-learnable structures is an novel approach to create highly accurate,
low-dimensional models of engineering systems incorporating both expert
knowledge and reality through measurement data. Our exploratory study focuses
on training universal differential equation (UDE) models for physical nonlinear
dynamical systems with limit cycles: an aerofoil undergoing flutter
oscillations and an electrodynamic nonlinear oscillator. We consider examples
where training data is generated by numerical simulations, whereas we also
employ the proposed modelling concept to physical experiments allowing us to
investigate problems with a wide range of complexity. To collect the training
data, the method of control-based continuation is used as it captures not just
the stable but also the unstable limit cycles of the observed system. This
feature makes it possible to extract more information about the observed system
than the standard, open-loop approach would allow. We use both neural networks
and Gaussian processes as universal approximators alongside the mechanistic
models to give a critical assessment of the accuracy and robustness of the UDE
modelling approach. We also highlight the potential issues one may run into
during the training procedure indicating the limits of the current modelling
framework.

    

### [[2110.11855] Auctions Between Regret-Minimizing Agents](http://arxiv.org/abs/2110.11855)


  We analyze a scenario in which software agents implemented as regret
minimizing algorithms engage in a repeated auction on behalf of their users. We
study first price and second price auctions, as well as their generalized
versions (e.g., as those used for ad auctions). Using both theoretical analysis
and simulations, we show that, surprisingly, in second price auctions the
players have incentives to mis-report their true valuations to their own
learning agents, while in the first price auction it is a dominant strategy for
all players to truthfully report their valuations to their agents.

    

### [[2110.11860] AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations](http://arxiv.org/abs/2110.11860)


  This paper introduces Attentive Implicit Representation Networks (AIR-Nets),
a simple, but highly effective architecture for 3D reconstruction from point
clouds. Since representing 3D shapes in a local and modular fashion increases
generalization and reconstruction quality, AIR-Nets encode an input point cloud
into a set of local latent vectors anchored in 3D space, which locally describe
the object's geometry, as well as a global latent description, enforcing global
consistency. Our model is the first grid-free, encoder-based approach that
locally describes an implicit function. The vector attention mechanism from
[Zhao et al. 2020] serves as main point cloud processing module, and allows for
permutation invariance and translation equivariance. When queried with a 3D
coordinate, our decoder gathers information from the global and nearby local
latent vectors in order to predict an occupancy value. Experiments on the
ShapeNet dataset show that AIR-Nets significantly outperform previous
state-of-the-art encoder-based, implicit shape learning methods and especially
dominate in the sparse setting. Furthermore, our model generalizes well to the
FAUST dataset in a zero-shot setting. Finally, since AIR-Nets use a sparse
latent representation and follow a simple operating scheme, the model offers
several exiting avenues for future work. Our code is available at
this https URL.

    

### [[2110.11862] Graph Filtration Kernels](http://arxiv.org/abs/2110.11862)


  The majority of popular graph kernels is based on the concept of Haussler's
$\mathcal{R}$-convolution kernel and defines graph similarities in terms of
mutual substructures. In this work, we enrich these similarity measures by
considering graph filtrations: Using meaningful orders on the set of edges,
which allow to construct a sequence of nested graphs, we can consider a graph
at multiple granularities. For one thing, this provides access to features on
different levels of resolution. Furthermore, rather than to simply compare
frequencies of features in graphs, it allows for their comparison in terms of
when and for how long they exist in the sequences. In this work, we propose a
family of graph kernels that incorporate these existence intervals of features.
While our approach can be applied to arbitrary graph features, we particularly
highlight Weisfeiler-Lehman vertex labels, leading to efficient kernels. We
show that using Weisfeiler-Lehman labels over certain filtrations strictly
increases the expressive power over the ordinary Weisfeiler-Lehman procedure in
terms of deciding graph isomorphism. In fact, this result directly yields more
powerful graph kernels based on such features and has implications to graph
neural networks due to their close relationship to the Weisfeiler-Lehman
method. We empirically validate the expressive power of our graph kernels and
show significant improvements over state-of-the-art graph kernels in terms of
predictive performance on various real-world benchmark datasets.

    

### [[2110.11869] FLiText: A Faster and Lighter Semi-Supervised Text Classification with Convolution Networks](http://arxiv.org/abs/2110.11869)


  In natural language processing (NLP), state-of-the-art (SOTA) semi-supervised
learning (SSL) frameworks have shown great performance on deep pre-trained
language models such as BERT, and are expected to significantly reduce the
demand for manual labeling. However, our empirical studies indicate that these
frameworks are not suitable for lightweight models such as TextCNN, LSTM and
etc. In this work, we develop a new SSL framework called FLiText, which stands
for Faster and Lighter semi-supervised Text classification. FLiText introduces
an inspirer network together with the consistency regularization framework,
which leverages a generalized regular constraint on the lightweight models for
efficient SSL. As a result, FLiText obtains new SOTA performance for
lightweight models across multiple SSL benchmarks on text classification.
Compared with existing SOTA SSL methods on TextCNN, FLiText improves the
accuracy of lightweight model TextCNN from 51.00% to 90.49% on IMDb, 39.8% to
58.06% on Yelp-5, and from 55.3% to 65.08% on Yahoo. In addition, compared with
the fully supervised method on the full dataset, FLiText just uses less than 1%
of labeled data to improve the accuracy by 6.59%, 3.94%, and 3.22% on the
datasets of IMDb, Yelp-5, and Yahoo respectively.

    

### [[2110.11872] Patient level simulation and reinforcement learning to discover novel strategies for treating ovarian cancer](http://arxiv.org/abs/2110.11872)


  The prognosis for patients with epithelial ovarian cancer remains dismal
despite improvements in survival for other cancers. Treatment involves multiple
lines of chemotherapy and becomes increasingly heterogeneous after first-line
therapy. Reinforcement learning with real-world outcomes data has the potential
to identify novel treatment strategies to improve overall survival. We design a
reinforcement learning environment to model epithelial ovarian cancer treatment
trajectories and use model free reinforcement learning to investigate
therapeutic regimens for simulated patients.

    

### [[2110.11875] GeneDisco: A Benchmark for Experimental Design in Drug Discovery](http://arxiv.org/abs/2110.11875)


  In vitro cellular experimentation with genetic interventions, using for
example CRISPR technologies, is an essential step in early-stage drug discovery
and target validation that serves to assess initial hypotheses about causal
associations between biological mechanisms and disease pathologies. With
billions of potential hypotheses to test, the experimental design space for in
vitro genetic experiments is extremely vast, and the available experimental
capacity - even at the largest research institutions in the world - pales in
relation to the size of this biological hypothesis space. Machine learning
methods, such as active and reinforcement learning, could aid in optimally
exploring the vast biological space by integrating prior knowledge from various
information sources as well as extrapolating to yet unexplored areas of the
experimental design space based on available data. However, there exist no
standardised benchmarks and data sets for this challenging task and little
research has been conducted in this area to date. Here, we introduce GeneDisco,
a benchmark suite for evaluating active learning algorithms for experimental
design in drug discovery. GeneDisco contains a curated set of multiple publicly
available experimental data sets as well as open-source implementations of
state-of-the-art active learning policies for experimental design and
exploration.

    

### [[2110.11876] Tight and Robust Private Mean Estimation with Few Users](http://arxiv.org/abs/2110.11876)


  In this work, we study high-dimensional mean estimation under user-level
differential privacy, and attempt to design an
$(\epsilon,\delta)$-differentially private mechanism using as few users as
possible. In particular, we provide a nearly optimal trade-off between the
number of users and the number of samples per user required for private mean
estimation, even when the number of users is as low as
$O(\frac{1}{\epsilon}\log\frac{1}{\delta})$. Interestingly our bound
$O(\frac{1}{\epsilon}\log\frac{1}{\delta})$ on the number of users is
independent of the dimension, unlike the previous work that depends
polynomially on the dimension, solving a problem left open by Amin et
al.~(ICML'2019). Our mechanism enjoys robustness up to the point that even if
the information of $49\%$ of the users are corrupted, our final estimation is
still approximately accurate. Finally, our results also apply to a broader
range of problems such as learning discrete distributions, stochastic convex
optimization, empirical risk minimization, and a variant of stochastic gradient
descent via a reduction to differentially private mean estimation.

    

### [[2110.11886] Conditional Gaussian PAC-Bayes](http://arxiv.org/abs/2110.11886)


  Recent studies have empirically investigated different methods to train a
stochastic classifier by optimising a PAC-Bayesian bound via stochastic
gradient descent. Most of these procedures need to replace the
misclassification error with a surrogate loss, leading to a mismatch between
the optimisation objective and the actual generalisation bound. The present
paper proposes a novel training algorithm that optimises the PAC-Bayesian
bound, without relying on any surrogate loss. Empirical results show that the
bounds obtained with this approach are tighter than those found in the
literature.

    

### [[2110.11891] On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning](http://arxiv.org/abs/2110.11891)


  Machine unlearning, i.e. having a model forget about some of its training
data, has become increasingly more important as privacy legislation promotes
variants of the right-to-be-forgotten. In the context of deep learning,
approaches for machine unlearning are broadly categorized into two classes:
exact unlearning methods, where an entity has formally removed the data point's
impact on the model by retraining the model from scratch, and approximate
unlearning, where an entity approximates the model parameters one would obtain
by exact unlearning to save on compute costs. In this paper we first show that
the definition that underlies approximate unlearning, which seeks to prove the
approximately unlearned model is close to an exactly retrained model, is
incorrect because one can obtain the same model using different datasets. Thus
one could unlearn without modifying the model at all. We then turn to exact
unlearning approaches and ask how to verify their claims of unlearning. Our
results show that even for a given training trajectory one cannot formally
prove the absence of certain data points used during training. We thus conclude
that unlearning is only well-defined at the algorithmic level, where an
entity's only possible auditable claim to unlearning is that they used a
particular algorithm designed to allow for external scrutiny during an audit.

    

### [[2110.11918] MIGS: Meta Image Generation from Scene Graphs](http://arxiv.org/abs/2110.11918)


  Generation of images from scene graphs is a promising direction towards
explicit scene generation and manipulation. However, the images generated from
the scene graphs lack quality, which in part comes due to high difficulty and
diversity in the data. We propose MIGS (Meta Image Generation from Scene
Graphs), a meta-learning based approach for few-shot image generation from
graphs that enables adapting the model to different scenes and increases the
image quality by training on diverse sets of tasks. By sampling the data in a
task-driven fashion, we train the generator using meta-learning on different
sets of tasks that are categorized based on the scene attributes. Our results
show that using this meta-learning approach for the generation of images from
scene graphs achieves state-of-the-art performance in terms of image quality
and capturing the semantic relationships in the scene. Project Website:
this https URL


### [[2110.11940] Logical Activation Functions: Logit-space equivalents of Boolean Operators](http://arxiv.org/abs/2110.11940)


  Neuronal representations within artificial neural networks are commonly
understood as logits, representing the log-odds score of presence (versus
absence) of features within the stimulus. Under this interpretation, we can
derive the probability $P(x_0 \land x_1)$ that a pair of independent features
are both present in the stimulus from their logits. By converting the resulting
probability back into a logit, we obtain a logit-space equivalent of the AND
operation. However, since this function involves taking multiple exponents and
logarithms, it is not well suited to be directly used within neural networks.
We thus constructed an efficient approximation named $\text{AND}_\text{AIL}$
(the AND operator Approximate for Independent Logits) utilizing only comparison
and addition operations, which can be deployed as an activation function in
neural networks. Like MaxOut, $\text{AND}_\text{AIL}$ is a generalization of
ReLU to two-dimensions. Additionally, we constructed efficient approximations
of the logit-space equivalents to the OR and XNOR operators. We deployed these
new activation functions, both in isolation and in conjunction, and
demonstrated their effectiveness on a variety of tasks including image
classification, transfer learning, abstract reasoning, and compositional
zero-shot learning.

    

### [[2110.11945] SOFT: Softmax-free Transformer with Linear Complexity](http://arxiv.org/abs/2110.11945)


  Vision transformers (ViTs) have pushed the state-of-the-art for various
visual recognition tasks by patch-wise image tokenization followed by
self-attention. However, the employment of self-attention modules results in a
quadratic complexity in both computation and memory usage. Various attempts on
approximating the self-attention computation with linear complexity have been
made in Natural Language Processing. However, an in-depth analysis in this work
shows that they are either theoretically flawed or empirically ineffective for
visual recognition. We further identify that their limitations are rooted in
keeping the softmax self-attention during approximations. Specifically,
conventional self-attention is computed by normalizing the scaled dot-product
between token feature vectors. Keeping this softmax operation challenges any
subsequent linearization efforts. Based on this insight, for the first time, a
softmax-free transformer or SOFT is proposed. To remove softmax in
self-attention, Gaussian kernel function is used to replace the dot-product
similarity without further normalization. This enables a full self-attention
matrix to be approximated via a low-rank matrix decomposition. The robustness
of the approximation is achieved by calculating its Moore-Penrose inverse using
a Newton-Raphson method. Extensive experiments on ImageNet show that our SOFT
significantly improves the computational efficiency of existing ViT variants.
Crucially, with a linear complexity, much longer token sequences are permitted
in SOFT, resulting in superior trade-off between accuracy and complexity.

    

### [[2110.11948] Learning Proposals for Practical Energy-Based Regression](http://arxiv.org/abs/2110.11948)


  Energy-based models (EBMs) have experienced a resurgence within machine
learning in recent years, including as a promising alternative for
probabilistic regression. However, energy-based regression requires a proposal
distribution to be manually designed for training, and an initial estimate has
to be provided at test-time. We address both of these issues by introducing a
conceptually simple method to automatically learn an effective proposal
distribution, which is parameterized by a separate network head. To this end,
we derive a surprising result, leading to a unified training objective that
jointly minimizes the KL divergence from the proposal to the EBM, and the
negative log-likelihood of the EBM. At test-time, we can then employ importance
sampling with the trained proposal to efficiently evaluate the learned EBM and
produce stand-alone predictions. Furthermore, we utilize our derived training
objective to learn mixture density networks (MDNs) with a jointly trained
energy-based teacher, consistently outperforming conventional MDN training on
four real-world regression tasks within computer vision. Code is available at
this https URL.

    

### [[2110.11950] Adversarial robustness for latent models: Revisiting the robust-standard accuracies tradeoff](http://arxiv.org/abs/2110.11950)


  Over the past few years, several adversarial training methods have been
proposed to improve the robustness of machine learning models against
adversarial perturbations in the input. Despite remarkable progress in this
regard, adversarial training is often observed to drop the standard test
accuracy. This phenomenon has intrigued the research community to investigate
the potential tradeoff between standard and robust accuracy as two performance
measures. In this paper, we revisit this tradeoff for latent models and argue
that this tradeoff is mitigated when the data enjoys a low-dimensional
structure. In particular, we consider binary classification under two data
generative models, namely Gaussian mixture model and generalized linear model,
where the feature data lie on a low-dimensional manifold. We show that as the
manifold dimension to the ambient dimension decreases, one can obtain models
that are nearly optimal with respect to both, the standard accuracy and the
robust accuracy measures.

    

### [[2110.11952] Optimal randomized classification trees](http://arxiv.org/abs/2110.11952)


  Classification and Regression Trees (CARTs) are off-the-shelf techniques in
modern Statistics and Machine Learning. CARTs are traditionally built by means
of a greedy procedure, sequentially deciding the splitting predictor
variable(s) and the associated threshold. This greedy approach trains trees
very fast, but, by its nature, their classification accuracy may not be
competitive against other state-of-the-art procedures. Moreover, controlling
critical issues, such as the misclassification rates in each of the classes, is
difficult. To address these shortcomings, optimal decision trees have been
recently proposed in the literature, which use discrete decision variables to
model the path each observation will follow in the tree. Instead, we propose a
new approach based on continuous optimization. Our classifier can be seen as a
randomized tree, since at each node of the decision tree a random decision is
made. The computational experience reported demonstrates the good performance
of our procedure.

    

### [[1912.02655] Obesity Prediction with EHR Data: A deep learning approach with interpretable elements](http://arxiv.org/abs/1912.02655)


  Childhood obesity is a major public health challenge. Early prediction and
identification of the children at a high risk of developing childhood obesity
may help in engaging earlier and more effective interventions to prevent and
manage obesity. Most existing predictive tools for childhood obesity primarily
rely on traditional regression-type methods using only a few hand-picked
features and without exploiting longitudinal patterns of children data. Deep
learning methods allow the use of high-dimensional longitudinal datasets. In
this paper, we present a deep learning model designed for predicting future
obesity patterns from generally available items on children medical history. To
do this, we use a large unaugmented electronic health records dataset from a
large pediatric health system. We adopt a general LSTM network architecture
which are known to better represent the longitudinal data. We train our
proposed model on both dynamic and static EHR data. Our model is used to
predict obesity for ages between 2-20 years. We compared the performance of our
LSTM model with other machine learning methods that aggregate over sequential
data and ignore temporality. To add interpretability, we have additionally
included an attention layer to calculate the attention scores for the
timestamps and rank features of each timestamp.

    

### [[2003.04315] LIMEADE: A General Framework for Explanation-Based Human Tuning of Opaque Machine Learners](http://arxiv.org/abs/2003.04315)


  Research in human-centered AI has shown the benefits of systems that can
explain their predictions. Methods that allow humans to tune a model in
response to the explanations are similarly useful. While both capabilities are
well-developed for transparent learning models (e.g., linear models and GA2Ms),
and recent techniques (e.g., LIME and SHAP) can generate explanations for
opaque models, no method for tuning opaque models in response to explanations
has been user-tested to date. This paper introduces LIMEADE, a general
framework for tuning an arbitrary machine learning model based on an
explanation of the model's prediction. We demonstrate the generality of our
approach with two case studies. First, we successfully utilize LIMEADE for the
human tuning of opaque image classifiers. Second, we apply our framework to a
neural recommender system for scientific papers on a public website and report
on a user study showing that our framework leads to significantly higher
perceived user control, trust, and satisfaction. Analyzing 300 user logs from
our publicly-deployed website, we uncover a tradeoff between canonical greedy
explanations and diverse explanations that better facilitate human tuning.

    

### [[2004.03808] Improving BERT with Self-Supervised Attention](http://arxiv.org/abs/2004.03808)


  One of the most popular paradigms of applying large pre-trained NLP models
such as BERT is to fine-tune it on a smaller dataset. However, one challenge
remains as the fine-tuned model often overfits on smaller datasets. A symptom
of this phenomenon is that irrelevant or misleading words in the sentence,
which are easy to understand for human beings, can substantially degrade the
performance of these finetuned BERT models. In this paper, we propose a novel
technique, called Self-Supervised Attention (SSA) to help facilitate this
generalization challenge. Specifically, SSA automatically generates weak,
token-level attention labels iteratively by probing the fine-tuned model from
the previous iteration. We investigate two different ways of integrating SSA
into BERT and propose a hybrid approach to combine their benefits. Empirically,
through a variety of public datasets, we illustrate significant performance
improvement using our SSA-enhanced BERT model.

    

### [[2005.13985] Mass Estimation of Galaxy Clusters with Deep Learning II: CMB Cluster Lensing](http://arxiv.org/abs/2005.13985)


  We present a new application of deep learning to reconstruct the cosmic
microwave background (CMB) temperature maps from the images of microwave sky,
and to use these reconstructed maps to estimate the masses of galaxy clusters.
We use a feed-forward deep learning network, mResUNet, for both steps of the
analysis. The first deep learning model, mResUNet-I, is trained to reconstruct
foreground and noise suppressed CMB maps from a set of simulated images of the
microwave sky that include signals from the cosmic microwave background,
astrophysical foregrounds like dusty and radio galaxies, instrumental noise as
well as the cluster's own thermal Sunyaev Zel'dovich signal. The second deep
learning model, mResUNet-II, is trained to estimate cluster masses from the
gravitational lensing signature in the reconstructed foreground and noise
suppressed CMB maps. For SPTpol-like noise levels, the trained mResUNet-II
model recovers the mass for $10^4$ galaxy cluster samples with a 1-$\sigma$
uncertainty $\Delta M_{\rm 200c}^{\rm est}/M_{\rm 200c}^{\rm est} =$ 0.108 and
0.016 for input cluster mass $M_{\rm 200c}^{\rm true}=10^{14}~\rm M_{\odot}$
and $8\times 10^{14}~\rm M_{\odot}$, respectively. We also test for potential
bias on recovered masses, finding that for a set of $10^5$ clusters the
estimator recovers $M_{\rm 200c}^{\rm est} = 2.02 \times 10^{14}~\rm
M_{\odot}$, consistent with the input at 1% level. The 2 $\sigma$ upper limit
on potential bias is at 3.5% level.

    

### [[2006.10858] Rehabilitating Isomap: Euclidean Representation of Geodesic Structure](http://arxiv.org/abs/2006.10858)


  Manifold learning techniques for nonlinear dimension reduction assume that
high-dimensional feature vectors lie on a low-dimensional manifold, then
attempt to exploit manifold structure to obtain useful low-dimensional
Euclidean representations of the data. Isomap, a seminal manifold learning
technique, is an elegant synthesis of two simple ideas: the approximation of
Riemannian distances with shortest path distances on a graph that localizes
manifold structure, and the approximation of shortest path distances with
Euclidean distances by multidimensional scaling. We revisit the rationale for
Isomap, clarifying what Isomap does and what it does not. In particular, we
explore the widespread perception that Isomap should only be used when the
manifold is parametrized by a convex region of Euclidean space. We argue that
this perception is based on an extremely narrow interpretation of manifold
learning as parametrization recovery, and we submit that Isomap is better
understood as constructing Euclidean representations of geodesic structure. We
reconsider a well-known example that was previously interpreted as evidence of
Isomap's limitations, and we re-examine the original analysis of Isomap's
convergence properties, concluding that convexity is not required for shortest
path distances to converge to Riemannian distances.

    

### [[2006.15138] Distributed Uplink Beamforming in Cell-Free Networks Using Deep Reinforcement Learning](http://arxiv.org/abs/2006.15138)


  The emergence of new wireless technologies together with the requirement of
massive connectivity results in several technical issues such as excessive
interference, high computational demand for signal processing, and lengthy
processing delays. In this work, we propose several beamforming techniques for
an uplink cell-free network with centralized, semi-distributed, and fully
distributed processing, all based on deep reinforcement learning (DRL). First,
we propose a fully centralized beamforming method that uses the deep
deterministic policy gradient algorithm (DDPG) with continuous space. We then
enhance this method by enabling distributed experience at access points (AP).
Indeed, we develop a beamforming scheme that uses the distributed
distributional deterministic policy gradients algorithm (D4PG) with the APs
representing the distributed agents. Finally, to decrease the computational
complexity, we propose a fully distributed beamforming scheme that divides the
beamforming computations among APs. The results show that the D4PG scheme with
distributed experience achieves the best performance irrespective of the
network size. Furthermore, the proposed distributed beamforming technique
performs better than the DDPG algorithm with centralized learning only for
small-scale networks. The performance superiority of the DDPG model becomes
more evident as the number of APs and/or users increases. Moreover, during the
operation stage, all DRL models demonstrate a significantly shorter processing
time than that of the conventional gradient descent (GD) solution.

    

### [[2007.00514] Regularized Online Allocation Problems: Fairness and Beyond](http://arxiv.org/abs/2007.00514)


  Online allocation problems with resource constraints have a rich history in
operations research. In this paper, we introduce the \emph{regularized online
allocation problem}, a variant that includes a non-linear regularizer acting on
the total resource consumption. In this problem, requests repeatedly arrive
over time and, for each request, a decision maker needs to take an action that
generates a reward and consumes resources. The objective is to simultaneously
maximize additively separable rewards and the value of a non-separable
regularizer subject to the resource constraints. Our primary motivation is
allowing decision makers to trade off separable objectives such as the economic
efficiency of an allocation with ancillary, non-separable objectives such as
the fairness or equity of an allocation. We design an algorithm that is simple,
fast, and attains good performance with both stochastic i.i.d.~and adversarial
inputs. In particular, our algorithm is asymptotically optimal under stochastic
i.i.d. input models and attains a fixed competitive ratio that depends on the
regularizer when the input is adversarial. Furthermore, the algorithm and
analysis do not require convexity or concavity of the reward function and the
consumption function, which allows more model flexibility. Numerical
experiments confirm the effectiveness of the proposed algorithm and of
regularization in an internet advertising application.

    

### [[2010.03106] Structured Logconcave Sampling with a Restricted Gaussian Oracle](http://arxiv.org/abs/2010.03106)


  We give algorithms for sampling several structured logconcave families to
high accuracy. We further develop a reduction framework, inspired by proximal
point methods in convex optimization, which bootstraps samplers for regularized
densities to improve dependences on problem conditioning. A key ingredient in
our framework is the notion of a "restricted Gaussian oracle" (RGO) for $g:
\mathbb{R}^d \rightarrow \mathbb{R}$, which is a sampler for distributions
whose negative log-likelihood sums a quadratic and $g$. By combining our
reduction framework with our new samplers, we obtain the following bounds for
sampling structured distributions to total variation distance $\epsilon$. For
composite densities $\exp(-f(x) - g(x))$, where $f$ has condition number
$\kappa$ and convex (but possibly non-smooth) $g$ admits an RGO, we obtain a
mixing time of $O(\kappa d \log^3\frac{\kappa d}{\epsilon})$, matching the
state-of-the-art non-composite bound; no composite samplers with better mixing
than general-purpose logconcave samplers were previously known. For logconcave
finite sums $\exp(-F(x))$, where $F(x) = \frac{1}{n}\sum_{i \in [n]} f_i(x)$
has condition number $\kappa$, we give a sampler querying $\widetilde{O}(n +
\kappa\max(d, \sqrt{nd}))$ gradient oracles to $\{f_i\}_{i \in [n]}$; no
high-accuracy samplers with nontrivial gradient query complexity were
previously known. For densities with condition number $\kappa$, we give an
algorithm obtaining mixing time $O(\kappa d \log^2\frac{\kappa d}{\epsilon})$,
improving the prior state-of-the-art by a logarithmic factor with a
significantly simpler analysis; we also show a zeroth-order algorithm attains
the same query complexity.

    

### [[2010.11289] Shedding Light on Blind Spots: Developing a Reference Architecture to Leverage Video Data for Process Mining](http://arxiv.org/abs/2010.11289)


  Process mining is one of the most active research streams in business process
management. In recent years, numerous methods have been proposed for analyzing
structured process data. Yet, in many cases, it is only the digitized parts of
processes that are directly captured from process-aware information systems,
and manual activities often result in blind spots. While the use of video
cameras to observe these activities could help to fill this gap, a standardized
approach to extracting event logs from unstructured video data remains lacking.
Here, we propose a reference architecture to bridge the gap between computer
vision and process mining. Various evaluation activities (i.e., competing
artifact analysis, prototyping, and real-world application) ensured that the
proposed reference architecture allows flexible, use-case-driven, and
context-specific instantiations. Our results also show that an exemplary
software prototype instantiation of the proposed reference architecture is
capable of automatically extracting most of the process-relevant events from
unstructured video data.

    

### [[2011.00580] Sparsity-Control Ternary Weight Networks](http://arxiv.org/abs/2011.00580)


  Deep neural networks (DNNs) have been widely and successfully applied to
various applications, but they require large amounts of memory and
computational power. This severely restricts their deployment on
resource-limited devices. To address this issue, many efforts have been made on
training low-bit weight DNNs. In this paper, we focus on training ternary
weight \{-1, 0, +1\} networks which can avoid multiplications and dramatically
reduce the memory and computation requirements. A ternary weight network can be
considered as a sparser version of the binary weight counterpart by replacing
some -1s or 1s in the binary weights with 0s, thus leading to more efficient
inference but more memory cost. However, the existing approaches to training
ternary weight networks cannot control the sparsity (i.e., percentage of 0s) of
the ternary weights, which undermines the advantage of ternary weights. In this
paper, we propose to our best knowledge the first sparsity-control approach
(SCA) to training ternary weight networks, which is simply achieved by a weight
discretization regularizer (WDR). SCA is different from all the existing
regularizer-based approaches in that it can control the sparsity of the ternary
weights through a controller $\alpha$ and does not rely on gradient estimators.
We theoretically and empirically show that the sparsity of the trained ternary
weights is positively related to $\alpha$. SCA is extremely simple,
easy-to-implement, and is shown to consistently outperform the state-of-the-art
approaches significantly over several benchmark datasets and even matches the
performances of the full-precision weight counterparts.

    

### [[2011.08545] Dynamic Hard Pruning of Neural Networks at the Edge of the Internet](http://arxiv.org/abs/2011.08545)


  Neural Networks (NN), although successfully applied to several Artificial
Intelligence tasks, are often unnecessarily over-parametrised. In edge/fog
computing, this might make their training prohibitive on resource-constrained
devices, contrasting with the current trend of decentralising intelligence from
remote data centres to local constrained devices. Therefore, we investigate the
problem of training effective NN models on constrained devices having a fixed,
potentially small, memory budget. We target techniques that are both
resource-efficient and performance effective while enabling significant network
compression. Our Dynamic Hard Pruning (DynHP) technique incrementally prunes
the network during training, identifying neurons that marginally contribute to
the model accuracy. DynHP enables a tunable size reduction of the final neural
network and reduces the NN memory occupancy during training. Freed memory is
reused by a \emph{dynamic batch sizing} approach to counterbalance the accuracy
degradation caused by the hard pruning strategy, improving its convergence and
effectiveness. We assess the performance of DynHP through reproducible
experiments on three public datasets, comparing them against reference
competitors. Results show that DynHP compresses a NN up to $10$ times without
significant performance drops (up to $3.5\%$ additional error w.r.t. the
competitors), reducing up to $80\%$ the training memory occupancy.

    

### [[2012.02679] What is a meaningful representation of protein sequences?](http://arxiv.org/abs/2012.02679)


  How we choose to represent our data has a fundamental impact on our ability
to subsequently extract information from them. Machine learning promises to
automatically determine efficient representations from large unstructured
datasets, such as those arising in biology. However, empirical evidence
suggests that seemingly minor changes to these machine learning models yield
drastically different data representations that result in different biological
interpretations of data. This begs the question of what even constitutes the
most meaningful representation. Here, we approach this question for
representations of protein sequences, which have received considerable
attention in the recent literature. We explore two key contexts in which
representations naturally arise: transfer learning and interpretable learning.
In the first context, we demonstrate that several contemporary practices yield
suboptimal performance, and in the latter we demonstrate that taking
representation geometry into account significantly improves interpretability
and lets the models reveal biological information that is otherwise obscured.

    

### [[2012.03790] Matching Distributions via Optimal Transport for Semi-Supervised Learning](http://arxiv.org/abs/2012.03790)


  Semi-Supervised Learning (SSL) approaches have been an influential framework
for the usage of unlabeled data when there is not a sufficient amount of
labeled data available over the course of training. SSL methods based on
Convolutional Neural Networks (CNNs) have recently provided successful results
on standard benchmark tasks such as image classification. In this work, we
consider the general setting of SSL problem where the labeled and unlabeled
data come from the same underlying probability distribution. We propose a new
approach that adopts an Optimal Transport (OT) technique serving as a metric of
similarity between discrete empirical probability measures to provide
pseudo-labels for the unlabeled data, which can then be used in conjunction
with the initial labeled data to train the CNN model in an SSL manner. We have
evaluated and compared our proposed method with state-of-the-art SSL algorithms
on standard datasets to demonstrate the superiority and effectiveness of our
SSL algorithm.

    

### [[2012.14906] Synthesizing Decentralized Controllers with Graph Neural Networks and Imitation Learning](http://arxiv.org/abs/2012.14906)


  Dynamical systems consisting of a set of autonomous agents face the challenge
of having to accomplish a global task, relying only on local information. While
centralized controllers are readily available, they face limitations in terms
of scalability and implementation, as they do not respect the distributed
information structure imposed by the network system of agents. Given the
difficulties in finding optimal decentralized controllers, we propose a novel
framework using graph neural networks (GNNs) to learn these controllers. GNNs
are well-suited for the task since they are naturally distributed architectures
and exhibit good scalability and transferability properties. We show that GNNs
learn appropriate decentralized controllers by means of imitation learning,
leverage their permutation invariance properties to successfully scale to
larger teams and transfer to unseen scenarios at deployment time. The problems
of flocking and multi-agent path planning are explored to illustrate the
potential of GNNs in learning decentralized controllers.

    

### [[2101.01881] MSD: Saliency-aware Knowledge Distillation for Multimodal Understanding](http://arxiv.org/abs/2101.01881)


  To reduce a model size but retain performance, we often rely on knowledge
distillation (KD) which transfers knowledge from a large "teacher" model to a
smaller "student" model. However, KD on multimodal datasets such as
vision-language tasks is relatively unexplored, and digesting multimodal
information is challenging since different modalities present different types
of information. In this paper, we perform a large-scale empirical study to
investigate the importance and effects of each modality in knowledge
distillation. Furthermore, we introduce a multimodal knowledge distillation
framework, modality-specific distillation (MSD), to transfer knowledge from a
teacher on multimodal tasks by learning the teacher's behavior within each
modality. The idea aims at mimicking a teacher's modality-specific predictions
by introducing auxiliary loss terms for each modality. Furthermore, because
each modality has different saliency for predictions, we define saliency scores
for each modality and investigate saliency-based weighting schemes for the
auxiliary losses. We further study a weight learning approach to learn the
optimal weights on these loss terms. In our empirical analysis, we examine the
saliency of each modality in KD, demonstrate the effectiveness of the weighting
scheme in MSD, and show that it achieves better performance than KD on four
multimodal datasets.

    

### [[2101.10027] Understanding and Achieving Efficient Robustness with Adversarial Supervised Contrastive Learning](http://arxiv.org/abs/2101.10027)


  Contrastive learning (CL) has recently emerged as an effective approach to
learning representation in a range of downstream tasks. Central to this
approach is the selection of positive (similar) and negative (dissimilar) sets
to provide the model the opportunity to `contrast' between data and class
representation in the latent space. In this paper, we investigate CL for
improving model robustness using adversarial samples. We first designed and
performed a comprehensive study to understand how adversarial vulnerability
behaves in the latent space. Based on this empirical evidence, we propose an
effective and efficient supervised contrastive learning to achieve model
robustness against adversarial attacks. Moreover, we propose a new sample
selection strategy that optimizes the positive/negative sets by removing
redundancy and improving correlation with the anchor. Extensive experiments
show that our Adversarial Supervised Contrastive Learning (ASCL) approach
achieves comparable performance with the state-of-the-art defenses while
significantly outperforms other CL-based defense methods by using only $42.8\%$
positives and $6.3\%$ negatives.

    

### [[2102.03509] Robust normalizing flows using Bernstein-type polynomials](http://arxiv.org/abs/2102.03509)


  Modeling real-world distributions can often be challenging due to sample data
that are subjected to perturbations, e.g., instrumentation errors, or added
random noise. Since flow models are typically nonlinear algorithms, they
amplify these initial errors, leading to poor generalizations. This paper
proposes a framework to construct Normalizing Flows (NF), which demonstrates
higher robustness against such initial errors. To this end, we utilize
Bernstein-type polynomials inspired by the optimal stability of the Bernstein
basis. Further, compared to the existing NF frameworks, our method provides
compelling advantages like theoretical upper bounds for the approximation
error, higher interpretability, suitability for compactly supported densities,
and the ability to employ higher degree polynomials without training
instability. We conduct a thorough theoretical analysis and empirically
demonstrate the efficacy of the proposed technique using experiments on both
real-world and synthetic datasets.

    

### [[2102.03824] Neural Termination Analysis](http://arxiv.org/abs/2102.03824)


  We introduce a novel approach to the automated termination analysis of
computer programs: we train neural networks to behave as ranking functions.
Ranking functions map program states to values that are bounded from below and
decrease as the program runs. The existence of a valid ranking function proves
that the program terminates. While existing methods usually construct ranking
functions from source or machine code using symbolic reasoning, we propose a
lightweight method that learns them from executions traces. We train a neural
network so that its output decreases along sampled executions as a ranking
function would; then, we use symbolic reasoning to verify whether it
generalises to all possible executions. We demonstrate that, thanks to the
ability of neural networks to generalise well, our method succeeds over a wide
variety of programs. This includes programs that use data structures. We have
built a prototype analyser for Java bytecode and show the efficacy of our
method over a standard dataset of benchmarks.

    

### [[2102.03868] U-vectors: Generating clusterable speaker embedding from unlabeled data](http://arxiv.org/abs/2102.03868)


  Speaker recognition deals with recognizing speakers by their speech. Most
speaker recognition systems are built upon two stages, the first stage extracts
low dimensional correlation embeddings from speech, and the second performs the
classification task. The robustness of a speaker recognition system mainly
depends on the extraction process of speech embeddings, which are primarily
pre-trained on a large-scale dataset. As the embedding systems are pre-trained,
the performance of speaker recognition models greatly depends on domain
adaptation policy, which may reduce if trained using inadequate data. This
paper introduces a speaker recognition strategy dealing with unlabeled data,
which generates clusterable embedding vectors from small fixed-size speech
frames. The unsupervised training strategy involves an assumption that a small
speech segment should include a single speaker. Depending on such a belief, a
pairwise constraint is constructed with noise augmentation policies, used to
train AutoEmbedder architecture that generates speaker embeddings. Without
relying on domain adaption policy, the process unsupervisely produces
clusterable speaker embeddings, termed unsupervised vectors (u-vectors). The
evaluation is concluded in two popular speaker recognition datasets for English
language, TIMIT, and LibriSpeech. Also, a Bengali dataset is included to
illustrate the diversity of the domain shifts for speaker recognition systems.
Finally, we conclude that the proposed approach achieves satisfactory
performance using pairwise architectures.

    

### [[2102.05379] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](http://arxiv.org/abs/2102.05379)


  Generative flows and diffusion models have been predominantly trained on
ordinal data, for example natural images. This paper introduces two extensions
of flows and diffusion for categorical data such as language or image
segmentation: Argmax Flows and Multinomial Diffusion. Argmax Flows are defined
by a composition of a continuous distribution (such as a normalizing flow), and
an argmax function. To optimize this model, we learn a probabilistic inverse
for the argmax that lifts the categorical data to a continuous space.
Multinomial Diffusion gradually adds categorical noise in a diffusion process,
for which the generative denoising process is learned. We demonstrate that our
method outperforms existing dequantization approaches on text modelling and
modelling on image segmentation maps in log-likelihood.

    

### [[2102.06289] When and How Mixup Improves Calibration](http://arxiv.org/abs/2102.06289)


  In many machine learning applications, it is important for the model to
provide confidence scores that accurately capture its prediction uncertainty.
Although modern learning methods have achieved great success in predictive
accuracy, generating calibrated confidence scores remains a major challenge.
Mixup, a popular yet simple data augmentation technique based on taking convex
combinations of pairs of training examples, has been empirically found to
significantly improve confidence calibration across diverse applications.
However, when and how Mixup helps calibration is still a mystery. In this
paper, we theoretically prove that Mixup improves calibration in
\textit{high-dimensional} settings by investigating natural statistical models.
Interestingly, the calibration benefit of Mixup increases as the model capacity
increases. We support our theories with experiments on common architectures and
datasets. In addition, we study how Mixup improves calibration in
semi-supervised learning. While incorporating unlabeled data can sometimes make
the model less calibrated, adding Mixup training mitigates this issue and
provably improves calibration. Our analysis provides new insights and a
framework to understand Mixup and calibration.

    

### [[2102.06665] Bayesian Uncertainty Estimation of Learned Variational MRI Reconstruction](http://arxiv.org/abs/2102.06665)


  Recent deep learning approaches focus on improving quantitative scores of
dedicated benchmarks, and therefore only reduce the observation-related
(aleatoric) uncertainty. However, the model-immanent (epistemic) uncertainty is
less frequently systematically analyzed. In this work, we introduce a Bayesian
variational framework to quantify the epistemic uncertainty. To this end, we
solve the linear inverse problem of undersampled MRI reconstruction in a
variational setting. The associated energy functional is composed of a data
fidelity term and the total deep variation (TDV) as a learned parametric
regularizer. To estimate the epistemic uncertainty we draw the parameters of
the TDV regularizer from a multivariate Gaussian distribution, whose mean and
covariance matrix are learned in a stochastic optimal control problem. In
several numerical experiments, we demonstrate that our approach yields
competitive results for undersampled MRI reconstruction. Moreover, we can
accurately quantify the pixelwise epistemic uncertainty, which can serve
radiologists as an additional resource to visualize reconstruction reliability.

    

### [[2102.06679] Adversarial Branch Architecture Search for Unsupervised Domain Adaptation](http://arxiv.org/abs/2102.06679)


  Unsupervised Domain Adaptation (UDA) is a key issue in visual recognition, as
it allows to bridge different visual domains enabling robust performances in
the real world. To date, all proposed approaches rely on human expertise to
manually adapt a given UDA method (e.g. DANN) to a specific backbone
architecture (e.g. ResNet). This dependency on handcrafted designs limits the
applicability of a given approach in time, as old methods need to be constantly
adapted to novel backbones.
Existing Neural Architecture Search (NAS) approaches cannot be directly
applied to mitigate this issue, as they rely on labels that are not available
in the UDA setting. Furthermore, most NAS methods search for full
architectures, which precludes the use of pre-trained models, essential in a
vast range of UDA settings for reaching SOTA results. To the best of our
knowledge, no prior work has addressed these aspects in the context of NAS for
UDA. Here we tackle both aspects with an Adversarial Branch Architecture Search
for UDA (ABAS): i. we address the lack of target labels by a novel data-driven
ensemble approach for model selection; and ii. we search for an auxiliary
adversarial branch, attached to a pre-trained backbone, which drives the domain
alignment.
We extensively validate ABAS to improve two modern UDA techniques, DANN and
ALDA, on three standard visual recognition datasets (Office31, Office-Home and
PACS). In all cases, ABAS robustly finds the adversarial branch architectures
and parameters which yield best performances.

    

### [[2102.12894] Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes](http://arxiv.org/abs/2102.12894)


  Deep neural networks (DNNs) are notorious for making more mistakes for the
classes that have substantially fewer samples than the others during training.
Such class imbalance is ubiquitous in clinical applications and very crucial to
handle because the classes with fewer samples most often correspond to critical
cases (e.g., cancer) where misclassifications can have severe consequences. Not
to miss such cases, binary classifiers need to be operated at high True
Positive Rates (TPRs) by setting a higher threshold, but this comes at the cost
of very high False Positive Rates (FPRs) for problems with class imbalance.
Existing methods for learning under class imbalance most often do not take this
into account. We argue that prediction accuracy should be improved by
emphasizing reducing FPRs at high TPRs for problems where misclassification of
the positive, i.e. critical, class samples are associated with higher cost. To
this end, we pose the training of a DNN for binary classification as a
constrained optimization problem and introduce a novel constraint that can be
used with existing loss functions to enforce maximal area under the ROC curve
(AUC) through prioritizing FPR reduction at high TPR. We solve the resulting
constrained optimization problem using an Augmented Lagrangian method (ALM).
Going beyond binary, we also propose two possible extensions of the proposed
constraint for multi-class classification problems. We present experimental
results for image-based binary and multi-class classification applications
using an in-house medical imaging dataset, CIFAR10, and CIFAR100. Our results
demonstrate that the proposed method improves the baselines in majority of the
cases by attaining higher accuracy on critical classes while reducing the
misclassification rate for the non-critical class samples.

    

### [[2104.13030] A Survey on Neural Recommendation: From Collaborative Filtering to Information-rich Recommendation](http://arxiv.org/abs/2104.13030)


  Influenced by the great success of deep learning in computer vision and
language understanding, research in recommendation has shifted to inventing new
recommender models based on neural networks. In recent years, we have witnessed
significant progress in developing neural recommender models, which generalize
and surpass traditional recommender models owing to the strong representation
power of neural networks. In this survey paper, we conduct a systematic review
on neural recommender models from the perspective of recommendation modeling
with the accuracy goal, aiming to summarize this field to facilitate
researchers and practitioners working on recommender systems. Specifically,
based on the data usage during recommendation modeling we divide the work into
collaborative filtering and information-rich recommendation: 1) collaborative
filtering, which leverages the key source of user-item interaction data; 2)
content enriched recommendation, which additionally utilizes the side
information associated with users and items, like user profile and item
knowledge graph; and 3) temporal/sequential recommendation, which accounts for
the contextual information associated with an interaction, such as time,
location, and the past interactions. After reviewing representative work for
each type, we finally discuss some promising directions in this field. We have
also summarized the related papers at
this https URL.

    

### [[2105.12806] A Universal Law of Robustness via Isoperimetry](http://arxiv.org/abs/2105.12806)


  Classically, data interpolation with a parametrized model class is possible
as long as the number of parameters is larger than the number of equations to
be satisfied. A puzzling phenomenon in deep learning is that models are trained
with many more parameters than what this classical theory would suggest. We
propose a theoretical explanation for this phenomenon. We prove that for a
broad class of data distributions and model classes, overparametrization is
necessary if one wants to interpolate the data smoothly. Namely we show that
smooth interpolation requires $d$ times more parameters than mere
interpolation, where $d$ is the ambient data dimension. We prove this universal
law of robustness for any smoothly parametrized function class with polynomial
size weights, and any covariate distribution verifying isoperimetry. In the
case of two-layers neural networks and Gaussian covariates, this law was
conjectured in prior work by Bubeck, Li and Nagaraj. We also give an
interpretation of our result as an improved generalization bound for model
classes consisting of smooth functions.

    

### [[2106.00797] QLSD: Quantised Langevin stochastic dynamics for Bayesian federated learning](http://arxiv.org/abs/2106.00797)


  The objective of Federated Learning (FL) is to perform statistical inference
for data which are decentralised and stored locally on networked clients. FL
raises many constraints which include privacy and data ownership, communication
overhead, statistical heterogeneity, and partial client participation. In this
paper, we address these problems in the framework of the Bayesian paradigm. To
this end, we propose a novel federated Markov Chain Monte Carlo algorithm,
referred to as Quantised Langevin Stochastic Dynamics which may be seen as an
extension to the FL setting of Stochastic Gradient Langevin Dynamics, which
handles the communication bottleneck using gradient compression. To improve
performance, we then introduce variance reduction techniques, which lead to two
improved versions coined \texttt{QLSD}$^{\star}$ and \texttt{QLSD}$^{++}$. We
give both non-asymptotic and asymptotic convergence guarantees for the proposed
algorithms. We illustrate their performances using various Bayesian Federated
Learning benchmarks.

    

### [[2106.02393] Multitask Online Mirror Descent](http://arxiv.org/abs/2106.02393)


  We introduce and analyze MT-OMD, a multitask generalization of Online Mirror
Descent (OMD) which operates by sharing updates between tasks. We prove that
the regret of MT-OMD is of order $\sqrt{1 + \sigma^2(N-1)}\sqrt{T}$, where
$\sigma^2$ is the task variance according to the geometry induced by the
regularizer, $N$ is the number of tasks, and $T$ is the time horizon. Whenever
tasks are similar, that is $\sigma^2 \le 1$, our method improves upon the
$\sqrt{NT}$ bound obtained by running independent OMDs on each task. We further
provide a matching lower bound, and show that our multitask extensions of
Online Gradient Descent and Exponentiated Gradient, two major instances of OMD,
enjoy closed-form updates, making them easy to use in practice. Finally, we
present experiments on both synthetic and real-world datasets supporting our
findings.

    

### [[2106.02636] MERLOT: Multimodal Neural Script Knowledge Models](http://arxiv.org/abs/2106.02636)


  As humans, we understand events in the visual world contextually, performing
multimodal reasoning across time to make inferences about the past, present,
and future. We introduce MERLOT, a model that learns multimodal script
knowledge by watching millions of YouTube videos with transcribed speech -- in
an entirely label-free, self-supervised manner. By pretraining with a mix of
both frame-level (spatial) and video-level (temporal) objectives, our model not
only learns to match images to temporally corresponding words, but also to
contextualize what is happening globally over time. As a result, MERLOT
exhibits strong out-of-the-box representations of temporal commonsense, and
achieves state-of-the-art performance on 12 different video QA datasets when
finetuned. It also transfers well to the world of static images, allowing
models to reason about the dynamic context behind visual scenes. On Visual
Commonsense Reasoning, MERLOT answers questions correctly with 80.6% accuracy,
outperforming state-of-the-art models of similar size by over 3%, even those
that make heavy use of auxiliary supervised data (like object bounding boxes).
Ablation analyses demonstrate the complementary importance of: 1) training on
videos versus static images; 2) scaling the magnitude and diversity of the
pretraining video corpus; and 3) using diverse objectives that encourage
full-stack multimodal reasoning, from the recognition to cognition level.

    

### [[2106.07769] The Flip Side of the Reweighted Coin: Duality of Adaptive Dropout and Regularization](http://arxiv.org/abs/2106.07769)


  Among the most successful methods for sparsifying deep (neural) networks are
those that adaptively mask the network weights throughout training. By
examining this masking, or dropout, in the linear case, we uncover a duality
between such adaptive methods and regularization through the so-called
"$\eta$-trick" that casts both as iteratively reweighted optimizations. We show
that any dropout strategy that adapts to the weights in a monotonic way
corresponds to an effective subquadratic regularization penalty, and therefore
leads to sparse solutions. We obtain the effective penalties for several
popular sparsification strategies, which are remarkably similar to classical
penalties commonly used in sparse optimization. Considering variational dropout
as a case study, we demonstrate similar empirical behavior between the adaptive
dropout method and classical methods on the task of deep network
sparsification, validating our theory.

    

### [[2106.09553] Do Large Scale Molecular Language Representations Capture Important Structural Information?](http://arxiv.org/abs/2106.09553)


  Predicting the chemical properties of a molecule is of great importance in
many applications, including drug discovery and material design. Machine
learning based molecular property prediction holds the promise of enabling
accurate predictions at much less computationally complex cost when compared
to, for example, Density Functional Theory (DFT) calculations. Various
representation learning methods in a supervised setting, including the features
extracted using graph neural nets, have emerged for such tasks. However, the
vast chemical space and the limited availability of labels make supervised
learning challenging, calling for learning a general-purpose molecular
representation. Recently, pre-trained transformer-based language models on
large unlabeled corpus have produced state-of-the-art results in many
downstream natural language processing tasks. Inspired by this development, we
present molecular embeddings obtained by training an efficient transformer
encoder model, MoLFormer. This model employs a linear attention mechanism
coupled with highly parallelized training on SMILES sequences of 1.1 billion
unlabeled molecules from the PubChem and ZINC datasets. Experiments show that
the learned molecular representation outperforms supervised and unsupervised
graph neural net baselines on several regression and classification tasks from
10 benchmark datasets, while performing competitively on others. Further
analyses, specifically through the lens of attention, demonstrate that
MoLFormer indeed learns a molecule's local and global structural aspects. These
results provide encouraging evidence that large-scale molecular language models
can capture sufficient structural information to be able to predict diverse
molecular properties, including quantum-chemical properties

    

### [[2106.09708] Multi-Label Learning from Single Positive Labels](http://arxiv.org/abs/2106.09708)


  Predicting all applicable labels for a given image is known as multi-label
classification. Compared to the standard multi-class case (where each image has
only one label), it is considerably more challenging to annotate training data
for multi-label classification. When the number of potential labels is large,
human annotators find it difficult to mention all applicable labels for each
training image. Furthermore, in some settings detection is intrinsically
difficult e.g. finding small object instances in high resolution images. As a
result, multi-label training data is often plagued by false negatives. We
consider the hardest version of this problem, where annotators provide only one
relevant label for each image. As a result, training sets will have only one
positive label per image and no confirmed negatives. We explore this special
case of learning from missing labels across four different multi-label image
classification datasets for both linear classifiers and end-to-end fine-tuned
deep networks. We extend existing multi-label losses to this setting and
propose novel variants that constrain the number of expected positive labels
during training. Surprisingly, we show that in some cases it is possible to
approach the performance of fully labeled classifiers despite training with
significantly fewer confirmed labels.

    

### [[2106.13082] On the relationship between predictive coding and backpropagation](http://arxiv.org/abs/2106.13082)


  Artificial neural networks are often interpreted as abstract models of
biological neuronal networks, but they are typically trained using the
biologically unrealistic backpropagation algorithm and its variants. Predictive
coding has been offered as a potentially more biologically realistic
alternative to backpropagation for training neural networks. In this
manuscript, I review and extend recent work on the mathematical relationship
between predictive coding and backpropagation for training feedforward
artificial neural networks on supervised learning tasks. I discuss some
implications of these results for the interpretation of predictive coding and
deep neural networks as models of biological learning and I describe a
repository of functions, Torch2PC, for performing predictive coding with
PyTorch neural network models.

    

### [[2109.12825] Probability Distribution on Full Rooted Trees](http://arxiv.org/abs/2109.12825)


  The recursive and hierarchical structure of full rooted trees is applicable
to represent statistical models in various areas, such as data compression,
image processing, and machine learning. In most of these cases, the full rooted
tree is not a random variable; as such, model selection to avoid overfitting
becomes problematic. A method to solve this problem is to assume a prior
distribution on the full rooted trees. This enables overfitting to be avoided
based on the Bayes decision theory. For example, by assigning a low prior
probability to a complex model, the maximum a posteriori estimator prevents
overfitting. Furthermore, overfitting can be avoided by averaging all the
models weighted by their posteriors. In this paper, we propose a probability
distribution on a set of full rooted trees. Its parametric representation is
suitable for calculating the properties of our distribution using recursive
functions, such as the mode, expectation, and posterior distribution. Although
such distributions have been proposed in previous studies, they are only
applicable to specific applications. Therefore, we extract their mathematically
essential components and derive new generalized methods to calculate the
expectation, posterior distribution, etc.

    

### [[2110.05035] Using Personality Detection Tools for Software Engineering Research: How Far Can We Go?](http://arxiv.org/abs/2110.05035)


  Assessing the personality of software engineers may help to match individual
traits with the characteristics of development activities such as code review
and testing, as well as support managers in team composition. However,
self-assessment questionnaires are not a practical solution for collecting
multiple observations on a large scale. Instead, automatic personality
detection, while overcoming these limitations, is based on off-the-shelf
solutions trained on non-technical corpora, which might not be readily
applicable to technical domains like Software Engineering (SE). In this paper,
we first assess the performance of general-purpose personality detection tools
when applied to a technical corpus of developers' emails retrieved from the
public archives of the Apache Software Foundation. We observe a general low
accuracy of predictions and an overall disagreement among the tools. Second, we
replicate two previous research studies in SE by replacing the personality
detection tool used to infer developers' personalities from pull-request
discussions and emails. We observe that the original results are not confirmed,
i.e., changing the tool used in the original study leads to diverging
conclusions. Our results suggest a need for personality detection tools
specially targeted for the software engineering domain.

    

### [[2110.11088] RoMA: a Method for Neural Network Robustness Measurement and Assessment](http://arxiv.org/abs/2110.11088)


  Neural network models have become the leading solution for a large variety of
tasks, such as classification, language processing, protein folding, and
others. However, their reliability is heavily plagued by adversarial inputs:
small input perturbations that cause the model to produce erroneous outputs.
Adversarial inputs can occur naturally when the system's environment behaves
randomly, even in the absence of a malicious adversary, and are a severe cause
for concern when attempting to deploy neural networks within critical systems.
In this paper, we present a new statistical method, called Robustness
Measurement and Assessment (RoMA), which can measure the expected robustness of
a neural network model. Specifically, RoMA determines the probability that a
random input perturbation might cause misclassification. The method allows us
to provide formal guarantees regarding the expected frequency of errors that a
trained model will encounter after deployment. Our approach can be applied to
large-scale, black-box neural networks, which is a significant advantage
compared to recently proposed verification methods. We apply our approach in
two ways: comparing the robustness of different models, and measuring how a
model's robustness is affected by the magnitude of input perturbation. One
interesting insight obtained through this work is that, in a classification
network, different output labels can exhibit very different robustness levels.
We term this phenomenon categorial robustness. Our ability to perform risk and
robustness assessments on a categorial basis opens the door to risk mitigation,
which may prove to be a significant step towards neural network certification
in safety-critical applications.

    

### [[2110.11154] Personalized Transfer of User Preferences for Cross-domain Recommendation](http://arxiv.org/abs/2110.11154)


  Cold-start problem is still a very challenging problem in recommender
systems. Fortunately, the interactions of the cold-start users in the auxiliary
source domain can help cold-start recommendations in the target domain. How to
transfer user's preferences from the source domain to the target domain, is the
key issue in Cross-domain Recommendation (CDR) which is a promising solution to
deal with the cold-start problem. Most existing methods model a common
preference bridge to transfer preferences for all users. Intuitively, since
preferences vary from user to user, the preference bridges of different users
should be different. Along this line, we propose a novel framework named
Personalized Transfer of User Preferences for Cross-domain Recommendation
(PTUPCDR). Specifically, a meta network fed with users' characteristic
embeddings is learned to generate personalized bridge functions to achieve
personalized transfer of preferences for each user. To learn the meta network
stably, we employ a task-oriented optimization procedure. With the
meta-generated personalized bridge function, the user's preference embedding in
the source domain can be transformed into the target domain, and the
transformed user preference embedding can be utilized as the initial embedding
for the cold-start user in the target domain. Using large real-world datasets,
we conduct extensive experiments to evaluate the effectiveness of PTUPCDR on
both cold-start and warm-start stages. The code has been available at
\url{this https URL}.

    

### [[2110.11191] Generative Adversarial Graph Convolutional Networks for Human Action Synthesis](http://arxiv.org/abs/2110.11191)


  Synthesising the spatial and temporal dynamics of the human body skeleton
remains a challenging task, not only in terms of the quality of the generated
shapes, but also of their diversity, particularly to synthesise realistic body
movements of a specific action (action conditioning). In this paper, we propose
Kinetic-GAN, a novel architecture that leverages the benefits of Generative
Adversarial Networks and Graph Convolutional Networks to synthesise the
kinetics of the human body. The proposed adversarial architecture can condition
up to 120 different actions over local and global body movements while
improving sample quality and diversity through latent space disentanglement and
stochastic variations. Our experiments were carried out in three well-known
datasets, where Kinetic-GAN notably surpasses the state-of-the-art methods in
terms of distribution quality metrics while having the ability to synthesise
more than one order of magnitude regarding the number of different actions. Our
code and models are publicly available at
this https URL.

    

### [[2110.11281] Super-resolution of multiphase materials by combining complementary 2D and 3D image data using generative adversarial networks](http://arxiv.org/abs/2110.11281)


  Modelling the impact of a material's mesostructure on device level
performance typically requires access to 3D image data containing all the
relevant information to define the geometry of the simulation domain. This
image data must include sufficient contrast between phases to distinguish each
material, be of high enough resolution to capture the key details, but also
have a large enough field-of-view to be representative of the material in
general. It is rarely possible to obtain data with all of these properties from
a single imaging technique. In this paper, we present a method for combining
information from pairs of distinct but complementary imaging techniques in
order to accurately reconstruct the desired multi-phase, high resolution,
representative, 3D images. Specifically, we use deep convolutional generative
adversarial networks to implement super-resolution, style transfer and
dimensionality expansion. To demonstrate the widespread applicability of this
tool, two pairs of datasets are used to validate the quality of the volumes
generated by fusing the information from paired imaging techniques. Three key
mesostructural metrics are calculated in each case to show the accuracy of this
method. Having confidence in the accuracy of our method, we then demonstrate
its power by applying to a real data pair from a lithium ion battery electrode,
where the required 3D high resolution image data is not available anywhere in
the literature. We believe this approach is superior to previously reported
statistical material reconstruction methods both in terms of its fidelity and
ease of use. Furthermore, much of the data required to train this algorithm
already exists in the literature, waiting to be combined. As such, our
open-access code could precipitate a step change by generating the hard to
obtain high quality image volumes necessary to simulate behaviour at the
mesoscale.

    

### [[2110.11182] SLURP: Side Learning Uncertainty for Regression Problems](http://arxiv.org/abs/2110.11182)


  It has become critical for deep learning algorithms to quantify their output
uncertainties to satisfy reliability constraints and provide accurate results.
Uncertainty estimation for regression has received less attention than
classification due to the more straightforward standardized output of the
latter class of tasks and their high importance. However, regression problems
are encountered in a wide range of applications in computer vision. We propose
SLURP, a generic approach for regression uncertainty estimation via a side
learner that exploits the output and the intermediate representations generated
by the main task model. We test SLURP on two critical regression tasks in
computer vision: monocular depth and optical flow estimation. In addition, we
conduct exhaustive benchmarks comprising transfer to different datasets and the
addition of aleatoric noise. The results show that our proposal is generic and
readily applicable to various regression problems and has a low computational
cost with respect to existing solutions.

    

### [[2110.11504] Maximum Power Point Tracking Circuit for an Energy Harvester in 130 nm CMOS Technology](http://arxiv.org/abs/2110.11504)


  This paper presents design of a Maximum Power Point Tracking (MPPT) circuit
and its functionality for tuning the maximum power transfer from an energy
harvester (EH) unit. Simple and practical Perturb and Observe algorithm is
investigated and implemented. We describe the circuit functionality and the
improvements that have been introduced to the original algorithm. The proposed
MPPT design is divided into three main blocks. The output signal is being
generated by the PWM or PFM block. The tracking speed has been enhanced by
implementing a variable step size in the Tracking Block. Finally, the overall
power consumption of the MPPT circuit itself is controlled by the Power
Management Block, which manages delivering the clock signal to the rest of the
circuit. The RTL code of the proposed MPPT has been described in Verilog, then
has been synthesized and placed-and-routed in a general purpose 130nm CMOS
technology.

    

### [[2110.11519] SiliFuzz: Fuzzing CPUs by proxy](http://arxiv.org/abs/2110.11519)


  CPUs are becoming more complex with every generation, at both the logical and
the physical levels. This potentially leads to more logic bugs and electrical
defects in CPUs being overlooked during testing, which causes data corruption
or other undesirable effects when these CPUs are used in production. These
ever-present problems may also have simply become more evident as more CPUs are
operated and monitored by large cloud providers.
If the RTL ("source code") of a CPU were available, we could apply greybox
fuzzing to the CPU model almost as we do to any other software
[arXiv:2102.02308]. However our targets are general purpose x86_64 CPUs
produced by third parties, where we do not have the RTL design, so in our case
CPU implementations are opaque. Moreover, we are more interested in electrical
defects as opposed to logic bugs.
We present SiliFuzz, a work-in-progress system that finds CPU defects by
fuzzing software proxies, like CPU simulators or disassemblers, and then
executing the accumulated test inputs (known as the corpus) on actual CPUs on a
large scale. The major difference between this work and traditional software
fuzzing is that a software bug fixed once will be fixed for all installations
of the software, while for CPU defects we have to test every individual core
repeatedly over its lifetime due to wear and tear. In this paper we also
analyze four groups of CPU defects that SiliFuzz has uncovered and describe
patterns shared by other SiliFuzz findings.

    

### [[2110.11520] Power Saving Evaluation with Automatic Offloading](http://arxiv.org/abs/2110.11520)


  Heterogeneous hardware other than small-core CPU such as GPU, FPGA, or
many-core CPU is increasingly being used. However, heterogeneous hardware usage
presents high technical skill barriers such as familiarity with CUDA. To
overcome this challenge, I previously proposed environment-adaptive software
that enables automatic conversion, automatic configuration, and
high-performance and low-power operation of once-written code, in accordance
with the hardware to be placed. I also previously verified performance
improvement of automatic GPU and FPGA offloading. In this paper, I verify
low-power operation with environment adaptation by evaluating power utilization
after automatic offloading. I compare Watt*seconds of existing applications
after automatic offloading with the case of CPU-only processing.

    

### [[2110.11521] High Level Synthesis Implementation of a Three-dimensional Systolic Array Architecture for Matrix Multiplications on Intel Stratix 10 FPGAs](http://arxiv.org/abs/2110.11521)


  In this paper, we consider the HLS implementation of a three-dimensional
systolic array architecture for matrix multiplication that targets specific
characteristics of Intel Stratix 10 FPGAs in order to produce designs that
achieve a high floating-point throughput using most of the DSPs at high
frequencies in a way that avoids the congestion of the routing fabric. The
investigated three-dimensional systolic array architecture is able to produce
hardware designs that use 99% of the available DSPs with maximum frequencies
that let us achieve performances above 3 TFLOPS.

    

### [[2110.11348] User Incentives for Blockchain-based Data Sharing Platforms](http://arxiv.org/abs/2110.11348)


  Data sharing is very important for accelerating scientific research, business
innovations, and for informing individuals. Yet, concerns over data privacy,
cost, and lack of secure data-sharing solutions have prevented data owners from
sharing data. To overcome these issues, several research works have proposed
blockchain-based data-sharing solutions for their ability to add transparency
and control to the data-sharing process. Yet, while models for decentralized
data sharing exist, how to incentivize these structures to enable data sharing
at scale remains largely unexplored. In this paper, we propose incentive
mechanisms for decentralized data-sharing platforms. We use smart contracts to
automate different payment options between data owners and data requesters. We
discuss multiple cost pricing scenarios for data owners to monetize their data.
Moreover, we simulate the incentive mechanisms on a blockchain-based
data-sharing platform. The evaluation of our simulation indicates that a cost
compensation model for the data owner can rapidly cover the cost of data
sharing and balance the overall incentives for all the actors in the platform.

    

### [[2110.11461] Three Practical Workflow Schedulers for Easy Maximum Parallelism](http://arxiv.org/abs/2110.11461)


  Runtime scheduling and workflow systems are an increasingly popular
algorithmic component in HPC because they allow full system utilization with
relaxed synchronization requirements. There are so many special-purpose tools
for task scheduling, one might wonder why more are needed. Use cases seen on
the Summit supercomputer needed better integration with MPI and greater
flexibility in job launch configurations. Preparation, execution, and analysis
of computational chemistry simulations at the scale of tens of thousands of
processors revealed three distinct workflow patterns. A separate job scheduler
was implemented for each one using extremely simple and robust designs:
file-based, task-list based, and bulk-synchronous. Comparing to existing
methods shows unique benefits of this work, including simplicity of design,
suitability for HPC centers, short startup time, and well-understood per-task
overhead. All three new tools have been shown to scale to full utilization of
Summit, and have been made publicly available with tests and documentation.
This work presents a complete characterization of the minimum effective task
granularity for efficient scheduler usage scenarios. These schedulers have the
same bottlenecks, and hence similar task granularities as those reported for
existing tools following comparable paradigms.

    

### [[2110.11462] A Fresh Look at the Architecture and Performance of Contemporary Isolation Platforms](http://arxiv.org/abs/2110.11462)


  With the ever-increasing pervasiveness of the cloud computing paradigm,
strong isolation guarantees and low performance overhead from isolation
platforms are paramount. An ideal isolation platform offers both: an
impermeable isolation boundary while imposing a negligible performance
overhead. In this paper, we examine various isolation platforms (containers,
secure containers, hypervisors, unikernels), and conduct a wide array of
experiments to measure the performance overhead and degree of isolation offered
by the platforms. We find that container platforms have the best, near-native,
performance while the newly emerging secure containers suffer from various
overheads. The highest degree of isolation is achieved by unikernels, closely
followed by traditional containers.

    

### [[2110.11644] EXSCALATE: An extreme-scale in-silico virtual screening platform to evaluate 1 trillion compounds in 60 hours on 81 PFLOPS supercomputers](http://arxiv.org/abs/2110.11644)


  The social and economic impact of the COVID-19 pandemic demands the reduction
of the time required to find a therapeutic cure. In the contest of urgent
computing, we re-designed the Exscalate molecular docking platform to benefit
from heterogeneous computation nodes and to avoid scaling issues. We deployed
the Exscalate platform on two top European supercomputers (CINECA-Marconi100
and ENI-HPC5), with a combined computational power of 81 PFLOPS, to evaluate
the interaction between 70 billions of small molecules and 15 binding-sites of
12 viral proteins of Sars-Cov2. The experiment lasted 60 hours and overall it
performed a trillion of evaluations.

    

### [[2110.11646] WebFed: Cross-platform Federated Learning Framework Based on Web Browser with Local Differential Privacy](http://arxiv.org/abs/2110.11646)


  For data isolated islands and privacy issues, federated learning has been
extensively invoking much interest since it allows clients to collaborate on
training a global model using their local data without sharing any with a third
party. However, the existing federated learning frameworks always need
sophisticated condition configurations (e.g., sophisticated driver
configuration of standalone graphics card like NVIDIA, compile environment)
that bring much inconvenience for large-scale development and deployment. To
facilitate the deployment of federated learning and the implementation of
related applications, we innovatively propose WebFed, a novel browser-based
federated learning framework that takes advantage of the browser's features
(e.g., Cross-platform, JavaScript Programming Features) and enhances the
privacy protection via local differential privacy mechanism. Finally, We
conduct experiments on heterogeneous devices to evaluate the performance of the
proposed WebFed framework.

    

### [[2110.11866] Morlet wavelet transform using attenuated sliding Fourier transform and kernel integral for graphic processing unit](http://arxiv.org/abs/2110.11866)


  Morlet or Gabor wavelet transforms as well as Gaussian smoothing, are widely
used in signal processing and image processing. However, the computational
complexity of their direct calculations is proportional not only to the number
of data points in a signal but also to the smoothing size, which is the
standard deviation in the Gaussian function in their transform functions. Thus,
when the standard deviation is large, its considerable computation time
diminishes the advantages of aforementioned transforms. Therefore, it is
important to formulate an algorithm to reduce the calculation time of the
transformations. In this paper, we first review calculation methods of Gaussian
smoothing by using the sliding Fourier transform (SFT) and our proposed
attenuated SFT (ASFT) \cite{YamashitaICPR2020}. Based on these methods, we
propose two types of calculation methods for Morlet wavelet transforms. We also
propose an algorithm to calculate SFT using the kernel integral on graphic
processing unit (GPU). When the number of calculation cores in GPU is not less
than the number of data points, the order of its calculation time is the
logarithm of the smoothing size and does not depend on the number of data
points. Using experiments, we compare the two methods for calculating the
Morlet wavelet transform and evaluate the calculation time of the proposed
algorithm using a kernel integral on GPU. For example, when the number of data
points and the standard deviation are 102400 and 8192.0, respectively, the
calculation time of the Morlet wavelet transform by the proposed method is
0.545 ms, which 413.6 times faster than a conventional method.

    

### [[2110.11927] Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP](http://arxiv.org/abs/2110.11927)


  Resource allocation problems in many computer systems can be formulated as
mathematical optimization problems. However, finding exact solutions to these
problems using off-the-shelf solvers is often intractable for large problem
sizes with tight SLAs, leading system designers to rely on cheap, heuristic
algorithms. We observe, however, that many allocation problems are granular:
they consist of a large number of clients and resources, each client requests a
small fraction of the total number of resources, and clients can
interchangeably use different resources. For these problems, we propose an
alternative approach that reuses the original optimization problem formulation
and leads to better allocations than domain-specific heuristics. Our technique,
Partitioned Optimization Problems (POP), randomly splits the problem into
smaller problems (with a subset of the clients and resources in the system) and
coalesces the resulting sub-allocations into a global allocation for all
clients. We provide theoretical and empirical evidence as to why random
partitioning works well. In our experiments, POP achieves allocations within
1.5% of the optimal with orders-of-magnitude improvements in runtime compared
to existing systems for cluster scheduling, traffic engineering, and load
balancing.

    

### [[1905.05254] Optimal Multithreaded Batch-Parallel 2-3 Trees](http://arxiv.org/abs/1905.05254)


  This paper presents a batch-parallel 2-3 tree T in an asynchronous dynamic
multithreading model that supports searches, insertions and deletions in sorted
batches and has essentially optimal parallelism, even under the restrictive
QRMW (queued read-modify-write) memory contention model where concurrent
accesses to the same memory location are queued and serviced one by one.
Specifically, if T has n items, then performing an item-sorted batch (given
as a leaf-based balanced binary tree) of b operations on T takes O( b *
log(n/b+1) + b ) work and O( log b + log n ) span (in the worst case as b,n ->
inf). This is information-theoretically work-optimal for b <= n, and also
span-optimal for pointer-based structures. Moreover, it is easy to support
optimal intersection, union and difference of instances of T with sizes m <= n,
namely within O( m * log(n/m+1) ) work and O( log m + log n ) span.
Furthermore, T supports other batch operations that make it a very useful
building block for parallel data structures.
To the author's knowledge, T is the first parallel sorted-set data structure
that can be used in an asynchronous multi-processor machine under a memory
model with queued contention and yet have asymptotically optimal work and span.
In fact, T is designed to have bounded contention and satisfy the claimed work
and span bounds regardless of the execution schedule.
Since all data structures and algorithms in this paper fit into the dynamic
multithreading paradigm, all their performance bounds are directly composable
with those of other data structures and algorithms in the same model. Finally,
the pipelining techniques in this paper are also likely to be very useful in
asynchronous parallelization of other recursive data structures.

    

### [[2105.11827] Narwhal and Tusk: A DAG-based Mempool and Efficient BFT Consensus](http://arxiv.org/abs/2105.11827)


  We propose separating the task of reliable transaction dissemination from
transaction ordering, to enable high-performance Byzantine fault-tolerant
quorum-based consensus. We design and evaluate a mempool protocol, Narwhal,
specializing in high-throughput reliable dissemination and storage of causal
histories of transactions. Narwhal tolerates an asynchronous network and
maintains high performance despite failures. Narwhal is designed to easily
scale-out using multiple workers at each validator, and we demonstrate that
there is no foreseeable limit to the throughput we can achieve. Composing
Narwhal with a partially synchronous consensus protocol (Narwhal-HotStuff)
yields significantly better throughput even in the presence of faults or
intermittent loss of liveness due to asynchrony. However, loss of liveness can
result in higher latency. To achieve overall good performance when faults occur
we design Tusk, a zero-message overhead asynchronous consensus protocol, to
work with Narwhal. We demonstrate its high performance under a variety of
configurations and faults. As a summary of results, on a WAN, Narwhal-Hotstuff
achieves over 130,000 tx/sec at less than 2-sec latency compared with 1,800
tx/sec at 1-sec latency for Hotstuff. Additional workers increase throughput
linearly to 600,000 tx/sec without any latency increase. Tusk achieves 160,000
tx/sec with about 3 seconds latency. Under faults, both protocols maintain high
throughput, but Narwhal-HotStuff suffers from increased latency.

    

### [[2110.11384] Decentralised Person Re-Identification with Selective Knowledge Aggregation](http://arxiv.org/abs/2110.11384)


  Existing person re-identification (Re-ID) methods mostly follow a centralised
learning paradigm which shares all training data to a collection for model
learning. This paradigm is limited when data from different sources cannot be
shared due to privacy concerns. To resolve this problem, two recent works have
introduced decentralised (federated) Re-ID learning for constructing a globally
generalised model (server)without any direct access to local training data nor
shared data across different source domains (clients). However, these methods
are poor on how to adapt the generalised model to maximise its performance on
individual client domain Re-ID tasks having different Re-ID label spaces, due
to a lack of understanding of data heterogeneity across domains. We call this
poor 'model personalisation'. In this work, we present a new Selective
Knowledge Aggregation approach to decentralised person Re-ID to optimise the
trade-off between model personalisation and generalisation. Specifically, we
incorporate attentive normalisation into the normalisation layers in a deep
ReID model and propose to learn local normalisation layers specific to each
domain, which are decoupled from the global model aggregation in federated
Re-ID learning. This helps to preserve model personalisation knowledge on each
local client domain and learn instance-specific information. Further, we
introduce a dual local normalisation mechanism to learn generalised
normalisation layers in each local model, which are then transmitted to the
global model for central aggregation. This facilitates selective knowledge
aggregation on the server to construct a global generalised model for
out-of-the-box deployment on unseen novel domains. Extensive experiments on
eight person Re-ID datasets show that the proposed approach to decentralised
Re-ID significantly outperforms the state-of-the-art decentralised methods.

    

### [[2110.11411] PROVES: Establishing Image Provenance using Semantic Signatures](http://arxiv.org/abs/2110.11411)


  Modern AI tools, such as generative adversarial networks, have transformed
our ability to create and modify visual data with photorealistic results.
However, one of the deleterious side-effects of these advances is the emergence
of nefarious uses in manipulating information in visual data, such as through
the use of deep fakes. We propose a novel architecture for preserving the
provenance of semantic information in images to make them less susceptible to
deep fake attacks. Our architecture includes semantic signing and verification
steps. We apply this architecture to verifying two types of semantic
information: individual identities (faces) and whether the photo was taken
indoors or outdoors. Verification accounts for a collection of common image
transformation, such as translation, scaling, cropping, and small rotations,
and rejects adversarial transformations, such as adversarially perturbed or, in
the case of face verification, swapped faces. Experiments demonstrate that in
the case of provenance of faces in an image, our approach is robust to
black-box adversarial transformations (which are rejected) as well as benign
transformations (which are accepted), with few false negatives and false
positives. Background verification, on the other hand, is susceptible to
black-box adversarial examples, but becomes significantly more robust after
adversarial training.

    

### [[2110.11417] HIRE-SNN: Harnessing the Inherent Robustness of Energy-Efficient Deep Spiking Neural Networks by Training with Crafted Input Noise](http://arxiv.org/abs/2110.11417)


  Low-latency deep spiking neural networks (SNNs) have become a promising
alternative to conventional artificial neural networks (ANNs) because of their
potential for increased energy efficiency on event-driven neuromorphic
hardware. Neural networks, including SNNs, however, are subject to various
adversarial attacks and must be trained to remain resilient against such
attacks for many applications. Nevertheless, due to prohibitively high training
costs associated with SNNs, analysis, and optimization of deep SNNs under
various adversarial attacks have been largely overlooked. In this paper, we
first present a detailed analysis of the inherent robustness of low-latency
SNNs against popular gradient-based attacks, namely fast gradient sign method
(FGSM) and projected gradient descent (PGD). Motivated by this analysis, to
harness the model robustness against these attacks we present an SNN training
algorithm that uses crafted input noise and incurs no additional training time.
To evaluate the merits of our algorithm, we conducted extensive experiments
with variants of VGG and ResNet on both CIFAR-10 and CIFAR-100 datasets.
Compared to standard trained direct input SNNs, our trained models yield
improved classification accuracy of up to 13.7% and 10.1% on FGSM and PGD
attack-generated images, respectively, with negligible loss in clean image
accuracy. Our models also outperform inherently robust SNNs trained on
rate-coded inputs with improved or similar classification performance on
attack-generated images while having up to 25x and 4.6x lower latency and
computation energy, respectively.

    

### [[2110.11482] Aware Adoption of AI: from Potential to Reusable Value](http://arxiv.org/abs/2110.11482)


  Artificial Intelligence (AI) provides practical advantages in different
applied domains. This is changing the way decision-makers reason about complex
systems. Indeed, broader visibility on greater information (re)sources, e.g.
Big Data, is now available to intelligent agents. On the other hand, decisions
are not always based on reusable, multi-purpose, and explainable knowledge.
Therefore, it is necessary to define new models to describe and manage this new
(re)source of uncertainty.
This contribution aims to introduce a multidimensional framework to deal with
the notion of Value in the AI context. In this model, Big Data represent a
distinguished dimension (characteristic) of Value rather than an intrinsic
property of Big Data. Great attention is paid to hidden dimensions of value,
which may be linked to emerging innovation processes. The requirements to
describe the framework are provided, and an associated mathematical structure
is presented to deal with comparison, combination, and update of states of
knowledge regarding Value. We introduce a notion of consistency of a state of
knowledge to investigate the relation between Human and Artificial
intelligences; this form of uncertainty is specified in analogy with two
scenarios concerning decision-making and non-classical measurements. Finally,
we propose future investigations aiming at the inclusion of this form of
uncertainty in the assessment of impact, risks, and structural modelling.

    

### [[2110.11514] SYNERGY: Building Task Bots at Scale Using Symbolic Knowledge and Machine Teaching](http://arxiv.org/abs/2110.11514)


  In this paper we explore the use of symbolic knowledge and machine teaching
to reduce human data labeling efforts in building neural task bots. We propose
SYNERGY, a hybrid learning framework where a task bot is developed in two
steps: (i) Symbolic knowledge to neural networks: Large amounts of simulated
dialog sessions are generated based on task-specific symbolic knowledge which
is represented as a task schema consisting of dialog flows and task-oriented
databases. Then a pre-trained neural dialog model, SOLOIST, is fine-tuned on
the simulated dialogs to build a bot for the task. (ii) Neural learning: The
fine-tuned neural dialog model is continually refined with a handful of real
task-specific dialogs via machine teaching, where training samples are
generated by human teachers interacting with the task bot. We validate SYNERGY
on four dialog tasks. Experimental results show that SYNERGY maps task-specific
knowledge into neural dialog models achieving greater diversity and coverage of
dialog flows, and continually improves model performance with machine teaching,
thus demonstrating strong synergistic effects of symbolic knowledge and machine
teaching.

    

### [[2110.11527] Proceedings Third Workshop on Formal Methods for Autonomous Systems](http://arxiv.org/abs/2110.11527)


  Autonomous systems are highly complex and present unique challenges for the
application of formal methods. Autonomous systems act without human
intervention, and are often embedded in a robotic system, so that they can
interact with the real world. As such, they exhibit the properties of
safety-critical, cyber-physical, hybrid, and real-time systems.
This EPTCS volume contains the proceedings for the third workshop on Formal
Methods for Autonomous Systems (FMAS 2021), which was held virtually on the
21st and 22nd of October 2021. Like the previous workshop, FMAS 2021 was an
online, stand-alone event, as an adaptation to the ongoing COVID-19
restrictions. Despite the challenges this brought, we were determined to build
on the success of the previous two FMAS workshops.
The goal of FMAS is to bring together leading researchers who are tackling
the unique challenges of autonomous systems using formal methods, to present
recent and ongoing work. We are interested in the use of formal methods to
specify, model, or verify autonomous and/or robotic systems; in whole or in
part. We are also interested in successful industrial applications and
potential future directions for this emerging application of formal methods.

    

### [[2110.11567] Logical Assessment Formula and its Principles for Evaluations without Accurate Ground-Truth Labels](http://arxiv.org/abs/2110.11567)


  Logical assessment formula (LAF) was proposed for evaluations without
accurate ground-truth labels (AGTL). In this paper, we reveal the principles of
LAF via comprehensive theoretical analyses. From the revealed principles, we
summarize the practicability of LAF: 1) LAF can be reasonably applied for
evaluations without AGTL on a more difficult task, just acting like usual
strategies for evaluations with AGTL; 2) LAF can be applied for evaluations
without AGTL from the logical perspective on an easier task, unable to be
acting like usual strategies for evaluations with AGTL. Experimental results
and analyses of LAF applied on tumour segmentation for breast cancer support
the practicability of LAF summarized from the revealed principles.

    

### [[2110.11573] ModEL: A Modularized End-to-end Reinforcement Learning Framework for Autonomous Driving](http://arxiv.org/abs/2110.11573)


  Heated debates continue over the best autonomous driving framework. The
classic modular pipeline is widely adopted in the industry owing to its great
interpretability and stability, whereas the end-to-end paradigm has
demonstrated considerable simplicity and learnability along with the rise of
deep learning. We introduce a new modularized end-to-end reinforcement learning
framework (ModEL) for autonomous driving, which combines the merits of both
previous approaches. The autonomous driving stack of ModEL is decomposed into
perception, planning, and control module, leveraging scene understanding,
end-to-end reinforcement learning, and PID control respectively. Furthermore,
we build a fully functional autonomous vehicle to deploy this framework.
Through extensive simulation and real-world experiments, our framework has
shown great generalizability to various complicated scenarios and outperforms
the competing baselines.

    

### [[2110.11583] EvoGAN: An Evolutionary Computation Assisted GAN](http://arxiv.org/abs/2110.11583)


  The image synthesis technique is relatively well established which can
generate facial images that are indistinguishable even by human beings.
However, all of these approaches uses gradients to condition the output,
resulting in the outputting the same image with the same input. Also, they can
only generate images with basic expression or mimic an expression instead of
generating compound expression. In real life, however, human expressions are of
great diversity and complexity. In this paper, we propose an evolutionary
algorithm (EA) assisted GAN, named EvoGAN, to generate various compound
expressions with any accurate target compound expression. EvoGAN uses an EA to
search target results in the data distribution learned by GAN. Specifically, we
use the Facial Action Coding System (FACS) as the encoding of an EA and use a
pre-trained GAN to generate human facial images, and then use a pre-trained
classifier to recognize the expression composition of the synthesized images as
the fitness function to guide the search of the EA. Combined random searching
algorithm, various images with the target expression can be easily sythesized.
Quantitative and Qualitative results are presented on several compound
expressions, and the experimental results demonstrate the feasibility and the
potential of EvoGAN.

    

### [[2110.11584] Multiwave COVID-19 Prediction via Social Awareness-Based Graph Neural Networks using Mobility and Web Search Data](http://arxiv.org/abs/2110.11584)


  Recurring outbreaks of COVID-19 have posed enduring effects on global
society, which calls for a predictor of pandemic waves using various data with
early availability. Existing prediction models that forecast the first outbreak
wave using mobility data may not be applicable to the multiwave prediction,
because the evidence in the USA and Japan has shown that mobility patterns
across different waves exhibit varying relationships with fluctuations in
infection cases. Therefore, to predict the multiwave pandemic, we propose a
Social Awareness-Based Graph Neural Network (SAB-GNN) that considers the decay
of symptom-related web search frequency to capture the changes in public
awareness across multiple waves. SAB-GNN combines GNN and LSTM to model the
complex relationships among urban districts, inter-district mobility patterns,
web search history, and future COVID-19 infections. We train our model to
predict future pandemic outbreaks in the Tokyo area using its mobility and web
search data from April 2020 to May 2021 across four pandemic waves collected by
_ANONYMOUS_COMPANY_ under strict privacy protection rules. Results show our
model outperforms other baselines including ST-GNN and MPNN+LSTM. Though our
model is not computationally expensive (only 3 layers and 10 hidden neurons),
the proposed model enables public agencies to anticipate and prepare for future
pandemic outbreaks.

    

### [[2110.11593] Automatic Detection of Injection and Press Mold Parts on 2D Drawing Using Deep Neural Network](http://arxiv.org/abs/2110.11593)


  This paper proposes a method to automatically detect the key feature parts in
a CAD of commercial TV and monitor using a deep neural network. We developed a
deep learning pipeline that can detect the injection parts such as hook, boss,
undercut and press parts such as DPS, Embo-Screwless, Embo-Burring, and EMBO in
the 2D CAD drawing images. We first cropped the drawing to a specific size for
the training efficiency of a deep neural network. Then, we use Cascade R-CNN to
find the position of injection and press parts and use Resnet-50 to predict the
orientation of the parts. Finally, we convert the position of the parts found
through the cropped image to the position of the original image. As a result,
we obtained detection accuracy of injection and press parts with 84.1% in AP
(Average Precision), 91.2% in AR(Average Recall), 72.0% in AP, 87.0% in AR, and
orientation accuracy of injection and press parts with 94.4% and 92.0%, which
can facilitate the faster design in industrial product design.

    

### [[2110.11624] SCICAP: Generating Captions for Scientific Figures](http://arxiv.org/abs/2110.11624)


  Researchers use figures to communicate rich, complex information in
scientific papers. The captions of these figures are critical to conveying
effective messages. However, low-quality figure captions commonly occur in
scientific articles and may decrease understanding. In this paper, we propose
an end-to-end neural framework to automatically generate informative,
high-quality captions for scientific figures. To this end, we introduce SCICAP,
a large-scale figure-caption dataset based on computer science arXiv papers
published between 2010 and 2020. After pre-processing - including figure-type
classification, sub-figure identification, text normalization, and caption text
selection - SCICAP contained more than two million figures extracted from over
290,000 papers. We then established baseline models that caption graph plots,
the dominant (19.2%) figure type. The experimental results showed both
opportunities and steep challenges of generating captions for scientific
figures.

    

### [[2110.11680] Deep Two-Stream Video Inference for Human Body Pose and Shape Estimation](http://arxiv.org/abs/2110.11680)


  Several video-based 3D pose and shape estimation algorithms have been
proposed to resolve the temporal inconsistency of single-image-based methods.
However it still remains challenging to have stable and accurate
reconstruction. In this paper, we propose a new framework Deep Two-Stream Video
Inference for Human Body Pose and Shape Estimation (DTS-VIBE), to generate 3D
human pose and mesh from RGB videos. We reformulate the task as a
multi-modality problem that fuses RGB and optical flow for more reliable
estimation. In order to fully utilize both sensory modalities (RGB or optical
flow), we train a two-stream temporal network based on transformer to predict
SMPL parameters. The supplementary modality, optical flow, helps to maintain
temporal consistency by leveraging motion knowledge between two consecutive
frames. The proposed algorithm is extensively evaluated on the Human3.6 and
3DPW datasets. The experimental results show that it outperforms other
state-of-the-art methods by a significant margin.

    

### [[2110.11709] Creating Knowledge Graphs Subsets using Shape Expressions](http://arxiv.org/abs/2110.11709)


  The initial adoption of knowledge graphs by Google and later by big companies
has increased their adoption and popularity. In this paper we present a formal
model for three different types of knowledge graphs which we call RDF-based
graphs, property graphs and wikibase graphs. In order to increase the quality
of Knowledge Graphs, several approaches have appeared to describe and validate
their contents. Shape Expressions (ShEx) has been proposed as concise language
for RDF validation. We give a brief introduction to ShEx and present two
extensions that can also be used to describe and validate property graphs
(PShEx) and wikibase graphs (WShEx). One problem of knowledge graphs is the
large amount of data they contain, which jeopardizes their practical
application. In order to palliate this problem, one approach is to create
subsets of those knowledge graphs for some domains. We propose the following
approaches to generate those subsets: Entity-matching, simple matching, ShEx
matching, ShEx plus Slurp and ShEx plus Pregel which are based on declaratively
defining the subsets by either matching some content or by Shape Expressions.
The last approach is based on a novel validation algorithm for ShEx based on
the Pregel algorithm that can handle big data graphs and has been implemented
on Apache Spark GraphX.

    

### [[2110.11737] Measuring the Non-Transitivity in Chess](http://arxiv.org/abs/2110.11737)


  It has long been believed that Chess is the \emph{Drosophila} of Artificial
Intelligence (AI). Studying Chess can productively provide valid knowledge
about complex systems. Although remarkable progress has been made on solving
Chess, the geometrical landscape of Chess in the strategy space is still
mysterious. Judging on AI-generated strategies, researchers hypothesised that
the strategy space of Chess possesses a spinning top geometry, with the upright
axis representing the \emph{transitive} dimension (e.g., A beats B, B beats C,
A beats C), and the radial axis representing the \emph{non-transitive}
dimension (e.g., A beats B, B beats C, C beats A). However, it is unclear
whether such a hypothesis holds for real-world strategies. In this paper, we
quantify the non-transitivity in Chess through real-world data from human
players. Specifically, we performed two ways of non-transitivity
quantifications -- Nash Clustering and counting the number of
Rock-Paper-Scissor cycles -- on over one billion match data from Lichess and
FICS. Our findings positively indicate that the strategy space occupied by
real-world Chess strategies demonstrates a spinning top geometry, and more
importantly, there exists a strong connection between the degree of
non-transitivity and the progression of a Chess player's rating. In particular,
high degrees of non-transitivity tend to prevent human players from making
progress on their Elo rating, whereas progressions are easier to make at the
level of ratings where the degree of non-transitivity is lower. Additionally,
we also investigate the implication of the degree of non-transitivity for
population-based training methods. By considering \emph{fixed-memory Fictitious
Play} as a proxy, we reach the conclusion that maintaining large-size and
diverse populations of strategies is imperative to training effective AI agents
in solving Chess types of games.

    

### [[2110.11742] Few-shot Semantic Segmentation with Self-supervision from Pseudo-classes](http://arxiv.org/abs/2110.11742)


  Despite the success of deep learning methods for semantic segmentation,
few-shot semantic segmentation remains a challenging task due to the limited
training data and the generalisation requirement for unseen classes. While
recent progress has been particularly encouraging, we discover that existing
methods tend to have poor performance in terms of meanIoU when query images
contain other semantic classes besides the target class. To address this issue,
we propose a novel self-supervised task that generates random pseudo-classes in
the background of the query images, providing extra training data that would
otherwise be unavailable when predicting individual target classes. To that
end, we adopted superpixel segmentation for generating the pseudo-classes. With
this extra supervision, we improved the meanIoU performance of the
state-of-the-art method by 2.5% and 5.1% on the one-shot tasks, as well as 6.7%
and 4.4% on the five-shot tasks, on the PASCAL-5i and COCO benchmarks,
respectively.

    

### [[2110.11767] Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning](http://arxiv.org/abs/2110.11767)


  The task of image captioning aims to generate captions directly from images
via the automatically learned cross-modal generator. To build a well-performing
generator, existing approaches usually need a large number of described images,
which requires a huge effects on manual labeling. However, in real-world
applications, a more general scenario is that we only have limited amount of
described images and a large number of undescribed images. Therefore, a
resulting challenge is how to effectively combine the undescribed images into
the learning of cross-modal generator. To solve this problem, we propose a
novel image captioning method by exploiting the Cross-modal Prediction and
Relation Consistency (CPRC), which aims to utilize the raw image input to
constrain the generated sentence in the commonly semantic space. In detail,
considering that the heterogeneous gap between modalities always leads to the
supervision difficulty of using the global embedding directly, CPRC turns to
transform both the raw image and corresponding generated sentence into the
shared semantic space, and measure the generated sentence from two aspects: 1)
Prediction consistency. CPRC utilizes the prediction of raw image as soft label
to distill useful supervision for the generated sentence, rather than employing
the traditional pseudo labeling; 2) Relation consistency. CPRC develops a novel
relation consistency between augmented images and corresponding generated
sentences to retain the important relational knowledge. In result, CPRC
supervises the generated sentence from both the informativeness and
representativeness perspectives, and can reasonably use the undescribed images
to learn a more effective generator under the semi-supervised scenario.

    

### [[2110.11822] Unraveling the hidden environmental impacts of AI solutions for environment](http://arxiv.org/abs/2110.11822)


  In the past ten years artificial intelligence has encountered such dramatic
progress that it is seen now as a tool of choice to solve environmental issues
and in the first place greenhouse gas emissions (GHG). At the same time the
deep learning community began to realize that training models with more and
more parameters required a lot of energy and as a consequence GHG emissions. To
our knowledge, questioning the complete environmental impacts of AI methods for
environment ("AI for green"), and not only GHG, has never been addressed
directly. In this article we propose to study the possible negative impact of
"AI for green" 1) by reviewing first the different types of AI impacts 2) by
presenting the different methodologies used to assess those impacts, in
particular life cycle assessment and 3) by discussing how to assess the
environmental usefulness of a general AI service.

    

### [[2110.11879] An N-gram based approach to auto-extracting topics from research articles](http://arxiv.org/abs/2110.11879)


  A lot of manual work goes into identifying a topic for an article. With a
large volume of articles, the manual process can be exhausting. Our approach
aims to address this issue by automatically extracting topics from the text of
large Numbers of articles. This approach takes into account the efficiency of
the process. Based on existing N-gram analysis, our research examines how often
certain words appear in documents in order to support automatic topic
extraction. In order to improve efficiency, we apply custom filtering standards
to our research. Additionally, delete as many noncritical or irrelevant phrases
as possible. In this way, we can ensure we are selecting unique keyphrases for
each article, which capture its core idea. For our research, we chose to center
on the autonomous vehicle domain, since the research is relevant to our daily
lives. We have to convert the PDF versions of most of the research papers into
editable types of files such as TXT. This is because most of the research
papers are only in PDF format. To test our proposed idea of automating,
numerous articles on robotics have been selected. Next, we evaluate our
approach by comparing the result with that obtained manually.

    

### [[2110.11924] Gapoera: Application Programming Interface for AI Environment of Indonesian Board Game](http://arxiv.org/abs/2110.11924)


  Currently, the development of computer games has shown a tremendous surge.
The ease and speed of internet access today have also influenced the
development of computer games, especially computer games that are played
online. Internet technology has allowed computer games to be played in
multiplayer mode. Interaction between players in a computer game can be built
in several ways, one of which is by providing balanced opponents. Opponents can
be developed using intelligent agents. On the other hand, research on
developing intelligent agents is also growing rapidly. In computer game
development, one of the easiest ways to measure the performance of an
intelligent agent is to develop a virtual environment that allows the
intelligent agent to interact with other players. In this research, we try to
develop an intelligent agent and virtual environment for the board game. To be
easily accessible, the intelligent agent and virtual environment are then
developed into an Application Programming Interface (API) service called
Gapoera API. The Gapoera API service that is built is expected to help game
developers develop a game without having to think much about the artificial
intelligence that will be embedded in the game. This service provides a basic
multilevel intelligent agent that can provide users with playing board games
commonly played in Indonesia. Although the Gapoera API can be used for various
types of games, in this paper, we will focus on the discussion on a popular
traditional board game in Indonesia, namely Mancala. The test results conclude
that the multilevel agent concept developed has worked as expected. On the
other hand, the development of the Gapoera API service has also been
successfully accessed on several game platforms.

    

### [[2110.11929] Double Trouble: How to not explain a text classifier's decisions using counterfactuals synthesized by masked language models?](http://arxiv.org/abs/2110.11929)


  Explaining how important each input feature is to a classifier's decision is
critical in high-stake applications. An underlying principle behind dozens of
explanation methods is to take the prediction difference between
before-and-after an input feature (here, a token) is removed as its attribution
- the individual treatment effect in causal inference. A recent method called
Input Marginalization (IM) (Kim et al., 2020) uses BERT to replace a token -
i.e. simulating the do(.) operator - yielding more plausible counterfactuals.
However, our rigorous evaluation using five metrics and on three datasets found
IM explanations to be consistently more biased, less accurate, and less
plausible than those derived from simply deleting a word.

    

### [[2104.07123] The MuSe 2021 Multimodal Sentiment Analysis Challenge: Sentiment, Emotion, Physiological-Emotion, and Stress](http://arxiv.org/abs/2104.07123)


  Multimodal Sentiment Analysis (MuSe) 2021 is a challenge focusing on the
tasks of sentiment and emotion, as well as physiological-emotion and
emotion-based stress recognition through more comprehensively integrating the
audio-visual, language, and biological signal modalities. The purpose of MuSe
2021 is to bring together communities from different disciplines; mainly, the
audio-visual emotion recognition community (signal-based), the sentiment
analysis community (symbol-based), and the health informatics community. We
present four distinct sub-challenges: MuSe-Wilder and MuSe-Stress which focus
on continuous emotion (valence and arousal) prediction; MuSe-Sent, in which
participants recognise five classes each for valence and arousal; and
MuSe-Physio, in which the novel aspect of `physiological-emotion' is to be
predicted. For this years' challenge, we utilise the MuSe-CaR dataset focusing
on user-generated reviews and introduce the Ulm-TSST dataset, which displays
people in stressful depositions. This paper also provides detail on the
state-of-the-art feature sets extracted from these datasets for utilisation by
our baseline model, a Long Short-Term Memory-Recurrent Neural Network. For each
sub-challenge, a competitive baseline for participants is set; namely, on test,
we report a Concordance Correlation Coefficient (CCC) of .4616 CCC for
MuSe-Wilder; .4717 CCC for MuSe-Stress, and .4606 CCC for MuSe-Physio. For
MuSe-Sent an F1 score of 32.82 % is obtained.

    

### [[2105.03726] Mental Models of Adversarial Machine Learning](http://arxiv.org/abs/2105.03726)


  Although machine learning (ML) is widely used in practice, little is known
about practitioners' actual understanding of potential security challenges. In
this work, we close this substantial gap in the literature and contribute a
qualitative study focusing on developers' mental models of the ML pipeline and
potentially vulnerable components. Studying mental models has helped in other
security fields to discover root causes or improve risk communication. Our
study reveals four characteristic ranges in mental models of industrial
practitioners. The first range concerns the intertwined relationship of
adversarial machine learning (AML) and classical security. The second range
describes structural and functional components. The third range expresses
individual variations of mental models, which are neither explained by the
application nor by the educational background of the corresponding subjects.
The fourth range corresponds to the varying levels of technical depth, which
are however not determined by our subjects' level of knowledge. Our
characteristic ranges have implications for the integration of AML into
corporate workflows, security enhancing tools for practitioners, and creating
appropriate regulatory frameworks for AML.

    

### [[2105.04447] SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation](http://arxiv.org/abs/2105.04447)


  We propose a novel scene flow estimation approach to capture and infer 3D
motions from point clouds. Estimating 3D motions for point clouds is
challenging, since a point cloud is unordered and its density is significantly
non-uniform. Such unstructured data poses difficulties in matching
corresponding points between point clouds, leading to inaccurate flow
estimation. We propose a novel architecture named Sparse
Convolution-Transformer Network (SCTN) that equips the sparse convolution with
the transformer. Specifically, by leveraging the sparse convolution, SCTN
transfers irregular point cloud into locally consistent flow features for
estimating continuous and consistent motions within an object/local object
part. We further propose to explicitly learn point relations using a point
transformer module, different from exiting methods. We show that the learned
relation-based contextual information is rich and helpful for matching
corresponding points, benefiting scene flow estimation. In addition, a novel
loss function is proposed to adaptively encourage flow consistency according to
feature similarity. Extensive experiments demonstrate that our proposed
approach achieves a new state of the art in scene flow estimation. Our approach
achieves an error of 0.038 and 0.037 (EPE3D) on FlyingThings3D and KITTI Scene
Flow respectively, which significantly outperforms previous methods by large
margins.

    

### [[2106.00200] Iterative Hierarchical Attention for Answering Complex Questions over Long Documents](http://arxiv.org/abs/2106.00200)


  We propose a new model, DocHopper, that iteratively attends to different
parts of long, hierarchically structured documents to answer complex questions.
Similar to multi-hop question-answering (QA) systems, at each step, DocHopper
uses a query $q$ to attend to information from a document, combines this
``retrieved'' information with $q$ to produce the next query. However, in
contrast to most previous multi-hop QA systems, DocHopper is able to
``retrieve'' either short passages or long sections of the document, thus
emulating a multi-step process of ``navigating'' through a long document to
answer a question. To enable this novel behavior, DocHopper does not combine
document information with $q$ by concatenating text to the text of $q$, but by
combining a compact neural representation of $q$ with a compact neural
representation of a hierarchical part of the document, which can potentially be
quite large. We experiment with DocHopper on four different QA tasks that
require reading long and complex documents to answer multi-hop questions, and
show that DocHopper achieves state-of-the-art results on three of the datasets.
Additionally, DocHopper is efficient at inference time, being 3--10 times
faster than the baselines.

    

### [[2110.10953] MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation](http://arxiv.org/abs/2110.10953)


  With the emergence of service robots and surveillance cameras, dynamic face
recognition (DFR) in wild has received much attention in recent years. Face
detection and head pose estimation are two important steps for DFR. Very often,
the pose is estimated after the face detection. However, such sequential
computations lead to higher latency. In this paper, we propose a low latency
and lightweight network for simultaneous face detection, landmark localization
and head pose estimation. Inspired by the observation that it is more
challenging to locate the facial landmarks for faces with large angles, a pose
loss is proposed to constrain the learning. Moreover, we also propose an
uncertainty multi-task loss to learn the weights of individual tasks
automatically. Another challenge is that robots often use low computational
units like ARM based computing core and we often need to use lightweight
networks instead of the heavy ones, which lead to performance drop especially
for small and hard faces. In this paper, we propose online feedback sampling to
augment the training samples across different scales, which increases the
diversity of training data automatically. Through validation in commonly used
WIDER FACE, AFLW and AFLW2000 datasets, the results show that the proposed
method achieves the state-of-the-art performance in low computational
resources.

    

### [[2110.11579] How to Schedule Near-Optimally under Real-World Constraints](http://arxiv.org/abs/2110.11579)


  Scheduling is a critical part of practical computer systems, and scheduling
has also been extensively studied from a theoretical perspective.
Unfortunately, there is a gap between theory and practice, as the optimal
scheduling policies presented by theory can be difficult or impossible to
perfectly implement in practice. In this work, we use recent breakthroughs in
queueing theory to begin to bridge this gap. We show how to translate
theoretically optimal policies -- which provably minimize mean response time
(a.k.a. latency) -- into near-optimal policies that are easily implemented in
practical settings. Specifically, we handle the following real-world
constraints:
- We show how to schedule in systems where job sizes (a.k.a. running time)
are unknown, or only partially known. We do so using simple policies that
achieve performance very close to the much more complicated theoretically
optimal policies.
- We show how to schedule in systems that have only a limited number of
priority levels available. We show how to adapt theoretically optimal policies
to this constrained setting and determine how many levels we need for
near-optimal performance.
- We show how to schedule in systems where job preemption can only happen at
specific checkpoints. Adding checkpoints allows for smarter scheduling, but
each checkpoint incurs time overhead. We give a rule of thumb that
near-optimally balances this tradeoff.

    

### [[2110.11719] Experience with PCIe streaming on FPGA for high throughput ML inferencing](http://arxiv.org/abs/2110.11719)


  Achieving maximum possible rate of inferencing with minimum hardware
resources plays a major role in reducing enterprise operational costs. In this
paper we explore use of PCIe streaming on FPGA based platforms to achieve high
throughput. PCIe streaming is a unique capability available on FPGA that
eliminates the need for memory copy overheads. We have presented our results
for inferences on a gradient boosted trees model, for online retail
recommendations. We compare the results achieved with the popular library
implementations on GPU and the CPU platforms and observe that the PCIe
streaming enabled FPGA implementation achieves the best overall measured
performance. We also measure power consumption across all platforms and find
that the PCIe streaming on FPGA platform achieves the 25x and 12x better energy
efficiency than an implementation on CPU and GPU platforms, respectively. We
discuss the conditions that need to be met, in order to achieve this kind of
acceleration on the FPGA. Further, we analyze the run time statistics on GPU
and FPGA and identify opportunities to enhance performance on both the
platforms.

    

### [[2110.11790] Automatic Guide Generation for Stan via NumPyro](http://arxiv.org/abs/2110.11790)


  Stan is a very popular probabilistic language with a state-of-the-art HMC
sampler but it only offers a limited choice of algorithms for black-box
variational inference. In this paper, we show that using our recently proposed
compiler from Stan to Pyro, Stan users can easily try the set of algorithms
implemented in Pyro for black-box variational inference. We evaluate our
approach on PosteriorDB, a database of Stan models with corresponding data and
reference posterior samples. Results show that the eight algorithms available
in Pyro offer a range of possible compromises between complexity and accuracy.
This paper illustrates that compiling Stan to another probabilistic language
can be used to leverage new features for Stan users, and give access to a large
set of examples for language developers who implement these new features.

    

### [[1907.05590] Revisiting Occurrence Typing](http://arxiv.org/abs/1907.05590)


  We revisit occurrence typing, a technique to refine the type of variables
occurring in type-cases and, thus, capturesome programming patterns used in
untyped languages. Although occurrence typing was tied from its inceptionto
set-theoretic types-union types, in particular-it never fully exploited the
capabilities of these types. Here weshow how, by using set-theoretic types, it
is possible to develop a general typing framework that encompasses
andgeneralizes several aspects of current occurrence typing proposals and that
can be applied to tackle other problemssuch as the reconstruction of
intersection types for unannotated or partially annotated functions and the
optimizationof the compilation of gradually typed languages.

    

### [[2105.02541] From Bounded Checking to Verification of Equivalence via Symbolic Up-to Techniques](http://arxiv.org/abs/2105.02541)


  We present a bounded equivalence verification technique for higher-order
programs with local state. This technique combines fully abstract symbolic
environmental bisimulations similar to symbolic game semantics, novel up-to
techniques, and lightweight state invariant annotations. This yields an
equivalence verification technique with no false positives or negatives. The
technique is bounded-complete, in that all inequivalences are automatically
detected given large enough bounds. Moreover, several hard equivalences are
proved automatically or after being annotated with state invariants. We realise
the technique in a tool prototype called Hobbit and benchmark it with an
extensive set of new and existing examples. Hobbit can prove many classical
equivalences including all Meyer and Sieber examples.

    