
## 2021-12-6

### [[2112.01743] A Parallel PageRank Algorithm For Undirected Graph](http://arxiv.org/abs/2112.01743)


  PageRank is a fundamental property of graph and there have been plenty of
PageRank algorithms. Generally, we consider undirected graph as a complicated
directed graph. However, some properties of undirected graph, such as symmetry,
are ignored when computing PageRank by existing algorithms. In this paper, we
propose a parallel PageRank algorithm which is specially for undirected graph.
We first demonstrate that the PageRank vector can be viewed as a linear
combination of eigenvectors of probability transition matrix and the
corresponding coefficients are the functions of eigenvalues. Then we introduce
the Chebyshev polynomial approximation by which PageRank vector can be computed
iteratively. Finally, we propose the parallel PageRank algorithm as the
Chebyshev polynomial approximating algorithm(CPAA). Experimental results show
that CPAA only takes 60% of iteration rounds of the power method and is at
least 4 times faster than the power method.

    

### [[2112.01945] A Study of the Impact of the Contention Window on the Performance of IEEE 802.11bd Networks with Channel Bonding](http://arxiv.org/abs/2112.01945)


  Nowadays, Vehicle-To-Everything (V2X) networks are actively developing. Most
of the already deployed V2X networks are based on the IEEE 802.11p standard.
However, these networks can provide only basic V2X applications and will
unlikely fulfill stringent requirements of modern V2X applications. Thus, the
IEEE has launched a new IEEE 802.11bd standard. A significant novelty of this
standard is channel bonding. IEEE 802.11bd describes two channel bonding
techniques, which differ from the legacy one used in modern Wi-Fi networks. Our
study performs a comparative analysis of the various channel bonding techniques
and a single-channel access method from IEEE 802.11p via simulation. We compare
them under different contention window sizes and demonstrate that the legacy
technique provides the best quality of service in terms of frame transmission
delays and packet loss ratio. Moreover, we have found a quasi-optimal
contention window size for the legacy technique.

    

### [[2112.01949] MARISA: A Self-configuring Metasurfaces Absorption and Reflection Solution Towards 6G](http://arxiv.org/abs/2112.01949)


  Reconfigurable Intelligent Surfaces (RISs) are considered one of the key
disruptive technologies towards future 6G networks. RISs revolutionize the
traditional wireless communication paradigm by controlling the wave propagation
properties of the impinging signals as required. A major roadblock for RIS is
though the need for a fast and complex control channel to continuously adapt to
the ever-changing wireless channel conditions. In this paper, we ask ourselves
the question: Would it be feasible to remove the need for control channels for
RISs? We analyze the feasibility of devising Self-Configuring Smart Surfaces
that can be easily and seamlessly installed throughout the environment,
following the new Internet-of-Surfaces (IoS) paradigm, without requiring
modifications of the deployed mobile network. To this aim, we design MARISA, a
self-configuring metasurfaces absorption and reflection solution, and show that
it can achieve a better-than-expected performance rivaling with control
channel-driven RISs.

    

### [[2112.02058] I-WKNN: Fast-Speed and High-Accuracy WIFI Positioning for Intelligent Stadiums](http://arxiv.org/abs/2112.02058)


  Based on various existing wireless fingerprint location algorithms in
intelligent sports venues, a high-precision and fast indoor location algorithm
improved weighted k-nearest neighbor (I-WKNN) is proposed. In order to meet the
complex environment of sports venues and the demand of high-speed sampling,
this paper proposes an AP selection algorithm for offline and online stages.
Based on the characteristics of the signal intensity distribution in
intelligent venues, an asymmetric Gaussian filter algorithm is proposed. This
paper introduces the application of the positioning algorithm in the
intelligent stadium system, and completes the data acquisition and real-time
positioning of the stadium. Compared with traditional WKNN and KNN algorithms,
the I-WKNN algorithm has advantages in fingerprint positioning database
processing, environmental noise adaptability, real-time positioning accuracy
and positioning speed, etc. The experimental results show that the I-WKNN
algorithm has obvious advantages in positioning accuracy and positioning time
in a complex noise environment and has obvious application potential in a smart
stadium.

    

### [[2112.02080] Improving the Reliability of Network Intrusion Detection Systems through Dataset Integration](http://arxiv.org/abs/2112.02080)


  This work presents Reliable-NIDS (R-NIDS), a novel methodology for Machine
Learning (ML) based Network Intrusion Detection Systems (NIDSs) that allows ML
models to work on integrated datasets, empowering the learning process with
diverse information from different datasets. Therefore, R-NIDS targets the
design of more robust models, that generalize better than traditional
approaches. We also propose a new dataset, called UNK21. It is built from three
of the most well-known network datasets (UGR'16, USNW-NB15 and NLS-KDD), each
one gathered from its own network environment, with different features and
classes, by using a data aggregation approach present in R-NIDS. Following
R-NIDS, in this work we propose to build two well-known ML models (a linear and
a non-linear one) based on the information of three of the most common datasets
in the literature for NIDS evaluation, those integrated in UNK21. The results
that the proposed methodology offers show how these two ML models trained as a
NIDS solution could benefit from this approach, being able to generalize better
when training on the newly proposed UNK21 dataset. Furthermore, these results
are carefully analyzed with statistical tools that provide high confidence on
our conclusions.

    

### [[2112.02083] Energy-Proportional Data Center Network Architecture Through OS, Switch and Laser Co-design](http://arxiv.org/abs/2112.02083)


  Optical interconnects are already the dominant technology in large-scale data
center networks. However, the high optical loss of many optical components
coupled with the low efficiency of laser sources result in high aggregate power
requirements for the thousands of optical transceivers used by these networks.
As optical interconnects stay always on even as traffic demands ebb and flow,
most of this power is wasted. We present LC/DC, a data center network system
architecture in which the operating system, the switch, and the optical
components are co-designed to achieve energy proportionality.
LC/DC capitalizes on the path divergence of data center networks to turn on
and off redundant paths according to traffic demand, while maintaining full
connectivity. Turning off redundant paths allows the optical transceivers and
their electronic drivers to power down and save energy. Maintaining full
connectivity hides the laser turn-on delay. At the node layer, intercepting
send requests within the OS allows for the NIC's laser turn-on delay to be
fully overlapped with TCP/IP packet processing, and thus egress links can
remain powered off until needed with zero performance penalty.
We demonstrate the feasibility of LC/DC by i) implementing the necessary
modifications in the Linux kernel and device drivers, ii) implementing a
10Gbit/s FPGA switch, and iii) performing physical experiments with optical
devices and circuit simulations. Our results on university data center traces
and models of Facebook and Microsoft data center traffic show that LC/DC saves
on average 60% of the optical transceivers power (68% max) at the cost of 6%
higher packet delay.

    

### [[2009.08044] Large-Scale Intelligent Microservices](http://arxiv.org/abs/2009.08044)


  Deploying Machine Learning (ML) algorithms within databases is a challenge
due to the varied computational footprints of modern ML algorithms and the
myriad of database technologies each with its own restrictive syntax. We
introduce an Apache Spark-based micro-service orchestration framework that
extends database operations to include web service primitives. Our system can
orchestrate web services across hundreds of machines and takes full advantage
of cluster, thread, and asynchronous parallelism. Using this framework, we
provide large scale clients for intelligent services such as speech, vision,
search, anomaly detection, and text analysis. This allows users to integrate
ready-to-use intelligence into any datastore with an Apache Spark connector. To
eliminate the majority of overhead from network communication, we also
introduce a low-latency containerized version of our architecture. Finally, we
demonstrate that the services we investigate are competitive on a variety of
benchmarks, and present two applications of this framework to create
intelligent search engines, and real-time auto race analytics systems.

    

### [[2101.02198] Federated Learning over Noisy Channels: Convergence Analysis and Design Examples](http://arxiv.org/abs/2101.02198)


  Does Federated Learning (FL) work when both uplink and downlink
communications have errors? How much communication noise can FL handle and what
is its impact to the learning performance? This work is devoted to answering
these practically important questions by explicitly incorporating both uplink
and downlink noisy channels in the FL pipeline. We present several novel
convergence analyses of FL over simultaneous uplink and downlink noisy
communication channels, which encompass full and partial clients participation,
direct model and model differential transmissions, and non-independent and
identically distributed (IID) local datasets. These analyses characterize the
sufficient conditions for FL over noisy channels to have the same convergence
behavior as the ideal case of no communication error. More specifically, in
order to maintain the O(1/T) convergence rate of FedAvg with perfect
communications, the uplink and downlink signal-to-noise ratio (SNR) for direct
model transmissions should be controlled such that they scale as O(t^2) where t
is the index of communication rounds, but can stay constant for model
differential transmissions. The key insight of these theoretical results is a
"flying under the radar" principle - stochastic gradient descent (SGD) is an
inherent noisy process and uplink/downlink communication noises can be
tolerated as long as they do not dominate the time-varying SGD noise. We
exemplify these theoretical findings with two widely adopted communication
techniques - transmit power control and diversity combining - and further
validating their performance advantages over the standard methods via extensive
numerical experiments using several real-world FL tasks.

    

### [[2112.01533] Automatic tumour segmentation in H&E-stained whole-slide images of the pancreas](http://arxiv.org/abs/2112.01533)


  Pancreatic cancer will soon be the second leading cause of cancer-related
death in Western society. Imaging techniques such as CT, MRI and ultrasound
typically help providing the initial diagnosis, but histopathological
assessment is still the gold standard for final confirmation of disease
presence and prognosis. In recent years machine learning approaches and
pathomics pipelines have shown potential in improving diagnostics and
prognostics in other cancerous entities, such as breast and prostate cancer. A
crucial first step in these pipelines is typically identification and
segmentation of the tumour area. Ideally this step is done automatically to
prevent time consuming manual annotation. We propose a multi-task convolutional
neural network to balance disease detection and segmentation accuracy. We
validated our approach on a dataset of 29 patients (for a total of 58 slides)
at different resolutions. The best single task segmentation network achieved a
median Dice of 0.885 (0.122) IQR at a resolution of 15.56 $\mu$m. Our
multi-task network improved on that with a median Dice score of 0.934 (0.077)
IQR.

    

### [[2112.01535] Robust End-to-End Focal Liver Lesion Detection using Unregistered Multiphase Computed Tomography Images](http://arxiv.org/abs/2112.01535)


  The computer-aided diagnosis of focal liver lesions (FLLs) can help improve
workflow and enable correct diagnoses; FLL detection is the first step in such
a computer-aided diagnosis. Despite the recent success of deep-learning-based
approaches in detecting FLLs, current methods are not sufficiently robust for
assessing misaligned multiphase data. By introducing an attention-guided
multiphase alignment in feature space, this study presents a fully automated,
end-to-end learning framework for detecting FLLs from multiphase computed
tomography (CT) images. Our method is robust to misaligned multiphase images
owing to its complete learning-based approach, which reduces the sensitivity of
the model's performance to the quality of registration and enables a standalone
deployment of the model in clinical practice. Evaluation on a large-scale
dataset with 280 patients confirmed that our method outperformed previous
state-of-the-art methods and significantly reduced the performance degradation
for detecting FLLs using misaligned multiphase CT images. The robustness of the
proposed method can enhance the clinical adoption of the deep-learning-based
computer-aided detection system.

    

### [[2112.01537] Improving mathematical questioning in teacher training](http://arxiv.org/abs/2112.01537)


  High-fidelity, AI-based simulated classroom systems enable teachers to
rehearse effective teaching strategies. However, dialogue-oriented open-ended
conversations such as teaching a student about scale factor can be difficult to
model. This paper presents a high-fidelity, AI-based classroom simulator to
help teachers rehearse research-based mathematical questioning skills. We take
a human centered approach to designing our system, relying advances in
deep-learning, uncertainty quantification and natural language processing while
acknowledging the limitations of conversational agents for specific pedagogical
needs. Using experts' input directly during the simulation, we demonstrate how
conversation success rate and high user satisfaction can be achieved.

    

### [[2112.01555] Is Approximation Universally Defensive Against Adversarial Attacks in Deep Neural Networks?](http://arxiv.org/abs/2112.01555)


  Approximate computing is known for its effectiveness in improvising the
energy efficiency of deep neural network (DNN) accelerators at the cost of
slight accuracy loss. Very recently, the inexact nature of approximate
components, such as approximate multipliers have also been reported successful
in defending adversarial attacks on DNNs models. Since the approximation errors
traverse through the DNN layers as masked or unmasked, this raises a key
research question-can approximate computing always offer a defense against
adversarial attacks in DNNs, i.e., are they universally defensive? Towards
this, we present an extensive adversarial robustness analysis of different
approximate DNN accelerators (AxDNNs) using the state-of-the-art approximate
multipliers. In particular, we evaluate the impact of ten adversarial attacks
on different AxDNNs using the MNIST and CIFAR-10 datasets. Our results
demonstrate that adversarial attacks on AxDNNs can cause 53% accuracy loss
whereas the same attack may lead to almost no accuracy loss (as low as 0.06%)
in the accurate DNN. Thus, approximate computing cannot be referred to as a
universal defense strategy against adversarial attacks.

    

### [[2112.01565] SparRL: Graph Sparsification via Deep Reinforcement Learning](http://arxiv.org/abs/2112.01565)


  Graph sparsification concerns data reduction where an edge-reduced graph of a
similar structure is preferred. Existing methods are mostly sampling-based,
which introduce high computation complexity in general and lack of flexibility
for a different reduction objective. We present SparRL, the first general and
effective reinforcement learning-based framework for graph sparsification.
SparRL can easily adapt to different reduction goals and promise
graph-size-independent complexity. Extensive experiments show that SparRL
outperforms all prevailing sparsification methods in producing high-quality
sparsified graphs concerning a variety of objectives.

    

### [[2112.01566] Theoretical Analysis of an XGBoost Framework for Product Cannibalization](http://arxiv.org/abs/2112.01566)


  This paper is an extension of our work where we presented a three-stage
XGBoost algorithm for forecasting sales under product cannibalization scenario.
Previously we developed the model based on our intuition and provided empirical
evidence on its performance. In this study we would briefly go over the
algorithm and then provide mathematical reasoning behind its working.

    

### [[2112.01570] Trajectory Clustering Performance Evaluation: If we know the answer, it's not clustering](http://arxiv.org/abs/2112.01570)


  Advancements in Intelligent Traffic Systems (ITS) have made huge amounts of
traffic data available through automatic data collection. A big part of this
data is stored as trajectories of moving vehicles and road users. Automatic
analysis of this data with minimal human supervision would both lower the costs
and eliminate subjectivity of the analysis. Trajectory clustering is an
unsupervised task.
In this paper, we perform a comprehensive comparison of similarity measures,
clustering algorithms and evaluation measures using trajectory data from seven
intersections. We also propose a method to automatically generate trajectory
reference clusters based on their origin and destination points to be used for
label-based evaluation measures. Therefore, the entire procedure remains
unsupervised both in clustering and evaluation levels. Finally, we use a
combination of evaluation measures to find the top performing similarity
measures and clustering algorithms for each intersection. The results show that
there is no single combination of distance and clustering algorithm that is
always among the top ten clustering setups.

    

### [[2112.01573] FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization](http://arxiv.org/abs/2112.01573)


  Generating images from natural language instructions is an intriguing yet
highly challenging task. We approach text-to-image generation by combining the
power of the retrained CLIP representation with an off-the-shelf image
generator (GANs), optimizing in the latent space of GAN to find images that
achieve maximum CLIP score with the given input text. Compared to traditional
methods that train generative models from text to image starting from scratch,
the CLIP+GAN approach is training-free, zero shot and can be easily customized
with different generators.
However, optimizing CLIP score in the GAN space casts a highly challenging
optimization problem and off-the-shelf optimizers such as Adam fail to yield
satisfying results. In this work, we propose a FuseDream pipeline, which
improves the CLIP+GAN approach with three key techniques: 1) an AugCLIP score
which robustifies the CLIP objective by introducing random augmentation on
image. 2) a novel initialization and over-parameterization strategy for
optimization which allows us to efficiently navigate the non-convex landscape
in GAN space. 3) a composed generation technique which, by leveraging a novel
bi-level optimization formulation, can compose multiple images to extend the
GAN space and overcome the data-bias.
When promoted by different input text, FuseDream can generate high-quality
images with varying objects, backgrounds, artistic styles, even novel
counterfactual concepts that do not appear in the training data of the GAN we
use. Quantitatively, the images generated by FuseDream yield top-level
Inception score and FID score on MS COCO dataset, without additional
architecture design or training. Our code is publicly available at
\url{this https URL}.

    

### [[2112.01574] Dimension-Free Average Treatment Effect Inference with Deep Neural Networks](http://arxiv.org/abs/2112.01574)


  This paper investigates the estimation and inference of the average treatment
effect (ATE) using deep neural networks (DNNs) in the potential outcomes
framework. Under some regularity conditions, the observed response can be
formulated as the response of a mean regression problem with both the
confounding variables and the treatment indicator as the independent variables.
Using such formulation, we investigate two methods for ATE estimation and
inference based on the estimated mean regression function via DNN regression
using a specific network architecture. We show that both DNN estimates of ATE
are consistent with dimension-free consistency rates under some assumptions on
the underlying true mean regression model. Our model assumptions accommodate
the potentially complicated dependence structure of the observed response on
the covariates, including latent factors and nonlinear interactions between the
treatment indicator and confounding variables. We also establish the asymptotic
normality of our estimators based on the idea of sample splitting, ensuring
precise inference and uncertainty quantification. Simulation studies and real
data application justify our theoretical findings and support our DNN
estimation and inference methods.

    

### [[2112.01575] Towards Intrinsic Interactive Reinforcement Learning: A Survey](http://arxiv.org/abs/2112.01575)


  Reinforcement learning (RL) and brain-computer interfaces (BCI) are two
fields that have been growing over the past decade. Until recently, these
fields have operated independently of one another. With the rising interest in
human-in-the-loop (HITL) applications, RL algorithms have been adapted to
account for human guidance giving rise to the sub-field of interactive
reinforcement learning (IRL). Adjacently, BCI applications have been long
interested in extracting intrinsic feedback from neural activity during
human-computer interactions. These two ideas have set RL and BCI on a collision
course for one another through the integration of BCI into the IRL framework
where intrinsic feedback can be utilized to help train an agent. This
intersection has been denoted as intrinsic IRL. To further help facilitate
deeper ingratiation of BCI and IRL, we provide a review of intrinsic IRL with
an emphasis on its parent field of feedback-driven IRL while also providing
discussions concerning the validity, challenges, and future research
directions.

    

### [[2112.01576] Scheduling to Learn In An Unsupervised Online Streaming Model](http://arxiv.org/abs/2112.01576)


  An unsupervised online streaming model is considered where samples arrive in
an online fashion over $T$ slots. There are $M$ classifiers, whose confusion
matrices are unknown a priori. In each slot, at most one sample can be labeled
by any classifier. The accuracy of a sample is a function of the set of labels
obtained for it from various classifiers. The utility of a sample is a scalar
multiple of its accuracy minus the response time (difference of the departure
slot and the arrival slot), where the departure slot is also decided by the
algorithm. Since each classifier can label at most one sample per slot, there
is a tradeoff between obtaining a larger set of labels for a particular sample
to improve its accuracy, and its response time. The problem of maximizing the
sum of the utilities of all samples is considered, where learning the confusion
matrices, sample-classifier matching assignment, and sample departure slot
decisions depend on each other. The proposed algorithm first learns the
confusion matrices, and then uses a greedy algorithm for sample-classifier
matching. A sample departs once its incremental utility turns non-positive. We
show that the competitive ratio of the proposed algorithm is
$\frac{1}{2}-{\mathcal O}\left(\frac{\log T}{T}\right)$.

    

### [[2112.01578] Invariant Priors for Bayesian Quadrature](http://arxiv.org/abs/2112.01578)


  Bayesian quadrature (BQ) is a model-based numerical integration method that
is able to increase sample efficiency by encoding and leveraging known
structure of the integration task at hand. In this paper, we explore priors
that encode invariance of the integrand under a set of bijective
transformations in the input domain, in particular some unitary
transformations, such as rotations, axis-flips, or point symmetries. We show
initial results on superior performance in comparison to standard Bayesian
quadrature on several synthetic and one real world application.

    

### [[2112.01582] LeapfrogLayers: A Trainable Framework for Effective Topological Sampling](http://arxiv.org/abs/2112.01582)


  We introduce LeapfrogLayers, an invertible neural network architecture that
can be trained to efficiently sample the topology of a 2D $U(1)$ lattice gauge
theory. We show an improvement in the integrated autocorrelation time of the
topological charge when compared with traditional HMC, and propose methods for
scaling our model to larger lattice volumes. Our implementation is open source,
and is publicly available on github at this https URL


### [[2112.01583] The Representation Jensen-Renyí Divergence](http://arxiv.org/abs/2112.01583)


  We introduce a divergence measure between data distributions based on
operators in reproducing kernel Hilbert spaces defined by infinitely divisible
kernels. The empirical estimator of the divergence is computed using the
eigenvalues of positive definite matrices that are obtained by evaluating the
kernel over pairs of samples. The new measure shares similar properties to
Jensen-Shannon divergence. Convergence of the proposed estimators follows from
concentration results based on the difference between the ordered spectrum of
the Gram matrices and the integral operators associated with the population
quantities. The proposed measure of divergence avoids the estimation of the
probability distribution underlying the data. Numerical experiments involving
comparing distributions and applications to sampling unbalanced data for
classification show that the proposed divergence can achieve state of the art
results.

    

### [[2112.01585] Differentially Private Exploration in Reinforcement Learning with Linear Representation](http://arxiv.org/abs/2112.01585)


  This paper studies privacy-preserving exploration in Markov Decision
Processes (MDPs) with linear representation. We first consider the setting of
linear-mixture MDPs (Ayoub et al., 2020) (a.k.a.\ model-based setting) and
provide an unified framework for analyzing joint and local differential private
(DP) exploration. Through this framework, we prove a
$\widetilde{O}(K^{3/4}/\sqrt{\epsilon})$ regret bound for
$(\epsilon,\delta)$-local DP exploration and a
$\widetilde{O}(\sqrt{K/\epsilon})$ regret bound for $(\epsilon,\delta)$-joint
DP.
We further study privacy-preserving exploration in linear MDPs (Jin et al.,
2020) (a.k.a.\ model-free setting) where we provide a
$\widetilde{O}(\sqrt{K/\epsilon})$ regret bound for $(\epsilon,\delta)$-joint
DP, with a novel algorithm based on low-switching. Finally, we provide insights
into the issues of designing local DP algorithms in this model-free setting.

    

### [[2112.01586] HMC with Normalizing Flows](http://arxiv.org/abs/2112.01586)


  We propose using Normalizing Flows as a trainable kernel within the molecular
dynamics update of Hamiltonian Monte Carlo (HMC). By learning (invertible)
transformations that simplify our dynamics, we can outperform traditional
methods at generating independent configurations. We show that, using a
carefully constructed network architecture, our approach can be easily scaled
to large lattice volumes with minimal retraining effort. The source code for
our implementation is publicly available online at
this https URL.

    

### [[2112.01591] PLSUM: Generating PT-BR Wikipedia by Summarizing Multiple Websites](http://arxiv.org/abs/2112.01591)


  Wikipedia is an important free source of intelligible knowledge. Despite
that, Brazilian Portuguese Wikipedia still lacks descriptions for many
subjects. In an effort to expand the Brazilian Wikipedia, we contribute PLSum,
a framework for generating wiki-like abstractive summaries from multiple
descriptive websites. The framework has an extractive stage followed by an
abstractive one. In particular, for the abstractive stage, we fine-tune and
compare two recent variations of the Transformer neural network, PTT5, and
Longformer. To fine-tune and evaluate the model, we created a dataset with
thousands of examples, linking reference websites to Wikipedia. Our results
show that it is possible to generate meaningful abstractive summaries from
Brazilian Portuguese web content.

    

### [[2112.01592] Online Search With Best-Price and Query-Based Predictions](http://arxiv.org/abs/2112.01592)


  In the online (time-series) search problem, a player is presented with a
sequence of prices which are revealed in an online manner. In the standard
definition of the problem, for each revealed price, the player must decide
irrevocably whether to accept or reject it, without knowledge of future prices
(other than an upper and a lower bound on their extreme values), and the
objective is to minimize the competitive ratio, namely the worst-case ratio
between the maximum price in the sequence and the one selected by the player.
The problem formulates several applications of decision-making in the face of
uncertainty on the revealed samples.
Previous work on this problem has largely assumed extreme scenarios in which
either the player has almost no information about the input, or the player is
provided with some powerful, and error-free advice. In this work, we study
learning-augmented algorithms, in which there is a potentially erroneous
prediction concerning the input. Specifically, we consider two different
settings: the setting in which the prediction is related to the maximum price
in the sequence, as well as the setting in which the prediction is obtained as
a response to a number of binary queries. For both settings, we provide tight,
or near-tight upper and lower bounds on the worst-case performance of search
algorithms as a function of the prediction error. We also provide experimental
results on data obtained from stock exchange markets that confirm the
theoretical analysis, and explain how our techniques can be applicable to other
learning-augmented applications.

    

### [[2112.01603] Neurosymbolic Systems of Perception & Cognition: The Role of Attention](http://arxiv.org/abs/2112.01603)


  A cognitive architecture aimed at cumulative learning must provide the
necessary information and control structures to allow agents to learn
incrementally and autonomously from their experience. This involves managing an
agent's goals as well as continuously relating sensory information to these in
its perception-cognition information stack. The more varied the environment of
a learning agent is, the more general and flexible must be these mechanisms to
handle a wider variety of relevant patterns, tasks, and goal structures. While
many researchers agree that information at different levels of abstraction
likely differs in its makeup and structure and processing mechanisms, agreement
on the particulars of such differences is not generally shared in the research
community. A binary processing architecture (often referred to as System-1 and
System-2) has been proposed as a model of cognitive processing for low- and
high-level information, respectively. We posit that cognition is not binary in
this way and that knowledge at any level of abstraction involves what we refer
to as neurosymbolic information, meaning that data at both high and low levels
must contain both symbolic and subsymbolic information. Further, we argue that
the main differentiating factor between the processing of high and low levels
of data abstraction can be largely attributed to the nature of the involved
attention mechanisms. We describe the key arguments behind this view and review
relevant evidence from the literature.

    

### [[2112.01617] Label noise detection under the Noise at Random model with ensemble filters](http://arxiv.org/abs/2112.01617)


  Label noise detection has been widely studied in Machine Learning because of
its importance in improving training data quality. Satisfactory noise detection
has been achieved by adopting ensembles of classifiers. In this approach, an
instance is assigned as mislabeled if a high proportion of members in the pool
misclassifies it. Previous authors have empirically evaluated this approach;
nevertheless, they mostly assumed that label noise is generated completely at
random in a dataset. This is a strong assumption since other types of label
noise are feasible in practice and can influence noise detection results. This
work investigates the performance of ensemble noise detection under two
different noise models: the Noisy at Random (NAR), in which the probability of
label noise depends on the instance class, in comparison to the Noisy
Completely at Random model, in which the probability of label noise is entirely
independent. In this setting, we investigate the effect of class distribution
on noise detection performance since it changes the total noise level observed
in a dataset under the NAR assumption. Further, an evaluation of the ensemble
vote threshold is conducted to contrast with the most common approaches in the
literature. In many performed experiments, choosing a noise generation model
over another can lead to different results when considering aspects such as
class imbalance and noise level ratio among different classes.

    

### [[2112.01624] A Survey on Awesome Korean NLP Datasets](http://arxiv.org/abs/2112.01624)


  English based datasets are commonly available from Kaggle, GitHub, or
recently published papers. Although benchmark tests with English datasets are
sufficient to show off the performances of new models and methods, still a
researcher need to train and validate the models on Korean based datasets to
produce a technology or product, suitable for Korean processing. This paper
introduces 15 popular Korean based NLP datasets with summarized details such as
volume, license, repositories, and other research results inspired by the
datasets. Also, I provide high-resolution instructions with sample or
statistics of datasets. The main characteristics of datasets are presented on a
single table to provide a rapid summarization of datasets for researchers.

    

### [[2112.01625] Sample-Efficient Generation of Novel Photo-acid Generator Molecules using a Deep Generative Model](http://arxiv.org/abs/2112.01625)


  Photo-acid generators (PAGs) are compounds that release acids ($H^+$ ions)
when exposed to light. These compounds are critical components of the
photolithography processes that are used in the manufacture of semiconductor
logic and memory chips. The exponential increase in the demand for
semiconductors has highlighted the need for discovering novel photo-acid
generators. While de novo molecule design using deep generative models has been
widely employed for drug discovery and material design, its application to the
creation of novel photo-acid generators poses several unique challenges, such
as lack of property labels. In this paper, we highlight these challenges and
propose a generative modeling approach that utilizes conditional generation
from a pre-trained deep autoencoder and expert-in-the-loop techniques. The
validity of the proposed approach was evaluated with the help of subject matter
experts, indicating the promise of such an approach for applications beyond the
creation of novel photo-acid generators.

    

### [[2112.01627] High-Precision Inversion of Dynamic Radiography Using Hydrodynamic Features](http://arxiv.org/abs/2112.01627)


  Radiography is often used to probe complex, evolving density fields in
dynamic systems and in so doing gain insight into the underlying physics. This
technique has been used in numerous fields including materials science, shock
physics, inertial confinement fusion, and other national security applications.
In many of these applications, however, complications resulting from noise,
scatter, complex beam dynamics, etc. prevent the reconstruction of density from
being accurate enough to identify the underlying physics with sufficient
confidence. As such, density reconstruction from static/dynamic radiography has
typically been limited to identifying discontinuous features such as cracks and
voids in a number of these applications.
In this work, we propose a fundamentally new approach to reconstructing
density from a temporal sequence of radiographic images. Using only the robust
features identifiable in radiographs, we combine them with the underlying
hydrodynamic equations of motion using a machine learning approach, namely,
conditional generative adversarial networks (cGAN), to determine the density
fields from a dynamic sequence of radiographs. Next, we seek to further enhance
the hydrodynamic consistency of the ML-based density reconstruction through a
process of parameter estimation and projection onto a hydrodynamic manifold. In
this context, we note that the distance from the hydrodynamic manifold given by
the training data to the test data in the parameter space considered both
serves as a diagnostic of the robustness of the predictions and serves to
augment the training database, with the expectation that the latter will
further reduce future density reconstruction errors. Finally, we demonstrate
the ability of this method to outperform a traditional radiographic
reconstruction in capturing allowable hydrodynamic paths even when relatively
small amounts of scatter are present.

    

### [[2112.01637] AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed Deep Learning](http://arxiv.org/abs/2112.01637)


  Distributed deep learning frameworks like federated learning (FL) and its
variants are enabling personalized experiences across a wide range of web
clients and mobile/IoT devices. However, FL-based frameworks are constrained by
computational resources at clients due to the exploding growth of model
parameters (eg. billion parameter model). Split learning (SL), a recent
framework, reduces client compute load by splitting the model training between
client and server. This flexibility is extremely useful for low-compute setups
but is often achieved at cost of increase in bandwidth consumption and may
result in sub-optimal convergence, especially when client data is
heterogeneous. In this work, we introduce AdaSplit which enables efficiently
scaling SL to low resource scenarios by reducing bandwidth consumption and
improving performance across heterogeneous clients. To capture and benchmark
this multi-dimensional nature of distributed deep learning, we also introduce
C3-Score, a metric to evaluate performance under resource budgets. We validate
the effectiveness of AdaSplit under limited resources through extensive
experimental comparison with strong federated and split learning baselines. We
also present a sensitivity analysis of key design choices in AdaSplit which
validates the ability of AdaSplit to provide adaptive trade-offs across
variable resource budgets.

    

### [[2112.01641] Hamiltonian prior to Disentangle Content and Motion in Image Sequences](http://arxiv.org/abs/2112.01641)


  We present a deep latent variable model for high dimensional sequential data.
Our model factorises the latent space into content and motion variables. To
model the diverse dynamics, we split the motion space into subspaces, and
introduce a unique Hamiltonian operator for each subspace. The Hamiltonian
formulation provides reversible dynamics that learn to constrain the motion
path to conserve invariant properties. The explicit split of the motion space
decomposes the Hamiltonian into symmetry groups and gives long-term
separability of the dynamics. This split also means representations can be
learnt that are easy to interpret and control. We demonstrate the utility of
our model for swapping the motion of two videos, generating sequences of
various actions from a given image and unconditional sequence generation.

    

### [[2112.01642] Probabilistic Contrastive Loss for Self-Supervised Learning](http://arxiv.org/abs/2112.01642)


  This paper proposes a probabilistic contrastive loss function for
self-supervised learning. The well-known contrastive loss is deterministic and
involves a temperature hyperparameter that scales the inner product between two
normed feature embeddings. By reinterpreting the temperature hyperparameter as
a quantity related to the radius of the hypersphere, we derive a new loss
function that involves a confidence measure which quantifies uncertainty in a
mathematically grounding manner. Some intriguing properties of the proposed
loss function are empirically demonstrated, which agree with human-like
predictions. We believe the present work brings up a new prospective to the
area of contrastive learning.

    

### [[2112.01653] Learning Curves for Sequential Training of Neural Networks: Self-Knowledge Transfer and Forgetting](http://arxiv.org/abs/2112.01653)


  Sequential training from task to task is becoming one of the major objects in
deep learning applications such as continual learning and transfer learning.
Nevertheless, it remains unclear under what conditions the trained model's
performance improves or deteriorates. To deepen our understanding of sequential
training, this study provides a theoretical analysis of generalization
performance in a solvable case of continual learning. We consider neural
networks in the neural tangent kernel (NTK) regime that continually learn
target functions from task to task, and investigate the generalization by using
an established statistical mechanical analysis of kernel ridge-less regression.
We first show characteristic transitions from positive to negative transfer.
More similar targets above a specific critical value can achieve positive
knowledge transfer for the subsequent task while catastrophic forgetting occurs
even with very similar targets. Next, we investigate a variant of continual
learning where the model learns the same target function in multiple tasks.
Even for the same target, the trained model shows some transfer and forgetting
depending on the sample size of each task. We can guarantee that the
generalization error monotonically decreases from task to task for equal sample
sizes while unbalanced sample sizes deteriorate the generalization. We
respectively refer to these improvement and deterioration as self-knowledge
transfer and forgetting, and empirically confirm them in realistic training of
deep neural networks as well.

    

### [[2112.01675] Challenges and Opportunities in Approximate Bayesian Deep Learning for Intelligent IoT Systems](http://arxiv.org/abs/2112.01675)


  Approximate Bayesian deep learning methods hold significant promise for
addressing several issues that occur when deploying deep learning components in
intelligent systems, including mitigating the occurrence of over-confident
errors and providing enhanced robustness to out of distribution examples.
However, the computational requirements of existing approximate Bayesian
inference methods can make them ill-suited for deployment in intelligent IoT
systems that include lower-powered edge devices. In this paper, we present a
range of approximate Bayesian inference methods for supervised deep learning
and highlight the challenges and opportunities when applying these methods on
current edge hardware. We highlight several potential solutions to decreasing
model storage requirements and improving computational scalability, including
model pruning and distillation methods.

    

### [[2112.01687] Differential Property Prediction: A Machine Learning Approach to Experimental Design in Advanced Manufacturing](http://arxiv.org/abs/2112.01687)


  Advanced manufacturing techniques have enabled the production of materials
with state-of-the-art properties. In many cases however, the development of
physics-based models of these techniques lags behind their use in the lab. This
means that designing and running experiments proceeds largely via trial and
error. This is sub-optimal since experiments are cost-, time-, and
labor-intensive. In this work we propose a machine learning framework,
differential property classification (DPC), which enables an experimenter to
leverage machine learning's unparalleled pattern matching capability to pursue
data-driven experimental design. DPC takes two possible experiment parameter
sets and outputs a prediction of which will produce a material with a more
desirable property specified by the operator. We demonstrate the success of DPC
on AA7075 tube manufacturing process and mechanical property data using shear
assisted processing and extrusion (ShAPE), a solid phase processing technology.
We show that by focusing on the experimenter's need to choose between multiple
candidate experimental parameters, we can reframe the challenging regression
task of predicting material properties from processing parameters, into a
classification task on which machine learning models can achieve good
performance.

    

### [[2112.01688] Machine Learning Subsystem for Autonomous Collision Avoidance on a small UAS with Embedded GPU](http://arxiv.org/abs/2112.01688)


  Interest in unmanned aerial system (UAS) powered solutions for 6G
communication networks has grown immensely with the widespread availability of
machine learning based autonomy modules and embedded graphical processing units
(GPUs). While these technologies have revolutionized the possibilities of UAS
solutions, designing an operable, robust autonomy framework for UAS remains a
multi-faceted and difficult problem. In this work, we present our novel,
modular framework for UAS autonomy, entitled MR-iFLY, and discuss how it may be
extended to enable 6G swarm solutions. We begin by detailing the challenges
associated with machine learning based UAS autonomy on resource constrained
devices. Next, we describe in depth, how MR-iFLY's novel depth estimation and
collision avoidance technology meets these challenges. Lastly, we describe the
various evaluation criteria we have used to measure performance, show how our
optimized machine vision components provide up to 15X speedup over baseline
models and present a flight demonstration video of MR-iFLY's vision-based
collision avoidance technology. We argue that these empirical results
substantiate MR-iFLY as a candidate for use in reducing communication overhead
between nodes in 6G communication swarms by providing standalone collision
avoidance and navigation capabilities.

    

### [[2112.01694] On the Existence of the Adversarial Bayes Classifier (Extended Version)](http://arxiv.org/abs/2112.01694)


  Adversarial robustness is a critical property in a variety of modern machine
learning applications. While it has been the subject of several recent
theoretical studies, many important questions related to adversarial robustness
are still open. In this work, we study a fundamental question regarding Bayes
optimality for adversarial robustness. We provide general sufficient conditions
under which the existence of a Bayes optimal classifier can be guaranteed for
adversarial robustness. Our results can provide a useful tool for a subsequent
study of surrogate losses in adversarial robustness and their consistency
properties. This manuscript is the extended version of the paper "On the
Existence of the Adversarial Bayes Classifier" published in NeurIPS. The
results of the original paper did not apply to some non-strictly convex norms.
Here we extend our results to all possible norms.

    

### [[2112.01697] LMR-CBT: Learning Modality-fused Representations with CB-Transformer for Multimodal Emotion Recognition from Unaligned Multimodal Sequences](http://arxiv.org/abs/2112.01697)


  Learning modality-fused representations and processing unaligned multimodal
sequences are meaningful and challenging in multimodal emotion recognition.
Existing approaches use directional pairwise attention or a message hub to fuse
language, visual, and audio modalities. However, those approaches introduce
information redundancy when fusing features and are inefficient without
considering the complementarity of modalities. In this paper, we propose an
efficient neural network to learn modality-fused representations with
CB-Transformer (LMR-CBT) for multimodal emotion recognition from unaligned
multimodal sequences. Specifically, we first perform feature extraction for the
three modalities respectively to obtain the local structure of the sequences.
Then, we design a novel transformer with cross-modal blocks (CB-Transformer)
that enables complementary learning of different modalities, mainly divided
into local temporal learning,cross-modal feature fusion and global
self-attention representations. In addition, we splice the fused features with
the original features to classify the emotions of the sequences. Finally, we
conduct word-aligned and unaligned experiments on three challenging datasets,
IEMOCAP, CMU-MOSI, and CMU-MOSEI. The experimental results show the superiority
and efficiency of our proposed method in both settings. Compared with the
mainstream methods, our approach reaches the state-of-the-art with a minimum
number of parameters.

    

### [[2112.01713] Contrastive Continual Learning with Feature Propagation](http://arxiv.org/abs/2112.01713)


  Classical machine learners are designed only to tackle one task without
capability of adopting new emerging tasks or classes whereas such capacity is
more practical and human-like in the real world. To address this shortcoming,
continual machine learners are elaborated to commendably learn a stream of
tasks with domain and class shifts among different tasks. In this paper, we
propose a general feature-propagation based contrastive continual learning
method which is capable of handling multiple continual learning scenarios.
Specifically, we align the current and previous representation spaces by means
of feature propagation and contrastive representation learning to bridge the
domain shifts among distinct tasks. To further mitigate the class-wise shifts
of the feature representation, a supervised contrastive loss is exploited to
make the example embeddings of the same class closer than those of different
classes. The extensive experimental results demonstrate the outstanding
performance of the proposed method on six continual learning benchmarks
compared to a group of cutting-edge continual learning methods.

    

### [[2112.01714] Structure-Aware Multi-Hop Graph Convolution for Graph Neural Networks](http://arxiv.org/abs/2112.01714)


  In this paper, we propose a spatial graph convolution (GC) to classify
signals on a graph. Existing GC methods are limited to using the structural
information in the feature space. Additionally, the single step of GCs only
uses features on the one-hop neighboring nodes from the target node. In this
paper, we propose two methods to improve the performance of GCs: 1) Utilizing
structural information in the feature space, and 2) exploiting the multi-hop
information in one GC step. In the first method, we define three structural
features in the feature space: feature angle, feature distance, and relational
embedding. The second method aggregates the node-wise features of multi-hop
neighbors in a GC. Both methods can be simultaneously used. We also propose
graph neural networks (GNNs) integrating the proposed GC for classifying nodes
in 3D point clouds and citation networks. In experiments, the proposed GNNs
exhibited a higher classification accuracy than existing methods.

    

### [[2112.01716] Reduced, Reused and Recycled: The Life of a Dataset in Machine Learning Research](http://arxiv.org/abs/2112.01716)


  Benchmark datasets play a central role in the organization of machine
learning research. They coordinate researchers around shared research problems
and serve as a measure of progress towards shared goals. Despite the
foundational role of benchmarking practices in this field, relatively little
attention has been paid to the dynamics of benchmark dataset use and reuse,
within or across machine learning subcommunities. In this paper, we dig into
these dynamics. We study how dataset usage patterns differ across machine
learning subcommunities and across time from 2015-2020. We find increasing
concentration on fewer and fewer datasets within task communities, significant
adoption of datasets from other tasks, and concentration across the field on
datasets that have been introduced by researchers situated within a small
number of elite institutions. Our results have implications for scientific
evaluation, AI ethics, and equity/access within the field.

    

### [[2112.01718] Improving Predictions of Tail-end Labels using Concatenated BioMed-Transformers for Long Medical Documents](http://arxiv.org/abs/2112.01718)


  Multi-label learning predicts a subset of labels from a given label set for
an unseen instance while considering label correlations. A known challenge with
multi-label classification is the long-tailed distribution of labels. Many
studies focus on improving the overall predictions of the model and thus do not
prioritise tail-end labels. Improving the tail-end label predictions in
multi-label classifications of medical text enables the potential to understand
patients better and improve care. The knowledge gained by one or more
infrequent labels can impact the cause of medical decisions and treatment
plans. This research presents variations of concatenated domain-specific
language models, including multi-BioMed-Transformers, to achieve two primary
goals. First, to improve F1 scores of infrequent labels across multi-label
problems, especially with long-tail labels; second, to handle long medical text
and multi-sourced electronic health records (EHRs), a challenging task for
standard transformers designed to work on short input sequences. A vital
contribution of this research is new state-of-the-art (SOTA) results obtained
using TransformerXL for predicting medical codes. A variety of experiments are
performed on the Medical Information Mart for Intensive Care (MIMIC-III)
database. Results show that concatenated BioMed-Transformers outperform
standard transformers in terms of overall micro and macro F1 scores and
individual F1 scores of tail-end labels, while incurring lower training times
than existing transformer-based solutions for long input sequences.

    

### [[2112.01719] Adaptive Poincaré Point to Set Distance for Few-Shot Classification](http://arxiv.org/abs/2112.01719)


  Learning and generalizing from limited examples, i,e, few-shot learning, is
of core importance to many real-world vision applications. A principal way of
achieving few-shot learning is to realize an embedding where samples from
different classes are distinctive. Recent studies suggest that embedding via
hyperbolic geometry enjoys low distortion for hierarchical and structured data,
making it suitable for few-shot learning. In this paper, we propose to learn a
context-aware hyperbolic metric to characterize the distance between a point
and a set associated with a learned set to set distance. To this end, we
formulate the metric as a weighted sum on the tangent bundle of the hyperbolic
space and develop a mechanism to obtain the weights adaptively and based on the
constellation of the points. This not only makes the metric local but also
dependent on the task in hand, meaning that the metric will adapt depending on
the samples that it compares. We empirically show that such metric yields
robustness in the presence of outliers and achieves a tangible improvement over
baseline models. This includes the state-of-the-art results on five popular
few-shot classification benchmarks, namely mini-ImageNet, tiered-ImageNet,
Caltech-UCSD Birds-200-2011 (CUB), CIFAR-FS, and FC100.

    

### [[2112.01741] Frame Averaging for Equivariant Shape Space Learning](http://arxiv.org/abs/2112.01741)


  The task of shape space learning involves mapping a train set of shapes to
and from a latent representation space with good generalization properties.
Often, real-world collections of shapes have symmetries, which can be defined
as transformations that do not change the essence of the shape. A natural way
to incorporate symmetries in shape space learning is to ask that the mapping to
the shape space (encoder) and mapping from the shape space (decoder) are
equivariant to the relevant symmetries.
In this paper, we present a framework for incorporating equivariance in
encoders and decoders by introducing two contributions: (i) adapting the recent
Frame Averaging (FA) framework for building generic, efficient, and maximally
expressive Equivariant autoencoders; and (ii) constructing autoencoders
equivariant to piecewise Euclidean motions applied to different parts of the
shape. To the best of our knowledge, this is the first fully piecewise
Euclidean equivariant autoencoder construction. Training our framework is
simple: it uses standard reconstruction losses and does not require the
introduction of new losses. Our architectures are built of standard (backbone)
architectures with the appropriate frame averaging to make them equivariant.
Testing our framework on both rigid shapes dataset using implicit neural
representations, and articulated shape datasets using mesh-based neural
networks show state-of-the-art generalization to unseen test shapes, improving
relevant baselines by a large margin. In particular, our method demonstrates
significant improvement in generalizing to unseen articulated poses.

    

### [[2112.01753] Probing Linguistic Information For Logical Inference In Pre-trained Language Models](http://arxiv.org/abs/2112.01753)


  Progress in pre-trained language models has led to a surge of impressive
results on downstream tasks for natural language understanding. Recent work on
probing pre-trained language models uncovered a wide range of linguistic
properties encoded in their contextualized representations. However, it is
unclear whether they encode semantic knowledge that is crucial to symbolic
inference methods. We propose a methodology for probing linguistic information
for logical inference in pre-trained language model representations. Our
probing datasets cover a list of linguistic phenomena required by major
symbolic inference systems. We find that (i) pre-trained language models do
encode several types of linguistic information for inference, but there are
also some types of information that are weakly encoded, (ii) language models
can effectively learn missing linguistic information through fine-tuning.
Overall, our findings provide insights into which aspects of linguistic
information for logical inference do language models and their pre-training
procedures capture. Moreover, we have demonstrated language models' potential
as semantic and background knowledge bases for supporting symbolic inference
methods.

    

### [[2112.01765] Learning Emergent Random Access Protocol for LEO Satellite Networks](http://arxiv.org/abs/2112.01765)


  A mega-constellation of low-altitude earth orbit (LEO) satellites (SATs) are
envisaged to provide a global coverage SAT network in beyond fifth-generation
(5G) cellular systems. LEO SAT networks exhibit extremely long link distances
of many users under time-varying SAT network topology. This makes existing
multiple access protocols, such as random access channel (RACH) based cellular
protocol designed for fixed terrestrial network topology, ill-suited. To
overcome this issue, in this paper, we propose a novel grant-free random access
solution for LEO SAT networks, dubbed emergent random access channel protocol
(eRACH). In stark contrast to existing model-based and standardized protocols,
eRACH is a model-free approach that emerges through interaction with the
non-stationary network environment, using multi-agent deep reinforcement
learning (MADRL). Furthermore, by exploiting known SAT orbiting patterns, eRACH
does not require central coordination or additional communication across users,
while training convergence is stabilized through the regular orbiting patterns.
Compared to RACH, we show from various simulations that our proposed eRACH
yields 54.6% higher average network throughput with around two times lower
average access delay while achieving 0.989 Jain's fairness index.

    

### [[2112.01767] MT-TransUNet: Mediating Multi-Task Tokens in Transformers for Skin Lesion Segmentation and Classification](http://arxiv.org/abs/2112.01767)


  Recent advances in automated skin cancer diagnosis have yielded performance
on par with board-certified dermatologists. However, these approaches
formulated skin cancer diagnosis as a simple classification task, dismissing
the potential benefit from lesion segmentation. We argue that an accurate
lesion segmentation can supplement the classification task with additive lesion
information, such as asymmetry, border, intensity, and physical size; in turn,
a faithful lesion classification can support the segmentation task with
discriminant lesion features. To this end, this paper proposes a new multi-task
framework, named MT-TransUNet, which is capable of segmenting and classifying
skin lesions collaboratively by mediating multi-task tokens in Transformers.
Furthermore, we have introduced dual-task and attended region consistency
losses to take advantage of those images without pixel-level annotation,
ensuring the model's robustness when it encounters the same image with an
account of augmentation. Our MT-TransUNet exceeds the previous state of the art
for lesion segmentation and classification tasks in ISIC-2017 and PH2; more
importantly, it preserves compelling computational efficiency regarding model
parameters (48M~vs.~130M) and inference speed (0.17s~vs.~2.02s per image). Code
will be available at this https URL.

    

### [[2112.01769] Prescriptive Process Monitoring: Quo Vadis?](http://arxiv.org/abs/2112.01769)


  Prescriptive process monitoring methods seek to optimize a business process
by recommending interventions at runtime to prevent negative outcomes or poorly
performing cases. In recent years, various prescriptive process monitoring
methods have been proposed. This paper studies existing methods in this field
via a Systematic Literature Review (SLR). In order to structure the field, the
paper proposes a framework for characterizing prescriptive process monitoring
methods according to their performance objective, performance metrics,
intervention types, modeling techniques, data inputs, and intervention
policies. The SLR provides insights into challenges and areas for future
research that could enhance the usefulness and applicability of prescriptive
process monitoring methods. The paper highlights the need to validate existing
and new methods in real-world settings, to extend the types of interventions
beyond those related to the temporal and cost perspectives, and to design
policies that take into account causality and second-order effects.

    

### [[2112.01771] Characterizing Performance Bugs in Deep Learning Systems](http://arxiv.org/abs/2112.01771)


  Deep learning (DL) has been increasingly applied to a variety of domains. The
programming paradigm shift from traditional systems to DL systems poses unique
challenges in engineering DL systems. Performance is one of the challenges, and
performance bugs(PBs) in DL systems can cause severe consequences such as
excessive resource consumption and financial loss. While bugs in DL systems
have been extensively investigated, PBs in DL systems have hardly been
explored. To bridge this gap, we present the first comprehensive study to
characterize symptoms, root causes, and introducing and exposing stages of PBs
in DL systems developed in TensorFLow and Keras, with a total of 238 PBs
collected from 225 StackOverflow posts. Our findings shed light on the
implications on developing high performance DL systems, and detecting and
localizing PBs in DL systems. We also build the first benchmark of 56 PBs in DL
systems, and assess the capability of existing approaches in tackling them.
Moreover, we develop a static checker DeepPerf to detect three types of PBs,
and identify 488 new PBs in 130 GitHub projects.62 and 18 of them have been
respectively confirmed and fixed by developers.

    

### [[2112.01773] Residual-Based Adaptive Coefficient and Noise-Immunity ZNN for Perturbed Time-Dependent Quadratic Minimization](http://arxiv.org/abs/2112.01773)


  The time-dependent quadratic minimization (TDQM) problem appears in many
applications and research projects. It has been reported that the zeroing
neural network (ZNN) models can effectively solve the TDQM problem. However,
the convergent and robust performance of the existing ZNN models are restricted
for lack of a joint-action mechanism of adaptive coefficient and integration
enhanced term. Consequently, the residual-based adaption coefficient zeroing
neural network (RACZNN) model with integration term is proposed in this paper
for solving the TDQM problem. The adaptive coefficient is proposed to improve
the performance of convergence and the integration term is embedded to ensure
the RACZNN model can maintain reliable robustness while perturbed by variant
measurement noises. Compared with the state-of-the-art models, the proposed
RACZNN model owns faster convergence and more reliable robustness. Then,
theorems are provided to prove the convergence of the RACZNN model. Finally,
corresponding quantitative numerical experiments are designed and performed in
this paper to verify the performance of the proposed RACZNN model.

    

### [[2112.01777] Attack-Centric Approach for Evaluating Transferability of Adversarial Samples in Machine Learning Models](http://arxiv.org/abs/2112.01777)


  Transferability of adversarial samples became a serious concern due to their
impact on the reliability of machine learning system deployments, as they find
their way into many critical applications. Knowing factors that influence
transferability of adversarial samples can assist experts to make informed
decisions on how to build robust and reliable machine learning systems. The
goal of this study is to provide insights on the mechanisms behind the
transferability of adversarial samples through an attack-centric approach. This
attack-centric perspective interprets how adversarial samples would transfer by
assessing the impact of machine learning attacks (that generated them) on a
given input dataset. To achieve this goal, we generated adversarial samples
using attacker models and transferred these samples to victim models. We
analyzed the behavior of adversarial samples on victim models and outlined four
factors that can influence the transferability of adversarial samples. Although
these factors are not necessarily exhaustive, they provide useful insights to
researchers and practitioners of machine learning systems.

    

### [[2112.01790] SSDL: Self-Supervised Dictionary Learning](http://arxiv.org/abs/2112.01790)


  The label-embedded dictionary learning (DL) algorithms generate influential
dictionaries by introducing discriminative information. However, there exists a
limitation: All the label-embedded DL methods rely on the labels due that this
way merely achieves ideal performances in supervised learning. While in
semi-supervised and unsupervised learning, it is no longer sufficient to be
effective. Inspired by the concept of self-supervised learning (e.g., setting
the pretext task to generate a universal model for the downstream task), we
propose a Self-Supervised Dictionary Learning (SSDL) framework to address this
challenge. Specifically, we first design a $p$-Laplacian Attention Hypergraph
Learning (pAHL) block as the pretext task to generate pseudo soft labels for
DL. Then, we adopt the pseudo labels to train a dictionary from a primary
label-embedded DL method. We evaluate our SSDL on two human activity
recognition datasets. The comparison results with other state-of-the-art
methods have demonstrated the efficiency of SSDL.

    

### [[2112.01796] The UniNAS framework: combining modules in arbitrarily complex configurations with argument trees](http://arxiv.org/abs/2112.01796)


  Designing code to be simplistic yet to offer choice is a tightrope walk.
Additional modules such as optimizers and data sets make a framework useful to
a broader audience, but the added complexity quickly becomes a problem.
Framework parameters may apply only to some modules but not others, be mutually
exclusive or depend on each other, often in unclear ways. Even so, many
frameworks are limited to a few specific use cases. This paper presents the
underlying concept of UniNAS, a framework designed to incorporate a variety of
Neural Architecture Search approaches. Since they differ in the number of
optimizers and networks, hyper-parameter optimization, network designs,
candidate operations, and more, a traditional approach can not solve the task.
Instead, every module defines its own hyper-parameters and a local tree
structure of module requirements. A configuration file specifies which modules
are used, their used parameters, and which other modules they use in turn This
concept of argument trees enables combining and reusing modules in complex
configurations while avoiding many problems mentioned above. Argument trees can
also be configured from a graphical user interface so that designing and
changing experiments becomes possible without writing a single line of code.
UniNAS is publicly available at this https URL


### [[2112.01797] Detection of Large Vessel Occlusions using Deep Learning by Deforming Vessel Tree Segmentations](http://arxiv.org/abs/2112.01797)


  Computed Tomography Angiography is a key modality providing insights into the
cerebrovascular vessel tree that are crucial for the diagnosis and treatment of
ischemic strokes, in particular in cases of large vessel occlusions (LVO).
Thus, the clinical workflow greatly benefits from an automated detection of
patients suffering from LVOs. This work uses convolutional neural networks for
case-level classification trained with elastic deformation of the vessel tree
segmentation masks to artificially augment training data. Using only masks as
the input to our model uniquely allows us to apply such deformations much more
aggressively than one could with conventional image volumes while retaining
sample realism.
The neural network classifies the presence of an LVO and the affected
hemisphere. In a 5-fold cross validated ablation study, we demonstrate that the
use of the suggested augmentation enables us to train robust models even from
few data sets. Training the EfficientNetB1 architecture on 100 data sets, the
proposed augmentation scheme was able to raise the ROC AUC to 0.85 from a
baseline value of 0.57 using no augmentation. The best performance was achieved
using a 3D-DenseNet yielding an AUC of 0.88. The augmentation had positive
impact in classification of the affected hemisphere as well, where the
3D-DenseNet reached an AUC of 0.93 on both sides.

    

### [[2112.01804] Computation of conditional expectations with guarantees](http://arxiv.org/abs/2112.01804)


  Theoretically, the conditional expectation of a square-integrable random
variable $Y$ given a $d$-dimensional random vector $X$ can be obtained by
minimizing the mean squared distance between $Y$ and $f(X)$ over all Borel
measurable functions $f \colon \mathbb{R}^d \to \mathbb{R}$. However, in many
applications this minimization problem cannot be solved exactly, and instead, a
numerical method that computes an approximate minimum over a suitable subfamily
of Borel functions has to be used. The quality of the result depends on the
adequacy of the subfamily and the performance of the numerical method. In this
paper, we derive an expected value representation of the minimal mean square
distance which in many applications can efficiently be approximated with a
standard Monte Carlo average. This enables us to provide guarantees for the
accuracy of any numerical approximation of a given conditional expectation. We
illustrate the method by assessing the quality of approximate conditional
expectations obtained by linear, polynomial as well as neural network
regression in different concrete examples.

    

### [[2112.01819] Chronological Causal Bandits](http://arxiv.org/abs/2112.01819)


  This paper studies an instance of the multi-armed bandit (MAB) problem,
specifically where several causal MABs operate chronologically in the same
dynamical system. Practically the reward distribution of each bandit is
governed by the same non-trivial dependence structure, which is a dynamic
causal model. Dynamic because we allow for each causal MAB to depend on the
preceding MAB and in doing so are able to transfer information between agents.
Our contribution, the Chronological Causal Bandit (CCB), is useful in discrete
decision-making settings where the causal effects are changing across time and
can be informed by earlier interventions in the same system. In this paper, we
present some early findings of the CCB as demonstrated on a toy problem.

    

### [[2112.01830] Table2Vec: Automated Universal Representation Learning to Encode All-round Data DNA for Benchmarkable and Explainable Enterprise Data Science](http://arxiv.org/abs/2112.01830)


  Enterprise data typically involves multiple heterogeneous data sources and
external data that respectively record business activities, transactions,
customer demographics, status, behaviors, interactions and communications with
the enterprise, and the consumption and feedback of its products, services,
production, marketing, operations, and management, etc. A critical challenge in
enterprise data science is to enable an effective whole-of-enterprise data
understanding and data-driven discovery and decision-making on all-round
enterprise DNA. We introduce a neural encoder Table2Vec for automated universal
representation learning of entities such as customers from all-round enterprise
DNA with automated data characteristics analysis and data quality augmentation.
The learned universal representations serve as representative and benchmarkable
enterprise data genomes and can be used for enterprise-wide and domain-specific
learning tasks. Table2Vec integrates automated universal representation
learning on low-quality enterprise data and downstream learning tasks. We
illustrate Table2Vec in characterizing all-round customer data DNA in an
enterprise on complex heterogeneous multi-relational big tables to build
universal customer vector representations. The learned universal representation
of each customer is all-round, representative and benchmarkable to support both
enterprise-wide and domain-specific learning goals and tasks in enterprise data
science. Table2Vec significantly outperforms the existing shallow, boosting and
deep learning methods typically used for enterprise analytics. We further
discuss the research opportunities, directions and applications of automated
universal enterprise representation and learning and the learned enterprise
data DNA for automated, all-purpose, whole-of-enterprise and ethical machine
learning and data science.

    

### [[2112.01836] Semantic Segmentation of Legal Documents via Rhetorical Roles](http://arxiv.org/abs/2112.01836)


  Legal documents are unstructured, use legal jargon, and have considerable
length, making it difficult to process automatically via conventional text
processing techniques. A legal document processing system would benefit
substantially if the documents could be semantically segmented into coherent
units of information. This paper proposes a Rhetorical Roles (RR) system for
segmenting a legal document into semantically coherent units: facts, arguments,
statute, issue, precedent, ruling, and ratio. With the help of legal experts,
we propose a set of 13 fine-grained rhetorical role labels and create a new
corpus of legal documents annotated with the proposed RR. We develop a system
for segmenting a document into rhetorical role units. In particular, we develop
a multitask learning-based deep learning model with document rhetorical role
label shift as an auxiliary task for segmenting a legal document. We experiment
extensively with various deep learning models for predicting rhetorical roles
in a document, and the proposed model shows superior performance over the
existing models. Further, we apply RR for predicting the judgment of legal
cases and show that the use of RR enhances the prediction compared to the
transformer-based models.

    

### [[2112.01838] Efficient Two-Stage Detection of Human-Object Interactions with a Novel Unary-Pairwise Transformer](http://arxiv.org/abs/2112.01838)


  Recent developments in transformer models for visual data have led to
significant improvements in recognition and detection tasks. In particular,
using learnable queries in place of region proposals has given rise to a new
class of one-stage detection models, spearheaded by the Detection Transformer
(DETR). Variations on this one-stage approach have since dominated human-object
interaction (HOI) detection. However, the success of such one-stage HOI
detectors can largely be attributed to the representation power of
transformers. We discovered that when equipped with the same transformer, their
two-stage counterparts can be more performant and memory-efficient, while
taking a fraction of the time to train. In this work, we propose the
Unary-Pairwise Transformer, a two-stage detector that exploits unary and
pairwise representations for HOIs. We observe that the unary and pairwise parts
of our transformer network specialise, with the former preferentially
increasing the scores of positive examples and the latter decreasing the scores
of negative examples. We evaluate our method on the HICO-DET and V-COCO
datasets, and significantly outperform state-of-the-art approaches. At
inference time, our model with ResNet50 approaches real-time performance on a
single GPU.

    

### [[2112.01839] Mind Your Clever Neighbours: Unsupervised Person Re-identification via Adaptive Clustering Relationship Modeling](http://arxiv.org/abs/2112.01839)


  Unsupervised person re-identification (Re-ID) attracts increasing attention
due to its potential to resolve the scalability problem of supervised Re-ID
models. Most existing unsupervised methods adopt an iterative clustering
mechanism, where the network was trained based on pseudo labels generated by
unsupervised clustering. However, clustering errors are inevitable. To generate
high-quality pseudo-labels and mitigate the impact of clustering errors, we
propose a novel clustering relationship modeling framework for unsupervised
person Re-ID. Specifically, before clustering, the relation between unlabeled
images is explored based on a graph correlation learning (GCL) module and the
refined features are then used for clustering to generate high-quality
pseudo-labels.Thus, GCL adaptively mines the relationship between samples in a
mini-batch to reduce the impact of abnormal clustering when training. To train
the network more effectively, we further propose a selective contrastive
learning (SCL) method with a selective memory bank update policy. Extensive
experiments demonstrate that our method shows much better results than most
state-of-the-art unsupervised methods on Market1501, DukeMTMC-reID and MSMT17
datasets. We will release the code for model reproduction.

    

### [[2112.01841] Reinforcement learning for options on target volatility funds](http://arxiv.org/abs/2112.01841)


  In this work we deal with the funding costs rising from hedging the risky
securities underlying a target volatility strategy (TVS), a portfolio of risky
assets and a risk-free one dynamically rebalanced in order to keep the realized
volatility of the portfolio on a certain level. The uncertainty in the TVS
risky portfolio composition along with the difference in hedging costs for each
component requires to solve a control problem to evaluate the option prices. We
derive an analytical solution of the problem in the Black and Scholes (BS)
scenario. Then we use Reinforcement Learning (RL) techniques to determine the
fund composition leading to the most conservative price under the local
volatility (LV) model, for which an a priori solution is not available. We show
how the performances of the RL agents are compatible with those obtained by
applying path-wise the BS analytical strategy to the TVS dynamics, which
therefore appears competitive also in the LV scenario.

    

### [[2112.01842] Automatic evaluation of scientific abstracts through natural language processing](http://arxiv.org/abs/2112.01842)


  This work presents a framework to classify and evaluate distinct research
abstract texts which are focused on the description of processes and their
applications. In this context, this paper proposes natural language processing
algorithms to classify, segment and evaluate the results of scientific work.
Initially, the proposed framework categorize the abstract texts into according
to the problems intended to be solved by employing a text classification
approach. Then, the abstract text is segmented into problem description,
methodology and results. Finally, the methodology of the abstract is ranked
based on the sentiment analysis of its results. The proposed framework allows
us to quickly rank the best methods to solve specific problems. To validate the
proposed framework, oil production anomaly abstracts were experimented and
achieved promising results.

    

### [[2112.01844] Combining Sub-Symbolic and Symbolic Methods for Explainability](http://arxiv.org/abs/2112.01844)


  Similarly to other connectionist models, Graph Neural Networks (GNNs) lack
transparency in their decision-making. A number of sub-symbolic approaches have
been developed to provide insights into the GNN decision making process. These
are first important steps on the way to explainability, but the generated
explanations are often hard to understand for users that are not AI experts. To
overcome this problem, we introduce a conceptual approach combining
sub-symbolic and symbolic methods for human-centric explanations, that
incorporate domain knowledge and causality. We furthermore introduce the notion
of fidelity as a metric for evaluating how close the explanation is to the
GNN's internal decision making process. The evaluation with a chemical dataset
and ontology shows the explanatory value and reliability of our method.

    

### [[2112.01849] Cross-modal Knowledge Distillation for Vision-to-Sensor Action Recognition](http://arxiv.org/abs/2112.01849)


  Human activity recognition (HAR) based on multi-modal approach has been
recently shown to improve the accuracy performance of HAR. However, restricted
computational resources associated with wearable devices, i.e., smartwatch,
failed to directly support such advanced methods. To tackle this issue, this
study introduces an end-to-end Vision-to-Sensor Knowledge Distillation (VSKD)
framework. In this VSKD framework, only time-series data, i.e., accelerometer
data, is needed from wearable devices during the testing phase. Therefore, this
framework will not only reduce the computational demands on edge devices, but
also produce a learning model that closely matches the performance of the
computational expensive multi-modal approach. In order to retain the local
temporal relationship and facilitate visual deep learning models, we first
convert time-series data to two-dimensional images by applying the Gramian
Angular Field ( GAF) based encoding method. We adopted ResNet18 and multi-scale
TRN with BN-Inception as teacher and student network in this study,
respectively. A novel loss function, named Distance and Angle-wised Semantic
Knowledge loss (DASK), is proposed to mitigate the modality variations between
the vision and the sensor domain. Extensive experimental results on UTD-MHAD,
MMAct, and Berkeley-MHAD datasets demonstrate the effectiveness and
competitiveness of the proposed VSKD model which can deployed on wearable
sensors.

    

### [[2112.01853] Episodic Policy Gradient Training](http://arxiv.org/abs/2112.01853)


  We introduce a novel training procedure for policy gradient methods wherein
episodic memory is used to optimize the hyperparameters of reinforcement
learning algorithms on-the-fly. Unlike other hyperparameter searches, we
formulate hyperparameter scheduling as a standard Markov Decision Process and
use episodic memory to store the outcome of used hyperparameters and their
training contexts. At any policy update step, the policy learner refers to the
stored experiences, and adaptively reconfigures its learning algorithm with the
new hyperparameters determined by the memory. This mechanism, dubbed as
Episodic Policy Gradient Training (EPGT), enables an episodic learning process,
and jointly learns the policy and the learning algorithm's hyperparameters
within a single run. Experimental results on both continuous and discrete
environments demonstrate the advantage of using the proposed method in boosting
the performance of various policy gradient algorithms.

    

### [[2112.01863] Discovery of Crime Event Sequences with Constricted Spatio-Temporal Sequential Patterns](http://arxiv.org/abs/2112.01863)


  In this article, we introduce a novel type of spatio-temporal sequential
patterns called Constricted Spatio-Temporal Sequential (CSTS) patterns and
thoroughly analyze their properties. We demonstrate that the set of CSTS
patterns is a concise representation of all spatio-temporal sequential patterns
that can be discovered in a given dataset. To measure significance of the
discovered CSTS patterns we adapt the participation index measure. We also
provide CSTS-Miner: an algorithm that discovers all participation index strong
CSTS patterns in event data. We experimentally evaluate the proposed algorithms
using two crime-related datasets: Pittsburgh Police Incident Blotter Dataset
and Boston Crime Incident Reports Dataset. In the experiments, the CSTS-Miner
algorithm is compared with the other four state-of-the-art algorithms:
STS-Miner, CSTPM, STBFM and CST-SPMiner. As the results of experiments suggest,
the proposed algorithm discovers much fewer patterns than the other selected
algorithms. Finally, we provide the examples of interesting crime-related
patterns discovered by the proposed CSTS-Miner algorithm.

    

### [[2112.01871] Active Inference in Robotics and Artificial Agents: Survey and Challenges](http://arxiv.org/abs/2112.01871)


  Active inference is a mathematical framework which originated in
computational neuroscience as a theory of how the brain implements action,
perception and learning. Recently, it has been shown to be a promising approach
to the problems of state-estimation and control under uncertainty, as well as a
foundation for the construction of goal-driven behaviours in robotics and
artificial agents in general. Here, we review the state-of-the-art theory and
implementations of active inference for state-estimation, control, planning and
learning; describing current achievements with a particular focus on robotics.
We showcase relevant experiments that illustrate its potential in terms of
adaptation, generalization and robustness. Furthermore, we connect this
approach with other frameworks and discuss its expected benefits and
challenges: a unified framework with functional biological plausibility using
variational Bayesian inference.

    

### [[2112.01873] Image-to-image Translation as a Unique Source of Knowledge](http://arxiv.org/abs/2112.01873)


  Image-to-image (I2I) translation is an established way of translating data
from one domain to another but the usability of the translated images in the
target domain when working with such dissimilar domains as the SAR/optical
satellite imagery ones and how much of the origin domain is translated to the
target domain is still not clear enough. This article address this by
performing translations of labelled datasets from the optical domain to the SAR
domain with different I2I algorithms from the state-of-the-art, learning from
transferred features in the destination domain and evaluating later how much
from the original dataset was transferred. Added to this, stacking is proposed
as a way of combining the knowledge learned from the different I2I translations
and evaluated against single models.

    

### [[2112.01875] A Flexible HLS Hoeffding Tree Implementation for Runtime Learning on FPGA](http://arxiv.org/abs/2112.01875)


  Decision trees are often preferred when implementing Machine Learning in
embedded systems for their simplicity and scalability. Hoeffding Trees are a
type of Decision Trees that take advantage of the Hoeffding Bound to allow them
to learn patterns in data without having to continuously store the data samples
for future reprocessing. This makes them especially suitable for deployment on
embedded devices. In this work we highlight the features of an HLS
implementation of the Hoeffding Tree. The implementation parameters include the
feature size of the samples (D), the number of output classes (K), and the
maximum number of nodes to which the tree is allowed to grow (Nd). We target a
Xilinx MPSoC ZCU102, and evaluate: the design's resource requirements and clock
frequency for different numbers of classes and feature size, the execution time
on several synthetic datasets of varying sample sizes (N), number of output
classes and the execution time and accuracy for two datasets from UCI. For a
problem size of D3, K5, and N40000, a single decision tree operating at 103MHz
is capable of 8.3x faster inference than the 1.2GHz ARM Cortex-A53 core.
Compared to a reference implementation of the Hoeffding tree, we achieve
comparable classification accuracy for the UCI datasets.

    

### [[2112.01878] Fast $L^2$ optimal mass transport via reduced basis methods for the Monge-Amp$\grave{\rm e}$re equation](http://arxiv.org/abs/2112.01878)


  Repeatedly solving the parameterized optimal mass transport (pOMT) problem is
a frequent task in applications such as image registration and adaptive grid
generation. It is thus critical to develop a highly efficient reduced solver
that is equally accurate as the full order model. In this paper, we propose
such a machine learning-like method for pOMT by adapting a new reduced basis
(RB) technique specifically designed for nonlinear equations, the reduced
residual reduced over-collocation (R2-ROC) approach, to the parameterized
Monge-Amp$\grave{\rm e}$re equation. It builds on top of a narrow-stencil
finite different method (FDM), a so-called truth solver, which we propose in
this paper for the Monge-Amp$\grave{\rm e}$re equation with a transport
boundary. Together with the R2-ROC approach, it allows us to handle the strong
and unique nonlinearity pertaining to the Monge-Amp$\grave{\rm e}$re equation
achieving online efficiency without resorting to any direct approximation of
the nonlinearity. Several challenging numerical tests demonstrate the accuracy
and high efficiency of our method for solving the Monge-Amp$\grave{\rm e}$re
equation with various parametric boundary conditions.

    

### [[2112.01879] Reinforcement Learning-Based Automatic Berthing System](http://arxiv.org/abs/2112.01879)


  Previous studies on automatic berthing systems based on artificial neural
network (ANN) showed great berthing performance by training the ANN with ship
berthing data as training data. However, because the ANN requires a large
amount of training data to yield robust performance, the ANN-based automatic
berthing system is somewhat limited due to the difficulty in obtaining the
berthing data. In this study, to overcome this difficulty, the automatic
berthing system based on one of the reinforcement learning (RL) algorithms,
proximal policy optimization (PPO), is proposed because the RL algorithms can
learn an optimal control policy through trial-and-error by interacting with a
given environment and does not require any pre-obtained training data, where
the control policy in the proposed PPO-based automatic berthing system controls
revolutions per second (RPS) and rudder angle of a ship. Finally, it is shown
that the proposed PPO-based automatic berthing system eliminates the need for
obtaining the training dataset and shows great potential for the actual
berthing application.

    

### [[2112.01880] Bayes in Wonderland! Predictive supervised classification inference hits unpredictability](http://arxiv.org/abs/2112.01880)


  The marginal Bayesian predictive classifiers (mBpc) as opposed to the
simultaneous Bayesian predictive classifiers (sBpc), handle each data
separately and hence tacitly assumes the independence of the observations.
However, due to saturation in learning of generative model parameters, the
adverse effect of this false assumption on the accuracy of mBpc tends to wear
out in face of increasing amount of training data; guaranteeing the convergence
of these two classifiers under de Finetti type of exchangeability. This result
however, is far from trivial for the sequences generated under Partition
exchangeability (PE), where even umpteen amount of training data is not ruling
out the possibility of an unobserved outcome (Wonderland!). We provide a
computational scheme that allows the generation of the sequences under PE.
Based on that, with controlled increase of the training data, we show the
convergence of the sBpc and mBpc. This underlies the use of simpler yet
computationally more efficient marginal classifiers instead of simultaneous. We
also provide a parameter estimation of the generative model giving rise to the
partition exchangeable sequence as well as a testing paradigm for the equality
of this parameter across different samples. The package for Bayesian predictive
supervised classifications, parameter estimation and hypothesis testing of the
Ewens Sampling formula generative model is deposited on CRAN as PEkit package
and free available from this https URL.

    

### [[2112.01896] Estimating the Value-at-Risk by Temporal VAE](http://arxiv.org/abs/2112.01896)


  Estimation of the value-at-risk (VaR) of a large portfolio of assets is an
important task for financial institutions. As the joint log-returns of asset
prices can often be projected to a latent space of a much smaller dimension,
the use of a variational autoencoder (VAE) for estimating the VaR is a natural
suggestion. To ensure the bottleneck structure of autoencoders when learning
sequential data, we use a temporal VAE (TempVAE) that avoids an auto-regressive
structure for the observation variables. However, the low signal- to-noise
ratio of financial data in combination with the auto-pruning property of a VAE
typically makes the use of a VAE prone to posterior collapse. Therefore, we
propose to use annealing of the regularization to mitigate this effect. As a
result, the auto-pruning of the TempVAE works properly which also results in
excellent estimation results for the VaR that beats classical GARCH-type and
historical simulation approaches when applied to real data.

    

### [[2112.01898] Linear algebra with transformers](http://arxiv.org/abs/2112.01898)


  Most applications of transformers to mathematics, from integration to theorem
proving, focus on symbolic computation. In this paper, we show that
transformers can be trained to perform numerical calculations with high
accuracy. We consider problems of linear algebra: matrix transposition,
addition, multiplication, eigenvalues and vectors, singular value
decomposition, and inversion. Training small transformers (up to six layers)
over datasets of random matrices, we achieve high accuracies (over 90%) on all
problems. We also show that trained models can generalize out of their training
distribution, and that out-of-domain accuracy can be greatly improved by
working from more diverse datasets (in particular, by training from matrices
with non-independent and identically distributed coefficients). Finally, we
show that few-shot learning can be leveraged to re-train models to solve larger
problems.

    

### [[2112.01903] Hybrid Digital Twin for process industry using Apros simulation environment](http://arxiv.org/abs/2112.01903)


  Making an updated and as-built model plays an important role in the
life-cycle of a process plant. In particular, Digital Twin models must be
precise to guarantee the efficiency and reliability of the systems. Data-driven
models can simulate the latest behavior of the sub-systems by considering
uncertainties and life-cycle related changes. This paper presents a
step-by-step concept for hybrid Digital Twin models of process plants using an
early implemented prototype as an example. It will detail the steps for
updating the first-principles model and Digital Twin of a brownfield process
system using data-driven models of the process equipment. The challenges for
generation of an as-built hybrid Digital Twin will also be discussed. With the
help of process history data to teach Machine Learning models, the implemented
Digital Twin can be continually improved over time and this work in progress
can be further optimized.

    

### [[2112.01907] Near-optimal estimation of smooth transport maps with kernel sums-of-squares](http://arxiv.org/abs/2112.01907)


  It was recently shown that under smoothness conditions, the squared
Wasserstein distance between two distributions could be efficiently computed
with appealing statistical error upper bounds. However, rather than the
distance itself, the object of interest for applications such as generative
modeling is the underlying optimal transport map. Hence, computational and
statistical guarantees need to be obtained for the estimated maps themselves.
In this paper, we propose the first tractable algorithm for which the
statistical $L^2$ error on the maps nearly matches the existing minimax
lower-bounds for smooth map estimation. Our method is based on solving the
semi-dual formulation of optimal transport with an infinite-dimensional
sum-of-squares reformulation, and leads to an algorithm which has
dimension-free polynomial rates in the number of samples, with potentially
exponentially dimension-dependent constants.

    

### [[2112.01908] Prediction of Household-level Heat-Consumption using PSO enhanced SVR Model](http://arxiv.org/abs/2112.01908)


  In combating climate change, an effective demand-based energy supply
operation of the district energy system (DES) for heating or cooling is
indispensable. As a consequence, an accurate forecast of heat consumption on
the consumer side poses an important first step towards an optimal energy
supply. However, due to the non-linearity and non-stationarity of heat
consumption data, the prediction of the thermal energy demand of DES remains
challenging. In this work, we propose a forecasting framework for thermal
energy consumption within a district heating system (DHS) based on kernel
Support Vector Regression (kSVR) using real-world smart meter data. Particle
Swarm Optimization (PSO) is employed to find the optimal hyper-parameter for
the kSVR model which leads to the superiority of the proposed methods when
compared to a state-of-the-art ARIMA model. The average MAPE is reduced to
2.07% and 2.64% for the individual meter-specific forecasting and for
forecasting of societal consumption, respectively.

    

### [[2112.01917] A Structured Dictionary Perspective on Implicit Neural Representations](http://arxiv.org/abs/2112.01917)


  Propelled by new designs that permit to circumvent the spectral bias,
implicit neural representations (INRs) have recently emerged as a promising
alternative to classical discretized representations of signals. Nevertheless,
despite their practical success, we still lack a proper theoretical
characterization of how INRs represent signals. In this work, we aim to fill
this gap, and we propose a novel unified perspective to theoretically analyse
INRs. Leveraging results from harmonic analysis and deep learning theory, we
show that most INR families are analogous to structured signal dictionaries
whose atoms are integer harmonics of the set of initial mapping frequencies.
This structure allows INRs to express signals with an exponentially increasing
frequency support using a number of parameters that only grows linearly with
depth. Afterwards, we explore the inductive bias of INRs exploiting recent
results about the empirical neural tangent kernel (NTK). Specifically, we show
that the eigenfunctions of the NTK can be seen as dictionary atoms whose inner
product with the target signal determines the final performance of their
reconstruction. In this regard, we reveal that meta-learning the initialization
has a reshaping effect of the NTK analogous to dictionary learning, building
dictionary atoms as a combination of the examples seen during meta-training.
Our results permit to design and tune novel INR architectures, but can also be
of interest for the wider deep learning theory community.

    

### [[2112.01918] Heuristic Search Planning with Deep Neural Networks using Imitation, Attention and Curriculum Learning](http://arxiv.org/abs/2112.01918)


  Learning a well-informed heuristic function for hard task planning domains is
an elusive problem. Although there are known neural network architectures to
represent such heuristic knowledge, it is not obvious what concrete information
is learned and whether techniques aimed at understanding the structure help in
improving the quality of the heuristics. This paper presents a network model to
learn a heuristic capable of relating distant parts of the state space via
optimal plan imitation using the attention mechanism, which drastically
improves the learning of a good heuristic function. To counter the limitation
of the method in the creation of problems of increasing difficulty, we
demonstrate the use of curriculum learning, where newly solved problem
instances are added to the training set, which, in turn, helps to solve
problems of higher complexities and far exceeds the performances of all
existing baselines including classical planning heuristics. We demonstrate its
effectiveness for grid-type PDDL domains.

    

### [[2112.01921] In situ process quality monitoring and defect detection for direct metal laser melting](http://arxiv.org/abs/2112.01921)


  Quality control and quality assurance are challenges in Direct Metal Laser
Melting (DMLM). Intermittent machine diagnostics and downstream part
inspections catch problems after undue cost has been incurred processing
defective parts. In this paper we demonstrate two methodologies for in-process
fault detection and part quality prediction that can be readily deployed on
existing commercial DMLM systems with minimal hardware modification. Novel
features were derived from the time series of common photodiode sensors along
with standard machine control signals. A Bayesian approach attributes
measurements to one of multiple process states and a least squares regression
model predicts severity of certain material defects.

    

### [[2112.01922] MetaQA: Combining Expert Agents for Multi-Skill Question Answering](http://arxiv.org/abs/2112.01922)


  The recent explosion of question answering (QA) datasets and models has
increased the interest in the generalization of models across multiple domains
and formats by either training models on multiple datasets or by combining
multiple models. We argue that despite the promising results of multi-dataset
models, some domains or QA formats may require specific architectures, and thus
the adaptability of these models might be limited. In addition, current
approaches for combining models disregard cues such as question-answer
compatibility. In this work, we propose to combine expert agents with a novel,
flexible, and training-efficient architecture that considers questions, answer
predictions, and answer-prediction confidence scores to select the best answer
among a list of answer candidates. Through quantitative and qualitative
experiments we show that our model i) creates a collaboration between agents
that outperforms previous multi-agent and multi-dataset approaches in both
in-domain and out-of-domain scenarios, ii) is extremely data-efficient to
train, and iii) can be adapted to any QA format.

    

### [[2112.01925] Generative Adversarial Networks for Synthetic Data Generation: A Comparative Study](http://arxiv.org/abs/2112.01925)


  Generative Adversarial Networks (GANs) are gaining increasing attention as a
means for synthesising data. So far much of this work has been applied to use
cases outside of the data confidentiality domain with a common application
being the production of artificial images. Here we consider the potential
application of GANs for the purpose of generating synthetic census microdata.
We employ a battery of utility metrics and a disclosure risk metric (the
Targeted Correct Attribution Probability) to compare the data produced by
tabular GANs with those produced using orthodox data synthesis methods.

    

### [[2112.01938] Shapes of Emotions: Multimodal Emotion Recognition in Conversations via Emotion Shifts](http://arxiv.org/abs/2112.01938)


  Emotion Recognition in Conversations (ERC) is an important and active
research problem. Recent work has shown the benefits of using multiple
modalities (e.g., text, audio, and video) for the ERC task. In a conversation,
participants tend to maintain a particular emotional state unless some external
stimuli evokes a change. There is a continuous ebb and flow of emotions in a
conversation. Inspired by this observation, we propose a multimodal ERC model
and augment it with an emotion-shift component. The proposed emotion-shift
component is modular and can be added to any existing multimodal ERC model
(with a few modifications), to improve emotion recognition. We experiment with
different variants of the model, and results show that the inclusion of emotion
shift signal helps the model to outperform existing multimodal models for ERC
and hence showing the state-of-the-art performance on MOSEI and IEMOCAP
datasets.

    

### [[2112.01939] Fast Projected Newton-like Method for Precision Matrix Estimation with Nonnegative Partial Correlations](http://arxiv.org/abs/2112.01939)


  We study the problem of estimating precision matrices in multivariate
Gaussian distributions where all partial correlations are nonnegative, also
known as multivariate totally positive of order two ($\mathrm{MTP}_2$). Such
models have received significant attention in recent years, primarily due to
interesting properties, e.g., the maximum likelihood estimator exists with as
few as two observations regardless of the underlying dimension. We formulate
this problem as a weighted $\ell_1$-norm regularized Gaussian maximum
likelihood estimation under $\mathrm{MTP}_2$ constraints. On this direction, we
propose a novel projected Newton-like algorithm that incorporates a
well-designed approximate Newton direction, which results in our algorithm
having the same orders of computation and memory costs as those of first-order
methods. We prove that the proposed projected Newton-like algorithm converges
to the minimizer of the problem. We further show, both theoretically and
experimentally, that the minimizer of our formulation using the weighted
$\ell_1$-norm is able to recover the support of the underlying precision matrix
correctly without requiring the incoherence condition present in $\ell_1$-norm
based methods. Experiments involving synthetic and real-world data demonstrate
that our proposed algorithm is significantly more efficient, from a
computational time perspective, than the state-of-the-art methods. Finally, we
apply our method in financial time-series data, which are well-known for
displaying positive dependencies, where we observe a significant performance in
terms of modularity value on the learned financial networks.

    

### [[2112.01948] Boosting Unsupervised Domain Adaptation with Soft Pseudo-label and Curriculum Learning](http://arxiv.org/abs/2112.01948)


  By leveraging data from a fully labeled source domain, unsupervised domain
adaptation (UDA) improves classification performance on an unlabeled target
domain through explicit discrepancy minimization of data distribution or
adversarial learning. As an enhancement, category alignment is involved during
adaptation to reinforce target feature discrimination by utilizing model
prediction. However, there remain unexplored problems about pseudo-label
inaccuracy incurred by wrong category predictions on target domain, and
distribution deviation caused by overfitting on source domain. In this paper,
we propose a model-agnostic two-stage learning framework, which greatly reduces
flawed model predictions using soft pseudo-label strategy and avoids
overfitting on source domain with a curriculum learning strategy.
Theoretically, it successfully decreases the combined risk in the upper bound
of expected error on the target domain. At the first stage, we train a model
with distribution alignment-based UDA method to obtain soft semantic label on
target domain with rather high confidence. To avoid overfitting on source
domain, at the second stage, we propose a curriculum learning strategy to
adaptively control the weighting between losses from the two domains so that
the focus of the training stage is gradually shifted from source distribution
to target distribution with prediction confidence boosted on the target domain.
Extensive experiments on two well-known benchmark datasets validate the
universal effectiveness of our proposed framework on promoting the performance
of the top-ranked UDA algorithms and demonstrate its consistent superior
performance.

    

### [[2112.01955] You Can't See the Forest for Its Trees: Assessing Deep Neural Network Testing via NeuraL Coverage](http://arxiv.org/abs/2112.01955)


  This paper summarizes eight design requirements for DNN testing criteria,
taking into account distribution properties and practical concerns. We then
propose a new criterion, NLC, that satisfies all of these design requirements.
NLC treats a single DNN layer as the basic computational unit (rather than a
single neuron) and captures four critical features of neuron output
distributions. Thus, NLC is denoted as NeuraL Coverage, which more accurately
describes how neural networks comprehend inputs via approximated distributions
rather than neurons. We demonstrate that NLC is significantly correlated with
the diversity of a test suite across a number of tasks (classification and
generation) and data formats (image and text). Its capacity to discover DNN
prediction errors is promising. Test input mutation guided by NLC result in a
greater quality and diversity of exposed erroneous behaviors.

    

### [[2112.01956] Enhancing Deep Neural Networks Testing by Traversing Data Manifold](http://arxiv.org/abs/2112.01956)


  We develop DEEPTRAVERSAL, a feedback-driven framework to test DNNs.
DEEPTRAVERSAL first launches an offline phase to map media data of various
forms to manifolds. Then, in its online testing phase, DEEPTRAVERSAL traverses
the prepared manifold space to maximize DNN coverage criteria and trigger
prediction errors. In our evaluation, DNNs executing various tasks (e.g.,
classification, self-driving, machine translation) and media data of different
types (image, audio, text) were used. DEEPTRAVERSAL exhibits better performance
than prior methods with respect to popular DNN coverage criteria and it can
discover a larger number and higher quality of error-triggering inputs. The
tested DNN models, after being repaired with findings of DEEPTRAVERSAL, achieve
better accuracy

    

### [[2112.01971] Dynamic fracture of a bicontinuously nanostructured copolymer: A deep learning analysis of big-data-generating experiment](http://arxiv.org/abs/2112.01971)


  Here, we report the dynamic fracture toughness as well as the cohesive
parameters of a bicontinuously nanostructured copolymer, polyurea, under an
extremely high crack-tip loading rate, from a deep-learning analysis of a
dynamic big-data-generating experiment. We first invented a novel Dynamic
Line-Image Shearing Interferometer (DL-ISI), which can generate the
displacement-gradient - time profiles along a line on a sample's back surface
projectively covering the crack initiation and growth process in a single plate
impact experiment. Then, we proposed a convolutional neural network (CNN) based
deep-learning framework that can inversely determine the accurate cohesive
parameters from DL-ISI fringe images. Plate-impact experiments on a polyurea
sample with a mid-plane crack have been performed, and the generated DL-ISI
fringe image has been inpainted by a Conditional Generative Adversarial
Networks (cGAN). For the first time, the dynamic cohesive parameters of
polyurea have been successfully obtained by the pre-trained CNN architecture
with the computational dataset, which is consistent with the correlation method
and the linear fracture mechanics estimation. Apparent dynamic toughening is
found in polyurea, where the cohesive strength is found to be nearly three
times higher than the spall strength under the symmetric impact with the same
impact speed. These experimental results fill the gap in the current
understanding of copolymer's cooperative-failure strength under extreme local
loading conditions near the crack tip. This experiment also demonstrates the
advantages of big-data-generating experiments, which combine innovative
high-throughput experimental techniques with state-of-the-art machine learning
algorithms.

    

### [[2112.01988] ROCA: Robust CAD Model Retrieval and Alignment from a Single Image](http://arxiv.org/abs/2112.01988)


  We present ROCA, a novel end-to-end approach that retrieves and aligns 3D CAD
models from a shape database to a single input image. This enables 3D
perception of an observed scene from a 2D RGB observation, characterized as a
lightweight, compact, clean CAD representation. Core to our approach is our
differentiable alignment optimization based on dense 2D-3D object
correspondences and Procrustes alignment. ROCA can thus provide a robust CAD
alignment while simultaneously informing CAD retrieval by leveraging the 2D-3D
correspondences to learn geometrically similar CAD models. Experiments on
challenging, real-world imagery from ScanNet show that ROCA significantly
improves on state of the art, from 9.5% to 17.6% in retrieval-aware CAD
alignment accuracy.

    

### [[2112.01989] Survey on English Entity Linking on Wikidata](http://arxiv.org/abs/2112.01989)


  Wikidata is a frequently updated, community-driven, and multilingual
knowledge graph. Hence, Wikidata is an attractive basis for Entity Linking,
which is evident by the recent increase in published papers. This survey
focuses on four subjects: (1) Which Wikidata Entity Linking datasets exist, how
widely used are they and how are they constructed? (2) Do the characteristics
of Wikidata matter for the design of Entity Linking datasets and if so, how?
(3) How do current Entity Linking approaches exploit the specific
characteristics of Wikidata? (4) Which Wikidata characteristics are unexploited
by existing Entity Linking approaches? This survey reveals that current
Wikidata-specific Entity Linking datasets do not differ in their annotation
scheme from schemes for other knowledge graphs like DBpedia. Thus, the
potential for multilingual and time-dependent datasets, naturally suited for
Wikidata, is not lifted. Furthermore, we show that most Entity Linking
approaches use Wikidata in the same way as any other knowledge graph missing
the chance to leverage Wikidata-specific characteristics to increase quality.
Almost all approaches employ specific properties like labels and sometimes
descriptions but ignore characteristics such as the hyper-relational structure.
Hence, there is still room for improvement, for example, by including
hyper-relational graph embeddings or type information. Many approaches also
include information from Wikipedia, which is easily combinable with Wikidata
and provides valuable textual information, which Wikidata lacks.

    

### [[2112.01998] Application of Machine Learning in understanding plant virus pathogenesis: Trends and perspectives on emergence, diagnosis, host-virus interplay and management](http://arxiv.org/abs/2112.01998)


  Inclusion of high throughput technologies in the field of biology has
generated massive amounts of biological data in the recent years. Now,
transforming these huge volumes of data into knowledge is the primary challenge
in computational biology. The traditional methods of data analysis have failed
to carry out the task. Hence, researchers are turning to machine learning based
approaches for the analysis of high-dimensional big data. In machine learning,
once a model is trained with a training dataset, it can be applied on a testing
dataset which is independent. In current times, deep learning algorithms
further promote the application of machine learning in several field of biology
including plant virology. Considering a significant progress in the application
of machine learning in understanding plant virology, this review highlights an
introductory note on machine learning and comprehensively discusses the trends
and prospects of machine learning in diagnosis of viral diseases, understanding
host-virus interplay and emergence of plant viruses.

    

### [[2112.02000] A Survey on Concept Drift in Process Mining](http://arxiv.org/abs/2112.02000)


  Concept drift in process mining (PM) is a challenge as classical methods
assume processes are in a steady-state, i.e., events share the same process
version. We conducted a systematic literature review on the intersection of
these areas, and thus, we review concept drift in process mining and bring
forward a taxonomy of existing techniques for drift detection and online
process mining for evolving environments. Existing works depict that (i) PM
still primarily focuses on offline analysis, and (ii) the assessment of concept
drift techniques in processes is cumbersome due to the lack of common
evaluation protocol, datasets, and metrics.

    

### [[2112.02002] Modelling and optimization of nanovector synthesis for applications in drug delivery systems](http://arxiv.org/abs/2112.02002)


  Nanovectors (NVs), based on nanostructured matter such as nanoparticles
(NPs), have proven to perform as excellent drug delivery systems. However, due
to the great variety of potential NVs, including NPs materials and their
functionalization, in addition to the plethora of molecules that could
transport, this fields presents a great challenge in terms of resources to find
NVs with the most optimal physicochemical properties such as particle size and
drug loading, where most of efforts rely on trial and error experimentation. In
this regard, Artificial intelligence (AI) and metaheuristic algorithms offer
efficient of the state-of-the-art modelling and optimization, respectively.
This review focuses, through a systematic search, on the use of artificial
intelligence and metaheuristic algorithms for nanoparticle synthesis in drug
delivery systems. The main findings are: neural networks are better at
modelling NVs properties than linear regression algorithms and response surface
methodology, there is a very limited number of studies comparing AI or
metaheuristic algorithm, and there is no information regarding the
appropriateness of calculations of the sample size. Based on these findings,
multilayer perceptron artificial neural network and adaptive neuro fuzzy
inference system were tested for their modelling performance with a NV dataset;
finding the latter the better algorithm. For metaheuristic algorithms,
benchmark functions were optimized with cuckoo search, firefly algorithm,
genetic algorithm and symbiotic organism search; finding cuckoo search and
symbiotic organism search with the best performance. Finally, methods to
estimate appropriate sample size for AI algorithms are discussed.

    

### [[2112.02006] User-click Modelling for Predicting Purchase Intent](http://arxiv.org/abs/2112.02006)


  This thesis contributes a structured inquiry into the open actuarial
mathematics problem of modelling user behaviour using machine learning methods,
in order to predict purchase intent of non-life insurance products. It is
valuable for a company to understand user interactions with their website as it
provides rich and individualized insight into consumer behaviour. Most of
existing research in user behaviour modelling aims to explain or predict clicks
on a search engine result page or to estimate click-through rate in sponsored
search. These models are based on concepts about users' examination patterns of
a web page and the web page's representation of items. Investigating the
problem of modelling user behaviour to predict purchase intent on a business
website, we observe that a user's intention yields high dependency on how the
user navigates the website in terms of how many different web pages the user
visited, what kind of web pages the user interacted with, and how much time the
user spent on each web page. Inspired by these findings, we propose two
different ways of representing features of a user session leading to two models
for user click-based purchase prediction: one based on a Feed Forward Neural
Network, and another based on a Recurrent Neural Network. We examine the
discriminativeness of user-clicks for predicting purchase intent by comparing
the above two models with a model using demographic features of the user. Our
experimental results show that our click-based models significantly outperform
the demographic model, in terms of standard classification evaluation metrics,
and that a model based on a sequential representation of user clicks yields
slightly greater performance than a model based on feature engineering of
clicks.

    

### [[2112.02012] Practitioner-Centric Approach for Early Incident Detection Using Crowdsourced Data for Emergency Services](http://arxiv.org/abs/2112.02012)


  Emergency response is highly dependent on the time of incident reporting.
Unfortunately, the traditional approach to receiving incident reports (e.g.,
calling 911 in the USA) has time delays. Crowdsourcing platforms such as Waze
provide an opportunity for early identification of incidents. However,
detecting incidents from crowdsourced data streams is difficult due to the
challenges of noise and uncertainty associated with such data. Further, simply
optimizing over detection accuracy can compromise spatial-temporal localization
of the inference, thereby making such approaches infeasible for real-world
deployment. This paper presents a novel problem formulation and solution
approach for practitioner-centered incident detection using crowdsourced data
by using emergency response management as a case-study. The proposed approach
CROME (Crowdsourced Multi-objective Event Detection) quantifies the
relationship between the performance metrics of incident classification (e.g.,
F1 score) and the requirements of model practitioners (e.g., 1 km. radius for
incident detection). First, we show how crowdsourced reports, ground-truth
historical data, and other relevant determinants such as traffic and weather
can be used together in a Convolutional Neural Network (CNN) architecture for
early detection of emergency incidents. Then, we use a Pareto
optimization-based approach to optimize the output of the CNN in tandem with
practitioner-centric parameters to balance detection accuracy and
spatial-temporal localization. Finally, we demonstrate the applicability of
this approach using crowdsourced data from Waze and traffic accident reports
from Nashville, TN, USA. Our experiments demonstrate that the proposed approach
outperforms existing approaches in incident detection while simultaneously
optimizing the needs for real-world deployment and usability.

    

### [[2112.02027] Divergent representations of ethological visual inputs emerge from supervised, unsupervised, and reinforcement learning](http://arxiv.org/abs/2112.02027)


  Artificial neural systems trained using reinforcement, supervised, and
unsupervised learning all acquire internal representations of high dimensional
input. To what extent these representations depend on the different learning
objectives is largely unknown. Here we compare the representations learned by
eight different convolutional neural networks, each with identical ResNet
architectures and trained on the same family of egocentric images, but embedded
within different learning systems. Specifically, the representations are
trained to guide action in a compound reinforcement learning task; to predict
one or a combination of three task-related targets with supervision; or using
one of three different unsupervised objectives. Using representational
similarity analysis, we find that the network trained with reinforcement
learning differs most from the other networks. Through further analysis using
metrics inspired by the neuroscience literature, we find that the model trained
with reinforcement learning has a sparse and high-dimensional representation
wherein individual images are represented with very different patterns of
neural activity. Further analysis suggests these representations may arise in
order to guide long-term behavior and goal-seeking in the RL agent. Our results
provide insights into how the properties of neural representations are
influenced by objective functions and can inform transfer learning approaches.

    

### [[2112.02039] Bridging the Gap: Point Clouds for Merging Neurons in Connectomics](http://arxiv.org/abs/2112.02039)


  In the field of Connectomics, a primary problem is that of 3D neuron
segmentation. Although Deep Learning based methods have achieved remarkable
accuracy, errors still exist, especially in regions with image defects. One
common type of defect is that of consecutive missing image sections. Here data
is lost along some axis, and the resulting neuron segmentations are split
across the gap. To address this problem, we propose a novel method based on
point cloud representations of neurons. We formulate this as a classification
problem and train CurveNet, a state-of-the-art point cloud classification
model, to identify which neurons should be merged. We show that our method not
only performs strongly but scales reasonably to gaps well beyond what other
methods have attempted to address. Additionally, our point cloud
representations are highly efficient in terms of data, maintaining high
performance with an amount of data that would be unfeasible for other methods.
We believe that this is an indicator of the viability of using point clouds
representations for other proofreading tasks.

    

### [[2112.02043] Multilingual training for Software Engineering](http://arxiv.org/abs/2112.02043)


  Well-trained machine-learning models, which leverage large amounts of
open-source software data, have now become an interesting approach to
automating many software engineering tasks. Several SE tasks have all been
subject to this approach, with performance gradually improving over the past
several years with better models and training methods. More, and more diverse,
clean, labeled data is better for training; but constructing good-quality
datasets is time-consuming and challenging. Ways of augmenting the volume and
diversity of clean, labeled data generally have wide applicability. For some
languages (e.g., Ruby) labeled data is less abundant; in others (e.g.,
JavaScript) the available data maybe more focused on some application domains,
and thus less diverse. As a way around such data bottlenecks, we present
evidence suggesting that human-written code in different languages (which
performs the same function), is rather similar, and particularly preserving of
identifier naming patterns; we further present evidence suggesting that
identifiers are a very important element of training data for software
engineering tasks. We leverage this rather fortuitous phenomenon to find
evidence that available multilingual training data (across different languages)
can be used to amplify performance. We study this for 3 different tasks: code
summarization, code retrieval, and function naming. We note that this
data-augmenting approach is broadly compatible with different tasks, languages,
and machine-learning models.

    

### [[2112.02048] Graph Neural Networks for Charged Particle Tracking on FPGAs](http://arxiv.org/abs/2112.02048)


  The determination of charged particle trajectories in collisions at the CERN
Large Hadron Collider (LHC) is an important but challenging problem, especially
in the high interaction density conditions expected during the future
high-luminosity phase of the LHC (HL-LHC). Graph neural networks (GNNs) are a
type of geometric deep learning algorithm that has successfully been applied to
this task by embedding tracker data as a graph -- nodes represent hits, while
edges represent possible track segments -- and classifying the edges as true or
fake track segments. However, their study in hardware- or software-based
trigger applications has been limited due to their large computational cost. In
this paper, we introduce an automated translation workflow, integrated into a
broader tool called $\texttt{hls4ml}$, for converting GNNs into firmware for
field-programmable gate arrays (FPGAs). We use this translation tool to
implement GNNs for charged particle tracking, trained using the TrackML
challenge dataset, on FPGAs with designs targeting different graph sizes, task
complexites, and latency/throughput requirements. This work could enable the
inclusion of charged particle tracking GNNs at the trigger level for HL-LHC
experiments.

    

### [[2112.02052] TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs](http://arxiv.org/abs/2112.02052)


  Recently, graph neural networks (GNNs), as the backbone of graph-based
machine learning, demonstrate great success in various domains (e.g.,
e-commerce). However, the performance of GNNs is usually unsatisfactory due to
the highly sparse and irregular graph-based operations. To this end, we
propose, TC-GNN, the first GPU Tensor Core Unit (TCU) based GNN acceleration
framework. The core idea is to reconcile the "Sparse" GNN computation with
"Dense" TCU. Specifically, we conduct an in-depth analysis of the sparse
operations in mainstream GNN computing frameworks. We introduce a novel sparse
graph translation technique to facilitate TCU processing of sparse GNN
workload. We also implement an effective CUDA core and TCU collaboration design
to fully utilize GPU resources. We fully integrate TC-GNN with the Pytorch
framework for ease of programming. Rigorous experiments show an average of
1.70X speedup over the state-of-the-art Deep Graph Library framework across
various GNN models and dataset settings.

    

### [[2112.02072] Identifying mass composition of ultra-high-energy cosmic rays using deep learning](http://arxiv.org/abs/2112.02072)


  We introduce a novel method for identifying the mass composition of
ultra-high-energy cosmic rays using deep learning. The key idea of the method
is to use a chain of two neural networks. The first network predicts the type
of a primary particle for individual events, while the second infers the mass
composition of an ensemble of events. We apply this method to the Monte-Carlo
data for the Telescope Array Surface Detectors readings, on which it yields an
unprecedented low error of 7% for 4-component approximation. The statistical
error is shown to be inferior to the systematic one related to the choice of
the hadronic interaction model used for simulations.

    

### [[2112.02073] Hierarchical Optimal Transport for Unsupervised Domain Adaptation](http://arxiv.org/abs/2112.02073)


  In this paper, we propose a novel approach for unsupervised domain
adaptation, that relates notions of optimal transport, learning probability
measures and unsupervised learning. The proposed approach, HOT-DA, is based on
a hierarchical formulation of optimal transport, that leverages beyond the
geometrical information captured by the ground metric, richer structural
information in the source and target domains. The additional information in the
labeled source domain is formed instinctively by grouping samples into
structures according to their class labels. While exploring hidden structures
in the unlabeled target domain is reduced to the problem of learning
probability measures through Wasserstein barycenter, which we prove to be
equivalent to spectral clustering. Experiments on a toy dataset with
controllable complexity and two challenging visual adaptation datasets show the
superiority of the proposed approach over the state-of-the-art.

    

### [[2112.02077] MD-inferred neural network monoclinic finite-strain hyperelasticity models for $β$-HMX: Sobolev training and validation against physical constraints](http://arxiv.org/abs/2112.02077)


  We present a machine learning framework to train and validate neural networks
to predict the anisotropic elastic response of the monoclinic organic molecular
crystal $\beta$-HMX in the geometrical nonlinear regime. A filtered molecular
dynamic (MD) simulations database is used to train the neural networks with a
Sobolev norm that uses the stress measure and a reference configuration to
deduce the elastic stored energy functional. To improve the accuracy of the
elasticity tangent predictions originating from the learned stored energy, a
transfer learning technique is used to introduce additional tangential
constraints from the data while necessary conditions (e.g. strong ellipticity,
crystallographic symmetry) for the correctness of the model are either
introduced as additional physical constraints or incorporated in the validation
tests. Assessment of the neural networks is based on (1) the accuracy with
which they reproduce the bottom-line constitutive responses predicted by MD,
(2) detailed examination of their stability and uniqueness, and (3)
admissibility of the predicted responses with respect to continuum mechanics
theory in the finite-deformation regime. We compare the neural networks'
training efficiency under different Sobolev constraints and assess the models'
accuracy and robustness against MD benchmarks for $\beta$-HMX.

    

### [[2112.02086] Data-Free Neural Architecture Search via Recursive Label Calibration](http://arxiv.org/abs/2112.02086)


  This paper aims to explore the feasibility of neural architecture search
(NAS) given only a pre-trained model without using any original training data.
This is an important circumstance for privacy protection, bias avoidance, etc.,
in real-world scenarios. To achieve this, we start by synthesizing usable data
through recovering the knowledge from a pre-trained deep neural network. Then
we use the synthesized data and their predicted soft-labels to guide neural
architecture search. We identify that the NAS task requires the synthesized
data (we target at image domain here) with enough semantics, diversity, and a
minimal domain gap from the natural images. For semantics, we propose recursive
label calibration to produce more informative outputs. For diversity, we
propose a regional update strategy to generate more diverse and
semantically-enriched synthetic data. For minimal domain gap, we use input and
feature-level regularization to mimic the original data distribution in latent
space. We instantiate our proposed framework with three popular NAS algorithms:
DARTS, ProxylessNAS and SPOS. Surprisingly, our results demonstrate that the
architectures discovered by searching with our synthetic data achieve accuracy
that is comparable to, or even higher than, architectures discovered by
searching from the original ones, for the first time, deriving the conclusion
that NAS can be done effectively with no need of access to the original or
called natural data if the synthesis method is well designed. Our code will be
publicly available.

    

### [[2112.02089] Regularized Newton Method with Global $O(1/k^2)$ Convergence](http://arxiv.org/abs/2112.02089)


  We present a Newton-type method that converges fast from any initialization
and for arbitrary convex objectives with Lipschitz Hessians. We achieve this by
merging the ideas of cubic regularization with a certain adaptive
Levenberg--Marquardt penalty. In particular, we show that the iterates given by
$x^{k+1}=x^k - \bigl(\nabla^2 f(x^k) + \sqrt{H\|\nabla f(x^k)\|}
\mathbf{I}\bigr)^{-1}\nabla f(x^k)$, where $H>0$ is a constant, converge
globally with a $\mathcal{O}(\frac{1}{k^2})$ rate. Our method is the first
variant of Newton's method that has both cheap iterations and provably fast
global convergence. Moreover, we prove that locally our method converges
superlinearly when the objective is strongly convex. To boost the method's
performance, we present a line search procedure that does not need
hyperparameters and is provably efficient.

    

### [[2112.02091] Class-agnostic Reconstruction of Dynamic Objects from Videos](http://arxiv.org/abs/2112.02091)


  We introduce REDO, a class-agnostic framework to REconstruct the Dynamic
Objects from RGBD or calibrated videos. Compared to prior work, our problem
setting is more realistic yet more challenging for three reasons: 1) due to
occlusion or camera settings an object of interest may never be entirely
visible, but we aim to reconstruct the complete shape; 2) we aim to handle
different object dynamics including rigid motion, non-rigid motion, and
articulation; 3) we aim to reconstruct different categories of objects with one
unified framework. To address these challenges, we develop two novel modules.
First, we introduce a canonical 4D implicit function which is pixel-aligned
with aggregated temporal visual cues. Second, we develop a 4D transformation
module which captures object dynamics to support temporal propagation and
aggregation. We study the efficacy of REDO in extensive experiments on
synthetic RGBD video datasets SAIL-VOS 3D and DeformingThings4D++, and on
real-world video data 3DPW. We find REDO outperforms state-of-the-art dynamic
reconstruction methods by a margin. In ablation studies we validate each
developed component.

    

### [[2112.02093] Causal-based Time Series Domain Generalization for Vehicle Intention Prediction](http://arxiv.org/abs/2112.02093)


  Accurately predicting possible behaviors of traffic participants is an
essential capability for autonomous vehicles. Since autonomous vehicles need to
navigate in dynamically changing environments, they are expected to make
accurate predictions regardless of where they are and what driving
circumstances they encountered. Therefore, generalization capability to unseen
domains is crucial for prediction models when autonomous vehicles are deployed
in the real world. In this paper, we aim to address the domain generalization
problem for vehicle intention prediction tasks and a causal-based time series
domain generalization (CTSDG) model is proposed. We construct a structural
causal model for vehicle intention prediction tasks to learn an invariant
representation of input driving data for domain generalization. We further
integrate a recurrent latent variable model into our structural causal model to
better capture temporal latent dependencies from time-series input data. The
effectiveness of our approach is evaluated via real-world driving data. We
demonstrate that our proposed method has consistent improvement on prediction
accuracy compared to other state-of-the-art domain generalization and behavior
prediction methods.

    

### [[2112.02094] Coupling Vision and Proprioception for Navigation of Legged Robots](http://arxiv.org/abs/2112.02094)


  We exploit the complementary strengths of vision and proprioception to
achieve point goal navigation in a legged robot. Legged systems are capable of
traversing more complex terrain than wheeled robots, but to fully exploit this
capability, we need the high-level path planner in the navigation system to be
aware of the walking capabilities of the low-level locomotion policy on varying
terrains. We achieve this by using proprioceptive feedback to estimate the safe
operating limits of the walking policy, and to sense unexpected obstacles and
terrain properties like smoothness or softness of the ground that may be missed
by vision. The navigation system uses onboard cameras to generate an occupancy
map and a corresponding cost map to reach the goal. The FMM (Fast Marching
Method) planner then generates a target path. The velocity command generator
takes this as input to generate the desired velocity for the locomotion policy
using as input additional constraints, from the safety advisor, of unexpected
obstacles and terrain determined speed limits. We show superior performance
compared to wheeled robot (LoCoBot) baselines, and other baselines which have
disjoint high-level planning and low-level control. We also show the real-world
deployment of our system on a quadruped robot with onboard sensors and compute.
Videos at this https URL


### [[1704.03296] Interpretable Explanations of Black Boxes by Meaningful Perturbation](http://arxiv.org/abs/1704.03296)


  As machine learning algorithms are increasingly applied to high impact yet
high risk tasks, such as medical diagnosis or autonomous driving, it is
critical that researchers can explain how such algorithms arrived at their
predictions. In recent years, a number of image saliency methods have been
developed to summarize where highly complex neural networks "look" in an image
for evidence for their predictions. However, these techniques are limited by
their heuristic nature and architectural constraints. In this paper, we make
two main contributions: First, we propose a general framework for learning
different kinds of explanations for any black box algorithm. Second, we
specialise the framework to find the part of an image most responsible for a
classifier decision. Unlike previous works, our method is model-agnostic and
testable because it is grounded in explicit and interpretable image
perturbations.

    

### [[1811.00741] Stronger Data Poisoning Attacks Break Data Sanitization Defenses](http://arxiv.org/abs/1811.00741)


  Machine learning models trained on data from the outside world can be
corrupted by data poisoning attacks that inject malicious points into the
models' training sets. A common defense against these attacks is data
sanitization: first filter out anomalous training points before training the
model. In this paper, we develop three attacks that can bypass a broad range of
common data sanitization defenses, including anomaly detectors based on nearest
neighbors, training loss, and singular-value decomposition. By adding just 3%
poisoned data, our attacks successfully increase test error on the Enron spam
detection dataset from 3% to 24% and on the IMDB sentiment classification
dataset from 12% to 29%. In contrast, existing attacks which do not explicitly
account for these data sanitization defenses are defeated by them. Our attacks
are based on two ideas: (i) we coordinate our attacks to place poisoned points
near one another, and (ii) we formulate each attack as a constrained
optimization problem, with constraints designed to ensure that the poisoned
points evade detection. As this optimization involves solving an expensive
bilevel problem, our three attacks correspond to different ways of
approximating this problem, based on influence functions; minimax duality; and
the Karush-Kuhn-Tucker (KKT) conditions. Our results underscore the need to
develop more robust defenses against data poisoning attacks.

    

### [[1905.11678] EEG-based Emotional Video Classification via Learning Connectivity Structure](http://arxiv.org/abs/1905.11678)


  Electroencephalography (EEG) is a useful way to implicitly monitor the users
perceptual state during multimedia consumption. One of the primary challenges
for the practical use of EEG-based monitoring is to achieve a satisfactory
level of accuracy in EEG classification. Connectivity between different brain
regions is an important property for the classification of EEG. However, how to
define the connectivity structure for a given task is still an open problem,
because there is no ground truth about how the connectivity structure should be
in order to maximize the classification performance. In this paper, we propose
an end-to-end neural network model for EEG-based emotional video
classification, which can extract an appropriate multi-layer graph structure
and signal features directly from a set of raw EEG signals and perform
classification using them. Experimental results demonstrate that our method
yields improved performance in comparison to the existing approaches where
manually defined connectivity structures and signal features are used.
Furthermore, we show that the graph structure extraction process is reliable in
terms of consistency, and the learned graph structures make much sense in the
viewpoint of emotional perception occurring in the brain.

    

### [[1906.10462] Policy Optimization with Stochastic Mirror Descent](http://arxiv.org/abs/1906.10462)


  Improving sample efficiency has been a longstanding goal in reinforcement
learning. This paper proposes $\mathtt{VRMPO}$ algorithm: a sample efficient
policy gradient method with stochastic mirror descent. In $\mathtt{VRMPO}$, a
novel variance-reduced policy gradient estimator is presented to improve sample
efficiency. We prove that the proposed $\mathtt{VRMPO}$ needs only
$\mathcal{O}(\epsilon^{-3})$ sample trajectories to achieve an
$\epsilon$-approximate first-order stationary point, which matches the best
sample complexity for policy optimization. The extensive experimental results
demonstrate that $\mathtt{VRMPO}$ outperforms the state-of-the-art policy
gradient methods in various settings.

    

### [[2002.03521] UGRWO-Sampling for COVID-19 dataset: A modified random walk under-sampling approach based on graphs to imbalanced data classification](http://arxiv.org/abs/2002.03521)


  This paper proposes a new RWO-Sampling (Random Walk Over-Sampling) based on
graphs for imbalanced datasets. In this method, two schemes based on
under-sampling and over-sampling methods are introduced to keep the proximity
information robust to noises and outliers. After constructing the first graph
on minority class, RWO-Sampling will be implemented on selected samples, and
the rest will remain unchanged. The second graph is constructed for the
majority class, and the samples in a low-density area (outliers) are removed.
Finally, in the proposed method, samples of the majority class in a
high-density area are selected, and the rest are eliminated. Furthermore,
utilizing RWO-sampling, the boundary of minority class is increased though the
outliers are not raised. This method is tested, and the number of evaluation
measures is compared to previous methods on nine continuous attribute datasets
with different over-sampling rates and one data set for the diagnosis of
COVID-19 disease. The experimental results indicated the high efficiency and
flexibility of the proposed method for the classification of imbalanced data

    

### [[2003.09711] Robust Out-of-distribution Detection for Neural Networks](http://arxiv.org/abs/2003.09711)


  Detecting out-of-distribution (OOD) inputs is critical for safely deploying
deep learning models in the real world. Existing approaches for detecting OOD
examples work well when evaluated on benign in-distribution and OOD samples.
However, in this paper, we show that existing detection mechanisms can be
extremely brittle when evaluating on in-distribution and OOD inputs with
minimal adversarial perturbations which don't change their semantics. Formally,
we extensively study the problem of Robust Out-of-Distribution Detection on
common OOD detection approaches, and show that state-of-the-art OOD detectors
can be easily fooled by adding small perturbations to the in-distribution and
OOD inputs. To counteract these threats, we propose an effective algorithm
called ALOE, which performs robust training by exposing the model to both
adversarially crafted inlier and outlier examples. Our method can be flexibly
combined with, and render existing methods robust. On common benchmark
datasets, we show that ALOE substantially improves the robustness of
state-of-the-art OOD detection, with 58.4% AUROC improvement on CIFAR-10 and
46.59% improvement on CIFAR-100.

    

### [[2005.06678] Activation functions are not needed: the ratio net](http://arxiv.org/abs/2005.06678)


  A deep neural network for classification tasks is essentially consist of two
components: feature extractors and function approximators. They usually work as
an integrated whole, however, improvements on any components can promote the
performance of the whole algorithm. This paper focus on designing a new
function approximator. Conventionally, to build a function approximator, one
usually uses the method based on the nonlinear activation function or the
nonlinear kernel function and yields classical networks such as the
feed-forward neural network (MLP) and the radial basis function network (RBF).
In this paper, a new function approximator that is effective and efficient is
proposed. Instead of designing new activation functions or kernel functions,
the new proposed network uses the fractional form. For the sake of convenience,
we name the network the ratio net. We compare the effectiveness and efficiency
of the ratio net and that of the RBF and the MLP with various kinds of
activation functions in the classification task on the mnist database of
handwritten digits and the Internet Movie Database (IMDb) which is a binary
sentiment analysis dataset. It shows that, in most cases, the ratio net
converges faster and outperforms both the MLP and the RBF.

    

### [[2006.00334] Consistent feature selection for neural networks via Adaptive Group Lasso](http://arxiv.org/abs/2006.00334)


  One main obstacle for the wide use of deep learning in medical and
engineering sciences is its interpretability. While neural network models are
strong tools for making predictions, they often provide little information
about which features play significant roles in influencing the prediction
accuracy. To overcome this issue, many regularization procedures for learning
with neural networks have been proposed for dropping non-significant features.
Unfortunately, the lack of theoretical results casts doubt on the applicability
of such pipelines. In this work, we propose and establish a theoretical
guarantee for the use of the adaptive group lasso for selecting important
features of neural networks. Specifically, we show that our feature selection
method is consistent for single-output feed-forward neural networks with one
hidden layer and hyperbolic tangent activation function. We demonstrate its
applicability using both simulation and data analysis.

    

### [[2006.06763] Stochastic Saddle-Point Optimization for Wasserstein Barycenters](http://arxiv.org/abs/2006.06763)


  We consider the population Wasserstein barycenter problem for random
probability measures supported on a finite set of points and generated by an
online stream of data. This leads to a complicated stochastic optimization
problem where the objective is given as an expectation of a function given as a
solution to a random optimization problem. We employ the structure of the
problem and obtain a convex-concave stochastic saddle-point reformulation of
this problem. In the setting when the distribution of random probability
measures is discrete, we propose a stochastic optimization algorithm and
estimate its complexity. The second result, based on kernel methods, extends
the previous one to the arbitrary distribution of random probability measures.
Moreover, this new algorithm has a total complexity better than the Stochastic
Approximation approach combined with the Sinkhorn algorithm in many cases. We
also illustrate our developments by a series of numerical experiments.

    

### [[2006.06889] Fast Objective & Duality Gap Convergence for Nonconvex-Strongly-Concave Min-Max Problems](http://arxiv.org/abs/2006.06889)


  This paper focuses on stochastic methods for solving smooth non-convex
strongly-concave min-max problems, which have received increasing attention due
to their potential applications in deep learning (e.g., deep AUC maximization,
distributionally robust optimization). However, most of the existing algorithms
are slow in practice, and their analysis revolves around the convergence to a
nearly stationary point. We consider leveraging the Polyak-Łojasiewicz (PL)
condition to design faster stochastic algorithms with stronger convergence
guarantee. Although PL condition has been utilized for designing many
stochastic minimization algorithms, their applications for non-convex min-max
optimization remain rare. In this paper, we propose and analyze a generic
framework of proximal epoch-based method with many well-known stochastic
updates embeddable. Fast convergence is established in terms of both {\bf the
primal objective gap and the duality gap}. Compared with existing studies, (i)
our analysis is based on a novel Lyapunov function consisting of the primal
objective gap and the duality gap of a regularized function, and (ii) the
results are more comprehensive with improved rates that have better dependence
on the condition number under different assumptions. We also conduct deep and
non-deep learning experiments to verify the effectiveness of our methods.

    

### [[2006.12169] Bidirectionally Self-Normalizing Neural Networks](http://arxiv.org/abs/2006.12169)


  The problem of vanishing and exploding gradients has been a long-standing
obstacle that hinders the effective training of neural networks. Despite
various tricks and techniques that have been employed to alleviate the problem
in practice, there still lacks satisfactory theories or provable solutions. In
this paper, we address the problem from the perspective of high-dimensional
probability theory. We provide a rigorous result that shows, under mild
conditions, how the vanishing/exploding gradients problem disappears with high
probability if the neural networks have sufficient width. Our main idea is to
constrain both forward and backward signal propagation in a nonlinear neural
network through a new class of activation functions, namely Gaussian-Poincaré
normalized functions, and orthogonal weight matrices. Experiments on both
synthetic and real-world data validate our theory and confirm its effectiveness
on very deep neural networks when applied in practice.

    

### [[2007.12173] Bridging the Imitation Gap by Adaptive Insubordination](http://arxiv.org/abs/2007.12173)


  In practice, imitation learning is preferred over pure reinforcement learning
whenever it is possible to design a teaching agent to provide expert
supervision. However, we show that when the teaching agent makes decisions with
access to privileged information that is unavailable to the student, this
information is marginalized during imitation learning, resulting in an
"imitation gap" and, potentially, poor results. Prior work bridges this gap via
a progression from imitation learning to reinforcement learning. While often
successful, gradual progression fails for tasks that require frequent switches
between exploration and memorization. To better address these tasks and
alleviate the imitation gap we propose 'Adaptive Insubordination' (ADVISOR).
ADVISOR dynamically weights imitation and reward-based reinforcement learning
losses during training, enabling on-the-fly switching between imitation and
exploration. On a suite of challenging tasks set within gridworlds, multi-agent
particle environments, and high-fidelity 3D simulators, we show that on-the-fly
switching with ADVISOR outperforms pure imitation, pure reinforcement learning,
as well as their sequential and parallel combinations.

    

### [[2008.13109] Brain Stroke Lesion Segmentation Using Consistent Perception Generative Adversarial Network](http://arxiv.org/abs/2008.13109)


  The state-of-the-art deep learning methods have demonstrated impressive
performance in segmentation tasks. However, the success of these methods
depends on a large amount of manually labeled masks, which are expensive and
time-consuming to be collected. In this work, a novel Consistent
PerceptionGenerative Adversarial Network (CPGAN) is proposed for
semi-supervised stroke lesion segmentation. The proposed CPGAN can reduce the
reliance on fully labeled samples. Specifically, A similarity connection module
(SCM) is designed to capture the information of multi-scale features. The
proposed SCM can selectively aggregate the features at each position by a
weighted sum. Moreover, a consistent perception strategy is introduced into the
proposed model to enhance the effect of brain stroke lesion prediction for the
unlabeled data. Furthermore, an assistant network is constructed to encourage
the discriminator to learn meaningful feature representations which are often
forgotten during training stage. The assistant network and the discriminator
are employed to jointly decide whether the segmentation results are real or
fake. The CPGAN was evaluated on the Anatomical Tracings of Lesions After
Stroke (ATLAS). The experimental results demonstrate that the proposed network
achieves superior segmentation performance. In semi-supervised segmentation
task, the proposed CPGAN using only two-fifths of labeled samples outperforms
some approaches using full labeled samples.

    

### [[2009.05303] CatGCN: Graph Convolutional Networks with Categorical Node Features](http://arxiv.org/abs/2009.05303)


  Recent studies on Graph Convolutional Networks (GCNs) reveal that the initial
node representations (i.e., the node representations before the first-time
graph convolution) largely affect the final model performance. However, when
learning the initial representation for a node, most existing work linearly
combines the embeddings of node features, without considering the interactions
among the features (or feature embeddings). We argue that when the node
features are categorical, e.g., in many real-world applications like user
profiling and recommender system, feature interactions usually carry important
signals for predictive analytics. Ignoring them will result in suboptimal
initial node representation and thus weaken the effectiveness of the follow-up
graph convolution. In this paper, we propose a new GCN model named CatGCN,
which is tailored for graph learning when the node features are categorical.
Specifically, we integrate two ways of explicit interaction modeling into the
learning of initial node representation, i.e., local interaction modeling on
each pair of node features and global interaction modeling on an artificial
feature graph. We then refine the enhanced initial node representations with
the neighborhood aggregation-based graph convolution. We train CatGCN in an
end-to-end fashion and demonstrate it on semi-supervised node classification.
Extensive experiments on three tasks of user profiling (the prediction of user
age, city, and purchase level) from Tencent and Alibaba datasets validate the
effectiveness of CatGCN, especially the positive effect of performing feature
interaction modeling before graph convolution.

    

### [[2011.12087] A Convenient Infinite Dimensional Framework for Generative Adversarial Learning](http://arxiv.org/abs/2011.12087)


  In recent years, generative adversarial networks (GANs) have demonstrated
impressive experimental results while there are only a few works that foster
statistical learning theory for GANs. In this work, we propose an infinite
dimensional theoretical framework for generative adversarial learning. Assuming
the class of uniformly bounded $k$-times $\alpha$-Hölder differentiable and
uniformly positive densities, we show that the Rosenblatt transformation
induces an optimal generator, which is realizable in the hypothesis space of
$\alpha$-Hölder differentiable generators. With a consistent definition of
the hypothesis space of discriminators, we further show that in our framework
the Jensen-Shannon divergence between the distribution induced by the generator
from the adversarial learning procedure and the data generating distribution
converges to zero. Under sufficiently strict regularity assumptions on the
density of the data generating process, we also provide rates of convergence
based on concentration and chaining.

    

### [[2012.09935] Increasing the efficiency of randomized trial estimates via linear adjustment for a prognostic score](http://arxiv.org/abs/2012.09935)


  Estimating causal effects from randomized experiments is central to clinical
research. Reducing the statistical uncertainty in these analyses is an
important objective for statisticians. Registries, prior trials, and health
records constitute a growing compendium of historical data on patients under
standard-of-care that may be exploitable to this end. However, most methods for
historical borrowing achieve reductions in variance by sacrificing strict
type-I error rate control. Here, we propose a use of historical data that
exploits linear covariate adjustment to improve the efficiency of trial
analyses without incurring bias. Specifically, we train a prognostic model on
the historical data, then estimate the treatment effect using a linear
regression while adjusting for the trial subjects' predicted outcomes (their
prognostic scores). We prove that, under certain conditions, this prognostic
covariate adjustment procedure attains the minimum variance possible among a
large class of estimators. When those conditions are not met, prognostic
covariate adjustment is still more efficient than raw covariate adjustment and
the gain in efficiency is proportional to a measure of the predictive accuracy
of the prognostic model above and beyond the linear relationship with the raw
covariates. We demonstrate the approach using simulations and a reanalysis of
an Alzheimer's Disease clinical trial and observe meaningful reductions in
mean-squared error and the estimated variance. Lastly, we provide a simplified
formula for asymptotic variance that enables power calculations that account
for these gains. Sample size reductions between 10% and 30% are attainable when
using prognostic models that explain a clinically realistic percentage of the
outcome variance.

    

### [[2012.11841] Residual Matrix Product State for Machine Learning](http://arxiv.org/abs/2012.11841)


  Tensor network, which originates from quantum physics, is emerging as an
efficient tool for classical and quantum machine learning. Nevertheless, there
still exists a considerable accuracy gap between tensor network and the
sophisticated neural network models for classical machine learning. In this
work, we combine the ideas of matrix product state (MPS), the simplest tensor
network structure, and residual neural network and propose the residual matrix
product state (ResMPS). The ResMPS can be treated as a network where its layers
map the "hidden" features to the outputs (e.g., classifications), and the
variational parameters of the layers are the functions of the features of the
samples (e.g., pixels of images). This is different from neural network, where
the layers map feed-forwardly the features to the output. The ResMPS can equip
with the non-linear activations and dropout layers, and outperforms the
state-of-the-art tensor network models in terms of efficiency, stability, and
expression power. Besides, ResMPS is interpretable from the perspective of
polynomial expansion, where the factorization and exponential machines
naturally emerge. Our work contributes to connecting and hybridizing neural and
tensor networks, which is crucial to further enhance our understand of the
working mechanisms and improve the performance of both models.

    

### [[2012.13341] AudioViewer: Learning to Visualize Sounds](http://arxiv.org/abs/2012.13341)


  A long-standing goal in the field of sensory substitution is enabling sound
perception for deaf people by visualizing audio content. Different from
existing models that translate between speech and text or text and images, we
target immediate and low-level audio to video translation that applies to
generic environment sounds as well as human speech. Since such a substitution
is artificial, without labels for supervised learning, our core contribution is
to build a mapping from audio to video that learns from unpaired examples via
high-level constraints. For speech, we additionally disentangle content
(phones) from style (gender and dialect) by mapping them to a common
disentangled latent space. Qualitative and quantitative results, including a
user study, demonstrate that our unpaired translation approach maintains
important audio features in the generated video and that videos of faces and
numbers are well suited for visualizing high-dimensional audio features that
can be parsed by humans to match and distinguish between sounds, words, and
speakers.

    

### [[2101.05779] Structured Prediction as Translation between Augmented Natural Languages](http://arxiv.org/abs/2101.05779)


  We propose a new framework, Translation between Augmented Natural Languages
(TANL), to solve many structured prediction language tasks including joint
entity and relation extraction, nested named entity recognition, relation
classification, semantic role labeling, event extraction, coreference
resolution, and dialogue state tracking. Instead of tackling the problem by
training task-specific discriminative classifiers, we frame it as a translation
task between augmented natural languages, from which the task-relevant
information can be easily extracted. Our approach can match or outperform
task-specific models on all tasks, and in particular, achieves new
state-of-the-art results on joint entity and relation extraction (CoNLL04, ADE,
NYT, and ACE2005 datasets), relation classification (FewRel and TACRED), and
semantic role labeling (CoNLL-2005 and CoNLL-2012). We accomplish this while
using the same architecture and hyperparameters for all tasks and even when
training a single model to solve all tasks at the same time (multi-task
learning). Finally, we show that our framework can also significantly improve
the performance in a low-resource regime, thanks to better use of label
semantics.

    

### [[2102.04402] Contrasting Centralized and Decentralized Critics in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2102.04402)


  Centralized Training for Decentralized Execution, where agents are trained
offline using centralized information but execute in a decentralized manner
online, has gained popularity in the multi-agent reinforcement learning
community. In particular, actor-critic methods with a centralized critic and
decentralized actors are a common instance of this idea. However, the
implications of using a centralized critic in this context are not fully
discussed and understood even though it is the standard choice of many
algorithms. We therefore formally analyze centralized and decentralized critic
approaches, providing a deeper understanding of the implications of critic
choice. Because our theory makes unrealistic assumptions, we also empirically
compare the centralized and decentralized critic methods over a wide set of
environments to validate our theories and to provide practical advice. We show
that there exist misconceptions regarding centralized critics in the current
literature and show that the centralized critic design is not strictly
beneficial, but rather both centralized and decentralized critics have
different pros and cons that should be taken into account by algorithm
designers.

    

### [[2102.07827] Deep Neural Networks for Radar Waveform Classification](http://arxiv.org/abs/2102.07827)


  We consider the problem of classifying radar pulses given raw I/Q waveforms
in the presence of noise and absence of synchronization. We also consider the
problem of classifying multiple superimposed radar pulses. For both, we design
deep neural networks (DNNs) that are robust to synchronization, pulse width,
and SNR. Our designs yield more than 100x reduction in error-rate over the
previous state-of-the-art.

    

### [[2102.11756] Deep Policy Dynamic Programming for Vehicle Routing Problems](http://arxiv.org/abs/2102.11756)


  Routing problems are a class of combinatorial problems with many practical
applications. Recently, end-to-end deep learning methods have been proposed to
learn approximate solution heuristics for such problems. In contrast, classical
dynamic programming (DP) algorithms guarantee optimal solutions, but scale
badly with the problem size. We propose Deep Policy Dynamic Programming (DPDP),
which aims to combine the strengths of learned neural heuristics with those of
DP algorithms. DPDP prioritizes and restricts the DP state space using a policy
derived from a deep neural network, which is trained to predict edges from
example solutions. We evaluate our framework on the travelling salesman problem
(TSP), the vehicle routing problem (VRP) and TSP with time windows (TSPTW) and
show that the neural policy improves the performance of (restricted) DP
algorithms, making them competitive to strong alternatives such as LKH, while
also outperforming most other 'neural approaches' for solving TSPs, VRPs and
TSPTWs with 100 nodes.

    

### [[2102.11845] Learning with User-Level Privacy](http://arxiv.org/abs/2102.11845)


  We propose and analyze algorithms to solve a range of learning tasks under
user-level differential privacy constraints. Rather than guaranteeing only the
privacy of individual samples, user-level DP protects a user's entire
contribution ($m \ge 1$ samples), providing more stringent but more realistic
protection against information leaks. We show that for high-dimensional mean
estimation, empirical risk minimization with smooth losses, stochastic convex
optimization, and learning hypothesis classes with finite metric entropy, the
privacy cost decreases as $O(1/\sqrt{m})$ as users provide more samples. In
contrast, when increasing the number of users $n$, the privacy cost decreases
at a faster $O(1/n)$ rate. We complement these results with lower bounds
showing the minimax optimality of our algorithms for mean estimation and
stochastic convex optimization. Our algorithms rely on novel techniques for
private mean estimation in arbitrary dimension with error scaling as the
concentration radius $\tau$ of the distribution rather than the entire range.

    

### [[2102.13092] Quantitative approximation results for complex-valued neural networks](http://arxiv.org/abs/2102.13092)


  Until recently, applications of neural networks in machine learning have
almost exclusively relied on real-valued networks. It was recently observed,
however, that complex-valued neural networks (CVNNs) exhibit superior
performance in applications in which the input is naturally complex-valued,
such as MRI fingerprinting. While the mathematical theory of real-valued
networks has, by now, reached some level of maturity, this is far from true for
complex-valued networks. In this paper, we analyze the expressivity of
complex-valued networks by providing explicit quantitative error bounds for
approximating $C^n$ functions on compact subsets of $\mathbb{C}^d$ by
complex-valued neural networks that employ the modReLU activation function,
given by $\sigma(z) = \mathrm{ReLU}(|z| - 1) \, \mathrm{sgn} (z)$, which is one
of the most popular complex activation functions used in practice. We show that
the derived approximation rates are optimal (up to log factors) in the class of
modReLU networks with weights of moderate growth.

    

### [[2103.00213] Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention](http://arxiv.org/abs/2103.00213)


  Discovering new materials better suited to specific purposes is an important
issue in improving the quality of human life. Here, a neural network that
creates molecules that meet some desired conditions based on a deep
understanding of chemical language is proposed (Generative Chemical
Transformer, GCT). The attention mechanism in GCT allows a deeper understanding
of molecular structures beyond the limitations of chemical language itself
which cause semantic discontinuity by paying attention to characters sparsely.
It is investigated that the significance of language models for inverse
molecular design problems by quantitatively evaluating the quality of the
generated molecules. GCT generates highly realistic chemical strings that
satisfy both chemical and linguistic grammar rules. Molecules parsed from
generated strings simultaneously satisfy the multiple target properties and
vary for a single condition set. These advances will contribute to improving
the quality of human life by accelerating the process of desired material
discovery.

    

### [[2103.01208] Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers](http://arxiv.org/abs/2103.01208)


  We show that when taking into account also the image domain $[0,1]^d$,
established $l_1$-projected gradient descent (PGD) attacks are suboptimal as
they do not consider that the effective threat model is the intersection of the
$l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest
descent step for this effective threat model and show that the exact projection
onto this set is computationally feasible and yields better performance.
Moreover, we propose an adaptive form of PGD which is highly effective even
with a small budget of iterations. Our resulting $l_1$-APGD is a strong
white-box attack showing that prior works overestimated their $l_1$-robustness.
Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA
$l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the
Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which
reliably assesses adversarial robustness for the threat model of $l_1$-ball
intersected with $[0,1]^d$.

    

### [[2103.05056] LCDNet: Deep Loop Closure Detection and Point Cloud Registration for LiDAR SLAM](http://arxiv.org/abs/2103.05056)


  Loop closure detection is an essential component of Simultaneous Localization
and Mapping (SLAM) systems, which reduces the drift accumulated over time. Over
the years, several deep learning approaches have been proposed to address this
task, however their performance has been subpar compared to handcrafted
techniques, especially while dealing with reverse loops. In this paper, we
introduce the novel LCDNet that effectively detects loop closures in LiDAR
point clouds by simultaneously identifying previously visited places and
estimating the 6-DoF relative transformation between the current scan and the
map. LCDNet is composed of a shared encoder, a place recognition head that
extracts global descriptors, and a relative pose head that estimates the
transformation between two point clouds. We introduce a novel relative pose
head based on the unbalanced optimal transport theory that we implement in a
differentiable manner to allow for end-to-end training. Extensive evaluations
of LCDNet on multiple real-world autonomous driving datasets show that our
approach outperforms state-of-the-art loop closure detection and point cloud
registration techniques by a large margin, especially while dealing with
reverse loops. Moreover, we integrate our proposed loop closure detection
approach into a LiDAR SLAM library to provide a complete mapping system and
demonstrate the generalization ability using different sensor setup in an
unseen city.

    

### [[2103.06064] Graph Neural Networks Inspired by Classical Iterative Algorithms](http://arxiv.org/abs/2103.06064)


  Despite the recent success of graph neural networks (GNN), common
architectures often exhibit significant limitations, including sensitivity to
oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur
as a result of graph heterophily or adversarial attacks. To at least partially
address these issues within a simple transparent framework, we consider a new
family of GNN layers designed to mimic and integrate the update rules of two
classical iterative algorithms, namely, proximal gradient descent and iterative
reweighted least squares (IRLS). The former defines an extensible base GNN
architecture that is immune to oversmoothing while nonetheless capturing
long-range dependencies by allowing arbitrary propagation steps. In contrast,
the latter produces a novel attention mechanism that is explicitly anchored to
an underlying end-to-end energy function, contributing stability with respect
to edge uncertainty. When combined we obtain an extremely simple yet robust
model that we evaluate across disparate scenarios including standardized
benchmarks, adversarially-perturbated graphs, graphs with heterophily, and
graphs involving long-range dependencies. In doing so, we compare against SOTA
GNN approaches that have been explicitly designed for the respective task,
achieving competitive or superior node classification accuracy. Our code is
available at this https URL.

    

### [[2103.09660] Scalable Hypergraph Embedding System](http://arxiv.org/abs/2103.09660)


  Many problems such as vertex classification andlink prediction in network
data can be solvedusing graph embeddings, and a number of algo-rithms are known
for constructing such embed-dings. However, it is difficult to use graphs
tocapture non-binary relations such as communitiesof vertices. These kinds of
complex relations areexpressed more naturally as hypergraphs. Whilehypergraphs
are a generalization of graphs, state-of-the-art graph embedding techniques are
notadequate for solving prediction and classificationtasks on large hypergraphs
accurately in reason-able time. In this paper, we introduce NetVec,a novel
multi-level framework for scalable un-supervised hypergraph embedding, that can
becoupled with any graph embedding algorithm toproduce embeddings of
hypergraphs with millionsof nodes and hyperedges in a few minutes.

    

### [[2104.06323] δ-CLUE: Diverse Sets of Explanations for Uncertainty Estimates](http://arxiv.org/abs/2104.06323)


  To interpret uncertainty estimates from differentiable probabilistic models,
recent work has proposed generating Counterfactual Latent Uncertainty
Explanations (CLUEs). However, for a single input, such approaches could output
a variety of explanations due to the lack of constraints placed on the
explanation. Here we augment the original CLUE approach, to provide what we
call $\delta$-CLUE. CLUE indicates $\it{one}$ way to change an input, while
remaining on the data manifold, such that the model becomes more confident
about its prediction. We instead return a $\it{set}$ of plausible CLUEs:
multiple, diverse inputs that are within a $\delta$ ball of the original input
in latent space, all yielding confident predictions.

    

### [[2105.00429] Controlling Smart Inverters using Proxies: A Chance-Constrained DNN-based Approach](http://arxiv.org/abs/2105.00429)


  Coordinating inverters at scale under uncertainty is the desideratum for
integrating renewables in distribution grids. Unless load demands and solar
generation are telemetered frequently, controlling inverters given approximate
grid conditions or proxies thereof becomes a key specification. Although deep
neural networks (DNNs) can learn optimal inverter schedules, guaranteeing
feasibility is largely elusive. Rather than training DNNs to imitate already
computed optimal power flow (OPF) solutions, this work integrates DNN-based
inverter policies into the OPF. The proposed DNNs are trained through two OPF
alternatives that confine voltage deviations on the average and as a convex
restriction of chance constraints. The trained DNNs can be driven by partial,
noisy, or proxy descriptors of the current grid conditions. This is important
when OPF has to be solved for an unobservable feeder. DNN weights are trained
via back-propagation and upon differentiating the AC power flow equations
assuming the network model is known. Otherwise, a gradient-free variant is put
forth. The latter is relevant when inverters are controlled by an aggregator
having access only to a power flow solver or a digital twin of the feeder.
Numerical tests compare the DNN-based inverter control schemes with the optimal
inverter setpoints in terms of optimality and feasibility.

    

### [[2105.09295] Online Selection of Diverse Committees](http://arxiv.org/abs/2105.09295)


  Citizens' assemblies need to represent subpopulations according to their
proportions in the general population. These large committees are often
constructed in an online fashion by contacting people, asking for the
demographic features of the volunteers, and deciding to include them or not.
This raises a trade-off between the number of people contacted (and the
incurring cost) and the representativeness of the committee. We study three
methods, theoretically and experimentally: a greedy algorithm that includes
volunteers as long as proportionality is not violated; a non-adaptive method
that includes a volunteer with a probability depending only on their features,
assuming that the joint feature distribution in the volunteer pool is known;
and a reinforcement learning based approach when this distribution is not known
a priori but learnt online.

    

### [[2106.06860] A Minimalist Approach to Offline Reinforcement Learning](http://arxiv.org/abs/2106.06860)


  Offline reinforcement learning (RL) defines the task of learning from a fixed
batch of data. Due to errors in value estimation from out-of-distribution
actions, most offline RL algorithms take the approach of constraining or
regularizing the policy with the actions contained in the dataset. Built on
pre-existing RL algorithms, modifications to make an RL algorithm work offline
comes at the cost of additional complexity. Offline RL algorithms introduce new
hyperparameters and often leverage secondary components such as generative
models, while adjusting the underlying RL algorithm. In this paper we aim to
make a deep RL algorithm work while making minimal changes. We find that we can
match the performance of state-of-the-art offline RL algorithms by simply
adding a behavior cloning term to the policy update of an online RL algorithm
and normalizing the data. The resulting algorithm is a simple to implement and
tune baseline, while more than halving the overall run time by removing the
additional computational overhead of previous methods.

    

### [[2106.08320] Self-Supervised Learning with Kernel Dependence Maximization](http://arxiv.org/abs/2106.08320)


  We approach self-supervised learning of image representations from a
statistical dependence perspective, proposing Self-Supervised Learning with the
Hilbert-Schmidt Independence Criterion (SSL-HSIC). SSL-HSIC maximizes
dependence between representations of transformations of an image and the image
identity, while minimizing the kernelized variance of those representations.
This framework yields a new understanding of InfoNCE, a variational lower bound
on the mutual information (MI) between different transformations. While the MI
itself is known to have pathologies which can result in learning meaningless
representations, its bound is much better behaved: we show that it implicitly
approximates SSL-HSIC (with a slightly different regularizer). Our approach
also gives us insight into BYOL, a negative-free SSL method, since SSL-HSIC
similarly learns local neighborhoods of samples. SSL-HSIC allows us to directly
optimize statistical dependence in time linear in the batch size, without
restrictive data assumptions or indirect mutual information estimators. Trained
with or without a target network, SSL-HSIC matches the current state-of-the-art
for standard linear evaluation on ImageNet, semi-supervised learning and
transfer to other classification and vision tasks such as semantic
segmentation, depth estimation and object recognition. Code is available at
this https URL .

    

### [[2106.11695] The Hitchhiker's Guide to Prior-Shift Adaptation](http://arxiv.org/abs/2106.11695)


  In many computer vision classification tasks, class priors at test time often
differ from priors on the training set. In the case of such prior shift,
classifiers must be adapted correspondingly to maintain close to optimal
performance. This paper analyzes methods for adaptation of probabilistic
classifiers to new priors and for estimating new priors on an unlabeled test
set. We propose a novel method to address a known issue of prior estimation
methods based on confusion matrices, where inconsistent estimates of decision
probabilities and confusion matrices lead to negative values in the estimated
priors. Experiments on fine-grained image classification datasets provide
insight into the best practice of prior shift estimation and classifier
adaptation, and show that the proposed method achieves state-of-the-art results
in prior adaptation. Applying the best practice to two tasks with naturally
imbalanced priors, learning from web-crawled images and plant species
classification, increased the recognition accuracy by 1.1% and 3.4%
respectively.

    

### [[2106.15587] Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation](http://arxiv.org/abs/2106.15587)


  The generalization gap in reinforcement learning (RL) has been a significant
obstacle that prevents the RL agent from learning general skills and adapting
to varying environments. Increasing the generalization capacity of the RL
systems can significantly improve their performance on real-world working
environments. In this work, we propose a novel policy-aware adversarial data
augmentation method to augment the standard policy learning method with
automatically generated trajectory data. Different from the commonly used
observation transformation based data augmentations, our proposed method
adversarially generates new trajectory data based on the policy gradient
objective and aims to more effectively increase the RL agent's generalization
ability with the policy-aware data augmentation. Moreover, we further deploy a
mixup step to integrate the original and generated data to enhance the
generalization capacity while mitigating the over-deviation of the adversarial
data. We conduct experiments on a number of RL tasks to investigate the
generalization performance of the proposed method by comparing it with the
standard baselines and the state-of-the-art mixreg approach. The results show
our method can generalize well with limited training diversity, and achieve the
state-of-the-art generalization test performance.

    

### [[2102.00184] Adversarially learning disentangled speech representations for robust multi-factor voice conversion](http://arxiv.org/abs/2102.00184)


  Factorizing speech as disentangled speech representations is vital to achieve
highly controllable style transfer in voice conversion (VC). Conventional
speech representation learning methods in VC only factorize speech as speaker
and content, lacking controllability on other prosody-related factors.
State-of-the-art speech representation learning methods for more speechfactors
are using primary disentangle algorithms such as random resampling and ad-hoc
bottleneck layer size adjustment,which however is hard to ensure robust speech
representationdisentanglement. To increase the robustness of highly
controllable style transfer on multiple factors in VC, we propose a
disentangled speech representation learning framework based on adversarial
learning. Four speech representations characterizing content, timbre, rhythm
and pitch are extracted, and further disentangled by an adversarial
Mask-And-Predict (MAP)network inspired by BERT. The adversarial network is used
tominimize the correlations between the speech representations,by randomly
masking and predicting one of the representationsfrom the others. Experimental
results show that the proposedframework significantly improves the robustness
of VC on multiple factors by increasing the speech quality MOS from 2.79 to3.30
and decreasing the MCD from 3.89 to 3.58.

    

### [[2112.01658] Virtual Coset Coding for Encrypted Non-Volatile Memories with Multi-Level Cells](http://arxiv.org/abs/2112.01658)


  PCM is a popular backing memory for DRAM main memory in tiered memory
systems. PCM has asymmetric access energy; writes dominate reads. MLC asymmetry
can vary by an order of magnitude. Many schemes have been developed to take
advantage of the asymmetric patterns of 0s and 1s in the data to reduce write
energy. Because the memory is non-volatile, data can be recovered via physical
attack or across system reboot cycles. To protect information stored in PCM
against these attacks requires encryption. Unfortunately, most encryption
algorithms scramble 0s and 1s in the data, effectively removing any patterns
and negatively impacting schemes that leverage data bias and similarity to
reduce write energy. In this paper, we introduce Virtual Coset Coding (VCC) as
a workload-independent approach that reduces costly symbol transitions for
storing encrypted data. VCC is based on two ideas. First, using coset encoding
with random coset candidates, it is possible to effectively reduce the
frequency of costly bit/symbol transitions when writing encrypted data. Second,
a small set of random substrings can be used to achieve the same encoding
efficiency as a large number of random coset candidates, but at a much lower
encoding/decoding cost. Additionally, we demonstrate how VCC can be leveraged
for energy reduction in combination with fault-mitigation and fault-tolerance
to dramatically increase the lifetimes of endurance-limited NVMs, such as PCM.
We evaluate the design of VCC and demonstrate that it can be implemented
on-chip with only a nominal area overhead. VCC reduces dynamic energy by 22-28%
while maintaining the same performance. Using our multi-objective optimization
approach achieves at least a 36% improvement in lifetime over the
state-of-the-art and at least a 50% improvement in lifetime vs. an unencoded
memory, while maintaining its energy savings and system performance.

    

### [[2112.01852] Grid on QPACE 4](http://arxiv.org/abs/2112.01852)


  In 2020 we deployed QPACE 4, which features 64 Fujitsu A64FX model FX700
processors interconnected by InfiniBand EDR. QPACE 4 runs an open-source
software stack. For Lattice QCD simulations we ported the Grid LQCD framework
to support the ARM Scalable Vector Extension (SVE). In this contribution we
discuss our SVE port of Grid, the status of SVE compilers and the performance
of Grid. We also present the benefits of an alternative data layout of complex
numbers for the Domain Wall operator.

    

### [[2112.01958] Genetic-based optimization in Fog Computing: current trends and research opportunities](http://arxiv.org/abs/2112.01958)


  Fog Computing is a new computational paradigm that emerges from the need of
reducing the network usage and latency in Internet of Things (IoT). Fog can be
understood as a continuum between the Cloud layer and the IoT users that allows
to execute applications or store/process data in the networks devices of the
infrastructure. This paper aims to review current trends in the use of Genetic
Algorithm for the optimization of resource management in Fog architecture.
Related papers are classified concerning the optimization scope and the
genetic. Finally, future research lines are also presented.

    

### [[2103.16251] The randomized local computation complexity of the Lovász local lemma](http://arxiv.org/abs/2103.16251)


  The Local Computation Algorithm (LCA) model is a popular model in the field
of sublinear-time algorithms that measures the complexity of an algorithm by
the number of probes the algorithm makes in the neighborhood of one node to
determine that node's output.
In this paper we show that the randomized LCA complexity of the Lovász
Local Lemma (LLL) on constant degree graphs is $\Theta(\log n)$. The lower
bound follows by proving an $\Omega(\log n)$ lower bound for the Sinkless
Orientation problem introduced in [Brandt et al. STOC 2016]. This answers a
question of [Rosenbaum, Suomela PODC 2020].
Additionally, we show that every randomized LCA algorithm for a locally
checkable problem with a probe complexity of $o(\sqrt{\log{n}})$ can be turned
into a deterministic LCA algorithm with a probe complexity of $O(\log^* n)$.
This improves exponentially upon the currently best known speed-up result from
$o(\log \log n)$ to $O(\log^* n)$ implied by the result of [Chang, Pettie FOCS
2017] in the LOCAL model.
Finally, we show that for every fixed constant $c \geq 2$, the deterministic
VOLUME complexity of $c$-coloring a bounded degree tree is $\Theta(n)$, where
the VOLUME model is a close relative of the LCA model that was recently
introduced by [Rosenbaum, Suomela PODC 2020].

    

### [[2112.01587] Quantifying the uncertainty of neural networks using Monte Carlo dropout for deep learning based quantitative MRI](http://arxiv.org/abs/2112.01587)


  Dropout is conventionally used during the training phase as regularization
method and for quantifying uncertainty in deep learning. We propose to use
dropout during training as well as inference steps, and average multiple
predictions to improve the accuracy, while reducing and quantifying the
uncertainty. The results are evaluated for fractional anisotropy (FA) and mean
diffusivity (MD) maps which are obtained from only 3 direction scans. With our
method, accuracy can be improved significantly compared to network outputs
without dropout, especially when the training dataset is small. Moreover,
confidence maps are generated which may aid in diagnosis of unseen pathology or
artifacts.

    

### [[2112.01589] InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation](http://arxiv.org/abs/2112.01589)


  Assessing the quality of natural language generation systems through human
annotation is very expensive. Additionally, human annotation campaigns are
time-consuming and include non-reusable human labour. In practice, researchers
rely on automatic metrics as a proxy of quality. In the last decade, many
string-based metrics (e.g., BLEU) have been introduced. However, such metrics
usually rely on exact matches and thus, do not robustly handle synonyms. In
this paper, we introduce InfoLM a family of untrained metrics that can be
viewed as a string-based metric that addresses the aforementioned flaws thanks
to a pre-trained masked language model. This family of metrics also makes use
of information measures allowing the adaptation of InfoLM to various evaluation
criteria. Using direct assessment, we demonstrate that InfoLM achieves
statistically significant improvement and over $10$ points of correlation gains
in many configurations on both summarization and data2text generation.

    

### [[2112.01616] Evaluator for Emotionally Consistent Chatbots](http://arxiv.org/abs/2112.01616)


  One challenge for evaluating current sequence- or dialogue-level chatbots,
such as Empathetic Open-domain Conversation Models, is to determine whether the
chatbot performs in an emotionally consistent way. The most recent work only
evaluates on the aspects of context coherence, language fluency, response
diversity, or logical self-consistency between dialogues. This work proposes
training an evaluator to determine the emotional consistency of chatbots.

    

### [[2112.01629] Engineering AI Tools for Systematic and Scalable Quality Assessment in Magnetic Resonance Imaging](http://arxiv.org/abs/2112.01629)


  A desire to achieve large medical imaging datasets keeps increasing as
machine learning algorithms, parallel computing, and hardware technology
evolve. Accordingly, there is a growing demand in pooling data from multiple
clinical and academic institutes to enable large-scale clinical or
translational research studies. Magnetic resonance imaging (MRI) is a
frequently used, non-invasive imaging modality. However, constructing a big MRI
data repository has multiple challenges related to privacy, data size, DICOM
format, logistics, and non-standardized images. Not only building the data
repository is difficult, but using data pooled from the repository is also
challenging, due to heterogeneity in image acquisition, reconstruction, and
processing pipelines across MRI vendors and imaging sites. This position paper
describes challenges in constructing a large MRI data repository and using data
downloaded from such data repositories in various aspects. To help address the
challenges, the paper proposes introducing a quality assessment pipeline, with
considerations and general design principles.

    

### [[2112.01640] LongChecker: Improving scientific claim verification by modeling full-abstract context](http://arxiv.org/abs/2112.01640)


  We introduce the LongChecker system for scientific claim verification. Given
a scientific claim and an evidence-containing research abstract, LongChecker
predicts a veracity label and identifies supporting rationales in a multitask
fashion based on a shared encoding of the claim and abstract. We perform
experiments on the SciFact dataset, and find that LongChecker achieves
state-of-the-art performance. We conduct analysis to understand the source of
this improvement, and find that identifying the relationship between a claim
and a rationale reporting a scientific finding often requires understanding the
context in which the rationale appears. By making labeling decisions based on
all available context, LongChecker achieves better performance on cases
requiring this type of understanding. In addition, we show that LongChecker is
able to leverage weakly-supervised in-domain data to facilitate few-shot domain
adaptation for scientific claim verification.

    

### [[2112.01646] Investigating the usefulness of Quantum Blur](http://arxiv.org/abs/2112.01646)


  Though some years remain before quantum computation can outperform
conventional computation, it already provides resources that be used for
exploratory purposes in various fields. This includes certain tasks for
procedural generation in computer games, music and art. The Quantum Blur method
was introduced as a proof-of-principle example, to show that it can be useful
to design methods for procedural generation using the principles of quantum
software. Here we analyse the effects of the method and compare it to
conventional blur effects. We also determine how the effects seen derive from
the manipulation of quantum superposition and entanglement.

    

### [[2112.01651] Multi-modal application: Image Memes Generation](http://arxiv.org/abs/2112.01651)


  Meme is an interesting word. Internet memes offer unique insights into the
changes in our perception of the world, the media and our own lives. If you
surf the Internet for long enough, you will see it somewhere on the Internet.
With the rise of social media platforms and convenient image dissemination,
Image Meme has gained fame. Image memes have become a kind of pop culture and
they play an important role in communication over social media, blogs, and open
messages. With the development of artificial intelligence and the widespread
use of deep learning, Natural Language Processing (NLP) and Computer Vision
(CV) can also be used to solve more problems in life, including meme
generation. An Internet meme commonly takes the form of an image and is created
by combining a meme template (image) and a caption (natural language sentence).
In our project, we propose an end-to-end encoder-decoder architecture meme
generator. For a given input sentence, we use the Meme template selection model
to determine the emotion it expresses and select the image template. Then
generate captions and memes through to the meme caption generator. Code and
models are available at github

    

### [[2112.01660] The Influence of Data Pre-processing and Post-processing on Long Document Summarization](http://arxiv.org/abs/2112.01660)


  Long document summarization is an important and hard task in the field of
natural language processing. A good performance of the long document
summarization reveals the model has a decent understanding of the human
language. Currently, most researches focus on how to modify the attention
mechanism of the transformer to achieve a higher ROUGE score. The study of data
pre-processing and post-processing are relatively few. In this paper, we use
two pre-processing methods and a post-processing method and analyze the effect
of these methods on various long document summarization models.

    

### [[2112.01671] An Automatic Approach for Generating Rich, Linked Geo-Metadata from Historical Map Images](http://arxiv.org/abs/2112.01671)


  Historical maps contain detailed geographic information difficult to find
elsewhere covering long-periods of time (e.g., 125 years for the historical
topographic maps in the US). However, these maps typically exist as scanned
images without searchable metadata. Existing approaches making historical maps
searchable rely on tedious manual work (including crowd-sourcing) to generate
the metadata (e.g., geolocations and keywords). Optical character recognition
(OCR) software could alleviate the required manual work, but the recognition
results are individual words instead of location phrases (e.g., "Black" and
"Mountain" vs. "Black Mountain"). This paper presents an end-to-end approach to
address the real-world problem of finding and indexing historical map images.
This approach automatically processes historical map images to extract their
text content and generates a set of metadata that is linked to large external
geospatial knowledge bases. The linked metadata in the RDF (Resource
Description Framework) format support complex queries for finding and indexing
historical maps, such as retrieving all historical maps covering mountain peaks
higher than 1,000 meters in California. We have implemented the approach in a
system called mapKurator. We have evaluated mapKurator using historical maps
from several sources with various map styles, scales, and coverage. Our results
show significant improvement over the state-of-the-art methods. The code has
been made publicly available as modules of the Kartta Labs project at
this https URL.

    

### [[2112.01683] TransZero: Attribute-guided Transformer for Zero-Shot Learning](http://arxiv.org/abs/2112.01683)


  Zero-shot learning (ZSL) aims to recognize novel classes by transferring
semantic knowledge from seen classes to unseen ones. Semantic knowledge is
learned from attribute descriptions shared between different classes, which act
as strong priors for localizing object attributes that represent discriminative
region features, enabling significant visual-semantic interaction. Although
some attention-based models have attempted to learn such region features in a
single image, the transferability and discriminative attribute localization of
visual features are typically neglected. In this paper, we propose an
attribute-guided Transformer network, termed TransZero, to refine visual
features and learn attribute localization for discriminative visual embedding
representations in ZSL. Specifically, TransZero takes a feature augmentation
encoder to alleviate the cross-dataset bias between ImageNet and ZSL
benchmarks, and improves the transferability of visual features by reducing the
entangled relative geometry relationships among region features. To learn
locality-augmented visual features, TransZero employs a visual-semantic decoder
to localize the image regions most relevant to each attribute in a given image,
under the guidance of semantic attribute information. Then, the
locality-augmented visual features and semantic vectors are used to conduct
effective visual-semantic interaction in a visual-semantic embedding network.
Extensive experiments show that TransZero achieves the new state of the art on
three ZSL benchmarks. The codes are available at:
\url{this https URL}.

    

### [[2112.01707] TransCouplet:Transformer based Chinese Couplet Generation](http://arxiv.org/abs/2112.01707)


  Chinese couplet is a special form of poetry composed of complex syntax with
ancient Chinese language. Due to the complexity of semantic and grammatical
rules, creation of a suitable couplet is a formidable challenge. This paper
presents a transformer-based sequence-to-sequence couplet generation model.
With the utilization of AnchiBERT, the model is able to capture ancient Chinese
language understanding. Moreover, we evaluate the Glyph, PinYin and
Part-of-Speech tagging on the couplet grammatical rules to further improve the
model.

    

### [[2112.01715] Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks](http://arxiv.org/abs/2112.01715)


  Self-supervised learning aims to learn image feature representations without
the usage of manually annotated labels. It is often used as a precursor step to
obtain useful initial network weights which contribute to faster convergence
and superior performance of downstream tasks. While self-supervision allows one
to reduce the domain gap between supervised and unsupervised learning without
the usage of labels, the self-supervised objective still requires a strong
inductive bias to downstream tasks for effective transfer learning. In this
work, we present our material and texture based self-supervision method named
MATTER (MATerial and TExture Representation Learning), which is inspired by
classical material and texture methods. Material and texture can effectively
describe any surface, including its tactile properties, color, and specularity.
By extension, effective representation of material and texture can describe
other semantic classes strongly associated with said material and texture.
MATTER leverages multi-temporal, spatially aligned remote sensing imagery over
unchanged regions to learn invariance to illumination and viewing angle as a
mechanism to achieve consistency of material and texture representation. We
show that our self-supervision pre-training method allows for up to 24.22% and
6.33% performance increase in unsupervised and fine-tuned setups, and up to 76%
faster convergence on change detection, land cover classification, and semantic
segmentation tasks.

    

### [[2112.01724] Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach](http://arxiv.org/abs/2112.01724)


  Deep Learning (DL)-based malware detectors are increasingly adopted for early
detection of malicious behavior in cybersecurity. However, their sensitivity to
adversarial malware variants has raised immense security concerns. Generating
such adversarial variants by the defender is crucial to improving the
resistance of DL-based malware detectors against them. This necessity has given
rise to an emerging stream of machine learning research, Adversarial Malware
example Generation (AMG), which aims to generate evasive adversarial malware
variants that preserve the malicious functionality of a given malware. Within
AMG research, black-box method has gained more attention than white-box
methods. However, most black-box AMG methods require numerous interactions with
the malware detectors to generate adversarial malware examples. Given that most
malware detectors enforce a query limit, this could result in generating
non-realistic adversarial examples that are likely to be detected in practice
due to lack of stealth. In this study, we show that a novel DL-based causal
language model enables single-shot evasion (i.e., with only one query to
malware detector) by treating the content of the malware executable as a byte
sequence and training a Generative Pre-Trained Transformer (GPT). Our proposed
method, MalGPT, significantly outperformed the leading benchmark methods on a
real-world malware dataset obtained from VirusTotal, achieving over 24.51\%
evasion rate. MalGPT enables cybersecurity researchers to develop advanced
defense capabilities by emulating large-scale realistic AMG.

    

### [[2112.01751] MaxRay: A Raytracing-based Integrated Sensing and Communication Framework](http://arxiv.org/abs/2112.01751)


  Integrated Sensing And Communication (ISAC)forms a symbiosis between the
human need for communication and the need for increasing productivity, by
extracting environmental information leveraging the communication network. As
multiple sensory already create a perception of the environment, an
investigation into the advantages of ISAC compare to such modalities is
required. Therefore, we introduce MaxRay, an ISAC framework allowing to
simulate communication, sensing, and additional sensory jointly. Emphasizing
the challenges for creating such sensing networks, we introduce the required
propagation properties for sensing and how they are leveraged. To compare the
performance of the different sensing techniques, we analyze four commonly used
metrics used in different fields and evaluate their advantages and
disadvantages for sensing. We depict that a metric based on prominence is
suitable to cover most algorithms. Further we highlight the requirement of
clutter removal algorithms, using two standard clutter removal techniques to
detect a target in a typical industrial scenario. In general a versatile
framework, allowing to create automatically labeled datasets to investigate a
large variety of tasks is demonstrated.

    

### [[2112.01759] NeRF-SR: High-Quality Neural Radiance Fields using Super-Sampling](http://arxiv.org/abs/2112.01759)


  We present NeRF-SR, a solution for high-resolution (HR) novel view synthesis
with mostly low-resolution (LR) inputs. Our method is built upon Neural
Radiance Fields (NeRF) that predicts per-point density and color with a
multi-layer perceptron. While producing images at arbitrary scales, NeRF
struggles with resolutions that go beyond observed images. Our key insight is
that NeRF has a local prior, which means predictions of a 3D point can be
propagated in the nearby region and remain accurate. We first exploit it by a
super-sampling strategy that shoots multiple rays at each image pixel, which
enforces multi-view constraint at a sub-pixel level. Then, we show that NeRF-SR
can further boost the performance of super-sampling by a refinement network
that leverages the estimated depth at hand to hallucinate details from related
patches on an HR reference image. Experiment results demonstrate that NeRF-SR
generates high-quality results for novel view synthesis at HR on both synthetic
and real-world datasets.

    

### [[2112.01787] Detect Faces Efficiently: A Survey and Evaluations](http://arxiv.org/abs/2112.01787)


  Face detection is to search all the possible regions for faces in images and
locate the faces if there are any. Many applications including face
recognition, facial expression recognition, face tracking and head-pose
estimation assume that both the location and the size of faces are known in the
image. In recent decades, researchers have created many typical and efficient
face detectors from the Viola-Jones face detector to current CNN-based ones.
However, with the tremendous increase in images and videos with variations in
face scale, appearance, expression, occlusion and pose, traditional face
detectors are challenged to detect various "in the wild" faces. The emergence
of deep learning techniques brought remarkable breakthroughs to face detection
along with the price of a considerable increase in computation. This paper
introduces representative deep learning-based methods and presents a deep and
thorough analysis in terms of accuracy and efficiency. We further compare and
discuss the popular and challenging datasets and their evaluation metrics. A
comprehensive comparison of several successful deep learning-based face
detectors is conducted to uncover their efficiency using two metrics: FLOPs and
latency. The paper can guide to choose appropriate face detectors for different
applications and also to develop more efficient and accurate detectors.

    

### [[2112.01793] A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization](http://arxiv.org/abs/2112.01793)


  Four-variable-independent-regression localization losses, such as
Smooth-$\ell_1$ Loss, are used by default in modern detectors. Nevertheless,
this kind of loss is oversimplified so that it is inconsistent with the final
evaluation metric, intersection over union (IoU). Directly employing the
standard IoU is also not infeasible, since the constant-zero plateau in the
case of non-overlapping boxes and the non-zero gradient at the minimum may make
it not trainable. Accordingly, we propose a systematic method to address these
problems. Firstly, we propose a new metric, the extended IoU (EIoU), which is
well-defined when two boxes are not overlapping and reduced to the standard IoU
when overlapping. Secondly, we present the convexification technique (CT) to
construct a loss on the basis of EIoU, which can guarantee the gradient at the
minimum to be zero. Thirdly, we propose a steady optimization technique (SOT)
to make the fractional EIoU loss approaching the minimum more steadily and
smoothly. Fourthly, to fully exploit the capability of the EIoU based loss, we
introduce an interrelated IoU-predicting head to further boost localization
accuracy. With the proposed contributions, the new method incorporated into
Faster R-CNN with ResNet50+FPN as the backbone yields \textbf{4.2 mAP} gain on
VOC2007 and \textbf{2.3 mAP} gain on COCO2017 over the baseline Smooth-$\ell_1$
Loss, at almost \textbf{no training and inferencing computational cost}.
Specifically, the stricter the metric is, the more notable the gain is,
improving \textbf{8.2 mAP} on VOC2007 and \textbf{5.4 mAP} on COCO2017 at
metric $AP_{90}$.

    

### [[2112.01800] A Survey: Deep Learning for Hyperspectral Image Classification with Few Labeled Samples](http://arxiv.org/abs/2112.01800)


  With the rapid development of deep learning technology and improvement in
computing capability, deep learning has been widely used in the field of
hyperspectral image (HSI) classification. In general, deep learning models
often contain many trainable parameters and require a massive number of labeled
samples to achieve optimal performance. However, in regard to HSI
classification, a large number of labeled samples is generally difficult to
acquire due to the difficulty and time-consuming nature of manual labeling.
Therefore, many research works focus on building a deep learning model for HSI
classification with few labeled samples. In this article, we concentrate on
this topic and provide a systematic review of the relevant literature.
Specifically, the contributions of this paper are twofold. First, the research
progress of related methods is categorized according to the learning paradigm,
including transfer learning, active learning and few-shot learning. Second, a
number of experiments with various state-of-the-art approaches has been carried
out, and the results are summarized to reveal the potential research
directions. More importantly, it is notable that although there is a vast gap
between deep learning models (that usually need sufficient labeled samples) and
the HSI scenario with few labeled samples, the issues of small-sample sets can
be well characterized by fusion of deep learning methods and related
techniques, such as transfer learning and a lightweight model. For
reproducibility, the source codes of the methods assessed in the paper can be
found at this https URL.

    

### [[2112.01840] Graph-Guided Deformation for Point Cloud Completion](http://arxiv.org/abs/2112.01840)


  For a long time, the point cloud completion task has been regarded as a pure
generation task. After obtaining the global shape code through the encoder, a
complete point cloud is generated using the shape priorly learnt by the
networks. However, such models are undesirably biased towards prior average
objects and inherently limited to fit geometry details. In this paper, we
propose a Graph-Guided Deformation Network, which respectively regards the
input data and intermediate generation as controlling and supporting points,
and models the optimization guided by a graph convolutional network(GCN) for
the point cloud completion task. Our key insight is to simulate the least
square Laplacian deformation process via mesh deformation methods, which brings
adaptivity for modeling variation in geometry details. By this means, we also
reduce the gap between the completion task and the mesh deformation algorithms.
As far as we know, we are the first to refine the point cloud completion task
by mimicing traditional graphics algorithms with GCN-guided deformation. We
have conducted extensive experiments on both the simulated indoor dataset
ShapeNet, outdoor dataset KITTI, and our self-collected autonomous driving
dataset Pandar40. The results show that our method outperforms the existing
state-of-the-art algorithms in the 3D point cloud completion task.

    

### [[2112.01846] A Proposal of Automatic Error Correction in Text](http://arxiv.org/abs/2112.01846)


  The great amount of information that can be stored in electronic media is
growing up daily. Many of them is got mainly by typing, such as the huge of
information obtained from web 2.0 sites; or scaned and processing by an Optical
Character Recognition software, like the texts of libraries and goverment
offices. Both processes introduce error in texts, so it is difficult to use the
data for other purposes than just to read it, i.e. the processing of those
texts by other applications like e-learning, learning of languages, electronic
tutorials, data minning, information retrieval and even more specialized
systems such as tiflologic software, specifically blinded people-oriented
applications like automatic reading, where the text would be error free as
possible in order to make easier the text to speech task, and so on. In this
paper it is showed an application of automatic recognition and correction of
ortographic errors in electronic texts. This task is composed of three stages:
a) error detection; b) candidate corrections generation; and c) correction
-selection of the best candidate. The proposal is based in part of speech text
categorization, word similarity, word diccionaries, statistical measures,
morphologic analisys and n-grams based language model of Spanish.

    

### [[2112.01847] Causal Homotopy](http://arxiv.org/abs/2112.01847)


  We characterize homotopical equivalences between causal DAG models,
exploiting the close connections between partially ordered set representations
of DAGs (posets) and finite Alexandroff topologies. Alexandroff spaces yield a
directional topological space: the topology is defined by a unique minimal
basis defined by an open set for each variable x, specified as the intersection
of all open sets containing x. Alexandroff spaces induce a (reflexive,
transitive) preorder. Alexandroff spaces satisfying the Kolmogorov T0
separation criterion, where open sets distinguish variables, converts the
preordering into a partial ordering. Our approach broadly is to construct a
topological representation of posets from data, and then use the poset
representation to build a conventional DAG causal model. We illustrate our
framework by showing how it unifies disparate algorithms and case studies
proposed previously. Topology plays two key roles in causal discovery. First,
topological separability constraints on datasets have been used in several
previous approaches to infer causal structure from observations and
interventions. Second, a diverse range ofgraphical models used to represent
causal structures can be represented in a unified way in terms of a topological
representation of the induced poset structure. We show that the homotopy theory
of Alexandroff spaces can be exploited to significantly efficiently reduce the
number of possible DAG structures, reducing the search space by several orders
of magnitude.

    

### [[2112.01894] The Catalan Language CLUB](http://arxiv.org/abs/2112.01894)


  The Catalan Language Understanding Benchmark (CLUB) encompasses various
datasets representative of different NLU tasks that enable accurate evaluations
of language models, following the General Language Understanding Evaluation
(GLUE) example. It is part of AINA and PlanTL, two public funding initiatives
to empower the Catalan language in the Artificial Intelligence era.

    

### [[2112.01905] Towards Super-Resolution CEST MRI for Visualization of Small Structures](http://arxiv.org/abs/2112.01905)


  The onset of rheumatic diseases such as rheumatoid arthritis is typically
subclinical, which results in challenging early detection of the disease.
However, characteristic changes in the anatomy can be detected using imaging
techniques such as MRI or CT. Modern imaging techniques such as chemical
exchange saturation transfer (CEST) MRI drive the hope to improve early
detection even further through the imaging of metabolites in the body. To image
small structures in the joints of patients, typically one of the first regions
where changes due to the disease occur, a high resolution for the CEST MR
imaging is necessary. Currently, however, CEST MR suffers from an inherently
low resolution due to the underlying physical constraints of the acquisition.
In this work we compared established up-sampling techniques to neural
network-based super-resolution approaches. We could show, that neural networks
are able to learn the mapping from low-resolution to high-resolution
unsaturated CEST images considerably better than present methods. On the test
set a PSNR of 32.29dB (+10%), a NRMSE of 0.14 (+28%), and a SSIM of 0.85 (+15%)
could be achieved using a ResNet neural network, improving the baseline
considerably. This work paves the way for the prospective investigation of
neural networks for super-resolution CEST MRI and, followingly, might lead to a
earlier detection of the onset of rheumatic diseases.

    

### [[2112.01991] A network analysis of decision strategies of human experts in steel manufacturing](http://arxiv.org/abs/2112.01991)


  Steel production scheduling is typically accomplished by human expert
planners. Hence, instead of fully automated scheduling systems steel
manufacturers prefer auxiliary recommendation algorithms. Through the
suggestion of suitable orders, these algorithms assist human expert planners
who are tasked with the selection and scheduling of production orders. However,
it is hard to estimate, what degree of complexity these algorithms should have
as steel campaign planning lacks precise rule-based procedures; in fact, it
requires extensive domain knowledge as well as intuition that can only be
acquired by years of business experience. Here, instead of developing new
algorithms or improving older ones, we introduce a shuffling-aided network
method to assess the complexity of the selection patterns established by a
human expert. This technique allows us to formalize and represent the tacit
knowledge that enters the campaign planning. As a result of the network
analysis, we have discovered that the choice of production orders is primarily
determined by the orders' carbon content. Surprisingly, trace elements like
manganese, silicon, and titanium have a lesser impact on the selection decision
than assumed by the pertinent literature. Our approach can serve as an input to
a range of decision-support systems, whenever a human expert needs to create
groups of orders ('campaigns') that fulfill certain implicit selection
criteria.

    

### [[2112.02034] Could AI Democratise Education? Socio-Technical Imaginaries of an EdTech Revolution](http://arxiv.org/abs/2112.02034)


  Artificial Intelligence (AI) in Education has been said to have the potential
for building more personalised curricula, as well as democratising education
worldwide and creating a Renaissance of new ways of teaching and learning.
Millions of students are already starting to benefit from the use of these
technologies, but millions more around the world are not. If this trend
continues, the first delivery of AI in Education could be greater educational
inequality, along with a global misallocation of educational resources
motivated by the current technological determinism narrative. In this paper, we
focus on speculating and posing questions around the future of AI in Education,
with the aim of starting the pressing conversation that would set the right
foundations for the new generation of education that is permeated by
technology. This paper starts by synthesising how AI might change how we learn
and teach, focusing specifically on the case of personalised learning
companions, and then move to discuss some socio-technical features that will be
crucial for avoiding the perils of these AI systems worldwide (and perhaps
ensuring their success). This paper also discusses the potential of using AI
together with free, participatory and democratic resources, such as Wikipedia,
Open Educational Resources and open-source tools. We also emphasise the need
for collectively designing human-centered, transparent, interactive and
collaborative AI-based algorithms that empower and give complete agency to
stakeholders, as well as support new emerging pedagogies. Finally, we ask what
would it take for this educational revolution to provide egalitarian and
empowering access to education, beyond any political, cultural, language,
geographical and learning ability barriers.

    

### [[2112.02045] An Analytical Update Rule for General Policy Optimization](http://arxiv.org/abs/2112.02045)


  We present an analytical policy update rule that is independent of
parameterized function approximators. The update rule is suitable for general
stochastic policies with monotonic improvement guarantee. The update rule is
derived from a closed-form trust-region solution using calculus of variation,
following a new theoretical result that tightens existing bounds for policy
search using trust-region methods. An explanation building a connection between
the policy update rule and value-function methods is provided. Based on a
recursive form of the update rule, an off-policy algorithm is derived
naturally, and the monotonic improvement guarantee remains. Furthermore, the
update rule extends immediately to multi-agent systems when updates are
performed by one agent at a time.

    

### [[2112.02070] Malakai: Music That Adapts to the Shape of Emotions](http://arxiv.org/abs/2112.02070)


  The advent of ML music models such as Google Magenta's MusicVAE now allow us
to extract and replicate compositional features from otherwise complex
datasets. These models allow computational composers to parameterize abstract
variables such as style and mood. By leveraging these models and combining them
with procedural algorithms from the last few decades, it is possible to create
a dynamic song that composes music in real-time to accompany interactive
experiences. Malakai is a tool that helps users of varying skill levels create,
listen to, remix and share such dynamic songs. Using Malakai, a Composer can
create a dynamic song that can be interacted with by a Listener

    

### [[2112.02079] Cyberphysical Sequencing for Distributed Asset Management with Broad Traceability](http://arxiv.org/abs/2112.02079)


  Cyber-Physical systems (CPS) have complex lifecycles involving multiple
stakeholders, and the transparency of both hardware and software components'
supply chain is opaque at best. This raises concerns for stakeholders who may
not trust that what they receive is what was requested. There is an opportunity
to build a cyberphysical titling process offering universal traceability and
the ability to differentiate systems based on provenance. Today, RFID tags and
barcodes address some of these needs, though they are easily manipulated due to
non-linkage with an object or system's intrinsic characteristics. We propose
cyberphysical sequencing as a low-cost, light-weight and pervasive means of
adding track-and-trace capabilities to any asset that ties a system's physical
identity to a unique and invariant digital identifier. CPS sequencing offers
benefits similar Digital Twins' for identifying and managing the provenance and
identity of an asset throughout its life with far fewer computational and other
resources.

    

### [[2112.01913] REMR: A Reliability Evaluation Method for Dynamic Edge Computing Network under Time Constraints](http://arxiv.org/abs/2112.01913)


  While the concept of Artificial Intelligent Internet of Things\ (AIoT) is
booming, computation and/or communication-intensive tasks accompanied by
several sub-tasks are slowly moving from centralized deployment to edge-side
deployment. The idea of edge computing also makes intelligent services sink
locally. But in actual scenarios like dynamic edge computing networks (DECN),
due to fluctuations in available computing resources of intermediate servers
and changes in bandwidth during data transmission, service reliability becomes
difficult to guarantee. Coupled with changes in the amount of data in a
service, the above three problems all make the existing reliability evaluation
methods no longer accurate. To study the effect of distributed service
deployment strategies under such a background, this paper proposes a
reliability evaluation method (REMR) based on lower boundary rule under time
constraint to study the degree of the rationality of a service deployment plan
combined with DECN. In this scenario, time delay is the main concern which
would be affected by three quantitative factors: data packet storing and
sending time, data transmission time and the calculation time of executing
sub-tasks on the node devices, specially while the last two are in dynamic
scenarios. In actual calculation, based on the idea of the minimal paths, the
solution set would to be found that can meet requirements in the current
deployment. Then the reliability of the service supported by the solution sets
would be found out based on the principle of inclusion-exclusion combined with
the distribution of available data transmission bandwidth and the distribution
of node available computing resources. Besides a illustrative example was
provided, to verify the calculated reliability of the designed service
deployment plan, the NS3 is utilized along with Google cluster data set for
simulation.

    

### [[1907.12650] How to Staff when Customers Arrive in Batches](http://arxiv.org/abs/1907.12650)


  In settings as diverse as autonomous vehicles, cloud computing, and pandemic
quarantines, requests for service can arrive in near or true simultaneity with
one another. This creates batches of arrivals to the underlying queueing
system. In this paper, we study the staffing problem for the batch arrival
queue. We show that batches place a significant stress on services, and thus
require a high amount of resources and preparation. In fact, we find that there
is no economy of scale as the number of customers in each batch increases,
creating a stark contrast with the square root safety staffing rules enjoyed by
systems with solitary arrivals of customers. Furthermore, when customers arrive
both quickly and in batches, an economy of scale can exist, but it is weaker
than what is typically expected. Methodologically, these staffing results
follow from novel large batch and hybrid large-batch-and-large-rate limits of
the general multi-server queueing model. In the pure large batch limit, we
establish the first formal connection between multi-server queues and storage
processes, another family of stochastic processes. By consequence, we show that
the limit of the batch scaled queue length process is not asymptotically
normal, and that, in fact, the fluid and diffusion-type limits coincide. This
is what drives our staffing analysis of the batch arrival queue, and what
implies that the (safety) staffing of this system must be directly proportional
to the batch size just to achieve a non-degenerate probability of customers
waiting.

    

### [[2112.01593] Types and Terms Translated: Unrestricted Resources in Encoding Functions as Processes (Extended Version)](http://arxiv.org/abs/2112.01593)


  Type-preserving translations are effective rigorous tools in the study of
core programming calculi. In this paper, we develop a new typed translation
that connects sequential and concurrent calculi; it is governed by expressive
type systems that control resource consumption. Our main contribution is the
source language, a new resource \lambda-calculus with non-determinism and
failures, dubbed \ulamf. In \ulamf, resources are sharply separated into linear
and unrestricted; failures are explicit and arise following this separation. We
equip \ulamf with a type system based on non-idempotent intersection types,
which controls resources and fail-prone computation. The target language is an
existing session-typed \pi-calculus, \spi, which results from a Curry-Howard
correspondence between linear logic and session types for concurrency. Our
typed translation of \ulamf into \spi subsumes our prior work; interestingly,
it elegantly treats unrestricted resources in \lamrfailunres as client-server
session behaviors in \spi.

    

### [<title>Speed of random forests in XGBoost on multiclass classification - XGBoost</title>](https://discuss.xgboost.ai/t/speed-of-random-forests-in-xgboost-on-multiclass-classification/2577/1)

### [<title>Getting into way too much detail with the Z80 netlist simulation</title>](https://floooh.github.io/2021/12/06/z80-instruction-timing.html)

### [<title>I installed XGBoost using "conda install -c anaconda py-xgboost", it is missing OpenMP runtime, but I installed that too! What's going on? - XGBoost</title>](https://discuss.xgboost.ai/t/i-installed-xgboost-using-conda-install-c-anaconda-py-xgboost-it-is-missing-openmp-runtime-but-i-installed-that-too-whats-going-on/2212/11)

### [<title>DNS doesn't "propagate"</title>](https://jvns.ca/blog/2021/12/06/dns-doesn-t-propagate/)