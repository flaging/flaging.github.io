
## 2021-8-25

### [<title>Can't turn off the validation messages during training - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/cant-turn-off-the-validation-messages-during-training/2445/2)

### [<title>XGBClassifier un-pickling fails if scikit-learn is not istalled - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/xgbclassifier-un-pickling-fails-if-scikit-learn-is-not-istalled/1520/7)

### [<title>XGBClassifier un-pickling fails if scikit-learn is not istalled - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/xgbclassifier-un-pickling-fails-if-scikit-learn-is-not-istalled/1520/6)

### [[2108.10319] Enhancing Security in VANETs with Efficient Sybil Attack Detection using Fog Computing](http://arxiv.org/abs/2108.10319)


  Vehicular ad hoc networks (VANETs) facilitate vehicles to broadcast beacon
messages to ensure road safety. Rogue nodes in VANETs cause a Sybil attack to
create an illusion of fake traffic congestion by broadcasting malicious
information leading to catastrophic consequences, such as the collision of
vehicles. Previous researchers used either cryptography, trust scores, or past
vehicle data to detect rogue nodes, but they suffer from high processing delay,
overhead, and false-positive rate (FPR). We propose a fog computing-based Sybil
attack detection for VANETs (FSDV), which utilizes onboard units (OBUs) of all
the vehicles in the region to create a dynamic fog for rogue nodes detection.
We aim to reduce the data processing delays, overhead, and FPR in detecting
rogue nodes causing Sybil attacks at high vehicle densities. The performance of
our framework was carried out with simulations using OMNET++ and SUMO
simulators. Results show that our framework ensures 43% lower processing
delays, 13% lower overhead, and 35% lower FPR at high vehicle densities
compared to existing Sybil attack detection schemes.

    

### [[2108.10505] Modulating Intelligent Surfaces for Multi-User MIMO Systems: Beamforming and Modulation Design](http://arxiv.org/abs/2108.10505)


  This paper introduces a novel approach of utilizing the reconfigurable
intelligent surface (RIS) for joint data modulation and signal beamforming in a
multi-user downlink cellular network by leveraging the idea of backscatter
communication. We present a general framework in which the RIS, referred to as
modulating intelligent surface (MIS) in this paper, is used to: i) beamform the
signals for a set of users whose data modulation is already performed by the
base station (BS), and at the same time, ii) embed the data of a different set
of users by passively modulating the deliberately sent carrier signals from the
BS to the RIS. To maximize each user's spectral efficiency, a joint non-convex
optimization problem is formulated under the sum minimum mean-square error
(MMSE) criterion. Alternating optimization is used to divide the original joint
problem into two tasks of: i) separately optimizing the MIS phase-shifts for
passive beamforming along with data embedding for the BS- and MIS-served users,
respectively, and ii) jointly optimizing the active precoder and the receive
scaling factor for the BS- and MIS-served users, respectively. While the
solution to the latter joint problem is found in closed-form using traditional
optimization techniques, the optimal phase-shifts at the MIS are obtained by
deriving the appropriate optimization-oriented vector approximate message
passing (OOVAMP) algorithm. Moreover, the original joint problem is solved
under both ideal and practical constraints on the MIS phase shifts, namely, the
unimodular constraint and assuming each MIS element to be terminated by a
variable reactive load. The proposed MIS-assisted scheme is compared against
state-of-the-art RIS-assisted wireless communication schemes and simulation
results reveal that it brings substantial improvements in terms of system
throughput while supporting a much higher number of users.

    

### [[2108.10572] Optimal UAV Hitching on Ground Vehicles](http://arxiv.org/abs/2108.10572)


  Due to its mobility and agility, unmanned aerial vehicle (UAV) has emerged as
a promising technology for various tasks, such as sensing, inspection and
delivery. However, a typical UAV has limited energy storage and cannot fly a
long distance without being recharged. This motivates several existing
proposals to use trucks and other ground vehicles to offer riding to help UAVs
save energy and expand the operation radius. We present the first theoretical
study regarding how UAVs should optimally hitch on ground vehicles, considering
vehicles' different travelling patterns and supporting capabilities. For a
single UAV, we derive closed-form optimal vehicle selection and hitching
strategy. When vehicles only support hitching, a UAV would prefer the vehicle
that can carry it closest to its final destination. When vehicles can offer
hitching plus charging, the UAV may hitch on a vehicle that carries it farther
away from its destination and hitch a longer distance. The UAV may also prefer
to hitch on a slower vehicle for the benefit of battery recharging. For
multiple UAVs in need of hitching, we develop the max-saving algorithm (MSA) to
optimally match UAV-vehicle collaboration. We prove that the MSA globally
optimizes the total hitching benefits for the UAVs.

    

### [[2108.10613] Outdoor Position Recovery from HeterogeneousTelco Cellular Data](http://arxiv.org/abs/2108.10613)


  Recent years have witnessed unprecedented amounts of data generated by
telecommunication (Telco) cellular networks. For example, measurement records
(MRs) are generated to report the connection states between mobile devices and
Telco networks, e.g., received signal strength. MR data have been widely used
to localize outdoor mobile devices for human mobility analysis, urban planning,
and traffic forecasting. Existing works using first-order sequence models such
as the Hidden Markov Model (HMM) attempt to capture spatio-temporal locality in
underlying mobility patterns for lower localization errors. The HMM approaches
typically assume stable mobility patterns of the underlying mobile devices. Yet
real MR datasets exhibit heterogeneous mobility patterns due to mixed
transportation modes of the underlying mobile devices and uneven distribution
of the positions associated with MR samples. Thus, the existing solutions
cannot handle these heterogeneous mobility patterns. we propose a multi-task
learning-based deep neural network (DNN) framework, namely PRNet+, to
incorporate outdoor position recovery and transportation mode detection. To
make sure the framework work, PRNet+ develops a feature extraction module to
precisely learn local-, short- and long-term spatio-temporal locality from
heterogeneous MR samples. Extensive evaluation on eight datasets collected at
three representative areas in Shanghai indicates that PRNet+ greatly
outperforms state-of-the-arts.

    

### [[2108.10615] Multi-UAV Assisted Data Gathering in WSN: A MILP Approach For Optimizing Network Lifetime](http://arxiv.org/abs/2108.10615)


  In this paper, we study the problem of gathering data from large-scale
wireless sensor networks using multiple unmanned air vehicles (UAVs) to gather
data at designated rendezvouses, where the goal is to maximize the network
lifetime. Previous proposals often consider a practical approach where the
problem of determining a data gathering scheme is decomposed into 2
sub-problems: i) partitioning the networks into clusters for determining the
rendezvouses as these obtained cluster heads; and ii) determining the paths for
a set of a given number of UAVs to come gathering data at these rendezvouses
which have been harvesting data within each local clusters, respectively. We
try to deal with this as a whole optimization problem, expecting a significant
increase in computation complexity which would bring new challenge in creating
practical solutions for large-scale WSNs. We introduce two alternatives
mixed-integer linear programming (MILP) formulations, namely the 2-index model
with $O(n^2)$ variables and the 3-index model that has $O(n^3)$ variables,
where $n$ denotes the number of sensor nodes. We show that our best model could
solve optimally the problem instances with up to 50 sensor nodes in less than
30 minutes. Next, we propose a heuristic idea to reduce the number of variables
in implementing the 3-index model to effectively handle larger-scale networks
with size in hundreds. The experiment results show that our heuristic approach
significantly prolongs the network lifetime compared to existing most efficient
proposals.

    

### [[2108.10651] Context-aware Telco Outdoor Localization](http://arxiv.org/abs/2108.10651)


  Recent years have witnessed the fast growth in telecommunication (Telco)
techniques from 2G to upcoming 5G. Precise outdoor localization is important
for Telco operators to manage, operate and optimize Telco networks. Differing
from GPS, Telco localization is a technique employed by Telco operators to
localize outdoor mobile devices by using measurement report (MR) data. When
given MR samples containing noisy signals (e.g., caused by Telco signal
interference and attenuation), Telco localization often suffers from high
errors. To this end, the main focus of this paper is how to improve Telco
localization accuracy via the algorithms to detect and repair outlier positions
with high errors. Specifically, we propose a context-aware Telco localization
technique, namely RLoc, which consists of three main components: a
machine-learning-based localization algorithm, a detection algorithm to find
flawed samples, and a repair algorithm to replace outlier localization results
by better ones (ideally ground truth positions). Unlike most existing works to
detect and repair every flawed MR sample independently, we instead take into
account spatio-temporal locality of MR locations and exploit trajectory context
to detect and repair flawed positions. Our experiments on the real MR data sets
from 2G GSM and 4G LTE Telco networks verify that our work RLoc can greatly
improve Telco location accuracy. For example, RLoc on a large 4G MR data set
can achieve 32.2 meters of median errors, around 17.4% better than
state-of-the-art.

    

### [[2004.10715] Redefining Wireless Communication for 6G: Signal Processing Meets Deep Learning with Deep Unfolding](http://arxiv.org/abs/2004.10715)


  The year 2019 witnessed the rollout of the 5G standard, which promises to
offer significant data rate improvement over 4G. While 5G is still in its
infancy, there has been an increased shift in the research community for
communication technologies beyond 5G. The recent emergence of machine learning
approaches for enhancing wireless communications and empowering them with
much-desired intelligence holds immense potential for redefining wireless
communication for 6G. The evolving communication systems will be bottlenecked
in terms of latency, throughput, and reliability by the underlying signal
processing at the physical layer. In this position paper, we motivate the need
to redesign iterative signal processing algorithms by leveraging deep unfolding
techniques to fulfill the physical layer requirements for 6G networks. To this
end, we begin by presenting the service requirements and the key challenges
posed by the envisioned 6G communication architecture. We outline the
deficiencies of the traditional algorithmic principles and data-hungry deep
learning (DL) approaches in the context of 6G networks. Specifically, deep
unfolded signal processing is presented by sketching the interplay between
domain knowledge and DL. The deep unfolded approaches reviewed in this article
are positioned explicitly in the context of the requirements imposed by the
next generation of cellular networks. Finally, this article motivates open
research challenges to truly realize hardware-efficient edge intelligence for
future 6G networks.

    

### [[2108.10335] edge-SR: Super-Resolution For The Masses](http://arxiv.org/abs/2108.10335)


  Classic image scaling (e.g. bicubic) can be seen as one convolutional layer
and a single upscaling filter. Its implementation is ubiquitous in all display
devices and image processing software. In the last decade deep learning systems
have been introduced for the task of image super-resolution (SR), using several
convolutional layers and numerous filters. These methods have taken over the
benchmarks of image quality for upscaling tasks. Would it be possible to
replace classic upscalers with deep learning architectures on edge devices such
as display panels, tablets, laptop computers, etc.? On one hand, the current
trend in Edge-AI chips shows a promising future in this direction, with rapid
development of hardware that can run deep-learning tasks efficiently. On the
other hand, in image SR only few architectures have pushed the limit to extreme
small sizes that can actually run on edge devices at real-time. We explore
possible solutions to this problem with the aim to fill the gap between classic
upscalers and small deep learning configurations. As a transition from classic
to deep-learning upscaling we propose edge-SR (eSR), a set of one-layer
architectures that use interpretable mechanisms to upscale images. Certainly, a
one-layer architecture cannot reach the quality of deep learning systems.
Nevertheless, we find that for high speed requirements, eSR becomes better at
trading-off image quality and runtime performance. Filling the gap between
classic and deep-learning architectures for image upscaling is critical for
massive adoption of this technology. It is equally important to have an
interpretable system that can reveal the inner strategies to solve this problem
and guide us to future improvements and better understanding of larger
networks.

    

### [[2108.10346] Explaining Bayesian Neural Networks](http://arxiv.org/abs/2108.10346)


  To make advanced learning machines such as Deep Neural Networks (DNNs) more
transparent in decision making, explainable AI (XAI) aims to provide
interpretations of DNNs' predictions. These interpretations are usually given
in the form of heatmaps, each one illustrating relevant patterns regarding the
prediction for a given instance. Bayesian approaches such as Bayesian Neural
Networks (BNNs) so far have a limited form of transparency (model transparency)
already built-in through their prior weight distribution, but notably, they
lack explanations of their predictions for given instances. In this work, we
bring together these two perspectives of transparency into a holistic
explanation framework for explaining BNNs. Within the Bayesian framework, the
network weights follow a probability distribution. Hence, the standard
(deterministic) prediction strategy of DNNs extends in BNNs to a predictive
distribution, and thus the standard explanation extends to an explanation
distribution. Exploiting this view, we uncover that BNNs implicitly employ
multiple heterogeneous prediction strategies. While some of these are inherited
from standard DNNs, others are revealed to us by considering the inherent
uncertainty in BNNs. Our quantitative and qualitative experiments on
toy/benchmark data and real-world data from pathology show that the proposed
approach of explaining BNNs can lead to more effective and insightful
explanations.

    

### [[2108.10352] Model-Free Learning of Optimal Deterministic Resource Allocations in Wireless Systems via Action-Space Exploration](http://arxiv.org/abs/2108.10352)


  Wireless systems resource allocation refers to perpetual and challenging
nonconvex constrained optimization tasks, which are especially timely in modern
communications and networking setups involving multiple users with
heterogeneous objectives and imprecise or even unknown models and/or channel
statistics. In this paper, we propose a technically grounded and scalable
primal-dual deterministic policy gradient method for efficiently learning
optimal parameterized resource allocation policies. Our method not only
efficiently exploits gradient availability of popular universal policy
representations, such as deep neural networks, but is also truly model-free, as
it relies on consistent zeroth-order gradient approximations of the associated
random network services constructed via low-dimensional perturbations in action
space, thus fully bypassing any dependence on critics. Both theory and
numerical simulations confirm the efficacy and applicability of the proposed
approach, as well as its superiority over the current state of the art in terms
of both achieving near-optimal performance and scalability.

    

### [[2108.10365] L1-regularized neural ranking for risk stratification and its application to prediction of time to distant metastasis in luminal node negative chemotherapy naïve breast cancer patients](http://arxiv.org/abs/2108.10365)


  Can we predict if an early stage cancer patient is at high risk of developing
distant metastasis and what clinicopathological factors are associated with
such a risk? In this paper, we propose a ranking based censoring-aware machine
learning model for answering such questions. The proposed model is able to
generate an interpretable formula for risk stratifi-cation using a minimal
number of clinicopathological covariates through L1-regulrization. Using this
approach, we analyze the association of time to distant metastasis (TTDM) with
various clinical parameters for early stage, luminal (ER+ or HER2-) breast
cancer patients who received endocrine therapy but no chemotherapy (n = 728).
The TTDM risk stratification formula obtained using the proposed approach is
primarily based on mitotic score, histolog-ical tumor type and lymphovascular
invasion. These findings corroborate with the known role of these covariates in
increased risk for distant metastasis. Our analysis shows that the proposed
risk stratification formula can discriminate between cases with high and low
risk of distant metastasis (p-value < 0.005) and can also rank cases based on
their time to distant metastasis with a concordance-index of 0.73.

    

### [[2108.10367] Marine vessel tracking using a monocular camera](http://arxiv.org/abs/2108.10367)


  In this paper, a new technique for camera calibration using only GPS data is
presented. A new way of tracking objects that move on a plane in a video is
achieved by using the location and size of the bounding box to estimate the
distance, achieving an average prediction error of 5.55m per 100m distance from
the camera. This solution can be run in real-time at the edge, achieving
efficient inference in a low-powered IoT environment while also being able to
track multiple different vessels.

    

### [[2108.10382] Learning Sparse Analytic Filters for Piano Transcription](http://arxiv.org/abs/2108.10382)


  In recent years, filterbank learning has become an increasingly popular
strategy for various audio-related machine learning tasks. This is partly due
to its ability to discover task-specific audio characteristics which can be
leveraged in downstream processing. It is also a natural extension of the
nearly ubiquitous deep learning methods employed to tackle a diverse array of
audio applications. In this work, several variations of a frontend filterbank
learning module are investigated for piano transcription, a challenging
low-level music information retrieval task. We build upon a standard piano
transcription model, modifying only the feature extraction stage. The
filterbank module is designed such that its complex filters are unconstrained
1D convolutional kernels with long receptive fields. Additional variations
employ the Hilbert transform to render the filters intrinsically analytic and
apply variational dropout to promote filterbank sparsity. Transcription results
are compared across all experiments, and we offer visualization and analysis of
the filterbanks.

    

### [[2108.10395] Using Neighborhood Context to Improve Information Extraction from Visual Documents Captured on Mobile Phones](http://arxiv.org/abs/2108.10395)


  Information Extraction from visual documents enables convenient and
intelligent assistance to end users. We present a Neighborhood-based
Information Extraction (NIE) approach that uses contextual language models and
pays attention to the local neighborhood context in the visual documents to
improve information extraction accuracy. We collect two different visual
document datasets and show that our approach outperforms the state-of-the-art
global context-based IE technique. In fact, NIE outperforms existing approaches
in both small and large model sizes. Our on-device implementation of NIE on a
mobile platform that generally requires small models showcases NIE's usefulness
in practical real-world applications.

    

### [[2108.10397] Predicting Vehicles' Longitudinal Trajectories and Lane Changes on Highway On-Ramps](http://arxiv.org/abs/2108.10397)


  Vehicles on highway on-ramps are one of the leading contributors to
congestion. In this paper, we propose a prediction framework that predicts the
longitudinal trajectories and lane changes (LCs) of vehicles on highway
on-ramps and tapers. Specifically, our framework adopts a combination of
prediction models that inputs a 4 seconds duration of a trajectory to output a
forecast of the longitudinal trajectories and LCs up to 15 seconds ahead.
Training and Validation based on next generation simulation (NGSIM) data show
that the prediction power of the developed model and its accuracy outperforms a
traditional long-short term memory (LSTM) model. Ultimately, the work presented
here can alleviate the congestion experienced on on-ramps, improve safety, and
guide effective traffic control strategies.

    

### [[2108.10403] Robust Risk-Aware Reinforcement Learning](http://arxiv.org/abs/2108.10403)


  We present a reinforcement learning (RL) approach for robust optimisation of
risk-aware performance criteria. To allow agents to express a wide variety of
risk-reward profiles, we assess the value of a policy using rank dependent
expected utility (RDEU). RDEU allows the agent to seek gains, while
simultaneously protecting themselves against downside events. To robustify
optimal policies against model uncertainty, we assess a policy not by its
distribution, but rather, by the worst possible distribution that lies within a
Wasserstein ball around it. Thus, our problem formulation may be viewed as an
actor choosing a policy (the outer problem), and the adversary then acting to
worsen the performance of that strategy (the inner problem). We develop
explicit policy gradient formulae for the inner and outer problems, and show
its efficacy on three prototypical financial problems: robust portfolio
allocation, optimising a benchmark, and statistical arbitrage

    

### [[2108.10411] SreaMRAK a Streaming Multi-Resolution Adaptive Kernel Algorithm](http://arxiv.org/abs/2108.10411)


  Kernel ridge regression (KRR) is a popular scheme for non-linear
non-parametric learning. However, existing implementations of KRR require that
all the data is stored in the main memory, which severely limits the use of KRR
in contexts where data size far exceeds the memory size. Such applications are
increasingly common in data mining, bioinformatics, and control. A powerful
paradigm for computing on data sets that are too large for memory is the
streaming model of computation, where we process one data sample at a time,
discarding each sample before moving on to the next one.
In this paper, we propose StreaMRAK - a streaming version of KRR. StreaMRAK
improves on existing KRR schemes by dividing the problem into several levels of
resolution, which allows continual refinement to the predictions. The algorithm
reduces the memory requirement by continuously and efficiently integrating new
samples into the training model. With a novel sub-sampling scheme, StreaMRAK
reduces memory and computational complexities by creating a sketch of the
original data, where the sub-sampling density is adapted to the bandwidth of
the kernel and the local dimensionality of the data.
We present a showcase study on two synthetic problems and the prediction of
the trajectory of a double pendulum. The results show that the proposed
algorithm is fast and accurate.

    

### [[2108.10417] Recurrent multiple shared layers in Depth for Neural Machine Translation](http://arxiv.org/abs/2108.10417)


  Learning deeper models is usually a simple and effective approach to improve
model performance, but deeper models have larger model parameters and are more
difficult to train. To get a deeper model, simply stacking more layers of the
model seems to work well, but previous works have claimed that it cannot
benefit the model. We propose to train a deeper model with recurrent mechanism,
which loops the encoder and decoder blocks of Transformer in the depth
direction. To address the increasing of model parameters, we choose to share
parameters in different recursive moments. We conduct our experiments on WMT16
English-to-German and WMT14 English-to-France translation tasks, our model
outperforms the shallow Transformer-Base/Big baseline by 0.35, 1.45 BLEU
points, which is 27.23% of Transformer-Big model parameters. Compared to the
deep Transformer(20-layer encoder, 6-layer decoder), our model has similar
model performance and infer speed, but our model parameters are 54.72% of the
former.

    

### [[2108.10420] Jointly Learnable Data Augmentations for Self-Supervised GNNs](http://arxiv.org/abs/2108.10420)


  Self-supervised Learning (SSL) aims at learning representations of objects
without relying on manual labeling. Recently, a number of SSL methods for graph
representation learning have achieved performance comparable to SOTA
semi-supervised GNNs. A Siamese network, which relies on data augmentation, is
the popular architecture used in these methods. However, these methods rely on
heuristically crafted data augmentation techniques. Furthermore, they use
either contrastive terms or other tricks (e.g., asymmetry) to avoid trivial
solutions that can occur in Siamese networks. In this study, we propose,
GraphSurgeon, a novel SSL method for GNNs with the following features. First,
instead of heuristics we propose a learnable data augmentation method that is
jointly learned with the embeddings by leveraging the inherent signal encoded
in the graph. In addition, we take advantage of the flexibility of the
learnable data augmentation and introduce a new strategy that augments in the
embedding space, called post augmentation. This strategy has a significantly
lower memory overhead and run-time cost. Second, as it is difficult to sample
truly contrastive terms, we avoid explicit negative sampling. Third, instead of
relying on engineering tricks, we use a scalable constrained optimization
objective motivated by Laplacian Eigenmaps to avoid trivial solutions. To
validate the practical use of GraphSurgeon, we perform empirical evaluation
using 14 public datasets across a number of domains and ranging from small to
large scale graphs with hundreds of millions of edges. Our finding shows that
GraphSurgeon is comparable to six SOTA semi-supervised and on par with five
SOTA self-supervised baselines in node classification tasks. The source code is
available at this https URL.

    

### [[2108.10424] Power Grid Cascading Failure Mitigation by Reinforcement Learning](http://arxiv.org/abs/2108.10424)


  This paper proposes a cascading failure mitigation strategy based on
Reinforcement Learning (RL). The motivation of the Multi-Stage Cascading
Failure (MSCF) problem and its connection with the challenge of climate change
are introduced. The bottom-level corrective control of the MCSF problem is
formulated based on DCOPF (Direct Current Optimal Power Flow). Then, to
mitigate the MSCF issue by a high-level RL-based strategy, physics-informed
reward, action, and state are devised. Besides, both shallow and deep neural
network architectures are tested. Experiments on the IEEE 118-bus system by the
proposed mitigation strategy demonstrate a promising performance in reducing
system collapses.

    

### [[2108.10427] Graph-LDA: Graph Structure Priors to Improve the Accuracy in Few-Shot Classification](http://arxiv.org/abs/2108.10427)


  It is very common to face classification problems where the number of
available labeled samples is small compared to their dimension. These
conditions are likely to cause underdetermined settings, with high risk of
overfitting. To improve the generalization ability of trained classifiers,
common solutions include using priors about the data distribution. Among many
options, data structure priors, often represented through graphs, are
increasingly popular in the field. In this paper, we introduce a generic model
where observed class signals are supposed to be deteriorated with two sources
of noise, one independent of the underlying graph structure and isotropic, and
the other colored by a known graph operator. Under this model, we derive an
optimal methodology to classify such signals. Interestingly, this methodology
includes a single parameter, making it particularly suitable for cases where
available data is scarce. Using various real datasets, we showcase the ability
of the proposed model to be implemented in real world scenarios, resulting in
increased generalization accuracy compared to popular alternatives.

    

### [[2108.10434] Adaptive shot allocation for fast convergence in variational quantum algorithms](http://arxiv.org/abs/2108.10434)


  Variational Quantum Algorithms (VQAs) are a promising approach for practical
applications like chemistry and materials science on near-term quantum
computers as they typically reduce quantum resource requirements. However, in
order to implement VQAs, an efficient classical optimization strategy is
required. Here we present a new stochastic gradient descent method using an
adaptive number of shots at each step, called the global Coupled Adaptive
Number of Shots (gCANS) method, which improves on prior art in both the number
of iterations as well as the number of shots required. These improvements
reduce both the time and money required to run VQAs on current cloud platforms.
We analytically prove that in a convex setting gCANS achieves geometric
convergence to the optimum. Further, we numerically investigate the performance
of gCANS on some chemical configuration problems. We also consider finding the
ground state for an Ising model with different numbers of spins to examine the
scaling of the method. We find that for these problems, gCANS compares
favorably to all of the other optimizers we consider.

    

### [[2108.10447] One TTS Alignment To Rule Them All](http://arxiv.org/abs/2108.10447)


  Speech-to-text alignment is a critical component of neural textto-speech
(TTS) models. Autoregressive TTS models typically use an attention mechanism to
learn these alignments on-line. However, these alignments tend to be brittle
and often fail to generalize to long utterances and out-of-domain text, leading
to missing or repeating words. Most non-autoregressive endto-end TTS models
rely on durations extracted from external sources. In this paper we leverage
the alignment mechanism proposed in RAD-TTS as a generic alignment learning
framework, easily applicable to a variety of neural TTS models. The framework
combines forward-sum algorithm, the Viterbi algorithm, and a simple and
efficient static prior. In our experiments, the alignment learning framework
improves all tested TTS architectures, both autoregressive (Flowtron, Tacotron
2) and non-autoregressive (FastPitch, FastSpeech 2, RAD-TTS). Specifically, it
improves alignment convergence speed of existing attention-based mechanisms,
simplifies the training pipeline, and makes the models more robust to errors on
long utterances. Most importantly, the framework improves the perceived speech
synthesis quality, as judged by human evaluators.

    

### [[2108.10448] Fast Robust Tensor Principal Component Analysis via Fiber CUR Decomposition](http://arxiv.org/abs/2108.10448)


  We study the problem of tensor robust principal component analysis (TRPCA),
which aims to separate an underlying low-multilinear-rank tensor and a sparse
outlier tensor from their sum. In this work, we propose a fast non-convex
algorithm, coined Robust Tensor CUR (RTCUR), for large-scale TRPCA problems.
RTCUR considers a framework of alternating projections and utilizes the
recently developed tensor Fiber CUR decomposition to dramatically lower the
computational complexity. The performance advantage of RTCUR is empirically
verified against the state-of-the-arts on the synthetic datasets and is further
demonstrated on the real-world application such as color video background
subtraction.

    

### [[2108.10451] Adversarial Robustness of Deep Learning: Theory, Algorithms, and Applications](http://arxiv.org/abs/2108.10451)


  This tutorial aims to introduce the fundamentals of adversarial robustness of
deep learning, presenting a well-structured review of up-to-date techniques to
assess the vulnerability of various types of deep learning models to
adversarial examples. This tutorial will particularly highlight
state-of-the-art techniques in adversarial attacks and robustness verification
of deep neural networks (DNNs). We will also introduce some effective
countermeasures to improve the robustness of deep learning models, with a
particular focus on adversarial training. We aim to provide a comprehensive
overall picture about this emerging direction and enable the community to be
aware of the urgency and importance of designing robust deep learning models in
safety-critical data analytical applications, ultimately enabling the end-users
to trust deep learning classifiers. We will also summarize potential research
directions concerning the adversarial robustness of deep learning, and its
potential benefits to enable accountable and trustworthy deep learning-based
data analytical systems and applications.

    

### [[2108.10453] Stochastic Treatment Recommendation with Deep Survival Dose Response Function (DeepSDRF)](http://arxiv.org/abs/2108.10453)


  We propose a general formulation for stochastic treatment recommendation
problems in settings with clinical survival data, which we call the Deep
Survival Dose Response Function (DeepSDRF). That is, we consider the problem of
learning the conditional average dose response (CADR) function solely from
historical data in which unobserved factors (confounders) affect both observed
treatment and time-to-event outcomes. The estimated treatment effect from
DeepSDRF enables us to develop recommender algorithms with explanatory
insights. We compared two recommender approaches based on random search and
reinforcement learning and found similar performance in terms of patient
outcome. We tested the DeepSDRF and the corresponding recommender on extensive
simulation studies and two empirical databases: 1) the Clinical Practice
Research Datalink (CPRD) and 2) the eICU Research Institute (eRI) database. To
the best of our knowledge, this is the first time that confounders are taken
into consideration for addressing the stochastic treatment effect with
observational data in a medical context.

    

### [[2108.10470] Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](http://arxiv.org/abs/2108.10470)


  Isaac Gym offers a high performance learning platform to train policies for
wide variety of robotics tasks directly on GPU. Both physics simulation and the
neural network policy training reside on GPU and communicate by directly
passing data from physics buffers to PyTorch tensors without ever going through
any CPU bottlenecks. This leads to blazing fast training times for complex
robotics tasks on a single GPU with 1-2 orders of magnitude improvements
compared to conventional RL training that uses a CPU based simulator and GPU
for neural networks. We host the results and videos at
\url{this https URL} and isaac gym can be
download at \url{this https URL}.

    

### [[2108.10504] Deep Signature FBSDE Algorithm](http://arxiv.org/abs/2108.10504)


  We propose a deep signature/log-signature FBSDE algorithm to solve
forward-backward stochastic differential equations (FBSDEs) with state and path
dependent features. By incorporating the deep signature/log-signature
transformation into the recurrent neural network (RNN) model, our algorithm
shortens the training time, improves the accuracy, and extends the time horizon
comparing to methods in the existing literature. Moreover, our algorithms can
be applied to a wide range of applications such as state and path dependent
option pricing involving high-frequency data, model ambiguity, and stochastic
games, which are linked to parabolic partial differential equations (PDEs), and
path-dependent PDEs (PPDEs). Lastly, we also derive the convergence analysis of
the deep signature/log-signature FBSDE algorithm.

    

### [[2108.10513] Maximum Likelihood Estimation for Multimodal Learning with Missing Modality](http://arxiv.org/abs/2108.10513)


  Multimodal learning has achieved great successes in many scenarios. Compared
with unimodal learning, it can effectively combine the information from
different modalities to improve the performance of learning tasks. In reality,
the multimodal data may have missing modalities due to various reasons, such as
sensor failure and data transmission error. In previous works, the information
of the modality-missing data has not been well exploited. To address this
problem, we propose an efficient approach based on maximum likelihood
estimation to incorporate the knowledge in the modality-missing data.
Specifically, we design a likelihood function to characterize the conditional
distribution of the modality-complete data and the modality-missing data, which
is theoretically optimal. Moreover, we develop a generalized form of the
softmax function to effectively implement maximum likelihood estimation in an
end-to-end manner. Such training strategy guarantees the computability of our
algorithm capably. Finally, we conduct a series of experiments on real-world
multimodal datasets. Our results demonstrate the effectiveness of the proposed
approach, even when 95% of the training data has missing modality.

    

### [[2108.10517] Uncertainty Quantification of the 4th kind; optimal posterior accuracy-uncertainty tradeoff with the minimum enclosing ball](http://arxiv.org/abs/2108.10517)


  There are essentially three kinds of approaches to Uncertainty Quantification
(UQ): (A) robust optimization, (B) Bayesian, (C) decision theory. Although (A)
is robust, it is unfavorable with respect to accuracy and data assimilation.
(B) requires a prior, it is generally brittle and posterior estimations can be
slow. Although (C) leads to the identification of an optimal prior, its
approximation suffers from the curse of dimensionality and the notion of risk
is one that is averaged with respect to the distribution of the data. We
introduce a 4th kind which is a hybrid between (A), (B), (C), and hypothesis
testing. It can be summarized as, after observing a sample $x$, (1) defining a
likelihood region through the relative likelihood and (2) playing a minmax game
in that region to define optimal estimators and their risk. The resulting
method has several desirable properties (a) an optimal prior is identified
after measuring the data, and the notion of risk is a posterior one, (b) the
determination of the optimal estimate and its risk can be reduced to computing
the minimum enclosing ball of the image of the likelihood region under the
quantity of interest map (which is fast and not subject to the curse of
dimensionality). The method is characterized by a parameter in $ [0,1]$ acting
as an assumed lower bound on the rarity of the observed data (the relative
likelihood). When that parameter is near $1$, the method produces a posterior
distribution concentrated around a maximum likelihood estimate with tight but
low confidence UQ estimates. When that parameter is near $0$, the method
produces a maximal risk posterior distribution with high confidence UQ
estimates. In addition to navigating the accuracy-uncertainty tradeoff, the
proposed method addresses the brittleness of Bayesian inference by navigating
the robustness-accuracy tradeoff associated with data assimilation.

    

### [[2108.10521] Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study](http://arxiv.org/abs/2108.10521)


  Training deep graph neural networks (GNNs) is notoriously hard. Besides the
standard plights in training deep architectures such as vanishing gradients and
overfitting, the training of deep GNNs also uniquely suffers from
over-smoothing, information squashing, and so on, which limits their potential
power on large-scale graphs. Although numerous efforts are proposed to address
these limitations, such as various forms of skip connections, graph
normalization, and random dropping, it is difficult to disentangle the
advantages brought by a deep GNN architecture from those "tricks" necessary to
train such an architecture. Moreover, the lack of a standardized benchmark with
fair and consistent experimental settings poses an almost insurmountable
obstacle to gauging the effectiveness of new mechanisms. In view of those, we
present the first fair and reproducible benchmark dedicated to assessing the
"tricks" of training deep GNNs. We categorize existing approaches, investigate
their hyperparameter sensitivity, and unify the basic configuration.
Comprehensive evaluations are then conducted on tens of representative graph
datasets including the recent large-scale Open Graph Benchmark (OGB), with
diverse deep GNN backbones. Based on synergistic studies, we discover the combo
of superior training tricks, that lead us to attain the new state-of-the-art
results for deep GCNs, across multiple representative graph datasets. We
demonstrate that an organic combo of initial connection, identity mapping,
group and batch normalization has the most ideal performance on large datasets.
Experiments also reveal a number of "surprises" when combining or scaling up
some of the tricks. All codes are available at
this https URL.

    

### [[2108.10531] Unsupervised Depth Completion with Calibrated Backprojection Layers](http://arxiv.org/abs/2108.10531)


  We propose a deep neural network architecture to infer dense depth from an
image and a sparse point cloud. It is trained using a video stream and
corresponding synchronized sparse point cloud, as obtained from a LIDAR or
other range sensor, along with the intrinsic calibration parameters of the
camera. At inference time, the calibration of the camera, which can be
different than the one used for training, is fed as an input to the network
along with the sparse point cloud and a single image. A Calibrated
Backprojection Layer backprojects each pixel in the image to three-dimensional
space using the calibration matrix and a depth feature descriptor. The
resulting 3D positional encoding is concatenated with the image descriptor and
the previous layer output to yield the input to the next layer of the encoder.
A decoder, exploiting skip-connections, produces a dense depth map. The
resulting Calibrated Backprojection Network, or KBNet, is trained without
supervision by minimizing the photometric reprojection error. KBNet imputes
missing depth value based on the training set, rather than on generic
regularization. We test KBNet on public depth completion benchmarks, where it
outperforms the state of the art by 30% indoor and 8% outdoor when the same
camera is used for training and testing. When the test camera is different, the
improvement reaches 62%. Code available at:
this https URL.

    

### [[2108.10533] Entropy-Aware Model Initialization for Effective Exploration in Deep Reinforcement Learning](http://arxiv.org/abs/2108.10533)


  Encouraging exploration is a critical issue in deep reinforcement learning.
We investigate the effect of initial entropy that significantly influences the
exploration, especially at the earlier stage. Our main observations are as
follows: 1) low initial entropy increases the probability of learning failure,
and 2) this initial entropy is biased towards a low value that inhibits
exploration. Inspired by the investigations, we devise entropy-aware model
initialization, a simple yet powerful learning strategy for effective
exploration. We show that the devised learning strategy significantly reduces
learning failures and enhances performance, stability, and learning speed
through experiments.

    

### [[2108.10539] Counterfactual Explainable Recommendation](http://arxiv.org/abs/2108.10539)


  By providing explanations for users and system designers to facilitate better
understanding and decision making, explainable recommendation has been an
important research problem. In this paper, we propose Counterfactual
Explainable Recommendation (CountER), which takes the insights of
counterfactual reasoning from causal inference for explainable recommendation.
CountER is able to formulate the complexity and the strength of explanations,
and it adopts a counterfactual learning framework to seek simple (low
complexity) and effective (high strength) explanations for the model decision.
Technically, for each item recommended to each user, CountER formulates a joint
optimization problem to generate minimal changes on the item aspects so as to
create a counterfactual item, such that the recommendation decision on the
counterfactual item is reversed. These altered aspects constitute the
explanation of why the original item is recommended. The counterfactual
explanation helps both the users for better understanding and the system
designers for better model debugging. Another contribution of the work is the
evaluation of explainable recommendation, which has been a challenging task.
Fortunately, counterfactual explanations are very suitable for standard
quantitative evaluation. To measure the explanation quality, we design two
types of evaluation metrics, one from user's perspective (i.e. why the user
likes the item), and the other from model's perspective (i.e. why the item is
recommended by the model). We apply our counterfactual learning algorithm on a
black-box recommender system and evaluate the generated explanations on five
real-world datasets. Results show that our model generates more accurate and
effective explanations than state-of-the-art explainable recommendation models.

    

### [[2108.10550] A generative adversarial approach to facilitate archival-quality histopathologic diagnoses from frozen tissue sections](http://arxiv.org/abs/2108.10550)


  In clinical diagnostics and research involving histopathology, formalin fixed
paraffin embedded (FFPE) tissue is almost universally favored for its superb
image quality. However, tissue processing time (more than 24 hours) can slow
decision-making. In contrast, fresh frozen (FF) processing (less than 1 hour)
can yield rapid information but diagnostic accuracy is suboptimal due to lack
of clearing, morphologic deformation and more frequent artifacts. Here, we
bridge this gap using artificial intelligence. We synthesize FFPE-like images
,virtual FFPE, from FF images using a generative adversarial network (GAN) from
98 paired kidney samples derived from 40 patients. Five board-certified
pathologists evaluated the results in a blinded test. Image quality of the
virtual FFPE data was assessed to be high and showed a close resemblance to
real FFPE images. Clinical assessments of disease on the virtual FFPE images
showed a higher inter-observer agreement compared to FF images. The nearly
instantaneously generated virtual FFPE images can not only reduce time to
information but can facilitate more precise diagnosis from routine FF images
without extraneous costs and effort.

    

### [[2108.10551] Lossless Image Compression Using a Multi-Scale Progressive Statistical Model](http://arxiv.org/abs/2108.10551)


  Lossless image compression is an important technique for image storage and
transmission when information loss is not allowed. With the fast development of
deep learning techniques, deep neural networks have been used in this field to
achieve a higher compression rate. Methods based on pixel-wise autoregressive
statistical models have shown good performance. However, the sequential
processing way prevents these methods to be used in practice. Recently,
multi-scale autoregressive models have been proposed to address this
limitation. Multi-scale approaches can use parallel computing systems
efficiently and build practical systems. Nevertheless, these approaches
sacrifice compression performance in exchange for speed. In this paper, we
propose a multi-scale progressive statistical model that takes advantage of the
pixel-wise approach and the multi-scale approach. We developed a flexible
mechanism where the processing order of the pixels can be adjusted easily. Our
proposed method outperforms the state-of-the-art lossless image compression
methods on two large benchmark datasets by a significant margin without
degrading the inference speed dramatically.

    

### [[2108.10557] Adaptation-Agnostic Meta-Training](http://arxiv.org/abs/2108.10557)


  Many meta-learning algorithms can be formulated into an interleaved process,
in the sense that task-specific predictors are learned during inner-task
adaptation and meta-parameters are updated during meta-update. The normal
meta-training strategy needs to differentiate through the inner-task adaptation
procedure to optimize the meta-parameters. This leads to a constraint that the
inner-task algorithms should be solved analytically. Under this constraint,
only simple algorithms with analytical solutions can be applied as the
inner-task algorithms, limiting the model expressiveness. To lift the
limitation, we propose an adaptation-agnostic meta-training strategy. Following
our proposed strategy, we can apply stronger algorithms (e.g., an ensemble of
different types of algorithms) as the inner-task algorithm to achieve superior
performance comparing with popular baselines. The source code is available at
this https URL.

    

### [[2108.10561] Taming the Beast: Learning to Control Neural Conversational Models](http://arxiv.org/abs/2108.10561)


  This thesis investigates the controllability of deep learning-based,
end-to-end, generative dialogue systems in both task-oriented and chit-chat
scenarios. In particular, we study the different aspects of controlling
generative dialogue systems, including controlling styles and topics and
continuously adding and combining dialogue skills. In the three decades since
the first dialogue system was commercialized, the basic architecture of such
systems has remained substantially unchanged, consisting of four pipelined
basic components, namely, natural language understanding (NLU), dialogue state
tracking (DST), a dialogue manager (DM) and natural language generation (NLG).
The dialogue manager, which is the critical component of the modularized
system, controls the response content and style. This module is usually
programmed by rules and is designed to be highly controllable and easily
extendable. With the emergence of powerful "deep learning" architectures,
end-to-end generative dialogue systems have been proposed to optimize overall
system performance and simplify training. However, these systems cannot be
easily controlled and extended as the modularized dialogue manager can. This is
because a single neural system is used, which is usually a large pre-trained
language model (e.g., GPT-2), and thus it is hard to surgically change
desirable attributes (e.g., style, topics, etc.). More importantly,
uncontrollable dialogue systems can generate offensive and even toxic
responses. Therefore, in this thesis, we study controllable methods for
end-to-end generative dialogue systems in task-oriented and chit-chat
scenarios. Throughout the chapters, we describe 1) how to control the style and
topics of chit-chat models, 2) how to continuously control and extend
task-oriented dialogue systems, and 3) how to compose and control multi-skill
dialogue models.

    

### [[2108.10566] sigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification](http://arxiv.org/abs/2108.10566)


  Multiclass multilabel classification refers to the task of attributing
multiple labels to examples via predictions. Current models formulate a
reduction of that multilabel setting into either multiple binary
classifications or multiclass classification, allowing for the use of existing
loss functions (sigmoid, cross-entropy, logistic, etc.). Empirically, these
methods have been reported to achieve good performance on different metrics (F1
score, Recall, Precision, etc.). Theoretically though, the multilabel
classification reductions does not accommodate for the prediction of varying
numbers of labels per example and the underlying losses are distant estimates
of the performance metrics.
We propose a loss function, sigmoidF1. It is an approximation of the F1 score
that (I) is smooth and tractable for stochastic gradient descent, (II)
naturally approximates a multilabel metric, (III) estimates label propensities
and label counts. More generally, we show that any confusion matrix metric can
be formulated with a smooth surrogate. We evaluate the proposed loss function
on different text and image datasets, and with a variety of metrics, to account
for the complexity of multilabel classification evaluation. In our experiments,
we embed the sigmoidF1 loss in a classification head that is attached to
state-of-the-art efficient pretrained neural networks MobileNetV2 and
DistilBERT.
Our experiments show that sigmoidF1 outperforms other loss functions on four
datasets and several metrics. These results show the effectiveness of using
inference-time metrics as loss function at training time in general and their
potential on non-trivial classification problems like multilabel
classification.

    

### [[2108.10573] The staircase property: How hierarchical structure can guide deep learning](http://arxiv.org/abs/2108.10573)


  This paper identifies a structural property of data distributions that
enables deep neural networks to learn hierarchically. We define the "staircase"
property for functions over the Boolean hypercube, which posits that high-order
Fourier coefficients are reachable from lower-order Fourier coefficients along
increasing chains. We prove that functions satisfying this property can be
learned in polynomial time using layerwise stochastic coordinate descent on
regular neural networks -- a class of network architectures and initializations
that have homogeneity properties. Our analysis shows that for such staircase
functions and neural networks, the gradient-based algorithm learns high-level
features by greedily combining lower-level features along the depth of the
network. We further back our theoretical results with experiments showing that
staircase functions are also learnable by more standard ResNet architectures
with stochastic gradient descent. Both the theoretical and experimental results
support the fact that staircase properties have a role to play in understanding
the capabilities of gradient-based learning on regular networks, in contrast to
general polynomial-size networks that can emulate any SQ or PAC algorithms as
recently shown.

    

### [[2108.10579] Lossy Medical Image Compression using Residual Learning-based Dual Autoencoder Model](http://arxiv.org/abs/2108.10579)


  In this work, we propose a two-stage autoencoder based
compressor-decompressor framework for compressing malaria RBC cell image
patches. We know that the medical images used for disease diagnosis are around
multiple gigabytes size, which is quite huge. The proposed residual-based dual
autoencoder network is trained to extract the unique features which are then
used to reconstruct the original image through the decompressor module. The two
latent space representations (first for the original image and second for the
residual image) are used to rebuild the final original image. Color-SSIM has
been exclusively used to check the quality of the chrominance part of the cell
images after decompression. The empirical results indicate that the proposed
work outperformed other neural network related compression technique for
medical images by approximately 35%, 10% and 5% in PSNR, Color SSIM and MS-SSIM
respectively. The algorithm exhibits a significant improvement in bit savings
of 76%, 78%, 75% & 74% over JPEG-LS, JP2K-LM, CALIC and recent neural network
approach respectively, making it a good compression-decompression technique.

    

### [[2108.10587] Pooling Architecture Search for Graph Classification](http://arxiv.org/abs/2108.10587)


  Graph classification is an important problem with applications across many
domains, like chemistry and bioinformatics, for which graph neural networks
(GNNs) have been state-of-the-art (SOTA) methods. GNNs are designed to learn
node-level representation based on neighborhood aggregation schemes, and to
obtain graph-level representation, pooling methods are applied after the
aggregation operation in existing GNN models to generate coarse-grained graphs.
However,due to highly diverse applications of graph classification, and the
performance of existing pooling methods vary on different graphs. In other
words, it is a challenging problem to design a universal pooling architecture
to perform well in most cases, leading to a demand for data-specific pooling
methods in real-world applications. To address this problem, we propose to use
neural architecture search (NAS) to search for adaptive pooling architectures
for graph classification. Firstly we designed a unified framework consisting of
four modules: Aggregation, Pooling, Readout, and Merge, which can cover
existing human-designed pooling methods for graph classification. Based on this
framework, a novel search space is designed by incorporating popular operations
in human-designed architectures. Then to enable efficient search, a coarsening
strategy is proposed to continuously relax the search space, thus a
differentiable search method can be adopted. Extensive experiments on six
real-world datasets from three domains are conducted, and the results
demonstrate the effectiveness and efficiency of the proposed framework.

    

### [[2108.10600] DeepSleepNet-Lite: A Simplified Automatic Sleep Stage Scoring Model with Uncertainty Estimates](http://arxiv.org/abs/2108.10600)


  Deep learning is widely used in the most recent automatic sleep scoring
algorithms. Its popularity stems from its excellent performance and from its
ability to directly process raw signals and to learn feature from the data.
Most of the existing scoring algorithms exploit very computationally demanding
architectures, due to their high number of training parameters, and process
lengthy time sequences in input (up to 12 minutes). Only few of these
architectures provide an estimate of the model uncertainty. In this study we
propose DeepSleepNet-Lite, a simplified and lightweight scoring architecture,
processing only 90-seconds EEG input sequences. We exploit, for the first time
in sleep scoring, the Monte Carlo dropout technique to enhance the performance
of the architecture and to also detect the uncertain instances. The evaluation
is performed on a single-channel EEG Fpz-Cz from the open source Sleep-EDF
expanded database. DeepSleepNet-Lite achieves slightly lower performance, if
not on par, compared to the existing state-of-the-art architectures, in overall
accuracy, macro F1-score and Cohen's kappa (on Sleep-EDF v1-2013 +/-30mins:
84.0%, 78.0%, 0.78; on Sleep-EDF v2-2018 +/-30mins: 80.3%, 75.2%, 0.73). Monte
Carlo dropout enables the estimate of the uncertain predictions. By rejecting
the uncertain instances, the model achieves higher performance on both versions
of the database (on Sleep-EDF v1-2013 +/-30mins: 86.1.0%, 79.6%, 0.81; on
Sleep-EDF v2-2018 +/-30mins: 82.3%, 76.7%, 0.76). Our lighter sleep scoring
approach paves the way to the application of scoring algorithms for sleep
analysis in real-time.

    

### [[2108.10608] Occlusion-robust Visual Markerless Bone Tracking for Computer-Assisted Orthopaedic Surgery](http://arxiv.org/abs/2108.10608)


  Conventional computer-assisted orthopaedic navigation systems rely on the
tracking of dedicated optical markers for patient poses, which makes the
surgical workflow more invasive, tedious, and expensive. Visual tracking has
recently been proposed to measure the target anatomy in a markerless and
effortless way, but the existing methods fail under real-world occlusion caused
by intraoperative interventions. Furthermore, such methods are
hardware-specific and not accurate enough for surgical applications. In this
paper, we propose a RGB-D sensing-based markerless tracking method that is
robust against occlusion. We design a new segmentation network that features
dynamic region-of-interest prediction and robust 3D point cloud segmentation.
As it is expensive to collect large-scale training data with occlusion
instances, we also propose a new method to create synthetic RGB-D images for
network training. Experimental results show that our proposed markerless
tracking method outperforms recent state-of-the-art approaches by a large
margin, especially when an occlusion exists. Furthermore, our method
generalises well to new cameras and new target models, including a cadaver,
without the need for network retraining. In practice, by using a high-quality
commercial RGB-D camera, our proposed visual tracking method achieves an
accuracy of 1-2 degress and 2-4 mm on a model knee, which meets the standard
for clinical applications.

    

### [[2108.10612] ProtoMIL: Multiple Instance Learning with Prototypical Parts for Fine-Grained Interpretability](http://arxiv.org/abs/2108.10612)


  Multiple Instance Learning (MIL) gains popularity in many real-life machine
learning applications due to its weakly supervised nature. However, the
corresponding effort on explaining MIL lags behind, and it is usually limited
to presenting instances of a bag that are crucial for a particular prediction.
In this paper, we fill this gap by introducing ProtoMIL, a novel
self-explainable MIL method inspired by the case-based reasoning process that
operates on visual prototypes. Thanks to incorporating prototypical features
into objects description, ProtoMIL unprecedentedly joins the model accuracy and
fine-grained interpretability, which we present with the experiments on five
recognized MIL datasets.

    

### [[2108.10623] Data-Free Evaluation of User Contributions in Federated Learning](http://arxiv.org/abs/2108.10623)


  Federated learning (FL) trains a machine learning model on mobile devices in
a distributed manner using each device's private data and computing resources.
A critical issues is to evaluate individual users' contributions so that (1)
users' effort in model training can be compensated with proper incentives and
(2) malicious and low-quality users can be detected and removed. The
state-of-the-art solutions require a representative test dataset for the
evaluation purpose, but such a dataset is often unavailable and hard to
synthesize. In this paper, we propose a method called Pairwise Correlated
Agreement (PCA) based on the idea of peer prediction to evaluate user
contribution in FL without a test dataset. PCA achieves this using the
statistical correlation of the model parameters uploaded by users. We then
apply PCA to designing (1) a new federated learning algorithm called Fed-PCA,
and (2) a new incentive mechanism that guarantees truthfulness. We evaluate the
performance of PCA and Fed-PCA using the MNIST dataset and a large industrial
product recommendation dataset. The results demonstrate that our Fed-PCA
outperforms the canonical FedAvg algorithm and other baseline methods in
accuracy, and at the same time, PCA effectively incentivizes users to behave
truthfully.

    

### [[2108.10636] Adaptive and Interpretable Graph Convolution Networks Using Generalized Pagerank](http://arxiv.org/abs/2108.10636)


  We investigate adaptive layer-wise graph convolution in deep GCN models. We
propose AdaGPR to learn generalized Pageranks at each layer of a GCNII network
to induce adaptive convolution. We show that the generalization bound for
AdaGPR is bounded by a polynomial of the eigenvalue spectrum of the normalized
adjacency matrix in the order of the number of generalized Pagerank
coefficients. By analysing the generalization bounds we show that oversmoothing
depends on both the convolutions by the higher orders of the normalized
adjacency matrix and the depth of the model. We performed evaluations on
node-classification using benchmark real data and show that AdaGPR provides
improved accuracies compared to existing graph convolution networks while
demonstrating robustness against oversmoothing. Further, we demonstrate that
analysis of coefficients of layer-wise generalized Pageranks allows us to
qualitatively understand convolution at each layer enabling model
interpretations.

    

### [[2108.10639] GrADE: A graph based data-driven solver for time-dependent nonlinear partial differential equations](http://arxiv.org/abs/2108.10639)


  The physical world is governed by the laws of physics, often represented in
form of nonlinear partial differential equations (PDEs). Unfortunately,
solution of PDEs is non-trivial and often involves significant computational
time. With recent developments in the field of artificial intelligence and
machine learning, the solution of PDEs using neural network has emerged as a
domain with huge potential. However, most of the developments in this field are
based on either fully connected neural networks (FNN) or convolutional neural
networks (CNN). While FNN is computationally inefficient as the number of
network parameters can be potentially huge, CNN necessitates regular grid and
simpler domain. In this work, we propose a novel framework referred to as the
Graph Attention Differential Equation (GrADE) for solving time dependent
nonlinear PDEs. The proposed approach couples FNN, graph neural network, and
recently developed Neural ODE framework. The primary idea is to use graph
neural network for modeling the spatial domain, and Neural ODE for modeling the
temporal domain. The attention mechanism identifies important inputs/features
and assign more weightage to the same; this enhances the performance of the
proposed framework. Neural ODE, on the other hand, results in constant memory
cost and allows trading of numerical precision for speed. We also propose depth
refinement as an effective technique for training the proposed architecture in
lesser time with better accuracy. The effectiveness of the proposed framework
is illustrated using 1D and 2D Burgers' equations. Results obtained illustrate
the capability of the proposed framework in modeling PDE and its scalability to
larger domains without the need for retraining.

    

### [[2108.10660] Data Aggregation for Reducing Training Data in Symbolic Regression](http://arxiv.org/abs/2108.10660)


  The growing volume of data makes the use of computationally intense machine
learning techniques such as symbolic regression with genetic programming more
and more impractical. This work discusses methods to reduce the training data
and thereby also the runtime of genetic programming. The data is aggregated in
a preprocessing step before running the actual machine learning algorithm.
K-means clustering and data binning is used for data aggregation and compared
with random sampling as the simplest data reduction method. We analyze the
achieved speed-up in training and the effects on the trained models test
accuracy for every method on four real-world data sets. The performance of
genetic programming is compared with random forests and linear regression. It
is shown, that k-means and random sampling lead to very small loss in test
accuracy when the data is reduced down to only 30% of the original data, while
the speed-up is proportional to the size of the data set. Binning on the
contrary, leads to models with very high test error.

    

### [[2108.10661] On the Effectiveness of Genetic Operations in Symbolic Regression](http://arxiv.org/abs/2108.10661)


  This paper describes a methodology for analyzing the evolutionary dynamics of
genetic programming (GP) using genealogical information, diversity measures and
information about the fitness variation from parent to offspring. We introduce
a new subtree tracing approach for identifying the origins of genes in the
structure of individuals, and we show that only a small fraction of ancestor
individuals are responsible for the evolvement of the best solutions in the
population.

    

### [[2108.10663] Energy time series forecasting-Analytical and empirical assessment of conventional and machine learning models](http://arxiv.org/abs/2108.10663)


  Machine learning methods have been adopted in the literature as contenders to
conventional methods to solve the energy time series forecasting (TSF)
problems. Recently, deep learning methods have been emerged in the artificial
intelligence field attaining astonishing performance in a wide range of
applications. Yet, the evidence about their performance in to solve the energy
TSF problems, in terms of accuracy and computational requirements, is scanty.
Most of the review articles that handle the energy TSF problem are systematic
reviews, however, a qualitative and quantitative study for the energy TSF
problem is not yet available in the literature. The purpose of this paper is
twofold, first it provides a comprehensive analytical assessment for
conventional,machine learning, and deep learning methods that can be utilized
to solve various energy TSF problems. Second, the paper carries out an
empirical assessment for many selected methods through three real-world
datasets. These datasets related to electrical energy consumption problem,
natural gas problem, and electric power consumption of an individual household
problem.The first two problems are univariate TSF and the third problem is a
multivariate TSF. Com-pared to both conventional and machine learning
contenders, the deep learning methods attain a significant improvement in terms
of accuracy and forecasting horizons examined. In the mean-time, their
computational requirements are notably greater than other contenders.
Eventually,the paper identifies a number of challenges, potential research
directions, and recommendations to the research community may serve as a basis
for further research in the energy forecasting domain.

    

### [[2108.10673] Out-of-Distribution Example Detection in Deep Neural Networks using Distance to Modelled Embedding](http://arxiv.org/abs/2108.10673)


  Adoption of deep learning in safety-critical systems raise the need for
understanding what deep neural networks do not understand after models have
been deployed. The behaviour of deep neural networks is undefined for so called
out-of-distribution examples. That is, examples from another distribution than
the training set. Several methodologies to detect out-of-distribution examples
during prediction-time have been proposed, but these methodologies constrain
either neural network architecture, how the neural network is trained, suffer
from performance overhead, or assume that the nature of out-of-distribution
examples are known a priori. We present Distance to Modelled Embedding (DIME)
that we use to detect out-of-distribution examples during prediction time. By
approximating the training set embedding into feature space as a linear
hyperplane, we derive a simple, unsupervised, highly performant and
computationally efficient method. DIME allows us to add prediction-time
detection of out-of-distribution examples to neural network models without
altering architecture or training while imposing minimal constraints on when it
is applicable. In our experiments, we demonstrate that by using DIME as an
add-on after training, we efficiently detect out-of-distribution examples
during prediction and match state-of-the-art methods while being more versatile
and introducing negligible computational overhead.

    

### [[2108.10684] Measuring Wikipedia Article Quality in One Dimension by Extending ORES with Ordinal Regression](http://arxiv.org/abs/2108.10684)


  Organizing complex peer production projects and advancing scientific
knowledge of open collaboration each depend on the ability to measure quality.
Article quality ratings on English language Wikipedia have been widely used by
both Wikipedia community members and academic researchers for purposes like
tracking knowledge gaps and studying how political polarization shapes
collaboration. Even so, measuring quality presents many methodological
challenges. The most widely used systems use labels on discrete ordinal scales
when assessing quality, but such labels can be inconvenient for statistics and
machine learning. Prior work handles this by assuming that different levels of
quality are "evenly spaced" from one another. This assumption runs counter to
intuitions about the relative degrees of effort needed to raise Wikipedia
encyclopedia articles to different quality levels. Furthermore, models from
prior work are fit to datasets that oversample high-quality articles. This
limits their accuracy for representative samples of articles or revisions. I
describe a technique extending the Wikimedia Foundations' ORES article quality
model to address these limitations. My method uses weighted ordinal regression
models to construct one-dimensional continuous measures of quality. While
scores from my technique and from prior approaches are correlated, my approach
improves accuracy for research datasets and provides evidence that the "evenly
spaced" assumption is unfounded in practice on English Wikipedia. I conclude
with recommendations for using quality scores in future research and include
the full code, data, and models.

    

### [[2108.10687] Deep Active Learning for Text Classification with Diverse Interpretations](http://arxiv.org/abs/2108.10687)


  Recently, Deep Neural Networks (DNNs) have made remarkable progress for text
classification, which, however, still require a large number of labeled data.
To train high-performing models with the minimal annotation cost, active
learning is proposed to select and label the most informative samples, yet it
is still challenging to measure informativeness of samples used in DNNs. In
this paper, inspired by piece-wise linear interpretability of DNNs, we propose
a novel Active Learning with DivErse iNterpretations (ALDEN) approach. With
local interpretations in DNNs, ALDEN identifies linearly separable regions of
samples. Then, it selects samples according to their diversity of local
interpretations and queries their labels. To tackle the text classification
problem, we choose the word with the most diverse interpretations to represent
the whole sentence. Extensive experiments demonstrate that ALDEN consistently
outperforms several state-of-the-art deep active learning methods.

    

### [[2108.10692] Generalized Optimal Linear Orders](http://arxiv.org/abs/2108.10692)


  The sequential structure of language, and the order of words in a sentence
specifically, plays a central role in human language processing. Consequently,
in designing computational models of language, the de facto approach is to
present sentences to machines with the words ordered in the same order as in
the original human-authored sentence. The very essence of this work is to
question the implicit assumption that this is desirable and inject theoretical
soundness into the consideration of word order in natural language processing.
In this thesis, we begin by uniting the disparate treatments of word order in
cognitive science, psycholinguistics, computational linguistics, and natural
language processing under a flexible algorithmic framework. We proceed to use
this heterogeneous theoretical foundation as the basis for exploring new word
orders with an undercurrent of psycholinguistic optimality. In particular, we
focus on notions of dependency length minimization given the difficulties in
human and computational language processing in handling long-distance
dependencies. We then discuss algorithms for finding optimal word orders
efficiently in spite of the combinatorial space of possibilities. We conclude
by addressing the implications of these word orders on human language and their
downstream impacts when integrated in computational models.

    

### [[2108.10697] Does Adversarial Oversampling Help us?](http://arxiv.org/abs/2108.10697)


  Traditional oversampling methods are generally employed to handle class
imbalance in datasets. This oversampling approach is independent of the
classifier; thus, it does not offer an end-to-end solution. To overcome this,
we propose a three-player adversarial game-based end-to-end method, where a
domain-constraints mixture of generators, a discriminator, and a multi-class
classifier are used. Rather than adversarial minority oversampling, we propose
an adversarial oversampling (AO) and a data-space oversampling (DO) approach.
In AO, the generator updates by fooling both the classifier and discriminator,
however, in DO, it updates by favoring the classifier and fooling the
discriminator. While updating the classifier, it considers both the real and
synthetically generated samples in AO. But, in DO, it favors the real samples
and fools the subset class-specific generated samples. To mitigate the biases
of a classifier towards the majority class, minority samples are over-sampled
at a fractional rate. Such implementation is shown to provide more robust
classification boundaries. The effectiveness of our proposed method has been
validated with high-dimensional, highly imbalanced and large-scale multi-class
tabular datasets. The results as measured by average class specific accuracy
(ACSA) clearly indicate that the proposed method provides better classification
accuracy (improvement in the range of 0.7% to 49.27%) as compared to the
baseline classifier.

    

### [[2108.10698] Efficacy of BERT embeddings on predicting disaster from Twitter data](http://arxiv.org/abs/2108.10698)


  Social media like Twitter provide a common platform to share and communicate
personal experiences with other people. People often post their life
experiences, local news, and events on social media to inform others. Many
rescue agencies monitor this type of data regularly to identify disasters and
reduce the risk of lives. However, it is impossible for humans to manually
check the mass amount of data and identify disasters in real-time. For this
purpose, many research works have been proposed to present words in
machine-understandable representations and apply machine learning methods on
the word representations to identify the sentiment of a text. The previous
research methods provide a single representation or embedding of a word from a
given document. However, the recent advanced contextual embedding method (BERT)
constructs different vectors for the same word in different contexts. BERT
embeddings have been successfully used in different natural language processing
(NLP) tasks, yet there is no concrete analysis of how these representations are
helpful in disaster-type tweet analysis. In this research work, we explore the
efficacy of BERT embeddings on predicting disaster from Twitter data and
compare these to traditional context-free word embedding methods (GloVe,
Skip-gram, and FastText). We use both traditional machine learning methods and
deep learning methods for this purpose. We provide both quantitative and
qualitative results for this study. The results show that the BERT embeddings
have the best results in disaster prediction task than the traditional word
embeddings. Our codes are made freely accessible to the research community.

    

### [[2108.10701] Sonic: A Sampling-based Online Controller for Streaming Applications](http://arxiv.org/abs/2108.10701)


  Many applications in important problem domains such as machine learning and
computer vision are streaming applications that take a sequence of inputs over
time. It is challenging to find knob settings that optimize the run-time
performance of such applications because the optimal knob settings are usually
functions of inputs, computing platforms, time as well as user's requirements,
which can be very diverse.
Most prior works address this problem by offline profiling followed by
training models for control. However, profiling-based approaches incur large
overhead before execution; it is also difficult to redeploy them in other
run-time configurations.
In this paper, we propose Sonic, a sampling-based online controller for
long-running streaming applications that does not require profiling ahead of
time. Within each phase of a streaming application's execution, Sonic utilizes
the beginning portion to sample the knob space strategically and aims to pick
the optimal knob setting for the rest of the phase, given a user-specified
constrained optimization problem. A hybrid approach of machine learning
regressions and Bayesian optimization are used for better overall sampling
choices.
Sonic is implemented independent of application, device, input, performance
objective and constraints. We evaluate Sonic on traditional parallel benchmarks
as well as on deep learning inference benchmarks across multiple platforms. Our
experiments show that when using Sonic to control knob settings, application
run-time performance is only 5.3% less than if optimal knob settings were used,
demonstrating that Sonic is able to find near-optimal knob settings under
diverse run-time configurations without prior knowledge quickly.

    

### [[2108.10703] REFINE: Random RangE FInder for Network Embedding](http://arxiv.org/abs/2108.10703)


  Network embedding approaches have recently attracted considerable interest as
they learn low-dimensional vector representations of nodes. Embeddings based on
the matrix factorization are effective but they are usually computationally
expensive due to the eigen-decomposition step. In this paper, we propose a
Random RangE FInder based Network Embedding (REFINE) algorithm, which can
perform embedding on one million of nodes (YouTube) within 30 seconds in a
single thread. REFINE is 10x faster than ProNE, which is 10-400x faster than
other methods such as LINE, DeepWalk, Node2Vec, GraRep, and Hope. Firstly, we
formulate our network embedding approach as a skip-gram model, but with an
orthogonal constraint, and we reformulate it into the matrix factorization
problem. Instead of using randomized tSVD (truncated SVD) as other methods, we
employ the Randomized Blocked QR decomposition to obtain the node
representation fast. Moreover, we design a simple but efficient spectral filter
for network enhancement to obtain higher-order information for node
representation. Experimental results prove that REFINE is very efficient on
datasets of different sizes (from thousand to million of nodes/edges) for node
classification, while enjoying a good performance.

    

### [[2108.10714] Curricular SincNet: Towards Robust Deep Speaker Recognition by Emphasizing Hard Samples in Latent Space](http://arxiv.org/abs/2108.10714)


  Deep learning models have become an increasingly preferred option for
biometric recognition systems, such as speaker recognition. SincNet, a deep
neural network architecture, gained popularity in speaker recognition tasks due
to its parameterized sinc functions that allow it to work directly on the
speech signal. The original SincNet architecture uses the softmax loss, which
may not be the most suitable choice for recognition-based tasks. Such loss
functions do not impose inter-class margins nor differentiate between easy and
hard training samples. Curriculum learning, particularly those leveraging
angular margin-based losses, has proven very successful in other biometric
applications such as face recognition. The advantage of such a curriculum
learning-based techniques is that it will impose inter-class margins as well as
taking to account easy and hard samples. In this paper, we propose Curricular
SincNet (CL-SincNet), an improved SincNet model where we use a curricular loss
function to train the SincNet architecture. The proposed model is evaluated on
multiple datasets using intra-dataset and inter-dataset evaluation protocols.
In both settings, the model performs competitively with other previously
published work. In the case of inter-dataset testing, it achieves the best
overall results with a reduction of 4\% error rate compare to SincNet and other
published work.

    

### [[2108.10715] From One to Many: A Deep Learning Coincident Gravitational-Wave Search](http://arxiv.org/abs/2108.10715)


  Gravitational waves from the coalescence of compact-binary sources are now
routinely observed by Earth bound detectors. The most sensitive search
algorithms convolve many different pre-calculated gravitational waveforms with
the detector data and look for coincident matches between different detectors.
Machine learning is being explored as an alternative approach to building a
search algorithm that has the prospect to reduce computational costs and target
more complex signals. In this work we construct a two-detector search for
gravitational waves from binary black hole mergers using neural networks
trained on non-spinning binary black hole data from a single detector. The
network is applied to the data from both observatories independently and we
check for events coincident in time between the two. This enables the efficient
analysis of large quantities of background data by time-shifting the
independent detector data. We find that while for a single detector the network
retains $91.5\%$ of the sensitivity matched filtering can achieve, this number
drops to $83.9\%$ for two observatories. To enable the network to check for
signal consistency in the detectors, we then construct a set of simple networks
that operate directly on data from both detectors. We find that none of these
simple two-detector networks are capable of improving the sensitivity over
applying networks individually to the data from the detectors and searching for
time coincidences.

    

### [[2108.10717] Improvement of a Prediction Model for Heart Failure Survival through Explainable Artificial Intelligence](http://arxiv.org/abs/2108.10717)


  Cardiovascular diseases and their associated disorder of heart failure are
one of the major death causes globally, being a priority for doctors to detect
and predict its onset and medical consequences. Artificial Intelligence (AI)
allows doctors to discover clinical indicators and enhance their diagnosis and
treatments. Specifically, explainable AI offers tools to improve the clinical
prediction models that experience poor interpretability of their results. This
work presents an explainability analysis and evaluation of a prediction model
for heart failure survival by using a dataset that comprises 299 patients who
suffered heart failure. The model employs a data workflow pipeline able to
select the best ensemble tree algorithm as well as the best feature selection
technique. Moreover, different post-hoc techniques have been used for the
explainability analysis of the model. The paper's main contribution is an
explainability-driven approach to select the best prediction model for HF
survival based on an accuracy-explainability balance. Therefore, the most
balanced explainable prediction model implements an Extra Trees classifier over
5 selected features (follow-up time, serum creatinine, ejection fraction, age
and diabetes) out of 12, achieving a balanced-accuracy of 85.1% and 79.5% with
cross-validation and new unseen data respectively. The follow-up time is the
most influencing feature followed by serum-creatinine and ejection-fraction.
The explainable prediction model for HF survival presented in this paper would
improve a further adoption of clinical prediction models by providing doctors
with intuitions to better understand the reasoning of, usually, black-box AI
clinical solutions, and make more reasonable and data-driven decisions.

    

### [[2108.10733] Graph Neural Networks: Methods, Applications, and Opportunities](http://arxiv.org/abs/2108.10733)


  In the last decade or so, we have witnessed deep learning reinvigorating the
machine learning field. It has solved many problems in the domains of computer
vision, speech recognition, natural language processing, and various other
tasks with state-of-the-art performance. The data is generally represented in
the Euclidean space in these domains. Various other domains conform to
non-Euclidean space, for which graph is an ideal representation. Graphs are
suitable for representing the dependencies and interrelationships between
various entities. Traditionally, handcrafted features for graphs are incapable
of providing the necessary inference for various tasks from this complex data
representation. Recently, there is an emergence of employing various advances
in deep learning to graph data-based tasks. This article provides a
comprehensive survey of graph neural networks (GNNs) in each learning setting:
supervised, unsupervised, semi-supervised, and self-supervised learning.
Taxonomy of each graph based learning setting is provided with logical
divisions of methods falling in the given learning setting. The approaches for
each learning task are analyzed from both theoretical as well as empirical
standpoints. Further, we provide general architecture guidelines for building
GNNs. Various applications and benchmark datasets are also provided, along with
open challenges still plaguing the general applicability of GNNs.

    

### [[2108.10744] Interpretable deep-learning models to help achieve the Sustainable Development Goals](http://arxiv.org/abs/2108.10744)


  We discuss our insights into interpretable artificial-intelligence (AI)
models, and how they are essential in the context of developing ethical AI
systems, as well as data-driven solutions compliant with the Sustainable
Development Goals (SDGs). We highlight the potential of extracting
truly-interpretable models from deep-learning methods, for instance via
symbolic models obtained through inductive biases, to ensure a sustainable
development of AI.

    

### [[2108.10748] Federated Learning for UAV Swarms Under Class Imbalance and Power Consumption Constraints](http://arxiv.org/abs/2108.10748)


  The usage of unmanned aerial vehicles (UAVs) in civil and military
applications continues to increase due to the numerous advantages that they
provide over conventional approaches. Despite the abundance of such advantages,
it is imperative to investigate the performance of UAV utilization while
considering their design limitations. This paper investigates the deployment of
UAV swarms when each UAV carries a machine learning classification task. To
avoid data exchange with ground-based processing nodes, a federated learning
approach is adopted between a UAV leader and the swarm members to improve the
local learning model while avoiding excessive air-to-ground and ground-to-air
communications. Moreover, the proposed deployment framework considers the
stringent energy constraints of UAVs and the problem of class imbalance, where
we show that considering these design parameters significantly improves the
performances of the UAV swarm in terms of classification accuracy, energy
consumption and availability of UAVs when compared with several baseline
algorithms.

    

### [[2108.10749] Federated Learning for Open Banking](http://arxiv.org/abs/2108.10749)


  Open banking enables individual customers to own their banking data, which
provides fundamental support for the boosting of a new ecosystem of data
marketplaces and financial services. In the near future, it is foreseeable to
have decentralized data ownership in the finance sector using federated
learning. This is a just-in-time technology that can learn intelligent models
in a decentralized training manner. The most attractive aspect of federated
learning is its ability to decompose model training into a centralized server
and distributed nodes without collecting private data. This kind of decomposed
learning framework has great potential to protect users' privacy and sensitive
data. Therefore, federated learning combines naturally with an open banking
data marketplaces. This chapter will discuss the possible challenges for
applying federated learning in the context of open banking, and the
corresponding solutions have been explored as well.

    

### [[2108.10751] Understanding the Basis of Graph Convolutional Neural Networks via an Intuitive Matched Filtering Approach](http://arxiv.org/abs/2108.10751)


  Graph Convolutional Neural Networks (GCNN) are becoming a preferred model for
data processing on irregular domains, yet their analysis and principles of
operation are rarely examined due to the black box nature of NNs. To this end,
we revisit the operation of GCNNs and show that their convolution layers
effectively perform matched filtering of input data with the chosen patterns
(features). This allows us to provide a unifying account of GCNNs through a
matched filter perspective, whereby the nonlinear ReLU and max-pooling layers
are also discussed within the matched filtering framework. This is followed by
a step-by-step guide on information propagation and learning in GCNNs. It is
also shown that standard CNNs and fully connected NNs can be obtained as a
special case of GCNNs. A carefully chosen numerical example guides the reader
through the various steps of GCNN operation and learning both visually and
numerically.

    

### [[2108.10761] Federated Learning for Privacy-Preserving Open Innovation Future on Digital Health](http://arxiv.org/abs/2108.10761)


  Privacy protection is an ethical issue with broad concern in Artificial
Intelligence (AI). Federated learning is a new machine learning paradigm to
learn a shared model across users or organisations without direct access to the
data. It has great potential to be the next-general AI model training framework
that offers privacy protection and therefore has broad implications for the
future of digital health and healthcare informatics. Implementing an open
innovation framework in the healthcare industry, namely open health, is to
enhance innovation and creative capability of health-related organisations by
building a next-generation collaborative framework with partner organisations
and the research community. In particular, this game-changing collaborative
framework offers knowledge sharing from diverse data with a privacy-preserving.
This chapter will discuss how federated learning can enable the development of
an open health ecosystem with the support of AI. Existing challenges and
solutions for federated learning will be discussed.

    

### [[2108.10763] ComSum: Commit Messages Summarization and Meaning Preservation](http://arxiv.org/abs/2108.10763)


  We present ComSum, a data set of 7 million commit messages for text
summarization. When documenting commits, software code changes, both a message
and its summary are posted. We gather and filter those to curate developers'
work summarization data set. Along with its growing size, practicality and
challenging language domain, the data set benefits from the living field of
empirical software engineering. As commits follow a typology, we propose to not
only evaluate outputs by Rouge, but by their meaning preservation.

    

### [[2108.10764] Regularizing Transformers With Deep Probabilistic Layers](http://arxiv.org/abs/2108.10764)


  Language models (LM) have grown with non-stop in the last decade, from
sequence-to-sequence architectures to the state-of-the-art and utter
attention-based Transformers. In this work, we demonstrate how the inclusion of
deep generative models within BERT can bring more versatile models, able to
impute missing/noisy words with richer text or even improve BLEU score. More
precisely, we use a Gaussian Mixture Variational Autoencoder (GMVAE) as a
regularizer layer and prove its effectiveness not only in Transformers but also
in the most relevant encoder-decoder based LM, seq2seq with and without
attention.

    

### [[2108.10781] Adaptive Explainable Continual Learning Framework for Regression Problems with Focus on Power Forecasts](http://arxiv.org/abs/2108.10781)


  Compared with traditional deep learning techniques, continual learning
enables deep neural networks to learn continually and adaptively. Deep neural
networks have to learn new tasks and overcome forgetting the knowledge obtained
from the old tasks as the amount of data keeps increasing in applications. In
this article, two continual learning scenarios will be proposed to describe the
potential challenges in this context. Besides, based on our previous work
regarding the CLeaR framework, which is short for continual learning for
regression tasks, the work will be further developed to enable models to extend
themselves and learn data successively. Research topics are related but not
limited to developing continual deep learning algorithms, strategies for
non-stationarity detection in data streams, explainable and visualizable
artificial intelligence, etc. Moreover, the framework- and algorithm-related
hyperparameters should be dynamically updated in applications. Forecasting
experiments will be conducted based on power generation and consumption data
collected from real-world applications. A series of comprehensive evaluation
metrics and visualization tools can help analyze the experimental results. The
proposed framework is expected to be generally applied to other constantly
changing scenarios.

    

### [[2108.10808] Greenformers: Improving Computation and Memory Efficiency in Transformer Models via Low-Rank Approximation](http://arxiv.org/abs/2108.10808)


  In this thesis, we introduce Greenformers, a collection of model efficiency
methods to improve the model efficiency of the recently renowned transformer
models with a low-rank approximation approach. The development trend of deep
learning models tends to results in a more complex and larger model. Although
it leads to a better and more accurate prediction, the resulting model becomes
even more costly, as it requires weeks of training with a huge amount of GPU
resources. Particularly, the size and computational cost of transformer-based
models have increased tremendously since its first debut in 2017 from ~100
million parameters up to ~1.6 trillion parameters in early 2021. This
computationally hungry model also incurs a substantial cost to the environment
and even reaches an alarming level of carbon footprint. Some of these models
are so massive that it is even impossible to run the model without a GPU
cluster.
Greenformers improve the model efficiency of transformer models by applying
low-rank approximation approaches. Specifically, we propose a low-rank
factorization approach to improve the efficiency of the transformer model
called Low-Rank Transformer. We further compare our model with an existing
low-rank factorization approach called Linformer. Based on our analysis, the
Low-Rank Transformer model is suitable for improving both the time and memory
efficiency in processing short-sequence (<= 512) input data, while the
Linformer model is suitable for improving the efficiency in processing
long-sequence input data (>= 512). We also show that Low-Rank Transformer is
more suitable for on-device deployment, as it significantly reduces the model
size. Additionally, we estimate that applying LRT to the existing BERT-base
model can significantly reduce the computational, economical, and environmental
costs for developing such models by more than 30% of its original costs.

    

### [[2108.10821] Graph Contrastive Pre-training for Effective Theorem Reasoning](http://arxiv.org/abs/2108.10821)


  Interactive theorem proving is a challenging and tedious process, which
requires non-trivial expertise and detailed low-level instructions (or tactics)
from human experts. Tactic prediction is a natural way to automate this
process. Existing methods show promising results on tactic prediction by
learning a deep neural network (DNN) based model from proofs written by human
experts. In this paper, we propose NeuroTactic, a novel extension with a
special focus on improving the representation learning for theorem proving.
NeuroTactic leverages graph neural networks (GNNs) to represent the theorems
and premises, and applies graph contrastive learning for pre-training. We
demonstrate that the representation learning of theorems is essential to
predict tactics. Compared with other methods, NeuroTactic achieves
state-of-the-art performance on the CoqGym dataset.

    

### [[2108.10825] Adaptive Group Lasso Neural Network Models for Functions of Few Variables and Time-Dependent Data](http://arxiv.org/abs/2108.10825)


  In this paper, we propose an adaptive group Lasso deep neural network for
high-dimensional function approximation where input data are generated from a
dynamical system and the target function depends on few active variables or few
linear combinations of variables. We approximate the target function by a deep
neural network and enforce an adaptive group Lasso constraint to the weights of
a suitable hidden layer in order to represent the constraint on the target
function. Our empirical studies show that the proposed method outperforms
recent state-of-the-art methods including the sparse dictionary matrix method,
neural networks with or without group Lasso penalty.

    

### [[2108.10826] S&P 500 Stock Price Prediction Using Technical, Fundamental and Text Data](http://arxiv.org/abs/2108.10826)


  We summarized both common and novel predictive models used for stock price
prediction and combined them with technical indices, fundamental
characteristics and text-based sentiment data to predict S&P stock prices. A
66.18% accuracy in S&P 500 index directional prediction and 62.09% accuracy in
individual stock directional prediction was achieved by combining different
machine learning models such as Random Forest and LSTM together into
state-of-the-art ensemble models. The data we use contains weekly historical
prices, finance reports, and text information from news items associated with
518 different common stocks issued by current and former S&P 500 large-cap
companies, from January 1, 2000 to December 31, 2019. Our study's innovation
includes utilizing deep language models to categorize and infer financial news
item sentiment; fusing different models containing different combinations of
variables and stocks to jointly make predictions; and overcoming the
insufficient data problem for machine learning models in time series by using
data across different stocks.

    

### [[2108.10828] Physics-Informed Deep Learning: A Promising Technique for System Reliability Assessment](http://arxiv.org/abs/2108.10828)


  Considerable research has been devoted to deep learning-based predictive
models for system prognostics and health management in the reliability and
safety community. However, there is limited study on the utilization of deep
learning for system reliability assessment. This paper aims to bridge this gap
and explore this new interface between deep learning and system reliability
assessment by exploiting the recent advances of physics-informed deep learning.
Particularly, we present an approach to frame system reliability assessment in
the context of physics-informed deep learning and discuss the potential value
of physics-informed generative adversarial networks for the uncertainty
quantification and measurement data incorporation in system reliability
assessment. The proposed approach is demonstrated by three numerical examples
involving a dual-processor computing system. The results indicate the potential
value of physics-informed deep learning to alleviate computational challenges
and combine measurement data and mathematical models for system reliability
assessment.

    

### [[2108.10842] imGHUM: Implicit Generative Models of 3D Human Shape and Articulated Pose](http://arxiv.org/abs/2108.10842)


  We present imGHUM, the first holistic generative model of 3D human shape and
articulated pose, represented as a signed distance function. In contrast to
prior work, we model the full human body implicitly as a function
zero-level-set and without the use of an explicit template mesh. We propose a
novel network architecture and a learning paradigm, which make it possible to
learn a detailed implicit generative model of human pose, shape, and semantics,
on par with state-of-the-art mesh-based models. Our model features desired
detail for human models, such as articulated pose including hand motion and
facial expressions, a broad spectrum of shape variations, and can be queried at
arbitrary resolutions and spatial locations. Additionally, our model has
attached spatial semantics making it straightforward to establish
correspondences between different shape instances, thus enabling applications
that are difficult to tackle using classical implicit representations. In
extensive experiments, we demonstrate the model accuracy and its applicability
to current research problems.

    

### [[2108.10859] Regret Analysis of Global Optimization in Univariate Functions with Lipschitz Derivatives](http://arxiv.org/abs/2108.10859)


  In this work, we study the problem of global optimization in univariate loss
functions, where we analyze the regret of the popular lower bounding algorithms
(e.g., Piyavskii-Shubert algorithm). For any given time $T$, instead of the
widely available simple regret (which is the difference of the losses between
the best estimation up to $T$ and the global optimizer), we study the
cumulative regret up to that time. With a suitable lower bounding algorithm, we
show that it is possible to achieve satisfactory cumulative regret bounds for
different classes of functions. For Lipschitz continuous functions with the
parameter $L$, we show that the cumulative regret is $O(L\log T)$. For
Lipschitz smooth functions with the parameter $H$, we show that the cumulative
regret is $O(H)$. We also analytically extend our results for a broader class
of functions that covers both the Lipschitz continuous and smooth functions
individually.

    

### [[2108.10873] A QuadTree Image Representation for Computational Pathology](http://arxiv.org/abs/2108.10873)


  The field of computational pathology presents many challenges for computer
vision algorithms due to the sheer size of pathology images. Histopathology
images are large and need to be split up into image tiles or patches so modern
convolutional neural networks (CNNs) can process them. In this work, we present
a method to generate an interpretable image representation of computational
pathology images using quadtrees and a pipeline to use these representations
for highly accurate downstream classification. To the best of our knowledge,
this is the first attempt to use quadtrees for pathology image data. We show it
is highly accurate, able to achieve as good results as the currently widely
adopted tissue mask patch extraction methods all while using over 38% less
data.

    

### [[1811.01545] PILAE: A Non-gradient Descent Learning Scheme for Deep Feedforward Neural Networks](http://arxiv.org/abs/1811.01545)


  In this work, a non-gradient descent learning (NGDL) scheme was proposed for
deep feedforward neural networks (DNN). It is known that an autoencoder can be
used as the building blocks of the multi-layer perceptron (MLP) DNN, the MLP is
taken as an example to illustrate the proposed scheme of pseudoinverse learning
algorithm for autoencoder (PILAE) in this paper. The PILAE with low rank
approximation is a NGDL algorithm, and the encoder weight matrix is set to be
the low rank approximation of the pseudoinverse of the input matrix, while the
decoder weight matrix is calculated by the pseudoinverse learning algorithm. It
is worth to note that only very few network structure hyper-parameters need to
be tuned compared with classical gradient descent learning algorithm. Hence,
the proposed algorithm could be regarded as a quasi-automated training
algorithm which could be utilized in automated machine learning field. The
experimental results show that the proposed learning scheme for DNN could
achieve better performance on considering the tradeoff between training
efficiency and classification accuracy.

    

### [[2002.06100] Analyzing Differentiable Fuzzy Logic Operators](http://arxiv.org/abs/2002.06100)


  The AI community is increasingly putting its attention towards combining
symbolic and neural approaches, as it is often argued that the strengths and
weaknesses of these approaches are complementary. One recent trend in the
literature are weakly supervised learning techniques that employ operators from
fuzzy logics. In particular, these use prior background knowledge described in
such logics to help the training of a neural network from unlabeled and noisy
data. By interpreting logical symbols using neural networks, this background
knowledge can be added to regular loss functions, hence making reasoning a part
of learning. We study, both formally and empirically, how a large collection of
logical operators from the fuzzy logic literature behave in a differentiable
learning setting. We find that many of these operators, including some of the
most well-known, are highly unsuitable in this setting. A further finding
concerns the treatment of implication in these fuzzy logics, and shows a strong
imbalance between gradients driven by the antecedent and the consequent of the
implication. Furthermore, we introduce a new family of fuzzy implications
(called sigmoidal implications) to tackle this phenomenon. Finally, we
empirically show that it is possible to use Differentiable Fuzzy Logics for
semi-supervised learning, and compare how different operators behave in
practice. We find that, to achieve the largest performance improvement over a
supervised baseline, we have to resort to non-standard combinations of logical
operators which perform well in learning, but no longer satisfy the usual
logical laws.

    

### [[2003.00295] Adaptive Federated Optimization](http://arxiv.org/abs/2003.00295)


  Federated learning is a distributed machine learning paradigm in which a
large number of clients coordinate with a central server to learn a model
without sharing their own training data. Standard federated optimization
methods such as Federated Averaging (FedAvg) are often difficult to tune and
exhibit unfavorable convergence behavior. In non-federated settings, adaptive
optimization methods have had notable success in combating such issues. In this
work, we propose federated versions of adaptive optimizers, including Adagrad,
Adam, and Yogi, and analyze their convergence in the presence of heterogeneous
data for general non-convex settings. Our results highlight the interplay
between client heterogeneity and communication efficiency. We also perform
extensive experiments on these methods and show that the use of adaptive
optimizers can significantly improve the performance of federated learning.

    

### [[2003.00856] Triangle-Net: Towards Robustness in Point Cloud Learning](http://arxiv.org/abs/2003.00856)


  Three dimensional (3D) object recognition is becoming a key desired
capability for many computer vision systems such as autonomous vehicles,
service robots and surveillance drones to operate more effectively in
unstructured environments. These real-time systems require effective
classification methods that are robust to various sampling resolutions, noisy
measurements, and unconstrained pose configurations. Previous research has
shown that points' sparsity, rotation and positional inherent variance can lead
to a significant drop in the performance of point cloud based classification
techniques. However, neither of them is sufficiently robust to multifactorial
variance and significant sparsity. In this regard, we propose a novel approach
for 3D classification that can simultaneously achieve invariance towards
rotation, positional shift, scaling, and is robust to point sparsity. To this
end, we introduce a new feature that utilizes graph structure of point clouds,
which can be learned end-to-end with our proposed neural network to acquire a
robust latent representation of the 3D object. We show that such latent
representations can significantly improve the performance of object
classification and retrieval tasks when points are sparse. Further, we show
that our approach outperforms PointNet and 3DmFV by 35.0% and 28.1%
respectively in ModelNet 40 classification tasks using sparse point clouds of
only 16 points under arbitrary SO(3) rotation.

    

### [[2006.06625] Cumulant GAN](http://arxiv.org/abs/2006.06625)


  In this paper, we propose a novel loss function for training Generative
Adversarial Networks (GANs) aiming towards deeper theoretical understanding as
well as improved stability and performance for the underlying optimization
problem. The new loss function is based on cumulant generating functions giving
rise to \emph{Cumulant GAN}. Relying on a recently-derived variational formula,
we show that the corresponding optimization problem is equivalent to R{é}nyi
divergence minimization, thus offering a (partially) unified perspective of GAN
losses: the R{é}nyi family encompasses Kullback-Leibler divergence (KLD),
reverse KLD, Hellinger distance and $\chi^2$-divergence. Wasserstein GAN is
also a member of cumulant GAN. In terms of stability, we rigorously prove the
linear convergence of cumulant GAN to the Nash equilibrium for a linear
discriminator, Gaussian distributions and the standard gradient descent ascent
algorithm. Finally, we experimentally demonstrate that image generation is more
robust relative to Wasserstein GAN and it is substantially improved in terms of
both inception score and Fréchet inception distance when both weaker and
stronger discriminators are considered.

    

### [[2007.01420] CoPhy-PGNN: Learning Physics-guided Neural Networks with Competing Loss Functions for Solving Eigenvalue Problems](http://arxiv.org/abs/2007.01420)


  Physics-guided Neural Networks (PGNNs) represent an emerging class of neural
networks that are trained using physics-guided (PG) loss functions (capturing
violations in network outputs with known physics), along with the supervision
contained in data. Existing work in PGNNs have demonstrated the efficacy of
adding single PG loss functions in the neural network objectives, using
constant trade-off parameters, to ensure better generalizability. However, in
the presence of multiple physics loss functions with competing gradient
directions, there is a need to adaptively tune the contribution of competing PG
loss functions during the course of training to arrive at generalizable
solutions. We demonstrate the presence of competing PG losses in the generic
neural network problem of solving for the lowest (or highest) eigenvector of a
physics-based eigenvalue equation, common to many scientific problems. We
present a novel approach to handle competing PG losses and demonstrate its
efficacy in learning generalizable solutions in two motivating applications of
quantum mechanics and electromagnetic propagation. All the code and data used
in this work is available at this https URL.

    

### [[2007.08902] A Unifying Perspective on Neighbor Embeddings along the Attraction-Repulsion Spectrum](http://arxiv.org/abs/2007.08902)


  Neighbor embeddings are a family of methods for visualizing complex
high-dimensional datasets using kNN graphs. To find the low-dimensional
embedding, these algorithms combine an attractive force between neighboring
pairs of points with a repulsive force between all points. One of the most
popular examples of such algorithms is t-SNE. Here we empirically show that
changing the balance between the attractive and the repulsive forces in t-SNE
using the exaggeration parameter yields a spectrum of embeddings, which is
characterized by a simple trade-off: stronger attraction can better represent
continuous manifold structures, while stronger repulsion can better represent
discrete cluster structures and yields higher kNN recall. We find that UMAP
embeddings correspond to t-SNE with increased attraction; mathematical analysis
shows that this is because the negative sampling optimisation strategy employed
by UMAP strongly lowers the effective repulsion. Likewise, ForceAtlas2,
commonly used for visualizing developmental single-cell transcriptomic data,
yields embeddings corresponding to t-SNE with the attraction increased even
more. At the extreme of this spectrum lie Laplacian Eigenmaps, corresponding to
the limit of infinite exaggeration. Our results demonstrate that many prominent
neighbor embedding algorithms can be placed onto the attraction-repulsion
spectrum, and highlight the inherent trade-offs between them.

    

### [[2008.02672] MFNets: Data efficient all-at-once learning of multifidelity surrogates as directed networks of information sources](http://arxiv.org/abs/2008.02672)


  We present an approach for constructing a surrogate from ensembles of
information sources of varying cost and accuracy. The multifidelity surrogate
encodes connections between information sources as a directed acyclic graph,
and is trained via gradient-based minimization of a nonlinear least squares
objective. While the vast majority of state-of-the-art assumes hierarchical
connections between information sources, our approach works with flexibly
structured information sources that may not admit a strict hierarchy. The
formulation has two advantages: (1) increased data efficiency due to
parsimonious multifidelity networks that can be tailored to the application;
and (2) no constraints on the training data -- we can combine noisy, non-nested
evaluations of the information sources. Numerical examples ranging from
synthetic to physics-based computational mechanics simulations indicate the
error in our approach can be orders-of-magnitude smaller, particularly in the
low-data regime, than single-fidelity and hierarchical multifidelity
approaches.

    

### [[2008.04733] Deep State-Space Gaussian Processes](http://arxiv.org/abs/2008.04733)


  This paper is concerned with a state-space approach to deep Gaussian process
(DGP) regression. We construct the DGP by hierarchically putting transformed
Gaussian process (GP) priors on the length scales and magnitudes of the next
level of Gaussian processes in the hierarchy. The idea of the state-space
approach is to represent the DGP as a non-linear hierarchical system of linear
stochastic differential equations (SDEs), where each SDE corresponds to a
conditional GP. The DGP regression problem then becomes a state estimation
problem, and we can estimate the state efficiently with sequential methods by
using the Markov property of the state-space DGP. The computational complexity
scales linearly with respect to the number of measurements. Based on this, we
formulate state-space MAP as well as Bayesian filtering and smoothing solutions
to the DGP regression problem. We demonstrate the performance of the proposed
models and methods on synthetic non-stationary signals and apply the
state-space DGP to detection of the gravitational waves from LIGO measurements.

    

### [[2010.00330] Workflow Provenance in the Lifecycle of Scientific Machine Learning](http://arxiv.org/abs/2010.00330)


  Machine Learning (ML) has already fundamentally changed several businesses.
More recently, it has also been profoundly impacting the computational science
and engineering domains, like geoscience, climate science, and health science.
In these domains, users need to perform comprehensive data analyses combining
scientific data and ML models to provide for critical requirements, such as
reproducibility, model explainability, and experiment data understanding.
However, scientific ML is multidisciplinary, heterogeneous, and affected by the
physical constraints of the domain, making such analyses even more challenging.
In this work, we leverage workflow provenance techniques to build a holistic
view to support the lifecycle of scientific ML. We contribute with (i)
characterization of the lifecycle and taxonomy for data analyses; (ii) design
principles to build this view, with a W3C PROV compliant data representation
and a reference system architecture; and (iii) lessons learned after an
evaluation in an Oil & Gas case using an HPC cluster with 393 nodes and 946
GPUs. The experiments show that the principles enable queries that integrate
domain semantics with ML models while keeping low overhead (<1%), high
scalability, and an order of magnitude of query acceleration under certain
workloads against without our representation.

    

### [[2011.11603] Interpretable Visual Reasoning via Induced Symbolic Space](http://arxiv.org/abs/2011.11603)


  We study the problem of concept induction in visual reasoning, i.e.,
identifying concepts and their hierarchical relationships from question-answer
pairs associated with images; and achieve an interpretable model via working on
the induced symbolic concept space. To this end, we first design a new
framework named object-centric compositional attention model (OCCAM) to perform
the visual reasoning task with object-level visual features. Then, we come up
with a method to induce concepts of objects and relations using clues from the
attention patterns between objects' visual features and question words.
Finally, we achieve a higher level of interpretability by imposing OCCAM on the
objects represented in the induced symbolic concept space. Our model design
makes this an easy adaption via first predicting the concepts of objects and
relations and then projecting the predicted concepts back to the visual feature
space so the compositional reasoning module can process normally. Experiments
on the CLEVR and GQA datasets demonstrate: 1) our OCCAM achieves a new state of
the art without human-annotated functional programs; 2) our induced concepts
are both accurate and sufficient as OCCAM achieves an on-par performance on
objects represented either in visual features or in the induced symbolic
concept space.

    

### [[2012.00517] One-Pixel Attack Deceives Computer-Assisted Diagnosis of Cancer](http://arxiv.org/abs/2012.00517)


  Computer vision and machine learning can be used to automate various tasks in
cancer diagnostic and detection. If an attacker can manipulate the automated
processing, the results can be devastating and in the worst case lead to wrong
diagnosis and treatment. In this research, the goal is to demonstrate the use
of one-pixel attacks in a real-life scenario with a real pathology dataset,
TUPAC16, which consists of digitized whole-slide images. We attack against the
IBM CODAIT's MAX breast cancer detector using adversarial images. These
adversarial examples are found using differential evolution to perform the
one-pixel modification to the images in the dataset. The results indicate that
a minor one-pixel modification of a whole slide image under analysis can affect
the diagnosis by reversing the automatic diagnosis result. The attack poses a
threat from the cyber security perspective: the one-pixel method can be used as
an attack vector by a motivated attacker.

    

### [[2012.03491] AI-enabled Prediction of eSports Player Performance Using the Data from Heterogeneous Sensors](http://arxiv.org/abs/2012.03491)


  The emerging progress of eSports lacks the tools for ensuring high-quality
analytics and training in Pro and amateur eSports teams. We report on an
Artificial Intelligence (AI) enabled solution for predicting the eSports player
in-game performance using exclusively the data from sensors. For this reason,
we collected the physiological, environmental, and the game chair data from Pro
and amateur players. The player performance is assessed from the game logs in a
multiplayer game for each moment of time using a recurrent neural network. We
have investigated that attention mechanism improves the generalization of the
network and provides the straightforward feature importance as well. The best
model achieves ROC AUC score 0.73. The prediction of the performance of
particular player is realized although his data are not utilized in the
training set. The proposed solution has a number of promising applications for
Pro eSports teams and amateur players, such as a learning tool or a performance
monitoring system.

    

### [[2012.04456] Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization](http://arxiv.org/abs/2012.04456)


  Dimension reduction (DR) techniques such as t-SNE, UMAP, and TriMAP have
demonstrated impressive visualization performance on many real world datasets.
One tension that has always faced these methods is the trade-off between
preservation of global structure and preservation of local structure: these
methods can either handle one or the other, but not both. In this work, our
main goal is to understand what aspects of DR methods are important for
preserving both local and global structure: it is difficult to design a better
method without a true understanding of the choices we make in our algorithms
and their empirical impact on the lower-dimensional embeddings they produce.
Towards the goal of local structure preservation, we provide several useful
design principles for DR loss functions based on our new understanding of the
mechanisms behind successful DR methods. Towards the goal of global structure
preservation, our analysis illuminates that the choice of which components to
preserve is important. We leverage these insights to design a new algorithm for
DR, called Pairwise Controlled Manifold Approximation Projection (PaCMAP),
which preserves both local and global structure. Our work provides several
unexpected insights into what design choices both to make and avoid when
constructing DR algorithms.

    

### [[2102.08075] Axial Residual Networks for CycleGAN-based Voice Conversion](http://arxiv.org/abs/2102.08075)


  We propose a novel architecture and improved training objectives for
non-parallel voice conversion. Our proposed CycleGAN-based model performs a
shape-preserving transformation directly on a high frequency-resolution
magnitude spectrogram, converting its style (i.e. speaker identity) while
preserving the speech content. Throughout the entire conversion process, the
model does not resort to compressed intermediate representations of any sort
(e.g. mel spectrogram, low resolution spectrogram, decomposed network feature).
We propose an efficient axial residual block architecture to support this
expensive procedure and various modifications to the CycleGAN losses to
stabilize the training process. We demonstrate via experiments that our
proposed model outperforms Scyclone and shows a comparable or better
performance to that of CycleGAN-VC2 even without employing a neural vocoder.

    

### [[2102.10271] Meta-Learning Dynamics Forecasting Using Task Inference](http://arxiv.org/abs/2102.10271)


  Current deep learning models for dynamics forecasting struggle with
generalization. They can only forecast in a specific domain and fail when
applied to systems with different parameters, external forces, or boundary
conditions. We propose a model-based meta-learning method called DyAd which can
generalize across heterogeneous domains by partitioning them into different
tasks. DyAd has two parts: an encoder which infers the time-invariant hidden
features of the task with weak supervision, and a forecaster which learns the
shared dynamics of the entire domain. The encoder adapts and controls the
forecaster during inference using adaptive instance normalization and adaptive
padding. Theoretically, we prove that the generalization error of such
procedure is related to the task relatedness in the source domain, as well as
the domain differences between source and target. Experimentally, we
demonstrate that our model outperforms state-of-the-art approaches on both
turbulent flow and real-world ocean data forecasting tasks.

    

### [[2103.01033] Tax Evasion Risk Management Using a Hybrid Unsupervised Outlier Detection Method](http://arxiv.org/abs/2103.01033)


  Big data methods are becoming an important tool for tax fraud detection
around the world. Unsupervised learning approach is the dominant framework due
to the lack of label and ground truth in corresponding data sets although these
methods suffer from low interpretability. HUNOD, a novel hybrid unsupervised
outlier detection method for tax evasion risk management, is presented in this
paper. In contrast to previous methods proposed in the literature, the HUNOD
method combines two outlier detection approaches based on two different machine
learning designs (i.e, clustering and representational learning) to detect and
internally validate outliers in a given tax dataset. The HUNOD method allows
its users to incorporate relevant domain knowledge into both constituent
outlier detection approaches in order to detect outliers relevant for a given
economic context. The interpretability of obtained outliers is achieved by
training explainable-by-design surrogate models over results of unsupervised
outlier detection methods. The experimental evaluation of the HUNOD method is
conducted on two datasets derived from the database on individual personal
income tax declarations collected by the Tax Administration of Serbia. The
obtained results show that the method indicates between 90% and 98% internally
validated outliers depending on the clustering configuration and employed
regularization mechanisms for representational learning.

    

### [[2103.01061] A Hybrid Quantum-Classical Hamiltonian Learning Algorithm](http://arxiv.org/abs/2103.01061)


  Hamiltonian learning is crucial to the certification of quantum devices and
quantum simulators. In this paper, we propose a hybrid quantum-classical
Hamiltonian learning algorithm to find the coefficients of the Pauli operator
components of the Hamiltonian. Its main subroutine is the practical
log-partition function estimation algorithm, which is based on the minimization
of the free energy of the system. Concretely, we devise a stochastic
variational quantum eigensolver (SVQE) to diagonalize the Hamiltonians and then
exploit the obtained eigenvalues to compute the free energy's global minimum
using convex optimization. Our approach not only avoids the challenge of
estimating von Neumann entropy in free energy minimization, but also reduces
the quantum resources via importance sampling in Hamiltonian diagonalization,
facilitating the implementation of our method on near-term quantum devices.
Finally, we demonstrate our approach's validity by conducting numerical
experiments with Hamiltonians of interest in quantum many-body physics.

    

### [[2103.01550] Label-Imbalanced and Group-Sensitive Classification under Overparameterization](http://arxiv.org/abs/2103.01550)


  The goal in label-imbalanced and group-sensitive classification is to
optimize relevant metrics such as balanced error and equal opportunity.
Classical methods, such as weighted cross-entropy, fail when used with the
modern practice of training deep nets to the terminal phase of training(TPT),
that is training beyond zero training error. This observation has motivated
recent flurry of activity in developing heuristic alternatives following the
intuitive mechanism of promoting larger margin for minorities. In contrast to
previous heuristics, we follow a principled analysis explaining how different
loss adjustments affect margins. First, we prove that for all linear
classifiers trained in TPT, it is necessary to introduce multiplicative, rather
than additive, logit adjustments so that the relative margins between classes
change appropriately. To show this, we discover a connection of the
multiplicative CE modification to the so-called cost-sensitive support-vector
machines. Perhaps counterintuitively, we also find that, at the start of the
training, the same multiplicative weights can actually harm the minority
classes. Thus, while additive adjustments are ineffective in the TPT, we show
numerically that they can speed up convergence by countering the initial
negative effect of the multiplicative weights. Motivated by these findings, we
formulate the vector-scaling(VS) loss, that captures existing techniques as
special cases. Moreover, we introduce a natural extension of the VS-loss to
group-sensitive classification, thus treating the two common types of
imbalances (label/group) in a unifying way. Importantly, our experiments on
state-of-the-art datasets are fully consistent with our theoretical insights
and confirm the superior performance of our algorithms. Finally, for imbalanced
Gaussian-mixtures data, we perform a generalization analysis, revealing
tradeoffs between different metrics.

    

### [[2103.01830] Audio scene monitoring using redundant ad-hoc microphone array networks](http://arxiv.org/abs/2103.01830)


  We present a system for localizing sound sources in a room with several
ad-hoc microphone arrays. Each circular array performs direction of arrival
(DOA) estimation independently using commercial software. The DOAs are fed to a
fusion center, concatenated, and used to perform the localization based on two
proposed methods, which require only few labeled source locations (anchor
points) for training. The first proposed method is based on principal component
analysis (PCA) of the observed DOA and does not require any knowledge of anchor
points. The array cluster can then perform localization on a manifold defined
by the PCA of concatenated DOAs over time. The second proposed method performs
localization using an affine transformation between the DOA vectors and the
room manifold. The PCA has fewer requirements on the training sequence, but is
less robust to missing DOAs from one of the arrays. The methods are
demonstrated with five IoT 8-microphone circular arrays, placed at unspecified
fixed locations in an office. Both the PCA and the affine method can easily map
out a rectangle based on a few anchor points with similar accuracy. The
proposed methods provide a step towards monitoring activities in a smart home
and require little installation effort as the array locations are not needed.

    

### [[2103.02565] GLAMOUR: Graph Learning over Macromolecule Representations](http://arxiv.org/abs/2103.02565)


  The near-infinite chemical diversity of natural and artificial macromolecules
arises from the vast range of possible component monomers, linkages, and
polymers topologies. This enormous variety contributes to the ubiquity and
indispensability of macromolecules but hinders the development of general
machine learning methods with macromolecules as input. To address this, we
developed GLAMOUR, a framework for chemistry-informed graph representation of
macromolecules that enables quantifying structural similarity, and
interpretable supervised learning for macromolecules.

    

### [[2103.05243] On the Generalization Power of Overfitted Two-Layer Neural Tangent Kernel Models](http://arxiv.org/abs/2103.05243)


  In this paper, we study the generalization performance of min $\ell_2$-norm
overfitting solutions for the neural tangent kernel (NTK) model of a two-layer
neural network with ReLU activation that has no bias term. We show that,
depending on the ground-truth function, the test error of overfitted NTK models
exhibits characteristics that are different from the "double-descent" of other
overparameterized linear models with simple Fourier or Gaussian features.
Specifically, for a class of learnable functions, we provide a new upper bound
of the generalization error that approaches a small limiting value, even when
the number of neurons $p$ approaches infinity. This limiting value further
decreases with the number of training samples $n$. For functions outside of
this class, we provide a lower bound on the generalization error that does not
diminish to zero even when $n$ and $p$ are both large.

    

### [[2103.06132] MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks](http://arxiv.org/abs/2103.06132)


  Recent strategies achieved ensembling "for free" by fitting concurrently
diverse subnetworks inside a single base network. The main idea during training
is that each subnetwork learns to classify only one of the multiple inputs
simultaneously provided. However, the question of how to best mix these
multiple inputs has not been studied so far. In this paper, we introduce MixMo,
a new generalized framework for learning multi-input multi-output deep
subnetworks. Our key motivation is to replace the suboptimal summing operation
hidden in previous approaches by a more appropriate mixing mechanism. For that
purpose, we draw inspiration from successful mixed sample data augmentations.
We show that binary mixing in features - particularly with rectangular patches
from CutMix - enhances results by making subnetworks stronger and more diverse.
We improve state of the art for image classification on CIFAR-100 and Tiny
ImageNet datasets. Our easy to implement models notably outperform data
augmented deep ensembles, without the inference and memory overheads. As we
operate in features and simply better leverage the expressiveness of large
networks, we open a new line of research complementary to previous works.

    

### [[2103.06685] Frame-independent vector-cloud neural network for nonlocal constitutive modeling on arbitrary grids](http://arxiv.org/abs/2103.06685)


  Constitutive models are widely used for modeling complex systems in science
and engineering, where first-principle-based, well-resolved simulations are
often prohibitively expensive. For example, in fluid dynamics, constitutive
models are required to describe nonlocal, unresolved physics such as turbulence
and laminar-turbulent transition. However, traditional constitutive models
based on partial differential equations (PDEs) often lack robustness and are
too rigid to accommodate diverse calibration datasets. We propose a
frame-independent, nonlocal constitutive model based on a vector-cloud neural
network that can be learned with data. The model predicts the closure variable
at a point based on the flow information in its neighborhood. Such nonlocal
information is represented by a group of points, each having a feature vector
attached to it, and thus the input is referred to as vector cloud. The cloud is
mapped to the closure variable through a frame-independent neural network,
invariant both to coordinate translation and rotation and to the ordering of
points in the cloud. As such, the network can deal with any number of
arbitrarily arranged grid points and thus is suitable for unstructured meshes
in fluid simulations. The merits of the proposed network are demonstrated for
scalar transport PDEs on a family of parameterized periodic hill geometries.
The vector-cloud neural network is a promising tool not only as nonlocal
constitutive models and but also as general surrogate models for PDEs on
irregular domains.

    

### [[2103.14172] Deep-RBF Networks for Anomaly Detection in Automotive Cyber-Physical Systems](http://arxiv.org/abs/2103.14172)


  Deep Neural Networks (DNNs) are popularly used for implementing autonomy
related tasks in automotive Cyber-Physical Systems (CPSs). However, these
networks have been shown to make erroneous predictions to anomalous inputs,
which manifests either due to Out-of-Distribution (OOD) data or adversarial
attacks. To detect these anomalies, a separate DNN called assurance monitor is
often trained and used in parallel to the controller DNN, increasing the
resource burden and latency. We hypothesize that a single network that can
perform controller predictions and anomaly detection is necessary to reduce the
resource requirements. Deep-Radial Basis Function (RBF) networks provide a
rejection class alongside the class predictions, which can be utilized for
detecting anomalies at runtime. However, the use of RBF activation functions
limits the applicability of these networks to only classification tasks. In
this paper, we show how the deep-RBF network can be used for detecting
anomalies in CPS regression tasks such as continuous steering predictions.
Further, we design deep-RBF networks using popular DNNs such as NVIDIA DAVE-II,
and ResNet20, and then use the resulting rejection class for detecting
adversarial attacks such as a physical attack and data poison attack. Finally,
we evaluate these attacks and the trained deep-RBF networks using a hardware
CPS testbed called DeepNNCar and a real-world German Traffic Sign Benchmark
(GTSB) dataset. Our results show that the deep-RBF networks can robustly detect
these attacks in a short time without additional resource requirements.

    

### [[2104.10771] Aedes-AI: Neural Network Models of Mosquito Abundance](http://arxiv.org/abs/2104.10771)


  We present artificial neural networks as a feasible replacement for a
mechanistic model of mosquito abundance. We develop a feed-forward neural
network, a long short-term memory recurrent neural network, and a gated
recurrent unit network. We evaluate the networks in their ability to replicate
the spatiotemporal features of mosquito populations predicted by the
mechanistic model, and discuss how augmenting the training data with time
series that emphasize specific dynamical behaviors affects model performance.
We conclude with an outlook on how such equation-free models may facilitate
vector control or the estimation of disease risk at arbitrary spatial scales.

    

### [[2106.03010] An Adaptive Framework for Learning Unsupervised Depth Completion](http://arxiv.org/abs/2106.03010)


  We present a method to infer a dense depth map from a color image and
associated sparse depth measurements. Our main contribution lies in the design
of an annealing process for determining co-visibility (occlusions,
disocclusions) and the degree of regularization to impose on the model. We show
that regularization and co-visibility are related via the fitness (residual) of
model to data and both can be unified into a single framework to improve the
learning process. Our method is an adaptive weighting scheme that guides
optimization by measuring the residual at each pixel location over each
training step for (i) estimating a soft visibility mask and (ii) determining
the amount of regularization. We demonstrate the effectiveness our method by
applying it to several recent unsupervised depth completion methods and
improving their performance on public benchmark datasets, without incurring
additional trainable parameters or increase in inference time. Code available
at: this https URL.

    

### [[2106.06134] Is Homophily a Necessity for Graph Neural Networks?](http://arxiv.org/abs/2106.06134)


  Graph neural networks (GNNs) have shown great prowess in learning
representations suitable for numerous graph-based machine learning tasks. When
applied to semi-supervised node classification, GNNs are widely believed to
work well due to the homophily assumption ("like attracts like"), and fail to
generalize to heterophilous graphs where dissimilar nodes connect. Recent works
design new architectures to overcome such heterophily-related limitations,
citing poor baseline performance and new architecture improvements on a few
heterophilous graph benchmark datasets as evidence for this notion. In our
experiments, we empirically find that standard graph convolutional networks
(GCNs) can actually achieve better performance than such carefully designed
methods on some commonly used heterophilous graphs. This motivates us to
reconsider whether homophily is truly necessary for good GNN performance. We
find that this claim is not quite true, and in fact, GCNs can achieve strong
performance on heterophilous graphs under certain conditions. Our work
carefully characterizes these conditions, and provides supporting theoretical
understanding and empirical observations. Finally, we examine existing
heterophilous graphs benchmarks and reconcile how the GCN (under)performs on
them based on this understanding.

    

### [[2106.08693] ParticleAugment: Sampling-Based Data Augmentation](http://arxiv.org/abs/2106.08693)


  We present an automated data augmentation approach for image classification.
We formulate the problem as Monte Carlo sampling where our goal is to
approximate the optimal augmentation policies. We propose a particle filtering
scheme for the policy search where the probability of applying a set of
augmentation operations forms the state of the filter. We measure the policy
performance based on the loss function difference between a reference and the
actual model, which we afterwards use to re-weight the particles and finally
update the policy. In our experiments, we show that our formulation for
automated augmentation reaches promising results on CIFAR-10, CIFAR-100, and
ImageNet datasets using the standard network architectures for this problem. By
comparing with the related work, our method reaches a balance between the
computational cost of policy search and the model performance. Our code will be
made publicly available.

    

### [[2106.10165] The Principles of Deep Learning Theory](http://arxiv.org/abs/2106.10165)


  This book develops an effective theory approach to understanding deep neural
networks of practical relevance. Beginning from a first-principles
component-level picture of networks, we explain how to determine an accurate
description of the output of trained networks by solving layer-to-layer
iteration equations and nonlinear learning dynamics. A main result is that the
predictions of networks are described by nearly-Gaussian distributions, with
the depth-to-width aspect ratio of the network controlling the deviations from
the infinite-width Gaussian description. We explain how these effectively-deep
networks learn nontrivial representations from training and more broadly
analyze the mechanism of representation learning for nonlinear models. From a
nearly-kernel-methods perspective, we find that the dependence of such models'
predictions on the underlying learning algorithm can be expressed in a simple
and universal way. To obtain these results, we develop the notion of
representation group flow (RG flow) to characterize the propagation of signals
through the network. By tuning networks to criticality, we give a practical
solution to the exploding and vanishing gradient problem. We further explain
how RG flow leads to near-universal behavior and lets us categorize networks
built from different activation functions into universality classes.
Altogether, we show that the depth-to-width ratio governs the effective model
complexity of the ensemble of trained networks. By using information-theoretic
techniques, we estimate the optimal aspect ratio at which we expect the network
to be practically most useful and show how residual connections can be used to
push this scale to arbitrary depths. With these tools, we can learn in detail
about the inductive bias of architectures, hyperparameters, and optimizers.

    

### [[2106.12045] Deep learning for improved global precipitation in numerical weather prediction systems](http://arxiv.org/abs/2106.12045)


  The formation of precipitation in state-of-the-art weather and climate models
is an important process. The understanding of its relationship with other
variables can lead to endless benefits, particularly for the world's monsoon
regions dependent on rainfall as a support for livelihood. Various factors play
a crucial role in the formation of rainfall, and those physical processes are
leading to significant biases in the operational weather forecasts. We use the
UNET architecture of a deep convolutional neural network with residual learning
as a proof of concept to learn global data-driven models of precipitation. The
models are trained on reanalysis datasets projected on the cubed-sphere
projection to minimize errors due to spherical distortion. The results are
compared with the operational dynamical model used by the India Meteorological
Department. The theoretical deep learning-based model shows doubling of the
grid point, as well as area averaged skill measured in Pearson correlation
coefficients relative to operational system. This study is a proof-of-concept
showing that residual learning-based UNET can unravel physical relationships to
target precipitation, and those physical constraints can be used in the
dynamical operational models towards improved precipitation forecasts. Our
results pave the way for the development of online, hybrid models in the
future.

    

### [[2106.12901] Recurrent Neural Network from Adder's Perspective: Carry-lookahead RNN](http://arxiv.org/abs/2106.12901)


  The recurrent network architecture is a widely used model in sequence
modeling, but its serial dependency hinders the computation parallelization,
which makes the operation inefficient. The same problem was encountered in
serial adder at the early stage of digital electronics. In this paper, we
discuss the similarities between recurrent neural network (RNN) and serial
adder. Inspired by carry-lookahead adder, we introduce carry-lookahead module
to RNN, which makes it possible for RNN to run in parallel. Then, we design the
method of parallel RNN computation, and finally Carry-lookahead RNN (CL-RNN) is
proposed. CL-RNN takes advantages in parallelism and flexible receptive field.
Through a comprehensive set of tests, we verify that CL-RNN can perform better
than existing typical RNNs in sequence modeling tasks which are specially
designed for RNNs.

    

### [[2012.07828] Robustness Threats of Differential Privacy](http://arxiv.org/abs/2012.07828)


  Differential privacy (DP) is a gold-standard concept of measuring and
guaranteeing privacy in data analysis. It is well-known that the cost of adding
DP to deep learning model is its accuracy. However, it remains unclear how it
affects robustness of the model. Standard neural networks are not robust to
different input perturbations: either adversarial attacks or common
corruptions. In this paper, we empirically observe an interesting trade-off
between privacy and robustness of neural networks. We experimentally
demonstrate that networks, trained with DP, in some settings might be even more
vulnerable in comparison to non-private versions. To explore this, we
extensively study different robustness measurements, including FGSM and PGD
adversaries, distance to linear decision boundaries, curvature profile, and
performance on a corrupted dataset. Finally, we study how the main ingredients
of differentially private neural networks training, such as gradient clipping
and noise addition, affect (decrease and increase) the robustness of the model.

    

### [[2108.10570] METRO: A Software-Hardware Co-Design of Interconnections for Spatial DNN Accelerators](http://arxiv.org/abs/2108.10570)


  Tiled spatial architectures have proved to be an effective solution to build
large-scale DNN accelerators. In particular, interconnections between tiles are
critical for high performance in these tile-based architectures. In this work,
we identify the inefficiency of the widely used traditional on-chip networks
and the opportunity of software-hardware co-design. We propose METRO with the
basic idea of decoupling the traffic scheduling policies from hardware fabrics
and moving them to the software level. METRO contains two modules working in
synergy: METRO software scheduling framework to coordinate the traffics and
METRO hardware facilities to deliver the data based on software configurations.
We evaluate the co-design using different flit sizes for synthetic study,
illustrating its effectiveness under various hardware resource constraints, in
addition to a wide range of DNN models selected from real-world workloads. The
results show that METRO achieves 56.3% communication speedup on average and up
to 73.6% overall processing time reduction compared with traditional on-chip
network designs.

    

### [[2108.10771] Transient Execution of Non-Canonical Accesses](http://arxiv.org/abs/2108.10771)


  Recent years have brought microarchitectural security intothe spotlight,
proving that modern CPUs are vulnerable toseveral classes of microarchitectural
attacks. These attacksbypass the basic isolation primitives provided by the
CPUs:process isolation, memory permissions, access checks, andso on.
Nevertheless, most of the research was focused on In-tel CPUs, with only a few
exceptions. As a result, few vulner-abilities have been found in other CPUs,
leading to specula-tions about their immunity to certain types of
microarchi-tectural attacks. In this paper, we provide a black-box anal-ysis of
one of these under-explored areas. Namely, we inves-tigate the flaw of AMD CPUs
which may lead to a transientexecution hijacking attack. Contrary to nominal
immunity,we discover that AMD Zen family CPUs exhibit transient ex-ecution
patterns similar for Meltdown/MDS. Our analysisof exploitation possibilities
shows that AMDs design deci-sions indeed limit the exploitability scope
comparing to In-tel CPUs, yet it may be possible to use them to amplify
othermicroarchitectural attacks.

    

### [[2108.10464] The Case for Task Sampling based Learning for Cluster Job Scheduling](http://arxiv.org/abs/2108.10464)


  The ability to accurately estimate job runtime properties allows a scheduler
to effectively schedule jobs. State-of-the-art online cluster job schedulers
use history-based learning, which uses past job execution information to
estimate the runtime properties of newly arrived jobs. However, with fast-paced
development in cluster technology (in both hardware and software) and changing
user inputs, job runtime properties can change over time, which lead to
inaccurate predictions. In this paper, we explore the potential and limitation
of real-time learning of job runtime properties, by proactively sampling and
scheduling a small fraction of the tasks of each job. Such a
task-sampling-based approach exploits the similarity among runtime properties
of the tasks of the same job and is inherently immune to changing job behavior.
Our study focuses on two key questions in comparing task-sampling-based
learning (learning in space) and history-based learning (learning in time): (1)
Can learning in space be more accurate than learning in time? (2) If so, can
delaying scheduling the remaining tasks of a job till the completion of sampled
tasks be more than compensated by the improved accuracy and result in improved
job performance? Our analytical and experimental analysis of 3 production
traces with different skew and job distribution shows that learning in space
can be substantially more accurate. Our simulation and testbed evaluation on
Azure of the two learning approaches anchored in a generic job scheduler using
3 production cluster job traces shows that despite its online overhead,
learning in space reduces the average Job Completion Time (JCT) by 1.28x,
1.56x, and 1.32x compared to the prior-art history-based predictor.

    

### [[2108.10496] The benefits of prefetching for large-scale cloud-based neuroimaging analysis workflows](http://arxiv.org/abs/2108.10496)


  To support the growing demands of neuroscience applications, researchers are
transitioning to cloud computing for its scalable, robust and elastic
infrastructure. Nevertheless, large datasets residing in object stores may
result in significant data transfer overheads during workflow execution.
Prefetching, a method to mitigate the cost of reading in mixed workloads, masks
data transfer costs within processing time of prior tasks. We present an
implementation of "Rolling Prefetch", a Python library that implements a
particular form of prefetching from AWS S3 object store, and we quantify its
benefits.
Rolling Prefetch extends S3Fs, a Python library exposing AWS S3 functionality
via a file object, to add prefetch capabilities. In measured analysis
performance of a 500 GB brain connectivity dataset stored on S3, we found that
prefetching provides significant speed-ups of up to 1.86x, even in applications
consisting entirely of data loading. The observed speed-up values are
consistent with our theoretical analysis. Our results demonstrate the
usefulness of prefetching for scientific data processing on cloud
infrastructures and provide an implementation applicable to various application
domains.

    

### [[2108.10565] An Efficient ADER-DG Local Time Stepping Scheme for 3D HPC Simulation of Seismic Waves in Poroelastic Media](http://arxiv.org/abs/2108.10565)


  Many applications from geosciences require simulations of seismic waves in
porous media. Biot's theory of poroelasticity describes the coupling between
solid and fluid phases and introduces a stiff source term, thereby increasing
computational cost and motivating efficient methods utilising High-Performance
Computing. We present a novel realisation of the discontinuous Galerkin scheme
with Arbitrary DERivative time stepping (ADER-DG) that copes with stiff source
terms.
To integrate this source term with a reasonable time step size, we use an
element-local space-time predictor, which needs to solve medium-sized linear
systems - with 1000 to 10000 unknowns - in each element update (i.e., billions
of times). We present a novel block-wise back-substitution algorithm for
solving these systems efficiently. In comparison to LU decomposition, we reduce
the number of floating-point operations by a factor of up to 25. The block-wise
back-substitution is mapped to a sequence of small matrix-matrix
multiplications, for which code generators are available to generate highly
optimised code.
We verify the new solver thoroughly in problems of increasing complexity. We
demonstrate high-order convergence for 3D problems. We verify the correct
treatment of point sources, material interfaces and traction-free boundary
conditions. In addition, we compare against a finite difference code for a
newly defined layer over half-space problem. We find that extremely high
accuracy is required to resolve the slow P-wave at a free surface, while solid
particle velocities are not affected by coarser resolutions. By using a
clustered local time stepping scheme, we reduce time to solution by a factor of
6 to 10 compared to global time stepping. We conclude our study with a scaling
and performance analysis, demonstrating our implementation's efficiency and its
potential for extreme-scale simulations.

    

### [[2108.10591] Communication-hiding pipelined BiCGSafe methods for solving large linear systems](http://arxiv.org/abs/2108.10591)


  Recently, a new variant of the BiCGStab method, known as the pipeline
BiCGStab, has been proposed. This method can achieve a higher degree of
scalability and speed-up rates through a mechanism in which the communication
phase for the computation of the inner product can be overlapped with the
computation of the matrix-vector product. On the other hand, there exist
several generalized iteration methods with better convergence behavior than
BiCGStab such as ssBiCGSafe, BiCGSafe, GPBi-CG. Of these methods, ssBiCGSafe,
which requires a single phase of computing inner products per one iteration, is
best suited for high-performance computing systems. In this paper, inspired by
the success of the pipelined BiCGStab method, we propose variations of the
ssBiCGSafe method, in which only one phase of inner product computation per
iteration is required and this inner product computation phase can be
overlapped with the matrix-vector computation. Through numerical experiments,
we show that the proposed methods lead to improvements in convergence behavior
and execution time compared to the pipelined BiCGStab and ssBiCGSafe methods.

    

### [[2108.10628] Towards Predictive Replica Placement for Distributed Data Stores in Fog Environments](http://arxiv.org/abs/2108.10628)


  Mobile clients that consume and produce data are abundant in fog
environments. Low latency access to this data can only be achieved by storing
it in close physical proximity to the clients. Current data store systems fall
short as they do not replicate data based on client movement. We propose an
approach to predictive replica placement that autonomously and proactively
replicates data close to likely client locations.

    

### [[2108.10721] Dependable IoT Data Stream Processing for Monitoring and Control of Urban Infrastructures](http://arxiv.org/abs/2108.10721)


  The Internet of Things describes a network of physical devices interacting
and producing vast streams of sensor data. At present there are a number of
general challenges which exist while developing solutions for use cases
involving the monitoring and control of urban infrastructures. These include
the need for a dependable method for extracting value from these high volume
streams of time sensitive data which is adaptive to changing workloads.
Low-latency access to the current state for live monitoring is a necessity as
well as the ability to perform queries on historical data. At the same time,
many design choices need to be made and the number of possible technology
options available further adds to the complexity.
In this paper we present a dependable IoT data processing platform for the
monitoring and control of urban infrastructures. We define requirements in
terms of dependability and then select a number of mature open-source
technologies to match these requirements. We examine the disparate parts
necessary for delivering a holistic overall architecture and describe the
dataflows between each of these components. We likewise present generalizable
methods for the enrichment and analysis of sensor data applicable across
various application areas. We demonstrate the usefulness of this approach by
providing an exemplary prototype platform executing on top of Kubernetes and
evaluate the effectiveness of jobs processing sensor data in this environment.

    

### [[2008.03909] ConnectIt: A Framework for Static and Incremental Parallel Graph Connectivity Algorithms](http://arxiv.org/abs/2008.03909)


  Connected components is a fundamental kernel in graph applications. The
fastest existing parallel multicore algorithms for connectivity are based on
some form of edge sampling and/or linking and compressing trees. However, many
combinations of these design choices have been left unexplored. In this paper,
we design the ConnectIt framework, which provides different sampling strategies
as well as various tree linking and compression schemes. ConnectIt enables us
to obtain several hundred new variants of connectivity algorithms, most of
which extend to computing spanning forest. In addition to static graphs, we
also extend ConnectIt to support mixes of insertions and connectivity queries
in the concurrent setting.
We present an experimental evaluation of ConnectIt on a 72-core machine,
which we believe is the most comprehensive evaluation of parallel connectivity
algorithms to date. Compared to a collection of state-of-the-art static
multicore algorithms, we obtain an average speedup of 12.4x (2.36x average
speedup over the fastest existing implementation for each graph). Using
ConnectIt, we are able to compute connectivity on the largest
publicly-available graph (with over 3.5 billion vertices and 128 billion edges)
in under 10 seconds using a 72-core machine, providing a 3.1x speedup over the
fastest existing connectivity result for this graph, in any computational
setting. For our incremental algorithms, we show that our algorithms can ingest
graph updates at up to several billion edges per second. To guide the user in
selecting the best variants in ConnectIt for different situations, we provide a
detailed analysis of the different strategies. Finally, we show how the
techniques in ConnectIt can be used to speed up two important graph
applications: approximate minimum spanning forest and SCAN clustering.

    

### [[2011.12879] Characterization and Derivation of Heard-Of Predicates for Asynchronous Message-Passing Models](http://arxiv.org/abs/2011.12879)


  In distributed computing, multiple processes interact to solve a problem
together. The main model of interaction is the message-passing model, where
processes communicate by exchanging messages. Nevertheless, there are several
models varying along important dimensions: degree of synchrony, kinds of
faults, number of faults... This variety is compounded by the lack of a general
formalism in which to abstract these models. One way to bring order is to
constrain these models to communicate in rounds. This is the setting of the
Heard-Of model, which captures many models through predicates on the messages
sent in a round and received on time. Yet, it is not easy to define the
predicate that captures a given operational model. The question is even harder
for the asynchronous case, as unbounded message delay means the implementation
of rounds must depend on details of the model. This paper shows that
characterising asynchronous models by heard-of predicates is indeed meaningful.
This characterization relies on delivered predicates, an intermediate
abstraction between the informal operational model and the heard-of predicates.
Our approach splits the problem into two steps: first extract the delivered
model capturing the informal model, and then characterize the heard-of
predicates that are generated by this delivered model. For the first part, we
provide examples of delivered predicates, and an approach to derive more. It
uses the intuition that complex models are a composition of simpler models. We
define operations like union, succession or repetition that make it easier to
derive complex delivered predicates from simple ones while retaining
expressivity. For the second part, we formalize and study strategies for when
to change rounds. Intuitively, the characterizing predicate of a model is the
one generated by a strategy that waits for as much messages as possible,
without blocking forever.

    

### [[2108.10315] Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control](http://arxiv.org/abs/2108.10315)


  In this paper we aim to provide analysis and insights (often based on
visualization), which explain the beneficial effects of on-line decision making
on top of off-line training. In particular, through a unifying abstract
mathematical framework, we show that the principal AlphaZero/TD-Gammon ideas of
approximation in value space and rollout apply very broadly to deterministic
and stochastic optimal control problems, involving both discrete and continuous
search spaces. Moreover, these ideas can be effectively integrated with other
important methodologies such as model predictive control, adaptive control,
decentralized control, discrete and Bayesian optimization, neural network-based
value and policy approximations, and heuristic algorithms for discrete
optimization.

    

### [[2108.10363] Knowledge-based XAI through CBR: There is more to explanations than models can tell](http://arxiv.org/abs/2108.10363)


  The underlying hypothesis of knowledge-based explainable artificial
intelligence is the data required for data-centric artificial intelligence
agents (e.g., neural networks) are less diverse in contents than the data
required to explain the decisions of such agents to humans. The idea is that a
classifier can attain high accuracy using data that express a phenomenon from
one perspective whereas the audience of explanations can entail multiple
stakeholders and span diverse perspectives. We hence propose to use domain
knowledge to complement the data used by agents. We formulate knowledge-based
explainable artificial intelligence as a supervised data classification problem
aligned with the CBR methodology. In this formulation, the inputs are case
problems composed of both the inputs and outputs of the data-centric agent and
case solutions, the outputs, are explanation categories obtained from domain
knowledge and subject matter experts. This formulation does not typically lead
to an accurate classification, preventing the selection of the correct
explanation category. Knowledge-based explainable artificial intelligence
extends the data in this formulation by adding features aligned with domain
knowledge that can increase accuracy when selecting explanation categories.

    

### [[2108.10399] Learning Motion Priors for 4D Human Body Capture in 3D Scenes](http://arxiv.org/abs/2108.10399)


  Recovering high-quality 3D human motion in complex scenes from monocular
videos is important for many applications, ranging from AR/VR to robotics.
However, capturing realistic human-scene interactions, while dealing with
occlusions and partial views, is challenging; current approaches are still far
from achieving compelling results. We address this problem by proposing LEMO:
LEarning human MOtion priors for 4D human body capture. By leveraging the
large-scale motion capture dataset AMASS, we introduce a novel motion
smoothness prior, which strongly reduces the jitters exhibited by poses
recovered over a sequence. Furthermore, to handle contacts and occlusions
occurring frequently in body-scene interactions, we design a contact friction
term and a contact-aware motion infiller obtained via per-instance
self-supervised training. To prove the effectiveness of the proposed motion
priors, we combine them into a novel pipeline for 4D human body capture in 3D
scenes. With our pipeline, we demonstrate high-quality 4D human body capture,
reconstructing smooth motions and physically plausible body-scene interactions.
The code and data are available at this https URL.

    

### [[2108.10437] Longitudinal Distance: Towards Accountable Instance Attribution](http://arxiv.org/abs/2108.10437)


  Previous research in interpretable machine learning (IML) and explainable
artificial intelligence (XAI) can be broadly categorized as either focusing on
seeking interpretability in the agent's model (i.e., IML) or focusing on the
context of the user in addition to the model (i.e., XAI). The former can be
categorized as feature or instance attribution. Example- or sample-based
methods such as those using or inspired by case-based reasoning (CBR) rely on
various approaches to select instances that are not necessarily attributing
instances responsible for an agent's decision. Furthermore, existing approaches
have focused on interpretability and explainability but fall short when it
comes to accountability. Inspired in case-based reasoning principles, this
paper introduces a pseudo-metric we call Longitudinal distance and its use to
attribute instances to a neural network agent's decision that can be
potentially used to build accountable CBR agents.

    

### [[2108.10446] All You Need is Color: Image based Spatial Gene Expression Prediction using Neural Stain Learning](http://arxiv.org/abs/2108.10446)


  "Is it possible to predict expression levels of different genes at a given
spatial location in the routine histology image of a tumor section by modeling
its stain absorption characteristics?" In this work, we propose a "stain-aware"
machine learning approach for prediction of spatial transcriptomic gene
expression profiles using digital pathology image of a routine Hematoxylin &
Eosin (H&E) histology section. Unlike recent deep learning methods which are
used for gene expression prediction, our proposed approach termed Neural Stain
Learning (NSL) explicitly models the association of stain absorption
characteristics of the tissue with gene expression patterns in spatial
transcriptomics by learning a problem-specific stain deconvolution matrix in an
end-to-end manner. The proposed method with only 11 trainable weight parameters
outperforms both classical regression models with cellular composition and
morphological features as well as deep learning methods. We have found that the
gene expression predictions from the proposed approach show higher correlations
with true expression values obtained through sequencing for a larger set of
genes in comparison to other approaches.

    

### [[2108.10449] Differential Music: Automated Music Generation Using LSTM Networks with Representation Based on Melodic and Harmonic Intervals](http://arxiv.org/abs/2108.10449)


  This paper presents a generative AI model for automated music composition
with LSTM networks that takes a novel approach at encoding musical information
which is based on movement in music rather than absolute pitch. Melodies are
encoded as a series of intervals rather than a series of pitches, and chords
are encoded as the set of intervals that each chord note makes with the melody
at each timestep. Experimental results show promise as they sound musical and
tonal. There are also weaknesses to this method, mainly excessive modulations
in the compositions, but that is expected from the nature of the encoding. This
issue is discussed later in the paper and is a potential topic for future work.

    

### [[2108.10511] CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation](http://arxiv.org/abs/2108.10511)


  Practical recommender systems experience a cold-start problem when observed
user-item interactions in the history are insufficient. Meta learning,
especially gradient based one, can be adopted to tackle this problem by
learning initial parameters of the model and thus allowing fast adaptation to a
specific task from limited data examples. Though with significant performance
improvement, it commonly suffers from two critical issues: the
non-compatibility with mainstream industrial deployment and the heavy
computational burdens, both due to the inner-loop gradient operation. These two
issues make them hard to be applied in practical recommender systems. To enjoy
the benefits of meta learning framework and mitigate these problems, we propose
a recommendation framework called Contextual Modulation Meta Learning (CMML).
CMML is composed of fully feed-forward operations so it is computationally
efficient and completely compatible with the mainstream industrial deployment.
CMML consists of three components, including a context encoder that can
generate context embedding to represent a specific task, a hybrid context
generator that aggregates specific user-item features with task-level context,
and a contextual modulation network, which can modulate the recommendation
model to adapt effectively. We validate our approach on both scenario-specific
and user-specific cold-start setting on various real-world datasets, showing
CMML can achieve comparable or even better performance with gradient based
methods yet with much higher computational efficiency and better
interpretability.

    

### [[2108.10604] Prompt-Learning for Fine-Grained Entity Typing](http://arxiv.org/abs/2108.10604)


  As an effective approach to tune pre-trained language models (PLMs) for
specific tasks, prompt-learning has recently attracted much attention from
researchers. By using \textit{cloze}-style language prompts to stimulate the
versatile knowledge of PLMs, prompt-learning can achieve promising results on a
series of NLP tasks, such as natural language inference, sentiment
classification, and knowledge probing. In this work, we investigate the
application of prompt-learning on fine-grained entity typing in fully
supervised, few-shot and zero-shot scenarios. We first develop a simple and
effective prompt-learning pipeline by constructing entity-oriented verbalizers
and templates and conducting masked language modeling. Further, to tackle the
zero-shot regime, we propose a self-supervised strategy that carries out
distribution-level optimization in prompt-learning to automatically summarize
the information of entity types. Extensive experiments on three fine-grained
entity typing benchmarks (with up to 86 classes) under fully supervised,
few-shot and zero-shot settings show that prompt-learning methods significantly
outperform fine-tuning baselines, especially when the training data is
insufficient.

    

### [[2108.10643] Morality-based Assertion and Homophily on Social Media: A Cultural Comparison between English and Japanese Languages](http://arxiv.org/abs/2108.10643)


  Moral psychology is a domain that deals with moral identity, appraisals and
emotions. Previous work has greatly focused on moral development and the
associated role of culture. Knowing that language is an inherent element of a
culture, we used the social media platform Twitter for comparing the moral
behaviors of Japanese users with English users. The five basic moral
foundations i.e., Care, Fairness, Ingroup, Authority and Purity, along with the
associated emotional valence are compared for English and Japanese tweets. The
tweets from Japanese users depicted relatively higher Fairness, Ingroup and
Purity. As far as emotions related to morality are concerned, the English
tweets expressed more positive emotions for all moral dimensions. Considering
the role of moral similarities in connecting users on social media, we
quantified homophily concerning different moral dimensions using our proposed
method. The moral dimensions Care, Authority and Purity for English and Ingroup
for Japanese depicted homophily on Twitter. Overall, our study uncovers the
underlying cultural differences with respect to moral behavior in English and
Japanese speaking users.

    

### [[2108.10709] MCUa: Multi-level Context and Uncertainty aware Dynamic Deep Ensemble for Breast Cancer Histology Image Classification](http://arxiv.org/abs/2108.10709)


  Breast histology image classification is a crucial step in the early
diagnosis of breast cancer. In breast pathological diagnosis, Convolutional
Neural Networks (CNNs) have demonstrated great success using digitized
histology slides. However, tissue classification is still challenging due to
the high visual variability of the large-sized digitized samples and the lack
of contextual information. In this paper, we propose a novel CNN, called
Multi-level Context and Uncertainty aware (MCUa) dynamic deep learning ensemble
model.MCUamodel consists of several multi-level context-aware models to learn
the spatial dependency between image patches in a layer-wise fashion. It
exploits the high sensitivity to the multi-level contextual information using
an uncertainty quantification component to accomplish a novel dynamic ensemble
model.MCUamodelhas achieved a high accuracy of 98.11% on a breast cancer
histology image dataset. Experimental results show the superior effectiveness
of the proposed solution compared to the state-of-the-art histology
classification models.

    

### [[2108.10752] Generalizing RNN-Transducer to Out-Domain Audio via Sparse Self-Attention Layers](http://arxiv.org/abs/2108.10752)


  Recurrent neural network transducers (RNN-T) are a promising end-to-end
speech recognition framework that transduces input acoustic frames into a
character sequence. The state-of-the-art encoder network for RNN-T is the
Conformer, which can effectively model the local-global context information via
its convolution and self-attention layers. Although Conformer RNN-T has shown
outstanding performance (measured by word error rate (WER) in general), most
studies have been verified in the setting where the train and test data are
drawn from the same domain. The domain mismatch problem for Conformer RNN-T has
not been intensively investigated yet, which is an important issue for the
product-level speech recognition system. In this study, we identified that
fully connected self-attention layers in the Conformer caused high deletion
errors, specifically in the long-form out-domain utterances. To address this
problem, we introduce sparse self-attention layers for Conformer-based encoder
networks, which can exploit local and generalized global information by pruning
most of the in-domain fitted global connections. Further, we propose a state
reset method for the generalization of the prediction network to cope with
long-form utterances. Applying proposed methods to an out-domain test, we
obtained 24.6\% and 6.5\% relative character error rate (CER) reduction
compared to the fully connected and local self-attention layer-based
Conformers, respectively.

    

### [[2108.10803] Reducing Exposure Bias in Training Recurrent Neural Network Transducers](http://arxiv.org/abs/2108.10803)


  When recurrent neural network transducers (RNNTs) are trained using the
typical maximum likelihood criterion, the prediction network is trained only on
ground truth label sequences. This leads to a mismatch during inference, known
as exposure bias, when the model must deal with label sequences containing
errors. In this paper we investigate approaches to reducing exposure bias in
training to improve the generalization of RNNT models for automatic speech
recognition (ASR). A label-preserving input perturbation to the prediction
network is introduced. The input token sequences are perturbed using SwitchOut
and scheduled sampling based on an additional token language model. Experiments
conducted on the 300-hour Switchboard dataset demonstrate their effectiveness.
By reducing the exposure bias, we show that we can further improve the accuracy
of a high-performance RNNT ASR model and obtain state-of-the-art results on the
300-hour Switchboard dataset.

    

### [[2108.10818] Identification of Pediatric Respiratory Diseases Using Fine-grained Diagnosis System](http://arxiv.org/abs/2108.10818)


  Respiratory diseases, including asthma, bronchitis, pneumonia, and upper
respiratory tract infection (RTI), are among the most common diseases in
clinics. The similarities among the symptoms of these diseases precludes prompt
diagnosis upon the patients' arrival. In pediatrics, the patients' limited
ability in expressing their situation makes precise diagnosis even harder. This
becomes worse in primary hospitals, where the lack of medical imaging devices
and the doctors' limited experience further increase the difficulty of
distinguishing among similar diseases. In this paper, a pediatric fine-grained
diagnosis-assistant system is proposed to provide prompt and precise diagnosis
using solely clinical notes upon admission, which would assist clinicians
without changing the diagnostic process. The proposed system consists of two
stages: a test result structuralization stage and a disease identification
stage. The first stage structuralizes test results by extracting relevant
numerical values from clinical notes, and the disease identification stage
provides a diagnosis based on text-form clinical notes and the structured data
obtained from the first stage. A novel deep learning algorithm was developed
for the disease identification stage, where techniques including adaptive
feature infusion and multi-modal attentive fusion were introduced to fuse
structured and text data together. Clinical notes from over 12000 patients with
respiratory diseases were used to train a deep learning model, and clinical
notes from a non-overlapping set of about 1800 patients were used to evaluate
the performance of the trained model. The average precisions (AP) for
pneumonia, RTI, bronchitis and asthma are 0.878, 0.857, 0.714, and 0.825,
respectively, achieving a mean AP (mAP) of 0.819.

    

### [[2108.10831] LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](http://arxiv.org/abs/2108.10831)


  It is very challenging for various visual tasks such as image fusion,
pedestrian detection and image-to-image translation in low light conditions due
to the loss of effective target areas. In this case, infrared and visible
images can be used together to provide both rich detail information and
effective target areas. In this paper, we present LLVIP, a visible-infrared
paired dataset for low-light vision. This dataset contains 33672 images, or
16836 pairs, most of which were taken at very dark scenes, and all of the
images are strictly aligned in time and space. Pedestrians in the dataset are
labeled. We compare the dataset with other visible-infrared datasets and
evaluate the performance of some popular visual algorithms including image
fusion, pedestrian detection and image-to-image translation on the dataset. The
experimental results demonstrate the complementary effect of fusion on image
information, and find the deficiency of existing algorithms of the three visual
tasks in very low-light conditions. We believe the LLVIP dataset will
contribute to the community of computer vision by promoting image fusion,
pedestrian detection and image-to-image translation in very low-light
applications. The dataset is being released in
this https URL.

    

### [[2108.10840] Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark](http://arxiv.org/abs/2108.10840)


  In recent years, deep learning-based methods have shown promising results in
computer vision area. However, a common deep learning model requires a large
amount of labeled data, which is labor-intensive to collect and label. What's
more, the model can be ruined due to the domain shift between training data and
testing data. Text recognition is a broadly studied field in computer vision
and suffers from the same problems noted above due to the diversity of fonts
and complicated backgrounds. In this paper, we focus on the text recognition
problem and mainly make three contributions toward these problems. First, we
collect a multi-source domain adaptation dataset for text recognition,
including five different domains with over five million images, which is the
first multi-domain text recognition dataset to our best knowledge. Secondly, we
propose a new method called Meta Self-Learning, which combines the
self-learning method with the meta-learning paradigm and achieves a better
recognition result under the scene of multi-domain adaptation. Thirdly,
extensive experiments are conducted on the dataset to provide a benchmark and
also show the effectiveness of our method. The code of our work and dataset are
available soon at this https URL.

    

### [[2108.10851] Autoencoder-based Semantic Novelty Detection: Towards Dependable AI-based Systems](http://arxiv.org/abs/2108.10851)


  Many autonomous systems, such as driverless taxis, perform safety critical
functions. Autonomous systems employ artificial intelligence (AI) techniques,
specifically for the environment perception. Engineers cannot completely test
or formally verify AI-based autonomous systems. The accuracy of AI-based
systems depends on the quality of training data. Thus, novelty detection -
identifying data that differ in some respect from the data used for training -
becomes a safety measure for system development and operation. In this paper,
we propose a new architecture for autoencoder-based semantic novelty detection
with two innovations: architectural guidelines for a semantic autoencoder
topology and a semantic error calculation as novelty criteria. We demonstrate
that such a semantic novelty detection outperforms autoencoder-based novelty
detection approaches known from literature by minimizing false negatives.

    

### [[2108.10855] Quantum Artificial Intelligence for the Science of Climate Change](http://arxiv.org/abs/2108.10855)


  Climate change has become one of the biggest global problems increasingly
compromising the Earth's habitability. Recent developments such as the
extraordinary heat waves in California & Canada, and the devastating floods in
Germany point to the role of climate change in the ever-increasing frequency of
extreme weather. Numerical modelling of the weather and climate have seen
tremendous improvements in the last five decades, yet stringent limitations
remain to be overcome. Spatially and temporally localized forecasting is the
need of the hour for effective adaptation measures towards minimizing the loss
of life and property. Artificial Intelligence-based methods are demonstrating
promising results in improving predictions, but are still limited by the
availability of requisite hardware and software required to process the vast
deluge of data at a scale of the planet Earth. Quantum computing is an emerging
paradigm that has found potential applicability in several fields. In this
opinion piece, we argue that new developments in Artificial Intelligence
algorithms designed for quantum computers - also known as Quantum Artificial
Intelligence (QAI) - may provide the key breakthroughs necessary to furthering
the science of climate change. The resultant improvements in weather and
climate forecasts are expected to cascade to numerous societal benefits.

    

### [[2108.10876] Quantum adaptive agents with efficient long-term memories](http://arxiv.org/abs/2108.10876)


  Central to the success of adaptive systems is their ability to interpret
signals from their environment and respond accordingly -- they act as agents
interacting with their surroundings. Such agents typically perform better when
able to execute increasingly complex strategies. This comes with a cost: the
more information the agent must recall from its past experiences, the more
memory it will need. Here we investigate the power of agents capable of quantum
information processing. We uncover the most general form a quantum agent need
adopt to maximise memory compression advantages, and provide a systematic means
of encoding their memory states. We show these encodings can exhibit extremely
favourable scaling advantages relative to memory-minimal classical agents when
information must be retained about events increasingly far into the past.

    

### [[1911.04942] RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers](http://arxiv.org/abs/1911.04942)


  When translating natural language questions into SQL queries to answer
questions from a database, contemporary semantic parsing models struggle to
generalize to unseen database schemas. The generalization challenge lies in (a)
encoding the database relations in an accessible way for the semantic parser,
and (b) modeling alignment between database columns and their mentions in a
given query. We present a unified framework, based on the relation-aware
self-attention mechanism, to address schema encoding, schema linking, and
feature representation within a text-to-SQL encoder. On the challenging Spider
dataset this framework boosts the exact match accuracy to 57.2%, surpassing its
best counterparts by 8.7% absolute improvement. Further augmented with BERT, it
achieves the new state-of-the-art performance of 65.6% on the Spider
leaderboard. In addition, we observe qualitative improvements in the model's
understanding of schema linking and alignment. Our implementation will be
open-sourced at this https URL.

    

### [[2011.09850] A Theoretical Computer Science Perspective on Consciousness](http://arxiv.org/abs/2011.09850)


  The quest to understand consciousness, once the purview of philosophers and
theologians, is now actively pursued by scientists of many stripes. This paper
studies consciousness from the perspective of theoretical computer science. It
formalizes the Global Workspace Theory (GWT) originated by cognitive
neuroscientist Bernard Baars and further developed by him, Stanislas Dehaene,
and others. Our major contribution lies in the precise formal definition of a
Conscious Turing Machine (CTM), also called a Conscious AI. We define the CTM
in the spirit of Alan Turing's simple yet powerful definition of a computer,
the Turing Machine (TM). We are not looking for a complex model of the brain
nor of cognition but for a simple model of (the admittedly complex concept of)
consciousness. After formally defining CTM, we give a formal definition of
consciousness in CTM. We then suggest why the CTM has the feeling of
consciousness. The reasonableness of the definitions and explanations can be
judged by how well they agree with commonly accepted intuitive concepts of
human consciousness, the breadth of related concepts that the model explains
easily and naturally, and the extent of its agreement with scientific evidence.

    

### [[2108.10779] Randomized C/C++ dynamic memory allocator](http://arxiv.org/abs/2108.10779)


  Dynamic memory management requires special attention in programming. It
should be fast and secure at the same time. This paper proposes a new
randomized dynamic memory management algorithm designed to meet these
requirements. Randomization is a key feature intended to protect applications
from "use-after-free" or similar attacks. At the same time, the state in the
algorithm consists only of one pointer, so it does not consume extra memory for
itself. However, our algorithm is not a universal solution. It does not solve
the memory fragmentation problem and it needs further development and testing.

    

### [[2108.10436] Rewrite Rule Inference Using Equality Saturation](http://arxiv.org/abs/2108.10436)


  Many compilers, synthesizers, and theorem provers rely on rewrite rules to
simplify expressions or prove equivalences. Developing rewrite rules can be
difficult: rules may be subtly incorrect, profitable rules are easy to miss,
and rulesets must be rechecked or extended whenever semantics are tweaked.
Large rulesets can also be challenging to apply: redundant rules slow down
rule-based search and frustrate debugging. This paper explores how equality
saturation, a promising technique that uses e-graphs to apply rewrite rules,
can also be used to infer rewrite rules. E-graphs can compactly represent the
exponentially large sets of enumerated terms and potential rewrite rules. We
show that equality saturation efficiently shrinks both sets, leading to faster
synthesis of smaller, more general rulesets.
We prototyped these strategies in a tool dubbed ruler. Compared to a similar
tool built on CVC4, ruler synthesizes 5.8X smaller rulesets 25X faster without
compromising on proving power. In an end-to-end case study, we show
ruler-synthesized rules which perform as well as those crafted by domain
experts, and addressed a longstanding issue in a popular open source tool.

    

### [[2108.10493] Language Transformations in the Classroom](http://arxiv.org/abs/2108.10493)


  Language transformations are algorithms that take a language specification in
input, and return the language specification modified. Language transformations
are useful for automatically adding features such as subtyping to programming
languages (PLs), and for automatically deriving abstract machines.
In this paper, we set forth the thesis that teaching programming languages
features with the help of language transformations, in addition to the planned
material, can be beneficial for students to help them deepen their
understanding of the features being taught.
We have conducted a study on integrating language transformations into an
undergraduate PL course. We describe our study, the material that we have
taught, and the exam submitted to students, and we present the results from
this study. Although we refrain from drawing general conclusions on the
effectiveness of language transformations, this paper offers encouraging data.
We also offer this paper to inspire similar studies.

    

### [[2108.10558] The Mays and Musts of Concurrent Strategies](http://arxiv.org/abs/2108.10558)


  Concurrent strategies based on event structures are examined from the
viewpoint of 'may' and 'must' testing in traditional process calculi. In their
pure form concurrent strategies fail to expose the deadlocks and divergences
that can arise in their composition. This motivates an extension of the
bicategory of concurrent strategies to treat the 'may' and 'must' behaviour of
strategies under testing. One extension adjoins neutral moves to strategies but
in so doing loses identities w.r.t. composition. This in turn motivates another
extension in which concurrent strategies are accompanied by stopping
configurations; the ensuing stopping strategies inherit the structure of a
bicategory from that of strategies. The technical developments converge in
providing characterisations of the 'may' and 'must' equivalences and preorders
on strategies.

    

### [[2108.10848] On Encoding LF in a Predicate Logic over Simply-Typed Lambda Terms](http://arxiv.org/abs/2108.10848)


  Felty and Miller have described what they claim to be a faithful encoding of
the dependently typed lambda calculus LF in the logic of hereditary Harrop
formulas, a sublogic of an intuitionistic variant of Church's Simple Theory of
Types. Their encoding is based roughly on translating object expressions in LF
into terms in a simply typed lambda calculus by erasing dependencies in typing
and then recapturing the erased dependencies through the use of predicates.
Unfortunately, this idea does not quite work. In particular, we provide a
counterexample to the claim that the described encoding is faithful. The
underlying reason for the falsity of the claim is that the mapping from
dependently typed lambda terms to simply typed ones is not one-to-one and hence
the inverse transformation is ambiguous. This observation has a broad
implication for other related encodings.

    

### [[2108.10865] On Specialization of a Program Model of Naive Pattern Matching in Strings (Extended Abstract)](http://arxiv.org/abs/2108.10865)


  We have proved that for any pattern p the tail recursive program model of
naive pattern matching may be automatically specialized w.r.t. the pattern p to
a specialized version of the so-called KMP-algorithm, using the Higman-Kruskal
relation that controls the unfolding/folding. Given an input string, the
corresponding residual program finds the first occurrence of p in the string in
linear time on the string length. The current state of the automated program
specialization art based on unfolding/folding is too weak in order to be able
to reproduce the proof, done by hands, of the uniform property above, while it
known before that program specialization is sometimes able to produce the
KMP-algorithm for a few concrete static patterns.

    

### [[1907.13227] Compiling With Classical Connectives](http://arxiv.org/abs/1907.13227)


  The study of polarity in computation has revealed that an "ideal" programming
language combines both call-by-value and call-by-name evaluation; the two
calling conventions are each ideal for half the types in a programming
language. But this binary choice leaves out call-by-need which is used in
practice to implement lazy-by-default languages like Haskell. We show how the
notion of polarity can be extended beyond the value/name dichotomy to include
call-by-need by adding a mechanism for sharing which is enough to compile a
Haskell-like functional language with user-defined types. The key to capturing
sharing in this mixed-evaluation setting is to generalize the usual notion of
polarity "shifts:" rather than just two shifts (between positive and negative)
we have a family of four dual shifts.
We expand on this idea of logical duality -- "and" is dual to "or;" proof is
dual to refutation -- for the purpose of compiling a variety of types. Based on
a general notion of data and codata, we show how classical connectives can be
used to encode a wide range of built-in and user-defined types. In contrast
with an intuitionistic logic corresponding to pure functional programming,
these classical connectives bring more of the pleasant symmetries of classical
logic to the computationally-relevant, constructive setting. In particular, an
involutive pair of negations bridges the gulf between the wide-spread notions
of parametric polymorphism and abstract data types in programming languages. To
complete the study of duality in compilation, we also consider the dual to
call-by-need evaluation, which shares the computation within the control flow
of a program instead of computation within the information flow.

    

### [<title>How are cached DMatrix are used in a Booster? - XGBoost</title>](https://discuss.xgboost.ai/t/how-are-cached-dmatrix-are-used-in-a-booster/28/5)

### [<title>How are cached DMatrix are used in a Booster? - XGBoost</title>](https://discuss.xgboost.ai/t/how-are-cached-dmatrix-are-used-in-a-booster/28/6)

### [<title>[Cpp] Non-Usable interface of xgboost/predictor.h - XGBoost</title>](https://discuss.xgboost.ai/t/cpp-non-usable-interface-of-xgboost-predictor-h/2421/8)

### [<title>Can't turn off the validation messages during training - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/cant-turn-off-the-validation-messages-during-training/2445/3)