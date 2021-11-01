
## 2021-11-1

### [[2110.15344] Learning to Jump from Pixels](http://arxiv.org/abs/2110.15344)


  Today's robotic quadruped systems can robustly walk over a diverse range of
rough but continuous terrains, where the terrain elevation varies gradually.
Locomotion on discontinuous terrains, such as those with gaps or obstacles,
presents a complementary set of challenges. In discontinuous settings, it
becomes necessary to plan ahead using visual inputs and to execute agile
behaviors beyond robust walking, such as jumps. Such dynamic motion results in
significant motion of onboard sensors, which introduces a new set of challenges
for real-time visual processing. The requirement for agility and terrain
awareness in this setting reinforces the need for robust control. We present
Depth-based Impulse Control (DIC), a method for synthesizing highly agile
visually-guided locomotion behaviors. DIC affords the flexibility of model-free
learning but regularizes behavior through explicit model-based optimization of
ground reaction forces. We evaluate the proposed method both in simulation and
in the real world.

    

### [[2110.15405] Stand-alone device for IoT applications](http://arxiv.org/abs/2110.15405)


  Internet of Things (IoT) is a digital world of connected and talking devices,
providing room for countless and diverse smart applications. This paper
proposes one such IoT enabled stand-alone device with numerous capabilities:
(i) interaction with user, (ii) required application selection, (iii) data
sensing, (iv) data publish, and (v) decision making and actuation. The
algorithm allows user to pick an application and input specific data for
calibration, which on completion enables the device for further working. To
verify and test its capability, a smart home garden environment is created
using this device, temperature, humidity, soil moisture sensors and actuators.
As it is implicit that real-time communication is inevitable for an IoT
application, the sensor data is published to a Mosquitto MQTT broker to permit
real-time remote access. The decision taken by the device is sent to actuators
via relay, thus a continuous monitoring process is achieved. Results are
obtained for the application which proves the device suitability for IoT
applications.

    

### [[2110.15747] A Survey on Threat Situation Awareness Systems: Framework, Techniques, and Insights](http://arxiv.org/abs/2110.15747)


  Cyberspace is full of uncertainty in terms of advanced and sophisticated
cyber threats which are equipped with novel approaches to learn the system and
propagate themselves, such as AI-powered threats. To debilitate these types of
threats, a modern and intelligent Cyber Situation Awareness (SA) system need to
be developed which has the ability of monitoring and capturing various types of
threats, analyzing and devising a plan to avoid further attacks. This paper
provides a comprehensive study on the current state-of-the-art in the cyber SA
to discuss the following aspects of SA: key design principles, framework,
classifications, data collection, and analysis of the techniques, and
evaluation methods. Lastly, we highlight misconceptions, insights and
limitations of this study and suggest some future work directions to address
the limitations.

    

### [[2110.15788] Towards Intelligent Load Balancing in Data Centers](http://arxiv.org/abs/2110.15788)


  Network load balancers are important components in data centers to provide
scalable services. Workload distribution algorithms are based on heuristics,
e.g., Equal-Cost Multi-Path (ECMP), Weighted-Cost Multi-Path (WCMP) or naive
machine learning (ML) algorithms, e.g., ridge regression. Advanced ML-based
approaches help achieve performance gain in different networking and system
problems. However, it is challenging to apply ML algorithms on networking
problems in real-life systems. It requires domain knowledge to collect features
from low-latency, high-throughput, and scalable networking systems, which are
dynamic and heterogenous. This paper proposes Aquarius to bridge the gap
between ML and networking systems and demonstrates its usage in the context of
network load balancers. This paper demonstrates its ability of conducting both
offline data analysis and online model deployment in realistic systems. The
results show that the ML model trained and deployed using Aquarius improves
load balancing performance yet they also reveals more challenges to be resolved
to apply ML for networking systems.

    

### [[2110.15841] Angle Diversity Trasmitter For High Speed Data Center Uplink Communications](http://arxiv.org/abs/2110.15841)


  This paper proposes an uplink optical wireless communication (OWC) link
design that can be used by data centers to support communication in spine and
leaf architectures between the top of rack leaf switches and large spine
switches whose access points are mounted in the ceiling. The use of optical
wireless links reduces cabling and allows easy reconfigurability for example
when data centres expand. We consider three racks in a data center where each
rack contains an Angle Diversity Transmitter (ADT) positioned on the top of the
rack to realize the uplink function of a top-of-the-rack (ToR) or a leaf
switch. Four receivers are considered to be installed on the ceiling where each
is connected to a spine switch. Two types of optical receivers are studied
which are a Wide Field-of-View Receiver (WFOVR) and an Angle Diversity Receiver
(ADR). The performance of the proposed system is evaluated when the links run
at data rates higher than 19 Gbps. The results indicate that the proposed
approach achieves excellent performance using simple On-Off Keying (OOK)

    

### [[2104.07923] Analytical Models of the Performance of IEEE 802.11p Vehicle to Vehicle Communications](http://arxiv.org/abs/2104.07923)


  The critical nature of vehicular communications requires their extensive
testing and evaluation. Analytical models can represent an attractive and
cost-effective approach for such evaluation if they can adequately model all
underlying effects that impact the performance of vehicular communications.
Several analytical models have been proposed to date to model vehicular
communications based on the IEEE 802.11p (or DSRC) standard. However, existing
models normally model in detail the MAC (Medium Access Control), and generally
simplify the propagation and interference effects. This reduces their value as
an alternative to evaluate the performance of vehicular communications. This
paper addresses this gap, and presents new analytical models that accurately
model the performance of vehicle-to-vehicle communications based on the IEEE
802.11p standard. The models jointly account for a detailed modeling of the
propagation and interference effects, as well as the impact of the hidden
terminal problem. The model quantifies the PDR (Packet Delivery Ratio) as a
function of the distance between transmitter and receiver. The paper also
presents new analytical models to quantify the probability of the four
different types of packet errors in IEEE 802.11p. In addition, the paper
presents the first analytical model capable to accurately estimate the Channel
Busy Ratio (CBR) metric even under high channel load levels. All the analytical
models are validated by means of simulation for a wide range of parameters,
including traffic densities, packet transmission frequencies, transmission
power levels, data rates and packet sizes. An implementation of the models is
provided openly to facilitate their use by the community.

    

### [[2110.15362] BitTrain: Sparse Bitmap Compression for Memory-Efficient Training on the Edge](http://arxiv.org/abs/2110.15362)


  Training on the Edge enables neural networks to learn continuously from new
data after deployment on memory-constrained edge devices. Previous work is
mostly concerned with reducing the number of model parameters which is only
beneficial for inference. However, memory footprint from activations is the
main bottleneck for training on the edge. Existing incremental training methods
fine-tune the last few layers sacrificing accuracy gains from re-training the
whole model. In this work, we investigate the memory footprint of training deep
learning models, and use our observations to propose BitTrain. In BitTrain, we
exploit activation sparsity and propose a novel bitmap compression technique
that reduces the memory footprint during training. We save the activations in
our proposed bitmap compression format during the forward pass of the training,
and restore them during the backward pass for the optimizer computations. The
proposed method can be integrated seamlessly in the computation graph of modern
deep learning frameworks. Our implementation is safe by construction, and has
no negative impact on the accuracy of model training. Experimental results show
up to 34% reduction in the memory footprint at a sparsity level of 50%. Further
pruning during training results in more than 70% sparsity, which can lead to up
to 56% reduction in memory footprint. BitTrain advances the efforts towards
bringing more machine learning capabilities to edge devices. Our source code is
available at this https URL.

    

### [[2110.15383] New SAR target recognition based on YOLO and very deep multi-canonical correlation analysis](http://arxiv.org/abs/2110.15383)


  Synthetic Aperture Radar (SAR) images are prone to be contaminated by noise,
which makes it very difficult to perform target recognition in SAR images.
Inspired by great success of very deep convolutional neural networks (CNNs),
this paper proposes a robust feature extraction method for SAR image target
classification by adaptively fusing effective features from different CNN
layers. First, YOLOv4 network is fine-tuned to detect the targets from the
respective MF SAR target images. Second, a very deep CNN is trained from
scratch on the moving and stationary target acquisition and recognition (MSTAR)
database by using small filters throughout the whole net to reduce the speckle
noise. Besides, using small-size convolution filters decreases the number of
parameters in each layer and, therefore, reduces computation cost as the CNN
goes deeper. The resulting CNN model is capable of extracting very deep
features from the target images without performing any noise filtering or
pre-processing techniques. Third, our approach proposes to use the
multi-canonical correlation analysis (MCCA) to adaptively learn CNN features
from different layers such that the resulting representations are highly
linearly correlated and therefore can achieve better classification accuracy
even if a simple linear support vector machine is used. Experimental results on
the MSTAR dataset demonstrate that the proposed method outperforms the
state-of-the-art methods.

    

### [[2110.15397] A Computationally Efficient Method for Learning Exponential Family Distributions](http://arxiv.org/abs/2110.15397)


  We consider the question of learning the natural parameters of a $k$
parameter minimal exponential family from i.i.d. samples in a computationally
and statistically efficient manner. We focus on the setting where the support
as well as the natural parameters are appropriately bounded. While the
traditional maximum likelihood estimator for this class of exponential family
is consistent, asymptotically normal, and asymptotically efficient, evaluating
it is computationally hard. In this work, we propose a computationally
efficient estimator that is consistent as well as asymptotically normal under
mild conditions. We provide finite sample guarantees to achieve an ($\ell_2$)
error of $\alpha$ in the parameter estimation with sample complexity
$O(\mathrm{poly}(k/\alpha))$ and computational complexity
${O}(\mathrm{poly}(k/\alpha))$. To establish these results, we show that, at
the population level, our method can be viewed as the maximum likelihood
estimation of a re-parameterized distribution belonging to the same class of
exponential family.

    

### [[2110.15403] Selective Regression Under Fairness Criteria](http://arxiv.org/abs/2110.15403)


  Selective regression allows abstention from prediction if the confidence to
make an accurate prediction is not sufficient. In general, by allowing a reject
option, one expects the performance of a regression model to increase at the
cost of reducing coverage (i.e., by predicting fewer samples). However, as
shown in this work, in some cases, the performance of minority group can
decrease while we reduce the coverage, and thus selective regression can
magnify disparities between different sensitive groups. We show that such an
unwanted behavior can be avoided if we can construct features satisfying the
sufficiency criterion, so that the mean prediction and the associated
uncertainty are calibrated across all the groups. Further, to mitigate the
disparity in the performance across groups, we introduce two approaches based
on this calibration criterion: (a) by regularizing an upper bound of
conditional mutual information under a Gaussian assumption and (b) by
regularizing a contrastive loss for mean and uncertainty prediction. The
effectiveness of these approaches are demonstrated on synthetic as well as
real-world datasets.

    

### [[2110.15409] What makes us curious? analysis of a corpus of open-domain questions](http://arxiv.org/abs/2110.15409)


  Every day people ask short questions through smart devices or online forums
to seek answers to all kinds of queries. With the increasing number of
questions collected it becomes difficult to provide answers to each of them,
which is one of the reasons behind the growing interest in automated question
answering. Some questions are similar to existing ones that have already been
answered, while others could be answered by an external knowledge source such
as Wikipedia. An important question is what can be revealed by analysing a
large set of questions. In 2017, "We the Curious" science centre in Bristol
started a project to capture the curiosity of Bristolians: the project
collected more than 10,000 questions on various topics. As no rules were given
during collection, the questions are truly open-domain, and ranged across a
variety of topics. One important aim for the science centre was to understand
what concerns its visitors had beyond science, particularly on societal and
cultural issues. We addressed this question by developing an Artificial
Intelligence tool that can be used to perform various processing tasks:
detection of equivalence between questions; detection of topic and type; and
answering of the question. As we focused on the creation of a "generalist"
tool, we trained it with labelled data from different datasets. We called the
resulting model QBERT. This paper describes what information we extracted from
the automated analysis of the WTC corpus of open-domain questions.

    

### [[2110.15412] Stochastic Mirror Descent: Convergence Analysis and Adaptive Variants via the Mirror Stochastic Polyak Stepsize](http://arxiv.org/abs/2110.15412)


  We investigate the convergence of stochastic mirror descent (SMD) in
relatively smooth and smooth convex optimization. In relatively smooth convex
optimization we provide new convergence guarantees for SMD with a constant
stepsize. For smooth convex optimization we propose a new adaptive stepsize
scheme -- the mirror stochastic Polyak stepsize (mSPS). Notably, our
convergence results in both settings do not make bounded gradient assumptions
or bounded variance assumptions, and we show convergence to a neighborhood that
vanishes under interpolation. mSPS generalizes the recently proposed stochastic
Polyak stepsize (SPS) (Loizou et al., 2021) to mirror descent and remains both
practical and efficient for modern machine learning applications while
inheriting the benefits of mirror descent. We complement our results with
experiments across various supervised learning tasks and different instances of
SMD, demonstrating the effectiveness of mSPS.

    

### [[2110.15424] Physics-Driven Learning of Wasserstein GAN for Density Reconstruction in Dynamic Tomography](http://arxiv.org/abs/2110.15424)


  Object density reconstruction from projections containing scattered radiation
and noise is of critical importance in many applications. Existing scatter
correction and density reconstruction methods may not provide the high accuracy
needed in many applications and can break down in the presence of unmodeled or
anomalous scatter and other experimental artifacts. Incorporating
machine-learned models could prove beneficial for accurate density
reconstruction particularly in dynamic imaging, where the time-evolution of the
density fields could be captured by partial differential equations or by
learning from hydrodynamics simulations. In this work, we demonstrate the
ability of learned deep neural networks to perform artifact removal in noisy
density reconstructions, where the noise is imperfectly characterized. We use a
Wasserstein generative adversarial network (WGAN), where the generator serves
as a denoiser that removes artifacts in densities obtained from traditional
reconstruction algorithms. We train the networks from large density time-series
datasets, with noise simulated according to parametric random distributions
that may mimic noise in experiments. The WGAN is trained with noisy density
frames as generator inputs, to match the generator outputs to the distribution
of clean densities (time-series) from simulations. A supervised loss is also
included in the training, which leads to improved density restoration
performance. In addition, we employ physics-based constraints such as mass
conservation during network training and application to further enable highly
accurate density reconstructions. Our preliminary numerical results show that
the models trained in our frameworks can remove significant portions of unknown
noise in density time-series data.

    

### [[2110.15426] RadBERT-CL: Factually-Aware Contrastive Learning For Radiology Report Classification](http://arxiv.org/abs/2110.15426)


  Radiology reports are unstructured and contain the imaging findings and
corresponding diagnoses transcribed by radiologists which include clinical
facts and negated and/or uncertain statements. Extracting pathologic findings
and diagnoses from radiology reports is important for quality control,
population health, and monitoring of disease progress. Existing works,
primarily rely either on rule-based systems or transformer-based pre-trained
model fine-tuning, but could not take the factual and uncertain information
into consideration, and therefore generate false-positive outputs. In this
work, we introduce three sedulous augmentation techniques which retain factual
and critical information while generating augmentations for contrastive
learning. We introduce RadBERT-CL, which fuses these information into BlueBert
via a self-supervised contrastive loss. Our experiments on MIMIC-CXR show
superior performance of RadBERT-CL on fine-tuning for multi-class, multi-label
report classification. We illustrate that when few labeled data are available,
RadBERT-CL outperforms conventional SOTA transformers (BERT/BlueBert) by
significantly larger margins (6-11%). We also show that the representations
learned by RadBERT-CL can capture critical medical information in the latent
space.

    

### [[2110.15431] Universal Decision Models](http://arxiv.org/abs/2110.15431)


  Humans are universal decision makers: we reason causally to understand the
world; we act competitively to gain advantage in commerce, games, and war; and
we are able to learn to make better decisions through trial and error. In this
paper, we propose Universal Decision Model (UDM), a mathematical formalism
based on category theory. Decision objects in a UDM correspond to instances of
decision tasks, ranging from causal models and dynamical systems such as Markov
decision processes and predictive state representations, to network multiplayer
games and Witsenhausen's intrinsic models, which generalizes all these previous
formalisms. A UDM is a category of objects, which include decision objects,
observation objects, and solution objects. Bisimulation morphisms map between
decision objects that capture structure-preserving abstractions. We formulate
universal properties of UDMs, including information integration, decision
solvability, and hierarchical abstraction. We describe universal functorial
representations of UDMs, and propose an algorithm for computing the minimal
object in a UDM using algebraic topology. We sketch out an application of UDMs
to causal inference in network economics, using a complex multiplayer
producer-consumer two-sided marketplace.

    

### [[2110.15438] InfoGCL: Information-Aware Graph Contrastive Learning](http://arxiv.org/abs/2110.15438)


  Various graph contrastive learning models have been proposed to improve the
performance of learning tasks on graph datasets in recent years. While
effective and prevalent, these models are usually carefully customized. In
particular, although all recent researches create two contrastive views, they
differ greatly in view augmentations, architectures, and objectives. It remains
an open question how to build your graph contrastive learning model from
scratch for particular graph learning tasks and datasets. In this work, we aim
to fill this gap by studying how graph information is transformed and
transferred during the contrastive learning process and proposing an
information-aware graph contrastive learning framework called InfoGCL. The key
point of this framework is to follow the Information Bottleneck principle to
reduce the mutual information between contrastive parts while keeping
task-relevant information intact at both the levels of the individual module
and the entire framework so that the information loss during graph
representation learning can be minimized. We show for the first time that all
recent graph contrastive learning methods can be unified by our framework. We
empirically validate our theoretical analysis on both node and graph
classification benchmark datasets, and demonstrate that our algorithm
significantly outperforms the state-of-the-arts.

    

### [[2110.15440] HD-cos Networks: Efficient Neural Architectures for Secure Multi-Party Computation](http://arxiv.org/abs/2110.15440)


  Multi-party computation (MPC) is a branch of cryptography where multiple
non-colluding parties execute a well designed protocol to securely compute a
function. With the non-colluding party assumption, MPC has a cryptographic
guarantee that the parties will not learn sensitive information from the
computation process, making it an appealing framework for applications that
involve privacy-sensitive user data. In this paper, we study training and
inference of neural networks under the MPC setup. This is challenging because
the elementary operations of neural networks such as the ReLU activation
function and matrix-vector multiplications are very expensive to compute due to
the added multi-party communication overhead. To address this, we propose the
HD-cos network that uses 1) cosine as activation function, 2) the
Hadamard-Diagonal transformation to replace the unstructured linear
transformations. We show that both of the approaches enjoy strong theoretical
motivations and efficient computation under the MPC setup. We demonstrate on
multiple public datasets that HD-cos matches the quality of the more expensive
baselines.

    

### [[2110.15442] Scalable Uni-directional Pareto Optimality for Multi-Task Learning with Constraints](http://arxiv.org/abs/2110.15442)


  We propose a scalable Pareto solver for Multi-Objective Optimization (MOO)
problems, including support for optimization under constraints. An important
application of this solver is to estimate high-dimensional neural models for
MOO classification tasks. We demonstrate significant runtime and space
improvement using our solver \vs prior methods, verify that solutions found are
truly Pareto optimal on a benchmark set of known non-convex MOO problems from
{\em operations research}, and provide a practical evaluation against prior
methods for Multi-Task Learning (MTL).

    

### [[2110.15444] 10 Security and Privacy Problems in Self-Supervised Learning](http://arxiv.org/abs/2110.15444)


  Self-supervised learning has achieved revolutionary progress in the past
several years and is commonly believed to be a promising approach for
general-purpose AI. In particular, self-supervised learning aims to pre-train
an encoder using a large amount of unlabeled data. The pre-trained encoder is
like an "operating system" of the AI ecosystem. In particular, the encoder can
be used as a feature extractor for many downstream tasks with little or no
labeled training data. Existing studies on self-supervised learning mainly
focused on pre-training a better encoder to improve its performance on
downstream tasks in non-adversarial settings, leaving its security and privacy
in adversarial settings largely unexplored. A security or privacy issue of a
pre-trained encoder leads to a single point of failure for the AI ecosystem. In
this book chapter, we discuss 10 basic security and privacy problems for the
pre-trained encoders in self-supervised learning, including six confidentiality
problems, three integrity problems, and one availability problem. For each
problem, we discuss potential opportunities and challenges. We hope our book
chapter will inspire future research on the security and privacy of
self-supervised learning.

    

### [[2110.15454] VigDet: Knowledge Informed Neural Temporal Point Process for Coordination Detection on Social Media](http://arxiv.org/abs/2110.15454)


  Recent years have witnessed an increasing use of coordinated accounts on
social media, operated by misinformation campaigns to influence public opinion
and manipulate social outcomes. Consequently, there is an urgent need to
develop an effective methodology for coordinated group detection to combat the
misinformation on social media. However, existing works suffer from various
drawbacks, such as, either limited performance due to extreme reliance on
predefined signatures of coordination, or instead an inability to address the
natural sparsity of account activities on social media with useful prior domain
knowledge. Therefore, in this paper, we propose a coordination detection
framework incorporating neural temporal point process with prior knowledge such
as temporal logic or pre-defined filtering functions. Specifically, when
modeling the observed data from social media with neural temporal point
process, we jointly learn a Gibbs-like distribution of group assignment based
on how consistent an assignment is to (1) the account embedding space and (2)
the prior knowledge. To address the challenge that the distribution is hard to
be efficiently computed and sampled from, we design a theoretically guaranteed
variational inference approach to learn a mean-field approximation for it.
Experimental results on a real-world dataset show the effectiveness of our
proposed method compared to the SOTA model in both unsupervised and
semi-supervised settings. We further apply our model on a COVID-19 Vaccine
Tweets dataset. The detection result suggests the presence of suspicious
coordinated efforts on spreading misinformation about COVID-19 vaccines.

    

### [[2110.15456] FAST: DNN Training Under Variable Precision Block Floating Point with Stochastic Rounding](http://arxiv.org/abs/2110.15456)


  Block Floating Point (BFP) can efficiently support quantization for Deep
Neural Network (DNN) training by providing a wide dynamic range via a shared
exponent across a group of values. In this paper, we propose a Fast First,
Accurate Second Training (FAST) system for DNNs, where the weights,
activations, and gradients are represented in BFP. FAST supports matrix
multiplication with variable precision BFP input operands, enabling incremental
increases in DNN precision throughout training. By increasing the BFP precision
across both training iterations and DNN layers, FAST can greatly shorten the
training time while reducing overall hardware resource usage. Our FAST
Multipler-Accumulator (fMAC) supports dot product computations under multiple
BFP precisions. We validate our FAST system on multiple DNNs with different
datasets, demonstrating a 2-6$\times$ speedup in training on a single-chip
platform over prior work based on \textbf{mixed-precision or block} floating
point number systems while achieving similar performance in validation
accuracy.

    

### [[2110.15458] Open Problem: Tight Online Confidence Intervals for RKHS Elements](http://arxiv.org/abs/2110.15458)


  Confidence intervals are a crucial building block in the analysis of various
online learning problems. The analysis of kernel based bandit and reinforcement
learning problems utilize confidence intervals applicable to the elements of a
reproducing kernel Hilbert space (RKHS). However, the existing confidence
bounds do not appear to be tight, resulting in suboptimal regret bounds. In
fact, the existing regret bounds for several kernelized bandit algorithms
(e.g., GP-UCB, GP-TS, and their variants) may fail to even be sublinear. It is
unclear whether the suboptimal regret bound is a fundamental shortcoming of
these algorithms or an artifact of the proof, and the main challenge seems to
stem from the online (sequential) nature of the observation points. We
formalize the question of online confidence intervals in the RKHS setting and
overview the existing results.

    

### [[2110.15481] Brick-by-Brick: Combinatorial Construction with Deep Reinforcement Learning](http://arxiv.org/abs/2110.15481)


  Discovering a solution in a combinatorial space is prevalent in many
real-world problems but it is also challenging due to diverse complex
constraints and the vast number of possible combinations. To address such a
problem, we introduce a novel formulation, combinatorial construction, which
requires a building agent to assemble unit primitives (i.e., LEGO bricks)
sequentially -- every connection between two bricks must follow a fixed rule,
while no bricks mutually overlap. To construct a target object, we provide
incomplete knowledge about the desired target (i.e., 2D images) instead of
exact and explicit volumetric information to the agent. This problem requires a
comprehensive understanding of partial information and long-term planning to
append a brick sequentially, which leads us to employ reinforcement learning.
The approach has to consider a variable-sized action space where a large number
of invalid actions, which would cause overlap between bricks, exist. To resolve
these issues, our model, dubbed Brick-by-Brick, adopts an action validity
prediction network that efficiently filters invalid actions for an actor-critic
network. We demonstrate that the proposed method successfully learns to
construct an unseen object conditioned on a single image or multiple views of a
target object.

    

### [[2110.15484] Cycle-Balanced Representation Learning For Counterfactual Inference](http://arxiv.org/abs/2110.15484)


  With the widespread accumulation of observational data, researchers obtain a
new direction to learn counterfactual effects in many domains (e.g., health
care and computational advertising) without Randomized Controlled Trials(RCTs).
However, observational data suffer from inherent missing counterfactual
outcomes, and distribution discrepancy between treatment and control groups due
to behaviour preference. Motivated by recent advances of representation
learning in the field of domain adaptation, we propose a novel framework based
on Cycle-Balanced REpresentation learning for counterfactual inference (CBRE),
to solve above problems. Specifically, we realize a robust balanced
representation for different groups using adversarial training, and meanwhile
construct an information loop, such that preserve original data properties
cyclically, which reduces information loss when transforming data into latent
representation space.Experimental results on three real-world datasets
demonstrate that CBRE matches/outperforms the state-of-the-art methods, and it
has a great potential to be applied to counterfactual inference.

    

### [[2110.15485] Location-routing Optimisation for Urban Logistics Using Mobile Parcel Locker Based on Hybrid Q-Learning Algorithm](http://arxiv.org/abs/2110.15485)


  Mobile parcel lockers (MPLs) have been recently introduced by urban logistics
operators as a means to reduce traffic congestion and operational cost. Their
capability to relocate their position during the day has the potential to
improve customer accessibility and convenience (if deployed and planned
accordingly), allowing customers to collect parcels at their preferred time
among one of the multiple locations. This paper proposes an integer programming
model to solve the Location Routing Problem for MPLs to determine the optimal
configuration and locker routes. In solving this model, a Hybrid Q-Learning
algorithm-based Method (HQM) integrated with global and local search mechanisms
is developed, the performance of which is examined for different problem sizes
and benchmarked with genetic algorithms. Furthermore, we introduced two route
adjustment strategies to resolve stochastic events that may cause delays. The
results show that HQM achieves 443.41% improvement on average in solution
improvement, compared with the 94.91% improvement of heuristic counterparts,
suggesting HQM enables a more efficient search for better solutions. Finally,
we identify critical factors that contribute to service delays and investigate
their effects.

    

### [[2110.15486] DOCKSTRING: easy molecular docking yields better benchmarks for ligand design](http://arxiv.org/abs/2110.15486)


  The field of machine learning for drug discovery is witnessing an explosion
of novel methods. These methods are often benchmarked on simple physicochemical
properties such as solubility or general druglikeness, which can be readily
computed. However, these properties are poor representatives of objective
functions in drug design, mainly because they do not depend on the candidate's
interaction with the target. By contrast, molecular docking is a widely
successful method in drug discovery to estimate binding affinities. However,
docking simulations require a significant amount of domain knowledge to set up
correctly which hampers adoption. To this end, we present DOCKSTRING, a bundle
for meaningful and robust comparison of ML models consisting of three
components: (1) an open-source Python package for straightforward computation
of docking scores; (2) an extensive dataset of docking scores and poses of more
than 260K ligands for 58 medically-relevant targets; and (3) a set of
pharmaceutically-relevant benchmark tasks including regression, virtual
screening, and de novo design. The Python package implements a robust ligand
and target preparation protocol that allows non-experts to obtain meaningful
docking scores. Our dataset is the first to include docking poses, as well as
the first of its size that is a full matrix, thus facilitating experiments in
multiobjective optimization and transfer learning. Overall, our results
indicate that docking scores are a more appropriate evaluation objective than
simple physicochemical properties, yielding more realistic benchmark tasks and
molecular candidates.

    

### [[2110.15489] GalilAI: Out-of-Task Distribution Detection using Causal Active Experimentation for Safe Transfer RL](http://arxiv.org/abs/2110.15489)


  Out-of-distribution (OOD) detection is a well-studied topic in supervised
learning. Extending the successes in supervised learning methods to the
reinforcement learning (RL) setting, however, is difficult due to the data
generating process - RL agents actively query their environment for data, and
the data are a function of the policy followed by the agent. An agent could
thus neglect a shift in the environment if its policy did not lead it to
explore the aspect of the environment that shifted. Therefore, to achieve safe
and robust generalization in RL, there exists an unmet need for OOD detection
through active experimentation. Here, we attempt to bridge this lacuna by first
defining a causal framework for OOD scenarios or environments encountered by RL
agents in the wild. Then, we propose a novel task: that of Out-of-Task
Distribution (OOTD) detection. We introduce an RL agent that actively
experiments in a test environment and subsequently concludes whether it is OOTD
or not. We name our method GalilAI, in honor of Galileo Galilei, as it
discovers, among other causal processes, that gravitational acceleration is
independent of the mass of a body. Finally, we propose a simple probabilistic
neural network baseline for comparison, which extends extant Model-Based RL. We
find that GalilAI outperforms the baseline significantly. See visualizations of
our method this https URL


### [[2110.15497] Unsupervised Foreground Extraction via Deep Region Competition](http://arxiv.org/abs/2110.15497)


  We present Deep Region Competition (DRC), an algorithm designed to extract
foreground objects from images in a fully unsupervised manner. Foreground
extraction can be viewed as a special case of generic image segmentation that
focuses on identifying and disentangling objects from the background. In this
work, we rethink the foreground extraction by reconciling energy-based prior
with generative image modeling in the form of Mixture of Experts (MoE), where
we further introduce the learned pixel re-assignment as the essential inductive
bias to capture the regularities of background regions. With this modeling, the
foreground-background partition can be naturally found through
Expectation-Maximization (EM). We show that the proposed method effectively
exploits the interaction between the mixture components during the partitioning
process, which closely connects to region competition, a seminal approach for
generic image segmentation. Experiments demonstrate that DRC exhibits more
competitive performances on complex real-world data and challenging
multi-object scenes compared with prior methods. Moreover, we show empirically
that DRC can potentially generalize to novel foreground objects even from
categories unseen during training.

    

### [[2110.15498] Learning Personal Food Preferences via Food Logs Embedding](http://arxiv.org/abs/2110.15498)


  Diet management is key to managing chronic diseases such as diabetes.
Automated food recommender systems may be able to assist by providing meal
recommendations that conform to a user's nutrition goals and food preferences.
Current recommendation systems suffer from a lack of accuracy that is in part
due to a lack of knowledge of food preferences, namely foods users like to and
are able to eat frequently. In this work, we propose a method for learning food
preferences from food logs, a comprehensive but noisy source of information
about users' dietary habits. We also introduce accompanying metrics. The method
generates and compares word embeddings to identify the parent food category of
each food entry and then calculates the most popular. Our proposed approach
identifies 82% of a user's ten most frequently eaten foods. Our method is
publicly available on (this https URL)

    

### [[2110.15501] Doubly Robust Interval Estimation for Optimal Policy Evaluation in Online Learning](http://arxiv.org/abs/2110.15501)


  Evaluating the performance of an ongoing policy plays a vital role in many
areas such as medicine and economics, to provide crucial instruction on the
early-stop of the online experiment and timely feedback from the environment.
Policy evaluation in online learning thus attracts increasing attention by
inferring the mean outcome of the optimal policy (i.e., the value) in
real-time. Yet, such a problem is particularly challenging due to the dependent
data generated in the online environment, the unknown optimal policy, and the
complex exploration and exploitation trade-off in the adaptive experiment. In
this paper, we aim to overcome these difficulties in policy evaluation for
online learning. We explicitly derive the probability of exploration that
quantifies the probability of exploring the non-optimal actions under commonly
used bandit algorithms. We use this probability to conduct valid inference on
the online conditional mean estimator under each action and develop the doubly
robust interval estimation (DREAM) method to infer the value under the
estimated optimal policy in online learning. The proposed value estimator
provides double protection on the consistency and is asymptotically normal with
a Wald-type confidence interval provided. Extensive simulations and real data
applications are conducted to demonstrate the empirical validity of the
proposed DREAM method.

    

### [[2110.15503] A Pre-processing Method for Fairness in Ranking](http://arxiv.org/abs/2110.15503)


  Fair ranking problems arise in many decision-making processes that often
necessitate a trade-off between accuracy and fairness.
Many existing studies have proposed correction methods such as adding
fairness constraints to a ranking model's loss.
However, the challenge of correcting the data bias for fair ranking remains,
and the trade-off of the ranking models leaves room for improvement.
In this paper, we propose a fair ranking framework that evaluates the order
of training data in a pairwise manner as well as various fairness measurements
in ranking.
This study is the first proposal of a pre-processing method that solves fair
ranking problems using the pairwise ordering method with our best knowledge.
The fair pairwise ordering method is prominent in training the fair ranking
models because it ensures that the resulting ranking likely becomes parity
across groups.
As far as the fairness measurements in ranking are represented as a linear
constraint of the ranking models, we proved that the minimization of loss
function subject to the constraints is reduced to the closed solution of the
minimization problem augmented by weights to training data.
This closed solution inspires us to present a practical and stable algorithm
that iterates the optimization of weights and model parameters.
The empirical results over real-world datasets demonstrated that our method
outperforms the existing methods in the trade-off between accuracy and fairness
over real-world datasets and various fairness measurements.

    

### [[2110.15520] On Label Shift in Domain Adaptation via Wasserstein Distance](http://arxiv.org/abs/2110.15520)


  We study the label shift problem between the source and target domains in
general domain adaptation (DA) settings. We consider transformations
transporting the target to source domains, which enable us to align the source
and target examples. Through those transformations, we define the label shift
between two domains via optimal transport and develop theory to investigate the
properties of DA under various DA settings (e.g., closed-set, partial-set,
open-set, and universal settings). Inspired from the developed theory, we
propose Label and Data Shift Reduction via Optimal Transport (LDROT) which can
mitigate the data and label shifts simultaneously. Finally, we conduct
comprehensive experiments to verify our theoretical findings and compare LDROT
with state-of-the-art baselines.

    

### [[2110.15522] ADDS: Adaptive Differentiable Sampling for Robust Multi-Party Learning](http://arxiv.org/abs/2110.15522)


  Distributed multi-party learning provides an effective approach for training
a joint model with scattered data under legal and practical constraints.
However, due to the quagmire of a skewed distribution of data labels across
participants and the computation bottleneck of local devices, how to build
smaller customized models for clients in various scenarios while providing
updates appliable to the central model remains a challenge. In this paper, we
propose a novel adaptive differentiable sampling framework (ADDS) for robust
and communication-efficient multi-party learning. Inspired by the idea of
dropout in neural networks, we introduce a network sampling strategy in the
multi-party setting, which distributes different subnets of the central model
to clients for updating, and the differentiable sampling rates allow each
client to extract optimal local architecture from the supernet according to its
private data distribution. The approach requires minimal modifications to the
existing multi-party learning structure, and it is capable of integrating local
updates of all subnets into the supernet, improving the robustness of the
central model. The proposed framework significantly reduces local computation
and communication costs while speeding up the central model convergence, as we
demonstrated through experiments on real-world datasets.

    

### [[2110.15528] Deconvolutional Networks on Graph Data](http://arxiv.org/abs/2110.15528)


  In this paper, we consider an inverse problem in graph learning domain --
``given the graph representations smoothed by Graph Convolutional Network
(GCN), how can we reconstruct the input graph signal?" We propose Graph
Deconvolutional Network (GDN) and motivate the design of GDN via a combination
of inverse filters in spectral domain and de-noising layers in wavelet domain,
as the inverse operation results in a high frequency amplifier and may amplify
the noise. We demonstrate the effectiveness of the proposed method on several
tasks including graph feature imputation and graph structure generation.

    

### [[2110.15529] Topological Relational Learning on Graphs](http://arxiv.org/abs/2110.15529)


  Graph neural networks (GNNs) have emerged as a powerful tool for graph
classification and representation learning. However, GNNs tend to suffer from
over-smoothing problems and are vulnerable to graph perturbations. To address
these challenges, we propose a novel topological neural framework of
topological relational inference (TRI) which allows for integrating
higher-order graph information to GNNs and for systematically learning a local
graph structure. The key idea is to rewire the original graph by using the
persistent homology of the small neighborhoods of nodes and then to incorporate
the extracted topological summaries as the side information into the local
algorithm. As a result, the new framework enables us to harness both the
conventional information on the graph structure and information on the graph
higher order topological properties. We derive theoretical stability guarantees
for the new local topological representation and discuss their implications on
the graph algebraic connectivity. The experimental results on node
classification tasks demonstrate that the new TRI-GNN outperforms all 14
state-of-the-art baselines on 6 out 7 graphs and exhibit higher robustness to
perturbations, yielding up to 10\% better performance under noisy scenarios.

    

### [[2110.15538] Model Fusion of Heterogeneous Neural Networks via Cross-Layer Alignment](http://arxiv.org/abs/2110.15538)


  Layer-wise model fusion via optimal transport, named OTFusion, applies soft
neuron association for unifying different pre-trained networks to save
computational resources. While enjoying its success, OTFusion requires the
input networks to have the same number of layers. To address this issue, we
propose a novel model fusion framework, named CLAFusion, to fuse neural
networks with a different number of layers, which we refer to as heterogeneous
neural networks, via cross-layer alignment. The cross-layer alignment problem,
which is an unbalanced assignment problem, can be solved efficiently using
dynamic programming. Based on the cross-layer alignment, our framework balances
the number of layers of neural networks before applying layer-wise model
fusion. Our synthetic experiments indicate that the fused network from
CLAFusion achieves a more favorable performance compared to the individual
networks trained on heterogeneous data without the need for any retraining.
With an extra fine-tuning process, it improves the accuracy of residual
networks on the CIFAR10 dataset. Finally, we explore its application for model
compression and knowledge distillation when applying to the teacher-student
setting.

    

### [[2110.15542] Automatic Hand Sign Recognition: Identify Unusuality through Latent Cognizance](http://arxiv.org/abs/2110.15542)


  Sign language is a main communication channel among hearing disability
community. Automatic sign language transcription could facilitate better
communication and understanding between hearing disability community and
hearing majority. As a recent work in automatic sign language transcription has
discussed, effectively handling or identifying a non-sign posture is one of the
key issues. A non-sign posture is a posture unintended for sign reading and
does not belong to any valid sign. A non-sign posture may arise during sign
transition or simply from an unaware posture. Confidence ratio has been
proposed to mitigate the issue. Confidence ratio is simple to compute and
readily available without extra training. However, confidence ratio is reported
to only partially address the problem. In addition, confidence ratio
formulation is susceptible to computational instability. This article proposes
alternative formulations to confidence ratio, investigates an issue of non-sign
identification for Thai Finger Spelling recognition, explores potential
solutions and has found a promising direction. Not only does this finding
address the issue of non-sign identification, it also provide some insight
behind a well-learned inference machine, revealing hidden meaning and new
interpretation of the underlying mechanism. Our proposed methods are evaluated
and shown to be effective for non-sign detection.

    

### [[2110.15545] Improving Fairness via Federated Learning](http://arxiv.org/abs/2110.15545)


  Recently, lots of algorithms have been proposed for learning a fair
classifier from centralized data. However, how to privately train a fair
classifier on decentralized data has not been fully studied yet. In this work,
we first propose a new theoretical framework, with which we analyze the value
of federated learning in improving fairness. Our analysis reveals that
federated learning can strictly boost model fairness compared with all
non-federated algorithms. We then theoretically and empirically show that the
performance tradeoff of FedAvg-based fair learning algorithms is strictly worse
than that of a fair classifier trained on centralized data. To resolve this, we
propose FedFB, a private fair learning algorithm on decentralized data with a
modified FedAvg protocol. Our extensive experimental results show that FedFB
significantly outperforms existing approaches, sometimes achieving a similar
tradeoff as the one trained on centralized data.

    

### [[2110.15547] Does Momentum Help? A Sample Complexity Analysis](http://arxiv.org/abs/2110.15547)


  Momentum methods are popularly used in accelerating stochastic iterative
methods. Although a fair amount of literature is dedicated to momentum in
stochastic optimisation, there are limited results that quantify the benefits
of using heavy ball momentum in the specific case of stochastic approximation
algorithms. We first show that the convergence rate with optimal step size does
not improve when momentum is used (under some assumptions). Secondly, to
quantify the behaviour in the initial phase we analyse the sample complexity of
iterates with and without momentum. We show that the sample complexity bound
for SA without momentum is
$\tilde{\mathcal{O}}(\frac{1}{\alpha\lambda_{min}(A)})$ while for SA with
momentum is $\tilde{\mathcal{O}}(\frac{1}{\sqrt{\alpha\lambda_{min}(A)}})$,
where $\alpha$ is the step size and $\lambda_{min}(A)$ is the smallest
eigenvalue of the driving matrix $A$. Although the sample complexity bound for
SA with momentum is better for small enough $\alpha$, it turns out that for
optimal choice of $\alpha$ in the two cases, the sample complexity bounds are
of the same order.

    

### [[2110.15548] Latent Cognizance: What Machine Really Learns](http://arxiv.org/abs/2110.15548)


  Despite overwhelming achievements in recognition accuracy, extending an
open-set capability -- ability to identify when the question is out of scope --
remains greatly challenging in a scalable machine learning inference. A recent
research has discovered Latent Cognizance (LC) -- an insight on a recognition
mechanism based on a new probabilistic interpretation, Bayesian theorem, and an
analysis of an internal structure of a commonly-used recognition inference
structure. The new interpretation emphasizes a latent assumption of an
overlooked probabilistic condition on a learned inference model. Viability of
LC has been shown on a task of sign language recognition, but its potential and
implication can reach far beyond a specific domain and can move object
recognition toward a scalable open-set recognition. However, LC new
probabilistic interpretation has not been directly investigated. This article
investigates the new interpretation under a traceable context. Our findings
support the rationale on which LC is based and reveal a hidden mechanism
underlying the learning classification inference. The ramification of these
findings could lead to a simple yet effective solution to an open-set
recognition.

    

### [[2110.15552] A Comprehensive Study on Learning-Based PE Malware Family Classification Methods](http://arxiv.org/abs/2110.15552)


  Driven by the high profit, Portable Executable (PE) malware has been
consistently evolving in terms of both volume and sophistication. PE malware
family classification has gained great attention and a large number of
approaches have been proposed. With the rapid development of machine learning
techniques and the exciting results they achieved on various tasks, machine
learning algorithms have also gained popularity in the PE malware family
classification task. Three mainstream approaches that use learning based
algorithms, as categorized by the input format the methods take, are
image-based, binary-based and disassembly-based approaches. Although a large
number of approaches are published, there is no consistent comparisons on those
approaches, especially from the practical industry adoption perspective.
Moreover, there is no comparison in the scenario of concept drift, which is a
fact for the malware classification task due to the fast evolving nature of
malware. In this work, we conduct a thorough empirical study on learning-based
PE malware classification approaches on 4 different datasets and consistent
experiment settings. Based on the experiment results and an interview with our
industry partners, we find that (1) there is no individual class of methods
that significantly outperforms the others; (2) All classes of methods show
performance degradation on concept drift (by an average F1-score of 32.23%);
and (3) the prediction time and high memory consumption hinder existing
approaches from being adopted for industry usage.

    

### [[2110.15557] Crowd-sensing Enhanced Parking Patrol using Sharing Bikes' Trajectories](http://arxiv.org/abs/2110.15557)


  Illegal vehicle parking is a common urban problem faced by major cities in
the world, as it incurs traffic jams, which lead to air pollution and traffic
accidents. The government highly relies on active human efforts to detect
illegal parking events. However, such an approach is extremely ineffective to
cover a large city since the police have to patrol over the entire city roads.
The massive and high-quality sharing bike trajectories from Mobike offer us a
unique opportunity to design a ubiquitous illegal parking detection approach,
as most of the illegal parking events happen at curbsides and have significant
impact on the bike users. The detection result can guide the patrol schedule,
i.e. send the patrol policemen to the region with higher illegal parking risks,
and further improve the patrol efficiency. Inspired by this idea, three main
components are employed in the proposed framework: 1)~{\em trajectory
pre-processing}, which filters outlier GPS points, performs map-matching, and
builds trajectory indexes; 2)~{\em illegal parking detection}, which models the
normal trajectories, extracts features from the evaluation trajectories, and
utilizes a distribution test-based method to discover the illegal parking
events; and 3)~{\em patrol scheduling}, which leverages the detection result as
reference context, and models the scheduling task as a multi-agent
reinforcement learning problem to guide the patrol police. Finally, extensive
experiments are presented to validate the effectiveness of illegal parking
detection, as well as the improvement of patrol efficiency.

    

### [[2110.15572] Understanding the Effect of Stochasticity in Policy Optimization](http://arxiv.org/abs/2110.15572)


  We study the effect of stochasticity in on-policy policy optimization, and
make the following four contributions. First, we show that the preferability of
optimization methods depends critically on whether stochastic versus exact
gradients are used. In particular, unlike the true gradient setting, geometric
information cannot be easily exploited in the stochastic case for accelerating
policy optimization without detrimental consequences or impractical
assumptions. Second, to explain these findings we introduce the concept of
committal rate for stochastic policy optimization, and show that this can serve
as a criterion for determining almost sure convergence to global optimality.
Third, we show that in the absence of external oracle information, which allows
an algorithm to determine the difference between optimal and sub-optimal
actions given only on-policy samples, there is an inherent trade-off between
exploiting geometry to accelerate convergence versus achieving optimality
almost surely. That is, an uninformed algorithm either converges to a globally
optimal policy with probability $1$ but at a rate no better than $O(1/t)$, or
it achieves faster than $O(1/t)$ convergence but then must fail to converge to
the globally optimal policy with some positive probability. Finally, we use the
committal rate theory to explain why practical policy optimization methods are
sensitive to random initialization, then develop an ensemble method that can be
guaranteed to achieve near-optimal solutions with high probability.

    

### [[2110.15573] A/B/n Testing with Control in the Presence of Subpopulations](http://arxiv.org/abs/2110.15573)


  Motivated by A/B/n testing applications, we consider a finite set of
distributions (called \emph{arms}), one of which is treated as a
\emph{control}. We assume that the population is stratified into homogeneous
subpopulations. At every time step, a subpopulation is sampled and an arm is
chosen: the resulting observation is an independent draw from the arm
conditioned on the subpopulation. The quality of each arm is assessed through a
weighted combination of its subpopulation means. We propose a strategy for
sequentially choosing one arm per time step so as to discover as fast as
possible which arms, if any, have higher weighted expectation than the control.
This strategy is shown to be asymptotically optimal in the following sense: if
$\tau_\delta$ is the first time when the strategy ensures that it is able to
output the correct answer with probability at least $1-\delta$, then
$\mathbb{E}[\tau_\delta]$ grows linearly with $\log(1/\delta)$ at the exact
optimal rate. This rate is identified in the paper in three different settings:
(1) when the experimenter does not observe the subpopulation information, (2)
when the subpopulation of each sample is observed but not chosen, and (3) when
the experimenter can select the subpopulation from which each response is
sampled. We illustrate the efficiency of the proposed strategy with numerical
simulations on synthetic and real data collected from an A/B/n experiment.

    

### [[2110.15596] Training Integrable Parameterizations of Deep Neural Networks in the Infinite-Width Limit](http://arxiv.org/abs/2110.15596)


  To theoretically understand the behavior of trained deep neural networks, it
is necessary to study the dynamics induced by gradient methods from a random
initialization. However, the nonlinear and compositional structure of these
models make these dynamics difficult to analyze. To overcome these challenges,
large-width asymptotics have recently emerged as a fruitful viewpoint and led
to practical insights on real-world deep networks. For two-layer neural
networks, it has been understood via these asymptotics that the nature of the
trained model radically changes depending on the scale of the initial random
weights, ranging from a kernel regime (for large initial variance) to a feature
learning regime (for small initial variance). For deeper networks more regimes
are possible, and in this paper we study in detail a specific choice of "small"
initialization corresponding to ''mean-field'' limits of neural networks, which
we call integrable parameterizations (IPs). First, we show that under standard
i.i.d. zero-mean initialization, integrable parameterizations of neural
networks with more than four layers start at a stationary point in the
infinite-width limit and no learning occurs. We then propose various methods to
avoid this trivial behavior and analyze in detail the resulting dynamics. In
particular, one of these methods consists in using large initial learning
rates, and we show that it is equivalent to a modification of the recently
proposed maximal update parameterization $\mu$P. We confirm our results with
numerical experiments on image classification tasks, which additionally show a
strong difference in behavior between various choices of activation functions
that is not yet captured by theory.

    

### [[2110.15622] Path-Enhanced Multi-Relational Question Answering with Knowledge Graph Embeddings](http://arxiv.org/abs/2110.15622)


  The multi-relational Knowledge Base Question Answering (KBQA) system performs
multi-hop reasoning over the knowledge graph (KG) to achieve the answer. Recent
approaches attempt to introduce the knowledge graph embedding (KGE) technique
to handle the KG incompleteness but only consider the triple facts and neglect
the significant semantic correlation between paths and multi-relational
questions. In this paper, we propose a Path and Knowledge Embedding-Enhanced
multi-relational Question Answering model (PKEEQA), which leverages multi-hop
paths between entities in the KG to evaluate the ambipolar correlation between
a path embedding and a multi-relational question embedding via a customizable
path representation mechanism, benefiting for achieving more accurate answers
from the perspective of both the triple facts and the extra paths. Experimental
results illustrate that PKEEQA improves KBQA models' performance for
multi-relational question answering with explainability to some extent derived
from paths.

    

### [[2110.15632] Bayesian Optimal Experimental Design for Simulator Models of Cognition](http://arxiv.org/abs/2110.15632)


  Bayesian optimal experimental design (BOED) is a methodology to identify
experiments that are expected to yield informative data. Recent work in
cognitive science considered BOED for computational models of human behavior
with tractable and known likelihood functions. However, tractability often
comes at the cost of realism; simulator models that can capture the richness of
human behavior are often intractable. In this work, we combine recent advances
in BOED and approximate inference for intractable models, using
machine-learning methods to find optimal experimental designs, approximate
sufficient summary statistics and amortized posterior distributions. Our
simulation experiments on multi-armed bandit tasks show that our method results
in improved model discrimination and parameter estimation, as compared to
experimental designs commonly used in the literature.

    

### [[2110.15660] Frame-Capture-Based CSI Recomposition Pertaining to Firmware-Agnostic WiFi Sensing](http://arxiv.org/abs/2110.15660)


  With regard to the implementation of WiFi sensing agnostic according to the
availability of channel state information (CSI), we investigate the possibility
of estimating a CSI matrix based on its compressed version, which is known as
beamforming feedback matrix (BFM). Being different from the CSI matrix that is
processed and discarded in physical layer components, the BFM can be captured
using a medium-access-layer frame-capturing technique because this is exchanged
among an access point (AP) and stations (STAs) over the air. This indicates
that WiFi sensing that leverages the BFM matrix is more practical to implement
using the pre-installed APs. However, the ability of BFM-based sensing has been
evaluated in a few tasks, and more general insights into its performance should
be provided. To fill this gap, we propose a CSI estimation method based on BFM,
approximating the estimation function with a machine learning model. In
addition, to improve the estimation accuracy, we leverage the inter-subcarrier
dependency using the BFMs at multiple subcarriers in orthogonal frequency
division multiplexing transmissions. Our simulation evaluation reveals that the
estimated CSI matches the ground-truth amplitude. Moreover, compared to CSI
estimation at each individual subcarrier, the effect of the BFMs at multiple
subcarriers on the CSI estimation accuracy is validated.

    

### [[2110.15667] QDCNN: Quantum Dilated Convolutional Neural Network](http://arxiv.org/abs/2110.15667)


  In recent years, with rapid progress in the development of quantum
technologies, quantum machine learning has attracted a lot of interest. In
particular, a family of hybrid quantum-classical neural networks, consisting of
classical and quantum elements, has been massively explored for the purpose of
improving the performance of classical neural networks. In this paper, we
propose a novel hybrid quantum-classical algorithm called quantum dilated
convolutional neural networks (QDCNNs). Our method extends the concept of
dilated convolution, which has been widely applied in modern deep learning
algorithms, to the context of hybrid neural networks. The proposed QDCNNs are
able to capture larger context during the quantum convolution process while
reducing the computational cost. We perform empirical experiments on MNIST and
Fashion-MNIST datasets for the task of image recognition and demonstrate that
QDCNN models generally enjoy better performances in terms of both accuracy and
computation efficiency compared to existing quantum convolutional neural
networks (QCNNs).

    

### [[2110.15672] Improved FRQI on superconducting processors and its restrictions in the NISQ era](http://arxiv.org/abs/2110.15672)


  In image processing, the amount of data to be processed grows rapidly, in
particular when imaging methods yield images of more than two dimensions or
time series of images. Thus, efficient processing is a challenge, as data sizes
may push even supercomputers to their limits. Quantum image processing promises
to encode images with logarithmically less qubits than classical pixels in the
image. In theory, this is a huge progress, but so far not many experiments have
been conducted in practice, in particular on real backends. Often, the precise
conversion of classical data to quantum states, the exact implementation, and
the interpretation of the measurements in the classical context are
challenging. We investigate these practical questions in this paper. In
particular, we study the feasibility of the Flexible Representation of Quantum
Images (FRQI). Furthermore, we check experimentally what is the limit in the
current noisy intermediate-scale quantum era, i.e. up to which image size an
image can be encoded, both on simulators and on real backends. Finally, we
propose a method for simplifying the circuits needed for the FRQI. With our
alteration, the number of gates needed, especially of the error-prone
controlled-NOT gates, can be reduced. As a consequence, the size of manageable
images increases.

    

### [[2110.15688] Variational Bayesian Optimistic Sampling](http://arxiv.org/abs/2110.15688)


  We consider online sequential decision problems where an agent must balance
exploration and exploitation. We derive a set of Bayesian `optimistic' policies
which, in the stochastic multi-armed bandit case, includes the Thompson
sampling policy. We provide a new analysis showing that any algorithm producing
policies in the optimistic set enjoys $\tilde O(\sqrt{AT})$ Bayesian regret for
a problem with $A$ actions after $T$ rounds. We extend the regret analysis for
optimistic policies to bilinear saddle-point problems which include zero-sum
matrix games and constrained bandits as special cases. In this case we show
that Thompson sampling can produce policies outside of the optimistic set and
suffer linear regret in some instances. Finding a policy inside the optimistic
set amounts to solving a convex optimization problem and we call the resulting
algorithm `variational Bayesian optimistic sampling' (VBOS). The procedure
works for any posteriors, \ie, it does not require the posterior to have any
special properties, such as log-concavity, unimodality, or smoothness. The
variational view of the problem has many useful properties, including the
ability to tune the exploration-exploitation tradeoff, add regularization,
incorporate constraints, and linearly parameterize the policy.

    

### [[2110.15700] Boosting Anomaly Detection Using Unsupervised Diverse Test-Time Augmentation](http://arxiv.org/abs/2110.15700)


  Anomaly detection is a well-known task that involves the identification of
abnormal events that occur relatively infrequently. Methods for improving
anomaly detection performance have been widely studied. However, no studies
utilizing test-time augmentation (TTA) for anomaly detection in tabular data
have been performed. TTA involves aggregating the predictions of several
synthetic versions of a given test sample; TTA produces different points of
view for a specific test instance and might decrease its prediction bias. We
propose the Test-Time Augmentation for anomaly Detection (TTAD) technique, a
TTA-based method aimed at improving anomaly detection performance. TTAD
augments a test instance based on its nearest neighbors; various methods,
including the k-Means centroid and SMOTE methods, are used to produce the
augmentations. Our technique utilizes a Siamese network to learn an advanced
distance metric when retrieving a test instance's neighbors. Our experiments
show that the anomaly detector that uses our TTA technique achieved
significantly higher AUC results on all datasets evaluated.

    

### [[2110.15701] Xi-Learning: Successor Feature Transfer Learning for General Reward Functions](http://arxiv.org/abs/2110.15701)


  Transfer in Reinforcement Learning aims to improve learning performance on
target tasks using knowledge from experienced source tasks. Successor features
(SF) are a prominent transfer mechanism in domains where the reward function
changes between tasks. They reevaluate the expected return of previously
learned policies in a new target task and to transfer their knowledge. A
limiting factor of the SF framework is its assumption that rewards linearly
decompose into successor features and a reward weight vector. We propose a
novel SF mechanism, $\xi$-learning, based on learning the cumulative discounted
probability of successor features. Crucially, $\xi$-learning allows to
reevaluate the expected return of policies for general reward functions. We
introduce two $\xi$-learning variations, prove its convergence, and provide a
guarantee on its transfer performance. Experimental evaluations based on
$\xi$-learning with function approximation demonstrate the prominent advantage
of $\xi$-learning over available mechanisms not only for general reward
functions, but also in the case of linearly decomposable reward functions.

    

### [[2110.15703] Navigating the Kaleidoscope of COVID-19 Misinformation Using Deep Learning](http://arxiv.org/abs/2110.15703)


  Irrespective of the success of the deep learning-based mixed-domain transfer
learning approach for solving various Natural Language Processing tasks, it
does not lend a generalizable solution for detecting misinformation from
COVID-19 social media data. Due to the inherent complexity of this type of
data, caused by its dynamic (context evolves rapidly), nuanced (misinformation
types are often ambiguous), and diverse (skewed, fine-grained, and overlapping
categories) nature, it is imperative for an effective model to capture both the
local and global context of the target domain. By conducting a systematic
investigation, we show that: (i) the deep Transformer-based pre-trained models,
utilized via the mixed-domain transfer learning, are only good at capturing the
local context, thus exhibits poor generalization, and (ii) a combination of
shallow network-based domain-specific models and convolutional neural networks
can efficiently extract local as well as global context directly from the
target data in a hierarchical fashion, enabling it to offer a more
generalizable solution.

    

### [[2110.15709] LegalNLP -- Natural Language Processing methods for the Brazilian Legal Language](http://arxiv.org/abs/2110.15709)


  We present and make available pre-trained language models (Phraser, Word2Vec,
Doc2Vec, FastText, and BERT) for the Brazilian legal language, a Python package
with functions to facilitate their use, and a set of demonstrations/tutorials
containing some applications involving them. Given that our material is built
upon legal texts coming from several Brazilian courts, this initiative is
extremely helpful for the Brazilian legal field, which lacks other open and
specific tools and language models. Our main objective is to catalyze the use
of natural language processing tools for legal texts analysis by the Brazilian
industry, government, and academia, providing the necessary tools and
accessible material.

    

### [[2110.15710] Classification of hierarchical text using geometric deep learning: the case of clinical trials corpus](http://arxiv.org/abs/2110.15710)


  We consider the hierarchical representation of documents as graphs and use
geometric deep learning to classify them into different categories. While graph
neural networks can efficiently handle the variable structure of hierarchical
documents using the permutation invariant message passing operations, we show
that we can gain extra performance improvements using our proposed selective
graph pooling operation that arises from the fact that some parts of the
hierarchy are invariable across different documents. We applied our model to
classify clinical trial (CT) protocols into completed and terminated
categories. We use bag-of-words based, as well as pre-trained transformer-based
embeddings to featurize the graph nodes, achieving f1-scores around 0.85 on a
publicly available large scale CT registry of around 360K protocols. We further
demonstrate how the selective pooling can add insights into the CT termination
status prediction. We make the source code and dataset splits accessible.

    

### [[2110.15717] LIDSNet: A Lightweight on-device Intent Detection model using Deep Siamese Network](http://arxiv.org/abs/2110.15717)


  Intent detection is a crucial task in any Natural Language Understanding
(NLU) system and forms the foundation of a task-oriented dialogue system. To
build high-quality real-world conversational solutions for edge devices, there
is a need for deploying intent detection model on device. This necessitates a
light-weight, fast, and accurate model that can perform efficiently in a
resource-constrained environment. To this end, we propose LIDSNet, a novel
lightweight on-device intent detection model, which accurately predicts the
message intent by utilizing a Deep Siamese Network for learning better sentence
representations. We use character-level features to enrich the sentence-level
representations and empirically demonstrate the advantage of transfer learning
by utilizing pre-trained embeddings. Furthermore, to investigate the efficacy
of the modules in our architecture, we conduct an ablation study and arrive at
our optimal model. Experimental results prove that LIDSNet achieves
state-of-the-art competitive accuracy of 98.00% and 95.97% on SNIPS and ATIS
public datasets respectively, with under 0.59M parameters. We further benchmark
LIDSNet against fine-tuned BERTs and show that our model is at least 41x
lighter and 30x faster during inference than MobileBERT on Samsung Galaxy S20
device, justifying its efficiency on resource-constrained edge devices.

    

### [[2110.15718] Deep convolutional forest: a dynamic deep ensemble approach for spam detection in text](http://arxiv.org/abs/2110.15718)


  The increase in people's use of mobile messaging services has led to the
spread of social engineering attacks like phishing, considering that spam text
is one of the main factors in the dissemination of phishing attacks to steal
sensitive data such as credit cards and passwords. In addition, rumors and
incorrect medical information regarding the COVID-19 pandemic are widely shared
on social media leading to people's fear and confusion. Thus, filtering spam
content is vital to reduce risks and threats. Previous studies relied on
machine learning and deep learning approaches for spam classification, but
these approaches have two limitations. Machine learning models require manual
feature engineering, whereas deep neural networks require a high computational
cost. This paper introduces a dynamic deep ensemble model for spam detection
that adjusts its complexity and extracts features automatically. The proposed
model utilizes convolutional and pooling layers for feature extraction along
with base classifiers such as random forests and extremely randomized trees for
classifying texts into spam or legitimate ones. Moreover, the model employs
ensemble learning procedures like boosting and bagging. As a result, the model
achieved high precision, recall, f1-score and accuracy of 98.38%.

    

### [[2110.15723] SP-GPT2: Semantics Improvement in Vietnamese Poetry Generation](http://arxiv.org/abs/2110.15723)


  Automatic text generation has garnered growing attention in recent years as
an essential step towards computer creativity. Generative Pretraining
Transformer 2 (GPT2) is one of the state of the art approaches that have
excellent successes. In this paper, we took the first step to investigate the
power of GPT2 in traditional Vietnamese poetry generation. In the earlier time,
our experiment with base GPT2 was quite good at generating the poem in the
proper template. Though it can learn the patterns, including rhyme and tone
rules, from the training data, like almost all other text generation
approaches, the poems generated still has a topic drift and semantic
inconsistency. To improve the cohesion within the poems, we proposed a new
model SP-GPT2 (semantic poem GPT2) which was built on the top GPT2 model and an
additional loss to constrain context throughout the entire poem. For better
evaluation, we examined the methods by both automatic quantitative evaluation
and human evaluation. Both automatic and human evaluation demonstrated that our
approach can generate poems that have better cohesion without losing the
quality due to additional loss. At the same time, we are the pioneers of this
topic. We released the first computational scoring module for poems generated
in the template containing the style rule dictionary. Additionally, we are the
first to publish a Luc-Bat dataset, including 87609 Luc Bat poems, which is
equivalent to about 2.6 million sentences, combined with about 83579 poems in
other styles was also published for further exploration. The code is available
at this https URL


### [[2110.15724] Learning to Learn End-to-End Goal-Oriented Dialog From Related Dialog Tasks](http://arxiv.org/abs/2110.15724)


  For each goal-oriented dialog task of interest, large amounts of data need to
be collected for end-to-end learning of a neural dialog system. Collecting that
data is a costly and time-consuming process. Instead, we show that we can use
only a small amount of data, supplemented with data from a related dialog task.
Naively learning from related data fails to improve performance as the related
data can be inconsistent with the target task. We describe a meta-learning
based method that selectively learns from the related dialog task data. Our
approach leads to significant accuracy improvements in an example dialog task.

    

### [[2110.15725] Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks](http://arxiv.org/abs/2110.15725)


  The use of contrastive loss for representation learning has become prominent
in computer vision, and it is now getting attention in Natural Language
Processing (NLP). Here, we explore the idea of using a batch-softmax
contrastive loss when fine-tuning large-scale pre-trained transformer models to
learn better task-specific sentence embeddings for pairwise sentence scoring
tasks. We introduce and study a number of variations in the calculation of the
loss as well as in the overall training procedure; in particular, we find that
data shuffling can be quite important. Our experimental results show sizable
improvements on a number of datasets and pairwise sentence scoring tasks
including classification, ranking, and regression. Finally, we offer detailed
analysis and discussion, which should be useful for researchers aiming to
explore the utility of contrastive loss in NLP.

    

### [[2110.15728] Deep Learning for Bias Detection: From Inception to Deployment](http://arxiv.org/abs/2110.15728)


  To create a more inclusive workplace, enterprises are actively investing in
identifying and eliminating unconscious bias (e.g., gender, race, age,
disability, elitism and religion) across their various functions. We propose a
deep learning model with a transfer learning based language model to learn from
manually tagged documents for automatically identifying bias in enterprise
content. We first pretrain a deep learning-based language-model using
Wikipedia, then fine tune the model with a large unlabelled data set related
with various types of enterprise content. Finally, a linear layer followed by
softmax layer is added at the end of the language model and the model is
trained on a labelled bias dataset consisting of enterprise content. The
trained model is thoroughly evaluated on independent datasets to ensure a
general application. We present the proposed method and its deployment detail
in a real-world application.

    

### [[2110.15733] Detecting Gender Bias in Transformer-based Models: A Case Study on BERT](http://arxiv.org/abs/2110.15733)


  In this paper, we propose a novel gender bias detection method by utilizing
attention map for transformer-based models. We 1) give an intuitive gender bias
judgement method by comparing the different relation degree between the genders
and the occupation according to the attention scores, 2) design a gender bias
detector by modifying the attention module, 3) insert the gender bias detector
into different positions of the model to present the internal gender bias flow,
and 4) draw the consistent gender bias conclusion by scanning the entire
Wikipedia, a BERT pretraining dataset. We observe that 1) the attention
matrices, Wq and Wk introduce much more gender bias than other modules
(including the embedding layer) and 2) the bias degree changes periodically
inside of the model (attention matrix Q, K, V, and the remaining part of the
attention layer (including the fully-connected layer, the residual connection,
and the layer normalization module) enhance the gender bias while the averaged
attentions reduces the bias).

    

### [[2110.15739] Scalable Inference in SDEs by Direct Matching of the Fokker-Planck-Kolmogorov Equation](http://arxiv.org/abs/2110.15739)


  Simulation-based techniques such as variants of stochastic Runge-Kutta are
the de facto approach for inference with stochastic differential equations
(SDEs) in machine learning. These methods are general-purpose and used with
parametric and non-parametric models, and neural SDEs. Stochastic Runge-Kutta
relies on the use of sampling schemes that can be inefficient in high
dimensions. We address this issue by revisiting the classical SDE literature
and derive direct approximations to the (typically intractable)
Fokker-Planck-Kolmogorov equation by matching moments. We show how this
workflow is fast, scales to high-dimensional latent spaces, and is applicable
to scarce-data applications, where a non-parametric SDE with a driving Gaussian
process velocity field specifies the model.

    

### [[2110.15742] Barlow Graph Auto-Encoder for Unsupervised Network Embedding](http://arxiv.org/abs/2110.15742)


  Network embedding has emerged as a promising research field for network
analysis. Recently, an approach, named Barlow Twins, has been proposed for
self-supervised learning in computer vision by applying the
redundancy-reduction principle to the embedding vectors corresponding to two
distorted versions of the image samples. Motivated by this, we propose Barlow
Graph Auto-Encoder, a simple yet effective architecture for learning network
embedding. It aims to maximize the similarity between the embedding vectors of
immediate and larger neighborhoods of a node, while minimizing the redundancy
between the components of these projections. In addition, we also present the
variation counterpart named as Barlow Variational Graph Auto-Encoder. Our
approach yields promising results for inductive link prediction and is also on
par with state of the art for clustering and downstream node classification, as
demonstrated by extensive comparisons with several well-known techniques on
three benchmark citation datasets.

    

### [[2110.15761] Aligned Multi-Task Gaussian Process](http://arxiv.org/abs/2110.15761)


  Multi-task learning requires accurate identification of the correlations
between tasks. In real-world time-series, tasks are rarely perfectly temporally
aligned; traditional multi-task models do not account for this and subsequent
errors in correlation estimation will result in poor predictive performance and
uncertainty quantification. We introduce a method that automatically accounts
for temporal misalignment in a unified generative model that improves
predictive performance. Our method uses Gaussian processes (GPs) to model the
correlations both within and between the tasks. Building on the previous work
by Kazlauskaiteet al. [2019], we include a separate monotonic warp of the input
data to model temporal misalignment. In contrast to previous work, we formulate
a lower bound that accounts for uncertainty in both the estimates of the
warping process and the underlying functions. Also, our new take on a monotonic
stochastic process, with efficient path-wise sampling for the warp functions,
allows us to perform full Bayesian inference in the model rather than MAP
estimates. Missing data experiments, on synthetic and real time-series,
demonstrate the advantages of accounting for misalignments (vs standard
unaligned method) as well as modelling the uncertainty in the warping
process(vs baseline MAP alignment approach).

    

### [[2110.15762] Mixed Cooperative-Competitive Communication Using Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2110.15762)


  By using communication between multiple agents in multi-agent environments,
one can reduce the effects of partial observability by combining one agent's
observation with that of others in the same dynamic environment. While a lot of
successful research has been done towards communication learning in cooperative
settings, communication learning in mixed cooperative-competitive settings is
also important and brings its own complexities such as the opposing team
overhearing the communication. In this paper, we apply differentiable
inter-agent learning (DIAL), designed for cooperative settings, to a mixed
cooperative-competitive setting. We look at the difference in performance
between communication that is private for a team and communication that can be
overheard by the other team. Our research shows that communicating agents are
able to achieve similar performance to fully observable agents after a given
training period in our chosen environment. Overall, we find that sharing
communication across teams results in decreased performance for the
communicating team in comparison to results achieved with private
communication.

    

### [[2110.15764] ε-weakened Robustness of Deep Neural Networks](http://arxiv.org/abs/2110.15764)


  This paper introduces a notation of $\varepsilon$-weakened robustness for
analyzing the reliability and stability of deep neural networks (DNNs). Unlike
the conventional robustness, which focuses on the "perfect" safe region in the
absence of adversarial examples, $\varepsilon$-weakened robustness focuses on
the region where the proportion of adversarial examples is bounded by
user-specified $\varepsilon$. Smaller $\varepsilon$ means a smaller chance of
failure. Under such robustness definition, we can give conclusive results for
the regions where conventional robustness ignores. We prove that the
$\varepsilon$-weakened robustness decision problem is PP-complete and give a
statistical decision algorithm with user-controllable error bound. Furthermore,
we derive an algorithm to find the maximum $\varepsilon$-weakened robustness
radius. The time complexity of our algorithms is polynomial in the dimension
and size of the network. So, they are scalable to large real-world networks.
Besides, We also show its potential application in analyzing quality issues.

    

### [[2110.15767] Adversarial Robustness with Semi-Infinite Constrained Learning](http://arxiv.org/abs/2110.15767)


  Despite strong performance in numerous applications, the fragility of deep
learning to input perturbations has raised serious questions about its use in
safety-critical domains. While adversarial training can mitigate this issue in
practice, state-of-the-art methods are increasingly application-dependent,
heuristic in nature, and suffer from fundamental trade-offs between nominal
performance and robustness. Moreover, the problem of finding worst-case
perturbations is non-convex and underparameterized, both of which engender a
non-favorable optimization landscape. Thus, there is a gap between the theory
and practice of adversarial training, particularly with respect to when and why
adversarial training works. In this paper, we take a constrained learning
approach to address these questions and to provide a theoretical foundation for
robust learning. In particular, we leverage semi-infinite optimization and
non-convex duality theory to show that adversarial training is equivalent to a
statistical problem over perturbation distributions, which we characterize
completely. Notably, we show that a myriad of previous robust training
techniques can be recovered for particular, sub-optimal choices of these
distributions. Using these insights, we then propose a hybrid Langevin Monte
Carlo approach of which several common algorithms (e.g., PGD) are special
cases. Finally, we show that our approach can mitigate the trade-off between
nominal and robust performance, yielding state-of-the-art results on MNIST and
CIFAR-10. Our code is available at: this https URL.

    

### [[2110.15771] Collaborative Pure Exploration in Kernel Bandit](http://arxiv.org/abs/2110.15771)


  In this paper, we formulate a Collaborative Pure Exploration in Kernel Bandit
problem (CoPE-KB), which provides a novel model for multi-agent multi-task
decision making under limited communication and general reward functions, and
is applicable to many online learning tasks, e.g., recommendation systems and
network scheduling. We consider two settings of CoPE-KB, i.e., Fixed-Confidence
(FC) and Fixed-Budget (FB), and design two optimal algorithms CoopKernelFC (for
FC) and CoopKernelFB (for FB). Our algorithms are equipped with innovative and
efficient kernelized estimators to simultaneously achieve computation and
communication efficiency. Matching upper and lower bounds under both the
statistical and communication metrics are established to demonstrate the
optimality of our algorithms. The theoretical bounds successfully quantify the
influences of task similarities on learning acceleration and only depend on the
effective dimension of the kernelized feature space. Our analytical techniques,
including data dimension decomposition, linear structured instance
transformation and (communication) round-speedup induction, are novel and
applicable to other bandit problems. Empirical evaluations are provided to
validate our theoretical results and demonstrate the performance superiority of
our algorithms.

    

### [[2110.15777] GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily](http://arxiv.org/abs/2110.15777)


  Graph Neural Networks (GNNs) are widely used on a variety of graph-based
machine learning tasks. For node-level tasks, GNNs have strong power to model
the homophily property of graphs (i.e., connected nodes are more similar) while
their ability to capture heterophily property is often doubtful. This is
partially caused by the design of the feature transformation with the same
kernel for the nodes in the same hop and the followed aggregation operator. One
kernel cannot model the similarity and the dissimilarity (i.e., the positive
and negative correlation) between node features simultaneously even though we
use attention mechanisms like Graph Attention Network (GAT), since the weight
calculated by attention is always a positive value. In this paper, we propose a
novel GNN model based on a bi-kernel feature transformation and a selection
gate. Two kernels capture homophily and heterophily information respectively,
and the gate is introduced to select which kernel we should use for the given
node pairs. We conduct extensive experiments on various datasets with different
homophily-heterophily properties. The experimental results show consistent and
significant improvements against state-of-the-art GNN methods.

    

### [[2110.15778] Comparing Machine Learning-Centered Approaches for Forecasting Language Patterns During Frustration in Early Childhood](http://arxiv.org/abs/2110.15778)


  When faced with self-regulation challenges, children have been known the use
their language to inhibit their emotions and behaviors. Yet, to date, there has
been a critical lack of evidence regarding what patterns in their speech
children use during these moments of frustration. In this paper, eXtreme
Gradient Boosting, Random Forest, Long Short-Term Memory Recurrent Neural
Networks, and Elastic Net Regression, have all been used to forecast these
language patterns in children. Based on the results of a comparative analysis
between these methods, the study reveals that when dealing with
high-dimensional and dense data, with very irregular and abnormal
distributions, as is the case with self-regulation patterns in children,
decision tree-based algorithms are able to outperform traditional regression
and neural network methods in their shortcomings.

    

### [[2110.15779] Learning to Communicate with Reinforcement Learning for an Adaptive Traffic Control System](http://arxiv.org/abs/2110.15779)


  Recent work in multi-agent reinforcement learning has investigated inter
agent communication which is learned simultaneously with the action policy in
order to improve the team reward. In this paper, we investigate independent
Q-learning (IQL) without communication and differentiable inter-agent learning
(DIAL) with learned communication on an adaptive traffic control system (ATCS).
In real world ATCS, it is impossible to present the full state of the
environment to every agent so in our simulation, the individual agents will
only have a limited observation of the full state of the environment. The ATCS
will be simulated using the Simulation of Urban MObility (SUMO) traffic
simulator in which two connected intersections are simulated. Every
intersection is controlled by an agent which has the ability to change the
direction of the traffic flow. Our results show that a DIAL agent outperforms
an independent Q-learner on both training time and on maximum achieved reward
as it is able to share relevant information with the other agents.

    

### [[2110.15781] Two-sided fairness in rankings via Lorenz dominance](http://arxiv.org/abs/2110.15781)


  We consider the problem of generating rankings that are fair towards both
users and item producers in recommender systems. We address both usual
recommendation (e.g., of music or movies) and reciprocal recommendation (e.g.,
dating). Following concepts of distributive justice in welfare economics, our
notion of fairness aims at increasing the utility of the worse-off individuals,
which we formalize using the criterion of Lorenz efficiency. It guarantees that
rankings are Pareto efficient, and that they maximally redistribute utility
from better-off to worse-off, at a given level of overall utility. We propose
to generate rankings by maximizing concave welfare functions, and develop an
efficient inference procedure based on the Frank-Wolfe algorithm. We prove that
unlike existing approaches based on fairness constraints, our approach always
produces fair rankings. Our experiments also show that it increases the utility
of the worse-off at lower costs in terms of overall utility.

    

### [[2110.15784] Convergence of Uncertainty Sampling for Active Learning](http://arxiv.org/abs/2110.15784)


  Uncertainty sampling in active learning is heavily used in practice to reduce
the annotation cost. However, there has been no wide consensus on the function
to be used for uncertainty estimation in binary classification tasks and
convergence guarantees of the corresponding active learning algorithms are not
well understood. The situation is even more challenging for multi-category
classification. In this work, we propose an efficient uncertainty estimator for
binary classification which we also extend to multiple classes, and provide a
non-asymptotic rate of convergence for our uncertainty sampling-based active
learning algorithm in both cases under no-noise conditions (i.e., linearly
separable data). We also extend our analysis to the noisy case and provide
theoretical guarantees for our algorithm under the influence of noise in the
task of binary and multi-class classification.

    

### [[2110.15789] On the Feasibility of Predicting Questions being Forgotten in Stack Overflow](http://arxiv.org/abs/2110.15789)


  For their attractiveness, comprehensiveness and dynamic coverage of relevant
topics, community-based question answering sites such as Stack Overflow heavily
rely on the engagement of their communities: Questions on new technologies,
technology features as well as technology versions come up and have to be
answered as technology evolves (and as community members gather experience with
it). At the same time, other questions cease in importance over time, finally
becoming irrelevant to users. Beyond filtering low-quality questions,
"forgetting" questions, which have become redundant, is an important step for
keeping the Stack Overflow content concise and useful. In this work, we study
this managed forgetting task for Stack Overflow. Our work is based on data from
more than a decade (2008 - 2019) - covering 18.1M questions, that are made
publicly available by the site itself. For establishing a deeper understanding,
we first analyze and characterize the set of questions about to be forgotten,
i.e., questions that get a considerable number of views in the current period
but become unattractive in the near future. Subsequently, we examine the
capability of a wide range of features in predicting such forgotten questions
in different categories. We find some categories in which those questions are
more predictable. We also discover that the text-based features are
surprisingly not helpful in this prediction task, while the meta information is
much more predictive.

    

### [[2110.15796] Properties from Mechanisms: An Equivariance Perspective on Identifiable Representation Learning](http://arxiv.org/abs/2110.15796)


  A key goal of unsupervised representation learning is "inverting" a data
generating process to recover its latent properties. Existing work that
provably achieves this goal relies on strong assumptions on relationships
between the latent variables (e.g., independence conditional on auxiliary
information). In this paper, we take a very different perspective on the
problem and ask, "Can we instead identify latent properties by leveraging
knowledge of the mechanisms that govern their evolution?" We provide a complete
characterization of the sources of non-identifiability as we vary knowledge
about a set of possible mechanisms. In particular, we prove that if we know the
exact mechanisms under which the latent properties evolve, then identification
can be achieved up to any equivariances that are shared by the underlying
mechanisms. We generalize this characterization to settings where we only know
some hypothesis class over possible mechanisms, as well as settings where the
mechanisms are stochastic. We demonstrate the power of this mechanism-based
perspective by showing that we can leverage our results to generalize existing
identifiable representation learning results. These results suggest that by
exploiting inductive biases on mechanisms, it is possible to design a range of
new identifiable representation learning approaches.

    

### [[2110.15797] Discovering Non-monotonic Autoregressive Orderings with Variational Inference](http://arxiv.org/abs/2110.15797)


  The predominant approach for language modeling is to process sequences from
left to right, but this eliminates a source of information: the order by which
the sequence was generated. One strategy to recover this information is to
decode both the content and ordering of tokens. Existing approaches supervise
content and ordering by designing problem-specific loss functions and
pre-training with an ordering pre-selected. Other recent works use iterative
search to discover problem-specific orderings for training, but suffer from
high time complexity and cannot be efficiently parallelized. We address these
limitations with an unsupervised parallelizable learner that discovers
high-quality generation orders purely from training data -- no domain knowledge
required. The learner contains an encoder network and decoder language model
that perform variational inference with autoregressive orders (represented as
permutation matrices) as latent variables. The corresponding ELBO is not
differentiable, so we develop a practical algorithm for end-to-end optimization
using policy gradients. We implement the encoder as a Transformer with
non-causal attention that outputs permutations in one forward pass.
Permutations then serve as target generation orders for training an
insertion-based Transformer language model. Empirical results in language
modeling tasks demonstrate that our method is context-aware and discovers
orderings that are competitive with or even better than fixed orders.

    

### [[2110.15801] Application of the Multi-label Residual Convolutional Neural Network text classifier using Content-Based Routing process](http://arxiv.org/abs/2110.15801)


  In this article, we will present an NLP application in text classifying
process using the content-based router. The ultimate goal throughout this
article is to predict the event described by a legal ad from the plain text of
the ad. This problem is purely a supervised problem that will involve the use
of NLP techniques and conventional modeling methodologies through the use of
the Multi-label Residual Convolutional Neural Network for text classification.
We will explain the approach put in place to solve the problem of classified
ads, the difficulties encountered and the experimental results.

    

### [[2110.15802] BERMo: What can BERT learn from ELMo?](http://arxiv.org/abs/2110.15802)


  We propose BERMo, an architectural modification to BERT, which makes
predictions based on a hierarchy of surface, syntactic and semantic language
features. We use linear combination scheme proposed in Embeddings from Language
Models (ELMo) to combine the scaled internal representations from different
network depths. Our approach has two-fold benefits: (1) improved gradient flow
for the downstream task as every layer has a direct connection to the gradients
of the loss function and (2) increased representative power as the model no
longer needs to copy the features learned in the shallower layer which are
necessary for the downstream task. Further, our model has a negligible
parameter overhead as there is a single scalar parameter associated with each
layer in the network. Experiments on the probing task from SentEval dataset
show that our model performs up to $4.65\%$ better in accuracy than the
baseline with an average improvement of $2.67\%$ on the semantic tasks. When
subject to compression techniques, we find that our model enables stable
pruning for compressing small datasets like SST-2, where the BERT model
commonly diverges. We observe that our approach converges $1.67\times$ and
$1.15\times$ faster than the baseline on MNLI and QQP tasks from GLUE dataset.
Moreover, our results show that our approach can obtain better parameter
efficiency for penalty based pruning approaches on QQP task.

    

### [[2110.15821] Landscape analysis of an improved power method for tensor decomposition](http://arxiv.org/abs/2110.15821)


  In this work, we consider the optimization formulation for symmetric tensor
decomposition recently introduced in the Subspace Power Method (SPM) of Kileel
and Pereira. Unlike popular alternative functionals for tensor decomposition,
the SPM objective function has the desirable properties that its maximal value
is known in advance, and its global optima are exactly the rank-1 components of
the tensor when the input is sufficiently low-rank. We analyze the non-convex
optimization landscape associated with the SPM objective. Our analysis accounts
for working with noisy tensors. We derive quantitative bounds such that any
second-order critical point with SPM objective value exceeding the bound must
equal a tensor component in the noiseless case, and must approximate a tensor
component in the noisy case. For decomposing tensors of size $D^{\times m}$, we
obtain a near-global guarantee up to rank $\widetilde{o}(D^{\lfloor m/2
\rfloor})$ under a random tensor model, and a global guarantee up to rank
$\mathcal{O}(D)$ assuming deterministic frame conditions. This implies that SPM
with suitable initialization is a provable, efficient, robust algorithm for
low-rank symmetric tensor decomposition. We conclude with numerics that show a
practical preferability for using the SPM functional over a more established
counterpart.

    

### [[2110.15824] Tractability from overparametrization: The example of the negative perceptron](http://arxiv.org/abs/2110.15824)


  In the negative perceptron problem we are given $n$ data points
$({\boldsymbol x}_i,y_i)$, where ${\boldsymbol x}_i$ is a $d$-dimensional
vector and $y_i\in\{+1,-1\}$ is a binary label. The data are not linearly
separable and hence we content ourselves to find a linear classifier with the
largest possible \emph{negative} margin. In other words, we want to find a unit
norm vector ${\boldsymbol \theta}$ that maximizes $\min_{i\le n}y_i\langle
{\boldsymbol \theta},{\boldsymbol x}_i\rangle$. This is a non-convex
optimization problem (it is equivalent to finding a maximum norm vector in a
polytope), and we study its typical properties under two random models for the
data.
We consider the proportional asymptotics in which $n,d\to \infty$ with
$n/d\to\delta$, and prove upper and lower bounds on the maximum margin
$\kappa_{\text{s}}(\delta)$ or -- equivalently -- on its inverse function
$\delta_{\text{s}}(\kappa)$. In other words, $\delta_{\text{s}}(\kappa)$ is the
overparametrization threshold: for $n/d\le
\delta_{\text{s}}(\kappa)-\varepsilon$ a classifier achieving vanishing
training error exists with high probability, while for $n/d\ge
\delta_{\text{s}}(\kappa)+\varepsilon$ it does not. Our bounds on
$\delta_{\text{s}}(\kappa)$ match to the leading order as $\kappa\to -\infty$.
We then analyze a linear programming algorithm to find a solution, and
characterize the corresponding threshold $\delta_{\text{lin}}(\kappa)$. We
observe a gap between the interpolation threshold $\delta_{\text{s}}(\kappa)$
and the linear programming threshold $\delta_{\text{lin}}(\kappa)$, raising the
question of the behavior of other algorithms.

    

### [[2110.15828] Resampling Base Distributions of Normalizing Flows](http://arxiv.org/abs/2110.15828)


  Normalizing flows are a popular class of models for approximating probability
distributions. However, their invertible nature limits their ability to model
target distributions with a complex topological structure, such as Boltzmann
distributions. Several procedures have been proposed to solve this problem but
many of them sacrifice invertibility and, thereby, tractability of the
log-likelihood as well as other desirable properties. To address these
limitations, we introduce a base distribution for normalizing flows based on
learned rejection sampling, allowing the resulting normalizing flow to model
complex topologies without giving up bijectivity. Furthermore, we develop
suitable learning algorithms using both maximizing the log-likelihood and the
optimization of the reverse Kullback-Leibler divergence, and apply them to
various sample problems, i.e.\ approximating 2D densities, density estimation
of tabular data, image generation, and modeling Boltzmann distributions. In
these experiments our method is competitive with or outperforms the baselines.

    

### [[2110.15829] Holistic Deep Learning](http://arxiv.org/abs/2110.15829)


  There is much interest in deep learning to solve challenges that arise in
applying neural network models in real-world environments. In particular, three
areas have received considerable attention: adversarial robustness, parameter
sparsity, and output stability. Despite numerous attempts on solving these
problems independently, there is very little work addressing the challenges
simultaneously. In this paper, we address this problem of constructing holistic
deep learning models by proposing a novel formulation that solves these issues
in combination. Real-world experiments on both tabular and MNIST dataset show
that our formulation is able to simultaneously improve the accuracy,
robustness, stability, and sparsity over traditional deep learning models among
many others.

    

### [[2110.15832] CAN-PINN: A Fast Physics-Informed Neural Network Based on Coupled-Automatic-Numerical Differentiation Method](http://arxiv.org/abs/2110.15832)


  In this study, novel physics-informed neural network (PINN) methods for
coupling neighboring support points and automatic differentiation (AD) through
Taylor series expansion are proposed to allow efficient training with improved
accuracy. The computation of differential operators required for PINNs loss
evaluation at collocation points are conventionally obtained via AD. Although
AD has the advantage of being able to compute the exact gradients at any point,
such PINNs can only achieve high accuracies with large numbers of collocation
points, otherwise they are prone to optimizing towards unphysical solution. To
make PINN training fast, the dual ideas of using numerical differentiation
(ND)-inspired method and coupling it with AD are employed to define the loss
function. The ND-based formulation for training loss can strongly link
neighboring collocation points to enable efficient training in sparse sample
regimes, but its accuracy is restricted by the interpolation scheme. The
proposed coupled-automatic-numerical differentiation framework, labeled as
can-PINN, unifies the advantages of AD and ND, providing more robust and
efficient training than AD-based PINNs, while further improving accuracy by up
to 1-2 orders of magnitude relative to ND-based PINNs. For a proof-of-concept
demonstration of this can-scheme to fluid dynamic problems, two
numerical-inspired instantiations of can-PINN schemes for the convection and
pressure gradient terms were derived to solve the incompressible Navier-Stokes
(N-S) equations. The superior performance of can-PINNs is demonstrated on
several challenging problems, including the flow mixing phenomena, lid driven
flow in a cavity, and channel flow over a backward facing step. The results
reveal that for challenging problems like these, can-PINNs can consistently
achieve very good accuracy whereas conventional AD-based PINNs fail.

    

### [[2110.15843] Adaptive Discretization in Online Reinforcement Learning](http://arxiv.org/abs/2110.15843)


  Discretization based approaches to solving online reinforcement learning
problems have been studied extensively in practice on applications ranging from
resource allocation to cache management. Two major questions in designing
discretization-based algorithms are how to create the discretization and when
to refine it. While there have been several experimental results investigating
heuristic solutions to these questions, there has been little theoretical
treatment. In this paper we provide a unified theoretical analysis of
tree-based hierarchical partitioning methods for online reinforcement learning,
providing model-free and model-based algorithms. We show how our algorithms are
able to take advantage of inherent structure of the problem by providing
guarantees that scale with respect to the 'zooming dimension' instead of the
ambient dimension, an instance-dependent quantity measuring the benignness of
the optimal $Q_h^\star$ function.
Many applications in computing systems and operations research requires
algorithms that compete on three facets: low sample complexity, mild storage
requirements, and low computational burden. Our algorithms are easily adapted
to operating constraints, and our theory provides explicit bounds across each
of the three facets. This motivates its use in practical applications as our
approach automatically adapts to underlying problem structure even when very
little is known a priori about the system.

    

### [[2110.15866] Towards Comparative Physical Interpretation of Spatial Variability Aware Neural Networks: A Summary of Results](http://arxiv.org/abs/2110.15866)


  Given Spatial Variability Aware Neural Networks (SVANNs), the goal is to
investigate mathematical (or computational) models for comparative physical
interpretation towards their transparency (e.g., simulatibility,
decomposability and algorithmic transparency). This problem is important due to
important use-cases such as reusability, debugging, and explainability to a
jury in a court of law. Challenges include a large number of model parameters,
vacuous bounds on generalization performance of neural networks, risk of
overfitting, sensitivity to noise, etc., which all detract from the ability to
interpret the models. Related work on either model-specific or model-agnostic
post-hoc interpretation is limited due to a lack of consideration of physical
constraints (e.g., mass balance) and properties (e.g., second law of
geography). This work investigates physical interpretation of SVANNs using
novel comparative approaches based on geographically heterogeneous features.
The proposed approach on feature-based physical interpretation is evaluated
using a case-study on wetland mapping. The proposed physical interpretation
improves the transparency of SVANN models and the analytical results highlight
the trade-off between model transparency and model performance (e.g.,
F1-score). We also describe an interpretation based on geographically
heterogeneous processes modeled as partial differential equations (PDEs).

    

### [[2110.15884] Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation](http://arxiv.org/abs/2110.15884)


  Most research on novel techniques for 3D Medical Image Segmentation (MIS) is
currently done using Deep Learning with GPU accelerators. The principal
challenge of such technique is that a single input can easily cope computing
resources, and require prohibitive amounts of time to be processed.
Distribution of deep learning and scalability over computing devices is an
actual need for progressing on such research field. Conventional distribution
of neural networks consist in data parallelism, where data is scattered over
resources (e.g., GPUs) to parallelize the training of the model. However,
experiment parallelism is also an option, where different training processes
are parallelized across resources. While the first option is much more common
on 3D image segmentation, the second provides a pipeline design with less
dependence among parallelized processes, allowing overhead reduction and more
potential scalability. In this work we present a design for distributed deep
learning training pipelines, focusing on multi-node and multi-GPU environments,
where the two different distribution approaches are deployed and benchmarked.
We take as proof of concept the 3D U-Net architecture, using the MSD Brain
Tumor Segmentation dataset, a state-of-art problem in medical image
segmentation with high computing and space requirements. Using the BSC
MareNostrum supercomputer as benchmarking environment, we use TensorFlow and
Ray as neural network training and experiment distribution platforms. We
evaluate the experiment speed-up, showing the potential for scaling out on GPUs
and nodes. Also comparing the different parallelism techniques, showing how
experiment distribution leverages better such resources through scaling.
Finally, we provide the implementation of the design open to the community, and
the non-trivial steps and methodology for adapting and deploying a MIS case as
the here presented.

    

### [[2110.15895] Application of 2-D Convolutional Neural Networks for Damage Detection in Steel Frame Structures](http://arxiv.org/abs/2110.15895)


  In this paper, we present an application of 2-D convolutional neural networks
(2-D CNNs) designed to perform both feature extraction and classification
stages as a single organism to solve the highlighted problems. The method uses
a network of lighted CNNs instead of deep and takes raw acceleration signals as
input. Using lighted CNNs, in which every one of them is optimized for a
specific element, increases the accuracy and makes the network faster to
perform. Also, a new framework is proposed for decreasing the data required in
the training phase. We verified our method on Qatar University Grandstand
Simulator (QUGS) benchmark data provided by Structural Dynamics Team. The
results showed improved accuracy over other methods, and running time was
adequate for real-time applications.

    

### [[2110.15900] Hyperparameter Tuning is All You Need for LISTA](http://arxiv.org/abs/2110.15900)


  Learned Iterative Shrinkage-Thresholding Algorithm (LISTA) introduces the
concept of unrolling an iterative algorithm and training it like a neural
network. It has had great success on sparse recovery. In this paper, we show
that adding momentum to intermediate variables in the LISTA network achieves a
better convergence rate and, in particular, the network with instance-optimal
parameters is superlinearly convergent. Moreover, our new theoretical results
lead to a practical approach of automatically and adaptively calculating the
parameters of a LISTA network layer based on its previous layers. Perhaps most
surprisingly, such an adaptive-parameter procedure reduces the training of
LISTA to tuning only three hyperparameters from data: a new record set in the
context of the recent advances on trimming down LISTA complexity. We call this
new ultra-light weight network HyperLISTA. Compared to state-of-the-art LISTA
models, HyperLISTA achieves almost the same performance on seen data
distributions and performs better when tested on unseen distributions
(specifically, those with different sparsity levels and nonzero magnitudes).
Code is available: this https URL.

    

### [[2110.15907] Learning to Be Cautious](http://arxiv.org/abs/2110.15907)


  A key challenge in the field of reinforcement learning is to develop agents
that behave cautiously in novel situations. It is generally impossible to
anticipate all situations that an autonomous system may face or what behavior
would best avoid bad outcomes. An agent that could learn to be cautious would
overcome this challenge by discovering for itself when and how to behave
cautiously. In contrast, current approaches typically embed task-specific
safety information or explicit cautious behaviors into the system, which is
error-prone and imposes extra burdens on practitioners. In this paper, we
present both a sequence of tasks where cautious behavior becomes increasingly
non-obvious, as well as an algorithm to demonstrate that it is possible for a
system to \emph{learn} to be cautious. The essential features of our algorithm
are that it characterizes reward function uncertainty without task-specific
safety information and uses this uncertainty to construct a robust policy.
Specifically, we construct robust policies with a $k$-of-$N$ counterfactual
regret minimization (CFR) subroutine given a learned reward function
uncertainty represented by a neural network ensemble belief. These policies
exhibit caution in each of our tasks without any task-specific safety tuning.

    

### [[2110.15909] Contrastive prediction strategies for unsupervised segmentation and categorization of phonemes and words](http://arxiv.org/abs/2110.15909)


  We investigate the performance on phoneme categorization and phoneme and word
segmentation of several self-supervised learning (SSL) methods based on
Contrastive Predictive Coding (CPC). Our experiments show that with the
existing algorithms there is a trade off between categorization and
segmentation performance. We investigate the source of this conflict and
conclude that the use of context building networks, albeit necessary for
superior performance on categorization tasks, harms segmentation performance by
causing a temporal shift on the learned representations. Aiming to bridge this
gap, we take inspiration from the leading approach on segmentation, which
simultaneously models the speech signal at the frame and phoneme level, and
incorporate multi-level modelling into Aligned CPC (ACPC), a variation of CPC
which exhibits the best performance on categorization tasks. Our multi-level
ACPC (mACPC) improves in all categorization metrics and achieves
state-of-the-art performance in word segmentation.

    

### [[2110.15911] Physics-informed linear regression is a competitive approach compared to Machine Learning methods in building MPC](http://arxiv.org/abs/2110.15911)


  Because physics-based building models are difficult to obtain as each
building is individual, there is an increasing interest in generating models
suitable for building MPC directly from measurement data. Machine learning
methods have been widely applied to this problem and validated mostly in
simulation; there are, however, few studies on a direct comparison of different
models or validation in real buildings to be found in the literature. Methods
that are indeed validated in application often lead to computationally complex
non-convex optimization problems. Here we compare physics-informed
Autoregressive-Moving-Average with Exogenous Inputs (ARMAX) models to Machine
Learning models based on Random Forests and Input Convex Neural Networks and
the resulting convex MPC schemes in experiments on a practical building
application with the goal of minimizing energy consumption while maintaining
occupant comfort, and in a numerical case study. We demonstrate that Predictive
Control in general leads to savings between 26% and 49% of heating and cooling
energy, compared to the building's baseline hysteresis controller. Moreover, we
show that all model types lead to satisfactory control performance in terms of
constraint satisfaction and energy reduction. However, we also see that the
physics-informed ARMAX models have a lower computational burden, and a superior
sample efficiency compared to the Machine Learning based models. Moreover, even
if abundant training data is available, the ARMAX models have a significantly
lower prediction error than the Machine Learning models, which indicates that
the encoded physics-based prior of the former cannot independently be found by
the latter.

    

### [[2110.15912] On the use of uncertainty in classifying Aedes Albopictus mosquitoes](http://arxiv.org/abs/2110.15912)


  The re-emergence of mosquito-borne diseases (MBDs), which kill hundreds of
thousands of people each year, has been attributed to increased human
population, migration, and environmental changes. Convolutional neural networks
(CNNs) have been used by several studies to recognise mosquitoes in images
provided by projects such as Mosquito Alert to assist entomologists in
identifying, monitoring, and managing MBD. Nonetheless, utilising CNNs to
automatically label input samples could involve incorrect predictions, which
may mislead future epidemiological studies. Furthermore, CNNs require large
numbers of manually annotated data. In order to address the mentioned issues,
this paper proposes using the Monte Carlo Dropout method to estimate the
uncertainty scores in order to rank the classified samples to reduce the need
for human supervision in recognising Aedes albopictus mosquitoes. The estimated
uncertainty was also used in an active learning framework, where just a portion
of the data from large training sets was manually labelled. The experimental
results show that the proposed classification method with rejection outperforms
the competing methods by improving overall performance and reducing
entomologist annotation workload. We also provide explainable visualisations of
the different regions that contribute to a set of samples' uncertainty
assessment.

    

### [[2110.15914] Improving the quality of generative models through Smirnov transformation](http://arxiv.org/abs/2110.15914)


  Solving the convergence issues of Generative Adversarial Networks (GANs) is
one of the most outstanding problems in generative models. In this work, we
propose a novel activation function to be used as output of the generator
agent. This activation function is based on the Smirnov probabilistic
transformation and it is specifically designed to improve the quality of the
generated data. In sharp contrast with previous works, our activation function
provides a more general approach that deals not only with the replication of
categorical variables but with any type of data distribution (continuous or
discrete). Moreover, our activation function is derivable and therefore, it can
be seamlessly integrated in the backpropagation computations during the GAN
training processes. To validate this approach, we evaluate our proposal against
two different data sets: a) an artificially rendered data set containing a
mixture of discrete and continuous variables, and b) a real data set of
flow-based network traffic data containing both normal connections and
cryptomining attacks. To evaluate the fidelity of the generated data, we
analyze both their results in terms of quality measures of statistical nature
and also regarding the use of these synthetic data to feed a nested machine
learning-based classifier. The experimental results evince a clear
outperformance of the GAN network tuned with this new activation function with
respect to both a naïve mean-based generator and a standard GAN. The quality
of the data is so high that the generated data can fully substitute real data
for training the nested classifier without a fall in the obtained accuracy.
This result encourages the use of GANs to produce high-quality synthetic data
that are applicable in scenarios in which data privacy must be guaranteed.

    

### [[2110.15941] Personalized breath based biometric authentication with wearable multimodality](http://arxiv.org/abs/2110.15941)


  Breath with nose sound features has been shown as a potential biometric in
personal identification and verification. In this paper, we show that
information that comes from other modalities captured by motion sensors on the
chest in addition to audio features could further improve the performance. Our
work is composed of three main contributions: hardware creation, dataset
publication, and proposed multimodal models. To be more specific, we design new
hardware which consists of an acoustic sensor to collect audio features from
the nose, as well as an accelerometer and gyroscope to collect movement on the
chest as a result of an individual's breathing. Using this hardware, we publish
a collected dataset from a number of sessions from different volunteers, each
session includes three common gestures: normal, deep, and strong breathing.
Finally, we experiment with two multimodal models based on Convolutional Long
Short Term Memory (CNN-LSTM) and Temporal Convolutional Networks (TCN)
architectures. The results demonstrate the suitability of our new hardware for
both verification and identification tasks.

    

### [[2110.15949] Sparsely Changing Latent States for Prediction and Planning in Partially Observable Domains](http://arxiv.org/abs/2110.15949)


  A common approach to prediction and planning in partially observable domains
is to use recurrent neural networks (RNNs), which ideally develop and maintain
a latent memory about hidden, task-relevant factors. We hypothesize that many
of these hidden factors in the physical world are constant over time, changing
only sparsely. Accordingly, we propose Gated $L_0$ Regularized Dynamics
(GateL0RD), a novel recurrent architecture that incorporates the inductive bias
to maintain stable, sparsely changing latent states. The bias is implemented by
means of a novel internal gating function and a penalty on the $L_0$ norm of
latent state changes. We demonstrate that GateL0RD can compete with or
outperform state-of-the-art RNNs in a variety of partially observable
prediction and control tasks. GateL0RD tends to encode the underlying
generative factors of the environment, ignores spurious temporal dependencies,
and generalizes better, improving sampling efficiency and prediction accuracy
as well as behavior in model-based planning and reinforcement learning tasks.
Moreover, we show that the developing latent states can be easily interpreted,
which is a step towards better explainability in RNNs.

    

### [[2110.15954] Limiting fluctuation and trajectorial stability of multilayer neural networks with mean field training](http://arxiv.org/abs/2110.15954)


  The mean field (MF) theory of multilayer neural networks centers around a
particular infinite-width scaling, where the learning dynamics is closely
tracked by the MF limit. A random fluctuation around this infinite-width limit
is expected from a large-width expansion to the next order. This fluctuation
has been studied only in shallow networks, where previous works employ heavily
technical notions or additional formulation ideas amenable only to that case.
Treatment of the multilayer case has been missing, with the chief difficulty in
finding a formulation that captures the stochastic dependency across not only
time but also depth.
In this work, we initiate the study of the fluctuation in the case of
multilayer networks, at any network depth. Leveraging on the neuronal embedding
framework recently introduced by Nguyen and Pham, we systematically derive a
system of dynamical equations, called the second-order MF limit, that captures
the limiting fluctuation distribution. We demonstrate through the framework the
complex interaction among neurons in this second-order MF limit, the
stochasticity with cross-layer dependency and the nonlinear time evolution
inherent in the limiting fluctuation. A limit theorem is proven to relate
quantitatively this limit to the fluctuation of large-width networks.
We apply the result to show a stability property of gradient descent MF
training: in the large-width regime, along the training trajectory, it
progressively biases towards a solution with "minimal fluctuation" (in fact,
vanishing fluctuation) in the learned output function, even after the network
has been initialized at or has converged (sufficiently fast) to a global
optimum. This extends a similar phenomenon previously shown only for shallow
networks with a squared loss in the ERM setting, to multilayer networks with a
loss function that is not necessarily convex in a more general setting.

    

### [[2110.15956] A deep convolutional neural network for classification of Aedes albopictus mosquitoes](http://arxiv.org/abs/2110.15956)


  Monitoring the spread of disease-carrying mosquitoes is a first and necessary
step to control severe diseases such as dengue, chikungunya, Zika or yellow
fever. Previous citizen science projects have been able to obtain large image
datasets with linked geo-tracking information. As the number of international
collaborators grows, the manual annotation by expert entomologists of the large
amount of data gathered by these users becomes too time demanding and
unscalable, posing a strong need for automated classification of mosquito
species from images. We introduce the application of two Deep Convolutional
Neural Networks in a comparative study to automate this classification task. We
use the transfer learning principle to train two state-of-the-art architectures
on the data provided by the Mosquito Alert project, obtaining testing accuracy
of 94%. In addition, we applied explainable models based on the Grad-CAM
algorithm to visualise the most discriminant regions of the classified images,
which coincide with the white band stripes located at the legs, abdomen, and
thorax of mosquitoes of the Aedes albopictus species. The model allows us to
further analyse the classification errors. Visual Grad-CAM models show that
they are linked to poor acquisition conditions and strong image occlusions.

    

### [[2110.15960] Support Recovery with Stochastic Gates: Theory and Application for Linear Models](http://arxiv.org/abs/2110.15960)


  We analyze the problem of simultaneous support recovery and estimation of the
coefficient vector ($\beta^*$) in a linear model with independent and
identically distributed Normal errors. We apply the penalised least square
estimator of $\beta^*$ based on non-linear penalties of stochastic gates (STG)
[YLNK20] to estimate the coefficients. Considering Gaussian design matrices we
show that under reasonable conditions on dimension and sparsity of $\beta^*$
the STG based estimator converges to the true data generating coefficient
vector and also detects its support set with high probability. We propose a new
projection based algorithm for the linear models setup to improve upon the
existing STG estimator that was originally designed for general non-linear
models. Our new procedure outperforms many classical estimators for sparse
support recovery in synthetic data analysis.

    

### [[1903.11991] Parabolic Approximation Line Search for DNNs](http://arxiv.org/abs/1903.11991)


  A major challenge in current optimization research for deep learning is to
automatically find optimal step sizes for each update step. The optimal step
size is closely related to the shape of the loss in the update step direction.
However, this shape has not yet been examined in detail. This work shows
empirically that the batch loss over lines in negative gradient direction is
mostly convex locally and well suited for one-dimensional parabolic
approximations. By exploiting this parabolic property we introduce a simple and
robust line search approach, which performs loss-shape dependent update steps.
Our approach combines well-known methods such as parabolic approximation, line
search and conjugate gradient, to perform efficiently. It surpasses other step
size estimating methods and competes with common optimization methods on a
large variety of experiments without the need of hand-designed step size
schedules. Thus, it is of interest for objectives where step-size schedules are
unknown or do not perform well. Our extensive evaluation includes multiple
comprehensive hyperparameter grid searches on several datasets and
architectures. Finally, we provide a general investigation of exact line
searches in the context of batch losses and exact losses, including their
relation to our line search approach.

    

### [[2001.00215] Histogram Layers for Texture Analysis](http://arxiv.org/abs/2001.00215)


  An essential aspect of texture analysis is the extraction of features that
describe the distribution of values in local, spatial regions. We present a
localized histogram layer for artificial neural networks. Instead of computing
global histograms as done previously, the proposed histogram layer directly
computes the local, spatial distribution of features for texture analysis and
parameters for the layer are estimated during backpropagation. We compare our
method with state-of-the-art texture encoding methods such as the Deep Encoding
Network Pooling, Deep Texture Encoding Network, Fisher Vector convolutional
neural network, and Multi-level Texture Encoding and Representation on three
material/texture datasets: (1) the Describable Texture Dataset; (2) an
extension of the ground terrain in outdoor scenes; (3) and a subset of the
Materials in Context dataset. Results indicate that the inclusion of the
proposed histogram layer improves performance. The source code for the
histogram layer is publicly available:
this https URL.

    

### [[2007.15645] Approximation of Smoothness Classes by Deep Rectifier Networks](http://arxiv.org/abs/2007.15645)


  We consider approximation rates of sparsely connected deep rectified linear
unit (ReLU) and rectified power unit (RePU) neural networks for functions in
Besov spaces $B^\alpha_{q}(L^p)$ in arbitrary dimension $d$, on general
domains. We show that \alert{deep rectifier} networks with a fixed activation
function attain optimal or near to optimal approximation rates for functions in
the Besov space $B^\alpha_{\tau}(L^\tau)$ on the critical embedding line
$1/\tau=\alpha/d+1/p$ for \emph{arbitrary} smoothness order $\alpha>0$. Using
interpolation theory, this implies that the entire range of smoothness classes
at or above the critical line is (near to) optimally approximated by deep
ReLU/RePU networks.

    

### [[2009.01521] Smoke Testing for Machine Learning: Simple Tests to Discover Severe Defects](http://arxiv.org/abs/2009.01521)


  Machine learning is nowadays a standard technique for data analysis within
software applications. Software engineers need quality assurance techniques
that are suitable for these new kinds of systems. Within this article, we
discuss the question whether standard software testing techniques that have
been part of textbooks since decades are also useful for the testing of machine
learning software. Concretely, we try to determine generic and simple smoke
tests that can be used to assert that basic functions can be executed without
crashing. We found that we can derive such tests using techniques similar to
equivalence classes and boundary value analysis. Moreover, we found that these
concepts can also be applied to hyperparameters, to further improve the quality
of the smoke tests. Even though our approach is almost trivial, we were able to
find bugs in all three machine learning libraries that we tested and severe
bugs in two of the three libraries. This demonstrates that common software
testing techniques are still valid in the age of machine learning and that
considerations how they can be adapted to this new context can help to find and
prevent severe bugs, even in mature machine learning libraries.

    

### [[2009.13251] Deep Learning for Predictive Business Process Monitoring: Review and Benchmark](http://arxiv.org/abs/2009.13251)


  Predictive monitoring of business processes is concerned with the prediction
of ongoing cases on a business process. Lately, the popularity of deep learning
techniques has propitiated an ever-growing set of approaches focused on
predictive monitoring based on these techniques. However, the high disparity of
process logs and experimental setups used to evaluate these approaches makes it
especially difficult to make a fair comparison. Furthermore, it also difficults
the selection of the most suitable approach to solve a specific problem. In
this paper, we provide both a systematic literature review of approaches that
use deep learning to tackle the predictive monitoring tasks. In addition, we
performed an exhaustive experimental evaluation of 10 different approaches over
12 publicly available process logs.

    

### [[2010.11660] Detecting Rewards Deterioration in Episodic Reinforcement Learning](http://arxiv.org/abs/2010.11660)


  In many RL applications, once training ends, it is vital to detect any
deterioration in the agent performance as soon as possible. Furthermore, it
often has to be done without modifying the policy and under minimal assumptions
regarding the environment. In this paper, we address this problem by focusing
directly on the rewards and testing for degradation. We consider an episodic
framework, where the rewards within each episode are not independent, nor
identically-distributed, nor Markov. We present this problem as a multivariate
mean-shift detection problem with possibly partial observations. We define the
mean-shift in a way corresponding to deterioration of a temporal signal (such
as the rewards), and derive a test for this problem with optimal statistical
power. Empirically, on deteriorated rewards in control problems (generated
using various environment modifications), the test is demonstrated to be more
powerful than standard tests - often by orders of magnitude. We also suggest a
novel Bootstrap mechanism for False Alarm Rate control (BFAR), applicable to
episodic (non-i.i.d) signal and allowing our test to run sequentially in an
online manner. Our method does not rely on a learned model of the environment,
is entirely external to the agent, and in fact can be applied to detect changes
or drifts in any episodic signal.

    

### [[2010.12305] FAME: Feature-Based Adversarial Meta-Embeddings for Robust Input Representations](http://arxiv.org/abs/2010.12305)


  Combining several embeddings typically improves performance in downstream
tasks as different embeddings encode different information. It has been shown
that even models using embeddings from transformers still benefit from the
inclusion of standard word embeddings. However, the combination of embeddings
of different types and dimensions is challenging. As an alternative to
attention-based meta-embeddings, we propose feature-based adversarial
meta-embeddings (FAME) with an attention function that is guided by features
reflecting word-specific properties, such as shape and frequency, and show that
this is beneficial to handle subword-based embeddings. In addition, FAME uses
adversarial training to optimize the mappings of differently-sized embeddings
to the same space. We demonstrate that FAME works effectively across languages
and domains for sequence labeling and sentence classification, in particular in
low-resource settings. FAME sets the new state of the art for POS tagging in 27
languages, various NER settings and question classification in different
domains.

    

### [[2010.13997] A Domain-Shrinking based Bayesian Optimization Algorithm with Order-Optimal Regret Performance](http://arxiv.org/abs/2010.13997)


  We consider sequential optimization of an unknown function in a reproducing
kernel Hilbert space. We propose a Gaussian process-based algorithm and
establish its order-optimal regret performance (up to a poly-logarithmic
factor). This is the first GP-based algorithm with an order-optimal regret
guarantee. The proposed algorithm is rooted in the methodology of domain
shrinking realized through a sequence of tree-based region pruning and refining
to concentrate queries in increasingly smaller high-performing regions of the
function domain. The search for high-performing regions is localized and guided
by an iterative estimation of the optimal function value to ensure both
learning efficiency and computational efficiency. Compared with the prevailing
GP-UCB family of algorithms, the proposed algorithm reduces computational
complexity by a factor of $O(T^{2d-1})$ (where $T$ is the time horizon and $d$
the dimension of the function domain).

    

### [[2011.06733] One Explanation is Not Enough: Structured Attention Graphs for Image Classification](http://arxiv.org/abs/2011.06733)


  Attention maps are a popular way of explaining the decisions of convolutional
networks for image classification. Typically, for each image of interest, a
single attention map is produced, which assigns weights to pixels based on
their importance to the classification. A single attention map, however,
provides an incomplete understanding since there are often many other maps that
explain a classification equally well. In this paper, we introduce structured
attention graphs (SAGs), which compactly represent sets of attention maps for
an image by capturing how different combinations of image regions impact a
classifier's confidence. We propose an approach to compute SAGs and a
visualization for SAGs so that deeper insight can be gained into a classifier's
decisions. We conduct a user study comparing the use of SAGs to traditional
attention maps for answering counterfactual questions about image
classifications. Our results show that the users are more correct when
answering comparative counterfactual questions based on SAGs compared to the
baselines.

    

### [[2011.07682] A Large-Scale Database for Graph Representation Learning](http://arxiv.org/abs/2011.07682)


  With the rapid emergence of graph representation learning, the construction
of new large-scale datasets is necessary to distinguish model capabilities and
accurately assess the strengths and weaknesses of each technique. By carefully
analyzing existing graph databases, we identify 3 critical components important
for advancing the field of graph representation learning: (1) large graphs, (2)
many graphs, and (3) class diversity. To date, no single graph database offers
all these desired properties. We introduce MalNet, the largest public graph
database ever constructed, representing a large-scale ontology of malicious
software function call graphs. MalNet contains over 1.2 million graphs,
averaging over 15k nodes and 35k edges per graph, across a hierarchy of 47
types and 696 families. Compared to the popular REDDIT-12K database, MalNet
offers 105x more graphs, 39x larger graphs on average, and 63x more classes. We
provide a detailed analysis of MalNet, discussing its properties and
provenance, along with the evaluation of state-of-the-art machine learning and
graph neural network techniques. The unprecedented scale and diversity of
MalNet offers exciting opportunities to advance the frontiers of graph
representation learning--enabling new discoveries and research into imbalanced
classification, explainability and the impact of class hardness. The database
is publicly available at this http URL.

    

### [[2011.12413] Wide-band butterfly network: stable and efficient inversion via multi-frequency neural networks](http://arxiv.org/abs/2011.12413)


  We introduce an end-to-end deep learning architecture called the wide-band
butterfly network (WideBNet) for approximating the inverse scattering map from
wide-band scattering data. This architecture incorporates tools from
computational harmonic analysis, such as the butterfly factorization, and
traditional multi-scale methods, such as the Cooley-Tukey FFT algorithm, to
drastically reduce the number of trainable parameters to match the inherent
complexity of the problem. As a result WideBNet is efficient: it requires fewer
training points than off-the-shelf architectures, and has stable training
dynamics, thus it can rely on standard weight initialization strategies. The
architecture automatically adapts to the dimensions of the data with only a few
hyper-parameters that the user must specify. WideBNet is able to produce images
that are competitive with optimization-based approaches, but at a fraction of
the cost, and we also demonstrate numerically that it learns to super-resolve
scatterers in the full aperture scattering setup.

    

### [[2011.12719] RLlib Flow: Distributed Reinforcement Learning is a Dataflow Problem](http://arxiv.org/abs/2011.12719)


  Researchers and practitioners in the field of reinforcement learning (RL)
frequently leverage parallel computation, which has led to a plethora of new
algorithms and systems in the last few years. In this paper, we re-examine the
challenges posed by distributed RL and try to view it through the lens of an
old idea: distributed dataflow. We show that viewing RL as a dataflow problem
leads to highly composable and performant implementations. We propose RLlib
Flow, a hybrid actor-dataflow programming model for distributed RL, and
validate its practicality by porting the full suite of algorithms in RLlib, a
widely adopted distributed RL library. Concretely, RLlib Flow provides 2-9 code
savings in real production code and enables the composition of multi-agent
algorithms not possible by end users before. The open-source code is available
as part of RLlib at this https URL.

    

### [[2012.03612] LCS Graph Kernel Based on Wasserstein Distance in Longest Common Subsequence Metric Space](http://arxiv.org/abs/2012.03612)


  For graph learning tasks, many existing methods utilize a message-passing
mechanism where vertex features are updated iteratively by aggregation of
neighbor information. This strategy provides an efficient means for graph
features extraction, but obtained features after many iterations might contain
too much information from other vertices, and tend to be similar to each other.
This makes their representations less expressive. Learning graphs using paths,
on the other hand, can be less adversely affected by this problem because it
does not involve all vertex neighbors. However, most of them can only compare
paths with the same length, which might engender information loss. To resolve
this difficulty, we propose a new Graph Kernel based on a Longest Common
Subsequence (LCS) similarity. Moreover, we found that the widely-used
R-convolution framework is unsuitable for path-based Graph Kernel because a
huge number of comparisons between dissimilar paths might deteriorate graph
distances calculation. Therefore, we propose a novel metric space by exploiting
the proposed LCS-based similarity, and compute a new Wasserstein-based graph
distance in this metric space, which emphasizes more the comparison between
similar paths. Furthermore, to reduce the computational cost, we propose an
adjacent point merging operation to sparsify point clouds in the metric space.

    

### [[2012.11552] OBoW: Online Bag-of-Visual-Words Generation for Self-Supervised Learning](http://arxiv.org/abs/2012.11552)


  Learning image representations without human supervision is an important and
active research field. Several recent approaches have successfully leveraged
the idea of making such a representation invariant under different types of
perturbations, especially via contrastive-based instance discrimination
training. Although effective visual representations should indeed exhibit such
invariances, there are other important characteristics, such as encoding
contextual reasoning skills, for which alternative reconstruction-based
approaches might be better suited.
With this in mind, we propose a teacher-student scheme to learn
representations by training a convolutional net to reconstruct a
bag-of-visual-words (BoW) representation of an image, given as input a
perturbed version of that same image. Our strategy performs an online training
of both the teacher network (whose role is to generate the BoW targets) and the
student network (whose role is to learn representations), along with an online
update of the visual-words vocabulary (used for the BoW targets). This idea
effectively enables fully online BoW-guided unsupervised learning. Extensive
experiments demonstrate the interest of our BoW-based strategy which surpasses
previous state-of-the-art methods (including contrastive-based ones) in several
applications. For instance, in downstream tasks such Pascal object detection,
Pascal classification and Places205 classification, our method improves over
all prior unsupervised approaches, thus establishing new state-of-the-art
results that are also significantly better even than those of supervised
pre-training. We provide the implementation code at
this https URL.

    

### [[2012.14844] Inference for Low-rank Tensors -- No Need to Debias](http://arxiv.org/abs/2012.14844)


  In this paper, we consider the statistical inference for several low-rank
tensor models. Specifically, in the Tucker low-rank tensor PCA or regression
model, provided with any estimates achieving some attainable error rate, we
develop the data-driven confidence regions for the singular subspace of the
parameter tensor based on the asymptotic distribution of an updated estimate by
two-iteration alternating minimization. The asymptotic distributions are
established under some essential conditions on the signal-to-noise ratio (in
PCA model) or sample size (in regression model). If the parameter tensor is
further orthogonally decomposable, we develop the methods and non-asymptotic
theory for inference on each individual singular vector. For the rank-one
tensor PCA model, we establish the asymptotic distribution for general linear
forms of principal components and confidence interval for each entry of the
parameter tensor. Finally, numerical simulations are presented to corroborate
our theoretical discoveries.
In all these models, we observe that different from many matrix/vector
settings in existing work, debiasing is not required to establish the
asymptotic distribution of estimates or to make statistical inference on
low-rank tensors. In fact, due to the widely observed
statistical-computational-gap for low-rank tensor estimation, one usually
requires stronger conditions than the statistical (or information-theoretic)
limit to ensure the computationally feasible estimation is achievable.
Surprisingly, such conditions ``incidentally" render a feasible low-rank tensor
inference without debiasing.

    

### [[2012.14905] Meta Learning Backpropagation And Improving It](http://arxiv.org/abs/2012.14905)


  Many concepts have been proposed for meta learning with neural networks
(NNs), e.g., NNs that learn to reprogram fast weights, Hebbian plasticity,
learned learning rules, and meta recurrent NNs. Our Variable Shared Meta
Learning (VSML) unifies the above and demonstrates that simple weight-sharing
and sparsity in an NN is sufficient to express powerful learning algorithms
(LAs) in a reusable fashion. A simple implementation of VSML where the weights
of a neural network are replaced by tiny LSTMs allows for implementing the
backpropagation LA solely by running in forward-mode. It can even meta learn
new LAs that differ from online backpropagation and generalize to datasets
outside of the meta training distribution without explicit gradient
calculation. Introspection reveals that our meta learned LAs learn through fast
association in a way that is qualitatively different from gradient descent.

    

### [[2102.02454] Hybrid Adversarial Imitation Learning](http://arxiv.org/abs/2102.02454)


  Extrapolating beyond-demonstrator (BD) performance through the imitation
learning (IL) algorithm aims to learn from and outperform the demonstrator.
Most existing BDIL algorithms are performed in two stages by first inferring a
reward function before learning a policy via reinforcement learning (RL).
However, such two-stage BDIL algorithms suffer from high computational
complexity, weak robustness, and large performance variations. In particular, a
poor reward function derived in the first stage will inevitably incur severe
performance loss in the second stage. In this work, we propose a hybrid
adversarial imitation learning (HAIL) algorithm that is one-stage, model-free,
generative-adversarial (GA) fashion and curiosity-driven. Thanks to the
one-stage design, the HAIL can integrate both the reward function learning and
the policy optimization into one procedure, which leads to many advantages such
as low computational complexity, high robustness, and strong adaptability. More
specifically, HAIL simultaneously imitates the demonstrator and explores BD
performance by utilizing hybrid rewards. Extensive simulation results confirm
that HAIL can achieve higher performance as compared to other similar BDIL
algorithms.

    

### [[2103.14431] Multimodal Knowledge Expansion](http://arxiv.org/abs/2103.14431)


  The popularity of multimodal sensors and the accessibility of the Internet
have brought us a massive amount of unlabeled multimodal data. Since existing
datasets and well-trained models are primarily unimodal, the modality gap
between a unimodal network and unlabeled multimodal data poses an interesting
problem: how to transfer a pre-trained unimodal network to perform the same
task on unlabeled multimodal data? In this work, we propose multimodal
knowledge expansion (MKE), a knowledge distillation-based framework to
effectively utilize multimodal data without requiring labels. Opposite to
traditional knowledge distillation, where the student is designed to be
lightweight and inferior to the teacher, we observe that a multimodal student
model consistently denoises pseudo labels and generalizes better than its
teacher. Extensive experiments on four tasks and different modalities verify
this finding. Furthermore, we connect the mechanism of MKE to semi-supervised
learning and offer both empirical and theoretical explanations to understand
the denoising capability of a multimodal student.

    

### [[2104.08078] To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning](http://arxiv.org/abs/2104.08078)


  In low-resource settings, model transfer can help to overcome a lack of
labeled data for many tasks and domains. However, predicting useful transfer
sources is a challenging problem, as even the most similar sources might lead
to unexpected negative transfer results. Thus, ranking methods based on task
and text similarity -- as suggested in prior work -- may not be sufficient to
identify promising sources. To tackle this problem, we propose a new approach
to automatically determine which and how many sources should be exploited. For
this, we study the effects of model transfer on sequence labeling across
various domains and tasks and show that our methods based on model similarity
and support vector machines are able to predict promising sources, resulting in
performance increases of up to 24 F1 points.

    

### [[2104.08308] Neural Transfer Learning for Repairing Security Vulnerabilities in C Code](http://arxiv.org/abs/2104.08308)


  In this paper, we address the problem of automatic repair of software
vulnerabilities with deep learning. The major problemwith data-driven
vulnerability repair is that the few existing datasets of known confirmed
vulnerabilities consist of only a few thousandexamples. However, training a
deep learning model often requires hundreds of thousands of examples. In this
work, we leverage theintuition that the bug fixing task and the vulnerability
fixing task are related, and that the knowledge learned from bug fixes can
betransferred to fixing vulnerabilities. In the machine learning community,
this technique is called transfer learning. In this paper, wepropose an
approach for repairing security vulnerabilities named VRepair which is based on
transfer learning. VRepair is first trainedon a large bug fix corpus and is
then tuned on a vulnerability fix dataset, which is an order of magnitude
smaller. In our experiments,we show that a model trained only on a bug fix
corpus can already fix some vulnerabilities. Then, we demonstrate that transfer
learningimproves the ability to repair vulnerable C functions. We also show
that the transfer learning model performs better than a modeltrained with a
denoising task and fine-tuned on the vulnerability fixing task. To sum up, this
paper shows that transfer learning workswell for repairing security
vulnerabilities in C compared to learning on a small dataset.

    

### [[2104.10219] Scalable Synthesis of Verified Controllers in Deep Reinforcement Learning](http://arxiv.org/abs/2104.10219)


  There has been significant recent interest in devising verification
techniques for learning-enabled controllers (LECs) that manage safety-critical
systems. Given the opacity and lack of interpretability of the neural policies
that govern the behavior of such controllers, many existing approaches enforce
safety properties through the use of shields, a dynamic monitoring and repair
mechanism that ensures a LEC does not emit actions that would violate desired
safety conditions. These methods, however, have shown to have significant
scalability limitations because verification costs grow as problem
dimensionality and objective complexity increase. In this paper, we propose a
new automated verification pipeline capable of synthesizing high-quality safety
shields even when the problem domain involves hundreds of dimensions, or when
the desired objective involves stochastic perturbations, liveness
considerations, and other complex non-functional properties. Our key insight
involves separating safety verification from neural controller, using
pre-computed verified safety shields to constrain neural controller training
which does not only focus on safety. Experimental results over a range of
realistic high-dimensional deep RL benchmarks demonstrate the effectiveness of
our approach.

    

### [[2104.13276] MULTIMODAL ANALYSIS: Informed content estimation and audio source separation](http://arxiv.org/abs/2104.13276)


  This dissertation proposes the study of multimodal learning in the context of
musical signals. Throughout, we focus on the interaction between audio signals
and text information. Among the many text sources related to music that can be
used (e.g. reviews, metadata, or social network feedback), we concentrate on
lyrics. The singing voice directly connects the audio signal and the text
information in a unique way, combining melody and lyrics where a linguistic
dimension complements the abstraction of musical instruments. Our study focuses
on the audio and lyrics interaction for targeting source separation and
informed content estimation.

    

### [[2105.04522] Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels](http://arxiv.org/abs/2105.04522)


  Prior works have found it beneficial to combine provably noise-robust loss
functions e.g., mean absolute error (MAE) with standard categorical loss
function e.g. cross entropy (CE) to improve their learnability. Here, we
propose to use Jensen-Shannon divergence as a noise-robust loss function and
show that it interestingly interpolate between CE and MAE with a controllable
mixing parameter. Furthermore, we make a crucial observation that CE exhibit
lower consistency around noisy data points. Based on this observation, we adopt
a generalized version of the Jensen-Shannon divergence for multiple
distributions to encourage consistency around data points. Using this loss
function, we show state-of-the-art results on both synthetic (CIFAR), and
real-world (e.g., WebVision) noise with varying noise rates.

    

### [[2105.06369] Neighborhood-Aware Neural Architecture Search](http://arxiv.org/abs/2105.06369)


  Existing neural architecture search (NAS) methods often return an
architecture with good search performance but generalizes poorly to the test
setting. To achieve better generalization, we propose a novel
neighborhood-aware NAS formulation to identify flat-minima architectures in the
search space, with the assumption that flat minima generalize better than sharp
minima. The phrase ``flat-minima architecture'' refers to architectures whose
performance is stable under small perturbations in the architecture (e.g.,
replacing a convolution with a skip connection). Our formulation takes the
``flatness'' of an architecture into account by aggregating the performance
over the neighborhood of this architecture. We demonstrate a principled way to
apply our formulation to existing search algorithms, including sampling-based
algorithms and gradient-based algorithms. To facilitate the application to
gradient-based algorithms, we also propose a differentiable representation for
the neighborhood of architectures. Based on our formulation, we propose
neighborhood-aware random search (NA-RS) and neighborhood-aware differentiable
architecture search (NA-DARTS). Notably, by simply augmenting DARTS with our
formulation, NA-DARTS outperforms DARTS and achieves state-of-the-art
performance on established benchmarks, including CIFAR-10, CIFAR-100 and
ImageNet.

    

### [[2106.00099] Multi-Objective SPIBB: Seldonian Offline Policy Improvement with Safety Constraints in Finite MDPs](http://arxiv.org/abs/2106.00099)


  We study the problem of Safe Policy Improvement (SPI) under constraints in
the offline Reinforcement Learning (RL) setting. We consider the scenario
where: (i) we have a dataset collected under a known baseline policy, (ii)
multiple reward signals are received from the environment inducing as many
objectives to optimize. We present an SPI formulation for this RL setting that
takes into account the preferences of the algorithm's user for handling the
trade-offs for different reward signals while ensuring that the new policy
performs at least as well as the baseline policy along each individual
objective. We build on traditional SPI algorithms and propose a novel method
based on Safe Policy Iteration with Baseline Bootstrapping (SPIBB, Laroche et
al., 2019) that provides high probability guarantees on the performance of the
agent in the true environment. We show the effectiveness of our method on a
synthetic grid-world safety task as well as in a real-world critical care
context to learn a policy for the administration of IV fluids and vasopressors
to treat sepsis.

    

### [[2106.01202] Framing RNN as a kernel method: A neural ODE approach](http://arxiv.org/abs/2106.01202)


  Building on the interpretation of a recurrent neural network (RNN) as a
continuous-time neural differential equation, we show, under appropriate
conditions, that the solution of a RNN can be viewed as a linear function of a
specific feature set of the input sequence, known as the signature. This
connection allows us to frame a RNN as a kernel method in a suitable
reproducing kernel Hilbert space. As a consequence, we obtain theoretical
guarantees on generalization and stability for a large class of recurrent
networks. Our results are illustrated on simulated datasets.

    

### [[2106.01413] Rectangular Flows for Manifold Learning](http://arxiv.org/abs/2106.01413)


  Normalizing flows are inevitable neural networks with tractable
change-of-volume terms, which allow optimization of their parameters to be
efficiently performed via maximum likelihood. However, data of interest are
typically assumed to live in some (often unknown) low-dimensional manifold
embedded in a high-dimensional ambient space. The result is a modelling
mismatch since -- by construction -- the invertibility requirement implies
high-dimensional support of the learned distribution. Injective flows, mappings
from low- to high-dimensional spaces, aim to fix this discrepancy by learning
distributions on manifolds, but the resulting volume-change term becomes more
challenging to evaluate. Current approaches either avoid computing this term
entirely using various heuristics, or assume the manifold is known beforehand
and therefore are not widely applicable. Instead, we propose two methods to
tractably calculate the gradient of this term with respect to the parameters of
the model, relying on careful use of automatic differentiation and techniques
from numerical linear algebra. Both approaches perform end-to-end nonlinear
manifold learning and density estimation for data projected onto this manifold.
We study the trade-offs between our proposed methods, empirically verify that
we outperform approaches ignoring the volume-change term by more accurately
learning manifolds and the corresponding distributions on them, and show
promising results on out-of-distribution detection. Our code is available at
this https URL.

    

### [[2106.02395] DOCTOR: A Simple Method for Detecting Misclassification Errors](http://arxiv.org/abs/2106.02395)


  Deep neural networks (DNNs) have shown to perform very well on large scale
object recognition problems and lead to widespread use for real-world
applications, including situations where DNN are implemented as "black boxes".
A promising approach to secure their use is to accept decisions that are likely
to be correct while discarding the others. In this work, we propose DOCTOR, a
simple method that aims to identify whether the prediction of a DNN classifier
should (or should not) be trusted so that, consequently, it would be possible
to accept it or to reject it. Two scenarios are investigated: Totally Black Box
(TBB) where only the soft-predictions are available and Partially Black Box
(PBB) where gradient-propagation to perform input pre-processing is allowed.
Empirically, we show that DOCTOR outperforms all state-of-the-art methods on
various well-known images and sentiment analysis datasets. In particular, we
observe a reduction of up to $4\%$ of the false rejection rate (FRR) in the PBB
scenario. DOCTOR can be applied to any pre-trained model, it does not require
prior information about the underlying dataset and is as simple as the simplest
available methods in the literature.

    

### [[2106.04024] Manifold Topology Divergence: a Framework for Comparing Data Manifolds](http://arxiv.org/abs/2106.04024)


  We develop a framework for comparing data manifolds, aimed, in particular,
towards the evaluation of deep generative models. We describe a novel tool,
Cross-Barcode(P,Q), that, given a pair of distributions in a high-dimensional
space, tracks multiscale topology spacial discrepancies between manifolds on
which the distributions are concentrated. Based on the Cross-Barcode, we
introduce the Manifold Topology Divergence score (MTop-Divergence) and apply it
to assess the performance of deep generative models in various domains: images,
3D-shapes, time-series, and on different datasets: MNIST, Fashion MNIST, SVHN,
CIFAR10, FFHQ, chest X-ray images, market stock data, ShapeNet. We demonstrate
that the MTop-Divergence accurately detects various degrees of mode-dropping,
intra-mode collapse, mode invention, and image disturbance. Our algorithm
scales well (essentially linearly) with the increase of the dimension of the
ambient high-dimensional space. It is one of the first TDA-based practical
methodologies that can be applied universally to datasets of different sizes
and dimensions, including the ones on which the most recent GANs in the visual
domain are trained. The proposed method is domain agnostic and does not rely on
pre-trained networks.

    

### [[2106.04186] What training reveals about neural network complexity](http://arxiv.org/abs/2106.04186)


  This work explores the Benevolent Training Hypothesis (BTH) which argues that
the complexity of the function a deep neural network (NN) is learning can be
deduced by its training dynamics. Our analysis provides evidence for BTH by
relating the NN's Lipschitz constant at different regions of the input space
with the behavior of the stochastic training procedure. We first observe that
the Lipschitz constant close to the training data affects various aspects of
the parameter trajectory, with more complex networks having a longer
trajectory, bigger variance, and often veering further from their
initialization. We then show that NNs whose 1st layer bias is trained more
steadily (i.e., slowly and with little variation) have bounded complexity even
in regions of the input space that are far from any training point. Finally, we
find that steady training with Dropout implies a training- and data-dependent
generalization bound that grows poly-logarithmically with the number of
parameters. Overall, our results support the intuition that good training
behavior can be a useful bias towards good generalization.

    

### [[2106.04480] There Is No Turning Back: A Self-Supervised Approach for Reversibility-Aware Reinforcement Learning](http://arxiv.org/abs/2106.04480)


  We propose to learn to distinguish reversible from irreversible actions for
better informed decision-making in Reinforcement Learning (RL). From
theoretical considerations, we show that approximate reversibility can be
learned through a simple surrogate task: ranking randomly sampled trajectory
events in chronological order. Intuitively, pairs of events that are always
observed in the same order are likely to be separated by an irreversible
sequence of actions. Conveniently, learning the temporal order of events can be
done in a fully self-supervised way, which we use to estimate the reversibility
of actions from experience, without any priors. We propose two different
strategies that incorporate reversibility in RL agents, one strategy for
exploration (RAE) and one strategy for control (RAC). We demonstrate the
potential of reversibility-aware agents in several environments, including the
challenging Sokoban game. In synthetic tasks, we show that we can learn control
policies that never fail and reduce to zero the side-effects of interactions,
even without access to the reward function.

    

### [[2106.06839] Intelligent Vision Based Wear Forecasting on Surfaces of Machine Tool Elements](http://arxiv.org/abs/2106.06839)


  This paper addresses the ability to enable machines to automatically detect
failures on machine tool components as well as estimating the severity of the
failures, which is a critical step towards autonomous production machines.
Extracting information about the severity of failures has been a substantial
part of classical, as well as Machine Learning based machine vision systems.
Efforts have been undertaken to automatically predict the severity of failures
on machine tool components for predictive maintenance purposes. Though, most
approaches only partly cover a completely automatic system from detecting
failures to the prognosis of their future severity. To the best of the authors
knowledge, this is the first time a vision-based system for defect detection
and prognosis of failures on metallic surfaces in general and on Ball Screw
Drives in specific has been proposed. The authors show that they can do both,
detect and prognose the evolution of a failure on the surface of a Ball Screw
Drive.

    

### [[2106.07770] Potato Crop Stress Identification in Aerial Images using Deep Learning-based Object Detection](http://arxiv.org/abs/2106.07770)


  Recent research on the application of remote sensing and deep learning-based
analysis in precision agriculture demonstrated a potential for improved crop
management and reduced environmental impacts of agricultural production.
Despite the promising results, the practical relevance of these technologies
for field deployment requires novel algorithms that are customized for analysis
of agricultural images and robust to implementation on natural field imagery.
The paper presents an approach for analyzing aerial images of a potato (Solanum
tuberosum L.) crop using deep neural networks. The main objective is to
demonstrate automated spatial recognition of healthy vs. stressed crop at a
plant level. Specifically, we examine premature plant senescence resulting in
drought stress on Russet Burbank potato plants. We propose a novel deep
learning (DL) model for detecting crop stress, named Retina-UNet-Ag. The
proposed architecture is a variant of Retina-UNet and includes connections from
low-level semantic representation maps to the feature pyramid network. The
paper also introduces a dataset of aerial field images acquired with a Parrot
Sequoia camera. The dataset includes manually annotated bounding boxes of
healthy and stressed plant regions. Experimental validation demonstrated the
ability for distinguishing healthy and stressed plants in field images,
achieving an average dice score coefficient (DSC) of 0.74. A comparison to
related state-of-the-art DL models for object detection revealed that the
presented approach is effective for this task. The proposed method is conducive
toward the assessment and recognition of potato crop stress in aerial field
images collected under natural conditions.

    

### [[2106.07887] Credit Assignment in Neural Networks through Deep Feedback Control](http://arxiv.org/abs/2106.07887)


  The success of deep learning sparked interest in whether the brain learns by
using similar techniques for assigning credit to each synaptic weight for its
contribution to the network output. However, the majority of current attempts
at biologically-plausible learning methods are either non-local in time,
require highly specific connectivity motives, or have no clear link to any
known mathematical optimization method. Here, we introduce Deep Feedback
Control (DFC), a new learning method that uses a feedback controller to drive a
deep neural network to match a desired output target and whose control signal
can be used for credit assignment. The resulting learning rule is fully local
in space and time and approximates Gauss-Newton optimization for a wide range
of feedback connectivity patterns. To further underline its biological
plausibility, we relate DFC to a multi-compartment model of cortical pyramidal
neurons with a local voltage-dependent synaptic plasticity rule, consistent
with recent theories of dendritic processing. By combining dynamical system
theory with mathematical optimization theory, we provide a strong theoretical
foundation for DFC that we corroborate with detailed results on toy experiments
and standard computer-vision benchmarks.

    

### [[2106.08929] KALE Flow: A Relaxed KL Gradient Flow for Probabilities with Disjoint Support](http://arxiv.org/abs/2106.08929)


  We study the gradient flow for a relaxed approximation to the
Kullback-Leibler (KL) divergence between a moving source and a fixed target
distribution. This approximation, termed the KALE (KL approximate lower-bound
estimator), solves a regularized version of the Fenchel dual problem defining
the KL over a restricted class of functions. When using a Reproducing Kernel
Hilbert Space (RKHS) to define the function class, we show that the KALE
continuously interpolates between the KL and the Maximum Mean Discrepancy
(MMD). Like the MMD and other Integral Probability Metrics, the KALE remains
well defined for mutually singular distributions. Nonetheless, the KALE
inherits from the limiting KL a greater sensitivity to mismatch in the support
of the distributions, compared with the MMD. These two properties make the KALE
gradient flow particularly well suited when the target distribution is
supported on a low-dimensional manifold. Under an assumption of sufficient
smoothness of the trajectories, we show the global convergence of the KALE
flow. We propose a particle implementation of the flow given initial samples
from the source and the target distribution, which we use to empirically
confirm the KALE's properties.

    

### [[2106.09211] Square Root Principal Component Pursuit: Tuning-Free Noisy Robust Matrix Recovery](http://arxiv.org/abs/2106.09211)


  We propose a new framework -- Square Root Principal Component Pursuit -- for
low-rank matrix recovery from observations corrupted with noise and outliers.
Inspired by the square root Lasso, this new formulation does not require prior
knowledge of the noise level. We show that a single, universal choice of the
regularization parameter suffices to achieve reconstruction error proportional
to the (a priori unknown) noise level. In comparison, previous formulations
such as stable PCP rely on noise-dependent parameters to achieve similar
performance, and are therefore challenging to deploy in applications where the
noise level is unknown. We validate the effectiveness of our new method through
experiments on simulated and real datasets. Our simulations corroborate the
claim that a universal choice of the regularization parameter yields near
optimal performance across a range of noise levels, indicating that the
proposed method outperforms the (somewhat loose) bound proved here.

    

### [[2106.10052] On Contrastive Representations of Stochastic Processes](http://arxiv.org/abs/2106.10052)


  Learning representations of stochastic processes is an emerging problem in
machine learning with applications from meta-learning to physical object models
to time series. Typical methods rely on exact reconstruction of observations,
but this approach breaks down as observations become high-dimensional or noise
distributions become complex. To address this, we propose a unifying framework
for learning contrastive representations of stochastic processes (CReSP) that
does away with exact reconstruction. We dissect potential use cases for
stochastic process representations, and propose methods that accommodate each.
Empirically, we show that our methods are effective for learning
representations of periodic functions, 3D objects and dynamical processes. Our
methods tolerate noisy high-dimensional observations better than traditional
approaches, and the learned representations transfer to a range of downstream
tasks.

    

### [[2106.14448] R-Drop: Regularized Dropout for Neural Networks](http://arxiv.org/abs/2106.14448)


  Dropout is a powerful and widely used technique to regularize the training of
deep neural networks. In this paper, we introduce a simple regularization
strategy upon dropout in model training, namely R-Drop, which forces the output
distributions of different sub models generated by dropout to be consistent
with each other. Specifically, for each training sample, R-Drop minimizes the
bidirectional KL-divergence between the output distributions of two sub models
sampled by dropout. Theoretical analysis reveals that R-Drop reduces the
freedom of the model parameters and complements dropout. Experiments on
$\bf{5}$ widely used deep learning tasks ($\bf{18}$ datasets in total),
including neural machine translation, abstractive summarization, language
understanding, language modeling, and image classification, show that R-Drop is
universally effective. In particular, it yields substantial improvements when
applied to fine-tune large-scale pre-trained models, e.g., ViT, RoBERTa-large,
and BART, and achieves state-of-the-art (SOTA) performances with the vanilla
Transformer model on WMT14 English$\to$German translation ($\bf{30.91}$ BLEU)
and WMT14 English$\to$French translation ($\bf{43.95}$ BLEU), even surpassing
models trained with extra large-scale data and expert-designed advanced
variants of Transformer models. Our code is available at
GitHub{\url{this https URL}}.

    

### [[2106.15845] Edge Representation Learning with Hypergraphs](http://arxiv.org/abs/2106.15845)


  Graph neural networks have recently achieved remarkable success in
representing graph-structured data, with rapid progress in both the node
embedding and graph pooling methods. Yet, they mostly focus on capturing
information from the nodes considering their connectivity, and not much work
has been done in representing the edges, which are essential components of a
graph. However, for tasks such as graph reconstruction and generation, as well
as graph classification tasks for which the edges are important for
discrimination, accurately representing edges of a given graph is crucial to
the success of the graph representation learning. To this end, we propose a
novel edge representation learning framework based on Dual Hypergraph
Transformation (DHT), which transforms the edges of a graph into the nodes of a
hypergraph. This dual hypergraph construction allows us to apply
message-passing techniques for node representations to edges. After obtaining
edge representations from the hypergraphs, we then cluster or drop edges to
obtain holistic graph-level edge representations. We validate our edge
representation learning method with hypergraphs on diverse graph datasets for
graph representation and generation performance, on which our method largely
outperforms existing graph representation learning methods. Moreover, our edge
representation learning and pooling method also largely outperforms
state-of-the-art graph pooling methods on graph classification, not only
because of its accurate edge representation learning, but also due to its
lossless compression of the nodes and removal of irrelevant edges for effective
message-passing.

    

### [[2108.10226] ECG-Based Heart Arrhythmia Diagnosis Through Attentional Convolutional Neural Networks](http://arxiv.org/abs/2108.10226)


  Electrocardiography (ECG) signal is a highly applied measurement for
individual heart condition, and much effort have been endeavored towards
automatic heart arrhythmia diagnosis based on machine learning. However,
traditional machine learning models require large investment of time and effort
for raw data preprocessing and feature extraction, as well as challenged by
poor classification performance. Here, we propose a novel deep learning model,
named Attention-Based Convolutional Neural Networks (ABCNN) that taking
advantage of CNN and multi-head attention, to directly work on the raw ECG
signals and automatically extract the informative dependencies for accurate
arrhythmia detection. To evaluate the proposed approach, we conduct extensive
experiments over a benchmark ECG dataset. Our main task is to find the
arrhythmia from normal heartbeats and, at the meantime, accurately recognize
the heart diseases from five arrhythmia types. We also provide convergence
analysis of ABCNN and intuitively show the meaningfulness of extracted
representation through visualization. The experimental results show that the
proposed ABCNN outperforms the widely used baselines, which puts one step
closer to intelligent heart disease diagnosis system.

    

### [[2109.09495] GhostShiftAddNet: More Features from Energy-Efficient Operations](http://arxiv.org/abs/2109.09495)


  Deep convolutional neural networks (CNNs) are computationally and memory
intensive. In CNNs, intensive multiplication can have resource implications
that may challenge the ability for effective deployment of inference on
resource-constrained edge devices. This paper proposes GhostShiftAddNet, where
the motivation is to implement a hardware-efficient deep network: a
multiplication-free CNN with fewer redundant features. We introduce a new
bottleneck block, GhostSA, that converts all multiplications in the block to
cheap operations. The bottleneck uses an appropriate number of bit-shift
filters to process intrinsic feature maps, then applies a series of
transformations that consist of bit-wise shifts with addition operations to
generate more feature maps that fully learn to capture information underlying
intrinsic features. We schedule the number of bit-shift and addition operations
for different hardware platforms. We conduct extensive experiments and ablation
studies with desktop and embedded (Jetson Nano) devices for implementation and
measurements. We demonstrate the proposed GhostSA block can replace bottleneck
blocks in the backbone of state-of-the-art networks architectures and gives
improved performance on image classification benchmarks. Further, our
GhostShiftAddNet can achieve higher classification accuracy with fewer FLOPs
and parameters (reduced by up to 3x) than GhostNet. When compared to GhostNet,
inference latency on the Jetson Nano is improved by 1.3x and 2x on the GPU and
CPU respectively.

    

### [[2110.04719] Structure learning in polynomial time: Greedy algorithms, Bregman information, and exponential families](http://arxiv.org/abs/2110.04719)


  Greedy algorithms have long been a workhorse for learning graphical models,
and more broadly for learning statistical models with sparse structure. In the
context of learning directed acyclic graphs, greedy algorithms are popular
despite their worst-case exponential runtime. In practice, however, they are
very efficient. We provide new insight into this phenomenon by studying a
general greedy score-based algorithm for learning DAGs. Unlike edge-greedy
algorithms such as the popular GES and hill-climbing algorithms, our approach
is vertex-greedy and requires at most a polynomial number of score evaluations.
We then show how recent polynomial-time algorithms for learning DAG models are
a special case of this algorithm, thereby illustrating how these order-based
algorithms can be rigourously interpreted as score-based algorithms. This
observation suggests new score functions and optimality conditions based on the
duality between Bregman divergences and exponential families, which we explore
in detail. Explicit sample and computational complexity bounds are derived.
Finally, we provide extensive experiments suggesting that this algorithm indeed
optimizes the score in a variety of settings.

    

### [[2110.04995] The Skellam Mechanism for Differentially Private Federated Learning](http://arxiv.org/abs/2110.04995)


  We introduce the multi-dimensional Skellam mechanism, a discrete differential
privacy mechanism based on the difference of two independent Poisson random
variables. To quantify its privacy guarantees, we analyze the privacy loss
distribution via a numerical evaluation and provide a sharp bound on the
Rényi divergence between two shifted Skellam distributions. While useful in
both centralized and distributed privacy applications, we investigate how it
can be applied in the context of federated learning with secure aggregation
under communication constraints. Our theoretical findings and extensive
experimental evaluations demonstrate that the Skellam mechanism provides the
same privacy-accuracy trade-offs as the continuous Gaussian mechanism, even
when the precision is low. More importantly, Skellam is closed under summation
and sampling from it only requires sampling from a Poisson distribution -- an
efficient routine that ships with all machine learning and data analysis
software packages. These features, along with its discrete nature and
competitive privacy-accuracy trade-offs, make it an attractive practical
alternative to the newly introduced discrete Gaussian mechanism.

    

### [[2110.05069] Efficient Training of Audio Transformers with Patchout](http://arxiv.org/abs/2110.05069)


  The great success of transformer-based models in natural language processing
(NLP) has led to various attempts at adapting these architectures to other
domains such as vision and audio. Recent work has shown that transformers can
outperform Convolutional Neural Networks (CNNs) on vision and audio tasks.
However, one of the main shortcomings of transformer models, compared to the
well-established CNNs, is the computational complexity. Compute and memory
complexity grow quadratically with the input length. Therefore, there has been
extensive work on optimizing transformers, but often at the cost of lower
predictive performance. In this work, we propose a novel method to optimize and
regularize transformers on audio spectrograms. The proposed models achieve a
new state-of-the-art performance on Audioset and can be trained on a single
consumer-grade GPU. Furthermore, we propose a transformer model that
outperforms CNNs in terms of both performance and training speed.

    

### [[2110.11269] Modeling the AC Power Flow Equations with Optimally Compact Neural Networks: Application to Unit Commitment](http://arxiv.org/abs/2110.11269)


  Nonlinear power flow constraints render a variety of power system
optimization problems computationally intractable. Emerging research shows,
however, that the nonlinear AC power flow equations can be successfully modeled
using Neural Networks (NNs). These NNs can be exactly transformed into Mixed
Integer Linear Programs (MILPs) and embedded inside challenging optimization
problems, thus replacing nonlinearities that are intractable for many
applications with tractable piecewise linear approximations. Such approaches,
though, suffer from an explosion of the number of binary variables needed to
represent the NN. Accordingly, this paper develops a technique for training an
"optimally compact" NN, i.e., one that can represent the power flow equations
with a sufficiently high degree of accuracy while still maintaining a tractable
number of binary variables. We show that the resulting NN model is more
expressive than both the DC and linearized power flow approximations when
embedded inside of a challenging optimization problem (i.e., the AC unit
commitment problem).

    

### [[1807.08221] A Preliminary Study On the Sustainability of Android Malware Detection](http://arxiv.org/abs/1807.08221)


  Machine learning-based malware detection dominates current security defense
approaches for Android apps. However, due to the evolution of Android platforms
and malware, existing such techniques are widely limited by their need for
constant retraining that are costly, and reliance on new malware samples that
may not be timely available. As a result, new and emerging malware slips
through, as seen from the continued surging of malware in the wild. Thus, a
more practical detector needs not only to be accurate but, more critically, to
be able to sustain its capabilities over time without frequent retraining. In
this paper, we study how Android apps evolve as a population over time, in
terms of their behaviors related to accesses to sensitive information and
operations. We first perform a longitudinal characterization of 6K benign and
malicious apps developed across seven years, with focus on these sensitive
accesses in app executions. Our study reveals, during the long evolution, a
consistent, clear differentiation between malware and benign apps regarding
such accesses, measured by relative statistics of relevant method calls.
Following these findings, we developed DroidSpan, a novel classification system
based on a new behavioral profile for Android apps. Through an extensive
evaluation, we showed that DroidSpan can not only effectively detect malware
but sustain high detection accuracy (93% F1 measure) for four years (with 81%
F1 for five years). Through a dedicated study, we also showed its resiliency to
sophisticated evasion schemes. By comparing to a state-of-the-art malware
detector, we demonstrated the largely superior sustainability of our approach
at reasonable costs.

    

### [[2110.15804] Doubt and Redundancy Kill Soft Errors -- Towards Detection and Correction of Silent Data Corruption in Task-based Numerical Software](http://arxiv.org/abs/2110.15804)


  Resilient algorithms in high-performance computing are subject to rigorous
non-functional constraints. Resiliency must not increase the runtime, memory
footprint or I/O demands too significantly. We propose a task-based soft error
detection scheme that relies on error criteria per task outcome. They formalise
how ``dubious'' an outcome is, i.e. how likely it contains an error. Our whole
simulation is replicated once, forming two teams of MPI ranks that share their
task results. Thus, ideally each team handles only around half of the workload.
If a task yields large error criteria values, i.e.~is dubious, we compute the
task redundantly and compare the outcomes. Whenever they disagree, the task
result with a lower error likeliness is accepted. We obtain a self-healing,
resilient algorithm which can compensate silent floating-point errors without a
significant performance, I/O or memory footprint penalty. Case studies however
suggest that a careful, domain-specific tailoring of the error criteria remains
essential.

    

### [[2110.15457] DFL: High-Performance Blockchain-Based Federated Learning](http://arxiv.org/abs/2110.15457)


  Many researchers are trying to replace the aggregation server in federated
learning with a blockchain system to achieve better privacy, robustness and
scalability. In this case, clients will upload their updated models to the
blockchain ledger, and use a smart contract on the blockchain system to perform
model averaging. However, running machine learning applications on the
blockchain is almost impossible because a blockchain system, which usually
takes over half minute to generate a block, is extremely slow and unable to
support machine learning applications.
This paper proposes a completely new public blockchain architecture called
DFL, which is specially optimized for distributed federated machine learning.
This architecture inherits most traditional blockchain merits and achieves
extremely high performance with low resource consumption by waiving global
consensus. To characterize the performance and robustness of our architecture,
we implement the architecture as a prototype and test it on a physical
four-node network. To test more nodes and more complex situations, we build a
simulator to simulate the network. The LeNet results indicate our system can
reach over 90% accuracy for non-I.I.D. datasets even while facing model
poisoning attacks, with the blockchain consuming less than 5% of hardware
resources.

    

### [[2110.15669] SDP: Scalable Real-time Dynamic Graph Partitioner](http://arxiv.org/abs/2110.15669)


  Time-evolving large graph has received attention due to their participation
in real-world applications such as social networks and PageRank calculation. It
is necessary to partition a large-scale dynamic graph in a streaming manner to
overcome the memory bottleneck while partitioning the computational load.
Reducing network communication and balancing the load between the partitions
are the criteria to achieve effective run-time performance in graph
partitioning. Moreover, an optimal resource allocation is needed to utilise the
resources while catering the graph streams into the partitions. A number of
existing partitioning algorithms (ADP, LogGP and LEOPARD) have been proposed to
address the above problem. However, these partitioning methods are incapable of
scaling the resources and handling the stream of data in real-time.
In this study, we propose a dynamic graph partitioning method called Scalable
Dynamic Graph Partitioner (SDP) using the streaming partitioning technique. The
SDP contributes a novel vertex assigning method, communication-aware balancing
method, and a scaling technique to produce an efficient dynamic graph
partitioner. Experiment results show that the proposed method achieves up to
90% reduction of communication cost and 60%-70% balancing the load dynamically,
compared with previous algorithms. Moreover, the proposed algorithm
significantly reduces the execution time during partitioning.

    

### [[2110.15702] DeF-DReL: Systematic Deployment of Serverless Functions in Fog and Cloud environments using Deep Reinforcement Learning](http://arxiv.org/abs/2110.15702)


  Fog computing is introduced by shifting cloud resources towards the users'
proximity to mitigate the limitations possessed by cloud computing. Fog
environment made its limited resource available to a large number of users to
deploy their serverless applications, composed of several serverless functions.
One of the primary intentions behind introducing the fog environment is to
fulfil the demand of latency and location-sensitive serverless applications
through its limited resources. The recent research mainly focuses on assigning
maximum resources to such applications from the fog node and not taking full
advantage of the cloud environment. This introduces a negative impact in
providing the resources to a maximum number of connected users. To address this
issue, in this paper, we investigated the optimum percentage of a user's
request that should be fulfilled by fog and cloud. As a result, we proposed
DeF-DReL, a Systematic Deployment of Serverless Functions in Fog and Cloud
environments using Deep Reinforcement Learning, using several real-life
parameters, such as distance and latency of the users from nearby fog node,
user's priority, the priority of the serverless applications and their resource
demand, etc. The performance of the DeF-DReL algorithm is further compared with
recent related algorithms. From the simulation and comparison results, its
superiority over other algorithms and its applicability to the real-life
scenario can be clearly observed.

    

### [[2110.15869] Trustworthy Pre-Processing of Sensor Data in Data On-chaining Workflows for Blockchain-based IoT Applications](http://arxiv.org/abs/2110.15869)


  Prior to provisioning sensor data to smart contracts, a pre-processing of the
data on intermediate off-chain nodes is often necessary. When doing so,
originally constructed cryptographic signatures cannot be verified on-chain
anymore. This exposes an opportunity for undetected manipulation and presents a
problem for applications in the Internet of Things where trustworthy sensor
data is required on-chain. In this paper, we propose trustworthy pre-processing
as enabler for end-to-end sensor data integrity in data on-chaining workflows.
We define requirements for trustworthy pre-processing, present a model and
common workflow for data on-chaining, select off-chain computation utilizing
Zero-knowledge Proofs (ZKPs) and Trusted Execution Environments (TEEs) as
promising solution approaches, and discuss both our proof-of-concept
implementations and initial experimental, comparative evaluation results. The
importance of trustworthy pre-processing and principle solution approaches are
presented, addressing the major problem of end-to-end sensor data integrity in
blockchain-based IoT applications.

    

### [[1310.2994] Depth-dependent Parallel Visualization with 3D Stylized Dense Tubes](http://arxiv.org/abs/1310.2994)


  We present a parallel visualization algorithm for the illustrative rendering
of depth-dependent stylized dense tube data at interactive frame rates. While
this computation could be efficiently performed on a GPU device, we target a
parallel framework to enable it to be efficiently running on an ordinary
multi-core CPU platform which is much more available than GPUs for common
users. Our approach is to map the depth information in each tube onto each of
the visual dimensions of shape, color, texture, value, and size on the basis of
Bertin's semiology theory. The purpose is to enable more legible displays in
the dense tube environments. A major contribution of our work is an efficient
and effective parallel depthordering algorithm that makes use of the message
passing interface (MPI) with VTK. We evaluated our framework with
visualizations of depth-stylized tubes derived from 3D diffusion tensor MRI
data by comparing its efficiency with several other alternative parallelization
platforms running the same computations. As our results show, the
parallelization framework we proposed can efficiently render highly dense 3D
data sets like the tube data and thus is useful as a complement to parallel
visualization environments that rely on GPUs.

    

### [[2110.15385] Data-driven Residual Generation for Early Fault Detection with Limited Data](http://arxiv.org/abs/2110.15385)


  Traditionally, fault detection and isolation community has used system
dynamic equations to generate diagnosers and to analyze detectability and
isolability of the dynamic systems. Model-based fault detection and isolation
methods use system model to generate a set of residuals as the bases for fault
detection and isolation. However, in many complex systems it is not feasible to
develop highly accurate models for the systems and to keep the models updated
during the system lifetime. Recently, data-driven solutions have received an
immense attention in the industries systems for several practical reasons.
First, these methods do not require the initial investment and expertise for
developing accurate models. Moreover, it is possible to automatically update
and retrain the diagnosers as the system or the environment change over time.
Finally, unlike the model-based methods it is straight forward to combine time
series measurements such as pressure and voltage with other sources of
information such as system operating hours to achieve a higher accuracy. In
this paper, we extend the traditional model-based fault detection and isolation
concepts such as residuals, and detectable and isolable faults to the
data-driven domain. We then propose an algorithm to automatically generate
residuals from the normal operating data. We present the performance of our
proposed approach through a comparative case study.

    

### [[2110.15387] Anticipation-driven Adaptive Architecture for Assisted Living](http://arxiv.org/abs/2110.15387)


  Anticipatory expression underlies human performance. Medical conditions and,
especially, aging result in diminished anticipatory action. In order to
mitigate the loss, means for engaging still available resources (capabilities)
can be provided. In particular, anticipation-driven adaptive environments could
be beneficial in medical care, as well as in assisted living for those seeking
such assistance. These adaptive environments are conceived to be individualized
and individualizable, in order to stimulate independent action instead of
creating dependencies.

    

### [[2110.15415] On the Use of CSI for the Generation of RF Fingerprints and Secret Keys](http://arxiv.org/abs/2110.15415)


  This paper presents a systematic approach to use channel state information
for authentication and secret key distillation for physical layer security
(PLS). We use popular machine learning (ML) methods and signal processing-based
approaches to disentangle the large scale fading and be used as a source of
uniqueness, from the small scale fading, to be treated as a source of shared
entropy secret key generation (SKG). The ML-based approaches are completely
unsupervised and hence avoid exhaustive measurement campaigns. We also propose
using the Hilbert Schmidt independence criterion (HSIC); our simulation results
demonstrate that the extracted stochastic part of the channel state information
(CSI) vectors are statistically independent.

    

### [[2110.15525] PEDENet: Image Anomaly Localization via Patch Embedding and Density Estimation](http://arxiv.org/abs/2110.15525)


  A neural network targeting at unsupervised image anomaly localization, called
the PEDENet, is proposed in this work. PEDENet contains a patch embedding (PE)
network, a density estimation (DE) network, and an auxiliary network called the
location prediction (LP) network. The PE network takes local image patches as
input and performs dimension reduction to get low-dimensional patch embeddings
via a deep encoder structure. Being inspired by the Gaussian Mixture Model
(GMM), the DE network takes those patch embeddings and then predicts the
cluster membership of an embedded patch. The sum of membership probabilities is
used as a loss term to guide the learning process. The LP network is a
Multi-layer Perception (MLP), which takes embeddings from two neighboring
patches as input and predicts their relative location. The performance of the
proposed PEDENet is evaluated extensively and benchmarked with that of
state-of-the-art methods.

    

### [[2110.15527] Pre-training Co-evolutionary Protein Representation via A Pairwise Masked Language Model](http://arxiv.org/abs/2110.15527)


  Understanding protein sequences is vital and urgent for biology, healthcare,
and medicine. Labeling approaches are expensive yet time-consuming, while the
amount of unlabeled data is increasing quite faster than that of the labeled
data due to low-cost, high-throughput sequencing methods. In order to extract
knowledge from these unlabeled data, representation learning is of significant
value for protein-related tasks and has great potential for helping us learn
more about protein functions and structures. The key problem in the protein
sequence representation learning is to capture the co-evolutionary information
reflected by the inter-residue co-variation in the sequences. Instead of
leveraging multiple sequence alignment as is usually done, we propose a novel
method to capture this information directly by pre-training via a dedicated
language model, i.e., Pairwise Masked Language Model (PMLM). In a conventional
masked language model, the masked tokens are modeled by conditioning on the
unmasked tokens only, but processed independently to each other. However, our
proposed PMLM takes the dependency among masked tokens into consideration,
i.e., the probability of a token pair is not equal to the product of the
probability of the two tokens. By applying this model, the pre-trained encoder
is able to generate a better representation for protein sequences. Our result
shows that the proposed method can effectively capture the inter-residue
correlations and improves the performance of contact prediction by up to 9%
compared to the MLM baseline under the same setting. The proposed model also
significantly outperforms the MSA baseline by more than 7% on the TAPE contact
prediction benchmark when pre-trained on a subset of the sequence database
which the MSA is generated from, revealing the potential of the sequence
pre-training method to surpass MSA based methods in general.

    

### [[2110.15599] Handshakes AI Research at CASE 2021 Task 1: Exploring different approaches for multilingual tasks](http://arxiv.org/abs/2110.15599)


  The aim of the CASE 2021 Shared Task 1 (Hürriyetoğlu et al., 2021) was
to detect and classify socio-political and crisis event information at
document, sentence, cross-sentence, and token levels in a multilingual setting,
with each of these subtasks being evaluated separately in each test language.
Our submission contained entries in all of the subtasks, and the scores
obtained validated our research finding: That the multilingual aspect of the
tasks should be embraced, so that modeling and training regimes use the
multilingual nature of the tasks to their mutual benefit, rather than trying to
tackle the different languages separately. Our code is available at
this https URL


### [[2110.15681] False Positive Detection and Prediction Quality Estimation for LiDAR Point Cloud Segmentation](http://arxiv.org/abs/2110.15681)


  We present a novel post-processing tool for semantic segmentation of LiDAR
point cloud data, called LidarMetaSeg, which estimates the prediction quality
segmentwise. For this purpose we compute dispersion measures based on network
probability outputs as well as feature measures based on point cloud input
features and aggregate them on segment level. These aggregated measures are
used to train a meta classification model to predict whether a predicted
segment is a false positive or not and a meta regression model to predict the
segmentwise intersection over union. Both models can then be applied to
semantic segmentation inferences without knowing the ground truth. In our
experiments we use different LiDAR segmentation models and datasets and analyze
the power of our method. We show that our results outperform other standard
approaches.

    

### [[2110.15695] A Protocol for Emotions](http://arxiv.org/abs/2110.15695)


  We tend to consider emotions a manifestation of our innermost nature of human
beings. Emotions characterize our lives in many ways and they chaperon every
rational activity we carry out. Despite their pervasiveness, there are still
many things we ignore about emotions. Among them, our understanding of how
living beings transfer emotions is limited. In particular, there are highly
sophisticated interactions between human beings that we would like to
comprehend. For instance, think of a movie director who knows in advance the
strong emotional impact that a certain scene will have on the spectators.
Although many artists rely on some emotional devices, their talent and vision
are still the key factors.
In this work we analyze high-level protocols for transferring emotions
between two intelligent agents. To the best of our knowledge, this is the first
attempt to use communication protocols for modeling the exchange of human
emotions. By means of a number of examples, we show that our protocols
adequately model the engagement of the two parties. Beyond the theoretical
interest, our proposal can provide a stepping stone for several applications
that we also discuss in this paper.

    

### [[2110.15706] Integrating Deep Event-Level and Script-Level Information for Script Event Prediction](http://arxiv.org/abs/2110.15706)


  Scripts are structured sequences of events together with the participants,
which are extracted from the texts.Script event prediction aims to predict the
subsequent event given the historical events in the script. Two kinds of
information facilitate this task, namely, the event-level information and the
script-level information. At the event level, existing studies view an event as
a verb with its participants, while neglecting other useful properties, such as
the state of the participants. At the script level, most existing studies only
consider a single event sequence corresponding to one common protagonist. In
this paper, we propose a Transformer-based model, called MCPredictor, which
integrates deep event-level and script-level information for script event
prediction. At the event level, MCPredictor utilizes the rich information in
the text to obtain more comprehensive event semantic representations. At the
script-level, it considers multiple event sequences corresponding to different
participants of the subsequent event. The experimental results on the
widely-used New York Times corpus demonstrate the effectiveness and superiority
of the proposed model.

    

### [[2110.15719] Generational Frameshifts in Technology: Computer Science and Neurosurgery, The VR Use Case](http://arxiv.org/abs/2110.15719)


  We are at a unique moment in history where there is a confluence of
technologies which will synergistically come together to transform the practice
of neurosurgery. These technological transformations will be all-encompassing,
including improved tools and methods for intraoperative performance of
neurosurgery, scalable solutions for asynchronous neurosurgical training and
simulation, as well as broad aggregation of operative data allowing fundamental
changes in quality assessment, billing, outcome measures, and dissemination of
surgical best practices. The ability to perform surgery more safely and more
efficiently while capturing the operative details and parsing each component of
the operation will open an entirely new epoch advancing our field and all
surgical specialties. The digitization of all components within the operating
room will allow us to leverage the various fields within computer and
computational science to obtain new insights that will improve care and
delivery of the highest quality neurosurgery regardless of location. The
democratization of neurosurgery is at hand and will be driven by our
development, extraction, and adoption of these tools of the modern world.
Virtual reality provides a good example of how consumer-facing technologies are
finding a clear role in industry and medicine and serves as a notable example
of the confluence of various computer science technologies creating a novel
paradigm for scaling human ability and interactions. The authors describe the
technology ecosystem that has come and highlight a myriad of computational and
data sciences that will be necessary to enable the operating room of the near
future.

    

### [[2110.15720] Weakly Supervised Concept Map Generation through Task-Guided Graph Translation](http://arxiv.org/abs/2110.15720)


  Recent years have witnessed the rapid development of concept map generation
techniques due to their advantages in providing well-structured summarization
of knowledge from free texts. Traditional unsupervised methods do not generate
task-oriented concept maps, whereas deep generative models require large
amounts of training data. In this work, we present GT-D2G (Graph Translation
based Document-To-Graph), an automatic concept map generation framework that
leverages generalized NLP pipelines to derive semantic-rich initial graphs, and
translates them into more concise structures under the weak supervision of
document labels. The quality and interpretability of such concept maps are
validated through human evaluation on three real-world corpora, and their
utility in the downstream task is further demonstrated in the controlled
experiments with scarce document labels.

    

### [[2110.15721] Paperswithtopic: Topic Identification from Paper Title Only](http://arxiv.org/abs/2110.15721)


  The deep learning field is growing rapidly as witnessed by the exponential
growth of papers submitted to journals, conferences, and pre-print servers. To
cope with the sheer number of papers, several text mining tools from natural
language processing (NLP) have been proposed that enable researchers to keep
track of recent findings. In this context, our paper makes two main
contributions: first, we collected and annotated a dataset of papers paired by
title and sub-field from the field of artificial intelligence (AI), and,
second, we present results on how to predict a paper's AI sub-field from a
given paper title only. Importantly, for the latter, short-text classification
task we compare several algorithms from conventional machine learning all the
way up to recent, larger transformer architectures. Finally, for the
transformer models, we also present gradient-based, attention visualizations to
further explain the model's classification process. All code can be found at
\url{this https URL}

    

### [[2110.15722] A Novel Sequence Tagging Framework for Consumer Event-Cause Extraction](http://arxiv.org/abs/2110.15722)


  Consumer Event-Cause Extraction, the task aimed at extracting the potential
causes behind certain events in the text, has gained much attention in recent
years due to its wide applications. The ICDM 2020 conference sets up an
evaluation competition that aims to extract events and the causes of the
extracted events with a specified subject (a brand or product). In this task,
we mainly focus on how to construct an end-to-end model, and extract multiple
event types and event-causes simultaneously. To this end, we introduce a fresh
perspective to revisit the relational event-cause extraction task and propose a
novel sequence tagging framework, instead of extracting event types and
events-causes separately. Experiments show our framework outperforms baseline
methods even when its encoder module uses an initialized pre-trained BERT
encoder, showing the power of the new tagging framework. In this competition,
our team achieved 1st place in the first stage leaderboard, and 3rd place in
the final stage leaderboard.

    

### [[2110.15726] Social Media Reveals Urban-Rural Differences in Stress across China](http://arxiv.org/abs/2110.15726)


  Modeling differential stress expressions in urban and rural regions in China
can provide a better understanding of the effects of urbanization on
psychological well-being in a country that has rapidly grown economically in
the last two decades. This paper studies linguistic differences in the
experiences and expressions of stress in urban-rural China from Weibo posts
from over 65,000 users across 329 counties using hierarchical mixed-effects
models. We analyzed phrases, topical themes, and psycho-linguistic word choices
in Weibo posts mentioning stress to better understand appraisal differences
surrounding psychological stress in urban and rural communities in China; we
then compared them with large-scale polls from Gallup. After controlling for
socioeconomic and gender differences, we found that rural communities tend to
express stress in emotional and personal themes such as relationships, health,
and opportunity while users in urban areas express stress using relative,
temporal, and external themes such as work, politics, and economics. These
differences exist beyond controlling for GDP and urbanization, indicating a
fundamentally different lifestyle between rural and urban residents in very
specific environments, arguably having different sources of stress. We found
corroborative trends in physical, financial, and social wellness with
urbanization in Gallup polls.

    

### [[2110.15727] Calling to CNN-LSTM for Rumor Detection: A Deep Multi-channel Model for Message Veracity Classification in Microblogs](http://arxiv.org/abs/2110.15727)


  Reputed by their low-cost, easy-access, real-time and valuable information,
social media also wildly spread unverified or fake news. Rumors can notably
cause severe damage on individuals and the society. Therefore, rumor detection
on social media has recently attracted tremendous attention. Most rumor
detection approaches focus on rumor feature analysis and social features, i.e.,
metadata in social media. Unfortunately, these features are data-specific and
may not always be available, e.g., when the rumor has just popped up and not
yet propagated. In contrast, post contents (including images or videos) play an
important role and can indicate the diffusion purpose of a rumor. Furthermore,
rumor classification is also closely related to opinion mining and sentiment
analysis. Yet, to the best of our knowledge, exploiting images and sentiments
is little investigated.Considering the available multimodal features from
microblogs, notably, we propose in this paper an end-to-end model called
deepMONITOR that is based on deep neural networks and allows quite accurate
automated rumor verification, by utilizing all three characteristics: post
textual and image contents, as well as sentiment. deepMONITOR concatenates
image features with the joint text and sentiment features to produce a
reliable, fused classification. We conduct extensive experiments on two
large-scale, real-world datasets. The results show that deepMONITOR achieves a
higher accuracy than state-of-the-art methods.

    

### [[2110.15730] E-Commerce Dispute Resolution Prediction](http://arxiv.org/abs/2110.15730)


  E-Commerce marketplaces support millions of daily transactions, and some
disagreements between buyers and sellers are unavoidable. Resolving disputes in
an accurate, fast, and fair manner is of great importance for maintaining a
trustworthy platform. Simple cases can be automated, but intricate cases are
not sufficiently addressed by hard-coded rules, and therefore most disputes are
currently resolved by people. In this work we take a first step towards
automatically assisting human agents in dispute resolution at scale. We
construct a large dataset of disputes from the eBay online marketplace, and
identify several interesting behavioral and linguistic patterns. We then train
classifiers to predict dispute outcomes with high accuracy. We explore the
model and the dataset, reporting interesting correlations, important features,
and insights.

    

### [[2110.15757] On Structural Parameterizations of the Offensive Alliance Problem](http://arxiv.org/abs/2110.15757)


  The Offensive Alliance problem has been studied extensively during the last
twenty years. A set $S\subseteq V$ of vertices is an offensive alliance in an
undirected graph $G=(V,E)$ if each $v\in N(S)$ has at least as many neighbours
in $S$ as it has neighbours (including itself) not in $S$. We study the
parameterized complexity of the Offensive Alliance problem, where the aim is to
find a minimum size offensive alliance. Our focus here lies on parameters that
measure the structural properties of the input instance. We enhance our
understanding of the problem from the viewpoint of parameterized complexity by
showing that the problem is W[1]-hard parameterized by a wide range of fairly
restrictive structural parameters such as the feedback vertex set number,
treewidth, pathwidth, and treedepth of the input graph.

    

### [[2110.15766] NxMTransformer: Semi-Structured Sparsification for Natural Language Understanding via ADMM](http://arxiv.org/abs/2110.15766)


  Natural Language Processing (NLP) has recently achieved success by using huge
pre-trained Transformer networks. However, these models often contain hundreds
of millions or even billions of parameters, bringing challenges to online
deployment due to latency constraints. Recently, hardware manufacturers have
introduced dedicated hardware for NxM sparsity to provide the flexibility of
unstructured pruning with the runtime efficiency of structured approaches. NxM
sparsity permits arbitrarily selecting M parameters to retain from a contiguous
group of N in the dense representation. However, due to the extremely high
complexity of pre-trained models, the standard sparse fine-tuning techniques
often fail to generalize well on downstream tasks, which have limited data
resources. To address such an issue in a principled manner, we introduce a new
learning framework, called NxMTransformer, to induce NxM semi-structured
sparsity on pretrained language models for natural language understanding to
obtain better performance. In particular, we propose to formulate the NxM
sparsity as a constrained optimization problem and use Alternating Direction
Method of Multipliers (ADMM) to optimize the downstream tasks while taking the
underlying hardware constraints into consideration. ADMM decomposes the NxM
sparsification problem into two sub-problems that can be solved sequentially,
generating sparsified Transformer networks that achieve high accuracy while
being able to effectively execute on newly released hardware. We apply our
approach to a wide range of NLP tasks, and our proposed method is able to
achieve 1.7 points higher accuracy in GLUE score than current practices.
Moreover, we perform detailed analysis on our approach and shed light on how
ADMM affects fine-tuning accuracy for downstream tasks. Finally, we illustrate
how NxMTransformer achieves performance improvement with knowledge
distillation.

    

### [[2110.15790] LSTM-RPA: A Simple but Effective Long Sequence Prediction Algorithm for Music Popularity Prediction](http://arxiv.org/abs/2110.15790)


  The big data about music history contains information about time and users'
behavior. Researchers could predict the trend of popular songs accurately by
analyzing this data. The traditional trend prediction models can better predict
the short trend than the long trend. In this paper, we proposed the improved
LSTM Rolling Prediction Algorithm (LSTM-RPA), which combines LSTM historical
input with current prediction results as model input for next time prediction.
Meanwhile, this algorithm converts the long trend prediction task into multiple
short trend prediction tasks. The evaluation results show that the LSTM-RPA
model increased F score by 13.03%, 16.74%, 11.91%, 18.52%, compared with LSTM,
BiLSTM, GRU and RNN. And our method outperforms tradi-tional sequence models,
which are ARIMA and SMA, by 10.67% and 3.43% improvement in F score.Code:
this https URL


### [[2110.15794] CLAUSEREC: A Clause Recommendation Framework for AI-aided Contract Authoring](http://arxiv.org/abs/2110.15794)


  Contracts are a common type of legal document that frequent in several
day-to-day business workflows. However, there has been very limited NLP
research in processing such documents, and even lesser in generating them.
These contracts are made up of clauses, and the unique nature of these clauses
calls for specific methods to understand and generate such documents. In this
paper, we introduce the task of clause recommendation, asa first step to aid
and accelerate the author-ing of contract documents. We propose a two-staged
pipeline to first predict if a specific clause type is relevant to be added in
a contract, and then recommend the top clauses for the given type based on the
contract context. We pretrain BERT on an existing library of clauses with two
additional tasks and use it for our prediction and recommendation. We
experiment with classification methods and similarity-based heuristics for
clause relevance prediction, and generation-based methods for clause
recommendation, and evaluate the results from various methods on several clause
types. We provide analyses on the results, and further outline the advantages
and limitations of the various methods for this line of research.

    

### [[2110.15799] Guided Policy Search for Parameterized Skills using Adverbs](http://arxiv.org/abs/2110.15799)


  We present a method for using adverb phrases to adjust skill parameters via
learned adverb-skill groundings. These groundings allow an agent to use adverb
feedback provided by a human to directly update a skill policy, in a manner
similar to traditional local policy search methods. We show that our method can
be used as a drop-in replacement for these policy search methods when dense
reward from the environment is not available but human language feedback is. We
demonstrate improved sample efficiency over modern policy search methods in two
experiments.

    

### [[2110.15803] Natural Language Processing for Smart Healthcare](http://arxiv.org/abs/2110.15803)


  Smart healthcare has achieved significant progress in recent years. Emerging
artificial intelligence (AI) technologies enable various smart applications
across various healthcare scenarios. As an essential technology powered by AI,
natural language processing (NLP) plays a key role in smart healthcare due to
its capability of analysing and understanding human language. In this work we
review existing studies that concern NLP for smart healthcare from the
perspectives of technique and application. We focus on feature extraction and
modelling for various NLP tasks encountered in smart healthcare from a
technical point of view. In the context of smart healthcare applications
employing NLP techniques, the elaboration largely attends to representative
smart healthcare scenarios, including clinical practice, hospital management,
personal care, public health, and drug development. We further discuss the
limitations of current works and identify the directions for future works.

    

### [[2110.15926] Delayed Propagation Transformer: A Universal Computation Engine towards Practical Control in Cyber-Physical Systems](http://arxiv.org/abs/2110.15926)


  Multi-agent control is a central theme in the Cyber-Physical Systems (CPS).
However, current control methods either receive non-Markovian states due to
insufficient sensing and decentralized design, or suffer from poor convergence.
This paper presents the Delayed Propagation Transformer (DePT), a new
transformer-based model that specializes in the global modeling of CPS while
taking into account the immutable constraints from the physical world. DePT
induces a cone-shaped spatial-temporal attention prior, which injects the
information propagation and aggregation principles and enables a global view.
With physical constraint inductive bias baked into its design, our DePT is
ready to plug and play for a broad class of multi-agent systems. The
experimental results on one of the most challenging CPS -- network-scale
traffic signal control system in the open world -- show that our model
outperformed the state-of-the-art expert methods on synthetic and real-world
datasets. Our codes are released at: this https URL.

    

### [[2110.15943] MetaICL: Learning to Learn In Context](http://arxiv.org/abs/2110.15943)


  We introduce MetaICL (Meta-training for In-Context Learning), a new
meta-training framework for few-shot learning where a pretrained language model
is tuned to do in-context learn-ing on a large set of training tasks. This
meta-training enables the model to more effectively learn a new task in context
at test time, by simply conditioning on a few training examples with no
parameter updates or task-specific templates. We experiment on a large, diverse
collection of tasks consisting of 142 NLP datasets including classification,
question answering, natural language inference, paraphrase detection and more,
across seven different meta-training/target splits. MetaICL outperforms a range
of baselines including in-context learning without meta-training and multi-task
learning followed by zero-shot transfer. We find that the gains are
particularly significant for target tasks that have domain shifts from the
meta-training tasks, and that using a diverse set of the meta-training tasks is
key to improvements. We also show that MetaICL approaches (and sometimes beats)
the performance of models fully finetuned on the target task training data, and
outperforms much bigger models with nearly 8x parameters.

    

### [[2102.10557] Contrastive Self-supervised Neural Architecture Search](http://arxiv.org/abs/2102.10557)


  This paper proposes a novel cell-based neural architecture search algorithm
(NAS), which completely alleviates the expensive costs of data labeling
inherited from supervised learning. Our algorithm capitalizes on the
effectiveness of self-supervised learning for image representations, which is
an increasingly crucial topic of computer vision. First, using only a small
amount of unlabeled train data under contrastive self-supervised learning allow
us to search on a more extensive search space, discovering better neural
architectures without surging the computational resources. Second, we entirely
relieve the cost for labeled data (by contrastive loss) in the search stage
without compromising architectures' final performance in the evaluation phase.
Finally, we tackle the inherent discrete search space of the NAS problem by
sequential model-based optimization via the tree-parzen estimator (SMBO-TPE),
enabling us to reduce the computational expense response surface significantly.
An extensive number of experiments empirically show that our search algorithm
can achieve state-of-the-art results with better efficiency in data labeling
cost, searching time, and accuracy in final validation.

    

### [[2110.13047] Drug Similarity and Link Prediction Using Graph Embeddings on Medical Knowledge Graphs](http://arxiv.org/abs/2110.13047)


  The paper utilizes the graph embeddings generated for entities of a large
biomedical database to perform link prediction to capture various new
relationships among different entities. A novel node similarity measure is
proposed that utilizes the graph embeddings and link prediction scores to find
similarity scores among various drugs which can be used by the medical experts
to recommend alternative drugs to avoid side effects from original one.
Utilizing machine learning on knowledge graph for drug similarity and
recommendation will be less costly and less time consuming with higher
scalability as compared to traditional biomedical methods due to the dependency
on costly medical equipment and experts of the latter ones.

    

### [[2104.06262] On Determinism of Game Engines used for Simulation-based Autonomous Vehicle Verification](http://arxiv.org/abs/2104.06262)


  Game engines are increasingly used as simulation platforms by the autonomous
vehicle (AV) community to develop vehicle control systems and test
environments. A key requirement for simulation-based development and
verification is determinism, since a deterministic process will always produce
the same output given the same initial conditions and event history. Thus, in a
deterministic simulation environment, tests are rendered repeatable and yield
simulation results that are trustworthy and straightforward to debug. However,
game engines are seldom deterministic. This paper reviews and identifies the
potential causes of non-deterministic behaviours in game engines. A case study
using CARLA, an open-source autonomous driving simulation environment powered
by Unreal Engine, is presented to highlight its inherent shortcomings in
providing sufficient precision in experimental results. Different
configurations and utilisations of the software and hardware are explored to
determine an operational domain where the simulation precision is sufficiently
low i.e.\ variance between repeated executions becomes negligible for
development and testing work. Finally, a method of a general nature is
proposed, that can be used to find the domains of permissible variance in game
engine simulations for any given system configuration.

    

### [[2110.15425] Cognac: Domain-Specific Compilation for Cognitive Models](http://arxiv.org/abs/2110.15425)


  This paper discusses our proposal and implementation of Cognac, a
domain-specific compilation tool based on LLVM to accelerate cognitive models.
Cognitive models explain the process of cognitive function and offer a path to
human-like artificial intelligence. However, cognitive modeling is laborious,
requiring composition of many types of computational tasks, and suffers from
poor performance as it relies on high-level languages like Python. In order to
continue enjoying the flexibility of Python while achieving high performance,
Cognac uses domain-specific knowledge to compile Python-based cognitive models
into LLVM IR, carefully stripping away features like dynamic typing and memory
management that add overheads to the actual model. As we show, this permits
significantly faster model execution. We also show that the code so generated
enables using classical compiler data flow analysis passes to reveal properties
about data flow in cognitive models that are useful to cognitive scientists.
Cognac is publicly available, is being used by researchers in cognitive
science, and has led to patches that are currently being evaluated for
integration into mainline LLVM.

    

### [[1903.06119] Correct Approximation of IEEE 754 Floating-Point Arithmetic for Program Verification](http://arxiv.org/abs/1903.06119)


  Verification of programs using floating-point arithmetic is challenging on
several accounts. One of the difficulties of reasoning about such programs is
due to the peculiarities of floating-point arithmetic: rounding errors,
infinities, non-numeric objects (NaNs), signed zeroes, denormal numbers,
different rounding modes, etc. One possibility to reason about floating-point
arithmetic is to model a program computation path by means of a set of ternary
constraints of the form z = x op y and use constraint propagation techniques to
infer new information on the variables' possible values. In this setting, we
define and prove the correctness of algorithms to precisely bound the value of
one of the variables x, y or z, starting from the bounds known for the other
two. We do this for each of the operations and for each rounding mode defined
by the IEEE 754 binary floating-point standard, even in the case the rounding
mode in effect is only partially known. This is the first time that such
so-called filtering algorithms are defined and their correctness is formally
proved. This is an important slab for paving the way to formal verification of
programs that use floating-point arithmetics.

    

### [<title data-react-helmet="true">NeurIPS 2021有哪些值得读的NLP论文？ - 知乎</title>](https://zhuanlan.zhihu.com/p/428057558)