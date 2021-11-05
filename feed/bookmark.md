
## 2021-11-5

### [<title>Pyspark3.1.2 with xgboost, where can I find the right version of sparkxgb.zip package - XGBoost</title>](https://discuss.xgboost.ai/t/pyspark3-1-2-with-xgboost-where-can-i-find-the-right-version-of-sparkxgb-zip-package/2525/1)

### [[2111.02630] Network Structure and Feature Learning from Rich but Noisy Data](http://arxiv.org/abs/2111.02630)


  In the study of network structures, much attention has been devoted to
network reconstruction, which relies on partial edge-related information or
dynamical processes on the network. However, there are cases where we are only
given incomplete nodal data, and the nodal data are measured with different
methodologies. In this work, we present an unsupervised learning framework to
construct networks from noisy and heterogeneous nodal data. First, we introduce
the creating nodes' context sets, which are used to generate random node
sequences. Then, a three-layer neural network is adopted to train the node
sequences to infer node vectors, enabling us to capture nodes with synergistic
roles within the network. Further, the effectiveness of the method is validated
through both synthetic data and real data. Finally, we compare the differences
between the global thresholding method and the entropy-based method in edge
selection. In summary, this work presents a neural network method for node
vector learning from heterogeneous nodal data and an entropy-based method for
edge selection.

    

### [[2111.02759] Count-Less: A Counting Sketch for the Data Plane of High Speed Switches](http://arxiv.org/abs/2111.02759)


  Demands are increasing to measure per-flow statistics in the data plane of
high-speed switches. Measuring flows with exact counting is infeasible due to
processing and memory constraints, but a sketch is a promising candidate for
collecting approximately per-flow statistics in data plane in real-time. Among
them, Count-Min sketch is a versatile tool to measure spectral density of high
volume data using a small amount of memory and low processing overhead. Due to
its simplicity and versatility, Count-Min sketch and its variants have been
adopted in many works as a stand alone or even as a supporting measurement
tool. However, Count-Min's estimation accuracy is limited owing to its data
structure not fully accommodating Zipfian distribution and the indiscriminate
update algorithm without considering a counter value. This in turn degrades the
accuracy of heavy hitter, heavy changer, cardinality, and entropy. To enhance
measurement accuracy of Count-Min, there have been many and various attempts.
One of the most notable approaches is to cascade multiple sketches in a
sequential manner so that either mouse or elephant flows should be filtered to
separate elephants from mouse flows such as Elastic sketch (an elephant filter
leveraging TCAM + Count-Min) and FCM sketch (Count-Min-based layered mouse
filters). In this paper, we first show that these cascaded filtering approaches
adopting a Pyramid-shaped data structure (allocating more counters for mouse
flows) still suffer from under-utilization of memory, which gives us a room for
better estimation. To this end, we are facing two challenges: one is (a) how to
make Count-Min's data structure accommodate more effectively Zipfian
distribution, and the other is (b) how to make update and query work without
delaying packet processing in the switch's data plane. Count-Less adopts a
different combination ...

    

### [[2111.02791] A Cyber Threat Intelligence Sharing Scheme based on Federated Learning for Network Intrusion Detection](http://arxiv.org/abs/2111.02791)


  The uses of Machine Learning (ML) in detection of network attacks have been
effective when designed and evaluated in a single organisation. However, it has
been very challenging to design an ML-based detection system by utilising
heterogeneous network data samples originating from several sources. This is
mainly due to privacy concerns and the lack of a universal format of datasets.
In this paper, we propose a collaborative federated learning scheme to address
these issues. The proposed framework allows multiple organisations to join
forces in the design, training, and evaluation of a robust ML-based network
intrusion detection system. The threat intelligence scheme utilises two
critical aspects for its application; the availability of network data traffic
in a common format to allow for the extraction of meaningful patterns across
data sources. Secondly, the adoption of a federated learning mechanism to avoid
the necessity of sharing sensitive users' information between organisations. As
a result, each organisation benefits from other organisations cyber threat
intelligence while maintaining the privacy of its data internally. The model is
trained locally and only the updated weights are shared with the remaining
participants in the federated averaging process. The framework has been
designed and evaluated in this paper by using two key datasets in a NetFlow
format known as NF-UNSW-NB15-v2 and NF-BoT-IoT-v2. Two other common scenarios
are considered in the evaluation process; a centralised training method where
the local data samples are shared with other organisations and a localised
training method where no threat intelligence is shared. The results demonstrate
the efficiency and effectiveness of the proposed framework by designing a
universal ML model effectively classifying benign and intrusive traffic
originating from multiple organisations without the need for local data
exchange.

    

### [[2111.02887] Self-Supervised Radio-Visual Representation Learning for 6G Sensing](http://arxiv.org/abs/2111.02887)


  In future 6G cellular networks, a joint communication and sensing protocol
will allow the network to perceive the environment, opening the door for many
new applications atop a unified communication-perception infrastructure.
However, interpreting the sparse radio representation of sensing scenes is
challenging, which hinders the potential of these emergent systems. We propose
to combine radio and vision to automatically learn a radio-only sensing model
with minimal human intervention. We want to build a radio sensing model that
can feed on millions of uncurated data points. To this end, we leverage recent
advances in self-supervised learning and formulate a new label-free
radio-visual co-learning scheme, whereby vision trains radio via cross-modal
mutual information. We implement and evaluate our scheme according to the
common linear classification benchmark, and report qualitative and quantitative
performance metrics. In our evaluation, the representation learnt by
radio-visual self-supervision works well for a downstream sensing demonstrator,
and outperforms its fully-supervised counterpart when less labelled data is
used. This indicates that self-supervised learning could be an important
enabler for future scalable radio sensing systems.

    

### [[2104.05117] Tracking Normalized Network Traffic Entropy to Detect DDoS Attacks in P4](http://arxiv.org/abs/2104.05117)


  Distributed Denial-of-Service (DDoS) attacks represent a persistent threat to
modern telecommunications networks: detecting and counteracting them is still a
crucial unresolved challenge for network operators. DDoS attack detection is
usually carried out in one or more central nodes that collect significant
amounts of monitoring data from networking devices, potentially creating issues
related to network overload or delay in detection. The dawn of programmable
data planes in Software-Defined Networks can help mitigate this issue, opening
the door to the detection of DDoS attacks directly in the data plane of the
switches. However, the most widely-adopted data plane programming language,
namely P4, lacks supporting many arithmetic operations, therefore, some of the
advanced network monitoring functionalities needed for DDoS detection cannot be
straightforwardly implemented in P4. This work overcomes such a limitation and
presents two novel strategies for flow cardinality and for normalized network
traffic entropy estimation that only use P4-supported operations and guarantee
a low relative error. Additionally, based on these contributions, we propose a
DDoS detection strategy relying on variations of the normalized network traffic
entropy. Results show that it has comparable or higher detection accuracy than
state-of-the-art solutions, yet being simpler and entirely executed in the data
plane.

    

### [[2105.13600] Placement Optimization and Power Control in Intelligent Reflecting Surface Aided Multiuser System](http://arxiv.org/abs/2105.13600)


  Intelligent reflecting surface (IRS) is a new and revolutionary technology
capable of reconfiguring the wireless propagation environment by controlling
its massive low-cost passive reflecting elements. Different from prior works
that focus on optimizing IRS reflection coefficients or single-IRS placement,
we aim to maximize the minimum throughput of a single-cell multiuser system
aided by multiple IRSs, by joint multi-IRS placement and power control at the
access point (AP), which is a mixed-integer non-convex problem with drastically
increased complexity with the number of IRSs/users. To tackle this challenge, a
ring-based IRS placement scheme is proposed along with a power control policy
that equalizes the users' non-outage probability. An efficient searching
algorithm is further proposed to obtain a close-to-optimal solution for
arbitrary number of IRSs/rings. Numerical results validate our analysis and
show that our proposed scheme significantly outperforms the benchmark schemes
without IRS and/or with other power control policies. Moreover, it is shown
that the IRSs are preferably deployed near AP for coverage range extension,
while with more IRSs, they tend to spread out over the cell to cover more and
get closer to target users.

    

### [[2106.00845] Energy-aware optimization of UAV base stations placement via decentralized multi-agent Q-learning](http://arxiv.org/abs/2106.00845)


  Unmanned aerial vehicles serving as aerial base stations (UAV-BSs) can be
deployed to provide wireless connectivity to ground devices in events of
increased network demand, points-of-failure in existing infrastructure, or
disasters. However, it is challenging to conserve the energy of UAVs during
prolonged coverage tasks, considering their limited on-board battery capacity.
Reinforcement learning-based (RL) approaches have been previously used to
improve energy utilization of multiple UAVs, however, a central cloud
controller is assumed to have complete knowledge of the end-devices' locations,
i.e., the controller periodically scans and sends updates for UAV
decision-making. This assumption is impractical in dynamic network environments
with UAVs serving mobile ground devices. To address this problem, we propose a
decentralized Q-learning approach, where each UAV-BS is equipped with an
autonomous agent that maximizes the connectivity of mobile ground devices while
improving its energy utilization. Experimental results show that the proposed
design significantly outperforms the centralized approaches in jointly
maximizing the number of connected ground devices and the energy utilization of
the UAV-BSs.

    

### [[2106.04228] Decentralized Learning in Online Queuing Systems](http://arxiv.org/abs/2106.04228)


  Motivated by packet routing in computer networks, online queuing systems are
composed of queues receiving packets at different rates. Repeatedly, they send
packets to servers, each of them treating only at most one packet at a time. In
the centralized case, the number of accumulated packets remains bounded (i.e.,
the system is \textit{stable}) as long as the ratio between service rates and
arrival rates is larger than $1$. In the decentralized case, individual
no-regret strategies ensures stability when this ratio is larger than $2$. Yet,
myopically minimizing regret disregards the long term effects due to the
carryover of packets to further rounds. On the other hand, minimizing long term
costs leads to stable Nash equilibria as soon as the ratio exceeds
$\frac{e}{e-1}$. Stability with decentralized learning strategies with a ratio
below $2$ was a major remaining question. We first argue that for ratios up to
$2$, cooperation is required for stability of learning strategies, as selfish
minimization of policy regret, a \textit{patient} notion of regret, might
indeed still be unstable in this case. We therefore consider cooperative queues
and propose the first learning decentralized algorithm guaranteeing stability
of the system as long as the ratio of rates is larger than $1$, thus reaching
performances comparable to centralized strategies.

    

### [[2106.09261] Federated Learning Framework with Straggling Mitigation and Privacy-Awareness for AI-based Mobile Application Services](http://arxiv.org/abs/2106.09261)


  In this work, we propose a novel framework to address straggling and privacy
issues for federated learning (FL)-based mobile application services, taking
into account limited computing/communications resources at mobile users
(MUs)/mobile application provider (MAP), privacy cost, the rationality and
incentive competition among MUs in contributing data to the MAP. Particularly,
the MAP first determines a set of the best MUs for the FL process based on the
MUs' provided information/features. To mitigate straggling problems with
privacy-awareness, each selected MU can then encrypt part of local data and
upload the encrypted data to the MAP for an encrypted training process, in
addition to the local training process. For that, each selected MU can propose
a contract to the MAP according to its expected trainable local data and
privacy-protected encrypted data. To find the optimal contracts that can
maximize utilities of the MAP and all the participating MUs while maintaining
high learning quality of the whole system, we first develop a multi-principal
one-agent contract-based problem leveraging FL-based multiple utility
functions. These utility functions account for the MUs' privacy cost, the MAP's
limited computing resources, and asymmetric information between the MAP and
MUs. Then, we transform the problem into an equivalent low-complexity problem
and develop a light-weight iterative algorithm to effectively find the optimal
solutions. Experiments with a real-world dataset show that our framework can
speed up training time up to 49% and improve prediction accuracy up to 4.6
times while enhancing the network's social welfare, i.e., total utility of all
participating entities, up to 114% under the privacy cost consideration
compared with those of baseline methods.

    

### [[2111.02398] Transparency of Deep Neural Networks for Medical Image Analysis: A Review of Interpretability Methods](http://arxiv.org/abs/2111.02398)


  Artificial Intelligence has emerged as a useful aid in numerous clinical
applications for diagnosis and treatment decisions. Deep neural networks have
shown same or better performance than clinicians in many tasks owing to the
rapid increase in the available data and computational power. In order to
conform to the principles of trustworthy AI, it is essential that the AI system
be transparent, robust, fair and ensure accountability. Current deep neural
solutions are referred to as black-boxes due to a lack of understanding of the
specifics concerning the decision making process. Therefore, there is a need to
ensure interpretability of deep neural networks before they can be incorporated
in the routine clinical workflow. In this narrative review, we utilized
systematic keyword searches and domain expertise to identify nine different
types of interpretability methods that have been used for understanding deep
learning models for medical image analysis applications based on the type of
generated explanations and technical similarities. Furthermore, we report the
progress made towards evaluating the explanations produced by various
interpretability methods. Finally we discuss limitations, provide guidelines
for using interpretability methods and future directions concerning the
interpretability of deep neural networks for medical imaging analysis.

    

### [[2111.02399] Learning Pruned Structure and Weights Simultaneously from Scratch: an Attention based Approach](http://arxiv.org/abs/2111.02399)


  As a deep learning model typically contains millions of trainable weights,
there has been a growing demand for a more efficient network structure with
reduced storage space and improved run-time efficiency. Pruning is one of the
most popular network compression techniques. In this paper, we propose a novel
unstructured pruning pipeline, Attention-based Simultaneous sparse structure
and Weight Learning (ASWL). Unlike traditional channel-wise or weight-wise
attention mechanism, ASWL proposed an efficient algorithm to calculate the
pruning ratio through layer-wise attention for each layer, and both weights for
the dense network and the sparse network are tracked so that the pruned
structure is simultaneously learned from randomly initialized weights. Our
experiments on MNIST, Cifar10, and ImageNet show that ASWL achieves superior
pruning results in terms of accuracy, pruning ratio and operating efficiency
when compared with state-of-the-art network pruning methods.

    

### [[2111.02400] Deep AUC Maximization for Medical Image Classification: Challenges and Opportunities](http://arxiv.org/abs/2111.02400)


  In this extended abstract, we will present and discuss opportunities and
challenges brought about by a new deep learning method by AUC maximization (aka
\underline{\bf D}eep \underline{\bf A}UC \underline{\bf M}aximization or {\bf
DAM}) for medical image classification. Since AUC (aka area under ROC curve) is
a standard performance measure for medical image classification, hence directly
optimizing AUC could achieve a better performance for learning a deep neural
network than minimizing a traditional loss function (e.g., cross-entropy loss).
Recently, there emerges a trend of using deep AUC maximization for large-scale
medical image classification. In this paper, we will discuss these recent
results by highlighting (i) the advancements brought by stochastic non-convex
optimization algorithms for DAM; (ii) the promising results on various medical
image classification problems. Then, we will discuss challenges and
opportunities of DAM for medical image classification from three perspectives,
feature learning, large-scale optimization, and learning trustworthy AI models.

    

### [[2111.02402] Skin Cancer Classification using Inception Network and Transfer Learning](http://arxiv.org/abs/2111.02402)


  Medical data classification is typically a challenging task due to imbalance
between classes. In this paper, we propose an approach to classify
dermatoscopic images from HAM10000 (Human Against Machine with 10000 training
images) dataset, consisting of seven imbalanced types of skin lesions, with
good precision and low resources requirements. Classification is done by using
a pretrained convolutional neural network. We evaluate the accuracy and
performance of the proposal and illustrate possible extensions.

    

### [[2111.02405] Unsupervised embedding and similarity detection of microregions using public transport schedules](http://arxiv.org/abs/2111.02405)


  The role of spatial data in tackling city-related tasks has been growing in
recent years. To use them in machine learning models, it is often necessary to
transform them into a vector representation, which has led to the development
in the field of spatial data representation learning. There is also a growing
variety of spatial data types for which representation learning methods are
proposed. Public transport timetables have so far not been used in the task of
learning representations of regions in a city. In this work, a method is
developed to embed public transport availability information into vector space.
To conduct experiments on its application, public transport timetables were
collected from 48 European cities. Using the H3 spatial indexing method, they
were divided into micro-regions. A method was also proposed to identify regions
with similar characteristics of public transport offers. On its basis, a
multi-level typology of public transport offers in the regions was defined.
This thesis shows that the proposed representation method makes it possible to
identify micro-regions with similar public transport characteristics between
the cities, and can be used to evaluate the quality of public transport
available in a city.

    

### [[2111.02408] Partial supervision for the FeTA challenge 2021](http://arxiv.org/abs/2111.02408)


  This paper describes our method for our participation in the FeTA
challenge2021 (team name: TRABIT). The performance of convolutional neural
networks for medical image segmentation is thought to correlate positively with
the number of training data. The FeTA challenge does not restrict participants
to using only the provided training data but also allows for using other
publicly available sources. Yet, open access fetal brain data remains limited.
An advantageous strategy could thus be to expand the training data to cover
broader perinatal brain imaging sources. Perinatal brain MRIs, other than the
FeTA challenge data, that are currently publicly available, span normal and
pathological fetal atlases as well as neonatal scans. However, perinatal brain
MRIs segmented in different datasets typically come with different annotation
protocols. This makes it challenging to combine those datasets to train a deep
neural network. We recently proposed a family of loss functions, the label-set
loss functions, for partially supervised learning. Label-set loss functions
allow to train deep neural networks with partially segmented images, i.e.
segmentations in which some classes may be grouped into super-classes. We
propose to use label-set loss functions to improve the segmentation performance
of a state-of-the-art deep learning pipeline for multi-class fetal brain
segmentation by merging several publicly available datasets. To promote
generalisability, our approach does not introduce any additional
hyper-parameters tuning.

    

### [[2111.02426] Weighted Quantum Channel Compiling through Proximal Policy Optimization](http://arxiv.org/abs/2111.02426)


  We propose a general and systematic strategy to compile arbitrary quantum
channels without using ancillary qubits, based on proximal policy optimization
-- a powerful deep reinforcement learning algorithm. We rigorously prove that,
in sharp contrast to the case of compiling unitary gates, it is impossible to
compile an arbitrary channel to arbitrary precision with any given finite
elementary channel set, regardless of the length of the decomposition sequence.
However, for a fixed accuracy $\epsilon$ one can construct a universal set with
constant number of $\epsilon$-dependent elementary channels, such that an
arbitrary quantum channel can be decomposed into a sequence of these elementary
channels followed by a unitary gate, with the sequence length bounded by
$O(\frac{1}{\epsilon}\log\frac{1}{\epsilon})$. Through a concrete example
concerning topological compiling of Majorana fermions, we show that our
proposed algorithm can conveniently and effectively reduce the use of expensive
elementary gates through adding the weighted cost into the reward function of
the proximal policy optimization.

    

### [[2111.02434] Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling](http://arxiv.org/abs/2111.02434)


  Sampling from an unnormalized probability distribution is a fundamental
problem in machine learning with applications including Bayesian modeling,
latent factor inference, and energy-based model training. After decades of
research, variations of MCMC remain the default approach to sampling despite
slow convergence. Auxiliary neural models can learn to speed up MCMC, but the
overhead for training the extra model can be prohibitive. We propose a
fundamentally different approach to this problem via a new Hamiltonian dynamics
with a non-Newtonian momentum. In contrast to MCMC approaches like Hamiltonian
Monte Carlo, no stochastic step is required. Instead, the proposed
deterministic dynamics in an extended state space exactly sample the target
distribution, specified by an energy function, under an assumption of
ergodicity. Alternatively, the dynamics can be interpreted as a normalizing
flow that samples a specified energy model without training. The proposed
Energy Sampling Hamiltonian (ESH) dynamics have a simple form that can be
solved with existing ODE solvers, but we derive a specialized solver that
exhibits much better performance. ESH dynamics converge faster than their MCMC
competitors enabling faster, more stable training of neural network energy
models.

    

### [[2111.02445] Autonomous Attack Mitigation for Industrial Control Systems](http://arxiv.org/abs/2111.02445)


  Defending computer networks from cyber attack requires timely responses to
alerts and threat intelligence. Decisions about how to respond involve
coordinating actions across multiple nodes based on imperfect indicators of
compromise while minimizing disruptions to network operations. Currently,
playbooks are used to automate portions of a response process, but often leave
complex decision-making to a human analyst. In this work, we present a deep
reinforcement learning approach to autonomous response and recovery in large
industrial control networks. We propose an attention-based neural architecture
that is flexible to the size of the network under protection. To train and
evaluate the autonomous defender agent, we present an industrial control
network simulation environment suitable for reinforcement learning. Experiments
show that the learned agent can effectively mitigate advanced attacks that
progress with few observable signals over several months before execution. The
proposed deep reinforcement learning approach outperforms a fully automated
playbook method in simulation, taking less disruptive actions while also
defending more nodes on the network. The learned policy is also more robust to
changes in attacker behavior than playbook approaches.

    

### [[2111.02458] Perturb-and-max-product: Sampling and learning in discrete energy-based models](http://arxiv.org/abs/2111.02458)


  Perturb-and-MAP offers an elegant approach to approximately sample from a
energy-based model (EBM) by computing the maximum-a-posteriori (MAP)
configuration of a perturbed version of the model. Sampling in turn enables
learning. However, this line of research has been hindered by the general
intractability of the MAP computation. Very few works venture outside tractable
models, and when they do, they use linear programming approaches, which as we
will show, have several limitations. In this work we present
perturb-and-max-product (PMP), a parallel and scalable mechanism for sampling
and learning in discrete EBMs. Models can be arbitrary as long as they are
built using tractable factors. We show that (a) for Ising models, PMP is orders
of magnitude faster than Gibbs and Gibbs-with-Gradients (GWG) at learning and
generating samples of similar or better quality; (b) PMP is able to learn and
sample from RBMs; (c) in a large, entangled graphical model in which Gibbs and
GWG fail to mix, PMP succeeds.

    

### [[2111.02461] Automatic ultrasound vessel segmentation with deep spatiotemporal context learning](http://arxiv.org/abs/2111.02461)


  Accurate, real-time segmentation of vessel structures in ultrasound image
sequences can aid in the measurement of lumen diameters and assessment of
vascular diseases. This, however, remains a challenging task, particularly for
extremely small vessels that are difficult to visualize. We propose to leverage
the rich spatiotemporal context available in ultrasound to improve segmentation
of small-scale lower-extremity arterial vasculature. We describe efficient deep
learning methods that incorporate temporal, spatial, and feature-aware
contextual embeddings at multiple resolution scales while jointly utilizing
information from B-mode and Color Doppler signals. Evaluating on femoral and
tibial artery scans performed on healthy subjects by an expert
ultrasonographer, and comparing to consensus expert ground-truth annotations of
inner lumen boundaries, we demonstrate real-time segmentation using the
context-aware models and show that they significantly outperform comparable
baseline approaches.

    

### [[2111.02484] Accelerated replica exchange stochastic gradient Langevin diffusion enhanced Bayesian DeepONet for solving noisy parametric PDEs](http://arxiv.org/abs/2111.02484)


  The Deep Operator Networks~(DeepONet) is a fundamentally different class of
neural networks that we train to approximate nonlinear operators, including the
solution operator of parametric partial differential equations (PDE). DeepONets
have shown remarkable approximation and generalization capabilities even when
trained with relatively small datasets. However, the performance of DeepONets
deteriorates when the training data is polluted with noise, a scenario that
occurs very often in practice. To enable DeepONets training with noisy data, we
propose using the Bayesian framework of replica-exchange Langevin diffusion.
Such a framework uses two particles, one for exploring and another for
exploiting the loss function landscape of DeepONets. We show that the proposed
framework's exploration and exploitation capabilities enable (1) improved
training convergence for DeepONets in noisy scenarios and (2) attaching an
uncertainty estimate for the predicted solutions of parametric PDEs. In
addition, we show that replica-exchange Langeving Diffusion (remarkably) also
improves the DeepONet's mean prediction accuracy in noisy scenarios compared
with vanilla DeepONets trained with state-of-the-art gradient-based
optimization algorithms (e.g. Adam). To reduce the potentially high
computational cost of replica, in this work, we propose an accelerated training
framework for replica-exchange Langevin diffusion that exploits the neural
network architecture of DeepONets to reduce its computational cost up to 25%
without compromising the proposed framework's performance. Finally, we
illustrate the effectiveness of the proposed Bayesian framework using a series
of experiments on four parametric PDE problems.

    

### [[2111.02489] Communication-Efficient Separable Neural Network for Distributed Inference on Edge Devices](http://arxiv.org/abs/2111.02489)


  The inference of Neural Networks is usually restricted by the resources
(e.g., computing power, memory, bandwidth) on edge devices. In addition to
improving the hardware design and deploying efficient models, it is possible to
aggregate the computing power of many devices to enable the machine learning
models. In this paper, we proposed a novel method of exploiting model
parallelism to separate a neural network for distributed inferences. To achieve
a better balance between communication latency, computation latency, and
performance, we adopt neural architecture search (NAS) to search for the best
transmission policy and reduce the amount of communication. The best model we
found decreases by 86.6% of the amount of data transmission compared to the
baseline and does not impact performance much. Under proper specifications of
devices and configurations of models, our experiments show that the inference
of large neural networks on edge clusters can be distributed and accelerated,
which provides a new solution for the deployment of intelligent applications in
the internet of things (IoT).

    

### [[2111.02500] Improving Pose Estimation through Contextual Activity Fusion](http://arxiv.org/abs/2111.02500)


  This research presents the idea of activity fusion into existing Pose
Estimation architectures to enhance their predictive ability. This is motivated
by the rise in higher level concepts found in modern machine learning
architectures, and the belief that activity context is a useful piece of
information for the problem of pose estimation. To analyse this concept we take
an existing deep learning architecture and augment it with an additional 1x1
convolution to fuse activity information into the model. We perform evaluation
and comparison on a common pose estimation dataset, and show a performance
improvement over our baseline model, especially in uncommon poses and on
typically difficult joints. Additionally, we perform an ablative analysis to
indicate that the performance improvement does in fact draw from the activity
information.

    

### [[2111.02508] AlphaD3M: Machine Learning Pipeline Synthesis](http://arxiv.org/abs/2111.02508)


  We introduce AlphaD3M, an automatic machine learning (AutoML) system based on
meta reinforcement learning using sequence models with self play. AlphaD3M is
based on edit operations performed over machine learning pipeline primitives
providing explainability. We compare AlphaD3M with state-of-the-art AutoML
systems: Autosklearn, Autostacker, and TPOT, on OpenML datasets. AlphaD3M
achieves competitive performance while being an order of magnitude faster,
reducing computation time from hours to minutes, and is explainable by design.

    

### [[2111.02513] Evaluation of Tree Based Regression over Multiple Linear Regression for Non-normally Distributed Data in Battery Performance](http://arxiv.org/abs/2111.02513)


  Battery performance datasets are typically non-normal and multicollinear.
Extrapolating such datasets for model predictions needs attention to such
characteristics. This study explores the impact of data normality in building
machine learning models. In this work, tree-based regression models and
multiple linear regressions models are each built from a highly skewed
non-normal dataset with multicollinearity and compared. Several techniques are
necessary, such as data transformation, to achieve a good multiple linear
regression model with this dataset; the most useful techniques are discussed.
With these techniques, the best multiple linear regression model achieved an
R^2 = 81.23% and exhibited no multicollinearity effect for the dataset used in
this study. Tree-based models perform better on this dataset, as they are
non-parametric, capable of handling complex relationships among variables and
not affected by multicollinearity. We show that bagging, in the use of Random
Forests, reduces overfitting. Our best tree-based model achieved accuracy of
R^2 = 97.73%. This study explains why tree-based regressions promise as a
machine learning model for non-normally distributed, multicollinear data.

    

### [[2111.02520] Resampling and super-resolution of hexagonally sampled images using deep learning](http://arxiv.org/abs/2111.02520)


  Super-resolution (SR) aims to increase the resolution of imagery.
Applications include security, medical imaging, and object recognition. We
propose a deep learning-based SR system that takes a hexagonally sampled
low-resolution image as an input and generates a rectangularly sampled SR image
as an output. For training and testing, we use a realistic observation model
that includes optical degradation from diffraction and sensor degradation from
detector integration. Our SR approach first uses non-uniform interpolation to
partially upsample the observed hexagonal imagery and convert it to a
rectangular grid. We then leverage a state-of-the-art convolutional neural
network (CNN) architecture designed for SR known as Residual Channel Attention
Network (RCAN). In particular, we use RCAN to further upsample and restore the
imagery to produce the final SR image estimate. We demonstrate that this system
is superior to applying RCAN directly to rectangularly sampled LR imagery with
equivalent sample density. The theoretical advantages of hexagonal sampling are
well known. However, to the best of our knowledge, the practical benefit of
hexagonal sampling in light of modern processing techniques such as RCAN SR is
heretofore untested. Our SR system demonstrates a notable advantage of
hexagonally sampled imagery when employing a modified RCAN for hexagonal SR.

    

### [[2111.02529] Shift Happens: Adjusting Classifiers](http://arxiv.org/abs/2111.02529)


  Minimizing expected loss measured by a proper scoring rule, such as Brier
score or log-loss (cross-entropy), is a common objective while training a
probabilistic classifier. If the data have experienced dataset shift where the
class distributions change post-training, then often the model's performance
will decrease, over-estimating the probabilities of some classes while
under-estimating the others on average. We propose unbounded and bounded
general adjustment (UGA and BGA) methods that transform all predictions to
(re-)equalize the average prediction and the class distribution. These methods
act differently depending on which proper scoring rule is to be minimized, and
we have a theoretical guarantee of reducing loss on test data, if the exact
class distribution is known. We also demonstrate experimentally that, when in
practice the class distribution is known only approximately, there is often
still a reduction in loss depending on the amount of shift and the precision to
which the class distribution is known.

    

### [[2111.02545] Multi-task Learning of Order-Consistent Causal Graphs](http://arxiv.org/abs/2111.02545)


  We consider the problem of discovering $K$ related Gaussian directed acyclic
graphs (DAGs), where the involved graph structures share a consistent causal
order and sparse unions of supports. Under the multi-task learning setting, we
propose a $l_1/l_2$-regularized maximum likelihood estimator (MLE) for learning
$K$ linear structural equation models. We theoretically show that the joint
estimator, by leveraging data across related tasks, can achieve a better sample
complexity for recovering the causal order (or topological order) than separate
estimations. Moreover, the joint estimator is able to recover non-identifiable
DAGs, by estimating them together with some identifiable DAGs. Lastly, our
analysis also shows the consistency of union support recovery of the
structures. To allow practical implementation, we design a continuous
optimization problem whose optimizer is the same as the joint estimator and can
be approximated efficiently by an iterative algorithm. We validate the
theoretical analysis and the effectiveness of the joint estimator in
experiments.

    

### [[2111.02552] Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies](http://arxiv.org/abs/2111.02552)


  Reinforcement learning (RL) for continuous control typically employs
distributions whose support covers the entire action space. In this work, we
investigate the colloquially known phenomenon that trained agents often prefer
actions at the boundaries of that space. We draw theoretical connections to the
emergence of bang-bang behavior in optimal control, and provide extensive
empirical evaluation across a variety of recent RL algorithms. We replace the
normal Gaussian by a Bernoulli distribution that solely considers the extremes
along each action dimension - a bang-bang controller. Surprisingly, this
achieves state-of-the-art performance on several continuous control benchmarks
- in contrast to robotic hardware, where energy and maintenance cost affect
controller choices. Since exploration, learning,and the final solution are
entangled in RL, we provide additional imitation learning experiments to reduce
the impact of exploration on our analysis. Finally, we show that our
observations generalize to environments that aim to model real-world challenges
and evaluate factors to mitigate the emergence of bang-bang solutions. Our
findings emphasize challenges for benchmarking continuous control algorithms,
particularly in light of potential real-world applications.

    

### [[2111.02557] A Meta-Learned Neuron model for Continual Learning](http://arxiv.org/abs/2111.02557)


  Continual learning is the ability to acquire new knowledge without forgetting
the previously learned one, assuming no further access to past training data.
Neural network approximators trained with gradient descent are known to fail in
this setting as they must learn from a stream of data-points sampled from a
stationary distribution to converge. In this work, we replace the standard
neuron by a meta-learned neuron model whom inference and update rules are
optimized to minimize catastrophic interference. Our approach can memorize
dataset-length sequences of training samples, and its learning capabilities
generalize to any domain. Unlike previous continual learning methods, our
method does not make any assumption about how tasks are constructed, delivered
and how they relate to each other: it simply absorbs and retains training
samples one by one, whether the stream of input data is time-correlated or not.

    

### [[2111.02569] RT-RCG: Neural Network and Accelerator Search Towards Effective and Real-time ECG Reconstruction from Intracardiac Electrograms](http://arxiv.org/abs/2111.02569)


  There exists a gap in terms of the signals provided by pacemakers (i.e.,
intracardiac electrogram (EGM)) and the signals doctors use (i.e., 12-lead
electrocardiogram (ECG)) to diagnose abnormal rhythms. Therefore, the former,
even if remotely transmitted, are not sufficient for doctors to provide a
precise diagnosis, let alone make a timely intervention. To close this gap and
make a heuristic step towards real-time critical intervention in instant
response to irregular and infrequent ventricular rhythms, we propose a new
framework dubbed RT-RCG to automatically search for (1) efficient Deep Neural
Network (DNN) structures and then (2)corresponding accelerators, to enable
Real-Time and high-quality Reconstruction of ECG signals from EGM signals.
Specifically, RT-RCG proposes a new DNN search space tailored for ECG
reconstruction from EGM signals, and incorporates a differentiable acceleration
search (DAS) engine to efficiently navigate over the large and discrete
accelerator design space to generate optimized accelerators. Extensive
experiments and ablation studies under various settings consistently validate
the effectiveness of our RT-RCG. To the best of our knowledge, RT-RCG is the
first to leverage neural architecture search (NAS) to simultaneously tackle
both reconstruction efficacy and efficiency.

    

### [[2111.02570] CLUES: Few-Shot Learning Evaluation in Natural Language Understanding](http://arxiv.org/abs/2111.02570)


  Most recent progress in natural language understanding (NLU) has been driven,
in part, by benchmarks such as GLUE, SuperGLUE, SQuAD, etc. In fact, many NLU
models have now matched or exceeded "human-level" performance on many tasks in
these benchmarks. Most of these benchmarks, however, give models access to
relatively large amounts of labeled data for training. As such, the models are
provided far more data than required by humans to achieve strong performance.
That has motivated a line of work that focuses on improving few-shot learning
performance of NLU models. However, there is a lack of standardized evaluation
benchmarks for few-shot NLU resulting in different experimental settings in
different papers. To help accelerate this line of work, we introduce CLUES
(Constrained Language Understanding Evaluation Standard), a benchmark for
evaluating the few-shot learning capabilities of NLU models. We demonstrate
that while recent models reach human performance when they have access to large
amounts of labeled data, there is a huge gap in performance in the few-shot
setting for most tasks. We also demonstrate differences between alternative
model families and adaptation techniques in the few shot setting. Finally, we
discuss several principles and choices in designing the experimental settings
for evaluating the true few-shot learning performance and suggest a unified
standardized approach to few-shot learning evaluation. We aim to encourage
research on NLU models that can generalize to new tasks with a small number of
examples. Code and data for CLUES are available at
this https URL.

    

### [[2111.02571] Learning suction graspability considering grasp quality and robot reachability for bin-picking](http://arxiv.org/abs/2111.02571)


  Deep learning has been widely used for inferring robust grasps. Although
human-labeled RGB-D datasets were initially used to learn grasp configurations,
preparation of this kind of large dataset is expensive. To address this
problem, images were generated by a physical simulator, and a physically
inspired model (e.g., a contact model between a suction vacuum cup and object)
was used as a grasp quality evaluation metric to annotate the synthesized
images. However, this kind of contact model is complicated and requires
parameter identification by experiments to ensure real world performance. In
addition, previous studies have not considered manipulator reachability such as
when a grasp configuration with high grasp quality is unable to reach the
target due to collisions or the physical limitations of the robot. In this
study, we propose an intuitive geometric analytic-based grasp quality
evaluation metric. We further incorporate a reachability evaluation metric. We
annotate the pixel-wise grasp quality and reachability by the proposed
evaluation metric on synthesized images in a simulator to train an
auto-encoder--decoder called suction graspability U-Net++ (SG-U-Net++).
Experiment results show that our intuitive grasp quality evaluation metric is
competitive with a physically-inspired metric. Learning the reachability helps
to reduce motion planning computation time by removing obviously unreachable
candidates. The system achieves an overall picking speed of 560 PPH (pieces per
hour).

    

### [[2111.02574] Contextual Semantic Parsing for Multilingual Task-Oriented Dialogues](http://arxiv.org/abs/2111.02574)


  Robust state tracking for task-oriented dialogue systems currently remains
restricted to a few popular languages. This paper shows that given a
large-scale dialogue data set in one language, we can automatically produce an
effective semantic parser for other languages using machine translation. We
propose automatic translation of dialogue datasets with alignment to ensure
faithful translation of slot values and eliminate costly human supervision used
in previous benchmarks. We also propose a new contextual semantic parsing
model, which encodes the formal slots and values, and only the last agent and
user utterances. We show that the succinct representation reduces the
compounding effect of translation errors, without harming the accuracy in
practice.
We evaluate our approach on several dialogue state tracking benchmarks. On
RiSAWOZ, CrossWOZ, CrossWOZ-EN, and MultiWOZ-ZH datasets we improve the state
of the art by 11%, 17%, 20%, and 0.3% in joint goal accuracy. We present a
comprehensive error analysis for all three datasets showing erroneous
annotations can obscure judgments on the quality of the model.
Finally, we present RiSAWOZ English and German datasets, created using our
translation methodology. On these datasets, accuracy is within 11% of the
original showing that high-accuracy multilingual dialogue datasets are possible
without relying on expensive human annotations.

    

### [[2111.02584] Real-time Wireless Transmitter Authorization: Adapting to Dynamic Authorized Sets with Information Retrieval](http://arxiv.org/abs/2111.02584)


  As the Internet of Things (IoT) continues to grow, ensuring the security of
systems that rely on wireless IoT devices has become critically important. Deep
learning-based passive physical layer transmitter authorization systems have
been introduced recently for this purpose, as they accommodate the limited
computational and power budget of such devices. These systems have been shown
to offer excellent outlier detection accuracies when trained and tested on a
fixed authorized transmitter set. However in a real-life deployment, a need may
arise for transmitters to be added and removed as the authorized set of
transmitters changes. In such cases, the system could experience long
down-times, as retraining the underlying deep learning model is often a
time-consuming process. In this paper, we draw inspiration from information
retrieval to address this problem: by utilizing feature vectors as RF
fingerprints, we first demonstrate that training could be simplified to
indexing those feature vectors into a database using locality sensitive hashing
(LSH). Then we show that approximate nearest neighbor search could be performed
on the database to perform transmitter authorization that matches the accuracy
of deep learning models, while allowing for more than 100x faster retraining.
Furthermore, dimensionality reduction techniques are used on the feature
vectors to show that the authorization latency of our technique could be
reduced to approach that of traditional deep learning-based systems.

    

### [[2111.02585] InQSS: a speech intelligibility assessment model using a multi-task learning network](http://arxiv.org/abs/2111.02585)


  Speech intelligibility assessment models are essential tools for researchers
to evaluate and improve speech processing models. In this study, we propose
InQSS, a speech intelligibility assessment model that uses both spectrogram and
scattering coefficients as input features. In addition, InQSS uses a multi-task
learning network in which quality scores can guide the training of the speech
intelligibility assessment. The resulting model can predict not only the
intelligibility scores but also the quality scores of a speech. The
experimental results confirm that the scattering coefficients and quality
scores are informative for intelligibility. Moreover, we released TMHINT-QI,
which is a Chinese speech dataset that records the quality and intelligibility
scores of clean, noisy, and enhanced speech.

    

### [[2111.02586] Building Damage Mapping with Self-PositiveUnlabeled Learning](http://arxiv.org/abs/2111.02586)


  Humanitarian organizations must have fast and reliable data to respond to
disasters. Deep learning approaches are difficult to implement in real-world
disasters because it might be challenging to collect ground truth data of the
damage situation (training data) soon after the event. The implementation of
recent self-paced positive-unlabeled learning (PU) is demonstrated in this work
by successfully applying to building damage assessment with very limited
labeled data and a large amount of unlabeled data. Self-PU learning is compared
with the supervised baselines and traditional PU learning using different
datasets collected from the 2011 Tohoku earthquake, the 2018 Palu tsunami, and
the 2018 Hurricane Michael. By utilizing only a portion of labeled damaged
samples, we show how models trained with self-PU techniques may achieve
comparable performance as supervised learning.

    

### [[2111.02592] Conformal prediction for text infilling and part-of-speech prediction](http://arxiv.org/abs/2111.02592)


  Modern machine learning algorithms are capable of providing remarkably
accurate point-predictions; however, questions remain about their statistical
reliability. Unlike conventional machine learning methods, conformal prediction
algorithms return confidence sets (i.e., set-valued predictions) that
correspond to a given significance level. Moreover, these confidence sets are
valid in the sense that they guarantee finite sample control over type 1 error
probabilities, allowing the practitioner to choose an acceptable error rate. In
our paper, we propose inductive conformal prediction (ICP) algorithms for the
tasks of text infilling and part-of-speech (POS) prediction for natural
language data. We construct new conformal prediction-enhanced bidirectional
encoder representations from transformers (BERT) and bidirectional long
short-term memory (BiLSTM) algorithms for POS tagging and a new conformal
prediction-enhanced BERT algorithm for text infilling. We analyze the
performance of the algorithms in simulations using the Brown Corpus, which
contains over 57,000 sentences. Our results demonstrate that the ICP algorithms
are able to produce valid set-valued predictions that are small enough to be
applicable in real-world applications. We also provide a real data example for
how our proposed set-valued predictions can improve machine generated audio
transcriptions.

    

### [[2111.02595] An Information-Theoretic Framework for Identifying Age-Related Genes Using Human Dermal Fibroblast Transcriptome Data](http://arxiv.org/abs/2111.02595)


  Investigation of age-related genes is of great importance for multiple
purposes, for instance, improving our understanding of the mechanism of ageing,
increasing life expectancy, age prediction, and other healthcare applications.
In his work, starting with a set of 27,142 genes, we develop an
information-theoretic framework for identifying genes that are associated with
aging by applying unsupervised and semi-supervised learning techniques on human
dermal fibroblast gene expression data. First, we use unsupervised learning and
apply information-theoretic measures to identify key features for effective
representation of gene expression values in the transcriptome data. Using the
identified features, we perform clustering on the data. Finally, we apply
semi-supervised learning on the clusters using different distance measures to
identify novel genes that are potentially associated with aging. Performance
assessment for both unsupervised and semi-supervised methods show the
effectiveness of the framework.

    

### [[2111.02599] Leveraging Time Irreversibility with Order-Contrastive Pre-training](http://arxiv.org/abs/2111.02599)


  Label-scarce, high-dimensional domains such as healthcare present a challenge
for modern machine learning techniques. To overcome the difficulties posed by a
lack of labeled data, we explore an "order-contrastive" method for
self-supervised pre-training on longitudinal data. We sample pairs of time
segments, switch the order for half of them, and train a model to predict
whether a given pair is in the correct order. Intuitively, the ordering task
allows the model to attend to the least time-reversible features (for example,
features that indicate progression of a chronic disease). The same features are
often useful for downstream tasks of interest. To quantify this, we study a
simple theoretical setting where we prove a finite-sample guarantee for the
downstream error of a representation learned with order-contrastive
pre-training. Empirically, in synthetic and longitudinal healthcare settings,
we demonstrate the effectiveness of order-contrastive pre-training in the
small-data regime over supervised learning and other self-supervised
pre-training baselines. Our results indicate that pre-training methods designed
for particular classes of distributions and downstream tasks can improve the
performance of self-supervised learning.

    

### [[2111.02601] Optimal Recovery from Inaccurate Data in Hilbert Spaces: Regularize, but what of the Parameter?](http://arxiv.org/abs/2111.02601)


  In Optimal Recovery, the task of learning a function from observational data
is tackled deterministically by adopting a worst-case perspective tied to an
explicit model assumption made on the functions to be learned. Working in the
framework of Hilbert spaces, this article considers a model assumption based on
approximability. It also incorporates observational inaccuracies modeled via
additive errors bounded in $\ell_2$. Earlier works have demonstrated that
regularization provide algorithms that are optimal in this situation, but did
not fully identify the desired hyperparameter. This article fills the gap in
both a local scenario and a global scenario. In the local scenario, which
amounts to the determination of Chebyshev centers, the semidefinite recipe of
Beck and Eldar (legitimately valid in the complex setting only) is complemented
by a more direct approach, with the proviso that the observational functionals
have orthonormal representers. In the said approach, the desired parameter is
the solution to an equation that can be resolved via standard methods. In the
global scenario, where linear algorithms rule, the parameter elusive in the
works of Micchelli et al. is found as the byproduct of a semidefinite program.
Additionally and quite surprisingly, in case of observational functionals with
orthonormal representers, it is established that any regularization parameter
is optimal.

    

### [[2111.02625] Qimera: Data-free Quantization with Synthetic Boundary Supporting Samples](http://arxiv.org/abs/2111.02625)


  Model quantization is known as a promising method to compress deep neural
networks, especially for inferences on lightweight mobile or edge devices.
However, model quantization usually requires access to the original training
data to maintain the accuracy of the full-precision models, which is often
infeasible in real-world scenarios for security and privacy issues. A popular
approach to perform quantization without access to the original data is to use
synthetically generated samples, based on batch-normalization statistics or
adversarial learning. However, the drawback of such approaches is that they
primarily rely on random noise input to the generator to attain diversity of
the synthetic samples. We find that this is often insufficient to capture the
distribution of the original data, especially around the decision boundaries.
To this end, we propose Qimera, a method that uses superposed latent embeddings
to generate synthetic boundary supporting samples. For the superposed
embeddings to better reflect the original distribution, we also propose using
an additional disentanglement mapping layer and extracting information from the
full-precision model. The experimental results show that Qimera achieves
state-of-the-art performances for various settings on data-free quantization.
Code is available at this https URL.

    

### [[2111.02627] A Personalized Federated Learning Algorithm: an Application in Anomaly Detection](http://arxiv.org/abs/2111.02627)


  Federated Learning (FL) has recently emerged as a promising method that
employs a distributed learning model structure to overcome data privacy and
transmission issues paused by central machine learning models. In FL, datasets
collected from different devices or sensors are used to train local models
(clients) each of which shares its learning with a centralized model (server).
However, this distributed learning approach presents unique learning challenges
as the data used at local clients can be non-IID (Independent and Identically
Distributed) and statistically diverse which decrease learning accuracy in the
central model. In this paper, we overcome this problem by proposing a novel
Personalized Conditional FedAvg (PC-FedAvg) which aims to control weights
communication and aggregation augmented with a tailored learning algorithm to
personalize the resulting models at each client. Our experimental validation on
two datasets showed that our PC-FedAvg precisely constructed generalized
clients' models and thus achieved higher accuracy compared to other
state-of-the-art methods.

    

### [[2111.02632] A Fast Parallel Tensor Decomposition with Optimal Stochastic Gradient Descent: an Application in Structural Damage Identification](http://arxiv.org/abs/2111.02632)


  Structural Health Monitoring (SHM) provides an economic approach which aims
to enhance understanding the behavior of structures by continuously collects
data through multiple networked sensors attached to the structure. This data is
then utilized to gain insight into the health of a structure and make timely
and economic decisions about its maintenance. The generated SHM sensing data is
non-stationary and exists in a correlated multi-way form which makes the
batch/off-line learning and standard two-way matrix analysis unable to capture
all of these correlations and relationships. In this sense, the online tensor
data analysis has become an essential tool for capturing underlying structures
in higher-order datasets stored in a tensor $\mathcal{X} \in \mathbb{R} ^{I_1
\times \dots \times I_N} $. The CANDECOMP/PARAFAC (CP) decomposition has been
extensively studied and applied to approximate X by N loading matrices A(1), .
. . ,A(N) where N represents the order of the tensor. We propose a novel
algorithm, FP-CPD, to parallelize the CANDECOMP/PARAFAC (CP) decomposition of a
tensor $\mathcal{X} \in \mathbb{R} ^{I_1 \times \dots \times I_N} $. Our
approach is based on stochastic gradient descent (SGD) algorithm which allows
us to parallelize the learning process and it is very useful in online setting
since it updates $\mathcal{X}^{t+1}$ in one single step. Our SGD algorithm is
augmented with Nesterov's Accelerated Gradient (NAG) and perturbation methods
to accelerate and guarantee convergence. The experimental results using
laboratory-based and real-life structural datasets indicate fast convergence
and good scalability.

    

### [[2111.02644] A Concentration Bound for LSPE($λ$)](http://arxiv.org/abs/2111.02644)


  The popular LSPE($\lambda$) algorithm for policy evaluation is revisited to
derive a concentration bound that gives high probability performance guarantees
from some time on.

    

### [[2111.02649] Logically Sound Arguments for the Effectiveness of ML Safety Measures](http://arxiv.org/abs/2111.02649)


  We investigate the issues of achieving sufficient rigor in the arguments for
the safety of machine learning functions. By considering the known weaknesses
of DNN-based 2D bounding box detection algorithms, we sharpen the metric of
imprecise pedestrian localization by associating it with the safety goal. The
sharpening leads to introducing a conservative post-processor after the
standard non-max-suppression as a counter-measure. We then propose a
semi-formal assurance case for arguing the effectiveness of the post-processor,
which is further translated into formal proof obligations for demonstrating the
soundness of the arguments. Applying theorem proving not only discovers the
need to introduce missing claims and mathematical concepts but also reveals the
limitation of Dempster-Shafer's rules used in semi-formal argumentation.

    

### [[2111.02666] Sensory attenuation develops as a result of sensorimotor experience](http://arxiv.org/abs/2111.02666)


  The brain attenuates its responses to self-produced exteroceptions (e.g., we
cannot tickle ourselves). Is this phenomenon, called sensory attenuation,
enabled innately, or is it acquired through learning? To explore the latter
possibility, we created a neural network model consisting of sensory
(proprioceptive and exteroceptive), association, and executive areas. A
simulated robot controlled by the network learned to acquire motor patterns
with self-produced or externally produced exteroceptive feedback. We found that
the robot first increased responses in sensory and association areas for both
self-produced and externally produced conditions in the early stage of
learning, but then, gradually it attenuated responses in sensory areas only for
self-produced conditions. The robot spontaneously acquired a capacity to switch
(attenuate or amplify) responses in sensory areas depending on the conditions
by switching the neural state of the executive area. This suggests that
proactive control of sensory-information flow inside the network was
self-organized through learning. We also found that innate alterations in the
modulation of sensory-information flow induced some characteristics analogous
to schizophrenia and autism spectrum disorder. This study provides a novel
perspective on neural mechanisms underlying perceptual phenomena and
psychiatric disorders.

    

### [[2111.02673] Recurrent Neural Network Training with Convex Loss and Regularization Functions by Extended Kalman Filtering](http://arxiv.org/abs/2111.02673)


  We investigate the use of extended Kalman filtering to train recurrent neural
networks for data-driven nonlinear, possibly adaptive, model-based control
design. We show that the approach can be applied to rather arbitrary convex
loss functions and regularization terms on the network parameters. We show that
the learning method outperforms stochastic gradient descent in a nonlinear
system identification benchmark and in training a linear system with binary
outputs. We also explore the use of the algorithm in data-driven nonlinear
model predictive control and its relation with disturbance models for
offset-free tracking.

    

### [[2111.02682] TimeMatch: Unsupervised Cross-Region Adaptation by Temporal Shift Estimation](http://arxiv.org/abs/2111.02682)


  The recent developments of deep learning models that capture the complex
temporal patterns of crop phenology have greatly advanced crop classification
of Satellite Image Time Series (SITS). However, when applied to target regions
spatially different from the training region, these models perform poorly
without any target labels due to the temporal shift of crop phenology between
regions. To address this unsupervised cross-region adaptation setting, existing
methods learn domain-invariant features without any target supervision, but not
the temporal shift itself. As a consequence, these techniques provide only
limited benefits for SITS. In this paper, we propose TimeMatch, a new
unsupervised domain adaptation method for SITS that directly accounts for the
temporal shift. TimeMatch consists of two components: 1) temporal shift
estimation, which estimates the temporal shift of the unlabeled target region
with a source-trained model, and 2) TimeMatch learning, which combines temporal
shift estimation with semi-supervised learning to adapt a classifier to an
unlabeled target region. We also introduce an open-access dataset for
cross-region adaptation with SITS from four different regions in Europe. On
this dataset, we demonstrate that TimeMatch outperforms all competing methods
by 11% in F1-score across five different adaptation scenarios, setting a new
state-of-the-art for cross-region adaptation.

    

### [[2111.02687] CoreLM: Coreference-aware Language Model Fine-Tuning](http://arxiv.org/abs/2111.02687)


  Language Models are the underpin of all modern Natural Language Processing
(NLP) tasks. The introduction of the Transformers architecture has contributed
significantly into making Language Modeling very effective across many NLP
task, leading to significant advancements in the field. However, Transformers
come with a big computational cost, which grows quadratically with respect to
the input length. This presents a challenge as to understand long texts
requires a lot of context. In this paper, we propose a Fine-Tuning framework,
named CoreLM, that extends the architecture of current Pretrained Language
Models so that they incorporate explicit entity information. By introducing
entity representations, we make available information outside the contextual
space of the model, which results in a better Language Model for a fraction of
the computational cost. We implement our approach using GPT2 and compare the
fine-tuned model to the original. Our proposed model achieves a lower
Perplexity in GUMBY and LAMBDADA datasets when compared to GPT2 and a
fine-tuned version of GPT2 without any changes. We also compare the models'
performance in terms of Accuracy in LAMBADA and Children's Book Test, with and
without the use of model-created coreference annotations.

    

### [[2111.02702] Ex$^2$MCMC: Sampling through Exploration Exploitation](http://arxiv.org/abs/2111.02702)


  We develop an Explore-Exploit Markov chain Monte Carlo algorithm
($\operatorname{Ex^2MCMC}$) that combines multiple global proposals and local
moves. The proposed method is massively parallelizable and extremely
computationally efficient. We prove $V$-uniform geometric ergodicity of
$\operatorname{Ex^2MCMC}$ under realistic conditions and compute explicit
bounds on the mixing rate showing the improvement brought by the multiple
global moves. We show that $\operatorname{Ex^2MCMC}$ allows fine-tuning of
exploitation (local moves) and exploration (global moves) via a novel approach
to proposing dependent global moves. Finally, we develop an adaptive scheme,
$\operatorname{FlEx^2MCMC}$, that learns the distribution of global moves using
normalizing flows. We illustrate the efficiency of $\operatorname{Ex^2MCMC}$
and its adaptive versions on many classical sampling benchmarks. We also show
that these algorithms improve the quality of sampling GANs as energy-based
models.

    

### [[2111.02705] Benchmarking Multimodal AutoML for Tabular Data with Text Fields](http://arxiv.org/abs/2111.02705)


  We consider the use of automated supervised learning systems for data tables
that not only contain numeric/categorical columns, but one or more text fields
as well. Here we assemble 18 multimodal data tables that each contain some text
fields and stem from a real business application. Our publicly-available
benchmark enables researchers to comprehensively evaluate their own methods for
supervised learning with numeric, categorical, and text features. To ensure
that any single modeling strategy which performs well over all 18 datasets will
serve as a practical foundation for multimodal text/tabular AutoML, the diverse
datasets in our benchmark vary greatly in: sample size, problem types (a mix of
classification and regression tasks), number of features (with the number of
text columns ranging from 1 to 28 between datasets), as well as how the
predictive signal is decomposed between text vs. numeric/categorical features
(and predictive interactions thereof). Over this benchmark, we evaluate various
straightforward pipelines to model such data, including standard two-stage
approaches where NLP is used to featurize the text such that AutoML for tabular
data can then be applied. Compared with human data science teams, the fully
automated methodology that performed best on our benchmark (stack ensembling a
multimodal Transformer with various tree models) also manages to rank 1st place
when fit to the raw text/tabular data in two MachineHack prediction
competitions and 2nd place (out of 2380 teams) in Kaggle's Mercari Price
Suggestion Challenge.

    

### [[2111.02708] Quasi-Newton Methods for Saddle Point Problems](http://arxiv.org/abs/2111.02708)


  This paper studies quasi-Newton methods for solving
strongly-convex-strongly-concave saddle point problems (SPP). We propose a
variant of general greedy Broyden family update for SPP, which has explicit
local superlinear convergence rate of ${\mathcal
O}\big(\big(1-\frac{1}{n\kappa^2}\big)^{k(k-1)/2}\big)$, where $n$ is
dimensions of the problem, $\kappa$ is the condition number and $k$ is the
number of iterations. The design and analysis of proposed algorithm are based
on estimating the square of indefinite Hessian matrix, which is different from
classical quasi-Newton methods in convex optimization. We also present two
specific Broyden family algorithms with BFGS-type and SR1-type updates, which
enjoy the faster local convergence rate of $\mathcal
O\big(\big(1-\frac{1}{n}\big)^{k(k-1)/2}\big)$.

    

### [[2111.02710] Towards dynamic multi-modal phenotyping using chest radiographs and physiological data](http://arxiv.org/abs/2111.02710)


  The healthcare domain is characterized by heterogeneous data modalities, such
as imaging and physiological data. In practice, the variety of medical data
assists clinicians in decision-making. However, most of the current
state-of-the-art deep learning models solely rely upon carefully curated data
of a single modality. In this paper, we propose a dynamic training approach to
learn modality-specific data representations and to integrate auxiliary
features, instead of solely relying on a single modality. Our preliminary
experiments results for a patient phenotyping task using physiological data in
MIMIC-IV & chest radiographs in the MIMIC- CXR dataset show that our proposed
approach achieves the highest area under the receiver operating characteristic
curve (AUROC) (0.764 AUROC) compared to the performance of the benchmark method
in previous work, which only used physiological data (0.740 AUROC). For a set
of five recurring or chronic diseases with periodic acute episodes, including
cardiac dysrhythmia, conduction disorders, and congestive heart failure, the
AUROC improves from 0.747 to 0.798. This illustrates the benefit of leveraging
the chest imaging modality in the phenotyping task and highlights the potential
of multi-modal learning in medical applications.

    

### [[2111.02732] When Neural Networks Using Different Sensors Create Similar Features](http://arxiv.org/abs/2111.02732)


  Multimodal problems are omnipresent in the real world: autonomous driving,
robotic grasping, scene understanding, etc... We draw from the well-developed
analysis of similarity to provide an example of a problem where neural networks
are trained from different sensors, and where the features extracted from these
sensors still carry similar information. More precisely, we demonstrate that
for each sensor, the linear combination of the features from the last layer
that correlates the most with other sensors corresponds to the classification
components of the classification layer.

    

### [[2111.02736] Deep Learning Methods for Daily Wildfire Danger Forecasting](http://arxiv.org/abs/2111.02736)


  Wildfire forecasting is of paramount importance for disaster risk reduction
and environmental sustainability. We approach daily fire danger prediction as a
machine learning task, using historical Earth observation data from the last
decade to predict next-day's fire danger. To that end, we collect, pre-process
and harmonize an open-access datacube, featuring a set of covariates that
jointly affect the fire occurrence and spread, such as weather conditions,
satellite-derived products, topography features and variables related to human
activity. We implement a variety of Deep Learning (DL) models to capture the
spatial, temporal or spatio-temporal context and compare them against a Random
Forest (RF) baseline. We find that either spatial or temporal context is enough
to surpass the RF, while a ConvLSTM that exploits the spatio-temporal context
performs best with a test Area Under the Receiver Operating Characteristic of
0.926. Our DL-based proof-of-concept provides national-scale daily fire danger
maps at a much higher spatial resolution than existing operational solutions.

    

### [[2111.02741] Multi-scale 2D Representation Learning for weakly-supervised moment retrieval](http://arxiv.org/abs/2111.02741)


  Video moment retrieval aims to search the moment most relevant to a given
language query. However, most existing methods in this community often require
temporal boundary annotations which are expensive and time-consuming to label.
Hence weakly supervised methods have been put forward recently by only using
coarse video-level label. Despite effectiveness, these methods usually process
moment candidates independently, while ignoring a critical issue that the
natural temporal dependencies between candidates in different temporal scales.
To cope with this issue, we propose a Multi-scale 2D Representation Learning
method for weakly supervised video moment retrieval. Specifically, we first
construct a two-dimensional map for each temporal scale to capture the temporal
dependencies between candidates. Two dimensions in this map indicate the start
and end time points of these candidates. Then, we select top-K candidates from
each scale-varied map with a learnable convolutional neural network. With a
newly designed Moments Evaluation Module, we obtain the alignment scores of the
selected candidates. At last, the similarity between captions and language
query is served as supervision for further training the candidates' selector.
Experiments on two benchmark datasets Charades-STA and ActivityNet Captions
demonstrate that our approach achieves superior performance to state-of-the-art
results.

    

### [[2111.02749] Label Ranking through Nonparametric Regression](http://arxiv.org/abs/2111.02749)


  Label Ranking (LR) corresponds to the problem of learning a hypothesis that
maps features to rankings over a finite set of labels. We adopt a nonparametric
regression approach to LR and obtain theoretical performance guarantees for
this fundamental practical problem. We introduce a generative model for Label
Ranking, in noiseless and noisy nonparametric regression settings, and provide
sample complexity bounds for learning algorithms in both cases. In the
noiseless setting, we study the LR problem with full rankings and provide
computationally efficient algorithms using decision trees and random forests in
the high-dimensional regime. In the noisy setting, we consider the more general
cases of LR with incomplete and partial rankings from a statistical viewpoint
and obtain sample complexity bounds using the One-Versus-One approach of
multiclass classification. Finally, we complement our theoretical contributions
with experiments, aiming to understand how the input regression noise affects
the observed output.

    

### [[2111.02751] FEAFA+: An Extended Well-Annotated Dataset for Facial Expression Analysis and 3D Facial Animation](http://arxiv.org/abs/2111.02751)


  Nearly all existing Facial Action Coding System-based datasets that include
facial action unit (AU) intensity information annotate the intensity values
hierarchically using A--E levels. However, facial expressions change
continuously and shift smoothly from one state to another. Therefore, it is
more effective to regress the intensity value of local facial AUs to represent
whole facial expression changes, particularly in the fields of expression
transfer and facial animation. We introduce an extension of FEAFA in
combination with the relabeled DISFA database, which is available at
this https URL now. Extended FEAFA (FEAFA+) includes 150 video
sequences from FEAFA and DISFA, with a total of 230,184 frames being manually
annotated on floating-point intensity value of 24 redefined AUs using the
Expression Quantitative Tool. We also list crude numerical results for posed
and spontaneous subsets and provide a baseline comparison for the AU intensity
regression task.

    

### [[2111.02757] Online Continual Learning via Multiple Deep Metric Learning and Uncertainty-guided Episodic Memory Replay -- 3rd Place Solution for ICCV 2021 Workshop SSLAD Track 3A Continual Object Classification](http://arxiv.org/abs/2111.02757)


  Online continual learning in the wild is a very difficult task in machine
learning. Non-stationarity in online continual learning potentially brings
about catastrophic forgetting in neural networks. Specifically, online
continual learning for autonomous driving with SODA10M dataset exhibits extra
problems on extremely long-tailed distribution with continuous distribution
shift. To address these problems, we propose multiple deep metric
representation learning via both contrastive and supervised contrastive
learning alongside soft labels distillation to improve model generalization.
Moreover, we exploit modified class-balanced focal loss for sensitive
penalization in class imbalanced and hard-easy samples. We also store some
samples under guidance of uncertainty metric for rehearsal and perform online
and periodical memory updates. Our proposed method achieves considerable
generalization with average mean class accuracy (AMCA) 64.01% on validation and
64.53% AMCA on test set.

    

### [[2111.02763] A Riemannian Accelerated Proximal Extragradient Framework and its Implications](http://arxiv.org/abs/2111.02763)


  The study of accelerated gradient methods in Riemannian optimization has
recently witnessed notable progress. However, in contrast with the Euclidean
setting, a systematic understanding of acceleration is still lacking in the
Riemannian setting. We revisit the \emph{Accelerated Hybrid Proximal
Extragradient} (A-HPE) method of \citet{monteiro2013accelerated}, a powerful
framework for obtaining accelerated Euclidean methods. Subsequently, we propose
a Riemannian version of A-HPE. The basis of our analysis of Riemannian A-HPE is
a set of insights into Euclidean A-HPE, which we combine with a careful control
of distortion caused by Riemannian geometry. We describe a number of Riemannian
accelerated gradient methods as concrete instances of our framework.

    

### [[2111.02767] RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning](http://arxiv.org/abs/2111.02767)


  We introduce RLDS (Reinforcement Learning Datasets), an ecosystem for
recording, replaying, manipulating, annotating and sharing data in the context
of Sequential Decision Making (SDM) including Reinforcement Learning (RL),
Learning from Demonstrations, Offline RL or Imitation Learning. RLDS enables
not only reproducibility of existing research and easy generation of new
datasets, but also accelerates novel research. By providing a standard and
lossless format of datasets it enables to quickly test new algorithms on a
wider range of tasks. The RLDS ecosystem makes it easy to share datasets
without any loss of information and to be agnostic to the underlying original
format when applying various data processing pipelines to large collections of
datasets. Besides, RLDS provides tools for collecting data generated by either
synthetic agents or humans, as well as for inspecting and manipulating the
collected data. Ultimately, integration with TFDS facilitates the sharing of RL
datasets with the research community.

    

### [[2111.02770] Representation Edit Distance as a Measure of Novelty](http://arxiv.org/abs/2111.02770)


  Adaptation to novelty is viewed as learning to change and augment existing
skills to confront unfamiliar situations. In this paper, we propose that the
amount of editing of an effective representation (the Representation Edit
Distance or RED) used in a set of skill programs in an agent's mental model is
a measure of difficulty for adaptation to novelty. The RED is an intuitive
approximation to the change in information content in bit strings measured by
comparing pre-novelty and post-novelty skill programs. We also present some
notional examples of how to use RED for predicting difficulty.

    

### [[2111.02771] The role of MRI physics in brain segmentation CNNs: achieving acquisition invariance and instructive uncertainties](http://arxiv.org/abs/2111.02771)


  Being able to adequately process and combine data arising from different
sites is crucial in neuroimaging, but is difficult, owing to site, sequence and
acquisition-parameter dependent biases. It is important therefore to design
algorithms that are not only robust to images of differing contrasts, but also
be able to generalise well to unseen ones, with a quantifiable measure of
uncertainty. In this paper we demonstrate the efficacy of a physics-informed,
uncertainty-aware, segmentation network that employs augmentation-time MR
simulations and homogeneous batch feature stratification to achieve acquisition
invariance. We show that the proposed approach also accurately extrapolates to
out-of-distribution sequence samples, providing well calibrated volumetric
bounds on these. We demonstrate a significant improvement in terms of
coefficients of variation, backed by uncertainty based volumetric validation.

    

### [[2111.02780] Flood forecasting with machine learning models in an operational framework](http://arxiv.org/abs/2111.02780)


  The operational flood forecasting system by Google was developed to provide
accurate real-time flood warnings to agencies and the public, with a focus on
riverine floods in large, gauged rivers. It became operational in 2018 and has
since expanded geographically. This forecasting system consists of four
subsystems: data validation, stage forecasting, inundation modeling, and alert
distribution. Machine learning is used for two of the subsystems. Stage
forecasting is modeled with the Long Short-Term Memory (LSTM) networks and the
Linear models. Flood inundation is computed with the Thresholding and the
Manifold models, where the former computes inundation extent and the latter
computes both inundation extent and depth. The Manifold model, presented here
for the first time, provides a machine-learning alternative to hydraulic
modeling of flood inundation. When evaluated on historical data, all models
achieve sufficiently high-performance metrics for operational use. The LSTM
showed higher skills than the Linear model, while the Thresholding and Manifold
models achieved similar performance metrics for modeling inundation extent.
During the 2021 monsoon season, the flood warning system was operational in
India and Bangladesh, covering flood-prone regions around rivers with a total
area of 287,000 km2, home to more than 350M people. More than 100M flood alerts
were sent to affected populations, to relevant authorities, and to emergency
organizations. Current and future work on the system includes extending
coverage to additional flood-prone locations, as well as improving modeling
capabilities and accuracy.

    

### [[2111.02784] On the Application of Data-Driven Deep Neural Networks in Linear and Nonlinear Structural Dynamics](http://arxiv.org/abs/2111.02784)


  The use of deep neural network (DNN) models as surrogates for linear and
nonlinear structural dynamical systems is explored. The goal is to develop DNN
based surrogates to predict structural response, i.e., displacements and
accelerations, for given input (harmonic) excitations. In particular, the focus
is on the development of efficient network architectures using fully-connected,
sparsely-connected, and convolutional network layers, and on the corresponding
training strategies that can provide a balance between the overall network
complexity and prediction accuracy in the target dataspaces. For linear
dynamics, sparsity patterns of the weight matrix in the network layers are used
to construct convolutional DNNs with sparse layers. For nonlinear dynamics, it
is shown that sparsity in network layers is lost, and efficient DNNs
architectures with fully-connected and convolutional network layers are
explored. A transfer learning strategy is also introduced to successfully train
the proposed DNNs, and various loading factors that influence the network
architectures are studied. It is shown that the proposed DNNs can be used as
effective and accurate surrogates for predicting linear and nonlinear dynamical
responses under harmonic loadings.

    

### [[2111.02787] Balanced Q-learning: Combining the Influence of Optimistic and Pessimistic Targets](http://arxiv.org/abs/2111.02787)


  The optimistic nature of the Q-learning target leads to an overestimation
bias, which is an inherent problem associated with standard $Q-$learning. Such
a bias fails to account for the possibility of low returns, particularly in
risky scenarios. However, the existence of biases, whether overestimation or
underestimation, need not necessarily be undesirable. In this paper, we
analytically examine the utility of biased learning, and show that specific
types of biases may be preferable, depending on the scenario. Based on this
finding, we design a novel reinforcement learning algorithm, Balanced
Q-learning, in which the target is modified to be a convex combination of a
pessimistic and an optimistic term, whose associated weights are determined
online, analytically. We prove the convergence of this algorithm in a tabular
setting, and empirically demonstrate its superior learning performance in
various environments.

    

### [[2111.02790] LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso](http://arxiv.org/abs/2111.02790)


  Even though Weighted Lasso regression has appealing statistical guarantees,
it is typically avoided due to its complex search space described with
thousands of hyperparameters. On the other hand, the latest progress with
high-dimensional HPO methods for black-box functions demonstrates that
high-dimensional applications can indeed be efficiently optimized. Despite this
initial success, the high-dimensional HPO approaches are typically applied to
synthetic problems with a moderate number of dimensions which limits its impact
in scientific and engineering applications. To address this limitation, we
propose LassoBench, a new benchmark suite tailored for an important open
research topic in the Lasso community that is Weighted Lasso regression.
LassoBench consists of benchmarks on both well-controlled synthetic setups
(number of samples, SNR, ambient and effective dimensionalities, and multiple
fidelities) and real-world datasets, which enable the use of many flavors of
HPO algorithms to be improved and extended to the high-dimensional setting. We
evaluate 5 state-of-the-art HPO methods and 3 baselines, and demonstrate that
Bayesian optimization, in particular, can improve over the methods commonly
used for sparse regression while highlighting limitations of these frameworks
in very high-dimensions. Remarkably, Bayesian optimization improve the Lasso
baselines on 60, 100, 300, and 1000 dimensional problems by 45.7%, 19.2%, 19.7%
and 15.5%, respectively.

    

### [[2111.02801] Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](http://arxiv.org/abs/2111.02801)


  Deep learning has been shown to be an effective tool in solving partial
differential equations (PDEs) through physics-informed neural networks (PINNs).
PINNs embed the PDE residual into the loss function of the neural network, and
have been successfully employed to solve diverse forward and inverse PDE
problems. However, one disadvantage of the first generation of PINNs is that
they usually have limited accuracy even with many training points. Here, we
propose a new method, gradient-enhanced physics-informed neural networks
(gPINNs), for improving the accuracy and training efficiency of PINNs. gPINNs
leverage gradient information of the PDE residual and embed the gradient into
the loss function. We tested gPINNs extensively and demonstrated the
effectiveness of gPINNs in both forward and inverse PDE problems. Our numerical
results show that gPINN performs better than PINN with fewer training points.
Furthermore, we combined gPINN with the method of residual-based adaptive
refinement (RAR), a method for improving the distribution of training points
adaptively during training, to further improve the performance of gPINN,
especially in PDEs with solutions that have steep gradients.

    

### [[2111.02802] Distributed Sparse Feature Selection in Communication-Restricted Networks](http://arxiv.org/abs/2111.02802)


  This paper aims to propose and theoretically analyze a new distributed scheme
for sparse linear regression and feature selection. The primary goal is to
learn the few causal features of a high-dimensional dataset based on noisy
observations from an unknown sparse linear model. However, the presumed
training set which includes $n$ data samples in $\mathbb{R}^p$ is already
distributed over a large network with $N$ clients connected through extremely
low-bandwidth links. Also, we consider the asymptotic configuration of $1\ll
N\ll n\ll p$. In order to infer the causal dimensions from the whole dataset,
we propose a simple, yet effective method for information sharing in the
network. In this regard, we theoretically show that the true causal features
can be reliably recovered with negligible bandwidth usage of $O\left(N\log
p\right)$ across the network. This yields a significantly lower communication
cost in comparison with the trivial case of transmitting all the samples to a
single node (centralized scenario), which requires $O\left(np\right)$
transmissions. Even more sophisticated schemes such as ADMM still have a
communication complexity of $O\left(Np\right)$. Surprisingly, our sample
complexity bound is proved to be the same (up to a constant factor) as the
optimal centralized approach for a fixed performance measure in each node,
while that of a naïve decentralized technique grows linearly with $N$.
Theoretical guarantees in this paper are based on the recent analytic framework
of debiased LASSO in Javanmard et al. (2019), and are supported by several
computer experiments performed on both synthetic and real-world datasets.

    

### [[2111.02803] On Similarity](http://arxiv.org/abs/2111.02803)


  The objective quantification of similarity between two mathematical
structures constitutes a recurrent issue in science and technology. In the
present work, we developed a principled approach that took the Kronecker's
delta function of two scalar values as the prototypical reference for
similarity quantification and then derived for more yielding indices, three of
which bound between 0 and 1. Generalizations of these indices to take into
account the sign of the scalar values were then presented and developed to
multisets, vectors, and functions in real spaces. Several important results
have been obtained, including the interpretation of the Jaccard index as a
yielding implementation of the Kronecker's delta function. When generalized to
real functions, the four described similarity indices become respective
functionals, which can then be employed to obtain associated operations of
convolution and correlation.

    

### [[2111.02813] WaveFake: A Data Set to Facilitate Audio Deepfake Detection](http://arxiv.org/abs/2111.02813)


  Deep generative modeling has the potential to cause significant harm to
society. Recognizing this threat, a magnitude of research into detecting
so-called "Deepfakes" has emerged. This research most often focuses on the
image domain, while studies exploring generated audio signals have, so-far,
been neglected. In this paper we make three key contributions to narrow this
gap. First, we provide researchers with an introduction to common signal
processing techniques used for analyzing audio signals. Second, we present a
novel data set, for which we collected nine sample sets from five different
network architectures, spanning two languages. Finally, we supply practitioners
with two baseline models, adopted from the signal processing community, to
facilitate further research in this area.

    

### [[2111.02823] Convolutional generative adversarial imputation networks for spatio-temporal missing data in storm surge simulations](http://arxiv.org/abs/2111.02823)


  Imputation of missing data is a task that plays a vital role in a number of
engineering and science applications. Often such missing data arise in
experimental observations from limitations of sensors or post-processing
transformation errors. Other times they arise from numerical and algorithmic
constraints in computer simulations. One such instance and the application
emphasis of this paper are numerical simulations of storm surge. The simulation
data corresponds to time-series surge predictions over a number of save points
within the geographic domain of interest, creating a spatio-temporal imputation
problem where the surge points are heavily correlated spatially and temporally,
and the missing values regions are structurally distributed at random. Very
recently, machine learning techniques such as neural network methods have been
developed and employed for missing data imputation tasks. Generative
Adversarial Nets (GANs) and GAN-based techniques have particularly attracted
attention as unsupervised machine learning methods. In this study, the
Generative Adversarial Imputation Nets (GAIN) performance is improved by
applying convolutional neural networks instead of fully connected layers to
better capture the correlation of data and promote learning from the adjacent
surge points. Another adjustment to the method needed specifically for the
studied data is to consider the coordinates of the points as additional
features to provide the model more information through the convolutional
layers. We name our proposed method as Convolutional Generative Adversarial
Imputation Nets (Conv-GAIN). The proposed method's performance by considering
the improvements and adaptations required for the storm surge data is assessed
and compared to the original GAIN and a few other techniques. The results show
that Conv-GAIN has better performance than the alternative methods on the
studied data.

    

### [[2111.02827] Towards Learning to Speak and Hear Through Multi-Agent Communication over a Continuous Acoustic Channel](http://arxiv.org/abs/2111.02827)


  While multi-agent reinforcement learning has been used as an effective means
to study emergent communication between agents, existing work has focused
almost exclusively on communication with discrete symbols. Human communication
often takes place (and emerged) over a continuous acoustic channel; human
infants acquire language in large part through continuous signalling with their
caregivers. We therefore ask: Are we able to observe emergent language between
agents with a continuous communication channel trained through reinforcement
learning? And if so, what is the impact of channel characteristics on the
emerging language? We propose an environment and training methodology to serve
as a means to carry out an initial exploration of these questions. We use a
simple messaging environment where a "speaker" agent needs to convey a concept
to a "listener". The Speaker is equipped with a vocoder that maps symbols to a
continuous waveform, this is passed over a lossy continuous channel, and the
Listener needs to map the continuous signal to the concept. Using deep
Q-learning, we show that basic compositionality emerges in the learned language
representations. We find that noise is essential in the communication channel
when conveying unseen concept combinations. And we show that we can ground the
emergent communication by introducing a caregiver predisposed to "hearing" or
"speaking" English. Finally, we describe how our platform serves as a starting
point for future work that uses a combination of deep reinforcement learning
and multi-agent systems to study our questions of continuous signalling in
language learning and emergence.

    

### [[2111.02840] Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models](http://arxiv.org/abs/2111.02840)


  Large-scale pre-trained language models have achieved tremendous success
across a wide range of natural language understanding (NLU) tasks, even
surpassing human performance. However, recent studies reveal that the
robustness of these models can be challenged by carefully crafted textual
adversarial examples. While several individual datasets have been proposed to
evaluate model robustness, a principled and comprehensive benchmark is still
missing. In this paper, we present Adversarial GLUE (AdvGLUE), a new multi-task
benchmark to quantitatively and thoroughly explore and evaluate the
vulnerabilities of modern large-scale language models under various types of
adversarial attacks. In particular, we systematically apply 14 textual
adversarial attack methods to GLUE tasks to construct AdvGLUE, which is further
validated by humans for reliable annotations. Our findings are summarized as
follows. (i) Most existing adversarial attack algorithms are prone to
generating invalid or ambiguous adversarial examples, with around 90% of them
either changing the original semantic meanings or misleading human annotators
as well. Therefore, we perform a careful filtering process to curate a
high-quality benchmark. (ii) All the language models and robust training
methods we tested perform poorly on AdvGLUE, with scores lagging far behind the
benign accuracy. We hope our work will motivate the development of new
adversarial attacks that are more stealthy and semantic-preserving, as well as
new robust language models against sophisticated adversarial attacks. AdvGLUE
is available at this https URL.

    

### [[2111.02842] Adversarial Attacks on Graph Classification via Bayesian Optimisation](http://arxiv.org/abs/2111.02842)


  Graph neural networks, a popular class of models effective in a wide range of
graph-based learning tasks, have been shown to be vulnerable to adversarial
attacks. While the majority of the literature focuses on such vulnerability in
node-level classification tasks, little effort has been dedicated to analysing
adversarial attacks on graph-level classification, an important problem with
numerous real-life applications such as biochemistry and social network
analysis. The few existing methods often require unrealistic setups, such as
access to internal information of the victim models, or an impractically-large
number of queries. We present a novel Bayesian optimisation-based attack method
for graph classification models. Our method is black-box, query-efficient and
parsimonious with respect to the perturbation applied. We empirically validate
the effectiveness and flexibility of the proposed method on a wide range of
graph classification tasks involving varying graph properties, constraints and
modes of attack. Finally, we analyse common interpretable patterns behind the
adversarial samples produced, which may shed further light on the adversarial
robustness of graph classification models.

    

### [[2111.02845] Attacking Deep Reinforcement Learning-Based Traffic Signal Control Systems with Colluding Vehicles](http://arxiv.org/abs/2111.02845)


  The rapid advancements of Internet of Things (IoT) and artificial
intelligence (AI) have catalyzed the development of adaptive traffic signal
control systems (ATCS) for smart cities. In particular, deep reinforcement
learning (DRL) methods produce the state-of-the-art performance and have great
potentials for practical applications. In the existing DRL-based ATCS, the
controlled signals collect traffic state information from nearby vehicles, and
then optimal actions (e.g., switching phases) can be determined based on the
collected information. The DRL models fully "trust" that vehicles are sending
the true information to the signals, making the ATCS vulnerable to adversarial
attacks with falsified information. In view of this, this paper first time
formulates a novel task in which a group of vehicles can cooperatively send
falsified information to "cheat" DRL-based ATCS in order to save their total
travel time. To solve the proposed task, we develop CollusionVeh, a generic and
effective vehicle-colluding framework composed of a road situation encoder, a
vehicle interpreter, and a communication mechanism. We employ our method to
attack established DRL-based ATCS and demonstrate that the total travel time
for the colluding vehicles can be significantly reduced with a reasonable
number of learning episodes, and the colluding effect will decrease if the
number of colluding vehicles increases. Additionally, insights and suggestions
for the real-world deployment of DRL-based ATCS are provided. The research
outcomes could help improve the reliability and robustness of the ATCS and
better protect the smart mobility systems.

    

### [[2111.02848] Data-Driven Market Segmentation in Hospitality Using Unsupervised Machine Learning](http://arxiv.org/abs/2111.02848)


  Within hospitality, marketing departments use segmentation to create tailored
strategies to ensure personalized marketing. This study provides a data-driven
approach by segmenting guest profiles via hierarchical clustering, based on an
extensive set of features. The industry requires understandable outcomes that
contribute to adaptability for marketing departments to make data-driven
decisions and ultimately driving profit. A marketing department specified a
business question that guides the unsupervised machine learning algorithm.
Features of guests change over time; therefore, there is a probability that
guests transition from one segment to another. The purpose of the study is to
provide steps in the process from raw data to actionable insights, which serve
as a guideline for how hospitality companies can adopt an algorithmic approach.

    

### [[2111.02862] Parameterized Knowledge Transfer for Personalized Federated Learning](http://arxiv.org/abs/2111.02862)


  In recent years, personalized federated learning (pFL) has attracted
increasing attention for its potential in dealing with statistical
heterogeneity among clients. However, the state-of-the-art pFL methods rely on
model parameters aggregation at the server side, which require all models to
have the same structure and size, and thus limits the application for more
heterogeneous scenarios. To deal with such model constraints, we exploit the
potentials of heterogeneous model settings and propose a novel training
framework to employ personalized models for different clients. Specifically, we
formulate the aggregation procedure in original pFL into a personalized group
knowledge transfer training algorithm, namely, KT-pFL, which enables each
client to maintain a personalized soft prediction at the server side to guide
the others' local training. KT-pFL updates the personalized soft prediction of
each client by a linear combination of all local soft predictions using a
knowledge coefficient matrix, which can adaptively reinforce the collaboration
among clients who own similar data distribution. Furthermore, to quantify the
contributions of each client to others' personalized training, the knowledge
coefficient matrix is parameterized so that it can be trained simultaneously
with the models. The knowledge coefficient matrix and the model parameters are
alternatively updated in each round following the gradient descent way.
Extensive experiments on various datasets (EMNIST, Fashion\_MNIST, CIFAR-10)
are conducted under different settings (heterogeneous models and data
distributions). It is demonstrated that the proposed framework is the first
federated learning paradigm that realizes personalized model training via
parameterized group knowledge transfer while achieving significant performance
gain comparing with state-of-the-art algorithms.

    

### [[2111.02865] Testing using Privileged Information by Adapting Features with Statistical Dependence](http://arxiv.org/abs/2111.02865)


  Given an imperfect predictor, we exploit additional features at test time to
improve the predictions made, without retraining and without knowledge of the
prediction function. This scenario arises if training labels or data are
proprietary, restricted, or no longer available, or if training itself is
prohibitively expensive. We assume that the additional features are useful if
they exhibit strong statistical dependence to the underlying perfect predictor.
Then, we empirically estimate and strengthen the statistical dependence between
the initial noisy predictor and the additional features via manifold denoising.
As an example, we show that this approach leads to improvement in real-world
visual attribute ranking. Project webpage: this http URL


### [[2111.02874] Deep Artificial Intelligence for Fantasy Football Language Understanding](http://arxiv.org/abs/2111.02874)


  Fantasy sports allow fans to manage a team of their favorite athletes and
compete with friends. The fantasy platform aligns the real-world statistical
performance of athletes to fantasy scoring and has steadily risen in popularity
to an estimated 9.1 million players per month with 4.4 billion player card
views on the ESPN Fantasy Football platform from 2018-2019. In parallel, the
sports media community produces news stories, blogs, forum posts, tweets,
videos, podcasts and opinion pieces that are both within and outside the
context of fantasy sports. However, human fantasy football players can only
analyze an average of 3.9 sources of information. Our work discusses the
results of a machine learning pipeline to manage an ESPN Fantasy Football team.
The use of trained statistical entity detectors and document2vector models
applied to over 100,000 news sources and 2.3 million articles, videos and
podcasts each day enables the system to comprehend natural language with an
analogy test accuracy of 100% and keyword test accuracy of 80%. Deep learning
feedforward neural networks provide player classifications such as if a player
will be a bust, boom, play with a hidden injury or play meaningful touches with
a cumulative 72% accuracy. Finally, a multiple regression ensemble uses the
deep learning output and ESPN projection data to provide a point projection for
each of the top 500+ fantasy football players in 2018. The point projection
maintained a RMSE of 6.78 points. The best fit probability density function
from a set of 24 is selected to visualize score spreads. Within the first 6
weeks of the product launch, the total number of users spent a cumulative time
of over 4.6 years viewing our AI insights. The training data for our models was
provided by a 2015 to 2016 web archive from Webhose, ESPN statistics, and
Rotowire injury reports. We used 2017 fantasy football data as a test set.

    

### [[2111.02893] Symmetry-Aware Autoencoders: s-PCA and s-nlPCA](http://arxiv.org/abs/2111.02893)


  Nonlinear principal component analysis (nlPCA) via autoencoders has attracted
attention in the dynamical systems community due to its larger compression rate
when compared to linear principal component analysis (PCA). These model
reduction methods experience an increase in the dimensionality of the latent
space when applied to datasets that exhibit globally invariant samples due to
the presence of symmetries. In this study, we introduce a novel machine
learning embedding in the autoencoder, which uses spatial transformer networks
and Siamese networks to account for continuous and discrete symmetries,
respectively. The spatial transformer network discovers the optimal shift for
the continuous translation or rotation so that invariant samples are aligned in
the periodic directions. Similarly, the Siamese networks collapse samples that
are invariant under discrete shifts and reflections. Thus, the proposed
symmetry-aware autoencoder is invariant to predetermined input transformations
dictating the dynamics of the underlying physical system. This embedding can be
employed with both linear and nonlinear reduction methods, which we term
symmetry-aware PCA (s-PCA) and symmetry-aware nlPCA (s-nlPCA). We apply the
proposed framework to 3 fluid flow problems: Burgers' equation, the simulation
of the flow through a step diffuser and the Kolmogorov flow to showcase the
capabilities for cases exhibiting only continuous symmetries, only discrete
symmetries or a combination of both.

    

### [[2111.02901] Certainty Volume Prediction for Unsupervised Domain Adaptation](http://arxiv.org/abs/2111.02901)


  Unsupervised domain adaptation (UDA) deals with the problem of classifying
unlabeled target domain data while labeled data is only available for a
different source domain. Unfortunately, commonly used classification methods
cannot fulfill this task adequately due to the domain gap between the source
and target data. In this paper, we propose a novel uncertainty-aware domain
adaptation setup that models uncertainty as a multivariate Gaussian
distribution in feature space. We show that our proposed uncertainty measure
correlates with other common uncertainty quantifications and relates to
smoothing the classifier's decision boundary, therefore improving the
generalization capabilities. We evaluate our proposed pipeline on challenging
UDA datasets and achieve state-of-the-art results. Code for our method is
available at this https URL.

    

### [[2111.02907] Model-Free Risk-Sensitive Reinforcement Learning](http://arxiv.org/abs/2111.02907)


  We extend temporal-difference (TD) learning in order to obtain
risk-sensitive, model-free reinforcement learning algorithms. This extension
can be regarded as modification of the Rescorla-Wagner rule, where the
(sigmoidal) stimulus is taken to be either the event of over- or
underestimating the TD target. As a result, one obtains a stochastic
approximation rule for estimating the free energy from i.i.d. samples generated
by a Gaussian distribution with unknown mean and variance. Since the Gaussian
free energy is known to be a certainty-equivalent sensitive to the mean and the
variance, the learning rule has applications in risk-sensitive decision-making.

    

### [[2111.02916] A Unified View of Relational Deep Learning for Polypharmacy Side Effect, Combination Synergy, and Drug-Drug Interaction Prediction](http://arxiv.org/abs/2111.02916)


  In recent years, numerous machine learning models which attempt to solve
polypharmacy side effect identification, drug-drug interaction prediction and
combination therapy design tasks have been proposed. Here, we present a unified
theoretical view of relational machine learning models which can address these
tasks. We provide fundamental definitions, compare existing model architectures
and discuss performance metrics, datasets and evaluation protocols. In
addition, we emphasize possible high impact applications and important future
research directions in this domain.

    

### [[2111.02922] Identifying nonlinear dynamical systems from multi-modal time series data](http://arxiv.org/abs/2111.02922)


  Empirically observed time series in physics, biology, or medicine, are
commonly generated by some underlying dynamical system (DS) which is the target
of scientific interest. There is an increasing interest to harvest machine
learning methods to reconstruct this latent DS in a completely data-driven,
unsupervised way. In many areas of science it is common to sample time series
observations from many data modalities simultaneously, e.g.
electrophysiological and behavioral time series in a typical neuroscience
experiment. However, current machine learning tools for reconstructing DSs
usually focus on just one data modality. Here we propose a general framework
for multi-modal data integration for the purpose of nonlinear DS identification
and cross-modal prediction. This framework is based on dynamically
interpretable recurrent neural networks as general approximators of nonlinear
DSs, coupled to sets of modality-specific decoder models from the class of
generalized linear models. Both an expectation-maximization and a variational
inference algorithm for model training are advanced and compared. We show on
nonlinear DS benchmarks that our algorithms can efficiently compensate for too
noisy or missing information in one data channel by exploiting other channels,
and demonstrate on experimental neuroscience data how the algorithm learns to
link different data domains to the underlying dynamics

    

### [[2111.02926] OpenFWI: Benchmark Seismic Datasets for Machine Learning-Based Full Waveform Inversion](http://arxiv.org/abs/2111.02926)


  We present OpenFWI, a collection of large-scale open-source benchmark
datasets for seismic full waveform inversion (FWI). OpenFWI is the
first-of-its-kind in the geoscience and machine learning community to
facilitate diversified, rigorous, and reproducible research on machine
learning-based FWI. OpenFWI includes datasets of multiple scales, encompasses
diverse domains, and covers various levels of model complexity. Along with the
dataset, we also perform an empirical study on each dataset with a
fully-convolutional deep learning model. OpenFWI has been meticulously
maintained and will be regularly updated with new data and experimental
results. We appreciate the inputs from the community to help us further improve
OpenFWI. At the current version, we publish seven datasets in OpenFWI, of which
one is specified for 3D FWI and the rest are for 2D scenarios. All datasets and
related information can be accessed through our website at
this https URL.

    

### [[2111.02930] Decoupled coordinates for machine learning-based molecular fragment linking](http://arxiv.org/abs/2111.02930)


  Recent developments in machine-learning based molecular fragment linking have
demonstrated the importance of informing the generation process with structural
information specifying the relative orientation of the fragments to be linked.
However, such structural information has not yet been provided in the form of a
complete relative coordinate system. Mathematical details for a decoupled set
of bond lengths, bond angles and torsion angles are elaborated and the
coordinate system is demonstrated to be complete. Significant impact on the
quality of the generated linkers is demonstrated numerically. The amount of
reliable information within the different types of degrees of freedom is
investigated. Ablation studies and an information-theoretical analysis are
performed. The presented benefits suggest the application of a complete and
decoupled relative coordinate system as a standard good practice in linker
design.

    

### [[2111.02936] Causal versus Marginal Shapley Values for Robotic Lever Manipulation Controlled using Deep Reinforcement Learning](http://arxiv.org/abs/2111.02936)


  We investigate the effect of including domain knowledge about a robotic
system's causal relations when generating explanations. To this end, we compare
two methods from explainable artificial intelligence, the popular KernelSHAP
and the recent causal SHAP, on a deep neural network trained using deep
reinforcement learning on the task of controlling a lever using a robotic
manipulator. A primary disadvantage of KernelSHAP is that its explanations
represent only the features' direct effects on a model's output, not
considering the indirect effects a feature can have on the output by affecting
other features. Causal SHAP uses a partial causal ordering to alter
KernelSHAP's sampling procedure to incorporate these indirect effects. This
partial causal ordering defines the causal relations between the features, and
we specify this using domain knowledge about the lever control task. We show
that enabling an explanation method to account for indirect effects and
incorporating some domain knowledge can lead to explanations that better agree
with human intuition. This is especially favorable for a real-world robotics
task, where there is considerable causality at play, and in addition, the
required domain knowledge is often handily available.

    

### [[2111.02947] Variational Inference with Holder Bounds](http://arxiv.org/abs/2111.02947)


  The recent introduction of thermodynamic integration techniques has provided
a new framework for understanding and improving variational inference (VI). In
this work, we present a careful analysis of the thermodynamic variational
objective (TVO), bridging the gap between existing variational objectives and
shedding new insights to advance the field. In particular, we elucidate how the
TVO naturally connects the three key variational schemes, namely the
importance-weighted VI, Renyi-VI, and MCMC-VI, which subsumes most VI
objectives employed in practice. To explain the performance gap between theory
and practice, we reveal how the pathological geometry of thermodynamic curves
negatively affects TVO. By generalizing the integration path from the geometric
mean to the weighted Holder mean, we extend the theory of TVO and identify new
opportunities for improving VI. This motivates our new VI objectives, named the
Holder bounds, which flatten the thermodynamic curves and promise to achieve a
one-step approximation of the exact marginal log-likelihood. A comprehensive
discussion on the choices of numerical estimators is provided. We present
strong empirical evidence on both synthetic and real-world datasets to support
our claims.

    

### [[2111.02949] Finite-Time Consensus Learning for Decentralized Optimization with Nonlinear Gossiping](http://arxiv.org/abs/2111.02949)


  Distributed learning has become an integral tool for scaling up machine
learning and addressing the growing need for data privacy. Although more robust
to the network topology, decentralized learning schemes have not gained the
same level of popularity as their centralized counterparts for being less
competitive performance-wise. In this work, we attribute this issue to the lack
of synchronization among decentralized learning workers, showing both
empirically and theoretically that the convergence rate is tied to the
synchronization level among the workers. Such motivated, we present a novel
decentralized learning framework based on nonlinear gossiping (NGO), that
enjoys an appealing finite-time consensus property to achieve better
synchronization. We provide a careful analysis of its convergence and discuss
its merits for modern distributed optimization applications, such as deep
neural networks. Our analysis on how communication delay and randomized chats
affect learning further enables the derivation of practical variants that
accommodate asynchronous and randomized communications. To validate the
effectiveness of our proposal, we benchmark NGO against competing solutions
through an extensive set of tests, with encouraging results reported.

    

### [[2111.02960] Use of low-fidelity models with machine-learning error correction for well placement optimization](http://arxiv.org/abs/2111.02960)


  Well placement optimization is commonly performed using population-based
global stochastic search algorithms. These optimizations are computationally
expensive due to the large number of multiphase flow simulations that must be
conducted. In this work, we present an optimization framework in which these
simulations are performed with low-fidelity (LF) models. These LF models are
constructed from the underlying high-fidelity (HF) geomodel using a global
transmissibility upscaling procedure. Tree-based machine-learning methods,
specifically random forest and light gradient boosting machine, are applied to
estimate the error in objective function value (in this case net present value,
NPV) associated with the LF models. In the offline (preprocessing) step,
preliminary optimizations are performed using LF models, and a clustering
procedure is applied to select a representative set of 100--150 well
configurations to use for training. HF simulation is then performed for these
configurations, and the tree-based models are trained using an appropriate set
of features. In the online (runtime) step, optimization with LF models, with
the machine-learning correction, is conducted. Differential evolution is used
for all optimizations. Results are presented for two example cases involving
the placement of vertical wells in 3D bimodal channelized geomodels. We compare
the performance of our procedure to optimization using HF models. In the first
case, 25 optimization runs are performed with both approaches. Our method
provides an overall speedup factor of 46 relative to optimization using HF
models, with the best-case NPV within 1% of the HF result. In the second case
fewer HF optimization runs are conducted (consistent with actual practice), and
the overall speedup factor with our approach is about 8. In this case, the
best-case NPV from our procedure exceeds the HF result by 3.8%

    

### [[2111.02966] Consistent Estimation for PCA and Sparse Regression with Oblivious Outliers](http://arxiv.org/abs/2111.02966)


  We develop machinery to design efficiently computable and consistent
estimators, achieving estimation error approaching zero as the number of
observations grows, when facing an oblivious adversary that may corrupt
responses in all but an $\alpha$ fraction of the samples. As concrete examples,
we investigate two problems: sparse regression and principal component analysis
(PCA). For sparse regression, we achieve consistency for optimal sample size
$n\gtrsim (k\log d)/\alpha^2$ and optimal error rate $O(\sqrt{(k\log d)/(n\cdot
\alpha^2)})$ where $n$ is the number of observations, $d$ is the number of
dimensions and $k$ is the sparsity of the parameter vector, allowing the
fraction of inliers to be inverse-polynomial in the number of samples. Prior to
this work, no estimator was known to be consistent when the fraction of inliers
$\alpha$ is $o(1/\log \log n)$, even for (non-spherical) Gaussian design
matrices. Results holding under weak design assumptions and in the presence of
such general noise have only been shown in dense setting (i.e., general linear
regression) very recently by d'Orsi et al. [dNS21]. In the context of PCA, we
attain optimal error guarantees under broad spikiness assumptions on the
parameter matrix (usually used in matrix completion). Previous works could
obtain non-trivial guarantees only under the assumptions that the measurement
noise corresponding to the inliers is polynomially small in $n$ (e.g., Gaussian
with variance $1/n^2$).
To devise our estimators, we equip the Huber loss with non-smooth
regularizers such as the $\ell_1$ norm or the nuclear norm, and extend d'Orsi
et al.'s approach [dNS21] in a novel way to analyze the loss function. Our
machinery appears to be easily applicable to a wide range of estimation
problems.

    

### [[2111.02987] Numerical Approximation in CFD Problems Using Physics Informed Machine Learning](http://arxiv.org/abs/2111.02987)


  The thesis focuses on various techniques to find an alternate approximation
method that could be universally used for a wide range of CFD problems but with
low computational cost and low runtime. Various techniques have been explored
within the field of machine learning to gauge the utility in fulfilling the
core ambition. Steady advection diffusion problem has been used as the test
case to understand the level of complexity up to which a method can provide
solution. Ultimately, the focus stays over physics informed machine learning
techniques where solving differential equations is possible without any
training with computed data. The prevalent methods by I.E. Lagaris this http URL. and
M. Raissi this http URL are explored thoroughly. The prevalent methods cannot solve
advection dominant problems. A physics informed method, called as Distributed
Physics Informed Neural Network (DPINN), is proposed to solve advection
dominant problems. It increases the lexibility and capability of older methods
by splitting the domain and introducing other physics-based constraints as mean
squared loss terms. Various experiments are done to explore the end to end
possibilities with the method. Parametric study is also done to understand the
behavior of the method to different tunable parameters. The method is tested
over steady advection-diffusion problems and unsteady square pulse problems.
Very accurate results are recorded. Extreme learning machine (ELM) is a very
fast neural network algorithm at the cost of tunable parameters. The ELM based
variant of the proposed model is tested over the advection-diffusion problem.
ELM makes the complex optimization simpler and Since the method is
non-iterative, the solution is recorded in a single shot. The ELM based variant
seems to work better than the simple DPINN method. Simultaneously scope for
various development in future are hinted throughout the thesis.

    

### [[2111.02994] Towards an Understanding of Default Policies in Multitask Policy Optimization](http://arxiv.org/abs/2111.02994)


  Much of the recent success of deep reinforcement learning has been driven by
regularized policy optimization (RPO) algorithms, with strong performance
across multiple domains. In this family of methods, agents are trained to
maximize cumulative reward while penalizing deviation in behavior from some
reference, or default policy. In addition to empirical success, there is a
strong theoretical foundation for understanding RPO methods applied to single
tasks, with connections to natural gradient, trust region, and variational
approaches. However, there is limited formal understanding of desirable
properties for default policies in the multitask setting, an increasingly
important domain as the field shifts towards training more generally capable
agents. Here, we take a first step towards filling this gap by formally linking
the quality of the default policy to its effect on optimization. Using these
results, we then derive a principled RPO algorithm for multitask learning with
strong performance guarantees.

    

### [[2111.02995] Unsupervised Change Detection of Extreme Events Using ML On-Board](http://arxiv.org/abs/2111.02995)


  In this paper, we introduce RaVAEn, a lightweight, unsupervised approach for
change detection in satellite data based on Variational Auto-Encoders (VAEs)
with the specific purpose of on-board deployment. Applications such as disaster
management enormously benefit from the rapid availability of satellite
observations. Traditionally, data analysis is performed on the ground after all
data is transferred - downlinked - to a ground station. Constraint on the
downlink capabilities therefore affects any downstream application. In
contrast, RaVAEn pre-processes the sampled data directly on the satellite and
flags changed areas to prioritise for downlink, shortening the response time.
We verified the efficacy of our system on a dataset composed of time series of
catastrophic events - which we plan to release alongside this publication -
demonstrating that RaVAEn outperforms pixel-wise baselines. Finally we tested
our approach on resource-limited hardware for assessing computational and
memory limitations.

    

### [[2111.02997] Global Optimality and Finite Sample Analysis of Softmax Off-Policy Actor Critic under State Distribution Mismatch](http://arxiv.org/abs/2111.02997)


  In this paper, we establish the global optimality and convergence rate of an
off-policy actor critic algorithm in the tabular setting without using density
ratio to correct the discrepancy between the state distribution of the behavior
policy and that of the target policy. Our work goes beyond existing works on
the optimality of policy gradient methods in that existing works use the exact
policy gradient for updating the policy parameters while we use an approximate
and stochastic update step. Our update step is not a gradient update because we
do not use a density ratio to correct the state distribution, which aligns well
with what practitioners do. Our update is approximate because we use a learned
critic instead of the true value function. Our update is stochastic because at
each step the update is done for only the current state action pair. Moreover,
we remove several restrictive assumptions from existing works in our analysis.
Central to our work is the finite sample analysis of a generic stochastic
approximation algorithm with time-inhomogeneous update operators on
time-inhomogeneous Markov chains, based on its uniform contraction properties.

    

### [[2111.03000] Reducing the impact of out of vocabulary words in the translation of natural language questions into SPARQL queries](http://arxiv.org/abs/2111.03000)


  Accessing the large volumes of information available in public knowledge
bases might be complicated for those users unfamiliar with the SPARQL query
language. Automatic translation of questions posed in natural language in
SPARQL has the potential of overcoming this problem. Existing systems based on
neural-machine translation are very effective but easily fail in recognizing
words that are Out Of the Vocabulary (OOV) of the training set. This is a
serious issue while querying large ontologies. In this paper, we combine Named
Entity Linking, Named Entity Recognition, and Neural Machine Translation to
perform automatic translation of natural language questions into SPARQL
queries. We demonstrate empirically that our approach is more effective and
resilient to OOV words than existing approaches by running the experiments on
Monument, QALD-9, and LC-QuAD v1, which are well-known datasets for Question
Answering over DBpedia.

    

### [[2111.03003] Scanflow: A multi-graph framework for Machine Learning workflow management, supervision, and debugging](http://arxiv.org/abs/2111.03003)


  Machine Learning (ML) is more than just training models, the whole workflow
must be considered. Once deployed, a ML model needs to be watched and
constantly supervised and debugged to guarantee its validity and robustness in
unexpected situations. Debugging in ML aims to identify (and address) the model
weaknesses in not trivial contexts. Several techniques have been proposed to
identify different types of model weaknesses, such as bias in classification,
model decay, adversarial attacks, etc., yet there is not a generic framework
that allows them to work in a collaborative, modular, portable, iterative way
and, more importantly, flexible enough to allow both human- and machine-driven
techniques. In this paper, we propose a novel containerized directed graph
framework to support and accelerate end-to-end ML workflow management,
supervision, and debugging. The framework allows defining and deploying ML
workflows in containers, tracking their metadata, checking their behavior in
production, and improving the models by using both learned and human-provided
knowledge. We demonstrate these capabilities by integrating in the framework
two hybrid systems to detect data drift distribution which identify the samples
that are far from the latent space of the original distribution, ask for human
intervention, and whether retrain the model or wrap it with a filter to remove
the noise of corrupted data at inference time. We test these systems on
MNIST-C, CIFAR-10-C, and FashionMNIST-C datasets, obtaining promising accuracy
results with the help of human involvement.

    

### [[2111.03015] Modeling Techniques for Machine Learning Fairness: A Survey](http://arxiv.org/abs/2111.03015)


  Machine learning models are becoming pervasive in high-stakes applications.
Despite their clear benefits in terms of performance, the models could show
bias against minority groups and result in fairness issues in a decision-making
process, leading to severe negative impacts on the individuals and the society.
In recent years, various techniques have been developed to mitigate the bias
for machine learning models. Among them, in-processing methods have drawn
increasing attention from the community, where fairness is directly taken into
consideration during model design to induce intrinsically fair models and
fundamentally mitigate fairness issues in outputs and representations. In this
survey, we review the current progress of in-processing bias mitigation
techniques. Based on where the fairness is achieved in the model, we categorize
them into explicit and implicit methods, where the former directly incorporates
fairness metrics in training objectives, and the latter focuses on refining
latent representation learning. Finally, we conclude the survey with a
discussion of the research challenges in this community to motivate future
exploration.

    

### [[2111.03016] Graph neural network initialisation of quantum approximate optimisation](http://arxiv.org/abs/2111.03016)


  Approximate combinatorial optimisation has emerged as one of the most
promising application areas for quantum computers, particularly those in the
near term. In this work, we focus on the quantum approximate optimisation
algorithm (QAOA) for solving the Max-Cut problem. Specifically, we address two
problems in the QAOA, how to select initial parameters, and how to subsequently
train the parameters to find an optimal solution. For the former, we propose
graph neural networks (GNNs) as an initialisation routine for the QAOA
parameters, adding to the literature on warm-starting techniques. We show the
GNN approach generalises across not only graph instances, but also to
increasing graph sizes, a feature not available to other warm-starting
techniques. For training the QAOA, we test several optimisers for the MaxCut
problem. These include quantum aware/agnostic optimisers proposed in literature
and we also incorporate machine learning techniques such as reinforcement and
meta-learning. With the incorporation of these initialisation and optimisation
toolkits, we demonstrate how the QAOA can be trained as an end-to-end
differentiable pipeline.

    

### [[2111.03017] MT3: Multi-Task Multitrack Music Transcription](http://arxiv.org/abs/2111.03017)


  Automatic Music Transcription (AMT), inferring musical notes from raw audio,
is a challenging task at the core of music understanding. Unlike Automatic
Speech Recognition (ASR), which typically focuses on the words of a single
speaker, AMT often requires transcribing multiple instruments simultaneously,
all while preserving fine-scale pitch and timing information. Further, many AMT
datasets are "low-resource", as even expert musicians find music transcription
difficult and time-consuming. Thus, prior work has focused on task-specific
architectures, tailored to the individual instruments of each task. In this
work, motivated by the promising results of sequence-to-sequence transfer
learning for low-resource Natural Language Processing (NLP), we demonstrate
that a general-purpose Transformer model can perform multi-task AMT, jointly
transcribing arbitrary combinations of musical instruments across several
transcription datasets. We show this unified training framework achieves
high-quality transcription results across a range of datasets, dramatically
improving performance for low-resource instruments (such as guitar), while
preserving strong performance for abundant instruments (such as piano).
Finally, by expanding the scope of AMT, we expose the need for more consistent
evaluation metrics and better dataset alignment, and provide a strong baseline
for this new direction of multi-task AMT.

    

### [[2111.03020] Efficacy the of Confinement Policies on the COVID-19 Spread Dynamics in the Early Period of the Pandemic](http://arxiv.org/abs/2111.03020)


  In this study, we propose a clustering-based approach on time-series data to
capture COVID-19 spread patterns in the early period of the pandemic. We
analyze the spread dynamics based on the early and post stages of COVID-19 for
different countries based on different geographical locations. Furthermore, we
investigate the confinement policies and the effect they made on the spread. We
found that implementations of the same confinement policies exhibit different
results in different countries. Specifically, lockdowns become less effective
in densely populated regions, because of the reluctance to comply with social
distancing measures. Lack of testing, contact tracing, and social awareness in
some countries forestall people from self-isolation and maintaining social
distance. Large labor camps with unhealthy living conditions also aid in high
community transmissions in countries depending on foreign labor. Distrust in
government policies and fake news instigate the spread in both developed and
under-developed countries. Large social gatherings play a vital role in causing
rapid outbreaks almost everywhere. While some countries were able to contain
the spread by implementing strict and widely adopted confinement policies, some
others contained the spread with the help of social distancing measures and
rigorous testing capacity. An early and rapid response at the beginning of the
pandemic is necessary to contain the spread, yet it is not always sufficient.

    

### [[2111.03026] B-Pref: Benchmarking Preference-Based Reinforcement Learning](http://arxiv.org/abs/2111.03026)


  Reinforcement learning (RL) requires access to a reward function that
incentivizes the right behavior, but these are notoriously hard to specify for
complex tasks. Preference-based RL provides an alternative: learning policies
using a teacher's preferences without pre-defined rewards, thus overcoming
concerns associated with reward engineering. However, it is difficult to
quantify the progress in preference-based RL due to the lack of a commonly
adopted benchmark. In this paper, we introduce B-Pref: a benchmark specially
designed for preference-based RL. A key challenge with such a benchmark is
providing the ability to evaluate candidate algorithms quickly, which makes
relying on real human input for evaluation prohibitive. At the same time,
simulating human input as giving perfect preferences for the ground truth
reward function is unrealistic. B-Pref alleviates this by simulating teachers
with a wide array of irrationalities, and proposes metrics not solely for
performance but also for robustness to these potential irrationalities. We
showcase the utility of B-Pref by using it to analyze algorithmic design
choices, such as selecting informative queries, for state-of-the-art
preference-based RL algorithms. We hope that B-Pref can serve as a common
starting point to study preference-based RL more systematically. Source code is
available at this https URL.

    

### [[2111.03029] Causal inference with imperfect instrumental variables](http://arxiv.org/abs/2111.03029)


  Instrumental variables allow for quantification of cause and effect
relationships even in the absence of interventions. To achieve this, a number
of causal assumptions must be met, the most important of which is the
independence assumption, which states that the instrument and any confounding
factor must be independent. However, if this independence condition is not met,
can we still work with imperfect instrumental variables? Imperfect instruments
can manifest themselves by violations of the instrumental inequalities that
constrain the set of correlations in the scenario. In this paper, we establish
a quantitative relationship between such violations of instrumental
inequalities and the minimal amount of measurement dependence required to
explain them. As a result, we provide adapted inequalities that are valid in
the presence of a relaxed measurement dependence assumption in the instrumental
scenario. This allows for the adaptation of existing and new lower bounds on
the average causal effect for instrumental scenarios with binary outcomes.
Finally, we discuss our findings in the context of quantum mechanics.

    

### [[2111.03030] An Interpretable Graph Generative Model with Heterophily](http://arxiv.org/abs/2111.03030)


  Many models for graphs fall under the framework of edge-independent dot
product models. These models output the probabilities of edges existing between
all pairs of nodes, and the probability of a link between two nodes increases
with the dot product of vectors associated with the nodes. Recent work has
shown that these models are unable to capture key structures in real-world
graphs, particularly heterophilous structures, wherein links occur between
dissimilar nodes. We propose the first edge-independent graph generative model
that is a) expressive enough to capture heterophily, b) produces nonnegative
embeddings, which allow link predictions to be interpreted in terms of
communities, and c) optimizes effectively on real-world graphs with gradient
descent on a cross-entropy loss. Our theoretical results demonstrate the
expressiveness of our model in its ability to exactly reconstruct a graph using
a number of clusters that is linear in the maximum degree, along with its
ability to capture both heterophily and homophily in the data. Further, our
experiments demonstrate the effectiveness of our model for a variety of
important application tasks such as multi-label clustering and link prediction.

    

### [[2111.03042] Unsupervised Learning of Compositional Energy Concepts](http://arxiv.org/abs/2111.03042)


  Humans are able to rapidly understand scenes by utilizing concepts extracted
from prior experience. Such concepts are diverse, and include global scene
descriptors, such as the weather or lighting, as well as local scene
descriptors, such as the color or size of a particular object. So far,
unsupervised discovery of concepts has focused on either modeling the global
scene-level or the local object-level factors of variation, but not both. In
this work, we propose COMET, which discovers and represents concepts as
separate energy functions, enabling us to represent both global concepts as
well as objects under a unified framework. COMET discovers energy functions
through recomposing the input image, which we find captures independent factors
without additional supervision. Sample generation in COMET is formulated as an
optimization process on underlying energy functions, enabling us to generate
images with permuted and composed concepts. Finally, discovered visual concepts
in COMET generalize well, enabling us to compose concepts between separate
modalities of images as well as with other concepts discovered by a separate
instance of COMET trained on a different dataset. Code and data available at
this https URL.

    

### [[2111.03043] A System for General In-Hand Object Re-Orientation](http://arxiv.org/abs/2111.03043)


  In-hand object reorientation has been a challenging problem in robotics due
to high dimensional actuation space and the frequent change in contact state
between the fingers and the objects. We present a simple model-free framework
that can learn to reorient objects with both the hand facing upwards and
downwards. We demonstrate the capability of reorienting over 2000 geometrically
different objects in both cases. The learned policies show strong zero-shot
transfer performance on new objects. We provide evidence that these policies
are amenable to real-world operation by distilling them to use observations
easily available in the real world. The videos of the learned policies are
available at: this https URL.

    

### [[2111.03044] A Unified Approach to Coreset Learning](http://arxiv.org/abs/2111.03044)


  Coreset of a given dataset and loss function is usually a small weighed set
that approximates this loss for every query from a given set of queries.
Coresets have shown to be very useful in many applications. However, coresets
construction is done in a problem dependent manner and it could take years to
design and prove the correctness of a coreset for a specific family of queries.
This could limit coresets use in practical applications. Moreover, small
coresets provably do not exist for many problems.
To address these limitations, we propose a generic, learning-based algorithm
for construction of coresets. Our approach offers a new definition of coreset,
which is a natural relaxation of the standard definition and aims at
approximating the \emph{average} loss of the original data over the queries.
This allows us to use a learning paradigm to compute a small coreset of a given
set of inputs with respect to a given loss function using a training set of
queries. We derive formal guarantees for the proposed approach. Experimental
evaluation on deep networks and classic machine learning problems show that our
learned coresets yield comparable or even better results than the existing
algorithms with worst-case theoretical guarantees (that may be too pessimistic
in practice). Furthermore, our approach applied to deep network pruning
provides the first coreset for a full deep network, i.e., compresses all the
network at once, and not layer by layer or similar divide-and-conquer methods.

    

### [[2111.03046] Introduction to Coresets: Approximated Mean](http://arxiv.org/abs/2111.03046)


  A \emph{strong coreset} for the mean queries of a set $P$ in ${\mathbb{R}}^d$
is a small weighted subset $C\subseteq P$, which provably approximates its sum
of squared distances to any center (point) $x\in {\mathbb{R}}^d$. A \emph{weak
coreset} is (also) a small weighted subset $C$ of $P$, whose mean approximates
the mean of $P$. While computing the mean of $P$ can be easily computed in
linear time, its coreset can be used to solve harder constrained version, and
is in the heart of generalizations such as coresets for $k$-means clustering.
In this paper, we survey most of the mean coreset construction techniques, and
suggest a unified analysis methodology for providing and explaining classical
and modern results including step-by-step proofs. In particular, we collected
folklore and scattered related results, some of which are not formally stated
elsewhere. Throughout this survey, we present, explain, and prove a set of
techniques, reductions, and algorithms very widespread and crucial in this
field. However, when put to use in the (relatively simple) mean problem, such
techniques are much simpler to grasp. The survey may help guide new researchers
unfamiliar with the field, and introduce them to the very basic foundations of
coresets, through a simple, yet fundamental, problem. Experts in this area
might appreciate the unified analysis flow, and the comparison table for
existing results. Finally, to encourage and help practitioners and software
engineers, we provide full open source code for all presented algorithms.

    

### [[2111.03047] A deep ensemble approach to X-ray polarimetry](http://arxiv.org/abs/2111.03047)


  X-ray polarimetry will soon open a new window on the high energy universe
with the launch of NASA's Imaging X-ray Polarimetry Explorer (IXPE).
Polarimeters are currently limited by their track reconstruction algorithms,
which typically use linear estimators and do not consider individual event
quality. We present a modern deep learning method for maximizing the
sensitivity of X-ray telescopic observations with imaging polarimeters, with a
focus on the gas pixel detectors (GPDs) to be flown on IXPE. We use a weighted
maximum likelihood combination of predictions from a deep ensemble of ResNets,
trained on Monte Carlo event simulations. We derive and apply the optimal event
weighting for maximizing the polarization signal-to-noise ratio (SNR) in track
reconstruction algorithms. For typical power-law source spectra, our method
improves on the current state of the art, providing a ~40% decrease in required
exposure times for a given SNR.

    

### [[2111.03062] Generalization in Dexterous Manipulation via Geometry-Aware Multi-Task Learning](http://arxiv.org/abs/2111.03062)


  Dexterous manipulation of arbitrary objects, a fundamental daily task for
humans, has been a grand challenge for autonomous robotic systems. Although
data-driven approaches using reinforcement learning can develop specialist
policies that discover behaviors to control a single object, they often exhibit
poor generalization to unseen ones. In this work, we show that policies learned
by existing reinforcement learning algorithms can in fact be generalist when
combined with multi-task learning and a well-chosen object representation. We
show that a single generalist policy can perform in-hand manipulation of over
100 geometrically-diverse real-world objects and generalize to new objects with
unseen shape or size. Interestingly, we find that multi-task learning with
object point cloud representations not only generalizes better but even
outperforms the single-object specialist policies on both training as well as
held-out test objects. Video results at
this https URL


### [[1911.04971] Deep Variational Semi-Supervised Novelty Detection](http://arxiv.org/abs/1911.04971)


  In anomaly detection (AD), one seeks to identify whether a test sample is
abnormal, given a data set of normal samples. A recent and promising approach
to AD relies on deep generative models, such as variational autoencoders
(VAEs), for unsupervised learning of the normal data distribution. In
semi-supervised AD (SSAD), the data also includes a small sample of labeled
anomalies. In this work, we propose two variational methods for training VAEs
for SSAD. The intuitive idea in both methods is to train the encoder to
`separate' between latent vectors for normal and outlier data. We show that
this idea can be derived from principled probabilistic formulations of the
problem, and propose simple and effective algorithms. Our methods can be
applied to various data types, as we demonstrate on SSAD datasets ranging from
natural images to astronomy and medicine, can be combined with any VAE model
architecture, and are naturally compatible with ensembling. When comparing to
state-of-the-art SSAD methods that are not specific to particular data types,
we obtain marked improvement in outlier detection.

    

### [[2001.00939] Relative Flatness and Generalization](http://arxiv.org/abs/2001.00939)


  Flatness of the loss curve is conjectured to be connected to the
generalization ability of machine learning models, in particular neural
networks. While it has been empirically observed that flatness measures
consistently correlate strongly with generalization, it is still an open
theoretical problem why and under which circumstances flatness is connected to
generalization, in particular in light of reparameterizations that change
certain flatness measures but leave generalization unchanged. We investigate
the connection between flatness and generalization by relating it to the
interpolation from representative data, deriving notions of representativeness,
and feature robustness. The notions allow us to rigorously connect flatness and
generalization and to identify conditions under which the connection holds.
Moreover, they give rise to a novel, but natural relative flatness measure that
correlates strongly with generalization, simplifies to ridge regression for
ordinary least squares, and solves the reparameterization issue.

    

### [[2006.10916] Probabilistic Fair Clustering](http://arxiv.org/abs/2006.10916)


  In clustering problems, a central decision-maker is given a complete metric
graph over vertices and must provide a clustering of vertices that minimizes
some objective function. In fair clustering problems, vertices are endowed with
a color (e.g., membership in a group), and the features of a valid clustering
might also include the representation of colors in that clustering. Prior work
in fair clustering assumes complete knowledge of group membership. In this
paper, we generalize prior work by assuming imperfect knowledge of group
membership through probabilistic assignments. We present clustering algorithms
in this more general setting with approximation ratio guarantees. We also
address the problem of "metric membership", where different groups have a
notion of order and distance. Experiments are conducted using our proposed
algorithms as well as baselines to validate our approach and also surface
nuanced concerns when group membership is not known deterministically.

    

### [[2009.13240] Texture Memory-Augmented Deep Patch-Based Image Inpainting](http://arxiv.org/abs/2009.13240)


  Patch-based methods and deep networks have been employed to tackle image
inpainting problem, with their own strengths and weaknesses. Patch-based
methods are capable of restoring a missing region with high-quality texture
through searching nearest neighbor patches from the unmasked regions. However,
these methods bring problematic contents when recovering large missing regions.
Deep networks, on the other hand, show promising results in completing large
regions. Nonetheless, the results often lack faithful and sharp details that
resemble the surrounding area. By bringing together the best of both paradigms,
we propose a new deep inpainting framework where texture generation is guided
by a texture memory of patch samples extracted from unmasked regions. The
framework has a novel design that allows texture memory retrieval to be trained
end-to-end with the deep inpainting network. In addition, we introduce a patch
distribution loss to encourage high-quality patch synthesis. The proposed
method shows superior performance both qualitatively and quantitatively on
three challenging image benchmarks, i.e., Places, CelebA-HQ, and Paris
Street-View datasets.

    

### [[2010.02502] Denoising Diffusion Implicit Models](http://arxiv.org/abs/2010.02502)


  Denoising diffusion probabilistic models (DDPMs) have achieved high quality
image generation without adversarial training, yet they require simulating a
Markov chain for many steps to produce a sample. To accelerate sampling, we
present denoising diffusion implicit models (DDIMs), a more efficient class of
iterative implicit probabilistic models with the same training procedure as
DDPMs. In DDPMs, the generative process is defined as the reverse of a
Markovian diffusion process. We construct a class of non-Markovian diffusion
processes that lead to the same training objective, but whose reverse process
can be much faster to sample from. We empirically demonstrate that DDIMs can
produce high quality samples $10 \times$ to $50 \times$ faster in terms of
wall-clock time compared to DDPMs, allow us to trade off computation for sample
quality, and can perform semantically meaningful image interpolation directly
in the latent space.

    

### [[2010.13018] Adversarial Robust Low Rank Matrix Estimation: Compressed Sensing and Matrix Completion](http://arxiv.org/abs/2010.13018)


  We consider robust low rank matrix estimation as a trace regression when
outputs are contaminated by adversaries. The adversaries are allowed to add
arbitrary values to arbitrary outputs. Such values can depend on any samples.
We deal with matrix compressed sensing, including lasso as a partial problem,
and matrix completion, and then we obtain sharp estimation error bounds. To
obtain the error bounds for different models such as matrix compressed sensing
and matrix completion, we propose a simple unified approach based on a
combination of the Huber loss function and the nuclear norm penalization, which
is a different approach from the conventional ones. Some error bounds obtained
in the present paper are sharper than the past ones.

    

### [[2010.16402] Why Do Better Loss Functions Lead to Less Transferable Features?](http://arxiv.org/abs/2010.16402)


  Previous work has proposed many new loss functions and regularizers that
improve test accuracy on image classification tasks. However, it is not clear
whether these loss functions learn better representations for downstream tasks.
This paper studies how the choice of training objective affects the
transferability of the hidden representations of convolutional neural networks
trained on ImageNet. We show that many objectives lead to statistically
significant improvements in ImageNet accuracy over vanilla softmax
cross-entropy, but the resulting fixed feature extractors transfer
substantially worse to downstream tasks, and the choice of loss has little
effect when networks are fully fine-tuned on the new tasks. Using centered
kernel alignment to measure similarity between hidden representations of
networks, we find that differences among loss functions are apparent only in
the last few layers of the network. We delve deeper into representations of the
penultimate layer, finding that different objectives and hyperparameter
combinations lead to dramatically different levels of class separation.
Representations with higher class separation obtain higher accuracy on the
original task, but their features are less useful for downstream tasks. Our
results suggest there exists a trade-off between learning invariant features
for the original task and features relevant for transfer tasks.

    

### [[2012.13344] Generating Long-term Continuous Multi-type Generation Profiles](http://arxiv.org/abs/2012.13344)


  Today, the adoption of new technologies has increased power system dynamics
significantly. Traditional long-term planning studies that most utility
companies perform based on discrete power levels such as peak or average values
cannot reflect system dynamics and often fail to accurately predict system
reliability deficiencies. As a result, long-term future continuous profiles
such as the 8760 hourly profiles are required to enable time-series based
long-term planning studies. However, unlike short-term profiles used for
operation studies, generating long-term continuous profiles that can reflect
both historical time-varying characteristics and future expected power
magnitude is very challenging. Current methods such as average profiling have
major drawbacks. To solve this challenge, this paper proposes a completely
novel approach to generate such profiles for multiple generation types. A
multi-level profile synthesis process is proposed to capture time-varying
characteristics at different time levels. The proposed approach was evaluated
based on a public dataset and demonstrated great performance and application
value for generating long-term continuous multi-type generation profiles.

    

### [[2102.01046] Impossible Tuning Made Possible: A New Expert Algorithm and Its Applications](http://arxiv.org/abs/2102.01046)


  We resolve the long-standing "impossible tuning" issue for the classic expert
problem and show that, it is in fact possible to achieve regret
$O\left(\sqrt{(\ln d)\sum_t \ell_{t,i}^2}\right)$ simultaneously for all expert
$i$ in a $T$-round $d$-expert problem where $\ell_{t,i}$ is the loss for expert
$i$ in round $t$. Our algorithm is based on the Mirror Descent framework with a
correction term and a weighted entropy regularizer. While natural, the
algorithm has not been studied before and requires a careful analysis. We also
generalize the bound to $O\left(\sqrt{(\ln d)\sum_t
(\ell_{t,i}-m_{t,i})^2}\right)$ for any prediction vector $m_t$ that the
learner receives, and recover or improve many existing results by choosing
different $m_t$. Furthermore, we use the same framework to create a master
algorithm that combines a set of base algorithms and learns the best one with
little overhead. The new guarantee of our master allows us to derive many new
results for both the expert problem and more generally Online Linear
Optimization.

    

### [[2102.06514] Large-Scale Representation Learning on Graphs via Bootstrapping](http://arxiv.org/abs/2102.06514)


  Self-supervised learning provides a promising path towards eliminating the
need for costly label information in representation learning on graphs.
However, to achieve state-of-the-art performance, methods often need large
numbers of negative examples and rely on complex augmentations. This can be
prohibitively expensive, especially for large graphs. To address these
challenges, we introduce Bootstrapped Graph Latents (BGRL) - a graph
representation learning method that learns by predicting alternative
augmentations of the input. BGRL uses only simple augmentations and alleviates
the need for contrasting with negative examples, and is thus scalable by
design. BGRL outperforms or matches prior methods on several established
benchmarks, while achieving a 2-10x reduction in memory costs. Furthermore, we
show that BGRL can be scaled up to extremely large graphs with hundreds of
millions of nodes in the semi-supervised regime - achieving state-of-the-art
performance and improving over supervised baselines where representations are
shaped only through label information. In particular, our solution centered on
BGRL constituted one of the winning entries to the Open Graph Benchmark - Large
Scale Challenge at KDD Cup 2021, on a graph orders of magnitudes larger than
all previously available benchmarks, thus demonstrating the scalability and
effectiveness of our approach.

    

### [[2102.08009] EfficientLPS: Efficient LiDAR Panoptic Segmentation](http://arxiv.org/abs/2102.08009)


  Panoptic segmentation of point clouds is a crucial task that enables
autonomous vehicles to comprehend their vicinity using their highly accurate
and reliable LiDAR sensors. Existing top-down approaches tackle this problem by
either combining independent task-specific networks or translating methods from
the image domain ignoring the intricacies of LiDAR data and thus often
resulting in sub-optimal performance. In this paper, we present the novel
top-down Efficient LiDAR Panoptic Segmentation (EfficientLPS) architecture that
addresses multiple challenges in segmenting LiDAR point clouds including
distance-dependent sparsity, severe occlusions, large scale-variations, and
re-projection errors. EfficientLPS comprises of a novel shared backbone that
encodes with strengthened geometric transformation modeling capacity and
aggregates semantically rich range-aware multi-scale features. It incorporates
new scale-invariant semantic and instance segmentation heads along with the
panoptic fusion module which is supervised by our proposed panoptic periphery
loss function. Additionally, we formulate a regularized pseudo labeling
framework to further improve the performance of EfficientLPS by training on
unlabelled data. We benchmark our proposed model on two large-scale LiDAR
datasets: nuScenes, for which we also provide ground truth annotations, and
SemanticKITTI. Notably, EfficientLPS sets the new state-of-the-art on both
these datasets.

    

### [[2102.08087] Making the most of your day: online learning for optimal allocation of time](http://arxiv.org/abs/2102.08087)


  We study online learning for optimal allocation when the resource to be
allocated is time. %Examples of possible applications include job scheduling
for a computing server, a driver filling a day with rides, a landlord renting
an estate, etc. An agent receives task proposals sequentially according to a
Poisson process and can either accept or reject a proposed task. If she accepts
the proposal, she is busy for the duration of the task and obtains a reward
that depends on the task duration. If she rejects it, she remains on hold until
a new task proposal arrives. We study the regret incurred by the agent, first
when she knows her reward function but does not know the distribution of the
task duration, and then when she does not know her reward function, either.
This natural setting bears similarities with contextual (one-armed) bandits,
but with the crucial difference that the normalized reward associated to a
context depends on the whole distribution of contexts.

    

### [[2102.09407] Deep Rational Reinforcement Learning](http://arxiv.org/abs/2102.09407)


  Latest insights from biology show that intelligence not only emerges from the
connections between neurons but that individual neurons shoulder more
computational responsibility than previously anticipated. This perspective
should be critical in the context of constantly changing distinct reinforcement
learning environments, yet current approaches still primarily employ static
activation functions. In this work, we motivate why rationals are suitable for
adaptable activation functions and why their inclusion into neural networks is
crucial. Inspired by recurrence in residual networks, we derive a condition
under which rational units are closed under residual connections and formulate
a naturally regularised version: the recurrent-rational. We demonstrate that
equipping popular algorithms with (recurrent-)rational activations leads to
consistent improvements on Atari games, especially turning simple DQN into a
solid approach, competitive to DDQN and Rainbow.

    

### [[2102.10395] On Calibration and Out-of-domain Generalization](http://arxiv.org/abs/2102.10395)


  Out-of-domain (OOD) generalization is a significant challenge for machine
learning models. Many techniques have been proposed to overcome this challenge,
often focused on learning models with certain invariance properties. In this
work, we draw a link between OOD performance and model calibration, arguing
that calibration across multiple domains can be viewed as a special case of an
invariant representation leading to better OOD generalization. Specifically, we
show that under certain conditions, models which achieve \emph{multi-domain
calibration} are provably free of spurious correlations. This leads us to
propose multi-domain calibration as a measurable and trainable surrogate for
the OOD performance of a classifier. We therefore introduce methods that are
easy to apply and allow practitioners to improve multi-domain calibration by
training or modifying an existing model, leading to better performance on
unseen domains. Using four datasets from the recently proposed WILDS OOD
benchmark, as well as the Colored MNIST dataset, we demonstrate that training
or tuning models so they are calibrated across multiple domains leads to
significantly improved performance on unseen test domains. We believe this
intriguing connection between calibration and OOD generalization is promising
from both a practical and theoretical point of view.

    

### [[2102.12060] Teach Me to Explain: A Review of Datasets for Explainable NLP](http://arxiv.org/abs/2102.12060)


  Explainable NLP (ExNLP) has increasingly focused on collecting
human-annotated textual explanations. These explanations are used downstream in
three ways: as data augmentation to improve performance on a predictive task,
as supervision to train models to produce explanations for their predictions,
and as a ground-truth to evaluate model-generated explanations. In this review,
we identify 65 datasets with three predominant classes of textual explanations
(highlights, free-text, and structured), organize the literature on annotating
each type, identify strengths and shortcomings of existing collection
methodologies, and give recommendations for collecting ExNLP datasets in the
future.

    

### [[2102.12061] Deep Video Prediction for Time Series Forecasting](http://arxiv.org/abs/2102.12061)


  Time series forecasting is essential for decision making in many domains. In
this work, we address the challenge of predicting prices evolution among
multiple potentially interacting financial assets. A solution to this problem
has obvious importance for governments, banks, and investors. Statistical
methods such as Auto Regressive Integrated Moving Average (ARIMA) are widely
applied to these problems. In this paper, we propose to approach economic time
series forecasting of multiple financial assets in a novel way via video
prediction. Given past prices of multiple potentially interacting financial
assets, we aim to predict the prices evolution in the future. Instead of
treating the snapshot of prices at each time point as a vector, we spatially
layout these prices in 2D as an image, such that we can harness the power of
CNNs in learning a latent representation for these financial assets. Thus, the
history of these prices becomes a sequence of images, and our goal becomes
predicting future images. We build on a state-of-the-art video prediction
method for forecasting future images. Our experiments involve the prediction
task of the price evolution of nine financial assets traded in U.S. stock
markets. The proposed method outperforms baselines including ARIMA, Prophet,
and variations of the proposed method, demonstrating the benefits of harnessing
the power of CNNs in the problem of economic time series forecasting.

    

### [[2103.03975] Global canopy height regression and uncertainty estimation from GEDI LIDAR waveforms with deep ensembles](http://arxiv.org/abs/2103.03975)


  NASA's Global Ecosystem Dynamics Investigation (GEDI) is a key climate
mission whose goal is to advance our understanding of the role of forests in
the global carbon cycle. While GEDI is the first space-based LIDAR explicitly
optimized to measure vertical forest structure predictive of aboveground
biomass, the accurate interpretation of this vast amount of waveform data
across the broad range of observational and environmental conditions is
challenging. Here, we present a novel supervised machine learning approach to
interpret GEDI waveforms and regress canopy top height globally. We propose a
probabilistic deep learning approach based on an ensemble of deep convolutional
neural networks(CNN) to avoid the explicit modelling of unknown effects, such
as atmospheric noise. The model learns to extract robust features that
generalize to unseen geographical regions and, in addition, yields reliable
estimates of predictive uncertainty. Ultimately, the global canopy top height
estimates produced by our model have an expected RMSE of 2.7 m with low bias.

    

### [[2103.06720] Variational inference with a quantum computer](http://arxiv.org/abs/2103.06720)


  Inference is the task of drawing conclusions about unobserved variables given
observations of related variables. Applications range from identifying diseases
from symptoms to classifying economic regimes from price movements.
Unfortunately, performing exact inference is intractable in general. One
alternative is variational inference, where a candidate probability
distribution is optimized to approximate the posterior distribution over
unobserved variables. For good approximations, a flexible and highly expressive
candidate distribution is desirable. In this work, we use quantum Born machines
as variational distributions over discrete variables. We apply the framework of
operator variational inference to achieve this goal. In particular, we adopt
two specific realizations: one with an adversarial objective and one based on
the kernelized Stein discrepancy. We demonstrate the approach numerically using
examples of Bayesian networks, and implement an experiment on an IBM quantum
computer. Our techniques enable efficient variational inference with
distributions beyond those that are efficiently representable on a classical
computer.

    

### [[2103.09748] On the Whitney extension problem for near isometries and beyond](http://arxiv.org/abs/2103.09748)


  In this memoir, we develop a general framework which allows for a
simultaneous study of labeled and unlabeled near alignment data problems in
$\mathbb R^D$ and the Whitney near isometry extension problem for discrete and
non-discrete subsets of $\mathbb R^D$ with certain geometries. In addition, we
survey related work of ours on clustering, dimension reduction, manifold
learning, vision as well as minimal energy partitions, discrepancy and min-max
optimization. Numerous open problems in harmonic analysis, computer vision,
manifold learning and signal processing connected to our work are given.
A significant portion of the work in this memoir is based on joint research
with Charles Fefferman in the papers [48], [49], [50], [51].

    

### [[2103.15798] Rethinking Neural Operations for Diverse Tasks](http://arxiv.org/abs/2103.15798)


  An important goal of AutoML is to automate-away the design of neural networks
on new tasks in under-explored domains. Motivated by this goal, we study the
problem of enabling users to discover the right neural operations given data
from their specific domain. We introduce a search space of operations called
XD-Operations that mimic the inductive bias of standard multi-channel
convolutions while being much more expressive: we prove that it includes many
named operations across multiple application areas. Starting with any standard
backbone such as ResNet, we show how to transform it into a search space over
XD-operations and how to traverse the space using a simple weight-sharing
scheme. On a diverse set of tasks -- solving PDEs, distance prediction for
protein folding, and music modeling -- our approach consistently yields models
with lower error than baseline networks and often even lower error than
expert-designed domain-specific approaches.

    

### [[2104.02821] Towards Measuring Fairness in AI: the Casual Conversations Dataset](http://arxiv.org/abs/2104.02821)


  This paper introduces a novel dataset to help researchers evaluate their
computer vision and audio models for accuracy across a diverse set of age,
genders, apparent skin tones and ambient lighting conditions. Our dataset is
composed of 3,011 subjects and contains over 45,000 videos, with an average of
15 videos per person. The videos were recorded in multiple U.S. states with a
diverse set of adults in various age, gender and apparent skin tone groups. A
key feature is that each subject agreed to participate for their likenesses to
be used. Additionally, our age and gender annotations are provided by the
subjects themselves. A group of trained annotators labeled the subjects'
apparent skin tone using the Fitzpatrick skin type scale. Moreover, annotations
for videos recorded in low ambient lighting are also provided. As an
application to measure robustness of predictions across certain attributes, we
provide a comprehensive study on the top five winners of the DeepFake Detection
Challenge (DFDC). Experimental evaluation shows that the winning models are
less performant on some specific groups of people, such as subjects with darker
skin tones and thus may not generalize to all people. In addition, we also
evaluate the state-of-the-art apparent age and gender classification methods.
Our experiments provides a thorough analysis on these models in terms of fair
treatment of people from various backgrounds.

    

### [[2104.11274] Landmark-Aware and Part-based Ensemble Transfer Learning Network for Facial Expression Recognition from Static images](http://arxiv.org/abs/2104.11274)


  Facial Expression Recognition from static images is a challenging problem in
computer vision applications. Convolutional Neural Network (CNN), the
state-of-the-art method for various computer vision tasks, has had limited
success in predicting expressions from faces having extreme poses,
illumination, and occlusion conditions. To mitigate this issue, CNNs are often
accompanied by techniques like transfer, multi-task, or ensemble learning that
often provide high accuracy at the cost of increased computational complexity.
In this work, we propose a Part-based Ensemble Transfer Learning network that
models how humans recognize facial expressions by correlating the spatial
orientation pattern of the facial features with a specific expression. It
consists of 5 sub-networks, and each sub-network performs transfer learning
from one of the five subsets of facial landmarks: eyebrows, eyes, nose, mouth,
or jaw to expression classification. We show that our proposed ensemble network
uses visual patterns emanating from facial muscles' motor movements to predict
expressions and demonstrate the usefulness of transfer learning from Facial
Landmark Localization to Facial Expression Recognition. We test the proposed
network on the CK+, JAFFE, and SFEW datasets, and it outperforms the benchmark
for CK+ and JAFFE datasets by 0.51% and 5.34%, respectively. Additionally, the
proposed ensemble network consists of only 1.65M model parameters, ensuring
computational efficiency during training and real-time deployment. Grad-CAM
visualizations of our proposed ensemble highlight the complementary nature of
its sub-networks, a key design parameter of an effective ensemble network.
Lastly, cross-dataset evaluation results reveal that our proposed ensemble has
a high generalization capacity, making it suitable for real-world usage.

    

### [[2105.02468] Relative stability toward diffeomorphisms indicates performance in deep nets](http://arxiv.org/abs/2105.02468)


  Understanding why deep nets can classify data in large dimensions remains a
challenge. It has been proposed that they do so by becoming stable to
diffeomorphisms, yet existing empirical measurements support that it is often
not the case. We revisit this question by defining a maximum-entropy
distribution on diffeomorphisms, that allows to study typical diffeomorphisms
of a given norm. We confirm that stability toward diffeomorphisms does not
strongly correlate to performance on benchmark data sets of images. By
contrast, we find that the stability toward diffeomorphisms relative to that of
generic transformations $R_f$ correlates remarkably with the test error
$\epsilon_t$. It is of order unity at initialization but decreases by several
decades during training for state-of-the-art architectures. For CIFAR10 and 15
known architectures, we find $\epsilon_t\approx 0.2\sqrt{R_f}$, suggesting that
obtaining a small $R_f$ is important to achieve good performance. We study how
$R_f$ depends on the size of the training set and compare it to a simple model
of invariant learning.

    

### [[2106.00421] OpenBox: A Generalized Black-box Optimization Service](http://arxiv.org/abs/2106.00421)


  Black-box optimization (BBO) has a broad range of applications, including
automatic machine learning, engineering, physics, and experimental design.
However, it remains a challenge for users to apply BBO methods to their
problems at hand with existing software packages, in terms of applicability,
performance, and efficiency. In this paper, we build OpenBox, an open-source
and general-purpose BBO service with improved usability. The modular design
behind OpenBox also facilitates flexible abstraction and optimization of basic
BBO components that are common in other existing systems. OpenBox is
distributed, fault-tolerant, and scalable. To improve efficiency, OpenBox
further utilizes "algorithm agnostic" parallelization and transfer learning.
Our experimental results demonstrate the effectiveness and efficiency of
OpenBox compared to existing systems.

    

### [[2106.00792] Latent Space Refinement for Deep Generative Models](http://arxiv.org/abs/2106.00792)


  Deep generative models are becoming widely used across science and industry
for a variety of purposes. A common challenge is achieving a precise implicit
or explicit representation of the data probability density. Recent proposals
have suggested using classifier weights to refine the learned density of deep
generative models. We extend this idea to all types of generative models and
show how latent space refinement via iterated generative modeling can
circumvent topological obstructions and improve precision. This methodology
also applies to cases were the target model is non-differentiable and has many
internal latent dimensions which must be marginalized over before refinement.
We demonstrate our Latent Space Refinement (LaSeR) protocol on a variety of
examples, focusing on the combinations of Normalizing Flows and Generative
Adversarial Networks.

    

### [[2106.02212] Fuzzy Clustering with Similarity Queries](http://arxiv.org/abs/2106.02212)


  The fuzzy or soft $k$-means objective is a popular generalization of the
well-known $k$-means problem, extending the clustering capability of the
$k$-means to datasets that are uncertain, vague, and otherwise hard to cluster.
In this paper, we propose a semi-supervised active clustering framework, where
the learner is allowed to interact with an oracle (domain expert), asking for
the similarity between a certain set of chosen items. We study the query and
computational complexities of clustering in this framework. We prove that
having a few of such similarity queries enables one to get a polynomial-time
approximation algorithm to an otherwise conjecturally NP-hard problem. In
particular, we provide algorithms for fuzzy clustering in this setting that
asks $O(\mathsf{poly}(k)\log n)$ similarity queries and run with
polynomial-time-complexity, where $n$ is the number of items. The fuzzy
$k$-means objective is nonconvex, with $k$-means as a special case, and is
equivalent to some other generic nonconvex problem such as non-negative matrix
factorization. The ubiquitous Lloyd-type algorithms (or alternating
minimization algorithms) can get stuck at a local minimum. Our results show
that by making a few similarity queries, the problem becomes easier to solve.
Finally, we test our algorithms over real-world datasets, showing their
effectiveness in real-world applications.

    

### [[2106.02666] Counterfactual Explanations Can Be Manipulated](http://arxiv.org/abs/2106.02666)


  Counterfactual explanations are emerging as an attractive option for
providing recourse to individuals adversely impacted by algorithmic decisions.
As they are deployed in critical applications (e.g. law enforcement, financial
lending), it becomes important to ensure that we clearly understand the
vulnerabilities of these methods and find ways to address them. However, there
is little understanding of the vulnerabilities and shortcomings of
counterfactual explanations. In this work, we introduce the first framework
that describes the vulnerabilities of counterfactual explanations and shows how
they can be manipulated. More specifically, we show counterfactual explanations
may converge to drastically different counterfactuals under a small
perturbation indicating they are not robust. Leveraging this insight, we
introduce a novel objective to train seemingly fair models where counterfactual
explanations find much lower cost recourse under a slight perturbation. We
describe how these models can unfairly provide low-cost recourse for specific
subgroups in the data while appearing fair to auditors. We perform experiments
on loan and violent crime prediction data sets where certain subgroups achieve
up to 20x lower cost recourse under the perturbation. These results raise
concerns regarding the dependability of current counterfactual explanation
techniques, which we hope will inspire investigations in robust counterfactual
explanations.

    

### [[2106.03279] Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Problems by Reinforcement Learning](http://arxiv.org/abs/2106.03279)


  In the predict-then-optimize framework, the objective is to train a
predictive model, mapping from environment features to parameters of an
optimization problem, which maximizes decision quality when the optimization is
subsequently solved. Recent work on decision-focused learning shows that
embedding the optimization problem in the training pipeline can improve
decision quality and help generalize better to unseen tasks compared to relying
on an intermediate loss function for evaluating prediction quality. We study
the predict-then-optimize framework in the context of sequential decision
problems (formulated as MDPs) that are solved via reinforcement learning. In
particular, we are given environment features and a set of trajectories from
training MDPs, which we use to train a predictive model that generalizes to
unseen test MDPs without trajectories. Two significant computational challenges
arise in applying decision-focused learning to MDPs: (i) large state and action
spaces make it infeasible for existing techniques to differentiate through MDP
problems, and (ii) the high-dimensional policy space, as parameterized by a
neural network, makes differentiating through a policy expensive. We resolve
the first challenge by sampling provably unbiased derivatives to approximate
and differentiate through optimality conditions, and the second challenge by
using a low-rank approximation to the high-dimensional sample-based
derivatives. We implement both Bellman--based and policy gradient--based
decision-focused learning on three different MDP problems with missing
parameters, and show that decision-focused learning performs better in
generalization to unseen tasks.

    

### [[2106.04502] Federated Hyperparameter Tuning: Challenges, Baselines, and Connections to Weight-Sharing](http://arxiv.org/abs/2106.04502)


  Tuning hyperparameters is a crucial but arduous part of the machine learning
pipeline. Hyperparameter optimization is even more challenging in federated
learning, where models are learned over a distributed network of heterogeneous
devices; here, the need to keep data on device and perform local training makes
it difficult to efficiently train and evaluate configurations. In this work, we
investigate the problem of federated hyperparameter tuning. We first identify
key challenges and show how standard approaches may be adapted to form
baselines for the federated setting. Then, by making a novel connection to the
neural architecture search technique of weight-sharing, we introduce a new
method, FedEx, to accelerate federated hyperparameter tuning that is applicable
to widely-used federated optimization methods such as FedAvg and recent
variants. Theoretically, we show that a FedEx variant correctly tunes the
on-device learning rate in the setting of online convex optimization across
devices. Empirically, we show that FedEx can outperform natural baselines for
federated hyperparameter tuning by several percentage points on the
Shakespeare, FEMNIST, and CIFAR-10 benchmarks, obtaining higher accuracy using
the same training budget.

    

### [[2106.05238] I Don't Need $\mathbf{u}$: Identifiable Non-Linear ICA Without Side Information](http://arxiv.org/abs/2106.05238)


  Recently there has been a renaissance in identifiability results in deep
generative models, not least for non-linear ICA. For i.i.d. data, prior works
have assumed access to a sufficiently-informative auxiliary set of
observations, denoted $\mathbf{u}$. We show here how identifiability can be
obtained in the absence of this side-information. Previous methods have had to
make strong assumptions in order to obtain identifiable models. Here we obtain
empirically identifiable models under a much looser set of constraints. In
particular, we focus on generative models which perform clustering in their
latent space -- a model structure which matches previous identifiable models,
but with the learnt clustering providing a synthetic form of auxiliary
information. We evaluate our proposals, including via statistical tests, and
find that the learned clusterings function effectively: deep generative models
with latent clusterings are empirically identifiable, to the same degree as
models which rely on side information.

    

### [[2106.05932] Early-stopped neural networks are consistent](http://arxiv.org/abs/2106.05932)


  This work studies the behavior of shallow ReLU networks trained with the
logistic loss via gradient descent on binary classification data where the
underlying data distribution is general, and the (optimal) Bayes risk is not
necessarily zero. In this setting, it is shown that gradient descent with early
stopping achieves population risk arbitrarily close to optimal in terms of not
just logistic and misclassification losses, but also in terms of calibration,
meaning the sigmoid mapping of its outputs approximates the true underlying
conditional distribution arbitrarily finely. Moreover, the necessary iteration,
sample, and architectural complexities of this analysis all scale naturally
with a certain complexity measure of the true conditional model. Lastly, while
it is not shown that early stopping is necessary, it is shown that any
univariate classifier satisfying a local interpolation property is
inconsistent.

    

### [[2106.05951] Support Recovery of Sparse Signals from a Mixture of Linear Measurements](http://arxiv.org/abs/2106.05951)


  Recovery of support of a sparse vector from simple measurements is a
widely-studied problem, considered under the frameworks of compressed sensing,
1-bit compressed sensing, and more general single index models. We consider
generalizations of this problem: mixtures of linear regressions, and mixtures
of linear classifiers, where the goal is to recover supports of multiple sparse
vectors using only a small number of possibly noisy linear, and 1-bit
measurements respectively. The key challenge is that the measurements from
different vectors are randomly mixed. Both of these problems have also received
attention recently. In mixtures of linear classifiers, the observations
correspond to the side of queried hyperplane a random unknown vector lies in,
whereas in mixtures of linear regressions we observe the projection of a random
unknown vector on the queried hyperplane. The primary step in recovering the
unknown vectors from the mixture is to first identify the support of all the
individual component vectors. In this work, we study the number of measurements
sufficient for recovering the supports of all the component vectors in a
mixture in both these models. We provide algorithms that use a number of
measurements polynomial in $k, \log n$ and quasi-polynomial in $\ell$, to
recover the support of all the $\ell$ unknown vectors in the mixture with high
probability when each individual component is a $k$-sparse $n$-dimensional
vector.

    

### [[2106.06515] Probability Paths and the Structure of Predictions over Time](http://arxiv.org/abs/2106.06515)


  In settings ranging from weather forecasts to political prognostications to
financial projections, probability estimates of future binary outcomes often
evolve over time. For example, the estimated likelihood of rain on a specific
day changes by the hour as new information becomes available. Given a
collection of such probability paths, we introduce a Bayesian framework --
which we call the Gaussian latent information martingale, or GLIM -- for
modeling the structure of dynamic predictions over time. Suppose, for example,
that the likelihood of rain in a week is 50 %, and consider two hypothetical
scenarios. In the first, one expects the forecast to be equally likely to
become either 25 % or 75 % tomorrow; in the second, one expects the forecast to
stay constant for the next several days. A time-sensitive decision-maker might
select a course of action immediately in the latter scenario, but may postpone
their decision in the former, knowing that new information is imminent. We
model these trajectories by assuming predictions update according to a latent
process of information flow, which is inferred from historical data. In
contrast to general methods for time series analysis, this approach preserves
important properties of probability paths such as the martingale structure and
appropriate amount of volatility and better quantifies future uncertainties
around probability paths. We show that GLIM outperforms three popular baseline
methods, producing better estimated posterior probability path distributions
measured by three different metrics. By elucidating the dynamic structure of
predictions over time, we hope to help individuals make more informed choices.

    

### [[2106.09352] Large Scale Private Learning via Low-rank Reparametrization](http://arxiv.org/abs/2106.09352)


  We propose a reparametrization scheme to address the challenges of applying
differentially private SGD on large neural networks, which are 1) the huge
memory cost of storing individual gradients, 2) the added noise suffering
notorious dimensional dependence. Specifically, we reparametrize each weight
matrix with two \emph{gradient-carrier} matrices of small dimension and a
\emph{residual weight} matrix. We argue that such reparametrization keeps the
forward/backward process unchanged while enabling us to compute the projected
gradient without computing the gradient itself. To learn with differential
privacy, we design \emph{reparametrized gradient perturbation (RGP)} that
perturbs the gradients on gradient-carrier matrices and reconstructs an update
for the original weight from the noisy gradients. Importantly, we use
historical updates to find the gradient-carrier matrices, whose optimality is
rigorously justified under linear regression and empirically verified with deep
learning tasks. RGP significantly reduces the memory cost and improves the
utility. For example, we are the first able to apply differential privacy on
the BERT model and achieve an average accuracy of $83.9\%$ on four downstream
tasks with $\epsilon=8$, which is within $5\%$ loss compared to the non-private
baseline but enjoys much lower privacy leakage risk.

    

### [[2106.10311] Universal Rate-Distortion-Perception Representations for Lossy Compression](http://arxiv.org/abs/2106.10311)


  In the context of lossy compression, Blau & Michaeli (2019) adopt a
mathematical notion of perceptual quality and define the information
rate-distortion-perception function, generalizing the classical rate-distortion
tradeoff. We consider the notion of universal representations in which one may
fix an encoder and vary the decoder to achieve any point within a collection of
distortion and perception constraints. We prove that the corresponding
information-theoretic universal rate-distortion-perception function is
operationally achievable in an approximate sense. Under MSE distortion, we show
that the entire distortion-perception tradeoff of a Gaussian source can be
achieved by a single encoder of the same rate asymptotically. We then
characterize the achievable distortion-perception region for a fixed
representation in the case of arbitrary distributions, identify conditions
under which the aforementioned results continue to hold approximately, and
study the case when the rate is not fixed in advance. This motivates the study
of practical constructions that are approximately universal across the RDP
tradeoff, thereby alleviating the need to design a new encoder for each
objective. We provide experimental results on MNIST and SVHN suggesting that on
image compression tasks, the operational tradeoffs achieved by machine learning
models with a fixed encoder suffer only a small penalty when compared to their
variable encoder counterparts.

    

### [[2106.11853] Credal Self-Supervised Learning](http://arxiv.org/abs/2106.11853)


  Self-training is an effective approach to semi-supervised learning. The key
idea is to let the learner itself iteratively generate "pseudo-supervision" for
unlabeled instances based on its current hypothesis. In combination with
consistency regularization, pseudo-labeling has shown promising performance in
various domains, for example in computer vision. To account for the
hypothetical nature of the pseudo-labels, these are commonly provided in the
form of probability distributions. Still, one may argue that even a probability
distribution represents an excessive level of informedness, as it suggests that
the learner precisely knows the ground-truth conditional probabilities. In our
approach, we therefore allow the learner to label instances in the form of
credal sets, that is, sets of (candidate) probability distributions. Thanks to
this increased expressiveness, the learner is able to represent uncertainty and
a lack of knowledge in a more flexible and more faithful manner. To learn from
weakly labeled data of that kind, we leverage methods that have recently been
proposed in the realm of so-called superset learning. In an exhaustive
empirical evaluation, we compare our methodology to state-of-the-art
self-supervision approaches, showing competitive to superior performance
especially in low-label scenarios incorporating a high degree of uncertainty.

    

### [[2106.15566] Near-Optimal Explainable $k$-Means for All Dimensions](http://arxiv.org/abs/2106.15566)


  Many clustering algorithms are guided by certain cost functions such as the
widely-used $k$-means cost. These algorithms divide data points into clusters
with often complicated boundaries, creating difficulties in explaining the
clustering decision. In a recent work, Dasgupta, Frost, Moshkovitz, and
Rashtchian (ICML 2020) introduced explainable clustering, where the cluster
boundaries are axis-parallel hyperplanes and the clustering is obtained by
applying a decision tree to the data. The central question here is: how much
does the explainability constraint increase the value of the cost function?
Given $d$-dimensional data points, we show an efficient algorithm that finds
an explainable clustering whose $k$-means cost is at most $k^{1 -
2/d}\,\mathrm{poly}(d\log k)$ times the minimum cost achievable by a clustering
without the explainability constraint, assuming $k,d\ge 2$. Taking the minimum
of this bound and the $k\,\mathrm{polylog} (k)$ bound in independent work by
Makarychev-Shan (ICML 2021), Gamlath-Jia-Polak-Svensson (2021), or
Esfandiari-Mirrokni-Narayanan (2021), we get an improved bound of $k^{1 -
2/d}\,\mathrm{polylog}(k)$, which we show is optimal for every choice of
$k,d\ge 2$ up to a poly-logarithmic factor in $k$. For $d = 2$ in particular,
we show an $O(\log k\log\log k)$ bound, improving near-exponentially over the
previous best bound of $O(k\log k)$ by Laber and Murtinho (ICML 2021).

    

### [[2107.00052] Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity](http://arxiv.org/abs/2107.00052)


  Two of the most prominent algorithms for solving unconstrained smooth games
are the classical stochastic gradient descent-ascent (SGDA) and the recently
introduced stochastic consensus optimization (SCO) [Mescheder et al., 2017].
SGDA is known to converge to a stationary point for specific classes of games,
but current convergence analyses require a bounded variance assumption. SCO is
used successfully for solving large-scale adversarial problems, but its
convergence guarantees are limited to its deterministic variant. In this work,
we introduce the expected co-coercivity condition, explain its benefits, and
provide the first last-iterate convergence guarantees of SGDA and SCO under
this condition for solving a class of stochastic variational inequality
problems that are potentially non-monotone. We prove linear convergence of both
methods to a neighborhood of the solution when they use constant step-size, and
we propose insightful stepsize-switching rules to guarantee convergence to the
exact solution. In addition, our convergence guarantees hold under the
arbitrary sampling paradigm, and as such, we give insights into the complexity
of minibatching.

    

### [[2107.00717] SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios](http://arxiv.org/abs/2107.00717)


  Active learning has proven to be useful for minimizing labeling costs by
selecting the most informative samples. However, existing active learning
methods do not work well in realistic scenarios such as imbalance or rare
classes, out-of-distribution data in the unlabeled set, and redundancy. In this
work, we propose SIMILAR (Submodular Information Measures based actIve
LeARning), a unified active learning framework using recently proposed
submodular information measures (SIM) as acquisition functions. We argue that
SIMILAR not only works in standard active learning, but also easily extends to
the realistic settings considered above and acts as a one-stop solution for
active learning that is scalable to large real-world datasets. Empirically, we
show that SIMILAR significantly outperforms existing active learning algorithms
by as much as ~5% - 18% in the case of rare classes and ~5% - 10% in the case
of out-of-distribution data on several image classification tasks like
CIFAR-10, MNIST, and ImageNet. SIMILAR is available as a part of the DISTIL
toolkit: "this https URL".

    

### [[2111.00658] RMNA: A Neighbor Aggregation-Based Knowledge Graph Representation Learning Model Using Rule Mining](http://arxiv.org/abs/2111.00658)


  Although the state-of-the-art traditional representation learning (TRL)
models show competitive performance on knowledge graph completion, there is no
parameter sharing between the embeddings of entities, and the connections
between entities are weak. Therefore, neighbor aggregation-based representation
learning (NARL) models are proposed, which encode the information in the
neighbors of an entity into its embeddings. However, existing NARL models
either only utilize one-hop neighbors, ignoring the information in multi-hop
neighbors, or utilize multi-hop neighbors by hierarchical neighbor aggregation,
destroying the completeness of multi-hop neighbors. In this paper, we propose a
NARL model named RMNA, which obtains and filters horn rules through a rule
mining algorithm, and uses selected horn rules to transform valuable multi-hop
neighbors into one-hop neighbors, therefore, the information in valuable
multi-hop neighbors can be completely utilized by aggregating these one-hop
neighbors. In experiments, we compare RMNA with the state-of-the-art TRL models
and NARL models. The results show that RMNA has a competitive performance.

    

### [[2111.00998] PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Learning](http://arxiv.org/abs/2111.00998)


  PDE discovery shows promise for uncovering predictive models for complex
physical systems but has difficulty when measurements are sparse and noisy. We
introduce a new approach for PDE discovery that uses two Rational Neural
Networks and a principled sparse regression algorithm to identify the hidden
dynamics that govern a system's response. The first network learns the system
response function, while the second learns a hidden PDE which drives the
system's evolution. We then use a parameter-free sparse regression algorithm to
extract a human-readable form of the hidden PDE from the second network. We
implement our approach in an open-source library called PDE-READ. Our approach
successfully identifies the Heat, Burgers, and Korteweg-De Vries equations with
remarkable consistency. We demonstrate that our approach is unprecedentedly
robust to both sparsity and noise and is, therefore, applicable to real-world
observational data.

    

### [[2111.02524] TOSCAdata: Modelling data pipeline applications in TOSCA](http://arxiv.org/abs/2111.02524)


  The serverless platform allows a customer to effectively use cloud resources
and pay for the exact amount of used resources. A number of dedicated open
source and commercial cloud data management tools are available to handle the
massive amount of data. Such modern cloud data management tools are not enough
matured to integrate the generic cloud application with the serverless platform
due to the lack of mature and stable standards. One of the most popular and
mature standards, TOSCA (Topology and Orchestration Specification for Cloud
Applications), mainly focuses on application and service portability and
automated management of the generic cloud application components. This paper
proposes the extension of the TOSCA standard, TOSCAdata, that focuses on the
modeling of data pipeline-based cloud applications. Keeping the requirements of
modern data pipeline cloud applications, TOSCAdata provides a number of TOSCA
models that are independently deployable, schedulable, scalable, and re-usable,
while effectively handling the flow and transformation of data in a pipeline
manner. We also demonstrate the applicability of proposed TOSCAdata models by
taking a web-based cloud application in the context of tourism promotion as a
use case scenario.

    

### [[2111.02604] Auto Tuning of Hadoop and Spark parameters](http://arxiv.org/abs/2111.02604)


  Data of the order of terabytes, petabytes, or beyond is known as Big Data.
This data cannot be processed using the traditional database software, and
hence there comes the need for Big Data Platforms. By combining the
capabilities and features of various big data applications and utilities, Big
Data Platforms form a single solution. It is a platform that helps to develop,
deploy and manage the big data environment. Hadoop and Spark are the two
open-source Big Data Platforms provided by Apache. Both these platforms have
many configurational parameters, which can have unforeseen effects on the
execution time, accuracy, etc. Manual tuning of these parameters can be
tiresome, and hence automatic ways should be needed to tune them. After
studying and analyzing various previous works in automating the tuning of these
parameters, this paper proposes two algorithms - Grid Search with Finer Tuning
and Controlled Random Search. The performance indicator studied in this paper
is Execution Time. These algorithms help to tune the parameters automatically.
Experimental results have shown a reduction in execution time of about 70% and
50% for Hadoop and 81.19% and 77.77% for Spark by Grid Search with Finer Tuning
and Controlled Random Search, respectively.

    

### [[2111.02706] A thread-safe Term Library](http://arxiv.org/abs/2111.02706)


  Terms are one of the fundamental data structures for computing. E.g. every
expression characterisable by a context free grammar is a term. Remarkably,
terms are not yet standard in common programming languages although term
libraries have already been proposed in the 1990-ies. We developed a
thread-safe Term Library. The biggest challenge is to implement hyper-efficient
multi-reader/single writer mutual exclusion for which we designed the new
busy-forbidden protocol. Model checking is used to show both the correctness of
the protocol and the library. Benchmarks show this Term Library to scale well,
and to compare favourably with sequential versions. Using the new library in an
existing state space generation tool, very substantial speed ups can be
obtained.

    

### [[2111.02709] Analog MIMO Communication for One-shot Distributed Principal Component Analysis](http://arxiv.org/abs/2111.02709)


  A fundamental algorithm for data analytics at the edge of wireless networks
is distributed principal component analysis (DPCA), which finds the most
important information embedded in a distributed high-dimensional dataset by
distributed computation of a reduced-dimension data subspace, called principal
components (PCs). In this paper, to support one-shot DPCA in wireless systems,
we propose a framework of analog MIMO transmission featuring the uncoded analog
transmission of local PCs for estimating the global PCs. To cope with channel
distortion and noise, two maximum-likelihood (global) PC estimators are
presented corresponding to the cases with and without receive channel state
information (CSI). The first design, termed coherent PC estimator, is derived
by solving a Procrustes problem and reveals the form of regularized channel
inversion where the regulation attempts to alleviate the effects of both
channel noise and data noise. The second one, termed blind PC estimator, is
designed based on the subspace channel-rotation-invariance property and
computes a centroid of received local PCs on a Grassmann manifold. Using the
manifold-perturbation theory, tight bounds on the mean square subspace distance
(MSSD) of both estimators are derived for performance evaluation. The results
reveal simple scaling laws of MSSD concerning device population, data and
channel signal-to-noise ratios (SNRs), and array sizes. More importantly, both
estimators are found to have identical scaling laws, suggesting the
dispensability of CSI to accelerate DPCA. Simulation results validate the
derived results and demonstrate the promising latency performance of the
proposed analog MIMO.

    

### [[2111.02719] SPEEDEX: A Scalable, Parallelizable, and Economically Efficient Digital EXchange](http://arxiv.org/abs/2111.02719)


  SPEEDEX is a decentralized exchange (DEX) letting participants securely trade
assets without giving any single party undue control over the market. SPEEDEX
offers several advantages over prior DEXes. It achieves high throughput -- over
100,000 transactions per second on 32-core servers, even with 70M open offers.
It eliminates internal arbitrage opportunities, so that a direct trade from
asset $A$ to $B$ always receives as good a price as trading through some third
asset such as USD\@. Finally, it prevents frontrunning attacks that would
otherwise increase the effective bid-ask spread for small traders. SPEEDEX's
key design insight is to use an Arrow-Debreu exchange market structure that
fixes the valuation of assets for all trades in a given block of transactions.
Not only does this market structure provide fairness across trades, it makes
trade operations commutative and hence efficiently parallelizable.

    

### [[2111.02725] Effect of Miner Incentive on the Confirmation Time of Bitcoin Transactions](http://arxiv.org/abs/2111.02725)


  Blockchain is a technology that provides a distributed ledger that stores
previous records while maintaining consistency and security. Bitcoin is the
first and largest decentralized electronic cryptographic system that uses
blockchain technology. It faces a challenge in making all the nodes synchronize
and have the same overall view with the cost of scalability and performance. In
addition, with miners' financial interest playing a significant role in
choosing transactions from the backlog, small fee or small fee per byte value
transactions will exhibit more delays. To study the issues related to the
system's performance, we developed an $M(t)/M^N/1$ model. The backlog's arrival
follows an inhomogeneous Poison process to the system that has infinite buffer
capacity, and the service time is distributed exponentially, which removes $N$
transactions at time. Besides validating the model with measurement data, we
have used the model to study the reward distribution when miners take
transaction selection strategies like fee per byte, fee-based, and FIFO. The
analysis shows that smaller fee transactions exhibit higher waiting times, even
with increasing the block size. Moreover, the miner transaction selection
strategy impacts the final gain.

    

### [[2111.02727] Failure Aware Semi-Centralized Virtual Network Embedding in Cloud Computing Fat-Tree Data Center Networks](http://arxiv.org/abs/2111.02727)


  In Cloud Computing, the tenants opting for the Infrastructure as a Service
(IaaS) send the resource requirements to the Cloud Service Provider (CSP) in
the form of Virtual Network (VN) consisting of a set of inter-connected Virtual
Machines (VM). Embedding the VN onto the existing physical network is known as
Virtual Network Embedding (VNE) problem. One of the major research challenges
is to allocate the physical resources such that the failure of the physical
resources would bring less impact onto the users' service. Additionally, the
major challenge is to handle the embedding process of growing number of
incoming users' VNs from the algorithm design point-of-view. Considering both
of the above-mentioned research issues, a novel Failure aware Semi-Centralized
VNE (FSC-VNE) algorithm is proposed for the Fat-Tree data center network with
the goal to reduce the impact of the resource failure onto the existing users.
The impact of failure of the Physical Machines (PMs), physical links and
network devices are taken into account while allocating the resources to the
users. The beauty of the proposed algorithm is that the VMs are assigned to
different PMs in a semi-centralized manner. In other words, the embedding
algorithm is executed by multiple physical servers in order to concurrently
embed the VMs of a VN and reduces the embedding time. Extensive simulation
results show that the proposed algorithm can outperform over other VNE
algorithms.

    

### [[2111.02737] MUVINE: Multi-stage Virtual Network Embedding in Cloud Data Centers using Reinforcement Learning based Predictions](http://arxiv.org/abs/2111.02737)


  The recent advances in virtualization technology have enabled the sharing of
computing and networking resources of cloud data centers among multiple users.
Virtual Network Embedding (VNE) is highly important and is an integral part of
the cloud resource management. The lack of historical knowledge on cloud
functioning and inability to foresee the future resource demand are two
fundamental shortcomings of the traditional VNE approaches. The consequence of
those shortcomings is the inefficient embedding of virtual resources on
Substrate Nodes (SNs). On the contrary, application of Artificial Intelligence
(AI) in VNE is still in the premature stage and needs further investigation.
Considering the underlying complexity of VNE that includes numerous parameters,
intelligent solutions are required to utilize the cloud resources efficiently
via careful selection of appropriate SNs for the VNE. In this paper,
Reinforcement Learning based prediction model is designed for the efficient
Multi-stage Virtual Network Embedding (MUVINE) among the cloud data centers.
The proposed MUVINE scheme is extensively simulated and evaluated against the
recent state-of-the-art schemes. The simulation outcomes show that the proposed
MUVINE scheme consistently outperforms over the existing schemes and provides
the promising results.

    

### [[2111.02869] Earthquake detection at the edge: IoT crowdsensing network](http://arxiv.org/abs/2111.02869)


  Earthquake Early Warning state of the art systems rely on a network of
sensors connected to a fusion center in a client-server paradigm. Instead, we
propose moving computation to the edge, with detector nodes that probe the
environment and process information from nearby probes to detect earthquakes
locally. Our approach tolerates multiple node faults and partial network
disruption and keeps all data locally, enhancing privacy. This paper describes
our proposal's rationale and explains its architecture. We then present an
implementation using Raspberry, NodeMCU, and the Crowdquake machine learning
model.

    

### [[2111.02925] SZ3: A Modular Framework for Composing Prediction-Based Error-Bounded Lossy Compressors](http://arxiv.org/abs/2111.02925)


  Today's scientific simulations require a significant reduction of data volume
because of extremely large amounts of data they produce and the limited I/O
bandwidth and storage space. Error-bounded lossy compressor has been considered
one of the most effective solutions to the above problem. In practice, however,
the best-fit compression method often needs to be customized/optimized in
particular because of diverse characteristics in different datasets and various
user requirements on the compression quality and performance. In this paper, we
develop a novel modular, composable compression framework (namely SZ3), which
involves three significant contributions. (1) SZ3 features a modular
abstraction for the prediction-based compression framework such that the new
compression modules can be plugged in easily. (2) SZ3 supports multialgorithm
predictors and can automatically select the best-fit predictor for each data
block based on the designed error estimation criterion. (3) SZ3 allows users to
easily compose different compression pipelines on demand, such that both
compression quality and performance can be significantly improved for their
specific datasets and requirements. (4) In addition, we evaluate several lossy
compressors composed from SZ3 using the real-world datasets. Specifically, we
leverage SZ3 to improve the compression quality and performance for different
use-cases, including GAMESS quantum chemistry dataset and Advanced Photon
Source (APS) instrument dataset. Experiments show that our customized
compression pipelines lead to up to 20% improvement in compression ratios under
the same data distortion compared with the state-of-the-art approaches.

    

### [[2111.02450] Unified 3D Mesh Recovery of Humans and Animals by Learning Animal Exercise](http://arxiv.org/abs/2111.02450)


  We propose an end-to-end unified 3D mesh recovery of humans and quadruped
animals trained in a weakly-supervised way. Unlike recent work focusing on a
single target class only, we aim to recover 3D mesh of broader classes with a
single multi-task model. However, there exists no dataset that can directly
enable multi-task learning due to the absence of both human and animal
annotations for a single object, e.g., a human image does not have animal pose
annotations; thus, we have to devise a new way to exploit heterogeneous
datasets. To make the unstable disjoint multi-task learning jointly trainable,
we propose to exploit the morphological similarity between humans and animals,
motivated by animal exercise where humans imitate animal poses. We realize the
morphological similarity by semantic correspondences, called sub-keypoint,
which enables joint training of human and animal mesh regression branches.
Besides, we propose class-sensitive regularization methods to avoid a
mean-shape bias and to improve the distinctiveness across multi-classes. Our
method performs favorably against recent uni-modal models on various human and
animal datasets while being far more compact.

    

### [[2111.02493] Roadmap on Signal Processing for Next Generation Measurement Systems](http://arxiv.org/abs/2111.02493)


  Signal processing is a fundamental component of almost any sensor-enabled
system, with a wide range of applications across different scientific
disciplines. Time series data, images, and video sequences comprise
representative forms of signals that can be enhanced and analysed for
information extraction and quantification. The recent advances in artificial
intelligence and machine learning are shifting the research attention towards
intelligent, data-driven, signal processing. This roadmap presents a critical
overview of the state-of-the-art methods and applications aiming to highlight
future challenges and research opportunities towards next generation
measurement systems. It covers a broad spectrum of topics ranging from basic to
industrial research, organized in concise thematic sections that reflect the
trends and the impacts of current and future developments per research field.
Furthermore, it offers guidance to researchers and funding agencies in
identifying new prospects.

    

### [[2111.02603] On Semantic Cognition, Inductive Generalization, and Language Models](http://arxiv.org/abs/2111.02603)


  My doctoral research focuses on understanding semantic knowledge in neural
network models trained solely to predict natural language (referred to as
language models, or LMs), by drawing on insights from the study of concepts and
categories grounded in cognitive science. I propose a framework inspired by
'inductive reasoning,' a phenomenon that sheds light on how humans utilize
background knowledge to make inductive leaps and generalize from new pieces of
information about concepts and their properties. Drawing from experiments that
study inductive reasoning, I propose to analyze semantic inductive
generalization in LMs using phenomena observed in human-induction literature,
investigate inductive behavior on tasks such as implicit reasoning and emergent
feature recognition, and analyze and relate induction dynamics to the learned
conceptual representation space.

    

### [[2111.02626] Characterizing Human Explanation Strategies to Inform the Design of Explainable AI for Building Damage Assessment](http://arxiv.org/abs/2111.02626)


  Explainable AI (XAI) is a promising means of supporting human-AI
collaborations for high-stakes visual detection tasks, such as damage detection
tasks from satellite imageries, as fully-automated approaches are unlikely to
be perfectly safe and reliable. However, most existing XAI techniques are not
informed by the understandings of task-specific needs of humans for
explanations. Thus, we took a first step toward understanding what forms of XAI
humans require in damage detection tasks. We conducted an online crowdsourced
study to understand how people explain their own assessments, when evaluating
the severity of building damage based on satellite imagery. Through the study
with 60 crowdworkers, we surfaced six major strategies that humans utilize to
explain their visual damage assessments. We present implications of our
findings for the design of XAI methods for such visual detection contexts, and
discuss opportunities for future research.

    

### [[2111.02636] A control method for solving high-dimensional Hamiltonian systems through deep neural networks](http://arxiv.org/abs/2111.02636)


  In this paper, we mainly focus on solving high-dimensional stochastic
Hamiltonian systems with boundary condition, and propose a novel method from
the view of the stochastic control. In order to obtain the approximated
solution of the Hamiltonian system, we first introduce a corresponding
stochastic optimal control problem such that the Hamiltonian system of control
problem is exactly what we need to solve, then develop two different algorithms
suitable for different cases of the control problem and approximate the
stochastic control via deep neural networks. From the numerical results,
comparing with the Deep FBSDE method which was developed previously from the
view of solving FBSDEs, the novel algorithms converge faster, which means that
they require fewer training steps, and demonstrate more stable convergences for
different Hamiltonian systems.

    

### [[2111.02671] GraphSearchNet: Enhancing GNNs via Capturing Global Dependency for Semantic Code Search](http://arxiv.org/abs/2111.02671)


  Code search aims to retrieve the relevant code fragments based on a natural
language query to improve the software productivity and quality. However,
automatic code search is challenging due to the semantic gap between the source
code and the query. Most existing approaches mainly consider the sequential
information for embedding, where the structure information behind the text is
not fully considered. In this paper, we design a novel neural network
framework, named GraphSearchNet, to enable an effective and accurate source
code search by jointly learning rich semantics of both source code and queries.
Specifically, we propose to encode both source code and queries into two graphs
with Bidirectional GGNN to capture the local structure information of the
graphs. Furthermore, we enhance BiGGNN by utilizing the effective multi-head
attention to supplement the global dependency that BiGGNN missed. The extensive
experiments on both Java and Python datasets illustrate that GraphSearchNet
outperforms current state-of-the-art works by a significant margin.

    

### [[2111.02692] Human Age Estimation from Gene Expression Data using Artificial Neural Networks](http://arxiv.org/abs/2111.02692)


  The study of signatures of aging in terms of genomic biomarkers can be
uniquely helpful in understanding the mechanisms of aging and developing models
to accurately predict the age. Prior studies have employed gene expression and
DNA methylation data aiming at accurate prediction of age. In this line, we
propose a new framework for human age estimation using information from human
dermal fibroblast gene expression data. First, we propose a new spatial
representation as well as a data augmentation approach for gene expression
data. Next in order to predict the age, we design an architecture of neural
network and apply it to this new representation of the original and augmented
data, as an ensemble classification approach. Our experimental results suggest
the superiority of the proposed framework over state-of-the-art age estimation
methods using DNA methylation and gene expression data.

    

### [[2111.02724] Tea Chrysanthemum Detection under Unstructured Environments Using the TC-YOLO Model](http://arxiv.org/abs/2111.02724)


  Tea chrysanthemum detection at its flowering stage is one of the key
components for selective chrysanthemum harvesting robot development. However,
it is a challenge to detect flowering chrysanthemums under unstructured field
environments given the variations on illumination, occlusion and object scale.
In this context, we propose a highly fused and lightweight deep learning
architecture based on YOLO for tea chrysanthemum detection (TC-YOLO). First, in
the backbone component and neck component, the method uses the Cross-Stage
Partially Dense Network (CSPDenseNet) as the main network, and embeds custom
feature fusion modules to guide the gradient flow. In the final head component,
the method combines the recursive feature pyramid (RFP) multiscale fusion
reflow structure and the Atrous Spatial Pyramid Pool (ASPP) module with cavity
convolution to achieve the detection task. The resulting model was tested on
300 field images, showing that under the NVIDIA Tesla P100 GPU environment, if
the inference speed is 47.23 FPS for each image (416 * 416), TC-YOLO can
achieve the average precision (AP) of 92.49% on our own tea chrysanthemum
dataset. In addition, this method (13.6M) can be deployed on a single mobile
GPU, and it could be further developed as a perception system for a selective
chrysanthemum harvesting robot in the future.

    

### [[2111.02825] Whistleblower protection in the digital age -- why 'anonymous' is not enough. Towards an interdisciplinary view of ethical dilemmas](http://arxiv.org/abs/2111.02825)


  When technology enters applications and processes with a long tradition of
controversial societal debate, multi-faceted new ethical and legal questions
arise. This paper focusses on the process of whistleblowing, an activity with
large impacts on democracy and business. Computer science can, for the first
time in history, provide for truly anonymous communication. We investigate this
in relation to the values and rights of accountability, fairness and data
protection, focusing on opportunities and limitations of the anonymity that can
be provided computationally; possible consequences of outsourcing
whistleblowing support; and challenges for the interpretation and use of some
relevant laws. We conclude that to address these questions, whistleblowing and
anonymous whistleblowing must rest on three pillars, forming a 'triangle of
whistleblowing protection and incentivisation' that combines anonymity in a
formal and technical sense; whistleblower protection through laws; and
organisational and political error culture.

    

### [[2111.02839] Optimised Playout Implementations for the Ludii General Game System](http://arxiv.org/abs/2111.02839)


  This paper describes three different optimised implementations of playouts,
as commonly used by game-playing algorithms such as Monte-Carlo Tree Search.
Each of the optimised implementations is applicable only to specific sets of
games, based on their rules. The Ludii general game system can automatically
infer, based on a game's description in its general game description language,
whether any optimised implementations are applicable. An empirical evaluation
demonstrates major speedups over a standard implementation, with a median
result of running playouts 5.08 times as fast, over 145 different games in
Ludii for which one of the optimised implementations is applicable.

    

### [[2111.02844] A text autoencoder from transformer for fast encoding language representation](http://arxiv.org/abs/2111.02844)


  In recent years BERT shows apparent advantages and great potential in natural
language processing tasks. However, both training and applying BERT requires
intensive time and resources for computing contextual language representations,
which hinders its universality and applicability. To overcome this bottleneck,
we propose a deep bidirectional language model by using window masking
mechanism at attention layer. This work computes contextual language
representations without random masking as does in BERT and maintains the deep
bidirectional architecture like BERT. To compute the same sentence
representation, our method shows O(n) complexity less compared to other
transformer-based models with O($n^2$). To further demonstrate its superiority,
computing context language representations on CPU environments is conducted, by
using the embeddings from the proposed method, logistic regression shows much
higher accuracy in terms of SMS classification. Moverover, the proposed method
also achieves significant higher performance in semantic similarity tasks.

    

### [[2111.02853] Big Data Testing Techniques: Taxonomy, Challenges and Future Trends](http://arxiv.org/abs/2111.02853)


  Big Data is reforming many industrial domains by providing decision support
through analyzing large volumes of data. Big Data testing aims to ensure that
Big Data systems run smoothly and error-free while maintaining the performance
and quality of data. However, because of the diversity and complexity of data,
testing Big Data is challenging. Though numerous researches deal with Big Data
testing, a comprehensive review to address testing techniques and challenges is
not conflate yet. Therefore, we have conducted a systematic review of the Big
Data testing techniques period (2010 - 2021). This paper discusses the
processing of testing data by highlighting the techniques used in every
processing phase. Furthermore, we discuss the challenges and future directions.
Our finding shows that diverse functional, non-functional and combined
(functional and non-functional) testing techniques have been used to solve
specific problems related to Big Data. At the same time, most of the testing
challenges have been faced during the MapReduce validation phase. In addition,
the combinatorial testing technique is one of the most applied techniques in
combination with other techniques (i.e., random testing, mutation testing,
input space partitioning and equivalence testing) to solve various functional
faults challenges faced during Big Data testing.

    

### [[2111.02859] Large Scale Diverse Combinatorial Optimization: ESPN Fantasy Football Player Trades](http://arxiv.org/abs/2111.02859)


  Even skilled fantasy football managers can be disappointed by their
mid-season rosters as some players inevitably fall short of draft day
expectations. Team managers can quickly discover that their team has a low
score ceiling even if they start their best active players. A novel and diverse
combinatorial optimization system proposes high volume and unique player trades
between complementary teams to balance trade fairness. Several algorithms
create the valuation of each fantasy football player with an ensemble of
computing models: Quantum Support Vector Classifier with Permutation Importance
(QSVC-PI), Quantum Support Vector Classifier with Accumulated Local Effects
(QSVC-ALE), Variational Quantum Circuit with Permutation Importance (VQC-PI),
Hybrid Quantum Neural Network with Permutation Importance (HQNN-PI), eXtreme
Gradient Boosting Classifier (XGB), and Subject Matter Expert (SME) rules. The
valuation of each player is personalized based on league rules, roster, and
selections. The cost of trading away a player is related to a team's roster,
such as the depth at a position, slot count, and position importance. Teams are
paired together for trading based on a cosine dissimilarity score so that teams
can offset their strengths and weaknesses. A knapsack 0-1 algorithm computes
outgoing players for each team. Postprocessors apply analytics and deep
learning models to measure 6 different objective measures about each trade.
Over the 2020 and 2021 National Football League (NFL) seasons, a group of 24
experts from IBM and ESPN evaluated trade quality through 10 Football Error
Analysis Tool (FEAT) sessions. Our system started with 76.9% of high-quality
trades and was deployed for the 2021 season with 97.3% of high-quality trades.
To increase trade quantity, our quantum, classical, and rules-based computing
have 100% trade uniqueness. We use Qiskit's quantum simulators throughout our
work.

    

### [[2111.03048] Imagine Networks](http://arxiv.org/abs/2111.03048)


  In this paper, we introduce an Imagine Network that can simulate itself
through graph tree neural networks. Among the graph tree neural networks
models, association, deduction, and memory networks are learned, and a network
is created by combining the discriminator and reinforcement learning models.
This model can learn various datasets or data samples generated in environments
and generate new data samples.

    

### [[2111.03059] Engagement Decision Support for Beyond Visual Range Air Combat](http://arxiv.org/abs/2111.03059)


  This work aims to provide an engagement decision support tool for Beyond
Visual Range (BVR) air combat in the context of Defensive Counter Air (DCA)
missions. In BVR air combat, engagement decision refers to the choice of the
moment the pilot engages a target by assuming an offensive stance and executing
corresponding maneuvers. To model this decision, we use the Brazilian Air
Force's Aerospace Simulation Environment (\textit{Ambiente de Simulação
Aeroespacial - ASA} in Portuguese), which generated 3,729 constructive
simulations lasting 12 minutes each and a total of 10,316 engagements. We
analyzed all samples by an operational metric called the DCA index, which
represents, based on the experience of subject matter experts, the degree of
success in this type of mission. This metric considers the distances of the
aircraft of the same team and the opposite team, the point of Combat Air
Patrol, and the number of missiles used. By defining the engagement status
right before it starts and the average of the DCA index throughout the
engagement, we create a supervised learning model to determine the quality of a
new engagement. An algorithm based on decision trees, working with the XGBoost
library, provides a regression model to predict the DCA index with a
coefficient of determination close to 0.8 and a Root Mean Square Error of 0.05
that can furnish parameters to the BVR pilot to decide whether or not to
engage. Thus, using data obtained through simulations, this work contributes by
building a decision support system based on machine learning for BVR air
combat.

    

### [[2106.12831] Extraction of common conceptual components from multiple ontologies](http://arxiv.org/abs/2106.12831)


  Understanding large ontologies is still an issue, and has an impact on many
ontology engineering tasks. We describe a novel method for identifying and
extracting conceptual components from domain ontologies, which are used to
understand and compare them. The method is applied to two corpora of ontologies
in the Cultural Heritage and Conference domain, respectively. The results,
which show good quality, are evaluated by manual inspection and by correlation
with datasets and tool performance from the ontology alignment evaluation
initiative.

    

### [[2106.15011] Are conditional GANs explicitly conditional?](http://arxiv.org/abs/2106.15011)


  This paper proposes two important contributions for conditional Generative
Adversarial Networks (cGANs) to improve the wide variety of applications that
exploit this architecture. The first main contribution is an analysis of cGANs
to show that they are not explicitly conditional. In particular, it will be
shown that the discriminator and subsequently the cGAN does not automatically
learn the conditionality between inputs. The second contribution is a new
method, called a contrario cGAN, that explicitly models conditionality for both
parts of the adversarial architecture via a novel a contrario loss that
involves training the discriminator to learn unconditional (adverse) examples.
This leads to a novel type of data augmentation approach for GANs (a contrario
learning) which allows to restrict the search space of the generator to
conditional outputs using adverse examples. Extensive experimentation is
carried out to evaluate the conditionality of the discriminator by proposing a
probability distribution analysis. Comparisons with the cGAN architecture for
different applications show significant improvements in performance on well
known datasets including, semantic image synthesis, image segmentation,
monocular depth prediction and "single label"-to-image using different metrics
including Fréchet Inception Distance (FID), mean Intersection over Union
(mIoU), Root Mean Square Error log (RMSE log) and Number of
statistically-Different Bins (NDB).

    

### [[2111.02938] Source-Level Bitwise Branching for Temporal Verification](http://arxiv.org/abs/2111.02938)


  There is increasing interest in applying verification tools to programs that
have bitvector operations. SMT solvers, which serve as a foundation for these
tools, have thus increased support for bitvector reasoning through bit-blasting
and linear arithmetic approximations. Still, verification tools are limited on
termination and LTL verification of bitvector programs. In this work, we show
that similar linear arithmetic approximation of bitvector operations can be
done at the source level through transformations. Specifically, we introduce
new paths that over-approximate bitvector operations with linear
conditions/constraints, increasing branching but allowing us to better exploit
the well-developed integer reasoning and interpolation of verification tools.
We present two sets of rules, namely rewriting rules and weakening rules, that
can be implemented as bitwise branching of program transformation, the
branching path can facilitate verification tools widen verification tasks over
bitvector programs. Our experiment shows this exploitation of integer reasoning
and interpolation enables competitive termination verification of bitvector
programs and leads to the first effective technique for LTL verification of
bitvector programs.

    