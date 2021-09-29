
## 2021-9-29

### [<title>XGBoost support for Windows 11? - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-support-for-windows-11/2477/2)

### [[2109.12331] Predicting Hidden Links and Missing Nodes in Scale-Free Networks with Artificial Neural Networks](http://arxiv.org/abs/2109.12331)


  There are many networks in real life which exist as form of Scale-free
networks such as World Wide Web, protein-protein inter action network, semantic
networks, airline networks, interbank payment networks, etc. If we want to
analyze these networks, it is really necessary to understand the properties of
scale-free networks. By using the properties of scale free networks, we can
identify any type of anomalies in those networks. In this research, we proposed
a methodology in a form of an algorithm to predict hidden links and missing
nodes in scale-free networks where we combined a generator of random networks
as a source of train data, on one hand, with artificial neural networks for
supervised classification, on the other, we aimed at training the neural
networks to discriminate between different subtypes of scale-free networks and
predicted the missing nodes and hidden links among (present and missing) nodes
in a given scale-free network. We chose Bela Bollobas's directed scale-free
random graph generation algorithm as a generator of random networks to generate
a large set of scale-free network's data.

    

### [[2109.13450] Two-Stage Channel Estimation Approach for Cell-Free IoT With Massive Random Access](http://arxiv.org/abs/2109.13450)


  We investigate the activity detection and channel estimation issues for
cell-free Internet of Things (IoT) networks with massive random access. In each
time slot, only partial devices are active and communicate with neighboring
access points (APs) using non-orthogonal random pilot sequences. Different from
the centralized processing in cellular networks, the activity detection and
channel estimation in cell-free IoT is more challenging due to the distributed
and user-centric architecture. We propose a two-stage approach to detect the
random activities of devices and estimate their channel states. In the first
stage, the activity of each device is jointly detected by its adjacent APs
based on the vector approximate message passing (Vector AMP) algorithm. In the
second stage, each AP re-estimates the channel using the linear minimum mean
square error (LMMSE) method based on the detected activities to improve the
channel estimation accuracy. We derive closed-form expressions for the activity
detection error probability and the mean-squared channel estimation errors for
a typical device. Finally, we analyze the performance of the entire cell-free
IoT network in terms of coverage probability. Simulation results validate the
derived closed-form expressions and show that the cell-free IoT significantly
outperforms the collocated massive MIMO and small-cell schemes in terms of
coverage probability.

    

### [[2109.13677] Frame Replication and Elimination for Reliability in Time-Sensitive Networks](http://arxiv.org/abs/2109.13677)


  In modern applications such as in the prospective smart factory, timely and
faultfree communication is one of the main concerns. Communication failures may
lead to huge economic losses. Moreover, they can even endanger human life.
Therefore, the TimeSensitive Networking (TSN) task group has introduced new
standards for real-time capable Ethernet, which also include a fault tolerance
mechanism called Frame Replication and Elimination for Reliability (FRER) as
IEEE standard 802.1CB. This standard introduces procedures and protocols for
bridges and end stations in time-sensitive networks. It also provides
mechanisms for the identification and duplication of frames to enable redundant
transmissions. In this paper, a simulation model is developed that implements
the IEEE 802.1CB standard in OMNeT++. In addition, as supplement to the
standard we propose a reliability mechanism for establishing redundant paths
and an error model to model transient and permanent errors. As proof of
concept, we evaluate the model with different topologies under various
conditions.

    

### [[2109.13774] PSSPR: A Source Location Privacy Protection Scheme Based on Sector Phantom Routing in WSNs](http://arxiv.org/abs/2109.13774)


  Source location privacy (SLP) protection is an emerging research topic in
wireless sensor networks (WSNs). Because the source location represents the
valuable information of the target being monitored and tracked, it is of great
practical significance to achieve a high degree of privacy of the source
location. Although many studies based on phantom nodes have alleviates the
protection of source location privacy to some extent. It is urgent to solve the
problems such as complicate the ac path between nodes, improve the centralized
distribution of Phantom nodes near the source nodes and reduce the network
communication overhead. In this paper, PSSPR routing is proposed as a visable
approach to address SLP issues. We use the coordinates of the center node V to
divide sector-domain, which act as important role in generating a new phantom
nodes. The phantom nodes perform specified routing policies to ensure that they
can choose various locations. In addition, the directed random route can ensure
that data packets avoid the visible range when they move to the sink node hop
by hop. Thus, the source location is protected. Theoretical analysis and
simulation experiments show that this protocol achieves higher security of
source node location with less communication overhead.

    

### [[2002.05814] Hoplite: Efficient and Fault-Tolerant Collective Communication for Task-Based Distributed Systems](http://arxiv.org/abs/2002.05814)


  Task-based distributed frameworks (e.g., Ray, Dask, Hydro) have become
increasingly popular for distributed applications that contain asynchronous and
dynamic workloads, including asynchronous gradient descent, reinforcement
learning, and model serving. As more data-intensive applications move to run on
top of task-based systems, collective communication efficiency has become an
important problem. Unfortunately, traditional collective communication
libraries (e.g., MPI, Horovod, NCCL) are an ill fit, because they require the
communication schedule to be known before runtime and they do not provide fault
tolerance.
We design and implement Hoplite, an efficient and fault-tolerant collective
communication layer for task-based distributed systems. Our key technique is to
compute data transfer schedules on the fly and execute the schedules
efficiently through fine-grained pipelining. At the same time, when a task
fails, the data transfer schedule adapts quickly to allow other tasks to keep
making progress. We apply Hoplite to a popular task-based distributed
framework, Ray. We show that Hoplite speeds up asynchronous stochastic gradient
descent, reinforcement learning, and serving an ensemble of machine learning
models that are difficult to execute efficiently with traditional collective
communication by up to 7.8x, 3.9x, and 3.3x, respectively.

    

### [[2012.05520] An Overview of 5G System Accessibility Differentiation and Control](http://arxiv.org/abs/2012.05520)


  5G system is characterized by its capability to support a wide range of use
cases and services. Supporting accessibility differentiation becomes therefore
essential to preserve a stable network condition during high traffic load,
while ensuring connection and service quality to prioritized devices and
services. In this article, we describe some use cases and requirements that
impact the 3GPP design of the 5G accessibility differentiation framework. We
then provide an overview of the supported mechanisms for accessibility
differentiation and control in 5G Stand Alone (SA) system, including cell
barring and reservation, unified access control, paging control, random access
control and admission control. We discuss how these functionalities can be used
to maintain the service quality and selectively limit the incoming traffic to
the network at high load situations, leveraging different connection-type
indicators and connection-priority identifiers.

    

### [[2104.12678] Semi-Decentralized Federated Edge Learning for Fast Convergence on Non-IID Data](http://arxiv.org/abs/2104.12678)


  Federated edge learning (FEEL) has emerged as an effective approach to reduce
the large communication latency in Cloud-based machine learning solutions,
while preserving data privacy. Unfortunately, the learning performance of FEEL
may be compromised due to limited training data in a single edge cluster. In
this paper, we investigate a novel framework of FEEL, namely semi-decentralized
federated edge learning (SD-FEEL). By allowing model aggregation across
different edge clusters, SD-FEEL enjoys the benefit of FEEL in reducing the
training latency, while improving the learning performance by accessing richer
training data from multiple edge clusters. A training algorithm for SD-FEEL
with three main procedures in each round is presented, including local model
updates, intra-cluster and inter-cluster model aggregations, which is proved to
converge on non-independent and identically distributed (non-IID) data. We also
characterize the interplay between the network topology of the edge servers and
the communication overhead of inter-cluster model aggregation on the training
performance. Experiment results corroborate our analysis and demonstrate the
effectiveness of SD-FFEL in achieving faster convergence than traditional
federated learning architectures. Besides, guidelines on choosing critical
hyper-parameters of the training algorithm are also provided.

    

### [[2106.04549] KIGLIS: Smart Networks for Smart Cities](http://arxiv.org/abs/2106.04549)


  Smart cities will be characterized by a variety of intelligent and networked
services, each with specific requirements for the underlying network
infrastructure. While smart city architectures and services have been studied
extensively, little attention has been paid to the network technology. The
KIGLIS research project, consisting of a consortium of companies, universities
and research institutions, focuses on artificial intelligence for optimizing
fiber-optic networks of a smart city, with a special focus on future mobility
applications, such as automated driving. In this paper, we present early
results on our process of collecting smart city requirements for communication
networks, which will lead towards reference infrastructure and architecture
solutions. Finally, we suggest directions in which artificial intelligence will
improve smart city networks.

    

### [[2109.13230] The Impact of Domain Shift on Left and Right Ventricle Segmentation in Short Axis Cardiac MR Images](http://arxiv.org/abs/2109.13230)


  Domain shift refers to the difference in the data distribution of two
datasets, normally between the training set and the test set for machine
learning algorithms. Domain shift is a serious problem for generalization of
machine learning models and it is well-established that a domain shift between
the training and test sets may cause a drastic drop in the model's performance.
In medical imaging, there can be many sources of domain shift such as different
scanners or scan protocols, different pathologies in the patient population,
anatomical differences in the patient population (e.g. men vs women) etc.
Therefore, in order to train models that have good generalization performance,
it is important to be aware of the domain shift problem, its potential causes
and to devise ways to address it. In this paper, we study the effect of domain
shift on left and right ventricle blood pool segmentation in short axis cardiac
MR images. Our dataset contains short axis images from 4 different MR scanners
and 3 different pathology groups. The training is performed with nnUNet. The
results show that scanner differences cause a greater drop in performance
compared to changing the pathology group, and that the impact of domain shift
is greater on right ventricle segmentation compared to left ventricle
segmentation. Increasing the number of training subjects increased
cross-scanner performance more than in-scanner performance at small training
set sizes, but this difference in improvement decreased with larger training
set sizes. Training models using data from multiple scanners improved
cross-domain performance.

    

### [[2109.13232] Contributions to Large Scale Bayesian Inference and Adversarial Machine Learning](http://arxiv.org/abs/2109.13232)


  The rampant adoption of ML methodologies has revealed that models are usually
adopted to make decisions without taking into account the uncertainties in
their predictions. More critically, they can be vulnerable to adversarial
examples. Thus, we believe that developing ML systems that take into account
predictive uncertainties and are robust against adversarial examples is a must
for critical, real-world tasks. We start with a case study in retailing. We
propose a robust implementation of the Nerlove-Arrow model using a Bayesian
structural time series model. Its Bayesian nature facilitates incorporating
prior information reflecting the manager's views, which can be updated with
relevant data. However, this case adopted classical Bayesian techniques, such
as the Gibbs sampler. Nowadays, the ML landscape is pervaded with neural
networks and this chapter also surveys current developments in this sub-field.
Then, we tackle the problem of scaling Bayesian inference to complex models and
large data regimes. In the first part, we propose a unifying view of two
different Bayesian inference algorithms, Stochastic Gradient Markov Chain Monte
Carlo (SG-MCMC) and Stein Variational Gradient Descent (SVGD), leading to
improved and efficient novel sampling schemes. In the second part, we develop a
framework to boost the efficiency of Bayesian inference in probabilistic models
by embedding a Markov chain sampler within a variational posterior
approximation. After that, we present an alternative perspective on adversarial
classification based on adversarial risk analysis, and leveraging the scalable
Bayesian approaches from chapter 2. In chapter 4 we turn to reinforcement
learning, introducing Threatened Markov Decision Processes, showing the
benefits of accounting for adversaries in RL while the agent learns.

    

### [[2109.13233] Bayesian Transfer Learning: An Overview of Probabilistic Graphical Models for Transfer Learning](http://arxiv.org/abs/2109.13233)


  Transfer learning where the behavior of extracting transferable knowledge
from the source domain(s) and reusing this knowledge to target domain has
become a research area of great interest in the field of artificial
intelligence. Probabilistic graphical models (PGMs) have been recognized as a
powerful tool for modeling complex systems with many advantages, e.g., the
ability to handle uncertainty and possessing good interpretability. Considering
the success of these two aforementioned research areas, it seems natural to
apply PGMs to transfer learning. However, although there are already some
excellent PGMs specific to transfer learning in the literature, the potential
of PGMs for this problem is still grossly underestimated. This paper aims to
boost the development of PGMs for transfer learning by 1) examining the pilot
studies on PGMs specific to transfer learning, i.e., analyzing and summarizing
the existing mechanisms particularly designed for knowledge transfer; 2)
discussing examples of real-world transfer problems where existing PGMs have
been successfully applied; and 3) exploring several potential research
directions on transfer learning using PGM.

    

### [[2109.13235] Probabilistic modeling of lake surface water temperature using a Bayesian spatio-temporal graph convolutional neural network](http://arxiv.org/abs/2109.13235)


  Accurate lake temperature estimation is essential for numerous problems
tackled in both hydrological and ecological domains. Nowadays physical models
are developed to estimate lake dynamics; however, computations needed for
accurate estimation of lake surface temperature can get prohibitively
expensive. We propose to aggregate simulations of lake temperature at a certain
depth together with a range of meteorological features to probabilistically
estimate lake surface temperature. Accordingly, we introduce a spatio-temporal
neural network that combines Bayesian recurrent neural networks and Bayesian
graph convolutional neural networks. This work demonstrates that the proposed
graphical model can deliver homogeneously good performance covering the whole
lake surface despite having sparse training data available. Quantitative
results are compared with a state-of-the-art Bayesian deep learning method.
Code for the developed architectural layers, as well as demo scripts, are
available on this https URL.

    

### [[2109.13236] FedIPR: Ownership Verification for Federated Deep Neural Network Models](http://arxiv.org/abs/2109.13236)


  Federated learning models must be protected against plagiarism since these
models are built upon valuable training data owned by multiple institutions or
people.This paper illustrates a novel federated deep neural network (FedDNN)
ownership verification scheme that allows ownership signatures to be embedded
and verified to claim legitimate intellectual property rights (IPR) of FedDNN
models, in case that models are illegally copied, re-distributed or misused.
The effectiveness of embedded ownership signatures is theoretically justified
by proved condition sunder which signatures can be embedded and detected by
multiple clients with-out disclosing private signatures. Extensive experimental
results on CIFAR10,CIFAR100 image datasets demonstrate that varying bit-lengths
signatures can be embedded and reliably detected without affecting models
classification performances. Signatures are also robust against removal attacks
including fine-tuning and pruning.

    

### [[2109.13237] DOODLER: Determining Out-Of-Distribution Likelihood from Encoder Reconstructions](http://arxiv.org/abs/2109.13237)


  Deep Learning models possess two key traits that, in combination, make their
use in the real world a risky prospect. One, they do not typically generalize
well outside of the distribution for which they were trained, and two, they
tend to exhibit confident behavior regardless of whether or not they are
producing meaningful outputs. While Deep Learning possesses immense power to
solve realistic, high-dimensional problems, these traits in concert make it
difficult to have confidence in their real-world applications. To overcome this
difficulty, the task of Out-Of-Distribution (OOD) Detection has been defined,
to determine when a model has received an input from outside of the
distribution for which it is trained to operate.
This paper introduces and examines a novel methodology, DOODLER, for OOD
Detection, which directly leverages the traits which result in its necessity.
By training a Variational Auto-Encoder (VAE) on the same data as another Deep
Learning model, the VAE learns to accurately reconstruct In-Distribution (ID)
inputs, but not to reconstruct OOD inputs, meaning that its failure state can
be used to perform OOD Detection. Unlike other work in the area, DOODLER
requires only very weak assumptions about the existence of an OOD dataset,
allowing for more realistic application. DOODLER also enables pixel-wise
segmentations of input images by OOD likelihood, and experimental results show
that it matches or outperforms methodologies that operate under the same
constraints.

    

### [[2109.13247] The edge of chaos: quantum field theory and deep neural networks](http://arxiv.org/abs/2109.13247)


  We explicitly construct the quantum field theory corresponding to a general
class of deep neural networks encompassing both recurrent and feedforward
architectures. We first consider the mean-field theory (MFT) obtained as the
leading saddlepoint in the action, and derive the condition for criticality via
the largest Lyapunov exponent. We then compute the loop corrections to the
correlation function in a perturbative expansion in the ratio of depth $T$ to
width $N$, and find a precise analogy with the well-studied $O(N)$ vector
model, in which the variance of the weight initializations plays the role of
the 't Hooft coupling. In particular, we compute both the $\mathcal{O}(1)$
corrections quantifying fluctuations from typicality in the ensemble of
networks, and the subleading $\mathcal{O}(T/N)$ corrections due to finite-width
effects. These provide corrections to the correlation length that controls the
depth to which information can propagate through the network, and thereby sets
the scale at which such networks are trainable by gradient descent. Our
analysis provides a first-principles approach to the rapidly emerging NN-QFT
correspondence, and opens several interesting avenues to the study of
criticality in deep neural networks.

    

### [[2109.13297] GANG-MAM: GAN based enGine for Modifying Android Malware](http://arxiv.org/abs/2109.13297)


  Malware detectors based on machine learning are vulnerable to adversarial
attacks. Generative Adversarial Networks (GAN) are architectures based on
Neural Networks that could produce successful adversarial samples. The interest
towards this technology is quickly growing. In this paper, we propose a system
that produces a feature vector for making an Android malware strongly evasive
and then modify the malicious program accordingly. Such a system could have a
twofold contribution: it could be used to generate datasets to validate systems
for detecting GAN-based malware and to enlarge the training and testing dataset
for making more robust malware classifiers.

    

### [[2109.13301] Exploring The Role of Local and Global Explanations in Recommender Systems](http://arxiv.org/abs/2109.13301)


  Explanations are well-known to improve recommender systems' transparency.
These explanations may be local, explaining an individual recommendation, or
global, explaining the recommender model in general. Despite their widespread
use, there has been little investigation into the relative benefits of these
two approaches. Do they provide the same benefits to users, or do they serve
different purposes? We conducted a 30-participant exploratory study and a
30-participant controlled user study with a research-paper recommender system
to analyze how providing participants local, global, or both explanations
influences user understanding of system behavior. Our results provide evidence
suggesting that both explanations are more helpful than either alone for
explaining how to improve recommendations, yet both appeared less helpful than
global alone for efficiency in identifying false positives and negatives.
However, we note that the two explanation approaches may be better compared in
the context of a higher-stakes or more opaque domain.

    

### [[2109.13305] ST-MAML: A Stochastic-Task based Method for Task-Heterogeneous Meta-Learning](http://arxiv.org/abs/2109.13305)


  Optimization-based meta-learning typically assumes tasks are sampled from a
single distribution - an assumption oversimplifies and limits the diversity of
tasks that meta-learning can model. Handling tasks from multiple different
distributions is challenging for meta-learning due to a so-called task
ambiguity issue. This paper proposes a novel method, ST-MAML, that empowers
model-agnostic meta-learning (MAML) to learn from multiple task distributions.
ST-MAML encodes tasks using a stochastic neural network module, that summarizes
every task with a stochastic representation. The proposed Stochastic Task (ST)
strategy allows a meta-model to get tailored for the current task and enables
us to learn a distribution of solutions for an ambiguous task. ST-MAML also
propagates the task representation to revise the encoding of input variables.
Empirically, we demonstrate that ST-MAML matches or outperforms the
state-of-the-art on two few-shot image classification tasks, one curve
regression benchmark, one image completion problem, and a real-world
temperature prediction application. To the best of authors' knowledge, this is
the first time optimization-based meta-learning method being applied on a
large-scale real-world task.

    

### [[2109.13306] Introducing the viewpoint in the resource description using machine learning](http://arxiv.org/abs/2109.13306)


  Search engines allow providing the user with data information according to
their interests and specialty. Thus, it is necessary to exploit descriptions of
the resources, which take into consideration viewpoints. Generally, the
resource descriptions are available in RDF (e.g., DBPedia of Wikipedia
content). However, these descriptions do not take into consideration
viewpoints. In this paper, we propose a new approach, which allows converting a
classic RDF resource description to a resource description that takes into
consideration viewpoints. To detect viewpoints in the document, a machine
learning technique will be exploited on an instanced ontology. This latter
allows representing the viewpoint in a given domain. An experimental study
shows that the conversion of the classic RDF resource description to a resource
description that takes into consideration viewpoints, allows giving very
relevant responses to the user's requests.

    

### [[2109.13318] Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end SL Translation](http://arxiv.org/abs/2109.13318)


  Automating sign language translation (SLT) is a challenging real world
application. Despite its societal importance, though, research progress in the
field remains rather poor. Crucially, existing methods that yield viable
performance necessitate the availability of laborious to obtain gloss sequence
groundtruth. In this paper, we attenuate this need, by introducing an
end-to-end SLT model that does not entail explicit use of glosses; the model
only needs text groundtruth. This is in stark contrast to existing end-to-end
models that use gloss sequence groundtruth, either in the form of a modality
that is recognized at an intermediate model stage, or in the form of a parallel
output process, jointly trained with the SLT model. Our approach constitutes a
Transformer network with a novel type of layers that combines: (i) local
winner-takes-all (LWTA) layers with stochastic winner sampling, instead of
conventional ReLU layers, (ii) stochastic weights with posterior distributions
estimated via variational inference, and (iii) a weight compression technique
at inference time that exploits estimated posterior variance to perform
massive, almost lossless compression. We demonstrate that our approach can
reach the currently best reported BLEU-4 score on the PHOENIX 2014T benchmark,
but without making use of glosses for model training, and with a memory
footprint reduced by more than 70%.

    

### [[2109.13333] Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients](http://arxiv.org/abs/2109.13333)


  In this work we are the first to present an offline policy gradient method
for learning imitative policies for complex urban driving from a large corpus
of real-world demonstrations. This is achieved by building a differentiable
data-driven simulator on top of perception outputs and high-fidelity HD maps of
the area. It allows us to synthesize new driving experiences from existing
demonstrations using mid-level representations. Using this simulator we then
train a policy network in closed-loop employing policy gradients. We train our
proposed method on 100 hours of expert demonstrations on urban roads and show
that it learns complex driving policies that generalize well and can perform a
variety of driving maneuvers. We demonstrate this in simulation as well as
deploy our model to self-driving vehicles in the real-world. Our method
outperforms previously demonstrated state-of-the-art for urban driving
scenarios -- all this without the need for complex state perturbations or
collecting additional on-policy data during training. We make code and data
publicly available.

    

### [[2109.13337] DEBOSH: Deep Bayesian Shape Optimization](http://arxiv.org/abs/2109.13337)


  Shape optimization is at the heart of many industrial applications, such as
aerodynamics, heat transfer, and structural analysis. It has recently been
shown that Graph Neural Networks (GNNs) can predict the performance of a shape
quickly and accurately and be used to optimize more effectively than
traditional techniques that rely on response-surfaces obtained by Kriging.
However, GNNs suffer from the fact that they do not evaluate their own
accuracy, which is something Bayesian Optimization methods require. Therefore,
estimating confidence in generated predictions is necessary to go beyond
straight deterministic optimization, which is less effective.
In this paper, we demonstrate that we can use Ensembles-based technique to
overcome this limitation and outperform the state-of-the-art. Our experiments
on diverse aerodynamics and structural analysis tasks prove that adding
uncertainty to shape optimization significantly improves the quality of
resulting shapes and reduces the time required for the optimization.

    

### [[2109.13354] Audio-to-Image Cross-Modal Generation](http://arxiv.org/abs/2109.13354)


  Cross-modal representation learning allows to integrate information from
different modalities into one representation. At the same time, research on
generative models tends to focus on the visual domain with less emphasis on
other domains, such as audio or text, potentially missing the benefits of
shared representations. Studies successfully linking more than one modality in
the generative setting are rare. In this context, we verify the possibility to
train variational autoencoders (VAEs) to reconstruct image archetypes from
audio data. Specifically, we consider VAEs in an adversarial training framework
in order to ensure more variability in the generated data and find that there
is a trade-off between the consistency and diversity of the generated images -
this trade-off can be governed by scaling the reconstruction loss up or down,
respectively. Our results further suggest that even in the case when the
generated images are relatively inconsistent (diverse), features that are
critical for proper image classification are preserved.

    

### [[2109.13359] Lyapunov-Net: A Deep Neural Network Architecture for Lyapunov Function Approximation](http://arxiv.org/abs/2109.13359)


  We develop a versatile deep neural network architecture, called Lyapunov-Net,
to approximate Lyapunov functions of dynamical systems in high dimensions.
Lyapunov-Net guarantees positive definiteness, and thus it can be easily
trained to satisfy the negative orbital derivative condition, which only
renders a single term in the empirical risk function in practice. This
significantly reduces the number of hyper-parameters compared to existing
methods. We also provide theoretical justifications on the approximation power
of Lyapunov-Net and its complexity bounds. We demonstrate the efficiency of the
proposed method on nonlinear dynamical systems involving up to 30-dimensional
state spaces, and show that the proposed approach significantly outperforms the
state-of-the-art methods.

    

### [[2109.13360] IGAN: Inferent and Generative Adversarial Networks](http://arxiv.org/abs/2109.13360)


  I present IGAN (Inferent Generative Adversarial Networks), a neural
architecture that learns both a generative and an inference model on a complex
high dimensional data distribution, i.e. a bidirectional mapping between data
samples and a simpler low-dimensional latent space. It extends the traditional
GAN framework with inference by rewriting the adversarial strategy in both the
image and the latent space with an entangled game between data-latent encoded
posteriors and priors. It brings a measurable stability and convergence to the
classical GAN scheme, while keeping its generative quality and remaining simple
and frugal in order to run on a lab PC. IGAN fosters the encoded latents to
span the full prior space: this enables the exploitation of an enlarged and
self-organised latent space in an unsupervised manner. An analysis of
previously published articles sets the theoretical ground for the proposed
algorithm. A qualitative demonstration of potential applications like
self-supervision or multi-modal data translation is given on common image
datasets including SAR and optical imagery.

    

### [[2109.13361] Graph Neural Network-based Resource AllocationStrategies for Multi-Object Spectroscopy](http://arxiv.org/abs/2109.13361)


  Resource allocation problems are often approached with linear program-ming
techniques. But many concrete allocation problems in the experimental and
ob-servational sciences cannot or should not be expressed in the form of linear
objectivefunctions. Even if the objective is linear, its parameters may not be
known beforehandbecause they depend on the results of the experiment for which
the allocation is to bedetermined. To address these challenges, we present a
bipartite Graph Neural Networkarchitecture for trainable resource allocation
strategies. Items of value and constraintsform the two sets of graph nodes,
which are connected by edges corresponding to pos-sible allocations. The GNN is
trained on simulations or past problem occurrences tomaximize any
user-supplied, scientifically motivated objective function, augmented byan
infeasibility penalty. The amount of feasibility violation can be tuned in
relation toany available slack in the system. We apply this method to optimize
the astronomicaltarget selection strategy for the highly multiplexed Subaru
Prime Focus Spectrographinstrument, where it shows superior results to direct
gradient descent optimization andextends the capabilities of the currently
employed solver which uses linear objectivefunctions. The development of this
method enables fast adjustment and deployment ofallocation strategies,
statistical analyses of allocation patterns, and fully
differentiable,science-driven solutions for resource allocation problems.

    

### [[2109.13373] Trustworthy AI and Robotics and the Implications for the AEC Industry: A Systematic Literature Review and Future Potentials](http://arxiv.org/abs/2109.13373)


  Human-technology interaction deals with trust as an inevitable requirement
for user acceptance. As the applications of artificial intelligence (AI) and
robotics emerge and with their ever-growing socio-economic influence in various
fields of research and practice, there is an imminent need to study trust in
such systems. With the opaque work mechanism of AI-based systems and the
prospect of intelligent robots as workers' companions, context-specific
interdisciplinary studies on trust are key in increasing their adoption.
Through a thorough systematic literature review on (1) trust in AI and robotics
(AIR) and (2) AIR applications in the architecture, engineering, and
construction (AEC) industry, this study identifies common trust dimensions in
the literature and uses them to organize the paper. Furthermore, the
connections of the identified dimensions to the existing and potential AEC
applications are determined and discussed. Finally, major future directions on
trustworthy AI and robotics in AEC research and practice are outlined.

    

### [[2109.13375] Automated Estimation of Construction Equipment Emission using Inertial Sensors and Machine Learning Models](http://arxiv.org/abs/2109.13375)


  The construction industry is one of the main producers of greenhouse gasses
(GHG). Quantifying the amount of air pollutants including GHG emissions during
a construction project has become an additional project objective to
traditional metrics such as time, cost, and safety in many parts of the world.
A major contributor to air pollution during construction is the use of heavy
equipment and thus their efficient operation and management can substantially
reduce the harm to the environment. Although the on-road vehicle emission
prediction is a widely researched topic, construction equipment emission
measurement and reduction have received very little attention. This paper
describes the development and deployment of a novel framework that uses machine
learning (ML) methods to predict the level of emissions from heavy construction
equipment monitored via an Internet of Things (IoT) system comprised of
accelerometer and gyroscope sensors. The developed framework was validated
using an excavator performing real-world construction work. A portable emission
measurement system (PEMS) was employed along with the inertial sensors to
record data including the amount of CO, NOX, CO2, SO2, and CH4 pollutions
emitted by the equipment. Different ML algorithms were developed and compared
to identify the best model to predict emission levels from inertial sensors
data. The results showed that Random Forest with the coefficient of
determination (R2) of 0.94, 0.91 and 0.94 for CO, NOX, CO2, respectively was
the best algorithm among different models evaluated in this study.

    

### [[2109.13388] Go-Blend behavior and affect](http://arxiv.org/abs/2109.13388)


  This paper proposes a paradigm shift for affective computing by viewing the
affect modeling task as a reinforcement learning process. According to our
proposed framework the context (environment) and the actions of an agent define
the common representation that interweaves behavior and affect. To realise this
framework we build on recent advances in reinforcement learning and use a
modified version of the Go-Explore algorithm which has showcased supreme
performance in hard exploration tasks. In this initial study, we test our
framework in an arcade game by training Go-Explore agents to both play
optimally and attempt to mimic human demonstrations of arousal. We vary the
degree of importance between optimal play and arousal imitation and create
agents that can effectively display a palette of affect and behavioral
patterns. Our Go-Explore implementation not only introduces a new paradigm for
affect modeling; it empowers believable AI-based game testing by providing
agents that can blend and express a multitude of behavioral and affective
patterns.

    

### [[2109.13398] Unrolling SGD: Understanding Factors Influencing Machine Unlearning](http://arxiv.org/abs/2109.13398)


  Machine unlearning is the process through which a deployed machine learning
model forgets about one of its training data points. While naively retraining
the model from scratch is an option, it is almost always associated with a
large computational effort for deep learning models. Thus, several approaches
to approximately unlearn have been proposed along with corresponding metrics
that formalize what it means for a model to forget about a data point. In this
work, we first taxonomize approaches and metrics of approximate unlearning. As
a result, we identify verification error, i.e., the L2 difference between the
weights of an approximately unlearned and a naively retrained model, as a
metric approximate unlearning should optimize for as it implies a large class
of other metrics. We theoretically analyze the canonical stochastic gradient
descent (SGD) training algorithm to surface the variables which are relevant to
reducing the verification error of approximate unlearning for SGD. From this
analysis, we first derive an easy-to-compute proxy for verification error
(termed unlearning error). The analysis also informs the design of a new
training objective penalty that limits the overall change in weights during SGD
and as a result facilitates approximate unlearning with lower verification
error. We validate our theoretical work through an empirical evaluation on
CIFAR-10, CIFAR-100, and IMDB sentiment analysis.

    

### [[2109.13412] Discriminative Attribution from Counterfactuals](http://arxiv.org/abs/2109.13412)


  We present a method for neural network interpretability by combining feature
attribution with counterfactual explanations to generate attribution maps that
highlight the most discriminative features between pairs of classes. We show
that this method can be used to quantitatively evaluate the performance of
feature attribution methods in an objective manner, thus preventing potential
observer bias. We evaluate the proposed method on three diverse datasets,
including a challenging artificial dataset and real-world biological data. We
show quantitatively and qualitatively that the highlighted features are
substantially more discriminative than those extracted using conventional
attribution methods and argue that this type of explanation is better suited
for understanding fine grained class differences as learned by a deep neural
network.

    

### [[2109.13417] Interactive Dynamic Walking: Learning Gait Switching Policies with Generalization Guarantees](http://arxiv.org/abs/2109.13417)


  In this paper, we consider the problem of adapting a dynamically walking
bipedal robot to follow a leading co-worker while engaging in tasks that
require physical interaction. Our approach relies on switching among a family
of Dynamic Movement Primitives (DMPs) as governed by a supervisor. We train the
supervisor to orchestrate the switching among the DMPs in order to adapt to the
leader's intentions, which are only implicitly available in the form of
interaction forces. The primary contribution of our approach is its ability to
furnish certificates of generalization to novel leader intentions for the
trained supervisor. This is achieved by leveraging the Probably Approximately
Correct (PAC)-Bayes bounds from generalization theory. We demonstrate the
efficacy of our approach by training a neural-network supervisor to adapt the
gait of a dynamically walking biped to a leading collaborator whose intended
trajectory is not known explicitly.

    

### [[2109.13419] The Role of Lookahead and Approximate Policy Evaluation in Policy Iteration with Linear Value Function Approximation](http://arxiv.org/abs/2109.13419)


  When the sizes of the state and action spaces are large, solving MDPs can be
computationally prohibitive even if the probability transition matrix is known.
So in practice, a number of techniques are used to approximately solve the
dynamic programming problem, including lookahead, approximate policy evaluation
using an m-step return, and function approximation. In a recent paper, (Efroni
et al. 2019) studied the impact of lookahead on the convergence rate of
approximate dynamic programming. In this paper, we show that these convergence
results change dramatically when function approximation is used in conjunction
with lookout and approximate policy evaluation using an m-step return.
Specifically, we show that when linear function approximation is used to
represent the value function, a certain minimum amount of lookahead and
multi-step return is needed for the algorithm to even converge. And when this
condition is met, we characterize the finite-time performance of policies
obtained using such approximate policy iteration. Our results are presented for
two different procedures to compute the function approximation: linear
least-squares regression and gradient descent.

    

### [[2109.13425] The JHU submission to VoxSRC-21: Track 3](http://arxiv.org/abs/2109.13425)


  This technical report describes Johns Hopkins University speaker recognition
system submitted to Voxceleb Speaker Recognition Challenge 2021 Track 3:
Self-supervised speaker verification (closed). Our overall training process is
similar to the proposed one from the first place team in the last year's
VoxSRC2020 challenge. The main difference is a recently proposed
non-contrastive self-supervised method in computer vision (CV), distillation
with no labels (DINO), is used to train our initial model, which outperformed
the last year's contrastive learning based on momentum contrast (MoCo). Also,
this requires only a few iterations in the iterative clustering stage, where
pseudo labels for supervised embedding learning are updated based on the
clusters of the embeddings generated from a model that is continually
fine-tuned over iterations. In the final stage, Res2Net50 is trained on the
final pseudo labels from the iterative clustering stage. This is our best
submitted model to the challenge, showing 1.89, 6.50, and 6.89 in EER(%) in
voxceleb1 test o, VoxSRC-21 validation, and test trials, respectively.

    

### [[2109.13432] Warp-Refine Propagation: Semi-Supervised Auto-labeling via Cycle-consistency](http://arxiv.org/abs/2109.13432)


  Deep learning models for semantic segmentation rely on expensive,
large-scale, manually annotated datasets. Labelling is a tedious process that
can take hours per image. Automatically annotating video sequences by
propagating sparsely labeled frames through time is a more scalable
alternative. In this work, we propose a novel label propagation method, termed
Warp-Refine Propagation, that combines semantic cues with geometric cues to
efficiently auto-label videos. Our method learns to refine geometrically-warped
labels and infuse them with learned semantic priors in a semi-supervised
setting by leveraging cycle consistency across time. We quantitatively show
that our method improves label-propagation by a noteworthy margin of 13.1 mIoU
on the ApolloScape dataset. Furthermore, by training with the auto-labelled
frames, we achieve competitive results on three semantic-segmentation
benchmarks, improving the state-of-the-art by a large margin of 1.8 and 3.61
mIoU on NYU-V2 and KITTI, while matching the current best results on
Cityscapes.

    

### [[2109.13441] DynG2G: An Efficient Stochastic Graph Embedding Method for Temporal Graphs](http://arxiv.org/abs/2109.13441)


  Dynamic graph embedding has gained great attention recently due to its
capability of learning low dimensional graph representations for complex
temporal graphs with high accuracy. However, recent advances mostly focus on
learning node embeddings as deterministic "vectors" for static graphs yet
disregarding the key graph temporal dynamics and the evolving uncertainties
associated with node embedding in the latent space. In this work, we propose an
efficient stochastic dynamic graph embedding method (DynG2G) that applies an
inductive feed-forward encoder trained with node triplet-based contrastive
loss. Every node per timestamp is encoded as a time-dependent probabilistic
multivariate Gaussian distribution in the latent space, hence we can quantify
the node embedding uncertainty on-the-fly. We adopted eight different
benchmarks that represent diversity in size (from 96 nodes to 87,626 and from
13,398 edges to 4,870,863) and diversity in dynamics. We demonstrate via
extensive experiments on these eight dynamic graph benchmarks that DynG2G
achieves new state-of-the-art performance in capturing the underlying temporal
node embeddings. We also demonstrate that DynG2G can predict the evolving node
embedding uncertainty, which plays a crucial role in quantifying the intrinsic
dimensionality of the dynamical system over time. We obtain a universal
relation of the optimal embedding dimension, $L_o$, versus the effective
dimensionality of uncertainty, $D_u$, and we infer that $L_o=D_u$ for all
cases. This implies that the uncertainty quantification approach we employ in
the DynG2G correctly captures the intrinsic dimensionality of the dynamics of
such evolving graphs despite the diverse nature and composition of the graphs
at each timestamp. Moreover, this $L_0 - D_u$ correlation provides a clear path
to select adaptively the optimum embedding size at each timestamp by setting $L
\ge D_u$.

    

### [[2109.13442] An Adaptive Deep Learning Framework for Day-ahead Forecasting of Photovoltaic Power Generation](http://arxiv.org/abs/2109.13442)


  Accurate forecasts of photovoltaic power generation (PVPG) are essential to
optimize operations between energy supply and demand. Recently, the propagation
of sensors and smart meters has produced an enormous volume of data, which
supports the development of data based PVPG forecasting. Although emerging deep
learning (DL) models, such as the long short-term memory (LSTM) model, based on
historical data, have provided effective solutions for PVPG forecasting with
great successes, these models utilize offline learning. As a result, DL models
cannot take advantage of the opportunity to learn from newly-arrived data, and
are unable to handle concept drift caused by installing extra PV units and
unforeseen PV unit failures. Consequently, to improve day-ahead PVPG
forecasting accuracy, as well as eliminate the impacts of concept drift, this
paper proposes an adaptive LSTM (AD-LSTM) model, which is a DL framework that
can not only acquire general knowledge from historical data, but also
dynamically learn specific knowledge from newly-arrived data. A two-phase
adaptive learning strategy (TP-ALS) is integrated into AD-LSTM, and a sliding
window (SDWIN) algorithm is proposed, to detect concept drift in PV systems.
Multiple datasets from PV systems are utilized to assess the feasibility and
effectiveness of the proposed approaches. The developed AD-LSTM model
demonstrates greater forecasting capability than the offline LSTM model,
particularly in the presence of concept drift. Additionally, the proposed
AD-LSTM model also achieves superior performance in terms of day-ahead PVPG
forecasting compared to other traditional machine learning models and
statistical models in the literature.

    

### [[2109.13445] To Which Out-Of-Distribution Object Orientations Are DNNs Capable of Generalizing?](http://arxiv.org/abs/2109.13445)


  The capability of Deep Neural Networks (DNNs) to recognize objects in
orientations outside the distribution of the training data, ie.
out-of-distribution (OoD) orientations, is not well understood. For humans,
behavioral studies showed that recognition accuracy varies across OoD
orientations, where generalization is much better for some orientations than
for others. In contrast, for DNNs, it remains unknown how generalization
abilities are distributed among OoD orientations. In this paper, we investigate
the limitations of DNNs' generalization capacities by systematically inspecting
patterns of success and failure of DNNs across OoD orientations. We use an
intuitive and controlled, yet challenging learning paradigm, in which some
instances of an object category are seen at only a few geometrically restricted
orientations, while other instances are seen at all orientations. The effect of
data diversity is also investigated by increasing the number of instances seen
at all orientations in the training set. We present a comprehensive analysis of
DNNs' generalization abilities and limitations for representative architectures
(ResNet, Inception, DenseNet and CORnet). Our results reveal an intriguing
pattern -- DNNs are only capable of generalizing to instances of objects that
appear like 2D, ie. in-plane, rotations of in-distribution orientations.

    

### [[2109.13448] Lithium-ion Battery State of Health Estimation based on Cycle Synchronization using Dynamic Time Warping](http://arxiv.org/abs/2109.13448)


  The state of health (SOH) estimation plays an essential role in
battery-powered applications to avoid unexpected breakdowns due to battery
capacity fading. However, few studies have paid attention to the problem of
uneven length of degrading cycles, simply employing manual operation or leaving
to the automatic processing mechanism of advanced machine learning models, like
long short-term memory (LSTM). As a result, this causes information loss and
caps the full capability of the data-driven SOH estimation models. To address
this challenge, this paper proposes an innovative cycle synchronization way to
change the existing coordinate system using dynamic time warping, not only
enabling the equal length inputs of the estimation model but also preserving
all information. By exploiting the time information of the time series, the
proposed method embeds the time index and the original measurements into a
novel indicator to reflect the battery degradation status, which could have the
same length over cycles. Adopting the LSTM as the basic estimation model, the
cycle synchronization-based SOH model could significantly improve the
prediction accuracy by more than 30% compared to the traditional LSTM.

    

### [[2109.13449] When in Doubt: Improving Classification Performance with Alternating Normalization](http://arxiv.org/abs/2109.13449)


  We introduce Classification with Alternating Normalization (CAN), a
non-parametric post-processing step for classification. CAN improves
classification accuracy for challenging examples by re-adjusting their
predicted class probability distribution using the predicted class
distributions of high-confidence validation examples. CAN is easily applicable
to any probabilistic classifier, with minimal computation overhead. We analyze
the properties of CAN using simulated experiments, and empirically demonstrate
its effectiveness across a diverse set of classification tasks.

    

### [[2109.13452] CateCom: a practical data-centric approach to categorization of computational models](http://arxiv.org/abs/2109.13452)


  The advent of data-driven science in the 21st century brought about the need
for well-organized structured data and associated infrastructure able to
facilitate the applications of Artificial Intelligence and Machine Learning. We
present an effort aimed at organizing the diverse landscape of physics-based
and data-driven computational models in order to facilitate the storage of
associated information as structured data. We apply object-oriented design
concepts and outline the foundations of an open-source collaborative framework
that is: (1) capable of uniquely describing the approaches in structured data,
(2) flexible enough to cover the majority of widely used models, and (3)
utilizes collective intelligence through community contributions. We present
example database schemas and corresponding data structures and explain how
these are deployed in software at the time of this writing.

    

### [[2109.13459] Multiwavelet-based Operator Learning for Differential Equations](http://arxiv.org/abs/2109.13459)


  The solution of a partial differential equation can be obtained by computing
the inverse operator map between the input and the solution space. Towards this
end, we introduce a \textit{multiwavelet-based neural operator learning scheme}
that compresses the associated operator's kernel using fine-grained wavelets.
By explicitly embedding the inverse multiwavelet filters, we learn the
projection of the kernel onto fixed multiwavelet polynomial bases. The
projected kernel is trained at multiple scales derived from using repeated
computation of multiwavelet transform. This allows learning the complex
dependencies at various scales and results in a resolution-independent scheme.
Compare to the prior works, we exploit the fundamental properties of the
operator's kernel which enable numerically efficient representation. We perform
experiments on the Korteweg-de Vries (KdV) equation, Burgers' equation, Darcy
Flow, and Navier-Stokes equation. Compared with the existing neural operator
approaches, our model shows significantly higher accuracy and achieves
state-of-the-art in a range of datasets. For the time-varying equations, the
proposed method exhibits a ($2X-10X$) improvement ($0.0018$ ($0.0033$) relative
$L2$ error for Burgers' (KdV) equation). By learning the mappings between
function spaces, the proposed method has the ability to find the solution of a
high-resolution input after learning from lower-resolution data.

    

### [[2109.13463] Deep Reinforcement Learning with Adjustments](http://arxiv.org/abs/2109.13463)


  Deep reinforcement learning (RL) algorithms can learn complex policies to
optimize agent operation over time. RL algorithms have shown promising results
in solving complicated problems in recent years. However, their application on
real-world physical systems remains limited. Despite the advancements in RL
algorithms, the industries often prefer traditional control strategies.
Traditional methods are simple, computationally efficient and easy to adjust.
In this paper, we first propose a new Q-learning algorithm for continuous
action space, which can bridge the control and RL algorithms and bring us the
best of both worlds. Our method can learn complex policies to achieve long-term
goals and at the same time it can be easily adjusted to address short-term
requirements without retraining. Next, we present an approximation of our
algorithm which can be applied to address short-term requirements of any
pre-trained RL algorithm. The case studies demonstrate that both our proposed
method as well as its practical approximation can achieve short-term and
long-term goals without complex reward functions.

    

### [[2109.13466] Delve into the Performance Degradation of Differentiable Architecture Search](http://arxiv.org/abs/2109.13466)


  Differentiable architecture search (DARTS) is widely considered to be easy to
overfit the validation set which leads to performance degradation. We first
employ a series of exploratory experiments to verify that neither high-strength
architecture parameters regularization nor warmup training scheme can
effectively solve this problem. Based on the insights from the experiments, we
conjecture that the performance of DARTS does not depend on the well-trained
supernet weights and argue that the architecture parameters should be trained
by the gradients which are obtained in the early stage rather than the final
stage of training. This argument is then verified by exchanging the learning
rate schemes of weights and parameters. Experimental results show that the
simple swap of the learning rates can effectively solve the degradation and
achieve competitive performance. Further empirical evidence suggests that the
degradation is not a simple problem of the validation set overfitting but
exhibit some links between the degradation and the operation selection bias
within bilevel optimization dynamics. We demonstrate the generalization of this
bias and propose to utilize this bias to achieve an operation-magnitude-based
selective stop.

    

### [[2109.13471] An Automated Approach to Causal Inference in Discrete Settings](http://arxiv.org/abs/2109.13471)


  When causal quantities cannot be point identified, researchers often pursue
partial identification to quantify the range of possible values. However, the
peculiarities of applied research conditions can make this analytically
intractable. We present a general and automated approach to causal inference in
discrete settings. We show causal questions with discrete data reduce to
polynomial programming problems, and we present an algorithm to automatically
bound causal effects using efficient dual relaxation and spatial
branch-and-bound techniques. The user declares an estimand, states assumptions,
and provides data (however incomplete or mismeasured). The algorithm then
searches over admissible data-generating processes and outputs the most precise
possible range consistent with available information -- i.e., sharp bounds --
including a point-identified solution if one exists. Because this search can be
computationally intensive, our procedure reports and continually refines
non-sharp ranges that are guaranteed to contain the truth at all times, even
when the algorithm is not run to completion. Moreover, it offers an additional
guarantee we refer to as $\epsilon$-sharpness, characterizing the worst-case
looseness of the incomplete bounds. Analytically validated simulations show the
algorithm accommodates classic obstacles, including confounding, selection,
measurement error, noncompliance, and nonresponse.

    

### [[2109.13477] Exploring More When It Needs in Deep Reinforcement Learning](http://arxiv.org/abs/2109.13477)


  We propose a exploration mechanism of policy in Deep Reinforcement Learning,
which is exploring more when agent needs, called Add Noise to Noise (AN2N). The
core idea is: when the Deep Reinforcement Learning agent is in a state of poor
performance in history, it needs to explore more. So we use cumulative rewards
to evaluate which past states the agents have not performed well, and use
cosine distance to measure whether the current state needs to be explored more.
This method shows that the exploration mechanism of the agent's policy is
conducive to efficient exploration. We combining the proposed exploration
mechanism AN2N with Deep Deterministic Policy Gradient (DDPG), Soft
Actor-Critic (SAC) algorithms, and apply it to the field of continuous control
tasks, such as halfCheetah, Hopper, and Swimmer, achieving considerable
improvement in performance and convergence speed.

    

### [[2109.13483] Metal Artifact Reduction in 2D CT Images with Self-supervised Cross-domain Learning](http://arxiv.org/abs/2109.13483)


  The presence of metallic implants often introduces severe metal artifacts in
the X-ray CT images, which could adversely influence clinical diagnosis or dose
calculation in radiation therapy. In this work, we present a novel
deep-learning-based approach for metal artifact reduction (MAR). In order to
alleviate the need for anatomically identical CT image pairs (i.e., metal
artifact-corrupted CT image and metal artifact-free CT image) for network
learning, we propose a self-supervised cross-domain learning framework.
Specifically, we train a neural network to restore the metal trace region
values in the given metal-free sinogram, where the metal trace is identified by
the forward projection of metal masks. We then design a novel FBP
reconstruction loss to encourage the network to generate more perfect
completion results and a residual-learning-based image refinement module to
reduce the secondary artifacts in the reconstructed CT images. To preserve the
fine structure details and fidelity of the final MAR image, instead of directly
adopting CNN-refined images as output, we incorporate the metal trace
replacement into our framework and replace the metal-affected projections of
the original sinogram with the prior sinogram generated by the forward
projection of the CNN output. We then use the filtered backward projection
(FBP) algorithms for final MAR image reconstruction. We conduct an extensive
evaluation on simulated and real artifact data to show the effectiveness of our
design. Our method produces superior MAR results and outperforms other
compelling methods. We also demonstrate the potential of our framework for
other organ sites.

    

### [[2109.13497] Instance-Based Neural Dependency Parsing](http://arxiv.org/abs/2109.13497)


  Interpretable rationales for model predictions are crucial in practical
applications. We develop neural models that possess an interpretable inference
process for dependency parsing. Our models adopt instance-based inference,
where dependency edges are extracted and labeled by comparing them to edges in
a training set. The training edges are explicitly used for the predictions;
thus, it is easy to grasp the contribution of each edge to the predictions. Our
experiments show that our instance-based models achieve competitive accuracy
with standard neural models and have the reasonable plausibility of
instance-based explanations.

    

### [[2109.13498] Learning to Superoptimize Real-world Programs](http://arxiv.org/abs/2109.13498)


  Program optimization is the process of modifying software to execute more
efficiently. Because finding the optimal program is generally undecidable,
modern compilers usually resort to expert-written heuristic optimizations. In
contrast, superoptimizers attempt to find the optimal program by employing
significantly more expensive search and constraint solving techniques.
Generally, these methods do not scale well to programs in real development
scenarios, and as a result superoptimization has largely been confined to
small-scale, domain-specific, and/or synthetic program benchmarks. In this
paper, we propose a framework to learn to superoptimize real-world programs by
using neural sequence-to-sequence models. We introduce the Big Assembly
benchmark, a dataset consisting of over 25K real-world functions mined from
open-source projects in x86-64 assembly, which enables experimentation on
large-scale optimization of real-world programs. We propose an approach, Self
Imitation Learning for Optimization (SILO) that is easy to implement and
outperforms a standard policy gradient learning approach on our Big Assembly
benchmark. Our method, SILO, superoptimizes programs an expected 6.2% of our
test set when compared with the gcc version 10.3 compiler's aggressive
optimization level -O3. We also report that SILO's rate of superoptimization on
our test set is over five times that of a standard policy gradient approach and
a model pre-trained on compiler optimization demonstration.

    

### [[2109.13510] VoxCeleb Enrichment for Age and Gender Recognition](http://arxiv.org/abs/2109.13510)


  VoxCeleb datasets are widely used in speaker recognition studies. Our work
serves two purposes. First, we provide speaker age labels and (an alternative)
annotation of speaker gender. Second, we demonstrate the use of this metadata
by constructing age and gender recognition models with different features and
classifiers. We query different celebrity databases and apply consensus rules
to derive age and gender labels. We also compare the original VoxCeleb gender
labels with our labels to identify records that might be mislabeled in the
original VoxCeleb data. On modeling side, we design a comprehensive study of
multiple features and models for recognizing gender and age. Our best system,
using i-vector features, achieved an F1-score of 0.9829 for gender recognition
task using logistic regression, and the lowest mean absolute error (MAE) in age
regression, 9.443 years, is obtained with ridge regression. This indicates
challenge in age estimation from in-the-wild style speech data.

    

### [[2109.13514] Convolutional Shapelet Transform: A new approach for time series shapelets](http://arxiv.org/abs/2109.13514)


  Shapelet-based algorithms are widely used for time series classification
because of their ease of interpretation, but they are currently outperformed,
notably by methods using convolutional kernels, capable of reaching
state-of-the-art performance while being highly scalable. We present a new
formulation of time series shapelets including the notion of dilation, and a
shapelet extraction method based on convolutional kernels, which is able to
target the discriminant information identified by convolutional kernels.
Experiments performed on 108 datasets show that our method improves on the
state-of-the-art for shapelet algorithms, and we show that it can be used to
interpret results from convolutional kernels.

    

### [[2109.13521] A multi-stage semi-supervised improved deep embedded clustering (MS-SSIDEC) method for bearing fault diagnosis under the situation of insufficient labeled samples](http://arxiv.org/abs/2109.13521)


  Intelligent data-driven fault diagnosis methods have been widely applied, but
most of these methods need a large number of high-quality labeled samples. It
costs a lot of labor and time to label data in actual industrial processes,
which challenges the application of intelligent fault diagnosis methods. To
solve this problem, a multi-stage semi-supervised improved deep embedded
clustering (MS-SSIDEC) method is proposed for the bearing fault diagnosis under
the insufficient labeled samples situation. This method includes three stages:
pre-training, deep clustering and enhanced supervised learning. In the first
stage, a skip-connection based convolutional auto-encoder (SCCAE) is proposed
and pre-trained to automatically learn low-dimensional representations. In the
second stage, a semi-supervised improved deep embedded clustering (SSIDEC)
model that integrates the pre-trained auto-encoder with a clustering layer is
proposed for deep clustering. Additionally, virtual adversarial training (VAT)
is introduced as a regularization term to overcome the overfitting in the
model's training. In the third stage, high-quality clustering results obtained
in the second stage are assigned to unlabeled samples as pseudo labels. The
labeled dataset is augmented by those pseudo-labeled samples and used to train
a bearing fault discriminative model. The effectiveness of the method is
evaluated on the Case Western Reserve University (CWRU) bearing dataset. The
results show that the method can not only satisfy the semi-supervised learning
under a small number of labeled samples, but also solve the problem of
unsupervised learning, and has achieved better results than traditional
diagnosis methods. This method provides a new research idea for fault diagnosis
with limited labeled samples by effectively using unsupervised data.

    

### [[2109.13527] Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation](http://arxiv.org/abs/2109.13527)


  Recently, micro-video sharing platforms such as Kuaishou and Tiktok have
become a major source of information for people's lives. Thanks to the large
traffic volume, short video lifespan and streaming fashion of these services,
it has become more and more pressing to improve the existing recommender
systems to accommodate these challenges in a cost-effective way. In this paper,
we propose a novel concept-aware denoising graph neural network (named CONDE)
for micro-video recommendation. CONDE consists of a three-phase graph
convolution process to derive user and micro-video representations: warm-up
propagation, graph denoising and preference refinement. A heterogeneous
tripartite graph is constructed by connecting user nodes with video nodes, and
video nodes with associated concept nodes, extracted from captions and comments
of the videos. To address the noisy information in the graph, we introduce a
user-oriented graph denoising phase to extract a subgraph which can better
reflect the user's preference. Despite the main focus of micro-video
recommendation in this paper, we also show that our method can be generalized
to other types of tasks. Therefore, we also conduct empirical studies on a
well-known public E-commerce dataset. The experimental results suggest that the
proposed CONDE achieves significantly better recommendation performance than
the existing state-of-the-art solutions.

    

### [[2109.13542] Convergence of Deep Convolutional Neural Networks](http://arxiv.org/abs/2109.13542)


  Convergence of deep neural networks as the depth of the networks tends to
infinity is fundamental in building the mathematical foundation for deep
learning. In a previous study, we investigated this question for deep ReLU
networks with a fixed width. This does not cover the important convolutional
neural networks where the widths are increasing from layer to layer. For this
reason, we first study convergence of general ReLU networks with increasing
widths and then apply the results obtained to deep convolutional neural
networks. It turns out the convergence reduces to convergence of infinite
products of matrices with increasing sizes, which has not been considered in
the literature. We establish sufficient conditions for convergence of such
infinite products of matrices. Based on the conditions, we present sufficient
conditions for piecewise convergence of general deep ReLU networks with
increasing widths, and as well as pointwise convergence of deep ReLU
convolutional neural networks.

    

### [[2109.13570] Adaptive Informative Path Planning Using Deep Reinforcement Learning for UAV-based Active Sensing](http://arxiv.org/abs/2109.13570)


  Aerial robots are increasingly being utilized for a wide range of
environmental monitoring and exploration tasks. However, a key challenge is
efficiently planning paths to maximize the information value of acquired data
as an initially unknown environment is explored. To address this, we propose a
new approach for informative path planning (IPP) based on deep reinforcement
learning (RL). Bridging the gap between recent advances in RL and robotic
applications, our method combines Monte Carlo tree search with an
offline-learned neural network predicting informative sensing actions. We
introduce several components making our approach applicable for robotic tasks
with continuous high-dimensional state spaces and large action spaces. By
deploying the trained network during a mission, our method enables
sample-efficient online replanning on physical platforms with limited
computational resources. Evaluations using synthetic data show that our
approach performs on par with existing information-gathering methods while
reducing runtime by a factor of 8-10. We validate the performance of our
framework using real-world surface temperature data from a crop field.

    

### [[2109.13576] Multimodality in Meta-Learning: A Comprehensive Survey](http://arxiv.org/abs/2109.13576)


  Meta-learning has gained wide popularity as a training framework that is more
data-efficient than traditional machine learning methods. However, its
generalization ability in complex task distributions, such as multimodal tasks,
has not been thoroughly studied. Recently, some studies on multimodality-based
meta-learning have emerged. This survey provides a comprehensive overview of
the multimodality-based meta-learning landscape in terms of the methodologies
and applications. We first formalize the definition of meta-learning and
multimodality, along with the research challenges in this growing field, such
as how to enrich the input in few-shot or zero-shot scenarios and how to
generalize the models to new tasks. We then propose a new taxonomy to
systematically discuss typical meta-learning algorithms combined with
multimodal tasks. We investigate the contributions of related papers and
summarize them by our taxonomy. Finally, we propose potential research
directions for this promising field.

    

### [[2109.13588] Making Curiosity Explicit in Vision-based RL](http://arxiv.org/abs/2109.13588)


  Vision-based reinforcement learning (RL) is a promising technique to solve
control tasks involving images as the main observation. State-of-the-art RL
algorithms still struggle in terms of sample efficiency, especially when using
image observations. This has led to an increased attention on integrating state
representation learning (SRL) techniques into the RL pipeline. Work in this
field demonstrates a substantial improvement in sample efficiency among other
benefits. However, to take full advantage of this paradigm, the quality of
samples used for training plays a crucial role. More importantly, the diversity
of these samples could affect the sample efficiency of vision-based RL, but
also its generalization capability. In this work, we present an approach to
improve the sample diversity. Our method enhances the exploration capability of
the RL algorithms by taking advantage of the SRL setup. Our experiments show
that the presented approach outperforms the baseline for all tested
environments. These results are most apparent for environments where the
baseline method struggles. Even in simple environments, our method stabilizes
the training, reduces the reward variance and boosts sample efficiency.

    

### [[2109.13589] Learning Ideological Embeddings from Information Cascades](http://arxiv.org/abs/2109.13589)


  Modeling information cascades in a social network through the lenses of the
ideological leaning of its users can help understanding phenomena such as
misinformation propagation and confirmation bias, and devising techniques for
mitigating their toxic effects.
In this paper we propose a stochastic model to learn the ideological leaning
of each user in a multidimensional ideological space, by analyzing the way
politically salient content propagates. In particular, our model assumes that
information propagates from one user to another if both users are interested in
the topic and ideologically aligned with each other. To infer the parameters of
our model, we devise a gradient-based optimization procedure maximizing the
likelihood of an observed set of information cascades. Our experiments on
real-world political discussions on Twitter and Reddit confirm that our model
is able to learn the political stance of the social media users in a
multidimensional ideological space.

    

### [[2109.13595] The Fragility of Optimized Bandit Algorithms](http://arxiv.org/abs/2109.13595)


  Much of the literature on optimal design of bandit algorithms is based on
minimization of expected regret. It is well known that designs that are optimal
over certain exponential families can achieve expected regret that grows
logarithmically in the number of arm plays, at a rate governed by the
Lai-Robbins lower bound. In this paper, we show that when one uses such
optimized designs, the associated algorithms necessarily have the undesirable
feature that the tail of the regret distribution behaves like that of a
truncated Cauchy distribution. Furthermore, for $p>1$, the $p$'th moment of the
regret distribution grows much faster than poly-logarithmically, in particular
as a power of the number of sub-optimal arm plays. We show that optimized
Thompson sampling and UCB bandit designs are also fragile, in the sense that
when the problem is even slightly mis-specified, the regret can grow much
faster than the conventional theory suggests. Our arguments are based on
standard change-of-measure ideas, and indicate that the most likely way that
regret becomes larger than expected is when the optimal arm returns
below-average rewards in the first few arm plays that make that arm appear to
be sub-optimal, thereby causing the algorithm to sample a truly sub-optimal arm
much more than would be optimal.

    

### [[2109.13596] Exploratory State Representation Learning](http://arxiv.org/abs/2109.13596)


  Not having access to compact and meaningful representations is known to
significantly increase the complexity of reinforcement learning (RL). For this
reason, it can be useful to perform state representation learning (SRL) before
tackling RL tasks. However, obtaining a good state representation can only be
done if a large diversity of transitions is observed, which can require a
difficult exploration, especially if the environment is initially reward-free.
To solve the problems of exploration and SRL in parallel, we propose a new
approach called XSRL (eXploratory State Representation Learning). On one hand,
it jointly learns compact state representations and a state transition
estimator which is used to remove unexploitable information from the
representations. On the other hand, it continuously trains an inverse model,
and adds to the prediction error of this model a $k$-step learning progress
bonus to form the maximization objective of a discovery policy. This results in
a policy that seeks complex transitions from which the trained models can
effectively learn. Our experimental results show that the approach leads to
efficient exploration in challenging environments with image observations, and
to state representations that significantly accelerate learning in RL tasks.

    

### [[2109.13602] SafetyNet: Safe planning for real-world self-driving vehicles using machine-learned policies](http://arxiv.org/abs/2109.13602)


  In this paper we present the first safe system for full control of
self-driving vehicles trained from human demonstrations and deployed in
challenging, real-world, urban environments. Current industry-standard
solutions use rule-based systems for planning. Although they perform reasonably
well in common scenarios, the engineering complexity renders this approach
incompatible with human-level performance. On the other hand, the performance
of machine-learned (ML) planning solutions can be improved by simply adding
more exemplar data. However, ML methods cannot offer safety guarantees and
sometimes behave unpredictably. To combat this, our approach uses a simple yet
effective rule-based fallback layer that performs sanity checks on an ML
planner's decisions (e.g. avoiding collision, assuring physical feasibility).
This allows us to leverage ML to handle complex situations while still assuring
the safety, reducing ML planner-only collisions by 95%. We train our ML planner
on 300 hours of expert driving demonstrations using imitation learning and
deploy it along with the fallback layer in downtown San Francisco, where it
takes complete control of a real vehicle and navigates a wide variety of
challenging urban driving scenarios.

    

### [[2109.13604] Real-Time Glaucoma Detection from Digital Fundus Images using Self-ONNs](http://arxiv.org/abs/2109.13604)


  Glaucoma leads to permanent vision disability by damaging the optical nerve
that transmits visual images to the brain. The fact that glaucoma does not show
any symptoms as it progresses and cannot be stopped at the later stages, makes
it critical to be diagnosed in its early stages. Although various deep learning
models have been applied for detecting glaucoma from digital fundus images, due
to the scarcity of labeled data, their generalization performance was limited
along with high computational complexity and special hardware requirements. In
this study, compact Self-Organized Operational Neural Networks (Self- ONNs) are
proposed for early detection of glaucoma in fundus images and their performance
is compared against the conventional (deep) Convolutional Neural Networks
(CNNs) over three benchmark datasets: ACRIMA, RIM-ONE, and ESOGU. The
experimental results demonstrate that Self-ONNs not only achieve superior
detection performance but can also significantly reduce the computational
complexity making it a potentially suitable network model for biomedical
datasets especially when the data is scarce.

    

### [[2109.13610] Confusion-based rank similarity filters for computationally-efficient machine learning on high dimensional data](http://arxiv.org/abs/2109.13610)


  We introduce a novel type of computationally efficient artificial neural
network (ANN) called the rank similarity filter (RSF). RSFs can be used to both
transform and classify nonlinearly separable datasets with many data points and
dimensions. The weights of RSF are set using the rank orders of features in a
data point, or optionally the 'confusion' adjusted ranks between features
(determined from their distributions in the dataset). The activation strength
of a filter determines its similarity to other points in the dataset, a measure
related to cosine similarity. The activation of many RSFs maps samples into a
new nonlinear space suitable for linear classification (the rank similarity
transform (RST)). We additionally used this method to create the nonlinear rank
similarity classifier (RSC), which is a fast and accurate multiclass
classifier, and the nonlinear rank similarity probabilistic classifier (RSPC),
which is an extension to the multilabel case. We evaluated the classifiers on
multiple datasets and RSC was competitive with existing classifiers but with
superior computational efficiency. Open-source code for RST, RSC and RSPC was
written in Python using the popular scikit-learn framework to make it easily
accessible. In future extensions the algorithm can be applied to specialised
hardware suitable for the parallelization of an ANN (GPU) and a Spiking Neural
Network (neuromorphic computing) with corresponding performance gains. This
makes RSF a promising solution to the problem of efficient analysis of
nonlinearly separable data.

    

### [[2109.13644] Clustering to the Fewest Clusters Under Intra-Cluster Dissimilarity Constraints](http://arxiv.org/abs/2109.13644)


  This paper introduces the equiwide clustering problem, where valid partitions
must satisfy intra-cluster dissimilarity constraints. Unlike most existing
clustering algorithms, equiwide clustering relies neither on density nor on a
predefined number of expected classes, but on a dissimilarity threshold. Its
main goal is to ensure an upper bound on the error induced by ultimately
replacing any object with its cluster representative. Under this constraint, we
then primarily focus on minimizing the number of clusters, along with potential
sub-objectives. We argue that equiwide clustering is a sound clustering
problem, and discuss its relationship with other optimization problems,
existing and novel implementations as well as approximation strategies. We
review and evaluate suitable clustering algorithms to identify trade-offs
between the various practical solutions for this clustering problem.

    

### [[2109.13662] DeepPSL: End-to-end perception and reasoning with applications to zero shot learning](http://arxiv.org/abs/2109.13662)


  We introduce DeepPSL a variant of Probabilistic Soft Logic (PSL) to produce
an end-to-end trainable system that integrates reasoning and perception. PSL
represents first-order logic in terms of a convex graphical model -- Hinge Loss
Markov random fields (HL-MRFs). PSL stands out among probabilistic logic
frameworks due to its tractability having been applied to systems of more than
1 billion ground rules. The key to our approach is to represent predicates in
first-order logic using deep neural networks and then to approximately
back-propagate through the HL-MRF and thus train every aspect of the
first-order system being represented. We believe that this approach represents
an interesting direction for the integration of deep learning and reasoning
techniques with applications to knowledge base learning, multi-task learning,
and explainability. We evaluate DeepPSL on a zero shot learning problem in
image classification. State of the art results demonstrate the utility and
flexibility of our approach.

    

### [[2109.13672] Improved prediction rule ensembling through model-based data generation](http://arxiv.org/abs/2109.13672)


  Prediction rule ensembles (PRE) provide interpretable prediction models with
relatively high accuracy.PRE obtain a large set of decision rules from a
(boosted) decision tree ensemble, and achieves sparsitythrough application of
Lasso-penalized regression. This article examines the use of surrogate modelsto
improve performance of PRE, wherein the Lasso regression is trained with the
help of a massivedataset generated by the (boosted) decision tree ensemble.
This use of model-based data generationmay improve the stability and
consistency of the Lasso step, thus leading to improved overallperformance. We
propose two surrogacy approaches, and evaluate them on simulated and
existingdatasets, in terms of sparsity and predictive accuracy. The results
indicate that the use of surrogacymodels can substantially improve the sparsity
of PRE, while retaining predictive accuracy, especiallythrough the use of a
nested surrogacy approach.

    

### [[2109.13675] FlowVocoder: A small Footprint Neural Vocoder based Normalizing flow for Speech Synthesis](http://arxiv.org/abs/2109.13675)


  Recently, non-autoregressive neural vocoders have provided remarkable
performance in generating high-fidelity speech and have been able to produce
synthetic speech in real-time. However, non-autoregressive neural vocoders such
as WaveGlow are far behind autoregressive neural vocoders like WaveFlow in
terms of modeling audio signals due to their limitation in expressiveness. In
addition, though NanoFlow is a state-of-the-art autoregressive neural vocoder
that has immensely small parameters, its performance is marginally lower than
WaveFlow. Therefore, in this paper, we propose a new type of autoregressive
neural vocoder called FlowVocoder, which has a small memory footprint and is
able to generate high-fidelity audio in real-time. Our proposed model improves
the expressiveness of flow blocks by operating a mixture of Cumulative
Distribution Function(CDF) for bipartite transformation. Hence, the proposed
model is capable of modeling waveform signals as well as WaveFlow, while its
memory footprint is much smaller thanWaveFlow. As shown in experiments,
FlowVocoder achieves competitive results with baseline methods in terms of both
subjective and objective evaluation, also, it is more suitable for real-time
text-to-speech applications.

    

### [[2109.13685] Machine learning methods for prediction of cancer driver genes: a survey paper](http://arxiv.org/abs/2109.13685)


  Identifying the genes and mutations that drive the emergence of tumors is a
major step to improve understanding of cancer and identify new directions for
disease diagnosis and treatment. Despite the large volume of genomics data, the
precise detection of driver mutations and their carrying genes, known as cancer
driver genes, from the millions of possible somatic mutations remains a
challenge. Computational methods play an increasingly important role in
identifying genomic patterns associated with cancer drivers and developing
models to predict driver events. Machine learning (ML) has been the engine
behind many of these efforts and provides excellent opportunities for tackling
remaining gaps in the field. Thus, this survey aims to perform a comprehensive
analysis of ML-based computational approaches to identify cancer driver
mutations and genes, providing an integrated, panoramic view of the broad data
and algorithmic landscape within this scientific problem. We discuss how the
interactions among data types and ML algorithms have been explored in previous
solutions and outline current analytical limitations that deserve further
attention from the scientific community. We hope that by helping readers become
more familiar with significant developments in the field brought by ML, we may
inspire new researchers to address open problems and advance our knowledge
towards cancer driver discovery.

    

### [[2109.13696] Improving Time Series Classification Algorithms Using Octave-Convolutional Layers](http://arxiv.org/abs/2109.13696)


  Deep learning models utilizing convolution layers have achieved
state-of-the-art performance on univariate time series classification tasks. In
this work, we propose improving CNN based time series classifiers by utilizing
Octave Convolutions (OctConv) to outperform themselves. These network
architectures include Fully Convolutional Networks (FCN), Residual Neural
Networks (ResNets), LSTM-Fully Convolutional Networks (LSTM-FCN), and Attention
LSTM-Fully Convolutional Networks (ALSTM-FCN). The proposed layers
significantly improve each of these models with minimally increased network
parameters. In this paper, we experimentally show that by substituting
convolutions with OctConv, we significantly improve accuracy for time series
classification tasks for most of the benchmark datasets. In addition, the
updated ALSTM-OctFCN performs statistically the same as the top two time series
classifers, TS-CHIEF and HIVE-COTE (both ensemble models). To further explore
the impact of the OctConv layers, we perform ablation tests of the augmented
model compared to their base model.

    

### [[2109.13698] Anomaly Detection for High-Dimensional Data Using Large Deviations Principle](http://arxiv.org/abs/2109.13698)


  Most current anomaly detection methods suffer from the curse of
dimensionality when dealing with high-dimensional data. We propose an anomaly
detection algorithm that can scale to high-dimensional data using concepts from
the theory of large deviations. The proposed Large Deviations Anomaly Detection
(LAD) algorithm is shown to outperform state of art anomaly detection methods
on a variety of large and high-dimensional benchmark data sets. Exploiting the
ability of the algorithm to scale to high-dimensional data, we propose an
online anomaly detection method to identify anomalies in a collection of
multivariate time series. We demonstrate the applicability of the online
algorithm in identifying counties in the United States with anomalous trends in
terms of COVID-19 related cases and deaths. Several of the identified anomalous
counties correlate with counties with documented poor response to the COVID
pandemic.

    

### [[2109.13705] Opportunistic Implicit User Authentication for Health-Tracking IoT Wearables](http://arxiv.org/abs/2109.13705)


  With the advancement of technologies, market wearables are becoming
increasingly popular with a range of services, including providing access to
bank accounts, accessing cars, monitoring patients remotely, among several
others. However, often these wearables collect various sensitive personal
information of a user with no to limited authentication, e.g., knowledge-based
external authentication techniques, such as PINs. While most of these external
authentication techniques suffer from multiple limitations, including recall
burden, human errors, or biases, researchers have started using various
physiological and behavioral data, such as gait and heart rate, collected by
the wearables to authenticate a wearable user implicitly with a limited
accuracy due to sensing and computing constraints of wearables. In this work,
we explore the usefulness of blood oxygen saturation SpO2 values collected from
the Oximeter device to distinguish a user from others. From a cohort of 25
subjects, we find that 92% of the cases SpO2 can distinguish pairs of users.
From detailed modeling and performance analysis, we observe that while SpO2
alone can obtain an average accuracy of 0.69 and F1 score of 0.69, the addition
of heart rate (HR) can improve the average identification accuracy by 15% and
F1 score by 13%. These results show promise in using SpO2 along with other
biometrics to develop implicit continuous authentications for wearables.

    

### [[2109.13723] Analyzing the Use of Character-Level Translation with Sparse and Noisy Datasets](http://arxiv.org/abs/2109.13723)


  This paper provides an analysis of character-level machine translation models
used in pivot-based translation when applied to sparse and noisy datasets, such
as crowdsourced movie subtitles. In our experiments, we find that such
character-level models cut the number of untranslated words by over 40% and are
especially competitive (improvements of 2-3 BLEU points) in the case of limited
training data. We explore the impact of character alignment, phrase table
filtering, bitext size and the choice of pivot language on translation quality.
We further compare cascaded translation models to the use of synthetic training
data via multiple pivots, and we find that the latter works significantly
better. Finally, we demonstrate that neither word-nor character-BLEU correlate
perfectly with human judgments, due to BLEU's sensitivity to length.

    

### [[2109.13724] Translating from Morphologically Complex Languages: A Paraphrase-Based Approach](http://arxiv.org/abs/2109.13724)


  We propose a novel approach to translating from a morphologically complex
language. Unlike previous research, which has targeted word inflections and
concatenations, we focus on the pairwise relationship between morphologically
related words, which we treat as potential paraphrases and handle using
paraphrasing techniques at the word, phrase, and sentence level. An important
advantage of this framework is that it can cope with derivational morphology,
which has so far remained largely beyond the capabilities of statistical
machine translation systems. Our experiments translating from Malay, whose
morphology is mostly derivational, into English show significant improvements
over rivaling approaches based on five automatic evaluation measures (for
320,000 sentence pairs; 9.5 million English word tokens).

    

### [[2109.13725] Sentiment Analysis in Twitter for Macedonian](http://arxiv.org/abs/2109.13725)


  We present work on sentiment analysis in Twitter for Macedonian. As this is
pioneering work for this combination of language and genre, we created suitable
resources for training and evaluating a system for sentiment analysis of
Macedonian tweets. In particular, we developed a corpus of tweets annotated
with tweet-level sentiment polarity (positive, negative, and neutral), as well
as with phrase-level sentiment, which we made freely available for research
purposes. We further bootstrapped several large-scale sentiment lexicons for
Macedonian, motivated by previous work for English. The impact of several
different pre-processing steps as well as of various features is shown in
experiments that represent the first attempt to build a system for sentiment
analysis in Twitter for the morphologically rich Macedonian language. Overall,
our experimental results show an F1-score of 92.16, which is very strong and is
on par with the best results for English, which were achieved in recent SemEval
competitions.

    

### [[2109.13726] Exposing Paid Opinion Manipulation Trolls](http://arxiv.org/abs/2109.13726)


  Recently, Web forums have been invaded by opinion manipulation trolls. Some
trolls try to influence the other users driven by their own convictions, while
in other cases they can be organized and paid, e.g., by a political party or a
PR agency that gives them specific instructions what to write. Finding paid
trolls automatically using machine learning is a hard task, as there is no
enough training data to train a classifier; yet some test data is possible to
obtain, as these trolls are sometimes caught and widely exposed. In this paper,
we solve the training data problem by assuming that a user who is called a
troll by several different people is likely to be such, and one who has never
been called a troll is unlikely to be such. We compare the profiles of (i) paid
trolls vs. (ii)"mentioned" trolls vs. (iii) non-trolls, and we further show
that a classifier trained to distinguish (ii) from (iii) does quite well also
at telling apart (i) from (iii).

    

### [[2109.13732] IRMAC: Interpretable Refined Motifs and Binary Classification for Rooftops PV Owners](http://arxiv.org/abs/2109.13732)


  In this paper, we seek to identify residential rooftop solar PV owners using
imported energy data. To solve this problem with an interpretable, fast,
secure, and maintainable solution, we propose Interpretable Refined Motifs And
binary Classification (IRMAC) method, which includes a shape-based
dimensionality reduction technique we call Refined Motif (RM), and a
classification technique with linear complexity to identify solar owners.
Furthermore, with the real data from Australia and Denmark, the proposed method
is tested and verified in identifying PV owners as well as identifying
electrical heating system users. The performances of the proposed method is
studied and compared with various of state-of-the-art methods, where the
proposed method outperformed the alternatives.

    

### [[2109.13736] Multi-Task Triplet Loss for Named Entity Recognition using Supplementary Text](http://arxiv.org/abs/2109.13736)


  Retail item data contains many different forms of text like the title of an
item, the description of an item, item name and reviews. It is of interest to
identify the item name in the other forms of text using a named entity tagger.
However, the title of an item and its description are syntactically different
(but semantically similar) in that the title is not necessarily a well formed
sentence while the description is made up of well formed sentences. In this
work, we use a triplet loss to contrast the embeddings of the item title with
the description to establish a proof of concept. We find that using the triplet
loss in a multi-task NER algorithm improves both the precision and recall by a
small percentage. While the improvement is small, we think it is a step in the
right direction of using various forms of text in a multi-task algorithm. In
addition to precision and recall, the multi task triplet loss method is also
found to significantly improve the exact match accuracy i.e. the accuracy of
tagging the entire set of tokens in the text with correct tags.

    

### [[2109.13742] Adaptive Attribute and Structure Subspace Clustering Network](http://arxiv.org/abs/2109.13742)


  Deep self-expressiveness-based subspace clustering methods have demonstrated
effectiveness. However, existing works only consider the attribute information
to conduct the self-expressiveness, which may limit the clustering performance.
In this paper, we propose a novel adaptive attribute and structure subspace
clustering network (AASSC-Net) to simultaneously consider the attribute and
structure information in an adaptive graph fusion manner. Specifically, we
first exploit an auto-encoder to represent input data samples with latent
features for the construction of an attribute matrix. We also construct a mixed
signed and symmetric structure matrix to capture the local geometric structure
underlying data samples. Then, we perform self-expressiveness on the
constructed attribute and structure matrices to learn their affinity graphs
separately. Finally, we design a novel attention-based fusion module to
adaptively leverage these two affinity graphs to construct a more
discriminative affinity graph. Extensive experimental results on commonly used
benchmark datasets demonstrate that our AASSC-Net significantly outperforms
state-of-the-art methods. In addition, we conduct comprehensive ablation
studies to discuss the effectiveness of the designed modules. The code will be
publicly available at this https URL.

    

### [[2109.13745] Meta-aprendizado para otimizacao de parametros de redes neurais](http://arxiv.org/abs/2109.13745)


  The optimization of Artificial Neural Networks (ANNs) is an important task to
the success of using these models in real-world applications. The solutions
adopted to this task are expensive in general, involving trial-and-error
procedures or expert knowledge which are not always available. In this work, we
investigated the use of meta-learning to the optimization of ANNs.
Meta-learning is a research field aiming to automatically acquiring knowledge
which relates features of the learning problems to the performance of the
learning algorithms. The meta-learning techniques were originally proposed and
evaluated to the algorithm selection problem and after to the optimization of
parameters for Support Vector Machines. However, meta-learning can be adopted
as a more general strategy to optimize ANN parameters, which motivates new
efforts in this research direction. In the current work, we performed a case
study using meta-learning to choose the number of hidden nodes for MLP
networks, which is an important parameter to be defined aiming a good networks
performance. In our work, we generated a base of meta-examples associated to 93
regression problems. Each meta-example was generated from a regression problem
and stored: 16 features describing the problem (e.g., number of attributes and
correlation among the problem attributes) and the best number of nodes for this
problem, empirically chosen from a range of possible values. This set of
meta-examples was given as input to a meta-learner which was able to predict
the best number of nodes for new problems based on their features. The
experiments performed in this case study revealed satisfactory results.

    

### [[2109.13748] Stable training of autoencoders for hyperspectral unmixing](http://arxiv.org/abs/2109.13748)


  Neural networks, autoencoders in particular, are one of the most promising
solutions for unmixing hyperspectral data, i.e. reconstructing the spectra of
observed substances (endmembers) and their relative mixing fractions
(abundances). Unmixing is needed for effective hyperspectral analysis and
classification. However, as we show in this paper, the training of autoencoders
for unmixing is highly dependent on weights initialisation. Some sets of
weights lead to degenerate or low performance solutions, introducing negative
bias in expected performance. In this work we present the results of
experiments investigating autoencoders' stability, verifying the dependence of
reconstruction error on initial weights and exploring conditions needed for
successful optimisation of autoencoder parameters.

    

### [[2109.13751] StereoSpike: Depth Learning with a Spiking Neural Network](http://arxiv.org/abs/2109.13751)


  Depth estimation is an important computer vision task, useful in particular
for navigation in autonomous vehicles, or for object manipulation in robotics.
Here we solved it using an end-to-end neuromorphic approach, combining two
event-based cameras and a Spiking Neural Network (SNN) with a slightly modified
U-Net-like encoder-decoder architecture, that we named StereoSpike. More
specifically, we used the Multi Vehicle Stereo Event Camera Dataset (MVSEC). It
provides a depth ground-truth, which was used to train StereoSpike in a
supervised manner, using surrogate gradient descent. We propose a novel readout
paradigm to obtain a dense analog prediction -- the depth of each pixel -- from
the spikes of the decoder. We demonstrate that this architecture generalizes
very well, even better than its non-spiking counterparts, leading to
state-of-the-art test accuracy. To the best of our knowledge, it is the first
time that such a large-scale regression problem is solved by a fully spiking
network. Finally, we show that low firing rates (<10%) can be obtained via
regularization, with a minimal cost in accuracy. This means that StereoSpike
could be efficiently implemented on neuromorphic chips, opening the door for
low power and real time embedded systems.

    

### [[2109.13754] Deep Generative Modeling for Protein Design](http://arxiv.org/abs/2109.13754)


  Deep learning approaches have produced substantial breakthroughs in fields
such as image classification and natural language processing and are making
rapid inroads in the area of protein design. Many generative models of proteins
have been developed that encompass all known protein sequences, model specific
protein families, or extrapolate the dynamics of individual proteins. Those
generative models can learn protein representations that are often more
informative of protein structure and function than hand-engineered features.
Furthermore, they can be used to quickly propose millions of novel proteins
that resemble the native counterparts in terms of expression level, stability,
or other attributes. The protein design process can further be guided by
discriminative oracles to select candidates with the highest probability of
having the desired properties. In this review, we discuss five classes of
generative models that have been most successful at modeling proteins and
provide a framework for model guided protein design.

    

### [[2109.13777] Macroeconomic forecasting with LSTM and mixed frequency time series data](http://arxiv.org/abs/2109.13777)


  This paper demonstrates the potentials of the long short-term memory (LSTM)
when applyingwith macroeconomic time series data sampled at different
frequencies. We first present how theconventional LSTM model can be adapted to
the time series observed at mixed frequencies when thesame mismatch ratio is
applied for all pairs of low-frequency output and higher-frequency variable.
Togeneralize the LSTM to the case of multiple mismatch ratios, we adopt the
unrestricted Mixed DAtaSampling (U-MIDAS) scheme (Foroni et al., 2015) into the
LSTM architecture. We assess via bothMonte Carlo simulations and empirical
application the out-of-sample predictive performance. Ourproposed models
outperform the restricted MIDAS model even in a set up favorable to the
MIDASestimator. For real world application, we study forecasting a quarterly
growth rate of Thai realGDP using a vast array of macroeconomic indicators both
quarterly and monthly. Our LSTM withU-MIDAS scheme easily beats the simple
benchmark AR(1) model at all horizons, but outperformsthe strong benchmark
univariate LSTM only at one and six months ahead. Nonetheless, we find thatour
proposed model could be very helpful in the period of large economic downturns
for short-termforecast. Simulation and empirical results seem to support the
use of our proposed LSTM withU-MIDAS scheme to nowcasting application.

    

### [[2109.13786] Near-Linear Time Algorithm with Near-Logarithmic Regret Per Switch for Mixable/Exp-Concave Losses](http://arxiv.org/abs/2109.13786)


  We investigate the problem of online learning, which has gained significant
attention in recent years due to its applicability in a wide range of fields
from machine learning to game theory. Specifically, we study the online
optimization of mixable loss functions with logarithmic static regret in a
dynamic environment. The best dynamic estimation sequence that we compete
against is selected in hindsight with full observation of the loss functions
and is allowed to select different optimal estimations in different time
intervals (segments). We propose an online mixture framework that uses these
static solvers as the base algorithm. We show that with the suitable selection
of hyper-expert creations and weighting strategies, we can achieve logarithmic
and squared logarithmic regret per switch in quadratic and linearithmic
computational complexity, respectively. For the first time in literature, we
show that it is also possible to achieve near-logarithmic regret per switch
with sub-polynomial complexity per time. Our results are guaranteed to hold in
a strong deterministic sense in an individual sequence manner.

    

### [[2109.13811] An Efficient Epileptic Seizure Detection Technique using Discrete Wavelet Transform and Machine Learning Classifiers](http://arxiv.org/abs/2109.13811)


  This paper presents an epilepsy detection method based on discrete wavelet
transform (DWT) and Machine learning classifiers. Here DWT has been used for
feature extraction as it provides a better decomposition of the signals in
different frequency bands. At first, DWT has been applied to the EEG signal to
extract the detail and approximate coefficients or different sub-bands. After
the extraction of the coefficients, principal component analysis (PCA) has been
applied on different sub-bands and then a feature level fusion technique is
used to extract the important features in low dimensional feature space. Three
classifiers namely: Support Vector Machine (SVM) classifier, K-Nearest-Neighbor
(KNN) classifier, and Naive Bayes (NB) Classifiers have been used in the
proposed work for classifying the EEG signals. The proposed method is tested on
Bonn databases and provides a maximum of 100% recognition accuracy for KNN,
SVM, NB classifiers.

    

### [[2109.13814] Text2Brain: Synthesis of Brain Activation Maps from Free-form Text Query](http://arxiv.org/abs/2109.13814)


  Most neuroimaging experiments are under-powered, limited by the number of
subjects and cognitive processes that an individual study can investigate.
Nonetheless, over decades of research, neuroscience has accumulated an
extensive wealth of results. It remains a challenge to digest this growing
knowledge base and obtain new insights since existing meta-analytic tools are
limited to keyword queries. In this work, we propose Text2Brain, a neural
network approach for coordinate-based meta-analysis of neuroimaging studies to
synthesize brain activation maps from open-ended text queries. Combining a
transformer-based text encoder and a 3D image generator, Text2Brain was trained
on variable-length text snippets and their corresponding activation maps
sampled from 13,000 published neuroimaging studies. We demonstrate that
Text2Brain can synthesize anatomically-plausible neural activation patterns
from free-form textual descriptions of cognitive concepts. Text2Brain is
available at this https URL as a web-based tool for retrieving
established priors and generating new hypotheses for neuroscience research.

    

### [[2109.13818] EEG Signal Processing using Wavelets for Accurate Seizure Detection through Cost Sensitive Data Mining](http://arxiv.org/abs/2109.13818)


  Epilepsy is one of the most common and yet diverse set of chronic
neurological disorders. This excessive or synchronous neuronal activity is
termed seizure. Electroencephalogram signal processing plays a significant role
in detection and prediction of epileptic seizures. In this paper we introduce
an approach that relies upon the properties of wavelets for seizure detection.
We utilise the Maximum Overlap Discrete Wavelet Transform which enables us to
reduce signal noise Then from the variance exhibited in wavelet coefficients we
develop connectivity and communication efficiency between the electrodes as
these properties differ significantly during a seizure period in comparison to
a non-seizure period. We use basic statistical parameters derived from the
reconstructed noise reduced signal, electrode connectivity and the efficiency
of information transfer to build the attribute space.
We have utilised data that are publicly available to test our method that is
found to be significantly better than some existing approaches.

    

### [[2109.13821] Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme](http://arxiv.org/abs/2109.13821)


  Voice conversion is a common speech synthesis task which can be solved in
different ways depending on a particular real-world scenario. The most
challenging one often referred to as one-shot many-to-many voice conversion
consists in copying the target voice from only one reference utterance in the
most general case when both source and target speakers do not belong to the
training dataset. We present a scalable high-quality solution based on
diffusion probabilistic modeling and demonstrate its superior quality compared
to state-of-the-art one-shot voice conversion approaches. Moreover, focusing on
real-time applications, we investigate general principles which can make
diffusion models faster while keeping synthesis quality at a high level. As a
result, we develop a novel Stochastic Differential Equations solver suitable
for various diffusion model types and generative tasks as shown through
empirical studies and justify it by theoretical analysis.

    

### [[2109.13825] What to Prioritize? Natural Language Processing for the Development of a Modern Bug Tracking Solution in Hardware Development](http://arxiv.org/abs/2109.13825)


  Managing large numbers of incoming bug reports and finding the most critical
issues in hardware development is time consuming, but crucial in order to
reduce development costs. In this paper, we present an approach to predict the
time to fix, the risk and the complexity of debugging and resolution of a bug
report using different supervised machine learning algorithms, namely Random
Forest, Naive Bayes, SVM, MLP and XGBoost. Further, we investigate the effect
of the application of active learning and we evaluate the impact of different
text representation techniques, namely TF-IDF, Word2Vec, Universal Sentence
Encoder and XLNet on the model's performance. The evaluation shows that a
combination of text embeddings generated through the Universal Sentence Encoder
and MLP as classifier outperforms all other methods, and is well suited to
predict the risk and complexity of bug tickets.

    

### [[2109.13828] An Automated Data Engineering Pipeline for Anomaly Detection of IoT Sensor Data](http://arxiv.org/abs/2109.13828)


  The rapid development in the field of System of Chip (SoC) technology,
Internet of Things (IoT), cloud computing, and artificial intelligence has
brought more possibilities of improving and solving the current problems. With
data analytics and the use of machine learning/deep learning, it is made
possible to learn the underlying patterns and make decisions based on what was
learned from massive data generated from IoT sensors. When combined with cloud
computing, the whole pipeline can be automated, and free of manual controls and
operations. In this paper, an implementation of an automated data engineering
pipeline for anomaly detection of IoT sensor data is studied and proposed. The
process involves the use of IoT sensors, Raspberry Pis, Amazon Web Services
(AWS) and multiple machine learning techniques with the intent to identify
anomalous cases for the smart home security system.

    

### [[2109.13841] Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation](http://arxiv.org/abs/2109.13841)


  We tackle real-world long-horizon robot manipulation tasks through skill
discovery. We present a bottom-up approach to learning a library of reusable
skills from unsegmented demonstrations and use these skills to synthesize
prolonged robot behaviors. Our method starts with constructing a hierarchical
task structure from each demonstration through agglomerative clustering. From
the task structures of multi-task demonstrations, we identify skills based on
the recurring patterns and train goal-conditioned sensorimotor policies with
hierarchical imitation learning. Finally, we train a meta controller to compose
these skills to solve long-horizon manipulation tasks. The entire model can be
trained on a small set of human demonstrations collected within 30 minutes
without further annotations, making it amendable to real-world deployment. We
systematically evaluated our method in simulation environments and on a real
robot. Our method has shown superior performance over state-of-the-art
imitation learning methods in multi-stage manipulation tasks. Furthermore,
skills discovered from multi-task demonstrations boost the average task success
by $8\%$ compared to those discovered from individual tasks.

    

### [[2109.13851] Reinforcement Learning for Quantitative Trading](http://arxiv.org/abs/2109.13851)


  Quantitative trading (QT), which refers to the usage of mathematical models
and data-driven techniques in analyzing the financial market, has been a
popular topic in both academia and financial industry since 1970s. In the last
decade, reinforcement learning (RL) has garnered significant interest in many
domains such as robotics and video games, owing to its outstanding ability on
solving complex sequential decision making problems. RL's impact is pervasive,
recently demonstrating its ability to conquer many challenging QT tasks. It is
a flourishing research direction to explore RL techniques' potential on QT
tasks. This paper aims at providing a comprehensive survey of research efforts
on RL-based methods for QT tasks. More concretely, we devise a taxonomy of
RL-based QT models, along with a comprehensive summary of the state of the art.
Finally, we discuss current challenges and propose future research directions
in this exciting field.

    

### [[2109.13858] Domain Generalization for Vision-based Driving Trajectory Generation](http://arxiv.org/abs/2109.13858)


  One of the challenges in vision-based driving trajectory generation is
dealing with out-of-distribution scenarios. In this paper, we propose a domain
generalization method for vision-based driving trajectory generation for
autonomous vehicles in urban environments, which can be seen as a solution to
extend the Invariant Risk Minimization (IRM) method in complex problems. We
leverage an adversarial learning approach to train a trajectory generator as
the decoder. Based on the pre-trained decoder, we infer the latent variables
corresponding to the trajectories, and pre-train the encoder by regressing the
inferred latent variable. Finally, we fix the decoder but fine-tune the encoder
with the final trajectory loss. We compare our proposed method with the
state-of-the-art trajectory generation method and some recent domain
generalization methods on both datasets and simulation, demonstrating that our
method has better generalization ability.

    

### [[2109.13862] 3N-GAN: Semi-Supervised Classification of X-Ray Images with a 3-Player Adversarial Framework](http://arxiv.org/abs/2109.13862)


  The success of deep learning for medical imaging tasks, such as
classification, is heavily reliant on the availability of large-scale datasets.
However, acquiring datasets with large quantities of labeled data is
challenging, as labeling is expensive and time-consuming. Semi-supervised
learning (SSL) is a growing alternative to fully-supervised learning, but
requires unlabeled samples for training. In medical imaging, many datasets lack
unlabeled data entirely, so SSL can't be conventionally utilized. We propose
3N-GAN, or 3 Network Generative Adversarial Networks, to perform
semi-supervised classification of medical images in fully-supervised settings.
We incorporate a classifier into the adversarial relationship such that the
generator trains adversarially against both the classifier and discriminator.
Our preliminary results show improved classification performance and GAN
generations over various algorithms. Our work can seamlessly integrate with
numerous other medical imaging model architectures and SSL methods for greater
performance.

    

### [[2109.13863] A First-Occupancy Representation for Reinforcement Learning](http://arxiv.org/abs/2109.13863)


  Both animals and artificial agents benefit from state representations that
support rapid transfer of learning across tasks and which enable them to
efficiently traverse their environments to reach rewarding states. The
successor representation (SR), which measures the expected cumulative,
discounted state occupancy under a fixed policy, enables efficient transfer to
different reward structures in an otherwise constant Markovian environment and
has been hypothesized to underlie aspects of biological behavior and neural
activity. However, in the real world, rewards may move or only be available for
consumption once, may shift location, or agents may simply aim to reach goal
states as rapidly as possible without the constraint of artificially imposed
task horizons. In such cases, the most behaviorally-relevant representation
would carry information about when the agent was likely to first reach states
of interest, rather than how often it should expect to visit them over a
potentially infinite time span. To reflect such demands, we introduce the
first-occupancy representation (FR), which measures the expected temporal
discount to the first time a state is accessed. We demonstrate that the FR
facilitates the selection of efficient paths to desired states, allows the
agent, under certain conditions, to plan provably optimal trajectories defined
by a sequence of subgoals, and induces similar behavior to animals avoiding
threatening stimuli.

    

### [[2109.13889] A PAC-Bayesian Analysis of Distance-Based Classifiers: Why Nearest-Neighbour works!](http://arxiv.org/abs/2109.13889)


  Abstract We present PAC-Bayesian bounds for the generalisation error of the
K-nearest-neighbour classifier (K-NN). This is achieved by casting the K-NN
classifier into a kernel space framework in the limit of vanishing kernel
bandwidth. We establish a relation between prior measures over the coefficients
in the kernel expansion and the induced measure on the weight vectors in kernel
space. Defining a sparse prior over the coefficients allows the application of
a PAC-Bayesian folk theorem that leads to a generalisation bound that is a
function of the number of redundant training examples: those that can be left
out without changing the solution. The presented bound requires to quantify a
prior belief in the sparseness of the solution and is evaluated after learning
when the actual redundancy level is known. Even for small sample size (m ~ 100)
the bound gives non-trivial results when both the expected sparseness and the
actual redundancy are high.

    

### [[2109.13891] Gaussian Processes to speed up MCMC with automatic exploratory-exploitation effect](http://arxiv.org/abs/2109.13891)


  We present a two-stage Metropolis-Hastings algorithm for sampling
probabilistic models, whose log-likelihood is computationally expensive to
evaluate, by using a surrogate Gaussian Process (GP) model. The key feature of
the approach, and the difference w.r.t. previous works, is the ability to learn
the target distribution from scratch (while sampling), and so without the need
of pre-training the GP. This is fundamental for automatic and inference in
Probabilistic Programming Languages In particular, we present an alternative
first stage acceptance scheme by marginalising out the GP distributed function,
which makes the acceptance ratio explicitly dependent on the variance of the
GP. This approach is extended to Metropolis-Adjusted Langevin algorithm (MALA).

    

### [[2109.13895] Symbolic Regression by Exhaustive Search: Reducing the Search Space Using Syntactical Constraints and Efficient Semantic Structure Deduplication](http://arxiv.org/abs/2109.13895)


  Symbolic regression is a powerful system identification technique in
industrial scenarios where no prior knowledge on model structure is available.
Such scenarios often require specific model properties such as
interpretability, robustness, trustworthiness and plausibility, that are not
easily achievable using standard approaches like genetic programming for
symbolic regression. In this chapter we introduce a deterministic symbolic
regression algorithm specifically designed to address these issues. The
algorithm uses a context-free grammar to produce models that are parameterized
by a non-linear least squares local optimization procedure. A finite
enumeration of all possible models is guaranteed by structural restrictions as
well as a caching mechanism for detecting semantically equivalent solutions.
Enumeration order is established via heuristics designed to improve search
efficiency. Empirical tests on a comprehensive benchmark suite show that our
approach is competitive with genetic programming in many noiseless problems
while maintaining desirable properties such as simple, reliable models and
reproducibility.

    

### [[2109.13898] Cluster Analysis of a Symbolic Regression Search Space](http://arxiv.org/abs/2109.13898)


  In this chapter we take a closer look at the distribution of symbolic
regression models generated by genetic programming in the search space. The
motivation for this work is to improve the search for well-fitting symbolic
regression models by using information about the similarity of models that can
be precomputed independently from the target function. For our analysis, we use
a restricted grammar for uni-variate symbolic regression models and generate
all possible models up to a fixed length limit. We identify unique models and
cluster them based on phenotypic as well as genotypic similarity. We find that
phenotypic similarity leads to well-defined clusters while genotypic similarity
does not produce a clear clustering. By mapping solution candidates visited by
GP to the enumerated search space we find that GP initially explores the whole
search space and later converges to the subspace of highest quality expressions
in a run for a simple benchmark problem.

    

### [[2109.13899] A Contrastive Learning Approach to Auroral Identification and Classification](http://arxiv.org/abs/2109.13899)


  Unsupervised learning algorithms are beginning to achieve accuracies
comparable to their supervised counterparts on benchmark computer vision tasks,
but their utility for practical applications has not yet been demonstrated. In
this work, we present a novel application of unsupervised learning to the task
of auroral image classification. Specifically, we modify and adapt the Simple
framework for Contrastive Learning of Representations (SimCLR) algorithm to
learn representations of auroral images in a recently released auroral image
dataset constructed using image data from Time History of Events and Macroscale
Interactions during Substorms (THEMIS) all-sky imagers. We demonstrate that (a)
simple linear classifiers fit to the learned representations of the images
achieve state-of-the-art classification performance, improving the
classification accuracy by almost 10 percentage points over the current
benchmark; and (b) the learned representations naturally cluster into more
clusters than exist manually assigned categories, suggesting that existing
categorizations are overly coarse and may obscure important connections between
auroral types, near-earth solar wind conditions, and geomagnetic disturbances
at the earth's surface. Moreover, our model is much lighter than the previous
benchmark on this dataset, requiring in the area of fewer than 25\% of the
number of parameters. Our approach exceeds an established threshold for
operational purposes, demonstrating readiness for deployment and utilization.

    

### [[2109.13901] Physics-Augmented Learning: A New Paradigm Beyond Physics-Informed Learning](http://arxiv.org/abs/2109.13901)


  Integrating physical inductive biases into machine learning can improve model
generalizability. We generalize the successful paradigm of physics-informed
learning (PIL) into a more general framework that also includes what we term
physics-augmented learning (PAL). PIL and PAL complement each other by handling
discriminative and generative properties, respectively. In numerical
experiments, we show that PAL performs well on examples where PIL is
inapplicable or inefficient.

    

### [[2109.13905] Intra-Day Price Simulation with Generative Adversarial Modelling of the Order Flow](http://arxiv.org/abs/2109.13905)


  Intra-day price variations in financial markets are driven by the sequence of
orders, called the order flow, that is submitted at high frequency by traders.
This paper introduces a novel application of the Sequence Generative
Adversarial Networks framework to model the order flow, such that random
sequences of the order flow can then be generated to simulate the intra-day
variation of prices. As a benchmark, a well-known parametric model from the
quantitative finance literature is selected. The models are fitted, and then
multiple random paths of the order flow sequences are sampled from each model.
Model performances are then evaluated by using the generated sequences to
simulate price variations, and we compare the empirical regularities between
the price variations produced by the generated and real sequences. The
empirical regularities considered include the distribution of the price
log-returns, the price volatility, and the heavy-tail of the log-returns
distributions. The results show that the order sequences from the generative
model are better able to reproduce the statistical behaviour of real price
variations than the sequences from the benchmark.

    

### [[2109.13913] $f$-Cal: Calibrated aleatoric uncertainty estimation from neural networks for robot perception](http://arxiv.org/abs/2109.13913)


  While modern deep neural networks are performant perception modules,
performance (accuracy) alone is insufficient, particularly for safety-critical
robotic applications such as self-driving vehicles. Robot autonomy stacks also
require these otherwise blackbox models to produce reliable and calibrated
measures of confidence on their predictions. Existing approaches estimate
uncertainty from these neural network perception stacks by modifying network
architectures, inference procedure, or loss functions. However, in general,
these methods lack calibration, meaning that the predictive uncertainties do
not faithfully represent the true underlying uncertainties (process noise). Our
key insight is that calibration is only achieved by imposing constraints across
multiple examples, such as those in a mini-batch; as opposed to existing
approaches which only impose constraints per-sample, often leading to
overconfident (thus miscalibrated) uncertainty estimates. By enforcing the
distribution of outputs of a neural network to resemble a target distribution
by minimizing an $f$-divergence, we obtain significantly better-calibrated
models compared to prior approaches. Our approach, $f$-Cal, outperforms
existing uncertainty calibration approaches on robot perception tasks such as
object detection and monocular depth estimation over multiple real-world
benchmarks.

    

### [[2109.13916] Unsolved Problems in ML Safety](http://arxiv.org/abs/2109.13916)


  Machine learning (ML) systems are rapidly increasing in size, are acquiring
new capabilities, and are increasingly deployed in high-stakes settings. As
with other powerful technologies, safety for ML should be a leading research
priority. In response to emerging safety challenges in ML, such as those
introduced by recent large-scale models, we provide a new roadmap for ML Safety
and refine the technical problems that the field needs to address. We present
four problems ready for research, namely withstanding hazards ("Robustness"),
identifying hazards ("Monitoring"), steering ML systems ("Alignment"), and
reducing risks to how ML systems are handled ("External Safety"). Throughout,
we clarify each problem's motivation and provide concrete research directions.

    

### [[1710.11431] Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling](http://arxiv.org/abs/1710.11431)


  This paper introduces a framework for combining scientific knowledge of
physics-based models with neural networks to advance scientific discovery. This
framework, termed physics-guided neural networks (PGNN), leverages the output
of physics-based model simulations along with observational features in a
hybrid modeling setup to generate predictions using a neural network
architecture. Further, this framework uses physics-based loss functions in the
learning objective of neural networks to ensure that the model predictions not
only show lower errors on the training set but are also scientifically
consistent with the known physics on the unlabeled set. We illustrate the
effectiveness of PGNN for the problem of lake temperature modeling, where
physical relationships between the temperature, density, and depth of water are
used to design a physics-based loss function. By using scientific knowledge to
guide the construction and learning of neural networks, we are able to show
that the proposed framework ensures better generalizability as well as
scientific consistency of results. All the code and datasets used in this study
have been made available on this link \url{this https URL}.

    

### [[1804.00308] Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](http://arxiv.org/abs/1804.00308)


  As machine learning becomes widely used for automated decisions, attackers
have strong incentives to manipulate the results and models generated by
machine learning algorithms. In this paper, we perform the first systematic
study of poisoning attacks and their countermeasures for linear regression
models. In poisoning attacks, attackers deliberately influence the training
data to manipulate the results of a predictive model. We propose a
theoretically-grounded optimization framework specifically designed for linear
regression and demonstrate its effectiveness on a range of datasets and models.
We also introduce a fast statistical attack that requires limited knowledge of
the training process. Finally, we design a new principled defense method that
is highly resilient against all poisoning attacks. We provide formal guarantees
about its convergence and an upper bound on the effect of poisoning attacks
when the defense is deployed. We evaluate extensively our attacks and defenses
on three realistic datasets from health care, loan assessment, and real estate
domains.

    

### [[1811.01198] A biconvex optimization for solving semidefinite programs via bilinear factorization](http://arxiv.org/abs/1811.01198)


  Many problems in machine learning can be reduced to learning a low-rank
positive semidefinite matrix (denoted as $Z$), which encounters semidefinite
program (SDP). Existing SDP solvers by classical convex optimization are
expensive to solve large-scale problems. Employing the low rank of solution,
Burer-Monteiro's method reformulated SDP as a nonconvex problem via the
$quadratic$ factorization ($Z$ as $XX^\top$). However, this would lose the
structure of problem in optimization. In this paper, we propose to convert SDP
into a biconvex problem via the $bilinear$ factorization ($Z$ as $XY^\top$),
and while adding the term $\frac{\gamma}{2}||X-Y||_F^2$ to penalize the
difference of $X$ and $Y$. Thus, the biconvex structure (w.r.t. $X$ and $Y$)
can be exploited naturally in optimization. As a theoretical result, we provide
a bound to the penalty parameter $\gamma$ under the assumption of $L$-Lipschitz
smoothness and $\sigma $-strongly biconvexity, such that, at stationary points,
the proposed bilinear factorization is equivalent to Burer-Monteiro's
factorization when the bound is arrived, that is
$\gamma>\frac{1}{4}(L-\sigma)_+$. Our proposal opens up a new way to surrogate
SDP by biconvex program. Experiments on two SDP-related applications
demonstrate that the proposed method is effective as the state-of-the-art.

    

### [[1909.08329] From Server-Based to Client-Based Machine Learning: A Comprehensive Survey](http://arxiv.org/abs/1909.08329)


  In recent years, mobile devices have gained increasing development with
stronger computation capability and larger storage space. Some of the
computation-intensive machine learning tasks can now be run on mobile devices.
To exploit the resources available on mobile devices and preserve personal
privacy, the concept of client-based machine learning has been proposed. It
leverages the users' local hardware and local data to solve machine learning
sub-problems on mobile devices and only uploads computation results rather than
the original data for the optimization of the global model. Such an
architecture can not only relieve computation and storage burdens on servers
but also protect the users' sensitive information. Another benefit is the
bandwidth reduction because various kinds of local data can be involved in the
training process without being uploaded. In this article, we provide a
literature review on the progressive development of machine learning from
server based to client based. We revisit a number of widely used server-based
and client-based machine learning methods and applications. We also extensively
discuss the challenges and future directions in this area. We believe that this
survey will give a clear overview of client-based machine learning and provide
guidelines on applying client-based machine learning to practice.

    

### [[1911.00232] Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding](http://arxiv.org/abs/1911.00232)


  Videos capture events that typically contain multiple sequential, and
simultaneous, actions even in the span of only a few seconds. However, most
large-scale datasets built to train models for action recognition in video only
provide a single label per video. Consequently, models can be incorrectly
penalized for classifying actions that exist in the videos but are not
explicitly labeled and do not learn the full spectrum of information present in
each video in training. Towards this goal, we present the Multi-Moments in Time
dataset (M-MiT) which includes over two million action labels for over one
million three second videos. This multi-label dataset introduces novel
challenges on how to train and analyze models for multi-action detection. Here,
we present baseline results for multi-action recognition using loss functions
adapted for long tail multi-label learning, provide improved methods for
visualizing and interpreting models trained for multi-label action detection
and show the strength of transferring models trained on M-MiT to smaller
datasets.

    

### [[2001.02522] On Interpretability of Artificial Neural Networks: A Survey](http://arxiv.org/abs/2001.02522)


  Deep learning as represented by the artificial deep neural networks (DNNs)
has achieved great success in many important areas that deal with text, images,
videos, graphs, and so on. However, the black-box nature of DNNs has become one
of the primary obstacles for their wide acceptance in mission-critical
applications such as medical diagnosis and therapy. Due to the huge potential
of deep learning, interpreting neural networks has recently attracted much
research attention. In this paper, based on our comprehensive taxonomy, we
systematically review recent studies in understanding the mechanism of neural
networks, describe applications of interpretability especially in medicine, and
discuss future directions of interpretability research, such as in relation to
fuzzy logic and brain science.

    

### [[2006.04877] A Causal Direction Test for Heterogeneous Populations](http://arxiv.org/abs/2006.04877)


  A probabilistic expert system emulates the decision-making ability of a human
expert through a directional graphical model. The first step in building such
systems is to understand data generation mechanism. To this end, one may try to
decompose a multivariate distribution into product of several conditionals, and
evolving a blackbox machine learning predictive models towards transparent
cause-and-effect discovery. Most causal models assume a single homogeneous
population, an assumption that may fail to hold in many applications. We show
that when the homogeneity assumption is violated, causal models developed based
on such assumption can fail to identify the correct causal direction. We
propose an adjustment to a commonly used causal direction test statistic by
using a $k$-means type clustering algorithm where both the labels and the
number of components are estimated from the collected data to adjust the test
statistic. Our simulation result show that the proposed adjustment
significantly improves the performance of the causal direction test statistic
for heterogeneous data. We study large sample behaviour of our proposed test
statistic and demonstrate the application of the proposed method using real
data.

    

### [[2006.10199] Head2Head++: Deep Facial Attributes Re-Targeting](http://arxiv.org/abs/2006.10199)


  Facial video re-targeting is a challenging problem aiming to modify the
facial attributes of a target subject in a seamless manner by a driving
monocular sequence. We leverage the 3D geometry of faces and Generative
Adversarial Networks (GANs) to design a novel deep learning architecture for
the task of facial and head reenactment. Our method is different to purely 3D
model-based approaches, or recent image-based methods that use Deep
Convolutional Neural Networks (DCNNs) to generate individual frames. We manage
to capture the complex non-rigid facial motion from the driving monocular
performances and synthesise temporally consistent videos, with the aid of a
sequential Generator and an ad-hoc Dynamics Discriminator network. We conduct a
comprehensive set of quantitative and qualitative tests and demonstrate
experimentally that our proposed method can successfully transfer facial
expressions, head pose and eye gaze from a source video to a target subject, in
a photo-realistic and faithful fashion, better than other state-of-the-art
methods. Most importantly, our system performs end-to-end reenactment in nearly
real-time speed (18 fps).

    

### [[2007.12160] Online Robust and Adaptive Learning from Data Streams](http://arxiv.org/abs/2007.12160)


  In online learning from non-stationary data streams, it is necessary to learn
robustly to outliers and to adapt quickly to changes in the underlying data
generating mechanism. In this paper, we refer to the former attribute of online
learning algorithms as robustness and to the latter as adaptivity. There is an
obvious tradeoff between the two attributes. It is a fundamental issue to
quantify and evaluate the tradeoff because it provides important information on
the data generating mechanism. However, no previous work has considered the
tradeoff quantitatively. We propose a novel algorithm called the stochastic
approximation-based robustness-adaptivity algorithm (SRA) to evaluate the
tradeoff. The key idea of SRA is to update parameters of distribution or
sufficient statistics with the biased stochastic approximation scheme, while
dropping data points with large values of the stochastic update. We address the
relation between the two parameters: one is the step size of the stochastic
approximation, and the other is the threshold parameter of the norm of the
stochastic update. The former controls the adaptivity and the latter does the
robustness. We give a theoretical analysis for the non-asymptotic convergence
of SRA in the presence of outliers, which depends on both the step size and
threshold parameter. Because SRA is formulated on the majorization-minimization
principle, it is a general algorithm that includes many algorithms, such as the
online EM algorithm and stochastic gradient descent. Empirical experiments for
both synthetic and real datasets demonstrated that SRA was superior to previous
methods.

    

### [[2008.07146] Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](http://arxiv.org/abs/2008.07146)


  Off-policy evaluation (OPE) aims to estimate the performance of hypothetical
policies using data generated by a different policy. Because of its huge
potential impact in practice, there has been growing research interest in this
field. There is, however, no real-world public dataset that enables the
evaluation of OPE, making its experimental studies unrealistic and
irreproducible. With the goal of enabling realistic and reproducible OPE
research, we publicize Open Bandit Dataset collected on a large-scale fashion
e-commerce platform, ZOZOTOWN. Our dataset is unique in that it contains a set
of multiple logged bandit feedback datasets collected by running different
policies on the same platform. This enables experimental comparisons of
different OPE estimators for the first time. We also develop Python software
called Open Bandit Pipeline to streamline and standardize the implementation of
batch bandit algorithms and OPE. Our open data and software will contribute to
the fair and transparent OPE research and help the community identify fruitful
research directions. We provide extensive benchmark experiments of existing OPE
estimators using our dataset and software. The results open up essential
challenges and new avenues for future OPE research.

    

### [[2010.01637] Understanding How Over-Parametrization Leads to Acceleration: A case of learning a single teacher neuron](http://arxiv.org/abs/2010.01637)


  Over-parametrization has become a popular technique in deep learning. It is
observed that by over-parametrization, a larger neural network needs a fewer
training iterations than a smaller one to achieve a certain level of
performance -- namely, over-parametrization leads to acceleration in
optimization. However, despite that over-parametrization is widely used
nowadays, little theory is available to explain the acceleration due to
over-parametrization. In this paper, we propose understanding it by studying a
simple problem first. Specifically, we consider the setting that there is a
single teacher neuron with quadratic activation, where over-parametrization is
realized by having multiple student neurons learn the data generated from the
teacher neuron. We provably show that over-parametrization helps the iterate
generated by gradient descent to enter the neighborhood of a global optimal
solution that achieves zero testing error faster. On the other hand, we also
point out an issue regarding the necessity of over-parametrization and study
how the scaling of the output neurons affects the convergence time.

    

### [[2011.10331] ANIMC: A Soft Framework for Auto-weighted Noisy and Incomplete Multi-view Clustering](http://arxiv.org/abs/2011.10331)


  Multi-view clustering has wide applications in many image processing
scenarios. In these scenarios, original image data often contain missing
instances and noises, which is ignored by most multi-view clustering methods.
However, missing instances may make these methods difficult to use directly and
noises will lead to unreliable clustering results. In this paper, we propose a
novel Auto-weighted Noisy and Incomplete Multi-view Clustering framework
(ANIMC) via a soft auto-weighted strategy and a doubly soft regular regression
model. Firstly, by designing adaptive semi-regularized nonnegative matrix
factorization (adaptive semi-RNMF), the soft auto-weighted strategy assigns a
proper weight to each view and adds a soft boundary to balance the influence of
noises and incompleteness. Secondly, by proposing{\theta}-norm, the doubly soft
regularized regression model adjusts the sparsity of our model by choosing
different{\theta}. Compared with existing methods, ANIMC has three unique
advantages: 1) it is a soft algorithm to adjust our framework in different
scenarios, thereby improving its generalization ability; 2) it automatically
learns a proper weight for each view, thereby reducing the influence of noises;
3) it performs doubly soft regularized regression that aligns the same
instances in different views, thereby decreasing the impact of missing
instances. Extensive experimental results demonstrate its superior advantages
over other state-of-the-art methods.

    

### [[2011.11576] Conjecturing-Based Computational Discovery of Patterns in Data](http://arxiv.org/abs/2011.11576)


  Modern machine learning methods are designed to exploit complex patterns in
data regardless of their form, while not necessarily revealing them to the
investigator. Here we demonstrate situations where modern machine learning
methods are ill-equipped to reveal feature interaction effects and other
nonlinear relationships. We propose the use of a conjecturing machine that
generates feature relationships in the form of bounds for numerical features
and boolean expressions for nominal features that are ignored by machine
learning algorithms. The proposed framework is demonstrated for a
classification problem with an interaction effect and a nonlinear regression
problem. In both settings, true underlying relationships are revealed and
generalization performance improves. The framework is then applied to
patient-level data regarding COVID-19 outcomes to suggest possible risk
factors.

    

### [[2012.04780] Open Knowledge Graphs Canonicalization using Variational Autoencoders](http://arxiv.org/abs/2012.04780)


  Noun phrases and Relation phrases in open knowledge graphs are not
canonicalized, leading to an explosion of redundant and ambiguous
subject-relation-object triples. Existing approaches to solve this problem take
a two-step approach. First, they generate embedding representations for both
noun and relation phrases, then a clustering algorithm is used to group them
using the embeddings as features. In this work, we propose Canonicalizing Using
Variational Autoencoders (CUVA), a joint model to learn both embeddings and
cluster assignments in an end-to-end approach, which leads to a better vector
representation for the noun and relation phrases. Our evaluation over multiple
benchmarks shows that CUVA outperforms the existing state-of-the-art
approaches. Moreover, we introduce CanonicNell, a novel dataset to evaluate
entity canonicalization systems.

    

### [[2102.00457] MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification](http://arxiv.org/abs/2102.00457)


  We propose MultiRocket, a fast time series classification (TSC) algorithm
that achieves state-of-the-art performance with a tiny fraction of the time and
without the complex ensembling structure of many state-of-the-art methods.
MultiRocket improves on MiniRocket, one of the fastest TSC algorithms to date,
by adding multiple pooling operators and transformations to improve the
diversity of the features generated. In addition to processing the raw input
series, MultiRocket also applies first order differences to transform the
original series. Convolutions are applied to both representations, and four
pooling operators are applied to the convolution outputs. When benchmarked
using the University of California Riverside TSC benchmark datasets,
MultiRocket is significantly more accurate than MiniRocket, and competitive
with the best ranked current method in terms of accuracy, HIVE-COTE 2.0, while
being orders of magnitude faster.

    

### [[2102.02969] Sign-RIP: A Robust Restricted Isometry Property for Low-rank Matrix Recovery](http://arxiv.org/abs/2102.02969)


  Restricted isometry property (RIP), essentially stating that the linear
measurements are approximately norm-preserving, plays a crucial role in
studying low-rank matrix recovery problem. However, RIP fails in the robust
setting, when a subset of the measurements are grossly corrupted with noise. In
this work, we propose a robust restricted isometry property, called Sign-RIP,
and show its broad applications in robust low-rank matrix recovery. In
particular, we show that Sign-RIP can guarantee the uniform convergence of the
subdifferentials of the robust matrix recovery with nonsmooth loss function,
even at the presence of arbitrarily dense and arbitrarily large outliers. Based
on Sign-RIP, we characterize the location of the critical points in the robust
rank-1 matrix recovery, and prove that they are either close to the true
solution, or have small norm. Moreover, in the over-parameterized regime, where
the rank of the true solution is over-estimated, we show that subgradient
method converges to the true solution at a (nearly) dimension-free rate.
Finally, we show that sign-RIP enjoys almost the same complexity as its
classical counterparts, but provides significantly better robustness against
noise.

    

### [[2102.07988] TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models](http://arxiv.org/abs/2102.07988)


  Model parallelism has become a necessity for training modern large-scale deep
language models. In this work, we identify a new and orthogonal dimension from
existing model parallel approaches: it is possible to perform pipeline
parallelism within a single training sequence for Transformer-based language
models thanks to its autoregressive property. This enables a more fine-grained
pipeline compared with previous work. With this key idea, we design TeraPipe, a
high-performance token-level pipeline parallel algorithm for synchronous
model-parallel training of Transformer-based language models. We develop a
novel dynamic programming-based algorithm to calculate the optimal pipelining
execution scheme given a specific model and cluster configuration. We show that
TeraPipe can speed up the training by 5.0x for the largest GPT-3 model with 175
billion parameters on an AWS cluster with 48 p3.16xlarge instances compared
with state-of-the-art model-parallel methods. The code for reproduction can be
found at this https URL


### [[2103.12803] JFB: Jacobian-Free Backpropagation for Implicit Networks](http://arxiv.org/abs/2103.12803)


  A promising trend in deep learning replaces traditional feedforward networks
with implicit networks. Unlike traditional networks, implicit networks solve a
fixed point equation to compute inferences. Solving for the fixed point varies
in complexity, depending on provided data and an error tolerance. Importantly,
implicit networks may be trained with fixed memory costs in stark contrast to
feedforward networks, whose memory requirements scale linearly with depth.
However, there is no free lunch -- backpropagation through implicit networks
often requires solving a costly Jacobian-based equation arising from the
implicit function theorem. We propose Jacobian-Free Backpropagation (JFB), a
fixed-memory approach that circumvents the need to solve Jacobian-based
equations. JFB makes implicit networks faster to train and significantly easier
to implement, without sacrificing test accuracy. Our experiments show implicit
networks trained with JFB are competitive with feedforward networks and prior
implicit networks given the same number of parameters.

    

### [[2103.13509] A Variational Inequality Approach to Bayesian Regression Games](http://arxiv.org/abs/2103.13509)


  Bayesian regression games are a special class of two-player general-sum
Bayesian games in which the learner is partially informed about the adversary's
objective through a Bayesian prior. This formulation captures the uncertainty
in regard to the adversary, and is useful in problems where the learner and
adversary may have conflicting, but not necessarily perfectly antagonistic
objectives. Although the Bayesian approach is a more general alternative to the
standard minimax formulation, the applications of Bayesian regression games
have been limited due to computational difficulties, and the existence and
uniqueness of a Bayesian equilibrium are only known for quadratic cost
functions. First, we prove the existence and uniqueness of a Bayesian
equilibrium for a class of convex and smooth Bayesian games by regarding it as
a solution of an infinite-dimensional variational inequality (VI) in Hilbert
space. We consider two special cases in which the infinite-dimensional VI
reduces to a high-dimensional VI or a nonconvex stochastic optimization, and
provide two simple algorithms of solving them with strong convergence
guarantees. Numerical results on real datasets demonstrate the promise of this
approach.

    

### [[2103.14417] Self-Supervised Learning in Multi-Task Graphs through Iterative Consensus Shift](http://arxiv.org/abs/2103.14417)


  The human ability to synchronize the feedback from all their senses inspired
recent works in multi-task and multi-modal learning. While these works rely on
expensive supervision, our multi-task graph requires only pseudo-labels from
expert models. Every graph node represents a task, and each edge learns between
tasks transformations. Once initialized, the graph learns self-supervised,
based on a novel consensus shift algorithm that intelligently exploits the
agreement between graph pathways to generate new pseudo-labels for the next
learning cycle. We demonstrate significant improvement from one unsupervised
learning iteration to the next, outperforming related recent methods in
extensive multi-task learning experiments on two challenging datasets.

    

### [[2104.00453] Learning Rates for Multi-task Regularization Networks](http://arxiv.org/abs/2104.00453)


  Multi-task learning is an important trend of machine learning in facing the
era of artificial intelligence and big data. Despite a large amount of
researches on learning rate estimates of various single-task machine learning
algorithms, there is little parallel work for multi-task learning. We present
mathematical analysis on the learning rate estimate of multi-task learning
based on the theory of vector-valued reproducing kernel Hilbert spaces and
matrix-valued reproducing kernels. For the typical multi-task regularization
networks, an explicit learning rate dependent both on the number of sample data
and the number of tasks is obtained. It reveals that the generalization ability
of multi-task learning algorithms is indeed affected as the number of tasks
increases.

    

### [[2104.02556] Physics-Informed Neural Nets for Control of Dynamical Systems](http://arxiv.org/abs/2104.02556)


  Physics-informed neural networks (PINNs) impose known physical laws into the
learning of deep neural networks, making sure they respect the physics of the
process while decreasing the demand of labeled data. For systems represented by
Ordinary Differential Equations (ODEs), the conventional PINN has a continuous
time input variable and outputs the solution of the corresponding ODE. In their
original form, PINNs do not allow control inputs neither can they simulate for
long-range intervals without serious degradation in their predictions. In this
context, this work presents a new framework called Physics-Informed Neural Nets
for Control (PINC), which proposes a novel PINN-based architecture that is
amenable to \emph{control} problems and able to simulate for longer-range time
horizons that are not fixed beforehand. The framework has new inputs to account
for the initial state of the system and the control action. In PINC, the
response over the complete time horizon is split such that each smaller
interval constitutes a solution of the ODE conditioned on the fixed values of
initial state and control action for that interval. The whole response is
formed by feeding back the predictions of the terminal state as the initial
state for the next interval. This proposal enables the optimal control of
dynamic systems, integrating a priori knowledge from experts and data collected
from plants into control applications. We showcase our proposal in the control
of two nonlinear dynamic systems: the Van der Pol oscillator and the four-tank
system.

    

### [[2106.01100] Prediction of the Position of External Markers Using a Recurrent Neural Network Trained With Unbiased Online Recurrent Optimization for Safe Lung Cancer Radiotherapy](http://arxiv.org/abs/2106.01100)


  During lung cancer radiotherapy, the position of infrared reflective objects
on the chest can be recorded to estimate the tumor location. However,
radiotherapy systems usually have a latency inherent to robot control
limitations that impedes the radiation delivery precision. Not taking this
phenomenon into account may cause unwanted damage to healthy tissues and lead
to side effects such as radiation pneumonitis. In this research, we use nine
observation records of the three-dimensional position of three external markers
on the chest and abdomen of healthy individuals breathing during intervals from
73s to 222s. The sampling frequency is equal to 10Hz and the amplitudes of the
recorded trajectories range from 6mm to 40mm in the superior-inferior
direction. We forecast the location of each marker simultaneously with a
horizon value (the time interval in advance for which the prediction is made)
between 0.1s and 2.0s, using a recurrent neural network (RNN) trained with
unbiased online recurrent optimization (UORO). We compare its performance with
an RNN trained with real-time recurrent learning, least mean squares (LMS), and
offline linear regression. Training and cross-validation are performed during
the first minute of each sequence. On average, UORO achieves the lowest
root-mean-square (RMS) and maximum error, equal respectively to 1.3mm and
8.8mm, with a prediction time per time step lower than 2.8ms (Dell Intel core
i9-9900K 3.60Ghz). Linear regression has the lowest RMS error for the horizon
values 0.1s and 0.2s, followed by LMS for horizon values between 0.3s and 0.5s,
and UORO for horizon values greater than 0.6s.

    

### [[2106.11908] Deep Phasor Networks: Connecting Conventional and Spiking Neural Networks](http://arxiv.org/abs/2106.11908)


  In this work, we extend standard neural networks by building upon an
assumption that neuronal activations correspond to the angle of a complex
number lying on the unit circle, or 'phasor.' Each layer in such a network
produces new activations by taking a weighted superposition of the previous
layer's phases and calculating the new phase value. This generalized
architecture allows models to reach high accuracy and carries the singular
advantage that mathematically equivalent versions of the network can be
executed with or without regard to a temporal variable. Importantly, the value
of a phase angle in the temporal domain can be sparsely represented by a
periodically repeating series of delta functions or 'spikes'. We demonstrate
the atemporal training of a phasor network on standard deep learning tasks and
show that these networks can then be executed in either the traditional
atemporal domain or spiking temporal domain with no conversion step needed.
This provides a novel basis for constructing deep networkswhich operate via
temporal, spike-based calculations suitable for neuromorphic computing
hardware.

    

### [[2109.12985] Synerise at RecSys 2021: Twitter user engagement prediction with a fast neural model](http://arxiv.org/abs/2109.12985)


  In this paper we present our 2nd place solution to ACM RecSys 2021 Challenge
organized by Twitter. The challenge aims to predict user engagement for a set
of tweets, offering an exceptionally large data set of 1 billion data points
sampled from over four weeks of real Twitter interactions. Each data point
contains multiple sources of information, such as tweet text along with
engagement features, user features, and tweet features. The challenge brings
the problem close to a real production environment by introducing strict
latency constraints in the model evaluation phase: the average inference time
for single tweet engagement prediction is limited to 6ms on a single CPU core
with 64GB memory. Our proposed model relies on extensive feature engineering
performed with methods such as the Efficient Manifold Density Estimator (EMDE)
- our previously introduced algorithm based on Locality Sensitive Hashing
method, and novel Fourier Feature Encoding, among others. In total, we create
numerous features describing a user's Twitter account status and the content of
a tweet. In order to adhere to the strict latency constraints, the underlying
model is a simple residual feed-forward neural network. The system is a
variation of our previous methods which proved successful in KDD Cup 2021, WSDM
Challenge 2021, and SIGIR eCom Challenge 2020. We release the source code at:
this https URL


### [[2109.13059] Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations](http://arxiv.org/abs/2109.13059)


  In NLP, a large volume of tasks involve pairwise comparison between two
sequences (e.g. sentence similarity and paraphrase identification).
Predominantly, two formulations are used for sentence-pair tasks: bi-encoders
and cross-encoders. Bi-encoders produce fixed-dimensional sentence
representations and are computationally efficient, however, they usually
underperform cross-encoders. Cross-encoders can leverage their attention heads
to exploit inter-sentence interactions for better performance but they require
task fine-tuning and are computationally more expensive. In this paper, we
present a completely unsupervised sentence representation model termed as
Trans-Encoder that combines the two learning paradigms into an iterative joint
framework to simultaneously learn enhanced bi- and cross-encoders.
Specifically, on top of a pre-trained Language Model (PLM), we start with
converting it to an unsupervised bi-encoder, and then alternate between the bi-
and cross-encoder task formulations. In each alternation, one task formulation
will produce pseudo-labels which are used as learning signals for the other
task formulation. We then propose an extension to conduct such
self-distillation approach on multiple PLMs in parallel and use the average of
their pseudo-labels for mutual-distillation. Trans-Encoder creates, to the best
of our knowledge, the first completely unsupervised cross-encoder and also a
state-of-the-art unsupervised bi-encoder for sentence similarity. Both the
bi-encoder and cross-encoder formulations of Trans-Encoder outperform recently
proposed state-of-the-art unsupervised sentence encoders such as Mirror-BERT
and SimCSE by up to 5% on the sentence similarity benchmarks.

    

### [[2109.13319] How Low Can You Go? Practical cold-start performance limits in FaaS](http://arxiv.org/abs/2109.13319)


  Function-as-a-Service (FaaS) has recently emerged as a new cloud computing
paradigm. It promises high utilization of data center resources through
allocating resources on demand at per-function request granularity. High
cold-start overheads, however, have been limiting FaaS systems' such potential.
Prior work has recognized that time redundancy exists across different cold
function invocations and has proposed varied snapshots that capture the
instantaneous execution state so allow for jump-starts through restoration.
However, it remains unclear what the cold-start performance limits are as the
previous snapshots employ different techniques and aredesigned for different
environments. In this paper, we summarize these snapshots from a taxonomic
perspective andpresent a model that depicts the cold-start performance
fromfirst principles. To approximate the performance limits, we propose a
snapshot design SnapFaaS. We evaluate SnapFaaS using real-world FaaS functions.
Our empirical results prove SnapFaaS' efficiency. It is 2-10x as fast as prior
work for most functions and achieves near-optimal cold-start performance.

    

### [[2109.13356] Efficient Computer Vision on Edge Devices with Pipeline-Parallel Hierarchical Neural Networks](http://arxiv.org/abs/2109.13356)


  Computer vision on low-power edge devices enables applications including
search-and-rescue and security. State-of-the-art computer vision algorithms,
such as Deep Neural Networks (DNNs), are too large for inference on low-power
edge devices. To improve efficiency, some existing approaches parallelize DNN
inference across multiple edge devices. However, these techniques introduce
significant communication and synchronization overheads or are unable to
balance workloads across devices. This paper demonstrates that the hierarchical
DNN architecture is well suited for parallel processing on multiple edge
devices. We design a novel method that creates a parallel inference pipeline
for computer vision problems that use hierarchical DNNs. The method balances
loads across the collaborating devices and reduces communication costs to
facilitate the processing of multiple video frames simultaneously with higher
throughput. Our experiments consider a representative computer vision problem
where image recognition is performed on each video frame, running on multiple
Raspberry Pi 4Bs. With four collaborating low-power edge devices, our approach
achieves 3.21X higher throughput, 68% less energy consumption per device per
frame, and 58% decrease in memory when compared with existing single-device
hierarchical DNNs.

    

### [[2109.13492] Restructuring Serverless Computing with Data-Centric Function Orchestration](http://arxiv.org/abs/2109.13492)


  Serverless applications are usually composed of multiple short-lived,
single-purpose functions exchanging data in reaction to events or changes of
states. Existing function orchestration services coordinate functions and
trigger their activation following some predefined rules (e.g., function
dependency and state machine), while being oblivious to the underlying data
exchange between functions. Such design has limited expressiveness and incurs
high orchestration overhead: developers often need to manage complex function
interactions by themselves, and the performance can still be unsatisfactory. In
this paper, we advocate data-centric orchestration where function invocations
are triggered by the flow of data. In our design, the platform provides data
trigger APIs through which developers can control when and how the output of
one or many functions is passed to other functions as input and triggers their
executions. With explicit support of data triggers, complex function
interactions can be easily implemented, and data locality can also be
satisfied. As a manifestation of this design, we present Pheromone, a scalable,
low-latency serverless platform. Pheromone schedules functions close to the
input with a two-level, shared-nothing scheduling hierarchy. Compared to
existing commercial and open-source platforms, Pheromone cuts the latencies of
function interactions and data exchanges by orders of magnitude and scales well
to complex workflows with long function chains and high parallelism. Case
studies further demonstrate that Pheromone enables easy implementations of many
applications, including real-time query, stream processing, and MapReduce sort.

    

### [[2109.13504] The Megopolis Resampler: Memory Coalesced Resampling on GPUs](http://arxiv.org/abs/2109.13504)


  The resampling process employed in widely used methods such as Importance
Sampling (IS), with its adaptive extension (AIS), are used to solve challenging
problems requiring approximate inference; for example, non-linear, non-Gaussian
state estimation problems. However, the re-sampling process can be
computationally prohibitive for practical problems with real-time requirements.
We consider the problem of developing highly parallelisable resampling
algorithms for massively parallel hardware architectures of modern graphics
processing units (GPUs) to accomplish real-time performance. We develop a new
variant of the Metropolis algorithm -- Megopolis -- that improves performance
without requiring a tuning parameter or reducing resampling quality. The
Megopolis algorithm is built upon exploiting the memory access patterns of
modern GPU units to reduce the number of memory transactions without the need
for tuning parameters. Extensive numerical experiments on GPU hardware
demonstrate that the proposed Megopolis algorithm is numerically stable and
outperforms the original Metropolis algorithm and its variants -- Metropolis-C1
and Metropolis-C2 -- in speed and quality metrics. Further, given the absence
of open tools in this domain and facilitating fair comparisons in the future
and supporting the signal processing community, we also open source the
complete project, including a repository of source code with Megopolis and all
other comparison methods.

    

### [[2109.13586] Dynamics in Coded Edge Computing for IoT: A Fractional Evolutionary Game Approach](http://arxiv.org/abs/2109.13586)


  Recently, coded distributed computing (CDC), with advantages in intensive
computation and reduced latency, has attracted a lot of research interest for
edge computing, in particular, IoT applications, including IoT data
pre-processing and data analytics. Nevertheless, it can be challenging for edge
infrastructure providers (EIPs) with limited edge resources to support IoT
applications performed in a CDC approach in edge networks, given the additional
computational resources required by CDC. In this paper, we propose coded edge
federation, in which different EIPs collaboratively provide edge resources for
CDC tasks. To study the Nash equilibrium, when no EIP has an incentive to
unilaterally alter its decision on edge resource allocation, we model the coded
edge federation based on evolutionary game theory. Since the replicator
dynamics of the classical evolutionary game are unable to model economic-aware
EIPs which memorize past decisions and utilities, we propose fractional
replicator dynamics with a power-law fading memory via Caputo fractional
derivatives. The proposed dynamics allow us to study a broad spectrum of EIP
dynamic behaviors, such as EIP sensitivity and aggressiveness in strategy
adaptation, which classical replicator dynamics cannot capture. Theoretical
analysis and extensive numerical results justify the existence, uniqueness, and
stability of the equilibrium in the fractional evolutionary game. The influence
of the content and the length of the memory on the rate of convergence is also
investigated.

    

### [[1510.06882] Trading off $t$-Resilience for Efficiency in Asynchronous Byzantine Reliable Broadcast](http://arxiv.org/abs/1510.06882)


  This paper presents a simple and efficient reliable broadcast algorithm for
asynchronous message-passing systems made up of $n$ processes, among which up
to $t<n/5$ may behave arbitrarily (Byzantine processes). This algorithm
requires two communication steps and $n^2-1$ messages. When compared to
Bracha's algorithm, which is resilience optimal ($t<n/3$) and requires three
communication steps and $2n^2-n-1$ messages, the proposed algorithm shows an
interesting tradeoff between communication efficiency and $t$-resilience.

    

### [[2007.00415] A Truly Self-Sovereign Identity System](http://arxiv.org/abs/2007.00415)


  Existing digital identity management systems fail to deliver the desirable
properties of control by the users of their own identity data, credibility of
disclosed identity data, and network-level anonymity. The recently proposed
Self-Sovereign Identity (SSI) approach promises to give users these properties.
However, we argue that without addressing privacy at the network level, SSI
systems cannot deliver on this promise. In this paper we present the design and
analysis of our solution TCID, created in collaboration with the Dutch
government. TCID is a system consisting of a set of components that together
satisfy seven functional requirements to guarantee the desirable system
properties. We show that the latency incurred by network-level anonymization in
TCID is significantly larger than that of identity data disclosure protocols
but is still low enough for practical situations. We conclude that current
research on SSI is too narrowly focused on these data disclosure protocols.

    

### [[2109.13238] Visually Grounded Reasoning across Languages and Cultures](http://arxiv.org/abs/2109.13238)


  The design of widespread vision-and-language datasets and pre-trained
encoders directly adopts, or draws inspiration from, the concepts and images of
ImageNet. While one can hardly overestimate how much this benchmark contributed
to progress in computer vision, it is mostly derived from lexical databases and
image queries in English, resulting in source material with a North American or
Western European bias. Therefore, we devise a new protocol to construct an
ImageNet-style hierarchy representative of more languages and cultures. In
particular, we let the selection of both concepts and images be entirely driven
by native speakers, rather than scraping them automatically. Specifically, we
focus on a typologically diverse set of languages, namely, Indonesian, Mandarin
Chinese, Swahili, Tamil, and Turkish. On top of the concepts and images
obtained through this new protocol, we create a multilingual dataset for
{M}ulticultur{a}l {R}easoning over {V}ision and {L}anguage (MaRVL) by eliciting
statements from native speaker annotators about pairs of images. The task
consists of discriminating whether each grounded statement is true or false. We
establish a series of baselines using state-of-the-art models and find that
their cross-lingual transfer performance lags dramatically behind supervised
performance in English. These results invite us to reassess the robustness and
accuracy of current state-of-the-art models beyond a narrow domain, but also
open up new exciting challenges for the development of truly multilingual and
multicultural systems.

    

### [[2109.13273] Design of quantum optical experiments with logic artificial intelligence](http://arxiv.org/abs/2109.13273)


  Logic artificial intelligence (AI) is a subfield of AI where variables can
take two defined arguments, True or False, and are arranged in clauses that
follow the rules of formal logic. Several problems that span from physical
systems to mathematical conjectures can be encoded into these clauses and be
solved by checking their satisfiability (SAT). Recently, SAT solvers have
become a sophisticated and powerful computational tool capable, among other
things, of solving long-standing mathematical conjectures. In this work, we
propose the use of logic AI for the design of optical quantum experiments. We
show how to map into a SAT problem the experimental preparation of an arbitrary
quantum state and propose a logic-based algorithm, called Klaus, to find an
interpretable representation of the photonic setup that generates it. We
compare the performance of Klaus with the state-of-the-art algorithm for this
purpose based on continuous optimization. We also combine both logic and
numeric strategies to find that the use of logic AI improves significantly the
resolution of this problem, paving the path to develop more formal-based
approaches in the context of quantum physics experiments.

    

### [[2109.13355] Evolving reversible circuits for the even-parity problem](http://arxiv.org/abs/2109.13355)


  Reversible computing basically means computation with less or not at all
electrical power. Since the standard binary gates are not usually reversible we
use the Fredkin gate in order to achieve reversibility. An algorithm for
designing reversible digital circuits is described in this paper. The algorithm
is based on Multi Expression Programming (MEP), a Genetic Programming variant
with a linear representation of individuals. The case of digital circuits for
the even-parity problem is investigated. Numerical experiments show that the
MEP-based algorithm is able to easily design reversible digital circuits for up
to the even-8-parity problem.

    

### [[2109.13367] A taxonomy of strategic human interactions in traffic conflicts](http://arxiv.org/abs/2109.13367)


  In order to enable autonomous vehicles (AV) to navigate busy traffic
situations, in recent years there has been a focus on game-theoretic models for
strategic behavior planning in AVs. However, a lack of common taxonomy impedes
a broader understanding of the strategies the models generate as well as the
development of safety specification to identity what strategies are safe for an
AV to execute. Based on common patterns of interaction in traffic conflicts, we
develop a taxonomy for strategic interactions along the dimensions of agents'
initial response to right-of-way rules and subsequent response to other agents'
behavior. Furthermore, we demonstrate a process of automatic mapping of
strategies generated by a strategic planner to the categories in the taxonomy,
and based on vehicle-vehicle and vehicle-pedestrian interaction simulation, we
evaluate two popular solution concepts used in strategic planning in AVs, QLk
and Subgame perfect $\epsilon$-Nash Equilibrium, with respect to those
categories.

    

### [[2109.13377] Model-Free Reinforcement Learning for Optimal Control of MarkovDecision Processes Under Signal Temporal Logic Specifications](http://arxiv.org/abs/2109.13377)


  We present a model-free reinforcement learning algorithm to find an optimal
policy for a finite-horizon Markov decision process while guaranteeing a
desired lower bound on the probability of satisfying a signal temporal logic
(STL) specification. We propose a method to effectively augment the MDP state
space to capture the required state history and express the STL objective as a
reachability objective. The planning problem can then be formulated as a
finite-horizon constrained Markov decision process (CMDP). For a general finite
horizon CMDP problem with unknown transition probability, we develop a
reinforcement learning scheme that can leverage any model-free RL algorithm to
provide an approximately optimal policy out of the general space of
non-stationary randomized policies. We illustrate the effectiveness of our
approach in the context of robotic motion planning for complex missions under
uncertainty and performance objectives.

    

### [[2109.13392] The Tensor Brain: A Unified Theory of Perception, Memory and Semantic Decoding](http://arxiv.org/abs/2109.13392)


  We present a unified computational theory of perception and memory. In our
model, perception, episodic memory, and semantic memory are realized by
different functional and operational modes of the oscillating interactions
between an index layer and a representation layer in a bilayer tensor network
(BTN). The memoryless semantic {representation layer} broadcasts information.
In cognitive neuroscience, it would be the "mental canvas", or the "global
workspace" and reflects the cognitive brain state. The symbolic {index layer}
represents concepts and past episodes, whose semantic embeddings are
implemented in the connection weights between both layers. In addition, we
propose a {working memory layer} as a processing center and information buffer.
Episodic and semantic memory realize memory-based reasoning, i.e., the recall
of relevant past information to enrich perception, and are personalized to an
agent's current state, as well as to an agent's unique memories. Episodic
memory stores and retrieves past observations and provides provenance and
context. Recent episodic memory enriches perception by the retrieval of
perceptual experiences, which provide the agent with a sense about the here and
now: to understand its own state, and the world's semantic state in general,
the agent needs to know what happened recently, in recent scenes, and on
recently perceived entities. Remote episodic memory retrieves relevant past
experiences, contributes to our conscious self, and, together with semantic
memory, to a large degree defines who we are as individuals.

    

### [[2109.13396] Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets](http://arxiv.org/abs/2109.13396)


  Robot learning holds the promise of learning policies that generalize
broadly. However, such generalization requires sufficiently diverse datasets of
the task of interest, which can be prohibitively expensive to collect. In other
fields, such as computer vision, it is common to utilize shared, reusable
datasets, such as ImageNet, to overcome this challenge, but this has proven
difficult in robotics. In this paper, we ask: what would it take to enable
practical data reuse in robotics for end-to-end skill learning? We hypothesize
that the key is to use datasets with multiple tasks and multiple domains, such
that a new user that wants to train their robot to perform a new task in a new
domain can include this dataset in their training process and benefit from
cross-task and cross-domain generalization. To evaluate this hypothesis, we
collect a large multi-domain and multi-task dataset, with 7,200 demonstrations
constituting 71 tasks across 10 environments, and empirically study how this
data can improve the learning of new tasks in new environments. We find that
jointly training with the proposed dataset and 50 demonstrations of a
never-before-seen task in a new domain on average leads to a 2x improvement in
success rate compared to using target domain data alone. We also find that data
for only a few tasks in a new domain can bridge the domain gap and make it
possible for a robot to perform a variety of prior tasks that were only seen in
other domains. These results suggest that reusing diverse multi-task and
multi-domain datasets, including our open-source dataset, may pave the way for
broader robot generalization, eliminating the need to re-collect data for each
new robot learning project.

    

### [[2109.13420] Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition](http://arxiv.org/abs/2109.13420)


  It has been well proved that deep networks are efficient at extracting
features from a given (source) labeled dataset. However, it is not always the
case that they can generalize well to other (target) datasets which very often
have a different underlying distribution. In this report, we evaluate four
different domain adaptation techniques for image classification tasks:
DeepCORAL, DeepDomainConfusion, CDAN and CDAN+E. These techniques are
unsupervised given that the target dataset dopes not carry any labels during
training phase. We evaluate model performance on the office-31 dataset. A link
to the github repository of this report can be found here:
this https URL.

    

### [[2109.13430] SYGMA: System for Generalizable Modular Question Answering OverKnowledge Bases](http://arxiv.org/abs/2109.13430)


  Knowledge Base Question Answering (KBQA) tasks that in-volve complex
reasoning are emerging as an important re-search direction. However, most KBQA
systems struggle withgeneralizability, particularly on two dimensions: (a)
acrossmultiple reasoning types where both datasets and systems haveprimarily
focused on multi-hop reasoning, and (b) across mul-tiple knowledge bases, where
KBQA approaches are specif-ically tuned to a single knowledge base. In this
paper, wepresent SYGMA, a modular approach facilitating general-izability
across multiple knowledge bases and multiple rea-soning types. Specifically,
SYGMA contains three high levelmodules: 1) KB-agnostic question understanding
module thatis common across KBs 2) Rules to support additional reason-ing types
and 3) KB-specific question mapping and answeringmodule to address the
KB-specific aspects of the answer ex-traction. We demonstrate effectiveness of
our system by evalu-ating on datasets belonging to two distinct knowledge
bases,DBpedia and Wikidata. In addition, to demonstrate extensi-bility to
additional reasoning types we evaluate on multi-hopreasoning datasets and a new
Temporal KBQA benchmarkdataset on Wikidata, namedTempQA-WD1, introduced in
thispaper. We show that our generalizable approach has bettercompetetive
performance on multiple datasets on DBpediaand Wikidata that requires both
multi-hop and temporal rea-soning

    

### [[2109.13456] SiamEvent: Event-based Object Tracking via Edge-aware Similarity Learning with Siamese Networks](http://arxiv.org/abs/2109.13456)


  Event cameras are novel sensors that perceive the per-pixel intensity changes
and output asynchronous event streams, showing lots of advantages over
traditional cameras, such as high dynamic range (HDR) and no motion blur. It
has been shown that events alone can be used for object tracking by motion
compensation or prediction. However, existing methods assume that the target
always moves and is the stand-alone object. Moreover, they fail to track the
stopped non-independent moving objects on fixed scenes. In this paper, we
propose a novel event-based object tracking framework, called SiamEvent, using
Siamese networks via edge-aware similarity learning. Importantly, to find the
part having the most similar edge structure of target, we propose to correlate
the embedded events at two timestamps to compute the target edge similarity.
The Siamese network enables tracking arbitrary target edge by finding the part
with the highest similarity score. This extends the possibility of event-based
object tracking applied not only for the independent stand-alone moving
objects, but also for various settings of the camera and scenes. In addition,
target edge initialization and edge detector are also proposed to prevent
SiamEvent from the drifting problem. Lastly, we built an open dataset including
various synthetic and real scenes to train and evaluate SiamEvent. Extensive
experiments demonstrate that SiamEvent achieves up to 15% tracking performance
enhancement than the baselines on the real-world scenes and more robust
tracking performance in the challenging HDR and motion blur conditions.

    

### [[2109.13479] Transfer Learning based Evolutionary Deep Neural Network for Intelligent Fault Diagnosis](http://arxiv.org/abs/2109.13479)


  The performance of a deep neural network (DNN) for fault diagnosis is very
much dependent on the network architecture. Also, the diagnostic performance is
reduced if the model trained on a laboratory case machine is used on a test
dataset from an industrial machine running under variable operating conditions.
Thus there are two challenges for the intelligent fault diagnosis of industrial
machines: (i) selection of suitable DNN architecture and (ii) domain adaptation
for the change in operating conditions. Therefore, we propose an evolutionary
Net2Net transformation (EvoNet2Net) that finds the best suitable DNN
architecture for the given dataset. Nondominated sorting genetic algorithm II
has been used to optimize the depth and width of the DNN architecture. We have
formulated a transfer learning-based fitness evaluation scheme for faster
evolution. It uses the concept of domain adaptation for quick learning of the
data pattern in the target domain. Also, we have introduced a hybrid crossover
technique for optimization of the depth and width of the deep neural network
encoded in a chromosome. We have used the Case Western Reserve University
dataset and Paderborn university dataset to demonstrate the effectiveness of
the proposed framework for the selection of the best suitable architecture
capable of excellent diagnostic performance, classification accuracy almost up
to 100\%.

    

### [[2109.13486] Exploring Teacher-Student Learning Approach for Multi-lingual Speech-to-Intent Classification](http://arxiv.org/abs/2109.13486)


  End-to-end speech-to-intent classification has shown its advantage in
harvesting information from both text and speech. In this paper, we study a
technique to develop such an end-to-end system that supports multiple
languages. To overcome the scarcity of multi-lingual speech corpus, we exploit
knowledge from a pre-trained multi-lingual natural language processing model.
Multi-lingual bidirectional encoder representations from transformers (mBERT)
models are trained on multiple languages and hence expected to perform well in
the multi-lingual scenario. In this work, we employ a teacher-student learning
approach to sufficiently extract information from an mBERT model to train a
multi-lingual speech model. In particular, we use synthesized speech generated
from an English-Mandarin text corpus for analysis and training of a
multi-lingual intent classification model. We also demonstrate that the
teacher-student learning approach obtains an improved performance (91.02%) over
the traditional end-to-end (89.40%) intent classification approach in a
practical multi-lingual scenario.

    

### [[2109.13493] Designed to Cooperate: A Kant-Inspired Ethic of Machine-to-Machine Cooperation](http://arxiv.org/abs/2109.13493)


  This position paper highlights an ethic of machine-to-machine cooperation and
machine pro-sociality, and argues that machines capable of autonomous sensing,
decision-making and action, such as automated vehicles and urban robots, owned
and used by different self-interested parties, and having their own agendas (or
interests of their owners) should be designed and built to be cooperative in
their behaviours, especially if they share public spaces. That is, by design,
the machine should first cooperate, and then only consider alternatives if
there are problems. It is argued that being cooperative is not only important
for their improved functioning, especially, when they use shared resources
(e.g., parking spaces, public roads, curbside space and walkways), but also as
a favourable requirement analogous to how humans cooperating with other humans
can be advantageous and often viewed favourably. The usefulness of such
machine-to-machine cooperation are illustrated via examples including
cooperative crowdsourcing, cooperative traffic routing and parking as well as
futuristic scenarios involving urban robots for delivery and shopping. It is
argued that just as privacy-by-design and security-by-design are important
considerations, in order to yield systems that fulfil ethical requirements,
cooperative-by-design should also be an imperative for autonomous systems that
are separately owned but co-inhabit the same spaces and use common resources.
If a machine using shared public spaces is not cooperative, as one might
expect, then it is not only anti-social but not behaving ethically. It is also
proposed that certification for urban robots that operate in public could be
explored.

    

### [[2109.13531] Multi-Semantic Image Recognition Model and Evaluating Index for explaining the deep learning models](http://arxiv.org/abs/2109.13531)


  Although deep learning models are powerful among various applications, most
deep learning models are still a black box, lacking verifiability and
interpretability, which means the decision-making process that human beings
cannot understand. Therefore, how to evaluate deep neural networks with
explanations is still an urgent task. In this paper, we first propose a
multi-semantic image recognition model, which enables human beings to
understand the decision-making process of the neural network. Then, we presents
a new evaluation index, which can quantitatively assess the model
interpretability. We also comprehensively summarize the semantic information
that affects the image classification results in the judgment process of neural
networks. Finally, this paper also exhibits the relevant baseline performance
with current state-of-the-art deep learning models.

    

### [[2109.13539] Extracting Attentive Social Temporal Excitation for Sequential Recommendation](http://arxiv.org/abs/2109.13539)


  In collaborative filtering, it is an important way to make full use of social
information to improve the recommendation quality, which has been proved to be
effective because user behavior will be affected by her friends. However,
existing works leverage the social relationship to aggregate user features from
friends' historical behavior sequences in a user-level indirect paradigm. A
significant defect of the indirect paradigm is that it ignores the temporal
relationships between behavior events across users. In this paper, we propose a
novel time-aware sequential recommendation framework called Social Temporal
Excitation Networks (STEN), which introduces temporal point processes to model
the fine-grained impact of friends' behaviors on the user s dynamic interests
in an event-level direct paradigm. Moreover, we propose to decompose the
temporal effect in sequential recommendation into social mutual temporal effect
and ego temporal effect. Specifically, we employ a social heterogeneous graph
embedding layer to refine user representation via structural information. To
enhance temporal information propagation, STEN directly extracts the
fine-grained temporal mutual influence of friends' behaviors through the
mutually exciting temporal network. Besides, the user s dynamic interests are
captured through the self-exciting temporal network. Extensive experiments on
three real-world datasets show that STEN outperforms state-of-the-art baseline
methods. Moreover, STEN provides event-level recommendation explainability,
which is also illustrated experimentally.

    

### [[2109.13563] Agreeing to Disagree: Annotating Offensive Language Datasets with Annotators' Disagreement](http://arxiv.org/abs/2109.13563)


  Since state-of-the-art approaches to offensive language detection rely on
supervised learning, it is crucial to quickly adapt them to the continuously
evolving scenario of social media. While several approaches have been proposed
to tackle the problem from an algorithmic perspective, so to reduce the need
for annotated data, less attention has been paid to the quality of these data.
Following a trend that has emerged recently, we focus on the level of agreement
among annotators while selecting data to create offensive language datasets, a
task involving a high level of subjectivity. Our study comprises the creation
of three novel datasets of English tweets covering different topics and having
five crowd-sourced judgments each. We also present an extensive set of
experiments showing that selecting training and test data according to
different levels of annotators' agreement has a strong effect on classifiers
performance and robustness. Our findings are further validated in cross-domain
experiments and studied using a popular benchmark dataset. We show that such
hard cases, where low agreement is present, are not necessarily due to
poor-quality annotation and we advocate for a higher presence of ambiguous
cases in future datasets, particularly in test sets, to better account for the
different points of view expressed online.

    

### [[2109.13626] An Efficient Network Design for Face Video Super-resolution](http://arxiv.org/abs/2109.13626)


  Face video super-resolution algorithm aims to reconstruct realistic face
details through continuous input video sequences. However, existing video
processing algorithms usually contain redundant parameters to guarantee
different super-resolution scenes. In this work, we focus on super-resolution
of face areas in original video scenes, while rest areas are interpolated. This
specific super-resolved task makes it possible to cut redundant parameters in
general video super-resolution networks. We construct a dataset consisting
entirely of face video sequences for network training and evaluation, and
conduct hyper-parameter optimization in our experiments. We use three combined
strategies to optimize the network parameters with a simultaneous
train-evaluation method to accelerate optimization process. Results show that
simultaneous train-evaluation method improves the training speed and
facilitates the generation of efficient networks. The generated network can
reduce at least 52.4% parameters and 20.7% FLOPs, achieve better performance on
PSNR, SSIM compared with state-of-art video super-resolution algorithms. When
processing 36x36x1x3 input video frame sequences, the efficient network
provides 47.62 FPS real-time processing performance. We name our proposal as
hyper-parameter optimization for face Video Super-Resolution (HO-FVSR), which
is open-sourced at this https URL.

    

### [[2109.13737] Evolving Evolutionary Algorithms using Multi Expression Programming](http://arxiv.org/abs/2109.13737)


  Finding the optimal parameter setting (i.e. the optimal population size, the
optimal mutation probability, the optimal evolutionary model etc) for an
Evolutionary Algorithm (EA) is a difficult task. Instead of evolving only the
parameters of the algorithm we will evolve an entire EA capable of solving a
particular problem. For this purpose the Multi Expression Programming (MEP)
technique is used. Each MEP chromosome will encode multiple EAs. An
nongenerational EA for function optimization is evolved in this paper.
Numerical experiments show the effectiveness of this approach.

    

### [[2109.13738] Searching for a practical evidence of the No Free Lunch theorems](http://arxiv.org/abs/2109.13738)


  According to the No Free Lunch (NFL) theorems all black-box algorithms
perform equally well when compared over the entire set of optimization
problems. An important problem related to NFL is finding a test problem for
which a given algorithm is better than another given algorithm. Of high
interest is finding a function for which Random Search is better than another
standard evolutionary algorithm. In this paper, we propose an evolutionary
approach for solving this problem: we will evolve test functions for which a
given algorithm A is better than another given algorithm B. Two ways for
representing the evolved functions are employed: as GP trees and as binary
strings. Several numerical experiments involving NFL-style Evolutionary
Algorithms for function optimization are performed. The results show the
effectiveness of the proposed approach. Several test functions for which Random
Search performs better than all other considered algorithms have been evolved.

    

### [[2109.13744] Death in Genetic Algorithms](http://arxiv.org/abs/2109.13744)


  Death has long been overlooked in evolutionary algorithms. Recent research
has shown that death (when applied properly) can benefit the overall fitness of
a population and can outperform sub-sections of a population that are
"immortal" when allowed to evolve together in an environment [1]. In this
paper, we strive to experimentally determine whether death is an adapted trait
and whether this adaptation can be used to enhance our implementations of
conventional genetic algorithms. Using some of the most widely accepted
evolutionary death and aging theories, we observed that senescent death (in
various forms) can lower the total run-time of genetic algorithms, increase the
optimality of a solution, and decrease the variance in an algorithm's
performance. We believe that death-enhanced genetic algorithms can accomplish
this through their unique ability to backtrack out of and/or avoid getting
trapped in local optima altogether.

    

### [[2109.13767] Identifying and Mitigating Gender Bias in Hyperbolic Word Embeddings](http://arxiv.org/abs/2109.13767)


  Euclidean word embedding models such as GloVe and Word2Vec have been shown to
reflect human-like gender biases. In this paper, we extend the study of gender
bias to the recently popularized hyperbolic word embeddings. We propose
gyrocosine bias, a novel measure for quantifying gender bias in hyperbolic word
representations and observe a significant presence of gender bias. To address
this problem, we propose Poincaré Gender Debias (PGD), a novel debiasing
procedure for hyperbolic word representations. Experiments on a suit of
evaluation tests show that PGD effectively reduces bias while adding a minimal
semantic offset.

    

### [[2109.13827] Intelligent Decision Assistance Versus Automated Decision-Making: Enhancing Knowledge Work Through Explainable Artificial Intelligence](http://arxiv.org/abs/2109.13827)


  While recent advances in AI-based automated decision-making have shown many
benefits for businesses and society, they also come at a cost. It has for long
been known that a high level of automation of decisions can lead to various
drawbacks, such as automation bias and deskilling. In particular, the
deskilling of knowledge workers is a major issue, as they are the same people
who should also train, challenge and evolve AI. To address this issue, we
conceptualize a new class of DSS, namely Intelligent Decision Assistance (IDA)
based on a literature review of two different research streams -- DSS and
automation. IDA supports knowledge workers without influencing them through
automated decision-making. Specifically, we propose to use techniques of
Explainable AI (XAI) while withholding concrete AI recommendations. To test
this conceptualization, we develop hypotheses on the impacts of IDA and provide
first evidence for their validity based on empirical studies in the literature.

    

### [[2109.13855] Chekhov's Gun Recognition](http://arxiv.org/abs/2109.13855)


  Chekhov's gun is a dramatic principle stating that every element in a story
must be necessary, and irrelevant elements should be removed. This paper
presents a new natural language processing task - Chekhov's gun recognition or
(CGR) - recognition of entities that are pivotal for the development of the
plot. Though similar to classical Named Entity Recognition (NER) it has
profound differences and is crucial for the tasks of narrative processing,
since Chekhov's guns have a profound impact on the causal relationship in a
story. The paper presents a new benchmark dataset for the CGR task that
includes 5550 descriptions with one or more Chekhov's Gun in each and validates
the task on two more datasets available in the natural language processing
(NLP) literature.

    

### [[2109.13857] ViT Cane: Visual Assistant for the Visually Impaired](http://arxiv.org/abs/2109.13857)


  Blind and visually challenged face multiple issues with navigating the world
independently. Some of these challenges include finding the shortest path to a
destination and detecting obstacles from a distance. To tackle this issue, this
paper proposes ViT Cane, which leverages a vision transformer model in order to
detect obstacles in real-time. Our entire system consists of a Pi Camera Module
v2, Raspberry Pi 4B with 8GB Ram and 4 motors. Based on tactile input using the
4 motors, the obstacle detection model is highly efficient in helping visually
impaired navigate unknown terrain and is designed to be easily reproduced. The
paper discusses the utility of a Visual Transformer model in comparison to
other CNN based models for this specific application. Through rigorous testing,
the proposed obstacle detection model has achieved higher performance on the
Common Object in Context (COCO) data set than its CNN counterpart.
Comprehensive field tests were conducted to verify the effectiveness of our
system for holistic indoor understanding and obstacle avoidance.

    

### [[2109.13885] Turning old models fashion again: Recycling classical CNN networks using the Lattice Transformation](http://arxiv.org/abs/2109.13885)


  In the early 1990s, the first signs of life of the CNN era were given: LeCun
et al. proposed a CNN model trained by the backpropagation algorithm to
classify low-resolution images of handwritten digits. Undoubtedly, it was a
breakthrough in the field of computer vision. But with the rise of other
classification methods, it fell out fashion. That was until 2012, when
Krizhevsky et al. revived the interest in CNNs by exhibiting considerably
higher image classification accuracy on the ImageNet challenge. Since then, the
complexity of the architectures are exponentially increasing and many
structures are rapidly becoming obsolete. Using multistream networks as a base
and the feature infusion precept, we explore the proposed LCNN cross-fusion
strategy to use the backbones of former state-of-the-art networks on image
classification in order to discover if the technique is able to put these
designs back in the game. In this paper, we showed that we can obtain an
increase of accuracy up to 63.21% on the NORB dataset we comparing with the
original structure. However, no technique is definitive. While our goal is to
try to reuse previous state-of-the-art architectures with few modifications, we
also expose the disadvantages of our explored strategy.

    

### [[2109.13892] Temporal Information and Event Markup Language: TIE-ML Markup Process and Schema Version 1.0](http://arxiv.org/abs/2109.13892)


  Temporal Information and Event Markup Language (TIE-ML) is a markup strategy
and annotation schema to improve the productivity and accuracy of temporal and
event related annotation of corpora to facilitate machine learning based model
training. For the annotation of events, temporal sequencing, and durations, it
is significantly simpler by providing an extremely reduced tag set for just
temporal relations and event enumeration. In comparison to other standards, as
for example the Time Markup Language (TimeML), it is much easier to use by
dropping sophisticated formalisms, theoretical concepts, and annotation
approaches. Annotations of corpora using TimeML can be mapped to TIE-ML with a
loss, and TIE-ML annotations can be fully mapped to TimeML with certain
under-specification.

    

### [[2109.13893] Explainable Machine Larning for liver transplantation](http://arxiv.org/abs/2109.13893)


  In this work, we present a flexible method for explaining, in human readable
terms, the predictions made by decision trees used as decision support in liver
transplantation. The decision trees have been obtained through machine learning
applied on a dataset collected at the liver transplantation unit at the
Coruña University Hospital Center and are used to predict long term (five
years) survival after transplantation. The method we propose is based on the
representation of the decision tree as a set of rules in a logic program (LP)
that is further annotated with text messages. This logic program is then
processed using the tool xclingo (based on Answer Set Programming) that allows
building compound explanations depending on the annotation text and the rules
effectively fired when a given input is provided. We explore two alternative LP
encodings: one in which rules respect the tree structure (more convenient to
reflect the learning process) and one where each rule corresponds to a
(previously simplified) tree path (more readable for decision making).

    

### [[1904.01497] Air Taxi Skyport Location Problem for Airport Access](http://arxiv.org/abs/1904.01497)


  Witnessing the rapid progress and accelerated commercialization made in
recent years for the introduction of air taxi services in near future across
metropolitan cities, our research focuses on one of the most important
consideration for such services, i.e., infrastructure planning (also known as
skyports). We consider design of skyport locations for air taxis accessing
airports, where we present the skyport location problem as a modified
single-allocation p-hub median location problem integrating choice-constrained
user mode choice behavior into the decision process. Our approach focuses on
two alternative objectives i.e., maximizing air taxi ridership and maximizing
air taxi revenue. The proposed models in the study incorporate trade-offs
between trip length and trip cost based on mode choice behavior of travelers to
determine optimal choices of skyports in an urban city. We examine the
sensitivity of skyport locations based on two objectives, three air taxi
pricing strategies, and varying transfer times at skyports. A case study of New
York City is conducted considering a network of 149 taxi zones and 3 airports
with over 20 million for-hire-vehicles trip data to the airports to discuss
insights around the choice of skyport locations in the city, and demand
allocation to different skyports under various parameter settings. Results
suggest that a minimum of 9 skyports located between Manhattan, Queens and
Brooklyn can adequately accommodate the airport access travel needs and are
sufficiently stable against transfer time increases. Findings from this study
can help air taxi providers strategize infrastructure design options and
investment decisions based on skyport location choices.

    

### [[1906.11110] Rethinking Formal Models of Partially Observable Multiagent Decision Making](http://arxiv.org/abs/1906.11110)


  Multiagent decision-making in partially observable environments is usually
modelled as either an extensive-form game (EFG) in game theory or a partially
observable stochastic game (POSG) in multiagent reinforcement learning (MARL).
One issue with the current situation is that while most practical problems can
be modelled in both formalisms, the relationship of the two models is unclear,
which hinders the transfer of ideas between the two communities. A second issue
is that while EFGs have recently seen significant algorithmic progress, their
classical formalization is unsuitable for efficient presentation of the
underlying ideas, such as those around decomposition.
To solve the first issue, we introduce factored-observation stochastic games
(FOSGs), a minor modification of the POSG formalism which distinguishes between
private and public observation and thereby greatly simplifies decomposition. To
remedy the second issue, we show that FOSGs and POSGs are naturally connected
to EFGs: by "unrolling" a FOSG into its tree form, we obtain an EFG.
Conversely, any perfect-recall timeable EFG corresponds to some underlying FOSG
in this manner. Moreover, this relationship justifies several minor
modifications to the classical EFG formalization that recently appeared as an
implicit response to the model's issues with decomposition. Finally, we
illustrate the transfer of ideas between EFGs and MARL by presenting three key
EFG techniques -- counterfactual regret minimization, sequence form, and
decomposition -- in the FOSG framework.

    

### [[2002.09440] A Toolkit for Generating Code Knowledge Graphs](http://arxiv.org/abs/2002.09440)


  Knowledge graphs have been proven extremely useful in powering diverse
applications in semantic search and natural language understanding. In this
paper, we present GraphGen4Code, a toolkit to build code knowledge graphs that
can similarly power various applications such as program search, code
understanding, bug detection, and code automation. GraphGen4Code uses generic
techniques to capture code semantics with the key nodes in the graph
representing classes, functions, and methods. Edges indicate function usage
(e.g., how data flows through function calls, as derived from program analysis
of real code), and documentation about functions (e.g., code documentation,
usage documentation, or forum discussions such as StackOverflow). Our toolkit
uses named graphs in RDF to model graphs per program, or can output graphs as
JSON. We show the scalability of the toolkit by applying it to 1.3 million
Python files drawn from GitHub, 2,300 Python modules, and 47 million forum
posts. This results in an integrated code graph with over 2 billion triples. We
make the toolkit to build such graphs as well as the sample extraction of the 2
billion triples graph publicly available to the community for use.

    

### [[2008.00520] Statistical Inference of Minimally Complex Models](http://arxiv.org/abs/2008.00520)


  Finding the model that best describes a high dimensional dataset is a
daunting task. For binary data, we show that this becomes feasible when
restricting the search to a family of simple models, that we call Minimally
Complex Models (MCMs). These are spin models, with interactions of arbitrary
order, that are composed of independent components of minimal complexity
(Beretta et al., 2018). They tend to be simple in information theoretic terms,
which means that they are well-fitted to specific types of data, and are
therefore easy to falsify. We show that Bayesian model selection restricted to
these models is computationally feasible and has many other advantages. First,
their evidence, which trades off goodness-of-fit against model complexity, can
be computed easily without any parameter fitting. This allows selecting the
best MCM among all, even though the number of models is astronomically large.
Furthermore, MCMs can be inferred and sampled from without any computational
effort. Finally, model selection among MCMs is invariant with respect to
changes in the representation of the data. MCMs portray the structure of
dependencies among variables in a simple way, as illustrated in several
examples, and thus provide robust predictions on dependencies in the data. MCMs
contain interactions of any order between variables, and thus may reveal the
presence of interactions of order higher than pairwise.

    

### [[2103.13020] deGraphCS: Embedding Variable-based Flow Graph for Neural Code Search](http://arxiv.org/abs/2103.13020)


  With the rapid increase in the amount of public code repositories, developers
maintain a great desire to retrieve precise code snippets by using natural
language. Despite existing deep learning based approaches(e.g., DeepCS and
MMAN) have provided the end-to-end solutions (i.e., accepts natural language as
queries and shows related code fragments retrieved directly from code corpus),
the accuracy of code search in the large-scale repositories is still limited by
the code representation (e.g., AST) and modeling (e.g., directly fusing the
features in the attention stage). In this paper, we propose a novel learnable
deep Graph for Code Search (calleddeGraphCS), to transfer source code into
variable-based flow graphs based on the intermediate representation technique,
which can model code semantics more precisely compared to process the code as
text directly or use the syntactic tree representation. Furthermore, we propose
a well-designed graph optimization mechanism to refine the code representation,
and apply an improved gated graph neural network to model variable-based flow
graphs. To evaluate the effectiveness of deGraphCS, we collect a large-scale
dataset from GitHub containing 41,152 code snippets written in C language, and
reproduce several typical deep code search methods for comparison. Besides, we
design a qualitative user study to verify the practical value of our approach.
The experimental results have shown that deGraphCS can achieve state-of-the-art
performances, and accurately retrieve code snippets satisfying the needs of the
users.

    

### [[2104.11079] Randomized Algorithms for Scientific Computing (RASC)](http://arxiv.org/abs/2104.11079)


  Randomized algorithms have propelled advances in artificial intelligence and
represent a foundational research area in advancing AI for Science. Future
advancements in DOE Office of Science priority areas such as climate science,
astrophysics, fusion, advanced materials, combustion, and quantum computing all
require randomized algorithms for surmounting challenges of complexity,
robustness, and scalability. This report summarizes the outcomes of that
workshop, "Randomized Algorithms for Scientific Computing (RASC)," held
virtually across four days in December 2020 and January 2021.

    

### [[2105.07889] HetMAML: Task-Heterogeneous Model-Agnostic Meta-Learning for Few-Shot Learning Across Modalities](http://arxiv.org/abs/2105.07889)


  Existing gradient-based meta-learning approaches to few-shot learning assume
that all tasks have the same input feature space. However, in the real world
scenarios, there are many cases that the input structures of tasks can be
different, that is, different tasks may vary in the number of input modalities
or data types. Existing meta-learners cannot handle the heterogeneous task
distribution (HTD) as there is not only global meta-knowledge shared across
tasks but also type-specific knowledge that distinguishes each type of tasks.
To deal with task heterogeneity and promote fast within-task adaptions for each
type of tasks, in this paper, we propose HetMAML, a task-heterogeneous
model-agnostic meta-learning framework, which can capture both the
type-specific and globally shared knowledge and can achieve the balance between
knowledge customization and generalization. Specifically, we design a
multi-channel backbone module that encodes the input of each type of tasks into
the same length sequence of modality-specific embeddings. Then, we propose a
task-aware iterative feature aggregation network which can automatically take
into account the context of task-specific input structures and adaptively
project the heterogeneous input spaces to the same lower-dimensional embedding
space of concepts. Our experiments on six task-heterogeneous datasets
demonstrate that HetMAML successfully leverages type-specific and globally
shared meta-parameters for heterogeneous tasks and achieves fast within-task
adaptions for each type of tasks.

    

### [[2109.13066] Prefix-to-SQL: Text-to-SQL Generation from Incomplete User Questions](http://arxiv.org/abs/2109.13066)


  Existing text-to-SQL research only considers complete questions as the input,
but lay-users might strive to formulate a complete question. To build a smarter
natural language interface to database systems (NLIDB) that also processes
incomplete questions, we propose a new task, prefix-to-SQL which takes question
prefix from users as the input and predicts the intended SQL. We construct a
new benchmark called PAGSAS that contains 124K user question prefixes and the
intended SQL for 5 sub-tasks Advising, GeoQuery, Scholar, ATIS, and Spider.
Additionally, we propose a new metric SAVE to measure how much effort can be
saved by users. Experimental results show that PAGSAS is challenging even for
strong baseline models such as T5. As we observe the difficulty of
prefix-to-SQL is related to the number of omitted tokens, we incorporate
curriculum learning of feeding examples with an increasing number of omitted
tokens. This improves scores on various sub-tasks by as much as 9% recall
scores on sub-task GeoQuery in PAGSAS.

    

### [[2109.13101] Half a Dozen Real-World Applications of Evolutionary Multitasking and More](http://arxiv.org/abs/2109.13101)


  Until recently, the potential to transfer evolved skills across distinct
optimization problem instances (or tasks) was seldom explored in evolutionary
computation. The concept of evolutionary multitasking (EMT) fills this gap. It
unlocks a population's implicit parallelism to jointly solve a set of tasks,
hence creating avenues for skills transfer between them. Despite it being early
days, the idea of EMT has begun to show promise in a range of real-world
applications. In the backdrop of recent advances, the contribution of this
paper is twofold. First, we present a review of several application-oriented
explorations of EMT in the literature, assimilating them into half a dozen
broad categories according to their respective application areas. Each category
elaborates fundamental motivations to multitask, and presents a representative
experimental study (referred from the literature). Second, we provide a set of
recipes by which general problem formulations of practical interest, those that
cut across different disciplines, could be transformed in the new light of EMT.
We intend our discussions to underscore the practical utility of existing EMT
methods, and spark future research toward novel algorithms crafted for
real-world deployment.

    

### [[2109.13103] Efficiently solving the thief orienteering problem with a max-min ant colony optimization approach](http://arxiv.org/abs/2109.13103)


  We tackle the Thief Orienteering Problem (ThOP), which is academic
multi-component problem: it combines two classical combinatorial problems,
namely the Knapsack Problem (KP) and the Orienteering Problem (OP). In this
problem, a thief has a time limit to steal items that distributed in a given
set of cities. While traveling, the thief collects items by storing them in
their knapsack, which in turn reduces the travel speed. The thief has as the
objective to maximize the total profit of the stolen items. In this article, we
present an approach that combines swarm-intelligence with a randomized packing
heuristic. Our solution approach outperforms existing works on almost all the
432 benchmarking instances, with significant improvements.

    

### [[2109.13665] A deep dive into the accuracy of IP Geolocation Databases and its impact on online advertising](http://arxiv.org/abs/2109.13665)


  The quest for every time more personalized Internet experience relies on the
enriched contextual information about each user. Online advertising also
follows this approach. Among the context information that advertising
stakeholders leverage, location information is certainly one of them. However,
when this information is not directly available from the end users, advertising
stakeholders infer it using geolocation databases, matching IP addresses to a
position on earth. The accuracy of this approach has often been questioned in
the past: however, the reality check on an advertising DSP shows that this
technique accounts for a large fraction of the served advertisements. In this
paper, we revisit the work in the field, that is mostly from almost one decade
ago, through the lenses of big data. More specifically, we, i) benchmark two
commercial Internet geolocation databases, evaluate the quality of their
information using a ground truth database of user positions containing more
than 2 billion samples, ii) analyze the internals of these databases, devising
a theoretical upper bound for the quality of the Internet geolocation approach,
and iii) we run an empirical study that unveils the monetary impact of this
technology by considering the costs associated with a real-world ad impressions
dataset. We show that when factoring cost in, IP geolocation technology may be,
under certain campaign characteristics, a better alternative than GPS from an
economic point of view, despite its inferior performance.

    