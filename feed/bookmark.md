
## 2021-11-18

### [[2111.08780] Optimal Oblivious Reconfigurable Networks](http://arxiv.org/abs/2111.08780)


  Oblivious routing has a long history in both the theory and practice of
networking. In this work we initiate the formal study of oblivious routing in
the context of reconfigurable networks, a new architecture that has recently
come to the fore in datacenter networking. These networks allow a rapidly
changing bounded-degree pattern of interconnections between nodes, but the
network topology and the selection of routing paths must both be oblivious to
the traffic demand matrix. Our focus is on the trade-off between maximizing
throughput and minimizing latency in these networks. For every constant
throughput rate, we characterize (up to a constant factor) the minimum latency
achievable by an oblivious reconfigurable network design that satisfies the
given throughput guarantee. The trade-off between these two objectives turns
out to be surprisingly subtle: the curve depicting it has an unexpected
scalloped shape reflecting the fact that load-balancing becomes more difficult
when the average length of routing paths is not an integer because equalizing
all the path lengths is not possible. The proof of our lower bound uses LP
duality to verify that Valiant load balancing is the most efficient oblivious
routing scheme when used in combination with an optimally-designed
reconfigurable network topology. The proof of our upper bound uses an algebraic
construction in which the network nodes are identified with vectors over a
finite field, the network topology is described by either the elementary basis
or a sequence of Vandermonde matrices, and routing paths are constructed by
selecting columns of these matrices to yield the appropriate mixture of path
lengths within the shortest possible time interval.

    

### [[2111.08871] Six Critical Challenges for 6G Wireless Systems](http://arxiv.org/abs/2111.08871)


  A large number of papers are now appearing on sixth-generation (6G) wireless
systems, covering different aspects, ranging from vision, architecture,
applications, and technology breakthroughs. With cellular systems in mind, this
paper presents six critical, yet fundamental challenges that must be overcome
before development and deployment of 6G systems. These include: Opening the
sub-terahertz (sub-THz) spectrum for increased bandwidths and the ability to
utilize these bandwidths, pushing the limits of semiconductor technologies for
operation within the sub-THz bands, transceiver design and architectures to
realize the high peak data rates, and realizations of sub-millisecond latencies
at the network-level to achieve the 6G key performance indicators.
Additionally, since 6G systems will not be introduced in a green fields
environment, backwards compatibility with existing systems is discussed. Where
possible, we present practical solutions to realize the identified challenges.

    

### [[2111.08936] Efficient Large-Scale Multiple Migration Planning and Scheduling in SDN-enabled Edge Computing](http://arxiv.org/abs/2111.08936)


  The containerized services allocated in the mobile edge clouds bring up the
opportunity for large-scale and real-time applications to have low latency
responses. Meanwhile, live container migration is introduced to support dynamic
resource management and users' mobility. However, with the expansion of network
topology scale and increasing migration requests, the current multiple
migration planning and scheduling algorithms of cloud data centers can not suit
large-scale scenarios in edge computing. The user mobility-induced live
migrations in edge computing require near real-time level scheduling.
Therefore, in this paper, through the Software-Defined Networking (SDN)
controller, the resource competitions among live migrations are modeled as a
dynamic resource dependency graph. We propose an iterative Maximal Independent
Set (MIS)-based multiple migration planning and scheduling algorithm. Using
real-world mobility traces of taxis and telecom base station coordinates, the
evaluation results indicate that our solution can efficiently schedule multiple
live container migrations in large-scale edge computing environments. It
improves the processing time by 3000 times compared with the state-of-the-art
migration planning algorithm in clouds while providing guaranteed migration
performance for time-critical migrations.

    

### [[2111.08942] CAMIG: Concurrency-Aware Live Migration Management of Multiple Virtual Machines in SDN-enabled Clouds](http://arxiv.org/abs/2111.08942)


  By integrating Software-Defined Networking and cloud computing, virtualized
networking and computing resources can be dynamically reallocated through live
migration of Virtual Machines (VMs). Dynamic resource management such as load
balancing and energy-saving policies can request multiple migrations when the
algorithms are triggered periodically. There exist notable research efforts in
dynamic resource management that alleviate single migration overheads, such as
single migration time and co-location interference while selecting the
potential VMs and migration destinations. However, by neglecting the resource
dependency among potential migration requests, the existing solutions of
dynamic resource management can result in the Quality of Service (QoS)
degradation and Service Level Agreement (SLA) violations during the migration
schedule. Therefore, it is essential to integrate both single and multiple
migration overheads into VM reallocation planning. In this paper, we propose a
concurrency-aware multiple migration selector that operates based on the
maximal cliques and independent sets of the resource dependency graph of
multiple migration requests. Our proposed method can be integrated with
existing dynamic resource management policies. The experimental results
demonstrate that our solution efficiently minimizes migration interference and
shortens the convergence time of reallocation by maximizing the multiple
migration performance while achieving the objective of dynamic resource
management.

    

### [[2111.08943] Edge Computing in IoT: A 6G Perspective](http://arxiv.org/abs/2111.08943)


  Edge computing is one of the key driving forces to enable Beyond 5G (B5G) and
6G networks. Due to the unprecedented increase in traffic volumes and
computation demands of future networks, Multi-access Edge Computing (MEC) is
considered as a promising solution to provide cloud-computing capabilities
within the radio access network (RAN) closer to the end users. There has been a
huge amount of research on MEC and its potential applications; however, very
little has been said about the key factors of MEC deployment to meet the
diverse demands of future applications. In this article, we present key
considerations for edge deployments in B5G/6G networks including edge
architecture, server location and capacity, user density, security etc. We
further provide state-of-the-art edge-centric services in future B5G/6G
networks. Lastly, we present some interesting insights and open research
problems in edge computing for 6G networks.

    

### [[2111.08956] Intelligence Reflecting Surface-Aided Integrated Data and Energy Networking Coexisting D2D Communications](http://arxiv.org/abs/2111.08956)


  In this paper, we consider an integrated data and energy network and D2D
communication coexistence (DED2D) system. The DED2D system allows a base
station (BS) to transfer data to information-demanded users (IUs) and energy to
energy-demanded users (EUs), i.e., using a time-fraction-based information and
energy transfer (TFIET) scheme. Furthermore, the DED2D system enables D2D
communications to share spectrum with the BS. Therefore, the DED2D system
addresses the growth of energy and spectrum demands of the next generation
networks. However, the interference caused by the D2D communications and
propagation loss of wireless links can significantly degrade the data
throughput of IUs. To deal with the issues, we propose to deploy an intelligent
reflecting surface (IRS) in the DED2D system. Then, we formulate an
optimization problem that aims to optimize the information beamformer for the
IUs, energy beamformer for EUs, time fractions of the TFIET, transmit power of
D2D transmitters, and reflection coefficients of the IRS to maximize IUs' worse
throughput while satisfying the harvested energy requirement of EUs and D2D
rate threshold. The max-min throughput optimization problem is computationally
intractable, and we develop an alternating descent algorithm to resolve it with
low computational complexity. The simulation results demonstrate the
effectiveness of the proposed algorithm.

    

### [[2111.09061] Exploring Unsupervised Learning Methods for Automated Protocol Analysis](http://arxiv.org/abs/2111.09061)


  The ability to analyse and differentiate network protocol traffic is crucial
for network resource management to provide differentiated services by Telcos.
Automated Protocol Analysis (APA) is crucial to significantly improve
efficiency and reduce reliance on human experts. There are numerous automated
state-of-the-art unsupervised methods for clustering unknown protocols in APA.
However, many such methods have not been sufficiently explored using diverse
test datasets. Thus failing to demonstrate their robustness to generalise. This
study proposed a comprehensive framework to evaluate various combinations of
feature extraction and clustering methods in APA. It also proposed a novel
approach to automate selection of dataset dependent model parameters for
feature extraction, resulting in improved performance. Promising results of a
novel field-based tokenisation approach also led to our proposal of a novel
automated hybrid approach for feature extraction and clustering of unknown
protocols in APA. Our proposed hybrid approach performed the best in 7 out of 9
of the diverse test datasets, thus displaying the robustness to generalise
across diverse unknown protocols. It also outperformed the unsupervised
clustering technique in state-of-the-art open-source APA tool, NETZOB in all
test datasets.

    

### [[2111.09161] MASS: Mobile Autonomous Station Simulation](http://arxiv.org/abs/2111.09161)


  We propose a set of tools to replay wireless network traffic traces, while
preserving the privacy of the original traces. Traces are generated by a user-
and context-aware trained generative adversarial network (GAN).
The replay allows for realistic traces from any number of users and of any
trace duration to be produced given contextual parameters like the type of
application and the real-time signal strength.
We demonstrate the usefulness of the tools in three replay scenarios: Linux-
and Android-station experiments and NS3 simulations.
We also evaluate the ability of the GAN model to generate traces that retain
key statistical properties of the original traces such as feature correlation,
statistical moments, and novelty. Our results show that we beat both
traditional statistical distribution fitting approaches as well as a
state-of-the-art GAN time series generator across these metrics. The ability of
our GAN model to generate any number of user traces regardless of the number of
users in the original trace also makes our tools more practically applicable
compared to previous GAN approaches.
Furthermore, we present a use case where our tools were employed in a Wi-Fi
research experiment.

    

### [[2111.09217] Information Freshness in Multi-Hop Wireless Networks](http://arxiv.org/abs/2111.09217)


  We consider the problem of minimizing age of information in multihop wireless
networks and propose three classes of policies to solve the problem -
stationary randomized, age difference, and age debt. For the unicast setting
with fixed routes between each source-destination pair, we first develop a
procedure to find age optimal Stationary Randomized policies. These policies
are easy to implement and allow us to derive closed-form expression for average
AoI. Next, for the same unicast setting, we develop a class of heuristic
policies, called Age Difference, based on the idea that if neighboring nodes
try to reduce their age differential then all nodes will have fresher updates.
This approach is useful in practice since it relies only on the local age
differential between nodes to make scheduling decisions. Finally, we propose
the class of policies called Age Debt, which can handle 1) non-linear AoI cost
functions; 2) unicast, multicast and broadcast flows; and 3) no fixed routes
specified per flow beforehand. Here, we convert AoI optimization problems into
equivalent network stability problems and use Lyapunov drift to find scheduling
and routing schemes that stabilize the network. We also provide numerical
results comparing our proposed classes of policies with the best known
scheduling and routing schemes available in the literature for a wide variety
of network settings.

    

### [[2111.09281] An Experimental Study of Latency for IEEE 802.11be Multi-link Operation](http://arxiv.org/abs/2111.09281)


  Will Multi-Link Operation (MLO) be able to improve the latency of Wi-Fi
networks? MLO is one of the most disruptive MAC-layer techniques included in
the IEEE 802.11be amendment. It allows a device to use multiple radios
simultaneously and in a coordinated way, providing a new framework to improve
the WLAN throughput and latency. In this paper, we investigate the potential
latency benefits of MLO by using a large dataset containing 5 GHz spectrum
occupancy measurements. Experimental results show that when the channels are
symmetrically occupied, MLO can improve latency by one order of magnitude. In
contrast, in asymmetrically occupied channels, MLO can sometimes be detrimental
and increase latency. To address this case, we introduce Opportunistic
Simultaneous Transmit and Receive (STR+) channel access and study its benefits.

    

### [[2111.07911] On the Tradeoff between Energy, Precision, and Accuracy in Federated Quantized Neural Networks](http://arxiv.org/abs/2111.07911)


  Deploying federated learning (FL) over wireless networks with
resource-constrained devices requires balancing between accuracy, energy
efficiency, and precision. Prior art on FL often requires devices to train deep
neural networks (DNNs) using a 32-bit precision level for data representation
to improve accuracy. However, such algorithms are impractical for
resource-constrained devices since DNNs could require execution of millions of
operations. Thus, training DNNs with a high precision level incurs a high
energy cost for FL. In this paper, a quantized FL framework, that represents
data with a finite level of precision in both local training and uplink
transmission, is proposed. Here, the finite level of precision is captured
through the use of quantized neural networks (QNNs) that quantize weights and
activations in fixed-precision format. In the considered FL model, each device
trains its QNN and transmits a quantized training result to the base station.
Energy models for the local training and the transmission with the quantization
are rigorously derived. An energy minimization problem is formulated with
respect to the level of precision while ensuring convergence. To solve the
problem, we first analytically derive the FL convergence rate and use a line
search method. Simulation results show that our FL framework can reduce energy
consumption by up to 53% compared to a standard FL model. The results also shed
light on the tradeoff between precision, energy, and accuracy in FL over
wireless networks.

    

### [[2111.08701] Interpretability Aware Model Training to Improve Robustness against Out-of-Distribution Magnetic Resonance Images in Alzheimer's Disease Classification](http://arxiv.org/abs/2111.08701)


  Owing to its pristine soft-tissue contrast and high resolution, structural
magnetic resonance imaging (MRI) is widely applied in neurology, making it a
valuable data source for image-based machine learning (ML) and deep learning
applications. The physical nature of MRI acquisition and reconstruction,
however, causes variations in image intensity, resolution, and signal-to-noise
ratio. Since ML models are sensitive to such variations, performance on
out-of-distribution data, which is inherent to the setting of a deployed
healthcare ML application, typically drops below acceptable levels. We propose
an interpretability aware adversarial training regime to improve robustness
against out-of-distribution samples originating from different MRI hardware.
The approach is applied to 1.5T and 3T MRIs obtained from the Alzheimer's
Disease Neuroimaging Initiative database. We present preliminary results
showing promising performance on out-of-distribution samples.

    

### [[2111.08706] How and When Random Feedback Works: A Case Study of Low-Rank Matrix Factorization](http://arxiv.org/abs/2111.08706)


  The success of gradient descent in ML and especially for learning neural
networks is remarkable and robust. In the context of how the brain learns, one
aspect of gradient descent that appears biologically difficult to realize (if
not implausible) is that its updates rely on feedback from later layers to
earlier layers through the same connections. Such bidirected links are
relatively few in brain networks, and even when reciprocal connections exist,
they may not be equi-weighted. Random Feedback Alignment (Lillicrap et al.,
2016), where the backward weights are random and fixed, has been proposed as a
bio-plausible alternative and found to be effective empirically. We investigate
how and when feedback alignment (FA) works, focusing on one of the most basic
problems with layered structure -- low-rank matrix factorization. In this
problem, given a matrix $Y_{n\times m}$, the goal is to find a low rank
factorization $Z_{n \times r}W_{r \times m}$ that minimizes the error
$\|ZW-Y\|_F$. Gradient descent solves this problem optimally. We show that FA
converges to the optimal solution when $r\ge \mbox{rank}(Y)$. We also shed
light on how FA works. It is observed empirically that the forward weight
matrices and (random) feedback matrices come closer during FA updates. Our
analysis rigorously derives this phenomenon and shows how it facilitates
convergence of FA. We also show that FA can be far from optimal when $r <
\mbox{rank}(Y)$. This is the first provable separation result between gradient
descent and FA. Moreover, the representations found by gradient descent and FA
can be almost orthogonal even when their error $\|ZW-Y\|_F$ is approximately
equal.

    

### [[2111.08710] CNN Filter Learning from Drawn Markers for the Detection of Suggestive Signs of COVID-19 in CT Images](http://arxiv.org/abs/2111.08710)


  Early detection of COVID-19 is vital to control its spread. Deep learning
methods have been presented to detect suggestive signs of COVID-19 from chest
CT images. However, due to the novelty of the disease, annotated volumetric
data are scarce. Here we propose a method that does not require either large
annotated datasets or backpropagation to estimate the filters of a
convolutional neural network (CNN). For a few CT images, the user draws markers
at representative normal and abnormal regions. The method generates a feature
extractor composed of a sequence of convolutional layers, whose kernels are
specialized in enhancing regions similar to the marked ones, and the decision
layer of our CNN is a support vector machine. As we have no control over the CT
image acquisition, we also propose an intensity standardization approach. Our
method can achieve mean accuracy and kappa values of $0.97$ and $0.93$,
respectively, on a dataset with 117 CT images extracted from different sites,
surpassing its counterpart in all scenarios.

    

### [[2111.08711] Two-step adversarial debiasing with partial learning -- medical image case-studies](http://arxiv.org/abs/2111.08711)


  The use of artificial intelligence (AI) in healthcare has become a very
active research area in the last few years. While significant progress has been
made in image classification tasks, only a few AI methods are actually being
deployed in hospitals. A major hurdle in actively using clinical AI models
currently is the trustworthiness of these models. More often than not, these
complex models are black boxes in which promising results are generated.
However, when scrutinized, these models begin to reveal implicit biases during
the decision making, such as detecting race and having bias towards ethnic
groups and subpopulations. In our ongoing study, we develop a two-step
adversarial debiasing approach with partial learning that can reduce the racial
disparity while preserving the performance of the targeted task. The
methodology has been evaluated on two independent medical image case-studies -
chest X-ray and mammograms, and showed promises in bias reduction while
preserving the targeted performance.

    

### [[2111.08712] Automatic Semantic Segmentation of the Lumbar Spine. Clinical Applicability in a Multi-parametric and Multi-centre MRI study](http://arxiv.org/abs/2111.08712)


  One of the major difficulties in medical image segmentation is the high
variability of these images, which is caused by their origin (multi-centre),
the acquisition protocols (multi-parametric), as well as the variability of
human anatomy, the severity of the illness, the effect of age and gender, among
others. The problem addressed in this work is the automatic semantic
segmentation of lumbar spine Magnetic Resonance images using convolutional
neural networks. The purpose is to assign a classes label to each pixel of an
image. Classes were defined by radiologists and correspond to different
structural elements like vertebrae, intervertebral discs, nerves, blood
vessels, and other tissues. The proposed network topologies are variants of the
U-Net architecture. Several complementary blocks were used to define the
variants: Three types of convolutional blocks, spatial attention models, deep
supervision and multilevel feature extractor. This document describes the
topologies and analyses the results of the neural network designs that obtained
the most accurate segmentations. Several of the proposed designs outperform the
standard U-Net used as baseline, especially when used in ensembles where the
output of multiple neural networks is combined according to different
strategies.

    

### [[2111.08723] Who Decides if AI is Fair? The Labels Problem in Algorithmic Auditing](http://arxiv.org/abs/2111.08723)


  Labelled "ground truth" datasets are routinely used to evaluate and audit AI
algorithms applied in high-stakes settings. However, there do not exist widely
accepted benchmarks for the quality of labels in these datasets. We provide
empirical evidence that quality of labels can significantly distort the results
of algorithmic audits in real-world settings. Using data annotators typically
hired by AI firms in India, we show that fidelity of the ground truth data can
lead to spurious differences in performance of ASRs between urban and rural
populations. After a rigorous, albeit expensive, label cleaning process, these
disparities between groups disappear. Our findings highlight how trade-offs
between label quality and data annotation costs can complicate algorithmic
audits in practice. They also emphasize the need for development of
consensus-driven, widely accepted benchmarks for label quality.

    

### [[2111.08733] Learning Provably Robust Motion Planners Using Funnel Libraries](http://arxiv.org/abs/2111.08733)


  This paper presents an approach for learning motion planners that are
accompanied with probabilistic guarantees of success on new environments that
hold uniformly for any disturbance to the robot's dynamics within an admissible
set. We achieve this by bringing together tools from generalization theory and
robust control. First, we curate a library of motion primitives where the
robustness of each primitive is characterized by an over-approximation of the
forward reachable set, i.e., a "funnel". Then, we optimize probably
approximately correct (PAC)-Bayes generalization bounds for training our
planner to compose these primitives such that the entire funnels respect the
problem specification. We demonstrate the ability of our approach to provide
strong guarantees on two simulated examples: (i) navigation of an autonomous
vehicle under external disturbances on a five-lane highway with multiple
vehicles, and (ii) navigation of a drone across an obstacle field in the
presence of wind disturbances.

    

### [[2111.08749] SMACE: A New Method for the Interpretability of Composite Decision Systems](http://arxiv.org/abs/2111.08749)


  Interpretability is a pressing issue for decision systems. Many post hoc
methods have been proposed to explain the predictions of any machine learning
model. However, business processes and decision systems are rarely centered
around a single, standalone model. These systems combine multiple models that
produce key predictions, and then apply decision rules to generate the final
decision. To explain such decision, we present SMACE, Semi-Model-Agnostic
Contextual Explainer, a novel interpretability method that combines a geometric
approach for decision rules with existing post hoc solutions for machine
learning models to generate an intuitive feature ranking tailored to the end
user. We show that established model-agnostic approaches produce poor results
in this framework.

    

### [[2111.08759] On the Potential of Execution Traces for Batch Processing Workload Optimization in Public Clouds](http://arxiv.org/abs/2111.08759)


  With the growing amount of data, data processing workloads and the management
of their resource usage becomes increasingly important. Since managing a
dedicated infrastructure is in many situations infeasible or uneconomical,
users progressively execute their respective workloads in the cloud. As the
configuration of workloads and resources is often challenging, various methods
have been proposed that either quickly profile towards a good configuration or
determine one based on data from previous runs. Still, performance data to
train such methods is often lacking and must be costly collected.
In this paper, we propose a collaborative approach for sharing anonymized
workload execution traces among users, mining them for general patterns, and
exploiting clusters of historical workloads for future optimizations. We
evaluate our prototype implementation for mining workload execution graphs on a
publicly available trace dataset and demonstrate the predictive value of
workload clusters determined through traces only.

    

### [[2111.08761] Stronger Generalization Guarantees for Robot Learning by Combining Generative Models and Real-World Data](http://arxiv.org/abs/2111.08761)


  We are motivated by the problem of learning policies for robotic systems with
rich sensory inputs (e.g., vision) in a manner that allows us to guarantee
generalization to environments unseen during training. We provide a framework
for providing such generalization guarantees by leveraging a finite dataset of
real-world environments in combination with a (potentially inaccurate)
generative model of environments. The key idea behind our approach is to
utilize the generative model in order to implicitly specify a prior over
policies. This prior is updated using the real-world dataset of environments by
minimizing an upper bound on the expected cost across novel environments
derived via Probably Approximately Correct (PAC)-Bayes generalization theory.
We demonstrate our approach on two simulated systems with nonlinear/hybrid
dynamics and rich sensing modalities: (i) quadrotor navigation with an onboard
vision sensor, and (ii) grasping objects using a depth sensor. Comparisons with
prior work demonstrate the ability of our approach to obtain stronger
generalization guarantees by utilizing generative models. We also present
hardware experiments for validating our bounds for the grasping task.

    

### [[2111.08792] PredProp: Bidirectional Stochastic Optimization with Precision Weighted Predictive Coding](http://arxiv.org/abs/2111.08792)


  We present PredProp, a method for bidirectional, parallel and local
optimisation of weights, activities and precision in neural networks. PredProp
jointly addresses inference and learning, scales learning rates dynamically and
weights gradients by the curvature of the loss function by optimizing
prediction error precision. PredProp optimizes network parameters with
Stochastic Gradient Descent and error forward propagation based strictly on
prediction errors and variables locally available to each layer. Neighboring
layers optimise shared activity variables so that prediction errors can
propagate forward in the network, while predictions propagate backwards. This
process minimises the negative Free Energy, or evidence lower bound of the
entire network. We show that networks trained with PredProp resemble gradient
based predictive coding when the number of weights between neighboring activity
variables is one. In contrast to related work, PredProp generalizes towards
backward connections of arbitrary depth and optimizes precision for any deep
network architecture. Due to the analogy between prediction error precision and
the Fisher information for each layer, PredProp implements a form of Natural
Gradient Descent. When optimizing DNN models, layer-wise PredProp renders the
model a bidirectional predictive coding network. Alternatively DNNs can
parameterize the weights between two activity variables. We evaluate PredProp
for dense DNNs on simple inference, learning and combined tasks. We show that,
without an explicit sampling step in the network, PredProp implements a form of
variational inference that allows to learn disentangled embeddings from low
amounts of data and leave evaluation on more complex tasks and datasets to
future work.

    

### [[2111.08794] Investigating Conversion from Mild Cognitive Impairment to Alzheimer's Disease using Latent Space Manipulation](http://arxiv.org/abs/2111.08794)


  Alzheimer's disease is the most common cause of dementia that affects
millions of lives worldwide. Investigating the underlying causes and risk
factors of Alzheimer's disease is essential to prevent its progression. Mild
Cognitive Impairment (MCI) is considered an intermediate stage before
Alzheimer's disease. Early prediction of the conversion from the MCI to
Alzheimer's is crucial to take necessary precautions for decelerating the
progression and developing suitable treatments. In this study, we propose a
deep learning framework to discover the variables which are identifiers of the
conversion from MCI to Alzheimer's disease. In particular, the latent space of
a variational auto-encoder network trained with the MCI and Alzheimer's
patients is manipulated to obtain the significant attributes and decipher their
behavior that leads to the conversion from MCI to Alzheimer's disease. By
utilizing a generative decoder and the dimensions that lead to the Alzheimer's
diagnosis, we generate synthetic dementia patients from MCI patients in the
dataset. Experimental results show promising quantitative and qualitative
results on one of the most extensive and commonly used Alzheimer's disease
neuroimaging datasets in literature.

    

### [[2111.08805] Online Estimation and Optimization of Utility-Based Shortfall Risk](http://arxiv.org/abs/2111.08805)


  Utility-Based Shortfall Risk (UBSR) is a risk metric that is increasingly
popular in financial applications, owing to certain desirable properties that
it enjoys. We consider the problem of estimating UBSR in a recursive setting,
where samples from the underlying loss distribution are available
one-at-a-time. We cast the UBSR estimation problem as a root finding problem,
and propose stochastic approximation-based estimations schemes. We derive
non-asymptotic bounds on the estimation error in the number of samples. We also
consider the problem of UBSR optimization within a parameterized class of
random variables. We propose a stochastic gradient descent based algorithm for
UBSR optimization, and derive non-asymptotic bounds on its convergence.

    

### [[2111.08819] CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms](http://arxiv.org/abs/2111.08819)


  CleanRL is an open-source library that provides high-quality single-file
implementations of Deep Reinforcement Learning algorithms. It provides a
simpler yet scalable developing experience by having a straightforward codebase
and integrating production tools to help interact and scale experiments. In
CleanRL, we put all details of an algorithm into a single file, making these
performance-relevant details easier to recognize. Additionally, an experiment
tracking feature is available to help log metrics, hyperparameters, videos of
an agent's gameplay, dependencies, and more to the cloud. Despite succinct
implementations, we have also designed tools to help scale, at one point
orchestrating experiments on more than 2000 machines simultaneously via Docker
and cloud providers. Finally, we have ensured the quality of the
implementations by benchmarking against a variety of environments. The source
code of CleanRL can be found at this https URL


### [[2111.08823] Meta-Auto-Decoder for Solving Parametric Partial Differential Equations](http://arxiv.org/abs/2111.08823)


  Partial Differential Equations (PDEs) are ubiquitous in many disciplines of
science and engineering and notoriously difficult to solve. In general,
closed-form solutions of PDEs are unavailable and numerical approximation
methods are computationally expensive. The parameters of PDEs are variable in
many applications, such as inverse problems, control and optimization, risk
assessment, and uncertainty quantification. In these applications, our goal is
to solve parametric PDEs rather than one instance of them. Our proposed
approach, called Meta-Auto-Decoder (MAD), treats solving parametric PDEs as a
meta-learning problem and utilizes the Auto-Decoder structure in
\cite{park2019deepsdf} to deal with different tasks/PDEs. Physics-informed
losses induced from the PDE governing equations and boundary conditions is used
as the training losses for different tasks. The goal of MAD is to learn a good
model initialization that can generalize across different tasks, and eventually
enables the unseen task to be learned faster. The inspiration of MAD comes from
(conjectured) low-dimensional structure of parametric PDE solutions and we
explain our approach from the perspective of manifold learning. Finally, we
demonstrate the power of MAD though extensive numerical studies, including
Burgers' equation, Laplace's equation and time-domain Maxwell's equations. MAD
exhibits faster convergence speed without losing the accuracy compared with
other deep learning methods.

    

### [[2111.08834] Federated Learning for Smart Healthcare: A Survey](http://arxiv.org/abs/2111.08834)


  Recent advances in communication technologies and Internet-of-Medical-Things
have transformed smart healthcare enabled by artificial intelligence (AI).
Traditionally, AI techniques require centralized data collection and processing
that may be infeasible in realistic healthcare scenarios due to the high
scalability of modern healthcare networks and growing data privacy concerns.
Federated Learning (FL), as an emerging distributed collaborative AI paradigm,
is particularly attractive for smart healthcare, by coordinating multiple
clients (e.g., hospitals) to perform AI training without sharing raw data.
Accordingly, we provide a comprehensive survey on the use of FL in smart
healthcare. First, we present the recent advances in FL, the motivations, and
the requirements of using FL in smart healthcare. The recent FL designs for
smart healthcare are then discussed, ranging from resource-aware FL, secure and
privacy-aware FL to incentive FL and personalized FL. Subsequently, we provide
a state-of-the-art review on the emerging applications of FL in key healthcare
domains, including health data management, remote health monitoring, medical
imaging, and COVID-19 detection. Several recent FL-based smart healthcare
projects are analyzed, and the key lessons learned from the survey are also
highlighted. Finally, we discuss interesting research challenges and possible
directions for future FL research in smart healthcare.

    

### [[2111.08840] Online Advertising Revenue Forecasting: An Interpretable Deep Learning Approach](http://arxiv.org/abs/2111.08840)


  Online advertising revenues account for an increasing share of publishers'
revenue streams, especially for small and medium-sized publishers who depend on
the advertisement networks of tech companies such as Google and Facebook. Thus
publishers may benefit significantly from accurate online advertising revenue
forecasts to better manage their website monetization strategies. However,
publishers who only have access to their own revenue data lack a holistic view
of the total ad market of publishers, which in turn limits their ability to
generate insights into their own future online advertising revenues. To address
this business issue, we leverage a proprietary database encompassing Google
Adsense revenues from a large collection of publishers in diverse areas. We
adopt the Temporal Fusion Transformer (TFT) model, a novel attention-based
architecture to predict publishers' advertising revenues. We leverage multiple
covariates, including not only the publisher's own characteristics but also
other publishers' advertising revenues. Our prediction results outperform
several benchmark deep-learning time-series forecast models over multiple time
horizons. Moreover, we interpret the results by analyzing variable importance
weights to identify significant features and self-attention weights to reveal
persistent temporal patterns.

    

### [[2111.08848] GNN-DSE: Automated Accelerator Optimization Aided by Graph Neural Networks](http://arxiv.org/abs/2111.08848)


  High-level synthesis (HLS) has freed the computer architects from developing
their designs in a very low-level language and needing to exactly specify how
the data should be transferred in register-level. With the help of HLS, the
hardware designers must describe only a high-level behavioral flow of the
design. Despite this, it still can take weeks to develop a high-performance
architecture mainly because there are many design choices at a higher level
that requires more time to explore. It also takes several minutes to hours to
get feedback from the HLS tool on the quality of each design candidate. In this
paper, we propose to solve this problem by modeling the HLS tool with a graph
neural network (GNN) that is trained to be used for a wide range of
applications. The experimental results demonstrate that by employing the
GNN-based model, we are able to estimate the quality of design in milliseconds
with high accuracy which can help us search through the solution space very
quickly.

    

### [[2111.08851] Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities](http://arxiv.org/abs/2111.08851)


  In recent times, deep neural networks achieved outstanding predictive
performance on various classification and pattern recognition tasks. However,
many real-world prediction problems have ordinal response variables, and this
ordering information is ignored by conventional classification losses such as
the multi-category cross-entropy. Ordinal regression methods for deep neural
networks address this. One such method is the CORAL method, which is based on
an earlier binary label extension framework and achieves rank consistency among
its output layer tasks by imposing a weight-sharing constraint. However, while
earlier experiments showed that CORAL's rank consistency is beneficial for
performance, the weight-sharing constraint could severely restrict the
expressiveness of a deep neural network. In this paper, we propose an
alternative method for rank-consistent ordinal regression that does not require
a weight-sharing constraint in a neural network's fully connected output layer.
We achieve this rank consistency by a novel training scheme using conditional
training sets to obtain the unconditional rank probabilities through applying
the chain rule for conditional probability distributions. Experiments on
various datasets demonstrate the efficacy of the proposed method to utilize the
ordinal target information, and the absence of the weight-sharing restriction
improves the performance substantially compared to the CORAL reference
approach.

    

### [[2111.08856] Fairness Testing of Deep Image Classification with Adequacy Metrics](http://arxiv.org/abs/2111.08856)


  As deep image classification applications, e.g., face recognition, become
increasingly prevalent in our daily lives, their fairness issues raise more and
more concern. It is thus crucial to comprehensively test the fairness of these
applications before deployment. Existing fairness testing methods suffer from
the following limitations: 1) applicability, i.e., they are only applicable for
structured data or text without handling the high-dimensional and abstract
domain sampling in the semantic level for image classification applications; 2)
functionality, i.e., they generate unfair samples without providing testing
criterion to characterize the model's fairness adequacy. To fill the gap, we
propose DeepFAIT, a systematic fairness testing framework specifically designed
for deep image classification applications. DeepFAIT consists of several
important components enabling effective fairness testing of deep image
classification applications: 1) a neuron selection strategy to identify the
fairness-related neurons; 2) a set of multi-granularity adequacy metrics to
evaluate the model's fairness; 3) a test selection algorithm for fixing the
fairness issues efficiently. We have conducted experiments on widely adopted
large-scale face recognition applications, i.e., VGGFace and FairFace. The
experimental results confirm that our approach can effectively identify the
fairness-related neurons, characterize the model's fairness, and select the
most valuable test cases to mitigate the model's fairness issues.

    

### [[2111.08857] SEIHAI: A Sample-efficient Hierarchical AI for the MineRL Competition](http://arxiv.org/abs/2111.08857)


  The MineRL competition is designed for the development of reinforcement
learning and imitation learning algorithms that can efficiently leverage human
demonstrations to drastically reduce the number of environment interactions
needed to solve the complex \emph{ObtainDiamond} task with sparse rewards. To
address the challenge, in this paper, we present \textbf{SEIHAI}, a
\textbf{S}ample-\textbf{e}ff\textbf{i}cient \textbf{H}ierarchical \textbf{AI},
that fully takes advantage of the human demonstrations and the task structure.
Specifically, we split the task into several sequentially dependent subtasks,
and train a suitable agent for each subtask using reinforcement learning and
imitation learning. We further design a scheduler to select different agents
for different subtasks automatically. SEIHAI takes the first place in the
preliminary and final of the NeurIPS-2020 MineRL competition.

    

### [[2111.08861] Label efficient two-sample test](http://arxiv.org/abs/2111.08861)


  Two-sample tests evaluate whether two samples are realizations of the same
distribution (the null hypothesis) or two different distributions (the
alternative hypothesis). In the traditional formulation of this problem, the
statistician has access to both the measurements (feature variables) and the
group variable (label variable). However, in several important applications,
feature variables can be easily measured but the binary label variable is
unknown and costly to obtain. In this paper, we consider this important
variation on the classical two-sample test problem and pose it as a problem of
obtaining the labels of only a small number of samples in service of performing
a two-sample test. We devise a label efficient three-stage framework: firstly,
a classifier is trained with samples uniformly labeled to model the posterior
probabilities of the labels; secondly, an innovative query scheme dubbed
\emph{bimodal query} is used to query labels of samples from both classes with
maximum posterior probabilities, and lastly, the classical Friedman-Rafsky (FR)
two-sample test is performed on the queried samples. Our theoretical analysis
shows that bimodal query is optimal for the FR test under reasonable conditions
and that the three-stage framework controls the Type I error. Extensive
experiments performed on synthetic, benchmark, and application-specific
datasets demonstrate that the three-stage framework has decreased Type II error
over uniform querying and certainty-based querying with same number of labels
while controlling the Type I error.

    

### [[2111.08862] Max-Min Grouped Bandits](http://arxiv.org/abs/2111.08862)


  In this paper, we introduce a multi-armed bandit problem termed max-min
grouped bandits, in which the arms are arranged in possibly-overlapping groups,
and the goal is to find a group whose worst arm has the highest mean reward.
This problem is of interest in applications such as recommendation systems, and
is also closely related to widely-studied robust optimization problems. We
present two algorithms based successive elimination and robust optimization,
and derive upper bounds on the number of samples to guarantee finding a max-min
optimal or near-optimal group, as well as an algorithm-independent lower bound.
We discuss the degree of tightness of our bounds in various cases of interest,
and the difficulties in deriving uniformly tight bounds.

    

### [[2111.08872] TorchGeo: deep learning with geospatial data](http://arxiv.org/abs/2111.08872)


  Remotely sensed geospatial data are critical for applications including
precision agriculture, urban planning, disaster monitoring and response, and
climate change research, among others. Deep learning methods are particularly
promising for modeling many remote sensing tasks given the success of deep
neural networks in similar computer vision tasks and the sheer volume of
remotely sensed imagery available. However, the variance in data collection
methods and handling of geospatial metadata make the application of deep
learning methodology to remotely sensed data nontrivial. For example, satellite
imagery often includes additional spectral bands beyond red, green, and blue
and must be joined to other geospatial data sources that can have differing
coordinate systems, bounds, and resolutions. To help realize the potential of
deep learning for remote sensing applications, we introduce TorchGeo, a Python
library for integrating geospatial data into the PyTorch deep learning
ecosystem. TorchGeo provides data loaders for a variety of benchmark datasets,
composable datasets for generic geospatial data sources, samplers for
geospatial data, and transforms that work with multispectral imagery. TorchGeo
is also the first library to provide pre-trained models for multispectral
satellite imagery (e.g. models that use all bands from the Sentinel 2
satellites), allowing for advances in transfer learning on downstream remote
sensing tasks with limited labeled data. We use TorchGeo to create reproducible
benchmark results on existing datasets and benchmark our proposed method for
preprocessing geospatial imagery on-the-fly. TorchGeo is open-source and
available on GitHub: this https URL.

    

### [[2111.08874] GN-Transformer: Fusing Sequence and Graph Representation for Improved Code Summarization](http://arxiv.org/abs/2111.08874)


  As opposed to natural languages, source code understanding is influenced by
grammatical relationships between tokens regardless of their identifier name.
Graph representations of source code such as Abstract Syntax Tree (AST) can
capture relationships between tokens that are not obvious from the source code.
We propose a novel method, GN-Transformer to learn end-to-end on a fused
sequence and graph modality we call Syntax-Code-Graph (SCG). GN-Transformer
expands on Graph Networks (GN) framework using a self-attention mechanism. SCG
is the result of the early fusion between a source code snippet and the AST
representation. We perform experiments on the structure of SCG, an ablation
study on the model design, and the hyper-parameters to conclude that the
performance advantage is from the fused representation. The proposed methods
achieve state-of-the-art performance in two code summarization datasets and
across three automatic code summarization metrics (BLEU, METEOR, ROUGE-L). We
further evaluate the human perceived quality of our model and previous work
with an expert-user study. Our model outperforms the state-of-the-art in human
perceived quality and accuracy.

    

### [[2111.08878] FAIRLEARN:Configurable and Interpretable Algorithmic Fairness](http://arxiv.org/abs/2111.08878)


  The rapid growth of data in the recent years has led to the development of
complex learning algorithms that are often used to make decisions in real
world. While the positive impact of the algorithms has been tremendous, there
is a need to mitigate any bias arising from either training samples or implicit
assumptions made about the data samples. This need becomes critical when
algorithms are used in automated decision making systems that can hugely impact
people's lives.
Many approaches have been proposed to make learning algorithms fair by
detecting and mitigating bias in different stages of optimization. However, due
to a lack of a universal definition of fairness, these algorithms optimize for
a particular interpretation of fairness which makes them limited for real world
use. Moreover, an underlying assumption that is common to all algorithms is the
apparent equivalence of achieving fairness and removing bias. In other words,
there is no user defined criteria that can be incorporated into the
optimization procedure for producing a fair algorithm. Motivated by these
shortcomings of existing methods, we propose the FAIRLEARN procedure that
produces a fair algorithm by incorporating user constraints into the
optimization procedure. Furthermore, we make the process interpretable by
estimating the most predictive features from data. We demonstrate the efficacy
of our approach on several real world datasets using different fairness
criteria.

    

### [[2111.08885] Jump Interval-Learning for Individualized Decision Making](http://arxiv.org/abs/2111.08885)


  An individualized decision rule (IDR) is a decision function that assigns
each individual a given treatment based on his/her observed characteristics.
Most of the existing works in the literature consider settings with binary or
finitely many treatment options. In this paper, we focus on the continuous
treatment setting and propose a jump interval-learning to develop an
individualized interval-valued decision rule (I2DR) that maximizes the expected
outcome. Unlike IDRs that recommend a single treatment, the proposed I2DR
yields an interval of treatment options for each individual, making it more
flexible to implement in practice. To derive an optimal I2DR, our jump
interval-learning method estimates the conditional mean of the outcome given
the treatment and the covariates via jump penalized regression, and derives the
corresponding optimal I2DR based on the estimated outcome regression function.
The regressor is allowed to be either linear for clear interpretation or deep
neural network to model complex treatment-covariates interactions. To implement
jump interval-learning, we develop a searching algorithm based on dynamic
programming that efficiently computes the outcome regression function.
Statistical properties of the resulting I2DR are established when the outcome
regression function is either a piecewise or continuous function over the
treatment space. We further develop a procedure to infer the mean outcome under
the (estimated) optimal policy. Extensive simulations and a real data
application to a warfarin study are conducted to demonstrate the empirical
validity of the proposed I2DR.

    

### [[2111.08888] Random Graph-Based Neuromorphic Learning with a Layer-Weaken Structure](http://arxiv.org/abs/2111.08888)


  Unified understanding of neuro networks (NNs) gets the users into great
trouble because they have been puzzled by what kind of rules should be obeyed
to optimize the internal structure of NNs. Considering the potential capability
of random graphs to alter how computation is performed, we demonstrate that
they can serve as architecture generators to optimize the internal structure of
NNs. To transform the random graph theory into an NN model with practical
meaning and based on clarifying the input-output relationship of each neuron,
we complete data feature mapping by calculating Fourier Random Features (FRFs).
Under the usage of this low-operation cost approach, neurons are assigned to
several groups of which connection relationships can be regarded as uniform
representations of random graphs they belong to, and random arrangement fuses
those neurons to establish the pattern matrix, markedly reducing manual
participation and computational cost without the fixed and deep architecture.
Leveraging this single neuromorphic learning model termed random graph-based
neuro network (RGNN) we develop a joint classification mechanism involving
information interaction between multiple RGNNs and realize significant
performance improvements in supervised learning for three benchmark tasks,
whereby they effectively avoid the adverse impact of the interpretability of
NNs on the structure design and engineering practice.

    

### [[2111.08900] A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction](http://arxiv.org/abs/2111.08900)


  Climate change is posing new challenges to crop-related concerns including
food insecurity, supply stability and economic planning. As one of the central
challenges, crop yield prediction has become a pressing task in the machine
learning field. Despite its importance, the prediction task is exceptionally
complicated since crop yields depend on various factors such as weather, land
surface, soil quality as well as their interactions. In recent years, machine
learning models have been successfully applied in this domain. However, these
models either restrict their tasks to a relatively small region, or only study
over a single or few years, which makes them hard to generalize spatially and
temporally. In this paper, we introduce a novel graph-based recurrent neural
network for crop yield prediction, to incorporate both geographical and
temporal knowledge in the model, and further boost predictive power. Our method
is trained, validated, and tested on over 2000 counties from 41 states in the
US mainland, covering years from 1981 to 2019. As far as we know, this is the
first machine learning method that embeds geographical knowledge in crop yield
prediction and predicts the crop yields at county level nationwide. We also
laid a solid foundation for the comparison with other machine learning
baselines by applying well-known linear models, tree-based models, deep
learning methods and comparing their performance. Experiments show that our
proposed method consistently outperforms the existing state-of-the-art methods
on various metrics, validating the effectiveness of geospatial and temporal
information.

    

### [[2111.08911] Fast Rates for Nonparametric Online Learning: From Realizability to Learning in Games](http://arxiv.org/abs/2111.08911)


  We study fast rates of convergence in the setting of nonparametric online
regression, namely where regret is defined with respect to an arbitrary
function class which has bounded complexity. Our contributions are two-fold:
- In the realizable setting of nonparametric online regression with the
absolute loss, we propose a randomized proper learning algorithm which gets a
near-optimal mistake bound in terms of the sequential fat-shattering dimension
of the hypothesis class. In the setting of online classification with a class
of Littlestone dimension $d$, our bound reduces to $d \cdot {\rm poly} \log T$.
This result answers a question as to whether proper learners could achieve
near-optimal mistake bounds; previously, even for online classification, the
best known mistake bound was $\tilde O( \sqrt{dT})$. Further, for the
real-valued (regression) setting, the optimal mistake bound was not even known
for improper learners, prior to this work.
- Using the above result, we exhibit an independent learning algorithm for
general-sum binary games of Littlestone dimension $d$, for which each player
achieves regret $\tilde O(d^{3/4} \cdot T^{1/4})$. This result generalizes
analogous results of Syrgkanis et al. (2015) who showed that in finite games
the optimal regret can be accelerated from $O(\sqrt{T})$ in the adversarial
setting to $O(T^{1/4})$ in the game setting.
To establish the above results, we introduce several new techniques,
including: a hierarchical aggregation rule to achieve the optimal mistake bound
for real-valued classes, a multi-scale extension of the proper online
realizable learner of Hanneke et al. (2021), an approach to show that the
output of such nonparametric learning algorithms is stable, and a proof that
the minimax theorem holds in all online learnable games.

    

### [[2111.08922] Traversing the Local Polytopes of ReLU Neural Networks: A Unified Approach for Network Verification](http://arxiv.org/abs/2111.08922)


  Although neural networks (NNs) with ReLU activation functions have found
success in a wide range of applications, their adoption in risk-sensitive
settings has been limited by the concerns on robustness and interpretability.
Previous works to examine robustness and to improve interpretability partially
exploited the piecewise linear function form of ReLU NNs. In this paper, we
explore the unique topological structure that ReLU NNs create in the input
space, identifying the adjacency among the partitioned local polytopes and
developing a traversing algorithm based on this adjacency. Our polytope
traversing algorithm can be adapted to verify a wide range of network
properties related to robustness and interpretability, providing an unified
approach to examine the network behavior. As the traversing algorithm
explicitly visits all local polytopes, it returns a clear and full picture of
the network behavior within the traversed region. The time and space complexity
of the traversing algorithm is determined by the number of a ReLU NN's
partitioning hyperplanes passing through the traversing region.

    

### [[2111.08947] Fast Yet Effective Machine Unlearning](http://arxiv.org/abs/2111.08947)


  Unlearning the data observed during the training of a machine learning (ML)
model is an important task that can play a pivotal role in fortifying the
privacy and security of ML-based applications. This paper raises the following
questions: (i) can we unlearn a class/classes of data from a ML model without
looking at the full training data even once? (ii) can we make the process of
unlearning fast and scalable to large datasets, and generalize it to different
deep networks? We introduce a novel machine unlearning framework with
error-maximizing noise generation and impair-repair based weight manipulation
that offers an efficient solution to the above questions. An error-maximizing
noise matrix is learned for the class to be unlearned using the original model.
The noise matrix is used to manipulate the model weights to unlearn the
targeted class of data. We introduce impair and repair steps for a controlled
manipulation of the network weights. In the impair step, the noise matrix along
with a very high learning rate is used to induce sharp unlearning in the model.
Thereafter, the repair step is used to regain the overall performance. With
very few update steps, we show excellent unlearning while substantially
retaining the overall model accuracy. Unlearning multiple classes requires a
similar number of update steps as for the single class, making our approach
scalable to large problems. Our method is quite efficient in comparison to the
existing methods, works for multi-class unlearning, doesn't put any constraints
on the original optimization mechanism or network design, and works well in
both small and large-scale vision tasks. This work is an important step towards
fast and easy implementation of unlearning in deep networks. We will make the
source code publicly available.

    

### [[2111.08952] A Generalized Proportionate-Type Normalized Subband Adaptive Filter](http://arxiv.org/abs/2111.08952)


  We show that a new design criterion, i.e., the least squares on subband
errors regularized by a weighted norm, can be used to generalize the
proportionate-type normalized subband adaptive filtering (PtNSAF) framework.
The new criterion directly penalizes subband errors and includes a sparsity
penalty term which is minimized using the damped regularized Newton's method.
The impact of the proposed generalized PtNSAF (GPtNSAF) is studied for the
system identification problem via computer simulations. Specifically, we study
the effects of using different numbers of subbands and various sparsity penalty
terms for quasi-sparse, sparse, and dispersive systems. The results show that
the benefit of increasing the number of subbands is larger than promoting
sparsity of the estimated filter coefficients when the target system is
quasi-sparse or dispersive. On the other hand, for sparse target systems,
promoting sparsity becomes more important. More importantly, the two aspects
provide complementary and additive benefits to the GPtNSAF for speeding up
convergence.

    

### [[2111.08953] Three approaches to supervised learning for compositional data with pairwise logratios](http://arxiv.org/abs/2111.08953)


  The common approach to compositional data analysis is to transform the data
by means of logratios. Logratios between pairs of compositional parts (pairwise
logratios) are the easiest to interpret in many research problems. When the
number of parts is large, some form of logratio selection is a must, for
instance by means of an unsupervised learning method based on a stepwise
selection of the pairwise logratios that explain the largest percentage of the
logratio variance in the compositional dataset. In this article we present
three alternative stepwise supervised learning methods to select the pairwise
logratios that best explain a dependent variable in a generalized linear model,
each geared for a specific problem. The first method features unrestricted
search, where any pairwise logratio can be selected. This method has a complex
interpretation if some pairs of parts in the logratios overlap, but it leads to
the most accurate predictions. The second method restricts parts to occur only
once, which makes the corresponding logratios intuitively interpretable. The
third method uses additive logratios, so that $K-1$ selected logratios involve
exactly $K$ parts. This method in fact searches for the subcomposition with the
highest explanatory power. Once the subcomposition is identified, the
researcher's favourite logratio representation may be used in subsequent
analyses, not only pairwise logratios. Our methodology allows logratios or
non-compositional covariates to be forced into the models based on theoretical
knowledge, and various stopping criteria are available based on information
measures or statistical significance with the Bonferroni correction. We present
an illustration of the three approaches on a dataset from a study predicting
Crohn's disease. The first method excels in terms of predictive power, and the
other two in interpretability.

    

### [[2111.08960] Compositional Transformers for Scene Generation](http://arxiv.org/abs/2111.08960)


  We introduce the GANformer2 model, an iterative object-oriented transformer,
explored for the task of generative modeling. The network incorporates strong
and explicit structural priors, to reflect the compositional nature of visual
scenes, and synthesizes images through a sequential process. It operates in two
stages: a fast and lightweight planning phase, where we draft a high-level
scene layout, followed by an attention-based execution phase, where the layout
is being refined, evolving into a rich and detailed picture. Our model moves
away from conventional black-box GAN architectures that feature a flat and
monolithic latent space towards a transparent design that encourages
efficiency, controllability and interpretability. We demonstrate GANformer2's
strengths and qualities through a careful evaluation over a range of datasets,
from multi-object CLEVR scenes to the challenging COCO images, showing it
successfully achieves state-of-the-art performance in terms of visual quality,
diversity and consistency. Further experiments demonstrate the model's
disentanglement and provide a deeper insight into its generative process, as it
proceeds step-by-step from a rough initial sketch, to a detailed layout that
accounts for objects' depths and dependencies, and up to the final
high-resolution depiction of vibrant and intricate real-world scenes. See
this https URL for model implementation.

    

### [[2111.08988] LVAC: Learned Volumetric Attribute Compression for Point Clouds using Coordinate Based Networks](http://arxiv.org/abs/2111.08988)


  We consider the attributes of a point cloud as samples of a vector-valued
volumetric function at discrete positions. To compress the attributes given the
positions, we compress the parameters of the volumetric function. We model the
volumetric function by tiling space into blocks, and representing the function
over each block by shifts of a coordinate-based, or implicit, neural network.
Inputs to the network include both spatial coordinates and a latent vector per
block. We represent the latent vectors using coefficients of the
region-adaptive hierarchical transform (RAHT) used in the MPEG geometry-based
point cloud codec G-PCC. The coefficients, which are highly compressible, are
rate-distortion optimized by back-propagation through a rate-distortion
Lagrangian loss in an auto-decoder configuration. The result outperforms RAHT
by 2--4 dB. This is the first work to compress volumetric functions represented
by local coordinate-based neural networks. As such, we expect it to be
applicable beyond point clouds, for example to compression of high-resolution
neural radiance fields.

    

### [[2111.08995] Self-Learning Tuning for Post-Silicon Validation](http://arxiv.org/abs/2111.08995)


  Increasing complexity of modern chips makes design validation more difficult.
Existing approaches are not able anymore to cope with the complexity of tasks
such as robust performance tuning in post-silicon validation. Therefore, we
propose a novel approach based on learn-to-optimize and reinforcement learning
in order to solve complex and mixed-type tuning tasks in a efficient and robust
way.

    

### [[2111.09014] Subject Enveloped Deep Sample Fuzzy Ensemble Learning Algorithm of Parkinson's Speech Data](http://arxiv.org/abs/2111.09014)


  Parkinson disease (PD)'s speech recognition is an effective way for its
diagnosis, which has become a hot and difficult research area in recent years.
As we know, there are large corpuses (segments) within one subject. However,
too large segments will increase the complexity of the classification model.
Besides, the clinicians interested in finding diagnostic speech markers that
reflect the pathology of the whole subject. Since the optimal relevant features
of each speech sample segment are different, it is difficult to find the
uniform diagnostic speech markers. Therefore, it is necessary to reconstruct
the existing large segments within one subject into few segments even one
segment within one subject, which can facilitate the extraction of relevant
speech features to characterize diagnostic markers for the whole subject. To
address this problem, an enveloped deep speech sample learning algorithm for
Parkinson's subjects based on multilayer fuzzy c-mean (MlFCM) clustering and
interlayer consistency preservation is proposed in this paper. The algorithm
can be used to achieve intra-subject sample reconstruction for Parkinson's
disease (PD) to obtain a small number of high-quality prototype sample
segments. At the end of the paper, several representative PD speech datasets
are selected and compared with the state-of-the-art related methods,
respectively. The experimental results show that the proposed algorithm is
effective signifcantly.

    

### [[2111.09030] Trustworthy Long-Tailed Classification](http://arxiv.org/abs/2111.09030)


  Classification on long-tailed distributed data is a challenging problem,
which suffers from serious class-imbalance and accordingly unpromising
performance especially on tail classes. Recently, the ensembling based methods
achieve the state-of-the-art performance and show great potential. However,
there are two limitations for current methods. First, their predictions are not
trustworthy for failure-sensitive applications. This is especially harmful for
the tail classes where the wrong predictions is basically frequent. Second,
they assign unified numbers of experts to all samples, which is redundant for
easy samples with excessive computational cost. To address these issues, we
propose a Trustworthy Long-tailed Classification (TLC) method to jointly
conduct classification and uncertainty estimation to identify hard samples in a
multi-expert framework. Our TLC obtains the evidence-based uncertainty (EvU)
and evidence for each expert, and then combines these uncertainties and
evidences under the Dempster-Shafer Evidence Theory (DST). Moreover, we propose
a dynamic expert engagement to reduce the number of engaged experts for easy
samples and achieve efficiency while maintaining promising performances.
Finally, we conduct comprehensive experiments on the tasks of classification,
tail detection, OOD detection and failure prediction. The experimental results
show that the proposed TLC outperforms the state-of-the-art methods and is
trustworthy with reliable uncertainty.

    

### [[2111.09035] Multi-Attribute Relation Extraction (MARE) -- Simplifying the Application of Relation Extraction](http://arxiv.org/abs/2111.09035)


  Natural language understanding's relation extraction makes innovative and
encouraging novel business concepts possible and facilitates new digitilized
decision-making processes. Current approaches allow the extraction of relations
with a fixed number of entities as attributes. Extracting relations with an
arbitrary amount of attributes requires complex systems and costly
relation-trigger annotations to assist these systems. We introduce
multi-attribute relation extraction (MARE) as an assumption-less problem
formulation with two approaches, facilitating an explicit mapping from business
use cases to the data annotations. Avoiding elaborated annotation constraints
simplifies the application of relation extraction approaches. The evaluation
compares our models to current state-of-the-art event extraction and binary
relation extraction methods. Our approaches show improvement compared to these
on the extraction of general multi-attribute relations.

    

### [[2111.09038] A Vertical Federated Learning Method For Multi-Institutional Credit Scoring: MICS](http://arxiv.org/abs/2111.09038)


  As more and more companies store their customers' data; various information
of a person is distributed among numerous companies' databases. Different
industrial sectors carry distinct features about the same customers. Also,
different companies within the same industrial sector carry similar kinds of
data about the customers with different data representations. Cooperation
between companies from different industrial sectors, called vertical
cooperation, and between the companies within the same sector, called
horizontal cooperation, can lead to more accurate machine learning models and
better estimations in tasks such as credit scoring. However, data privacy
regulations and compatibility issues for different data representations are
huge obstacles to cooperative model training. By proposing the training
framework MICS and experimentation on several numerical data sets, we showed
that companies would have an incentive to cooperate with other companies from
their sector and with other industrial sectors to jointly train more robust and
accurate global models without explicitly sharing their customers' private
data.

    

### [[2111.09043] ORSA: Outlier Robust Stacked Aggregation for Best- and Worst-Case Approximations of Ensemble Systems\](http://arxiv.org/abs/2111.09043)


  In recent years, the usage of ensemble learning in applications has grown
significantly due to increasing computational power allowing the training of
large ensembles in reasonable time frames. Many applications, e.g., malware
detection, face recognition, or financial decision-making, use a finite set of
learning algorithms and do aggregate them in a way that a better predictive
performance is obtained than any other of the individual learning algorithms.
In the field of Post-Silicon Validation for semiconductor devices (PSV), data
sets are typically provided that consist of various devices like, e.g., chips
of different manufacturing lines. In PSV, the task is to approximate the
underlying function of the data with multiple learning algorithms, each trained
on a device-specific subset, instead of improving the performance of arbitrary
classifiers on the entire data set. Furthermore, the expectation is that an
unknown number of subsets describe functions showing very different
characteristics. Corresponding ensemble members, which are called outliers, can
heavily influence the approximation. Our method aims to find a suitable
approximation that is robust to outliers and represents the best or worst case
in a way that will apply to as many types as possible. A 'soft-max' or
'soft-min' function is used in place of a maximum or minimum operator. A Neural
Network (NN) is trained to learn this 'soft-function' in a two-stage process.
First, we select a subset of ensemble members that is representative of the
best or worst case. Second, we combine these members and define a weighting
that uses the properties of the Local Outlier Factor (LOF) to increase the
influence of non-outliers and to decrease outliers. The weighting ensures
robustness to outliers and makes sure that approximations are suitable for most
types.

    

### [[2111.09052] High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency](http://arxiv.org/abs/2111.09052)


  This paper presents an end-to-end text-to-speech system with low latency on a
CPU, suitable for real-time applications. The system is composed of an
autoregressive attention-based sequence-to-sequence acoustic model and the
LPCNet vocoder for waveform generation. An acoustic model architecture that
adopts modules from both the Tacotron 1 and 2 models is proposed, while
stability is ensured by using a recently proposed purely location-based
attention mechanism, suitable for arbitrary sentence length generation. During
inference, the decoder is unrolled and acoustic feature generation is performed
in a streaming manner, allowing for a nearly constant latency which is
independent from the sentence length. Experimental results show that the
acoustic model can produce feature sequences with minimal latency about 31
times faster than real-time on a computer CPU and 6.5 times on a mobile CPU,
enabling it to meet the conditions required for real-time applications on both
devices. The full end-to-end system can generate almost natural quality speech,
which is verified by listening tests.

    

### [[2111.09065] Sampling To Improve Predictions For Underrepresented Observations In Imbalanced Data](http://arxiv.org/abs/2111.09065)


  Data imbalance is common in production data, where controlled production
settings require data to fall within a narrow range of variation and data are
collected with quality assessment in mind, rather than data analytic insights.
This imbalance negatively impacts the predictive performance of models on
underrepresented observations. We propose sampling to adjust for this imbalance
with the goal of improving the performance of models trained on historical
production data. We investigate the use of three sampling approaches to adjust
for imbalance. The goal is to downsample the covariates in the training data
and subsequently fit a regression model. We investigate how the predictive
power of the model changes when using either the sampled or the original data
for training. We apply our methods on a large biopharmaceutical manufacturing
data set from an advanced simulation of penicillin production and find that
fitting a model using the sampled data gives a small reduction in the overall
predictive performance, but yields a systematically better performance on
underrepresented observations. In addition, the results emphasize the need for
alternative, fair, and balanced model evaluations.

    

### [[2111.09074] Surrogate-Assisted Genetic Algorithm for Wrapper Feature Selection](http://arxiv.org/abs/2111.09074)


  Feature selection is an intractable problem, therefore practical algorithms
often trade off the solution accuracy against the computation time. In this
paper, we propose a novel multi-stage feature selection framework utilizing
multiple levels of approximations, or surrogates. Such a framework allows for
using wrapper approaches in a much more computationally efficient way,
significantly increasing the quality of feature selection solutions achievable,
especially on large datasets. We design and evaluate a Surrogate-Assisted
Genetic Algorithm (SAGA) which utilizes this concept to guide the evolutionary
search during the early phase of exploration. SAGA only switches to evaluating
the original function at the final exploitation phase.
We prove that the run-time upper bound of SAGA surrogate-assisted stage is at
worse equal to the wrapper GA, and it scales better for induction algorithms of
high order of complexity in number of instances. We demonstrate, using 14
datasets from the UCI ML repository, that in practice SAGA significantly
reduces the computation time compared to a baseline wrapper Genetic Algorithm
(GA), while converging to solutions of significantly higher accuracy. Our
experiments show that SAGA can arrive at near-optimal solutions three times
faster than a wrapper GA, on average. We also showcase the importance of
evolution control approach designed to prevent surrogates from misleading the
evolutionary search towards false optima.

    

### [[2111.09075] Cross-lingual Low Resource Speaker Adaptation Using Phonological Features](http://arxiv.org/abs/2111.09075)


  The idea of using phonological features instead of phonemes as input to
sequence-to-sequence TTS has been recently proposed for zero-shot multilingual
speech synthesis. This approach is useful for code-switching, as it facilitates
the seamless uttering of foreign text embedded in a stream of native text. In
our work, we train a language-agnostic multispeaker model conditioned on a set
of phonologically derived features common across different languages, with the
goal of achieving cross-lingual speaker adaptation. We first experiment with
the effect of language phonological similarity on cross-lingual TTS of several
source-target language combinations. Subsequently, we fine-tune the model with
very limited data of a new speaker's voice in either a seen or an unseen
language, and achieve synthetic speech of equal quality, while preserving the
target speaker's identity. With as few as 32 and 8 utterances of target speaker
data, we obtain high speaker similarity scores and naturalness comparable to
the corresponding literature. In the extreme case of only 2 available
adaptation utterances, we find that our model behaves as a few-shot learner, as
the performance is similar in both the seen and unseen adaptation language
scenarios.

    

### [[2111.09076] Do Not Trust Prediction Scores for Membership Inference Attacks](http://arxiv.org/abs/2111.09076)


  Membership inference attacks (MIAs) aim to determine whether a specific
sample was used to train a predictive model. Knowing this may indeed lead to a
privacy breach. Arguably, most MIAs, however, make use of the model's
prediction scores - the probability of each output given some input - following
the intuition that the trained model tends to behave differently on its
training data. We argue that this is a fallacy for many modern deep network
architectures, e.g., ReLU type neural networks produce almost always high
prediction scores far away from the training data. Consequently, MIAs will
miserably fail since this behavior leads to high false-positive rates not only
on known domains but also on out-of-distribution data and implicitly acts as a
defense against MIAs. Specifically, using generative adversarial networks, we
are able to produce a potentially infinite number of samples falsely classified
as part of the training data. In other words, the threat of MIAs is
overestimated and less information is leaked than previously assumed. Moreover,
there is actually a trade-off between the overconfidence of classifiers and
their susceptibility to MIAs: the more classifiers know when they do not know,
making low confidence predictions far away from the training data, the more
they reveal the training data.

    

### [[2111.09081] Unsupervised Spectral Unmixing For Telluric Correction Using A Neural Network Autoencoder](http://arxiv.org/abs/2111.09081)


  The absorption of light by molecules in the atmosphere of Earth is a
complication for ground-based observations of astrophysical objects.
Comprehensive information on various molecular species is required to correct
for this so called telluric absorption. We present a neural network autoencoder
approach for extracting a telluric transmission spectrum from a large set of
high-precision observed solar spectra from the HARPS-N radial velocity
spectrograph. We accomplish this by reducing the data into a compressed
representation, which allows us to unveil the underlying solar spectrum and
simultaneously uncover the different modes of variation in the observed spectra
relating to the absorption of $\mathrm{H_2O}$ and $\mathrm{O_2}$ in the
atmosphere of Earth. We demonstrate how the extracted components can be used to
remove $\mathrm{H_2O}$ and $\mathrm{O_2}$ tellurics in a validation observation
with similar accuracy and at less computational expense than a synthetic
approach with molecfit.

    

### [[2111.09085] Network Generation with Differential Privacy](http://arxiv.org/abs/2111.09085)


  We consider the problem of generating private synthetic versions of
real-world graphs containing private information while maintaining the utility
of generated graphs. Differential privacy is a gold standard for data privacy,
and the introduction of the differentially private stochastic gradient descent
(DP-SGD) algorithm has facilitated the training of private neural models in a
number of domains. Recent advances in graph generation via deep generative
networks have produced several high performing models. We evaluate and compare
state-of-the-art models including adjacency matrix based models and edge based
models, and show a practical implementation that favours the edge-list approach
utilizing the Gaussian noise mechanism when evaluated on commonly used graph
datasets. Based on our findings, we propose a generative model that can
reproduce the properties of real-world networks while maintaining
edge-differential privacy. The proposed model is based on a stochastic neural
network that generates discrete edge-list samples and is trained using the
Wasserstein GAN objective with the DP-SGD optimizer. Being the first approach
to combine these beneficial properties, our model contributes to further
research on graph data privacy.

    

### [[2111.09098] Unifying Heterogenous Electronic Health Records Systems via Text-Based Code Embedding](http://arxiv.org/abs/2111.09098)


  EHR systems lack a unified code system forrepresenting medical concepts,
which acts asa barrier for the deployment of deep learningmodels in large scale
to multiple clinics and hos-pitals. To overcome this problem, we
introduceDescription-based Embedding,DescEmb, a code-agnostic representation
learning framework forEHR. DescEmb takes advantage of the flexibil-ity of
neural language understanding models toembed clinical events using their
textual descrip-tions rather than directly mapping each event toa dedicated
embedding. DescEmb outperformedtraditional code-based embedding in
extensiveexperiments, especially in a zero-shot transfertask (one hospital to
another), and was able totrain a single unified model for heterogeneousEHR
datasets.

    

### [[2111.09099] Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](http://arxiv.org/abs/2111.09099)


  Anomaly detection is commonly pursued as a one-class classification problem,
where models can only learn from normal training samples, while being evaluated
on both normal and abnormal test samples. Among the successful approaches for
anomaly detection, a distinguished category of methods relies on predicting
masked information (e.g. patches, future frames, etc.) and leveraging the
reconstruction error with respect to the masked information as an abnormality
score. Different from related methods, we propose to integrate the
reconstruction-based functionality into a novel self-supervised predictive
architectural building block. The proposed self-supervised block is generic and
can easily be incorporated into various state-of-the-art anomaly detection
methods. Our block starts with a convolutional layer with dilated filters,
where the center area of the receptive field is masked. The resulting
activation maps are passed through a channel attention module. Our block is
equipped with a loss that minimizes the reconstruction error with respect to
the masked area in the receptive field. We demonstrate the generality of our
block by integrating it into several state-of-the-art frameworks for anomaly
detection on image and video, providing empirical evidence that shows
considerable performance improvements on MVTec AD, Avenue, and ShanghaiTech.

    

### [[2111.09109] Physics-guided Loss Functions Improve Deep Learning Performance in Inverse Scattering](http://arxiv.org/abs/2111.09109)


  Solving electromagnetic inverse scattering problems (ISPs) is challenging due
to the intrinsic nonlinearity, ill-posedness, and expensive computational cost.
Recently, deep neural network (DNN) techniques have been successfully applied
on ISPs and shown potential of superior imaging over conventional methods. In
this paper, we analyse the analogy between DNN solvers and traditional
iterative algorithms and discuss how important physical phenomena cannot be
effectively incorporated in the training process. We show the importance of
including near-field priors in the learning process of DNNs. To this end, we
propose new designs of loss functions which incorporate multiple-scattering
based near-field quantities (such as scattered fields or induced currents
within domain of interest). Effects of physics-guided loss functions are
studied using a variety of numerical experiments. Pros and cons of the
investigated ISP solvers with different loss functions are summarized.

    

### [[2111.09111] Forecasting Crude Oil Price Using Event Extraction](http://arxiv.org/abs/2111.09111)


  Research on crude oil price forecasting has attracted tremendous attention
from scholars and policymakers due to its significant effect on the global
economy. Besides supply and demand, crude oil prices are largely influenced by
various factors, such as economic development, financial markets, conflicts,
wars, and political events. Most previous research treats crude oil price
forecasting as a time series or econometric variable prediction problem.
Although recently there have been researches considering the effects of
real-time news events, most of these works mainly use raw news headlines or
topic models to extract text features without profoundly exploring the event
information. In this study, a novel crude oil price forecasting framework,
AGESL, is proposed to deal with this problem. In our approach, an open domain
event extraction algorithm is utilized to extract underlying related events,
and a text sentiment analysis algorithm is used to extract sentiment from
massive news. Then a deep neural network integrating the news event features,
sentimental features, and historical price features is built to predict future
crude oil prices. Empirical experiments are performed on West Texas
Intermediate (WTI) crude oil price data, and the results show that our approach
obtains superior performance compared with several benchmark methods.

    

### [[2111.09114] Cryo-shift: Reducing domain shift in cryo-electron subtomograms with unsupervised domain adaptation and randomization](http://arxiv.org/abs/2111.09114)


  Cryo-Electron Tomography (cryo-ET) is a 3D imaging technology that enables
the visualization of subcellular structures in situ at near-atomic resolution.
Cellular cryo-ET images help in resolving the structures of macromolecules and
determining their spatial relationship in a single cell, which has broad
significance in cell and structural biology. Subtomogram classification and
recognition constitute a primary step in the systematic recovery of these
macromolecular structures. Supervised deep learning methods have been proven to
be highly accurate and efficient for subtomogram classification, but suffer
from limited applicability due to scarcity of annotated data. While generating
simulated data for training supervised models is a potential solution, a
sizeable difference in the image intensity distribution in generated data as
compared to real experimental data will cause the trained models to perform
poorly in predicting classes on real subtomograms. In this work, we present
Cryo-Shift, a fully unsupervised domain adaptation and randomization framework
for deep learning-based cross-domain subtomogram classification. We use
unsupervised multi-adversarial domain adaption to reduce the domain shift
between features of simulated and experimental data. We develop a
network-driven domain randomization procedure with `warp' modules to alter the
simulated data and help the classifier generalize better on experimental data.
We do not use any labeled experimental data to train our model, whereas some of
the existing alternative approaches require labeled experimental samples for
cross-domain classification. Nevertheless, Cryo-Shift outperforms the existing
alternative approaches in cross-domain subtomogram classification in extensive
evaluation studies demonstrated herein using both simulated and experimental
data.

    

### [[2111.09115] Using Deep Learning to Identify Patients with Cognitive Impairment in Electronic Health Records](http://arxiv.org/abs/2111.09115)


  Dementia is a neurodegenerative disorder that causes cognitive decline and
affects more than 50 million people worldwide. Dementia is under-diagnosed by
healthcare professionals - only one in four people who suffer from dementia are
diagnosed. Even when a diagnosis is made, it may not be entered as a structured
International Classification of Diseases (ICD) diagnosis code in a patient's
charts. Information relevant to cognitive impairment (CI) is often found within
electronic health records (EHR), but manual review of clinician notes by
experts is both time consuming and often prone to errors. Automated mining of
these notes presents an opportunity to label patients with cognitive impairment
in EHR data. We developed natural language processing (NLP) tools to identify
patients with cognitive impairment and demonstrate that linguistic context
enhances performance for the cognitive impairment classification task. We
fine-tuned our attention based deep learning model, which can learn from
complex language structures, and substantially improved accuracy (0.93)
relative to a baseline NLP model (0.84). Further, we show that deep learning
NLP can successfully identify dementia patients without dementia-related ICD
codes or medications.

    

### [[2111.09121] Uncertainty Quantification of Surrogate Explanations: an Ordinal Consensus Approach](http://arxiv.org/abs/2111.09121)


  Explainability of black-box machine learning models is crucial, in particular
when deployed in critical applications such as medicine or autonomous cars.
Existing approaches produce explanations for the predictions of models,
however, how to assess the quality and reliability of such explanations remains
an open question. In this paper we take a step further in order to provide the
practitioner with tools to judge the trustworthiness of an explanation. To this
end, we produce estimates of the uncertainty of a given explanation by
measuring the ordinal consensus amongst a set of diverse bootstrapped surrogate
explainers. While we encourage diversity by using ensemble techniques, we
propose and analyse metrics to aggregate the information contained within the
set of explainers through a rating scheme. We empirically illustrate the
properties of this approach through experiments on state-of-the-art
Convolutional Neural Network ensembles. Furthermore, through tailored
visualisations, we show specific examples of situations where uncertainty
estimates offer concrete actionable insights to the user beyond those arising
from standard surrogate explainers.

    

### [[2111.09124] Route Optimization via Environment-Aware Deep Network and Reinforcement Learning](http://arxiv.org/abs/2111.09124)


  Vehicle mobility optimization in urban areas is a long-standing problem in
smart city and spatial data analysis. Given the complex urban scenario and
unpredictable social events, our work focuses on developing a mobile sequential
recommendation system to maximize the profitability of vehicle service
providers (e.g., taxi drivers). In particular, we treat the dynamic route
optimization problem as a long-term sequential decision-making task. A
reinforcement-learning framework is proposed to tackle this problem, by
integrating a self-check mechanism and a deep neural network for customer
pick-up point monitoring. To account for unexpected situations (e.g., the
COVID-19 outbreak), our method is designed to be capable of handling related
environment changes with a self-adaptive parameter determination mechanism.
Based on the yellow taxi data in New York City and vicinity before and after
the COVID-19 outbreak, we have conducted comprehensive experiments to evaluate
the effectiveness of our method. The results show consistently excellent
performance, from hourly to weekly measures, to support the superiority of our
method over the state-of-the-art methods (i.e., with more than 98% improvement
in terms of the profitability for taxi drivers).

    

### [[2111.09127] Outlier Detection as Instance Selection Method for Feature Selection in Time Series Classification](http://arxiv.org/abs/2111.09127)


  In order to allow machine learning algorithms to extract knowledge from raw
data, these data must first be cleaned, transformed, and put into
machine-appropriate form. These often very time-consuming phase is referred to
as preprocessing. An important step in the preprocessing phase is feature
selection, which aims at better performance of prediction models by reducing
the amount of features of a data set. Within these datasets, instances of
different events are often imbalanced, which means that certain normal events
are over-represented while other rare events are very limited. Typically, these
rare events are of special interest since they have more discriminative power
than normal events. The aim of this work was to filter instances provided to
feature selection methods for these rare instances, and thus positively
influence the feature selection process. In the course of this work, we were
able to show that this filtering has a positive effect on the performance of
classification models and that outlier detection methods are suitable for this
filtering. For some data sets, the resulting increase in performance was only a
few percent, but for other datasets, we were able to achieve increases in
performance of up to 16 percent. This work should lead to the improvement of
the predictive models and the better interpretability of feature selection in
the course of the preprocessing phase. In the spirit of open science and to
increase transparency within our research field, we have made all our source
code and the results of our experiments available in a publicly available
repository.

    

### [[2111.09128] Smart Data Representations: Impact on the Accuracy of Deep Neural Networks](http://arxiv.org/abs/2111.09128)


  Deep Neural Networks are able to solve many complex tasks with less
engineering effort and better performance. However, these networks often use
data for training and evaluation without investigating its representation,
i.e.~the form of the used data. In the present paper, we analyze the impact of
data representations on the performance of Deep Neural Networks using energy
time series forecasting. Based on an overview of exemplary data
representations, we select four exemplary data representations and evaluate
them using two different Deep Neural Network architectures and three
forecasting horizons on real-world energy time series. The results show that,
depending on the forecast horizon, the same data representations can have a
positive or negative impact on the accuracy of Deep Neural Networks.

    

### [[2111.09138] Interpreting multi-variate models with setPCA](http://arxiv.org/abs/2111.09138)


  Principal Component Analysis (PCA) and other multi-variate models are often
used in the analysis of "omics" data. These models contain much information
which is currently neither easily accessible nor interpretable. Here we present
an algorithmic method which has been developed to integrate this information
with existing databases of background knowledge, stored in the form of known
sets (for instance genesets or pathways). To make this accessible we have
produced a Graphical User Interface (GUI) in Matlab which allows the overlay of
known set information onto the loadings plot and thus improves the
interpretability of the multi-variate model. For each known set the optimal
convex hull, covering a subset of elements from the known set, is found through
a search algorithm and displayed. In this paper we discuss two main topics; the
details of the search algorithm for the optimal convex hull for this problem
and the GUI interface which is freely available for download for academic use.

    

### [[2111.09139] Airport Taxi Time Prediction and Alerting: A Convolutional Neural Network Approach](http://arxiv.org/abs/2111.09139)


  This paper proposes a novel approach to predict and determine whether the
average taxi- out time at an airport will exceed a pre-defined threshold within
the next hour of operations. Prior work in this domain has focused exclusively
on predicting taxi-out times on a flight-by-flight basis, which requires
significant efforts and data on modeling taxiing activities from gates to
runways. Learning directly from surface radar information with minimal
processing, a computer vision-based model is proposed that incorporates airport
surface data in such a way that adaptation-specific information (e.g., runway
configuration, the state of aircraft in the taxiing process) is inferred
implicitly and automatically by Artificial Intelligence (AI).

    

### [[2111.09145] Interpretable Models via Pairwise permutations algorithm](http://arxiv.org/abs/2111.09145)


  One of the most common pitfalls often found in high dimensional biological
data sets are correlations between the features. This may lead to statistical
and machine learning methodologies overvaluing or undervaluing these correlated
predictors, while the truly relevant ones are ignored. In this paper, we will
define a new method called \textit{pairwise permutation algorithm} (PPA) with
the aim of mitigating the correlation bias in feature importance values.
Firstly, we provide a theoretical foundation, which builds upon previous work
on permutation importance. PPA is then applied to a toy data set, where we
demonstrate its ability to correct the correlation effect. We further test PPA
on a microbiome shotgun dataset, to show that the PPA is already able to obtain
biological relevant biomarkers.

    

### [[2111.09146] Rapping-Singing Voice Synthesis based on Phoneme-level Prosody Control](http://arxiv.org/abs/2111.09146)


  In this paper, a text-to-rapping/singing system is introduced, which can be
adapted to any speaker's voice. It utilizes a Tacotron-based multispeaker
acoustic model trained on read-only speech data and which provides prosody
control at the phoneme level. Dataset augmentation and additional prosody
manipulation based on traditional DSP algorithms are also investigated. The
neural TTS model is fine-tuned to an unseen speaker's limited recordings,
allowing rapping/singing synthesis with the target's speaker voice. The
detailed pipeline of the system is described, which includes the extraction of
the target pitch and duration values from an a capella song and their
conversion into target speaker's valid range of notes before synthesis. An
additional stage of prosodic manipulation of the output via WSOLA is also
investigated for better matching the target duration values. The synthesized
utterances can be mixed with an instrumental accompaniment track to produce a
complete song. The proposed system is evaluated via subjective listening tests
as well as in comparison to an available alternate system which also aims to
produce synthetic singing voice from read-only training data. Results show that
the proposed approach can produce high quality rapping/singing voice with
increased naturalness.

    

### [[2111.09159] Aggressive Q-Learning with Ensembles: Achieving Both High Sample Efficiency and High Asymptotic Performance](http://arxiv.org/abs/2111.09159)


  Recently, Truncated Quantile Critics (TQC), using distributional
representation of critics, was shown to provide state-of-the-art asymptotic
training performance on all environments from the MuJoCo continuous control
benchmark suite. Also recently, Randomized Ensemble Double Q-Learning (REDQ),
using a high update-to-data ratio and target randomization, was shown to
achieve high sample efficiency that is competitive with state-of-the-art
model-based methods. In this paper, we propose a novel model-free algorithm,
Aggressive Q-Learning with Ensembles (AQE), which improves the
sample-efficiency performance of REDQ and the asymptotic performance of TQC,
thereby providing overall state-of-the-art performance during all stages of
training. Moreover, AQE is very simple, requiring neither distributional
representation of critics nor target randomization.

    

### [[2111.09162] It's About Time: Analog Clock Reading in the Wild](http://arxiv.org/abs/2111.09162)


  In this paper, we present a framework for reading analog clocks in natural
images or videos. Specifically, we make the following contributions: First, we
create a scalable pipeline for generating synthetic clocks, significantly
reducing the requirements for the labour-intensive annotations; Second, we
introduce a clock recognition architecture based on spatial transformer
networks (STN), which is trained end-to-end for clock alignment and
recognition. We show that the model trained on the proposed synthetic dataset
generalises towards real clocks with good accuracy, advocating a Sim2Real
training regime; Third, to further reduce the gap between simulation and real
data, we leverage the special property of time, i.e. uniformity, to generate
reliable pseudo-labels on real unlabelled clock videos, and show that training
on these videos offers further improvements while still requiring zero manual
annotations. Lastly, we introduce three benchmark datasets based on COCO, Open
Images, and The Clock movie, totalling 4,472 images with clocks, with full
annotations for time, accurate to the minute.

    

### [[2111.09189] ToM2C: Target-oriented Multi-agent Communication and Cooperation with Theory of Mind](http://arxiv.org/abs/2111.09189)


  Being able to predict the mental states of others is a key factor to
effective social interaction. It is also crucial for distributed multi-agent
systems, where agents are required to communicate and cooperate. In this paper,
we introduce such an important social-cognitive skill, i.e. Theory of Mind
(ToM), to build socially intelligent agents who are able to communicate and
cooperate effectively to accomplish challenging tasks. With ToM, each agent is
capable of inferring the mental states and intentions of others according to
its (local) observation. Based on the inferred states, the agents decide "when"
and with "whom" to share their intentions. With the information observed,
inferred, and received, the agents decide their sub-goals and reach a consensus
among the team. In the end, the low-level executors independently take
primitive actions to accomplish the sub-goals. We demonstrate the idea in two
typical target-oriented multi-agent tasks: cooperative navigation and
multi-sensor target coverage. The experiments show that the proposed model not
only outperforms the state-of-the-art methods on reward and communication
efficiency, but also shows good generalization across different scales of the
environment.

    

### [[2111.09190] Understanding and Testing Generalization of Deep Networks on Out-of-Distribution Data](http://arxiv.org/abs/2111.09190)


  Deep network models perform excellently on In-Distribution (ID) data, but can
significantly fail on Out-Of-Distribution (OOD) data. While developing methods
focus on improving OOD generalization, few attention has been paid to
evaluating the capability of models to handle OOD data. This study is devoted
to analyzing the problem of experimental ID test and designing OOD test
paradigm to accurately evaluate the practical performance. Our analysis is
based on an introduced categorization of three types of distribution shifts to
generate OOD data. Main observations include: (1) ID test fails in neither
reflecting the actual performance of a single model nor comparing between
different models under OOD data. (2) The ID test failure can be ascribed to the
learned marginal and conditional spurious correlations resulted from the
corresponding distribution shifts. Based on this, we propose novel OOD test
paradigms to evaluate the generalization capacity of models to unseen data, and
discuss how to use OOD test results to find bugs of models to guide model
debugging.

    

### [[2111.09191] Preference Communication in Multi-Objective Normal-Form Games](http://arxiv.org/abs/2111.09191)


  We study the problem of multiple agents learning concurrently in a
multi-objective environment. Specifically, we consider two agents that
repeatedly play a multi-objective normal-form game. In such games, the payoffs
resulting from joint actions are vector valued. Taking a utility-based
approach, we assume a utility function exists that maps vectors to scalar
utilities and consider agents that aim to maximise the utility of expected
payoff vectors. As agents do not necessarily know their opponent's utility
function or strategy, they must learn optimal policies to interact with each
other. To aid agents in arriving at adequate solutions, we introduce four novel
preference communication protocols for both cooperative as well as
self-interested communication. Each approach describes a specific protocol for
one agent communicating preferences over their actions and how another agent
responds. These protocols are subsequently evaluated on a set of five benchmark
games against baseline agents that do not communicate. We find that preference
communication can drastically alter the learning process and lead to the
emergence of cyclic Nash equilibria which had not been previously observed in
this setting. Additionally, we introduce a communication scheme where agents
must learn when to communicate. For agents in games with Nash equilibria, we
find that communication can be beneficial but difficult to learn when agents
have different preferred equilibria. When this is not the case, agents become
indifferent to communication. In games without Nash equilibria, our results
show differences across learning rates. When using faster learners, we observe
that explicit communication becomes more prevalent at around 50% of the time,
as it helps them in learning a compromise joint policy. Slower learners retain
this pattern to a lesser degree, but show increased indifference.

    

### [[2111.09194] IV-GNN : Interval Valued Data Handling Using Graph Neural Network](http://arxiv.org/abs/2111.09194)


  Graph Neural Network (GNN) is a powerful tool to perform standard machine
learning on graphs. To have a Euclidean representation of every node in the
Non-Euclidean graph-like data, GNN follows neighbourhood aggregation and
combination of information recursively along the edges of the graph. Despite
having many GNN variants in the literature, no model can deal with graphs
having nodes with interval-valued features. This article proposes an
Interval-ValuedGraph Neural Network, a novel GNN model where, for the first
time, we relax the restriction of the feature space being countable. Our model
is much more general than existing models as any countable set is always a
subset of the universal set $R^{n}$, which is uncountable. Here, to deal with
interval-valued feature vectors, we propose a new aggregation scheme of
intervals and show its expressive power to capture different interval
structures. We validate our theoretical findings about our model for graph
classification tasks by comparing its performance with those of the
state-of-the-art models on several benchmark network and synthetic datasets.

    

### [[2111.09212] Single-pass Object-adaptive Data Undersampling and Reconstruction for MRI](http://arxiv.org/abs/2111.09212)


  There is much recent interest in techniques to accelerate the data
acquisition process in MRI by acquiring limited measurements. Often
sophisticated reconstruction algorithms are deployed to maintain high image
quality in such settings. In this work, we propose a data-driven sampler using
a convolutional neural network, MNet, to provide object-specific sampling
patterns adaptive to each scanned object. The network observes very limited
low-frequency k-space data for each object and rapidly predicts the desired
undersampling pattern in one go that achieves high image reconstruction
quality.
We propose an accompanying alternating-type training framework with a
mask-backward procedure that efficiently generates training labels for the
sampler network and jointly trains an image reconstruction network.
Experimental results on the fastMRI knee dataset demonstrate the ability of the
proposed learned undersampling network to generate object-specific masks at
fourfold and eightfold acceleration that achieve superior image reconstruction
performance than several existing schemes. The source code for the proposed
joint sampling and reconstruction learning framework is available at
this https URL.

    

### [[2111.09248] Secure Federated Learning for Residential Short Term Load Forecasting](http://arxiv.org/abs/2111.09248)


  The inclusion of intermittent and renewable energy sources has increased the
importance of demand forecasting in power systems. Smart meters can play a
critical role in demand forecasting due to the measurement granularity they
provide. Consumers' privacy concerns, reluctance of utilities and vendors to
share data with competitors or third parties, and regulatory constraints are
some constraints smart meter forecasting faces. This paper examines a
collaborative machine learning method for short-term demand forecasting using
smart meter data as a solution to the previous constraints. Privacy preserving
techniques and federated learning enable to ensure consumers' confidentiality
concerning both, their data, the models generated using it (Differential
Privacy), and the communication mean (Secure Aggregation). The methods
evaluated take into account several scenarios that explore how traditional
centralized approaches could be projected in the direction of a decentralized,
collaborative and private system. The results obtained over the evaluations
provided almost perfect privacy budgets (1.39,$10e^{-5}$) and (2.01,$10e^{-5}$)
with a negligible performance compromise.

    

### [[2111.09254] Universal Inference Meets Random Projections: A Scalable Test for Log-concavity](http://arxiv.org/abs/2111.09254)


  Shape constraints yield flexible middle grounds between fully nonparametric
and fully parametric approaches to modeling distributions of data. The specific
assumption of log-concavity is motivated by applications across economics,
survival modeling, and reliability theory. However, there do not currently
exist valid tests for whether the underlying density of given data is
log-concave. The recent universal likelihood ratio test provides a valid test.
The universal test relies on maximum likelihood estimation (MLE), and efficient
methods already exist for finding the log-concave MLE. This yields the first
test of log-concavity that is provably valid in finite samples in any
dimension, for which we also establish asymptotic consistency results.
Empirically, we find that the highest power is obtained by using random
projections to convert the d-dimensional testing problem into many
one-dimensional problems, leading to a simple procedure that is statistically
and computationally efficient.

    

### [[2111.09266] GFlowNet Foundations](http://arxiv.org/abs/2111.09266)


  Generative Flow Networks (GFlowNets) have been introduced as a method to
sample a diverse set of candidates in an active learning context, with a
training objective that makes them approximately sample in proportion to a
given reward function. In this paper, we show a number of additional
theoretical properties of GFlowNets. They can be used to estimate joint
probability distributions and the corresponding marginal distributions where
some variables are unspecified and, of particular interest, can represent
distributions over composite objects like sets and graphs. GFlowNets amortize
the work typically done by computationally expensive MCMC methods in a single
but trained generative pass. They could also be used to estimate partition
functions and free energies, conditional probabilities of supersets
(supergraphs) given a subset (subgraph), as well as marginal distributions over
all supersets (supergraphs) of a given set (graph). We introduce variations
enabling the estimation of entropy and mutual information, sampling from a
Pareto frontier, connections to reward-maximizing policies, and extensions to
stochastic environments, continuous actions and modular energy functions.

    

### [[2111.09277] SmoothMix: Training Confidence-calibrated Smoothed Classifiers for Certified Robustness](http://arxiv.org/abs/2111.09277)


  Randomized smoothing is currently a state-of-the-art method to construct a
certifiably robust classifier from neural networks against $\ell_2$-adversarial
perturbations. Under the paradigm, the robustness of a classifier is aligned
with the prediction confidence, i.e., the higher confidence from a smoothed
classifier implies the better robustness. This motivates us to rethink the
fundamental trade-off between accuracy and robustness in terms of calibrating
confidences of a smoothed classifier. In this paper, we propose a simple
training scheme, coined SmoothMix, to control the robustness of smoothed
classifiers via self-mixup: it trains on convex combinations of samples along
the direction of adversarial perturbation for each input. The proposed
procedure effectively identifies over-confident, near off-class samples as a
cause of limited robustness in case of smoothed classifiers, and offers an
intuitive way to adaptively set a new decision boundary between these samples
for better robustness. Our experimental results demonstrate that the proposed
method can significantly improve the certified $\ell_2$-robustness of smoothed
classifiers compared to existing state-of-the-art robust training methods.

    

### [[2111.09278] Differentially Private Federated Learning on Heterogeneous Data](http://arxiv.org/abs/2111.09278)


  Federated Learning (FL) is a paradigm for large-scale distributed learning
which faces two key challenges: (i) efficient training from highly
heterogeneous user data, and (ii) protecting the privacy of participating
users. In this work, we propose a novel FL approach (DP-SCAFFOLD) to tackle
these two challenges together by incorporating Differential Privacy (DP)
constraints into the popular SCAFFOLD algorithm. We focus on the challenging
setting where users communicate with a ''honest-but-curious'' server without
any trusted intermediary, which requires to ensure privacy not only towards a
third-party with access to the final model but also towards the server who
observes all user communications. Using advanced results from DP theory, we
establish the convergence of our algorithm for convex and non-convex
objectives. Our analysis clearly highlights the privacy-utility trade-off under
data heterogeneity, and demonstrates the superiority of DP-SCAFFOLD over the
state-of-the-art algorithm DP-FedAvg when the number of local updates and the
level of heterogeneity grow. Our numerical results confirm our analysis and
show that DP-SCAFFOLD provides significant gains in practice.

    

### [[2111.09293] Fast BATLLNN: Fast Box Analysis of Two-Level Lattice Neural Networks](http://arxiv.org/abs/2111.09293)


  In this paper, we present the tool Fast Box Analysis of Two-Level Lattice
Neural Networks (Fast BATLLNN) as a fast verifier of box-like output
constraints for Two-Level Lattice (TLL) Neural Networks (NNs). In particular,
Fast BATLLNN can verify whether the output of a given TLL NN always lies within
a specified hyper-rectangle whenever its input constrained to a specified
convex polytope (not necessarily a hyper-rectangle). Fast BATLLNN uses the
unique semantics of the TLL architecture and the decoupled nature of box-like
output constraints to dramatically improve verification performance relative to
known polynomial-time verification algorithms for TLLs with generic polytopic
output constraints. In this paper, we evaluate the performance and scalability
of Fast BATLLNN, both in its own right and compared to state-of-the-art NN
verifiers applied to TLL NNs. Fast BATLLNN compares very favorably to even the
fastest NN verifiers, completing our synthetic TLL test bench more than 400x
faster than its nearest competitor.

    

### [[2111.09297] Learning to Compose Visual Relations](http://arxiv.org/abs/2111.09297)


  The visual world around us can be described as a structured set of objects
and their associated relations. An image of a room may be conjured given only
the description of the underlying objects and their associated relations. While
there has been significant work on designing deep neural networks which may
compose individual objects together, less work has been done on composing the
individual relations between objects. A principal difficulty is that while the
placement of objects is mutually independent, their relations are entangled and
dependent on each other. To circumvent this issue, existing works primarily
compose relations by utilizing a holistic encoder, in the form of text or
graphs. In this work, we instead propose to represent each relation as an
unnormalized density (an energy-based model), enabling us to compose separate
relations in a factorized manner. We show that such a factorized decomposition
allows the model to both generate and edit scenes that have multiple sets of
relations more faithfully. We further show that decomposition enables our model
to effectively understand the underlying relational scene structure. Project
page at: this https URL.

    

### [[2111.09304] Quantum-Assisted Support Vector Regression for Detecting Facial Landmarks](http://arxiv.org/abs/2111.09304)


  The classical machine-learning model for support vector regression (SVR) is
widely used for regression tasks, including weather prediction, stock-market
and real-estate pricing. However, a practically realisable quantum version for
SVR remains to be formulated. We devise annealing-based algorithms, namely
simulated and quantum-classical hybrid, for training two SVR models, and
compare their empirical performances against the SVR implementation of Python's
scikit-learn package and the SVR-based state-of-the-art algorithm for the
facial landmark detection (FLD) problem. Our method is to derive a
quadratic-unconstrained-binary formulation for the optimisation problem used
for training a SVR model and solve this problem using annealing. Using D-Wave's
Hybrid Solver, we construct a quantum-assisted SVR model, thereby demonstrating
a slight advantage over classical models regarding landmark-detection accuracy.
Furthermore, we observe that annealing-based SVR models predict landmarks with
lower variances compared to the SVR models trained by greedy optimisation
procedures. Our work is a proof-of-concept example for applying quantu-assisted
SVR to a supervised learning task with a small training dataset.

    

### [[1704.00454] Clustering in Hilbert simplex geometry](http://arxiv.org/abs/1704.00454)


  Clustering categorical distributions in the finite-dimensional probability
simplex is a fundamental task met in many applications dealing with normalized
histograms. Traditionally, the differential-geometric structures of the
probability simplex have been used either by (i) setting the Riemannian metric
tensor to the Fisher information matrix of the categorical distributions, or
(ii) defining the dualistic information-geometric structure induced by a smooth
dissimilarity measure, the Kullback-Leibler divergence. In this work, we
introduce for clustering tasks a novel computationally-friendly framework for
modeling geometrically the probability simplex: The {\em Hilbert simplex
geometry}. In the Hilbert simplex geometry, the distance is the non-separable
Hilbert's metric distance which satisfies the property of information
monotonicity with distance level set functions described by polytope
boundaries. We show that both the Aitchison and Hilbert simplex distances are
norm distances on normalized logarithmic representations with respect to the
$\ell_2$ and variation norms, respectively. We discuss the pros and cons of
those different statistical modelings, and benchmark experimentally these
different kind of geometries for center-based $k$-means and $k$-center
clustering. Furthermore, since a canonical Hilbert distance can be defined on
any bounded convex subset of the Euclidean space, we also consider Hilbert's
geometry of the elliptope of correlation matrices and study its clustering
performances compared to Fröbenius and log-det divergences.

    

### [[1902.00600] Efficient Learning of Discrete Graphical Models](http://arxiv.org/abs/1902.00600)


  Graphical models are useful tools for describing structured high-dimensional
probability distributions. Development of efficient algorithms for learning
graphical models with least amount of data remains an active research topic.
Reconstruction of graphical models that describe the statistics of discrete
variables is a particularly challenging problem, for which the maximum
likelihood approach is intractable. In this work, we provide the first
sample-efficient method based on the Interaction Screening framework that
allows one to provably learn fully general discrete factor models with
node-specific discrete alphabets and multi-body interactions, specified in an
arbitrary basis. We identify a single condition related to model
parametrization that leads to rigorous guarantees on the recovery of model
structure and parameters in any error norm, and is readily verifiable for a
large class of models. Importantly, our bounds make explicit distinction
between parameters that are proper to the model and priors used as an input to
the algorithm. Finally, we show that the Interaction Screening framework
includes all models previously considered in the literature as special cases,
and for which our analysis shows a systematic improvement in sample complexity.

    

### [[1910.08693] Online Pricing with Offline Data: Phase Transition and Inverse Square Law](http://arxiv.org/abs/1910.08693)


  This paper investigates the impact of pre-existing offline data on online
learning, in the context of dynamic pricing. We study a single-product dynamic
pricing problem over a selling horizon of $T$ periods. The demand in each
period is determined by the price of the product according to a linear demand
model with unknown parameters. We assume that before the start of the selling
horizon, the seller already has some pre-existing offline data. The offline
data set contains $n$ samples, each of which is an input-output pair consisting
of a historical price and an associated demand observation. The seller wants to
utilize both the pre-existing offline data and the sequential online data to
minimize the regret of the online learning process.
We characterize the joint effect of the size, location and dispersion of the
offline data on the optimal regret of the online learning process.
Specifically, the size, location and dispersion of the offline data are
measured by the number of historical samples $n$, the distance between the
average historical price and the optimal price $\delta$, and the standard
deviation of the historical prices $\sigma$, respectively. We show that the
optimal regret is $\widetilde \Theta\left(\sqrt{T}\wedge \frac{T}{(n\wedge
T)\delta^2+n\sigma^2}\right)$, and design a learning algorithm based on the
"optimism in the face of uncertainty" principle, whose regret is optimal up to
a logarithmic factor. Our results reveal surprising transformations of the
optimal regret rate with respect to the size of the offline data, which we
refer to as phase transitions. In addition, our results demonstrate that the
location and dispersion of the offline data also have an intrinsic effect on
the optimal regret, and we quantify this effect via the inverse-square law.

    

### [[1911.03810] Parameter Estimation in Adaptive Control of Time-Varying Systems Under a Range of Excitation Conditions](http://arxiv.org/abs/1911.03810)


  This paper presents a new parameter estimation algorithm for the adaptive
control of a class of time-varying plants. The main feature of this algorithm
is a matrix of time-varying learning rates, which enables parameter estimation
error trajectories to tend exponentially fast towards a compact set whenever
excitation conditions are satisfied. This algorithm is employed in a large
class of problems where unknown parameters are present and are time-varying. It
is shown that this algorithm guarantees global boundedness of the state and
parameter errors of the system, and avoids an often used filtering approach for
constructing key regressor signals. In addition, intervals of time over which
these errors tend exponentially fast toward a compact set are provided, both in
the presence of finite and persistent excitation. A projection operator is used
to ensure the boundedness of the learning rate matrix, as compared to a
time-varying forgetting factor. Numerical simulations are provided to
complement the theoretical analysis.

    

### [[1911.07605] Commit2Vec: Learning Distributed Representations of Code Changes](http://arxiv.org/abs/1911.07605)


  Deep learning methods, which have found successful applications in fields
like image classification and natural language processing, have recently been
applied to source code analysis too, due to the enormous amount of freely
available source code (e.g., from open-source software repositories).
In this work, we elaborate upon a state-of-the-art approach to the
representation of source code that uses information about its syntactic
structure, and we adapt it to represent source changes (i.e., commits). We use
this representation to classify security-relevant commits.
Because our method uses transfer learning (that is, we train a network on a
"pretext task" for which abundant labeled data is available, and then we use
such network for the target task of commit classification, for which fewer
labeled instances are available), we studied the impact of pre-training the
network using two different pretext tasks versus a randomly initialized model.
Our results indicate that representations that leverage the structural
information obtained through code syntax outperform token-based
representations. Furthermore, the performance metrics obtained when
pre-training on a loosely related pretext task with a very large dataset
($>10^6$ samples) were surpassed when pretraining on a smaller dataset ($>10^4$
samples) but for a pretext task that is more closely related to the target
task.

    

### [[2002.05135] On the Convergence Theory of Debiased Model-Agnostic Meta-Reinforcement Learning](http://arxiv.org/abs/2002.05135)


  We consider Model-Agnostic Meta-Learning (MAML) methods for Reinforcement
Learning (RL) problems, where the goal is to find a policy using data from
several tasks represented by Markov Decision Processes (MDPs) that can be
updated by one step of stochastic policy gradient for the realized MDP. In
particular, using stochastic gradients in MAML update steps is crucial for RL
problems since computation of exact gradients requires access to a large number
of possible trajectories. For this formulation, we propose a variant of the
MAML method, named Stochastic Gradient Meta-Reinforcement Learning (SG-MRL),
and study its convergence properties. We derive the iteration and sample
complexity of SG-MRL to find an $\epsilon$-first-order stationary point, which,
to the best of our knowledge, provides the first convergence guarantee for
model-agnostic meta-reinforcement learning algorithms. We further show how our
results extend to the case where more than one step of stochastic policy
gradient method is used at test time. Finally, we empirically compare SG-MRL
and MAML in several deep RL environments.

    

### [[2002.05474] Metric-Free Individual Fairness in Online Learning](http://arxiv.org/abs/2002.05474)


  We study an online learning problem subject to the constraint of individual
fairness, which requires that similar individuals are treated similarly. Unlike
prior work on individual fairness, we do not assume the similarity measure
among individuals is known, nor do we assume that such measure takes a certain
parametric form. Instead, we leverage the existence of an auditor who detects
fairness violations without enunciating the quantitative measure. In each
round, the auditor examines the learner's decisions and attempts to identify a
pair of individuals that are treated unfairly by the learner. We provide a
general reduction framework that reduces online classification in our model to
standard online classification, which allows us to leverage existing online
learning algorithms to achieve sub-linear regret and number of fairness
violations. Surprisingly, in the stochastic setting where the data are drawn
independently from a distribution, we are also able to establish PAC-style
fairness and accuracy generalization guarantees (Yona and Rothblum [2018]),
despite only having access to a very restricted form of fairness feedback. Our
fairness generalization bound qualitatively matches the uniform convergence
bound of Yona and Rothblum [2018], while also providing a meaningful accuracy
generalization guarantee. Our results resolve an open question by Gillen et al.
[2018] by showing that online learning under an unknown individual fairness
constraint is possible even without assuming a strong parametric form of the
underlying similarity measure.

    

### [[2003.04035] Transformation-based Adversarial Video Prediction on Large-Scale Data](http://arxiv.org/abs/2003.04035)


  Recent breakthroughs in adversarial generative modeling have led to models
capable of producing video samples of high quality, even on large and complex
datasets of real-world video. In this work, we focus on the task of video
prediction, where given a sequence of frames extracted from a video, the goal
is to generate a plausible future sequence. We first improve the state of the
art by performing a systematic empirical study of discriminator decompositions
and proposing an architecture that yields faster convergence and higher
performance than previous approaches. We then analyze recurrent units in the
generator, and propose a novel recurrent unit which transforms its past hidden
state according to predicted motion-like features, and refines it to handle
dis-occlusions, scene changes and other complex behavior. We show that this
recurrent unit consistently outperforms previous designs. Our final model leads
to a leap in the state-of-the-art performance, obtaining a test set Frechet
Video Distance of 25.7, down from 69.2, on the large-scale Kinetics-600
dataset.

    

### [[2008.01158] Classification of Multiple Diseases on Body CT Scans using Weakly Supervised Deep Learning](http://arxiv.org/abs/2008.01158)


  Purpose: To design multi-disease classifiers for body CT scans for three
different organ systems using automatically extracted labels from radiology
text reports.Materials & Methods: This retrospective study included a total of
12,092 patients (mean age 57 +- 18; 6,172 women) for model development and
testing (from 2012-2017). Rule-based algorithms were used to extract 19,225
disease labels from 13,667 body CT scans from 12,092 patients. Using a
three-dimensional DenseVNet, three organ systems were segmented: lungs and
pleura; liver and gallbladder; and kidneys and ureters. For each organ, a
three-dimensional convolutional neural network classified no apparent disease
versus four common diseases for a total of 15 different labels across all three
models. Testing was performed on a subset of 2,158 CT volumes relative to 2,875
manually derived reference labels from 2133 patients (mean age 58 +- 18;1079
women). Performance was reported as receiver operating characteristic area
under the curve (AUC) with 95% confidence intervals by the DeLong method.
Results: Manual validation of the extracted labels confirmed 91% to 99%
accuracy across the 15 different labels. AUCs for lungs and pleura labels were:
atelectasis 0.77 (95% CI: 0.74, 0.81), nodule 0.65 (0.61, 0.69), emphysema 0.89
(0.86, 0.92), effusion 0.97 (0.96, 0.98), and no apparent disease 0.89 (0.87,
0.91). AUCs for liver and gallbladder were: hepatobiliary calcification 0.62
(95% CI: 0.56, 0.67), lesion 0.73 (0.69, 0.77), dilation 0.87 (0.84, 0.90),
fatty 0.89 (0.86, 0.92), and no apparent disease 0.82 (0.78, 0.85). AUCs for
kidneys and ureters were: stone 0.83 (95% CI: 0.79, 0.87), atrophy 0.92 (0.89,
0.94), lesion 0.68 (0.64, 0.72), cyst 0.70 (0.66, 0.73), and no apparent
disease 0.79 (0.75, 0.83). Conclusion: Weakly-supervised deep learning models
were able to classify diverse diseases in multiple organ systems.

    

### [[2010.03324] Automated Human Activity Recognition by Colliding Bodies Optimization-based Optimal Feature Selection with Recurrent Neural Network](http://arxiv.org/abs/2010.03324)


  In smart healthcare, Human Activity Recognition (HAR) is considered to be an
efficient model in pervasive computation from sensor readings. The Ambient
Assisted Living (AAL) in the home or community helps the people in providing
independent care and enhanced living quality. However, many AAL models were
restricted using many factors that include computational cost and system
complexity. Moreover, the HAR concept has more relevance because of its
applications. Hence, this paper tempts to implement the HAR system using deep
learning with the data collected from smart sensors that are publicly available
in the UC Irvine Machine Learning Repository (UCI). The proposed model involves
three processes: (1) Data collection, (b) Optimal feature selection, (c)
Recognition. The data gathered from the benchmark repository is initially
subjected to optimal feature selection that helps to select the most
significant features. The proposed optimal feature selection is based on a new
meta-heuristic algorithm called Colliding Bodies Optimization (CBO). An
objective function derived by the recognition accuracy is used for
accomplishing the optimal feature selection. Here, the deep learning model
called Recurrent Neural Network (RNN) is used for activity recognition. The
proposed model on the concerned benchmark dataset outperforms existing learning
methods, providing high performance compared to the conventional models.

    

### [[2011.08184] Deep Learning -- A first Meta-Survey of selected Reviews across Scientific Disciplines, their Commonalities, Challenges and Research Impact](http://arxiv.org/abs/2011.08184)


  Deep learning belongs to the field of artificial intelligence, where machines
perform tasks that typically require some kind of human intelligence. Similar
to the basic structure of a brain, a deep learning algorithm consists of an
artificial neural network, which resembles the biological brain structure.
Mimicking the learning process of humans with their senses, deep learning
networks are fed with (sensory) data, like texts, images, videos or sounds.
These networks outperform the state-of-the-art methods in different tasks and,
because of this, the whole field saw an exponential growth during the last
years. This growth resulted in way over 10,000 publications per year in the
last years. For example, the search engine PubMed alone, which covers only a
sub-set of all publications in the medical field, provides already over 11,000
results in Q3 2020 for the search term 'deep learning', and around 90% of these
results are from the last three years. Consequently, a complete overview over
the field of deep learning is already impossible to obtain and, in the near
future, it will potentially become difficult to obtain an overview over a
subfield. However, there are several review articles about deep learning, which
are focused on specific scientific fields or applications, for example deep
learning advances in computer vision or in specific tasks like object
detection. With these surveys as a foundation, the aim of this contribution is
to provide a first high-level, categorized meta-survey of selected reviews on
deep learning across different scientific disciplines. The categories (computer
vision, language processing, medical informatics and additional works) have
been chosen according to the underlying data sources (image, language, medical,
mixed). In addition, we review the common architectures, methods, pros, cons,
evaluations, challenges and future directions for every sub-category.

    

### [[2011.09573] Learning Recurrent Neural Net Models of Nonlinear Systems](http://arxiv.org/abs/2011.09573)


  We consider the following learning problem: Given sample pairs of input and
output signals generated by an unknown nonlinear system (which is not assumed
to be causal or time-invariant), we wish to find a continuous-time recurrent
neural net with hyperbolic tangent activation function that approximately
reproduces the underlying i/o behavior with high confidence. Leveraging earlier
work concerned with matching output derivatives up to a given finite order, we
reformulate the learning problem in familiar system-theoretic language and
derive quantitative guarantees on the sup-norm risk of the learned model in
terms of the number of neurons, the sample size, the number of derivatives
being matched, and the regularity properties of the inputs, the outputs, and
the unknown i/o map.

    

### [[2011.11197] The Emerging Trends of Multi-Label Learning](http://arxiv.org/abs/2011.11197)


  Exabytes of data are generated daily by humans, leading to the growing need
for new efforts in dealing with the grand challenges for multi-label learning
brought by big data. For example, extreme multi-label classification is an
active and rapidly growing research area that deals with classification tasks
with an extremely large number of classes or labels; utilizing massive data
with limited supervision to build a multi-label classification model becomes
valuable for practical applications, etc. Besides these, there are tremendous
efforts on how to harvest the strong learning capability of deep learning to
better capture the label dependencies in multi-label learning, which is the key
for deep learning to address real-world classification tasks. However, it is
noted that there has been a lack of systemic studies that focus explicitly on
analyzing the emerging trends and new challenges of multi-label learning in the
era of big data. It is imperative to call for a comprehensive survey to fulfill
this mission and delineate future research directions and new applications.

    

### [[2012.09613] Model-based Reinforcement Learning for Continuous Control with Posterior Sampling](http://arxiv.org/abs/2012.09613)


  Balancing exploration and exploitation is crucial in reinforcement learning
(RL). In this paper, we study model-based posterior sampling for reinforcement
learning (PSRL) in continuous state-action spaces theoretically and
empirically. First, we show the first regret bound of PSRL in continuous spaces
which is polynomial in the episode length to the best of our knowledge. With
the assumption that reward and transition functions can be modeled by Bayesian
linear regression, we develop a regret bound of $\tilde{O}(H^{3/2}d\sqrt{T})$,
where $H$ is the episode length, $d$ is the dimension of the state-action
space, and $T$ indicates the total time steps. This result matches the
best-known regret bound of non-PSRL methods in linear MDPs. Our bound can be
extended to nonlinear cases as well with feature embedding: using linear
kernels on the feature representation $\phi$, the regret bound becomes
$\tilde{O}(H^{3/2}d_{\phi}\sqrt{T})$, where $d_\phi$ is the dimension of the
representation space. Moreover, we present MPC-PSRL, a model-based posterior
sampling algorithm with model predictive control for action selection. To
capture the uncertainty in models, we use Bayesian linear regression on the
penultimate layer (the feature representation layer $\phi$) of neural networks.
Empirical results show that our algorithm achieves the state-of-the-art sample
efficiency in benchmark continuous control tasks compared to prior model-based
algorithms, and matches the asymptotic performance of model-free algorithms.

    

### [[2102.00313] Cortical Features for Defense Against Adversarial Audio Attacks](http://arxiv.org/abs/2102.00313)


  We propose using a computational model of the auditory cortex as a defense
against adversarial attacks on audio. We apply several white-box iterative
optimization-based adversarial attacks to an implementation of Amazon Alexa's
HW network, and a modified version of this network with an integrated cortical
representation, and show that the cortical features help defend against
universal adversarial examples. At the same level of distortion, the
adversarial noises found for the cortical network are always less effective for
universal audio attacks. We make our code publicly available at
this https URL.

    

### [[2102.01243] PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation](http://arxiv.org/abs/2102.01243)


  Audio tagging is an active research area and has a wide range of
applications. Since the release of AudioSet, great progress has been made in
advancing model performance, which mostly comes from the development of novel
model architectures and attention modules. However, we find that appropriate
training techniques are equally important for building audio tagging models
with AudioSet, but have not received the attention they deserve. To fill the
gap, in this work, we present PSLA, a collection of training techniques that
can noticeably boost the model accuracy including ImageNet pretraining,
balanced sampling, data augmentation, label enhancement, model aggregation and
their design choices. By training an EfficientNet with these techniques, we
obtain a single model (with 13.6M parameters) and an ensemble model that
achieve mean average precision (mAP) scores of 0.444 and 0.474 on AudioSet,
respectively, outperforming the previous best system of 0.439 with 81M
parameters. In addition, our model also achieves a new state-of-the-art mAP of
0.567 on FSD50K.

    

### [[2102.03832] Generalization of Model-Agnostic Meta-Learning Algorithms: Recurring and Unseen Tasks](http://arxiv.org/abs/2102.03832)


  In this paper, we study the generalization properties of Model-Agnostic
Meta-Learning (MAML) algorithms for supervised learning problems. We focus on
the setting in which we train the MAML model over $m$ tasks, each with $n$ data
points, and characterize its generalization error from two points of view:
First, we assume the new task at test time is one of the training tasks, and we
show that, for strongly convex objective functions, the expected excess
population loss is bounded by ${\mathcal{O}}(1/mn)$. Second, we consider the
MAML algorithm's generalization to an unseen task and show that the resulting
generalization error depends on the total variation distance between the
underlying distributions of the new task and the tasks observed during the
training process. Our proof techniques rely on the connections between
algorithmic stability and generalization bounds of algorithms. In particular,
we propose a new definition of stability for meta-learning algorithms, which
allows us to capture the role of both the number of tasks $m$ and number of
samples per task $n$ on the generalization error of MAML.

    

### [[2102.13347] MDA for random forests: inconsistency, and a practical solution via the Sobol-MDA](http://arxiv.org/abs/2102.13347)


  Variable importance measures are the main tools to analyze the black-box
mechanisms of random forests. Although the mean decrease accuracy (MDA) is
widely accepted as the most efficient variable importance measure for random
forests, little is known about its statistical properties. In fact, the exact
MDA definition varies across the main random forest software. In this article,
our objective is to rigorously analyze the behavior of the main MDA
implementations. Consequently, we mathematically formalize the various
implemented MDA algorithms, and then establish their limits when the sample
size increases. In particular, we break down these limits in three components:
the first one is related to Sobol indices, which are well-defined measures of a
covariate contribution to the response variance, widely used in the sensitivity
analysis field, as opposed to thethird term, whose value increases with
dependence within covariates. Thus, we theoretically demonstrate that the MDA
does not target the right quantity when covariates are dependent, a fact that
has already been noticed experimentally. To address this issue, we define a new
importance measure for random forests, the Sobol-MDA, which fixes the flaws of
the original MDA. We prove the consistency of the Sobol-MDA and show thatthe
Sobol-MDA empirically outperforms its competitors on both simulated and real
data. An open source implementation in R and C++ is available online.

    

### [[2103.11003] Differentially private inference via noisy optimization](http://arxiv.org/abs/2103.11003)


  We propose a general optimization-based framework for computing
differentially private M-estimators and a new method for constructing
differentially private confidence regions. Firstly, we show that robust
statistics can be used in conjunction with noisy gradient descent or noisy
Newton methods in order to obtain optimal private estimators with global linear
or quadratic convergence, respectively. We establish local and global
convergence guarantees, under both local strong convexity and self-concordance,
showing that our private estimators converge with high probability to a nearly
optimal neighborhood of the non-private M-estimators. Secondly, we tackle the
problem of parametric inference by constructing differentially private
estimators of the asymptotic variance of our private M-estimators. This
naturally leads to approximate pivotal statistics for constructing confidence
regions and conducting hypothesis testing. We demonstrate the effectiveness of
a bias correction that leads to enhanced small-sample empirical performance in
simulations. We illustrate the benefits of our methods in several numerical
examples.

    

### [[2104.07917] Hop-Count Based Self-Supervised Anomaly Detection on Attributed Networks](http://arxiv.org/abs/2104.07917)


  Recent years have witnessed an upsurge of interest in the problem of anomaly
detection on attributed networks due to its importance in both research and
practice. Although various approaches have been proposed to solve this problem,
two major limitations exist: (1) unsupervised approaches usually work much less
efficiently due to the lack of supervisory signal, and (2) existing anomaly
detection methods only use local contextual information to detect anomalous
nodes, e.g., one- or two-hop information, but ignore the global contextual
information. Since anomalous nodes differ from normal nodes in structures and
attributes, it is intuitive that the distance between anomalous nodes and their
neighbors should be larger than that between normal nodes and their neighbors
if we remove the edges connecting anomalous and normal nodes. Thus, hop counts
based on both global and local contextual information can be served as the
indicators of anomaly. Motivated by this intuition, we propose a hop-count
based model (HCM) to detect anomalies by modeling both local and global
contextual information. To make better use of hop counts for anomaly
identification, we propose to use hop counts prediction as a self-supervised
task. We design two anomaly scores based on the hop counts prediction via HCM
model to identify anomalies. Besides, we employ Bayesian learning to train HCM
model for capturing uncertainty in learned parameters and avoiding overfitting.
Extensive experiments on real-world attributed networks demonstrate that our
proposed model is effective in anomaly detection.

    

### [[2104.10481] SSLM: Self-Supervised Learning for Medical Diagnosis from MR Video](http://arxiv.org/abs/2104.10481)


  In medical image analysis, the cost of acquiring high-quality data and their
annotation by experts is a barrier in many medical applications. Most of the
techniques used are based on supervised learning framework and need a large
amount of annotated data to achieve satisfactory performance. As an
alternative, in this paper, we propose a self-supervised learning approach to
learn the spatial anatomical representations from the frames of magnetic
resonance (MR) video clips for the diagnosis of knee medical conditions. The
pretext model learns meaningful spatial context-invariant representations. The
downstream task in our paper is a class imbalanced multi-label classification.
Different experiments show that the features learnt by the pretext model
provide explainable performance in the downstream task. Moreover, the
efficiency and reliability of the proposed pretext model in learning
representations of minority classes without applying any strategy towards
imbalance in the dataset can be seen from the results. To the best of our
knowledge, this work is the first work of its kind in showing the effectiveness
and reliability of self-supervised learning algorithms in class imbalanced
multi-label classification tasks on MR video.
The code for evaluation of the proposed work is available at
this https URL


### [[2104.14579] LIDAR and Position-Aided mmWave Beam Selection with Non-local CNNs and Curriculum Training](http://arxiv.org/abs/2104.14579)


  Efficient millimeter wave (mmWave) beam selection in
vehicle-to-infrastructure (V2I) communication is a crucial yet challenging task
due to the narrow mmWave beamwidth and high user mobility. To reduce the search
overhead of iterative beam discovery procedures, contextual information from
light detection and ranging (LIDAR) sensors mounted on vehicles has been
leveraged by data-driven methods to produce useful side information. In this
paper, we propose a lightweight neural network (NN) architecture along with the
corresponding LIDAR preprocessing, which significantly outperforms previous
works. Our solution comprises multiple novelties that improve both the
convergence speed and the final accuracy of the model. In particular, we define
a novel loss function inspired by the knowledge distillation idea, introduce a
curriculum training approach exploiting line-of-sight (LOS)/non-line-of-sight
(NLOS) information, and we propose a non-local attention module to improve the
performance for the more challenging NLOS cases. Simulation results on
benchmark datasets show that, utilizing solely LIDAR data and the receiver
position, our NN-based beam selection scheme can achieve 79.9% throughput of an
exhaustive beam sweeping approach without any beam search overhead and 95% by
searching among as few as 6 beams. In a typical mmWave V2I scenario, our
proposed method considerably reduces the beam search time required to achieve a
desired throughput, in comparison with the inverse fingerprinting and
hierarchical beam selection schemes.

    

### [[2105.07831] How to Explain Neural Networks: an Approximation Perspective](http://arxiv.org/abs/2105.07831)


  The lack of interpretability has hindered the large-scale adoption of AI
technologies. However, the fundamental idea of interpretability, as well as how
to put it into practice, remains unclear. We provide notions of
interpretability based on approximation theory in this study. We first
implement this approximation interpretation on a specific model (fully
connected neural network) and then propose to use MLP as a universal
interpreter to explain arbitrary black-box models. Extensive experiments
demonstrate the effectiveness of our approach.

    

### [[2105.13762] The Feature-First Block Model](http://arxiv.org/abs/2105.13762)


  Labelled networks are an important class of data, naturally appearing in
numerous applications in science and engineering. A typical inference goal is
to determine how the vertex labels (or features) affect the network's
structure. In this work, we introduce a new generative model, the feature-first
block model (FFBM), that facilitates the use of rich queries on labelled
networks. We develop a Bayesian framework and devise a two-level Markov chain
Monte Carlo approach to efficiently sample from the relevant posterior
distribution of the FFBM parameters. This allows us to infer if and how the
observed vertex-features affect macro-structure. We apply the proposed methods
to a variety of network data to extract the most important features along which
the vertices are partitioned. The main advantages of the proposed approach are
that the whole feature-space is used automatically and that features can be
rank-ordered implicitly according to impact.

    

### [[2106.03917] Mixture Outlier Exposure for Out-of-Distribution Detection in Fine-grained Settings](http://arxiv.org/abs/2106.03917)


  Enabling out-of-distribution (OOD) detection for DNNs is critical for their
safe and reliable operation in the open world. Despite recent progress, current
works often consider a coarse level of granularity in the OOD problem, which
fail to approximate many real-world fine-grained tasks where high granularity
may be expected between the in-distribution (ID) data and the OOD data (e.g.,
identifying novel bird species for a bird classification system in the wild).
In this work, we start by carefully constructing four large-scale fine-grained
test environments in which existing methods are shown to have difficulties. We
find that current methods, including ones that include a large/diverse set of
outliers during DNN training, have poor coverage over the broad region where
fine-grained OOD samples locate. We then propose Mixture Outlier Exposure
(MixOE), which effectively expands the covered OOD region by mixing ID data and
training outliers, and regularizes the model behaviour by linearly decaying the
prediction confidence as the input transitions from ID to OOD. Extensive
experiments and analyses demonstrate the effectiveness of MixOE for improving
OOD detection in fine-grained settings.

    

### [[2106.04463] PolypGen: A multi-center polyp detection and segmentation dataset for generalisability assessment](http://arxiv.org/abs/2106.04463)


  Polyps in the colon are widely known as cancer precursors identified by
colonoscopy either related to diagnostic work-up for symptoms, colorectal
cancer screening or systematic surveillance of certain diseases. Whilst most
polyps are benign, the number, size and the surface structure of the polyp are
tightly linked to the risk of colon cancer. There exists a high missed
detection rate and incomplete removal of colon polyps due to the variable
nature, difficulties to delineate the abnormality, high recurrence rates and
the anatomical topography of the colon. In the past, several methods have been
built to automate polyp detection and segmentation. However, the key issue of
most methods is that they have not been tested rigorously on a large
multi-center purpose-built dataset. Thus, these methods may not generalise to
different population datasets as they overfit to a specific population and
endoscopic surveillance. To this extent, we have curated a dataset from 6
different centers incorporating more than 300 patients. The dataset includes
both single frame and sequence data with 3446 annotated polyp labels with
precise delineation of polyp boundaries verified by six senior
gastroenterologists. To our knowledge, this is the most comprehensive detection
and pixel-level segmentation dataset curated by a team of computational
scientists and expert gastroenterologists. This dataset has been originated as
the part of the Endocv2021 challenge aimed at addressing generalisability in
polyp detection and segmentation. In this paper, we provide comprehensive
insight into data construction and annotation strategies, annotation quality
assurance and technical validation for our extended EndoCV2021 dataset which we
refer to as PolypGen.

    

### [[2106.07804] Controlling Neural Networks with Rule Representations](http://arxiv.org/abs/2106.07804)


  We propose a novel training method that integrates rules into deep learning,
in a way the strengths of the rules are controllable at inference. Deep Neural
Networks with Controllable Rule Representations (DeepCTRL) incorporates a rule
encoder into the model coupled with a rule-based objective, enabling a shared
representation for decision making. DeepCTRL is agnostic to data type and model
architecture. It can be applied to any kind of rule defined for inputs and
outputs. The key aspect of DeepCTRL is that it does not require retraining to
adapt the rule strength -- at inference, the user can adjust it based on the
desired operation point on accuracy vs. rule verification ratio. In real-world
domains where incorporating rules is critical -- such as Physics, Retail and
Healthcare -- we show the effectiveness of DeepCTRL in teaching rules for deep
learning. DeepCTRL improves the trust and reliability of the trained models by
significantly increasing their rule verification ratio, while also providing
accuracy gains at downstream tasks. Additionally, DeepCTRL enables novel use
cases such as hypothesis testing of the rules on data samples, and unsupervised
adaptation based on shared rules between datasets.

    

### [[2106.12929] Lettuce: PyTorch-based Lattice Boltzmann Framework](http://arxiv.org/abs/2106.12929)


  The lattice Boltzmann method (LBM) is an efficient simulation technique for
computational fluid mechanics and beyond. It is based on a simple
stream-and-collide algorithm on Cartesian grids, which is easily compatible
with modern machine learning architectures. While it is becoming increasingly
clear that deep learning can provide a decisive stimulus for classical
simulation techniques, recent studies have not addressed possible connections
between machine learning and LBM. Here, we introduce Lettuce, a PyTorch-based
LBM code with a threefold aim. Lettuce enables GPU accelerated calculations
with minimal source code, facilitates rapid prototyping of LBM models, and
enables integrating LBM simulations with PyTorch's deep learning and automatic
differentiation facility. As a proof of concept for combining machine learning
with the LBM, a neural collision model is developed, trained on a doubly
periodic shear layer and then transferred to a different flow, a decaying
turbulence. We also exemplify the added benefit of PyTorch's automatic
differentiation framework in flow control and optimization. To this end, the
spectrum of a forced isotropic turbulence is maintained without further
constraining the velocity field. The source code is freely available from
this https URL.

    

### [[2109.13202] MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research](http://arxiv.org/abs/2109.13202)


  Progress in deep reinforcement learning (RL) is heavily driven by the
availability of challenging benchmarks used for training agents. However,
benchmarks that are widely adopted by the community are not explicitly designed
for evaluating specific capabilities of RL methods. While there exist
environments for assessing particular open problems in RL (such as exploration,
transfer learning, unsupervised environment design, or even language-assisted
RL), it is generally difficult to extend these to richer, more complex
environments once research goes beyond proof-of-concept results. We present
MiniHack, a powerful sandbox framework for easily designing novel RL
environments. MiniHack is a one-stop shop for RL experiments with environments
ranging from small rooms to complex, procedurally generated worlds. By
leveraging the full set of entities and environment dynamics from NetHack, one
of the richest grid-based video games, MiniHack allows designing custom RL
testbeds that are fast and convenient to use. With this sandbox framework,
novel environments can be designed easily, either using a human-readable
description language or a simple Python interface. In addition to a variety of
RL tasks and baselines, MiniHack can wrap existing RL benchmarks and provide
ways to seamlessly add additional complexity.

    

### [[2110.01235] Identifiability in Two-Layer Sparse Matrix Factorization](http://arxiv.org/abs/2110.01235)


  Sparse matrix factorization is the problem of approximating a matrix
$\mathbf{Z}$ by a product of $J$ sparse factors $\mathbf{X}^{(J)}
\mathbf{X}^{(J-1)} \ldots \mathbf{X}^{(1)}$. This paper focuses on
identifiability issues that appear in this problem, in view of better
understanding under which sparsity constraints the problem is well-posed. We
give conditions under which the problem of factorizing a matrix into \emph{two}
sparse factors admits a unique solution, up to unavoidable permutation and
scaling equivalences. Our general framework considers an arbitrary family of
prescribed sparsity patterns, allowing us to capture more structured notions of
sparsity than simply the count of nonzero entries. These conditions are shown
to be related to essential uniqueness of exact matrix decomposition into a sum
of rank-one matrices, with structured sparsity constraints. In particular, in
the case of fixed-support sparse matrix factorization, we give a general
sufficient condition for identifiability based on rank-one matrix
completability, and we derive from it a completion algorithm that can verify if
this sufficient condition is satisfied, and recover the entries in the two
sparse factors if this is the case. A companion paper further exploits these
conditions to derive identifiability properties and theoretically sound
factorization methods for multi-layer sparse matrix factorization with support
constraints associated to some well-known fast transforms such as the Hadamard
or the Discrete Fourier Transforms.

    

### [[2110.11312] Towards modelling hazard factors in unstructured data spaces using gradient-based latent interpolation](http://arxiv.org/abs/2110.11312)


  The application of deep learning in survival analysis (SA) allows utilizing
unstructured and high-dimensional data types uncommon in traditional survival
methods. This allows to advance methods in fields such as digital health,
predictive maintenance, and churn analysis, but often yields less interpretable
and intuitively understandable models due to the black-box character of deep
learning-based approaches. We close this gap by proposing 1) a multi-task
variational autoencoder (VAE) with survival objective, yielding
survival-oriented embeddings, and 2) a novel method HazardWalk that allows to
model hazard factors in the original data space. HazardWalk transforms the
latent distribution of our autoencoder into areas of maximized/minimized hazard
and then uses the decoder to project changes to the original domain. Our
procedure is evaluated on a simulated dataset as well as on a dataset of CT
imaging data of patients with liver metastases.

    

### [[2110.12827] Automatic segmentation of novel coronavirus pneumonia lesions in CT images utilizing deep-supervised ensemble learning network](http://arxiv.org/abs/2110.12827)


  Background: The 2019 novel coronavirus disease (COVID-19) has been spread
widely in the world, causing a huge threat to people's living environment.
Objective: Under computed tomography (CT) imaging, the structure features of
COVID-19 lesions are complicated and varied greatly in different cases. To
accurately locate COVID-19 lesions and assist doctors to make the best
diagnosis and treatment plan, a deep-supervised ensemble learning network is
presented for COVID-19 lesion segmentation in CT images. Methods: Considering
the fact that a large number of COVID-19 CT images and the corresponding lesion
annotations are difficult to obtained, a transfer learning strategy is employed
to make up for the shortcoming and alleviate the overfitting problem. Based on
the reality that traditional single deep learning framework is difficult to
extract COVID-19 lesion features effectively, which may cause some lesions to
be undetected. To overcome the problem, a deep-supervised ensemble learning
network is presented to combine with local and global features for COVID-19
lesion segmentation. Results: The performance of the proposed method was
validated in experiments with a publicly available dataset. Compared with
manual annotations, the proposed method acquired a high intersection over union
(IoU) of 0.7279. Conclusion: A deep-supervised ensemble learning network was
presented for coronavirus pneumonia lesion segmentation in CT images. The
effectiveness of the proposed method was verified by visual inspection and
quantitative evaluation. Experimental results shown that the proposed mehtod
has a perfect performance in COVID-19 lesion segmentation.

    

### [[2111.07941] Distribution Compression in Near-linear Time](http://arxiv.org/abs/2111.07941)


  In distribution compression, one aims to accurately summarize a probability
distribution $\mathbb{P}$ using a small number of representative points.
Near-optimal thinning procedures achieve this goal by sampling $n$ points from
a Markov chain and identifying $\sqrt{n}$ points with
$\widetilde{\mathcal{O}}(1/\sqrt{n})$ discrepancy to $\mathbb{P}$.
Unfortunately, these algorithms suffer from quadratic or super-quadratic
runtime in the sample size $n$. To address this deficiency, we introduce
Compress++, a simple meta-procedure for speeding up any thinning algorithm
while suffering at most a factor of $4$ in error. When combined with the
quadratic-time kernel halving and kernel thinning algorithms of Dwivedi and
Mackey (2021), Compress++ delivers $\sqrt{n}$ points with
$\mathcal{O}(\sqrt{\log n/n})$ integration error and better-than-Monte-Carlo
maximum mean discrepancy in $\mathcal{O}(n \log^3 n)$ time and $\mathcal{O}(
\sqrt{n} \log^2 n )$ space. Moreover, Compress++ enjoys the same near-linear
runtime given any quadratic-time input and reduces the runtime of
super-quadratic algorithms by a square-root factor. In our benchmarks with
high-dimensional Monte Carlo samples and Markov chains targeting challenging
differential equation posteriors, Compress++ matches or nearly matches the
accuracy of its input algorithm in orders of magnitude less time.

    

### [[1707.02294] A case study of Empirical Bayes in User-Movie Recommendation system](http://arxiv.org/abs/1707.02294)


  In this article we provide a formulation of empirical bayes described by
Atchade (2011) to tune the hyperparameters of priors used in bayesian set up of
collaborative filter. We implement the same in MovieLens small dataset. We see
that it can be used to get a good initial choice for the parameters. It can
also be used to guess an initial choice for hyper-parameters in grid search
procedure even for the datasets where MCMC oscillates around the true value or
takes long time to converge.

    

### [[2111.09222] Early DSE and Automatic Generation of Coarse Grained Merged Accelerators](http://arxiv.org/abs/2111.09222)


  Post-Moore's law area-constrained systems rely on accelerators to deliver
performance enhancements. Coarse grained accelerators can offer substantial
domain acceleration, but manual, ad-hoc identification of code to accelerate is
prohibitively expensive. Because cycle-accurate simulators and high-level
synthesis flows are so time-consuming, manual creation of high-utilization
accelerators that exploit control and data flow patterns at optimal
granularities is rarely successful. To address these challenges, we present
AccelMerger, the first automated methodology to create coarse grained, control-
and data-flow-rich, merged accelerators. AccelMerger uses sequence alignment
matching to recognize similar function call-graphs and loops, and neural
networks to quickly evaluate their post-HLS characteristics. It accurately
identifies which functions to accelerate, and it merges accelerators to respect
an area budget and to accommodate system communication characteristics like
latency and bandwidth. Merging two accelerators can save as much as 99% of the
area of one. The space saved is used by a globally optimal integer linear
program to allocate more accelerators for increased performance. We demonstate
AccelMerger's effectiveness using HLS flows without any manual effort to
fine-tune the resulting designs. On FPGA-based systems, AccelMerger yields
application performance improvements of up to 16.7x over software
implementations, and 1.91x on average with respect to state-of-the-art
early-stage design space exploration tools.

    

### [[2111.09272] ReaLPrune: ReRAM Crossbar-aware Lottery Ticket Pruned CNNs](http://arxiv.org/abs/2111.09272)


  ReRAM-based architectures offer high-performance yet energy efficient
computing platforms for CNN training/inferencing. However, ReRAM-based
architectures are not scalable with the size of the CNN. Larger CNNs have more
weights, which requires more ReRAM cells that cannot be integrated in a single
chip. Pruning is an effective way to solve this problem. However, existing
pruning techniques are either targeted for inferencing only, or they are not
crossbar-aware. This leads to sub-optimal hardware savings and performance
benefits for CNN training on ReRAM-based architectures. In this paper, we
address this problem by proposing a novel crossbar-aware pruning strategy,
referred as ReaLPrune, which can prune more than 90% of CNN weights. The pruned
model can be trained from scratch without any accuracy loss. Experimental
results indicate that ReaLPrune reduces hardware requirements by 77.2% and
accelerates CNN training by ~20x compared to unpruned CNNs. ReaLPrune also
outperforms other state-of-the-art crossbar-aware pruning techniques in terms
of both performance and hardware savings

    

### [[2106.12169] APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Cores](http://arxiv.org/abs/2106.12169)


  Over the years, accelerating neural networks with quantization has been
widely studied. Unfortunately, prior efforts with diverse precisions (e.g.,
1-bit weights and 2-bit activations) are usually restricted by limited
precision support on GPUs (e.g., int1 and int4). To break such restrictions, we
introduce the first Arbitrary Precision Neural Network framework (APNN-TC) to
fully exploit quantization benefits on Ampere GPU Tensor Cores. Specifically,
APNN-TC first incorporates a novel emulation algorithm to support arbitrary
short bit-width computation with int1 compute primitives and XOR/AND Boolean
operations. Second, APNN-TC integrates arbitrary precision layer designs to
efficiently map our emulation algorithm to Tensor Cores with novel batching
strategies and specialized memory organization. Third, APNN-TC embodies a novel
arbitrary precision NN design to minimize memory access across layers and
further improve performance. Extensive evaluations show that APNN-TC can
achieve significant speedup over CUTLASS kernels and various NN models, such as
ResNet and VGG.

    

### [[2111.08978] An energy-efficient scheduling algorithm for shared facility supercomputer centers](http://arxiv.org/abs/2111.08978)


  The evolution of high-performance computing is associated with the growth of
energy consumption. Performance of cluster computes (is increased via rising in
performance and the number of used processors, GPUs, and coprocessors. An
increment in the number of computing elements results in significant growth of
energy consumption. Power grids limits for supercomputer centers (SCC) are
driving the transition to more energy-efficient solutions. Often upgrade of
computing resources is done step-by-step, i.e. parts of older supercomputers
are removed from service and replaced with newer ones. A single SCC at any time
can operate several computing systems with different performance and power
consumption. That is why the problem of scheduling parallel programs execution
on SCC resources to optimize energy consumption and minimize the increase in
execution time (energy-efficient scheduling) is important. The goal of the
presented work was the development of a new energy-efficient algorithm for
scheduling computing resources at SCC. To reach the goal the authors analyzed
methods of scheduling computing resources in a shared facility, including
energy consumption minimizing methods. The study made it possible to formulate
the problem of energy-efficient scheduling for a set of CCs and propose an
algorithm for its solution. Experiments on NPB benchmarks allowed achieving
significant reduction in energy consumption with a minor increase of runtime.

    

### [[2111.09153] Top-k Dynamic Service Composition in Skyway Networks](http://arxiv.org/abs/2111.09153)


  We propose a novel top-k service composition framework for drone services
under a dynamic environment. We develop a system model for formal modelling of
drone services in a skyway network. The composition process is accomplished in
two phases, i.e., computing top-k compositions and extending and ranking top-k
compositions using probabilistic wait and recharge times under congestion
conditions. We propose a top-k composition algorithm to compute the best
service composition plan meeting user's requirements. A set of experiments with
a real dataset is conducted to demonstrate the effectiveness of the proposed
approach.

    

### [[2111.09219] Accelerating JPEG Decompression on GPUs](http://arxiv.org/abs/2111.09219)


  The JPEG compression format has been the standard for lossy image compression
for over multiple decades, offering high compression rates at minor perceptual
loss in image quality. For GPU-accelerated computer vision and deep learning
tasks, such as the training of image classification models, efficient JPEG
decoding is essential due to limitations in memory bandwidth. As many decoder
implementations are CPU-based, decoded image data has to be transferred to
accelerators like GPUs via interconnects such as PCI-E, implying decreased
throughput rates. JPEG decoding therefore represents a considerable bottleneck
in these pipelines. In contrast, efficiency could be vastly increased by
utilizing a GPU-accelerated decoder. In this case, only compressed data needs
to be transferred, as decoding will be handled by the accelerators. In order to
design such a GPU-based decoder, the respective algorithms must be parallelized
on a fine-grained level. However, parallel decoding of individual JPEG files
represents a complex task. In this paper, we present an efficient method for
JPEG image decompression on GPUs, which implements an important subset of the
JPEG standard. The proposed algorithm evaluates codeword locations at arbitrary
positions in the bitstream, thereby enabling parallel decompression of
independent chunks. Our performance evaluation shows that on an A100 (V100) GPU
our implementation can outperform the state-of-the-art implementations
libjpeg-turbo (CPU) and nvJPEG (GPU) by a factor of up to 51 (34) and 8.0
(5.7). Furthermore, it achieves a speedup of up to 3.4 over nvJPEG accelerated
with the dedicated hardware JPEG decoder on an A100.

    

### [[2103.13351] RDMA is Turing complete, we just did not know it yet!](http://arxiv.org/abs/2103.13351)


  It is becoming increasingly popular for distributed systems to exploit
offload to reduce load on the CPU. Remote Direct Memory Access (RDMA) offload,
in particular, has become popular. However, RDMA still requires CPU
intervention for complex offloads that go beyond simple remote memory access.
As such, the offload potential is limited and RDMA-based systems usually have
to work around such limitations.
We present RedN, a principled, practical approach to implementing complex
RDMA offloads, without requiring any hardware modifications. Using
self-modifying RDMA chains, we lift the existing RDMA verbs interface to a
Turing complete set of programming abstractions. We explore what is possible in
terms of offload complexity and performance with a commodity RDMA NIC. We show
how to integrate these RDMA chains into applications, such as the Memcached
key-value store, allowing us to offload complex tasks such as key lookups. RedN
can reduce the latency of key-value get operations by up to 2.6x compared to
state-of-the-art KV designs that use one-sided RDMA primitives (e.g., FaRM-KV),
as well as traditional RPC-over-RDMA approaches. Moreover, compared to these
baselines, RedN provides performance isolation and, in the presence of
contention, can reduce latency by up to 35x while providing applications with
failure resiliency to OS and process crashes.

    

### [[2106.02942] Time-Optimal Sublinear Algorithms for Matching and Vertex Cover](http://arxiv.org/abs/2106.02942)


  We study the problem of estimating the size of maximum matching and minimum
vertex cover in sublinear time. Denoting the number of vertices by $n$ and the
average degree in the graph by $\bar{d}$, we obtain the following results for
both problems:
* A multiplicative $(2+\epsilon)$-approximation that takes
$\tilde{O}(n/\epsilon^2)$ time using adjacency list queries.
* A multiplicative-additive $(2, \epsilon n)$-approximation in
$\tilde{O}((\bar{d} + 1)/\epsilon^2)$ time using adjacency list queries.
* A multiplicative-additive $(2, \epsilon n)$-approximation in
$\tilde{O}(n/\epsilon^{3})$ time using adjacency matrix queries.
All three results are provably time-optimal up to polylogarithmic factors
culminating a long line of work on these problems.
Our main contribution and the key ingredient leading to the bounds above is a
new and near-tight analysis of the average query complexity of the randomized
greedy maximal matching algorithm which improves upon a seminal result of
Yoshida, Yamamoto, and Ito [STOC'09].

    

### [[2111.08738] Synthesis-Guided Feature Learning for Cross-Spectral Periocular Recognition](http://arxiv.org/abs/2111.08738)


  A common yet challenging scenario in periocular biometrics is cross-spectral
matching - in particular, the matching of visible wavelength against
near-infrared (NIR) periocular images. We propose a novel approach to
cross-spectral periocular verification that primarily focuses on learning a
mapping from visible and NIR periocular images to a shared latent
representational subspace, and supports this effort by simultaneously learning
intra-spectral image reconstruction. We show the auxiliary image reconstruction
task (and in particular the reconstruction of high-level, semantic features)
results in learning a more discriminative, domain-invariant subspace compared
to the baseline while incurring no additional computational or memory costs at
test-time. The proposed Coupled Conditional Generative Adversarial Network
(CoGAN) architecture uses paired generator networks (one operating on visible
images and the other on NIR) composed of U-Nets with ResNet-18 encoders trained
for feature learning via contrastive loss and for intra-spectral image
reconstruction with adversarial, pixel-based, and perceptual reconstruction
losses. Moreover, the proposed CoGAN model beats the current state-of-art
(SotA) in cross-spectral periocular recognition. On the Hong Kong PolyU
benchmark dataset, we achieve 98.65% AUC and 5.14% EER compared to the SotA EER
of 8.02%. On the Cross-Eyed dataset, we achieve 99.31% AUC and 3.99% EER versus
SotA EER of 4.39%.

    

### [[2111.08755] Learning Scene Dynamics from Point Cloud Sequences](http://arxiv.org/abs/2111.08755)


  Understanding 3D scenes is a critical prerequisite for autonomous agents.
Recently, LiDAR and other sensors have made large amounts of data available in
the form of temporal sequences of point cloud frames. In this work, we propose
a novel problem -- sequential scene flow estimation (SSFE) -- that aims to
predict 3D scene flow for all pairs of point clouds in a given sequence. This
is unlike the previously studied problem of scene flow estimation which focuses
on two frames.
We introduce the SPCM-Net architecture, which solves this problem by
computing multi-scale spatiotemporal correlations between neighboring point
clouds and then aggregating the correlation across time with an order-invariant
recurrent unit. Our experimental evaluation confirms that recurrent processing
of point cloud sequences results in significantly better SSFE compared to using
only two frames. Additionally, we demonstrate that this approach can be
effectively modified for sequential point cloud forecasting (SPF), a related
problem that demands forecasting future point cloud frames.
Our experimental results are evaluated using a new benchmark for both SSFE
and SPF consisting of synthetic and real datasets. Previously, datasets for
scene flow estimation have been limited to two frames. We provide non-trivial
extensions to these datasets for multi-frame estimation and prediction. Due to
the difficulty of obtaining ground truth motion for real-world datasets, we use
self-supervised training and evaluation metrics. We believe that this benchmark
will be pivotal to future research in this area. All code for benchmark and
models will be made accessible.

    

### [[2111.08817] Compressive Features in Offline Reinforcement Learning for Recommender Systems](http://arxiv.org/abs/2111.08817)


  In this paper, we develop a recommender system for a game that suggests
potential items to players based on their interactive behaviors to maximize
revenue for the game provider. Our approach is built on a reinforcement
learning-based technique and is trained on an offline data set that is publicly
available on an IEEE Big Data Cup challenge. The limitation of the offline data
set and the curse of high dimensionality pose significant obstacles to solving
this problem. Our proposed method focuses on improving the total rewards and
performance by tackling these main difficulties. More specifically, we utilized
sparse PCA to extract important features of user behaviors. Our
Q-learning-based system is then trained from the processed offline data set. To
exploit all possible information from the provided data set, we cluster user
features to different groups and build an independent Q-table for each group.
Furthermore, to tackle the challenge of unknown formula for evaluation metrics,
we design a metric to self-evaluate our system's performance based on the
potential value the game provider might achieve and a small collection of
actual evaluation metrics that we obtain from the live scoring environment. Our
experiments show that our proposed metric is consistent with the results
published by the challenge organizers. We have implemented the proposed
training pipeline, and the results show that our method outperforms current
state-of-the-art methods in terms of both total rewards and training speed. By
addressing the main challenges and leveraging the state-of-the-art techniques,
we have achieved the best public leaderboard result in the challenge.
Furthermore, our proposed method achieved an estimated score of approximately
20% better and can be trained faster by 30 times than the best of the current
state-of-the-art methods.

    

### [[2111.08826] A Benchmark for Modeling Violation-of-Expectation in Physical Reasoning Across Event Categories](http://arxiv.org/abs/2111.08826)


  Recent work in computer vision and cognitive reasoning has given rise to an
increasing adoption of the Violation-of-Expectation (VoE) paradigm in synthetic
datasets. Inspired by infant psychology, researchers are now evaluating a
model's ability to label scenes as either expected or surprising with knowledge
of only expected scenes. However, existing VoE-based 3D datasets in physical
reasoning provide mainly vision data with little to no heuristics or inductive
biases. Cognitive models of physical reasoning reveal infants create high-level
abstract representations of objects and interactions. Capitalizing on this
knowledge, we established a benchmark to study physical reasoning by curating a
novel large-scale synthetic 3D VoE dataset armed with ground-truth heuristic
labels of causally relevant features and rules. To validate our dataset in five
event categories of physical reasoning, we benchmarked and analyzed human
performance. We also proposed the Object File Physical Reasoning Network
(OFPR-Net) which exploits the dataset's novel heuristics to outperform our
baseline and ablation models. The OFPR-Net is also flexible in learning an
alternate physical reality, showcasing its ability to learn universal causal
relationships in physical reasoning to create systems with better
interpretability.

    

### [[2111.08858] A Normative and Biologically Plausible Algorithm for Independent Component Analysis](http://arxiv.org/abs/2111.08858)


  The brain effortlessly solves blind source separation (BSS) problems, but the
algorithm it uses remains elusive. In signal processing, linear BSS problems
are often solved by Independent Component Analysis (ICA). To serve as a model
of a biological circuit, the ICA neural network (NN) must satisfy at least the
following requirements: 1. The algorithm must operate in the online setting
where data samples are streamed one at a time, and the NN computes the sources
on the fly without storing any significant fraction of the data in memory. 2.
The synaptic weight update is local, i.e., it depends only on the biophysical
variables present in the vicinity of a synapse. Here, we propose a novel
objective function for ICA from which we derive a biologically plausible NN,
including both the neural architecture and the synaptic learning rules.
Interestingly, our algorithm relies on modulating synaptic plasticity by the
total activity of the output neurons. In the brain, this could be accomplished
by neuromodulators, extracellular calcium, local field potential, or nitric
oxide.

    

### [[2111.08897] ARKitScenes -- A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data](http://arxiv.org/abs/2111.08897)


  Scene understanding is an active research area. Commercial depth sensors,
such as Kinect, have enabled the release of several RGB-D datasets over the
past few years which spawned novel methods in 3D scene understanding. More
recently with the launch of the LiDAR sensor in Apple's iPads and iPhones, high
quality RGB-D data is accessible to millions of people on a device they
commonly use. This opens a whole new era in scene understanding for the
Computer Vision community as well as app developers. The fundamental research
in scene understanding together with the advances in machine learning can now
impact people's everyday experiences. However, transforming these scene
understanding methods to real-world experiences requires additional innovation
and development. In this paper we introduce ARKitScenes. It is not only the
first RGB-D dataset that is captured with a now widely available depth sensor,
but to our best knowledge, it also is the largest indoor scene understanding
data released. In addition to the raw and processed data from the mobile
device, ARKitScenes includes high resolution depth maps captured using a
stationary laser scanner, as well as manually labeled 3D oriented bounding
boxes for a large taxonomy of furniture. We further analyze the usefulness of
the data for two downstream tasks: 3D object detection and color-guided depth
upsampling. We demonstrate that our dataset can help push the boundaries of
existing state-of-the-art methods and it introduces new challenges that better
represent real-world scenarios.

    

### [[2111.08951] Exploring Student Representation For Neural Cognitive Diagnosis](http://arxiv.org/abs/2111.08951)


  Cognitive diagnosis, the goal of which is to obtain the proficiency level of
students on specific knowledge concepts, is an fundamental task in smart
educational systems. Previous works usually represent each student as a
trainable knowledge proficiency vector, which cannot capture the relations of
concepts and the basic profile(e.g. memory or comprehension) of students. In
this paper, we propose a method of student representation with the exploration
of the hierarchical relations of knowledge concepts and student embedding.
Specifically, since the proficiency on parent knowledge concepts reflects the
correlation between knowledge concepts, we get the first knowledge proficiency
with a parent-child concepts projection layer. In addition, a low-dimension
dense vector is adopted as the embedding of each student, and obtain the second
knowledge proficiency with a full connection layer. Then, we combine the two
proficiency vector above to get the final representation of students.
Experiments show the effectiveness of proposed representation method.

    

### [[2111.09078] Green CWS: Extreme Distillation and Efficient Decode Method Towards Industrial Application](http://arxiv.org/abs/2111.09078)


  Benefiting from the strong ability of the pre-trained model, the research on
Chinese Word Segmentation (CWS) has made great progress in recent years.
However, due to massive computation, large and complex models are incapable of
empowering their ability for industrial use. On the other hand, for
low-resource scenarios, the prevalent decode method, such as Conditional Random
Field (CRF), fails to exploit the full information of the training data. This
work proposes a fast and accurate CWS framework that incorporates a
light-weighted model and an upgraded decode method (PCRF) towards industrially
low-resource CWS scenarios. First, we distill a Transformer-based student model
as an encoder, which not only accelerates the inference speed but also combines
open knowledge and domain-specific knowledge. Second, the perplexity score to
evaluate the language model is fused into the CRF module to better identify the
word boundaries. Experiments show that our work obtains relatively high
performance on multiple datasets with as low as 14\% of time consumption
compared with the original BERT-based model. Moreover, under the low-resource
setting, we get superior results in comparison with the traditional decoding
methods.

    

### [[2111.09084] A Graph-based Imputation Method for Sparse Medical Records](http://arxiv.org/abs/2111.09084)


  Electronic Medical Records (EHR) are extremely sparse. Only a small
proportion of events (symptoms, diagnoses, and treatments) are observed in the
lifetime of an individual. The high degree of missingness of EHR can be
attributed to a large number of factors, including device failure, privacy
concerns, or other unexpected reasons. Unfortunately, many traditional
imputation methods are not well suited for highly sparse data and scale poorly
to high dimensional datasets. In this paper, we propose a graph-based
imputation method that is both robust to sparsity and to unreliable unmeasured
events. Our approach compares favourably to several standard and
state-of-the-art imputation methods in terms of performance and runtime.
Moreover, results indicate that the model learns to embed different event types
in a clinically meaningful way. Our work can facilitate the diagnosis of novel
diseases based on the clinical history of past events, with the potential to
increase our understanding of the landscape of comorbidities.

    

### [[2111.09093] The Faulty GPS Problem: Shortest Time Paths in Networks with Unreliable Directions](http://arxiv.org/abs/2111.09093)


  This paper optimizes motion planning when there is a known risk that the road
choice suggested by a Satnav (GPS) is not on a shortest path. At every branch
node of a network Q, a Satnav (GPS) points to the arc leading to the
destination, or home node, H - but only with a high known probability p. Always
trusting the Satnav's suggestion may lead to an infinite cycle. If one wishes
to reach H in least expected time, with what probability q=q(Q,p) should one
trust the pointer (if not, one chooses randomly among the other arcs)? We call
this the Faulty Satnav (GPS) Problem. We also consider versions where the trust
probability q can depend on the degree of the current node and a `treasure
hunt' where two searchers try to reach H first. The agent searching for H need
not be a car, that is just a familiar example -- it could equally be a UAV
receiving unreliable GPS information. This problem has its origin not in driver
frustration but in the work of Fonio et al (2017) on ant navigation, where the
pointers correspond to pheromone markers pointing to the nest. Neither the
driver or ant will know the exact process by which a choice (arc) is suggested,
which puts the problem into the domain of how much to trust an option suggested
by AI.

    

### [[2111.09152] Improved cooperation by balancing exploration and exploitation in intertemporal social dilemma tasks](http://arxiv.org/abs/2111.09152)


  When an individual's behavior has rational characteristics, this may lead to
irrational collective actions for the group. A wide range of organisms from
animals to humans often evolve the social attribute of cooperation to meet this
challenge. Therefore, cooperation among individuals is of great significance
for allowing social organisms to adapt to changes in the natural environment.
Based on multi-agent reinforcement learning, we propose a new learning strategy
for achieving coordination by incorporating a learning rate that can balance
exploration and exploitation. We demonstrate that agents that use the simple
strategy improve a relatively collective return in a decision task called the
intertemporal social dilemma, where the conflict between the individual and the
group is particularly sharp. We also explore the effects of the diversity of
learning rates on the population of reinforcement learning agents and show that
agents trained in heterogeneous populations develop particularly coordinated
policies relative to those trained in homogeneous populations.

    

### [[2111.09259] Acquisition of Chess Knowledge in AlphaZero](http://arxiv.org/abs/2111.09259)


  What is being learned by superhuman neural network agents such as AlphaZero?
This question is of both scientific and practical interest. If the
representations of strong neural networks bear no resemblance to human
concepts, our ability to understand faithful explanations of their decisions
will be restricted, ultimately limiting what we can achieve with neural network
interpretability. In this work we provide evidence that human knowledge is
acquired by the AlphaZero neural network as it trains on the game of chess. By
probing for a broad range of human chess concepts we show when and where these
concepts are represented in the AlphaZero network. We also provide a
behavioural analysis focusing on opening play, including qualitative analysis
from chess Grandmaster Vladimir Kramnik. Finally, we carry out a preliminary
investigation looking at the low-level details of AlphaZero's representations,
and make the resulting behavioural and representational analyses available
online.

    

### [[2111.09267] DiverGAN: An Efficient and Effective Single-Stage Framework for Diverse Text-to-Image Generation](http://arxiv.org/abs/2111.09267)


  In this paper, we present an efficient and effective single-stage framework
(DiverGAN) to generate diverse, plausible and semantically consistent images
according to a natural-language description. DiverGAN adopts two novel
word-level attention modules, i.e., a channel-attention module (CAM) and a
pixel-attention module (PAM), which model the importance of each word in the
given sentence while allowing the network to assign larger weights to the
significant channels and pixels semantically aligning with the salient words.
After that, Conditional Adaptive Instance-Layer Normalization (CAdaILN) is
introduced to enable the linguistic cues from the sentence embedding to
flexibly manipulate the amount of change in shape and texture, further
improving visual-semantic representation and helping stabilize the training.
Also, a dual-residual structure is developed to preserve more original visual
features while allowing for deeper networks, resulting in faster convergence
speed and more vivid details. Furthermore, we propose to plug a fully-connected
layer into the pipeline to address the lack-of-diversity problem, since we
observe that a dense layer will remarkably enhance the generative capability of
the network, balancing the trade-off between a low-dimensional random latent
code contributing to variants and modulation modules that use high-dimensional
and textual contexts to strength feature maps. Inserting a linear layer after
the second residual block achieves the best variety and quality. Both
qualitative and quantitative results on benchmark data sets demonstrate the
superiority of our DiverGAN for realizing diversity, without harming quality
and semantic consistency.

    

### [[2111.09301] Learning to Align Sequential Actions in the Wild](http://arxiv.org/abs/2111.09301)


  State-of-the-art methods for self-supervised sequential action alignment rely
on deep networks that find correspondences across videos in time. They either
learn frame-to-frame mapping across sequences, which does not leverage temporal
information, or assume monotonic alignment between each video pair, which
ignores variations in the order of actions. As such, these methods are not able
to deal with common real-world scenarios that involve background frames or
videos that contain non-monotonic sequence of actions.
In this paper, we propose an approach to align sequential actions in the wild
that involve diverse temporal variations. To this end, we propose an approach
to enforce temporal priors on the optimal transport matrix, which leverages
temporal consistency, while allowing for variations in the order of actions.
Our model accounts for both monotonic and non-monotonic sequences and handles
background frames that should not be aligned. We demonstrate that our approach
consistently outperforms the state-of-the-art in self-supervised sequential
action representation learning on four different benchmark datasets.

    

### [[2103.13268] Address Behaviour Vulnerabilities in the Next Generation of Autonomous Robots](http://arxiv.org/abs/2103.13268)


  Robots applications in our daily life increase at an unprecedented pace. As
robots will soon operate "out in the wild", we must identify the safety and
security vulnerabilities they will face. Robotics researchers and manufacturers
focus their attention on new, cheaper, and more reliable applications. Still,
they often disregard the operability in adversarial environments where a
trusted or untrusted user can jeopardize or even alter the robot's task.
In this paper, we identify a new paradigm of security threats in the next
generation of robots. These threats fall beyond the known hardware or
network-based ones, and we must find new solutions to address them. These new
threats include malicious use of the robot's privileged access, tampering with
the robot sensors system, and tricking the robot's deliberation into harmful
behaviors. We provide a taxonomy of attacks that exploit these vulnerabilities
with realistic examples, and we outline effective countermeasures to prevent
better, detect, and mitigate them.

    

### [[2104.06890] An Introduction of mini-AlphaStar](http://arxiv.org/abs/2104.06890)


  StarCraft II (SC2) is a real-time strategy game in which players produce and
control multiple units to fight against opponent's units. Due to its
difficulties, such as huge state space, various action space, a long time
horizon, and imperfect information, SC2 has been a research hotspot in
reinforcement learning. Recently, an agent called AlphaStar (AS) has been
proposed, which shows good performance, obtaining a high win rate of 99.8%
against human players. We implemented a mini-scaled version of it called
mini-AlphaStar (mAS) based on AS's paper and pseudocode. The difference between
AS and mAS is that we substituted the hyper-parameters of AS with smaller ones
for mini-scale training. Codes of mAS are all open-sourced
(this https URL) for future research.

    

### [[2111.09022] Context-Bounded Verification of Thread Pools](http://arxiv.org/abs/2111.09022)


  Thread pooling is a common programming idiom in which a fixed set of worker
threads are maintained to execute tasks concurrently. The workers repeatedly
pick tasks and execute them to completion. Each task is sequential, with
possibly recursive code, and tasks communicate over shared memory. Executing a
task can lead to more new tasks being spawned. We consider the safety
verification problem for thread-pooled programs. We parameterize the problem
with two parameters: the size of the thread pool as well as the number of
context switches for each task. The size of the thread pool determines the
number of workers running concurrently. The number of context switches
determines how many times a worker can be swapped out while executing a single
task - like many verification problems for multithreaded recursive programs,
the context bounding is important for decidability.
We show that the safety verification problem for thread-pooled,
context-bounded, Boolean programs is EXPSPACE-complete, even if the size of the
thread pool and the context bound are given in binary. Our main result, the
EXPSPACE upper bound, is derived using a sequence of new succinct encoding
techniques of independent language-theoretic interest. In particular, we show a
polynomial-time construction of downward closures of languages accepted by
succinct pushdown automata as doubly succinct nondeterministic finite automata.
While there are explicit doubly exponential lower bounds on the size of
nondeterministic finite automata accepting the downward closure, our result
shows these automata can be compressed. We show that thread pooling
significantly reduces computational power: in contrast, if only the context
bound is provided in binary, but there is no thread pooling, the safety
verification problem becomes 3EXPSPACE-complete.

    

### [[2102.10784] SigVM: Enabling Event-Driven Execution for Autonomous Smart Contracts](http://arxiv.org/abs/2102.10784)


  This paper presents SigVM, a novel blockchain virtual machine that supports
an event-driven execution model, enabling developers to build autonomous smart
contracts. Contracts in SigVM can emit signal events, on which other contracts
can listen. Once an event is triggered, corresponding handler functions are
automatically executed as signal transactions. We build an end-to-end
blockchain platform SigChain and a contract language compiler SigSolid to
realize the potential of SigVM. Experimental results show that our benchmark
applications can be reimplemented with SigVM in an autonomous way, eliminating
the dependency on unreliable mechanisms like off-chain relay servers. The
development effort of reimplementing these contracts with SigVM is small, i.e.,
we modified on average 2.6% of the contract code.

    

### [<title>ValueError: Input contains NaN, infinity or a value too large for dtype('float32') - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtype-float32/1095/8)