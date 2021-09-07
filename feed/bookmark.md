
## 2021-9-7

### [[2109.01737] AppSlice: A system for application-centric design of 5G and edge computing applications](http://arxiv.org/abs/2109.01737)


  Applications that use edge computing and 5G to improve response times consume
both compute and network resources. However, 5G networks manage only network
resources without considering the application's compute requirements, and
container orchestration frameworks manage only compute resources without
considering the application's network requirements. We observe that there is a
complex coupling between an application's compute and network usage, which can
be leveraged to improve application performance and resource utilization. We
propose a new, declarative abstraction called AppSlice that jointly considers
the application's compute and network requirements. This abstraction leverages
container management systems to manage edge computing resources, and 5G network
stacks to manage network resources, while the joint consideration of coupling
between compute and network usage is explicitly managed by a new runtime
system, which delivers the declarative semantics of the app slice. The runtime
system also jointly manages the edge compute and network resource usage
automatically across different edge computing environments and 5G networks by
using two adaptive algorithms. We implement a complex, real-world, real-time
monitoring application using the proposed app slice abstraction, and
demonstrate on a private 5G/LTE testbed that the proposed runtime system
significantly improves the application performance and resource usage when
compared with the case where the coupling between the compute and network
resource usage is ignored.

    

### [[2109.01855] Network Traffic Characteristics of IoT Devices in Smart Homes](http://arxiv.org/abs/2109.01855)


  Understanding network traffic characteristics of IoT devices plays a critical
role in improving both the performance and security of IoT devices, including
IoT device identification, classification, and anomaly detection. Although a
number of existing research efforts have developed machine-learning based
algorithms to help address the challenges in improving the security of IoT
devices, none of them have provided detailed studies on the network traffic
characteristics of IoT devices. In this paper we collect and analyze the
network traffic generated in a typical smart homes environment consisting of a
set of common IoT (and non-IoT) devices. We analyze the network traffic
characteristics of IoT devices from three complementary aspects: remote network
servers and port numbers that IoT devices connect to, flow-level traffic
characteristics such as flow duration, and packet-level traffic characteristics
such as packet inter-arrival time. Our study provides critical insights into
the operational and behavioral characteristics of IoT devices, which can help
develop more effective security and performance algorithms for IoT devices.

    

### [[2109.01971] Horizontal and Vertical Collaboration for VR Delivery in MEC-Enabled Small-Cell Networks](http://arxiv.org/abs/2109.01971)


  Due to the large bandwidth, low latency and computationally intensive
features of virtual reality (VR) video applications, the current
resource-constrained wireless and edge networks cannot meet the requirements of
on-demand VR delivery. In this letter, we propose a joint horizontal and
vertical collaboration architecture in mobile edge computing (MEC)-enabled
small-cell networks for VR delivery. In the proposed architecture, multiple MEC
servers can jointly provide VR head-mounted devices (HMDs) with edge caching
and viewpoint computation services, while the computation tasks can also be
performed at HMDs or on the cloud. Power allocation at base stations (BSs) is
considered in coordination with horizontal collaboration (HC) and vertical
collaboration (VC) of MEC servers to obtain lower end-to-end latency of VR
delivery. A joint caching, power allocation and task offloading problem is then
formulated, and a discrete branch-reduce-and-bound (DBRB) algorithm inspired by
monotone optimization is proposed to effectively solve the problem. Simulation
results demonstrate the advantage of the proposed architecture and algorithm in
terms of existing ones.

    

### [[2109.02042] A Survey on IoT Smart Healthcare: Emerging Technologies, Applications, Challenges, and Future Trends](http://arxiv.org/abs/2109.02042)


  The internet of things (IoT) refers to a framework of interrelated, web
associated objects that can gather and move information over a remote network
without human interference. With a quick development in the arrangement of IoT
gadgets and expanding want to make medical care more financially savvy,
customized, and proactive, IoT is ready to assume a solid function in all
perspectives of the healthcare industry. In this context, IoT-based healthcare
provides several benefits such as instant and reliable treatment, cost
reduction, improved communication, etc. by using different new technologies.
Wireless Body Area Networks (WBAN) technologies can enhance the quality of data
gathering and data transferring in smart systems. Machine Learning(ML) are put
to use at every level of smart healthcare systems. Fog computing reduces
communication cost and provides low latency. Software-Defined Networking (SDN)
and Network Function Virtualization (NFV) technologies provide less complex and
more flexible network structures. Blockchain technology offers a better way of
protection of users' sensitive information. This paper aims to provide
comprehensive descriptions of ongoing research projects and the utilization of
the above-mentioned technologies in smart healthcare systems. In this paper,
the latest studies, proposed techniques, and the current solutions of smart
healthcare systems are elaborated in the context of emerging technologies,
applications and challenges of these systems to provide a better understanding
of what IoT means in the healthcare industry now and what it will mean in the
future.

    

### [[2109.02186] Achieving QoS for Real-Time Bursty Applications over Passive Optical Networks](http://arxiv.org/abs/2109.02186)


  Emerging real-time applications such as those classified under ultra-reliable
low latency (uRLLC) generate bursty traffic and have strict Quality of Service
(QoS) requirements. Passive Optical Network (PON) is a popular access network
technology, which is envisioned to handle such applications at the access
segment of the network. However, the existing standards cannot handle strict
QoS constraints. The available solutions rely on instantaneous heuristic
decisions and maintain QoS constraints (mostly bandwidth) in an average sense.
Existing works with optimal strategies are computationally complex and are not
suitable for uRLLC applications. This paper presents a novel
computationally-efficient, far-sighted bandwidth allocation policy design for
facilitating bursty traffic in a PON framework while satisfying strict QoS (age
of information/delay and bandwidth) requirements of modern applications. To
this purpose, first we design a delay-tracking mechanism which allows us to
model the resource allocation problem from a control-theoretic viewpoint as a
Model Predictive Control (MPC). MPC helps in taking far-sighted decisions
regarding resource allocations and captures the time-varying dynamics of the
network. We provide computationally efficient polynomial-time solutions and
show its implementation in the PON framework. Compared to existing approaches,
MPC reduces delay violations by approximately 15% for a delay-constrained
application of 1ms target. Our approach is also robust to varying traffic
arrivals.

    

### [[2109.02206] Achieving Deterministic Service in Mobile Edge Computing (MEC) Networks](http://arxiv.org/abs/2109.02206)


  Mobile edge computing (MEC) is proposed to boost high-efficient and
time-sensitive 5G applications. However, the "microburst" may occur even in
lightly-loaded scenarios, which leads to the indeterministic service latency
(i.e., unpredictable delay or delay variation), hence hindering the deployment
of MEC. Deterministic IP networking (DIP) has been proposed that can provide
bounds on latency, and high reliability in the large-scale networks.
Nevertheless, the direct migration of DIP into the MEC network is non-trivial
owing to its original design for the Ethernet with homogeneous devices.
Meanwhile, DIP also faces the challenges on the network throughput and
scheduling flexibility. In this paper, we delve into the adoption of DIP for
the MEC networks and some of the relevant aspects. A deterministic MEC (D-MEC)
network is proposed to deliver the deterministic service (i.e., providing the
MEC service with bounded service latency). In the D-MEC network, two
mechanisms, including the cycle mapping and cycle shifting, are designed to
enable: (i) seamless and deterministic transmission with heterogeneous
underlaid resources; and (ii) traffic shaping on the edges to improve the
resource utilization. We also formulate a joint configuration to maximize the
network throughput with deterministic QoS guarantees. Extensive simulations
verify that the proposed D-MEC network can achieve a deterministic MEC service,
even in the highly-loaded scenarios.

    

### [[2109.02278] TSCH Evaluation under heterogeneous Mobile Scenarios](http://arxiv.org/abs/2109.02278)


  Time Slotted Channel Hopping (TSCH) is a medium access protocol defined in
the IEEE 802.15.4 standard. It has been demonstrated to be one of the most
reliable options when it comes to industrial applications. TSCH offers a degree
of large flexibility and can be tailored to the requirements of specific
applications. Several performance aspects of TSCH have been investigated so
far, such as the energy consumption, the reliability, scalability and many
more. However, mobility in TSCH networks remains an aspect that has not been
thoroughly explored. In this paper we examine how TSCH performs under mobility
situations. We define two mobile scenarios: one where autonomous agriculture
vehicles move on a predefined trail, and a warehouse logistics scenario, where
autonomous robots/vehicles and workers move randomly. We examine how different
TSCH scheduling approaches perform on these mobility patterns and when
different number of nodes are operating. The results show that the current TSCH
scheduling approaches are not able to handle mobile scenarios efficiently.
Moreover, the results provide insights on how TSCH scheduling can be improved
for mobile applications.

    

### [[2109.02336] Towards an Approach to Contextual Detection of Multi-Stage Cyber Attacks in Smart Grids](http://arxiv.org/abs/2109.02336)


  Electric power grids are at risk of being compromised by high-impact
cyber-security threats such as coordinated, timed attacks. Navigating this new
threat landscape requires a deep understanding of the potential risks and
complex attack processes in energy information systems, which in turn demands
an unmanageable manual effort to timely process a large amount of cross-domain
information. To provide an adequate basis to contextually assess and understand
the situation of smart grids in case of coordinated cyber-attacks, we need a
systematic and coherent approach to identify cyber incidents. In this paper, we
present an approach that collects and correlates cross-domain cyber threat
information to detect multi-stage cyber-attacks in energy information systems.
We investigate the applicability and performance of the presented correlation
approach and discuss the results to highlight challenges in domain-specific
detection mechanisms.

    

### [[2109.02353] Reconfigurable Intelligent Surface Empowered Over-the-Air Federated Edge Learning](http://arxiv.org/abs/2109.02353)


  Federated edge learning (FEEL) has emerged as a revolutionary paradigm to
develop AI services at the edge of 6G wireless networks as it supports
collaborative model training at a massive number of mobile devices. However,
model communication over wireless channels, especially in uplink model
uploading of FEEL, has been widely recognized as a bottleneck that critically
limits the efficiency of FEEL. Although over-the-air computation can alleviate
the excessive cost of radio resources in FEEL model uploading, practical
implementations of over-the-air FEEL still suffer from several challenges,
including strong straggler issues, large communication overheads, and potential
privacy leakage. In this article, we study these challenges in over-the-air
FEEL and leverage reconfigurable intelligent surface (RIS), a key enabler of
future wireless systems, to address these challenges. We study the
state-of-the-art solutions on RIS-empowered FEEL and explore the promising
research opportunities for adopting RIS to enhance FEEL performance.

    

### [[2102.04288] Revocation Statuses on the Internet](http://arxiv.org/abs/2102.04288)


  The modern Internet is highly dependent on the trust communicated via X.509
certificates. However, in some cases certificates become untrusted and it is
necessary to revoke them. In practice, the problem of secure certificate
revocation has not yet been solved, and today no revocation procedure (similar
to Certificate Transparency w.r.t. certificate issuance) has been adopted to
provide transparent and immutable history of all revocations. Instead, the
status of most certificates can only be checked with Online Certificate Status
Protocol (OCSP) and/or Certificate Revocation Lists (CRLs). In this paper, we
present the first longitudinal characterization of the revocation statuses
delivered by CRLs and OCSP servers from the time of certificate expiration to
status disappearance. The analysis captures the status history of over 1
million revoked certificates, including 773K certificates mass-revoked by Let's
Encrypt. Our characterization provides a new perspective on the Internet's
revocation rates, quantifies how short-lived the revocation statuses are,
highlights differences in revocation practices within and between different
CAs, and captures biases and oddities in the handling of revoked certificates.
Combined, the findings motivate the development and adoption of a revocation
transparency standard.

    

### [[2109.01656] Thompson Sampling for Bandits with Clustered Arms](http://arxiv.org/abs/2109.01656)


  We propose algorithms based on a multi-level Thompson sampling scheme, for
the stochastic multi-armed bandit and its contextual variant with linear
expected rewards, in the setting where arms are clustered. We show, both
theoretically and empirically, how exploiting a given cluster structure can
significantly improve the regret and computational cost compared to using
standard Thompson sampling. In the case of the stochastic multi-armed bandit we
give upper bounds on the expected cumulative regret showing how it depends on
the quality of the clustering. Finally, we perform an empirical evaluation
showing that our algorithms perform well compared to previously proposed
algorithms for bandits with clustered arms.

    

### [[2109.01657] A Multi-view Multi-task Learning Framework for Multi-variate Time Series Forecasting](http://arxiv.org/abs/2109.01657)


  Multi-variate time series (MTS) data is a ubiquitous class of data
abstraction in the real world. Any instance of MTS is generated from a hybrid
dynamical system and their specific dynamics are usually unknown. The hybrid
nature of such a dynamical system is a result of complex external attributes,
such as geographic location and time of day, each of which can be categorized
into either spatial attributes or temporal attributes. Therefore, there are two
fundamental views which can be used to analyze MTS data, namely the spatial
view and the temporal view. Moreover, from each of these two views, we can
partition the set of data samples of MTS into disjoint forecasting tasks in
accordance with their associated attribute values. Then, samples of the same
task will manifest similar forthcoming pattern, which is less sophisticated to
be predicted in comparison with the original single-view setting. Considering
this insight, we propose a novel multi-view multi-task (MVMT) learning
framework for MTS forecasting. Instead of being explicitly presented in most
scenarios, MVMT information is deeply concealed in the MTS data, which severely
hinders the model from capturing it naturally. To this end, we develop two
kinds of basic operations, namely task-wise affine transformation and task-wise
normalization, respectively. Applying these two operations with prior knowledge
on the spatial and temporal view allows the model to adaptively extract MVMT
information while predicting. Extensive experiments on three datasets are
conducted to illustrate that canonical architectures can be greatly enhanced by
the MVMT learning framework in terms of both effectiveness and efficiency. In
addition, we design rich case studies to reveal the properties of
representations produced at different phases in the entire prediction
procedure.

    

### [[2109.01658] Artificial Intelligence in Dry Eye Disease](http://arxiv.org/abs/2109.01658)


  Dry eye disease (DED) has a prevalence of between 5 and 50\%, depending on
the diagnostic criteria used and population under study. However, it remains
one of the most underdiagnosed and undertreated conditions in ophthalmology.
Many tests used in the diagnosis of DED rely on an experienced observer for
image interpretation, which may be considered subjective and result in
variation in diagnosis. Since artificial intelligence (AI) systems are capable
of advanced problem solving, use of such techniques could lead to more
objective diagnosis. Although the term `AI' is commonly used, recent success in
its applications to medicine is mainly due to advancements in the sub-field of
machine learning, which has been used to automatically classify images and
predict medical outcomes. Powerful machine learning techniques have been
harnessed to understand nuances in patient data and medical images, aiming for
consistent diagnosis and stratification of disease severity. This is the first
literature review on the use of AI in DED. We provide a brief introduction to
AI, report its current use in DED research and its potential for application in
the clinic. Our review found that AI has been employed in a wide range of DED
clinical tests and research applications, primarily for interpretation of
interferometry, slit-lamp and meibography images. While initial results are
promising, much work is still needed on model development, clinical testing and
standardisation.

    

### [[2109.01659] Reinforcement Learning for Battery Energy Storage Dispatch augmented with Model-based Optimizer](http://arxiv.org/abs/2109.01659)


  Reinforcement learning has been found useful in solving optimal power flow
(OPF) problems in electric power distribution systems. However, the use of
largely model-free reinforcement learning algorithms that completely ignore the
physics-based modeling of the power grid compromises the optimizer performance
and poses scalability challenges. This paper proposes a novel approach to
synergistically combine the physics-based models with learning-based algorithms
using imitation learning to solve distribution-level OPF problems.
Specifically, we propose imitation learning based improvements in deep
reinforcement learning (DRL) methods to solve the OPF problem for a specific
case of battery storage dispatch in the power distribution systems. The
proposed imitation learning algorithm uses the approximate optimal solutions
obtained from a linearized model-based OPF solver to provide a good initial
policy for the DRL algorithms while improving the training efficiency. The
effectiveness of the proposed approach is demonstrated using IEEE 34-bus and
123-bus distribution feeders with numerous distribution-level battery storage
systems.

    

### [[2109.01661] Data science and Machine learning in the Clouds: A Perspective for the Future](http://arxiv.org/abs/2109.01661)


  As we are fast approaching the beginning of a paradigm shift in the field of
science, Data driven science (the so called fourth science paradigm) is going
to be the driving force in research and innovation. From medicine to
biodiversity and astronomy to geology, all these terms are somehow going to be
affected by this paradigm shift. The huge amount of data to be processed under
this new paradigm will be a major concern in the future and one will strongly
require cloud based services in all the aspects of these computations (from
storage to compute and other services). Another aspect will be energy
consumption and performance of prediction jobs and tasks within such a
scientific paradigm which will change the way one sees computation. Data
science has heavily impacted or rather triggered the emergence of Machine
Learning, Signal/Image/Video processing related algorithms, Artificial
intelligence, Robotics, health informatics, geoinformatics, and many more such
areas of interest. Hence, we envisage an era where Data science can deliver its
promises with the help of the existing cloud based platforms and services with
the addition of new services. In this article, we discuss about data driven
science and Machine learning and how they are going to be linked through cloud
based services in the future. It also discusses the rise of paradigms like
approximate computing, quantum computing and many more in recent times and
their applicability in big data processing, data science, analytics, prediction
and machine learning in the cloud environments.

    

### [[2109.01667] Hierarchical 3D Feature Learning for Pancreas Segmentation](http://arxiv.org/abs/2109.01667)


  We propose a novel 3D fully convolutional deep network for automated pancreas
segmentation from both MRI and CT scans. More specifically, the proposed model
consists of a 3D encoder that learns to extract volume features at different
scales; features taken at different points of the encoder hierarchy are then
sent to multiple 3D decoders that individually predict intermediate
segmentation maps. Finally, all segmentation maps are combined to obtain a
unique detailed segmentation mask. We test our model on both CT and MRI imaging
data: the publicly available NIH Pancreas-CT dataset (consisting of 82
contrast-enhanced CTs) and a private MRI dataset (consisting of 40 MRI scans).
Experimental results show that our model outperforms existing methods on CT
pancreas segmentation, obtaining an average Dice score of about 88%, and yields
promising segmentation performance on a very challenging MRI data set (average
Dice score is about 77%). Additional control experiments demonstrate that the
achieved performance is due to the combination of our 3D fully-convolutional
deep network and the hierarchical representation decoding, thus substantiating
our architectural design.

    

### [[2109.01668] How Reliable Are Out-of-Distribution Generalization Methods for Medical Image Segmentation?](http://arxiv.org/abs/2109.01668)


  The recent achievements of Deep Learning rely on the test data being similar
in distribution to the training data. In an ideal case, Deep Learning models
would achieve Out-of-Distribution (OoD) Generalization, i.e. reliably make
predictions on out-of-distribution data. Yet in practice, models usually fail
to generalize well when facing a shift in distribution. Several methods were
thereby designed to improve the robustness of the features learned by a model
through Regularization- or Domain-Prediction-based schemes. Segmenting medical
images such as MRIs of the hippocampus is essential for the diagnosis and
treatment of neuropsychiatric disorders. But these brain images often suffer
from distribution shift due to the patient's age and various pathologies
affecting the shape of the organ. In this work, we evaluate OoD Generalization
solutions for the problem of hippocampus segmentation in MR data using both
fully- and semi-supervised training. We find that no method performs reliably
in all experiments. Only the V-REx loss stands out as it remains easy to tune,
while it outperforms a standard U-Net in most cases.

    

### [[2109.01669] Multimodal Detection of COVID-19 Symptoms using Deep Learning & Probability-based Weighting of Modes](http://arxiv.org/abs/2109.01669)


  The COVID-19 pandemic is one of the most challenging healthcare crises during
the 21st century. As the virus continues to spread on a global scale, the
majority of efforts have been on the development of vaccines and the mass
immunization of the public. While the daily case numbers were following a
decreasing trend, the emergent of new virus mutations and variants still pose a
significant threat. As economies start recovering and societies start opening
up with people going back into office buildings, schools, and malls, we still
need to have the ability to detect and minimize the spread of COVID-19.
Individuals with COVID-19 may show multiple symptoms such as cough, fever, and
shortness of breath. Many of the existing detection techniques focus on
symptoms having the same equal importance. However, it has been shown that some
symptoms are more prevalent than others. In this paper, we present a multimodal
method to predict COVID-19 by incorporating existing deep learning classifiers
using convolutional neural networks and our novel probability-based weighting
function that considers the prevalence of each symptom. The experiments were
performed on an existing dataset with respect to the three considered modes of
coughs, fever, and shortness of breath. The results show considerable
improvements in the detection of COVID-19 using our weighting function when
compared to an equal weighting function.

    

### [[2109.01690] High-quality Thermal Gibbs Sampling with Quantum Annealing Hardware](http://arxiv.org/abs/2109.01690)


  Quantum Annealing (QA) was originally intended for accelerating the solution
of combinatorial optimization tasks that have natural encodings as Ising
models. However, recent experiments on QA hardware platforms have demonstrated
that, in the operating regime corresponding to weak interactions, the QA
hardware behaves like a noisy Gibbs sampler at a hardware-specific effective
temperature. This work builds on those insights and identifies a class of small
hardware-native Ising models that are robust to noise effects and proposes a
novel procedure for executing these models on QA hardware to maximize Gibbs
sampling performance. Experimental results indicate that the proposed protocol
results in high-quality Gibbs samples from a hardware-specific effective
temperature and that the QA annealing time can be used to adjust the effective
temperature of the output distribution. The procedure proposed in this work
provides a new approach to using QA hardware for Ising model sampling
presenting potential new opportunities for applications in machine learning and
physics simulation.

    

### [[2109.01691] ALLWAS: Active Learning on Language models in WASserstein space](http://arxiv.org/abs/2109.01691)


  Active learning has emerged as a standard paradigm in areas with scarcity of
labeled training data, such as in the medical domain. Language models have
emerged as the prevalent choice of several natural language tasks due to the
performance boost offered by these models. However, in several domains, such as
medicine, the scarcity of labeled training data is a common issue. Also, these
models may not work well in cases where class imbalance is prevalent. Active
learning may prove helpful in these cases to boost the performance with a
limited label budget. To this end, we propose a novel method using sampling
techniques based on submodular optimization and optimal transport for active
learning in language models, dubbed ALLWAS. We construct a sampling strategy
based on submodular optimization of the designed objective in the gradient
domain. Furthermore, to enable learning from few samples, we propose a novel
strategy for sampling from the Wasserstein barycenters. Our empirical
evaluations on standard benchmark datasets for text classification show that
our methods perform significantly better (>20% relative increase in some cases)
than existing approaches for active learning on language models.

    

### [[2109.01693] Weakly Supervised Few-Shot Segmentation Via Meta-Learning](http://arxiv.org/abs/2109.01693)


  Semantic segmentation is a classic computer vision task with multiple
applications, which includes medical and remote sensing image analysis. Despite
recent advances with deep-based approaches, labeling samples (pixels) for
training models is laborious and, in some cases, unfeasible. In this paper, we
present two novel meta learning methods, named WeaSeL and ProtoSeg, for the
few-shot semantic segmentation task with sparse annotations. We conducted
extensive evaluation of the proposed methods in different applications (12
datasets) in medical imaging and agricultural remote sensing, which are very
distinct fields of knowledge and usually subject to data scarcity. The results
demonstrated the potential of our method, achieving suitable results for
segmenting both coffee/orange crops and anatomical parts of the human body in
comparison with full dense annotation.

    

### [[2109.01696] Revisiting 3D ResNets for Video Recognition](http://arxiv.org/abs/2109.01696)


  A recent work from Bello shows that training and scaling strategies may be
more significant than model architectures for visual recognition. This short
note studies effective training and scaling strategies for video recognition
models. We propose a simple scaling strategy for 3D ResNets, in combination
with improved training strategies and minor architectural changes. The
resulting models, termed 3D ResNet-RS, attain competitive performance of 81.0
on Kinetics-400 and 83.8 on Kinetics-600 without pre-training. When pre-trained
on a large Web Video Text dataset, our best model achieves 83.5 and 84.3 on
Kinetics-400 and Kinetics-600. The proposed scaling rule is further evaluated
in a self-supervised setup using contrastive learning, demonstrating improved
performance. Code is available at:
this https URL.

    

### [[2109.01718] Communication Efficient Tensor Factorization for Decentralized Healthcare Networks](http://arxiv.org/abs/2109.01718)


  Tensor factorization has been proved as an efficient unsupervised learning
approach for health data analysis, especially for computational phenotyping,
where the high-dimensional Electronic Health Records (EHRs) with patients
history of medical procedures, medications, diagnosis, lab tests, etc., are
converted to meaningful and interpretable medical concepts. Federated tensor
factorization distributes the tensor computation to multiple workers under the
coordination of a central server, which enables jointly learning the phenotypes
across multiple hospitals while preserving the privacy of the patient
information. However, existing federated tensor factorization algorithms
encounter the single-point-failure issue with the involvement of the central
server, which is not only easily exposed to external attacks, but also limits
the number of clients sharing information with the server under restricted
uplink bandwidth. In this paper, we propose CiderTF, a communication-efficient
decentralized generalized tensor factorization, which reduces the uplink
communication cost by leveraging a four-level communication reduction strategy
designed for a generalized tensor factorization, which has the flexibility of
modeling different tensor distribution with multiple kinds of loss functions.
Experiments on two real-world EHR datasets demonstrate that CiderTF achieves
comparable convergence with the communication reduction up to 99.99%.

    

### [[2109.01730] Nonasymptotic one-and two-sample tests in high dimension with unknown covariance structure](http://arxiv.org/abs/2109.01730)


  Let $\mathbf{X} = (X_i)_{1\leq i \leq n}$ be an i.i.d. sample of
square-integrable variables in $\mathbb{R}^d$, with common expectation $\mu$
and covariance matrix $\Sigma$, both unknown. We consider the problem of
testing if $\mu$ is $\eta$-close to zero, i.e. $\|\mu\| \leq \eta $ against
$\|\mu\| \geq (\eta + \delta)$; we also tackle the more general two-sample mean
closeness testing problem. The aim of this paper is to obtain nonasymptotic
upper and lower bounds on the minimal separation distance $\delta$ such that we
can control both the Type I and Type II errors at a given level. The main
technical tools are concentration inequalities, first for a suitable estimator
of $\|\mu\|^2$ used a test statistic, and secondly for estimating the operator
and Frobenius norms of $\Sigma$ coming into the quantiles of said test
statistic. These properties are obtained for Gaussian and bounded
distributions. A particular attention is given to the dependence in the
pseudo-dimension $d_*$ of the distribution, defined as $d_* :=
\|\Sigma\|_2^2/\|\Sigma\|_\infty^2$. In particular, for $\eta=0$, the minimum
separation distance is ${\Theta}(d_*^{\frac{1}{4}}\sqrt{\|\Sigma\|_\infty/n})$,
in contrast with the minimax estimation distance for $\mu$, which is
${\Theta}(d_e^{\frac{1}{2}}\sqrt{\|\Sigma\|_\infty/n})$ (where
$d_e:=\|\Sigma\|_1/\|\Sigma\|_\infty$). This generalizes a phenomenon spelled
out in particular by Baraud (2002).

    

### [[2109.01731] Acceleration Method for Learning Fine-Layered Optical Neural Networks](http://arxiv.org/abs/2109.01731)


  An optical neural network (ONN) is a promising system due to its high-speed
and low-power operation. Its linear unit performs a multiplication of an input
vector and a weight matrix in optical analog circuits. Among them, a circuit
with a multiple-layered structure of programmable Mach-Zehnder interferometers
(MZIs) can realize a specific class of unitary matrices with a limited number
of MZIs as its weight matrix. The circuit is effective for balancing the number
of programmable MZIs and ONN performance. However, it takes a lot of time to
learn MZI parameters of the circuit with a conventional automatic
differentiation (AD), which machine learning platforms are equipped with. To
solve the time-consuming problem, we propose an acceleration method for
learning MZI parameters. We create customized complex-valued derivatives for an
MZI, exploiting Wirtinger derivatives and a chain rule. They are incorporated
into our newly developed function module implemented in C++ to collectively
calculate their values in a multi-layered structure. Our method is simple,
fast, and versatile as well as compatible with the conventional AD. We
demonstrate that our method works 20 times faster than the conventional AD when
a pixel-by-pixel MNIST task is performed in a complex-valued recurrent neural
network with an MZI-based hidden unit.

    

### [[2109.01739] Cohort Characteristics and Factors Associated with Cannabis Use among Adolescents in Canada Using Pattern Discovery and Disentanglement Method](http://arxiv.org/abs/2109.01739)


  COMPASS is a longitudinal, prospective cohort study collecting data annually
from students attending high school in jurisdictions across Canada. We aimed to
discover significant frequent/rare associations of behavioral factors among
Canadian adolescents related to cannabis use. We use a subset of COMPASS
dataset which contains 18,761 records of students in grades 9 to 12 with 31
selected features (attributes) involving various characteristics, from living
habits to academic performance. We then used the Pattern Discovery and
Disentanglement (PDD) algorithm that we have developed to detect strong and
rare (yet statistically significant) associations from the dataset. PDD used
the criteria derived from disentangled statistical spaces (known as
Re-projected Adjusted-Standardized Residual Vector Spaces, notated as RARV). It
outperformed methods using other criteria (i.e. support and confidence) popular
as reported in the literature. Association results showed that PDD can
discover: i) a smaller set of succinct significant associations in clusters;
ii) frequent and rare, yet significant, patterns supported by population health
relevant study; iii) patterns from a dataset with extremely imbalanced groups
(majority class: minority class = 88.3%: 11.7%).

    

### [[2109.01745] A realistic approach to generate masked faces applied on two novel masked face recognition data sets](http://arxiv.org/abs/2109.01745)


  The COVID-19 pandemic raises the problem of adapting face recognition systems
to the new reality, where people may wear surgical masks to cover their noses
and mouths. Traditional data sets (e.g., CelebA, CASIA-WebFace) used for
training these systems were released before the pandemic, so they now seem
unsuited due to the lack of examples of people wearing masks. We propose a
method for enhancing data sets containing faces without masks by creating
synthetic masks and overlaying them on faces in the original images. Our method
relies on Spark AR Studio, a developer program made by Facebook that is used to
create Instagram face filters. In our approach, we use 9 masks of different
colors, shapes and fabrics. We employ our method to generate a number of
445,446 (90%) samples of masks for the CASIA-WebFace data set and 196,254
(96.8%) masks for the CelebA data set, releasing the mask images at
this https URL. We show that our method produces
significantly more realistic training examples of masks overlaid on faces by
asking volunteers to qualitatively compare it to other methods or data sets
designed for the same task. We also demonstrate the usefulness of our method by
evaluating state-of-the-art face recognition systems (FaceNet, VGG-face,
ArcFace) trained on the enhanced data sets and showing that they outperform
equivalent systems trained on the original data sets (containing faces without
masks), when the test benchmark contains masked faces.

    

### [[2109.01750] CodeNeRF: Disentangled Neural Radiance Fields for Object Categories](http://arxiv.org/abs/2109.01750)


  CodeNeRF is an implicit 3D neural representation that learns the variation of
object shapes and textures across a category and can be trained, from a set of
posed images, to synthesize novel views of unseen objects. Unlike the original
NeRF, which is scene specific, CodeNeRF learns to disentangle shape and texture
by learning separate embeddings. At test time, given a single unposed image of
an unseen object, CodeNeRF jointly estimates camera viewpoint, and shape and
appearance codes via optimization. Unseen objects can be reconstructed from a
single image, and then rendered from new viewpoints or their shape and texture
edited by varying the latent codes. We conduct experiments on the SRN
benchmark, which show that CodeNeRF generalises well to unseen objects and
achieves on-par performance with methods that require known camera pose at test
time. Our results on real-world images demonstrate that CodeNeRF can bridge the
sim-to-real gap. Project page: \url{this https URL}

    

### [[2109.01753] Assessing the Knowledge State of Online Students -- New Data, New Approaches, Improved Accuracy](http://arxiv.org/abs/2109.01753)


  We consider the problem of assessing the changing knowledge state of
individual students as they go through online courses. This student performance
(SP) modeling problem, also known as knowledge tracing, is a critical step for
building adaptive online teaching systems. Specifically, we conduct a study of
how to utilize various types and large amounts of students log data to train
accurate machine learning models that predict the knowledge state of future
students. This study is the first to use four very large datasets made
available recently from four distinct intelligent tutoring systems. Our results
include a new machine learning approach that defines a new state of the art for
SP modeling, improving over earlier methods in several ways: First, we achieve
improved accuracy by introducing new features that can be easily computed from
conventional question-response logs (e.g., the pattern in the student's most
recent answers). Second, we take advantage of features of the student history
that go beyond question-response pairs (e.g., which video segments the student
watched, or skipped) as well as information about prerequisite structure in the
curriculum. Third, we train multiple specialized modeling models for different
aspects of the curriculum (e.g., specializing in early versus later segments of
the student history), then combine these specialized models to create a group
prediction of student knowledge. Taken together, these innovations yield an
average AUC score across these four datasets of 0.807 compared to the previous
best logistic regression approach score of 0.766, and also outperforming
state-of-the-art deep neural net approaches. Importantly, we observe consistent
improvements from each of our three methodological innovations, in each
dataset, suggesting that our methods are of general utility and likely to
produce improvements for other online tutoring systems as well.

    

### [[2109.01754] Error Detection in Large-Scale Natural Language Understanding Systems Using Transformer Models](http://arxiv.org/abs/2109.01754)


  Large-scale conversational assistants like Alexa, Siri, Cortana and Google
Assistant process every utterance using multiple models for domain, intent and
named entity recognition. Given the decoupled nature of model development and
large traffic volumes, it is extremely difficult to identify utterances
processed erroneously by such systems. We address this challenge to detect
domain classification errors using offline Transformer models. We combine
utterance encodings from a RoBERTa model with the Nbest hypothesis produced by
the production system. We then fine-tune end-to-end in a multitask setting
using a small dataset of humanannotated utterances with domain classification
errors. We tested our approach for detecting misclassifications from one domain
that accounts for <0.5% of the traffic in a large-scale conversational AI
system. Our approach achieves an F1 score of 30% outperforming a bi- LSTM
baseline by 16.9% and a standalone RoBERTa model by 4.8%. We improve this
further by 2.2% to 32.2% by ensembling multiple models.

    

### [[2109.01761] An empirical evaluation of attention-based multi-head models for improved turbofan engine remaining useful life prediction](http://arxiv.org/abs/2109.01761)


  A single unit (head) is the conventional input feature extractor in deep
learning architectures trained on multivariate time series signals. The
importance of the fixed-dimensional vector representation generated by the
single-head network has been demonstrated for industrial machinery condition
monitoring and predictive maintenance. However, processing heterogeneous sensor
signals with a single head may result in a model that cannot explicitly account
for the diversity in time-varying multivariate inputs. This work extends the
conventional single-head deep learning models to a more robust form by
developing context-specific heads to independently capture the inherent pattern
of each sensor reading in multivariate time series signals. Using the turbofan
aircraft engine benchmark dataset (CMAPSS), an extensive experiment is
performed to verify the effectiveness and benefits of multi-head fully
connected neurons, recurrent networks, convolution network, the
transformer-style stand-alone attention network, and their variants for
remaining useful life estimation. Moreover, the effect of different attention
mechanisms on the multi-head models is also evaluated. In addition, each
architecture's relative advantage and computational overhead are analyzed.
Results show that utilizing the attention layer is task-sensitive and
model-dependent, as it does not provide consistent improvement across the
models investigated. The result is further compared with five state-of-the-art
models, and the comparison shows that a relatively simple multi-head
architecture performs better than the state-of-the-art models. The results
presented in this study demonstrate the importance of multi-head models and
attention mechanisms to improved understanding of the remaining useful life of
industrial assets.

    

### [[2109.01766] SEC4SR: A Security Analysis Platform for Speaker Recognition](http://arxiv.org/abs/2109.01766)


  Adversarial attacks have been expanded to speaker recognition (SR). However,
existing attacks are often assessed using different SR models, recognition
tasks and datasets, and only few adversarial defenses borrowed from computer
vision are considered. Yet,these defenses have not been thoroughly evaluated
against adaptive attacks. Thus, there is still a lack of quantitative
understanding about the strengths and limitations of adversarial attacks and
defenses. More effective defenses are also required for securing SR systems. To
bridge this gap, we present SEC4SR, the first platform enabling researchers to
systematically and comprehensively evaluate adversarial attacks and defenses in
SR. SEC4SR incorporates 4 white-box and 2 black-box attacks, 24 defenses
including our novel feature-level transformations. It also contains techniques
for mounting adaptive attacks. Using SEC4SR, we conduct thus far the
largest-scale empirical study on adversarial attacks and defenses in SR,
involving 23 defenses, 15 attacks and 4 attack settings. Our study provides
lots of useful findings that may advance future research: such as (1) all the
transformations slightly degrade accuracy on benign examples and their
effectiveness vary with attacks; (2) most transformations become less effective
under adaptive attacks, but some transformations become more effective; (3) few
transformations combined with adversarial training yield stronger defenses over
some but not all attacks, while our feature-level transformation combined with
adversarial training yields the strongest defense over all the attacks.
Extensive experiments demonstrate capabilities and advantages of SEC4SR which
can benefit future research in SR.

    

### [[2109.01768] Eden: A Unified Environment Framework for Booming Reinforcement Learning Algorithms](http://arxiv.org/abs/2109.01768)


  With AlphaGo defeats top human players, reinforcement learning(RL) algorithms
have gradually become the code-base of building stronger artificial
intelligence(AI). The RL algorithm design firstly needs to adapt to the
specific environment, so the designed environment guides the rapid and profound
development of RL algorithms. However, the existing environments, which can be
divided into real world games and customized toy environments, have obvious
shortcomings. For real world games, it is designed for human entertainment, and
too much difficult for most of RL researchers. For customized toy environments,
there is no widely accepted unified evaluation standard for all RL algorithms.
Therefore, we introduce the first virtual user-friendly environment framework
for RL. In this framework, the environment can be easily configured to realize
all kinds of RL tasks in the mainstream research. Then all the mainstream
state-of-the-art(SOTA) RL algorithms can be conveniently evaluated and
compared. Therefore, our contributions mainly includes the following aspects:
1.single configured environment for all classification of SOTA RL algorithms;
2.combined environment of more than one classification RL algorithms; 3.the
evaluation standard for all kinds of RL algorithms. With all these efforts, a
possibility for breeding an AI with capability of general competency in a
variety of tasks is provided, and maybe it will open up a new chapter for AI.

    

### [[2109.01773] MLCTR: A Fast Scalable Coupled Tensor Completion Based on Multi-Layer Non-Linear Matrix Factorization](http://arxiv.org/abs/2109.01773)


  Firms earning prediction plays a vital role in investment decisions,
dividends expectation, and share price. It often involves multiple
tensor-compatible datasets with non-linear multi-way relationships,
spatiotemporal structures, and different levels of sparsity. Current non-linear
tensor completion algorithms tend to learn noisy embedding and incur
overfitting. This paper focuses on the embedding learning aspect of the tensor
completion problem and proposes a new multi-layer neural network architecture
for tensor factorization and completion (MLCTR). The network architecture
entails multiple advantages: a series of low-rank matrix factorizations (MF)
building blocks to minimize overfitting, interleaved transfer functions in each
layer for non-linearity, and by-pass connections to reduce the gradient
diminishing problem and increase the depths of neural networks. Furthermore,
the model employs Stochastic Gradient Descent(SGD) based optimization for fast
convergence in training. Our algorithm is highly efficient for imputing missing
values in the EPS data. Experiments confirm that our strategy of incorporating
non-linearity in factor matrices demonstrates impressive performance in
embedding learning and end-to-end tensor models, and outperforms approaches
with non-linearity in the phase of reconstructing tensors from factor matrices.

    

### [[2109.01784] Model retraining and information sharing in a supply chain with long-term fluctuating demands](http://arxiv.org/abs/2109.01784)


  Demand forecasting based on empirical data is a viable approach for
optimizing a supply chain. However, in this approach, a model constructed from
past data occasionally becomes outdated due to long-term changes in the
environment, in which case the model should be updated (i.e., retrained) using
the latest data. In this study, we examine the effects of updating models in a
supply chain using a minimal setting. We demonstrate that when each party in
the supply chain has its own forecasting model, uncoordinated model retraining
causes the bullwhip effect even if a very simple replenishment policy is
applied. Our results also indicate that sharing the forecasting model among the
parties involved significantly reduces the bullwhip effect.

    

### [[2109.01785] Node Feature Kernels Increase Graph Convolutional Network Robustness](http://arxiv.org/abs/2109.01785)


  The robustness of the much-used Graph Convolutional Networks (GCNs) to
perturbations of their input is becoming a topic of increasing importance. In
this paper, the random GCN is introduced for which a random matrix theory
analysis is possible. This analysis suggests that if the graph is sufficiently
perturbed, or in the extreme case random, then the GCN fails to benefit from
the node features. It is furthermore observed that enhancing the message
passing step in GCNs by adding the node feature kernel to the adjacency matrix
of the graph structure solves this problem. An empirical study of a GCN
utilised for node classification on six real datasets further confirms the
theoretical findings and demonstrates that perturbations of the graph structure
can result in GCNs performing significantly worse than Multi-Layer Perceptrons
run on the node features alone. In practice, adding a node feature kernel to
the message passing of perturbed graphs results in a significant improvement of
the GCN's performance, thereby rendering it more robust to graph perturbations.
Our code is publicly available at:this https URL.

    

### [[2109.01795] On the Complexity of Computing Markov Perfect Equilibrium in General-Sum Stochastic Games](http://arxiv.org/abs/2109.01795)


  Similar to the role of Markov decision processes in reinforcement learning,
Stochastic Games (SGs) lay the foundation for the study of multi-agent
reinforcement learning (MARL) and sequential agent interactions. In this paper,
we derive that computing an approximate Markov Perfect Equilibrium (MPE) in a
finite-state discounted Stochastic Game within the exponential precision is
\textbf{PPAD}-complete. We adopt a function with a polynomially bounded
description in the strategy space to convert the MPE computation to a
fixed-point problem, even though the stochastic game may demand an exponential
number of pure strategies, in the number of states, for each agent. The
completeness result follows the reduction of the fixed-point problem to {\sc
End of the Line}. Our results indicate that finding an MPE in SGs is highly
unlikely to be \textbf{NP}-hard unless \textbf{NP}=\textbf{co-NP}. Our work
offers confidence for MARL research to study MPE computation on general-sum SGs
and to develop fruitful algorithms as currently on zero-sum SGs.

    

### [[2109.01806] On Faster Convergence of Scaled Sign Gradient Descent](http://arxiv.org/abs/2109.01806)


  Communication has been seen as a significant bottleneck in industrial
applications over large-scale networks. To alleviate the communication burden,
sign-based optimization algorithms have gained popularity recently in both
industrial and academic communities, which is shown to be closely related to
adaptive gradient methods, such as Adam. Along this line, this paper
investigates faster convergence for a variant of sign-based gradient descent,
called scaled signGD, in three cases: 1) the objective function is strongly
convex; 2) the objective function is nonconvex but satisfies the
Polyak-Lojasiewicz (PL) inequality; 3) the gradient is stochastic, called
scaled signGD in this case. For the first two cases, it can be shown that the
scaled signGD converges at a linear rate. For case 3), the algorithm is shown
to converge linearly to a neighborhood of the optimal value when a constant
learning rate is employed, and the algorithm converges at a rate of $O(1/k)$
when using a diminishing learning rate, where $k$ is the iteration number. The
results are also extended to the distributed setting by majority vote in a
parameter-server framework. Finally, numerical experiments on logistic
regression are performed to corroborate the theoretical findings.

    

### [[2109.01815] Representation Learning for Efficient and Effective Similarity Search and Recommendation](http://arxiv.org/abs/2109.01815)


  How data is represented and operationalized is critical for building
computational solutions that are both effective and efficient. A common
approach is to represent data objects as binary vectors, denoted \textit{hash
codes}, which require little storage and enable efficient similarity search
through direct indexing into a hash table or through similarity computations in
an appropriate space. Due to the limited expressibility of hash codes, compared
to real-valued representations, a core open challenge is how to generate hash
codes that well capture semantic content or latent properties using a small
number of bits, while ensuring that the hash codes are distributed in a way
that does not reduce their search efficiency. State of the art methods use
representation learning for generating such hash codes, focusing on neural
autoencoder architectures where semantics are encoded into the hash codes by
learning to reconstruct the original inputs of the hash codes. This thesis
addresses the above challenge and makes a number of contributions to
representation learning that (i) improve effectiveness of hash codes through
more expressive representations and a more effective similarity measure than
the current state of the art, namely the Hamming distance, and (ii) improve
efficiency of hash codes by learning representations that are especially suited
to the choice of search method. The contributions are empirically validated on
several tasks related to similarity search and recommendation.

    

### [[2109.01818] Estimating permeability of 3D micro-CT images by physics-informed CNNs based on DNS](http://arxiv.org/abs/2109.01818)


  In recent years, convolutional neural networks (CNNs) have experienced an
increasing interest for their ability to perform fast approximation of
effective hydrodynamic parameters in porous media research and applications.
This paper presents a novel methodology for permeability prediction from
micro-CT scans of geological rock samples. The training data set for CNNs
dedicated to permeability prediction consists of permeability labels that are
typically generated by classical lattice Boltzmann methods (LBM) that simulate
the flow through the pore space of the segmented image data. We instead perform
direct numerical simulation (DNS) by solving the stationary Stokes equation in
an efficient and distributed-parallel manner. As such, we circumvent the
convergence issues of LBM that frequently are observed on complex pore
geometries, and therefore, improve on the generality and accuracy of our
training data set. Using the DNS-computed permeabilities, a physics-informed
CNN PhyCNN) is trained by additionally providing a tailored characteristic
quantity of the pore space. More precisely, by exploiting the connection to
flow problems on a graph representation of the pore space, additional
information about confined structures is provided to the network in terms of
the maximum flow value, which is the key innovative component of our workflow.
As a result, unprecedented prediction accuracy and robustness are observed for
a variety of sandstone samples from archetypal rock formations.

    

### [[2109.01819] Frustratingly Simple Pretraining Alternatives to Masked Language Modeling](http://arxiv.org/abs/2109.01819)


  Masked language modeling (MLM), a self-supervised pretraining objective, is
widely used in natural language processing for learning text representations.
MLM trains a model to predict a random sample of input tokens that have been
replaced by a [MASK] placeholder in a multi-class setting over the entire
vocabulary. When pretraining, it is common to use alongside MLM other auxiliary
objectives on the token or sequence level to improve downstream performance
(e.g. next sentence prediction). However, no previous work so far has attempted
in examining whether other simpler linguistically intuitive or not objectives
can be used standalone as main pretraining objectives. In this paper, we
explore five simple pretraining objectives based on token-level classification
tasks as replacements of MLM. Empirical results on GLUE and SQuAD show that our
proposed methods achieve comparable or better performance to MLM using a
BERT-BASE architecture. We further validate our methods using smaller models,
showing that pretraining a model with 41% of the BERT-BASE's parameters,
BERT-MEDIUM results in only a 1% drop in GLUE scores with our best objective.

    

### [[2109.01824] Multi-View Spatial-Temporal Graph Convolutional Networks with Domain Generalization for Sleep Stage Classification](http://arxiv.org/abs/2109.01824)


  Sleep stage classification is essential for sleep assessment and disease
diagnosis. Although previous attempts to classify sleep stages have achieved
high classification performance, several challenges remain open: 1) How to
effectively utilize time-varying spatial and temporal features from
multi-channel brain signals remains challenging. Prior works have not been able
to fully utilize the spatial topological information among brain regions. 2)
Due to the many differences found in individual biological signals, how to
overcome the differences of subjects and improve the generalization of deep
neural networks is important. 3) Most deep learning methods ignore the
interpretability of the model to the brain. To address the above challenges, we
propose a multi-view spatial-temporal graph convolutional networks (MSTGCN)
with domain generalization for sleep stage classification. Specifically, we
construct two brain view graphs for MSTGCN based on the functional connectivity
and physical distance proximity of the brain regions. The MSTGCN consists of
graph convolutions for extracting spatial features and temporal convolutions
for capturing the transition rules among sleep stages. In addition, attention
mechanism is employed for capturing the most relevant spatial-temporal
information for sleep stage classification. Finally, domain generalization and
MSTGCN are integrated into a unified framework to extract subject-invariant
sleep features. Experiments on two public datasets demonstrate that the
proposed model outperforms the state-of-the-art baselines.

    

### [[2109.01838] RAMA: A Rapid Multicut Algorithm on GPU](http://arxiv.org/abs/2109.01838)


  We propose a highly parallel primal-dual algorithm for the multicut (a.k.a.
correlation clustering) problem, a classical graph clustering problem widely
used in machine learning and computer vision. Our algorithm consists of three
steps executed recursively: (1) Finding conflicted cycles that correspond to
violated inequalities of the underlying multicut relaxation, (2) Performing
message passing between the edges and cycles to optimize the Lagrange
relaxation coming from the found violated cycles producing reduced costs and
(3) Contracting edges with high reduced costs through matrix-matrix
multiplications.
Our algorithm produces primal solutions and dual lower bounds that estimate
the distance to optimum. We implement our algorithm on GPUs and show resulting
one to two order-of-magnitudes improvements in execution speed without
sacrificing solution quality compared to traditional serial algorithms that run
on CPUs. We can solve very large scale benchmark problems with up to
$\mathcal{O}(10^8)$ variables in a few seconds with small primal-dual gaps. We
make our code available at this https URL.

    

### [[2109.01844] On robustness of generative representations against catastrophic forgetting](http://arxiv.org/abs/2109.01844)


  Catastrophic forgetting of previously learned knowledge while learning new
tasks is a widely observed limitation of contemporary neural networks. Although
many continual learning methods are proposed to mitigate this drawback, the
main question remains unanswered: what is the root cause of catastrophic
forgetting? In this work, we aim at answering this question by posing and
validating a set of research hypotheses related to the specificity of
representations built internally by neural models. More specifically, we design
a set of empirical evaluations that compare the robustness of representations
in discriminative and generative models against catastrophic forgetting. We
observe that representations learned by discriminative models are more prone to
catastrophic forgetting than their generative counterparts, which sheds new
light on the advantages of developing generative models for continual learning.
Finally, our work opens new research pathways and possibilities to adopt
generative models in continual learning beyond mere replay mechanisms.

    

### [[2109.01863] Customer 360-degree Insights in Predicting Chronic Diabetes](http://arxiv.org/abs/2109.01863)


  Chronic diseases such as diabetes are quite prevalent in the world and are
responsible for a significant number of deaths per year. In addition,
treatments for such chronic diseases account for a high healthcare cost.
However, research has shown that diabetes can be proactively managed and
prevented while lowering these healthcare costs. We have mined a sample of ten
million customers' 360-degree data representing the state of Texas, USA, with
attributes current as of late 2018. The sample received from a market research
data vendor has over 1000 customer attributes consisting of demography,
lifestyle, and in some cases self-reported chronic conditions. In this study,
we have developed a classification model to predict chronic diabetes with an
accuracy of 80%. We demonstrate a use case where a large volume of 360-degree
customer data can be useful to predict and hence proactively prevent chronic
diseases such as diabetes.

    

### [[2109.01876] Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting](http://arxiv.org/abs/2109.01876)


  Neural networks inspired by differential equations have proliferated for the
past several years. Neural ordinary differential equations (NODEs) and neural
controlled differential equations (NCDEs) are two representative examples of
them. In theory, NCDEs provide better representation learning capability for
time-series data than NODEs. In particular, it is known that NCDEs are suitable
for processing irregular time-series data. Whereas NODEs have been successfully
extended after adopting attention, however, it had not been studied yet how to
integrate attention into NCDEs. To this end, we present the method of Attentive
Neural Controlled Differential Equations (ANCDEs) for time-series
classification and forecasting, where dual NCDEs are used: one for generating
attention values, and the other for evolving hidden vectors for a downstream
machine learning task. We conduct experiments with three real-world time-series
datasets and 10 baselines. After dropping some values, we also conduct
irregular time-series experiments. Our method consistently shows the best
accuracy in all cases by non-trivial margins. Our visualizations also show that
the presented attention mechanism works as intended by focusing on crucial
information.

    

### [[2109.01880] Deep learning facilitates fully automated brain image registration of optoacoustic tomography and magnetic resonance imaging](http://arxiv.org/abs/2109.01880)


  Multi-spectral optoacoustic tomography (MSOT) is an emerging optical imaging
method providing multiplex molecular and functional information from the rodent
brain. It can be greatly augmented by magnetic resonance imaging (MRI) that
offers excellent soft-tissue contrast and high-resolution brain anatomy.
Nevertheless, registration of multi-modal images remains challenging, chiefly
due to the entirely different image contrast rendered by these modalities.
Previously reported registration algorithms mostly relied on manual
user-dependent brain segmentation, which compromised data interpretation and
accurate quantification. Here we propose a fully automated registration method
for MSOT-MRI multimodal imaging empowered by deep learning. The automated
workflow includes neural network-based image segmentation to generate suitable
masks, which are subsequently registered using an additional neural network.
Performance of the algorithm is showcased with datasets acquired by
cross-sectional MSOT and high-field MRI preclinical scanners. The automated
registration method is further validated with manual and half-automated
registration, demonstrating its robustness and accuracy.

    

### [[2109.01887] Weakly supervised semantic segmentation of tomographic images in the diagnosis of stroke](http://arxiv.org/abs/2109.01887)


  This paper presents an automatic algorithm for the segmentation of areas
affected by an acute stroke on the non-contrast computed tomography brain
images. The proposed algorithm is designed for learning in a weakly supervised
scenario when some images are labeled accurately, and some images are labeled
inaccurately. Wrong labels appear as a result of inaccuracy made by a
radiologist in the process of manual annotation of computed tomography images.
We propose methods for solving the segmentation problem in the case of
inaccurately labeled training data. We use the U-Net neural network
architecture with several modifications. Experiments on real computed
tomography scans show that the proposed methods increase the segmentation
accuracy.

    

### [[2109.01900] Uncovering the Limits of Text-based Emotion Detection](http://arxiv.org/abs/2109.01900)


  Identifying emotions from text is crucial for a variety of real world tasks.
We consider the two largest now-available corpora for emotion classification:
GoEmotions, with 58k messages labelled by readers, and Vent, with 33M
writer-labelled messages. We design a benchmark and evaluate several feature
spaces and learning algorithms, including two simple yet novel models on top of
BERT that outperform previous strong baselines on GoEmotions. Through an
experiment with human participants, we also analyze the differences between how
writers express emotions and how readers perceive them. Our results suggest
that emotions expressed by writers are harder to identify than emotions that
readers perceive. We share a public web interface for researchers to explore
our models.

    

### [[2109.01902] Barycenteric distribution alignment and manifold-restricted invertibility for domain generalization](http://arxiv.org/abs/2109.01902)


  For the Domain Generalization (DG) problem where the hypotheses are composed
of a common representation function followed by a labeling function, we point
out a shortcoming in existing approaches that fail to explicitly optimize for a
term, appearing in a well-known and widely adopted upper bound to the risk on
the unseen domain, that is dependent on the representation to be learned. To
this end, we first derive a novel upper bound to the prediction risk. We show
that imposing a mild assumption on the representation to be learned, namely
manifold restricted invertibility, is sufficient to deal with this issue.
Further, unlike existing approaches, our novel upper bound doesn't require the
assumption of Lipschitzness of the loss function. In addition, the
distributional discrepancy in the representation space is handled via the
Wasserstein-2 barycenter cost. In this context, we creatively leverage old and
recent transport inequalities, which link various optimal transport metrics, in
particular the $L^1$ distance (also known as the total variation distance) and
the Wasserstein-2 distances, with the Kullback-Liebler divergence. These
analyses and insights motivate a new representation learning cost for DG that
additively balances three competing objectives: 1) minimizing classification
error across seen domains via cross-entropy, 2) enforcing domain-invariance in
the representation space via the Wasserstein-2 barycenter cost, and 3)
promoting non-degenerate, nearly-invertible representation via one of two
mechanisms, viz., an autoencoder-based reconstruction loss or a mutual
information loss. It is to be noted that the proposed algorithms completely
bypass the use of any adversarial training mechanism that is typical of many
current domain generalization approaches. Simulation results on several
standard datasets demonstrate superior performance compared to several
well-known DG algorithms.

    

### [[2109.01903] Robust fine-tuning of zero-shot models](http://arxiv.org/abs/2109.01903)


  Large pre-trained models such as CLIP offer consistent accuracy across a
range of data distributions when performing zero-shot inference (i.e., without
fine-tuning on a specific dataset). Although existing fine-tuning approaches
substantially improve accuracy in-distribution, they also reduce
out-of-distribution robustness. We address this tension by introducing a simple
and effective method for improving robustness: ensembling the weights of the
zero-shot and fine-tuned models. Compared to standard fine-tuning, the
resulting weight-space ensembles provide large accuracy improvements
out-of-distribution, while matching or improving in-distribution accuracy. On
ImageNet and five derived distribution shifts, weight-space ensembles improve
out-of-distribution accuracy by 2 to 10 percentage points while increasing
in-distribution accuracy by nearly 1 percentage point relative to standard
fine-tuning. These improvements come at no additional computational cost during
fine-tuning or inference.

    

### [[2109.01904] Estimating the probabilities of causation via deep monotonic twin networks](http://arxiv.org/abs/2109.01904)


  There has been much recent work using machine learning to answer causal
queries. Most focus on interventional queries, such as the conditional average
treatment effect. However, as noted by Pearl, interventional queries only form
part of a larger hierarchy of causal queries, with counterfactuals sitting at
the top. Despite this, our community has not fully succeeded in adapting
machine learning tools to answer counterfactual queries. This work addresses
this challenge by showing how to implement twin network counterfactual
inference -- an alternative to abduction, action, & prediction counterfactual
inference -- with deep learning to estimate counterfactual queries. We show how
the graphical nature of twin networks makes them particularly amenable to deep
learning, yielding simple neural network architectures that, when trained, are
capable of counterfactual inference. Importantly, we show how to enforce known
identifiability constraints during training, ensuring the answer to each
counterfactual query is uniquely determined. We demonstrate our approach by
using it to accurately estimate the probabilities of causation -- important
counterfactual queries that quantify the degree to which one event was a
necessary or sufficient cause of another -- on both synthetic and real data.

    

### [[2109.01918] Training Graph Neural Networks by Graphon Estimation](http://arxiv.org/abs/2109.01918)


  In this work, we propose to train a graph neural network via resampling from
a graphon estimate obtained from the underlying network data. More
specifically, the graphon or the link probability matrix of the underlying
network is first obtained from which a new network will be resampled and used
during the training process at each layer. Due to the uncertainty induced from
the resampling, it helps mitigate the well-known issue of over-smoothing in a
graph neural network (GNN) model. Our framework is general, computationally
efficient, and conceptually simple. Another appealing feature of our method is
that it requires minimal additional tuning during the training process.
Extensive numerical results show that our approach is competitive with and in
many cases outperform the other over-smoothing reducing GNN training methods.

    

### [[2109.01924] A Neural Network-Based Linguistic Similarity Measure for Entrainment in Conversations](http://arxiv.org/abs/2109.01924)


  Linguistic entrainment is a phenomenon where people tend to mimic each other
in conversation. The core instrument to quantify entrainment is a linguistic
similarity measure between conversational partners. Most of the current
similarity measures are based on bag-of-words approaches that rely on
linguistic markers, ignoring the overall language structure and dialogue
context. To address this issue, we propose to use a neural network model to
perform the similarity measure for entrainment. Our model is context-aware, and
it further leverages a novel component to learn the shared high-level
linguistic features across dialogues. We first investigate the effectiveness of
our novel component. Then we use the model to perform similarity measure in a
corpus-based entrainment analysis. We observe promising results for both
evaluation tasks.

    

### [[2109.01934] Weakly Supervised Relative Spatial Reasoning for Visual Question Answering](http://arxiv.org/abs/2109.01934)


  Vision-and-language (V\&L) reasoning necessitates perception of visual
concepts such as objects and actions, understanding semantics and language
grounding, and reasoning about the interplay between the two modalities. One
crucial aspect of visual reasoning is spatial understanding, which involves
understanding relative locations of objects, i.e.\ implicitly learning the
geometry of the scene. In this work, we evaluate the faithfulness of V\&L
models to such geometric understanding, by formulating the prediction of
pair-wise relative locations of objects as a classification as well as a
regression task. Our findings suggest that state-of-the-art transformer-based
V\&L models lack sufficient abilities to excel at this task. Motivated by this,
we design two objectives as proxies for 3D spatial reasoning (SR) -- object
centroid estimation, and relative position estimation, and train V\&L with weak
supervision from off-the-shelf depth estimators. This leads to considerable
improvements in accuracy for the "GQA" visual question answering challenge (in
fully supervised, few-shot, and O.O.D settings) as well as improvements in
relative spatial reasoning. Code and data will be released
\href{this https URL}{here}.

    

### [[2109.01949] Improving Joint Learning of Chest X-Ray and Radiology Report by Word Region Alignment](http://arxiv.org/abs/2109.01949)


  Self-supervised learning provides an opportunity to explore unlabeled chest
X-rays and their associated free-text reports accumulated in clinical routine
without manual supervision. This paper proposes a Joint Image Text
Representation Learning Network (JoImTeRNet) for pre-training on chest X-ray
images and their radiology reports. The model was pre-trained on both the
global image-sentence level and the local image region-word level for
visual-textual matching. Both are bidirectionally constrained on Cross-Entropy
based and ranking-based Triplet Matching Losses. The region-word matching is
calculated using the attention mechanism without direct supervision about their
mapping. The pre-trained multi-modal representation learning paves the way for
downstream tasks concerning image and/or text encoding. We demonstrate the
representation learning quality by cross-modality retrievals and multi-label
classifications on two datasets: OpenI-IU and MIMIC-CXR

    

### [[2109.01951] FewshotQA: A simple framework for few-shot learning of question answering tasks using pre-trained text-to-text models](http://arxiv.org/abs/2109.01951)


  The task of learning from only a few examples (called a few-shot setting) is
of key importance and relevance to a real-world setting. For question answering
(QA), the current state-of-the-art pre-trained models typically need
fine-tuning on tens of thousands of examples to obtain good results. Their
performance degrades significantly in a few-shot setting (< 100 examples). To
address this, we propose a simple fine-tuning framework that leverages
pre-trained text-to-text models and is directly aligned with their pre-training
framework. Specifically, we construct the input as a concatenation of the
question, a mask token representing the answer span and a context. Given this
input, the model is fine-tuned using the same objective as that of its
pre-training objective. Through experimental studies on various few-shot
configurations, we show that this formulation leads to significant gains on
multiple QA benchmarks (an absolute gain of 34.2 F1 points on average when
there are only 16 training examples). The gains extend further when used with
larger models (Eg:- 72.3 F1 on SQuAD using BART-large with only 32 examples)
and translate well to a multilingual setting . On the multilingual TydiQA
benchmark, our model outperforms the XLM-Roberta-large by an absolute margin of
upto 40 F1 points and an average of 33 F1 points in a few-shot setting (<= 64
training examples). We conduct detailed ablation studies to analyze factors
contributing to these gains.

    

### [[2109.01965] Scalable Feature Selection for (Multitask) Gradient Boosted Trees](http://arxiv.org/abs/2109.01965)


  Gradient Boosted Decision Trees (GBDTs) are widely used for building ranking
and relevance models in search and recommendation. Considerations such as
latency and interpretability dictate the use of as few features as possible to
train these models. Feature selection in GBDT models typically involves
heuristically ranking the features by importance and selecting the top few, or
by performing a full backward feature elimination routine. On-the-fly feature
selection methods proposed previously scale suboptimally with the number of
features, which can be daunting in high dimensional settings. We develop a
scalable forward feature selection variant for GBDT, via a novel group testing
procedure that works well in high dimensions, and enjoys favorable theoretical
performance and computational guarantees. We show via extensive experiments on
both public and proprietary datasets that the proposed method offers
significant speedups in training time, while being as competitive as existing
GBDT methods in terms of model performance metrics. We also extend the method
to the multitask setting, allowing the practitioner to select common features
across tasks, as well as selecting task-specific features.

    

### [[2109.01980] Deep Saliency Prior for Reducing Visual Distraction](http://arxiv.org/abs/2109.01980)


  Using only a model that was trained to predict where people look at images,
and no additional training data, we can produce a range of powerful editing
effects for reducing distraction in images. Given an image and a mask
specifying the region to edit, we backpropagate through a state-of-the-art
saliency model to parameterize a differentiable editing operator, such that the
saliency within the masked region is reduced. We demonstrate several operators,
including: a recoloring operator, which learns to apply a color transform that
camouflages and blends distractors into their surroundings; a warping operator,
which warps less salient image regions to cover distractors, gradually
collapsing objects into themselves and effectively removing them (an effect
akin to inpainting); a GAN operator, which uses a semantic prior to fully
replace image regions with plausible, less salient alternatives. The resulting
effects are consistent with cognitive research on the human visual system
(e.g., since color mismatch is salient, the recoloring operator learns to
harmonize objects' colors with their surrounding to reduce their saliency),
and, importantly, are all achieved solely through the guidance of the
pretrained saliency model, with no additional supervision. We present results
on a variety of natural images and conduct a perceptual study to evaluate and
validate the changes in viewers' eye-gaze between the original images and our
edited results.

    

### [[2109.01983] Training Meta-Surrogate Model for Transferable Adversarial Attack](http://arxiv.org/abs/2109.01983)


  We consider adversarial attacks to a black-box model when no queries are
allowed. In this setting, many methods directly attack surrogate models and
transfer the obtained adversarial examples to fool the target model. Plenty of
previous works investigated what kind of attacks to the surrogate model can
generate more transferable adversarial examples, but their performances are
still limited due to the mismatches between surrogate models and the target
model. In this paper, we tackle this problem from a novel angle -- instead of
using the original surrogate models, can we obtain a Meta-Surrogate Model (MSM)
such that attacks to this model can be easier transferred to other models? We
show that this goal can be mathematically formulated as a well-posed
(bi-level-like) optimization problem and design a differentiable attacker to
make training feasible. Given one or a set of surrogate models, our method can
thus obtain an MSM such that adversarial examples generated on MSM enjoy
eximious transferability. Comprehensive experiments on Cifar-10 and ImageNet
demonstrate that by attacking the MSM, we can obtain stronger transferable
adversarial examples to fool black-box models including adversarially trained
ones, with much higher success rates than existing methods. The proposed method
reveals significant security challenges of deep models and is promising to be
served as a state-of-the-art benchmark for evaluating the robustness of deep
models in the black-box setting.

    

### [[2109.01991] Optimal transport weights for causal inference](http://arxiv.org/abs/2109.01991)


  Weighting methods are a common tool to de-bias estimates of causal effects.
And though there are an increasing number of seemingly disparate methods, many
of them can be folded into one unifying regime: causal optimal transport. This
new method directly targets distributional balance by minimizing optimal
transport distances between treatment and control groups or, more generally,
between a source and target population. Our approach is model-free but can also
incorporate moments or any other important functions of covariates that the
researcher desires to balance. We find that the causal optimal transport
outperforms competitor methods when both the propensity score and outcome
models are misspecified, indicating it is a robust alternative to common
weighting methods. Finally, we demonstrate the utility of our method in an
external control study examining the effect of misoprostol versus oxytocin for
treatment of post-partum hemorrhage.

    

### [[2109.01996] Automatic Online Multi-Source Domain Adaptation](http://arxiv.org/abs/2109.01996)


  Knowledge transfer across several streaming processes remain challenging
problem not only because of different distributions of each stream but also
because of rapidly changing and never-ending environments of data streams.
Albeit growing research achievements in this area, most of existing works are
developed for a single source domain which limits its resilience to exploit
multi-source domains being beneficial to recover from concept drifts quickly
and to avoid the negative transfer problem. An online domain adaptation
technique under multisource streaming processes, namely automatic online
multi-source domain adaptation (AOMSDA), is proposed in this paper. The online
domain adaptation strategy of AOMSDA is formulated under a coupled generative
and discriminative approach of denoising autoencoder (DAE) where the central
moment discrepancy (CMD)-based regularizer is integrated to handle the
existence of multi-source domains thereby taking advantage of complementary
information sources. The asynchronous concept drifts taking place at different
time periods are addressed by a self-organizing structure and a node
re-weighting strategy. Our numerical study demonstrates that AOMSDA is capable
of outperforming its counterparts in 5 of 8 study cases while the ablation
study depicts the advantage of each learning component. In addition, AOMSDA is
general for any number of source streams. The source code of AOMSDA is shared
publicly in this https URL.

    

### [[2109.02008] Sparse-MLP: A Fully-MLP Architecture with Conditional Computation](http://arxiv.org/abs/2109.02008)


  Mixture of Experts (MoE) with sparse conditional computation has been proved
an effective architecture for scaling attention-based models to more parameters
with comparable computation cost. In this paper, we propose Sparse-MLP, scaling
the recent MLP-Mixer model with sparse MoE layers, to achieve a more
computation-efficient architecture. We replace a subset of dense MLP blocks in
the MLP-Mixer model with Sparse blocks. In each Sparse block, we apply two
stages of MoE layers: one with MLP experts mixing information within channels
along image patch dimension, one with MLP experts mixing information within
patches along the channel dimension. Besides, to reduce computational cost in
routing and improve experts capacity, we design Re-represent layers in each
Sparse block. These layers are to re-scale image representations by two simple
but effective linear transformations. By pre-training on ImageNet-1k with MoCo
v3 algorithm, our models can outperform dense MLP models with comparable
parameters and less computational cost on several downstream image
classification tasks.

    

### [[2109.02018] Tolerating Adversarial Attacks and Byzantine Faults in Distributed Machine Learning](http://arxiv.org/abs/2109.02018)


  Adversarial attacks attempt to disrupt the training, retraining and utilizing
of artificial intelligence and machine learning models in large-scale
distributed machine learning systems. This causes security risks on its
prediction outcome. For example, attackers attempt to poison the model by
either presenting inaccurate misrepresentative data or altering the models'
parameters. In addition, Byzantine faults including software, hardware, network
issues occur in distributed systems which also lead to a negative impact on the
prediction outcome. In this paper, we propose a novel distributed training
algorithm, partial synchronous stochastic gradient descent (ParSGD), which
defends adversarial attacks and/or tolerates Byzantine faults. We demonstrate
the effectiveness of our algorithm under three common adversarial attacks again
the ML models and a Byzantine fault during the training phase. Our results show
that using ParSGD, ML models can still produce accurate predictions as if it is
not being attacked nor having failures at all when almost half of the nodes are
being compromised or failed. We will report the experimental evaluations of
ParSGD in comparison with other algorithms.

    

### [[2109.02027] Structural Optimization Makes Graph Classification Simpler and Better](http://arxiv.org/abs/2109.02027)


  In deep neural networks, better results can often be obtained by increasing
the complexity of previously developed basic models. However, it is unclear
whether there is a way to boost performance by decreasing the complexity of
such models. Here, based on an optimization method, we investigate the
feasibility of improving graph classification performance while simplifying the
model learning process. Inspired by progress in structural information
assessment, we optimize the given data sample from graphs to encoding trees. In
particular, we minimize the structural entropy of the transformed encoding tree
to decode the key structure underlying a graph. This transformation is denoted
as structural optimization. Furthermore, we propose a novel feature combination
scheme, termed hierarchical reporting, for encoding trees. In this scheme,
features are transferred from leaf nodes to root nodes by following the
hierarchical structures of encoding trees. We then present an implementation of
the scheme in a tree kernel and a convolutional network to perform graph
classification. The tree kernel follows label propagation in the
Weisfeiler-Lehman (WL) subtree kernel, but it has a lower runtime complexity
$O(n)$. The convolutional network is a special implementation of our tree
kernel in the deep learning field and is called Encoding Tree Learning (ETL).
We empirically validate our tree kernel and convolutional network with several
graph classification benchmarks and demonstrate that our methods achieve better
performance and lower computational consumption than competing approaches.

    

### [[2109.02032] Soft Hierarchical Graph Recurrent Networks for Many-Agent Partially Observable Environments](http://arxiv.org/abs/2109.02032)


  The recent progress in multi-agent deep reinforcement learning(MADRL) makes
it more practical in real-world tasks, but its relatively poor scalability and
the partially observable constraints raise challenges to its performance and
deployment. Based on our intuitive observation that the human society could be
regarded as a large-scale partially observable environment, where each
individual has the function of communicating with neighbors and remembering its
own experience, we propose a novel network structure called hierarchical graph
recurrent network(HGRN) for multi-agent cooperation under partial
observability. Specifically, we construct the multi-agent system as a graph,
use the hierarchical graph attention network(HGAT) to achieve communication
between neighboring agents, and exploit GRU to enable agents to record
historical information. To encourage exploration and improve robustness, we
design a maximum-entropy learning method to learn stochastic policies of a
configurable target action entropy. Based on the above technologies, we
proposed a value-based MADRL algorithm called Soft-HGRN and its actor-critic
variant named SAC-HRGN. Experimental results based on three homogeneous tasks
and one heterogeneous environment not only show that our approach achieves
clear improvements compared with four baselines, but also demonstrates the
interpretability, scalability, and transferability of the proposed model.
Ablation studies prove the function and necessity of each component.

    

### [[2109.02035] Variational Physics Informed Neural Networks: the role of quadratures and test functions](http://arxiv.org/abs/2109.02035)


  In this work we analyze how Gaussian or Newton-Cotes quadrature rules of
different precisions and piecewise polynomial test functions of different
degrees affect the convergence rate of Variational Physics Informed Neural
Networks (VPINN) with respect to mesh refinement, while solving elliptic
boundary-value problems. Using a Petrov-Galerkin framework relying on an
inf-sup condition, we derive an a priori error estimate in the energy norm
between the exact solution and a suitable high-order piecewise interpolant of a
computed neural network. Numerical experiments confirm the theoretical
predictions, and also indicate that the error decay follows the same behavior
when the neural network is not interpolated. Our results suggest, somehow
counterintuitively, that for smooth solutions the best strategy to achieve a
high decay rate of the error consists in choosing test functions of the lowest
polynomial degree, while using quadrature formulas of suitably high precision.

    

### [[2109.02038] NAS-OoD: Neural Architecture Search for Out-of-Distribution Generalization](http://arxiv.org/abs/2109.02038)


  Recent advances on Out-of-Distribution (OoD) generalization reveal the
robustness of deep learning models against distribution shifts. However,
existing works focus on OoD algorithms, such as invariant risk minimization,
domain generalization, or stable learning, without considering the influence of
deep model architectures on OoD generalization, which may lead to sub-optimal
performance. Neural Architecture Search (NAS) methods search for architecture
based on its performance on the training data, which may result in poor
generalization for OoD tasks. In this work, we propose robust Neural
Architecture Search for OoD generalization (NAS-OoD), which optimizes the
architecture with respect to its performance on generated OoD data by gradient
descent. Specifically, a data generator is learned to synthesize OoD data by
maximizing losses computed by different neural architectures, while the goal
for architecture search is to find the optimal architecture parameters that
minimize the synthetic OoD data losses. The data generator and the neural
architecture are jointly optimized in an end-to-end manner, and the minimax
training process effectively discovers robust architectures that generalize
well for different distribution shifts. Extensive experimental results show
that NAS-OoD achieves superior performance on various OoD generalization
benchmarks with deep models having a much fewer number of parameters. In
addition, on a real industry dataset, the proposed NAS-OoD method reduces the
error rate by more than 70% compared with the state-of-the-art method,
demonstrating the proposed method's practicality for real applications.

    

### [[2109.02040] Data Efficient Masked Language Modeling for Vision and Language](http://arxiv.org/abs/2109.02040)


  Masked language modeling (MLM) is one of the key sub-tasks in vision-language
pretraining. In the cross-modal setting, tokens in the sentence are masked at
random, and the model predicts the masked tokens given the image and the text.
In this paper, we observe several key disadvantages of MLM in this setting.
First, as captions tend to be short, in a third of the sentences no token is
sampled. Second, the majority of masked tokens are stop-words and punctuation,
leading to under-utilization of the image. We investigate a range of
alternative masking strategies specific to the cross-modal setting that address
these shortcomings, aiming for better fusion of text and image in the learned
representation. When pre-training the LXMERT model, our alternative masking
strategies consistently improve over the original masking strategy on three
downstream tasks, especially in low resource settings. Further, our
pre-training approach substantially outperforms the baseline model on a
prompt-based probing task designed to elicit image objects. These results and
our analysis indicate that our method allows for better utilization of the
training data.

    

### [[2109.02052] The Phonexia VoxCeleb Speaker Recognition Challenge 2021 System Description](http://arxiv.org/abs/2109.02052)


  We describe the Phonexia submission for the VoxCeleb Speaker Recognition
Challenge 2021 (VoxSRC-21) in the unsupervised speaker verification track. Our
solution was very similar to IDLab's winning submission for VoxSRC-20. An
embedding extractor was bootstrapped using momentum contrastive learning, with
input augmentations as the only source of supervision. This was followed by
several iterations of clustering to assign pseudo-speaker labels that were then
used for supervised embedding extractor training. Finally, a score fusion was
done, by averaging the zt-normalized cosine scores of five different embedding
extractors. We briefly also describe unsuccessful solutions involving i-vectors
instead of DNN embeddings and PLDA instead of cosine scoring.

    

### [[2109.02074] Global-Local Item Embedding for Temporal Set Prediction](http://arxiv.org/abs/2109.02074)


  Temporal set prediction is becoming increasingly important as many companies
employ recommender systems in their online businesses, e.g., personalized
purchase prediction of shopping baskets. While most previous techniques have
focused on leveraging a user's history, the study of combining it with others'
histories remains untapped potential. This paper proposes Global-Local Item
Embedding (GLOIE) that learns to utilize the temporal properties of sets across
whole users as well as within a user by coining the names as global and local
information to distinguish the two temporal patterns. GLOIE uses Variational
Autoencoder (VAE) and dynamic graph-based model to capture global and local
information and then applies attention to integrate resulting item embeddings.
Additionally, we propose to use Tweedie output for the decoder of VAE as it can
easily model zero-inflated and long-tailed distribution, which is more suitable
for several real-world data distributions than Gaussian or multinomial
counterparts. When evaluated on three public benchmarks, our algorithm
consistently outperforms previous state-of-the-art methods in most ranking
metrics.

    

### [[2109.02080] Providing an Approach to Predicting Customer Quality in E-Commerce Social Networks Based on Big Data and Unsupervised Learning Method](http://arxiv.org/abs/2109.02080)


  One of the goals of every business enterprise is to increase customer
loyalty. The degree of customer loyalty is called customer quality which its
forecasting will affect strategic marketing practices. The purpose of this
study is to predict the quality of customers of large e-commerce social
networks by big data algorithms and unsupervised learning. For this purpose, a
graph-based social network analysis framework was used for community detection
in the Stanford Network Analysis Platform (SNAP). Then in the found
communities, the quality of customers was predicted. The results showed that
various visits with an impact of 37.13% can have the greatest impact on
customer quality and the order of impact of other parameters were from highest
to lowest: number of frequent customer visits (28.56%), role in social networks
(28.37%), Indirect transactions (26.74%), activity days (25.62%) and customer
social network size (25.06%).

    

### [[2109.02082] Nonparametric Extrema Analysis in Time Series for Envelope Extraction, Peak Detection and Clustering](http://arxiv.org/abs/2109.02082)


  In this paper, we propose a nonparametric approach that can be used in
envelope extraction, peak-burst detection and clustering in time series. Our
problem formalization results in a naturally defined splitting/forking of the
time series. With a possibly hierarchical implementation, it can be used for
various applications in machine learning, signal processing and mathematical
finance. From an incoming input signal, our iterative procedure sequentially
creates two signals (one upper bounding and one lower bounding signal) by
minimizing the cumulative $L_1$ drift. We show that a solution can be
efficiently calculated by use of a Viterbi-like path tracking algorithm
together with an optimal elimination rule. We consider many interesting
settings, where our algorithm has near-linear time complexities.

    

### [[2109.02084] (M)SLAe-Net: Multi-Scale Multi-Level Attention embedded Network for Retinal Vessel Segmentation](http://arxiv.org/abs/2109.02084)


  Segmentation plays a crucial role in diagnosis. Studying the retinal
vasculatures from fundus images help identify early signs of many crucial
illnesses such as diabetic retinopathy. Due to the varying shape, size, and
patterns of retinal vessels, along with artefacts and noises in fundus images,
no one-stage method can accurately segment retinal vessels. In this work, we
propose a multi-scale, multi-level attention embedded CNN architecture
((M)SLAe-Net) to address the issue of multi-stage processing for robust and
precise segmentation of retinal vessels. We do this by extracting features at
multiple scales and multiple levels of the network, enabling our model to
holistically extracts the local and global features. Multi-scale features are
extracted using our novel dynamic dilated pyramid pooling (D-DPP) module. We
also aggregate the features from all the network levels. These effectively
resolved the issues of varying shapes and artefacts and hence the need for
multiple stages. To assist in better pixel-level classification, we use the
Squeeze and Attention(SA) module, a smartly adapted version of the Squeeze and
Excitation(SE) module for segmentation tasks in our network to facilitate
pixel-group attention. Our unique network design and novel D-DPP module with
efficient task-specific loss function for thin vessels enabled our model for
better cross data performance. Exhaustive experimental results on DRIVE, STARE,
HRF, and CHASE-DB1 show the superiority of our method.

    

### [[2109.02096] Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks](http://arxiv.org/abs/2109.02096)


  This research project investigates the application of deep learning to timbre
transfer, where the timbre of a source audio can be converted to the timbre of
a target audio with minimal loss in quality. The adopted approach combines
Variational Autoencoders with Generative Adversarial Networks to construct
meaningful representations of the source audio and produce realistic
generations of the target audio and is applied to the Flickr 8k Audio dataset
for transferring the vocal timbre between speakers and the URMP dataset for
transferring the musical timbre between instruments. Furthermore, variations of
the adopted approach are trained, and generalised performance is compared using
the metrics SSIM (Structural Similarity Index) and FAD (Frechét Audio
Distance). It was found that a many-to-many approach supersedes a one-to-one
approach in terms of reconstructive capabilities, and that the adoption of a
basic over a bottleneck residual block design is more suitable for enriching
content information about a latent space. It was also found that the decision
on whether cyclic loss takes on a variational autoencoder or vanilla
autoencoder approach does not have a significant impact on reconstructive and
adversarial translation aspects of the model.

    

### [[2109.02100] Cluster-Promoting Quantization with Bit-Drop for Minimizing Network Quantization Loss](http://arxiv.org/abs/2109.02100)


  Network quantization, which aims to reduce the bit-lengths of the network
weights and activations, has emerged for their deployments to resource-limited
devices. Although recent studies have successfully discretized a full-precision
network, they still incur large quantization errors after training, thus giving
rise to a significant performance gap between a full-precision network and its
quantized counterpart. In this work, we propose a novel quantization method for
neural networks, Cluster-Promoting Quantization (CPQ) that finds the optimal
quantization grids while naturally encouraging the underlying full-precision
weights to gather around those quantization grids cohesively during training.
This property of CPQ is thanks to our two main ingredients that enable
differentiable quantization: i) the use of the categorical distribution
designed by a specific probabilistic parametrization in the forward pass and
ii) our proposed multi-class straight-through estimator (STE) in the backward
pass. Since our second component, multi-class STE, is intrinsically biased, we
additionally propose a new bit-drop technique, DropBits, that revises the
standard dropout regularization to randomly drop bits instead of neurons. As a
natural extension of DropBits, we further introduce the way of learning
heterogeneous quantization levels to find proper bit-length for each layer by
imposing an additional regularization on DropBits. We experimentally validate
our method on various benchmark datasets and network architectures, and also
support a new hypothesis for quantization: learning heterogeneous quantization
levels outperforms the case using the same but fixed quantization levels from
scratch.

    

### [[2109.02117] VARGAN: Variance Enforcing Network Enhanced GAN](http://arxiv.org/abs/2109.02117)


  Generative adversarial networks (GANs) are one of the most widely used
generative models. GANs can learn complex multi-modal distributions, and
generate real-like samples. Despite the major success of GANs in generating
synthetic data, they might suffer from unstable training process, and mode
collapse. In this paper, we introduce a new GAN architecture called variance
enforcing GAN (VARGAN), which incorporates a third network to introduce
diversity in the generated samples. The third network measures the diversity of
the generated samples, which is used to penalize the generator's loss for low
diversity samples. The network is trained on the available training data and
undesired distributions with limited modality. On a set of synthetic and
real-world image data, VARGAN generates a more diverse set of samples compared
to the recent state-of-the-art models. High diversity and low computational
complexity, as well as fast convergence, make VARGAN a promising model to
alleviate mode collapse.

    

### [[2109.02119] Identification of Driver Phone Usage Violations via State-of-the-Art Object Detection with Tracking](http://arxiv.org/abs/2109.02119)


  The use of mobiles phones when driving have been a major factor when it comes
to road traffic incidents and the process of capturing such violations can be a
laborious task. Advancements in both modern object detection frameworks and
high-performance hardware has paved the way for a more automated approach when
it comes to video surveillance. In this work, we propose a custom-trained
state-of-the-art object detector to work with roadside cameras to capture
driver phone usage without the need for human intervention. The proposed
approach also addresses the issues caused by windscreen glare and introduces
the steps required to remedy this. Twelve pre-trained models are fine-tuned
with our custom dataset using four popular object detection methods: YOLO, SSD,
Faster R-CNN, and CenterNet. Out of all the object detectors tested, the YOLO
yields the highest accuracy levels of up to 96% (AP10) and frame rates of up to
~30 FPS. DeepSort object tracking algorithm is also integrated into the
best-performing model to collect records of only the unique violations, and
enable the proposed approach to count the number of vehicles. The proposed
automated system will collect the output images of the identified violations,
timestamps of each violation, and total vehicle count. Data can be accessed via
a purpose-built user interface.

    

### [[2109.02138] A Transformer-based Model to Detect Phishing URLs](http://arxiv.org/abs/2109.02138)


  Phishing attacks are among emerging security issues that recently draws
significant attention in the cyber security community. There are numerous
existing approaches for phishing URL detection. However, malicious URL
detection is still a research hotspot because attackers can bypass newly
introduced detection mechanisms by changing their tactics. This paper will
introduce a transformer-based malicious URL detection model, which has
significant accuracy and outperforms current detection methods. We conduct
experiments and compare them with six existing classical detection models.
Experiments demonstrate that our transformer-based model is the best performing
model from all perspectives among the seven models and achieves 97.3 % of
detection accuracy.

    

### [[2109.02145] Temporal Aware Deep Reinforcement Learning](http://arxiv.org/abs/2109.02145)


  The function approximators employed by traditional image based Deep
Reinforcement Learning (DRL) algorithms usually lack a temporal learning
component and instead focus on learning the spatial component. We propose a
technique wherein both temporal as well as spatial components are jointly
learned. Our tested was tested with a generic DQN and it outperformed it in
terms of maximum rewards as well as sample complexity. This algorithm has
implications in the robotics as well as sequential decision making domains.

    

### [[2109.02150] Robust Importance Sampling for Error Estimation in the Context of Optimal Bayesian Transfer Learning](http://arxiv.org/abs/2109.02150)


  Classification has been a major task for building intelligent systems as it
enables decision-making under uncertainty. Classifier design aims at building
models from training data for representing feature-label distributions--either
explicitly or implicitly. In many scientific or clinical settings, training
data are typically limited, which makes designing accurate classifiers and
evaluating their classification error extremely challenging. While transfer
learning (TL) can alleviate this issue by incorporating data from relevant
source domains to improve learning in a different target domain, it has
received little attention for performance assessment, notably in error
estimation. In this paper, we fill this gap by investigating knowledge
transferability in the context of classification error estimation within a
Bayesian paradigm. We introduce a novel class of Bayesian minimum mean-square
error (MMSE) estimators for optimal Bayesian transfer learning (OBTL), which
enables rigorous evaluation of classification error under uncertainty in a
small-sample setting. Using Monte Carlo importance sampling, we employ the
proposed estimator to evaluate the classification accuracy of a broad family of
classifiers that span diverse learning capabilities. Experimental results based
on both synthetic data as well as real-world RNA sequencing (RNA-seq) data show
that our proposed OBTL error estimation scheme clearly outperforms standard
error estimators, especially in a small-sample setting, by tapping into the
data from other relevant domains.

    

### [[2109.02157] Learning with Holographic Reduced Representations](http://arxiv.org/abs/2109.02157)


  Holographic Reduced Representations (HRR) are a method for performing
symbolic AI on top of real-valued vectors \cite{Plate1995} by associating each
vector with an abstract concept, and providing mathematical operations to
manipulate vectors as if they were classic symbolic objects. This method has
seen little use outside of older symbolic AI work and cognitive science. Our
goal is to revisit this approach to understand if it is viable for enabling a
hybrid neural-symbolic approach to learning as a differentiable component of a
deep learning architecture. HRRs today are not effective in a differentiable
solution due to numerical instability, a problem we solve by introducing a
projection step that forces the vectors to exist in a well behaved point in
space. In doing so we improve the concept retrieval efficacy of HRRs by over
$100\times$. Using multi-label classification we demonstrate how to leverage
the symbolic HRR properties to develop an output layer and loss function that
is able to learn effectively, and allows us to investigate some of the pros and
cons of an HRR neuro-symbolic learning approach.

    

### [[2109.02160] Urban Fire Station Location Planning: A Systematic Approach using Predicted Demand and Service Quality Index](http://arxiv.org/abs/2109.02160)


  In this article, we propose a systematic approach for fire station location
planning. We develop a machine learning model, based on Random Forest, for
demand prediction and utilize the model further to define a generalized index
to measure quality of fire service in urban settings. Our model is built upon
spatial data collected from multiple different sources. Efficacy of proper
facility planning depends on choice of candidates where fire stations can be
located along with existing stations, if any. Also, the travel time from these
candidates to demand locations need to be taken care of to maintain fire safety
standard. Here, we propose a travel time based clustering technique to identify
suitable candidates. Finally, we develop an optimization problem to select best
locations to install new fire stations. Our optimization problem is built upon
maximum coverage problem, based on integer programming. We present a detailed
experimental study of our proposed approach in collaboration with city of
Victoria Fire Department, MN, USA. Our demand prediction model achieves true
positive rate of 70% and false positive rate of 22% approximately. We aid
Victoria Fire Department to select a location for a new fire station using our
approach. We present detailed results on improvement statistics by locating a
new facility, as suggested by our methodology, in the city of Victoria.

    

### [[2109.02165] FBCNN: A Deep Neural Network Architecture for Portable and Fast Brain-Computer Interfaces](http://arxiv.org/abs/2109.02165)


  Objective: To propose a novel deep neural network (DNN) architecture -- the
filter bank convolutional neural network (FBCNN) -- to improve SSVEP
classification in single-channel BCIs with small data lengths.
Methods: We propose two models: the FBCNN-2D and the FBCNN-3D. The FBCNN-2D
utilizes a filter bank to create sub-band components of the
electroencephalography (EEG) signal, which it transforms using the fast Fourier
transform (FFT) and analyzes with a 2D CNN. The FBCNN-3D utilizes the same
filter bank, but it transforms the sub-band components into spectrograms via
short-time Fourier transform (STFT), and analyzes them with a 3D CNN. We made
use of transfer learning. To train the FBCNN-3D, we proposed a new technique,
called inter-dimensional transfer learning, to transfer knowledge from a 2D DNN
to a 3D DNN. Our BCI was conceived so as not to require calibration from the
final user: therefore, the test subject data was separated from training and
validation.
Results: The mean test accuracy was 85.7% for the FBCCA-2D and 85% for the
FBCCA-3D. Mean F1-Scores were 0.858 and 0.853. Alternative classification
methods, SVM, FBCCA and a CNN, had mean accuracy of 79.2%, 80.1% and 81.4%,
respectively.
Conclusion: The FBCNNs surpassed traditional SSVEP classification methods in
our simulated BCI, by a considerable margin (about 5% higher accuracy).
Transfer learning and inter-dimensional transfer learning made training much
faster and more predictable.
Significance: We proposed a new and flexible type of DNN, which had a better
performance than standard methods in SSVEP classification for portable and fast
BCIs.

    

### [[2109.02173] Multi-Agent Variational Occlusion Inference Using People as Sensors](http://arxiv.org/abs/2109.02173)


  Autonomous vehicles must reason about spatial occlusions in urban
environments to ensure safety without being overly cautious. Prior work
explored occlusion inference from observed social behaviors of road agents.
Inferring occupancy from agent behaviors is an inherently multimodal problem; a
driver may behave in the same manner for different occupancy patterns ahead of
them (e.g., a driver may move at constant speed in traffic or on an open road).
Past work, however, does not account for this multimodality, thus neglecting to
model this source of aleatoric uncertainty in the relationship between driver
behaviors and their environment. We propose an occlusion inference method that
characterizes observed behaviors of human agents as sensor measurements, and
fuses them with those from a standard sensor suite. To capture the aleatoric
uncertainty, we train a conditional variational autoencoder with a discrete
latent space to learn a multimodal mapping from observed driver trajectories to
an occupancy grid representation of the view ahead of the driver. Our method
handles multi-agent scenarios, combining measurements from multiple observed
drivers using evidential theory to solve the sensor fusion problem. Our
approach is validated on a real-world dataset, outperforming baselines and
demonstrating real-time capable performance. Our code is available at
this https URL .

    

### [[2109.02178] A Critical Review of the state-of-the-art on Deep Neural Networks for Blood Glucose Prediction in Patients with Diabetes](http://arxiv.org/abs/2109.02178)


  This article compares ten recently proposed neural networks and proposes two
ensemble neural network-based models for blood glucose prediction. All of them
are tested under the same dataset, preprocessing workflow, and tools using the
OhioT1DM Dataset at three different prediction horizons: 30, 60, and 120
minutes. We compare their performance using the most common metrics in blood
glucose prediction and rank the best-performing ones using three methods
devised for the statistical comparison of the performance of multiple
algorithms: scmamp, model confidence set, and superior predictive ability. Our
analysis highlights those models with the highest probability of being the best
predictors, estimates the increase in error of the models that perform more
poorly with respect to the best ones, and provides a guide for their use in
clinical practice.

    

### [[2109.02183] Towards high-accuracy deep learning inference of compressible turbulent flows over aerofoils](http://arxiv.org/abs/2109.02183)


  The present study investigates the accurate inference of Reynolds-averaged
Navier-Stokes solutions for the compressible flow over aerofoils in two
dimensions with a deep neural network. Our approach yields networks that learn
to generate precise flow fields for varying body-fitted, structured grids by
providing them with an encoding of the corresponding mapping to a canonical
space for the solutions. We apply the deep neural network model to a benchmark
case of incompressible flow at randomly given angles of attack and Reynolds
numbers and achieve an improvement of more than an order of magnitude compared
to previous work. Further, for transonic flow cases, the deep neural network
model accurately predicts complex flow behaviour at high Reynolds numbers, such
as shock wave/boundary layer interaction, and quantitative distributions like
pressure coefficient, skin friction coefficient as well as wake total pressure
profiles downstream of aerofoils. The proposed deep learning method
significantly speeds up the predictions of flow fields and shows promise for
enabling fast aerodynamic designs.

    

### [[2109.02202] Fairness via AI: Bias Reduction in Medical Information](http://arxiv.org/abs/2109.02202)


  Most Fairness in AI research focuses on exposing biases in AI systems. A
broader lens on fairness reveals that AI can serve a greater aspiration:
rooting out societal inequities from their source. Specifically, we focus on
inequities in health information, and aim to reduce bias in that domain using
AI. The AI algorithms under the hood of search engines and social media, many
of which are based on recommender systems, have an outsized impact on the
quality of medical and health information online. Therefore, embedding bias
detection and reduction into these recommender systems serving up medical and
health content online could have an outsized positive impact on patient
outcomes and wellbeing.
In this position paper, we offer the following contributions: (1) we propose
a novel framework of Fairness via AI, inspired by insights from medical
education, sociology and antiracism; (2) we define a new term, bisinformation,
which is related to, but distinct from, misinformation, and encourage
researchers to study it; (3) we propose using AI to study, detect and mitigate
biased, harmful, and/or false health information that disproportionately hurts
minority groups in society; and (4) we suggest several pillars and pose several
open problems in order to seed inquiry in this new space. While part (3) of
this work specifically focuses on the health domain, the fundamental computer
science advances and contributions stemming from research efforts in bias
reduction and Fairness via AI have broad implications in all areas of society.

    

### [[2109.02230] Non-Euclidean Analysis of Joint Variations in Multi-Object Shapes](http://arxiv.org/abs/2109.02230)


  This paper considers joint analysis of multiple functionally related
structures in classification tasks. In particular, our method developed is
driven by how functionally correlated brain structures vary together between
autism and control groups. To do so, we devised a method based on a novel
combination of (1) non-Euclidean statistics that can faithfully represent
non-Euclidean data in Euclidean spaces and (2) a non-parametric integrative
analysis method that can decompose multi-block Euclidean data into joint,
individual, and residual structures. We find that the resulting joint structure
is effective, robust, and interpretable in recognizing the underlying patterns
of the joint variation of multi-block non-Euclidean data. We verified the
method in classifying the structural shape data collected from cases that
developed and did not develop into Autistic Spectrum Disorder (ASD).

    

### [[2109.02235] Gradient Normalization for Generative Adversarial Networks](http://arxiv.org/abs/2109.02235)


  In this paper, we propose a novel normalization method called gradient
normalization (GN) to tackle the training instability of Generative Adversarial
Networks (GANs) caused by the sharp gradient space. Unlike existing work such
as gradient penalty and spectral normalization, the proposed GN only imposes a
hard 1-Lipschitz constraint on the discriminator function, which increases the
capacity of the discriminator. Moreover, the proposed gradient normalization
can be applied to different GAN architectures with little modification.
Extensive experiments on four datasets show that GANs trained with gradient
normalization outperform existing methods in terms of both Frechet Inception
Distance and Inception Score.

    

### [[2109.02241] Supervised DKRC with Images for Offline System Identification](http://arxiv.org/abs/2109.02241)


  Koopman spectral theory has provided a new perspective in the field of
dynamical systems in recent years. Modern dynamical systems are becoming
increasingly non-linear and complex, and there is a need for a framework to
model these systems in a compact and comprehensive representation for
prediction and control. The central problem in applying Koopman theory to a
system of interest is that the choice of finite-dimensional basis functions is
typically done apriori, using expert knowledge of the systems dynamics. Our
approach learns these basis functions using a supervised learning approach
where a combination of autoencoders and deep neural networks learn the basis
functions for any given system. We demonstrate this approach on a simple
pendulum example in which we obtain a linear representation of the non-linear
system and then predict the future state trajectories given some initial
conditions. We also explore how changing the input representation of the
dynamic systems time series data can impact the quality of learned basis
functions. This alternative representation is compared to the traditional raw
time series data approach to determine which method results in lower
reconstruction and prediction error of the true non-linear dynamics of the
system.

    

### [[2109.02248] Quantifying the Reproducibility of Graph Neural Networks using Multigraph Brain Data](http://arxiv.org/abs/2109.02248)


  Graph neural networks (GNNs) have witnessed an unprecedented proliferation in
tackling several problems in computer vision, computer-aided diagnosis, and
related fields. While prior studies have focused on boosting the model
accuracy, quantifying the reproducibility of the most discriminative features
identified by GNNs is still an intact problem that yields concerns about their
reliability in clinical applications in particular. Specifically, the
reproducibility of biological markers across clinical datasets and distribution
shifts across classes (e.g., healthy and disordered brains) is of paramount
importance in revealing the underpinning mechanisms of diseases as well as
propelling the development of personalized treatment. Motivated by these
issues, we propose, for the first time, reproducibility-based GNN selection
(RG-Select), a framework for GNN reproducibility assessment via the
quantification of the most discriminative features (i.e., biomarkers) shared
between different models. To ascertain the soundness of our framework, the
reproducibility assessment embraces variations of different factors such as
training strategies and data perturbations. Despite these challenges, our
framework successfully yielded replicable conclusions across different training
strategies and various clinical datasets. Our findings could thus pave the way
for the development of biomarker trustworthiness and reliability assessment
methods for computer-aided diagnosis and prognosis tasks. RG-Select code is
available on GitHub at this https URL.

    

### [[2109.02250] Estimating Leaf Water Content using Remotely Sensed Hyperspectral Data](http://arxiv.org/abs/2109.02250)


  Plant water stress may occur due to the limited availability of water to the
roots/soil or due to increased transpiration. These factors adversely affect
plant physiology and photosynthetic ability to the extent that it has been
shown to have inhibitory effects in both growth and yield [18]. Early
identification of plant water stress status enables suitable corrective
measures to be applied to obtain the expected crop yield. Further, improving
crop yield through precision agriculture methods is a key component of climate
policy and the UN sustainable development goals [1]. Leaf water content (LWC)
is a measure that can be used to estimate water content and identify stressed
plants. LWC during the early crop growth stages is an important indicator of
plant productivity and yield. The effect of water stress can be instantaneous
[15], affecting gaseous exchange or long-term, significantly reducing [9, 18,
22]. It is thus necessary to identify potential plant water stress during the
early stages of growth [15] to introduce corrective irrigation and alleviate
stress. LWC is also useful for identifying plant genotypes that are tolerant to
water stress and salinity by measuring the stability of LWC even under
artificially induced water stress [18, 25]. Such experiments generally employ
destructive procedures to obtain the LWC, which is time-consuming and labor
intensive. Accordingly, this research has developed a non-destructive method to
estimate LWC from UAV-based hyperspectral data.

    

### [[2109.02255] Data-Driven Learning of 3-Point Correlation Functions as Microstructure Representations](http://arxiv.org/abs/2109.02255)


  This paper considers the open challenge of identifying complete, concise, and
explainable quantitative microstructure representations for disordered
heterogeneous material systems. Completeness and conciseness have been achieved
through existing data-driven methods, e.g., deep generative models, which,
however, do not provide mathematically explainable latent representations. This
study investigates representations composed of three-point correlation
functions, which are a special type of spatial convolutions. We show that a
variety of microstructures can be characterized by a concise subset of
three-point correlations, and the identification of such subsets can be
achieved by Bayesian optimization. Lastly, we show that the proposed
representation can directly be used to compute material properties based on the
effective medium theory.

    

### [[2109.02273] Postulating Exoplanetary Habitability via a Novel Anomaly Detection Method](http://arxiv.org/abs/2109.02273)


  A profound shift in the study of cosmology came with the discovery of
thousands of exoplanets and the possibility of the existence of billions of
them in our Galaxy. The biggest goal in these searches is whether there are
other life-harbouring planets. However, the question which of these detected
planets are habitable, potentially-habitable, or maybe even inhabited, is still
not answered. Some potentially habitable exoplanets have been hypothesized, but
since Earth is the only known habitable planet, measures of habitability are
necessarily determined with Earth as the reference. Several recent works
introduced new habitability metrics based on optimization methods.
Classification of potentially habitable exoplanets using supervised learning is
another emerging area of study. However, both modeling and supervised learning
approaches suffer from drawbacks. We propose an anomaly detection method, the
Multi-Stage Memetic Algorithm (MSMA), to detect anomalies and extend it to an
unsupervised clustering algorithm MSMVMCA to use it to detect potentially
habitable exoplanets as anomalies. The algorithm is based on the postulate that
Earth is an anomaly, with the possibility of existence of few other anomalies
among thousands of data points. We describe an MSMA-based clustering approach
with a novel distance function to detect habitable candidates as anomalies
(including Earth). The results are cross-matched with the habitable exoplanet
catalog (PHL-HEC) of the Planetary Habitability Laboratory (PHL) with both
optimistic and conservative lists of potentially habitable exoplanets.

    

### [[2109.02283] Does Melania Trump have a body double from the perspective of automatic face recognition?](http://arxiv.org/abs/2109.02283)


  In this paper, we explore whether automatic face recognition can help in
verifying widespread misinformation on social media, particularly conspiracy
theories that are based on the existence of body doubles. The conspiracy theory
addressed in this paper is the case of the Melania Trump body double. We
employed four different state-of-the-art descriptors for face recognition to
verify the integrity of the claim of the studied conspiracy theory. In
addition, we assessed the impact of different image quality metrics on the
variation of face recognition results. Two sets of image quality metrics were
considered: acquisition-related metrics and subject-related metrics.

    

### [[2109.02314] Fast Hypergraph Regularized Nonnegative Tensor Ring Factorization Based on Low-Rank Approximation](http://arxiv.org/abs/2109.02314)


  For the high dimensional data representation, nonnegative tensor ring (NTR)
decomposition equipped with manifold learning has become a promising model to
exploit the multi-dimensional structure and extract the feature from tensor
data. However, the existing methods such as graph regularized tensor ring
decomposition (GNTR) only models the pair-wise similarities of objects. For
tensor data with complex manifold structure, the graph can not exactly
construct similarity relationships. In this paper, in order to effectively
utilize the higher-dimensional and complicated similarities among objects, we
introduce hypergraph to the framework of NTR to further enhance the feature
extraction, upon which a hypergraph regularized nonnegative tensor ring
decomposition (HGNTR) method is developed. To reduce the computational
complexity and suppress the noise, we apply the low-rank approximation trick to
accelerate HGNTR (called LraHGNTR). Our experimental results show that compared
with other state-of-the-art algorithms, the proposed HGNTR and LraHGNTR can
achieve higher performance in clustering tasks, in addition, LraHGNTR can
greatly reduce running time without decreasing accuracy.

    

### [[2109.02332] Hindsight Reward Tweaking via Conditional Deep Reinforcement Learning](http://arxiv.org/abs/2109.02332)


  Designing optimal reward functions has been desired but extremely difficult
in reinforcement learning (RL). When it comes to modern complex tasks,
sophisticated reward functions are widely used to simplify policy learning yet
even a tiny adjustment on them is expensive to evaluate due to the drastically
increasing cost of training. To this end, we propose a hindsight reward
tweaking approach by designing a novel paradigm for deep reinforcement learning
to model the influences of reward functions within a near-optimal space. We
simply extend the input observation with a condition vector linearly correlated
with the effective environment reward parameters and train the model in a
conventional manner except for randomizing reward configurations, obtaining a
hyper-policy whose characteristics are sensitively regulated over the condition
space. We demonstrate the feasibility of this approach and study one of its
potential application in policy performance boosting with multiple MuJoCo
tasks.

    

### [[2109.02344] Information Theory-Guided Heuristic Progressive Multi-View Coding](http://arxiv.org/abs/2109.02344)


  Multi-view representation learning captures comprehensive information from
multiple views of a shared context. Recent works intuitively apply contrastive
learning (CL) to learn representations, regarded as a pairwise manner, which is
still scalable: view-specific noise is not filtered in learning view-shared
representations; the fake negative pairs, where the negative terms are actually
within the same class as the positive, and the real negative pairs are
coequally treated; and evenly measuring the similarities between terms might
interfere with optimization. Importantly, few works research the theoretical
framework of generalized self-supervised multi-view learning, especially for
more than two views. To this end, we rethink the existing multi-view learning
paradigm from the information theoretical perspective and then propose a novel
information theoretical framework for generalized multi-view learning. Guided
by it, we build a multi-view coding method with a three-tier progressive
architecture, namely Information theory-guided heuristic Progressive Multi-view
Coding (IPMC). In the distribution-tier, IPMC aligns the distribution between
views to reduce view-specific noise. In the set-tier, IPMC builds self-adjusted
pools for contrasting, which utilizes a view filter to adaptively modify the
pools. Lastly, in the instance-tier, we adopt a designed unified loss to learn
discriminative representations and reduce the gradient interference.
Theoretically and empirically, we demonstrate the superiority of IPMC over
state-of-the-art methods.

    

### [[2109.02345] Tensor Normalization and Full Distribution Training](http://arxiv.org/abs/2109.02345)


  In this work, we introduce pixel wise tensor normalization, which is inserted
after rectifier linear units and, together with batch normalization, provides a
significant improvement in the accuracy of modern deep neural networks. In
addition, this work deals with the robustness of networks. We show that the
factorized superposition of images from the training set and the reformulation
of the multi class problem into a multi-label problem yields significantly more
robust networks. The reformulation and the adjustment of the multi class log
loss also improves the results compared to the overlay with only one class as
label.
this https URL


### [[2109.02351] Fair Federated Learning for Heterogeneous Face Data](http://arxiv.org/abs/2109.02351)


  We consider the problem of achieving fair classification in Federated
Learning (FL) under data heterogeneity. Most of the approaches proposed for
fair classification require diverse data that represent the different
demographic groups involved. In contrast, it is common for each client to own
data that represents only a single demographic group. Hence the existing
approaches cannot be adopted for fair classification models at the client
level. To resolve this challenge, we propose several aggregation techniques. We
empirically validate these techniques by comparing the resulting fairness
metrics and accuracy on CelebA, UTK, and FairFace datasets.

    

### [[2109.02355] A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning](http://arxiv.org/abs/2109.02355)


  The rapid recent progress in machine learning (ML) has raised a number of
scientific questions that challenge the longstanding dogma of the field. One of
the most important riddles is the good empirical generalization of
overparameterized models. Overparameterized models are excessively complex with
respect to the size of the training dataset, which results in them perfectly
fitting (i.e., interpolating) the training data, which is usually noisy. Such
interpolation of noisy data is traditionally associated with detrimental
overfitting, and yet a wide range of interpolating models -- from simple linear
models to deep neural networks -- have recently been observed to generalize
extremely well on fresh test data. Indeed, the recently discovered double
descent phenomenon has revealed that highly overparameterized models often
improve over the best underparameterized model in test performance.
Understanding learning in this overparameterized regime requires new theory
and foundational empirical studies, even for the simplest case of the linear
model. The underpinnings of this understanding have been laid in very recent
analyses of overparameterized linear regression and related statistical
learning tasks, which resulted in precise analytic characterizations of double
descent. This paper provides a succinct overview of this emerging theory of
overparameterized ML (henceforth abbreviated as TOPML) that explains these
recent findings through a statistical signal processing perspective. We
emphasize the unique aspects that define the TOPML research area as a subfield
of modern ML theory and outline interesting open questions that remain.

    

### [[2109.02357] Visual Recognition with Deep Learning from Biased Image Datasets](http://arxiv.org/abs/2109.02357)


  In practice, and more especially when training deep neural networks, visual
recognition rules are often learned based on various sources of information. On
the other hand, the recent deployment of facial recognition systems with uneven
predictive performances on different population segments highlights the
representativeness issues possibly induced by a naive aggregation of image
datasets. Indeed, sampling bias does not vanish simply by considering larger
datasets, and ignoring its impact may completely jeopardize the generalization
capacity of the learned prediction rules. In this paper, we show how biasing
models, originally introduced for nonparametric estimation in (Gill et al.,
1988), and recently revisited from the perspective of statistical learning
theory in (Laforgue and Clémençon, 2019), can be applied to remedy these
problems in the context of visual recognition. Based on the (approximate)
knowledge of the biasing mechanisms at work, our approach consists in
reweighting the observations, so as to form a nearly debiased estimator of the
target distribution. One key condition for our method to be theoretically valid
is that the supports of the distributions generating the biased datasets at
disposal must overlap, and cover the support of the target distribution. In
order to meet this requirement in practice, we propose to use a low dimensional
image representation, shared across the image databases. Finally, we provide
numerical experiments highlighting the relevance of our approach whenever the
biasing functions are appropriately chosen.

    

### [[2109.02358] Pointspectrum: Equivariance Meets Laplacian Filtering for Graph Representation Learning](http://arxiv.org/abs/2109.02358)


  Graph Representation Learning (GRL) has become essential for modern graph
data mining and learning tasks. GRL aims to capture the graph's structural
information and exploit it in combination with node and edge attributes to
compute low-dimensional representations. While Graph Neural Networks (GNNs)
have been used in state-of-the-art GRL architectures, they have been shown to
suffer from over smoothing when many GNN layers need to be stacked. In a
different GRL approach, spectral methods based on graph filtering have emerged
addressing over smoothing; however, up to now, they employ traditional neural
networks that cannot efficiently exploit the structure of graph data. Motivated
by this, we propose PointSpectrum, a spectral method that incorporates a set
equivariant network to account for a graph's structure. PointSpectrum enhances
the efficiency and expressiveness of spectral methods, while it outperforms or
competes with state-of-the-art GRL methods. Overall, PointSpectrum addresses
over smoothing by employing a graph filter and captures a graph's structure
through set equivariance, lying on the intersection of GNNs and spectral
methods. Our findings are promising for the benefits and applicability of this
architectural shift for spectral methods and GRL.

    

### [[2109.02362] Comparing the Machine Readability of Traffic Sign Pictograms in Austria and Germany](http://arxiv.org/abs/2109.02362)


  We compare the machine readability of pictograms found on Austrian and German
traffic signs. To that end, we train classification models on synthetic data
sets and evaluate their classification accuracy in a controlled setting. In
particular, we focus on differences between currently deployed pictograms in
the two countries, and a set of new pictograms designed to increase human
readability. Besides other results, we find that machine-learning models
generalize poorly to data sets with pictogram designs they have not been
trained on. We conclude that manufacturers of advanced driver-assistance
systems (ADAS) must take special care to properly address small visual
differences between current and newly designed traffic sign pictograms, as well
as between pictograms from different countries.

    

### [[1710.03113] Toward Multidiversified Ensemble Clustering of High-Dimensional Data: From Subspaces to Metrics and Beyond](http://arxiv.org/abs/1710.03113)


  The rapid emergence of high-dimensional data in various areas has brought new
challenges to current ensemble clustering research. To deal with the curse of
dimensionality, recently considerable efforts in ensemble clustering have been
made by means of different subspace-based techniques. However, besides the
emphasis on subspaces, rather limited attention has been paid to the potential
diversity in similarity/dissimilarity metrics. It remains a surprisingly open
problem in ensemble clustering how to create and aggregate a large population
of diversified metrics, and furthermore, how to jointly investigate the
multi-level diversity in the large populations of metrics, subspaces, and
clusters in a unified framework. To tackle this problem, this paper proposes a
novel multidiversified ensemble clustering approach. In particular, we create a
large number of diversified metrics by randomizing a scaled exponential
similarity kernel, which are then coupled with random subspaces to form a large
set of metric-subspace pairs. Based on the similarity matrices derived from
these metric-subspace pairs, an ensemble of diversified base clusterings can
thereby be constructed. Further, an entropy-based criterion is utilized to
explore the cluster-wise diversity in ensembles, based on which three specific
ensemble clustering algorithms are presented by incorporating three types of
consensus functions. Extensive experiments are conducted on 30 high-dimensional
datasets, including 18 cancer gene expression datasets and 12 image/speech
datasets, which demonstrate the superiority of our algorithms over the
state-of-the-art. The source code is available at
this https URL.

    

### [[1809.03066] Multi-agent online learning in time-varying games](http://arxiv.org/abs/1809.03066)


  We examine the long-run behavior of multi-agent online learning in games that
evolve over time. Specifically, we focus on a wide class of policies based on
mirror descent, and we show that the induced sequence of play (a) converges to
Nash equilibrium in time-varying games that stabilize in the long run to a
strictly monotone limit; and (b) it stays asymptotically close to the evolving
equilibrium of the sequence of stage games (assuming they are strongly
monotone). Our results apply to both gradient-based and payoff-based feedback -
i.e., the "bandit feedback" case where players only get to observe the payoffs
of their chosen actions.

    

### [[1905.10115] Multi-Kernel Correntropy for Robust Learning](http://arxiv.org/abs/1905.10115)


  As a novel similarity measure that is defined as the expectation of a kernel
function between two random variables, correntropy has been successfully
applied in robust machine learning and signal processing to combat large
outliers. The kernel function in correntropy is usually a zero-mean Gaussian
kernel. In a recent work, the concept of mixture correntropy (MC) was proposed
to improve the learning performance, where the kernel function is a mixture
Gaussian kernel, namely a linear combination of several zero-mean Gaussian
kernels with different widths. In both correntropy and mixture correntropy, the
center of the kernel function is, however, always located at zero. In the
present work, to further improve the learning performance, we propose the
concept of multi-kernel correntropy (MKC), in which each component of the
mixture Gaussian kernel can be centered at a different location. The properties
of the MKC are investigated and an efficient approach is proposed to determine
the free parameters in MKC. Experimental results show that the learning
algorithms under the maximum multi-kernel correntropy criterion (MMKCC) can
outperform those under the original maximum correntropy criterion (MCC) and the
maximum mixture correntropy criterion (MMCC).

    

### [[1910.01723] Using Logical Specifications of Objectives in Multi-Objective Reinforcement Learning](http://arxiv.org/abs/1910.01723)


  It is notoriously difficult to control the behavior of reinforcement learning
agents. Agents often learn to exploit the environment or reward signal and need
to be retrained multiple times. The multi-objective reinforcement learning
(MORL) framework separates a reward function into several objectives. An ideal
MORL agent learns to generalize to novel combinations of objectives allowing
for better control of an agent's behavior without requiring retraining. Many
MORL approaches use a weight vector to parameterize the importance of each
objective. However, this approach suffers from lack of expressiveness and
interpretability. We propose using propositional logic to specify the
importance of multiple objectives. By using a logic where predicates correspond
directly to objectives, specifications are inherently more interpretable.
Additionally the set of specifications that can be expressed with formal
languages is a superset of what can be expressed by weight vectors. In this
paper, we define a formal language based on propositional logic with
quantitative semantics. We encode logical specifications using a recurrent
neural network and show that MORL agents parameterized by these encodings are
able to generalize to novel specifications over objectives and achieve
performance comparable to single objective baselines.

    

### [[1911.05248] What Do Compressed Deep Neural Networks Forget?](http://arxiv.org/abs/1911.05248)


  Deep neural network pruning and quantization techniques have demonstrated it
is possible to achieve high levels of compression with surprisingly little
degradation to test set accuracy. However, this measure of performance conceals
significant differences in how different classes and images are impacted by
model compression techniques. We find that models with radically different
numbers of weights have comparable top-line performance metrics but diverge
considerably in behavior on a narrow subset of the dataset. This small subset
of data points, which we term Pruning Identified Exemplars (PIEs) are
systematically more impacted by the introduction of sparsity. Compression
disproportionately impacts model performance on the underrepresented long-tail
of the data distribution. PIEs over-index on atypical or noisy images that are
far more challenging for both humans and algorithms to classify. Our work
provides intuition into the role of capacity in deep neural networks and the
trade-offs incurred by compression. An understanding of this disparate impact
is critical given the widespread deployment of compressed models in the wild.

    

### [[1912.02877] Training Agents using Upside-Down Reinforcement Learning](http://arxiv.org/abs/1912.02877)


  We develop Upside-Down Reinforcement Learning (UDRL), a method for learning
to act using only supervised learning techniques. Unlike traditional
algorithms, UDRL does not use reward prediction or search for an optimal
policy. Instead, it trains agents to follow commands such as "obtain so much
total reward in so much time." Many of its general principles are outlined in a
companion report; the goal of this paper is to develop a practical learning
algorithm and show that this conceptually simple perspective on agent training
can produce a range of rewarding behaviors for multiple episodic environments.
Experiments show that on some tasks UDRL's performance can be surprisingly
competitive with, and even exceed that of some traditional baseline algorithms
developed over decades of research. Based on these results, we suggest that
alternative approaches to expected reward maximization have an important role
to play in training useful autonomous agents.

    

### [[2001.02478] Questioning the AI: Informing Design Practices for Explainable AI User Experiences](http://arxiv.org/abs/2001.02478)


  A surge of interest in explainable AI (XAI) has led to a vast collection of
algorithmic work on the topic. While many recognize the necessity to
incorporate explainability features in AI systems, how to address real-world
user needs for understanding AI remains an open question. By interviewing 20 UX
and design practitioners working on various AI products, we seek to identify
gaps between the current XAI algorithmic work and practices to create
explainable AI products. To do so, we develop an algorithm-informed XAI
question bank in which user needs for explainability are represented as
prototypical questions users might ask about the AI, and use it as a study
probe. Our work contributes insights into the design space of XAI, informs
efforts to support design practices in this space, and identifies opportunities
for future XAI work. We also provide an extended XAI question bank and discuss
how it can be used for creating user-centered XAI.

    

### [[2002.03614] RDFFrames: Knowledge Graph Access for Machine Learning Tools](http://arxiv.org/abs/2002.03614)


  Knowledge graphs represented as RDF datasets are integral to many machine
learning applications. RDF is supported by a rich ecosystem of data management
systems and tools, most notably RDF database systems that provide a SPARQL
query interface. Surprisingly, machine learning tools for knowledge graphs do
not use SPARQL, despite the obvious advantages of using a database system. This
is due to the mismatch between SPARQL and machine learning tools in terms of
data model and programming style. Machine learning tools work on data in
tabular format and process it using an imperative programming style, while
SPARQL is declarative and has as its basic operation matching graph patterns to
RDF triples. We posit that a good interface to knowledge graphs from a machine
learning software stack should use an imperative, navigational programming
paradigm based on graph traversal rather than the SPARQL query paradigm based
on graph patterns. In this paper, we present RDFFrames, a framework that
provides such an interface. RDFFrames provides an imperative Python API that
gets internally translated to SPARQL, and it is integrated with the PyData
machine learning software stack. RDFFrames enables the user to make a sequence
of Python calls to define the data to be extracted from a knowledge graph
stored in an RDF database system, and it translates these calls into a compact
SPQARL query, executes it on the database system, and returns the results in a
standard tabular format. Thus, RDFFrames is a useful tool for data preparation
that combines the usability of PyData with the flexibility and performance of
RDF database systems.

    

### [[2002.03757] Distributed Learning with Dependent Samples](http://arxiv.org/abs/2002.03757)


  This paper focuses on learning rate analysis of distributed kernel ridge
regression for strong mixing sequences. Using a recently developed integral
operator approach and a classical covariance inequality for Banach-valued
strong mixing sequences, we succeed in deriving optimal learning rate for
distributed kernel ridge regression. As a byproduct, we also deduce a
sufficient condition for the mixing property to guarantee the optimal learning
rates for kernel ridge regression. Our results extend the applicable range of
distributed learning from i.i.d. samples to non-i.i.d. sequences.

    

### [[2002.04112] Connecting GANs, MFGs, and OT](http://arxiv.org/abs/2002.04112)


  Generative adversarial networks (GANs) have enjoyed tremendous success in
image generation and processing, and have recently attracted growing interests
in financial modelings. This paper analyzes GANs from the perspectives of
mean-field games (MFGs) and optimal transport. More specifically, from the game
theoretical perspective, GANs are interpreted as MFGs under Pareto Optimality
criterion or mean-field controls; from the optimal transport perspective, GANs
are to minimize the optimal transport cost indexed by the generator from the
known latent distribution to the unknown true distribution of data. The MFGs
perspective of GANs leads to a GAN-based computational method (MFGANs) to solve
MFGs: one neural network for the backward Hamilton-Jacobi-Bellman equation and
one neural network for the forward Fokker-Planck equation, with the two neural
networks trained in an adversarial way. Numerical experiments demonstrate
superior performance of this proposed algorithm, especially in the higher
dimensional case, when compared with existing neural network approaches.

    

### [[2002.10940] Stochastic-Sign SGD for Federated Learning with Theoretical Guarantees](http://arxiv.org/abs/2002.10940)


  Federated learning (FL) has emerged as a prominent distributed learning
paradigm. FL entails some pressing needs for developing novel parameter
estimation approaches with theoretical guarantees of convergence, which are
also communication efficient, differentially private and Byzantine resilient in
the heterogeneous data distribution settings. Quantization-based SGD solvers
have been widely adopted in FL and the recently proposed SIGNSGD with majority
vote shows a promising direction. However, no existing methods enjoy all the
aforementioned properties. In this paper, we propose an intuitively-simple yet
theoretically-sound method based on SIGNSGD to bridge the gap. We present
Stochastic-Sign SGD which utilizes novel stochastic-sign based gradient
compressors enabling the aforementioned properties in a unified framework. We
also present an error-feedback variant of the proposed Stochastic-Sign SGD
which further improves the learning performance in FL. We test the proposed
method with extensive experiments using deep neural networks on the MNIST
dataset and the CIFAR-10 dataset. The experimental results corroborate the
effectiveness of the proposed method.

    

### [[2002.11187] Analysis of Discriminator in RKHS Function Space for Kullback-Leibler Divergence Estimation](http://arxiv.org/abs/2002.11187)


  Several scalable sample-based methods to compute the Kullback Leibler (KL)
divergence between two distributions have been proposed and applied in
large-scale machine learning models. While they have been found to be unstable,
the theoretical root cause of the problem is not clear. In this paper, we study
a generative adversarial network based approach that uses a neural network
discriminator to estimate KL divergence. We argue that, in such case, high
fluctuations in the estimates are a consequence of not controlling the
complexity of the discriminator function space. We provide a theoretical
underpinning and remedy for this problem by first constructing a discriminator
in the Reproducing Kernel Hilbert Space (RKHS). This enables us to leverage
sample complexity and mean embedding to theoretically relate the error
probability bound of the KL estimates to the complexity of the discriminator in
RKHS. Based on this theory, we then present a scalable way to control the
complexity of the discriminator for a reliable estimation of KL divergence. We
support both our proposed theory and method to control the complexity of the
RKHS discriminator through controlled experiments.

    

### [[2002.11561] An Open-set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments](http://arxiv.org/abs/2002.11561)


  The problem of training with a small set of positive samples is known as
few-shot learning (FSL). It is widely known that traditional deep learning (DL)
algorithms usually show very good performance when trained with large datasets.
However, in many applications, it is not possible to obtain such a high number
of samples. In the image domain, typical FSL applications include those related
to face recognition. In the audio domain, music fraud or speaker recognition
can be clearly benefited from FSL methods. This paper deals with the
application of FSL to the detection of specific and intentional acoustic events
given by different types of sound alarms, such as door bells or fire alarms,
using a limited number of samples. These sounds typically occur in domestic
environments where many events corresponding to a wide variety of sound classes
take place. Therefore, the detection of such alarms in a practical scenario can
be considered an open-set recognition (OSR) problem. To address the lack of a
dedicated public dataset for audio FSL, researchers usually make modifications
on other available datasets. This paper is aimed at poviding the audio
recognition community with a carefully annotated dataset
(this https URL) for FSL in an OSR context comprised of 1360
clips from 34 classes divided into pattern sounds} and unwanted sounds. To
facilitate and promote research on this area, results with state-of-the-art
baseline systems based on transfer learning are also presented.

    

### [[2004.00436] Exploring Long Tail Visual Relationship Recognition with Large Vocabulary](http://arxiv.org/abs/2004.00436)


  Several approaches have been proposed in recent literature to alleviate the
long-tail problem, mainly in object classification tasks. In this paper, we
make the first large-scale study concerning the task of Long-Tail Visual
Relationship Recognition (LTVRR). LTVRR aims at improving the learning of
structured visual relationships that come from the long-tail (e.g., "rabbit
grazing on grass"). In this setup, the subject, relation, and object classes
each follow a long-tail distribution. To begin our study and make a future
benchmark for the community, we introduce two LTVRR-related benchmarks, dubbed
VG8K-LT and GQA-LT, built upon the widely used Visual Genome and GQA datasets.
We use these benchmarks to study the performance of several state-of-the-art
long-tail models on the LTVRR setup. Lastly, we propose a visiolinguistic
hubless (VilHub) loss and a Mixup augmentation technique adapted to LTVRR
setup, dubbed as RelMix. Both VilHub and RelMix can be easily integrated on top
of existing models and despite being simple, our results show that they can
remarkably improve the performance, especially on tail classes. Benchmarks,
code, and models have been made available at:
this https URL.

    

### [[2004.07351] Communication Efficient Federated Learning with Energy Awareness over Wireless Networks](http://arxiv.org/abs/2004.07351)


  In federated learning (FL), reducing the communication overhead is one of the
most critical challenges since the parameter server and the mobile devices
share the training parameters over wireless links. With such consideration, we
adopt the idea of SignSGD in which only the signs of the gradients are
exchanged. Moreover, most of the existing works assume Channel State
Information (CSI) available at both the mobile devices and the parameter
server, and thus the mobile devices can adopt fixed transmission rates dictated
by the channel capacity. In this work, only the parameter server side CSI is
assumed, and channel capacity with outage is considered. In this case, an
essential problem for the mobile devices is to select appropriate local
processing and communication parameters (including the transmission rates) to
achieve a desired balance between the overall learning performance and their
energy consumption. Two optimization problems are formulated and solved, which
optimize the learning performance given the energy consumption requirement, and
vice versa. Furthermore, considering that the data may be distributed across
the mobile devices in a highly uneven fashion in FL, a stochastic sign-based
algorithm is proposed. Extensive simulations are performed to demonstrate the
effectiveness of the proposed methods.

    

### [[2004.14724] Learning Bayesian Networks Under Sparsity Constraints: A Parameterized Complexity Analysis](http://arxiv.org/abs/2004.14724)


  We study the problem of learning the structure of an optimal Bayesian network
when additional constraints are posed on the network or on its moralized graph.
More precisely, we consider the constraint that the network or its moralized
graph are close, in terms of vertex or edge deletions, to a sparse graph class
$\Pi$. For example, we show that learning an optimal network whose moralized
graph has vertex deletion distance at most $k$ from a graph with maximum degree
1 can be computed in polynomial time when $k$ is constant. This extends
previous work that gave an algorithm with such a running time for the vertex
deletion distance to edgeless graphs [Korhonen & Parviainen, NIPS 2015]. We
then show that further extensions or improvements are presumably impossible.
For example, we show that learning optimal networks where the network or its
moralized graph have maximum degree $2$ or connected components of size at most
$c$, $c\ge 3$, is NP-hard. Finally, we show that learning an optimal network
with at most $k$ edges in the moralized graph presumably has no $f(k)\cdot
|I|^{O(1)}$-time algorithm and that, in contrast, an optimal network with at
most $k$ arcs can be computed in $2^{O(k)}\cdot |I|^{O(1)}$ time where $|I|$ is
the total input size.

    

### [[2005.04563] Efficient Privacy Preserving Edge Computing Framework for Image Classification](http://arxiv.org/abs/2005.04563)


  In order to extract knowledge from the large data collected by edge devices,
traditional cloud based approach that requires data upload may not be feasible
due to communication bandwidth limitation as well as privacy and security
concerns of end users. To address these challenges, a novel privacy preserving
edge computing framework is proposed in this paper for image classification.
Specifically, autoencoder will be trained unsupervised at each edge device
individually, then the obtained latent vectors will be transmitted to the edge
server for the training of a classifier. This framework would reduce the
communications overhead and protect the data of the end users. Comparing to
federated learning, the training of the classifier in the proposed framework
does not subject to the constraints of the edge devices, and the autoencoder
can be trained independently at each edge device without any server
involvement. Furthermore, the privacy of the end users' data is protected by
transmitting latent vectors without additional cost of encryption. Experimental
results provide insights on the image classification performance vs. various
design parameters such as the data compression ratio of the autoencoder and the
model complexity.

    

### [[2006.02585] Online mirror descent and dual averaging: keeping pace in the dynamic case](http://arxiv.org/abs/2006.02585)


  Online mirror descent (OMD) and dual averaging (DA) -- two fundamental
algorithms for online convex optimization -- are known to have very similar
(and sometimes identical) performance guarantees when used with a fixed
learning rate. Under dynamic learning rates, however, OMD is provably inferior
to DA and suffers a linear regret, even in common settings such as prediction
with expert advice. We modify the OMD algorithm through a simple technique that
we call stabilization. We give essentially the same abstract regret bound for
OMD with stabilization and for DA by modifying the classical OMD convergence
analysis in a careful and modular way that allows for straightforward and
flexible proofs. Simple corollaries of these bounds show that OMD with
stabilization and DA enjoy the same performance guarantees in many applications
-- even under dynamic learning rates. We also shed light on the similarities
between OMD and DA and show simple conditions under which stabilized-OMD and DA
generate the same iterates.

    

### [[2006.06332] A Variational Approach to Privacy and Fairness](http://arxiv.org/abs/2006.06332)


  In this article, we propose a new variational approach to learn private
and/or fair representations. This approach is based on the Lagrangians of a new
formulation of the privacy and fairness optimization problems that we propose.
In this formulation, we aim to generate representations of the data that keep a
prescribed level of the relevant information that is not shared by the private
or sensitive data, while minimizing the remaining information they keep. The
proposed approach (i) exhibits the similarities of the privacy and fairness
problems, (ii) allows us to control the trade-off between utility and privacy
or fairness through the Lagrange multiplier parameter, and (iii) can be
comfortably incorporated to common representation learning algorithms such as
the VAE, the $\beta$-VAE, the VIB, or the nonlinear IB.

    

### [[2006.13866] Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks](http://arxiv.org/abs/2006.13866)


  Sampling methods (e.g., node-wise, layer-wise, or subgraph) has become an
indispensable strategy to speed up training large-scale Graph Neural Networks
(GNNs). However, existing sampling methods are mostly based on the graph
structural information and ignore the dynamicity of optimization, which leads
to high variance in estimating the stochastic gradients. The high variance
issue can be very pronounced in extremely large graphs, where it results in
slow convergence and poor generalization. In this paper, we theoretically
analyze the variance of sampling methods and show that, due to the composite
structure of empirical risk, the variance of any sampling method can be
decomposed into \textit{embedding approximation variance} in the forward stage
and \textit{stochastic gradient variance} in the backward stage that
necessities mitigating both types of variance to obtain faster convergence
rate. We propose a decoupled variance reduction strategy that employs
(approximate) gradient information to adaptively sample nodes with minimal
variance, and explicitly reduces the variance introduced by embedding
approximation. We show theoretically and empirically that the proposed method,
even with smaller mini-batch sizes, enjoys a faster convergence rate and
entails a better generalization compared to the existing methods.

    

### [[2008.01897] Counterfactual Explanation Based on Gradual Construction for Deep Networks](http://arxiv.org/abs/2008.01897)


  To understand the black-box characteristics of deep networks, counterfactual
explanation that deduces not only the important features of an input space but
also how those features should be modified to classify input as a target class
has gained an increasing interest. The patterns that deep networks have learned
from a training dataset can be grasped by observing the feature variation among
various classes. However, current approaches perform the feature modification
to increase the classification probability for the target class irrespective of
the internal characteristics of deep networks. This often leads to unclear
explanations that deviate from real-world data distributions. To address this
problem, we propose a counterfactual explanation method that exploits the
statistics learned from a training dataset. Especially, we gradually construct
an explanation by iterating over masking and composition steps. The masking
step aims to select an important feature from the input data to be classified
as a target class. Meanwhile, the composition step aims to optimize the
previously selected feature by ensuring that its output score is close to the
logit space of the training data that are classified as the target class.
Experimental results show that our method produces human-friendly
interpretations on various classification datasets and verify that such
interpretations can be achieved with fewer feature modification.

    

### [[2009.08392] Impact and dynamics of hate and counter speech online](http://arxiv.org/abs/2009.08392)


  Citizen-generated counter speech is a promising way to fight hate speech and
promote peaceful, non-polarized discourse. However, there is a lack of
large-scale longitudinal studies of its effectiveness for reducing hate speech.
To this end, we perform an exploratory analysis of the effectiveness of counter
speech using several different macro- and micro-level measures to analyze
180,000 political conversations that took place on German Twitter over four
years. We report on the dynamic interactions of hate and counter speech over
time and provide insights into whether, as in `classic' bullying situations,
organized efforts are more effective than independent individuals in steering
online discourse. Taken together, our results build a multifaceted picture of
the dynamics of hate and counter speech online. While we make no causal claims
due to the complexity of discourse dynamics, our findings suggest that
organized hate speech is associated with changes in public discourse and that
counter speech -- especially when organized -- may help curb hateful rhetoric
in online discourse.

    

### [[2009.13233] Sense and Learn: Self-Supervision for Omnipresent Sensors](http://arxiv.org/abs/2009.13233)


  Learning general-purpose representations from multisensor data produced by
the omnipresent sensing systems (or IoT in general) has numerous applications
in diverse use cases. Existing purely supervised end-to-end deep learning
techniques depend on the availability of a massive amount of well-curated data,
acquiring which is notoriously difficult but required to achieve a sufficient
level of generalization on a task of interest. In this work, we leverage the
self-supervised learning paradigm towards realizing the vision of continual
learning from unlabeled inputs. We present a generalized framework named Sense
and Learn for representation or feature learning from raw sensory data. It
consists of several auxiliary tasks that can learn high-level and broadly
useful features entirely from unannotated data without any human involvement in
the tedious labeling process. We demonstrate the efficacy of our approach on
several publicly available datasets from different domains and in various
settings, including linear separability, semi-supervised or few shot learning,
and transfer learning. Our methodology achieves results that are competitive
with the supervised approaches and close the gap through fine-tuning a network
while learning the downstream tasks in most cases. In particular, we show that
the self-supervised network can be utilized as initialization to significantly
boost the performance in a low-data regime with as few as 5 labeled instances
per class, which is of high practical importance to real-world problems.
Likewise, the learned representations with self-supervision are found to be
highly transferable between related datasets, even when few labeled instances
are available from the target domains. The self-learning nature of our
methodology opens up exciting possibilities for on-device continual learning.

    

### [[2010.01404] Mean-Variance Efficient Reinforcement Learning by Expected Quadratic Utility Maximization](http://arxiv.org/abs/2010.01404)


  Risk management is critical in decision making, and mean-variance (MV)
trade-off is one of the most common criteria. However, in reinforcement
learning (RL) for sequential decision making under uncertainty, most of the
existing methods for MV control suffer from computational difficulties caused
by the double sampling problem. In this paper, in contrast to strict MV
control, we consider learning MV efficient policies that achieve Pareto
efficiency regarding MV trade-off. To achieve this purpose, we train an agent
to maximize the expected quadratic utility function, a common objective of risk
management in finance and economics. We call our approach direct expected
quadratic utility maximization (EQUM). The EQUM does not suffer from the double
sampling issue because it does not include gradient estimation of variance. We
confirm that the maximizer of the objective in the EQUM directly corresponds to
an MV efficient policy under a certain condition. We conduct experiments with
benchmark settings to demonstrate the effectiveness of the EQUM.

    

### [[2010.05073] Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series](http://arxiv.org/abs/2010.05073)


  Access to high-quality data repositories and benchmarks have been
instrumental in advancing the state of the art in many experimental research
domains. While advanced analytics tasks over time series data have been gaining
lots of attention, lack of such community resources severely limits scientific
progress. In this paper, we present Exathlon, the first comprehensive public
benchmark for explainable anomaly detection over high-dimensional time series
data. Exathlon has been systematically constructed based on real data traces
from repeated executions of large-scale stream processing jobs on an Apache
Spark cluster. Some of these executions were intentionally disturbed by
introducing instances of six different types of anomalous events (e.g.,
misbehaving inputs, resource contention, process failures). For each of the
anomaly instances, ground truth labels for the root cause interval as well as
those for the extended effect interval are provided, supporting the development
and evaluation of a wide range of anomaly detection (AD) and explanation
discovery (ED) tasks. We demonstrate the practical utility of Exathlon's
dataset, evaluation methodology, and end-to-end data science pipeline design
through an experimental study with three state-of-the-art AD and ED techniques.

    

### [[2010.08843] Approximate information state for approximate planning and reinforcement learning in partially observed systems](http://arxiv.org/abs/2010.08843)


  We propose a theoretical framework for approximate planning and learning in
partially observed systems. Our framework is based on the fundamental notion of
information state. We provide two equivalent definitions of information state
-- i) a function of history which is sufficient to compute the expected reward
and predict its next value; ii) equivalently, a function of the history which
can be recursively updated and is sufficient to compute the expected reward and
predict the next observation. An information state always leads to a dynamic
programming decomposition. Our key result is to show that if a function of the
history (called approximate information state (AIS)) approximately satisfies
the properties of the information state, then there is a corresponding
approximate dynamic program. We show that the policy computed using this is
approximately optimal with bounded loss of optimality. We show that several
approximations in state, observation and action spaces in literature can be
viewed as instances of AIS. In some of these cases, we obtain tighter bounds. A
salient feature of AIS is that it can be learnt from data. We present AIS based
multi-time scale policy gradient algorithms. and detailed numerical experiments
with low, moderate and high dimensional environments.

    

### [[2010.11665] Spike and slab variational Bayes for high dimensional logistic regression](http://arxiv.org/abs/2010.11665)


  Variational Bayes (VB) is a popular scalable alternative to Markov chain
Monte Carlo for Bayesian inference. We study a mean-field spike and slab VB
approximation of widely used Bayesian model selection priors in sparse
high-dimensional logistic regression. We provide non-asymptotic theoretical
guarantees for the VB posterior in both $\ell_2$ and prediction loss for a
sparse truth, giving optimal (minimax) convergence rates. Since the VB
algorithm does not depend on the unknown truth to achieve optimality, our
results shed light on effective prior choices. We confirm the improved
performance of our VB algorithm over common sparse VB approaches in a numerical
study.

    

### [[2010.16413] Artificial Intelligence (AI) in Action: Addressing the COVID-19 Pandemic with Natural Language Processing (NLP)](http://arxiv.org/abs/2010.16413)


  The COVID-19 pandemic has had a significant impact on society, both because
of the serious health effects of COVID-19 and because of public health measures
implemented to slow its spread. Many of these difficulties are fundamentally
information needs; attempts to address these needs have caused an information
overload for both researchers and the public. Natural language processing
(NLP), the branch of artificial intelligence that interprets human language,
can be applied to address many of the information needs made urgent by the
COVID-19 pandemic. This review surveys approximately 150 NLP studies and more
than 50 systems and datasets addressing the COVID-19 pandemic. We detail work
on four core NLP tasks: information retrieval, named entity recognition,
literature-based discovery, and question answering. We also describe work that
directly addresses aspects of the pandemic through four additional tasks: topic
modeling, sentiment and emotion analysis, caseload forecasting, and
misinformation detection. We conclude by discussing observable trends and
remaining challenges.

    

### [[2011.04020] High-Dimensional Sparse Linear Bandits](http://arxiv.org/abs/2011.04020)


  Stochastic linear bandits with high-dimensional sparse features are a
practical model for a variety of domains, including personalized medicine and
online advertising. We derive a novel $\Omega(n^{2/3})$ dimension-free minimax
regret lower bound for sparse linear bandits in the data-poor regime where the
horizon is smaller than the ambient dimension and where the feature vectors
admit a well-conditioned exploration distribution. This is complemented by a
nearly matching upper bound for an explore-then-commit algorithm showing that
that $\Theta(n^{2/3})$ is the optimal rate in the data-poor regime. The results
complement existing bounds for the data-rich regime and provide another example
where carefully balancing the trade-off between information and regret is
necessary. Finally, we prove a dimension-free $O(\sqrt{n})$ regret upper bound
under an additional assumption on the magnitude of the signal for relevant
features.

    

### [[2011.06806] On the stability properties of Gated Recurrent Units neural networks](http://arxiv.org/abs/2011.06806)


  The goal of this paper is to provide sufficient conditions for guaranteeing
the Input-to-State Stability (ISS) and the Incremental Input-to-State Stability
({\delta}ISS) of Gated Recurrent Units (GRUs) neural networks. These
conditions, devised for both single-layer and multi-layer architectures,
consist of nonlinear inequalities on network's weights. They can be employed to
check the stability of trained networks, or can be enforced as constraints
during the training procedure of a GRU. The resulting training procedure is
tested on a Quadruple Tank nonlinear benchmark system, showing satisfactory
modeling performances.

    

### [[2012.02972] Empirical observation of negligible fairness-accuracy trade-offs in machine learning for public policy](http://arxiv.org/abs/2012.02972)


  Growing use of machine learning in policy and social impact settings have
raised concerns for fairness implications, especially for racial minorities.
These concerns have generated considerable interest among machine learning and
artificial intelligence researchers, who have developed new methods and
established theoretical bounds for improving fairness, focusing on the source
data, regularization and model training, or post-hoc adjustments to model
scores. However, little work has studied the practical trade-offs between
fairness and accuracy in real-world settings to understand how these bounds and
methods translate into policy choices and impact on society. Our empirical
study fills this gap by investigating the impact of mitigating disparities on
accuracy, focusing on the common context of using machine learning to inform
benefit allocation in resource-constrained programs across education, mental
health, criminal justice, and housing safety. Here we describe applied work in
which we find fairness-accuracy trade-offs to be negligible in practice. In
each setting studied, explicitly focusing on achieving equity and using our
proposed post-hoc disparity mitigation methods, fairness was substantially
improved without sacrificing accuracy. This observation was robust across
policy contexts studied, scale of resources available for intervention, time,
and relative size of the protected groups. These empirical results challenge a
commonly held assumption that reducing disparities either requires accepting an
appreciable drop in accuracy or the development of novel, complex methods,
making reducing disparities in these applications more practical.

    

### [[2012.05722] Using Differentiable Programming for Flexible Statistical Modeling](http://arxiv.org/abs/2012.05722)


  Differentiable programming has recently received much interest as a paradigm
that facilitates taking gradients of computer programs. While the corresponding
flexible gradient-based optimization approaches so far have been used
predominantly for deep learning or enriching the latter with modeling
components, we want to demonstrate that they can also be useful for statistical
modeling per se, e.g., for quick prototyping when classical maximum likelihood
approaches are challenging or not feasible. In an application from a COVID-19
setting, we utilize differentiable programming to quickly build and optimize a
flexible prediction model adapted to the data quality challenges at hand.
Specifically, we develop a regression model, inspired by delay differential
equations, that can bridge temporal gaps of observations in the central German
registry of COVID-19 intensive care cases for predicting future demand. With
this exemplary modeling challenge, we illustrate how differentiable programming
can enable simple gradient-based optimization of the model by automatic
differentiation. This allowed us to quickly prototype a model under time
pressure that outperforms simpler benchmark models. We thus exemplify the
potential of differentiable programming also outside deep learning
applications, to provide more options for flexible applied statistical
modeling.

    

### [[2012.09790] Neural Radiance Flow for 4D View Synthesis and Video Processing](http://arxiv.org/abs/2012.09790)


  We present a method, Neural Radiance Flow (NeRFlow),to learn a 4D
spatial-temporal representation of a dynamic scene from a set of RGB images.
Key to our approach is the use of a neural implicit representation that learns
to capture the 3D occupancy, radiance, and dynamics of the scene. By enforcing
consistency across different modalities, our representation enables multi-view
rendering in diverse dynamic scenes, including water pouring, robotic
interaction, and real images, outperforming state-of-the-art methods for
spatial-temporal view synthesis. Our approach works even when inputs images are
captured with only one camera. We further demonstrate that the learned
representation can serve as an implicit scene prior, enabling video processing
tasks such as image super-resolution and de-noising without any additional
supervision.

    

### [[2012.11148] Efficient On-Chip Learning for Optical Neural Networks Through Power-Aware Sparse Zeroth-Order Optimization](http://arxiv.org/abs/2012.11148)


  Optical neural networks (ONNs) have demonstrated record-breaking potential in
high-performance neuromorphic computing due to their ultra-high execution speed
and low energy consumption. However, current learning protocols fail to provide
scalable and efficient solutions to photonic circuit optimization in practical
applications. In this work, we propose a novel on-chip learning framework to
release the full potential of ONNs for power-efficient in situ training.
Instead of deploying implementation-costly back-propagation, we directly
optimize the device configurations with computation budgets and power
constraints. We are the first to model the ONN on-chip learning as a
resource-constrained stochastic noisy zeroth-order optimization problem, and
propose a novel mixed-training strategy with two-level sparsity and power-aware
dynamic pruning to offer a scalable on-chip training solution in practical ONN
deployment. Compared with previous methods, we are the first to optimize over
2,500 optical components on chip. We can achieve much better optimization
stability, 3.7x-7.6x higher efficiency, and save >90% power under practical
device variations and thermal crosstalk.

    

### [[2101.02558] Using BART to Perform Pareto Optimization and Quantify its Uncertainties](http://arxiv.org/abs/2101.02558)


  Techniques to reduce the energy burden of an industrial ecosystem often
require solving a multiobjective optimization problem. However, collecting
experimental data can often be either expensive or time-consuming. In such
cases, statistical methods can be helpful. This article proposes Pareto Front
(PF) and Pareto Set (PS) estimation methods using Bayesian Additive Regression
Trees (BART), which is a non-parametric model whose assumptions are typically
less restrictive than popular alternatives, such as Gaussian Processes (GPs).
These less restrictive assumptions allow BART to handle scenarios (e.g.
high-dimensional input spaces, nonsmooth responses, large datasets) that GPs
find difficult. The performance of our BART-based method is compared to a
GP-based method using analytic test functions, demonstrating convincing
advantages. Finally, our BART-based methodology is applied to a motivating
engineering problem. Supplementary materials, which include a theorem proof,
algorithms, and R code, for this article are available online.

    

### [[2101.10423] Online Continual Learning in Image Classification: An Empirical Survey](http://arxiv.org/abs/2101.10423)


  Online continual learning for image classification studies the problem of
learning to classify images from an online stream of data and tasks, where
tasks may include new classes (class incremental) or data nonstationarity
(domain incremental). One of the key challenges of continual learning is to
avoid catastrophic forgetting (CF), i.e., forgetting old tasks in the presence
of more recent tasks. Over the past few years, many methods and tricks have
been introduced to address this problem, but many have not been fairly and
systematically compared under a variety of realistic and practical settings. To
better understand the relative advantages of various approaches and the
settings where they work best, this survey aims to (1) compare state-of-the-art
methods such as MIR, iCARL, and GDumb and determine which works best at
different experimental settings; (2) determine if the best class incremental
methods are also competitive in domain incremental setting; (3) evaluate the
performance of 7 simple but effective trick such as "review" trick and nearest
class mean (NCM) classifier to assess their relative impact. Regarding (1), we
observe iCaRL remains competitive when the memory buffer is small; GDumb
outperforms many recently proposed methods in medium-size datasets and MIR
performs the best in larger-scale datasets. For (2), we note that GDumb
performs quite poorly while MIR -- already competitive for (1) -- is also
strongly competitive in this very different but important setting. Overall,
this allows us to conclude that MIR is overall a strong and versatile method
across a wide variety of settings. For (3), we find that all 7 tricks are
beneficial, and when augmented with the "review" trick and NCM classifier, MIR
produces performance levels that bring online continual learning much closer to
its ultimate goal of matching offline training.

    

### [[2102.03746] Bacteriophage classification for assembled contigs using Graph Convolutional Network](http://arxiv.org/abs/2102.03746)


  Motivation: Bacteriophages (aka phages), which mainly infect bacteria, play
key roles in the biology of microbes. As the most abundant biological entities
on the planet, the number of discovered phages is only the tip of the iceberg.
Recently, many new phages have been revealed using high throughput sequencing,
particularly metagenomic sequencing. Compared to the fast accumulation of
phage-like sequences, there is a serious lag in taxonomic classification of
phages. High diversity, abundance, and limited known phages pose great
challenges for taxonomic analysis. In particular, alignment-based tools have
difficulty in classifying fast accumulating contigs assembled from metagenomic
data. Results: In this work, we present a novel semi-supervised learning model,
named PhaGCN, to conduct taxonomic classification for phage contigs. In this
learning model, we construct a knowledge graph by combining the DNA sequence
features learned by convolutional neural network (CNN) and protein sequence
similarity gained from gene-sharing network. Then we apply graph convolutional
network (GCN) to utilize both the labeled and unlabeled samples in training to
enhance the learning ability. We tested PhaGCN on both simulated and real
sequencing data. The results clearly show that our method competes favorably
against available phage classification tools.

    

### [[2102.10806] Provably Correct Training of Neural Network Controllers Using Reachability Analysis](http://arxiv.org/abs/2102.10806)


  In this paper, we consider the problem of training neural network (NN)
controllers for nonlinear dynamical systems that are guaranteed to satisfy
safety and liveness (e.g., reach-avoid) properties. Our approach is to combine
model-based design methodologies for dynamical systems with data-driven
approaches to achieve this target. We confine our attention to NNs with
Rectifier Linear Unit (ReLU) nonlinearity which are known to represent
Continuous Piece-Wise Affine (CPWA) functions. Given a mathematical model of
the dynamical system, we compute a finite-state abstract model that captures
the closed-loop behavior under all possible CPWA controllers. Using this
finite-state abstract model, our framework identifies a family of CPWA
functions guaranteed to satisfy the safety requirements. We augment the
learning algorithm with a NN weight projection operator during training that
enforces the resulting NN to represent a CPWA function from the provably safe
family of CPWA functions. Moreover, the proposed framework uses the
finite-state abstract model to identify candidate CPWA functions that may
satisfy the liveness properties. Using such candidate CPWA functions, the
proposed framework biases the NN training to achieve the liveness
specification. We show the efficacy of the proposed framework both in
simulation and on an actual robotic vehicle.

    

### [[2103.00542] Deep Neural Networks with ReLU-Sine-Exponential Activations Break Curse of Dimensionality on Hölder Class](http://arxiv.org/abs/2103.00542)


  In this paper, we construct neural networks with ReLU, sine and $2^x$ as
activation functions. For general continuous $f$ defined on $[0,1]^d$ with
continuity modulus $\omega_f(\cdot)$, we construct ReLU-sine-$2^x$ networks
that enjoy an approximation rate
$\mathcal{O}(\omega_f(\sqrt{d})\cdot2^{-M}+\omega_{f}\left(\frac{\sqrt{d}}{N}\right))$,
where $M,N\in \mathbb{N}^{+}$ denote the hyperparameters related to widths of
the networks. As a consequence, we can construct ReLU-sine-$2^x$ network with
the depth $5$ and width
$\max\left\{\left\lceil2d^{3/2}\left(\frac{3\mu}{\epsilon}\right)^{1/{\alpha}}\right\rceil,2\left\lceil\log_2\frac{3\mu
d^{\alpha/2}}{2\epsilon}\right\rceil+2\right\}$ that approximates $f\in
\mathcal{H}_{\mu}^{\alpha}([0,1]^d)$ within a given tolerance $\epsilon >0$
measured in $L^p$ norm $p\in[1,\infty)$, where
$\mathcal{H}_{\mu}^{\alpha}([0,1]^d)$ denotes the Hölder continuous function
class defined on $[0,1]^d$ with order $\alpha \in (0,1]$ and constant $\mu >
0$. Therefore, the ReLU-sine-$2^x$ networks overcome the curse of
dimensionality on $\mathcal{H}_{\mu}^{\alpha}([0,1]^d)$. In addition to its
supper expressive power, functions implemented by ReLU-sine-$2^x$ networks are
(generalized) differentiable, enabling us to apply SGD to train.

    

### [[2103.00674] BEAUTY Powered BEAST](http://arxiv.org/abs/2103.00674)


  We study nonparametric dependence detection with the proposed binary
expansion approximation of uniformity (BEAUTY) approach, which generalizes the
celebrated Euler's formula, and approximates the characteristic function of any
copula with a linear combination of expectations of binary interactions from
marginal binary expansions. This novel theory enables a unification of many
important tests through approximations from some quadratic forms of symmetry
statistics, where the deterministic weight matrix characterizes the power
properties of each test. To achieve a robust power, we study test statistics
with data-adaptive weights, referred to as the binary expansion adaptive
symmetry test (BEAST). By utilizing the properties of the binary expansion
filtration, we show that the Neyman-Pearson test of uniformity can be
approximated by an oracle weighted sum of symmetry statistics. The BEAST with
this oracle provides a benchmark of feasible power against any alternative by
leading all existing tests with a substantial margin. To approach this oracle
power, we develop the BEAST through a regularized resampling approximation of
the oracle test. The BEAST improves the empirical power of many existing tests
against a wide spectrum of common alternatives while providing clear
interpretation of the form of dependency upon rejection.

    

### [[2103.01511] Multi-label Classification via Adaptive Resonance Theory-based Clustering](http://arxiv.org/abs/2103.01511)


  This paper proposes a multi-label classification algorithm capable of
continual learning by applying an Adaptive Resonance Theory (ART)-based
clustering algorithm and the Bayesian approach for label probability
computation. The ART-based clustering algorithm adaptively and continually
generates prototype nodes corresponding to given data, and the generated nodes
are used as classifiers. The label probability computation independently counts
the number of label appearances for each class and calculates the Bayesian
probabilities. Thus, the label probability computation can cope with an
increase in the number of labels. Experimental results with synthetic and
real-world multi-label datasets show that the proposed algorithm has
competitive classification performance to other well-known algorithms while
realizing continual learning.

    

### [[2103.02843] Pandemic Drugs at Pandemic Speed: Infrastructure for Accelerating COVID-19 Drug Discovery with Hybrid Machine Learning- and Physics-based Simulations on High Performance Computers](http://arxiv.org/abs/2103.02843)


  The race to meet the challenges of the global pandemic has served as a
reminder that the existing drug discovery process is expensive, inefficient and
slow. There is a major bottleneck screening the vast number of potential small
molecules to shortlist lead compounds for antiviral drug development. New
opportunities to accelerate drug discovery lie at the interface between machine
learning methods, in this case developed for linear accelerators, and
physics-based methods. The two in silico methods, each have their own
advantages and limitations which, interestingly, complement each other. Here,
we present an innovative infrastructural development that combines both
approaches to accelerate drug discovery. The scale of the potential resulting
workflow is such that it is dependent on supercomputing to achieve extremely
high throughput. We have demonstrated the viability of this workflow for the
study of inhibitors for four COVID-19 target proteins and our ability to
perform the required large-scale calculations to identify lead antiviral
compounds through repurposing on a variety of supercomputers.

    

### [[2103.10901] From Static to Dynamic Prediction: Wildfire Risk Assessment Based on Multiple Environmental Factors](http://arxiv.org/abs/2103.10901)


  Wildfire is one of the biggest disasters that frequently occurs on the west
coast of the United States. Many efforts have been made to understand the
causes of the increases in wildfire intensity and frequency in recent years. In
this work, we propose static and dynamic prediction models to analyze and
assess the areas with high wildfire risks in California by utilizing a
multitude of environmental data including population density, Normalized
Difference Vegetation Index (NDVI), Palmer Drought Severity Index (PDSI), tree
mortality area, tree mortality number, and altitude. Moreover, we focus on a
better understanding of the impacts of different factors so as to inform
preventive actions. To validate our models and findings, we divide the land of
California into 4,242 grids of 0.1 degrees $\times$ 0.1 degrees in latitude and
longitude, and compute the risk of each grid based on spatial and temporal
conditions. To verify the generalizability of our models, we further expand the
scope of wildfire risk assessment from California to Washington without any
fine tuning. By performing counterfactual analysis, we uncover the effects of
several possible methods on reducing the number of high risk wildfires. Taken
together, our study has the potential to estimate, monitor, and reduce the
risks of wildfires across diverse areas provided that such environment data is
available.

    

### [[2103.13973] Learning Temporal Quantum Tomography](http://arxiv.org/abs/2103.13973)


  Quantifying and verifying the control level in preparing a quantum state are
central challenges in building quantum devices. The quantum state is
characterized from experimental measurements, using a procedure known as
tomography, which requires a vast number of resources. Furthermore, the
tomography for a quantum device with temporal processing, which is
fundamentally different from the standard tomography, has not been formulated.
We develop a practical and approximate tomography method using a recurrent
machine learning framework for this intriguing situation. The method is based
on repeated quantum interactions between a system called quantum reservoir with
a stream of quantum states. Measurement data from the reservoir are connected
to a linear readout to train a recurrent relation between quantum channels
applied to the input stream. We demonstrate our algorithms for quantum learning
tasks followed by the proposal of a quantum short-term memory capacity to
evaluate the temporal processing ability of near-term quantum devices.

    

### [[2103.14561] User-Oriented Smart General AI System under Causal Inference](http://arxiv.org/abs/2103.14561)


  General AI system solves a wide range of tasks with high performance in an
automated fashion. The best general AI algorithm designed by one individual is
different from that devised by another. The best performance records achieved
by different users are also different. An inevitable component of general AI is
tacit knowledge that depends upon user-specific comprehension of task
information and individual model design preferences that are related to user
technical experiences. Tacit knowledge affects model performance but cannot be
automatically optimized in general AI algorithms. In this paper, we propose
User-Oriented Smart General AI System under Causal Inference, abbreviated as
UOGASuCI, where UOGAS represents User-Oriented General AI System and uCI means
under the framework of causal inference. User characteristics that have a
significant influence upon tacit knowledge can be extracted from observed model
training experiences of many users in external memory modules. Under the
framework of causal inference, we manage to identify the optimal value of user
characteristics that are connected with the best model performance designed by
users. We make suggestions to users about how different user characteristics
can improve the best model performance achieved by users. By recommending
updating user characteristics associated with individualized tacit knowledge
comprehension and technical preferences, UOGAS helps users design models with
better performance.

    

### [[2103.16894] Mobility Functional Areas and COVID-19 Spread](http://arxiv.org/abs/2103.16894)


  This work introduces a new concept of functional areas called Mobility
Functional Areas (MFAs), i.e., the geographic zones highly interconnected
according to the analysis of mobile positioning data. The MFAs do not coincide
necessarily with administrative borders as they are built observing natural
human mobility and, therefore, they can be used to inform, in a bottom-up
approach, local transportation, spatial planning, health and economic policies.
After presenting the methodology behind the MFAs, this study focuses on the
link between the COVID-19 pandemic and the MFAs in Austria. It emerges that the
MFAs registered an average number of infections statistically larger than the
areas in the rest of the country, suggesting the usefulness of the MFAs in the
context of targeted re-escalation policy responses to this health crisis. The
MFAs dataset is openly available to other scholars for further analyses.

    

### [[2104.08773] Cross-Task Generalization via Natural Language Crowdsourcing Instructions](http://arxiv.org/abs/2104.08773)


  Humans (e.g., crowdworkers) have a remarkable ability in solving different
tasks, by simply reading textual instructions that define them and looking at a
few examples. NLP models built with the conventional paradigm, however, often
struggle with generalization across tasks (e.g., a question-answering system
cannot solve classification tasks). A long-standing challenge in AI is to build
a model that is equipped with the understanding of human-readable instructions
that define the tasks, and can generalize to new tasks. To study this, we
introduce NATURAL INSTRUCTIONS, a dataset of 61 distinct tasks, their
human-authored instructions and 193k task instances. The instructions are
obtained from crowdsourcing instructions used to collect existing NLP datasets
and mapped to a unified schema. We adopt generative pre-trained language models
to encode task-specific instructions along with input and generate task output.
Our results indicate that models can benefit from instructions to generalize
across tasks. These models, however, are far behind supervised task-specific
models, indicating significant room for more progress in this direction.

    

### [[2105.00636] Learning to drive from a world on rails](http://arxiv.org/abs/2105.00636)


  We learn an interactive vision-based driving policy from pre-recorded driving
logs via a model-based approach. A forward model of the world supervises a
driving policy that predicts the outcome of any potential driving trajectory.
To support learning from pre-recorded logs, we assume that the world is on
rails, meaning neither the agent nor its actions influence the environment.
This assumption greatly simplifies the learning problem, factorizing the
dynamics into a nonreactive world model and a low-dimensional and compact
forward model of the ego-vehicle. Our approach computes action-values for each
training trajectory using a tabular dynamic-programming evaluation of the
Bellman equations; these action-values in turn supervise the final vision-based
driving policy. Despite the world-on-rails assumption, the final driving policy
acts well in a dynamic and reactive world. At the time of writing, our method
ranks first on the CARLA leaderboard, attaining a 25% higher driving score
while using 40 times less data. Our method is also an order of magnitude more
sample-efficient than state-of-the-art model-free reinforcement learning
techniques on navigational tasks in the ProcGen benchmark.

    

### [[2105.10059] Model Compression](http://arxiv.org/abs/2105.10059)


  With time, machine learning models have increased in their scope,
functionality and size. Consequently, the increased functionality and size of
such models requires high-end hardware to both train and provide inference
after the fact. This paper aims to explore the possibilities within the domain
of model compression, discuss the efficiency of combining various levels of
pruning and quantization, while proposing a quality measurement metric to
objectively decide which combination is best in terms of minimizing the
accuracy delta and maximizing the size reduction factor.

    

### [[2105.13645] Learning to Select Cuts for Efficient Mixed-Integer Programming](http://arxiv.org/abs/2105.13645)


  Cutting plane methods play a significant role in modern solvers for tackling
mixed-integer programming (MIP) problems. Proper selection of cuts would remove
infeasible solutions in the early stage, thus largely reducing the
computational burden without hurting the solution accuracy. However, the major
cut selection approaches heavily rely on heuristics, which strongly depend on
the specific problem at hand and thus limit their generalization capability. In
this paper, we propose a data-driven and generalizable cut selection approach,
named Cut Ranking, in the settings of multiple instance learning. To measure
the quality of the candidate cuts, a scoring function, which takes the
instance-specific cut features as inputs, is trained and applied in cut ranking
and selection. In order to evaluate our method, we conduct extensive
experiments on both synthetic datasets and real-world datasets. Compared with
commonly used heuristics for cut selection, the learning-based policy has shown
to be more effective, and is capable of generalizing over multiple problems
with different properties. Cut Ranking has been deployed in an industrial
solver for large-scale MIPs. In the online A/B testing of the product planning
problems with more than $10^7$ variables and constraints daily, Cut Ranking has
achieved the average speedup ratio of 12.42% over the production solver without
any accuracy loss of solution.

    

### [[2106.09415] Trainable Discrete Feature Embeddings for Variational Quantum Classifier](http://arxiv.org/abs/2106.09415)


  Quantum classifiers provide sophisticated embeddings of input data in Hilbert
space promising quantum advantage. The advantage stems from quantum feature
maps encoding the inputs into quantum states with variational quantum circuits.
A recent work shows how to map discrete features with fewer quantum bits using
Quantum Random Access Coding (QRAC), an important primitive to encode binary
strings into quantum states. We propose a new method to embed discrete features
with trainable quantum circuits by combining QRAC and a recently proposed
strategy for training quantum feature map called quantum metric learning. We
show that the proposed trainable embedding requires not only as few qubits as
QRAC but also overcomes the limitations of QRAC to classify inputs whose
classes are based on hard Boolean functions. We numerically demonstrate its use
in variational quantum classifiers to achieve better performances in
classifying real-world datasets, and thus its possibility to leverage near-term
quantum computers for quantum machine learning.

    

### [[2106.11272] Neural Marching Cubes](http://arxiv.org/abs/2106.11272)


  We introduce Neural Marching Cubes (NMC), a data-driven approach for
extracting a triangle mesh from a discretized implicit field. Classical MC is
defined by coarse tessellation templates isolated to individual cubes. While
more refined tessellations have been proposed, they all make heuristic
assumptions, such as trilinearity, when determining the vertex positions and
local mesh topologies in each cube. In principle, none of these approaches can
reconstruct geometric features that reveal coherence or dependencies between
nearby cubes (e.g., a sharp edge), as such information is unaccounted for,
resulting in poor estimates of the true underlying implicit field. To tackle
these challenges, we re-cast MC from a deep learning perspective, by designing
tessellation templates more apt at preserving geometric features, and learning
the vertex positions and mesh topologies from training meshes, to account for
contextual information from nearby cubes. We develop a compact per-cube
parameterization to represent the output triangle mesh, while being compatible
with neural processing, so that a simple 3D convolutional network can be
employed for the training. We show that all topological cases in each cube that
are applicable to our design can be easily derived using our representation,
and the resulting tessellations can also be obtained naturally and efficiently
by following a few design guidelines. In addition, our network learns local
features with limited receptive fields, hence it generalizes well to new shapes
and new datasets. We evaluate our neural MC approach by quantitative and
qualitative comparisons to all well-known MC variants. In particular, we
demonstrate the ability of our network to recover sharp features such as edges
and corners, a long-standing issue of MC and its variants. Our network also
reconstructs local mesh topologies more accurately than previous approaches.

    

### [[2106.12996] Multi-Reference Alignment for sparse signals, Uniform Uncertainty Principles and the Beltway Problem](http://arxiv.org/abs/2106.12996)


  Motivated by cutting-edge applications like cryo-electron microscopy
(cryo-EM), the Multi-Reference Alignment (MRA) model entails the learning of an
unknown signal from repeated measurements of its images under the latent action
of a group of isometries and additive noise of magnitude $\sigma$. Despite
significant interest, a clear picture for understanding rates of estimation in
this model has emerged only recently, particularly in the high-noise regime
$\sigma \gg 1$ that is highly relevant in applications. Recent investigations
have revealed a remarkable asymptotic sample complexity of order $\sigma^6$ for
certain signals whose Fourier transforms have full support, in stark contrast
to the traditional $\sigma^2$ that arise in regular models. Often prohibitively
large in practice, these results have prompted the investigation of variations
around the MRA model where better sample complexity may be achieved. In this
paper, we show that \emph{sparse} signals exhibit an intermediate $\sigma^4$
sample complexity even in the classical MRA model. Our results explore and
exploit connections of the MRA estimation problem with two classical topics in
applied mathematics: the \textit{beltway problem} from combinatorial
optimization, and \textit{uniform uncertainty principles} from harmonic
analysis.

    

### [[2109.01943] Application Checkpoint and Power Study on Large Scale Systems](http://arxiv.org/abs/2109.01943)


  Power efficiency is critical in high performance computing (HPC) systems. To
achieve high power efficiency on application level, it is vital importance to
efficiently distribute power used by application checkpoints. In this study, we
analyze the relation of application checkpoints and their power consumption.
The observations could guide the design of power management.

    

### [[2109.01961] A Fully-Integrated 5mW, 0.8Gbps Energy-Efficient Chip-to-Chip Data Link for Ultra-Low-Power IoT End-Nodes in 65-nm CMOS](http://arxiv.org/abs/2109.01961)


  The increasing complexity of Internet-of-Things (IoT) applications and
near-sensor processing algorithms is pushing the computational power of
low-power, battery-operated end-node systems. This trend also reveals growing
demands for high-speed and energy-efficient inter-chip communications to manage
the increasing amount of data coming from off-chip sensors and memories. While
traditional micro-controller interfaces such as SPIs cannot cope with tight
energy and large bandwidth requirements, low-voltage swing transceivers can
tackle this challenge thanks to their capability to achieve several Gbps of the
communication speed at milliwatt power levels. However, recent research on
high-speed serial links focused on high-performance systems, with a power
consumption significantly larger than the one of low-power IoT end-nodes, or on
stand-alone designs not integrated at a system level. This paper presents a
low-swing transceiver for the energy-efficient and low power chip-to-chip
communication fully integrated within an IoT end-node System-on-Chip,
fabricated in CMOS 65nm technology. The transceiver can be easily controlled
via a software interface; thus, we can consider realistic scenarios for the
data communication, which cannot be assessed in stand-alone prototypes. Chip
measurements show that the transceiver achieves 8.46x higher energy efficiency
at 15.9x higher performance than a traditional microcontroller interface such
as a single-SPI.

    

### [[2011.14203] EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference](http://arxiv.org/abs/2011.14203)


  Transformer-based language models such as BERT provide significant accuracy
improvement for a multitude of natural language processing (NLP) tasks.
However, their hefty computational and memory demands make them challenging to
deploy to resource-constrained edge platforms with strict latency requirements.
We present EdgeBERT, an in-depth algorithm-hardware co-design for latency-aware
energy optimization for multi-task NLP. EdgeBERT employs entropy-based early
exit predication in order to perform dynamic voltage-frequency scaling (DVFS),
at a sentence granularity, for minimal energy consumption while adhering to a
prescribed target latency. Computation and memory footprint overheads are
further alleviated by employing a calibrated combination of adaptive attention
span, selective network pruning, and floating-point quantization. Furthermore,
in order to maximize the synergistic benefits of these algorithms in always-on
and intermediate edge computing settings, we specialize a 12nm scalable
hardware accelerator system, integrating a fast-switching low-dropout voltage
regulator (LDO), an all-digital phase-locked loop (ADPLL), as well as,
high-density embedded non-volatile memories (eNVMs) wherein the sparse
floating-point bit encodings of the shared multi-task parameters are carefully
stored. Altogether, latency-aware multi-task NLP inference acceleration on the
EdgeBERT hardware system generates up to 7x, 2.5x, and 53x lower energy
compared to the conventional inference without early stopping, the
latency-unbounded early exit approach, and CUDA adaptations on an Nvidia Jetson
Tegra X2 mobile GPU, respectively.

    

### [[2109.01719] Comparative Analysis for Quick Sort Algorithm under four Different Modes of Execution](http://arxiv.org/abs/2109.01719)


  This work presents a comparison for the performance of sequential sorting
algorithms under four different modes of execution, the sequential processing
mode, a conventional multi-threading implementation, multi-threading with
OpenMP Library and finally parallel processing on a super computer. Quick Sort
algorithm was selected to run the experiments performed by this effort and the
algorithm was run using different arrays sizes and different number of
processors. The results and findings were analyzed uncovering limitations as
well as enhancement potentials of sequential sorting algorithms using
parallelism.

    

### [[2109.01994] UC Modelling and Security Analysis of the Estonian IVXV Internet Voting System](http://arxiv.org/abs/2109.01994)


  Estonian Internet voting has been used in national-wide elections since 2005.
However, the system was initially designed in a heuristic manner, with very few
proven security guarantees. The Estonian Internet voting system has constantly
been evolving throughout the years, with the latest version (code-named IVXV)
implemented in 2018. Nevertheless, to date, no formal security analysis of the
system has been given. In this work, for the first time, we provide a rigorous
security modeling for the Estonian IVXV system as a ceremony, attempting to
capture the effect of actual human behavior on election verifiability in the
universal composability (UC) framework. Based on the voter behavior statistics
collected from three actual election events in Estonia, we show that IVXV
achieves end-to-end verifiability in practice despite the fact that only $4\%$
(on average) of the Estonian voters audit their ballots.

    

### [[2109.02012] Post-Quantum VRF and its Applications in Future-Proof Blockchain System](http://arxiv.org/abs/2109.02012)


  A verifiable random function (VRF in short) is a powerful pseudo-random
function that provides a non-interactively public verifiable proof for the
correctness of its output. Recently, VRFs have found essential applications in
blockchain design, such as random beacons and proof-of-stake consensus
protocols. To our knowledge, the first generation of blockchain systems used
inherently inefficient proof-of-work consensuses, and the research community
tried to achieve the same properties by proposing proof-of-stake schemes where
resource-intensive proof-of-work is emulated by cryptographic constructions.
Unfortunately, those most discussed proof-of-stake consensuses (e.g., Algorand
and Ouroborous family) are not future-proof because the building blocks are
secure only under the classical hard assumptions; in particular, their designs
ignore the advent of quantum computing and its implications. In this paper, we
propose a generic compiler to obtain the post-quantum VRF from the simple VRF
solution using symmetric-key primitives (e.g., non-interactive zero-knowledge
system) with an intrinsic property of quantum-secure. Our novel solution is
realized via two efficient zero-knowledge systems ZKBoo and ZKB++,
respectively, to validate the compiler correctness. Our proof-of-concept
implementation indicates that even today, the overheads introduced by our
solution are acceptable in real-world deployments. We also demonstrate
potential applications of a quantum-secure VRF, such as quantum-secure
decentralized random beacon and lottery-based proof of stake consensus
blockchain protocol.

    

### [[2109.02166] Assessing the Use Cases of Persistent Memory in High-Performance Scientific Computing](http://arxiv.org/abs/2109.02166)


  As the High Performance Computing world moves towards the Exa-Scale era, huge
amounts of data should be analyzed, manipulated and stored. In the traditional
storage/memory hierarchy, each compute node retains its data objects in its
local volatile DRAM. Whenever the DRAM's capacity becomes insufficient for
storing this data, the computation should either be distributed between several
compute nodes, or some portion of these data objects must be stored in a
non-volatile block device such as a hard disk drive or an SSD storage device.
Optane DataCenter Persistent Memory Module (DCPMM), a new technology introduced
by Intel, provides non-volatile memory that can be plugged into standard memory
bus slots and therefore be accessed much faster than standard storage devices.
In this work, we present and analyze the results of a comprehensive performance
assessment of several ways in which DCPMM can 1) replace standard storage
devices, and 2) replace or augment DRAM for improving the performance of HPC
scientific computations. To achieve this goal, we have configured an HPC system
such that DCPMM can service I/O operations of scientific applications, replace
standard storage devices and file systems (specifically for diagnostics and
checkpoint-restarting), and serve for expanding applications' main memory. We
focus on keeping the scientific codes with as few changes as possible, while
allowing them to access the NVM transparently as if they access persistent
storage. Our results show that DCPMM allows scientific applications to fully
utilize nodes' locality by providing them with sufficiently-large main memory.
Moreover, it can be used for providing a high-performance replacement for
persistent storage. Thus, the usage of DCPMM has the potential of replacing
standard HDD and SSD storage devices in HPC architectures and enabling a more
efficient platform for modern supercomputing applications.

    

### [[2109.02328] A Survey on Resilience in the IoT: Taxonomy, Classification and Discussion of Resilience Mechanisms](http://arxiv.org/abs/2109.02328)


  Internet-of-Things (IoT) ecosystems tend to grow both in scale and complexity
as they consist of a variety of heterogeneous devices, which span over multiple
architectural IoT layers (e.g., cloud, edge, sensors). Further, IoT systems
increasingly demand the resilient operability of services as they become part
of critical infrastructures. This leads to a broad variety of research works
that aim to increase the resilience of these systems. In this paper, we create
a systematization of knowledge about existing scientific efforts of making IoT
systems resilient. In particular, we first discuss the taxonomy and
classification of resilience and resilience mechanisms and subsequently survey
state-of-the-art resilience mechanisms that have been proposed by research work
and are applicable to IoT. As part of the survey, we also discuss questions
that focus on the practical aspects of resilience, e.g., which constraints
resilience mechanisms impose on developers when designing resilient systems by
incorporating a specific mechanism into IoT systems.

    

### [[2109.02340] Khaos: Dynamically Optimizing Checkpointing for Dependable Distributed Stream Processing](http://arxiv.org/abs/2109.02340)


  Distributed Stream Processing systems are becoming an increasingly essential
part of Big Data processing platforms as users grow ever more reliant on their
ability to provide fast access to new results. As such, making timely decisions
based on these results is dependent on a system's ability to tolerate failure.
Typically, these systems achieve fault tolerance and the ability to recover
automatically from partial failures by implementing checkpoint and rollback
recovery. However, owing to the statistical probability of partial failures
occurring in these distributed environments and the variability of workloads
upon which jobs are expected to operate, static configurations will often not
meet Quality of Service constraints with low overhead.
In this paper we present Khaos, a new approach which utilizes the parallel
processing capabilities of virtual cloud automation technologies for the
automatic runtime optimization of fault tolerance configurations in Distributed
Stream Processing jobs. Our approach employs three subsequent phases which
borrows from the principles of Chaos Engineering: establish the steady-state
processing conditions, conduct experiments to better understand how the system
performs under failure, and use this knowledge to continuously minimize Quality
of Service violations. We implemented Khaos prototypically together with Apache
Flink and demonstrate its usefulness experimentally.

    

### [[2007.01459] A New Theoretical Framework of Pyramid Markov Processes for Blockchain Selfish Mining](http://arxiv.org/abs/2007.01459)


  In this paper, we provide a new theoretical framework of pyramid Markov
processes to solve some open and fundamental problems of blockchain selfish
mining under a rigorous mathematical setting. We first describe a more general
model of blockchain selfish mining with both a two-block leading competitive
criterion and a new economic incentive mechanism. Then we establish a pyramid
Markov process and show that it is irreducible and positive recurrent, and its
stationary probability vector is matrix-geometric with an explicitly
representable rate matrix. Also, we use the stationary probability vector to
study the influence of many orphan blocks on the waste of computing resource.
Next, we set up a pyramid Markov reward process to investigate the long-run
average profits of the honest and dishonest mining pools, respectively. As a
by-product, we build three approximative Markov processes and provide some new
interesting interpretation on the Markov chain and the revenue analysis
reported in the seminal work by Eyal and Sirer (2014). Note that the pyramid
Markov (reward) processes can open up a new avenue in the study of blockchain
selfish mining. Thus we hope that the methodology and results developed in this
paper shed light on the blockchain selfish mining such that a series of
promising research can be developed potentially.

    

### [[2011.04511] Deterministic Distributed Vertex Coloring: Simpler, Faster, and without Network Decomposition](http://arxiv.org/abs/2011.04511)


  We present a simple deterministic distributed algorithm that computes a
$(\Delta+1)$-vertex coloring in $O(\log^2 \Delta \cdot \log n)$ rounds. The
algorithm can be implemented with $O(\log n)$-bit messages. The algorithm can
also be extended to the more general $(degree+1)$-list coloring problem.
Obtaining a polylogarithmic-time deterministic algorithm for
$(\Delta+1)$-vertex coloring had remained a central open question in the area
of distributed graph algorithms since the 1980s, until a recent network
decomposition algorithm of Rozhoň and Ghaffari [STOC'20]. The current state
of the art is based on an improved variant of their decomposition, which leads
to an $O(\log^5 n)$-round algorithm for $(\Delta+1)$-vertex coloring.
Our coloring algorithm is completely different and considerably simpler and
faster. It solves the coloring problem in a direct way, without using network
decomposition, by gradually rounding a certain fractional color assignment
until reaching an integral color assignments. Moreover, via the approach of
Chang, Li, and Pettie [STOC'18], this improved deterministic algorithm also
leads to an improvement in the complexity of randomized algorithms for
$(\Delta+1)$-coloring, now reaching the bound of $O(\log^3\log n)$ rounds.
As a further application, we also provide faster deterministic distributed
algorithms for the following variants of the vertex coloring problem. In graphs
of arboricity $a$, we show that a $(2+\epsilon)a$-vertex coloring can be
computed in $O(\log^3 a\cdot\log n)$ rounds. We also show that for $\Delta\geq
3$, a $\Delta$-coloring of a $\Delta$-colorable graph $G$ can be computed in
$O(\log^2 \Delta\cdot\log^2 n)$ rounds.

    

### [[2102.05743] Temporal Parallelization of Inference in Hidden Markov Models](http://arxiv.org/abs/2102.05743)


  This paper presents algorithms for parallelization of inference in hidden
Markov models (HMMs). In particular, we propose parallel backward-forward type
of filtering and smoothing algorithm as well as parallel Viterbi-type
maximum-a-posteriori (MAP) algorithm. We define associative elements and
operators to pose these inference problems as parallel-prefix-sum computations
in sum-product and max-product algorithms and parallelize them using
parallel-scan algorithms. The advantage of the proposed algorithms is that they
are computationally efficient in HMM inference problems with long time
horizons. We empirically compare the performance of the proposed methods to
classical methods on a highly parallel graphical processing unit (GPU).

    

### [[2109.01703] Will bots take over the supply chain? Revisiting Agent-based supply chain automation](http://arxiv.org/abs/2109.01703)


  Agent-based systems have the capability to fuse information from many
distributed sources and create better plans faster. This feature makes
agent-based systems naturally suitable to address the challenges in Supply
Chain Management (SCM). Although agent-based supply chains systems have been
proposed since early 2000; industrial uptake of them has been lagging. The
reasons quoted include the immaturity of the technology, a lack of
interoperability with supply chain information systems, and a lack of trust in
Artificial Intelligence (AI). In this paper, we revisit the agent-based supply
chain and review the state of the art. We find that agent-based technology has
matured, and other supporting technologies that are penetrating supply chains;
are filling in gaps, leaving the concept applicable to a wider range of
functions. For example, the ubiquity of IoT technology helps agents "sense" the
state of affairs in a supply chain and opens up new possibilities for
automation. Digital ledgers help securely transfer data between third parties,
making agent-based information sharing possible, without the need to integrate
Enterprise Resource Planning (ERP) systems. Learning functionality in agents
enables agents to move beyond automation and towards autonomy. We note this
convergence effect through conceptualising an agent-based supply chain
framework, reviewing its components, and highlighting research challenges that
need to be addressed in moving forward.

    

### [[2109.01765] Effective user intent mining with unsupervised word representation models and topic modelling](http://arxiv.org/abs/2109.01765)


  Understanding the intent behind chat between customers and customer service
agents has become a crucial problem nowadays due to an exponential increase in
the use of the Internet by people from different cultures and educational
backgrounds. More importantly, the explosion of e-commerce has led to a
significant increase in text conversation between customers and agents. In this
paper, we propose an approach to data mining the conversation intents behind
the textual data. Using the customer service data set, we train unsupervised
text representation models, and then develop an intent mapping model which
would rank the predefined intents base on cosine similarity between sentences
and intents. Topic-modeling techniques are used to define intents and domain
experts are also involved to interpret topic modelling results. With this
approach, we can get a good understanding of the user intentions behind the
unlabelled customer service textual data.

    

### [[2109.01782] Automata for dynamic answer set solving: Preliminary report](http://arxiv.org/abs/2109.01782)


  We explore different ways of implementing temporal constraints expressed in
an extension of Answer Set Programming (ASP) with language constructs from
dynamic logic. Foremost, we investigate how automata can be used for enforcing
such constraints. The idea is to transform a dynamic constraint into an
automaton expressed in terms of a logic program that enforces the satisfaction
of the original constraint. What makes this approach attractive is its
independence of time stamps and the potential to detect unsatisfiability. On
the one hand, we elaborate upon a transformation of dynamic formulas into
alternating automata that relies on meta-programming in ASP. This is the first
application of reification applied to theory expressions in gringo. On the
other hand, we propose two transformations of dynamic formulas into monadic
second-order formulas. These can then be used by off-the-shelf tools to
construct the corresponding automata. We contrast both approaches empirically
with the one of the temporal ASP solver telingo that directly maps dynamic
constraints to logic programs. Since this preliminary study is restricted to
dynamic formulas in integrity constraints, its implementations and (empirical)
results readily apply to conventional linear dynamic logic, too.

    

### [[2109.01797] Hybrid Contrastive Learning of Tri-Modal Representation for Multimodal Sentiment Analysis](http://arxiv.org/abs/2109.01797)


  The wide application of smart devices enables the availability of multimodal
data, which can be utilized in many tasks. In the field of multimodal sentiment
analysis (MSA), most previous works focus on exploring intra- and inter-modal
interactions. However, training a network with cross-modal information
(language, visual, audio) is still challenging due to the modality gap, and
existing methods still cannot ensure to sufficiently learn intra-/inter-modal
dynamics. Besides, while learning dynamics within each sample draws great
attention, the learning of inter-class relationships is neglected. Moreover,
the size of datasets limits the generalization ability of existing methods. To
address the afore-mentioned issues, we propose a novel framework HyCon for
hybrid contrastive learning of tri-modal representation. Specifically, we
simultaneously perform intra-/inter-modal contrastive learning and
semi-contrastive learning (that is why we call it hybrid contrastive learning),
with which the model can fully explore cross-modal interactions, preserve
inter-class relationships and reduce the modality gap. Besides, a refinement
term is devised to prevent the model falling into a sub-optimal solution.
Moreover, HyCon can naturally generate a large amount of training pairs for
better generalization and reduce the negative effect of limited datasets.
Extensive experiments on public datasets demonstrate that our proposed method
outperforms existing works.

    

### [[2109.01812] Stimuli-Aware Visual Emotion Analysis](http://arxiv.org/abs/2109.01812)


  Visual emotion analysis (VEA) has attracted great attention recently, due to
the increasing tendency of expressing and understanding emotions through images
on social networks. Different from traditional vision tasks, VEA is inherently
more challenging since it involves a much higher level of complexity and
ambiguity in human cognitive process. Most of the existing methods adopt deep
learning techniques to extract general features from the whole image,
disregarding the specific features evoked by various emotional stimuli.
Inspired by the \textit{Stimuli-Organism-Response (S-O-R)} emotion model in
psychological theory, we proposed a stimuli-aware VEA method consisting of
three stages, namely stimuli selection (S), feature extraction (O) and emotion
prediction (R). First, specific emotional stimuli (i.e., color, object, face)
are selected from images by employing the off-the-shelf tools. To the best of
our knowledge, it is the first time to introduce stimuli selection process into
VEA in an end-to-end network. Then, we design three specific networks, i.e.,
Global-Net, Semantic-Net and Expression-Net, to extract distinct emotional
features from different stimuli simultaneously. Finally, benefiting from the
inherent structure of Mikel's wheel, we design a novel hierarchical
cross-entropy loss to distinguish hard false examples from easy ones in an
emotion-specific manner. Experiments demonstrate that the proposed method
consistently outperforms the state-of-the-art approaches on four public visual
emotion datasets. Ablation study and visualizations further prove the validity
and interpretability of our method.

    

### [[2109.01862] Pushing Paraphrase Away from Original Sentence: A Multi-Round Paraphrase Generation Approach](http://arxiv.org/abs/2109.01862)


  In recent years, neural paraphrase generation based on Seq2Seq has achieved
superior performance, however, the generated paraphrase still has the problem
of lack of diversity. In this paper, we focus on improving the diversity
between the generated paraphrase and the original sentence, i.e., making
generated paraphrase different from the original sentence as much as possible.
We propose BTmPG (Back-Translation guided multi-round Paraphrase Generation),
which leverages multi-round paraphrase generation to improve diversity and
employs back-translation to preserve semantic information. We evaluate BTmPG on
two benchmark datasets. Both automatic and human evaluation show BTmPG can
improve the diversity of paraphrase while preserving the semantics of the
original sentence.

    

### [[2109.01879] Moving Object Detection for Event-based Vision using k-means Clustering](http://arxiv.org/abs/2109.01879)


  Moving object detection is a crucial task in computer vision. Event-based
cameras are bio-inspired cameras that work by mimicking the working of the
human eye. These cameras have multiple advantages over conventional frame-based
cameras, like reduced latency, HDR, reduced motion blur during high motion, low
power consumption, etc. However, these advantages come at a high cost, as
event-based cameras are noise sensitive and have low resolution. Moreover, the
task of moving object detection in these cameras is difficult, as event-based
sensors capture only the binary changes in brightness of a scene, lacking
useful visual features like texture and color. In this paper, we investigate
the application of the k-means clustering technique in detecting moving objects
in event-based data. Experimental results in publicly available datasets using
k-means show significant improvement in performance over the state-of-the-art
methods.

    

### [[2109.01889] Fast Image-Anomaly Mitigation for Autonomous Mobile Robots](http://arxiv.org/abs/2109.01889)


  Camera anomalies like rain or dust can severelydegrade image quality and its
related tasks, such as localizationand segmentation. In this work we address
this importantissue by implementing a pre-processing step that can
effectivelymitigate such artifacts in a real-time fashion, thus supportingthe
deployment of autonomous systems with limited computecapabilities. We propose a
shallow generator with aggregation,trained in an adversarial setting to solve
the ill-posed problemof reconstructing the occluded regions. We add an enhancer
tofurther preserve high-frequency details and image colorization.We also
produce one of the largest publicly available datasets1to train our
architecture and use realistic synthetic raindrops toobtain an improved
initialization of the model. We benchmarkour framework on existing datasets and
on our own imagesobtaining state-of-the-art results while enabling real-time
per-formance, with up to 40x faster inference time than existingapproaches.

    

### [[2109.01905] Spiking Neural Networks with Improved Inherent Recurrence Dynamics for Sequential Learning](http://arxiv.org/abs/2109.01905)


  Spiking neural networks (SNNs) with leaky integrate and fire (LIF) neurons,
can be operated in an event-driven manner and have internal states to retain
information over time, providing opportunities for energy-efficient
neuromorphic computing, especially on edge devices. Note, however, many
representative works on SNNs do not fully demonstrate the usefulness of their
inherent recurrence (membrane potentials retaining information about the past)
for sequential learning. Most of the works train SNNs to recognize static
images by artificially expanded input representation in time through rate
coding. We show that SNNs can be trained for sequential tasks and propose
modifications to a network of LIF neurons that enable internal states to learn
long sequences and make their inherent recurrence resilient to the vanishing
gradient problem. We then develop a training scheme to train the proposed SNNs
with improved inherent recurrence dynamics. Our training scheme allows spiking
neurons to produce multi-bit outputs (as opposed to binary spikes) which help
mitigate the mismatch between a derivative of spiking neurons' activation
function and a surrogate derivative used to overcome spiking neurons'
non-differentiability. Our experimental results indicate that the proposed SNN
architecture on TIMIT and LibriSpeech 100h dataset yields accuracy comparable
to that of LSTMs (within 1.10% and 0.36%, respectively), but with 2x fewer
parameters than LSTMs. The sparse SNN outputs also lead to 10.13x and 11.14x
savings in multiplication operations compared to GRUs, which is generally
con-sidered as a lightweight alternative to LSTMs, on TIMIT and LibriSpeech
100h datasets, respectively.

    

### [[2109.01954] An Exploration of Deep Learning Methods in Hungry Geese](http://arxiv.org/abs/2109.01954)


  Hungry Geese is a n-player variation of the popular game snake. This paper
looks at state of the art Deep Reinforcement Learning Value Methods. The goal
of the paper is to aggregate research of value based methods and apply it as an
exercise to other environments. A vanilla Deep Q Network, a Double Q-network
and a Dueling Q-Network were all examined and tested with the Hungry Geese
environment. The best performing model was the vanilla Deep Q Network due to
its simple state representation and smaller network structure. Converging
towards an optimal policy was found to be difficult due to random geese
initialization and food generation. Therefore we show that Deep Q Networks may
not be the appropriate model for such a stochastic environment and lastly we
present improvements that can be made along with more suitable models for the
environment.

    

### [[2109.02011] A Two-stage Complex Network using Cycle-consistent Generative Adversarial Networks for Speech Enhancement](http://arxiv.org/abs/2109.02011)


  Cycle-consistent generative adversarial networks (CycleGAN) have shown their
promising performance for speech enhancement (SE), while one intractable
shortcoming of these CycleGAN-based SE systems is that the noise components
propagate throughout the cycle and cannot be completely eliminated.
Additionally, conventional CycleGAN-based SE systems only estimate the spectral
magnitude, while the phase is unaltered. Motivated by the multi-stage learning
concept, we propose a novel two-stage denoising system that combines a
CycleGAN-based magnitude enhancing network and a subsequent complex spectral
refining network in this paper. Specifically, in the first stage, a
CycleGAN-based model is responsible for only estimating magnitude, which is
subsequently coupled with the original noisy phase to obtain a coarsely
enhanced complex spectrum. After that, the second stage is applied to further
suppress the residual noise components and estimate the clean phase by a
complex spectral mapping network, which is a pure complex-valued network
composed of complex 2D convolution/deconvolution and complex temporal-frequency
attention blocks. Experimental results on two public datasets demonstrate that
the proposed approach consistently surpasses previous one-stage CycleGANs and
other state-of-the-art SE systems in terms of various evaluation metrics,
especially in background noise suppression.

    

### [[2109.02046] Attentive Knowledge-aware Graph Convolutional Networks with Collaborative Guidance for Recommendation](http://arxiv.org/abs/2109.02046)


  To alleviate data sparsity and cold-start problems of traditional recommender
systems (RSs), incorporating knowledge graphs (KGs) to supplement auxiliary
information has attracted considerable attention recently. However, simply
integrating KGs in current KG-based RS models is not necessarily a guarantee to
improve the recommendation performance, which may even weaken the holistic
model capability. This is because the construction of these KGs is independent
of the collection of historical user-item interactions; hence, information in
these KGs may not always be helpful for recommendation to all users.
In this paper, we propose attentive Knowledge-aware Graph convolutional
networks with Collaborative Guidance for personalized Recommendation (CG-KGR).
CG-KGR is a novel knowledge-aware recommendation model that enables ample and
coherent learning of KGs and user-item interactions, via our proposed
Collaborative Guidance Mechanism. Specifically, CG-KGR first encapsulates
historical interactions to interactive information summarization. Then CG-KGR
utilizes it as guidance to extract information out of KGs, which eventually
provides more precise personalized recommendation. We conduct extensive
experiments on four real-world datasets over two recommendation tasks, i.e.,
Top-K recommendation and Click-Through rate (CTR) prediction. The experimental
results show that the CG-KGR model significantly outperforms recent
state-of-the-art models by 4.0-53.2% and 0.4-3.2%, in terms of Recall metric on
Top-K recommendation and AUC on CTR prediction, respectively.

    

### [[2109.02053] GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning](http://arxiv.org/abs/2109.02053)


  Federated Learning (FL) bridges the gap between collaborative machine
learning and preserving data privacy. To sustain the long-term operation of an
FL ecosystem, it is important to attract high quality data owners with
appropriate incentive schemes. As an important building block of such incentive
schemes, it is essential to fairly evaluate participants' contribution to the
performance of the final FL model without exposing their private data. Shapley
Value (SV)-based techniques have been widely adopted to provide fair evaluation
of FL participant contributions. However, existing approaches incur significant
computation costs, making them difficult to apply in practice. In this paper,
we propose the Guided Truncation Gradient Shapley (GTG-Shapley) approach to
address this challenge. It reconstructs FL models from gradient updates for SV
calculation instead of repeatedly training with different combinations of FL
participants. In addition, we design a guided Monte Carlo sampling approach
combined with within-round and between-round truncation to further reduce the
number of model reconstructions and evaluations required, through extensive
experiments under diverse realistic data distribution settings. The results
demonstrate that GTG-Shapley can closely approximate actual Shapley values,
while significantly increasing computational efficiency compared to the state
of the art, especially under non-i.i.d. settings.

    

### [[2109.02058] Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model](http://arxiv.org/abs/2109.02058)


  Community detection, aiming to group the graph nodes into clusters with dense
inner-connection, is a fundamental graph mining task. Recently, it has been
studied on the heterogeneous graph, which contains multiple types of nodes and
edges, posing great challenges for modeling the high-order relationship between
nodes. With the surge of graph embedding mechanism, it has also been adopted to
community detection. A remarkable group of works use the meta-path to capture
the high-order relationship between nodes and embed them into nodes' embedding
to facilitate community detection. However, defining meaningful meta-paths
requires much domain knowledge, which largely limits their applications,
especially on schema-rich heterogeneous graphs like knowledge graphs. To
alleviate this issue, in this paper, we propose to exploit the context path to
capture the high-order relationship between nodes, and build a Context
Path-based Graph Neural Network (CP-GNN) model. It recursively embeds the
high-order relationship between nodes into the node embedding with attention
mechanisms to discriminate the importance of different relationships. By
maximizing the expectation of the co-occurrence of nodes connected by context
paths, the model can learn the nodes' embeddings that both well preserve the
high-order relationship between nodes and are helpful for community detection.
Extensive experimental results on four real-world datasets show that CP-GNN
outperforms the state-of-the-art community detection methods.

    

### [[2109.02099] Knowing False Negatives: An Adversarial Training Method for Distantly Supervised Relation Extraction](http://arxiv.org/abs/2109.02099)


  Distantly supervised relation extraction (RE) automatically aligns
unstructured text with relation instances in a knowledge base (KB). Due to the
incompleteness of current KBs, sentences implying certain relations may be
annotated as N/A instances, which causes the so-called false negative (FN)
problem. Current RE methods usually overlook this problem, inducing improper
biases in both training and testing procedures. To address this issue, we
propose a two-stage approach. First, it finds out possible FN samples by
heuristically leveraging the memory mechanism of deep neural networks. Then, it
aligns those unlabeled data with the training data into a unified feature space
by adversarial training to assign pseudo labels and further utilize the
information contained in them. Experiments on two wildly-used benchmark
datasets demonstrate the effectiveness of our approach.

    

### [[2109.02102] Teaching Autoregressive Language Models Complex Tasks By Demonstration](http://arxiv.org/abs/2109.02102)


  This paper demonstrates that by fine-tuning an autoregressive language model
(GPT-Neo) on appropriately structured step-by-step demonstrations, it is
possible to teach it to execute a mathematical task that has previously proved
difficult for Transformers - longhand modulo operations - with a relatively
small number of examples. Specifically, we fine-tune GPT-Neo to solve the
numbers__div_remainder task from the DeepMind Mathematics Dataset; Saxton et
al. (arXiv:1904.01557) reported below 40% accuracy on this task with 2 million
training examples. We show that after fine-tuning on 200 appropriately
structured demonstrations of solving long division problems and reporting the
remainders, the smallest available GPT-Neo model achieves over 80% accuracy.
This is achieved by constructing an appropriate dataset for fine-tuning, with
no changes to the learning algorithm. These results suggest that fine-tuning
autoregressive language models on small sets of well-crafted demonstrations may
be a useful paradigm for enabling individuals without training in machine
learning to coax such models to perform some kinds of complex multi-step tasks.

    

### [[2109.02137] Efficient Action Recognition Using Confidence Distillation](http://arxiv.org/abs/2109.02137)


  Modern neural networks are powerful predictive models. However, when it comes
to recognizing that they may be wrong about their predictions, they perform
poorly. For example, for one of the most common activation functions, the ReLU
and its variants, even a well-calibrated model can produce incorrect but high
confidence predictions. In the related task of action recognition, most current
classification methods are based on clip-level classifiers that densely sample
a given video for non-overlapping, same-sized clips and aggregate the results
using an aggregation function - typically averaging - to achieve video level
predictions. While this approach has shown to be effective, it is sub-optimal
in recognition accuracy and has a high computational overhead. To mitigate both
these issues, we propose the confidence distillation framework to teach a
representation of uncertainty of the teacher to the student sampler and divide
the task of full video prediction between the student and the teacher models.
We conduct extensive experiments on three action recognition datasets and
demonstrate that our framework achieves significant improvements in action
recognition accuracy (up to 20%) and computational efficiency (more than 40%).

    

### [[2109.02161] Modular Framework for Visuomotor Language Grounding](http://arxiv.org/abs/2109.02161)


  Natural language instruction following tasks serve as a valuable test-bed for
grounded language and robotics research. However, data collection for these
tasks is expensive and end-to-end approaches suffer from data inefficiency. We
propose the structuring of language, acting, and visual tasks into separate
modules that can be trained independently. Using a Language, Action, and Vision
(LAV) framework removes the dependence of action and vision modules on
instruction following datasets, making them more efficient to train. We also
present a preliminary evaluation of LAV on the ALFRED task for visual and
interactive instruction following.

    

### [[2109.02194] Learning-Based Strategy Design for Robot-Assisted Reminiscence Therapy Based on a Developed Model for People with Dementia](http://arxiv.org/abs/2109.02194)


  In this paper, the robot-assisted Reminiscence Therapy (RT) is studied as a
psychosocial intervention to persons with dementia (PwDs). We aim at a
conversation strategy for the robot by reinforcement learning to stimulate the
PwD to talk. Specifically, to characterize the stochastic reactions of a PwD to
the robot's actions, a simulation model of a PwD is developed which features
the transition probabilities among different PwD states consisting of the
response relevance, emotion levels and confusion conditions. A Q-learning (QL)
algorithm is then designed to achieve the best conversation strategy for the
robot. The objective is to stimulate the PwD to talk as much as possible while
keeping the PwD's states as positive as possible. In certain conditions, the
achieved strategy gives the PwD choices to continue or change the topic, or
stop the conversation, so that the PwD has a sense of control to mitigate the
conversation stress. To achieve this, the standard QL algorithm is revised to
deliberately integrate the impact of PwD's choices into the Q-value updates.
Finally, the simulation results demonstrate the learning convergence and
validate the efficacy of the achieved strategy. Tests show that the strategy is
capable to duly adjust the difficulty level of prompt according to the PwD's
states, take actions (e.g., repeat or explain the prompt, or comfort) to help
the PwD out of bad states, and allow the PwD to control the conversation
tendency when bad states continue.

    

### [[2109.02237] BERT might be Overkill: A Tiny but Effective Biomedical Entity Linker based on Residual Convolutional Neural Networks](http://arxiv.org/abs/2109.02237)


  Biomedical entity linking is the task of linking entity mentions in a
biomedical document to referent entities in a knowledge base. Recently, many
BERT-based models have been introduced for the task. While these models have
achieved competitive results on many datasets, they are computationally
expensive and contain about 110M parameters. Little is known about the factors
contributing to their impressive performance and whether the
over-parameterization is needed. In this work, we shed some light on the inner
working mechanisms of these large BERT-based models. Through a set of probing
experiments, we have found that the entity linking performance only changes
slightly when the input word order is shuffled or when the attention scope is
limited to a fixed window size. From these observations, we propose an
efficient convolutional neural network with residual connections for biomedical
entity linking. Because of the sparse connectivity and weight sharing
properties, our model has a small number of parameters and is highly efficient.
On five public datasets, our model achieves comparable or even better linking
accuracy than the state-of-the-art BERT-based models while having about 60
times fewer parameters.

    

### [[2109.02247] STaCK: Sentence Ordering with Temporal Commonsense Knowledge](http://arxiv.org/abs/2109.02247)


  Sentence order prediction is the task of finding the correct order of
sentences in a randomly ordered document. Correctly ordering the sentences
requires an understanding of coherence with respect to the chronological
sequence of events described in the text. Document-level contextual
understanding and commonsense knowledge centered around these events are often
essential in uncovering this coherence and predicting the exact chronological
order. In this paper, we introduce STaCK -- a framework based on graph neural
networks and temporal commonsense knowledge to model global information and
predict the relative order of sentences. Our graph network accumulates temporal
evidence using knowledge of `past' and `future' and formulates sentence
ordering as a constrained edge classification problem. We report results on
five different datasets, and empirically show that the proposed method is
naturally suitable for order prediction. The implementation of this work is
publicly available at: this https URL.

    

### [[2109.02271] MONITOR: A Multimodal Fusion Framework to Assess Message Veracity in Social Networks](http://arxiv.org/abs/2109.02271)


  Users of social networks tend to post and share content with little
restraint. Hence, rumors and fake news can quickly spread on a huge scale. This
may pose a threat to the credibility of social media and can cause serious
consequences in real life. Therefore, the task of rumor detection and
verification has become extremely important. Assessing the veracity of a social
media message (e.g., by fact checkers) involves analyzing the text of the
message, its context and any multimedia attachment. This is a very
time-consuming task that can be much helped by machine learning. In the
literature, most message veracity verification methods only exploit textual
contents and metadata. Very few take both textual and visual contents, and more
particularly images, into account. In this paper, we second the hypothesis that
exploiting all of the components of a social media post enhances the accuracy
of veracity detection. To further the state of the art, we first propose using
a set of advanced image features that are inspired from the field of image
quality assessment, which effectively contributes to rumor detection. These
metrics are good indicators for the detection of fake images, even for those
generated by advanced techniques like generative adversarial networks (GANs).
Then, we introduce the Multimodal fusiON framework to assess message veracIty
in social neTwORks (MONITOR), which exploits all message features (i.e., text,
social context, and image features) by supervised machine learning. Such
algorithms provide interpretability and explainability in the decisions taken,
which we believe is particularly important in the context of rumor
verification. Experimental results show that MONITOR can detect rumors with an
accuracy of 96% and 89% on the MediaEval benchmark and the FakeNewsNet dataset,
respectively. These results are significantly better than those of
state-of-the-art machine learning baselines.

    

### [[2109.02289] Improving Numerical Reasoning Skills in the Modular Approach for Complex Question Answering on Text](http://arxiv.org/abs/2109.02289)


  Numerical reasoning skills are essential for complex question answering (CQA)
over text. It requires opertaions including counting, comparison, addition and
subtraction. A successful approach to CQA on text, Neural Module Networks
(NMNs), follows the programmer-interpreter paradigm and leverages specialised
modules to perform compositional reasoning. However, the NMNs framework does
not consider the relationship between numbers and entities in both questions
and paragraphs. We propose effective techniques to improve NMNs' numerical
reasoning capabilities by making the interpreter question-aware and capturing
the relationship between entities and numbers. On the same subset of the DROP
dataset for CQA on text, experimental results show that our additions
outperform the original NMNs by 3.0 points for the overall F1 score.

    

### [[2109.02297] Enhancing Visual Dialog Questioner with Entity-based Strategy Learning and Augmented Guesser](http://arxiv.org/abs/2109.02297)


  Considering the importance of building a good Visual Dialog (VD) Questioner,
many researchers study the topic under a Q-Bot-A-Bot image-guessing game
setting, where the Questioner needs to raise a series of questions to collect
information of an undisclosed image. Despite progress has been made in
Supervised Learning (SL) and Reinforcement Learning (RL), issues still exist.
Firstly, previous methods do not provide explicit and effective guidance for
Questioner to generate visually related and informative questions. Secondly,
the effect of RL is hampered by an incompetent component, i.e., the Guesser,
who makes image predictions based on the generated dialogs and assigns rewards
accordingly. To enhance VD Questioner: 1) we propose a Related entity enhanced
Questioner (ReeQ) that generates questions under the guidance of related
entities and learns entity-based questioning strategy from human dialogs; 2) we
propose an Augmented Guesser (AugG) that is strong and is optimized for the VD
setting especially. Experimental results on the VisDial v1.0 dataset show that
our approach achieves state-of-theart performance on both image-guessing task
and question diversity. Human study further proves that our model generates
more visually related, informative and coherent questions.

    

### [[2109.02320] LightTag: Text Annotation Platform](http://arxiv.org/abs/2109.02320)


  Text annotation tools assume that their user's goal is to create a labeled
corpus. However, users view annotation as a necessary evil on the way to
deliver business value through NLP. Thus an annotation tool should optimize for
the throughput of the global NLP process, not only the productivity of
individual annotators. LightTag is a text annotation tool designed and built on
that principle. This paper shares our design rationale, data modeling choices,
and user interface decisions then illustrates how those choices serve the full
NLP lifecycle.

    

### [[2109.02354] Method for making multi-attribute decisions in wargames by combining intuitionistic fuzzy numbers with reinforcement learning](http://arxiv.org/abs/2109.02354)


  Researchers are increasingly focusing on intelligent games as a hot research
area.The article proposes an algorithm that combines the multi-attribute
management and reinforcement learning methods, and that combined their effect
on wargaming, it solves the problem of the agent's low rate of winning against
specific rules and its inability to quickly converge during intelligent wargame
this http URL the same time, this paper studied a multi-attribute decision making
and reinforcement learning algorithm in a wargame simulation environment, and
obtained data on red and blue conflict.Calculate the weight of each attribute
based on the intuitionistic fuzzy number weight calculations. Then determine
the threat posed by each opponent's chess pieces.Using the red side
reinforcement learning reward function, the AC framework is trained on the
reward function, and an algorithm combining multi-attribute decision-making
with reinforcement learning is obtained. A simulation experiment confirms that
the algorithm of multi-attribute decision-making combined with reinforcement
learning presented in this paper is significantly more intelligent than the
pure reinforcement learning this http URL resolving the shortcomings of the
agent's neural network, coupled with sparse rewards in large-map combat games,
this robust algorithm effectively reduces the difficulties of convergence. It
is also the first time in this field that an algorithm design for intelligent
wargaming combines multi-attribute decision making with reinforcement
learning.Attempt interdisciplinary cross-innovation in the academic field, like
designing intelligent wargames and improving reinforcement learning algorithms.

    

### [[1810.05659] Mobility Offer Allocations in Corporate Settings](http://arxiv.org/abs/1810.05659)


  Corporate mobility is often based on a fixed assignment of vehicles to
employees. Relaxing this fixation and including alternatives such as public
transportation or taxis for business and private trips could increase fleet
utilization and foster the use of battery electric vehicles. We introduce the
mobility offer allocation problem as the core concept of a flexible booking
system for corporate mobility. The problem is equivalent to interval scheduling
on dedicated unrelated parallel machines. We show that the problem is NP-hard
to approximate within any factor. We describe problem specific conflict graphs
for representing and exploring the structure of feasible solutions. A
characterization of all maximum cliques in these conflict graphs reveals
symmetries which allow to formulate stronger integer linear programming models.
We also present an adaptive large neighborhood search based approach which
makes use of conflict graphs as well. In a computational study, the approaches
are evaluated. It was found that greedy heuristics perform best if very tight
run-time requirements are given, a solver for the integer linear programming
model performs best on small and medium instances, and the adaptive large
neighborhood search performs best on large instances.

    

### [[1901.06230] Computing large market equilibria using abstractions](http://arxiv.org/abs/1901.06230)


  Computing market equilibria is an important practical problem for market
design, for example in fair division of items. However, computing equilibria
requires large amounts of information (typically the valuation of every buyer
for every item) and computing power. We consider ameliorating these issues by
applying a method used for solving complex games: constructing a coarsened
abstraction of a given market, solving for the equilibrium in the abstraction,
and lifting the prices and allocations back to the original market. We show how
to bound important quantities such as regret, envy, Nash social welfare, Pareto
optimality, and maximin share/proportionality when the abstracted prices and
allocations are used in place of the real equilibrium. We then study two
abstraction methods of interest for practitioners: (1) filling in unknown
valuations using techniques from matrix completion, (2) reducing the problem
size by aggregating groups of buyers/items into smaller numbers of
representative buyers/items and solving for equilibrium in this coarsened
market. We find that in real data allocations/prices that are relatively close
to equilibria can be computed from even very coarse abstractions.

    

### [[2002.02334] Self-recognition in conversational agents](http://arxiv.org/abs/2002.02334)


  In a standard Turing test, a machine has to prove its humanness to the
judges. By successfully imitating a thinking entity such as a human, this
machine then proves that it can also think. Some objections claim that Turing
test is not a tool to demonstrate the existence of general intelligence or
thinking activity. A compelling alternative is the Lovelace test, in which the
agent must originate a product that the agent's creator cannot explain.
Therefore, the agent must be the owner of an original product. However, for
this to happen the agent must exhibit the idea of self and distinguish oneself
from others. Sustaining the idea of self within the Turing test is still
possible if the judge decides to act as a textual mirror. Self-recognition
tests applied on animals through mirrors appear to be viable tools to
demonstrate the existence of a type of general intelligence. Methodology here
constructs a textual version of the mirror test by placing the agent as the one
and only judge to figure out whether the contacted one is an other, a mimicker,
or oneself in an unsupervised manner. This textual version of the mirror test
is objective, self-contained, and devoid of humanness. Any agent passing this
textual mirror test should have or can acquire a thought mechanism that can be
referred to as the inner-voice, answering the original and long lasting
question of Turing "Can machines think?" in a constructive manner still within
the bounds of the Turing test. Moreover, it is possible that a successful
self-recognition might pave way to stronger notions of self-awareness in
artificial beings.

    

### [[2008.02025] Verifying Tight Logic Programs with anthem and Vampire](http://arxiv.org/abs/2008.02025)


  This paper continues the line of research aimed at investigating the
relationship between logic programs and first-order theories. We extend the
definition of program completion to programs with input and output in a subset
of the input language of the ASP grounder gringo, study the relationship
between stable models and completion in this context, and describe preliminary
experiments with the use of two software tools, anthem and vampire, for
verifying the correctness of programs with input and output. Proofs of theorems
are based on a lemma that relates the semantics of programs studied in this
paper to stable models of first-order formulas. Under consideration for
acceptance in TPLP.

    

### [[2008.06824] Conjunctive Queries: Unique Characterizations and Exact Learnability](http://arxiv.org/abs/2008.06824)


  We answer the question which conjunctive queries are uniquely characterized
by polynomially many positive and negative examples, and how to construct such
examples efficiently. As a consequence, we obtain a new efficient exact
learning algorithm for a class of conjunctive queries. At the core of our
contributions lie two new polynomial-time algorithms for constructing frontiers
in the homomorphism lattice of finite structures. We also discuss implications
for the unique characterizability and learnability of schema mappings and of
description logic concepts.

    

### [[2010.13319] Migratable AI : Investigating users' affect on identity and information migration of a conversational AI agent](http://arxiv.org/abs/2010.13319)


  Conversational AI agents are becoming ubiquitous and provide assistance to us
in our everyday activities. In recent years, researchers have explored the
migration of these agents across different embodiments in order to maintain the
continuity of the task and improve user experience. In this paper, we
investigate user's affective responses in different configurations of the
migration parameters. We present a 2x2 between-subjects study in a task-based
scenario using information migration and identity migration as parameters. We
outline the affect processing pipeline from the video footage collected during
the study and report user's responses in each condition. Our results show that
users reported highest joy and were most surprised when both the information
and identity was migrated; and reported most anger when the information was
migrated without the identity of their agent.

    

### [[2010.14619] Ensembles of Spiking Neural Networks](http://arxiv.org/abs/2010.14619)


  This paper demonstrates how to construct ensembles of spiking neural networks
producing state-of-the-art results, achieving classification accuracies of
98.71%, 100.0%, and 99.09%, on the MNIST, NMNIST and DVS Gesture datasets
respectively. Furthermore, this performance is achieved using simplified
individual models, with ensembles containing less than 50% of the parameters of
published reference models. We provide comprehensive exploration on the effect
of spike train interpretation methods, and derive the theoretical methodology
for combining model predictions such that performance improvements are
guaranteed for spiking ensembles. For this, we formalize spiking neural
networks as GLM predictors, identifying a suitable representation for their
target domain. Further, we show how the diversity of our spiking ensembles can
be measured using the Ambiguity Decomposition. The work demonstrates how
ensembling can overcome the challenges of producing individual SNN models which
can compete with traditional deep neural networks, and creates systems with
fewer trainable parameters and smaller memory footprints, opening the door to
low-power edge applications, e.g. implemented on neuromorphic hardware.

    

### [[2102.00311] Welfare-based Fairness through Optimization](http://arxiv.org/abs/2102.00311)


  We propose optimization as a general paradigm for formalizing welfare-based
fairness in AI systems. We argue that optimization models allow formulation of
a wide range of fairness criteria as social welfare functions, while enabling
AI to take advantage of highly advanced solution technology. In particular, we
highlight that social welfare optimization supports a broad perspective on
fairness motivated by general distributive justice considerations. We
illustrate this advantage by reviewing a collection of social welfare functions
that capture various concepts of equity. Most of these functions have tractable
optimization formulations that can be efficiently solved by state-of-the-art
methods. To further demonstrate the potentials of social welfare optimization
in AI, we show how to integrate optimization with rule-based AI and machine
learning, and outline research directions to explore for practical
implementation of integrated methods.

    

### [[2103.10069] Constructive and Toxic Speech Detection for Open-domain Social Media Comments in Vietnamese](http://arxiv.org/abs/2103.10069)


  The rise of social media has led to the increasing of comments on online
forums. However, there still exists invalid comments which are not informative
for users. Moreover, those comments are also quite toxic and harmful to people.
In this paper, we create a dataset for constructive and toxic speech detection,
named UIT-ViCTSD (Vietnamese Constructive and Toxic Speech Detection dataset)
with 10,000 human-annotated comments. For these tasks, we propose a system for
constructive and toxic speech detection with the state-of-the-art transfer
learning model in Vietnamese NLP as PhoBERT. With this system, we obtain
F1-scores of 78.59% and 59.40% for classifying constructive and toxic comments,
respectively. Besides, we implement various baseline models as traditional
Machine Learning and Deep Neural Network-Based models to evaluate the dataset.
With the results, we can solve several tasks on the online discussions and
develop the framework for identifying constructiveness and toxicity of
Vietnamese social media comments automatically.

    

### [[2104.03483] Question-Driven Design Process for Explainable AI User Experiences](http://arxiv.org/abs/2104.03483)


  A pervasive design issue of AI systems is their explainability--how to
provide appropriate information to help users understand the AI. The technical
field of explainable AI (XAI) has produced a rich toolbox of techniques.
Designers are now tasked with the challenges of how to select the most suitable
XAI techniques and translate them into UX solutions. Informed by our previous
work studying design challenges around XAI UX, this work proposes a design
process to tackle these challenges. We review our and related prior work to
identify requirements that the process should fulfill, and accordingly, propose
a Question-Driven Design Process that grounds the user needs, choices of XAI
techniques, design, and evaluation of XAI UX all in the user questions. We
provide a mapping guide between prototypical user questions and exemplars of
XAI techniques to reframe the technical space of XAI, also serving as boundary
objects to support collaboration between designers and AI engineers. We
demonstrate it with a use case of designing XAI for healthcare adverse events
prediction, and discuss lessons learned for tackling design challenges of AI
systems.

    

### [[2105.00525] Planning for Proactive Assistance in Environments with Partial Observability](http://arxiv.org/abs/2105.00525)


  This paper addresses the problem of synthesizing the behavior of an AI agent
that provides proactive task assistance to a human in settings like factory
floors where they may coexist in a common environment. Unlike in the case of
requested assistance, the human may not be expecting proactive assistance and
hence it is crucial for the agent to ensure that the human is aware of how the
assistance affects her task. This becomes harder when there is a possibility
that the human may neither have full knowledge of the AI agent's capabilities
nor have full observability of its activities. Therefore, our \textit{proactive
assistant} is guided by the following three principles: \textbf{(1)} its
activity decreases the human's cost towards her goal; \textbf{(2)} the human is
able to recognize the potential reduction in her cost; \textbf{(3)} its
activity optimizes the human's overall cost (time/resources) of achieving her
goal. Through empirical evaluation and user studies, we demonstrate the
usefulness of our approach.

    

### [[2105.08692] Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition](http://arxiv.org/abs/2105.08692)


  In real-world multi-agent systems, agents with different capabilities may
join or leave without altering the team's overarching goals. Coordinating teams
with such dynamic composition is challenging: the optimal team strategy varies
with the composition. We propose COPA, a coach-player framework to tackle this
problem. We assume the coach has a global view of the environment and
coordinates the players, who only have partial views, by distributing
individual strategies. Specifically, we 1) adopt the attention mechanism for
both the coach and the players; 2) propose a variational objective to
regularize learning; and 3) design an adaptive communication method to let the
coach decide when to communicate with the players. We validate our methods on a
resource collection task, a rescue game, and the StarCraft micromanagement
tasks. We demonstrate zero-shot generalization to new team compositions. Our
method achieves comparable or better performance than the setting where all
players have a full view of the environment. Moreover, we see that the
performance remains high even when the coach communicates as little as 13% of
the time using the adaptive communication strategy.

    

### [[2107.09045] On the Veracity of Local, Model-agnostic Explanations in Audio Classification: Targeted Investigations with Adversarial Examples](http://arxiv.org/abs/2107.09045)


  Local explanation methods such as LIME have become popular in MIR as tools
for generating post-hoc, model-agnostic explanations of a model's
classification decisions. The basic idea is to identify a small set of
human-understandable features of the classified example that are most
influential on the classifier's prediction. These are then presented as an
explanation. Evaluation of such explanations in publications often resorts to
accepting what matches the expectation of a human without actually being able
to verify if what the explanation shows is what really caused the model's
prediction. This paper reports on targeted investigations where we try to get
more insight into the actual veracity of LIME's explanations in an audio
classification task. We deliberately design adversarial examples for the
classifier, in a way that gives us knowledge about which parts of the input are
potentially responsible for the model's (wrong) prediction. Asking LIME to
explain the predictions for these adversaries permits us to study whether local
explanations do indeed detect these regions of interest. We also look at
whether LIME is more successful in finding perturbations that are more
prominent and easily noticeable for a human. Our results suggest that LIME does
not necessarily manage to identify the most relevant input features and hence
it remains unclear whether explanations are useful or even misleading.

    

### [[2104.10426] Stability and Optimization of Speculative Queueing Networks](http://arxiv.org/abs/2104.10426)


  We provide a queueing-theoretic framework for job replication schemes based
on the principle "\emph{replicate a job as soon as the system detects it as a
\emph{straggler}}". This is called job \emph{speculation}. Recent works have
analyzed {replication} on arrival, which we refer to as \emph{replication}.
Replication is motivated by its implementation in Google's BigTable. However,
systems such as Apache Spark and Hadoop MapReduce implement speculative job
execution. The performance and optimization of speculative job execution is not
well understood. To this end, we propose a queueing network model for load
balancing where each server can speculate on the execution time of a job.
Specifically, each job is initially assigned to a single server by a frontend
dispatcher. Then, when its execution begins, the server sets a timeout. If the
job completes before the timeout, it leaves the network, otherwise the job is
terminated and relaunched or resumed at another server where it will complete.
We provide a necessary and sufficient condition for the stability of
speculative queueing networks with heterogeneous servers, general job sizes and
scheduling disciplines. We find that speculation can increase the stability
region of the network when compared with standard load balancing models and
replication schemes. We provide general conditions under which timeouts
increase the size of the stability region and derive a formula for the optimal
speculation time, i.e., the timeout that minimizes the load induced through
speculation. We compare speculation with redundant-$d$ and
redundant-to-idle-queue-$d$ rules under an $S\& X$ model. For light loaded
systems, redundancy schemes provide better response times. However, for
moderate to heavy loadings, redundancy schemes can lose capacity and have
markedly worse response times when compared with a speculative scheme.

    

### [[2109.01950] Type Stability in Julia: Avoiding Performance Pathologies in JIT Compilation (Extended Version)](http://arxiv.org/abs/2109.01950)


  Performance is serious business for a scientific programming language.
Success in that niche hinges on fostering a rich ecosystem of highly optimized
mathematical libraries. The Julia language is predicated on the bet that its
users can write efficient numerical code in the language itself, without having
to resort to C or Fortran. To avoid performance pathologies, Julia programmers
strive to write code that is type stable. This paper provides a formal
definition of type stability as well as a stronger property of type
groundedness, shows that groundedness enables compiler optimizations, and
proves the compiler correct. We also perform a corpus analysis to uncover how
these type-related properties manifest in practice.

    

### [[2109.02001] Proceedings of the 9th International Workshop on Verification and Program Transformation](http://arxiv.org/abs/2109.02001)


  The previous VPT 2020 workshop was organized in honour of Professor Alberto
Pettorossi on the occasion of his academic retirement from Università di Roma
Tor Vergata. Due to the pandemic the VPT 2020 meeting was cancelled but its
proceeding have already appeared in the EPTCS 320 volume. The joint VPT-20-21
event has subsumed the original programme of VPT 2020 and provided an
opportunity to meet and celebrate the achievements of Professor Alberto
Pettorossi; its programme was further expanded with the newly submitted
presentations for VPT 2021. The aim of the VPT workshop series is to provide a
forum where people from the areas of program transformation and program
verification can fruitfully exchange ideas and gain a deeper understanding of
the interactions between those two fields.

    

### [[2109.02196] Quantum CPOs](http://arxiv.org/abs/2109.02196)


  We introduce the monoidal closed category qCPO of quantum cpos, whose objects
are "quantized" analogs of omega-complete partial orders (cpos). The category
qCPO is enriched over the category CPO of cpos, and contains both CPO, and the
opposite of the category FdAlg of finite-dimensional von Neumann algebras as
monoidal subcategories. We use qCPO to construct a sound model for the quantum
programming language Proto-Quipper-M (PQM) extended with term recursion, as
well as a sound and computationally adequate model for the Linear/Non-Linear
Fixpoint Calculus (LNL-FPC), which is both an extension of the Fixpoint
Calculus (FPC) with linear types, and an extension of a circuit-free fragment
of PQM that includes recursive types.

    

### [[2109.02197] Gottesman Types for Quantum Programs](http://arxiv.org/abs/2109.02197)


  The Heisenberg representation of quantum operators provides a powerful
technique for reasoning about quantum circuits, albeit those restricted to the
common (non-universal) Clifford set H, S and CNOT. The Gottesman-Knill theorem
showed that we can use this representation to efficiently simulate Clifford
circuits. We show that Gottesman's semantics for quantum programs can be
treated as a type system, allowing us to efficiently characterize a common
subset of quantum programs. We also show that it can be extended beyond the
Clifford set to partially characterize a broad range of programs. We apply
these types to reason about separable states and the superdense coding
algorithm.

    

### [[2109.02198] Quantum Hoare Type Theory: Extended Abstract](http://arxiv.org/abs/2109.02198)


  As quantum computers become real, it is high time we come up with effective
techniques that help programmers write correct quantum programs. In classical
computing, formal verification and sound static type systems prevent several
classes of bugs from being introduced. There is a need for similar techniques
in the quantum regime. Inspired by Hoare Type Theory in the classical paradigm,
we propose Quantum Hoare Types by extending the Quantum IO Monad by indexing it
with pre- and post-conditions that serve as program specifications. In this
paper, we introduce Quantum Hoare Type Theory (QHTT), present its syntax and
typing rules, and demonstrate its effectiveness with the help of examples.
QHTT has the potential to be a unified system for programming, specifying,
and reasoning about quantum programs. This is a work in progress.

    

### [[1911.09421] The Linear Algebra Mapping Problem. Current state of linear algebra languages and libraries](http://arxiv.org/abs/1911.09421)


  We observe a disconnect between the developers and the end users of linear
algebra libraries. On the one hand, the numerical linear algebra and the
high-performance communities invest significant effort in the development and
optimization of highly sophisticated numerical kernels and libraries, aiming at
the maximum exploitation of both the properties of the input matrices, and the
architectural features of the target computing platform. On the other hand, end
users are progressively less likely to go through the error-prone and time
consuming process of directly using said libraries by writing their code in C
or Fortran; instead, languages and libraries such as Matlab, Julia, Eigen and
Armadillo, which offer a higher level of abstraction, are becoming more and
more popular. Users are given the opportunity to code matrix computations with
a syntax that closely resembles the mathematical description; it is then a
compiler or an interpreter that internally maps the input program to lower
level kernels, as provided by libraries such as BLAS and LAPACK. Unfortunately,
our experience suggests that in terms of performance, this translation is
typically vastly suboptimal.
In this paper, we first introduce the Linear Algebra Mapping Problem, and
then investigate how effectively a benchmark of test problems is solved by
popular high-level programming languages. Specifically, we consider Matlab,
Octave, Julia, R, Armadillo (C++), Eigen (C++), and NumPy (Python); the
benchmark is meant to test both standard compiler optimizations such as common
subexpression elimination and loop-invariant code motion, as well as linear
algebra specific optimizations such as optimal parenthesization of a matrix
product and kernel selection for matrices with properties. The aim of this
study is to give concrete guidelines for the development of languages and
libraries that support linear algebra computations.

    

### [[2011.11763] The Reads-From Equivalence for the TSO and PSO Memory Models](http://arxiv.org/abs/2011.11763)


  The verification of concurrent programs remains an open challenge due to the
non-determinism in inter-process communication. One algorithmic problem in this
challenge is the consistency verification of concurrent executions. Consistency
verification under a reads-from map allows to compute the reads-from (RF)
equivalence between concurrent traces, with direct applications to areas such
as Stateless Model Checking (SMC). The RF equivalence was recently shown to be
coarser than the standard Mazurkiewicz equivalence, leading to impressive
scalability improvements for SMC under SC (sequential consistency). However,
for the relaxed memory models of TSO and PSO (total/partial store order), the
algorithmic problem of deciding the RF equivalence, as well as its impact on
SMC, has been elusive. In this work we solve the problem of consistency
verification for the TSO and PSO memory models given a reads-from map, denoted
VTSO-rf and VPSO-rf, respectively. For an execution of $n$ events over $k$
threads and $d$ variables, we establish novel bounds that scale as $n^{k+1}$
for TSO and as $n^{k+1}\cdot \min(n^{k^2}, 2^{k\cdot d})$ for PSO. Based on our
solution to these problems, we develop an SMC algorithm under TSO and PSO that
uses the RF equivalence. The algorithm is exploration-optimal, in the sense
that it is guaranteed to explore each class of the RF partitioning exactly
once, and spends polynomial time per class when $k$ is bounded. We implement
all our algorithms in the SMC tool Nidhugg, and perform a large number of
experiments over benchmarks from existing literature. Our experimental results
show that our algorithms for VTSO-rf and VPSO-rf provide significant
scalability improvements over standard alternatives. When used for SMC, the RF
partitioning is often much coarser than the standard Shasha-Snir partitioning
for TSO/PSO, which yields a significant speedup in the model checking task.

    