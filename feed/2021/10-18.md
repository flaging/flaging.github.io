
## 2021-10-18

### [[2110.07704] Uplink Power Control in Integrated Access and Backhaul Networks](http://arxiv.org/abs/2110.07704)


  Integrated access and backhaul (IAB) network is a novel radio access network
(RAN) solution, enabling network densification for 5G and beyond. In this
paper, we use power control combined with resource allocation algorithms to
develop efficient IAB networks with high service coverage. Particularly, we
develop a genetic algorithm-based solution for the power control of both user
equipments and IAB nodes such that the network uplink service coverage
probability is maximized. Finally, considering millimeter wave channel models,
we study the effect of different parameters including minimum data rate
requirement, coverage distance and transmit power on the network performance.
As we show, a power allocation schemes with well-tuned parameters can improve
the uplink performance of IAB networks considerably. Moreover, with millimeter
wave communications and a proper network deployment, the effect of interference
on the service coverage probability is negligible.

    

### [[2110.07808] Mobility Aware Edge Computing Segmentation Towards Localized Orchestration](http://arxiv.org/abs/2110.07808)


  The current trend in end-user devices' advancements in computing and
communication capabilities makes edge computing an attractive solution to pave
the way for the coveted ultra-low latency services. The success of the edge
computing networking paradigm depends on the proper orchestration of the edge
servers. Several Edge applications and services are intolerant to latency,
especially in 5G and beyond networks, such as intelligent video surveillance,
E-health, Internet of Vehicles, and augmented reality applications. The edge
devices underwent rapid growth in both capabilities and size to cope with the
service demands. Orchestrating it on the cloud was a prominent trend during the
past decade. However, the increasing number of edge devices poses a significant
burden on the orchestration delay. In addition to the growth in edge devices,
the high mobility of users renders traditional orchestration schemes
impractical for contemporary edge networks. Proper segmentation of the edge
space becomes necessary to adapt these schemes to address these challenges. In
this paper, we introduce a segmentation technique employing lax clustering and
segregated mobility-based clustering. We then apply latency mapping to these
clusters. The proposed scheme's main objective is to create subspaces
(segments) that enable light and efficient edge orchestration by reducing the
processing time and the core cloud communication overhead. A bench-marking
simulation is conducted with the results showing decreased mobility-related
failures and reduced orchestration delay.

    

### [[2110.07926] Structured Nonnegative Matrix Factorization for Traffic Flow Estimation of Large Cloud Networks](http://arxiv.org/abs/2110.07926)


  Network traffic matrix estimation is an ill-posed linear inverse problem: it
requires to estimate the unobservable origin destination traffic flows, X,
given the observable link traffic flows, Y, and a binary routing matrix, A,
which are such that Y = AX. This is a challenging but vital problem as accurate
estimation of OD flows is required for several network management tasks. In
this paper, we propose a novel model for the network traffic matrix estimation
problem which maps high-dimension OD flows to low-dimension latent flows with
the following three constraints: (1) nonnegativity constraint on the estimated
OD flows, (2) autoregression constraint that enables the proposed model to
effectively capture temporal patterns of the OD flows, and (3) orthogonality
constraint that ensures the mapping between low-dimensional latent flows and
the corresponding link flows to be distance preserving. The parameters of the
proposed model are estimated with a training algorithm based on Nesterov
accelerated gradient and generally shows fast convergence. We validate the
proposed traffic flow estimation model on two real backbone IP network
datasets, namely Internet2 and G'EANT. Empirical results show that the proposed
model outperforms the state-of-the-art models not only in terms of tracking the
individual OD flows but also in terms of standard performance metrics. The
proposed model is also found to be highly scalable compared to the existing
state-of-the-art approaches.

    

### [[2110.07954] HTTPA: HTTPS Attestable Protocol](http://arxiv.org/abs/2110.07954)


  Hypertext Transfer Protocol Secure (HTTPS) protocol has become integral part
of the modern internet technology. It is currently the primary protocol for
commercialized web applications. It can provide a fast, secure connection with
a certain level of privacy and integrity, and it has become a basic assumption
on most web services on the internet. However, HTTPS cannot provide security
assurances on the request data in compute, so the computing environment remains
uncertain risks and vulnerabilities. A hardware-based trusted execution
environment (TEE) such as Intel Software Guard Extension (SGX) provides
in-memory encryption to help protect the runtime computation to reduce risks of
illegal leaking or modifying private information. The central concept of SGX
enables the computation happening inside the enclave, a protected environment
that encrypts the codes and data pertaining to a security-sensitive
computation. In addition, SGX provides provide security assurances via remote
attestation to the web client, including TCB identity, vendor identity and
verification identity. Here we propose a HTTP protocol, called HTTPS Attestable
(HTTPA), by including remote attestation process onto the HTTPS protocol to
address the privacy and security concerns on web and the access over the
Internet. With HTTPA, we can provide security assurances to establish
trustworthiness with web services and ensure integrity of request handling for
web users. We expect that remote attestation will become a new trend adopted to
reduce web services security risks, and propose the HTTPA protocol to unify the
web attestation and accessing services in a standard and efficient way.

    

### [[2110.07964] Federated Route Leak Detection in Inter-domain Routing with Privacy Guarantee](http://arxiv.org/abs/2110.07964)


  In the inter-domain network, a route leak occurs when a routing announcement
is propagated outside of its intended scope, which is a violation of the agreed
routing policy. The route leaks can disrupt the internet traffic and cause
large outages. The accurately detection of route leaks requires the share of AS
business relationship information of ASes. However, the business relationship
information between ASes is confidential due to economic issues. Thus, ASes are
usually unwilling to revealing this information to the other ASes, especially
their competitors. Recent advancements in federated learning make it possible
to share data while maintaining privacy. Motivated by this, in this paper we
study the route leak problem by considering the privacy of business
relationships between ASes, and propose a method for route leak detection with
privacy guarantee by using blockchain-based federated learning framework, in
which ASes can train a global detection model without revealing their business
relationships directly. Moreover, the proposed method provides a
self-validation scheme by labeling AS triples with local routing policies,
which mitigates route leaks' lack of ground truth. We evaluate the proposed
method under a variety of datasets including unbalanced and balanced datasets.
The different deployment strategies of the proposed method under different
topologies are also examined. The results show that the proposed method has a
better performance in detecting route leaks than a single AS detection
regardless of whether using balanced or unbalanced datasets. In the analysis of
the deployment, the results show that ASes with more peers have more possible
route leaks and can contribute more on the detection of route leaks with the
proposed method.

    

### [[2110.08031] Multi-layer Space Information Networks: Access Design and Softwarization](http://arxiv.org/abs/2110.08031)


  In this paper, we propose an approach for constructing a multi-layer
multi-orbit space information network (SIN) to provide high-speed continuous
broadband connectivity for space missions (nanosatellite terminals) from the
emerging space-based Internet providers. This notion has been motivated by the
rapid developments in satellite technologies in terms of satellite
miniaturization and reusable rocket launch, as well as the increased number of
nanosatellite constellations in lower orbits for space downstream applications,
such as earth observation, remote sensing, and Internet of Things (IoT) data
collection. Specifically, space-based Internet providers, such as Starlink,
OneWeb, and SES O3b, can be utilized for broadband connectivity directly
to/from the nanosatellites, which allows a larger degree of connectivity in
space network topologies. Besides, this kind of establishment is more
economically efficient and eliminates the need for an excessive number of
ground stations while achieving real-time and reliable space communications.
This objective necessitates developing suitable radio access schemes and
efficient scalable space backhauling using inter-satellite links (ISLs) and
inter-orbit links (IOLs). Particularly, service-oriented radio access methods
in addition to software-defined networking (SDN)-based architecture employing
optimal routing mechanisms over multiple ISLs and IOLs are the most essential
enablers for this novel concept. Thus, developing this symbiotic interaction
between versatile satellite nodes across different orbits will lead to a
breakthrough in the way that future downstream space missions and satellite
networks are designed and operated.

    

### [[2012.15545] Vehicular Network Slicing for Reliable Access and Deadline-Constrained Data Offloading: A Multi-Agent On-Device Learning Approach](http://arxiv.org/abs/2012.15545)


  Efficient data offloading plays a pivotal role in computational-intensive
platforms as data rate over wireless channels is fundamentally limited. On top
of that, high mobility adds an extra burden in vehicular edge networks (VENs),
bolstering the desire for efficient user-centric solutions. Therefore, unlike
the legacy inflexible network-centric approach, this paper exploits a
software-defined flexible, open, and programmable networking platform for an
efficient user-centric, fast, reliable, and deadline-constrained offloading
solution in VENs. In the proposed model, each active vehicle user (VU) is
served from multiple low-powered access points (APs) by creating a noble
virtual cell (VC). A joint node association, power allocation, and distributed
resource allocation problem is formulated. As centralized learning is not
practical in many real-world problems, following the distributed nature of
autonomous VUs, each VU is considered an edge learning agent. To that end,
considering practical location-aware node associations, a joint radio and power
resource allocation non-cooperative stochastic game is formulated. Leveraging
reinforcement learning's (RL) efficacy, a multi-agent RL (MARL) solution is
proposed where the edge learners aim to learn the Nash equilibrium (NE)
strategies to solve the game efficiently. Besides, real-world map data, with a
practical microscopic mobility model, are used for the simulation. Results
suggest that the proposed user-centric approach can deliver remarkable
performances in VENs. Moreover, the proposed MARL solution delivers
near-optimal performances with approximately 3% collision probabilities in case
of distributed random access in the uplink.

    

### [[2104.09202] Monitoring Data Requests in Decentralized Data Storage Systems: A Case Study of IPFS](http://arxiv.org/abs/2104.09202)


  Decentralized data storage systems like the Interplanetary Filesystem (IPFS)
are becoming increasingly popular, e. g., as a data layer in blockchain
applications and for sharing content in a censorship-resistant manner. In IPFS,
data is hosted by an open set of peers, requests to which are broadcast to all
directly connected peers and routed via a distributed hash table (DHT). In this
paper, we showcase how the monitoring of said data requests allows for profound
insights about the IPFS network while simultaneously breaching individual
users' privacy. To this end, we present a passive monitoring methodology that
enables us to collect data requests of a significant and upscalable portion of
the total IPFS node population. Using a measurement setup implementing our
approach and data collected over a period of fifteen months, we demonstrate the
estimation of, among other things: the size of the IPFS network, activity
levels and structure, and content popularity distributions. We furthermore
present how our methodology can be abused for attacks on users' privacy. As a
demonstration, we identify and successfully surveil public IPFS/HTTP gateways,
thereby also uncovering their (normally hidden) node identifiers. We find that
the number of requests by public gateways is substantial, suggesting
substantial usage of these gateways. We give a detailed analysis of the
mechanics and reasons behind implied privacy threats and discuss possible
countermeasures.

    

### [[2110.07618] Sparse Implicit Processes for Approximate Inference](http://arxiv.org/abs/2110.07618)


  Implicit Processes (IPs) are flexible priors that can describe models such as
Bayesian neural networks, neural samplers and data generators. IPs allow for
approximate inference in function-space. This avoids some degenerate problems
of parameter-space approximate inference due to the high number of parameters
and strong dependencies. For this, an extra IP is often used to approximate the
posterior of the prior IP. However, simultaneously adjusting the parameters of
the prior IP and the approximate posterior IP is a challenging task. Existing
methods that can tune the prior IP result in a Gaussian predictive
distribution, which fails to capture important data patterns. By contrast,
methods producing flexible predictive distributions by using another IP to
approximate the posterior process cannot fit the prior IP to the observed data.
We propose here a method that can carry out both tasks. For this, we rely on an
inducing-point representation of the prior IP, as often done in the context of
sparse Gaussian processes. The result is a scalable method for approximate
inference with IPs that can tune the prior IP parameters to the data, and that
provides accurate non-Gaussian predictive distributions.

    

### [[2110.07631] More Efficient Sampling for Tensor Decomposition](http://arxiv.org/abs/2110.07631)


  Recent papers have developed alternating least squares (ALS) methods for CP
and tensor ring decomposition with a per-iteration cost which is sublinear in
the number of input tensor entries for low-rank decomposition. However, the
per-iteration cost of these methods still has an exponential dependence on the
number of tensor modes. In this paper, we propose sampling-based ALS methods
for the CP and tensor ring decompositions whose cost does not have this
exponential dependence, thereby significantly improving on the previous
state-of-the-art. We provide a detailed theoretical analysis and also apply the
methods in a feature extraction experiment.

    

### [[2110.07636] A Survey of Machine Learning Algorithms for Detecting Ransomware Encryption Activity](http://arxiv.org/abs/2110.07636)


  A survey of machine learning techniques trained to detect ransomware is
presented. This work builds upon the efforts of Taylor et al. in using
sensor-based methods that utilize data collected from built-in instruments like
CPU power and temperature monitors to identify encryption activity. Exploratory
data analysis (EDA) shows the features most useful from this simulated data are
clock speed, temperature, and CPU load. These features are used in training
multiple algorithms to determine an optimal detection approach. Performance is
evaluated with accuracy, F1 score, and false-negative rate metrics. The
Multilayer Perceptron with three hidden layers achieves scores of 97% in
accuracy and F1 and robust data preparation. A random forest model produces
scores of 93% accuracy and 92% F1, showing that sensor-based detection is
currently a viable option to detect even zero-day ransomware attacks before the
code fully executes.

    

### [[2110.07647] Towards Understanding the Data Dependency of Mixup-style Training](http://arxiv.org/abs/2110.07647)


  In the Mixup training paradigm, a model is trained using convex combinations
of data points and their associated labels. Despite seeing very few true data
points during training, models trained using Mixup seem to still minimize the
original empirical risk and exhibit better generalization and robustness on
various tasks when compared to standard training. In this paper, we investigate
how these benefits of Mixup training rely on properties of the data in the
context of classification. For minimizing the original empirical risk, we
compute a closed form for the Mixup-optimal classification, which allows us to
construct a simple dataset on which minimizing the Mixup loss can provably lead
to learning a classifier that does not minimize the empirical loss on the data.
On the other hand, we also give sufficient conditions for Mixup training to
also minimize the original empirical risk. For generalization, we characterize
the margin of a Mixup classifier, and use this to understand why the decision
boundary of a Mixup classifier can adapt better to the full structure of the
training data when compared to standard training. In contrast, we also show
that, for a large class of linear models and linearly separable datasets, Mixup
training leads to learning the same classifier as standard training.

    

### [[2110.07649] Federated learning and next generation wireless communications: A survey on bidirectional relationship](http://arxiv.org/abs/2110.07649)


  In order to meet the extremely heterogeneous requirements of the next
generation wireless communication networks, research community is increasingly
dependent on using machine learning solutions for real-time decision-making and
radio resource management. Traditional machine learning employs fully
centralized architecture in which the entire training data is collected at one
node e.g., cloud server, that significantly increases the communication
overheads and also raises severe privacy concerns. Towards this end, a
distributed machine learning paradigm termed as Federated learning (FL) has
been proposed recently. In FL, each participating edge device trains its local
model by using its own training data. Then, via the wireless channels the
weights or parameters of the locally trained models are sent to the central PS,
that aggregates them and updates the global model. On one hand, FL plays an
important role for optimizing the resources of wireless communication networks,
on the other hand, wireless communications is crucial for FL. Thus, a
`bidirectional' relationship exists between FL and wireless communications.
Although FL is an emerging concept, many publications have already been
published in the domain of FL and its applications for next generation wireless
networks. Nevertheless, we noticed that none of the works have highlighted the
bidirectional relationship between FL and wireless communications. Therefore,
the purpose of this survey paper is to bridge this gap in literature by
providing a timely and comprehensive discussion on the interdependency between
FL and wireless communications.

    

### [[2110.07654] Residual2Vec: Debiasing graph embedding with random graphs](http://arxiv.org/abs/2110.07654)


  Graph embedding maps a graph into a convenient vector-space representation
for graph analysis and machine learning applications. Many graph embedding
methods hinge on a sampling of context nodes based on random walks. However,
random walks can be a biased sampler due to the structural properties of
graphs. Most notably, random walks are biased by the degree of each node, where
a node is sampled proportionally to its degree. The implication of such biases
has not been clear, particularly in the context of graph representation
learning. Here, we investigate the impact of the random walks' bias on graph
embedding and propose residual2vec, a general graph embedding method that can
debias various structural biases in graphs by using random graphs. We
demonstrate that this debiasing not only improves link prediction and
clustering performance but also allows us to explicitly model salient
structural properties in graph embedding.

    

### [[2110.07658] Predicting Solar Flares with Remote Sensing and Machine Learning](http://arxiv.org/abs/2110.07658)


  High energy solar flares and coronal mass ejections have the potential to
destroy Earth's ground and satellite infrastructures, causing trillions of
dollars in damage and mass human suffering. Destruction of these critical
systems would disable power grids and satellites, crippling communications and
transportation. This would lead to food shortages and an inability to respond
to emergencies. A solution to this impending problem is proposed herein using
satellites in solar orbit that continuously monitor the Sun, use artificial
intelligence and machine learning to calculate the probability of massive solar
explosions from this sensed data, and then signal defense mechanisms that will
mitigate the threat. With modern technology there may be only safeguards that
can be implemented with enough warning, which is why the best algorithm must be
identified and continuously trained with existing and new data to maximize true
positive rates while minimizing false negatives. This paper conducts a survey
of current machine learning models using open source solar flare prediction
data. The rise of edge computing allows machine learning hardware to be placed
on the same satellites as the sensor arrays, saving critical time by not having
to transmit remote sensing data across the vast distances of space. A system of
systems approach will allow enough warning for safety measures to be put into
place mitigating the risk of disaster.

    

### [[2110.07660] A Semi-Supervised Approach for Abnormal Event Prediction on Large Operational Network Time-Series Data](http://arxiv.org/abs/2110.07660)


  Large network logs, recording multivariate time series generated from
heterogeneous devices and sensors in a network, can often reveal important
information about abnormal activities, such as network intrusions and device
malfunctions. Existing machine learning methods for anomaly detection on
multivariate time series typically assume that 1) normal sequences would have
consistent behavior for training unsupervised models, or 2) require a large set
of labeled normal and abnormal sequences for supervised models. However, in
practice, normal network activities can demonstrate significantly varying
sequence patterns (e.g., before and after rerouting partial network traffic).
Also, the recorded abnormal events can be sparse. This paper presents a novel
semi-supervised method that efficiently captures dependencies between network
time series and across time points to generate meaningful representations of
network activities for predicting abnormal events. The method can use the
limited labeled data to explicitly learn separable embedding space for normal
and abnormal samples and effectively leverage unlabeled data to handle training
data scarcity. The experiments demonstrate that our approach significantly
outperformed state-of-the-art approaches for event detection on a large
real-world network log.

    

### [[2110.07661] Distribution-Free Federated Learning with Conformal Predictions](http://arxiv.org/abs/2110.07661)


  Federated learning has attracted considerable interest for collaborative
machine learning in healthcare to leverage separate institutional datasets
while maintaining patient privacy.
However, additional challenges such as poor calibration and lack of
interpretability may also hamper widespread deployment of federated models into
clinical practice and lead to user distrust or misuse of ML tools in
high-stakes clinical decision-making.
In this paper, we propose to address these challenges by incorporating an
adaptive conformal framework into federated learning to ensure
distribution-free prediction sets that provide coverage guarantees and
uncertainty estimates without requiring any additional modifications to the
model or assumptions.
Empirical results on the MedMNIST medical imaging benchmark demonstrate our
federated method provide tighter coverage in lower average cardinality over
local conformal predictions on 6 different medical imaging benchmark datasets
in 2D and 3D multi-class classification tasks.
Further, we correlate class entropy and prediction set size to assess task
uncertainty with conformal methods.

    

### [[2110.07682] Sound and Complete Neural Network Repair with Minimality and Locality Guarantees](http://arxiv.org/abs/2110.07682)


  We present a novel methodology for repairing neural networks that use ReLU
activation functions. Unlike existing methods that rely on modifying the
weights of a neural network which can induce a global change in the function
space, our approach applies only a localized change in the function space while
still guaranteeing the removal of the buggy behavior. By leveraging the
piecewise linear nature of ReLU networks, our approach can efficiently
construct a patch network tailored to the linear region where the buggy input
resides, which when combined with the original network, provably corrects the
behavior on the buggy input. Our method is both sound and complete -- the
repaired network is guaranteed to fix the buggy input, and a patch is
guaranteed to be found for any buggy input. Moreover, our approach preserves
the continuous piecewise linear nature of ReLU networks, automatically
generalizes the repair to all the points including other undetected buggy
inputs inside the repair region, is minimal in terms of changes in the function
space, and guarantees that outputs on inputs away from the repair region are
unaltered. On several benchmarks, we show that our approach significantly
outperforms existing methods in terms of locality and limiting negative side
effects.

    

### [[2110.07683] An Optimization Perspective on Realizing Backdoor Injection Attacks on Deep Neural Networks in Hardware](http://arxiv.org/abs/2110.07683)


  State-of-the-art deep neural networks (DNNs) have been proven to be
vulnerable to adversarial manipulation and backdoor attacks. Backdoored models
deviate from expected behavior on inputs with predefined triggers while
retaining performance on clean data. Recent works focus on software simulation
of backdoor injection during the inference phase by modifying network weights,
which we find often unrealistic in practice due to the hardware restriction
such as bit allocation in memory. In contrast, in this work, we investigate the
viability of backdoor injection attacks in real-life deployments of DNNs on
hardware and address such practical issues in hardware implementation from a
novel optimization perspective. We are motivated by the fact that the
vulnerable memory locations are very rare, device-specific, and sparsely
distributed. Consequently, we propose a novel network training algorithm based
on constrained optimization for realistic backdoor injection attack in
hardware. By modifying parameters uniformly across the convolutional and
fully-connected layers as well as optimizing the trigger pattern together, we
achieve the state-of-the-art attack performance with fewer bit flips. For
instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10
can achieve over 91% test accuracy and 94% attack success rate by flipping only
10 bits out of 2.2 million bits.

    

### [[2110.07699] Safety-aware Policy Optimisation for Autonomous Racing](http://arxiv.org/abs/2110.07699)


  To be viable for safety-critical applications, such as autonomous driving and
assistive robotics, autonomous agents should adhere to safety constraints
throughout the interactions with their environments. Instead of learning about
safety by collecting samples, including unsafe ones, methods such as
Hamilton-Jacobi (HJ) reachability compute safe sets with theoretical guarantees
using models of the system dynamics. However, HJ reachability is not scalable
to high-dimensional systems, and the guarantees hinge on the quality of the
model. In this work, we inject HJ reachability theory into the constrained
Markov decision process (CMDP) framework, as a control-theoretical approach for
safety analysis via model-free updates on state-action pairs. Furthermore, we
demonstrate that the HJ safety value can be learned directly on vision context,
the highest-dimensional problem studied via the method to-date. We evaluate our
method on several benchmark tasks, including Safety Gym and Learn-to-Race
(L2R), a recently-released high-fidelity autonomous racing environment. Our
approach has significantly fewer constraint violations in comparison to other
constrained RL baselines, and achieve the new state-of-the-art results on the
L2R benchmark task.

    

### [[2110.07700] Hindsight Network Credit Assignment: Efficient Credit Assignment in Networks of Discrete Stochastic Units](http://arxiv.org/abs/2110.07700)


  Training neural networks with discrete stochastic variables presents a unique
challenge. Backpropagation is not directly applicable, nor are the
reparameterization tricks used in networks with continuous stochastic
variables. To address this challenge, we present Hindsight Network Credit
Assignment (HNCA), a novel learning algorithm for networks of discrete
stochastic units. HNCA works by assigning credit to each unit based on the
degree to which its output influences its immediate children in the network. We
prove that HNCA produces unbiased gradient estimates with reduced variance
compared to the REINFORCE estimator, while the computational cost is similar to
that of backpropagation. We first apply HNCA in a contextual bandit setting to
optimize a reward function that is unknown to the agent. In this setting, we
empirically demonstrate that HNCA significantly outperforms REINFORCE,
indicating that the variance reduction implied by our theoretical analysis is
significant and impactful. We then show how HNCA can be extended to optimize a
more general function of the outputs of a network of stochastic units, where
the function is known to the agent. We apply this extended version of HNCA to
train a discrete variational auto-encoder and empirically show it compares
favourably to other strong methods. We believe that the ideas underlying HNCA
can help stimulate new ways of thinking about efficient credit assignment in
stochastic compute graphs.

    

### [[2110.07701] Exposing Query Identification for Search Transparency](http://arxiv.org/abs/2110.07701)


  Search systems control the exposure of ranked content to searchers. In many
cases, creators value not only the exposure of their content but, moreover, an
understanding of the specific searches where the content is surfaced. The
problem of identifying which queries expose a given piece of content in the
ranking results is an important and relatively under-explored search
transparency challenge. Exposing queries are useful for quantifying various
issues of search bias, privacy, data protection, security, and search engine
optimization.
Exact identification of exposing queries in a given system is computationally
expensive, especially in dynamic contexts such as web search. In quest of a
more lightweight solution, we explore the feasibility of approximate exposing
query identification (EQI) as a retrieval task by reversing the role of queries
and documents in two classes of search systems: dense dual-encoder models and
traditional BM25 models. We then propose how this approach can be improved
through metric learning over the retrieval embedding space. We further derive
an evaluation metric to measure the quality of a ranking of exposing queries,
as well as conducting an empirical analysis focusing on various practical
aspects of approximate EQI.

    

### [[2110.07719] Certified Patch Robustness via Smoothed Vision Transformers](http://arxiv.org/abs/2110.07719)


  Certified patch defenses can guarantee robustness of an image classifier to
arbitrary changes within a bounded contiguous region. But, currently, this
robustness comes at a cost of degraded standard accuracies and slower inference
times. We demonstrate how using vision transformers enables significantly
better certified patch robustness that is also more computationally efficient
and does not incur a substantial drop in standard accuracy. These improvements
stem from the inherent ability of the vision transformer to gracefully handle
largely masked images. Our code is available at
this https URL.

    

### [[2110.07728] Pre-training Molecular Graph Representation with 3D Geometry](http://arxiv.org/abs/2110.07728)


  Molecular graph representation learning is a fundamental problem in modern
drug and material discovery. Molecular graphs are typically modeled by their 2D
topological structures, but it has been recently discovered that 3D geometric
information plays a more vital role in predicting molecular functionalities.
However, the lack of 3D information in real-world scenarios has significantly
impeded the learning of geometric graph representation. To cope with this
challenge, we propose the Graph Multi-View Pre-training (GraphMVP) framework
where self-supervised learning (SSL) is performed by leveraging the
correspondence and consistency between 2D topological structures and 3D
geometric views. GraphMVP effectively learns a 2D molecular graph encoder that
is enhanced by richer and more discriminative 3D geometry. We further provide
theoretical insights to justify the effectiveness of GraphMVP. Finally,
comprehensive experiments show that GraphMVP can consistently outperform
existing graph SSL methods.

    

### [[2110.07731] CCQA: A New Web-Scale Question Answering Dataset for Model Pre-Training](http://arxiv.org/abs/2110.07731)


  With the rise of large-scale pre-trained language models, open-domain
question-answering (ODQA) has become an important research topic in NLP. Based
on the popular pre-training fine-tuning approach, we posit that an additional
in-domain pre-training stage using a large-scale, natural, and diverse
question-answering (QA) dataset can be beneficial for ODQA. Consequently, we
propose a novel QA dataset based on the Common Crawl project in this paper.
Using the readily available this http URL annotation, we extract around 130
million multilingual question-answer pairs, including about 60 million English
data-points. With this previously unseen number of natural QA pairs, we
pre-train popular language models to show the potential of large-scale
in-domain pre-training for the task of question-answering. In our experiments,
we find that pre-training question-answering models on our Common Crawl
Question Answering dataset (CCQA) achieves promising results in zero-shot, low
resource and fine-tuned settings across multiple tasks, models and benchmarks.

    

### [[2110.07732] The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization](http://arxiv.org/abs/2110.07732)


  Despite successes across a broad range of applications, Transformers have
limited success in systematic generalization. The situation is especially
frustrating in the case of algorithmic tasks, where they often fail to find
intuitive solutions that route relevant information to the right node/operation
at the right time in the grid represented by Transformer columns. To facilitate
the learning of useful control flow, we propose two modifications to the
Transformer architecture, copy gate and geometric attention. Our novel Neural
Data Router (NDR) achieves 100% length generalization accuracy on the classic
compositional table lookup task, as well as near-perfect accuracy on the simple
arithmetic task and a new variant of ListOps testing for generalization across
computational depth. NDR's attention and gating patterns tend to be
interpretable as an intuitive form of neural routing. Our code is public.

    

### [[2110.07735] Continual Learning on Noisy Data Streams via Self-Purified Replay](http://arxiv.org/abs/2110.07735)


  Continually learning in the real world must overcome many challenges, among
which noisy labels are a common and inevitable issue. In this work, we present
a repla-ybased continual learning framework that simultaneously addresses both
catastrophic forgetting and noisy labels for the first time. Our solution is
based on two observations; (i) forgetting can be mitigated even with noisy
labels via self-supervised learning, and (ii) the purity of the replay buffer
is crucial. Building on this regard, we propose two key components of our
method: (i) a self-supervised replay technique named Self-Replay which can
circumvent erroneous training signals arising from noisy labeled data, and (ii)
the Self-Centered filter that maintains a purified replay buffer via
centrality-based stochastic graph ensembles. The empirical results on MNIST,
CIFAR-10, CIFAR-100, and WebVision with real-world noise demonstrate that our
framework can maintain a highly pure replay buffer amidst noisy streamed data
while greatly outperforming the combinations of the state-of-the-art continual
learning and noisy label learning methods. The source code is available at
this http URL


### [[2110.07739] Model-Change Active Learning in Graph-Based Semi-Supervised Learning](http://arxiv.org/abs/2110.07739)


  Active learning in semi-supervised classification involves introducing
additional labels for unlabelled data to improve the accuracy of the underlying
classifier. A challenge is to identify which points to label to best improve
performance while limiting the number of new labels. "Model-change" active
learning quantifies the resulting change incurred in the classifier by
introducing the additional label(s). We pair this idea with graph-based
semi-supervised learning methods, that use the spectrum of the graph Laplacian
matrix, which can be truncated to avoid prohibitively large computational and
storage costs. We consider a family of convex loss functions for which the
acquisition function can be efficiently approximated using the Laplace
approximation of the posterior distribution. We show a variety of multiclass
examples that illustrate improved performance over prior state-of-art.

    

### [[2110.07749] Attention-Free Keyword Spotting](http://arxiv.org/abs/2110.07749)


  Till now, attention-based models have been used with great success in the
keyword spotting problem domain. However, in light of recent advances in deep
learning, the question arises whether self-attention is truly irreplaceable for
recognizing speech keywords. We thus explore the usage of gated MLPs --
previously shown to be alternatives to transformers in vision tasks -- for the
keyword spotting task. We verify our approach on the Google Speech Commands
V2-35 dataset and show that it is possible to obtain performance comparable to
the state of the art without any apparent usage of self-attention.

    

### [[2110.07751] Leveraging Spatial and Temporal Correlations in Sparsified Mean Estimation](http://arxiv.org/abs/2110.07751)


  We study the problem of estimating at a central server the mean of a set of
vectors distributed across several nodes (one vector per node). When the
vectors are high-dimensional, the communication cost of sending entire vectors
may be prohibitive, and it may be imperative for them to use sparsification
techniques. While most existing work on sparsified mean estimation is agnostic
to the characteristics of the data vectors, in many practical applications such
as federated learning, there may be spatial correlations (similarities in the
vectors sent by different nodes) or temporal correlations (similarities in the
data sent by a single node over different iterations of the algorithm) in the
data vectors. We leverage these correlations by simply modifying the decoding
method used by the server to estimate the mean. We provide an analysis of the
resulting estimation error as well as experiments for PCA, K-Means and Logistic
Regression, which show that our estimators consistently outperform more
sophisticated and expensive sparsification methods.

    

### [[2110.07756] Learning Mean-Field Equations from Particle Data Using WSINDy](http://arxiv.org/abs/2110.07756)


  We develop a weak-form sparse identification method for interacting particle
systems (IPS) with the primary goals of reducing computational complexity for
large particle number $N$ and offering robustness to either intrinsic or
extrinsic noise. In particular, we use concepts from mean-field theory of IPS
in combination with the weak-form sparse identification of nonlinear dynamics
algorithm (WSINDy) to provide a fast and reliable system identification scheme
for recovering the governing stochastic differential equations for an IPS when
the number of particles per experiment $N$ is on the order of several thousand
and the number of experiments $M$ is less than 100. This is in contrast to
existing work showing that system identification for $N$ less than 100 and $M$
on the order of several thousand is feasible using strong-form methods. We
prove that under some standard regularity assumptions the scheme converges with
rate $\mathcal{O}(N^{-1/2})$ in the ordinary least squares setting and we
demonstrate the convergence rate numerically on several systems in one and two
spatial dimensions. Our examples include a canonical problem from
homogenization theory (as a first step towards learning coarse-grained models),
the dynamics of an attractive-repulsive swarm, and the IPS description of the
parabolic-elliptic Keller-Segel model for chemotaxis.

    

### [[2110.07773] Areas on the space of smooth probability density functions on $S^2$](http://arxiv.org/abs/2110.07773)


  We present symbolic and numerical methods for computing Poisson brackets on
the spaces of measures with positive densities of the plane, the 2-torus, and
the 2-sphere. We apply our methods to compute symplectic areas of finite
regions for the case of the 2-sphere, including an explicit example for
Gaussian measures with positive densities.

    

### [[2110.07775] Creating User Interface Mock-ups from High-Level Text Descriptions with Deep-Learning Models](http://arxiv.org/abs/2110.07775)


  The design process of user interfaces (UIs) often begins with articulating
high-level design goals. Translating these high-level design goals into
concrete design mock-ups, however, requires extensive effort and UI design
expertise. To facilitate this process for app designers and developers, we
introduce three deep-learning techniques to create low-fidelity UI mock-ups
from a natural language phrase that describes the high-level design goal (e.g.
"pop up displaying an image and other options"). In particular, we contribute
two retrieval-based methods and one generative method, as well as
pre-processing and post-processing techniques to ensure the quality of the
created UI mock-ups. We quantitatively and qualitatively compare and contrast
each method's ability in suggesting coherent, diverse and relevant UI design
mock-ups. We further evaluate these methods with 15 professional UI designers
and practitioners to understand each method's advantages and disadvantages. The
designers responded positively to the potential of these methods for assisting
the design process.

    

### [[2110.07778] NeuroView: Explainable Deep Network Decision Making](http://arxiv.org/abs/2110.07778)


  Deep neural networks (DNs) provide superhuman performance in numerous
computer vision tasks, yet it remains unclear exactly which of a DN's units
contribute to a particular decision. NeuroView is a new family of DN
architectures that are interpretable/explainable by design. Each member of the
family is derived from a standard DN architecture by vector quantizing the unit
output values and feeding them into a global linear classifier. The resulting
architecture establishes a direct, causal link between the state of each unit
and the classification decision. We validate NeuroView on standard datasets and
classification tasks to show that how its unit/class mapping aids in
understanding the decision-making process.

    

### [[2110.07785] Scalable Causal Structure Learning: New Opportunities in Biomedicine](http://arxiv.org/abs/2110.07785)


  This paper gives a practical tutorial on popular causal structure learning
models with examples of real-world data to help healthcare audiences understand
and apply them. We review prominent traditional, score-based and
machine-learning based schemes for causal structure discovery, study some of
their performance over some benchmark datasets, and discuss some of the
applications to biomedicine. In the case of sufficient data, machine
learning-based approaches can be scalable, can include a greater number of
variables than traditional approaches, and can potentially be applied in many
biomedical applications.

    

### [[2110.07786] Learning the Koopman Eigendecomposition: A Diffeomorphic Approach](http://arxiv.org/abs/2110.07786)


  We present a novel data-driven approach for learning linear representations
of a class of stable nonlinear systems using Koopman eigenfunctions. By
learning the conjugacy map between a nonlinear system and its Jacobian
linearization through a Normalizing Flow one can guarantee the learned function
is a diffeomorphism. Using this diffeomorphism, we construct eigenfunctions of
the nonlinear system via the spectral equivalence of conjugate systems -
allowing the construction of linear predictors for nonlinear systems. The
universality of the diffeomorphism learner leads to the universal approximation
of the nonlinear system's Koopman eigenfunctions. The developed method is also
safe as it guarantees the model is asymptotically stable regardless of the
representation accuracy. To our best knowledge, this is the first work to close
the gap between the operator, system and learning theories. The efficacy of our
approach is shown through simulation examples.

    

### [[2110.07788] Gaussian Process Bandit Optimization with Few Batches](http://arxiv.org/abs/2110.07788)


  In this paper, we consider the problem of black-box optimization using
Gaussian Process (GP) bandit optimization with a small number of batches.
Assuming the unknown function has a low norm in the Reproducing Kernel Hilbert
Space (RKHS), we introduce a batch algorithm inspired by batched finite-arm
bandit algorithms, and show that it achieves the cumulative regret upper bound
$O^\ast(\sqrt{T\gamma_T})$ using $O(\log\log T)$ batches within time horizon
$T$, where the $O^\ast(\cdot)$ notation hides dimension-independent logarithmic
factors and $\gamma_T$ is the maximum information gain associated with the
kernel. This bound is near-optimal for several kernels of interest and improves
on the typical $O^\ast(\sqrt{T}\gamma_T)$ bound, and our approach is arguably
the simplest among algorithms attaining this improvement. In addition, in the
case of a constant number of batches (not depending on $T$), we propose a
modified version of our algorithm, and characterize how the regret is impacted
by the number of batches, focusing on the squared exponential and Matérn
kernels. The algorithmic upper bounds are shown to be nearly minimax optimal
via analogous algorithm-independent lower bounds.

    

### [[2110.07801] Adversarial Purification through Representation Disentanglement](http://arxiv.org/abs/2110.07801)


  Deep learning models are vulnerable to adversarial examples and make
incomprehensible mistakes, which puts a threat on their real-world deployment.
Combined with the idea of adversarial training, preprocessing-based defenses
are popular and convenient to use because of their task independence and good
generalizability. Current defense methods, especially purification, tend to
remove ``noise" by learning and recovering the natural images. However,
different from random noise, the adversarial patterns are much easier to be
overfitted during model training due to their strong correlation to the images.
In this work, we propose a novel adversarial purification scheme by presenting
disentanglement of natural images and adversarial perturbations as a
preprocessing defense. With extensive experiments, our defense is shown to be
generalizable and make significant protection against unseen strong adversarial
attacks. It reduces the success rates of state-of-the-art \textbf{ensemble}
attacks from \textbf{61.7\%} to \textbf{14.9\%} on average, superior to a
number of existing methods. Notably, our defense restores the perturbed images
perfectly and does not hurt the clean accuracy of backbone models, which is
highly desirable in practice.

    

### [[2110.07807] Provable Regret Bounds for Deep Online Learning and Control](http://arxiv.org/abs/2110.07807)


  The use of deep neural networks has been highly successful in reinforcement
learning and control, although few theoretical guarantees for deep learning
exist for these problems. There are two main challenges for deriving
performance guarantees: a) control has state information and thus is inherently
online and b) deep networks are non-convex predictors for which online learning
cannot provide provable guarantees in general.
Building on the linearization technique for overparameterized neural
networks, we derive provable regret bounds for efficient online learning with
deep neural networks. Specifically, we show that over any sequence of convex
loss functions, any low-regret algorithm can be adapted to optimize the
parameters of a neural network such that it competes with the best net in
hindsight. As an application of these results in the online setting, we obtain
provable bounds for online episodic control with deep neural network
controllers.

    

### [[2110.07809] PTQ-SL: Exploring the Sub-layerwise Post-training Quantization](http://arxiv.org/abs/2110.07809)


  Network quantization is a powerful technique to compress convolutional neural
networks. The quantization granularity determines how to share the scaling
factors in weights, which affects the performance of network quantization. Most
existing approaches share the scaling factors layerwisely or channelwisely for
quantization of convolutional layers. Channelwise quantization and layerwise
quantization have been widely used in various applications. However, other
quantization granularities are rarely explored. In this paper, we will explore
the sub-layerwise granularity that shares the scaling factor across multiple
input and output channels. We propose an efficient post-training quantization
method in sub-layerwise granularity (PTQ-SL). Then we systematically experiment
on various granularities and observe that the prediction accuracy of the
quantized neural network has a strong correlation with the granularity.
Moreover, we find that adjusting the position of the channels can improve the
performance of sub-layerwise quantization. Therefore, we propose a method to
reorder the channels for sub-layerwise quantization. The experiments
demonstrate that the sub-layerwise quantization with appropriate channel
reordering can outperform the channelwise quantization.

    

### [[2110.07810] Towards Statistical and Computational Complexities of Polyak Step Size Gradient Descent](http://arxiv.org/abs/2110.07810)


  We study the statistical and computational complexities of the Polyak step
size gradient descent algorithm under generalized smoothness and Lojasiewicz
conditions of the population loss function, namely, the limit of the empirical
loss function when the sample size goes to infinity, and the stability between
the gradients of the empirical and population loss functions, namely, the
polynomial growth on the concentration bound between the gradients of sample
and population loss functions. We demonstrate that the Polyak step size
gradient descent iterates reach a final statistical radius of convergence
around the true parameter after logarithmic number of iterations in terms of
the sample size. It is computationally cheaper than the polynomial number of
iterations on the sample size of the fixed-step size gradient descent algorithm
to reach the same final statistical radius when the population loss function is
not locally strongly convex. Finally, we illustrate our general theory under
three statistical examples: generalized linear model, mixture model, and mixed
linear regression model.

    

### [[2110.07814] Meta-learning via Language Model In-context Tuning](http://arxiv.org/abs/2110.07814)


  The goal of meta-learning is to learn to adapt to a new task with only a few
labeled examples. To tackle this problem in NLP, we propose $\textit{in-context
tuning}$, which recasts adaptation and prediction as a simple sequence
prediction problem: to form the input sequence, we concatenate the task
instruction, the labeled examples, and the target input to predict; to
meta-train the model to learn from in-context examples, we fine-tune a
pre-trained language model (LM) to predict the target label from the input
sequences on a collection of tasks.
We benchmark our method on two collections of text classification tasks: LAMA
and BinaryClfs. Compared to first-order MAML which adapts the model with
gradient descent, our method better leverages the inductive bias of LMs to
perform pattern matching, and outperforms MAML by an absolute $6\%$ AUC ROC
score on BinaryClfs, with increasing advantage w.r.t. model size. Compared to
non-fine-tuned in-context learning (i.e. prompting a raw LM), in-context tuning
directly learns to learn from in-context examples. On BinaryClfs, in-context
tuning improves the average AUC-ROC score by an absolute $10\%$, and reduces
the variance with respect to example ordering by 6x and example choices by 2x.

    

### [[2110.07821] Gait-based Frailty Assessment using Image Representation of IMU Signals and Deep CNN](http://arxiv.org/abs/2110.07821)


  Frailty is a common and critical condition in elderly adults, which may lead
to further deterioration of health. However, difficulties and complexities
exist in traditional frailty assessments based on activity-related
questionnaires. These can be overcome by monitoring the effects of frailty on
the gait. In this paper, it is shown that by encoding gait signals as images,
deep learning-based models can be utilized for the classification of gait type.
Two deep learning models (a) SS-CNN, based on single stride input images, and
(b) MS-CNN, based on 3 consecutive strides were proposed. It was shown that
MS-CNN performs best with an accuracy of 85.1\%, while SS-CNN achieved an
accuracy of 77.3\%. This is because MS-CNN can observe more features
corresponding to stride-to-stride variations which is one of the key symptoms
of frailty. Gait signals were encoded as images using STFT, CWT, and GAF. While
the MS-CNN model using GAF images achieved the best overall accuracy and
precision, CWT has a slightly better recall. This study demonstrates how image
encoded gait data can be used to exploit the full potential of deep learning
CNN models for the assessment of frailty.

    

### [[2110.07822] On Extending Amdahl's law to Learn Computer Performance](http://arxiv.org/abs/2110.07822)


  The problem of learning parallel computer performance is investigated in the
context of multicore processors. Given a fixed workload, the effect of varying
system configuration on performance is sought. Conventionally, the performance
speedup due to a single resource enhancement is formulated using Amdahl's law.
However, in case of multiple configurable resources the conventional
formulation results in several disconnected speedup equations that cannot be
combined together to determine the overall speedup. To solve this problem, we
propose to (1) extend Amdahl's law to accommodate multiple configurable
resources into the overall speedup equation, and (2) transform the speedup
equation into a multivariable regression problem suitable for machine learning.
Using experimental data from two benchmarks (SPECCPU 2017 and PCMark 10) and
four hardware platforms (Intel Xeon 8180M, AMD EPYC 7702P, Intel CoffeeLake
8700K, and AMD Ryzen 3900X), analytical models are developed and
cross-validated. Findings indicate that in most cases, the models result in an
average cross-validated accuracy higher than 95%, thereby validating the
proposed extension of Amdahl's law. The proposed methodology enables rapid
generation of intelligent analytical models to support future industrial
development, optimization, and simulation needs.

    

### [[2110.07829] FedSEAL: Semi-Supervised Federated Learning with Self-Ensemble Learning and Negative Learning](http://arxiv.org/abs/2110.07829)


  Federated learning (FL), a popular decentralized and privacy-preserving
machine learning (FL) framework, has received extensive research attention in
recent years. The majority of existing works focus on supervised learning (SL)
problems where it is assumed that clients carry labeled datasets while the
server has no data. However, in realistic scenarios, clients are often unable
to label their data due to the lack of expertise and motivation while the
server may host a small amount of labeled data. How to reasonably utilize the
server labeled data and the clients' unlabeled data is thus of paramount
practical importance. In this paper, we propose a new FL algorithm, called
FedSEAL, to solve this Semi-Supervised Federated Learning (SSFL) problem. Our
algorithm utilizes self-ensemble learning and complementary negative learning
to enhance both the accuracy and the efficiency of clients' unsupervised
learning on unlabeled data, and orchestrates the model training on both the
server side and the clients' side. Our experimental results on Fashion-MNIST
and CIFAR10 datasets in the SSFL setting validate the effectiveness of our
method, which outperforms the state-of-the-art SSFL methods by a large margin.

    

### [[2110.07831] RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models](http://arxiv.org/abs/2110.07831)


  Backdoor attacks, which maliciously control a well-trained model's outputs of
the instances with specific triggers, are recently shown to be serious threats
to the safety of reusing deep neural networks (DNNs). In this work, we propose
an efficient online defense mechanism based on robustness-aware perturbations.
Specifically, by analyzing the backdoor training process, we point out that
there exists a big gap of robustness between poisoned and clean samples.
Motivated by this observation, we construct a word-based robustness-aware
perturbation to distinguish poisoned samples from clean samples to defend
against the backdoor attacks on natural language processing (NLP) models.
Moreover, we give a theoretical analysis about the feasibility of our
robustness-aware perturbation-based defense method. Experimental results on
sentiment analysis and toxic detection tasks show that our method achieves
better defending performance and much lower computational costs than existing
online defense methods. Our code is available at
this https URL.

    

### [[2110.07832] A Modern Analysis of Aging Machine Learning Based IoT Cybersecurity Methods](http://arxiv.org/abs/2110.07832)


  Modern scientific advancements often contribute to the introduction and
refinement of never-before-seen technologies. This can be quite the task for
humans to maintain and monitor and as a result, our society has become reliant
on machine learning to assist in this task. With new technology comes new
methods and thus new ways to circumvent existing cyber security measures. This
study examines the effectiveness of three distinct Internet of Things cyber
security algorithms currently used in industry today for malware and intrusion
detection: Random Forest (RF), Support-Vector Machine (SVM), and K-Nearest
Neighbor (KNN). Each algorithm was trained and tested on the Aposemat IoT-23
dataset which was published in January 2020 with the earliest of captures from
2018 and latest from 2019. The RF, SVM, and KNN reached peak accuracies of
92.96%, 86.23%, and 91.48%, respectively, in intrusion detection and 92.27%,
83.52%, and 89.80% in malware detection. It was found all three algorithms are
capable of being effectively utilized for the current landscape of IoT cyber
security in 2021.

    

### [[2110.07837] Cross-Lingual Fine-Grained Entity Typing](http://arxiv.org/abs/2110.07837)


  The growth of cross-lingual pre-trained models has enabled NLP tools to
rapidly generalize to new languages. While these models have been applied to
tasks involving entities, their ability to explicitly predict typological
features of these entities across languages has not been established. In this
paper, we present a unified cross-lingual fine-grained entity typing model
capable of handling over 100 languages and analyze this model's ability to
generalize to languages and entities unseen during training. We train this
model on cross-lingual training data collected from Wikipedia hyperlinks in
multiple languages (training languages). During inference, our model takes an
entity mention and context in a particular language (test language, possibly
not in the training languages) and predicts fine-grained types for that entity.
Generalizing to new languages and unseen entities are the fundamental
challenges of this entity typing setup, so we focus our evaluation on these
settings and compare against simple yet powerful string match baselines.
Experimental results show that our approach outperforms the baselines on unseen
languages such as Japanese, Tamil, Arabic, Serbian, and Persian. In addition,
our approach substantially improves performance on unseen entities (even in
unseen languages) over the baselines, and human evaluation shows a strong
ability to predict relevant types in these settings.

    

### [[2110.07843] FOLD-R++: A Toolset for Automated Inductive Learning of Default Theories from Mixed Data](http://arxiv.org/abs/2110.07843)


  FOLD-R is an automated inductive learning algorithm for learning default
rules with exceptions for mixed (numerical and categorical) data. It generates
an (explainable) answer set programming (ASP) rule set for classification
tasks. We present an improved FOLD-R algorithm, called FOLD-R++, that
significantly increases the efficiency and scalability of FOLD-R. FOLD-R++
improves upon FOLD-R without compromising or losing information in the input
training data during the encoding or feature selection phase. The FOLD-R++
algorithm is competitive in performance with the widely-used XGBoost algorithm,
however, unlike XGBoost, the FOLD-R++ algorithm produces an explainable model.
Next, we create a powerful tool-set by combining FOLD-R++ with s(CASP)-a
goal-directed ASP execution engine-to make predictions on new data samples
using the answer set program generated by FOLD-R++. The s(CASP) system also
produces a justification for the prediction. Experiments presented in this
paper show that our improved FOLD-R++ algorithm is a significant improvement
over the original design and that the s(CASP) system can make predictions in an
efficient manner as well.

    

### [[2110.07858] Understanding and Improving Robustness of Vision Transformers through Patch-based Negative Augmentation](http://arxiv.org/abs/2110.07858)


  We investigate the robustness of vision transformers (ViTs) through the lens
of their special patch-based architectural structure, i.e., they process an
image as a sequence of image patches. We find that ViTs are surprisingly
insensitive to patch-based transformations, even when the transformation
largely destroys the original semantics and makes the image unrecognizable by
humans. This indicates that ViTs heavily use features that survived such
transformations but are generally not indicative of the semantic class to
humans. Further investigations show that these features are useful but
non-robust, as ViTs trained on them can achieve high in-distribution accuracy,
but break down under distribution shifts. From this understanding, we ask: can
training the model to rely less on these features improve ViT robustness and
out-of-distribution performance? We use the images transformed with our
patch-based operations as negatively augmented views and offer losses to
regularize the training away from using non-robust features. This is a
complementary view to existing research that mostly focuses on augmenting
inputs with semantic-preserving transformations to enforce models' invariance.
We show that patch-based negative augmentation consistently improves robustness
of ViTs across a wide set of ImageNet based robustness benchmarks. Furthermore,
we find our patch-based negative augmentation are complementary to traditional
(positive) data augmentation, and together boost the performance further. All
the code in this work will be open-sourced.

    

### [[2110.07868] FedMe: Federated Learning via Model Exchange](http://arxiv.org/abs/2110.07868)


  Federated learning is a distributed machine learning method in which a single
server and multiple clients collaboratively build machine learning models
without sharing datasets on clients. Numerous methods have been proposed to
cope with the data heterogeneity issue in federated learning. Existing
solutions require a model architecture tuned by the central server, yet a major
technical challenge is that it is difficult to tune the model architecture due
to the absence of local data on the central server. In this paper, we propose
Federated learning via Model exchange (FedMe), which personalizes models with
automatic model architecture tuning during the learning process. The novelty of
FedMe lies in its learning process: clients exchange their models for model
architecture tuning and model training. First, to optimize the model
architectures for local data, clients tune their own personalized models by
comparing to exchanged models and picking the one that yields the best
performance. Second, clients train both personalized models and exchanged
models by using deep mutual learning, in spite of different model architectures
across the clients. We perform experiments on three real datasets and show that
FedMe outperforms state-of-the-art federated learning methods while tuning
model architectures automatically.

    

### [[2110.07869] A Dual-Perception Graph Neural Network with Multi-hop Graph Generator](http://arxiv.org/abs/2110.07869)


  Graph neural networks (GNNs) have drawn increasing attention in recent years
and achieved remarkable performance in many graph-based tasks, especially in
semi-supervised learning on graphs. However, most existing GNNs excessively
rely on topological structures and aggregate multi-hop neighborhood information
by simply stacking network layers, which may introduce superfluous noise
information, limit the expressive power of GNNs and lead to the over-smoothing
problem ultimately. In light of this, we propose a novel Dual-Perception Graph
Neural Network (DPGNN) to address these issues. In DPGNN, we utilize node
features to construct a feature graph, and perform node representations
learning based on the original topology graph and the constructed feature graph
simultaneously, which conduce to capture the structural neighborhood
information and the feature-related information. Furthermore, we design a
Multi-Hop Graph Generator (MHGG), which applies a node-to-hop attention
mechanism to aggregate node-specific multi-hop neighborhood information
adaptively. Finally, we apply self-ensembling to form a consistent prediction
for unlabeled node representations. Experimental results on five datasets with
different topological structures demonstrate that our proposed DPGNN achieves
competitive performance across all datasets, four of which the results
outperform the latest state-of-the-art models. The source code of our model is
available at this https URL.

    

### [[2110.07875] Graph Neural Networks with Learnable Structural and Positional Representations](http://arxiv.org/abs/2110.07875)


  Graph neural networks (GNNs) have become the standard learning architectures
for graphs. GNNs have been applied to numerous domains ranging from quantum
chemistry, recommender systems to knowledge graphs and natural language
processing. A major issue with arbitrary graphs is the absence of canonical
positional information of nodes, which decreases the representation power of
GNNs to distinguish e.g. isomorphic nodes and other graph symmetries. An
approach to tackle this issue is to introduce Positional Encoding (PE) of
nodes, and inject it into the input layer, like in Transformers. Possible graph
PE are Laplacian eigenvectors. In this work, we propose to decouple structural
and positional representations to make easy for the network to learn these two
essential properties. We introduce a novel generic architecture which we call
LSPE (Learnable Structural and Positional Encodings). We investigate several
sparse and fully-connected (Transformer-like) GNNs, and observe a performance
increase for molecular datasets, from 2.87% up to 64.14% when considering
learnable PE for both GNN classes.

    

### [[2110.07881] $k\texttt{-experts}$ -- Online Policies and Fundamental Limits](http://arxiv.org/abs/2110.07881)


  This paper introduces and studies the $k\texttt{-experts}$ problem -- a
generalization of the classic Prediction with Expert's Advice (i.e., the
$\texttt{Experts}$) problem. Unlike the $\texttt{Experts}$ problem, where the
learner chooses exactly one expert, in this problem, the learner selects a
subset of $k$ experts from a pool of $N$ experts at each round. The reward
obtained by the learner at any round depends on the rewards of the selected
experts. The $k\texttt{-experts}$ problem arises in many practical settings,
including online ad placements, personalized news recommendations, and paging.
Our primary goal is to design an online learning policy having a small regret.
In this pursuit, we propose $\texttt{SAGE}$ ($\textbf{Sa}$mpled
Hed$\textbf{ge}$) - a framework for designing efficient online learning
policies by leveraging statistical sampling techniques. We show that, for many
related problems, $\texttt{SAGE}$ improves upon the state-of-the-art bounds for
regret and computational complexity. Furthermore, going beyond the notion of
regret, we characterize the mistake bounds achievable by online learning
policies for a class of stable loss functions. We conclude the paper by
establishing a tight regret lower bound for a variant of the
$k\texttt{-experts}$ problem and carrying out experiments with standard
datasets.

    

### [[2110.07888] ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network](http://arxiv.org/abs/2110.07888)


  Graph Neural Networks (GNNs) have been widely studied in various graph data
mining tasks. Most existingGNNs embed graph data into Euclidean space and thus
are less effective to capture the ubiquitous hierarchical structures in
real-world networks. Hyperbolic Graph Neural Networks(HGNNs) extend GNNs to
hyperbolic space and thus are more effective to capture the hierarchical
structures of graphs in node representation learning. In hyperbolic geometry,
the graph hierarchical structure can be reflected by the curvatures of the
hyperbolic space, and different curvatures can model different hierarchical
structures of a graph. However, most existing HGNNs manually set the curvature
to a fixed value for simplicity, which achieves a suboptimal performance of
graph learning due to the complex and diverse hierarchical structures of the
graphs. To resolve this problem, we propose an Adaptive Curvature Exploration
Hyperbolic Graph NeuralNetwork named ACE-HGNN to adaptively learn the optimal
curvature according to the input graph and downstream tasks. Specifically,
ACE-HGNN exploits a multi-agent reinforcement learning framework and contains
two agents, ACE-Agent andHGNN-Agent for learning the curvature and node
representations, respectively. The two agents are updated by a NashQ-leaning
algorithm collaboratively, seeking the optimal hyperbolic space indexed by the
curvature. Extensive experiments on multiple real-world graph datasets
demonstrate a significant and consistent performance improvement in model
quality with competitive performance and good generalization ability.

    

### [[2110.07897] Improving Unsupervised Domain Adaptive Re-Identification via Source-Guided Selection of Pseudo-Labeling Hyperparameters](http://arxiv.org/abs/2110.07897)


  Unsupervised Domain Adaptation (UDA) for re-identification (re-ID) is a
challenging task: to avoid a costly annotation of additional data, it aims at
transferring knowledge from a domain with annotated data to a domain of
interest with only unlabeled data. Pseudo-labeling approaches have proven to be
effective for UDA re-ID. However, the effectiveness of these approaches heavily
depends on the choice of some hyperparameters (HP) that affect the generation
of pseudo-labels by clustering. The lack of annotation in the domain of
interest makes this choice non-trivial. Current approaches simply reuse the
same empirical value for all adaptation tasks and regardless of the target data
representation that changes through pseudo-labeling training phases. As this
simplistic choice may limit their performance, we aim at addressing this issue.
We propose new theoretical grounds on HP selection for clustering UDA re-ID as
well as method of automatic and cyclic HP tuning for pseudo-labeling UDA
clustering: HyPASS. HyPASS consists in incorporating two modules in
pseudo-labeling methods: (i) HP selection based on a labeled source validation
set and (ii) conditional domain alignment of feature discriminativeness to
improve HP selection based on source samples. Experiments on commonly used
person re-ID and vehicle re-ID datasets show that our proposed HyPASS
consistently improves the best state-of-the-art methods in re-ID compared to
the commonly used empirical HP setting.

    

### [[2110.07905] Towards Better Plasticity-Stability Trade-off in Incremental Learning: A simple Linear Connector](http://arxiv.org/abs/2110.07905)


  Plasticity-stability dilemma is a main problem for incremental learning, with
plasticity referring to the ability to learn new knowledge, and stability
retaining the knowledge of previous tasks. Due to the lack of training samples
from previous tasks, it is hard to balance the plasticity and stability. For
example, the recent null-space projection methods (e.g., Adam-NSCL) have shown
promising performance on preserving previous knowledge, while such strong
projection also causes the performance degradation of the current task. To
achieve better plasticity-stability trade-off, in this paper, we show that a
simple averaging of two independently optimized optima of networks, null-space
projection for past tasks and simple SGD for the current task, can attain a
meaningful balance between preserving already learned knowledge and granting
sufficient flexibility for learning a new task. This simple linear connector
also provides us a new perspective and technology to control the trade-off
between plasticity and stability. We evaluate the proposed method on several
benchmark datasets. The results indicate our simple method can achieve notable
improvement, and perform well on both the past and current tasks. In short, our
method is an extremely simple approach and achieves a better balance model.

    

### [[2110.07910] SaLinA: Sequential Learning of Agents](http://arxiv.org/abs/2110.07910)


  SaLinA is a simple library that makes implementing complex sequential
learning models easy, including reinforcement learning algorithms. It is built
as an extension of PyTorch: algorithms coded with \SALINA{} can be understood
in few minutes by PyTorch users and modified easily. Moreover, SaLinA naturally
works with multiple CPUs and GPUs at train and test time, thus being a good fit
for the large-scale training use cases. In comparison to existing RL libraries,
SaLinA has a very low adoption cost and capture a large variety of settings
(model-based RL, batch RL, hierarchical RL, multi-agent RL, etc.). But SaLinA
does not only target RL practitioners, it aims at providing sequential learning
capabilities to any deep learning programmer.

    

### [[2110.07922] Anomaly Detection in Multi-Agent Trajectories for Automated Driving](http://arxiv.org/abs/2110.07922)


  Human drivers can recognise fast abnormal driving situations to avoid
accidents. Similar to humans, automated vehicles are supposed to perform
anomaly detection. In this work, we propose the spatio-temporal graph
auto-encoder for learning normal driving behaviours. Our innovation is the
ability to jointly learn multiple trajectories of a dynamic number of agents.
To perform anomaly detection, we first estimate a density function of the
learned trajectory feature representation and then detect anomalies in
low-density regions. Due to the lack of multi-agent trajectory datasets for
anomaly detection in automated driving, we introduce our dataset using a
driving simulator for normal and abnormal manoeuvres. Our evaluations show that
our approach learns the relation between different agents and delivers
promising results compared to the related works. The code, simulation and the
dataset are publicly available on the project page:
this https URL.

    

### [[2110.07923] Value Penalized Q-Learning for Recommender Systems](http://arxiv.org/abs/2110.07923)


  Scaling reinforcement learning (RL) to recommender systems (RS) is promising
since maximizing the expected cumulative rewards for RL agents meets the
objective of RS, i.e., improving customers' long-term satisfaction. A key
approach to this goal is offline RL, which aims to learn policies from logged
data. However, the high-dimensional action space and the non-stationary
dynamics in commercial RS intensify distributional shift issues, making it
challenging to apply offline RL methods to RS. To alleviate the action
distribution shift problem in extracting RL policy from static trajectories, we
propose Value Penalized Q-learning (VPQ), an uncertainty-based offline RL
algorithm. It penalizes the unstable Q-values in the regression target by
uncertainty-aware weights, without the need to estimate the behavior policy,
suitable for RS with a large number of items. We derive the penalty weights
from the variances across an ensemble of Q-functions. To alleviate
distributional shift issues at test time, we further introduce the critic
framework to integrate the proposed method with classic RS models. Extensive
experiments conducted on two real-world datasets show that the proposed method
could serve as a gain plugin for existing RS models.

    

### [[2110.07940] Wasserstein Unsupervised Reinforcement Learning](http://arxiv.org/abs/2110.07940)


  Unsupervised reinforcement learning aims to train agents to learn a handful
of policies or skills in environments without external reward. These
pre-trained policies can accelerate learning when endowed with external reward,
and can also be used as primitive options in hierarchical reinforcement
learning. Conventional approaches of unsupervised skill discovery feed a latent
variable to the agent and shed its empowerment on agent's behavior by mutual
information (MI) maximization. However, the policies learned by MI-based
methods cannot sufficiently explore the state space, despite they can be
successfully identified from each other. Therefore we propose a new framework
Wasserstein unsupervised reinforcement learning (WURL) where we directly
maximize the distance of state distributions induced by different policies.
Additionally, we overcome difficulties in simultaneously training N(N >2)
policies, and amortizing the overall reward to each step. Experiments show
policies learned by our approach outperform MI-based methods on the metric of
Wasserstein distance while keeping high discriminability. Furthermore, the
agents trained by WURL can sufficiently explore the state space in mazes and
MuJoCo tasks and the pre-trained policies can be applied to downstream tasks by
hierarchical learning.

    

### [[2110.07959] Low-rank Matrix Recovery With Unknown Correspondence](http://arxiv.org/abs/2110.07959)


  We study a matrix recovery problem with unknown correspondence: given the
observation matrix $M_o=[A,\tilde P B]$, where $\tilde P$ is an unknown
permutation matrix, we aim to recover the underlying matrix $M=[A,B]$. Such
problem commonly arises in many applications where heterogeneous data are
utilized and the correspondence among them are unknown, e.g., due to privacy
concerns. We show that it is possible to recover $M$ via solving a nuclear norm
minimization problem under a proper low-rank condition on $M$, with provable
non-asymptotic error bound for the recovery of $M$. We propose an algorithm,
$\text{M}^3\text{O}$ (Matrix recovery via Min-Max Optimization) which recasts
this combinatorial problem as a continuous minimax optimization problem and
solves it by proximal gradient with a Max-Oracle. $\text{M}^3\text{O}$ can also
be applied to a more general scenario where we have missing entries in $M_o$
and multiple groups of data with distinct unknown correspondence. Experiments
on simulated data, the MovieLens 100K dataset and Yale B database show that
$\text{M}^3\text{O}$ achieves state-of-the-art performance over several
baselines and can recover the ground-truth correspondence with high accuracy.

    

### [[2110.07981] Reappraising Domain Generalization in Neural Networks](http://arxiv.org/abs/2110.07981)


  Domain generalization (DG) of machine learning algorithms is defined as their
ability to learn a domain agnostic hypothesis from multiple training
distributions, which generalizes onto data from an unseen domain. DG is vital
in scenarios where the target domain with distinct characteristics has sparse
data for training. Aligning with recent work~\cite{gulrajani2020search}, we
find that a straightforward Empirical Risk Minimization (ERM) baseline
consistently outperforms existing DG methods. We present ablation studies
indicating that the choice of backbone, data augmentation, and optimization
algorithms overshadows the many tricks and trades explored in the prior art.
Our work leads to a new state of the art on the four popular DG datasets,
surpassing previous methods by large margins. Furthermore, as a key
contribution, we propose a classwise-DG formulation, where for each class, we
randomly select one of the domains and keep it aside for testing. We argue that
this benchmarking is closer to human learning and relevant in real-world
scenarios. We comprehensively benchmark classwise-DG on the DomainBed and
propose a method combining ERM and reverse gradients to achieve the
state-of-the-art results. To our surprise, despite being exposed to all domains
during training, the classwise DG is more challenging than traditional DG
evaluation and motivates more fundamental rethinking on the problem of DG.

    

### [[2110.07983] NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem](http://arxiv.org/abs/2110.07983)


  We present NeuroLKH, a novel algorithm that combines deep learning with the
strong traditional heuristic Lin-Kernighan-Helsgaun (LKH) for solving Traveling
Salesman Problem. Specifically, we train a Sparse Graph Network (SGN) with
supervised learning for edge scores and unsupervised learning for node
penalties, both of which are critical for improving the performance of LKH.
Based on the output of SGN, NeuroLKH creates the edge candidate set and
transforms edge distances to guide the searching process of LKH. Extensive
experiments firmly demonstrate that, by training one model on a wide range of
problem sizes, NeuroLKH significantly outperforms LKH and generalizes well to
much larger sizes. Also, we show that NeuroLKH can be applied to other routing
problems such as Capacitated Vehicle Routing Problem (CVRP), Pickup and
Delivery Problem (PDP), and CVRP with Time Windows (CVRPTW).

    

### [[2110.07985] On-Policy Model Errors in Reinforcement Learning](http://arxiv.org/abs/2110.07985)


  Model-free reinforcement learning algorithms can compute policy gradients
given sampled environment transitions, but require large amounts of data. In
contrast, model-based methods can use the learned model to generate new data,
but model errors and bias can render learning unstable or sub-optimal. In this
paper, we present a novel method that combines real world data and a learned
model in order to get the best of both worlds. The core idea is to exploit the
real world data for on-policy predictions and use the learned model only to
generalize to different actions. Specifically, we use the data as
time-dependent on-policy correction terms on top of a learned model, to retain
the ability to generate data without accumulating errors over long prediction
horizons. We motivate this method theoretically and show that it counteracts an
error term for model-based policy improvement. Experiments on MuJoCo- and
PyBullet-benchmarks show that our method can drastically improve existing
model-based approaches without introducing additional tuning parameters.

    

### [[2110.07992] BayesAoA: A Bayesian method for Computation Efficient Angle of Arrival Estimation](http://arxiv.org/abs/2110.07992)


  The angle of Arrival (AoA) estimation is of great interest in modern
communication systems. Traditional maximum likelihood-based iterative
algorithms are sensitive to initialization and cannot be used online. We
propose a Bayesian method to find AoA that is insensitive towards
initialization. The proposed method is less complex and needs fewer computing
resources than traditional deep learning-based methods. It has a faster
convergence than the brute-force methods. Further, a Hedge type solution is
proposed that helps to deploy the method online to handle the situations where
the channel noise and antenna configuration in the receiver change over time.
The proposed method achieves $92\%$ accuracy in a channel of noise variance
$10^{-6}$ with $19.3\%$ of the brute-force method's computation.

    

### [[2110.08009] MaGNET: Uniform Sampling from Deep Generative Network Manifolds Without Retraining](http://arxiv.org/abs/2110.08009)


  Deep Generative Networks (DGNs) are extensively employed in Generative
Adversarial Networks (GANs), Variational Autoencoders (VAEs), and their
variants to approximate the data manifold, and data distribution on that
manifold. However, training samples are often obtained based on preferences,
costs, or convenience producing artifacts in the empirical data distribution
e.g., the large fraction of smiling faces in the CelebA dataset or the large
fraction of dark-haired individuals in FFHQ. These inconsistencies will be
reproduced when sampling from the trained DGN, which has far-reaching potential
implications for fairness, data augmentation, anomaly detection, domain
adaptation, and beyond. In response, we develop a differential geometry based
sampler -- coined MaGNET -- that, given any trained DGN, produces samples that
are uniformly distributed on the learned manifold. We prove theoretically and
empirically that our technique produces a uniform distribution on the manifold
regardless of the training set distribution. We perform a range of experiments
on various datasets and DGNs. One of them considers the state-of-the-art
StyleGAN2 trained on FFHQ dataset, where uniform sampling via MaGNET increases
distribution precision and recall by 4.1% & 3.0% and decreases gender bias by
41.2%, without requiring labels or retraining.

    

### [[2110.08021] StreaMulT: Streaming Multimodal Transformer for Heterogeneous and Arbitrary Long Sequential Data](http://arxiv.org/abs/2110.08021)


  This paper tackles the problem of processing and combining efficiently
arbitrary long data streams, coming from different modalities with different
acquisition frequencies. Common applications can be, for instance, long-time
industrial or real-life systems monitoring from multimodal heterogeneous data
(sensor data, monitoring report, images, etc.). To tackle this problem, we
propose StreaMulT, a Streaming Multimodal Transformer, relying on cross-modal
attention and an augmented memory bank to process arbitrary long input
sequences at training time and run in a streaming way at inference. StreaMulT
reproduces state-of-the-art results on CMU-MOSEI dataset, while being able to
deal with much longer inputs than other models such as previous Multimodal
Transformer.

    

### [[2110.08028] Improving Hyperparameter Optimization by Planning Ahead](http://arxiv.org/abs/2110.08028)


  Hyperparameter optimization (HPO) is generally treated as a bi-level
optimization problem that involves fitting a (probabilistic) surrogate model to
a set of observed hyperparameter responses, e.g. validation loss, and
consequently maximizing an acquisition function using a surrogate model to
identify good hyperparameter candidates for evaluation. The choice of a
surrogate and/or acquisition function can be further improved via knowledge
transfer across related tasks. In this paper, we propose a novel transfer
learning approach, defined within the context of model-based reinforcement
learning, where we represent the surrogate as an ensemble of probabilistic
models that allows trajectory sampling. We further propose a new variant of
model predictive control which employs a simple look-ahead strategy as a policy
that optimizes a sequence of actions, representing hyperparameter candidates to
expedite HPO. Our experiments on three meta-datasets comparing to
state-of-the-art HPO algorithms including a model-free reinforcement learning
approach show that the proposed method can outperform all baselines by
exploiting a simple planning-based policy.

    

### [[2110.08030] Identifying Incorrect Classifications with Balanced Uncertainty](http://arxiv.org/abs/2110.08030)


  Uncertainty estimation is critical for cost-sensitive deep-learning
applications (i.e. disease diagnosis). It is very challenging partly due to the
inaccessibility of uncertainty groundtruth in most datasets. Previous works
proposed to estimate the uncertainty from softmax calibration, Monte Carlo
sampling, subjective logic and so on. However, these existing methods tend to
be over-confident about their predictions with unreasonably low overall
uncertainty, which originates from the imbalance between positive (correct
classifications) and negative (incorrect classifications) samples. For this
issue, we firstly propose the distributional imbalance to model the imbalance
in uncertainty estimation as two kinds of distribution biases, and secondly
propose Balanced True Class Probability (BTCP) framework, which learns an
uncertainty estimator with a novel Distributional Focal Loss (DFL) objective.
Finally, we evaluate the BTCP in terms of failure prediction and
out-of-distribution (OOD) detection on multiple datasets. The experimental
results show that BTCP outperforms other uncertainty estimation methods
especially in identifying incorrect classifications.

    

### [[2110.08037] Tensor-to-Image: Image-to-Image Translation with Vision Transformers](http://arxiv.org/abs/2110.08037)


  Transformers gain huge attention since they are first introduced and have a
wide range of applications. Transformers start to take over all areas of deep
learning and the Vision transformers paper also proved that they can be used
for computer vision tasks. In this paper, we utilized a vision
transformer-based custom-designed model, tensor-to-image, for the image to
image translation. With the help of self-attention, our model was able to
generalize and apply to different problems without a single modification.

    

### [[2110.08038] Toward Annotator Group Bias in Crowdsourcing](http://arxiv.org/abs/2110.08038)


  Crowdsourcing has emerged as a popular approach for collecting annotated data
to train supervised machine learning models. However, annotator bias can lead
to defective annotations. Though there are a few works investigating individual
annotator bias, the group effects in annotators are largely overlooked. In this
work, we reveal that annotators within the same demographic group tend to show
consistent group bias in annotation tasks and thus we conduct an initial study
on annotator group bias. We first empirically verify the existence of annotator
group bias in various real-world crowdsourcing datasets. Then, we develop a
novel probabilistic graphical framework GroupAnno to capture annotator group
bias with a new extended Expectation Maximization (EM) training algorithm. We
conduct experiments on both synthetic and real-world datasets. Experimental
results demonstrate the effectiveness of our model in modeling annotator group
bias in label aggregation and model learning over competitive baselines.

    

### [[2110.08042] Adversarial Attacks on ML Defense Models Competition](http://arxiv.org/abs/2110.08042)


  Due to the vulnerability of deep neural networks (DNNs) to adversarial
examples, a large number of defense techniques have been proposed to alleviate
this problem in recent years. However, the progress of building more robust
models is usually hampered by the incomplete or incorrect robustness
evaluation. To accelerate the research on reliable evaluation of adversarial
robustness of the current defense models in image classification, the TSAIL
group at Tsinghua University and the Alibaba Security group organized this
competition along with a CVPR 2021 workshop on adversarial machine learning
(this https URL). The purpose of this
competition is to motivate novel attack algorithms to evaluate adversarial
robustness more effectively and reliably. The participants were encouraged to
develop stronger white-box attack algorithms to find the worst-case robustness
of different defenses. This competition was conducted on an adversarial
robustness evaluation platform -- ARES (this https URL), and is
held on the TianChi platform
(this https URL) as one of
the series of AI Security Challengers Program. After the competition, we
summarized the results and established a new adversarial robustness benchmark
at this https URL, which allows users to upload
adversarial attack algorithms and defense models for evaluation.

    

### [[2110.08045] Compressive Independent Component Analysis: Theory and Algorithms](http://arxiv.org/abs/2110.08045)


  Compressive learning forms the exciting intersection between compressed
sensing and statistical learning where one exploits forms of sparsity and
structure to reduce the memory and/or computational complexity of the learning
task. In this paper, we look at the independent component analysis (ICA) model
through the compressive learning lens. In particular, we show that solutions to
the cumulant based ICA model have particular structure that induces a low
dimensional model set that resides in the cumulant tensor space. By showing a
restricted isometry property holds for random cumulants e.g. Gaussian
ensembles, we prove the existence of a compressive ICA scheme. Thereafter, we
propose two algorithms of the form of an iterative projection gradient (IPG)
and an alternating steepest descent (ASD) algorithm for compressive ICA, where
the order of compression asserted from the restricted isometry property is
realised through empirical results. We provide analysis of the CICA algorithms
including the effects of finite samples. The effects of compression are
characterised by a trade-off between the sketch size and the statistical
efficiency of the ICA estimates. By considering synthetic and real datasets, we
show the substantial memory gains achieved over well-known ICA algorithms by
using one of the proposed CICA algorithms. Finally, we conclude the paper with
open problems including interesting challenges from the emerging field of
compressive learning.

    

### [[2110.08057] Almost Optimal Batch-Regret Tradeoff for Batch Linear Contextual Bandits](http://arxiv.org/abs/2110.08057)


  We study the optimal batch-regret tradeoff for batch linear contextual
bandits. For any batch number $M$, number of actions $K$, time horizon $T$, and
dimension $d$, we provide an algorithm and prove its regret guarantee, which,
due to technical reasons, features a two-phase expression as the time horizon
$T$ grows. We also prove a lower bound theorem that surprisingly shows the
optimality of our two-phase regret upper bound (up to logarithmic factors) in
the \emph{full range} of the problem parameters, therefore establishing the
exact batch-regret tradeoff.
Compared to the recent work \citep{ruan2020linear} which showed that $M =
O(\log \log T)$ batches suffice to achieve the asymptotically minimax-optimal
regret without the batch constraints, our algorithm is simpler and easier for
practical implementation. Furthermore, our algorithm achieves the optimal
regret for all $T \geq d$, while \citep{ruan2020linear} requires that $T$
greater than an unrealistically large polynomial of $d$.
Along our analysis, we also prove a new matrix concentration inequality with
dependence on their dynamic upper bounds, which, to the best of our knowledge,
is the first of its kind in literature and maybe of independent interest.

    

### [[2110.08058] Detecting Modularity in Deep Neural Networks](http://arxiv.org/abs/2110.08058)


  A neural network is modular to the extent that parts of its computational
graph (i.e. structure) can be represented as performing some comprehensible
subtask relevant to the overall task (i.e. functionality). Are modern deep
neural networks modular? How can this be quantified? In this paper, we consider
the problem of assessing the modularity exhibited by a partitioning of a
network's neurons. We propose two proxies for this: importance, which reflects
how crucial sets of neurons are to network performance; and coherence, which
reflects how consistently their neurons associate with features of the inputs.
To measure these proxies, we develop a set of statistical methods based on
techniques conventionally used to interpret individual neurons. We apply the
proxies to partitionings generated by spectrally clustering a graph
representation of the network's neurons with edges determined either by network
weights or correlations of activations. We show that these partitionings, even
ones based only on weights (i.e. strictly from non-runtime analysis), reveal
groups of neurons that are important and coherent. These results suggest that
graph-based partitioning can reveal modularity and help us understand how deep
neural networks function.

    

### [[2110.08059] FlexConv: Continuous Kernel Convolutions with Differentiable Kernel Sizes](http://arxiv.org/abs/2110.08059)


  When designing Convolutional Neural Networks (CNNs), one must select the size
of the convolutional kernels before training. Recent works show CNNs benefit
from different kernel sizes at different layers, but exploring all possible
combinations is unfeasible in practice. A more efficient approach is to learn
the kernel size during training. However, existing works that learn the kernel
size have a limited bandwidth. These approaches scale kernels by dilation, and
thus the detail they can describe is limited. In this work, we propose
FlexConv, a novel convolutional operation with which high bandwidth
convolutional kernels of learnable kernel size can be learned at a fixed
parameter cost. FlexNets model long-term dependencies without the use of
pooling, achieve state-of-the-art performance on several sequential datasets,
outperform recent works with learned kernel sizes, and are competitive with
much deeper ResNets on image benchmark datasets. Additionally, FlexNets can be
deployed at higher resolutions than those seen during training. To avoid
aliasing, we propose a novel kernel parameterization with which the frequency
of the kernels can be analytically controlled. Our novel kernel
parameterization shows higher descriptive power and faster convergence speed
than existing parameterizations. This leads to important improvements in
classification accuracy.

    

### [[2110.08066] Dual-Arm Adversarial Robot Learning](http://arxiv.org/abs/2110.08066)


  Robot learning is a very promising topic for the future of automation and
machine intelligence. Future robots should be able to autonomously acquire
skills, learn to represent their environment, and interact with it. While these
topics have been explored in simulation, real-world robot learning research
seems to be still limited. This is due to the additional challenges encountered
in the real-world, such as noisy sensors and actuators, safe exploration,
non-stationary dynamics, autonomous environment resetting as well as the cost
of running experiments for long periods of time. Unless we develop scalable
solutions to these problems, learning complex tasks involving hand-eye
coordination and rich contacts will remain an untouched vision that is only
feasible in controlled lab environments. We propose dual-arm settings as
platforms for robot learning. Such settings enable safe data collection for
acquiring manipulation skills as well as training perception modules in a
robot-supervised manner. They also ease the processes of resetting the
environment. Furthermore, adversarial learning could potentially boost the
generalization capability of robot learning methods by maximizing the
exploration based on game-theoretic objectives while ensuring safety based on
collaborative task spaces. In this paper, we will discuss the potential
benefits of this setup as well as the challenges and research directions that
can be pursued.

    

### [[2110.08084] Gradient Descent on Infinitely Wide Neural Networks: Global Convergence and Generalization](http://arxiv.org/abs/2110.08084)


  Many supervised machine learning methods are naturally cast as optimization
problems. For prediction models which are linear in their parameters, this
often leads to convex problems for which many mathematical guarantees exist.
Models which are non-linear in their parameters such as neural networks lead to
non-convex optimization problems for which guarantees are harder to obtain. In
this review paper, we consider two-layer neural networks with homogeneous
activation functions where the number of hidden neurons tends to infinity, and
show how qualitative convergence guarantees may be derived.

    

### [[2110.08087] Causal Identification with Additive Noise Models: Quantifying the Effect of Noise](http://arxiv.org/abs/2110.08087)


  In recent years, a lot of research has been conducted within the area of
causal inference and causal learning. Many methods have been developed to
identify the cause-effect pairs in models and have been successfully applied to
observational real-world data to determine the direction of causal
relationships. Yet in bivariate situations, causal discovery problems remain
challenging. One class of such methods, that also allows tackling the bivariate
case, is based on Additive Noise Models (ANMs). Unfortunately, one aspect of
these methods has not received much attention until now: what is the impact of
different noise levels on the ability of these methods to identify the
direction of the causal relationship. This work aims to bridge this gap with
the help of an empirical study. We test Regression with Subsequent Independence
Test (RESIT) using an exhaustive range of models where the level of additive
noise gradually changes from 1\% to 10000\% of the causes' noise level (the
latter remains fixed). Additionally, the experiments in this work consider
several different types of distributions as well as linear and non-linear
models. The results of the experiments show that ANMs methods can fail to
capture the true causal direction for some levels of noise.

    

### [[2110.08092] Equivariant and Invariant Reynolds Networks](http://arxiv.org/abs/2110.08092)


  Invariant and equivariant networks are useful in learning data with symmetry,
including images, sets, point clouds, and graphs. In this paper, we consider
invariant and equivariant networks for symmetries of finite groups. Invariant
and equivariant networks have been constructed by various researchers using
Reynolds operators. However, Reynolds operators are computationally expensive
when the order of the group is large because they use the sum over the whole
group, which poses an implementation difficulty. To overcome this difficulty,
we consider representing the Reynolds operator as a sum over a subset instead
of a sum over the whole group. We call such a subset a Reynolds design, and an
operator defined by a sum over a Reynolds design a reductive Reynolds operator.
For example, in the case of a graph with $n$ nodes, the computational
complexity of the reductive Reynolds operator is reduced to $O(n^2)$, while the
computational complexity of the Reynolds operator is $O(n!)$. We construct
learning models based on the reductive Reynolds operator called equivariant and
invariant Reynolds networks (ReyNets) and prove that they have universal
approximation property. Reynolds designs for equivariant ReyNets are derived
from combinatorial observations with Young diagrams, while Reynolds designs for
invariant ReyNets are derived from invariants called Reynolds dimensions
defined on the set of invariant polynomials. Numerical experiments show that
the performance of our models is comparable to state-of-the-art methods.

    

### [[2110.08100] A Survey of Evolutionary Multi-Objective Clustering Approaches](http://arxiv.org/abs/2110.08100)


  This article presents how the studies of the evolutionary multi-objective
clustering have been evolving over the years, based on a mapping of the indexed
articles in the ACM, IEEE, and Scopus. We present the most relevant approaches
considering the high impact journals and conferences to provide an overview of
this study field. We analyzed the algorithms based on the features and
components presented in the proposed general architecture of the evolutionary
multi-objective clustering. These algorithms were grouped considering common
clustering strategies and applications. Furthermore, issues regarding the
difficulty in defining appropriate clustering criteria applied to evolutionary
multi-objective clustering and the importance of the evolutionary process
evaluation to have a clear view of the optimization efficiency are discussed.
It is essential to observe these aspects besides specific clustering properties
when designing new approaches or selecting/using the existing ones. Finally, we
present other potential subjects of future research, in which this article can
contribute to newcomers or busy researchers who want to have a wide vision of
the field.

    

### [[2110.08101] An Artificial Neural Network-Based Model Predictive Control for Three-phase Flying Capacitor Multi-Level Inverter](http://arxiv.org/abs/2110.08101)


  Model predictive control (MPC) has been used widely in power electronics due
to its simple concept, fast dynamic response, and good reference tracking.
However, it suffers from parametric uncertainties, since it directly relies on
the mathematical model of the system to predict the optimal switching states to
be used at the next sampling time. As a result, uncertain parameters lead to an
ill-designed MPC. Thus, this paper offers a model-free control strategy on the
basis of artificial neural networks (ANNs), for mitigating the effects of
parameter mismatching while having a little negative impact on the inverter's
performance. This method includes two related stages. First, MPC is used as an
expert to control the studied converter in order to provide the training data;
while, in the second stage, the obtained dataset is utilized to train the
proposed ANN which will be used directly to control the inverter without the
requirement for the mathematical model of the system. The case study herein is
based on a four-level three-cell flying capacitor inverter. In this study,
MATLAB/Simulink is used to simulate the performance of the proposed control
strategy, taking into account various operating conditions. Afterward, the
simulation results are reported in comparison with the conventional MPC scheme,
demonstrating the superior performance of the proposed control strategy in
terms of getting low total harmonic distortion (THD) and the robustness against
parameters mismatch, especially when changes occur in the system parameters.

    

### [[2110.08105] Interpretable Neural Networks with Frank-Wolfe: Sparse Relevance Maps and Relevance Orderings](http://arxiv.org/abs/2110.08105)


  We study the effects of constrained optimization formulations and Frank-Wolfe
algorithms for obtaining interpretable neural network predictions.
Reformulating the Rate-Distortion Explanations (RDE) method for relevance
attribution as a constrained optimization problem provides precise control over
the sparsity of relevance maps. This enables a novel multi-rate as well as a
relevance-ordering variant of RDE that both empirically outperform standard RDE
in a well-established comparison test. We showcase several deterministic and
stochastic variants of the Frank-Wolfe algorithm and their effectiveness for
RDE.

    

### [[2110.08111] An active learning approach for improving the performance of equilibrium based chemical simulations](http://arxiv.org/abs/2110.08111)


  In this paper, we propose a novel sequential data-driven method for dealing
with equilibrium based chemical simulations, which can be seen as a specific
machine learning approach called active learning. The underlying idea of our
approach is to consider the function to estimate as a sample of a Gaussian
process which allows us to compute the global uncertainty on the function
estimation. Thanks to this estimation and with almost no parameter to tune, the
proposed method sequentially chooses the most relevant input data at which the
function to estimate has to be evaluated to build a surrogate model. Hence, the
number of evaluations of the function to estimate is dramatically limited. Our
active learning method is validated through numerical experiments and applied
to a complex chemical system commonly used in geoscience.

    

### [[2110.08113] Hand Me Your PIN! Inferring ATM PINs of Users Typing with a Covered Hand](http://arxiv.org/abs/2110.08113)


  Automated Teller Machines (ATMs) represent the most used system for
withdrawing cash. The European Central Bank reported more than 11 billion cash
withdrawals and loading/unloading transactions on the European ATMs in 2019.
Although ATMs have undergone various technological evolutions, Personal
Identification Numbers (PINs) are still the most common authentication method
for these devices. Unfortunately, the PIN mechanism is vulnerable to
shoulder-surfing attacks performed via hidden cameras installed near the ATM to
catch the PIN pad. To overcome this problem, people get used to covering the
typing hand with the other hand. While such users probably believe this
behavior is safe enough to protect against mentioned attacks, there is no clear
assessment of this countermeasure in the scientific literature.
This paper proposes a novel attack to reconstruct PINs entered by victims
covering the typing hand with the other hand. We consider the setting where the
attacker can access an ATM PIN pad of the same brand/model as the target one.
Afterward, the attacker uses that model to infer the digits pressed by the
victim while entering the PIN. Our attack owes its success to a carefully
selected deep learning architecture that can infer the PIN from the typing hand
position and movements. We run a detailed experimental analysis including 58
users. With our approach, we can guess 30% of the 5-digit PINs within three
attempts -- the ones usually allowed by ATM before blocking the card. We also
conducted a survey with 78 users that managed to reach an accuracy of only
7.92% on average for the same setting. Finally, we evaluate a shielding
countermeasure that proved to be rather inefficient unless the whole keypad is
shielded.

    

### [[2110.08128] Label-Wise Message Passing Graph Neural Network on Heterophilic Graphs](http://arxiv.org/abs/2110.08128)


  Graph Neural Networks (GNNs) have achieved remarkable performance in modeling
graphs for various applications. However, most existing GNNs assume the graphs
exhibit strong homophily in node labels, i.e., nodes with similar labels are
connected in the graphs. They fail to generalize to heterophilic graphs where
linked nodes may have dissimilar labels and attributes. Therefore, in this
paper, we investigate a novel framework that performs well on graphs with
either homophily or heterophily. More specifically, to address the challenge
brought by the heterophily in graphs, we propose a label-wise message passing
mechanism. In label-wise message-passing, neighbors with similar pseudo labels
will be aggregated together, which will avoid the negative effects caused by
aggregating dissimilar node representations. We further propose a bi-level
optimization method to automatically select the model for graphs with
homophily/heterophily. Extensive experiments demonstrate the effectiveness of
our proposed framework for node classification on both homophilic and
heterophilic graphs.

    

### [[2110.08133] Trade-offs of Local SGD at Scale: An Empirical Study](http://arxiv.org/abs/2110.08133)


  As datasets and models become increasingly large, distributed training has
become a necessary component to allow deep neural networks to train in
reasonable amounts of time. However, distributed training can have substantial
communication overhead that hinders its scalability. One strategy for reducing
this overhead is to perform multiple unsynchronized SGD steps independently on
each worker between synchronization steps, a technique known as local SGD. We
conduct a comprehensive empirical study of local SGD and related methods on a
large-scale image classification task. We find that performing local SGD comes
at a price: lower communication costs (and thereby faster training) are
accompanied by lower accuracy. This finding is in contrast from the
smaller-scale experiments in prior work, suggesting that local SGD encounters
challenges at scale. We further show that incorporating the slow momentum
framework of Wang et al. (2020) consistently improves accuracy without
requiring additional communication, hinting at future directions for
potentially escaping this trade-off.

    

### [[2110.08169] Containerized Distributed Value-Based Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2110.08169)


  Multi-agent reinforcement learning tasks put a high demand on the volume of
training samples. Different from its single-agent counterpart, distributed
value-based multi-agent reinforcement learning faces the unique challenges of
demanding data transfer, inter-process communication management, and high
requirement of exploration. We propose a containerized learning framework to
solve these problems. We pack several environment instances, a local learner
and buffer, and a carefully designed multi-queue manager which avoids blocking
into a container. Local policies of each container are encouraged to be as
diverse as possible, and only trajectories with highest priority are sent to a
global learner. In this way, we achieve a scalable, time-efficient, and diverse
distributed MARL learning framework with high system throughput. To own
knowledge, our method is the first to solve the challenging Google Research
Football full game $5\_v\_5$. On the StarCraft II micromanagement benchmark,
our method gets $4$-$18\times$ better results compared to state-of-the-art
non-distributed MARL algorithms.

    

### [[2110.08176] Collaborating with Humans without Human Data](http://arxiv.org/abs/2110.08176)


  Collaborating with humans requires rapidly adapting to their individual
strengths, weaknesses, and preferences. Unfortunately, most standard
multi-agent reinforcement learning techniques, such as self-play (SP) or
population play (PP), produce agents that overfit to their training partners
and do not generalize well to humans. Alternatively, researchers can collect
human data, train a human model using behavioral cloning, and then use that
model to train "human-aware" agents ("behavioral cloning play", or BCP). While
such an approach can improve the generalization of agents to new human
co-players, it involves the onerous and expensive step of collecting large
amounts of human data first. Here, we study the problem of how to train agents
that collaborate well with human partners without using human data. We argue
that the crux of the problem is to produce a diverse set of training partners.
Drawing inspiration from successful multi-agent approaches in competitive
domains, we find that a surprisingly simple approach is highly effective. We
train our agent partner as the best response to a population of self-play
agents and their past checkpoints taken throughout training, a method we call
Fictitious Co-Play (FCP). Our experiments focus on a two-player collaborative
cooking simulator that has recently been proposed as a challenge problem for
coordination with humans. We find that FCP agents score significantly higher
than SP, PP, and BCP when paired with novel agent and human partners.
Furthermore, humans also report a strong subjective preference to partnering
with FCP agents over all baselines.

    

### [[2110.08185] Propagation on Multi-relational Graphs for Node Regression](http://arxiv.org/abs/2110.08185)


  Recent years have witnessed a rise in real-world data captured with rich
structural information that can be conveniently depicted by multi-relational
graphs. While inference of continuous node features across a simple graph is
rather under-studied by the current relational learning research, we go one
step further and focus on node regression problem on multi-relational graphs.
We take inspiration from the well-known label propagation algorithm aiming at
completing categorical features across a simple graph and propose a novel
propagation framework for completing missing continuous features at the nodes
of a multi-relational and directed graph. Our multi-relational propagation
algorithm is composed of iterative neighborhood aggregations which originate
from a relational local generative model. Our findings show the benefit of
exploiting the multi-relational structure of the data in several node
regression scenarios in different settings.

    

### [[2110.08202] Evaluation of Hyperparameter-Optimization Approaches in an Industrial Federated Learning System](http://arxiv.org/abs/2110.08202)


  Federated Learning (FL) decouples model training from the need for direct
access to the data and allows organizations to collaborate with industry
partners to reach a satisfying level of performance without sharing vulnerable
business information. The performance of a machine learning algorithm is highly
sensitive to the choice of its hyperparameters. In an FL setting,
hyperparameter optimization poses new challenges. In this work, we investigated
the impact of different hyperparameter optimization approaches in an FL system.
In an effort to reduce communication costs, a critical bottleneck in FL, we
investigated a local hyperparameter optimization approach that -- in contrast
to a global hyperparameter optimization approach -- allows every client to have
its own hyperparameter configuration. We implemented these approaches based on
grid search and Bayesian optimization and evaluated the algorithms on the MNIST
data set using an i.i.d. partition and on an Internet of Things (IoT) sensor
based industrial data set using a non-i.i.d. partition.

    

### [[2110.08203] Shared Visual Representations of Drawing for Communication: How do different biases affect human interpretability and intent?](http://arxiv.org/abs/2110.08203)


  We present an investigation into how representational losses can affect the
drawings produced by artificial agents playing a communication game. Building
upon recent advances, we show that a combination of powerful pretrained encoder
networks, with appropriate inductive biases, can lead to agents that draw
recognisable sketches, whilst still communicating well. Further, we start to
develop an approach to help automatically analyse the semantic content being
conveyed by a sketch and demonstrate that current approaches to inducing
perceptual biases lead to a notion of objectness being a key feature despite
the agent training being self-supervised.

    

### [[2110.08207] Multitask Prompted Training Enables Zero-Shot Task Generalization](http://arxiv.org/abs/2110.08207)


  Large language models have recently been shown to attain reasonable zero-shot
generalization on a diverse set of tasks. It has been hypothesized that this is
a consequence of implicit multitask learning in language model training. Can
zero-shot generalization instead be directly induced by explicit multitask
learning? To test this question at scale, we develop a system for easily
mapping general natural language tasks into a human-readable prompted form. We
convert a large set of supervised datasets, each with multiple prompts using
varying natural language. These prompted datasets allow for benchmarking the
ability of a model to perform completely unseen tasks specified in natural
language. We fine-tune a pretrained encoder-decoder model on this multitask
mixture covering a wide variety of tasks. The model attains strong zero-shot
performance on several standard datasets, often outperforming models 16x its
size. Further, our approach attains strong performance on a subset of tasks
from the BIG-Bench benchmark, outperforming models 6x its size. All prompts and
trained models are available at this http URL.

    

### [[2110.08212] NNK-Means: Dictionary Learning using Non-Negative Kernel regression](http://arxiv.org/abs/2110.08212)


  An increasing number of systems are being designed by first gathering
significant amounts of data, and then optimizing the system parameters directly
using the obtained data. Often this is done without analyzing the dataset
structure. As task complexity, data size, and parameters all increase to
millions or even billions, data summarization is becoming a major challenge. In
this work, we investigate data summarization via dictionary learning,
leveraging the properties of recently introduced non-negative kernel regression
(NNK) graphs. Our proposed NNK-Means, unlike competing techniques, such askSVD,
learns geometric dictionaries with atoms that lie in the input data space.
Experiments show that summaries using NNK-Meanscan provide better
discrimination compared to linear and kernel versions of kMeans and kSVD.
Moreover, NNK-Means has a scalable implementation, with runtime complexity
similar to that of kMeans.

    

### [[2110.08217] Choice functions based multi-objective Bayesian optimisation](http://arxiv.org/abs/2110.08217)


  In this work we introduce a new framework for multi-objective Bayesian
optimisation where the multi-objective functions can only be accessed via
choice judgements, such as ``I pick options A,B,C among this set of five
options A,B,C,D,E''. The fact that the option D is rejected means that there is
at least one option among the selected ones A,B,C that I strictly prefer over D
(but I do not have to specify which one). We assume that there is a latent
vector function f for some dimension $n_e$ which embeds the options into the
real vector space of dimension n, so that the choice set can be represented
through a Pareto set of non-dominated options. By placing a Gaussian process
prior on f and deriving a novel likelihood model for choice data, we propose a
Bayesian framework for choice functions learning. We then apply this surrogate
model to solve a novel multi-objective Bayesian optimisation from choice data
problem.

    

### [[2110.08220] Combining Diverse Feature Priors](http://arxiv.org/abs/2110.08220)


  To improve model generalization, model designers often restrict the features
that their models use, either implicitly or explicitly. In this work, we
explore the design space of leveraging such feature priors by viewing them as
distinct perspectives on the data. Specifically, we find that models trained
with diverse sets of feature priors have less overlapping failure modes, and
can thus be combined more effectively. Moreover, we demonstrate that jointly
training such models on additional (unlabeled) data allows them to correct each
other's mistakes, which, in turn, leads to better generalization and resilience
to spurious correlations. Code available at
this https URL.

    

### [[2110.08223] VICause: Simultaneous Missing Value Imputation and Causal Discovery with Groups](http://arxiv.org/abs/2110.08223)


  Missing values constitute an important challenge in real-world machine
learning for both prediction and causal discovery tasks. However, existing
imputation methods are agnostic to causality, while only few methods in
traditional causal discovery can handle missing data in an efficient way. In
this work we propose VICause, a novel approach to simultaneously tackle missing
value imputation and causal discovery efficiently with deep learning.
Particularly, we propose a generative model with a structured latent space and
a graph neural network-based architecture, scaling to large number of
variables. Moreover, our method can discover relationships between groups of
variables which is useful in many real-world applications. VICause shows
improved performance compared to popular and recent approaches in both missing
value imputation and causal discovery.

    

### [[2110.08226] Guiding Visual Question Generation](http://arxiv.org/abs/2110.08226)


  In traditional Visual Question Generation (VQG), most images have multiple
concepts (e.g. objects and categories) for which a question could be generated,
but models are trained to mimic an arbitrary choice of concept as given in
their training data. This makes training difficult and also poses issues for
evaluation -- multiple valid questions exist for most images but only one or a
few are captured by the human references. We present Guiding Visual Question
Generation - a variant of VQG which conditions the question generator on
categorical information based on expectations on the type of question and the
objects it should explore. We propose two variants: (i) an explicitly guided
model that enables an actor (human or automated) to select which objects and
categories to generate a question for; and (ii) an implicitly guided model that
learns which objects and categories to condition on, based on discrete latent
variables. The proposed models are evaluated on an answer-category augmented
VQA dataset and our quantitative results show a substantial improvement over
the current state of the art (over 9 BLEU-4 increase). Human evaluation
validates that guidance helps the generation of questions that are
grammatically coherent and relevant to the given image and objects.

    

### [[2110.08229] Influencing Towards Stable Multi-Agent Interactions](http://arxiv.org/abs/2110.08229)


  Learning in multi-agent environments is difficult due to the non-stationarity
introduced by an opponent's or partner's changing behaviors. Instead of
reactively adapting to the other agent's (opponent or partner) behavior, we
propose an algorithm to proactively influence the other agent's strategy to
stabilize -- which can restrain the non-stationarity caused by the other agent.
We learn a low-dimensional latent representation of the other agent's strategy
and the dynamics of how the latent strategy evolves with respect to our robot's
behavior. With this learned dynamics model, we can define an unsupervised
stability reward to train our robot to deliberately influence the other agent
to stabilize towards a single strategy. We demonstrate the effectiveness of
stabilizing in improving efficiency of maximizing the task reward in a variety
of simulated environments, including autonomous driving, emergent
communication, and robotic manipulation. We show qualitative results on our
website: this https URL.

    

### [[2110.08232] Fire Together Wire Together: A Dynamic Pruning Approach with Self-Supervised Mask Prediction](http://arxiv.org/abs/2110.08232)


  Dynamic model pruning is a recent direction that allows for the inference of
a different sub-network for each input sample during deployment. However,
current dynamic methods rely on learning a continuous channel gating through
regularization by inducing sparsity loss. This formulation introduces
complexity in balancing different losses (e.g task loss, regularization loss).
In addition, regularization-based methods lack transparent tradeoff
hyperparameter selection to realize computational budget. Our contribution is
twofold: 1) decoupled task and pruning training. 2) Simple hyperparameter
selection that enables FLOPs reduction estimation before training. We propose
to predict a mask to process k filters in a layer based on the activation of
its previous layer. We pose the problem as a self-supervised binary
classification problem. Each mask predictor module is trained to predict if the
log-likelihood of each filter in the current layer belongs to the top-k
activated filters. The value k is dynamically estimated for each input based on
a novel criterion using the mass of heatmaps. We show experiments on several
neural architectures, such as VGG, ResNet, and MobileNet on CIFAR and ImageNet
datasets. On CIFAR, we reach similar accuracy to SOTA methods with 15% and 24%
higher FLOPs reduction. Similarly in ImageNet, we achieve a lower drop in
accuracy with up to 13% improvement in FLOPs reduction.

    

### [[2110.08239] Learn Proportional Derivative Controllable Latent Space from Pixels](http://arxiv.org/abs/2110.08239)


  Recent advances in latent space dynamics model from pixels show promising
progress in vision-based model predictive control (MPC). However, executing MPC
in real time can be challenging due to its intensive computational cost in each
timestep. We propose to introduce additional learning objectives to enforce
that the learned latent space is proportional derivative controllable. In
execution time, the simple PD-controller can be applied directly to the latent
space encoded from pixels, to produce simple and effective control to systems
with visual observations. We show that our method outperforms baseline methods
to produce robust goal reaching and trajectory tracking in various
environments.

    

### [[2110.08243] Neural Dubber: Dubbing for Silent Videos According to Scripts](http://arxiv.org/abs/2110.08243)


  Dubbing is a post-production process of re-recording actors' dialogues, which
is extensively used in filmmaking and video production. It is usually performed
manually by professional voice actors who read lines with proper prosody, and
in synchronization with the pre-recorded videos. In this work, we propose
Neural Dubber, the first neural network model to solve a novel automatic video
dubbing (AVD) task: synthesizing human speech synchronized with the given
silent video from the text. Neural Dubber is a multi-modal text-to-speech (TTS)
model that utilizes the lip movement in the video to control the prosody of the
generated speech. Furthermore, an image-based speaker embedding (ISE) module is
developed for the multi-speaker setting, which enables Neural Dubber to
generate speech with a reasonable timbre according to the speaker's face.
Experiments on the chemistry lecture single-speaker dataset and LRS2
multi-speaker dataset show that Neural Dubber can generate speech audios on par
with state-of-the-art TTS models in terms of speech quality. Most importantly,
both qualitative and quantitative evaluations show that Neural Dubber can
control the prosody of synthesized speech by the video, and generate
high-fidelity speech temporally synchronized with the video.

    

### [[2110.08245] LPRules: Rule Induction in Knowledge Graphs Using Linear Programming](http://arxiv.org/abs/2110.08245)


  Knowledge graph (KG) completion is a well-studied problem in AI. Rule-based
methods and embedding-based methods form two of the solution techniques.
Rule-based methods learn first-order logic rules that capture existing facts in
an input graph and then use these rules for reasoning about missing facts. A
major drawback of such methods is the lack of scalability to large datasets. In
this paper, we present a simple linear programming (LP) model to choose rules
from a list of candidate rules and assign weights to them. For smaller KGs, we
use simple heuristics to create the candidate list. For larger KGs, we start
with a small initial candidate list, and then use standard column generation
ideas to add more rules in order to improve the LP model objective value. To
foster interpretability and generalizability, we limit the complexity of the
set of chosen rules via explicit constraints, and tune the complexity
hyperparameter for individual datasets. We show that our method can obtain
state-of-the-art results for three out of four widely used KG datasets, while
taking significantly less computing time than other popular rule learners
including some based on neuro-symbolic methods. The improved scalability of our
method allows us to tackle large datasets such as YAGO3-10.

    

### [[2110.08248] Transforming Autoregression: Interpretable and Expressive Time Series Forecast](http://arxiv.org/abs/2110.08248)


  Probabilistic forecasting of time series is an important matter in many
applications and research fields. In order to draw conclusions from a
probabilistic forecast, we must ensure that the model class used to approximate
the true forecasting distribution is expressive enough. Yet, characteristics of
the model itself, such as its uncertainty or its general functioning are not of
lesser importance. In this paper, we propose Autoregressive Transformation
Models (ATMs), a model class inspired from various research directions such as
normalizing flows and autoregressive models. ATMs unite expressive
distributional forecasts using a semi-parametric distribution assumption with
an interpretable model specification and allow for uncertainty quantification
based on (asymptotic) Maximum Likelihood theory. We demonstrate the properties
of ATMs both theoretically and through empirical evaluation on several
simulated and real-world forecasting datasets.

    

### [[1810.00303] Newton-MR: Inexact Newton Method With Minimum Residual Sub-problem Solver](http://arxiv.org/abs/1810.00303)


  We consider a variant of inexact Newton Method, called Newton-MR, in which
the least-squares sub-problems are solved approximately using Minimum Residual
method. By construction, Newton-MR can be readily applied for unconstrained
optimization of a class of non-convex problems known as invex, which subsumes
convexity as a sub-class. For invex optimization, instead of the classical
Lipschitz continuity assumptions on gradient and Hessian, Newton-MR's global
convergence can be guaranteed under a weaker notion of joint regularity of
Hessian and gradient. We also obtain Newton-MR's problem-independent local
convergence to the set of minima. We show that fast local/global convergence
can be guaranteed under a novel inexactness condition, which, to our knowledge,
is much weaker than the prior related works. Numerical results demonstrate the
performance of Newton-MR as compared with several other Newton-type
alternatives on a few machine learning problems.

    

### [[1907.08356] New Era of Deeplearning-Based Malware Intrusion Detection: The Malware Detection and Prediction Based On Deep Learning](http://arxiv.org/abs/1907.08356)


  With the development of artificial intelligence algorithms like deep learning
models and the successful applications in many different fields, further
similar trails of deep learning technology have been made in cyber security
area. It shows the preferable performance not only in academic security
research but also in industry practices when dealing with part of cyber
security issues by deep learning methods compared to those conventional rules.
Especially for the malware detection and classification tasks, it saves
generous time cost and promotes the accuracy for a total pipeline of malware
detection system. In this paper, we construct special deep neural network, ie,
MalDeepNet (TB-Malnet and IB-Malnet) for malware dynamic behavior
classification tasks. Then we build the family clustering algorithm based on
deep learning and fulfil related testing. Except that, we also design a novel
malware prediction model which could detect the malware coming in future
through the Mal Generative Adversarial Network (Mal-GAN) implementation. All
those algorithms present fairly considerable value in related datasets
afterwards.

    

### [[1912.08140] On-the-fly Global Embeddings Using Random Projections for Extreme Multi-label Classification](http://arxiv.org/abs/1912.08140)


  The goal of eXtreme Multi-label Learning (XML) is to automatically annotate a
given data point with the most relevant subset of labels from an extremely
large vocabulary of labels (e.g., a million labels). Lately, many attempts have
been made to address this problem that achieve reasonable performance on
benchmark datasets. In this paper, rather than coming-up with an altogether new
method, our objective is to present and validate a simple baseline for this
task. Precisely, we investigate an on-the-fly global and structure preserving
feature embedding technique using random projections whose learning phase is
independent of training samples and label vocabulary. Further, we show how an
ensemble of multiple such learners can be used to achieve further boost in
prediction accuracy with only linear increase in training and prediction time.
Experiments on three public XML benchmarks show that the proposed approach
obtains competitive accuracy compared with many existing methods. Additionally,
it also provides around 6572x speed-up ratio in terms of training time and
around 14.7x reduction in model-size compared to the closest competitors on the
largest publicly available dataset.

    

### [[2002.12036] Complexity Measures and Features for Times Series classification](http://arxiv.org/abs/2002.12036)


  Classification of time series is a growing problem in different disciplines
due to the progressive digitalization of the world. Currently, the
state-of-the-art in time series classification is dominated by The Hierarchical
Vote Collective of Transformation-based Ensembles. This algorithm is composed
of several classifiers of different domains distributed in five large modules.
The combination of the results obtained by each module weighed based on an
internal evaluation process allows this algorithm to obtain the best results in
state-of-the-art. One Nearest Neighbour with Dynamic Time Warping remains the
base classifier in any time series classification problem for its simplicity
and good results. Despite their performance, they share a weakness, which is
that they are not interpretable. In the field of time series classification,
there is a tradeoff between accuracy and interpretability. In this work, we
propose a set of characteristics capable of extracting information on the
structure of the time series to face time series classification problems. The
use of these characteristics allows the use of traditional classification
algorithms in time series problems. The experimental results of our proposal
show no statistically significant differences from the second and third best
models of the state-of-the-art. Apart from competitive results in accuracy, our
proposal is able to offer interpretable results based on the set of
characteristics proposed

    

### [[2006.06592] The Backbone Method for Ultra-High Dimensional Sparse Machine Learning](http://arxiv.org/abs/2006.06592)


  We present the backbone method, a generic framework that enables sparse and
interpretable supervised machine learning methods to scale to ultra-high
dimensional problems. We solve sparse regression problems with $10^7$ features
in minutes and $10^8$ features in hours, as well as decision tree problems with
$10^5$ features in minutes.The proposed method operates in two phases: we first
determine the backbone set, consisting of potentially relevant features, by
solving a number of tractable subproblems; then, we solve a reduced problem,
considering only the backbone features. For the sparse regression problem, our
theoretical analysis shows that, under certain assumptions and with high
probability, the backbone set consists of the truly relevant features.
Numerical experiments on both synthetic and real-world datasets demonstrate
that our method outperforms or competes with state-of-the-art methods in
ultra-high dimensional problems, and competes with optimal solutions in
problems where exact methods scale, both in terms of recovering the truly
relevant features and in its out-of-sample predictive performance.

    

### [[2007.05139] Mechanisms for Hiding Sensitive Genotypes with Information-Theoretic Privacy](http://arxiv.org/abs/2007.05139)


  Motivated by the growing availability of personal genomics services, we study
an information-theoretic privacy problem that arises when sharing genomic data:
a user wants to share his or her genome sequence while keeping the genotypes at
certain positions hidden, which could otherwise reveal critical health-related
information. A straightforward solution of erasing (masking) the chosen
genotypes does not ensure privacy, because the correlation between nearby
positions can leak the masked genotypes. We introduce an erasure-based privacy
mechanism with perfect information-theoretic privacy, whereby the released
sequence is statistically independent of the sensitive genotypes. Our mechanism
can be interpreted as a locally-optimal greedy algorithm for a given processing
order of sequence positions, where utility is measured by the number of
positions released without erasure. We show that finding an optimal order is
NP-hard in general and provide an upper bound on the optimal utility. For
sequences from hidden Markov models, a standard modeling approach in genetics,
we propose an efficient algorithmic implementation of our mechanism with
complexity polynomial in sequence length. Moreover, we illustrate the
robustness of the mechanism by bounding the privacy leakage from erroneous
prior distributions. Our work is a step towards more rigorous control of
privacy in genomic data sharing.

    

### [[2007.12652] MurTree: Optimal Classification Trees via Dynamic Programming and Search](http://arxiv.org/abs/2007.12652)


  Decision tree learning is a widely used approach in machine learning,
favoured in applications that require concise and interpretable models.
Heuristic methods are traditionally used to quickly produce models with
reasonably high accuracy. A commonly criticised point, however, is that the
resulting trees may not necessarily be the best representation of the data in
terms of accuracy and size. In recent years, this motivated the development of
optimal classification tree algorithms that globally optimise the decision tree
in contrast to heuristic methods that perform a sequence of locally optimal
decisions. We follow this line of work and provide a novel algorithm for
learning optimal classification trees based on dynamic programming and search.
Our algorithm supports constraints on the depth of the tree and number of
nodes. The success of our approach is attributed to a series of specialised
techniques that exploit properties unique to classification trees. Whereas
algorithms for optimal classification trees have traditionally been plagued by
high runtimes and limited scalability, we show in a detailed experimental study
that our approach uses only a fraction of the time required by the
state-of-the-art and can handle datasets with tens of thousands of instances,
providing several orders of magnitude improvements and notably contributing
towards the practical realisation of optimal decision trees.

    

### [[2009.06921] Optimal Decision Trees for Nonlinear Metrics](http://arxiv.org/abs/2009.06921)


  Nonlinear metrics, such as the F1-score, Matthews correlation coefficient,
and Fowlkes-Mallows index, are often used to evaluate the performance of
machine learning models, in particular, when facing imbalanced datasets that
contain more samples of one class than the other. Recent optimal decision tree
algorithms have shown remarkable progress in producing trees that are optimal
with respect to linear criteria, such as accuracy, but unfortunately nonlinear
metrics remain a challenge. To address this gap, we propose a novel algorithm
based on bi-objective optimisation, which treats misclassifications of each
binary class as a separate objective. We show that, for a large class of
metrics, the optimal tree lies on the Pareto frontier. Consequently, we obtain
the optimal tree by using our method to generate the set of all nondominated
trees. To the best of our knowledge, this is the first method to compute
provably optimal decision trees for nonlinear metrics. Our approach leads to a
trade-off when compared to optimising linear metrics: the resulting trees may
be more desirable according to the given nonlinear metric at the expense of
higher runtimes. Nevertheless, the experiments illustrate that runtimes are
reasonable for majority of the tested datasets.

    

### [[2010.15335] Learning Sampling Distributions Using Local 3D Workspace Decompositions for Motion Planning in High Dimensions](http://arxiv.org/abs/2010.15335)


  Earlier work has shown that reusing experience from prior motion planning
problems can improve the efficiency of similar, future motion planning queries.
However, for robots with many degrees-of-freedom, these methods exhibit poor
generalization across different environments and often require large datasets
that are impractical to gather. We present SPARK and FLAME , two
experience-based frameworks for sampling-based planning applicable to complex
manipulators in 3 D environments. Both combine samplers associated with
features from a workspace decomposition into a global biased sampling
distribution. SPARK decomposes the environment based on exact geometry while
FLAME is more general, and uses an octree-based decomposition obtained from
sensor data. We demonstrate the effectiveness of SPARK and FLAME on a Fetch
robot tasked with challenging pick-and-place manipulation problems. Our
approaches can be trained incrementally and significantly improve performance
with only a handful of examples, generalizing better over diverse tasks and
environments as compared to prior approaches.

    

### [[2011.06070] Quantifying and Learning Linear Symmetry-Based Disentanglement](http://arxiv.org/abs/2011.06070)


  The definition of Linear Symmetry-Based Disentanglement (LSBD) formalizes the
notion of linearly disentangled representations, but there is currently no
metric to quantify LSBD. Such a metric is crucial to evaluate LSBD methods and
to compare to previous understandings of disentanglement. We propose
$\mathcal{D}_\mathrm{LSBD}$, a mathematically sound metric to quantify LSBD,
and provide a practical implementation for $\mathrm{SO}(2)$ groups.
Furthermore, from this metric we derive LSBD-VAE, a semi-supervised method to
learn LSBD representations. We demonstrate the utility of our metric by showing
that (1) common VAE-based disentanglement methods don't learn LSBD
representations, (2) LSBD-VAE as well as other recent methods can learn LSBD
representations, needing only limited supervision on transformations, and (3)
various desirable properties expressed by existing disentanglement metrics are
also achieved by LSBD representations.

    

### [[2101.09324] Generating Black-Box Adversarial Examples in Sparse Domain](http://arxiv.org/abs/2101.09324)


  Applications of machine learning (ML) models and convolutional neural
networks (CNNs) have been rapidly increased. Although state-of-the-art CNNs
provide high accuracy in many applications, recent investigations show that
such networks are highly vulnerable to adversarial attacks. The black-box
adversarial attack is one type of attack that the attacker does not have any
knowledge about the model or the training dataset, but it has some input data
set and their labels. In this paper, we propose a novel approach to generate a
black-box attack in sparse domain whereas the most important information of an
image can be observed. Our investigation shows that large sparse (LaS)
components play a critical role in the performance of image classifiers. Under
this presumption, to generate adversarial example, we transfer an image into a
sparse domain and put a threshold to choose only k LaS components. In contrast
to the very recent works that randomly perturb k low frequency (LoF)
components, we perturb k LaS components either randomly (query-based) or in the
direction of the most correlated sparse signal from a different class. We show
that LaS components contain some middle or higher frequency components
information which leads fooling image classifiers with a fewer number of
queries. We demonstrate the effectiveness of this approach by fooling six
state-of-the-art image classifiers, the TensorFlow Lite (TFLite) model of
Google Cloud Vision platform, and YOLOv5 model as an object detection
algorithm. Mean squared error (MSE) and peak signal to noise ratio (PSNR) are
used as quality metrics. We also present a theoretical proof to connect these
metrics to the level of perturbation in the sparse domain.

    

### [[2102.10774] Provably Improved Context-Based Offline Meta-RL with Attention and Contrastive Learning](http://arxiv.org/abs/2102.10774)


  Meta-learning for offline reinforcement learning (OMRL) is an understudied
problem with tremendous potential impact by enabling RL algorithms in many
real-world applications. A popular solution to the problem is to infer task
identity as augmented state using a context-based encoder, for which efficient
learning of robust task representations remains an open challenge. In this
work, we provably improve upon one of the SOTA OMRL algorithms, FOCAL, by
incorporating intra-task attention mechanism and inter-task contrastive
learning objectives, to robustify task representation learning against sparse
reward and distribution shift. Theoretical analysis and experiments are
presented to demonstrate the superior performance and robustness of our
end-to-end and model-free framework compared to prior algorithms across
multiple meta-RL benchmarks.

    

### [[2102.11382] Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity](http://arxiv.org/abs/2102.11382)


  We present Sandwich Batch Normalization (SaBN), a frustratingly easy
improvement of Batch Normalization (BN) with only a few lines of code changes.
SaBN is motivated by addressing the inherent feature distribution heterogeneity
that one can be identified in many tasks, which can arise from data
heterogeneity (multiple input domains) or model heterogeneity (dynamic
architectures, model conditioning, etc.). Our SaBN factorizes the BN affine
layer into one shared sandwich affine layer, cascaded by several parallel
independent affine layers. Concrete analysis reveals that, during optimization,
SaBN promotes balanced gradient norms while still preserving diverse gradient
directions -- a property that many application tasks seem to favor. We
demonstrate the prevailing effectiveness of SaBN as a drop-in replacement in
four tasks: conditional image generation, neural architecture search (NAS),
adversarial training, and arbitrary style transfer. Leveraging SaBN immediately
achieves better Inception Score and FID on CIFAR-10 and ImageNet conditional
image generation with three state-of-the-art GANs; boosts the performance of a
state-of-the-art weight-sharing NAS algorithm significantly on NAS-Bench-201;
substantially improves the robust and standard accuracies for adversarial
defense; and produces superior arbitrary stylized results. We also provide
visualizations and analysis to help understand why SaBN works. Codes are
available at: this https URL.

    

### [[2102.13088] Even your Teacher Needs Guidance: Ground-Truth Targets Dampen Regularization Imposed by Self-Distillation](http://arxiv.org/abs/2102.13088)


  Knowledge distillation is classically a procedure where a neural network is
trained on the output of another network along with the original targets in
order to transfer knowledge between the architectures. The special case of
self-distillation, where the network architectures are identical, has been
observed to improve generalization accuracy. In this paper, we consider an
iterative variant of self-distillation in a kernel regression setting, in which
successive steps incorporate both model outputs and the ground-truth targets.
This allows us to provide the first theoretical results on the importance of
using the weighted ground-truth targets in self-distillation. Our focus is on
fitting nonlinear functions to training data with a weighted mean square error
objective function suitable for distillation, subject to $\ell_2$
regularization of the model parameters. We show that any such function obtained
with self-distillation can be calculated directly as a function of the initial
fit, and that infinite distillation steps yields the same optimization problem
as the original with amplified regularization. Furthermore, we provide a closed
form solution for the optimal choice of weighting parameter at each step, and
show how to efficiently estimate this weighting parameter for deep learning and
significantly reduce the computational requirements compared to a grid search.

    

### [[2103.15670] On the Adversarial Robustness of Vision Transformers](http://arxiv.org/abs/2103.15670)


  Following the success in advancing natural language processing and
understanding, transformers are expected to bring revolutionary changes to
computer vision. This work provides the first and comprehensive study on the
robustness of vision transformers (ViTs) against adversarial perturbations.
Tested on various white-box and transfer attack settings, we find that ViTs
possess better adversarial robustness when compared with convolutional neural
networks (CNNs). This observation also holds for certified robustness. We
summarize the following main observations contributing to the improved
robustness of ViTs:
1) Features learned by ViTs contain less low-level information and are more
generalizable, which contributes to superior robustness against adversarial
perturbations.
2) Introducing convolutional or tokens-to-token blocks for learning low-level
features in ViTs can improve classification accuracy but at the cost of
adversarial robustness.
3) Increasing the proportion of transformers in the model structure (when the
model consists of both transformer and CNN blocks) leads to better robustness.
But for a pure transformer model, simply increasing the size or adding layers
cannot guarantee a similar effect.
4) Pre-training on larger datasets does not significantly improve adversarial
robustness though it is critical for training ViTs.
5) Adversarial training is also applicable to ViT for training robust models.
Furthermore, feature visualization and frequency analysis are conducted for
explanation. The results show that ViTs are less sensitive to high-frequency
perturbations than CNNs and there is a high correlation between how well the
model learns low-level features and its robustness against different
frequency-based perturbations.

    

### [[2103.16701] Charged particle tracking via edge-classifying interaction networks](http://arxiv.org/abs/2103.16701)


  Recent work has demonstrated that geometric deep learning methods such as
graph neural networks (GNNs) are well suited to address a variety of
reconstruction problems in high energy particle physics. In particular,
particle tracking data is naturally represented as a graph by identifying
silicon tracker hits as nodes and particle trajectories as edges; given a set
of hypothesized edges, edge-classifying GNNs identify those corresponding to
real particle trajectories. In this work, we adapt the physics-motivated
interaction network (IN) GNN toward the problem of particle tracking in pileup
conditions similar to those expected at the high-luminosity Large Hadron
Collider. Assuming idealized hit filtering at various particle momenta
thresholds, we demonstrate the IN's excellent edge-classification accuracy and
tracking efficiency through a suite of measurements at each stage of GNN-based
tracking: graph construction, edge classification, and track building. The
proposed IN architecture is substantially smaller than previously studied GNN
tracking architectures; this is particularly promising as a reduction in size
is critical for enabling GNN-based tracking in constrained computing
environments. Furthermore, the IN may be represented as either a set of
explicit matrix operations or a message passing GNN. Efforts are underway to
accelerate each representation via heterogeneous computing resources towards
both high-level and low-latency triggering applications.

    

### [[2104.02935] TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition](http://arxiv.org/abs/2104.02935)


  In this paper, we propose TSception, a multi-scale convolutional neural
network, to learn temporal dynamics and spatial asymmetry from
electroencephalogram (EEG). TSception consists of dynamic temporal, asymmetric
spatial, and high-level fusion layers, which learn discriminative
representations in the time and channel dimensions simultaneously. The dynamic
temporal layer consists of multi-scale 1D convolutional kernels whose lengths
are related to the sampling rate of the EEG signal, which learns the dynamic
temporal and frequency representations of EEG. The asymmetric spatial layer
takes advantage of the asymmetric neural activations underlying emotional
responses, learning the discriminative global and hemisphere representations.
The learned spatial representations will be fused by a high-level fusion layer.
Using more generalized cross-validation settings, the proposed method is
evaluated on two publicly available datasets DEAP and MAHNOB-HCI. The
performance of the proposed network is compared with prior reported methods
such as SVM, KNN, FBFgMDM, FBTSC, Unsupervised learning, DeepConvNet,
ShallowConvNet, and EEGNet. Our method achieves higher classification
accuracies and F1 scores than the compared methods in most of the experiments.
The proposed methods can be utilized in emotion regulation therapy for emotion
recognition in the future. The source code can be found at
this https URL


### [[2104.07353] Fast Private Parameter Learning and Inference for Sum-Product Networks](http://arxiv.org/abs/2104.07353)


  A sum-product network (SPN) is a graphical model that allows several types of
inferences to be drawn efficiently. There are two types of learning for SPNs:
Learning the architecture of the model, and learning the parameters. In this
paper, we tackle the second problem: We show how to learn the weights for the
sum nodes, assuming the architecture is fixed, and the data is horizontally
partitioned between multiple parties. The computations will preserve the
privacy of each participant. Furthermore, we will use secret sharing instead of
(homomorphic) encryption, which allows fast computations and requires little
computational resources. To this end, we use a novel integer division to
compute approximate real divisions. We also show how simple and private
inferences can be performed using the learned SPN.

    

### [[2104.08737] Low-Rank Subspaces for Unsupervised Entity Linking](http://arxiv.org/abs/2104.08737)


  Entity linking is an important problem with many applications. Most previous
solutions were designed for settings where annotated training data is
available, which is, however, not the case in numerous domains. We propose a
light-weight and scalable entity linking method, Eigenthemes, that relies
solely on the availability of entity names and a referent knowledge base.
Eigenthemes exploits the fact that the entities that are truly mentioned in a
document (the "gold entities") tend to form a semantically dense subset of the
set of all candidate entities in the document. Geometrically speaking, when
representing entities as vectors via some given embedding, the gold entities
tend to lie in a low-rank subspace of the full embedding space. Eigenthemes
identifies this subspace using the singular value decomposition and scores
candidate entities according to their proximity to the subspace. On the
empirical front, we introduce multiple strong baselines that compare favorably
to (and sometimes even outperform) the existing state of the art. Extensive
experiments on benchmark datasets from a variety of real-world domains showcase
the effectiveness of our approach.

    

### [[2105.04906] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](http://arxiv.org/abs/2105.04906)


  Recent self-supervised methods for image representation learning are based on
maximizing the agreement between embedding vectors from different views of the
same image. A trivial solution is obtained when the encoder outputs constant
vectors. This collapse problem is often avoided through implicit biases in the
learning architecture, that often lack a clear justification or interpretation.
In this paper, we introduce VICReg (Variance-Invariance-Covariance
Regularization), a method that explicitly avoids the collapse problem with a
simple regularization term on the variance of the embeddings along each
dimension individually. VICReg combines the variance term with a decorrelation
mechanism based on redundancy reduction and covariance regularization, and
achieves results on par with the state of the art on several downstream tasks.
In addition, we show that incorporating our new variance term into other
methods helps stabilize the training and leads to performance improvements.

    

### [[2105.07610] Cross-Cluster Weighted Forests](http://arxiv.org/abs/2105.07610)


  Adapting machine learning algorithms to better handle clustering or batch
effects within training data sets is important across a wide variety of
biological applications. This article considers the effect of ensembling Random
Forest learners trained on clusters within a single data set with heterogeneity
in the distribution of the features. We find that constructing ensembles of
forests trained on clusters determined by algorithms such as k-means results in
significant improvements in accuracy and generalizability over the traditional
Random Forest algorithm. We denote our novel approach as the Cross-Cluster
Weighted Forest, and examine its robustness to various data-generating
scenarios and outcome models. Furthermore, we explore the influence of the
data-partitioning and ensemble weighting strategies the benefits of our method
over the existing paradigm. Finally, we apply our approach to cancer molecular
profiling and gene expression data sets that are naturally divisible into
clusters and illustrate that our approach outperforms classic Random Forest.
Code and supplementary material are available at
this https URL.

    

### [[2105.11590] A Quantum Hopfield Associative Memory Implemented on an Actual Quantum Processor](http://arxiv.org/abs/2105.11590)


  In this work, we present a Quantum Hopfield Associative Memory (QHAM) and
demonstrate its capabilities in simulation and hardware using IBM Quantum
Experience. The QHAM is based on a quantum neuron design which can be utilized
for many different machine learning applications and can be implemented on real
quantum hardware without requiring mid-circuit measurement or reset operations.
We analyze the accuracy of the neuron and the full QHAM considering hardware
errors via simulation with hardware noise models as well as with implementation
on the 15-qubit ibmq_16_melbourne device. The quantum neuron and the QHAM are
shown to be resilient to noise and require low qubit overhead and gate
complexity. We benchmark the QHAM by testing its effective memory capacity and
demonstrate its capabilities in the NISQ-era of quantum hardware. This
demonstration of the first functional QHAM to be implemented in NISQ-era
quantum hardware is a significant step in machine learning at the leading edge
of quantum computing.

    

### [[2105.12787] Self-Supervised Bug Detection and Repair](http://arxiv.org/abs/2105.12787)


  Machine learning-based program analyses have recently shown the promise of
integrating formal and probabilistic reasoning towards aiding software
development. However, in the absence of large annotated corpora, training these
analyses is challenging. Towards addressing this, we present BugLab, an
approach for self-supervised learning of bug detection and repair. BugLab
co-trains two models: (1) a detector model that learns to detect and repair
bugs in code, (2) a selector model that learns to create buggy code for the
detector to use as training data. A Python implementation of BugLab improves by
up to 30% upon baseline methods on a test dataset of 2374 real-life bugs and
finds 19 previously unknown bugs in open-source software.

    

### [[2106.02713] Learning Curves for SGD on Structured Features](http://arxiv.org/abs/2106.02713)


  The generalization performance of a machine learning algorithm such as a
neural network depends in a non-trivial way on the structure of the data
distribution. To analyze the influence of data structure on test loss dynamics,
we study an exactly solveable model of stochastic gradient descent (SGD) on
mean square loss which predicts test loss when training on features with
arbitrary covariance structure. We solve the theory exactly for both Gaussian
features and arbitrary features and we show that the simpler Gaussian model
accurately predicts test loss of nonlinear random-feature models and deep
neural networks trained with SGD on real datasets such as MNIST and CIFAR-10.
We show that the optimal batch size at a fixed compute budget is typically
small and depends on the feature correlation structure, demonstrating the
computational benefits of SGD with small batch sizes. Lastly, we extend our
theory to the more usual setting of stochastic gradient descent on a fixed
subsampled training set, showing that both training and test error can be
accurately predicted in our framework on real data.

    

### [[2106.02886] Context-Aware Sparse Deep Coordination Graphs](http://arxiv.org/abs/2106.02886)


  Learning sparse coordination graphs adaptive to the coordination dynamics
among agents is a long-standing problem in cooperative multi-agent learning.
This paper studies this problem and proposes a novel method using the variance
of payoff functions to construct context-aware sparse coordination topologies.
We theoretically consolidate our method by proving that the smaller the
variance of payoff functions is, the less likely action selection will change
after removing the corresponding edge. Moreover, we propose to learn action
representations to effectively reduce the influence of payoff functions'
estimation errors on graph construction. To empirically evaluate our method, we
present the Multi-Agent COordination (MACO) benchmark by collecting classic
coordination problems in the literature, increasing their difficulty, and
classifying them into different types. We carry out a case study and
experiments on the MACO and StarCraft II micromanagement benchmark to
demonstrate the dynamics of sparse graph learning, the influence of graph
sparseness, and the learning performance of our method.

    

### [[2106.06603] A Shuffling Framework for Local Differential Privacy](http://arxiv.org/abs/2106.06603)


  ldp deployments are vulnerable to inference attacks as an adversary can link
the noisy responses to their identity and subsequently, auxiliary information
using the order of the data. An alternative model, shuffle DP, prevents this by
shuffling the noisy responses uniformly at random. However, this limits the
data learnability -- only symmetric functions (input order agnostic) can be
learned. In this paper, we strike a balance and show that systematic shuffling
of the noisy responses can thwart specific inference attacks while retaining
some meaningful data learnability. To this end, we propose a novel privacy
guarantee, d-sigma-privacy, that captures the privacy of the order of a data
sequence. d-sigma-privacy allows tuning the granularity at which the ordinal
information is maintained, which formalizes the degree the resistance to
inference attacks trading it off with data learnability. Additionally, we
propose a novel shuffling mechanism that can achieve \name-privacy and
demonstrate the practicality of our mechanism via evaluation on real-world
datasets.

    

### [[2106.11609] Distributional Gradient Matching for Learning Uncertain Neural Dynamics Models](http://arxiv.org/abs/2106.11609)


  Differential equations in general and neural ODEs in particular are an
essential technique in continuous-time system identification. While many
deterministic learning algorithms have been designed based on numerical
integration via the adjoint method, many downstream tasks such as active
learning, exploration in reinforcement learning, robust control, or filtering
require accurate estimates of predictive uncertainties. In this work, we
propose a novel approach towards estimating epistemically uncertain neural
ODEs, avoiding the numerical integration bottleneck. Instead of modeling
uncertainty in the ODE parameters, we directly model uncertainties in the state
space. Our algorithm - distributional gradient matching (DGM) - jointly trains
a smoother and a dynamics model and matches their gradients via minimizing a
Wasserstein loss. Our experiments show that, compared to traditional
approximate inference methods based on numerical integration, our approach is
faster to train, faster at predicting previously unseen trajectories, and in
the context of neural ODEs, significantly more accurate.

    

### [[2106.12248] ADAVI: Automatic Dual Amortized Variational Inference Applied To Pyramidal Bayesian Models](http://arxiv.org/abs/2106.12248)


  Frequently, population studies feature pyramidally-organized data represented
using Hierarchical Bayesian Models (HBM) enriched with plates.These models can
become prohibitively large in settings such as neuroimaging, where a sample is
composed of a functional MRI signal measured on 64 thousand brain locations,
across 4 measurement sessions, and at least tens of subjects. Even a reduced
example on a specific cortical region of 300 brain locations features around 1
million parameters, hampering the usage of modern density estimation techniques
such as Simulation-Based Inference (SBI) or structured Variational Inference
(VI).To infer parameter posterior distributions in this challenging class of
problems, we designed a novel methodology that automatically produces a
variational family dual to a target HBM. This variational family, represented
as a neural network, consists in the combination of an attention-based
hierarchical encoder feeding summary statistics to a set of normalizing flows.
Our automatically-derived neural network exploits exchangeability in the
plate-enriched HBM and factorizes its parameter space. The resulting
architecture reduces by orders of magnitude its parameterization with respect
to that of a typical SBI or structured VI representation, while maintaining
expressivity.Our method performs inference on the specified HBM in an amortized
setup: once trained, it can readily be applied to a new data sample to compute
the parameters' full posterior.We demonstrate the capability and scalability of
our method on simulated data, as well as a challenging high-dimensional brain
parcellation experiment. We also open up several questions that lie at the
intersection between SBI techniques, structured Variational Inference, and
inference amortization.

    

### [[2110.01387] Machine Learning with Knowledge Constraints for Process Optimization of Open-Air Perovskite Solar Cell Manufacturing](http://arxiv.org/abs/2110.01387)


  Perovskite photovoltaics (PV) have achieved rapid development in the past
decade in terms of power conversion efficiency of small-area lab-scale devices;
however, successful commercialization still requires further development of
low-cost, scalable, and high-throughput manufacturing techniques. One of the
key challenges to the development of a new fabrication technique is the
high-dimensional parameter space, and machine learning (ML) can be used to
accelerate perovskite PV scaling. Here, we present an ML-guided framework of
sequential learning for manufacturing process optimization. We apply our
methodology to the Rapid Spray Plasma Processing (RSPP) technique for
perovskite thin films in ambient conditions. With a limited experimental budget
of screening 100 conditions process conditions, we demonstrated an efficiency
improvement to 18.5% for the best device, and we also experimentally found 10
unique conditions to produce the top-performing devices of more than 17%
efficiency, which is 5 times higher rate of success than pseudo-random Latin
hypercube sampling. Our model is enabled by three innovations: (a) flexible
knowledge transfer between experimental processes by incorporating data from
prior experimental data as a soft constraint; (b) incorporation of both
subjective human observations and ML insights when selecting next experiments;
(c) adaptive strategy of locating the region of interest using Bayesian
optimization first, and then conducting local exploration for high-efficiency
devices. Furthermore, in virtual benchmarking, our framework achieves faster
improvements with limited experimental budgets than traditional
design-of-experiments methods (e.g., one-variable-at-a-time sampling).

    

### [[2110.04745] Reinforcement Learning for Systematic FX Trading](http://arxiv.org/abs/2110.04745)


  We conduct a detailed experiment on major cash fx pairs, accurately
accounting for transaction and funding costs. These sources of profit and loss,
including the price trends that occur in the currency markets, are made
available to our recurrent reinforcement learner via a quadratic utility, which
learns to target a position directly. We improve upon earlier work, by casting
the problem of learning to target a risk position, in an online learning
context. This online learning occurs sequentially in time, but also in the form
of transfer learning. We transfer the output of radial basis function hidden
processing units, whose means, covariances and overall size are determined by
Gaussian mixture models, to the recurrent reinforcement learner and baseline
momentum trader. Thus the intrinsic nature of the feature space is learnt and
made available to the upstream models. The recurrent reinforcement learning
trader achieves an annualised portfolio information ratio of 0.52 with compound
return of 9.3%, net of execution and funding cost, over a 7 year test set. This
is despite forcing the model to trade at the close of the trading day 5pm EST,
when trading costs are statistically the most expensive. These results are
comparable with the momentum baseline trader, reflecting the low interest
differential environment since the the 2008 financial crisis, and very obvious
currency trends since then. The recurrent reinforcement learner does
nevertheless maintain an important advantage, in that the model's weights can
be adapted to reflect the different sources of profit and loss variation. This
is demonstrated visually by a USDRUB trading agent, who learns to target
different positions, that reflect trading in the absence or presence of cost.

    

### [[2110.05324] Learnable Adaptive Cosine Estimator (LACE) for Image Classification](http://arxiv.org/abs/2110.05324)


  In this work, we propose a new loss to improve feature discriminability and
classification performance. Motivated by the adaptive cosine/coherence
estimator (ACE), our proposed method incorporates angular information that is
inherently learned by artificial neural networks. Our learnable ACE (LACE)
transforms the data into a new "whitened" space that improves the inter-class
separability and intra-class compactness. We compare our LACE to alternative
state-of-the art softmax-based and feature regularization approaches. Our
results show that the proposed method can serve as a viable alternative to
cross entropy and angular softmax approaches. Our code is publicly available:
this https URL.

    

### [[2110.05354] Internal Language Model Adaptation with Text-Only Data for End-to-End Speech Recognition](http://arxiv.org/abs/2110.05354)


  Text-only adaptation of an end-to-end (E2E) model remains a challenging task
for automatic speech recognition (ASR). Language model (LM) fusion-based
approaches require an additional external LM during inference, significantly
increasing the computation cost. To overcome this, we propose an internal LM
adaptation (ILMA) of the E2E model using text-only data. Trained with
audio-transcript pairs, an E2E model implicitly learns an internal LM that
characterizes the token sequence probability which is approximated by the E2E
model output after zeroing out the encoder contribution. During ILMA, we
fine-tune the internal LM, i.e., the E2E components excluding the encoder, to
minimize a cross-entropy loss. To make ILMA effective, it is essential to train
the E2E model with an internal LM loss besides the standard E2E loss.
Furthermore, we propose to regularize ILMA by minimizing the Kullback-Leibler
divergence between the output distributions of the adapted and unadapted
internal LMs. ILMA is the most effective when we update only the last linear
layer of the joint network. ILMA enables a fast text-only adaptation of the E2E
model without increasing the run-time computational cost. Experimented with
30K-hour trained transformer transducer models, ILMA achieves up to 34.9%
relative word error rate reduction from the unadapted baseline.

    

### [[2103.11575] Learn-to-Race: A Multimodal Control Environment for Autonomous Racing](http://arxiv.org/abs/2103.11575)


  Existing research on autonomous driving primarily focuses on urban driving,
which is insufficient for characterising the complex driving behaviour
underlying high-speed racing. At the same time, existing racing simulation
frameworks struggle in capturing realism, with respect to visual rendering,
vehicular dynamics, and task objectives, inhibiting the transfer of learning
agents to real-world contexts. We introduce a new environment, where agents
Learn-to-Race (L2R) in simulated competition-style racing, using multimodal
information--from virtual cameras to a comprehensive array of inertial
measurement sensors. Our environment, which includes a simulator and an
interfacing training framework, accurately models vehicle dynamics and racing
conditions. In this paper, we release the Arrival simulator for autonomous
racing. Next, we propose the L2R task with challenging metrics, inspired by
learning-to-drive challenges, Formula-style racing, and multimodal trajectory
prediction for autonomous driving. Additionally, we provide the L2R framework
suite, facilitating simulated racing on high-precision models of real-world
tracks. Finally, we provide an official L2R task dataset of expert
demonstrations, as well as a series of baseline experiments and reference
implementations. We make all code available:
this https URL.

    

### [[2110.04924] High-dimensional Inference for Dynamic Treatment Effects](http://arxiv.org/abs/2110.04924)


  This paper proposes a confidence interval construction for heterogeneous
treatment effects in the context of multi-stage experiments with $N$ samples
and high-dimensional, $d$, confounders. Our focus is on the case of $d\gg N$,
but the results obtained also apply to low-dimensional cases. We showcase that
the bias of regularized estimation, unavoidable in high-dimensional covariate
spaces, is mitigated with a simple double-robust score. In this way, no
additional bias removal is necessary, and we obtain root-$N$ inference results
while allowing multi-stage interdependency of the treatments and covariates.
Memoryless property is also not assumed; treatment can possibly depend on all
previous treatment assignments and all previous multi-stage confounders. Our
results rely on certain sparsity assumptions of the underlying dependencies. We
discover new product rate conditions necessary for robust inference with
dynamic treatments.

    

### [[2110.08221] Metrics and Design of an Instruction Roofline Model for AMD GPUs](http://arxiv.org/abs/2110.08221)


  Due to the recent announcement of the Frontier supercomputer, many scientific
application developers are working to make their applications compatible with
AMD architectures (CPU-GPU), which means moving away from the traditional CPU
and NVIDIA-GPU systems. Due to the current limitations of profiling tools for
AMD GPUs, this shift leaves a void in how to measure application performance on
AMD GPUs. In this paper, we design an instruction roofline model for AMD GPUs
using AMD's ROCProfiler and a benchmarking tool, BabelStream (the HIP
implementation), as a way to measure an application's performance in
instructions and memory transactions on new AMD hardware. Specifically, we
create instruction roofline models for a case study scientific application,
PIConGPU, an open source particle-in-cell (PIC) simulations application used
for plasma and laser-plasma physics on the NVIDIA V100, AMD Radeon Instinct
MI60, and AMD Instinct MI100 GPUs. When looking at the performance of multiple
kernels of interest in PIConGPU we find that although the AMD MI100 GPU
achieves a similar, or better, execution time compared to the NVIDIA V100 GPU,
profiling tool differences make comparing performance of these two
architectures hard. When looking at execution time, GIPS, and instruction
intensity, the AMD MI60 achieves the worst performance out of the three GPUs
used in this work.

    

### [[2002.08101] The Sum of Its Parts: Analysis of Federated Byzantine Agreement Systems](http://arxiv.org/abs/2002.08101)


  Federated Byzantine Agreement Systems (FBASs) are a fascinating new paradigm
in the context of consensus protocols. Originally proposed for powering the
Stellar payment network, FBASs can instantiate Byzantine quorum systems without
requiring out-of-band agreement on a common set of validators; every node is
free to decide for itself with whom it requires agreement. Sybil-resistant and
yet energy-efficient consensus protocols can therefore be built upon FBASs, and
the "decentrality" possible with the FBAS paradigm might be sufficient to
reduce the use of environmentally unsustainable proof-of-work protocols. In
this paper, we first demonstrate how the robustness of individual FBASs can be
determined, by precisely determining their safety and liveness buffers and
therefore enabling a comparison with threshold-based quorum systems. Using
simulations and example node configuration strategies, we then empirically
investigate the hypothesis that while FBASs can be bootstrapped in a bottom-up
fashion from individual preferences, strategic considerations should
additionally be applied by node operators in order to arrive at FBASs that are
robust and amenable to monitoring. Finally, we investigate the reported
"open-membership" property of FBASs. We observe that an often small group of
nodes is exclusively relevant for determining safety and liveness buffers, and
prove that membership in this top tier is conditional on the approval by
current top tier nodes if maintaining safety is a core requirement.

    

### [[2007.10541] ZLB: A Blockchain to Tolerate Colluding Majorities](http://arxiv.org/abs/2007.10541)


  In the general setting, consensus cannot be solved if an adversary controls a
third of the system. Yet, blockchain participants typically reach consensus
"eventually" despite an adversary controlling a minority of the system.
Exceeding this $\frac{1}{3}$ cap is made possible by tolerating transient
disagreements, where distinct participants select distinct blocks for the same
index, before eventually agreeing to select the same block. Until now, no
blockchain could tolerate an attacker controlling a majority of the system.
In this paper, we present Zero-Loss Blockchain (ZLB), the first blockchain
that tolerates an adversary controlling more than half of the system. ZLB is an
open blockchain that combines recent theoretical advances in accountable
Byzantine agreement to exclude undeniably deceitful replicas. progressively
reduces the portion of deceitful replicas below $\frac{1}{3}$, and reaches
consensus. Geo-distributed experiments show that ZLB outperforms HotStuff and
is almost as fast as the scalable Red Belly Blockchain that cannot tolerate
$n/3$ faults.

    

### [[2110.07641] Non-deep Networks](http://arxiv.org/abs/2110.07641)


  Depth is the hallmark of deep neural networks. But more depth means more
sequential computation and higher latency. This begs the question -- is it
possible to build high-performing "non-deep" neural networks? We show that it
is. To do so, we use parallel subnetworks instead of stacking one layer after
another. This helps effectively reduce depth while maintaining high
performance. By utilizing parallel substructures, we show, for the first time,
that a network with a depth of just 12 can achieve top-1 accuracy over 80% on
ImageNet, 96% on CIFAR10, and 81% on CIFAR100. We also show that a network with
a low-depth (12) backbone can achieve an AP of 48% on MS-COCO. We analyze the
scaling rules for our design and show how to increase performance without
changing the network's depth. Finally, we provide a proof of concept for how
non-deep networks could be used to build low-latency recognition systems. Code
is available at this https URL.

    

### [[2110.07667] Interactive Analysis of CNN Robustness](http://arxiv.org/abs/2110.07667)


  While convolutional neural networks (CNNs) have found wide adoption as
state-of-the-art models for image-related tasks, their predictions are often
highly sensitive to small input perturbations, which the human vision is robust
against. This paper presents Perturber, a web-based application that allows
users to instantaneously explore how CNN activations and predictions evolve
when a 3D input scene is interactively perturbed. Perturber offers a large
variety of scene modifications, such as camera controls, lighting and shading
effects, background modifications, object morphing, as well as adversarial
attacks, to facilitate the discovery of potential vulnerabilities. Fine-tuned
model versions can be directly compared for qualitative evaluation of their
robustness. Case studies with machine learning experts have shown that
Perturber helps users to quickly generate hypotheses about model
vulnerabilities and to qualitatively compare model behavior. Using quantitative
analyses, we could replicate users' insights with other CNN architectures and
input images, yielding new insights about the vulnerability of adversarially
trained models.

    

### [[2110.07679] GlobalWoZ: Globalizing MultiWoZ to Develop Multilingual Task-Oriented Dialogue Systems](http://arxiv.org/abs/2110.07679)


  Much recent progress in task-oriented dialogue (ToD) systems has been driven
by available annotation data across multiple domains for training. Over the
last few years, there has been a move towards data curation for multilingual
ToD systems that are applicable to serve people speaking different languages.
However, existing multilingual ToD datasets either have a limited coverage of
languages due to the high cost of data curation, or ignore the fact that
dialogue entities barely exist in countries speaking these languages. To tackle
these limitations, we introduce a novel data curation method that generates
GlobalWoZ -- a large-scale multilingual ToD dataset globalized from an English
ToD dataset for three unexplored use cases. Our method is based on translating
dialogue templates and filling them with local entities in the target-language
countries. We release our dataset as well as a set of strong baselines to
encourage research on learning multilingual ToD systems for real use cases.

    

### [[2110.07686] Making Document-Level Information Extraction Right for the Right Reasons](http://arxiv.org/abs/2110.07686)


  Document-level information extraction is a flexible framework compatible with
applications where information is not necessarily localized in a single
sentence. For example, key features of a diagnosis in radiology a report may
not be explicitly stated, but nevertheless can be inferred from the report's
text. However, document-level neural models can easily learn spurious
correlations from irrelevant information. This work studies how to ensure that
these models make correct inferences from complex text and make those
inferences in an auditable way: beyond just being right, are these models
"right for the right reasons?" We experiment with post-hoc evidence extraction
in a predict-select-verify framework using feature attribution techniques.
While this basic approach can extract reasonable evidence, it can be
regularized with small amounts of evidence supervision during training, which
substantially improves the quality of extracted evidence. We evaluate on two
domains: a small-scale labeled dataset of brain MRI reports and a large-scale
modified version of DocRED (Yao et al., 2019) and show that models'
plausibility can be improved with no loss in accuracy.

    

### [[2110.07710] Semi-automated checking for regulatory compliance in e-Health](http://arxiv.org/abs/2110.07710)


  One of the main issues of every business process is to be compliant with
legal rules. This work presents a methodology to check in a semi-automated way
the regulatory compliance of a business process. We analyse an e-Health
hospital service in particular: the Hospital at Home (HaH) service. The paper
shows, at first, the analysis of the hospital business using the Business
Process Management and Notation (BPMN) standard language, then, the
formalization in Defeasible Deontic Logic (DDL) of some rules of the European
General Data Protection Regulation (GDPR). The aim is to show how to combine a
set of tasks of a business with a set of rules to be compliant with, using a
tool.

    

### [[2110.07717] Deep Human-guided Conditional Variational Generative Modeling for Automated Urban Planning](http://arxiv.org/abs/2110.07717)


  Urban planning designs land-use configurations and can benefit building
livable, sustainable, safe communities. Inspired by image generation, deep
urban planning aims to leverage deep learning to generate land-use
configurations. However, urban planning is a complex process. Existing studies
usually ignore the need of personalized human guidance in planning, and spatial
hierarchical structure in planning generation. Moreover, the lack of
large-scale land-use configuration samples poses a data sparsity challenge.
This paper studies a novel deep human guided urban planning method to jointly
solve the above challenges. Specifically, we formulate the problem into a deep
conditional variational autoencoder based framework. In this framework, we
exploit the deep encoder-decoder design to generate land-use configurations. To
capture the spatial hierarchy structure of land uses, we enforce the decoder to
generate both the coarse-grained layer of functional zones, and the
fine-grained layer of POI distributions. To integrate human guidance, we allow
humans to describe what they need as texts and use these texts as a model
condition input. To mitigate training data sparsity and improve model
robustness, we introduce a variational Gaussian embedding mechanism. It not
just allows us to better approximate the embedding space distribution of
training data and sample a larger population to overcome sparsity, but also
adds more probabilistic randomness into the urban planning generation to
improve embedding diversity so as to improve robustness. Finally, we present
extensive experiments to validate the enhanced performances of our method.

    

### [[2110.07722] The Sigma-Max System Induced from Randomness and Fuzziness](http://arxiv.org/abs/2110.07722)


  This paper managed to induce probability theory (sigma system) and
possibility theory (max system) respectively from randomness and fuzziness,
through which the premature theory of possibility is expected to be well
founded. Such an objective is achieved by addressing three open key issues: a)
the lack of clear mathematical definitions of randomness and fuzziness; b) the
lack of intuitive mathematical definition of possibility; c) the lack of
abstraction procedure of the axiomatic definitions of probability/possibility
from their intuitive definitions. Especially, the last issue involves the
question why the key axiom of "maxitivity" is adopted for possibility measure.
By taking advantage of properties of the well-defined randomness and fuzziness,
we derived the important conclusion that "max" is the only but un-strict
disjunctive operator that is applicable across the fuzzy event space, and is an
exact operator for fuzzy feature extraction that assures the max inference is
an exact mechanism. It is fair to claim that the long-standing problem of lack
of consensus to the foundation of possibility theory is well resolved, which
would facilitate wider adoption of possibility theory in practice and promote
cross prosperity of the two uncertainty theories of probability and
possibility.

    

### [[2110.07729] An Independent Study of Reinforcement Learning and Autonomous Driving](http://arxiv.org/abs/2110.07729)


  Reinforcement learning has become one of the most trending subjects in the
recent decade. It has seen applications in various fields such as robot
manipulations, autonomous driving, path planning, computer gaming, etc. We
accomplished three tasks during the course of this project. Firstly, we studied
the Q-learning algorithm for tabular environments and applied it successfully
to an OpenAi Gym environment, Taxi. Secondly, we gained an understanding of and
implemented the deep Q-network algorithm for Cart-Pole environment. Thirdly, we
also studied the application of reinforcement learning in autonomous driving
and its combination with safety check constraints (safety controllers). We
trained a rough autonomous driving agent using highway-gym environment and
explored the effects of various environment configurations like reward
functions on the agent training performance.

    

### [[2110.07796] Occupancy Estimation from Thermal Images](http://arxiv.org/abs/2110.07796)


  We propose a non-intrusive, and privacy-preserving occupancy estimation
system for smart environments. The proposed scheme uses thermal images to
detect the number of people in a given area. The occupancy estimation model is
designed using the concepts of intensity-based and motion-based human
segmentation. The notion of difference catcher, connected component labeling,
noise filter, and memory propagation are utilized to estimate the occupancy
number. We use a real dataset to demonstrate the effectiveness of the proposed
system.

    

### [[2110.07803] ContraQA: Question Answering under Contradicting Contexts](http://arxiv.org/abs/2110.07803)


  With a rise in false, inaccurate, and misleading information in propaganda,
news, and social media, real-world Question Answering (QA) systems face the
challenges of synthesizing and reasoning over contradicting information to
derive correct answers. This urgency gives rise to the need to make QA systems
robust to misinformation, a topic previously unexplored. We study the risk of
misinformation to QA models by investigating the behavior of the QA model under
contradicting contexts that are mixed with both real and fake information. We
create the first large-scale dataset for this problem, namely Contra-QA, which
contains over 10K human-written and model-generated contradicting pairs of
contexts. Experiments show that QA models are vulnerable under contradicting
contexts brought by misinformation. To defend against such a threat, we build a
misinformation-aware QA system as a counter-measure that integrates question
answering and misinformation detection in a joint fashion.

    

### [[2110.07816] Multilingual Neural Machine Translation:Can Linguistic Hierarchies Help?](http://arxiv.org/abs/2110.07816)


  Multilingual Neural Machine Translation (MNMT) trains a single NMT model that
supports translation between multiple languages, rather than training separate
models for different languages. Learning a single model can enhance the
low-resource translation by leveraging data from multiple languages. However,
the performance of an MNMT model is highly dependent on the type of languages
used in training, as transferring knowledge from a diverse set of languages
degrades the translation performance due to negative transfer. In this paper,
we propose a Hierarchical Knowledge Distillation (HKD) approach for MNMT which
capitalises on language groups generated according to typological features and
phylogeny of languages to overcome the issue of negative transfer. HKD
generates a set of multilingual teacher-assistant models via a selective
knowledge distillation mechanism based on the language groups, and then distils
the ultimate multilingual model from those assistants in an adaptive way.
Experimental results derived from the TED dataset with 53 languages demonstrate
the effectiveness of our approach in avoiding the negative transfer effect in
MNMT, leading to an improved translation performance (about 1 BLEU score on
average) compared to strong baselines.

    

### [[2110.07826] Machine Learning Algorithms In User Authentication Schemes](http://arxiv.org/abs/2110.07826)


  In the past two decades, the number of mobile products being created by
companies has grown exponentially. However, although these devices are
constantly being upgraded with the newest features, the security measures used
to protect these devices has stayed relatively the same over the past two
decades. The vast difference in growth patterns between devices and their
security is opening up the risk for more and more devices to easily become
infiltrated by nefarious users. Working off of previous work in the field, this
study looks at the different Machine Learning algorithms used in user
authentication schemes involving touch dynamics and device movement. This study
aims to give a comprehensive overview of the current uses of different machine
learning algorithms that are frequently used in user authentication schemas
involving touch dynamics and device movement. The benefits, limitations, and
suggestions for future work will be thoroughly discussed throughout this paper.

    

### [[2110.07867] Exploring Low-dimensional Intrinsic Task Subspace via Prompt Tuning](http://arxiv.org/abs/2110.07867)


  How can pre-trained language models (PLMs) learn universal representations
and effectively adapt to broad NLP tasks differing a lot superficially? In this
work, we empirically find evidences indicating that the adaptations of PLMs to
various tasks can be reparameterized as optimizing only a few free parameters
in a common low-dimensional intrinsic task subspace, which may help us
understand why PLMs could easily adapt to various NLP tasks with small-scale
data. Specifically, to find such a subspace and examine its universality, we
resort to the recent success of prompt tuning and decompose the soft prompts of
multiple NLP tasks into the same low-dimensional nonlinear subspace, then we
learn to adapt the PLM to unseen tasks or data by only tuning parameters in the
subspace. We dub this pipeline as intrinsic prompt tuning (IPT). In
experiments, we study diverse few-shot NLP tasks and surprisingly find that in
a 5-dimensional subspace found with 100 random tasks, by only tuning 5 free
parameters, we can recover 87% and 65% of the full prompt tuning performance
for 100 seen tasks (using different training data) and 20 unseen tasks,
respectively, showing great generalization ability of the found intrinsic task
subspace. Besides being an analysis tool, IPT could further bring practical
benefits, such as improving the prompt tuning stability.

    

### [[2110.07872] Role Similarity Metric Based on Spanning Rooted Forest](http://arxiv.org/abs/2110.07872)


  As a fundamental issue in network analysis, structural node similarity has
received much attention in academia and is adopted in a wide range of
applications. Among these proposed structural node similarity measures, role
similarity stands out because of satisfying several axiomatic properties
including automorphism conformation. Existing role similarity metrics cannot
handle top-k queries on large real-world networks due to the high time and
space cost. In this paper, we propose a new role similarity metric, namely
\textsf{ForestSim}. We prove that \textsf{ForestSim} is an admissible role
similarity metric and devise the corresponding top-k similarity search
algorithm, namely \textsf{ForestSimSearch}, which is able to process a top-k
query in $O(k)$ time once the precomputation is finished. Moreover, we speed up
the precomputation by using a fast approximate algorithm to compute the
diagonal entries of the forest matrix, which reduces the time and space
complexity of the precomputation to
$O(\epsilon^{-2}m\log^5{n}\log{\frac{1}{\epsilon}})$ and $O(m\log^3{n})$,
respectively. Finally, we conduct extensive experiments on 26 real-world
networks. The results show that \textsf{ForestSim} works efficiently on
million-scale networks and achieves comparable performance to the state-of-art
methods.

    

### [[2110.07895] A Machine Learning Approach for Delineating Similar Sound Symptoms of Respiratory Conditions on a Smartphone](http://arxiv.org/abs/2110.07895)


  Clinical characterization and interpretation of respiratory sound symptoms
have remained a challenge due to the similarities in the audio properties that
manifest during auscultation in medical diagnosis. The misinterpretation and
conflation of these sounds coupled with the comorbidity cases of the associated
ailments particularly, exercised-induced respiratory conditions; result in the
under-diagnosis and under-treatment of the conditions. Though several studies
have proposed computerized systems for objective classification and evaluation
of these sounds, most of the algorithms run on desktop and backend systems. In
this study, we leverage the improved computational and storage capabilities of
modern smartphones to distinguish the respiratory sound symptoms using machine
learning algorithms namely: Random Forest (RF), Support Vector Machine (SVM),
and k-Nearest Neighbour (k-NN). The appreciable performance of these
classifiers on a mobile phone shows smartphone as an alternate tool for
recognition and discrimination of respiratory symptoms in real-time scenarios.
Further, the objective clinical data provided by the machine learning process
could aid physicians in the screening and treatment of a patient during
ambulatory care where specialized medical devices may not be readily available.

    

### [[2110.07898] Certainty Modeling of a Decision Support System for Mobile Monitoring of Exercise induced Respiratory Conditions](http://arxiv.org/abs/2110.07898)


  Mobile health systems in recent times, have notably improved the healthcare
sector by empowering patients to actively participate in their health, and by
facilitating access to healthcare professionals. Effective operation of these
mobile systems nonetheless, requires high level of intelligence and expertise
implemented in the form of decision support systems (DSS). However, common
challenges in the implementation include generalization and reliability, due to
the dynamics and incompleteness of information presented to the inference
models. In this paper, we advance the use of ad hoc mobile decision support
system to monitor and detect triggers and early symptoms of respiratory
distress provoked by strenuous physical exertion. The focus is on the
application of certainty theory to model inexact reasoning by the mobile
monitoring system. The aim is to develop a mobile tool to assist patients in
managing their conditions, and to provide objective clinical data to aid
physicians in the screening, diagnosis, and treatment of the respiratory
ailments. We present the proposed model architecture and then describe an
application scenario in a clinical setting. We also show implementation of an
aspect of the system that enables patients in the self-management of their
conditions.

    

### [[2110.07953] Estimation and Prediction of Deterministic Human Intent Signal to augment Haptic Glove aided Control of Robotic Hand](http://arxiv.org/abs/2110.07953)


  The paper focuses on Haptic Glove (HG) based control of a Robotic Hand (RH)
executing in-hand manipulation. A control algorithm is presented to allow the
RH relocate the object held to a goal pose. The motion signals for both the HG
and the RH are high dimensional. The RH kinematics is usually different from
the HG kinematics. The variability of kinematics of the two devices, added with
the incomplete information about the human hand kinematics result in difficulty
in direct mapping of the high dimensional motion signal of the HG to the RH.
Hence, a method is proposed to estimate the human intent from the high
dimensional HG motion signal and reconstruct the signal at the RH to ensure
object relocation. It is also shown that the lag in synthesis of the motion
signal of the human hand added with the control latency of the RH leads to a
requirement of the prediction of the human intent signal. Then, a recurrent
neural network (RNN) is proposed to predict the human intent signal ahead of
time.

    

### [[2110.08003] A Broad-persistent Advising Approach for Deep Interactive Reinforcement Learning in Robotic Environments](http://arxiv.org/abs/2110.08003)


  Deep Reinforcement Learning (DeepRL) methods have been widely used in
robotics to learn about the environment and acquire behaviors autonomously.
Deep Interactive Reinforcement Learning (DeepIRL) includes interactive feedback
from an external trainer or expert giving advice to help learners choosing
actions to speed up the learning process. However, current research has been
limited to interactions that offer actionable advice to only the current state
of the agent. Additionally, the information is discarded by the agent after a
single use that causes a duplicate process at the same state for a revisit. In
this paper, we present Broad-persistent Advising (BPA), a broad-persistent
advising approach that retains and reuses the processed information. It not
only helps trainers to give more general advice relevant to similar states
instead of only the current state but also allows the agent to speed up the
learning process. We test the proposed approach in two continuous robotic
scenarios, namely, a cart pole balancing task and a simulated robot navigation
task. The obtained results show that the performance of the agent using BPA
improves while keeping the number of interactions required for the trainer in
comparison to the DeepIRL approach.

    

### [[2110.08012] A Survey on State-of-the-art Techniques for Knowledge Graphs Construction and Challenges ahead](http://arxiv.org/abs/2110.08012)


  Global datasphere is increasing fast, and it is expected to reach 175
Zettabytes by 20251 . However, most of the content is unstructured and is not
understandable by machines. Structuring this data into a knowledge graph
enables multitudes of intelligent applications such as deep question answering,
recommendation systems, semantic search, etc. The knowledge graph is an
emerging technology that allows logical reasoning and uncovers new insights
using content along with the context. Thereby, it provides necessary syntax and
reasoning semantics that enable machines to solve complex healthcare, security,
financial institutions, economics, and business problems. As an outcome,
enterprises are putting their effort into constructing and maintaining
knowledge graphs to support various downstream applications. Manual approaches
are too expensive. Automated schemes can reduce the cost of building knowledge
graphs up to 15-250 times. This paper critiques state-of-the-art automated
techniques to produce knowledge graphs of near-human quality autonomously.
Additionally, it highlights different research issues that need to be addressed
to deliver high-quality knowledge graphs

    

### [[2110.08016] Efficiently Solve the Max-cut Problem via a Quantum Qubit Rotation Algorithm](http://arxiv.org/abs/2110.08016)


  Optimizing parameterized quantum circuits promises efficient use of near-term
quantum computers to achieve the potential quantum advantage. However, there is
a notorious tradeoff between the expressibility and trainability of the
parameter ansatz. We find that in combinatorial optimization problems, since
the solutions are described by bit strings, one can trade the expressiveness of
the ansatz for high trainability. To be specific, by focusing on the max-cut
problem we introduce a simple yet efficient algorithm named Quantum Qubit
Rotation Algorithm (QQRA). The quantum circuits are comprised with single-qubit
rotation gates implementing on each qubit. The rotation angles of the gates can
be trained free of barren plateaus. Thus, the approximate solution of the
max-cut problem can be obtained with probability close to 1. To illustrate the
effectiveness of QQRA, we compare it with the well known quantum approximate
optimization algorithm and the classical Goemans-Williamson algorithm.

    

### [[2110.08036] Generating Natural Language Adversarial Examples through An Improved Beam Search Algorithm](http://arxiv.org/abs/2110.08036)


  The research of adversarial attacks in the text domain attracts many
interests in the last few years, and many methods with a high attack success
rate have been proposed. However, these attack methods are inefficient as they
require lots of queries for the victim model when crafting text adversarial
examples. In this paper, a novel attack model is proposed, its attack success
rate surpasses the benchmark attack methods, but more importantly, its attack
efficiency is much higher than the benchmark attack methods. The novel method
is empirically evaluated by attacking WordCNN, LSTM, BiLSTM, and BERT on four
benchmark datasets. For instance, it achieves a 100\% attack success rate
higher than the state-of-the-art method when attacking BERT and BiLSTM on IMDB,
but the number of queries for the victim models only is 1/4 and 1/6.5 of the
state-of-the-art method, respectively. Also, further experiments show the novel
method has a good transferability on the generated adversarial examples.

    

### [[2110.08049] Learning Semantics: An Opportunity for Effective 6G Communications](http://arxiv.org/abs/2110.08049)


  Recently, semantic communications are envisioned as a key enabler of future
6G networks. Back to Shannon's information theory, the goal of communication
has long been to guarantee the correct reception of transmitted messages
irrespective of their meaning. However, in general, whenever communication
occurs to convey a meaning, what matters is the receiver's understanding of the
transmitted message and not necessarily its correct reconstruction. Hence,
semantic communications introduce a new paradigm: transmitting only relevant
information sufficient for the receiver to capture the meaning intended can
save significant communication bandwidth. Thus, this work explores the
opportunity offered by semantic communications for beyond 5G networks. In
particular, we focus on the benefit of semantic compression. We refer to
semantic message as a sequence of well-formed symbols learned from the
"meaning" underlying data, which have to be interpreted at the receiver. This
requires a reasoning unit, here artificial, on a knowledge base: a symbolic
knowledge representation of the specific application. Therefore, we present and
detail a novel architecture that enables representation learning of semantic
symbols for effective semantic communications. We first discuss theoretical
aspects and successfully design objective functions, which help learn effective
semantic encoders and decoders. Eventually, we show promising numerical results
for the scenario of text transmission, especially when the sender and receiver
speak different languages.

    

### [[2110.08068] SAT Encodings for Pseudo-Boolean Constraints Together With At-Most-One Constraints](http://arxiv.org/abs/2110.08068)


  When solving a combinatorial problem using propositional satisfiability
(SAT), the encoding of the problem is of vital importance. We study encodings
of Pseudo-Boolean (PB) constraints, a common type of arithmetic constraint that
appears in a wide variety of combinatorial problems such as timetabling,
scheduling, and resource allocation. In some cases PB constraints occur
together with at-most-one (AMO) constraints over subsets of their variables
(forming PB(AMO) constraints). Recent work has shown that taking account of
AMOs when encoding PB constraints using decision diagrams can produce a
dramatic improvement in solver efficiency. In this paper we extend the approach
to other state-of-the-art encodings of PB constraints, developing several new
encodings for PB(AMO) constraints. Also, we present a more compact and
efficient version of the popular Generalized Totalizer encoding, named Reduced
Generalized Totalizer. This new encoding is also adapted for PB(AMO)
constraints for a further gain. Our experiments show that the encodings of
PB(AMO) constraints can be substantially smaller than those of PB constraints.
PB(AMO) encodings allow many more instances to be solved within a time limit,
and solving time is improved by more than one order of magnitude in some cases.
We also observed that there is no single overall winner among the considered
encodings, but efficiency of each encoding may depend on PB(AMO)
characteristics such as the magnitude of coefficient values.

    

### [[2110.08079] Automated Quality Control of Vacuum Insulated Glazing by Convolutional Neural Network Image Classification](http://arxiv.org/abs/2110.08079)


  Vacuum Insulated Glazing (VIG) is a highly thermally insulating window
technology, which boasts an extremely thin profile and lower weight as compared
to gas-filled insulated glazing units of equivalent performance. The VIG is a
double-pane configuration with a submillimeter vacuum gap between the panes and
therefore under constant atmospheric pressure over their service life. Small
pillars are positioned between the panes to maintain the gap, which can damage
the glass reducing the lifetime of the VIG unit. To efficiently assess any
surface damage on the glass, an automated damage detection system is highly
desirable. For the purpose of classifying the damage, we have developed,
trained, and tested a deep learning computer vision system using convolutional
neural networks. The classification model flawlessly classified the test
dataset with an area under the curve (AUC) for the receiver operating
characteristic (ROC) of 100%. We have automatically cropped the images down to
their relevant information by using Faster-RCNN to locate the position of the
pillars. We employ the state-of-the-art methods Grad-CAM and Score-CAM of
explainable Artificial Intelligence (XAI) to provide an understanding of the
internal mechanisms and were able to show that our classifier outperforms
ResNet50V2 for identification of crack locations and geometry. The proposed
methods can therefore be used to detect systematic defects even without large
amounts of training data. Further analyses of our model's predictive
capabilities demonstrates its superiority over state-of-the-art models
(ResNet50V2, ResNet101V2 and ResNet152V2) in terms of convergence speed,
accuracy, precision at 100% recall and AUC for ROC.

    

### [[2110.08090] Using DeepProbLog to perform Complex Event Processing on an Audio Stream](http://arxiv.org/abs/2110.08090)


  In this paper, we present an approach to Complex Event Processing (CEP) that
is based on DeepProbLog. This approach has the following objectives: (i)
allowing the use of subsymbolic data as an input, (ii) retaining the
flexibility and modularity on the definitions of complex event rules, (iii)
allowing the system to be trained in an end-to-end manner and (iv) being robust
against noisily labelled data. Our approach makes use of DeepProbLog to create
a neuro-symbolic architecture that combines a neural network to process the
subsymbolic data with a probabilistic logic layer to allow the user to define
the rules for the complex events. We demonstrate that our approach is capable
of detecting complex events from an audio stream. We also demonstrate that our
approach is capable of training even with a dataset that has a moderate
proportion of noisy data.

    

### [[2110.08118] Few-Shot Bot: Prompt-Based Learning for Dialogue Systems](http://arxiv.org/abs/2110.08118)


  Learning to converse using only a few examples is a great challenge in
conversational AI. The current best conversational models, which are either
good chit-chatters (e.g., BlenderBot) or goal-oriented systems (e.g., MinTL),
are language models (LMs) fine-tuned on large conversational datasets. Training
these models is expensive, both in terms of computational resources and time,
and it is hard to keep them up to date with new conversational skills. A simple
yet unexplored solution is prompt-based few-shot learning (Brown et al. 2020)
which does not require gradient-based fine-tuning but instead uses a few
examples in the LM context as the only source of learning. In this paper, we
explore prompt-based few-shot learning in dialogue tasks. We benchmark LMs of
different sizes in nine response generation tasks, which include four
knowledge-grounded tasks, a task-oriented generations task, three open-chat
tasks, and controlled stylistic generation, and five conversational parsing
tasks, which include dialogue state tracking, graph path generation, persona
information extraction, document retrieval, and internet query generation. The
current largest released LM (GPT-J-6B) using prompt-based few-shot learning,
and thus requiring no training, achieves competitive performance to fully
trained state-of-the-art models. Moreover, we propose a novel prompt-based
few-shot classifier, that also does not require any fine-tuning, to select the
most appropriate prompt given a dialogue history. Finally, by combining the
power of prompt-based few-shot learning and a Skill Selector, we create an
end-to-end chatbot named the Few-Shot Bot (FSB), which automatically selects
the most appropriate conversational skill, queries different knowledge bases or
the internet, and uses the retrieved knowledge to generate a human-like
response, all using only few dialogue examples per skill.

    

### [[2110.08122] Effects of Different Optimization Formulations in Evolutionary Reinforcement Learning on Diverse Behavior Generation](http://arxiv.org/abs/2110.08122)


  Generating various strategies for a given task is challenging. However, it
has already proven to bring many assets to the main learning process, such as
improved behavior exploration. With the growth in the interest of heterogeneity
in solution in evolutionary computation and reinforcement learning, many
promising approaches have emerged. To better understand how one guides multiple
policies toward distinct strategies and benefit from diversity, we need to
analyze further the influence of the reward signal modulation and other
evolutionary mechanisms on the obtained behaviors. To that effect, this paper
considers an existing evolutionary reinforcement learning framework which
exploits multi-objective optimization as a way to obtain policies that succeed
at behavior-related tasks as well as completing the main goal. Experiments on
the Atari games stress that optimization formulations which do not consider
objectives equally fail at generating diversity and even output agents that are
worse at solving the problem at hand, regardless of the obtained behaviors.

    

### [[2110.08124] Decentralized Cooperative Lane Changing at Freeway Weaving Areas Using Multi-Agent Deep Reinforcement Learning](http://arxiv.org/abs/2110.08124)


  Frequent lane changes during congestion at freeway bottlenecks such as merge
and weaving areas further reduce roadway capacity. The emergence of deep
reinforcement learning (RL) and connected and automated vehicle technology
provides a possible solution to improve mobility and energy efficiency at
freeway bottlenecks through cooperative lane changing. Deep RL is a collection
of machine-learning methods that enables an agent to improve its performance by
learning from the environment. In this study, a decentralized cooperative
lane-changing controller was developed using proximal policy optimization by
adopting a multi-agent deep RL paradigm. In the decentralized control strategy,
policy learning and action reward are evaluated locally, with each agent
(vehicle) getting access to global state information. Multi-agent deep RL
requires lower computational resources and is more scalable than single-agent
deep RL, making it a powerful tool for time-sensitive applications such as
cooperative lane changing. The results of this study show that cooperative lane
changing enabled by multi-agent deep RL yields superior performance to human
drivers in term of traffic throughput, vehicle speed, number of stops per
vehicle, vehicle fuel efficiency, and emissions. The trained RL policy is
transferable and can be generalized to uncongested, moderately congested, and
extremely congested traffic conditions.

    

### [[2110.08125] Towards a Multi-Agent System Architecture for Supply Chain Management](http://arxiv.org/abs/2110.08125)


  Individual business processes have been changing since the Internet was
created, and they are now oriented towards a more distributed and collaborative
business model, in an e-commerce environment that adapts itself to the
competitive and changing market conditions. This paper presents a multi-agent
system architecture for supply chain management, which explores different
strategies and offers solutions in a distributed e-commerce environment. The
system is designed to support different types of interfaces, which allow
interoperating with other business models already developed. In order to show
how the entire multi-agent system is being developed, the implementation of a
collaborative agent is presented and explained.

    

### [[2110.08144] Integrating diverse extraction pathways using iterative predictions for Multilingual Open Information Extraction](http://arxiv.org/abs/2110.08144)


  In this paper we investigate a simple hypothesis for the Open Information
Extraction (OpenIE) task, that it may be easier to extract some elements of an
triple if the extraction is conditioned on prior extractions which may be
easier to extract. We successfully exploit this and propose a neural
multilingual OpenIE system that iteratively extracts triples by conditioning
extractions on different elements of the triple leading to a rich set of
extractions. The iterative nature of MiLIE also allows for seamlessly
integrating rule based extraction systems with a neural end-to-end system
leading to improved performance. MiLIE outperforms SOTA systems on multiple
languages ranging from Chinese to Galician thanks to it's ability of combining
multiple extraction pathways. Our analysis confirms that it is indeed true that
certain elements of an extraction are easier to extract than others. Finally,
we introduce OpenIE evaluation datasets for two low resource languages namely
Japanese and Galician.

    

### [[2110.08170] Simulation of emergence in artificial societies: a practical model-based approach with the EB-DEVS formalism](http://arxiv.org/abs/2110.08170)


  Modelling and simulation of complex systems is key to exploring and
understanding social processes, benefiting from formal mechanisms to derive
global-level properties from local-level interactions. In this paper we extend
the body of knowledge on formal methods in complex systems by applying EB-DEVS,
a novel formalism tailored for the modelling, simulation and live
identification of emergent properties. We guide the reader through the
implementation of different classical models for varied social systems to
introduce good modelling practices and showcase the advantages and limitations
of modelling emergence with EB-DEVS, in particular through its live emergence
detection capability. This work provides case study-driven evidence for the
neatness and compactness of the approach to modelling communication structures
that can be explicit or implicit, static or dynamic, with or without multilevel
interactions, and with weak or strong emergent behaviour. Throughout examples
we show that EB-DEVS permits conceptualising the analysed societies by
incorporating emergent behaviour when required, namely by integrating as a
macro-level aggregate the Gini index in the Sugarscape model, Fads and Fashion
in the Dissemination of Culture model, size-biased degree distribution in a
Preferential Attachment model, happiness index in the Segregation model and
quarantines in the SIR epidemic model. In each example we discuss the role of
communication structures in the development of multilevel simulation models,
and illustrate how micro-macro feedback loops enable the modelling of
macro-level properties. Our results stress the relevance of multilevel features
to support a robust approach in the modelling and simulation of complex
systems.

    

### [[2110.08187] Crop Rotation Modeling for Deep Learning-Based Parcel Classification from Satellite Time Series](http://arxiv.org/abs/2110.08187)


  While annual crop rotations play a crucial role for agricultural
optimization, they have been largely ignored for automated crop type mapping.
In this paper, we take advantage of the increasing quantity of annotated
satellite data to propose the first deep learning approach modeling
simultaneously the inter- and intra-annual agricultural dynamics of parcel
classification. Along with simple training adjustments, our model provides an
improvement of over 6.6 mIoU points over the current state-of-the-art of crop
classification. Furthermore, we release the first large-scale multi-year
agricultural dataset with over 300,000 annotated parcels.

    

### [[2110.08228] Cross-Domain Data Integration for Named Entity Disambiguation in Biomedical Text](http://arxiv.org/abs/2110.08228)


  Named entity disambiguation (NED), which involves mapping textual mentions to
structured entities, is particularly challenging in the medical domain due to
the presence of rare entities. Existing approaches are limited by the presence
of coarse-grained structural resources in biomedical knowledge bases as well as
the use of training datasets that provide low coverage over uncommon resources.
In this work, we address these issues by proposing a cross-domain data
integration method that transfers structural knowledge from a general text
knowledge base to the medical domain. We utilize our integration scheme to
augment structural resources and generate a large biomedical NED dataset for
pretraining. Our pretrained model with injected structural knowledge achieves
state-of-the-art performance on two benchmark medical NED datasets: MedMentions
and BC5CDR. Furthermore, we improve disambiguation of rare entities by up to 57
accuracy points.

    

### [[2110.08247] Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks](http://arxiv.org/abs/2110.08247)


  Backdoor attacks are a kind of emergent security threat in deep learning.
When a deep neural model is injected with a backdoor, it will behave normally
on standard inputs but give adversary-specified predictions once the input
contains specific backdoor triggers. Current textual backdoor attacks have poor
attack performance in some tough situations. In this paper, we find two simple
tricks that can make existing textual backdoor attacks much more harmful. The
first trick is to add an extra training task to distinguish poisoned and clean
data during the training of the victim model, and the second one is to use all
the clean training data rather than remove the original clean data
corresponding to the poisoned data. These two tricks are universally applicable
to different attack models. We conduct experiments in three tough situations
including clean data fine-tuning, low poisoning rate, and label-consistent
attacks. Experimental results show that the two tricks can significantly
improve attack performance. This paper exhibits the great potential harmfulness
of backdoor attacks. All the code and data will be made public to facilitate
further research.

    

### [[2110.07663] Auto-Tuned Preconditioners for the Spectral Element Method on GPUs](http://arxiv.org/abs/2110.07663)


  The Poisson pressure solve resulting from the spectral element discretization
of the incompressible Navier-Stokes equation requires fast, robust, and
scalable preconditioning. In the current work, a parallel scaling study of
Chebyshev-accelerated Schwarz and Jacobi preconditioning schemes is presented,
with special focus on GPU architectures, such as OLCF's Summit. Convergence
properties of the Chebyshev-accelerated schemes are compared with alternative
methods, such as low-order preconditioners combined with algebraic multigrid.
Performance and scalability results are presented for a variety of
preconditioner and solver settings. The authors demonstrate that
Chebyshev-accelerated-Schwarz methods provide a robust and effective smoothing
strategy when using $p$-multigrid as a preconditioner in a Krylov-subspace
projector. At the same time, optimal preconditioning parameters can vary for
different geometries, problem sizes, and processor counts. This variance
motivates the development of an autotuner to optimize solver parameters
on-line, during the course of production simulations.

    

### [[2110.08154] Distributed Resource Allocation Optimization for User-Centric Cell-Free MIMO Networks](http://arxiv.org/abs/2110.08154)


  We develop two distributed downlink resource allocation algorithms for
user-centric, cell-free, spatially-distributed, multiple-input multiple-output
(MIMO) networks. In such networks, each user is served by a subset of nearby
transmitters that we call distributed units or DUs. The operation of the DUs in
a region is controlled by a central unit (CU). Our first scheme is implemented
at the DUs, while the second is implemented at the CUs controlling these DUs.
We define a hybrid quality of service metric that enables distributed
optimization of system resources in a proportional fair manner. Specifically,
each of our algorithms performs user scheduling, beamforming, and power control
while accounting for channel estimation errors. Importantly, our algorithm does
not require information exchange amongst DUs (CUs) for the DU-distributed
(CU-distributed) system, while also smoothly converging. Our results show that
our CU-distributed system provides 1.3- to 1.8-fold network throughput compared
to the DU-distributed system, with minor increases in complexity and front-haul
load - and substantial gains over benchmark schemes like local zero-forcing. We
also analyze the trade-offs provided by the CU-distributed system, hence
highlighting the significance of deploying multiple CUs in user-centric
cell-free networks.

    

### [[2110.07811] Cascaded Fast and Slow Models for Efficient Semantic Code Search](http://arxiv.org/abs/2110.07811)


  The goal of natural language semantic code search is to retrieve a
semantically relevant code snippet from a fixed set of candidates using a
natural language query. Existing approaches are neither effective nor efficient
enough towards a practical semantic code search system. In this paper, we
propose an efficient and accurate semantic code search framework with cascaded
fast and slow models, in which a fast transformer encoder model is learned to
optimize a scalable index for fast retrieval followed by learning a slow
classification-based re-ranking model to improve the performance of the top K
results from the fast retrieval. To further reduce the high memory cost of
deploying two separate models in practice, we propose to jointly train the fast
and slow model based on a single transformer encoder with shared parameters.
The proposed cascaded approach is not only efficient and scalable, but also
achieves state-of-the-art results with an average mean reciprocal ranking (MRR)
score of 0.7795 (across 6 programming languages) as opposed to the previous
state-of-the-art result of 0.713 MRR on the CodeSearchNet benchmark.

    

### [[2110.07902] Zipping Strategies and Attribute Grammars](http://arxiv.org/abs/2110.07902)


  Strategic term rewriting and attribute grammars are two powerful programming
techniques widely used in language engineering. The former, relies on
strategies to apply term rewrite rules in defining language transformations,
while the latter is suitable to express context-dependent language processing
algorithms. Each of these techniques, however, is usually implemented by its
own powerful and large language processor system. As a result, it makes such
systems harder to extend and to combine.
In this paper, we present the embedding of both strategic tree rewriting and
attribute grammars in a zipper-based, purely functional setting. Zippers
provide a simple, but generic tree-walk mechanism that is the building block
technique we use to express the purely-functional embedding of both techniques.
The embedding of the two techniques in the same setting has several advantages:
First, we easily combine/zip attribute grammars and strategies, thus providing
language engineers the best of the two worlds. Second, the combined embedding
is easier to maintain and extend since it is written in a concise and uniform
setting. This results in a very small library which is able to express advanced
(static) analysis and transformation tasks. We show the expressive power of our
library in optimizing |Haskell| let expressions, expressing several |Haskell|
refactorings and solving several language processing tasks of the LDTA Tool
Challenge.

    

### [<title>Huge AUC drop when upgrading from XGBoost4J-Spark 0.90 - XGBoost</title>](https://discuss.xgboost.ai/t/huge-auc-drop-when-upgrading-from-xgboost4j-spark-0-90/2500/1)