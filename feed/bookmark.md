
## 2021-10-15

### [<title>XGBoost iteration_range defined differently in sklearn API and docs - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-iteration-range-defined-differently-in-sklearn-api-and-docs/2495/3)

### [<title>XGBoost iteration_range defined differently in sklearn API and docs - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-iteration-range-defined-differently-in-sklearn-api-and-docs/2495/2)

### [[2110.07050] Competitive Multi-Agent Load Balancing with Adaptive Policies in Wireless Networks](http://arxiv.org/abs/2110.07050)


  Using Machine Learning (ML) techniques for the next generation wireless
networks have shown promising results in the recent years, due to high learning
and adaptation capability of ML algorithms. More specifically, ML techniques
have been used for load balancing in Self-Organizing Networks (SON). In the
context of load balancing and ML, several studies propose network management
automation (NMA) from the perspective of a single and centralized agent.
However, a single agent domain does not consider the interaction among the
agents. In this paper, we propose a more realistic load balancing approach
using novel Multi-Agent Deep Deterministic Policy Gradient with Adaptive
Policies (MADDPG-AP) scheme that considers throughput, resource block
utilization and latency in the network. We compare our proposal with a
single-agent RL algorithm named Clipped Double Q-Learning (CDQL) . Simulation
results reveal a significant improvement in latency, packet loss ratio and
convergence time

    

### [[2110.07077] Federated Learning Over Cellular-Connected UAV Networks with Non-IID Datasets](http://arxiv.org/abs/2110.07077)


  Federated learning (FL) is a promising distributed learning technique
particularly suitable for wireless learning scenarios since it can accomplish a
learning task without raw data transportation so as to preserve data privacy
and lower network resource consumption. However, current works on FL over
wireless communication do not profoundly study the fundamental performance of
FL that suffers from data delivery outage due to network interference and data
heterogeneity among mobile clients. To accurately exploit the performance of FL
over wireless communication, this paper proposes a new FL model over a
cellular-connected unmanned aerial vehicle (UAV) network, which characterizes
data delivery outage from UAV clients to their server and data heterogeneity
among the datasets of UAV clients. We devise a simulation-based approach to
evaluating the convergence performance of the proposed FL model. We then
propose a tractable analytical framework of the uplink outage probability in
the cellular-connected UAV network and derive a neat expression of the uplink
outage probability, which reveals how the proposed FL model is impacted by data
delivery outage and UAV deployment. Extensive numerical simulations are
conducted to show the consistency between the estimated and simulated
performances.

    

### [[2110.07365] DynoLoc: Infrastructure-free RF Tracking in Dynamic Indoor Environments](http://arxiv.org/abs/2110.07365)


  Promising solutions exist today that can accurately track mobile entities
indoor using visual inertial odometry in favorable visual conditions, or by
leveraging fine-grained ranging (RF, ultrasonic, IR, etc.) to reference
anchors. However, they are unable to directly cater to "dynamic" indoor
environments (e.g. first responder scenarios, multi-player AR/VR gaming in
everyday spaces, etc.) that are devoid of such favorable conditions. Indeed, we
show that the need for "infrastructure-free", and robustness to "node mobility"
and "visual conditions" in such environments, motivates a robust RF-based
approach along with the need to address a novel and challenging variant of its
infrastructure-free (i.e. peer-to-peer) localization problem that is
latency-bounded - accurate tracking of mobile entities imposes a latency budget
that not only affects the solution computation but also the collection of
peer-to-peer ranges themselves.
In this work, we present the design and deployment of DynoLoc that addresses
this latency-bounded infrastructure-free RF localization problem. To this end,
DynoLoc unravels the fundamental tradeoff between latency and localization
accuracy and incorporates design elements that judiciously leverage the
available ranging resources to adaptively estimate the joint topology of nodes,
coupled with robust algorithm that maximizes the localization accuracy even in
the face of practical environmental artifacts (wireless connectivity and
multipath, node mobility, etc.). This allows DynoLoc to track (every second) a
network of few tens of mobile entities even at speeds of 1-2 m/s with median
accuracies under 1-2 m (compared to 5m+ with baselines), without infrastructure
support. We demonstrate DynoLoc's potential in a real-world firefighters'
drill, as well as two other use cases of (i) multi-player AR/VR gaming, and
(ii) active shooter tracking by first responders.

    

### [[2110.07424] Towards a modern CMake workflow](http://arxiv.org/abs/2110.07424)


  Modern CMake offers the features to manage versatile and complex projects
with ease. With respect to OMNeT++ projects, a workflow relying on CMake
enables projects to combine discrete event simulation and production code in a
common development environment. Such a combination means less maintenance
effort and thus potentially more sustainable and long-living software. This
paper highlights the significant improvements since the first attempt of using
CMake in OMNeT++ projects. In particular, a state-of-the-art integration of
OMNeT++ in Visual Studio Code including support for debugging and
multi-platform compilation is presented. Last but not least, an exemplary use
case demonstrates the powerful mix of production and simulation code in a
common software architecture supported by the OMNeT++ CMake package.

    

### [[2110.07547] Reconfigurable, Intelligent, and SustainableWireless Environments for 6G Smart Connectivity](http://arxiv.org/abs/2110.07547)


  Various visions on the forthcoming sixth Generation (6G) networks point
towards flexible connect-and-compute technologies to support future innovative
services and the corresponding use cases. 6G should be capable to accommodate
ever-evolving and heterogeneous applications, future regulations, and diverse
user-, service-, and location-based requirements. A key element towards
building smart and energy sustainable wireless systems beyond 5G is the
Reconfigurable Intelligent Surface (RIS), which offers programmable control and
shaping of the wireless propagation environment.
Capitalizing on this technology potential, in this article we introduce two
new concepts: i) wireless environment as a service, which leverages a novel
RIS-empowered networking paradigm to trade off diverse, and usually
conflicting, connectivity objectives; and ii) performance-boosted areas enabled
by RIS-based connectivity, representing competing service provisioning areas
that are highly spatially and temporally focused. We discuss the key
technological enablers and research challenges with the proposed networking
paradigm, and highlight the potential profound role of RISs in the recent Open
Radio Access Network (O-RAN) architecture.

    

### [[2110.07567] Resource-constrained Federated Edge Learning with Heterogeneous Data: Formulation and Analysis](http://arxiv.org/abs/2110.07567)


  Efficient collaboration between collaborative machine learning and wireless
communication technology, forming a Federated Edge Learning (FEEL), has spawned
a series of next-generation intelligent applications. However, due to the
openness of network connections, the FEEL framework generally involves hundreds
of remote devices (or clients), resulting in expensive communication costs,
which is not friendly to resource-constrained FEEL. To address this issue, we
propose a distributed approximate Newton-type algorithm with fast convergence
speed to alleviate the problem of FEEL resource (in terms of communication
resources) constraints. Specifically, the proposed algorithm is improved based
on distributed L-BFGS algorithm and allows each client to approximate the
high-cost Hessian matrix by computing the low-cost Fisher matrix in a
distributed manner to find a "better" descent direction, thereby speeding up
convergence. Second, we prove that the proposed algorithm has linear
convergence in strongly convex and non-convex cases and analyze its
computational and communication complexity. Similarly, due to the heterogeneity
of the connected remote devices, FEEL faces the challenge of heterogeneous data
and non-IID (Independent and Identically Distributed) data. To this end, we
design a simple but elegant training scheme, namely FedOVA, to solve the
heterogeneous statistical challenge brought by heterogeneous data. In this way,
FedOVA first decomposes a multi-class classification problem into more
straightforward binary classification problems and then combines their
respective outputs using ensemble learning. In particular, the scheme can be
well integrated with our communication efficient algorithm to serve FEEL.
Numerical results verify the effectiveness and superiority of the proposed
algorithm.

    

### [[2106.03504] Boosting 5G mm-Wave IAB Reliability with Reconfigurable Intelligent Surfaces](http://arxiv.org/abs/2106.03504)


  The introduction of the mm-Wave spectrum into 5G NR promises to bring about
unprecedented data throughput to future mobile wireless networks but comes with
several challenges. Network densification has been proposed as a viable
solution to increase RAN resilience, and the newly introduced IAB is considered
a key enabling technology with compelling cost-reducing opportunities for such
dense deployments. Reconfigurable Intelligent Surfaces (RIS) have recently
gained extreme popularity as they can create Smart Radio Environments by EM
wave manipulation and behave as inexpensive passive relays. However, it is not
yet clear what role this technology can play in a large RAN deployment. With
the scope of filling this gap, we study the blockage resilience of realistic
mm-Wave RAN deployments that use IAB and RIS. The RAN layouts have been
optimised by means of a novel mm-Wave planning tool based on MILP formulation.
Numerical results show how adding RISs to IAB deployments can provide high
blockage resistance levels while significantly reducing the overall network
planning cost.

    

### [[2106.13950] Self-Evolving Integrated Vertical Heterogeneous Networks](http://arxiv.org/abs/2106.13950)


  6G and beyond networks tend towards fully intelligent and adaptive design in
order to provide better operational agility in maintaining universal wireless
access and supporting a wide range of services and use cases while dealing with
network complexity efficiently. Such enhanced network agility will require
developing a self-evolving capability in designing both the network
architecture and resource management to intelligently utilize resources, reduce
operational costs, and achieve the coveted quality of service (QoS). To enable
this capability, the necessity of considering an integrated vertical
heterogeneous network (VHetNet) architecture appears to be inevitable due to
its high inherent agility. Moreover, employing an intelligent framework is
another crucial requirement for self-evolving networks to deal with real-time
network optimization problems. Hence, in this work, to provide a better insight
on network architecture design in support of self-evolving networks, we
highlight the merits of integrated VHetNet architecture while proposing an
intelligent framework for self-evolving integrated vertical heterogeneous
networks (SEI-VHetNets). The impact of the challenges associated with
SEI-VHetNet architecture, on network management is also studied considering a
generalized network model. Furthermore, the current literature on network
management of integrated VHetNets along with the recent advancements in
artificial intelligence (AI)/machine learning (ML) solutions are discussed.
Accordingly, the core challenges of integrating AI/ML in SEI-VHetNets are
identified. Finally, the potential future research directions for advancing the
autonomous and self-evolving capabilities of SEI-VHetNets are discussed.

    

### [[2110.06931] A neural simulation-based inference approach for characterizing the Galactic Center $γ$-ray excess](http://arxiv.org/abs/2110.06931)


  The nature of the Fermi gamma-ray Galactic Center Excess (GCE) has remained a
persistent mystery for over a decade. Although the excess is broadly compatible
with emission expected due to dark matter annihilation, an explanation in terms
of a population of unresolved astrophysical point sources e.g., millisecond
pulsars, remains viable. The effort to uncover the origin of the GCE is
hampered in particular by an incomplete understanding of diffuse emission of
Galactic origin. This can lead to spurious features that make it difficult to
robustly differentiate smooth emission, as expected for a dark matter origin,
from more "clumpy" emission expected for a population of relatively bright,
unresolved point sources. We use recent advancements in the field of
simulation-based inference, in particular density estimation techniques using
normalizing flows, in order to characterize the contribution of modeled
components, including unresolved point source populations, to the GCE. Compared
to traditional techniques based on the statistical distribution of photon
counts, our machine learning-based method is able to utilize more of the
information contained in a given model of the Galactic Center emission, and in
particular can perform posterior parameter estimation while accounting for
pixel-to-pixel spatial correlations in the gamma-ray map. This makes the method
demonstrably more resilient to certain forms of model misspecification. On
application to Fermi data, the method generically attributes a smaller fraction
of the GCE flux to unresolved point sources when compared to traditional
approaches. We nevertheless infer such a contribution to make up a
non-negligible fraction of the GCE across all analysis variations considered,
with at least $38^{+9}_{-19}\%$ of the excess attributed to unresolved points
sources in our baseline analysis.

    

### [[2110.06933] Style-based quantum generative adversarial networks for Monte Carlo events](http://arxiv.org/abs/2110.06933)


  We propose and assess an alternative quantum generator architecture in the
context of generative adversarial learning for Monte Carlo event generation,
used to simulate particle physics processes at the Large Hadron Collider (LHC).
We validate this methodology by implementing the quantum network on artificial
data generated from known underlying distributions. The network is then applied
to Monte Carlo-generated datasets of specific LHC scattering processes. The new
quantum generator architecture leads to an improvement in state-of-the-art
implementations while maintaining shallow-depth networks. Moreover, the quantum
generator successfully learns the underlying distribution functions even if
trained with small training sample sets; this is particularly interesting for
data augmentation applications. We deploy this novel methodology on two
different quantum hardware architectures, trapped-ion and superconducting
technologies, to test its hardware-independent viability.

    

### [[2110.06948] Challenges for Unsupervised Anomaly Detection in Particle Physics](http://arxiv.org/abs/2110.06948)


  Anomaly detection relies on designing a score to determine whether a
particular event is uncharacteristic of a given background distribution. One
way to define a score is to use autoencoders, which rely on the ability to
reconstruct certain types of data (background) but not others (signals). In
this paper, we study some challenges associated with variational autoencoders,
such as the dependence on hyperparameters and the metric used, in the context
of anomalous signal (top and $W$) jets in a QCD background. We find that the
hyperparameter choices strongly affect the network performance and that the
optimal parameters for one signal are non-optimal for another. In exploring the
networks, we uncover a connection between the latent space of a variational
autoencoder trained using mean-squared-error and the optimal transport
distances within the dataset. We then show that optimal transport distances to
representative events in the background dataset can be used directly for
anomaly detection, with performance comparable to the autoencoders. Whether
using autoencoders or optimal transport distances for anomaly detection, we
find that the choices that best represent the background are not necessarily
best for signal identification. These challenges with unsupervised anomaly
detection bolster the case for additional exploration of semi-supervised or
alternative approaches.

    

### [[2110.06961] Language Modelling via Learning to Rank](http://arxiv.org/abs/2110.06961)


  We consider language modelling (LM) as a multi-label structured prediction
task by re-framing training from solely predicting a single ground-truth word
to ranking a set of words which could continue a given context. To avoid
annotating top-$k$ ranks, we generate them using pre-trained LMs: GPT-2, BERT,
and Born-Again models. This leads to a rank-based form of knowledge
distillation (KD). We also develop a method using $N$-grams to create a
non-probabilistic teacher which generates the ranks without the need of a
pre-trained LM.
We confirm the hypotheses that we can treat LMing as a ranking task and that
we can do so without the use of a pre-trained LM. We show that rank-based KD
generally improves perplexity (PPL), often with statistical significance, when
compared to Kullback-Leibler-based KD. Surprisingly, given the simplicity of
the method, $N$-grams act as competitive teachers and achieve similar
performance as using either BERT or a Born-Again model teachers. GPT-2 always
acts as the best teacher, though, and using it and a Transformer-XL student on
Wiki-02, rank-based KD reduces a cross-entropy baseline from 65.27 to 55.94 and
against a KL-based KD of 56.70.

    

### [[2110.06972] Block Contextual MDPs for Continual Learning](http://arxiv.org/abs/2110.06972)


  In reinforcement learning (RL), when defining a Markov Decision Process
(MDP), the environment dynamics is implicitly assumed to be stationary. This
assumption of stationarity, while simplifying, can be unrealistic in many
scenarios. In the continual reinforcement learning scenario, the sequence of
tasks is another source of nonstationarity. In this work, we propose to examine
this continual reinforcement learning setting through the block contextual MDP
(BC-MDP) framework, which enables us to relax the assumption of stationarity.
This framework challenges RL algorithms to handle both nonstationarity and rich
observation settings and, by additionally leveraging smoothness properties,
enables us to study generalization bounds for this setting. Finally, we take
inspiration from adaptive control to propose a novel algorithm that addresses
the challenges introduced by this more realistic BC-MDP setting, allows for
zero-shot adaptation at evaluation time, and achieves strong performance on
several nonstationary environments.

    

### [[2110.06976] Rethinking the Representational Continuity: Towards Unsupervised Continual Learning](http://arxiv.org/abs/2110.06976)


  Continual learning (CL) aims to learn a sequence of tasks without forgetting
the previously acquired knowledge. However, recent advances in continual
learning are restricted to supervised continual learning (SCL) scenarios.
Consequently, they are not scalable to real-world applications where the data
distribution is often biased and unannotated. In this work, we focus on
unsupervised continual learning (UCL), where we learn the feature
representations on an unlabelled sequence of tasks and show that reliance on
annotated data is not necessary for continual learning. We conduct a systematic
study analyzing the learned feature representations and show that unsupervised
visual representations are surprisingly more robust to catastrophic forgetting,
consistently achieve better performance, and generalize better to
out-of-distribution tasks than SCL. Furthermore, we find that UCL achieves a
smoother loss landscape through qualitative analysis of the learned
representations and learns meaningful feature representations. Additionally, we
propose Lifelong Unsupervised Mixup (LUMP), a simple yet effective technique
that leverages the interpolation between the current task and previous tasks'
instances to alleviate catastrophic forgetting for unsupervised
representations.

    

### [[2110.06978] WAFFLE: Weighted Averaging for Personalized Federated Learning](http://arxiv.org/abs/2110.06978)


  In collaborative or federated learning, model personalization can be a very
effective strategy to deal with heterogeneous training data across clients. We
introduce WAFFLE (Weighted Averaging For Federated LEarning), a personalized
collaborative machine learning algorithm based on SCAFFOLD. SCAFFOLD uses
stochastic control variates to converge towards a model close to the globally
optimal model even in tasks where the distribution of data and labels across
clients is highly skewed. In contrast, WAFFLE uses the Euclidean distance
between clients' updates to weigh their individual contributions and thus
minimize the trained personalized model loss on the specific agent of interest.
Through a series of experiments, we compare our proposed new method to two
recent personalized federated learning methods, Weight Erosion and APFL, as
well as two global learning methods, federated averaging and SCAFFOLD. We
evaluate our method using two categories of non-identical client data
distributions (concept shift and label skew) on two benchmark image data sets,
MNIST and CIFAR10. Our experiments demonstrate the effectiveness of WAFFLE
compared with other methods, as it achieves or improves accuracy with faster
convergence.

    

### [[2110.06980] Output Space Entropy Search Framework for Multi-Objective Bayesian Optimization](http://arxiv.org/abs/2110.06980)


  We consider the problem of black-box multi-objective optimization (MOO) using
expensive function evaluations (also referred to as experiments), where the
goal is to approximate the true Pareto set of solutions by minimizing the total
resource cost of experiments. For example, in hardware design optimization, we
need to find the designs that trade-off performance, energy, and area overhead
using expensive computational simulations. The key challenge is to select the
sequence of experiments to uncover high-quality solutions using minimal
resources. In this paper, we propose a general framework for solving MOO
problems based on the principle of output space entropy (OSE) search: select
the experiment that maximizes the information gained per unit resource cost
about the true Pareto front. We appropriately instantiate the principle of OSE
search to derive efficient algorithms for the following four MOO problem
settings: 1) The most basic em single-fidelity setting, where experiments are
expensive and accurate; 2) Handling em black-box constraints} which cannot be
evaluated without performing experiments; 3) The discrete multi-fidelity
setting, where experiments can vary in the amount of resources consumed and
their evaluation accuracy; and 4) The em continuous-fidelity setting, where
continuous function approximations result in a huge space of experiments.
Experiments on diverse synthetic and real-world benchmarks show that our OSE
search based algorithms improve over state-of-the-art methods in terms of both
computational-efficiency and accuracy of MOO solutions.

    

### [[2110.06983] Bundle Networks: Fiber Bundles, Local Trivializations, and a Generative Approach to Exploring Many-to-one Maps](http://arxiv.org/abs/2110.06983)


  Many-to-one maps are ubiquitous in machine learning, from the image
recognition model that assigns a multitude of distinct images to the concept of
"cat" to the time series forecasting model which assigns a range of distinct
time-series to a single scalar regression value. While the primary use of such
models is naturally to associate correct output to each input, in many problems
it is also useful to be able to explore, understand, and sample from a model's
fibers, which are the set of input values $x$ such that $f(x) = y$, for fixed
$y$ in the output space. In this paper we show that popular generative
architectures are ill-suited to such tasks. Motivated by this we introduce a
novel generative architecture, a Bundle Network, based on the concept of a
fiber bundle from (differential) topology. BundleNets exploit the idea of a
local trivialization wherein a space can be locally decomposed into a product
space that cleanly encodes the many-to-one nature of the map. By enforcing this
decomposition in BundleNets and by utilizing state-of-the-art invertible
components, investigating a network's fibers becomes natural.

    

### [[2110.06986] ADMM-DAD net: a deep unfolding network for analysis compressed sensing](http://arxiv.org/abs/2110.06986)


  In this paper, we propose a new deep unfolding neural network based on the
ADMM algorithm for analysis Compressed Sensing. The proposed network jointly
learns a redundant analysis operator for sparsification and reconstructs the
signal of interest. We compare our proposed network with a state-of-the-art
unfolded ISTA decoder, that also learns an orthogonal sparsifier. Moreover, we
consider not only image, but also speech datasets as test examples.
Computational experiments demonstrate that our proposed network outperforms the
state-of-the-art deep unfolding networks, consistently for both real-world
image and speech datasets.

    

### [[2110.06990] Scaling Laws for the Few-Shot Adaptation of Pre-trained Image Classifiers](http://arxiv.org/abs/2110.06990)


  Empirical science of neural scaling laws is a rapidly growing area of
significant importance to the future of machine learning, particularly in the
light of recent breakthroughs achieved by large-scale pre-trained models such
as GPT-3, CLIP and DALL-e. Accurately predicting the neural network performance
with increasing resources such as data, compute and model size provides a more
comprehensive evaluation of different approaches across multiple scales, as
opposed to traditional point-wise comparisons of fixed-size models on
fixed-size benchmarks, and, most importantly, allows for focus on the
best-scaling, and thus most promising in the future, approaches. In this work,
we consider a challenging problem of few-shot learning in image classification,
especially when the target data distribution in the few-shot phase is different
from the source, training, data distribution, in a sense that it includes new
image classes not encountered during training. Our current main goal is to
investigate how the amount of pre-training data affects the few-shot
generalization performance of standard image classifiers. Our key observations
are that (1) such performance improvements are well-approximated by power laws
(linear log-log plots) as the training set size increases, (2) this applies to
both cases of target data coming from either the same or from a different
domain (i.e., new classes) as the training data, and (3) few-shot performance
on new classes converges at a faster rate than the standard classification
performance on previously seen classes. Our findings shed new light on the
relationship between scale and generalization.

    

### [[2110.06991] Scalable Graph Embedding LearningOn A Single GPU](http://arxiv.org/abs/2110.06991)


  Graph embedding techniques have attracted growing interest since they convert
the graph data into continuous and low-dimensional space. Effective graph
analytic provides users a deeper understanding of what is behind the data and
thus can benefit a variety of machine learning tasks. With the current scale of
real-world applications, most graph analytic methods suffer high computation
and space costs. These methods and systems can process a network with thousands
to a few million nodes. However, scaling to large-scale networks remains a
challenge. The complexity of training graph embedding system requires the use
of existing accelerators such as GPU. In this paper, we introduce a hybrid
CPU-GPU framework that addresses the challenges of learning embedding of
large-scale graphs. The performance of our method is compared qualitatively and
quantitatively with the existing embedding systems on common benchmarks. We
also show that our system can scale training to datasets with an order of
magnitude greater than a single machine's total memory capacity. The
effectiveness of the learned embedding is evaluated within multiple downstream
applications. The experimental results indicate the effectiveness of the
learned embedding in terms of performance and accuracy.

    

### [[2110.06996] A Novel Clustering-Based Algorithm for Continuous and Non-invasive Cuff-Less Blood Pressure Estimation](http://arxiv.org/abs/2110.06996)


  Continuous blood pressure (BP) measurements can reflect a bodys response to
diseases and serve as a predictor of cardiovascular and other health
conditions. While current cuff-based BP measurement methods are incapable of
providing continuous BP readings, invasive BP monitoring methods also tend to
cause patient dissatisfaction and can potentially cause infection. In this
research, we developed a method for estimating blood pressure based on the
features extracted from Electrocardiogram (ECG) and Photoplethysmogram (PPG)
signals and the Arterial Blood Pressure (ABP) data. The vector of features
extracted from the preprocessed ECG and PPG signals is used in this approach,
which include Pulse Transit Time (PTT), PPG Intensity Ratio (PIR), and Heart
Rate (HR), as the input of a clustering algorithm and then developing separate
regression models like Random Forest Regression, Gradient Boosting Regression,
and Multilayer Perceptron Regression algorithms for each resulting cluster. We
evaluated and compared the findings to create the model with the highest
accuracy by applying the clustering approach and identifying the optimal number
of clusters, and eventually the acceptable prediction model. The paper compares
the results obtained with and without this clustering. The results show that
the proposed clustering approach helps obtain more accurate estimates of
Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP). Given the
inconsistency, high dispersion, and multitude of trends in the datasets for
different features, using the clustering approach improved the estimation
accuracy by 50-60%.

    

### [[2110.06999] Study of positional encoding approaches for Audio Spectrogram Transformers](http://arxiv.org/abs/2110.06999)


  Transformers have revolutionized the world of deep learning, specially in the
field of natural language processing. Recently, the Audio Spectrogram
Transformer (AST) was proposed for audio classification, leading to state of
the art results in several datasets. However, in order for ASTs to outperform
CNNs, pretraining with ImageNet is needed. In this paper, we study one
component of the AST, the positional encoding, and propose several variants to
improve the performance of ASTs trained from scratch, without ImageNet
pretraining. Our best model, which incorporates conditional positional
encodings, significantly improves performance on Audioset and ESC-50 compared
to the original AST.

    

### [[2110.07002] Bag-of-Vectors Autoencoders for Unsupervised Conditional Text Generation](http://arxiv.org/abs/2110.07002)


  Text autoencoders are often used for unsupervised conditional text generation
by applying mappings in the latent space to change attributes to the desired
values. Recently, Mai et al. (2020) proposed Emb2Emb, a method to learn these
mappings in the embedding space of an autoencoder. However, their method is
restricted to autoencoders with a single-vector embedding, which limits how
much information can be retained. We address this issue by extending their
method to Bag-of-Vectors Autoencoders (BoV-AEs), which encode the text into a
variable-size bag of vectors that grows with the size of the text, as in
attention-based models. This allows to encode and reconstruct much longer texts
than standard autoencoders. Analogous to conventional autoencoders, we propose
regularization techniques that facilitate learning meaningful operations in the
latent space. Finally, we adapt for a training scheme that learns to map an
input bag to an output bag, including a novel loss function and neural
architecture. Our experimental evaluations on unsupervised sentiment transfer
and sentence summarization show that our method performs substantially better
than a standard autoencoder.

    

### [[2110.07003] Sustainability Through Cognition Aware Safety Systems -- Next Level Human-Machine-Interaction](http://arxiv.org/abs/2110.07003)


  Industrial Safety deals with the physical integrity of humans, machines and
the environment when they interact during production scenarios. Industrial
Safety is subject to a rigorous certification process that leads to inflexible
settings, in which all changes are forbidden. With the progressing introduction
of smart robotics and smart machinery to the factory floor, combined with an
increasing shortage of skilled workers, it becomes imperative that safety
scenarios incorporate a flexible handling of the boundary between humans,
machines and the environment. In order to increase the well-being of workers,
reduce accidents, and compensate for different skill sets, the configuration of
machines and the factory floor should be dynamically adapted, while still
enforcing functional safety requirements. The contribution of this paper is as
follows: (1) We present a set of three scenarios, and discuss how industrial
safety mechanisms could be augmented through dynamic changes to the work
environment in order to decrease potential accidents, and thus increase
productivity. (2) We introduce the concept of a Cognition Aware Safety System
(CASS) and its architecture. The idea behind CASS is to integrate AI based
reasoning about human load, stress, and attention with AI based selection of
actions to avoid the triggering of safety stops. (3) And finally, we will
describe the required performance measurement dimensions for a quantitative
performance measurement model to enable a comprehensive (triple bottom line)
impact assessment of CASS. Additionally we introduce a detailed guideline for
expert interviews to explore the feasibility of the approach for given
scenarios.

    

### [[2110.07004] ES-Based Jacobian Enables Faster Bilevel Optimization](http://arxiv.org/abs/2110.07004)


  Bilevel optimization (BO) has arisen as a powerful tool for solving many
modern machine learning problems. However, due to the nested structure of BO,
existing gradient-based methods require second-order derivative approximations
via Jacobian- or/and Hessian-vector computations, which can be very costly in
practice, especially with large neural network models. In this work, we propose
a novel BO algorithm, which adopts Evolution Strategies (ES) based method to
approximate the response Jacobian matrix in the hypergradient of BO, and hence
fully eliminates all second-order computations. We call our algorithm as ESJ
(which stands for the ES-based Jacobian method) and further extend it to the
stochastic setting as ESJ-S. Theoretically, we characterize the convergence
guarantee and computational complexity for our algorithms. Experimentally, we
demonstrate the superiority of our proposed algorithms compared to the state of
the art methods on various bilevel problems. Particularly, in our experiment in
the few-shot meta-learning problem, we meta-learn the twelve millions
parameters of a ResNet-12 network over the miniImageNet dataset, which
evidently demonstrates the scalability of our ES-based bilevel approach and its
feasibility in the large-scale setting.

    

### [[2110.07007] Out-of-Distribution Robustness in Deep Learning Compression](http://arxiv.org/abs/2110.07007)


  In recent years, deep neural network (DNN) compression systems have proved to
be highly effective for designing source codes for many natural sources.
However, like many other machine learning systems, these compressors suffer
from vulnerabilities to distribution shifts as well as out-of-distribution
(OOD) data, which reduces their real-world applications. In this paper, we
initiate the study of OOD robust compression. Considering robustness to two
types of ambiguity sets (Wasserstein balls and group shifts), we propose
algorithmic and architectural frameworks built on two principled methods: one
that trains DNN compressors using distributionally-robust optimization (DRO),
and the other which uses a structured latent code. Our results demonstrate that
both methods enforce robustness compared to a standard DNN compressor, and that
using a structured code can be superior to the DRO compressor. We observe
tradeoffs between robustness and distortion and corroborate these findings
theoretically for a specific class of sources.

    

### [[2110.07009] Covert Message Passing over Public Internet Platforms Using Model-Based Format-Transforming Encryption](http://arxiv.org/abs/2110.07009)


  We introduce a new type of format-transforming encryption where the format of
ciphertexts is implicitly encoded within a machine-learned generative model.
Around this primitive, we build a system for covert messaging over large,
public internet platforms (e.g., Twitter). Loosely, our system composes an
authenticated encryption scheme, with a method for encoding random ciphertext
bits into samples from the generative model's family of seed-indexed
token-distributions. By fixing a deployment scenario, we are forced to consider
system-level and algorithmic solutions to real challenges -- such as
receiver-side parsing ambiguities, and the low information-carrying capacity of
actual token-distributions -- that were elided in prior work. We use GPT-2 as
our generative model so that our system cryptographically transforms plaintext
bitstrings into natural-language covertexts suitable for posting to public
platforms. We consider adversaries with full view of the internet platform's
content, whose goal is to surface posts that are using our system for covert
messaging. We carry out a suite of experiments to provide heuristic evidence of
security and to explore tradeoffs between operational efficiency and
detectability.

    

### [[2110.07020] Top 3 in FG 2021 Families In the Wild Kinship Verification Challenge](http://arxiv.org/abs/2110.07020)


  Kinship verification is the task of determining whether a parent-child,
sibling, or grandparent-grandchild relationship exists between two people and
is important in social media applications, forensic investigations, finding
missing children, and reuniting families. We demonstrate high quality kinship
verification by participating in the FG 2021 Recognizing Families in the Wild
challenge which provides the largest publicly available dataset in the field.
Our approach is among the top 3 winning entries in the competition. We ensemble
models written by both human experts and OpenAI Codex. We make our models and
code publicly available.

    

### [[2110.07028] AI Total: Analyzing Security ML Models with Imperfect Data in Production](http://arxiv.org/abs/2110.07028)


  Development of new machine learning models is typically done on manually
curated data sets, making them unsuitable for evaluating the models'
performance during operations, where the evaluation needs to be performed
automatically on incoming streams of new data. Unfortunately, pure reliance on
a fully automatic pipeline for monitoring model performance makes it difficult
to understand if any observed performance issues are due to model performance,
pipeline issues, emerging data distribution biases, or some combination of the
above. With this in mind, we developed a web-based visualization system that
allows the users to quickly gather headline performance numbers while
maintaining confidence that the underlying data pipeline is functioning
properly. It also enables the users to immediately observe the root cause of an
issue when something goes wrong. We introduce a novel way to analyze
performance under data issues using a data coverage equalizer. We describe the
various modifications and additional plots, filters, and drill-downs that we
added on top of the standard evaluation metrics typically tracked in machine
learning (ML) applications, and walk through some real world examples that
proved valuable for introspecting our models.

    

### [[2110.07029] Adaptive Elastic Training for Sparse Deep Learning on Heterogeneous Multi-GPU Servers](http://arxiv.org/abs/2110.07029)


  Motivated by extreme multi-label classification applications, we consider
training deep learning models over sparse data in multi-GPU servers. The
variance in the number of non-zero features across training batches and the
intrinsic GPU heterogeneity combine to limit accuracy and increase the time to
convergence. We address these challenges with Adaptive SGD, an adaptive elastic
model averaging stochastic gradient descent algorithm for heterogeneous
multi-GPUs that is characterized by dynamic scheduling, adaptive batch size
scaling, and normalized model merging. Instead of statically partitioning
batches to GPUs, batches are routed based on the relative processing speed.
Batch size scaling assigns larger batches to the faster GPUs and smaller
batches to the slower ones, with the goal to arrive at a steady state in which
all the GPUs perform the same number of model updates. Normalized model merging
computes optimal weights for every GPU based on the assigned batches such that
the combined model achieves better accuracy. We show experimentally that
Adaptive SGD outperforms four state-of-the-art solutions in time-to-accuracy
and is scalable with the number of GPUs.

    

### [[2110.07034] How Does Momentum Benefit Deep Neural Networks Architecture Design? A Few Case Studies](http://arxiv.org/abs/2110.07034)


  We present and review an algorithmic and theoretical framework for improving
neural network architecture design via momentum. As case studies, we consider
how momentum can improve the architecture design for recurrent neural networks
(RNNs), neural ordinary differential equations (ODEs), and transformers. We
show that integrating momentum into neural network architectures has several
remarkable theoretical and empirical benefits, including 1) integrating
momentum into RNNs and neural ODEs can overcome the vanishing gradient issues
in training RNNs and neural ODEs, resulting in effective learning long-term
dependencies. 2) momentum in neural ODEs can reduce the stiffness of the ODE
dynamics, which significantly enhances the computational efficiency in training
and testing. 3) momentum can improve the efficiency and accuracy of
transformers.

    

### [[2110.07035] Bond Default Prediction with Text Embeddings, Undersampling and Deep Learning](http://arxiv.org/abs/2110.07035)


  The special and important problems of default prediction for municipal bonds
are addressed using a combination of text embeddings from a pre-trained
transformer network, a fully connected neural network, and synthetic
oversampling. The combination of these techniques provides significant
improvement in performance over human estimates, linear models, and boosted
ensemble models, on data with extreme imbalance. Less than 0.2% of municipal
bonds default, but our technique predicts 9 out of 10 defaults at the time of
issue, without using bond ratings, at a cost of false positives on less than
0.1% non-defaulting bonds. The results hold the promise of reducing the cost of
capital for local public goods, which are vital for society, and bring
techniques previously used in personal credit and public equities (or national
fixed income), as well as the current generation of embedding techniques, to
sub-sovereign credit decisions.

    

### [[2110.07040] Data Incubation -- Synthesizing Missing Data for Handwriting Recognition](http://arxiv.org/abs/2110.07040)


  In this paper, we demonstrate how a generative model can be used to build a
better recognizer through the control of content and style. We are building an
online handwriting recognizer from a modest amount of training samples. By
training our controllable handwriting synthesizer on the same data, we can
synthesize handwriting with previously underrepresented content (e.g., URLs and
email addresses) and style (e.g., cursive and slanted). Moreover, we propose a
framework to analyze a recognizer that is trained with a mixture of real and
synthetic training data. We use the framework to optimize data synthesis and
demonstrate significant improvement on handwriting recognition over a model
trained on real data only. Overall, we achieve a 66% reduction in Character
Error Rate.

    

### [[2110.07043] Why Out-of-distribution Detection in CNNs Does Not Like Mahalanobis -- and What to Use Instead](http://arxiv.org/abs/2110.07043)


  Convolutional neural networks applied for real-world classification tasks
need to recognize inputs that are far or out-of-distribution (OoD) with respect
to the known or training data. To achieve this, many methods estimate
class-conditional posterior probabilities and use confidence scores obtained
from the posterior distributions. Recent works propose to use multivariate
Gaussian distributions as models of posterior distributions at different layers
of the CNN (i.e., for low- and upper-level features), which leads to the
confidence scores based on the Mahalanobis distance. However, this procedure
involves estimating probability density in high dimensional data using the
insufficient number of observations (e.g. the dimensionality of features at the
last two layers in the ResNet-101 model are 2048 and 1024, with ca. 1000
observations per class used to estimate density). In this work, we want to
address this problem. We show that in many OoD studies in high-dimensional
data, LOF-based (Local Outlierness-Factor) methods outperform the parametric,
Mahalanobis distance-based methods. This motivates us to propose the
nonparametric, LOF-based method of generating the confidence scores for CNNs.
We performed several feasibility studies involving ResNet-101 and
EffcientNet-B3, based on CIFAR-10 and ImageNet (as known data), and CIFAR-100,
SVHN, ImageNet2010, Places365, or ImageNet-O (as outliers). We demonstrated
that nonparametric LOF-based confidence estimation can improve current
Mahalanobis-based SOTA or obtain similar performance in a simpler way.

    

### [[2110.07045] Investigating Health-Aware Smart-Nudging with Machine Learning to Help People Pursue Healthier Eating-Habits](http://arxiv.org/abs/2110.07045)


  Food-choices and eating-habits directly contribute to our long-term health.
This makes the food recommender system a potential tool to address the global
crisis of obesity and malnutrition. Over the past decade,
artificial-intelligence and medical researchers became more invested in
researching tools that can guide and help people make healthy and thoughtful
decisions around food and diet. In many typical (Recommender System) RS
domains, smart nudges have been proven effective in shaping users' consumption
patterns. In recent years, knowledgeable nudging and incentifying choices
started getting attention in the food domain as well. To develop smart nudging
for promoting healthier food choices, we combined Machine Learning and RS
technology with food-healthiness guidelines from recognized health
organizations, such as the World Health Organization, Food Standards Agency,
and the National Health Service United Kingdom. In this paper, we discuss our
research on, persuasive visualization for making users aware of the healthiness
of the recommended recipes. Here, we propose three novel nudging technology,
the WHO-BubbleSlider, the FSA-ColorCoading, and the DRCI-MLCP, that encourage
users to choose healthier recipes. We also propose a Topic Modeling based
portion-size recommendation algorithm. To evaluate our proposed smart-nudges,
we conducted an online user study with 96 participants and 92250 recipes.
Results showed that, during the food decision-making process, appropriate
healthiness cues make users more likely to click, browse, and choose healthier
recipes over less healthy ones.

    

### [[2110.07046] Deep Metric Learning with Locality Sensitive Angular Loss for Self-Correcting Source Separation of Neural Spiking Signals](http://arxiv.org/abs/2110.07046)


  Neurophysiological time series, such as electromyographic signal and
intracortical recordings, are typically composed of many individual spiking
sources, the recovery of which can give fundamental insights into the
biological system of interest or provide neural information for man-machine
interfaces. For this reason, source separation algorithms have become an
increasingly important tool in neuroscience and neuroengineering. However, in
noisy or highly multivariate recordings these decomposition techniques often
make a large number of errors, which degrades human-machine interfacing
applications and often requires costly post-hoc manual cleaning of the output
label set of spike timestamps. To address both the need for automated post-hoc
cleaning and robust separation filters we propose a methodology based on deep
metric learning, using a novel loss function which maintains intra-class
variance, creating a rich embedding space suitable for both label cleaning and
the discovery of new activations. We then validate this method with an
artificially corrupted label set based on source-separated high-density surface
electromyography recordings, recovering the original timestamps even in extreme
degrees of feature and class-dependent label noise. This approach enables a
neural network to learn to accurately decode neurophysiological time series
using any imperfect method of labelling the signal.

    

### [[2110.07053] Robust MIMO Detection using Hypernetworks with Learned Regularizers](http://arxiv.org/abs/2110.07053)


  Optimal symbol detection in multiple-input multiple-output (MIMO) systems is
known to be an NP-hard problem. Recently, there has been a growing interest to
get reasonably close to the optimal solution using neural networks while
keeping the computational complexity in check. However, existing work based on
deep learning shows that it is difficult to design a generic network that works
well for a variety of channels. In this work, we propose a method that tries to
strike a balance between symbol error rate (SER) performance and generality of
channels. Our method is based on hypernetworks that generate the parameters of
a neural network-based detector that works well on a specific channel. We
propose a general framework by regularizing the training of the hypernetwork
with some pre-trained instances of the channel-specific method. Through
numerical experiments, we show that our proposed method yields high performance
for a set of prespecified channel realizations while generalizing well to all
channels drawn from a specific distribution.

    

### [[2110.07059] Subspace Regularizers for Few-Shot Class Incremental Learning](http://arxiv.org/abs/2110.07059)


  Few-shot class incremental learning -- the problem of updating a trained
classifier to discriminate among an expanded set of classes with limited
labeled data -- is a key challenge for machine learning systems deployed in
non-stationary environments. Existing approaches to the problem rely on complex
model architectures and training procedures that are difficult to tune and
re-use. In this paper, we present an extremely simple approach that enables the
use of ordinary logistic regression classifiers for few-shot incremental
learning. The key to this approach is a new family of subspace regularization
schemes that encourage weight vectors for new classes to lie close to the
subspace spanned by the weights of existing classes. When combined with
pretrained convolutional feature extractors, logistic regression models trained
with subspace regularization outperform specialized, state-of-the-art
approaches to few-shot incremental image classification by up to 22% on the
miniImageNet dataset. Because of its simplicity, subspace regularization can be
straightforwardly extended to incorporate additional background information
about the new classes (including class names and descriptions specified in
natural language); these further improve accuracy by up to 2%. Our results show
that simple geometric regularization of class representations offers an
effective tool for continual learning.

    

### [[2110.07064] Variance Minimization in the Wasserstein Space for Invariant Causal Prediction](http://arxiv.org/abs/2110.07064)


  Selecting powerful predictors for an outcome is a cornerstone task for
machine learning. However, some types of questions can only be answered by
identifying the predictors that causally affect the outcome. A recent approach
to this causal inference problem leverages the invariance property of a causal
mechanism across differing experimental environments (Peters et al., 2016;
Heinze-Deml et al., 2018). This method, invariant causal prediction (ICP), has
a substantial computational defect -- the runtime scales exponentially with the
number of possible causal variables. In this work, we show that the approach
taken in ICP may be reformulated as a series of nonparametric tests that scales
linearly in the number of predictors. Each of these tests relies on the
minimization of a novel loss function -- the Wasserstein variance -- that is
derived from tools in optimal transport theory and is used to quantify
distributional variability across environments. We prove under mild assumptions
that our method is able to recover the set of identifiable direct causes, and
we demonstrate in our experiments that it is competitive with other benchmark
causal discovery algorithms.

    

### [[2110.07069] CloudPred: Predicting Patient Phenotypes From Single-cell RNA-seq](http://arxiv.org/abs/2110.07069)


  Single-cell RNA sequencing (scRNA-seq) has the potential to provide powerful,
high-resolution signatures to inform disease prognosis and precision medicine.
This paper takes an important first step towards this goal by developing an
interpretable machine learning algorithm, CloudPred, to predict individuals'
disease phenotypes from their scRNA-seq data. Predicting phenotype from
scRNA-seq is challenging for standard machine learning methods -- the number of
cells measured can vary by orders of magnitude across individuals and the cell
populations are also highly heterogeneous. Typical analysis creates pseudo-bulk
samples which are biased toward prior annotations and also lose the single cell
resolution. CloudPred addresses these challenges via a novel end-to-end
differentiable learning algorithm which is coupled with a biologically informed
mixture of cell types model. CloudPred automatically infers the cell
subpopulation that are salient for the phenotype without prior annotations. We
developed a systematic simulation platform to evaluate the performance of
CloudPred and several alternative methods we propose, and find that CloudPred
outperforms the alternative methods across several settings. We further
validated CloudPred on a real scRNA-seq dataset of 142 lupus patients and
controls. CloudPred achieves AUROC of 0.98 while identifying a specific
subpopulation of CD4 T cells whose presence is highly indicative of lupus.
CloudPred is a powerful new framework to predict clinical phenotypes from
scRNA-seq data and to identify relevant cells.

    

### [[2110.07097] A Comprehensive Study on Torchvision Pre-trained Models for Fine-grained Inter-species Classification](http://arxiv.org/abs/2110.07097)


  This study aims to explore different pre-trained models offered in the
Torchvision package which is available in the PyTorch library. And investigate
their effectiveness on fine-grained images classification. Transfer Learning is
an effective method of achieving extremely good performance with insufficient
training data. In many real-world situations, people cannot collect sufficient
data required to train a deep neural network model efficiently. Transfer
Learning models are pre-trained on a large data set, and can bring a good
performance on smaller datasets with significantly lower training time.
Torchvision package offers us many models to apply the Transfer Learning on
smaller datasets. Therefore, researchers may need a guideline for the selection
of a good model. We investigate Torchvision pre-trained models on four
different data sets: 10 Monkey Species, 225 Bird Species, Fruits 360, and
Oxford 102 Flowers. These data sets have images of different resolutions, class
numbers, and different achievable accuracies. We also apply their usual
fully-connected layer and the Spinal fully-connected layer to investigate the
effectiveness of SpinalNet. The Spinal fully-connected layer brings better
performance in most situations. We apply the same augmentation for different
models for the same data set for a fair comparison. This paper may help future
Computer Vision researchers in choosing a proper Transfer Learning model.

    

### [[2110.07098] Escaping Saddle Points in Nonconvex Minimax Optimization via Cubic-Regularized Gradient Descent-Ascent](http://arxiv.org/abs/2110.07098)


  The gradient descent-ascent (GDA) algorithm has been widely applied to solve
nonconvex minimax optimization problems. However, the existing GDA-type
algorithms can only find first-order stationary points of the envelope function
of nonconvex minimax optimization problems, which does not rule out the
possibility to get stuck at suboptimal saddle points. In this paper, we develop
Cubic-GDA -- the first GDA-type algorithm for escaping strict saddle points in
nonconvex-strongly-concave minimax optimization. Specifically, the algorithm
uses gradient ascent to estimate the second-order information of the minimax
objective function, and it leverages the cubic regularization technique to
efficiently escape the strict saddle points. Under standard smoothness
assumptions on the objective function, we show that Cubic-GDA admits an
intrinsic potential function whose value monotonically decreases in the minimax
optimization process. Such a property leads to a desired global convergence of
Cubic-GDA to a second-order stationary point at a sublinear rate. Moreover, we
analyze the convergence rate of Cubic-GDA in the full spectrum of a gradient
dominant-type nonconvex geometry. Our result shows that Cubic-GDA achieves an
orderwise faster convergence rate than the standard GDA for a wide spectrum of
gradient dominant geometry. Our study bridges minimax optimization with
second-order optimization and may inspire new developments along this
direction.

    

### [[2110.07112] On the Sample Complexity of Decentralized Linear Quadratic Regulator with Partially Nested Information Structure](http://arxiv.org/abs/2110.07112)


  We study the problem of control policy design for decentralized
state-feedback linear quadratic control with a partially nested information
structure, when the system model is unknown. We propose a model-based learning
solution, which consists of two steps. First, we estimate the unknown system
model from a single system trajectory of finite length, using least squares
estimation. Next, based on the estimated system model, we design a control
policy that satisfies the desired information structure. We show that the
suboptimality gap between our control policy and the optimal decentralized
control policy (designed using accurate knowledge of the system model) scales
linearly with the estimation error of the system model. Using this result, we
provide an end-to-end sample complexity result for learning decentralized
controllers for a linear quadratic control problem with a partially nested
information structure.

    

### [[2110.07120] Brittle interpretations: The Vulnerability of TCAV and Other Concept-based Explainability Tools to Adversarial Attack](http://arxiv.org/abs/2110.07120)


  Methods for model explainability have become increasingly critical for
testing the fairness and soundness of deep learning. A number of explainability
techniques have been developed which use a set of examples to represent a
human-interpretable concept in a model's activations. In this work we show that
these explainability methods can suffer the same vulnerability to adversarial
attacks as the models they are meant to analyze. We demonstrate this phenomenon
on two well-known concept-based approaches to the explainability of deep
learning models: TCAV and faceted feature visualization. We show that by
carefully perturbing the examples of the concept that is being investigated, we
can radically change the output of the interpretability method, e.g. showing
that stripes are not an important factor in identifying images of a zebra. Our
work highlights the fact that in safety-critical applications, there is need
for security around not only the machine learning pipeline but also the model
interpretation process.

    

### [[2110.07121] Secure Precoding in MIMO-NOMA: A Deep Learning Approach](http://arxiv.org/abs/2110.07121)


  A novel signaling design for secure transmission over two-user multiple-input
multiple-output non-orthogonal multiple access channel using deep neural
networks (DNNs) is proposed. The goal of the DNN is to form the covariance
matrix of users' signals such that the message of each user is transmitted
reliably while being confidential from its counterpart. The proposed DNN
linearly precodes each user's signal before superimposing them and achieves
near-optimal performance with significantly lower run time. Simulation results
show that the proposed models reach about 98% of the secrecy capacity rates.
The spectral efficiency of the DNN precoder is much higher than that of
existing analytical linear precoders--e.g., generalized singular value
decomposition--and its on-the-fly complexity is several times less than the
existing iterative methods.

    

### [[2110.07122] Deconfounded Causal Collaborative Filtering](http://arxiv.org/abs/2110.07122)


  Recommender systems may be confounded by various types of confounding factors
(also called confounders) that may lead to inaccurate recommendations and
sacrificed recommendation performance. Current approaches to solving the
problem usually design each specific model for each specific confounder.
However, real-world systems may include a huge number of confounders and thus
designing each specific model for each specific confounder is unrealistic. More
importantly, except for those "explicit confounders" that researchers can
manually identify and process such as item's position in the ranking list,
there are also many "latent confounders" that are beyond the imagination of
researchers. For example, users' rating on a song may depend on their current
mood or the current weather, and users' preference on ice creams may depend on
the air temperature. Such latent confounders may be unobservable in the
recorded training data. To solve the problem, we propose a deconfounded causal
collaborative filtering model. We first frame user behaviors with unobserved
confounders into a causal graph, and then we design a front-door adjustment
model carefully fused with machine learning to deconfound the influence of
unobserved confounders. The proposed model is able to handle both global
confounders and personalized confounders. Experiments on real-world e-commerce
datasets show that our method is able to deconfound unobserved confounders to
achieve better recommendation performance.

    

### [[2110.07130] Region Semantically Aligned Network for Zero-Shot Learning](http://arxiv.org/abs/2110.07130)


  Zero-shot learning (ZSL) aims to recognize unseen classes based on the
knowledge of seen classes. Previous methods focused on learning direct
embeddings from global features to the semantic space in hope of knowledge
transfer from seen classes to unseen classes. However, an unseen class shares
local visual features with a set of seen classes and leveraging global visual
features makes the knowledge transfer ineffective. To tackle this problem, we
propose a Region Semantically Aligned Network (RSAN), which maps local features
of unseen classes to their semantic attributes. Instead of using global
features which are obtained by an average pooling layer after an image encoder,
we directly utilize the output of the image encoder which maintains local
information of the image. Concretely, we obtain each attribute from a specific
region of the output and exploit these attributes for recognition. As a result,
the knowledge of seen classes can be successfully transferred to unseen classes
in a region-bases manner. In addition, we regularize the image encoder through
attribute regression with a semantic knowledge to extract robust and
attribute-related visual features. Experiments on several standard ZSL datasets
reveal the benefit of the proposed RSAN method, outperforming state-of-the-art
methods.

    

### [[2110.07141] SoGCN: Second-Order Graph Convolutional Networks](http://arxiv.org/abs/2110.07141)


  Graph Convolutional Networks (GCN) with multi-hop aggregation is more
expressive than one-hop GCN but suffers from higher model complexity. Finding
the shortest aggregation range that achieves comparable expressiveness and
minimizes this side effect remains an open question. We answer this question by
showing that multi-layer second-order graph convolution (SoGC) is sufficient to
attain the ability of expressing polynomial spectral filters with arbitrary
coefficients. Compared to models with one-hop aggregation, multi-hop
propagation, and jump connections, SoGC possesses filter representational
completeness while being lightweight, efficient, and easy to implement.
Thereby, we suggest that SoGC is a simple design capable of forming the basic
building block of GCNs, playing the same role as $3 \times 3$ kernels in CNNs.
We build our Second-Order Graph Convolutional Networks (SoGCN) with SoGC and
design a synthetic dataset to verify its filter fitting capability to validate
these points. For real-world tasks, we present the state-of-the-art performance
of SoGCN on the benchmark of node classification, graph classification, and
graph regression datasets.

    

### [[2110.07152] DeepSSM: A Blueprint for Image-to-Shape Deep Learning Models](http://arxiv.org/abs/2110.07152)


  Statistical shape modeling (SSM) characterizes anatomical variations in a
population of shapes generated from medical images. SSM requires consistent
shape representation across samples in shape cohort. Establishing this
representation entails a processing pipeline that includes anatomy
segmentation, re-sampling, registration, and non-linear optimization. These
shape representations are then used to extract low-dimensional shape
descriptors that facilitate subsequent analyses in different applications.
However, the current process of obtaining these shape descriptors from imaging
data relies on human and computational resources, requiring domain expertise
for segmenting anatomies of interest. Moreover, this same taxing pipeline needs
to be repeated to infer shape descriptors for new image data using a
pre-trained/existing shape model. Here, we propose DeepSSM, a deep
learning-based framework for learning the functional mapping from images to
low-dimensional shape descriptors and their associated shape representations,
thereby inferring statistical representation of anatomy directly from 3D
images. Once trained using an existing shape model, DeepSSM circumvents the
heavy and manual pre-processing and segmentation and significantly improves the
computational time, making it a viable solution for fully end-to-end SSM
applications. In addition, we introduce a model-based data-augmentation
strategy to address data scarcity. Finally, this paper presents and analyzes
two different architectural variants of DeepSSM with different loss functions
using three medical datasets and their downstream clinical application.
Experiments showcase that DeepSSM performs comparably or better to the
state-of-the-art SSM both quantitatively and on application-driven downstream
tasks. Therefore, DeepSSM aims to provide a comprehensive blueprint for deep
learning-based image-to-shape models.

    

### [[2110.07161] Neural Attention-Aware Hierarchical Topic Model](http://arxiv.org/abs/2110.07161)


  Neural topic models (NTMs) apply deep neural networks to topic modelling.
Despite their success, NTMs generally ignore two important aspects: (1) only
document-level word count information is utilized for the training, while more
fine-grained sentence-level information is ignored, and (2) external semantic
knowledge regarding documents, sentences and words are not exploited for the
training. To address these issues, we propose a variational autoencoder (VAE)
NTM model that jointly reconstructs the sentence and document word counts using
combinations of bag-of-words (BoW) topical embeddings and pre-trained semantic
embeddings. The pre-trained embeddings are first transformed into a common
latent topical space to align their semantics with the BoW embeddings. Our
model also features hierarchical KL divergence to leverage embeddings of each
document to regularize those of their sentences, thereby paying more attention
to semantically relevant sentences. Both quantitative and qualitative
experiments have shown the efficacy of our model in 1) lowering the
reconstruction errors at both the sentence and document levels, and 2)
discovering more coherent topics from real-world datasets.

    

### [[2110.07174] Context-gloss Augmentation for Improving Word Sense Disambiguation](http://arxiv.org/abs/2110.07174)


  The goal of Word Sense Disambiguation (WSD) is to identify the sense of a
polysemous word in a specific context. Deep-learning techniques using BERT have
achieved very promising results in the field and different methods have been
proposed to integrate structured knowledge to enhance performance. At the same
time, an increasing number of data augmentation techniques have been proven to
be useful for NLP tasks. Building upon previous works leveraging BERT and
WordNet knowledge, we explore different data augmentation techniques on
context-gloss pairs to improve the performance of WSD. In our experiment, we
show that both sentence-level and word-level augmentation methods are effective
strategies for WSD. Also, we find out that performance can be improved by
adding hypernyms' glosses obtained from a lexical knowledge base. We compare
and analyze different context-gloss augmentation techniques, and the results
show that applying back translation on gloss performs the best.

    

### [[2110.07182] Adversarial examples by perturbing high-level features in intermediate decoder layers](http://arxiv.org/abs/2110.07182)


  We propose a novel method for creating adversarial examples. Instead of
perturbing pixels, we use an encoder-decoder representation of the input image
and perturb intermediate layers in the decoder. This changes the high-level
features provided by the generative model. Therefore, our perturbation
possesses semantic meaning, such as a longer beak or green tints. We formulate
this task as an optimization problem by minimizing the Wasserstein distance
between the adversarial and initial images under a misclassification
constraint. We employ the projected gradient method with a simple inexact
projection. Due to the projection, all iterations are feasible, and our method
always generates adversarial images. We perform numerical experiments on the
MNIST and ImageNet datasets in both targeted and untargeted settings. We
demonstrate that our adversarial images are much less vulnerable to
steganographic defence techniques than pixel-based attacks. Moreover, we show
that our method modifies key features such as edges and that defence techniques
based on adversarial training are vulnerable to our attacks.

    

### [[2110.07185] VLBInet: Radio Interferometry Data Classification for EHT with Neural Networks](http://arxiv.org/abs/2110.07185)


  The Event Horizon Telescope (EHT) recently released the first horizon-scale
images of the black hole in M87. Combined with other astronomical data, these
images constrain the mass and spin of the hole as well as the accretion rate
and magnetic flux trapped on the hole. An important question for the EHT is how
well key parameters, such as trapped magnetic flux and the associated disk
models, can be extracted from present and future EHT VLBI data products. The
process of modeling visibilities and analyzing them is complicated by the fact
that the data are sparsely sampled in the Fourier domain while most of the
theory/simulation is constructed in the image domain. Here we propose a
data-driven approach to analyze complex visibilities and closure quantities for
radio interferometric data with neural networks. Using mock interferometric
data, we show that our neural networks are able to infer the accretion state as
either high magnetic flux (MAD) or low magnetic flux (SANE), suggesting that it
is possible to perform parameter extraction directly in the visibility domain
without image reconstruction. We have applied VLBInet to real M87 EHT data
taken on four different days in 2017 (April 5, 6, 10, 11), and our neural
networks give a score prediction 0.52, 0.4, 0.43, 0.76 for each day, with an
average score 0.53, which shows no significant indication for the data to lean
toward either the MAD or SANE state.

    

### [[2110.07190] Why Propagate Alone? Parallel Use of Labels and Features on Graphs](http://arxiv.org/abs/2110.07190)


  Graph neural networks (GNNs) and label propagation represent two interrelated
modeling strategies designed to exploit graph structure in tasks such as node
property prediction. The former is typically based on stacked message-passing
layers that share neighborhood information to transform node features into
predictive embeddings. In contrast, the latter involves spreading label
information to unlabeled nodes via a parameter-free diffusion process, but
operates independently of the node features. Given then that the material
difference is merely whether features or labels are smoothed across the graph,
it is natural to consider combinations of the two for improving performance. In
this regard, it has recently been proposed to use a randomly-selected portion
of the training labels as GNN inputs, concatenated with the original node
features for making predictions on the remaining labels. This so-called label
trick accommodates the parallel use of features and labels, and is foundational
to many of the top-ranking submissions on the Open Graph Benchmark (OGB)
leaderboard. And yet despite its wide-spread adoption, thus far there has been
little attempt to carefully unpack exactly what statistical properties the
label trick introduces into the training pipeline, intended or otherwise. To
this end, we prove that under certain simplifying assumptions, the stochastic
label trick can be reduced to an interpretable, deterministic training
objective composed of two factors. The first is a data-fitting term that
naturally resolves potential label leakage issues, while the second serves as a
regularization factor conditioned on graph structure that adapts to graph size
and connectivity. Later, we leverage this perspective to motivate a broader
range of label trick use cases, and provide experiments to verify the efficacy
of these extensions.

    

### [[2110.07191] CNN-DST: ensemble deep learning based on Dempster-Shafer theory for vibration-based fault recognition](http://arxiv.org/abs/2110.07191)


  Nowadays, using vibration data in conjunction with pattern recognition
methods is one of the most common fault detection strategies for structures.
However, their performances depend on the features extracted from vibration
data, the features selected to train the classifier, and the classifier used
for pattern recognition. Deep learning facilitates the fault detection
procedure by automating the feature extraction and selection, and
classification procedure. Though, deep learning approaches have challenges in
designing its structure and tuning its hyperparameters, which may result in a
low generalization capability. Therefore, this study proposes an ensemble deep
learning framework based on a convolutional neural network (CNN) and
Dempster-Shafer theory (DST), called CNN-DST. In this framework, several CNNs
with the proposed structure are first trained, and then, the outputs of the
CNNs selected by the proposed technique are combined by using an improved
DST-based method. To validate the proposed CNN-DST framework, it is applied to
an experimental dataset created by the broadband vibrational responses of
polycrystalline Nickel alloy first-stage turbine blades with different types
and severities of damage. Through statistical analysis, it is shown that the
proposed CNN-DST framework classifies the turbine blades with an average
prediction accuracy of 97.19%. The proposed CNN-DST framework is benchmarked
with other state-of-the-art classification methods, demonstrating its high
performance. The robustness of the proposed CNN-DST framework with respect to
measurement noise is investigated, showing its high noise-resistance. Further,
bandwidth analysis reveals that most of the required information for detecting
faulty samples is available in a small frequency range.

    

### [[2110.07205] SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing](http://arxiv.org/abs/2110.07205)


  Motivated by the success of T5 (Text-To-Text Transfer Transformer) in
pre-training natural language processing models, we propose a unified-modal
SpeechT5 framework that explores the encoder-decoder pre-training for
self-supervised speech/text representation learning. The SpeechT5 framework
consists of a shared encoder-decoder network and six modal-specific
(speech/text) pre/post-nets. After preprocessing the speech/text input through
the pre-nets, the shared encoder-decoder network models the sequence to
sequence transformation, and then the post-nets generate the output in the
speech/text modality based on the decoder output. Particularly, SpeechT5 can
pre-train on a large scale of unlabeled speech and text data to improve the
capability of the speech and textual modeling. To align the textual and speech
information into a unified semantic space, we propose a cross-modal vector
quantization method with random mixing-up to bridge speech and text. Extensive
evaluations on a wide variety of spoken language processing tasks, including
voice conversion, automatic speech recognition, text to speech, and speaker
identification, show the superiority of the proposed SpeechT5 framework.

    

### [[2110.07221] Learning a Compressive Sensing Matrix with Structural Constraints via Maximum Mean Discrepancy Optimization](http://arxiv.org/abs/2110.07221)


  We introduce a learning-based algorithm to obtain a measurement matrix for
compressive sensing related recovery problems. The focus lies on matrices with
a constant modulus constraint which typically represent a network of analog
phase shifters in hybrid precoding/combining architectures. We interpret a
matrix with restricted isometry property as a mapping of points from a high- to
a low-dimensional hypersphere. We argue that points on the low-dimensional
hypersphere, namely, in the range of the matrix, should be uniformly
distributed to increase robustness against measurement noise. This notion is
formalized in an optimization problem which uses one of the maximum mean
discrepancy metrics in the objective function. Recent success of such metrics
in neural network related topics motivate a solution of the problem based on
machine learning. Numerical experiments show better performance than random
measurement matrices that are generally employed in compressive sensing
contexts. Further, we adapt a method from the literature to the constant
modulus constraint. This method can also compete with random matrices and it is
shown to harmonize well with the proposed learning-based approach if it is used
as an initialization. Lastly, we describe how other structural matrix
constraints, e.g., a Toeplitz constraint, can be taken into account, too.

    

### [[2110.07232] Procrastinated Tree Search: Black-box Optimization with Delayed, Noisy, and Multi-fidelity Feedback](http://arxiv.org/abs/2110.07232)


  In black-box optimization problems, we aim to maximize an unknown objective
function, where the function is only accessible through feedbacks of an
evaluation or simulation oracle. In real-life, the feedbacks of such oracles
are often noisy and available after some unknown delay that may depend on the
computation time of the oracle. Additionally, if the exact evaluations are
expensive but coarse approximations are available at a lower cost, the
feedbacks can have multi-fidelity. In order to address this problem, we propose
a generic extension of hierarchical optimistic tree search (HOO), called
ProCrastinated Tree Search (PCTS), that flexibly accommodates a delay and
noise-tolerant bandit algorithm. We provide a generic proof technique to
quantify regret of PCTS under delayed, noisy, and multi-fidelity feedbacks.
Specifically, we derive regret bounds of PCTS enabled with delayed-UCB1 (DUCB1)
and delayed-UCB-V (DUCBV) algorithms. Given a horizon $T$, PCTS retains the
regret bound of non-delayed HOO for expected delay of $O(\log T)$ and worsens
by $O(T^{\frac{1-\alpha}{d+2}})$ for expected delays of $O(T^{1-\alpha})$ for
$\alpha \in (0,1]$. We experimentally validate on multiple synthetic functions
and hyperparameter tuning problems that PCTS outperforms the state-of-the-art
black-box optimization methods for feedbacks with different noise levels,
delays, and fidelity.

    

### [[2110.07234] On the Stability of Low Pass Graph Filter With a Large Number of Edge Rewires](http://arxiv.org/abs/2110.07234)


  Recently, the stability of graph filters has been studied as one of the key
theoretical properties driving the highly successful graph convolutional neural
networks (GCNs). The stability of a graph filter characterizes the effect of
topology perturbation on the output of a graph filter, a fundamental building
block for GCNs. Many existing results have focused on the regime of small
perturbation with a small number of edge rewires. However, the number of edge
rewires can be large in many applications. To study the latter case, this work
departs from the previous analysis and proves a bound on the stability of graph
filter relying on the filter's frequency response. Assuming the graph filter is
low pass, we show that the stability of the filter depends on perturbation to
the community structure. As an application, we show that for stochastic block
model graphs, the graph filter distance converges to zero when the number of
nodes approaches infinity. Numerical simulations validate our findings.

    

### [[2110.07235] HUMAN4D: A Human-Centric Multimodal Dataset for Motions and Immersive Media](http://arxiv.org/abs/2110.07235)


  We introduce HUMAN4D, a large and multimodal 4D dataset that contains a
variety of human activities simultaneously captured by a professional
marker-based MoCap, a volumetric capture and an audio recording system. By
capturing 2 female and $2$ male professional actors performing various
full-body movements and expressions, HUMAN4D provides a diverse set of motions
and poses encountered as part of single- and multi-person daily, physical and
social activities (jumping, dancing, etc.), along with multi-RGBD (mRGBD),
volumetric and audio data. Despite the existence of multi-view color datasets
captured with the use of hardware (HW) synchronization, to the best of our
knowledge, HUMAN4D is the first and only public resource that provides
volumetric depth maps with high synchronization precision due to the use of
intra- and inter-sensor HW-SYNC. Moreover, a spatio-temporally aligned scanned
and rigged 3D character complements HUMAN4D to enable joint research on
time-varying and high-quality dynamic meshes. We provide evaluation baselines
by benchmarking HUMAN4D with state-of-the-art human pose estimation and 3D
compression methods. For the former, we apply 2D and 3D pose estimation
algorithms both on single- and multi-view data cues. For the latter, we
benchmark open-source 3D codecs on volumetric data respecting online volumetric
video encoding and steady bit-rates. Furthermore, qualitative and quantitative
visual comparison between mesh-based volumetric data reconstructed in different
qualities showcases the available options with respect to 4D representations.
HUMAN4D is introduced to the computer vision and graphics research communities
to enable joint research on spatio-temporally aligned pose, volumetric, mRGBD
and audio data cues. The dataset and its code are available
this https URL.

    

### [[2110.07238] How to train RNNs on chaotic data?](http://arxiv.org/abs/2110.07238)


  Recurrent neural networks (RNNs) are wide-spread machine learning tools for
modeling sequential and time series data. They are notoriously hard to train
because their loss gradients backpropagated in time tend to saturate or diverge
during training. This is known as the exploding and vanishing gradient problem.
Previous solutions to this issue either built on rather complicated,
purpose-engineered architectures with gated memory buffers, or - more recently
- imposed constraints that ensure convergence to a fixed point or restrict (the
eigenspectrum of) the recurrence matrix. Such constraints, however, convey
severe limitations on the expressivity of the RNN. Essential intrinsic dynamics
such as multistability or chaos are disabled. This is inherently at disaccord
with the chaotic nature of many, if not most, time series encountered in nature
and society. Here we offer a comprehensive theoretical treatment of this
problem by relating the loss gradients during RNN training to the Lyapunov
spectrum of RNN-generated orbits. We mathematically prove that RNNs producing
stable equilibrium or cyclic behavior have bounded gradients, whereas the
gradients of RNNs with chaotic dynamics always diverge. Based on these analyses
and insights, we offer an effective yet simple training technique for chaotic
data and guidance on how to choose relevant hyperparameters according to the
Lyapunov spectrum.

    

### [[2110.07246] HAVEN: Hierarchical Cooperative Multi-Agent Reinforcement Learning with Dual Coordination Mechanism](http://arxiv.org/abs/2110.07246)


  Multi-agent reinforcement learning often suffers from the exponentially
larger action space caused by a large number of agents. In this paper, we
propose a novel value decomposition framework HAVEN based on hierarchical
reinforcement learning for the fully cooperative multi-agent problems. In order
to address instabilities that arise from the concurrent optimization of
high-level and low-level policies and another concurrent optimization of
agents, we introduce the dual coordination mechanism of inter-layer strategies
and inter-agent strategies. HAVEN does not require domain knowledge and
pretraining at all, and can be applied to any value decomposition variants. Our
method is demonstrated to achieve superior results to many baselines on
StarCraft II micromanagement tasks and offers an efficient solution to
multi-agent hierarchical reinforcement learning in fully cooperative scenarios.

    

### [[2110.07275] Order Constraints in Optimal Transport](http://arxiv.org/abs/2110.07275)


  Optimal transport is a framework for comparing measures whereby a cost is
incurred for transporting one measure to another. Recent works have aimed to
improve optimal transport plans through the introduction of various forms of
structure. We introduce novel order constraints into the optimal transport
formulation to allow for the incorporation of structure. While there will are
now quadratically many constraints as before, we prove a $\delta-$approximate
solution to the order-constrained optimal transport problem can be obtained in
$\mathcal{O}(L^2\delta^{-2} \kappa(\delta(2cL_\infty (1+(mn)^{1/2}))^{-1})
\cdot mn\log mn)$ time. We derive computationally efficient lower bounds that
allow for an explainable approach to adding structure to the optimal transport
plan through order constraints. We demonstrate experimentally that order
constraints improve explainability using the e-SNLI (Stanford Natural Language
Inference) dataset that includes human-annotated rationales for each
assignment.

    

### [[2110.07276] Carousel Memory: Rethinking the Design of Episodic Memory for Continual Learning](http://arxiv.org/abs/2110.07276)


  Continual Learning (CL) is an emerging machine learning paradigm that aims to
learn from a continuous stream of tasks without forgetting knowledge learned
from the previous tasks. To avoid performance decrease caused by forgetting,
prior studies exploit episodic memory (EM), which stores a subset of the past
observed samples while learning from new non-i.i.d. data. Despite the promising
results, since CL is often assumed to execute on mobile or IoT devices, the EM
size is bounded by the small hardware memory capacity and makes it infeasible
to meet the accuracy requirements for real-world applications. Specifically,
all prior CL methods discard samples overflowed from the EM and can never
retrieve them back for subsequent training steps, incurring loss of information
that would exacerbate catastrophic forgetting. We explore a novel hierarchical
EM management strategy to address the forgetting issue. In particular, in
mobile and IoT devices, real-time data can be stored not just in high-speed
RAMs but in internal storage devices as well, which offer significantly larger
capacity than the RAMs. Based on this insight, we propose to exploit the
abundant storage to preserve past experiences and alleviate the forgetting by
allowing CL to efficiently migrate samples between memory and storage without
being interfered by the slow access speed of the storage. We call it Carousel
Memory (CarM). As CarM is complementary to existing CL methods, we conduct
extensive evaluations of our method with seven popular CL methods and show that
CarM significantly improves the accuracy of the methods across different
settings by large margins in final average accuracy (up to 28.4%) while
retaining the same training efficiency.

    

### [[2110.07283] DeepMoCap: Deep Optical Motion Capture Using Multiple Depth Sensors and Retro-Reflectors](http://arxiv.org/abs/2110.07283)


  In this paper, a marker-based, single-person optical motion capture method
(DeepMoCap) is proposed using multiple spatio-temporally aligned infrared-depth
sensors and retro-reflective straps and patches (reflectors). DeepMoCap
explores motion capture by automatically localizing and labeling reflectors on
depth images and, subsequently, on 3D space. Introducing a non-parametric
representation to encode the temporal correlation among pairs of colorized
depthmaps and 3D optical flow frames, a multi-stage Fully Convolutional Network
(FCN) architecture is proposed to jointly learn reflector locations and their
temporal dependency among sequential frames. The extracted reflector 2D
locations are spatially mapped in 3D space, resulting in robust 3D optical data
extraction. The subject's motion is efficiently captured by applying a
template-based fitting technique on the extracted optical data. Two datasets
have been created and made publicly available for evaluation purposes; one
comprising multi-view depth and 3D optical flow annotated images (DMC2.5D), and
a second, consisting of spatio-temporally aligned multi-view depth images along
with skeleton, inertial and ground truth MoCap data (DMC3D). The FCN model
outperforms its competitors on the DMC2.5D dataset using 2D Percentage of
Correct Keypoints (PCK) metric, while the motion capture outcome is evaluated
against RGB-D and inertial data fusion approaches on DMC3D, outperforming the
next best method by 4.5% in total 3D PCK accuracy.

    

### [[2110.07292] Sign and Relevance learning](http://arxiv.org/abs/2110.07292)


  Standard models of biologically realistic, or inspired, reinforcement
learning employ a global error signal which implies shallow networks. However,
deep networks could offer a drastically superior performance by feeding the
error signal backwards through such a network which in turn is not biologically
realistic as it requires symmetric weights between top-down and bottom-up
pathways. Instead, we present a network combining local learning with global
modulation where neuromodulation controls the amount of plasticity change in
the whole network, while only the sign of the error is backpropagated through
the network. The neuromodulation can be understood as a rectified error, or
relevance, signal while the bottom-up sign of the error signal decides between
long-term potentiation and long-term depression. We demonstrate the performance
of this paradigm with a real robotic task.

    

### [[2110.07301] Multi-task problems are not multi-objective](http://arxiv.org/abs/2110.07301)


  Multi-objective optimization (MOO) aims at finding a set of optimal
configurations for a given set of objectives. A recent line of work applies MOO
methods to the typical Machine Learning (ML) setting, which becomes
multi-objective if a model should optimize more than one objective, for
instance in fair machine learning. These works also use Multi-Task Learning
(MTL) problems to benchmark MOO algorithms treating each task as independent
objective.
In this work we show that MTL problems do not resemble the characteristics of
MOO problems. In particular, MTL losses are not competing in case of a
sufficiently expressive single model. As a consequence, a single model can
perform just as well as optimizing all objectives with independent models,
rendering MOO inapplicable. We provide evidence with extensive experiments on
the widely used Multi-Fashion-MNIST datasets. Our results call for new
benchmarks to evaluate MOO algorithms for ML. Our code is available at:
this https URL.

    

### [[2110.07305] DI-AA: An Interpretable White-box Attack for Fooling Deep Neural Networks](http://arxiv.org/abs/2110.07305)


  White-box Adversarial Example (AE) attacks towards Deep Neural Networks
(DNNs) have a more powerful destructive capacity than black-box AE attacks in
the fields of AE strategies. However, almost all the white-box approaches lack
interpretation from the point of view of DNNs. That is, adversaries did not
investigate the attacks from the perspective of interpretable features, and few
of these approaches considered what features the DNN actually learns. In this
paper, we propose an interpretable white-box AE attack approach, DI-AA, which
explores the application of the interpretable approach of the deep Taylor
decomposition in the selection of the most contributing features and adopts the
Lagrangian relaxation optimization of the logit output and L_p norm to further
decrease the perturbation. We compare DI-AA with six baseline attacks
(including the state-of-the-art attack AutoAttack) on three datasets.
Experimental results reveal that our proposed approach can 1) attack non-robust
models with comparatively low perturbation, where the perturbation is closer to
or lower than the AutoAttack approach; 2) break the TRADES adversarial training
models with the highest success rate; 3) the generated AE can reduce the robust
accuracy of the robust black-box models by 16% to 31% in the black-box transfer
attack.

    

### [[2110.07311] SpecSinGAN: Sound Effect Variation Synthesis Using Single-Image GANs](http://arxiv.org/abs/2110.07311)


  Single-image generative adversarial networks learn from the internal
distribution of a single training example to generate variations of it,
removing the need of a large dataset. In this paper we introduce SpecSinGAN, an
unconditional generative architecture that takes a single one-shot sound effect
(e.g., a footstep; a character jump) and produces novel variations of it, as if
they were different takes from the same recording session. We explore the use
of multi-channel spectrograms to train the model on the various layers that
comprise a single sound effect. A listening study comparing our model to real
recordings and to digital signal processing procedural audio models in terms of
sound plausibility and variation revealed that SpecSinGAN is more plausible and
varied than the procedural audio models considered, when using multi-channel
spectrograms. Sound examples can be found at the project website:
this https URL


### [[2110.07313] Conformer-Based Self-Supervised Learning for Non-Speech Audio Tasks](http://arxiv.org/abs/2110.07313)


  Representation learning from unlabeled data has been of major interest in
artificial intelligence research. While self-supervised speech representation
learning has been popular in the speech research community, very few works have
comprehensively analyzed audio representation learning for non-speech audio
tasks. In this paper, we propose a self-supervised audio representation
learning method and apply it to a variety of downstream non-speech audio tasks.
We combine the well-known wav2vec 2.0 framework, which has shown success in
self-supervised learning for speech tasks, with parameter-efficient conformer
architectures. On the AudioSet benchmark, we achieve a mean average precision
(mAP) score of 0.415, which is a new state-of-the-art on this dataset through
audio-only self-supervised learning. Our fine-tuned conformers also surpass or
match the performance of previous systems pre-trained in a supervised way on
several downstream tasks. We further discuss the important design
considerations for both pre-training and fine-tuning.

    

### [[2110.07317] ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection](http://arxiv.org/abs/2110.07317)


  Identifying vulnerabilities in the source code is essential to protect the
software systems from cyber security attacks. It, however, is also a
challenging step that requires specialized expertise in security and code
representation. Inspired by the successful applications of pre-trained
programming language (PL) models such as CodeBERT and graph neural networks
(GNNs), we propose ReGVD, a general and novel graph neural network-based model
for vulnerability detection. In particular, ReGVD views a given source code as
a flat sequence of tokens and then examines two effective methods of utilizing
unique tokens and indexes respectively to construct a single graph as an input,
wherein node features are initialized only by the embedding layer of a
pre-trained PL model. Next, ReGVD leverages a practical advantage of residual
connection among GNN layers and explores a beneficial mixture of graph-level
sum and max poolings to return a graph embedding for the given source code.
Experimental results demonstrate that ReGVD outperforms the existing
state-of-the-art models and obtain the highest accuracy on the real-world
benchmark dataset from CodeXGLUE for vulnerability detection.

    

### [[2110.07336] RPT: Toward Transferable Model on Heterogeneous Researcher Data via Pre-Training](http://arxiv.org/abs/2110.07336)


  With the growth of the academic engines, the mining and analysis acquisition
of massive researcher data, such as collaborator recommendation and researcher
retrieval, has become indispensable. It can improve the quality of services and
intelligence of academic engines. Most of the existing studies for researcher
data mining focus on a single task for a particular application scenario and
learning a task-specific model, which is usually unable to transfer to
out-of-scope tasks. The pre-training technology provides a generalized and
sharing model to capture valuable information from enormous unlabeled data. The
model can accomplish multiple downstream tasks via a few fine-tuning steps. In
this paper, we propose a multi-task self-supervised learning-based researcher
data pre-training model named RPT. Specifically, we divide the researchers'
data into semantic document sets and community graph. We design the
hierarchical Transformer and the local community encoder to capture information
from the two categories of data, respectively. Then, we propose three
self-supervised learning objectives to train the whole model. Finally, we also
propose two transfer modes of RPT for fine-tuning in different scenarios. We
conduct extensive experiments to evaluate RPT, results on three downstream
tasks verify the effectiveness of pre-training for researcher data mining.

    

### [[2110.07342] FILM: Following Instructions in Language with Modular Methods](http://arxiv.org/abs/2110.07342)


  Recent methods for embodied instruction following are typically trained
end-to-end using imitation learning. This requires the use of expert
trajectories and low-level language instructions. Such approaches assume
learned hidden states will simultaneously integrate semantics from the language
and vision to perform state tracking, spatial memory, exploration, and
long-term planning. In contrast, we propose a modular method with structured
representations that (1) builds a semantic map of the scene, and (2) performs
exploration with a semantic search policy, to achieve the natural language
goal. Our modular method achieves SOTA performance (24.46%) with a substantial
(8.17 % absolute) gap from previous work while using less data by eschewing
both expert trajectories and low-level instructions. Leveraging low-level
language, however, can further increase our performance (26.49%). Our findings
suggest that an explicit spatial memory and a semantic search policy can
provide a stronger and more general representation for state-tracking and
guidance, even in the absence of expert trajectories or low-level instructions.

    

### [[2110.07347] Improved Drug-target Interaction Prediction with Intermolecular Graph Transformer](http://arxiv.org/abs/2110.07347)


  The identification of active binding drugs for target proteins (termed as
drug-target interaction prediction) is the key challenge in virtual screening,
which plays an essential role in drug discovery. Although recent deep
learning-based approaches achieved better performance than molecular docking,
existing models often neglect certain aspects of the intermolecular
information, hindering the performance of prediction. We recognize this problem
and propose a novel approach named Intermolecular Graph Transformer (IGT) that
employs a dedicated attention mechanism to model intermolecular information
with a three-way Transformer-based architecture. IGT outperforms
state-of-the-art approaches by 9.1% and 20.5% over the second best for binding
activity and binding pose prediction respectively, and shows superior
generalization ability to unseen receptor proteins. Furthermore, IGT exhibits
promising drug screening ability against SARS-CoV-2 by identifying 83.1% active
drugs that have been validated by wet-lab experiments with near-native
predicted binding poses.

    

### [[2110.07353] Interpretable transformed ANOVA approximation on the example of the prevention of forest fires](http://arxiv.org/abs/2110.07353)


  The distribution of data points is a key component in machine learning. In
most cases, one uses min-max normalization to obtain nodes in $[0,1]$ or
Z-score normalization for standard normal distributed data. In this paper, we
apply transformation ideas in order to design a complete orthonormal system in
the $\mathrm{L}_2$ space of functions with the standard normal distribution as
integration weight. Subsequently, we are able to apply the explainable ANOVA
approximation for this basis and use Z-score transformed data in the method. We
demonstrate the applicability of this procedure on the well-known forest fires
data set from the UCI machine learning repository. The attribute ranking
obtained from the ANOVA approximation provides us with crucial information
about which variables in the data set are the most important for the detection
of fires.

    

### [[2110.07354] Music Playlist Title Generation: A Machine-Translation Approach](http://arxiv.org/abs/2110.07354)


  We propose a machine-translation approach to automatically generate a
playlist title from a set of music tracks. We take a sequence of track IDs as
input and a sequence of words in a playlist title as output, adapting the
sequence-to-sequence framework based on Recurrent Neural Network (RNN) and
Transformer to the music data. Considering the orderless nature of music tracks
in a playlist, we propose two techniques that remove the order of the input
sequence. One is data augmentation by shuffling and the other is deleting the
positional encoding. We also reorganize the existing music playlist datasets to
generate phrase-level playlist titles. The result shows that the Transformer
models generally outperform the RNN model. Also, removing the order of input
sequence improves the performance further.

    

### [[2110.07356] Medically Aware GPT-3 as a Data Generator for Medical Dialogue Summarization](http://arxiv.org/abs/2110.07356)


  In medical dialogue summarization, summaries must be coherent and must
capture all the medically relevant information in the dialogue. However,
learning effective models for summarization require large amounts of labeled
data which is especially hard to obtain. We present an algorithm to create
synthetic training data with an explicit focus on capturing medically relevant
information. We utilize GPT-3 as the backbone of our algorithm and scale 210
human labeled examples to yield results comparable to using 6400 human labeled
examples (~30x) leveraging low-shot learning and an ensemble method. In
detailed experiments, we show that this approach produces high quality training
data that can further be combined with human labeled data to get summaries that
are strongly preferable to those produced by models trained on human data alone
both in terms of medical accuracy and coherency.

    

### [[2110.07374] Physics informed neural networks for continuum micromechanics](http://arxiv.org/abs/2110.07374)


  Recently, physics informed neural networks have successfully been applied to
a broad variety of problems in applied mathematics and engineering. The
principle idea is to use a neural network as a global ansatz function to
partial differential equations. Due to the global approximation, physics
informed neural networks have difficulties in displaying localized effects and
strong non-linear solutions by optimization. In this work we consider material
non-linearities invoked by material inhomogeneities with sharp phase
interfaces. This constitutes a challenging problem for a method relying on a
global ansatz. To overcome convergence issues, adaptive training strategies and
domain decomposition are studied. It is shown, that the domain decomposition
approach is able to accurately resolve nonlinear stress, displacement and
energy fields in heterogeneous microstructures obtained from real-world
$\mu$CT-scans.

    

### [[2110.07383] The Neglected Sibling: Isotropic Gaussian Posterior for VAE](http://arxiv.org/abs/2110.07383)


  Deep generative models have been widely used in several areas of NLP, and
various techniques have been proposed to augment them or address their training
challenges. In this paper, we propose a simple modification to Variational
Autoencoders (VAEs) by using an Isotropic Gaussian Posterior (IGP) that allows
for better utilisation of their latent representation space. This model avoids
the sub-optimal behavior of VAEs related to inactive dimensions in the
representation space. We provide both theoretical analysis, and empirical
evidence on various datasets and tasks that show IGP leads to consistent
improvement on several quantitative and qualitative grounds, from downstream
task performance and sample efficiency to robustness. Additionally, we give
insights about the representational properties encouraged by IGP and also show
that its gain generalises to image domain as well.

    

### [[2110.07385] Few-shot Controllable Style Transfer for Low-Resource Settings: A Study in Indian Languages](http://arxiv.org/abs/2110.07385)


  Style transfer is the task of rewriting an input sentence into a target style
while approximately preserving its content. While most prior literature assumes
access to large style-labelled corpora, recent work (Riley et al. 2021) has
attempted "few-shot" style transfer using only 3-10 sentences at inference for
extracting the target style. In this work we consider one such low resource
setting where no datasets are available: style transfer for Indian languages.
We find that existing few-shot methods perform this task poorly, with a strong
tendency to copy inputs verbatim. We push the state-of-the-art for few-shot
style transfer with a new method modeling the stylistic difference between
paraphrases. When compared to prior work using automatic and human evaluations,
our model achieves 2-3x better performance and output diversity in formality
transfer and code-mixing addition across five Indian languages. Moreover, our
method is better able to control the amount of style transfer using an input
scalar knob. We report promising qualitative results for several attribute
transfer directions, including sentiment transfer, text simplification, gender
neutralization and text anonymization, all without retraining the model.
Finally we found model evaluation to be difficult due to the lack of evaluation
datasets and metrics for Indian languages. To facilitate further research in
formality transfer for Indic languages, we crowdsource annotations for 4000
sentence pairs in four languages, and use this dataset to design our automatic
evaluation suite.

    

### [[2110.07392] Provably Efficient Multi-Agent Reinforcement Learning with Fully Decentralized Communication](http://arxiv.org/abs/2110.07392)


  A challenge in reinforcement learning (RL) is minimizing the cost of sampling
associated with exploration. Distributed exploration reduces sampling
complexity in multi-agent RL (MARL). We investigate the benefits to performance
in MARL when exploration is fully decentralized. Specifically, we consider a
class of online, episodic, tabular $Q$-learning problems under time-varying
reward and transition dynamics, in which agents can communicate in a
decentralized manner.We show that group performance, as measured by the bound
on regret, can be significantly improved through communication when each agent
uses a decentralized message-passing protocol, even when limited to sending
information up to its $\gamma$-hop neighbors. We prove regret and sample
complexity bounds that depend on the number of agents, communication network
structure and $\gamma.$ We show that incorporating more agents and more
information sharing into the group learning scheme speeds up convergence to the
optimal policy. Numerical simulations illustrate our results and validate our
theoretical claims.

    

### [[2110.07395] Offline Reinforcement Learning with Soft Behavior Regularization](http://arxiv.org/abs/2110.07395)


  Most prior approaches to offline reinforcement learning (RL) utilize
\textit{behavior regularization}, typically augmenting existing off-policy
actor critic algorithms with a penalty measuring divergence between the policy
and the offline data. However, these approaches lack guaranteed performance
improvement over the behavior policy. In this work, we start from the
performance difference between the learned policy and the behavior policy, we
derive a new policy learning objective that can be used in the offline setting,
which corresponds to the advantage function value of the behavior policy,
multiplying by a state-marginal density ratio. We propose a practical way to
compute the density ratio and demonstrate its equivalence to a state-dependent
behavior regularization. Unlike state-independent regularization used in prior
approaches, this \textit{soft} regularization allows more freedom of policy
deviation at high confidence states, leading to better performance and
stability. We thus term our resulting algorithm Soft Behavior-regularized Actor
Critic (SBAC). Our experimental results show that SBAC matches or outperforms
the state-of-the-art on a set of continuous control locomotion and manipulation
tasks.

    

### [[2110.07402] Self-Supervised Learning by Estimating Twin Class Distributions](http://arxiv.org/abs/2110.07402)


  We present TWIST, a novel self-supervised representation learning method by
classifying large-scale unlabeled datasets in an end-to-end way. We employ a
siamese network terminated by a softmax operation to produce twin class
distributions of two augmented images. Without supervision, we enforce the
class distributions of different augmentations to be consistent. In the
meantime, we regularize the class distributions to make them sharp and diverse.
Specifically, we minimize the entropy of the distribution for each sample to
make the class prediction for each sample assertive and maximize the entropy of
the mean distribution to make the predictions of different samples diverse. In
this way, TWIST can naturally avoid the trivial solutions without specific
designs such as asymmetric network, stop-gradient operation, or momentum
encoder. Different from the clustering-based methods which alternate between
clustering and learning, our method is a single learning process guided by a
unified loss function. As a result, TWIST outperforms state-of-the-art methods
on a wide range of tasks, including unsupervised classification, linear
classification, semi-supervised learning, transfer learning, and some dense
prediction tasks such as detection and segmentation.

    

### [[2110.07409] The Geometry of Memoryless Stochastic Policy Optimization in Infinite-Horizon POMDPs](http://arxiv.org/abs/2110.07409)


  We consider the problem of finding the best memoryless stochastic policy for
an infinite-horizon partially observable Markov decision process (POMDP) with
finite state and action spaces with respect to either the discounted or mean
reward criterion. We show that the (discounted) state-action frequencies and
the expected cumulative reward are rational functions of the policy, whereby
the degree is determined by the degree of partial observability. We then
describe the optimization problem as a linear optimization problem in the space
of feasible state-action frequencies subject to polynomial constraints that we
characterize explicitly. This allows us to address the combinatorial and
geometric complexity of the optimization problem using recent tools from
polynomial optimization. In particular, we demonstrate how the partial
observability constraints can lead to multiple smooth and non-smooth local
optimizers and we estimate the number of critical points.

    

### [[2110.07410] Evaluating Off-the-Shelf Machine Listening and Natural Language Models for Automated Audio Captioning](http://arxiv.org/abs/2110.07410)


  Automated audio captioning (AAC) is the task of automatically generating
textual descriptions for general audio signals. A captioning system has to
identify various information from the input signal and express it with natural
language. Existing works mainly focus on investigating new methods and try to
improve their performance measured on existing datasets. Having attracted
attention only recently, very few works on AAC study the performance of
existing pre-trained audio and natural language processing resources. In this
paper, we evaluate the performance of off-the-shelf models with a
Transformer-based captioning approach. We utilize the freely available Clotho
dataset to compare four different pre-trained machine listening models, four
word embedding models, and their combinations in many different settings. Our
evaluation suggests that YAMNet combined with BERT embeddings produces the best
captions. Moreover, in general, fine-tuning pre-trained word embeddings can
lead to better performance. Finally, we show that sequences of audio embeddings
can be processed using a Transformer encoder to produce higher-quality
captions.

    

### [[2110.07430] Detecting Renewal States in Chains of Variable Length via Intrinsic Bayes Factors](http://arxiv.org/abs/2110.07430)


  Markov chains with variable length are useful parsimonious stochastic models
able to generate most stationary sequence of discrete symbols. The idea is to
identify the suffixes of the past, called contexts, that are relevant to
predict the future symbol. Sometimes a single state is a context, and looking
at the past and finding this specific state makes the further past irrelevant.
These states are called renewal states and they split the chain into
independent blocks. In order to identify renewal states for chains with
variable length, we propose the use of Intrinsic Bayes Factor to evaluate the
plausibility of each set of renewal states. In this case, the difficulty lies
in finding the marginal posterior distribution for the random context trees for
general prior distribution on the space of context trees and Dirichlet prior
for the transition probabilities. To show the strength of our method, we
analyzed artificial datasets generated from two binary models models and one
example coming from the field of Linguistics.

    

### [[2110.07433] Possibilistic Fuzzy Local Information C-Means with Automated Feature Selection for Seafloor Segmentation](http://arxiv.org/abs/2110.07433)


  The Possibilistic Fuzzy Local Information C-Means (PFLICM) method is
presented as a technique to segment side-look synthetic aperture sonar (SAS)
imagery into distinct regions of the sea-floor. In this work, we investigate
and present the results of an automated feature selection approach for SAS
image segmentation. The chosen features and resulting segmentation from the
image will be assessed based on a select quantitative clustering validity
criterion and the subset of the features that reach a desired threshold will be
used for the segmentation process.

    

### [[2110.07435] Adaptive Differentially Private Empirical Risk Minimization](http://arxiv.org/abs/2110.07435)


  We propose an adaptive (stochastic) gradient perturbation method for
differentially private empirical risk minimization. At each iteration, the
random noise added to the gradient is optimally adapted to the stepsize; we
name this process adaptive differentially private (ADP) learning. Given the
same privacy budget, we prove that the ADP method considerably improves the
utility guarantee compared to the standard differentially private method in
which vanilla random noise is added. Our method is particularly useful for
gradient-based algorithms with time-varying learning rates, including variants
of AdaGrad (Duchi et al., 2011). We provide extensive numerical experiments to
demonstrate the effectiveness of the proposed adaptive differentially private
algorithm.

    

### [[2110.07436] Asymmetric Graph Representation Learning](http://arxiv.org/abs/2110.07436)


  Despite the enormous success of graph neural networks (GNNs), most existing
GNNs can only be applicable to undirected graphs where relationships among
connected nodes are two-way symmetric (i.e., information can be passed back and
forth). However, there is a vast amount of applications where the information
flow is asymmetric, leading to directed graphs where information can only be
passed in one direction. For example, a directed edge indicates that the
information can only be conveyed forwardly from the start node to the end node,
but not backwardly. To accommodate such an asymmetric structure of directed
graphs within the framework of GNNs, we propose a simple yet remarkably
effective framework for directed graph analysis to incorporate such one-way
information passing. We define an incoming embedding and an outgoing embedding
for each node to model its sending and receiving features respectively. We
further develop two steps in our directed GNN model with the first one to
aggregate/update the incoming features of nodes and the second one to
aggregate/update the outgoing features. By imposing the two roles for each
node, the likelihood of a directed edge can be calculated based on the outgoing
embedding of the start node and the incoming embedding of the end node. The
log-likelihood of all edges plays a natural role of regularization for the
proposed model, which can alleviate the over-smoothing problem of the deep
GNNs. Extensive experiments on multiple real-world directed graphs demonstrate
outstanding performances of the proposed model in both node-level and
graph-level tasks.

    

### [[2110.07439] Inverse Problems Leveraging Pre-trained Contrastive Representations](http://arxiv.org/abs/2110.07439)


  We study a new family of inverse problems for recovering representations of
corrupted data. We assume access to a pre-trained representation learning
network R(x) that operates on clean images, like CLIP. The problem is to
recover the representation of an image R(x), if we are only given a corrupted
version A(x), for some known forward operator A. We propose a supervised
inversion method that uses a contrastive objective to obtain excellent
representations for highly corrupted images. Using a linear probe on our robust
representations, we achieve a higher accuracy than end-to-end supervised
baselines when classifying images with various types of distortions, including
blurring, additive noise, and random pixel masking. We evaluate on a subset of
ImageNet and observe that our method is robust to varying levels of distortion.
Our method outperforms end-to-end baselines even with a fraction of the labeled
data in a wide range of forward operators.

    

### [[2110.07443] DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing](http://arxiv.org/abs/2110.07443)


  Continuous integration testing is an important step in the modern software
engineering life cycle. Test prioritization is a method that can improve the
efficiency of continuous integration testing by selecting test cases that can
detect faults in the early stage of each cycle. As continuous integration
testing produces voluminous test execution data, test history is a commonly
used artifact in test prioritization. However, existing test prioritization
techniques for continuous integration either cannot handle large test history
or are optimized for using a limited number of historical test cycles. We show
that such a limitation can decrease fault detection effectiveness of
prioritized test suites.
This work introduces DeepOrder, a deep learning-based model that works on the
basis of regression machine learning. DeepOrder ranks test cases based on the
historical record of test executions from any number of previous test cycles.
DeepOrder learns failed test cases based on multiple factors including the
duration and execution status of test cases. We experimentally show that deep
neural networks, as a simple regression model, can be efficiently used for test
case prioritization in continuous integration testing. DeepOrder is evaluated
with respect to time-effectiveness and fault detection effectiveness in
comparison with an industry practice and the state of the art approaches. The
results show that DeepOrder outperforms the industry practice and
state-of-the-art test prioritization approaches in terms of these two metrics.

    

### [[2110.07448] Human-Robot Collaboration and Machine Learning: A Systematic Review of Recent Research](http://arxiv.org/abs/2110.07448)


  Technological progress increasingly envisions the use of robots interacting
with people in everyday life. Human-robot collaboration (HRC) is the approach
that explores the interaction between a human and a robot, during the
completion of an actual physical task. Such interplay is explored both at the
cognitive and physical level, by respectively analysing the mutual exchange of
information and mechanical power. In HRC works, a cognitive model is typically
built, which collects inputs from the environment and from the user, elaborates
and translates these into information that can be used by the robot itself. HRC
studies progressively employ machine learning algorithms to build the cognitive
models and behavioural block that elaborates the acquired external inputs. This
is a promising approach still in its early stages and with the potential of
significant benefit from the growing field of machine learning. Consequently,
this paper proposes a thorough literature review of the use of machine learning
techniques in the context of human-robot collaboration. The
collection,selection and analysis of the set of 45 key papers, selected from
the wide review of the literature on robotics and machine learning, allowed the
identification of the current trends in HRC. In particular, a clustering of
works based on the type of collaborative tasks, evaluation metrics and
cognitive variables modelled is proposed. With these premises, a deep analysis
on different families of machine learning algorithms and their properties,
along with the sensing modalities used, was carried out. The salient aspects of
the analysis are discussed to show trends and suggest possible challenges to
tackle in the future research.

    

### [[2110.07460] IB-GAN: A Unified Approach for Multivariate Time Series Classification under Class Imbalance](http://arxiv.org/abs/2110.07460)


  Classification of large multivariate time series with strong class imbalance
is an important task in real-world applications. Standard methods of class
weights, oversampling, or parametric data augmentation do not always yield
significant improvements for predicting minority classes of interest.
Non-parametric data augmentation with Generative Adversarial Networks (GANs)
offers a promising solution. We propose Imputation Balanced GAN (IB-GAN), a
novel method that joins data augmentation and classification in a one-step
process via an imputation-balancing approach. IB-GAN uses imputation and
resampling techniques to generate higher quality samples from randomly masked
vectors than from white noise, and augments classification through a
class-balanced set of real and synthetic samples. Imputation hyperparameter
$p_{miss}$ allows for regularization of classifier variability by tuning
innovations introduced via generator imputation. IB-GAN is simple to train and
model-agnostic, pairing any deep learning classifier with a
generator-discriminator duo and resulting in higher accuracy for under-observed
classes. Empirical experiments on open-source UCR data and proprietary 90K
product dataset show significant performance gains against state-of-the-art
parametric and GAN baselines.

    

### [[2110.07462] On Adversarial Vulnerability of PHM algorithms: An Initial Study](http://arxiv.org/abs/2110.07462)


  With proliferation of deep learning (DL) applications in diverse domains,
vulnerability of DL models to adversarial attacks has become an increasingly
interesting research topic in the domains of Computer Vision (CV) and Natural
Language Processing (NLP). DL has also been widely adopted to diverse PHM
applications, where data are primarily time-series sensor measurements. While
those advanced DL algorithms/models have resulted in an improved PHM
algorithms' performance, the vulnerability of those PHM algorithms to
adversarial attacks has not drawn much attention in the PHM community. In this
paper we attempt to explore the vulnerability of PHM algorithms. More
specifically, we investigate the strategies of attacking PHM algorithms by
considering several unique characteristics associated with time-series sensor
measurements data. We use two real-world PHM applications as examples to
validate our attack strategies and to demonstrate that PHM algorithms indeed
are vulnerable to adversarial attacks.

    

### [[2110.07467] Hybrid Quantum-Classical Neural Network for Cloud-supported In-Vehicle Cyberattack Detection](http://arxiv.org/abs/2110.07467)


  A classical computer works with ones and zeros, whereas a quantum computer
uses ones, zeros, and superpositions of ones and zeros, which enables quantum
computers to perform a vast number of calculations simultaneously compared to
classical computers. In a cloud-supported cyber-physical system environment,
running a machine learning application in quantum computers is often difficult,
due to the existing limitations of the current quantum devices. However, with
the combination of quantum-classical neural networks (NN), complex and
high-dimensional features can be extracted by the classical NN to a reduced but
more informative feature space to be processed by the existing quantum
computers. In this study, we develop a hybrid quantum-classical NN to detect an
amplitude shift cyber-attack on an in-vehicle control area network (CAN)
dataset. We show that using the hybrid quantum classical NN, it is possible to
achieve an attack detection accuracy of 94%, which is higher than a Long
short-term memory (LSTM) NN (87%) or quantum NN alone (62%)

    

### [[2110.07470] Universally Rank Consistent Ordinal Regression in Neural Networks](http://arxiv.org/abs/2110.07470)


  Despite the pervasiveness of ordinal labels in supervised learning, it
remains common practice in deep learning to treat such problems as categorical
classification using the categorical cross entropy loss. Recent methods
attempting to address this issue while respecting the ordinal structure of the
labels have resorted to converting ordinal regression into a series of extended
binary classification subtasks. However, the adoption of such methods remains
inconsistent due to theoretical and practical limitations. Here we address
these limitations by demonstrating that the subtask probabilities form a Markov
chain. We show how to straightforwardly modify neural network architectures to
exploit this fact and thereby constrain predictions to be universally rank
consistent. We furthermore prove that all rank consistent solutions can be
represented within this formulation. Using diverse benchmarks and the
real-world application of a specialized recurrent neural network for COVID-19
prognosis, we demonstrate the practical superiority of this method versus the
current state-of-the-art. The method is open sourced as user-friendly PyTorch
and TensorFlow packages.

    

### [[2110.07471] Stability Analysis of Unfolded WMMSE for Power Allocation](http://arxiv.org/abs/2110.07471)


  Power allocation is one of the fundamental problems in wireless networks and
a wide variety of algorithms address this problem from different perspectives.
A common element among these algorithms is that they rely on an estimation of
the channel state, which may be inaccurate on account of hardware defects,
noisy feedback systems, and environmental and adversarial disturbances.
Therefore, it is essential that the output power allocation of these algorithms
is stable with respect to input perturbations, to the extent that the
variations in the output are bounded for bounded variations in the input. In
this paper, we focus on UWMMSE -- a modern algorithm leveraging graph neural
networks --, and illustrate its stability to additive input perturbations of
bounded energy through both theoretical analysis and empirical validation.

    

### [[2110.07472] Capacity of Group-invariant Linear Readouts from Equivariant Representations: How Many Objects can be Linearly Classified Under All Possible Views?](http://arxiv.org/abs/2110.07472)


  Equivariance has emerged as a desirable property of representations of
objects subject to identity-preserving transformations that constitute a group,
such as translations and rotations. However, the expressivity of a
representation constrained by group equivariance is still not fully understood.
We address this gap by providing a generalization of Cover's Function Counting
Theorem that quantifies the number of linearly separable and group-invariant
binary dichotomies that can be assigned to equivariant representations of
objects. We find that the fraction of separable dichotomies is determined by
the dimension of the space that is fixed by the group action. We show how this
relation extends to operations such as convolutions, element-wise
nonlinearities, and global and local pooling. While other operations do not
change the fraction of separable dichotomies, local pooling decreases the
fraction, despite being a highly nonlinear operation. Finally, we test our
theory on intermediate representations of randomly initialized and fully
trained convolutional neural networks and find perfect agreement.

    

### [[2110.07478] Inferring Manifolds From Noisy Data Using Gaussian Processes](http://arxiv.org/abs/2110.07478)


  In analyzing complex datasets, it is often of interest to infer lower
dimensional structure underlying the higher dimensional observations. As a
flexible class of nonlinear structures, it is common to focus on Riemannian
manifolds. Most existing manifold learning algorithms replace the original data
with lower dimensional coordinates without providing an estimate of the
manifold in the observation space or using the manifold to denoise the original
data. This article proposes a new methodology for addressing these problems,
allowing interpolation of the estimated manifold between fitted data points.
The proposed approach is motivated by novel theoretical properties of local
covariance matrices constructed from noisy samples on a manifold. Our results
enable us to turn a global manifold reconstruction problem into a local
regression problem, allowing application of Gaussian processes for
probabilistic manifold reconstruction. In addition to theory justifying the
algorithm, we provide simulated and real data examples to illustrate the
performance.

    

### [[2110.07479] VABO: Violation-Aware Bayesian Optimization for Closed-Loop Control Performance Optimization with Unmodeled Constraints](http://arxiv.org/abs/2110.07479)


  We study the problem of performance optimization of closed-loop control
systems with unmodeled dynamics. Bayesian optimization (BO) has been
demonstrated effective for improving closed-loop performance by automatically
tuning controller gains or reference setpoints in a model-free manner. However,
BO methods have rarely been tested on dynamical systems with unmodeled
constraints. In this paper, we propose a violation-aware BO algorithm (VABO)
that optimizes closed-loop performance while simultaneously learning
constraint-feasible solutions. Unlike classical constrained BO methods which
allow an unlimited constraint violations, or safe BO algorithms that are
conservative and try to operate with near-zero violations, we allow budgeted
constraint violations to improve constraint learning and accelerate
optimization. We demonstrate the effectiveness of our proposed VABO method for
energy minimization of industrial vapor compression systems.

    

### [[2110.07510] Omni-Training for Data-Efficient Deep Learning](http://arxiv.org/abs/2110.07510)


  Learning a generalizable deep model from a few examples in a short time
remains a major challenge of machine learning, which has impeded its wide
deployment to many scenarios. Recent advances reveal that a properly
pre-trained model endows an important property: transferability. A higher
transferability of the learned representations indicates a better
generalizability across domains of different distributions (domain
transferability), or across tasks of different semantics (task
transferability). Transferability has become the key to enable data-efficient
deep learning, however, existing pre-training methods focus only on the domain
transferability while meta-training methods only on the task transferability.
This restricts their data-efficiency in downstream scenarios of diverging
domains and tasks. A finding of this paper is that even a tight combination of
pre-training and meta-training cannot achieve both kinds of transferability.
This motivates the proposed Omni-Training framework towards data-efficient deep
learning. Our first contribution is Omni-Net, a tri-flow architecture. Besides
the joint representation flow, Omni-Net introduces two new parallel flows for
pre-training and meta-training, respectively responsible for learning
representations of domain transferability and task transferability. Omni-Net
coordinates the parallel flows by routing them via the joint-flow, making each
gain the other kind of transferability. Our second contribution is Omni-Loss,
in which a mean-teacher regularization is imposed to learn generalizable and
stabilized representations. Omni-Training is a general framework that
accommodates many existing pre-training and meta-training algorithms. A
thorough evaluation on cross-task and cross-domain datasets in classification,
regression and reinforcement learning problems shows that Omni-Training
consistently outperforms the state-of-the-art methods.

    

### [[2110.07521] Multi-objective Clustering: A Data-driven Analysis of MOCLE, MOCK and $Δ$-MOCK](http://arxiv.org/abs/2110.07521)


  We present a data-driven analysis of MOCK, $\Delta$-MOCK, and MOCLE. These
are three closely related approaches that use multi-objective optimization for
crisp clustering. More specifically, based on a collection of 12 datasets
presenting different proprieties, we investigate the performance of MOCLE and
MOCK compared to the recently proposed $\Delta$-MOCK. Besides performing a
quantitative analysis identifying which method presents a good/poor performance
with respect to another, we also conduct a more detailed analysis on why such a
behavior happened. Indeed, the results of our analysis provide useful insights
into the strengths and weaknesses of the methods investigated.

    

### [[2110.07531] Predictive models of RNA degradation through dual crowdsourcing](http://arxiv.org/abs/2110.07531)


  Messenger RNA-based medicines hold immense potential, as evidenced by their
rapid deployment as COVID-19 vaccines. However, worldwide distribution of mRNA
molecules has been limited by their thermostability, which is fundamentally
limited by the intrinsic instability of RNA molecules to a chemical degradation
reaction called in-line hydrolysis. Predicting the degradation of an RNA
molecule is a key task in designing more stable RNA-based therapeutics. Here,
we describe a crowdsourced machine learning competition ("Stanford
OpenVaccine") on Kaggle, involving single-nucleotide resolution measurements on
6043 102-130-nucleotide diverse RNA constructs that were themselves solicited
through crowdsourcing on the RNA design platform Eterna. The entire experiment
was completed in less than 6 months. Winning models demonstrated test set
errors that were better by 50% than the previous state-of-the-art DegScore
model. Furthermore, these models generalized to blindly predicting orthogonal
degradation data on much longer mRNA molecules (504-1588 nucleotides) with
improved accuracy over DegScore and other models. Top teams integrated natural
language processing architectures and data augmentation techniques with
predictions from previous dynamic programming models for RNA secondary
structure. These results indicate that such models are capable of representing
in-line hydrolysis with excellent accuracy, supporting their use for designing
stabilized messenger RNAs. The integration of two crowdsourcing platforms, one
for data set creation and another for machine learning, may be fruitful for
other urgent problems that demand scientific discovery on rapid timescales.

    

### [[2110.07537] Toward Degradation-Robust Voice Conversion](http://arxiv.org/abs/2110.07537)


  Any-to-any voice conversion technologies convert the vocal timbre of an
utterance to any speaker even unseen during training. Although there have been
several state-of-the-art any-to-any voice conversion models, they were all
based on clean utterances to convert successfully. However, in real-world
scenarios, it is difficult to collect clean utterances of a speaker, and they
are usually degraded by noises or reverberations. It thus becomes highly
desired to understand how these degradations affect voice conversion and build
a degradation-robust model. We report in this paper the first comprehensive
study on the degradation robustness of any-to-any voice conversion. We show
that the performance of state-of-the-art models nowadays was severely hampered
given degraded utterances. To this end, we then propose speech enhancement
concatenation and denoising training to improve the robustness. In addition to
common degradations, we also consider adversarial noises, which alter the model
output significantly yet are human-imperceptible. It was shown that both
concatenations with off-the-shelf speech enhancement models and denoising
training on voice conversion models could improve the robustness, while each of
them had pros and cons.

    

### [[2110.07549] Time Series Clustering for Human Behavior Pattern Mining](http://arxiv.org/abs/2110.07549)


  Human behavior modeling deals with learning and understanding of behavior
patterns inherent in humans' daily routines. Existing pattern mining techniques
either assume human dynamics is strictly periodic, or require the number of
modes as input, or do not consider uncertainty in the sensor data. To handle
these issues, in this paper, we propose a novel clustering approach for
modeling human behavior (named, MTpattern) from time-series data. For mining
frequent human behavior patterns effectively, we utilize a three-stage
pipeline: (1) represent time series data into sequence of regularly sampled
equal-sized unit time intervals for better analysis, (2) a new distance measure
scheme is proposed to cluster similar sequences which can handle temporal
variation and uncertainty in the data, and (3) exploit an exemplar-based
clustering mechanism and fine-tune its parameters to output minimum number of
clusters with given permissible distance constraints and without knowing the
number of modes present in the data. Then, the average of all sequences in a
cluster is considered as a human behavior pattern. Empirical studies on two
real-world datasets and a simulated dataset demonstrate the effectiveness of
MTpattern this http URL internal and external measures of clustering.

    

### [[2110.07554] Looper: An end-to-end ML platform for product decisions](http://arxiv.org/abs/2110.07554)


  Modern software systems and products increasingly rely on machine learning
models to make data-driven decisions based on interactions with users and
systems, e.g., compute infrastructure. For broader adoption, this practice must
(i) accommodate software engineers without ML backgrounds, and (ii) provide
mechanisms to optimize for product goals. In this work, we describe general
principles and a specific end-to-end ML platform, Looper, which offers
easy-to-use APIs for decision-making and feedback collection. Looper supports
the full end-to-end ML lifecycle from online data collection to model training,
deployment, inference, and extends support to evaluation and tuning against
product goals. We outline the platform architecture and overall impact of
production deployment. We also describe the learning curve and summarize
experiences from platform adopters.

    

### [[2110.07557] AIR-Net: Adaptive and Implicit Regularization Neural Network for Matrix Completion](http://arxiv.org/abs/2110.07557)


  Conventionally, the matrix completion (MC) model aims to recover a matrix
from partially observed elements. Accurate recovery necessarily requires a
regularization encoding priors of the unknown matrix/signal properly. However,
encoding the priors accurately for the complex natural signal is difficult, and
even then, the model might not generalize well outside the particular matrix
type. This work combines adaptive and implicit low-rank regularization that
captures the prior dynamically according to the current recovered matrix.
Furthermore, we aim to answer the question: how does adaptive regularization
affect implicit regularization? We utilize neural networks to represent
Adaptive and Implicit Regularization and named the proposed model
\textit{AIR-Net}. Theoretical analyses show that the adaptive part of the
AIR-Net enhances implicit regularization. In addition, the adaptive regularizer
vanishes at the end, thus can avoid saturation issues. Numerical experiments
for various data demonstrate the effectiveness of AIR-Net, especially when the
locations of missing elements are not randomly chosen. With complete
flexibility to select neural networks for matrix representation, AIR-Net can be
extended to solve more general inverse problems.

    

### [[2110.07566] Practical Benefits of Feature Feedback Under Distribution Shift](http://arxiv.org/abs/2110.07566)


  In attempts to develop sample-efficient algorithms, researcher have explored
myriad mechanisms for collecting and exploiting feature feedback, auxiliary
annotations provided for training (but not test) instances that highlight
salient evidence. Examples include bounding boxes around objects and salient
spans in text. Despite its intuitive appeal, feature feedback has not delivered
significant gains in practical problems as assessed on iid holdout sets.
However, recent works on counterfactually augmented data suggest an alternative
benefit of supplemental annotations: lessening sensitivity to spurious patterns
and consequently delivering gains in out-of-domain evaluations. Inspired by
these findings, we hypothesize that while the numerous existing methods for
incorporating feature feedback have delivered negligible in-sample gains, they
may nevertheless generalize better out-of-domain. In experiments addressing
sentiment analysis, we show that feature feedback methods perform significantly
better on various natural out-of-domain datasets even absent differences on
in-domain evaluation. By contrast, on natural language inference tasks,
performance remains comparable. Finally, we compare those tasks where feature
feedback does (and does not) help.

    

### [[2110.07570] sMGC: A Complex-Valued Graph Convolutional Network via Magnetic Laplacian for Directed Graphs](http://arxiv.org/abs/2110.07570)


  Recent advancements in Graph Neural Networks have led to state-of-the-art
performance on representation learning of graphs for node classification.
However, the majority of existing works process directed graphs by
symmetrization, which may cause loss of directional information. In this paper,
we propose the magnetic Laplacian that preserves edge directionality by
encoding it into complex phase as a deformation of the combinatorial Laplacian.
In addition, we design an Auto-Regressive Moving-Average (ARMA) filter that is
capable of learning global features from graphs. To reduce time complexity,
Taylor expansion is applied to approximate the filter. We derive complex-valued
operations in graph neural network and devise a simplified Magnetic Graph
Convolution network, namely sMGC. Our experiment results demonstrate that sMGC
is a fast, powerful, and widely applicable GNN.

    

### [[2110.07577] UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning](http://arxiv.org/abs/2110.07577)


  Conventional fine-tuning of pre-trained language models tunes all model
parameters and stores a full model copy for each downstream task, which has
become increasingly infeasible as the model size grows larger. Recent
parameter-efficient language model tuning (PELT) methods manage to match the
performance of fine-tuning with much fewer trainable parameters and perform
especially well when the training data is limited. However, different PELT
methods may perform rather differently on the same task, making it nontrivial
to select the most appropriate method for a specific task, especially
considering the fast-growing number of new PELT methods and downstream tasks.
In light of model diversity and the difficulty of model selection, we propose a
unified framework, UniPELT, which incorporates different PELT methods as
submodules and learns to activate the ones that best suit the current data or
task setup. Remarkably, on the GLUE benchmark, UniPELT consistently achieves
1~3pt gains compared to the best individual PELT method that it incorporates
and even outperforms fine-tuning under different setups. Moreover, UniPELT
often surpasses the upper bound when taking the best performance of all its
submodules used individually on each task, indicating that a mixture of
multiple PELT methods may be inherently more effective than single methods.

    

### [[2110.07579] Diffusion Normalizing Flow](http://arxiv.org/abs/2110.07579)


  We present a novel generative modeling method called diffusion normalizing
flow based on stochastic differential equations (SDEs). The algorithm consists
of two neural SDEs: a forward SDE that gradually adds noise to the data to
transform the data into Gaussian random noise, and a backward SDE that
gradually removes the noise to sample from the data distribution. By jointly
training the two neural SDEs to minimize a common cost function that quantifies
the difference between the two, the backward SDE converges to a diffusion
process the starts with a Gaussian distribution and ends with the desired data
distribution. Our method is closely related to normalizing flow and diffusion
probabilistic models and can be viewed as a combination of the two. Compared
with normalizing flow, diffusion normalizing flow is able to learn
distributions with sharp boundaries. Compared with diffusion probabilistic
models, diffusion normalizing flow requires fewer discretization steps and thus
has better sampling efficiency. Our algorithm demonstrates competitive
performance in both high-dimension data density estimation and image generation
tasks.

    

### [[2110.07580] Graph Condensation for Graph Neural Networks](http://arxiv.org/abs/2110.07580)


  Given the prevalence of large-scale graphs in real-world applications, the
storage and time for training neural models have raised increasing concerns. To
alleviate the concerns, we propose and study the problem of graph condensation
for graph neural networks (GNNs). Specifically, we aim to condense the large,
original graph into a small, synthetic and highly-informative graph, such that
GNNs trained on the small graph and large graph have comparable performance. We
approach the condensation problem by imitating the GNN training trajectory on
the original graph through the optimization of a gradient matching loss and
design a strategy to condense node futures and structural information
simultaneously. Extensive experiments have demonstrated the effectiveness of
the proposed framework in condensing different graph datasets into informative
smaller graphs. In particular, we are able to approximate the original test
accuracy by 95.3% on Reddit, 99.8% on Flickr and 99.0% on Citeseer, while
reducing their graph size by more than 99.9%, and the condensed graphs can be
used to train various GNN architectures.

    

### [[2110.07581] Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations](http://arxiv.org/abs/2110.07581)


  Dense retrieval (DR) methods conduct text retrieval by first encoding texts
in the embedding space and then matching them by nearest neighbor search. This
requires strong locality properties from the representation space, i.e, the
close allocations of each small group of relevant texts, which are hard to
generalize to domains without sufficient training data. In this paper, we aim
to improve the generalization ability of DR models from source training domains
with rich supervision signals to target domains without any relevant labels, in
the zero-shot setting. To achieve that, we propose Momentum adversarial Domain
Invariant Representation learning (MoDIR), which introduces a momentum method
in the DR training process to train a domain classifier distinguishing source
versus target, and then adversarially updates the DR encoder to learn domain
invariant representations. Our experiments show that MoDIR robustly outperforms
its baselines on 10+ ranking datasets from the BEIR benchmark in the zero-shot
setup, with more than 10% relative gains on datasets with enough sensitivity
for DR models' evaluation. Source code of this paper will be released.

    

### [[2110.07582] Network Representation Learning: From Preprocessing, Feature Extraction to Node Embedding](http://arxiv.org/abs/2110.07582)


  Network representation learning (NRL) advances the conventional graph mining
of social networks, knowledge graphs, and complex biomedical and physics
information networks. Over dozens of network representation learning algorithms
have been reported in the literature. Most of them focus on learning node
embeddings for homogeneous networks, but they differ in the specific encoding
schemes and specific types of node semantics captured and used for learning
node embedding. This survey paper reviews the design principles and the
different node embedding techniques for network representation learning over
homogeneous networks. To facilitate the comparison of different node embedding
algorithms, we introduce a unified reference framework to divide and generalize
the node embedding learning process on a given network into preprocessing
steps, node feature extraction steps and node embedding model training for a
NRL task such as link prediction and node clustering. With this unifying
reference framework, we highlight the representative methods, models, and
techniques used at different stages of the node embedding model learning
process. This survey not only helps researchers and practitioners to gain an
in-depth understanding of different network representation learning techniques
but also provides practical guidelines for designing and developing the next
generation of network representation learning algorithms and systems.

    

### [[2110.07583] Near optimal sample complexity for matrix and tensor normal models via geodesic convexity](http://arxiv.org/abs/2110.07583)


  The matrix normal model, the family of Gaussian matrix-variate distributions
whose covariance matrix is the Kronecker product of two lower dimensional
factors, is frequently used to model matrix-variate data. The tensor normal
model generalizes this family to Kronecker products of three or more factors.
We study the estimation of the Kronecker factors of the covariance matrix in
the matrix and tensor models. We show nonasymptotic bounds for the error
achieved by the maximum likelihood estimator (MLE) in several natural metrics.
In contrast to existing bounds, our results do not rely on the factors being
well-conditioned or sparse. For the matrix normal model, all our bounds are
minimax optimal up to logarithmic factors, and for the tensor normal model our
bound for the largest factor and overall covariance matrix are minimax optimal
up to constant factors provided there are enough samples for any estimator to
obtain constant Frobenius error. In the same regimes as our sample complexity
bounds, we show that an iterative procedure to compute the MLE known as the
flip-flop algorithm converges linearly with high probability. Our main tool is
geodesic strong convexity in the geometry on positive-definite matrices induced
by the Fisher information metric. This strong convexity is determined by the
expansion of certain random quantum channels. We also provide numerical
evidence that combining the flip-flop algorithm with a simple shrinkage
estimator can improve performance in the undersampled regime.

    

### [[2110.07584] Unsupervised Learning of Full-Waveform Inversion: Connecting CNN and Partial Differential Equation in a Loop](http://arxiv.org/abs/2110.07584)


  This paper investigates unsupervised learning of Full-Waveform Inversion
(FWI), which has been widely used in geophysics to estimate subsurface velocity
maps from seismic data. This problem is mathematically formulated by a second
order partial differential equation (PDE), but is hard to solve. Moreover,
acquiring velocity map is extremely expensive, making it impractical to scale
up a supervised approach to train the mapping from seismic data to velocity
maps with convolutional neural networks (CNN). We address these difficulties by
integrating PDE and CNN in a loop, thus shifting the paradigm to unsupervised
learning that only requires seismic data. In particular, we use finite
difference to approximate the forward modeling of PDE as a differentiable
operator (from velocity map to seismic data) and model its inversion by CNN
(from seismic data to velocity map). Hence, we transform the supervised
inversion task into an unsupervised seismic data reconstruction task. We also
introduce a new large-scale dataset OpenFWI, to establish a more challenging
benchmark for the community. Experiment results show that our model (using
seismic data alone) yields comparable accuracy to the supervised counterpart
(using both seismic data and velocity map). Furthermore, it outperforms the
supervised model when involving more seismic data.

    

### [[2110.07594] The Neural MMO Platform for Massively Multiagent Research](http://arxiv.org/abs/2110.07594)


  Neural MMO is a computationally accessible research platform that combines
large agent populations, long time horizons, open-ended tasks, and modular game
systems. Existing environments feature subsets of these properties, but Neural
MMO is the first to combine them all. We present Neural MMO as free and open
source software with active support, ongoing development, documentation, and
additional training, logging, and visualization tools to help users adapt to
this new setting. Initial baselines on the platform demonstrate that agents
trained in large populations explore more and learn a progression of skills. We
raise other more difficult problems such as many-team cooperation as open
research questions which Neural MMO is well-suited to answer. Finally, we
discuss current limitations of the platform, potential mitigations, and plans
for continued development.

    

### [[2110.07595] Compressibility of Distributed Document Representations](http://arxiv.org/abs/2110.07595)


  Contemporary natural language processing (NLP) revolves around learning from
latent document representations, generated either implicitly by neural language
models or explicitly by methods such as doc2vec or similar. One of the key
properties of the obtained representations is their dimension. Whilst the
commonly adopted dimensions of 256 and 768 offer sufficient performance on many
tasks, it is many times unclear whether the default dimension is the most
suitable choice for the subsequent downstream learning tasks. Furthermore,
representation dimensions are seldom subject to hyperparameter tuning due to
computational constraints. The purpose of this paper is to demonstrate that a
surprisingly simple and efficient recursive compression procedure can be
sufficient to both significantly compress the initial representation, but also
potentially improve its performance when considering the task of text
classification. Having smaller and less noisy representations is the desired
property during deployment, as orders of magnitude smaller models can
significantly reduce the computational overload and with it the deployment
costs. We propose CoRe, a straightforward, representation learner-agnostic
framework suitable for representation compression. The CoRe's performance is
showcased and studied on a collection of 17 real-life corpora from biomedical,
news, social media, and literary domains. We explored CoRe's behavior when
considering contextual and non-contextual document representations, different
compression levels, and 9 different compression algorithms. Current results
based on more than 100,000 compression experiments indicate that recursive
Singular Value Decomposition offers a very good trade-off between the
compression efficiency and performance, making CoRe useful in many existing,
representation-dependent NLP pipelines.

    

### [[2110.07604] NeRS: Neural Reflectance Surfaces for Sparse-view 3D Reconstruction in the Wild](http://arxiv.org/abs/2110.07604)


  Recent history has seen a tremendous growth of work exploring implicit
representations of geometry and radiance, popularized through Neural Radiance
Fields (NeRF). Such works are fundamentally based on a (implicit) {\em
volumetric} representation of occupancy, allowing them to model diverse scene
structure including translucent objects and atmospheric obscurants. But because
the vast majority of real-world scenes are composed of well-defined surfaces,
we introduce a {\em surface} analog of such implicit models called Neural
Reflectance Surfaces (NeRS). NeRS learns a neural shape representation of a
closed surface that is diffeomorphic to a sphere, guaranteeing water-tight
reconstructions. Even more importantly, surface parameterizations allow NeRS to
learn (neural) bidirectional surface reflectance functions (BRDFs) that
factorize view-dependent appearance into environmental illumination, diffuse
color (albedo), and specular "shininess." Finally, rather than illustrating our
results on synthetic scenes or controlled in-the-lab capture, we assemble a
novel dataset of multi-view images from online marketplaces for selling goods.
Such "in-the-wild" multi-view image sets pose a number of challenges, including
a small number of views with unknown/rough camera estimates. We demonstrate
that surface-based neural reconstructions enable learning from such data,
outperforming volumetric neural rendering-based reconstructions. We hope that
NeRS serves as a first step toward building scalable, high-quality libraries of
real-world shape, materials, and illumination. The project page with code and
video visualizations can be found at
this https URL}{this http URL.

    

### [[1801.08640] Considerations When Learning Additive Explanations for Black-Box Models](http://arxiv.org/abs/1801.08640)


  Many methods to explain black-box models, whether local or global, are
additive. In this paper, we study global additive explanations for non-additive
models, focusing on four explanation methods: partial dependence, Shapley
explanations adapted to a global setting, distilled additive explanations, and
gradient-based explanations. We show that different explanation methods
characterize non-additive components in a black-box model's prediction function
in different ways. We use the concepts of main and total effects to anchor
additive explanations, and quantitatively evaluate additive and non-additive
explanations. Even though distilled explanations are generally the most
accurate additive explanations, non-additive explanations such as tree
explanations that explicitly model non-additive components tend to be even more
accurate. Despite this, our user study showed that machine learning
practitioners were better able to leverage additive explanations for various
tasks. These considerations should be taken into account when considering which
explanation to trust and use to explain black-box models.

    

### [[1811.12471] Unlabeled Compression Schemes Exceeding the VC-dimension](http://arxiv.org/abs/1811.12471)


  In this note we disprove a conjecture of Kuzmin and Warmuth claiming that
every family whose VC-dimension is at most d admits an unlabeled compression
scheme to a sample of size at most d. We also study the unlabeled compression
schemes of the joins of some families and conjecture that these give a larger
gap between the VC-dimension and the size of the smallest unlabeled compression
scheme for them.

    

### [[1902.03667] Differential Similarity in Higher Dimensional Spaces: Theory and Applications](http://arxiv.org/abs/1902.03667)


  This paper presents an extension and an elaboration of the theory of
differential similarity, which was originally proposed in arXiv:1401.2411
[cs.LG]. The goal is to develop an algorithm for clustering and coding that
combines a geometric model with a probabilistic model in a principled way. For
simplicity, the geometric model in the earlier paper was restricted to the
three-dimensional case. The present paper removes this restriction, and
considers the full $n$-dimensional case. Although the mathematical model is the
same, the strategies for computing solutions in the $n$-dimensional case are
different, and one of the main purposes of this paper is to develop and analyze
these strategies. Another main purpose is to devise techniques for estimating
the parameters of the model from sample data, again in $n$ dimensions. We
evaluate the solution strategies and the estimation techniques by applying them
to two familiar real-world examples: the classical MNIST dataset and the
CIFAR-10 dataset.

    

### [[1909.09902] Deep Reinforcement Learning with Modulated Hebbian plus Q Network Architecture](http://arxiv.org/abs/1909.09902)


  This paper presents a new neural architecture that combines a modulated
Hebbian network (MOHN) with DQN, which we call modulated Hebbian plus Q network
architecture (MOHQA). The hypothesis is that such a combination allows MOHQA to
solve difficult partially observable Markov decision process (POMDP) problems
which impair temporal difference (TD)-based RL algorithms such as DQN, as the
TD error cannot be easily derived from observations. The key idea is to use a
Hebbian network with bio-inspired neural traces in order to bridge temporal
delays between actions and rewards when confounding observations and sparse
rewards result in inaccurate TD errors. In MOHQA, DQN learns low level features
and control, while the MOHN contributes to the high-level decisions by
associating rewards with past states and actions. Thus the proposed
architecture combines two modules with significantly different learning
algorithms, a Hebbian associative network and a classical DQN pipeline,
exploiting the advantages of both. Simulations on a set of POMDPs and on the
MALMO environment show that the proposed algorithm improved DQN's results and
even outperformed control tests with A2C, QRDQN+LSTM and REINFORCE algorithms
on some POMDPs with confounding stimuli and sparse rewards.

    

### [[2002.03016] Safe Wasserstein Constrained Deep Q-Learning](http://arxiv.org/abs/2002.03016)


  This paper presents a distributionally robust Q-Learning algorithm (DrQ)
which leverages Wasserstein ambiguity sets to provide probabilistic
out-of-sample safety guarantees during online learning. First, we follow past
work by separating the constraint functions from the principal objective to
create a hierarchy of machines which estimate the feasible state-action space
within the constrained Markov decision process (CMDP). DrQ works within this
framework by augmenting constraint costs with tightening offset variables
obtained through Wasserstein distributionally robust optimization (DRO). These
offset variables correspond to worst-case distributions of modeling error
characterized by the TD-errors of the constraint Q-functions. This procedure
allows us to safely approach the nominal constraint boundaries with strong
probabilistic safety guarantees. Using a case study of safe lithium-ion battery
fast charging, we demonstrate dramatic improvements in safety and performance
relative to conventional methods.

    

### [[2003.02437] Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning](http://arxiv.org/abs/2003.02437)


  Drone-based vehicle detection aims at finding the vehicle locations and
categories in an aerial image. It empowers smart city traffic management and
disaster rescue. Researchers have made mount of efforts in this area and
achieved considerable progress. Nevertheless, it is still a challenge when the
objects are hard to distinguish, especially in low light conditions. To tackle
this problem, we construct a large-scale drone-based RGB-Infrared vehicle
detection dataset, termed DroneVehicle. Our DroneVehicle collects 28, 439
RGB-Infrared image pairs, covering urban roads, residential areas, parking
lots, and other scenarios from day to night. Due to the great gap between RGB
and infrared images, cross-modal images provide both effective information and
redundant information. To address this dilemma, we further propose an
uncertainty-aware cross-modality vehicle detection (UA-CMDet) framework to
extract complementary information from cross-modal images, which can
significantly improve the detection performance in low light conditions. An
uncertainty-aware module (UAM) is designed to quantify the uncertainty weights
of each modality, which is calculated by the cross-modal Intersection over
Union (IoU) and the RGB illumination value. Furthermore, we design an
illumination-aware cross-modal non-maximum suppression algorithm to better
integrate the modal-specific information in the inference phase. Extensive
experiments on the DroneVehicle dataset demonstrate the flexibility and
effectiveness of the proposed method for crossmodality vehicle detection. The
dataset can be download from this https URL.

    

### [[2005.02979] A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems](http://arxiv.org/abs/2005.02979)


  Autonomous cyber-physical systems (CPS) can improve safety and efficiency for
safety-critical applications, but require rigorous testing before deployment.
The complexity of these systems often precludes the use of formal verification
and real-world testing can be too dangerous during development. Therefore,
simulation-based techniques have been developed that treat the system under
test as a black box operating in a simulated environment. Safety validation
tasks include finding disturbances in the environment that cause the system to
fail (falsification), finding the most-likely failure, and estimating the
probability that the system fails. Motivated by the prevalence of
safety-critical artificial intelligence, this work provides a survey of
state-of-the-art safety validation techniques for CPS with a focus on applied
algorithms and their modifications for the safety validation problem. We
present and discuss algorithms in the domains of optimization, path planning,
reinforcement learning, and importance sampling. Problem decomposition
techniques are presented to help scale algorithms to large state spaces, which
are common for CPS. A brief overview of safety-critical applications is given,
including autonomous vehicles and aircraft collision avoidance systems.
Finally, we present a survey of existing academic and commercially available
safety validation tools.

    

### [[2005.08054] Classification vs regression in overparameterized regimes: Does the loss function matter?](http://arxiv.org/abs/2005.08054)


  We compare classification and regression tasks in an overparameterized linear
model with Gaussian features. On the one hand, we show that with sufficient
overparameterization all training points are support vectors: solutions
obtained by least-squares minimum-norm interpolation, typically used for
regression, are identical to those produced by the hard-margin support vector
machine (SVM) that minimizes the hinge loss, typically used for training
classifiers. On the other hand, we show that there exist regimes where these
interpolating solutions generalize well when evaluated by the 0-1 test loss
function, but do not generalize if evaluated by the square loss function, i.e.
they approach the null risk. Our results demonstrate the very different roles
and properties of loss functions used at the training phase (optimization) and
the testing phase (generalization).

    

### [[2006.09365] Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](http://arxiv.org/abs/2006.09365)


  In Byzantine robust distributed or federated learning, a central server wants
to train a machine learning model over data distributed across multiple
workers. However, a fraction of these workers may deviate from the prescribed
algorithm and send arbitrary messages. While this problem has received
significant attention recently, most current defenses assume that the workers
have identical data. For realistic cases when the data across workers are
heterogeneous (non-iid), we design new attacks which circumvent current
defenses, leading to significant loss of performance. We then propose a simple
bucketing scheme that adapts existing robust algorithms to heterogeneous
datasets at a negligible computational cost. We also theoretically and
experimentally validate our approach, showing that combining bucketing with
existing robust algorithms is effective against challenging attacks. Our work
is the first to establish guaranteed convergence for the non-iid Byzantine
robust problem under realistic assumptions.

    

### [[2008.01169] Deep Knowledge Tracing with Learning Curves](http://arxiv.org/abs/2008.01169)


  Knowledge tracing (KT) has recently been an active research area of
computational pedagogy. The task is to model students' mastery level of
knowledge concepts based on their responses to the questions in the past, as
well as predict the probabilities that they correctly answer subsequent
questions in the future. KT tasks were historically solved using statistical
modeling methods such as Bayesian inference and factor analysis, but recent
advances in deep learning have led to the successive proposals that leverage
deep neural networks, including long short-term memory networks,
memory-augmented networks and self-attention networks. While those deep models
demonstrate superior performance over the traditional approaches, they all
neglect the explicit modeling of the learning curve theory, which generally
says that more practice on the same knowledge concept enhances one's mastery
level of the concept. Based on this theory, we propose a Convolution-Augmented
Knowledge Tracing (CAKT) model in this paper. The model employs
three-dimensional convolutional neural networks to explicitly learn a student's
recent experience on applying the same knowledge concept with that in the next
question, and fuses the learnt feature with the feature representing her
overall latent knowledge state obtained using a classic LSTM network. The fused
feature is then fed into a second LSTM network to predict the student's
response to the next question. Experimental results show that CAKT achieves the
new state-of-the-art performance in predicting students' responses compared
with existing models. We also conduct extensive sensitivity analysis and
ablation study to show the stability of the results and justify the particular
architecture of CAKT, respectively.

    

### [[2008.09872] LT4REC:A Lottery Ticket Hypothesis Based Multi-task Practice for Video Recommendation System](http://arxiv.org/abs/2008.09872)


  Click-through rate prediction (CTR) and post-click conversion rate prediction
(CVR) play key roles across all industrial ranking systems, such as
recommendation systems, online advertising, and search engines. Different from
the extensive research on CTR, there is much less research on CVR estimation,
whose main challenge is extreme data sparsity with one or two orders of
magnitude reduction in the number of samples than CTR. People try to solve this
problem with the paradigm of multi-task learning with the sufficient samples of
CTR, but the typical hard sharing method can't effectively solve this problem,
because it is difficult to analyze which parts of network components can be
shared and which parts are in conflict, i.e., there is a large inaccuracy with
artificially designed neurons sharing. In this paper, we model CVR in a
brand-new method by adopting the lottery-ticket-hypothesis-based sparse sharing
multi-task learning, which can automatically and flexibly learn which neuron
weights to be shared without artificial experience. Experiments on the dataset
gathered from traffic logs of Tencent video's recommendation system demonstrate
that sparse sharing in the CVR model significantly outperforms competitive
methods. Due to the nature of weight sparsity in sparse sharing, it can also
significantly reduce computational complexity and memory usage which are very
important in the industrial recommendation system.

    

### [[2010.01089] Unsupervised Point Cloud Pre-Training via Occlusion Completion](http://arxiv.org/abs/2010.01089)


  We describe a simple pre-training approach for point clouds. It works in
three steps: 1. Mask all points occluded in a camera view; 2. Learn an
encoder-decoder model to reconstruct the occluded points; 3. Use the encoder
weights as initialisation for downstream point cloud tasks. We find that even
when we construct a single pre-training dataset (from ModelNet40), this
pre-training method improves accuracy across different datasets and encoders,
on a wide range of downstream tasks. Specifically, we show that our method
outperforms previous pre-training methods in object classification, and both
part-based and semantic segmentation tasks. We study the pre-trained features
and find that they lead to wide downstream minima, have high transformation
invariance, and have activations that are highly correlated with part labels.
Code and data are available at: this https URL


### [[2010.07027] A Light Heterogeneous Graph Collaborative Filtering Model using Textual Information](http://arxiv.org/abs/2010.07027)


  Due to the development of graph neural networks, graph-based representation
learning methods have made great progress in recommender systems. However, data
sparsity is still a challenging problem that most graph-based recommendation
methods are confronted with. Recent works try to address this problem by
utilizing side information. In this paper, we exploit the relevant and easily
accessible textual information by advanced natural language processing (NLP)
models and propose a light RGCN-based (RGCN, relational graph convolutional
network) collaborative filtering method based on heterogeneous graphs.
Specifically, to incorporate rich textual knowledge, we utilize a pre-trained
NLP model to initialize the embeddings of text nodes. Afterward, by performing
a simplified RGCN-based node information propagation on the constructed
heterogeneous graph, the embeddings of users and items can be adjusted with
textual knowledge, which effectively alleviates the negative effects of data
sparsity. Moreover, the matching function used by most graph-based
representation learning methods is the inner product, which is not appropriate
for the obtained embeddings that contain complex semantics. We design a
predictive network that combines graph-based representation learning with
neural matching function learning, and demonstrate that this architecture can
bring a significant performance improvement. Extensive experiments are
conducted on three publicly available datasets, and the results verify the
superior performance of our method over several baselines.

    

### [[2010.15690] Analyzing the tree-layer structure of Deep Forests](http://arxiv.org/abs/2010.15690)


  Random forests on the one hand, and neural networks on the other hand, have
met great success in the machine learning community for their predictive
performance. Combinations of both have been proposed in the literature, notably
leading to the so-called deep forests (DF) (Zhou \& Feng,2019). In this paper,
our aim is not to benchmark DF performances but to investigate instead their
underlying mechanisms. Additionally, DF architecture can be generally
simplified into more simple and computationally efficient shallow forest
networks. Despite some instability, the latter may outperform standard
predictive tree-based methods. We exhibit a theoretical framework in which a
shallow tree network is shown to enhance the performance of classical decision
trees. In such a setting, we provide tight theoretical lower and upper bounds
on its excess risk. These theoretical results show the interest of tree-network
architectures for well-structured data provided that the first layer, acting as
a data encoder, is rich enough.

    

### [[2102.06605] Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning](http://arxiv.org/abs/2102.06605)


  Contrastive self-supervised learning (CSL) has attracted increasing attention
for model pre-training via unlabeled data. The resulted CSL models provide
instance-discriminative visual features that are uniformly scattered in the
feature space. During deployment, the common practice is to directly fine-tune
CSL models with cross-entropy, which however may not be the best strategy in
practice. Although cross-entropy tends to separate inter-class features, the
resulting models still have limited capability for reducing intra-class feature
scattering that exists in CSL models. In this paper, we investigate whether
applying contrastive learning to fine-tuning would bring further benefits, and
analytically find that optimizing the contrastive loss benefits both
discriminative representation learning and model optimization during
fine-tuning. Inspired by these findings, we propose Contrast-regularized tuning
(Core-tuning), a new approach for fine-tuning CSL models. Instead of simply
adding the contrastive loss to the objective of fine-tuning, Core-tuning
further applies a novel hard pair mining strategy for more effective
contrastive fine-tuning, as well as smoothing the decision boundary to better
exploit the learned discriminative feature space. Extensive experiments on
image classification and semantic segmentation verify the effectiveness of
Core-tuning.

    

### [[2103.03413] Routing algorithms as tools for integrating social distancing with emergency evacuation](http://arxiv.org/abs/2103.03413)


  One of the lessons from the COVID-19 pandemic is the importance of social
distancing, even in challenging circumstances such as pre-hurricane evacuation.
To explore the implications of integrating social distancing with evacuation
operations, we describe this evacuation process as a Capacitated Vehicle
Routing Problem (CVRP) and solve it using a DNN (Deep Neural Network)-based
solution (Deep Reinforcement Learning) and a non-DNN solution (Sweep
Algorithm). A central question is whether Deep Reinforcement Learning provides
sufficient extra routing efficiency to accommodate increased social distancing
in a time-constrained evacuation operation. We found that, in comparison to the
Sweep Algorithm, Deep Reinforcement Learning can provide decision-makers with
more efficient routing. However, the evacuation time saved by Deep
Reinforcement Learning does not come close to compensating for the extra time
required for social distancing, and its advantage disappears as the emergency
vehicle capacity approaches the number of people per household.

    

### [[2103.08255] Sample-efficient Reinforcement Learning Representation Learning with Curiosity Contrastive Forward Dynamics Model](http://arxiv.org/abs/2103.08255)


  Developing an agent in reinforcement learning (RL) that is capable of
performing complex control tasks directly from high-dimensional observation
such as raw pixels is yet a challenge as efforts are made towards improving
sample efficiency and generalization. This paper considers a learning framework
for Curiosity Contrastive Forward Dynamics Model (CCFDM) in achieving a more
sample-efficient RL based directly on raw pixels. CCFDM incorporates a forward
dynamics model (FDM) and performs contrastive learning to train its deep
convolutional neural network-based image encoder (IE) to extract conducive
spatial and temporal information for achieving a more sample efficiency for RL.
In addition, during training, CCFDM provides intrinsic rewards, produced based
on FDM prediction error, encourages the curiosity of the RL agent to improve
exploration. The diverge and less-repetitive observations provide by both our
exploration strategy and data augmentation available in contrastive learning
improve not only the sample efficiency but also the generalization. Performance
of existing model-free RL methods such as Soft Actor-Critic built on top of
CCFDM outperforms prior state-of-the-art pixel-based RL methods on the DeepMind
Control Suite benchmark.

    

### [[2103.12591] BoXHED 2.0: Scalable boosting of dynamic survival analysis](http://arxiv.org/abs/2103.12591)


  Modern applications of survival analysis increasingly involve time-dependent
covariates. In healthcare settings, such covariates provide dynamic patient
histories that can be used to assess health risks in realtime by tracking the
hazard function. Hazard learning is thus particularly useful in healthcare
analytics, and the open-source package BoXHED 1.0 provides the first
implementation of a gradient boosted hazard estimator that is fully
nonparametric. This paper introduces BoXHED 2.0, a quantum leap over BoXHED 1.0
in several ways. Crucially, BoXHED 2.0 can deal with survival data that goes
far beyond right-censoring and it also supports recurring events. To our
knowledge, this is the only nonparametric machine learning implementation that
is able to do so. Another major improvement is that BoXHED 2.0 is orders of
magnitude more scalable, due in part to a novel data preprocessing step that
sidesteps the need for explicit quadrature when dealing with time-dependent
covariates. BoXHED 2.0 supports the use of GPUs and multicore CPUs, and is
available from GitHub: this http URL.

    

### [[2103.13392] Scalable Pareto Front Approximation for Deep Multi-Objective Learning](http://arxiv.org/abs/2103.13392)


  Multi-objective optimization (MOO) is a prevalent challenge for Deep
Learning, however, there exists no scalable MOO solution for truly deep neural
networks. Prior work either demand optimizing a new network for every point on
the Pareto front, or induce a large overhead to the number of trainable
parameters by using hyper-networks conditioned on modifiable preferences. In
this paper, we propose to condition the network directly on these preferences
by augmenting them to the feature space. Furthermore, we ensure a well-spread
Pareto front by penalizing the solutions to maintain a small angle to the
preference vector. In a series of experiments, we demonstrate that our Pareto
fronts achieve state-of-the-art quality despite being computed significantly
faster. Furthermore, we showcase the scalability as our method approximates the
full Pareto front on the CelebA dataset with an EfficientNet network at a tiny
training time overhead of 7% compared to a simple single-objective
optimization. We make our code publicly available at
this https URL.

    

### [[2104.08261] Adaptive Robust Model Predictive Control with Matched and Unmatched Uncertainty](http://arxiv.org/abs/2104.08261)


  We propose a learning-based robust predictive control algorithm that
compensates for significant uncertainty in the dynamics for a class of
discrete-time systems that are nominally linear with an additive nonlinear
component. Such systems commonly model the nonlinear effects of an unknown
environment on a nominal system. We optimize over a class of nonlinear feedback
policies inspired by certainty equivalent "estimate-and-cancel" control laws
pioneered in classical adaptive control to achieve significant performance
improvements in the presence of uncertainties of large magnitude, a setting in
which existing learning-based predictive control algorithms often struggle to
guarantee safety. In contrast to previous work in robust adaptive MPC, our
approach allows us to take advantage of structure (i.e., the numerical
predictions) in the a priori unknown dynamics learned online through function
approximation. Our approach also extends typical nonlinear adaptive control
methods to systems with state and input constraints even when we cannot
directly cancel the additive uncertain function from the dynamics. Moreover, we
apply contemporary statistical estimation techniques to certify the system's
safety through persistent constraint satisfaction with high probability.
Finally, we show in simulation that our method can accommodate more significant
unknown dynamics terms than existing methods.

    

### [[2105.13203] Conic Blackwell Algorithm: Parameter-Free Convex-Concave Saddle-Point Solving](http://arxiv.org/abs/2105.13203)


  We develop new parameter-free and scale-free algorithms for solving
convex-concave saddle-point problems. Our results are based on a new simple
regret minimizer, the Conic Blackwell Algorithm$^+$ (CBA$^+$), which attains
$O(1/\sqrt{T})$ average regret. Intuitively, our approach generalizes to other
decision sets of interest ideas from the Counterfactual Regret minimization
(CFR$^+$) algorithm, which has very strong practical performance for solving
sequential games on simplexes. We show how to implement CBA$^+$ for the
simplex, $\ell_{p}$ norm balls, and ellipsoidal confidence regions in the
simplex, and we present numerical experiments for solving matrix games and
distributionally robust optimization problems. Our empirical results show that
CBA$^+$ is a simple algorithm that outperforms state-of-the-art methods on
synthetic data and real data instances, without the need for any choice of step
sizes or other algorithmic parameters.

    

### [[2106.02078] Improving Neural Network Robustness via Persistency of Excitation](http://arxiv.org/abs/2106.02078)


  Improving adversarial robustness of neural networks remains a major
challenge. Fundamentally, training a neural network via gradient descent is a
parameter estimation problem. In adaptive control, maintaining persistency of
excitation (PoE) is integral to ensuring convergence of parameter estimates in
dynamical systems to their true values. We show that parameter estimation with
gradient descent can be modeled as a sampling of an adaptive linear
time-varying continuous system. Leveraging this model, and with inspiration
from Model-Reference Adaptive Control (MRAC), we prove a sufficient condition
to constrain gradient descent updates to reference persistently excited
trajectories converging to the true parameters. The sufficient condition is
achieved when the learning rate is less than the inverse of the Lipschitz
constant of the gradient of loss function. We provide an efficient technique
for estimating the corresponding Lipschitz constant in practice using extreme
value theory. Our experimental results in both standard and adversarial
training illustrate that networks trained with the PoE-motivated learning rate
schedule have similar clean accuracy but are significantly more robust to
adversarial attacks than models trained using current state-of-the-art
heuristics.

    

### [[2106.02356] PCA Initialization for Approximate Message Passing in Rotationally Invariant Models](http://arxiv.org/abs/2106.02356)


  We study the problem of estimating a rank-$1$ signal in the presence of
rotationally invariant noise-a class of perturbations more general than
Gaussian noise. Principal Component Analysis (PCA) provides a natural
estimator, and sharp results on its performance have been obtained in the
high-dimensional regime. Recently, an Approximate Message Passing (AMP)
algorithm has been proposed as an alternative estimator with the potential to
improve the accuracy of PCA. However, the existing analysis of AMP requires an
initialization that is both correlated with the signal and independent of the
noise, which is often unrealistic in practice. In this work, we combine the two
methods, and propose to initialize AMP with PCA. Our main result is a rigorous
asymptotic characterization of the performance of this estimator. Both the AMP
algorithm and its analysis differ from those previously derived in the Gaussian
setting: at every iteration, our AMP algorithm requires a specific term to
account for PCA initialization, while in the Gaussian case, PCA initialization
affects only the first iteration of AMP. The proof is based on a two-phase
artificial AMP that first approximates the PCA estimator and then mimics the
true AMP. Our numerical simulations show an excellent agreement between AMP
results and theoretical predictions, and suggest an interesting open direction
on achieving Bayes-optimal performance.

    

### [[2106.02615] Consensus Multiplicative Weights Update: Learning to Learn using Projector-based Game Signatures](http://arxiv.org/abs/2106.02615)


  Cheung and Piliouras (2020) recently showed that two variants of the
Multiplicative Weights Update method - OMWU and MWU - display opposite
convergence properties depending on whether the game is zero-sum or
cooperative. Inspired by this work and the recent literature on learning to
optimize for single functions, we introduce a new framework for learning
last-iterate convergence to Nash Equilibria in games, where the update rule's
coefficients (learning rates) along a trajectory are learnt by a reinforcement
learning policy that is conditioned on the nature of the game: \textit{the game
signature}. We construct the latter using a new decomposition of two-player
games into eight components corresponding to commutative projection operators,
generalizing and unifying recent game concepts studied in the literature. We
compare the performance of various update rules when their coefficients are
learnt, and show that the RL policy is able to exploit the game signature
across a wide range of game types. In doing so, we introduce CMWU, a new
algorithm that extends consensus optimization to the constrained case, has
local convergence guarantees for zero-sum bimatrix games, and show that it
enjoys competitive performance on both zero-sum games with constant
coefficients and across a spectrum of games when its coefficients are learnt.

    

### [[2106.05306] DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact](http://arxiv.org/abs/2106.05306)


  Cloth simulation has wide applications in computer animation, garment design,
and robot-assisted dressing. This work presents a differentiable cloth
simulator whose additional gradient information facilitates cloth-related
applications. Our differentiable simulator extends a state-of-the-art cloth
simulator based on Projective Dynamics (PD) and with dry frictional contact. We
draw inspiration from previous work to propose a fast and novel method for
deriving gradients in PD-based cloth simulation with dry frictional contact.
Furthermore, we conduct a comprehensive analysis and evaluation of the
usefulness of gradients in contact-rich cloth simulation. Finally, we
demonstrate the efficacy of our simulator in a number of downstream
applications, including system identification, trajectory optimization for
assisted dressing, closed-loop control, inverse design, and real-to-sim
transfer. We observe a substantial speedup obtained from using our gradient
information in solving most of these applications.

    

### [[2106.06333] Invariant Information Bottleneck for Domain Generalization](http://arxiv.org/abs/2106.06333)


  The main challenge for domain generalization (DG) is to overcome the
potential distributional shift between multiple training domains and unseen
test domains. One popular class of DG algorithms aims to learn representations
that have an invariant causal relation across the training domains. However,
certain features, called \emph{pseudo-invariant features}, may be invariant in
the training domain but not the test domain and can substantially decreases the
performance of existing algorithms. To address this issue, we propose a novel
algorithm, called Invariant Information Bottleneck (IIB), that learns a
minimally sufficient representation that is invariant across training and
testing domains. By minimizing the mutual information between the
representation and inputs, IIB alleviates its reliance on pseudo-invariant
features, which is desirable for DG. To verify the effectiveness of the IIB
principle, we conduct extensive experiments on large-scale DG benchmarks. The
results show that IIB outperforms invariant learning baseline (e.g. IRM) by an
average of 2.8\% and 3.8\% accuracy over two evaluation metrics.

    

### [[2106.07847] Learning Stable Classifiers by Transferring Unstable Features](http://arxiv.org/abs/2106.07847)


  While unbiased machine learning models are essential for many applications,
bias is a human-defined concept that can vary across tasks. Given only
input-label pairs, algorithms may lack sufficient information to distinguish
stable (causal) features from unstable (spurious) features. However, related
tasks often share similar biases -- an observation we may leverage to develop
stable classifiers in the transfer setting. In this work, we explicitly inform
the target classifier about unstable features in the source tasks.
Specifically, we derive a representation that encodes the unstable features by
contrasting different data environments in the source task. We achieve
robustness by clustering data of the target task according to this
representation and minimizing the worst-case risk across these clusters. We
evaluate our method on both text and image classifications. Empirical results
demonstrate that our algorithm is able to maintain robustness on the target
task, outperforming the best baseline by 22.9% in absolute accuracy across 12
transfer settings. Our code is available at this https URL.

    

### [[2106.09669] Improving On-Screen Sound Separation for Open-Domain Videos with Audio-Visual Self-Attention](http://arxiv.org/abs/2106.09669)


  We introduce a state-of-the-art audio-visual on-screen sound separation
system which is capable of learning to separate sounds and associate them with
on-screen objects by looking at in-the-wild videos. We identify limitations of
previous work on audio-visual on-screen sound separation, including the
simplicity and coarse resolution of spatio-temporal attention, and poor
convergence of the audio separation model. Our proposed model addresses these
issues using cross-modal and self-attention modules that capture audio-visual
dependencies at a finer resolution over time, and by unsupervised pre-training
of audio separation model. These improvements allow the model to generalize to
a much wider set of unseen videos. We also show a robust way to further improve
the generalization capability of our models by calibrating the probabilities of
our audio-visual on-screen classifier, using only a small amount of in-domain
videos labeled for their on-screen presence. For evaluation and semi-supervised
training, we collected human annotations of on-screen audio from a large
database of in-the-wild videos (YFCC100m). Our results show marked improvements
in on-screen separation performance, in more general conditions than previous
methods.

    

### [[2106.14568] Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity](http://arxiv.org/abs/2106.14568)


  Recent works on sparse neural networks have demonstrated the possibility to
train a sparse subnetwork independently from scratch, to match the performance
of its corresponding dense network. However, identifying such sparse
subnetworks (winning tickets) either involves a costly iterative
train-prune-retrain process (e.g., Lottery Ticket Hypothesis) or an
over-extended training time (e.g., Dynamic Sparse Training). In this work, we
draw a unique connection between sparse neural network training and the deep
ensembling technique, yielding a novel ensemble learning framework called
FreeTickets. Instead of starting from a dense network, FreeTickets randomly
initializes a sparse subnetwork and then trains the subnetwork while
dynamically adjusting its sparse mask, resulting in many diverse sparse
subnetworks throughout the training process. FreeTickets is defined as the
ensemble of these sparse subnetworks freely obtained during this one-pass,
sparse-to-sparse training, which uses only a fraction of the computational
resources required by the vanilla dense training. Moreover, despite being an
ensemble of models, FreeTickets has even fewer parameters and training FLOPs
compared to a single dense model: this seemingly counter-intuitive outcome is
due to the high sparsity of each subnetwork. FreeTickets is observed to
demonstrate a significant all-round improvement compared to standard dense
baselines, in prediction accuracy, uncertainty estimation, robustness, and
efficiency. FreeTickets easily outperforms the naive deep ensemble with
ResNet50 on ImageNet using only a quarter of the training FLOPs required by the
latter. Our results provide insights into the strength of sparse neural
networks and suggest that the benefits of sparsity go way beyond the usually
expected inference efficiency.

    

### [[2107.09003] Constraints Penalized Q-Learning for Safe Offline Reinforcement Learning](http://arxiv.org/abs/2107.09003)


  We study the problem of safe offline reinforcement learning (RL), the goal is
to learn a policy that maximizes long-term reward while satisfying safety
constraints given only offline data, without further interaction with the
environment. This problem is more appealing for real world RL applications, in
which data collection is costly or dangerous. Enforcing constraint satisfaction
is non-trivial, especially in offline settings, as there is a potential large
discrepancy between the policy distribution and the data distribution, causing
errors in estimating the value of safety constraints. We show that naïve
approaches that combine techniques from safe RL and offline RL can only learn
sub-optimal solutions. We thus develop a simple yet effective algorithm,
Constraints Penalized Q-Learning (CPQ), to solve the problem. Our method admits
the use of data generated by mixed behavior policies. We present a theoretical
analysis and demonstrate empirically that our approach can learn robustly
across a variety of benchmark control tasks, outperforming several baselines.

    

### [[2110.01601] NeuFENet: Neural Finite Element Solutions with Theoretical Bounds for Parametric PDEs](http://arxiv.org/abs/2110.01601)


  We consider a mesh-based approach for training a neural network to produce
field predictions of solutions to parametric partial differential equations
(PDEs). This approach contrasts current approaches for "neural PDE solvers"
that employ collocation-based methods to make point-wise predictions of
solutions to PDEs. This approach has the advantage of naturally enforcing
different boundary conditions as well as ease of invoking well-developed PDE
theory -- including analysis of numerical stability and convergence -- to
obtain capacity bounds for our proposed neural networks in discretized domains.
We explore our mesh-based strategy, called NeuFENet, using a weighted Galerkin
loss function based on the Finite Element Method (FEM) on a parametric elliptic
PDE. The weighted Galerkin loss (FEM loss) is similar to an energy functional
that produces improved solutions, satisfies a priori mesh convergence, and can
model Dirichlet and Neumann boundary conditions. We prove theoretically, and
illustrate with experiments, convergence results analogous to mesh convergence
analysis deployed in finite element solutions to PDEs. These results suggest
that a mesh-based neural network approach serves as a promising approach for
solving parametric PDEs with theoretical bounds.

    

### [[2110.05263] 2021 Drexel Society of Artificial Intelligence Research Conference](http://arxiv.org/abs/2110.05263)


  The 2021 Drexel Society of Artificial Intelligence Research Conference
highlights papers focused on a broad set of papers in machine learning. This
was our organizations' first annual conference. It was conducted virtually via
Zoom. The highlights are currently posted on YouTube.

    

### [[2110.05286] Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning](http://arxiv.org/abs/2110.05286)


  Our work aims at efficiently leveraging ambiguous demonstrations for the
training of a reinforcement learning (RL) agent. An ambiguous demonstration can
usually be interpreted in multiple ways, which severely hinders the RL-Agent
from learning stably and efficiently. Since an optimal demonstration may also
suffer from being ambiguous, previous works that combine RL and learning from
demonstration (RLfD works) may not work well. Inspired by how humans handle
such situations, we propose to use self-explanation (an agent generates
explanations for itself) to recognize valuable high-level relational features
as an interpretation of why a successful trajectory is successful. This way,
the agent can provide some guidance for its RL learning. Our main contribution
is to propose the Self-Explanation for RL from Demonstrations (SERLfD)
framework, which can overcome the limitations of traditional RLfD works. Our
experimental results show that an RLfD model can be improved by using our
SERLfD framework in terms of training stability and performance.

    

### [[2110.07186] An FPGA-Based Fully Pipelined Bilateral Grid for Real-Time Image Denoising](http://arxiv.org/abs/2110.07186)


  The bilateral filter (BF) is widely used in image processing because it can
perform denoising while preserving edges. It has disadvantages in that it is
nonlinear, and its computational complexity and hardware resources are directly
proportional to its window size. Thus far, several approximation methods and
hardware implementations have been proposed to solve these problems. However,
processing large-scale and high-resolution images in real time under severe
hardware resource constraints remains a challenge. This paper proposes a
real-time image denoising system that uses an FPGA based on the bilateral grid
(BG). In the BG, a 2D image consisting of x- and y-axes is projected onto a 3D
space called a "grid," which consists of axes that correlate to the
x-component, y-component, and intensity value of the input image. This grid is
then blurred using the Gaussian filter, and the output image is generated by
interpolating the grid. Although it is possible to change the window size in
the BF, it is impossible to change it on the input image in the BG. This makes
it difficult to associate the BG with the BF and to obtain the property of
suppressing the increase in hardware resources when the window radius is
enlarged. This study demonstrates that a BG with a variable-sized window can be
realized by introducing the window radius parameter wherein the window radius
on the grid is always 1. We then implement this BG on an FPGA in a fully
pipelined manner. Further, we verify that our design suppresses the increase in
hardware resources even when the window size is enlarged and outperforms the
existing designs in terms of computation speed and hardware resources.

    

### [[2110.07600] PointAcc: Efficient Point Cloud Accelerator](http://arxiv.org/abs/2110.07600)


  Deep learning on point clouds plays a vital role in a wide range of
applications such as autonomous driving and AR/VR. These applications interact
with people in real-time on edge devices and thus require low latency and low
energy. Compared to projecting the point cloud to 2D space, directly processing
the 3D point cloud yields higher accuracy and lower #MACs. However, the
extremely sparse nature of point cloud poses challenges to hardware
acceleration. For example, we need to explicitly determine the nonzero outputs
and search for the nonzero neighbors (mapping operation), which is unsupported
in existing accelerators. Furthermore, explicit gather and scatter of sparse
features are required, resulting in large data movement overhead. In this
paper, we comprehensively analyze the performance bottleneck of modern point
cloud networks on CPU/GPU/TPU. To address the challenges, we then present
PointAcc, a novel point cloud deep learning accelerator. PointAcc maps diverse
mapping operations onto one versatile ranking-based kernel, streams the sparse
computation with configurable caching, and temporally fuses consecutive dense
layers to reduce the memory footprint. Evaluated on 8 point cloud models across
4 applications, PointAcc achieves 3.7X speedup and 22X energy savings over RTX
2080Ti GPU. Co-designed with light-weight neural networks, PointAcc rivals the
prior accelerator Mesorasi by 100X speedup with 9.1% higher accuracy running
segmentation on the S3DIS dataset. PointAcc paves the way for efficient point
cloud recognition.

    

### [[2110.07083] Dynamic Conflict Resolution of IoT Services in Smart Homes](http://arxiv.org/abs/2110.07083)


  We propose a novel conflict resolution framework for IoT services in
multi-resident smart homes. The proposed framework employs a preference
extraction model based on a temporal proximity strategy. We design a preference
aggregation model using a matrix factorization-based approach (i.e., singular
value decomposition). The concepts of current resident item matrix and ideal
resident item matrix are introduced as key criteria to cater to the conflict
resolution framework. Finally, a set of experiments on real-world datasets are
conducted to show the effectiveness of the proposed approach.

    

### [[2106.16064] Efficient Sparse Matrix Kernels based on Adaptive Workload-Balancing and Parallel-Reduction](http://arxiv.org/abs/2106.16064)


  Sparse matrix-vector and matrix-matrix multiplication (SpMV and SpMM) are
fundamental in both conventional (graph analytics, scientific computing) and
emerging (sparse DNN, GNN) domains. Workload-balancing and parallel-reduction
are widely-used design principles for efficient SpMV. However, prior work fails
to resolve how to implement and adaptively use the two principles for SpMV/MM.
To overcome this obstacle, we first complete the implementation space with
optimizations by filling three missing pieces in prior work, including: (1) We
show that workload-balancing and parallel-reduction can be combined through a
segment-reduction algorithm implemented with SIMD-shuffle primitives. (2) We
show that parallel-reduction can be implemented in SpMM through loading the
dense-matrix rows with vector memory operations. (3) We show that vectorized
loading of sparse rows, being a part of the benefit of parallel-reduction, can
co-exist with sequential-reduction in SpMM through temporally caching
sparse-matrix elements in the shared memory. In terms of adaptive use, we
analyze how the benefit of two principles change with two characteristics from
the input data space: the diverse sparsity pattern and dense-matrix width. We
find the benefit of the two principles fades along with the increased total
workload, i.e. the increased dense-matrix width. We also identify, for SpMV and
SpMM, different sparse-matrix features that impact workload-balancing
effectiveness. Our design consistently exceeds cuSPARSE by 1.07-1.57x on
different GPUs and dense matrix width, and the kernel selection rules involve
5-12% performance loss compared with optimal choices. Our kernel is being
integrated into popular graph learning frameworks to accelerate GNN training.

    

### [[2110.06956] Considering user agreement in learning to predict the aesthetic quality](http://arxiv.org/abs/2110.06956)


  How to robustly rank the aesthetic quality of given images has been a
long-standing ill-posed topic. Such challenge stems mainly from the diverse
subjective opinions of different observers about the varied types of content.
There is a growing interest in estimating the user agreement by considering the
standard deviation of the scores, instead of only predicting the mean aesthetic
opinion score. Nevertheless, when comparing a pair of contents, few studies
consider how confident are we regarding the difference in the aesthetic scores.
In this paper, we thus propose (1) a re-adapted multi-task attention network to
predict both the mean opinion score and the standard deviation in an end-to-end
manner; (2) a brand-new confidence interval ranking loss that encourages the
model to focus on image-pairs that are less certain about the difference of
their aesthetic scores. With such loss, the model is encouraged to learn the
uncertainty of the content that is relevant to the diversity of observers'
opinions, i.e., user disagreement. Extensive experiments have demonstrated that
the proposed multi-task aesthetic model achieves state-of-the-art performance
on two different types of aesthetic datasets, i.e., AVA and TMGA.

    

### [[2110.06968] Interpretable AI forecasting for numerical relativity waveforms of quasi-circular, spinning, non-precessing binary black hole mergers](http://arxiv.org/abs/2110.06968)


  We present a deep-learning artificial intelligence model that is capable of
learning and forecasting the late-inspiral, merger and ringdown of numerical
relativity waveforms that describe quasi-circular, spinning, non-precessing
binary black hole mergers. We used the NRHybSur3dq8 surrogate model to produce
train, validation and test sets of $\ell=|m|=2$ waveforms that cover the
parameter space of binary black hole mergers with mass-ratios $q\leq8$ and
individual spins $|s^z_{\{1,2\}}| \leq 0.8$. These waveforms cover the time
range $t\in[-5000\textrm{M}, 130\textrm{M}]$, where $t=0M$ marks the merger
event, defined as the maximum value of the waveform amplitude. We harnessed the
ThetaGPU supercomputer at the Argonne Leadership Computing Facility to train
our AI model using a training set of 1.5 million waveforms. We used 16 NVIDIA
DGX A100 nodes, each consisting of 8 NVIDIA A100 Tensor Core GPUs and 2 AMD
Rome CPUs, to fully train our model within 3.5 hours. Our findings show that
artificial intelligence can accurately forecast the dynamical evolution of
numerical relativity waveforms in the time range $t\in[-100\textrm{M},
130\textrm{M}]$. Sampling a test set of 190,000 waveforms, we find that the
average overlap between target and predicted waveforms is $\gtrsim99\%$ over
the entire parameter space under consideration. We also combined scientific
visualization and accelerated computing to identify what components of our
model take in knowledge from the early and late-time waveform evolution to
accurately forecast the latter part of numerical relativity waveforms. This
work aims to accelerate the creation of scalable, computationally efficient and
interpretable artificial intelligence models for gravitational wave
astrophysics.

    

### [[2110.06997] Bandits Don't Follow Rules: Balancing Multi-Facet Machine Translation with Multi-Armed Bandits](http://arxiv.org/abs/2110.06997)


  Training data for machine translation (MT) is often sourced from a multitude
of large corpora that are multi-faceted in nature, e.g. containing contents
from multiple domains or different levels of quality or complexity. Naturally,
these facets do not occur with equal frequency, nor are they equally important
for the test scenario at hand. In this work, we propose to optimize this
balance jointly with MT model parameters to relieve system developers from
manual schedule design. A multi-armed bandit is trained to dynamically choose
between facets in a way that is most beneficial for the MT system. We evaluate
it on three different multi-facet applications: balancing translationese and
natural training data, or data from multiple domains or multiple language
pairs. We find that bandit learning leads to competitive MT systems across
tasks, and our analysis provides insights into its learned strategies and the
underlying data sets.

    

### [[2110.07031] Improving the Robustness to Variations of Objects and Instructions with a Neuro-Symbolic Approach for Interactive Instruction Following](http://arxiv.org/abs/2110.07031)


  An interactive instruction following task has been proposed as a benchmark
for learning to map natural language instructions and first-person vision into
sequences of actions to interact with objects in a 3D simulated environment. We
find that an existing end-to-end neural model for this task is not robust to
variations of objects and language instructions. We assume that this problem is
due to the high sensitiveness of neural feature extraction to small changes in
vision and language inputs. To mitigate this problem, we propose a
neuro-symbolic approach that performs reasoning over high-level symbolic
representations that are robust to small changes in raw inputs. Our experiments
on the ALFRED dataset show that our approach significantly outperforms the
existing model by 18, 52, and 73 points in the success rate on the
ToggleObject, PickupObject, and SliceObject subtasks in unseen environments
respectively.

    

### [[2110.07033] Compliance checking in reified IO logic via SHACL](http://arxiv.org/abs/2110.07033)


  Reified Input/Output (I/O) logic[21] has been recently proposed to model
real-world norms in terms of the logic in [11]. This is massively grounded on
the notion of reification, and it has specifically designed to model meaning of
natural language sentences, such as the ones occurring in existing legislation.
This paper presents a methodology to carry out compliance checking on reified
I/O logic formulae. These are translated in SHACL (Shapes Constraint Language)
shapes, a recent W3C recommendation to validate and reason with RDF
triplestores. Compliance checking is then enforced by validating RDF graphs
describing states of affairs with respect to these SHACL shapes.

    

### [[2110.07038] Towards Efficient NLP: A Standard Evaluation and A Strong Baseline](http://arxiv.org/abs/2110.07038)


  Supersized pre-trained language models have pushed the accuracy of various
NLP tasks to a new state-of-the-art (SOTA). Rather than pursuing the reachless
SOTA accuracy, most works are pursuing improvement on other dimensions such as
efficiency, leading to "Pareto SOTA". Different from accuracy, the metric for
efficiency varies across different studies, making them hard to be fairly
compared. To that end, this work presents ELUE (Efficient Language
Understanding Evaluation), a standard evaluation, and a public leaderboard for
efficient NLP models. ELUE is dedicated to depicting the Pareto Front for
various language understanding tasks, such that it can tell whether and how
much a method achieves Pareto improvement. Along with the benchmark, we also
pre-train and release a strong baseline, ElasticBERT, whose elasticity is both
static and dynamic. ElasticBERT is static in that it allows reducing model
layers on demand. ElasticBERT is dynamic in that it selectively executes parts
of model layers conditioned on the input. We demonstrate the ElasticBERT,
despite its simplicity, outperforms or performs on par with SOTA compressed and
early exiting models. The ELUE benchmark is publicly available at
http://eluebenchmark.fastnlp.top/.

    

### [[2110.07058] Ego4D: Around the World in 3,000 Hours of Egocentric Video](http://arxiv.org/abs/2110.07058)


  We introduce Ego4D, a massive-scale egocentric video dataset and benchmark
suite. It offers 3,025 hours of daily-life activity video spanning hundreds of
scenarios (household, outdoor, workplace, leisure, etc.) captured by 855 unique
camera wearers from 74 worldwide locations and 9 different countries. The
approach to collection is designed to uphold rigorous privacy and ethics
standards with consenting participants and robust de-identification procedures
where relevant. Ego4D dramatically expands the volume of diverse egocentric
video footage publicly available to the research community. Portions of the
video are accompanied by audio, 3D meshes of the environment, eye gaze, stereo,
and/or synchronized videos from multiple egocentric cameras at the same event.
Furthermore, we present a host of new benchmark challenges centered around
understanding the first-person visual experience in the past (querying an
episodic memory), present (analyzing hand-object manipulation, audio-visual
conversation, and social interactions), and future (forecasting activities). By
publicly sharing this massive annotated dataset and benchmark suite, we aim to
push the frontier of first-person perception. Project page:
this https URL


### [[2110.07066] An algorithm for a fairer and better voting system](http://arxiv.org/abs/2110.07066)


  The major finding, of this article, is an ensemble method, but more exactly,
a novel, better ranked voting system (and other variations of it), that aims to
solve the problem of finding the best candidate to represent the voters. We
have the source code on GitHub, for making realistic simulations of elections,
based on artificial intelligence for comparing different variations of the
algorithm, and other already known algorithms.
We have convincing evidence that our algorithm is better than Instant-Runoff
Voting, Preferential Block Voting, Single Transferable Vote, and First Past The
Post (if certain, natural conditions are met, to support the wisdom of the
crowds). By also comparing with the best voter, we demonstrated the wisdom of
the crowds, suggesting that democracy (distributed system) is a better option
than dictatorship (centralized system), if those certain, natural conditions
are met.
Voting systems are not restricted to politics, they are ensemble methods for
artificial intelligence, but the context of this article is natural
intelligence. It is important to find a system that is fair (e.g. freedom of
expression on the ballot exists), especially when the outcome of the voting
system has social impact: some voting systems have the unfair inevitability to
trend (over time) towards the same two major candidates (Duverger's law).

    

### [[2110.07139] Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer](http://arxiv.org/abs/2110.07139)


  Adversarial attacks and backdoor attacks are two common security threats that
hang over deep learning. Both of them harness task-irrelevant features of data
in their implementation. Text style is a feature that is naturally irrelevant
to most NLP tasks, and thus suitable for adversarial and backdoor attacks. In
this paper, we make the first attempt to conduct adversarial and backdoor
attacks based on text style transfer, which is aimed at altering the style of a
sentence while preserving its meaning. We design an adversarial attack method
and a backdoor attack method, and conduct extensive experiments to evaluate
them. Experimental results show that popular NLP models are vulnerable to both
adversarial and backdoor attacks based on text style transfer -- the attack
success rates can exceed 90% without much effort. It reflects the limited
ability of NLP models to handle the feature of text style that has not been
widely realized. In addition, the style transfer-based adversarial and backdoor
attack methods show superiority to baselines in many aspects. All the code and
data of this paper can be obtained at this https URL.

    

### [[2110.07181] Relation-aware Heterogeneous Graph for User Profiling](http://arxiv.org/abs/2110.07181)


  User profiling has long been an important problem that investigates user
interests in many real applications. Some recent works regard users and their
interacted objects as entities of a graph and turn the problem into a node
classification task. However, they neglect the difference of distinct
interaction types, e.g. user clicks an item v.s.user purchases an item, and
thus cannot incorporate such information well. To solve these issues, we
propose to leverage the relation-aware heterogeneous graph method for user
profiling, which also allows capturing significant meta relations. We adopt the
query, key, and value mechanism in a transformer fashion for heterogeneous
message passing so that entities can effectively interact with each other. Via
such interactions on different relation types, our model can generate
representations with rich information for the user profile prediction. We
conduct experiments on two real-world e-commerce datasets and observe a
significant performance boost of our approach.

    

### [[2110.07202] Unrolled Variational Bayesian Algorithm for Image Blind Deconvolution](http://arxiv.org/abs/2110.07202)


  In this paper, we introduce a variational Bayesian algorithm (VBA) for image
blind deconvolution. Our generic framework incorporates smoothness priors on
the unknown blur/image and possible affine constraints (e.g., sum to one) on
the blur kernel. One of our main contributions is the integration of VBA within
a neural network paradigm, following an unrolling methodology. The proposed
architecture is trained in a supervised fashion, which allows us to optimally
set two key hyperparameters of the VBA model and lead to further improvements
in terms of resulting visual quality. Various experiments involving
grayscale/color images and diverse kernel shapes, are performed. The numerical
examples illustrate the high performance of our approach when compared to
state-of-the-art techniques based on optimization, Bayesian estimation, or deep
learning.

    

### [[2110.07209] A Dual-Attention Neural Network for Pun Location and Using Pun-Gloss Pairs for Interpretation](http://arxiv.org/abs/2110.07209)


  Pun location is to identify the punning word (usually a word or a phrase that
makes the text ambiguous) in a given short text, and pun interpretation is to
find out two different meanings of the punning word. Most previous studies
adopt limited word senses obtained by WSD(Word Sense Disambiguation) technique
or pronunciation information in isolation to address pun location. For the task
of pun interpretation, related work pays attention to various WSD algorithms.
In this paper, a model called DANN (Dual-Attentive Neural Network) is proposed
for pun location, effectively integrates word senses and pronunciation with
context information to address two kinds of pun at the same time. Furthermore,
we treat pun interpretation as a classification task and construct pungloss
pairs as processing data to solve this task. Experiments on the two benchmark
datasets show that our proposed methods achieve new state-of-the-art results.
Our source code is available in the public code repository.

    

### [[2110.07239] Solving Large Break Minimization Problems in a Mirrored Double Round-robin Tournament Using Quantum Annealing](http://arxiv.org/abs/2110.07239)


  Quantum annealing (QA) has gained considerable attention because it can be
applied to combinatorial optimization problems, which have numerous
applications in logistics, scheduling, and finance. In recent years, research
on solving practical combinatorial optimization problems using them has
accelerated. However, researchers struggle to find practical combinatorial
optimization problems, for which quantum annealers outperform other
mathematical optimization solvers. Moreover, there are only a few studies that
compare the performance of quantum annealers with one of the most sophisticated
mathematical optimization solvers, such as Gurobi and CPLEX. In our study, we
determine that QA demonstrates better performance than the solvers in the break
minimization problem in a mirrored double round-robin tournament (MDRRT). We
also explain the desirable performance of QA for the sparse interaction between
variables and a problem without constraints. In this process, we demonstrate
that the break minimization problem in an MDRRT can be expressed as a 4-regular
graph. Through computational experiments, we solve this problem using our QA
approach and two-integer programming approaches, which were performed using the
latest quantum annealer D-Wave Advantage, and the sophisticated mathematical
optimization solver, Gurobi, respectively. Further, we compare the quality of
the solutions and the computational time. QA was able to determine the exact
solution in 0.05 seconds for problems with 20 teams, which is a practical size.
In the case of 36 teams, it took 84.8 s for the integer programming method to
reach the objective function value, which was obtained by the quantum annealer
in 0.05 s. These results not only present the break minimization problem in an
MDRRT as an example of applying QA to practical optimization problems, but also
contribute to find problems that can be effectively solved by QA.

    

### [[2110.07244] Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](http://arxiv.org/abs/2110.07244)


  Pre-trained language models (PLMs), such as BERT and GPT, have revolutionized
the field of NLP, not only in the general domain but also in the biomedical
domain. Most prior efforts in building biomedical PLMs have resorted simply to
domain adaptation and focused mainly on English. In this work we introduce
eHealth, a biomedical PLM in Chinese built with a new pre-training framework.
This new framework trains eHealth as a discriminator through both token-level
and sequence-level discrimination. The former is to detect input tokens
corrupted by a generator and select their original signals from plausible
candidates, while the latter is to further distinguish corruptions of a same
original sequence from those of the others. As such, eHealth can learn language
semantics at both the token and sequence levels. Extensive experiments on 11
Chinese biomedical language understanding tasks of various forms verify the
effectiveness and superiority of our approach. The pre-trained model is
available to the public at
\url{this https URL} and the
code will also be released later.

    

### [[2110.07331] Plug-Tagger: A Pluggable Sequence Labeling Framework Using Language Models](http://arxiv.org/abs/2110.07331)


  Plug-and-play functionality allows deep learning models to adapt well to
different tasks without requiring any parameters modified. Recently,
prefix-tuning was shown to be a plug-and-play method on various text generation
tasks by simply inserting corresponding continuous vectors into the inputs.
However, sequence labeling tasks invalidate existing plug-and-play methods
since different label sets demand changes to the architecture of the model
classifier. In this work, we propose the use of label word prediction instead
of classification to totally reuse the architecture of pre-trained models for
sequence labeling tasks. Specifically, for each task, a label word set is first
constructed by selecting a high-frequency word for each class respectively, and
then, task-specific vectors are inserted into the inputs and optimized to
manipulate the model predictions towards the corresponding label words. As a
result, by simply switching the plugin vectors on the input, a frozen
pre-trained language model is allowed to perform different tasks. Experimental
results on three sequence labeling tasks show that the performance of the
proposed method can achieve comparable performance with standard fine-tuning
with only 0.1\% task-specific parameters. In addition, our method is up to 70
times faster than non-plug-and-play methods while switching different tasks
under the resource-constrained scenario.

    

### [[2110.07333] A Survey on Legal Question Answering Systems](http://arxiv.org/abs/2110.07333)


  Many legal professionals think that the explosion of information about local,
regional, national, and international legislation makes their practice more
costly, time-consuming, and even error-prone. The two main reasons for this are
that most legislation is usually unstructured, and the tremendous amount and
pace with which laws are released causes information overload in their daily
tasks. In the case of the legal domain, the research community agrees that a
system allowing to generate automatic responses to legal questions could
substantially impact many practical implications in daily activities. The
degree of usefulness is such that even a semi-automatic solution could
significantly help to reduce the workload to be faced. This is mainly because a
Question Answering system could be able to automatically process a massive
amount of legal resources to answer a question or doubt in seconds, which means
that it could save resources in the form of effort, money, and time to many
professionals in the legal sector. In this work, we quantitatively and
qualitatively survey the solutions that currently exist to meet this challenge.

    

### [[2110.07337] Topic-time Heatmaps for Human-in-the-loop Topic Detection and Tracking](http://arxiv.org/abs/2110.07337)


  The essential task of Topic Detection and Tracking (TDT) is to organize a
collection of news media into clusters of stories that pertain to the same
real-world event. To apply TDT models to practical applications such as search
engines and discovery tools, human guidance is needed to pin down the scope of
an "event" for the corpus of interest. In this work in progress, we explore a
human-in-the-loop method that helps users iteratively fine-tune TDT algorithms
so that both the algorithms and the users themselves better understand the
nature of the events. We generate a visual overview of the entire corpus,
allowing the user to select regions of interest from the overview, and then ask
a series of questions to affirm (or reject) that the selected documents belong
to the same event. The answers to these questions supplement the training data
for the event similarity model that underlies the system.

    

### [[2110.07376] Domain Adaptation on Semantic Segmentation with Separate Affine Transformation in Batch Normalization](http://arxiv.org/abs/2110.07376)


  In recent years, unsupervised domain adaptation (UDA) for semantic
segmentation has brought many researchers'attention. Many of them take an
approach to design a complex system so as to better align the gap between
source and target domain. Instead, we focus on the very basic structure of the
deep neural network, Batch Normalization, and propose to replace the Sharing
Affine Transformation with our proposed Separate Affine Transformation (SEAT).
The proposed SEAT is simple, easily implemented and easy to integrate into
existing adversarial learning based UDA methods. Also, to further improve the
adaptation quality, we introduce multi level adaptation by adding the
lower-level features to the higher-level ones before feeding them to the
discriminator, without adding extra discriminator like others. Experiments show
that the proposed methods is less complex without losing performance accuracy
when compared with other UDA methods.

    

### [[2110.07476] Query and Extract: Refining Event Extraction as Type-oriented Binary Decoding](http://arxiv.org/abs/2110.07476)


  Event extraction is typically modeled as a multi-class classification problem
where both event types and argument roles are treated as atomic symbols. These
approaches are usually limited to a set of pre-defined types. We propose a
novel event extraction framework that takes event types and argument roles as
natural language queries to extract candidate triggers and arguments from the
input text. With the rich semantics in the queries, our framework benefits from
the attention mechanisms to better capture the semantic correlation between the
event types or argument roles and the input text. Furthermore, the
query-and-extract formulation allows our approach to leverage all available
event annotations from various ontologies as a unified model. Experiments on
two public benchmarks, ACE and ERE, demonstrate that our approach achieves
state-of-the-art performance on each dataset and significantly outperforms
existing methods on zero-shot event extraction. We will make all the programs
publicly available once the paper is accepted.

    

### [[2110.07518] SaFeRDialogues: Taking Feedback Gracefully after Conversational Safety Failures](http://arxiv.org/abs/2110.07518)


  Current open-domain conversational models can easily be made to talk in
inadequate ways. Online learning from conversational feedback given by the
conversation partner is a promising avenue for a model to improve and adapt, so
as to generate fewer of these safety failures. However, current
state-of-the-art models tend to react to feedback with defensive or oblivious
responses. This makes for an unpleasant experience and may discourage
conversation partners from giving feedback in the future. This work proposes
SaFeRDialogues, a task and dataset of graceful responses to conversational
feedback about safety failures. We collect a dataset of 10k dialogues
demonstrating safety failures, feedback signaling them, and a response
acknowledging the feedback. We show how fine-tuning on this dataset results in
conversations that human raters deem considerably more likely to lead to a
civil conversation, without sacrificing engagingness or general conversational
ability.

    

### [[2110.07552] BI-RADS BERT & Using Section Tokenization to Understand Radiology Reports](http://arxiv.org/abs/2110.07552)


  Radiology reports are the main form of communication between radiologists and
other clinicians, and contain important information for patient care. However
in order to use this information for research it is necessary to convert the
raw text into structured data suitable for analysis. Domain specific contextual
word embeddings have been shown to achieve impressive accuracy at such natural
language processing tasks in medicine. In this work we pre-trained a contextual
embedding BERT model using breast radiology reports and developed a classifier
that incorporated the embedding with auxiliary global textual features in order
to perform a section tokenization task. This model achieved a 98% accuracy at
segregating free text reports into sections of information outlined in the
Breast Imaging Reporting and Data System (BI-RADS) lexicon, a significant
improvement over the Classic BERT model without auxiliary information. We then
evaluated whether using section tokenization improved the downstream extraction
of the following fields: modality/procedure, previous cancer, menopausal
status, purpose of exam, breast density and background parenchymal enhancement.
Using the BERT model pre-trained on breast radiology reports combined with
section tokenization resulted in an overall accuracy of 95.9% in field
extraction. This is a 17% improvement compared to an overall accuracy of 78.9%
for field extraction for models without section tokenization and with Classic
BERT embeddings. Our work shows the strength of using BERT in radiology report
analysis and the advantages of section tokenization in identifying key features
of patient factors recorded in breast radiology reports.

    

### [[2110.07572] LAGr: Labeling Aligned Graphs for Improving Systematic Generalization in Semantic Parsing](http://arxiv.org/abs/2110.07572)


  Semantic parsing is the task of producing a structured meaning representation
for natural language utterances or questions. Recent research has pointed out
that the commonly-used sequence-to-sequence (seq2seq) semantic parsers struggle
to generalize systematically, i.e. to handle examples that require recombining
known knowledge in novel settings. In this work, we show that better systematic
generalization can be achieved by producing the meaning representation (MR)
directly as a graph and not as a sequence. To this end we propose LAGr, the
Labeling Aligned Graphs algorithm that produces semantic parses by predicting
node and edge labels for a complete multi-layer input-aligned graph. The
strongly-supervised LAGr algorithm requires aligned graphs as inputs, whereas
weakly-supervised LAGr infers alignments for originally unaligned target graphs
using an approximate MAP inference procedure. On the COGS and CFQ compositional
generalization benchmarks the strongly- and weakly- supervised LAGr algorithms
achieve significant improvements upon the baseline seq2seq parsers.

    

### [[2110.07596] Retrieval-guided Counterfactual Generation for QA](http://arxiv.org/abs/2110.07596)


  Deep NLP models have been shown to learn spurious correlations, leaving them
brittle to input perturbations. Recent work has shown that counterfactual or
contrastive data -- i.e. minimally perturbed inputs -- can reveal these
weaknesses, and that data augmentation using counterfactuals can help
ameliorate them. Proposed techniques for generating counterfactuals rely on
human annotations, perturbations based on simple heuristics, and meaning
representation frameworks. We focus on the task of creating counterfactuals for
question answering, which presents unique challenges related to world
knowledge, semantic diversity, and answerability. To address these challenges,
we develop a Retrieve-Generate-Filter(RGF) technique to create counterfactual
evaluation and training data with minimal human supervision. Using an
open-domain QA framework and question generation model trained on original task
data, we create counterfactuals that are fluent, semantically diverse, and
automatically labeled. Data augmentation with RGF counterfactuals improves
performance on out-of-domain and challenging evaluation sets over and above
existing methods, in both the reading comprehension and open-domain QA
settings. Moreover, we find that RGF data leads to significant improvements in
a model's robustness to local perturbations.

    

### [[1812.00301] Plan-Recognition-Driven Attention Modeling for Visual Recognition](http://arxiv.org/abs/1812.00301)


  Human visual recognition of activities or external agents involves an
interplay between high-level plan recognition and low-level perception. Given
that, a natural question to ask is: can low-level perception be improved by
high-level plan recognition? We formulate the problem of leveraging recognized
plans to generate better top-down attention maps
\cite{gazzaniga2009,baluch2011} to improve the perception performance. We call
these top-down attention maps specifically as plan-recognition-driven attention
maps. To address this problem, we introduce the Pixel Dynamics Network. Pixel
Dynamics Network serves as an observation model, which predicts next states of
object points at each pixel location given observation of pixels and
pixel-level action feature. This is like internally learning a pixel-level
dynamics model. Pixel Dynamics Network is a kind of Convolutional Neural
Network (ConvNet), with specially-designed architecture. Therefore, Pixel
Dynamics Network could take the advantage of parallel computation of ConvNets,
while learning the pixel-level dynamics model. We further prove the equivalence
between Pixel Dynamics Network as an observation model, and the belief update
in partially observable Markov decision process (POMDP) framework. We evaluate
our Pixel Dynamics Network in event recognition tasks. We build an event
recognition system, ER-PRN, which takes Pixel Dynamics Network as a subroutine,
to recognize events based on observations augmented by plan-recognition-driven
attention.

    

### [[2010.01496] Explaining Deep Neural Networks](http://arxiv.org/abs/2010.01496)


  Deep neural networks are becoming more and more popular due to their
revolutionary success in diverse areas, such as computer vision, natural
language processing, and speech recognition. However, the decision-making
processes of these models are generally not interpretable to users. In various
domains, such as healthcare, finance, or law, it is critical to know the
reasons behind a decision made by an artificial intelligence system. Therefore,
several directions for explaining neural models have recently been explored. In
this thesis, I investigate two major directions for explaining deep neural
networks. The first direction consists of feature-based post-hoc explanatory
methods, that is, methods that aim to explain an already trained and fixed
model (post-hoc), and that provide explanations in terms of input features,
such as tokens for text and superpixels for images (feature-based). The second
direction consists of self-explanatory neural models that generate natural
language explanations, that is, models that have a built-in module that
generates explanations for the predictions of the model.

    

### [[2105.08832] A Contraction Theory Approach to Optimization Algorithms from Acceleration Flows](http://arxiv.org/abs/2105.08832)


  Much recent interest has focused on the design of optimization algorithms
from the discretization of an associated optimization flow, i.e., a system of
differential equations (ODEs) whose trajectories solve an associated
optimization problem. Such a design approach poses an important problem: how to
find a principled methodology to design and discretize appropriate ODEs. This
paper aims to provide a solution to this problem through the use of contraction
theory. We first introduce general mathematical results that explain how
contraction theory guarantees the stability of the implicit and explicit Euler
integration methods. Then, we propose a novel system of ODEs, namely the
Accelerated-Contracting-Nesterov flow, and use contraction theory to establish
it is an optimization flow with exponential convergence rate, from which the
linear convergence rate of its associated optimization algorithm is immediately
established. Remarkably, a simple explicit Euler discretization of this flow
corresponds to the Nesterov acceleration method. Finally, we present how our
approach leads to performance guarantees in the design of optimization
algorithms for time-varying optimization problems.

    

### [[2106.10989] Pre-training also Transfers Non-Robustness](http://arxiv.org/abs/2106.10989)


  Pre-training has enabled state-of-the-art results on many tasks. In spite of
its recognized contribution to generalization, we observed in this study that
pre-training also transfers adversarial non-robustness from pre-trained model
into fine-tuned model in the downstream tasks. Using image classification as an
example, we first conducted experiments on various datasets and network
backbones to uncover the adversarial non-robustness in fine-tuned model.
Further analysis was conducted on examining the learned knowledge of fine-tuned
model and standard model, and revealed that the reason leading to the
non-robustness is the non-robust features transferred from pre-trained model.
Finally, we analyzed the preference for feature learning of the pre-trained
model, explored the factors influencing robustness, and introduced a simple
robust pre-traning solution.

    

### [[2110.07018] Algebraic Reasoning of Quantum Programs via Non-Idempotent Kleene Algebra](http://arxiv.org/abs/2110.07018)


  We investigate the algebraic reasoning of quantum programs inspired by the
success of classical program analysis based on Kleene algebra. One prominent
example of such is the famous Kleene Algebra with Tests (KAT), which has
furnished both theoretical insights and practical tools. The succinctness of
algebraic reasoning would be especially desirable for scalable analysis of
quantum programs, given the involvement of exponential-size matrices in most of
the existing methods. A few key features of KAT including the idempotent law
and the nice properties of classical tests, however, fail to hold in the
context of quantum programs due to their unique quantum features, especially in
branching. We propose the Non-idempotent Kleena Algebra (NKA) as a natural
alternative and identify complete and sound semantic models for NKA as well as
their appropriate quantum interpretations. In light of applications of KAT, we
are able to demonstrate algebraic proofs in NKA of quantum compiler
optimization and the normal form of quantum while-programs. Moreover, we extend
NKA with Tests (i.e., NKAT), where tests model quantum predicates following the
rules of effect algebra, and illustrate how to encode propositional quantum
Hoare logic as NKAT theorems.

    

### [[2110.07349] A Functional Abstraction of Typed Invocation Contexts](http://arxiv.org/abs/2110.07349)


  In their paper "A Functional Abstraction of Typed Contexts", Danvy and
Filinski show how to derive a type system of the shift and reset operators from
a CPS semantics. In this paper, we show how this method scales to Felleisen's
control and prompt operators. Compared to shift and reset, control and prompt
exhibit a more dynamic behavior, in that they can manipulate a trail of
contexts surrounding the invocation of previously captured continuations. Our
key observation is that, by adopting a functional representation of trails in
the CPS semantics, we can derive a type system that encodes all and only
constraints imposed by the CPS semantics.

    

### [[2110.07493] Parallel Algebraic Effect Handlers](http://arxiv.org/abs/2110.07493)


  Algebraic effects and handlers support composable and structured control-flow
abstraction. However, existing designs of algebraic effects often require
effects to be executed sequentially. This paper studies parallel algebraic
effect handlers. In particular, we formalize {\lambda}p, an untyped lambda
calculus which models two key features, effect handlers and parallelizable
computations, the latter of which takes the form of a for expression as
inspired by the Dex programming language. We present various interesting
examples expressible in our calculus, and provide a Haskell implementation. We
hope this paper provides a basis for future designs and implementations of
parallel algebraic effect handlers.

    

### [[2110.07542] ALFRED: Virtual Memory for Intermittent Computing](http://arxiv.org/abs/2110.07542)


  We present ALFRED: a virtual memory abstraction that resolves the dichotomy
between volatile and non-volatile memory in intermittent computing.
Mixed-volatile microcontrollers allow programmers to allocate part of the
application state onto non-volatile main memory. Programmers are therefore to
explore manually the trade-off between simpler management of persistent state
against the energy overhead for non-volatile memory operations and
intermittence anomalies due to re-execution of non-idempotent code. This
approach is laborious and yields sub-optimal performance. We take a different
stand with ALFRED: we provide programmers with a virtual memory abstraction
detached from the specific volatile nature of memory and automatically
determine an efficient mapping from virtual to volatile or non-volatile memory.
Unlike existing works, ALFRED does not require programmers to learn a new
programming model or language syntax, while the mapping is entirely resolved at
compile-time, reducing the run-time energy overhead. We implement ALFRED
through a series of program machine-level code transformations. Compared to
existing systems, we demonstrate that ALFRED reduces energy consumption by up
to two orders of magnitude given a fixed workload. This enables the workloads
to finish sooner, as the use of available energy shifts from ensuring forward
progress to useful application processing.

    

### [[2105.01632] Solo: A Lightweight Static Analysis for Differential Privacy](http://arxiv.org/abs/2105.01632)


  All current approaches for statically enforcing differential privacy in
higher order languages make use of either linear or relational refinement
types. A barrier to adoption for these approaches is the lack of support for
expressing these "fancy types" in mainstream programming languages. For
example, no mainstream language supports relational refinement types, and
although Rust and modern versions of Haskell both employ some linear typing
techniques, they are inadequate for embedding enforcement of differential
privacy, which requires "full" linear types a la Girard. We propose a new type
system that enforces differential privacy, avoids the use of linear and
relational refinement types, and can be easily embedded in mainstream richly
typed programming languages such as Scala, OCaml and Haskell. We demonstrate
such an embedding in Haskell, demonstrate its expressiveness on case studies,
and prove that our type-based enforcement of differential privacy is sound.

    