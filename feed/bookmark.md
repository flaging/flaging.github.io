
## 2021-12-21

### [[2112.09769] Self-Organizing Networks in the 6G Era: State-of-the-Art, Opportunities, Challenges, and Future Trends](http://arxiv.org/abs/2112.09769)


  Self-organizing networks (SONs) need to be endowed with self-coordination
capabilities to manage the complex relations between their internal components
and to avoid their destructive interactions. Existing communication
technologies commonly implement responsive self-coordination mechanisms that
can be very slow in run-time situations. The sixth generation (6G) networks,
being in their early stages of research and standardization activities, open
new opportunities to opt for a design-driven approach when developing
self-coordination capabilities. This can be achieved through the use of hybrid
weakly coupled SON designs. In this article, we review the history of SONs
including the inherent self-coordination feature. We then delve into the
concept of hybrid SONs (H-SONs), and we summarize the challenges,
opportunities, and future trends for H-SON development. We provide a
comprehensive collection of standardization activities and recommendations,
discussing the key contributions and potential work to continue the evolution
and push for a wide adoption of the H-SON paradigm. More importantly, we
propose that H-SONs must be weakly coupled networks, i.e., the various feedback
loops must be almost isolated from each other to improve the stability and to
avoid chaotic situations. We finally conclude the paper with the key hints
about the future landscape and the key drivers of 6G H-SONs.

    

### [[2112.09981] Multi-step LRU: SIMD-based Cache Replacement for Lower Overhead and Higher Precision](http://arxiv.org/abs/2112.09981)


  A key-value cache is a key component of many services to provide low-latency
and high-throughput data accesses to a huge amount of data. To improve the
end-to-end performance of such services, a key-value cache must achieve a high
cache hit ratio with high throughput. In this paper, we propose a new cache
replacement algorithm, multi-step LRU, which achieves high throughput by
efficiently exploiting SIMD instructions without using per-item additional
memory (LRU metadata) to record information such as the last access timestamp.
For a small set of items that can fit within a vector register, SIMD-based LRU
management without LRU metadata is known (in-vector LRU). It remembers the
access history by reordering items in one vector using vector shuffle
instruction. In-vector LRU alone cannot be used for a caching system since it
can manage only few items. Set-associative cache is a straightforward way to
build a large cache using in-vector LRU as a building block. However, a naive
set-associative cache based on in-vector LRU has a poorer cache hit ratio than
the original LRU although it can achieve a high throughput. Our multi-step LRU
enhances naive set-associative cache based on in-vector LRU for improving cache
accuracy by taking both access frequency and access recency of items into
account while keeping the efficiency by SIMD instructions. Our results indicate
that multi-step LRU outperforms the original LRU and GCLOCK algorithms in terms
of both execution speed and cache hit ratio. Multi-step LRU improves the cache
hit ratios over the original LRU by implicitly taking access frequency of items
as well as access recency into account. The cache hit ratios of multi-step LRU
are similar to those of ARC, which achieves a higher a cache hit ratio in a
tradeoff for using more LRU metadata.

    

### [[2112.10086] Heterogeneous Transformer: A Scale Adaptable Neural Network Architecture for Device Activity Detection](http://arxiv.org/abs/2112.10086)


  To support the modern machine-type communications, a crucial task during the
random access phase is device activity detection, which is to detect the active
devices from a large number of potential devices based on the received signal
at the access point. By utilizing the statistical properties of the channel,
state-of-the-art covariance based methods have been demonstrated to achieve
better activity detection performance than compressed sensing based methods.
However, covariance based methods require to solve a high dimensional nonconvex
optimization problem by updating the estimate of the activity status of each
device sequentially. Since the number of updates is proportional to the device
number, the computational complexity and delay make the iterative updates
difficult for real-time implementation especially when the device number scales
up. Inspired by the success of deep learning for real-time inference, this
paper proposes a learning based method with a customized heterogeneous
transformer architecture for device activity detection. By adopting an
attention mechanism in the architecture design, the proposed method is able to
extract the relevance between device pilots and received signal, is permutation
equivariant with respect to devices, and is scale adaptable to different
numbers of devices. Simulation results demonstrate that the proposed method
achieves better activity detection performance with much shorter computation
time than state-of-the-art covariance approach, and generalizes well to
different numbers of devices, BS-antennas, and different signal-to-noise
ratios.

    

### [[2112.10339] Smart Home: Application using HTTP and MQTT as Communication Protocols](http://arxiv.org/abs/2112.10339)


  This study discloses the development of a solution for realizing a smart home
in the post COVID-19 era using the Internet of Things domain knowledge.
COVID-19 outbreak has been catastrophic and impacted everyone's lives due to
its rapid transmission from one body to another. This study aims to reduce
virus transmission by eliminating the need to touch any common-point surface in
a home, such as switches, doorknobs, and remotes. We provide a generic solution
by coupling things with the internet to control them remotely. The project aims
to showcase a working solution to controlling devices like smart bulbs, smart
fans, smart ACs, and smart door locks in our self-developed emulator over WWW
securely using two different protocols, viz., HTTP and MQTT-over-WSS.
Additionally, intent authentication over HTTP is based on digital signature
that is demonstrated using RSA (encryption) and MD5 (hashing) when the system
is deployed in insecure environment(s). RESTful API deployed on AWS EC2 is used
to realize HTTP communication protocol, and MQTT is realized using AWS IoT
service. The developed project can be applied to any smart home setting like a
hotel or public place using AWS IoT, Lambda, or similar infrastructure as a
broker.

    

### [[2112.10352] Transformation from 5G for Verticals Towards a 6G-enabled Internet of Verticals](http://arxiv.org/abs/2112.10352)


  5G will enable and greatly accelerate digital transformation of vertical
sectors. In the longer term this will evolve from per-vertical connectivity
services in 5G to the emergence of the 6G-enabled Internet of Verticals
(6G-IoV). In this paper we describe and examine enabling technologies for this
further transformation of verticals and examine some examples of 6G-IoV, next
generation cloud manufacturing and manufacturing as a service, next generation
smart energy grids and the Internet of Robotics (IoR).

    

### [[2112.10387] Performance analysis of SDN controllers: POX, Floodlight and Opendaylight](http://arxiv.org/abs/2112.10387)


  The IP network is time-consuming for configuration and troubleshooting
because it requires access to every device command line interface (CLI) or
Graphical User Interface (GUI). In other words, the control plane (gathering
information to forward data) and data plane (data forwarding plane) are run on
every intermediary device. To solve this problem, the software defined network
emerged and separated the control plane (done by software) from the data plane
(done by hardware) [1]. In addition, the control plane is operated in the
central point named the controller. There are many controller software programs
in simulation and production network environments. In this paper, we compare
three open source controllers called POX, Floodlight and Opendaylight (ODL) in
simulation network created by MININET SDN emulation program in terms of TCP and
UDP throughput and average Round Trip Time (RTT) of the first packet of the
flow in the mesh and tree topology.

    

### [[2112.10389] Decentralized Stochastic Proximal Gradient Descent with Variance Reduction over Time-varying Networks](http://arxiv.org/abs/2112.10389)


  In decentralized learning, a network of nodes cooperate to minimize an
overall objective function that is usually the finite-sum of their local
objectives, and incorporates a non-smooth regularization term for the better
generalization ability. Decentralized stochastic proximal gradient (DSPG)
method is commonly used to train this type of learning models, while the
convergence rate is retarded by the variance of stochastic gradients. In this
paper, we propose a novel algorithm, namely DPSVRG, to accelerate the
decentralized training by leveraging the variance reduction technique. The
basic idea is to introduce an estimator in each node, which tracks the local
full gradient periodically, to correct the stochastic gradient at each
iteration. By transforming our decentralized algorithm into a centralized
inexact proximal gradient algorithm with variance reduction, and controlling
the bounds of error sequences, we prove that DPSVRG converges at the rate of
$O(1/T)$ for general convex objectives plus a non-smooth term with $T$ as the
number of iterations, while DSPG converges at the rate $O(\frac{1}{\sqrt{T}})$.
Our experiments on different applications, network topologies and learning
models demonstrate that DPSVRG converges much faster than DSPG, and the loss
function of DPSVRG decreases smoothly along with the training epochs.

    

### [[2112.10400] Sending Timely Status Updates through Channel with Random Delay via Online Learning](http://arxiv.org/abs/2112.10400)


  In this work, we study a status update system with a source node sending
timely information to the destination through a channel with random delay. We
measure the timeliness of the information stored at the receiver via the Age of
Information (AoI), the time elapsed since the freshest sample stored at the
receiver is generated. The goal is to design a sampling strategy that minimizes
the total cost of the expected time average AoI and sampling cost in the
absence of transmission delay statistics. We reformulate the total cost
minimization problem as the optimization of a renewal-reward process, and
propose an online sampling strategy based on the Robbins-Monro algorithm. We
show that, when the transmission delay is bounded, the expected time average
total cost obtained by the proposed online algorithm converges to the minimum
cost when $K$ goes to infinity, and the optimality gap decays with rate
$\mathcal{O}\left(\ln K/K\right)$, where $K$ is the number of samples we have
taken. Simulation results validate the performance of our proposed algorithm.

    

### [[2102.00542] Follow the Scent: Defeating IPv6 Prefix Rotation Privacy](http://arxiv.org/abs/2102.00542)


  IPv6's large address space allows ample freedom for choosing and assigning
addresses. To improve client privacy and resist IP-based tracking, standardized
techniques leverage this large address space, including privacy extensions and
provider prefix rotation. Ephemeral and dynamic IPv6 addresses confound not
only tracking and traffic correlation attempts, but also traditional network
measurements, logging, and defense mechanisms. We show that the intended
anti-tracking capability of these widely deployed mechanisms is unwittingly
subverted by edge routers using legacy IPv6 addressing schemes that embed
unique identifiers.
We develop measurement techniques that exploit these legacy devices to make
tracking such moving IPv6 clients feasible by combining intelligent search
space reduction with modern high-speed active probing. Via an Internet-wide
measurement campaign, we discover more than 9M affected edge routers and
approximately 13k /48 prefixes employing prefix rotation in hundreds of ASes
worldwide. We mount a six-week campaign to characterize the size and dynamics
of these deployed IPv6 rotation pools, and demonstrate via a case study the
ability to remotely track client address movements over time. We responsibly
disclosed our findings to equipment manufacturers, at least one of which
subsequently changed their default addressing logic.

    

### [[2107.00957] Ascent Similarity Caching with Approximate Indexes](http://arxiv.org/abs/2107.00957)


  Similarity search is a key operation in multimedia retrieval systems and
recommender systems, and it will play an important role also for future machine
learning and augmented reality applications. When these systems need to serve
large objects with tight delay constraints, edge servers close to the end-user
can operate as similarity caches to speed up the retrieval. In this paper we
present AÇAI, a new similarity caching policy which improves on the state
of the art by using (i) an (approximate) index for the whole catalog to decide
which objects to serve locally and which to retrieve from the remote server,
and (ii) a mirror ascent algorithm to update the set of local objects with
strong guarantees even when the request process does not exhibit any
statistical regularity.

    

### [[2111.07764] Fidelity-Guarantee Entanglement Routing in Quantum Networks](http://arxiv.org/abs/2111.07764)


  Entanglement routing establishes remote entanglement connection between two
arbitrary nodes, which is one of the most important functions in quantum
networks. The existing routing mechanisms mainly improve the robustness and
throughput facing the failure of entanglement generations, which, however,
rarely include the considerations on the most important metric to evaluate the
quality of connection, entanglement fidelity. To solve this problem, we propose
purification-enabled entanglement routing designs to provide fidelity guarantee
for multiple Source-Destination (SD) pairs in quantum networks. In our
proposal, we first consider the single S-D pair scenario and design an
iterative routing algorithm, Q-PATH, to find the optimal purification decisions
along the routing path with minimum entangled pair cost. Further, a
low-complexity routing algorithm using an extended Dijkstra algorithm, Q-LEAP,
is designed to reduce the computational complexity by using a simple but
effective purification decision method. Then we consider the common scenario
with multiple S-D pairs and design a greedy-based algorithm considering
resource allocation and rerouting process for multiple routing requests. To
verify the effectiveness and superiority of the proposed algorithms, extensive
simulations are conducted, and the simulation results show that the proposed
algorithms not only can provide fidelity-guarantee routing solutions, but also
has superior performance in terms of throughput, fidelity of end-to-end
entanglement connection, and resource utilization ratio, compared with the
existing routing scheme.

    

### [[2112.09693] Can uncertainty boost the reliability of AI-based diagnostic methods in digital pathology?](http://arxiv.org/abs/2112.09693)


  Deep learning (DL) has shown great potential in digital pathology
applications. The robustness of a diagnostic DL-based solution is essential for
safe clinical deployment. In this work we evaluate if adding uncertainty
estimates for DL predictions in digital pathology could result in increased
value for the clinical applications, by boosting the general predictive
performance or by detecting mispredictions. We compare the effectiveness of
model-integrated methods (MC dropout and Deep ensembles) with a model-agnostic
approach (Test time augmentation, TTA). Moreover, four uncertainty metrics are
compared. Our experiments focus on two domain shift scenarios: a shift to a
different medical center and to an underrepresented subtype of cancer. Our
results show that uncertainty estimates can add some reliability and reduce
sensitivity to classification threshold selection. While advanced metrics and
deep ensembles perform best in our comparison, the added value over simpler
metrics and TTA is small. Importantly, the benefit of all evaluated uncertainty
estimation methods is diminished by domain shift.

    

### [[2112.09694] Interpretable and Interactive Deep Multiple Instance Learning for Dental Caries Classification in Bitewing X-rays](http://arxiv.org/abs/2112.09694)


  We propose a simple and efficient image classification architecture based on
deep multiple instance learning, and apply it to the challenging task of caries
detection in dental radiographs. Technically, our approach contributes in two
ways: First, it outputs a heatmap of local patch classification probabilities
despite being trained with weak image-level labels. Second, it is amenable to
learning from segmentation labels to guide training. In contrast to existing
methods, the human user can faithfully interpret predictions and interact with
the model to decide which regions to attend to. Experiments are conducted on a
large clinical dataset of $\sim$38k bitewings ($\sim$316k teeth), where we
achieve competitive performance compared to various baselines. When guided by
an external caries segmentation model, a significant improvement in
classification and localization performance is observed.

    

### [[2112.09697] On the Evolution of the MCTS Upper Confidence Bounds for Trees by Means of Evolutionary Algorithms in the Game of Carcassonne](http://arxiv.org/abs/2112.09697)


  Monte Carlo Tree Search (MCTS) is a sampling best-first method to search for
optimal decisions. The MCTS's popularity is based on its extraordinary results
in the challenging two-player based game Go, a game considered much harder than
Chess and that until very recently was considered infeasible for Artificial
Intelligence methods. The success of MCTS depends heavily on how the tree is
built and the selection process plays a fundamental role in this. One
particular selection mechanism that has proved to be reliable is based on the
Upper Confidence Bounds for Trees, commonly referred as UCT. The UCT attempts
to nicely balance exploration and exploitation by considering the values stored
in the statistical tree of the MCTS. However, some tuning of the MCTS UCT is
necessary for this to work well. In this work, we use Evolutionary Algorithms
(EAs) to evolve mathematical expressions with the goal to substitute the UCT
mathematical expression. We compare our proposed approach, called Evolution
Strategy in MCTS (ES-MCTS) against five variants of the MCTS UCT, three
variants of the star-minimax family of algorithms as well as a random
controller in the Game of Carcassonne. We also use a variant of our proposed
EA-based controller, dubbed ES partially integrated in MCTS. We show how the
ES-MCTS controller, is able to outperform all these 10 intelligent controllers,
including robust MCTS UCT controllers.

    

### [[2112.09727] Rank4Class: A Ranking Formulation for Multiclass Classification](http://arxiv.org/abs/2112.09727)


  Multiclass classification (MCC) is a fundamental machine learning problem
which aims to classify each instance into one of a predefined set of classes.
Given an instance, a classification model computes a score for each class, all
of which are then used to sort the classes. The performance of a classification
model is usually measured by Top-K Accuracy/Error (e.g., K=1 or 5). In this
paper, we do not aim to propose new neural representation learning models as
most recent works do, but to show that it is easy to boost MCC performance with
a novel formulation through the lens of ranking. In particular, by viewing MCC
as to rank classes for an instance, we first argue that ranking metrics, such
as Normalized Discounted Cumulative Gain (NDCG), can be more informative than
existing Top-K metrics. We further demonstrate that the dominant neural MCC
architecture can be formulated as a neural ranking framework with a specific
set of design choices. Based on such generalization, we show that it is
straightforward and intuitive to leverage techniques from the rich information
retrieval literature to improve the MCC performance out of the box. Extensive
empirical results on both text and image classification tasks with diverse
datasets and backbone models (e.g., BERT and ResNet for text and image
classification) show the value of our proposed framework.

    

### [[2112.09738] Improving Ethical Outcomes with Machine-in-the-Loop: Broadening Human Understanding of Data Annotations](http://arxiv.org/abs/2112.09738)


  We introduce a machine-in-the-loop pipeline that aims to address root causes
of unwanted bias in natural language based supervised machine learning tasks in
the education domain. Learning from the experiences of students is foundational
for education researchers, and academic administrators. 21st-century skills
learned from experience are becoming a core part of college and career
readiness as well as the hiring process in the new knowledge economy.
Minoritized students demonstrate these skills in their daily lives, but
documenting, assessing, and validating these skills is a huge problem for
educational institutions. As an equity focused online platform, LivedX
translates minoritized students' lived experiences into the 21st century
skills, issues micro-credentials, and creates personal 21st century skills
portfolio. To automate the micro credential mining from the natural language
texts received from the students' submitted essays, we employed a bag-of-word
model to construct a multi-output classifier. Despite our goal, our model
initially exacerbated disparate impact on minoritized students. We used a
machine-in-the-loop model development pipeline to address the problem and
refine the aforementioned model to ensure fairness in its prediction.

    

### [[2112.09741] Neurashed: A Phenomenological Model for Imitating Deep Learning Training](http://arxiv.org/abs/2112.09741)


  To advance deep learning methodologies in the next decade, a theoretical
framework for reasoning about modern neural networks is needed. While efforts
are increasing toward demystifying why deep learning is so effective, a
comprehensive picture remains lacking, suggesting that a better theory is
possible. We argue that a future deep learning theory should inherit three
characteristics: a \textit{hierarchically} structured network architecture,
parameters \textit{iteratively} optimized using stochastic gradient-based
methods, and information from the data that evolves \textit{compressively}. As
an instantiation, we integrate these characteristics into a graphical model
called \textit{neurashed}. This model effectively explains some common
empirical patterns in deep learning. In particular, neurashed enables insights
into implicit regularization, information bottleneck, and local elasticity.
Finally, we discuss how neurashed can guide the development of deep learning
theories.

    

### [[2112.09745] Interpretable Data-Based Explanations for Fairness Debugging](http://arxiv.org/abs/2112.09745)


  A wide variety of fairness metrics and eXplainable Artificial Intelligence
(XAI) approaches have been proposed in the literature to identify bias in
machine learning models that are used in critical real-life contexts. However,
merely reporting on a model's bias, or generating explanations using existing
XAI techniques is insufficient to locate and eventually mitigate sources of
bias. In this work, we introduce Gopher, a system that produces compact,
interpretable, and causal explanations for bias or unexpected model behavior by
identifying coherent subsets of the training data that are root-causes for this
behavior. Specifically, we introduce the concept of causal responsibility that
quantifies the extent to which intervening on training data by removing or
updating subsets of it can resolve the bias. Building on this concept, we
develop an efficient approach for generating the top-k patterns that explain
model bias that utilizes techniques from the ML community to approximate causal
responsibility and uses pruning rules to manage the large search space for
patterns. Our experimental evaluation demonstrates the effectiveness of Gopher
in generating interpretable explanations for identifying and debugging sources
of bias.

    

### [[2112.09746] Supervised Multivariate Learning with Simultaneous Feature Auto-grouping and Dimension Reduction](http://arxiv.org/abs/2112.09746)


  Modern high-dimensional methods often adopt the ``bet on sparsity''
principle, while in supervised multivariate learning statisticians may face
``dense'' problems with a large number of nonzero coefficients. This paper
proposes a novel clustered reduced-rank learning (CRL) framework that imposes
two joint matrix regularizations to automatically group the features in
constructing predictive factors. CRL is more interpretable than low-rank
modeling and relaxes the stringent sparsity assumption in variable selection.
In this paper, new information-theoretical limits are presented to reveal the
intrinsic cost of seeking for clusters, as well as the blessing from
dimensionality in multivariate learning. Moreover, an efficient optimization
algorithm is developed, which performs subspace learning and clustering with
guaranteed convergence. The obtained fixed-point estimators, though not
necessarily globally optimal, enjoy the desired statistical accuracy beyond the
standard likelihood setup under some regularity conditions. Moreover, a new
kind of information criterion, as well as its scale-free form, is proposed for
cluster and rank selection, and has a rigorous theoretical support without
assuming an infinite sample size. Extensive simulations and real-data
experiments demonstrate the statistical accuracy and interpretability of the
proposed method.

    

### [[2112.09752] Set Twister for Single-hop Node Classification](http://arxiv.org/abs/2112.09752)


  Node classification is a central task in relational learning, with the
current state-of-the-art hinging on two key principles: (i) predictions are
permutation-invariant to the ordering of a node's neighbors, and (ii)
predictions are a function of the node's $r$-hop neighborhood topology and
attributes, $r \geq 2$. Both graph neural networks and collective inference
methods (e.g., belief propagation) rely on information from up to $r$-hops
away. In this work, we study if the use of more powerful permutation-invariant
functions can sometimes avoid the need for classifiers to collect information
beyond $1$-hop. Towards this, we introduce a new architecture, the Set Twister,
which generalizes DeepSets (Zaheer et al., 2017), a simple and widely-used
permutation-invariant representation. Set Twister theoretically increases
expressiveness of DeepSets, allowing it to capture higher-order dependencies,
while keeping its simplicity and low computational cost. Empirically, we see
accuracy improvements of Set Twister over DeepSets as well as a variety of
graph neural networks and collective inference schemes in several tasks, while
showcasing its implementation simplicity and computational efficiency.

    

### [[2112.09754] Probabilistic Inverse Optimal Transport](http://arxiv.org/abs/2112.09754)


  Optimal transport (OT) formalizes the problem of finding an optimal coupling
between probability measures given a cost matrix. The inverse problem of
inferring the cost given a coupling is Inverse Optimal Transport (IOT). IOT is
less well understood than OT. We formalize and systematically analyze the
properties of IOT using tools from the study of entropy-regularized OT.
Theoretical contributions include characterization of the manifold of
cross-ratio equivalent costs, the implications of model priors, and derivation
of an MCMC sampler. Empirical contributions include visualizations of
cross-ratio equivalent effect on basic examples and simulations validating
theoretical results.

    

### [[2112.09788] Heavy-tailed denoising score matching](http://arxiv.org/abs/2112.09788)


  Score-based model research in the last few years has produced state of the
art generative models by employing Gaussian denoising score-matching (DSM).
However, the Gaussian noise assumption has several high-dimensional
limitations, motivating a more concrete route toward even higher dimension PDF
estimation in future. We outline this limitation, before extending the theory
to a broader family of noising distributions -- namely, the generalised normal
distribution. To theoretically ground this, we relax a key assumption in
(denoising) score matching theory, demonstrating that distributions which are
differentiable \textit{almost everywhere} permit the same objective
simplification as Gaussians. For noise vector length distributions, we
demonstrate favourable concentration of measure in the high-dimensional spaces
prevalent in deep learning. In the process, we uncover a skewed noise vector
length distribution and develop an iterative noise scaling algorithm to
consistently initialise the multiple levels of noise in annealed Langevin
dynamics. On the practical side, our use of heavy-tailed DSM leads to improved
score estimation, controllable sampling convergence, and more balanced
unconditional generative performance for imbalanced datasets.

    

### [[2112.09792] A data-centric weak supervised learning for highway traffic incident detection](http://arxiv.org/abs/2112.09792)


  Using the data from loop detector sensors for near-real-time detection of
traffic incidents in highways is crucial to averting major traffic congestion.
While recent supervised machine learning methods offer solutions to incident
detection by leveraging human-labeled incident data, the false alarm rate is
often too high to be used in practice. Specifically, the inconsistency in the
human labeling of the incidents significantly affects the performance of
supervised learning models. To that end, we focus on a data-centric approach to
improve the accuracy and reduce the false alarm rate of traffic incident
detection on highways. We develop a weak supervised learning workflow to
generate high-quality training labels for the incident data without the ground
truth labels, and we use those generated labels in the supervised learning
setup for final detection. This approach comprises three stages. First, we
introduce a data preprocessing and curation pipeline that processes traffic
sensor data to generate high-quality training data through leveraging labeling
functions, which can be domain knowledge-related or simple heuristic rules.
Second, we evaluate the training data generated by weak supervision using three
supervised learning models -- random forest, k-nearest neighbors, and a support
vector machine ensemble -- and long short-term memory classifiers. The results
show that the accuracy of all of the models improves significantly after using
the training data generated by weak supervision. Third, we develop an online
real-time incident detection approach that leverages the model ensemble and the
uncertainty quantification while detecting incidents. Overall, we show that our
proposed weak supervised learning workflow achieves a high incident detection
rate (0.90) and low false alarm rate (0.08).

    

### [[2112.09794] Coded Consensus Monte Carlo: Robust One-Shot Distributed Bayesian Learning with Stragglers](http://arxiv.org/abs/2112.09794)


  This letter studies distributed Bayesian learning in a setting encompassing a
central server and multiple workers by focusing on the problem of mitigating
the impact of stragglers. The standard one-shot, or embarrassingly parallel,
Bayesian learning protocol known as consensus Monte Carlo (CMC) is generalized
by proposing two straggler-resilient solutions based on grouping and coding.
The proposed methods, referred to as Group-based CMC (G-CMC) and Coded CMC
(C-CMC), leverage redundant computing at the workers in order to enable the
estimation of global posterior samples at the server based on partial outputs
from the workers. Simulation results show that C-CMC may outperform G-GCMC for
a small number of workers, while G-CMC is generally preferable for a larger
number of workers.

    

### [[2112.09796] AutoTransfer: Subject Transfer Learning with Censored Representations on Biosignals Data](http://arxiv.org/abs/2112.09796)


  We provide a regularization framework for subject transfer learning in which
we seek to train an encoder and classifier to minimize classification loss,
subject to a penalty measuring independence between the latent representation
and the subject label. We introduce three notions of independence and
corresponding penalty terms using mutual information or divergence as a proxy
for independence. For each penalty term, we provide several concrete estimation
algorithms, using analytic methods as well as neural critic functions. We
provide a hands-off strategy for applying this diverse family of regularization
algorithms to a new dataset, which we call "AutoTransfer". We evaluate the
performance of these individual regularization strategies and our AutoTransfer
method on EEG, EMG, and ECoG datasets, showing that these approaches can
improve subject transfer learning for challenging real-world datasets.

    

### [[2112.09802] Improving Multi-Domain Generalization through Domain Re-labeling](http://arxiv.org/abs/2112.09802)


  Domain generalization (DG) methods aim to develop models that generalize to
settings where the test distribution is different from the training data. In
this paper, we focus on the challenging problem of multi-source zero-shot DG,
where labeled training data from multiple source domains is available but with
no access to data from the target domain. Though this problem has become an
important topic of research, surprisingly, the simple solution of pooling all
source data together and training a single classifier is highly competitive on
standard benchmarks. More importantly, even sophisticated approaches that
explicitly optimize for invariance across different domains do not necessarily
provide non-trivial gains over ERM. In this paper, for the first time, we study
the important link between pre-specified domain labels and the generalization
performance. Using a motivating case-study and a new variant of a
distributional robust optimization algorithm, GroupDRO++, we first demonstrate
how inferring custom domain groups can lead to consistent improvements over the
original domain labels that come with the dataset. Subsequently, we introduce a
general approach for multi-domain generalization, MulDEns, that uses an
ERM-based deep ensembling backbone and performs implicit domain re-labeling
through a meta-optimization algorithm. Using empirical studies on multiple
standard benchmarks, we show that MulDEns does not require tailoring the
augmentation strategy or the training process specific to a dataset,
consistently outperforms ERM by significant margins, and produces
state-of-the-art generalization performance, even when compared to existing
methods that exploit the domain labels.

    

### [[2112.09810] Meta Propagation Networks for Graph Few-shot Semi-supervised Learning](http://arxiv.org/abs/2112.09810)


  Inspired by the extensive success of deep learning, graph neural networks
(GNNs) have been proposed to learn expressive node representations and
demonstrated promising performance in various graph learning tasks. However,
existing endeavors predominately focus on the conventional semi-supervised
setting where relatively abundant gold-labeled nodes are provided. While it is
often impractical due to the fact that data labeling is unbearably laborious
and requires intensive domain knowledge, especially when considering the
heterogeneity of graph-structured data. Under the few-shot semi-supervised
setting, the performance of most of the existing GNNs is inevitably undermined
by the overfitting and oversmoothing issues, largely owing to the shortage of
labeled data. In this paper, we propose a decoupled network architecture
equipped with a novel meta-learning algorithm to solve this problem. In
essence, our framework Meta-PN infers high-quality pseudo labels on unlabeled
nodes via a meta-learned label propagation strategy, which effectively augments
the scarce labeled data while enabling large receptive fields during training.
Extensive experiments demonstrate that our approach offers easy and substantial
performance gains compared to existing techniques on various benchmark
datasets.

    

### [[2112.09815] Gradient-based Novelty Detection Boosted by Self-supervised Binary Classification](http://arxiv.org/abs/2112.09815)


  Novelty detection aims to automatically identify out-of-distribution (OOD)
data, without any prior knowledge of them. It is a critical step in data
monitoring, behavior analysis and other applications, helping enable continual
learning in the field. Conventional methods of OOD detection perform
multi-variate analysis on an ensemble of data or features, and usually resort
to the supervision with OOD data to improve the accuracy. In reality, such
supervision is impractical as one cannot anticipate the anomalous data. In this
paper, we propose a novel, self-supervised approach that does not rely on any
pre-defined OOD data: (1) The new method evaluates the Mahalanobis distance of
the gradients between the in-distribution and OOD data. (2) It is assisted by a
self-supervised binary classifier to guide the label selection to generate the
gradients, and maximize the Mahalanobis distance. In the evaluation with
multiple datasets, such as CIFAR-10, CIFAR-100, SVHN and TinyImageNet, the
proposed approach consistently outperforms state-of-the-art supervised and
unsupervised methods in the area under the receiver operating characteristic
(AUROC) and area under the precision-recall curve (AUPR) metrics. We further
demonstrate that this detector is able to accurately learn one OOD class in
continual learning.

    

### [[2112.09820] GPEX, A Framework For Interpreting Artificial Neural Networks](http://arxiv.org/abs/2112.09820)


  Machine learning researchers have long noted a trade-off between
interpretability and prediction performance. On the one hand, traditional
models are often interpretable to humans but they cannot achieve high
prediction performances. At the opposite end of the spectrum, deep models can
achieve state-of-the-art performances in many tasks. However, deep models'
predictions are known to be uninterpretable to humans. In this paper we present
a framework that shortens the gap between the two aforementioned groups of
methods. Given an artificial neural network (ANN), our method finds a Gaussian
process (GP) whose predictions almost match those of the ANN. As GPs are highly
interpretable, we use the trained GP to explain the ANN's decisions. We use our
method to explain ANNs' decisions on may datasets. The explanations provide
intriguing insights about the ANNs' decisions. With the best of our knowledge,
our inference formulation for GPs is the first one in which an ANN and a
similarly behaving Gaussian process naturally appear. Furthermore, we examine
some of the known theoretical conditions under which an ANN is interpretable by
GPs. Some of those theoretical conditions are too restrictive for modern
architectures. However, we hypothesize that only a subset of those theoretical
conditions are sufficient. Finally, we implement our framework as a publicly
available tool called GPEX. Given any pytorch feed-forward module, GPEX allows
users to interpret any ANN subcomponent of the module effortlessly and without
having to be involved in the inference algorithm. GPEX is publicly available
online:this http URL


### [[2112.09822] Multimeasurement Generative Models](http://arxiv.org/abs/2112.09822)


  We formally map the problem of sampling from an unknown distribution with
density $p_X$ in $\mathbb{R}^d$ to the problem of learning and sampling
$p_\mathbf{Y}$ in $\mathbb{R}^{Md}$ obtained by convolving $p_X$ with a fixed
factorial kernel: $p_\mathbf{Y}$ is referred to as M-density and the factorial
kernel as multimeasurement noise model (MNM). The M-density is smoother than
$p_X$, easier to learn and sample from, yet for large $M$ the two problems are
mathematically equivalent since $X$ can be estimated exactly given
$\mathbf{Y}=\mathbf{y}$ using the Bayes estimator
$\widehat{x}(\mathbf{y})=\mathbb{E}[X\vert\mathbf{Y}=\mathbf{y}]$. To formulate
the problem, we derive $\widehat{x}(\mathbf{y})$ for Poisson and Gaussian MNMs
expressed in closed form in terms of unnormalized $p_\mathbf{Y}$. This leads to
a simple least-squares objective for learning parametric energy and score
functions. We present various parametrization schemes of interest, including
one in which studying Gaussian M-densities directly leads to multidenoising
autoencoders--this is the first theoretical connection made between denoising
autoencoders and empirical Bayes in the literature. Samples from $p_X$ are
obtained by walk-jump sampling (Saremi & Hyvarinen, 2019) via underdamped
Langevin MCMC (walk) to sample from $p_\mathbf{Y}$ and the multimeasurement
Bayes estimation of $X$ (jump). We study permutation invariant Gaussian
M-densities on MNIST, CIFAR-10, and FFHQ-256 datasets, and demonstrate the
effectiveness of this framework for realizing fast-mixing stable Markov chains
in high dimensions.

    

### [[2112.09824] Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better](http://arxiv.org/abs/2112.09824)


  Federated learning (FL) enables distribution of machine learning workloads
from the cloud to resource-limited edge devices. Unfortunately, current deep
networks remain not only too compute-heavy for inference and training on edge
devices, but also too large for communicating updates over
bandwidth-constrained networks. In this paper, we develop, implement, and
experimentally validate a novel FL framework termed Federated Dynamic Sparse
Training (FedDST) by which complex neural networks can be deployed and trained
with substantially improved efficiency in both on-device computation and
in-network communication. At the core of FedDST is a dynamic process that
extracts and trains sparse sub-networks from the target full network. With this
scheme, "two birds are killed with one stone:" instead of full models, each
client performs efficient training of its own sparse networks, and only sparse
networks are transmitted between devices and the cloud. Furthermore, our
results reveal that the dynamic sparsity during FL training more flexibly
accommodates local heterogeneity in FL agents than the fixed, shared sparse
masks. Moreover, dynamic sparsity naturally introduces an "in-time
self-ensembling effect" into the training dynamics and improves the FL
performance even over dense training. In a realistic and challenging non i.i.d.
FL setting, FedDST consistently outperforms competing algorithms in our
experiments: for instance, at any fixed upload data cap on non-iid CIFAR-10, it
gains an impressive accuracy advantage of 10% over FedAvgM when given the same
upload data cap; the accuracy gap remains 3% even when FedAvgM is given 2x the
upload data cap, further demonstrating efficacy of FedDST. Code is available
at: this https URL.

    

### [[2112.09834] Improving the performance of bagging ensembles for data streams through mini-batching](http://arxiv.org/abs/2112.09834)


  Often, machine learning applications have to cope with dynamic environments
where data are collected in the form of continuous data streams with
potentially infinite length and transient behavior. Compared to traditional
(batch) data mining, stream processing algorithms have additional requirements
regarding computational resources and adaptability to data evolution. They must
process instances incrementally because the data's continuous flow prohibits
storing data for multiple passes. Ensemble learning achieved remarkable
predictive performance in this scenario. Implemented as a set of (several)
individual classifiers, ensembles are naturally amendable for task parallelism.
However, the incremental learning and dynamic data structures used to capture
the concept drift increase the cache misses and hinder the benefit of
parallelism. This paper proposes a mini-batching strategy that can improve
memory access locality and performance of several ensemble algorithms for
stream mining in multi-core environments. With the aid of a formal framework,
we demonstrate that mini-batching can significantly decrease the reuse distance
(and the number of cache misses). Experiments on six different state-of-the-art
ensemble algorithms applying four benchmark datasets with varied
characteristics show speedups of up to 5X on 8-core processors. These benefits
come at the expense of a small reduction in predictive performance.

    

### [[2112.09836] Creativity of AI: Automatic Symbolic Option Discovery for Facilitating Deep Reinforcement Learning](http://arxiv.org/abs/2112.09836)


  Despite of achieving great success in real life, Deep Reinforcement Learning
(DRL) is still suffering from three critical issues, which are data efficiency,
lack of the interpretability and transferability. Recent research shows that
embedding symbolic knowledge into DRL is promising in addressing those
challenges. Inspired by this, we introduce a novel deep reinforcement learning
framework with symbolic options. This framework features a loop training
procedure, which enables guiding the improvement of policy by planning with
action models and symbolic options learned from interactive trajectories
automatically. The learned symbolic options alleviate the dense requirement of
expert domain knowledge and provide inherent interpretability of policies.
Moreover, the transferability and data efficiency can be further improved by
planning with the action models. To validate the effectiveness of this
framework, we conduct experiments on two domains, Montezuma's Revenge and
Office World, respectively. The results demonstrate the comparable performance,
improved data efficiency, interpretability and transferability.

    

### [[2112.09842] Manifold embedding data-driven mechanics](http://arxiv.org/abs/2112.09842)


  This article introduces a new data-driven approach that leverages a manifold
embedding generated by the invertible neural network to improve the robustness,
efficiency, and accuracy of the constitutive-law-free simulations with limited
data. We achieve this by training a deep neural network to globally map data
from the constitutive manifold onto a lower-dimensional Euclidean vector space.
As such, we establish the relation between the norm of the mapped Euclidean
vector space and the metric of the manifold and lead to a more physically
consistent notion of distance for the material data. This treatment in return
allows us to bypass the expensive combinatorial optimization, which may
significantly speed up the model-free simulations when data are abundant and of
high dimensions. Meanwhile, the learning of embedding also improves the
robustness of the algorithm when the data is sparse or distributed unevenly in
the parametric space. Numerical experiments are provided to demonstrate and
measure the performance of the manifold embedding technique under different
circumstances. Results obtained from the proposed method and those obtained via
the classical energy norms are compared.

    

### [[2112.09858] Deep Learning for Stability Analysis of a Freely Vibrating Sphere at Moderate Reynolds Number](http://arxiv.org/abs/2112.09858)


  In this paper, we present a deep learning-based reduced-order model (DL-ROM)
for the stability prediction of unsteady 3D fluid-structure interaction
systems. The proposed DL-ROM has the format of a nonlinear state-space model
and employs a recurrent neural network with long short-term memory (LSTM). We
consider a canonical fluid-structure system of an elastically-mounted sphere
coupled with incompressible fluid flow in a state-space format. We develop a
nonlinear data-driven coupling for predicting unsteady forces and
vortex-induced vibration (VIV) lock-in of the freely vibrating sphere in a
transverse direction. We design an input-output relationship as a temporal
sequence of force and displacement datasets for a low-dimensional approximation
of the fluid-structure system. Based on the prior knowledge of the VIV lock-in
process, the input function contains a range of frequencies and amplitudes,
which enables an efficient DL-ROM without the need for a massive training
dataset for the low-dimensional modeling. Once trained, the network provides a
nonlinear mapping of input-output dynamics that can predict the coupled
fluid-structure dynamics for a longer horizon via the feedback process. By
integrating the LSTM network with the eigensystem realization algorithm (ERA),
we construct a data-driven state-space model for the reduced-order stability
analysis. We investigate the underlying mechanism and stability characteristics
of VIV via an eigenvalue selection process. To understand the frequency lock-in
mechanism, we study the eigenvalue trajectories for a range of the reduced
oscillation frequencies and the mass ratios. Consistent with the full-order
simulations, the frequency lock-in branches are accurately captured by the
combined LSTM-ERA procedure. The proposed DL-ROM aligns with the development of
physics-based digital twin of engineering systems involving fluid-structure
interactions.

    

### [[2112.09859] Improved No-Regret Algorithms for Stochastic Shortest Path with Linear MDP](http://arxiv.org/abs/2112.09859)


  We introduce two new no-regret algorithms for the stochastic shortest path
(SSP) problem with a linear MDP that significantly improve over the only
existing results of (Vial et al., 2021). Our first algorithm is computationally
efficient and achieves a regret bound
$\widetilde{O}\left(\sqrt{d^3B_{\star}^2T_{\star} K}\right)$, where $d$ is the
dimension of the feature space, $B_{\star}$ and $T_{\star}$ are upper bounds of
the expected costs and hitting time of the optimal policy respectively, and $K$
is the number of episodes. The same algorithm with a slight modification also
achieves logarithmic regret of order
$O\left(\frac{d^3B_{\star}^4}{c_{\min}^2\text{gap}_{\min}}\ln^5\frac{dB_{\star}
K}{c_{\min}} \right)$, where $\text{gap}_{\min}$ is the minimum sub-optimality
gap and $c_{\min}$ is the minimum cost over all state-action pairs. Our result
is obtained by developing a simpler and improved analysis for the
finite-horizon approximation of (Cohen et al., 2021) with a smaller
approximation error, which might be of independent interest. On the other hand,
using variance-aware confidence sets in a global optimization problem, our
second algorithm is computationally inefficient but achieves the first
"horizon-free" regret bound $\widetilde{O}(d^{3.5}B_{\star}\sqrt{K})$ with no
polynomial dependency on $T_{\star}$ or $1/c_{\min}$, almost matching the
$\Omega(dB_{\star}\sqrt{K})$ lower bound from (Min et al., 2021).

    

### [[2112.09860] Morpheme Boundary Detection & Grammatical Feature Prediction for Gujarati : Dataset & Model](http://arxiv.org/abs/2112.09860)


  Developing Natural Language Processing resources for a low resource language
is a challenging but essential task. In this paper, we present a Morphological
Analyzer for Gujarati. We have used a Bi-Directional LSTM based approach to
perform morpheme boundary detection and grammatical feature tagging. We have
created a data set of Gujarati words with lemma and grammatical features. The
Bi-LSTM based model of Morph Analyzer discussed in the paper handles the
language morphology effectively without the knowledge of any hand-crafted
suffix rules. To the best of our knowledge, this is the first dataset and morph
analyzer model for the Gujarati language which performs both grammatical
feature tagging and morpheme boundary detection tasks.

    

### [[2112.09865] Off-Policy Evaluation Using Information Borrowing and Context-Based Switching](http://arxiv.org/abs/2112.09865)


  We consider the off-policy evaluation (OPE) problem in contextual bandits,
where the goal is to estimate the value of a target policy using the data
collected by a logging policy. Most popular approaches to the OPE are variants
of the doubly robust (DR) estimator obtained by combining a direct method (DM)
estimator and a correction term involving the inverse propensity score (IPS).
Existing algorithms primarily focus on strategies to reduce the variance of the
DR estimator arising from large IPS. We propose a new approach called the
Doubly Robust with Information borrowing and Context-based switching (DR-IC)
estimator that focuses on reducing both bias and variance. The DR-IC estimator
replaces the standard DM estimator with a parametric reward model that borrows
information from the 'closer' contexts through a correlation structure that
depends on the IPS. The DR-IC estimator also adaptively interpolates between
this modified DM estimator and a modified DR estimator based on a
context-specific switching rule. We give provable guarantees on the performance
of the DR-IC estimator. We also demonstrate the superior performance of the
DR-IC estimator compared to the state-of-the-art OPE algorithms on a number of
benchmark problems.

    

### [[2112.09866] Cascading Adaptors to Leverage English Data to Improve Performance of Question Answering for Low-Resource Languages](http://arxiv.org/abs/2112.09866)


  Transformer based architectures have shown notable results on many down
streaming tasks including question answering. The availability of data, on the
other hand, impedes obtaining legitimate performance for low-resource
languages. In this paper, we investigate the applicability of pre-trained
multilingual models to improve the performance of question answering in
low-resource languages. We tested four combinations of language and task
adapters using multilingual transformer architectures on seven languages
similar to MLQA dataset. Additionally, we have also proposed zero-shot transfer
learning of low-resource question answering using language and task adapters.
We observed that stacking the language and the task adapters improves the
multilingual transformer models' performance significantly for low-resource
languages.

    

### [[2112.09891] Equilibrated Zeroth-Order Unrolled Deep Networks for Accelerated MRI](http://arxiv.org/abs/2112.09891)


  Recently, model-driven deep learning unrolls a certain iterative algorithm of
a regularization model into a cascade network by replacing the first-order
information (i.e., (sub)gradient or proximal operator) of the regularizer with
a network module, which appears more explainable and predictable compared to
common data-driven networks. Conversely, in theory, there is not necessarily
such a functional regularizer whose first-order information matches the
replaced network module, which means the network output may not be covered by
the original regularization model. Moreover, up to now, there is also no theory
to guarantee the global convergence and robustness (regularity) of unrolled
networks under realistic assumptions. To bridge this gap, this paper propose to
present a safeguarded methodology on network unrolling. Specifically, focusing
on accelerated MRI, we unroll a zeroth-order algorithm, of which the network
module represents the regularizer itself, so that the network output can be
still covered by the regularization model. Furthermore, inspired by the ideal
of deep equilibrium models, before backpropagating, we carry out the unrolled
iterative network to converge to a fixed point to ensure the convergence. In
case the measurement data contains noise, we prove that the proposed network is
robust against noisy interference. Finally, numerical experiments show that the
proposed network consistently outperforms the state-of-the-art MRI
reconstruction methods including traditional regularization methods and other
deep learning methods.

    

### [[2112.09893] Revisiting Memory Efficient Kernel Approximation: An Indefinite Learning Perspective](http://arxiv.org/abs/2112.09893)


  Matrix approximations are a key element in large-scale algebraic machine
learning approaches. The recently proposed method MEKA (Si et al., 2014)
effectively employs two common assumptions in Hilbert spaces: the low-rank
property of an inner product matrix obtained from a shift-invariant kernel
function and a data compactness hypothesis by means of an inherent
block-cluster structure. In this work, we extend MEKA to be applicable not only
for shift-invariant kernels but also for non-stationary kernels like polynomial
kernels and an extreme learning kernel. We also address in detail how to handle
non-positive semi-definite kernel functions within MEKA, either caused by the
approximation itself or by the intentional use of general kernel functions. We
present a Lanczos-based estimation of a spectrum shift to develop a stable
positive semi-definite MEKA approximation, also usable in classical convex
optimization frameworks. Furthermore, we support our findings with theoretical
considerations and a variety of experiments on synthetic and real-world data.

    

### [[2112.09895] Towards the Explanation of Graph Neural Networks in Digital Pathology with Information Flows](http://arxiv.org/abs/2112.09895)


  As Graph Neural Networks (GNNs) are widely adopted in digital pathology,
there is increasing attention to developing explanation models (explainers) of
GNNs for improved transparency in clinical decisions.
Existing explainers discover an explanatory subgraph relevant to the
prediction.
However, such a subgraph is insufficient to reveal all the critical
biological substructures for the prediction because the prediction will remain
unchanged after removing that subgraph.
Hence, an explanatory subgraph should be not only necessary for prediction,
but also sufficient to uncover the most predictive regions for the explanation.
Such explanation requires a measurement of information transferred from
different input subgraphs to the predictive output, which we define as
information flow.
In this work, we address these key challenges and propose IFEXPLAINER, which
generates a necessary and sufficient explanation for GNNs.
To evaluate the information flow within GNN's prediction, we first propose a
novel notion of predictiveness, named $f$-information, which is directional and
incorporates the realistic capacity of the GNN model.
Based on it, IFEXPLAINER generates the explanatory subgraph with maximal
information flow to the prediction.
Meanwhile, it minimizes the information flow from the input to the predictive
result after removing the explanation.
Thus, the produced explanation is necessarily important to the prediction and
sufficient to reveal the most crucial substructures.
We evaluate IFEXPLAINER to interpret GNN's predictions on breast cancer
subtyping.
Experimental results on the BRACS dataset show the superior performance of
the proposed method.

    

### [[2112.09898] Does Explainable Machine Learning Uncover the Black Box in Vision Applications?](http://arxiv.org/abs/2112.09898)


  Machine learning (ML) in general and deep learning (DL) in particular has
become an extremely popular tool in several vision applications (like object
detection, super resolution, segmentation, object tracking etc.). Almost in
parallel, the issue of explainability in ML (i.e. the ability to
explain/elaborate the way a trained ML model arrived at its decision) in vision
has also received fairly significant attention from various quarters. However,
we argue that the current philosophy behind explainable ML suffers from certain
limitations, and the resulting explanations may not meaningfully uncover black
box ML models. To elaborate our assertion, we first raise a few fundamental
questions which have not been adequately discussed in the corresponding
literature. We also provide perspectives on how explainablity in ML can benefit
by relying on more rigorous principles in the related areas.

    

### [[2112.09899] Improving Subgraph Recognition with Variational Graph Information Bottleneck](http://arxiv.org/abs/2112.09899)


  Subgraph recognition aims at discovering a compressed substructure of a graph
that is most informative to the graph property. It can be formulated by
optimizing Graph Information Bottleneck (GIB) with a mutual information
estimator. However, GIB suffers from training instability since the mutual
information of graph data is intrinsically difficult to estimate. This paper
introduces a noise injection method to compress the information in the
subgraphs, which leads to a novel Variational Graph Information Bottleneck
(VGIB) framework. VGIB allows a tractable variational approximation to its
objective under mild assumptions. Therefore, VGIB enjoys more stable and
efficient training process - we find that VGIB converges 10 times faster than
GIB with improved performances in practice. Extensive experiments on graph
interpretation, explainability of Graph Neural Networks, and graph
classification show that VGIB finds better subgraphs than existing methods.

    

### [[2112.09902] 3D Instance Segmentation of MVS Buildings](http://arxiv.org/abs/2112.09902)


  We present a novel framework for instance segmentation of 3D buildings from
Multi-view Stereo (MVS) urban scenes. Unlike existing works focusing on
semantic segmentation of an urban scene, the emphasis of this work lies in
detecting and segmenting 3D building instances even if they are attached and
embedded in a large and imprecise 3D surface model. Multi-view RGB images are
first enhanced to RGBH images by adding a heightmap and are segmented to obtain
all roof instances using a fine-tuned 2D instance segmentation neural network.
Roof instance masks from different multi-view images are then clustered into
global masks. Our mask clustering accounts for spatial occlusion and
overlapping, which can eliminate segmentation ambiguities among multi-view
images. Based on these global masks, 3D roof instances are segmented out by
mask back-projections and extended to the entire building instances through a
Markov random field (MRF) optimization. Quantitative evaluations and ablation
studies have shown the effectiveness of all major steps of the method. A
dataset for the evaluation of instance segmentation of 3D building models is
provided as well. To the best of our knowledge, it is the first dataset for 3D
urban buildings on the instance segmentation level.

    

### [[2112.09906] Learning to Model the Relationship Between Brain Structural and Functional Connectomes](http://arxiv.org/abs/2112.09906)


  Recent advances in neuroimaging along with algorithmic innovations in
statistical learning from network data offer a unique pathway to integrate
brain structure and function, and thus facilitate revealing some of the brain's
organizing principles at the system level. In this direction, we develop a
supervised graph representation learning framework to model the relationship
between brain structural connectivity (SC) and functional connectivity (FC) via
a graph encoder-decoder system, where the SC is used as input to predict
empirical FC. A trainable graph convolutional encoder captures direct and
indirect interactions between brain regions-of-interest that mimic actual
neural communications, as well as to integrate information from both the
structural network topology and nodal (i.e., region-specific) attributes. The
encoder learns node-level SC embeddings which are combined to generate (whole
brain) graph-level representations for reconstructing empirical FC networks.
The proposed end-to-end model utilizes a multi-objective loss function to
jointly reconstruct FC networks and learn discriminative graph representations
of the SC-to-FC mapping for downstream subject (i.e., graph-level)
classification. Comprehensive experiments demonstrate that the learnt
representations of said relationship capture valuable information from the
intrinsic properties of the subject's brain networks and lead to improved
accuracy in classifying a large population of heavy drinkers and non-drinkers
from the Human Connectome Project. Our work offers new insights on the
relationship between brain networks that support the promising prospect of
using graph representation learning to discover more about human brain activity
and function.

    

### [[2112.09924] The Web Is Your Oyster -- Knowledge-Intensive NLP against a Very Large Web Corpus](http://arxiv.org/abs/2112.09924)


  In order to address the increasing demands of real-world applications, the
research for knowledge-intensive NLP (KI-NLP) should advance by capturing the
challenges of a truly open-domain environment: web scale knowledge, lack of
structure, inconsistent quality, and noise. To this end, we propose a new setup
for evaluating existing KI-NLP tasks in which we generalize the background
corpus to a universal web snapshot. We repurpose KILT, a standard KI-NLP
benchmark initially developed for Wikipedia, and ask systems to use a subset of
CCNet - the Sphere corpus - as a knowledge source. In contrast to Wikipedia,
Sphere is orders of magnitude larger and better reflects the full diversity of
knowledge on the Internet. We find that despite potential gaps of coverage,
challenges of scale, lack of structure and lower quality, retrieval from Sphere
enables a state-of-the-art retrieve-and-read system to match and even
outperform Wikipedia-based models on several KILT tasks - even if we
aggressively filter content that looks like Wikipedia. We also observe that
while a single dense passage index over Wikipedia can outperform a sparse BM25
version, on Sphere this is not yet possible. To facilitate further research
into this area, and minimise the community's reliance on proprietary black box
search engines, we will share our indices, evaluation metrics and
infrastructure.

    

### [[2112.09933] DegreEmbed: incorporating entity embedding into logic rule learning for knowledge graph reasoning](http://arxiv.org/abs/2112.09933)


  Knowledge graphs (KGs), as structured representations of real world facts,
are intelligent databases incorporating human knowledge that can help machine
imitate the way of human problem solving. However, due to the nature of rapid
iteration as well as incompleteness of data, KGs are usually huge and there are
inevitably missing facts in KGs. Link prediction for knowledge graphs is the
task aiming to complete missing facts by reasoning based on the existing
knowledge. Two main streams of research are widely studied: one learns
low-dimensional embeddings for entities and relations that can capture latent
patterns, and the other gains good interpretability by mining logical rules.
Unfortunately, previous studies rarely pay attention to heterogeneous KGs. In
this paper, we propose DegreEmbed, a model that combines embedding-based
learning and logic rule mining for inferring on KGs. Specifically, we study the
problem of predicting missing links in heterogeneous KGs that involve entities
and relations of various types from the perspective of the degrees of nodes.
Experimentally, we demonstrate that our DegreEmbed model outperforms the
state-of-the-art methods on real world datasets. Meanwhile, the rules mined by
our model are of high quality and interpretability.

    

### [[2112.09943] Exploiting Expert-guided Symmetry Detection in Markov Decision Processes](http://arxiv.org/abs/2112.09943)


  Offline estimation of the dynamical model of a Markov Decision Process (MDP)
is a non-trivial task that greatly depends on the data available to the
learning phase. Sometimes the dynamics of the model is invariant with respect
to some transformations of the current state and action. Recent works showed
that an expert-guided pipeline relying on Density Estimation methods as Deep
Neural Network based Normalizing Flows effectively detects this structure in
deterministic environments, both categorical and continuous-valued. The
acquired knowledge can be exploited to augment the original data set, leading
eventually to a reduction in the distributional shift between the true and the
learnt model. In this work we extend the paradigm to also tackle non
deterministic MDPs, in particular 1) we propose a detection threshold in
categorical environments based on statistical distances, 2) we introduce a
benchmark of the distributional shift in continuous environments based on the
Wilcoxon signed-rank statistical test and 3) we show that the former results
lead to a performance improvement when solving the learnt MDP and then applying
the optimal policy in the real environment.

    

### [[2112.09951] Rapid Face Mask Detection and Person Identification Model based on Deep Neural Networks](http://arxiv.org/abs/2112.09951)


  As Covid-19 has been constantly getting mutated and in three or four months a
new variant gets introduced to us and it comes with more deadly problems. The
things that prevent us from getting Covid is getting vaccinated and wearing a
face mask. In this paper, we have implemented a new Face Mask Detection and
Person Recognition model named Insight face which is based on SoftMax loss
classification algorithm Arc Face loss and names it as RFMPI-DNN(Rapid Face
Detection and Peron Identification Model based on Deep Neural Networks) to
detect face mask and person identity rapidly as compared to other models
available. To compare our new model, we have used previous MobileNet_V2 model
and face recognition module for effective comparison on the basis of time. The
proposed model implemented in the system has outperformed the model compared in
this paper in every aspect

    

### [[2112.09964] GOPHER: Categorical probabilistic forecasting with graph structure via local continuous-time dynamics](http://arxiv.org/abs/2112.09964)


  We consider the problem of probabilistic forecasting over categories with
graph structure, where the dynamics at a vertex depends on its local
connectivity structure. We present GOPHER, a method that combines the inductive
bias of graph neural networks with neural ODEs to capture the intrinsic local
continuous-time dynamics of our probabilistic forecasts. We study the benefits
of these two inductive biases by comparing against baseline models that help
disentangle the benefits of each. We find that capturing the graph structure is
crucial for accurate in-domain probabilistic predictions and more sample
efficient models. Surprisingly, our experiments demonstrate that the continuous
time evolution inductive bias brings little to no benefit despite reflecting
the true probability dynamics.

    

### [[2112.09968] Being Friends Instead of Adversaries: Deep Networks Learn from Data Simplified by Other Networks](http://arxiv.org/abs/2112.09968)


  Amongst a variety of approaches aimed at making the learning procedure of
neural networks more effective, the scientific community developed strategies
to order the examples according to their estimated complexity, to distil
knowledge from larger networks, or to exploit the principles behind adversarial
machine learning. A different idea has been recently proposed, named Friendly
Training, which consists in altering the input data by adding an automatically
estimated perturbation, with the goal of facilitating the learning process of a
neural classifier. The transformation progressively fades-out as long as
training proceeds, until it completely vanishes. In this work we revisit and
extend this idea, introducing a radically different and novel approach inspired
by the effectiveness of neural generators in the context of Adversarial Machine
Learning. We propose an auxiliary multi-layer network that is responsible of
altering the input data to make them easier to be handled by the classifier at
the current stage of the training procedure. The auxiliary network is trained
jointly with the neural classifier, thus intrinsically increasing the 'depth'
of the classifier, and it is expected to spot general regularities in the data
alteration process. The effect of the auxiliary network is progressively
reduced up to the end of training, when it is fully dropped and the classifier
is deployed for applications. We refer to this approach as Neural Friendly
Training. An extended experimental procedure involving several datasets and
different neural architectures shows that Neural Friendly Training overcomes
the originally proposed Friendly Training technique, improving the
generalization of the classifier, especially in the case of noisy data.

    

### [[2112.09970] 3D Structural Analysis of the Optic Nerve Head to Robustly Discriminate Between Papilledema and Optic Disc Drusen](http://arxiv.org/abs/2112.09970)


  Purpose: (1) To develop a deep learning algorithm to identify major tissue
structures of the optic nerve head (ONH) in 3D optical coherence tomography
(OCT) scans; (2) to exploit such information to robustly differentiate among
healthy, optic disc drusen (ODD), and papilledema ONHs.
It was a cross-sectional comparative study with confirmed ODD (105 eyes),
papilledema due to high intracranial pressure (51 eyes), and healthy controls
(100 eyes). 3D scans of the ONHs were acquired using OCT, then processed to
improve deep-tissue visibility. At first, a deep learning algorithm was
developed using 984 B-scans (from 130 eyes) in order to identify: major
neural/connective tissues, and ODD regions. The performance of our algorithm
was assessed using the Dice coefficient (DC). In a 2nd step, a classification
algorithm (random forest) was designed using 150 OCT volumes to perform 3-class
classifications (1: ODD, 2: papilledema, 3: healthy) strictly from their drusen
and prelamina swelling scores (derived from the segmentations). To assess
performance, we reported the area under the receiver operating characteristic
curves (AUCs) for each class.
Our segmentation algorithm was able to isolate neural and connective tissues,
and ODD regions whenever present. This was confirmed by an average DC of
0.93$\pm$0.03 on the test set, corresponding to good performance.
Classification was achieved with high AUCs, i.e. 0.99$\pm$0.01 for the
detection of ODD, 0.99 $\pm$ 0.01 for the detection of papilledema, and
0.98$\pm$0.02 for the detection of healthy ONHs.
Our AI approach accurately discriminated ODD from papilledema, using a single
OCT scan. Our classification performance was excellent, with the caveat that
validation in a much larger population is warranted. Our approach may have the
potential to establish OCT as the mainstay of diagnostic imaging in
neuro-ophthalmology.

    

### [[2112.09986] Leveraging Transformers for Hate Speech Detection in Conversational Code-Mixed Tweets](http://arxiv.org/abs/2112.09986)


  In the current era of the internet, where social media platforms are easily
accessible for everyone, people often have to deal with threats, identity
attacks, hate, and bullying due to their association with a cast, creed,
gender, religion, or even acceptance or rejection of a notion. Existing works
in hate speech detection primarily focus on individual comment classification
as a sequence labeling task and often fail to consider the context of the
conversation. The context of a conversation often plays a substantial role when
determining the author's intent and sentiment behind the tweet. This paper
describes the system proposed by team MIDAS-IIITD for HASOC 2021 subtask 2, one
of the first shared tasks focusing on detecting hate speech from Hindi-English
code-mixed conversations on Twitter. We approach this problem using neural
networks, leveraging the transformer's cross-lingual embeddings and further
finetuning them for low-resource hate-speech classification in transliterated
Hindi text. Our best performing system, a hard voting ensemble of Indic-BERT,
XLM-RoBERTa, and Multilingual BERT, achieved a macro F1 score of 0.7253,
placing us first on the overall leaderboard standings.

    

### [[2112.09990] FlowPool: Pooling Graph Representations with Wasserstein Gradient Flows](http://arxiv.org/abs/2112.09990)


  In several machine learning tasks for graph structured data, the graphs under
consideration may be composed of a varying number of nodes. Therefore, it is
necessary to design pooling methods that aggregate the graph representations of
varying size to representations of fixed size which can be used in downstream
tasks, such as graph classification. Existing graph pooling methods offer no
guarantee with regards to the similarity of a graph representation and its
pooled version. In this work we address this limitation by proposing FlowPool,
a pooling method that optimally preserves the statistics of a graph
representation to its pooled counterpart by minimizing their Wasserstein
distance. This is achieved by performing a Wasserstein gradient flow with
respect to the pooled graph representation. We propose a versatile
implementation of our method which can take into account the geometry of the
representation space through any ground cost. This implementation relies on the
computation of the gradient of the Wasserstein distance with recently proposed
implicit differentiation schemes. Our pooling method is amenable to automatic
differentiation and can be integrated in end-to-end deep learning
architectures. Further, FlowPool is invariant to permutations and can therefore
be combined with permutation equivariant feature extraction layers in GNNs in
order to obtain predictions that are independent of the ordering of the nodes.
Experimental results demonstrate that our method leads to an increase in
performance compared to existing pooling methods when evaluated in graph
classification tasks.

    

### [[2112.09992] Weisfeiler and Leman go Machine Learning: The Story so far](http://arxiv.org/abs/2112.09992)


  In recent years, algorithms and neural architectures based on the
Weisfeiler-Leman algorithm, a well-known heuristic for the graph isomorphism
problem, emerged as a powerful tool for machine learning with graphs and
relational data. Here, we give a comprehensive overview of the algorithm's use
in a machine learning setting, focusing on the supervised regime. We discuss
the theoretical background, show how to use it for supervised graph- and node
representation learning, discuss recent extensions, and outline the algorithm's
connection to (permutation-)equivariant neural architectures. Moreover, we give
an overview of current applications and future directions to stimulate further
research.

    

### [[2112.09995] Data-Driven Reachability analysis and Support set Estimation with Christoffel Functions](http://arxiv.org/abs/2112.09995)


  We present algorithms for estimating the forward reachable set of a dynamical
system using only a finite collection of independent and identically
distributed samples. The produced estimate is the sublevel set of a function
called an empirical inverse Christoffel function: empirical inverse Christoffel
functions are known to provide good approximations to the support of
probability distributions. In addition to reachability analysis, the same
approach can be applied to general problems of estimating the support of a
random variable, which has applications in data science towards detection of
novelties and outliers in data sets. In applications where safety is a concern,
having a guarantee of accuracy that holds on finite data sets is critical. In
this paper, we prove such bounds for our algorithms under the Probably
Approximately Correct (PAC) framework. In addition to applying classical
Vapnik-Chervonenkis (VC) dimension bound arguments, we apply the PAC-Bayes
theorem by leveraging a formal connection between kernelized empirical inverse
Christoffel functions and Gaussian process regression models. The bound based
on PAC-Bayes applies to a more general class of Christoffel functions than the
VC dimension argument, and achieves greater sample efficiency in experiments.

    

### [[2112.09998] Learning-based methods to model small body gravity fields for proximity operations: Safety and Robustness](http://arxiv.org/abs/2112.09998)


  Accurate gravity field models are essential for safe proximity operations
around small bodies. State-of-the-art techniques use spherical harmonics or
high-fidelity polyhedron shape models. Unfortunately, these techniques can
become inaccurate near the surface of the small body or have high computational
costs, especially for binary or heterogeneous small bodies. New learning-based
techniques do not encode a predefined structure and are more versatile. In
exchange for versatility, learning-based techniques can be less robust outside
the training data domain. In deployment, the spacecraft trajectory is the
primary source of dynamics data. Therefore, the training data domain should
include spacecraft trajectories to accurately evaluate the learned model's
safety and robustness. We have developed a novel method for learning-based
gravity models that directly uses the spacecraft's past trajectories. We
further introduce a method to evaluate the safety and robustness of
learning-based techniques via comparing accuracy within and outside of the
training domain. We demonstrate this safety and robustness method for two
learning-based frameworks: Gaussian processes and neural networks. Along with
the detailed analysis provided, we empirically establish the need for
robustness verification of learned gravity models when used for proximity
operations.

    

### [[2112.10001] Cross-Domain Federated Learning in Medical Imaging](http://arxiv.org/abs/2112.10001)


  Federated learning is increasingly being explored in the field of medical
imaging to train deep learning models on large scale datasets distributed
across different data centers while preserving privacy by avoiding the need to
transfer sensitive patient information. In this manuscript, we explore
federated learning in a multi-domain, multi-task setting wherein different
participating nodes may contain datasets sourced from different domains and are
trained to solve different tasks. We evaluated cross-domain federated learning
for the tasks of object detection and segmentation across two different
experimental settings: multi-modal and multi-organ. The result from our
experiments on cross-domain federated learning framework were very encouraging
with an overlap similarity of 0.79 for organ localization and 0.65 for lesion
segmentation. Our results demonstrate the potential of federated learning in
developing multi-domain, multi-task deep learning models without sharing data
from different domains.

    

### [[2112.10006] Low-resource Learning with Knowledge Graphs: A Comprehensive Survey](http://arxiv.org/abs/2112.10006)


  Machine learning methods especially deep neural networks have achieved great
success but many of them often rely on a number of labeled samples for
training. In real-world applications, we often need to address sample shortage
due to e.g., dynamic contexts with emerging prediction targets and costly
sample annotation. Therefore, low-resource learning, which aims to learn robust
prediction models with no enough resources (especially training samples), is
now being widely investigated. Among all the low-resource learning studies,
many prefer to utilize some auxiliary information in form of Knowledge Graph
(KG), which is becoming more and more popular for knowledge representation, to
reduce the reliance on labeled samples. In this survey, we very comprehensively
reviewed over $90$ papers about KG-aware research for two major low-resource
learning settings -- zero-shot learning (ZSL) where new classes for prediction
have never appeared in training, and few-shot learning (FSL) where new classes
for prediction have only a small number of labeled samples that are available.
We first introduced the KGs used in ZSL and FSL studies as well as the existing
and potential KG construction solutions, and then systematically categorized
and summarized KG-aware ZSL and FSL methods, dividing them into different
paradigms such as the mapping-based, the data augmentation, the
propagation-based and the optimization-based. We next presented different
applications, including both KG augmented prediction tasks in Computer Vision
and Natural Language Processing but also tasks for KG completion, and some
typical evaluation resources for each task. We eventually discussed some
challenges and future directions on aspects such as new learning and reasoning
paradigms, and the construction of high quality KGs.

    

### [[2112.10017] Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](http://arxiv.org/abs/2112.10017)


  Existing research on continual learning of a sequence of tasks focused on
dealing with catastrophic forgetting, where the tasks are assumed to be
dissimilar and have little shared knowledge. Some work has also been done to
transfer previously learned knowledge to the new task when the tasks are
similar and have shared knowledge. To the best of our knowledge, no technique
has been proposed to learn a sequence of mixed similar and dissimilar tasks
that can deal with forgetting and also transfer knowledge forward and backward.
This paper proposes such a technique to learn both types of tasks in the same
network. For dissimilar tasks, the algorithm focuses on dealing with
forgetting, and for similar tasks, the algorithm focuses on selectively
transferring the knowledge learned from some similar previous tasks to improve
the new task learning. Additionally, the algorithm automatically detects
whether a new task is similar to any previous tasks. Empirical evaluation using
sequences of mixed tasks demonstrates the effectiveness of the proposed model.

    

### [[2112.10021] Continual Learning with Knowledge Transfer for Sentiment Classification](http://arxiv.org/abs/2112.10021)


  This paper studies continual learning (CL) for sentiment classification (SC).
In this setting, the CL system learns a sequence of SC tasks incrementally in a
neural network, where each task builds a classifier to classify the sentiment
of reviews of a particular product category or domain. Two natural questions
are: Can the system transfer the knowledge learned in the past from the
previous tasks to the new task to help it learn a better model for the new
task? And, can old models for previous tasks be improved in the process as
well? This paper proposes a novel technique called KAN to achieve these
objectives. KAN can markedly improve the SC accuracy of both the new task and
the old tasks via forward and backward knowledge transfer. The effectiveness of
KAN is demonstrated through extensive experiments.

    

### [[2112.10039] Wasserstein Generative Learning of Conditional Distribution](http://arxiv.org/abs/2112.10039)


  Conditional distribution is a fundamental quantity for describing the
relationship between a response and a predictor. We propose a Wasserstein
generative approach to learning a conditional distribution. The proposed
approach uses a conditional generator to transform a known distribution to the
target conditional distribution. The conditional generator is estimated by
matching a joint distribution involving the conditional generator and the
target joint distribution, using the Wasserstein distance as the discrepancy
measure for these joint distributions. We establish non-asymptotic error bound
of the conditional sampling distribution generated by the proposed method and
show that it is able to mitigate the curse of dimensionality, assuming that the
data distribution is supported on a lower-dimensional set. We conduct numerical
experiments to validate proposed method and illustrate its applications to
conditional sample generation, nonparametric conditional density estimation,
prediction uncertainty quantification, bivariate response data, image
reconstruction and image generation.

    

### [[2112.10046] A-ESRGAN: Training Real-World Blind Super-Resolution with Attention U-Net Discriminators](http://arxiv.org/abs/2112.10046)


  Blind image super-resolution(SR) is a long-standing task in CV that aims to
restore low-resolution images suffering from unknown and complex distortions.
Recent work has largely focused on adopting more complicated degradation models
to emulate real-world degradations. The resulting models have made
breakthroughs in perceptual loss and yield perceptually convincing results.
However, the limitation brought by current generative adversarial network
structures is still significant: treating pixels equally leads to the ignorance
of the image's structural features, and results in performance drawbacks such
as twisted lines and background over-sharpening or blurring. In this paper, we
present A-ESRGAN, a GAN model for blind SR tasks featuring an attention U-Net
based, multi-scale discriminator that can be seamlessly integrated with other
generators. To our knowledge, this is the first work to introduce attention
U-Net structure as the discriminator of GAN to solve blind SR problems. And the
paper also gives an interpretation for the mechanism behind multi-scale
attention U-Net that brings performance breakthrough to the model. Through
comparison experiments with prior works, our model presents state-of-the-art
level performance on the non-reference natural image quality evaluator metric.
And our ablation studies have shown that with our discriminator, the RRDB based
generator can leverage the structural features of an image in multiple scales,
and consequently yields more perceptually realistic high-resolution images
compared to prior works.

    

### [[2112.10063] Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation](http://arxiv.org/abs/2112.10063)


  Graph-level anomaly detection (GAD) describes the problem of detecting graphs
that are abnormal in their structure and/or the features of their nodes, as
compared to other graphs. One of the challenges in GAD is to devise graph
representations that enable the detection of both locally- and
globally-anomalous graphs, i.e., graphs that are abnormal in their fine-grained
(node-level) or holistic (graph-level) properties, respectively. To tackle this
challenge we introduce a novel deep anomaly detection approach for GAD that
learns rich global and local normal pattern information by joint random
distillation of graph and node representations. The random distillation is
achieved by training one GNN to predict another GNN with randomly initialized
network weights. Extensive experiments on 16 real-world graph datasets from
diverse domains show that our model significantly outperforms seven
state-of-the-art models. Code and datasets are available at
this https URL.

    

### [[2112.10065] Efficient Strong Scaling Through Burst Parallel Training](http://arxiv.org/abs/2112.10065)


  As emerging deep neural network (DNN) models continue to grow in size, using
large GPU clusters to train DNNs is becoming an essential requirement to
achieving acceptable training times. In this paper, we consider the case where
future increases in cluster size will cause the global batch size that can be
used to train models to reach a fundamental limit: beyond a certain point,
larger global batch sizes cause sample efficiency to degrade, increasing
overall time to accuracy. As a result, to achieve further improvements in
training performance, we must instead consider "strong scaling" strategies that
hold the global batch size constant and allocate smaller batches to each GPU.
Unfortunately, this makes it significantly more difficult to use cluster
resources efficiently. We present DeepPool, a system that addresses this
efficiency challenge through two key ideas. First, burst parallelism allocates
large numbers of GPUs to foreground jobs in bursts to exploit the unevenness in
parallelism across layers. Second, GPU multiplexing prioritizes throughput for
foreground training jobs, while packing in background training jobs to reclaim
underutilized GPU resources, thereby improving cluster-wide utilization.
Together, these two ideas enable DeepPool to deliver a 2.2 - 2.4x improvement
in total cluster throughput over standard data parallelism with a single task
when the cluster scale is large.

    

### [[2112.10067] CORE: A Knowledge Graph Entity Type Prediction Method via Complex Space Regression and Embedding](http://arxiv.org/abs/2112.10067)


  Entity type prediction is an important problem in knowledge graph (KG)
research. A new KG entity type prediction method, named CORE (COmplex space
Regression and Embedding), is proposed in this work. The proposed CORE method
leverages the expressive power of two complex space embedding models; namely,
RotatE and ComplEx models. It embeds entities and types in two different
complex spaces using either RotatE or ComplEx. Then, we derive a complex
regression model to link these two spaces. Finally, a mechanism to optimize
embedding and regression parameters jointly is introduced. Experiments show
that CORE outperforms benchmarking methods on representative KG entity type
inference datasets. Strengths and weaknesses of various entity type prediction
methods are analyzed.

    

### [[2112.10068] Lerna: Transformer Architectures for Configuring Error Correction Tools for Short- and Long-Read Genome Sequencing](http://arxiv.org/abs/2112.10068)


  Sequencing technologies are prone to errors, making error correction (EC)
necessary for downstream applications. EC tools need to be manually configured
for optimal performance. We find that the optimal parameters (e.g., k-mer size)
are both tool- and dataset-dependent. Moreover, evaluating the performance
(i.e., Alignment-rate or Gain) of a given tool usually relies on a reference
genome, but quality reference genomes are not always available. We introduce
Lerna for the automated configuration of k-mer-based EC tools. Lerna first
creates a language model (LM) of the uncorrected genomic reads; then,
calculates the perplexity metric to evaluate the corrected reads for different
parameter choices. Next, it finds the one that produces the highest alignment
rate without using a reference genome. The fundamental intuition of our
approach is that the perplexity metric is inversely correlated with the quality
of the assembly after error correction. Results: First, we show that the best
k-mer value can vary for different datasets, even for the same EC tool. Second,
we show the gains of our LM using its component attention-based transformers.
We show the model's estimation of the perplexity metric before and after error
correction. The lower the perplexity after correction, the better the k-mer
size. We also show that the alignment rate and assembly quality computed for
the corrected reads are strongly negatively correlated with the perplexity,
enabling the automated selection of k-mer values for better error correction,
and hence, improved assembly quality. Additionally, we show that our
attention-based models have significant runtime improvement for the entire
pipeline -- 18X faster than previous works, due to parallelizing the attention
mechanism and the use of JIT compilation for GPU inferencing.

    

### [[2112.10074] QU-BraTS: MICCAI BraTS 2020 Challenge on Quantifying Uncertainty in Brain Tumor Segmentation -- Analysis of Ranking Metrics and Benchmarking Results](http://arxiv.org/abs/2112.10074)


  Deep learning (DL) models have provided the state-of-the-art performance in a
wide variety of medical imaging benchmarking challenges, including the Brain
Tumor Segmentation (BraTS) challenges. However, the task of focal pathology
multi-compartment segmentation (e.g., tumor and lesion sub-regions) is
particularly challenging, and potential errors hinder the translation of DL
models into clinical workflows. Quantifying the reliability of DL model
predictions in the form of uncertainties, could enable clinical review of the
most uncertain regions, thereby building trust and paving the way towards
clinical translation. Recently, a number of uncertainty estimation methods have
been introduced for DL medical image segmentation tasks. Developing metrics to
evaluate and compare the performance of uncertainty measures will assist the
end-user in making more informed decisions. In this study, we explore and
evaluate a metric developed during the BraTS 2019-2020 task on uncertainty
quantification (QU-BraTS), and designed to assess and rank uncertainty
estimates for brain tumor multi-compartment segmentation. This metric (1)
rewards uncertainty estimates that produce high confidence in correct
assertions, and those that assign low confidence levels at incorrect
assertions, and (2) penalizes uncertainty measures that lead to a higher
percentages of under-confident correct assertions. We further benchmark the
segmentation uncertainties generated by 14 independent participating teams of
QU-BraTS 2020, all of which also participated in the main BraTS segmentation
task. Overall, our findings confirm the importance and complementary value that
uncertainty estimates provide to segmentation algorithms, and hence highlight
the need for uncertainty quantification in medical image analyses. Our
evaluation code is made publicly available at
this https URL.

    

### [[2112.10078] Managing dataset shift by adversarial validation for credit scoring](http://arxiv.org/abs/2112.10078)


  Dataset shift is common in credit scoring scenarios, and the inconsistency
between the distribution of training data and the data that actually needs to
be predicted is likely to cause poor model performance. However, most of the
current studies do not take this into account, and they directly mix data from
different time periods when training the models. This brings about two
problems. Firstly, there is a risk of data leakage, i.e., using future data to
predict the past. This can result in inflated results in offline validation,
but unsatisfactory results in practical applications. Secondly, the
macroeconomic environment and risk control strategies are likely to be
different in different time periods, and the behavior patterns of borrowers may
also change. The model trained with past data may not be applicable to the
recent stage. Therefore, we propose a method based on adversarial validation to
alleviate the dataset shift problem in credit scoring scenarios. In this
method, partial training set samples with the closest distribution to the
predicted data are selected for cross-validation by adversarial validation to
ensure the generalization performance of the trained model on the predicted
samples. In addition, through a simple splicing method, samples in the training
data that are inconsistent with the test data distribution are also involved in
the training process of cross-validation, which makes full use of all the data
and further improves the model performance. To verify the effectiveness of the
proposed method, comparative experiments with several other data split methods
are conducted with the data provided by Lending Club. The experimental results
demonstrate the importance of dataset shift in the field of credit scoring and
the superiority of the proposed method.

    

### [[2112.10101] ArcFace Knows the Gender, Too!](http://arxiv.org/abs/2112.10101)


  The main idea of this paper is that if a model can recognize a person, of
course, it must be able to know the gender of that person, too. Therefore,
instead of defining a new model for gender classification, this paper uses
ArcFace features to determine gender, based on the facial features. A face
image is given to ArcFace and 512 features are obtained for the face. Then,
with the help of traditional machine learning models, gender is determined.
Discriminative methods such as Support Vector Machine (SVM), Linear
Discriminant, and Logistic Regression well demonstrate that the features
extracted from the ArcFace create a remarkable distinction between the gender
classes. Experiments on the Gender Classification Dataset show that SVM with
Gaussian kernel is able to classify gender with an accuracy of 96.4% using
ArcFace features.

    

### [[2112.10107] Expression is enough: Improving traffic signal control with advanced traffic state representation](http://arxiv.org/abs/2112.10107)


  Recently, finding fundamental properties for traffic state representation is
more critical than complex algorithms for traffic signal control (TSC).In this
paper, we (1) present a novel, flexible and straightforward method advanced max
pressure (Advanced-MP), taking both running and queueing vehicles into
consideration to decide whether to change current phase; (2) novelty design the
traffic movement representation with the efficient pressure and effective
running vehicles from Advanced-MP, namely advanced traffic state (ATS); (3)
develop an RL-based algorithm template Advanced-XLight, by combining ATS with
current RL approaches and generate two RL algorithms, "Advanced-MPLight" and
"Advanced-CoLight". Comprehensive experiments on multiple real-world datasets
show that: (1) the Advanced-MP outperforms baseline methods, which is efficient
and reliable for deployment; (2) Advanced-MPLight and Advanced-CoLight could
achieve new state-of-the-art. Our code is released on Github.

    

### [[2112.10108] Investigation of Densely Connected Convolutional Networks with Domain Adversarial Learning for Noise Robust Speech Recognition](http://arxiv.org/abs/2112.10108)


  We investigate densely connected convolutional networks (DenseNets) and their
extension with domain adversarial training for noise robust speech recognition.
DenseNets are very deep, compact convolutional neural networks which have
demonstrated incredible improvements over the state-of-the-art results in
computer vision. Our experimental results reveal that DenseNets are more robust
against noise than other neural network based models such as deep feed forward
neural networks and convolutional neural networks. Moreover, domain adversarial
learning can further improve the robustness of DenseNets against both, known
and unknown noise conditions.

    

### [[2112.10121] Evaluating System Identification Methods for Predicting Thermal Dissipation of Heterogeneous SoCs](http://arxiv.org/abs/2112.10121)


  In this paper we evaluate the use of system identification methods to build a
thermal prediction model of heterogeneous SoC platforms that can be used to
quickly predict the temperature of different configurations without the need of
hardware. Specifically, we focus on modeling approaches that can predict the
temperature based on the clock frequency and the utilization percentage of each
core. We investigate three methods with respect to their prediction accuracy: a
linear state-space identification approach using polynomial regressors, a NARX
neural network approach and a recurrent neural network approach configured in
an FIR model structure. We evaluate the methods on an Odroid-XU4 board
featuring an Exynos 5422 SoC. The results show that the model based on
polynomial regressors significantly outperformed the other two models when
trained with 1 hour and 6 hours of data.

    

### [[2112.10123] Early Detection of Security-Relevant Bug Reports using Machine Learning: How Far Are We?](http://arxiv.org/abs/2112.10123)


  Bug reports are common artefacts in software development. They serve as the
main channel for users to communicate to developers information about the
issues that they encounter when using released versions of software programs.
In the descriptions of issues, however, a user may, intentionally or not,
expose a vulnerability. In a typical maintenance scenario, such
security-relevant bug reports are prioritised by the development team when
preparing corrective patches. Nevertheless, when security relevance is not
immediately expressed (e.g., via a tag) or rapidly identified by triaging
teams, the open security-relevant bug report can become a critical leak of
sensitive information that attackers can leverage to perform zero-day attacks.
To support practitioners in triaging bug reports, the research community has
proposed a number of approaches for the detection of security-relevant bug
reports. In recent years, approaches in this respect based on machine learning
have been reported with promising performance. Our work focuses on such
approaches, and revisits their building blocks to provide a comprehensive view
on the current achievements. To that end, we built a large experimental dataset
and performed extensive experiments with variations in feature sets and
learning algorithms. Eventually, our study highlights different approach
configurations that yield best performing classifiers.

    

### [[2112.10133] Information Field Theory as Artificial Intelligence](http://arxiv.org/abs/2112.10133)


  Information field theory (IFT), the information theory for fields, is a
mathematical framework for signal reconstruction and non-parametric inverse
problems. Here, fields denote physical quantities that change continuously as a
function of space (and time) and information theory refers to Bayesian
probabilistic logic equipped with the associated entropic information measures.
Reconstructing a signal with IFT is a computational problem similar to training
a generative neural network (GNN). In this paper, the inference in IFT is
reformulated in terms of GNN training and the cross-fertilization of numerical
variational inference methods used in IFT and machine learning are discussed.
The discussion suggests that IFT inference can be regarded as a specific form
of artificial intelligence. In contrast to classical neural networks, IFT based
GNNs can operate without pre-training thanks to incorporating expert knowledge
into their architecture.

    

### [[2112.10139] Denoised Labels for Financial Time-Series Data via Self-Supervised Learning](http://arxiv.org/abs/2112.10139)


  The introduction of electronic trading platforms effectively changed the
organisation of traditional systemic trading from quote-driven markets into
order-driven markets. Its convenience led to an exponentially increasing amount
of financial data, which is however hard to use for the prediction of future
prices, due to the low signal-to-noise ratio and the non-stationarity of
financial time series. Simpler classification tasks -- where the goal is to
predict the directions of future price movement -- via supervised learning
algorithms, need sufficiently reliable labels to generalise well. Labelling
financial data is however less well defined than other domains: did the price
go up because of noise or because of signal? The existing labelling methods
have limited countermeasures against noise and limited effects in improving
learning algorithms. This work takes inspiration from image classification in
trading and success in self-supervised learning. We investigate the idea of
applying computer vision techniques to financial time-series to reduce the
noise exposure and hence generate correct labels. We look at the label
generation as the pretext task of a self-supervised learning approach and
compare the naive (and noisy) labels, commonly used in the literature, with the
labels generated by a denoising autoencoder for the same downstream
classification task. Our results show that our denoised labels improve the
performances of the downstream learning algorithm, for both small and large
datasets. We further show that the signals we obtain can be used to effectively
trade with binary strategies. We suggest that with proposed techniques,
self-supervised learning constitutes a powerful framework for generating
"better" financial labels that are useful for studying the underlying patterns
of the market.

    

### [[2112.10143] RoboAssembly: Learning Generalizable Furniture Assembly Policy in a Novel Multi-robot Contact-rich Simulation Environment](http://arxiv.org/abs/2112.10143)


  Part assembly is a typical but challenging task in robotics, where robots
assemble a set of individual parts into a complete shape. In this paper, we
develop a robotic assembly simulation environment for furniture assembly. We
formulate the part assembly task as a concrete reinforcement learning problem
and propose a pipeline for robots to learn to assemble a diverse set of chairs.
Experiments show that when testing with unseen chairs, our approach achieves a
success rate of 74.5% under the object-centric setting and 50.0% under the full
setting. We adopt an RRT-Connect algorithm as the baseline, which only achieves
a success rate of 18.8% after a significantly longer computation time.
Supplemental materials and videos are available on our project webpage.

    

### [[2112.10150] Active Weighted Aging Ensemble for Drifted Data Stream Classification](http://arxiv.org/abs/2112.10150)


  One of the significant problems of streaming data classification is the
occurrence of concept drift, consisting of the change of probabilistic
characteristics of the classification task. This phenomenon destabilizes the
performance of the classification model and seriously degrades its quality. An
appropriate strategy counteracting this phenomenon is required to adapt the
classifier to the changing probabilistic characteristics. One of the
significant problems in implementing such a solution is the access to data
labels. It is usually costly, so to minimize the expenses related to this
process, learning strategies based on semi-supervised learning are proposed,
e.g., employing active learning methods indicating which of the incoming
objects are valuable to be labeled for improving the classifier's performance.
This paper proposes a novel chunk-based method for non-stationary data streams
based on classifier ensemble learning and an active learning strategy
considering a limited budget that can be successfully applied to any data
stream classification algorithm. The proposed method has been evaluated through
computer experiments using both real and generated data streams. The results
confirm the high quality of the proposed algorithm over state-of-the-art
methods.

    

### [[2112.10152] TECM: Transfer Evidential C-means Clustering](http://arxiv.org/abs/2112.10152)


  Clustering is widely used in text analysis, natural language processing,
image segmentation, and other data mining fields. As a promising clustering
algorithm, the evidential c-means (ECM) can provide a deeper insight on the
data by allowing an object to belong to several subsets of classes, which
extends those of hard, fuzzy, and possibilistic clustering. However, as it
needs to estimate much more parameters than the other classical partition-based
algorithms, it only works well when the available data is sufficient and of
good quality. In order to overcome these shortcomings, this paper proposes a
transfer evidential c-means (TECM) algorithm, by introducing the strategy of
transfer learning. The objective function of TECM is obtained by introducing
barycenters in the source domain on the basis of the objective function of ECM,
and the iterative optimization strategy is used to solve the objective
function. In addition, the TECM can adapt to situation where the number of
clusters in the source domain and the target domain is different. The proposed
algorithm has been validated on synthetic and real-world datasets. Experimental
results demonstrate the effectiveness of TECM in comparison with the original
ECM as well as other representative multitask or transfer clustering
algorithms.

    

### [[2112.10154] Representation Learning for Dynamic Hyperedges](http://arxiv.org/abs/2112.10154)


  Recently there has been a massive interest in extracting information from
interaction data. Traditionally this is done by modeling it as pair-wise
interaction at a particular time in a dynamic network. However, real-world
interactions are seldom pair-wise; they can involve more than two nodes. In
literature, these types of group interactions are modeled by
hyperedges/hyperlinks. The existing works for hyperedge modeling focused only
on static networks, and they cannot model the temporal evolution of nodes as
they interact with other nodes. Also, they cannot answer temporal queries like
which type of interaction will occur next and when the interaction will occur.
To address these limitations, in this paper, we develop a temporal point
process model for hyperlink prediction. Our proposed model uses dynamic
representation techniques for nodes to model the evolution and uses this
representation in a neural point process framework to make inferences. We
evaluate our models on five real-world interaction data and show that our
dynamic model has significant performance gain over the static model. Further,
we also demonstrate the advantages of our technique over the pair-wise
interaction modeling technique.

    

### [[2112.10157] Rethinking Importance Weighting for Transfer Learning](http://arxiv.org/abs/2112.10157)


  A key assumption in supervised learning is that training and test data follow
the same probability distribution. However, this fundamental assumption is not
always satisfied in practice, e.g., due to changing environments, sample
selection bias, privacy concerns, or high labeling costs. Transfer learning
(TL) relaxes this assumption and allows us to learn under distribution shift.
Classical TL methods typically rely on importance-weighting -- a predictor is
trained based on the training losses weighted according to the importance
(i.e., the test-over-training density ratio). However, as real-world machine
learning tasks are becoming increasingly complex, high-dimensional, and
dynamical, novel approaches are explored to cope with such challenges recently.
In this article, after introducing the foundation of TL based on
importance-weighting, we review recent advances based on joint and dynamic
importance-predictor estimation. Furthermore, we introduce a method of causal
mechanism transfer that incorporates causal structure in TL. Finally, we
discuss future perspectives of TL research.

    

### [[2112.10161] RELAX: Representation Learning Explainability](http://arxiv.org/abs/2112.10161)


  Despite the significant improvements that representation learning via
self-supervision has led to when learning from unlabeled data, no methods exist
that explain what influences the learned representation. We address this need
through our proposed approach, RELAX, which is the first approach for
attribution-based explanations of representations. Our approach can also model
the uncertainty in its explanations, which is essential to produce trustworthy
explanations. RELAX explains representations by measuring similarities in the
representation space between an input and masked out versions of itself,
providing intuitive explanations and significantly outperforming the
gradient-based baseline. We provide theoretical interpretations of RELAX and
conduct a novel analysis of feature extractors trained using supervised and
unsupervised learning, providing insights into different learning strategies.
Finally, we illustrate the usability of RELAX in multi-view clustering and
highlight that incorporating uncertainty can be essential for providing
low-complexity explanations, taking a crucial step towards explaining
representations.

    

### [[2112.10166] FedNI: Federated Graph Learning with Network Inpainting for Population-Based Disease Prediction](http://arxiv.org/abs/2112.10166)


  Graph Convolutional Neural Networks (GCNs) are widely used for graph
analysis. Specifically, in medical applications, GCNs can be used for disease
prediction on a population graph, where graph nodes represent individuals and
edges represent individual similarities. However, GCNs rely on a vast amount of
data, which is challenging to collect for a single medical institution. In
addition, a critical challenge that most medical institutions continue to face
is addressing disease prediction in isolation with incomplete data information.
To address these issues, Federated Learning (FL) allows isolated local
institutions to collaboratively train a global model without data sharing. In
this work, we propose a framework, FedNI, to leverage network inpainting and
inter-institutional data via FL. Specifically, we first federatively train
missing node and edge predictor using a graph generative adversarial network
(GAN) to complete the missing information of local networks. Then we train a
global GCN node classifier across institutions using a federated graph learning
platform. The novel design enables us to build more accurate machine learning
models by leveraging federated learning and also graph learning approaches. We
demonstrate that our federated model outperforms local and baseline FL methods
with significant margins on two public neuroimaging datasets.

    

### [[2112.10168] The Preliminary Results on Analysis of TAIGA-IACT Images Using Convolutional Neural Networks](http://arxiv.org/abs/2112.10168)


  The imaging Cherenkov telescopes TAIGA-IACT, located in the Tunka valley of
the republic Buryatia, accumulate a lot of data in a short period of time which
must be efficiently and quickly analyzed. One of the methods of such analysis
is the machine learning, which has proven its effectiveness in many
technological and scientific fields in recent years. The aim of the work is to
study the possibility of the machine learning application to solve the tasks
set for TAIGA-IACT: the identification of the primary particle of cosmic rays
and reconstruction their physical parameters. In the work the method of
Convolutional Neural Networks (CNN) was applied to process and analyze
Monte-Carlo events simulated with CORSIKA. Also various CNN architectures for
the processing were considered. It has been demonstrated that this method gives
good results in the determining the type of primary particles of Extensive Air
Shower (EAS) and the reconstruction of gamma-rays energy. The results are
significantly improved in the case of stereoscopic observations.

    

### [[2112.10170] Analysis of the HiSCORE Simulated Events in TAIGA Experiment Using Convolutional Neural Networks](http://arxiv.org/abs/2112.10170)


  TAIGA is a hybrid observatory for gamma-ray astronomy at high energies in
range from 10 TeV to several EeV. It consists of instruments such as
TAIGA-IACT, TAIGA-HiSCORE, and others. TAIGA-HiSCORE, in particular, is an
array of wide-angle timing Cherenkov light stations. TAIGA-HiSCORE data enable
to reconstruct air shower characteristics, such as air shower energy, arrival
direction, and axis coordinates. In this report, we propose to consider the use
of convolution neural networks in task of air shower characteristics
determination. We use Convolutional Neural Networks (CNN) to analyze HiSCORE
events, treating them like images. For this, the times and amplitudes of events
recorded at HiSCORE stations are used. The work discusses a simple
convolutional neural network and its training. In addition, we present some
preliminary results on the determination of the parameters of air showers such
as the direction and position of the shower axis and the energy of the primary
particle and compare them with the results obtained by the traditional method.

    

### [[2112.10214] Modelling of Received Signals in Molecular Communication Systems based machine learning: Comparison of azure machine learning and Python tools](http://arxiv.org/abs/2112.10214)


  Molecular communication (MC) implemented on Nano networks has extremely
attractive characteristics in terms of energy efficiency, dependability, and
robustness. Even though, the impact of incredibly slow molecule diffusion and
high variability environments remains unknown. Analysis and designs of
communication systems usually rely on developing mathematical models that
describe the communication channel. However, the underlying channel models are
unknown in some systems, such as MC systems, where chemical signals are used to
transfer information. In these cases, a new method to analyze and design is
needed. In this paper, we concentrate on one critical aspect of the MC system,
modelling MC received signal until time t , and demonstrate that using tools
from ML makes it promising to train detectors that can be executed well without
any information about the channel model. Machine learning (ML) is one of the
intelligent methodologies that has shown promising results in the domain. This
paper applies Azure Machine Learning (Azure ML) for flexible pavement
maintenance regressions problems and solutions. For prediction, four parameters
are used as inputs: the receiver radius, transmitter radius, distance between
receiver and transmitter, and diffusion coefficient, while the output is mAP
(mean average precision) of the received signal. Azure ML enables algorithms
that can learn from data and experiences and accomplish tasks without having to
be coded. In the established Azure ML, the regression algorithms such as, boost
decision tree regression, Bayesian linear regression, neural network, and
decision forest regression are selected. The best performance is chosen as an
optimality criterion. Finally, a comparison that shows the potential benefits
of Azure ML tool over programmed based tool (Python), used by developers on
local PCs, is demonstrated

    

### [[2112.10219] Quantum Approximate Optimization Algorithm applied to the binary perceptron](http://arxiv.org/abs/2112.10219)


  We apply digitized Quantum Annealing (QA) and Quantum Approximate
Optimization Algorithm (QAOA) to a paradigmatic task of supervised learning in
artificial neural networks: the optimization of synaptic weights for the binary
perceptron. At variance with the usual QAOA applications to MaxCut, or to
quantum spin-chains ground state preparation, the classical Hamiltonian is
characterized by highly non-local multi-spin interactions. Yet, we provide
evidence for the existence of optimal smooth solutions for the QAOA parameters,
which are transferable among typical instances of the same problem, and we
prove numerically an enhanced performance of QAOA over traditional QA. We also
investigate on the role of the QAOA optimization landscape geometry in this
problem, showing that the detrimental effect of a gap-closing transition
encountered in QA is also negatively affecting the performance of our
implementation of QAOA.

    

### [[2112.10224] Stable Conformal Prediction Sets](http://arxiv.org/abs/2112.10224)


  When one observes a sequence of variables $(x_1, y_1), ..., (x_n, y_n)$,
conformal prediction is a methodology that allows to estimate a confidence set
for $y_{n+1}$ given $x_{n+1}$ by merely assuming that the distribution of the
data is exchangeable. While appealing, the computation of such set turns out to
be infeasible in general, e.g. when the unknown variable $y_{n+1}$ is
continuous. In this paper, we combine conformal prediction techniques with
algorithmic stability bounds to derive a prediction set computable with a
single model fit. We perform some numerical experiments that illustrate the
tightness of our estimation when the sample size is sufficiently large.

    

### [[2112.10229] On Causal Inference for Data-free Structured Pruning](http://arxiv.org/abs/2112.10229)


  Neural networks (NNs) are making a large impact both on research and
industry. Nevertheless, as NNs' accuracy increases, it is followed by an
expansion in their size, required number of compute operations and energy
consumption. Increase in resource consumption results in NNs' reduced adoption
rate and real-world deployment impracticality. Therefore, NNs need to be
compressed to make them available to a wider audience and at the same time
decrease their runtime costs. In this work, we approach this challenge from a
causal inference perspective, and we propose a scoring mechanism to facilitate
structured pruning of NNs. The approach is based on measuring mutual
information under a maximum entropy perturbation, sequentially propagated
through the NN. We demonstrate the method's performance on two datasets and
various NNs' sizes, and we show that our approach achieves competitive
performance under challenging conditions.

    

### [[2112.10251] SSDNet: State Space Decomposition Neural Network for Time Series Forecasting](http://arxiv.org/abs/2112.10251)


  In this paper, we present SSDNet, a novel deep learning approach for time
series forecasting. SSDNet combines the Transformer architecture with state
space models to provide probabilistic and interpretable forecasts, including
trend and seasonality components and previous time steps important for the
prediction. The Transformer architecture is used to learn the temporal patterns
and estimate the parameters of the state space model directly and efficiently,
without the need for Kalman filters. We comprehensively evaluate the
performance of SSDNet on five data sets, showing that SSDNet is an effective
method in terms of accuracy and speed, outperforming state-of-the-art deep
learning and statistical methods, and able to provide meaningful trend and
seasonality components.

    

### [[2112.10254] Inverse deep learning methods and benchmarks for artificial electromagnetic material design](http://arxiv.org/abs/2112.10254)


  Deep learning (DL) inverse techniques have increased the speed of artificial
electromagnetic material (AEM) design and improved the quality of resulting
devices. Many DL inverse techniques have succeeded on a number of AEM design
tasks, but to compare, contrast, and evaluate assorted techniques it is
critical to clarify the underlying ill-posedness of inverse problems. Here we
review state-of-the-art approaches and present a comprehensive survey of deep
learning inverse methods and invertible and conditional invertible neural
networks to AEM design. We produce easily accessible and rapidly implementable
AEM design benchmarks, which offers a methodology to efficiently determine the
DL technique best suited to solving different design challenges. Our
methodology is guided by constraints on repeated simulation and an easily
integrated metric, which we propose expresses the relative ill-posedness of any
AEM design problem. We show that as the problem becomes increasingly ill-posed,
the neural adjoint with boundary loss (NA) generates better solutions faster,
regardless of simulation constraints. On simpler AEM design tasks, direct
neural networks (NN) fare better when simulations are limited, while geometries
predicted by mixture density networks (MDN) and conditional variational
auto-encoders (VAE) can improve with continued sampling and re-simulation.

    

### [[2112.10264] Exploration-exploitation trade-off for continuous-time episodic reinforcement learning with linear-convex models](http://arxiv.org/abs/2112.10264)


  We develop a probabilistic framework for analysing model-based reinforcement
learning in the episodic setting. We then apply it to study finite-time horizon
stochastic control problems with linear dynamics but unknown coefficients and
convex, but possibly irregular, objective function. Using probabilistic
representations, we study regularity of the associated cost functions and
establish precise estimates for the performance gap between applying optimal
feedback control derived from estimated and true model parameters. We identify
conditions under which this performance gap is quadratic, improving the linear
performance gap in recent work [X. Guo, A. Hu, and Y. Zhang, arXiv preprint,
arXiv:2104.09311, (2021)], which matches the results obtained for stochastic
linear-quadratic problems. Next, we propose a phase-based learning algorithm
for which we show how to optimise exploration-exploitation trade-off and
achieve sublinear regrets in high probability and expectation. When assumptions
needed for the quadratic performance gap hold, the algorithm achieves an order
$\mathcal{O}(\sqrt{N} \ln N)$ high probability regret, in the general case, and
an order $\mathcal{O}((\ln N)^2)$ expected regret, in self-exploration case,
over $N$ episodes, matching the best possible results from the literature. The
analysis requires novel concentration inequalities for correlated
continuous-time observations, which we derive.

    

### [[2112.10274] Estimating Causal Effects of Multi-Aspect Online Reviews with Multi-Modal Proxies](http://arxiv.org/abs/2112.10274)


  Online reviews enable consumers to engage with companies and provide
important feedback. Due to the complexity of the high-dimensional text, these
reviews are often simplified as a single numerical score, e.g., ratings or
sentiment scores. This work empirically examines the causal effects of
user-generated online reviews on a granular level: we consider multiple
aspects, e.g., the Food and Service of a restaurant. Understanding consumers'
opinions toward different aspects can help evaluate business performance in
detail and strategize business operations effectively. Specifically, we aim to
answer interventional questions such as What will the restaurant popularity be
if the quality w.r.t. its aspect Service is increased by 10%? The defining
challenge of causal inference with observational data is the presence of
"confounder", which might not be observed or measured, e.g., consumers'
preference to food type, rendering the estimated effects biased and
high-variance. To address this challenge, we have recourse to the multi-modal
proxies such as the consumer profile information and interactions between
consumers and businesses. We show how to effectively leverage the rich
information to identify and estimate causal effects of multiple aspects
embedded in online reviews. Empirical evaluations on synthetic and real-world
data corroborate the efficacy and shed light on the actionable insight of the
proposed approach.

    

### [[2112.10290] Distributionally Robust Group Backwards Compatibility](http://arxiv.org/abs/2112.10290)


  Machine learning models are updated as new data is acquired or new
architectures are developed. These updates usually increase model performance,
but may introduce backward compatibility errors, where individual users or
groups of users see their performance on the updated model adversely affected.
This problem can also be present when training datasets do not accurately
reflect overall population demographics, with some groups having overall lower
participation in the data collection process, posing a significant fairness
concern. We analyze how ideas from distributional robustness and minimax
fairness can aid backward compatibility in this scenario, and propose two
methods to directly address this issue. Our theoretical analysis is backed by
experimental results on CIFAR-10, CelebA, and Waterbirds, three standard image
classification datasets. Code available at this http URL


### [[2112.10296] Deep Surrogate for Direct Time Fluid Dynamics](http://arxiv.org/abs/2112.10296)


  The ubiquity of fluids in the physical world explains the need to accurately
simulate their dynamics for many scientific and engineering applications.
Traditionally, well established but resource intensive CFD solvers provide such
simulations. The recent years have seen a surge of deep learning surrogate
models substituting these solvers to alleviate the simulation process. Some
approaches to build data-driven surrogates mimic the solver iterative process.
They infer the next state of the fluid given its previous one. Others directly
infer the state from time input. Approaches also differ in their management of
the spatial information. Graph Neural Networks (GNN) can address the
specificity of the irregular meshes commonly used in CFD simulations. In this
article, we present our ongoing work to design a novel direct time GNN
architecture for irregular meshes. It consists of a succession of graphs of
increasing size connected by spline convolutions. We test our architecture on
the Von K{á}rm{á}n's vortex street benchmark. It achieves small
generalization errors while mitigating error accumulation along the trajectory.

    

### [[2112.10297] DXML: Distributed Extreme Multilabel Classification](http://arxiv.org/abs/2112.10297)


  As a big data application, extreme multilabel classification has emerged as
an important research topic with applications in ranking and recommendation of
products and items. A scalable hybrid distributed and shared memory
implementation of extreme classification for large scale ranking and
recommendation is proposed. In particular, the implementation is a mix of
message passing using MPI across nodes and using multithreading on the nodes
using OpenMP. The expression for communication latency and communication volume
is derived. Parallelism using work-span model is derived for shared memory
architecture. This throws light on the expected scalability of similar extreme
classification methods. Experiments show that the implementation is relatively
faster to train and test on some large datasets. In some cases, model size is
relatively small.

    

### [[2112.10307] Skin lesion segmentation and classification using deep learning and handcrafted features](http://arxiv.org/abs/2112.10307)


  Accurate diagnostics of a skin lesion is a critical task in classification
dermoscopic images. In this research, we form a new type of image features,
called hybrid features, which has stronger discrimination ability than single
method features. This study involves a new technique where we inject the
handcrafted features or feature transfer into the fully connected layer of
Convolutional Neural Network (CNN) model during the training process. Based on
our literature review until now, no study has examined or investigated the
impact on classification performance by injecting the handcrafted features into
the CNN model during the training process. In addition, we also investigated
the impact of segmentation mask and its effect on the overall classification
performance. Our model achieves an 92.3% balanced multiclass accuracy, which is
6.8% better than the typical single method classifier architecture for deep
learning.

    

### [[2112.10313] Semi-Decentralized Federated Edge Learning with Data and Device Heterogeneity](http://arxiv.org/abs/2112.10313)


  Federated edge learning (FEEL) has attracted much attention as a
privacy-preserving paradigm to effectively incorporate the distributed data at
the network edge for training deep learning models. Nevertheless, the limited
coverage of a single edge server results in an insufficient number of
participated client nodes, which may impair the learning performance. In this
paper, we investigate a novel framework of FEEL, namely semi-decentralized
federated edge learning (SD-FEEL), where multiple edge servers are employed to
collectively coordinate a large number of client nodes. By exploiting the
low-latency communication among edge servers for efficient model sharing,
SD-FEEL can incorporate more training data, while enjoying much lower latency
compared with conventional federated learning. We detail the training algorithm
for SD-FEEL with three main steps, including local model update, intra-cluster,
and inter-cluster model aggregations. The convergence of this algorithm is
proved on non-independent and identically distributed (non-IID) data, which
also helps to reveal the effects of key parameters on the training efficiency
and provides practical design guidelines. Meanwhile, the heterogeneity of edge
devices may cause the straggler effect and deteriorate the convergence speed of
SD-FEEL. To resolve this issue, we propose an asynchronous training algorithm
with a staleness-aware aggregation scheme for SD-FEEL, of which, the
convergence performance is also analyzed. The simulation results demonstrate
the effectiveness and efficiency of the proposed algorithms for SD-FEEL and
corroborate our analysis.

    

### [[2112.10327] Classifier Calibration: How to assess and improve predicted class probabilities: a survey](http://arxiv.org/abs/2112.10327)


  This paper provides both an introduction to and a detailed overview of the
principles and practice of classifier calibration. A well-calibrated classifier
correctly quantifies the level of uncertainty or confidence associated with its
instance-wise predictions. This is essential for critical applications, optimal
decision making, cost-sensitive classification, and for some types of context
change. Calibration research has a rich history which predates the birth of
machine learning as an academic field by decades. However, a recent increase in
the interest on calibration has led to new methods and the extension from
binary to the multiclass setting. The space of options and issues to consider
is large, and navigating it requires the right set of concepts and tools. We
provide both introductory material and up-to-date technical details of the main
concepts and methods, including proper scoring rules and other evaluation
metrics, visualisation approaches, a comprehensive account of post-hoc
calibration methods for binary and multiclass classification, and several
advanced topics.

    

### [[2112.10369] Feature Selection for Efficient Local-to-Global Bayesian Network Structure Learning](http://arxiv.org/abs/2112.10369)


  Local-to-global learning approach plays an essential role in Bayesian network
(BN) structure learning. Existing local-to-global learning algorithms first
construct the skeleton of a DAG (directed acyclic graph) by learning the MB
(Markov blanket) or PC (parents and children) of each variable in a data set,
then orient edges in the skeleton. However, existing MB or PC learning methods
are often computationally expensive especially with a large-sized BN, resulting
in inefficient local-to-global learning algorithms. To tackle the problem, in
this paper, we develop an efficient local-to-global learning approach using
feature selection. Specifically, we first analyze the rationale of the
well-known Minimum-Redundancy and Maximum-Relevance (MRMR) feature selection
approach for learning a PC set of a variable. Based on the analysis, we propose
an efficient F2SL (feature selection-based structure learning) approach to
local-to-global BN structure learning. The F2SL approach first employs the MRMR
approach to learn a DAG skeleton, then orients edges in the skeleton. Employing
independence tests or score functions for orienting edges, we instantiate the
F2SL approach into two new algorithms, F2SL-c (using independence tests) and
F2SL-s (using score functions). Compared to the state-of-the-art
local-to-global BN learning algorithms, the experiments validated that the
proposed algorithms in this paper are more efficient and provide competitive
structure learning quality than the compared algorithms.

    

### [[2112.10372] A Comprehensive Analytical Survey on Unsupervised and Semi-Supervised Graph Representation Learning Methods](http://arxiv.org/abs/2112.10372)


  Graph representation learning is a fast-growing field where one of the main
objectives is to generate meaningful representations of graphs in
lower-dimensional spaces. The learned embeddings have been successfully applied
to perform various prediction tasks, such as link prediction, node
classification, clustering, and visualization. The collective effort of the
graph learning community has delivered hundreds of methods, but no single
method excels under all evaluation metrics such as prediction accuracy, running
time, scalability, etc. This survey aims to evaluate all major classes of graph
embedding methods by considering algorithmic variations, parameter selections,
scalability, hardware and software platforms, downstream ML tasks, and diverse
datasets. We organized graph embedding techniques using a taxonomy that
includes methods from manual feature engineering, matrix factorization, shallow
neural networks, and deep graph convolutional networks. We evaluated these
classes of algorithms for node classification, link prediction, clustering, and
visualization tasks using widely used benchmark graphs. We designed our
experiments on top of PyTorch Geometric and DGL libraries and run experiments
on different multicore CPU and GPU platforms. We rigorously scrutinize the
performance of embedding methods under various performance metrics and
summarize the results. Thus, this paper may serve as a comparative guide to
help users select methods that are most suitable for their tasks.

    

### [[2112.10377] Learning for Robust Combinatorial Optimization: Algorithm and Application](http://arxiv.org/abs/2112.10377)


  Learning to optimize (L2O) has recently emerged as a promising approach to
solving optimization problems by exploiting the strong prediction power of
neural networks and offering lower runtime complexity than conventional
solvers. While L2O has been applied to various problems, a crucial yet
challenging class of problems -- robust combinatorial optimization in the form
of minimax optimization -- have largely remained under-explored. In addition to
the exponentially large decision space, a key challenge for robust
combinatorial optimization lies in the inner optimization problem, which is
typically non-convex and entangled with outer optimization. In this paper, we
study robust combinatorial optimization and propose a novel learning-based
optimizer, called LRCO (Learning for Robust Combinatorial Optimization), which
quickly outputs a robust solution in the presence of uncertain context. LRCO
leverages a pair of learning-based optimizers -- one for the minimizer and the
other for the maximizer -- that use their respective objective functions as
losses and can be trained without the need of labels for training problem
instances. To evaluate the performance of LRCO, we perform simulations for the
task offloading problem in vehicular edge computing. Our results highlight that
LRCO can greatly reduce the worst-case cost and improve robustness, while
having a very low runtime complexity.

    

### [[2112.10384] Multimodal Adversarially Learned Inference with Factorized Discriminators](http://arxiv.org/abs/2112.10384)


  Learning from multimodal data is an important research topic in machine
learning, which has the potential to obtain better representations. In this
work, we propose a novel approach to generative modeling of multimodal data
based on generative adversarial networks. To learn a coherent multimodal
generative model, we show that it is necessary to align different encoder
distributions with the joint decoder distribution simultaneously. To this end,
we construct a specific form of the discriminator to enable our model to
utilize data efficiently, which can be trained constrastively. By taking
advantage of contrastive learning through factorizing the discriminator, we
train our model on unimodal data. We have conducted experiments on the
benchmark datasets, whose promising results show that our proposed approach
outperforms the-state-of-the-art methods on a variety of metrics. The source
code will be made publicly available.

    

### [[2112.10401] Quasi-uniform designs with optimal and near-optimal uniformity constant](http://arxiv.org/abs/2112.10401)


  A design is a collection of distinct points in a given set $X$, which is
assumed to be a compact subset of $R^d$, and the mesh-ratio of a design is the
ratio of its fill distance to its separation radius. The uniformity constant of
a sequence of nested designs is the smallest upper bound for the mesh-ratios of
the designs. We derive a lower bound on this uniformity constant and show that
a simple greedy construction achieves this lower bound. We then extend this
scheme to allow more flexibility in the design construction.

    

### [[2112.10408] Efficient Wind Speed Nowcasting with GPU-Accelerated Nearest Neighbors Algorithm](http://arxiv.org/abs/2112.10408)


  This paper proposes a simple yet efficient high-altitude wind nowcasting
pipeline. It processes efficiently a vast amount of live data recorded by
airplanes over the whole airspace and reconstructs the wind field with good
accuracy. It creates a unique context for each point in the dataset and then
extrapolates from it. As creating such context is computationally intensive,
this paper proposes a novel algorithm that reduces the time and memory cost by
efficiently fetching nearest neighbors in a data set whose elements are
organized along smooth trajectories that can be approximated with piece-wise
linear structures.
We introduce an efficient and exact strategy implemented through algebraic
tensorial operations, which is well-suited to modern GPU-based computing
infrastructure. This method employs a scalable Euclidean metric and allows
masking data points along one dimension. When applied, this method is more
efficient than plain Euclidean k-NN and other well-known data selection methods
such as KDTrees and provides a several-fold speedup. We provide an
implementation in PyTorch and a novel data set to allow the replication of
empirical results.

    

### [[2112.10425] Model-based Clustering with Missing Not At Random Data](http://arxiv.org/abs/2112.10425)


  In recent decades, technological advances have made it possible to collect
large data sets. In this context, the model-based clustering is a very popular,
flexible and interpretable methodology for data exploration in a well-defined
statistical framework. One of the ironies of the increase of large datasets is
that missing values are more frequent. However, traditional ways (as discarding
observations with missing values or imputation methods) are not designed for
the clustering purpose. In addition, they rarely apply to the general case,
though frequent in practice, of Missing Not At Random (MNAR) values, i.e. when
the missingness depends on the unobserved data values and possibly on the
observed data values. The goal of this paper is to propose a novel approach by
embedding MNAR data directly within model-based clustering algorithms. We
introduce a selection model for the joint distribution of data and missing-data
indicator. It corresponds to a mixture model for the data distribution and a
general MNAR model for the missing-data mechanism, which may depend on the
underlying classes (unknown) and/or the values of the missing variables
themselves. A large set of meaningful MNAR sub-models is derived and the
identifiability of the parameters is studied for each of the sub-models, which
is usually a key issue for any MNAR proposals. The EM and Stochastic EM
algorithms are considered for estimation. Finally, we perform empirical
evaluations for the proposed submodels on synthetic data and we illustrate the
relevance of our method on a medical register, the TraumaBase (R) dataset.

    

### [[2112.10441] Towards Trustworthy Cross-patient Model Development](http://arxiv.org/abs/2112.10441)


  Machine learning is used in medicine to support physicians in examination,
diagnosis, and predicting outcomes. One of the most dynamic area is the usage
of patient generated health data from intensive care units. The goal of this
paper is to demonstrate how we advance cross-patient ML model development by
combining the patient's demographics data with their physiological data. We
used a population of patients undergoing Carotid Enderarterectomy (CEA), where
we studied differences in model performance and explainability when trained for
all patients and one patient at a time. The results show that patients'
demographics has a large impact on the performance and explainability and thus
trustworthiness. We conclude that we can increase trust in ML models in a
cross-patient context, by careful selection of models and patients based on
their demographics and the surgical procedure.

    

### [[2112.10459] Safe multi-agent deep reinforcement learning for joint bidding and maintenance scheduling of generation units](http://arxiv.org/abs/2112.10459)


  This paper proposes a safe reinforcement learning algorithm for generation
bidding decisions and unit maintenance scheduling in a competitive electricity
market environment. In this problem, each unit aims to find a bidding strategy
that maximizes its revenue while concurrently retaining its reliability by
scheduling preventive maintenance. The maintenance scheduling provides some
safety constraints which should be satisfied at all times. Satisfying the
critical safety and reliability constraints while the generation units have an
incomplete information of each others' bidding strategy is a challenging
problem. Bi-level optimization and reinforcement learning are state of the art
approaches for solving this type of problems. However, neither bi-level
optimization nor reinforcement learning can handle the challenges of incomplete
information and critical safety constraints. To tackle these challenges, we
propose the safe deep deterministic policy gradient reinforcement learning
algorithm which is based on a combination of reinforcement learning and a
predicted safety filter. The case study demonstrates that the proposed approach
can achieve a higher profit compared to other state of the art methods while
concurrently satisfying the system safety constraints.

    

### [[2112.10467] An iterative clustering algorithm for the Contextual Stochastic Block Model with optimality guarantees](http://arxiv.org/abs/2112.10467)


  Real-world networks often come with side information that can help to improve
the performance of network analysis tasks such as clustering. Despite a large
number of empirical and theoretical studies conducted on network clustering
methods during the past decade, the added value of side information and the
methods used to incorporate it optimally in clustering algorithms are
relatively less understood. We propose a new iterative algorithm to cluster
networks with side information for nodes (in the form of covariates) and show
that our algorithm is optimal under the Contextual Symmetric Stochastic Block
Model. Our algorithm can be applied to general Contextual Stochastic Block
Models and avoids hyperparameter tuning in contrast to previously proposed
methods. We confirm our theoretical results on synthetic data experiments where
our algorithm significantly outperforms other methods, and show that it can
also be applied to signed graphs. Finally we demonstrate the practical interest
of our method on real data.

    

### [[2112.10504] Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic](http://arxiv.org/abs/2112.10504)


  Model-based reinforcement learning algorithms, which aim to learn a model of
the environment to make decisions, are more sample efficient than their
model-free counterparts. The sample efficiency of model-based approaches relies
on whether the model can well approximate the environment. However, learning an
accurate model is challenging, especially in complex and noisy environments. To
tackle this problem, we propose the conservative model-based actor-critic
(CMBAC), a novel approach that achieves high sample efficiency without the
strong reliance on accurate learned models. Specifically, CMBAC learns multiple
estimates of the Q-value function from a set of inaccurate models and uses the
average of the bottom-k estimates -- a conservative estimate -- to optimize the
policy. An appealing feature of CMBAC is that the conservative estimates
effectively encourage the agent to avoid unreliable "promising actions" --
whose values are high in only a small fraction of the models. Experiments
demonstrate that CMBAC significantly outperforms state-of-the-art approaches in
terms of sample efficiency on several challenging tasks, and the proposed
method is more robust than previous methods in noisy environments.

    

### [[2112.10508] Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP](http://arxiv.org/abs/2112.10508)


  What are the units of text that we want to model? From bytes to multi-word
expressions, text can be analyzed and generated at many granularities. Until
recently, most natural language processing (NLP) models operated over words,
treating those as discrete and atomic tokens, but starting with byte-pair
encoding (BPE), subword-based approaches have become dominant in many areas,
enabling small vocabularies while still allowing for fast inference. Is the end
of the road character-level model or byte-level processing? In this survey, we
connect several lines of work from the pre-neural and neural era, by showing
how hybrid approaches of words and characters as well as subword-based
approaches based on learned segmentation have been proposed and evaluated. We
conclude that there is and likely will never be a silver bullet singular
solution for all applications and that thinking seriously about tokenization
remains important for many applications.

    

### [[2112.10510] Transformers Can Do Bayesian Inference](http://arxiv.org/abs/2112.10510)


  Currently, it is hard to reap the benefits of deep learning for Bayesian
methods, which allow the explicit specification of prior knowledge and
accurately capture model uncertainty. We present Prior-Data Fitted Networks
(PFNs). PFNs leverage large-scale machine learning techniques to approximate a
large set of posteriors. The only requirement for PFNs to work is the ability
to sample from a prior distribution over supervised learning tasks (or
functions). Our method restates the objective of posterior approximation as a
supervised classification problem with a set-valued input: it repeatedly draws
a task (or function) from the prior, draws a set of data points and their
labels from it, masks one of the labels and learns to make probabilistic
predictions for it based on the set-valued input of the rest of the data
points. Presented with a set of samples from a new supervised learning task as
input, PFNs make probabilistic predictions for arbitrary other data points in a
single forward propagation, having learned to approximate Bayesian inference.
We demonstrate that PFNs can near-perfectly mimic Gaussian processes and also
enable efficient Bayesian inference for intractable problems, with over
200-fold speedups in multiple setups compared to current methods. We obtain
strong results in very diverse areas such as Gaussian process regression,
Bayesian neural networks, classification for small tabular data sets, and
few-shot image classification, demonstrating the generality of PFNs. Code and
trained PFNs are released at
this https URL.

    

### [[2112.10513] Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization](http://arxiv.org/abs/2112.10513)


  Deep reinforcement learning algorithms can perform poorly in real-world tasks
due to the discrepancy between source and target environments. This discrepancy
is commonly viewed as the disturbance in transition dynamics. Many existing
algorithms learn robust policies by modeling the disturbance and applying it to
source environments during training, which usually requires prior knowledge
about the disturbance and control of simulators. However, these algorithms can
fail in scenarios where the disturbance from target environments is unknown or
is intractable to model in simulators. To tackle this problem, we propose a
novel model-free actor-critic algorithm -- namely, state-conservative policy
optimization (SCPO) -- to learn robust policies without modeling the
disturbance in advance. Specifically, SCPO reduces the disturbance in
transition dynamics to that in state space and then approximates it by a simple
gradient-based regularizer. The appealing features of SCPO include that it is
simple to implement and does not require additional knowledge about the
disturbance or specially designed simulators. Experiments in several robot
control tasks demonstrate that SCPO learns robust policies against the
disturbance in transition dynamics.

    

### [[2112.10525] Certified Federated Adversarial Training](http://arxiv.org/abs/2112.10525)


  In federated learning (FL), robust aggregation schemes have been developed to
protect against malicious clients. Many robust aggregation schemes rely on
certain numbers of benign clients being present in a quorum of workers. This
can be hard to guarantee when clients can join at will, or join based on
factors such as idle system status, and connected to power and WiFi. We tackle
the scenario of securing FL systems conducting adversarial training when a
quorum of workers could be completely malicious. We model an attacker who
poisons the model to insert a weakness into the adversarial training such that
the model displays apparent adversarial robustness, while the attacker can
exploit the inserted weakness to bypass the adversarial training and force the
model to misclassify adversarial examples. We use abstract interpretation
techniques to detect such stealthy attacks and block the corrupted model
updates. We show that this defence can preserve adversarial robustness even
against an adaptive attacker.

    

### [[2112.10526] NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems](http://arxiv.org/abs/2112.10526)


  We introduce version 3 of NetKet, the machine learning toolbox for many-body
quantum physics. NetKet is built around neural-network quantum states and
provides efficient algorithms for their evaluation and optimization. This new
version is built on top of JAX, a differentiable programming and accelerated
linear algebra framework for the Python programming language. The most
significant new feature is the possibility to define arbitrary neural network
ansätze in pure Python code using the concise notation of machine-learning
frameworks, which allows for just-in-time compilation as well as the implicit
generation of gradients thanks to automatic differentiation. NetKet 3 also
comes with support for GPU and TPU accelerators, advanced support for discrete
symmetry groups, chunking to scale up to thousands of degrees of freedom,
drivers for quantum dynamics applications, and improved modularity, allowing
users to use only parts of the toolbox as a foundation for their own code.

    

### [[2112.10551] Scope and Sense of Explainability for AI-Systems](http://arxiv.org/abs/2112.10551)


  Certain aspects of the explainability of AI systems will be critically
discussed. This especially with focus on the feasibility of the task of making
every AI system explainable. Emphasis will be given to difficulties related to
the explainability of highly complex and efficient AI systems which deliver
decisions whose explanation defies classical logical schemes of cause and
effect. AI systems have provably delivered unintelligible solutions which in
retrospect were characterized as ingenious (for example move 37 of the game 2
of AlphaGo). It will be elaborated on arguments supporting the notion that if
AI-solutions were to be discarded in advance because of their not being
thoroughly comprehensible, a great deal of the potentiality of intelligent
systems would be wasted.

    

### [[2112.10558] Lifelong Learning in Evolving Graphs with Limited Labeled Data and Unseen Class Detection](http://arxiv.org/abs/2112.10558)


  Large-scale graph data in the real-world are often dynamic rather than
static. The data are changing with new nodes, edges, and even classes appearing
over time, such as in citation networks and research-and-development
collaboration networks. Graph neural networks (GNNs) have emerged as the
standard method for numerous tasks on graph-structured data. In this work, we
employ a two-step procedure to explore how GNNs can be incrementally adapted to
new unseen graph data. First, we analyze the verge between transductive and
inductive learning on standard benchmark datasets. After inductive pretraining,
we add unlabeled data to the graph and show that the models are stable. Then,
we explore the case of continually adding more and more labeled data, while
considering cases, where not all past instances are annotated with class
labels. Furthermore, we introduce new classes while the graph evolves and
explore methods that automatically detect instances from previously unseen
classes. In order to deal with evolving graphs in a principled way, we propose
a lifelong learning framework for graph data along with an evaluation protocol.
In this framework, we evaluate representative GNN architectures. We observe
that implicit knowledge within model parameters becomes more important when
explicit knowledge, i.e., data from past tasks, is limited. We find that in
open-world node classification, the data from surprisingly few past tasks are
sufficient to reach the performance reached by remembering data from all past
tasks. In the challenging task of unseen class detection, we find that using a
weighted cross-entropy loss is important for stability.

    

### [[2112.10572] General Greedy De-bias Learning](http://arxiv.org/abs/2112.10572)


  Neural networks often make predictions relying on the spurious correlations
from the datasets rather than the intrinsic properties of the task of interest,
facing sharp degradation on out-of-distribution (OOD) test data. Existing
de-bias learning frameworks try to capture specific dataset bias by bias
annotations, they fail to handle complicated OOD scenarios. Others implicitly
identify the dataset bias by the special design on the low capability biased
model or the loss, but they degrade when the training and testing data are from
the same distribution. In this paper, we propose a General Greedy De-bias
learning framework (GGD), which greedily trains the biased models and the base
model like gradient descent in functional space. It encourages the base model
to focus on examples that are hard to solve with biased models, thus remaining
robust against spurious correlations in the test stage. GGD largely improves
models' OOD generalization ability on various tasks, but sometimes
over-estimates the bias level and degrades on the in-distribution test. We
further re-analyze the ensemble process of GGD and introduce the Curriculum
Regularization into GGD inspired by curriculum learning, which achieves a good
trade-off between in-distribution and out-of-distribution performance.
Extensive experiments on image classification, adversarial question answering,
and visual question answering demonstrate the effectiveness of our method. GGD
can learn a more robust base model under the settings of both task-specific
biased models with prior knowledge and self-ensemble biased model without prior
knowledge.

    

### [[2112.10574] Hybrid Bayesian network discovery with latent variables by scoring multiple interventions](http://arxiv.org/abs/2112.10574)


  In Bayesian Networks (BNs), the direction of edges is crucial for causal
reasoning and inference. However, Markov equivalence class considerations mean
it is not always possible to establish edge orientations, which is why many BN
structure learning algorithms cannot orientate all edges from purely
observational data. Moreover, latent confounders can lead to false positive
edges. Relatively few methods have been proposed to address these issues. In
this work, we present the hybrid mFGS-BS (majority rule and Fast Greedy
equivalence Search with Bayesian Scoring) algorithm for structure learning from
discrete data that involves an observational data set and one or more
interventional data sets. The algorithm assumes causal insufficiency in the
presence of latent variables and produces a Partial Ancestral Graph (PAG).
Structure learning relies on a hybrid approach and a novel Bayesian scoring
paradigm that calculates the posterior probability of each directed edge being
added to the learnt graph. Experimental results based on well-known networks of
up to 109 variables and 10k sample size show that mFGS-BS improves structure
learning accuracy relative to the state-of-the-art and it is computationally
efficient.

    

### [[2112.10577] NFTGAN: Non-Fungible Token Art Generation Using Generative Adversatial Networks](http://arxiv.org/abs/2112.10577)


  Digital arts have gained an unprecedented level of popularity with the
emergence of non-fungible tokens (NFTs). NFTs are cryptographic assets that are
stored on blockchain networks and represent a digital certificate of ownership
that cannot be forged. NFTs can be incorporated into a smart contract which
allows the owner to benefit from a future sale percentage. While digital art
producers can benefit immensely with NFTs, their production is time consuming.
Therefore, this paper explores the possibility of using generative adversarial
networks (GANs) for automatic generation of digital arts. GANs are deep
learning architectures that are widely and effectively used for synthesis of
audio, images, and video contents. However, their application to NFT arts have
been limited. In this paper, a GAN-based architecture is implemented and
evaluated for digital arts generation. Results from the qualitative case study
indicate the generated artworks are comparable to the real samples.

    

### [[2112.10583] A singular Riemannian geometry approach to Deep Neural Networks II. Reconstruction of 1-D equivalence classes](http://arxiv.org/abs/2112.10583)


  In a previous work, we proposed a geometric framework to study a deep neural
network, seen as sequence of maps between manifolds, employing singular
Riemannian geometry. In this paper, we present an application of this
framework, proposing a way to build the class of equivalence of an input point:
such class is defined as the set of the points on the input manifold mapped to
the same output by the neural network. In other words, we build the preimage of
a point in the output manifold in the input space. In particular. we focus for
simplicity on the case of neural networks maps from n-dimensional real spaces
to (n - 1)-dimensional real spaces, we propose an algorithm allowing to build
the set of points lying on the same class of equivalence. This approach leads
to two main applications: the generation of new synthetic data and it may
provides some insights on how a classifier can be confused by small
perturbation on the input data (e.g. a penguin image classified as an image
containing a chihuahua). In addition, for neural networks from 2D to 1D real
spaces, we also discuss how to find the preimages of closed intervals of the
real line. We also present some numerical experiments with several neural
networks trained to perform non-linear regression tasks, including the case of
a binary classifier.

    

### [[2112.10588] CGAN-EB: A Non-parametric Empirical Bayes Method for Crash Hotspot Identification Using Conditional Generative Adversarial Networks: A Real-world Crash Data Study](http://arxiv.org/abs/2112.10588)


  The empirical Bayes (EB) method based on parametric statistical models such
as the negative binomial (NB) has been widely used for ranking sites in road
network safety screening process. This paper is the continuation of the authors
previous research, where a novel non-parametric EB method for modelling crash
frequency data data based on Conditional Generative Adversarial Networks (CGAN)
was proposed and evaluated over several simulated crash data sets. Unlike
parametric approaches, there is no need for a pre-specified underlying
relationship between dependent and independent variables in the proposed
CGAN-EB and they are able to model any types of distributions. The proposed
methodology is now applied to a real-world data set collected for road segments
from 2012 to 2017 in Washington State. The performance of CGAN-EB in terms of
model fit, predictive performance and network screening outcomes is compared
with the conventional approach (NB-EB) as a benchmark. The results indicate
that the proposed CGAN-EB approach outperforms NB-EB in terms of prediction
power and hotspot identification tests.

    

### [[2112.10592] Exact Shapley Values for Local and Model-True Explanations of Decision Tree Ensembles](http://arxiv.org/abs/2112.10592)


  Additive feature explanations using Shapley values have become popular for
providing transparency into the relative importance of each feature to an
individual prediction of a machine learning model. While Shapley values provide
a unique additive feature attribution in cooperative game theory, the Shapley
values that can be generated for even a single machine learning model are far
from unique, with theoretical and implementational decisions affecting the
resulting attributions. Here, we consider the application of Shapley values for
explaining decision tree ensembles and present a novel approach to Shapley
value-based feature attribution that can be applied to random forests and
boosted decision trees. This new method provides attributions that accurately
reflect details of the model prediction algorithm for individual instances,
while being computationally competitive with one of the most widely used
current methods. We explain the theoretical differences between the standard
and novel approaches and compare their performance using synthetic and real
data.

    

### [[2112.10593] Benchmarking Safe Deep Reinforcement Learning in Aquatic Navigation](http://arxiv.org/abs/2112.10593)


  We propose a novel benchmark environment for Safe Reinforcement Learning
focusing on aquatic navigation. Aquatic navigation is an extremely challenging
task due to the non-stationary environment and the uncertainties of the robotic
platform, hence it is crucial to consider the safety aspect of the problem, by
analyzing the behavior of the trained network to avoid dangerous situations
(e.g., collisions). To this end, we consider a value-based and policy-gradient
Deep Reinforcement Learning (DRL) and we propose a crossover-based strategy
that combines gradient-based and gradient-free DRL to improve
sample-efficiency. Moreover, we propose a verification strategy based on
interval analysis that checks the behavior of the trained models over a set of
desired properties. Our results show that the crossover-based training
outperforms prior DRL approaches, while our verification allows us to quantify
the number of configurations that violate the behaviors that are described by
the properties. Crucially, this will serve as a benchmark for future research
in this domain of applications.

    

### [[2112.10599] Differentially Private Regret Minimization in Episodic Markov Decision Processes](http://arxiv.org/abs/2112.10599)


  We study regret minimization in finite horizon tabular Markov decision
processes (MDPs) under the constraints of differential privacy (DP). This is
motivated by the widespread applications of reinforcement learning (RL) in
real-world sequential decision making problems, where protecting users'
sensitive and private information is becoming paramount. We consider two
variants of DP -- joint DP (JDP), where a centralized agent is responsible for
protecting users' sensitive data and local DP (LDP), where information needs to
be protected directly on the user side. We first propose two general frameworks
-- one for policy optimization and another for value iteration -- for designing
private, optimistic RL algorithms. We then instantiate these frameworks with
suitable privacy mechanisms to satisfy JDP and LDP requirements, and
simultaneously obtain sublinear regret guarantees. The regret bounds show that
under JDP, the cost of privacy is only a lower order additive term, while for a
stronger privacy protection under LDP, the cost suffered is multiplicative.
Finally, the regret bounds are obtained by a unified analysis, which, we
believe, can be extended beyond tabular MDPs.

    

### [[2112.10609] An ensemble deep learning technique for detecting suicidal ideation from posts in social media platforms](http://arxiv.org/abs/2112.10609)


  Suicidal ideation detection from social media is an evolving research with
great challenges. Many of the people who have the tendency to suicide share
their thoughts and opinions through social media platforms. As part of many
researches it is observed that the publicly available posts from social media
contain valuable criteria to effectively detect individuals with suicidal
thoughts. The most difficult part to prevent suicide is to detect and
understand the complex risk factors and warning signs that may lead to suicide.
This can be achieved by identifying the sudden changes in a user behavior
automatically. Natural language processing techniques can be used to collect
behavioral and textual features from social media interactions and these
features can be passed to a specially designed framework to detect anomalies in
human interactions that are indicators of suicidal intentions. We can achieve
fast detection of suicidal ideation using deep learning and/or machine learning
based classification approaches. For such a purpose, we can employ the
combination of LSTM and CNN models to detect such emotions from posts of the
users. In order to improve the accuracy, some approaches like using more data
for training, using attention model to improve the efficiency of existing
models etc. could be done. This paper proposes a LSTM-Attention-CNN combined
model to analyze social media submissions to detect any underlying suicidal
intentions. During evaluations, the proposed model demonstrated an accuracy of
90.3 percent and an F1-score of 92.6 percent, which is greater than the
baseline models.

    

### [[2112.10612] Context-Based Music Recommendation Algorithm Evaluation](http://arxiv.org/abs/2112.10612)


  Artificial Intelligence (AI ) has been very successful in creating and
predicting music playlists for online users based on their data; data received
from users experience using the app such as searching the songs they like.
There are lots of current technological advancements in AI due to the
competition between music platform owners such as Spotify, Pandora, and more.
In this paper, 6 machine learning algorithms and their individual accuracy for
predicting whether a user will like a song are explored across 3 different
platforms including Weka, SKLearn, and Orange. The algorithms explored include
Logistic Regression, Naive Bayes, Sequential Minimal Optimization (SMO),
Multilayer Perceptron (Neural Network), Nearest Neighbor, and Random Forest.
With the analysis of the specific characteristics of each song provided by the
Spotify API [1], Random Forest is the most successful algorithm for predicting
whether a user will like a song with an accuracy of 84%. This is higher than
the accuracy of 82.72% found by Mungekar using the Random Forest technique and
slightly different characteristics of a song [2]. The characteristics in
Mungekars Random Forest algorithm focus more on the artist and popularity
rather than the sonic features of the songs. Removing the popularity aspect and
focusing purely on the sonic qualities improve the accuracy of recommendations.
Finally, this paper shows how song prediction can be accomplished without any
monetary investments, and thus, inspires an idea of what amazing results can be
accomplished with full financial research.

    

### [[1803.08784] Causal Modeling of Dynamical Systems](http://arxiv.org/abs/1803.08784)


  Dynamical systems are widely used in science and engineering to model systems
consisting of several interacting components. Often, they can be given a causal
interpretation in the sense that they not only model the evolution of the
states of the system's components over time, but also describe how their
evolution is affected by external interventions on the system that perturb the
dynamics. We introduce the formal framework of structural dynamical causal
models (SDCMs) that explicates the causal semantics of the system's components
as part of the model. SDCMs represent a dynamical system as a collection of
stochastic processes and specify the basic causal mechanisms that govern the
dynamics of each component as a structured system of random differential
equations of arbitrary order. SDCMs extend the versatile causal modeling
framework of structural causal models (SCMs), also known as structural equation
models (SEMs), by explicitly allowing for time-dependence. An SDCM can be
thought of as the stochastic-process version of an SCM, where the static random
variables of the SCM are replaced by dynamic stochastic processes and their
derivatives. We provide the foundations for a theory of SDCMs, by (i) formally
defining SDCMs, their solutions, stochastic interventions, and a graphical
representation; (ii) studying existence and uniqueness of the solutions for
given initial conditions; (iii) discussing under which conditions SDCMs
equilibrate to SCMs as time tends to infinity; (iv) relating the properties of
the SDCM to those of the equilibrium SCM. This correspondence enables one to
leverage the wealth of statistical tools and discovery methods available for
SCMs when studying the causal semantics of a large class of stochastic
dynamical systems. The theory is illustrated with several well-known examples
from different scientific domains.

    

### [[1903.01998] Statistically-informed deep learning for gravitational wave parameter estimation](http://arxiv.org/abs/1903.01998)


  We introduce deep learning models to estimate the masses of the binary
components of black hole mergers, $(m_1,m_2)$, and three astrophysical
properties of the post-merger compact remnant, namely, the final spin, $a_f$,
and the frequency and damping time of the ringdown oscillations of the
fundamental $\ell=m=2$ bar mode, $(\omega_R, \omega_I)$. Our neural networks
combine a modified $\texttt{WaveNet}$ architecture with contrastive learning
and normalizing flow. We validate these models against a Gaussian conjugate
prior family whose posterior distribution is described by a closed analytical
expression. Upon confirming that our models produce statistically consistent
results, we used them to estimate the astrophysical parameters $(m_1,m_2, a_f,
\omega_R, \omega_I)$ of five binary black holes: $\texttt{GW150914},
\texttt{GW170104}, \texttt{GW170814}, \texttt{GW190521}$ and
$\texttt{GW190630}$. We use $\texttt{PyCBC Inference}$ to directly compare
traditional Bayesian methodologies for parameter estimation with our
deep-learning-based posterior distributions. Our results show that our neural
network models predict posterior distributions that encode physical
correlations, and that our data-driven median results and 90$\%$ confidence
intervals are similar to those produced with gravitational wave Bayesian
analyses. This methodology requires a single V100 $\texttt{NVIDIA}$ GPU to
produce median values and posterior distributions within two milliseconds for
each event. This neural network, and a tutorial for its use, are available at
the $\texttt{Data and Learning Hub for Science}$.

    

### [[1909.05207] Introduction to Online Convex Optimization](http://arxiv.org/abs/1909.05207)


  This manuscript portrays optimization as a process. In many practical
applications the environment is so complex that it is infeasible to lay out a
comprehensive theoretical model and use classical algorithmic theory and
mathematical optimization. It is necessary as well as beneficial to take a
robust approach, by applying an optimization method that learns as one goes
along, learning from experience as more aspects of the problem are observed.
This view of optimization as a process has become prominent in varied fields
and has led to some spectacular success in modeling and systems that are now
part of our daily lives.

    

### [[1910.06358] Asymmetric Shapley values: incorporating causal knowledge into model-agnostic explainability](http://arxiv.org/abs/1910.06358)


  Explaining AI systems is fundamental both to the development of high
performing models and to the trust placed in them by their users. The Shapley
framework for explainability has strength in its general applicability combined
with its precise, rigorous foundation: it provides a common, model-agnostic
language for AI explainability and uniquely satisfies a set of intuitive
mathematical axioms. However, Shapley values are too restrictive in one
significant regard: they ignore all causal structure in the data. We introduce
a less restrictive framework, Asymmetric Shapley values (ASVs), which are
rigorously founded on a set of axioms, applicable to any AI system, and
flexible enough to incorporate any causal structure known to be respected by
the data. We demonstrate that ASVs can (i) improve model explanations by
incorporating causal information, (ii) provide an unambiguous test for unfair
discrimination in model predictions, (iii) enable sequentially incremental
explanations in time-series models, and (iv) support feature-selection studies
without the need for model retraining.

    

### [[2001.01870] MW-GAN: Multi-Warping GAN for Caricature Generation with Multi-Style Geometric Exaggeration](http://arxiv.org/abs/2001.01870)


  Given an input face photo, the goal of caricature generation is to produce
stylized, exaggerated caricatures that share the same identity as the photo. It
requires simultaneous style transfer and shape exaggeration with rich
diversity, and meanwhile preserving the identity of the input. To address this
challenging problem, we propose a novel framework called Multi-Warping GAN
(MW-GAN), including a style network and a geometric network that are designed
to conduct style transfer and geometric exaggeration respectively. We bridge
the gap between the style and landmarks of an image with corresponding latent
code spaces by a dual way design, so as to generate caricatures with arbitrary
styles and geometric exaggeration, which can be specified either through random
sampling of latent code or from a given caricature sample. Besides, we apply
identity preserving loss to both image space and landmark space, leading to a
great improvement in quality of generated caricatures. Experiments show that
caricatures generated by MW-GAN have better quality than existing methods.

    

### [[2002.08972] Differential Privacy for Eye Tracking with Temporal Correlations](http://arxiv.org/abs/2002.08972)


  New generation head-mounted displays, such as VR and AR glasses, are coming
into the market with already integrated eye tracking and are expected to enable
novel ways of human-computer interaction in numerous applications. However,
since eye movement properties contain biometric information, privacy concerns
have to be handled properly. Privacy-preservation techniques such as
differential privacy mechanisms have recently been applied to eye movement data
obtained from such displays. Standard differential privacy mechanisms; however,
are vulnerable due to temporal correlations between the eye movement
observations. In this work, we propose a novel transform-coding based
differential privacy mechanism to further adapt it to the statistics of eye
movement feature data and compare various low-complexity methods. We extend the
Fourier perturbation algorithm, which is a differential privacy mechanism, and
correct a scaling mistake in its proof. Furthermore, we illustrate significant
reductions in sample correlations in addition to query sensitivities, which
provide the best utility-privacy trade-off in the eye tracking literature. Our
results provide significantly high privacy without any essential loss in
classification accuracies while hiding personal identifiers.

    

### [[2006.01272] Shapley explainability on the data manifold](http://arxiv.org/abs/2006.01272)


  Explainability in AI is crucial for model development, compliance with
regulation, and providing operational nuance to predictions. The Shapley
framework for explainability attributes a model's predictions to its input
features in a mathematically principled and model-agnostic way. However,
general implementations of Shapley explainability make an untenable assumption:
that the model's features are uncorrelated. In this work, we demonstrate
unambiguous drawbacks of this assumption and develop two solutions to Shapley
explainability that respect the data manifold. One solution, based on
generative modelling, provides flexible access to data imputations; the other
directly learns the Shapley value-function, providing performance and stability
at the cost of flexibility. While "off-manifold" Shapley values can (i) give
rise to incorrect explanations, (ii) hide implicit model dependence on
sensitive attributes, and (iii) lead to unintelligible explanations in
higher-dimensional data, on-manifold explainability overcomes these problems.

    

### [[2006.12463] Slimming Neural Networks using Adaptive Connectivity Scores](http://arxiv.org/abs/2006.12463)


  In general, deep neural network (DNN) pruning methods fall into two
categories: 1) Weight-based deterministic constraints, and 2) Probabilistic
frameworks. While each approach has its merits and limitations there are a set
of common practical issues such as, trial-and-error to analyze sensitivity and
hyper-parameters to prune DNNs, which plague them both. In this work, we
propose a new single-shot, fully automated pruning algorithm called Slimming
Neural networks using Adaptive Connectivity Scores (SNACS). Our proposed
approach combines a probabilistic pruning framework with constraints on the
underlying weight matrices, via a novel connectivity measure, at multiple
levels to capitalize on the strengths of both approaches while solving their
deficiencies. In \alg{}, we propose a fast hash-based estimator of Adaptive
Conditional Mutual Information (ACMI), that uses a weight-based scaling
criterion, to evaluate the connectivity between filters and prune unimportant
ones. To automatically determine the limit up to which a layer can be pruned,
we propose a set of operating constraints that jointly define the upper pruning
percentage limits across all the layers in a deep network. Finally, we define a
novel sensitivity criterion for filters that measures the strength of their
contributions to the succeeding layer and highlights critical filters that need
to be completely protected from pruning. Through our experimental validation we
show that SNACS is faster by over 17x the nearest comparable method and is the
state of the art single-shot pruning method across three standard Dataset-DNN
pruning benchmarks: CIFAR10-VGG16, CIFAR10-ResNet56 and ILSVRC2012-ResNet50.

    

### [[2006.14422] Lifelong Learning of Graph Neural Networks for Open-World Node Classification](http://arxiv.org/abs/2006.14422)


  Graph neural networks (GNNs) have emerged as the standard method for numerous
tasks on graph-structured data such as node classification. However, real-world
graphs are often evolving over time and even new classes may arise. We model
these challenges as an instance of lifelong learning, in which a learner faces
a sequence of tasks and may take over knowledge acquired in past tasks. Such
knowledge may be stored explicitly as historic data or implicitly within model
parameters. In this work, we systematically analyze the influence of implicit
and explicit knowledge. Therefore, we present an incremental training method
for lifelong learning on graphs and introduce a new measure based on
$k$-neighborhood time differences to address variances in the historic data. We
apply our training method to five representative GNN architectures and evaluate
them on three new lifelong node classification datasets. Our results show that
no more than 50% of the GNN's receptive field is necessary to retain at least
95% accuracy compared to training over the complete history of the graph data.
Furthermore, our experiments confirm that implicit knowledge becomes more
important when fewer explicit knowledge is available.

    

### [[2008.01798] Physics-informed Tensor-train ConvLSTM for Volumetric Velocity Forecasting of Loop Current](http://arxiv.org/abs/2008.01798)


  According to the National Academies, a weekly forecast of velocity, vertical
structure, and duration of the Loop Current (LC) and its eddies is critical for
understanding the oceanography and ecosystem, and for mitigating outcomes of
anthropogenic and natural disasters in the Gulf of Mexico (GoM). However, this
forecast is a challenging problem since the LC behaviour is dominated by
long-range spatial connections across multiple timescales. In this paper, we
extend spatiotemporal predictive learning, showing its effectiveness beyond
video prediction, to a 4D model, i.e., a novel Physics-informed Tensor-train
ConvLSTM (PITT-ConvLSTM) for temporal sequences of 3D geospatial data
forecasting. Specifically, we propose 1) a novel 4D higher-order recurrent
neural network with empirical orthogonal function analysis to capture the
hidden uncorrelated patterns of each hierarchy, 2) a convolutional tensor-train
decomposition to capture higher-order space-time correlations, and 3) to
incorporate prior physic knowledge that is provided from domain experts by
informing the learning in latent space. The advantage of our proposed method is
clear: constrained by physical laws, it simultaneously learns good
representations for frame dependencies (both short-term and long-term
high-level dependency) and inter-hierarchical relations within each time frame.
Experiments on geospatial data collected from the GoM demonstrate that
PITT-ConvLSTM outperforms the state-of-the-art methods in forecasting the
volumetric velocity of the LC and its eddies for a period of over one week.

    

### [[2008.10749] Breaking the Communities: Characterizing community changing users using text mining and graph machine learning on Twitter](http://arxiv.org/abs/2008.10749)


  Even though the Internet and social media have increased the amount of news
and information people can consume, most users are only exposed to content that
reinforces their positions and isolates them from other ideological
communities. This environment has real consequences with great impact on our
lives like severe political polarization, easy spread of fake news, political
extremism, hate groups and the lack of enriching debates, among others.
Therefore, encouraging conversations between different groups of users and
breaking the closed community is of importance for healthy societies. In this
paper, we characterize and study users who break their community on Twitter
using natural language processing techniques and graph machine learning
algorithms. In particular, we collected 9 million Twitter messages from 1.5
million users and constructed the retweet networks. We identified their
communities and topics of discussion associated to them. With this data, we
present a machine learning framework for social media users classification
which detects "community breakers", i.e. users that swing from their closed
community to another one. A feature importance analysis in three Twitter
polarized political datasets showed that these users have low values of
PageRank, suggesting that changes are driven because their messages have no
response in their communities. This methodology also allowed us to identify
their specific topics of interest, providing a fully characterization of this
kind of users.

    

### [[2010.07230] An Evasion Attack against Stacked Capsule Autoencoder](http://arxiv.org/abs/2010.07230)


  Capsule network is a type of neural network that uses the spatial
relationship between features to classify images. By capturing the poses and
relative positions between features, its ability to recognize affine
transformation is improved, and it surpasses traditional convolutional neural
networks (CNNs) when handling translation, rotation and scaling. The Stacked
Capsule Autoencoder (SCAE) is the state-of-the-art capsule network. The SCAE
encodes an image as capsules, each of which contains poses of features and
their correlations. The encoded contents are then input into the downstream
classifier to predict the categories of the images. Existing research mainly
focuses on the security of capsule networks with dynamic routing or EM routing,
and little attention has been given to the security and robustness of the SCAE.
In this paper, we propose an evasion attack against the SCAE. After a
perturbation is generated based on the output of the object capsules in the
model, it is added to an image to reduce the contribution of the object
capsules related to the original category of the image so that the perturbed
image will be misclassified. We evaluate the attack using an image
classification experiment, and the experimental results indicate that the
attack can achieve high success rates and stealthiness. It confirms that the
SCAE has a security vulnerability whereby it is possible to craft adversarial
samples without changing the original structure of the image to fool the
classifiers. We hope that our work will make the community aware of the threat
of this attack and raise the attention given to the SCAE's security.

    

### [[2010.07384] Human-interpretable model explainability on high-dimensional data](http://arxiv.org/abs/2010.07384)


  The importance of explainability in machine learning continues to grow, as
both neural-network architectures and the data they model become increasingly
complex. Unique challenges arise when a model's input features become high
dimensional: on one hand, principled model-agnostic approaches to
explainability become too computationally expensive; on the other, more
efficient explainability algorithms lack natural interpretations for general
users. In this work, we introduce a framework for human-interpretable
explainability on high-dimensional data, consisting of two modules. First, we
apply a semantically meaningful latent representation, both to reduce the raw
dimensionality of the data, and to ensure its human interpretability. These
latent features can be learnt, e.g. explicitly as disentangled representations
or implicitly through image-to-image translation, or they can be based on any
computable quantities the user chooses. Second, we adapt the Shapley paradigm
for model-agnostic explainability to operate on these latent features. This
leads to interpretable model explanations that are both theoretically
controlled and computationally tractable. We benchmark our approach on
synthetic data and demonstrate its effectiveness on several
image-classification tasks.

    

### [[2010.09770] Learning by Competition of Self-Interested Reinforcement Learning Agents](http://arxiv.org/abs/2010.09770)


  An artificial neural network can be trained by uniformly broadcasting a
reward signal to units that implement a REINFORCE learning rule. Though this
presents a biologically plausible alternative to backpropagation in training a
network, the high variance associated with it renders it impractical to train
deep networks. The high variance arises from the inefficient structural credit
assignment since a single reward signal is used to evaluate the collective
action of all units. To facilitate structural credit assignment, we propose
replacing the reward signal to hidden units with the change in the $L^2$ norm
of the unit's outgoing weight. As such, each hidden unit in the network is
trying to maximize the norm of its outgoing weight instead of the global
reward, and thus we call this learning method \emph{Weight Maximization}. We
prove that Weight Maximization is approximately following the gradient of
rewards in expectation. In contrast to backpropagation, Weight Maximization can
be used to train both continuous-valued and discrete-valued units. Moreover,
Weight Maximization solves several major issues of backpropagation relating to
biological plausibility. Our experiments show that a network trained with
Weight Maximization can learn significantly faster than REINFORCE and slightly
slower than backpropagation. Weight Maximization illustrates an example of
cooperative behavior automatically arising from a population of self-interested
agents in a competitive game without any central coordination.

    

### [[2010.11327] Meta-Learning Guarantees for Online Receding Horizon Learning Control](http://arxiv.org/abs/2010.11327)


  In this paper we provide provable regret guarantees for an online
meta-learning receding horizon control algorithm in an iterative control
setting. We consider the setting where, in each iteration the system to be
controlled is a linear deterministic system that is different and unknown, the
cost for the controller in an iteration is a general additive cost function and
there are affine control input constraints. By analysing conditions under which
sub-linear regret is achievable, we prove that the meta-learning online
receding horizon controller achieves an average of the dynamic regret for the
controller cost that is $\tilde{O}((1+1/\sqrt{N})T^{3/4})$ with the number of
iterations $N$. Thus, we show that the worst regret for learning within an
iteration improves with experience of more iterations, with guarantee on rate
of improvement.

    

### [[2011.06445] How to Measure Gender Bias in Machine Translation: Optimal Translators, Multiple Reference Points](http://arxiv.org/abs/2011.06445)


  In this paper, as a case study, we present a systematic study of gender bias
in machine translation with Google Translate. We translated sentences
containing names of occupations from Hungarian, a language with gender-neutral
pronouns, into English. Our aim was to present a fair measure for bias by
comparing the translations to an optimal non-biased translator. When assessing
bias, we used the following reference points: (1) the distribution of men and
women among occupations in both the source and the target language countries,
as well as (2) the results of a Hungarian survey that examined if certain jobs
are generally perceived as feminine or masculine. We also studied how expanding
sentences with adjectives referring to occupations effect the gender of the
translated pronouns. As a result, we found bias against both genders, but
biased results against women are much more frequent. Translations are closer to
our perception of occupations than to objective occupational statistics.
Finally, occupations have a greater effect on translation than adjectives.

    

### [[2012.00634] Deep dynamic modeling with just two time points: Can we still allow for individual trajectories?](http://arxiv.org/abs/2012.00634)


  Longitudinal biomedical data are often characterized by a sparse time grid
and individual-specific development patterns. Specifically, in epidemiological
cohort studies and clinical registries we are facing the question of what can
be learned from the data in an early phase of the study, when only a baseline
characterization and one follow-up measurement are available. Inspired by
recent advances that allow to combine deep learning with dynamic modeling, we
investigate whether such approaches can be useful for uncovering complex
structure, in particular for an extreme small data setting with only two
observations time points for each individual. Irregular spacing in time could
then be used to gain more information on individual dynamics by leveraging
similarity of individuals. We provide a brief overview of how variational
autoencoders (VAEs), as a deep learning approach, can be linked to ordinary
differential equations (ODEs) for dynamic modeling, and then specifically
investigate the feasibility of such an approach that infers individual-specific
latent trajectories by including regularity assumptions and individuals'
similarity. We also provide a description of this deep learning approach as a
filtering task to give a statistical perspective. Using simulated data, we show
to what extent the approach can recover individual trajectories from ODE
systems with two and four unknown parameters and infer groups of individuals
with similar trajectories, and where it breaks down. The results show that such
dynamic deep learning approaches can be useful even in extreme small data
settings, but need to be carefully adapted.

    

### [[2012.01252] From One to All: Learning to Match Heterogeneous and Partially Overlapped Graphs](http://arxiv.org/abs/2012.01252)


  Recent years have witnessed a flurry of research activity in graph matching,
which aims at finding the correspondence of nodes across two graphs and lies at
the heart of many artificial intelligence applications. However, matching
heterogeneous graphs with partial overlap remains a challenging problem in
real-world applications. This paper proposes the first practical
learning-to-match method to meet this challenge. The proposed unsupervised
method adopts a novel partial OT paradigm to learn a transport plan and node
embeddings simultaneously. In a from-one-to-all manner, the entire learning
procedure is decomposed into a series of easy-to-solve sub-procedures, each of
which only handles the alignment of a single type of nodes. A mechanism for
searching the transport mass is also proposed. Experimental results demonstrate
that the proposed method outperforms state-of-the-art graph matching methods.

    

### [[2012.15755] Am I Rare? An Intelligent Summarization Approach for Identifying Hidden Anomalies](http://arxiv.org/abs/2012.15755)


  Monitoring network traffic data to detect any hidden patterns of anomalies is
a challenging and time-consuming task that requires high computing resources.
To this end, an appropriate summarization technique is of great importance,
where it can be a substitute for the original data. However, the summarized
data is under the threat of removing anomalies. Therefore, it is vital to
create a summary that can reflect the same pattern as the original data.
Therefore, in this paper, we propose an INtelligent Summarization approach for
IDENTifying hidden anomalies, called INSIDENT. The proposed approach guarantees
to keep the original data distribution in summarized data. Our approach is a
clustering-based algorithm that dynamically maps original feature space to a
new feature space by locally weighting features in each cluster. Therefore, in
new feature space, similar samples are closer, and consequently, outliers are
more detectable. Besides, selecting representatives based on cluster size keeps
the same distribution as the original data in summarized data. INSIDENT can be
used both as the preprocess approach before performing anomaly detection
algorithms and anomaly detection algorithm. The experimental results on
benchmark datasets prove a summary of the data can be a substitute for original
data in the anomaly detection task.

    

### [[2101.06570] Membership Inference Attack on Graph Neural Networks](http://arxiv.org/abs/2101.06570)


  Graph Neural Networks (GNNs), which generalize traditional deep neural
networks on graph data, have achieved state-of-the-art performance on several
graph analytical tasks. We focus on how trained GNN models could leak
information about the \emph{member} nodes that they were trained on. We
introduce two realistic settings for performing a membership inference (MI)
attack on GNNs. While choosing the simplest possible attack model that utilizes
the posteriors of the trained model (black-box access), we thoroughly analyze
the properties of GNNs and the datasets which dictate the differences in their
robustness towards MI attack. While in traditional machine learning models,
overfitting is considered the main cause of such leakage, we show that in GNNs
the additional structural information is the major contributing factor. We
support our findings by extensive experiments on four representative GNN
models. To prevent MI attacks on GNN, we propose two effective defenses that
significantly decreases the attacker's inference by up to 60% without
degradation to the target model's performance. Our code is available at
this https URL.

    

### [[2101.07385] Autonomous synthesis of metastable materials](http://arxiv.org/abs/2101.07385)


  Autonomous experimentation enabled by artificial intelligence (AI) offers a
new paradigm for accelerating scientific discovery. Non-equilibrium materials
synthesis is emblematic of complex, resource-intensive experimentation whose
acceleration would be a watershed for materials discovery and development. The
mapping of non-equilibrium synthesis phase diagrams has recently been
accelerated via high throughput experimentation but still limits materials
research because the parameter space is too vast to be exhaustively explored.
We demonstrate accelerated synthesis and exploration of metastable materials
through hierarchical autonomous experimentation governed by the Scientific
Autonomous Reasoning Agent (SARA). SARA integrates robotic materials synthesis
and characterization along with a hierarchy of AI methods that efficiently
reveal the structure of processing phase diagrams. SARA designs lateral
gradient laser spike annealing (lg-LSA) experiments for parallel materials
synthesis and employs optical spectroscopy to rapidly identify phase
transitions. Efficient exploration of the multi-dimensional parameter space is
achieved with nested active learning (AL) cycles built upon advanced machine
learning models that incorporate the underlying physics of the experiments as
well as end-to-end uncertainty quantification. With this, and the coordination
of AL at multiple scales, SARA embodies AI harnessing of complex scientific
tasks. We demonstrate its performance by autonomously mapping synthesis phase
boundaries for the Bi$_2$O$_3$ system, leading to orders-of-magnitude
acceleration in establishment of a synthesis phase diagram that includes
conditions for kinetically stabilizing $\delta$-Bi$_2$O$_3$ at room
temperature, a critical development for electrochemical technologies such as
solid oxide fuel cells.

    

### [[2103.00742] Automated Machine Learning on Graphs: A Survey](http://arxiv.org/abs/2103.00742)


  Machine learning on graphs has been extensively studied in both academic and
industry. However, as the literature on graph learning booms with a vast number
of emerging methods and techniques, it becomes increasingly difficult to
manually design the optimal machine learning algorithm for different
graph-related tasks. To solve this critical challenge, automated machine
learning (AutoML) on graphs which combines the strength of graph machine
learning and AutoML together, is gaining attention from the research community.
Therefore, we comprehensively survey AutoML on graphs in this paper, primarily
focusing on hyper-parameter optimization (HPO) and neural architecture search
(NAS) for graph machine learning. We further overview libraries related to
automated graph machine learning and in-depth discuss AutoGL, the first
dedicated open-source library for AutoML on graphs. In the end, we share our
insights on future research directions for automated graph machine learning.
This paper is the first systematic and comprehensive review of automated
machine learning on graphs to the best of our knowledge.

    

### [[2103.12857] Embracing the Disharmony in Medical Imaging: A Simple and Effective Framework for Domain Adaptation](http://arxiv.org/abs/2103.12857)


  Domain shift, the mismatch between training and testing data characteristics,
causes significant degradation in the predictive performance in multi-source
imaging scenarios. In medical imaging, the heterogeneity of population,
scanners and acquisition protocols at different sites presents a significant
domain shift challenge and has limited the widespread clinical adoption of
machine learning models. Harmonization methods which aim to learn a
representation of data invariant to these differences are the prevalent tools
to address domain shift, but they typically result in degradation of predictive
accuracy. This paper takes a different perspective of the problem: we embrace
this disharmony in data and design a simple but effective framework for
tackling domain shift. The key idea, based on our theoretical arguments, is to
build a pretrained classifier on the source data and adapt this model to new
data. The classifier can be fine-tuned for intra-site domain adaptation. We can
also tackle situations where we do not have access to ground-truth labels on
target data; we show how one can use auxiliary tasks for adaptation; these
tasks employ covariates such as age, gender and race which are easy to obtain
but nevertheless correlated to the main task. We demonstrate substantial
improvements in both intra-site domain adaptation and inter-site domain
generalization on large-scale real-world 3D brain MRI datasets for classifying
Alzheimer's disease and schizophrenia.

    

### [[2104.03220] DoubleML -- An Object-Oriented Implementation of Double Machine Learning in Python](http://arxiv.org/abs/2104.03220)


  DoubleML is an open-source Python library implementing the double machine
learning framework of Chernozhukov et al. (2018) for a variety of causal
models. It contains functionalities for valid statistical inference on causal
parameters when the estimation of nuisance parameters is based on machine
learning methods. The object-oriented implementation of DoubleML provides a
high flexibility in terms of model specifications and makes it easily
extendable. The package is distributed under the MIT license and relies on core
libraries from the scientific Python ecosystem: scikit-learn, numpy, pandas,
scipy, statsmodels and joblib. Source code, documentation and an extensive user
guide can be found at this https URL and
this https URL.

    

### [[2104.04310] Context-self contrastive pretraining for crop type semantic segmentation](http://arxiv.org/abs/2104.04310)


  In this paper, we propose a fully supervised pre-training scheme based on
contrastive learning particularly tailored to dense classification tasks. The
proposed Context-Self Contrastive Loss (CSCL) learns an embedding space that
makes semantic boundaries pop-up by use of a similarity metric between every
location in a training sample and its local context. For crop type semantic
segmentation from Satellite Image Time Series (SITS) we find performance at
parcel boundaries to be a critical bottleneck and explain how CSCL tackles the
underlying cause of that problem, improving the state-of-the-art performance in
this task. Additionally, using images from the Sentinel-2 (S2) satellite
missions we compile the largest, to our knowledge, SITS dataset densely
annotated by crop type and parcel identities, which we make publicly available
together with the data generation pipeline. Using that data we find CSCL, even
with minimal pre-training, to improve all respective baselines and present a
process for semantic segmentation at super-resolution for obtaining crop
classes at a more granular level. The code and instructions to download the
data can be found in this https URL.

    

### [[2104.10935] SoT: Delving Deeper into Classification Head for Transformer](http://arxiv.org/abs/2104.10935)


  Transformer models are not only successful in natural language processing
(NLP) but also demonstrate high potential in computer vision (CV). Despite
great advance, most of works only focus on improvement of architectures but pay
little attention to the classification head. For years transformer models base
exclusively on classification token to construct the final classifier, without
explicitly harnessing high-level word tokens. In this paper, we propose a novel
transformer model called second-order transformer (SoT), exploiting
simultaneously the classification token and word tokens for the classifier.
Specifically, we empirically disclose that high-level word tokens contain rich
information, which per se are very competent with the classifier and moreover,
are complementary to the classification token. To effectively harness such rich
information, we propose multi-headed global cross-covariance pooling with
singular value power normalization, which shares similar philosophy and thus is
compatible with the transformer block, better than commonly used pooling
methods. Then, we study comprehensively how to explicitly combine word tokens
with classification token for building the final classification head. For CV
tasks, our SoT significantly improves state-of-the-art vision transformers on
challenging benchmarks including ImageNet and ImageNet-A. For NLP tasks,
through fine-tuning based on pretrained language transformers including GPT and
BERT, our SoT greatly boosts the performance on widely used tasks such as CoLA
and RTE. Code will be available at this https URL


### [[2104.11408] Neural Mean Discrepancy for Efficient Out-of-Distribution Detection](http://arxiv.org/abs/2104.11408)


  Various approaches have been proposed for out-of-distribution (OOD) detection
by augmenting models, input examples, training sets, and optimization
objectives. Deviating from existing work, we have a simple hypothesis that
standard off-the-shelf models may already contain sufficient information about
the training set distribution which can be leveraged for reliable OOD
detection. Our empirical study on validating this hypothesis, which measures
the model activation's mean for OOD and in-distribution (ID) mini-batches,
surprisingly finds that activation means of OOD mini-batches consistently
deviate more from those of the training data. In addition, training data's
activation means can be computed offline efficiently or retrieved from batch
normalization layers as a `free lunch'. Based upon this observation, we propose
a novel metric called Neural Mean Discrepancy (NMD), which compares neural
means of the input examples and training data. Leveraging the simplicity of
NMD, we propose an efficient OOD detector that computes neural means by a
standard forward pass followed by a lightweight classifier. Extensive
experiments show that NMD outperforms state-of-the-art OOD approaches across
multiple datasets and model architectures in terms of both detection accuracy
and computational cost.

    

### [[2105.06228] SIDE: State Inference for Partially Observable Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2105.06228)


  As one of the solutions to the decentralized partially observable Markov
decision process (Dec-POMDP) problems, the value decomposition method has
achieved significant results recently. However, most value decomposition
methods require the fully observable state of the environment during training,
but this is not feasible in some scenarios where only incomplete and noisy
observations can be obtained. Therefore, we propose a novel value decomposition
framework, named State Inference for value DEcomposition (SIDE), which
eliminates the need to know the global state by simultaneously seeking
solutions to the two problems of optimal control and state inference. SIDE can
be extended to any value decomposition method to tackle partially observable
problems. By comparing with the performance of different algorithms in
StarCraft II micromanagement tasks, we verified that though without accessible
states, SIDE can infer the current state that contributes to the reinforcement
learning process based on past local observations and even achieve superior
results to many baselines in some complex scenarios.

    

### [[2105.08351] Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks](http://arxiv.org/abs/2105.08351)


  Accurate numerical solutions for the Schrödinger equation are of utmost
importance in quantum chemistry. However, the computational cost of current
high-accuracy methods scales poorly with the number of interacting particles.
Combining Monte Carlo methods with unsupervised training of neural networks has
recently been proposed as a promising approach to overcome the curse of
dimensionality in this setting and to obtain accurate wavefunctions for
individual molecules at a moderately scaling computational cost. These methods
currently do not exploit the regularity exhibited by wavefunctions with respect
to their molecular geometries. Inspired by recent successful applications of
deep transfer learning in machine translation and computer vision tasks, we
attempt to leverage this regularity by introducing a weight-sharing constraint
when optimizing neural network-based models for different molecular geometries.
That is, we restrict the optimization process such that up to 95 percent of
weights in a neural network model are in fact equal across varying molecular
geometries. We find that this technique can accelerate optimization when
considering sets of nuclear geometries of the same molecule by an order of
magnitude and that it opens a promising route towards pre-trained neural
network wavefunctions that yield high accuracy even across different molecules.

    

### [[2105.13727] Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](http://arxiv.org/abs/2105.13727)


  Momentum strategies are an important part of alternative investments and are
at the heart of commodity trading advisors (CTAs). These strategies have,
however, been found to have difficulties adjusting to rapid changes in market
conditions, such as during the 2020 market crash. In particular, immediately
after momentum turning points, where a trend reverses from an uptrend
(downtrend) to a downtrend (uptrend), time-series momentum (TSMOM) strategies
are prone to making bad bets. To improve the response to regime change, we
introduce a novel approach, where we insert an online changepoint detection
(CPD) module into a Deep Momentum Network (DMN) [1904.04912] pipeline, which
uses an LSTM deep-learning architecture to simultaneously learn both trend
estimation and position sizing. Furthermore, our model is able to optimise the
way in which it balances 1) a slow momentum strategy which exploits persisting
trends, but does not overreact to localised price moves, and 2) a fast
mean-reversion strategy regime by quickly flipping its position, then swapping
it back again to exploit localised price moves. Our CPD module outputs a
changepoint location and severity score, allowing our model to learn to respond
to varying degrees of disequilibrium, or smaller and more localised
changepoints, in a data driven manner. Back-testing our model over the period
1995-2020, the addition of the CPD module leads to an improvement in Sharpe
ratio of one-third. The module is especially beneficial in periods of
significant nonstationarity, and in particular, over the most recent years
tested (2015-2020) the performance boost is approximately two-thirds. This is
interesting as traditional momentum strategies have been underperforming in
this period.

    

### [[2106.01655] Hierarchical Representation Learning for Markov Decision Processes](http://arxiv.org/abs/2106.01655)


  In this paper we present a novel method for learning hierarchical
representations of Markov decision processes. Our method works by partitioning
the state space into subsets, and defines subtasks for performing transitions
between the partitions. We formulate the problem of partitioning the state
space as an optimization problem that can be solved using gradient descent
given a set of sampled trajectories, making our method suitable for
high-dimensional problems with large state spaces. We empirically validate the
method, by showing that it can successfully learn a useful hierarchical
representation in a navigation domain. Once learned, the hierarchical
representation can be used to solve different tasks in the given domain, thus
generalizing knowledge across tasks.

    

### [[2106.02346] Provably Strict Generalisation Benefit for Invariance in Kernel Methods](http://arxiv.org/abs/2106.02346)


  It is a commonly held belief that enforcing invariance improves
generalisation. Although this approach enjoys widespread popularity, it is only
very recently that a rigorous theoretical demonstration of this benefit has
been established. In this work we build on the function space perspective of
Elesedy and Zaidi arXiv:2102.10333 to derive a strictly non-zero generalisation
benefit of incorporating invariance in kernel ridge regression when the target
is invariant to the action of a compact group. We study invariance enforced by
feature averaging and find that generalisation is governed by a notion of
effective dimension that arises from the interplay between the kernel and the
group. In building towards this result, we find that the action of the group
induces an orthogonal decomposition of both the reproducing kernel Hilbert
space and its kernel, which may be of interest in its own right.

    

### [[2106.06788] Learngene: From Open-World to Your Learning Task](http://arxiv.org/abs/2106.06788)


  Although deep learning has made significant progress on fixed large-scale
datasets, it typically encounters challenges regarding improperly detecting
unknown/unseen classes in the open-world scenario, over-parametrized, and
overfitting small samples. Since biological systems can overcome the above
difficulties very well, individuals inherit an innate gene from collective
creatures that have evolved over hundreds of millions of years and then learn
new skills through few examples. Inspired by this, we propose a practical
collective-individual paradigm where an evolution (expandable) network is
trained on sequential tasks and then recognize unknown classes in real-world.
Moreover, the learngene, i.e., the gene for learning initialization rules of
the target model, is proposed to inherit the meta-knowledge from the collective
model and reconstruct a lightweight individual model on the target task.
Particularly, a novel criterion is proposed to discover learngene in the
collective model, according to the gradient information. Finally, the
individual model is trained only with few samples on the target learning tasks.
We demonstrate the effectiveness of our approach in an extensive empirical
study and theoretical analysis.

    

### [[2106.10086] It's FLAN time! Summing feature-wise latent representations for interpretability](http://arxiv.org/abs/2106.10086)


  Interpretability has become a necessary feature for machine learning models
deployed in critical scenarios, e.g. legal system, healthcare. In these
situations, algorithmic decisions may have (potentially negative) long-lasting
effects on the end-user affected by the decision. In many cases, the
representational power of deep learning models is not needed, therefore simple
and interpretable models (e.g. linear models) should be preferred. However, in
high-dimensional and/or complex domains (e.g. computer vision), the universal
approximation capabilities of neural networks are required. Inspired by linear
models and the Kolmogorov-Arnold representation theorem, we propose a novel
class of structurally-constrained neural networks, which we call FLANs
(Feature-wise Latent Additive Networks). Crucially, FLANs process each input
feature separately, computing for each of them a representation in a common
latent space. These feature-wise latent representations are then simply summed,
and the aggregated representation is used for prediction. These constraints
(which are at the core of the interpretability of linear models) allow a user
to estimate the effect of each individual feature independently from the
others, enhancing interpretability. In a set of experiments across different
domains, we show how without compromising excessively the test performance, the
structural constraints proposed in FLANs indeed facilitates the
interpretability of deep learning models. We quantitatively compare FLANs
interpretability to post-hoc methods using recently introduced metrics,
discussing the advantages of natively interpretable models over a post-hoc
analysis.

    

### [[2106.13687] panda-gym: Open-source goal-conditioned environments for robotic learning](http://arxiv.org/abs/2106.13687)


  This paper presents panda-gym, a set of Reinforcement Learning (RL)
environments for the Franka Emika Panda robot integrated with OpenAI Gym. Five
tasks are included: reach, push, slide, pick & place and stack. They all follow
a Multi-Goal RL framework, allowing to use goal-oriented RL algorithms. To
foster open-research, we chose to use the open-source physics engine PyBullet.
The implementation chosen for this package allows to define very easily new
tasks or new robots. This paper also presents a baseline of results obtained
with state-of-the-art model-free off-policy algorithms. panda-gym is
open-source and freely available at this https URL.

    

### [[2106.15098] Graph Piece: Efficiently Generating High-Quality Molecular Graphs with Substructures](http://arxiv.org/abs/2106.15098)


  Molecule generation, which requires generating valid molecules with desired
properties, is a fundamental but challenging task. Recent years have witnessed
the rapid development of atom-level auto-regressive models, which usually
construct graphs following sequential actions of adding atom-level nodes and
edges. However, these atom-level models ignore high-frequency substructures,
which not only capture the regularities of atomic combination in molecules but
are also often related to desired chemical properties, and therefore may be
sub-optimal for generating high-quality molecules. In this paper, we propose a
method to automatically discover such common substructures, which we call graph
pieces, from given molecular graphs. We also present a graph piece variational
autoencoder (GP-VAE) for generating molecular graphs based on graph pieces.
Experiments show that our GP-VAE models not only achieve better performance
than the state-of-the-art baseline for distribution-learning, property
optimization, and constrained property optimization tasks but are also
computationally efficient.

    

### [[2106.15214] Joint Majorization-Minimization for Nonnegative Matrix Factorization with the $β$-divergence](http://arxiv.org/abs/2106.15214)


  This article proposes new multiplicative updates for nonnegative matrix
factorization (NMF) with the $\beta$-divergence objective function. Our new
updates are derived from a joint majorization-minimization (MM) scheme, in
which an auxiliary function (a tight upper bound of the objective function) is
built for the two factors jointly and minimized at each iteration. This is in
contrast with the classic approach in which a majorizer is derived for each
factor separately. Like that classic approach, our joint MM algorithm also
results in multiplicative updates that are simple to implement. They however
yield a significant drop of computation time (for equally good solutions), in
particular for some $\beta$-divergences of important applicative interest, such
as the squared Euclidean distance and the Kullback-Leibler or Itakura-Saito
divergences. We report experimental results using diverse datasets: face
images, an audio spectrogram, hyperspectral data and song play counts.
Depending on the value of $\beta$ and on the dataset, our joint MM approach can
yield CPU time reductions from about $13\%$ to $78\%$ in comparison to the
classic alternating scheme.

    

### [[2107.00996] DeformRS: Certifying Input Deformations with Randomized Smoothing](http://arxiv.org/abs/2107.00996)


  Deep neural networks are vulnerable to input deformations in the form of
vector fields of pixel displacements and to other parameterized geometric
deformations e.g. translations, rotations, etc. Current input deformation
certification methods either 1. do not scale to deep networks on large input
datasets, or 2. can only certify a specific class of deformations, e.g. only
rotations. We reformulate certification in randomized smoothing setting for
both general vector field and parameterized deformations and propose
DeformRS-VF and DeformRS-Par, respectively. Our new formulation scales to large
networks on large input datasets. For instance, DeformRS-Par certifies rich
deformations, covering translations, rotations, scaling, affine deformations,
and other visually aligned deformations such as ones parameterized by
Discrete-Cosine-Transform basis. Extensive experiments on MNIST, CIFAR10, and
ImageNet show competitive performance of DeformRS-Par achieving a certified
accuracy of $39\%$ against perturbed rotations in the set
$[-10\degree,10\degree]$ on ImageNet.

    

### [[2111.00903] Towards a theory of quantum gravity from neural networks](http://arxiv.org/abs/2111.00903)


  Neural network is a dynamical system described by two different types of
degrees of freedom: fast-changing non-trainable variables (e.g. state of
neurons) and slow-changing trainable variables (e.g. weights and biases). We
show that the non-equilibrium dynamics of trainable variables can be described
by the Madelung equations, if the number of neurons is fixed, and by the
Schrodinger equation, if the learning system is capable of adjusting its own
parameters such as the number of neurons, step size and mini-batch size. We
argue that the Lorentz symmetries and curved space-time can emerge from the
interplay between stochastic entropy production and entropy destruction due to
learning. We show that the non-equilibrium dynamics of non-trainable variables
can be described by the geodesic equation (in the emergent space-time) for
localized states of neurons, and by the Einstein equations (with cosmological
constant) for the entire network. We conclude that the quantum description of
trainable variables and the gravitational description of non-trainable
variables are dual in the sense that they provide alternative macroscopic
descriptions of the same learning system, defined microscopically as a neural
network.

    

### [[2111.00916] A multi-task learning-based optimization approach for finding diverse sets of material microstructures with desired properties and its application to texture optimization](http://arxiv.org/abs/2111.00916)


  The optimization along the chain processing-structure-properties-performance
is one of the core objectives in data-driven materials science. In this sense,
processes are supposed to manufacture workpieces with targeted material
microstructures. These microstructures are defined by the material properties
of interest and identifying them is a question of materials design. In the
present paper, we addresse this issue and introduce a generic multi-task
learning-based optimization approach. The approach enables the identification
of sets of highly diverse microstructures for given desired properties and
corresponding tolerances. Basically, the approach consists of an optimization
algorithm that interacts with a machine learning model that combines multi-task
learning with siamese neural networks. The resulting model (1) relates
microstructures and properties, (2) estimates the likelihood of a
microstructure of being producible, and (3) performs a distance preserving
microstructure feature extraction in order to generate a lower dimensional
latent feature space to enable efficient optimization. The proposed approach is
applied on a crystallographic texture optimization problem for rolled steel
sheets given desired properties.

    

### [[2111.14737] Optimal No-Regret Learning in General Games: Bounded Regret with Unbounded Step-Sizes via Clairvoyant MWU](http://arxiv.org/abs/2111.14737)


  In this paper we solve the problem of no-regret learning in general games.
Specifically, we provide a simple and practical algorithm that achieves
constant regret with fixed step-sizes. The cumulative regret of our algorithm
provably decreases linearly as the step-size increases. Our findings depart
from the prevailing paradigm that vanishing step-sizes are a prerequisite for
low regret as championed by all state-of-the-art methods to date.
We shift away from this paradigm by defining a novel algorithm that we call
Clairvoyant Multiplicative Weights Updates (CMWU). CMWU is Multiplicative
Weights Updates (MWU) equipped with a mental model (jointly shared across all
agents) about the state of the system in its next period. Each agent records
its mixed strategy, i.e., its belief about what it expects to play in the next
period, in this shared mental model which is internally updated using MWU
without any changes to the real-world behavior up until it equilibrates, thus
marking its consistency with the next day's real-world outcome. It is then and
only then that agents take action in the real-world, effectively doing so with
the "full knowledge" of the state of the system on the next day, i.e., they are
clairvoyant. CMWU effectively acts as MWU with one day look-ahead, achieving
bounded regret. At a technical level, we establish that self-consistent mental
models exist for any choice of step-sizes and provide bounds on the step-size
under which their uniqueness and linear-time computation are guaranteed via
contraction mapping arguments. Our arguments extend well beyond normal-form
games with little effort.

    

### [[2111.14822] Vector Quantized Diffusion Model for Text-to-Image Synthesis](http://arxiv.org/abs/2111.14822)


  We present the vector quantized diffusion (VQ-Diffusion) model for
text-to-image generation. This method is based on a vector quantized
variational autoencoder (VQ-VAE) whose latent space is modeled by a conditional
variant of the recently developed Denoising Diffusion Probabilistic Model
(DDPM). We find that this latent-space method is well-suited for text-to-image
generation tasks because it not only eliminates the unidirectional bias with
existing methods but also allows us to incorporate a mask-and-replace diffusion
strategy to avoid the accumulation of errors, which is a serious problem with
existing methods. Our experiments show that the VQ-Diffusion produces
significantly better text-to-image generation results when compared with
conventional autoregressive (AR) models with similar numbers of parameters.
Compared with previous GAN-based text-to-image methods, our VQ-Diffusion can
handle more complex scenes and improve the synthesized image quality by a large
margin. Finally, we show that the image generation computation in our method
can be made highly efficient by reparameterization. With traditional AR
methods, the text-to-image generation time increases linearly with the output
image resolution and hence is quite time consuming even for normal size images.
The VQ-Diffusion allows us to achieve a better trade-off between quality and
speed. Our experiments indicate that the VQ-Diffusion model with the
reparameterization is fifteen times faster than traditional AR methods while
achieving a better image quality.

    

### [[2112.02845] Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks](http://arxiv.org/abs/2112.02845)


  Offline reinforcement learning leverages previously-collected offline
datasets to learn optimal policies with no necessity to access the real
environment. Such a paradigm is also desirable for multi-agent reinforcement
learning (MARL) tasks, given the increased interactions among agents and with
the enviroment. Yet, in MARL, the paradigm of offline pre-training with online
fine-tuning has not been studied, nor datasets or benchmarks for offline MARL
research are available. In this paper, we facilitate the research by providing
large-scale datasets, and use them to examine the usage of the Decision
Transformer in the context of MARL. We investigate the generalisation of MARL
offline pre-training in the following three aspects: 1) between single agents
and multiple agents, 2) from offline pretraining to the online fine-tuning, and
3) to that of multiple downstream tasks with few-shot and zero-shot
capabilities. We start by introducing the first offline MARL dataset with
diverse quality levels based on the StarCraftII environment, and then propose
the novel architecture of multi-agent decision transformer (MADT) for effective
offline learning. MADT leverages transformer's modelling ability of sequence
modelling and integrates it seamlessly with both offline and online MARL tasks.
A crucial benefit of MADT is that it learns generalizable policies that can
transfer between different types of agents under different task scenarios. On
StarCraft II offline dataset, MADT outperforms the state-of-the-art offline RL
baselines. When applied to online tasks, the pre-trained MADT significantly
improves sample efficiency, and enjoys strong performance both few-short and
zero-shot cases. To our best knowledge, this is the first work that studies and
demonstrates the effectiveness of offline pre-trained models in terms of sample
efficiency and generalisability enhancements in MARL.

    

### [[2112.02706] Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](http://arxiv.org/abs/2112.02706)


  Continual learning (CL) learns a sequence of tasks incrementally with the
goal of achieving two main objectives: overcoming catastrophic forgetting (CF)
and encouraging knowledge transfer (KT) across tasks. However, most existing
techniques focus only on overcoming CF and have no mechanism to encourage KT,
and thus do not do well in KT. Although several papers have tried to deal with
both CF and KT, our experiments show that they suffer from serious CF when the
tasks do not have much shared knowledge. Another observation is that most
current CL methods do not use pre-trained models, but it has been shown that
such models can significantly improve the end task performance. For example, in
natural language processing, fine-tuning a BERT-like pre-trained language model
is one of the most effective approaches. However, for CL, this approach suffers
from serious CF. An interesting question is how to make the best use of
pre-trained models for CL. This paper proposes a novel model called CTR to
solve these problems. Our experimental results demonstrate the effectiveness of
CTR

    

### [[2112.02714] CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks](http://arxiv.org/abs/2112.02714)


  This paper studies continual learning (CL) of a sequence of aspect sentiment
classification(ASC) tasks in a particular CL setting called domain incremental
learning (DIL). Each task is from a different domain or product. The DIL
setting is particularly suited to ASC because in testing the system needs not
know the task/domain to which the test data belongs. To our knowledge, this
setting has not been studied before for ASC. This paper proposes a novel model
called CLASSIC. The key novelty is a contrastive continual learning method that
enables both knowledge transfer across tasks and knowledge distillation from
old tasks to the new task, which eliminates the need for task ids in testing.
Experimental results show the high effectiveness of CLASSIC.

    

### [[2112.03271] Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks](http://arxiv.org/abs/2112.03271)


  This paper studies continual learning (CL) of a sequence of aspect sentiment
classification (ASC) tasks. Although some CL techniques have been proposed for
document sentiment classification, we are not aware of any CL work on ASC. A CL
system that incrementally learns a sequence of ASC tasks should address the
following two issues: (1) transfer knowledge learned from previous tasks to the
new task to help it learn a better model, and (2) maintain the performance of
the models for previous tasks so that they are not forgotten. This paper
proposes a novel capsule network based model called B-CL to address these
issues. B-CL markedly improves the ASC performance on both the new task and the
old tasks via forward and backward knowledge transfer. The effectiveness of
B-CL is demonstrated through extensive experiments.

    

### [[2112.08451] Quantum Algorithms for Reinforcement Learning with a Generative Model](http://arxiv.org/abs/2112.08451)


  Reinforcement learning studies how an agent should interact with an
environment to maximize its cumulative reward. A standard way to study this
question abstractly is to ask how many samples an agent needs from the
environment to learn an optimal policy for a $\gamma$-discounted Markov
decision process (MDP). For such an MDP, we design quantum algorithms that
approximate an optimal policy ($\pi^*$), the optimal value function ($v^*$),
and the optimal $Q$-function ($q^*$), assuming the algorithms can access
samples from the environment in quantum superposition. This assumption is
justified whenever there exists a simulator for the environment; for example,
if the environment is a video game or some other program. Our quantum
algorithms, inspired by value iteration, achieve quadratic speedups over the
best-possible classical sample complexities in the approximation accuracy
($\epsilon$) and two main parameters of the MDP: the effective time horizon
($\frac{1}{1-\gamma}$) and the size of the action space ($A$). Moreover, we
show that our quantum algorithm for computing $q^*$ is optimal by proving a
matching quantum lower bound.

    

### [[2112.09775] Adaptive Subsampling for ROI-based Visual Tracking: Algorithms and FPGA Implementation](http://arxiv.org/abs/2112.09775)


  There is tremendous scope for improving the energy efficiency of embedded
vision systems by incorporating programmable region-of-interest (ROI) readout
in the image sensor design. In this work, we study how ROI programmability can
be leveraged for tracking applications by anticipating where the ROI will be
located in future frames and switching pixels off outside of this region. We
refer to this process of ROI prediction and corresponding sensor configuration
as adaptive subsampling. Our adaptive subsampling algorithms comprise an object
detector and an ROI predictor (Kalman filter) which operate in conjunction to
optimize the energy efficiency of the vision pipeline with the end task being
object tracking. To further facilitate the implementation of our adaptive
algorithms in real life, we select a candidate algorithm and map it onto an
FPGA. Leveraging Xilinx Vitis AI tools, we designed and accelerated a YOLO
object detector-based adaptive subsampling algorithm. In order to further
improve the algorithm post-deployment, we evaluated several competing baselines
on the OTB100 and LaSOT datasets. We found that coupling the ECO tracker with
the Kalman filter has a competitive AUC score of 0.4568 and 0.3471 on the
OTB100 and LaSOT datasets respectively. Further, the power efficiency of this
algorithm is on par with, and in a couple of instances superior to, the other
baselines. The ECO-based algorithm incurs a power consumption of approximately
4 W averaged across both datasets while the YOLO-based approach requires power
consumption of approximately 6 W (as per our power consumption model). In terms
of accuracy-latency tradeoff, the ECO-based algorithm provides near-real-time
performance (19.23 FPS) while managing to attain competitive tracking
precision.

    

### [[2112.10028] Attacks of the Knights: Exploiting Non Uniform Cache Access Time](http://arxiv.org/abs/2112.10028)


  Intel Knights Landing Processors have shared last level cache (LLC) across
all the tiles using MESIF protocol and uses a mesh network of Caching and
Homing Agents(CHA)s. Due to the structure of the network, the cache access is
non uniform in nature having significant difference in cache hit times. In this
paper, we try to exploit this idea to leak secret from a victim process. First,
we show a naive implementation of the attack using a gem5 simulator that
achieves 100\% accuracy of extracting the secret bits. Then we replicate the
attack in a Intel Xeon Phi 7290@ 1.50 GHz Knight's Landing CPU to show the
efficacy of the attack. In real machine we can leak the secret from a victim
process at 85\% accuracy and ~350 kbps bandwidth. All the attacks were done on
a machine without any root or sudo privileges, so this shows the strength of
the attack. This can be further extended to leak secrets from different
processes given the vulnerable patterns may exist in many libraries. Other
processors with similar architecture (last level distributed cache in mesh
networks) can also be vulnerable to similar attack strategy.

    

### [[2112.10034] COX: CUDA on X86 by Exposing Warp-Level Functions to CPUs](http://arxiv.org/abs/2112.10034)


  As CUDA programs become the de facto program among data parallel applications
such as high-performance computing or machine learning applications, running
CUDA on other platforms has been a compelling option. Although several efforts
have attempted to support CUDA on other than NVIDIA GPU devices, due to extra
steps in the translation, the support is always behind a few years from
supporting CUDA's latest features. The examples are DPC, Hipfy, where CUDA
source code have to be translated to their native supporting language and then
they are supported. In particular, the new CUDA programming model exposes the
warp concept in the programming language, which greatly changes the way the
CUDA code should be mapped to CPU programs. In this paper, hierarchical
collapsing that \emph{correctly} supports CUDA warp-level functions on CPUs is
proposed. Based on hierarchical collapsing, a framework, COX, is developed that
allows CUDA programs with the latest features to be executed efficiently on CPU
platforms. COX consists of a compiler IR transformation (new LLVM pass) and a
runtime system to execute the transformed programs on CPU devices. COX can
support the most recent CUDA features, and the application coverage is much
higher (90%) than for previous frameworks (68%) with comparable performance. We
also show that the warp-level functions in CUDA can be efficiently executed by
utilizing CPU SIMD (AVX) instructions.

    

### [[2112.10037] FSpGEMM: An OpenCL-based HPC Framework for Accelerating General Sparse Matrix-Matrix Multiplication on FPGAs](http://arxiv.org/abs/2112.10037)


  General sparse matrix-matrix multiplication (SpGEMM) is an integral part of
many scientific computing, high-performance computing (HPC), and graph analytic
applications. This paper presents a new compressed sparse vector (CSV) format
for representing sparse matrices and FSpGEMM, an OpenCL-based HPC framework for
accelerating general sparse matrix-matrix multiplication on FPGAs. The proposed
FSpGEMM framework includes an FPGA kernel implementing a throughput-optimized
hardware architecture based on Gustavson's algorithm and a host program
implementing pre-processing functions for converting input matrices to the CSV
format tailored for the proposed architecture. FSpGEMM utilizes a new buffering
scheme tailored to Gustavson's algorithm. We compare FSpGEMM implemented on an
Intel Arria 10 GX FPGA development board with Intel Math Kernel Library (MKL)
implemented on an Intel Xeon E5-2637 CPU and cuSPARSE on an NVIDIA GTX TITAN X
GPU, respectively, for multiplying a set of sparse matrices selected from
SuiteSparse Matrix Collection. The experiment results show that the proposed
FSpGEMM solution achieves on average 4.9x and 1.7x higher performance with
31.9x and 13.1x lower energy consumption per SpGEMM computation than the CPU
and GPU implementations, respectively.

    

### [[2112.10486] Dijkstra-Through-Time: Ahead of time hardware scheduling method for deterministic workloads](http://arxiv.org/abs/2112.10486)


  Most of the previous works on data flow optimizations for Machine Learning
hardware accelerators try to find algorithmic re-factorization such as
loop-reordering and loop-tiling. However, the analysis and information they
provide are still at very high level and one must further map them onto
instructions that hardware can understand. This paper presents
"Dijkstra-Through-Time" (DTT), an ahead of time compute and memory
scheduling-mapping algorithm for deterministic workloads. It provides a simple
implementation and supports accelerators with complex NoC configurations, at
the expense of a long compilation process. This initial paper illustrates a
proof of concept implementation to merge scheduling and data cache coherence
mechanisms to get more optimized data flows.

    

### [[2112.10511] Relational Models of Microarchitectures for Formal Security Analyses](http://arxiv.org/abs/2112.10511)


  There is a growing need for hardware-software contracts which precisely
define the implications of microarchitecture on software security-i.e.,
security contracts. It is our view that such contracts should explicitly
account for microarchitecture-level implementation details that underpin
hardware leakage, thereby establishing a direct correspondence between a
contract and the microarchitecture it represents. At the same time, these
contracts should remain as abstract as possible so as to support efficient
formal analyses. With these goals in mind, we propose leakage containment
models (LCMs)-novel axiomatic security contracts which support formally
reasoning about the security guarantees of programs when they run on particular
microarchitectures. Our core contribution is an axiomatic vocabulary for
formally defining LCMs, derived from the established axiomatic vocabulary used
to formalize processor memory consistency models. Using this vocabulary, we
formalize microarchitectural leakage-focusing on leakage through hardware
memory systems-so that it can be automatically detected in programs. To
illustrate the efficacy of LCMs, we present two case studies. First, we
demonstrate that our leakage definition faithfully captures a sampling of
(transient and non-transient) microarchitectural attacks from the literature.
Second, we develop a static analysis tool based on LCMs which automatically
identifies Spectre vulnerabilities in programs and scales to analyze
realistic-sized codebases, like libsodium.

    

### [[2112.10537] Efficient Floating Point Arithmetic for Quantum Computers](http://arxiv.org/abs/2112.10537)


  One of the major promises of quantum computing is the realization of SIMD
(single instruction - multiple data) operations using the phenomenon of
superposition. Since the dimension of the state space grows exponentially with
the number of qubits, we can easily reach situations where we pay less than a
single quantum gate per data point for data-processing instructions which would
be rather expensive in classical computing. Formulating such instructions in
terms of quantum gates, however, still remains a challenging task. Laying out
the foundational functions for more advanced data-processing is therefore a
subject of paramount importance for advancing the realm of quantum computing.
In this paper, we introduce the formalism of encoding so called-semi-boolean
polynomials. As it turns out, arithmetic $\mathbb{Z}/2^n\mathbb{Z}$ ring
operations can be formulated as semi-boolean polynomial evaluations, which
allows convenient generation of unsigned integer arithmetic quantum circuits.
For arithmetic evaluations, the resulting algorithm has been known as
Fourier-arithmetic. We extend this type of algorithm with additional features,
such as ancilla-free in-place multiplication and integer coefficient polynomial
evaluation. Furthermore, we introduce a tailor-made method for encoding signed
integers succeeded by an encoding for arbitrary floating-point numbers. This
representation of floating-point numbers and their processing can be applied to
any quantum algorithm that performs unsigned modular integer arithmetic. We
discuss some further performance enhancements of the semi boolean polynomial
encoder and finally supply a complexity estimation. The application of our
methods to a 32-bit unsigned integer multiplication demonstrated a 90\% circuit
depth reduction compared to carry-ripple approaches.

    

### [[2112.09691] Past, Present and Future of Computational Storage: A Survey](http://arxiv.org/abs/2112.09691)


  We live in a data-centric world where we are heading to generate close to 200
Zettabytes of data by the year 2025. Our data processing requirements have also
increased as we push to build data processing frameworks that can process large
volumes of data in a short duration, a few milli- and even micro-seconds. In
the prevalent computer systems designs, data is stored passively in storage
devices which is brought in for processing and then the results are written
out. As the volume of data explodes this constant data movement has led to a
"data movement wall" which hinders further process and optimizations in data
processing systems designs. One promising alternative to this architecture is
to push computation to the data (instead of the other way around), and design a
computational-storage device or CSD. The idea of CSD is not new and can trace
its root to the pioneering work done in the 1970s and 1990s. More recently,
with the emergence of non-volatile memory (NVM) storage in the mainstream
computing (e.g., NAND flash and Optane), the idea has again gained a lot of
traction with multiple academic and commercial prototypes being available now.
In this brief survey we present a systematic analysis of work done in the area
of computation storage and present future directions.

    

### [[2112.09756] Towards Harmonious Decentralization of Energy Systems: A Vision of Interoperable Peer-to-Peer Energy Markets](http://arxiv.org/abs/2112.09756)


  We present a hierarchical framework aimed at decentralizing the distribution
systems market operations using localized peer-to-peer energy markets.
Hierarchically designed decision-making algorithm approaches the power systems
market operations from a bottom-up perspective. The three layers of the
hierarchical framework operate in orchestration to enable prosumers (the
grass-root actors) to maximize their revenues - hence, a prosumer-centric
framework. The design of the framework incorporates existing smart grid
technologies (Virtual Power Plants, Microgrids, Distributed Energy Resources)
and redefine their functional objectives to align them with the
decentralization paradigm focused on empowering bottom-up grid operations
approach. On one hand, the framework is enabling prosumers with simultaneous
access to the buy-sell choices that help them maximize their cost savings while
ensuring their consumption patterns and preferences are not being tradeoff as a
result of top-down operational decisions. On the other hand, it is designed to
operate in harmony with the existing top-down grid operations mechanisms -
thereby reducing the potential friction in its adaptation. This marriage of the
top-down and bottom-up operational approaches is facilitated through meticulous
orchestration of operational timescales. Framework's novel design also
incorporates scalability and interoperability considerations, thereby tackling
the challenge of decentralization holistically.

    

### [[2112.09761] Efficient and Scalable Graph Pattern Mining on GPUs](http://arxiv.org/abs/2112.09761)


  We describe G2Miner, the first Graph Pattern Mining (GPM) framework that runs
on multiple GPUs. G2Miner uses pattern-aware, input-aware and
architecture-aware search strategies to achieve high efficiency on GPUs. To
simplify programming, it provides a code generator that automatically generates
pattern-aware CUDA code. G2Miner flexibly supports both breadth-first search
(BFS) and depth-first search (DFS) to maximize memory utilization and generate
sufficient parallelism for GPUs. For the scalability of G2Miner, we use a
customized scheduling policy to balance work among multiple GPUs. Experiments
on a V100 GPU show that G2Miner achieves average speedups of 5.4x and 7.2x over
two state-of-the-art single-GPU systems, Pangolin and PBE, respectively. In the
multi-GPU setting, G2Miner achieves linear speedups from 1 to 8 GPUs, for
various patterns and data graphs. We also show that G2Miner on a V100 GPU is
48.3x and 15.2x faster than the state-of-the-art CPU-based system, Peregrine
and GraphZero, on a 56-core CPU machine.

    

### [[2112.09762] Reproducible and Portable Big Data Analytics in the Cloud](http://arxiv.org/abs/2112.09762)


  Cloud computing has become a major approach to enable reproducible
computational experiments because of its support of on-demand hardware and
software resource provisioning. Yet there are still two main difficulties in
reproducing big data applications in the cloud. The first is how to automate
end-to-end execution of big data analytics in the cloud including virtual
distributed environment provisioning, network and security group setup, and big
data analytics pipeline description and execution. The second is an application
developed for one cloud, such as AWS or Azure, is difficult to reproduce in
another cloud, a.k.a. vendor lock-in problem. To tackle these problems, we
leverage serverless computing and containerization techniques for automatic
scalable big data application execution and reproducibility, and utilize the
adapter design pattern to enable application portability and reproducibility
across different clouds. Based on the approach, we propose and develop an
open-source toolkit that supports 1) on-demand distributed hardware and
software environment provisioning, 2) automatic data and configuration storage
for each execution, 3) flexible client modes based on user preferences, 4)
execution history query, and 5) simple reproducibility of existing executions
in the same environment or a different environment. We did extensive
experiments on both AWS and Azure using three big data analytics applications
that run on a virtual CPU/GPU cluster. Three main behaviors of our toolkit were
benchmarked: i) execution overhead ratio for reproducibility support, ii)
differences of reproducing the same application on AWS and Azure in terms of
execution time, budgetary cost and cost-performance ratio, iii) differences
between scale-out and scale-up approach for the same application on AWS and
Azure.

    

### [[2112.09780] Exploring the Impact of Virtualization on the Usability of the Deep Learning Applications](http://arxiv.org/abs/2112.09780)


  Deep Learning-based (DL) applications are becoming increasingly popular and
advancing at an unprecedented pace. While many research works are being
undertaken to enhance Deep Neural Networks (DNN) -- the centerpiece of DL
applications -- practical deployment challenges of these applications in the
Cloud and Edge systems, and their impact on the usability of the applications
have not been sufficiently investigated. In particular, the impact of deploying
different virtualization platforms, offered by the Cloud and Edge, on the
usability of DL applications (in terms of the End-to-End (E2E) inference time)
has remained an open question. Importantly, resource elasticity (by means of
scale-up), CPU pinning, and processor type (CPU vs GPU) configurations have
shown to be influential on the virtualization overhead. Accordingly, the goal
of this research is to study the impact of these potentially decisive
deployment options on the E2E performance, thus, usability of the DL
applications. To that end, we measure the impact of four popular execution
platforms (namely, bare-metal, virtual machine (VM), container, and container
in VM) on the E2E inference time of four types of DL applications, upon
changing processor configuration (scale-up, CPU pinning) and processor types.
This study reveals a set of interesting and sometimes counter-intuitive
findings that can be used as best practices by Cloud solution architects to
efficiently deploy DL applications in various systems. The notable finding is
that the solution architects must be aware of the DL application
characteristics, particularly, their pre- and post-processing requirements, to
be able to optimally choose and configure an execution platform, determine the
use of GPU, and decide the efficient scale-up range.

    

### [[2112.09974] Serverless data pipeline approaches for IoT data in fog and cloud computing](http://arxiv.org/abs/2112.09974)


  With the increasing number of Internet of Things (IoT) devices, massive
amounts of raw data is being generated. The latency, cost, and other challenges
in cloud-based IoT data processing have driven the adoption of Edge and Fog
computing models, where some data processing tasks are moved closer to data
sources. Properly dealing with the flow of such data requires building data
pipelines, to control the complete life cycle of data streams from data
acquisition at the data source, edge and fog processing, to Cloud side storage
and analytics. Data analytics tasks need to be executed dynamically at
different distances from the data sources and often on very heterogeneous
hardware devices. This can be streamlined by the use of a Serverless (or FaaS)
cloud computing model, where tasks are defined as virtual functions, which can
be migrated from edge to cloud (and vice versa) and executed in an event-driven
manner on data streams. In this work, we investigate the benefits of building
Serverless data pipelines (SDP) for IoT data analytics and evaluate three
different approaches for designing SDPs: 1) Off-the-shelf data flow tool (DFT)
based, 2) Object storage service (OSS) based and 3) MQTT based. Further, we
applied these strategies on three fog applications (Aeneas, PocketSphinx, and
custom Video processing application) and evaluated the performance by comparing
their processing time (computation time, network communication and disk access
time), and resource utilization. Results show that DFT is unsuitable for
compute-intensive applications such as video or image processing, whereas OSS
is best suitable for this task. However, DFT is nicely fit for
bandwidth-intensive applications due to the minimum use of network resources.
On the other hand, MQTT-based SDP is observed with increase in CPU and Memory
usage as the number of...<truncted to fit character limit in Arxiv>

    

### [[2112.10091] A Novel Load Balancing Scheme for Mobile Edge Computing](http://arxiv.org/abs/2112.10091)


  To overcome long propagation delays for data exchange between the remote
cloud data center and end devices in Mobile Cloud Computing (MCC), Mobile Edge
Computing (MEC) is merging to push mobile computing, network control and
storage to the network edges. A cloudlet in MEC is a mobility-enhanced
small-scale cloud, which contains several MEC servers located in close
proximity to mobile devices. The main purpose of a cloudlet is to stably
provide services to mobile devices with low latency. When a cloudlet offers
hundreds kinds of services to millions of mobile devices, it is critical to
balance the loads so as to improve performance. In this paper, we propose a
three-layer mobile hybrid hierarchical P2P (MHP2P) model as a cloudlet. MHP2P
performs satisfactory service lookup efficiency and system scalability as well
as high stability. More importantly, a load balance scheme is provided so that
the loads of MHP2P can be well balanced with the increasing of MEC servers and
query load. A large number of simulation experiments indicate that the proposed
scheme is efficient for enhancing the load balancing performance in MHP2P based
MEC systems.

    

### [[2112.10127] Proactive Autoscaling for Edge Computing Systems with Kubernetes](http://arxiv.org/abs/2112.10127)


  With the emergence of the Internet of Things and 5G technologies, the edge
computing paradigm is playing increasingly important roles with better
availability, latency-control and performance. However, existing autoscaling
tools for edge computing applications do not utilize heterogeneous resources of
edge systems efficiently, leaving scope for performance improvement. In this
work, we propose a Proactive Pod Autoscaler (PPA) for edge computing
applications on Kubernetes. The proposed PPA is able to forecast workloads in
advance with multiple user-defined/customized metrics and to scale edge
computing applications up and down correspondingly. The PPA is optimized and
evaluated on an example CPU-intensive edge computing application further. It
can be concluded that the proposed PPA outperforms the default pod autoscaler
of Kubernetes on both efficiency of resource utilization and application
performance. The article also highlights future possible improvements on the
proposed PPA.

    

### [[2112.10134] An Experimental and Comparative Benchmark Study Examining Resource Utilization in Managed Hadoop Context](http://arxiv.org/abs/2112.10134)


  Transitioning cloud-based Hadoop from IaaS to PaaS, which are commercially
conceptualized as pay-as-you-go or pay-per-use, often reduces the associated
system costs. However, managed Hadoop systems do present a black-box behavior
to the end-users who cannot be clear on the inner performance dynamics, hence,
on the benefits of leveraging them. In the study, we aimed to understand
managed Hadoop context in terms of resource utilization. We utilized three
experimental Hadoop-on-PaaS proposals as they come out-of-the-box and conducted
Hadoop specific workloads of the HiBench Benchmark Suite. During the benchmark
executions, we collected system resource utilization data on the worker nodes.
The results indicated that the same property specifications among cloud
services do not guarantee nearby performance outputs, nor consistent results
within themselves. We assume that the managed systems' architectures and
pre-configurations play a significant role in the performance.

    

### [[2112.10223] Parallel Algorithms for Adding a Collection of Sparse Matrices](http://arxiv.org/abs/2112.10223)


  We develop a family of parallel algorithms for the SpKAdd operation that adds
a collection of k sparse matrices. SpKAdd is a much needed operation in many
applications including distributed memory sparse matrix-matrix multiplication
(SpGEMM), streaming accumulations of graphs, and algorithmic sparsification of
the gradient updates in deep learning. While adding two sparse matrices is a
common operation in Matlab, Python, Intel MKL, and various GraphBLAS libraries,
these implementations do not perform well when adding a large collection of
sparse matrices. We develop a series of algorithms using tree merging, heap,
sparse accumulator, hash table, and sliding hash table data structures. Among
them, hash-based algorithms attain the theoretical lower bounds both on the
computational and I/O complexities and perform the best in practice. The
newly-developed hash SpKAdd makes the computation of a distributed-memory
SpGEMM algorithm at least 2x faster than that the previous state-of-the-art
algorithms.

    

### [[2112.10364] NavP: Enabling Navigational Programming for Science Data Processing via Application-Initiated Checkpointing](http://arxiv.org/abs/2112.10364)


  Science Data Systems (SDS) handle science data from acquisition through
processing to distribution. They are deployed in the Cloud today, and the
efficiency of Cloud instance utilization is critical to success. Conventional
SDS are unable to take advantage of a cost-effective Amazon EC2 spot market,
especially for long-running tasks. Some of the difficulties found in current
practice at NASA/JPL are: a lack of mechanism for app programmers to save
valuable partial results for future processing continuation, the heavy weight
from using container-based (Singularity) sandboxes with more than 200,000
OS-level files; and the gap between scientists developing algorithms/programs
on a laptop and the SDS experts deploying software in Cloud computing or
supercomputing.
We present a first proof-of-principle of this using NavP (Navigational
Programming) and fault-tolerant computing (FTC) in SDS, by employing program
state migration facilitated by Checkpoint-Restart (C/R). NavP provides a new
navigational view of computations in a distributed world for the application
programmers. The tool of DHP (DMTCP Hop and Publish) we developed enables the
application programmers to navigate the computation among instances or nodes by
inserting hop(destination) statements in their app code, and choose when to
publish partial results at stages of their algorithms that they think
worthwhile for future continuation. The result of using DHP is that a parallel
distributed SDS becomes easier to program and deploy, and this enables more
efficient leveraging of the Amazon EC2 Spot market. This technical report
describes a high-level design and an initial implementation.

    

### [[2010.02147] Diversity/Parallelism Trade-off in Distributed Systems with Redundancy](http://arxiv.org/abs/2010.02147)


  As numerous machine learning and other algorithms increase in complexity and
data requirements, distributed computing becomes necessary to satisfy the
growing computational and storage demands, because it enables parallel
execution of smaller tasks that make up a large computing job. However, random
fluctuations in task service times lead to straggling tasks with long execution
times. Redundancy, in the form of task replication and erasure coding, provides
diversity that allows a job to be completed when only a subset of redundant
tasks is executed, thus removing the dependency on the straggling tasks. In
situations of constrained resources (here a fixed number of parallel servers),
increasing redundancy reduces the available resources for parallelism. In this
paper, we characterize the diversity vs. parallelism trade-off and identify the
optimal strategy, among replication, coding and splitting, which minimizes the
expected job completion time. We consider three common service time
distributions and establish three models that describe scaling of these
distributions with the task size. We find that different distributions with
different scaling models operate optimally at different levels of redundancy,
and thus may require very different code rates.

    

### [[2103.10280] Computing Parameterized Invariants of Parameterized Petri Nets](http://arxiv.org/abs/2103.10280)


  A fundamental advantage of Petri net models is the possibility to
automatically compute useful system invariants from the syntax of the net.
Classical techniques used for this are place invariants, P-components, siphons
or traps. Recently, Bozga et al. have presented a novel technique for the
\emph{parameterized} verification of safety properties of systems with a ring
or array architecture. They show that the statement \enquote{for every instance
of the parameterized Petri net, all markings satisfying the linear invariants
associated to all the P-components, siphons and traps of the instance are safe}
can be encoded in \acs{WS1S} and checked using tools like MONA. However, while
the technique certifies that this infinite set of linear invariants extracted
from P-components, siphons or traps are strong enough to prove safety, it does
not return an explanation of this fact understandable by humans. We present a
CEGAR loop that constructs a \emph{finite} set of \emph{parameterized}
P-components, siphons or traps, whose infinitely many instances are strong
enough to prove safety. For this we design parameterization procedures for
different architectures.

    

### [[2105.04880] Consistent Multiple Graph Embedding for Multi-View Clustering](http://arxiv.org/abs/2105.04880)


  Graph-based multi-view clustering aiming to obtain a partition of data across
multiple views, has received considerable attention in recent years. Although
great efforts have been made for graph-based multi-view clustering, it remains
a challenge to fuse characteristics from various views to learn a common
representation for clustering. In this paper, we propose a novel Consistent
Multiple Graph Embedding Clustering framework(CMGEC). Specifically, a multiple
graph auto-encoder(M-GAE) is designed to flexibly encode the complementary
information of multi-view data using a multi-graph attention fusion encoder. To
guide the learned common representation maintaining the similarity of the
neighboring characteristics in each view, a Multi-view Mutual Information
Maximization module(MMIM) is introduced. Furthermore, a graph fusion
network(GFN) is devised to explore the relationship among graphs from different
views and provide a common consensus graph needed in M-GAE. By jointly training
these models, the common latent representation can be obtained which encodes
more complementary information from multiple views and depicts data more
comprehensively. Experiments on three types of multi-view datasets demonstrate
CMGEC outperforms the state-of-the-art clustering methods.

    

### [[2112.09737] Improving scripts with a memory of natural feedback](http://arxiv.org/abs/2112.09737)


  How can an end-user provide feedback if a deployed structured prediction
model generates incorrect output? Our goal is to allow users to correct errors
directly through interaction, without retraining, by giving feedback on the
model's output. We create a dynamic memory architecture with a growing memory
of feedbacks about errors in the output. Given a new, unseen input, our model
can use feedback from a similar, past erroneous state. On a script generation
task, we show empirically that the model learns to apply feedback effectively
(up to 30 points improvement), while avoiding similar past mistakes after
deployment (up to 10 points improvement on an unseen set). This is a first step
towards strengthening deployed models, potentially broadening their utility.

    

### [[2112.09791] Query Adaptive Few-Shot Object Detection with Heterogeneous Graph Convolutional Networks](http://arxiv.org/abs/2112.09791)


  Few-shot object detection (FSOD) aims to detect never-seen objects using few
examples. This field sees recent improvement owing to the meta-learning
techniques by learning how to match between the query image and few-shot class
examples, such that the learned model can generalize to few-shot novel classes.
However, currently, most of the meta-learning-based methods perform pairwise
matching between query image regions (usually proposals) and novel classes
separately, therefore failing to take into account multiple relationships among
them. In this paper, we propose a novel FSOD model using heterogeneous graph
convolutional networks. Through efficient message passing among all the
proposal and class nodes with three different types of edges, we could obtain
context-aware proposal features and query-adaptive, multiclass-enhanced
prototype representations for each class, which could help promote the pairwise
matching and improve final FSOD accuracy. Extensive experimental results show
that our proposed model, denoted as QA-FewDet, outperforms the current
state-of-the-art approaches on the PASCAL VOC and MSCOCO FSOD benchmarks under
different shots and evaluation metrics.

    

### [[2112.09831] Neural Born Iteration Method For Solving Inverse Scattering Problems: 2D Cases](http://arxiv.org/abs/2112.09831)


  In this paper, we propose the neural Born iteration method (NeuralBIM) for
solving 2D inverse scattering problems (ISPs) by drawing on the scheme of
physics-informed supervised residual learning (PhiSRL) to emulate the computing
process of the traditional Born iteration method (TBIM). NeuralBIM employs
independent convolutional neural networks (CNNs) to learn the alternate update
rules of two different candidate solutions with their corresponding residuals.
Two different schemes of NeuralBIMs are presented in this paper including
supervised and unsupervised learning schemes. With the data set generated by
method of moments (MoM), supervised NeuralBIMs are trained with the knowledge
of total fields and contrasts. Unsupervised NeuralBIM is guided by the
physics-embedded loss functions founding on the governing equations of ISPs,
which results in no requirements of total fields and contrasts for training.
Representative numerical results further validate the effectiveness and
competitiveness of both supervised and unsupervised NeuralBIMs.

    

### [[2112.09833] Face Deblurring Based on Separable Normalization and Adaptive Denormalization](http://arxiv.org/abs/2112.09833)


  Face deblurring aims to restore a clear face image from a blurred input image
with more explicit structure and facial details. However, most conventional
image and face deblurring methods focus on the whole generated image resolution
without consideration of special face part texture and generally produce
unsufficient details. Considering that faces and backgrounds have different
distribution information, in this study, we designed an effective face
deblurring network based on separable normalization and adaptive
denormalization (SNADNet). First, We fine-tuned the face parsing network to
obtain an accurate face structure. Then, we divided the face parsing feature
into face foreground and background. Moreover, we constructed a new feature
adaptive denormalization to regularize fafcial structures as a condition of the
auxiliary to generate more harmonious and undistorted face structure. In
addition, we proposed a texture extractor and multi-patch discriminator to
enhance the generated facial texture information. Experimental results on both
CelebA and CelebA-HQ datasets demonstrate that the proposed face deblurring
network restores face structure with more facial details and performs favorably
against state-of-the-art methods in terms of structured similarity indexing
method (SSIM), peak signal-to-noise ratio (PSNR), Frechet inception distance
(FID) and L1, and qualitative comparisons.

    

### [[2112.09839] Calorie Aware Automatic Meal Kit Generation from an Image](http://arxiv.org/abs/2112.09839)


  Calorie and nutrition research has attained increased interest in recent
years. But, due to the complexity of the problem, literature in this area
focuses on a limited subset of ingredients or dish types and simple
convolutional neural networks or traditional machine learning. Simultaneously,
estimation of ingredient portions can help improve calorie estimation and meal
re-production from a given image. In this paper, given a single cooking image,
a pipeline for calorie estimation and meal re-production for different servings
of the meal is proposed. The pipeline contains two stages. In the first stage,
a set of ingredients associated with the meal in the given image are predicted.
In the second stage, given image features and ingredients, portions of the
ingredients and finally the total meal calorie are simultaneously estimated
using a deep transformer-based model. Portion estimation introduced in the
model helps improve calorie estimation and is also beneficial for meal
re-production in different serving sizes. To demonstrate the benefits of the
pipeline, the model can be used for meal kits generation. To evaluate the
pipeline, the large scale dataset Recipe1M is used. Prior to experiments, the
Recipe1M dataset is parsed and explicitly annotated with portions of
ingredients. Experiments show that using ingredients and their portions
significantly improves calorie estimation. Also, a visual interface is created
in which a user can interact with the pipeline to reach accurate calorie
estimations and generate a meal kit for cooking purposes.

    

### [[2112.09844] Enhanced Object Detection in Floor-plan through Super Resolution](http://arxiv.org/abs/2112.09844)


  Building Information Modelling (BIM) software use scalable vector formats to
enable flexible designing of floor plans in the industry. Floor plans in the
architectural domain can come from many sources that may or may not be in
scalable vector format. The conversion of floor plan images to fully annotated
vector images is a process that can now be realized by computer vision. Novel
datasets in this field have been used to train Convolutional Neural Network
(CNN) architectures for object detection. Image enhancement through
Super-Resolution (SR) is also an established CNN based network in computer
vision that is used for converting low resolution images to high resolution
ones. This work focuses on creating a multi-component module that stacks a SR
model on a floor plan object detection model. The proposed stacked model shows
greater performance than the corresponding vanilla object detection model. For
the best case, the the inclusion of SR showed an improvement of 39.47% in
object detection over the vanilla network. Data and code are made publicly
available at this https URL.

    

### [[2112.09845] Time-Aware Neighbor Sampling for Temporal Graph Networks](http://arxiv.org/abs/2112.09845)


  We present a new neighbor sampling method on temporal graphs. In a temporal
graph, predicting different nodes' time-varying properties can require the
receptive neighborhood of various temporal scales. In this work, we propose the
TNS (Time-aware Neighbor Sampling) method: TNS learns from temporal information
to provide an adaptive receptive neighborhood for every node at any time.
Learning how to sample neighbors is non-trivial, since the neighbor indices in
time order are discrete and not differentiable. To address this challenge, we
transform neighbor indices from discrete values to continuous ones by
interpolating the neighbors' messages. TNS can be flexibly incorporated into
popular temporal graph networks to improve their effectiveness without
increasing their time complexity. TNS can be trained in an end-to-end manner.
It needs no extra supervision and is automatically and implicitly guided to
sample the neighbors that are most beneficial for prediction. Empirical results
on multiple standard datasets show that TNS yields significant gains on edge
prediction and node classification.

    

### [[2112.09939] Syntactic-GCN Bert based Chinese Event Extraction](http://arxiv.org/abs/2112.09939)


  With the rapid development of information technology, online platforms (e.g.,
news portals and social media) generate enormous web information every moment.
Therefore, it is crucial to extract structured representations of events from
social streams. Generally, existing event extraction research utilizes pattern
matching, machine learning, or deep learning methods to perform event
extraction tasks. However, the performance of Chinese event extraction is not
as good as English due to the unique characteristics of the Chinese language.
In this paper, we propose an integrated framework to perform Chinese event
extraction. The proposed approach is a multiple channel input neural framework
that integrates semantic features and syntactic features. The semantic features
are captured by BERT architecture. The Part of Speech (POS) features and
Dependency Parsing (DP) features are captured by profiling embeddings and Graph
Convolutional Network (GCN), respectively. We also evaluate our model on a
real-world dataset. Experimental results show that the proposed method
outperforms the benchmark approaches significantly.

    

### [[2112.09996] Curriculum Based Reinforcement Learning of Grid Topology Controllers to Prevent Thermal Cascading](http://arxiv.org/abs/2112.09996)


  This paper describes how domain knowledge of power system operators can be
integrated into reinforcement learning (RL) frameworks to effectively learn
agents that control the grid's topology to prevent thermal cascading. Typical
RL-based topology controllers fail to perform well due to the large
search/optimization space. Here, we propose an actor-critic-based agent to
address the problem's combinatorial nature and train the agent using the RL
environment developed by RTE, the French TSO. To address the challenge of the
large optimization space, a curriculum-based approach with reward tuning is
incorporated into the training procedure by modifying the environment using
network physics for enhanced agent learning. Further, a parallel training
approach on multiple scenarios is employed to avoid biasing the agent to a few
scenarios and make it robust to the natural variability in grid operations.
Without these modifications to the training procedure, the RL agent failed for
most test scenarios, illustrating the importance of properly integrating domain
knowledge of physical systems for real-world RL learning. The agent was tested
by RTE for the 2019 learning to run the power network challenge and was awarded
the 2nd place in accuracy and 1st place in speed. The developed code is
open-sourced for public use.

    

### [[2112.10007] Online Grounding of PDDL Domains by Acting and Sensing in Unknown Environments](http://arxiv.org/abs/2112.10007)


  To effectively use an abstract (PDDL) planning domain to achieve goals in an
unknown environment, an agent must instantiate such a domain with the objects
of the environment and their properties. If the agent has an egocentric and
partial view of the environment, it needs to act, sense, and abstract the
perceived data in the planning domain. Furthermore, the agent needs to compile
the plans computed by a symbolic planner into low level actions executable by
its actuators. This paper proposes a framework that aims to accomplish the
aforementioned perspective and allows an agent to perform different tasks. For
this purpose, we integrate machine learning models to abstract the sensory
data, symbolic planning for goal achievement and path planning for navigation.
We evaluate the proposed method in accurate simulated environments, where the
sensors are RGB-D on-board camera, GPS and compass.

    

### [[2112.10024] Supervised laser-speckle image sampling of skin tissue to detect very early stage of diabetes by its effects on skin subcellular properties](http://arxiv.org/abs/2112.10024)


  This paper investigates the effectiveness of an expert system based on
K-nearest neighbors algorithm for laser speckle image sampling applied to the
early detection of diabetes. With the latest developments in artificial
intelligent guided laser speckle imaging technologies, it may be possible to
optimise laser parameters, such as wavelength, energy level and image texture
measures in association with a suitable AI technique to interact effectively
with the subcellular properties of a skin tissue to detect early signs of
diabetes. The new approach is potentially more effective than the classical
skin glucose level observation because of its optimised combination of laser
physics and AI techniques, and additionally, it allows non-expert individuals
to perform more frequent skin tissue tests for an early detection of diabetes.

    

### [[2112.10047] Controlling the Quality of Distillation in Response-Based Network Compression](http://arxiv.org/abs/2112.10047)


  The performance of a distillation-based compressed network is governed by the
quality of distillation. The reason for the suboptimal distillation of a large
network (teacher) to a smaller network (student) is largely attributed to the
gap in the learning capacities of given teacher-student pair. While it is hard
to distill all the knowledge of a teacher, the quality of distillation can be
controlled to a large extent to achieve better performance. Our experiments
show that the quality of distillation is largely governed by the quality of
teacher's response, which in turn is heavily affected by the presence of
similarity information in its response. A well-trained large capacity teacher
loses similarity information between classes in the process of learning
fine-grained discriminative properties for classification. The absence of
similarity information causes the distillation process to be reduced from one
example-many class learning to one example-one class learning, thereby
throttling the flow of diverse knowledge from the teacher. With the implicit
assumption that only the instilled knowledge can be distilled, instead of
focusing only on the knowledge distilling process, we scrutinize the knowledge
inculcation process. We argue that for a given teacher-student pair, the
quality of distillation can be improved by finding the sweet spot between batch
size and number of epochs while training the teacher. We discuss the steps to
find this sweet spot for better distillation. We also propose the distillation
hypothesis to differentiate the behavior of the distillation process between
knowledge distillation and regularization effect. We conduct all our
experiments on three different datasets.

    

### [[2112.10085] D-HAN: Dynamic News Recommendation with Hierarchical Attention Network](http://arxiv.org/abs/2112.10085)


  News recommendation is an effective information dissemination solution in
modern society. While recent years have witnessed many promising news
recommendation models, they mostly capture the user-news interactions on the
document-level in a static manner. However, in real-world scenarios, the news
can be quite complex and diverse, blindly squeezing all the contents into an
embedding vector can be less effective in extracting information compatible
with the personalized preference of the users. In addition, user preferences in
the news recommendation scenario can be highly dynamic, and a tailored dynamic
mechanism should be designed for better recommendation performance. In this
paper, we propose a novel dynamic news recommender model. For better
understanding the news content, we leverage the attention mechanism to
represent the news from the sentence-, element- and document-levels,
respectively. For capturing users' dynamic preferences, the continuous time
information is seamlessly incorporated into the computing of the attention
weights. More specifically, we design a hierarchical attention network, where
the lower layer learns the importance of different sentences and elements, and
the upper layer captures the correlations between the previously interacted and
the target news. To comprehensively model the dynamic characters, we firstly
enhance the traditional attention mechanism by incorporating both absolute and
relative time information, and then we propose a dynamic negative sampling
method to optimize the users' implicit feedback. We conduct extensive
experiments based on three real-world datasets to demonstrate our model's
effectiveness. Our source code and pre-trained representations are available at
this https URL.

    

### [[2112.10098] Initiative Defense against Facial Manipulation](http://arxiv.org/abs/2112.10098)


  Benefiting from the development of generative adversarial networks (GAN),
facial manipulation has achieved significant progress in both academia and
industry recently. It inspires an increasing number of entertainment
applications but also incurs severe threats to individual privacy and even
political security meanwhile. To mitigate such risks, many countermeasures have
been proposed. However, the great majority methods are designed in a passive
manner, which is to detect whether the facial images or videos are tampered
after their wide propagation. These detection-based methods have a fatal
limitation, that is, they only work for ex-post forensics but can not prevent
the engendering of malicious behavior. To address the limitation, in this
paper, we propose a novel framework of initiative defense to degrade the
performance of facial manipulation models controlled by malicious users. The
basic idea is to actively inject imperceptible venom into target facial data
before manipulation. To this end, we first imitate the target manipulation
model with a surrogate model, and then devise a poison perturbation generator
to obtain the desired venom. An alternating training strategy are further
leveraged to train both the surrogate model and the perturbation generator. Two
typical facial manipulation tasks: face attribute editing and face reenactment,
are considered in our initiative defense framework. Extensive experiments
demonstrate the effectiveness and robustness of our framework in different
settings. Finally, we hope this work can shed some light on initiative
countermeasures against more adversarial scenarios.

    

### [[2112.10125] Masked Deep Q-Recommender for Effective Question Scheduling](http://arxiv.org/abs/2112.10125)


  Providing appropriate questions according to a student's knowledge level is
imperative in personalized learning. However, It requires a lot of manual
effort for teachers to understand students' knowledge status and provide
optimal questions accordingly. To address this problem, we introduce a question
scheduling model that can effectively boost student knowledge level using
Reinforcement Learning (RL). Our proposed method first evaluates students'
concept-level knowledge using knowledge tracing (KT) model. Given predicted
student knowledge, RL-based recommender predicts the benefits of each question.
With curriculum range restriction and duplicate penalty, the recommender
selects questions sequentially until it reaches the predefined number of
questions. In an experimental setting using a student simulator, which gives 20
questions per day for two weeks, questions recommended by the proposed method
increased average student knowledge level by 21.3%, superior to an
expert-designed schedule baseline with a 10% increase in student knowledge
levels.

    

### [[2112.10190] Demanding and Designing Aligned Cognitive Architectures](http://arxiv.org/abs/2112.10190)


  With AI systems becoming more powerful and pervasive, there is increasing
debate about keeping their actions aligned with the broader goals and needs of
humanity. This multi-disciplinary and multi-stakeholder debate must resolve
many issues, here we examine three of them. The first issue is to clarify what
demands stakeholders might usefully make on the designers of AI systems, useful
because the technology exists to implement them. We make this technical topic
more accessible by using the framing of cognitive architectures. The second
issue is to move beyond an analytical framing that treats useful intelligence
as being reward maximization only. To support this move, we define several AI
cognitive architectures that combine reward maximization with other technical
elements designed to improve alignment. The third issue is how stakeholders
should calibrate their interactions with modern machine learning researchers.
We consider how current fashions in machine learning create a narrative pull
that participants in technical and policy discussions should be aware of, so
that they can compensate for it. We identify several technically tractable but
currently unfashionable options for improving AI alignment.

    

### [[2112.10275] Parallel Multi-Scale Networks with Deep Supervision for Hand Keypoint Detection](http://arxiv.org/abs/2112.10275)


  Keypoint detection plays an important role in a wide range of applications.
However, predicting keypoints of small objects such as human hands is a
challenging problem. Recent works fuse feature maps of deep Convolutional
Neural Networks (CNNs), either via multi-level feature integration or
multi-resolution aggregation. Despite achieving some success, the feature
fusion approaches increase the complexity and the opacity of CNNs. To address
this issue, we propose a novel CNN model named Multi-Scale Deep Supervision
Network (P-MSDSNet) that learns feature maps at different scales with deep
supervisions to produce attention maps for adaptive feature propagation from
layers to layers. P-MSDSNet has a multi-stage architecture which makes it
scalable while its deep supervision with spatial attention improves
transparency to the feature learning at each stage. We show that P-MSDSNet
outperforms the state-of-the-art approaches on benchmark datasets while
requiring fewer number of parameters. We also show the application of P-MSDSNet
to quantify finger tapping hand movements in a neuroscience study.

    

### [[2112.10316] CSSR: A Context-Aware Sequential Software Service Recommendation Model](http://arxiv.org/abs/2112.10316)


  We propose a novel software service recommendation model to help users find
their suitable repositories in GitHub. Our model first designs a novel
context-induced repository graph embedding method to leverage rich contextual
information of repositories to alleviate the difficulties caused by the data
sparsity issue. It then leverages sequence information of user-repository
interactions for the first time in the software service recommendation field.
Specifically, a deep-learning based sequential recommendation technique is
adopted to capture the dynamics of user preferences. Comprehensive experiments
have been conducted on a large dataset collected from GitHub against a list of
existing methods. The results illustrate the superiority of our method in
various aspects.

    

### [[2112.10374] Multi-agent Communication with Graph Information Bottleneck under Limited Bandwidth (a position paper)](http://arxiv.org/abs/2112.10374)


  Recent studies have shown that introducing communication between agents can
significantly improve overall performance in cooperative Multi-agent
reinforcement learning (MARL). In many real-world scenarios, communication can
be expensive and the bandwidth of the multi-agent system is subject to certain
constraints. Redundant messages who occupy the communication resources can
block the transmission of informative messages and thus jeopardize the
performance. In this paper, we aim to learn the minimal sufficient
communication messages. First, we initiate the communication between agents by
a complete graph. Then we introduce the graph information bottleneck (GIB)
principle into this complete graph and derive the optimization over graph
structures. Based on the optimization, a novel multi-agent communication
module, called CommGIB, is proposed, which effectively compresses the structure
information and node information in the communication graph to deal with
bandwidth-constrained settings. Extensive experiments in Traffic Control and
StanCraft II are conducted. The results indicate that the proposed methods can
achieve better performance in bandwidth-restricted settings compared with
state-of-the-art algorithms, with especially large margins in large-scale
multi-agent tasks.

    

### [[2112.10390] Evaluation and Comparison of Deep Learning Methods for Pavement Crack Identification with Visual Images](http://arxiv.org/abs/2112.10390)


  Compared with contact detection techniques, pavement crack identification
with visual images via deep learning algorithms has the advantages of not being
limited by the material of object to be detected, fast speed and low cost. The
fundamental frameworks and typical model architectures of transfer learning
(TL), encoder-decoder (ED), generative adversarial networks (GAN), and their
common modules were first reviewed, and then the evolution of convolutional
neural network (CNN) backbone models and GAN models were summarized. The crack
classification, segmentation performance, and effect were tested on the
SDNET2018 and CFD public data sets. In the aspect of patch sample
classification, the fine-tuned TL models can be equivalent to or even slightly
better than the ED models in accuracy, and the predicting time is faster; In
the aspect of accurate crack location, both ED and GAN algorithms can achieve
pixel-level segmentation and is expected to be detected in real time on low
computing power platform. Furthermore, a weakly supervised learning framework
of combined TL-SSGAN and its performance enhancement measures are proposed,
which can maintain comparable crack identification performance with that of the
supervised learning, while greatly reducing the number of labeled samples
required.

    

### [[2112.10424] Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction](http://arxiv.org/abs/2112.10424)


  Recent works have shown explainability and robustness are two crucial
ingredients of trustworthy and reliable text classification. However, previous
works usually address one of two aspects: i) how to extract accurate rationales
for explainability while being beneficial to prediction; ii) how to make the
predictive model robust to different types of adversarial attacks. Intuitively,
a model that produces helpful explanations should be more robust against
adversarial attacks, because we cannot trust the model that outputs
explanations but changes its prediction under small perturbations. To this end,
we propose a joint classification and rationale extraction model named AT-BMC.
It includes two key mechanisms: mixed Adversarial Training (AT) is designed to
use various perturbations in discrete and embedding space to improve the
model's robustness, and Boundary Match Constraint (BMC) helps to locate
rationales more precisely with the guidance of boundary information.
Performances on benchmark datasets demonstrate that the proposed AT-BMC
outperforms baselines on both classification and rationale extraction by a
large margin. Robustness analysis shows that the proposed AT-BMC decreases the
attack success rate effectively by up to 69%. The empirical results indicate
that there are connections between robust models and better explanations.

    

### [[2112.10433] Diaformer: Automatic Diagnosis via Symptoms Sequence Generation](http://arxiv.org/abs/2112.10433)


  Automatic diagnosis has attracted increasing attention but remains
challenging due to multi-step reasoning. Recent works usually address it by
reinforcement learning methods. However, these methods show low efficiency and
require taskspecific reward functions. Considering the conversation between
doctor and patient allows doctors to probe for symptoms and make diagnoses, the
diagnosis process can be naturally seen as the generation of a sequence
including symptoms and diagnoses. Inspired by this, we reformulate automatic
diagnosis as a symptoms Sequence Generation (SG) task and propose a simple but
effective automatic Diagnosis model based on Transformer (Diaformer). We
firstly design the symptom attention framework to learn the generation of
symptom inquiry and the disease diagnosis. To alleviate the discrepancy between
sequential generation and disorder of implicit symptoms, we further design
three orderless training mechanisms. Experiments on three public datasets show
that our model outperforms baselines on disease diagnosis by 1%, 6% and 11.5%
with the highest training efficiency. Detailed analysis on symptom inquiry
prediction demonstrates that the potential of applying symptoms sequence
generation for automatic diagnosis.

    

### [[2112.10531] Object Recognition as Classification of Visual Properties](http://arxiv.org/abs/2112.10531)


  We base our work on the teleosemantic modelling of concepts as abilities
implementing the distinct functions of recognition and classification.
Accordingly, we model two types of concepts - substance concepts suited for
object recognition exploiting visual properties, and classification concepts
suited for classification of substance concepts exploiting linguistically
grounded properties. The goal in this paper is to demonstrate that object
recognition can be construed as classification of visual properties, as
distinct from work in mainstream computer vision. Towards that, we present an
object recognition process based on Ranganathan's four-phased faceted knowledge
organization process, grounded in the teleosemantic distinctions of substance
concept and classification concept. We also briefly introduce the ongoing
project MultiMedia UKC, whose aim is to build an object recognition resource
following our proposed process

    

### [[2112.10543] Spiral Language Modeling](http://arxiv.org/abs/2112.10543)


  In almost all text generation applications, word sequences are constructed in
a left-to-right (L2R) or right-to-left (R2L) manner, as natural language
sentences are written either L2R or R2L. However, we find that the natural
language written order is not essential for text generation. In this paper, we
propose Spiral Language Modeling (SLM), a general approach that enables one to
construct natural language sentences beyond the L2R and R2L order. SLM allows
one to form natural language text by starting from an arbitrary token inside
the result text and expanding the rest tokens around the selected ones. It
makes the decoding order a new optimization objective besides the language
model perplexity, which further improves the diversity and quality of the
generated text. Furthermore, SLM makes it possible to manipulate the text
construction process by selecting a proper starting token. SLM also introduces
generation orderings as additional regularization to improve model robustness
in low-resource scenarios. Experiments on 8 widely studied Neural Machine
Translation (NMT) tasks show that SLM is constantly effective with up to 4.7
BLEU increase comparing to the conventional L2R decoding approach.

    

### [[2112.10613] Intelligent Online Selling Point Extraction for E-Commerce Recommendation](http://arxiv.org/abs/2112.10613)


  In the past decade, automatic product description generation for e-commerce
have witnessed significant advancement. As the services provided by e-commerce
platforms become diverse, it is necessary to dynamically adapt the patterns of
descriptions generated. The selling point of products is an important type of
product description for which the length should be as short as possible while
still conveying key information. In addition, this kind of product description
should be eye-catching to the readers. Currently, product selling points are
normally written by human experts. Thus, the creation and maintenance of these
contents incur high costs. These costs can be significantly reduced if product
selling points can be automatically generated by machines. In this paper, we
report our experience developing and deploying the Intelligent Online Selling
Point Extraction (IOSPE) system to serve the recommendation system in the
this http URL e-commerce platform. Since July 2020, IOSPE has become a core service
for 62 key categories of products (covering more than 4 million products). So
far, it has generated more than 0.1 billion selling points, thereby
significantly scaling up the selling point creation operation and saving human
labour. These IOSPE generated selling points have increased the click-through
rate (CTR) by 1.89\% and the average duration the customers spent on the
products by more than 2.03\% compared to the previous practice, which are
significant improvements for such a large-scale e-commerce platform.

    

### [[1909.00885] Simplified decision making in the belief space using belief sparsification](http://arxiv.org/abs/1909.00885)


  In this work, we introduce a new and efficient solution approach for the
problem of decision making under uncertainty, which can be formulated as
decision making in a belief space, over a possibly high-dimensional state
space. Typically, to solve a decision problem, one should identify the optimal
action from a set of candidates, according to some objective. We claim that one
can often generate and solve an analogous yet simplified decision problem,
which can be solved more efficiently. A wise simplification method can lead to
the same action selection, or one for which the maximal loss in optimality can
be guaranteed. Furthermore, such simplification is separated from the state
inference, and does not compromise its accuracy, as the selected action would
finally be applied on the original state. First, we present the concept for
general decision problems, and provide a theoretical framework for a coherent
formulation of the approach. We then practically apply these ideas to decision
problems in the belief space, which can be simplified by considering a sparse
approximation of their initial belief. The scalable belief sparsification
algorithm we provide is able to yield solutions which are guaranteed to be
consistent with the original problem. We demonstrate the benefits of the
approach in the solution of a realistic active-SLAM problem, and manage to
significantly reduce computation time, with no loss in the quality of solution.
This work is both fundamental and practical, and holds numerous possible
extensions.

    

### [[2009.00326] PYCSP3: Modeling Combinatorial Constrained Problems in Python](http://arxiv.org/abs/2009.00326)


  In this document, we introduce PyCSP$3$, a Python library that allows us to
write models of combinatorial constrained problems in a declarative manner.
Currently, with PyCSP$3$, you can write models of constraint satisfaction and
optimization problems. More specifically, you can build CSP (Constraint
Satisfaction Problem) and COP (Constraint Optimization Problem) models.
Importantly, there is a complete separation between the modeling and solving
phases: you write a model, you compile it (while providing some data) in order
to generate an XCSP$3$ instance (file), and you solve that problem instance by
means of a constraint solver. You can also directly pilot the solving procedure
in PyCSP$3$, possibly conducting an incremental solving strategy. In this
document, you will find all that you need to know about PyCSP$3$, with more
than 50 illustrative models.

    

### [[2012.07499] A learning perspective on the emergence of abstractions: the curious case of phonemes](http://arxiv.org/abs/2012.07499)


  In the present paper we use a range of modeling techniques to investigate
whether an abstract phone could emerge from exposure to speech sounds. In
effect, the study represents an attempt for operationalize a theoretical device
of Usage-based Linguistics of emergence of an abstraction from language use.
Our quest focuses on the simplest of such hypothesized abstractions. We test
two opposing principles regarding the development of language knowledge in
linguistically untrained language users: Memory-Based Learning (MBL) and
Error-Correction Learning (ECL). A process of generalization underlies the
abstractions linguists operate with, and we probed whether MBL and ECL could
give rise to a type of language knowledge that resembles linguistic
abstractions. Each model was presented with a significant amount of
pre-processed speech produced by one speaker. We assessed the consistency or
stability of what these simple models have learned and their ability to give
rise to abstract categories. Both types of models fare differently with regard
to these tests. We show that ECL models can learn abstractions and that at
least part of the phone inventory and grouping into traditional types can be
reliably identified from the input.

    

### [[2012.13387] Adaptive Summaries: A Personalized Concept-based Summarization Approach by Learning from Users' Feedback](http://arxiv.org/abs/2012.13387)


  Exploring the tremendous amount of data efficiently to make a decision,
similar to answering a complicated question, is challenging with many
real-world application scenarios. In this context, automatic summarization has
substantial importance as it will provide the foundation for big data analytic.
Traditional summarization approaches optimize the system to produce a short
static summary that fits all users that do not consider the subjectivity aspect
of summarization, i.e., what is deemed valuable for different users, making
these approaches impractical in real-world use cases. This paper proposes an
interactive concept-based summarization model, called Adaptive Summaries, that
helps users make their desired summary instead of producing a single inflexible
summary. The system learns from users' provided information gradually while
interacting with the system by giving feedback in an iterative loop. Users can
choose either reject or accept action for selecting a concept being included in
the summary with the importance of that concept from users' perspectives and
confidence level of their feedback. The proposed approach can guarantee
interactive speed to keep the user engaged in the process. Furthermore, it
eliminates the need for reference summaries, which is a challenging issue for
summarization tasks. Evaluations show that Adaptive Summaries helps users make
high-quality summaries based on their preferences by maximizing the
user-desired content in the generated summaries.

    

### [[2106.10138] Classical Planning as QBF without Grounding (extended version)](http://arxiv.org/abs/2106.10138)


  Most classical planners use grounding as a preprocessing step, essentially
reducing planning to propositional logic. However, grounding involves
instantiating all action rules with concrete object combinations, and results
in large encodings for SAT/QBF-based planners. This severe cost in memory
becomes a main bottleneck when actions have many parameters, such as in the
Organic Synthesis problems from the IPC 2018 competition. We provide a compact
QBF encoding that is logarithmic in the number of objects and avoids grounding
completely, by using universal quantification for object combinations. We show
that we can solve some of the Organic Synthesis problems, which could not be
handled before by any SAT/QBF based planners due to grounding.

    

### [[1910.14560] A Relational Program Logic with Data Abstraction and Dynamic Framing](http://arxiv.org/abs/1910.14560)


  In a paper published in 1972 Hoare articulated the fundamental notions of
hiding invariants and simulations. Hiding: invariants on encapsulated data
representations need not be mentioned in specifications that comprise the API
of a module. Simulation: correctness of a new data representation and
implementation can be established by proving simulation between the old and new
implementations using a coupling relation defined on the encapsulated state.
These results were formalized semantically and for a simple model of state,
though the paper claimed this could be extended to encompass dynamically
allocated objects. In recent years, progress has been made towards formalizing
the claim, for simulation, though mainly in semantic developments. In this
paper, the ideas are combined with the idea in Hoare's 1969 paper: a logic of
programs. For a language with dynamically allocated shared mutable objects, we
introduce a relational Hoare logic that formalizes encapsulation, hiding of
invariants, and relating two implementations via coupling relations. Relations
and other assertions are expressed in first order logic. Specifications can
express a wide range of relational properties such as conditional equivalence
and noninterference with declassification. The proof rules facilitate reasoning
by means of convenient alignments and are shown sound with respect to a
conventional operational semantics. Applicability to representative examples of
data abstraction is demonstrated using an SMT-based implementation.

    

### [[2001.10490] Beyond Notations: Hygienic Macro Expansion for Theorem Proving Languages](http://arxiv.org/abs/2001.10490)


  In interactive theorem provers (ITPs), extensible syntax is not only crucial
to lower the cognitive burden of manipulating complex mathematical objects, but
plays a critical role in developing reusable abstractions in libraries. Most
ITPs support such extensions in the form of restrictive "syntax sugar"
substitutions and other ad hoc mechanisms, which are too rudimentary to support
many desirable abstractions. As a result, libraries are littered with
unnecessary redundancy. Tactic languages in these systems are plagued by a
seemingly unrelated issue: accidental name capture, which often produces
unexpected and counterintuitive behavior. We take ideas from the Scheme family
of programming languages and solve these two problems simultaneously by
proposing a novel hygienic macro system custom-built for ITPs. We further
describe how our approach can be extended to cover type-directed macro
expansion resulting in a single, uniform system offering multiple abstraction
levels that range from supporting simplest syntax sugars to elaboration of
formerly baked-in syntax. We have implemented our new macro system and
integrated it into the new version of the Lean theorem prover, Lean 4. Despite
its expressivity, the macro system is simple enough that it can easily be
integrated into other systems.

    

### [[2102.09920] Overcoming Restraint: Modular Refinement using Cogent's Principled Foreign Function Interface](http://arxiv.org/abs/2102.09920)


  Cogent is a restricted functional language designed to reduce the cost of
developing verified systems code. Because of its sometimes-onerous
restrictions, such as the lack of support for recursion and its strict
uniqueness type system, Cogent provides an escape hatch in the form of a
foreign function interface (FFI) to C code. This poses a problem when verifying
Cogent programs, as imported C components do not enjoy the same level of static
guarantees that Cogent does. Previous verification of file systems implemented
in Cogent merely assumed that their C components were correct and that they
preserved the invariants of Cogent's type system. In this paper, we instead
prove such obligations. We demonstrate how they smoothly compose with existing
Cogent theorems, and result in a correctness theorem of the overall Cogent-C
system. The Cogent FFI constraints ensure that key invariants of Cogent's type
system are maintained even when calling C code. We verify reusable higher-order
and polymorphic functions including a generic loop combinator and array
iterators and demonstrate their application to several examples including
binary search and the BilbyFs file system. We demonstrate the feasibility of
verification of mixed Cogent-C systems, and provide some insight into
verification of software comprised of code in multiple languages with differing
levels of static guarantees.

    

### [[2109.02458] On non-structural subtype entailment](http://arxiv.org/abs/2109.02458)


  We prove that the non-structural subtype entailment problem for finite and
regular type expressions is in PSPACE. In this way we close a decidability and
complexity gap pending since 1996.

    