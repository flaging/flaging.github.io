
## 2021-10-11

### [[2110.03762] Group-based Delivery of Critical Traffic in Cellular IoT Networks](http://arxiv.org/abs/2110.03762)


  Fifth generation (5G) networks are expected to connect a huge number of
Internet of Things (IoT) devices in many usage scenarios. The challenges of
typical massive IoT applications with sporadic and short packet uplink
transmissions are well studied, while not enough attention is given to the
delivery of content of common interest, such as software/firmware updates and
remote control, towards IoT devices in emerging point-to-multipoint scenarios.
Moreover, the delivery of delay-sensitive IoT traffic is not sufficiently
addressed in the literature. In this work we (i) identify the drawbacks of the
current Single-Cell Point-to-Multipoint (SC-PTM) solution for unplanned
critical traffic delivery in cellular IoT (cIoT) networks, and (ii) propose
paging and multicast schemes for a fast distribution of critical updates after,
e.g., bug fixes or system failures. We benchmark the performance of the
proposed paging scheme against similar solutions available in the literature.
Our extended SC-PTM framework is energy efficient and guarantees low service
latency, as demonstrated both analytically and by simulations.

    

### [[2110.03766] Trusted and Secured D2D-Aided Communications in 5G Networks](http://arxiv.org/abs/2110.03766)


  The design of the forthcoming fifth generation (5G) system shall meet the
severe requirement of managing an always increasing amount of traffic generated
by both humans and machines, while guaranteeing data security. Among the
enabling technologies that will turn 5G into a reality, Device-to-Device (D2D)
and Multicasting will certainly play a key role because of their capability to
largely improve network resources utilization and to address emerging use cases
requiring the delivery of the same content to a large number of devices. D2D
communications can help to improve traditional point-to-multipoint
transmissions by reducing the multicast coverage area and exploiting properly
selected relay nodes as data forwarders towards users with worse channel
conditions. However, security issues are even more challenging for D2D
connections, as data exchange happens directly between nodes in proximity. To
enhance performance and security of delivered traffic in 5G-oriented networks,
in this paper we design SeT-D2D (Secure and Trust D2D), according to which
trustworthiness inferred from both direct interactions and social-awareness
parameters is exploited to properly select relay nodes. Main contributions of
our research consist in the introduction of a model for the assessment of
network nodes' trustworthiness and the implementation of security mechanisms to
protect the data transmitted in D2D communications and the privacy of the
involved users. The conducted simulation campaign testifies to the ability of
the proposed solution to effectively select relay nodes, which leads to an
improved network performance.

    

### [[2110.03781] 5G Traffic Prediction with Time Series Analysis](http://arxiv.org/abs/2110.03781)


  In todays day and age, a mobile phone has become a basic requirement needed
for anyone to thrive. With the cellular traffic demand increasing so
dramatically, it is now necessary to accurately predict the user traffic in
cellular networks, so as to improve the performance in terms of resource
allocation and utilisation. By leveraging the power of machine learning and
identifying its usefulness in the field of cellular networks we try to achieve
three main objectives classification of the application generating the traffic,
prediction of packet arrival intensity and burst occurrence. The design of the
prediction and classification system is done using Long Short Term Memory
model. The LSTM predictor developed in this experiment would return the number
of uplink packets and also estimate the probability of burst occurrence in the
specified future time interval. For the purpose of classification, the
regression layer in our LSTM prediction model is replaced by a softmax
classifier which is used to classify the application generating the cellular
traffic into one of the four applications including surfing, video calling,
voice calling, and video streaming.

    

### [[2110.03794] Optimal Deployment and Operation of Robotic Aerial 6G Small Cells with Grasping End Effectors](http://arxiv.org/abs/2110.03794)


  Although airborne base stations (ABSs) mounted on drones show a significant
potential to enhance network capacity and coverage due to their flexible
deployment, the system performance is limited by the endurance of the on-board
battery. To overcome this key shortcoming, we are exploring robotic airborne
base station (RABS) with energy neutral grasping end-effectors able to
autonomously perch at tall urban landforms. To this end, we propose novel
integer linear programming (ILP) optimization models and computational
efficient reformulation by proving total unimodularity problem structure to
allow optimal deployment and operation of robotic small cells based on the
spatio-temporal characteristics of underlying traffic demand from end-users. A
wide set of numerical investigations reveal that a single robotic aerial small
cell is able to outperform five (5) fixed small cells in terms of served user
generated traffic within a 16 to 41 hours period. This is because robotic
aerial small cell is able to alter its location based on actual traffic demand
rather than on average values used for fixed small cell network deployment.

    

### [[2110.03816] AS-Level BGP Community Usage Classification](http://arxiv.org/abs/2110.03816)


  BGP communities are a popular mechanism used by network operators for traffic
engineering, blackholing, and to realize network policies and business
strategies. In recent years, many research works have contributed to our
understanding of how BGP communities are utilized, as well as how they can
reveal secondary insights into real-world events such as outages and security
attacks. However, one fundamental question remains unanswered: "Which ASes tag
announcements with BGP communities and which remove communities in the
announcements they receive?" A grounded understanding of where BGP communities
are added or removed can help better model and predict BGP-based actions in the
Internet and characterize the strategies of network operators.
In this paper we develop, validate, and share data from the first algorithm
that can infer BGP community tagging and cleaning behavior at the AS-level. The
algorithm is entirely passive and uses BGP update messages and snapshots, e.g.
from public route collectors, as input. First, we quantify the correctness and
accuracy of the algorithm in controlled experiments with simulated topologies.
To validate in the wild, we announce prefixes with communities and confirm that
more than 90% of the ASes that we classify behave as our algorithm predicts.
Finally, we apply the algorithm to data from four sets of BGP collectors: RIPE,
RouteViews, Isolario, and PCH. Tuned conservatively, our algorithm ascribes
community tagging and cleaning behaviors to more than 13k ASes, the majority of
which are large networks and providers. We make our algorithm and inferences
available as a public resource to the BGP research community.

    

### [[2110.03818] Social Groups Based Content Caching in Wireless Networks](http://arxiv.org/abs/2110.03818)


  The unprecedented growth of wireless mobile traffic, mainly due to multimedia
traffic over online social platforms has strained the resources in the mobile
backhaul network. A promising approach to reduce the backhaul load is to
proactively cache content at the network edge, taking into account the overlaid
social network. Known caching schemes require complete knowledge of the social
graph and mainly focus on one-to-one interactions forgoing the prevalent mode
of content sharing among circles of 'friends'. We propose Bingo, a proactive
content caching scheme that leverages the presence of interest groups in online
social networks. The mobile network operator (MNO) can choose to incrementally
deploy Bingo at select network nodes (base stations, packet core, data center)
based on user profiles and revenue numbers. We approximate the group
memberships of users using the available user-content request logs without any
prior knowledge of the overlaid social graph. Bingo can cater to the evolving
nature of online social groups and file popularity distribution for making
caching decisions. We use synthetically generated group structures and simulate
user requests at the base station for empirical evaluation against traditional
and recent caching schemes. Bingo achieves up to 30%-34% gain over the best
baseline.

    

### [[2110.04116] Entanglement Swapping in Quantum Switches: Protocol Design and Stability Analysis](http://arxiv.org/abs/2110.04116)


  Quantum switches are critical components in quantum networks, distributing
maximally entangled pairs among end nodes by entanglement swapping. In this
work, we design protocols that schedule entanglement swapping operations in
quantum switches. Entanglement requests randomly arrive at the switch, and the
goal of an entanglement swapping protocol is to stabilize the quantum switch so
that the number of unfinished entanglement requests is bounded with a high
probability. We determine the capacity region for the rates of entanglement
requests and develop entanglement swapping protocols to stabilize the switch.
Among these protocols, the on-demand protocols are not only computationally
efficient, but also achieve high fidelity and low latency demonstrated by
results obtained using a quantum network discrete event simulator.

    

### [[2110.04259] A Wireless Intrusion Detection System for 802.11 WPA3 Networks](http://arxiv.org/abs/2110.04259)


  Wi-Fi (802.11) networks have become an essential part of our daily lives;
hence, their security is of utmost importance. However, Wi-Fi Protected Access
3 (WPA3), the latest security certification for 802.11 standards, has recently
been shown to be vulnerable to several attacks. In this paper, we first
describe the attacks on WPA3 networks that have been reported in prior work;
additionally, we show that a deauthentication attack and a beacon flood attack,
known to be possible on a WPA2 network, are still possible with WPA3. We launch
and test all the above (a total of nine) attacks using a testbed that contains
an enterprise Access Point (AP) and Intrusion Detection System (IDS). Our
experimental results show that the AP is vulnerable to eight out of the nine
attacks and the IDS is unable to detect any of them. We propose a design for a
signature-based IDS, which incorporates techniques to detect all the above
attacks. Also, we implement these techniques on our testbed and verify that our
IDS is able to successfully detect all the above attacks. We provide schemes
for mitigating the impact of the above attacks once they are detected. We make
the code to perform the above attacks as well as that of our IDS publicly
available, so that it can be used for future work by the research community at
large.

    

### [[2101.01768] Multi-Cell, Multi-Channel URLLC with Probabilistic Per-Packet Real-Time Guarantee](http://arxiv.org/abs/2101.01768)


  Consider a wireless network where each communication link has transmission
reliability and real-time guarantee. Certain pairs of wireless links interfere
with each others and this interference is modeled by a interference model which
enable predictable link reliability. Given the conflict graph and link QoS
requirements, the objective is to determine whether and how the demands of all
links can be satisfied. Prior studies have considered probabilistic per-packet
real-time guarantees for single-cell, single-channel networks with implicit
deadline constraints, but they have not considered real-world complexities such
as inter-cell interference and multiple communication channels.
Towards ensuring URLLC in multi-cell, multi-channel wireless networks, we
propose a real-time scheduling algorithm based on
\emph{local-deadline-partition (LDP)}. The LDP algorithm is suitable for
distributed implementation, and it ensures probabilistic per-packet real-time
guarantee for multi-cell, multi-channel networks with general deadline
constraints. We also address the associated challenge of the schedulability
test of URLLC traffic. In particular, we propose the concept of \emph{feasible
set} and identify a closed-form sufficient condition for the schedulability of
URLLC traffic.
We propose a distributed algorithm for the schedulability test, and the
algorithm includes a procedure for finding the minimum sum work density of
feasible sets which is of interest by itself. We also identify a necessary
condition for the schedulability of URLLC traffic, and use numerical studies to
understand a lower bound on the approximation ratio of the LDP algorithm.
We experimentally study the properties of the LDP algorithm and observe that
the URLLC traffic supportable by the LDP algorithm is significantly higher than
that of a state-of-the-art algorithm.

    

### [[2110.03687] Protecting Retail Investors from Order Book Spoofing using a GRU-based Detection Model](http://arxiv.org/abs/2110.03687)


  Market manipulation is tackled through regulation in traditional markets
because of its detrimental effect on market efficiency and many participating
financial actors. The recent increase of private retail investors due to new
low-fee platforms and new asset classes such as decentralised digital
currencies has increased the number of vulnerable actors due to lack of
institutional sophistication and strong regulation. This paper proposes a
method to detect illicit activity and inform investors on spoofing attempts, a
well-known market manipulation technique. Our framework is based on a highly
extendable Gated Recurrent Unit (GRU) model and allows the inclusion of market
variables that can explain spoofing and potentially other illicit activities.
The model is tested on granular order book data, in one of the most unregulated
markets prone to spoofing with a large number of non-institutional traders. The
results show that the model is performing well in an early detection context,
allowing the identification of spoofing attempts soon enough to allow investors
to react. This is the first step to a fully comprehensive model that will
protect investors in various unregulated trading environments and regulators to
identify illicit activity.

    

### [[2110.03688] Predicting Chemical Hazard across Taxa through Machine Learning](http://arxiv.org/abs/2110.03688)


  We apply machine learning methods to predict chemical hazards focusing on
fish acute toxicity across taxa. We analyze the relevance of taxonomy and
experimental setup, and show that taking them into account can lead to
considerable improvements in the classification performance. We quantify the
gain obtained by introducing the taxonomic and experimental information,
compared to classifying based on chemical information alone. We use our
approach with standard machine learning models (K-nearest neighbors, random
forests and deep neural networks), as well as the recently proposed Read-Across
Structure Activity Relationship (RASAR) models, which were very successful in
predicting chemical hazards to mammals based on chemical similarity. We are
able to obtain accuracies of over 0.93 on datasets where, due to noise in the
data, the maximum achievable accuracy is expected to be below 0.95, which
results in an effective accuracy of 0.98. The best performances are obtained by
random forests and RASAR models. We analyze metrics to compare our results with
animal test reproducibility, and despite most of our models 'outperform animal
test reproducibility' as measured through recently proposed metrics, we show
that the comparison between machine learning performance and animal test
reproducibility should be addressed with particular care. While we focus on
fish mortality, our approach, provided that the right data is available, is
valid for any combination of chemicals, effects and taxa.

    

### [[2110.03689] DeepECMP: Predicting Extracellular Matrix Proteins using Deep Learning](http://arxiv.org/abs/2110.03689)


  Introduction: The extracellular matrix (ECM) is a networkof proteins and
carbohydrates that has a structural and bio-chemical function. The ECM plays an
important role in dif-ferentiation, migration and signaling. Several studies
havepredicted ECM proteins using machine learning algorithmssuch as Random
Forests, K-nearest neighbours and supportvector machines but is yet to be
explored using deep learn-ing. Method: DeepECMP was developed using several
previ-ously used ECM datasets, asymmetric undersampling andan ensemble of 11
feed-forward neural networks. Results: The performance of DeepECMP was 83.6%
bal-anced accuracy which outperformed several algorithms. Inaddition, the
pipeline of DeepECMP has been shown to behighly efficient. Conclusion: This
paper is the first to focus on utilizingdeep learning for ECM prediction.
Several limitations areovercome by DeepECMP such as computational
expense,availability to the public and usability outside of the humanspecies

    

### [[2110.03690] Learning Higher-Order Dynamics in Video-Based Cardiac Measurement](http://arxiv.org/abs/2110.03690)


  Computer vision methods typically optimize for first-order dynamics (e.g.,
optical flow). However, in many cases the properties of interest are subtle
variations in higher-order changes, such as acceleration. This is true in the
cardiac pulse, where the second derivative can be used as an indicator of blood
pressure and arterial disease. Recent developments in camera-based vital sign
measurement have shown that cardiac measurements can be recovered with
impressive accuracy from videos; however, the majority of research has focused
on extracting summary statistics such as heart rate. Less emphasis has been put
on the accuracy of waveform morphology that is necessary for many clinically
impactful scenarios. In this work, we provide evidence that higher-order
dynamics are better estimated by neural models when explicitly optimized for in
the loss function. Furthermore, adding second-derivative inputs also improves
performance when estimating second-order dynamics. By incorporating the second
derivative of both the input frames and the target vital sign signals into the
training procedure, our model is better able to estimate left ventricle
ejection time (LVET) intervals.

    

### [[2110.03691] Direct design of biquad filter cascades with deep learning by sampling random polynomials](http://arxiv.org/abs/2110.03691)


  Designing infinite impulse response filters to match an arbitrary magnitude
response requires specialized techniques. Methods like modified Yule-Walker are
relatively efficient, but may not be sufficiently accurate in matching high
order responses. On the other hand, iterative optimization techniques often
enable superior performance, but come at the cost of longer run-times and are
sensitive to initial conditions, requiring manual tuning. In this work, we
address some of these limitations by learning a direct mapping from the target
magnitude response to the filter coefficient space with a neural network
trained on millions of random filters. We demonstrate our approach enables both
fast and accurate estimation of filter coefficients given a desired response.
We investigate training with different families of random filters, and find
training with a variety of filter families enables better generalization when
estimating real-world filters, using head-related transfer functions and guitar
cabinets as case studies. We compare our method against existing methods
including modified Yule-Walker and gradient descent and show IIRNet is, on
average, both faster and more accurate.

    

### [[2110.03706] SVG-Net: An SVG-based Trajectory Prediction Model](http://arxiv.org/abs/2110.03706)


  Anticipating motions of vehicles in a scene is an essential problem for safe
autonomous driving systems. To this end, the comprehension of the scene's
infrastructure is often the main clue for predicting future trajectories. Most
of the proposed approaches represent the scene with a rasterized format and
some of the more recent approaches leverage custom vectorized formats. In
contrast, we propose representing the scene's information by employing Scalable
Vector Graphics (SVG). SVG is a well-established format that matches the
problem of trajectory prediction better than rasterized formats while being
more general than arbitrary vectorized formats. SVG has the potential to
provide the convenience and generality of raster-based solutions if coupled
with a powerful tool such as CNNs, for which we introduce SVG-Net. SVG-Net is a
Transformer-based Neural Network that can effectively capture the scene's
information from SVG inputs. Thanks to the self-attention mechanism in its
Transformers, SVG-Net can also adequately apprehend relations amongst the scene
and the agents. We demonstrate SVG-Net's effectiveness by evaluating its
performance on the publicly available Argoverse forecasting dataset. Finally,
we illustrate how, by using SVG, one can benefit from datasets and advancements
in other research fronts that also utilize the same input format. Our code is
available at this https URL.

    

### [[2110.03712] Gaussian Process for Trajectories](http://arxiv.org/abs/2110.03712)


  The Gaussian process is a powerful and flexible technique for interpolating
spatiotemporal data, especially with its ability to capture complex trends and
uncertainty from the input signal. This chapter describes Gaussian processes as
an interpolation technique for geospatial trajectories. A Gaussian process
models measurements of a trajectory as coming from a multidimensional Gaussian,
and it produces for each timestamp a Gaussian distribution as a prediction. We
discuss elements that need to be considered when applying Gaussian process to
trajectories, common choices for those elements, and provide a concrete example
of implementing a Gaussian process.

    

### [[2110.03722] A Meta-learning Approach to Reservoir Computing: Time Series Prediction with Limited Data](http://arxiv.org/abs/2110.03722)


  Recent research has established the effectiveness of machine learning for
data-driven prediction of the future evolution of unknown dynamical systems,
including chaotic systems. However, these approaches require large amounts of
measured time series data from the process to be predicted. When only limited
data is available, forecasters are forced to impose significant model structure
that may or may not accurately represent the process of interest. In this work,
we present a Meta-learning Approach to Reservoir Computing (MARC), a
data-driven approach to automatically extract an appropriate model structure
from experimentally observed "related" processes that can be used to vastly
reduce the amount of data required to successfully train a predictive model. We
demonstrate our approach on a simple benchmark problem, where it beats the
state of the art meta-learning techniques, as well as a challenging chaotic
problem.

    

### [[2110.03726] Bisimulations for Neural Network Reduction](http://arxiv.org/abs/2110.03726)


  We present a notion of bisimulation that induces a reduced network which is
semantically equivalent to the given neural network. We provide a minimization
algorithm to construct the smallest bisimulation equivalent network. Reductions
that construct bisimulation equivalent neural networks are limited in the scale
of reduction. We present an approximate notion of bisimulation that provides
semantic closeness, rather than, semantic equivalence, and quantify semantic
deviation between the neural networks that are approximately bisimilar. The
latter provides a trade-off between the amount of reduction and deviations in
the semantics.

    

### [[2110.03727] Contextual Sentence Classification: Detecting Sustainability Initiatives in Company Reports](http://arxiv.org/abs/2110.03727)


  We introduce the novel task of detecting sustainability initiatives in
company reports. Given a full report, the aim is to automatically identify
mentions of practical activities that a company has performed in order to
tackle specific societal issues. As a single initiative can often be described
over multiples sentences, new methods for identifying continuous sentence spans
needs to be developed. We release a new dataset of company reports in which the
text has been manually annotated with sustainability initiatives. We also
evaluate different models for initiative detection, introducing a novel
aggregation and evaluation methodology. Our proposed architecture uses
sequences of five consecutive sentences to account for contextual information
when making classification decisions at the individual sentence level.

    

### [[2110.03735] Adversarial Unlearning of Backdoors via Implicit Hypergradient](http://arxiv.org/abs/2110.03735)


  We propose a minimax formulation for removing backdoors from a given poisoned
model based on a small set of clean data. This formulation encompasses much of
prior work on backdoor removal. We propose the Implicit Bacdoor Adversarial
Unlearning (I-BAU) algorithm to solve the minimax. Unlike previous work, which
breaks down the minimax into separate inner and outer problems, our algorithm
utilizes the implicit hypergradient to account for the interdependence between
inner and outer optimization. We theoretically analyze its convergence and the
generalizability of the robustness gained by solving minimax on clean data to
unseen test data. In our evaluation, we compare I-BAU with six state-of-art
backdoor defenses on seven backdoor attacks over two datasets and various
attack settings, including the common setting where the attacker targets one
class as well as important but underexplored settings where multiple classes
are targeted. I-BAU's performance is comparable to and most often significantly
better than the best baseline. Particularly, its performance is more robust to
the variation on triggers, attack settings, poison ratio, and clean data size.
Moreover, I-BAU requires less computation to take effect; particularly, it is
more than $13\times$ faster than the most efficient baseline in the
single-target attack setting. Furthermore, it can remain effective in the
extreme case where the defender can only access 100 clean samples -- a setting
where all the baselines fail to produce acceptable results.

    

### [[2110.03740] Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](http://arxiv.org/abs/2110.03740)


  Deep learning in the presence of noisy annotations has been studied
extensively in classification, but much less in segmentation tasks. In this
work, we study the learning dynamics of deep segmentation networks trained on
inaccurately-annotated data. We discover a phenomenon that has been previously
reported in the context of classification: the networks tend to first fit the
clean pixel-level labels during an "early-learning" phase, before eventually
memorizing the false annotations. However, in contrast to classification,
memorization in segmentation does not arise simultaneously for all semantic
categories. Inspired by these findings, we propose a new method for
segmentation from noisy annotations with two key elements. First, we detect the
beginning of the memorization phase separately for each category during
training. This allows us to adaptively correct the noisy annotations in order
to exploit early learning. Second, we incorporate a regularization term that
enforces consistency across scales to boost robustness against annotation
noise. Our method outperforms standard approaches on a medical-imaging
segmentation task where noises are synthesized to mimic human annotation
errors. It also provides robustness to realistic noisy annotations present in
weakly-supervised semantic segmentation, achieving state-of-the-art results on
PASCAL VOC 2012.

    

### [[2110.03742] Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference](http://arxiv.org/abs/2110.03742)


  Sparse Mixture-of-Experts (MoE) has been a successful approach for scaling
multilingual translation models to billions of parameters without a
proportional increase in training computation. However, MoE models are
prohibitively large and practitioners often resort to methods such as
distillation for serving. In this work, we investigate routing strategies at
different granularity (token, sentence, task) in MoE models to bypass
distillation. Experiments on WMT and a web-scale dataset suggest that
task-level routing (task-MoE) enables us to extract smaller, ready-to-deploy
sub-networks from large sparse models. On WMT, our task-MoE with 32 experts
(533M parameters) outperforms the best performing token-level MoE model
(token-MoE) by +1.0 BLEU on average across 30 language pairs. The peak
inference throughput is also improved by a factor of 1.9x when we route by
tasks instead of tokens. While distilling a token-MoE to a smaller dense model
preserves only 32% of the BLEU gains, our sub-network task-MoE, by design,
preserves all the gains with the same inference cost as the distilled student
model. Finally, when scaling up to 200 language pairs, our 128-expert task-MoE
(13B parameters) performs competitively with a token-level counterpart, while
improving the peak inference throughput by a factor of 2.6x.

    

### [[2110.03743] Reinforcement Learning in Reward-Mixing MDPs](http://arxiv.org/abs/2110.03743)


  Learning a near optimal policy in a partially observable system remains an
elusive challenge in contemporary reinforcement learning. In this work, we
consider episodic reinforcement learning in a reward-mixing Markov decision
process (MDP). There, a reward function is drawn from one of multiple possible
reward models at the beginning of every episode, but the identity of the chosen
reward model is not revealed to the agent. Hence, the latent state space, for
which the dynamics are Markovian, is not given to the agent. We study the
problem of learning a near optimal policy for two reward-mixing MDPs. Unlike
existing approaches that rely on strong assumptions on the dynamics, we make no
assumptions and study the problem in full generality. Indeed, with no further
assumptions, even for two switching reward-models, the problem requires several
new ideas beyond existing algorithmic and analysis techniques for efficient
exploration. We provide the first polynomial-time algorithm that finds an
$\epsilon$-optimal policy after exploring $\tilde{O}(poly(H,\epsilon^{-1})
\cdot S^2 A^2)$ episodes, where $H$ is time-horizon and $S, A$ are the number
of states and actions respectively. This is the first efficient algorithm that
does not require any assumptions in partially observed environments where the
observation space is smaller than the latent state space.

    

### [[2110.03749] Global sensitivity analysis in probabilistic graphical models](http://arxiv.org/abs/2110.03749)


  We show how to apply Sobol's method of global sensitivity analysis to measure
the influence exerted by a set of nodes' evidence on a quantity of interest
expressed by a Bayesian network. Our method exploits the network structure so
as to transform the problem of Sobol index estimation into that of
marginalization inference. This way, we can efficiently compute indices for
networks where brute-force or Monte Carlo based estimators for variance-based
sensitivity analysis would require millions of costly samples. Moreover, our
method gives exact results when exact inference is used, and also supports the
case of correlated inputs. The proposed algorithm is inspired by the field of
tensor networks, and generalizes earlier tensor sensitivity techniques from the
acyclic to the cyclic case. We demonstrate the method on three medium to large
Bayesian networks that cover the areas of project risk management and
reliability engineering.

    

### [[2110.03753] From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness](http://arxiv.org/abs/2110.03753)


  Message Passing Neural Networks (MPNNs) are a common type of Graph Neural
Network (GNN), in which each node's representation is computed recursively by
aggregating representations (messages) from its immediate neighbors akin to a
star-shaped pattern. MPNNs are appealing for being efficient and scalable,
how-ever their expressiveness is upper-bounded by the 1st-order
Weisfeiler-Lehman isomorphism test (1-WL). In response, prior works propose
highly expressive models at the cost of scalability and sometimes
generalization performance. Our work stands between these two regimes: we
introduce a general framework to uplift any MPNN to be more expressive, with
limited scalability overhead and greatly improved practical performance. We
achieve this by extending local aggregation in MPNNs from star patterns to
general subgraph patterns (e.g.,k-egonets):in our framework, each node
representation is computed as the encoding of a surrounding induced subgraph
rather than encoding of immediate neighbors only (i.e. a star). We choose the
subgraph encoder to be a GNN (mainly MPNNs, considering scalability) to design
a general framework that serves as a wrapper to up-lift any GNN. We call our
proposed method GNN-AK(GNN As Kernel), as the framework resembles a
convolutional neural network by replacing the kernel with GNNs. Theoretically,
we show that our framework is strictly more powerful than 1&2-WL, and is not
less powerful than 3-WL. We also design subgraph sampling strategies which
greatly reduce memory footprint and improve speed while maintaining
performance. Our method sets new state-of-the-art performance by large margins
for several well-known graph ML tasks; specifically, 0.08 MAE on ZINC,74.79%
and 86.887% accuracy on CIFAR10 and PATTERN respectively.

    

### [[2110.03757] Predictive Maintenance for General Aviation Using Convolutional Transformers](http://arxiv.org/abs/2110.03757)


  Predictive maintenance systems have the potential to significantly reduce
costs for maintaining aircraft fleets as well as provide improved safety by
detecting maintenance issues before they come severe. However, the development
of such systems has been limited due to a lack of publicly labeled multivariate
time series (MTS) sensor data. MTS classification has advanced greatly over the
past decade, but there is a lack of sufficiently challenging benchmarks for new
methods. This work introduces the NGAFID Maintenance Classification (NGAFID-MC)
dataset as a novel benchmark in terms of difficulty, number of samples, and
sequence length. NGAFID-MC consists of over 7,500 labeled flights, representing
over 11,500 hours of per second flight data recorder readings of 23 sensor
parameters. Using this benchmark, we demonstrate that Recurrent Neural Network
(RNN) methods are not well suited for capturing temporally distant
relationships and propose a new architecture called Convolutional Multiheaded
Self Attention (Conv-MHSA) that achieves greater classification performance at
greater computational efficiency. We also demonstrate that image inspired
augmentations of cutout, mixup, and cutmix, can be used to reduce overfitting
and improve generalization in MTS classification. Our best trained models have
been incorporated back into the NGAFID to allow users to potentially detect
flights that require maintenance as well as provide feedback to further expand
and refine the NGAFID-MC dataset.

    

### [[2110.03759] Explanation as a process: user-centric construction of multi-level and multi-modal explanations](http://arxiv.org/abs/2110.03759)


  In the last years, XAI research has mainly been concerned with developing new
technical approaches to explain deep learning models. Just recent research has
started to acknowledge the need to tailor explanations to different contexts
and requirements of stakeholders. Explanations must not only suit developers of
models, but also domain experts as well as end users. Thus, in order to satisfy
different stakeholders, explanation methods need to be combined. While
multi-modal explanations have been used to make model predictions more
transparent, less research has focused on treating explanation as a process,
where users can ask for information according to the level of understanding
gained at a certain point in time. Consequently, an opportunity to explore
explanations on different levels of abstraction should be provided besides
multi-modal explanations. We present a process-based approach that combines
multi-level and multi-modal explanations. The user can ask for textual
explanations or visualizations through conversational interaction in a
drill-down manner. We use Inductive Logic Programming, an interpretable machine
learning approach, to learn a comprehensible model. Further, we present an
algorithm that creates an explanatory tree for each example for which a
classifier decision is to be explained. The explanatory tree can be navigated
by the user to get answers of different levels of detail. We provide a
proof-of-concept implementation for concepts induced from a semantic net about
living beings.

    

### [[2110.03761] A simple equivariant machine learning method for dynamics based on scalars](http://arxiv.org/abs/2110.03761)


  Physical systems obey strict symmetry principles. We expect that machine
learning methods that intrinsically respect these symmetries should perform
better than those that do not. In this work we implement a principled model
based on invariant scalars, and release open-source code. We apply this
\textsl{Scalars} method to a simple chaotic dynamical system, the springy
double pendulum. We show that the Scalars method outperforms state-of-the-art
approaches for learning the properties of physical systems with symmetries,
both in terms of accuracy and speed. Because the method incorporates the
fundamental symmetries, we expect it to generalize to different settings, such
as changes in the force laws in the system.

    

### [[2110.03763] Label Propagation across Graphs: Node Classification using Graph Neural Tangent Kernels](http://arxiv.org/abs/2110.03763)


  Graph neural networks (GNNs) have achieved superior performance on node
classification tasks in the last few years. Commonly, this is framed in a
transductive semi-supervised learning setup wherein the entire graph, including
the target nodes to be labeled, is available for training. Driven in part by
scalability, recent works have focused on the inductive case where only the
labeled portion of a graph is available for training. In this context, our
current work considers a challenging inductive setting where a set of labeled
graphs are available for training while the unlabeled target graph is
completely separate, i.e., there are no connections between labeled and
unlabeled nodes. Under the implicit assumption that the testing and training
graphs come from similar distributions, our goal is to develop a labeling
function that generalizes to unobserved connectivity structures. To that end,
we employ a graph neural tangent kernel (GNTK) that corresponds to infinitely
wide GNNs to find correspondences between nodes in different graphs based on
both the topology and the node features. We augment the capabilities of the
GNTK with residual connections and empirically illustrate its performance gains
on standard benchmarks.

    

### [[2110.03765] Food Science Spectroscopy Model Training: Improving Data Efficiency Using Active Learning and Semi-Supervised Learning](http://arxiv.org/abs/2110.03765)


  The past decade witnesses a rapid development in the measurement and
monitoring technologies for food science. Among these technologies,
spectroscopy has been widely used for the analysis of food quality, safety, and
nutritional properties. Due to the complexity of food systems and the lack of
comprehensive predictive models, rapid and simple measurements to predict
complex properties in food systems are largely missing. Machine Learning (ML)
has shown great potential to improve classification and prediction of these
properties. However, the barriers to collect large datasets for ML applications
still persists. In this paper, we explore different approaches of data
annotation and model training to improve data efficiency for ML applications.
Specifically, we leverage Active Learning (AL) and Semi-Supervised Learning
(SSL) and investigate four approaches: baseline passive learning, AL, SSL, and
a hybrid of AL and SSL. To evaluate these approaches, we collect two
spectroscopy datasets: predicting plasma dosage and detecting foodborne
pathogen. Our experimental results show that, compared to the de facto passive
learning approach, AL and SSL methods reduce the number of labeled samples by
50% and 25% for each ML application, respectively.

    

### [[2110.03768] De-randomizing MCMC dynamics with the diffusion Stein operator](http://arxiv.org/abs/2110.03768)


  Approximate Bayesian inference estimates descriptors of an intractable target
distribution - in essence, an optimization problem within a family of
distributions. For example, Langevin dynamics (LD) extracts asymptotically
exact samples from a diffusion process because the time evolution of its
marginal distributions constitutes a curve that minimizes the KL-divergence via
steepest descent in the Wasserstein space. Parallel to LD, Stein variational
gradient descent (SVGD) similarly minimizes the KL, albeit endowed with a novel
Stein-Wasserstein distance, by deterministically transporting a set of particle
samples, thus de-randomizes the stochastic diffusion process. We propose
de-randomized kernel-based particle samplers to all diffusion-based samplers
known as MCMC dynamics. Following previous work in interpreting MCMC dynamics,
we equip the Stein-Wasserstein space with a fiber-Riemannian Poisson structure,
with the capacity of characterizing a fiber-gradient Hamiltonian flow that
simulates MCMC dynamics. Such dynamics discretizes into generalized SVGD
(GSVGD), a Stein-type deterministic particle sampler, with particle updates
coinciding with applying the diffusion Stein operator to a kernel function. We
demonstrate empirically that GSVGD can de-randomize complex MCMC dynamics,
which combine the advantages of auxiliary momentum variables and Riemannian
structure, while maintaining the high sample quality from an interacting
particle system.

    

### [[2110.03771] Wake-Cough: cough spotting and cougher identification for personalised long-term cough monitoring](http://arxiv.org/abs/2110.03771)


  We present 'wake-cough', an application of wake-word spotting to coughs using
Resnet50 and identifying coughers using i-vectors, for the purpose of a
long-term, personalised cough monitoring system. Coughs, recorded in a quiet
(73$\pm$5 dB) and noisy (34$\pm$17 dB) environment, were used to extract
i-vectors, x-vectors and d-vectors, used as features to the classifiers. The
system achieves 90.02\% accuracy from an MLP to discriminate 51 coughers using
2-sec long cough segments in the noisy environment. When discriminating between
5 and 14 coughers using longer (100 sec) segments in the quiet environment,
this accuracy rises to 99.78\% and 98.39\% respectively. Unlike speech,
i-vectors outperform x-vectors and d-vectors in identifying coughers. These
coughs were added as an extra class in the Google Speech Commands dataset and
features were extracted by preserving the end-to-end time-domain information in
an event. The highest accuracy of 88.58\% is achieved in spotting coughs among
35 other trigger phrases using a Resnet50. Wake-cough represents a
personalised, non-intrusive, cough monitoring system, which is power efficient
as using wake-word detection method can keep a smartphone-based monitoring
device mostly dormant. This makes wake-cough extremely attractive in multi-bed
ward environments to monitor patient's long-term recovery from lung ailments
such as tuberculosis and COVID-19.

    

### [[2110.03774] Automatically Polyconvex Strain Energy Functions using Neural Ordinary Differential Equations](http://arxiv.org/abs/2110.03774)


  Data-driven methods are becoming an essential part of computational mechanics
due to their unique advantages over traditional material modeling. Deep neural
networks are able to learn complex material response without the constraints of
closed-form approximations. However, imposing the physics-based mathematical
requirements that any material model must comply with is not straightforward
for data-driven approaches. In this study, we use a novel class of neural
networks, known as neural ordinary differential equations (N-ODEs), to develop
data-driven material models that automatically satisfy polyconvexity of the
strain energy function with respect to the deformation gradient, a condition
needed for the existence of minimizers for boundary value problems in
elasticity. We take advantage of the properties of ordinary differential
equations to create monotonic functions that approximate the derivatives of the
strain energy function with respect to the invariants of the right Cauchy-Green
deformation tensor. The monotonicity of the derivatives guarantees the
convexity of the energy. The N-ODE material model is able to capture synthetic
data generated from closed-form material models, and it outperforms
conventional models when tested against experimental data on skin, a highly
nonlinear and anisotropic material. We also showcase the use of the N-ODE
material model in finite element simulations. The framework is general and can
be used to model a large class of materials. Here we focus on hyperelasticity,
but polyconvex strain energies are a core building block for other problems in
elasticity such as viscous and plastic deformations. We therefore expect our
methodology to further enable data-driven methods in computational mechanics

    

### [[2110.03780] A composable autoencoder-based iterative algorithm for accelerating numerical simulations](http://arxiv.org/abs/2110.03780)


  Numerical simulations for engineering applications solve partial differential
equations (PDE) to model various physical processes. Traditional PDE solvers
are very accurate but computationally costly. On the other hand, Machine
Learning (ML) methods offer a significant computational speedup but face
challenges with accuracy and generalization to different PDE conditions, such
as geometry, boundary conditions, initial conditions and PDE source terms. In
this work, we propose a novel ML-based approach, CoAE-MLSim (Composable
AutoEncoder Machine Learning Simulation), which is an unsupervised,
lower-dimensional, local method, that is motivated from key ideas used in
commercial PDE solvers. This allows our approach to learn better with
relatively fewer samples of PDE solutions. The proposed ML-approach is compared
against commercial solvers for better benchmarks as well as latest
ML-approaches for solving PDEs. It is tested for a variety of complex
engineering cases to demonstrate its computational speed, accuracy,
scalability, and generalization across different PDE conditions. The results
show that our approach captures physics accurately across all metrics of
comparison (including measures such as results on section cuts and lines).

    

### [[2110.03783] Machine Learning approaches to do size based reasoning on Retail Shelf objects to classify product variants](http://arxiv.org/abs/2110.03783)


  There has been a surge in the number of Machine Learning methods to analyze
products kept on retail shelves images. Deep learning based computer vision
methods can be used to detect products on retail shelves and then classify
them. However, there are different sized variants of products which look
exactly the same visually and the method to differentiate them is to look at
their relative sizes with other products on shelves. This makes the process of
deciphering the sized based variants from each other using computer vision
algorithms alone impractical. In this work, we propose methods to ascertain the
size variant of the product as a downstream task to an object detector which
extracts products from shelf and a classifier which determines product brand.
Product variant determination is the task which assigns a product variant to
products of a brand based on the size of bounding boxes and brands predicted by
classifier. While gradient boosting based methods work well for products whose
facings are clear and distinct, a noise accommodating Neural Network method is
proposed for cases where the products are stacked irregularly.

    

### [[2110.03785] Addressing practical challenges in Active Learning via a hybrid query strategy](http://arxiv.org/abs/2110.03785)


  Active Learning (AL) is a powerful tool to address modern machine learning
problems with significantly fewer labeled training instances. However,
implementation of traditional AL methodologies in practical scenarios is
accompanied by multiple challenges due to the inherent assumptions. There are
several hindrances, such as unavailability of labels for the AL algorithm at
the beginning; unreliable external source of labels during the querying
process; or incompatible mechanisms to evaluate the performance of Active
Learner. Inspired by these practical challenges, we present a hybrid query
strategy-based AL framework that addresses three practical challenges
simultaneously: cold-start, oracle uncertainty and performance evaluation of
Active Learner in the absence of ground truth. While a pre-clustering approach
is employed to address the cold-start problem, the uncertainty surrounding the
expertise of labeler and confidence in the given labels is incorporated to
handle oracle uncertainty. The heuristics obtained during the querying process
serve as the fundamental premise for accessing the performance of Active
Learner. The robustness of the proposed AL framework is evaluated across three
different environments and industrial settings. The results demonstrate the
capability of the proposed framework to tackle practical challenges during AL
implementation in real-world scenarios.

    

### [[2110.03786] Efficient large-scale image retrieval with deep feature orthogonality and Hybrid-Swin-Transformers](http://arxiv.org/abs/2110.03786)


  We present an efficient end-to-end pipeline for largescale landmark
recognition and retrieval. We show how to combine and enhance concepts from
recent research in image retrieval and introduce two architectures especially
suited for large-scale landmark identification. A model with deep orthogonal
fusion of local and global features (DOLG) using an EfficientNet backbone as
well as a novel Hybrid-Swin-Transformer is discussed and details how to train
both architectures efficiently using a step-wise approach and a sub-center
arcface loss with dynamic margins are provided. Furthermore, we elaborate a
novel discriminative re-ranking methodology for image retrieval. The
superiority of our approach was demonstrated by winning the recognition and
retrieval track of the Google Landmark Competition 2021.

    

### [[2110.03789] Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding](http://arxiv.org/abs/2110.03789)


  Knowledge graph embedding involves learning representations of entities --
the vertices of the graph -- and relations -- the edges of the graph -- such
that the resulting representations encode the known factual information
represented by the knowledge graph are internally consistent and can be used in
the inference of new relations. We show that knowledge graph embedding is
naturally expressed in the topological and categorical language of
\textit{cellular sheaves}: learning a knowledge graph embedding corresponds to
learning a \textit{knowledge sheaf} over the graph, subject to certain
constraints. In addition to providing a generalized framework for reasoning
about knowledge graph embedding models, this sheaf-theoretic perspective admits
the expression of a broad class of prior constraints on embeddings and offers
novel inferential capabilities. We leverage the recently developed spectral
theory of sheaf Laplacians to understand the local and global consistency of
embeddings and develop new methods for reasoning over composite relations
through harmonic extension with respect to the sheaf Laplacian. We then
implement these ideas to highlight the benefits of the extensions inspired by
this new perspective.

    

### [[2110.03790] Scaling Bayesian Optimization With Game Theory](http://arxiv.org/abs/2110.03790)


  We introduce the algorithm Bayesian Optimization (BO) with Fictitious Play
(BOFiP) for the optimization of high dimensional black box functions. BOFiP
decomposes the original, high dimensional, space into several sub-spaces
defined by non-overlapping sets of dimensions. These sets are randomly
generated at the start of the algorithm, and they form a partition of the
dimensions of the original space. BOFiP searches the original space with
alternating BO, within sub-spaces, and information exchange among sub-spaces,
to update the sub-space function evaluation. The basic idea is to distribute
the high dimensional optimization across low dimensional sub-spaces, where each
sub-space is a player in an equal interest game. At each iteration, BO produces
approximate best replies that update the players belief distribution. The
belief update and BO alternate until a stopping condition is met.
High dimensional problems are common in real applications, and several
contributions in the BO literature have highlighted the difficulty in scaling
to high dimensions due to the computational complexity associated to the
estimation of the model hyperparameters. Such complexity is exponential in the
problem dimension, resulting in substantial loss of performance for most
techniques with the increase of the input dimensionality.
We compare BOFiP to several state-of-the-art approaches in the field of high
dimensional black box optimization. The numerical experiments show the
performance over three benchmark objective functions from 20 up to 1000
dimensions. A neural network architecture design problem is tested with 42 up
to 911 nodes in 6 up to 92 layers, respectively, resulting into networks with
500 up to 10,000 weights. These sets of experiments empirically show that BOFiP
outperforms its competitors, showing consistent performance across different
problems and increasing problem dimensionality.

    

### [[2110.03800] CCGG: A Deep Autoregressive Model for Class-Conditional Graph Generation](http://arxiv.org/abs/2110.03800)


  Graph data structures are fundamental for studying connected entities. With
an increase in the number of applications where data is represented as graphs,
the problem of graph generation has recently become a hot topic in many signal
processing areas. However, despite its significance, conditional graph
generation that creates graphs with desired features is relatively less
explored in previous studies. This paper addresses the problem of
class-conditional graph generation that uses class labels as generation
constraints by introducing the Class Conditioned Graph Generator (CCGG). We
built CCGG by adding the class information as an additional input to a graph
generator model and including a classification loss in its total loss along
with a gradient passing trick. Our experiments show that CCGG outperforms
existing conditional graph generation methods on various datasets. It also
manages to maintain the quality of the generated graphs in terms of
distribution-based evaluation metrics.

    

### [[2110.03802] Hitting the Target: Stopping Active Learning at the Cost-Based Optimum](http://arxiv.org/abs/2110.03802)


  Active learning allows machine learning models to be trained using fewer
labels while retaining similar performance to traditional fully supervised
learning. An active learner selects the most informative data points, requests
their labels, and retrains itself. While this approach is promising, it leaves
an open problem of how to determine when the model is `good enough' without the
additional labels required for traditional evaluation. In the past, different
stopping criteria have been proposed aiming to identify the optimal stopping
point. However, optimality can only be expressed as a domain-dependent
trade-off between accuracy and the number of labels, and no criterion is
superior in all applications. This paper is the first to give actionable advice
to practitioners on what stopping criteria they should use in a given
real-world scenario. We contribute the first large-scale comparison of stopping
criteria, using a cost measure to quantify the accuracy/label trade-off, public
implementations of all stopping criteria we evaluate, and an open-source
framework for evaluating stopping criteria. Our research enables practitioners
to substantially reduce labelling costs by utilizing the stopping criterion
which best suits their domain.

    

### [[2110.03804] FOCUS: Familiar Objects in Common and Uncommon Settings](http://arxiv.org/abs/2110.03804)


  Standard training datasets for deep learning often contain objects in common
settings (e.g., "a horse on grass" or "a ship in water") since they are usually
collected by randomly scraping the web. Uncommon and rare settings (e.g., "a
plane on water", "a car in snowy weather") are thus severely under-represented
in the training data. This can lead to an undesirable bias in model predictions
towards common settings and create a false sense of accuracy. In this paper, we
introduce FOCUS (Familiar Objects in Common and Uncommon Settings), a dataset
for stress-testing the generalization power of deep image classifiers. By
leveraging the power of modern search engines, we deliberately gather data
containing objects in common and uncommon settings in a wide range of
locations, weather conditions, and time of day. We present a detailed analysis
of the performance of various popular image classifiers on our dataset and
demonstrate a clear drop in performance when classifying images in uncommon
settings. By analyzing deep features of these models, we show that such errors
can be due to the use of spurious features in model predictions. We believe
that our dataset will aid researchers in understanding the inability of deep
models to generalize well to uncommon settings and drive future work on
improving their distributional robustness.

    

### [[2110.03825] Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks](http://arxiv.org/abs/2110.03825)


  Deep neural networks (DNNs) are known to be vulnerable to adversarial
attacks. A range of defense methods have been proposed to train adversarially
robust DNNs, among which adversarial training has demonstrated promising
results. However, despite preliminary understandings developed for adversarial
training, it is still not clear, from the architectural perspective, what
configurations can lead to more robust DNNs. In this paper, we address this gap
via a comprehensive investigation on the impact of network width and depth on
the robustness of adversarially trained DNNs. Specifically, we make the
following key observations: 1) more parameters (higher model capacity) does not
necessarily help adversarial robustness; 2) reducing capacity at the last stage
(the last group of blocks) of the network can actually improve adversarial
robustness; and 3) under the same parameter budget, there exists an optimal
architectural configuration for adversarial robustness. We also provide a
theoretical analysis explaning why such network configuration can help
robustness. These architectural insights can help design adversarially robust
DNNs. Code is available at \url{this https URL}.

    

### [[2110.03848] Speeding up Deep Model Training by Sharing Weights and Then Unsharing](http://arxiv.org/abs/2110.03848)


  We propose a simple and efficient approach for training the BERT model. Our
approach exploits the special structure of BERT that contains a stack of
repeated modules (i.e., transformer encoders). Our proposed approach first
trains BERT with the weights shared across all the repeated modules till some
point. This is for learning the commonly shared component of weights across all
repeated layers. We then stop weight sharing and continue training until
convergence. We present theoretic insights for training by sharing weights then
unsharing with analysis for simplified models. Empirical experiments on the
BERT model show that our method yields better performance of trained models,
and significantly reduces the number of training iterations.

    

### [[2110.03860] Token Pooling in Visual Transformers](http://arxiv.org/abs/2110.03860)


  Despite the recent success in many applications, the high computational
requirements of vision transformers limit their use in resource-constrained
settings. While many existing methods improve the quadratic complexity of
attention, in most vision transformers, self-attention is not the major
computation bottleneck, e.g., more than 80% of the computation is spent on
fully-connected layers. To improve the computational complexity of all layers,
we propose a novel token downsampling method, called Token Pooling, efficiently
exploiting redundancies in the images and intermediate token representations.
We show that, under mild assumptions, softmax-attention acts as a
high-dimensional low-pass (smoothing) filter. Thus, its output contains
redundancy that can be pruned to achieve a better trade-off between the
computational cost and accuracy. Our new technique accurately approximates a
set of tokens by minimizing the reconstruction error caused by downsampling. We
solve this optimization problem via cost-efficient clustering. We rigorously
analyze and compare to prior downsampling methods. Our experiments show that
Token Pooling significantly improves the cost-accuracy trade-off over the
state-of-the-art downsampling. Token Pooling is a simple and effective operator
that can benefit many architectures. Applied to DeiT, it achieves the same
ImageNet top-1 accuracy using 42% fewer computations.

    

### [[2110.03861] QTN-VQC: An End-to-End Learning framework for Quantum Neural Networks](http://arxiv.org/abs/2110.03861)


  The advent of noisy intermediate-scale quantum (NISQ) computers raises a
crucial challenge to design quantum neural networks for fully quantum learning
tasks. To bridge the gap, this work proposes an end-to-end learning framework
named QTN-VQC, by introducing a trainable quantum tensor network (QTN) for
quantum embedding on a variational quantum circuit (VQC). The architecture of
QTN is composed of a parametric tensor-train network for feature extraction and
a tensor product encoding for quantum encoding. We highlight the QTN for
quantum embedding in terms of two perspectives: (1) we theoretically
characterize QTN by analyzing its representation power of input features; (2)
QTN enables an end-to-end parametric model pipeline, namely QTN-VQC, from the
generation of quantum embedding to the output measurement. Our experiments on
the MNIST dataset demonstrate the advantages of QTN for quantum embedding over
other quantum embedding approaches.

    

### [[2110.03865] Stable Prediction on Graphs with Agnostic Distribution Shift](http://arxiv.org/abs/2110.03865)


  Graph is a flexible and effective tool to represent complex structures in
practice and graph neural networks (GNNs) have been shown to be effective on
various graph tasks with randomly separated training and testing data. In real
applications, however, the distribution of training graph might be different
from that of the test one (e.g., users' interactions on the user-item training
graph and their actual preference on items, i.e., testing environment, are
known to have inconsistencies in recommender systems). Moreover, the
distribution of test data is always agnostic when GNNs are trained. Hence, we
are facing the agnostic distribution shift between training and testing on
graph learning, which would lead to unstable inference of traditional GNNs
across different test environments. To address this problem, we propose a novel
stable prediction framework for GNNs, which permits both locally and globally
stable learning and prediction on graphs. In particular, since each node is
partially represented by its neighbors in GNNs, we propose to capture the
stable properties for each node (locally stable) by re-weighting the
information propagation/aggregation processes. For global stability, we propose
a stable regularizer that reduces the training losses on heterogeneous
environments and thus warping the GNNs to generalize well. We conduct extensive
experiments on several graph benchmarks and a noisy industrial recommendation
dataset that is collected from 5 consecutive days during a product promotion
festival. The results demonstrate that our method outperforms various SOTA GNNs
for stable prediction on graphs with agnostic distribution shift, including
shift caused by node labels and attributes.

    

### [[2110.03868] Contrastive Learning for Source Code with Structural and Functional Properties](http://arxiv.org/abs/2110.03868)


  Pre-trained transformer models have recently shown promises for understanding
the source code. Most existing works expect to understand code from the textual
features and limited structural knowledge of code. However, the program
functionalities sometimes cannot be fully revealed by the code sequence, even
with structure information. Programs can contain very different tokens and
structures while sharing the same functionality, but changing only one or a few
code tokens can introduce unexpected or malicious program behaviors while
preserving the syntax and most tokens. In this work, we present BOOST, a novel
self-supervised model to focus pre-training based on the characteristics of
source code. We first employ automated, structure-guided code transformation
algorithms that generate (i.) functionally equivalent code that looks
drastically different from the original one, and (ii.) textually and
syntactically very similar code that is functionally distinct from the
original. We train our model in a way that brings the functionally equivalent
code closer and distinct code further through a contrastive learning objective.
To encode the structure information, we introduce a new node-type masked
language model objective that helps the model learn about structural context.
We pre-train BOOST with a much smaller dataset than the state-of-the-art
models, but our small models can still match or outperform these large models
in code understanding and generation tasks.

    

### [[2110.03873] Representation of professions in entertainment media: Insights into frequency and sentiment trends through computational text analysis](http://arxiv.org/abs/2110.03873)


  Societal ideas and trends dictate media narratives and cinematic depictions
which in turn influences people's beliefs and perceptions of the real world.
Media portrayal of culture, education, government, religion, and family affect
their function and evolution over time as people interpret and perceive these
representations and incorporate them into their beliefs and actions. It is
important to study media depictions of these social structures so that they do
not propagate or reinforce negative stereotypes, or discriminate against any
demographic section. In this work, we examine media representation of
professions and provide computational insights into their incidence, and
sentiment expressed, in entertainment media content. We create a searchable
taxonomy of professional groups and titles to facilitate their retrieval from
speaker-agnostic text passages like movie and television (TV) show subtitles.
We leverage this taxonomy and relevant natural language processing (NLP) models
to create a corpus of professional mentions in media content, spanning more
than 136,000 IMDb titles over seven decades (1950-2017). We analyze the
frequency and sentiment trends of different occupations, study the effect of
media attributes like genre, country of production, and title type on these
trends, and investigate if the incidence of professions in media subtitles
correlate with their real-world employment statistics. We observe increased
media mentions of STEM, arts, sports, and entertainment occupations in the
analyzed subtitles, and a decreased frequency of manual labor jobs and military
occupations. The sentiment expressed toward lawyers, police, and doctors is
becoming negative over time, whereas astronauts, musicians, singers, and
engineers are mentioned favorably. Professions that employ more people have
increased media frequency, supporting our hypothesis that media acts as a
mirror to society.

    

### [[2110.03879] Explaining the Attention Mechanism of End-to-End Speech Recognition Using Decision Trees](http://arxiv.org/abs/2110.03879)


  The attention mechanism has largely improved the performance of end-to-end
speech recognition systems. However, the underlying behaviours of attention is
not yet clearer. In this study, we use decision trees to explain how the
attention mechanism impact itself in speech recognition. The results indicate
that attention levels are largely impacted by their previous states rather than
the encoder and decoder patterns. Additionally, the default attention mechanism
seems to put more weights on closer states, but behaves poorly on modelling
long-term dependencies of attention states.

    

### [[2110.03882] ModeRNN: Harnessing Spatiotemporal Mode Collapse in Unsupervised Predictive Learning](http://arxiv.org/abs/2110.03882)


  Learning predictive models for unlabeled spatiotemporal data is challenging
in part because visual dynamics can be highly entangled in real scenes, making
existing approaches prone to overfit partial modes of physical processes while
neglecting to reason about others. We name this phenomenon spatiotemporal mode
collapse and explore it for the first time in predictive learning. The key is
to provide the model with a strong inductive bias to discover the compositional
structures of latent modes. To this end, we propose ModeRNN, which introduces a
novel method to learn structured hidden representations between recurrent
states. The core idea of this framework is to first extract various components
of visual dynamics using a set of spatiotemporal slots with independent
parameters. Considering that multiple space-time patterns may co-exist in a
sequence, we leverage learnable importance weights to adaptively aggregate slot
features into a unified hidden representation, which is then used to update the
recurrent states. Across the entire dataset, different modes result in
different responses on the mixtures of slots, which enhances the ability of
ModeRNN to build structured representations and thus prevents the so-called
mode collapse. Unlike existing models, ModeRNN is shown to prevent
spatiotemporal mode collapse and further benefit from learning mixed visual
dynamics.

    

### [[2110.03888] M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion Parameter Pretraining](http://arxiv.org/abs/2110.03888)


  Recent expeditious developments in deep learning algorithms, distributed
training, and even hardware design for large models have enabled training
extreme-scale models, say GPT-3 and Switch Transformer possessing hundreds of
billions or even trillions of parameters. However, under limited resources,
extreme-scale model training that requires enormous amounts of computes and
memory footprint suffers from frustratingly low efficiency in model
convergence. In this paper, we propose a simple training strategy called
"Pseudo-to-Real" for high-memory-footprint-required large models.
Pseudo-to-Real is compatible with large models with architecture of sequential
layers. We demonstrate a practice of pretraining unprecedented
10-trillion-parameter model, an order of magnitude larger than the
state-of-the-art, on solely 512 GPUs within 10 days. Besides demonstrating the
application of Pseudo-to-Real, we also provide a technique, Granular CPU
offloading, to manage CPU memory for training large model and maintain high GPU
utilities. Fast training of extreme-scale models on a decent amount of
resources can bring much smaller carbon footprint and contribute to greener AI.

    

### [[2110.03891] Momentum Doesn't Change the Implicit Bias](http://arxiv.org/abs/2110.03891)


  The momentum acceleration technique is widely adopted in many optimization
algorithms. However, the theoretical understanding of how the momentum affects
the generalization performance of the optimization algorithms is still unknown.
In this paper, we answer this question through analyzing the implicit bias of
momentum-based optimization. We prove that both SGD with momentum and Adam
converge to the $L_2$ max-margin solution for exponential-tailed loss, which is
the same as vanilla gradient descent. That means, these optimizers with
momentum acceleration still converge to a model with low complexity, which
provides guarantees on their generalization. Technically, to overcome the
difficulty brought by the error accumulation in analyzing the momentum, we
construct new Lyapunov functions as a tool to analyze the gap between the model
parameter and the max-margin solution.

    

### [[2110.03894] A Study of Low-Resource Speech Commands Recognition based on Adversarial Reprogramming](http://arxiv.org/abs/2110.03894)


  In this study, we propose a novel adversarial reprogramming (AR) approach for
low-resource spoken command recognition (SCR), and build an AR-SCR system. The
AR procedure aims to modify the acoustic signals (from the target domain) to
repurpose a pretrained SCR model (from the source domain). To solve the label
mismatches between source and target domains, and further improve the stability
of AR, we propose a novel similarity-based label mapping technique to align
classes. In addition, the transfer learning (TL) technique is combined with the
original AR process to improve the model adaptation capability. We evaluate the
proposed AR-SCR system on three low-resource SCR datasets, including Arabic,
Lithuanian, and dysarthric Mandarin speech. Experimental results show that with
a pretrained AM trained on a large-scale English dataset, the proposed AR-SCR
system outperforms the current state-of-the-art results on Arabic and
Lithuanian speech commands datasets, with only a limited amount of training
data.

    

### [[2110.03898] Differentiable Programming of Isometric Tensor Networks](http://arxiv.org/abs/2110.03898)


  Differentiable programming is a new programming paradigm which enables large
scale optimization through automatic calculation of gradients also known as
auto-differentiation. This concept emerges from deep learning, and has also
been generalized to tensor network optimizations. Here, we extend the
differentiable programming to tensor networks with isometric constraints with
applications to multiscale entanglement renormalization ansatz (MERA) and
tensor network renormalization (TNR). By introducing several gradient-based
optimization methods for the isometric tensor network and comparing with
Evenbly-Vidal method, we show that auto-differentiation has a better
performance for both stability and accuracy. We numerically tested our methods
on 1D critical quantum Ising spin chain and 2D classical Ising model. We
calculate the ground state energy for the 1D quantum model and internal energy
for the classical model, and scaling dimensions of scaling operators and find
they all agree with the theory well.

    

### [[2110.03903] Kinematically consistent recurrent neural networks for learning inverse problems in wave propagation](http://arxiv.org/abs/2110.03903)


  Although machine learning (ML) is increasingly employed recently for
mechanistic problems, the black-box nature of conventional ML architectures
lacks the physical knowledge to infer unforeseen input conditions. This implies
both severe overfitting during a dearth of training data and inadequate
physical interpretability, which motivates us to propose a new kinematically
consistent, physics-based ML model. In particular, we attempt to perform
physically interpretable learning of inverse problems in wave propagation
without suffering overfitting restrictions. Towards this goal, we employ long
short-term memory (LSTM) networks endowed with a physical,
hyperparameter-driven regularizer, performing penalty-based enforcement of the
characteristic geometries. Since these characteristics are the kinematical
invariances of wave propagation phenomena, maintaining their structure provides
kinematical consistency to the network. Even with modest training data, the
kinematically consistent network can reduce the $L_1$ and $L_\infty$ error
norms of the plain LSTM predictions by about 45% and 55%, respectively. It can
also increase the horizon of the plain LSTM's forecasting by almost two times.
To achieve this, an optimal range of the physical hyperparameter, analogous to
an artificial bulk modulus, has been established through numerical experiments.
The efficacy of the proposed method in alleviating overfitting, and the
physical interpretability of the learning mechanism, are also discussed. Such
an application of kinematically consistent LSTM networks for wave propagation
learning is presented here for the first time.

    

### [[2110.03905] COVID-19 Monitoring System using Social Distancing and Face Mask Detection on Surveillance video datasets](http://arxiv.org/abs/2110.03905)


  In the current times, the fear and danger of COVID-19 virus still stands
large. Manual monitoring of social distancing norms is impractical with a large
population moving about and with insufficient task force and resources to
administer them. There is a need for a lightweight, robust and 24X7
video-monitoring system that automates this process. This paper proposes a
comprehensive and effective solution to perform person detection, social
distancing violation detection, face detection and face mask classification
using object detection, clustering and Convolution Neural Network (CNN) based
binary classifier. For this, YOLOv3, Density-based spatial clustering of
applications with noise (DBSCAN), Dual Shot Face Detector (DSFD) and
MobileNetV2 based binary classifier have been employed on surveillance video
datasets. This paper also provides a comparative study of different face
detection and face mask classification models. Finally, a video dataset
labelling method is proposed along with the labelled video dataset to
compensate for the lack of dataset in the community and is used for evaluation
of the system. The system performance is evaluated in terms of accuracy, F1
score as well as the prediction time, which has to be low for practical
applicability. The system performs with an accuracy of 91.2% and F1 score of
90.79% on the labelled video dataset and has an average prediction time of 7.12
seconds for 78 frames of a video.

    

### [[2110.03906] Nash Convergence of Mean-Based Learning Algorithms in First Price Auctions](http://arxiv.org/abs/2110.03906)


  We consider repeated first price auctions where each bidder, having a
deterministic type, learns to bid using a mean-based learning algorithm. We
completely characterize the Nash convergence property of the bidding dynamics
in two senses: (1) time-average: the fraction of rounds where bidders play a
Nash equilibrium approaches to 1 in the limit; (2) last-iterate: the mixed
strategy profile of bidders approaches to a Nash equilibrium in the limit.
Specifically, the results depend on the number of bidders with the highest
value: - If the number is at least three, the bidding dynamics almost surely
converges to a Nash equilibrium of the auction, both in time-average and in
last-iterate. - If the number is two, the bidding dynamics almost surely
converges to a Nash equilibrium in time-average but not necessarily in
last-iterate. - If the number is one, the bidding dynamics may not converge to
a Nash equilibrium in time-average nor in last-iterate. Our discovery opens up
new possibilities in the study of convergence dynamics of learning algorithms.

    

### [[2110.03909] Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning](http://arxiv.org/abs/2110.03909)


  In few-shot learning scenarios, the challenge is to generalize and perform
well on new unseen examples when only very few labeled examples are available
for each task. Model-agnostic meta-learning (MAML) has gained the popularity as
one of the representative few-shot learning methods for its flexibility and
applicability to diverse problems. However, MAML and its variants often resort
to a simple loss function without any auxiliary loss function or regularization
terms that can help achieve better generalization. The problem lies in that
each application and task may require different auxiliary loss function,
especially when tasks are diverse and distinct. Instead of attempting to
hand-design an auxiliary loss function for each application and task, we
introduce a new meta-learning framework with a loss function that adapts to
each task. Our proposed framework, named Meta-Learning with Task-Adaptive Loss
Function (MeTAL), demonstrates the effectiveness and the flexibility across
various domains, such as few-shot classification and few-shot regression.

    

### [[2110.03921] ViDT: An Efficient and Effective Fully Transformer-based Object Detector](http://arxiv.org/abs/2110.03921)


  Transformers are transforming the landscape of computer vision, especially
for recognition tasks. Detection transformers are the first fully end-to-end
learning systems for object detection, while vision transformers are the first
fully transformer-based architecture for image classification. In this paper,
we integrate Vision and Detection Transformers (ViDT) to build an effective and
efficient object detector. ViDT introduces a reconfigured attention module to
extend the recent Swin Transformer to be a standalone object detector, followed
by a computationally efficient transformer decoder that exploits multi-scale
features and auxiliary techniques essential to boost the detection performance
without much increase in computational load. Extensive evaluation results on
the Microsoft COCO benchmark dataset demonstrate that ViDT obtains the best AP
and latency trade-off among existing fully transformer-based object detectors,
and achieves 49.2AP owing to its high scalability for large models. We will
release the code and trained models athttps://github.com/naver-ai/vidt

    

### [[2110.03922] Neural Tangent Kernel Eigenvalues Accurately Predict Generalization](http://arxiv.org/abs/2110.03922)


  Finding a quantitative theory of neural network generalization has long been
a central goal of deep learning research. We extend recent results to
demonstrate that, by examining the eigensystem of a neural network's "neural
tangent kernel", one can predict its generalization performance when learning
arbitrary functions. Our theory accurately predicts not only test
mean-squared-error but all first- and second-order statistics of the network's
learned function. Furthermore, using a measure quantifying the "learnability"
of a given target function, we prove a new "no-free-lunch" theorem
characterizing a fundamental tradeoff in the inductive bias of wide neural
networks: improving a network's generalization for a given target function must
worsen its generalization for orthogonal functions. We further demonstrate the
utility of our theory by analytically predicting two surprising phenomena -
worse-than-chance generalization on hard-to-learn functions and nonmonotonic
error curves in the small data regime - which we subsequently observe in
experiments. Though our theory is derived for infinite-width architectures, we
find it agrees with networks as narrow as width 20, suggesting it is predictive
of generalization in practical neural networks. Code replicating our results is
available at this https URL .

    

### [[2110.03923] Opportunities for Machine Learning to Accelerate Halide Perovskite Commercialization and Scale-Up](http://arxiv.org/abs/2110.03923)


  While halide perovskites attract significant academic attention, examples of
at-scale industrial production are still sparse. In this perspective, we review
practical challenges hindering the commercialization of halide perovskites, and
discuss how machine-learning (ML) tools could help: (1) active-learning
algorithms that blend institutional knowledge and human expertise could help
stabilize and rapidly update baseline manufacturing processes; (2) ML-powered
metrology, including computer imaging, could help narrow the performance gap
between large- and small-area devices; and (3) inference methods could help
accelerate root-cause analysis by reconciling multiple data streams and
simulations, focusing research effort on areas with highest probability for
improvement. We conclude that to satisfy many of these challenges, incremental
-- not radical -- adaptations of existing ML and statistical methods are
needed. We identify resources to help develop in-house data-science talent, and
propose how industry-academic partnerships could help adapt "ready-now" ML
tools to specific industry needs, further improve process control by revealing
underlying mechanisms, and develop "gamechanger" discovery-oriented algorithms
to better navigate vast materials combination spaces and the literature.

    

### [[2110.03945] Anomaly Detection in Beehives: An Algorithm Comparison](http://arxiv.org/abs/2110.03945)


  Sensor-equipped beehives allow monitoring the living conditions of bees.
Machine learning models can use the data of such hives to learn behavioral
patterns and find anomalous events. One type of event that is of particular
interest to apiarists for economical reasons is bee swarming. Other events of
interest are behavioral anomalies from illness and technical anomalies, e.g.
sensor failure. Beekeepers can be supported by suitable machine learning models
which can detect these events. In this paper we compare multiple machine
learning models for anomaly detection and evaluate them for their applicability
in the context of beehives. Namely we employed Deep Recurrent Autoencoder,
Elliptic Envelope, Isolation Forest, Local Outlier Factor and One-Class SVM.
Through evaluation with real world datasets of different hives and with
different sensor setups we find that the autoencoder is the best multi-purpose
anomaly detector in comparison.

    

### [[2110.03950] Nonconvex-Nonconcave Min-Max Optimization with a Small Maximization Domain](http://arxiv.org/abs/2110.03950)


  We study the problem of finding approximate first-order stationary points in
optimization problems of the form $\min_{x \in X} \max_{y \in Y} f(x,y)$, where
the sets $X,Y$ are convex and $Y$ is compact. The objective function $f$ is
smooth, but assumed neither convex in $x$ nor concave in $y$. Our approach
relies upon replacing the function $f(x,\cdot)$ with its $k$th order Taylor
approximation (in $y$) and finding a near-stationary point in the resulting
surrogate problem. To guarantee its success, we establish the following result:
let the Euclidean diameter of $Y$ be small in terms of the target accuracy
$\varepsilon$, namely $O(\varepsilon^{\frac{2}{k+1}})$ for $k \in \mathbb{N}$
and $O(\varepsilon)$ for $k = 0$, with the constant factors controlled by
certain regularity parameters of $f$; then any $\varepsilon$-stationary point
in the surrogate problem remains $O(\varepsilon)$-stationary for the initial
problem. Moreover, we show that these upper bounds are nearly optimal: the
aforementioned reduction provably fails when the diameter of $Y$ is larger. For
$0 \le k \le 2$ the surrogate function can be efficiently maximized in $y$; our
general approximation result then leads to efficient algorithms for finding a
near-stationary point in nonconvex-nonconcave min-max problems, for which we
also provide convergence guarantees.

    

### [[2110.03960] Mixability made efficient: Fast online multiclass logistic regression](http://arxiv.org/abs/2110.03960)


  Mixability has been shown to be a powerful tool to obtain algorithms with
optimal regret. However, the resulting methods often suffer from high
computational complexity which has reduced their practical applicability. For
example, in the case of multiclass logistic regression, the aggregating
forecaster (Foster et al. (2018)) achieves a regret of $O(\log(Bn))$ whereas
Online Newton Step achieves $O(e^B\log(n))$ obtaining a double exponential gain
in $B$ (a bound on the norm of comparative functions). However, this high
statistical performance is at the price of a prohibitive computational
complexity $O(n^{37})$.

    

### [[2110.03975] Tensor train completion: local recovery guarantees via Riemannian optimization](http://arxiv.org/abs/2110.03975)


  In this work we estimate the number of randomly selected elements of a tensor
that with high probability guarantees local convergence of Riemannian gradient
descent for tensor train completion. We derive a new bound for the orthogonal
projections onto the tangent spaces based on the harmonic mean of the
unfoldings' singular values and introduce a notion of core coherence for tensor
trains. We also extend the results to tensor train completion with side
information and obtain the corresponding local convergence guarantees.

    

### [[2110.03979] MilliTRACE-IR: Contact Tracing and Temperature Screening via mm-Wave and Infrared Sensing](http://arxiv.org/abs/2110.03979)


  In this work, we present milliTRACE-IR, a joint mm-wave radar and infrared
imaging sensing system performing unobtrusive and privacy preserving human body
temperature screening and contact tracing in indoor spaces. Social distancing
and fever detection have been widely employed to counteract the COVID-19
pandemic, sparking great interest from academia, industry and public
administrations worldwide. While most solutions have dealt with the two aspects
separately, milliTRACE-IR combines, via a robust sensor fusion approach,
mm-wave radars and infrared thermal cameras. The system achieves fully
automated measurement of distancing and body temperature, by jointly tracking
the faces of the subjects in the thermal camera image plane and the human
motion in the radar reference system. It achieves decimeter-level accuracy in
distance estimation, inter-personal distance estimation (effective for subjects
getting as close as 0.2 m), and accurate temperature monitoring (max. errors of
0.5 C). Moreover, milliTRACE-IR performs contact tracing: a person with high
body temperature is reliably detected by the thermal camera sensor and
subsequently traced across a large indoor area in a non-invasive way by the
radars. When entering a new room, this subject is re-identified among several
other individuals with high accuracy (95%), by computing gait-related features
from the radar reflections through a deep neural network and using a weighted
extreme learning machine as the final re-identification tool.

    

### [[2110.03991] Combining Differential Privacy and Byzantine Resilience in Distributed SGD](http://arxiv.org/abs/2110.03991)


  Privacy and Byzantine resilience (BR) are two crucial requirements of
modern-day distributed machine learning. The two concepts have been extensively
studied individually but the question of how to combine them effectively
remains unanswered. This paper contributes to addressing this question by
studying the extent to which the distributed SGD algorithm, in the standard
parameter-server architecture, can learn an accurate model despite (a) a
fraction of the workers being malicious (Byzantine), and (b) the other
fraction, whilst being honest, providing noisy information to the server to
ensure differential privacy (DP). We first observe that the integration of
standard practices in DP and BR is not straightforward. In fact, we show that
many existing results on the convergence of distributed SGD under Byzantine
faults, especially those relying on $(\alpha,f)$-Byzantine resilience, are
rendered invalid when honest workers enforce DP. To circumvent this
shortcoming, we revisit the theory of $(\alpha,f)$-BR to obtain an approximate
convergence guarantee. Our analysis provides key insights on how to improve
this guarantee through hyperparameter optimization. Essentially, our
theoretical and empirical results show that (1) an imprudent combination of
standard approaches to DP and BR might be fruitless, but (2) by carefully
re-tuning the learning algorithm, we can obtain reasonable learning accuracy
while simultaneously guaranteeing DP and BR.

    

### [[2110.03995] Statistical Regeneration Guarantees of the Wasserstein Autoencoder with Latent Space Consistency](http://arxiv.org/abs/2110.03995)


  The introduction of Variational Autoencoders (VAE) has been marked as a
breakthrough in the history of representation learning models. Besides having
several accolades of its own, VAE has successfully flagged off a series of
inventions in the form of its immediate successors. Wasserstein Autoencoder
(WAE), being an heir to that realm carries with it all of the goodness and
heightened generative promises, matching even the generative adversarial
networks (GANs). Needless to say, recent years have witnessed a remarkable
resurgence in statistical analyses of the GANs. Similar examinations for
Autoencoders, however, despite their diverse applicability and notable
empirical performance, remain largely absent. To close this gap, in this paper,
we investigate the statistical properties of WAE. Firstly, we provide
statistical guarantees that WAE achieves the target distribution in the latent
space, utilizing the Vapnik Chervonenkis (VC) theory. The main result,
consequently ensures the regeneration of the input distribution, harnessing the
potential offered by Optimal Transport of measures under the Wasserstein
metric. This study, in turn, hints at the class of distributions WAE can
reconstruct after suffering a compression in the form of a latent law.

    

### [[2110.03999] Graphs as Tools to Improve Deep Learning Methods](http://arxiv.org/abs/2110.03999)


  In recent years, deep neural networks (DNNs) have known an important rise in
popularity. However, although they are state-of-the-art in many machine
learning challenges, they still suffer from several limitations. For example,
DNNs require a lot of training data, which might not be available in some
practical applications. In addition, when small perturbations are added to the
inputs, DNNs are prone to misclassification errors. DNNs are also viewed as
black-boxes and as such their decisions are often criticized for their lack of
interpretability.
In this chapter, we review recent works that aim at using graphs as tools to
improve deep learning methods. These graphs are defined considering a specific
layer in a deep learning architecture. Their vertices represent distinct
samples, and their edges depend on the similarity of the corresponding
intermediate representations. These graphs can then be leveraged using various
methodologies, many of which built on top of graph signal processing.
This chapter is composed of four main parts: tools for visualizing
intermediate layers in a DNN, denoising data representations, optimizing graph
objective functions and regularizing the learning process.

    

### [[2110.04003] Learning to Centralize Dual-Arm Assembly](http://arxiv.org/abs/2110.04003)


  Even though industrial manipulators are widely used in modern manufacturing
processes, deployment in unstructured environments remains an open problem. To
deal with variety, complexity and uncertainty of real world manipulation tasks
a general framework is essential. In this work we want to focus on assembly
with humanoid robots by providing a framework for dual-arm peg-in-hole
manipulation. As we aim to contribute towards an approach which is not limited
to dual-arm peg-in-hole, but dual-arm manipulation in general, we keep modeling
effort at a minimum. While reinforcement learning has shown great results for
single-arm robotic manipulation in recent years, research focusing on dual-arm
manipulation is still rare. Solving such tasks often involves complex modeling
of interaction between two manipulators and their coupling at a control level.
In this paper, we explore the applicability of model-free reinforcement
learning to dual-arm manipulation based on a modular approach with two
decentralized single-arm controllers and a single centralized policy. We reduce
modeling effort to a minimum by using sparse rewards only. We demonstrate the
effectiveness of the framework on dual-arm peg-in-hole and analyze sample
efficiency and success rates for different action spaces. Moreover, we compare
results on different clearances and showcase disturbance recovery and
robustness, when dealing with position uncertainties. Finally we zero-shot
transfer policies trained in simulation to the real-world and evaluate their
performance.

    

### [[2110.04005] KaraSinger: Score-Free Singing Voice Synthesis with VQ-VAE using Mel-spectrograms](http://arxiv.org/abs/2110.04005)


  In this paper, we propose a novel neural network model called KaraSinger for
a less-studied singing voice synthesis (SVS) task named score-free SVS, in
which the prosody and melody are spontaneously decided by machine. KaraSinger
comprises a vector-quantized variational autoencoder (VQ-VAE) that compresses
the Mel-spectrograms of singing audio to sequences of discrete codes, and a
language model (LM) that learns to predict the discrete codes given the
corresponding lyrics. For the VQ-VAE part, we employ a Connectionist Temporal
Classification (CTC) loss to encourage the discrete codes to carry
phoneme-related information. For the LM part, we use location-sensitive
attention for learning a robust alignment between the input phoneme sequence
and the output discrete code. We keep the architecture of both the VQ-VAE and
LM light-weight for fast training and inference speed. We validate the
effectiveness of the proposed design choices using a proprietary collection of
550 English pop songs sung by multiple amateur singers. The result of a
listening test shows that KaraSinger achieves high scores in intelligibility,
musicality, and the overall quality.

    

### [[2110.04020] Pathologies in priors and inference for Bayesian transformers](http://arxiv.org/abs/2110.04020)


  In recent years, the transformer has established itself as a workhorse in
many applications ranging from natural language processing to reinforcement
learning. Similarly, Bayesian deep learning has become the gold-standard for
uncertainty estimation in safety-critical applications, where robustness and
calibration are crucial. Surprisingly, no successful attempts to improve
transformer models in terms of predictive uncertainty using Bayesian inference
exist. In this work, we study this curiously underpopulated area of Bayesian
transformers. We find that weight-space inference in transformers does not work
well, regardless of the approximate posterior. We also find that the prior is
at least partially at fault, but that it is very hard to find well-specified
weight priors for these models. We hypothesize that these problems stem from
the complexity of obtaining a meaningful mapping from weight-space to
function-space distributions in the transformer. Therefore, moving closer to
function-space, we propose a novel method based on the implicit
reparameterization of the Dirichlet distribution to apply variational inference
directly to the attention weights. We find that this proposed method performs
competitively with our baselines.

    

### [[2110.04022] Learning Sparse Graphs with a Core-periphery Structure](http://arxiv.org/abs/2110.04022)


  In this paper, we focus on learning sparse graphs with a core-periphery
structure. We propose a generative model for data associated with
core-periphery structured networks to model the dependence of node attributes
on core scores of the nodes of a graph through a latent graph structure. Using
the proposed model, we jointly infer a sparse graph and nodal core scores that
induce dense (sparse) connections in core (respectively, peripheral) parts of
the network. Numerical experiments on a variety of real-world data indicate
that the proposed method learns a core-periphery structured graph from node
attributes alone, while simultaneously learning core score assignments that
agree well with existing works that estimate core scores using graph as input
and ignoring commonly available node attributes.

    

### [[2110.04038] Traffic Flow Forecasting with Spatial-Temporal Graph Diffusion Network](http://arxiv.org/abs/2110.04038)


  Accurate forecasting of citywide traffic flow has been playing critical role
in a variety of spatial-temporal mining applications, such as intelligent
traffic control and public risk assessment. While previous work has made
significant efforts to learn traffic temporal dynamics and spatial
dependencies, two key limitations exist in current models. First, only the
neighboring spatial correlations among adjacent regions are considered in most
existing methods, and the global inter-region dependency is ignored.
Additionally, these methods fail to encode the complex traffic transition
regularities exhibited with time-dependent and multi-resolution in nature. To
tackle these challenges, we develop a new traffic prediction
framework-Spatial-Temporal Graph Diffusion Network (ST-GDN). In particular,
ST-GDN is a hierarchically structured graph neural architecture which learns
not only the local region-wise geographical dependencies, but also the spatial
semantics from a global perspective. Furthermore, a multi-scale attention
network is developed to empower ST-GDN with the capability of capturing
multi-level temporal dynamics. Experiments on several real-life traffic
datasets demonstrate that ST-GDN outperforms different types of
state-of-the-art baselines. Source codes of implementations are available at
this https URL.

    

### [[2110.04039] Global Context Enhanced Social Recommendation with Hierarchical Graph Neural Networks](http://arxiv.org/abs/2110.04039)


  Social recommendation which aims to leverage social connections among users
to enhance the recommendation performance. With the revival of deep learning
techniques, many efforts have been devoted to developing various neural
network-based social recommender systems, such as attention mechanisms and
graph-based message passing frameworks. However, two important challenges have
not been well addressed yet: (i) Most of existing social recommendation models
fail to fully explore the multi-type user-item interactive behavior as well as
the underlying cross-relational inter-dependencies. (ii) While the learned
social state vector is able to model pair-wise user dependencies, it still has
limited representation capacity in capturing the global social context across
users. To tackle these limitations, we propose a new Social Recommendation
framework with Hierarchical Graph Neural Networks (SR-HGNN). In particular, we
first design a relation-aware reconstructed graph neural network to inject the
cross-type collaborative semantics into the recommendation framework. In
addition, we further augment SR-HGNN with a social relation encoder based on
the mutual information learning paradigm between low-level user embeddings and
high-level global representation, which endows SR-HGNN with the capability of
capturing the global social contextual signals. Empirical results on three
public benchmarks demonstrate that SR-HGNN significantly outperforms
state-of-the-art recommendation methods. Source codes are available at:
this https URL.

    

### [[2110.04044] Subspace Change-Point Detection via Low-Rank Matrix Factorisation](http://arxiv.org/abs/2110.04044)


  Multivariate time series can often have a large number of dimensions, whether
it is due to the vast amount of collected features or due to how the data
sources are processed. Frequently, the main structure of the high-dimensional
time series can be well represented by a lower dimensional subspace. As vast
quantities of data are being collected over long periods of time, it is
reasonable to assume that the underlying subspace structure would change over
time. In this work, we propose a change-point detection method based on
low-rank matrix factorisation that can detect multiple changes in the
underlying subspace of a multivariate time series. Experimental results on both
synthetic and real data sets demonstrate the effectiveness of our approach and
its advantages against various state-of-the-art methods.

    

### [[2110.04049] Minimal-Configuration Anomaly Detection for IIoT Sensors](http://arxiv.org/abs/2110.04049)


  The increasing deployment of low-cost IoT sensor platforms in industry boosts
the demand for anomaly detection solutions that fulfill two key requirements:
minimal configuration effort and easy transferability across equipment. Recent
advances in deep learning, especially long-short-term memory (LSTM) and
autoencoders, offer promising methods for detecting anomalies in sensor data
recordings. We compared autoencoders with various architectures such as deep
neural networks (DNN), LSTMs and convolutional neural networks (CNN) using a
simple benchmark dataset, which we generated by operating a peristaltic pump
under various operating conditions and inducing anomalies manually. Our
preliminary results indicate that a single model can detect anomalies under
various operating conditions on a four-dimensional data set without any
specific feature engineering for each operating condition. We consider this
work as being the first step towards a generic anomaly detection method, which
is applicable for a wide range of industrial equipment.

    

### [[2110.04056] Improving Pseudo-label Training For End-to-end Speech Recognition Using Gradient Mask](http://arxiv.org/abs/2110.04056)


  In the recent trend of semi-supervised speech recognition, both
self-supervised representation learning and pseudo-labeling have shown
promising results. In this paper, we propose a novel approach to combine their
ideas for end-to-end speech recognition model. Without any extra loss function,
we utilize the Gradient Mask to optimize the model when training on
pseudo-label. This method forces the speech recognition model to predict from
the masked input to learn strong acoustic representation and make training
robust to label noise. In our semi-supervised experiments, the method can
improve the model performance when training on pseudo-label and our method
achieved competitive results comparing with other semi-supervised approaches on
the Librispeech 100 hours experiments.

    

### [[2110.04057] FAST-RIR: Fast neural diffuse room impulse response generator](http://arxiv.org/abs/2110.04057)


  We present a neural-network-based fast diffuse room impulse response
generator (FAST-RIR) for generating room impulse responses (RIRs) for a given
acoustic environment. Our FAST-RIR takes rectangular room dimensions, listener
and speaker positions, and reverberation time as inputs and generates specular
and diffuse reflections for a given acoustic environment. Our FAST-RIR is
capable of generating RIRs for a given input reverberation time with an average
error of 0.02s. We evaluate our generated RIRs in automatic speech recognition
(ASR) applications using Google Speech API, Microsoft Speech API, and Kaldi
tools. We show that our proposed FAST-RIR with batch size 1 is 400 times faster
than a state-of-the-art diffuse acoustic simulator (DAS) on a CPU and gives
similar performance to DAS in ASR experiments. Our FAST-RIR is 12 times faster
than an existing GPU-based RIR generator (gpuRIR). We show that our FAST-RIR
outperforms gpuRIR by 2.5% in an AMI far-field ASR benchmark.

    

### [[2110.04060] New Insights into Graph Convolutional Networks using Neural Tangent Kernels](http://arxiv.org/abs/2110.04060)


  Graph Convolutional Networks (GCNs) have emerged as powerful tools for
learning on network structured data. Although empirically successful, GCNs
exhibit certain behaviour that has no rigorous explanation -- for instance, the
performance of GCNs significantly degrades with increasing network depth,
whereas it improves marginally with depth using skip connections. This paper
focuses on semi-supervised learning on graphs, and explains the above
observations through the lens of Neural Tangent Kernels (NTKs). We derive NTKs
corresponding to infinitely wide GCNs (with and without skip connections).
Subsequently, we use the derived NTKs to identify that, with suitable
normalisation, network depth does not always drastically reduce the performance
of GCNs -- a fact that we also validate through extensive simulation.
Furthermore, we propose NTK as an efficient `surrogate model' for GCNs that
does not suffer from performance fluctuations due to hyper-parameter tuning
since it is a hyper-parameter free deterministic kernel. The efficacy of this
idea is demonstrated through a comparison of different skip connections for
GCNs using the surrogate NTKs.

    

### [[2110.04063] A New Weakly Supervised Learning Approach for Real-time Iron Ore Feed Load Estimation](http://arxiv.org/abs/2110.04063)


  Iron ore feed load control is one of the most critical settings in a mineral
grinding process, directly impacting the quality of final products. The setting
of the feed load is mainly determined by the characteristics of the ore
pellets. However, the characterisation of ore is challenging to acquire in many
production environments, leading to poor feed load settings and inefficient
production processes. This paper presents our work using deep learning models
for direct ore feed load estimation from ore pellet images. To address the
challenges caused by the large size of a full ore pellets image and the
shortage of accurately annotated data, we treat the whole modelling process as
a weakly supervised learning problem. A two-stage model training algorithm and
two neural network architectures are proposed. The experiment results show
competitive model performance, and the trained models can be used for real-time
feed load estimation for grind process optimisation.

    

### [[2110.04069] BI-RADS-Net: An Explainable Multitask Learning Approach for Cancer Diagnosis in Breast Ultrasound Images](http://arxiv.org/abs/2110.04069)


  In healthcare, it is essential to explain the decision-making process of
machine learning models to establish the trustworthiness of clinicians. This
paper introduces BI-RADS-Net, a novel explainable deep learning approach for
cancer detection in breast ultrasound images. The proposed approach
incorporates tasks for explaining and classifying breast tumors, by learning
feature representations relevant to clinical diagnosis. Explanations of the
predictions (benign or malignant) are provided in terms of morphological
features that are used by clinicians for diagnosis and reporting in medical
practice. The employed features include the BI-RADS descriptors of shape,
orientation, margin, echo pattern, and posterior features. Additionally, our
approach predicts the likelihood of malignancy of the findings, which relates
to the BI-RADS assessment category reported by clinicians. Experimental
validation on a dataset consisting of 1,192 images indicates improved model
accuracy, supported by explanations in clinical terms using the BI-RADS
lexicon.

    

### [[2110.04070] Dataset Structural Index: Understanding a machine's perspective towards visual data](http://arxiv.org/abs/2110.04070)


  With advances in vision and perception architectures, we have realized that
working with data is equally crucial, if not more, than the algorithms. Till
today, we have trained machines based on our knowledge and perspective of the
world. The entire concept of Dataset Structural Index(DSI) revolves around
understanding a machine`s perspective of the dataset. With DSI, I show two meta
values with which we can get more information over a visual dataset and use it
to optimize data, create better architectures, and have an ability to guess
which model would work best. These two values are the Variety contribution
ratio and Similarity matrix. In the paper, I show many applications of DSI, one
of which is how the same level of accuracy can be achieved with the same model
architectures trained over less amount of data.

    

### [[2110.04071] Generative Pre-Trained Transformer for Cardiac Abnormality Detection](http://arxiv.org/abs/2110.04071)


  ECG heartbeat classification plays a vital role in diagnosis of cardiac
arrhythmia. The goal of the Physionet/CinC 2021 challenge was to accurately
classify clinical diagnosis based on 12, 6, 4, 3 or 2-lead ECG recordings in
order to aid doctors in the diagnoses of different heart conditions.
Transformers have had great success in the field of natural language processing
in the past years. Our team, CinCSEM, proposes to draw the parallel between
text and periodic time series signals by viewing the repeated period as words
and the whole signal as a sequence of such words. In this way, the attention
mechanisms of the transformers can be applied to periodic time series signals.
In our implementation, we follow the Transformer Encoder architecture, which
combines several encoder layers followed by a dense layer with linear or
sigmoid activation for generative pre-training or classification, respectively.
The use case presented here is multi-label classification of heartbeat
abnormalities of ECG recordings shared by the challenge. Our best entry, not
exceeding the challenge's hardware limitations, achieved a score of 0.12, 0.07,
0.10, 0.10 and 0.07 on 12-lead, 6-lead, 4-lead, 3-lead and 2-lead test set
respectively. Unfortunately, our team was unable to be ranked because of a
missing pre-print.

    

### [[2110.04074] Active inference, Bayesian optimal design, and expected utility](http://arxiv.org/abs/2110.04074)


  Active inference, a corollary of the free energy principle, is a formal way
of describing the behavior of certain kinds of random dynamical systems that
have the appearance of sentience. In this chapter, we describe how active
inference combines Bayesian decision theory and optimal Bayesian design
principles under a single imperative to minimize expected free energy. It is
this aspect of active inference that allows for the natural emergence of
information-seeking behavior. When removing prior outcomes preferences from
expected free energy, active inference reduces to optimal Bayesian design,
i.e., information gain maximization. Conversely, active inference reduces to
Bayesian decision theory in the absence of ambiguity and relative risk, i.e.,
expected utility maximization. Using these limiting cases, we illustrate how
behaviors differ when agents select actions that optimize expected utility,
expected information gain, and expected free energy. Our T-maze simulations
show optimizing expected free energy produces goal-directed information-seeking
behavior while optimizing expected utility induces purely exploitive behavior
and maximizing information gain engenders intrinsically motivated behavior.

    

### [[2110.04079] A Hybrid Spatial-temporal Sequence-to-one Neural Network Model for Lane Detection](http://arxiv.org/abs/2110.04079)


  Reliable and accurate lane detection is of vital importance for the safe
performance of Lane Keeping Assistance and Lane Departure Warning systems.
However, under certain challenging peculiar circumstances (e.g., marking
degradation, serious vehicle occlusion), it is difficult to get satisfactory
performance in accurately detecting the lane markings from one single image
which is often the case in current literature. Since road markings are
continuous lines on the road, the lanes that are difficult to be accurately
detected in the current image frame might potentially be better inferred out if
information from previous frames is incorporated. For this, we propose a novel
hybrid spatial-temporal sequence-to-one deep learning architecture making full
use of the spatial-temporal information in multiple frames of a continuous
sequence of images to detect lane markings in the very last current image
frame. Specifically, the hybrid model integrates the spatial convolutional
neural network (SCNN), which is powerful in extracting spatial features and
relationships in one single image, with convolutional long-short term memory
(ConvLSTM) neural network, which can capture the spatial-temporal correlations
and time dependencies among the image sequences. With the proposed model
architecture, the advantages of both SCNN and ConvLSTM are fully combined and
the spatial-temporal information is fully exploited. Treating lane detection as
the image segmentation problem, we applied encoder-decoder structures to make
it work in an end-to-end way. Extensive experiments on two large-scale datasets
reveal that our proposed model can effectively handle challenging driving
scenes and outperforms previous state-of-the-art methods.

    

### [[2110.04081] Flow Plugin Network for conditional generation](http://arxiv.org/abs/2110.04081)


  Generative models have gained many researchers' attention in the last years
resulting in models such as StyleGAN for human face generation or PointFlow for
the 3D point cloud generation. However, by default, we cannot control its
sampling process, i.e., we cannot generate a sample with a specific set of
attributes. The current approach is model retraining with additional inputs and
different architecture, which requires time and computational resources. We
propose a novel approach that enables to a generation of objects with a given
set of attributes without retraining the base model. For this purpose, we
utilize the normalizing flow models - Conditional Masked Autoregressive Flow
and Conditional Real NVP, as a Flow Plugin Network (FPN).

    

### [[2110.04094] Privacy-Aware Communication Over the Wiretap Channel with Generative Networks](http://arxiv.org/abs/2110.04094)


  We study privacy-aware communication over a wiretap channel using end-to-end
learning. Alice wants to transmit a source signal to Bob over a binary
symmetric channel, while passive eavesdropper Eve tries to infer some sensitive
attribute of Alice's source based on its overheard signal. Since we usually do
not have access to true distributions, we propose a data-driven approach using
variational autoencoder (VAE)-based joint source channel coding (JSCC). We show
through simulations with the colored MNIST dataset that our approach provides
high reconstruction quality at the receiver while confusing the eavesdropper
about the latent sensitive attribute, which consists of the color and thickness
of the digits. Finally, we consider a parallel-channel scenario, and show that
our approach arranges the information transmission such that the channels with
higher noise levels at the eavesdropper carry the sensitive information, while
the non-sensitive information is transmitted over more vulnerable channels.

    

### [[2110.04099] Topology-Imbalance Learning for Semi-Supervised Node Classification](http://arxiv.org/abs/2110.04099)


  The class imbalance problem, as an important issue in learning node
representations, has drawn increasing attention from the community. Although
the imbalance considered by existing studies roots from the unequal quantity of
labeled examples in different classes (quantity imbalance), we argue that graph
data expose a unique source of imbalance from the asymmetric topological
properties of the labeled nodes, i.e., labeled nodes are not equal in terms of
their structural role in the graph (topology imbalance). In this work, we first
probe the previously unknown topology-imbalance issue, including its
characteristics, causes, and threats to semi-supervised node classification
learning. We then provide a unified view to jointly analyzing the quantity- and
topology- imbalance issues by considering the node influence shift phenomenon
with the Label Propagation algorithm. In light of our analysis, we devise an
influence conflict detection -- based metric Totoro to measure the degree of
graph topology imbalance and propose a model-agnostic method ReNode to address
the topology-imbalance issue by re-weighting the influence of labeled nodes
adaptively based on their relative positions to class boundaries. Systematic
experiments demonstrate the effectiveness and generalizability of our method in
relieving topology-imbalance issue and promoting semi-supervised node
classification. The further analysis unveils varied sensitivity of different
graph neural networks (GNNs) to topology imbalance, which may serve as a new
perspective in evaluating GNN architectures.

    

### [[2110.04111] Discover, Hallucinate, and Adapt: Open Compound Domain Adaptation for Semantic Segmentation](http://arxiv.org/abs/2110.04111)


  Unsupervised domain adaptation (UDA) for semantic segmentation has been
attracting attention recently, as it could be beneficial for various
label-scarce real-world scenarios (e.g., robot control, autonomous driving,
medical imaging, etc.). Despite the significant progress in this field, current
works mainly focus on a single-source single-target setting, which cannot
handle more practical settings of multiple targets or even unseen targets. In
this paper, we investigate open compound domain adaptation (OCDA), which deals
with mixed and novel situations at the same time, for semantic segmentation. We
present a novel framework based on three main design principles: discover,
hallucinate, and adapt. The scheme first clusters compound target data based on
style, discovering multiple latent domains (discover). Then, it hallucinates
multiple latent target domains in source by using image-translation
(hallucinate). This step ensures the latent domains in the source and the
target to be paired. Finally, target-to-source alignment is learned separately
between domains (adapt). In high-level, our solution replaces a hard OCDA
problem with much easier multiple UDA problems. We evaluate our solution on
standard benchmark GTA to C-driving, and achieved new state-of-the-art results.

    

### [[2110.04117] Detecting adversaries in Crowdsourcing](http://arxiv.org/abs/2110.04117)


  Despite its successes in various machine learning and data science tasks,
crowdsourcing can be susceptible to attacks from dedicated adversaries. This
work investigates the effects of adversaries on crowdsourced classification,
under the popular Dawid and Skene model. The adversaries are allowed to deviate
arbitrarily from the considered crowdsourcing model, and may potentially
cooperate. To address this scenario, we develop an approach that leverages the
structure of second-order moments of annotator responses, to identify large
numbers of adversaries, and mitigate their impact on the crowdsourcing task.
The potential of the proposed approach is empirically demonstrated on synthetic
and real crowdsourcing datasets.

    

### [[2110.04119] A Multi-viewpoint Outdoor Dataset for Human Action Recognition](http://arxiv.org/abs/2110.04119)


  Advancements in deep neural networks have contributed to near perfect results
for many computer vision problems such as object recognition, face recognition
and pose estimation. However, human action recognition is still far from
human-level performance. Owing to the articulated nature of the human body, it
is challenging to detect an action from multiple viewpoints, particularly from
an aerial viewpoint. This is further compounded by a scarcity of datasets that
cover multiple viewpoints of actions. To fill this gap and enable research in
wider application areas, we present a multi-viewpoint outdoor action
recognition dataset collected from YouTube and our own drone. The dataset
consists of 20 dynamic human action classes, 2324 video clips and 503086
frames. All videos are cropped and resized to 720x720 without distorting the
original aspect ratio of the human subjects in videos. This dataset should be
useful to many research areas including action recognition, surveillance and
situational awareness. We evaluated the dataset with a two-stream CNN
architecture coupled with a recently proposed temporal pooling scheme called
kernelized rank pooling that produces nonlinear feature subspace
representations. The overall baseline action recognition accuracy is 74.0%.

    

### [[2110.04121] On the Limitations of Multimodal VAEs](http://arxiv.org/abs/2110.04121)


  Multimodal variational autoencoders (VAEs) have shown promise as efficient
generative models for weakly-supervised data. Yet, despite their advantage of
weak supervision, they exhibit a gap in generative quality compared to unimodal
VAEs, which are completely unsupervised. In an attempt to explain this gap, we
uncover a fundamental limitation that applies to a large family of
mixture-based multimodal VAEs. We prove that the sub-sampling of modalities
enforces an undesirable upper bound on the multimodal ELBO and thereby limits
the generative quality of the respective models. Empirically, we showcase the
generative quality gap on both synthetic and real data and present the
tradeoffs between different variants of multimodal VAEs. We find that none of
the existing approaches fulfills all desired criteria of an effective
multimodal generative model when applied on more complex datasets than those
used in previous benchmarks. In summary, we identify, formalize, and validate
fundamental limitations of VAE-based approaches for modeling weakly-supervised
data and discuss implications for real-world applications.

    

### [[2110.04124] Ensemble Neural Representation Networks](http://arxiv.org/abs/2110.04124)


  Implicit Neural Representation (INR) has recently attracted considerable
attention for storing various types of signals in continuous forms. The
existing INR networks require lengthy training processes and high-performance
computational resources. In this paper, we propose a novel sub-optimal ensemble
architecture for INR that resolves the aforementioned problems. In this
architecture, the representation task is divided into several sub-tasks done by
independent sub-networks. We show that the performance of the proposed ensemble
INR architecture may decrease if the dimensions of sub-networks increase.
Hence, it is vital to suggest an optimization algorithm to find the sub-optimal
structure of the ensemble network, which is done in this paper. According to
the simulation results, the proposed architecture not only has significantly
fewer floating-point operations (FLOPs) and less training time, but it also has
better performance in terms of Peak Signal to Noise Ratio (PSNR) compared to
those of its counterparts.

    

### [[2110.04126] 3D Infomax improves GNNs for Molecular Property Prediction](http://arxiv.org/abs/2110.04126)


  Molecular property prediction is one of the fastest-growing applications of
deep learning with critical real-world impacts. Including 3D molecular
structure as input to learned models their performance for many molecular
tasks. However, this information is infeasible to compute at the scale required
by several real-world applications. We propose pre-training a model to reason
about the geometry of molecules given only their 2D molecular graphs. Using
methods from self-supervised learning, we maximize the mutual information
between 3D summary vectors and the representations of a Graph Neural Network
(GNN) such that they contain latent 3D information. During fine-tuning on
molecules with unknown geometry, the GNN still generates implicit 3D
information and can use it to improve downstream tasks. We show that 3D
pre-training provides significant improvements for a wide range of properties,
such as a 22% average MAE reduction on eight quantum mechanical properties.
Moreover, the learned representations can be effectively transferred between
datasets in different molecular spaces.

    

### [[2110.04127] Deep Upper Confidence Bound Algorithm for Contextual Bandit Ranking of Information Selection](http://arxiv.org/abs/2110.04127)


  Contextual multi-armed bandits (CMAB) have been widely used for learning to
filter and prioritize information according to a user's interest. In this work,
we analyze top-K ranking under the CMAB framework where the top-K arms are
chosen iteratively to maximize a reward. The context, which represents a set of
observable factors related to the user, is used to increase prediction accuracy
compared to a standard multi-armed bandit. Contextual bandit methods have
mostly been studied under strict linearity assumptions, but we drop that
assumption and learn non-linear stochastic reward functions with deep neural
networks. We introduce a novel algorithm called the Deep Upper Confidence Bound
(UCB) algorithm. Deep UCB balances exploration and exploitation with a separate
neural network to model the learning convergence. We compare the performance of
many bandit algorithms varying K over real-world data sets with
high-dimensional data and non-linear reward functions. Empirical results show
that the performance of Deep UCB often outperforms though it is sensitive to
the problem and reward setup. Additionally, we prove theoretical regret bounds
on Deep UCB giving convergence to optimality for the weak class of CMAB
problems.

    

### [[2110.04130] Learning post-processing for QRS detection using Recurrent Neural Network](http://arxiv.org/abs/2110.04130)


  Deep-learning based QRS-detection algorithms often require essential
post-processing to refine the prediction streams for R-peak localisation. The
post-processing performs signal-processing tasks from as simple as, removing
isolated 0s or 1s in the prediction-stream to sophisticated steps, which
require domain-specific knowledge, including the minimum threshold of a
QRS-complex extent or R-R interval. Often these thresholds vary among
QRS-detection studies and are empirically determined for the target dataset,
which may have implications if the target dataset differs. Moreover, these
studies, in general, fail to identify the relative strengths of deep-learning
models and post-processing to weigh them appropriately. This study classifies
post-processing, as found in the QRS-detection literature, into two levels -
moderate, and advanced - and advocates that the thresholds be learned by an
appropriate deep-learning module, called a Gated Recurrent Unit (GRU), to avoid
explicitly setting post-processing thresholds. This is done by utilising the
same philosophy of shifting from hand-crafted feature-engineering to
deep-learning-based feature-extraction. The results suggest that GRU learns the
post-processing level and the QRS detection performance using GRU-based
post-processing marginally follows the domain-specific manual post-processing,
without requiring usage of domain-specific threshold parameters. To the best of
our knowledge, the use of GRU to learn QRS-detection post-processing from CNN
model generated prediction streams is the first of its kind. The outcome was
used to recommend a modular design for a QRS-detection system, where the level
of complexity of the CNN model and post-processing can be tuned based on the
deployment environment.

    

### [[2110.04133] Quantifying Inequality in Underreported Medical Conditions](http://arxiv.org/abs/2110.04133)


  Estimating the prevalence of a medical condition, or the proportion of the
population in which it occurs, is a fundamental problem in healthcare and
public health. Accurate estimates of the relative prevalence across groups --
capturing, for example, that a condition affects women more frequently than men
-- facilitate effective and equitable health policy which prioritizes groups
who are disproportionately affected by a condition. However, it is difficult to
estimate relative prevalence when a medical condition is underreported. In this
work, we provide a method for accurately estimating the relative prevalence of
underreported medical conditions, building upon the positive unlabeled learning
framework. We show that under the commonly made covariate shift assumption --
i.e., that the probability of having a disease conditional on symptoms remains
constant across groups -- we can recover the relative prevalence, even without
restrictive assumptions commonly made in positive unlabeled learning and even
if it is impossible to recover the absolute prevalence. We provide a suite of
experiments on synthetic and real health data that demonstrate our method's
ability to recover the relative prevalence more accurately than do baselines,
and the method's robustness to plausible violations of the covariate shift
assumption.

    

### [[2110.04135] Revisiting Design Choices in Model-Based Offline Reinforcement Learning](http://arxiv.org/abs/2110.04135)


  Offline reinforcement learning enables agents to leverage large pre-collected
datasets of environment transitions to learn control policies, circumventing
the need for potentially expensive or unsafe online data collection.
Significant progress has been made recently in offline model-based
reinforcement learning, approaches which leverage a learned dynamics model.
This typically involves constructing a probabilistic model, and using the model
uncertainty to penalize rewards where there is insufficient data, solving for a
pessimistic MDP that lower bounds the true MDP. Existing methods, however,
exhibit a breakdown between theory and practice, whereby pessimistic return
ought to be bounded by the total variation distance of the model from the true
dynamics, but is instead implemented through a penalty based on estimated model
uncertainty. This has spawned a variety of uncertainty heuristics, with little
to no comparison between differing approaches. In this paper, we compare these
heuristics, and design novel protocols to investigate their interaction with
other hyperparameters, such as the number of models, or imaginary rollout
horizon. Using these insights, we show that selecting these key hyperparameters
using Bayesian Optimization produces superior configurations that are vastly
different to those currently used in existing hand-tuned state-of-the-art
methods, and result in drastically stronger performance.

    

### [[2110.04136] Adaptive Sampling for Heterogeneous Rank Aggregation from Noisy Pairwise Comparisons](http://arxiv.org/abs/2110.04136)


  In heterogeneous rank aggregation problems, users often exhibit various
accuracy levels when comparing pairs of items. Thus a uniform querying strategy
over users may not be optimal. To address this issue, we propose an
elimination-based active sampling strategy, which estimates the ranking of
items via noisy pairwise comparisons from users and improves the users' average
accuracy by maintaining an active set of users. We prove that our algorithm can
return the true ranking of items with high probability. We also provide a
sample complexity bound for the proposed algorithm which is better than that of
non-active strategies in the literature. Experiments are provided to show the
empirical advantage of the proposed methods over the state-of-the-art
baselines.

    

### [[2110.04146] Arachnophobia Exposure Therapy using Experience-driven Procedural Content Generation via Reinforcement Learning (EDPCGRL)](http://arxiv.org/abs/2110.04146)


  Personalized therapy, in which a therapeutic practice is adapted to an
individual patient, leads to better health outcomes. Typically, this is
accomplished by relying on a therapist's training and intuition along with
feedback from a patient. While there exist approaches to automatically adapt
therapeutic content to a patient, they rely on hand-authored, pre-defined
rules, which may not generalize to all individuals. In this paper, we propose
an approach to automatically adapt therapeutic content to patients based on
physiological measures. We implement our approach in the context of
arachnophobia exposure therapy, and rely on experience-driven procedural
content generation via reinforcement learning (EDPCGRL) to generate virtual
spiders to match an individual patient. In this initial implementation, and due
to the ongoing pandemic, we make use of virtual or artificial humans
implemented based on prior arachnophobia psychology research. Our EDPCGRL
method is able to more quickly adapt to these virtual humans with high accuracy
in comparison to existing, search-based EDPCG approaches.

    

### [[2110.04156] Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters](http://arxiv.org/abs/2110.04156)


  Over the recent years, vast progress has been made in Offline Reinforcement
Learning (Offline-RL) for various decision-making domains: from finance to
robotics. However, comparing and reporting new Offline-RL algorithms has been
noted as underdeveloped: (1) use of unlimited online evaluation budget for
hyperparameter search (2) sidestepping offline policy selection (3) ad-hoc
performance statistics reporting. In this work, we propose an evaluation
technique addressing these issues, Expected Online Performance, that provides a
performance estimate for a best-found policy given a fixed online evaluation
budget. Using our approach, we can estimate the number of online evaluations
required to surpass a given behavioral policy performance. Applying it to
several Offline-RL baselines, we find that with a limited online evaluation
budget, (1) Behavioral Cloning constitutes a strong baseline over various
expert levels and data regimes, and (2) offline uniform policy selection is
competitive with value-based approaches. We hope the proposed technique will
make it into the toolsets of Offline-RL practitioners to help them arrive at
informed conclusions when deploying RL in real-world systems.

    

### [[2110.04160] Federated Learning for Big Data: A Survey on Opportunities, Applications, and Future Directions](http://arxiv.org/abs/2110.04160)


  Big data has remarkably evolved over the last few years to realize an
enormous volume of data generated from newly emerging services and applications
and a massive number of Internet-of-Things (IoT) devices. The potential of big
data can be realized via analytic and learning techniques, in which the data
from various sources is transferred to a central cloud for central storage,
processing, and training. However, this conventional approach faces critical
issues in terms of data privacy as the data may include sensitive data such as
personal information, governments, banking accounts. To overcome this
challenge, federated learning (FL) appeared to be a promising learning
technique. However, a gap exists in the literature that a comprehensive survey
on FL for big data services and applications is yet to be conducted. In this
article, we present a survey on the use of FL for big data services and
applications, aiming to provide general readers with an overview of FL, big
data, and the motivations behind the use of FL for big data. In particular, we
extensively review the use of FL for key big data services, including big data
acquisition, big data storage, big data analytics, and big data privacy
preservation. Subsequently, we review the potential of FL for big data
applications, such as smart city, smart healthcare, smart transportation, smart
grid, and social media. Further, we summarize a number of important projects on
FL-big data and discuss key challenges of this interesting topic along with
several promising solutions and directions.

    

### [[2110.04169] Iterative Decoding for Compositional Generalization in Transformers](http://arxiv.org/abs/2110.04169)


  Deep learning models do well at generalizing to in-distribution data but
struggle to generalize compositionally, i.e., to combine a set of learned
primitives to solve more complex tasks. In particular, in sequence-to-sequence
(seq2seq) learning, transformers are often unable to predict correct outputs
for even marginally longer examples than those seen during training. This paper
introduces iterative decoding, an alternative to seq2seq learning that (i)
improves transformer compositional generalization and (ii) evidences that, in
general, seq2seq transformers do not learn iterations that are not unrolled.
Inspired by the idea of compositionality -- that complex tasks can be solved by
composing basic primitives -- training examples are broken down into a sequence
of intermediate steps that the transformer then learns iteratively. At
inference time, the intermediate outputs are fed back to the transformer as
intermediate inputs until an end-of-iteration token is predicted. Through
numerical experiments, we show that transfomers trained via iterative decoding
outperform their seq2seq counterparts on the PCFG dataset, and solve the
problem of calculating Cartesian products between vectors longer than those
seen during training with 100% accuracy, a task at which seq2seq models have
been shown to fail. We also illustrate a limitation of iterative decoding,
specifically, that it can make sorting harder to learn on the CFQ dataset.

    

### [[2110.04173] TopoDetect: Framework for Topological Features Detection in Graph Embeddings](http://arxiv.org/abs/2110.04173)


  TopoDetect is a Python package that allows the user to investigate if
important topological features, such as the Degree of the nodes, their Triangle
Count, or their Local Clustering Score, are preserved in the embeddings of
graph representation models. Additionally, the framework enables the
visualization of the embeddings according to the distribution of the
topological features among the nodes. Moreover, TopoDetect enables us to study
the effect of the preservation of these features by evaluating the performance
of the embeddings on downstream learning tasks such as clustering and
classification.

    

### [[2110.04175] RelaySum for Decentralized Deep Learning on Heterogeneous Data](http://arxiv.org/abs/2110.04175)


  In decentralized machine learning, workers compute model updates on their
local data. Because the workers only communicate with few neighbors without
central coordination, these updates propagate progressively over the network.
This paradigm enables distributed training on networks without all-to-all
connectivity, helping to protect data privacy as well as to reduce the
communication cost of distributed training in data centers. A key challenge,
primarily in decentralized deep learning, remains the handling of differences
between the workers' local data distributions. To tackle this challenge, we
introduce the RelaySum mechanism for information propagation in decentralized
learning. RelaySum uses spanning trees to distribute information exactly
uniformly across all workers with finite delays depending on the distance
between nodes. In contrast, the typical gossip averaging mechanism only
distributes data uniformly asymptotically while using the same communication
volume per step as RelaySum. We prove that RelaySGD, based on this mechanism,
is independent of data heterogeneity and scales to many workers, enabling
highly accurate decentralized deep learning on heterogeneous data. Our code is
available at this http URL.

    

### [[2110.04176] Lightweight Convolutional Neural Networks By Hypercomplex Parameterization](http://arxiv.org/abs/2110.04176)


  Hypercomplex neural networks have proved to reduce the overall number of
parameters while ensuring valuable performances by leveraging the properties of
Clifford algebras. Recently, hypercomplex linear layers have been further
improved by involving efficient parameterized Kronecker products. In this
paper, we define the parameterization of hypercomplex convolutional layers to
develop lightweight and efficient large-scale convolutional models. Our method
grasps the convolution rules and the filters organization directly from data
without requiring a rigidly predefined domain structure to follow. The proposed
approach is flexible to operate in any user-defined or tuned domain, from 1D to
$n$D regardless of whether the algebra rules are preset. Such a malleability
allows processing multidimensional inputs in their natural domain without
annexing further dimensions, as done, instead, in quaternion neural networks
for 3D inputs like color images. As a result, the proposed method operates with
$1/n$ free parameters as regards its analog in the real domain. We demonstrate
the versatility of this approach to multiple domains of application by
performing experiments on various image datasets as well as audio datasets in
which our method outperforms real and quaternion-valued counterparts.

    

### [[2110.04181] Dataset Condensation with Distribution Matching](http://arxiv.org/abs/2110.04181)


  Computational cost to train state-of-the-art deep models in many learning
problems is rapidly increasing due to more sophisticated models and larger
datasets. A recent promising direction to reduce training time is dataset
condensation that aims to replace the original large training set with a
significantly smaller learned synthetic set while preserving its information.
While training deep models on the small set of condensed images can be
extremely fast, their synthesis remains computationally expensive due to the
complex bi-level optimization and second-order derivative computation. In this
work, we propose a simple yet effective dataset condensation technique that
requires significantly lower training cost with comparable performance by
matching feature distributions of the synthetic and original training images in
sampled embedding spaces. Thanks to its efficiency, we apply our method to more
realistic and larger datasets with sophisticated neural architectures and
achieve a significant performance boost while using larger synthetic training
set. We also show various practical benefits of our method in continual
learning and neural architecture search.

    

### [[2110.04182] Temporal Convolutions for Multi-Step Quadrotor Motion Prediction](http://arxiv.org/abs/2110.04182)


  Model-based control methods for robotic systems such as quadrotors,
autonomous driving vehicles and flexible manipulators require motion models
that generate accurate predictions of complex nonlinear system dynamics over
long periods of time. Temporal Convolutional Networks (TCNs) can be adapted to
this challenge by formulating multi-step prediction as a sequence-to-sequence
modeling problem. We present End2End-TCN: a fully convolutional architecture
that integrates future control inputs to compute multi-step motion predictions
in one forward pass. We demonstrate the approach with a thorough analysis of
TCN performance for the quadrotor modeling task, which includes an
investigation of scaling effects and ablation studies. Ultimately, End2End-TCN
provides 55% error reduction over the state of the art in multi-step prediction
on an aggressive indoor quadrotor flight dataset. The model yields accurate
predictions across 90 timestep horizons over a 900 ms interval.

    

### [[2110.04184] When Can We Learn General-Sum Markov Games with a Large Number of Players Sample-Efficiently?](http://arxiv.org/abs/2110.04184)


  Multi-agent reinforcement learning has made substantial empirical progresses
in solving games with a large number of players. However, theoretically, the
best known sample complexity for finding a Nash equilibrium in general-sum
games scales exponentially in the number of players due to the size of the
joint action space, and there is a matching exponential lower bound. This paper
investigates what learning goals admit better sample complexities in the
setting of $m$-player general-sum Markov games with $H$ steps, $S$ states, and
$A_i$ actions per player. First, we design algorithms for learning an
$\epsilon$-Coarse Correlated Equilibrium (CCE) in
$\widetilde{\mathcal{O}}(H^5S\max_{i\le m} A_i / \epsilon^2)$ episodes, and an
$\epsilon$-Correlated Equilibrium (CE) in
$\widetilde{\mathcal{O}}(H^6S\max_{i\le m} A_i^2 / \epsilon^2)$ episodes. This
is the first line of results for learning CCE and CE with sample complexities
polynomial in $\max_{i\le m} A_i$. Our algorithm for learning CE integrates an
adversarial bandit subroutine which minimizes a weighted swap regret, along
with several novel designs in the outer loop. Second, we consider the important
special case of Markov Potential Games, and design an algorithm that learns an
$\epsilon$-approximate Nash equilibrium within
$\widetilde{\mathcal{O}}(S\sum_{i\le m} A_i / \epsilon^3)$ episodes (when only
highlighting the dependence on $S$, $A_i$, and $\epsilon$), which only depends
linearly in $\sum_{i\le m} A_i$ and significantly improves over the best known
algorithm in the $\epsilon$ dependence. Overall, our results shed light on what
equilibria or structural assumptions on the game may enable sample-efficient
learning with many players.

    

### [[2110.04186] Medical Dead-ends and Learning to Identify High-risk States and Treatments](http://arxiv.org/abs/2110.04186)


  Machine learning has successfully framed many sequential decision making
problems as either supervised prediction, or optimal decision-making policy
identification via reinforcement learning. In data-constrained offline
settings, both approaches may fail as they assume fully optimal behavior or
rely on exploring alternatives that may not exist. We introduce an inherently
different approach that identifies possible ``dead-ends'' of a state space. We
focus on the condition of patients in the intensive care unit, where a
``medical dead-end'' indicates that a patient will expire, regardless of all
potential future treatment sequences. We postulate ``treatment security'' as
avoiding treatments with probability proportional to their chance of leading to
dead-ends, present a formal proof, and frame discovery as an RL problem. We
then train three independent deep neural models for automated state
construction, dead-end discovery and confirmation. Our empirical results
discover that dead-ends exist in real clinical data among septic patients, and
further reveal gaps between secure treatments and those that were administered.

    

### [[2110.04187] SCaLa: Supervised Contrastive Learning for End-to-End Automatic Speech Recognition](http://arxiv.org/abs/2110.04187)


  End-to-end Automatic Speech Recognition (ASR) models are usually trained to
reduce the losses of the whole token sequences, while neglecting explicit
phonemic-granularity supervision. This could lead to recognition errors due to
similar-phoneme confusion or phoneme reduction. To alleviate this problem, this
paper proposes a novel framework of Supervised Contrastive Learning (SCaLa) to
enhance phonemic information learning for end-to-end ASR systems. Specifically,
we introduce the self-supervised Masked Contrastive Predictive Coding (MCPC)
into the fully-supervised setting. To supervise phoneme learning explicitly,
SCaLa first masks the variable-length encoder features corresponding to
phonemes given phoneme forced-alignment extracted from a pre-trained acoustic
model, and then predicts the masked phonemes via contrastive learning. The
phoneme forced-alignment can mitigate the noise of positive-negative pairs in
self-supervised MCPC. Experimental results conducted on reading and spontaneous
speech datasets show that the proposed approach achieves 2.84% and 1.38%
Character Error Rate (CER) reductions compared to the baseline, respectively.

    

### [[2110.04193] On Fast Johnson-Lindernstrauss Embeddings of Compact Submanifolds of $\mathbb{R}^N$ with Boundary](http://arxiv.org/abs/2110.04193)


  Let $\mathcal{M}$ be a smooth $d$-dimensional submanifold of $\mathbb{R}^N$
with boundary that's equipped with the Euclidean (chordal) metric, and choose
$m \leq N$. In this paper we consider the probability that a random matrix $A
\in \mathbb{R}^{m \times N}$ will serve as a bi-Lipschitz function $A:
\mathcal{M} \rightarrow \mathbb{R}^m$ with bi-Lipschitz constants close to one
for three different types of distributions on the $m \times N$ matrices $A$,
including two whose realizations are guaranteed to have fast matrix-vector
multiplies. In doing so we generalize prior randomized metric space embedding
results of this type for submanifolds of $\mathbb{R}^N$ by allowing for the
presence of boundary while also retaining, and in some cases improving, prior
lower bounds on the achievable embedding dimensions $m$ for which one can
expect small distortion with high probability. In particular, motivated by
recent modewise embedding constructions for tensor data, herein we present a
new class of highly structured distributions on matrices which outperform prior
structured matrix distributions for embedding sufficiently low-dimensional
submanifolds of $\mathbb{R}^N$ (with $d \lesssim \sqrt{N}$) with respect to
both achievable embedding dimension, and computationally efficient
realizations. As a consequence we are able to present, for example, a general
new class of Johnson-Lindenstrauss embedding matrices for $\mathcal{O}(\log^c
N)$-dimensional submanifolds of $\mathbb{R}^N$ which enjoy $\mathcal{O}(N \log
\log N))$-time matrix vector multiplications.

    

### [[2110.04202] Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation](http://arxiv.org/abs/2110.04202)


  Domain adaptation (DA) aims to alleviate the domain shift between source
domain and target domain. Most DA methods require access to the source data,
but often that is not possible (e.g. due to data privacy or intellectual
property). In this paper, we address the challenging source-free domain
adaptation (SFDA) problem, where the source pretrained model is adapted to the
target domain in the absence of source data. Our method is based on the
observation that target data, which might no longer align with the source
domain classifier, still forms clear clusters. We capture this intrinsic
structure by defining local affinity of the target data, and encourage label
consistency among data with high local affinity. We observe that higher
affinity should be assigned to reciprocal neighbors, and propose a self
regularization loss to decrease the negative impact of noisy neighbors.
Furthermore, to aggregate information with more context, we consider expanded
neighborhoods with small affinity values. In the experimental results we verify
that the inherent structure of the target features is an important source of
information for domain adaptation. We demonstrate that this local structure can
be efficiently captured by considering the local neighbors, the reciprocal
neighbors, and the expanded neighborhood. Finally, we achieve state-of-the-art
performance on several 2D image and 3D point cloud recognition datasets. Code
is available in this https URL.

    

### [[2110.04227] Universal Joint Approximation of Manifolds and Densities by Simple Injective Flows](http://arxiv.org/abs/2110.04227)


  We analyze neural networks composed of bijective flows and injective
expansive elements. We find that such networks universally approximate a large
class of manifolds simultaneously with densities supported on them. Among
others, our results apply to the well-known coupling and autoregressive flows.
We build on the work of Teshima et al. 2020 on bijective flows and study
injective architectures proposed in Brehmer et al. 2020 and Kothari et al.
2021. Our results leverage a new theoretical device called the embedding gap,
which measures how far one continuous manifold is from embedding another. We
relate the embedding gap to a relaxation of universally we call the manifold
embedding property, capturing the geometric part of universality. Our proof
also establishes that optimality of a network can be established in reverse,
resolving a conjecture made in Brehmer et al. 2020 and opening the door for
simple layer-wise training schemes. Finally, we show that the studied networks
admit an exact layer-wise projection result, Bayesian uncertainty
quantification, and black-box recovery of network weights.

    

### [[2110.04228] Hybrid Graph Embedding Techniques in Estimated Time of Arrival Task](http://arxiv.org/abs/2110.04228)


  Recently, deep learning has achieved promising results in the calculation of
Estimated Time of Arrival (ETA), which is considered as predicting the travel
time from the start point to a certain place along a given path. ETA plays an
essential role in intelligent taxi services or automotive navigation systems. A
common practice is to use embedding vectors to represent the elements of a road
network, such as road segments and crossroads. Road elements have their own
attributes like length, presence of crosswalks, lanes number, etc. However,
many links in the road network are traversed by too few floating cars even in
large ride-hailing platforms and affected by the wide range of temporal events.
As the primary goal of the research, we explore the generalization ability of
different spatial embedding strategies and propose a two-stage approach to deal
with such problems.

    

### [[2110.04232] Learning Topic Models: Identifiability and Finite-Sample Analysis](http://arxiv.org/abs/2110.04232)


  Topic models provide a useful text-mining tool for learning, extracting and
discovering latent structures in large text corpora. Although a plethora of
methods have been proposed for topic modeling, a formal theoretical
investigation on the statistical identifiability and accuracy of latent topic
estimation is lacking in the literature. In this paper, we propose a maximum
likelihood estimator (MLE) of latent topics based on a specific integrated
likelihood, which is naturally connected to the concept of volume minimization
in computational geometry. Theoretically, we introduce a new set of geometric
conditions for topic model identifiability, which are weaker than conventional
separability conditions relying on the existence of anchor words or pure topic
documents. We conduct finite-sample error analysis for the proposed estimator
and discuss the connection of our results with existing ones. We conclude with
empirical studies on both simulated and real datasets.

    

### [[2110.04241] Cognitive Coding of Speech](http://arxiv.org/abs/2110.04241)


  We propose an approach for cognitive coding of speech by unsupervised
extraction of contextual representations in two hierarchical levels of
abstraction. Speech attributes such as phoneme identity that last one hundred
milliseconds or less are captured in the lower level of abstraction, while
speech attributes such as speaker identity and emotion that persist up to one
second are captured in the higher level of abstraction. This decomposition is
achieved by a two-stage neural network, with a lower and an upper stage
operating at different time scales. Both stages are trained to predict the
content of the signal in their respective latent spaces. A top-down pathway
between stages further improves the predictive capability of the network. With
an application in speech compression in mind, we investigate the effect of
dimensionality reduction and low bitrate quantization on the extracted
representations. The performance measured on the LibriSpeech and EmoV-DB
datasets reaches, and for some speech attributes even exceeds, that of
state-of-the-art approaches.

    

### [[2110.04243] Heavy Ball Momentum for Conditional Gradient](http://arxiv.org/abs/2110.04243)


  Conditional gradient, aka Frank Wolfe (FW) algorithms, have well-documented
merits in machine learning and signal processing applications. Unlike
projection-based methods, momentum cannot improve the convergence rate of FW,
in general. This limitation motivates the present work, which deals with heavy
ball momentum, and its impact to FW. Specifically, it is established that heavy
ball offers a unifying perspective on the primal-dual (PD) convergence, and
enjoys a tighter per iteration PD error rate, for multiple choices of step
sizes, where PD error can serve as the stopping criterion in practice. In
addition, it is asserted that restart, a scheme typically employed jointly with
Nesterov's momentum, can further tighten this PD error bound. Numerical results
demonstrate the usefulness of heavy ball momentum in FW iterations.

    

### [[2110.04248] Observations on K-image Expansion of Image-Mixing Augmentation for Classification](http://arxiv.org/abs/2110.04248)


  Image-mixing augmentations (e.g., Mixup or CutMix), which typically mix two
images, have become de-facto training tricks for image classification. Despite
their huge success on image classification, the number of images to mix has not
been profoundly investigated by the previous works, only showing the naive
K-image expansion leads to poor performance degradation. This paper derives a
new K-image mixing augmentation based on the stick-breaking process under
Dirichlet prior. We show that our method can train more robust and generalized
classifiers through extensive experiments and analysis on classification
accuracy, a shape of a loss landscape and adversarial robustness, than the
usual two-image methods. Furthermore, we show that our probabilistic model can
measure the sample-wise uncertainty and can boost the efficiency for Network
Architecture Search (NAS) with 7x reduced search time.

    

### [[2110.04252] LCS: Learning Compressible Subspaces for Adaptive Network Compression at Inference Time](http://arxiv.org/abs/2110.04252)


  When deploying deep learning models to a device, it is traditionally assumed
that available computational resources (compute, memory, and power) remain
static. However, real-world computing systems do not always provide stable
resource guarantees. Computational resources need to be conserved when load
from other processes is high or battery power is low. Inspired by recent works
on neural network subspaces, we propose a method for training a "compressible
subspace" of neural networks that contains a fine-grained spectrum of models
that range from highly efficient to highly accurate. Our models require no
retraining, thus our subspace of models can be deployed entirely on-device to
allow adaptive network compression at inference time. We present results for
achieving arbitrarily fine-grained accuracy-efficiency trade-offs at inference
time for structured and unstructured sparsity. We achieve accuracies on-par
with standard models when testing our uncompressed models, and maintain high
accuracy for sparsity rates above 90% when testing our compressed models. We
also demonstrate that our algorithm extends to quantization at variable bit
widths, achieving accuracy on par with individually trained networks.

    

### [[2110.04253] F-Divergences and Cost Function Locality in Generative Modelling with Quantum Circuits](http://arxiv.org/abs/2110.04253)


  Generative modelling is an important unsupervised task in machine learning.
In this work, we study a hybrid quantum-classical approach to this task, based
on the use of a quantum circuit Born machine. In particular, we consider
training a quantum circuit Born machine using $f$-divergences. We first discuss
the adversarial framework for generative modelling, which enables the
estimation of any $f$-divergence in the near term. Based on this capability, we
introduce two heuristics which demonstrably improve the training of the Born
machine. The first is based on $f$-divergence switching during training. The
second introduces locality to the divergence, a strategy which has proved
important in similar applications in terms of mitigating barren plateaus.
Finally, we discuss the long-term implications of quantum devices for computing
$f$-divergences, including algorithms which provide quadratic speedups to their
estimation. In particular, we generalise existing algorithms for estimating the
Kullback-Leibler divergence and the total variation distance to obtain a
fault-tolerant quantum algorithm for estimating another $f$-divergence, namely,
the Pearson divergence.

    

### [[2110.04254] Assessment of Neural Networks for Stream-Water-Temperature Prediction](http://arxiv.org/abs/2110.04254)


  Climate change results in altered air and water temperatures. Increases
affect physicochemical properties, such as oxygen concentration, and can shift
species distribution and survival, with consequences for ecosystem functioning
and services. These ecosystem services have integral value for humankind and
are forecasted to alter under climate warming. A mechanistic understanding of
the drivers and magnitude of expected changes is essential in identifying
system resilience and mitigation measures. In this work, we present a selection
of state-of-the-art Neural Networks (NN) for the prediction of water
temperatures in six streams in Germany. We show that the use of methods that
compare observed and predicted values, exemplified with the Root Mean Square
Error (RMSE), is not sufficient for their assessment. Hence we introduce
additional analysis methods for our models to complement the state-of-the-art
metrics. These analyses evaluate the NN's robustness, possible maximal and
minimal values, and the impact of single input parameters on the output. We
thus contribute to understanding the processes within the NN and help
applicants choose architectures and input parameters for reliable water
temperature prediction models.

    

### [[2110.04256] Big Machinery Data Preprocessing Methodology for Data-Driven Models in Prognostics and Health Management](http://arxiv.org/abs/2110.04256)


  Sensor monitoring networks and advances in big data analytics have guided the
reliability engineering landscape to a new era of big machinery data. Low-cost
sensors, along with the evolution of the internet of things and industry 4.0,
have resulted in rich databases that can be analyzed through prognostics and
health management (PHM) frameworks. Several da-ta-driven models (DDMs) have
been proposed and applied for diagnostics and prognostics purposes in complex
systems. However, many of these models are developed using simulated or
experimental data sets, and there is still a knowledge gap for applications in
real operating systems. Furthermore, little attention has been given to the
required data preprocessing steps compared to the training processes of these
DDMs. Up to date, research works do not follow a formal and consistent data
preprocessing guideline for PHM applications. This paper presents a
comprehensive, step-by-step pipeline for the preprocessing of monitoring data
from complex systems aimed for DDMs. The importance of expert knowledge is
discussed in the context of data selection and label generation. Two case
studies are presented for validation, with the end goal of creating clean data
sets with healthy and unhealthy labels that are then used to train machinery
health state classifiers.

    

### [[2110.04260] Taming Sparsely Activated Transformer with Stochastic Experts](http://arxiv.org/abs/2110.04260)


  Sparsely activated models (SAMs), such as Mixture-of-Experts (MoE), can
easily scale to have outrageously large amounts of parameters without
significant increase in computational cost. However, SAMs are reported to be
parameter inefficient such that larger models do not always lead to better
performance. While most on-going research focuses on improving SAMs models by
exploring methods of routing inputs to experts, our analysis reveals that such
research might not lead to the solution we expect, i.e., the commonly-used
routing methods based on gating mechanisms do not work better than randomly
routing inputs to experts. In this paper, we propose a new expert-based model,
THOR (Transformer witH StOchastic ExpeRts). Unlike classic expert-based models,
such as the Switch Transformer, experts in THOR are randomly activated for each
input during training and inference. THOR models are trained using a
consistency regularized loss, where experts learn not only from training data
but also from other experts as teachers, such that all the experts make
consistent predictions. We validate the effectiveness of THOR on machine
translation tasks. Results show that THOR models are more parameter efficient
in that they significantly outperform the Transformer and MoE models across
various settings. For example, in multilingual translation, THOR outperforms
the Switch Transformer by 2 BLEU scores, and obtains the same BLEU score as
that of a state-of-the-art MoE model that is 18 times larger. Our code is
publicly available at: this http URL.

    

### [[2110.04261] Extragradient Method: $O(1/K)$ Last-Iterate Convergence for Monotone Variational Inequalities and Connections With Cocoercivity](http://arxiv.org/abs/2110.04261)


  Extragradient method (EG) Korpelevich [1976] is one of the most popular
methods for solving saddle point and variational inequalities problems (VIP).
Despite its long history and significant attention in the optimization
community, there remain important open questions about convergence of EG. In
this paper, we resolve one of such questions and derive the first last-iterate
$O(1/K)$ convergence rate for EG for monotone and Lipschitz VIP without any
additional assumptions on the operator. The rate is given in terms of reducing
the squared norm of the operator. Moreover, we establish several results on the
(non-)cocoercivity of the update operators of EG, Optimistic Gradient Method,
and Hamiltonian Gradient Method, when the original operator is monotone and
Lipschitz.

    

### [[2110.04267] Exploring Heterogeneous Characteristics of Layers in ASR Models for More Efficient Training](http://arxiv.org/abs/2110.04267)


  Transformer-based architectures have been the subject of research aimed at
understanding their overparameterization and the non-uniform importance of
their layers. Applying these approaches to Automatic Speech Recognition, we
demonstrate that the state-of-the-art Conformer models generally have multiple
ambient layers. We study the stability of these layers across runs and model
sizes, propose that group normalization may be used without disrupting their
formation, and examine their correlation with model weight updates in each
layer. Finally, we apply these findings to Federated Learning in order to
improve the training procedure, by targeting Federated Dropout to layers by
importance. This allows us to reduce the model size optimized by clients
without quality degradation, and shows potential for future exploration.

    

### [[2110.04274] On the Implicit Biases of Architecture & Gradient Descent](http://arxiv.org/abs/2110.04274)


  Do neural networks generalise because of bias in the functions returned by
gradient descent, or bias already present in the network architecture? Por
qué no los dos?
This paper finds that while typical networks that fit the training data
already generalise fairly well, gradient descent can further improve
generalisation by selecting networks with a large margin. This conclusion is
based on a careful study of the behaviour of infinite width networks trained by
Bayesian inference and finite width networks trained by gradient descent. To
measure the implicit bias of architecture, new technical tools are developed to
both analytically bound and consistently estimate the average test error of the
neural network--Gaussian process (NNGP) posterior. This error is found to be
already better than chance, corroborating the findings of Valle-Pérez et al.
(2019) and underscoring the importance of architecture. Going beyond this
result, this paper finds that test performance can be substantially improved by
selecting a function with much larger margin than is typical under the NNGP
posterior. This highlights a curious fact: minimum a posteriori functions can
generalise best, and gradient descent can select for those functions. In
summary, new technical tools suggest a nuanced portrait of generalisation
involving both the implicit biases of architecture and gradient descent.
Code for this paper is available at: this https URL.

    

### [[2110.04279] StairwayGraphNet for Inter- and Intra-modality Multi-resolution Brain Graph Alignment and Synthesis](http://arxiv.org/abs/2110.04279)


  Synthesizing multimodality medical data provides complementary knowledge and
helps doctors make precise clinical decisions. Although promising, existing
multimodal brain graph synthesis frameworks have several limitations. First,
they mainly tackle only one problem (intra- or inter-modality), limiting their
generalizability to synthesizing inter- and intra-modality simultaneously.
Second, while few techniques work on super-resolving low-resolution brain
graphs within a single modality (i.e., intra), inter-modality graph
super-resolution remains unexplored though this would avoid the need for costly
data collection and processing. More importantly, both target and source
domains might have different distributions, which causes a domain fracture
between them. To fill these gaps, we propose a multi-resolution
StairwayGraphNet (SG-Net) framework to jointly infer a target graph modality
based on a given modality and super-resolve brain graphs in both inter and
intra domains. Our SG-Net is grounded in three main contributions: (i)
predicting a target graph from a source one based on a novel graph generative
adversarial network in both inter (e.g., morphological-functional) and intra
(e.g., functional-functional) domains, (ii) generating high-resolution brain
graphs without resorting to the time consuming and expensive MRI processing
steps, and (iii) enforcing the source distribution to match that of the ground
truth graphs using an inter-modality aligner to relax the loss function to
optimize. Moreover, we design a new Ground Truth-Preserving loss function to
guide both generators in learning the topological structure of ground truth
brain graphs more accurately. Our comprehensive experiments on predicting
target brain graphs from source graphs using a multi-resolution stairway showed
the outperformance of our method in comparison with its variants and
state-of-the-art method.

    

### [[2110.04280] Pyxis: An Open-Source Performance Dataset of Sparse Accelerators](http://arxiv.org/abs/2110.04280)


  Specialized accelerators provide gains of performance and efficiency in
specific domains of applications. Sparse data structures or/and representations
exist in a wide range of applications. However, it is challenging to design
accelerators for sparse applications because no analytic architecture or
performance-level models are able to fully capture the spectrum of the sparse
data. Accelerator researchers rely on real execution to get precise feedback
for their designs. In this work, we present PYXIS, a performance dataset for
specialized accelerators on sparse data. PYXIS collects accelerator designs and
real execution performance statistics. Currently, there are 73.8 K instances in
PYXIS. PYXIS is open-source, and we are constantly growing PYXIS with new
accelerator designs and performance statistics. PYXIS can benefit researchers
in the fields of accelerator, architecture, performance, algorithm, and many
related topics.

    

### [[2110.04281] Collaging Class-specific GANs for Semantic Image Synthesis](http://arxiv.org/abs/2110.04281)


  We propose a new approach for high resolution semantic image synthesis. It
consists of one base image generator and multiple class-specific generators.
The base generator generates high quality images based on a segmentation map.
To further improve the quality of different objects, we create a bank of
Generative Adversarial Networks (GANs) by separately training class-specific
models. This has several benefits including -- dedicated weights for each
class; centrally aligned data for each model; additional training data from
other sources, potential of higher resolution and quality; and easy
manipulation of a specific object in the scene. Experiments show that our
approach can generate high quality images in high resolution while having
flexibility of object-level control by using class-specific generators.

    

### [[2110.04286] Is MC Dropout Bayesian?](http://arxiv.org/abs/2110.04286)


  MC Dropout is a mainstream "free lunch" method in medical imaging for
approximate Bayesian computations (ABC). Its appeal is to solve out-of-the-box
the daunting task of ABC and uncertainty quantification in Neural Networks
(NNs); to fall within the variational inference (VI) framework; and to propose
a highly multimodal, faithful predictive posterior. We question the properties
of MC Dropout for approximate inference, as in fact MC Dropout changes the
Bayesian model; its predictive posterior assigns $0$ probability to the true
model on closed-form benchmarks; the multimodality of its predictive posterior
is not a property of the true predictive posterior but a design artefact. To
address the need for VI on arbitrary models, we share a generic VI engine
within the pytorch framework. The code includes a carefully designed
implementation of structured (diagonal plus low-rank) multivariate normal
variational families, and mixtures thereof. It is intended as a go-to
no-free-lunch approach, addressing shortcomings of mean-field VI with an
adjustable trade-off between expressivity and computational complexity.

    

### [[2110.04291] Local and Global Context-Based Pairwise Models for Sentence Ordering](http://arxiv.org/abs/2110.04291)


  Sentence Ordering refers to the task of rearranging a set of sentences into
the appropriate coherent order. For this task, most previous approaches have
explored global context-based end-to-end methods using Sequence Generation
techniques. In this paper, we put forward a set of robust local and global
context-based pairwise ordering strategies, leveraging which our prediction
strategies outperform all previous works in this domain. Our proposed encoding
method utilizes the paragraph's rich global contextual information to predict
the pairwise order using novel transformer architectures. Analysis of the two
proposed decoding strategies helps better explain error propagation in pairwise
models. This approach is the most accurate pure pairwise model and our encoding
strategy also significantly improves the performance of other recent approaches
that use pairwise models, including the previous state-of-the-art,
demonstrating the research novelty and generalizability of this work.
Additionally, we show how the pre-training task for ALBERT helps it to
significantly outperform BERT, despite having considerably lesser parameters.
The extensive experimental results, architectural analysis and ablation studies
demonstrate the effectiveness and superiority of the proposed models compared
to the previous state-of-the-art, besides providing a much better understanding
of the functioning of pairwise models.

    

### [[1808.02266] Multi-Output Convolution Spectral Mixture for Gaussian Processes](http://arxiv.org/abs/1808.02266)


  Multi-output Gaussian processes (MOGPs) are an extension of Gaussian
Processes (GPs) for predicting multiple output variables (also called channels,
tasks) simultaneously. In this paper we use the convolution theorem to design a
new kernel for MOGPs, by modeling cross channel dependencies through cross
convolution of time and phase delayed components in the spectral domain. The
resulting kernel is called Multi-Output Convolution Spectral Mixture (MOCSM)
kernel. Results of extensive experiments on synthetic and real-life datasets
demonstrate the advantages of the proposed kernel and its state of the art
performance. MOCSM enjoys the desirable property to reduce to the well known
Spectral Mixture (SM) kernel when a single-channel is considered. A comparison
with the recently introduced Multi-Output Spectral Mixture kernel reveals that
this is not the case for the latter kernel, which contains quadratic terms that
generate undesirable scale effects when the spectral densities of different
channels are either very close or very far from each other in the frequency
domain.

    

### [[1905.03151] Unrestricted Permutation forces Extrapolation: Variable Importance Requires at least One More Model, or There Is No Free Variable Importance](http://arxiv.org/abs/1905.03151)


  This paper reviews and advocates against the use of permute-and-predict (PaP)
methods for interpreting black box functions. Methods such as the variable
importance measures proposed for random forests, partial dependence plots, and
individual conditional expectation plots remain popular because they are both
model-agnostic and depend only on the pre-trained model output, making them
computationally efficient and widely available in software. However, numerous
studies have found that these tools can produce diagnostics that are highly
misleading, particularly when there is strong dependence among features. The
purpose of our work here is to (i) review this growing body of literature, (ii)
provide further demonstrations of these drawbacks along with a detailed
explanation as to why they occur, and (iii) advocate for alternative measures
that involve additional modeling. In particular, we describe how breaking
dependencies between features in hold-out data places undue emphasis on sparse
regions of the feature space by forcing the original model to extrapolate to
regions where there is little to no data. We explore these effects across
various model setups and find support for previous claims in the literature
that PaP metrics can vastly over-emphasize correlated features in both variable
importance measures and partial dependence plots. As an alternative, we discuss
and recommend more direct approaches that involve measuring the change in model
performance after muting the effects of the features under investigation.

    

### [[1906.07869] Identifiability of Hierarchical Latent Attribute Models](http://arxiv.org/abs/1906.07869)


  Hierarchical Latent Attribute Models (HLAMs) are a family of discrete latent
variable models that are attracting increasing attention in educational,
psychological, and behavioral sciences. The key ingredients of an HLAM include
a binary structural matrix and a directed acyclic graph specifying hierarchical
constraints on the configurations of latent attributes. These components encode
practitioners' design information and carry important scientific meanings.
Despite the popularity of HLAMs, the fundamental identifiability issue remains
unaddressed. The existence of the attribute hierarchy graph leads to degenerate
parameter space, and the potentially unknown structural matrix further
complicates the identifiability problem. This paper addresses this issue of
identifying the latent structure and model parameters underlying an HLAM. We
develop sufficient and necessary identifiability conditions. These results
directly and sharply characterize the different impacts on identifiability cast
by different attribute types in the graph. The proposed conditions not only
provide insights into diagnostic test designs under the attribute hierarchy,
but also serve as tools to assess the validity of an estimated HLAM.

    

### [[1910.04287] Deep localization of protein structures in fluorescence microscopy images](http://arxiv.org/abs/1910.04287)


  Accurate localization of proteins from fluorescence microscopy images is
challenging due to the inter-class similarities and intra-class disparities
introducing grave concerns in addressing multi-class classification problems.
Conventional machine learning-based image prediction pipelines rely heavily on
pre-processing such as normalization and segmentation followed by hand-crafted
feature extraction to identify useful, informative, and application-specific
features. Here, we demonstrate that deep learning-based pipelines can
effectively classify protein images from different datasets. We propose an
end-to-end Protein Localization Convolutional Neural Network (PLCNN) that
classifies protein images more accurately and reliably. PLCNN processes raw
imagery without involving any pre-processing steps and produces outputs without
any customization or parameter adjustment for a particular dataset.
Experimental analysis is performed on five benchmark datasets. PLCNN
consistently outperformed the existing state-of-the-art approaches from
traditional machine learning and deep architectures. This study highlights the
importance of deep learning for the analysis of fluorescence microscopy protein
imagery. The proposed deep pipeline can better guide drug designing procedures
in the pharmaceutical industry and open new avenues for researchers in
computational biology and bioinformatics.

    

### [[1910.08412] On the Sample Complexity of Actor-Critic Method for Reinforcement Learning with Function Approximation](http://arxiv.org/abs/1910.08412)


  Reinforcement learning, mathematically described by Markov Decision Problems,
may be approached either through dynamic programming or policy search.
Actor-critic algorithms combine the merits of both approaches by alternating
between steps to estimate the value function and policy gradient updates. Due
to the fact that the updates exhibit correlated noise and biased gradient
updates, only the asymptotic behavior of actor-critic is known by connecting
its behavior to dynamical systems. This work puts forth a new variant of
actor-critic that employs Monte Carlo rollouts during the policy search
updates, which results in controllable bias that depends on the number of
critic evaluations. As a result, we are able to provide for the first time the
convergence rate of actor-critic algorithms when the policy search step employs
policy gradient, agnostic to the choice of policy evaluation technique. In
particular, we establish conditions under which the sample complexity is
comparable to stochastic gradient method for non-convex problems or slower as a
result of the critic estimation error, which is the main complexity bottleneck.
These results hold in continuous state and action spaces with linear function
approximation for the value function. We then specialize these conceptual
results to the case where the critic is estimated by Temporal Difference,
Gradient Temporal Difference, and Accelerated Gradient Temporal Difference.
These learning rates are then corroborated on a navigation problem involving an
obstacle, providing insight into the interplay between optimization and
generalization in reinforcement learning.

    

### [[2005.07036] Infant Crying Detection in Real-World Environments](http://arxiv.org/abs/2005.07036)


  This paper addresses the problem of infant cry detection in real-world
settings. While most existing cry detection models have been tested with data
collected in controlled settings, the extent to which they generalize to noisy
and lived environments, i.e., people's homes, is unclear. In this paper, we
evaluated several established machine learning-based approaches as well as a
promising modeling strategy leveraging both deep spectrum and acoustic
features. This model was able to recognize crying events with F1 score 0.630
(Precision: 0.697, Recall: 0.567), showing improved external validity over
existing methods at cry detection in everyday real-world settings. As part of
our evaluation, we collected and annotated a novel dataset of infant crying
compiled from over 780 hours of high-quality labeled real-world audio data,
captured via recorders worn by infants in their homes, which we make publicly
available. Our findings confirmed that a cry detection model trained on in-lab
data underperforms when presented with real-world data (in-lab test F1: 0.656,
real-world test F1: 0.243), highlighting the value of our new dataset and
model.

    

### [[2008.00422] Rule-based Bayesian regression](http://arxiv.org/abs/2008.00422)


  We introduce a novel rule-based approach for handling regression problems.
The new methodology carries elements from two frameworks: (i) it provides
information about the uncertainty of the parameters of interest using Bayesian
inference, and (ii) it allows the incorporation of expert knowledge through
rule-based systems. The blending of those two different frameworks can be
particularly beneficial for various domains (e.g. engineering), where, even
though the significance of uncertainty quantification motivates a Bayesian
approach, there is no simple way to incorporate researcher intuition into the
model. We validate our models by applying them to synthetic applications: a
simple linear regression problem and two more complex structures based on
partial differential equations. Finally, we review the advantages of our
methodology, which include the simplicity of the implementation, the
uncertainty reduction due to the added information and, in some occasions, the
derivation of better point predictions, and we address limitations, mainly from
the computational complexity perspective, such as the difficulty in choosing an
appropriate algorithm and the added computational burden.

    

### [[2008.05913] Semi-supervised learning objectives as log-likelihoods in a generative model of data curation](http://arxiv.org/abs/2008.05913)


  We currently do not have an understanding of semi-supervised learning (SSL)
objectives such as pseudo-labelling and entropy minimization as
log-likelihoods, which precludes the development of e.g. Bayesian SSL. Here, we
note that benchmark image datasets such as CIFAR-10 are carefully curated, and
we formulate SSL objectives as a log-likelihood in a generative model of data
curation that was initially developed to explain the cold-posterior effect
(Aitchison 2020). SSL methods, from entropy minimization and pseudo-labelling,
to state-of-the-art techniques similar to FixMatch can be understood as
lower-bounds on our principled log-likelihood. We are thus able to give a
proof-of-principle for Bayesian SSL on toy data. Finally, our theory suggests
that SSL is effective in part due to the statistical patterns induced by data
curation. This provides an explanation of past results which show SSL performs
better on clean datasets without any "out of distribution" examples. Confirming
these results we find that SSL gave much larger performance improvements on
curated than on uncurated data, using matched curated and uncurated datasets
based on Galaxy Zoo 2.

    

### [[2012.04713] Classical symmetries and the Quantum Approximate Optimization Algorithm](http://arxiv.org/abs/2012.04713)


  We study the relationship between the Quantum Approximate Optimization
Algorithm (QAOA) and the underlying symmetries of the objective function to be
optimized. Our approach formalizes the connection between quantum symmetry
properties of the QAOA dynamics and the group of classical symmetries of the
objective function. The connection is general and includes but is not limited
to problems defined on graphs. We show a series of results exploring the
connection and highlight examples of hard problem classes where a nontrivial
symmetry subgroup can be obtained efficiently. In particular we show how
classical objective function symmetries lead to invariant measurement outcome
probabilities across states connected by such symmetries, independent of the
choice of algorithm parameters or number of layers. To illustrate the power of
the developed connection, we apply machine learning techniques towards
predicting QAOA performance based on symmetry considerations. We provide
numerical evidence that a small set of graph symmetry properties suffices to
predict the minimum QAOA depth required to achieve a target approximation ratio
on the MaxCut problem, in a practically important setting where QAOA parameter
schedules are constrained to be linear and hence easier to optimize.

    

### [[2012.07032] Neural network approaches to point lattice decoding](http://arxiv.org/abs/2012.07032)


  We characterize the complexity of the lattice decoding problem from a neural
network perspective. The notion of Voronoi-reduced basis is introduced to
restrict the space of solutions to a binary set. On the one hand, this problem
is shown to be equivalent to computing a continuous piecewise linear (CPWL)
function restricted to the fundamental parallelotope. On the other hand, it is
known that any function computed by a ReLU feed-forward neural network is CPWL.
As a result, we count the number of affine pieces in the CPWL decoding function
to characterize the complexity of the decoding problem. It is exponential in
the space dimension $n$, which induces shallow neural networks of exponential
size. For structured lattices we show that folding, a technique equivalent to
using a deep neural network, enables to reduce this complexity from exponential
in $n$ to polynomial in $n$. Regarding unstructured MIMO lattices, in contrary
to dense lattices many pieces in the CPWL decoding function can be neglected
for quasi-optimal decoding on the Gaussian channel. This makes the decoding
problem easier and it explains why shallow neural networks of reasonable size
are more efficient with this category of lattices (in low to moderate
dimensions).

    

### [[2101.07968] DynaComm: Accelerating Distributed CNN Training between Edges and Clouds through Dynamic Communication Scheduling](http://arxiv.org/abs/2101.07968)


  To reduce uploading bandwidth and address privacy concerns, deep learning at
the network edge has been an emerging topic. Typically, edge devices
collaboratively train a shared model using real-time generated data through the
Parameter Server framework. Although all the edge devices can share the
computing workloads, the distributed training processes over edge networks are
still time-consuming due to the parameters and gradients transmission
procedures between parameter servers and edge devices. Focusing on accelerating
distributed Convolutional Neural Networks (CNNs) training at the network edge,
we present DynaComm, a novel scheduler that dynamically decomposes each
transmission procedure into several segments to achieve optimal layer-wise
communications and computations overlapping during run-time. Through
experiments, we verify that DynaComm manages to achieve optimal layer-wise
scheduling for all cases compared to competing strategies while the model
accuracy remains untouched.

    

### [[2101.12010] Modeling Spatial Nonstationarity via Deformable Convolutions for Deep Traffic Flow Prediction](http://arxiv.org/abs/2101.12010)


  Deep neural networks are being increasingly used for short-term traffic flow
prediction, which can be generally categorized as convolutional (CNNs) or graph
neural networks (GNNs). CNNs are preferable for region-wise traffic prediction
by taking advantage of localized spatial correlations, whilst GNNs achieves
better performance for graph-structured traffic data. When applied to
region-wise traffic prediction, CNNs typically partition an underlying
territory into grid-like spatial units, and employ standard convolutions to
learn spatial dependence among the units. However, standard convolutions with
fixed geometric structures cannot fully model the nonstationary characteristics
of local traffic flows. To overcome the deficiency, we introduce deformable
convolution that augments the spatial sampling locations with additional
offsets, to enhance the modeling capability of spatial nonstationarity. On this
basis, we design a deep deformable convolutional residual network, namely
DeFlow-Net, that can effectively model global spatial dependence, local spatial
nonstationarity, and temporal periodicity of traffic flows. Furthermore, to
better fit with convolutions, we suggest to first aggregate traffic flows
according to pre-conceived regions or self-organized regions based on traffic
flows, then dispose to sequentially organized raster images for network input.
Extensive experiments on real-world traffic flows demonstrate that DeFlow-Net
outperforms GNNs and existing CNNs using standard convolutions, and spatial
partition by pre-conceived regions or self-organized regions further enhances
the performance. We also demonstrate the advantage of DeFlow-Net in maintaining
spatial autocorrelation, and reveal the impacts of partition shapes and scales
on deep traffic flow prediction.

    

### [[2102.03129] Integer Programming for Causal Structure Learning in the Presence of Latent Variables](http://arxiv.org/abs/2102.03129)


  The problem of finding an ancestral acyclic directed mixed graph (ADMG) that
represents the causal relationships between a set of variables is an important
area of research on causal inference. Most existing score-based structure
learning methods focus on learning directed acyclic graph (DAG) models without
latent variables. A number of score-based methods have recently been proposed
for the ADMG learning, yet they are heuristic in nature and do not guarantee an
optimal solution. We propose a novel exact score-based method that solves an
integer programming (IP) formulation and returns a score-maximizing ancestral
ADMG for a set of continuous variables that follow a multivariate Gaussian
distribution. We generalize the state-of-the-art IP model for DAG learning
problems and derive new classes of valid inequalities to formulate an IP model
for ADMG learning. Empirically, our model can be solved efficiently for
medium-sized problems and achieves better accuracy than state-of-the-art
score-based methods as well as benchmark constraint-based methods.

    

### [[2102.04399] Escaping Stochastic Traps with Aleatoric Mapping Agents](http://arxiv.org/abs/2102.04399)


  Exploration in environments with sparse rewards is difficult for artificial
agents. Curiosity driven learning -- using feed-forward prediction errors as
intrinsic rewards -- has achieved some success in these scenarios, but fails
when faced with action-dependent noise sources. We present aleatoric mapping
agents (AMAs), a neuroscience inspired solution modeled on the cholinergic
system of the mammalian brain. AMAs aim to explicitly ascertain which dynamics
of the environment are unpredictable, regardless of whether those dynamics are
induced by the actions of the agent. This is achieved by generating separate
forward predictions for the mean and variance of future states and reducing
intrinsic rewards for those transitions with high aleatoric variance. We show
AMAs are able to effectively circumvent action-dependent stochastic traps that
immobilise conventional curiosity driven agents. The code for all experiments
presented in this paper is open sourced:
this http URL.

    

### [[2103.00902] Manifold optimization for non-linear optimal transport problems](http://arxiv.org/abs/2103.00902)


  Optimal transport (OT) has recently found widespread interest in machine
learning. It allows to define novel distances between probability measures,
which have shown promise in several applications. In this work, we discuss how
to computationally approach general non-linear OT problems within the framework
of Riemannian manifold optimization. The basis of this is the manifold of
doubly stochastic matrices (and their generalization). Even though the manifold
geometry is not new, surprisingly, its usefulness for solving general
non-linear OT problems has not been popular. To this end, we specifically
discuss optimization-related ingredients that allow modeling the OT problem on
smooth Riemannian manifolds by exploiting the geometry of the search space. We
also discuss extensions where we reuse the developed optimization ingredients.
We make available the Manifold optimization-based Optimal Transport, or MOT,
repository with codes useful in solving OT problems in Python and Matlab. The
codes are available at \url{this https URL}.

    

### [[2103.14586] Understanding Robustness of Transformers for Image Classification](http://arxiv.org/abs/2103.14586)


  Deep Convolutional Neural Networks (CNNs) have long been the architecture of
choice for computer vision tasks. Recently, Transformer-based architectures
like Vision Transformer (ViT) have matched or even surpassed ResNets for image
classification. However, details of the Transformer architecture -- such as the
use of non-overlapping patches -- lead one to wonder whether these networks are
as robust. In this paper, we perform an extensive study of a variety of
different measures of robustness of ViT models and compare the findings to
ResNet baselines. We investigate robustness to input perturbations as well as
robustness to model perturbations. We find that when pre-trained with a
sufficient amount of data, ViT models are at least as robust as the ResNet
counterparts on a broad range of perturbations. We also find that Transformers
are robust to the removal of almost any single layer, and that while
activations from later layers are highly correlated with each other, they
nevertheless play an important role in classification.

    

### [[2103.15722] Transformer-based end-to-end speech recognition with residual Gaussian-based self-attention](http://arxiv.org/abs/2103.15722)


  Self-attention (SA), which encodes vector sequences according to their
pairwise similarity, is widely used in speech recognition due to its strong
context modeling ability. However, when applied to long sequence data, its
accuracy is reduced. This is caused by the fact that its weighted average
operator may lead to the dispersion of the attention distribution, which
results in the relationship between adjacent signals ignored. To address this
issue, in this paper, we introduce relative-position-awareness self-attention
(RPSA). It not only maintains the global-range dependency modeling ability of
self-attention, but also improves the localness modeling ability. Because the
local window length of the original RPSA is fixed and sensitive to different
test data, here we propose Gaussian-based self-attention (GSA) whose window
length is learnable and adaptive to the test data automatically. We further
generalize GSA to a new residual Gaussian self-attention (resGSA) for the
performance improvement. We apply RPSA, GSA, and resGSA to Transformer-based
speech recognition respectively. Experimental results on the AISHELL-1 Mandarin
speech recognition corpus demonstrate the effectiveness of the proposed
methods. For example, the resGSA-Transformer achieves a character error rate
(CER) of 5.86% on the test set, which is relative 7.8% lower than that of the
SA-Transformer. Although the performance of the proposed resGSA-Transformer is
only slightly better than that of the RPSA-Transformer, it does not have to
tune the window length manually.

    

### [[2103.15947] Federated Learning with Taskonomy for Non-IID Data](http://arxiv.org/abs/2103.15947)


  Classical federated learning approaches incur significant performance
degradation in the presence of non-IID client data. A possible direction to
address this issue is forming clusters of clients with roughly IID data. Most
solutions following this direction are iterative and relatively slow, also
prone to convergence issues in discovering underlying cluster formations. We
introduce federated learning with taskonomy (FLT) that generalizes this
direction by learning the task-relatedness between clients for more efficient
federated aggregation of heterogeneous data. In a one-off process, the server
provides the clients with a pretrained (and fine-tunable) encoder to compress
their data into a latent representation, and transmit the signature of their
data back to the server. The server then learns the task-relatedness among
clients via manifold learning, and performs a generalization of federated
averaging. FLT can flexibly handle a generic client relatedness graph, when
there are no explicit clusters of clients, as well as efficiently decompose it
into (disjoint) clusters for clustered federated learning. We demonstrate that
FLT not only outperforms the existing state-of-the-art baselines in non-IID
scenarios but also offers improved fairness across clients.

    

### [[2103.16940] Learning with Memory-based Virtual Classes for Deep Metric Learning](http://arxiv.org/abs/2103.16940)


  The core of deep metric learning (DML) involves learning visual similarities
in high-dimensional embedding space. One of the main challenges is to
generalize from seen classes of training data to unseen classes of test data.
Recent works have focused on exploiting past embeddings to increase the number
of instances for the seen classes. Such methods achieve performance improvement
via augmentation, while the strong focus on seen classes still remains. This
can be undesirable for DML, where training and test data exhibit entirely
different classes. In this work, we present a novel training strategy for DML
called MemVir. Unlike previous works, MemVir memorizes both embedding features
and class weights to utilize them as additional virtual classes. The
exploitation of virtual classes not only utilizes augmented information for
training but also alleviates a strong focus on seen classes for better
generalization. Moreover, we embed the idea of curriculum learning by slowly
adding virtual classes for a gradual increase in learning difficulty, which
improves the learning stability as well as the final performance. MemVir can be
easily applied to many existing loss functions without any modification.
Extensive experimental results on famous benchmarks demonstrate the superiority
of MemVir over state-of-the-art competitors. Code of MemVir is publicly
available.

    

### [[2104.12835] Less is more: Selecting informative and diverse subsets with balancing constraints](http://arxiv.org/abs/2104.12835)


  Deep learning has yielded extraordinary results in vision and natural
language processing, but this achievement comes at a cost. Most models require
enormous resources during training, both in terms of computation and in human
labeling effort. We show that we can identify informative and diverse subsets
of data that lead to deep learning models with similar performance as the ones
trained with the original dataset. Prior methods have exploited diversity and
uncertainty in submodular objective functions for choosing subsets. In addition
to these measures, we show that balancing constraints on predicted class labels
and decision boundaries are beneficial. We propose a novel formulation of these
constraints using matroids, an algebraic structure that generalizes linear
independence in vector spaces, and present an efficient greedy algorithm with
constant approximation guarantees. We outperform competing baselines on
standard classification datasets such as CIFAR-10, CIFAR-100, ImageNet, as well
as long-tailed datasets such as CIFAR-100-LT.

    

### [[2105.06535] Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies](http://arxiv.org/abs/2105.06535)


  Resting-state fMRI has been shown to provide surrogate biomarkers for the
analysis of various diseases. In addition, fMRI data helps in understanding the
brain's functional working during resting state and task-induced activity. To
improve the statistical power of biomarkers and the understanding mechanism of
the brain, pooling of multi-center studies has become increasingly popular. But
pooling the data from multiple sites introduces variations due to hardware,
software, and environment. In this paper, we look at the estimation problem of
hierarchical Sparsity Connectivity Patterns (hSCPs) in fMRI data acquired on
multiple sites. We introduce a simple yet effective matrix factorization based
formulation to reduce site-related effects while preserving biologically
relevant variations. We leverage adversarial learning in the unsupervised
regime to improve the reproducibility of the components. Experiments on
simulated datasets display that the proposed method can estimate components
with improved accuracy and reproducibility. We also demonstrate the improved
reproducibility of the components while preserving age-related variation on a
real dataset compiled from multiple sites.

    

### [[2105.09371] VOILA: Visual-Observation-Only Imitation Learning for Autonomous Navigation](http://arxiv.org/abs/2105.09371)


  While imitation learning for vision based autonomous mobile robot navigation
has recently received a great deal of attention in the research community,
existing approaches typically require state action demonstrations that were
gathered using the deployment platform. However, what if one cannot easily
outfit their platform to record these demonstration signals or worse yet the
demonstrator does not have access to the platform at all? Is imitation learning
for vision based autonomous navigation even possible in such scenarios? In this
work, we hypothesize that the answer is yes and that recent ideas from the
Imitation from Observation (IfO) literature can be brought to bear such that a
robot can learn to navigate using only ego centric video collected by a
demonstrator, even in the presence of viewpoint mismatch. To this end, we
introduce a new algorithm, Visual Observation only Imitation Learning for
Autonomous navigation (VOILA), that can successfully learn navigation policies
from a single video demonstration collected from a physically different agent.
We evaluate VOILA in the photorealistic AirSim simulator and show that VOILA
not only successfully imitates the expert, but that it also learns navigation
policies that can generalize to novel environments. Further, we demonstrate the
effectiveness of VOILA in a real world setting by showing that it allows a
wheeled Jackal robot to successfully imitate a human walking in an environment
using a video recorded using a mobile phone camera.

    

### [[2105.12272] Provable Representation Learning for Imitation with Contrastive Fourier Features](http://arxiv.org/abs/2105.12272)


  In imitation learning, it is common to learn a behavior policy to match an
unknown target policy via max-likelihood training on a collected set of target
demonstrations. In this work, we consider using offline experience datasets -
potentially far from the target distribution - to learn low-dimensional state
representations that provably accelerate the sample-efficiency of downstream
imitation learning. A central challenge in this setting is that the unknown
target policy itself may not exhibit low-dimensional behavior, and so there is
a potential for the representation learning objective to alias states in which
the target policy acts differently. Circumventing this challenge, we derive a
representation learning objective that provides an upper bound on the
performance difference between the target policy and a lowdimensional policy
trained with max-likelihood, and this bound is tight regardless of whether the
target policy itself exhibits low-dimensional structure. Moving to the
practicality of our method, we show that our objective can be implemented as
contrastive learning, in which the transition dynamics are approximated by
either an implicit energy-based model or, in some special cases, an implicit
linear model with representations given by random Fourier features. Experiments
on both tabular environments and high-dimensional Atari games provide
quantitative evidence for the practical benefits of our proposed objective.

    

### [[2105.14750] Active Hierarchical Exploration with Stable Subgoal Representation Learning](http://arxiv.org/abs/2105.14750)


  Goal-conditioned hierarchical reinforcement learning (GCHRL) provides a
promising approach to solving long-horizon tasks. Recently, its success has
been extended to more general settings by concurrently learning hierarchical
policies and subgoal representations. Although GCHRL possesses superior
exploration ability by decomposing tasks via subgoals, existing GCHRL methods
struggle in temporally extended tasks with sparse external rewards, since the
high-level policy learning relies on external rewards. As the high-level policy
selects subgoals in an online learned representation space, the dynamic change
of the subgoal space severely hinders effective high-level exploration. In this
paper, we propose a novel regularization that contributes to both stable and
efficient subgoal representation learning. Building upon the stable
representation, we design measures of novelty and potential for subgoals, and
develop an active hierarchical exploration strategy that seeks out new
promising subgoals and states without intrinsic rewards. Experimental results
show that our approach significantly outperforms state-of-the-art baselines in
continuous control tasks with sparse rewards.

    

### [[2106.01425] Gradient Assisted Learning](http://arxiv.org/abs/2106.01425)


  In distributed settings, collaborations between different entities, such as
financial institutions, medical centers, and retail markets, are crucial to
providing improved service and performance. However, the underlying entities
may have little interest in sharing their private data, proprietary models, and
objective functions. These privacy requirements have created new challenges for
collaboration. In this work, we propose Gradient Assisted Learning (GAL), a new
method for various entities to assist each other in supervised learning tasks
without sharing data, models, and objective functions. In this framework, all
participants collaboratively optimize the aggregate of local loss functions,
and each participant autonomously builds its own model by iteratively fitting
the gradients of the objective function. Experimental studies demonstrate that
Gradient Assisted Learning can achieve performance close to centralized
learning when all data, models, and objective functions are fully disclosed.

    

### [[2106.01432] SemiFL: Communication Efficient Semi-Supervised Federated Learning with Unlabeled Clients](http://arxiv.org/abs/2106.01432)


  Federated Learning allows training machine learning models by using the
computation and private data resources of many distributed clients such as
smartphones and IoT devices. Most existing works on Federated Learning (FL)
assume the clients have ground-truth labels. However, in many practical
scenarios, clients may be unable to label task-specific data, e.g., due to a
lack of expertise. This work considers a server that hosts a labeled dataset
and wishes to leverage clients with unlabeled data for supervised learning. We
propose a new Federated Learning framework referred to as SemiFL to address
Semi-Supervised Federated Learning (SSFL). In SemiFL, clients have completely
unlabeled data, while the server has a small amount of labeled data. SemiFL is
communication efficient since it separates the training of server-side
supervised data and client-side unsupervised data. We demonstrate several
strategies of SemiFL that enhance efficiency and prediction and develop
intuitions of why they work. In particular, we provide a theoretical
understanding of the use of strong data augmentation for Semi-Supervised
Learning (SSL), which can be interesting in its own right. Extensive empirical
evaluations demonstrate that our communication efficient method can
significantly improve the performance of a labeled server with unlabeled
clients. Moreover, we demonstrate that SemiFL can outperform many existing FL
results trained with fully supervised data, and perform competitively with the
state-of-the-art centralized SSL methods. For instance, in standard
communication efficient scenarios, our method can perform $93\%$ accuracy on
the CIFAR10 dataset with only $4000$ labeled samples at the server. Such
accuracy is only $2\%$ away from the result trained from $50000$ fully labeled
data, and it improves about $30\%$ upon existing SSFL methods in the
communication efficient setting.

    

### [[2106.03374] MixRL: Data Mixing Augmentation for Regression using Reinforcement Learning](http://arxiv.org/abs/2106.03374)


  Data augmentation is becoming essential for improving regression accuracy in
critical applications including manufacturing and finance. Existing techniques
for data augmentation largely focus on classification tasks and do not readily
apply to regression tasks. In particular, the recent Mixup techniques for
classification rely on the key assumption that linearity holds among training
examples, which is reasonable if the label space is discrete, but has
limitations when the label space is continuous as in regression. We show that
mixing examples that either have a large data or label distance may have an
increasingly-negative effect on model performance. Hence, we use the stricter
assumption that linearity only holds within certain data or label distances for
regression where the degree may vary by each example. We then propose MixRL, a
data augmentation meta learning framework for regression that learns for each
example how many nearest neighbors it should be mixed with for the best model
performance using a small validation set. MixRL achieves these objectives using
Monte Carlo policy gradient reinforcement learning. Our experiments conducted
both on synthetic and real datasets show that MixRL significantly outperforms
state-of-the-art data augmentation baselines. MixRL can also be integrated with
other classification Mixup techniques for better results.

    

### [[2106.03485] Representation mitosis in wide neural networks](http://arxiv.org/abs/2106.03485)


  Deep neural networks (DNNs) defy the classical bias-variance trade-off:
adding parameters to a DNN that interpolates its training data will typically
improve its generalization performance. Explaining the mechanism behind this
``benign overfitting'' in deep networks remains an outstanding challenge. Here,
we study the last hidden layer representations of various state-of-the-art
convolutional neural networks and find evidence for an underlying mechanism
that we call "representation mitosis": if the last hidden representation is
wide enough, its neurons tend to split into groups which carry identical
information, and differ from each other only by a statistically independent
noise. Like in a mitosis process, the number of such groups, or ``clones'',
increases linearly with the width of the layer, but only if the width is above
a critical value. We show that a key ingredient to activate mitosis is
continuing the training process until the training error is zero.

    

### [[2106.03693] Increase and Conquer: Training Graph Neural Networks on Growing Graphs](http://arxiv.org/abs/2106.03693)


  Graph neural networks (GNNs) use graph convolutions to exploit network
invariances and learn meaningful features from network data. However, on
large-scale graphs convolutions incur in high computational cost, leading to
scalability limitations. Leveraging the graphon -- the limit object of a graph
-- in this paper we consider the problem of learning a graphon neural network
(WNN) -- the limit object of a GNN -- by training GNNs on graphs sampled
Bernoulli from the graphon. Under smoothness conditions, we show that: (i) the
expected distance between the learning steps on the GNN and on the WNN
decreases asymptotically with the size of the graph, and (ii) when training on
a sequence of growing graphs, gradient descent follows the learning direction
of the WNN. Inspired by these results, we propose a novel algorithm to learn
GNNs on large-scale graphs that, starting from a moderate number of nodes,
successively increases the size of the graph during training. This algorithm is
benchmarked on both a recommendation system and a decentralized control problem
where it is shown to retain comparable performance, to its large-scale
counterpart, at a reduced computational cost.

    

### [[2106.04149] Understanding Generalized Label Smoothing when Learning with Noisy Labels](http://arxiv.org/abs/2106.04149)


  Label smoothing (LS) is an arising learning paradigm that uses the positively
weighted average of both the hard training labels and uniformly distributed
soft labels. It was shown that LS serves as a regularizer for training data
with hard labels and therefore improves the generalization of the model. Later
it was reported LS even helps with improving robustness when learning with
noisy labels. However, we observe that the advantage of LS vanishes when we
operate in a high label noise regime. Puzzled by the observation, we proceeded
to discover that several proposed learning-with-noisy-labels solutions in the
literature instead relate more closely to negative label smoothing (NLS), which
defines as using a negative weight to combine the hard and soft labels! We show
that NLS differs substantially from LS in their achieved model confidence. To
differentiate the two cases, we will call LS the positive label smoothing
(PLS), and this paper unifies PLS and NLS into generalized label smoothing
(GLS). We provide understandings for the properties of GLS when learning with
noisy labels. Among other established properties, we theoretically show NLS is
considered more beneficial when the label noise rates are high. We provide
extensive experimental results on multiple benchmarks to support our findings
too.

    

### [[2106.04221] Multi-output Gaussian Processes for Uncertainty-aware Recommender Systems](http://arxiv.org/abs/2106.04221)


  Recommender systems are often designed based on a collaborative filtering
approach, where user preferences are predicted by modelling interactions
between users and items. Many common approaches to solve the collaborative
filtering task are based on learning representations of users and items,
including simple matrix factorization, Gaussian process latent variable models,
and neural-network based embeddings. While matrix factorization approaches fail
to model nonlinear relations, neural networks can potentially capture such
complex relations with unprecedented predictive power and are highly scalable.
However, neither of them is able to model predictive uncertainties. In
contrast, Gaussian Process based models can generate a predictive distribution,
but cannot scale to large amounts of data. In this manuscript, we propose a
novel approach combining the representation learning paradigm of collaborative
filtering with multi-output Gaussian processes in a joint framework to generate
uncertainty-aware recommendations. We introduce an efficient strategy for model
training and inference, resulting in a model that scales to very large and
sparse datasets and achieves competitive performance in terms of classical
metrics quantifying the reconstruction error. In addition to accurately
predicting user preferences, our model also provides meaningful uncertainty
estimates about that prediction.

    

### [[2106.05710] DNN-Based Topology Optimisation: Spatial Invariance and Neural Tangent Kernel](http://arxiv.org/abs/2106.05710)


  We study the Solid Isotropic Material Penalisation (SIMP) method with a
density field generated by a fully-connected neural network, taking the
coordinates as inputs. In the large width limit, we show that the use of DNNs
leads to a filtering effect similar to traditional filtering techniques for
SIMP, with a filter described by the Neural Tangent Kernel (NTK). This filter
is however not invariant under translation, leading to visual artifacts and
non-optimal shapes. We propose two embeddings of the input coordinates, which
lead to (approximate) spatial invariance of the NTK and of the filter. We
empirically confirm our theoretical observations and study how the filter size
is affected by the architecture of the network. Our solution can easily be
applied to any other coordinates-based generation method.

    

### [[2106.14117] Graph Convolutional Memory using Topological Priors](http://arxiv.org/abs/2106.14117)


  Solving partially-observable Markov decision processes (POMDPs) is critical
when applying reinforcement learning to real-world problems, where agents have
an incomplete view of the world. We present graph convolutional memory (GCM),
the first hybrid memory model for solving POMDPs using reinforcement learning.
GCM uses either human-defined or data-driven topological priors to form graph
neighborhoods, combining them into a larger network topology using dynamic
programming. We query the graph using graph convolution, coalescing relevant
memories into a context-dependent belief. When used without human priors, GCM
performs similarly to state-of-the-art methods. When used with human priors,
GCM outperforms these methods on control, memorization, and navigation tasks
while using significantly fewer parameters.

    

### [[2106.14300] ASK: Adversarial Soft k-Nearest Neighbor Attack and Defense](http://arxiv.org/abs/2106.14300)


  K-Nearest Neighbor (kNN)-based deep learning methods have been applied to
many applications due to their simplicity and geometric interpretability.
However, the robustness of kNN-based classification models has not been
thoroughly explored and kNN attack strategies are underdeveloped. In this
paper, we propose an Adversarial Soft kNN (ASK) loss to both design more
effective kNN attack strategies and to develop better defenses against them.
Our ASK loss approach has two advantages. First, ASK loss can better
approximate the kNN's probability of classification error than objectives
proposed in previous works. Second, the ASK loss is interpretable: it preserves
the mutual information between the perturbed input and the in-class-reference
data. We use the ASK loss to generate a novel attack method called the
ASK-Attack (ASK-Atk), which shows superior attack efficiency and accuracy
degradation relative to previous kNN attacks. Based on the ASK-Atk, we then
derive an ASK-\underline{Def}ense (ASK-Def) method that optimizes the
worst-case training loss induced by ASK-Atk. Experiments on CIFAR-10 (ImageNet)
show that (i) ASK-Atk achieves $\geq 13\%$ ($\geq 13\%$) improvement in attack
success rate over previous kNN attacks, and (ii) ASK-Def outperforms the
conventional adversarial training method by $\geq 6.9\%$ ($\geq 3.5\%$) in
terms of robustness improvement.

    

### [[2107.05446] Source-Free Adaptation to Measurement Shift via Bottom-Up Feature Restoration](http://arxiv.org/abs/2107.05446)


  Source-free domain adaptation (SFDA) aims to adapt a model trained on
labelled data in a source domain to unlabelled data in a target domain without
access to the source-domain data during adaptation. Existing methods for SFDA
leverage entropy-minimization techniques which: (i) apply only to
classification; (ii) destroy model calibration; and (iii) rely on the source
model achieving a good level of feature-space class-separation in the target
domain. We address these issues for a particularly pervasive type of domain
shift called measurement shift -- characterized by a change in measurement
system -- which can be resolved by restoring the source features. In the source
domain, we store a lightweight and flexible approximation of the feature
distribution under the source data. In the target domain, we adapt the
feature-extractor such that the approximate feature distribution under the
target data realigns with that saved on the source. We call this method Feature
Restoration (FR) as it seeks to extract features with the same semantics from
the target domain as were previously extracted from the source, rather than
extracting new ones. We additionally propose Bottom-Up Feature Restoration
(BUFR) -- a bottom-up training scheme for FR which boosts performance by
preserving learnt structure in the later layers of a network. We demonstrate
that BUFR outperforms existing SFDA methods on real and synthetic data in terms
of accuracy, calibration, and data efficiency, while being less reliant on the
performance of the source model in the target domain.

    

### [[2108.10155] Construction Cost Index Forecasting: A Multi-feature Fusion Approach](http://arxiv.org/abs/2108.10155)


  The construction cost index is an important indicator of the construction
industry. Predicting CCI has important practical significance. This paper
combines information fusion with machine learning, and proposes a multi-feature
fusion (MFF) module for time series forecasting. Compared with the convolution
module, the MFF module is a module that extracts certain features. Experiments
have proved that the combination of MFF module and multi-layer perceptron has a
relatively good prediction effect. The MFF neural network model has high
prediction accuracy and efficient prediction efficiency. At the same time, MFF
continues to improve the potential of prediction accuracy, which is a study of
continuous attention.

    

### [[2109.06099] Uniform Generalization Bounds for Overparameterized Neural Networks](http://arxiv.org/abs/2109.06099)


  An interesting observation in artificial neural networks is their favorable
generalization error despite typically being extremely overparameterized. It is
well known that the classical statistical learning methods often result in
vacuous generalization errors in the case of overparameterized neural networks.
Adopting the recently developed Neural Tangent (NT) kernel theory, we prove
uniform generalization bounds for overparameterized neural networks in kernel
regimes, when the true data generating model belongs to the reproducing kernel
Hilbert space (RKHS) corresponding to the NT kernel. Importantly, our bounds
capture the exact error rates depending on the differentiability of the
activation functions. In order to establish these bounds, we propose the
information gain of the NT kernel as a measure of complexity of the learning
problem. Our analysis uses a Mercer decomposition of the NT kernel in the basis
of spherical harmonics and the decay rate of the corresponding eigenvalues. As
a byproduct of our results, we show the equivalence between the RKHS
corresponding to the NT kernel and its counterpart corresponding to the
Matérn family of kernels, showing the NT kernels induce a very general class
of models. We further discuss the implications of our analysis for some recent
results on the regret bounds for reinforcement learning and bandit algorithms,
which use overparameterized neural networks.

    

### [[2109.06165] CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation](http://arxiv.org/abs/2109.06165)


  Unsupervised domain adaptation (UDA) aims to transfer knowledge learned from
a labeled source domain to a different unlabeled target domain. Most existing
UDA methods focus on learning domain-invariant feature representation, either
from the domain level or category level, using convolution neural networks
(CNNs)-based frameworks. One fundamental problem for the category level based
UDA is the production of pseudo labels for samples in target domain, which are
usually too noisy for accurate domain alignment, inevitably compromising the
UDA performance. With the success of Transformer in various tasks, we find that
the cross-attention in Transformer is robust to the noisy input pairs for
better feature alignment, thus in this paper Transformer is adopted for the
challenging UDA task. Specifically, to generate accurate input pairs, we design
a two-way center-aware labeling algorithm to produce pseudo labels for target
samples. Along with the pseudo labels, a weight-sharing triple-branch
transformer framework is proposed to apply self-attention and cross-attention
for source/target feature learning and source-target domain alignment,
respectively. Such design explicitly enforces the framework to learn
discriminative domain-specific and domain-invariant representations
simultaneously. The proposed method is dubbed CDTrans (cross-domain
transformer), and it provides one of the first attempts to solve UDA tasks with
a pure transformer solution. Extensive experiments show that our proposed
method achieves the best performance on Office-Home, VisDA-2017, and DomainNet
datasets.

    

### [[2109.09510] Conditionally Parameterized, Discretization-Aware Neural Networks for Mesh-Based Modeling of Physical Systems](http://arxiv.org/abs/2109.09510)


  The numerical simulations of physical systems are heavily dependent on
mesh-based models. While neural networks have been extensively explored to
assist such tasks, they often ignore the interactions or hierarchical relations
between input features, and process them as concatenated mixtures. In this
work, we generalize the idea of conditional parametrization -- using trainable
functions of input parameters to generate the weights of a neural network, and
extend them in a flexible way to encode information critical to the numerical
simulations. Inspired by discretized numerical methods, choices of the
parameters include physical quantities and mesh topology features. The
functional relation between the modeled features and the parameters are built
into the network architecture. The method is implemented on different networks,
which are applied to several frontier scientific machine learning tasks,
including the discovery of unmodeled physics, super-resolution of coarse
fields, and the simulation of unsteady flows with chemical reactions. The
results show that the conditionally parameterized networks provide superior
performance compared to their traditional counterparts. A network architecture
named CP-GNet is also proposed as the first deep learning model capable of
standalone prediction of reacting flows on irregular meshes.

    

### [[2110.01406] MedPerf: Open Benchmarking Platform for Medical Artificial Intelligence using Federated Evaluation](http://arxiv.org/abs/2110.01406)


  Medical AI has tremendous potential to advance healthcare by supporting the
evidence-based practice of medicine, personalizing patient treatment, reducing
costs, and improving provider and patient experience. We argue that unlocking
this potential requires a systematic way to measure the performance of medical
AI models on large-scale heterogeneous data. To meet this need, we are building
MedPerf, an open framework for benchmarking machine learning in the medical
domain. MedPerf will enable federated evaluation in which models are securely
distributed to different facilities for evaluation, thereby empowering
healthcare organizations to assess and verify the performance of AI models in
an efficient and human-supervised process, while prioritizing privacy. We
describe the current challenges healthcare and AI communities face, the need
for an open platform, the design philosophy of MedPerf, its current
implementation status, and our roadmap. We call for researchers and
organizations to join us in creating the MedPerf open benchmarking platform.

    

### [[2110.03855] Hardware Functional Obfuscation With Ferroelectric Active Interconnects](http://arxiv.org/abs/2110.03855)


  Camouflaging gate techniques are typically used in hardware security to
prevent reverse engineering. Layout level camouflaging by adding dummy contacts
ensures some level of protection against extracting the correct netlist.
Threshold voltage manipulation for multi-functional logic with identical
layouts has also been introduced for functional obfuscation. All these
techniques are implemented at the expense of circuit-complexity and with
significant area, energy, and delay penalty. In this paper, we propose an
efficient hardware encryption technique with minimal complexity and overheads
based on ferroelectric field-effect transistor (FeFET) active interconnects.
The active interconnect provides run-time reconfigurable inverter-buffer logic
by utilizing the threshold voltage programmability of the FeFETs. Our method
utilizes only two FeFETs and an inverter to realize the masking function
compared to recent reconfigurable logic gate implementations using several
FeFETs and complex differential logic. We fabricate the proposed circuit and
demonstrate the functionality. Judicious placement of the proposed logic in the
IC makes it acts as a hardware encryption key and enables encoding and decoding
of the functional output without affecting the critical path timing delay.
Also, we achieve comparable encryption probability with a limited number of
encryption units. In addition, we show a peripheral programming scheme for
reconfigurable logic by reusing the existing scan chain logic, hence obviating
the need for specialized programming logic and circuitry for keybit
distribution. Our analysis shows an average encryption probability of 97.43\%
with an increase of 2.24\%/ 3.67\% delay for the most critical path/ sum of 100
critical paths delay for ISCAS85 benchmarks.

    

### [[2110.03901] Characterizing and Demystifying the Implicit Convolution Algorithm on Commercial Matrix-Multiplication Accelerators](http://arxiv.org/abs/2110.03901)


  Many of today's deep neural network accelerators, e.g., Google's TPU and
NVIDIA's tensor core, are built around accelerating the general matrix
multiplication (i.e., GEMM). However, supporting convolution on GEMM-based
accelerators is not trivial. The naive method explicitly lowers the convolution
to GEMM, commonly known as im2col, which introduces significant performance and
memory overhead. Existing implicit im2col algorithms require unscalable
hardware and are inefficient in supporting important convolution variants such
as strided convolution. In this paper, we propose a memory-efficient and
hardware-friendly implicit im2col algorithm used by Google's TPU, which
dynamically converts a convolution into a GEMM with practically zero
performance and memory overhead, fully unleashing the power of GEMM engines.
Through comprehensive experimental results, we quantitatively argue that this
algorithm has been adopted in commercial closed-source platforms, and we are
the first to describe its high-level idea and implementation details. Finally,
we show that our algorithm can also be generally applied to Nvidia's Tensor
Cores (TC), matching and out-performing the measured performance on TCs.

    

### [[2110.04101] TFix+: Self-configuring Hybrid Timeout Bug Fixing for Cloud Systems](http://arxiv.org/abs/2110.04101)


  Timeout bugs can cause serious availability and performance issues which are
often difficult to fix due to the lack of diagnostic information. Previous work
proposed solutions for fixing specific type of timeout-related performance
bugs. In this paper, we present TFix+, a self-configuring timeout bug fixing
framework for automatically correcting two major kinds of timeout bugs (i.e.,
misused timeout bugs and missing timeout bugs) with dynamic timeout value
predictions. TFix+ provides two new hybrid schemes for fixing misused and
missing timeout bugs, respectively. TFix+ further provides prediction-driven
timeout variable configuration based on runtime function tracing. We have
implemented a prototype of TFix+ and conducted experiments on 16 real world
timeout bugs. Our experimental results show that TFix+ can effectively fix 15
out of tested 16 timeout bugs.

    

### [[2110.03754] Process Extraction from Text: state of the art and challenges for the future](http://arxiv.org/abs/2110.03754)


  Automatic Process Discovery aims at developing algorithmic methodologies for
the extraction and elicitation of process models as described in data. While
Process Discovery from event-log data is a well established area, that has
already moved from research to concrete adoption in a mature manner, Process
Discovery from text is still a research area at an early stage of development,
which rarely scales to real world documents. In this paper we analyze, in a
comparative manner, reference state-of-the-art literature, especially for what
concerns the techniques used, the process elements extracted and the
evaluations performed. As a result of the analysis we discuss important
limitations that hamper the exploitation of recent Natural Language Processing
techniques in this field and we discuss fundamental limitations and challenges
for the future concerning the datasets, the techniques, the experimental
evaluations, and the pipelines currently adopted and to be developed in the
future.

    

### [[2110.03760] Design Strategy Network: A deep hierarchical framework to represent generative design strategies in complex action spaces](http://arxiv.org/abs/2110.03760)


  Generative design problems often encompass complex action spaces that may be
divergent over time, contain state-dependent constraints, or involve hybrid
(discrete and continuous) domains. To address those challenges, this work
introduces Design Strategy Network (DSN), a data-driven deep hierarchical
framework that can learn strategies over these arbitrary complex action spaces.
The hierarchical architecture decomposes every action decision into first
predicting a preferred spatial region in the design space and then outputting a
probability distribution over a set of possible actions from that region. This
framework comprises a convolutional encoder to work with image-based design
state representations, a multi-layer perceptron to predict a spatial region,
and a weight-sharing network to generate a probability distribution over
unordered set-based inputs of feasible actions. Applied to a truss design
study, the framework learns to predict the actions of human designers in the
study, capturing their truss generation strategies in the process. Results show
that DSNs significantly outperform non-hierarchical methods of policy
representation, demonstrating their superiority in complex action space
problems.

    

### [[2110.03858] ABCP: Automatic Block-wise and Channel-wise Network Pruning via Joint Search](http://arxiv.org/abs/2110.03858)


  Currently, an increasing number of model pruning methods are proposed to
resolve the contradictions between the computer powers required by the deep
learning models and the resource-constrained devices. However, most of the
traditional rule-based network pruning methods can not reach a sufficient
compression ratio with low accuracy loss and are time-consuming as well as
laborious. In this paper, we propose Automatic Block-wise and Channel-wise
Network Pruning (ABCP) to jointly search the block-wise and channel-wise
pruning action with deep reinforcement learning. A joint sample algorithm is
proposed to simultaneously generate the pruning choice of each residual block
and the channel pruning ratio of each convolutional layer from the discrete and
continuous search space respectively. The best pruning action taking both the
accuracy and the complexity of the model into account is obtained finally.
Compared with the traditional rule-based pruning method, this pipeline saves
human labor and achieves a higher compression ratio with lower accuracy loss.
Tested on the mobile robot detection dataset, the pruned YOLOv3 model saves
99.5% FLOPs, reduces 99.5% parameters, and achieves 37.3 times speed up with
only 2.8% mAP loss. The results of the transfer task on the sim2real detection
dataset also show that our pruned model has much better robustness performance.

    

### [[2110.03875] Dyn-Backdoor: Backdoor Attack on Dynamic Link Prediction](http://arxiv.org/abs/2110.03875)


  Dynamic link prediction (DLP) makes graph prediction based on historical
information. Since most DLP methods are highly dependent on the training data
to achieve satisfying prediction performance, the quality of the training data
is crucial. Backdoor attacks induce the DLP methods to make wrong prediction by
the malicious training data, i.e., generating a subgraph sequence as the
trigger and embedding it to the training data. However, the vulnerability of
DLP toward backdoor attacks has not been studied yet. To address the issue, we
propose a novel backdoor attack framework on DLP, denoted as Dyn-Backdoor.
Specifically, Dyn-Backdoor generates diverse initial-triggers by a generative
adversarial network (GAN). Then partial links of the initial-triggers are
selected to form a trigger set, according to the gradient information of the
attack discriminator in the GAN, so as to reduce the size of triggers and
improve the concealment of the attack. Experimental results show that
Dyn-Backdoor launches successful backdoor attacks on the state-of-the-art DLP
models with success rate more than 90%. Additionally, we conduct a possible
defense against Dyn-Backdoor to testify its resistance in defensive settings,
highlighting the needs of defenses for backdoor attacks on DLP.

    

### [[2110.03877] Diabetic Retinopathy Screening Using Custom-Designed Convolutional Neural Network](http://arxiv.org/abs/2110.03877)


  The prevalence of diabetic retinopathy (DR) has reached 34.6% worldwide and
is a major cause of blindness among middle-aged diabetic patients. Regular DR
screening using fundus photography helps detect its complications and prevent
its progression to advanced levels. As manual screening is time-consuming and
subjective, machine learning (ML) and deep learning (DL) have been employed to
aid graders. However, the existing CNN-based methods use either pre-trained CNN
models or a brute force approach to design new CNN models, which are not
customized to the complexity of fundus images. To overcome this issue, we
introduce an approach for custom-design of CNN models, whose architectures are
adapted to the structural patterns of fundus images and better represent the
DR-relevant features. It takes the leverage of k-medoid clustering, principal
component analysis (PCA), and inter-class and intra-class variations to
automatically determine the depth and width of a CNN model. The designed models
are lightweight, adapted to the internal structures of fundus images, and
encode the discriminative patterns of DR lesions. The technique is validated on
a local dataset from King Saud University Medical City, Saudi Arabia, and two
challenging benchmark datasets from Kaggle: EyePACS and APTOS2019. The
custom-designed models outperform the famous pre-trained CNN models like
ResNet152, Densnet121, and ResNeSt50 with a significant decrease in the number
of parameters and compete well with the state-of-the-art CNN-based DR screening
methods. The proposed approach is helpful for DR screening under diverse
clinical settings and referring the patients who may need further assessment
and treatment to expert ophthalmologists.

    

### [[2110.03895] ALL-IN-ONE: Multi-Task Learning BERT models for Evaluating Peer Assessments](http://arxiv.org/abs/2110.03895)


  Peer assessment has been widely applied across diverse academic fields over
the last few decades and has demonstrated its effectiveness. However, the
advantages of peer assessment can only be achieved with high-quality peer
reviews. Previous studies have found that high-quality review comments usually
comprise several features (e.g., contain suggestions, mention problems, use a
positive tone). Thus, researchers have attempted to evaluate peer-review
comments by detecting different features using various machine learning and
deep learning models. However, there is no single study that investigates using
a multi-task learning (MTL) model to detect multiple features simultaneously.
This paper presents two MTL models for evaluating peer-review comments by
leveraging the state-of-the-art pre-trained language representation models BERT
and DistilBERT. Our results demonstrate that BERT-based models significantly
outperform previous GloVe-based methods by around 6% in F1-score on tasks of
detecting a single feature, and MTL further improves performance while reducing
model size.

    

### [[2110.03912] Stereo Dense Scene Reconstruction and Accurate Laparoscope Localization for Learning-Based Navigation in Robot-Assisted Surgery](http://arxiv.org/abs/2110.03912)


  The computation of anatomical information and laparoscope position is a
fundamental block of robot-assisted surgical navigation in Minimally Invasive
Surgery (MIS). Recovering a dense 3D structure of surgical scene using visual
cues remains a challenge, and the online laparoscopic tracking mostly relies on
external sensors, which increases system complexity. In this paper, we propose
a learning-driven framework, in which an image-guided laparoscopic localization
with 3D reconstructions of complex anatomical structures is hereby achieved. To
reconstruct the 3D structure of the whole surgical environment, we first
fine-tune a learning-based stereoscopic depth perception method, which is
robust to the texture-less and variant soft tissues, for depth estimation.
Then, we develop a dense visual reconstruction algorithm to represent the scene
by surfels, estimate the laparoscope pose and fuse the depth data into a
unified reference coordinate for tissue reconstruction. To estimate poses of
new laparoscope views, we realize a coarse-to-fine localization method, which
incorporates our reconstructed 3D model. We evaluate the reconstruction method
and the localization module on three datasets, namely, the stereo
correspondence and reconstruction of endoscopic data (SCARED), the ex-vivo
phantom and tissue data collected with Universal Robot (UR) and Karl Storz
Laparoscope, and the in-vivo DaVinci robotic surgery dataset. Extensive
experiments have been conducted to prove the superior performance of our method
in 3D anatomy reconstruction and laparoscopic localization, which demonstrates
its potential implementation to surgical navigation system.

    

### [[2110.03939] Ranking Cost: Building An Efficient and Scalable Circuit Routing Planner with Evolution-Based Optimization](http://arxiv.org/abs/2110.03939)


  Circuit routing has been a historically challenging problem in designing
electronic systems such as very large-scale integration (VLSI) and printed
circuit boards (PCBs). The main challenge is that connecting a large number of
electronic components under specific design rules involves a very large search
space. Early solutions are typically designed with hard-coded heuristics, which
suffer from problems of non-optimal solutions and lack of flexibility for new
design needs. Although a few learning-based methods have been proposed
recently, they are typically cumbersome and hard to extend to large-scale
applications. In this work, we propose a new algorithm for circuit routing,
named Ranking Cost, which innovatively combines search-based methods (i.e., A*
algorithm) and learning-based methods (i.e., Evolution Strategies) to form an
efficient and trainable router. In our method, we introduce a new set of
variables called cost maps, which can help the A* router to find out proper
paths to achieve the global objective. We also train a ranking parameter, which
can produce the ranking order and further improve the performance of our
method. Our algorithm is trained in an end-to-end manner and does not use any
artificial data or human demonstration. In the experiments, we compare with the
sequential A* algorithm and a canonical reinforcement learning approach, and
results show that our method outperforms these baselines with higher
connectivity rates and better scalability.

    

### [[2110.03958] Social Recommendation with Self-Supervised Metagraph Informax Network](http://arxiv.org/abs/2110.03958)


  In recent years, researchers attempt to utilize online social information to
alleviate data sparsity for collaborative filtering, based on the rationale
that social networks offers the insights to understand the behavioral patterns.
However, due to the overlook of inter-dependent knowledge across items (e.g.,
categories of products), existing social recommender systems are insufficient
to distill the heterogeneous collaborative signals from both user and item
sides. In this work, we propose a Self-Supervised Metagraph Infor-max Network
(SMIN) which investigates the potential of jointly incorporating social- and
knowledge-aware relational structures into the user preference representation
for recommendation. To model relation heterogeneity, we design a
metapath-guided heterogeneous graph neural network to aggregate feature
embeddings from different types of meta-relations across users and items,
em-powering SMIN to maintain dedicated representations for multi-faceted user-
and item-wise dependencies. Additionally, to inject high-order collaborative
signals, we generalize the mutual information learning paradigm under the
self-supervised graph-based collaborative filtering. This endows the expressive
modeling of user-item interactive patterns, by exploring global-level
collaborative relations and underlying isomorphic transformation property of
graph topology. Experimental results on several real-world datasets demonstrate
the effectiveness of our SMIN model over various state-of-the-art
recommendation methods. We release our source code at
this https URL.

    

### [[2110.03966] Novel EEG-based BCIs for Elderly Rehabilitation Enhancement](http://arxiv.org/abs/2110.03966)


  The ageing process may lead to cognitive and physical impairments, which may
affect elderly everyday life. In recent years, the use of Brain Computer
Interfaces (BCIs) based on Electroencephalography (EEG) has revealed to be
particularly effective to promote and enhance rehabilitation procedures,
especially by exploiting motor imagery experimental paradigms. Moreover, BCIs
seem to increase patients' engagement and have proved to be reliable tools for
elderly overall wellness improvement. However, EEG signals usually present a
low signal-to-noise ratio and can be recorded for a limited time. Thus,
irrelevant information and faulty samples could affect the BCI performance.
Introducing a methodology that allows the extraction of informative components
from the EEG signal while maintaining its intrinsic characteristics, may
provide a solution to both the described issues: noisy data may be avoided by
having only relevant components and combining relevant components may represent
a good strategy to substitute the data without requiring long or repeated EEG
recordings. Moreover, substituting faulty trials may significantly improve the
classification performances of a BCI when translating imagined movement to
rehabilitation systems. To this end, in this work the EEG signal decomposition
by means of multivariate empirical mode decomposition is proposed to obtain its
oscillatory modes, called Intrinsic Mode Functions (IMFs). Subsequently, a
novel procedure for relevant IMF selection criterion based on the IMF
time-frequency representation and entropy is provided. After having verified
the reliability of the EEG signal reconstruction with the relevant IMFs only,
the relevant IMFs are combined to produce new artificial data and provide new
samples to use for BCI training.

    

### [[2110.03969] Graph Meta Network for Multi-Behavior Recommendation](http://arxiv.org/abs/2110.03969)


  Modern recommender systems often embed users and items into low-dimensional
latent representations, based on their observed interactions. In practical
recommendation scenarios, users often exhibit various intents which drive them
to interact with items with multiple behavior types (e.g., click,
tag-as-favorite, purchase). However, the diversity of user behaviors is ignored
in most of the existing approaches, which makes them difficult to capture
heterogeneous relational structures across different types of interactive
behaviors. Exploring multi-typed behavior patterns is of great importance to
recommendation systems, yet is very challenging because of two aspects: i) The
complex dependencies across different types of user-item interactions; ii)
Diversity of such multi-behavior patterns may vary by users due to their
personalized preference. To tackle the above challenges, we propose a
Multi-Behavior recommendation framework with Graph Meta Network to incorporate
the multi-behavior pattern modeling into a meta-learning paradigm. Our
developed MB-GMN empowers the user-item interaction learning with the
capability of uncovering type-dependent behavior representations, which
automatically distills the behavior heterogeneity and interaction diversity for
recommendations. Extensive experiments on three real-world datasets show the
effectiveness of MB-GMN by significantly boosting the recommendation
performance as compared to various state-of-the-art baselines. The source code
is available athttps://github.com/akaxlh/MB-GMN.

    

### [[2110.03982] Maximize the Exploration of Congeneric Semantics for Weakly Supervised Semantic Segmentation](http://arxiv.org/abs/2110.03982)


  With the increase in the number of image data and the lack of corresponding
labels, weakly supervised learning has drawn a lot of attention recently in
computer vision tasks, especially in the fine-grained semantic segmentation
problem. To alleviate human efforts from expensive pixel-by-pixel annotations,
our method focuses on weakly supervised semantic segmentation (WSSS) with
image-level tags, which are much easier to obtain. As a huge gap exists between
pixel-level segmentation and image-level labels, how to reflect the image-level
semantic information on each pixel is an important question. To explore the
congeneric semantic regions from the same class to the maximum, we construct
the patch-level graph neural network (P-GNN) based on the self-detected patches
from different images that contain the same class labels. Patches can frame the
objects as much as possible and include as little background as possible. The
graph network that is established with patches as the nodes can maximize the
mutual learning of similar objects. We regard the embedding vectors of patches
as nodes, and use transformer-based complementary learning module to construct
weighted edges according to the embedding similarity between different nodes.
Moreover, to better supplement semantic information, we propose
soft-complementary loss functions matched with the whole network structure. We
conduct experiments on the popular PASCAL VOC 2012 benchmarks, and our model
yields state-of-the-art performance.

    

### [[2110.03987] Knowledge-aware Coupled Graph Neural Network for Social Recommendation](http://arxiv.org/abs/2110.03987)


  Social recommendation task aims to predict users' preferences over items with
the incorporation of social connections among users, so as to alleviate the
sparse issue of collaborative filtering. While many recent efforts show the
effectiveness of neural network-based social recommender systems, several
important challenges have not been well addressed yet: (i) The majority of
models only consider users' social connections, while ignoring the
inter-dependent knowledge across items; (ii) Most of existing solutions are
designed for singular type of user-item interactions, making them infeasible to
capture the interaction heterogeneity; (iii) The dynamic nature of user-item
interactions has been less explored in many social-aware recommendation
techniques. To tackle the above challenges, this work proposes a
Knowledge-aware Coupled Graph Neural Network (KCGN) that jointly injects the
inter-dependent knowledge across items and users into the recommendation
framework. KCGN enables the high-order user- and item-wise relation encoding by
exploiting the mutual information for global graph structure awareness.
Additionally, we further augment KCGN with the capability of capturing dynamic
multi-typed user-item interactive patterns. Experimental studies on real-world
datasets show the effectiveness of our method against many strong baselines in
a variety of settings. Source codes are available at:
this https URL.

    

### [[2110.03996] Graph-Enhanced Multi-Task Learning of Multi-Level Transition Dynamics for Session-based Recommendation](http://arxiv.org/abs/2110.03996)


  Session-based recommendation plays a central role in a wide spectrum of
online applications, ranging from e-commerce to online advertising services.
However, the majority of existing session-based recommendation techniques
(e.g., attention-based recurrent network or graph neural network) are not
well-designed for capturing the complex transition dynamics exhibited with
temporally-ordered and multi-level inter-dependent relation structures. These
methods largely overlook the relation hierarchy of item transitional patterns.
In this paper, we propose a multi-task learning framework with Multi-level
Transition Dynamics (MTD), which enables the jointly learning of intra- and
inter-session item transition dynamics in automatic and hierarchical manner.
Towards this end, we first develop a position-aware attention mechanism to
learn item transitional regularities within individual session. Then, a
graph-structured hierarchical relation encoder is proposed to explicitly
capture the cross-session item transitions in the form of high-order
connectivities by performing embedding propagation with the global graph
context. The learning process of intra- and inter-session transition dynamics
are integrated, to preserve the underlying low- and high-level item
relationships in a common latent space. Extensive experiments on three
real-world datasets demonstrate the superiority of MTD as compared to
state-of-the-art baselines.

    

### [[2110.04000] Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation](http://arxiv.org/abs/2110.04000)


  Accurate user and item embedding learning is crucial for modern recommender
systems. However, most existing recommendation techniques have thus far focused
on modeling users' preferences over singular type of user-item interactions.
Many practical recommendation scenarios involve multi-typed user interactive
behaviors (e.g., page view, add-to-favorite and purchase), which presents
unique challenges that cannot be handled by current recommendation solutions.
In particular: i) complex inter-dependencies across different types of user
behaviors; ii) the incorporation of knowledge-aware item relations into the
multi-behavior recommendation framework; iii) dynamic characteristics of
multi-typed user-item interactions. To tackle these challenges, this work
proposes a Knowledge-Enhanced Hierarchical Graph Transformer Network (KHGT), to
investigate multi-typed interactive patterns between users and items in
recommender systems. Specifically, KHGT is built upon a graph-structured neural
architecture to i) capture type-specific behavior characteristics; ii)
explicitly discriminate which types of user-item interactions are more
important in assisting the forecasting task on the target behavior.
Additionally, we further integrate the graph attention layer with the temporal
encoding strategy, to empower the learned embeddings be reflective of both
dedicated multiplex user-item and item-item relations, as well as the
underlying interaction dynamics. Extensive experiments conducted on three
real-world datasets show that KHGT consistently outperforms many
state-of-the-art recommendation methods across various evaluation settings. Our
implementation code is available at this https URL.

    

### [[2110.04002] Multiplex Behavioral Relation Learning for Recommendation via Memory Augmented Transformer Network](http://arxiv.org/abs/2110.04002)


  Capturing users' precise preferences is of great importance in various
recommender systems (eg., e-commerce platforms), which is the basis of how to
present personalized interesting product lists to individual users. In spite of
significant progress has been made to consider relations between users and
items, most of the existing recommendation techniques solely focus on singular
type of user-item interactions. However, user-item interactive behavior is
often exhibited with multi-type (e.g., page view, add-to-favorite and purchase)
and inter-dependent in nature. The overlook of multiplex behavior relations can
hardly recognize the multi-modal contextual signals across different types of
interactions, which limit the feasibility of current recommendation methods. To
tackle the above challenge, this work proposes a Memory-Augmented Transformer
Networks (MATN), to enable the recommendation with multiplex behavioral
relational information, and joint modeling of type-specific behavioral context
and type-wise behavior inter-dependencies, in a fully automatic manner. In our
MATN framework, we first develop a transformer-based multi-behavior relation
encoder, to make the learned interaction representations be reflective of the
cross-type behavior relations. Furthermore, a memory attention network is
proposed to supercharge MATN capturing the contextual signals of different
types of behavior into the category-specific latent embedding space. Finally, a
cross-behavior aggregation component is introduced to promote the comprehensive
collaboration across type-aware interaction behavior representations, and
discriminate their inherent contributions in assisting recommendations.
Extensive experiments on two benchmark datasets and a real-world e-commence
user behavior data demonstrate significant improvements obtained by MATN over
baselines. Codes are available at: this https URL.

    

### [[2110.04032] Symbolic Register Automata for Complex Event Recognition and Forecasting](http://arxiv.org/abs/2110.04032)


  We propose an automaton model which is a combination of symbolic and register
automata, i.e., we enrich symbolic automata with memory. We call such automata
Symbolic Register Automata (SRA). SRA extend the expressive power of symbolic
automata, by allowing Boolean formulas to be applied not only to the last
element read from the input string, but to multiple elements, stored in their
registers. SRA also extend register automata, by allowing arbitrary Boolean
formulas, besides equality predicates. We study the closure properties of SRA
under union, intersection, concatenation, Kleene closure, complement and
determinization and show that SRA, contrary to symbolic automata, are not in
general closed under complement and they are not determinizable. However, they
are closed under these operations when a window operator, quintessential in
Complex Event Recognition, is used. We show how SRA can be used in Complex
Event Recognition in order to detect patterns upon streams of events, using our
framework that provides declarative and compositional semantics, and that
allows for a systematic treatment of such automata. We also show how the
behavior of SRA, as they consume streams of events, can be given a
probabilistic description with the help of prediction suffix trees. This allows
us to go one step beyond Complex Event Recognition to Complex Event
Forecasting, where, besides detecting complex patterns, we can also efficiently
forecast their occurrence.

    

### [[2110.04040] Towards Math-Aware Automated Classification and Similarity Search of Scientific Publications: Methods of Mathematical Content Representations](http://arxiv.org/abs/2110.04040)


  In this paper, we investigate mathematical content representations suitable
for the automated classification of and the similarity search in STEM documents
using standard machine learning algorithms: the Latent Dirichlet Allocation
(LDA) and the Latent Semantic Indexing (LSI). The methods are evaluated on a
subset of arXiv.org papers with the Mathematics Subject Classification (MSC) as
a reference classification and using the standard precision/recall/F1-measure
metrics. The results give insight into how different math representations may
influence the performance of the classification and similarity search tasks in
STEM repositories. Non-surprisingly, machine learning methods are able to grab
distributional semantics from textual tokens. A proper selection of weighted
tokens representing math may improve the quality of the results slightly. A
structured math representation that imitates successful text-processing
techniques with math is shown to yield better results than flat TeX tokens.

    

### [[2110.04041] Pick Your Battles: Interaction Graphs as Population-Level Objectives for Strategic Diversity](http://arxiv.org/abs/2110.04041)


  Strategic diversity is often essential in games: in multi-player games, for
example, evaluating a player against a diverse set of strategies will yield a
more accurate estimate of its performance. Furthermore, in games with
non-transitivities diversity allows a player to cover several winning
strategies. However, despite the significance of strategic diversity, training
agents that exhibit diverse behaviour remains a challenge. In this paper we
study how to construct diverse populations of agents by carefully structuring
how individuals within a population interact. Our approach is based on
interaction graphs, which control the flow of information between agents during
training and can encourage agents to specialise on different strategies,
leading to improved overall performance. We provide evidence for the importance
of diversity in multi-agent training and analyse the effect of applying
different interaction graphs on the training trajectories, diversity and
performance of populations in a range of games. This is an extended version of
the long abstract published at AAMAS.

    

### [[2110.04064] A Neural Anthropometer Learning from Body Dimensions Computed on Human 3D Meshes](http://arxiv.org/abs/2110.04064)


  Human shape estimation has become increasingly important both theoretically
and practically, for instance, in 3D mesh estimation, distance garment
production and computational forensics, to mention just a few examples. As a
further specialization, \emph{Human Body Dimensions Estimation} (HBDE) focuses
on estimating human body measurements like shoulder width or chest
circumference from images or 3D meshes usually using supervised learning
approaches. The main obstacle in this context is the data scarcity problem, as
collecting this ground truth requires expensive and difficult procedures. This
obstacle can be overcome by obtaining realistic human measurements from 3D
human meshes. However, a) there are no well established methods to calculate
HBDs from 3D meshes and b) there are no benchmarks to fairly compare results on
the HBDE task. Our contribution is twofold. On the one hand, we present a
method to calculate right and left arm length, shoulder width, and inseam
(crotch height) from 3D meshes with focus on potential medical, virtual try-on
and distance tailoring applications. On the other hand, we use four additional
body dimensions calculated using recently published methods to assemble a set
of eight body dimensions which we use as a supervision signal to our Neural
Anthropometer: a convolutional neural network capable of estimating these
dimensions. To assess the estimation, we train the Neural Anthropometer with
synthetic images of 3D meshes, from which we calculated the HBDs and observed
that the network's overall mean estimate error is $20.89$ mm (relative error of
2.84\%). The results we present are fully reproducible and establish a fair
baseline for research on the task of HBDE, therefore enabling the community
with a valuable method.

    

### [[2110.04065] Test-time Batch Statistics Calibration for Covariate Shift](http://arxiv.org/abs/2110.04065)


  Deep neural networks have a clear degradation when applying to the unseen
environment due to the covariate shift. Conventional approaches like domain
adaptation requires the pre-collected target data for iterative training, which
is impractical in real-world applications. In this paper, we propose to adapt
the deep models to the novel environment during inference. An previous solution
is test time normalization, which substitutes the source statistics in BN
layers with the target batch statistics. However, we show that test time
normalization may potentially deteriorate the discriminative structures due to
the mismatch between target batch statistics and source parameters. To this
end, we present a general formulation $\alpha$-BN to calibrate the batch
statistics by mixing up the source and target statistics for both alleviating
the domain shift and preserving the discriminative structures. Based on
$\alpha$-BN, we further present a novel loss function to form a unified test
time adaptation framework Core, which performs the pairwise class correlation
online optimization. Extensive experiments show that our approaches achieve the
state-of-the-art performance on total twelve datasets from three topics,
including model robustness to corruptions, domain generalization on image
classification and semantic segmentation. Particularly, our $\alpha$-BN
improves 28.4\% to 43.9\% on GTA5 $\rightarrow$ Cityscapes without any
training, even outperforms the latest source-free domain adaptation method.

    

### [[2110.04066] MToFNet: Object Anti-Spoofing with Mobile Time-of-Flight Data](http://arxiv.org/abs/2110.04066)


  In online markets, sellers can maliciously recapture others' images on
display screens to utilize as spoof images, which can be challenging to
distinguish in human eyes. To prevent such harm, we propose an anti-spoofing
method using the paired rgb images and depth maps provided by the mobile camera
with a Time-of-Fight sensor. When images are recaptured on display screens,
various patterns differing by the screens as known as the moiré patterns can
be also captured in spoof images. These patterns lead the anti-spoofing model
to be overfitted and unable to detect spoof images recaptured on unseen media.
To avoid the issue, we build a novel representation model composed of two
embedding models, which can be trained without considering the recaptured
images. Also, we newly introduce mToF dataset, the largest and most diverse
object anti-spoofing dataset, and the first to utilize ToF data. Experimental
results confirm that our model achieves robust generalization even across
unseen domains.

    

### [[2110.04077] Physical Context and Timing Aware Sequence Generating GANs](http://arxiv.org/abs/2110.04077)


  Generative Adversarial Networks (GANs) have shown remarkable successes in
generating realistic images and interpolating changes between images. Existing
models, however, do not take into account physical contexts behind images in
generating the images, which may cause unrealistic changes. Furthermore, it is
difficult to generate the changes at a specific timing and they often do not
match with actual changes. This paper proposes a novel GAN, named Physical
Context and Timing aware sequence generating GANs (PCTGAN), that generates an
image in a sequence at a specific timing between two images with considering
physical contexts behind them. Our method consists of three components: an
encoder, a generator, and a discriminator. The encoder estimates latent vectors
from the beginning and ending images, their timings, and a target timing. The
generator generates images and the physical contexts at the beginning, ending,
and target timing from the corresponding latent vectors. The discriminator
discriminates whether the generated images and contexts are real or not. In the
experiments, PCTGAN is applied to a data set of sequential changes of shapes in
die forging processes. We show that both timing and physical contexts are
effective in generating sequential images.

    

### [[2110.04089] Task Allocation for Multi-Robot Task and Motion Planning: a case for Object Picking in Cluttered Workspaces](http://arxiv.org/abs/2110.04089)


  We present an AND/OR graph-based, integrated multi-robot task and motion
planning approach which (i) performs task allocation coordinating the activity
of a given number of robots, and (ii) is capable of handling tasks which
involve an a priori unknown number of object re-arrangements, such as those
involved in retrieving objects from cluttered workspaces. Such situations may
arise, for example, in search and rescue scenarios, while locating/picking a
cluttered object of interest. The corresponding problem falls under the
category of planning in clutter. One of the challenges while planning in
clutter is that the number of object re-arrangements required to pick the
target object is not known beforehand, in general. Moreover, such tasks can be
decomposed in a variety of ways, since different cluttering object
re-arrangements are possible to reach the target object. In our approach, task
allocation and decomposition is achieved by maximizing a combined utility
function. The allocated tasks are performed by an integrated task and motion
planner, which is robust to the requirement of an unknown number of
re-arrangement tasks. We demonstrate our results with experiments in simulation
on two Franka Emika manipulators.

    

### [[2110.04095] A Mining Software Repository Extended Cookbook: Lessons learned from a literature review](http://arxiv.org/abs/2110.04095)


  The main purpose of Mining Software Repositories (MSR) is to discover the
latest enhancements and provide an insight into how to make improvements in a
software project. In light of it, this paper updates the MSR findings of the
original MSR Cookbook, by first conducting a systematic mapping study to elicit
and analyze the state-of-the-art, and then proposing an extended version of the
Cookbook. This extended Cookbook was built on four high-level themes, which
were derived from the analysis of a list of 112 selected studies. Hence, it was
used to consolidate the extended Cookbook as a contribution to practice and
research in the following areas by: 1) including studies published in all
available and relevant publication venues; 2) including and updating
recommendations in all four high-level themes, with an increase of 84% in
comments in this study when compared with the original MSR Cookbook; 3)
summarizing the tools employed for each high-level theme; and 4) providing
lessons learned for future studies. Thus, the extended Cookbook examined in
this work can support new research projects, as upgraded recommendations and
the lessons learned are available with the aid of samples and tools.

    

### [[2110.04147] The Impact of Visualizing Design Gradients for Human Designers](http://arxiv.org/abs/2110.04147)


  Mixed-initiative Procedural Content Generation (PCG) refers to tools or
systems in which a human designer works with an algorithm to produce game
content. This area of research remains relatively under-explored, with the
majority of mixed-initiative PCG level design systems using a common set of
search-based PCG algorithms. In this paper, we introduce a mixed-initiative
tool employing Exhaustive PCG (EPCG) for puzzle level design to further explore
mixed-initiative PCG. We run an online human subject study in which individuals
use the tool with an EPCG component turned on or off. Our analysis of the
results demonstrates that, although a majority of users did not prefer the
tool, it made the level design process significantly easier, and that the tool
impacted the subjects' design process. This paper describes the study results
and draws lessons for mixed-initiative PCG tool design.

    

### [[2110.04192] Explaining Reward Functions to Humans for Better Human-Robot Collaboration](http://arxiv.org/abs/2110.04192)


  Explainable AI techniques that describe agent reward functions can enhance
human-robot collaboration in a variety of settings. One context where human
understanding of agent reward functions is particularly beneficial is in the
value alignment setting. In the value alignment context, an agent aims to infer
a human's reward function through interaction so that it can assist the human
with their tasks. If the human can understand where gaps exist in the agent's
reward understanding, they will be able to teach more efficiently and
effectively, leading to quicker human-agent team performance improvements. In
order to support human collaborators in the value alignment setting and similar
contexts, it is first important to understand the effectiveness of different
reward explanation techniques in a variety of domains. In this paper, we
introduce a categorization of information modalities for reward explanation
techniques, suggest a suite of assessment techniques for human reward
understanding, and introduce four axes of domain complexity. We then propose an
experiment to study the relative efficacy of a broad set of reward explanation
techniques covering multiple modalities of information in a set of domains of
varying complexity.

    

### [[2110.04203] Toward a Human-Level Video Understanding Intelligence](http://arxiv.org/abs/2110.04203)


  We aim to develop an AI agent that can watch video clips and have a
conversation with human about the video story. Developing video understanding
intelligence is a significantly challenging task, and evaluation methods for
adequately measuring and analyzing the progress of AI agent are lacking as
well. In this paper, we propose the Video Turing Test to provide effective and
practical assessments of video understanding intelligence as well as
human-likeness evaluation of AI agents. We define a general format and
procedure of the Video Turing Test and present a case study to confirm the
effectiveness and usefulness of the proposed test.

    

### [[2110.04222] Inferring Offensiveness In Images From Natural Language Supervision](http://arxiv.org/abs/2110.04222)


  Probing or fine-tuning (large-scale) pre-trained models results in
state-of-the-art performance for many NLP tasks and, more recently, even for
computer vision tasks when combined with image data. Unfortunately, these
approaches also entail severe risks. In particular, large image datasets
automatically scraped from the web may contain derogatory terms as categories
and offensive images, and may also underrepresent specific classes.
Consequently, there is an urgent need to carefully document datasets and curate
their content. Unfortunately, this process is tedious and error-prone. We show
that pre-trained transformers themselves provide a methodology for the
automated curation of large-scale vision datasets. Based on human-annotated
examples and the implicit knowledge of a CLIP based model, we demonstrate that
one can select relevant prompts for rating the offensiveness of an image. In
addition to e.g. privacy violation and pornographic content previously
identified in ImageNet, we demonstrate that our approach identifies further
inappropriate and potentially offensive content.

    

### [[2110.04236] lambeq: An Efficient High-Level Python Library for Quantum NLP](http://arxiv.org/abs/2110.04236)


  We present lambeq, the first high-level Python library for Quantum Natural
Language Processing (QNLP). The open-source toolkit offers a detailed hierarchy
of modules and classes implementing all stages of a pipeline for converting
sentences to string diagrams, tensor networks, and quantum circuits ready to be
used on a quantum computer. lambeq supports syntactic parsing, rewriting and
simplification of string diagrams, ansatz creation and manipulation, as well as
a number of compositional models for preparing quantum-friendly representations
of sentences, employing various degrees of syntax sensitivity. We present the
generic architecture and describe the most important modules in detail,
demonstrating the usage with illustrative examples. Further, we test the
toolkit in practice by using it to perform a number of experiments on simple
NLP tasks, implementing both classical and quantum pipelines.

    

### [[2110.04249] How Can AI Recognize Pain and Express Empathy](http://arxiv.org/abs/2110.04249)


  Sensory and emotional experiences such as pain and empathy are relevant to
mental and physical health. The current drive for automated pain recognition is
motivated by a growing number of healthcare requirements and demands for social
interaction make it increasingly essential. Despite being a trending area, they
have not been explored in great detail. Over the past decades, behavioral
science and neuroscience have uncovered mechanisms that explain the
manifestations of pain. Recently, also artificial intelligence research has
allowed empathic machine learning methods to be approachable. Generally, the
purpose of this paper is to review the current developments for computational
pain recognition and artificial empathy implementation. Our discussion covers
the following topics: How can AI recognize pain from unimodality and
multimodality? Is it necessary for AI to be empathic? How can we create an AI
agent with proactive and reactive empathy? This article explores the challenges
and opportunities of real-world multimodal pain recognition from a
psychological, neuroscientific, and artificial intelligence perspective.
Finally, we identify possible future implementations of artificial empathy and
analyze how humans might benefit from an AI agent equipped with empathy.

    

### [[2110.04282] Field Extraction from Forms with Unlabeled Data](http://arxiv.org/abs/2110.04282)


  We propose a novel framework to conduct field extraction from forms with
unlabeled data. To bootstrap the training process, we develop a rule-based
method for mining noisy pseudo-labels from unlabeled forms. Using the
supervisory signal from the pseudo-labels, we extract a discriminative token
representation from a transformer-based model by modeling the interaction
between text in the form. To prevent the model from overfitting to label noise,
we introduce a refinement module based on a progressive pseudo-label ensemble.
Experimental results demonstrate the effectiveness of our framework.

    

### [[2110.04292] Toward a Visual Concept Vocabulary for GAN Latent Space](http://arxiv.org/abs/2110.04292)


  A large body of recent work has identified transformations in the latent
spaces of generative adversarial networks (GANs) that consistently and
interpretably transform generated images. But existing techniques for
identifying these transformations rely on either a fixed vocabulary of
pre-specified visual concepts, or on unsupervised disentanglement techniques
whose alignment with human judgments about perceptual salience is unknown. This
paper introduces a new method for building open-ended vocabularies of primitive
visual concepts represented in a GAN's latent space. Our approach is built from
three components: (1) automatic identification of perceptually salient
directions based on their layer selectivity; (2) human annotation of these
directions with free-form, compositional natural language descriptions; and (3)
decomposition of these annotations into a visual concept vocabulary, consisting
of distilled directions labeled with single words. Experiments show that
concepts learned with our approach are reliable and composable -- generalizing
across classes, contexts, and observers, and enabling fine-grained manipulation
of image style and content.

    

### [[1805.07429] Designing communication systems via iterative improvement: error correction coding with Bayes decoder and codebook optimized for source symbol error](http://arxiv.org/abs/1805.07429)


  In most error correction coding (ECC) frameworks, the typical error metric is
the bit error rate (BER) which measures the number of bit errors. For this
metric, the positions of the bits are not relevant to the decoding, and in many
noise models, not relevant to the BER either. In many applications this is
unsatisfactory as typically all bits are not equal and have different
significance. We consider the problem of bit error correction and mitigation
where bits in different positions have different importance. For error
correction, we look at ECC from a Bayesian perspective and introduce Bayes
estimators with general loss functions to take into account the bit
significance. We propose ECC schemes that optimize this error metric. As the
problem is highly nonlinear, traditional ECC construction techniques are not
applicable. Using exhaustive search is cost prohibitive, and thus we use
iterative improvement search techniques to find good codebooks. We optimize
both general codebooks and linear codes. We provide numerical experiments to
show that they can be superior to classical linear block codes such as Hamming
codes and decoding methods such as minimum distance decoding.
For error mitigation, we study the case where ECC is not possible or not
desirable, but significance aware encoding of information is still beneficial
in reducing the average error. We propose a novel number presentation format
suitable for emerging storage media where the noise magnitude is unknown and
possibly large and show that it has lower mean error than the traditional
number format.

    

### [[2011.04527] AAAI FSS-20: Artificial Intelligence in Government and Public Sector Proceedings](http://arxiv.org/abs/2011.04527)


  Proceedings of the AAAI Fall Symposium on Artificial Intelligence in
Government and Public Sector, Washington, DC, USA, November 13-14, 2020

    

### [[2102.04760] Improving Scene Graph Classification by Exploiting Knowledge from Texts](http://arxiv.org/abs/2102.04760)


  Training scene graph classification models requires a large amount of
annotated image data. Meanwhile, scene graphs represent relational knowledge
that can be modeled with symbolic data from texts or knowledge graphs. While
image annotation demands extensive labor, collecting textual descriptions of
natural scenes requires less effort. In this work, we investigate whether
textual scene descriptions can substitute for annotated image data. To this
end, we employ a scene graph classification framework that is trained not only
from annotated images but also from symbolic data. In our architecture, the
symbolic entities are first mapped to their correspondent image-grounded
representations and then fed into the relational reasoning pipeline. Even
though a structured form of knowledge, such as the form in knowledge graphs, is
not always available, we can generate it from unstructured texts using a
transformer-based language model. We show that by fine-tuning the
classification pipeline with the extracted knowledge from texts, we can achieve
~8x more accurate results in scene graph classification, ~3x in object
classification, and ~1.5x in predicate classification, compared to the
supervised baselines with only 1% of the annotated images.

    

### [[2103.13027] Unveiling the Power of Mixup for Stronger Classifiers](http://arxiv.org/abs/2103.13027)


  Mixup-based data augmentations have achieved great success as regularizers
for deep neural networks. However, existing methods rely on deliberately
handcrafted mixup policies, which ignore or oversell the semantic matching
between mixed samples and labels. Driven by their prior assumptions, early
methods attempt to smooth decision boundaries by random linear interpolation
while others focus on maximizing class-related information via offline saliency
optimization. As a result, the issue of label mismatch has not been well
addressed. Additionally, the optimization stability of mixup training is
constantly troubled by the label mismatch. To address these challenges, we
first reformulate mixup for supervised classification as two sub-tasks, mixup
sample generation and classification, then propose Automatic Mixup (AutoMix), a
revolutionary mixup framework. Specifically, a learnable lightweight Mix Block
(MB) with a cross-attention mechanism is proposed to generate a mixed sample by
modeling a fair relationship between the pair of samples under direct
supervision of the corresponding mixed label. Moreover, the proposed Momentum
Pipeline (MP) enhances training stability and accelerates convergence on top of
making the Mix Block fully trained end-to-end. Extensive experiments on five
popular classification benchmarks show that the proposed approach consistently
outperforms leading methods by a large margin.

    

### [[2110.03361] Multi-scale speaker embedding-based graph attention networks for speaker diarisation](http://arxiv.org/abs/2110.03361)


  The objective of this work is effective speaker diarisation using multi-scale
speaker embeddings. Typically, there is a trade-off between the ability to
recognise short speaker segments and the discriminative power of the embedding,
according to the segment length used for embedding extraction. To this end,
recent works have proposed the use of multi-scale embeddings where segments
with varying lengths are used. However, the scores are combined using a
weighted summation scheme where the weights are fixed after the training phase,
whereas the importance of segment lengths can differ with in a single session.
To address this issue, we present three key contributions in this paper: (1) we
propose graph attention networks for multi-scale speaker diarisation; (2) we
design scale indicators to utilise scale information of each embedding; (3) we
adapt the attention-based aggregation to utilise a pre-computed affinity matrix
from multi-scale embeddings. We demonstrate the effectiveness of our method in
various datasets where the speaker confusion which constitutes the primary
metric drops over 10% in average relative compared to the baseline.

    

### [[2110.03806] Toward a Theory of Programming Language and Reasoning Assistant Design: Minimizing Cognitive Load](http://arxiv.org/abs/2110.03806)


  Current approaches to making programming languages and reasoning assistants
more effective for people focus on leveraging feedback from users and on
evaluating the success of particular techniques. These approaches, although
helpful, may not result in systems that are as usable as possible, and may not
lead to general design principles. This paper advocates for leveraging theories
from cognitive science, using cognitive load theory as an example, to design
more effective programming languages and reasoning assistants. Development of
these theories may enable designers to create more effective programming
languages and reasoning assistants at lower cost.

    