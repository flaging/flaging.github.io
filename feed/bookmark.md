
## 2021-9-3

### [[2109.00672] Clock Skew Compensation Algorithm Immune to Floating-Point Precision Loss](http://arxiv.org/abs/2109.00672)


  We propose a novel clock skew compensation algorithm based on Bresenham's
line drawing algorithm. The proposed algorithm can avoid the effect of limited
floating-point precision (e.g., 32-bit single precision) on clock skew
compensation and thereby provide high-precision time synchronization even with
resource-constrained sensor nodes in wireless sensor networks.

    

### [[2109.00757] Energy-Efficient Multi-Orchestrator Mobile Edge Learning](http://arxiv.org/abs/2109.00757)


  Mobile Edge Learning (MEL) is a collaborative learning paradigm that features
distributed training of Machine Learning (ML) models over edge devices (e.g.,
IoT devices). In MEL, possible coexistence of multiple learning tasks with
different datasets may arise. The heterogeneity in edge devices' capabilities
will require the joint optimization of the learners-orchestrator association
and task allocation. To this end, we aim to develop an energy-efficient
framework for learners-orchestrator association and learning task allocation,
in which each orchestrator gets associated with a group of learners with the
same learning task based on their communication channel qualities and
computational resources, and allocate the tasks accordingly. Therein, a multi
objective optimization problem is formulated to minimize the total energy
consumption and maximize the learning tasks' accuracy. However, solving such
optimization problem requires centralization and the presence of the whole
environment information at a single entity, which becomes impractical in
large-scale systems. To reduce the solution complexity and to enable solution
decentralization, we propose lightweight heuristic algorithms that can achieve
near-optimal performance and facilitate the trade-offs between energy
consumption, accuracy, and solution complexity. Simulation results show that
the proposed approaches reduce the energy consumption significantly while
executing multiple learning tasks compared to recent state-of-the-art methods.

    

### [[2109.00833] Towards a Reference Architecture for Future Industrial Internet of Things Networks](http://arxiv.org/abs/2109.00833)


  With the continuing decrease of sensor technology prices as well as the
increase of communication and analytical capabilities of modern internet of
things devices, the continuously generated amount of data is constantly
growing. Various use cases show the untapped potential of this data for new
business models. However, conventional industrial IT networks of traditional
manufacturing companies can hardly meet the modern requirements emerging with
today's and future industrial internet of things applications. Outdated and
rigid network infrastructures are one of the main reasons for hesitant
innovation efforts and cross-organizational collaborations as well as the slow
adoption of modern business models by traditional manufacturing companies.
Following the design science research paradigm, our work contributes by
elaborating on a comprehensive list of requirements for future industrial
internet of things networks from a theoretical and practical perspective as
well as a proposed reference architecture acting as a blueprint for future
implementations.

    

### [[2109.00868] Load Balancing in Heterogeneous Server Clusters: Insights From a Product-Form Queueing Model](http://arxiv.org/abs/2109.00868)


  Efficiently exploiting servers in data centers requires performance analysis
methods that account not only for the stochastic nature of demand but also for
server heterogeneity. Although several recent works proved optimality results
for heterogeneity-aware variants of classical load-balancing algorithms in the
many-server regime, we still lack a fundamental understanding of the impact of
heterogeneity on performance in finite-size systems. In this paper, we consider
a load-balancing algorithm that leads to a product-form queueing model and can
therefore be analyzed exactly even when the number of servers is finite. We
develop new analytical methods that exploit its product-form stationary
distribution to understand the joint impact of the speeds and buffer lengths of
servers on performance. These analytical results are supported and complemented
by numerical evaluations that cover a large variety of scenarios.

    

### [[2109.00979] ROFA: An OFDMA system for Ultra-Reliable Wireless Industrial Networking](http://arxiv.org/abs/2109.00979)


  This paper proposes and demonstrates a PHY-layer design of a real-time
prototype that supports Ultra-Reliable Communication (URC) in wireless
infrastructure networks. The design makes use of Orthogonal Frequency Division
Multiple Access (OFDMA) as a means to achieve URC. Compared with Time-Division
Multiple Access (TDMA), OFDMA concentrates the transmit power to a narrower
bandwidth, resulting in higher effective SNR. Compared with Frequency-Division
Multiple Access (FDMA), OFDMA has higher spectrum efficiency thanks to the
smaller subcarrier spacing. Although OFDMA has been introduced in 802.11ax, the
purpose was to add flexibility in spectrum usage. Our Reliable OFDMA design,
referred to as ROFA, is a clean-slate design with a single goal of
ultra-reliable packet delivery. ROFA solves a number of key challenges to
ensure the ultra-reliability: (1) a downlink-coordinated time-synchronization
mechanism to synchronize the uplink transmission of users, with at most $0.1us$
timing offset; (2) an "STF-free" packet reception synchronization method that
makes use of the property of synchronous systems to avoid packet misdetection;
and (3) an uplink precoding mechanism to reduce the CFOs between users and the
AP to a negligible level. We implemented ROFA on the Universal Software Radio
Peripheral (USRP) SDR platform with real-time signal processing. Extensive
experimental results show that ROFA can achieve ultra-reliable packet delivery
($PER<10^5$) with $11.5dB$ less transmit power compared with OFDM-TDMA when
they use $3$ and $52$ subcarriers respectively.

    

### [[2109.01104] The Far Side of DNS Amplification: Tracing the DDoS Attack Ecosystem from the Internet Core](http://arxiv.org/abs/2109.01104)


  In this paper, we shed new light on the DNS amplification ecosystem, by
studying complementary data sources, bolstered by orthogonal methodologies.
First, we introduce a passive attack detection method for the Internet core,
i.e., at Internet eXchange Points (IXPs). Surprisingly, IXPs and honeypots
observe mostly disjoint sets of attacks: 96% of IXP-inferred attacks were
invisible to a sizable honeypot platform. Second, we assess the effectiveness
of observed DNS attacks by studying IXP traces jointly with diverse data from
independent measurement infrastructures. We find that attackers efficiently
detect new reflectors and purposefully rotate between them. At the same time,
we reveal that attackers are a small step removed from bringing about
significantly higher amplification factors (14x). Third, we identify and
fingerprint a major attack entity by studying patterns in attack traces. We
show that this entity dominates the DNS amplification ecosystem by carrying out
59% of the attacks, and provide an in-depth analysis of its behavior over time.
Finally, our results reveal that operators of various .gov names adhere to a
DNSSEC key rollover scheme, which exacerbates amplification potential, and
which we can verifiably connect to misuses and attacker decision-making.

    

### [[2109.01106] QUICsand: Quantifying QUIC Reconnaissance Scans and DoS Flooding Events](http://arxiv.org/abs/2109.01106)


  In this paper, we present first measurements of Internet background radiation
originating from the emerging transport protocol QUIC. Our analysis is based on
the UCSD network telescope, correlated with active measurements.
We find that research projects dominate the QUIC scanning ecosystem but also
discover traffic from non-benign sources. We argue that although QUIC has been
carefully designed to restrict reflective amplification attacks, the QUIC
handshake is prone to resource exhaustion attacks, similar to TCP SYN floods.
We confirm this conjecture by showing how this attack vector is already
exploited in multi-vector attacks: On average, the Internet is exposed to four
QUIC floods per hour and half of these attacks occur concurrently with other
common attack types such as TCP/ICMP floods.

    

### [[2109.00516] Multistage Pruning of CNN Based ECG Classifiers for Edge Devices](http://arxiv.org/abs/2109.00516)


  Using smart wearable devices to monitor patients electrocardiogram (ECG) for
real-time detection of arrhythmias can significantly improve healthcare
outcomes. Convolutional neural network (CNN) based deep learning has been used
successfully to detect anomalous beats in ECG. However, the computational
complexity of existing CNN models prohibits them from being implemented in
low-powered edge devices. Usually, such models are complex with lots of model
parameters which results in large number of computations, memory, and power
usage in edge devices. Network pruning techniques can reduce model complexity
at the expense of performance in CNN models. This paper presents a novel
multistage pruning technique that reduces CNN model complexity with negligible
loss in performance compared to existing pruning techniques. An existing CNN
model for ECG classification is used as a baseline reference. At 60% sparsity,
the proposed technique achieves 97.7% accuracy and an F1 score of 93.59% for
ECG classification tasks. This is an improvement of 3.3% and 9% for accuracy
and F1 Score respectively, compared to traditional pruning with fine-tuning
approach. Compared to the baseline model, we also achieve a 60.4% decrease in
run-time complexity.

    

### [[2109.00520] The Role of Explainability in Assuring Safety of Machine Learning in Healthcare](http://arxiv.org/abs/2109.00520)


  Established approaches to assuring safety-critical systems and software are
difficult to apply to systems employing machine learning (ML). In many cases,
ML is used on ill-defined problems, e.g. optimising sepsis treatment, where
there is no clear, pre-defined specification against which to assess validity.
This problem is exacerbated by the "opaque" nature of ML where the learnt model
is not amenable to human scrutiny. Explainable AI methods have been proposed to
tackle this issue by producing human-interpretable representations of ML models
which can help users to gain confidence and build trust in the ML system.
However, there is not much work explicitly investigating the role of
explainability for safety assurance in the context of ML development. This
paper identifies ways in which explainable AI methods can contribute to safety
assurance of ML-based systems. It then uses a concrete ML-based clinical
decision support system, concerning weaning of patients from mechanical
ventilation, to demonstrate how explainable AI methods can be employed to
produce evidence to support safety assurance. The results are also represented
in a safety argument to show where, and in what way, explainable AI methods can
contribute to a safety case. Overall, we conclude that explainable AI methods
have a valuable role in safety assurance of ML-based systems in healthcare but
that they are not sufficient in themselves to assure safety.

    

### [[2109.00521] Don't Discard All the Biased Instances: Investigating a Core Assumption in Dataset Bias Mitigation Techniques](http://arxiv.org/abs/2109.00521)


  Existing techniques for mitigating dataset bias often leverage a biased model
to identify biased instances. The role of these biased instances is then
reduced during the training of the main model to enhance its robustness to
out-of-distribution data. A common core assumption of these techniques is that
the main model handles biased instances similarly to the biased model, in that
it will resort to biases whenever available. In this paper, we show that this
assumption does not hold in general. We carry out a critical investigation on
two well-known datasets in the domain, MNLI and FEVER, along with two biased
instance detection methods, partial-input and limited-capacity models. Our
experiments show that in around a third to a half of instances, the biased
model is unable to predict the main model's behavior, highlighted by the
significantly different parts of the input on which they base their decisions.
Based on a manual validation, we also show that this estimate is highly in line
with human interpretation. Our findings suggest that down-weighting of
instances detected by bias detection methods, which is a widely-practiced
procedure, is an unnecessary waste of training data. We release our code to
facilitate reproducibility and future research.

    

### [[2109.00522] Conditional Extreme Value Theory for Open Set Video Domain Adaptation](http://arxiv.org/abs/2109.00522)


  With the advent of media streaming, video action recognition has become
progressively important for various applications, yet at the high expense of
requiring large-scale data labelling. To overcome the problem of expensive data
labelling, domain adaptation techniques have been proposed that transfers
knowledge from fully labelled data (i.e., source domain) to unlabelled data
(i.e., target domain). The majority of video domain adaptation algorithms are
proposed for closed-set scenarios in which all the classes are shared among the
domains. In this work, we propose an open-set video domain adaptation approach
to mitigate the domain discrepancy between the source and target data, allowing
the target data to contain additional classes that do not belong to the source
domain. Different from previous works, which only focus on improving accuracy
for shared classes, we aim to jointly enhance the alignment of shared classes
and recognition of unknown samples. Towards this goal, class-conditional
extreme value theory is applied to enhance the unknown recognition.
Specifically, the entropy values of target samples are modelled as generalised
extreme value distributions, which allows separating unknown samples lying in
the tail of the distribution. To alleviate the negative transfer issue, weights
computed by the distance from the sample entropy to the threshold are leveraged
in adversarial learning in the sense that confident source and target samples
are aligned, and unconfident samples are pushed away. The proposed method has
been thoroughly evaluated on both small-scale and large-scale cross-domain
video datasets and achieved the state-of-the-art performance.

    

### [[2109.00523] Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification](http://arxiv.org/abs/2109.00523)


  Data augmentation aims to enrich training samples for alleviating the
overfitting issue in low-resource or class-imbalanced situations. Traditional
methods first devise task-specific operations such as Synonym Substitute, then
preset the corresponding parameters such as the substitution rate artificially,
which require a lot of prior knowledge and are prone to fall into the
sub-optimum. Besides, the number of editing operations is limited in the
previous methods, which decreases the diversity of the augmented data and thus
restricts the performance gain. To overcome the above limitations, we propose a
framework named Text AutoAugment (TAA) to establish a compositional and
learnable paradigm for data augmentation. We regard a combination of various
operations as an augmentation policy and utilize an efficient Bayesian
Optimization algorithm to automatically search for the best policy, which
substantially improves the generalization capability of models. Experiments on
six benchmark datasets show that TAA boosts classification accuracy in
low-resource and class-imbalanced regimes by an average of 8.8% and 9.7%,
respectively, outperforming strong baselines.

    

### [[2109.00524] On the Limits of Pseudo Ground Truth in Visual Camera Re-localisation](http://arxiv.org/abs/2109.00524)


  Benchmark datasets that measure camera pose accuracy have driven progress in
visual re-localisation research. To obtain poses for thousands of images, it is
common to use a reference algorithm to generate pseudo ground truth. Popular
choices include Structure-from-Motion (SfM) and
Simultaneous-Localisation-and-Mapping (SLAM) using additional sensors like
depth cameras if available. Re-localisation benchmarks thus measure how well
each method replicates the results of the reference algorithm. This begs the
question whether the choice of the reference algorithm favours a certain family
of re-localisation methods. This paper analyzes two widely used re-localisation
datasets and shows that evaluation outcomes indeed vary with the choice of the
reference algorithm. We thus question common beliefs in the re-localisation
literature, namely that learning-based scene coordinate regression outperforms
classical feature-based methods, and that RGB-D-based methods outperform
RGB-based methods. We argue that any claims on ranking re-localisation methods
should take the type of the reference algorithm, and the similarity of the
methods to the reference algorithm, into account.

    

### [[2109.00525] Catastrophic Interference in Reinforcement Learning: A Solution Based on Context Division and Knowledge Distillation](http://arxiv.org/abs/2109.00525)


  The powerful learning ability of deep neural networks enables reinforcement
learning (RL) agents to learn competent control policies directly from
high-dimensional and continuous environments. In theory, to achieve stable
performance, neural networks assume i.i.d. inputs, which unfortunately does no
hold in the general RL paradigm where the training data is temporally
correlated and non-stationary. This issue may lead to the phenomenon of
"catastrophic interference" and the collapse in performance as later training
is likely to overwrite and interfer with previously learned policies. In this
paper, we introduce the concept of "context" into single-task RL and develop a
novel scheme, termed as Context Division and Knowledge Distillation (CDaKD)
driven RL, to divide all states experienced during training into a series of
contexts. Its motivation is to mitigate the challenge of aforementioned
catastrophic interference in deep RL, thereby improving the stability and
plasticity of RL models. At the heart of CDaKD is a value function,
parameterized by a neural network feature extractor shared across all contexts,
and a set of output heads, each specializing on an individual context. In
CDaKD, we exploit online clustering to achieve context division, and
interference is further alleviated by a knowledge distillation regularization
term on the output layers for learned contexts. In addition, to effectively
obtain the context division in high-dimensional state spaces (e.g., image
inputs), we perform clustering in the lower-dimensional representation space of
a randomly initialized convolutional encoder, which is fixed throughout
training. Our results show that, with various replay memory capacities, CDaKD
can consistently improve the performance of existing RL algorithms on classic
OpenAI Gym tasks and the more complex high-dimensional Atari tasks, incurring
only moderate computational overhead.

    

### [[2109.00526] Physics-informed Neural Network for Nonlinear Dynamics in Fiber Optics](http://arxiv.org/abs/2109.00526)


  A physics-informed neural network (PINN) that combines deep learning with
physics is studied to solve the nonlinear Schrödinger equation for learning
nonlinear dynamics in fiber optics. We carry out a systematic investigation and
comprehensive verification on PINN for multiple physical effects in optical
fibers, including dispersion, self-phase modulation, and higher-order nonlinear
effects. Moreover, both special case (soliton propagation) and general case
(multi-pulse propagation) are investigated and realized with PINN. In the
previous studies, the PINN was mainly effective for single scenario. To
overcome this problem, the physical parameters (pulse peak power and amplitudes
of sub-pulses) are hereby embedded as additional input parameter controllers,
which allow PINN to learn the physical constraints of different scenarios and
perform good generalizability. Furthermore, PINN exhibits better performance
than the data-driven neural network using much less data, and its computational
complexity (in terms of number of multiplications) is much lower than that of
the split-step Fourier method. The results report here show that the PINN is
not only an effective partial differential equation solver, but also a
prospective technique to advance the scientific computing and automatic
modeling in fiber optics.

    

### [[2109.00527] Boosting Search Engines with Interactive Agents](http://arxiv.org/abs/2109.00527)


  Can machines learn to use a search engine as an interactive tool for finding
information? That would have far reaching consequences for making the world's
knowledge more accessible. This paper presents first steps in designing agents
that learn meta-strategies for contextual query refinements. Our approach uses
machine reading to guide the selection of refinement terms from aggregated
search results. Agents are then empowered with simple but effective search
operators to exert fine-grained and transparent control over queries and search
results. We develop a novel way of generating synthetic search sessions, which
leverages the power of transformer-based generative language models through
(self-)supervised learning. We also present a reinforcement learning agent with
dynamically constrained actions that can learn interactive search strategies
completely from scratch. In both cases, we obtain significant improvements over
one-shot search with a strong information retrieval baseline. Finally, we
provide an in-depth analysis of the learned search policies.

    

### [[2109.00528] Wasserstein GANs with Gradient Penalty Compute Congested Transport](http://arxiv.org/abs/2109.00528)


  Wasserstein GANs with Gradient Penalty (WGAN-GP) are an extremely popular
method for training generative models to produce high quality synthetic data.
While WGAN-GP were initially developed to calculate the Wasserstein 1 distance
between generated and real data, recent works (e.g. Stanczuk et al. (2021))
have provided empirical evidence that this does not occur, and have argued that
WGAN-GP perform well not in spite of this issue, but because of it. In this
paper we show for the first time that WGAN-GP compute the minimum of a
different optimal transport problem, the so-called congested transport (Carlier
et al. (2008)). Congested transport determines the cost of moving one
distribution to another under a transport model that penalizes congestion. For
WGAN-GP, we find that the congestion penalty has a spatially varying component
determined by the sampling strategy used in Gulrajani et al. (2017) which acts
like a local speed limit, making congestion cost less in some regions than
others. This aspect of the congested transport problem is new in that the
congestion penalty turns out to be unbounded and depend on the distributions to
be transported, and so we provide the necessary mathematical proofs for this
setting. We use our discovery to show that the gradients of solutions to the
optimization problem in WGAN-GP determine the time averaged momentum of optimal
mass flow. This is in contrast to the gradients of Kantorovich potentials for
the Wasserstein 1 distance, which only determine the normalized direction of
flow. This may explain, in support of Stanczuk et al. (2021), the success of
WGAN-GP, since the training of the generator is based on these gradients.

    

### [[2109.00530] A Gradient Sampling Algorithm for Stratified Maps with Applications to Topological Data Analysis](http://arxiv.org/abs/2109.00530)


  We introduce a novel gradient descent algorithm extending the well-known
Gradient Sampling methodology to the class of stratifiably smooth objective
functions, which are defined as locally Lipschitz functions that are smooth on
some regular pieces-called the strata-of the ambient Euclidean space. For this
class of functions, our algorithm achieves a sub-linear convergence rate. We
then apply our method to objective functions based on the (extended) persistent
homology map computed over lower-star filters, which is a central tool of
Topological Data Analysis. For this, we propose an efficient exploration of the
corresponding stratification by using the Cayley graph of the permutation
group. Finally, we provide benchmark and novel topological optimization
problems, in order to demonstrate the utility and applicability of our
framework.

    

### [[2109.00531] Under-bagging Nearest Neighbors for Imbalanced Classification](http://arxiv.org/abs/2109.00531)


  In this paper, we propose an ensemble learning algorithm called
\textit{under-bagging $k$-nearest neighbors} (\textit{under-bagging $k$-NN})
for imbalanced classification problems. On the theoretical side, by developing
a new learning theory analysis, we show that with properly chosen parameters,
i.e., the number of nearest neighbors $k$, the expected sub-sample size $s$,
and the bagging rounds $B$, optimal convergence rates for under-bagging $k$-NN
can be achieved under mild assumptions w.r.t.~the arithmetic mean (AM) of
recalls. Moreover, we show that with a relatively small $B$, the expected
sub-sample size $s$ can be much smaller than the number of training data $n$ at
each bagging round, and the number of nearest neighbors $k$ can be reduced
simultaneously, especially when the data are highly imbalanced, which leads to
substantially lower time complexity and roughly the same space complexity. On
the practical side, we conduct numerical experiments to verify the theoretical
results on the benefits of the under-bagging technique by the promising AM
performance and efficiency of our proposed algorithm.

    

### [[2109.00532] TransforMesh: A Transformer Network for Longitudinal modeling of Anatomical Meshes](http://arxiv.org/abs/2109.00532)


  The longitudinal modeling of neuroanatomical changes related to Alzheimer's
disease (AD) is crucial for studying the progression of the disease. To this
end, we introduce TransforMesh, a spatio-temporal network based on transformers
that models longitudinal shape changes on 3D anatomical meshes. While
transformer and mesh networks have recently shown impressive performances in
natural language processing and computer vision, their application to medical
image analysis has been very limited. To the best of our knowledge, this is the
first work that combines transformer and mesh networks. Our results show that
TransforMesh can model shape trajectories better than other baseline
architectures that do not capture temporal dependencies. Moreover, we also
explore the capabilities of TransforMesh in detecting structural anomalies of
the hippocampus in patients developing AD.

    

### [[2109.00533] R-SNN: An Analysis and Design Methodology for Robustifying Spiking Neural Networks against Adversarial Attacks through Noise Filters for Dynamic Vision Sensors](http://arxiv.org/abs/2109.00533)


  Spiking Neural Networks (SNNs) aim at providing energy-efficient learning
capabilities when implemented on neuromorphic chips with event-based Dynamic
Vision Sensors (DVS). This paper studies the robustness of SNNs against
adversarial attacks on such DVS-based systems, and proposes R-SNN, a novel
methodology for robustifying SNNs through efficient DVS-noise filtering. We are
the first to generate adversarial attacks on DVS signals (i.e., frames of
events in the spatio-temporal domain) and to apply noise filters for DVS
sensors in the quest for defending against adversarial attacks. Our results
show that the noise filters effectively prevent the SNNs from being fooled. The
SNNs in our experiments provide more than 90% accuracy on the DVS-Gesture and
NMNIST datasets under different adversarial threat models.

    

### [[2109.00534] The Minimax Complexity of Distributed Optimization](http://arxiv.org/abs/2109.00534)


  In this thesis, I study the minimax oracle complexity of distributed
stochastic optimization. First, I present the "graph oracle model", an
extension of the classic oracle complexity framework that can be applied to
study distributed optimization algorithms. Next, I describe a general approach
to proving optimization lower bounds for arbitrary randomized algorithms (as
opposed to more restricted classes of algorithms, e.g., deterministic or
"zero-respecting" algorithms), which is used extensively throughout the thesis.
For the remainder of the thesis, I focus on the specific case of the
"intermittent communication setting", where multiple computing devices work in
parallel with limited communication amongst themselves. In this setting, I
analyze the theoretical properties of the popular Local Stochastic Gradient
Descent (SGD) algorithm in convex setting, both for homogeneous and
heterogeneous objectives. I provide the first guarantees for Local SGD that
improve over simple baseline methods, but show that Local SGD is not optimal in
general. In pursuit of optimal methods in the intermittent communication
setting, I then show matching upper and lower bounds for the intermittent
communication setting with homogeneous convex, heterogeneous convex, and
homogeneous non-convex objectives. These upper bounds are attained by simple
variants of SGD which are therefore optimal. Finally, I discuss several
additional assumptions about the objective or more powerful oracles that might
be exploitable in order to develop better intermittent communication algorithms
with better guarantees than our lower bounds allow.

    

### [[2109.00535] ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan](http://arxiv.org/abs/2109.00535)


  The automatic speaker verification spoofing and countermeasures (ASVspoof)
challenge series is a community-led initiative which aims to promote the
consideration of spoofing and the development of countermeasures. ASVspoof 2021
is the 4th in a series of bi-annual, competitive challenges where the goal is
to develop countermeasures capable of discriminating between bona fide and
spoofed or deepfake speech. This document provides a technical description of
the ASVspoof 2021 challenge, including details of training, development and
evaluation data, metrics, baselines, evaluation rules, submission procedures
and the schedule.

    

### [[2109.00537] ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection](http://arxiv.org/abs/2109.00537)


  ASVspoof 2021 is the forth edition in the series of bi-annual challenges
which aim to promote the study of spoofing and the design of countermeasures to
protect automatic speaker verification systems from manipulation. In addition
to a continued focus upon logical and physical access tasks in which there are
a number of advances compared to previous editions, ASVspoof 2021 introduces a
new task involving deepfake speech detection. This paper describes all three
tasks, the new databases for each of them, the evaluation metrics, four
challenge baselines, the evaluation platform and a summary of challenge
results. Despite the introduction of channel and compression variability which
compound the difficulty, results for the logical access and deepfake tasks are
close to those from previous ASVspoof editions. Results for the physical access
task show the difficulty in detecting attacks in real, variable physical
spaces. With ASVspoof 2021 being the first edition for which participants were
not provided with any matched training or development data and with this
reflecting real conditions in which the nature of spoofed and deepfake speech
can never be predicated with confidence, the results are extremely encouraging
and demonstrate the substantial progress made in the field in recent years.

    

### [[2109.00538] Physics-integrated hybrid framework for model form error identification in nonlinear dynamical systems](http://arxiv.org/abs/2109.00538)


  For real-life nonlinear systems, the exact form of nonlinearity is often not
known and the known governing equations are often based on certain assumptions
and approximations. Such representation introduced model-form error into the
system. In this paper, we propose a novel gray-box modeling approach that not
only identifies the model-form error but also utilizes it to improve the
predictive capability of the known but approximate governing equation. The
primary idea is to treat the unknown model-form error as a residual force and
estimate it using duel Bayesian filter based joint input-state estimation
algorithms. For improving the predictive capability of the underlying physics,
we first use machine learning algorithm to learn a mapping between the
estimated state and the input (model-form error) and then introduce it into the
governing equation as an additional term. This helps in improving the
predictive capability of the governing physics and allows the model to
generalize to unseen environment. Although in theory, any machine learning
algorithm can be used within the proposed framework, we use Gaussian process in
this work. To test the performance of proposed framework, case studies
discussing four different dynamical systems are discussed; results for which
indicate that the framework is applicable to a wide variety of systems and can
produce reliable estimates of original system's states.

    

### [[2109.00539] Spatially and Robustly Hybrid Mixture Regression Model for Inference of Spatial Dependence](http://arxiv.org/abs/2109.00539)


  In this paper, we propose a Spatial Robust Mixture Regression model to
investigate the relationship between a response variable and a set of
explanatory variables over the spatial domain, assuming that the relationships
may exhibit complex spatially dynamic patterns that cannot be captured by
constant regression coefficients. Our method integrates the robust finite
mixture Gaussian regression model with spatial constraints, to simultaneously
handle the spatial nonstationarity, local homogeneity, and outlier
contaminations. Compared with existing spatial regression models, our proposed
model assumes the existence a few distinct regression models that are estimated
based on observations that exhibit similar response-predictor relationships. As
such, the proposed model not only accounts for nonstationarity in the spatial
trend, but also clusters observations into a few distinct and homogenous
groups. This provides an advantage on interpretation with a few stationary
sub-processes identified that capture the predominant relationships between
response and predictor variables. Moreover, the proposed method incorporates
robust procedures to handle contaminations from both regression outliers and
spatial outliers. By doing so, we robustly segment the spatial domain into
distinct local regions with similar regression coefficients, and sporadic
locations that are purely outliers. Rigorous statistical hypothesis testing
procedure has been designed to test the significance of such segmentation.
Experimental results on many synthetic and real-world datasets demonstrate the
robustness, accuracy, and effectiveness of our proposed method, compared with
other robust finite mixture regression, spatial regression and spatial
segmentation methods.

    

### [[2109.00540] Variational Quantum Reinforcement Learning via Evolutionary Optimization](http://arxiv.org/abs/2109.00540)


  Recent advance in classical reinforcement learning (RL) and quantum
computation (QC) points to a promising direction of performing RL on a quantum
computer. However, potential applications in quantum RL are limited by the
number of qubits available in the modern quantum devices. Here we present two
frameworks of deep quantum RL tasks using a gradient-free evolution
optimization: First, we apply the amplitude encoding scheme to the Cart-Pole
problem; Second, we propose a hybrid framework where the quantum RL agents are
equipped with hybrid tensor network-variational quantum circuit (TN-VQC)
architecture to handle inputs with dimensions exceeding the number of qubits.
This allows us to perform quantum RL on the MiniGrid environment with
147-dimensional inputs. We demonstrate the quantum advantage of parameter
saving using the amplitude encoding. The hybrid TN-VQC architecture provides a
natural way to perform efficient compression of the input dimension, enabling
further quantum RL applications on noisy intermediate-scale quantum devices.

    

### [[2109.00541] Active Inference and Epistemic Value in Graphical Models](http://arxiv.org/abs/2109.00541)


  The Free Energy Principle (FEP) postulates that biological agents perceive
and interact with their environment in order to minimize a Variational Free
Energy (VFE) with respect to a generative model of their environment. The
inference of a policy (future control sequence) according to the FEP is known
as Active Inference (AIF). The AIF literature describes multiple VFE objectives
for policy planning that lead to epistemic (information-seeking) behavior.
However, most objectives have limited modeling flexibility. This paper
approaches epistemic behavior from a constrained Bethe Free Energy (CBFE)
perspective. Crucially, variational optimization of the CBFE can be expressed
in terms of message passing on free-form generative models. The key intuition
behind the CBFE is that we impose a point-mass constraint on predicted
outcomes, which explicitly encodes the assumption that the agent will make
observations in the future. We interpret the CBFE objective in terms of its
constituent behavioral drives. We then illustrate resulting behavior of the
CBFE by planning and interacting with a simulated T-maze environment.
Simulations for the T-maze task illustrate how the CBFE agent exhibits an
epistemic drive, and actively plans ahead to account for the impact of
predicted outcomes. Compared to an EFE agent, the CBFE agent incurs expected
reward in significantly more environmental scenarios. We conclude that CBFE
optimization by message passing suggests a general mechanism for
epistemic-aware AIF in free-form generative models.

    

### [[2109.00542] Proof Transfer for Neural Network Verification](http://arxiv.org/abs/2109.00542)


  We introduce the novel concept of proof transfer for neural network
verification. We show that by generating proof templates that capture and
generalize existing proofs, we can speed up subsequent proofs. In particular we
create these templates from previous proofs on the same neural network and
consider two cases: (i) where the proofs are created online when verifying
other properties and (ii) where the templates are created offline using a
dataset. We base our methods on three key hypotheses of neural network
robustness proofs. Our evaluation shows the potential of proof transfer for
benefitting robustness verification of neural networks against adversarial
patches, geometric, and $\ell_{\infty}$-perturbations.

    

### [[2109.00544] Towards Improving Adversarial Training of NLP Models](http://arxiv.org/abs/2109.00544)


  Adversarial training, a method for learning robust deep neural networks,
constructs adversarial examples during training. However, recent methods for
generating NLP adversarial examples involve combinatorial search and expensive
sentence encoders for constraining the generated instances. As a result, it
remains challenging to use vanilla adversarial training to improve NLP models'
performance, and the benefits are mainly uninvestigated. This paper proposes a
simple and improved vanilla adversarial training process for NLP, which we name
Attacking to Training ($\texttt{A2T}$). The core part of $\texttt{A2T}$ is a
new and cheaper word substitution attack optimized for vanilla adversarial
training. We use $\texttt{A2T}$ to train BERT and RoBERTa models on IMDB,
Rotten Tomatoes, Yelp, and SNLI datasets. Our results show that it is possible
to train empirically robust NLP models using a much cheaper adversary. We
demonstrate that vanilla adversarial training with $\texttt{A2T}$ can improve
an NLP model's robustness to the attack it was originally trained with and also
defend the model against other types of attacks. Furthermore, we show that
$\texttt{A2T}$ can improve NLP models' standard accuracy, cross-domain
generalization, and interpretability. Code is available at
this http URL .

    

### [[2109.00545] Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks](http://arxiv.org/abs/2109.00545)


  Motivated by scenarios where data is used for diverse prediction tasks, we
study whether fair representation can be used to guarantee fairness for unknown
tasks and for multiple fairness notions simultaneously. We consider seven group
fairness notions that cover the concepts of independence, separation, and
calibration. Against the backdrop of the fairness impossibility results, we
explore approximate fairness. We prove that, although fair representation might
not guarantee fairness for all prediction tasks, it does guarantee fairness for
an important subset of tasks -- the tasks for which the representation is
discriminative. Specifically, all seven group fairness notions are linearly
controlled by fairness and discriminativeness of the representation. When an
incompatibility exists between different fairness notions, fair and
discriminative representation hits the sweet spot that approximately satisfies
all notions. Motivated by our theoretical findings, we propose to learn both
fair and discriminative representations using pretext loss which
self-supervises learning, and Maximum Mean Discrepancy as a fair regularizer.
Experiments on tabular, image, and face datasets show that using the learned
representation, downstream predictions that we are unaware of when learning the
representation indeed become fairer for seven group fairness notions, and the
fairness guarantees computed from our theoretical results are all valid.

    

### [[2109.00570] Process parameter optimization of Friction Stir Welding on 6061AA using Supervised Machine Learning Regression-based Algorithms](http://arxiv.org/abs/2109.00570)


  The highest strength-to-weight ratio criterion has fascinated curiosity
increasingly in virtually all areas where heft reduction is indispensable.
Lightweight materials and their joining processes are also a recent point of
research demands in the manufacturing industries. Friction Stir Welding (FSW)
is one of the recent advancements for joining materials without adding any
third material (filler rod) and joining below the melting point of the parent
material. The process is widely used for joining similar and dissimilar metals,
especially lightweight non-ferrous materials like aluminum, copper, and
magnesium alloys. This paper presents verdicts of optimum process parameters on
attaining enhanced mechanical properties of the weld joint. The experiment was
conducted on a 5 mm 6061 aluminum alloy sheet. Process parameters; tool
material, rotational speed, traverse speed, and axial forces were utilized.
Mechanical properties of the weld joint are examined employing a tensile test,
and the maximum joint strength efficiency was reached 94.2%. Supervised Machine
Learning based Regression algorithms such as Decision Trees, Random Forest, and
Gradient Boosting Algorithm were used. The results showed that the Random
Forest algorithm yielded highest coefficient of determination value of 0.926
which means it gives a best fit in comparison to other algorithms.

    

### [[2109.00573] Pulmonary Disease Classification Using Globally Correlated Maximum Likelihood: an Auxiliary Attention mechanism for Convolutional Neural Networks](http://arxiv.org/abs/2109.00573)


  Convolutional neural networks (CNN) are now being widely used for classifying
and detecting pulmonary abnormalities in chest radiographs. Two complementary
generalization properties of CNNs, translation invariance and equivariance, are
particularly useful in detecting manifested abnormalities associated with
pulmonary disease, regardless of their spatial locations within the image.
However, these properties also come with the loss of exact spatial information
and global relative positions of abnormalities detected in local regions.
Global relative positions of such abnormalities may help distinguish similar
conditions, such as COVID-19 and viral pneumonia. In such instances, a global
attention mechanism is needed, which CNNs do not support in their traditional
architectures that aim for generalization afforded by translation invariance
and equivariance. Vision Transformers provide a global attention mechanism, but
lack translation invariance and equivariance, requiring significantly more
training data samples to match generalization of CNNs. To address the loss of
spatial information and global relations between features, while preserving the
inductive biases of CNNs, we present a novel technique that serves as an
auxiliary attention mechanism to existing CNN architectures, in order to
extract global correlations between salient features.

    

### [[2109.00574] Active label cleaning: Improving dataset quality under resource constraints](http://arxiv.org/abs/2109.00574)


  Imperfections in data annotation, known as label noise, are detrimental to
the training of machine learning models and have an often-overlooked
confounding effect on the assessment of model performance. Nevertheless,
employing experts to remove label noise by fully re-annotating large datasets
is infeasible in resource-constrained settings, such as healthcare. This work
advocates for a data-driven approach to prioritising samples for re-annotation
- which we term "active label cleaning". We propose to rank instances according
to estimated label correctness and labelling difficulty of each sample, and
introduce a simulation framework to evaluate relabelling efficacy. Our
experiments on natural images and on a new medical imaging benchmark show that
cleaning noisy labels mitigates their negative impact on model training,
evaluation, and selection. Crucially, the proposed active label cleaning
enables correcting labels up to 4 times more effectively than typical random
selection in realistic conditions, making better use of experts' valuable time
for improving dataset quality.

    

### [[2109.00577] FaVoA: Face-Voice Association Favours Ambiguous Speaker Detection](http://arxiv.org/abs/2109.00577)


  The strong relation between face and voice can aid active speaker detection
systems when faces are visible, even in difficult settings, when the face of a
speaker is not clear or when there are several people in the same scene. By
being capable of estimating the frontal facial representation of a person from
his/her speech, it becomes easier to determine whether he/she is a potential
candidate for being classified as an active speaker, even in challenging cases
in which no mouth movement is detected from any person in that same scene. By
incorporating a face-voice association neural network into an existing
state-of-the-art active speaker detection model, we introduce FaVoA (Face-Voice
Association Ambiguous Speaker Detector), a neural network model that can
correctly classify particularly ambiguous scenarios. FaVoA not only finds
positive associations, but helps to rule out non-matching face-voice
associations, where a face does not match a voice. Its use of a
gated-bimodal-unit architecture for the fusion of those models offers a way to
quantitatively determine how much each modality contributes to the
classification.

    

### [[2109.00582] Information-theoretic Classification Accuracy: A Criterion that Guides Data-driven Combination of Ambiguous Outcome Labels in Multi-class Classification](http://arxiv.org/abs/2109.00582)


  Outcome labeling ambiguity and subjectivity are ubiquitous in real-world
datasets. While practitioners commonly combine ambiguous outcome labels in an
ad hoc way to improve the accuracy of multi-class classification, there lacks a
principled approach to guide label combination by any optimality criterion. To
address this problem, we propose the information-theoretic classification
accuracy (ITCA), a criterion of outcome "information" conditional on outcome
prediction, to guide practitioners on how to combine ambiguous outcome labels.
ITCA indicates a balance in the trade-off between prediction accuracy (how well
do predicted labels agree with actual labels) and prediction resolution (how
many labels are predictable). To find the optimal label combination indicated
by ITCA, we develop two search strategies: greedy search and breadth-first
search. Notably, ITCA and the two search strategies are adaptive to all
machine-learning classification algorithms. Coupled with a classification
algorithm and a search strategy, ITCA has two uses: to improve prediction
accuracy and to identify ambiguous labels. We first verify that ITCA achieves
high accuracy with both search strategies in finding the correct label
combinations on synthetic and real data. Then we demonstrate the effectiveness
of ITCA in diverse applications including medical prognosis, cancer survival
prediction, user demographics prediction, and cell type classification.

    

### [[2109.00590] WebQA: Multihop and Multimodal QA](http://arxiv.org/abs/2109.00590)


  Web search is fundamentally multimodal and multihop. Often, even before
asking a question we choose to go directly to image search to find our answers.
Further, rarely do we find an answer from a single source but aggregate
information and reason through implications. Despite the frequency of this
everyday occurrence, at present, there is no unified question answering
benchmark that requires a single model to answer long-form natural language
questions from text and open-ended visual sources -- akin to a human's
experience. We propose to bridge this gap between the natural language and
computer vision communities with WebQA. We show that A. our multihop text
queries are difficult for a large-scale transformer model, and B. existing
multi-modal transformers and visual representations do not perform well on
open-domain visual queries. Our challenge for the community is to create a
unified multimodal reasoning model that seamlessly transitions and reasons
regardless of the source modality.

    

### [[2109.00594] Wearable-based Classification of Running Styles with Deep Learning](http://arxiv.org/abs/2109.00594)


  Automatic classification of running styles can enable runners to obtain
feedback with the aim of optimizing performance in terms of minimizing energy
expenditure, fatigue, and risk of injury. To develop a system capable of
classifying running styles using wearables, we collect a dataset from 10
healthy runners performing 8 different pre-defined running styles. Five
wearable devices are used to record accelerometer data from different parts of
the lower body, namely left and right foot, left and right medial tibia, and
lower back. Using the collected dataset, we develop a deep learning solution
which consists of a Convolutional Neural Network and Long Short-Term Memory
network to first automatically extract effective features, followed by learning
temporal relationships. Score-level fusion is used to aggregate the
classification results from the different sensors. Experiments show that the
proposed model is capable of automatically classifying different running styles
in a subject-dependant manner, outperforming several classical machine learning
methods (following manual feature extraction) and a convolutional neural
network baseline. Moreover, our study finds that subject-independent
classification of running styles is considerably more challenging than a
subject-dependant scheme, indicating a high level of personalization in such
running styles. Finally, we demonstrate that by fine-tuning the model with as
few as 5% subject-specific samples, considerable performance boost is obtained.

    

### [[2109.00596] Streaming data preprocessing via online tensor recovery for large environmental sensor networks](http://arxiv.org/abs/2109.00596)


  Measuring the built and natural environment at a fine-grained scale is now
possible with low-cost urban environmental sensor networks. However,
fine-grained city-scale data analysis is complicated by tedious data cleaning
including removing outliers and imputing missing data. While many methods exist
to automatically correct anomalies and impute missing entries, challenges still
exist on data with large spatial-temporal scales and shifting patterns. To
address these challenges, we propose an online robust tensor recovery (OLRTR)
method to preprocess streaming high-dimensional urban environmental datasets. A
small-sized dictionary that captures the underlying patterns of the data is
computed and constantly updated with new data. OLRTR enables online recovery
for large-scale sensor networks that provide continuous data streams, with a
lower computational memory usage compared to offline batch counterparts. In
addition, we formulate the objective function so that OLRTR can detect
structured outliers, such as faulty readings over a long period of time. We
validate OLRTR on a synthetically degraded National Oceanic and Atmospheric
Administration temperature dataset, with a recovery error of 0.05, and apply it
to the Array of Things city-scale sensor network in Chicago, IL, showing
superior results compared with several established online and batch-based low
rank decomposition methods.

    

### [[2109.00617] LinEasyBO: Scalable Bayesian Optimization Approach for Analog Circuit Synthesis via One-Dimensional Subspaces](http://arxiv.org/abs/2109.00617)


  A large body of literature has proved that the Bayesian optimization
framework is especially efficient and effective in analog circuit synthesis.
However, most of the previous research works only focus on designing
informative surrogate models or efficient acquisition functions. Even if
searching for the global optimum over the acquisition function surface is
itself a difficult task, it has been largely ignored. In this paper, we propose
a fast and robust Bayesian optimization approach via one-dimensional subspaces
for analog circuit synthesis. By solely focusing on optimizing one-dimension
subspaces at each iteration, we greatly reduce the computational overhead of
the Bayesian optimization framework while safely maximizing the acquisition
function. By combining the benefits of different dimension selection
strategies, we adaptively balancing between searching globally and locally. By
leveraging the batch Bayesian optimization framework, we further accelerate the
optimization procedure by making full use of the hardware resources.
Experimental results quantitatively show that our proposed algorithm can
accelerate the optimization procedure by up to 9x and 38x compared to LP-EI and
REMBOpBO respectively when the batch size is 15.

    

### [[2109.00619] Learning compositional programs with arguments and sampling](http://arxiv.org/abs/2109.00619)


  One of the most challenging goals in designing intelligent systems is
empowering them with the ability to synthesize programs from data. Namely,
given specific requirements in the form of input/output pairs, the goal is to
train a machine learning model to discover a program that satisfies those
requirements. A recent class of methods exploits combinatorial search
procedures and deep learning to learn compositional programs. However, they
usually generate only toy programs using a domain-specific language that does
not provide any high-level feature, such as function arguments, which reduces
their applicability in real-world settings. We extend upon a state of the art
model, AlphaNPI, by learning to generate functions that can accept arguments.
This improvement will enable us to move closer to real computer programs.
Moreover, we investigate employing an Approximate version of Monte Carlo Tree
Search (A-MCTS) to speed up convergence. We showcase the potential of our
approach by learning the Quicksort algorithm, showing how the ability to deal
with arguments is crucial for learning and generalization.

    

### [[2109.00622] An End-to-End learnable Flow Regularized Model for Brain Tumor Segmentation](http://arxiv.org/abs/2109.00622)


  Many segmentation tasks for biomedical images can be modeled as the
minimization of an energy function and solved by a class of max-flow and
min-cut optimization algorithms. However, the segmentation accuracy is
sensitive to the contrasting of semantic features of different segmenting
objects, as the traditional energy function usually uses hand-crafted features
in their energy functions. To address these limitations, we propose to
incorporate end-to-end trainable neural network features into the energy
functions. Our deep neural network features are extracted from the
down-sampling and up-sampling layers with skip-connections of a U-net. In the
inference stage, the learned features are fed into the energy functions. And
the segmentations are solved in a primal-dual form by ADMM solvers. In the
training stage, we train our neural networks by optimizing the energy function
in the primal form with regularizations on the min-cut and flow-conservation
functions, which are derived from the optimal conditions in the dual form. We
evaluate our methods, both qualitatively and quantitatively, in a brain tumor
segmentation task. As the energy minimization model achieves a balance on
sensitivity and smooth boundaries, we would show how our segmentation contours
evolve actively through iterations as ensemble references for doctor diagnosis.

    

### [[2109.00630] A Novel Multi-Centroid Template Matching Algorithm and Its Application to Cough Detection](http://arxiv.org/abs/2109.00630)


  Cough is a major symptom of respiratory-related diseases. There exists a
tremendous amount of work in detecting coughs from audio but there has been no
effort to identify coughs from solely inertial measurement unit (IMU). Coughing
causes motion across the whole body and especially on the neck and head.
Therefore, head motion data during coughing captured by a head-worn IMU sensor
could be leveraged to detect coughs using a template matching algorithm. In
time series template matching problems, K-Nearest Neighbors (KNN) combined with
elastic distance measurement (esp. Dynamic Time Warping (DTW)) achieves
outstanding performance. However, it is often regarded as prohibitively
time-consuming. Nearest Centroid Classifier is thereafter proposed. But the
accuracy is comprised of only one centroid obtained for each class.
Centroid-based Classifier performs clustering and averaging for each cluster,
but requires manually setting the number of clusters. We propose a novel
self-tuning multi-centroid template-matching algorithm, which can automatically
adjust the number of clusters to balance accuracy and inference time. Through
experiments conducted on synthetic datasets and a real-world earbud-based cough
dataset, we demonstrate the superiority of our proposed algorithm and present
the result of cough detection with a single accelerometer sensor on the earbuds
platform.

    

### [[2109.00635] Selecting Optimal Trace Clustering Pipelines with AutoML](http://arxiv.org/abs/2109.00635)


  Trace clustering has been extensively used to preprocess event logs. By
grouping similar behavior, these techniques guide the identification of
sub-logs, producing more understandable models and conformance analytics.
Nevertheless, little attention has been posed to the relationship between event
log properties and clustering quality. In this work, we propose an Automatic
Machine Learning (AutoML) framework to recommend the most suitable pipeline for
trace clustering given an event log, which encompasses the encoding method,
clustering algorithm, and its hyperparameters. Our experiments were conducted
using a thousand event logs, four encoding techniques, and three clustering
methods. Results indicate that our framework sheds light on the trace
clustering problem and can assist users in choosing the best pipeline
considering their scenario.

    

### [[2109.00637] Properly learning decision trees in almost polynomial time](http://arxiv.org/abs/2109.00637)


  We give an $n^{O(\log\log n)}$-time membership query algorithm for properly
and agnostically learning decision trees under the uniform distribution over
$\{\pm 1\}^n$. Even in the realizable setting, the previous fastest runtime was
$n^{O(\log n)}$, a consequence of a classic algorithm of Ehrenfeucht and
Haussler.
Our algorithm shares similarities with practical heuristics for learning
decision trees, which we augment with additional ideas to circumvent known
lower bounds against these heuristics. To analyze our algorithm, we prove a new
structural result for decision trees that strengthens a theorem of O'Donnell,
Saks, Schramm, and Servedio. While the OSSS theorem says that every decision
tree has an influential variable, we show how every decision tree can be
"pruned" so that every variable in the resulting tree is influential.

    

### [[2109.00644] RIFLE: Robust Inference from Low Order Marginals](http://arxiv.org/abs/2109.00644)


  The ubiquity of missing values in real-world datasets poses a challenge for
statistical inference and can prevent similar datasets from being analyzed in
the same study, precluding many existing datasets from being used for new
analyses. While an extensive collection of packages and algorithms have been
developed for data imputation, the overwhelming majority perform poorly if
there are many missing values and low sample size, which are unfortunately
common characteristics in empirical data. Such low-accuracy estimations
adversely affect the performance of downstream statistical models. We develop a
statistical inference framework for predicting the target variable without
imputing missing values. Our framework, RIFLE (Robust InFerence via Low-order
moment Estimations), estimates low-order moments with corresponding confidence
intervals to learn a distributionally robust model. We specialize our framework
to linear regression and normal discriminant analysis, and we provide
convergence and performance guarantees. This framework can also be adapted to
impute missing data. In numerical experiments, we compare RIFLE with
state-of-the-art approaches (including MICE, Amelia, MissForest, KNN-imputer,
MIDA, and Mean Imputer). Our experiments demonstrate that RIFLE outperforms
other benchmark algorithms when the percentage of missing values is high and/or
when the number of data points is relatively small. RIFLE is publicly
available.

    

### [[2109.00650] Dash: Semi-Supervised Learning with Dynamic Thresholding](http://arxiv.org/abs/2109.00650)


  While semi-supervised learning (SSL) has received tremendous attentions in
many machine learning tasks due to its successful use of unlabeled data,
existing SSL algorithms use either all unlabeled examples or the unlabeled
examples with a fixed high-confidence prediction during the training progress.
However, it is possible that too many correct/wrong pseudo labeled examples are
eliminated/selected. In this work we develop a simple yet powerful framework,
whose key idea is to select a subset of training examples from the unlabeled
data when performing existing SSL methods so that only the unlabeled examples
with pseudo labels related to the labeled data will be used to train models.
The selection is performed at each updating iteration by only keeping the
examples whose losses are smaller than a given threshold that is dynamically
adjusted through the iteration. Our proposed approach, Dash, enjoys its
adaptivity in terms of unlabeled data selection and its theoretical guarantee.
Specifically, we theoretically establish the convergence rate of Dash from the
view of non-convex optimization. Finally, we empirically demonstrate the
effectiveness of the proposed method in comparison with state-of-the-art over
benchmarks.

    

### [[2109.00666] TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks](http://arxiv.org/abs/2109.00666)


  With the increasing reliance on automated decision making, the issue of
algorithmic fairness has gained increasing importance. In this paper, we
propose a Generative Adversarial Network for tabular data generation. The model
includes two phases of training. In the first phase, the model is trained to
accurately generate synthetic data similar to the reference dataset. In the
second phase we modify the value function to add fairness constraint, and
continue training the network to generate data that is both accurate and fair.
We test our results in both cases of unconstrained, and constrained fair data
generation. In the unconstrained case, i.e. when the model is only trained in
the first phase and is only meant to generate accurate data following the same
joint probability distribution of the real data, the results show that the
model beats state-of-the-art GANs proposed in the literature to produce
synthetic tabular data. Also, in the constrained case in which the first phase
of training is followed by the second phase, we train the network and test it
on four datasets studied in the fairness literature and compare our results
with another state-of-the-art pre-processing method, and present the promising
results that it achieves. Comparing to other studies utilizing GANs for fair
data generation, our model is comparably more stable by using only one critic,
and also by avoiding major problems of original GAN model, such as
mode-dropping and non-convergence, by implementing a Wasserstein GAN.

    

### [[2109.00670] Variable Augmented Network for Invertible Modality Synthesis-Fusion](http://arxiv.org/abs/2109.00670)


  As an effective way to integrate the information contained in multiple
medical images under different modalities, medical image synthesis and fusion
have emerged in various clinical applications such as disease diagnosis and
treatment planning. In this paper, an invertible and variable augmented network
(iVAN) is proposed for medical image synthesis and fusion. In iVAN, the channel
number of the network input and output is the same through variable
augmentation technology, and data relevance is enhanced, which is conducive to
the generation of characterization information. Meanwhile, the invertible
network is used to achieve the bidirectional inference processes. Due to the
invertible and variable augmentation schemes, iVAN can not only be applied to
the mappings of multi-input to one-output and multi-input to multi-output, but
also be applied to one-input to multi-output. Experimental results demonstrated
that the proposed method can obtain competitive or superior performance in
comparison to representative medical image synthesis and fusion methods.

    

### [[2109.00675] FLASHE: Additively Symmetric Homomorphic Encryption for Cross-Silo Federated Learning](http://arxiv.org/abs/2109.00675)


  Homomorphic encryption (HE) is a promising privacy-preserving technique for
cross-silo federated learning (FL), where organizations perform collaborative
model training on decentralized data. Despite the strong privacy guarantee,
general HE schemes result in significant computation and communication
overhead. Prior works employ batch encryption to address this problem, but it
is still suboptimal in mitigating communication overhead and is incompatible
with sparsification techniques.
In this paper, we propose FLASHE, an HE scheme tailored for cross-silo FL. To
capture the minimum requirements of security and functionality, FLASHE drops
the asymmetric-key design and only involves modular addition operations with
random numbers. Depending on whether to accommodate sparsification techniques,
FLASHE is optimized in computation efficiency with different approaches. We
have implemented FLASHE as a pluggable module atop FATE, an industrial platform
for cross-silo FL. Compared to plaintext training, FLASHE slightly increases
the training time by $\leq6\%$, with no communication overhead.

    

### [[2109.00678] Regional Adversarial Training for Better Robust Generalization](http://arxiv.org/abs/2109.00678)


  Adversarial training (AT) has been demonstrated as one of the most promising
defense methods against various adversarial attacks. To our knowledge, existing
AT-based methods usually train with the locally most adversarial perturbed
points and treat all the perturbed points equally, which may lead to
considerably weaker adversarial robust generalization on test data. In this
work, we introduce a new adversarial training framework that considers the
diversity as well as characteristics of the perturbed points in the vicinity of
benign samples. To realize the framework, we propose a Regional Adversarial
Training (RAT) defense method that first utilizes the attack path generated by
the typical iterative attack method of projected gradient descent (PGD), and
constructs an adversarial region based on the attack path. Then, RAT samples
diverse perturbed training points efficiently inside this region, and utilizes
a distance-aware label smoothing mechanism to capture our intuition that
perturbed points at different locations should have different impact on the
model performance. Extensive experiments on several benchmark datasets show
that RAT consistently makes significant improvement on standard adversarial
training (SAT), and exhibits better robust generalization.

    

### [[2109.00685] Excess Capacity and Backdoor Poisoning](http://arxiv.org/abs/2109.00685)


  A backdoor data poisoning attack is an adversarial attack wherein the
attacker injects several watermarked, mislabeled training examples into a
training set. The watermark does not impact the test-time performance of the
model on typical data; however, the model reliably errs on watermarked
examples.
To gain a better foundational understanding of backdoor data poisoning
attacks, we present a formal theoretical framework within which one can discuss
backdoor data poisoning attacks for classification problems. We then use this
to analyze important statistical and computational issues surrounding these
attacks.
On the statistical front, we identify a parameter we call the memorization
capacity that captures the intrinsic vulnerability of a learning problem to a
backdoor attack. This allows us to argue about the robustness of several
natural learning problems to backdoor attacks. Our results favoring the
attacker involve presenting explicit constructions of backdoor attacks, and our
robustness results show that some natural problem settings cannot yield
successful backdoor attacks.
From a computational standpoint, we show that under certain assumptions,
adversarial training can detect the presence of backdoors in a training set. We
then show that under similar assumptions, two closely related problems we call
backdoor filtering and robust generalization are nearly equivalent. This
implies that it is both asymptotically necessary and sufficient to design
algorithms that can identify watermarked examples in the training set in order
to obtain a learning algorithm that both generalizes well to unseen data and is
robust to backdoors.

    

### [[2109.00691] Global Convolutional Neural Processes](http://arxiv.org/abs/2109.00691)


  The ability to deal with uncertainty in machine learning models has become
equally, if not more, crucial to their predictive ability itself. For instance,
during the pandemic, governmental policies and personal decisions are
constantly made around uncertainties. Targeting this, Neural Process Families
(NPFs) have recently shone a light on prediction with uncertainties by bridging
Gaussian processes and neural networks. Latent neural process, a member of NPF,
is believed to be capable of modelling the uncertainty on certain points (local
uncertainty) as well as the general function priors (global uncertainties).
Nonetheless, some critical questions remain unresolved, such as a formal
definition of global uncertainties, the causality behind global uncertainties,
and the manipulation of global uncertainties for generative models. Regarding
this, we build a member GloBal Convolutional Neural Process(GBCoNP) that
achieves the SOTA log-likelihood in latent NPFs. It designs a global
uncertainty representation p(z), which is an aggregation on a discretized input
space. The causal effect between the degree of global uncertainty and the
intra-task diversity is discussed. The learnt prior is analyzed on a variety of
scenarios, including 1D, 2D, and a newly proposed spatial-temporal COVID
dataset. Our manipulation of the global uncertainty not only achieves
generating the desired samples to tackle few-shot learning, but also enables
the probability evaluation on the functional priors.

    

### [[2109.00700] Machine learning moment closure models for the radiative transfer equation III: enforcing hyperbolicity and physical characteristic speeds](http://arxiv.org/abs/2109.00700)


  This is the third paper in a series in which we develop machine learning (ML)
moment closure models for the radiative transfer equation (RTE). In our
previous work \cite{huang2021gradient}, we proposed an approach to learn the
gradient of the unclosed high order moment, which performs much better than
learning the moment itself and the conventional $P_N$ closure. However, while
the ML moment closure has better accuracy, it is not able to guarantee
hyperbolicity and has issues with long time stability. In our second paper
\cite{huang2021hyperbolic}, we identified a symmetrizer which leads to
conditions that enforce that the gradient based ML closure is symmetrizable
hyperbolic and stable over long time. The limitation of this approach is that
in practice the highest moment can only be related to four, or fewer, lower
moments.
In this paper, we propose a new method to enforce the hyperbolicity of the ML
closure model. Motivated by the observation that the coefficient matrix of the
closure system is a lower Hessenberg matrix, we relate its eigenvalues to the
roots of an associated polynomial. We design two new neural network
architectures based on this relation. The ML closure model resulting from the
first neural network is weakly hyperbolic and guarantees the physical
characteristic speeds, i.e., the eigenvalues are bounded by the speed of light.
The second model is strictly hyperbolic and does not guarantee the boundedness
of the eigenvalues. Several benchmark tests including the Gaussian source
problem and the two-material problem show the good accuracy, stability and
generalizability of our hyperbolic ML closure model.

    

### [[2109.00707] Cross-Model Consensus of Explanations and Beyond for Image Classification Models: An Empirical Study](http://arxiv.org/abs/2109.00707)


  Existing interpretation algorithms have found that, even deep models make the
same and right predictions on the same image, they might rely on different sets
of input features for classification. However, among these sets of features,
some common features might be used by the majority of models. In this paper, we
are wondering what are the common features used by various models for
classification and whether the models with better performance may favor those
common features. For this purpose, our works uses an interpretation algorithm
to attribute the importance of features (e.g., pixels or superpixels) as
explanations, and proposes the cross-model consensus of explanations to capture
the common features. Specifically, we first prepare a set of deep models as a
committee, then deduce the explanation for every model, and obtain the
consensus of explanations across the entire committee through voting. With the
cross-model consensus of explanations, we conduct extensive experiments using
80+ models on 5 datasets/tasks. We find three interesting phenomena as follows:
(1) the consensus obtained from image classification models is aligned with the
ground truth of semantic segmentation; (2) we measure the similarity of the
explanation result of each model in the committee to the consensus (namely
consensus score), and find positive correlations between the consensus score
and model performance; and (3) the consensus score coincidentally correlates to
the interpretability.

    

### [[2109.00708] Efficient Algorithms For Fair Clustering with a New Fairness Notion](http://arxiv.org/abs/2109.00708)


  We revisit the problem of fair clustering, first introduced by Chierichetti
et al., that requires each protected attribute to have approximately equal
representation in every cluster; i.e., a balance property. Existing solutions
to fair clustering are either not scalable or do not achieve an optimal
trade-off between clustering objective and fairness. In this paper, we propose
a new notion of fairness, which we call $tau$-fair fairness, that strictly
generalizes the balance property and enables a fine-grained efficiency vs.
fairness trade-off. Furthermore, we show that simple greedy round-robin based
algorithms achieve this trade-off efficiently. Under a more general setting of
multi-valued protected attributes, we rigorously analyze the theoretical
properties of the our algorithms. Our experimental results suggest that the
proposed solution outperforms all the state-of-the-art algorithms and works
exceptionally well even for a large number of clusters.

    

### [[2109.00711] Heterogeneous relational message passing networks for molecular dynamics simulations](http://arxiv.org/abs/2109.00711)


  With many frameworks based on message passing neural networks proposed to
predict molecular and bulk properties, machine learning methods have
tremendously shifted the paradigms of computational sciences underpinning
physics, material science, chemistry, and biology. While existing machine
learning models have yielded superior performances in many occasions, most of
them model and process molecular systems in terms of homogeneous graph, which
severely limits the expressive power for representing diverse interactions. In
practice, graph data with multiple node and edge types is ubiquitous and more
appropriate for molecular systems. Thus, we propose the heterogeneous
relational message passing network (HermNet), an end-to-end heterogeneous graph
neural networks, to efficiently express multiple interactions in a single model
with {\it ab initio} accuracy. HermNet performs impressively against many
top-performing models on both molecular and extended systems. Specifically,
HermNet outperforms other tested models in nearly 75\%, 83\% and 94\% of tasks
on MD17, QM9 and extended systems datasets, respectively. Finally, we elucidate
how the design of HermNet is compatible with quantum mechanics from the
perspective of the density functional theory. Besides, HermNet is a universal
framework, whose sub-networks could be replaced by other advanced models.

    

### [[2109.00724] RF-LighGBM: A probabilistic ensemble way to predict customer repurchase behaviour in community e-commerce](http://arxiv.org/abs/2109.00724)


  It is reported that the number of online payment users in China has reached
854 million; with the emergence of community e-commerce platforms, the trend of
integration of e-commerce and social applications is increasingly intense.
Community e-commerce is not a mature and sound comprehensive e-commerce with
fewer categories and low brand value. To effectively retain community users and
fully explore customer value has become an important challenge for community
e-commerce operators. Given the above problems, this paper uses the data-driven
method to study the prediction of community e-commerce customers' repurchase
behaviour. The main research contents include 1. Given the complex problem of
feature engineering, the classic model RFM in the field of customer
relationship management is improved, and an improved model is proposed to
describe the characteristics of customer buying behaviour, which includes five
indicators. 2. In view of the imbalance of machine learning training samples in
SMOTE-ENN, a training sample balance using SMOTE-ENN is proposed. The
experimental results show that the machine learning model can be trained more
effectively on balanced samples. 3. Aiming at the complexity of the parameter
adjustment process, an automatic hyperparameter optimization method based on
the TPE method was proposed. Compared with other methods, the model's
prediction performance is improved, and the training time is reduced by more
than 450%. 4. Aiming at the weak prediction ability of a single model, the soft
voting based RF-LightgBM model was proposed. The experimental results show that
the RF-LighTGBM model proposed in this paper can effectively predict customer
repurchase behaviour, and the F1 value is 0.859, which is better than the
single model and previous research results.

    

### [[2109.00725] Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond](http://arxiv.org/abs/2109.00725)


  A fundamental goal of scientific research is to learn about causal
relationships. However, despite its critical role in the life and social
sciences, causality has not had the same importance in Natural Language
Processing (NLP), which has traditionally placed more emphasis on predictive
tasks. This distinction is beginning to fade, with an emerging area of
interdisciplinary research at the convergence of causal inference and language
processing. Still, research on causality in NLP remains scattered across
domains without unified definitions, benchmark datasets and clear articulations
of the remaining challenges. In this survey, we consolidate research across
academic areas and situate it in the broader NLP landscape. We introduce the
statistical challenge of estimating causal effects, encompassing settings where
text is used as an outcome, treatment, or as a means to address confounding. In
addition, we explore potential uses of causal inference to improve the
performance, robustness, fairness, and interpretability of NLP models. We thus
provide a unified overview of causal inference for the computational
linguistics community.

    

### [[2109.00727] Some Inapproximability Results of MAP Inference and Exponentiated Determinantal Point Processes](http://arxiv.org/abs/2109.00727)


  We study the computational complexity of two hard problems on determinantal
point processes (DPPs). One is maximum a posteriori (MAP) inference, i.e., to
find a principal submatrix having the maximum determinant. The other is
probabilistic inference on exponentiated DPPs (E-DPPs), which can sharpen or
weaken the diversity preference of DPPs with an exponent parameter $p$. We
prove the following complexity-theoretic hardness results that explain the
difficulty in approximating MAP inference and the normalizing constant for
E-DPPs.
1. Unconstrained MAP inference for an $n \times n$ matrix is NP-hard to
approximate within a factor of $2^{\beta n}$, where $\beta = 10^{-10^{13}} $.
This result improves upon a $(\frac{9}{8}-\epsilon)$-factor inapproximability
given by Kulesza and Taskar (2012).
2. Log-determinant maximization is NP-hard to approximate within a factor of
$\frac{5}{4}$ for the unconstrained case and within a factor of
$1+10^{-10^{13}}$ for the size-constrained monotone case.
3. The normalizing constant for E-DPPs of any (fixed) constant exponent $p
\geq \beta^{-1} = 10^{10^{13}}$ is NP-hard to approximate within a factor of
$2^{\beta pn}$. This gives a(nother) negative answer to open questions posed by
Kulesza and Taskar (2012); Ohsaka and Matsuoka (2020).

    

### [[2109.00741] End-to-End Demand Response Model Identification and Baseline Estimation with Deep Learning](http://arxiv.org/abs/2109.00741)


  This paper proposes a novel end-to-end deep learning framework that
simultaneously identifies demand baselines and the incentive-based agent demand
response model, from the net demand measurements and incentive signals. This
learning framework is modularized as two modules: 1) the decision making
process of a demand response participant is represented as a differentiable
optimization layer, which takes the incentive signal as input and predicts
user's response; 2) the baseline demand forecast is represented as a standard
neural network model, which takes relevant features and predicts user's
baseline demand. These two intermediate predictions are integrated, to form the
net demand forecast. We then propose a gradient-descent approach that
backpropagates the net demand forecast errors to update the weights of the
agent model and the weights of baseline demand forecast, jointly. We
demonstrate the effectiveness of our approach through computation experiments
with synthetic demand response traces and a large-scale real world demand
response dataset. Our results show that the approach accurately identifies the
demand response model, even without any prior knowledge about the baseline
demand.

    

### [[2109.00749] Co-Separable Nonnegative Matrix Factorization](http://arxiv.org/abs/2109.00749)


  Nonnegative matrix factorization (NMF) is a popular model in the field of
pattern recognition. It aims to find a low rank approximation for nonnegative
data M by a product of two nonnegative matrices W and H. In general, NMF is
NP-hard to solve while it can be solved efficiently under separability
assumption, which requires the columns of factor matrix are equal to columns of
the input matrix. In this paper, we generalize separability assumption based on
3-factor NMF M=P_1SP_2, and require that S is a sub-matrix of the input matrix.
We refer to this NMF as a Co-Separable NMF (CoS-NMF). We discuss some
mathematics properties of CoS-NMF, and present the relationships with other
related matrix factorizations such as CUR decomposition, generalized separable
NMF(GS-NMF), and bi-orthogonal tri-factorization (BiOR-NM3F). An optimization
model for CoS-NMF is proposed and alternated fast gradient method is employed
to solve the model. Numerical experiments on synthetic datasets, document
datasets and facial databases are conducted to verify the effectiveness of our
CoS-NMF model. Compared to state-of-the-art methods, CoS-NMF model performs
very well in co-clustering task, and preserves a good approximation to the
input data matrix as well.

    

### [[2109.00768] Direct PET Image Reconstruction Incorporating Deep Image Prior and a Forward Projection Model](http://arxiv.org/abs/2109.00768)


  Convolutional neural networks (CNNs) have recently achieved remarkable
performance in positron emission tomography (PET) image reconstruction. In
particular, CNN-based direct PET image reconstruction, which directly generates
the reconstructed image from the sinogram, has potential applicability to PET
image enhancements because it does not require image reconstruction algorithms,
which often produce some artifacts. However, these deep learning-based, direct
PET image reconstruction algorithms have the disadvantage that they require a
large number of high-quality training datasets. In this study, we propose an
unsupervised direct PET image reconstruction method that incorporates a deep
image prior framework. Our proposed method incorporates a forward projection
model with a loss function to achieve unsupervised direct PET image
reconstruction from sinograms. To compare our proposed direct reconstruction
method with the filtered back projection (FBP) and maximum likelihood
expectation maximization (ML-EM) algorithms, we evaluated using Monte Carlo
simulation data of brain [$^{18}$F]FDG PET scans. The results demonstrate that
our proposed direct reconstruction quantitatively and qualitatively outperforms
the FBP and ML-EM algorithms with respect to peak signal-to-noise ratio and
structural similarity index.

    

### [[2109.00783] VIbCReg: Variance-Invariance-better-Covariance Regularization for Self-Supervised Learning on Time Series](http://arxiv.org/abs/2109.00783)


  Self-supervised learning for image representations has recently had many
breakthroughs with respect to linear evaluation and fine-tuning evaluation.
These approaches rely on both cleverly crafted loss functions and training
setups to avoid the feature collapse problem. In this paper, we improve on the
recently proposed VICReg paper, which introduced a loss function that does not
rely on specialized training loops to converge to useful representations. Our
method improves on a covariance term proposed in VICReg, and in addition we
augment the head of the architecture by an IterNorm layer that greatly
accelerates convergence of the model. Our model achieves superior performance
on linear evaluation and fine-tuning evaluation on a subset of the UCR time
series classification archive and the PTB-XL ECG dataset.

    

### [[2109.00794] Semi-Supervised Learning using Siamese Networks](http://arxiv.org/abs/2109.00794)


  Neural networks have been successfully used as classification models yielding
state-of-the-art results when trained on a large number of labeled samples.
These models, however, are more difficult to train successfully for
semi-supervised problems where small amounts of labeled instances are available
along with a large number of unlabeled instances. This work explores a new
training method for semi-supervised learning that is based on similarity
function learning using a Siamese network to obtain a suitable embedding. The
learned representations are discriminative in Euclidean space, and hence can be
used for labeling unlabeled instances using a nearest-neighbor classifier.
Confident predictions of unlabeled instances are used as true labels for
retraining the Siamese network on the expanded training set. This process is
applied iteratively. We perform an empirical study of this iterative
self-training algorithm. For improving unlabeled predictions, local learning
with global consistency [22] is also evaluated.

    

### [[2109.00802] Anatomical-Guided Attention Enhances Unsupervised PET Image Denoising Performance](http://arxiv.org/abs/2109.00802)


  Although supervised convolutional neural networks (CNNs) often outperform
conventional alternatives for denoising positron emission tomography (PET)
images, they require many low- and high-quality reference PET image pairs.
Herein, we propose an unsupervised 3D PET image denoising method based on
anatomical information-guided attention mechanism. Our proposed magnetic
resonance-guided deep decoder (MR-GDD) utilizes the spatial details and
semantic features of MR-guidance image more effectively by introducing
encoder-decoder and deep decoder subnetworks. Moreover, the specific shapes and
patterns of the guidance image do not affect the denoised PET image, because
the guidance image is input to the network through an attention gate. Monte
Carlo simulation using the [$^{18}$F]fluoro-2-deoxy-D-glucose (FDG) shows that
the proposed method outperforms other denoising algorithms in terms of the
highest peak signal-to-noise ratio and structural similarity (28.33 dB/0.886).
Furthermore, we experimentally visualized the behavior of the optimization
process, which is often unknown in unsupervised CNN-based restoration problems.
For preclinical (using [$^{18}$F]FDG and [$^{11}$C]raclopride) and clinical
(using [$^{18}$F]florbetapir) studies, the proposed method demonstrates
state-of-the-art denoising performance while retaining spatial resolution and
quantitative accuracy, despite using only a single architecture for various
noisy PET images with 1/10th of the full counts. These results suggest that the
proposed MR-GDD can reduce PET scan times and PET tracer doses considerably
without impacting patients.

    

### [[2109.00805] Brief View and Analysis to Latest Android Security Issues and Approaches](http://arxiv.org/abs/2109.00805)


  Due to the continuous improvement of performance and functions, Android
remains the most popular operating system on mobile phone today. However,
various malicious applications bring great threats to the system. Over the past
few years, significant changes occured in both malwares and counter measures.
Specifically, malwares are continuously evolving, and advanced approaches are
adopted for more accurate detection. To keep up with the latest situation, in
this paper, we conduct a wide range of analysis, including latest malwares,
Android security features, and approaches. We also provide some finding when we
are gathering information and carrying on experiments, which we think is useful
for further researches and has not been mentioned in previous works.

    

### [[2109.00816] Deep Learning-based mitosis detection in breast cancer histologic samples](http://arxiv.org/abs/2109.00816)


  This is the submission for mitosis detection in the context of the MIDOG 2021
challenge. It is based on the two-stage objection model Faster RCNN as well as
DenseNet as a backbone for the neural network architecture. It achieves a
F1-score of 0.6645 on the Preliminary Test Phase Leaderboard.

    

### [[2109.00817] NASI: Label- and Data-agnostic Neural Architecture Search at Initialization](http://arxiv.org/abs/2109.00817)


  Recent years have witnessed a surging interest in Neural Architecture Search
(NAS). Various algorithms have been proposed to improve the search efficiency
and effectiveness of NAS, i.e., to reduce the search cost and improve the
generalization performance of the selected architectures, respectively.
However, the search efficiency of these algorithms is severely limited by the
need for model training during the search process. To overcome this limitation,
we propose a novel NAS algorithm called NAS at Initialization (NASI) that
exploits the capability of a Neural Tangent Kernel in being able to
characterize the converged performance of candidate architectures at
initialization, hence allowing model training to be completely avoided to boost
the search efficiency. Besides the improved search efficiency, NASI also
achieves competitive search effectiveness on various datasets like CIFAR-10/100
and ImageNet. Further, NASI is shown to be label- and data-agnostic under mild
conditions, which guarantees the transferability of architectures selected by
our NASI over different datasets.

    

### [[2109.00829] SlowFast Rolling-Unrolling LSTMs for Action Anticipation in Egocentric Videos](http://arxiv.org/abs/2109.00829)


  Action anticipation in egocentric videos is a difficult task due to the
inherently multi-modal nature of human actions. Additionally, some actions
happen faster or slower than others depending on the actor or surrounding
context which could vary each time and lead to different predictions. Based on
this idea, we build upon RULSTM architecture, which is specifically designed
for anticipating human actions, and propose a novel attention-based technique
to evaluate, simultaneously, slow and fast features extracted from three
different modalities, namely RGB, optical flow, and extracted objects. Two
branches process information at different time scales, i.e., frame-rates, and
several fusion schemes are considered to improve prediction accuracy. We
perform extensive experiments on EpicKitchens-55 and EGTEA Gaze+ datasets, and
demonstrate that our technique systematically improves the results of RULSTM
architecture for Top-5 accuracy metric at different anticipation times.

    

### [[2109.00846] Self-timed Reinforcement Learning using Tsetlin Machine](http://arxiv.org/abs/2109.00846)


  We present a hardware design for the learning datapath of the Tsetlin machine
algorithm, along with a latency analysis of the inference datapath. In order to
generate a low energy hardware which is suitable for pervasive artificial
intelligence applications, we use a mixture of asynchronous design techniques -
including Petri nets, signal transition graphs, dual-rail and bundled-data. The
work builds on previous design of the inference hardware, and includes an
in-depth breakdown of the automaton feedback, probability generation and
Tsetlin automata. Results illustrate the advantages of asynchronous design in
applications such as personalized healthcare and battery-powered internet of
things devices, where energy is limited and latency is an important figure of
merit. Challenges of static timing analysis in asynchronous circuits are also
addressed.

    

### [[2109.00855] Inferring feature importance with uncertainties in high-dimensional data](http://arxiv.org/abs/2109.00855)


  Estimating feature importance is a significant aspect of explaining
data-based models. Besides explaining the model itself, an equally relevant
question is which features are important in the underlying data generating
process. We present a Shapley value based framework for inferring the
importance of individual features, including uncertainty in the estimator. We
build upon the recently published feature importance measure of SAGE (Shapley
additive global importance) and introduce sub-SAGE which can be estimated
without resampling for tree-based models. We argue that the uncertainties can
be estimated from bootstrapping and demonstrate the approach for tree ensemble
methods. The framework is exemplified on synthetic data as well as
high-dimensional genomics data.

    

### [[2109.00882] MACRPO: Multi-Agent Cooperative Recurrent Policy Optimization](http://arxiv.org/abs/2109.00882)


  This work considers the problem of learning cooperative policies in
multi-agent settings with partially observable and non-stationary environments
without a communication channel. We focus on improving information sharing
between agents and propose a new multi-agent actor-critic method called
\textit{Multi-Agent Cooperative Recurrent Proximal Policy Optimization}
(MACRPO). We propose two novel ways of integrating information across agents
and time in MACRPO: First, we use a recurrent layer in critic's network
architecture and propose a new framework to use a meta-trajectory to train the
recurrent layer. This allows the network to learn the cooperation and dynamics
of interactions between agents, and also handle partial observability. Second,
we propose a new advantage function that incorporates other agents' rewards and
value functions. We evaluate our algorithm on three challenging multi-agent
environments with continuous and discrete action spaces, Deepdrive-Zero,
Multi-Walker, and Particle environment. We compare the results with several
ablations and state-of-the-art multi-agent algorithms such as QMIX and MADDPG
and also single-agent methods with shared parameters between agents such as
IMPALA and APEX. The results show superior performance against other
algorithms. The code is available online at
this https URL.

    

### [[2109.00884] Tracking Hand Hygiene Gestures with Leap Motion Controller](http://arxiv.org/abs/2109.00884)


  The process of hand washing, according to the WHO, is divided into stages
with clearly defined two handed dynamic gestures. In this paper, videos of hand
washing experts are segmented and analyzed with the goal of extracting their
corresponding features. These features can be further processed in software to
classify particular hand movements, determine whether the stages have been
successfully completed by the user and also assess the quality of washing.
Having identified the important features, a 3D gesture tracker, the Leap Motion
Controller (LEAP), was used to track and detect the hand features associated
with these stages. With the help of sequential programming and threshold
values, the hand features were combined together to detect the initiation and
completion of a sample WHO Stage 2 (Rub hands Palm to Palm). The LEAP provides
accurate raw positional data for tracking single hand gestures and two hands in
separation but suffers from occlusion when hands are in contact. Other than
hand hygiene the approaches shown here can be applied in other biomedical
applications requiring close hand gesture analysis.

    

### [[2109.00885] Unsupervised Learning for Target Tracking and Background Subtraction in Satellite Imagery](http://arxiv.org/abs/2109.00885)


  This paper describes an unsupervised machine learning methodology capable of
target tracking and background suppression via a novel dual-model approach.
``Jekyll`` produces a video bit-mask describing an estimate of the locations of
moving objects, and ``Hyde`` outputs a pseudo-background frame to subtract from
the original input image sequence. These models were trained with a
custom-modified version of Cross Entropy Loss.
Simulated data were used to compare the performance of Jekyll and Hyde
against a more traditional supervised Machine Learning approach. The results
from these comparisons show that the unsupervised methods developed are
competitive in output quality with supervised techniques, without the
associated cost of acquiring labeled training data.

    

### [[2109.00886] Contrast Limited Adaptive Histogram Equalization (CLAHE) Approach for Enhancement of the Microstructures of Friction Stir Welded Joints](http://arxiv.org/abs/2109.00886)


  Image processing algorithms are finding various applications in manufacturing
and materials industries such as identification of cracks in the fabricated
samples, calculating the geometrical properties of the given microstructure,
presence of surface defects, etc. The present work deals with the application
of Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm for
improving the quality of the microstructure images of the Friction Stir Welded
joints. The obtained results showed that the obtained value of quantitative
metric features such as Entropy value and RMS Contrast value were high which
resulted in enhanced microstructure images.

    

### [[2109.00909] Sparsifying the Update Step in Graph Neural Networks](http://arxiv.org/abs/2109.00909)


  Message-Passing Neural Networks (MPNNs), the most prominent Graph Neural
Network (GNN) framework, celebrate much success in the analysis of
graph-structured data. Concurrently, the sparsification of Neural Network
models attracts a great amount of academic and industrial interest. In this
paper, we conduct a structured study of the effect of sparsification on the
trainable part of MPNNs known as the Update step. To this end, we design a
series of models to successively sparsify the linear transform in the Update
step. Specifically, we propose the ExpanderGNN model with a tuneable
sparsification rate and the Activation-Only GNN, which has no linear transform
in the Update step. In agreement with a growing trend in the literature, the
sparsification paradigm is changed by initialising sparse neural network
architectures rather than expensively sparsifying already trained
architectures. Our novel benchmark models enable a better understanding of the
influence of the Update step on model performance and outperform existing
simplified benchmark models such as the Simple Graph Convolution. The
ExpanderGNNs, and in some cases the Activation-Only models, achieve performance
on par with their vanilla counterparts on several downstream tasks while
containing significantly fewer trainable parameters. In experiments with
matching parameter numbers, our benchmark models outperform the
state-of-the-art GNN models. Our code is publicly available at:
this https URL.

    

### [[2109.00911] BiHPF: Bilateral High-Pass Filters for Robust Deepfake Detection](http://arxiv.org/abs/2109.00911)


  The advancement in numerous generative models has a two-fold effect: a simple
and easy generation of realistic synthesized images, but also an increased risk
of malicious abuse of those images. Thus, it is important to develop a
generalized detector for synthesized images of any GAN model or object
category, including those unseen during the training phase. However, the
conventional methods heavily depend on the training settings, which cause a
dramatic decline in performance when tested with unknown domains. To resolve
the issue and obtain a generalized detection ability, we propose Bilateral
High-Pass Filters (BiHPF), which amplify the effect of the frequency-level
artifacts that are known to be found in the synthesized images of generative
models. Numerous experimental results validate that our method outperforms
other state-of-the-art methods, even when tested with unseen domains.

    

### [[2109.00921] Exploring Retraining-Free Speech Recognition for Intra-sentential Code-Switching](http://arxiv.org/abs/2109.00921)


  In this paper, we present our initial efforts for building a code-switching
(CS) speech recognition system leveraging existing acoustic models (AMs) and
language models (LMs), i.e., no training required, and specifically targeting
intra-sentential switching. To achieve such an ambitious goal, new mechanisms
for foreign pronunciation generation and language model (LM) enrichment have
been devised. Specifically, we have designed an automatic approach to obtain
high quality pronunciation of foreign language (FL) words in the native
language (NL) phoneme set using existing acoustic phone decoders and an
LSTM-based grapheme-to-phoneme (G2P) model. Improved accented pronunciations
have thus been obtained by learning foreign pronunciations directly from data.
Furthermore, a code-switching LM was deployed by converting the original NL LM
into a CS LM using translated word pairs and borrowing statistics for the NL
LM. Experimental evidence clearly demonstrates that our approach better deals
with accented foreign pronunciations than techniques based on human labeling.
Moreover, our best system achieves a 55.5% relative word error rate reduction
from 34.4%, obtained with a conventional monolingual ASR system, to 15.3% on an
intra-sentential CS task without harming the monolingual recognition accuracy.

    

### [[2109.00922] Improving Multimodal fusion via Mutual Dependency Maximisation](http://arxiv.org/abs/2109.00922)


  Multimodal sentiment analysis is a trending area of research, and the
multimodal fusion is one of its most active topic. Acknowledging humans
communicate through a variety of channels (i.e visual, acoustic, linguistic),
multimodal systems aim at integrating different unimodal representations into a
synthetic one. So far, a consequent effort has been made on developing complex
architectures allowing the fusion of these modalities. However, such systems
are mainly trained by minimising simple losses such as $L_1$ or cross-entropy.
In this work, we investigate unexplored penalties and propose a set of new
objectives that measure the dependency between modalities. We demonstrate that
our new penalties lead to a consistent improvement (up to $4.3$ on accuracy)
across a large variety of state-of-the-art models on two well-known sentiment
analysis datasets: \texttt{CMU-MOSI} and \texttt{CMU-MOSEI}. Our method not
only achieves a new SOTA on both datasets but also produces representations
that are more robust to modality drops. Finally, a by-product of our methods
includes a statistical network which can be used to interpret the high
dimensional representations learnt by the model.

    

### [[2109.00923] Auctions and Prediction Markets for Scientific Peer Review](http://arxiv.org/abs/2109.00923)


  Peer reviewed publications are considered the gold standard in certifying and
disseminating ideas that a research community considers valuable. However, we
identify two major drawbacks of the current system: (1) the overwhelming demand
for reviewers due to a large volume of submissions, and (2) the lack of
incentives for reviewers to participate and expend the necessary effort to
provide high-quality reviews. In this work, we adopt a mechanism-design
approach to propose improvements to the peer review process. We present a
two-stage mechanism which ties together the paper submission and review
process, simultaneously incentivizing high-quality reviews and high-quality
submissions. In the first stage, authors participate in a VCG auction for
review slots by submitting their papers along with a bid that represents their
expected value for having their paper reviewed. For the second stage, we
propose a novel prediction market-style mechanism (H-DIPP) building on recent
work in the information elicitation literature, which incentivizes
participating reviewers to provide honest and effortful reviews. The revenue
raised by the Stage I auction is used in Stage II to pay reviewers based on the
quality of their reviews.

    

### [[2109.00924] Parallel Multi-Graph Convolution Network For Metro Passenger Volume Prediction](http://arxiv.org/abs/2109.00924)


  Accurate prediction of metro passenger volume (number of passengers) is
valuable to realize real-time metro system management, which is a pivotal yet
challenging task in intelligent transportation. Due to the complex spatial
correlation and temporal variation of urban subway ridership behavior, deep
learning has been widely used to capture non-linear spatial-temporal
dependencies. Unfortunately, the current deep learning methods only adopt graph
convolutional network as a component to model spatial relationship, without
making full use of the different spatial correlation patterns between stations.
In order to further improve the accuracy of metro passenger volume prediction,
a deep learning model composed of Parallel multi-graph convolution and stacked
Bidirectional unidirectional Gated Recurrent Unit (PB-GRU) was proposed in this
paper. The parallel multi-graph convolution captures the origin-destination
(OD) distribution and similar flow pattern between the metro stations, while
bidirectional gated recurrent unit considers the passenger volume sequence in
forward and backward directions and learns complex temporal features. Extensive
experiments on two real-world datasets of subway passenger flow show the
efficacy of the model. Surprisingly, compared with the existing methods, PB-GRU
achieves much lower prediction error.

    

### [[2109.00928] Speaker-Conditioned Hierarchical Modeling for Automated Speech Scoring](http://arxiv.org/abs/2109.00928)


  Automatic Speech Scoring (ASS) is the computer-assisted evaluation of a
candidate's speaking proficiency in a language. ASS systems face many
challenges like open grammar, variable pronunciations, and unstructured or
semi-structured content. Recent deep learning approaches have shown some
promise in this domain. However, most of these approaches focus on extracting
features from a single audio, making them suffer from the lack of
speaker-specific context required to model such a complex task. We propose a
novel deep learning technique for non-native ASS, called speaker-conditioned
hierarchical modeling. In our technique, we take advantage of the fact that
oral proficiency tests rate multiple responses for a candidate. We extract
context vectors from these responses and feed them as additional
speaker-specific context to our network to score a particular response. We
compare our technique with strong baselines and find that such modeling
improves the model's average performance by 6.92% (maximum = 12.86%, minimum =
4.51%). We further show both quantitative and qualitative insights into the
importance of this additional context in solving the problem of ASS.

    

### [[2109.00937] A Comparative Study of Algorithms for Intelligent Traffic Signal Control](http://arxiv.org/abs/2109.00937)


  In this paper, methods have been explored to effectively optimise traffic
signal control to minimise waiting times and queue lengths, thereby increasing
traffic flow. The traffic intersection was first defined as a Markov Decision
Process, and a state representation, actions and rewards were chosen.
Simulation of Urban MObility (SUMO) was used to simulate an intersection and
then compare a Round Robin Scheduler, a Feedback Control mechanism and two
Reinforcement Learning techniques - Deep Q Network (DQN) and Advantage
Actor-Critic (A2C), as the policy for the traffic signal in the simulation
under different scenarios. Finally, the methods were tested on a simulation of
a real-world intersection in Bengaluru, India.

    

### [[2109.00946] Adversarial Robustness for Unsupervised Domain Adaptation](http://arxiv.org/abs/2109.00946)


  Extensive Unsupervised Domain Adaptation (UDA) studies have shown great
success in practice by learning transferable representations across a labeled
source domain and an unlabeled target domain with deep models. However,
previous works focus on improving the generalization ability of UDA models on
clean examples without considering the adversarial robustness, which is crucial
in real-world applications. Conventional adversarial training methods are not
suitable for the adversarial robustness on the unlabeled target domain of UDA
since they train models with adversarial examples generated by the supervised
loss function. In this work, we leverage intermediate representations learned
by multiple robust ImageNet models to improve the robustness of UDA models. Our
method works by aligning the features of the UDA model with the robust features
learned by ImageNet pre-trained models along with domain adaptation training.
It utilizes both labeled and unlabeled domains and instills robustness without
any adversarial intervention or label requirement during domain adaptation
training. Experimental results show that our method significantly improves
adversarial robustness compared to the baseline while keeping clean accuracy on
various UDA benchmarks.

    

### [[2109.00951] GAM: Explainable Visual Similarity and Classification via Gradient Activation Maps](http://arxiv.org/abs/2109.00951)


  We present Gradient Activation Maps (GAM) - a machinery for explaining
predictions made by visual similarity and classification models. By gleaning
localized gradient and activation information from multiple network layers, GAM
offers improved visual explanations, when compared to existing alternatives.
The algorithmic advantages of GAM are explained in detail, and validated
empirically, where it is shown that GAM outperforms its alternatives across
various tasks and datasets.

    

### [[2109.00959] Building Compact and Robust Deep Neural Networks with Toeplitz Matrices](http://arxiv.org/abs/2109.00959)


  Deep neural networks are state-of-the-art in a wide variety of tasks,
however, they exhibit important limitations which hinder their use and
deployment in real-world applications. When developing and training neural
networks, the accuracy should not be the only concern, neural networks must
also be cost-effective and reliable. Although accurate, large neural networks
often lack these properties. This thesis focuses on the problem of training
neural networks which are not only accurate but also compact, easy to train,
reliable and robust to adversarial examples. To tackle these problems, we
leverage the properties of structured matrices from the Toeplitz family to
build compact and secure neural networks.

    

### [[2109.00962] You Only Hear Once: A YOLO-like Algorithm for Audio Segmentation and Sound Event Detection](http://arxiv.org/abs/2109.00962)


  Audio segmentation and sound event detection are crucial topics in machine
listening that aim to detect acoustic classes and their respective boundaries.
It is useful for audio-content analysis, speech recognition, audio-indexing,
and music information retrieval. In recent years, most research articles adopt
segmentation-by-classification. This technique divides audio into small frames
and individually performs classification on these frames. In this paper, we
present a novel approach called You Only Hear Once (YOHO), which is inspired by
the YOLO algorithm popularly adopted in Computer Vision. We convert the
detection of acoustic boundaries into a regression problem instead of
frame-based classification. This is done by having separate output neurons to
detect the presence of an audio class and predict its start and end points.
YOHO obtained a higher F-measure and lower error rate than the state-of-the-art
Convolutional Recurrent Neural Network on multiple datasets. As YOHO is purely
a convolutional neural network and has no recurrent layers, it is faster during
inference. In addition, as this approach is more end-to-end and predicts
acoustic boundaries directly, it is significantly quicker during
post-processing and smoothing.

    

### [[2109.00983] Bilinear Input Normalization for Neural Networks in Financial Forecasting](http://arxiv.org/abs/2109.00983)


  Data normalization is one of the most important preprocessing steps when
building a machine learning model, especially when the model of interest is a
deep neural network. This is because deep neural network optimized with
stochastic gradient descent is sensitive to the input variable range and prone
to numerical issues. Different than other types of signals, financial
time-series often exhibit unique characteristics such as high volatility,
non-stationarity and multi-modality that make them challenging to work with,
often requiring expert domain knowledge for devising a suitable processing
pipeline. In this paper, we propose a novel data-driven normalization method
for deep neural networks that handle high-frequency financial time-series. The
proposed normalization scheme, which takes into account the bimodal
characteristic of financial multivariate time-series, requires no expert
knowledge to preprocess a financial time-series since this step is formulated
as part of the end-to-end optimization process. Our experiments, conducted with
state-of-the-arts neural networks and high-frequency data from two large-scale
limit order books coming from the Nordic and US markets, show significant
improvements over other normalization techniques in forecasting future stock
price dynamics.

    

### [[2109.00984] CrypTen: Secure Multi-Party Computation Meets Machine Learning](http://arxiv.org/abs/2109.00984)


  Secure multi-party computation (MPC) allows parties to perform computations
on data while keeping that data private. This capability has great potential
for machine-learning applications: it facilitates training of machine-learning
models on private data sets owned by different parties, evaluation of one
party's private model using another party's private data, etc. Although a range
of studies implement machine-learning models via secure MPC, such
implementations are not yet mainstream. Adoption of secure MPC is hampered by
the absence of flexible software frameworks that "speak the language" of
machine-learning researchers and engineers. To foster adoption of secure MPC in
machine learning, we present CrypTen: a software framework that exposes popular
secure MPC primitives via abstractions that are common in modern
machine-learning frameworks, such as tensor computations, automatic
differentiation, and modular neural networks. This paper describes the design
of CrypTen and measure its performance on state-of-the-art models for text
classification, speech recognition, and image classification. Our benchmarks
show that CrypTen's GPU support and high-performance communication between (an
arbitrary number of) parties allows it to perform efficient private evaluation
of modern machine-learning models under a semi-honest threat model. For
example, two parties using CrypTen can securely predict phonemes in speech
recordings using Wav2Letter faster than real-time. We hope that CrypTen will
spur adoption of secure MPC in the machine-learning community.

    

### [[2109.00998] Waveform Learning for Next-Generation Wireless Communication Systems](http://arxiv.org/abs/2109.00998)


  We propose a learning-based method for the joint design of a transmit and
receive filter, the constellation geometry and associated bit labeling, as well
as a neural network (NN)-based detector. The method maximizes an achievable
information rate, while simultaneously satisfying constraints on the adjacent
channel leakage ratio (ACLR) and peak-to-average power ratio (PAPR). This
allows control of the tradeoff between spectral containment, peak power, and
communication rate. Evaluation on an additive white Gaussian noise (AWGN)
channel shows significant reduction of ACLR and PAPR compared to a conventional
baseline relying on quadrature amplitude modulation (QAM) and
root-raised-cosine (RRC), without significant loss of information rate. When
considering a 3rd Generation Partnership Project (3GPP) multipath channel, the
learned waveform and neural receiver enable competitive or higher rates than an
orthogonal frequency division multiplexing (OFDM) baseline, while reducing the
ACLR by 10 dB and the PAPR by 2 dB. The proposed method incurs no additional
complexity on the transmitter side and might be an attractive tool for waveform
design of beyond-5G systems.

    

### [[2109.01036] MrSQM: Fast Time Series Classification with Symbolic Representations](http://arxiv.org/abs/2109.01036)


  Symbolic representations of time series have proven to be effective for time
series classification, with many recent approaches including SAX-VSM, BOSS,
WEASEL, and MrSEQL. The key idea is to transform numerical time series to
symbolic representations in the time or frequency domain, i.e., sequences of
symbols, and then extract features from these sequences. While achieving high
accuracy, existing symbolic classifiers are computationally expensive. In this
paper we present MrSQM, a new time series classifier which uses multiple
symbolic representations and efficient sequence mining, to extract important
time series features. We study four feature selection approaches on symbolic
sequences, ranging from fully supervised, to unsupervised and hybrids. We
propose a new approach for optimal supervised symbolic feature selection in
all-subsequence space, by adapting a Chi-squared bound developed for
discriminative pattern mining, to time series. Our extensive experiments on 112
datasets of the UEA/UCR benchmark demonstrate that MrSQM can quickly extract
useful features and learn accurate classifiers with the classic logistic
regression algorithm. Interestingly, we find that a very simple and fast
feature selection strategy can be highly effective as compared with more
sophisticated and expensive methods. MrSQM advances the state-of-the-art for
symbolic time series classifiers and it is an effective method to achieve high
accuracy, with fast runtime.

    

### [[2109.01044] Forecasting High-Dimensional Covariance Matrices of Asset Returns with Hybrid GARCH-LSTMs](http://arxiv.org/abs/2109.01044)


  Several academics have studied the ability of hybrid models mixing univariate
Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models and
neural networks to deliver better volatility predictions than purely
econometric models. Despite presenting very promising results, the
generalization of such models to the multivariate case has yet to be studied.
Moreover, very few papers have examined the ability of neural networks to
predict the covariance matrix of asset returns, and all use a rather small
number of assets, thus not addressing what is known as the curse of
dimensionality. The goal of this paper is to investigate the ability of hybrid
models, mixing GARCH processes and neural networks, to forecast covariance
matrices of asset returns. To do so, we propose a new model, based on
multivariate GARCHs that decompose volatility and correlation predictions. The
volatilities are here forecast using hybrid neural networks while correlations
follow a traditional econometric process. After implementing the models in a
minimum variance portfolio framework, our results are as follows. First, the
addition of GARCH parameters as inputs is beneficial to the model proposed.
Second, the use of one-hot-encoding to help the neural network differentiate
between each stock improves the performance. Third, the new model proposed is
very promising as it not only outperforms the equally weighted portfolio, but
also by a significant margin its econometric counterpart that uses univariate
GARCHs to predict the volatilities.

    

### [[2109.01050] Characterizing possible failure modes in physics-informed neural networks](http://arxiv.org/abs/2109.01050)


  Recent work in scientific machine learning has developed so-called
physics-informed neural network (PINN) models. The typical approach is to
incorporate physical domain knowledge as soft constraints on an empirical loss
function and use existing machine learning methodologies to train the model. We
demonstrate that, while existing PINN methodologies can learn good models for
relatively trivial problems, they can easily fail to learn relevant physical
phenomena even for simple PDEs. In particular, we analyze several distinct
situations of widespread physical interest, including learning differential
equations with convection, reaction, and diffusion operators. We provide
evidence that the soft regularization in PINNs, which involves differential
operators, can introduce a number of subtle problems, including making the
problem ill-conditioned. Importantly, we show that these possible failure modes
are not due to the lack of expressivity in the NN architecture, but that the
PINN's setup makes the loss landscape very hard to optimize. We then describe
two promising solutions to address these failure modes. The first approach is
to use curriculum regularization, where the PINN's loss term starts from a
simple PDE regularization, and becomes progressively more complex as the NN
gets trained. The second approach is to pose the problem as a
sequence-to-sequence learning task, rather than learning to predict the entire
space-time at once. Extensive testing shows that we can achieve up to 1-2
orders of magnitude lower error with these methods as compared to regular PINN
training.

    

### [[2109.01051] Can Error Mitigation Improve Trainability of Noisy Variational Quantum Algorithms?](http://arxiv.org/abs/2109.01051)


  Variational Quantum Algorithms (VQAs) are widely viewed as the best hope for
near-term quantum advantage. However, recent studies have shown that noise can
severely limit the trainability of VQAs, e.g., by exponentially flattening the
cost landscape and suppressing the magnitudes of cost gradients. Error
Mitigation (EM) shows promise in reducing the impact of noise on near-term
devices. Thus, it is natural to ask whether EM can improve the trainability of
VQAs. In this work, we first show that, for a broad class of EM strategies,
exponential cost concentration cannot be resolved without committing
exponential resources elsewhere. This class of strategies includes as special
cases Zero Noise Extrapolation, Virtual Distillation, Probabilistic Error
Cancellation, and Clifford Data Regression. Second, we perform analytical and
numerical analysis of these EM protocols, and we find that some of them (e.g.,
Virtual Distillation) can make it harder to resolve cost function values
compared to running no EM at all. As a positive result, we do find numerical
evidence that Clifford Data Regression (CDR) can aid the training process in
certain settings where cost concentration is not too severe. Our results show
that care should be taken in applying EM protocols as they can either worsen or
not improve trainability. On the other hand, our positive results for CDR
highlight the possibility of engineering error mitigation methods to improve
trainability.

    

### [[2109.01064] Lower Bounds on the Total Variation Distance Between Mixtures of Two Gaussians](http://arxiv.org/abs/2109.01064)


  Mixtures of high dimensional Gaussian distributions have been studied
extensively in statistics and learning theory. While the total variation
distance appears naturally in the sample complexity of distribution learning,
it is analytically difficult to obtain tight lower bounds for mixtures.
Exploiting a connection between total variation distance and the characteristic
function of the mixture, we provide fairly tight functional approximations.
This enables us to derive new lower bounds on the total variation distance
between pairs of two-component Gaussian mixtures that have a shared covariance
matrix.

    

### [[2109.01077] Optimal subgroup selection](http://arxiv.org/abs/2109.01077)


  In clinical trials and other applications, we often see regions of the
feature space that appear to exhibit interesting behaviour, but it is unclear
whether these observed phenomena are reflected at the population level.
Focusing on a regression setting, we consider the subgroup selection challenge
of identifying a region of the feature space on which the regression function
exceeds a pre-determined threshold. We formulate the problem as one of
constrained optimisation, where we seek a low-complexity, data-dependent
selection set on which, with a guaranteed probability, the regression function
is uniformly at least as large as the threshold; subject to this constraint, we
would like the region to contain as much mass under the marginal feature
distribution as possible. This leads to a natural notion of regret, and our
main contribution is to determine the minimax optimal rate for this regret in
both the sample size and the Type I error probability. The rate involves a
delicate interplay between parameters that control the smoothness of the
regression function, as well as exponents that quantify the extent to which the
optimal selection set at the population level can be approximated by families
of well-behaved subsets. Finally, we expand the scope of our previous results
by illustrating how they may be generalised to a treatment and control setting,
where interest lies in the heterogeneous treatment effect.

    

### [[2109.01080] Optimization and Sampling Under Continuous Symmetry: Examples and Lie Theory](http://arxiv.org/abs/2109.01080)


  In the last few years, the notion of symmetry has provided a powerful and
essential lens to view several optimization or sampling problems that arise in
areas such as theoretical computer science, statistics, machine learning,
quantum inference, and privacy. Here, we present two examples of nonconvex
problems in optimization and sampling where continuous symmetries play --
implicitly or explicitly -- a key role in the development of efficient
algorithms. These examples rely on deep and hidden connections between
nonconvex symmetric manifolds and convex polytopes, and are heavily
generalizable. To formulate and understand these generalizations, we then
present an introduction to Lie theory -- an indispensable mathematical toolkit
for capturing and working with continuous symmetries. We first present the
basics of Lie groups, Lie algebras, and the adjoint actions associated with
them, and we also mention the classification theorem for Lie algebras.
Subsequently, we present Kostant's convexity theorem and show how it allows us
to reduce linear optimization problems over orbits of Lie groups to linear
optimization problems over polytopes. Finally, we present the Harish-Chandra
and the Harish-Chandra--Itzykson--Zuber (HCIZ) formulas, which convert
partition functions (integrals) over Lie groups into sums over the
corresponding (discrete) Weyl groups, enabling efficient sampling algorithms.

    

### [[2109.01081] Transformer Networks for Data Augmentation of Human Physical Activity Recognition](http://arxiv.org/abs/2109.01081)


  Data augmentation is a widely used technique in classification to increase
data used in training. It improves generalization and reduces amount of
annotated human activity data needed for training which reduces labour and time
needed with the dataset. Sensor time-series data, unlike images, cannot be
augmented by computationally simple transformation algorithms. State of the art
models like Recurrent Generative Adversarial Networks (RGAN) are used to
generate realistic synthetic data. In this paper, transformer based generative
adversarial networks which have global attention on data, are compared on
PAMAP2 and Real World Human Activity Recognition data sets with RGAN. The newer
approach provides improvements in time and savings in computational resources
needed for data augmentation than previous approach.

    

### [[2109.01084] Text Classification for Predicting Multi-level Product Categories](http://arxiv.org/abs/2109.01084)


  In an online shopping platform, a detailed classification of the products
facilitates user navigation. It also helps online retailers keep track of the
price fluctuations in a certain industry or special discounts on a specific
product category. Moreover, an automated classification system may help to
pinpoint incorrect or subjective categories suggested by an operator. In this
study, we focus on product title classification of the grocery products. We
perform a comprehensive comparison of six different text classification models
to establish a strong baseline for this task, which involves testing both
traditional and recent machine learning methods. In our experiments, we
investigate the generalizability of the trained models to the products of other
online retailers, the dynamic masking of infeasible subcategories for
pretrained language models, and the benefits of incorporating product titles in
multiple languages. Our numerical results indicate that dynamic masking of
subcategories is effective in improving prediction accuracy. In addition, we
observe that using bilingual product titles is generally beneficial, and neural
network-based models perform significantly better than SVM and XGBoost models.
Lastly, we investigate the reasons for the misclassified products and propose
future research directions to further enhance the prediction models.

    

### [[2109.01085] Cascade RCNN for MIDOG Challenge](http://arxiv.org/abs/2109.01085)


  Mitotic counts are one of the key indicators of breast cancer prognosis.
However, accurate mitotic cell counting is still a difficult problem and is
labourious. Automated methods have been proposed for this task, but are usually
dependent on the training images and show poor performance on unseen domains.
In this work, we present a multi-stage mitosis detection method based on a
Cascade RCNN developed to be sequentially more selective against false
positives. On the preliminary test set, the algorithm scores an F1-score of
0.7492.

    

### [[2109.01087] On-target Adaptation](http://arxiv.org/abs/2109.01087)


  Domain adaptation seeks to mitigate the shift between training on the
\emph{source} domain and testing on the \emph{target} domain. Most adaptation
methods rely on the source data by joint optimization over source data and
target data. Source-free methods replace the source data with a source model by
fine-tuning it on target. Either way, the majority of the parameter updates for
the model representation and the classifier are derived from the source, and
not the target. However, target accuracy is the goal, and so we argue for
optimizing as much as possible on the target data. We show significant
improvement by on-target adaptation, which learns the representation purely
from target data while taking only the source predictions for supervision. In
the long-tailed classification setting, we show further improvement by
on-target class distribution learning, which learns the (im)balance of classes
from target data.

    

### [[2109.01093] What Users Want? WARHOL: A Generative Model for Recommendation](http://arxiv.org/abs/2109.01093)


  Current recommendation approaches help online merchants predict, for each
visiting user, which subset of their existing products is the most relevant.
However, besides being interested in matching users with existing products,
merchants are also interested in understanding their users' underlying
preferences. This could indeed help them produce or acquire better matching
products in the future. We argue that existing recommendation models cannot
directly be used to predict the optimal combination of features that will make
new products serve better the needs of the target audience. To tackle this, we
turn to generative models, which allow us to learn explicitly distributions
over product feature combinations both in text and visual space. We develop
WARHOL, a product generation and recommendation architecture that takes as
input past user shopping activity and generates relevant textual and visual
descriptions of novel products. We show that WARHOL can approach the
performance of state-of-the-art recommendation models, while being able to
generate entirely new products that are relevant to the given user profiles.

    

### [[2109.01097] The Functional Correspondence Problem](http://arxiv.org/abs/2109.01097)


  The ability to find correspondences in visual data is the essence of most
computer vision tasks. But what are the right correspondences? The task of
visual correspondence is well defined for two different images of same object
instance. In case of two images of objects belonging to same category, visual
correspondence is reasonably well-defined in most cases. But what about
correspondence between two objects of completely different category -- e.g., a
shoe and a bottle? Does there exist any correspondence? Inspired by humans'
ability to: (a) generalize beyond semantic categories and; (b) infer functional
affordances, we introduce the problem of functional correspondences in this
paper. Given images of two objects, we ask a simple question: what is the set
of correspondences between these two images for a given task? For example, what
are the correspondences between a bottle and shoe for the task of pounding or
the task of pouring. We introduce a new dataset: FunKPoint that has ground
truth correspondences for 10 tasks and 20 object categories. We also introduce
a modular task-driven representation for attacking this problem and demonstrate
that our learned representation is effective for this task. But most
importantly, because our supervision signal is not bound by semantics, we show
that our learned representation can generalize better on few-shot
classification problem. We hope this paper will inspire our community to think
beyond semantics and focus more on cross-category generalization and learning
representations for robotics tasks.

    

### [[2109.01105] Solving Inverse Problems with Conditional-GAN Prior via Fast Network-Projected Gradient Descent](http://arxiv.org/abs/2109.01105)


  The projected gradient descent (PGD) method has shown to be effective in
recovering compressed signals described in a data-driven way by a generative
model, i.e., a generator which has learned the data distribution. Further
reconstruction improvements for such inverse problems can be achieved by
conditioning the generator on the measurement. The boundary equilibrium
generative adversarial network (BEGAN) implements an equilibrium based loss
function and an auto-encoding discriminator to better balance the performance
of the generator and the discriminator. In this work we investigate a
network-based projected gradient descent (NPGD) algorithm for
measurement-conditional generative models to solve the inverse problem much
faster than regular PGD. We combine the NPGD with conditional GAN/BEGAN to
evaluate their effectiveness in solving compressed sensing type problems. Our
experiments on the MNIST and CelebA datasets show that the combination of
measurement conditional model with NPGD works well in recovering the compressed
signal while achieving similar or in some cases even better performance along
with a much faster reconstruction. The achieved reconstruction speed-up in our
experiments is up to 140-175.

    

### [[2109.01115] Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation](http://arxiv.org/abs/2109.01115)


  We study the problem of learning a range of vision-based manipulation tasks
from a large offline dataset of robot interaction. In order to accomplish this,
humans need easy and effective ways of specifying tasks to the robot. Goal
images are one popular form of task specification, as they are already grounded
in the robot's observation space. However, goal images also have a number of
drawbacks: they are inconvenient for humans to provide, they can over-specify
the desired behavior leading to a sparse reward signal, or under-specify task
information in the case of non-goal reaching tasks. Natural language provides a
convenient and flexible alternative for task specification, but comes with the
challenge of grounding language in the robot's observation space. To scalably
learn this grounding we propose to leverage offline robot datasets (including
highly sub-optimal, autonomously collected data) with crowd-sourced natural
language labels. With this data, we learn a simple classifier which predicts if
a change in state completes a language instruction. This provides a
language-conditioned reward function that can then be used for offline
multi-task RL. In our experiments, we find that on language-conditioned
manipulation tasks our approach outperforms both goal-image specifications and
language conditioned imitation techniques by more than 25%, and is able to
perform visuomotor tasks from natural language, such as "open the right drawer"
and "move the stapler", on a Franka Emika Panda robot.

    

### [[2109.01116] An Empirical Study of Graph Contrastive Learning](http://arxiv.org/abs/2109.01116)


  Graph Contrastive Learning (GCL) establishes a new paradigm for learning
graph representations without human annotations. Although remarkable progress
has been witnessed recently, the success behind GCL is still left somewhat
mysterious. In this work, we first identify several critical design
considerations within a general GCL paradigm, including augmentation functions,
contrasting modes, contrastive objectives, and negative mining techniques.
Then, to understand the interplay of different GCL components, we conduct
extensive, controlled experiments over a set of benchmark tasks on datasets
across various domains. Our empirical studies suggest a set of general receipts
for effective GCL, e.g., simple topology augmentations that produce sparse
graph views bring promising performance improvements; contrasting modes should
be aligned with the granularities of end tasks. In addition, to foster future
research and ease the implementation of GCL algorithms, we develop an
easy-to-use library PyGCL, featuring modularized CL components, standardized
evaluation, and experiment management. We envision this work to provide useful
empirical evidence of effective GCL algorithms and offer several insights for
future research.

    

### [[2109.01120] Automatic Diagnosis of Schizophrenia using EEG Signals and CNN-LSTM Models](http://arxiv.org/abs/2109.01120)


  Schizophrenia (SZ) is a mental disorder whereby due to the secretion of
specific chemicals in the brain, the function of some brain regions is out of
balance, leading to the lack of coordination between thoughts, actions, and
emotions. This study provides various intelligent Deep Learning (DL)-based
methods for automated SZ diagnosis via EEG signals. The obtained results are
compared with those of conventional intelligent methods. In order to implement
the proposed methods, the dataset of the Institute of Psychiatry and Neurology
in Warsaw, Poland, has been used. First, EEG signals are divided into
25-seconds time frames and then were normalized by z-score or norm L2. In the
classification step, two different approaches are considered for SZ diagnosis
via EEG signals. In this step, the classification of EEG signals is first
carried out by conventional DL methods, e.g., KNN, DT, SVM, Bayes, bagging, RF,
and ET. Various proposed DL models, including LSTMs, 1D-CNNs, and 1D-CNN-LSTMs,
are used in the following. In this step, the DL models were implemented and
compared with different activation functions. Among the proposed DL models, the
CNN-LSTM architecture has had the best performance. In this architecture, the
ReLU activation function and the z-score and L2 combined normalization are
used. The proposed CNN-LSTM model has achieved an accuracy percentage of
99.25\%, better than the results of most former studies in this field. It is
worth mentioning that in order to perform all simulations, the k-fold
cross-validation method with k=5 has been used.

    

### [[2109.01123] Benchmarking the Robustness of Instance Segmentation Models](http://arxiv.org/abs/2109.01123)


  This paper presents a comprehensive evaluation of instance segmentation
models with respect to real-world image corruptions and out-of-domain image
collections, e.g. datasets collected with different set-ups than the training
datasets the models learned from. The out-of-domain image evaluation shows the
generalization capability of models, an essential aspect of real-world
applications, and an extensively studied topic of domain adaptation. These
presented robustness and generalization evaluations are important when
designing instance segmentation models for real-world applications and picking
an off-the-shelf pretrained model to directly use for the task at hand.
Specifically, this benchmark study includes state-of-the-art network
architectures, network backbones, normalization layers, models trained starting
from scratch or ImageNet pretrained networks, and the effect of multi-task
training on robustness and generalization. Through this study, we gain several
insights e.g. we find that normalization layers play an essential role in
robustness, ImageNet pretraining does not help the robustness and the
generalization of models, excluding JPEG corruption, and network backbones and
copy-paste augmentations affect robustness significantly.

    

### [[2109.01134] Learning to Prompt for Vision-Language Models](http://arxiv.org/abs/2109.01134)


  Vision-language pre-training has recently emerged as a promising alternative
for representation learning. It shifts from the tradition of using images and
discrete labels for learning a fixed set of weights, seen as visual concepts,
to aligning images and raw text for two separate encoders. Such a paradigm
benefits from a broader source of supervision and allows zero-shot transfer to
downstream tasks since visual concepts can be diametrically generated from
natural language, known as prompt. In this paper, we identify that a major
challenge of deploying such models in practice is prompt engineering. This is
because designing a proper prompt, especially for context words surrounding a
class name, requires domain expertise and typically takes a significant amount
of time for words tuning since a slight change in wording could have a huge
impact on performance. Moreover, different downstream tasks require specific
designs, further hampering the efficiency of deployment. To overcome this
challenge, we propose a novel approach named context optimization (CoOp). The
main idea is to model context in prompts using continuous representations and
perform end-to-end learning from data while keeping the pre-trained parameters
fixed. In this way, the design of task-relevant prompts can be fully automated.
Experiments on 11 datasets show that CoOp effectively turns pre-trained
vision-language models into data-efficient visual learners, requiring as few as
one or two shots to beat hand-crafted prompts with a decent margin and able to
gain significant improvements when using more shots (e.g., at 16 shots the
average gain is around 17% with the highest reaching over 50%). CoOp also
exhibits strong robustness to distribution shift.

    

### [[2109.01135] Sequence-to-Sequence Learning with Latent Neural Grammars](http://arxiv.org/abs/2109.01135)


  Sequence-to-sequence learning with neural networks has become the de facto
standard for sequence prediction tasks. This approach typically models the
local distribution over the next word with a powerful neural network that can
condition on arbitrary context. While flexible and performant, these models
often require large datasets for training and can fail spectacularly on
benchmarks designed to test for compositional generalization. This work
explores an alternative, hierarchical approach to sequence-to-sequence learning
with quasi-synchronous grammars, where each node in the target tree is
transduced by a node in the source tree. Both the source and target trees are
treated as latent and induced during training. We develop a neural
parameterization of the grammar which enables parameter sharing over the
combinatorial space of derivation rules without the need for manual feature
engineering. We apply this latent neural grammar to various domains -- a
diagnostic language navigation task designed to test for compositional
generalization (SCAN), style transfer, and small-scale machine translation --
and find that it performs respectably compared to standard baselines.

    

### [[1812.11039] On the Benefit of Width for Neural Networks: Disappearance of Bad Basins](http://arxiv.org/abs/1812.11039)


  Wide networks are often believed to have a nice optimization landscape, but
what rigorous results can we prove? To understand the benefit of width, it is
important to identify the difference between wide and narrow networks. In this
work, we prove that from narrow to wide networks, there is a phase transition
from having sub-optimal basins to no sub-optimal basins. Specifically, we prove
two results: on the positive side, for any continuous activation functions, the
loss surface of a class of wide networks has no sub-optimal basins, where
"basin" is defined as the set-wise strict local minimum; on the negative side,
for a large class of networks with width below a threshold, we construct strict
local minima that are not global. These two results together show the phase
transition from narrow to wide networks.

    

### [[1902.07848] Gradient Scheduling with Global Momentum for Non-IID Data Distributed Asynchronous Training](http://arxiv.org/abs/1902.07848)


  Distributed asynchronous offline training has received widespread attention
in recent years because of its high performance on large-scale data and complex
models. As data are distributed from cloud-centric to edge nodes, a big
challenge for distributed machine learning systems is how to handle native and
natural non-independent and identically distributed (non-IID) data for
training. Previous asynchronous training methods do not have a satisfying
performance on non-IID data because it would result in that the training
process fluctuates greatly which leads to an abnormal convergence. We propose a
gradient scheduling algorithm with partly averaged gradients and global
momentum (GSGM) for non-IID data distributed asynchronous training. Our key
idea is to apply global momentum and local average to the biased gradient after
scheduling, in order to make the training process steady. Experimental results
show that for non-IID data training under the same experimental conditions,
GSGM on popular optimization algorithms can achieve a 20% increase in training
stability with a slight improvement in accuracy on Fashion-Mnist and CIFAR-10
datasets. Meanwhile, when expanding distributed scale on CIFAR-100 dataset that
results in sparse data distribution, GSGM can perform a 37% improvement on
training stability. Moreover, only GSGM can converge well when the number of
computing nodes grows to 30, compared to the state-of-the-art distributed
asynchronous algorithms. At the same time, GSGM is robust to different degrees
of non-IID data.

    

### [[1907.00325] Random Forests for Adaptive Nearest Neighbor Estimation of Information-Theoretic Quantities](http://arxiv.org/abs/1907.00325)


  Information-theoretic quantities, such as conditional entropy and mutual
information, are critical data summaries for quantifying uncertainty. Current
widely used approaches for computing such quantities rely on nearest neighbor
methods and exhibit both strong performance and theoretical guarantees in
certain simple scenarios. However, existing approaches fail in high-dimensional
settings and when different features are measured on different scales.We
propose decision forest-based adaptive nearest neighbor estimators and show
that they are able to effectively estimate posterior probabilities, conditional
entropies, and mutual information even in the aforementioned settings.We
provide an extensive study of efficacy for classification and posterior
probability estimation, and prove certain forest-based approaches to be
consistent estimators of the true posteriors and derived information-theoretic
quantities under certain assumptions. In a real-world connectome application,
we quantify the uncertainty about neuron type given various cellular features
in the Drosophila larva mushroom body, a key challenge for modern neuroscience.

    

### [[1912.13213] A Modern Introduction to Online Learning](http://arxiv.org/abs/1912.13213)


  In this monograph, I introduce the basic concepts of Online Learning through
a modern view of Online Convex Optimization. Here, online learning refers to
the framework of regret minimization under worst-case assumptions. I present
first-order and second-order algorithms for online learning with convex losses,
in Euclidean and non-Euclidean settings. All the algorithms are clearly
presented as instantiation of Online Mirror Descent or
Follow-The-Regularized-Leader and their variants. Particular attention is given
to the issue of tuning the parameters of the algorithms and learning in
unbounded domains, through adaptive and parameter-free online learning
algorithms. Non-convex losses are dealt through convex surrogate losses and
through randomization. The bandit setting is also briefly discussed, touching
on the problem of adversarial and stochastic multi-armed bandits. These notes
do not require prior knowledge of convex analysis and all the required
mathematical tools are rigorously explained. Moreover, all the proofs have been
carefully chosen to be as simple and as short as possible.

    

### [[2005.08479] Large-Scale Secure XGB for Vertical Federated Learning](http://arxiv.org/abs/2005.08479)


  Privacy-preserving machine learning has drawn increasingly attention
recently, especially with kinds of privacy regulations come into force. Under
such situation, Federated Learning (FL) appears to facilitate
privacy-preserving joint modeling among multiple parties. Although many
federated algorithms have been extensively studied, there is still a lack of
secure and practical gradient tree boosting models (e.g., XGB) in literature.
In this paper, we aim to build large-scale secure XGB under vertically
federated learning setting. We guarantee data privacy from three aspects.
Specifically, (i) we employ secure multi-party computation techniques to avoid
leaking intermediate information during training, (ii) we store the output
model in a distributed manner in order to minimize information release, and
(iii) we provide a novel algorithm for secure XGB predict with the distributed
model. Furthermore, by proposing secure permutation protocols, we can improve
the training efficiency and make the framework scale to large dataset. We
conduct extensive experiments on both public datasets and real-world datasets,
and the results demonstrate that our proposed XGB models provide not only
competitive accuracy but also practical performance.

    

### [[2006.07817] Topology-aware Differential Privacy for Decentralized Image Classification](http://arxiv.org/abs/2006.07817)


  In this paper, we design Top-DP, a novel solution to optimize the
differential privacy protection of decentralized image classification systems.
The key insight of our solution is to leverage the unique features of
decentralized communication topologies to reduce the noise scale and improve
the model usability. (1) We enhance the DP-SGD algorithm with this
topology-aware noise reduction strategy, and integrate the time-aware noise
decay technique. (2) We design two novel learning protocols (synchronous and
asynchronous) to protect systems with different network connectivities and
topologies. We formally analyze and prove the DP requirement of our proposed
solutions. Experimental evaluations demonstrate that our solution achieves a
better trade-off between usability and privacy than prior works. To the best of
our knowledge, this is the first DP optimization work from the perspective of
network topologies.

    

### [[2007.04074] Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning](http://arxiv.org/abs/2007.04074)


  Automated Machine Learning (AutoML) supports practitioners and researchers
with the tedious task of designing machine learning pipelines and has recently
achieved substantial success. In this paper we introduce new AutoML approaches
motivated by our winning submission to the second ChaLearn AutoML challenge. We
develop PoSH Auto-sklearn, which enables AutoML systems to work well on large
datasets under rigid time limits using a new, simple and meta-feature-free
meta-learning technique and employs a successful bandit strategy for budget
allocation. However, PoSH Auto-sklearn introduces even more ways of running
AutoML and might make it harder for users to set it up correctly. Therefore, we
also go one step further and study the design space of AutoML itself, proposing
a solution towards truly hands-free AutoML. Together, these changes give rise
to the next generation of our AutoML system, Auto-sklearn 2.0 . We verify the
improvements by these additions in a large experimental study on 39 AutoML
benchmark datasets and conclude the paper by comparing to other popular AutoML
frameworks and Auto-sklearn 1.0 , reducing the relative error by up to a factor
of 4.5, and yielding a performance in 10 minutes that is substantially better
than what Auto-sklearn 1.0 achieves within an hour.

    

### [[2007.06711] A Bayesian Evaluation Framework for Subjectively Annotated Visual Recognition Tasks](http://arxiv.org/abs/2007.06711)


  An interesting development in automatic visual recognition has been the
emergence of tasks where it is not possible to assign objective labels to
images, yet still feasible to collect annotations that reflect human judgements
about them. Machine learning-based predictors for these tasks rely on
supervised training that models the behavior of the annotators, i.e., what
would the average person's judgement be for an image? A key open question for
this type of work, especially for applications where inconsistency with human
behavior can lead to ethical lapses, is how to evaluate the epistemic
uncertainty of trained predictors, i.e., the uncertainty that comes from the
predictor's model. We propose a Bayesian framework for evaluating black box
predictors in this regime, agnostic to the predictor's internal structure. The
framework specifies how to estimate the epistemic uncertainty that comes from
the predictor with respect to human labels by approximating a conditional
distribution and producing a credible interval for the predictions and their
measures of performance. The framework is successfully applied to four image
classification tasks that use subjective human judgements: facial beauty
assessment, social attribute assignment, apparent age estimation, and ambiguous
scene labeling.

    

### [[2007.07804] Newton Optimization on Helmholtz Decomposition for Continuous Games](http://arxiv.org/abs/2007.07804)


  Many learning problems involve multiple agents optimizing different
interactive functions. In these problems, the standard policy gradient
algorithms fail due to the non-stationarity of the setting and the different
interests of each agent. In fact, algorithms must take into account the complex
dynamics of these systems to guarantee rapid convergence towards a (local) Nash
equilibrium. In this paper, we propose NOHD (Newton Optimization on Helmholtz
Decomposition), a Newton-like algorithm for multi-agent learning problems based
on the decomposition of the dynamics of the system in its irrotational
(Potential) and solenoidal (Hamiltonian) component. This method ensures
quadratic convergence in purely irrotational systems and pure solenoidal
systems. Furthermore, we show that NOHD is attracted to stable fixed points in
general multi-agent systems and repelled by strict saddle ones. Finally, we
empirically compare the NOHD's performance with that of state-of-the-art
algorithms on some bimatrix games and in a continuous Gridworld environment.

    

### [[2007.14717] Almost exact recovery in noisy semi-supervised learning](http://arxiv.org/abs/2007.14717)


  Graph-based semi-supervised learning methods combine the graph structure and
labeled data to classify unlabeled data. In this work, we study the effect of a
noisy oracle on classification. In particular, we derive the Maximum A
Posteriori (MAP) estimator for clustering a Degree Corrected Stochastic Block
Model (DC-SBM) when a noisy oracle reveals a fraction of the labels. We then
propose an algorithm derived from a continuous relaxation of the MAP, and we
establish its consistency. Numerical experiments show that our approach
achieves promising performance on synthetic and real data sets, even in the
case of very noisy labeled data.

    

### [[2010.03180] Not All Datasets Are Born Equal: On Heterogeneous Data and Adversarial Examples](http://arxiv.org/abs/2010.03180)


  Recent work on adversarial learning has focused mainly on neural networks and
domains where those networks excel, such as computer vision, or audio
processing. The data in these domains is typically homogeneous, whereas
heterogeneous tabular datasets domains remain underexplored despite their
prevalence. When searching for adversarial patterns within heterogeneous input
spaces, an attacker must simultaneously preserve the complex domain-specific
validity rules of the data, as well as the adversarial nature of the identified
samples. As such, applying adversarial manipulations to heterogeneous datasets
has proved to be a challenging task, and no generic attack method was suggested
thus far. We, however, argue that machine learning models trained on
heterogeneous tabular data are as susceptible to adversarial manipulations as
those trained on continuous or homogeneous data such as images. To support our
claim, we introduce a generic optimization framework for identifying
adversarial perturbations in heterogeneous input spaces. We define
distribution-aware constraints for preserving the consistency of the
adversarial examples and incorporate them by embedding the heterogeneous input
into a continuous latent space. Due to the nature of the underlying datasets We
focus on $\ell_0$ perturbations, and demonstrate their applicability in real
life. We demonstrate the effectiveness of our approach using three datasets
from different content domains. Our results demonstrate that despite the
constraints imposed on input validity in heterogeneous datasets, machine
learning models trained using such data are still equally susceptible to
adversarial examples.

    

### [[2010.04254] Exploring Sensitivity of ICF Outputs to Design Parameters in Experiments Using Machine Learning](http://arxiv.org/abs/2010.04254)


  Building a sustainable burn platform in inertial confinement fusion (ICF)
requires an understanding of the complex coupling of physical processes and the
effects that key experimental design changes have on implosion performance.
While simulation codes are used to model ICF implosions, incomplete physics and
the need for approximations deteriorate their predictive capability.
Identification of relationships between controllable design inputs and
measurable outcomes can help guide the future design of experiments and
development of simulation codes, which can potentially improve the accuracy of
the computational models used to simulate ICF implosions. In this paper, we
leverage developments in machine learning (ML) and methods for ML feature
importance/sensitivity analysis to identify complex relationships in ways that
are difficult to process using expert judgment alone. We present work using
random forest (RF) regression for prediction of yield, velocity, and other
experimental outcomes given a suite of design parameters, along with an
assessment of important relationships and uncertainties in the prediction
model. We show that RF models are capable of learning and predicting on ICF
experimental data with high accuracy, and we extract feature importance metrics
that provide insight into the physical significance of different controllable
design inputs for various ICF design configurations. These results can be used
to augment expert intuition and simulation results for optimal design of future
ICF experiments.

    

### [[2010.09246] Taking Over the Stock Market: Adversarial Perturbations Against Algorithmic Traders](http://arxiv.org/abs/2010.09246)


  In recent years, machine learning has become prevalent in numerous tasks,
including algorithmic trading. Stock market traders utilize machine learning
models to predict the market's behavior and execute an investment strategy
accordingly. However, machine learning models have been shown to be susceptible
to input manipulations called adversarial examples. Despite this risk, the
trading domain remains largely unexplored in the context of adversarial
learning. In this study, we present a realistic scenario in which an attacker
influences algorithmic trading systems by using adversarial learning techniques
to manipulate the input data stream in real time. The attacker creates a
universal perturbation that is agnostic to the target model and time of use,
which, when added to the input stream, remains imperceptible. We evaluate our
attack on a real-world market data stream and target three different trading
algorithms. We show that when added to the input stream, our perturbation can
fool the trading algorithms at future unseen data points, in both white-box and
black-box settings. Finally, we present various mitigation methods and discuss
their limitations, which stem from the algorithmic trading domain. We believe
that these findings should serve as an alert to the finance community about the
threats in this area and promote further research on the risks associated with
using automated learning models in the trading domain.

    

### [[2010.12809] Stop Bugging Me! Evading Modern-Day Wiretapping Using Adversarial Perturbations](http://arxiv.org/abs/2010.12809)


  Mass surveillance systems for voice over IP (VoIP) conversations pose a great
risk to privacy. These automated systems use learning models to analyze
conversations, and calls that involve specific topics are routed to a human
agent for further examination. In this study, we present an
adversarial-learning-based framework for privacy protection for VoIP
conversations. We present a novel method that finds a universal adversarial
perturbation (UAP), which, when added to the audio stream, prevents an
eavesdropper from automatically detecting the conversation's topic. As shown in
our experiments, the UAP is agnostic to the speaker or audio length, and its
volume can be changed in real time, as needed. Our real-world solution uses a
Teensy microcontroller that acts as an external microphone and adds the UAP to
the audio in real time. We examine different speakers, VoIP applications
(Skype, Zoom, Slack, and Google Meet), and audio lengths. Our results in the
real world suggest that our approach is a feasible solution for privacy
protection.

    

### [[2010.13713] Self-supervised Human Activity Recognition by Learning to Predict Cross-Dimensional Motion](http://arxiv.org/abs/2010.13713)


  We propose the use of self-supervised learning for human activity recognition
with smartphone accelerometer data. Our proposed solution consists of two
steps. First, the representations of unlabeled input signals are learned by
training a deep convolutional neural network to predict a segment of
accelerometer values. Our model exploits a novel scheme to leverage past and
present motion in x and y dimensions, as well as past values of the z axis to
predict values in the z dimension. This cross-dimensional prediction approach
results in effective pretext training with which our model learns to extract
strong representations. Next, we freeze the convolution blocks and transfer the
weights to our downstream network aimed at human activity recognition. For this
task, we add a number of fully connected layers to the end of the frozen
network and train the added layers with labeled accelerometer signals to learn
to classify human activities. We evaluate the performance of our method on
three publicly available human activity datasets: UCI HAR, MotionSense, and
HAPT. The results show that our approach outperforms the existing methods and
sets new state-of-the-art results.

    

### [[2010.15421] Scalable Graph Neural Networks via Bidirectional Propagation](http://arxiv.org/abs/2010.15421)


  Graph Neural Networks (GNN) is an emerging field for learning on
non-Euclidean data. Recently, there has been increased interest in designing
GNN that scales to large graphs. Most existing methods use "graph sampling" or
"layer-wise sampling" techniques to reduce training time. However, these
methods still suffer from degrading performance and scalability problems when
applying to graphs with billions of edges. This paper presents GBP, a scalable
GNN that utilizes a localized bidirectional propagation process from both the
feature vectors and the training/testing nodes. Theoretical analysis shows that
GBP is the first method that achieves sub-linear time complexity for both the
precomputation and the training phases. An extensive empirical study
demonstrates that GBP achieves state-of-the-art performance with significantly
less training/testing time. Most notably, GBP can deliver superior performance
on a graph with over 60 million nodes and 1.8 billion edges in less than half
an hour on a single machine. The codes of GBP can be found at
this https URL .

    

### [[2011.06042] Robust multi-stage model-based design of optimal experiments for nonlinear estimation](http://arxiv.org/abs/2011.06042)


  We study approaches to robust model-based design of experiments in the
context of maximum-likelihood estimation. These approaches provide
robustification of model-based methodologies for the design of optimal
experiments by accounting for the effect of the parametric uncertainty. We
study the problem of robust optimal design of experiments in the framework of
nonlinear least-squares parameter estimation using linearized confidence
regions. We investigate several well-known robustification frameworks in this
respect and propose a novel methodology based on multi-stage robust
optimization. The proposed methodology aims at problems, where the experiments
are designed sequentially with a possibility of re-estimation in-between the
experiments. The multi-stage formalism aids in identifying experiments that are
better conducted in the early phase of experimentation, where parameter
knowledge is poor. We demonstrate the findings and effectiveness of the
proposed methodology using four case studies of varying complexity.

    

### [[2012.01744] Sample-Efficient L0-L2 Constrained Structure Learning of Sparse Ising Models](http://arxiv.org/abs/2012.01744)


  We consider the problem of learning the underlying graph of a sparse Ising
model with $p$ nodes from $n$ i.i.d. samples. The most recent and best
performing approaches combine an empirical loss (the logistic regression loss
or the interaction screening loss) with a regularizer (an L1 penalty or an L1
constraint). This results in a convex problem that can be solved separately for
each node of the graph. In this work, we leverage the cardinality constraint L0
norm, which is known to properly induce sparsity, and further combine it with
an L2 norm to better model the non-zero coefficients. We show that our proposed
estimators achieve an improved sample complexity, both (a) theoretically, by
reaching new state-of-the-art upper bounds for recovery guarantees, and (b)
empirically, by showing sharper phase transitions between poor and full
recovery for graph topologies studied in the literature, when compared to their
L1-based state-of-the-art methods.

    

### [[2012.04307] Cross-lingual Transfer of Abstractive Summarizer to Less-resource Language](http://arxiv.org/abs/2012.04307)


  Automatic text summarization extracts important information from texts and
presents the information in the form of a summary. Abstractive summarization
approaches progressed significantly by switching to deep neural networks, but
results are not yet satisfactory, especially for languages where large training
sets do not exist. In several natural language processing tasks, a
cross-lingual model transfer is successfully applied in less-resource
languages. For summarization, the cross-lingual model transfer was not
attempted due to a non-reusable decoder side of neural models that cannot
correct target language generation. In our work, we use a pre-trained English
summarization model based on deep neural networks and sequence-to-sequence
architecture to summarize Slovene news articles. We address the problem of
inadequate decoder by using an additional language model for the evaluation of
the generated text in target language. We test several cross-lingual
summarization models with different amounts of target data for fine-tuning. We
assess the models with automatic evaluation measures and conduct a small-scale
human evaluation. Automatic evaluation shows that the summaries of our best
cross-lingual model are useful and of quality similar to the model trained only
in the target language. Human evaluation shows that our best model generates
summaries with high accuracy and acceptable readability. However, similar to
other abstractive models, our models are not perfect and may occasionally
produce misleading or absurd content.

    

### [[2012.06274] A Topic Coverage Approach to Evaluation of Topic Models](http://arxiv.org/abs/2012.06274)


  Topic models are widely used unsupervised models capable of learning topics -
weighted lists of words and documents - from large collections of text
documents. When topic models are used for discovery of topics in text
collections, a question that arises naturally is how well the model-induced
topics correspond to topics of interest to the analyst. In this paper we
revisit and extend a so far neglected approach to topic model evaluation based
on measuring topic coverage - computationally matching model topics with a set
of reference topics that models are expected to uncover. The approach is well
suited for analyzing models' performance in topic discovery and for large-scale
analysis of both topic models and measures of model quality. We propose new
measures of coverage and evaluate, in a series of experiments, different types
of topic models on two distinct text domains for which interest for topic
discovery exists. The experiments include evaluation of model quality, analysis
of coverage of distinct topic categories, and the analysis of the relationship
between coverage and other methods of topic model evaluation. The paper
contributes a new supervised measure of coverage, and the first unsupervised
measure of coverage. The supervised measure achieves topic matching accuracy
close to human agreement. The unsupervised measure correlates highly with the
supervised one (Spearman's $\rho \geq 0.95$). Other contributions include
insights into both topic models and different methods of model evaluation, and
the datasets and code for facilitating future research on topic coverage.

    

### [[2101.00157] Active Learning Under Malicious Mislabeling and Poisoning Attacks](http://arxiv.org/abs/2101.00157)


  Deep neural networks usually require large labeled datasets for training to
achieve state-of-the-art performance in many tasks, such as image
classification and natural language processing. Although a lot of data is
created each day by active Internet users, most of these data are unlabeled and
are vulnerable to data poisoning attacks. In this paper, we develop an
efficient active learning method that requires fewer labeled instances and
incorporates the technique of adversarial retraining in which additional
labeled artificial data are generated without increasing the budget of the
labeling. The generated adversarial examples also provide a way to measure the
vulnerability of the model. To check the performance of the proposed method
under an adversarial setting, i.e., malicious mislabeling and data poisoning
attacks, we perform an extensive evaluation on the reduced CIFAR-10 dataset,
which contains only two classes: airplane and frog. Our experimental results
demonstrate that the proposed active learning method is efficient for defending
against malicious mislabeling and data poisoning attacks. Specifically, whereas
the baseline active learning method based on the random sampling strategy
performs poorly (about 50%) under a malicious mislabeling attack, the proposed
active learning method can achieve the desired accuracy of 89% using only
one-third of the dataset on average.

    

### [[2102.00751] Learning to Combat Noisy Labels via Classification Margins](http://arxiv.org/abs/2102.00751)


  A deep neural network trained on noisy labels is known to quickly lose its
power to discriminate clean instances from noisy ones. After the early learning
phase has ended, the network memorizes the noisy instances, which leads to a
significant degradation in its generalization performance. To resolve this
issue, we propose MARVEL (MARgins Via Early Learning), a new robust learning
method where the memorization of the noisy instances is curbed. We propose a
new test statistic that tracks the goodness of "fit" of every instance based on
the epoch-history of its classification margins. If its classification margin
is small in a sequence of consecutive learning epochs, that instance is
declared noisy and the network abandons learning on it. Consequently, the
network first flags a possibly noisy instance, and then waits to see if
learning on that instance can be improved and if not, the network learns with
confidence that this instance can be safely abandoned. We also propose MARVEL+,
where arduous instances can be upweighted, enabling the network to focus and
improve its learning on them and consequently its generalization. Experimental
results on benchmark datasets with synthetic label noise and real-world
datasets show that MARVEL outperforms other baselines consistently across
different noise levels, with a significantly larger margin under asymmetric
noise.

    

### [[2102.05334] Enhancing Real-World Adversarial Patches through 3D Modeling of Complex Target Scenes](http://arxiv.org/abs/2102.05334)


  Adversarial examples have proven to be a concerning threat to deep learning
models, particularly in the image domain. However, while many studies have
examined adversarial examples in the real world, most of them relied on 2D
photos of the attack scene. As a result, the attacks proposed may have limited
effectiveness when implemented in realistic environments with 3D objects or
varied conditions. There are few studies on adversarial learning that use 3D
objects, and in many cases, other researchers are unable to replicate the
real-world evaluation process. In this study, we present a framework that uses
3D modeling to craft adversarial patches for an existing real-world scene. Our
approach uses a 3D digital approximation of the scene as a simulation of the
real world. With the ability to add and manipulate any element in the digital
scene, our framework enables the attacker to improve the adversarial patch's
impact in real-world settings. We use the framework to create a patch for an
everyday scene and evaluate its performance using a novel evaluation process
that ensures that our results are reproducible in both the digital space and
the real world. Our evaluation results show that the framework can generate
adversarial patches that are robust to different settings in the real world.

    

### [[2102.06635] ReLU Neural Networks of Polynomial Size for Exact Maximum Flow Computation](http://arxiv.org/abs/2102.06635)


  Understanding the great empirical success of artificial neural networks (NNs)
from a theoretical point of view is currently one of the hottest research
topics in computer science. In order to study the expressive power of NNs with
rectified linear units, we propose to view them as a model of computation and
investigate the complexity of combinatorial optimization problems in that
model. Using a result from arithmetic circuit complexity, we show as a first
immediate result that the value of a minimum spanning tree in a graph with $n$
nodes can be computed by an NN of size $\mathcal{O}(n^3)$. Our primary result,
however, is that, given a directed graph with $n$ nodes and $m$ arcs, there
exists an NN of size $\mathcal{O}(m^2n^2)$ that computes a maximum flow from
any possible real-valued arc capacities as input. This settles the former open
questions whether such NNs with polynomial size exist. To prove our results, we
develop the pseudo-code language Max-Affine Arithmetic Programs (MAAPs) and
show equivalence between MAAPs and NNs concerning natural complexity measures.
We then design MAAPs that exactly solve the corresponding optimization problems
and translate to NNs of the claimed size.

    

### [[2102.10336] Physical Reasoning Using Dynamics-Aware Models](http://arxiv.org/abs/2102.10336)


  A common approach to solving physical reasoning tasks is to train a value
learner on example tasks. A limitation of such an approach is that it requires
learning about object dynamics solely from reward values assigned to the final
state of a rollout of the environment. This study aims to address this
limitation by augmenting the reward value with self-supervised signals about
object dynamics. Specifically, we train the model to characterize the
similarity of two environment rollouts, jointly with predicting the outcome of
the reasoning task. This similarity can be defined as a distance measure
between the trajectory of objects in the two rollouts, or learned directly from
pixels using a contrastive formulation. Empirically, we find that this approach
leads to substantial performance improvements on the PHYRE benchmark for
physical reasoning (Bakhtin et al., 2019), establishing a new state-of-the-art.

    

### [[2103.00073] CURE: Code-Aware Neural Machine Translation for Automatic Program Repair](http://arxiv.org/abs/2103.00073)


  Automatic program repair (APR) is crucial to improve software reliability.
Recently, neural machine translation (NMT) techniques have been used to fix
software bugs automatically. While promising, these approaches have two major
limitations. Their search space often does not contain the correct fix, and
their search strategy ignores software knowledge such as strict code syntax.
Due to these limitations, existing NMT-based techniques underperform the best
template-based approaches.
We propose CURE, a new NMT-based APR technique with three major novelties.
First, CURE pre-trains a programming language (PL) model on a large software
codebase to learn developer-like source code before the APR task. Second, CURE
designs a new code-aware search strategy that finds more correct fixes by
focusing on compilable patches and patches that are close in length to the
buggy code. Finally, CURE uses a subword tokenization technique to generate a
smaller search space that contains more correct fixes.
Our evaluation on two widely-used benchmarks shows that CURE correctly fixes
57 Defects4J bugs and 26 QuixBugs bugs, outperforming all existing APR
techniques on both benchmarks.

    

### [[2103.05872] Sampling methods for efficient training of graph convolutional networks: A survey](http://arxiv.org/abs/2103.05872)


  Graph Convolutional Networks (GCNs) have received significant attention from
various research fields due to the excellent performance in learning graph
representations. Although GCN performs well compared with other methods, it
still faces challenges. Training a GCN model for large-scale graphs in a
conventional way requires high computation and storage costs. Therefore,
motivated by an urgent need in terms of efficiency and scalability in training
GCN, sampling methods have been proposed and achieved a significant effect. In
this paper, we categorize sampling methods based on the sampling mechanisms and
provide a comprehensive survey of sampling methods for efficient training of
GCN. To highlight the characteristics and differences of sampling methods, we
present a detailed comparison within each category and further give an overall
comparative analysis for the sampling methods in all categories. Finally, we
discuss some challenges and future research directions of the sampling methods.

    

### [[2103.08403] Quantum federated learning through blind quantum computing](http://arxiv.org/abs/2103.08403)


  Private distributed learning studies the problem of how multiple distributed
entities collaboratively train a shared deep network with their private data
unrevealed. With the security provided by the protocols of blind quantum
computation, the cooperation between quantum physics and machine learning may
lead to unparalleled prospect for solving private distributed learning tasks.
In this paper, we introduce a quantum protocol for distributed learning that is
able to utilize the computational power of the remote quantum servers while
keeping the private data safe. For concreteness, we first introduce a protocol
for private single-party delegated training of variational quantum classifiers
based on blind quantum computing and then extend this protocol to multiparty
private distributed learning incorporated with differential privacy. We carry
out extensive numerical simulations with different real-life datasets and
encoding strategies to benchmark the effectiveness of our protocol. We find
that our protocol is robust to experimental imperfections and is secure under
the gradient attack after the incorporation of differential privacy. Our
results show the potential for handling computationally expensive distributed
learning tasks with privacy guarantees, thus providing a valuable guide for
exploring quantum advantages from the security perspective in the field of
machine learning with real-life applications.

    

### [[2104.02548] White Box Methods for Explanations of Convolutional Neural Networks in Image Classification Tasks](http://arxiv.org/abs/2104.02548)


  In recent years, deep learning has become prevalent to solve applications
from multiple domains. Convolutional Neural Networks (CNNs) particularly have
demonstrated state of the art performance for the task of image classification.
However, the decisions made by these networks are not transparent and cannot be
directly interpreted by a human. Several approaches have been proposed to
explain to understand the reasoning behind a prediction made by a network. In
this paper, we propose a topology of grouping these methods based on their
assumptions and implementations. We focus primarily on white box methods that
leverage the information of the internal architecture of a network to explain
its decision. Given the task of image classification and a trained CNN, this
work aims to provide a comprehensive and detailed overview of a set of methods
that can be used to create explanation maps for a particular image, that assign
an importance score to each pixel of the image based on its contribution to the
decision of the network. We also propose a further classification of the white
box methods based on their implementations to enable better comparisons and
help researchers find methods best suited for different scenarios.

    

### [[2104.03902] The Autodidactic Universe](http://arxiv.org/abs/2104.03902)


  We present an approach to cosmology in which the Universe learns its own
physical laws. It does so by exploring a landscape of possible laws, which we
express as a certain class of matrix models. We discover maps that put each of
these matrix models in correspondence with both a gauge/gravity theory and a
mathematical model of a learning machine, such as a deep recurrent, cyclic
neural network. This establishes a correspondence between each solution of the
physical theory and a run of a neural network. This correspondence is not an
equivalence, partly because gauge theories emerge from $N \rightarrow \infty $
limits of the matrix models, whereas the same limits of the neural networks
used here are not well-defined. We discuss in detail what it means to say that
learning takes place in autodidactic systems, where there is no supervision. We
propose that if the neural network model can be said to learn without
supervision, the same can be said for the corresponding physical theory. We
consider other protocols for autodidactic physical systems, such as
optimization of graph variety, subset-replication using self-attention and
look-ahead, geometrogenesis guided by reinforcement learning, structural
learning using renormalization group techniques, and extensions. These
protocols together provide a number of directions in which to explore the
origin of physical laws based on putting machine learning architectures in
correspondence with physical theories.

    

### [[2104.04466] Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking](http://arxiv.org/abs/2104.04466)


  Dialogue State Tracking is central to multi-domain task-oriented dialogue
systems, responsible for extracting information from user utterances. We
present a novel hybrid architecture that augments GPT-2 with representations
derived from Graph Attention Networks in such a way to allow causal, sequential
prediction of slot values. The model architecture captures inter-slot
relationships and dependencies across domains that otherwise can be lost in
sequential prediction. We report improvements in state tracking performance in
MultiWOZ 2.0 against a strong GPT-2 baseline and investigate a simplified
sparse training scenario in which DST models are trained only on session-level
annotations but evaluated at the turn level. We further report detailed
analyses to demonstrate the effectiveness of graph models in DST by showing
that the proposed graph modules capture inter-slot dependencies and improve the
predictions of values that are common to multiple domains.

    

### [[2104.09574] Probing Commonsense Explanation in Dialogue Response Generation](http://arxiv.org/abs/2104.09574)


  Humans use commonsense reasoning (CSR) implicitly to produce natural and
coherent responses in conversations. Aiming to close the gap between current
response generation (RG) models and human communication abilities, we want to
understand why RG models respond as they do by probing RG model's understanding
of commonsense reasoning that elicits proper responses. We formalize the
problem by framing commonsense as a latent variable in the RG task and using
explanations for responses as textual form of commonsense. We collect 6k
annotated explanations justifying responses from four dialogue datasets and ask
humans to verify them and propose two probing settings to evaluate RG models'
CSR capabilities. Probing results show that models fail to capture the logical
relations between commonsense explanations and responses and fine-tuning on
in-domain data and increasing model sizes do not lead to understanding of CSR
for RG. We hope our study motivates more research in making RG models emulate
the human reasoning process in pursuit of smooth human-AI communication.

    

### [[2104.11347] Restoring degraded speech via a modified diffusion model](http://arxiv.org/abs/2104.11347)


  There are many deterministic mathematical operations (e.g. compression,
clipping, downsampling) that degrade speech quality considerably. In this paper
we introduce a neural network architecture, based on a modification of the
DiffWave model, that aims to restore the original speech signal. DiffWave, a
recently published diffusion-based vocoder, has shown state-of-the-art
synthesized speech quality and relatively shorter waveform generation times,
with only a small set of parameters. We replace the mel-spectrum upsampler in
DiffWave with a deep CNN upsampler, which is trained to alter the degraded
speech mel-spectrum to match that of the original speech. The model is trained
using the original speech waveform, but conditioned on the degraded speech
mel-spectrum. Post-training, only the degraded mel-spectrum is used as input
and the model generates an estimate of the original speech. Our model results
in improved speech quality (original DiffWave model as baseline) on several
different experiments. These include improving the quality of speech degraded
by LPC-10 compression, AMR-NB compression, and signal clipping. Compared to the
original DiffWave architecture, our scheme achieves better performance on
several objective perceptual metrics and in subjective comparisons.
Improvements over baseline are further amplified in a out-of-corpus evaluation
setting.

    

### [[2105.00827] AMMU : A Survey of Transformer-based Biomedical Pretrained Language Models](http://arxiv.org/abs/2105.00827)


  Transformer-based pretrained language models (PLMs) have started a new era in
modern natural language processing (NLP). These models combine the power of
transformers, transfer learning, and self-supervised learning (SSL). Following
the success of these models in the general domain, the biomedical research
community has developed various in-domain PLMs starting from BioBERT to the
latest BioELECTRA and BioALBERT models. We strongly believe there is a need for
a survey paper that can provide a comprehensive survey of various
transformer-based biomedical pretrained language models (BPLMs). In this
survey, we start with a brief overview of foundational concepts like
self-supervised learning, embedding layer and transformer encoder layers. We
discuss core concepts of transformer-based PLMs like pretraining methods,
pretraining tasks, fine-tuning methods, and various embedding types specific to
biomedical domain. We introduce a taxonomy for transformer-based BPLMs and then
discuss all the models. We discuss various challenges and present possible
solutions. We conclude by highlighting some of the open issues which will drive
the research community to further improve transformer-based BPLMs.

    

### [[2105.01984] Software Engineering for AI-Based Systems: A Survey](http://arxiv.org/abs/2105.01984)


  AI-based systems are software systems with functionalities enabled by at
least one AI component (e.g., for image- and speech-recognition, and autonomous
driving). AI-based systems are becoming pervasive in society due to advances in
AI. However, there is limited synthesized knowledge on Software Engineering
(SE) approaches for building, operating, and maintaining AI-based systems. To
collect and analyze state-of-the-art knowledge about SE for AI-based systems,
we conducted a systematic mapping study. We considered 248 studies published
between January 2010 and March 2020. SE for AI-based systems is an emerging
research area, where more than 2/3 of the studies have been published since
2018. The most studied properties of AI-based systems are dependability and
safety. We identified multiple SE approaches for AI-based systems, which we
classified according to the SWEBOK areas. Studies related to software testing
and software quality are very prevalent, while areas like software maintenance
seem neglected. Data-related issues are the most recurrent challenges. Our
results are valuable for: researchers, to quickly understand the state of the
art and learn which topics need more research; practitioners, to learn about
the approaches and challenges that SE entails for AI-based systems; and,
educators, to bridge the gap among SE and AI in their curricula.

    

### [[2105.10759] Universal set of Observables for the Koopman Operator through Causal Embedding](http://arxiv.org/abs/2105.10759)


  Obtaining repeated measurements through observables of underlying physical
and natural systems to build dynamical models is engraved in modern science. A
key to the success of such methods is that the dynamics in the observed space
can often be described by a map that has much lower functional complexity than
the one that describes the unknown underlying system. Finding observables that
can empirically reduce the functional complexity of the map to be learned, and
at the same time, theoretically guarantee exact reconstruction in the new phase
space is an open challenge. Here, we determine a set of observables for the
Koopman operator of the inverse-limit system of a dynamical system that
guarantees exact reconstruction of the underlying dynamical system. Similar to
the delay coordinate maps being universal observables in Takens delay
embedding, the observables we determine are universal, and hence do not need to
be changed while the underlying system is changed. They are determined by a
class of driven systems that are comparable to those used in reservoir
computing, but which also can causally embed a dynamical system, a phenomenon
which we newly describe. Dynamics in the observed space is then shown to be
topologically conjugate to the underlying system. Deep learning methods can be
used to learn accurate equations from data as a consequence of the topological
conjugacy. Besides stability, amenability for hardware implementations, causal
embedding-based models provide long-term consistency even for systems that have
failed with previously reported data-driven or machine learning methods.

    

### [[2105.12818] Anomaly Detection in Predictive Maintenance: A New Evaluation Framework for Temporal Unsupervised Anomaly Detection Algorithms](http://arxiv.org/abs/2105.12818)


  The research in anomaly detection lacks a unified definition of what
represents an anomalous instance. Discrepancies in the nature itself of an
anomaly lead to multiple paradigms of algorithms design and experimentation.
Predictive maintenance is a special case, where the anomaly represents a
failure that must be prevented. Related time-series research as outlier and
novelty detection or time-series classification does not apply to the concept
of an anomaly in this field, because they are not single points which have not
been seen previously and may not be precisely annotated. Moreover, due to the
lack of annotated anomalous data, many benchmarks are adapted from supervised
scenarios.
To address these issues, we generalise the concept of positive and negative
instances to intervals to be able to evaluate unsupervised anomaly detection
algorithms. We also preserve the imbalance scheme for evaluation through the
proposal of the Preceding Window ROC, a generalisation for the calculation of
ROC curves for time-series scenarios. We also adapt the mechanism from a
established time-series anomaly detection benchmark to the proposed
generalisations to reward early detection. Therefore, the proposal represents a
flexible evaluation framework for the different scenarios. To show the
usefulness of this definition, we include a case study of Big Data algorithms
with a real-world time-series problem provided by the company ArcelorMittal,
and compare the proposal with an evaluation method.

    

### [[2106.02676] A novel multi-scale loss function for classification problems in machine learning](http://arxiv.org/abs/2106.02676)


  We introduce two-scale loss functions for use in various gradient descent
algorithms applied to classification problems via deep neural networks. This
new method is generic in the sense that it can be applied to a wide range of
machine learning architectures, from deep neural networks to support vector
machines for example. These two-scale loss functions allow to focus the
training onto objects in the training set which are not well classified. This
leads to an increase in several measures of performance for
appropriately-defined two-scale loss functions with respect to the more
classical cross-entropy when tested on traditional deep neural networks on the
MNIST, CIFAR10, and CIFAR100 data-sets.

    

### [[2106.03134] Semi-Riemannian Graph Convolutional Networks](http://arxiv.org/abs/2106.03134)


  Graph Convolutional Networks (GCNs) are typically studied through the lens of
Euclidean geometry. Non-Euclidean Riemannian manifolds provide specific
inductive biases for embedding hierarchical or spherical data, but cannot align
well with data of mixed topologies. We consider a larger class of
semi-Riemannian manifolds with indefinite metric that generalize hyperboloid
and sphere as well as their submanifolds. We develop new geodesic tools that
allow for extending neural network operations into geodesically disconnected
semi-Riemannian manifolds. As a consequence, we derive a principled
Semi-Riemannian GCN that first models data in semi-Riemannian manifolds of
constant nonzero curvature in the context of graph neural networks. Our method
provides a geometric inductive bias that is sufficiently flexible to model
mixed heterogeneous topologies like hierarchical graphs with cycles. Empirical
results demonstrate that our method outperforms Riemannian counterparts when
embedding graphs of complex topologies.

    

### [[2106.07592] No more glowing in the dark: How deep learning improves exposure date estimation in thermoluminescence dosimetry](http://arxiv.org/abs/2106.07592)


  The time- or temperature-resolved detector signal from a thermoluminescence
dosimeter can reveal additional information about circumstances of an exposure
to ionizing irradiation. We present studies using deep neural networks to
estimate the date of a single irradiation with 12 mSv within a monitoring
interval of 42 days from glow curves of novel TL-DOS personal dosimeters
developed by the Materialprüfungsamt NRW in cooperation with TU Dortmund
University. Using a deep convolutional network, the irradiation date can be
predicted from raw time-resolved glow curve data with an uncertainty of roughly
1-2 days on a 68% confidence level without the need for a prior transformation
into temperature space and a subsequent glow curve deconvolution. This
corresponds to a significant improvement in prediction accuracy compared to a
prior publication, which yielded a prediction uncertainty of 2-4 days using
features obtained from a glow curve deconvolution as input to a neural network.

    

### [[2106.07976] Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection](http://arxiv.org/abs/2106.07976)


  Federated learning can be a promising solution for enabling IoT cybersecurity
(i.e., anomaly detection in the IoT environment) while preserving data privacy
and mitigating the high communication/storage overhead (e.g., high-frequency
data from time-series sensors) of centralized over-the-cloud approaches. In
this paper, to further push forward this direction with a comprehensive study
in both algorithm and system design, we build FedIoT platform that contains
FedDetect algorithm for on-device anomaly data detection and a system design
for realistic evaluation of federated learning on IoT devices. Furthermore, the
proposed FedDetect learning framework improves the performance by utilizing a
local adaptive optimizer (e.g., Adam) and a cross-round learning rate
scheduler. In a network of realistic IoT devices (Raspberry PI), we evaluate
FedIoT platform and FedDetect algorithm in both model and system performance.
Our results demonstrate the efficacy of federated learning in detecting a wider
range of attack types occurred at multiple devices. The system efficiency
analysis indicates that both end-to-end training time and memory cost are
affordable and promising for resource-constrained IoT devices. The source code
is publicly available at this https URL


### [[2107.05407] PonderNet: Learning to Ponder](http://arxiv.org/abs/2107.05407)


  In standard neural networks the amount of computation used grows with the
size of the inputs, but not with the complexity of the problem being learnt. To
overcome this limitation we introduce PonderNet, a new algorithm that learns to
adapt the amount of computation based on the complexity of the problem at hand.
PonderNet learns end-to-end the number of computational steps to achieve an
effective compromise between training prediction accuracy, computational cost
and generalization. On a complex synthetic problem, PonderNet dramatically
improves performance over previous adaptive computation methods and
additionally succeeds at extrapolation tests where traditional neural networks
fail. Also, our method matched the current state of the art results on a real
world question and answering dataset, but using less compute. Finally,
PonderNet reached state of the art results on a complex task designed to test
the reasoning capabilities of neural networks.1

    

### [[2107.08987] Analysis of training and seed bias in small molecules generated with a conditional graph-based variational autoencoder -- Insights for practical AI-driven molecule generation](http://arxiv.org/abs/2107.08987)


  The application of deep learning to generative molecule design has shown
early promise for accelerating lead series development. However, questions
remain concerning how factors like training, dataset, and seed bias impact the
technology's utility to medicine and computational chemists. In this work, we
analyze the impact of seed and training bias on the output of an
activity-conditioned graph-based variational autoencoder (VAE). Leveraging a
massive, labeled dataset corresponding to the dopamine D2 receptor, our
graph-based generative model is shown to excel in producing desired conditioned
activities and favorable unconditioned physical properties in generated
molecules. We implement an activity swapping method that allows for the
activation, deactivation, or retention of activity of molecular seeds, and we
apply independent deep learning classifiers to verify the generative results.
Overall, we uncover relationships between noise, molecular seeds, and training
set selection across a range of latent-space sampling procedures, providing
important insights for practical AI-driven molecule generation.

    

### [[2109.00665] Agon: A Scalable Competitive Scheduler for Large Heterogeneous Systems](http://arxiv.org/abs/2109.00665)


  This work proposes a competitive scheduling approach, designed to scale to
large heterogeneous multicore systems. This scheduler overcomes the challenges
of (1) the high computation overhead of near-optimal schedulers, and (2) the
error introduced by inaccurate performance predictions. This paper presents
Agon, a neural network-based classifier that selects from a range of
schedulers, from simple to very accurate, and learns which scheduler provides
the right balance of accuracy and overhead for each scheduling interval. Agon
also employs a de-noising frontend allowing the individual schedulers to be
tolerant towards noise in performance predictions, producing better overall
schedules. By avoiding expensive scheduling overheads, Agon improves average
system performance by 6\% on average, approaching the performance of an
oracular scheduler (99.1% of oracle performance).

    

### [[2109.00958] A Novel Compaction Approach for SBST Test Programs](http://arxiv.org/abs/2109.00958)


  In-field test of processor-based devices is a must when considering
safety-critical systems (e.g., in robotics, aerospace, and automotive
applications). During in-field testing, different solutions can be adopted,
depending on the specific constraints of each scenario. In the last years,
Self-Test Libraries (STLs) developed by IP or semiconductor companies became
widely adopted. Given the strict constraints of in-field test, the size and
time duration of a STL is a crucial parameter. This work introduces a novel
approach to compress functional test programs belonging to an STL. The proposed
approach is based on analyzing (via logic simulation) the interaction between
the micro-architectural operation performed by each instruction and its
capacity to propagate fault effects on any observable output, reducing the
required fault simulations to only one. The proposed compaction strategy was
validated by resorting to a RISC-V processor and several test programs stemming
from diverse generation strategies. Results showed that the proposed compaction
approach can reduce the length of test programs by up to 93.9% and their
duration by up to 95%, with minimal effect on fault coverage.

    

### [[2109.01126] An Electro-Photonic System for Accelerating Deep Neural Networks](http://arxiv.org/abs/2109.01126)


  The number of parameters in deep neural networks (DNNs) is scaling at about
5$\times$ the rate of Moore's Law. To sustain the pace of growth of the DNNs,
new technologies and computing architectures are needed. Photonic computing
systems are promising avenues, since they can perform the dominant general
matrix-matrix multiplication (GEMM) operations in DNNs at a higher throughput
than their electrical counterpart. However, purely photonic systems face
several challenges including a lack of photonic memory, the need for conversion
circuits, and the accumulation of noise. In this paper, we propose a hybrid
electro-photonic system realizing the best of both worlds to accelerate DNNs.
In contrast to prior work in photonic and electronic accelerators, we adopt a
system-level perspective. Our electro-photonic system includes an electronic
host processor and DRAM, and a custom electro-photonic hardware accelerator
called ADEPT. The fused hardware accelerator leverages a photonic computing
unit for performing highly-efficient GEMM operations and a digital electronic
ASIC for storage and for performing non-GEMM operations. We also identify
architectural optimization opportunities for improving the overall ADEPT's
efficiency. We evaluate ADEPT using three state-of-the-art neural
networks-ResNet-50, BERT-large, and RNN-T-to show its general applicability in
accelerating today's DNNs. A head-to-head comparison of ADEPT with systolic
array architectures shows that ADEPT can provide, on average, 7.19$\times$
higher inference throughput per watt.

    

### [[2012.05079] Page Tables: Keeping them Flat and Hot (Cached)](http://arxiv.org/abs/2012.05079)


  As memory capacity has outstripped TLB coverage, large data applications
suffer from frequent page table walks. We investigate two complementary
techniques for addressing this cost: reducing the number of accesses required
and reducing the latency of each access. The first approach is accomplished by
opportunistically "flattening" the page table: merging two levels of
traditional 4KB page table nodes into a single 2MB node, thereby reducing the
table's depth and the number of indirections required to search it. The second
is accomplished by biasing the cache replacement algorithm to keep page table
entries during periods of high TLB miss rates, as these periods also see high
data miss rates and are therefore more likely to benefit from having the
smaller page table in the cache than to suffer from increased data cache
misses.
We evaluate these approaches for both native and virtualized systems and
across a range of realistic memory fragmentation scenarios, describe the
limited changes needed in our kernel implementation and hardware design,
identify and address challenges related to self-referencing page tables and
kernel memory allocation, and compare results across server and mobile systems
using both academic and industrial simulators for robustness.
We find that flattening does reduce the number of accesses required on a page
walk (to 1.0), but its performance impact (+2.3%) is small due to Page Walker
Caches (already 1.5 accesses). Prioritizing caching has a larger effect
(+6.8%), and the combination improves performance by +9.2%. Flattening is more
effective on virtualized systems (4.4 to 2.8 accesses, +7.1% performance), due
to 2D page walks. By combining the two techniques we demonstrate a
state-of-the-art +14.0% performance gain and -8.7% dynamic cache energy and
-4.7% dynamic DRAM energy for virtualized execution with very simple hardware
and software changes.

    

### [[2109.00657] Multi-Queues Can Be State-of-the-Art Priority Schedulers](http://arxiv.org/abs/2109.00657)


  Designing and implementing efficient parallel priority schedulers is an
active research area. An intriguing proposed design is the Multi-Queue: given
$n$ threads and $m\ge n$ distinct priority queues, task insertions are
performed uniformly at random, while, to delete, a thread picks two queues
uniformly at random, and removes the observed task of higher priority. This
approach scales well, and has probabilistic rank guarantees: roughly, the rank
of each task removed, relative to remaining tasks in all other queues, is
$O(m)$ in expectation. Yet, the performance of this pattern is below that of
well-engineered schedulers, which eschew theoretical guarantees for practical
efficiency.
We investigate whether it is possible to design and implement a
Multi-Queue-based task scheduler that is both highly efficient and has
analytical guarantees. We propose a new variant called the Stealing Multi-Queue
(SMQ), a cache-efficient variant of the Multi-Queue, which leverages both queue
affinity -- each thread has a local queue, from which tasks are usually
removed; but, with some probability, threads also attempt to steal
higher-priority tasks from the other queues -- and task batching, that is, the
processing of several tasks in a single insert / delete step. These ideas are
well-known for task scheduling without priorities; our theoretical contribution
is showing that, despite relaxations, this design can still provide rank
guarantees, which in turn implies bounds on total work performed. We provide a
general SMQ implementation that can surpass state-of-the-art schedulers such as
Galois and PMOD in terms of performance on popular graph-processing benchmarks.
Notably, the performance improvement comes mainly from the superior rank
guarantees provided by our scheduler, confirming that analytically-reasoned
approaches can still provide performance improvements for priority task
scheduling.

    

### [[2109.00857] GPU-accelerated Optimal Path Planning in Stochastic Dynamic Environments](http://arxiv.org/abs/2109.00857)


  Autonomous marine vehicles play an essential role in many ocean science and
engineering applications. Planning time and energy optimal paths for these
vehicles to navigate in stochastic dynamic ocean environments is essential to
reduce operational costs. In some missions, they must also harvest solar, wind,
or wave energy (modeled as a stochastic scalar field) and move in optimal paths
that minimize net energy consumption. Markov Decision Processes (MDPs) provide
a natural framework for sequential decision-making for robotic agents in such
environments. However, building a realistic model and solving the modeled MDP
becomes computationally expensive in large-scale real-time applications,
warranting the need for parallel algorithms and efficient implementation. In
the present work, we introduce an efficient end-to-end GPU-accelerated
algorithm that (i) builds the MDP model (computing transition probabilities and
expected one-step rewards); and (ii) solves the MDP to compute an optimal
policy. We develop methodical and algorithmic solutions to overcome the limited
global memory of GPUs by (i) using a dynamic reduced-order representation of
the ocean flows, (ii) leveraging the sparse nature of the state transition
probability matrix, (iii) introducing a neighbouring sub-grid concept and (iv)
proving that it is sufficient to use only the stochastic scalar field's mean to
compute the expected one-step rewards for missions involving energy harvesting
from the environment; thereby saving memory and reducing the computational
effort. We demonstrate the algorithm on a simulated stochastic dynamic
environment and highlight that it builds the MDP model and computes the optimal
policy 600-1000x faster than conventional CPU implementations, making it
suitable for real-time use.

    

### [[2109.01047] Crypto Currency Regulation and Law Enforcement Perspectives](http://arxiv.org/abs/2109.01047)


  This paper provides an overview of how crypto currency and blockchain
engineering interacts with the law enforcement. We point out that a large
proportion of crypto users are amateur investors and the dominant and the
largest segment in crypto crime are simply investment scams (!). We look at
various questions of criminal use and misuse of technology, especially in the
areas of money laundering or cashing out the profits originating from illicit
activities. The aim of the paper is to raise a set of concerns arising in the
criminal justice and policing circles, based on the interviews with law
enforcement practitioners, and to see how cryptos could be reconciled with
public security and safety. We propose a simplified classification of crimes
related to crypto currency. We study the development of blockchains in a
broader context of applied cryptography and payment technology. Ransomware is a
big threat but we also need protection against corporate misconduct or
negligence, with untested financial services breaching customer trust or
government regulations. Not paying taxes is illegal, but there is more at
stake: exposing crypto holders to losing all their savings in scams or thefts.
Interestingly, privacy helps to defend on multiple fronts: against social
engineering, targeted crime, scams, and also against cybersecurity thefts and
hacks.

    

### [[2109.01102] DAG-Oriented Protocols PHANTOM and GHOSTDAG under Incentive Attack via Transaction Selection Strategy](http://arxiv.org/abs/2109.01102)


  In response to the bottleneck of processing throughput inherent to single
chain PoW blockchains, several proposals have substituted a single chain for
Directed Acyclic Graphs (DAGs). In this work, we investigate two notable
DAG-oriented designs. We focus on PHANTOM (and its optimization GHOSTDAG),
which proposes a custom transaction selection strategy that enables to increase
the throughput of the network. However, the related work lacks a thorough
investigation of corner cases that deviate from the protocol in terms of
transaction selection strategy. Therefore, we build a custom simulator that
extends open source simulation tools to support multiple chains and enables us
to investigate such corner cases. Our experiments show that malicious actors
who diverge from the proposed transaction selection strategy make more profit
as compared to honest miners. Moreover, they have a detrimental effect on the
processing throughput of the PHANTOM (and GHOSTDAG) due to same transactions
being included in more than one block of different chains. Finally, we show
that multiple miners not following the transaction selection strategy are
incentivized to create a shared mining pool instead of mining independently,
which has a negative impact on decentralization.

    

### [[2109.00591] Fight Fire with Fire: Fine-tuning Hate Detectors using Large Samples of Generated Hate Speech](http://arxiv.org/abs/2109.00591)


  Automatic hate speech detection is hampered by the scarcity of labeled
datasetd, leading to poor generalization. We employ pretrained language models
(LMs) to alleviate this data bottleneck. We utilize the GPT LM for generating
large amounts of synthetic hate speech sequences from available labeled
examples, and leverage the generated data in fine-tuning large pretrained LMs
on hate detection. An empirical study using the models of BERT, RoBERTa and
ALBERT, shows that this approach improves generalization significantly and
consistently within and across data distributions. In fact, we find that
generating relevant labeled hate speech sequences is preferable to using
out-of-domain, and sometimes also within-domain, human-labeled examples.

    

### [[2109.00663] Controllable deep melody generation via hierarchical music structure representation](http://arxiv.org/abs/2109.00663)


  Recent advances in deep learning have expanded possibilities to generate
music, but generating a customizable full piece of music with consistent
long-term structure remains a challenge. This paper introduces MusicFrameworks,
a hierarchical music structure representation and a multi-step generative
process to create a full-length melody guided by long-term repetitive
structure, chord, melodic contour, and rhythm constraints. We first organize
the full melody with section and phrase-level structure. To generate melody in
each phrase, we generate rhythm and basic melody using two separate
transformer-based networks, and then generate the melody conditioned on the
basic melody, rhythm and chords in an auto-regressive manner. By factoring
music generation into sub-problems, our approach allows simpler models and
requires less data. To customize or add variety, one can alter chords, basic
melody, and rhythm structure in the music frameworks, letting our networks
generate the melody accordingly. Additionally, we introduce new features to
encode musical positional information, rhythm patterns, and melodic contours
based on musical domain knowledge. A listening test reveals that melodies
generated by our method are rated as good as or better than human-composed
music in the POP909 dataset about half the time.

    

### [[2109.00693] AnANet: Modeling Association and Alignment for Cross-modal Correlation Classification](http://arxiv.org/abs/2109.00693)


  The explosive increase of multimodal data makes a great demand in many
cross-modal applications that follow the strict prior related assumption. Thus
researchers study the definition of cross-modal correlation category and
construct various classification systems and predictive models. However, those
systems pay more attention to the fine-grained relevant types of cross-modal
correlation, ignoring lots of implicit relevant data which are often divided
into irrelevant types. What's worse is that none of previous predictive models
manifest the essence of cross-modal correlation according to their definition
at the modeling stage. In this paper, we present a comprehensive analysis of
the image-text correlation and redefine a new classification system based on
implicit association and explicit alignment. To predict the type of image-text
correlation, we propose the Association and Alignment Network according to our
proposed definition (namely AnANet) which implicitly represents the global
discrepancy and commonality between image and text and explicitly captures the
cross-modal local relevance. The experimental results on our constructed new
image-text correlation dataset show the effectiveness of our model.

    

### [[2109.00729] ConQX: Semantic Expansion of Spoken Queries for Intent Detection based on Conditioned Text Generation](http://arxiv.org/abs/2109.00729)


  Intent detection of spoken queries is a challenging task due to their noisy
structure and short length. To provide additional information regarding the
query and enhance the performance of intent detection, we propose a method for
semantic expansion of spoken queries, called ConQX, which utilizes the text
generation ability of an auto-regressive language model, GPT-2. To avoid
off-topic text generation, we condition the input query to a structured context
with prompt mining. We then apply zero-shot, one-shot, and few-shot learning.
We lastly use the expanded queries to fine-tune BERT and RoBERTa for intent
detection. The experimental results show that the performance of intent
detection can be improved by our semantic expansion method.

    

### [[2109.00831] Knot invariants and their relations: a topological perspective](http://arxiv.org/abs/2109.00831)


  This work brings methods from topological data analysis to knot theory and
develops new data analysis tools inspired by this application. We explore a
vast collection of knot invariants and relations between then using Mapper and
Ball Mapper algorithms. In particular, we develop versions of the Ball Mapper
algorithm that incorporate symmetries and other relations within the data, and
provide ways to compare data arising from different descriptors, such as knot
invariants. Additionally, we extend the Mapper construction to the case where
the range of the lens function is high dimensional rather than a 1-dimensional
space, that also provides ways of visualizing functions between
high-dimensional spaces. We illustrate the use of these techniques on knot
theory data and draw attention to potential implications of our findings in
knot theory.

    

### [[2109.00838] An Automated Framework for Supporting Data-Governance Rule Compliance in Decentralized MIMO Contexts](http://arxiv.org/abs/2109.00838)


  We propose Dr.Aid, a logic-based AI framework for automated compliance
checking of data governance rules over data-flow graphs. The rules are modelled
using a formal language based on situation calculus and are suitable for
decentralized contexts with multi-input-multi-output (MIMO) processes. Dr.Aid
models data rules and flow rules and checks compliance by reasoning about the
propagation, combination, modification and application of data rules over the
data flow graphs. Our approach is driven and evaluated by real-world datasets
using provenance graphs from data-intensive research.

    

### [[2109.00840] Imposing Relation Structure in Language-Model EmbeddingsUsing Contrastive Learning](http://arxiv.org/abs/2109.00840)


  Though language model text embeddings have revolutionized NLP research, their
ability to capture high-level semantic information, such as relations between
entities in text, is limited. In this paper, we propose a novel contrastive
learning framework that trains sentence embeddings to encode the relations in a
graph structure. Given a sentence (unstructured text) and its graph, we use
contrastive learning to impose relation-related structure on the token-level
representations of the sentence obtained with a CharacterBERT (El Boukkouri et
al.,2020) model. The resulting relation-aware sentence embeddings achieve
state-of-the-art results on the relation extraction task using only a simple
KNN classifier, thereby demonstrating the success of the proposed method.
Additional visualization by a tSNE analysis shows the effectiveness of the
learned representation space compared to baselines. Furthermore, we show that
we can learn a different space for named entity recognition, again using a
contrastive learning objective, and demonstrate how to successfully combine
both representation spaces in an entity-relation task.

    

### [[2109.00862] VORRT-COLREGs: A Hybrid Velocity Obstacles and RRT Based COLREGs-Compliant Path Planner for Autonomous Surface Vessels](http://arxiv.org/abs/2109.00862)


  This paper presents VORRT-COLREGs, a hybrid technique that combines velocity
obstacles (VO) and rapidly-exploring random trees (RRT) to generate safe
trajectories for autonomous surface vessels (ASVs) while following nautical
rules of the road. RRT generates a set of way points and the velocity obstacles
method ensures safe travel between way points. We also ensure that the actions
of ASVs do not violate maritime collision guidelines. Earlier work has used RRT
and VO separately to generate paths for ASVs. However, RRT does not handle
highly dynamic situations well and and VO seems most suitable as a local path
planner. Combining both approaches, VORRT-COLREGs is a global path planner that
uses a joint forward simulation to ensure that generated paths remain valid and
collision free as the situation changes. Experiments were conducted in
different types of collision scenarios and with different numbers of ASVs.
Results show that VORRT-COLREGS generated collision regulations (COLREGs)
complaint paths in open ocean scenarios. Furthermore, VORRT-COLREGS
successfully generated compliant paths within traffic separation schemes. These
results show the applicability of our technique for generating paths for ASVs
in different collision scenarios. To the best of our knowledge, this is the
first work that combines velocity obstacles and RRT to produce safe and COLREGs
complaint path for ASVs.

    

### [[2109.00866] Habitual and Reflective Control in Hierarchical Predictive Coding](http://arxiv.org/abs/2109.00866)


  In cognitive science, behaviour is often separated into two types. Reflexive
control is habitual and immediate, whereas reflective is deliberative and time
consuming. We examine the argument that Hierarchical Predictive Coding (HPC)
can explain both types of behaviour as a continuum operating across a
multi-layered network, removing the need for separate circuits in the brain. On
this view, "fast" actions may be triggered using only the lower layers of the
HPC schema, whereas more deliberative actions need higher layers. We
demonstrate that HPC can distribute learning throughout its hierarchy, with
higher layers called into use only as required.

    

### [[2109.00895] Knowledge Perceived Multi-modal Pretraining in E-commerce](http://arxiv.org/abs/2109.00895)


  In this paper, we address multi-modal pretraining of product data in the
field of E-commerce. Current multi-modal pretraining methods proposed for image
and text modalities lack robustness in the face of modality-missing and
modality-noise, which are two pervasive problems of multi-modal product data in
real E-commerce scenarios. To this end, we propose a novel method, K3M, which
introduces knowledge modality in multi-modal pretraining to correct the noise
and supplement the missing of image and text modalities. The modal-encoding
layer extracts the features of each modality. The modal-interaction layer is
capable of effectively modeling the interaction of multiple modalities, where
an initial-interactive feature fusion model is designed to maintain the
independence of image modality and text modality, and a structure aggregation
module is designed to fuse the information of image, text, and knowledge
modalities. We pretrain K3M with three pretraining tasks, including masked
object modeling (MOM), masked language modeling (MLM), and link prediction
modeling (LPM). Experimental results on a real-world E-commerce dataset and a
series of product-based downstream tasks demonstrate that K3M achieves
significant improvements in performances than the baseline and state-of-the-art
methods when modality-noise or modality-missing exists.

    

### [[2109.00903] Effect of the output activation function on the probabilities and errors in medical image segmentation](http://arxiv.org/abs/2109.00903)


  The sigmoid activation is the standard output activation function in binary
classification and segmentation with neural networks. Still, there exist a
variety of other potential output activation functions, which may lead to
improved results in medical image segmentation. In this work, we consider how
the asymptotic behavior of different output activation and loss functions
affects the prediction probabilities and the corresponding segmentation errors.
For cross entropy, we show that a faster rate of change of the activation
function correlates with better predictions, while a slower rate of change can
improve the calibration of probabilities. For dice loss, we found that the
arctangent activation function is superior to the sigmoid function.
Furthermore, we provide a test space for arbitrary output activation functions
in the area of medical image segmentation. We tested seven activation functions
in combination with three loss functions on four different medical image
segmentation tasks to provide a classification of which function is best suited
in this application scenario.

    

### [[2109.00918] Multi-task learning from fixed-wing UAV images for 2D/3D city modeling](http://arxiv.org/abs/2109.00918)


  Single-task learning in artificial neural networks will be able to learn the
model very well, and the benefits brought by transferring knowledge thus become
limited. In this regard, when the number of tasks increases (e.g., semantic
segmentation, panoptic segmentation, monocular depth estimation, and 3D point
cloud), duplicate information may exist across tasks, and the improvement
becomes less significant. Multi-task learning has emerged as a solution to
knowledge-transfer issues and is an approach to scene understanding which
involves multiple related tasks each with potentially limited training data.
Multi-task learning improves generalization by leveraging the domain-specific
information contained in the training data of related tasks. In urban
management applications such as infrastructure development, traffic monitoring,
smart 3D cities, and change detection, automated multi-task data analysis for
scene understanding based on the semantic, instance, and panoptic annotation,
as well as monocular depth estimation, is required to generate precise urban
models. In this study, a common framework for the performance assessment of
multi-task learning methods from fixed-wing UAV images for 2D/3D city modeling
is presented.

    

### [[2109.00927] Autonomous Curiosity for Real-Time Training Onboard Robotic Agents](http://arxiv.org/abs/2109.00927)


  Learning requires both study and curiosity. A good learner is not only good
at extracting information from the data given to it, but also skilled at
finding the right new information to learn from. This is especially true when a
human operator is required to provide the ground truth - such a source should
only be queried sparingly. In this work, we address the problem of curiosity as
it relates to online, real-time, human-in-the-loop training of an object
detection algorithm onboard a robotic platform, one where motion produces new
views of the subject. We propose a deep reinforcement learning approach that
decides when to ask the human user for ground truth, and when to move. Through
a series of experiments, we demonstrate that our agent learns a movement and
request policy that is at least 3x more effective at using human user
interactions to train an object detector than untrained approaches, and is
generalizable to a variety of subjects and environments.

    

### [[2109.00953] TrouSPI-Net: Spatio-temporal attention on parallel atrous convolutions and U-GRUs for skeletal pedestrian crossing prediction](http://arxiv.org/abs/2109.00953)


  Understanding the behaviors and intentions of pedestrians is still one of the
main challenges for vehicle autonomy, as accurate predictions of their
intentions can guarantee their safety and driving comfort of vehicles. In this
paper, we address pedestrian crossing prediction in urban traffic environments
by linking the dynamics of a pedestrian's skeleton to a binary crossing
intention. We introduce TrouSPI-Net: a context-free, lightweight, multi-branch
predictor. TrouSPI-Net extracts spatio-temporal features for different time
resolutions by encoding pseudo-images sequences of skeletal joints' positions
and processes them with parallel attention modules and atrous convolutions. The
proposed approach is then enhanced by processing features such as relative
distances of skeletal joints, bounding box positions, or ego-vehicle speed with
U-GRUs. Using the newly proposed evaluation procedures for two large public
naturalistic data sets for studying pedestrian behavior in traffic: JAAD and
PIE, we evaluate TrouSPI-Net and analyze its performance. Experimental results
show that TrouSPI-Net achieved 0.76 F1 score on JAAD and 0.80 F1 score on PIE,
therefore outperforming current state-of-the-art while being lightweight and
context-free.

    

### [[2109.00960] Infrared Image Super-Resolution via Heterogeneous Convolutional WGAN](http://arxiv.org/abs/2109.00960)


  Image super-resolution is important in many fields, such as surveillance and
remote sensing. However, infrared (IR) images normally have low resolution
since the optical equipment is relatively expensive. Recently, deep learning
methods have dominated image super-resolution and achieved remarkable
performance on visible images; however, IR images have received less attention.
IR images have fewer patterns, and hence, it is difficult for deep neural
networks (DNNs) to learn diverse features from IR images. In this paper, we
present a framework that employs heterogeneous convolution and adversarial
training, namely, heterogeneous kernel-based super-resolution Wasserstein GAN
(HetSRWGAN), for IR image super-resolution. The HetSRWGAN algorithm is a
lightweight GAN architecture that applies a plug-and-play heterogeneous
kernel-based residual block. Moreover, a novel loss function that employs image
gradients is adopted, which can be applied to an arbitrary model. The proposed
HetSRWGAN achieves consistently better performance in both qualitative and
quantitative evaluations. According to the experimental results, the whole
training process is more stable.

    

### [[2109.01013] On Dedicated CDCL Strategies for PB Solvers](http://arxiv.org/abs/2109.01013)


  Current implementations of pseudo-Boolean (PB) solvers working on native PB
constraints are based on the CDCL architecture which empowers highly efficient
modern SAT solvers. In particular, such PB solvers not only implement a
(cutting-planes-based) conflict analysis procedure, but also complementary
strategies for components that are crucial for the efficiency of CDCL, namely
branching heuristics, learned constraint deletion and restarts. However, these
strategies are mostly reused by PB solvers without considering the particular
form of the PB constraints they deal with. In this paper, we present and
evaluate different ways of adapting CDCL strategies to take the specificities
of PB constraints into account while preserving the behavior they have in the
clausal setting. We implemented these strategies in two different solvers,
namely Sat4j (for which we consider three configurations) and RoundingSat. Our
experiments show that these dedicated strategies allow to improve, sometimes
significantly, the performance of these solvers, both on decision and
optimization problems.

    

### [[2109.01071] Towards disease-aware image editing of chest X-rays](http://arxiv.org/abs/2109.01071)


  Disease-aware image editing by means of generative adversarial networks
(GANs) constitutes a promising avenue for advancing the use of AI in the
healthcare sector. Here, we present a proof of concept of this idea. While
GAN-based techniques have been successful in generating and manipulating
natural images, their application to the medical domain, however, is still in
its infancy. Working with the CheXpert data set, we show that StyleGAN can be
trained to generate realistic chest X-rays. Inspired by the Cyclic Reverse
Generator (CRG) framework, we train an encoder that allows for faithfully
inverting the generator on synthetic X-rays and provides organ-level
reconstructions of real ones. Employing a guided manipulation of latent codes,
we confer the medical condition of cardiomegaly (increased heart size) onto
real X-rays from healthy patients. This work was presented in the Medical
Imaging meets Neurips Workshop 2020, which was held as part of the 34th
Conference on Neural Information Processing Systems (NeurIPS 2020) in
Vancouver, Canada

    

### [[2109.01121] A Reasoning Engine for the Gamification of Loop-Invariant Discovery](http://arxiv.org/abs/2109.01121)


  We describe the design and implementation of a reasoning engine that
facilitates the gamification of loop-invariant discovery. Our reasoning engine
enables students, computational agents and regular software engineers with no
formal methods expertise to collaboratively prove interesting theorems about
simple programs using browser-based, online games. Within an hour, players are
able to specify and verify properties of programs that are beyond the
capabilities of fully-automated tools. The hour limit includes the time for
setting up the system, completing a short tutorial explaining game play and
reasoning about simple imperative programs. Players are never required to
understand formal proofs; they only provide insights by proposing invariants.
The reasoning engine is responsible for managing and evaluating the proposed
invariants, as well as generating actionable feedback.

    

### [[2006.13615] Explainable robotic systems: Understanding goal-driven actions in a reinforcement learning scenario](http://arxiv.org/abs/2006.13615)


  Robotic systems are more present in our society everyday. In human-robot
environments, it is crucial that end-users may correctly understand their
robotic team-partners, in order to collaboratively complete a task. To increase
action understanding, users demand more explainability about the decisions by
the robot in particular situations. Recently, explainable robotic systems have
emerged as an alternative focused not only on completing a task satisfactorily,
but also on justifying, in a human-like manner, the reasons that lead to making
a decision. In reinforcement learning scenarios, a great effort has been
focused on providing explanations using data-driven approaches, particularly
from the visual input modality in deep learning-based systems. In this work, we
focus rather on the decision-making process of reinforcement learning agents
performing a task in a robotic scenario. Experimental results are obtained
using 3 different set-ups, namely, a deterministic navigation task, a
stochastic navigation task, and a continuous visual-based sorting object task.
As a way to explain the goal-driven robot's actions, we use the probability of
success computed by three different proposed approaches: memory-based,
learning-based, and introspection-based. The difference between these
approaches is the amount of memory required to compute or estimate the
probability of success as well as the kind of reinforcement learning
representation where they could be used. In this regard, we use the
memory-based approach as a baseline since it is obtained directly from the
agent's observations. When comparing the learning-based and the
introspection-based approaches to this baseline, both are found to be suitable
alternatives to compute the probability of success, obtaining high levels of
similarity when compared using both the Pearson's correlation and the mean
squared error.

    

### [[2010.07620] GMH: A General Multi-hop Reasoning Model for KG Completion](http://arxiv.org/abs/2010.07620)


  Knowledge graphs are essential for numerous downstream natural language
processing applications, but are typically incomplete with many facts missing.
This results in research efforts on multi-hop reasoning task, which can be
formulated as a search process and current models typically perform short
distance reasoning. However, the long-distance reasoning is also vital with the
ability to connect the superficially unrelated entities. To the best of our
knowledge, there lacks a general framework that approaches multi-hop reasoning
in mixed long-short distance reasoning scenarios. We argue that there are two
key issues for a general multi-hop reasoning model: i) where to go, and ii)
when to stop. Therefore, we propose a general model which resolves the issues
with three modules: 1) the local-global knowledge module to estimate the
possible paths, 2) the differentiated action dropout module to explore a
diverse set of paths, and 3) the adaptive stopping search module to avoid over
searching. The comprehensive results on three datasets demonstrate the
superiority of our model with significant improvements against baselines in
both short and long distance reasoning scenarios.

    

### [[2011.13066] USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning](http://arxiv.org/abs/2011.13066)


  Most deep neural networks (DNNs) based ultrasound (US) medical image analysis
models use pretrained backbones (e.g., ImageNet) for better model
generalization. However, the domain gap between natural and medical images
causes an inevitable performance bottleneck. To alleviate this problem, an US
dataset named US-4 is constructed for direct pretraining on the same domain. It
contains over 23,000 images from four US video sub-datasets. To learn robust
features from US-4, we propose an US semi-supervised contrastive learning
method, named USCL, for pretraining. In order to avoid high similarities
between negative pairs as well as mine abundant visual features from limited US
videos, USCL adopts a sample pair generation method to enrich the feature
involved in a single step of contrastive optimization. Extensive experiments on
several downstream tasks show the superiority of USCL pretraining against
ImageNet pretraining and other state-of-the-art (SOTA) pretraining approaches.
In particular, USCL pretrained backbone achieves fine-tuning accuracy of over
94% on POCUS dataset, which is 10% higher than 84% of the ImageNet pretrained
model. The source codes of this work are available at
this https URL.

    

### [[2012.03208] Factorizing Perception and Policy for Interactive Instruction Following](http://arxiv.org/abs/2012.03208)


  Performing simple household tasks based on language directives is very
natural to humans, yet it remains an open challenge for AI agents. The
'interactive instruction following' task attempts to make progress towards
building agents that jointly navigate, interact, and reason in the environment
at every step. To address the multifaceted problem, we propose a model that
factorizes the task into interactive perception and action policy streams with
enhanced components and name it as MOCA, a Modular Object-Centric Approach. We
empirically validate that MOCA outperforms prior arts by significant margins on
the ALFRED benchmark with improved generalization.

    

### [[2106.00306] Understanding peacefulness through the world news](http://arxiv.org/abs/2106.00306)


  Peacefulness is a principal dimension of well-being for all humankind and is
the way out of inequity and every single form of violence. Thus, its
measurement has lately drawn the attention of researchers and policy-makers.
During the last years, novel digital data streams have drastically changed the
research in this field. In the current study, we exploit information extracted
from Global Data on Events, Location, and Tone (GDELT) digital news database,
to capture peacefulness through the Global Peace Index (GPI). Applying
predictive machine learning models, we demonstrate that news media attention
from GDELT can be used as a proxy for measuring GPI at a monthly level.
Additionally, we use the SHAP methodology to obtain the most important
variables that drive the predictions. This analysis highlights each country's
profile and provides explanations for the predictions overall, and particularly
for the errors and the events that drive these errors. We believe that digital
data exploited by Social Good researchers, policy-makers, and peace-builders,
with data science tools as powerful as machine learning, could contribute to
maximize the societal benefits and minimize the risks to peacefulness.

    

### [[2108.10141] Improving Accuracy of Permutation DAG Search using Best Order Score Search](http://arxiv.org/abs/2108.10141)


  The Sparsest Permutation (SP) algorithm is accurate but limited to about 9
variables in practice; the Greedy Sparest Permutation (GSP) algorithm is faster
but less weak theoretically. A compromise can be given, the Best Order Score
Search, which gives results as accurate as SP but for much larger and denser
graphs. BOSS (Best Order Score Search) is more accurate for two reason: (a) It
assumes the "brute faithfuness" assumption, which is weaker than faithfulness,
and (b) it uses a different traversal of permutations than the depth first
traversal used by GSP, obtained by taking each variable in turn and moving it
to the position in the permutation that optimizes the model score. Results are
given comparing BOSS to several related papers in the literature in terms of
performance, for linear, Gaussian data. In all cases, with the proper parameter
settings, accuracy of BOSS is lifted considerably with respect to competing
approaches. In configurations tested, models with 60 variables are feasible
with large samples out to about an average degree of 12 in reasonable time,
with near-perfect accuracy, and sparse models with an average degree of 4 are
feasible out to about 300 variables on a laptop, again with near-perfect
accuracy. Mixed continuous discrete and all-discrete datasets were also tested.
The mixed data analysis showed advantage for BOSS over GES more apparent at
higher depths with the same score; the discrete data analysis showed a very
small advantage for BOSS over GES with the same score, perhaps not enough to
prefer it.

    

### [[2109.00673] Supporting CUDA for an extended RISC-V GPU architecture](http://arxiv.org/abs/2109.00673)


  With the rapid development of scientific computation, more and more
researchers and developers are committed to implementing various
workloads/operations on different devices. Among all these devices, NVIDIA GPU
is the most popular choice due to its comprehensive documentation and excellent
development tools. As a result, there are abundant resources for hand-writing
high-performance CUDA codes. However, CUDA is mainly supported by only
commercial products and there has been no support for open-source H/W
platforms. RISC-V is the most popular choice for hardware ISA, thanks to its
elegant design and open-source license. In this project, we aim to utilize
these existing CUDA codes with RISC-V devices. More specifically, we design and
implement a pipeline that can execute CUDA source code on an RISC-V GPU
architecture. We have succeeded in executing CUDA kernels with several
important features, like multi-thread and atomic instructions, on an RISC-V GPU
architecture.

    

### [[2109.00859] CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](http://arxiv.org/abs/2109.00859)


  Pre-trained models for Natural Languages (NL) like BERT and GPT have been
recently shown to transfer well to Programming Languages (PL) and largely
benefit a broad set of code-related tasks. Despite their success, most current
methods either rely on an encoder-only (or decoder-only) pre-training that is
suboptimal for generation (resp. understanding) tasks or process the code
snippet in the same way as NL, neglecting the special characteristics of PL
such as token types. We present CodeT5, a unified pre-trained encoder-decoder
Transformer model that better leverages the code semantics conveyed from the
developer-assigned identifiers. Our model employs a unified framework to
seamlessly support both code understanding and generation tasks and allows for
multi-task learning. Besides, we propose a novel identifier-aware pre-training
task that enables the model to distinguish which code tokens are identifiers
and to recover them when they are masked. Furthermore, we propose to exploit
the user-written code comments with a bimodal dual generation task for better
NL-PL alignment. Comprehensive experiments show that CodeT5 significantly
outperforms prior methods on understanding tasks such as code defect detection
and clone detection, and generation tasks across various directions including
PL-NL, NL-PL, and PL-PL. Further analysis reveals that our model can better
capture semantic information from code. Our code and pre-trained models are
released at https: //github.com/salesforce/CodeT5 .

    

### [<title>XGBoost4J - Scala dataframe to sparse dmatrix - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost4j-scala-dataframe-to-sparse-dmatrix/2457/1)