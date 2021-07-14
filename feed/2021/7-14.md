
## 2021-7-14

### [<title>Dask has different prediction results between single GPU and multi GPUs - XGBoost</title>](https://discuss.xgboost.ai/t/dask-has-different-prediction-results-between-single-gpu-and-multi-gpus/2368/1)

### [[2107.05728] Toward Efficient Transfer Learning in 6G](http://arxiv.org/abs/2107.05728)


  6G networks will greatly expand the support for data-oriented, autonomous
applications for over the top (OTT) and networking use cases. The success of
these use cases will depend on the availability of big data sets which is not
practical in many real scenarios due to the highly dynamic behavior of systems
and the cost of data collection procedures. Transfer learning (TL) is a
promising approach to deal with these challenges through the sharing of
knowledge among diverse learning algorithms. with TL, the learning rate and
learning accuracy can be considerably improved. However, there are
implementation challenges to efficiently deploy and utilize TL in 6G. In this
paper, we initiate this discussion by providing some performance metrics to
measure the TL success. Then, we show how infrastructure, application,
management, and training planes of 6G can be adapted to handle TL. We provide
examples of TL in 6G and highlight the spatio-temporal features of data in 6G
that can lead to efficient TL. By simulation results, we demonstrate how
transferring the quantized neural network weights between two use cases can
make a trade-off between overheads and performance and attain more efficient TL
in 6G. We also provide a list of future research directions in TL for 6G.

    

### [[2107.05939] A QUIC(K) Way Through Your Firewall?](http://arxiv.org/abs/2107.05939)


  The QUIC protocol is a new approach to combine encryption and transport layer
stream abstraction into one protocol to lower latency and improve security.
However, the decision to encrypt transport layer functionality may limit the
capabilities of firewalls to protect networks. To identify these limitations we
created a test environment and analyzed generated QUIC traffic from the
viewpoint of a middlebox. This paper shows that QUIC indeed exposes traditional
stateful firewalls to UDP hole punching bypass attacks. On the contrary we show
the robustness against censorship of QUIC through the encrypted transport layer
design and analyze the capabilities to re-gain stateful tracking capabilities
by deep packet inspection of the few exposed QUIC header fields.

    

### [[2107.05954] MVPipe: Enabling Lightweight Updates and Fast Convergence in Hierarchical Heavy Hitter Detection](http://arxiv.org/abs/2107.05954)


  Finding hierarchical heavy hitters (HHHs) (i.e., hierarchical aggregates with
exceptionally huge amounts of traffic) is critical to network management, yet
it is often challenged by the requirements of fast packet processing, real-time
and accurate detection, as well as resource efficiency. Existing HHH detection
schemes either incur expensive packet updates for multiple aggregation levels
in the IP address hierarchy, or need to process sufficient packets to converge
to the required detection accuracy. We present MVPipe, an invertible sketch
that achieves both lightweight updates and fast convergence in HHH detection.
MVPipe builds on the skewness property of IP traffic to process packets via a
pipeline of majority voting executions, such that most packets can be updated
for only one or few aggregation levels in the IP address hierarchy. We show how
MVPipe can be feasibly deployed in P4-based programmable switches subject to
limited switch resources. We also theoretically analyze the accuracy and
coverage properties of MVPipe. Evaluation with real-world Internet traces shows
that MVPipe achieves high accuracy, high throughput, and fast convergence
compared to six state-of-the-art HHH detection schemes. It also incurs low
resource overhead in the Barefoot Tofino switch deployment.

    

### [[2107.05991] Learning based E2E Energy Efficient in Joint Radio and NFV Resource Allocation for 5G and Beyond Networks](http://arxiv.org/abs/2107.05991)


  In this paper, we propose a joint radio and core resource allocation
framework for NFV-enabled networks. In the proposed system model, the goal is
to maximize energy efficiency (EE), by guaranteeing end-to-end (E2E) quality of
service (QoS) for different service types. To this end, we formulate an
optimization problem in which power and spectrum resources are allocated in the
radio part. In the core part, the chaining, placement, and scheduling of
functions are performed to ensure the QoS of all users. This joint optimization
problem is modeled as a Markov decision process (MDP), considering time-varying
characteristics of the available resources and wireless channels. A soft
actor-critic deep reinforcement learning (SAC-DRL) algorithm based on the
maximum entropy framework is subsequently utilized to solve the above MDP.
Numerical results reveal that the proposed joint approach based on the SAC-DRL
algorithm could significantly reduce energy consumption compared to the case in
which R-RA and NFV-RA problems are optimized separately.

    

### [[2107.06080] Practical and Configurable Network Traffic Classification Using Probabilistic Machine Learning](http://arxiv.org/abs/2107.06080)


  Network traffic classification that is widely applicable and highly accurate
is valuable for many network security and management tasks. A flexible and
easily configurable classification framework is ideal, as it can be customized
for use in a wide variety of networks. In this paper, we propose a highly
configurable and flexible machine learning traffic classification method that
relies only on statistics of sequences of packets to distinguish known, or
approved, traffic from unknown traffic. Our method is based on likelihood
estimation, provides a measure of certainty for classification decisions, and
can classify traffic at adjustable certainty levels. Our classification method
can also be applied in different classification scenarios, each prioritizing a
different classification goal. We demonstrate how our classification scheme and
all its configurations perform well on real-world traffic from a high
performance computing network environment.

    

### [[2107.06190] Towards Machine Learning-Enabled Context Adaption for Reliable Aerial Mesh Routing](http://arxiv.org/abs/2107.06190)


  In this paper, we present Context-Adaptive PARRoT (CA-PARRoT) as an extension
of our previous work Predictive Ad-hoc Routing fueled by Reinforcement learning
and Trajectory knowledge (PARRoT). Short-term effects, as occurring in urban
surroundings, have shown to have a negative impact on the Reinforcement
Learning (RL)-based routing process. Therefore, we add a timer-based
compensation mechanism to the update process and introduce a hybrid Machine
Learning (ML) approach to classify Radio Environment Prototypes (REPs) with a
dedicated ML component and enable the protocol for autonomous context adaption.
The performance of the novel protocol is evaluated in comprehensive network
simulations considering different REPs and is compared to well-known
established routing protocols for Mobile Ad-hoc Networks (MANETs). The results
show, that CA-PARRoT is capable to compensate the challenges confronted with in
different REPs and to improve its Key Performance Indicators (KPIs) up to 23%
compared to PARRoT, and outperform established routing protocols by up to 50 %.

    

### [[2105.12827] Massive MIMO Adaptive Modulation and Coding Using Online Deep Learning](http://arxiv.org/abs/2105.12827)


  The paper describes an online deep learning algorithm for the adaptive
modulation and coding in 5G Massive MIMO. The algorithm is based on a fully
connected neural network, which is initially trained on the output of the
traditional algorithm and then is incrementally retrained by the service
feedback of its output. We show the advantage of our solution over the
state-of-the-art Q-Learning approach. We provide system-level simulation
results to support this conclusion in various scenarios with different channel
characteristics and different user speeds. Compared with traditional OLLA our
algorithm shows 10% to 20% improvement of user throughput in full buffer case.

    

### [[2107.05630] Challenges for machine learning in clinical translation of big data imaging studies](http://arxiv.org/abs/2107.05630)


  The combination of deep learning image analysis methods and large-scale
imaging datasets offers many opportunities to imaging neuroscience and
epidemiology. However, despite the success of deep learning when applied to
many neuroimaging tasks, there remain barriers to the clinical translation of
large-scale datasets and processing tools. Here, we explore the main challenges
and the approaches that have been explored to overcome them. We focus on issues
relating to data availability, interpretability, evaluation and logistical
challenges, and discuss the challenges we believe are still to be overcome to
enable the full success of big data deep learning approaches to be experienced
outside of the research field.

    

### [[2107.05634] DDCNet-Multires: Effective Receptive Field Guided Multiresolution CNN for Dense Prediction](http://arxiv.org/abs/2107.05634)


  Dense optical flow estimation is challenging when there are large
displacements in a scene with heterogeneous motion dynamics, occlusion, and
scene homogeneity. Traditional approaches to handle these challenges include
hierarchical and multiresolution processing methods. Learning-based optical
flow methods typically use a multiresolution approach with image warping when a
broad range of flow velocities and heterogeneous motion is present. Accuracy of
such coarse-to-fine methods is affected by the ghosting artifacts when images
are warped across multiple resolutions and by the vanishing problem in smaller
scene extents with higher motion contrast. Previously, we devised strategies
for building compact dense prediction networks guided by the effective
receptive field (ERF) characteristics of the network (DDCNet). The DDCNet
design was intentionally simple and compact allowing it to be used as a
building block for designing more complex yet compact networks. In this work,
we extend the DDCNet strategies to handle heterogeneous motion dynamics by
cascading DDCNet based sub-nets with decreasing extents of their ERF. Our
DDCNet with multiresolution capability (DDCNet-Multires) is compact without any
specialized network layers. We evaluate the performance of the DDCNet-Multires
network using standard optical flow benchmark datasets. Our experiments
demonstrate that DDCNet-Multires improves over the DDCNet-B0 and -B1 and
provides optical flow estimates with accuracy comparable to similar lightweight
learning-based methods.

    

### [[2107.05666] Stress Classification and Personalization: Getting the most out of the least](http://arxiv.org/abs/2107.05666)


  Stress detection and monitoring is an active area of research with important
implications for the personal, professional, and social health of an
individual. Current approaches for affective state classification use
traditional machine learning algorithms with features computed from multiple
sensor modalities. These methods are data-intensive and rely on hand-crafted
features which impede the practical applicability of these sensor systems in
daily lives. To overcome these shortcomings, we propose a novel Convolutional
Neural Network (CNN) based stress detection and classification framework
without any feature computation using data from only one sensor modality. Our
method is competitive and outperforms current state-of-the-art techniques and
achieves a classification accuracy of $92.85\%$ and an $f1$ score of $0.89$.
Through our leave-one-subject-out analysis, we also show the importance of
personalizing stress models.

    

### [[2107.05675] Quality of Service Guarantees for Physical Unclonable Functions](http://arxiv.org/abs/2107.05675)


  We consider a secret key agreement problem in which noisy physical unclonable
function (PUF) outputs facilitate reliable, secure, and private key agreement
with the help of public, noiseless, and authenticated storage. PUF outputs are
highly correlated, so transform coding methods have been combined with scalar
quantizers to extract uncorrelated bit sequences with reliability guarantees.
For PUF circuits with continuous-valued outputs, the models for transformed
outputs are made more realistic by replacing the fitted distributions with
corresponding truncated ones. The state-of-the-art PUF methods that provide
reliability guarantees to each extracted bit are shown to be inadequate to
guarantee the same reliability level for all PUF outputs. Thus, a quality of
service parameter is introduced to control the percentage of PUF outputs for
which a target reliability level can be guaranteed. A public ring oscillator
(RO) output dataset is used to illustrate that a truncated Gaussian
distribution can be fitted to transformed RO outputs that are inputs to uniform
scalar quantizers such that reliability guarantees can be provided for each bit
extracted from any PUF device under additive Gaussian noise components by
eliminating a small subset of PUF outputs. Furthermore, we conversely show that
it is not possible to provide such reliability guarantees without eliminating
any PUF output if no extra secrecy and privacy leakage is allowed.

    

### [[2107.05677] Codified audio language modeling learns useful representations for music information retrieval](http://arxiv.org/abs/2107.05677)


  We demonstrate that language models pre-trained on codified
(discretely-encoded) music audio learn representations that are useful for
downstream MIR tasks. Specifically, we explore representations from Jukebox
(Dhariwal et al. 2020): a music generation system containing a language model
trained on codified audio from 1M songs. To determine if Jukebox's
representations contain useful information for MIR, we use them as input
features to train shallow models on several MIR tasks. Relative to
representations from conventional MIR models which are pre-trained on tagging,
we find that using representations from Jukebox as input features yields 30%
stronger performance on average across four MIR tasks: tagging, genre
classification, emotion recognition, and key detection. For key detection, we
observe that representations from Jukebox are considerably stronger than those
from models pre-trained on tagging, suggesting that pre-training via codified
audio language modeling may address blind spots in conventional approaches. We
interpret the strength of Jukebox's representations as evidence that modeling
audio instead of tags provides richer representations for MIR.

    

### [[2107.05680] Hidden Convexity of Wasserstein GANs: Interpretable Generative Models with Closed-Form Solutions](http://arxiv.org/abs/2107.05680)


  Generative Adversarial Networks (GANs) are commonly used for modeling complex
distributions of data. Both the generators and discriminators of GANs are often
modeled by neural networks, posing a non-transparent optimization problem which
is non-convex and non-concave over the generator and discriminator,
respectively. Such networks are often heuristically optimized with gradient
descent-ascent (GDA), but it is unclear whether the optimization problem
contains any saddle points, or whether heuristic methods can find them in
practice. In this work, we analyze the training of Wasserstein GANs with
two-layer neural network discriminators through the lens of convex duality, and
for a variety of generators expose the conditions under which Wasserstein GANs
can be solved exactly with convex optimization approaches, or can be
represented as convex-concave games. Using this convex duality interpretation,
we further demonstrate the impact of different activation functions of the
discriminator. Our observations are verified with numerical results
demonstrating the power of the convex interpretation, with applications in
progressive training of convex architectures corresponding to linear generators
and quadratic-activation discriminators for CelebA image generation. The code
for our experiments is available at this https URL.

    

### [[2107.05682] Least-Squares Linear Dilation-Erosion Regressor Trained using Stochastic Descent Gradient or the Difference of Convex Methods](http://arxiv.org/abs/2107.05682)


  This paper presents a hybrid morphological neural network for regression
tasks called linear dilation-erosion regression ($\ell$-DER). In few words, an
$\ell$-DER model is given by a convex combination of the composition of linear
and elementary morphological operators. As a result, they yield continuous
piecewise linear functions and, thus, are universal approximators. Apart from
introducing the $\ell$-DER models, we present three approaches for training
these models: one based on stochastic descent gradient and two based on the
difference of convex programming problems. Finally, we evaluate the performance
of the $\ell$-DER model using 14 regression tasks. Although the approach based
on SDG revealed faster than the other two, the $\ell$-DER trained using a
disciplined convex-concave programming problem outperformed the others in terms
of the least mean absolute error score.

    

### [[2107.05686] Representation Learning for Out-Of-Distribution Generalization in Reinforcement Learning](http://arxiv.org/abs/2107.05686)


  Learning data representations that are useful for various downstream tasks is
a cornerstone of artificial intelligence. While existing methods are typically
evaluated on downstream tasks such as classification or generative image
quality, we propose to assess representations through their usefulness in
downstream control tasks, such as reaching or pushing objects. By training over
10,000 reinforcement learning policies, we extensively evaluate to what extent
different representation properties affect out-of-distribution (OOD)
generalization. Finally, we demonstrate zero-shot transfer of these policies
from simulation to the real world, without any domain randomization or
fine-tuning. This paper aims to establish the first systematic characterization
of the usefulness of learned representations for real-world OOD downstream
tasks.

    

### [[2107.05687] Uncertainty-based Query Strategies for Active Learning with Transformers](http://arxiv.org/abs/2107.05687)


  Active learning is the iterative construction of a classification model
through targeted labeling, enabling significant labeling cost savings. As most
research on active learning has been carried out before transformer-based
language models ("transformers") became popular, despite its practical
importance, comparably few papers have investigated how transformers can be
combined with active learning to date. This can be attributed to the fact that
using state-of-the-art query strategies for transformers induces a prohibitive
runtime overhead, which effectively cancels out, or even outweighs
aforementioned cost savings. In this paper, we revisit uncertainty-based query
strategies, which had been largely outperformed before, but are particularly
suited in the context of fine-tuning transformers. In an extensive evaluation
on five widely used text classification benchmarks, we show that considerable
improvements of up to 14.4 percentage points in area under the learning curve
are achieved, as well as a final accuracy close to the state of the art for all
but one benchmark, using only between 0.4% and 15% of the training data.

    

### [[2107.05693] Quantifying Explainability in NLP and Analyzing Algorithms for Performance-Explainability Tradeoff](http://arxiv.org/abs/2107.05693)


  The healthcare domain is one of the most exciting application areas for
machine learning, but a lack of model transparency contributes to a lag in
adoption within the industry. In this work, we explore the current art of
explainability and interpretability within a case study in clinical text
classification, using a task of mortality prediction within MIMIC-III clinical
notes. We demonstrate various visualization techniques for fully interpretable
methods as well as model-agnostic post hoc attributions, and we provide a
generalized method for evaluating the quality of explanations using infidelity
and local Lipschitz across model types from logistic regression to BERT
variants. With these metrics, we introduce a framework through which
practitioners and researchers can assess the frontier between a model's
predictive performance and the quality of its available explanations. We make
our code available to encourage continued refinement of these methods.

    

### [[2107.05707] Computational modelling and data-driven homogenisation of knitted membranes](http://arxiv.org/abs/2107.05707)


  Knitting is an effective technique for producing complex three-dimensional
surfaces owing to the inherent flexibility of interlooped yarns and recent
advances in manufacturing providing better control of local stitch patterns.
Fully yarn-level modelling of large-scale knitted membranes is not feasible.
Therefore, we consider a two-scale homogenisation approach and model the
membrane as a Kirchhoff-Love shell on the macroscale and as Euler-Bernoulli
rods on the microscale. The governing equations for both the shell and the rod
are discretised with cubic B-spline basis functions. The solution of the
nonlinear microscale problem requires a significant amount of time due to the
large deformations and the enforcement of contact constraints, rendering
conventional online computational homogenisation approaches infeasible. To
sidestep this problem, we use a pre-trained statistical Gaussian Process
Regression (GPR) model to map the macroscale deformations to macroscale
stresses. During the offline learning phase, the GPR model is trained by
solving the microscale problem for a sufficiently rich set of deformation
states obtained by either uniform or Sobol sampling. The trained GPR model
encodes the nonlinearities and anisotropies present in the microscale and
serves as a material model for the macroscale Kirchhoff-Love shell. After
verifying and validating the different components of the proposed approach, we
introduce several examples involving membranes subjected to tension and shear
to demonstrate its versatility and good performance.

    

### [[2107.05709] Optimal input representation in neural systems at the edge of chaos](http://arxiv.org/abs/2107.05709)


  Shedding light onto how biological systems represent, process and store
information in noisy environments is a key and challenging goal. A stimulating,
though controversial, hypothesis poses that operating in dynamical regimes near
the edge of a phase transition, i.e. at criticality or the "edge of chaos", can
provide information-processing living systems with important operational
advantages, creating, e.g., an optimal trade-off between robustness and
flexibility. Here, we elaborate on a recent theoretical result, which
establishes that the spectrum of covariance matrices of neural networks
representing complex inputs in a robust way needs to decay as a power-law of
the rank, with an exponent close to unity, a result that has been indeed
experimentally verified in neurons of the mouse visual cortex. Aimed at
understanding and mimicking these results, we construct an artificial neural
network and train it to classify images. Remarkably, we find that the best
performance in such a task is obtained when the network operates near the
critical point, at which the eigenspectrum of the covariance matrix follows the
very same statistics as actual neurons do. Thus, we conclude that operating
near criticality can also have -- besides the usually alleged virtues -- the
advantage of allowing for flexible, robust and efficient input representations.

    

### [[2107.05712] A Closer Look at the Adversarial Robustness of Information Bottleneck Models](http://arxiv.org/abs/2107.05712)


  We study the adversarial robustness of information bottleneck models for
classification. Previous works showed that the robustness of models trained
with information bottlenecks can improve upon adversarial training. Our
evaluation under a diverse range of white-box $l_{\infty}$ attacks suggests
that information bottlenecks alone are not a strong defense strategy, and that
previous results were likely influenced by gradient obfuscation.

    

### [[2107.05719] Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration](http://arxiv.org/abs/2107.05719)


  When facing uncertainty, decision-makers want predictions they can trust. A
machine learning provider can convey confidence to decision-makers by
guaranteeing their predictions are distribution calibrated -- amongst the
inputs that receive a predicted class probabilities vector $q$, the actual
distribution over classes is $q$. For multi-class prediction problems, however,
achieving distribution calibration tends to be infeasible, requiring sample
complexity exponential in the number of classes $C$. In this work, we introduce
a new notion -- \emph{decision calibration} -- that requires the predicted
distribution and true distribution to be ``indistinguishable'' to a set of
downstream decision-makers. When all possible decision makers are under
consideration, decision calibration is the same as distribution calibration.
However, when we only consider decision makers choosing between a bounded
number of actions (e.g. polynomial in $C$), our main result shows that
decisions calibration becomes feasible -- we design a recalibration algorithm
that requires sample complexity polynomial in the number of actions and the
number of classes. We validate our recalibration algorithm empirically:
compared to existing methods, decision calibration improves decision-making on
skin lesion and ImageNet classification with modern neural network predictors.

    

### [[2107.05729] Generalization of graph network inferences in higher-order probabilistic graphical models](http://arxiv.org/abs/2107.05729)


  Probabilistic graphical models provide a powerful tool to describe complex
statistical structure, with many real-world applications in science and
engineering from controlling robotic arms to understanding neuronal
computations. A major challenge for these graphical models is that inferences
such as marginalization are intractable for general graphs. These inferences
are often approximated by a distributed message-passing algorithm such as
Belief Propagation, which does not always perform well on graphs with cycles,
nor can it always be easily specified for complex continuous probability
distributions. Such difficulties arise frequently in expressive graphical
models that include intractable higher-order interactions. In this paper we
construct iterative message-passing algorithms using Graph Neural Networks
defined on factor graphs to achieve fast approximate inference on graphical
models that involve many-variable interactions. Experimental results on several
families of graphical models demonstrate the out-of-distribution generalization
capability of our method to different sized graphs, and indicate the domain in
which our method gains advantage over Belief Propagation.

    

### [[2107.05745] Adapting to Misspecification in Contextual Bandits](http://arxiv.org/abs/2107.05745)


  A major research direction in contextual bandits is to develop algorithms
that are computationally efficient, yet support flexible, general-purpose
function approximation. Algorithms based on modeling rewards have shown strong
empirical performance, but typically require a well-specified model, and can
fail when this assumption does not hold. Can we design algorithms that are
efficient and flexible, yet degrade gracefully in the face of model
misspecification? We introduce a new family of oracle-efficient algorithms for
$\varepsilon$-misspecified contextual bandits that adapt to unknown model
misspecification -- both for finite and infinite action settings. Given access
to an online oracle for square loss regression, our algorithm attains optimal
regret and -- in particular -- optimal dependence on the misspecification
level, with no prior knowledge. Specializing to linear contextual bandits with
infinite actions in $d$ dimensions, we obtain the first algorithm that achieves
the optimal $O(d\sqrt{T} + \varepsilon\sqrt{d}T)$ regret bound for unknown
misspecification level $\varepsilon$.
On a conceptual level, our results are enabled by a new optimization-based
perspective on the regression oracle reduction framework of Foster and Rakhlin,
which we anticipate will find broader use.

    

### [[2107.05747] SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks](http://arxiv.org/abs/2107.05747)


  State-of-the-art artificial neural networks (ANNs) require labelled data or
feedback between layers, are often biologically implausible, and are vulnerable
to adversarial attacks that humans are not susceptible to. On the other hand,
Hebbian learning in winner-take-all (WTA) networks, is unsupervised,
feed-forward, and biologically plausible. However, an objective optimization
theory for WTA networks has been missing, except under very limiting
assumptions. Here we derive formally such a theory, based on biologically
plausible but generic ANN elements. Through Hebbian learning, network
parameters maintain a Bayesian generative model of the data. There is no
supervisory loss function, but the network does minimize cross-entropy between
its activations and the input distribution. The key is a "soft" WTA where there
is no absolute "hard" winner neuron, and a specific type of Hebbian-like
plasticity of weights and biases. We confirm our theory in practice, where, in
handwritten digit (MNIST) recognition, our Hebbian algorithm, SoftHebb,
minimizes cross-entropy without having access to it, and outperforms the more
frequently used, hard-WTA-based method. Strikingly, it even outperforms
supervised end-to-end backpropagation, under certain conditions. Specifically,
in a two-layered network, SoftHebb outperforms backpropagation when the
training dataset is only presented once, when the testing data is noisy, and
under gradient-based adversarial attacks. Adversarial attacks that confuse
SoftHebb are also confusing to the human eye. Finally, the model can generate
interpolations of objects from its input distribution.

    

### [[2107.05754] EvoBA: An Evolution Strategy as a Strong Baseline forBlack-Box Adversarial Attacks](http://arxiv.org/abs/2107.05754)


  Recent work has shown how easily white-box adversarial attacks can be applied
to state-of-the-art image classifiers. However, real-life scenarios resemble
more the black-box adversarial conditions, lacking transparency and usually
imposing natural, hard constraints on the query budget.
We propose $\textbf{EvoBA}$, a black-box adversarial attack based on a
surprisingly simple evolutionary search strategy. $\textbf{EvoBA}$ is
query-efficient, minimizes $L_0$ adversarial perturbations, and does not
require any form of training.
$\textbf{EvoBA}$ shows efficiency and efficacy through results that are in
line with much more complex state-of-the-art black-box attacks such as
$\textbf{AutoZOOM}$. It is more query-efficient than $\textbf{SimBA}$, a simple
and powerful baseline black-box attack, and has a similar level of complexity.
Therefore, we propose it both as a new strong baseline for black-box
adversarial attacks and as a fast and general tool for gaining empirical
insight into how robust image classifiers are with respect to $L_0$ adversarial
perturbations.
There exist fast and reliable $L_2$ black-box attacks, such as
$\textbf{SimBA}$, and $L_{\infty}$ black-box attacks, such as
$\textbf{DeepSearch}$. We propose $\textbf{EvoBA}$ as a query-efficient $L_0$
black-box adversarial attack which, together with the aforementioned methods,
can serve as a generic tool to assess the empirical robustness of image
classifiers. The main advantages of such methods are that they run fast, are
query-efficient, and can easily be integrated in image classifiers development
pipelines.
While our attack minimises the $L_0$ adversarial perturbation, we also report
$L_2$, and notice that we compare favorably to the state-of-the-art $L_2$
black-box attack, $\textbf{AutoZOOM}$, and of the $L_2$ strong baseline,
$\textbf{SimBA}$.

    

### [[2107.05757] Kernel Continual Learning](http://arxiv.org/abs/2107.05757)


  This paper introduces kernel continual learning, a simple but effective
variant of continual learning that leverages the non-parametric nature of
kernel methods to tackle catastrophic forgetting. We deploy an episodic memory
unit that stores a subset of samples for each task to learn task-specific
classifiers based on kernel ridge regression. This does not require memory
replay and systematically avoids task interference in the classifiers. We
further introduce variational random features to learn a data-driven kernel for
each task. To do so, we formulate kernel continual learning as a variational
inference problem, where a random Fourier basis is incorporated as the latent
variable. The variational posterior distribution over the random Fourier basis
is inferred from the coreset of each task. In this way, we are able to generate
more informative kernels specific to each task, and, more importantly, the
coreset size can be reduced to achieve more compact memory, resulting in more
efficient continual learning based on episodic memory. Extensive evaluation on
four benchmarks demonstrates the effectiveness and promise of kernels for
continual learning.

    

### [[2107.05762] Strategic Instrumental Variable Regression: Recovering Causal Relationships From Strategic Responses](http://arxiv.org/abs/2107.05762)


  Machine Learning algorithms often prompt individuals to strategically modify
their observable attributes to receive more favorable predictions. As a result,
the distribution the predictive model is trained on may differ from the one it
operates on in deployment. While such distribution shifts, in general, hinder
accurate predictions, our work identifies a unique opportunity associated with
shifts due to strategic responses: We show that we can use strategic responses
effectively to recover causal relationships between the observable features and
outcomes we wish to predict. More specifically, we study a game-theoretic model
in which a principal deploys a sequence of models to predict an outcome of
interest (e.g., college GPA) for a sequence of strategic agents (e.g., college
applicants). In response, strategic agents invest efforts and modify their
features for better predictions. In such settings, unobserved confounding
variables can influence both an agent's observable features (e.g., high school
records) and outcomes. Therefore, standard regression methods generally produce
biased estimators. In order to address this issue, our work establishes a novel
connection between strategic responses to machine learning models and
instrumental variable (IV) regression, by observing that the sequence of
deployed models can be viewed as an instrument that affects agents' observable
features but does not directly influence their outcomes. Therefore, two-stage
least squares (2SLS) regression can recover the causal relationships between
observable features and outcomes. Beyond causal recovery, we can build on our
2SLS method to address two additional relevant optimization objectives: agent
outcome maximization and predictive risk minimization. Finally, our numerical
simulations on semi-synthetic data show that our methods significantly
outperform OLS regression in causal relationship estimation.

    

### [[2107.05767] Effects of personality traits in predicting grade retention of Brazilian students](http://arxiv.org/abs/2107.05767)


  Student's grade retention is a key issue faced by many education systems,
especially those in developing countries. In this paper, we seek to gauge the
relevance of students' personality traits in predicting grade retention in
Brazil. For that, we used data collected in 2012 and 2017, in the city of
Sertaozinho, countryside of the state of Sao Paulo, Brazil. The surveys taken
in Sertaozinho included several socioeconomic questions, standardized tests,
and a personality test. Moreover, students were in grades 4, 5, and 6 in 2012.
Our approach was based on training machine learning models on the surveys' data
to predict grade retention between 2012 and 2017 using information from 2012 or
before, and then using some strategies to quantify personality traits'
predictive power. We concluded that, besides proving to be fairly better than a
random classifier when isolated, personality traits contribute to prediction
even when using socioeconomic variables and standardized tests results.

    

### [[2107.05768] Combiner: Full Attention Transformer with Sparse Computation Cost](http://arxiv.org/abs/2107.05768)


  Transformers provide a class of expressive architectures that are extremely
effective for sequence modeling. However, the key limitation of transformers is
their quadratic memory and time complexity $\mathcal{O}(L^2)$ with respect to
the sequence length in attention layers, which restricts application in
extremely long sequences. Most existing approaches leverage sparsity or
low-rank assumptions in the attention matrix to reduce cost, but sacrifice
expressiveness. Instead, we propose Combiner, which provides full attention
capability in each attention head while maintaining low computation and memory
complexity. The key idea is to treat the self-attention mechanism as a
conditional expectation over embeddings at each location, and approximate the
conditional distribution with a structured factorization. Each location can
attend to all other locations, either via direct attention, or through indirect
attention to abstractions, which are again conditional expectations of
embeddings from corresponding local regions. We show that most sparse attention
patterns used in existing sparse transformers are able to inspire the design of
such factorization for full attention, resulting in the same sub-quadratic cost
($\mathcal{O}(L\log(L))$ or $\mathcal{O}(L\sqrt{L})$). Combiner is a drop-in
replacement for attention layers in existing transformers and can be easily
implemented in common frameworks. An experimental evaluation on both
autoregressive and bidirectional sequence tasks demonstrates the effectiveness
of this approach, yielding state-of-the-art results on several image and text
modeling tasks.

    

### [[2107.05775] Fast and Explicit Neural View Synthesis](http://arxiv.org/abs/2107.05775)


  We study the problem of novel view synthesis of a scene comprised of 3D
objects. We propose a simple yet effective approach that is neither continuous
nor implicit, challenging recent trends on view synthesis. We demonstrate that
although continuous radiance field representations have gained a lot of
attention due to their expressive power, our simple approach obtains comparable
or even better novel view reconstruction quality comparing with
state-of-the-art baselines while increasing rendering speed by over 400x. Our
model is trained in a category-agnostic manner and does not require
scene-specific optimization. Therefore, it is able to generalize novel view
synthesis to object categories not seen during training. In addition, we show
that with our simple formulation, we can use view synthesis as a
self-supervision signal for efficient learning of 3D geometry without explicit
3D supervision.

    

### [[2107.05786] Carle's Game: An Open-Ended Challenge in Exploratory Machine Creativity](http://arxiv.org/abs/2107.05786)


  This paper is both an introduction and an invitation. It is an introduction
to CARLE, a Life-like cellular automata simulator and reinforcement learning
environment. It is also an invitation to Carle's Game, a challenge in
open-ended machine exploration and creativity. Inducing machine agents to excel
at creating interesting patterns across multiple cellular automata universes is
a substantial challenge, and approaching this challenge is likely to require
contributions from the fields of artificial life, AI, machine learning, and
complexity, at multiple levels of interest. Carle's Game is based on machine
agent interaction with CARLE, a Cellular Automata Reinforcement Learning
Environment. CARLE is flexible, capable of simulating any of the 262,144
different rules defining Life-like cellular automaton universes. CARLE is also
fast and can simulate automata universes at a rate of tens of thousands of
steps per second through a combination of vectorization and GPU acceleration.
Finally, CARLE is simple. Compared to high-fidelity physics simulators and
video games designed for human players, CARLE's two-dimensional grid world
offers a discrete, deterministic, and atomic universal playground, despite its
complexity. In combination with CARLE, Carle's Game offers an initial set of
agent policies, learning and meta-learning algorithms, and reward wrappers that
can be tailored to encourage exploration or specific tasks.

    

### [[2107.05787] Data-Driven Low-Rank Neural Network Compression](http://arxiv.org/abs/2107.05787)


  Despite many modern applications of Deep Neural Networks (DNNs), the large
number of parameters in the hidden layers makes them unattractive for
deployment on devices with storage capacity constraints. In this paper we
propose a Data-Driven Low-rank (DDLR) method to reduce the number of parameters
of pretrained DNNs and expedite inference by imposing low-rank structure on the
fully connected layers, while controlling for the overall accuracy and without
requiring any retraining. We pose the problem as finding the lowest rank
approximation of each fully connected layer with given performance guarantees
and relax it to a tractable convex optimization problem. We show that it is
possible to significantly reduce the number of parameters in common DNN
architectures with only a small reduction in classification accuracy. We
compare DDLR with Net-Trim, which is another data-driven DNN compression
technique based on sparsity and show that DDLR consistently produces more
compressed neural networks while maintaining higher accuracy.

    

### [[2107.05798] Cautious Policy Programming: Exploiting KL Regularization in Monotonic Policy Improvement for Reinforcement Learning](http://arxiv.org/abs/2107.05798)


  In this paper, we propose cautious policy programming (CPP), a novel
value-based reinforcement learning (RL) algorithm that can ensure monotonic
policy improvement during learning. Based on the nature of entropy-regularized
RL, we derive a new entropy regularization-aware lower bound of policy
improvement that only requires estimating the expected policy advantage
function. CPP leverages this lower bound as a criterion for adjusting the
degree of a policy update for alleviating policy oscillation. Different from
similar algorithms that are mostly theory-oriented, we also propose a novel
interpolation scheme that makes CPP better scale in high dimensional control
problems. We demonstrate that the proposed algorithm can trade o? performance
and stability in both didactic classic control problems and challenging
high-dimensional Atari games.

    

### [[2107.05802] How many degrees of freedom do we need to train deep networks: a loss landscape perspective](http://arxiv.org/abs/2107.05802)


  A variety of recent works, spanning pruning, lottery tickets, and training
within random subspaces, have shown that deep neural networks can be trained
using far fewer degrees of freedom than the total number of parameters. We
explain this phenomenon by first examining the success probability of hitting a
training loss sub-level set when training within a random subspace of a given
training dimensionality. We find a sharp phase transition in the success
probability from $0$ to $1$ as the training dimension surpasses a threshold.
This threshold training dimension increases as the desired final loss
decreases, but decreases as the initial loss decreases. We then theoretically
explain the origin of this phase transition, and its dependence on
initialization and final desired loss, in terms of precise properties of the
high dimensional geometry of the loss landscape. In particular, we show via
Gordon's escape theorem, that the training dimension plus the Gaussian width of
the desired loss sub-level set, projected onto a unit sphere surrounding the
initialization, must exceed the total number of parameters for the success
probability to be large. In several architectures and datasets, we measure the
threshold training dimension as a function of initialization and demonstrate
that it is a small fraction of the total number of parameters, thereby
implying, by our theory, that successful training with so few dimensions is
possible precisely because the Gaussian width of low loss sub-level sets is
very large. Moreover, this threshold training dimension provides a strong null
model for assessing the efficacy of more sophisticated ways to reduce training
degrees of freedom, including lottery tickets as well a more optimal method we
introduce: lottery subspaces.

    

### [[2107.05804] AlterSGD: Finding Flat Minima for Continual Learning by Alternative Training](http://arxiv.org/abs/2107.05804)


  Deep neural networks suffer from catastrophic forgetting when learning
multiple knowledge sequentially, and a growing number of approaches have been
proposed to mitigate this problem. Some of these methods achieved considerable
performance by associating the flat local minima with forgetting mitigation in
continual learning. However, they inevitably need (1) tedious hyperparameters
tuning, and (2) additional computational cost. To alleviate these problems, in
this paper, we propose a simple yet effective optimization method, called
AlterSGD, to search for a flat minima in the loss landscape. In AlterSGD, we
conduct gradient descent and ascent alternatively when the network tends to
converge at each session of learning new knowledge. Moreover, we theoretically
prove that such a strategy can encourage the optimization to converge to a flat
minima. We verify AlterSGD on continual learning benchmark for semantic
segmentation and the empirical results show that we can significantly mitigate
the forgetting and outperform the state-of-the-art methods with a large margin
under challenging continual learning protocols.

    

### [[2107.05818] A Hierarchical Bayesian model for Inverse RL in Partially-Controlled Environments](http://arxiv.org/abs/2107.05818)


  Robots learning from observations in the real world using inverse
reinforcement learning (IRL) may encounter objects or agents in the
environment, other than the expert, that cause nuisance observations during the
demonstration. These confounding elements are typically removed in
fully-controlled environments such as virtual simulations or lab settings. When
complete removal is impossible the nuisance observations must be filtered out.
However, identifying the source of observations when large amounts of
observations are made is difficult. To address this, we present a hierarchical
Bayesian model that incorporates both the expert's and the confounding
elements' observations thereby explicitly modeling the diverse observations a
robot may receive. We extend an existing IRL algorithm originally designed to
work under partial occlusion of the expert to consider the diverse
observations. In a simulated robotic sorting domain containing both occlusion
and confounding elements, we demonstrate the model's effectiveness. In
particular, our technique outperforms several other comparative methods, second
only to having perfect knowledge of the subject's trajectory.

    

### [[2107.05825] Recent Advances in Leveraging Human Guidance for Sequential Decision-Making Tasks](http://arxiv.org/abs/2107.05825)


  A longstanding goal of artificial intelligence is to create artificial agents
capable of learning to perform tasks that require sequential decision making.
Importantly, while it is the artificial agent that learns and acts, it is still
up to humans to specify the particular task to be performed. Classical
task-specification approaches typically involve humans providing stationary
reward functions or explicit demonstrations of the desired tasks. However,
there has recently been a great deal of research energy invested in exploring
alternative ways in which humans may guide learning agents that may, e.g., be
more suitable for certain tasks or require less human effort. This survey
provides a high-level overview of five recent machine learning frameworks that
primarily rely on human guidance apart from pre-specified reward functions or
conventional, step-by-step action demonstrations. We review the motivation,
assumptions, and implementation of each framework, and we discuss possible
future research directions.

    

### [[2107.05834] Oversampling Divide-and-conquer for Response-skewed Kernel Ridge Regression](http://arxiv.org/abs/2107.05834)


  The divide-and-conquer method has been widely used for estimating large-scale
kernel ridge regression estimates. Unfortunately, when the response variable is
highly skewed, the divide-and-conquer kernel ridge regression (dacKRR) may
overlook the underrepresented region and result in unacceptable results. We
develop a novel response-adaptive partition strategy to overcome the
limitation. In particular, we propose to allocate the replicates of some
carefully identified informative observations to multiple nodes (local
processors). The idea is analogous to the popular oversampling technique.
Although such a technique has been widely used for addressing discrete label
skewness, extending it to the dacKRR setting is nontrivial. We provide both
theoretical and practical guidance on how to effectively over-sample the
observations under the dacKRR setting. Furthermore, we show the proposed
estimate has a smaller asymptotic mean squared error (AMSE) than that of the
classical dacKRR estimate under mild conditions. Our theoretical findings are
supported by both simulated and real-data analyses.

    

### [[2107.05842] Motion Planning by Learning the Solution Manifold in Trajectory Optimization](http://arxiv.org/abs/2107.05842)


  The objective function used in trajectory optimization is often non-convex
and can have an infinite set of local optima. In such cases, there are diverse
solutions to perform a given task. Although there are a few methods to find
multiple solutions for motion planning, they are limited to generating a finite
set of solutions. To address this issue, we presents an optimization method
that learns an infinite set of solutions in trajectory optimization. In our
framework, diverse solutions are obtained by learning latent representations of
solutions. Our approach can be interpreted as training a deep generative model
of collision-free trajectories for motion planning. The experimental results
indicate that the trained model represents an infinite set of homotopic
solutions for motion planning problems.

    

### [[2107.05847] Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges](http://arxiv.org/abs/2107.05847)


  Most machine learning algorithms are configured by one or several
hyperparameters that must be carefully chosen and often considerably impact
performance. To avoid a time consuming and unreproducible manual
trial-and-error process to find well-performing hyperparameter configurations,
various automatic hyperparameter optimization (HPO) methods, e.g., based on
resampling error estimation for supervised machine learning, can be employed.
After introducing HPO from a general perspective, this paper reviews important
HPO methods such as grid or random search, evolutionary algorithms, Bayesian
optimization, Hyperband and racing. It gives practical recommendations
regarding important choices to be made when conducting HPO, including the HPO
algorithms themselves, performance evaluation, how to combine HPO with ML
pipelines, runtime improvements, and parallelization.

    

### [[2107.05849] Model Selection with Near Optimal Rates for Reinforcement Learning with General Model Classes](http://arxiv.org/abs/2107.05849)


  We address the problem of model selection for the finite horizon episodic
Reinforcement Learning (RL) problem where the transition kernel $P^*$ belongs
to a family of models $\mathcal{P}^*$ with finite metric entropy. In the model
selection framework, instead of $\mathcal{P}^*$, we are given $M$ nested
families of transition kernels $\cP_1 \subset \cP_2 \subset \ldots \subset
\cP_M$. We propose and analyze a novel algorithm, namely \emph{Adaptive
Reinforcement Learning (General)} (\texttt{ARL-GEN}) that adapts to the
smallest such family where the true transition kernel $P^*$ lies.
\texttt{ARL-GEN} uses the Upper Confidence Reinforcement Learning
(\texttt{UCRL}) algorithm with value targeted regression as a blackbox and puts
a model selection module at the beginning of each epoch. Under a mild
separability assumption on the model classes, we show that \texttt{ARL-GEN}
obtains a regret of
$\Tilde{\mathcal{O}}(d_{\mathcal{E}}^*H^2+\sqrt{d_{\mathcal{E}}^* \mathbb{M}^*
H^2 T})$, with high probability, where $H$ is the horizon length, $T$ is the
total number of steps, $d_{\mathcal{E}}^*$ is the Eluder dimension and
$\mathbb{M}^*$ is the metric entropy corresponding to $\mathcal{P}^*$. Note
that this regret scaling matches that of an oracle that knows $\mathcal{P}^*$
in advance. We show that the cost of model selection for \texttt{ARL-GEN} is an
additive term in the regret having a weak dependence on $T$. Subsequently, we
remove the separability assumption and consider the setup of linear mixture
MDPs, where the transition kernel $P^*$ has a linear function approximation.
With this low rank structure, we propose novel adaptive algorithms for model
selection, and obtain (order-wise) regret identical to that of an oracle with
knowledge of the true model class.

    

### [[2107.05855] Automated Learning Rate Scheduler for Large-batch Training](http://arxiv.org/abs/2107.05855)


  Large-batch training has been essential in leveraging large-scale datasets
and models in deep learning. While it is computationally beneficial to use
large batch sizes, it often requires a specially designed learning rate (LR)
schedule to achieve a comparable level of performance as in smaller batch
training. Especially, when the number of training epochs is constrained, the
use of a large LR and a warmup strategy is critical in the final performance of
large-batch training due to the reduced number of updating steps. In this work,
we propose an automated LR scheduling algorithm which is effective for neural
network training with a large batch size under the given epoch budget. In
specific, the whole schedule consists of two phases: adaptive warmup and
predefined decay, where the LR is increased until the training loss no longer
decreases and decreased to zero until the end of training. Here, whether the
training loss has reached the minimum value is robustly checked with Gaussian
process smoothing in an online manner with a low computational burden. Coupled
with adaptive stochastic optimizers such as AdamP and LAMB, the proposed
scheduler successfully adjusts the LRs without cumbersome hyperparameter tuning
and achieves comparable or better performances than tuned baselines on various
image classification benchmarks and architectures with a wide range of batch
sizes.

    

### [[2107.05884] Auto IV: Counterfactual Prediction via Automatic Instrumental Variable Decomposition](http://arxiv.org/abs/2107.05884)


  Instrumental variables (IVs), sources of treatment randomization that are
conditionally independent of the outcome, play an important role in causal
inference with unobserved confounders. However, the existing IV-based
counterfactual prediction methods need well-predefined IVs, while it's an art
rather than science to find valid IVs in many real-world scenes. Moreover, the
predefined hand-made IVs could be weak or erroneous by violating the conditions
of valid IVs. These thorny facts hinder the application of the IV-based
counterfactual prediction methods. In this paper, we propose a novel Automatic
Instrumental Variable decomposition (AutoIV) algorithm to automatically
generate representations serving the role of IVs from observed variables (IV
candidates). Specifically, we let the learned IV representations satisfy the
relevance condition with the treatment and exclusion condition with the outcome
via mutual information maximization and minimization constraints, respectively.
We also learn confounder representations by encouraging them to be relevant to
both the treatment and the outcome. The IV and confounder representations
compete for the information with their constraints in an adversarial game,
which allows us to get valid IV representations for IV-based counterfactual
prediction. Extensive experiments demonstrate that our method generates valid
IV representations for accurate IV-based counterfactual prediction.

    

### [[2107.05885] Exploiting Network Structures to Improve Semantic Representation for the Financial Domain](http://arxiv.org/abs/2107.05885)


  This paper presents the participation of the MiniTrue team in the FinSim-3
shared task on learning semantic similarities for the financial domain in
English language. Our approach combines contextual embeddings learned by
transformer-based language models with network structures embeddings extracted
on external knowledge sources, to create more meaningful representations of
financial domain entities and terms. For this, two BERT based language models
and a knowledge graph embedding model are used. Besides, we propose a voting
function to joint three basic models for the final inference. Experimental
results show that the model with the knowledge graph embeddings has achieved a
superior result than these models with only contextual embeddings.
Nevertheless, we also observe that our voting function brings an extra benefit
to the final system.

    

### [[2107.05901] Fast approximations of the Jeffreys divergence between univariate Gaussian mixture models via exponential polynomial densities](http://arxiv.org/abs/2107.05901)


  The Jeffreys divergence is a renown symmetrization of the statistical
Kullback-Leibler divergence which is often used in machine learning, signal
processing, and information sciences. Since the Jeffreys divergence between the
ubiquitous Gaussian Mixture Models are not available in closed-form, many
techniques with various pros and cons have been proposed in the literature to
either (i) estimate, (ii) approximate, or (iii) lower and upper bound this
divergence. In this work, we propose a simple yet fast heuristic to approximate
the Jeffreys divergence between two GMMs of arbitrary number of components. The
heuristic relies on converting GMMs into pairs of dually parameterized
probability densities belonging to exponential families. In particular, we
consider Polynomial Exponential Densities, and design a goodness-of-fit
criterion to measure the dissimilarity between a GMM and a PED which is a
generalization of the Hyvärinen divergence. This criterion allows one to
select the orders of the PEDs to approximate the GMMs. We demonstrate
experimentally that the computational time of our heuristic improves over the
stochastic Monte Carlo estimation baseline by several orders of magnitude while
approximating reasonably well the Jeffreys divergence, specially when the
univariate mixtures have a small number of modes.

    

### [[2107.05908] Experience Report: Deep Learning-based System Log Analysis for Anomaly Detection](http://arxiv.org/abs/2107.05908)


  Logs have been an imperative resource to ensure the reliability and
continuity of many software systems, especially large-scale distributed
systems. They faithfully record runtime information to facilitate system
troubleshooting and behavior understanding. Due to the large scale and
complexity of modern software systems, the volume of logs has reached an
unprecedented level. Consequently, for log-based anomaly detection,
conventional methods of manual inspection or even traditional machine
learning-based methods become impractical, which serve as a catalyst for the
rapid development of deep learning-based solutions. However, there is currently
a lack of rigorous comparison among the representative log-based anomaly
detectors which resort to neural network models. Moreover, the
re-implementation process demands non-trivial efforts and bias can be easily
introduced. To better understand the characteristics of different anomaly
detectors, in this paper, we provide a comprehensive review and evaluation on
five popular models used by six state-of-the-art methods. Particularly, four of
the selected methods are unsupervised and the remaining two are supervised.
These methods are evaluated with two publicly-available log datasets, which
contain nearly 16 millions log messages and 0.4 million anomaly instances in
total. We believe our work can serve as a basis in this field and contribute to
the future academic researches and industrial applications.

    

### [[2107.05911] Induced Domain Adaptation](http://arxiv.org/abs/2107.05911)


  We formulate the problem of induced domain adaptation (IDA) when the
underlying distribution/domain shift is introduced by the model being deployed.
Our formulation is motivated by applications where the deployed machine
learning models interact with human agents, and will ultimately face responsive
and interactive data distributions. We formalize the discussions of the
transferability of learning in our IDA setting by studying how the model
trained on the available source distribution (data) would translate to the
performance on the induced domain. We provide both upper bounds for the
performance gap due to the induced domain shift, as well as lower bound for the
trade-offs a classifier has to suffer on either the source training
distribution or the induced target distribution. We provide further
instantiated analysis for two popular domain adaptation settings with covariate
shift and label shift. We highlight some key properties of IDA, as well as
computational and learning challenges.

    

### [[2107.05913] Can Less be More? When Increasing-to-Balancing Label Noise Rates Considered Beneficial](http://arxiv.org/abs/2107.05913)


  In this paper, we answer the question when inserting label noise (less
informative labels) can instead return us more accurate and fair models. We are
primarily inspired by two observations that 1) increasing a certain class of
instances' label noise to balance the noise rates (increasing-to-balancing)
results in an easier learning problem; 2) Increasing-to-balancing improves
fairness guarantees against label bias. In this paper, we will first quantify
the trade-offs introduced by increasing a certain group of instances' label
noise rate w.r.t. the learning difficulties and performance guarantees. We
analytically demonstrate when such an increase proves to be beneficial, in
terms of either improved generalization errors or the fairness guarantees. Then
we present a method to leverage our idea of inserting label noise for the task
of learning with noisy labels, either without or with a fairness constraint.
The primary technical challenge we face is due to the fact that we would not
know which data instances are suffering from higher noise, and we would not
have the ground truth labels to verify any possible hypothesis. We propose a
detection method that informs us which group of labels might suffer from higher
noise, without using ground truth information. We formally establish the
effectiveness of the proposed solution and demonstrate it with extensive
experiments.

    

### [[2107.05916] Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music](http://arxiv.org/abs/2107.05916)


  Modern keyboards allow a musician to play multiple instruments at the same
time by assigning zones -- fixed pitch ranges of the keyboard -- to different
instruments. In this paper, we aim to further extend this idea and examine the
feasibility of automatic instrumentation -- dynamically assigning instruments
to notes in solo music during performance. In addition to the online,
real-time-capable setting for performative use cases, automatic instrumentation
can also find applications in assistive composing tools in an offline setting.
Due to the lack of paired data of original solo music and their full
arrangements, we approach automatic instrumentation by learning to separate
parts (e.g., voices, instruments and tracks) from their mixture in symbolic
multitrack music, assuming that the mixture is to be played on a keyboard. We
frame the task of part separation as a sequential multi-class classification
problem and adopt machine learning to map sequences of notes into sequences of
part labels. To examine the effectiveness of our proposed models, we conduct a
comprehensive empirical evaluation over four diverse datasets of different
genres and ensembles -- Bach chorales, string quartets, game music and pop
music. Our experiments show that the proposed models outperform various
baselines. We also demonstrate the potential for our proposed models to produce
alternative convincing instrumentations for an existing arrangement by
separating its mixture into parts. All source code and audio samples can be
found at this https URL .

    

### [[2107.05917] Towards Representation Identical Privacy-Preserving Graph Neural Network via Split Learning](http://arxiv.org/abs/2107.05917)


  In recent years, the fast rise in number of studies on graph neural network
(GNN) has put it from the theories research to reality application stage.
Despite the encouraging performance achieved by GNN, less attention has been
paid to the privacy-preserving training and inference over distributed graph
data in the related literature. Due to the particularity of graph structure, it
is challenging to extend the existing private learning framework to GNN.
Motivated by the idea of split learning, we propose a \textbf{S}erver
\textbf{A}ided \textbf{P}rivacy-preserving \textbf{GNN} (SAPGNN) for the node
level task on horizontally partitioned cross-silo scenario. It offers a natural
extension of centralized GNN to isolated graph with max/min pooling
aggregation, while guaranteeing that all the private data involved in
computation still stays at local data holders. To further enhancing the data
privacy, a secure pooling aggregation mechanism is proposed. Theoretical and
experimental results show that the proposed model achieves the same accuracy as
the one learned over the combined data.

    

### [[2107.05941] Multi-Scale Label Relation Learning for Multi-Label Classification Using 1-Dimensional Convolutional Neural Networks](http://arxiv.org/abs/2107.05941)


  We present Multi-Scale Label Dependence Relation Networks (MSDN), a novel
approach to multi-label classification (MLC) using 1-dimensional convolution
kernels to learn label dependencies at multi-scale. Modern multi-label
classifiers have been adopting recurrent neural networks (RNNs) as a memory
structure to capture and exploit label dependency relations. The RNN-based MLC
models however tend to introduce a very large number of parameters that may
cause under-/over-fitting problems. The proposed method uses the 1-dimensional
convolutional neural network (1D-CNN) to serve the same purpose in a more
efficient manner. By training a model with multiple kernel sizes, the method is
able to learn the dependency relations among labels at multiple scales, while
it uses a drastically smaller number of parameters. With public benchmark
datasets, we demonstrate that our model can achieve better accuracies with much
smaller number of model parameters compared to RNN-based MLC models.

    

### [[2107.05948] On Designing Good Representation Learning Models](http://arxiv.org/abs/2107.05948)


  The goal of representation learning is different from the ultimate objective
of machine learning such as decision making, it is therefore very difficult to
establish clear and direct objectives for training representation learning
models. It has been argued that a good representation should disentangle the
underlying variation factors, yet how to translate this into training
objectives remains unknown. This paper presents an attempt to establish direct
training criterions and design principles for developing good representation
learning models. We propose that a good representation learning model should be
maximally expressive, i.e., capable of distinguishing the maximum number of
input configurations. We formally define expressiveness and introduce the
maximum expressiveness (MEXS) theorem of a general learning model. We propose
to train a model by maximizing its expressiveness while at the same time
incorporating general priors such as model smoothness. We present a conscience
competitive learning algorithm which encourages the model to reach its MEXS
whilst at the same time adheres to model smoothness prior. We also introduce a
label consistent training (LCT) technique to boost model smoothness by
encouraging it to assign consistent labels to similar samples. We present
extensive experimental results to show that our method can indeed design
representation learning models capable of developing representations that are
as good as or better than state of the art. We also show that our technique is
computationally efficient, robust against different parameter settings and can
work effectively on a variety of datasets.

    

### [[2107.05975] Detecting when pre-trained nnU-Net models fail silently for Covid-19](http://arxiv.org/abs/2107.05975)


  Automatic segmentation of lung lesions in computer tomography has the
potential to ease the burden of clinicians during the Covid-19 pandemic. Yet
predictive deep learning models are not trusted in the clinical routine due to
failing silently in out-of-distribution (OOD) data. We propose a lightweight
OOD detection method that exploits the Mahalanobis distance in the feature
space. The proposed approach can be seamlessly integrated into state-of-the-art
segmentation pipelines without requiring changes in model architecture or
training procedure, and can therefore be used to assess the suitability of
pre-trained models to new data. We validate our method with a patch-based
nnU-Net architecture trained with a multi-institutional dataset and find that
it effectively detects samples that the model segments incorrectly.

    

### [[2107.05978] DIVINE: Diverse Influential Training Points for Data Visualization and Model Refinement](http://arxiv.org/abs/2107.05978)


  As the complexity of machine learning (ML) models increases, resulting in a
lack of prediction explainability, several methods have been developed to
explain a model's behavior in terms of the training data points that most
influence the model. However, these methods tend to mark outliers as highly
influential points, limiting the insights that practitioners can draw from
points that are not representative of the training data. In this work, we take
a step towards finding influential training points that also represent the
training data well. We first review methods for assigning importance scores to
training points. Given importance scores, we propose a method to select a set
of DIVerse INfluEntial (DIVINE) training points as a useful explanation of
model behavior. As practitioners might not only be interested in finding data
points influential with respect to model accuracy, but also with respect to
other important metrics, we show how to evaluate training data points on the
basis of group fairness. Our method can identify unfairness-inducing training
points, which can be removed to improve fairness outcomes. Our quantitative
experiments and user studies show that visualizing DIVINE points helps
practitioners understand and explain model behavior better than earlier
approaches.

    

### [[2107.05984] Deep Autoregressive Models with Spectral Attention](http://arxiv.org/abs/2107.05984)


  Time series forecasting is an important problem across many domains, playing
a crucial role in multiple real-world applications. In this paper, we propose a
forecasting architecture that combines deep autoregressive models with a
Spectral Attention (SA) module, which merges global and local frequency domain
information in the model's embedded space. By characterizing in the spectral
domain the embedding of the time series as occurrences of a random process, our
method can identify global trends and seasonality patterns. Two spectral
attention models, global and local to the time series, integrate this
information within the forecast and perform spectral filtering to remove time
series's noise. The proposed architecture has a number of useful properties: it
can be effectively incorporated into well-know forecast architectures,
requiring a low number of parameters and producing interpretable results that
improve forecasting accuracy. We test the Spectral Attention Autoregressive
Model (SAAM) on several well-know forecast datasets, consistently demonstrating
that our model compares favorably to state-of-the-art approaches.

    

### [[2107.05989] Emotion Recognition for Healthcare Surveillance Systems Using Neural Networks: A Survey](http://arxiv.org/abs/2107.05989)


  Recognizing the patient's emotions using deep learning techniques has
attracted significant attention recently due to technological advancements.
Automatically identifying the emotions can help build smart healthcare centers
that can detect depression and stress among the patients in order to start the
medication early. Using advanced technology to identify emotions is one of the
most exciting topics as it defines the relationships between humans and
machines. Machines learned how to predict emotions by adopting various methods.
In this survey, we present recent research in the field of using neural
networks to recognize emotions. We focus on studying emotions' recognition from
speech, facial expressions, and audio-visual input and show the different
techniques of deploying these algorithms in the real world. These three emotion
recognition techniques can be used as a surveillance system in healthcare
centers to monitor patients. We conclude the survey with a presentation of the
challenges and the related future work to provide an insight into the
applications of using emotion recognition.

    

### [[2107.05990] Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform](http://arxiv.org/abs/2107.05990)


  Prior work on diagnosing Alzheimer's disease from magnetic resonance images
of the brain established that convolutional neural networks (CNNs) can leverage
the high-dimensional image information for classifying patients. However,
little research focused on how these models can utilize the usually
low-dimensional tabular information, such as patient demographics or laboratory
measurements. We introduce the Dynamic Affine Feature Map Transform (DAFT), a
general-purpose module for CNNs that dynamically rescales and shifts the
feature maps of a convolutional layer, conditional on a patient's tabular
clinical information. We show that DAFT is highly effective in combining 3D
image and tabular information for diagnosis and time-to-dementia prediction,
where it outperforms competing CNNs with a mean balanced accuracy of 0.622 and
mean c-index of 0.748, respectively. Our extensive ablation study provides
valuable insights into the architectural properties of DAFT. Our implementation
is available at this https URL.

    

### [[2107.05997] Scalable, Axiomatic Explanations of Deep Alzheimer's Diagnosis from Heterogeneous Data](http://arxiv.org/abs/2107.05997)


  Deep Neural Networks (DNNs) have an enormous potential to learn from complex
biomedical data. In particular, DNNs have been used to seamlessly fuse
heterogeneous information from neuroanatomy, genetics, biomarkers, and
neuropsychological tests for highly accurate Alzheimer's disease diagnosis. On
the other hand, their black-box nature is still a barrier for the adoption of
such a system in the clinic, where interpretability is absolutely essential. We
propose Shapley Value Explanation of Heterogeneous Neural Networks (SVEHNN) for
explaining the Alzheimer's diagnosis made by a DNN from the 3D point cloud of
the neuroanatomy and tabular biomarkers. Our explanations are based on the
Shapley value, which is the unique method that satisfies all fundamental axioms
for local explanations previously established in the literature. Thus, SVEHNN
has many desirable characteristics that previous work on interpretability for
medical decision making is lacking. To avoid the exponential time complexity of
the Shapley value, we propose to transform a given DNN into a Lightweight
Probabilistic Deep Network without re-training, thus achieving a complexity
only quadratic in the number of features. In our experiments on synthetic and
real data, we show that we can closely approximate the exact Shapley value with
a dramatically reduced runtime and can reveal the hidden knowledge the network
has learned from the data.

    

### [[2107.06008] Wasserstein GAN: Deep Generation applied on Bitcoins financial time series](http://arxiv.org/abs/2107.06008)


  Modeling financial time series is challenging due to their high volatility
and unexpected happenings on the market. Most financial models and algorithms
trying to fill the lack of historical financial time series struggle to perform
and are highly vulnerable to overfitting. As an alternative, we introduce in
this paper a deep neural network called the WGAN-GP, a data-driven model that
focuses on sample generation. The WGAN-GP consists of a generator and
discriminator function which utilize an LSTM architecture. The WGAN-GP is
supposed to learn the underlying structure of the input data, which in our
case, is the Bitcoin. Bitcoin is unique in its behavior; the prices fluctuate
what makes guessing the price trend hardly impossible. Through adversarial
training, the WGAN-GP should learn the underlying structure of the bitcoin and
generate very similar samples of the bitcoin distribution. The generated
synthetic time series are visually indistinguishable from the real data. But
the numerical results show that the generated data were close to the real data
distribution but distinguishable. The model mainly shows a stable learning
behavior. However, the model has space for optimization, which could be
achieved by adjusting the hyperparameters.

    

### [[2107.06011] Teaching Agents how to Map: Spatial Reasoning for Multi-Object Navigation](http://arxiv.org/abs/2107.06011)


  In the context of visual navigation, the capacity to map a novel environment
is necessary for an agent to exploit its observation history in the considered
place and efficiently reach known goals. This ability can be associated with
spatial reasoning, where an agent is able to perceive spatial relationships and
regularities, and discover object affordances. In classical Reinforcement
Learning (RL) setups, this capacity is learned from reward alone. We introduce
supplementary supervision in the form of auxiliary tasks designed to favor the
emergence of spatial perception capabilities in agents trained for a
goal-reaching downstream objective. We show that learning to estimate metrics
quantifying the spatial relationships between an agent at a given location and
a goal to reach has a high positive impact in Multi-Object Navigation settings.
Our method significantly improves the performance of different baseline agents,
that either build an explicit or implicit representation of the environment,
even matching the performance of incomparable oracle agents taking ground-truth
maps as input.

    

### [[2107.06020] A Deep Generative Artificial Intelligence system to decipher species coexistence patterns](http://arxiv.org/abs/2107.06020)


  1. Deciphering coexistence patterns is a current challenge to understanding
diversity maintenance, especially in rich communities where the complexity of
these patterns is magnified through indirect interactions that prevent their
approximation with classical experimental approaches. 2. We explore
cutting-edge Machine Learning techniques called Generative Artificial
Intelligence (GenAI) to decipher species coexistence patterns in vegetation
patches, training generative adversarial networks (GAN) and variational
AutoEncoders (VAE) that are then used to unravel some of the mechanisms behind
community assemblage. 3. The GAN accurately reproduces the species composition
of real patches as well as the affinity of plant species to different soil
types, and the VAE also reaches a high level of accuracy, above 99%. Using the
artificially generated patches, we found that high order interactions tend to
suppress the positive effects of low order interactions. Finally, by
reconstructing successional trajectories we could identify the pioneer species
with larger potential to generate a high diversity of distinct patches in terms
of species composition. 4. Understanding the complexity of species coexistence
patterns in diverse ecological communities requires new approaches beyond
heuristic rules. Generative Artificial Intelligence can be a powerful tool to
this end as it allows to overcome the inherent dimensionality of this
challenge.

    

### [[2107.06039] AutoScore-Imbalance: An interpretable machine learning tool for development of clinical scores with rare events data](http://arxiv.org/abs/2107.06039)


  Background: Medical decision-making impacts both individual and public
health. Clinical scores are commonly used among a wide variety of
decision-making models for determining the degree of disease deterioration at
the bedside. AutoScore was proposed as a useful clinical score generator based
on machine learning and a generalized linear model. Its current framework,
however, still leaves room for improvement when addressing unbalanced data of
rare events. Methods: Using machine intelligence approaches, we developed
AutoScore-Imbalance, which comprises three components: training dataset
optimization, sample weight optimization, and adjusted AutoScore. All scoring
models were evaluated on the basis of their area under the curve (AUC) in the
receiver operating characteristic analysis and balanced accuracy (i.e., mean
value of sensitivity and specificity). By utilizing a publicly accessible
dataset from Beth Israel Deaconess Medical Center, we assessed the proposed
model and baseline approaches in the prediction of inpatient mortality.
Results: AutoScore-Imbalance outperformed baselines in terms of AUC and
balanced accuracy. The nine-variable AutoScore-Imbalance sub-model achieved the
highest AUC of 0.786 (0.732-0.839) while the eleven-variable original AutoScore
obtained an AUC of 0.723 (0.663-0.783), and the logistic regression with 21
variables obtained an AUC of 0.743 (0.685-0.800). The AutoScore-Imbalance
sub-model (using down-sampling algorithm) yielded an AUC of 0. 0.771
(0.718-0.823) with only five variables, demonstrating a good balance between
performance and variable sparsity. Conclusions: The AutoScore-Imbalance tool
has the potential to be applied to highly unbalanced datasets to gain further
insight into rare medical events and to facilitate real-world clinical
decision-making.

    

### [[2107.06048] A Graph Data Augmentation Strategy with Entropy Preserving](http://arxiv.org/abs/2107.06048)


  The Graph Convolutional Networks (GCNs) proposed by Kipf and Welling are
effective models for semi-supervised learning, but facing the obstacle of
over-smoothing, which will weaken the representation ability of GCNs. Recently
some works are proposed to tackle with above limitation by randomly perturbing
graph topology or feature matrix to generate data augmentations as input for
training. However, these operations have to pay the price of information
structure integrity breaking, and inevitably sacrifice information
stochastically from original graph. In this paper, we introduce a novel graph
entropy definition as an quantitative index to evaluate feature information
diffusion among a graph. Under considerations of preserving graph entropy, we
propose an effective strategy to generate perturbed training data using a
stochastic mechanism but guaranteeing graph topology integrity and with only a
small amount of graph entropy decaying. Extensive experiments have been
conducted on real-world datasets and the results verify the effectiveness of
our proposed method in improving semi-supervised node classification accuracy
compared with a surge of baselines. Beyond that, our proposed approach
significantly enhances the robustness and generalization ability of GCNs during
the training process.

    

### [[2107.06050] Force-in-domain GAN inversion](http://arxiv.org/abs/2107.06050)


  Empirical works suggest that various semantics emerge in the latent space of
Generative Adversarial Networks (GANs) when being trained to generate images.
To perform real image editing, it requires an accurate mapping from the real
image to the latent space to leveraging these learned semantics, which is
important yet difficult. An in-domain GAN inversion approach is recently
proposed to constraint the inverted code within the latent space by forcing the
reconstructed image obtained from the inverted code within the real image
space. Empirically, we find that the inverted code by the in-domain GAN can
deviate from the latent space significantly. To solve this problem, we propose
a force-in-domain GAN based on the in-domain GAN, which utilizes a
discriminator to force the inverted code within the latent space. The
force-in-domain GAN can also be interpreted by a cycle-GAN with slight
modification. Extensive experiments show that our force-in-domain GAN not only
reconstructs the target image at the pixel level, but also align the inverted
code with the latent space well for semantic editing.

    

### [[2107.06057] Fast-Slow Streamflow Model Using Mass-Conserving LSTM](http://arxiv.org/abs/2107.06057)


  Streamflow forecasting is key to effectively managing water resources and
preparing for the occurrence of natural calamities being exacerbated by climate
change. Here we use the concept of fast and slow flow components to create a
new mass-conserving Long Short-Term Memory (LSTM) neural network model. It uses
hydrometeorological time series and catchment attributes to predict daily river
discharges. Preliminary results evidence improvement in skills for different
scores compared to the recent literature.

    

### [[2107.06064] Model of the Weak Reset Process in HfOx Resistive Memory for Deep Learning Frameworks](http://arxiv.org/abs/2107.06064)


  The implementation of current deep learning training algorithms is
power-hungry, owing to data transfer between memory and logic units.
Oxide-based RRAMs are outstanding candidates to implement in-memory computing,
which is less power-intensive. Their weak RESET regime, is particularly
attractive for learning, as it allows tuning the resistance of the devices with
remarkable endurance. However, the resistive change behavior in this regime
suffers many fluctuations and is particularly challenging to model, especially
in a way compatible with tools used for simulating deep learning. In this work,
we present a model of the weak RESET process in hafnium oxide RRAM and
integrate this model within the PyTorch deep learning framework. Validated on
experiments on a hybrid CMOS/RRAM technology, our model reproduces both the
noisy progressive behavior and the device-to-device (D2D) variability. We use
this tool to train Binarized Neural Networks for the MNIST handwritten digit
recognition task and the CIFAR-10 object classification task. We simulate our
model with and without various aspects of device imperfections to understand
their impact on the training process and identify that the D2D variability is
the most detrimental aspect. The framework can be used in the same manner for
other types of memories to identify the device imperfections that cause the
most degradation, which can, in turn, be used to optimize the devices to reduce
the impact of these imperfections.

    

### [[2107.06065] Pattern Discovery and Validation Using Scientific Research Methods](http://arxiv.org/abs/2107.06065)


  Pattern discovery, the process of discovering previously unrecognized
patterns, is often performed as an ad-hoc process with little resulting
certainty in the quality of the proposed patterns. Pattern validation, the
process of validating the accuracy of proposed patterns, remains dominated by
the simple heuristic of "the rule of three". This article shows how to use
established scientific research methods for the purpose of pattern discovery
and validation. We present a specific approach, called the handbook method,
that uses the qualitative survey, action research, and case study research for
pattern discovery and evaluation, and we discuss the underlying principle of
using scientific methods in general. We evaluate the handbook method using
three exploratory studies and demonstrate its usefulness.

    

### [[2107.06068] Calibrated Uncertainty for Molecular Property Prediction using Ensembles of Message Passing Neural Networks](http://arxiv.org/abs/2107.06068)


  Data-driven methods based on machine learning have the potential to
accelerate analysis of atomic structures. However, machine learning models can
produce overconfident predictions and it is therefore crucial to detect and
handle uncertainty carefully. Here, we extend a message passing neural network
designed specifically for predicting properties of molecules and materials with
a calibrated probabilistic predictive distribution. The method presented in
this paper differs from the previous work by considering both aleatoric and
epistemic uncertainty in a unified framework, and by re-calibrating the
predictive distribution on unseen data. Through computer experiments, we show
that our approach results in accurate models for predicting molecular formation
energies with calibrated uncertainty in and out of the training data
distribution on two public molecular benchmark datasets, QM9 and PC9. The
proposed method provides a general framework for training and evaluating neural
network ensemble models that are able to produce accurate predictions of
properties of molecules with calibrated uncertainty.

    

### [[2107.06074] On Choice of Hyper-parameter in Extreme Value Theory based on Machine Learning Techniques](http://arxiv.org/abs/2107.06074)


  Extreme value theory (EVT) is a statistical tool for analysis of extreme
events. It has a strong theoretical background, however, we need to choose
hyper-parameters
to apply EVT. In recent studies of machine learning, techniques of choosing
hyper-parameters have been well-studied. In this paper, we propose a new method
of choosing hyper-parameters in EVT based on machine learning techniques. We
also experiment our method to real-world data and show good usability of our
method.

    

### [[2107.06097] Transformer-Based Behavioral Representation Learning Enables Transfer Learning for Mobile Sensing in Small Datasets](http://arxiv.org/abs/2107.06097)


  While deep learning has revolutionized research and applications in NLP and
computer vision, this has not yet been the case for behavioral modeling and
behavioral health applications. This is because the domain's datasets are
smaller, have heterogeneous datatypes, and typically exhibit a large degree of
missingness. Therefore, off-the-shelf deep learning models require significant,
often prohibitive, adaptation. Accordingly, many research applications still
rely on manually coded features with boosted tree models, sometimes with
task-specific features handcrafted by experts. Here, we address these
challenges by providing a neural architecture framework for mobile sensing data
that can learn generalizable feature representations from time series and
demonstrates the feasibility of transfer learning on small data domains through
finetuning. This architecture combines benefits from CNN and Trans-former
architectures to (1) enable better prediction performance by learning directly
from raw minute-level sensor data without the need for handcrafted features by
up to 0.33 ROC AUC, and (2) use pretraining to outperform simpler neural models
and boosted decision trees with data from as few a dozen participants.

    

### [[2107.06098] Using Causal Analysis for Conceptual Deep Learning Explanation](http://arxiv.org/abs/2107.06098)


  Model explainability is essential for the creation of trustworthy Machine
Learning models in healthcare. An ideal explanation resembles the
decision-making process of a domain expert and is expressed using concepts or
terminology that is meaningful to the clinicians. To provide such an
explanation, we first associate the hidden units of the classifier to
clinically relevant concepts. We take advantage of radiology reports
accompanying the chest X-ray images to define concepts. We discover sparse
associations between concepts and hidden units using a linear sparse logistic
regression. To ensure that the identified units truly influence the
classifier's outcome, we adopt tools from Causal Inference literature and, more
specifically, mediation analysis through counterfactual interventions. Finally,
we construct a low-depth decision tree to translate all the discovered concepts
into a straightforward decision rule, expressed to the radiologist. We
evaluated our approach on a large chest x-ray dataset, where our model produces
a global explanation consistent with clinical knowledge.

    

### [[2107.06099] Drug-Target Interaction Prediction with Graph Attention networks](http://arxiv.org/abs/2107.06099)


  Motivation: Predicting Drug-Target Interaction (DTI) is a well-studied topic
in bioinformatics due to its relevance in the fields of proteomics and
pharmaceutical research. Although many machine learning methods have been
successfully applied in this task, few of them aim at leveraging the inherent
heterogeneous graph structure in the DTI network to address the challenge. For
better learning and interpreting the DTI topological structure and the
similarity, it is desirable to have methods specifically for predicting
interactions from the graph structure.
Results: We present an end-to-end framework, DTI-GAT (Drug-Target Interaction
prediction with Graph Attention networks) for DTI predictions. DTI-GAT
incorporates a deep neural network architecture that operates on
graph-structured data with the attention mechanism, which leverages both the
interaction patterns and the features of drug and protein sequences. DTI-GAT
facilitates the interpretation of the DTI topological structure by assigning
different attention weights to each node with the self-attention mechanism.
Experimental evaluations show that DTI-GAT outperforms various state-of-the-art
systems on the binary DTI prediction problem. Moreover, the independent study
results further demonstrate that our model can be generalized better than other
conventional methods.
Availability: The source code and all datasets are available at
this https URL


### [[2107.06104] Functional Magnetic Resonance Imaging data augmentation through conditional ICA](http://arxiv.org/abs/2107.06104)


  Advances in computational cognitive neuroimaging research are related to the
availability of large amounts of labeled brain imaging data, but such data are
scarce and expensive to generate. While powerful data generation mechanisms,
such as Generative Adversarial Networks (GANs), have been designed in the last
decade for computer vision, such improvements have not yet carried over to
brain imaging. A likely reason is that GANs training is ill-suited to the
noisy, high-dimensional and small-sample data available in functional
this http URL this paper, we introduce Conditional Independent Components
Analysis (Conditional ICA): a fast functional Magnetic Resonance Imaging (fMRI)
data augmentation technique, that leverages abundant resting-state data to
create images by sampling from an ICA decomposition. We then propose a
mechanism to condition the generator on classes observed with few samples. We
first show that the generative mechanism is successful at synthesizing data
indistinguishable from observations, and that it yields gains in classification
accuracy in brain decoding problems. In particular it outperforms GANs while
being much easier to optimize and interpret. Lastly, Conditional ICA enhances
classification accuracy in eight datasets without further parameters tuning.

    

### [[2107.06106] Conservative Offline Distributional Reinforcement Learning](http://arxiv.org/abs/2107.06106)


  Many reinforcement learning (RL) problems in practice are offline, learning
purely from observational data. A key challenge is how to ensure the learned
policy is safe, which requires quantifying the risk associated with different
actions. In the online setting, distributional RL algorithms do so by learning
the distribution over returns (i.e., cumulative rewards) instead of the
expected return; beyond quantifying risk, they have also been shown to learn
better representations for planning. We propose Conservative Offline
Distributional Actor Critic (CODAC), an offline RL algorithm suitable for both
risk-neutral and risk-averse domains. CODAC adapts distributional RL to the
offline setting by penalizing the predicted quantiles of the return for
out-of-distribution actions. We prove that CODAC learns a conservative return
distribution -- in particular, for finite MDPs, CODAC converges to an uniform
lower bound on the quantiles of the return distribution; our proof relies on a
novel analysis of the distributional Bellman operator. In our experiments, on
two challenging robot navigation tasks, CODAC successfully learns risk-averse
policies using offline data collected purely from risk-neutral agents.
Furthermore, CODAC is state-of-the-art on the D4RL MuJoCo benchmark in terms of
both expected and risk-sensitive performance.

    

### [[2107.06115] A Deep Reinforcement Learning Approach for Traffic Signal Control Optimization](http://arxiv.org/abs/2107.06115)


  Inefficient traffic signal control methods may cause numerous problems, such
as traffic congestion and waste of energy. Reinforcement learning (RL) is a
trending data-driven approach for adaptive traffic signal control in complex
urban traffic networks. Although the development of deep neural networks (DNN)
further enhances its learning capability, there are still some challenges in
applying deep RLs to transportation networks with multiple signalized
intersections, including non-stationarity environment, exploration-exploitation
dilemma, multi-agent training schemes, continuous action spaces, etc. In order
to address these issues, this paper first proposes a multi-agent deep
deterministic policy gradient (MADDPG) method by extending the actor-critic
policy gradient algorithms. MADDPG has a centralized learning and decentralized
execution paradigm in which critics use additional information to streamline
the training process, while actors act on their own local observations. The
model is evaluated via simulation on the Simulation of Urban MObility (SUMO)
platform. Model comparison results show the efficiency of the proposed
algorithm in controlling traffic lights.

    

### [[2107.06126] DiCOVA-Net: Diagnosing COVID-19 using Acoustics based on Deep Residual Network for the DiCOVA Challenge 2021](http://arxiv.org/abs/2107.06126)


  In this paper, we propose a deep residual network-based method, namely the
DiCOVA-Net, to identify COVID-19 infected patients based on the acoustic
recording of their coughs. Since there are far more healthy people than
infected patients, this classification problem faces the challenge of
imbalanced data. To improve the model's ability to recognize minority class
(the infected patients), we introduce data augmentation and cost-sensitive
methods into our model. Besides, considering the particularity of this task, we
deploy some fine-tuning techniques to adjust the pre-training ResNet50.
Furthermore, to improve the model's generalizability, we use ensemble learning
to integrate prediction results from multiple base classifiers generated using
different random seeds. To evaluate the proposed DiCOVA-Net's performance, we
conducted experiments with the DiCOVA challenge dataset. The results show that
our method has achieved 85.43\% in AUC, among the top of all competing teams.

    

### [[2107.06131] Identification of Dynamical Systems using Symbolic Regression](http://arxiv.org/abs/2107.06131)


  We describe a method for the identification of models for dynamical systems
from observational data. The method is based on the concept of symbolic
regression and uses genetic programming to evolve a system of ordinary
differential equations (ODE). The novelty is that we add a step of
gradient-based optimization of the ODE parameters. For this we calculate the
sensitivities of the solution to the initial value problem (IVP) using
automatic differentiation. The proposed approach is tested on a set of 19
problem instances taken from the literature which includes datasets from
simulated systems as well as datasets captured from mechanical systems. We find
that gradient-based optimization of parameters improves predictive accuracy of
the models. The best results are obtained when we first fit the individual
equations to the numeric differences and then subsequently fine-tune the
identified parameter values by fitting the IVP solution to the observed
variable values.

    

### [[2107.06158] Correlation Analysis between the Robustness of Sparse Neural Networks and their Random Hidden Structural Priors](http://arxiv.org/abs/2107.06158)


  Deep learning models have been shown to be vulnerable to adversarial attacks.
This perception led to analyzing deep learning models not only from the
perspective of their performance measures but also their robustness to certain
types of adversarial attacks. We take another step forward in relating the
architectural structure of neural networks from a graph theoretic perspective
to their robustness. We aim to investigate any existing correlations between
graph theoretic properties and the robustness of Sparse Neural Networks. Our
hypothesis is, that graph theoretic properties as a prior of neural network
structures are related to their robustness. To answer to this hypothesis, we
designed an empirical study with neural network models obtained through random
graphs used as sparse structural priors for the networks. We additionally
investigated the evaluation of a randomly pruned fully connected network as a
point of reference.
We found that robustness measures are independent of initialization methods
but show weak correlations with graph properties: higher graph densities
correlate with lower robustness, but higher average path lengths and average
node eccentricities show negative correlations with robustness measures. We
hope to motivate further empirical and analytical research to tightening an
answer to our hypothesis.

    

### [[2107.06174] National-scale electricity peak load forecasting: Traditional, machine learning, or hybrid model?](http://arxiv.org/abs/2107.06174)


  As the volatility of electricity demand increases owing to climate change and
electrification, the importance of accurate peak load forecasting is
increasing. Traditional peak load forecasting has been conducted through time
series-based models; however, recently, new models based on machine or deep
learning are being introduced. This study performs a comparative analysis to
determine the most accurate peak load-forecasting model for Korea, by comparing
the performance of time series, machine learning, and hybrid models. Seasonal
autoregressive integrated moving average with exogenous variables (SARIMAX) is
used for the time series model. Artificial neural network (ANN), support vector
regression (SVR), and long short-term memory (LSTM) are used for the machine
learning models. SARIMAX-ANN, SARIMAX-SVR, and SARIMAX-LSTM are used for the
hybrid models. The results indicate that the hybrid models exhibit significant
improvement over the SARIMAX model. The LSTM-based models outperformed the
others; the single and hybrid LSTM models did not exhibit a significant
performance difference. In the case of Korea's highest peak load in 2019, the
predictive power of the LSTM model proved to be greater than that of the
SARIMAX-LSTM model. The LSTM, SARIMAX-SVR, and SARIMAX-LSTM models outperformed
the current time series-based forecasting model used in Korea. Thus, Korea's
peak load-forecasting performance can be improved by including machine learning
or hybrid models.

    

### [[2107.06181] Intermittent Jamming against Telemetry and Telecommand of Satellite Systems and A Learning-driven Detection Strategy](http://arxiv.org/abs/2107.06181)


  Towards sixth-generation networks (6G), satellite communication systems,
especially based on Low Earth Orbit (LEO) networks, become promising due to
their unique and comprehensive capabilities. These advantages are accompanied
by a variety of challenges such as security vulnerabilities, management of
hybrid systems, and high mobility. In this paper, firstly, a security
deficiency in the physical layer is addressed with a conceptual framework,
considering the cyber-physical nature of the satellite systems, highlighting
the potential attacks. Secondly, a learning-driven detection scheme is
proposed, and the lightweight convolutional neural network (CNN) is designed.
The performance of the designed CNN architecture is compared with a prevalent
machine learning algorithm, support vector machine (SVM). The results show that
deficiency attacks against the satellite systems can be detected by employing
the proposed scheme.

    

### [[2107.06182] Predictive models for wind speed using artificial intelligence and copula](http://arxiv.org/abs/2107.06182)


  Electricity generation from burning fossil fuels is one of the major
contributors to global warming. Renewable energy sources are a viable
alternative to produce electrical energy and to reduce the emission from the
power industry. These energy sources are the building blocks of green energy,
which all have different characteristics. Their availabilities are also
diverse, depending on geographical locations and other parameters. Low
implementation cost and distributed availability all over the world uplifts
their popularity exponentially. Therefore, it has unlocked opportunities for
consumers to produce electricity locally and use it on-site, which reduces
dependency on centralized utility companies. The research considers two main
objectives: the prediction of wind speed that simplifies wind farm planning and
feasibility study. Secondly, the need to understand the dependency structure of
the wind speeds of multiple distant locations. To address the first objective,
twelve artificial intelligence algorithms were used for wind speed prediction
from collected meteorological parameters. The model performances were compared
to determine the wind speed prediction accuracy. The results show a deep
learning approach, long short-term memory (LSTM) outperforms other models with
the highest accuracy of 97.8%. For dependency, a multivariate cumulative
distribution function, Copula, was used to find the joint distribution of two
or more distant location wind speeds, followed by a case study. We found that
the appropriate copula family and the parameters vary based on the distance in
between. For the case study, Joe-Frank (BB8) copula shows an efficient joint
distribution fit for a wind speed pair with a standard error of 0.0094.
Finally, some insights about the uncertainty aspects of wind speed dependency
were addressed.

    

### [[2107.06187] Deep Ranking with Adaptive Margin Triplet Loss](http://arxiv.org/abs/2107.06187)


  We propose a simple modification from a fixed margin triplet loss to an
adaptive margin triplet loss. While the original triplet loss is used widely in
classification problems such as face recognition, face re-identification and
fine-grained similarity, our proposed loss is well suited for rating datasets
in which the ratings are continuous values. In contrast to original triplet
loss where we have to sample data carefully, in out method, we can generate
triplets using the whole dataset, and the optimization can still converge
without frequently running into a model collapsing issue. The adaptive margins
only need to be computed once before the training, which is much less expensive
than generating triplets after every epoch as in the fixed margin case. Besides
substantially improved training stability (the proposed model never collapsed
in our experiments compared to a couple of times that the training collapsed on
existing triplet loss), we achieved slightly better performance than the
original triplet loss on various rating datasets and network architectures.

    

### [[2107.06195] Transfer Learning in Multi-Agent Reinforcement Learning with Double Q-Networks for Distributed Resource Sharing in V2X Communication](http://arxiv.org/abs/2107.06195)


  This paper addresses the problem of decentralized spectrum sharing in
vehicle-to-everything (V2X) communication networks. The aim is to provide
resource-efficient coexistence of vehicle-to-infrastructure(V2I) and
vehicle-to-vehicle(V2V) links. A recent work on the topic proposes a
multi-agent reinforcement learning (MARL) approach based on deep Q-learning,
which leverages a fingerprint-based deep Q-network (DQN) architecture. This
work considers an extension of this framework by combining Double Q-learning
(via Double DQN) and transfer learning. The motivation behind is that Double
Q-learning can alleviate the problem of overestimation of the action values
present in conventional Q-learning, while transfer learning can leverage
knowledge acquired by an expert model to accelerate learning in the MARL
setting. The proposed algorithm is evaluated in a realistic V2X setting, with
synthetic data generated based on a geometry-based propagation model that
incorporates location-specific geographical descriptors of the simulated
environment(outlines of buildings, foliage, and vehicles). The advantages of
the proposed approach are demonstrated via numerical simulations.

    

### [[2107.06196] No Regrets for Learning the Prior in Bandits](http://arxiv.org/abs/2107.06196)


  We propose ${\tt AdaTS}$, a Thompson sampling algorithm that adapts
sequentially to bandit tasks that it interacts with. The key idea in ${\tt
AdaTS}$ is to adapt to an unknown task prior distribution by maintaining a
distribution over its parameters. When solving a bandit task, that uncertainty
is marginalized out and properly accounted for. ${\tt AdaTS}$ is a
fully-Bayesian algorithm that can be implemented efficiently in several classes
of bandit problems. We derive upper bounds on its Bayes regret that quantify
the loss due to not knowing the task prior, and show that it is small. Our
theory is supported by experiments, where ${\tt AdaTS}$ outperforms prior
algorithms and works well even in challenging real-world problems.

    

### [[2107.06197] Generative Adversarial Learning via Kernel Density Discrimination](http://arxiv.org/abs/2107.06197)


  We introduce Kernel Density Discrimination GAN (KDD GAN), a novel method for
generative adversarial learning. KDD GAN formulates the training as a
likelihood ratio optimization problem where the data distributions are written
explicitly via (local) Kernel Density Estimates (KDE). This is inspired by the
recent progress in contrastive learning and its relation to KDE. We define the
KDEs directly in feature space and forgo the requirement of invertibility of
the kernel feature mappings. In our approach, features are no longer optimized
for linear separability, as in the original GAN formulation, but for the more
general discrimination of distributions in the feature space. We analyze the
gradient of our loss with respect to the feature representation and show that
it is better behaved than that of the original hinge loss. We perform
experiments with the proposed KDE-based loss, used either as a training loss or
a regularization term, on both CIFAR10 and scaled versions of ImageNet. We use
BigGAN/SA-GAN as a backbone and baseline, since our focus is not to design the
architecture of the networks. We show a boost in the quality of generated
samples with respect to FID from 10% to 40% compared to the baseline. Code will
be made available.

    

### [[2107.06206] ML-Quest: A Game for Introducing Machine Learning Concepts to K-12 Students](http://arxiv.org/abs/2107.06206)


  Today, Machine Learning (ML) is of a great importance to society due to the
availability of huge data and high computational resources. This ultimately led
to the introduction of ML concepts at multiple levels of education including
K-12 students to promote computational thinking. However, teaching these
concepts to K-12 through traditional methodologies such as video lectures and
books is challenging. Many studies in the literature have reported that using
interactive environments such as games to teach computational thinking and
programming improves retention capacity and motivation among students.
Therefore, introducing ML concepts using a game might enhance students'
understanding of the subject and motivate them to learn further. However, we
are not aware of any existing game which explicitly focuses on introducing ML
concepts to students using game play. Hence, in this paper, we propose
ML-Quest, a 3D video game to provide conceptual overview of three ML concepts:
Supervised Learning, Gradient Descent and K-Nearest Neighbor (KNN)
Classification. The crux of the game is to introduce the definition and working
of these concepts, which we call conceptual overview, in a simulated scenario
without overwhelming students with the intricacies of ML. The game has been
predominantly evaluated for its usefulness and player experience using the
Technology Acceptance Model (TAM) model with the help of 23 higher-secondary
school students. The survey result shows that around 70% of the participants
either agree or strongly agree that the ML-Quest is quite interactive and
useful in introducing them to ML concepts.

    

### [[2107.06207] Adaptive Machine Learning for Time-Varying Systems: Low Dimensional Latent Space Tuning](http://arxiv.org/abs/2107.06207)


  Machine learning (ML) tools such as encoder-decoder convolutional neural
networks (CNN) can represent incredibly complex nonlinear functions which map
between combinations of images and scalars. For example, CNNs can be used to
map combinations of accelerator parameters and images which are 2D projections
of the 6D phase space distributions of charged particle beams as they are
transported between various particle accelerator locations. Despite their
strengths, applying ML to time-varying systems, or systems with shifting
distributions, is an open problem, especially for large systems for which
collecting new data for re-training is impractical or interrupts operations.
Particle accelerators are one example of large time-varying systems for which
collecting detailed training data requires lengthy dedicated beam measurements
which may no longer be available during regular operations. We present a
recently developed method of adaptive ML for time-varying systems. Our approach
is to map very high (N>100k) dimensional inputs (a combination of scalar
parameters and images) into the low dimensional (N~2) latent space at the
output of the encoder section of an encoder-decoder CNN. We then actively tune
the low dimensional latent space-based representation of complex system
dynamics by the addition of an adaptively tuned feedback vector directly before
the decoder sections builds back up to our image-based high-dimensional phase
space density representations. This method allows us to learn correlations
within and to quickly tune the characteristics of incredibly high parameter
systems and to track their evolution in real time based on feedback without
massive new data sets for re-training.

    

### [[2107.06209] Learning a Discriminant Latent Space with Neural Discriminant Analysis](http://arxiv.org/abs/2107.06209)


  Discriminative features play an important role in image and object
classification and also in other fields of research such as semi-supervised
learning, fine-grained classification, out of distribution detection. Inspired
by Linear Discriminant Analysis (LDA), we propose an optimization called Neural
Discriminant Analysis (NDA) for Deep Convolutional Neural Networks (DCNNs). NDA
transforms deep features to become more discriminative and, therefore, improves
the performances in various tasks. Our proposed optimization has two primary
goals for inter- and intra-class variances. The first one is to minimize
variances within each individual class. The second goal is to maximize pairwise
distances between features coming from different classes. We evaluate our NDA
optimization in different research fields: general supervised classification,
fine-grained classification, semi-supervised learning, and out of distribution
detection. We achieve performance improvements in all the fields compared to
baseline methods that do not use NDA. Besides, using NDA, we also surpass the
state of the art on the four tasks on various testing datasets.

    

### [[2107.06212] 'CADSketchNet' -- An Annotated Sketch dataset for 3D CAD Model Retrieval with Deep Neural Networks](http://arxiv.org/abs/2107.06212)


  Ongoing advancements in the fields of 3D modelling and digital archiving have
led to an outburst in the amount of data stored digitally. Consequently,
several retrieval systems have been developed depending on the type of data
stored in these databases. However, unlike text data or images, performing a
search for 3D models is non-trivial. Among 3D models, retrieving 3D
Engineering/CAD models or mechanical components is even more challenging due to
the presence of holes, volumetric features, presence of sharp edges etc., which
make CAD a domain unto itself. The research work presented in this paper aims
at developing a dataset suitable for building a retrieval system for 3D CAD
models based on deep learning. 3D CAD models from the available CAD databases
are collected, and a dataset of computer-generated sketch data, termed
'CADSketchNet', has been prepared. Additionally, hand-drawn sketches of the
components are also added to CADSketchNet. Using the sketch images from this
dataset, the paper also aims at evaluating the performance of various retrieval
system or a search engine for 3D CAD models that accepts a sketch image as the
input query. Many experimental models are constructed and tested on
CADSketchNet. These experiments, along with the model architecture, choice of
similarity metrics are reported along with the search results.

    

### [[2107.06217] What classifiers know what they don't?](http://arxiv.org/abs/2107.06217)


  Being uncertain when facing the unknown is key to intelligent decision
making. However, machine learning algorithms lack reliable estimates about
their predictive uncertainty. This leads to wrong and overly-confident
decisions when encountering classes unseen during training. Despite the
importance of equipping classifiers with uncertainty estimates ready for the
real world, prior work has focused on small datasets and little or no class
discrepancy between training and testing data. To close this gap, we introduce
UIMNET: a realistic, ImageNet-scale test-bed to evaluate predictive uncertainty
estimates for deep image classifiers. Our benchmark provides implementations of
eight state-of-the-art algorithms, six uncertainty measures, four in-domain
metrics, three out-domain metrics, and a fully automated pipeline to train,
calibrate, ensemble, select, and evaluate models. Our test-bed is open-source
and all of our results are reproducible from a fixed commit in our repository.
Adding new datasets, algorithms, measures, or metrics is a matter of a few
lines of code-in so hoping that UIMNET becomes a stepping stone towards
realistic, rigorous, and reproducible research in uncertainty estimation. Our
results show that ensembles of ERM classifiers as well as single MIMO
classifiers are the two best alternatives currently available to measure
uncertainty about both in-domain and out-domain classes.

    

### [[2107.06219] Domain-Irrelevant Representation Learning for Unsupervised Domain Generalization](http://arxiv.org/abs/2107.06219)


  Domain generalization (DG) aims to help models trained on a set of source
domains generalize better on unseen target domains. The performances of current
DG methods largely rely on sufficient labeled data, which however are usually
costly or unavailable. While unlabeled data are far more accessible, we seek to
explore how unsupervised learning can help deep models generalizes across
domains. Specifically, we study a novel generalization problem called
unsupervised domain generalization, which aims to learn generalizable models
with unlabeled data. Furthermore, we propose a Domain-Irrelevant Unsupervised
Learning (DIUL) method to cope with the significant and misleading
heterogeneity within unlabeled data and severe distribution shifts between
source and target data. Surprisingly we observe that DIUL can not only
counterbalance the scarcity of labeled data but also further strengthen the
generalization ability of models when the labeled data are sufficient. As a
pretraining approach, DIUL shows superior to ImageNet pretraining protocol even
when the available data are unlabeled and of a greatly smaller amount compared
to ImageNet. Extensive experiments clearly demonstrate the effectiveness of our
method compared with state-of-the-art unsupervised learning counterparts.

    

### [[2107.06226] Pessimistic Model-based Offline RL: PAC Bounds and Posterior Sampling under Partial Coverage](http://arxiv.org/abs/2107.06226)


  We study model-based offline Reinforcement Learning with general function
approximation. We present an algorithm named Constrained Pessimistic Policy
Optimization (CPPO) which leverages a general function class and uses a
constraint to encode pessimism. Under the assumption that the ground truth
model belongs to our function class, CPPO can learn with the offline data only
providing partial coverage, i.e., it can learn a policy that completes against
any policy that is covered by the offline data, in polynomial sample complexity
with respect to the statistical complexity of the function class. We then
demonstrate that this algorithmic framework can be applied to many specialized
Markov Decision Processes where the additional structural assumptions can
further refine the concept of partial coverage. One notable example is low-rank
MDP with representation learning where the partial coverage is defined using
the concept of relative condition number measured by the underlying unknown
ground truth feature representation. Finally, we introduce and study the
Bayesian setting in offline RL. The key benefit of Bayesian offline RL is that
algorithmically, we do not need to explicitly construct pessimism or reward
penalty which could be hard beyond models with linear structures. We present a
posterior sampling-based incremental policy optimization algorithm (PS-PO)
which proceeds by iteratively sampling a model from the posterior distribution
and performing one-step incremental policy optimization inside the sampled
model. Theoretically, in expectation with respect to the prior distribution,
PS-PO can learn a near optimal policy under partial coverage with polynomial
sample complexity.

    

### [[2107.06231] Timbre Classification of Musical Instruments with a Deep Learning Multi-Head Attention-Based Model](http://arxiv.org/abs/2107.06231)


  The aim of this work is to define a model based on deep learning that is able
to identify different instrument timbres with as few parameters as possible.
For this purpose, we have worked with classical orchestral instruments played
with different dynamics, which are part of a few instrument families and which
play notes in the same pitch range. It has been possible to assess the ability
to classify instruments by timbre even if the instruments are playing the same
note with the same intensity. The network employed uses a multi-head attention
mechanism, with 8 heads and a dense network at the output taking as input the
log-mel magnitude spectrograms of the sound samples. This network allows the
identification of 20 instrument classes of the classical orchestra, achieving
an overall F$_1$ value of 0.62. An analysis of the weights of the attention
layer has been performed and the confusion matrix of the model is presented,
allowing us to assess the ability of the proposed architecture to distinguish
timbre and to establish the aspects on which future work should focus.

    

### [[2107.06239] Everybody Is Unique: Towards Unbiased Human Mesh Recovery](http://arxiv.org/abs/2107.06239)


  We consider the problem of obese human mesh recovery, i.e., fitting a
parametric human mesh to images of obese people. Despite obese person mesh
fitting being an important problem with numerous applications (e.g.,
healthcare), much recent progress in mesh recovery has been restricted to
images of non-obese people. In this work, we identify this crucial gap in the
current literature by presenting and discussing limitations of existing
algorithms. Next, we present a simple baseline to address this problem that is
scalable and can be easily used in conjunction with existing algorithms to
improve their performance. Finally, we present a generalized human mesh
optimization algorithm that substantially improves the performance of existing
methods on both obese person images as well as community-standard benchmark
datasets. A key innovation of this technique is that it does not rely on
supervision from expensive-to-create mesh parameters. Instead, starting from
widely and cheaply available 2D keypoints annotations, our method automatically
generates mesh parameters that can in turn be used to re-train and fine-tune
any existing mesh estimation algorithm. This way, we show our method acts as a
drop-in to improve the performance of a wide variety of contemporary mesh
estimation methods. We conduct extensive experiments on multiple datasets
comprising both standard and obese person images and demonstrate the efficacy
of our proposed techniques.

    

### [[2107.06256] Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval](http://arxiv.org/abs/2107.06256)


  We present Retrieve in Style (RIS), an unsupervised framework for
fine-grained facial feature transfer and retrieval on real images. Recent work
shows that it is possible to learn a catalog that allows local semantic
transfers of facial features on generated images by capitalizing on the
disentanglement property of the StyleGAN latent space. RIS improves existing
art on: 1) feature disentanglement and allows for challenging transfers (i.e.,
hair and pose) that were not shown possible in SoTA methods. 2) eliminating the
need for per-image hyperparameter tuning, and for computing a catalog over a
large batch of images. 3) enabling face retrieval using the proposed facial
features (e.g., eyes), and to our best knowledge, is the first work to retrieve
face images at the fine-grained level. 4) robustness and natural application to
real images. Our qualitative and quantitative analyses show RIS achieves both
high-fidelity feature transfers and accurate fine-grained retrievals on real
images. We discuss the responsible application of RIS.

    

### [[2107.06257] Object Tracking and Geo-localization from Street Images](http://arxiv.org/abs/2107.06257)


  Geo-localizing static objects from street images is challenging but also very
important for road asset mapping and autonomous driving. In this paper we
present a two-stage framework that detects and geolocalizes traffic signs from
low frame rate street videos. Our proposed system uses a modified version of
RetinaNet (GPS-RetinaNet), which predicts a positional offset for each sign
relative to the camera, in addition to performing the standard classification
and bounding box regression. Candidate sign detections from GPS-RetinaNet are
condensed into geolocalized signs by our custom tracker, which consists of a
learned metric network and a variant of the Hungarian Algorithm. Our metric
network estimates the similarity between pairs of detections, then the
Hungarian Algorithm matches detections across images using the similarity
scores provided by the metric network. Our models were trained using an updated
version of the ARTS dataset, which contains 25,544 images and 47.589 sign
annotations ~\cite{arts}. The proposed dataset covers a diverse set of
environments gathered from a broad selection of roads. Each annotaiton contains
a sign class label, its geospatial location, an assembly label, a side of road
indicator, and unique identifiers that aid in the evaluation. This dataset will
support future progress in the field, and the proposed system demonstrates how
to take advantage of some of the unique characteristics of a realistic
geolocalization dataset.

    

### [[2107.06264] Parameterization of Forced Isotropic Turbulent Flow using Autoencoders and Generative Adversarial Networks](http://arxiv.org/abs/2107.06264)


  Autoencoders and generative neural network models have recently gained
popularity in fluid mechanics due to their spontaneity and low processing time
instead of high fidelity CFD simulations. Auto encoders are used as model order
reduction tools in applications of fluid mechanics by compressing input
high-dimensional data using an encoder to map the input space into a
lower-dimensional latent space. Whereas, generative models such as Variational
Auto-encoders (VAEs) and Generative Adversarial Networks (GANs) are proving to
be effective in generating solutions to chaotic models with high 'randomness'
such as turbulent flows. In this study, forced isotropic turbulence flow is
generated by parameterizing into some basic statistical characteristics. The
models trained on pre-simulated data from dependencies on these characteristics
and the flow generation is then affected by varying these parameters. The
latent vectors pushed along the generator models like the decoders and
generators contain independent entries which can be used to create different
outputs with similar properties. The use of neural network-based architecture
removes the need for dependency on the classical mesh-based Navier-Stoke
equation estimation which is prominent in many CFD softwares.

    

### [[2107.06268] Smoothed Bernstein Online Aggregation for Day-Ahead Electricity Demand Forecasting](http://arxiv.org/abs/2107.06268)


  We present a winning method of the IEEE DataPort Competition on Day-Ahead
Electricity Demand Forecasting: Post-COVID Paradigm. The day-ahead load
forecasting approach is based on online forecast combination of multiple point
prediction models. It contains four steps: i) data cleaning and preprocessing,
ii) a holiday adjustment procedure, iii) training of individual forecasting
models, iv) forecast combination by smoothed Bernstein Online Aggregation
(BOA). The approach is flexible and can quickly adopt to new energy system
situations as they occurred during and after COVID-19 shutdowns. The pool of
individual prediction models ranges from rather simple time series models to
sophisticated models like generalized additive models (GAMs) and
high-dimensional linear models estimated by lasso. They incorporate
autoregressive, calendar and weather effects efficiently. All steps contain
novel concepts that contribute to the excellent forecasting performance of the
proposed method. This holds particularly for the holiday adjustment procedure
and the fully adaptive smoothed BOA approach.

    

### [[2107.06277] Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability](http://arxiv.org/abs/2107.06277)


  Generalization is a central challenge for the deployment of reinforcement
learning (RL) systems in the real world. In this paper, we show that the
sequential structure of the RL problem necessitates new approaches to
generalization beyond the well-studied techniques used in supervised learning.
While supervised learning methods can generalize effectively without explicitly
accounting for epistemic uncertainty, we show that, perhaps surprisingly, this
is not the case in RL. We show that generalization to unseen test conditions
from a limited number of training conditions induces implicit partial
observability, effectively turning even fully-observed MDPs into POMDPs.
Informed by this observation, we recast the problem of generalization in RL as
solving the induced partially observed Markov decision process, which we call
the epistemic POMDP. We demonstrate the failure modes of algorithms that do not
appropriately handle this partial observability, and suggest a simple
ensemble-based technique for approximately solving the partially observed
problem. Empirically, we demonstrate that our simple algorithm derived from the
epistemic POMDP achieves significant gains in generalization over current
methods on the Procgen benchmark suite.

    

### [[1904.08962] Constrained Restless Bandits for Dynamic Scheduling in Cyber-Physical Systems](http://arxiv.org/abs/1904.08962)


  This paper studies a class of constrained restless multi-armed bandits
(CRMAB). The constraints are in the form of time varying set of actions (set of
available arms). This variation can be either stochastic or semi-deterministic.
Given a set of arms, a fixed number of them can be chosen to be played in each
decision interval. The play of each arm yields a state dependent reward. The
current states of arms are partially observable through binary feedback signals
from arms that are played. The current availability of arms is fully
observable. The objective is to maximize long term cumulative reward. The
uncertainty about future availability of arms along with partial state
information makes this objective challenging. Applications for CRMAB abound in
the domain of cyber-physical systems. First, this optimization problem is
analyzed using Whittle's index policy. To this end, a constrained restless
single-armed bandit is studied. It is shown to admit a threshold-type optimal
policy and is also indexable. An algorithm to compute Whittle's index is
presented. An alternate solution method with lower complexity is also presented
in the form of an online rollout policy. Further, upper bounds on the value
function are derived in order to estimate the degree of sub-optimality of
various solutions. The simulation study compares the performance of Whittle's
index, online rollout, myopic and modified Whittle's index policies.

    

### [[1911.00773] Design and Challenges of Cloze-Style Reading Comprehension Tasks on Multiparty Dialogue](http://arxiv.org/abs/1911.00773)


  This paper analyzes challenges in cloze-style reading comprehension on
multiparty dialogue and suggests two new tasks for more comprehensive
predictions of personal entities in daily conversations. We first demonstrate
that there are substantial limitations to the evaluation methods of previous
work, namely that randomized assignment of samples to training and test data
substantially decreases the complexity of cloze-style reading comprehension.
According to our analysis, replacing the random data split with a chronological
data split reduces test accuracy on previous single-variable passage completion
task from 72\% to 34\%, that leaves much more room to improve. Our proposed
tasks extend the previous single-variable passage completion task by replacing
more character mentions with variables. Several deep learning models are
developed to validate these three tasks. A thorough error analysis is provided
to understand the challenges and guide the future direction of this research.

    

### [[1911.07936] Privacy Preserving Gaze Estimation using Synthetic Images via a Randomized Encoding Based Framework](http://arxiv.org/abs/1911.07936)


  Eye tracking is handled as one of the key technologies for applications that
assess and evaluate human attention, behavior, and biometrics, especially using
gaze, pupillary, and blink behaviors. One of the challenges with regard to the
social acceptance of eye tracking technology is however the preserving of
sensitive and personal information. To tackle this challenge, we employ a
privacy-preserving framework based on randomized encoding to train a Support
Vector Regression model using synthetic eye images privately to estimate the
human gaze. During the computation, none of the parties learn about the data or
the result that any other party has. Furthermore, the party that trains the
model cannot reconstruct pupil, blinks or visual scanpath. The experimental
results show that our privacy-preserving framework is capable of working in
real-time, with the same accuracy as compared to non-private version and could
be extended to other eye tracking related problems.

    

### [[2002.01129] Bayesian Meta-Prior Learning Using Empirical Bayes](http://arxiv.org/abs/2002.01129)


  Adding domain knowledge to a learning system is known to improve results. In
multi-parameter Bayesian frameworks, such knowledge is incorporated as a prior.
On the other hand, various model parameters can have different learning rates
in real-world problems, especially with skewed data. Two often-faced challenges
in Operation Management and Management Science applications are the absence of
informative priors, and the inability to control parameter learning rates. In
this study, we propose a hierarchical Empirical Bayes approach that addresses
both challenges, and that can generalize to any Bayesian framework. Our method
learns empirical meta-priors from the data itself and uses them to decouple the
learning rates of first-order and second-order features (or any other given
feature grouping) in a Generalized Linear Model. As the first-order features
are likely to have a more pronounced effect on the outcome, focusing on
learning first-order weights first is likely to improve performance and
convergence time. Our Empirical Bayes method clamps features in each group
together and uses the deployed model's observed data to empirically compute a
hierarchical prior in hindsight. We report theoretical results for the
unbiasedness, strong consistency, and optimal frequentist cumulative regret
properties of our meta-prior variance estimator. We apply our method to a
standard supervised learning optimization problem, as well as an online
combinatorial optimization problem in a contextual bandit setting implemented
in an Amazon production system. Both during simulations and live experiments,
our method shows marked improvements, especially in cases of small traffic. Our
findings are promising, as optimizing over sparse data is often a challenge.

    

### [[2002.10516] Modeling Continuous Stochastic Processes with Dynamic Normalizing Flows](http://arxiv.org/abs/2002.10516)


  Normalizing flows transform a simple base distribution into a complex target
distribution and have proved to be powerful models for data generation and
density estimation. In this work, we propose a novel type of normalizing flow
driven by a differential deformation of the Wiener process. As a result, we
obtain a rich time series model whose observable process inherits many of the
appealing properties of its base process, such as efficient computation of
likelihoods and marginals. Furthermore, our continuous treatment provides a
natural framework for irregular time series with an independent arrival
process, including straightforward interpolation. We illustrate the desirable
properties of the proposed model on popular stochastic processes and
demonstrate its superior flexibility to variational RNN and latent ODE
baselines in a series of experiments on synthetic and real-world data.

    

### [[2003.08579] Adaptive Batching for Gaussian Process Surrogates with Application in Noisy Level Set Estimation](http://arxiv.org/abs/2003.08579)


  We develop adaptive replicated designs for Gaussian process metamodels of
stochastic experiments. Adaptive batching is a natural extension of sequential
design heuristics with the benefit of replication growing as response features
are learned, inputs concentrate, and the metamodeling overhead rises. Motivated
by the problem of learning the level set of the mean simulator response we
develop four novel schemes: Multi-Level Batching (MLB), Ratchet Batching (RB),
Adaptive Batched Stepwise Uncertainty Reduction (ABSUR), Adaptive Design with
Stepwise Allocation (ADSA) and Deterministic Design with Stepwise Allocation
(DDSA). Our algorithms simultaneously (MLB, RB and ABSUR) or sequentially (ADSA
and DDSA) determine the sequential design inputs and the respective number of
replicates. Illustrations using synthetic examples and an application in
quantitative finance (Bermudan option pricing via Regression Monte Carlo) show
that adaptive batching brings significant computational speed-ups with minimal
loss of modeling fidelity.

    

### [[2007.06631] T-Basis: a Compact Representation for Neural Networks](http://arxiv.org/abs/2007.06631)


  We introduce T-Basis, a novel concept for a compact representation of a set
of tensors, each of an arbitrary shape, which is often seen in Neural Networks.
Each of the tensors in the set is modeled using Tensor Rings, though the
concept applies to other Tensor Networks. Owing its name to the T-shape of
nodes in diagram notation of Tensor Rings, T-Basis is simply a list of equally
shaped three-dimensional tensors, used to represent Tensor Ring nodes. Such
representation allows us to parameterize the tensor set with a small number of
parameters (coefficients of the T-Basis tensors), scaling logarithmically with
each tensor's size in the set and linearly with the dimensionality of T-Basis.
We evaluate the proposed approach on the task of neural network compression and
demonstrate that it reaches high compression rates at acceptable performance
drops. Finally, we analyze memory and operation requirements of the compressed
networks and conclude that T-Basis networks are equally well suited for
training and inference in resource-constrained environments and usage on the
edge devices.

    

### [[2007.11117] Interpretable Anomaly Detection with DIFFI: Depth-based Isolation Forest Feature Importance](http://arxiv.org/abs/2007.11117)


  Anomaly Detection is an unsupervised learning task aimed at detecting
anomalous behaviours with respect to historical data. In particular,
multivariate Anomaly Detection has an important role in many applications
thanks to the capability of summarizing the status of a complex system or
observed phenomenon with a single indicator (typically called `Anomaly Score')
and thanks to the unsupervised nature of the task that does not require human
tagging. The Isolation Forest is one of the most commonly adopted algorithms in
the field of Anomaly Detection, due to its proven effectiveness and low
computational complexity. A major problem affecting Isolation Forest is
represented by the lack of interpretability, an effect of the inherent
randomness governing the splits performed by the Isolation Trees, the building
blocks of the Isolation Forest. In this paper we propose effective, yet
computationally inexpensive, methods to define feature importance scores at
both global and local level for the Isolation Forest. Moreover, we define a
procedure to perform unsupervised feature selection for Anomaly Detection
problems based on our interpretability method; such procedure also serves the
purpose of tackling the challenging task of feature importance evaluation in
unsupervised anomaly detection. We assess the performance on several synthetic
and real-world datasets, including comparisons against state-of-the-art
interpretability techniques, and make the code publicly available to enhance
reproducibility and foster research in the field.

    

### [[2008.01724] Convex and Nonconvex Optimization Are Both Minimax-Optimal for Noisy Blind Deconvolution under Random Designs](http://arxiv.org/abs/2008.01724)


  We investigate the effectiveness of convex relaxation and nonconvex
optimization in solving bilinear systems of equations under two different
designs (i.e.$~$a sort of random Fourier design and Gaussian design). Despite
the wide applicability, the theoretical understanding about these two paradigms
remains largely inadequate in the presence of random noise. The current paper
makes two contributions by demonstrating that: (1) a two-stage nonconvex
algorithm attains minimax-optimal accuracy within a logarithmic number of
iterations. (2) convex relaxation also achieves minimax-optimal statistical
accuracy vis-à-vis random noise. Both results significantly improve upon the
state-of-the-art theoretical guarantees.

    

### [[2008.02627] Notes on the Behavior of MC Dropout](http://arxiv.org/abs/2008.02627)


  Among the various options to estimate uncertainty in deep neural networks,
Monte-Carlo dropout is widely popular for its simplicity and effectiveness.
However the quality of the uncertainty estimated through this method varies and
choices in architecture design and in training procedures have to be carefully
considered and tested to obtain satisfactory results. In this paper we present
a study offering a different point of view on the behavior of Monte-Carlo
dropout, which enables us to observe a few interesting properties of the
technique to keep in mind when considering its use for uncertainty estimation.

    

### [[2009.03871] Intraoperative Liver Surface Completion with Graph Convolutional VAE](http://arxiv.org/abs/2009.03871)


  In this work we propose a method based on geometric deep learning to predict
the complete surface of the liver, given a partial point cloud of the organ
obtained during the surgical laparoscopic procedure. We introduce a new data
augmentation technique that randomly perturbs shapes in their frequency domain
to compensate the limited size of our dataset. The core of our method is a
variational autoencoder (VAE) that is trained to learn a latent space for
complete shapes of the liver. At inference time, the generative part of the
model is embedded in an optimisation procedure where the latent representation
is iteratively updated to generate a model that matches the intraoperative
partial point cloud. The effect of this optimisation is a progressive non-rigid
deformation of the initially generated shape. Our method is qualitatively
evaluated on real data and quantitatively evaluated on synthetic data. We
compared with a state-of-the-art rigid registration algorithm, that our method
outperformed in visible areas.

    

### [[2009.06117] The Platform Design Problem](http://arxiv.org/abs/2009.06117)


  On-line firms deploy suites of software platforms, where each platform is
designed to interact with users during a certain activity, such as browsing,
chatting, socializing, emailing, driving, etc. The economic and incentive
structure of this exchange, as well as its algorithmic nature, have not been
explored to our knowledge. We model this interaction as a Stackelberg game
between a Designer and one or more Agents. We model an Agent as a Markov chain
whose states are activities; we assume that the Agent's utility is a linear
function of the steady-state distribution of this chain. The Designer may
design a platform for each of these activities/states; if a platform is adopted
by the Agent, the transition probabilities of the Markov chain are affected,
and so is the objective of the Agent. The Designer's utility is a linear
function of the steady state probabilities of the accessible states minus the
development cost of the platforms. The underlying optimization problem of the
Agent -- how to choose the states for which to adopt the platform -- is an MDP.
If this MDP has a simple yet plausible structure (the transition probabilities
from one state to another only depend on the target state and the recurrent
probability of the current state) the Agent's problem can be solved by a greedy
algorithm. The Designer's optimization problem (designing a custom suite for
the Agent so as to optimize, through the Agent's optimum reaction, the
Designer's revenue), is NP-hard to approximate within any finite ratio;
however, the special case, while still NP-hard, has an FPTAS. These results
generalize from a single Agent to a distribution of Agents with finite support,
as well as to the setting where the Designer must find the best response to the
existing strategies of other Designers. We discuss other implications of our
results and directions of future research.

    

### [[2010.04891] Online Optimal Control with Affine Constraints](http://arxiv.org/abs/2010.04891)


  This paper considers online optimal control with affine constraints on the
states and actions under linear dynamics with bounded random disturbances. The
system dynamics and constraints are assumed to be known and time-invariant but
the convex stage cost functions change adversarially. To solve this problem, we
propose Online Gradient Descent with Buffer Zones (OGD-BZ). Theoretically, we
show that OGD-BZ with proper parameters can guarantee the system to satisfy all
the constraints despite any admissible disturbances. Further, we investigate
the policy regret of OGD-BZ, which compares OGD-BZ's performance with the
performance of the optimal linear policy in hindsight. We show that OGD-BZ can
achieve a policy regret upper bound that is the square root of the horizon
length multiplied by some logarithmic terms of the horizon length under proper
algorithm parameters.

    

### [[2010.12970] Deep Denoising For Scientific Discovery: A Case Study In Electron Microscopy](http://arxiv.org/abs/2010.12970)


  Denoising is a fundamental challenge in scientific imaging. Deep
convolutional neural networks (CNNs) provide the current state of the art in
denoising natural images, where they produce impressive results. However, their
potential has barely been explored in the context of scientific imaging.
Denoising CNNs are typically trained on real natural images artificially
corrupted with simulated noise. In contrast, in scientific applications,
noiseless ground-truth images are usually not available. To address this issue,
we propose a simulation-based denoising (SBD) framework, in which CNNs are
trained on simulated images. We test the framework on data obtained from
transmission electron microscopy (TEM), an imaging technique with widespread
applications in material science, biology, and medicine. SBD outperforms
existing techniques by a wide margin on a simulated benchmark dataset, as well
as on real data. Apart from the denoised images, SBD generates likelihood maps
to visualize the agreement between the structure of the denoised image and the
observed data. Our results reveal shortcomings of state-of-the-art denoising
architectures, such as their small field-of-view: substantially increasing the
field-of-view of the CNNs allows them to exploit non-local periodic patterns in
the data, which is crucial at high noise levels. In addition, we analyze the
generalization capability of SBD, demonstrating that the trained networks are
robust to variations of imaging parameters and of the underlying signal
structure. Finally, we release the first publicly available benchmark dataset
of TEM images, containing 18,000 examples.

    

### [[2010.13988] Toward Better Generalization Bounds with Locally Elastic Stability](http://arxiv.org/abs/2010.13988)


  Algorithmic stability is a key characteristic to ensure the generalization
ability of a learning algorithm. Among different notions of stability,
\emph{uniform stability} is arguably the most popular one, which yields
exponential generalization bounds. However, uniform stability only considers
the worst-case loss change (or so-called sensitivity) by removing a single data
point, which is distribution-independent and therefore undesirable. There are
many cases that the worst-case sensitivity of the loss is much larger than the
average sensitivity taken over the single data point that is removed,
especially in some advanced models such as random feature models or neural
networks. Many previous works try to mitigate the distribution independent
issue by proposing weaker notions of stability, however, they either only yield
polynomial bounds or the bounds derived do not vanish as sample size goes to
infinity. Given that, we propose \emph{locally elastic stability} as a weaker
and distribution-dependent stability notion, which still yields exponential
generalization bounds. We further demonstrate that locally elastic stability
implies tighter generalization bounds than those derived based on uniform
stability in many situations by revisiting the examples of bounded support
vector machines, regularized least square regressions, and stochastic gradient
descent.

    

### [[2010.16011] POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](http://arxiv.org/abs/2010.16011)


  In neural combinatorial optimization (CO), reinforcement learning (RL) can
turn a deep neural net into a fast, powerful heuristic solver of NP-hard
problems. This approach has a great potential in practical applications because
it allows near-optimal solutions to be found without expert guides armed with
substantial domain knowledge. We introduce Policy Optimization with Multiple
Optima (POMO), an end-to-end approach for building such a heuristic solver.
POMO is applicable to a wide range of CO problems. It is designed to exploit
the symmetries in the representation of a CO solution. POMO uses a modified
REINFORCE algorithm that forces diverse rollouts towards all optimal solutions.
Empirically, the low-variance baseline of POMO makes RL training fast and
stable, and it is more resistant to local minima compared to previous
approaches. We also introduce a new augmentation-based inference method, which
accompanies POMO nicely. We demonstrate the effectiveness of POMO by solving
three popular NP-hard problems, namely, traveling salesman (TSP), capacitated
vehicle routing (CVRP), and 0-1 knapsack (KP). For all three, our solver based
on POMO shows a significant improvement in performance over all recent learned
heuristics. In particular, we achieve the optimality gap of 0.14% with TSP100
while reducing inference time by more than an order of magnitude.

    

### [[2012.11619] Defence against adversarial attacks using classical and quantum-enhanced Boltzmann machines](http://arxiv.org/abs/2012.11619)


  We provide a robust defence to adversarial attacks on discriminative
algorithms. Neural networks are naturally vulnerable to small, tailored
perturbations in the input data that lead to wrong predictions. On the
contrary, generative models attempt to learn the distribution underlying a
dataset, making them inherently more robust to small perturbations. We use
Boltzmann machines for discrimination purposes as attack-resistant classifiers,
and compare them against standard state-of-the-art adversarial defences. We
find improvements ranging from 5% to 72% against attacks with Boltzmann
machines on the MNIST dataset. We furthermore complement the training with
quantum-enhanced sampling from the D-Wave 2000Q annealer, finding results
comparable with classical techniques and with marginal improvements in some
cases. These results underline the relevance of probabilistic methods in
constructing neural networks and highlight a novel scenario of practical
relevance where quantum computers, even with limited hardware capabilites,
could provide advantages over classical computers. This work is dedicated to
the memory of Peter Wittek.

    

### [[2101.00387] What all do audio transformer models hear? Probing Acoustic Representations for Language Delivery and its Structure](http://arxiv.org/abs/2101.00387)


  In recent times, BERT based transformer models have become an inseparable
part of the 'tech stack' of text processing models. Similar progress is being
observed in the speech domain with a multitude of models observing
state-of-the-art results by using audio transformer models to encode speech.
This begs the question of what are these audio transformer models learning.
Moreover, although the standard methodology is to choose the last layer
embedding for any downstream task, but is it the optimal choice? We try to
answer these questions for the two recent audio transformer models, Mockingjay
and wave2vec2.0. We compare them on a comprehensive set of language delivery
and structure features including audio, fluency and pronunciation features.
Additionally, we probe the audio models' understanding of textual surface,
syntax, and semantic features and compare them to BERT. We do this over
exhaustive settings for native, non-native, synthetic, read and spontaneous
speech datasets

    

### [[2101.10643] Casual Inference using Deep Bayesian Dynamic Survival Model (CDS)](http://arxiv.org/abs/2101.10643)


  Causal inference in longitudinal observational health data often requires the
accurate estimation of treatment effects on time-to-event outcomes in the
presence of time-varying covariates. To tackle this sequential treatment effect
estimation problem, we have developed a causal dynamic survival (CDS) model
that uses the potential outcomes framework with the recurrent sub-networks with
random seed ensembles to estimate the difference in survival curves of its
confidence interval. Using simulated survival datasets, the CDS model has shown
good causal effect estimation performance across scenarios of sample dimension,
event rate, confounding and overlapping. However, increasing the sample size is
not effective to alleviate the adverse impact from high level of confounding.
In two large clinical cohort studies, our model identified the expected
conditional average treatment effect and detected individual effect
heterogeneity over time and patient subgroups. CDS provides individualised
absolute treatment effect estimations to improve clinical decisions.

    

### [[2101.11046] Generalized Doubly Reparameterized Gradient Estimators](http://arxiv.org/abs/2101.11046)


  Efficient low-variance gradient estimation enabled by the reparameterization
trick (RT) has been essential to the success of variational autoencoders.
Doubly-reparameterized gradients (DReGs) improve on the RT for multi-sample
variational bounds by applying reparameterization a second time for an
additional reduction in variance. Here, we develop two generalizations of the
DReGs estimator and show that they can be used to train conditional and
hierarchical VAEs on image modelling tasks more effectively. First, we extend
the estimator to hierarchical models with several stochastic layers by showing
how to treat additional score function terms due to the hierarchical
variational posterior. We then generalize DReGs to score functions of arbitrary
distributions instead of just those of the sampling distribution, which makes
the estimator applicable to the parameters of the prior in addition to those of
the posterior.

    

### [[2102.00504] Exact Recovery of Clusters in Finite Metric Spaces Using Oracle Queries](http://arxiv.org/abs/2102.00504)


  We investigate the problem of exact cluster recovery using oracle queries.
Previous results show that clusters in Euclidean spaces that are convex and
separated with a margin can be reconstructed exactly using only $O(\log n)$
same-cluster queries, where $n$ is the number of input points. In this work, we
study this problem in the more challenging non-convex setting. We introduce a
structural characterization of clusters, called $(\beta,\gamma)$-convexity,
that can be applied to any finite set of points equipped with a metric (or even
a semimetric, as the triangle inequality is not needed). Using
$(\beta,\gamma)$-convexity, we can translate natural density properties of
clusters (which include, for instance, clusters that are strongly non-convex in
$\mathbb{R}^d$) into a graph-theoretic notion of convexity. By exploiting this
convexity notion, we design a deterministic algorithm that recovers
$(\beta,\gamma)$-convex clusters using $O(k^2 \log n + k^2
(6/\beta\gamma)^{dens(X)})$ same-cluster queries, where $k$ is the number of
clusters and $dens(X)$ is the density dimension of the semimetric. We show that
an exponential dependence on the density dimension is necessary, and we also
show that, if we are allowed to make $O(k^2 + k\log n)$ additional queries to a
"cluster separation" oracle, then we can recover clusters that have different
and arbitrary scales, even when the scale of each cluster is unknown.

    

### [[2102.03594] Online nonparametric regression with Sobolev kernels](http://arxiv.org/abs/2102.03594)


  In this work we investigate the variation of the online kernelized ridge
regression algorithm in the setting of $d-$dimensional adversarial
nonparametric regression. We derive the regret upper bounds on the classes of
Sobolev spaces $W_{p}^{\beta}(\mathcal{X})$, $p\geq 2, \beta>\frac{d}{p}$. The
upper bounds are supported by the minimax regret analysis, which reveals that
in the cases $\beta> \frac{d}{2}$ or $p=\infty$ these rates are (essentially)
optimal. Finally, we compare the performance of the kernelized ridge regression
forecaster to the known non-parametric forecasters in terms of the regret rates
and their computational complexity as well as to the excess risk rates in the
setting of statistical (i.i.d.) nonparametric regression.

    

### [[2102.05336] Understanding Instance-Level Label Noise: Disparate Impacts and Treatments](http://arxiv.org/abs/2102.05336)


  This paper aims to provide understandings for the effect of an
over-parameterized model, e.g. a deep neural network, memorizing
instance-dependent noisy labels. We first quantify the harms caused by
memorizing noisy instances, and show the disparate impacts of noisy labels for
sample instances with different representation frequencies. We then analyze how
several popular solutions for learning with noisy labels mitigate this harm at
the instance level. Our analysis reveals that existing approaches lead to
disparate treatments when handling noisy instances. While higher-frequency
instances often enjoy a high probability of an improvement by applying these
solutions, lower-frequency instances do not. Our analysis reveals new
understandings for when these approaches work, and provides theoretical
justifications for previously reported empirical observations. This observation
requires us to rethink the distribution of label noise across instances and
calls for different treatments for instances in different regimes.

    

### [[2102.05347] From Sampling to Optimization on Discrete Domains withApplications to Determinant Maximization](http://arxiv.org/abs/2102.05347)


  We show a connection between sampling and optimization on discrete domains.
For a family of distributions $\mu$ defined on size $k$ subsets of a ground set
of elements that is closed under external fields, we show that rapid mixing of
natural local random walks implies the existence of simple approximation
algorithms to find $\max \mu(\cdot)$. More precisely we show that if
(multi-step) down-up random walks have spectral gap at least inverse
polynomially large in $k$, then (multi-step) local search can find $\max
\mu(\cdot)$ within a factor of $k^{O(k)}$. As the main application of our
result, we show a simple nearly-optimal $k^{O(k)}$-factor approximation
algorithm for MAP inference on nonsymmetric DPPs. This is the first nontrivial
multiplicative approximation for finding the largest size $k$ principal minor
of a square (not-necessarily-symmetric) matrix $L$ with $L+L^\intercal\succeq
0$.
We establish the connection between sampling and optimization by showing that
an exchange inequality, a concept rooted in discrete convex analysis, can be
derived from fast mixing of local random walks. We further connect exchange
inequalities with composable core-sets for optimization, generalizing recent
results on composable core-sets for DPP maximization to arbitrary distributions
that satisfy either the strongly Rayleigh property or that have a log-concave
generating polynomial.

    

### [[2102.07868] GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning](http://arxiv.org/abs/2102.07868)


  Gaussian processes (GPs) are non-parametric, flexible, models that work well
in many tasks. Combining GPs with deep learning methods via deep kernel
learning (DKL) is especially compelling due to the strong representational
power induced by the network. However, inference in GPs, whether with or
without DKL, can be computationally challenging on large datasets. Here, we
propose GP-Tree, a novel method for multi-class classification with Gaussian
processes and DKL. We develop a tree-based hierarchical model in which each
internal node of the tree fits a GP to the data using the Pólya Gamma
augmentation scheme. As a result, our method scales well with both the number
of classes and data size. We demonstrate the effectiveness of our method
against other Gaussian process training baselines, and we show how our general
GP approach achieves improved accuracy on standard incremental few-shot
learning benchmarks.

    

### [[2102.09351] A Comprehensive Review of Deep Learning-based Single Image Super-resolution](http://arxiv.org/abs/2102.09351)


  Image super-resolution (SR) is one of the vital image processing methods that
improve the resolution of an image in the field of computer vision. In the last
two decades, significant progress has been made in the field of
super-resolution, especially by utilizing deep learning methods. This survey is
an effort to provide a detailed survey of recent progress in single-image
super-resolution in the perspective of deep learning while also informing about
the initial classical methods used for image super-resolution. The survey
classifies the image SR methods into four categories, i.e., classical methods,
supervised learning-based methods, unsupervised learning-based methods, and
domain-specific SR methods. We also introduce the problem of SR to provide
intuition about image quality metrics, available reference datasets, and SR
challenges. Deep learning-based approaches of SR are evaluated using a
reference dataset. Some of the reviewed state-of-the-art image SR methods
include the enhanced deep SR network (EDSR), cycle-in-cycle GAN (CinCGAN),
multiscale residual network (MSRN), meta residual dense network (Meta-RDN),
recurrent back-projection network (RBPN), second-order attention network (SAN),
SR feedback network (SRFBN) and the wavelet-based residual attention network
(WRAN). Finally, this survey is concluded with future directions and trends in
SR and open problems in SR to be addressed by the researchers.

    

### [[2102.09907] Instrumental Variable Value Iteration for Causal Offline Reinforcement Learning](http://arxiv.org/abs/2102.09907)


  In offline reinforcement learning (RL) an optimal policy is learnt solely
from a priori collected observational data. However, in observational data,
actions are often confounded by unobserved variables. Instrumental variables
(IVs), in the context of RL, are the variables whose influence on the state
variables are all mediated through the action. When a valid instrument is
present, we can recover the confounded transition dynamics through
observational data. We study a confounded Markov decision process where the
transition dynamics admit an additive nonlinear functional form. Using IVs, we
derive a conditional moment restriction (CMR) through which we can identify
transition dynamics based on observational data. We propose a provably
efficient IV-aided Value Iteration (IVVI) algorithm based on a primal-dual
reformulation of CMR. To the best of our knowledge, this is the first provably
efficient algorithm for instrument-aided offline RL.

    

### [[2102.10264] On Proximal Policy Optimization's Heavy-tailed Gradients](http://arxiv.org/abs/2102.10264)


  Modern policy gradient algorithms such as Proximal Policy Optimization (PPO)
rely on an arsenal of heuristics, including loss clipping and gradient
clipping, to ensure successful learning. These heuristics are reminiscent of
techniques from robust statistics, commonly used for estimation in outlier-rich
(``heavy-tailed'') regimes. In this paper, we present a detailed empirical
study to characterize the heavy-tailed nature of the gradients of the PPO
surrogate reward function. We demonstrate that the gradients, especially for
the actor network, exhibit pronounced heavy-tailedness and that it increases as
the agent's policy diverges from the behavioral policy (i.e., as the agent goes
further off policy). Further examination implicates the likelihood ratios and
advantages in the surrogate reward as the main sources of the observed
heavy-tailedness. We then highlight issues arising due to the heavy-tailed
nature of the gradients. In this light, we study the effects of the standard
PPO clipping heuristics, demonstrating that these tricks primarily serve to
offset heavy-tailedness in gradients. Thus motivated, we propose incorporating
GMOM, a high-dimensional robust estimator, into PPO as a substitute for three
clipping tricks. Despite requiring less hyperparameter tuning, our method
matches the performance of PPO (with all heuristics enabled) on a battery of
MuJoCo continuous control tasks.

    

### [[2102.11436] Model-Based Domain Generalization](http://arxiv.org/abs/2102.11436)


  Despite remarkable success in a variety of applications, it is well-known
that deep learning can fail catastrophically when presented with
out-of-distribution data. Toward addressing this challenge, we consider the
domain generalization problem, wherein predictors are trained using data drawn
from a family of related training domains and then evaluated on a distinct and
unseen test domain. We show that under a natural model of data generation and a
concomitant invariance condition, the domain generalization problem is
equivalent to an infinite-dimensional constrained statistical learning problem;
this problem forms the basis of our approach, which we call Model-Based Domain
Generalization. Due to the inherent challenges in solving constrained
optimization problems in deep learning, we exploit nonconvex duality theory to
develop unconstrained relaxations of this statistical problem with tight bounds
on the duality gap. Based on this theoretical motivation, we propose a novel
domain generalization algorithm with convergence guarantees. In our
experiments, we report improvements of up to 30 percentage points over
state-of-the-art domain generalization baselines on several benchmarks
including ColoredMNIST, Camelyon17-WILDS, FMoW-WILDS, and PACS.

    

### [[2102.13620] Towards Robust and Reliable Algorithmic Recourse](http://arxiv.org/abs/2102.13620)


  As predictive models are increasingly being deployed in high-stakes decision
making (e.g., loan approvals), there has been growing interest in post hoc
techniques which provide recourse to affected individuals. These techniques
generate recourses under the assumption that the underlying predictive model
does not change. However, in practice, models are often regularly updated for a
variety of reasons (e.g., dataset shifts), thereby rendering previously
prescribed recourses ineffective. To address this problem, we propose a novel
framework, RObust Algorithmic Recourse (ROAR), that leverages adversarial
training for finding recourses that are robust to model shifts. To the best of
our knowledge, this work proposes the first solution to this critical problem.
We also carry out detailed theoretical analysis which underscores the
importance of constructing recourses that are robust to model shifts: 1) we
derive a lower bound on the probability of invalidation of recourses generated
by existing approaches which are not robust to model shifts. 2) we prove that
the additional cost incurred due to the robust recourses output by our
framework is bounded. Experimental evaluation on multiple synthetic and
real-world datasets demonstrates the efficacy of the proposed framework and
supports our theoretical findings.

    

### [[2103.01931] Categorical Foundations of Gradient-Based Learning](http://arxiv.org/abs/2103.01931)


  We propose a categorical semantics of gradient-based machine learning
algorithms in terms of lenses, parametrised maps, and reverse derivative
categories. This foundation provides a powerful explanatory and unifying
framework: it encompasses a variety of gradient descent algorithms such as
ADAM, AdaGrad, and Nesterov momentum, as well as a variety of loss functions
such as as MSE and Softmax cross-entropy, shedding new light on their
similarities and differences. Our approach to gradient-based learning has
examples generalising beyond the familiar continuous domains (modelled in
categories of smooth maps) and can be realized in the discrete setting of
boolean circuits. Finally, we demonstrate the practical significance of our
framework with an implementation in Python.

    

### [[2103.03097] Generalizing to Unseen Domains: A Survey on Domain Generalization](http://arxiv.org/abs/2103.03097)


  Machine learning systems generally assume that the training and testing
distributions are the same. To this end, a key requirement is to develop models
that can generalize to unseen distributions. Domain generalization (DG), i.e.,
out-of-distribution generalization, has attracted increasing interests in
recent years. Domain generalization deals with a challenging setting where one
or several different but related domain(s) are given, and the goal is to learn
a model that can generalize to an unseen test domain. Great progress has been
made in the area of domain generalization for years. This paper presents the
first review of recent advances in this area. First, we provide a formal
definition of domain generalization and discuss several related fields. We then
thoroughly review the theories related to domain generalization and carefully
analyze the theory behind generalization. We categorize recent algorithms into
three classes: data manipulation, representation learning, and learning
strategy, and present several popular algorithms in detail for each category.
Third, we introduce the commonly used datasets and applications. Finally, we
summarize existing literature and present some potential research topics for
the future.

    

### [[2103.09327] SoWaF: Shuffling of Weights and Feature Maps: A Novel Hardware Intrinsic Attack (HIA) on Convolutional Neural Network (CNN)](http://arxiv.org/abs/2103.09327)


  Security of inference phase deployment of Convolutional neural network (CNN)
into resource constrained embedded systems (e.g. low end FPGAs) is a growing
research area. Using secure practices, third party FPGA designers can be
provided with no knowledge of initial and final classification layers. In this
work, we demonstrate that hardware intrinsic attack (HIA) in such a "secure"
design is still possible. Proposed HIA is inserted inside mathematical
operations of individual layers of CNN, which propagates erroneous operations
in all the subsequent CNN layers that lead to misclassification. The attack is
non-periodic and completely random, hence it becomes difficult to detect. Five
different attack scenarios with respect to each CNN layer are designed and
evaluated based on the overhead resources and the rate of triggering in
comparison to the original implementation. Our results for two CNN
architectures show that in all the attack scenarios, additional latency is
negligible (<0.61%), increment in DSP, LUT, FF is also less than 2.36%. Three
attack scenarios do not require any additional BRAM resources, while in two
scenarios BRAM increases, which compensates with the corresponding decrease in
FF and LUTs. To the authors' best knowledge this work is the first to address
the hardware intrinsic CNN attack with the attacker does not have knowledge of
the full CNN.

    

### [[2103.09791] An Overflow/Underflow-Free Fixed-Point Bit-Width Optimization Method for OS-ELM Digital Circuit](http://arxiv.org/abs/2103.09791)


  Currently there has been increasing demand for real-time training on
resource-limited IoT devices such as smart sensors, which realizes standalone
online adaptation for streaming data without data transfers to remote servers.
OS-ELM (Online Sequential Extreme Learning Machine) has been one of promising
neural-network-based online algorithms for on-chip learning because it can
perform online training at low computational cost and is easy to implement as a
digital circuit. Existing OS-ELM digital circuits employ fixed-point data
format and the bit-widths are often manually tuned, however, this may cause
overflow or underflow which can lead to unexpected behavior of the circuit. For
on-chip learning systems, an overflow/underflow-free design has a great impact
since online training is continuously performed and the intervals of
intermediate variables will dynamically change as time goes by. In this paper,
we propose an overflow/underflow-free bit-width optimization method for
fixed-point digital circuits of OS-ELM. Experimental results show that our
method realizes overflow/underflow-free OS-ELM digital circuits with 1.0x -
1.5x more area cost compared to the baseline simulation method where overflow
or underflow can happen.

    

### [[2103.14082] Learning Stable Representations with Full Encoder](http://arxiv.org/abs/2103.14082)


  While the beta-VAE family is aiming to find disentangled representations and
acquire human-interpretable generative factors, like what an ICA (from the
linear domain) does, we propose Full Encoder, a novel unified autoencoder
framework as a correspondence to PCA in the non-linear domain. The idea is to
train an autoencoder with one latent variable first, then involve more latent
variables progressively to refine the reconstruction results. The Full Encoder
is also a latent variable predictive model that the latent variables acquired
are stable and robust, as they always learn the same representation regardless
of the network initial states. Full Encoder can be used to determine the
degrees of freedom in a simple non-linear system and can be useful for data
compression or anomaly detection. Full Encoder can also be combined with the
beta-VAE framework to sort out the importance of the generative factors,
providing more insights for non-linear system analysis. These qualities will
make FE useful for analyzing real-life industrial non-linear systems. To
validate, we created a toy dataset with a custom-made non-linear system to test
it and compare its properties to those of VAE and beta-VAE's.

    

### [[2103.16111] A resource-efficient method for repeated HPO and NAS problems](http://arxiv.org/abs/2103.16111)


  In this work we consider the problem of repeated hyperparameter and neural
architecture search (HNAS). We propose an extension of Successive Halving that
is able to leverage information gained in previous HNAS problems with the goal
of saving computational resources. We empirically demonstrate that our solution
is able to drastically decrease costs while maintaining accuracy and being
robust to negative transfer. Our method is significantly simpler than competing
transfer learning approaches, setting a new baseline for transfer learning in
HNAS.

    

### [[2103.16440] Neural Transformation Learning for Deep Anomaly Detection Beyond Images](http://arxiv.org/abs/2103.16440)


  Data transformations (e.g. rotations, reflections, and cropping) play an
important role in self-supervised learning. Typically, images are transformed
into different views, and neural networks trained on tasks involving these
views produce useful feature representations for downstream tasks, including
anomaly detection. However, for anomaly detection beyond image data, it is
often unclear which transformations to use. Here we present a simple end-to-end
procedure for anomaly detection with learnable transformations. The key idea is
to embed the transformed data into a semantic space such that the transformed
data still resemble their untransformed form, while different transformations
are easily distinguishable. Extensive experiments on time series demonstrate
that our proposed method outperforms existing approaches in the one-vs.-rest
setting and is competitive in the more challenging n-vs.-rest anomaly detection
task. On tabular datasets from the medical and cyber-security domains, our
method learns domain-specific transformations and detects anomalies more
accurately than previous work.

    

### [[2104.00138] Rapid quantification of COVID-19 pneumonia burden from computed tomography with convolutional LSTM networks](http://arxiv.org/abs/2104.00138)


  Quantitative lung measures derived from computed tomography (CT) have been
demonstrated to improve prognostication in coronavirus disease (COVID-19)
patients, but are not part of the clinical routine since required manual
segmentation of lung lesions is prohibitively time-consuming. We propose a new
fully automated deep learning framework for rapid quantification and
differentiation between lung lesions in COVID-19 pneumonia from both contrast
and non-contrast CT images using convolutional Long Short-Term Memory
(ConvLSTM) networks. Utilizing the expert annotations, model training was
performed 5 times with separate hold-out sets using 5-fold cross-validation to
segment ground-glass opacity and high opacity (including consolidation and
pleural effusion). The performance of the method was evaluated on CT data sets
from 197 patients with positive reverse transcription polymerase chain reaction
test result for SARS-CoV-2. Strong agreement between expert manual and
automatic segmentation was obtained for lung lesions with a Dice score
coefficient of 0.876 $\pm$ 0.005; excellent correlations of 0.978 and 0.981 for
ground-glass opacity and high opacity volumes. In the external validation set
of 67 patients, there was dice score coefficient of 0.767 $\pm$ 0.009 as well
as excellent correlations of 0.989 and 0.996 for ground-glass opacity and high
opacity volumes. Computations for a CT scan comprising 120 slices were
performed under 2 seconds on a personal computer equipped with NVIDIA Titan RTX
graphics processing unit. Therefore, our deep learning-based method allows
rapid fully-automated quantitative measurement of pneumonia burden from CT and
may generate results with an accuracy similar to the expert readers.

    

### [[2104.01874] Deep Learning of Conjugate Mappings](http://arxiv.org/abs/2104.01874)


  Despite many of the most common chaotic dynamical systems being continuous in
time, it is through discrete time mappings that much of the understanding of
chaos is formed. Henri Poincaré first made this connection by tracking
consecutive iterations of the continuous flow with a lower-dimensional,
transverse subspace. The mapping that iterates the dynamics through consecutive
intersections of the flow with the subspace is now referred to as a Poincaré
map, and it is the primary method available for interpreting and classifying
chaotic dynamics. Unfortunately, in all but the simplest systems, an explicit
form for such a mapping remains outstanding. This work proposes a method for
obtaining explicit Poincaré mappings by using deep learning to construct an
invertible coordinate transformation into a conjugate representation where the
dynamics are governed by a relatively simple chaotic mapping. The invertible
change of variable is based on an autoencoder, which allows for dimensionality
reduction, and has the advantage of classifying chaotic systems using the
equivalence relation of topological conjugacies. Indeed, the enforcement of
topological conjugacies is the critical neural network regularization for
learning the coordinate and dynamics pairing. We provide expository
applications of the method to low-dimensional systems such as the Rössler and
Lorenz systems, while also demonstrating the utility of the method on
infinite-dimensional systems, such as the Kuramoto--Sivashinsky equation.

    

### [[2104.02938] Deep Interpretable Models of Theory of Mind](http://arxiv.org/abs/2104.02938)


  When developing AI systems that interact with humans, it is essential to
design both a system that can understand humans, and a system that humans can
understand. Most deep network based agent-modeling approaches are 1) not
interpretable and 2) only model external behavior, ignoring internal mental
states, which potentially limits their capability for assistance,
interventions, discovering false beliefs, etc. To this end, we develop an
interpretable modular neural framework for modeling the intentions of other
observed entities. We demonstrate the efficacy of our approach with experiments
on data from human participants on a search and rescue task in Minecraft, and
show that incorporating interpretability can significantly increase predictive
performance under the right conditions.

    

### [[2104.03674] Explainability-based Backdoor Attacks Against Graph Neural Networks](http://arxiv.org/abs/2104.03674)


  Backdoor attacks represent a serious threat to neural network models. A
backdoored model will misclassify the trigger-embedded inputs into an
attacker-chosen target label while performing normally on other benign inputs.
There are already numerous works on backdoor attacks on neural networks, but
only a few works consider graph neural networks (GNNs). As such, there is no
intensive research on explaining the impact of trigger injecting position on
the performance of backdoor attacks on GNNs.
To bridge this gap, we conduct an experimental investigation on the
performance of backdoor attacks on GNNs. We apply two powerful GNN
explainability approaches to select the optimal trigger injecting position to
achieve two attacker objectives -- high attack success rate and low clean
accuracy drop. Our empirical results on benchmark datasets and state-of-the-art
neural network models demonstrate the proposed method's effectiveness in
selecting trigger injecting position for backdoor attacks on GNNs. For
instance, on the node classification task, the backdoor attack with trigger
injecting position selected by GraphLIME reaches over $84 \%$ attack success
rate with less than $2.5 \%$ accuracy drop

    

### [[2104.05379] Comparing the Benefit of Synthetic Training Data for Various Automatic Speech Recognition Architectures](http://arxiv.org/abs/2104.05379)


  Recent publications on automatic-speech-recognition (ASR) have a strong focus
on attention encoder-decoder (AED) architectures which tend to suffer from
over-fitting in low resource scenarios. One solution to tackle this issue is to
generate synthetic data with a trained text-to-speech system (TTS) if
additional text is available. This was successfully applied in many
publications with AED systems, but only very limited in the context of other
ASR architectures. We investigate the effect of varying pre-processing, the
speaker embedding and input encoding of the TTS system w.r.t. the effectiveness
of the synthesized data for AED-ASR training. Additionally, we also consider
internal language model subtraction for the first time, resulting in up to 38%
relative improvement. We compare the AED results to a state-of-the-art hybrid
ASR system, a monophone based system using
connectionist-temporal-classification (CTC) and a monotonic transducer based
system. We show that for the later systems the addition of synthetic data has
no relevant effect, but they still outperform the AED systems on
LibriSpeech-100h. We achieve a final word-error-rate of 3.3%/10.0% with a
hybrid system on the clean/noisy test-sets, surpassing any previous
state-of-the-art systems on Librispeech-100h that do not include unlabeled
audio data.

    

### [[2104.05942] Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness](http://arxiv.org/abs/2104.05942)


  This paper introduces recurrent equilibrium networks (RENs), a new class of
nonlinear dynamical models for applications in machine learning, system
identification and control. The new model class has ``built in'' guarantees of
stability and robustness: all models in the class are contracting - a strong
form of nonlinear stability - and models can satisfy prescribed incremental
integral quadratic constraints (IQC), including Lipschitz bounds and
incremental passivity. RENs are otherwise very flexible: they can represent all
stable linear systems, all previously-known sets of contracting recurrent
neural networks and echo state networks, all deep feedforward neural networks,
and all stable Wiener/Hammerstein models. RENs are parameterized directly by a
vector in R^N, i.e. stability and robustness are ensured without parameter
constraints, which simplifies learning since generic methods for unconstrained
optimization can be used. The performance and robustness of the new model set
is evaluated on benchmark nonlinear system identification problems, and the
paper also presents applications in data-driven nonlinear observer design and
control with stability guarantees.

    

### [[2104.10482] GraphSVX: Shapley Value Explanations for Graph Neural Networks](http://arxiv.org/abs/2104.10482)


  Graph Neural Networks (GNNs) achieve significant performance for various
learning tasks on geometric data due to the incorporation of graph structure
into the learning of node representations, which renders their comprehension
challenging. In this paper, we first propose a unified framework satisfied by
most existing GNN explainers. Then, we introduce GraphSVX, a post hoc local
model-agnostic explanation method specifically designed for GNNs. GraphSVX is a
decomposition technique that captures the "fair" contribution of each feature
and node towards the explained prediction by constructing a surrogate model on
a perturbed dataset. It extends to graphs and ultimately provides as
explanation the Shapley Values from game theory. Experiments on real-world and
synthetic datasets demonstrate that GraphSVX achieves state-of-the-art
performance compared to baseline models while presenting core theoretical and
human-centric properties.

    

### [[2104.12556] COVID-19 Modeling: A Review](http://arxiv.org/abs/2104.12556)


  The SARS-CoV-2 virus and COVID-19 disease have posed unprecedented and
overwhelming challenges and opportunities to data and domain-driven modeling.
This paper makes a comprehensive review of the challenges, tasks, methods, gaps
and opportunities on modeling COVID-19 problems and data. It constructs a
research landscape of COVID-19 modeling, and further categorizes, compares and
discusses the related work on modeling COVID-19 epidemic transmission processes
and dynamics, case identification and tracing, infection diagnosis and trends,
medical treatments, non-pharmaceutical intervention effect, drug and vaccine
development, psychological, economic and social impact, and misinformation,
etc. The modeling methods involve mathematical and statistical models,
domain-driven modeling by epidemiological compartmental models, medical and
biomedical analysis, data-driven learning by shallow and deep machine learning,
simulation systems, social science methods, and hybrid methods.

    

### [[2105.03793] Stability and Generalization of Stochastic Gradient Methods for Minimax Problems](http://arxiv.org/abs/2105.03793)


  Many machine learning problems can be formulated as minimax problems such as
Generative Adversarial Networks (GANs), AUC maximization and robust estimation,
to mention but a few. A substantial amount of studies are devoted to studying
the convergence behavior of their stochastic gradient-type algorithms. In
contrast, there is relatively little work on their generalization, i.e., how
the learning models built from training examples would behave on test examples.
In this paper, we provide a comprehensive generalization analysis of stochastic
gradient methods for minimax problems under both convex-concave and
nonconvex-nonconcave cases through the lens of algorithmic stability. We
establish a quantitative connection between stability and several
generalization measures both in expectation and with high probability. For the
convex-concave setting, our stability analysis shows that stochastic gradient
descent ascent attains optimal generalization bounds for both smooth and
nonsmooth minimax problems. We also establish generalization bounds for both
weakly-convex-weakly-concave and gradient-dominated problems.

    

### [[2105.07671] Classifying variety of customer's online engagement for churn prediction with mixed-penalty logistic regression](http://arxiv.org/abs/2105.07671)


  Using big data to analyze consumer behavior can provide effective
decision-making tools for preventing customer attrition (churn) in customer
relationship management (CRM). Focusing on a CRM dataset with several different
categories of factors that impact customer heterogeneity (i.e., usage of
self-care service channels, duration of service, and responsiveness to
marketing actions), we provide new predictive analytics of customer churn rate
based on a machine learning method that enhances the classification of logistic
regression by adding a mixed penalty term. The proposed penalized logistic
regression can prevent overfitting when dealing with big data and minimize the
loss function when balancing the cost from the median (absolute value) and mean
(squared value) regularization. We show the analytical properties of the
proposed method and its computational advantage in this research. In addition,
we investigate the performance of the proposed method with a CRM data set (that
has a large number of features) under different settings by efficiently
eliminating the disturbance of (1) least important features and (2) sensitivity
from the minority (churn) class. Our empirical results confirm the expected
performance of the proposed method in full compliance with the common
classification criteria (i.e., accuracy, precision, and recall) for evaluating
machine learning methods.

    

### [[2105.11681] Deep Neural Networks and End-to-End Learning for Audio Compression](http://arxiv.org/abs/2105.11681)


  Recent achievements in end-to-end deep learning have encouraged the
exploration of tasks dealing with highly structured data with unified deep
network models. Having such models for compressing audio signals has been
challenging since it requires discrete representations that are not easy to
train with end-to-end backpropagation. In this paper, we present an end-to-end
deep learning approach that combines recurrent neural networks (RNNs) within
the training strategy of variational autoencoders (VAEs) with a binary
representation of the latent space. We apply a reparametrization trick for the
Bernoulli distribution for the discrete representations, which allows smooth
backpropagation. In addition, our approach allows the separation of the encoder
and decoder, which is necessary for compression tasks. To our best knowledge,
this is the first end-to-end learning for a single audio compression model with
RNNs, and our model achieves a Signal to Distortion Ratio (SDR) of 20.54.

    

### [[2105.15183] Efficient and Modular Implicit Differentiation](http://arxiv.org/abs/2105.15183)


  Automatic differentiation (autodiff) has revolutionized machine learning. It
allows expressing complex computations by composing elementary ones in creative
ways and removes the burden of computing their derivatives by hand. More
recently, differentiation of optimization problem solutions has attracted
widespread attention with applications such as optimization as a layer, and in
bi-level problems such as hyper-parameter optimization and meta-learning.
However, the formulas for these derivatives often involve case-by-case tedious
mathematical derivations. In this paper, we propose a unified, efficient and
modular approach for implicit differentiation of optimization problems. In our
approach, the user defines (in Python in the case of our implementation) a
function $F$ capturing the optimality conditions of the problem to be
differentiated. Once this is done, we leverage autodiff of $F$ and implicit
differentiation to automatically differentiate the optimization problem. Our
approach thus combines the benefits of implicit differentiation and autodiff.
It is efficient as it can be added on top of any state-of-the-art solver and
modular as the optimality condition specification is decoupled from the
implicit differentiation mechanism. We show that seemingly simple principles
allow to recover many recently proposed implicit differentiation methods and
create new ones easily. We demonstrate the ease of formulating and solving
bi-level optimization problems using our framework. We also showcase an
application to the sensitivity analysis of molecular dynamics.

    

### [[2106.03843] Equivariant Graph Neural Networks for 3D Macromolecular Structure](http://arxiv.org/abs/2106.03843)


  Representing and reasoning about 3D structures of macromolecules is emerging
as a distinct challenge in machine learning. Here, we extend recent work on
geometric vector perceptrons and apply equivariant graph neural networks to a
wide range of tasks from structural biology. Our method outperforms all
reference architectures on three out of eight tasks in the ATOM3D benchmark, is
tied for first on two others, and is competitive with equivariant networks
using higher-order representations and spherical harmonic convolutions. In
addition, we demonstrate that transfer learning can further improve performance
on certain downstream tasks. Code is available at
this https URL.

    

### [[2106.04718] FastSeq: Make Sequence Generation Faster](http://arxiv.org/abs/2106.04718)


  Transformer-based models have made tremendous impacts in natural language
generation. However the inference speed is a bottleneck due to large model size
and intensive computing involved in auto-regressive decoding process. We
develop FastSeq framework to accelerate sequence generation without accuracy
loss. The proposed optimization techniques include an attention cache
optimization, an efficient algorithm for detecting repeated n-grams, and an
asynchronous generation pipeline with parallel I/O. These optimizations are
general enough to be applicable to Transformer-based models (e.g., T5, GPT2,
and UniLM). Our benchmark results on a set of widely used and diverse models
demonstrate 4-9x inference speed gain. Additionally, FastSeq is easy to use
with a simple one-line code change. The source code is available at
this https URL.

    

### [[2106.05386] Artificial Intelligence in Drug Discovery: Applications and Techniques](http://arxiv.org/abs/2106.05386)


  Artificial intelligence (AI) has been transforming the practice of drug
discovery in the past decade. Various AI techniques have been used in a wide
range of applications, such as virtual screening and drug design. In this
survey, we first give an overview on drug discovery and discuss related
applications, which can be reduced to two major tasks, i.e., molecular property
prediction and molecule generation. We then discuss common data resources,
molecule representations and benchmark platforms. Furthermore, to summarize the
progress of AI in drug discovery, we present the relevant AI techniques
including model architectures and learning paradigms in the papers surveyed. We
expect that this survey will serve as a guide for researchers who are
interested in working at the interface of artificial intelligence and drug
discovery. We also provide a GitHub repository
(this https URL) with the collection
of papers and codes, if applicable, as a learning resource, which is regularly
updated.

    

### [[2106.06046] Information Theoretic Evaluation of Privacy-Leakage, Interpretability, and Transferability for a Novel Trustworthy AI Framework](http://arxiv.org/abs/2106.06046)


  Guidelines and principles of trustworthy AI should be adhered to in practice
during the development of AI systems. This work suggests a novel information
theoretic trustworthy AI framework based on the hypothesis that information
theory enables taking into account the ethical AI principles during the
development of machine learning and deep learning models via providing a way to
study and optimize the inherent tradeoffs between trustworthy AI principles.
Under the proposed framework, a unified approach to ``privacy-preserving
interpretable and transferable learning'' is considered to introduce the
information theoretic measures for privacy-leakage, interpretability, and
transferability. A technique based on variational optimization, employing
\emph{conditionally deep autoencoders}, is developed for practically
calculating the defined information theoretic measures for privacy-leakage,
interpretability, and transferability.

    

### [[2106.06080] Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent](http://arxiv.org/abs/2106.06080)


  We focus on the problem of domain adaptation when the goal is shifting the
model towards the target distribution, rather than learning domain invariant
representations. It has been shown that under the following two assumptions:
(a) access to samples from intermediate distributions, and (b) samples being
annotated with the amount of change from the source distribution, self-training
can be successfully applied on gradually shifted samples to adapt the model
toward the target distribution. We hypothesize having (a) is enough to enable
iterative self-training to slowly adapt the model to the target distribution,
by making use of an implicit curriculum. In the case where (a) does not hold,
we observe that iterative self-training falls short. We propose GIFT, a method
that creates virtual samples from intermediate distributions by interpolating
representations of examples from source and target domains. We evaluate an
iterative-self-training method on datasets with natural distribution shifts,
and show that when applied on top of other domain adaptation methods, it
improves the performance of the model on the target dataset. We run an analysis
on a synthetic dataset to show that in the presence of (a)
iterative-self-training naturally forms a curriculum of samples. Furthermore,
we show that when (a) does not hold, GIFT performs better than iterative
self-training.

    

### [[2106.06232] GDI: Rethinking What Makes Reinforcement Learning Different From Supervised Learning](http://arxiv.org/abs/2106.06232)


  Deep Q Network (DQN) firstly kicked the door of deep reinforcement learning
(DRL) via combining deep learning (DL) with reinforcement learning (RL), which
has noticed that the distribution of the acquired data would change during the
training process. DQN found this property might cause instability for training,
so it proposed effective methods to handle the downside of the property.
Instead of focusing on the unfavourable aspects, we find it critical for RL to
ease the gap between the estimated data distribution and the ground truth data
distribution while supervised learning (SL) fails to do so. From this new
perspective, we extend the basic paradigm of RL called the Generalized Policy
Iteration (GPI) into a more generalized version, which is called the
Generalized Data Distribution Iteration (GDI). We see massive RL algorithms and
techniques can be unified into the GDI paradigm, which can be considered as one
of the special cases of GDI. We provide theoretical proof of why GDI is better
than GPI and how it works. Several practical algorithms based on GDI have been
proposed to verify the effectiveness and extensiveness of it. Empirical
experiments prove our state-of-the-art (SOTA) performance on Arcade Learning
Environment (ALE), wherein our algorithm has achieved 9620.98% mean human
normalized score (HNS), 1146.39% median HNS and 22 human world record
breakthroughs (HWRB) using only 200 training frames. Our work aims to lead the
RL research to step into the journey of conquering the human world records and
seek real superhuman agents on both performance and efficiency.

    

### [[2107.00630] Variational Diffusion Models](http://arxiv.org/abs/2107.00630)


  Diffusion-based generative models have demonstrated a capacity for
perceptually impressive synthesis, but can they also be great likelihood-based
models? We answer this in the affirmative, and introduce a family of
diffusion-based generative models that obtain state-of-the-art likelihoods on
standard image density estimation benchmarks. Unlike other diffusion-based
models, our method allows for efficient optimization of the noise schedule
jointly with the rest of the model. We show that the variational lower bound
(VLB) simplifies to a remarkably short expression in terms of the
signal-to-noise ratio of the diffused data, thereby improving our theoretical
understanding of this model class. Using this insight, we prove an equivalence
between several models proposed in the literature. In addition, we show that
the continuous-time VLB is invariant to the noise schedule, except for the
signal-to-noise ratio at its endpoints. This enables us to learn a noise
schedule that minimizes the variance of the resulting VLB estimator, leading to
faster optimization. Combining these advances with architectural improvements,
we obtain state-of-the-art likelihoods on image density estimation benchmarks,
outperforming autoregressive models that have dominated these benchmarks for
many years, with often significantly faster optimization. In addition, we show
how to turn the model into a bits-back compression scheme, and demonstrate
lossless compression rates close to the theoretical optimum.

    

### [[2107.05384] Fine-Grained AutoAugmentation for Multi-Label Classification](http://arxiv.org/abs/2107.05384)


  Data augmentation is a commonly used approach to improving the generalization
of deep learning models. Recent works show that learned data augmentation
policies can achieve better generalization than hand-crafted ones. However,
most of these works use unified augmentation policies for all samples in a
dataset, which is observed not necessarily beneficial for all labels in
multi-label classification tasks, i.e., some policies may have negative impacts
on some labels while benefitting the others. To tackle this problem, we propose
a novel Label-Based AutoAugmentation (LB-Aug) method for multi-label scenarios,
where augmentation policies are generated with respect to labels by an
augmentation-policy network. The policies are learned via reinforcement
learning using policy gradient methods, providing a mapping from instance
labels to their optimal augmentation policies. Numerical experiments show that
our LB-Aug outperforms previous state-of-the-art augmentation methods by large
margins in multiple benchmarks on image and video classification.

    

### [[2107.05541] End-to-End Natural Language Understanding Pipeline for Bangla Conversational Agents](http://arxiv.org/abs/2107.05541)


  Chatbots are intelligent software built to be used as a replacement for human
interaction. However, existing studies typically do not provide enough support
for low-resource languages like Bangla. Moreover, due to the increasing
popularity of social media, we can also see the rise of interactions in Bangla
transliteration (mostly in English) among the native Bangla speakers. In this
paper, we propose a novel approach to build a Bangla chatbot aimed to be used
as a business assistant which can communicate in Bangla and Bangla
Transliteration in English with high confidence consistently. Since annotated
data was not available for this purpose, we had to work on the whole machine
learning life cycle (data preparation, machine learning modeling, and model
deployment) using Rasa Open Source Framework, fastText embeddings, Polyglot
embeddings, Flask, and other systems as building blocks. While working with the
skewed annotated dataset, we try out different setups and pipelines to evaluate
which works best and provide possible reasoning behind the observed results.
Finally, we present a pipeline for intent classification and entity extraction
which achieves reasonable performance (accuracy: 83.02\%, precision: 80.82\%,
recall: 83.02\%, F1-score: 80\%).

    

### [[2107.05791] Monotonic Filtering for Distributed Collection](http://arxiv.org/abs/2107.05791)


  Distributed data collection is a fundamental task in open systems. In such
networks, data is aggregated across a network to produce a single aggregated
result at a source device. Though self-stabilizing, algorithms performing data
collection can produce large overestimates in the transient phase. For example,
in [1] we demonstrated that in a line graph, a switch of sources after initial
stabilization may produce overestimates that are quadratic in the network
diameter. We also proposed monotonic filtering as a strategy for removing such
large overestimates. Monotonic filtering prevents the transfer of data from
device A to device B unless the distance estimate at A is more than that at B
at the previous iteration. For a line graph, [1] shows that monotonic filtering
prevents quadratic overestimates. This paper analyzes monotonic filtering for
an arbitrary graph topology, showing that for an N device network, the largest
overestimate after switching sources is at most 2N.

    

### [[2107.05793] A Parallel Approximation Algorithm for Maximizing Submodular $b$-Matching](http://arxiv.org/abs/2107.05793)


  We design new serial and parallel approximation algorithms for computing a
maximum weight $b$-matching in an edge-weighted graph with a submodular
objective function. This problem is NP-hard; the new algorithms have
approximation ratio $1/3$, and are relaxations of the Greedy algorithm that
rely only on local information in the graph, making them parallelizable. We
have designed and implemented Local Lazy Greedy algorithms for both serial and
parallel computers. We have applied the approximate submodular $b$-matching
algorithm to assign tasks to processors in the computation of Fock matrices in
quantum chemistry on parallel computers. The assignment seeks to reduce the run
time by balancing the computational load on the processors and bounding the
number of messages that each processor sends. We show that the new assignment
of tasks to processors provides a four fold speedup over the currently used
assignment in the NWChemEx software on $8000$ processors on the Summit
supercomputer at Oak Ridge National Lab.

    

### [[2107.05951] One-Point Gradient-Free Methods for Composite Optimization with Applications to Distributed Optimization](http://arxiv.org/abs/2107.05951)


  This work is devoted to solving the composite optimization problem with the
mixture oracle: for the smooth part of the problem, we have access to the
gradient, and for the non-smooth part, only to the one-point zero-order oracle.
We present a method based on the sliding algorithm. Our method allows to
separate the oracle complexities and compute the gradient for one of the
function as rarely as possible. The paper also examines the applicability of
this method to the problems of distributed optimization and federated learning.

    

### [[2107.06005] Recent Advances in Energy Efficient Resource Management Techniques in Cloud Computing Environments](http://arxiv.org/abs/2107.06005)


  Nowadays cloud computing adoption as a form of hosted application and
services is widespread due to decreasing costs of hardware, software, and
maintenance. Cloud enables access to a shared pool of virtual resources hosted
in large energy-hungry data centers for diverse information and communication
services with dynamic workloads. The huge energy consumption of cloud data
centers results in high electricity bills as well as emission of a large amount
of carbon dioxide gas. Needless to say, efficient resource management in cloud
environments has become one of the most important priorities of cloud providers
and consequently has increased the interest of researchers to propose novel
energy saving solutions. This chapter presents a scientific and taxonomic
survey of recent energy efficient cloud resource management' solutions in cloud
environments. The main objective of this study is to propose a novel complete
taxonomy for energy-efficient cloud resource management solutions, review
recent research advancements in this area, classify the existing techniques
based on our proposed taxonomy, and open up new research directions. Besides,
it reviews and surveys the literature in the range of 2015 through 2021 in the
subject of energy-efficient cloud resource management techniques and maps them
to its proposed taxonomy, which unveils novel research directions and
facilitates the conduction of future researches.

    

### [[2107.06108] Transitioning from file-based HPC workflows to streaming data pipelines with openPMD and ADIOS2](http://arxiv.org/abs/2107.06108)


  This paper aims to create a transition path from file-based IO to
streaming-based workflows for scientific applications in an HPC environment. By
using the openPMP-api, traditional workflows limited by filesystem bottlenecks
can be overcome and flexibly extended for in situ analysis. The openPMD-api is
a library for the description of scientific data according to the Open Standard
for Particle-Mesh Data (openPMD). Its approach towards recent challenges posed
by hardware heterogeneity lies in the decoupling of data description in domain
sciences, such as plasma physics simulations, from concrete implementations in
hardware and IO. The streaming backend is provided by the ADIOS2 framework,
developed at Oak Ridge National Laboratory. This paper surveys two
openPMD-based loosely coupled setups to demonstrate flexible applicability and
to evaluate performance. In loose coupling, as opposed to tight coupling, two
(or more) applications are executed separately, e.g. in individual MPI
contexts, yet cooperate by exchanging data. This way, a streaming-based
workflow allows for standalone codes instead of tightly-coupled plugins, using
a unified streaming-aware API and leveraging high-speed communication
infrastructure available in modern compute clusters for massive data exchange.
We determine new challenges in resource allocation and in the need of
strategies for a flexible data distribution, demonstrating their influence on
efficiency and scaling on the Summit compute system. The presented setups show
the potential for a more flexible use of compute resources brought by streaming
IO as well as the ability to increase throughput by avoiding filesystem
bottlenecks.

    

### [[2102.08304] Speeding Up Private Distributed Matrix Multiplication via Bivariate Polynomial Codes](http://arxiv.org/abs/2102.08304)


  We consider the problem of private distributed matrix multiplication under
limited resources. Coded computation has been shown to be an effective solution
in distributed matrix multiplication, both providing privacy against the
workers and boosting the computation speed by efficiently mitigating
stragglers. In this work, we propose the use of recently-introduced bivariate
polynomial codes to further speed up private distributed matrix multiplication
by exploiting the partial work done by the stragglers rather than completely
ignoring them. We show that the proposed approach reduces the average
computation time of private distributed matrix multiplication compared to its
competitors in the literature while improving the upload communication cost and
the workers' storage efficiency.

    

### [[2104.14354] SoCRATES: System-on-Chip Resource Adaptive Scheduling using Deep Reinforcement Learning](http://arxiv.org/abs/2104.14354)


  Deep Reinforcement Learning (DRL) is being increasingly applied to the
problem of resource allocation for emerging System-on-Chip (SoC) applications,
and has shown remarkable promises. In this paper, we introduce SoCRATES (SoC
Resource AdapTivE Scheduler), an extremely efficient DRL-based SoC scheduler
which maps a wide range of hierarchical jobs to heterogeneous resources within
SoC using the Eclectic Interaction Matching (EIM) technique. It is noted that
the majority of SoC resource management approaches have been targeting makespan
minimization with fixed number of jobs in the system. In contrast, SoCRATES
aims at minimizing average latency in a steady-state condition while assigning
tasks in the ready queue to heterogeneous resources (processing elements). We
first show that existing DRL-based schedulers developed with the makespan
minimization objective are ineffective for the latency-minimization-driven SoC
applications due to their characteristics such as high-frequency job workload
and distributed/parallel job execution. We then demonstrate that through its
EIM technique, SoCRATES successfully addresses the challenge of concurrent
observations caused by the task dependency inherent in the latency minimization
objective. Extensive tests show that SoCRATES outperforms other existing neural
and non-neural schedulers with as high as 38% gain in latency reduction under a
variety of job types, queue length, and incoming rates. The resulting model is
also compact in size and has very favorable energy consumption behaviors,
making it highly practical for deployment in future SoC systems with built-in
neural accelerator.

    

### [[2105.12083] Efficient Assignment of Identities in Anonymous Populations](http://arxiv.org/abs/2105.12083)


  We consider the fundamental problem of assigning distinct labels to agents in
the probabilistic model of population protocols. Our protocols operate under
the assumption that the size $n$ of the population is embedded in the
transition function. They are silent, i.e., eventually each agent reaches its
final state and remains in it forever, as well as are safe, i.e., they can
produce a valid agent labeling in a finite number of interactions, and
guarantee that at any step of the protocol no two agents have the same label.
We first present a fast, silent and safe labeling protocol for which the
required number of interactions is asymptotically optimal, i.e., $O(n \log
n/\epsilon)$ w.h.p. It uses $(2+\epsilon)n+O(n^c)$ states, for any $c<1,$ and
the label range $1,\dots,(1+\epsilon)n.$ Furthermore, we consider the so-called
pool labeling protocols that include our fast protocol. We show that the
expected number of interactions required by any pool protocol is $\ge
\frac{n^2}{r+1}$, when the labels range is $1,\dots, n+r<2n.$ Next, we provide
a silent and safe protocol which uses only $n+5\sqrt n +O(n^c)$ states, for any
$c<1,$ and draws labels from the range $1,\dots,n.$ . The expected number of
interactions required by the protocol is $O(n^3).$ On the other hand, we show
that any safe protocol, as well as any silent protocol which provides a valid
labeling with probability $>1-\frac 1n$, uses $\ge n+\sqrt n-1$ states. Hence,
our protocol is almost state-optimal. We also present a generalization of the
protocol to include a trade-off between the number of states and the expected
number of interactions. Furthermore, we show that for any safe labeling
protocol utilizing $n+t<2n$ states the expected number of interactions required
to achieve a valid labeling is $\ge \frac{n^2}{t+1}$.

    

### [[2106.08290] Coded Privacy-Preserving Computation at Edge Networks](http://arxiv.org/abs/2106.08290)


  Multi-party computation (MPC) is promising for privacy-preserving machine
learning algorithms at edge networks, like federated learning. Despite their
potential, existing MPC algorithms fail short of adapting to the limited
resources of edge devices. A promising solution, and the focus of this work, is
coded computation, which advocates the use of error-correcting codes to improve
the performance of distributed computing through "smart" data redundancy. In
this paper, we focus on coded privacy-preserving computation using Shamir's
secret sharing. In particular, we design novel coded privacy-preserving
computation mechanisms; MatDot coded MPC (MatDot-CMPC) and PolyDot coded MPC
(PolyDot-CMPC) by employing recently proposed coded computation algorithms;
MatDot and PolyDot. We take advantage of the "garbage terms" that naturally
arise when polynomials are constructed in the design of MatDot-CMPC and
PolyDot-CMPC to reduce the number of workers needed for privacy-preserving
computation. Also, we analyze MatDot-CMPC and PolyDot-CMPC in terms of their
computation, storage, communication overhead as well as recovery threshold, so
they can easily adapt to the limited resources of edge devices.

    

### [[2107.05473] GPTPU: Accelerating Applications using Edge Tensor Processing Units](http://arxiv.org/abs/2107.05473)


  Neural network (NN) accelerators have been integrated into a wide-spectrum of
computer systems to accommodate the rapidly growing demands for artificial
intelligence (AI) and machine learning (ML) applications. NN accelerators share
the idea of providing native hardware support for operations on
multidimensional tensor data. Therefore, NN accelerators are theoretically
tensor processors that can improve system performance for any problem that uses
tensors as inputs/outputs. Unfortunately, commercially available NN
accelerators only expose computation capabilities through AI/ML-specific
interfaces. Furthermore, NN accelerators reveal very few hardware design
details, so applications cannot easily leverage the tensor operations NN
accelerators provide.
This paper introduces General-Purpose Computing on Edge Tensor Processing
Units (GPTPU), an open-source, open-architecture framework that allows the
developer and research communities to discover opportunities that NN
accelerators enable for applications. GPTPU includes a powerful programming
interface with efficient runtime system-level support -- similar to that of
CUDA/OpenCL in GPGPU computing -- to bridge the gap between application demands
and mismatched hardware/software interfaces.
We built GPTPU machine uses Edge Tensor Processing Units (Edge TPUs), which
are widely available and representative of many commercial NN accelerators. We
identified several novel use cases and revisited the algorithms. By leveraging
the underlying Edge TPUs to perform tensor-algorithm-based compute kernels, our
results reveal that GPTPU can achieve a 2.46x speedup over high-end CPUs and
reduce energy consumption by 40%.

    

### [[2107.05731] Detecting Ideal Instagram Influencer Using Social Network Analysis](http://arxiv.org/abs/2107.05731)


  Social Media is a key aspect of modern society where people share their
thoughts, views, feelings and sentiments. Over the last few years, the
inflation in popularity of social media has resulted in a monumental increase
in data. Users use this medium to express their thoughts, feelings, and
opinions on a wide variety of subjects, including politics and celebrities.
Social Media has thus evolved into a lucrative platform for companies to expand
their scope and improve their prospects. The paper focuses on social network
analysis (SNA) for a real-world online marketing strategy. The study
contributes by comparing various centrality measures to identify the most
central nodes in the network and uses a linear threshold model to understand
the spreading behaviour of individual users. In conclusion, the paper
correlates different centrality measures and spreading behaviour to identify
the most influential user in the network

    

### [[2107.05756] Reinforcement Learning based Proactive Control for Transmission Grid Resilience to Wildfire](http://arxiv.org/abs/2107.05756)


  Power grid operation subject to an extreme event requires decision-making by
human operators under stressful condition with high cognitive load. Decision
support under adverse dynamic events, specially if forecasted, can be
supplemented by intelligent proactive control. Power system operation during
wildfires require resiliency-driven proactive control for load shedding, line
switching and resource allocation considering the dynamics of the wildfire and
failure propagation. However, possible number of line- and load-switching in a
large system during an event make traditional prediction-driven and stochastic
approaches computationally intractable, leading operators to often use greedy
algorithms. We model and solve the proactive control problem as a Markov
decision process and introduce an integrated testbed for spatio-temporal
wildfire propagation and proactive power-system operation. We transform the
enormous wildfire-propagation observation space and utilize it as part of a
heuristic for proactive de-energization of transmission assets. We integrate
this heuristic with a reinforcement-learning based proactive policy for
controlling the generating assets. Our approach allows this controller to
provide setpoints for a part of the generation fleet, while a myopic operator
can determine the setpoints for the remaining set, which results in a symbiotic
action. We evaluate our approach utilizing the IEEE 24-node system mapped on a
hypothetical terrain. Our results show that the proposed approach can help the
operator to reduce load loss during an extreme event, reduce power flow through
lines that are to be de-energized, and reduce the likelihood of infeasible
power-flow solutions, which would indicate violation of short-term thermal
limits of transmission lines.

    

### [[2107.05777] An active dendritic tree can mitigate fan-in limitations in superconducting neurons](http://arxiv.org/abs/2107.05777)


  Superconducting electronic circuits have much to offer with regard to
neuromorphic hardware. Superconducting quantum interference devices (SQUIDs)
can serve as an active element to perform the thresholding operation of a
neuron's soma. However, a SQUID has a response function that is periodic in the
applied signal. We show theoretically that if one restricts the total input to
a SQUID to maintain a monotonically increasing response, a large fraction of
synapses must be active to drive a neuron to threshold. We then demonstrate
that an active dendritic tree (also based on SQUIDs) can significantly reduce
the fraction of synapses that must be active to drive the neuron to threshold.
In this context, the inclusion of a dendritic tree provides the dual benefits
of enhancing the computational abilities of each neuron and allowing the neuron
to spike with sparse input activity.

    

### [[2107.05789] Kit-Net: Self-Supervised Learning to Kit Novel 3D Objects into Novel 3D Cavities](http://arxiv.org/abs/2107.05789)


  In industrial part kitting, 3D objects are inserted into cavities for
transportation or subsequent assembly. Kitting is a critical step as it can
decrease downstream processing and handling times and enable lower storage and
shipping costs. We present Kit-Net, a framework for kitting previously unseen
3D objects into cavities given depth images of both the target cavity and an
object held by a gripper in an unknown initial orientation. Kit-Net uses
self-supervised deep learning and data augmentation to train a convolutional
neural network (CNN) to robustly estimate 3D rotations between objects and
matching concave or convex cavities using a large training dataset of simulated
depth images pairs. Kit-Net then uses the trained CNN to implement a controller
to orient and position novel objects for insertion into novel prismatic and
conformal 3D cavities. Experiments in simulation suggest that Kit-Net can
orient objects to have a 98.9% average intersection volume between the object
mesh and that of the target cavity. Physical experiments with industrial
objects succeed in 18% of trials using a baseline method and in 63% of trials
with Kit-Net. Video, code, and data are available at
this https URL.

    

### [[2107.05799] Deep Neural Networks Evolve Human-like Attention Distribution during Reading Comprehension](http://arxiv.org/abs/2107.05799)


  Attention is a key mechanism for information selection in both biological
brains and many state-of-the-art deep neural networks (DNNs). Here, we
investigate whether humans and DNNs allocate attention in comparable ways when
reading a text passage to subsequently answer a specific question. We analyze 3
transformer-based DNNs that reach human-level performance when trained to
perform the reading comprehension task. We find that the DNN attention
distribution quantitatively resembles human attention distribution measured by
fixation times. Human readers fixate longer on words that are more relevant to
the question-answering task, demonstrating that attention is modulated by
top-down reading goals, on top of lower-level visual and text features of the
stimulus. Further analyses reveal that the attention weights in DNNs are also
influenced by both top-down reading goals and lower-level stimulus features,
with the shallow layers more strongly influenced by lower-level text features
and the deep layers attending more to task-relevant words. Additionally, deep
layers' attention to task-relevant words gradually emerges when pre-trained DNN
models are fine-tuned to perform the reading comprehension task, which
coincides with the improvement in task performance. These results demonstrate
that DNNs can evolve human-like attention distribution through task
optimization, which suggests that human attention during goal-directed reading
comprehension is a consequence of task optimization.

    

### [[2107.05850] Encoding Compositionality in Classical Planning Solutions](http://arxiv.org/abs/2107.05850)


  Classical AI planners provide solutions to planning problems in the form of
long and opaque text outputs. To aid in the understanding transferability of
planning solutions, it is necessary to have a rich and comprehensible
representation for both human and computers beyond the current line-by-line
text notation. In particular, it is desirable to encode the trace of literals
throughout the plan to capture the dependencies between actions selected. The
approach of this paper is to view the actions as maps between literals and the
selected plan as a composition of those maps. The mathematical theory, called
category theory, provides the relevant structures for capturing maps, their
compositions, and maps between compositions. We employ this theory to propose
an algorithm agnostic, model-based representation for domains, problems, and
plans expressed in the commonly used planning description language, PDDL. This
category theoretic representation is accompanied by a graphical syntax in
addition to a linear notation, similar to algebraic expressions, that can be
used to infer literals used at every step of the plan. This provides the
appropriate constructive abstraction and facilitates comprehension for human
operators. In this paper, we demonstrate this on a plan within the Blocksworld
domain.

    

### [[2107.05877] GA and ILS for optimizing the size of NFA models](http://arxiv.org/abs/2107.05877)


  Grammatical inference consists in learning a formal grammar (as a set of
rewrite rules or a finite state machine). We are concerned with learning
Nondeterministic Finite Automata (NFA) of a given size from samples of positive
and negative words. NFA can naturally be modeled in SAT. The standard model [1]
being enormous, we also try a model based on prefixes [2] which generates
smaller instances. We also propose a new model based on suffixes and a hybrid
model based on prefixes and suffixes. We then focus on optimizing the size of
generated SAT instances issued from the hybrid models. We present two
techniques to optimize this combination, one based on Iterated Local Search
(ILS), the second one based on Genetic Algorithm (GA). Optimizing the
combination significantly reduces the SAT instances and their solving time, but
at the cost of longer generation time. We, therefore, study the balance between
generation time and solving time thanks to some experimental comparisons, and
we analyze our various model improvements.

    

### [[2107.05904] Region attention and graph embedding network for occlusion objective class-based micro-expression recognition](http://arxiv.org/abs/2107.05904)


  Micro-expression recognition (\textbf{MER}) has attracted lots of
researchers' attention in a decade. However, occlusion will occur for MER in
real-world scenarios. This paper deeply investigates an interesting but
unexplored challenging issue in MER, \ie, occlusion MER. First, to research MER
under real-world occlusion, synthetic occluded micro-expression databases are
created by using various mask for the community. Second, to suppress the
influence of occlusion, a \underline{R}egion-inspired \underline{R}elation
\underline{R}easoning \underline{N}etwork (\textbf{RRRN}) is proposed to model
relations between various facial regions. RRRN consists of a backbone network,
the Region-Inspired (\textbf{RI}) module and Relation Reasoning (\textbf{RR})
module. More specifically, the backbone network aims at extracting feature
representations from different facial regions, RI module computing an adaptive
weight from the region itself based on attention mechanism with respect to the
unobstructedness and importance for suppressing the influence of occlusion, and
RR module exploiting the progressive interactions among these regions by
performing graph convolutions. Experiments are conducted on handout-database
evaluation and composite database evaluation tasks of MEGC 2018 protocol.
Experimental results show that RRRN can significantly explore the importance of
facial regions and capture the cooperative complementary relationship of facial
regions for MER. The results also demonstrate RRRN outperforms the
state-of-the-art approaches, especially on occlusion, and RRRN acts more robust
to occlusion.

    

### [[2107.05944] The Piano Inpainting Application](http://arxiv.org/abs/2107.05944)


  Autoregressive models are now capable of generating high-quality minute-long
expressive MIDI piano performances. Even though this progress suggests new
tools to assist music composition, we observe that generative algorithms are
still not widely used by artists due to the limited control they offer,
prohibitive inference times or the lack of integration within musicians'
workflows. In this work, we present the Piano Inpainting Application (PIA), a
generative model focused on inpainting piano performances, as we believe that
this elementary operation (restoring missing parts of a piano performance)
encourages human-machine interaction and opens up new ways to approach music
composition. Our approach relies on an encoder-decoder Linear Transformer
architecture trained on a novel representation for MIDI piano performances
termed Structured MIDI Encoding. By uncovering an interesting synergy between
Linear Transformers and our inpainting task, we are able to efficiently inpaint
contiguous regions of a piano performance, which makes our model suitable for
interactive and responsive A.I.-assisted composition. Finally, we introduce our
freely-available Ableton Live PIA plugin, which allows musicians to smoothly
generate or modify any MIDI clip using PIA within a widely-used professional
Digital Audio Workstation.

    

### [[2107.05946] HAT: Hierarchical Aggregation Transformers for Person Re-identification](http://arxiv.org/abs/2107.05946)


  Recently, with the advance of deep Convolutional Neural Networks (CNNs),
person Re-Identification (Re-ID) has witnessed great success in various
applications. However, with limited receptive fields of CNNs, it is still
challenging to extract discriminative representations in a global view for
persons under non-overlapped cameras. Meanwhile, Transformers demonstrate
strong abilities of modeling long-range dependencies for spatial and sequential
data. In this work, we take advantages of both CNNs and Transformers, and
propose a novel learning framework named Hierarchical Aggregation Transformer
(HAT) for image-based person Re-ID with high performance. To achieve this goal,
we first propose a Deeply Supervised Aggregation (DSA) to recurrently aggregate
hierarchical features from CNN backbones. With multi-granularity supervisions,
the DSA can enhance multi-scale features for person retrieval, which is very
different from previous methods. Then, we introduce a Transformer-based Feature
Calibration (TFC) to integrate low-level detail information as the global prior
for high-level semantic information. The proposed TFC is inserted to each level
of hierarchical features, resulting in great performance improvements. To our
best knowledge, this work is the first to take advantages of both CNNs and
Transformers for image-based person Re-ID. Comprehensive experiments on four
large-scale Re-ID benchmarks demonstrate that our method shows better results
than several state-of-the-art methods. The code is released at
this https URL.

    

### [[2107.05949] Q-SMASH: Q-Learning-based Self-Adaptation of Human-Centered Internet of Things](http://arxiv.org/abs/2107.05949)


  As the number of Human-Centered Internet of Things (HCIoT) applications
increases, the self-adaptation of its services and devices is becoming a
fundamental requirement for addressing the uncertainties of the environment in
decision-making processes. Self-adaptation of HCIoT aims to manage run-time
changes in a dynamic environment and to adjust the functionality of IoT objects
in order to achieve desired goals during execution. SMASH is a semantic-enabled
multi-agent system for self-adaptation of HCIoT that autonomously adapts IoT
objects to uncertainties of their environment. SMASH addresses the
self-adaptation of IoT applications only according to the human values of
users, while the behavior of users is not addressed. This article presents
Q-SMASH: a multi-agent reinforcement learning-based approach for
self-adaptation of IoT objects in human-centered environments. Q-SMASH aims to
learn the behaviors of users along with respecting human values. The learning
ability of Q-SMASH allows it to adapt itself to the behavioral change of users
and make more accurate decisions in different states and situations.

    

### [[2107.05992] Identifying Influential Users in Unknown Social Networks for Adaptive Incentive Allocation Under Budget Restriction](http://arxiv.org/abs/2107.05992)


  In recent years, recommendation systems have been widely applied in many
domains. These systems are impotent in affecting users to choose the behavior
that the system expects. Meanwhile, providing incentives has been proven to be
a more proactive way to affect users' behaviors. Due to the budget limitation,
the number of users who can be incentivized is restricted. In this light, we
intend to utilize social influence existing among users to enhance the effect
of incentivization. Through incentivizing influential users directly, their
followers in the social network are possibly incentivized indirectly. However,
in many real-world scenarios, the topological structure of the network is
usually unknown, which makes identifying influential users difficult. To tackle
the aforementioned challenges, in this paper, we propose a novel algorithm for
exploring influential users in unknown networks, which can estimate the
influential relationships among users based on their historical behaviors and
without knowing the topology of the network. Meanwhile, we design an adaptive
incentive allocation approach that determines incentive values based on users'
preferences and their influence ability. We evaluate the performance of the
proposed approaches by conducting experiments on both synthetic and real-world
datasets. The experimental results demonstrate the effectiveness of the
proposed approaches.

    

### [[2107.06015] A Classification of Artificial Intelligence Systems for Mathematics Education](http://arxiv.org/abs/2107.06015)


  This chapter provides an overview of the different Artificial Intelligence
(AI) systems that are being used in contemporary digital tools for Mathematics
Education (ME). It is aimed at researchers in AI and Machine Learning (ML), for
whom we shed some light on the specific technologies that are being used in
educational applications; and at researchers in ME, for whom we clarify: i)
what the possibilities of the current AI technologies are, ii) what is still
out of reach and iii) what is to be expected in the near future. We start our
analysis by establishing a high-level taxonomy of AI tools that are found as
components in digital ME applications. Then, we describe in detail how these AI
tools, and in particular ML, are being used in two key applications,
specifically AI-based calculators and intelligent tutoring systems. We finish
the chapter with a discussion about student modeling systems and their
relationship to artificial general intelligence.

    

### [[2107.06018] This Person (Probably) Exists. Identity Membership Attacks Against GAN Generated Faces](http://arxiv.org/abs/2107.06018)


  Recently, generative adversarial networks (GANs) have achieved stunning
realism, fooling even human observers. Indeed, the popular tongue-in-cheek
website {\small \url{ this http URL}}, taunts users with
GAN generated images that seem too real to believe. On the other hand, GANs do
leak information about their training data, as evidenced by membership attacks
recently demonstrated in the literature. In this work, we challenge the
assumption that GAN faces really are novel creations, by constructing a
successful membership attack of a new kind. Unlike previous works, our attack
can accurately discern samples sharing the same identity as training samples
without being the same samples. We demonstrate the interest of our attack
across several popular face datasets and GAN training procedures. Notably, we
show that even in the presence of significant dataset diversity, an over
represented person can pose a privacy concern.

    

### [[2107.06031] Understanding Factors Affecting Fuel Consumption of Vehicles Through Explainable AI: A Use Case With Explainable Boosting Machines](http://arxiv.org/abs/2107.06031)


  A significant economic cost for many companies that operate with fleets of
vehicles is related to their fuel consumption. This consumption can be reduced
by acting over some aspects, such as the driving behaviour style of vehicle
drivers. Improving driving behaviour (and other features) can save fuel on a
fleet of vehicles without needing to change other aspects, such as the planned
routes or stops. This is important not only for mitigating economic costs
within a company, but also for reducing the emissions associated to fuel
consumption, mainly when the vehicles have petrol or diesel engines. In this
paper we show how Explainable Artificial Intelligence (XAI) can be useful for
quantifying the impact that different feature groups have on the fuel
consumption of a particular fleet. For that, we use Explainable Boosting
Machines (EBM) that are trained over different features (up to 70) in order to
first model the relationship between them and the fuel consumption, and then
explain it. With it, we compare the explanations provided by the EBM with
general references from the literature that estimate the potential impact that
those features may have on the fuel consumption, in order to validate this
approach. We work with several real-world industry datasets that represent
different types of fleets, from ones that have passenger cars to others that
include heavy-duty vehicles such as trucks.

    

### [[2107.06054] Parallelisable Existential Rules: a Story of Pieces](http://arxiv.org/abs/2107.06054)


  In this paper, we consider existential rules, an expressive formalism well
suited to the representation of ontological knowledge and data-to-ontology
mappings in the context of ontology-based data integration. The chase is a
fundamental tool to do reasoning with existential rules as it computes all the
facts entailed by the rules from a database instance. We introduce
parallelisable sets of existential rules, for which the chase can be computed
in a single breadth-first step from any instance. The question we investigate
is the characterization of such rule sets. We show that parallelisable rule
sets are exactly those rule sets both bounded for the chase and belonging to a
novel class of rules, called pieceful. The pieceful class includes in
particular frontier-guarded existential rules and (plain) datalog. We also give
another characterization of parallelisable rule sets in terms of rule
composition based on rewriting.

    

### [[2107.06056] Indian Legal NLP Benchmarks : A Survey](http://arxiv.org/abs/2107.06056)


  Availability of challenging benchmarks is the key to advancement of AI in a
specific field.Since Legal Text is significantly different than normal English
text, there is a need to create separate Natural Language Processing benchmarks
for Indian Legal Text which are challenging and focus on tasks specific to
Legal Systems. This will spur innovation in applications of Natural language
Processing for Indian Legal Text and will benefit AI community and Legal
fraternity. We review the existing work in this area and propose ideas to
create new benchmarks for Indian Legal Natural Language Processing.

    

### [[2107.06071] aiSTROM -- A roadmap for developing a successful AI strategy](http://arxiv.org/abs/2107.06071)


  A total of 34% of AI research and development projects fails or are
abandoned, according to a recent survey by Rackspace Technology of 1,870
companies. We propose a new strategic framework, aiSTROM, that empowers
managers to create a successful AI strategy based on a thorough literature
review. This provides a unique and integrated approach that guides managers and
lead developers through the various challenges in the implementation process.
In the aiSTROM framework, we start by identifying the top n potential projects
(typically 3-5). For each of those, seven areas of focus are thoroughly
analysed. These areas include creating a data strategy that takes into account
unique cross-departmental machine learning data requirements, security, and
legal requirements. aiSTROM then guides managers to think about how to put
together an interdisciplinary artificial intelligence (AI) implementation team
given the scarcity of AI talent. Once an AI team strategy has been established,
it needs to be positioned within the organization, either cross-departmental or
as a separate division. Other considerations include AI as a service (AIaas),
or outsourcing development. Looking at new technologies, we have to consider
challenges such as bias, legality of black-box-models, and keeping humans in
the loop. Next, like any project, we need value-based key performance
indicators (KPIs) to track and validate the progress. Depending on the
company's risk-strategy, a SWOT analysis (strengths, weaknesses, opportunities,
and threats) can help further classify the shortlisted projects. Finally, we
should make sure that our strategy includes continuous education of employees
to enable a culture of adoption. This unique and comprehensive framework offers
a valuable, literature supported, tool for managers and lead developers.

    

### [[2107.06075] A Rational Entailment for Expressive Description Logics via Description Logic Programs](http://arxiv.org/abs/2107.06075)


  Lehmann and Magidor's rational closure is acknowledged as a landmark in the
field of non-monotonic logics and it has also been re-formulated in the context
of Description Logics (DLs).
We show here how to model a rational form of entailment for expressive DLs,
such as SROIQ, providing a novel reasoning procedure that compiles a
non-monotone DL knowledge base into a description logic program (dl-program).

    

### [[2107.06083] A New Approach for Semantic Web Matching](http://arxiv.org/abs/2107.06083)


  In this work we propose a new approach for semantic web matching to improve
the performance of Web Service replacement. Because in automatic systems we
should ensure the self-healing, self-configuration, self-optimization and
self-management, all services should be always available and if one of them
crashes, it should be replaced with the most similar one. Candidate services
are advertised in Universal Description, Discovery and Integration (UDDI) all
in Web Ontology Language (OWL). By the help of bipartite graph, we did the
matching between the crashed service and a Candidate one. Then we chose the
best service, which had the maximum rate of matching. In fact we compare two
services` functionalities and capabilities to see how much they match. We found
that the best way for matching two web services, is comparing the
functionalities of them.

    

### [[2107.06132] Deep learning approaches to Earth Observation change detection](http://arxiv.org/abs/2107.06132)


  The interest for change detection in the field of remote sensing has
increased in the last few years. Searching for changes in satellite images has
many useful applications, ranging from land cover and land use analysis to
anomaly detection. In particular, urban change detection provides an efficient
tool to study urban spread and growth through several years of observation. At
the same time, change detection is often a computationally challenging and
time-consuming task, which requires innovative methods to guarantee optimal
results with unquestionable value and within reasonable time. In this paper we
present two different approaches to change detection (semantic segmentation and
classification) that both exploit convolutional neural networks to achieve good
results, which can be further refined and used in a post-processing workflow
for a large variety of applications.

    

### [[2107.06146] Ontology-Based Process Modelling -- Will we live to see it?](http://arxiv.org/abs/2107.06146)


  In theory, ontology-based process modelling (OBPM) bares great potential to
extend business process management. Many works have studied OBPM and are clear
on the potential amenities, such as eliminating ambiguities or enabling
advanced reasoning over company processes. However, despite this approval in
academia, a widespread industry adoption is still nowhere to be seen. This can
be mainly attributed to the fact, that it still requires high amounts of manual
labour to initially create ontologies and annotations to process models. As
long as these problems are not addressed, implementing OBPM seems unfeasible in
practice. In this work, we therefore identify requirements needed for a
successful implementation of OBPM and assess the current state of research
w.r.t. these requirements. Our results indicate that the research progress for
means to facilitate OBPM are still alarmingly low and there needs to be urgent
work on extending existing approaches.

    

### [[2107.06243] Fairness-aware Summarization for Justified Decision-Making](http://arxiv.org/abs/2107.06243)


  In many applications such as recidivism prediction, facility inspection, and
benefit assignment, it's important for individuals to know the
decision-relevant information for the model's prediction. In addition, the
model's predictions should be fairly justified. Essentially, decision-relevant
features should provide sufficient information for the predicted outcome and
should be independent of the membership of individuals in protected groups such
as race and gender. In this work, we focus on the problem of (un)fairness in
the justification of the text-based neural models. We tie the explanatory power
of the model to fairness in the outcome and propose a fairness-aware
summarization mechanism to detect and counteract the bias in such models. Given
a potentially biased natural language explanation for a decision, we use a
multi-task neural model and an attribution mechanism based on integrated
gradients to extract the high-utility and discrimination-free justifications in
the form of a summary. The extracted summary is then used for training a model
to make decisions for individuals. Results on several real-world datasets
suggests that our method: (i) assists users to understand what information is
used for the model's decision and (ii) enhances the fairness in outcomes while
significantly reducing the demographic leakage.

    

### [[2005.09253] Safe Learning for Near Optimal Scheduling](http://arxiv.org/abs/2005.09253)


  In this paper, we investigate the combination of synthesis, model-based
learning, and online sampling techniques to obtain safe and near-optimal
schedulers for a preemptible task scheduling problem. Our algorithms can handle
Markov decision processes (MDPs) that have 1020 states and beyond which cannot
be handled with state-of-the art probabilistic model-checkers. We provide
probably approximately correct (PAC) guarantees for learning the model.
Additionally, we extend Monte-Carlo tree search with advice, computed using
safety games or obtained using the earliest-deadline-first scheduler, to safely
explore the learned model online. Finally, we implemented and compared our
algorithms empirically against shielded deep Q-learning on large task systems.

    

### [[2012.06154] ParsiNLU: A Suite of Language Understanding Challenges for Persian](http://arxiv.org/abs/2012.06154)


  Despite the progress made in recent years in addressing natural language
understanding (NLU) challenges, the majority of this progress remains to be
concentrated on resource-rich languages like English. This work focuses on
Persian language, one of the widely spoken languages in the world, and yet
there are few NLU datasets available for this rich language. The availability
of high-quality evaluation datasets is a necessity for reliable assessment of
the progress on different NLU tasks and domains. We introduce ParsiNLU, the
first benchmark in Persian language that includes a range of high-level tasks
-- Reading Comprehension, Textual Entailment, etc. These datasets are collected
in a multitude of ways, often involving manual annotations by native speakers.
This results in over 14.5$k$ new instances across 6 distinct NLU tasks.
Besides, we present the first results on state-of-the-art monolingual and
multi-lingual pre-trained language-models on this benchmark and compare them
with human performance, which provides valuable insights into our ability to
tackle natural language understanding challenges in Persian. We hope ParsiNLU
fosters further research and advances in Persian language understanding.

    

### [[2103.00355] SUM: A Benchmark Dataset of Semantic Urban Meshes](http://arxiv.org/abs/2103.00355)


  Recent developments in data acquisition technology allow us to collect 3D
texture meshes quickly. Those can help us understand and analyse the urban
environment, and as a consequence are useful for several applications like
spatial analysis and urban planning. Semantic segmentation of texture meshes
through deep learning methods can enhance this understanding, but it requires a
lot of labelled data. The contributions of this work are threefold: (1) a new
benchmark dataset of semantic urban meshes, (2) a novel semi-automatic
annotation framework, and (3) an annotation tool for 3D meshes. In particular,
our dataset covers about 4 km2 in Helsinki (Finland), with six classes, and we
estimate that we save about 600 hours of labelling work using our annotation
framework, which includes initial segmentation and interactive refinement. We
also compare the performance of several state-of-theart 3D semantic
segmentation methods on the new benchmark dataset. Other researchers can use
our results to train their networks: the dataset is publicly available, and the
annotation tool is released as open-source.

    

### [[2103.04167] Imbalance-Aware Self-Supervised Learning for 3D Radiomic Representations](http://arxiv.org/abs/2103.04167)


  Radiomic representations can quantify properties of regions of interest in
medical image data. Classically, they account for pre-defined statistics of
shape, texture, and other low-level image features. Alternatively, deep
learning-based representations are derived from supervised learning but require
expensive annotations from experts and often suffer from overfitting and data
imbalance issues. In this work, we address the challenge of learning
representations of 3D medical images for an effective quantification under data
imbalance. We propose a \emph{self-supervised} representation learning
framework to learn high-level features of 3D volumes as a complement to
existing radiomics features. Specifically, we demonstrate how to learn image
representations in a self-supervised fashion using a 3D Siamese network. More
importantly, we deal with data imbalance by exploiting two unsupervised
strategies: a) sample re-weighting, and b) balancing the composition of
training batches. When combining our learned self-supervised feature with
traditional radiomics, we show significant improvement in brain tumor
classification and lung cancer staging tasks covering MRI and CT imaging
modalities.

    

### [[2103.05661] On complementing end-to-end human behavior predictors with planning](http://arxiv.org/abs/2103.05661)


  High capacity end-to-end approaches for human motion (behavior) prediction
have the ability to represent subtle nuances in human behavior, but struggle
with robustness to out of distribution inputs and tail events. Planning-based
prediction, on the other hand, can reliably output decent-but-not-great
predictions: it is much more stable in the face of distribution shift (as we
verify in this work), but it has high inductive bias, missing important aspects
that drive human decisions, and ignoring cognitive biases that make human
behavior suboptimal. In this work, we analyze one family of approaches that
strive to get the best of both worlds: use the end-to-end predictor on common
cases, but do not rely on it for tail events / out-of-distribution inputs --
switch to the planning-based predictor there. We contribute an analysis of
different approaches for detecting when to make this switch, using an
autonomous driving domain. We find that promising approaches based on
ensembling or generative modeling of the training distribution might not be
reliable, but that there very simple methods which can perform surprisingly
well -- including training a classifier to pick up on tell-tale issues in
predicted trajectories.

    

### [[2104.01939] NQMIX: Non-monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2104.01939)


  Multi-agent value-based approaches recently make great progress, especially
value decomposition methods. However, there are still a lot of limitations in
value function factorization. In VDN, the joint action-value function is the
sum of per-agent action-value function while the joint action-value function of
QMIX is the monotonic mixing of per-agent action-value function. To some
extent, QTRAN reduces the limitation of joint action-value functions that can
be represented, but it has unsatisfied performance in complex tasks. In this
paper, in order to extend the class of joint value functions that can be
represented, we propose a novel actor-critic method called NQMIX. NQMIX
introduces an off-policy policy gradient on QMIX and modify its network
architecture, which can remove the monotonicity constraint of QMIX and
implement a non-monotonic value function factorization for the joint
action-value function. In addition, NQMIX takes the state-value as the learning
target, which overcomes the problem in QMIX that the learning target is
overestimated. Furthermore, NQMIX can be extended to continuous action space
settings by introducing deterministic policy gradient on itself. Finally, we
evaluate our actor-critic methods on SMAC domain, and show that it has a
stronger performance than COMA and QMIX on complex maps with heterogeneous
agent types. In addition, our ablation results show that our modification of
mixer is effective.

    

### [[2106.13052] Autonomous Driving Strategies at Intersections: Scenarios, State-of-the-Art, and Future Outlooks](http://arxiv.org/abs/2106.13052)


  Due to the complex and dynamic character of intersection scenarios, the
autonomous driving strategy at intersections has been a difficult problem and a
hot point in the research of intelligent transportation systems in recent
years. This paper gives a brief summary of state-of-the-art autonomous driving
strategies at intersections. Firstly, we enumerate and analyze common types of
intersection scenarios, corresponding simulation platforms, as well as related
datasets. Secondly, by reviewing previous studies, we have summarized
characteristics of existing autonomous driving strategies and classified them
into several categories. Finally, we point out problems of the existing
autonomous driving strategies and put forward several valuable research
outlooks.

    

### [[2107.05681] CFM: SIMT Thread Divergence Reduction by Melding Similar Control-Flow Regions in GPGPU Programs](http://arxiv.org/abs/2107.05681)


  GPGPUs use the Single-Instruction-Multiple-Thread (SIMT) execution model
where a group of threads--wavefront or war--execute instructions in lockstep.
When threads in a group encounter a branching instruction, not all threads in
the group take the same path, a phenomenon known as control-flow divergence.
The control-flow divergence causes performance degradation because both paths
of the branch must be executed one after the other. Prior research has
primarily addressed this issue through architectural modifications. We observe
that certain GPGPU kernels with control-flow divergence have similar
control-flow structures with similar instructions on both sides of a branch.
This structure can be exploited to reduce control-flow divergence by melding
the two sides of the branch allowing threads to reconverge early, reducing
divergence. In this work, we present CFM, a compiler analysis and
transformation framework that can meld divergent control-flow structures with
similar instruction sequences. We show that CFM can reduce the performance
degradation from control-flow divergence.

    

### [[2107.06127] On the impact of Performance Antipatterns in multi-objective software model refactoring optimization](http://arxiv.org/abs/2107.06127)


  Software quality estimation is a challenging and time-consuming activity, and
models are crucial to face the complexity of such activity on modern software
applications.
One main challenge is that the improvement of distinctive quality attributes
may require contrasting refactoring actions on an application, as for trade-off
between performance and reliability. In such cases, multi-objective
optimization can provide the designer with a wider view on these trade-offs
and, consequently, can lead to identify suitable actions that take into account
independent or even competing objectives.
In this paper, we present an approach that exploits the NSGA-II
multi-objective evolutionary algorithm to search optimal Pareto solution
frontiers for software refactoring while considering as objectives: i)
performance variation, ii) reliability, iii) amount of performance
antipatterns, and iv) architectural distance. The algorithm combines randomly
generated refactoring actions into solutions (i.e., sequences of actions) and
compares them according to the objectives.
We have applied our approach on a train ticket booking service case study,
and we have focused the analysis on the impact of performance antipatterns on
the quality of solutions. Indeed, we observe that the approach finds better
solutions when antipatterns enter the multi-objective optimization. In
particular, performance antipatterns objective leads to solutions improving the
performance by up to 15% with respect to the case where antipatterns are not
considered, without affecting the solution quality on other objectives.

    

### [[2107.05679] Teaching Design by Contract using Snap!](http://arxiv.org/abs/2107.05679)


  With the progress in deductive program verification research, new tools and
techniques have become available to support design-by-contract reasoning about
non-trivial programs written in widely-used programming languages. However,
deductive program verification remains an activity for experts, with ample
experience in programming, specification and verification. We would like to
change this situation, by developing program verification techniques that are
available to a larger audience. In this paper, we present how we developed
prototypal program verification support for Snap!. Snap! is a visual
programming language, aiming in particular at high school students. We added
specification language constructs in a similar visual style, designed to make
the intended semantics clear from the look and feel of the specification
constructs. We provide support both for static and dynamic verification of
Snap! programs. Special attention is given to the error messaging, to make this
as intuitive as possible.

    

### [[2107.06253] Bottom-up Synthesis of Recursive Functional Programs using Angelic Execution](http://arxiv.org/abs/2107.06253)


  We present a novel bottom-up method for the synthesis of functional recursive
programs. While bottom-up synthesis techniques can work better than top-down
methods in certain settings, there is no prior technique for synthesizing
recursive programs from logical specifications in a purely bottom-up fashion.
The main challenge is that effective bottom-up methods need to execute
sub-expressions of the code being synthesized, but it is impossible to execute
a recursive subexpression of a program that has not been fully constructed yet.
In this paper, we address this challenge using the concept of angelic
semantics. Specifically, our method finds a program that satisfies the
specification under angelic semantics (we refer to this as angelic synthesis),
analyzes the assumptions made during its angelic execution, uses this analysis
to strengthen the specification, and finally reattempts synthesis with the
strengthened specification. Our proposed angelic synthesis algorithm is based
on version space learning and therefore deals effectively with many incremental
synthesis calls made during the overall algorithm. We have implemented this
approach in a prototype called Burst and evaluate it on synthesis problems from
prior work. Our experiments show that Burst is able to synthesize a solution to
95% of the benchmarks in our benchmark suite, outperforming prior work.

    