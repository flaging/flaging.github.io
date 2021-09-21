
## 2021-9-21

### [[2109.08783] From the Beginning: Key Transitions in the First 15 Years of DNSSEC](http://arxiv.org/abs/2109.08783)


  When the global rollout of the DNS Security Extensions (DNSSEC) began in
2005, it started a first-of-its-kind trial: increasing complexity of a core
Internet protocol in favor of better security for the overall Internet. The
necessary cryptographic key management is made particularly challenging by DNS'
loosely-federated delegation substrate and unprecedented cryptographic scale.
Though fundamental for current and future operational success, our community
lacks a clear notion of how to empirically evaluate the process of securely
changing (or transitioning) keys.
In this paper, we propose two building blocks to fundamentally understand and
assess key transitions. First, the anatomy of key transitions: measurable and
well-defined properties of key changes; and second a novel classification model
based on this anatomy to describe key transitions practices in abstract terms.
Our anatomy enables the evaluation of cryptographic keys' life cycles in
general, and comparison of operational practices with prescribed key management
processes, e.g., RFC key rollover guidelines. The fine-grained transition
anatomy is then abstracted through our classification model to characterize
transitions in abstract terms which rather describe a transition's behavior
than its specific features.
The applicability and utility of our proposed transition anatomy and
transition classes are exemplified for the global DNSSEC deployment.
Specifically, we use measurements from the first 15 years of the DNSSEC rollout
to detect and measure which key rollover/transitions have been used, to what
degree, and what their rates of errors and warnings have been. Our results show
measurable gaps between prescribed key management processes and key transitions
in the wild. We also find evidence that such noncompliant transitions are
inevitable in the wild.

    

### [[2109.08976] A Tutorial on Trusted and Untrusted non-3GPP Accesses in 5G Systems -- First Steps Towards a Unified Communications Infrastructure](http://arxiv.org/abs/2109.08976)


  Fifth-generation (5G) systems are designed to enable convergent
access-agnostic service availability. This means that 5G services will be
available over 5G New Radio air interface and also through other non-Third
Generation Partnership Project (3GPP) access networks, e.g., IEEE 802.11
(Wi-Fi). 3GPP has recently published the Release 16 that includes trusted
non-3GPP access network concept and wireless wireline convergence. The main
goal of this tutorial is to present an overview of access to 5G core via
non-3GPP access networks specified by 3GPP until Release 16 (i.e., untrusted,
trusted, and wireline access). The tutorial describes aspects of the
convergence of a 5G system and these non-3GPP access networks, such as the
authentication and authorization procedures and the data session establishment
from the point of view of protocol stack and exchanged messages between the
network functions. In order to illustrate several concepts and part of 3GPP
specification, we present a basic but fully operational implementation of
untrusted non-3GPP access using WLAN. We perform experiments that demonstrate
how a Wi-Fi user is authorized in a 5G core and establishes user plane
connectivity to a data network. Moreover, we evaluate the performance of this
access in terms of time consumed, number of messages, and protocol overhead to
established data sessions.

    

### [[2109.08989] Passive Optical Networking for 5G and Beyond 5G Low-Latency Mobile Fronthauling Services](http://arxiv.org/abs/2109.08989)


  Passive optical network (PON) technology offers an attractive cost-efficient
alternative to support 5G and Beyond 5G mobile network fronthauling (MFH).
However, MFH for such networks is challenging given its high bandwidth and
strict latency requirements. To reduce these requirements, radio access network
functional splitting has been introduced in 5G networks; this provides more
flexibility in resource allocation since the protocol stack is distributed
between the centralized and the distributed units. In contrast to the
conventional MFH requirement of physical-layer splittings, the MFH traffic
produced by link-layer splittings becomes more dependent on the actual user
traffic load. By capitalizing on the characteristics of the new MFH traffic
with functional splitting, this article introduces a mechanism to improve the
performance of PONs serving MFH.

    

### [[2109.09015] Distributed Joint Power and Rate Control for NOMA/OFDMA in 5G and Beyond](http://arxiv.org/abs/2109.09015)


  In this paper, we study the problem of minimizing the uplink aggregate
transmit power subject to the users' minimum data rate and peak power
constraint on each sub-channel for multi-cell wireless networks. To address
this problem, a distributed sub-optimal joint power and rate control algorithm
called JPRC is proposed, which is applicable to both non-orthogonal
frequency-division multiple access (NOMA) and orthogonal frequency-division
multiple access (OFDMA) schemes. Employing JPRC, each user updates its transmit
power using only local information. Simulation results illustrate that the JPRC
algorithm can reach a performance close to that obtained by the optimal
solution via exhaustive search, with the NOMA scheme achieving a 59\%
improvement on the aggregate transmit power over the OFDMA counterpart. It is
also shown that the JPRC algorithm can outperform existing distributed power
control algorithms.

    

### [[2109.09132] Secrecy Capacity and Energy Efficiency of Spectrum Sharing Networks Incorporating Physical Layer Security](http://arxiv.org/abs/2109.09132)


  Underutilized wireless channel is a waste of spectral resource. Eavesdropping
compromises data secrecy. How to overcome the two problems with one solution?
In this paper, we propose a spectrum sharing model that defends against
eavesdropping. Consider a source-destination channel that is being
eavesdropped. Cognitive radio can help jamming the eavesdropper. Friendly
jamming is a physical layer security method of protecting data secrecy based on
radio propagation characteristic. In return, the helper is allowed to access
the unused channel of the source. The desirable characteristic of cognitive
radio is its capability of sensing the occupancy or vacancy of the channel.
This work investigates the secrecy capacity $C_S$ and energy efficiency $\mu$
of such a spectrum sharing network that deploys physical layer security method.
The main factors that affect $C_S$ and $\mu$ are the transmit powers of the
source and cognitive radio. We present a novel expression that permits finding
the sensing duration $t$ that optimizes $\mu$.

    

### [[2103.03288] The Effect of Network Topology on Credit Network Throughput](http://arxiv.org/abs/2103.03288)


  Credit networks rely on decentralized, pairwise trust relationships
(channels) to exchange money or goods. Credit networks arise naturally in many
financial systems, including the recent construct of payment channel networks
in blockchain systems. An important performance metric for these networks is
their transaction throughput. However, predicting the throughput of a credit
network is nontrivial. Unlike traditional communication channels, credit
channels can become imbalanced; they are unable to support more transactions in
a given direction once the credit limit has been reached. This potential for
imbalance creates a complex dependency between a network's throughput and its
topology, path choices, and the credit balances (state) on every channel. Even
worse, certain combinations of these factors can lead the credit network to
deadlocked states where no transactions can make progress. In this paper, we
study the relationship between the throughput of a credit network and its
topology and credit state. We show that the presence of deadlocks completely
characterizes a network's throughput sensitivity to different credit states.
Although we show that identifying deadlocks in an arbitrary topology is
NP-hard, we propose a peeling algorithm inspired by decoding algorithms for
erasure codes that upper bounds the severity of the deadlock. We use the
peeling algorithm as a tool to compare the performance of different topologies
as well as to aid in the synthesis of topologies robust to deadlocks.

    

### [[2109.08684] Asymmetric 3D Context Fusion for Universal Lesion Detection](http://arxiv.org/abs/2109.08684)


  Modeling 3D context is essential for high-performance 3D medical image
analysis. Although 2D networks benefit from large-scale 2D supervised
pretraining, it is weak in capturing 3D context. 3D networks are strong in 3D
context yet lack supervised pretraining. As an emerging technique, \emph{3D
context fusion operator}, which enables conversion from 2D pretrained networks,
leverages the advantages of both and has achieved great success. Existing 3D
context fusion operators are designed to be spatially symmetric, i.e.,
performing identical operations on each 2D slice like convolutions. However,
these operators are not truly equivariant to translation, especially when only
a few 3D slices are used as inputs. In this paper, we propose a novel
asymmetric 3D context fusion operator (A3D), which uses different weights to
fuse 3D context from different 2D slices. Notably, A3D is NOT
translation-equivariant while it significantly outperforms existing symmetric
context fusion operators without introducing large computational overhead. We
validate the effectiveness of the proposed method by extensive experiments on
DeepLesion benchmark, a large-scale public dataset for universal lesion
detection from computed tomography (CT). The proposed A3D consistently
outperforms symmetric context fusion operators by considerable margins, and
establishes a new \emph{state of the art} on DeepLesion. To facilitate open
research, our code and model in PyTorch are available at
this https URL.

    

### [[2109.08685] Self-supervised learning methods and applications in medical imaging analysis: A survey](http://arxiv.org/abs/2109.08685)


  The availability of high quality annotated medical imaging datasets is a
major problem that collides with machine learning applications in the field of
medical imaging analysis and impedes its advancement. Self-supervised learning
is a recent training paradigm that enables learning robust representations
without the need for human annotation which can be considered as an effective
solution for the scarcity in annotated medical data. This article reviews the
state-of-the-art research directions in self-supervised learning approaches for
image data with concentration on their applications in the field of medical
imaging analysis. The article covers a set of the most recent self-supervised
learning methods from the computer vision field as they are applicable to the
medical imaging analysis and categorize them as predictive, generative and
contrastive approaches. Moreover, the article covers (40) of the most recent
researches in the field of self-supervised learning in medical imaging analysis
aiming at shedding the light on the recent innovation in the field. Ultimately,
the article concludes with possible future research directions in the field.

    

### [[2109.08688] Segmentation of Brain MRI using an Altruistic Harris Hawks' Optimization algorithm](http://arxiv.org/abs/2109.08688)


  Segmentation is an essential requirement in medicine when digital images are
used in illness diagnosis, especially, in posterior tasks as analysis and
disease identification. An efficient segmentation of brain Magnetic Resonance
Images (MRIs) is of prime concern to radiologists due to their poor
illumination and other conditions related to de acquisition of the images.
Thresholding is a popular method for segmentation that uses the histogram of an
image to label different homogeneous groups of pixels into different classes.
However, the computational cost increases exponentially according to the number
of thresholds. In this paper, we perform the multi-level thresholding using an
evolutionary metaheuristic. It is an improved version of the Harris Hawks
Optimization (HHO) algorithm that combines the chaotic initialization and the
concept of altruism. Further, for fitness assignment, we use a hybrid objective
function where along with the cross-entropy minimization, we apply a new
entropy function, and leverage weights to the two objective functions to form a
new hybrid approach. The HHO was originally designed to solve numerical
optimization problems. Earlier, the statistical results and comparisons have
demonstrated that the HHO provides very promising results compared with
well-established metaheuristic techniques. In this article, the altruism has
been incorporated into the HHO algorithm to enhance its exploitation
capabilities. We evaluate the proposed method over 10 benchmark images from the
WBA database of the Harvard Medical School and 8 benchmark images from the
Brainweb dataset using some standard evaluation metrics.

    

### [[2109.08705] Relating Neural Text Degeneration to Exposure Bias](http://arxiv.org/abs/2109.08705)


  This work focuses on relating two mysteries in neural-based text generation:
exposure bias, and text degeneration. Despite the long time since exposure bias
was mentioned and the numerous studies for its remedy, to our knowledge, its
impact on text generation has not yet been verified. Text degeneration is a
problem that the widely-used pre-trained language model GPT-2 was recently
found to suffer from (Holtzman et al., 2020). Motivated by the unknown
causation of the text degeneration, in this paper we attempt to relate these
two mysteries. Specifically, we first qualitatively quantitatively identify
mistakes made before text degeneration occurs. Then we investigate the
significance of the mistakes by inspecting the hidden states in GPT-2. Our
results show that text degeneration is likely to be partly caused by exposure
bias. We also study the self-reinforcing mechanism of text degeneration,
explaining why the mistakes amplify. In sum, our study provides a more concrete
foundation for further investigation on exposure bias and text degeneration
problems.

    

### [[2109.08711] Experimental Evaluation of Computational Complexity for Different Neural Network Equalizers in Optical Communications](http://arxiv.org/abs/2109.08711)


  Addressing the neural network-based optical channel equalizers, we quantify
the trade-off between their performance and complexity by carrying out the
comparative analysis of several neural network architectures, presenting the
results for TWC and SSMF set-ups.

    

### [[2109.08712] Back-translation for Large-Scale Multilingual Machine Translation](http://arxiv.org/abs/2109.08712)


  This paper illustrates our approach to the shared task on large-scale
multilingual machine translation in the sixth conference on machine translation
(WMT-21). This work aims to build a single multilingual translation system with
a hypothesis that a universal cross-language representation leads to better
multilingual translation performance. We extend the exploration of different
back-translation methods from bilingual translation to multilingual
translation. Better performance is obtained by the constrained sampling method,
which is different from the finding of the bilingual translation. Besides, we
also explore the effect of vocabularies and the amount of synthetic data.
Surprisingly, the smaller size of vocabularies perform better, and the
extensive monolingual English data offers a modest improvement. We submitted to
both the small tasks and achieved the second place.

    

### [[2109.08717] The Optimization of the Constant Flow Parallel Micropump Using RBF Neural Network](http://arxiv.org/abs/2109.08717)


  The objective of this work is to optimize the performance of a constant flow
parallel mechanical displacement micropump, which has parallel pump chambers
and incorporates passive check valves. The critical task is to minimize the
pressure pulse caused by regurgitation, which negatively impacts the constant
flow rate, during the reciprocating motion when the left and right pumps
interchange their role of aspiration and transfusion. Previous works attempt to
solve this issue via the mechanical design of passive check valves. In this
work, the novel concept of overlap time is proposed, and the issue is solved
from the aspect of control theory by implementing a RBF neural network trained
by both unsupervised and supervised learning. The experimental results indicate
that the pressure pulse is optimized in the range of 0.15 - 0.25 MPa, which is
a significant improvement compared to the maximum pump working pressure of 40
MPa.

    

### [[2109.08718] Proteome-informed machine learning studies of cocaine addiction](http://arxiv.org/abs/2109.08718)


  Cocaine addiction accounts for a large portion of substance use disorders and
threatens millions of lives worldwide. There is an urgent need to come up with
efficient anti-cocaine addiction drugs. Unfortunately, no medications have been
approved by the Food and Drug Administration (FDA), despite the extensive
effort in the past few decades. The main challenge is the intricate molecular
mechanisms of cocaine addiction, involving synergistic interactions among
proteins upstream and downstream of dopamine transporter (DAT) functions
impacted by cocaine. However, traditional in vivo or in vitro experiments can
not address the roles of so many proteins, highlighting the need for innovative
strategies in the field. We propose a proteome-informed machine learning/deep
learning (ML/DL) platform to discover nearly optimal anti-cocaine addiction
lead compounds. We construct and analyze proteomic protein-protein interaction
(PPI) networks for cocaine dependence to identify 141 involved drug targets and
represent over 60,000 associated drug candidates or experimental drugs in the
latent space using an autoencoder (EA) model trained from over 104 million
molecules. We build 32 ML models for cross-target analysis of these drug
candidates for side effects and repurposing potential. We further screen the
absorption, distribution, metabolism, excretion, and toxicity (ADMET)
properties of these candidates. Our platform reveals that essentially all of
the existing drug candidates, including dozens of experimental drugs, fail to
pass our cross-target and ADMET screenings. Nonetheless, we have identified two
nearly optimal leads for further optimization.

    

### [[2109.08722] Efficient Variational Graph Autoencoders for Unsupervised Cross-domain Prerequisite Chains](http://arxiv.org/abs/2109.08722)


  Prerequisite chain learning helps people acquire new knowledge efficiently.
While people may quickly determine learning paths over concepts in a domain,
finding such paths in other domains can be challenging. We introduce
Domain-Adversarial Variational Graph Autoencoders (DAVGAE) to solve this
cross-domain prerequisite chain learning task efficiently. Our novel model
consists of a variational graph autoencoder (VGAE) and a domain discriminator.
The VGAE is trained to predict concept relations through link prediction, while
the domain discriminator takes both source and target domain data as input and
is trained to predict domain labels. Most importantly, this method only needs
simple homogeneous graphs as input, compared with the current state-of-the-art
model. We evaluate our model on the LectureBankCD dataset, and results show
that our model outperforms recent graph-based benchmarks while using only 1/10
of graph scale and 1/3 computation time.

    

### [[2109.08735] Analyzing the Habitable Zones of Circumbinary Planets Using Machine Learning](http://arxiv.org/abs/2109.08735)


  Exoplanet detection in the past decade by efforts including NASA's Kepler and
TESS missions has discovered many worlds that differ substantially from planets
in our own Solar System, including more than 150 exoplanets orbiting binary or
multi-star systems. This not only broadens our understanding of the diversity
of exoplanets, but also promotes our study of exoplanets in the complex binary
systems and provides motivation to explore their habitability. In this study,
we investigate the Habitable Zones of circumbinary planets based on planetary
trajectory and dynamically informed habitable zones. Our results indicate that
the mass ratio and orbital eccentricity of binary stars are important factors
affecting the orbital stability and habitability of planetary systems.
Moreover, planetary trajectory and dynamically informed habitable zones divide
planetary habitability into three categories: habitable, part-habitable and
uninhabitable. Therefore, we train a machine learning model to quickly and
efficiently classify these planetary systems.

    

### [[2109.08744] Dual-Encoder Architecture with Encoder Selection for Joint Close-Talk and Far-Talk Speech Recognition](http://arxiv.org/abs/2109.08744)


  In this paper, we propose a dual-encoder ASR architecture for joint modeling
of close-talk (CT) and far-talk (FT) speech, in order to combine the advantages
of CT and FT devices for better accuracy. The key idea is to add an encoder
selection network to choose the optimal input source (CT or FT) and the
corresponding encoder. We use a single-channel encoder for CT speech and a
multi-channel encoder with Spatial Filtering neural beamforming for FT speech,
which are jointly trained with the encoder selection. We validate our approach
on both attention-based and RNN Transducer end-to-end ASR systems. The
experiments are done with conversational speech from a medical use case, which
is recorded simultaneously with a CT device and a microphone array. Our results
show that the proposed dual-encoder architecture obtains up to 9% relative WER
reduction when using both CT and FT input, compared to the best single-encoder
system trained and tested in matched condition.

    

### [[2109.08776] Exploring the Robustness of Distributional Reinforcement Learning against Noisy State Observations](http://arxiv.org/abs/2109.08776)


  In real scenarios, state observations that an agent observes may contain
measurement errors or adversarial noises, misleading the agent to take
suboptimal actions or even collapse while training. In this paper, we study the
training robustness of distributional Reinforcement Learning~(RL), a class of
state-of-the-art methods that estimate the whole distribution, as opposed to
only the expectation, of the total return. Firstly, we propose State-Noisy
Markov Decision Process~(SN-MDP) in the tabular case to incorporate both random
and adversarial state observation noises, in which the contraction of both
expectation-based and distributional Bellman operators is derived. Beyond
SN-MDP with the function approximation, we theoretically characterize the
bounded gradient norm of histogram-based distributional loss, accounting for
the better training robustness of distribution RL. We also provide stricter
convergence conditions of the Temporal-Difference~(TD) learning under more
flexible state noises, as well as the sensitivity analysis by the leverage of
influence function. Finally, extensive experiments on the suite of games show
that distributional RL enjoys better training robustness compared with its
expectation-based counterpart across various state observation noises.

    

### [[2109.08779] Capacitance Resistance Model and Recurrent Neural Network for Well Connectivity Estimation : A Comparison Study](http://arxiv.org/abs/2109.08779)


  In this report, two commonly used data-driven models for predicting well
production under a waterflood setting: the capacitance resistance model (CRM)
and recurrent neural networks (RNN) are compared. Both models are completely
data-driven and are intended to learn the reservoir behavior during a water
flood from historical data. This report serves as a technical guide to the
python-based implementation of the CRM model available from the associated
GitHub repository.

    

### [[2109.08780] Long-Range Modeling of Source Code Files with eWASH: Extended Window Access by Syntax Hierarchy](http://arxiv.org/abs/2109.08780)


  Statistical language modeling and translation with transformers have found
many successful applications in program understanding and generation tasks,
setting high benchmarks for tools in modern software development environments.
The finite context window of these neural models means, however, that they will
be unable to leverage the entire relevant context of large files and packages
for any given task. While there are many efforts to extend the context window,
we introduce an architecture-independent approach for leveraging the syntactic
hierarchies of source code for incorporating entire file-level context into a
fixed-length window. Using concrete syntax trees of each source file we extract
syntactic hierarchies and integrate them into context window by selectively
removing from view more specific, less relevant scopes for a given task. We
evaluate this approach on code generation tasks and joint translation of
natural language and source code in Python programming language, achieving a
new state-of-the-art in code completion and summarization for Python in the
CodeXGLUE benchmark. We also introduce new CodeXGLUE benchmarks for
user-experience-motivated tasks: code completion with normalized literals,
method body completion/code summarization conditioned on file-level context.

    

### [[2109.08786] A Deep-Learning Based Optimization Approach to Address Stop-Skipping Strategy in Urban Rail Transit Lines](http://arxiv.org/abs/2109.08786)


  Different passenger demand rates in transit stations underscore the
importance of adopting operational strategies to provide a demand-responsive
service. Aiming at improving passengers' travel time, the present study
introduces an advanced data-driven optimization approach to determine the
optimal stop-skip pattern in urban rail transit lines. In detail, first, using
the time-series smart card data for an entire month, we employ a Long
Short-Term Memory (LSTM) deep learning model to predict the station-level
demand rates for the peak hour. This prediction is based on four preceding
hours and is especially important knowing that the true demand rates of the
peak hour are posterior information that can be obtained only after the peak
hour operation is finished. Moreover, utilizing a real-time prediction instead
of assuming fixed demand rates, allows us to account for unexpected real-time
changes which can be detrimental to the subsequent analyses. Then, we integrate
the output of the LSTM model as an input to an optimization model with the
objective of minimizing patrons' total travel time. Considering the exponential
nature of the problem, we propose an Ant Colony Optimization technique to solve
the problem in a desirable amount of time. Finally, the performance of the
proposed models and the solution algorithm is assessed using real case data.
The results suggest that the proposed approach can enhance the performance of
the service by improving both passengers' in-vehicle time as well as
passengers' waiting time.

    

### [[2109.08791] Small Lesion Segmentation in Brain MRIs with Subpixel Embedding](http://arxiv.org/abs/2109.08791)


  We present a method to segment MRI scans of the human brain into ischemic
stroke lesion and normal tissues. We propose a neural network architecture in
the form of a standard encoder-decoder where predictions are guided by a
spatial expansion embedding network. Our embedding network learns features that
can resolve detailed structures in the brain without the need for
high-resolution training images, which are often unavailable and expensive to
acquire. Alternatively, the encoder-decoder learns global structures by means
of striding and max pooling. Our embedding network complements the
encoder-decoder architecture by guiding the decoder with fine-grained details
lost to spatial downsampling during the encoder stage. Unlike previous works,
our decoder outputs at 2 times the input resolution, where a single pixel in
the input resolution is predicted by four neighboring subpixels in our output.
To obtain the output at the original scale, we propose a learnable downsampler
(as opposed to hand-crafted ones e.g. bilinear) that combines subpixel
predictions. Our approach improves the baseline architecture by approximately
11.7% and achieves the state of the art on the ATLAS public benchmark dataset
with a smaller memory footprint and faster runtime than the best competing
method. Our source code has been made available at:
this https URL.

    

### [[2109.08792] Learning to be Fair: A Consequentialist Approach to Equitable Decision-Making](http://arxiv.org/abs/2109.08792)


  In the dominant paradigm for designing equitable machine learning systems,
one works to ensure that model predictions satisfy various fairness criteria,
such as parity in error rates across race, gender, and other legally protected
traits. That approach, however, typically divorces predictions from the
downstream outcomes they ultimately affect, and, as a result, can induce
unexpected harms. Here we present an alternative framework for fairness that
directly anticipates the consequences of actions. Stakeholders first specify
preferences over the possible outcomes of an algorithmically informed
decision-making process. For example, lenders may prefer extending credit to
those most likely to repay a loan, while also preferring similar lending rates
across neighborhoods. One then searches the space of decision policies to
maximize the specified utility. We develop and describe a method for
efficiently learning these optimal policies from data for a large family of
expressive utility functions, facilitating a more holistic approach to
equitable decision-making.

    

### [[2109.08795] An Empirical Evaluation of the t-SNE Algorithm for Data Visualization in Structural Engineering](http://arxiv.org/abs/2109.08795)


  A fundamental task in machine learning involves visualizing high-dimensional
data sets that arise in high-impact application domains. When considering the
context of large imbalanced data, this problem becomes much more challenging.
In this paper, the t-Distributed Stochastic Neighbor Embedding (t-SNE)
algorithm is used to reduce the dimensions of an earthquake engineering related
data set for visualization purposes. Since imbalanced data sets greatly affect
the accuracy of classifiers, we employ Synthetic Minority Oversampling
Technique (SMOTE) to tackle the imbalanced nature of such data set. We present
the result obtained from t-SNE and SMOTE and compare it to the basic approaches
with various aspects. Considering four options and six classification
algorithms, we show that using t-SNE on the imbalanced data and SMOTE on the
training data set, neural network classifiers have promising results without
sacrificing accuracy. Hence, we can transform the studied scientific data into
a two-dimensional (2D) space, enabling the visualization of the classifier and
the resulting decision surface using a 2D plot.

    

### [[2109.08800] A Robust and Efficient Multi-Scale Seasonal-Trend Decomposition](http://arxiv.org/abs/2109.08800)


  Many real-world time series exhibit multiple seasonality with different
lengths. The removal of seasonal components is crucial in numerous applications
of time series, including forecasting and anomaly detection. However, many
seasonal-trend decomposition algorithms suffer from high computational cost and
require a large amount of data when multiple seasonal components exist,
especially when the periodic length is long. In this paper, we propose a
general and efficient multi-scale seasonal-trend decomposition algorithm for
time series with multiple seasonality. We first down-sample the original time
series onto a lower resolution, and then convert it to a time series with
single seasonality. Thus, existing seasonal-trend decomposition algorithms can
be applied directly to obtain the rough estimates of trend and the seasonal
component corresponding to the longer periodic length. By considering the
relationship between different resolutions, we formulate the recovery of
different components on the high resolution as an optimization problem, which
is solved efficiently by our alternative direction multiplier method (ADMM)
based algorithm. Our experimental results demonstrate the accurate
decomposition results with significantly improved efficiency.

    

### [[2109.08805] BERT-Beta: A Proactive Probabilistic Approach to Text Moderation](http://arxiv.org/abs/2109.08805)


  Text moderation for user generated content, which helps to promote healthy
interaction among users, has been widely studied and many machine learning
models have been proposed. In this work, we explore an alternative perspective
by augmenting reactive reviews with proactive forecasting. Specifically, we
propose a new concept {\it text toxicity propensity} to characterize the extent
to which a text tends to attract toxic comments. Beta regression is then
introduced to do the probabilistic modeling, which is demonstrated to function
well in comprehensive experiments. We also propose an explanation method to
communicate the model decision clearly. Both propensity scoring and
interpretation benefit text moderation in a novel manner. Finally, the proposed
scaling mechanism for the linear model offers useful insights beyond this work.

    

### [[2109.08815] Probabilistic Inference of Simulation Parameters via Parallel Differentiable Simulation](http://arxiv.org/abs/2109.08815)


  To accurately reproduce measurements from the real world, simulators need to
have an adequate model of the physical system and require the parameters of the
model be identified.
We address the latter problem of estimating parameters through a Bayesian
inference approach that approximates a posterior distribution over simulation
parameters given real sensor measurements. By extending the commonly used
Gaussian likelihood model for trajectories via the multiple-shooting
formulation, our chosen particle-based inference algorithm Stein Variational
Gradient Descent is able to identify highly nonlinear, underactuated systems.
We leverage GPU code generation and differentiable simulation to evaluate the
likelihood and its gradient for many particles in parallel.
Our algorithm infers non-parametric distributions over simulation parameters
more accurately than comparable baselines and handles constraints over
parameters efficiently through gradient-based optimization. We evaluate
estimation performance on several physical experiments. On an underactuated
mechanism where a 7-DOF robot arm excites an object with an unknown mass
configuration, we demonstrate how our inference technique can identify
symmetries between the parameters and provide highly accurate predictions.
Project website: this https URL


### [[2109.08817] Learning to Regrasp by Learning to Place](http://arxiv.org/abs/2109.08817)


  In this paper, we explore whether a robot can learn to regrasp a diverse set
of objects to achieve various desired grasp poses. Regrasping is needed
whenever a robot's current grasp pose fails to perform desired manipulation
tasks. Endowing robots with such an ability has applications in many domains
such as manufacturing or domestic services. Yet, it is a challenging task due
to the large diversity of geometry in everyday objects and the high
dimensionality of the state and action space. In this paper, we propose a
system for robots to take partial point clouds of an object and the supporting
environment as inputs and output a sequence of pick-and-place operations to
transform an initial object grasp pose to the desired object grasp poses. The
key technique includes a neural stable placement predictor and a regrasp graph
based solution through leveraging and changing the surrounding environment. We
introduce a new and challenging synthetic dataset for learning and evaluating
the proposed approach. In this dataset, we show that our system is able to
achieve 73.3% success rate of regrasping diverse objects.

    

### [[2109.08819] Toward Efficient Federated Learning in Multi-Channeled Mobile Edge Network with Layerd Gradient Compression](http://arxiv.org/abs/2109.08819)


  A fundamental issue for federated learning (FL) is how to achieve optimal
model performance under highly dynamic communication environments. This issue
can be alleviated by the fact that modern edge devices usually can connect to
the edge FL server via multiple communication channels (e.g., 4G, LTE and 5G).
However, having an edge device send copies of local models to the FL server
along multiple channels is redundant, time-consuming, and would waste resources
(e.g., bandwidth, battery life and monetary cost). In this paper, motivated by
the layered coding techniques in video streaming, we propose a novel FL
framework called layered gradient compression (LGC). Specifically, in LGC,
local gradients from a device is coded into several layers and each layer is
sent to the FL server along a different channel. The FL server aggregates the
received layers of local gradients from devices to update the global model, and
sends the result back to the devices. We prove the convergence of LGC, and
formally define the problem of resource-efficient federated learning with LGC.
We then propose a learning based algorithm for each device to dynamically
adjust its local computation (i.e., the number of local stochastic descent) and
communication decisions (i.e.,the compression level of different layers and the
layer to channel mapping) in each iteration. Results from extensive experiments
show that using our algorithm, LGC significantly reduces the training time,
improves the resource utilization, while achieving a similar accuracy, compared
with well-known FL mechanisms.

    

### [[2109.08820] Towards Zero and Few-shot Knowledge-seeking Turn Detection in Task-orientated Dialogue Systems](http://arxiv.org/abs/2109.08820)


  Most prior work on task-oriented dialogue systems is restricted to supporting
domain APIs. However, users may have requests that are out of the scope of
these APIs. This work focuses on identifying such user requests. Existing
methods for this task mainly rely on fine-tuning pre-trained models on large
annotated data. We propose a novel method, REDE, based on adaptive
representation learning and density estimation. REDE can be applied to
zero-shot cases, and quickly learns a high-performing detector with only a few
shots by updating less than 3K parameters. We demonstrate REDE's competitive
performance on DSTC9 data and our newly collected test set.

    

### [[2109.08830] MM-Deacon: Multimodal molecular domain embedding analysis via contrastive learning](http://arxiv.org/abs/2109.08830)


  Molecular representation learning plays an essential role in cheminformatics.
Recently, language model-based approaches have been popular as an alternative
to traditional expert-designed features to encode molecules. However, these
approaches only utilize a single modality for representing molecules. Driven by
the fact that a given molecule can be described through different modalities
such as Simplified Molecular Line Entry System (SMILES), The International
Union of Pure and Applied Chemistry (IUPAC), and The IUPAC International
Chemical Identifier (InChI), we propose a multimodal molecular embedding
generation approach called MM-Deacon (multimodal molecular domain embedding
analysis via contrastive learning). MM-Deacon is trained using SMILES and IUPAC
molecule representations as two different modalities. First, SMILES and IUPAC
strings are encoded by using two different transformer-based language models
independently, then the contrastive loss is utilized to bring these encoded
representations from different modalities closer to each other if they belong
to the same molecule, and to push embeddings farther from each other if they
belong to different molecules. We evaluate the robustness of our molecule
embeddings on molecule clustering, cross-modal molecule search, drug similarity
assessment and drug-drug interaction tasks.

    

### [[2109.08839] SpeechNAS: Towards Better Trade-off between Latency and Accuracy for Large-Scale Speaker Verification](http://arxiv.org/abs/2109.08839)


  Recently, x-vector has been a successful and popular approach for speaker
verification, which employs a time delay neural network (TDNN) and statistics
pooling to extract speaker characterizing embedding from variable-length
utterances. Improvement upon the x-vector has been an active research area, and
enormous neural networks have been elaborately designed based on the x-vector,
eg, extended TDNN (E-TDNN), factorized TDNN (F-TDNN), and densely connected
TDNN (D-TDNN). In this work, we try to identify the optimal architectures from
a TDNN based search space employing neural architecture search (NAS), named
SpeechNAS. Leveraging the recent advances in the speaker recognition, such as
high-order statistics pooling, multi-branch mechanism, D-TDNN and angular
additive margin softmax (AAM) loss with a minimum hyper-spherical energy (MHE),
SpeechNAS automatically discovers five network architectures, from SpeechNAS-1
to SpeechNAS-5, of various numbers of parameters and GFLOPs on the large-scale
text-independent speaker recognition dataset VoxCeleb1. Our derived best neural
network achieves an equal error rate (EER) of 1.02% on the standard test set of
VoxCeleb1, which surpasses previous TDNN based state-of-the-art approaches by a
large margin. Code and trained weights are in
this https URL


### [[2109.08844] Near-Minimax Optimal Estimation With Shallow ReLU Neural Networks](http://arxiv.org/abs/2109.08844)


  We study the problem of estimating an unknown function from noisy data using
shallow (single-hidden layer) ReLU neural networks. The estimators we study
minimize the sum of squared data-fitting errors plus a regularization term
proportional to the Euclidean norm of the network weights. This minimization
corresponds to the common approach of training a neural network with weight
decay. We quantify the performance (mean-squared error) of these neural network
estimators when the data-generating function belongs to the space of functions
of second-order bounded variation in the Radon domain. This space of functions
was recently proposed as the natural function space associated with shallow
ReLU neural networks. We derive a minimax lower bound for the estimation
problem for this function space and show that the neural network estimators are
minimax optimal up to logarithmic factors. We also show that this is a "mixed
variation" function space that contains classical multivariate function spaces
including certain Sobolev spaces and certain spectral Barron spaces. Finally,
we use these results to quantify a gap between neural networks and linear
methods (which include kernel methods). This paper sheds light on the
phenomenon that neural networks seem to break the curse of dimensionality.

    

### [[2109.08850] Coordinate Descent for MCP/SCAD Penalized Least Squares Converges Linearly](http://arxiv.org/abs/2109.08850)


  Recovering sparse signals from observed data is an important topic in
signal/imaging processing, statistics and machine learning. Nonconvex penalized
least squares have been attracted a lot of attentions since they enjoy nice
statistical properties. Computationally, coordinate descent (CD) is a workhorse
for minimizing the nonconvex penalized least squares criterion due to its
simplicity and scalability. In this work, we prove the linear convergence rate
to CD for solving MCP/SCAD penalized least squares problems.

    

### [[2109.08853] A survey on deep learning approaches for breast cancer diagnosis](http://arxiv.org/abs/2109.08853)


  Deep learning has introduced several learning-based methods to recognize
breast tumours and presents high applicability in breast cancer diagnostics. It
has presented itself as a practical installment in Computer-Aided Diagnostic
(CAD) systems to further assist radiologists in diagnostics for different
modalities. A deep learning network trained on images provided by hospitals or
public databases can perform classification, detection, and segmentation of
lesion types. Significant progress has been made in recognizing tumours on 2D
images but recognizing 3D images remains a frontier so far. The interconnection
of deep learning networks between different fields of study help propels
discoveries for more efficient, accurate, and robust networks. In this review
paper, the following topics will be explored: (i) theory and application of
deep learning, (ii) progress of 2D, 2.5D, and 3D CNN approaches in breast
tumour recognition from a performance metric perspective, and (iii) challenges
faced in CNN approaches.

    

### [[2109.08857] Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts](http://arxiv.org/abs/2109.08857)


  Evolutionary algorithms have been used in the digital art scene since the
1970s. A popular application of genetic algorithms is to optimize the
procedural placement of vector graphic primitives to resemble a given painting.
In recent years, deep learning-based approaches have also been proposed to
generate procedural drawings, which can be optimized using gradient descent. In
this work, we revisit the use of evolutionary algorithms for computational
creativity. We find that modern evolution strategies (ES) algorithms, when
tasked with the placement of shapes, offer large improvements in both quality
and efficiency compared to traditional genetic algorithms, and even comparable
to gradient-based methods. We demonstrate that ES is also well suited at
optimizing the placement of shapes to fit the CLIP model, and can produce
diverse, distinct geometric abstractions that are aligned with human
interpretation of language. Videos and demo: this https URL


### [[2109.08858] An Accelerated Variance-Reduced Conditional Gradient Sliding Algorithm for First-order and Zeroth-order Optimization](http://arxiv.org/abs/2109.08858)


  The conditional gradient algorithm (also known as the Frank-Wolfe algorithm)
has recently regained popularity in the machine learning community due to its
projection-free property to solve constrained problems. Although many variants
of the conditional gradient algorithm have been proposed to improve
performance, they depend on first-order information (gradient) to optimize.
Naturally, these algorithms are unable to function properly in the field of
increasingly popular zeroth-order optimization, where only zeroth-order
information (function value) is available. To fill in this gap, we propose a
novel Accelerated variance-Reduced Conditional gradient Sliding (ARCS)
algorithm for finite-sum problems, which can use either first-order or
zeroth-order information to optimize. To the best of our knowledge, ARCS is the
first zeroth-order conditional gradient sliding type algorithms solving convex
problems in zeroth-order optimization. In first-order optimization, the
convergence results of ARCS substantially outperform previous algorithms in
terms of the number of gradient query oracle. Finally we validated the
superiority of ARCS by experiments on real-world datasets.

    

### [[2109.08865] Interest-oriented Universal User Representation via Contrastive Learning](http://arxiv.org/abs/2109.08865)


  User representation is essential for providing high-quality commercial
services in industry. Universal user representation has received many interests
recently, with which we can be free from the cumbersome work of training a
specific model for each downstream application. In this paper, we attempt to
improve universal user representation from two points of views. First, a
contrastive self-supervised learning paradigm is presented to guide the
representation model training. It provides a unified framework that allows for
long-term or short-term interest representation learning in a data-driven
manner. Moreover, a novel multi-interest extraction module is presented. The
module introduces an interest dictionary to capture principal interests of the
given user, and then generate his/her interest-oriented representations via
behavior aggregation. Experimental results demonstrate the effectiveness and
applicability of the learned user representations.

    

### [[2109.08901] S$^3$VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation](http://arxiv.org/abs/2109.08901)


  Unsupervised domain adaptation (DA) methods have focused on achieving maximal
performance through aligning features from source and target domains without
using labeled data in the target domain. Whereas, in the real-world scenario's
it might be feasible to get labels for a small proportion of target data. In
these scenarios, it is important to select maximally-informative samples to
label and find an effective way to combine them with the existing knowledge
from source data. Towards achieving this, we propose S$^3$VAADA which i)
introduces a novel submodular criterion to select a maximally informative
subset to label and ii) enhances a cluster-based DA procedure through novel
improvements to effectively utilize all the available data for improving
generalization on target. Our approach consistently outperforms the competing
state-of-the-art approaches on datasets with varying degrees of domain shifts.

    

### [[2109.08904] Towards Resilient Artificial Intelligence: Survey and Research Issues](http://arxiv.org/abs/2109.08904)


  Artificial intelligence (AI) systems are becoming critical components of
today's IT landscapes. Their resilience against attacks and other environmental
influences needs to be ensured just like for other IT assets. Considering the
particular nature of AI, and machine learning (ML) in particular, this paper
provides an overview of the emerging field of resilient AI and presents
research issues the authors identify as potential future work.

    

### [[2109.08907] Releasing Graph Neural Networks with Differential Privacy Guarantees](http://arxiv.org/abs/2109.08907)


  With the increasing popularity of Graph Neural Networks (GNNs) in several
sensitive applications like healthcare and medicine, concerns have been raised
over the privacy aspects of trained GNNs. More notably, GNNs are vulnerable to
privacy attacks, such as membership inference attacks, even if only blackbox
access to the trained model is granted. To build defenses, differential privacy
has emerged as a mechanism to disguise the sensitive data in training datasets.
Following the strategy of Private Aggregation of Teacher Ensembles (PATE),
recent methods leverage a large ensemble of teacher models. These teachers are
trained on disjoint subsets of private data and are employed to transfer
knowledge to a student model, which is then released with privacy guarantees.
However, splitting graph data into many disjoint training sets may destroy the
structural information and adversely affect accuracy. We propose a new
graph-specific scheme of releasing a student GNN, which avoids splitting
private training data altogether. The student GNN is trained using public data,
partly labeled privately using the teacher GNN models trained exclusively for
each query node. We theoretically analyze our approach in the Rènyi
differential privacy framework and provide privacy guarantees. Besides, we show
the solid experimental performance of our method compared to several baselines,
including the PATE baseline adapted for graph-structured data. Our anonymized
code is available.

    

### [[2109.08908] Intra-Inter Subject Self-supervised Learning for Multivariate Cardiac Signals](http://arxiv.org/abs/2109.08908)


  Learning information-rich and generalizable representations effectively from
unlabeled multivariate cardiac signals to identify abnormal heart rhythms
(cardiac arrhythmias) is valuable in real-world clinical settings but often
challenging due to its complex temporal dynamics. Cardiac arrhythmias can vary
significantly in temporal patterns even for the same patient ($i.e.$, intra
subject difference). Meanwhile, the same type of cardiac arrhythmia can show
different temporal patterns among different patients due to different cardiac
structures ($i.e.$, inter subject difference). In this paper, we address the
challenges by proposing an Intra-inter Subject self-supervised Learning (ISL)
model that is customized for multivariate cardiac signals. Our proposed ISL
model integrates medical knowledge into self-supervision to effectively learn
from intra-inter subject differences. In intra subject self-supervision, ISL
model first extracts heartbeat-level features from each subject using a
channel-wise attentional CNN-RNN encoder. Then a stationarity test module is
employed to capture the temporal dependencies between heartbeats. In inter
subject self-supervision, we design a set of data augmentations according to
the clinical characteristics of cardiac signals and perform contrastive
learning among subjects to learn distinctive representations for various types
of patients. Extensive experiments on three real-world datasets were conducted.
In a semi-supervised transfer learning scenario, our pre-trained ISL model
leads about 10% improvement over supervised training when only 1% labeled data
is available, suggesting strong generalizability and robustness of the model.

    

### [[2109.08914] Text Detoxification using Large Pre-trained Neural Models](http://arxiv.org/abs/2109.08914)


  We present two novel unsupervised methods for eliminating toxicity in text.
Our first method combines two recent ideas: (1) guidance of the generation
process with small style-conditional language models and (2) use of
paraphrasing models to perform style transfer. We use a well-performing
paraphraser guided by style-trained language models to keep the text content
and remove toxicity. Our second method uses BERT to replace toxic words with
their non-offensive synonyms. We make the method more flexible by enabling BERT
to replace mask tokens with a variable number of words. Finally, we present the
first large-scale comparative study of style transfer models on the task of
toxicity removal. We compare our models with a number of methods for style
transfer. The models are evaluated in a reference-free way using a combination
of unsupervised style transfer metrics. Both methods we suggest yield new SOTA
results.

    

### [[2109.08916] Underwater Image Enhancement Using Convolutional Neural Network](http://arxiv.org/abs/2109.08916)


  This work proposes a method for underwater image enhancement using the
principle of histogram equalization. Since underwater images have a global
strong dominant colour, their colourfulness and contrast are often degraded.
Before applying the histogram equalisation technique on the image, the image is
converted from coloured image to a gray scale image for further operations.
Histogram equalization is a technique for adjusting image intensities to
enhance contrast. The colours of the image are retained using a convolutional
neural network model which is trained by the datasets of underwater images to
give better results.

    

### [[2109.08917] KNN Learning Techniques for Proportional Myocontrol in Prosthetics](http://arxiv.org/abs/2109.08917)


  This work has been conducted in the context of pattern-recognition-based
control for electromyographic prostheses. It presents a k-nearest neighbour
(kNN) classification technique for gesture recognition, extended by a
proportionality scheme. The methods proposed are practically implemented and
validated. Datasets are captured by means of a state-of-the-art 8-channel
electromyography (EMG) armband positioned on the forearm. Based on this data,
the influence of kNN's parameters is analyzed in pilot experiments. Moreover,
the effect of proportionality scaling and rest thresholding schemes is
investigated. A randomized, double-blind user study is conducted to compare the
implemented method with the state-of-research algorithm Ridge Regression with
Random Fourier Features (RR-RFF) for different levels of gesture exertion. The
results from these experiments show a statistically significant improvement in
favour of the kNN-based algorithm.

    

### [[2109.08945] Removing Noise from Extracellular Neural Recordings Using Fully Convolutional Denoising Autoencoders](http://arxiv.org/abs/2109.08945)


  Extracellular recordings are severely contaminated by a considerable amount
of noise sources, rendering the denoising process an extremely challenging task
that should be tackled for efficient spike sorting. To this end, we propose an
end-to-end deep learning approach to the problem, utilizing a Fully
Convolutional Denoising Autoencoder, which learns to produce a clean neuronal
activity signal from a noisy multichannel input. The experimental results on
simulated data show that our proposed method can improve significantly the
quality of noise-corrupted neural signals, outperforming widely-used wavelet
denoising techniques.

    

### [[2109.08949] Inductive Conformal Recommender System](http://arxiv.org/abs/2109.08949)


  Traditional recommendation algorithms develop techniques that can help people
to choose desirable items. However, in many real-world applications, along with
a set of recommendations, it is also essential to quantify each
recommendation's (un)certainty. The conformal recommender system uses the
experience of a user to output a set of recommendations, each associated with a
precise confidence value. Given a significance level $\varepsilon$, it provides
a bound $\varepsilon$ on the probability of making a wrong recommendation. The
conformal framework uses a key concept called nonconformity measure that
measure the strangeness of an item concerning other items. One of the
significant design challenges of any conformal recommendation framework is
integrating nonconformity measure with the recommendation algorithm. In this
paper, we introduce an inductive variant of a conformal recommender system. We
propose and analyze different nonconformity measures in the inductive setting.
We also provide theoretical proofs on the error-bound and the time complexity.
Extensive empirical analysis on ten benchmark datasets demonstrates that the
inductive variant substantially improves the performance in computation time
while preserving the accuracy.

    

### [[2109.08955] Manifold-preserved GANs](http://arxiv.org/abs/2109.08955)


  Generative Adversarial Networks (GANs) have been widely adopted in various
fields. However, existing GANs generally are not able to preserve the manifold
of data space, mainly due to the simple representation of discriminator for the
real/generated data. To address such open challenges, this paper proposes
Manifold-preserved GANs (MaF-GANs), which generalize Wasserstein GANs into
high-dimensional form. Specifically, to improve the representation of data, the
discriminator in MaF-GANs is designed to map data into a high-dimensional
manifold. Furthermore, to stabilize the training of MaF-GANs, an operation with
precise and universal solution for any K-Lipschitz continuity, called
Topological Consistency is proposed. The effectiveness of the proposed method
is justified by both theoretical analysis and empirical results. When adopting
DCGAN as the backbone on CelebA (256*256), the proposed method achieved 12.43
FID, which outperforms the state-of-the-art model like Realness GAN (23.51 FID)
by a large margin. Code will be made publicly available.

    

### [[2109.08956] Scenario adaptive disruption prediction study for next generation burning-plasma tokamaks](http://arxiv.org/abs/2109.08956)


  Next generation high performance (HP) tokamaks risk damage from unmitigated
disruptions at high current and power. Achieving reliable disruption prediction
for a device's HP operation based on its low performance (LP) data is key to
success. In this letter, through explorative data analysis and dedicated
numerical experiments on multiple existing tokamaks, we demonstrate how the
operational regimes of tokamaks can affect the power of a trained disruption
predictor. First, our results suggest data-driven disruption predictors trained
on abundant LP discharges work poorly on the HP regime of the same tokamak,
which is a consequence of the distinct distributions of the tightly correlated
signals related to disruptions in these two regimes. Second, we find that
matching operational parameters among tokamaks strongly improves cross-machine
accuracy which implies our model learns from the underlying scalings of
dimensionless physics parameters like q_{95}, \beta_{p} and confirms the
importance of these parameters in disruption physics and cross machine domain
matching from the data-driven perspective. Finally, our results show how in the
absence of HP data from the target devices, the best predictivity of the HP
regime for the target machine can be achieved by combining LP data from the
target with HP data from other machines. These results provide a possible
disruption predictor development strategy for next generation tokamaks, such as
ITER and SPARC, and highlight the importance of developing on existing machines
baseline scenario discharges of future tokamaks to collect more relevant
disruptive data.

    

### [[2109.08957] AI Accelerator Survey and Trends](http://arxiv.org/abs/2109.08957)


  Over the past several years, new machine learning accelerators were being
announced and released every month for a variety of applications from speech
recognition, video object detection, assisted driving, and many data center
applications. This paper updates the survey of AI accelerators and processors
from past two years. This paper collects and summarizes the current commercial
accelerators that have been publicly announced with peak performance and power
consumption numbers. The performance and power values are plotted on a scatter
graph, and a number of dimensions and observations from the trends on this plot
are again discussed and analyzed. This year, we also compile a list of
benchmarking performance results and compute the computational efficiency with
respect to peak performance.

    

### [[2109.08958] AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks](http://arxiv.org/abs/2109.08958)


  Neural networks require careful weight initialization to prevent signals from
exploding or vanishing. Existing initialization schemes solve this problem in
specific cases by assuming that the network has a certain activation function
or topology. It is difficult to derive such weight initialization strategies,
and modern architectures therefore often use these same initialization schemes
even though their assumptions do not hold. This paper introduces AutoInit, a
weight initialization algorithm that automatically adapts to different neural
network architectures. By analytically tracking the mean and variance of
signals as they propagate through the network, AutoInit is able to
appropriately scale the weights at each layer to avoid exploding or vanishing
signals. Experiments demonstrate that AutoInit improves performance of various
convolutional and residual networks across a range of activation function,
dropout, weight decay, learning rate, and normalizer settings. Further, in
neural architecture search and activation function meta-learning, AutoInit
automatically calculates specialized weight initialization strategies for
thousands of unique architectures and hundreds of unique activation functions,
and improves performance in vision, language, tabular, multi-task, and transfer
learning scenarios. AutoInit thus serves as an automatic configuration tool
that makes design of new neural network architectures more robust. The AutoInit
package provides a wrapper around existing TensorFlow models and is available
at this https URL.

    

### [[2109.08967] Ensemble Learning using Error Correcting Output Codes: New Classification Error Bounds](http://arxiv.org/abs/2109.08967)


  New bounds on classification error rates for the error-correcting output code
(ECOC) approach in machine learning are presented. These bounds have
exponential decay complexity with respect to codeword length and theoretically
validate the effectiveness of the ECOC approach. Bounds are derived for two
different models: the first under the assumption that all base classifiers are
independent and the second under the assumption that all base classifiers are
mutually correlated up to first-order. Moreover, we perform ECOC classification
on six datasets and compare their error rates with our bounds to experimentally
validate our work and show the effect of correlation on classification
accuracy.

    

### [[2109.08968] Visual Representation Learning for Preference-Aware Path Planning](http://arxiv.org/abs/2109.08968)


  Autonomous mobile robots deployed in outdoor environments must reason about
different types of terrain for both safety (e.g., prefer dirt over mud) and
deployer preferences (e.g., prefer dirt path over flower beds). Most existing
solutions to this preference-aware path planning problem use semantic
segmentation to classify terrain types from camera images, and then ascribe
costs to each type. Unfortunately, there are three key limitations of such
approaches -- they 1) require pre-enumeration of the discrete terrain types, 2)
are unable to handle hybrid terrain types (e.g., grassy dirt), and 3) require
expensive labelled data to train visual semantic segmentation. We introduce
Visual Representation Learning for Preference-Aware Path Planning (VRL-PAP), an
alternative approach that overcomes all three limitations: VRL-PAP leverages
unlabeled human demonstrations of navigation to autonomously generate triplets
for learning visual representations of terrain that are viewpoint invariant and
encode terrain types in a continuous representation space. The learned
representations are then used along with the same unlabeled human navigation
demonstrations to learn a mapping from the representation space to terrain
costs. At run time, VRL-PAP maps from images to representations and then
representations to costs to perform preference-aware path planning. We present
empirical results from challenging outdoor settings that demonstrate VRL-PAP 1)
is successfully able to pick paths that reflect demonstrated preferences, 2) is
comparable in execution to geometric navigation with a highly detailed manually
annotated map (without requiring such annotations), 3) is able to generalize to
novel terrain types with minimal additional unlabeled demonstrations.

    

### [[2109.08970] Temporal Knowledge Graph Completion using Box Embeddings](http://arxiv.org/abs/2109.08970)


  Knowledge graph completion is the task of inferring missing facts based on
existing data in a knowledge graph. Temporal knowledge graph completion (TKGC)
is an extension of this task to temporal knowledge graphs, where each fact is
additionally associated with a time stamp. Current approaches for TKGC
primarily build on existing embedding models which are developed for (static)
knowledge graph completion, and extend these models to incorporate time, where
the idea is to learn latent representations for entities, relations, and
timestamps and then use the learned representations to predict missing facts at
various time steps. In this paper, we propose BoxTE, a box embedding model for
TKGC, building on the static knowledge graph embedding model BoxE. We show that
BoxTE is fully expressive, and possesses strong inductive capacity in the
temporal setting. We then empirically evaluate our model and show that it
achieves state-of-the-art results on several TKGC benchmarks.

    

### [[2109.08974] Atrial Fibrillation: A Medical and Technological Review](http://arxiv.org/abs/2109.08974)


  Atrial Fibrillation (AF) is the most common type of arrhythmia (Greek a-,
loss + rhythmos, rhythm = loss of rhythm) leading to hospitalization in the
United States. Though sometimes AF is asymptomatic, it increases the risk of
stroke and heart failure in patients, in addition to lowering the
health-related quality of life (HRQOL). AF-related care costs the healthcare
system between $6.0 to $26 billion each year. Early detection of AF and
clinical attention can help improve symptoms and HRQOL of the patient, as well
as bring down the cost of care. However, the prevalent paradigm of AF detection
depends on electrocardiogram (ECG) recorded at a single point in time and does
not shed light on the relation of the symptoms with heart rhythm or AF. In the
recent decade, due to the democratization of health monitors and the advent of
high-performing computers, Machine Learning algorithms have been proven
effective in identifying AF, from the ECG of patients. This paper provides an
overview of the symptoms of AF, its diagnosis, and future prospects for
research in the field.

    

### [[2109.08983] G-CoS: GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency](http://arxiv.org/abs/2109.08983)


  Graph Neural Networks (GNNs) have emerged as the state-of-the-art (SOTA)
method for graph-based learning tasks. However, it still remains prohibitively
challenging to inference GNNs over large graph datasets, limiting their
application to large-scale real-world tasks. While end-to-end jointly
optimizing GNNs and their accelerators is promising in boosting GNNs' inference
efficiency and expediting the design process, it is still underexplored due to
the vast and distinct design spaces of GNNs and their accelerators. In this
work, we propose G-CoS, a GNN and accelerator co-search framework that can
automatically search for matched GNN structures and accelerators to maximize
both task accuracy and acceleration efficiency. Specifically, GCoS integrates
two major enabling components: (1) a generic GNN accelerator search space which
is applicable to various GNN structures and (2) a one-shot GNN and accelerator
co-search algorithm that enables simultaneous and efficient search for optimal
GNN structures and their matched accelerators. To the best of our knowledge,
G-CoS is the first co-search framework for GNNs and their accelerators.
Extensive experiments and ablation studies show that the GNNs and accelerators
generated by G-CoS consistently outperform SOTA GNNs and GNN accelerators in
terms of both task accuracy and hardware efficiency, while only requiring a few
hours for the end-to-end generation of the best matched GNNs and their
accelerators.

    

### [[2109.08996] Dynamic and Systematic Survey of Deep Learning Approaches for Driving Behavior Analysis](http://arxiv.org/abs/2109.08996)


  Improper driving results in fatalities, damages, increased energy
consumptions, and depreciation of the vehicles. Analyzing driving behaviour
could lead to optimize and avoid mentioned issues. By identifying the type of
driving and mapping them to the consequences of that type of driving, we can
get a model to prevent them. In this regard, we try to create a dynamic survey
paper to review and present driving behaviour survey data for future
researchers in our research. By analyzing 58 articles, we attempt to classify
standard methods and provide a framework for future articles to be examined and
studied in different dashboards and updated about trends.

    

### [[2109.09001] Development of patients triage algorithm from nationwide COVID-19 registry data based on machine learning](http://arxiv.org/abs/2109.09001)


  Prompt severity assessment model of confirmed patients who were infected with
infectious diseases could enable efficient diagnosis and alleviate the burden
on the medical system. This paper provides the development processes of the
severity assessment model using machine learning techniques and its application
on SARS-CoV-2 patients. Here, we highlight that our model only requires basic
patients' basic personal data, allowing for them to judge their own severity.
We selected the boosting-based decision tree model as a classifier and
interpreted mortality as a probability score after modeling. Specifically,
hyperparameters that determine the structure of the tree model were tuned using
the Bayesian optimization technique without any knowledge of medical
information. As a result, we measured model performance and identified the
variables affecting the severity through the model. Finally, we aim to
establish a medical system that allows patients to check their own severity and
informs them to visit the appropriate clinic center based on the past treatment
details of other patients with similar severity.

    

### [[2109.09010] Augmenting semantic lexicons using word embeddings and transfer learning](http://arxiv.org/abs/2109.09010)


  Sentiment-aware intelligent systems are essential to a wide array of
applications including marketing, political campaigns, recommender systems,
behavioral economics, social psychology, and national security. These
sentiment-aware intelligent systems are driven by language models which broadly
fall into two paradigms: 1. Lexicon-based and 2. Contextual. Although recent
contextual models are increasingly dominant, we still see demand for
lexicon-based models because of their interpretability and ease of use. For
example, lexicon-based models allow researchers to readily determine which
words and phrases contribute most to a change in measured sentiment. A
challenge for any lexicon-based approach is that the lexicon needs to be
routinely expanded with new words and expressions. Crowdsourcing annotations
for semantic dictionaries may be an expensive and time-consuming task. Here, we
propose two models for predicting sentiment scores to augment semantic lexicons
at a relatively low cost using word embeddings and transfer learning. Our first
model establishes a baseline employing a simple and shallow neural network
initialized with pre-trained word embeddings using a non-contextual approach.
Our second model improves upon our baseline, featuring a deep Transformer-based
network that brings to bear word definitions to estimate their lexical
polarity. Our evaluation shows that both models are able to score new words
with a similar accuracy to reviewers from Amazon Mechanical Turk, but at a
fraction of the cost.

    

### [[2109.09011] PluGeN: Multi-Label Conditional Generation From Pre-Trained Models](http://arxiv.org/abs/2109.09011)


  Modern generative models achieve excellent quality in a variety of tasks
including image or text generation and chemical molecule modeling. However,
existing methods often lack the essential ability to generate examples with
requested properties, such as the age of the person in the photo or the weight
of the generated molecule. Incorporating such additional conditioning factors
would require rebuilding the entire architecture and optimizing the parameters
from scratch. Moreover, it is difficult to disentangle selected attributes so
that to perform edits of only one attribute while leaving the others unchanged.
To overcome these limitations we propose PluGeN (Plugin Generative Network), a
simple yet effective generative technique that can be used as a plugin to
pre-trained generative models. The idea behind our approach is to transform the
entangled latent representation using a flow-based module into a
multi-dimensional space where the values of each attribute are modeled as an
independent one-dimensional distribution. In consequence, PluGeN can generate
new samples with desired attributes as well as manipulate labeled attributes of
existing examples. Due to the disentangling of the latent representation, we
are even able to generate samples with rare or unseen combinations of
attributes in the dataset, such as a young person with gray hair, men with
make-up, or women with beards. We combined PluGeN with GAN and VAE models and
applied it to conditional generation and manipulation of images and chemical
molecule modeling. Experiments demonstrate that PluGeN preserves the quality of
backbone models while adding the ability to control the values of labeled
attributes.

    

### [[2109.09013] Hydroelectric Generation Forecasting with Long Short Term Memory (LSTM) Based Deep Learning Model for Turkey](http://arxiv.org/abs/2109.09013)


  Hydroelectricity is one of the renewable energy source, has been used for
many years in Turkey. The production of hydraulic power plants based on water
reservoirs varies based on different parameters. For this reason, the
estimation of hydraulic production gains importance in terms of the planning of
electricity generation. In this article, the estimation of Turkey's monthly
hydroelectricity production has been made with the long-short-term memory
(LSTM) network-based deep learning model. The designed deep learning model is
based on hydraulic production time series and future production planning for
many years. By using real production data and different LSTM deep learning
models, their performance on the monthly forecast of hydraulic electricity
generation of the next year has been examined. The obtained results showed that
the use of time series based on real production data for many years and deep
learning model together is successful in long-term prediction. In the study, it
is seen that the 100-layer LSTM model, in which 120 months (10 years)
hydroelectric generation time data are used according to the RMSE and MAPE
values, are the highest model in terms of estimation accuracy, with a MAPE
value of 0.1311 (13.1%) in the annual total and 1.09% as the monthly average
distribution. In this model, the best results were obtained for the 100-layer
LSTM model, in which the time data of 144 months (12 years) hydroelectric
generation data are used, with a RMSE value of 29,689 annually and 2474.08 in
monthly distribution. According to the results of the study, time data covering
at least 120 months of production is recommended to create an acceptable
hydropower forecasting model with LSTM.

    

### [[2109.09014] A Machine Learning Pipeline to Examine Political Bias with Congressional Speeches](http://arxiv.org/abs/2109.09014)


  Computational methods to model political bias in social media involve several
challenges due to heterogeneity, high-dimensional, multiple modalities, and the
scale of the data. Political bias in social media has been studied in multiple
viewpoints like media bias, political ideology, echo chambers, and
controversies using machine learning pipelines. Most of the current methods
rely heavily on the manually-labeled ground-truth data for the underlying
political bias prediction tasks. Limitations of such methods include
human-intensive labeling, labels related to only a specific problem, and the
inability to determine the near future bias state of a social media
conversation. In this work, we address such problems and give machine learning
approaches to study political bias in two ideologically diverse social media
forums: Gab and Twitter without the availability of human-annotated data. Our
proposed methods exploit the use of transcripts collected from political
speeches in US congress to label the data and achieve the highest accuracy of
70.5% and 65.1% in Twitter and Gab data respectively to predict political bias.
We also present a machine learning approach that combines features from
cascades and text to forecast cascade's political bias with an accuracy of
about 85%.

    

### [[2109.09020] Multimodal Classification: Current Landscape, Taxonomy and Future Directions](http://arxiv.org/abs/2109.09020)


  Multimodal classification research has been gaining popularity in many
domains that collect more data from multiple sources including satellite
imagery, biometrics, and medicine. However, the lack of consistent terminology
and architectural descriptions makes it difficult to compare different existing
solutions. We address these challenges by proposing a new taxonomy for
describing such systems based on trends found in recent publications on
multimodal classification. Many of the most difficult aspects of unimodal
classification have not yet been fully addressed for multimodal datasets
including big data, class imbalance, and instance level difficulty. We also
provide a discussion of these challenges and future directions.

    

### [[2109.09022] Change of human mobility during COVID-19: A United States case study](http://arxiv.org/abs/2109.09022)


  With the onset of COVID-19 and the resulting shelter in place guidelines
combined with remote working practices, human mobility in 2020 has been
dramatically impacted. Existing studies typically examine whether mobility in
specific localities increases or decreases at specific points in time and
relate these changes to certain pandemic and policy events. In this paper, we
study mobility change in the US through a five-step process using mobility
footprint data. (Step 1) Propose the delta Time Spent in Public Places
(Delta-TSPP) as a measure to quantify daily changes in mobility for each US
county from 2019-2020. (Step 2) Conduct Principal Component Analysis (PCA) to
reduce the Delta-TSPP time series of each county to lower-dimensional latent
components of change in mobility. (Step 3) Conduct clustering analysis to find
counties that exhibit similar latent components. (Step 4) Investigate local and
global spatial autocorrelation for each component. (Step 5) Conduct correlation
analysis to investigate how various population characteristics and behavior
correlate with mobility patterns. Results show that by describing each county
as a linear combination of the three latent components, we can explain 59% of
the variation in mobility trends across all US counties. Specifically, change
in mobility in 2020 for US counties can be explained as a combination of three
latent components: 1) long-term reduction in mobility, 2) no change in
mobility, and 3) short-term reduction in mobility. We observe significant
correlations between the three latent components of mobility change and various
population characteristics, including political leaning, population, COVID-19
cases and deaths, and unemployment. We find that our analysis provides a
comprehensive understanding of mobility change in response to the COVID-19
pandemic.

    

### [[2109.09023] Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Model Training](http://arxiv.org/abs/2109.09023)


  In this paper, we raise up an emerging personal data protection problem where
user personal data (e.g. images) could be inappropriately exploited to train
deep neural network models without authorization. To solve this problem, we
revisit traditional watermarking in advanced machine learning settings. By
embedding a watermarking signature using specialized linear color
transformation to user images, neural models will be imprinted with such a
signature if training data include watermarked images. Then, a third-party
verifier can verify potential unauthorized usage by inferring the watermark
signature from neural models. We further explore the desired properties of
watermarking and signature space for convincing verification. Through extensive
experiments, we show empirically that linear color transformation is effective
in protecting user's personal images for various realistic settings. To the
best of our knowledge, this is the first work to protect users' personal data
from unauthorized usage in neural network training.

    

### [[2109.09026] Hybrid Data Augmentation and Deep Attention-based Dilated Convolutional-Recurrent Neural Networks for Speech Emotion Recognition](http://arxiv.org/abs/2109.09026)


  Speech emotion recognition (SER) has been one of the significant tasks in
Human-Computer Interaction (HCI) applications. However, it is hard to choose
the optimal features and deal with imbalance labeled data. In this article, we
investigate hybrid data augmentation (HDA) methods to generate and balance data
based on traditional and generative adversarial networks (GAN) methods. To
evaluate the effectiveness of HDA methods, a deep learning framework namely
(ADCRNN) is designed by integrating deep dilated convolutional-recurrent neural
networks with an attention mechanism. Besides, we choose 3D log Mel-spectrogram
(MelSpec) features as the inputs for the deep learning framework. Furthermore,
we reconfigure a loss function by combining a softmax loss and a center loss to
classify the emotions. For validating our proposed methods, we use the EmoDB
dataset that consists of several emotions with imbalanced samples. Experimental
results prove that the proposed methods achieve better accuracy than the
state-of-the-art methods on the EmoDB with 87.12% and 88.47% for the
traditional and GAN-based methods, respectively.

    

### [[2109.09031] Hindsight Foresight Relabeling for Meta-Reinforcement Learning](http://arxiv.org/abs/2109.09031)


  Meta-reinforcement learning (meta-RL) algorithms allow for agents to learn
new behaviors from small amounts of experience, mitigating the sample
inefficiency problem in RL. However, while meta-RL agents can adapt quickly to
new tasks at test time after experiencing only a few trajectories, the
meta-training process is still sample-inefficient. Prior works have found that
in the multi-task RL setting, relabeling past transitions and thus sharing
experience among tasks can improve sample efficiency and asymptotic
performance. We apply this idea to the meta-RL setting and devise a new
relabeling method called Hindsight Foresight Relabeling (HFR). We construct a
relabeling distribution using the combination of "hindsight", which is used to
relabel trajectories using reward functions from the training task
distribution, and "foresight", which takes the relabeled trajectories and
computes the utility of each trajectory for each task. HFR is easy to implement
and readily compatible with existing meta-RL algorithms. We find that HFR
improves performance when compared to other relabeling methods on a variety of
meta-RL tasks.

    

### [[2109.09032] JEM++: Improved Techniques for Training JEM](http://arxiv.org/abs/2109.09032)


  Joint Energy-based Model (JEM) is a recently proposed hybrid model that
retains strong discriminative power of modern CNN classifiers, while generating
samples rivaling the quality of GAN-based approaches. In this paper, we propose
a variety of new training procedures and architecture features to improve JEM's
accuracy, training stability, and speed altogether. 1) We propose a proximal
SGLD to generate samples in the proximity of samples from the previous step,
which improves the stability. 2) We further treat the approximate maximum
likelihood learning of EBM as a multi-step differential game, and extend the
YOPO framework to cut out redundant calculations during backpropagation, which
accelerates the training substantially. 3) Rather than initializing SGLD chain
from random noise, we introduce a new informative initialization that samples
from a distribution estimated from training data. 4) This informative
initialization allows us to enable batch normalization in JEM, which further
releases the power of modern CNN architectures for hybrid modeling. Code:
this https URL


### [[2109.09034] Greedy UnMixing for Q-Learning in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2109.09034)


  This paper introduces Greedy UnMix (GUM) for cooperative multi-agent
reinforcement learning (MARL). Greedy UnMix aims to avoid scenarios where MARL
methods fail due to overestimation of values as part of the large joint
state-action space. It aims to address this through a conservative Q-learning
approach through restricting the state-marginal in the dataset to avoid
unobserved joint state action spaces, whilst concurrently attempting to unmix
or simplify the problem space under the centralized training with decentralized
execution paradigm. We demonstrate the adherence to Q-function lower bounds in
the Q-learning for MARL scenarios, and demonstrate superior performance to
existing Q-learning MARL approaches as well as more general MARL algorithms
over a set of benchmark MARL tasks, despite its relative simplicity compared
with state-of-the-art approaches.

    

### [[2109.09037] Dual Behavior Regularized Reinforcement Learning](http://arxiv.org/abs/2109.09037)


  Reinforcement learning has been shown to perform a range of complex tasks
through interaction with an environment or collected leveraging experience.
However, many of these approaches presume optimal or near optimal experiences
or the presence of a consistent environment. In this work we propose dual,
advantage-based behavior policy based on counterfactual regret minimization. We
demonstrate the flexibility of this approach and how it can be adapted to
online contexts where the environment is available to collect experiences and a
variety of other contexts. We demonstrate this new algorithm can outperform
several strong baseline models in different contexts based on a range of
continuous environments. Additional ablations provide insights into how our
dual behavior regularized reinforcement learning approach is designed compared
with other plausible modifications and demonstrates its ability to generalize.

    

### [[2109.09038] Regularize! Don't Mix: Multi-Agent Reinforcement Learning without Explicit Centralized Structures](http://arxiv.org/abs/2109.09038)


  We propose using regularization for Multi-Agent Reinforcement Learning rather
than learning explicit cooperative structures called {\em Multi-Agent
Regularized Q-learning} (MARQ). Many MARL approaches leverage centralized
structures in order to exploit global state information or removing
communication constraints when the agents act in a decentralized manner.
Instead of learning redundant structures which is removed during agent
execution, we propose instead to leverage shared experiences of the agents to
regularize the individual policies in order to promote structured exploration.
We examine several different approaches to how MARQ can either explicitly or
implicitly regularize our policies in a multi-agent setting. MARQ aims to
address these limitations in the MARL context through applying regularization
constraints which can correct bias in off-policy out-of-distribution agent
experiences and promote diverse exploration. Our algorithm is evaluated on
several benchmark multi-agent environments and we show that MARQ consistently
outperforms several baselines and state-of-the-art algorithms; learning in
fewer steps and converging to higher returns.

    

### [[2109.09046] Improving Fairness for Data Valuation in Federated Learning](http://arxiv.org/abs/2109.09046)


  Federated learning is an emerging decentralized machine learning scheme that
allows multiple data owners to work collaboratively while ensuring data
privacy. The success of federated learning depends largely on the participation
of data owners. To sustain and encourage data owners' participation, it is
crucial to fairly evaluate the quality of the data provided by the data owners
and reward them correspondingly. Federated Shapley value, recently proposed by
Wang et al. [Federated Learning, 2020], is a measure for data value under the
framework of federated learning that satisfies many desired properties for data
valuation. However, there are still factors of potential unfairness in the
design of federated Shapley value because two data owners with the same local
data may not receive the same evaluation. We propose a new measure called
completed federated Shapley value to improve the fairness of federated Shapley
value. The design depends on completing a matrix consisting of all the possible
contributions by different subsets of the data owners. It is shown under mild
conditions that this matrix is approximately low-rank by leveraging concepts
and tools from optimization. Both theoretical analysis and empirical evaluation
verify that the proposed measure does improve fairness in many circumstances.

    

### [[2109.09048] A framework for benchmarking uncertainty in deep regression](http://arxiv.org/abs/2109.09048)


  We propose a framework for the assessment of uncertainty quantification in
deep regression. The framework is based on regression problems where the
regression function is a linear combination of nonlinear functions. Basically,
any level of complexity can be realized through the choice of the nonlinear
functions and the dimensionality of their domain. Results of an uncertainty
quantification for deep regression are compared against those obtained by a
statistical reference method. The reference method utilizes knowledge of the
underlying nonlinear functions and is based on a Bayesian linear regression
using a reference prior. Reliability of uncertainty quantification is assessed
in terms of coverage probabilities, and accuracy through the size of calculated
uncertainties. We illustrate the proposed framework by applying it to current
approaches for uncertainty quantification in deep regression. The flexibility,
together with the availability of a reference solution, makes the framework
suitable for defining benchmark sets for uncertainty quantification.

    

### [[2109.09060] On the Noise Stability and Robustness of Adversarially Trained Networks on NVM Crossbars](http://arxiv.org/abs/2109.09060)


  Applications based on Deep Neural Networks (DNNs) have grown exponentially in
the past decade. To match their increasing computational needs, several
Non-Volatile Memory (NVM) crossbar-based accelerators have been proposed. Apart
from improved energy efficiency and performance, these approximate hardware
also possess intrinsic robustness for defense against Adversarial Attacks,
which is an important security concern for DNNs. Prior works have focused on
quantifying this intrinsic robustness for vanilla networks, that is DNNs
trained on unperturbed inputs. However, adversarial training of DNNs is the
benchmark technique for robustness, and sole reliance on intrinsic robustness
of the hardware may not be sufficient. In this work, we explore the design of
robust DNNs through the amalgamation of adversarial training and the intrinsic
robustness offered by NVM crossbar-based analog hardware. First, we study the
noise stability of such networks on unperturbed inputs and observe that
internal activations of adversarially trained networks have lower
Signal-to-Noise Ratio (SNR), and are sensitive to noise than vanilla networks.
As a result, they suffer significantly higher performance degradation due to
the non-ideal computations; on an average 2x accuracy drop. On the other hand,
for adversarial images generated using Projected-Gradient-Descent (PGD)
White-Box attacks, ResNet-10/20 adversarially trained on CIFAR-10/100 display a
5-10% gain in robust accuracy due to the underlying NVM crossbar when the
attack epsilon ($\epsilon_{attack}$, the degree of input perturbations) is
greater than the epsilon of the adversarial training ($\epsilon_{train}$). Our
results indicate that implementing adversarially trained networks on analog
hardware requires careful calibration between hardware non-idealities and
$\epsilon_{train}$ to achieve optimum robustness and performance.

    

### [[2109.09061] Model-Based Approach for Measuring the Fairness in ASR](http://arxiv.org/abs/2109.09061)


  The issue of fairness arises when the automatic speech recognition (ASR)
systems do not perform equally well for all subgroups of the population. In any
fairness measurement studies for ASR, the open questions of how to control the
nuisance factors, how to handle unobserved heterogeneity across speakers, and
how to trace the source of any word error rate (WER) gap among different
subgroups are especially important - if not appropriately accounted for,
incorrect conclusions will be drawn. In this paper, we introduce mixed-effects
Poisson regression to better measure and interpret any WER difference among
subgroups of interest. Particularly, the presented method can effectively
address the three problems raised above and is very flexible to use in
practical disparity analyses. We demonstrate the validity of proposed
model-based approach on both synthetic and real-world speech data.

    

### [[2109.09063] Ontology-based n-ball Concept Embeddings Informing Few-shot Image Classification](http://arxiv.org/abs/2109.09063)


  We propose a novel framework named ViOCE that integrates ontology-based
background knowledge in the form of $n$-ball concept embeddings into a neural
network based vision architecture. The approach consists of two components -
converting symbolic knowledge of an ontology into continuous space by learning
n-ball embeddings that capture properties of subsumption and disjointness, and
guiding the training and inference of a vision model using the learnt
embeddings. We evaluate ViOCE using the task of few-shot image classification,
where it demonstrates superior performance on two standard benchmarks.

    

### [[2109.09076] Towards Representation Learning for Atmospheric Dynamics](http://arxiv.org/abs/2109.09076)


  The prediction of future climate scenarios under anthropogenic forcing is
critical to understand climate change and to assess the impact of potentially
counter-acting technologies. Machine learning and hybrid techniques for this
prediction rely on informative metrics that are sensitive to pertinent but
often subtle influences. For atmospheric dynamics, a critical part of the
climate system, the "eyeball metric", i.e. a visual inspection by an expert, is
currently still the gold standard. However, it cannot be used as metric in
machine learning systems where an algorithmic description is required.
Motivated by the success of intermediate neural network activations as basis
for learned metrics, e.g. in computer vision, we present a novel,
self-supervised representation learning approach specifically designed for
atmospheric dynamics. Our approach, called AtmoDist, trains a neural network on
a simple, auxiliary task: predicting the temporal distance between elements of
a shuffled sequence of atmospheric fields (e.g. the components of the wind
field from a reanalysis or simulation). The task forces the network to learn
important intrinsic aspects of the data as activations in its layers and from
these hence a discriminative metric can be obtained. We demonstrate this by
using AtmoDist to define a metric for GAN-based super resolution of vorticity
and divergence. Our upscaled data matches closely the true statistics of a high
resolution reference and it significantly outperform the state-of-the-art based
on mean squared error. Since AtmoDist is unsupervised, only requires a temporal
sequence of fields, and uses a simple auxiliary task, it can be used in a wide
range of applications that aim to understand and mitigate climate change.

    

### [[2109.09077] DECORAS: detection and characterization of radio-astronomical sources using deep learning](http://arxiv.org/abs/2109.09077)


  We present DECORAS, a deep learning based approach to detect both point and
extended sources from Very Long Baseline Interferometry (VLBI) observations.
Our approach is based on an encoder-decoder neural network architecture that
uses a low number of convolutional layers to provide a scalable solution for
source detection. In addition, DECORAS performs source characterization in
terms of the position, effective radius and peak brightness of the detected
sources. We have trained and tested the network with images that are based on
realistic Very Long Baseline Array (VLBA) observations at 20 cm. Also, these
images have not gone through any prior de-convolution step and are directly
related to the visibility data via a Fourier transform. We find that the source
catalog generated by DECORAS has a better overall completeness and purity, when
compared to a traditional source detection algorithm. DECORAS is complete at
the 7.5$\sigma$ level, and has an almost factor of two improvement in
reliability at 5.5$\sigma$. We find that DECORAS can recover the position of
the detected sources to within 0.61 $\pm$ 0.69 mas, and the effective radius
and peak surface brightness are recovered to within 20 per cent for 98 and 94
per cent of the sources, respectively. Overall, we find that DECORAS provides a
reliable source detection and characterization solution for future wide-field
VLBI surveys.

    

### [[2109.09105] What BERT Based Language Models Learn in Spoken Transcripts: An Empirical Study](http://arxiv.org/abs/2109.09105)


  Language Models (LMs) have been ubiquitously leveraged in various tasks
including spoken language understanding (SLU). Spoken language requires careful
understanding of speaker interactions, dialog states and speech induced
multimodal behaviors to generate a meaningful representation of the
this http URL this work, we propose to dissect SLU into three representative
properties:conversational(disfluency, pause, overtalk), channel(speaker-type,
turn-tasks) andASR(insertion, deletion,substitution). We probe BERT based
language models (BERT, RoBERTa) trained on spoken transcripts to investigate
its ability to understand multifarious properties in absence of any speech
cues. Empirical results indicate that LM is surprisingly good at capturing
conversational properties such as pause prediction and overtalk detection from
lexical tokens. On the downsides, the LM scores low on turn-tasks and ASR
errors predictions. Additionally, pre-training the LM on spoken transcripts
restrain its linguistic understanding. Finally,we establish the efficacy and
transferability of the mentioned properties on two benchmark datasets:
Switchboard Dialog Act and Disfluency datasets.

    

### [[2109.09120] ComicGAN: Text-to-Comic Generative Adversarial Network](http://arxiv.org/abs/2109.09120)


  Drawing and annotating comic illustrations is a complex and difficult
process. No existing machine learning algorithms have been developed to create
comic illustrations based on descriptions of illustrations, or the dialogue in
comics. Moreover, it is not known if a generative adversarial network (GAN) can
generate original comics that correspond to the dialogue and/or descriptions.
GANs are successful in producing photo-realistic images, but this technology
does not necessarily translate to generation of flawless comics. What is more,
comic evaluation is a prominent challenge as common metrics such as Inception
Score will not perform comparably, as they are designed to work on photos. In
this paper: 1. We implement ComicGAN, a novel text-to-comic pipeline based on a
text-to-image GAN that synthesizes comics according to text descriptions. 2. We
describe an in-depth empirical study of the technical difficulties of comic
generation using GAN's. ComicGAN has two novel features: (i) text description
creation from labels via permutation and augmentation, and (ii) custom image
encoding with Convolutional Neural Networks. We extensively evaluate the
proposed ComicGAN in two scenarios, namely image generation from descriptions,
and image generation from dialogue. Our results on 1000 Dilbert comic panels
and 6000 descriptions show synthetic comic panels from text inputs resemble
original Dilbert panels. Novel methods for text description creation and custom
image encoding brought improvements to Frechet Inception Distance, detail, and
overall image quality over baseline algorithms. Generating illustrations from
descriptions provided clear comics including characters and colours that were
specified in the descriptions.

    

### [[2109.09150] A Study of the Generalizability of Self-Supervised Representations](http://arxiv.org/abs/2109.09150)


  Recent advancements in self-supervised learning (SSL) made it possible to
learn generalizable visual representations from unlabeled data. The performance
of Deep Learning models fine-tuned on pretrained SSL representations is on par
with models fine-tuned on the state-of-the-art supervised learning (SL)
representations. Irrespective of the progress made in SSL, its generalizability
has not been studied extensively. In this article, we perform a deeper analysis
of the generalizability of pretrained SSL and SL representations by conducting
a domain-based study for transfer learning classification tasks. The
representations are learned from the ImageNet source data, which are then
fine-tuned using two types of target datasets: similar to the source dataset,
and significantly different from the source dataset. We study generalizability
of the SSL and SL-based models via their prediction accuracy as well as
prediction confidence. In addition to this, we analyze the attribution of the
final convolutional layer of these models to understand how they reason about
the semantic identity of the data. We show that the SSL representations are
more generalizable as compared to the SL representations. We explain the
generalizability of the SSL representations by investigating its invariance
property, which is shown to be better than that observed in the SL
representations.

    

### [[2109.09151] Locally-symplectic neural networks for learning volume-preserving dynamics](http://arxiv.org/abs/2109.09151)


  We propose locally-symplectic neural networks LocSympNets for learning
volume-preserving dynamics. The construction of LocSympNets stems from the
theorem of local Hamiltonian description of the vector field of a
volume-preserving dynamical system and the splitting methods based on
symplectic integrators. Modified gradient modules of recently proposed
symplecticity-preserving neural networks SympNets are used to construct
locally-symplectic modules, which composition results in volume-preserving
neural networks. LocSympNets are studied numerically considering linear and
nonlinear dynamics, i.e., semi-discretized advection equation and Euler
equations of the motion of a free rigid body, respectively. LocSympNets are
able to learn linear and nonlinear dynamics to high degree of accuracy. When
learning a single trajectory of the rigid body dynamics LocSympNets are able to
learn both invariants of the system with absolute relative errors below 1% in
long-time predictions and produce qualitatively good short-time predictions,
when the learning of the whole system from randomly sampled data is considered.

    

### [[2109.09164] Optimal Ensemble Construction for Multi-Study Prediction with Applications to COVID-19 Excess Mortality Estimation](http://arxiv.org/abs/2109.09164)


  It is increasingly common to encounter prediction tasks in the biomedical
sciences for which multiple datasets are available for model training. Common
approaches such as pooling datasets and applying standard statistical learning
methods can result in poor out-of-study prediction performance when datasets
are heterogeneous. Theoretical and applied work has shown $\textit{multi-study
ensembling}$ to be a viable alternative that leverages the variability across
datasets in a manner that promotes model generalizability. Multi-study
ensembling uses a two-stage $\textit{stacking}$ strategy which fits
study-specific models and estimates ensemble weights separately. This approach
ignores, however, the ensemble properties at the model-fitting stage,
potentially resulting in a loss of efficiency. We therefore propose
$\textit{optimal ensemble construction}$, an $\textit{all-in-one}$ approach to
multi-study stacking whereby we jointly estimate ensemble weights as well as
parameters associated with each study-specific model. We prove that limiting
cases of our approach yield existing methods such as multi-study stacking and
pooling datasets before model fitting. We propose an efficient block coordinate
descent algorithm to optimize the proposed loss function. We compare our
approach to standard methods by applying it to a multi-country COVID-19 dataset
for baseline mortality prediction. We show that when little data is available
for a country before the onset of the pandemic, leveraging data from other
countries can substantially improve prediction accuracy. Importantly, our
approach outperforms multi-study stacking and other standard methods in this
application. We further characterize the method's performance in data-driven
and other simulations. Our method remains competitive with or outperforms
multi-study stacking and other earlier methods across a range of between-study
heterogeneity levels.

    

### [[2109.09165] Traffic-Net: 3D Traffic Monitoring Using a Single Camera](http://arxiv.org/abs/2109.09165)


  Computer Vision has played a major role in Intelligent Transportation Systems
(ITS) and traffic surveillance. Along with the rapidly growing automated
vehicles and crowded cities, the automated and advanced traffic management
systems (ATMS) using video surveillance infrastructures have been evolved by
the implementation of Deep Neural Networks. In this research, we provide a
practical platform for real-time traffic monitoring, including 3D
vehicle/pedestrian detection, speed detection, trajectory estimation,
congestion detection, as well as monitoring the interaction of vehicles and
pedestrians, all using a single CCTV traffic camera. We adapt a custom YOLOv5
deep neural network model for vehicle/pedestrian detection and an enhanced SORT
tracking algorithm. For the first time, a hybrid satellite-ground based inverse
perspective mapping (SG-IPM) method for camera auto-calibration is also
developed which leads to an accurate 3D object detection and visualisation. We
also develop a hierarchical traffic modelling solution based on short- and
long-term temporal video data stream to understand the traffic flow,
bottlenecks, and risky spots for vulnerable road users. Several experiments on
real-world scenarios and comparisons with state-of-the-art are conducted using
various traffic monitoring datasets, including MIO-TCD, UA-DETRAC and GRAM-RTM
collected from highways, intersections, and urban areas under different
lighting and weather conditions.

    

### [[2109.09180] Lifelong Robotic Reinforcement Learning by Retaining Experiences](http://arxiv.org/abs/2109.09180)


  Multi-task learning ideally allows robots to acquire a diverse repertoire of
useful skills. However, many multi-task reinforcement learning efforts assume
the robot can collect data from all tasks at all times. In reality, the tasks
that the robot learns arrive sequentially, depending on the user and the
robot's current environment. In this work, we study a practical sequential
multi-task RL problem that is motivated by the practical constraints of
physical robotic systems, and derive an approach that effectively leverages the
data and policies learned for previous tasks to cumulatively grow the robot's
skill-set. In a series of simulated robotic manipulation experiments, our
approach requires less than half the samples than learning each task from
scratch, while avoiding impractical round-robin data collection. On a Franka
Emika Panda robot arm, our approach incrementally learns ten challenging tasks,
including bottle capping and block insertion.

    

### [[2109.09189] Probabilistic Bearing Fault Diagnosis Using Gaussian Process with Tailored Feature Extraction](http://arxiv.org/abs/2109.09189)


  Rolling bearings are subject to various faults due to its long-time operation
under harsh environment, which will lead to unexpected breakdown of machinery
system and cause severe accidents. Deep learning methods recently have gained
growing interests and extensively applied in the data-driven bearing fault
diagnosis. However, current deep learning methods perform the bearing fault
diagnosis in the form of deterministic classification, which overlook the
uncertainties that inevitably exist in actual practice. To tackle this issue,
in this research we develop a probabilistic fault diagnosis framework that can
account for the uncertainty effect in prediction, which bears practical
significance. This framework fully leverages the probabilistic feature of
Gaussian process classifier (GPC). To facilitate the establishment of
high-fidelity GPC, the tailored feature extraction with dimensionality
reduction method can be optimally determined through the cross validation-based
grid search upon a prespecified method pool consisting of various kernel
principal component analysis (KPCA) methods and stacked autoencoder. This
strategy can ensure the complex nonlinear relations between the features and
faults to be adequately characterized. Furthermore, the sensor fusion concept
is adopted to enhance the diagnosis performance. As compared with the
traditional deep learning methods, this proposed framework usually requires
less labeled data and less effort for parameter tuning. Systematic case studies
using the publicly accessible experimental rolling bearing dataset are carried
out to validate this new framework. Various influencing factors on fault
diagnosis performance also are thoroughly investigated.

    

### [[2109.09190] Harnessing the Power of Ego Network Layers for Link Prediction in Online Social Networks](http://arxiv.org/abs/2109.09190)


  Being able to recommend links between users in online social networks is
important for users to connect with like-minded individuals as well as for the
platforms themselves and third parties leveraging social media information to
grow their business. Predictions are typically based on unsupervised or
supervised learning, often leveraging simple yet effective graph topological
information, such as the number of common neighbors. However, we argue that
richer information about personal social structure of individuals might lead to
better predictions. In this paper, we propose to leverage well-established
social cognitive theories to improve link prediction performance. According to
these theories, individuals arrange their social relationships along, on
average, five concentric circles of decreasing intimacy. We postulate that
relationships in different circles have different importance in predicting new
links. In order to validate this claim, we focus on popular feature-extraction
prediction algorithms (both unsupervised and supervised) and we extend them to
include social-circles awareness. We validate the prediction performance of
these circle-aware algorithms against several benchmarks (including their
baseline versions as well as node-embedding- and GNN-based link prediction),
leveraging two Twitter datasets comprising a community of video gamers and
generic users. We show that social-awareness generally provides significant
improvements in the prediction performance, beating also state-of-the-art
solutions like node2vec and SEAL, and without increasing the computational
complexity. Finally, we show that social-awareness can be used in place of
using a classifier (which may be costly or impractical) for targeting a
specific category of users.

    

### [[2109.09193] Towards Zero-Label Language Learning](http://arxiv.org/abs/2109.09193)


  This paper explores zero-label learning in Natural Language Processing (NLP),
whereby no human-annotated data is used anywhere during training and models are
trained purely on synthetic data. At the core of our framework is a novel
approach for better leveraging the powerful pretrained language models.
Specifically, inspired by the recent success of few-shot inference on GPT-3, we
present a training data creation procedure named Unsupervised Data Generation
(UDG), which leverages few-shot prompts to synthesize high-quality training
data without real human annotations. Our method enables zero-label learning as
we train task-specific models solely on the synthetic data, yet we achieve
better or comparable results from strong baseline models trained on
human-labeled data. Furthermore, when mixed with labeled data, our approach
serves as a highly effective data augmentation procedure, achieving new
state-of-the-art results on the SuperGLUE benchmark.

    

### [[2109.09199] Co-occurrence of medical conditions: Exposing patterns through probabilistic topic modeling of SNOMED codes](http://arxiv.org/abs/2109.09199)


  Patients associated with multiple co-occurring health conditions often face
aggravated complications and less favorable outcomes. Co-occurring conditions
are especially prevalent among individuals suffering from kidney disease, an
increasingly widespread condition affecting 13% of the general population in
the US. This study aims to identify and characterize patterns of co-occurring
medical conditions in patients employing a probabilistic framework.
Specifically, we apply topic modeling in a non-traditional way to find
associations across SNOMEDCT codes assigned and recorded in the EHRs of>13,000
patients diagnosed with kidney disease. Unlike most prior work on topic
modeling, we apply the method to codes rather than to natural language.
Moreover, we quantitatively evaluate the topics, assessing their tightness and
distinctiveness, and also assess the medical validity of our results. Our
experiments show that each topic is succinctly characterized by a few highly
probable and unique disease codes, indicating that the topics are tight.
Furthermore, inter-topic distance between each pair of topics is typically
high, illustrating distinctiveness. Last, most coded conditions grouped
together within a topic, are indeed reported to co-occur in the medical
literature. Notably, our results uncover a few indirect associations among
conditions that have hitherto not been reported as correlated in the medical
literature.

    

### [[2109.09203] Topology, Convergence, and Reconstruction of Predictive States](http://arxiv.org/abs/2109.09203)


  Predictive equivalence in discrete stochastic processes have been applied
with great success to identify randomness and structure in statistical physics
and chaotic dynamical systems and to inferring hidden Markov models. We examine
the conditions under which they can be reliably reconstructed from time-series
data, showing that convergence of predictive states can be achieved from
empirical samples in the weak topology of measures. Moreover, predictive states
may be represented in Hilbert spaces that replicate the weak topology. We
mathematically explain how these representations are particularly beneficial
when reconstructing high-memory processes and connect them to reproducing
kernel Hilbert spaces.

    

### [[2109.09207] Machine Learning Methods for Identifying Atrial Fibrillation Cases and Their Predictors in Patients With Hypertrophic Cardiomyopathy: The HCM-AF-Risk Model](http://arxiv.org/abs/2109.09207)


  Hypertrophic cardiomyopathy (HCM) patients have a high incidence of atrial
fibrillation (AF) and increased stroke risk, even with low risk of congestive
heart failure, hypertension, age, diabetes, previous stroke/transient ischemic
attack scores. Hence, there is a need to understand the pathophysiology of AF
and stroke in HCM. In this retrospective study, we develop and apply a
data-driven, machine learning based method to identify AF cases, and clinical
and imaging features associated with AF, using electronic health record data.
HCM patients with documented paroxysmal/persistent/permanent AF (n = 191) were
considered AF cases, and the remaining patients in sinus rhythm (n = 640) were
tagged as No-AF. We evaluated 93 clinical variables and the most informative
variables useful for distinguishing AF from No-AF cases were selected based on
the 2-sample t test and the information gain criterion. We identified 18 highly
informative variables that are positively (n = 11) and negatively (n = 7)
correlated with AF in HCM. Next, patient records were represented via these 18
variables. Data imbalance resulting from the relatively low number of AF cases
was addressed via a combination of oversampling and under-sampling strategies.
We trained and tested multiple classifiers under this sampling approach,
showing effective classification. Specifically, an ensemble of logistic
regression and naive Bayes classifiers, trained based on the 18 variables and
corrected for data imbalance, proved most effective for separating AF from
No-AF cases (sensitivity = 0.74, specificity = 0.70, C-index = 0.80). Our model
is the first machine learning based method for identification of AF cases in
HCM. This model demonstrates good performance, addresses data imbalance, and
suggests that AF is associated with a more severe cardiac HCM phenotype.

    

### [[2109.09210] Identifying Ventricular Arrhythmias and Their Predictors by Applying Machine Learning Methods to Electronic Health Records in Patients With Hypertrophic Cardiomyopathy(HCM-VAr-Risk Model)](http://arxiv.org/abs/2109.09210)


  Clinical risk stratification for sudden cardiac death (SCD) in hypertrophic
cardiomyopathy (HC) employs rules derived from American College of Cardiology
Foundation/American Heart Association (ACCF/AHA) guidelines or the HCM Risk-SCD
model (C-index of 0.69), which utilize a few clinical variables. We assessed
whether data-driven machine learning methods that consider a wider range of
variables can effectively identify HC patients with ventricular arrhythmias
(VAr) that lead to SCD. We scanned the electronic health records of 711 HC
patients for sustained ventricular tachycardia or ventricular fibrillation.
Patients with ventricular tachycardia or ventricular fibrillation (n = 61) were
tagged as VAr cases and the remaining (n = 650) as non-VAr. The 2-sample t test
and information gain criterion were used to identify the most informative
clinical variables that distinguish VAr from non-VAr; patient records were
reduced to include only these variables. Data imbalance stemming from low
number of VAr cases was addressed by applying a combination of over- and
under-sampling strategies.We trained and tested multiple classifiers under this
sampling approach, showing effective classification. We evaluated 93 clinical
variables, of which 22 proved predictive of VAr. The ensemble of logistic
regression and naive Bayes classifiers, trained based on these 22 variables and
corrected for data imbalance, was most effective in separating VAr from non-VAr
cases (sensitivity = 0.73, specificity = 0.76, C-index = 0.83). Our method
(HCM-VAr-Risk Model) identified 12 new predictors of VAr, in addition to 10
established SCD predictors. In conclusion, this is the first application of
machine learning for identifying HC patients with VAr, using clinical
attributes.

    

### [[2109.09212] Generalized Translation and Scale Invariant Online Algorithm for Adversarial Multi-Armed Bandits](http://arxiv.org/abs/2109.09212)


  We study the adversarial multi-armed bandit problem and create a completely
online algorithmic framework that is invariant under arbitrary translations and
scales of the arm losses. We study the expected performance of our algorithm
against a generic competition class, which makes it applicable for a wide
variety of problem scenarios. Our algorithm works from a universal prediction
perspective and the performance measure used is the expected regret against
arbitrary arm selection sequences, which is the difference between our losses
and a competing loss sequence. The competition class can be designed to include
fixed arm selections, switching bandits, contextual bandits, or any other
competition of interest. The sequences in the competition class are generally
determined by the specific application at hand and should be designed
accordingly. Our algorithm neither uses nor needs any preliminary information
about the loss sequences and is completely online. Its performance bounds are
the second order bounds in terms of sum of the squared losses, where any affine
transform of the losses has no effect on the normalized regret.

    

### [[2109.09222] Multiscale Manifold Warping](http://arxiv.org/abs/2109.09222)


  Many real-world applications require aligning two temporal sequences,
including bioinformatics, handwriting recognition, activity recognition, and
human-robot coordination. Dynamic Time Warping (DTW) is a popular alignment
method, but can fail on high-dimensional real-world data where the dimensions
of aligned sequences are often unequal. In this paper, we show that exploiting
the multiscale manifold latent structure of real-world data can yield improved
alignment. We introduce a novel framework called Warping on Wavelets (WOW) that
integrates DTW with a a multi-scale manifold learning framework called
Diffusion Wavelets. We present a theoretical analysis of the WOW family of
algorithms and show that it outperforms previous state of the art methods, such
as canonical time warping (CTW) and manifold warping, on several real-world
datasets.

    

### [[2109.09227] ARCA23K: An audio dataset for investigating open-set label noise](http://arxiv.org/abs/2109.09227)


  The availability of audio data on sound sharing platforms such as Freesound
gives users access to large amounts of annotated audio. Utilising such data for
training is becoming increasingly popular, but the problem of label noise that
is often prevalent in such datasets requires further investigation. This paper
introduces ARCA23K, an Automatically Retrieved and Curated Audio dataset
comprised of over 23000 labelled Freesound clips. Unlike past datasets such as
FSDKaggle2018 and FSDnoisy18K, ARCA23K facilitates the study of label noise in
a more controlled manner. We describe the entire process of creating the
dataset such that it is fully reproducible, meaning researchers can extend our
work with little effort. We show that the majority of labelling errors in
ARCA23K are due to out-of-vocabulary audio clips, and we refer to this type of
label noise as open-set label noise. Experiments are carried out in which we
study the impact of label noise in terms of classification performance and
representation learning.

    

### [[2109.09228] Rethnicity: Predicting Ethnicity from Names](http://arxiv.org/abs/2109.09228)


  I provide an R package, \texttt{rethnicity}, for predicting ethnicity from
names. I use the Bidirectional LSTM as the model and Florida Voter Registration
as training data. Special care is given for the accuracy of minority groups, by
adjusting the imbalance in the dataset. I also compare the availability,
accuracy, and performance with other solutions for predicting ethnicity from
names. Sample code snippet and analysis of the DIME dataset are also shown as
applications of the package.

    

### [[2109.09232] UPV at CheckThat! 2021: Mitigating Cultural Differences for Identifying Multilingual Check-worthy Claims](http://arxiv.org/abs/2109.09232)


  Identifying check-worthy claims is often the first step of automated
fact-checking systems. Tackling this task in a multilingual setting has been
understudied. Encoding inputs with multilingual text representations could be
one approach to solve the multilingual check-worthiness detection. However,
this approach could suffer if cultural bias exists within the communities on
determining what is this http URL this paper, we propose a language
identification task as an auxiliary task to mitigate unintended bias.With this
purpose, we experiment joint training by using the datasets from CLEF-2021
CheckThat!, that contain tweets in English, Arabic, Bulgarian, Spanish and
Turkish. Our results show that joint training of language identification and
check-worthy claim detection tasks can provide performance gains for some of
the selected languages.

    

### [[2109.09233] Unified and Multilingual Author Profiling for Detecting Haters](http://arxiv.org/abs/2109.09233)


  This paper presents a unified user profiling framework to identify hate
speech spreaders by processing their tweets regardless of the language. The
framework encodes the tweets with sentence transformers and applies an
attention mechanism to select important tweets for learning user profiles.
Furthermore, the attention layer helps to explain why a user is a hate speech
spreader by producing attention weights at both token and post level. Our
proposed model outperformed the state-of-the-art multilingual transformer
models.

    

### [[2109.09238] A Data-Driven Convergence Bidding Strategy Based on Reverse Engineering of Market Participants' Performance: A Case of California ISO](http://arxiv.org/abs/2109.09238)


  Convergence bidding, a.k.a., virtual bidding, has been widely adopted in
wholesale electricity markets in recent years. It provides opportunities for
market participants to arbitrage on the difference between the day-ahead market
locational marginal prices and the real-time market locational marginal prices.
Given the fact that convergence bids (CBs) have a significant impact on the
operation of electricity markets, it is important to understand how market
participants strategically select their CBs in real-world. We address this open
problem with focus on the electricity market that is operated by the California
ISO. In this regard, we use the publicly available electricity market data to
learn, characterize, and evaluate different types of convergence bidding
strategies that are currently used by market participants. Our analysis
includes developing a data-driven reverse engineering method that we apply to
three years of real-world data. Our analysis involves feature selection and
density-based data clustering. It results in identifying three main clusters of
CB strategies in the California ISO market. Different characteristics and the
performance of each cluster of strategies are analyzed. Interestingly, we
unmask a common real-world strategy that does not match any of the existing
strategic convergence bidding methods in the literature. Next, we build upon
the lessons learned from the existing real-world strategies to propose a new CB
strategy that can significantly outperform them. Our analysis includes
developing a new strategy for convergence bidding. The new strategy has three
steps: net profit maximization by capturing price spikes, dynamic node
labeling, and strategy selection algorithm. We show through case studies that
the annual net profit for the most lucrative market participants can increase
by over 40% if the proposed convergence bidding strategy is used.

    

### [[2109.09241] Robust Automated Framework for COVID-19 Disease Identification from a Multicenter Dataset of Chest CT Scans](http://arxiv.org/abs/2109.09241)


  The objective of this study is to develop a robust deep learning-based
framework to distinguish COVID-19, Community-Acquired Pneumonia (CAP), and
Normal cases based on chest CT scans acquired in different imaging centers
using various protocols, and radiation doses. We showed that while our proposed
model is trained on a relatively small dataset acquired from only one imaging
center using a specific scanning protocol, the model performs well on
heterogeneous test sets obtained by multiple scanners using different technical
parameters. We also showed that the model can be updated via an unsupervised
approach to cope with the data shift between the train and test sets and
enhance the robustness of the model upon receiving a new external dataset from
a different center. We adopted an ensemble architecture to aggregate the
predictions from multiple versions of the model. For initial training and
development purposes, an in-house dataset of 171 COVID-19, 60 CAP, and 76
Normal cases was used, which contained volumetric CT scans acquired from one
imaging center using a constant standard radiation dose scanning protocol. To
evaluate the model, we collected four different test sets retrospectively to
investigate the effects of the shifts in the data characteristics on the
model's performance. Among the test cases, there were CT scans with similar
characteristics as the train set as well as noisy low-dose and ultra-low dose
CT scans. In addition, some test CT scans were obtained from patients with a
history of cardiovascular diseases or surgeries. The entire test dataset used
in this study contained 51 COVID-19, 28 CAP, and 51 Normal cases. Experimental
results indicate that our proposed framework performs well on all test sets
achieving total accuracy of 96.15% (95%CI: [91.25-98.74]), COVID-19 sensitivity
of 96.08% (95%CI: [86.54-99.5]), CAP sensitivity of 92.86% (95%CI:
[76.50-99.19]).

    

### [[1810.03393] Hierarchical clustering that takes advantage of both density-peak and density-connectivity](http://arxiv.org/abs/1810.03393)


  This paper focuses on density-based clustering, particularly the Density Peak
(DP) algorithm and the one based on density-connectivity DBSCAN; and proposes a
new method which takes advantage of the individual strengths of these two
methods to yield a density-based hierarchical clustering algorithm. Our
investigation begins with formally defining the types of clusters DP and DBSCAN
are designed to detect; and then identifies the kinds of distributions that DP
and DBSCAN individually fail to detect all clusters in a dataset. These
identified weaknesses inspire us to formally define a new kind of clusters and
propose a new method called DC-HDP to overcome these weaknesses to identify
clusters with arbitrary shapes and varied densities. In addition, the new
method produces a richer clustering result in terms of hierarchy or dendrogram
for better cluster structures understanding. Our empirical evaluation results
show that DC-HDP produces the best clustering results on 14 datasets in
comparison with 7 state-of-the-art clustering algorithms.

    

### [[1912.04792] Training Provably Robust Models by Polyhedral Envelope Regularization](http://arxiv.org/abs/1912.04792)


  Training certifiable neural networks enables one to obtain models with
robustness guarantees against adversarial attacks. In this work, we introduce a
framework to bound the adversary-free region in the neighborhood of the input
data by a polyhedral envelope, which yields finer-grained certified robustness.
We further introduce polyhedral envelope regularization (PER) to encourage
larger polyhedral envelopes and thus improve the provable robustness of the
models. We demonstrate the flexibility and effectiveness of our framework on
standard benchmarks; it applies to networks of different architectures and
general activation functions. Compared with the state-of-the-art methods, PER
has very little computational overhead and better robustness guarantees without
over-regularizing the model.

    

### [[1912.06087] Attention network forecasts time-to-failure in laboratory shear experiments](http://arxiv.org/abs/1912.06087)


  Rocks under stress deform by creep mechanisms that include formation and slip
on small-scale internal cracks. Intragranular cracks and slip along grain
contacts release energy as elastic waves termed acoustic emissions (AE). AEs
are thought to contain predictive information that can be used for fault
failure forecasting. Here we present a method using unsupervised classification
and an attention network to forecast labquakes using AE waveform features. Our
data were generated in a laboratory setting using a biaxial shearing device
with granular fault gouge intended to mimic the conditions of tectonic faults.
Here we analyzed the temporal evolution of AEs generated throughout several
hundred laboratory earthquake cycles. We used a Conscience Self-Organizing Map
(CSOM) to perform topologically ordered vector quantization based on waveform
properties. The resulting map was used to interactively cluster AEs. We
examined the clusters over time to identify those with predictive ability.
Finally, we used a variety of LSTM and attention-based networks to test the
predictive power of the AE clusters. By tracking cumulative waveform features
over the seismic cycle, the network is able to forecast the time-to-failure
(TTF) of lab earthquakes. Our results show that analyzing the data to isolate
predictive signals and using a more sophisticated network architecture are key
to robustly forecasting labquakes. In the future, this method could be applied
on tectonic faults monitor earthquakes and augment current early warning
systems.

    

### [[2002.05082] Results on the algebraic matroid of the determinantal variety](http://arxiv.org/abs/2002.05082)


  We make progress towards characterizing the algebraic matroid of the
determinantal variety. We present a family of base sets of the matroid and
conjecture these are all the base sets. This conjecture is reduced to a purely
combinatorial statement, which is proved for special cases. Our results rely on
the combinatorial notion of relaxed supports of linkage matching fields that we
introduce, our interpretation of the problem of completing a matrix of bounded
rank from a subset of its entries as a linear section problem on the
Grassmannian, and a connection that we draw with a class of local coordinates
on the Grassmannian described by Sturmfels $\&$ Zelevinsky.

    

### [[2002.06478] Learning to Group: A Bottom-Up Framework for 3D Part Discovery in Unseen Categories](http://arxiv.org/abs/2002.06478)


  We address the problem of discovering 3D parts for objects in unseen
categories. Being able to learn the geometry prior of parts and transfer this
prior to unseen categories pose fundamental challenges on data-driven shape
segmentation approaches. Formulated as a contextual bandit problem, we propose
a learning-based agglomerative clustering framework which learns a grouping
policy to progressively group small part proposals into bigger ones in a
bottom-up fashion. At the core of our approach is to restrict the local context
for extracting part-level features, which encourages the generalizability to
unseen categories. On the large-scale fine-grained 3D part dataset, PartNet, we
demonstrate that our method can transfer knowledge of parts learned from 3
training categories to 21 unseen testing categories without seeing any
annotated samples. Quantitative comparisons against four shape segmentation
baselines shows that our approach achieve the state-of-the-art performance.

    

### [[2002.08246] A Unified Convergence Analysis for Shuffling-Type Gradient Methods](http://arxiv.org/abs/2002.08246)


  In this paper, we propose a unified convergence analysis for a class of
generic shuffling-type gradient methods for solving finite-sum optimization
problems. Our analysis works with any sampling without replacement strategy and
covers many known variants such as randomized reshuffling, deterministic or
randomized single permutation, and cyclic and incremental gradient schemes. We
focus on two different settings: strongly convex and nonconvex problems, but
also discuss the non-strongly convex case. Our main contribution consists of
new non-asymptotic and asymptotic convergence rates for a wide class of
shuffling-type gradient methods in both nonconvex and convex settings. We also
study uniformly randomized shuffling variants with different learning rates and
model assumptions. While our rate in the nonconvex case is new and
significantly improved over existing works under standard assumptions, the rate
on the strongly convex one matches the existing best-known rates prior to this
paper up to a constant factor without imposing a bounded gradient condition.
Finally, we empirically illustrate our theoretical results via two numerical
examples: nonconvex logistic regression and neural network training examples.
As byproducts, our results suggest some appropriate choices for diminishing
learning rates in certain shuffling variants.

    

### [[2002.08937] Geometric Interpretation of Running Nyström-Based Kernel Machines and Error Analysis](http://arxiv.org/abs/2002.08937)


  Recently, Nyström method has proved its prominence empirically and
theoretically in speeding up the training of kernel machines while retaining
satisfactory performances and accuracy. So far, there are several different
approaches proposed to exploit Nyström method in scaling up kernel
machines. However, there is no comparative study over these approaches, and
they were individually analyzed for specific types of kernel machines.
Therefore, it remains a question that the philosophy of which approach is more
promising when it extends to other kernel machines. In this work, motivated by
the column inclusion property of Gram matrices, we develop a new approach with
a clear geometric interpretation for running Nyström-based kernel machines.
We show that the other two well-studied approaches can be equivalently
transformed to be our proposed one. Consequently, analysis established for the
proposed approach also works for these two. Particularly, our proposed approach
makes it possible to develop approximation errors in a general setting.
Besides, our analysis also manifests the relations among the aforementioned two
approaches and another naive one. First, the analytical forms of the
corresponding approximate solutions are only at odds with one term. Second, the
naive approach can be implemented efficiently by sharing the same training
procedure with others. These analytical results lead to the conjecture that the
naive approach can provide more accurate approximate solutions than the other
two sophisticated approaches. Since our analysis also offers ways for computing
the accuracy of these approximate solutions, we run experiments with
classification tasks to confirm our conjecture.

    

### [[2002.09545] RobustTAD: Robust Time Series Anomaly Detection via Decomposition and Convolutional Neural Networks](http://arxiv.org/abs/2002.09545)


  The monitoring and management of numerous and diverse time series data at
Alibaba Group calls for an effective and scalable time series anomaly detection
service. In this paper, we propose RobustTAD, a Robust Time series Anomaly
Detection framework by integrating robust seasonal-trend decomposition and
convolutional neural network for time series data. The seasonal-trend
decomposition can effectively handle complicated patterns in time series, and
meanwhile significantly simplifies the architecture of the neural network,
which is an encoder-decoder architecture with skip connections. This
architecture can effectively capture the multi-scale information from time
series, which is very useful in anomaly detection. Due to the limited labeled
data in time series anomaly detection, we systematically investigate data
augmentation methods in both time and frequency domains. We also introduce
label-based weight and value-based weight in the loss function by utilizing the
unbalanced nature of the time series anomaly detection problem. Compared with
the widely used forecasting-based anomaly detection algorithms,
decomposition-based algorithms, traditional statistical algorithms, as well as
recent neural network based algorithms, RobustTAD performs significantly better
on public benchmark datasets. It is deployed as a public online service and
widely adopted in different business scenarios at Alibaba Group.

    

### [[2002.12478] Time Series Data Augmentation for Deep Learning: A Survey](http://arxiv.org/abs/2002.12478)


  Deep learning performs remarkably well on many time series analysis tasks
recently. The superior performance of deep neural networks relies heavily on a
large number of training data to avoid overfitting. However, the labeled data
of many real-world time series applications may be limited such as
classification in medical time series and anomaly detection in AIOps. As an
effective way to enhance the size and quality of the training data, data
augmentation is crucial to the successful application of deep learning models
on time series data. In this paper, we systematically review different data
augmentation methods for time series. We propose a taxonomy for the reviewed
methods, and then provide a structured review for these methods by highlighting
their strengths and limitations. We also empirically compare different data
augmentation methods for different tasks including time series classification,
anomaly detection, and forecasting. Finally, we discuss and highlight five
future directions to provide useful research guidance.

    

### [[2003.06210] Identification of AC Networks via Online Learning](http://arxiv.org/abs/2003.06210)


  The increasing penetration of intermittent distributed energy resources in
power networks calls for novel planning and control methodologies which hinge
on detailed knowledge of the grid. However, reliable information concerning the
system topology and parameters may be missing or outdated for temporally
varying electric distribution networks. This paper proposes an online learning
procedure to estimate the network admittance matrix capturing topological
information and line parameters. We start off by providing a recursive
identification algorithm exploiting phasor measurements of voltages and
currents. With the goal of accelerating convergence, we subsequently complement
our base algorithm with a design-of-experiment procedure which maximizes the
information content of data at each step by computing optimal voltage
excitations. Our approach improves on existing techniques, and its
effectiveness is substantiated by numerical studies on realistic testbeds.

    

### [[2004.01382] Effective Fusion of Deep Multitasking Representations for Robust Visual Tracking](http://arxiv.org/abs/2004.01382)


  Visual object tracking remains an active research field in computer vision
due to persisting challenges with various problem-specific factors in
real-world scenes. Many existing tracking methods based on discriminative
correlation filters (DCFs) employ feature extraction networks (FENs) to model
the target appearance during the learning process. However, using deep feature
maps extracted from FENs based on different residual neural networks (ResNets)
has not previously been investigated. This paper aims to evaluate the
performance of twelve state-of-the-art ResNet-based FENs in a DCF-based
framework to determine the best for visual tracking purposes. First, it ranks
their best feature maps and explores the generalized adoption of the best
ResNet-based FEN into another DCF-based method. Then, the proposed method
extracts deep semantic information from a fully convolutional FEN and fuses it
with the best ResNet-based feature maps to strengthen the target representation
in the learning process of continuous convolution filters. Finally, it
introduces a new and efficient semantic weighting method (using semantic
segmentation feature maps on each video frame) to reduce the drift problem.
Extensive experimental results on the well-known OTB-2013, OTB-2015, TC-128 and
VOT-2018 visual tracking datasets demonstrate that the proposed method
effectively outperforms state-of-the-art methods in terms of precision and
robustness of visual tracking.

    

### [[2006.09245] An empirical study on using CNNs for fast radio signal prediction](http://arxiv.org/abs/2006.09245)


  Accurate radio frequency power prediction in a geographic region is a
computationally expensive part of finding the optimal transmitter location
using a ray tracing software. We empirically analyze the viability of deep
learning models to speed up this process. Specifically, deep learning methods
including CNNs and UNET are typically used for segmentation, and can also be
employed in power prediction tasks. We consider a dataset that consists of
radio frequency power values for five different regions with four different
frame dimensions. We compare deep learning-based prediction models including
RadioUNET and four different variations of the UNET model for the power
prediction task. More complex UNET variations improve the model on higher
resolution frames such as 256x256. However, using the same models on lower
resolutions results in overfitting and simpler models perform better. Our
detailed numerical analysis shows that the deep learning models are effective
in power prediction and they are able to generalize well to the new regions.

    

### [[2006.09373] The shape and simplicity biases of adversarially robust ImageNet-trained CNNs](http://arxiv.org/abs/2006.09373)


  Adversarial training has been the topic of dozens of studies and a leading
method for defending against adversarial attacks. Yet, it remains largely
unknown (a) how adversarially-robust ImageNet classifiers (R classifiers)
generalize to out-of-distribution examples; and (b) how their generalization
capability relates to their hidden representations. In this paper, we perform a
thorough, systematic study to answer these two questions across AlexNet,
GoogLeNet, and ResNet-50 architectures. We found that while standard ImageNet
classifiers have a strong texture bias, their R counterparts rely heavily on
shapes. Remarkably, adversarial training induces three simplicity biases into
hidden neurons in the process of 'robustifying' the network. That is, each
convolutional neuron in R networks often changes to detecting (1) pixel-wise
smoother patterns i.e. a mechanism that blocks high-frequency noise from
passing through the network; (2) more lower-level features i.e. textures and
colors (instead of objects); and (3) fewer types of inputs. Our findings reveal
the interesting mechanisms that made networks more adversarially robust and
also explain some recent findings e.g. why R networks benefit from much larger
capacity (Xie and Yuille, 2020) and can act as a strong image prior in image
synthesis (Santurkar et al., 2019).

    

### [[2007.01544] A Conceptual Framework for Externally-influenced Agents: An Assisted Reinforcement Learning Review](http://arxiv.org/abs/2007.01544)


  A long-term goal of reinforcement learning agents is to be able to perform
tasks in complex real-world scenarios. The use of external information is one
way of scaling agents to more complex problems. However, there is a general
lack of collaboration or interoperability between different approaches using
external information. In this work, while reviewing externally-influenced
methods, we propose a conceptual framework and taxonomy for assisted
reinforcement learning, aimed at fostering collaboration by classifying and
comparing various methods that use external information in the learning
process. The proposed taxonomy details the relationship between the external
information source and the learner agent, highlighting the process of
information decomposition, structure, retention, and how it can be used to
influence agent learning. As well as reviewing state-of-the-art methods, we
identify current streams of reinforcement learning that use external
information in order to improve the agent's performance and its decision-making
process. These include heuristic reinforcement learning, interactive
reinforcement learning, learning from demonstration, transfer learning, and
learning from multiple sources, among others. These streams of reinforcement
learning operate with the shared objective of scaffolding the learner agent.
Lastly, we discuss further possibilities for future work in the field of
assisted reinforcement learning systems.

    

### [[2007.03253] Doubly infinite residual neural networks: a diffusion process approach](http://arxiv.org/abs/2007.03253)


  Modern neural networks (NN) featuring a large number of layers (depth) and
units per layer (width) have achieved a remarkable performance across many
domains. While there exists a vast literature on the interplay between
infinitely wide NNs and Gaussian processes, a little is known about analogous
interplays with respect to infinitely deep NNs. NNs with independent and
identically distributed (i.i.d.) initializations exhibit undesirable forward
and backward propagation properties as the number of layers increases. To
overcome these drawbacks, Peluchetti and Favaro (2020) considered
fully-connected residual networks (ResNets) with network's parameters
initialized by means of distributions that shrink as the number of layers
increases, thus establishing an interplay between infinitely deep ResNets and
solutions to stochastic differential equations, i.e. diffusion processes, and
showing that infinitely deep ResNets does not suffer from undesirable
forward-propagation properties. In this paper, we review the results of
Peluchetti and Favaro (2020), extending them to convolutional ResNets, and we
establish analogous backward-propagation results, which directly relate to the
problem of training fully-connected deep ResNets. Then, we investigate the more
general setting of doubly infinite NNs, where both network's width and
network's depth grow unboundedly. We focus on doubly infinite fully-connected
ResNets, for which we consider i.i.d. initializations. Under this setting, we
show that the dynamics of quantities of interest converge, at initialization,
to deterministic limits. This allow us to provide analytical expressions for
inference, both in the case of weakly trained and fully trained ResNets. Our
results highlight a limited expressive power of doubly infinite ResNets when
the unscaled network's parameters are i.i.d. and the residual blocks are
shallow.

    

### [[2008.07428] Fast decentralized non-convex finite-sum optimization with recursive variance reduction](http://arxiv.org/abs/2008.07428)


  This paper considers decentralized minimization of $N:=nm$ smooth non-convex
cost functions equally divided over a directed network of $n$ nodes.
Specifically, we describe a stochastic first-order gradient method, called
GT-SARAH, that employs a SARAH-type variance reduction technique and gradient
tracking (GT) to address the stochastic and decentralized nature of the
problem. We show that GT-SARAH, with appropriate algorithmic parameters, finds
an $\epsilon$-accurate first-order stationary point with
$O\big(\max\big\{N^{\frac{1}{2}},n(1-\lambda)^{-2},n^{\frac{2}{3}}m^{\frac{1}{3}}(1-\lambda)^{-1}\big\}L\epsilon^{-2}\big)$
gradient complexity, where ${(1-\lambda)\in(0,1]}$ is the spectral gap of the
network weight matrix and $L$ is the smoothness parameter of the cost
functions. This gradient complexity outperforms that of the existing
decentralized stochastic gradient methods. In particular, in a big-data regime
such that ${n = O(N^{\frac{1}{2}}(1-\lambda)^{3})}$, this gradient complexity
furthers reduces to ${O(N^{\frac{1}{2}}L\epsilon^{-2})}$, independent of the
network topology, and matches that of the centralized near-optimal
variance-reduced methods. Moreover, in this regime GT-SARAH achieves a
non-asymptotic linear speedup, in that, the total number of gradient
computations at each node is reduced by a factor of $1/n$ compared to the
centralized near-optimal algorithms that perform all gradient computations at a
single node. To the best of our knowledge, GT-SARAH is the first algorithm that
achieves this property. In addition, we show that appropriate choices of local
minibatch size balance the trade-offs between the gradient and communication
complexity of GT-SARAH. Over infinite time horizon, we establish that all nodes
in GT-SARAH asymptotically achieve consensus and converge to a first-order
stationary point in the almost sure and mean-squared sense.

    

### [[2009.02467] Binary Classification as a Phase Separation Process](http://arxiv.org/abs/2009.02467)


  We propose a new binary classification model called Phase Separation Binary
Classifier (PSBC). It consists of a discretization of a nonlinear
reaction-diffusion equation coupled with an Ordinary Differential Equation, and
is inspired by fluids behavior, namely, on how binary fluids phase separate.
Thus, parameters and hyperparameters have physical meaning, whose effects are
studied in several different scenarios.
PSBC's equations can be seen as a dynamical system whose coefficients are
trainable weights, with a similar architecture to that of a Recurrent Neural
Network. As such, forward propagation amounts to an initial value problem.
Boundary conditions are also present, bearing similarity with figure padding
techniques in Computer Vision. Model compression is exploited in several ways,
with weight sharing taking place both across and within layers.
The model is tested on pairs of digits of the classical MNIST database. An
associated multiclass classifier is also constructed using a combination of
Ensemble Learning and one versus one techniques. It is also shown how the PSBC
can be combined with other methods - like aggregation and PCA - in order to
construct better binary classifiers. The role of boundary conditions and
viscosity is thoroughly studied in the case of digits ``0'' and ``1''.

    

### [[2010.09621] Importance Reweighting for Biquality Learning](http://arxiv.org/abs/2010.09621)


  The field of Weakly Supervised Learning (WSL) has recently seen a surge of
popularity, with numerous papers addressing different types of "supervision
deficiencies", namely: poor quality, non adaptability, and insufficient
quantity of labels. Regarding quality, label noise can be of different types,
including completely-at-random, at-random or even not-at-random. All these
kinds of label noise are addressed separately in the literature, leading to
highly specialized approaches. This paper proposes an original, encompassing,
view of Weakly Supervised Learning, which results in the design of generic
approaches capable of dealing with any kind of label noise. For this purpose,
an alternative setting called "Biquality data" is used. It assumes that a small
trusted dataset of correctly labeled examples is available, in addition to an
untrusted dataset of noisy examples. In this paper, we propose a new
reweigthing scheme capable of identifying noncorrupted examples in the
untrusted dataset. This allows one to learn classifiers using both datasets.
Extensive experiments that simulate several types of label noise and that vary
the quality and quantity of untrusted examples, demonstrate that the proposed
approach outperforms baselines and state-of-the-art approaches.

    

### [[2011.08091] Tweet Sentiment Quantification: An Experimental Re-Evaluation](http://arxiv.org/abs/2011.08091)


  Sentiment quantification is the task of training, by means of supervised
learning, estimators of the relative frequency (also called ``prevalence'') of
sentiment-related classes (such as \textsf{Positive}, \textsf{Neutral},
\textsf{Negative}) in a sample of unlabelled texts. This task is especially
important when these texts are tweets, since the final goal of most sentiment
classification efforts carried out on Twitter data is actually quantification
(and not the classification of individual tweets). It is well-known that
solving quantification by means of ``classify and count'' (i.e., by classifying
all unlabelled items by means of a standard classifier and counting the items
that have been assigned to a given class) is less than optimal in terms of
accuracy, and that more accurate quantification methods exist. Gao and
Sebastiani (2016) carried out a systematic comparison of quantification methods
on the task of tweet sentiment quantification. In hindsight, we observe that
the experimental protocol followed in that work was weak, and that the
reliability of the conclusions that were drawn from the results is thus
questionable. We now re-evaluate those quantification methods (plus a few more
modern ones) on exactly the same same datasets, this time following a now
consolidated and much more robust experimental protocol (which also involves
simulating the presence, in the test data, of class prevalence values very
different from those of the training set). This experimental protocol (even
without counting the newly added methods) involves a number of experiments
5,775 times larger than that of the original study. The results of our
experiments are dramatically different from those obtained by Gao and
Sebastiani, and they provide a different, much more solid understanding of the
relative strengths and weaknesses of different sentiment quantification
methods.

    

### [[2011.08484] Combining Reinforcement Learning with Model Predictive Control for On-Ramp Merging](http://arxiv.org/abs/2011.08484)


  We consider the problem of designing an algorithm to allow a car to
autonomously merge on to a highway from an on-ramp. Two broad classes of
techniques have been proposed to solve motion planning problems in autonomous
driving: Model Predictive Control (MPC) and Reinforcement Learning (RL). In
this paper, we first establish the strengths and weaknesses of state-of-the-art
MPC and RL-based techniques through simulations. We show that the performance
of the RL agent is worse than that of the MPC solution from the perspective of
safety and robustness to out-of-distribution traffic patterns, i.e., traffic
patterns which were not seen by the RL agent during training. On the other
hand, the performance of the RL agent is better than that of the MPC solution
when it comes to efficiency and passenger comfort. We subsequently present an
algorithm which blends the model-free RL agent with the MPC solution and show
that it provides better trade-offs between all metrics -- passenger comfort,
efficiency, crash rate and robustness.

    

### [[2011.08932] Analyzing and Mitigating JPEG Compression Defects in Deep Learning](http://arxiv.org/abs/2011.08932)


  With the proliferation of deep learning methods, many computer vision
problems which were considered academic are now viable in the consumer setting.
One drawback of consumer applications is lossy compression, which is necessary
from an engineering standpoint to efficiently and cheaply store and transmit
user images. Despite this, there has been little study of the effect of
compression on deep neural networks and benchmark datasets are often losslessly
compressed or compressed at high quality. Here we present a unified study of
the effects of JPEG compression on a range of common tasks and datasets. We
show that there is a significant penalty on common performance metrics for high
compression. We test several methods for mitigating this penalty, including a
novel method based on artifact correction which requires no labels to train.

    

### [[2011.12956] Reinforcement Learning for Robust Missile Autopilot Design](http://arxiv.org/abs/2011.12956)


  Designing missiles' autopilot controllers has been a complex task, given the
extensive flight envelope and the nonlinear flight dynamics. A solution that
can excel both in nominal performance and in robustness to uncertainties is
still to be found. While Control Theory often debouches into parameters'
scheduling procedures, Reinforcement Learning has presented interesting results
in ever more complex tasks, going from videogames to robotic tasks with
continuous action domains. However, it still lacks clearer insights on how to
find adequate reward functions and exploration strategies. To the best of our
knowledge, this work is pioneer in proposing Reinforcement Learning as a
framework for flight control. In fact, it aims at training a model-free agent
that can control the longitudinal flight of a missile, achieving optimal
performance and robustness to uncertainties. To that end, under TRPO's
methodology, the collected experience is augmented according to HER, stored in
a replay buffer and sampled according to its significance. Not only does this
work enhance the concept of prioritized experience replay into BPER, but it
also reformulates HER, activating them both only when the training progress
converges to suboptimal policies, in what is proposed as the SER methodology.
Besides, the Reward Engineering process is carefully detailed. The results show
that it is possible both to achieve the optimal performance and to improve the
agent's robustness to uncertainties (with low damage on nominal performance) by
further training it in non-nominal environments, therefore validating the
proposed approach and encouraging future research in this field.

    

### [[2012.04030] Statistical Mechanics of Deep Linear Neural Networks: The Back-Propagating Kernel Renormalization](http://arxiv.org/abs/2012.04030)


  The success of deep learning in many real-world tasks has triggered an
intense effort to understand the power and limitations of deep learning in the
training and generalization of complex tasks, so far with limited progress. In
this work, we study the statistical mechanics of learning in Deep Linear Neural
Networks (DLNNs) in which the input-output function of an individual unit is
linear. Despite the linearity of the units, learning in DLNNs is nonlinear,
hence studying its properties reveals some of the features of nonlinear Deep
Neural Networks (DNNs). Importantly, we solve exactly the network properties
following supervised learning using an equilibrium Gibbs distribution in the
weight space. To do this, we introduce the Back-Propagating Kernel
Renormalization (BPKR), which allows for the incremental integration of the
network weights starting from the network output layer and progressing backward
until the first layer's weights are integrated out. This procedure allows us to
evaluate important network properties, such as its generalization error, the
role of network width and depth, the impact of the size of the training set,
and the effects of weight regularization and learning stochasticity. BPKR does
not assume specific statistics of the input or the task's output. Furthermore,
by performing partial integration of the layers, the BPKR allows us to compute
the properties of the neural representations across the different hidden
layers. We have proposed an extension of the BPKR to nonlinear DNNs with ReLU.
Surprisingly, our numerical simulations reveal that despite the nonlinearity,
the predictions of our theory are largely shared by ReLU networks in a wide
regime of parameters. Our work is the first exact statistical mechanical study
of learning in a family of DNNs, and the first successful theory of learning
through successive integration of DoFs in the learned weight space.

    

### [[2012.11329] CARLA Real Traffic Scenarios -- novel training ground and benchmark for autonomous driving](http://arxiv.org/abs/2012.11329)


  This work introduces interactive traffic scenarios in the CARLA simulator,
which are based on real-world traffic. We concentrate on tactical tasks lasting
several seconds, which are especially challenging for current control methods.
The CARLA Real Traffic Scenarios (CRTS) is intended to be a training and
testing ground for autonomous driving systems. To this end, we open-source the
code under a permissive license and present a set of baseline policies. CRTS
combines the realism of traffic scenarios and the flexibility of simulation. We
use it to train agents using a reinforcement learning algorithm. We show how to
obtain competitive polices and evaluate experimentally how observation types
and reward schemes affect the training process and the resulting agent's
behavior.

    

### [[2101.03418] Deep Reinforcement Learning with Function Properties in Mean Reversion Strategies](http://arxiv.org/abs/2101.03418)


  Over the past decades, researchers have been pushing the limits of Deep
Reinforcement Learning (DRL). Although DRL has attracted substantial interest
from practitioners, many are blocked by having to search through a plethora of
available methodologies that are seemingly alike, while others are still
building RL agents from scratch based on classical theories. To address the
aforementioned gaps in adopting the latest DRL methods, I am particularly
interested in testing out if any of the recent technology developed by the
leads in the field can be readily applied to a class of optimal trading
problems. Unsurprisingly, many prominent breakthroughs in DRL are investigated
and tested on strategic games: from AlphaGo to AlphaStar and at about the same
time, OpenAI Five. Thus, in this writing, I want to show precisely how to use a
DRL library that is initially built for games in a fundamental trading problem;
mean reversion. And by introducing a framework that incorporates
economically-motivated function properties, I also demonstrate, through the
library, a highly-performant and convergent DRL solution to decision-making
financial problems in general.

    

### [[2101.12100] Increasing the Confidence of Deep Neural Networks by Coverage Analysis](http://arxiv.org/abs/2101.12100)


  The great performance of machine learning algorithms and deep neural networks
in several perception and control tasks is pushing the industry to adopt such
technologies in safety-critical applications, as autonomous robots and
self-driving vehicles. At present, however, several issues need to be solved to
make deep learning methods more trustworthy, predictable, safe, and secure
against adversarial attacks. Although several methods have been proposed to
improve the trustworthiness of deep neural networks, most of them are tailored
for specific classes of adversarial examples, hence failing to detect other
corner cases or unsafe inputs that heavily deviate from the training samples.
This paper presents a lightweight monitoring architecture based on coverage
paradigms to enhance the model robustness against different unsafe inputs. In
particular, four coverage analysis methods are proposed and tested in the
architecture for evaluating multiple detection logics. Experimental results
show that the proposed approach is effective in detecting both powerful
adversarial examples and out-of-distribution inputs, introducing limited
extra-execution time and memory requirements.

    

### [[2102.01875] AHAR: Adaptive CNN for Energy-efficient Human Activity Recognition in Low-power Edge Devices](http://arxiv.org/abs/2102.01875)


  Human Activity Recognition (HAR) is one of the key applications of health
monitoring that requires continuous use of wearable devices to track daily
activities. This paper proposes an Adaptive CNN for energy-efficient HAR (AHAR)
suitable for low-power edge devices. Unlike traditional early exit architecture
that makes the exit decision based on classification confidence, AHAR proposes
a novel adaptive architecture that uses an output block predictor to select a
portion of the baseline architecture to use during the inference phase.
Experimental results show that traditional early exit architectures suffer from
performance loss whereas our adaptive architecture provides similar or better
performance as the baseline one while being energy-efficient. We validate our
methodology in classifying locomotion activities from two datasets- Opportunity
and w-HAR. Compared to the fog/cloud computing approaches for the Opportunity
dataset, our baseline and adaptive architecture shows a comparable weighted F1
score of 91.79%, and 91.57%, respectively. For the w-HAR dataset, our baseline
and adaptive architecture outperforms the state-of-the-art works with a
weighted F1 score of 97.55%, and 97.64%, respectively. Evaluation on real
hardware shows that our baseline architecture is significantly energy-efficient
(422.38x less) and memory-efficient (14.29x less) compared to the works on the
Opportunity dataset. For the w-HAR dataset, our baseline architecture requires
2.04x less energy and 2.18x less memory compared to the state-of-the-art work.
Moreover, experimental results show that our adaptive architecture is 12.32%
(Opportunity) and 11.14% (w-HAR) energy-efficient than our baseline while
providing similar (Opportunity) or better (w-HAR) performance with no
significant memory overhead.

    

### [[2102.03826] Effective and Scalable Clustering on Massive Attributed Graphs](http://arxiv.org/abs/2102.03826)


  Given a graph G where each node is associated with a set of attributes, and a
parameter k specifying the number of output clusters, k-attributed graph
clustering (k-AGC) groups nodes in G into k disjoint clusters, such that nodes
within the same cluster share similar topological and attribute
characteristics, while those in different clusters are dissimilar. This problem
is challenging on massive graphs, e.g., with millions of nodes and billions of
edges. For such graphs, existing solutions either incur prohibitively high
costs, or produce clustering results with compromised quality.
In this paper, we propose ACMin, an effective approach to k-AGC that yields
high-quality clusters with cost linear to the size of the input graph G. The
main contributions of ACMin are twofold: (i) a novel formulation of the k-AGC
problem based on an attributed multi-hop conductance quality measure
custom-made for this problem setting, which effectively captures cluster
coherence in terms of both topological proximities and attribute similarities,
and (ii) a linear-time optimization solver that obtains high-quality clusters
iteratively, based on efficient matrix operations such as orthogonal
iterations, an alternative optimization approach, as well as an initialization
technique that significantly speeds up the convergence of ACMin in practice.
Extensive experiments, comparing 11 competitors on 6 real datasets,
demonstrate that ACMin consistently outperforms all competitors in terms of
result quality measured against ground-truth labels, while being up to orders
of magnitude faster. In particular, on the Microsoft Academic Knowledge Graph
dataset with 265.2 million edges and 1.1 billion attribute values, ACMin
outputs high-quality results for 5-AGC within 1.68 hours using a single CPU
core, while none of the 11 competitors finish within 3 days.

    

### [[2102.06602] Modeling Dynamic User Interests: A Neural Matrix Factorization Approach](http://arxiv.org/abs/2102.06602)


  In recent years, there has been significant interest in understanding users'
online content consumption patterns. But, the unstructured, high-dimensional,
and dynamic nature of such data makes extracting valuable insights challenging.
Here we propose a model that combines the simplicity of matrix factorization
with the flexibility of neural networks to efficiently extract nonlinear
patterns from massive text data collections relevant to consumers' online
consumption patterns. Our model decomposes a user's content consumption journey
into nonlinear user and content factors that are used to model their dynamic
interests. This natural decomposition allows us to summarize each user's
content consumption journey with a dynamic probabilistic weighting over a set
of underlying content attributes. The model is fast to estimate, easy to
interpret and can harness external data sources as an empirical prior. These
advantages make our method well suited to the challenges posed by modern
datasets. We use our model to understand the dynamic news consumption interests
of Boston Globe readers over five years. Thorough qualitative studies,
including a crowdsourced evaluation, highlight our model's ability to
accurately identify nuanced and coherent consumption patterns. These results
are supported by our model's superior and robust predictive performance over
several competitive baseline methods.

    

### [[2102.07227] Learning by Turning: Neural Architecture Aware Optimisation](http://arxiv.org/abs/2102.07227)


  Descent methods for deep networks are notoriously capricious: they require
careful tuning of step size, momentum and weight decay, and which method will
work best on a new benchmark is a priori unclear. To address this problem, this
paper conducts a combined study of neural architecture and optimisation,
leading to a new optimiser called Nero: the neuronal rotator. Nero trains
reliably without momentum or weight decay, works in situations where Adam and
SGD fail, and requires little to no learning rate tuning. Also, Nero's memory
footprint is ~ square root that of Adam or LAMB. Nero combines two ideas: (1)
projected gradient descent over the space of balanced networks; (2)
neuron-specific updates, where the step size sets the angle through which each
neuron's hyperplane turns. The paper concludes by discussing how this geometric
connection between architecture and optimisation may impact theories of
generalisation in deep learning.

    

### [[2102.08244] Differentially Private Quantiles](http://arxiv.org/abs/2102.08244)


  Quantiles are often used for summarizing and understanding data. If that data
is sensitive, it may be necessary to compute quantiles in a way that is
differentially private, providing theoretical guarantees that the result does
not reveal private information. However, when multiple quantiles are needed,
existing differentially private algorithms fare poorly: they either compute
quantiles individually, splitting the privacy budget, or summarize the entire
distribution, wasting effort. In either case the result is reduced accuracy. In
this work we propose an instance of the exponential mechanism that
simultaneously estimates exactly $m$ quantiles from $n$ data points while
guaranteeing differential privacy. The utility function is carefully structured
to allow for an efficient implementation that returns estimates of all $m$
quantiles in time $O(mn\log(n) + m^2n)$. Experiments show that our method
significantly outperforms the current state of the art on both real and
synthetic data while remaining efficient enough to be practical.

    

### [[2103.06966] Efficient Pairwise Neuroimage Analysis using the Soft Jaccard Index and 3D Keypoint Sets](http://arxiv.org/abs/2103.06966)


  We propose a novel pairwise distance measure between image keypoint sets, for
the purpose of large-scale medical image indexing. Our measure generalizes the
Jaccard index to account for soft set equivalence (SSE) between keypoint
elements, via an adaptive kernel framework modeling uncertainty in keypoint
appearance and geometry. A new kernel is proposed to quantify the variability
of keypoint geometry in location and scale. Our distance measure may be
estimated between $O(N^2)$ image pairs in $O(N~\log~N)$ operations via keypoint
indexing. Experiments report the first results for the task of predicting
family relationships from medical images, using 1010 T1-weighted MRI brain
volumes of 434 families including monozygotic and dizygotic twins, siblings and
half-siblings sharing 100%-25% of their polymorphic genes. Soft set equivalence
and the keypoint geometry kernel improve upon standard hard set equivalence
(HSE) and appearance kernels alone in predicting family relationships.
Monozygotic twin identification is near 100%, and three subjects with uncertain
genotyping are automatically paired with their self-reported families, the
first reported practical application of image-based family identification. Our
distance measure can also be used to predict group categories, sex is predicted
with an AUC=0.97. Software is provided for efficient fine-grained curation of
large, generic image datasets.

    

### [[2103.07960] Diagrammatic Differentiation for Quantum Machine Learning](http://arxiv.org/abs/2103.07960)


  We introduce diagrammatic differentiation for tensor calculus by generalising
the dual number construction from rigs to monoidal categories. Applying this to
ZX diagrams, we show how to calculate diagrammatically the gradient of a linear
map with respect to a phase parameter. For diagrams of parametrised quantum
circuits, we get the well-known parameter-shift rule at the basis of many
variational quantum algorithms. We then extend our method to the automatic
differentation of hybrid classical-quantum circuits, using diagrams with
bubbles to encode arbitrary non-linear operators. Moreover, diagrammatic
differentiation comes with an open-source implementation in DisCoPy, the Python
library for monoidal categories. Diagrammatic gradients of classical-quantum
circuits can then be simplified using the PyZX library and executed on quantum
hardware via the tket compiler. This opens the door to many practical
applications harnessing both the structure of string diagrams and the
computational power of quantum machine learning.

    

### [[2103.16149] Time-domain Speech Enhancement with Generative Adversarial Learning](http://arxiv.org/abs/2103.16149)


  Speech enhancement aims to obtain speech signals with high intelligibility
and quality from noisy speech. Recent work has demonstrated the excellent
performance of time-domain deep learning methods, such as Conv-TasNet. However,
these methods can be degraded by the arbitrary scales of the waveform induced
by the scale-invariant signal-to-noise ratio (SI-SNR) loss. This paper proposes
a new framework called Time-domain Speech Enhancement Generative Adversarial
Network (TSEGAN), which is an extension of the generative adversarial network
(GAN) in time-domain with metric evaluation to mitigate the scaling problem,
and provide model training stability, thus achieving performance improvement.
In addition, we provide a new method based on objective function mapping for
the theoretical analysis of the performance of Metric GAN, and explain why it
is better than the Wasserstein GAN. Experiments conducted demonstrate the
effectiveness of our proposed method, and illustrate the advantage of Metric
GAN.

    

### [[2104.00641] Dynamic Silos: Increased Modularity in Intra-organizational Communication Networks during the Covid-19 Pandemic](http://arxiv.org/abs/2104.00641)


  Workplace communications around the world were drastically altered by
Covid-19 and the resulting work-from-home orders and rise of remote work. We
analyze aggregated, anonymized metadata from over 360 billion emails within
over 4,000 organizations worldwide to examine changes in network community
structures over 24 months. We find that, in 2020, organizations around the
world became more siloed than in 2019, evidenced by increased modularity. This
shift was concurrent with decreased stability, indicating that organizational
siloes had less stable membership. We provide initial insights into the meaning
and implications of these network changes -- which we term dynamic silos -- for
new models of work.

    

### [[2104.01375] Evaluating explainable artificial intelligence methods for multi-label deep learning classification tasks in remote sensing](http://arxiv.org/abs/2104.01375)


  Although deep neural networks hold the state-of-the-art in several remote
sensing tasks, their black-box operation hinders the understanding of their
decisions, concealing any bias and other shortcomings in datasets and model
performance. To this end, we have applied explainable artificial intelligence
(XAI) methods in remote sensing multi-label classification tasks towards
producing human-interpretable explanations and improve transparency. In
particular, we utilized and trained deep learning models with state-of-the-art
performance in the benchmark BigEarthNet and SEN12MS datasets. Ten XAI methods
were employed towards understanding and interpreting models' predictions, along
with quantitative metrics to assess and compare their performance. Numerous
experiments were performed to assess the overall performance of XAI methods for
straightforward prediction cases, competing multiple labels, as well as
misclassification cases. According to our findings, Occlusion, Grad-CAM and
Lime were the most interpretable and reliable XAI methods. However, none
delivers high-resolution outputs, while apart from Grad-CAM, both Lime and
Occlusion are computationally expensive. We also highlight different aspects of
XAI performance and elaborate with insights on black-box decisions in order to
improve transparency, understand their behavior and reveal, as well, datasets'
particularities.

    

### [[2104.03017] Utilizing Self-supervised Representations for MOS Prediction](http://arxiv.org/abs/2104.03017)


  Speech quality assessment has been a critical issue in speech processing for
decades. Existing automatic evaluations usually require clean references or
parallel ground truth data, which is infeasible when the amount of data soars.
Subjective tests, on the other hand, do not need any additional clean or
parallel data and correlates better to human perception. However, such a test
is expensive and time-consuming because crowd work is necessary. It thus
becomes highly desired to develop an automatic evaluation approach that
correlates well with human perception while not requiring ground truth data. In
this paper, we use self-supervised pre-trained models for MOS prediction. We
show their representations can distinguish between clean and noisy audios.
Then, we fine-tune these pre-trained models followed by simple linear layers in
an end-to-end manner. The experiment results showed that our framework
outperforms the two previous state-of-the-art models by a significant
improvement on Voice Conversion Challenge 2018 and achieves comparable or
superior performance on Voice Conversion Challenge 2016. We also conducted an
ablation study to further investigate how each module benefits the task. The
experiment results are implemented and reproducible with publicly available
toolkits.

    

### [[2104.03838] Speech Denoising Without Clean Training Data: A Noise2Noise Approach](http://arxiv.org/abs/2104.03838)


  This paper tackles the problem of the heavy dependence of clean speech data
required by deep learning based audio-denoising methods by showing that it is
possible to train deep speech denoising networks using only noisy speech
samples. Conventional wisdom dictates that in order to achieve good speech
denoising performance, there is a requirement for a large quantity of both
noisy speech samples and perfectly clean speech samples, resulting in a need
for expensive audio recording equipment and extremely controlled soundproof
recording studios. These requirements pose significant challenges in data
collection, especially in economically disadvantaged regions and for low
resource languages. This work shows that speech denoising deep neural networks
can be successfully trained utilizing only noisy training audio. Furthermore it
is revealed that such training regimes achieve superior denoising performance
over conventional training regimes utilizing clean training audio targets, in
cases involving complex noise distributions and low Signal-to-Noise ratios
(high noise environments). This is demonstrated through experiments studying
the efficacy of our proposed approach over both real-world noises and synthetic
noises using the 20 layered Deep Complex U-Net architecture.

    

### [[2104.05608] Equivariant geometric learning for digital rock physics: estimating formation factor and effective permeability tensors from Morse graph](http://arxiv.org/abs/2104.05608)


  We present a SE(3)-equivariant graph neural network (GNN) approach that
directly predicting the formation factor and effective permeability from
micro-CT images. FFT solvers are established to compute both the formation
factor and effective permeability, while the topology and geometry of the pore
space are represented by a persistence-based Morse graph. Together, they
constitute the database for training, validating, and testing the neural
networks. While the graph and Euclidean convolutional approaches both employ
neural networks to generate low-dimensional latent space to represent the
features of the micro-structures for forward predictions, the SE(3) equivariant
neural network is found to generate more accurate predictions, especially when
the training data is limited. Numerical experiments have also shown that the
new SE(3) approach leads to predictions that fulfill the material frame
indifference whereas the predictions from classical convolutional neural
networks (CNN) may suffer from spurious dependence on the coordinate system of
the training data. Comparisons among predictions inferred from training the CNN
and those from graph convolutional neural networks (GNN) with and without the
equivariant constraint indicate that the equivariant graph neural network seems
to perform better than the CNN and GNN without enforcing equivariant
constraints.

    

### [[2104.07972] Language Models are Few-Shot Butlers](http://arxiv.org/abs/2104.07972)


  Pretrained language models demonstrate strong performance in most NLP tasks
when fine-tuned on small task-specific datasets. Hence, these autoregressive
models constitute ideal agents to operate in text-based environments where
language understanding and generative capabilities are essential. Nonetheless,
collecting expert demonstrations in such environments is a time-consuming
endeavour. We introduce a two-stage procedure to learn from a small set of
demonstrations and further improve by interacting with an environment. We show
that language models fine-tuned with only 1.2% of the expert demonstrations and
a simple reinforcement learning algorithm achieve a 51% absolute improvement in
success rate over existing methods in the ALFWorld environment.

    

### [[2104.13299] From Human Explanation to Model Interpretability: A Framework Based on Weight of Evidence](http://arxiv.org/abs/2104.13299)


  We take inspiration from the study of human explanation to inform the design
and evaluation of interpretability methods in machine learning. First, we
survey the literature on human explanation in philosophy, cognitive science,
and the social sciences, and propose a list of design principles for
machine-generated explanations that are meaningful to humans. Using the concept
of weight of evidence from information theory, we develop a method for
generating explanations that adhere to these principles. We show that this
method can be adapted to handle high-dimensional, multi-class settings,
yielding a flexible framework for generating explanations. We demonstrate that
these explanations can be estimated accurately from finite samples and are
robust to small perturbations of the inputs. We also evaluate our method
through a qualitative user study with machine learning practitioners, where we
observe that the resulting explanations are usable despite some participants
struggling with background concepts like prior class probabilities. Finally, we
conclude by surfacing~design~implications for interpretability tools in
general.

    

### [[2104.14289] Multi-class Text Classification using BERT-based Active Learning](http://arxiv.org/abs/2104.14289)


  Text Classification finds interesting applications in the pickup and delivery
services industry where customers require one or more items to be picked up
from a location and delivered to a certain destination. Classifying these
customer transactions into multiple categories helps understand the market
needs for different customer segments. Each transaction is accompanied by a
text description provided by the customer to describe the products being picked
up and delivered which can be used to classify the transaction. BERT-based
models have proven to perform well in Natural Language Understanding. However,
the product descriptions provided by the customers tend to be short, incoherent
and code-mixed (Hindi-English) text which demands fine-tuning of such models
with manually labelled data to achieve high accuracy. Collecting this labelled
data can prove to be expensive. In this paper, we explore Active Learning
strategies to label transaction descriptions cost effectively while using BERT
to train a transaction classification model. On TREC-6, AG's News Corpus and an
internal dataset, we benchmark the performance of BERT across different Active
Learning strategies in Multi-Class Text Classification.

    

### [[2105.05029] Adversarial examples attack based on random warm restart mechanism and improved Nesterov momentum](http://arxiv.org/abs/2105.05029)


  The deep learning algorithm has achieved great success in the field of
computer vision, but some studies have pointed out that the deep learning model
is vulnerable to attacks adversarial examples and makes false decisions. This
challenges the further development of deep learning, and urges researchers to
pay more attention to the relationship between adversarial examples attacks and
deep learning security. This work focuses on adversarial examples, optimizes
the generation of adversarial examples from the view of adversarial robustness,
takes the perturbations added in adversarial examples as the optimization
parameter. We propose RWR-NM-PGD attack algorithm based on random warm restart
mechanism and improved Nesterov momentum from the view of gradient
optimization. The algorithm introduces improved Nesterov momentum, using its
characteristics of accelerating convergence and improving gradient update
direction in optimization algorithm to accelerate the generation of adversarial
examples. In addition, the random warm restart mechanism is used for
optimization, and the projected gradient descent algorithm is used to limit the
range of the generated perturbations in each warm restart, which can obtain
better attack effect. Experiments on two public datasets show that the
algorithm proposed in this work can improve the success rate of attacking deep
learning models without extra time cost. Compared with the benchmark attack
method, the algorithm proposed in this work can achieve better attack success
rate for both normal training model and defense model. Our method has average
attack success rate of 46.3077%, which is 27.19% higher than I-FGSM and 9.27%
higher than PGD. The attack results in 13 defense models show that the attack
algorithm proposed in this work is superior to the benchmark algorithm in
attack universality and transferability.

    

### [[2105.06072] Leveraging Non-uniformity in First-order Non-convex Optimization](http://arxiv.org/abs/2105.06072)


  Classical global convergence results for first-order methods rely on uniform
smoothness and the Łojasiewicz inequality. Motivated by properties of
objective functions that arise in machine learning, we propose a non-uniform
refinement of these notions, leading to \emph{Non-uniform Smoothness} (NS) and
\emph{Non-uniform Łojasiewicz inequality} (NŁ). The new definitions
inspire new geometry-aware first-order methods that are able to converge to
global optimality faster than the classical $\Omega(1/t^2)$ lower bounds. To
illustrate the power of these geometry-aware methods and their corresponding
non-uniform analysis, we consider two important problems in machine learning:
policy gradient optimization in reinforcement learning (PG), and generalized
linear model training in supervised learning (GLM). For PG, we find that
normalizing the gradient ascent method can accelerate convergence to
$O(e^{-t})$ while incurring less overhead than existing algorithms. For GLM, we
show that geometry-aware normalized gradient descent can also achieve a linear
convergence rate, which significantly improves the best known results. We
additionally show that the proposed geometry-aware descent methods escape
landscape plateaus faster than standard gradient descent. Experimental results
are used to illustrate and complement the theoretical findings.

    

### [[2105.06844] Predicting speech intelligibility from EEG using a dilated convolutional network](http://arxiv.org/abs/2105.06844)


  Objective: Currently, only behavioral speech understanding tests are
available, which require active participation of the person being tested. As
this is infeasible for certain populations, an objective measure of speech
intelligibility is required. Recently, brain imaging data has been used to
establish a relationship between stimulus and brain response. Linear models
have been successfully linked to speech intelligibility but require per-subject
training. We present a deep-learning-based model incorporating dilated
convolutions that operates in a match/mismatch paradigm. The accuracy of the
model's match/mismatch predictions can be used as a proxy for speech
intelligibility without subject-specific (re)training. Approach: We evaluated
the performance of the model as a function of input segment length, EEG
frequency band and receptive field size while comparing it to multiple baseline
models. Next, we evaluated performance on held-out data and finetuning.
Finally, we established a link between the accuracy of our model and the
state-of-the-art behavioral MATRIX test. Main results: The dilated
convolutional model significantly outperformed the baseline models for every
input segment length, for all EEG frequency bands except the delta and theta
band, and receptive field sizes between 250 and 500 ms. Additionally,
finetuning significantly increased the accuracy on a held-out dataset. Finally,
a significant correlation (r=0.59, p=0.0154) was found between the speech
reception threshold estimated using the behavioral MATRIX test and our
objective method. Significance: Our method is the first to predict the speech
reception threshold from EEG for unseen subjects, contributing to objective
measures of speech intelligibility.

    

### [[2105.08601] Graph Neural Networks for Decentralized Multi-Robot Submodular Action Selection](http://arxiv.org/abs/2105.08601)


  In this paper, we develop a learning-based approach for decentralized
submodular maximization. We focus on applications where robots are required to
jointly select actions, e.g., motion primitives, to maximize team submodular
objectives with local communications only. Such applications are essential for
large-scale multi-robot coordination such as multi-robot motion planning for
area coverage, environment exploration, and target tracking. But the current
decentralized submodular maximization algorithms either require assumptions on
the inter-robot communication or lose some suboptimal guarantees. In this work,
we propose a general-purpose learning architecture towards submodular
maximization at scale, with decentralized communications. Particularly, our
learning architecture leverages a graph neural network (GNN) to capture local
interactions of the robots and learns decentralized decision-making for the
robots. We train the learning model by imitating an expert solution and
implement the resulting model for decentralized action selection involving
local observations and communications only. We demonstrate the performance of
our GNN-based learning approach in a scenario of active target coverage with
large networks of robots. The simulation results show our approach nearly
matches the coverage performance of the expert algorithm, and yet runs several
orders faster with up to 50 robots. Moreover, its coverage performance is
superior to the existing decentralized greedy algorithms. The results also
exhibit our approach's generalization capability in previously unseen
scenarios, e.g., larger environments and larger networks of robots.

    

### [[2105.12441] DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling](http://arxiv.org/abs/2105.12441)


  Since 2014 transfer learning has become the key driver for the improvement of
spatial saliency prediction; however, with stagnant progress in the last 3-5
years. We conduct a large-scale transfer learning study which tests different
ImageNet backbones, always using the same read out architecture and learning
protocol adopted from DeepGaze II. By replacing the VGG19 backbone of DeepGaze
II with ResNet50 features we improve the performance on saliency prediction
from 78% to 85%. However, as we continue to test better ImageNet models as
backbones (such as EfficientNetB5) we observe no additional improvement on
saliency prediction. By analyzing the backbones further, we find that
generalization to other datasets differs substantially, with models being
consistently overconfident in their fixation predictions. We show that by
combining multiple backbones in a principled manner a good confidence
calibration on unseen datasets can be achieved. This new model, "DeepGaze IIE",
yields a significant leap in benchmark performance in and out-of-domain with a
15 percent point improvement over DeepGaze II to 93% on MIT1003, marking a new
state of the art on the MIT/Tuebingen Saliency Benchmark in all available
metrics (AUC: 88.3%, sAUC: 79.4%, CC: 82.4%).

    

### [[2106.03142] A Physics-Informed Deep Learning Paradigm for Traffic State Estimation and Fundamental Diagram Discovery](http://arxiv.org/abs/2106.03142)


  Traffic state estimation (TSE) bifurcates into two main categories,
model-driven and data-driven (e.g., machine learning, ML) approaches, while
each suffers from either deficient physics or small data. To mitigate these
limitations, recent studies introduced hybrid methods, such as physics-informed
deep learning (PIDL), which contains both model-driven and data-driven
components. This paper contributes an improved paradigm, called
physics-informed deep learning with a fundamental diagram learner (PIDL+FDL),
which integrates ML terms into the model-driven component to learn a functional
form of a fundamental diagram (FD), i.e., a mapping from traffic density to
flow or velocity. The proposed PIDL+FDL has the advantages of performing the
TSE learning, model parameter discovery, and FD discovery simultaneously. This
paper focuses on highway TSE with observed data from loop detectors, using
traffic density or velocity as traffic variables. We demonstrate the use of
PIDL+FDL to solve popular first-order and second-order traffic flow models and
reconstruct the FD relation as well as model parameters that are outside the FD
term. We then evaluate the PIDL+FDL-based TSE using the Next Generation
SIMulation (NGSIM) dataset. The experimental results show the superiority of
the PIDL+FDL in terms of improved estimation accuracy and data efficiency over
advanced baseline TSE methods, and additionally, the capacity to properly learn
the unknown underlying FD relation.

    

### [[2106.03911] XIRL: Cross-embodiment Inverse Reinforcement Learning](http://arxiv.org/abs/2106.03911)


  We investigate the visual cross-embodiment imitation setting, in which agents
learn policies from videos of other agents (such as humans) demonstrating the
same task, but with stark differences in their embodiments -- shape, actions,
end-effector dynamics, etc. In this work, we demonstrate that it is possible to
automatically discover and learn vision-based reward functions from
cross-embodiment demonstration videos that are robust to these differences.
Specifically, we present a self-supervised method for Cross-embodiment Inverse
Reinforcement Learning (XIRL) that leverages temporal cycle-consistency
constraints to learn deep visual embeddings that capture task progression from
offline videos of demonstrations across multiple expert agents, each performing
the same task differently due to embodiment differences. Prior to our work,
producing rewards from self-supervised embeddings typically required alignment
with a reference trajectory, which may be difficult to acquire under stark
embodiment differences. We show empirically that if the embeddings are aware of
task progress, simply taking the negative distance between the current state
and goal state in the learned embedding space is useful as a reward for
training policies with reinforcement learning. We find our learned reward
function not only works for embodiments seen during training, but also
generalizes to entirely new embodiments. Additionally, when transferring
real-world human demonstrations to a simulated robot, we find that XIRL is more
sample efficient than current best methods. Qualitative results, code, and
datasets are available at this https URL


### [[2106.05648] Unsupervised Behaviour Discovery with Quality-Diversity Optimisation](http://arxiv.org/abs/2106.05648)


  Quality-Diversity algorithms refer to a class of evolutionary algorithms
designed to find a collection of diverse and high-performing solutions to a
given problem. In robotics, such algorithms can be used for generating a
collection of controllers covering most of the possible behaviours of a robot.
To do so, these algorithms associate a behavioural descriptor to each of these
behaviours. Each behavioural descriptor is used for estimating the novelty of
one behaviour compared to the others. In most existing algorithms, the
behavioural descriptor needs to be hand-coded, thus requiring prior knowledge
about the task to solve. In this paper, we introduce: Autonomous Robots
Realising their Abilities, an algorithm that uses a dimensionality reduction
technique to automatically learn behavioural descriptors based on raw sensory
data. The performance of this algorithm is assessed on three robotic tasks in
simulation. The experimental results show that it performs similarly to
traditional hand-coded approaches without the requirement to provide any
hand-coded behavioural descriptor. In the collection of diverse and
high-performing solutions, it also manages to find behaviours that are novel
with respect to more features than its hand-coded baselines. Finally, we
introduce a variant of the algorithm which is robust to the dimensionality of
the behavioural descriptor space.

    

### [[2106.05784] Programming Puzzles](http://arxiv.org/abs/2106.05784)


  We introduce a new type of programming challenge called programming puzzles,
as an objective and comprehensive evaluation of program synthesis, and release
an open-source dataset of Python Programming Puzzles (P3). Each puzzle is
defined by a short Python program $f$, and the goal is to find an input $x$
which makes $f$ output "True". The puzzles are objective in that each one is
specified entirely by the source code of its verifier $f$, so evaluating $f(x)$
is all that is needed to test a candidate solution $x$. They do not require an
answer key or input/output examples, nor do they depend on natural language
understanding. The dataset is comprehensive in that it spans problems of a
range of difficulties and domains, ranging from trivial string manipulation
problems that are immediately obvious to human programmers (but not necessarily
to AI), to classic programming puzzles (e.g., Towers of Hanoi), to
interview/competitive-programming problems (e.g., dynamic programming), to
longstanding open problems in algorithms and mathematics (e.g., factoring). The
objective nature of P3 readily supports self-supervised bootstrapping. We
develop baseline enumerative program synthesis and GPT-3 solvers that are
capable of solving easy puzzles -- even without access to any reference
solutions -- by learning from their own past solutions. Based on a small user
study, we find puzzle difficulty to correlate between human programmers and the
baseline AI solvers.

    

### [[2106.10562] Score-Based Explanations in Data Management and Machine Learning: An Answer-Set Programming Approach to Counterfactual Analysis](http://arxiv.org/abs/2106.10562)


  We describe some recent approaches to score-based explanations for query
answers in databases and outcomes from classification models in machine
learning. The focus is on work done by the author and collaborators. Special
emphasis is placed on declarative approaches based on answer-set programming to
the use of counterfactual reasoning for score specification and computation.
Several examples that illustrate the flexibility of these methods are shown.

    

### [[2106.13703] Task-Driven Out-of-Distribution Detection with Statistical Guarantees for Robot Learning](http://arxiv.org/abs/2106.13703)


  Our goal is to perform out-of-distribution (OOD) detection, i.e., to detect
when a robot is operating in environments that are drawn from a different
distribution than the environments used to train the robot. We leverage
Probably Approximately Correct (PAC)-Bayes theory in order to train a policy
with a guaranteed bound on performance on the training distribution. Our key
idea for OOD detection then relies on the following intuition: violation of the
performance bound on test environments provides evidence that the robot is
operating OOD. We formalize this via statistical techniques based on p-values
and concentration inequalities. The resulting approach (i) provides guaranteed
confidence bounds on OOD detection, and (ii) is task-driven and sensitive only
to changes that impact the robot's performance. We demonstrate our approach on
a simulated example of grasping objects with unfamiliar poses or shapes. We
also present both simulation and hardware experiments for a drone performing
vision-based obstacle avoidance in unfamiliar environments (including wind
disturbances and different obstacle densities). Our examples demonstrate that
we can perform task-driven OOD detection within just a handful of trials.
Comparisons with baselines also demonstrate the advantages of our approach in
terms of providing statistical guarantees and being insensitive to
task-irrelevant distribution shifts.

    

### [[2106.15178] Deep Inertial Navigation using Continuous Domain Adaptation and Optimal Transport](http://arxiv.org/abs/2106.15178)


  In this paper, we propose a new strategy for learning inertial robotic
navigation models. The proposed strategy enhances the generalisability of
end-to-end inertial modelling, and is aimed at wheeled robotic deployments.
Concretely, the paper describes the following. (1) Using precision robotics, we
empirically characterise the effect of changing the sensor position during
navigation on the distribution of raw inertial signals, as well as the
corresponding impact on learnt latent spaces. (2) We propose neural
architectures and algorithms to assimilate knowledge from an indexed set of
sensor positions in order to enhance the robustness and generalisability of
robotic inertial tracking in the field. Our scheme of choice uses continuous
domain adaptation (DA) and optimal transport (OT). (3) In our evaluation,
continuous OT DA outperforms a continuous adversarial DA baseline, while also
showing quantifiable learning benefits over simple data augmentation. We will
release our dataset to help foster future research.

    

### [[2106.16174] Hierarchical Phenotyping and Graph Modeling of Spatial Architecture in Lymphoid Neoplasms](http://arxiv.org/abs/2106.16174)


  The cells and their spatial patterns in the tumor microenvironment (TME) play
a key role in tumor evolution, and yet the latter remains an understudied topic
in computational pathology. This study, to the best of our knowledge, is among
the first to hybridize local and global graph methods to profile orchestration
and interaction of cellular components. To address the challenge in
hematolymphoid cancers, where the cell classes in TME may be unclear, we first
implemented cell-level unsupervised learning and identified two new cell
subtypes. Local cell graphs or supercells were built for each image by
considering the individual cell's geospatial location and classes. Then, we
applied supercell level clustering and identified two new cell communities. In
the end, we built global graphs to abstract spatial interaction patterns and
extract features for disease diagnosis. We evaluate the proposed algorithm on
H&E slides of 60 hematolymphoid neoplasms and further compared it with three
cell level graph-based algorithms, including the global cell graph, cluster
cell graph, and FLocK. The proposed algorithm achieved a mean diagnosis
accuracy of 0.703 with the repeated 5-fold cross-validation scheme. In
conclusion, our algorithm shows superior performance over the existing methods
and can be potentially applied to other cancer types.

    

### [[2107.00877] Conflict-free collective stochastic decision making by orbital angular momentum of photons through quantum interference](http://arxiv.org/abs/2107.00877)


  In recent cross-disciplinary studies involving both optics and computing,
single-photon-based decision-making has been demonstrated by utilizing the
wave-particle duality of light to solve multi-armed bandit problems.
Furthermore, entangled-photon-based decision-making has managed to solve a
competitive multi-armed bandit problem in such a way that conflicts of
decisions among players are avoided while ensuring equality. However, as these
studies are based on the polarization of light, the number of available choices
is limited to two, corresponding to two orthogonal polarization states. Here we
propose a scalable principle to solve competitive decision-making situations by
using the orbital angular momentum of photons based on its high dimensionality,
which theoretically allows an unlimited number of arms. Moreover, by extending
the Hong-Ou-Mandel effect to more than two states, we theoretically establish
an experimental configuration able to generate multi-photon states with orbital
angular momentum and conditions that provide conflict-free selections at every
turn. We numerically examine total rewards regarding three-armed bandit
problems, for which the proposed strategy accomplishes almost the theoretical
maximum, which is greater than a conventional mixed strategy intending to
realize Nash equilibrium. This is thanks to the quantum interference effect
that achieves no-conflict selections, even in the exploring phase to find the
best arms.

    

### [[2107.05466] Learning and Adaptation for Millimeter-Wave Beam Tracking and Training: a Dual Timescale Variational Framework](http://arxiv.org/abs/2107.05466)


  Millimeter-wave vehicular networks incur enormous beam-training overhead to
enable narrow-beam communications. This paper proposes a learning and
adaptation framework in which the dynamics of the communication beams are
learned and then exploited to design adaptive beam-tracking and training with
low overhead: on a long-timescale, a deep recurrent variational autoencoder
(DR-VAE) uses noisy beam-training feedback to learn a probabilistic model of
beam dynamics and enable predictive beam-tracking; on a short-timescale, an
adaptive beam-training procedure is formulated as a partially observable (PO-)
Markov decision process (MDP) and optimized via point-based value iteration
(PBVI) by leveraging beam-training feedback and a probabilistic prediction of
the strongest beam pair provided by the DR-VAE. In turn, beam-training feedback
is used to refine the DR-VAE via stochastic gradient ascent in a continuous
process of learning and adaptation. The proposed DR-VAE learning framework
learns accurate beam dynamics: it reduces the Kullback-Leibler divergence
between the ground truth and the learned model of beam dynamics by 95% over the
Baum-Welch algorithm and a naive learning approach that neglects feedback
errors. Numerical results on a line-of-sight (LOS) scenario with multipath
reveal that the proposed dual timescale approach yields near-optimal spectral
efficiency, and improves it by 130% over a policy that scans exhaustively over
the dominant beam pairs, and by 20% over a state-of-the-art POMDP policy.
Finally, a low-complexity policy is proposed by reducing the POMDP to an
error-robust MDP, and is shown to perform well in regimes with infrequent
feedback errors.

    

### [[2103.02071] Sibyl: Understanding and Addressing the Usability Challenges of Machine Learning In High-Stakes Decision Making](http://arxiv.org/abs/2103.02071)


  Machine learning (ML) is being applied to a diverse and ever-growing set of
domains. In many cases, domain experts - who often have no expertise in ML or
data science - are asked to use ML predictions to make high-stakes decisions.
Multiple ML usability challenges can appear as result, such as lack of user
trust in the model, inability to reconcile human-ML disagreement, and ethical
concerns about oversimplification of complex problems to a single algorithm
output. In this paper, we investigate the ML usability challenges that present
in the domain of child welfare screening through a series of collaborations
with child welfare screeners. Following the iterative design process between
the ML scientists, visualization researchers, and domain experts (child
screeners), we first identified four key ML challenges and honed in on one
promising explainable ML technique to address them (local factor
contributions). Then we implemented and evaluated our visual analytics tool,
Sibyl, to increase the interpretability and interactivity of local factor
contributions. The effectiveness of our tool is demonstrated by two formal user
studies with 12 non-expert participants and 13 expert participants
respectively. Valuable feedback was collected, from which we composed a list of
design implications as a useful guideline for researchers who aim to develop an
interpretable and interactive visualization tool for ML prediction models
deployed for child welfare screeners and other similar domain experts.

    

### [[2103.09002] Hebbian Semi-Supervised Learning in a Sample Efficiency Setting](http://arxiv.org/abs/2103.09002)


  We propose to address the issue of sample efficiency, in Deep Convolutional
Neural Networks (DCNN), with a semi-supervised training strategy that combines
Hebbian learning with gradient descent: all internal layers (both convolutional
and fully connected) are pre-trained using an unsupervised approach based on
Hebbian learning, and the last fully connected layer (the classification layer)
is trained using Stochastic Gradient Descent (SGD). In fact, as Hebbian
learning is an unsupervised learning method, its potential lies in the
possibility of training the internal layers of a DCNN without labels. Only the
final fully connected layer has to be trained with labeled examples.
We performed experiments on various object recognition datasets, in different
regimes of sample efficiency, comparing our semi-supervised (Hebbian for
internal layers + SGD for the final fully connected layer) approach with
end-to-end supervised backprop training, and with semi-supervised learning
based on Variational Auto-Encoder (VAE). The results show that, in regimes
where the number of available labeled samples is low, our semi-supervised
approach outperforms the other approaches in almost all the cases.

    

### [[2109.08874] Reconfigurable Low-latency Memory System for Sparse Matricized Tensor Times Khatri-Rao Product on FPGA](http://arxiv.org/abs/2109.08874)


  Tensor decomposition has become an essential tool in many applications in
various domains, including machine learning. Sparse Matricized Tensor Times
Khatri-Rao Product (MTTKRP) is one of the most computationally expensive
kernels in tensor computations. Despite having significant computational
parallelism, MTTKRP is a challenging kernel to optimize due to its irregular
memory access characteristics. This paper focuses on a multi-faceted memory
system, which explores the spatial and temporal locality of the data structures
of MTTKRP. Further, users can reconfigure our design depending on the behavior
of the compute units used in the FPGA accelerator. Our system efficiently
accesses all the MTTKRP data structures while reducing the total memory access
time, using a distributed cache and Direct Memory Access (DMA) subsystem.
Moreover, our work improves the memory access time by 3.5x compared with
commercial memory controller IPs. Also, our system shows 2x and 1.26x speedups
compared with cache-only and DMA-only memory systems, respectively.

    

### [[2108.13378] MultPIM: Fast Stateful Multiplication for Processing-in-Memory](http://arxiv.org/abs/2108.13378)


  Processing-in-memory (PIM) seeks to eliminate computation/memory data
transfer using devices that support both storage and logic. Stateful logic
techniques such as IMPLY, MAGIC and FELIX can perform logic gates within
memristive crossbar arrays with massive parallelism. Multiplication via
stateful logic is an active field of research due to the wide implications.
Recently, RIME has become the state-of-the-art algorithm for stateful
single-row multiplication by using memristive partitions, reducing the latency
of the previous state-of-the-art by 5.1x. In this paper, we begin by proposing
novel partition-based computation techniques for broadcasting and shifting
data. Then, we design an in-memory multiplication algorithm based on the
carry-save add-shift (CSAS) technique. Finally, we develop a novel stateful
full-adder that significantly improves the state-of-the-art (FELIX) design.
These contributions constitute MultPIM, a multiplier that reduces
state-of-the-art time complexity from quadratic to linear-log. For 32-bit
numbers, MultPIM improves latency by an additional 4.2x over RIME, while even
slightly reducing area overhead. Furthermore, we optimize MultPIM for
full-precision matrix-vector multiplication and improve latency by 25.5x over
FloatPIM matrix-vector multiplication.

    

### [[2109.08751] Sparbit: a new logarithmic-cost and data locality-aware MPI Allgather algorithm](http://arxiv.org/abs/2109.08751)


  The collective operations are considered critical for improving the
performance of exascale-ready and high-performance computing applications. On
this paper we focus on the Message-Passing Interface (MPI) Allgather many to
many collective, which is amongst the most called and time-consuming
operations. Each MPI algorithm for this call suffers from different operational
and performance limitations, that might include only working for restricted
cases, requiring linear amounts of communication steps with the growth in
number of processes, memory copies and shifts to assure correct data
organization, and non-local data exchange patterns, most of which negatively
contribute to the total operation time. All these characteristics create an
environment where there is no algorithm which is the best for all cases and
this consequently implies that careful choices of alternatives must be made to
execute the call. Considering such aspects, we propose the Stripe Parallel
Binomial Trees (Sparbit) algorithm, which has optimal latency and bandwidth
time costs with no usage restrictions. It also maintains a much more local
communication pattern that minimizes the delays due to long range exchanges,
allowing the extraction of more performance from current systems when compared
with asymptotically equivalent alternatives. On its best scenario, Sparbit
surpassed the traditional MPI algorithms on 46.43% of test cases with mean
(median) improvements of 34.7% (26.16%) and highest reaching 84.16%.

    

### [[2109.08930] Regular Sequential Serializability and Regular Sequential Consistency](http://arxiv.org/abs/2109.08930)


  Strictly serializable (linearizable) services appear to execute transactions
(operations) sequentially, in an order consistent with real time. This
restricts a transaction's (operation's) possible return values and in turn,
simplifies application programming. In exchange, strictly serializable
(linearizable) services perform worse than those with weaker consistency.
Switching to such services, however, can break applications.
This work introduces two new consistency models to ease this trade-off:
regular sequential serializability (RSS) and regular sequential consistency
(RSC). They are just as "strong" for applications; we prove any application
invariant that holds when using a strictly serializable (linearizable) service
also holds when using an RSS (RSC) service. Yet they are "weaker" for services;
they allow new, better-performing designs. To demonstrate this, we design,
implement, and evaluate variants of two systems, Spanner and Gryff, weakening
their consistency to RSS and RSC, respectively. The new variants achieve better
read-only transaction and read tail latency than their counterparts.

    

### [[2109.08933] Optimization-based Block Coordinate Gradient Coding](http://arxiv.org/abs/2109.08933)


  Existing gradient coding schemes introduce identical redundancy across the
coordinates of gradients and hence cannot fully utilize the computation results
from partial stragglers. This motivates the introduction of diverse
redundancies across the coordinates of gradients. This paper considers a
distributed computation system consisting of one master and $N$ workers
characterized by a general partial straggler model and focuses on solving a
general large-scale machine learning problem with $L$ model parameters. We show
that it is sufficient to provide at most $N$ levels of redundancies for
tolerating $0, 1,\cdots, N-1$ stragglers, respectively. Consequently, we
propose an optimal block coordinate gradient coding scheme based on a
stochastic optimization problem that optimizes the partition of the $L$
coordinates into $N$ blocks, each with identical redundancy, to minimize the
expected overall runtime for collaboratively computing the gradient. We obtain
an optimal solution using a stochastic projected subgradient method and propose
two low-complexity approximate solutions with closed-from expressions, for the
stochastic optimization problem. We also show that under a shifted-exponential
distribution, for any $L$, the expected overall runtimes of the two approximate
solutions and the minimum overall runtime have sub-linear multiplicative gaps
in $N$. To the best of our knowledge, this is the first work that optimizes the
redundancies of gradient coding introduced across the coordinates of gradients.

    

### [[2109.09056] Enabling particle applications for exascale computing platforms](http://arxiv.org/abs/2109.09056)


  The Exascale Computing Project (ECP) is invested in co-design to assure that
key applications are ready for exascale computing. Within ECP, the Co-design
Center for Particle Applications (CoPA) is addressing challenges faced by
particle-based applications across four sub-motifs: short-range
particle-particle interactions (e.g., those which often dominate molecular
dynamics (MD) and smoothed particle hydrodynamics (SPH) methods), long-range
particle-particle interactions (e.g., electrostatic MD and gravitational
N-body), particle-in-cell (PIC) methods, and linear-scaling electronic
structure and quantum molecular dynamics (QMD) algorithms. Our crosscutting
co-designed technologies fall into two categories: proxy applications (or apps)
and libraries. Proxy apps are vehicles used to evaluate the viability of
incorporating various types of algorithms, data structures, and
architecture-specific optimizations and the associated trade-offs; examples
include ExaMiniMD, CabanaMD, CabanaPIC, and ExaSP2. Libraries are modular
instantiations that multiple applications can utilize or be built upon; CoPA
has developed the Cabana particle library, PROGRESS/BML libraries for QMD, and
the SWFFT and fftMPI parallel FFT libraries. Success is measured by
identifiable lessons learned that are translated either directly into parent
production application codes or into libraries, with demonstrated performance
and/or productivity improvement. The libraries and their use in CoPA's ECP
application partner codes are also addressed.

    

### [[2109.09142] Decentralized Wireless Federated Learning with Differential Privacy](http://arxiv.org/abs/2109.09142)


  This paper studies decentralized federated learning algorithms in wireless
IoT networks. The traditional parameter server architecture for federated
learning faces some problems such as low fault tolerance, large communication
overhead and inaccessibility of private data. To solve these problems, we
propose a Decentralized-Wireless-Federated-Learning algorithm called DWFL. The
algorithm works in a system where the workers are organized in a peer-to-peer
and server-less manner, and the workers exchange their privacy preserving data
with the anolog transmission scheme over wireless channels in parallel. With
rigorous analysis, we show that DWFL satisfies $(\epsilon,\delta)$-differential
privacy and the privacy budget per worker scale as
$\mathcal{O}(\frac{1}{\sqrt{N}})$, in contrast with the constant budget in the
orthogonal transmission approach. Furthermore, DWFL converges at the same rate
of $\sqrt{\frac{1}{TN}}$ as the best known centralized algorithm with a central
parameter server. Extensive experiments demonstrate that our algorithm DWFL
also performs well in real settings.

    

### [[2101.03733] A Fault Tolerant Mechanism for Partitioning and Offloading Framework in Pervasive Environments](http://arxiv.org/abs/2101.03733)


  Application partitioning and code offloading are being researched extensively
during the past few years. Several frameworks for code offloading have been
proposed. However, fewer works attempted to address issues occurred with its
implementation in pervasive environments such as frequent network disconnection
due to high mobility of users. Thus, in this paper, we proposed a fault
tolerant algorithm that helps in consolidating the efficiency and robustness of
application partitioning and offloading frameworks. To permit the usage of
different fault tolerant policies such as replication and checkpointing, the
devices are grouped into high and low reliability clusters. Experimental
results shown that the fault tolerant algorithm can easily adapt to different
execution conditions while incurring minimum overhead.

    

### [[2103.09683] Accelerating Radiation Therapy Dose Calculation with Nvidia GPUs](http://arxiv.org/abs/2103.09683)


  Radiation Treatment Planning (RTP) is the process of planning the appropriate
external beam radiotherapy to combat cancer in human patients. RTP is a complex
and compute-intensive task, which often takes a long time (several hours) to
compute. Reducing this time allows for higher productivity at clinics and more
sophisticated treatment planning, which can materialize in better treatments.
The state-of-the-art in medical facilities uses general-purpose processors
(CPUs) to perform many steps in the RTP process. In this paper, we explore the
use of accelerators to reduce RTP calculating time. We focus on the step that
calculates the dose using the Graphics Processing Unit (GPU), which we believe
is an excellent candidate for this computation type. Next, we create a highly
optimized implementation for a custom Sparse Matrix-Vector Multiplication
(SpMV) that operates on numerical formats unavailable in state-of-the-art SpMV
libraries (e.g., Ginkgo and cuSPARSE). We show that our implementation is
several times faster than the baseline (up-to 4x) and has a higher operational
intensity than similar (but different) versions such as Ginkgo and cuSPARSE.

    

### [[2109.08755] Solving infinite-horizon Dec-POMDPs using Finite State Controllers within JESP](http://arxiv.org/abs/2109.08755)


  This paper looks at solving collaborative planning problems formalized as
Decentralized POMDPs (Dec-POMDPs) by searching for Nash equilibria, i.e.,
situations where each agent's policy is a best response to the other agents'
(fixed) policies. While the Joint Equilibrium-based Search for Policies (JESP)
algorithm does this in the finite-horizon setting relying on policy trees, we
propose here to adapt it to infinite-horizon Dec-POMDPs by using finite state
controller (FSC) policy representations. In this article, we (1) explain how to
turn a Dec-POMDP with $N-1$ fixed FSCs into an infinite-horizon POMDP whose
solution is an $N^\text{th}$ agent best response; (2) propose a JESP variant,
called \infJESP, using this to solve infinite-horizon Dec-POMDPs; (3) introduce
heuristic initializations for JESP aiming at leading to good solutions; and (4)
conduct experiments on state-of-the-art benchmark problems to evaluate our
approach.

    

### [[2109.08794] A Comprehensive Overview of Recommender System and Sentiment Analysis](http://arxiv.org/abs/2109.08794)


  Recommender system has been proven to be significantly crucial in many fields
and is widely used by various domains. Most of the conventional recommender
systems rely on the numeric rating given by a user to reflect his opinion about
a consumed item; however, these ratings are not available in many domains. As a
result, a new source of information represented by the user-generated reviews
is incorporated in the recommendation process to compensate for the lack of
these ratings. The reviews contain prosperous and numerous information related
to the whole item or a specific feature that can be extracted using the
sentiment analysis field. This paper gives a comprehensive overview to help
researchers who aim to work with recommender system and sentiment analysis. It
includes a background of the recommender system concept, including phases,
approaches, and performance metrics used in recommender systems. Then, it
discusses the sentiment analysis concept and highlights the main points in the
sentiment analysis, including level, approaches, and focuses on aspect-based
sentiment analysis.

    

### [[2109.08814] Structured Pattern Pruning Using Regularization](http://arxiv.org/abs/2109.08814)


  Iterative Magnitude Pruning (IMP) is a network pruning method that repeats
the process of removing weights with the least magnitudes and retraining the
model. When visualizing the weight matrices of language models pruned by IMP,
previous research has shown that a structured pattern emerges, wherein the
resulting surviving weights tend to prominently cluster in a select few rows
and columns of the matrix. Though the need for further research in utilizing
these structured patterns for potential performance gains has previously been
indicated, it has yet to be thoroughly studied. We propose SPUR (Structured
Pattern pruning Using Regularization), a novel pruning mechanism that
preemptively induces structured patterns in compression by adding a
regularization term to the objective function in the IMP. Our results show that
SPUR can significantly preserve model performance under high sparsity settings
regardless of the language or the task. Our contributions are as follows: (i)
We propose SPUR, a network pruning mechanism that improves upon IMP regardless
of the language or the task. (ii) We are the first to empirically verify the
efficacy of "structured patterns" observed previously in pruning research.
(iii) SPUR is a resource-efficient mechanism in that it does not require
significant additional computations.

    

### [[2109.08834] Generating Active Explicable Plans in Human-Robot Teaming](http://arxiv.org/abs/2109.08834)


  Intelligent robots are redefining a multitude of critical domains but are
still far from being fully capable of assisting human peers in day-to-day
tasks. An important requirement of collaboration is for each teammate to
maintain and respect an understanding of the others' expectations of itself.
Lack of which may lead to serious issues such as loose coordination between
teammates, reduced situation awareness, and ultimately teaming failures. Hence,
it is important for robots to behave explicably by meeting the human's
expectations. One of the challenges here is that the expectations of the human
are often hidden and can change dynamically as the human interacts with the
robot. However, existing approaches to generating explicable plans often assume
that the human's expectations are known and static. In this paper, we propose
the idea of active explicable planning to relax this assumption. We apply a
Bayesian approach to model and predict dynamic human belief and expectations to
make explicable planning more anticipatory. We hypothesize that active
explicable plans can be more efficient and explicable at the same time, when
compared to explicable plans generated by the existing methods. In our
experimental evaluation, we verify that our approach generates more efficient
explicable plans while successfully capturing the dynamic belief change of the
human teammate.

    

### [[2109.08856] Favoring Eagerness for Remaining Items: Achieving Efficient and Fair Assignments](http://arxiv.org/abs/2109.08856)


  In the assignment problem, items must be assigned to agents who have unit
demands, based on agents' ordinal preferences. Often the goal is to design a
mechanism that is both fair and efficient. In this paper, we first prove that,
unfortunately, the desirable efficiency notions rank-maximality, ex-post
favoring-higher-ranks, and ex-ante favoring-higher-ranks, which aim to allocate
each item to agents who rank it highest over all the items, are incompatible
with the desirable fairness notions strong equal treatment of equals (SETE) and
sd-weak-envy-freeness (sd-WEF) simultaneously. In light of this, we propose
novel properties of efficiency based on a subtly different notion to favoring
higher ranks, by favoring "eagerness" for remaining items and aiming to
guarantee that each item is allocated to agents who rank it highest among
remaining items. Specifically, we propose ex-post
favoring-eagerness-for-remaining-items (ep-FERI) and ex-ante
favoring-eagerness-for-remaining-items (ea-FERI). We prove that the eager
Boston mechanism satisfies ep-FERI and sd-WSP and that the uniform
probabilistic respecting eagerness mechanism satisfies ea-FERI. We also prove
that both mechanisms satisfy SETE and sd-WEF, and show that no mechanism can
satisfy stronger versions of envy-freeness and strategyproofness while
simultaneously maintaining SETE, and either ep-FERI or ea-FERI.

    

### [[2109.08868] Clean-label Backdoor Attack against Deep Hashing based Retrieval](http://arxiv.org/abs/2109.08868)


  Deep hashing has become a popular method in large-scale image retrieval due
to its computational and storage efficiency. However, recent works raise the
security concerns of deep hashing. Although existing works focus on the
vulnerability of deep hashing in terms of adversarial perturbations, we
identify a more pressing threat, backdoor attack, when the attacker has access
to the training data. A backdoored deep hashing model behaves normally on
original query images, while returning the images with the target label when
the trigger presents, which makes the attack hard to be detected. In this
paper, we uncover this security concern by utilizing clean-label data
poisoning. To the best of our knowledge, this is the first attempt at the
backdoor attack against deep hashing models. To craft the poisoned images, we
first generate the targeted adversarial patch as the backdoor trigger.
Furthermore, we propose the confusing perturbations to disturb the hashing code
learning, such that the hashing model can learn more about the trigger. The
confusing perturbations are imperceptible and generated by dispersing the
images with the target label in the Hamming space. We have conducted extensive
experiments to verify the efficacy of our backdoor attack under various
settings. For instance, it can achieve 63% targeted mean average precision on
ImageNet under 48 bits code length with only 40 poisoned images.

    

### [[2109.08877] DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation](http://arxiv.org/abs/2109.08877)


  In this paper, we provide a bilingual parallel human-to-human recommendation
dialog dataset (DuRecDial 2.0) to enable researchers to explore a challenging
task of multilingual and cross-lingual conversational recommendation. The
difference between DuRecDial 2.0 and existing conversational recommendation
datasets is that the data item (Profile, Goal, Knowledge, Context, Response) in
DuRecDial 2.0 is annotated in two languages, both English and Chinese, while
other datasets are built with the setting of a single language. We collect 8.2k
dialogs aligned across English and Chinese languages (16.5k dialogs and 255k
utterances in total) that are annotated by crowdsourced workers with strict
quality control procedure. We then build monolingual, multilingual, and
cross-lingual conversational recommendation baselines on DuRecDial 2.0.
Experiment results show that the use of additional English data can bring
performance improvement for Chinese conversational recommendation, indicating
the benefits of DuRecDial 2.0. Finally, this dataset provides a challenging
testbed for future studies of monolingual, multilingual, and cross-lingual
conversational recommendation.

    

### [[2109.08880] Computational Imaging and Artificial Intelligence: The Next Revolution of Mobile Vision](http://arxiv.org/abs/2109.08880)


  Signal capture stands in the forefront to perceive and understand the
environment and thus imaging plays the pivotal role in mobile vision. Recent
explosive progresses in Artificial Intelligence (AI) have shown great potential
to develop advanced mobile platforms with new imaging devices. Traditional
imaging systems based on the "capturing images first and processing afterwards"
mechanism cannot meet this unprecedented demand. Differently, Computational
Imaging (CI) systems are designed to capture high-dimensional data in an
encoded manner to provide more information for mobile vision systems.Thanks to
AI, CI can now be used in real systems by integrating deep learning algorithms
into the mobile vision platform to achieve the closed loop of intelligent
acquisition, processing and decision making, thus leading to the next
revolution of mobile vision.Starting from the history of mobile vision using
digital cameras, this work first introduces the advances of CI in diverse
applications and then conducts a comprehensive review of current research
topics combining CI and AI. Motivated by the fact that most existing studies
only loosely connect CI and AI (usually using AI to improve the performance of
CI and only limited works have deeply connected them), in this work, we propose
a framework to deeply integrate CI and AI by using the example of self-driving
vehicles with high-speed communication, edge computing and traffic planning.
Finally, we outlook the future of CI plus AI by investigating new materials,
brain science and new computing techniques to shed light on new directions of
mobile vision systems.

    

### [[2109.08884] Design and Results of ICCMA 2021](http://arxiv.org/abs/2109.08884)


  Since 2015, the International Competition on Computational Models of
Argumentation (ICCMA) provides a systematic comparison of the different
algorithms for solving some classical reasoning problems in the domain of
abstract argumentation. This paper discusses the design of the Fourth
International Competition on Computational Models of Argumentation. We describe
the rules of the competition and the benchmark selection method that we used.
After a brief presentation of the competitors, we give an overview of the
results.

    

### [[2109.08910] MS-SincResNet: Joint learning of 1D and 2D kernels using multi-scale SincNet and ResNet for music genre classification](http://arxiv.org/abs/2109.08910)


  In this study, we proposed a new end-to-end convolutional neural network,
called MS-SincResNet, for music genre classification. MS-SincResNet appends 1D
multi-scale SincNet (MS-SincNet) to 2D ResNet as the first convolutional layer
in an attempt to jointly learn 1D kernels and 2D kernels during the training
stage. First, an input music signal is divided into a number of fixed-duration
(3 seconds in this study) music clips, and the raw waveform of each music clip
is fed into 1D MS-SincNet filter learning module to obtain three-channel 2D
representations. The learned representations carry rich timbral, harmonic, and
percussive characteristics comparing with spectrograms, harmonic spectrograms,
percussive spectrograms and Mel-spectrograms. ResNet is then used to extract
discriminative embeddings from these 2D representations. The spatial pyramid
pooling (SPP) module is further used to enhance the feature discriminability,
in terms of both time and frequency aspects, to obtain the classification label
of each music clip. Finally, the voting strategy is applied to summarize the
classification results from all 3-second music clips. In our experimental
results, we demonstrate that the proposed MS-SincResNet outperforms the
baseline SincNet and many well-known hand-crafted features. Considering
individual 2D representation, MS-SincResNet also yields competitive results
with the state-of-the-art methods on the GTZAN dataset and the ISMIR2004
dataset. The code is available at this https URL


### [[2109.08927] Weakly Supervised Explainable Phrasal Reasoning with Neural Fuzzy Logic](http://arxiv.org/abs/2109.08927)


  Natural language inference (NLI) aims to determine the logical relationship
between two sentences among the target labels Entailment, Contradiction, and
Neutral. In recent years, deep learning models have become a prevailing
approach to NLI, but they lack interpretability and explainability. In this
work, we address the explainability for NLI by weakly supervised logical
reasoning, and propose an Explainable Phrasal Reasoning (EPR) approach. Our
model first detects phrases as the semantic unit and aligns corresponding
phrases. Then, the model predicts the NLI label for the aligned phrases, and
induces the sentence label by fuzzy logic formulas. Our EPR is almost
everywhere differentiable and thus the system can be trained end-to-end in a
weakly supervised manner. We annotated a corpus and developed a set of metrics
to evaluate phrasal reasoning. Results show that our EPR yields much more
meaningful explanations in terms of F scores than previous studies. To the best
of our knowledge, we are the first to develop a weakly supervised phrasal
reasoning model for the NLI task.

    

### [[2109.08947] Risk-averse autonomous systems: A brief history and recent developments from the perspective of optimal control](http://arxiv.org/abs/2109.08947)


  We offer a historical overview of methodologies for quantifying the notion of
risk and optimizing risk-aware autonomous systems, with emphasis on risk-averse
settings in which safety may be critical. We categorize and present
state-of-the-art approaches, and we describe connections between such
approaches and ideas from the fields of decision theory, operations research,
reinforcement learning, and stochastic control. The first part of the review
focuses on model-based risk-averse methods. The second part discusses methods
that blend model-based and model-free techniques for the purpose of designing
policies with improved adaptive capabilities. We conclude by highlighting areas
for future research.

    

### [[2109.08965] PCNN: A physics-constrained neural network for multiphase flows](http://arxiv.org/abs/2109.08965)


  The present study develops a physics-constrained neural network (PCNN) to
predict sequential patterns and motions of multiphase flows (MPFs), which
includes strong interactions among various fluid phases. To predict the order
parameters, which locate individual phases, in the future time, the conditional
neural processes and long short-term memory (CNP-LSTM) are applied to quickly
infer the dynamics of the phases after encoding only a few observations. After
that, the multiphase consistent and conservative boundedness mapping algorithm
(MCBOM) is implemented to correct the order parameters predicted from CNP-LSTM
in order to strictly satisfy the mass conservation, the summation of the volume
fractions of the phases to be unity, the consistency of reduction, and the
boundedness of the order parameters. Then, the density of the fluid mixture is
updated from the corrected order parameters. Finally, the velocity in the
future time is predicted by a physics-informed CNP-LSTM (PICNP-LSTM) where
conservation of momentum is included in the loss function with the observed
density and velocity as the inputs. The proposed PCNN for MPFs sequentially
performs (CNP-LSTM)-(MCBOM)-(PICNP-LSTM), which avoids unphysical behaviors of
the order parameters, accelerates the convergence, and requires fewer data to
make predictions. Numerical experiments demonstrate that the proposed PCNN is
capable of predicting MPFs effectively.

    

### [[2109.08973] Hierarchical Policy for Non-prehensile Multi-object Rearrangement with Deep Reinforcement Learning and Monte Carlo Tree Search](http://arxiv.org/abs/2109.08973)


  Non-prehensile multi-object rearrangement is a robotic task of planning
feasible paths and transferring multiple objects to their predefined target
poses without grasping. It needs to consider how each object reaches the target
and the order of object movement, which significantly deepens the complexity of
the problem. To address these challenges, we propose a hierarchical policy to
divide and conquer for non-prehensile multi-object rearrangement. In the
high-level policy, guided by a designed policy network, the Monte Carlo Tree
Search efficiently searches for the optimal rearrangement sequence among
multiple objects, which benefits from imitation and reinforcement. In the
low-level policy, the robot plans the paths according to the order of path
primitives and manipulates the objects to approach the goal poses one by one.
We verify through experiments that the proposed method can achieve a higher
success rate, fewer steps, and shorter path length compared with the
state-of-the-art.

    

### [[2109.09016] The Unreasonable Effectiveness of the Final Batch Normalization Layer](http://arxiv.org/abs/2109.09016)


  Early-stage disease indications are rarely recorded in real-world domains,
such as Agriculture and Healthcare, and yet, their accurate identification is
critical in that point of time. In this type of highly imbalanced
classification problems, which encompass complex features, deep learning (DL)
is much needed because of its strong detection capabilities. At the same time,
DL is observed in practice to favor majority over minority classes and
consequently suffer from inaccurate detection of the targeted early-stage
indications. In this work, we extend the study done by Kocaman et al., 2020,
showing that the final BN layer, when placed before the softmax output layer,
has a considerable impact in highly imbalanced image classification problems as
well as undermines the role of the softmax outputs as an uncertainty measure.
This current study addresses additional hypotheses and reports on the following
findings: (i) the performance gain after adding the final BN layer in highly
imbalanced settings could still be achieved after removing this additional BN
layer in inference; (ii) there is a certain threshold for the imbalance ratio
upon which the progress gained by the final BN layer reaches its peak; (iii)
the batch size also plays a role and affects the outcome of the final BN
application; (iv) the impact of the BN application is also reproducible on
other datasets and when utilizing much simpler neural architectures; (v) the
reported BN effect occurs only per a single majority class and multiple
minority classes i.e., no improvements are evident when there are two majority
classes; and finally, (vi) utilizing this BN layer with sigmoid activation has
almost no impact when dealing with a strongly imbalanced image classification
tasks.

    

### [[2109.09083] Towards robustness under occlusion for face recognition](http://arxiv.org/abs/2109.09083)


  In this paper, we evaluate the effects of occlusions in the performance of a
face recognition pipeline that uses a ResNet backbone. The classifier was
trained on a subset of the CelebA-HQ dataset containing 5,478 images from 307
classes, to achieve top-1 error rate of 17.91%. We designed 8 different
occlusion masks which were applied to the input images. This caused a
significant drop in the classifier performance: its error rate for each mask
became at least two times worse than before. In order to increase robustness
under occlusions, we followed two approaches. The first is image inpainting
using the pre-trained pluralistic image completion network. The second is
Cutmix, a regularization strategy consisting of mixing training images and
their labels using rectangular patches, making the classifier more robust
against input corruptions. Both strategies revealed effective and interesting
results were observed. In particular, the Cutmix approach makes the network
more robust without requiring additional steps at the application time, though
its training time is considerably longer. Our datasets containing the different
occlusion masks as well as their inpainted counterparts are made publicly
available to promote research on the field.

    

### [[2109.09103] A Framework for Institutional Risk Identification using Knowledge Graphs and Automated News Profiling](http://arxiv.org/abs/2109.09103)


  Organizations around the world face an array of risks impacting their
operations globally. It is imperative to have a robust risk identification
process to detect and evaluate the impact of potential risks before they
materialize. Given the nature of the task and the current requirements of deep
subject matter expertise, most organizations utilize a heavily manual process.
In our work, we develop an automated system that (a) continuously monitors
global news, (b) is able to autonomously identify and characterize risks, (c)
is able to determine the proximity of reaching triggers to determine the
distance from the manifestation of the risk impact and (d) identifies
organization's operational areas that may be most impacted by the risk. Other
contributions also include: (a) a knowledge graph representation of risks and
(b) relevant news matching to risks identified by the organization utilizing a
neural embedding model to match the textual description of a given risk with
multi-lingual news.

    

### [[2109.09113] HPTQ: Hardware-Friendly Post Training Quantization](http://arxiv.org/abs/2109.09113)


  Neural network quantization enables the deployment of models on edge devices.
An essential requirement for their hardware efficiency is that the quantizers
are hardware-friendly: uniform, symmetric, and with power-of-two thresholds. To
the best of our knowledge, current post-training quantization methods do not
support all of these constraints simultaneously. In this work, we introduce a
hardware-friendly post training quantization (HPTQ) framework, which addresses
this problem by synergistically combining several known quantization methods.
We perform a large-scale study on four tasks: classification, object detection,
semantic segmentation and pose estimation over a wide variety of network
architectures. Our extensive experiments show that competitive results can be
obtained under hardware-friendly constraints.

    

### [[2109.09138] Multi-Task Learning in Natural Language Processing: An Overview](http://arxiv.org/abs/2109.09138)


  Deep learning approaches have achieved great success in the field of Natural
Language Processing (NLP). However, deep neural models often suffer from
overfitting and data scarcity problems that are pervasive in NLP tasks. In
recent years, Multi-Task Learning (MTL), which can leverage useful information
of related tasks to achieve simultaneous performance improvement on multiple
related tasks, has been used to handle these problems. In this paper, we give
an overview of the use of MTL in NLP tasks. We first review MTL architectures
used in NLP tasks and categorize them into four classes, including the parallel
architecture, hierarchical architecture, modular architecture, and generative
adversarial architecture. Then we present optimization techniques on loss
construction, data sampling, and task scheduling to properly train a multi-task
model. After presenting applications of MTL in a variety of NLP tasks, we
introduce some benchmark datasets. Finally, we make a conclusion and discuss
several possible research directions in this field.

    

### [[2109.09148] RSI-Net: Two-Stream Deep Neural Network Integrating GCN and Atrous CNN for Semantic Segmentation of High-resolution Remote Sensing Images](http://arxiv.org/abs/2109.09148)


  For semantic segmentation of remote sensing images (RSI), trade-off between
representation power and location accuracy is quite important. How to get the
trade-off effectively is an open question, where current approaches of
utilizing attention schemes or very deep models result in complex models with
large memory consumption. Compared with the popularly-used convolutional neural
network (CNN) with fixed square kernels, graph convolutional network (GCN) can
explicitly utilize correlations between adjacent land covers and conduct
flexible convolution on arbitrarily irregular image regions. However, the
problems of large variations of target scales and blurred boundary cannot be
easily solved by GCN, while densely connected atrous convolution network
(DenseAtrousCNet) with multi-scale atrous convolution can expand the receptive
fields and obtain image global information. Inspired by the advantages of both
GCN and Atrous CNN, a two-stream deep neural network for semantic segmentation
of RSI (RSI-Net) is proposed in this paper to obtain improved performance
through modeling and propagating spatial contextual structure effectively and a
novel decoding scheme with image-level and graph-level combination. Extensive
experiments are implemented on the Vaihingen, Potsdam and Gaofen RSI datasets,
where the comparison results demonstrate the superior performance of RSI-Net in
terms of overall accuracy, F1 score and kappa coefficient when compared with
six state-of-the-art RSI semantic segmentation methods.

    

### [[2109.09160] An Exploration And Validation of Visual Factors in Understanding Classification Rule Sets](http://arxiv.org/abs/2109.09160)


  Rule sets are often used in Machine Learning (ML) as a way to communicate the
model logic in settings where transparency and intelligibility are necessary.
Rule sets are typically presented as a text-based list of logical statements
(rules). Surprisingly, to date there has been limited work on exploring visual
alternatives for presenting rules. In this paper, we explore the idea of
designing alternative representations of rules, focusing on a number of visual
factors we believe have a positive impact on rule readability and
understanding. We then presents a user study exploring their impact. The
results show that some design factors have a strong impact on how efficiently
readers can process the rules while having minimal impact on accuracy. This
work can help practitioners employ more effective solutions when using rules as
a communication strategy to understand ML models.

    

### [[2109.09163] CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation](http://arxiv.org/abs/2109.09163)


  Task-relevant grasping is critical for industrial assembly, where downstream
manipulation tasks constrain the set of valid grasps. Learning how to perform
this task, however, is challenging, since task-relevant grasp labels are hard
to define and annotate. There is also yet no consensus on proper
representations for modeling or off-the-shelf tools for performing
task-relevant grasps. This work proposes a framework to learn task-relevant
grasping for industrial objects without the need of time-consuming real-world
data collection or manual annotation. To achieve this, the entire framework is
trained solely in simulation, including supervised training with synthetic
label generation and self-supervised, hand-object interaction. In the context
of this framework, this paper proposes a novel, object-centric canonical
representation at the category level, which allows establishing dense
correspondence across object instances and transferring task-relevant grasps to
novel instances. Extensive experiments on task-relevant grasping of
densely-cluttered industrial objects are conducted in both simulation and
real-world setups, demonstrating the effectiveness of the proposed framework.
Code and data will be released upon acceptance at
this https URL.

    

### [[2109.09191] Training Dynamic based data filtering may not work for NLP datasets](http://arxiv.org/abs/2109.09191)


  The recent increase in dataset size has brought about significant advances in
natural language understanding. These large datasets are usually collected
through automation (search engines or web crawlers) or crowdsourcing which
inherently introduces incorrectly labeled data. Training on these datasets
leads to memorization and poor generalization. Thus, it is pertinent to develop
techniques that help in the identification and isolation of mislabelled data.
In this paper, we study the applicability of the Area Under the Margin (AUM)
metric to identify and remove/rectify mislabelled examples in NLP datasets. We
find that mislabelled samples can be filtered using the AUM metric in NLP
datasets but it also removes a significant number of correctly labeled points
and leads to the loss of a large amount of relevant language information. We
show that models rely on the distributional information instead of relying on
syntactic and semantic representations.

    

### [[2109.09202] Automated and Explainable Ontology Extension Based on Deep Learning: A Case Study in the Chemical Domain](http://arxiv.org/abs/2109.09202)


  Reference ontologies provide a shared vocabulary and knowledge resource for
their domain. Manual construction enables them to maintain a high quality,
allowing them to be widely accepted across their community. However, the manual
development process does not scale for large domains. We present a new
methodology for automatic ontology extension and apply it to the ChEBI
ontology, a prominent reference ontology for life sciences chemistry. We
trained a Transformer-based deep learning model on the leaf node structures
from the ChEBI ontology and the classes to which they belong. The model is then
capable of automatically classifying previously unseen chemical structures. The
proposed model achieved an overall F1 score of 0.80, an improvement of 6
percentage points over our previous results on the same dataset. Additionally,
we demonstrate how visualizing the model's attention weights can help to
explain the results by providing insight into how the model made its decisions.

    

### [[2109.09213] Capsule networks with non-iterative cluster routing](http://arxiv.org/abs/2109.09213)


  Capsule networks use routing algorithms to flow information between
consecutive layers. In the existing routing procedures, capsules produce
predictions (termed votes) for capsules of the next layer. In a nutshell, the
next-layer capsule's input is a weighted sum over all the votes it receives. In
this paper, we propose non-iterative cluster routing for capsule networks. In
the proposed cluster routing, capsules produce vote clusters instead of
individual votes for next-layer capsules, and each vote cluster sends its
centroid to a next-layer capsule. Generally speaking, the next-layer capsule's
input is a weighted sum over the centroid of each vote cluster it receives. The
centroid that comes from a cluster with a smaller variance is assigned a larger
weight in the weighted sum process. Compared with the state-of-the-art capsule
networks, the proposed capsule networks achieve the best accuracy on the
Fashion-MNIST and SVHN datasets with fewer parameters, and achieve the best
accuracy on the smallNORB and CIFAR-10 datasets with a moderate number of
parameters. The proposed capsule networks also produce capsules with
disentangled representation and generalize well to images captured at novel
viewpoints. The proposed capsule networks also preserve 2D spatial information
of an input image in the capsule channels: if the capsule channels are rotated,
the object reconstructed from these channels will be rotated by the same
transformation. Codes are available at
this https URL.

    

### [[1609.01995] Unifying task specification in reinforcement learning](http://arxiv.org/abs/1609.01995)


  Reinforcement learning tasks are typically specified as Markov decision
processes. This formalism has been highly successful, though specifications
often couple the dynamics of the environment and the learning objective. This
lack of modularity can complicate generalization of the task specification, as
well as obfuscate connections between different task settings, such as episodic
and continuing. In this work, we introduce the RL task formalism, that provides
a unification through simple constructs including a generalization to
transition-based discounting. Through a series of examples, we demonstrate the
generality and utility of this formalism. Finally, we extend standard learning
constructs, including Bellman operators, and extend some seminal theoretical
results, including approximation errors bounds. Overall, we provide a
well-understood and sound formalism on which to build theoretical results and
simplify algorithm use and development.

    

### [[2008.06692] How to build your own ASP-based system?!](http://arxiv.org/abs/2008.06692)


  Answer Set Programming (ASP) has become a popular and quite sophisticated
approach to declarative problem solving. This is arguably due to its attractive
modeling-grounding-solving workflow that provides an easy approach to problem
solving, even for laypersons outside computer science. Unlike this, the high
degree of sophistication of the underlying technology makes it increasingly
hard for ASP experts to put ideas into practice.
For addressing this issue, this tutorial aims at enabling users to build
their own ASP-based systems. More precisely, we show how the ASP system CLINGO
can be used for extending ASP and for implementing customized special-purpose
systems. To this end, we propose two alternatives. We begin with a traditional
AI technique and show how meta programming can be used for extending ASP. This
is a rather light approach that relies on CLINGO's reification feature to use
ASP itself for expressing new functionalities. Unlike this, the major part of
this tutorial uses traditional programming (in PYTHON) for manipulating CLINGO
via its application programming interface. This approach allows for changing
and controlling the entire model-ground-solve workflow of ASP. Central to this
is CLINGO's new Application class that allows us to draw on CLINGO's
infrastructure by customizing processes similar to the one in CLINGO. For
instance, we may engage manipulations to programs' abstract syntax trees,
control various forms of multi-shot solving, and set up theory propagators for
foreign inferences. Another cross-sectional structure, spanning meta as well as
application programming, is CLINGO's intermediate format, ASPIF, that specifies
the interface among the underlying grounder and solver. We illustrate the
aforementioned concepts and techniques throughout this tutorial by means of
examples and several non-trivial case-studies.

    

### [[2101.00133] NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned](http://arxiv.org/abs/2101.00133)


  We review the EfficientQA competition from NeurIPS 2020. The competition
focused on open-domain question answering (QA), where systems take natural
language questions as input and return natural language answers. The aim of the
competition was to build systems that can predict correct answers while also
satisfying strict on-disk memory budgets. These memory budgets were designed to
encourage contestants to explore the trade-off between storing retrieval
corpora or the parameters of learned models. In this report, we describe the
motivation and organization of the competition, review the best submissions,
and analyze system predictions to inform a discussion of evaluation for
open-domain QA.

    

### [[2103.11731] Meta-DETR: Image-Level Few-Shot Object Detection with Inter-Class Correlation Exploitation](http://arxiv.org/abs/2103.11731)


  Few-shot object detection has been extensively investigated by incorporating
meta-learning into region-based detection frameworks. Despite its success, the
said paradigm is constrained by several factors, such as (i) low-quality region
proposals for novel classes and (ii) negligence of the inter-class correlation
among different classes. Such limitations hinder the generalization of
base-class knowledge for the detection of novel-class objects. In this work, we
design Meta-DETR, a novel few-shot detection framework that incorporates
correlational aggregation for meta-learning into DETR detection frameworks.
Meta-DETR works entirely at image level without any region proposals, which
circumvents the constraint of inaccurate proposals in prevalent few-shot
detection frameworks. Besides, Meta-DETR can simultaneously attend to multiple
support classes within a single feed-forward. This unique design allows
capturing the inter-class correlation among different classes, which
significantly reduces the misclassification of similar classes and enhances
knowledge generalization to novel classes. Experiments over multiple few-shot
object detection benchmarks show that the proposed Meta-DETR outperforms
state-of-the-art methods by large margins. The implementation codes will be
released at this https URL.

    

### [[2104.07190] An Alignment-Agnostic Model for Chinese Text Error Correction](http://arxiv.org/abs/2104.07190)


  This paper investigates how to correct Chinese text errors with types of
mistaken, missing and redundant characters, which is common for Chinese native
speakers. Most existing models based on detect-correct framework can correct
mistaken characters errors, but they cannot deal with missing or redundant
characters. The reason is that lengths of sentences before and after correction
are not the same, leading to the inconsistence between model inputs and
outputs. Although the Seq2Seq-based or sequence tagging methods provide
solutions to the problem and achieved relatively good results on English
context, but they do not perform well in Chinese context according to our
experimental results. In our work, we propose a novel detect-correct framework
which is alignment-agnostic, meaning that it can handle both text aligned and
non-aligned occasions, and it can also serve as a cold start model when there
are no annotated data provided. Experimental results on three datasets
demonstrate that our method is effective and achieves the best performance
among existing published models.

    

### [[2104.08445] Joint Passage Ranking for Diverse Multi-Answer Retrieval](http://arxiv.org/abs/2104.08445)


  We study multi-answer retrieval, an under-explored problem that requires
retrieving passages to cover multiple distinct answers for a given question.
This task requires joint modeling of retrieved passages, as models should not
repeatedly retrieve passages containing the same answer at the cost of missing
a different valid answer. In this paper, we introduce JPR, the first joint
passage retrieval model for multi-answer retrieval. JPR makes use of an
autoregressive reranker that selects a sequence of passages, each conditioned
on previously selected passages. JPR is trained to select passages that cover
new answers at each timestep and uses a tree-decoding algorithm to enable
flexibility in the degree of diversity. Compared to prior approaches, JPR
achieves significantly better answer coverage on three multi-answer datasets.
When combined with downstream question answering, the improved retrieval
enables larger answer generation models since they need to consider fewer
passages, establishing a new state-of-the-art.

    

### [[2104.08451] Context-Aware Interaction Network for Question Matching](http://arxiv.org/abs/2104.08451)


  Impressive milestones have been achieved in text matching by adopting a
cross-attention mechanism to capture pertinent semantic connections between two
sentence representations. However, regular cross-attention focuses on
word-level links between the two input sequences, neglecting the importance of
contextual information. We propose a context-aware interaction network (COIN)
to properly align two sequences and infer their semantic relationship.
Specifically, each interaction block includes (1) a context-aware
cross-attention mechanism to effectively integrate contextual information when
aligning two sequences, and (2) a gate fusion layer to flexibly interpolate
aligned representations. We apply multiple stacked interaction blocks to
produce alignments at different levels and gradually refine the attention
results. Experiments on two question matching datasets and detailed analyses
demonstrate the effectiveness of our model.

    

### [[2104.08661] Explaining Answers with Entailment Trees](http://arxiv.org/abs/2104.08661)


  Our goal, in the context of open-domain textual question-answering (QA), is
to explain answers by showing the line of reasoning from what is known to the
answer, rather than simply showing a fragment of textual evidence (a
"rationale'"). If this could be done, new opportunities for understanding and
debugging the system's reasoning become possible. Our approach is to generate
explanations in the form of entailment trees, namely a tree of multipremise
entailment steps from facts that are known, through intermediate conclusions,
to the hypothesis of interest (namely the question + answer). To train a model
with this skill, we created ENTAILMENTBANK, the first dataset to contain
multistep entailment trees. Given a hypothesis (question + answer), we define
three increasingly difficult explanation tasks: generate a valid entailment
tree given (a) all relevant sentences (b) all relevant and some irrelevant
sentences, or (c) a corpus. We show that a strong language model can partially
solve these tasks, in particular when the relevant sentences are included in
the input (e.g., 35% of trees for (a) are perfect), and with indications of
generalization to other domains. This work is significant as it provides a new
type of dataset (multistep entailments) and baselines, offering a new avenue
for the community to generate richer, more systematic explanations.

    

### [[2106.00588] TransVOS: Video Object Segmentation with Transformers](http://arxiv.org/abs/2106.00588)


  Recently, Space-Time Memory Network (STM) based methods have achieved
state-of-the-art performance in semi-supervised video object segmentation
(VOS). A crucial problem in this task is how to model the dependency both among
different frames and inside every frame. However, most of these methods neglect
the spatial relationships (inside each frame) and do not make full use of the
temporal relationships (among different frames). In this paper, we propose a
new transformer-based framework, termed TransVOS, introducing a vision
transformer to fully exploit and model both the temporal and spatial
relationships. Moreover, most STM-based approaches employ two separate encoders
to extract features of two significant inputs, i.e., reference sets (history
frames with predicted masks) and query frame (current frame), respectively,
increasing the models' parameters and complexity. To slim the popular
two-encoder pipeline while keeping the effectiveness, we design a single
two-path feature extractor to encode the above two inputs in a unified way.
Extensive experiments demonstrate the superiority of our TransVOS over
state-of-the-art methods on both DAVIS and YouTube-VOS datasets.

    

### [[2106.07176] SAS: Self-Augmented Strategy for Language Model Pre-training](http://arxiv.org/abs/2106.07176)


  The core of a self-supervised learning method for pre-training language
models includes the design of appropriate data augmentation and corresponding
pre-training task(s). Most data augmentations in language model pre-training
are context-independent. The seminal contextualized augmentation recently
proposed by the ELECTRA requires a separate generator, which leads to extra
computation cost as well as the challenge in adjusting the capability of its
generator relative to that of the other model component(s). We propose a
self-augmented strategy (SAS) that uses a single forward pass through the model
to augment the input data for model training in the next epoch. Essentially our
strategy eliminates a separate generator network and uses only one network to
generate the data augmentation and undertake two pre-training tasks (the MLM
task and the RTD task) jointly, which naturally avoids the challenge in
adjusting the generator's capability as well as reduces the computation cost.
Additionally, our SAS is a general strategy such that it can seamlessly
incorporate many new techniques emerging recently or in the future, such as the
disentangled attention mechanism recently proposed by the DeBERTa model. Our
experiments show that our SAS is able to outperform the ELECTRA and other
state-of-the-art models in the GLUE tasks with the same or less computation
cost.

    

### [[2106.07857] Bilateral Personalized Dialogue Generation with Dynamic Persona-Aware Fusion](http://arxiv.org/abs/2106.07857)


  Generating personalized responses is one of the major challenges in natural
human-robot interaction. Current researches in this field mainly focus on
generating responses consistent with the robot's pre-assigned persona, while
ignoring the user's persona. Such responses may be inappropriate or even
offensive, which may lead to the bad user experience. Therefore, we propose a
bilateral personalized dialogue generation (BPDG) method with dynamic
persona-aware fusion via multi-task transfer learning to generate responses
consistent with both personas. The proposed method aims to accomplish three
learning tasks: 1) an encoder is trained with dialogue utterances added with
corresponded personalized attributes and relative position (language model
task), 2) a dynamic persona-aware fusion module predicts the persona presence
to adaptively fuse the contextual and bilateral personas encodings (persona
prediction task) and 3) a decoder generates natural, fluent and personalized
responses (dialogue generation task). To make the generated responses more
personalized and bilateral persona-consistent, the Conditional Mutual
Information Maximum (CMIM) criterion is adopted to select the final response
from the generated candidates. The experimental results show that the proposed
method outperforms several state-of-the-art methods in terms of both automatic
and manual evaluations.

    

### [[2106.14556] Contrastive Counterfactual Visual Explanations With Overdetermination](http://arxiv.org/abs/2106.14556)


  A novel explainable AI method called CLEAR Image is introduced in this paper.
CLEAR Image is based on the view that a satisfactory explanation should be
contrastive, counterfactual and measurable. CLEAR Image explains an image's
classification probability by contrasting the image with a corresponding image
generated automatically via adversarial learning. This enables both salient
segmentation and perturbations that faithfully determine each segment's
importance. CLEAR Image was successfully applied to a medical imaging case
study where it outperformed methods such as Grad-CAM and LIME by an average of
27% using a novel pointing game metric. CLEAR Image excels in identifying cases
of "causal overdetermination" where there are multiple patches in an image, any
one of which is sufficient by itself to cause the classification probability to
be close to one.

    

### [[2109.08710] On-device neural speech synthesis](http://arxiv.org/abs/2109.08710)


  Recent advances in text-to-speech (TTS) synthesis, such as Tacotron and
WaveRNN, have made it possible to construct a fully neural network based TTS
system, by coupling the two components together. Such a system is conceptually
simple as it only takes grapheme or phoneme input, uses Mel-spectrogram as an
intermediate feature, and directly generates speech samples. The system
achieves quality equal or close to natural speech. However, the high
computational cost of the system and issues with robustness have limited their
usage in real-world speech synthesis applications and products. In this paper,
we present key modeling improvements and optimization strategies that enable
deploying these models, not only on GPU servers, but also on mobile devices.
The proposed system can generate high-quality 24 kHz speech at 5x faster than
real time on server and 3x faster than real time on mobile devices.

    

### [[2109.09244] A domain specific modeling and analysis environment for complex IoT applications](http://arxiv.org/abs/2109.09244)


  To cope with the complexities found in the Internet of Things domain,
designers and developers of IoT applications demand practical tools. Several
model-driven engineering methodologies and tools have been developed to address
such difficulties, but few of them address the analysis aspects. In this
extended abstract, we introduce CHESSIoT, a domain-specific modeling
environment for complex IoT applications. In addition, the existing supported
real-time analysis mechanism, as well as a proposed code generation approach,
are presented

    

### [[2006.14969] Fully Abstract and Robust Compilation and How to Reconcile the Two, Abstractly](http://arxiv.org/abs/2006.14969)


  The most prominent formal criterion for secure compilation is full
abstraction, the preservation and reflection of contextual equivalence. Recent
work introduced robust compilation, defined as the preservation of robust
satisfaction of hyperproperties, i.e., their satisfaction against arbitrary
attackers. In this paper, we initially set out to compare these two approaches
to secure compilation. To that end, we provide an exact description of the
hyperproperties that are robustly satisfied by programs compiled with a fully
abstract compiler, and show that they can be meaningless or trivial. We then
propose a novel criterion for secure compilation formulated in the framework of
Mathematical Operational Semantics (MOS), guaranteeing both full abstraction
and the preservation of robust satisfaction of hyperproperties in a more
sensible manner.

    