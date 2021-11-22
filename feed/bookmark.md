
## 2021-11-22

### [[2111.09955] SmartSlice: Dynamic, self-optimization of applications QoS requests to 5G networks](http://arxiv.org/abs/2111.09955)


  Applications can tailor a network slice by specifying a variety of QoS
attributes related to application-specific performance, function or operation.
However, some QoS attributes like guaranteed bandwidth required by the
application do vary over time. For example, network bandwidth needs of video
streams from surveillance cameras can vary a lot depending on the environmental
conditions and the content in the video streams. In this paper, we propose a
novel, dynamic QoS attribute prediction technique that assists any application
to make optimal resource reservation requests at all times. Standard
forecasting using traditional cost functions like MAE, MSE, RMSE, MDA, etc.
don't work well because they do not take into account the direction (whether
the forecasting of resources is more or less than needed), magnitude (by how
much the forecast deviates, and in which direction), or frequency (how many
times the forecast deviates from actual needs, and in which direction). The
direction, magnitude and frequency have a direct impact on the application's
accuracy of insights, and the operational costs. We propose a new,
parameterized cost function that takes into account all three of them, and
guides the design of a new prediction technique. To the best of our knowledge,
this is the first work that considers time-varying application requirements and
dynamically adjusts slice QoS requests to 5G networks in order to ensure a
balance between application's accuracy and operational costs. In a real-world
deployment of a surveillance video analytics application over 17 cameras, we
show that our technique outperforms other traditional forecasting methods, and
it saves 34% of network bandwidth (over a ~24 hour period) when compared to a
static, one-time reservation.

    

### [[2111.09960] Reining in Mobile Web Performance with Document and Permission Policies](http://arxiv.org/abs/2111.09960)


  The quality of experience with the mobile web remains poor, partially as a
result of complex websites and design choices that worsen performance,
particularly for users in suboptimal networks or devices. Prior proposed
solutions have seen limited adoption due in part to the demand they place on
developers and content providers, and the performing infrastructure needed to
support them. We argue that Document and Permissions Policies -- an ongoing
effort to enforce good practices on web design -- may offer the basis for a
readily-available and easily-adoptable solution. In this paper, we evaluate the
potential performance cost of violating well-understood policies and how common
such violations are in today's web. Our analysis show, for example, that
controlling for unsized-media policy, something applicable to 70% of the
top-1million websites, can indeed reduce Cumulative Layout Shift metric.

    

### [[2111.10073] An Asynchronous Multi-Beam MAC Protocol for Multi-Hop Wireless Networks](http://arxiv.org/abs/2111.10073)


  A node equipped with a multi-beam antenna can achieve a throughput of up to m
times as compared to a single-beam antenna, by simultaneously communicating on
its m non-interfering beams. However, the existing multi-beam medium access
control (MAC) schemes can achieve concurrent data communication only when the
transmitter nodes are locally synchronized. Asynchronous packet arrival at a
multi-beam receiver node would increase the node deafness and MAC layer capture
problems, and thereby limit the data throughput. This paper presents an
asynchronous multi-beam MAC protocol for multi-hop wireless networks, which
makes the following enhancements to the existing multi-beam MAC schemes (i) A
windowing mechanism to achieve concurrent communication when the packet arrival
is asynchronous, (ii) A smart packet processing mechanism which reduces the
node deafness, hidden terminals and MAC-layer capture problems, and (iii) A
channel access mechanism which decreases resource wastage and node starvation.
Our proposed protocol also works in heterogeneous networks that deploy the
nodes equipped with single-beam as well as multi-beam antennas. Simulation
results demonstrate a superior performance of our proposed protocol.

    

### [[2111.10076] Edge Computing vs Centralized Cloud: Impact of Communication Latency on the Energy Consumption of LTE Terminal Nodes](http://arxiv.org/abs/2111.10076)


  Edge computing brings several advantages, such as reduced latency, increased
bandwidth, and improved locality of traffic. One aspect that is not
sufficiently understood is the impact of the different communication latency
experienced in the edge-cloud continuum on the energy consumption of clients.
We studied how a request-response communication scheme is influenced by
different placements of the server, when communication is based on LTE. Results
show that by accurately selecting the operational parameters a significant
amount of energy can be saved.

    

### [[2111.10115] IRONWAN: Increasing Reliability of Overlapping Networks in LoRaWAN](http://arxiv.org/abs/2111.10115)


  LoRaWAN deployments follow an ad-hoc deployment model that has organically
led to overlapping communication networks, sharing the wireless spectrum, and
completely unaware of each other. LoRaWAN uses ALOHA-style communication where
it is almost impossible to schedule transmission between networks belonging to
different owners properly. The inability to schedule overlapping networks will
cause inter-network interference, which will increase node-to-gateway message
losses and gateway-to-node acknowledgement failures. This problem is likely to
get worse as the number of LoRaWAN networks increase. In response to this
problem, we propose IRONWAN, a wireless overlay network that shares
communication resources without modifications to underlying protocols. It
utilises the broadcast nature of radio communication and enables
gateway-to-gateway communication to facilitate the search for failed messages
and transmit failed acknowledgements already received and cached in overlapping
network's gateways. IRONWAN uses two novel algorithms, a Real-time Message
Inter-arrival Predictor, to highlight when a server has not received an
expected uplink message. The Interference Predictor ensures that extra
gateway-to-gateway communication does not negatively impact communication
bandwidth. We evaluate IRONWAN on a 1000-node simulator with up to ten gateways
and a 10-node testbed with 2-gateways. Results show that IRONWAN can achieve up
to 12\% higher packet delivery ratio (PDR) and total messages received per node
while increasing the minimum PDR by up to 28\%. These improvements save up to
50\% node's energy. Finally, we demonstrate that IRONWAN has comparable
performance to an optimal solution (wired, centralised) but with 2-32 times
lower communication costs. IRONWAN also has up to 14\% better PDR when compared
to FLIP, a wired-distributed gateway-to-gateway protocol in certain scenarios.

    

### [[2111.10219] A Survey on Rural Internet Connectivity in India](http://arxiv.org/abs/2111.10219)


  Rural connectivity is widely research topic for several years. In India,
around 70% of the population have poor or no connectivity to access digital
services. Different solutions are being tested and trialled around the world,
especially in India. They key driving factor for reducing digital divide is
exploring different solutions both technologically and economically to lower
the cost for the network deployments and improving service adoption rate. In
this survey, we aim to study the rural connectivity use-cases, state of art
projects and initiatives, challenges, and technologies to improve digital
connectivity in rural parts of India. The strengths and weakness of different
technologies which are being tested for rural connectivity is analyzed. We also
explore the rural use-case of 6G communication system which would be suitable
for rural Indian scenario.

    

### [[2111.10261] Optimal Association Strategy of Multi-gateway Wireless Sensor Networks Against Smart Jammers](http://arxiv.org/abs/2111.10261)


  Engineers have numerous low-power wireless sensor devices in the current
network setup for the Internet of Things, such as ZigBee, LoRaWAN, ANT, or
Bluetooth. These low-power wireless sensors are the best candidates to transfer
and collect data. But they are all vulnerable to the physical jamming attack
since it is not costly for the attackers to run low power jammer sources in
these networks. Having multiple gateways and providing alternative connections
to sensors would help these networks to mitigate successful jamming. In this
paper, we propose an analytical model to solve the problem of gateway selection
and association based on a Stackelberg game, where the jammer is the follower.
We first formulate the payoffs of both sensor network and attacker and then
establish and prove the conditions leading to NASH equilibrium. With numerical
investigation, we also present how our model can capture the performance of
sensor networks under jamming with a varying number of gateways. Our results
show that compared to the single gateway scenario, the network's throughput
will improve by 26% and 60% when we deploy two and four gateways in the
presence of a single jammer.

    

### [[2111.09902] A transformer-based model for default prediction in mid-cap corporate markets](http://arxiv.org/abs/2111.09902)


  In this paper, we study mid-cap companies, i.e. publicly traded companies
with less than US $10 billion in market capitalisation. Using a large dataset
of US mid-cap companies observed over 30 years, we look to predict the default
probability term structure over the medium term and understand which data
sources (i.e. fundamental, market or pricing data) contribute most to the
default risk. Whereas existing methods typically require that data from
different time periods are first aggregated and turned into cross-sectional
features, we frame the problem as a multi-label time-series classification
problem. We adapt transformer models, a state-of-the-art deep learning model
emanating from the natural language processing domain, to the credit risk
modelling setting. We also interpret the predictions of these models using
attention heat maps. To optimise the model further, we present a custom loss
function for multi-label classification and a novel multi-channel architecture
with differential training that gives the model the ability to use all input
data efficiently. Our results show the proposed deep learning architecture's
superior performance, resulting in a 13% improvement in AUC (Area Under the
receiver operating characteristic Curve) over traditional models. We also
demonstrate how to produce an importance ranking for the different data sources
and the temporal relationships using a Shapley approach specific to these
models.

    

### [[2111.09930] Learning To Estimate Regions Of Attraction Of Autonomous Dynamical Systems Using Physics-Informed Neural Networks](http://arxiv.org/abs/2111.09930)


  When learning to perform motor tasks in a simulated environment, neural
networks must be allowed to explore their action space to discover new
potentially viable solutions. However, in an online learning scenario with
physical hardware, this exploration must be constrained by relevant safety
considerations in order to avoid damage to the agent's hardware and
environment. We aim to address this problem by training a neural network, which
we will refer to as a "safety network", to estimate the region of attraction
(ROA) of a controlled autonomous dynamical system. This safety network can
thereby be used to quantify the relative safety of proposed control actions and
prevent the selection of damaging actions. Here we present our development of
the safety network by training an artificial neural network (ANN) to represent
the ROA of several autonomous dynamical system benchmark problems. The training
of this network is predicated upon both Lyapunov theory and neural solutions to
partial differential equations (PDEs). By learning to approximate the viscosity
solution to a specially chosen PDE that contains the dynamics of the system of
interest, the safety network learns to approximate a particular function,
similar to a Lyapunov function, whose zero level set is boundary of the ROA. We
train our safety network to solve these PDEs in a semi-supervised manner
following a modified version of the Physics Informed Neural Network (PINN)
approach, utilizing a loss function that penalizes disagreement with the PDE's
initial and boundary conditions, as well as non-zero residual and variational
terms. In future work we intend to apply this technique to reinforcement
learning agents during motor learning tasks.

    

### [[2111.09933] Loss Functions for Discrete Contextual Pricing with Observational Data](http://arxiv.org/abs/2111.09933)


  We study a pricing setting where each customer is offered a contextualized
price based on customer and/or product features that are predictive of the
customer's valuation for that product. Often only historical sales records are
available, where we observe whether each customer purchased a product at the
price prescribed rather than the customer's true valuation. As such, the data
is influenced by the historical sales policy which introduces difficulties in
a) estimating future loss/regret for pricing policies without the possibility
of conducting real experiments and b) optimizing new policies for downstream
tasks such as revenue management. We study how to formulate loss functions
which can be used for optimizing pricing policies directly, rather than going
through an intermediate demand estimation stage, which can be biased in
practice due to model misspecification, regularization or poor calibration.
While existing approaches have been proposed when valuation data is available,
we propose loss functions for the observational data setting. To achieve this,
we adapt ideas from machine learning with corrupted labels, where we can
consider each observed customer's outcome (purchased or not for a prescribed
price), as a (known) probabilistic transformation of the customer's valuation.
From this transformation we derive a class of suitable unbiased loss functions.
Within this class we identify minimum variance estimators, those which are
robust to poor demand function estimation, and provide guidance on when the
estimated demand function is useful. Furthermore, we also show that when
applied to our contextual pricing setting, estimators popular in the off-policy
evaluation literature fall within this class of loss functions, and also offer
managerial insights on when each estimator is likely to perform well in
practice.

    

### [[2111.09939] Explainable predictions of different machine learning algorithms used to predict Early Stage diabetes](http://arxiv.org/abs/2111.09939)


  Machine Learning and Artificial Intelligence can be widely used to diagnose
chronic diseases so that necessary precautionary treatment can be done in
critical time. Diabetes Mellitus which is one of the major diseases can be
easily diagnosed by several Machine Learning algorithms. Early stage diagnosis
is crucial to prevent dangerous consequences. In this paper we have made a
comparative analysis of several machine learning algorithms viz. Random Forest,
Decision Tree, Artificial Neural Networks, K Nearest Neighbor, Support Vector
Machine, and XGBoost along with feature attribution using SHAP to identify the
most important feature in predicting the diabetes on a dataset collected from
Sylhet Hospital. As per the experimental results obtained, the Random Forest
algorithm has outperformed all the other algorithms with an accuracy of 99
percent on this particular dataset.

    

### [[2111.09954] MS-nowcasting: Operational Precipitation Nowcasting with Convolutional LSTMs at Microsoft Weather](http://arxiv.org/abs/2111.09954)


  We present the encoder-forecaster convolutional long short-term memory (LSTM)
deep-learning model that powers Microsoft Weather's operational precipitation
nowcasting product. This model takes as input a sequence of weather radar
mosaics and deterministically predicts future radar reflectivity at lead times
up to 6 hours. By stacking a large input receptive field along the feature
dimension and conditioning the model's forecaster with predictions from the
physics-based High Resolution Rapid Refresh (HRRR) model, we are able to
outperform optical flow and HRRR baselines by 20-25% on multiple metrics
averaged over all lead times.

    

### [[2111.09961] A Review of Adversarial Attack and Defense for Classification Methods](http://arxiv.org/abs/2111.09961)


  Despite the efficiency and scalability of machine learning systems, recent
studies have demonstrated that many classification methods, especially deep
neural networks (DNNs), are vulnerable to adversarial examples; i.e., examples
that are carefully crafted to fool a well-trained classification model while
being indistinguishable from natural data to human. This makes it potentially
unsafe to apply DNNs or related methods in security-critical areas. Since this
issue was first identified by Biggio et al. (2013) and Szegedy et al.(2014),
much work has been done in this field, including the development of attack
methods to generate adversarial examples and the construction of defense
techniques to guard against such examples. This paper aims to introduce this
topic and its latest developments to the statistical community, primarily
focusing on the generation and guarding of adversarial examples. Computing
codes (in python and R) used in the numerical experiments are publicly
available for readers to explore the surveyed methods. It is the hope of the
authors that this paper will encourage more statisticians to work on this
important and exciting field of generating and defending against adversarial
examples.

    

### [[2111.09963] Beyond NDCG: behavioral testing of recommender systems with RecList](http://arxiv.org/abs/2111.09963)


  As with most Machine Learning systems, recommender systems are typically
evaluated through performance metrics computed over held-out data points.
However, real-world behavior is undoubtedly nuanced: ad hoc error analysis and
deployment-specific tests must be employed to ensure the desired quality in
actual deployments. In this paper, we propose RecList, a behavioral-based
testing methodology. RecList organizes recommender systems by use case and
introduces a general plug-and-play procedure to scale up behavioral testing. We
demonstrate its capabilities by analyzing known algorithms and black-box
commercial systems, and we release RecList as an open source, extensible
package for the community.

    

### [[2111.09964] Deep IDA: A Deep Learning Method for Integrative Discriminant Analysis of Multi-View Data with Feature Ranking -- An Application to COVID-19 severity](http://arxiv.org/abs/2111.09964)


  COVID-19 severity is due to complications from SARS-Cov-2 but the clinical
course of the infection varies for individuals, emphasizing the need to better
understand the disease at the molecular level. We use clinical and multiple
molecular data (or views) obtained from patients with and without COVID-19 who
were (or not) admitted to the intensive care unit to shed light on COVID-19
severity. Methods for jointly associating the views and separating the COVID-19
groups (i.e., one-step methods) have focused on linear relationships. The
relationships between the views and COVID-19 patient groups, however, are too
complex to be understood solely by linear methods. Existing nonlinear one-step
methods cannot be used to identify signatures to aid in our understanding of
the complexity of the disease. We propose Deep IDA (Integrative Discriminant
Analysis) to address analytical challenges in our problem of interest. Deep IDA
learns nonlinear projections of two or more views that maximally associate the
views and separate the classes in each view, and permits feature ranking for
interpretable findings. Our applications demonstrate that Deep IDA has
competitive classification rates compared to other state-of-the-art methods and
is able to identify molecular signatures that facilitate an understanding of
COVID-19 severity.

    

### [[2111.09971] Learning Robust Output Control Barrier Functions from Safe Expert Demonstrations](http://arxiv.org/abs/2111.09971)


  This paper addresses learning safe control laws from expert demonstrations.
We assume that appropriate models of the system dynamics and the output
measurement map are available, along with corresponding error bounds. We first
propose robust output control barrier functions (ROCBFs) as a means to
guarantee safety, as defined through controlled forward invariance of a safe
set. We then present an optimization problem to learn ROCBFs from expert
demonstrations that exhibit safe system behavior, e.g., data collected from a
human operator. Along with the optimization problem, we provide verifiable
conditions that guarantee validity of the obtained ROCBF. These conditions are
stated in terms of the density of the data and on Lipschitz and boundedness
constants of the learned function and the models of the system dynamics and the
output measurement map. When the parametrization of the ROCBF is linear, then,
under mild assumptions, the optimization problem is convex. We validate our
findings in the autonomous driving simulator CARLA and show how to learn safe
control laws from RGB camera images.

    

### [[2111.09982] Second-Order Mirror Descent: Convergence in Games Beyond Averaging and Discounting](http://arxiv.org/abs/2111.09982)


  In this paper, we propose a second-order extension of the continuous-time
game-theoretic mirror descent (MD) dynamics, referred to as MD2, which
converges to mere (but not necessarily strict) variationally stable states
(VSS) without using common auxiliary techniques such as averaging or
discounting. We show that MD2 enjoys no-regret as well as exponential rate of
convergence towards a strong VSS upon a slight modification. Furthermore, MD2
can be used to derive many novel primal-space dynamics. Lastly, using
stochastic approximation techniques, we provide a convergence guarantee of
discrete-time MD2 with noisy observations towards interior mere VSS. Selected
simulations are provided to illustrate our results.

    

### [[2111.09990] Gaussian Determinantal Processes: a new model for directionality in data](http://arxiv.org/abs/2111.09990)


  Determinantal point processes (a.k.a. DPPs) have recently become popular
tools for modeling the phenomenon of negative dependence, or repulsion, in
data. However, our understanding of an analogue of a classical parametric
statistical theory is rather limited for this class of models. In this work, we
investigate a parametric family of Gaussian DPPs with a clearly interpretable
effect of parametric modulation on the observed points. We show that parameter
modulation impacts the observed points by introducing directionality in their
repulsion structure, and the principal directions correspond to the directions
of maximal (i.e. the most long ranged) dependency.
This model readily yields a novel and viable alternative to Principal
Component Analysis (PCA) as a dimension reduction tool that favors directions
along which the data is most spread out. This methodological contribution is
complemented by a statistical analysis of a spiked model similar to that
employed for covariance matrices as a framework to study PCA. These theoretical
investigations unveil intriguing questions for further examination in random
matrix theory, stochastic geometry and related topics.

    

### [[2111.09991] Sketch-based Creativity Support Tools using Deep Learning](http://arxiv.org/abs/2111.09991)


  Sketching is a natural and effective visual communication medium commonly
used in creative processes. Recent developments in deep-learning models
drastically improved machines' ability in understanding and generating visual
content. An exciting area of development explores deep-learning approaches used
to model human sketches, opening opportunities for creative applications. This
chapter describes three fundamental steps in developing deep-learning-driven
creativity support tools that consumes and generates sketches: 1) a data
collection effort that generated a new paired dataset between sketches and
mobile user interfaces; 2) a sketch-based user interface retrieval system
adapted from state-of-the-art computer vision techniques; and, 3) a
conversational sketching system that supports the novel interaction of a
natural-language-based sketch/critique authoring process. In this chapter, we
survey relevant prior work in both the deep-learning and
human-computer-interaction communities, document the data collection process
and the systems' architectures in detail, present qualitative and quantitative
results, and paint the landscape of several future research directions in this
exciting area.

    

### [[2111.09993] Esophageal virtual disease landscape using mechanics-informed machine learning](http://arxiv.org/abs/2111.09993)


  The pathogenesis of esophageal disorders is related to the esophageal wall
mechanics. Therefore, to understand the underlying fundamental mechanisms
behind various esophageal disorders, it is crucial to map the esophageal wall
mechanics-based parameters onto physiological and pathophysiological conditions
corresponding to altered bolus transit and supraphysiologic IBP. In this work,
we present a hybrid framework that combines fluid mechanics and machine
learning to identify the underlying physics of the various esophageal disorders
and maps them onto a parameter space which we call the virtual disease
landscape (VDL). A one-dimensional inverse model processes the output from an
esophageal diagnostic device called endoscopic functional lumen imaging probe
(EndoFLIP) to estimate the mechanical "health" of the esophagus by predicting a
set of mechanics-based parameters such as esophageal wall stiffness, muscle
contraction pattern and active relaxation of esophageal walls. The
mechanics-based parameters were then used to train a neural network that
consists of a variational autoencoder (VAE) that generates a latent space and a
side network that predicts mechanical work metrics for estimating
esophagogastric junction motility. The latent vectors along with a set of
discrete mechanics-based parameters define the VDL and form clusters
corresponding to the various esophageal disorders. The VDL not only
distinguishes different disorders but can also be used to predict disease
progression in time. Finally, we also demonstrate the clinical applicability of
this framework for estimating the effectiveness of a treatment and track
patient condition after a treatment.

    

### [[2111.10003] Differentiable Wavetable Synthesis](http://arxiv.org/abs/2111.10003)


  Differentiable Wavetable Synthesis (DWTS) is a technique for neural audio
synthesis which learns a dictionary of one-period waveforms i.e. wavetables,
through end-to-end training. We achieve high-fidelity audio synthesis with as
little as 10 to 20 wavetables and demonstrate how a data-driven dictionary of
waveforms opens up unprecedented one-shot learning paradigms on short audio
clips. Notably, we show audio manipulations, such as high quality
pitch-shifting, using only a few seconds of input audio. Lastly, we investigate
performance gains from using learned wavetables for realtime and interactive
audio synthesis.

    

### [[2111.10009] ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier to Mine Exoplanets](http://arxiv.org/abs/2111.10009)


  The kepler and TESS missions have generated over 100,000 potential transit
signals that must be processed in order to create a catalog of planet
candidates. During the last few years, there has been a growing interest in
using machine learning to analyze these data in search of new exoplanets.
Different from the existing machine learning works, ExoMiner, the proposed deep
learning classifier in this work, mimics how domain experts examine diagnostic
tests to vet a transit signal. ExoMiner is a highly accurate, explainable, and
robust classifier that 1) allows us to validate 301 new exoplanets from the
MAST Kepler Archive and 2) is general enough to be applied across missions such
as the on-going TESS mission. We perform an extensive experimental study to
verify that ExoMiner is more reliable and accurate than the existing transit
signal classifiers in terms of different classification and ranking metrics.
For example, for a fixed precision value of 99%, ExoMiner retrieves 93.6% of
all exoplanets in the test set (i.e., recall=0.936) while this rate is 76.3%
for the best existing classifier. Furthermore, the modular design of ExoMiner
favors its explainability. We introduce a simple explainability framework that
provides experts with feedback on why ExoMiner classifies a transit signal into
a specific class label (e.g., planet candidate or not planet candidate).

    

### [[2111.10010] UN-AVOIDS: Unsupervised and Nonparametric Approach for Visualizing Outliers and Invariant Detection Scoring](http://arxiv.org/abs/2111.10010)


  The visualization and detection of anomalies (outliers) are of crucial
importance to many fields, particularly cybersecurity. Several approaches have
been proposed in these fields, yet to the best of our knowledge, none of them
has fulfilled both objectives, simultaneously or cooperatively, in one coherent
framework. The visualization methods of these approaches were introduced for
explaining the output of a detection algorithm, not for data exploration that
facilitates a standalone visual detection. This is our point of departure:
UN-AVOIDS, an unsupervised and nonparametric approach for both visualization (a
human process) and detection (an algorithmic process) of outliers, that assigns
invariant anomalous scores (normalized to $[0,1]$), rather than hard
binary-decision. The main aspect of novelty of UN-AVOIDS is that it transforms
data into a new space, which is introduced in this paper as neighborhood
cumulative density function (NCDF), in which both visualization and detection
are carried out. In this space, outliers are remarkably visually
distinguishable, and therefore the anomaly scores assigned by the detection
algorithm achieved a high area under the ROC curve (AUC). We assessed UN-AVOIDS
on both simulated and two recently published cybersecurity datasets, and
compared it to three of the most successful anomaly detection methods: LOF, IF,
and FABOD. In terms of AUC, UN-AVOIDS was almost an overall winner. The article
concludes by providing a preview of new theoretical and practical avenues for
UN-AVOIDS. Among them is designing a visualization aided anomaly detection
(VAAD), a type of software that aids analysts by providing UN-AVOIDS' detection
algorithm (running in a back engine), NCDF visualization space (rendered to
plots), along with other conventional methods of visualization in the original
feature space, all of which are linked in one interactive environment.

    

### [[2111.10021] Achievability and Impossibility of Exact Pairwise Ranking](http://arxiv.org/abs/2111.10021)


  We consider the problem of recovering the rank of a set of $n$ items based on
noisy pairwise comparisons. We assume the SST class as the family of generative
models. Our analysis gave sharp information theoretic upper and lower bound for
the exact requirement, which matches exactly in the parametric limit. Our tight
analysis on the algorithm induced by the moment method gave better constant in
Minimax optimal rate than ~\citet{shah2017simple} and contribute to their open
problem. The strategy we used in this work to obtain information theoretic
bounds is based on combinatorial arguments and is of independent interest.

    

### [[2111.10026] IC-U-Net: A U-Net-based Denoising Autoencoder Using Mixtures of Independent Components for Automatic EEG Artifact Removal](http://arxiv.org/abs/2111.10026)


  Electroencephalography (EEG) signals are often contaminated with artifacts.
It is imperative to develop a practical and reliable artifact removal method to
prevent misinterpretations of neural signals and underperformance of
brain-computer interfaces. This study developed a new artifact removal method,
IC-U-Net, which is based on the U-Net architecture for removing pervasive EEG
artifacts and reconstructing brain sources. The IC-U-Net was trained using
mixtures of brain and non-brain sources decomposed by independent component
analysis and employed an ensemble of loss functions to model complex signal
fluctuations in EEG recordings. The effectiveness of the proposed method in
recovering brain sources and removing various artifacts (e.g., eye
blinks/movements, muscle activities, and line/channel noises) was demonstrated
in a simulation study and three real-world EEG datasets collected at rest and
while driving and walking. IC-U-Net is user-friendly and publicly available,
does not require parameter tuning or artifact type designations, and has no
limitations on channel numbers. Given the increasing need to image natural
brain dynamics in a mobile setting, IC-U-Net offers a promising end-to-end
solution for automatically removing artifacts from EEG recordings.

    

### [[2111.10037] Explaining GNN over Evolving Graphs using Information Flow](http://arxiv.org/abs/2111.10037)


  Graphs are ubiquitous in many applications, such as social networks,
knowledge graphs, smart grids, etc.. Graph neural networks (GNN) are the
current state-of-the-art for these applications, and yet remain obscure to
humans. Explaining the GNN predictions can add transparency. However, as many
graphs are not static but continuously evolving, explaining changes in
predictions between two graph snapshots is different but equally important.
Prior methods only explain static predictions or generate coarse or irrelevant
explanations for dynamic predictions. We define the problem of explaining
evolving GNN predictions and propose an axiomatic attribution method to
uniquely decompose the change in a prediction to paths on computation graphs.
The attribution to many paths involving high-degree nodes is still not
interpretable, while simply selecting the top important paths can be suboptimal
in approximating the change. We formulate a novel convex optimization problem
to optimally select the paths that explain the prediction evolution.
Theoretically, we prove that the existing method based on
Layer-Relevance-Propagation (LRP) is a special case of the proposed algorithm
when an empty graph is compared with. Empirically, on seven graph datasets,
with a novel metric designed for evaluating explanations of prediction change,
we demonstrate the superiority of the proposed approach over existing methods,
including LRP, DeepLIFT, and other path selection methods.

    

### [[2111.10039] Modeling Flash Memory Channels Using Conditional Generative Nets](http://arxiv.org/abs/2111.10039)


  Understanding the NAND flash memory channel has become more and more
challenging due to the continually increasing density and the complex
distortions arising from the write and read mechanisms. In this work, we
propose a data-driven generative modeling method to characterize the flash
memory channel. The learned model can reconstruct the read voltage from an
individual memory cell based on the program levels of the cell and its
surrounding array of cells. Experimental results show that the statistical
distribution of the reconstructed read voltages accurately reflects the
measured distribution on a commercial flash memory chip, both qualitatively and
as quantified by the total variation distance. Moreover, we observe that the
learned model can capture precise inter-cell interference (ICI) effects, as
verified by comparison of the error probabilities of specific patterns in
wordlines and bitlines.

    

### [[2111.10041] Embeddings and labeling schemes for A*](http://arxiv.org/abs/2111.10041)


  A* is a classic and popular method for graphs search and path finding. It
assumes the existence of a heuristic function $h(u,t)$ that estimates the
shortest distance from any input node $u$ to the destination $t$.
Traditionally, heuristics have been handcrafted by domain experts. However,
over the last few years, there has been a growing interest in learning
heuristic functions. Such learned heuristics estimate the distance between
given nodes based on "features" of those nodes.
In this paper we formalize and initiate the study of such feature-based
heuristics. In particular, we consider heuristics induced by norm embeddings
and distance labeling schemes, and provide lower bounds for the tradeoffs
between the number of dimensions or bits used to represent each graph node, and
the running time of the A* algorithm. We also show that, under natural
assumptions, our lower bounds are almost optimal.

    

### [[2111.10046] YMIR: A Rapid Data-centric Development Platform for Vision Applications](http://arxiv.org/abs/2111.10046)


  This paper introduces an open source platform for rapid development of
computer vision applications. The platform puts the efficient data development
at the center of the machine learning development process, integrates active
learning methods, data and model version control, and uses concepts such as
projects to enable fast iteration of multiple task specific datasets in
parallel. We make it an open platform by abstracting the development process
into core states and operations, and design open APIs to integrate third party
tools as implementations of the operations. This open design reduces the
development cost and adoption cost for ML teams with existing tools. At the
same time, the platform supports recording project development history, through
which successful projects can be shared to further boost model production
efficiency on similar tasks. The platform is open source and is already used
internally to meet the increasing demand from custom real world computer vision
applications.

    

### [[2111.10048] Uniform Brackets, Containers, and Combinatorial Macbeath Regions](http://arxiv.org/abs/2111.10048)


  We study the connections between three seemingly different combinatorial
structures - "uniform" brackets in statistics and probability theory,
"containers" in online and distributed learning theory, and "combinatorial
Macbeath regions", or Mnets in discrete and computational geometry. We show
that these three concepts are manifestations of a single combinatorial property
that can be expressed under a unified framework along the lines of
Vapnik-Chervonenkis type theory for uniform convergence. These new connections
help us to bring tools from discrete and computational geometry to prove
improved bounds for these objects. Our improved bounds help to get an optimal
algorithm for distributed learning of halfspaces, an improved algorithm for the
distributed convex set disjointness problem, and improved regret bounds for
online algorithms against a smoothed adversary for a large class of
semi-algebraic threshold functions.

    

### [[2111.10050] Combined Scaling for Zero-shot Transfer Learning](http://arxiv.org/abs/2111.10050)


  We present a combined scaling method called BASIC that achieves 85.7% top-1
zero-shot accuracy on the ImageNet ILSVRC-2012 validation set, surpassing the
best-published zero-shot models - CLIP and ALIGN - by 9.3%. Our BASIC model
also shows significant improvements in robustness benchmarks. For instance, on
5 test sets with natural distribution shifts such as ImageNet-{A,R,V2,Sketch}
and ObjectNet, our model achieves 83.7% top-1 average accuracy, only a small
drop from the its original ImageNet accuracy.
To achieve these results, we scale up the contrastive learning framework of
CLIP and ALIGN in three dimensions: data size, model size, and batch size. Our
dataset has 6.6B noisy image-text pairs, which is 4x larger than ALIGN, and 16x
larger than CLIP. Our largest model has 3B weights, which is 3.75x larger in
parameters and 8x larger in FLOPs than ALIGN and CLIP. Our batch size is 65536
which is 2x more than CLIP and 4x more than ALIGN. The main challenge with
scaling is the limited memory of our accelerators such as GPUs and TPUs. We
hence propose a simple method of online gradient caching to overcome this
limit.

    

### [[2111.10055] Towards Efficiently Evaluating the Robustness of Deep Neural Networks in IoT Systems: A GAN-based Method](http://arxiv.org/abs/2111.10055)


  Intelligent Internet of Things (IoT) systems based on deep neural networks
(DNNs) have been widely deployed in the real world. However, DNNs are found to
be vulnerable to adversarial examples, which raises people's concerns about
intelligent IoT systems' reliability and security. Testing and evaluating the
robustness of IoT systems becomes necessary and essential. Recently various
attacks and strategies have been proposed, but the efficiency problem remains
unsolved properly. Existing methods are either computationally extensive or
time-consuming, which is not applicable in practice. In this paper, we propose
a novel framework called Attack-Inspired GAN (AI-GAN) to generate adversarial
examples conditionally. Once trained, it can generate adversarial perturbations
efficiently given input images and target classes. We apply AI-GAN on different
datasets in white-box settings, black-box settings and targeted models
protected by state-of-the-art defenses. Through extensive experiments, AI-GAN
achieves high attack success rates, outperforming existing methods, and reduces
generation time significantly. Moreover, for the first time, AI-GAN
successfully scales to complex datasets e.g. CIFAR-100 and ImageNet, with about
$90\%$ success rates among all classes.

    

### [[2111.10058] DeepQR: Neural-based Quality Ratings for Learnersourced Multiple-Choice Questions](http://arxiv.org/abs/2111.10058)


  Automated question quality rating (AQQR) aims to evaluate question quality
through computational means, thereby addressing emerging challenges in online
learnersourced question repositories. Existing methods for AQQR rely solely on
explicitly-defined criteria such as readability and word count, while not fully
utilising the power of state-of-the-art deep-learning techniques. We propose
DeepQR, a novel neural-network model for AQQR that is trained using
multiple-choice-question (MCQ) datasets collected from PeerWise, a widely-used
learnersourcing platform. Along with designing DeepQR, we investigate models
based on explicitly-defined features, or semantic features, or both. We also
introduce a self-attention mechanism to capture semantic correlations between
MCQ components, and a contrastive-learning approach to acquire question
representations using quality ratings. Extensive experiments on datasets
collected from eight university-level courses illustrate that DeepQR has
superior performance over six comparative models.

    

### [[2111.10066] Assessment of Fetal and Maternal Well-Being During Pregnancy Using Passive Wearable Inertial Sensor](http://arxiv.org/abs/2111.10066)


  Assessing the health of both the fetus and mother is vital in preventing and
identifying possible complications in pregnancy. This paper focuses on a device
that can be used effectively by the mother herself with minimal supervision and
provide a reasonable estimation of fetal and maternal health while being safe,
comfortable, and easy to use. The device proposed uses a belt with a single
accelerometer over the mother's uterus to record the required information. The
device is expected to monitor both the mother and the fetus constantly over a
long period and provide medical professionals with useful information, which
they would otherwise overlook due to the low frequency that health monitoring
is carried out at the present. The paper shows that simultaneous measurement of
respiratory information of the mother and fetal movement is in fact possible
even in the presence of mild interferences, which needs to be accounted for if
the device is expected to be worn for extended times.

    

### [[2111.10075] Enhanced countering adversarial attacks via input denoising and feature restoring](http://arxiv.org/abs/2111.10075)


  Despite the fact that deep neural networks (DNNs) have achieved prominent
performance in various applications, it is well known that DNNs are vulnerable
to adversarial examples/samples (AEs) with imperceptible perturbations in
clean/original samples. To overcome the weakness of the existing defense
methods against adversarial attacks, which damages the information on the
original samples, leading to the decrease of the target classifier accuracy,
this paper presents an enhanced countering adversarial attack method IDFR (via
Input Denoising and Feature Restoring). The proposed IDFR is made up of an
enhanced input denoiser (ID) and a hidden lossy feature restorer (FR) based on
the convex hull optimization. Extensive experiments conducted on benchmark
datasets show that the proposed IDFR outperforms the various state-of-the-art
defense methods, and is highly effective for protecting target models against
various adversarial black-box or white-box attacks. \footnote{Souce code is
released at:
\href{this https URL}{this https URL}}

    

### [[2111.10078] Defeating Catastrophic Forgetting via Enhanced Orthogonal Weights Modification](http://arxiv.org/abs/2111.10078)


  The ability of neural networks (NNs) to learn and remember multiple tasks
sequentially is facing tough challenges in achieving general artificial
intelligence due to their catastrophic forgetting (CF) issues. Fortunately, the
latest OWM Orthogonal Weights Modification) and other several continual
learning (CL) methods suggest some promising ways to overcome the CF issue.
However, none of existing CL methods explores the following three crucial
questions for effectively overcoming the CF issue: that is, what knowledge does
it contribute to the effective weights modification of the NN during its
sequential tasks learning? When the data distribution of a new learning task
changes corresponding to the previous learned tasks, should a uniform/specific
weight modification strategy be adopted or not? what is the upper bound of the
learningable tasks sequentially for a given CL method? ect. To achieve this, in
this paper, we first reveals the fact that of the weight gradient of a new
learning task is determined by both the input space of the new task and the
weight space of the previous learned tasks sequentially. On this observation
and the recursive least square optimal method, we propose a new efficient and
effective continual learning method EOWM via enhanced OWM. And we have
theoretically and definitively given the upper bound of the learningable tasks
sequentially of our EOWM. Extensive experiments conducted on the benchmarks
demonstrate that our EOWM is effectiveness and outperform all of the
state-of-the-art CL baselines.

    

### [[2111.10079] Evaluating Self and Semi-Supervised Methods for Remote Sensing Segmentation Tasks](http://arxiv.org/abs/2111.10079)


  We perform a rigorous evaluation of recent self and semi-supervised ML
techniques that leverage unlabeled data for improving downstream task
performance, on three remote sensing tasks of riverbed segmentation, land cover
mapping and flood mapping. These methods are especially valuable for remote
sensing tasks since there is easy access to unlabeled imagery and getting
ground truth labels can often be expensive. We quantify performance
improvements one can expect on these remote sensing segmentation tasks when
unlabeled imagery (outside of the labeled dataset) is made available for
training. We also design experiments to test the effectiveness of these
techniques when the test set has a domain shift relative to the training and
validation sets.

    

### [[2111.10085] Exposing Weaknesses of Malware Detectors with Explainability-Guided Evasion Attacks](http://arxiv.org/abs/2111.10085)


  Numerous open-source and commercial malware detectors are available. However,
the efficacy of these tools has been threatened by new adversarial attacks,
whereby malware attempts to evade detection using, for example, machine
learning techniques. In this work, we design an adversarial evasion attack that
relies on both feature-space and problem-space manipulation. It uses
explainability-guided feature selection to maximize evasion by identifying the
most critical features that impact detection. We then use this attack as a
benchmark to evaluate several state-of-the-art malware detectors. We find that
(i) state-of-the-art malware detectors are vulnerable to even simple evasion
strategies, and they can easily be tricked using off-the-shelf techniques; (ii)
feature-space manipulation and problem-space obfuscation can be combined to
enable evasion without needing white-box understanding of the detector; (iii)
we can use explainability approaches (e.g., SHAP) to guide the feature
manipulation and explain how attacks can transfer across multiple detectors.
Our findings shed light on the weaknesses of current malware detectors, as well
as how they can be improved.

    

### [[2111.10088] Data imputation and comparison of custom ensemble models with existing libraries like XGBoost, Scikit learn, etc. for Predictive Equipment failure](http://arxiv.org/abs/2111.10088)


  This paper presents comparison of custom ensemble models with the models
trained using existing libraries Like Xgboost, Scikit Learn, etc. in case of
predictive equipment failure for the case of oil extracting equipment setup.
The dataset that is used contains many missing values and the paper proposes
different model-based data imputation strategies to impute the missing values.
The architecture and the training and testing process of the custom ensemble
models are explained in detail.

    

### [[2111.10102] Graph Neural Networks with Feature and Structure Aware Random Walk](http://arxiv.org/abs/2111.10102)


  Graph Neural Networks (GNNs) have received increasing attention for
representation learning in various machine learning tasks. However, most
existing GNNs applying neighborhood aggregation usually perform poorly on the
graph with heterophily where adjacent nodes belong to different classes. In
this paper, we show that in typical heterphilous graphs, the edges may be
directed, and whether to treat the edges as is or simply make them undirected
greatly affects the performance of the GNN models. Furthermore, due to the
limitation of heterophily, it is highly beneficial for the nodes to aggregate
messages from similar nodes beyond local neighborhood.These motivate us to
develop a model that adaptively learns the directionality of the graph, and
exploits the underlying long-distance correlations between nodes. We first
generalize the graph Laplacian to digraph based on the proposed Feature-Aware
PageRank algorithm, which simultaneously considers the graph directionality and
long-distance feature similarity between nodes. Then digraph Laplacian defines
a graph propagation matrix that leads to a model called {\em DiglacianGCN}.
Based on this, we further leverage the node proximity measured by commute times
between nodes, in order to preserve the nodes' long-distance correlation on the
topology level. Extensive experiments on ten datasets with different levels of
homophily demonstrate the effectiveness of our method over existing solutions
in the task of node classification.

    

### [[2111.10103] Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning](http://arxiv.org/abs/2111.10103)


  Value estimation is one key problem in Reinforcement Learning. Albeit many
successes have been achieved by Deep Reinforcement Learning (DRL) in different
fields, the underlying structure and learning dynamics of value function,
especially with complex function approximation, are not fully understood. In
this paper, we report that decreasing rank of $Q$-matrix widely exists during
learning process across a series of continuous control tasks for different
popular algorithms. We hypothesize that the low-rank phenomenon indicates the
common learning dynamics of $Q$-matrix from stochastic high dimensional space
to smooth low dimensional space. Moreover, we reveal a positive correlation
between value matrix rank and value estimation uncertainty. Inspired by above
evidence, we propose a novel Uncertainty-Aware Low-rank Q-matrix Estimation
(UA-LQE) algorithm as a general framework to facilitate the learning of value
function. Through quantifying the uncertainty of state-action value estimation,
we selectively erase the entries of highly uncertain values in state-action
value matrix and conduct low-rank matrix reconstruction for them to recover
their values. Such a reconstruction exploits the underlying structure of value
matrix to improve the value approximation, thus leading to a more efficient
learning process of value function. In the experiments, we evaluate the
efficacy of UA-LQE in several representative OpenAI MuJoCo continuous control
tasks.

    

### [[2111.10106] A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling](http://arxiv.org/abs/2111.10106)


  Individual Treatment Effect (ITE) prediction is an important area of research
in machine learning which aims at explaining and estimating the causal impact
of an action at the granular level. It represents a problem of growing interest
in multiple sectors of application such as healthcare, online advertising or
socioeconomics. To foster research on this topic we release a publicly
available collection of 13.9 million samples collected from several randomized
control trials, scaling up previously available datasets by a healthy 210x
factor. We provide details on the data collection and perform sanity checks to
validate the use of this data for causal inference tasks. First, we formalize
the task of uplift modeling (UM) that can be performed with this data, along
with the relevant evaluation metrics. Then, we propose synthetic response
surfaces and heterogeneous treatment assignment providing a general set-up for
ITE prediction. Finally, we report experiments to validate key characteristics
of the dataset leveraging its size to evaluate and compare - with high
statistical significance - a selection of baseline UM and ITE prediction
methods.

    

### [[2111.10130] Fooling Adversarial Training with Inducing Noise](http://arxiv.org/abs/2111.10130)


  Adversarial training is widely believed to be a reliable approach to improve
model robustness against adversarial attack. However, in this paper, we show
that when trained on one type of poisoned data, adversarial training can also
be fooled to have catastrophic behavior, e.g., $<1\%$ robust test accuracy with
$>90\%$ robust training accuracy on CIFAR-10 dataset. Previously, there are
other types of noise poisoned in the training data that have successfully
fooled standard training ($15.8\%$ standard test accuracy with $99.9\%$
standard training accuracy on CIFAR-10 dataset), but their poisonings can be
easily removed when adopting adversarial training. Therefore, we aim to design
a new type of inducing noise, named ADVIN, which is an irremovable poisoning of
training data. ADVIN can not only degrade the robustness of adversarial
training by a large margin, for example, from $51.7\%$ to $0.57\%$ on CIFAR-10
dataset, but also be effective for fooling standard training ($13.1\%$ standard
test accuracy with $100\%$ standard training accuracy). Additionally, ADVIN can
be applied to preventing personal data (like selfies) from being exploited
without authorization under whether standard or adversarial training.

    

### [[2111.10135] Grounded Situation Recognition with Transformers](http://arxiv.org/abs/2111.10135)


  Grounded Situation Recognition (GSR) is the task that not only classifies a
salient action (verb), but also predicts entities (nouns) associated with
semantic roles and their locations in the given image. Inspired by the
remarkable success of Transformers in vision tasks, we propose a GSR model
based on a Transformer encoder-decoder architecture. The attention mechanism of
our model enables accurate verb classification by capturing high-level semantic
feature of an image effectively, and allows the model to flexibly deal with the
complicated and image-dependent relations between entities for improved noun
classification and localization. Our model is the first Transformer
architecture for GSR, and achieves the state of the art in every evaluation
metric on the SWiG benchmark. Our code is available at
this https URL .

    

### [[2111.10140] Learning in High-Dimensional Feature Spaces Using ANOVA-Based Fast Matrix-Vector Multiplication](http://arxiv.org/abs/2111.10140)


  Kernel matrices are crucial in many learning tasks such as support vector
machines or kernel ridge regression. The kernel matrix is typically dense and
large-scale. Depending on the dimension of the feature space even the
computation of all of its entries in reasonable time becomes a challenging
task. For such dense matrices the cost of a matrix-vector product scales
quadratically in the number of entries, if no customized methods are applied.
We propose the use of an ANOVA kernel, where we construct several kernels based
on lower-dimensional feature spaces for which we provide fast algorithms
realizing the matrix-vector products. We employ the non-equispaced fast Fourier
transform (NFFT), which is of linear complexity for fixed accuracy. Based on a
feature grouping approach, we then show how the fast matrix-vector products can
be embedded into a learning method choosing kernel ridge regression and the
preconditioned conjugate gradient solver. We illustrate the performance of our
approach on several data sets.

    

### [[2111.10144] Positional Encoder Graph Neural Networks for Geographic Data](http://arxiv.org/abs/2111.10144)


  Graph neural networks (GNNs) provide a powerful and scalable solution for
modeling continuous spatial data. However, in the absence of further context on
the geometric structure of the data, they often rely on Euclidean distances to
construct the input graphs. This assumption can be improbable in many
real-world settings, where the spatial structure is more complex and explicitly
non-Euclidean (e.g., road networks). In this paper, we propose PE-GNN, a new
framework that incorporates spatial context and correlation explicitly into the
models. Building on recent advances in geospatial auxiliary task learning and
semantic spatial embeddings, our proposed method (1) learns a context-aware
vector encoding of the geographic coordinates and (2) predicts spatial
autocorrelation in the data in parallel with the main task. On spatial
regression tasks, we show the effectiveness of our approach, improving
performance over different state-of-the-art GNN approaches. We also test our
approach for spatial interpolation, i.e., spatial regression without node
features, a task that GNNs are currently not competitive at. We observe that
our approach not only vastly improves over the GNN baselines, but can match
Gaussian processes, the most commonly utilized method for spatial interpolation
problems.

    

### [[2111.10168] Improved Prosodic Clustering for Multispeaker and Speaker-independent Phoneme-level Prosody Control](http://arxiv.org/abs/2111.10168)


  This paper presents a method for phoneme-level prosody control of F0 and
duration on a multispeaker text-to-speech setup, which is based on prosodic
clustering. An autoregressive attention-based model is used, incorporating
multispeaker architecture modules in parallel to a prosody encoder. Several
improvements over the basic single-speaker method are proposed that increase
the prosodic control range and coverage. More specifically we employ data
augmentation, F0 normalization, balanced clustering for duration, and
speaker-independent prosodic clustering. These modifications enable
fine-grained phoneme-level prosody control for all speakers contained in the
training set, while maintaining the speaker identity. The model is also
fine-tuned to unseen speakers with limited amounts of data and it is shown to
maintain its prosody control capabilities, verifying that the
speaker-independent prosodic clustering is effective. Experimental results
verify that the model maintains high output speech quality and that the
proposed method allows efficient prosody control within each speaker's range
despite the variability that a multispeaker setting introduces.

    

### [[2111.10173] Word-Level Style Control for Expressive, Non-attentive Speech Synthesis](http://arxiv.org/abs/2111.10173)


  This paper presents an expressive speech synthesis architecture for modeling
and controlling the speaking style at a word level. It attempts to learn
word-level stylistic and prosodic representations of the speech data, with the
aid of two encoders. The first one models style by finding a combination of
style tokens for each word given the acoustic features, and the second outputs
a word-level sequence conditioned only on the phonetic information in order to
disentangle it from the style information. The two encoder outputs are aligned
and concatenated with the phoneme encoder outputs and then decoded with a
Non-Attentive Tacotron model. An extra prior encoder is used to predict the
style tokens autoregressively, in order for the model to be able to run without
a reference utterance. We find that the resulting model gives both word-level
and global control over the style, as well as prosody transfer capabilities.

    

### [[2111.10175] Randomized Algorithms for Monotone Submodular Function Maximization on the Integer Lattice](http://arxiv.org/abs/2111.10175)


  Optimization problems with set submodular objective functions have many
real-world applications. In discrete scenarios, where the same item can be
selected more than once, the domain is generalized from a 2-element set to a
bounded integer lattice. In this work, we consider the problem of maximizing a
monotone submodular function on the bounded integer lattice subject to a
cardinality constraint. In particular, we focus on maximizing DR-submodular
functions, i.e., functions defined on the integer lattice that exhibit the
diminishing returns property. Given any epsilon > 0, we present a randomized
algorithm with probabilistic guarantees of O(1 - 1/e - epsilon) approximation,
using a framework inspired by a Stochastic Greedy algorithm developed for set
submodular functions by Mirzasoleiman et al. We then show that, on synthetic
DR-submodular functions, applying our proposed algorithm on the integer lattice
is faster than the alternatives, including reducing a target problem to the set
domain and then applying the fastest known set submodular maximization
algorithm.

    

### [[2111.10177] Prosodic Clustering for Phoneme-level Prosody Control in End-to-End Speech Synthesis](http://arxiv.org/abs/2111.10177)


  This paper presents a method for controlling the prosody at the phoneme level
in an autoregressive attention-based text-to-speech system. Instead of learning
latent prosodic features with a variational framework as is commonly done, we
directly extract phoneme-level F0 and duration features from the speech data in
the training set. Each prosodic feature is discretized using unsupervised
clustering in order to produce a sequence of prosodic labels for each
utterance. This sequence is used in parallel to the phoneme sequence in order
to condition the decoder with the utilization of a prosodic encoder and a
corresponding attention module. Experimental results show that the proposed
method retains the high quality of generated speech, while allowing
phoneme-level control of F0 and duration. By replacing the F0 cluster centroids
with musical notes, the model can also provide control over the note and octave
within the range of the speaker.

    

### [[2111.10178] Understanding Training-Data Leakage from Gradients in Neural Networks for Image Classification](http://arxiv.org/abs/2111.10178)


  Federated learning of deep learning models for supervised tasks, e.g. image
classification and segmentation, has found many applications: for example in
human-in-the-loop tasks such as film post-production where it enables sharing
of domain expertise of human artists in an efficient and effective fashion. In
many such applications, we need to protect the training data from being leaked
when gradients are shared in the training process due to IP or privacy
concerns. Recent works have demonstrated that it is possible to reconstruct the
training data from gradients for an image-classification model when its
architecture is known. However, there is still an incomplete theoretical
understanding of the efficacy and failure of such attacks. In this paper, we
analyse the source of training-data leakage from gradients. We formulate the
problem of training data reconstruction as solving an optimisation problem
iteratively for each layer. The layer-wise objective function is primarily
defined by weights and gradients from the current layer as well as the output
from the reconstruction of the subsequent layer, but it might also involve a
'pull-back' constraint from the preceding layer. Training data can be
reconstructed when we solve the problem backward from the output of the network
through each layer. Based on this formulation, we are able to attribute the
potential leakage of the training data in a deep network to its architecture.
We also propose a metric to measure the level of security of a deep learning
model against gradient-based attacks on the training data.

    

### [[2111.10183] Benchmarking Small-Scale Quantum Devices on Computing Graph Edit Distance](http://arxiv.org/abs/2111.10183)


  Distance measures provide the foundation for many popular algorithms in
Machine Learning and Pattern Recognition. Different notions of distance can be
used depending on the types of the data the algorithm is working on. For
graph-shaped data, an important notion is the Graph Edit Distance (GED) that
measures the degree of (dis)similarity between two graphs in terms of the
operations needed to make them identical. As the complexity of computing GED is
the same as NP-hard problems, it is reasonable to consider approximate
solutions. In this paper we present a comparative study of two quantum
approaches to computing GED: quantum annealing and variational quantum
algorithms, which refer to the two types of quantum hardware currently
available, namely quantum annealer and gate-based quantum computer,
respectively. Considering the current state of noisy intermediate-scale quantum
computers, we base our study on proof-of-principle tests of the performance of
these quantum algorithms.

    

### [[2111.10189] Analysis of autocorrelation times in Neural Markov Chain Monte Carlo simulations](http://arxiv.org/abs/2111.10189)


  We provide a deepened study of autocorrelations in Neural Markov Chain Monte
Carlo simulations, a version of the traditional Metropolis algorithm which
employs neural networks to provide independent proposals. We illustrate our
ideas using the two-dimensional Ising model. We propose several estimates of
autocorrelation times, some inspired by analytical results derived for the
Metropolized Independent Sampler, which we compare and study as a function of
inverse temperature $\beta$. Based on that we propose an alternative loss
function and study its impact on the autocorelation times. Furthermore, we
investigate the impact of imposing system symmetries ($Z_2$ and/or
translational) in the neural network training process on the autocorrelation
times. Eventually, we propose a scheme which incorporates partial heat-bath
updates. The impact of the above enhancements is discussed for a $16 \times 16$
spin system. The summary of our findings may serve as a guide to the
implementation of Neural Markov Chain Monte Carlo simulations of more
complicated models.

    

### [[2111.10192] An Expectation-Maximization Perspective on Federated Learning](http://arxiv.org/abs/2111.10192)


  Federated learning describes the distributed training of models across
multiple clients while keeping the data private on-device. In this work, we
view the server-orchestrated federated learning process as a hierarchical
latent variable model where the server provides the parameters of a prior
distribution over the client-specific model parameters. We show that with
simple Gaussian priors and a hard version of the well known
Expectation-Maximization (EM) algorithm, learning in such a model corresponds
to FedAvg, the most popular algorithm for the federated learning setting. This
perspective on FedAvg unifies several recent works in the field and opens up
the possibility for extensions through different choices for the hierarchical
model. Based on this view, we further propose a variant of the hierarchical
model that employs prior distributions to promote sparsity. By similarly using
the hard-EM algorithm for learning, we obtain FedSparse, a procedure that can
learn sparse neural networks in the federated learning setting. FedSparse
reduces communication costs from client to server and vice-versa, as well as
the computational costs for inference with the sparsified network - both of
which are of great practical importance in federated learning.

    

### [[2111.10196] Towards Traffic Scene Description: The Semantic Scene Graph](http://arxiv.org/abs/2111.10196)


  For the classification of traffic scenes, a description model is necessary
that can describe the scene in a uniform way, independent of its domain. A
model to describe a traffic scene in a semantic way is described in this paper.
The description model allows to describe a traffic scene independently of the
road geometry and road topology. Here, the traffic participants are projected
onto the road network and represented as nodes in a graph. Depending on the
relative location between two traffic participants with respect to the road
topology, semantic classified edges are created between the corresponding
nodes. For concretization, the edge attributes are extended by relative
distances and velocities between both traffic participants with regard to the
course of the lane. An important aspect of the description is that it can be
converted easily into a machine-readable format. The current description
focuses on dynamic objects of a traffic scene and considers traffic
participants, such as pedestrians or vehicles.

    

### [[2111.10204] Augmentation of base classifier performance via HMMs on a handwritten character data set](http://arxiv.org/abs/2111.10204)


  This paper presents results of a study of the performance of several base
classifiers for recognition of handwritten characters of the modern Latin
alphabet. Base classification performance is further enhanced by utilizing
Viterbi error correction by determining the Viterbi sequence. Hidden Markov
Models (HMMs) models exploit relationships between letters within a word to
determine the most likely sequence of characters. Four base classifiers are
studied along with eight feature sets extracted from the handwritten dataset.
The best classification performance after correction was 89.8%, and the average
was 68.1%

    

### [[2111.10206] Subspace Graph Physics: Real-Time Rigid Body-Driven Granular Flow Simulation](http://arxiv.org/abs/2111.10206)


  An important challenge in robotics is understanding the interactions between
robots and deformable terrains that consist of granular material. Granular
flows and their interactions with rigid bodies still pose several open
questions. A promising direction for accurate, yet efficient, modeling is using
continuum methods. Also, a new direction for real-time physics modeling is the
use of deep learning. This research advances machine learning methods for
modeling rigid body-driven granular flows, for application to terrestrial
industrial machines as well as space robotics (where the effect of gravity is
an important factor). In particular, this research considers the development of
a subspace machine learning simulation approach. To generate training datasets,
we utilize our high-fidelity continuum method, material point method (MPM).
Principal component analysis (PCA) is used to reduce the dimensionality of
data. We show that the first few principal components of our high-dimensional
data keep almost the entire variance in data. A graph network simulator (GNS)
is trained to learn the underlying subspace dynamics. The learned GNS is then
able to predict particle positions and interaction forces with good accuracy.
More importantly, PCA significantly enhances the time and memory efficiency of
GNS in both training and rollout. This enables GNS to be trained using a single
desktop GPU with moderate VRAM. This also makes the GNS real-time on
large-scale 3D physics configurations (700x faster than our continuum method).

    

### [[2111.10207] Comparative Study of Speech Analysis Methods to Predict Parkinson's Disease](http://arxiv.org/abs/2111.10207)


  One of the symptoms observed in the early stages of Parkinson's Disease (PD)
is speech impairment. Speech disorders can be used to detect this disease
before it degenerates. This work analyzes speech features and machine learning
approaches to predict PD. Acoustic features such as shimmer and jitter
variants, and Mel Frequency Cepstral Coefficients (MFCC) are extracted from
speech signals. We use two datasets in this work: the MDVR-KCL and the Italian
Parkinson's Voice and Speech database. To separate PD and non-PD speech
signals, seven classification models were implemented: K-Nearest Neighbor,
Decision Trees, Support Vector Machines, Naive Bayes, Logistic Regression,
Gradient Boosting, Random Forests. Three feature sets were used for each of the
models: (a) Acoustic features only, (b) All the acoustic features and MFCC, (c)
Selected subset of features from acoustic features and MFCC. Using all the
acoustic features and MFCC, together with SVM produced the highest performance
with an accuracy of 98% and F1-Score of 99%. When compared with prior art, this
shows a better performance. Our code and related documentation is available in
a public domain repository.

    

### [[2111.10208] Attention based end to end Speech Recognition for Voice Search in Hindi and English](http://arxiv.org/abs/2111.10208)


  We describe here our work with automatic speech recognition (ASR) in the
context of voice search functionality on the Flipkart e-Commerce platform.
Starting with the deep learning architecture of Listen-Attend-Spell (LAS), we
build upon and expand the model design and attention mechanisms to incorporate
innovative approaches including multi-objective training, multi-pass training,
and external rescoring using language models and phoneme based losses. We
report a relative WER improvement of 15.7% on top of state-of-the-art LAS
models using these modifications. Overall, we report an improvement of 36.9%
over the phoneme-CTC system. The paper also provides an overview of different
components that can be tuned in a LAS-based system.

    

### [[2111.10210] The Application of Zig-Zag Sampler in Sequential Markov Chain Monte Carlo](http://arxiv.org/abs/2111.10210)


  Particle filtering methods are widely applied in sequential state estimation
within nonlinear non-Gaussian state space model. However, the traditional
particle filtering methods suffer the weight degeneracy in the high-dimensional
state space model. Currently, there are many methods to improve the performance
of particle filtering in high-dimensional state space model. Among these, the
more advanced method is to construct the Sequential Makov chian Monte Carlo
(SMCMC) framework by implementing the Composite Metropolis-Hasting (MH) Kernel.
In this paper, we proposed to discrete the Zig-Zag Sampler and apply the
Zig-Zag Sampler in the refinement stage of the Composite MH Kernel within the
SMCMC framework which is implemented the invertible particle flow in the joint
draw stage. We evaluate the performance of proposed method through numerical
experiments of the challenging complex high-dimensional filtering examples.
Nemurical experiments show that in high-dimensional state estimation examples,
the proposed method improves estimation accuracy and increases the acceptance
ratio compared with state-of-the-art filtering methods.

    

### [[2111.10227] Policy Gradient Approach to Compilation of Variational Quantum Circuits](http://arxiv.org/abs/2111.10227)


  We propose a method for finding approximate compilations of quantum circuits,
based on techniques from policy gradient reinforcement learning. The choice of
a stochastic policy allows us to rephrase the optimization problem in terms of
probability distributions, rather than variational parameters. This implies
that searching for the optimal configuration is done by optimizing over the
distribution parameters, rather than over the circuit free angles. The upshot
of this is that we can always compute a gradient, provided that the policy is
differentiable. We show numerically that this approach is more competitive than
those using gradient-free methods, even in the presence of depolarizing noise,
and argue analytically why this is the case. Another interesting feature of
this approach to variational compilation is that it does not need a separate
register and long-range interactions to estimate the end-point fidelity. We
expect these techniques to be relevant for training variational circuit in
other contexts

    

### [[2111.10233] Xp-GAN: Unsupervised Multi-object Controllable Video Generation](http://arxiv.org/abs/2111.10233)


  Video Generation is a relatively new and yet popular subject in machine
learning due to its vast variety of potential applications and its numerous
challenges. Current methods in Video Generation provide the user with little or
no control over the exact specification of how the objects in the generate
video are to be moved and located at each frame, that is, the user can't
explicitly control how each object in the video should move. In this paper we
propose a novel method that allows the user to move any number of objects of a
single initial frame just by drawing bounding boxes over those objects and then
moving those boxes in the desired path. Our model utilizes two Autoencoders to
fully decompose the motion and content information in a video and achieves
results comparable to well-known baseline and state of the art methods.

    

### [[2111.10235] Interpreting deep urban sound classification using Layer-wise Relevance Propagation](http://arxiv.org/abs/2111.10235)


  After constructing a deep neural network for urban sound classification, this
work focuses on the sensitive application of assisting drivers suffering from
hearing loss. As such, clear etiology justifying and interpreting model
predictions comprise a strong requirement. To this end, we used two different
representations of audio signals, i.e. Mel and constant-Q spectrograms, while
the decisions made by the deep neural network are explained via layer-wise
relevance propagation. At the same time, frequency content assigned with high
relevance in both feature sets, indicates extremely discriminative information
characterizing the present classification task. Overall, we present an
explainable AI framework for understanding deep urban sound classification.

    

### [[2111.10243] Posterior concentration and fast convergence rates for generalized Bayesian learning](http://arxiv.org/abs/2111.10243)


  In this paper, we study the learning rate of generalized Bayes estimators in
a general setting where the hypothesis class can be uncountable and have an
irregular shape, the loss function can have heavy tails, and the optimal
hypothesis may not be unique. We prove that under the multi-scale Bernstein's
condition, the generalized posterior distribution concentrates around the set
of optimal hypotheses and the generalized Bayes estimator can achieve fast
learning rate. Our results are applied to show that the standard Bayesian
linear regression is robust to heavy-tailed distributions.

    

### [[2111.10245] Ubi-SleepNet: Advanced Multimodal Fusion Techniques for Three-stage Sleep Classification Using Ubiquitous Sensing](http://arxiv.org/abs/2111.10245)


  Sleep is a fundamental physiological process that is essential for sustaining
a healthy body and mind. The gold standard for clinical sleep monitoring is
polysomnography(PSG), based on which sleep can be categorized into five stages,
including wake/rapid eye movement sleep (REM sleep)/Non-REM sleep 1
(N1)/Non-REM sleep 2 (N2)/Non-REM sleep 3 (N3). However, PSG is expensive,
burdensome, and not suitable for daily use. For long-term sleep monitoring,
ubiquitous sensing may be a solution. Most recently, cardiac and movement
sensing has become popular in classifying three-stage sleep, since both
modalities can be easily acquired from research-grade or consumer-grade devices
(e.g., Apple Watch). However, how best to fuse the data for the greatest
accuracy remains an open question. In this work, we comprehensively studied
deep learning (DL)-based advanced fusion techniques consisting of three fusion
strategies alongside three fusion methods for three-stage sleep classification
based on two publicly available datasets. Experimental results demonstrate
important evidence that three-stage sleep can be reliably classified by fusing
cardiac/movement sensing modalities, which may potentially become a practical
tool to conduct large-scale sleep stage assessment studies or long-term
self-tracking on sleep. To accelerate the progression of sleep research in the
ubiquitous/wearable computing community, we made this project open source, and
the code can be found at: this https URL.

    

### [[2111.10247] Fast and Data-Efficient Training of Rainbow: an Experimental Study on Atari](http://arxiv.org/abs/2111.10247)


  Across the Arcade Learning Environment, Rainbow achieves a level of
performance competitive with humans and modern RL algorithms. However,
attaining this level of performance requires large amounts of data and hardware
resources, making research in this area computationally expensive and use in
practical applications often infeasible. This paper's contribution is
threefold: We (1) propose an improved version of Rainbow, seeking to
drastically reduce Rainbow's data, training time, and compute requirements
while maintaining its competitive performance; (2) we empirically demonstrate
the effectiveness of our approach through experiments on the Arcade Learning
Environment, and (3) we conduct a number of ablation studies to investigate the
effect of the individual proposed modifications. Our improved version of
Rainbow reaches a median human normalized score close to classic Rainbow's,
while using 20 times less data and requiring only 7.5 hours of training time on
a single GPU. We also provide our full implementation including pre-trained
models.

    

### [[2111.10248] Non asymptotic bounds in asynchronous sum-weight gossip protocols](http://arxiv.org/abs/2111.10248)


  This paper focuses on non-asymptotic diffusion time in asynchronous gossip
protocols. Asynchronous gossip protocols are designed to perform distributed
computation in a network of nodes by randomly exchanging messages on the
associated graph. To achieve consensus among nodes, a minimal number of
messages has to be exchanged. We provides a probabilistic bound to such number
for the general case. We provide a explicit formula for fully connected graphs
depending only on the number of nodes and an approximation for any graph
depending on the spectrum of the graph.

    

### [[2111.10262] Residual fourier neural operator for thermochemical curing of composites](http://arxiv.org/abs/2111.10262)


  During the curing process of composites, the temperature history heavily
determines the evolutions of the field of degree of cure as well as the
residual stress, which will further influence the mechanical properties of
composite, thus it is important to simulate the real temperature history to
optimize the curing process of composites. Since thermochemical analysis using
Finite Element (FE) simulations requires heavy computational loads and
data-driven approaches suffer from the complexity of highdimensional mapping.
This paper proposes a Residual Fourier Neural Operator (ResFNO) to establish
the direct high-dimensional mapping from any given cure cycle to the
corresponding temperature histories. By integrating domain knowledge into a
time-resolution independent parameterized neural network, the mapping between
cure cycles to temperature histories can be learned using limited number of
labelled data. Besides, a novel Fourier residual mapping is designed based on
mode decomposition to accelerate the training and boost the performance
significantly. Several cases are carried out to evaluate the superior
performance and generalizability of the proposed method comprehensively.

    

### [[2111.10265] ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation](http://arxiv.org/abs/2111.10265)


  There has been a recent surge in methods that aim to decompose and segment
scenes into multiple objects in an unsupervised manner, i.e., unsupervised
multi-object segmentation. Performing such a task is a long-standing goal of
computer vision, offering to unlock object-level reasoning without requiring
dense annotations to train segmentation models. Despite significant progress,
current models are developed and trained on visually simple scenes depicting
mono-colored objects on plain backgrounds. The natural world, however, is
visually complex with confounding aspects such as diverse textures and
complicated lighting effects. In this study, we present a new benchmark called
ClevrTex, designed as the next challenge to compare, evaluate and analyze
algorithms. ClevrTex features synthetic scenes with diverse shapes, textures
and photo-mapped materials, created using physically based rendering
techniques. It includes 50k examples depicting 3-10 objects arranged on a
background, created using a catalog of 60 materials, and a further test set
featuring 10k images created using 25 different materials. We benchmark a large
set of recent unsupervised multi-object segmentation models on ClevrTex and
find all state-of-the-art approaches fail to learn good representations in the
textured setting, despite impressive performance on simpler data. We also
create variants of the ClevrTex dataset, controlling for different aspects of
scene complexity, and probe current approaches for individual shortcomings.
Dataset and code are available at
this https URL.

    

### [[2111.10267] Over-the-Air Federated Learning with Retransmissions (Extended Version)](http://arxiv.org/abs/2111.10267)


  Motivated by increasing computational capabilities of wireless devices, as
well as unprecedented levels of user- and device-generated data, new
distributed machine learning (ML) methods have emerged. In the wireless
community, Federated Learning (FL) is of particular interest due to its
communication efficiency and its ability to deal with the problem of non-IID
data. FL training can be accelerated by a wireless communication method called
Over-the-Air Computation (AirComp) which harnesses the interference of
simultaneous uplink transmissions to efficiently aggregate model updates.
However, since AirComp utilizes analog communication, it introduces inevitable
estimation errors. In this paper, we study the impact of such estimation errors
on the convergence of FL and propose retransmissions as a method to improve FL
convergence over resource-constrained wireless networks. First, we derive the
optimal AirComp power control scheme with retransmissions over static channels.
Then, we investigate the performance of Over-the-Air FL with retransmissions
and find two upper bounds on the FL loss function. Finally, we propose a
heuristic for selecting the optimal number of retransmissions, which can be
calculated before training the ML model. Numerical results demonstrate that the
introduction of retransmissions can lead to improved ML performance, without
incurring extra costs in terms of communication or computation. Additionally,
we provide simulation results on our heuristic which indicate that it can
correctly identify the optimal number of retransmissions for different wireless
network setups and machine learning problems.

    

### [[2111.10269] Pointer over Attention: An Improved Bangla Text Summarization Approach Using Hybrid Pointer Generator Network](http://arxiv.org/abs/2111.10269)


  Despite the success of the neural sequence-to-sequence model for abstractive
text summarization, it has a few shortcomings, such as repeating inaccurate
factual details and tending to repeat themselves. We propose a hybrid pointer
generator network to solve the shortcomings of reproducing factual details
inadequately and phrase repetition. We augment the attention-based
sequence-to-sequence using a hybrid pointer generator network that can generate
Out-of-Vocabulary words and enhance accuracy in reproducing authentic details
and a coverage mechanism that discourages repetition. It produces a
reasonable-sized output text that preserves the conceptual integrity and
factual information of the input article. For evaluation, we primarily employed
"BANSData" - a highly adopted publicly available Bengali dataset. Additionally,
we prepared a large-scale dataset called "BANS-133" which consists of 133k
Bangla news articles associated with human-generated summaries. Experimenting
with the proposed model, we achieved ROUGE-1 and ROUGE-2 scores of 0.66, 0.41
for the "BANSData" dataset and 0.67, 0.42 for the BANS-133k" dataset,
respectively. We demonstrated that the proposed system surpasses previous
state-of-the-art Bengali abstractive summarization techniques and its stability
on a larger dataset. "BANS-133" datasets and code-base will be publicly
available for research.

    

### [[2111.10272] Resilience from Diversity: Population-based approach to harden models against adversarial attacks](http://arxiv.org/abs/2111.10272)


  Traditional deep learning models exhibit intriguing vulnerabilities that
allow an attacker to force them to fail at their task. Notorious attacks such
as the Fast Gradient Sign Method (FGSM) and the more powerful Projected
Gradient Descent (PGD) generate adversarial examples by adding a magnitude of
perturbation $\epsilon$ to the input's computed gradient, resulting in a
deterioration of the effectiveness of the model's classification. This work
introduces a model that is resilient to adversarial attacks. Our model
leverages a well established principle from biological sciences: population
diversity produces resilience against environmental changes. More precisely,
our model consists of a population of $n$ diverse submodels, each one of them
trained to individually obtain a high accuracy for the task at hand, while
forced to maintain meaningful differences in their weight tensors. Each time
our model receives a classification query, it selects a submodel from its
population at random to answer the query. To introduce and maintain diversity
in population of submodels, we introduce the concept of counter linking
weights. A Counter-Linked Model (CLM) consists of submodels of the same
architecture where a periodic random similarity examination is conducted during
the simultaneous training to guarantee diversity while maintaining accuracy. In
our testing, CLM robustness got enhanced by around 20% when tested on the MNIST
dataset and at least 15% when tested on the CIFAR-10 dataset. When implemented
with adversarially trained submodels, this methodology achieves
state-of-the-art robustness. On the MNIST dataset with $\epsilon=0.3$, it
achieved 94.34% against FGSM and 91% against PGD. On the CIFAR-10 dataset with
$\epsilon=8/255$, it achieved 62.97% against FGSM and 59.16% against PGD.

    

### [[2111.10275] Composite Goodness-of-fit Tests with Kernels](http://arxiv.org/abs/2111.10275)


  Model misspecification can create significant challenges for the
implementation of probabilistic models, and this has led to development of a
range of inference methods which directly account for this issue. However,
whether these more involved methods are required will depend on whether the
model is really misspecified, and there is a lack of generally applicable
methods to answer this question. One set of tools which can help are
goodness-of-fit tests, where we test whether a dataset could have been
generated by a fixed distribution. Kernel-based tests have been developed to
for this problem, and these are popular due to their flexibility, strong
theoretical guarantees and ease of implementation in a wide range of scenarios.
In this paper, we extend this line of work to the more challenging composite
goodness-of-fit problem, where we are instead interested in whether the data
comes from any distribution in some parametric family. This is equivalent to
testing whether a parametric model is well-specified for the data.

    

### [[2111.10283] The Joy of Neural Painting](http://arxiv.org/abs/2111.10283)


  Neural Painters is a class of models that follows a GAN framework to generate
brushstrokes, which are then composed to create paintings. GANs are great
generative models for AI Art but they are known to be notoriously difficult to
train. To overcome GAN's limitations and to speed up the Neural Painter
training, we applied Transfer Learning to the process reducing it from days to
only hours, while achieving the same level of visual aesthetics in the final
paintings generated. We report our approach and results in this work.

    

### [[2111.10285] Adversarial Deep Learning for Online Resource Allocation](http://arxiv.org/abs/2111.10285)


  Online algorithm is an important branch in algorithm design. Designing online
algorithms with a bounded competitive ratio (in terms of worst-case
performance) can be hard and usually relies on problem-specific assumptions.
Inspired by adversarial training from Generative Adversarial Net (GAN) and the
fact that competitive ratio of an online algorithm is based on worst-case
input, we adopt deep neural networks to learn an online algorithm for a
resource allocation and pricing problem from scratch, with the goal that the
performance gap between offline optimum and the learned online algorithm can be
minimized for worst-case input.
Specifically, we leverage two neural networks as algorithm and adversary
respectively and let them play a zero sum game, with the adversary being
responsible for generating worst-case input while the algorithm learns the best
strategy based on the input provided by the adversary. To ensure better
convergence of the algorithm network (to the desired online algorithm), we
propose a novel per-round update method to handle sequential decision making to
break complex dependency among different rounds so that update can be done for
every possible action, instead of only sampled actions. To the best of our
knowledge, our work is the first using deep neural networks to design an online
algorithm from the perspective of worst-case performance guarantee. Empirical
studies show that our updating methods ensure convergence to Nash equilibrium
and the learned algorithm outperforms state-of-the-art online algorithms under
various settings.

    

### [[2111.10291] Meta Adversarial Perturbations](http://arxiv.org/abs/2111.10291)


  A plethora of attack methods have been proposed to generate adversarial
examples, among which the iterative methods have been demonstrated the ability
to find a strong attack. However, the computation of an adversarial
perturbation for a new data point requires solving a time-consuming
optimization problem from scratch. To generate a stronger attack, it normally
requires updating a data point with more iterations. In this paper, we show the
existence of a meta adversarial perturbation (MAP), a better initialization
that causes natural images to be misclassified with high probability after
being updated through only a one-step gradient ascent update, and propose an
algorithm for computing such perturbations. We conduct extensive experiments,
and the empirical results demonstrate that state-of-the-art deep neural
networks are vulnerable to meta perturbations. We further show that these
perturbations are not only image-agnostic, but also model-agnostic, as a single
perturbation generalizes well across unseen data points and different neural
network architectures.

    

### [[2111.10293] A 3D 2D convolutional Neural Network Model for Hyperspectral Image Classification](http://arxiv.org/abs/2111.10293)


  In the proposed SEHybridSN model, a dense block was used to reuse shallow
feature and aimed at better exploiting hierarchical spatial spectral feature.
Subsequent depth separable convolutional layers were used to discriminate the
spatial information. Further refinement of spatial spectral features was
realized by the channel attention method, which were performed behind every 3D
convolutional layer and every 2D convolutional layer. Experiment results
indicate that our proposed model learn more discriminative spatial spectral
features using very few training data. SEHybridSN using only 0.05 and 0.01
labeled data for training, a very satisfactory performance is obtained.

    

### [[2111.10297] Expert-Guided Symmetry Detection in Markov Decision Processes](http://arxiv.org/abs/2111.10297)


  Learning a Markov Decision Process (MDP) from a fixed batch of trajectories
is a non-trivial task whose outcome's quality depends on both the amount and
the diversity of the sampled regions of the state-action space. Yet, many MDPs
are endowed with invariant reward and transition functions with respect to some
transformations of the current state and action. Being able to detect and
exploit these structures could benefit not only the learning of the MDP but
also the computation of its subsequent optimal control policy. In this work we
propose a paradigm, based on Density Estimation methods, that aims to detect
the presence of some already supposed transformations of the state-action space
for which the MDP dynamics is invariant. We tested the proposed approach in a
discrete toroidal grid environment and in two notorious environments of
OpenAI's Gym Learning Suite. The results demonstrate that the model
distributional shift is reduced when the dataset is augmented with the data
obtained by using the detected symmetries, allowing for a more thorough and
data-efficient learning of the transition functions.

    

### [[2111.10298] An Asymptotic Equivalence between the Mean-Shift Algorithm and the Cluster Tree](http://arxiv.org/abs/2111.10298)


  Two important nonparametric approaches to clustering emerged in the 1970's:
clustering by level sets or cluster tree as proposed by Hartigan, and
clustering by gradient lines or gradient flow as proposed by Fukunaga and
Hosteler. In a recent paper, we argue the thesis that these two approaches are
fundamentally the same by showing that the gradient flow provides a way to move
along the cluster tree. In making a stronger case, we are confronted with the
fact the cluster tree does not define a partition of the entire support of the
underlying density, while the gradient flow does. In the present paper, we
resolve this conundrum by proposing two ways of obtaining a partition from the
cluster tree -- each one of them very natural in its own right -- and showing
that both of them reduce to the partition given by the gradient flow under
standard assumptions on the sampling density.

    

### [[2111.10302] Instance-Adaptive Video Compression: Improving Neural Codecs by Training on the Test Set](http://arxiv.org/abs/2111.10302)


  We introduce a video compression algorithm based on instance-adaptive
learning. On each video sequence to be transmitted, we finetune a pretrained
compression model. The optimal parameters are transmitted to the receiver along
with the latent code. By entropy-coding the parameter updates under a suitable
mixture model prior, we ensure that the network parameters can be encoded
efficiently. This instance-adaptive compression algorithm is agnostic about the
choice of base model and has the potential to improve any neural video codec.
On UVG, HEVC, and Xiph datasets, our codec improves the performance of a
low-latency scale-space flow model by between 21% and 26% BD-rate savings, and
that of a state-of-the-art B-frame model by 17 to 20% BD-rate savings. We also
demonstrate that instance-adaptive finetuning improves the robustness to domain
shift. Finally, our approach reduces the capacity requirements on compression
models. We show that it enables a state-of-the-art performance even after
reducing the network size by 72%.

    

### [[2111.10309] Unsupervised Visual Time-Series Representation Learning and Clustering](http://arxiv.org/abs/2111.10309)


  Time-series data is generated ubiquitously from Internet-of-Things (IoT)
infrastructure, connected and wearable devices, remote sensing, autonomous
driving research and, audio-video communications, in enormous volumes. This
paper investigates the potential of unsupervised representation learning for
these time-series. In this paper, we use a novel data transformation along with
novel unsupervised learning regime to transfer the learning from other domains
to time-series where the former have extensive models heavily trained on very
large labelled datasets. We conduct extensive experiments to demonstrate the
potential of the proposed approach through time-series clustering.

    

### [[2111.10320] Toward Compact Parameter Representations for Architecture-Agnostic Neural Network Compression](http://arxiv.org/abs/2111.10320)


  This paper investigates deep neural network (DNN) compression from the
perspective of compactly representing and storing trained parameters. We
explore the previously overlooked opportunity of cross-layer
architecture-agnostic representation sharing for DNN parameters. To do this, we
decouple feedforward parameters from DNN architectures and leverage additive
quantization, an extreme lossy compression method invented for image
descriptors, to compactly represent the parameters. The representations are
then finetuned on task objectives to improve task accuracy. We conduct
extensive experiments on MobileNet-v2, VGG-11, ResNet-50, Feature Pyramid
Networks, and pruned DNNs trained for classification, detection, and
segmentation tasks. The conceptually simple scheme consistently outperforms
iterative unstructured pruning. Applied to ResNet-50 with 76.1% top-1 accuracy
on the ILSVRC12 classification challenge, it achieves a $7.2\times$ compression
ratio with no accuracy loss and a $15.3\times$ compression ratio at 74.79%
accuracy. Further analyses suggest that representation sharing can frequently
happen across network layers and that learning shared representations for an
entire DNN can achieve better accuracy at the same compression ratio than
compressing the model as multiple separate parts. We release PyTorch code to
facilitate DNN deployment on resource-constrained devices and spur future
research on efficient representations and storage of DNN parameters.

    

### [[2111.10329] Physics-enhanced Neural Networks in the Small Data Regime](http://arxiv.org/abs/2111.10329)


  Identifying the dynamics of physical systems requires a machine learning
model that can assimilate observational data, but also incorporate the laws of
physics. Neural Networks based on physical principles such as the Hamiltonian
or Lagrangian NNs have recently shown promising results in generating
extrapolative predictions and accurately representing the system's dynamics. We
show that by additionally considering the actual energy level as a
regularization term during training and thus using physical information as
inductive bias, the results can be further improved. Especially in the case
where only small amounts of data are available, these improvements can
significantly enhance the predictive capability. We apply the proposed
regularization term to a Hamiltonian Neural Network (HNN) and Constrained
Hamiltonian Neural Network (CHHN) for a single and double pendulum, generate
predictions under unseen initial conditions and report significant gains in
predictive accuracy.

    

### [[2111.10332] DSPoint: Dual-scale Point Cloud Recognition with High-frequency Fusion](http://arxiv.org/abs/2111.10332)


  Point cloud processing is a challenging task due to its sparsity and
irregularity. Prior works introduce delicate designs on either local feature
aggregator or global geometric architecture, but few combine both advantages.
We propose Dual-Scale Point Cloud Recognition with High-frequency Fusion
(DSPoint) to extract local-global features by concurrently operating on voxels
and points. We reverse the conventional design of applying convolution on
voxels and attention to points. Specifically, we disentangle point features
through channel dimension for dual-scale processing: one by point-wise
convolution for fine-grained geometry parsing, the other by voxel-wise global
attention for long-range structural exploration. We design a co-attention
fusion module for feature alignment to blend local-global modalities, which
conducts inter-scale cross-modality interaction by communicating high-frequency
coordinates information. Experiments and ablations on widely-adopted
ModelNet40, ShapeNet, and S3DIS demonstrate the state-of-the-art performance of
our DSPoint.

    

### [[2111.10342] GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation](http://arxiv.org/abs/2111.10342)


  In this paper, we present GRecX, an open-source TensorFlow framework for
benchmarking GNN-based recommendation models in an efficient and unified way.
GRecX consists of core libraries for building GNN-based recommendation
benchmarks, as well as the implementations of popular GNN-based recommendation
models. The core libraries provide essential components for building efficient
and unified benchmarks, including FastMetrics (efficient metrics computation
libraries), VectorSearch (efficient similarity search libraries for dense
vectors), BatchEval (efficient mini-batch evaluation libraries), and
DataManager (unified dataset management libraries). Especially, to provide a
unified benchmark for the fair comparison of different complex GNN-based
recommendation models, we design a new metric GRMF-X and integrate it into the
FastMetrics component. Based on a TensorFlow GNN library tf_geometric, GRecX
carefully implements a variety of popular GNN-based recommendation models. We
carefully implement these baseline models to reproduce the performance reported
in the literature, and our implementations are usually more efficient and
friendly. In conclusion, GRecX enables uses to train and benchmark GNN-based
recommendation baselines in an efficient and unified way. We conduct
experiments with GRecX, and the experimental results show that GRecX allows us
to train and benchmark GNN-based recommendation baselines in an efficient and
unified way. The source code of GRecX is available at
this https URL.

    

### [[2111.10344] Maximum Mean Discrepancy for Generalization in the Presence of Distribution and Missingness Shift](http://arxiv.org/abs/2111.10344)


  Covariate shifts are a common problem in predictive modeling on real-world
problems. This paper proposes addressing the covariate shift problem by
minimizing Maximum Mean Discrepancy (MMD) statistics between the training and
test sets in either feature input space, feature representation space, or both.
We designed three techniques that we call MMD Representation, MMD Mask, and MMD
Hybrid to deal with the scenarios where only a distribution shift exists, only
a missingness shift exists, or both types of shift exist, respectively. We find
that integrating an MMD loss component helps models use the best features for
generalization and avoid dangerous extrapolation as much as possible for each
test sample. Models treated with this MMD approach show better performance,
calibration, and extrapolation on the test set.

    

### [[2111.10352] On the power of adaptivity in statistical adversaries](http://arxiv.org/abs/2111.10352)


  We study a fundamental question concerning adversarial noise models in
statistical problems where the algorithm receives i.i.d. draws from a
distribution $\mathcal{D}$. The definitions of these adversaries specify the
type of allowable corruptions (noise model) as well as when these corruptions
can be made (adaptivity); the latter differentiates between oblivious
adversaries that can only corrupt the distribution $\mathcal{D}$ and adaptive
adversaries that can have their corruptions depend on the specific sample $S$
that is drawn from $\mathcal{D}$.
In this work, we investigate whether oblivious adversaries are effectively
equivalent to adaptive adversaries, across all noise models studied in the
literature. Specifically, can the behavior of an algorithm $\mathcal{A}$ in the
presence of oblivious adversaries always be well-approximated by that of an
algorithm $\mathcal{A}'$ in the presence of adaptive adversaries? Our first
result shows that this is indeed the case for the broad class of statistical
query algorithms, under all reasonable noise models. We then show that in the
specific case of additive noise, this equivalence holds for all algorithms.
Finally, we map out an approach towards proving this statement in its fullest
generality, for all algorithms and under all reasonable noise models.

    

### [[2111.10361] Solving Visual Analogies Using Neural Algorithmic Reasoning](http://arxiv.org/abs/2111.10361)


  We consider a class of visual analogical reasoning problems that involve
discovering the sequence of transformations by which pairs of input/output
images are related, so as to analogously transform future inputs. This program
synthesis task can be easily solved via symbolic search. Using a variation of
the `neural analogical reasoning' approach of (Velickovic and Blundell 2021),
we instead search for a sequence of elementary neural network transformations
that manipulate distributed representations derived from a symbolic space, to
which input images are directly encoded. We evaluate the extent to which our
`neural reasoning' approach generalizes for images with unseen shapes and
positions.

    

### [[2111.10364] Generalized Decision Transformer for Offline Hindsight Information Matching](http://arxiv.org/abs/2111.10364)


  How to extract as much learning signal from each trajectory data has been a
key problem in reinforcement learning (RL), where sample inefficiency has posed
serious challenges for practical applications. Recent works have shown that
using expressive policy function approximators and conditioning on future
trajectory information -- such as future states in hindsight experience replay
or returns-to-go in Decision Transformer (DT) -- enables efficient learning of
multi-task policies, where at times online RL is fully replaced by offline
behavioral cloning, e.g. sequence modeling. We demonstrate that all these
approaches are doing hindsight information matching (HIM) -- training policies
that can output the rest of trajectory that matches some statistics of future
state information. We present Generalized Decision Transformer (GDT) for
solving any HIM problem, and show how different choices for the feature
function and the anti-causal aggregator not only recover DT as a special case,
but also lead to novel Categorical DT (CDT) and Bi-directional DT (BDT) for
matching different statistics of the future. For evaluating CDT and BDT, we
define offline multi-task state-marginal matching (SMM) and imitation learning
(IL) as two generic HIM problems, propose a Wasserstein distance loss as a
metric for both, and empirically study them on MuJoCo continuous control
benchmarks. CDT, which simply replaces anti-causal summation with anti-causal
binning in DT, enables the first effective offline multi-task SMM algorithm
that generalizes well to unseen and even synthetic multi-modal state-feature
distributions. BDT, which uses an anti-causal second transformer as the
aggregator, can learn to model any statistics of the future and outperforms DT
variants in offline multi-task IL. Our generalized formulations from HIM and
GDT greatly expand the role of powerful sequence modeling architectures in
modern RL.

    

### [[2111.10367] SLUE: New Benchmark Tasks for Spoken Language Understanding Evaluation on Natural Speech](http://arxiv.org/abs/2111.10367)


  Progress in speech processing has been facilitated by shared datasets and
benchmarks. Historically these have focused on automatic speech recognition
(ASR), speaker identification, or other lower-level tasks. Interest has been
growing in higher-level spoken language understanding tasks, including using
end-to-end models, but there are fewer annotated datasets for such tasks. At
the same time, recent work shows the possibility of pre-training generic
representations and then fine-tuning for several tasks using relatively little
labeled data. We propose to create a suite of benchmark tasks for Spoken
Language Understanding Evaluation (SLUE) consisting of limited-size labeled
training sets and corresponding evaluation sets. This resource would allow the
research community to track progress, evaluate pre-trained representations for
higher-level tasks, and study open questions such as the utility of pipeline
versus end-to-end approaches. We present the first phase of the SLUE benchmark
suite, consisting of named entity recognition, sentiment analysis, and ASR on
the corresponding datasets. We focus on naturally produced (not read or
synthesized) speech, and freely available datasets. We provide new
transcriptions and annotations on subsets of the VoxCeleb and VoxPopuli
datasets, evaluation metrics and results for baseline models, and an
open-source toolkit to reproduce the baselines and evaluate new models.

    

### [[1805.08969] Semantic Network Interpretation](http://arxiv.org/abs/1805.08969)


  Network interpretation as an effort to reveal the features learned by a
network remains largely visualization-based. In this paper, our goal is to
tackle semantic network interpretation at both filter and decision level. For
filter-level interpretation, we represent the concepts a filter encodes with a
probability distribution of visual attributes. The decision-level
interpretation is achieved by textual summarization that generates an
explanatory sentence containing clues behind a network's decision. A Bayesian
inference algorithm is proposed to automatically associate filters and network
decisions with visual attributes. Human study confirms that the semantic
interpretation is a beneficial alternative or complement to visualization
methods. We demonstrate the crucial role that semantic network interpretation
can play in understanding a network's failure patterns. More importantly,
semantic network interpretation enables a better understanding of the
correlation between a model's performance and its distribution metrics like
filter selectivity and concept sparseness.

    

### [[1910.07567] Active Learning for Graph Neural Networks via Node Feature Propagation](http://arxiv.org/abs/1910.07567)


  Graph Neural Networks (GNNs) for prediction tasks like node classification or
edge prediction have received increasing attention in recent machine learning
from graphically structured data. However, a large quantity of labeled graphs
is difficult to obtain, which significantly limits the true success of GNNs.
Although active learning has been widely studied for addressing label-sparse
issues with other data types like text, images, etc., how to make it effective
over graphs is an open question for research. In this paper, we present an
investigation on active learning with GNNs for node classification tasks.
Specifically, we propose a new method, which uses node feature propagation
followed by K-Medoids clustering of the nodes for instance selection in active
learning. With a theoretical bound analysis we justify the design choice of our
approach. In our experiments on four benchmark datasets, the proposed method
outperforms other representative baseline methods consistently and
significantly.

    

### [[1912.09323] NFAD: Fixing anomaly detection using normalizing flows](http://arxiv.org/abs/1912.09323)


  Anomaly detection is a challenging task that frequently arises in practically
all areas of industry and science, from fraud detection and data quality
monitoring to finding rare cases of diseases and searching for new physics.
Most of the conventional approaches to anomaly detection, such as one-class SVM
and Robust Auto-Encoder, are one-class classification methods, i.e. focus on
separating normal data from the rest of the space. Such methods are based on
the assumption of separability of normal and anomalous classes, and
subsequently do not take into account any available samples of anomalies.
Nonetheless, in practical settings, some anomalous samples are often available;
however, usually in amounts far lower than required for a balanced
classification task, and the separability assumption might not always hold.
This leads to an important task -- incorporating known anomalous samples into
training procedures of anomaly detection models.
In this work, we propose a novel model-agnostic training procedure to address
this task. We reformulate one-class classification as a binary classification
problem with normal data being distinguished from pseudo-anomalous samples. The
pseudo-anomalous samples are drawn from low-density regions of a normalizing
flow model by feeding tails of the latent distribution into the model. Such an
approach allows to easily include known anomalies into the training process of
an arbitrary classifier. We demonstrate that our approach shows comparable
performance on one-class problems, and, most importantly, achieves comparable
or superior results on tasks with variable amounts of known anomalies.

    

### [[2003.03229] Non-linear Neurons with Human-like Apical Dendrite Activations](http://arxiv.org/abs/2003.03229)


  In order to classify linearly non-separable data, neurons are typically
organized into multi-layer neural networks that are equipped with at least one
hidden layer. Inspired by some recent discoveries in neuroscience, we propose a
new neuron model along with a novel activation function enabling the learning
of non-linear decision boundaries using a single neuron. We show that a
standard neuron followed by the novel apical dendrite activation (ADA) can
learn the XOR logical function with 100\% accuracy. Furthermore, we conduct
experiments on five benchmark data sets from computer vision, signal processing
and natural language processing, i.e. MOROCO, UTKFace, CREMA-D, Fashion-MNIST,
and Tiny ImageNet, showing that ADA and the leaky ADA functions provide
superior results to Rectified Linear Units (ReLU), leaky ReLU, RBF and Swish,
for various neural network architectures, e.g. one-hidden-layer or
two-hidden-layer multi-layer perceptrons (MLPs) and convolutional neural
networks (CNNs) such as LeNet, VGG, ResNet and Character-level CNN. We obtain
further performance improvements when we change the standard model of the
neuron with our pyramidal neuron with apical dendrite activations (PyNADA). Our
code is available at: this https URL.

    

### [[2004.03955] Dendrite Net: A White-Box Module for Classification, Regression, and System Identification](http://arxiv.org/abs/2004.03955)


  The simulation of biological dendrite computations is vital for the
development of artificial intelligence (AI). This paper presents a basic
machine learning algorithm, named Dendrite Net or DD, just like Support Vector
Machine (SVM) or Multilayer Perceptron (MLP). DD's main concept is that the
algorithm can recognize this class after learning, if the output's logical
expression contains the corresponding class's logical relationship among inputs
(and$\backslash$or$\backslash$not). Experiments and main results: DD, a
white-box machine learning algorithm, showed excellent system identification
performance for the black-box system. Secondly, it was verified by nine
real-world applications that DD brought better generalization capability
relative to MLP architecture that imitated neurons' cell body (Cell body Net)
for regression. Thirdly, by MNIST and FASHION-MNIST datasets, it was verified
that DD showed higher testing accuracy under greater training loss than Cell
body Net for classification. The number of modules can effectively adjust DD's
logical expression capacity, which avoids over-fitting and makes it easy to get
a model with outstanding generalization capability. Finally, repeated
experiments in MATLAB and PyTorch (Python) demonstrated that DD was faster than
Cell body Net both in epoch and forward-propagation. The main contribution of
this paper is the basic machine learning algorithm (DD) with a white-box
attribute, controllable precision for better generalization capability, and
lower computational complexity. Not only can DD be used for generalized
engineering, but DD has vast development potential as a module for deep
learning. DD code is available at GitHub:
this https URL .

    

### [[2005.01097] Adaptive Learning of the Optimal Batch Size of SGD](http://arxiv.org/abs/2005.01097)


  Recent advances in the theoretical understanding of SGD led to a formula for
the optimal batch size minimizing the number of effective data passes, i.e.,
the number of iterations times the batch size. However, this formula is of no
practical value as it depends on the knowledge of the variance of the
stochastic gradients evaluated at the optimum. In this paper we design a
practical SGD method capable of learning the optimal batch size adaptively
throughout its iterations for strongly convex and smooth functions. Our method
does this provably, and in our experiments with synthetic and real data
robustly exhibits nearly optimal behaviour; that is, it works as if the optimal
batch size was known a-priori. Further, we generalize our method to several new
batch strategies not considered in the literature before, including a sampling
suitable for distributed implementations.

    

### [[2005.14501] A Performance-Explainability Framework to Benchmark Machine Learning Methods: Application to Multivariate Time Series Classifiers](http://arxiv.org/abs/2005.14501)


  Our research aims to propose a new performance-explainability analytical
framework to assess and benchmark machine learning methods. The framework
details a set of characteristics that systematize the
performance-explainability assessment of existing machine learning methods. In
order to illustrate the use of the framework, we apply it to benchmark the
current state-of-the-art multivariate time series classifiers.

    

### [[2006.07682] Rethinking Clustering for Robustness](http://arxiv.org/abs/2006.07682)


  This paper studies how encouraging semantically-aligned features during deep
neural network training can increase network robustness. Recent works observed
that Adversarial Training leads to robust models, whose learnt features appear
to correlate with human perception. Inspired by this connection from robustness
to semantics, we study the complementary connection: from semantics to
robustness. To do so, we provide a robustness certificate for distance-based
classification models (clustering-based classifiers). Moreover, we show that
this certificate is tight, and we leverage it to propose ClusTR (Clustering
Training for Robustness), a clustering-based and adversary-free training
framework to learn robust models. Interestingly, \textit{ClusTR} outperforms
adversarially-trained networks by up to $4\%$ under strong PGD attacks.

    

### [[2008.10797] The Fairness-Accuracy Pareto Front](http://arxiv.org/abs/2008.10797)


  Algorithmic fairness seeks to identify and correct sources of bias in machine
learning algorithms. Confoundingly, ensuring fairness often comes at the cost
of accuracy. We provide formal tools in this work for reconciling this
fundamental tension in algorithm fairness. Specifically, we put to use the
concept of Pareto optimality from multi-objective optimization and seek the
fairness-accuracy Pareto front of a neural network classifier. We demonstrate
that many existing algorithmic fairness methods are performing the so-called
linear scalarization scheme which has severe limitations in recovering Pareto
optimal solutions. We instead apply the Chebyshev scalarization scheme which is
provably superior theoretically and no more computationally burdensome at
recovering Pareto optimal solutions compared to the linear scheme.

    

### [[2009.03979] A Distance-preserving Matrix Sketch](http://arxiv.org/abs/2009.03979)


  Visualizing very large matrices involves many formidable problems. Various
popular solutions to these problems involve sampling, clustering, projection,
or feature selection to reduce the size and complexity of the original task. An
important aspect of these methods is how to preserve relative distances between
points in the higher-dimensional space after reducing rows and columns to fit
in a lower dimensional space. This aspect is important because conclusions
based on faulty visual reasoning can be harmful. Judging dissimilar points as
similar or similar points as dissimilar on the basis of a visualization can
lead to false conclusions. To ameliorate this bias and to make visualizations
of very large datasets feasible, we introduce two new algorithms that
respectively select a subset of rows and columns of a rectangular matrix. This
selection is designed to preserve relative distances as closely as possible. We
compare our matrix sketch to more traditional alternatives on a variety of
artificial and real datasets.

    

### [[2010.16344] Marginalised Gaussian Processes with Nested Sampling](http://arxiv.org/abs/2010.16344)


  Gaussian Process (GPs) models are a rich distribution over functions with
inductive biases controlled by a kernel function. Learning occurs through the
optimisation of kernel hyperparameters using the marginal likelihood as the
objective. This classical approach known as Type-II maximum likelihood (ML-II)
yields point estimates of the hyperparameters, and continues to be the default
method for training GPs. However, this approach risks underestimating
predictive uncertainty and is prone to overfitting especially when there are
many hyperparameters. Furthermore, gradient based optimisation makes ML-II
point estimates highly susceptible to the presence of local minima. This work
presents an alternative learning procedure where the hyperparameters of the
kernel function are marginalised using Nested Sampling (NS), a technique that
is well suited to sample from complex, multi-modal distributions. We focus on
regression tasks with the spectral mixture (SM) class of kernels and find that
a principled approach to quantifying model uncertainty leads to substantial
gains in predictive performance across a range of synthetic and benchmark data
sets. In this context, nested sampling is also found to offer a speed advantage
over Hamiltonian Monte Carlo (HMC), widely considered to be the gold-standard
in MCMC based inference.

    

### [[2011.07126] Zero-shot Relation Classification from Side Information](http://arxiv.org/abs/2011.07126)


  We propose a zero-shot learning relation classification (ZSLRC) framework
that improves on state-of-the-art by its ability to recognize novel relations
that were not present in training data. The zero-shot learning approach mimics
the way humans learn and recognize new concepts with no prior knowledge. To
achieve this, ZSLRC uses advanced prototypical networks that are modified to
utilize weighted side (auxiliary) information. ZSLRC's side information is
built from keywords, hypernyms of name entities, and labels and their synonyms.
ZSLRC also includes an automatic hypernym extraction framework that acquires
hypernyms of various name entities directly from the web. ZSLRC improves on
state-of-the-art few-shot learning relation classification methods that rely on
labeled training data and is therefore applicable more widely even in
real-world scenarios where some relations have no corresponding labeled
examples for training. We present results using extensive experiments on two
public datasets (NYT and FewRel) and show that ZSLRC significantly outperforms
state-of-the-art methods on supervised learning, few-shot learning, and
zero-shot learning tasks. Our experimental results also demonstrate the
effectiveness and robustness of our proposed model.

    

### [[2012.04207] Efficient Estimation of Influence of a Training Instance](http://arxiv.org/abs/2012.04207)


  Understanding the influence of a training instance on a neural network model
leads to improving interpretability. However, it is difficult and inefficient
to evaluate the influence, which shows how a model's prediction would be
changed if a training instance were not used. In this paper, we propose an
efficient method for estimating the influence. Our method is inspired by
dropout, which zero-masks a sub-network and prevents the sub-network from
learning each training instance. By switching between dropout masks, we can use
sub-networks that learned or did not learn each training instance and estimate
its influence. Through experiments with BERT and VGGNet on classification
datasets, we demonstrate that the proposed method can capture training
influences, enhance the interpretability of error predictions, and cleanse the
training dataset for improving generalization.

    

### [[2012.06047] KNN Classification with One-step Computation](http://arxiv.org/abs/2012.06047)


  KNN classification is an improvisational learning mode, in which they are
carried out only when a test data is predicted that set a suitable K value and
search the K nearest neighbors from the whole training sample space, referred
them to the lazy part of KNN classification. This lazy part has been the
bottleneck problem of applying KNN classification due to the complete search of
K nearest neighbors. In this paper, a one-step computation is proposed to
replace the lazy part of KNN classification. The one-step computation actually
transforms the lazy part to a matrix computation as follows. Given a test data,
training samples are first applied to fit the test data with the least squares
loss function. And then, a relationship matrix is generated by weighting all
training samples according to their influence on the test data. Finally, a
group lasso is employed to perform sparse learning of the relationship matrix.
In this way, setting K value and searching K nearest neighbors are both
integrated to a unified computation. In addition, a new classification rule is
proposed for improving the performance of one-step KNN classification. The
proposed approach is experimentally evaluated, and demonstrated that the
one-step KNN classification is efficient and promising

    

### [[2102.09798] Training Neural Networks is $\exists\mathbb R$-complete](http://arxiv.org/abs/2102.09798)


  Given a neural network, training data, and a threshold, it was known that it
is NP-hard to find weights for the neural network such that the total error is
below the threshold. We determine the algorithmic complexity of this
fundamental problem precisely, by showing that it is $\exists\mathbb
R$-complete. This means that the problem is equivalent, up to polynomial-time
reductions, to deciding whether a system of polynomial equations and
inequalities with integer coefficients and real unknowns has a solution. If, as
widely expected, $\exists\mathbb R$ is strictly larger than NP, our work
implies that the problem of training neural networks is not even in NP.
Neural networks are usually trained using some variation of backpropagation.
The result of this paper offers an explanation why techniques commonly used to
solve big instances of NP-complete problems seem not to be of use for this
task. Examples of such techniques are SAT solvers, IP solvers, local search,
dynamic programming, to name a few general ones.

    

### [[2102.10773] Slowly Varying Regression under Sparsity](http://arxiv.org/abs/2102.10773)


  We consider the problem of parameter estimation in slowly varying regression
models with sparsity constraints. We formulate the problem as a mixed integer
optimization problem and demonstrate that it can be reformulated exactly as a
binary convex optimization problem through a novel exact relaxation. The
relaxation utilizes a new equality on Moore-Penrose inverses that convexifies
the non-convex objective function while coinciding with the original objective
on all feasible binary points. This allows us to solve the problem
significantly more efficiently and to provable optimality using a cutting
plane-type algorithm. We develop a highly optimized implementation of such
algorithm, which substantially improves upon the asymptotic computational
complexity of a straightforward implementation. We further develop a heuristic
method that is guaranteed to produce a feasible solution and, as we empirically
illustrate, generates high quality warm-start solutions for the binary
optimization problem. We show, on both synthetic and real-world datasets, that
the resulting algorithm outperforms competing formulations in comparable times
across a variety of metrics including out-of-sample predictive performance,
support recovery accuracy, and false positive rate. The algorithm enables us to
train models with 10,000s of parameters, is robust to noise, and able to
effectively capture the underlying slowly changing support of the data
generating process.

    

### [[2102.11273] On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness](http://arxiv.org/abs/2102.11273)


  Invariance to a broad array of image corruptions, such as warping, noise, or
color shifts, is an important aspect of building robust models in computer
vision. Recently, several new data augmentations have been proposed that
significantly improve performance on ImageNet-C, a benchmark of such
corruptions. However, there is still a lack of basic understanding on the
relationship between data augmentations and test-time corruptions. To this end,
we develop a feature space for image transforms, and then use a new measure in
this space between augmentations and corruptions called the Minimal Sample
Distance to demonstrate a strong correlation between similarity and
performance. We then investigate recent data augmentations and observe a
significant degradation in corruption robustness when the test-time corruptions
are sampled to be perceptually dissimilar from ImageNet-C in this feature
space. Our results suggest that test error can be improved by training on
perceptually similar augmentations, and data augmentations may not generalize
well beyond the existing benchmark. We hope our results and tools will allow
for more robust progress towards improving robustness to image corruptions. We
provide code at this https URL.

    

### [[2102.12827] Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints](http://arxiv.org/abs/2102.12827)


  Evaluating adversarial robustness amounts to finding the minimum perturbation
needed to have an input sample misclassified. The inherent complexity of the
underlying optimization requires current gradient-based attacks to be carefully
tuned, initialized, and possibly executed for many computationally-demanding
iterations, even if specialized to a given perturbation model. In this work, we
overcome these limitations by proposing a fast minimum-norm (FMN) attack that
works with different $\ell_p$-norm perturbation models ($p=0, 1, 2, \infty$),
is robust to hyperparameter choices, does not require adversarial starting
points, and converges within few lightweight steps. It works by iteratively
finding the sample misclassified with maximum confidence within an
$\ell_p$-norm constraint of size $\epsilon$, while adapting $\epsilon$ to
minimize the distance of the current sample to the decision boundary. Extensive
experiments show that FMN significantly outperforms existing attacks in terms
of convergence speed and computation time, while reporting comparable or even
smaller perturbation sizes.

    

### [[2103.10334] Structure Inducing Pre-Training](http://arxiv.org/abs/2103.10334)


  We present a theoretical analysis from first principles that establishes a
novel connection between relational inductive bias of pre-training and
fine-tuning performance while providing an extended view on general
pre-training models. We further explore how existing pre-training methods
impose relational inductive biases, finding that the vast majority of existing
approaches focus almost exclusively on modelling relationships in an
intra-sample manner, rather than a per-sample manner. We build upon these
findings with simulations and empirical studies on standard benchmarks spanning
3 data modalities and 10 downstream tasks. These investigations validate our
theoretical analyses, and provides a recipe to produce new pre-training methods
which incorporate provably richer inductive biases than do existing methods in
line with user specified relational graphs.

    

### [[2103.13278] Safe Linear-Quadratic Dual Control with Almost Sure Performance Guarantee](http://arxiv.org/abs/2103.13278)


  This paper considers the linear-quadratic dual control problem where the
system parameters need to be identified and the control objective needs to be
optimized in the meantime. Contrary to existing works on data-driven
linear-quadratic regulation, which typically provide error or regret bounds
within a certain probability, we propose an online algorithm that guarantees
the asymptotic optimality of the controller in the almost sure sense. Our dual
control strategy consists of two parts: a switched controller with
time-decaying exploration noise and Markov parameter inference based on the
cross-correlation between the exploration noise and system output. Central to
the almost sure performance guarantee is a safe switched control strategy that
falls back to a known conservative but stable controller when the actual state
deviates significantly from the target state. We prove that this switching
strategy rules out any potential destabilizing controllers from being applied,
while the performance gap between our switching strategy and the optimal linear
state feedback is exponentially small. Under our dual control scheme, the
parameter inference error scales as $O(T^{-1/4+\epsilon})$, while the
suboptimality gap of control performance scales as $O(T^{-1/2+\epsilon})$,
where $T$ is the number of time steps, and $\epsilon$ is an arbitrarily small
positive number. Simulation results on an industrial process example are
provided to illustrate the effectiveness of our proposed strategy.

    

### [[2103.14886] Generalization over different cellular automata rules learned by a deep feed-forward neural network](http://arxiv.org/abs/2103.14886)


  To test generalization ability of a class of deep neural networks, we
randomly generate a large number of different rule sets for 2-D cellular
automata (CA), based on John Conway's Game of Life. Using these rules, we
compute several trajectories for each CA instance. A deep convolutional
encoder-decoder network with short and long range skip connections is trained
on various generated CA trajectories to predict the next CA state given its
previous states. Results show that the network is able to learn the rules of
various, complex cellular automata and generalize to unseen configurations. To
some extent, the network shows generalization to rule sets and neighborhood
sizes that were not seen during the training at all. Code to reproduce the
experiments is publicly available at:
this https URL


### [[2105.11748] Dense Regression Activation Maps For Lesion Segmentation in CT scans of COVID-19 patients](http://arxiv.org/abs/2105.11748)


  Automatic lesion segmentation on thoracic CT enables rapid quantitative
analysis of lung involvement in COVID-19 infections. However, obtaining a large
amount of voxel-level annotations for training segmentation networks is
prohibitively expensive. Therefore, we propose a weakly-supervised segmentation
method based on dense regression activation maps (dRAMs). Most
weakly-supervised segmentation approaches exploit class activation maps (CAMs)
to localize objects. However, because CAMs were trained for classification,
they do not align precisely with the object segmentations. Instead, we produce
high-resolution activation maps using dense features from a segmentation
network that was trained to estimate a per-lobe lesion percentage. In this way,
the network can exploit knowledge regarding the required lesion volume. In
addition, we propose an attention neural network module to refine dRAMs,
optimized together with the main regression task. We evaluated our algorithm on
90 subjects. Results show our method achieved 70.2% Dice coefficient,
substantially outperforming the CAM-based baseline at 48.6%.

    

### [[2106.04399] Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](http://arxiv.org/abs/2106.04399)


  This paper is about the problem of learning a stochastic policy for
generating an object (like a molecular graph) from a sequence of actions, such
that the probability of generating an object is proportional to a given
positive reward for that object. Whereas standard return maximization tends to
converge to a single return-maximizing sequence, there are cases where we would
like to sample a diverse set of high-return solutions. These arise, for
example, in black-box function optimization when few rounds are possible, each
with large batches of queries, where the batches should be diverse, e.g., in
the design of new molecules. One can also see this as a problem of
approximately converting an energy function to a generative distribution. While
MCMC methods can achieve that, they are expensive and generally only perform
local exploration. Instead, training a generative policy amortizes the cost of
search during training and yields to fast generation. Using insights from
Temporal Difference learning, we propose GFlowNet, based on a view of the
generative process as a flow network, making it possible to handle the tricky
case where different trajectories can yield the same final state, e.g., there
are many ways to sequentially add atoms to generate some molecular graph. We
cast the set of trajectories as a flow and convert the flow consistency
equations into a learning objective, akin to the casting of the Bellman
equations into Temporal Difference methods. We prove that any global minimum of
the proposed objectives yields a policy which samples from the desired
distribution, and demonstrate the improved performance and diversity of
GFlowNet on a simple domain where there are many modes to the reward function,
and on a molecule synthesis task.

    

### [[2106.08185] Kernel Identification Through Transformers](http://arxiv.org/abs/2106.08185)


  Kernel selection plays a central role in determining the performance of
Gaussian Process (GP) models, as the chosen kernel determines both the
inductive biases and prior support of functions under the GP prior. This work
addresses the challenge of constructing custom kernel functions for
high-dimensional GP regression models. Drawing inspiration from recent progress
in deep learning, we introduce a novel approach named KITT: Kernel
Identification Through Transformers. KITT exploits a transformer-based
architecture to generate kernel recommendations in under 0.1 seconds, which is
several orders of magnitude faster than conventional kernel search algorithms.
We train our model using synthetic data generated from priors over a vocabulary
of known kernels. By exploiting the nature of the self-attention mechanism,
KITT is able to process datasets with inputs of arbitrary dimension. We
demonstrate that kernels chosen by KITT yield strong performance over a diverse
collection of regression benchmarks.

    

### [[2106.12718] Sparse Flows: Pruning Continuous-depth Models](http://arxiv.org/abs/2106.12718)


  Continuous deep learning architectures enable learning of flexible
probabilistic models for predictive modeling as neural ordinary differential
equations (ODEs), and for generative modeling as continuous normalizing flows.
In this work, we design a framework to decipher the internal dynamics of these
continuous depth models by pruning their network architectures. Our empirical
results suggest that pruning improves generalization for neural ODEs in
generative modeling. We empirically show that the improvement is because
pruning helps avoid mode-collapse and flatten the loss surface. Moreover,
pruning finds efficient neural ODE representations with up to 98% less
parameters compared to the original network, without loss of accuracy. We hope
our results will invigorate further research into the performance-size
trade-offs of modern continuous-depth models.

    

### [[2106.13239] Federated Noisy Client Learning](http://arxiv.org/abs/2106.13239)


  Federated learning (FL) collaboratively aggregates a shared global model
depending on multiple local clients, while keeping the training data
decentralized in order to preserve data privacy. However, standard FL methods
ignore the noisy client issue, which may harm the overall performance of the
aggregated model. In this paper, we first analyze the noisy client statement,
and then model noisy clients with different noise distributions (e.g.,
Bernoulli and truncated Gaussian distributions). To learn with noisy clients,
we propose a simple yet effective FL framework, named Federated Noisy Client
Learning (Fed-NCL), which is a plug-and-play algorithm and contains two main
components: a data quality measurement (DQM) to dynamically quantify the data
quality of each participating client, and a noise robust aggregation (NRA) to
adaptively aggregate the local models of each client by jointly considering the
amount of local training data and the data quality of each client. Our Fed-NCL
can be easily applied in any standard FL workflow to handle the noisy client
issue. Experimental results on various datasets demonstrate that our algorithm
boosts the performances of different state-of-the-art systems with noisy
clients.

    

### [[2110.05313] Unsupervised Source Separation via Bayesian Inference in the Latent Domain](http://arxiv.org/abs/2110.05313)


  State of the art audio source separation models rely on supervised
data-driven approaches, which can be expensive in terms of labeling resources.
On the other hand, approaches for training these models without any direct
supervision are typically high-demanding in terms of memory and time
requirements, and remain impractical to be used at inference time. We aim to
tackle these limitations by proposing a simple yet effective unsupervised
separation algorithm, which operates directly on a latent representation of
time-domain signals. Our algorithm relies on deep Bayesian priors in the form
of pre-trained autoregressive networks to model the probability distributions
of each source. We leverage the low cardinality of the discrete latent space,
trained with a novel loss term imposing a precise arithmetic structure on it,
to perform exact Bayesian inference without relying on an approximation
strategy. We validate our approach on the Slakh dataset arXiv:1909.08494,
demonstrating results in line with state of the art supervised approaches while
requiring fewer resources with respect to other unsupervised methods.

    

### [[2110.05428] Learning Temporally Causal Latent Processes from General Temporal Data](http://arxiv.org/abs/2110.05428)


  Our goal is to recover time-delayed latent causal variables and identify
their relations from measured temporal data. Estimating causally-related latent
variables from observations is particularly challenging as the latent variables
are not uniquely recoverable in the most general case. In this work, we
consider both a nonparametric, nonstationary setting and a parametric setting
for the latent processes and propose two provable conditions under which
temporally causal latent processes can be identified from their nonlinear
mixtures. We propose LEAP, a theoretically-grounded architecture that extends
Variational Autoencoders (VAEs) by enforcing our conditions through proper
constraints in causal process prior. Experimental results on various data sets
demonstrate that temporally causal latent processes are reliably identified
from observed variables under different dependency structures and that our
approach considerably outperforms baselines that do not leverage history or
nonstationarity information. This is one of the first works that successfully
recover time-delayed latent processes from nonlinear mixtures without using
sparsity or minimality assumptions.

    

### [[2110.12612] DelightfulTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2021](http://arxiv.org/abs/2110.12612)


  This paper describes the Microsoft end-to-end neural text to speech (TTS)
system: DelightfulTTS for Blizzard Challenge 2021. The goal of this challenge
is to synthesize natural and high-quality speech from text, and we approach
this goal in two perspectives: The first is to directly model and generate
waveform in 48 kHz sampling rate, which brings higher perception quality than
previous systems with 16 kHz or 24 kHz sampling rate; The second is to model
the variation information in speech through a systematic design, which improves
the prosody and naturalness. Specifically, for 48 kHz modeling, we predict 16
kHz mel-spectrogram in acoustic model, and propose a vocoder called HiFiNet to
directly generate 48 kHz waveform from predicted 16 kHz mel-spectrogram, which
can better trade off training efficiency, modelling stability and voice
quality. We model variation information systematically from both explicit
(speaker ID, language ID, pitch and duration) and implicit (utterance-level and
phoneme-level prosody) perspectives: 1) For speaker and language ID, we use
lookup embedding in training and inference; 2) For pitch and duration, we
extract the values from paired text-speech data in training and use two
predictors to predict the values in inference; 3) For utterance-level and
phoneme-level prosody, we use two reference encoders to extract the values in
training, and use two separate predictors to predict the values in inference.
Additionally, we introduce an improved Conformer block to better model the
local and global dependency in acoustic model. For task SH1, DelightfulTTS
achieves 4.17 mean score in MOS test and 4.35 in SMOS test, which indicates the
effectiveness of our proposed system

    

### [[2111.08243] CAR -- Cityscapes Attributes Recognition A Multi-category Attributes Dataset for Autonomous Vehicles](http://arxiv.org/abs/2111.08243)


  Self-driving vehicles are the future of transportation. With current
advancements in this field, the world is getting closer to safe roads with
almost zero probability of having accidents and eliminating human errors.
However, there is still plenty of research and development necessary to reach a
level of robustness. One important aspect is to understand a scene fully
including all details. As some characteristics (attributes) of objects in a
scene (drivers' behavior for instance) could be imperative for correct decision
making. However, current algorithms suffer from low-quality datasets with such
rich attributes. Therefore, in this paper, we present a new dataset for
attributes recognition -- Cityscapes Attributes Recognition (CAR). The new
dataset extends the well-known dataset Cityscapes by adding an additional yet
important annotation layer of attributes of objects in each image. Currently,
we have annotated more than 32k instances of various categories (Vehicles,
Pedestrians, etc.). The dataset has a structured and tailored taxonomy where
each category has its own set of possible attributes. The tailored taxonomy
focuses on attributes that is of most beneficent for developing better
self-driving algorithms that depend on accurate computer vision and scene
comprehension. We have also created an API for the dataset to ease the usage of
CAR. The API can be accessed through this https URL.

    

### [[2111.09459] Gradient flows on graphons: existence, convergence, continuity equations](http://arxiv.org/abs/2111.09459)


  Wasserstein gradient flows on probability measures have found a host of
applications in various optimization problems. They typically arise as the
continuum limit of exchangeable particle systems evolving by some mean-field
interaction involving a gradient-type potential. However, in many problems,
such as in multi-layer neural networks, the so-called particles are edge
weights on large graphs whose nodes are exchangeable. Such large graphs are
known to converge to continuum limits called graphons as their size grow to
infinity. We show that the Euclidean gradient flow of a suitable function of
the edge-weights converges to a novel continuum limit given by a curve on the
space of graphons that can be appropriately described as a gradient flow or,
more technically, a curve of maximal slope. Several natural functions on
graphons, such as homomorphism functions and the scalar entropy, are covered by
our set-up, and the examples have been worked out in detail.

    

### [[2111.09888] Simple but Effective: CLIP Embeddings for Embodied AI](http://arxiv.org/abs/2111.09888)


  Contrastive language image pretraining (CLIP) encoders have been shown to be
beneficial for a range of visual tasks from classification and detection to
captioning and image manipulation. We investigate the effectiveness of CLIP
visual backbones for embodied AI tasks. We build incredibly simple baselines,
named EmbCLIP, with no task specific architectures, inductive biases (such as
the use of semantic maps), auxiliary tasks during training, or depth maps --
yet we find that our improved baselines perform very well across a range of
tasks and simulators. EmbCLIP tops the RoboTHOR ObjectNav leaderboard by a huge
margin of 20 pts (Success Rate). It tops the iTHOR 1-Phase Rearrangement
leaderboard, beating the next best submission, which employs Active Neural
Mapping, and more than doubling the % Fixed Strict metric (0.08 to 0.17). It
also beats the winners of the 2021 Habitat ObjectNav Challenge, which employ
auxiliary tasks, depth maps, and human demonstrations, and those of the 2019
Habitat PointNav Challenge. We evaluate the ability of CLIP's visual
representations at capturing semantic information about input observations --
primitives that are useful for navigation-heavy embodied tasks -- and find that
CLIP's representations encode these primitives more effectively than
ImageNet-pretrained backbones. Finally, we extend one of our baselines,
producing an agent capable of zero-shot object navigation that can navigate to
objects that were not used as targets during training.

    

### [[2111.10027] E3NE: An End-to-End Framework for Accelerating Spiking Neural Networks with Emerging Neural Encoding on FPGAs](http://arxiv.org/abs/2111.10027)


  Compiler frameworks are crucial for the widespread use of FPGA-based deep
learning accelerators. They allow researchers and developers, who are not
familiar with hardware engineering, to harness the performance attained by
domain-specific logic. There exists a variety of frameworks for conventional
artificial neural networks. However, not much research effort has been put into
the creation of frameworks optimized for spiking neural networks (SNNs). This
new generation of neural networks becomes increasingly interesting for the
deployment of AI on edge devices, which have tight power and resource
constraints. Our end-to-end framework E3NE automates the generation of
efficient SNN inference logic for FPGAs. Based on a PyTorch model and user
parameters, it applies various optimizations and assesses trade-offs inherent
to spike-based accelerators. Multiple levels of parallelism and the use of an
emerging neural encoding scheme result in an efficiency superior to previous
SNN hardware implementations. For a similar model, E3NE uses less than 50% of
hardware resources and 20% less power, while reducing the latency by an order
of magnitude. Furthermore, scalability and generality allowed the deployment of
the large-scale SNN models AlexNet and VGG.

    

### [[2111.10200] Optimisation of job scheduling for supercomputers with burst buffers](http://arxiv.org/abs/2111.10200)


  The ever-increasing gap between compute and I/O performance in HPC platforms,
together with the development of novel NVMe storage devices (NVRAM), led to the
emergence of the burst buffer concept - an intermediate persistent storage
layer logically positioned between random-access main memory and a parallel
file system. Since the appearance of this technology, numerous supercomputers
have been equipped with burst buffers exploring various architectures. Despite
the development of real-world architectures as well as research concepts,
Resource and Job Management Systems, such as Slurm, provide only marginal
support for scheduling jobs with burst buffer requirements. This research is
primarily motivated by the alerting observation that burst buffers are omitted
from reservations in the procedure of backfilling in existing job schedulers.
In this dissertation, we forge a detailed supercomputer simulator based on
Batsim and SimGrid, which is capable of simulating I/O contention and I/O
congestion effects. Due to the lack of publicly available workloads with burst
buffer requests, we create a burst buffer request distribution model derived
from Parallel Workload Archive logs. We investigate the impact of burst buffer
reservations on the overall efficiency of online job scheduling for canonical
algorithms: First-Come-First-Served (FCFS) and Shortest-Job-First (SJF)
EASY-backfilling. Our results indicate that the lack of burst buffer
reservations in backfilling may significantly deteriorate the performance of
scheduling. [...] Furthermore, this lack of reservations may cause the
starvation of medium-size and wide jobs. Finally, we propose a
burst-buffer-aware plan-based scheduling algorithm with simulated annealing
optimisation, which improves the mean waiting time by over 20% and mean bounded
slowdown by 27% compared to the SJF EASY-backfilling.

    

### [[2111.09947] Parallel Algorithms for Masked Sparse Matrix-Matrix Products](http://arxiv.org/abs/2111.09947)


  Computing the product of two sparse matrices (SpGEMM) is a fundamental
operation in various combinatorial and graph algorithms as well as various
bioinformatics and data analytics applications for computing inner-product
similarities. For an important class of algorithms, only a subset of the output
entries are needed, and the resulting operation is known as Masked SpGEMM since
a subset of the output entries is considered to be "masked out". Existing
algorithms for Masked SpGEMM usually do not consider mask as part of
multiplication and either first compute a regular SpGEMM followed by masking,
or perform a sparse inner product only for output elements that are not masked
out. In this work, we investigate various novel algorithms and data structures
for this rather challenging and important computation, and provide guidelines
on how to design a fast Masked-SpGEMM for shared-memory architectures. Our
evaluations show that factors such as matrix and mask density, mask structure
and cache behavior play a vital role in attaining high performance for Masked
SpGEMM. We evaluate our algorithms on a large number of matrices using several
real-world benchmarks and show that our algorithms in most cases significantly
outperform the state of the art for Masked SpGEMM implementations.

    

### [[2111.10091] A Voting-Based Blockchain Interoperability Oracle](http://arxiv.org/abs/2111.10091)


  Today's blockchain landscape is severely fragmented as more and more
heterogeneous blockchain platforms have been developed in recent years. These
blockchain platforms are not able to interact with each other or with the
outside world since only little emphasis is placed on the interoperability
between them. Already proposed solutions for blockchain interoperability such
as naive relay or oracle solutions are usually not broadly applicable since
they are either too expensive to operate or very resource-intensive.
For that reason, we propose a blockchain interoperability oracle that follows
a voting-based approach based on threshold signatures. The oracle nodes
generate a distributed private key to execute an off-chain aggregation
mechanism to collectively respond to requests. Compared to state-of-the-art
relay schemes, our approach does not incur any ongoing costs and since the
on-chain component only needs to verify a single signature, we can achieve
remarkable cost savings compared to conventional oracle solutions.

    

### [[2111.10241] START: Straggler Prediction and Mitigation for Cloud Computing Environments using Encoder LSTM Networks](http://arxiv.org/abs/2111.10241)


  Modern large-scale computing systems distribute jobs into multiple smaller
tasks which execute in parallel to accelerate job completion rates and reduce
energy consumption. However, a common performance problem in such systems is
dealing with straggler tasks that are slow running instances that increase the
overall response time. Such tasks can significantly impact the system's Quality
of Service (QoS) and the Service Level Agreements (SLA). To combat this issue,
there is a need for automatic straggler detection and mitigation mechanisms
that execute jobs without violating the SLA. Prior work typically builds
reactive models that focus first on detection and then mitigation of straggler
tasks, which leads to delays. Other works use prediction based proactive
mechanisms, but ignore heterogeneous host or volatile task characteristics. In
this paper, we propose a Straggler Prediction and Mitigation Technique (START)
that is able to predict which tasks might be stragglers and dynamically adapt
scheduling to achieve lower response times. Our technique analyzes all tasks
and hosts based on compute and network resource consumption using an Encoder
Long-Short-Term-Memory (LSTM) network. The output of this network is then used
to predict and mitigate expected straggler tasks. This reduces the SLA
violation rate and execution time without compromising QoS. Specifically, we
use the CloudSim toolkit to simulate START in a cloud environment and compare
it with state-of-the-art techniques (IGRU-SD, SGC, Dolly, GRASS, NearestFit and
Wrangler) in terms of QoS parameters such as energy consumption, execution
time, resource contention, CPU utilization and SLA violation rate. Experiments
show that START reduces execution time, resource contention, energy and SLA
violations by 13%, 11%, 16% and 19%, respectively, compared to the
state-of-the-art approaches.

    

### [[2111.10270] FastDOG: Fast Discrete Optimization on GPU](http://arxiv.org/abs/2111.10270)


  We present a massively parallel Lagrange decomposition method for solving 0-1
integer linear programs occurring in structured prediction. We propose a new
iterative update scheme for solving the Lagrangean dual and a perturbation
technique for decoding primal solutions. For representing subproblems we follow
Lange et al. (2021) and use binary decision diagrams (BDDs). Our primal and
dual algorithms require little synchronization between subproblems and
optimization over BDDs needs only elementary operations without complicated
control flow. This allows us to exploit the parallelism offered by GPUs for all
components of our method. We present experimental results on combinatorial
problems from MAP inference for Markov Random Fields, quadratic assignment and
cell tracking for developmental biology. Our highly parallel GPU implementation
improves upon the running times of the algorithms from Lange et al. (2021) by
up to an order of magnitude. In particular, we come close to or outperform some
state-of-the-art specialized heuristics while being problem agnostic.

    

### [[2111.10333] Improving a High Productivity Data Analytics Chapel Framework](http://arxiv.org/abs/2111.10333)


  Most state of the art exploratory data analysis frameworks fall into one of
the two extremes: they either focus on the high-performance computational, or
on the interactive and open-ended aspects of the analysis. Arkouda is a
framework that attempts to integrate the interactive approach with the high
performance computation by using a novel client-server architecture, with a
Python interpreter on the client side for the interactions with the scientist
and a Chapel server for performing the demanding high-performance computations.
The Arkouda Python interpreter overloads the Python operators and transforms
them into messages to the Chapel server that performs the actual computation.
In this paper, we are proposing several client-side optimization techniques
for the Arkouda framework that maintain the interactive nature of the Arkouda
framework, but at the same time significantly improve the performance of the
programs that perform operations running on the high-performance Chapel server.
We do this by intercepting the Python operations in the interpreter, and
delaying their execution until the user requires the data, or we fill out the
instruction buffer. We implement caching and reuse of the Arkouda arrays on the
Chapel server side (thus saving on the allocation, initialization and
deallocation of the Chapel arrays), tracking and caching the results of
function calls on the Arkouda arrays (thus avoiding repeated computation) and
reusing the results of array operations by performing common subexpression
elimination.
We evaluate our approach on several Arkouda benchmarks and a large collection
of real-world and synthetic data inputs and show significant performance
improvements between 20% and 120% across the board, while fully maintaining the
interactive nature of the Arkouda framework.

    

### [[2006.04200] A Review of Incident Prediction, Resource Allocation, and Dispatch Models for Emergency Management](http://arxiv.org/abs/2006.04200)


  In the last fifty years, researchers have developed statistical, data-driven,
analytical, and algorithmic approaches for designing and improving emergency
response management (ERM) systems. The problem has been noted as inherently
difficult and constitutes spatio-temporal decision making under uncertainty,
which has been addressed in the literature with varying assumptions and
approaches. This survey provides a detailed review of these approaches,
focusing on the key challenges and issues regarding four sub-processes: (a)
incident prediction, (b) incident detection, (c) resource allocation, and (c)
computer-aided dispatch for emergency response. We highlight the strengths and
weaknesses of prior work in this domain and explore the similarities and
differences between different modeling paradigms. We conclude by illustrating
open challenges and opportunities for future research in this complex domain.

    

### [[2105.07782] A fast vectorized sorting implementation based on the ARM scalable vector extension (SVE)](http://arxiv.org/abs/2105.07782)


  The way developers implement their algorithms and how these implementations
behave on modern CPUs are governed by the design and organization of these. The
vectorization units (SIMD) are among the few CPUs' parts that can and must be
explicitly controlled. In the HPC community, the x86 CPUs and their
vectorization instruction sets were de-facto the standard for decades. Each new
release of an instruction set was usually a doubling of the vector length
coupled with new operations. Each generation was pushing for adapting and
improving previous implementations. The release of the ARM scalable vector
extension (SVE) changed things radically for several reasons. First, we expect
ARM processors to equip many supercomputers in the next years. Second, SVE's
interface is different in several aspects from the x86 extensions as it
provides different instructions, uses a predicate to control most operations,
and has a vector size that is only known at execution time. Therefore, using
SVE opens new challenges on how to adapt algorithms including the ones that are
already well-optimized on x86. In this paper, we port a hybrid sort based on
the well-known Quicksort and Bitonic-sort algorithms. We use a Bitonic sort to
process small partitions/arrays and a vectorized partitioning implementation to
divide the partitions. We explain how we use the predicates and how we manage
the non-static vector size. We explain how we efficiently implement the sorting
kernels. Our approach only needs an array of O(log N) for the recursive calls
in the partitioning phase, both in the sequential and in the parallel case. We
test the performance of our approach on a modern ARMv8.2 and assess the
different layers of our implementation by sorting/partitioning integers, double
floating-point numbers, and key/value pairs of integers. Our approach is faster
than the GNU C++ sort algorithm by a speedup factor of 4 on average.

    

### [[2111.09908] Visual Goal-Directed Meta-Learning with Contextual Planning Networks](http://arxiv.org/abs/2111.09908)


  The goal of meta-learning is to generalize to new tasks and goals as quickly
as possible. Ideally, we would like approaches that generalize to new goals and
tasks on the first attempt. Toward that end, we introduce contextual planning
networks (CPN). Tasks are represented as goal images and used to condition the
approach. We evaluate CPN along with several other approaches adapted for
zero-shot goal-directed meta-learning. We evaluate these approaches across 24
distinct manipulation tasks using Metaworld benchmark tasks. We found that CPN
outperformed several approaches and baselines on one task and was competitive
with existing approaches on others. We demonstrate the approach on a physical
platform on Jenga tasks using a Kinova Jaco robotic arm.

    

### [[2111.09934] Constraint-based Diversification of JOP Gadgets](http://arxiv.org/abs/2111.09934)


  Modern software deployment process produces software that is uniform and
hence vulnerable to large-scale code-reuse attacks, such as Jump-Oriented
Programming (JOP) attacks. Compiler-based diversification improves the
resilience of software systems by automatically generating different assembly
code versions of a given program. Existing techniques are efficient but do not
have a precise control over the quality of the generated variants. This paper
introduces Diversity by Construction (DivCon), a constraint-based approach to
software diversification. Unlike previous approaches, DivCon allows users to
control and adjust the conflicting goals of diversity and code quality. A key
enabler is the use of Large Neighborhood Search (LNS) to generate highly
diverse code efficiently. For larger problems, we propose a combination of LNS
with a structural decomposition of the problem. To further improve the
diversification efficiency of DivCon against JOP attacks, we propose an
application-specific distance measure tailored to the characteristics of JOP
attacks. We evaluate DivCon with 20 functions from a popular benchmark suite
for embedded systems. These experiments show that the combination of LNS and
our application-specific distance measure generates binary programs that are
highly resilient against JOP attacks. Our results confirm that there is a
trade-off between the quality of each assembly code version and the diversity
of the entire pool of versions. In particular, the experiments show that DivCon
generates near-optimal binary programs that share a small number of gadgets.
For constraint programming researchers and practitioners, this paper
demonstrates that LNS is a valuable technique for finding diverse solutions.
For security researchers and software engineers, DivCon extends the scope of
compiler-based diversification to performance-critical and resource-constrained
applications.

    

### [[2111.10005] Reinforcement Learning with Adaptive Curriculum Dynamics Randomization for Fault-Tolerant Robot Control](http://arxiv.org/abs/2111.10005)


  This study is aimed at addressing the problem of fault tolerance of quadruped
robots to actuator failure, which is critical for robots operating in remote or
extreme environments. In particular, an adaptive curriculum reinforcement
learning algorithm with dynamics randomization (ACDR) is established. The ACDR
algorithm can adaptively train a quadruped robot in random actuator failure
conditions and formulate a single robust policy for fault-tolerant robot
control. It is noted that the hard2easy curriculum is more effective than the
easy2hard curriculum for quadruped robot locomotion. The ACDR algorithm can be
used to build a robot system that does not require additional modules for
detecting actuator failures and switching policies. Experimental results show
that the ACDR algorithm outperforms conventional algorithms in terms of the
average reward and walking distance.

    

### [[2111.10044] Building a Question Answering System for the Manufacturing Domain](http://arxiv.org/abs/2111.10044)


  The design or simulation analysis of special equipment products must follow
the national standards, and hence it may be necessary to repeatedly consult the
contents of the standards in the design process. However, it is difficult for
the traditional question answering system based on keyword retrieval to give
accurate answers to technical questions. Therefore, we use natural language
processing techniques to design a question answering system for the
decision-making process in pressure vessel design. To solve the problem of
insufficient training data for the technology question answering system, we
propose a method to generate questions according to a declarative sentence from
several different dimensions so that multiple question-answer pairs can be
obtained from a declarative sentence. In addition, we designed an interactive
attention model based on a bidirectional long short-term memory (BiLSTM)
network to improve the performance of the similarity comparison of two question
sentences. Finally, the performance of the question answering system was tested
on public and technical domain datasets.

    

### [[2111.10056] Medical Visual Question Answering: A Survey](http://arxiv.org/abs/2111.10056)


  Medical Visual Question Answering (VQA) is a combination of medical
artificial intelligence and popular VQA challenges. Given a medical image and a
clinically relevant question in natural language, the medical VQA system is
expected to predict a plausible and convincing answer. Although the
general-domain VQA has been extensively studied, the medical VQA still needs
specific investigation and exploration due to its task features. In the first
part of this survey, we cover and discuss the publicly available medical VQA
datasets up to date about the data source, data quantity, and task feature. In
the second part, we review the approaches used in medical VQA tasks. In the
last part, we analyze some medical-specific challenges for the field and
discuss future research directions.

    

### [[2111.10061] An Activity-Based Model of Transport Demand for Greater Melbourne](http://arxiv.org/abs/2111.10061)


  In this paper, we present an algorithm for creating a synthetic population
for the Greater Melbourne area using a combination of machine learning,
probabilistic, and gravity-based approaches. We combine these techniques in a
hybrid model with three primary innovations: 1. when assigning activity
patterns, we generate individual activity chains for every agent, tailored to
their cohort; 2. when selecting destinations, we aim to strike a balance
between the distance-decay of trip lengths and the activity-based attraction of
destination locations; and 3. we take into account the number of trips
remaining for an agent so as to ensure they do not select a destination that
would be unreasonable to return home from. Our method is completely open and
replicable, requiring only publicly available data to generate a synthetic
population of agents compatible with commonly used agent-based modeling
software such as MATSim. The synthetic population was found to be accurate in
terms of distance distribution, mode choice, and destination choice for a
variety of population sizes.

    

### [[2111.10093] RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation](http://arxiv.org/abs/2111.10093)


  Cross-domain recommendation can help alleviate the data sparsity issue in
traditional sequential recommender systems. In this paper, we propose the
RecGURU algorithm framework to generate a Generalized User Representation (GUR)
incorporating user information across domains in sequential recommendation,
even when there is minimum or no common users in the two domains. We propose a
self-attentive autoencoder to derive latent user representations, and a domain
discriminator, which aims to predict the origin domain of a generated latent
representation. We propose a novel adversarial learning method to train the two
modules to unify user embeddings generated from different domains into a single
global GUR for each user. The learned GUR captures the overall preferences and
characteristics of a user and thus can be used to augment the behavior data and
improve recommendations in any single domain in which the user is involved.
Extensive experiments have been conducted on two public cross-domain
recommendation datasets as well as a large dataset collected from real-world
applications. The results demonstrate that RecGURU boosts performance and
outperforms various state-of-the-art sequential recommendation and cross-domain
recommendation methods. The collected data will be released to facilitate
future research.

    

### [[2111.10127] Neural Image Beauty Predictor Based on Bradley-Terry Model](http://arxiv.org/abs/2111.10127)


  Image beauty assessment is an important subject of computer vision.
Therefore, building a model to mimic the image beauty assessment becomes an
important task. To better imitate the behaviours of the human visual system
(HVS), a complete survey about images of different categories should be
implemented. This work focuses on image beauty assessment. In this study, the
pairwise evaluation method was used, which is based on the Bradley-Terry model.
We believe that this method is more accurate than other image rating methods
within an image group. Additionally, Convolution neural network (CNN), which is
fit for image quality assessment, is used in this work. The first part of this
study is a survey about the image beauty comparison of different images. The
Bradley-Terry model is used for the calculated scores, which are the target of
CNN model. The second part of this work focuses on the results of the image
beauty prediction, including landscape images, architecture images and portrait
images. The models are pretrained by the AVA dataset to improve the performance
later. Then, the CNN model is trained with the surveyed images and
corresponding scores. Furthermore, this work compares the results of four CNN
base networks, i.e., Alex net, VGG net, Squeeze net and LSiM net, as discussed
in literature. In the end, the model is evaluated by the accuracy in pairs,
correlation coefficient and relative error calculated by survey results.
Satisfactory results are achieved by our proposed methods with about 70 percent
accuracy in pairs. Our work sheds more light on the novel image beauty
assessment method. While more studies should be conducted, this method is a
promising step.

    

### [[2111.10280] A Hybrid Approach for an Interpretable and Explainable Intrusion Detection System](http://arxiv.org/abs/2111.10280)


  Cybersecurity has been a concern for quite a while now. In the latest years,
cyberattacks have been increasing in size and complexity, fueled by significant
advances in technology. Nowadays, there is an unavoidable necessity of
protecting systems and data crucial for business continuity. Hence, many
intrusion detection systems have been created in an attempt to mitigate these
threats and contribute to a timelier detection. This work proposes an
interpretable and explainable hybrid intrusion detection system, which makes
use of artificial intelligence methods to achieve better and more long-lasting
security. The system combines experts' written rules and dynamic knowledge
continuously generated by a decision tree algorithm as new shreds of evidence
emerge from network activity.

    

### [[2007.00714] Quantifying intrinsic causal contributions via structure preserving interventions](http://arxiv.org/abs/2007.00714)


  We propose a new notion of causal contribution which describes the
'intrinsic' part of the contribution of a node on a target node in a DAG. We
show that in some scenarios the existing causal quantification methods failed
to capture this notion exactly. By recursively writing each node as a function
of the upstream noise terms, we separate the intrinsic information added by
each node from the one obtained from its ancestors. To interpret the intrinsic
information as a causal contribution, we consider 'structure-preserving
interventions' that randomize each node in a way that mimics the usual
dependence on the parents and do not perturb the observed joint distribution.
To get a measure that is invariant across arbitrary orderings of nodes we
propose Shapley based symmetrization. We describe our contribution analysis for
variance and entropy, but contributions for other target metrics can be defined
analogously.

    

### [[2012.12130] A Computational Framework for Solving Nonlinear Binary OptimizationProblems in Robust Causal Inference](http://arxiv.org/abs/2012.12130)


  Identifying cause-effect relations among variables is a key step in the
decision-making process. While causal inference requires randomized
experiments, researchers and policymakers are increasingly using observational
studies to test causal hypotheses due to the wide availability of observational
data and the infeasibility of experiments. The matching method is the most used
technique to make causal inference from observational data. However, the pair
assignment process in one-to-one matching creates uncertainty in the inference
because of different choices made by the experimenter. Recently, discrete
optimization models are proposed to tackle such uncertainty. Although a robust
inference is possible with discrete optimization models, they produce nonlinear
problems and lack scalability. In this work, we propose greedy algorithms to
solve the robust causal inference test instances from observational data with
continuous outcomes. We propose a unique framework to reformulate the nonlinear
binary optimization problems as feasibility problems. By leveraging the
structure of the feasibility formulation, we develop greedy schemes that are
efficient in solving robust test problems. In many cases, the proposed
algorithms achieve global optimal solutions. We perform experiments on three
real-world datasets to demonstrate the effectiveness of the proposed algorithms
and compare our result with the state-of-the-art solver. Our experiments show
that the proposed algorithms significantly outperform the exact method in terms
of computation time while achieving the same conclusion for causal tests. Both
numerical experiments and complexity analysis demonstrate that the proposed
algorithms ensure the scalability required for harnessing the power of big data
in the decision-making process.

    

### [[2101.07217] Is it a great Autonomous FX Trading Strategy or you are just fooling yourself](http://arxiv.org/abs/2101.07217)


  In this paper, we propose a method for evaluating autonomous trading
strategies that provides realistic expectations, regarding the strategy's
long-term performance. This method addresses This method addresses many
pitfalls that currently fool even experienced software developers and
researchers, not to mention the customers that purchase these products. We
present the results of applying our method to several famous autonomous trading
strategies, which are used to manage a diverse selection of financial assets.
The results show that many of these published strategies are far from being
reliable vehicles for financial investment. Our method exposes the difficulties
involved in building a reliable, long-term strategy and provides a means to
compare potential strategies and select the most promising one by establishing
minimal periods and requirements for the test executions. There are many
developers that create software to buy and sell financial assets autonomously
and some of them present great performance when simulating with historical
price series (commonly called backtests). Nevertheless, when these strategies
are used in real markets (or data not used in their training or evaluation),
quite often they perform very poorly. The proposed method can be used to
evaluate potential strategies. In this way, the method helps to tell if you
really have a great trading strategy or you are just fooling yourself.

    

### [[2103.06854] A conditional, a fuzzy and a probabilistic interpretation of self-organising maps](http://arxiv.org/abs/2103.06854)


  In this paper we establish a link between fuzzy and preferential semantics
for description logics and Self-Organising Maps, which have been proposed as
possible candidates to explain the psychological mechanisms underlying category
generalisation. In particular, we show that the input/output behavior of a
Self-Organising Map after training can be described by a fuzzy description
logic interpretation as well as by a preferential interpretation, based on a
concept-wise multipreference semantics, which takes into account preferences
with respect to different concepts and has been recently proposed for ranked
and for weighted defeasible description logics. Properties of the network can
be proven by model checking on the fuzzy or on the preferential interpretation.
Starting from the fuzzy interpretation, we also provide a probabilistic account
for this neural network model.

    

### [[2004.12859] Static Race Detection and Mutex Safety and Liveness for Go Programs (extended version)](http://arxiv.org/abs/2004.12859)


  Go is a popular concurrent programming language thanks to its ability to
efficiently combine concurrency and systems programming. In Go programs, a
number of concurrency bugs can be caused by a mixture of data races and
communication problems. In this paper, we develop a theory based on behavioural
types to statically detect data races and deadlocks in Go programs. We first
specify lock safety and liveness and data race properties over a Go program
model, using the happens-before relation defined in the Go memory model. We
represent these properties of programs in a $\mu$-calculus model of types, and
validate them using type-level model-checking. We then extend the framework to
account for Go's channels, and implement a static verification tool which can
detect concurrency errors. This is, to the best of our knowledge, the first
static verification framework of this kind for the Go language, uniformly
analysing concurrency errors caused by a mix of shared memory accesses and
asynchronous message-passing communications.

    