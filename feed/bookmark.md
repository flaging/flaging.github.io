
## 2021-8-27

### [[2108.11472] Bandwidth Allocation and Service Differentiation in D2D Wireless Networks](http://arxiv.org/abs/2108.11472)


  Inspired by a new feature in 5G NR called bandwidth part (BWP), this paper
presents a bandwidth allocation (BA) model that allows one to adapt the
bandwidth allocated to users depending on their data rate needs. Specifically,
in adaptive BA, a wide bandwidth is divided into chunks of smaller bandwidths
and the number of bandwidth chunks allocated to a user depends on its needs or
type. Although BWP in 5G NR mandates allocation of a set of contiguous
bandwidth chunks, our BA model also allows other assumptions on chunk
allocation such as the allocation of any set of bandwidth chunks, as in, e.g.,
LTE resource allocation, where chunks are selected uniformly at random. The BA
model studied here is probabilistic in that the user locations are assumed to
form a realization of a Poisson point process and each user decides
independently to be of a certain type with some probability. This model allows
one to quantify spectrum sharing and service differentiation in this context,
namely to predict what performance a user gets depending on its type as well as
the overall performance. This is based on exact representations of key
performance metrics for each user type, namely its success probability, the
meta distribution of its signal-to-interference ratio, and its Shannon
throughput. We show that, surprisingly, the higher traffic variability stemming
from adaptive BA is beneficial: when comparing two networks using adaptive BA
and having the same mean signal and the same mean interference powers, the
network with higher traffic variability performs better for all these
performance metrics. With respect to Shannon throughput, we observe that our BA
model is roughly egalitarian per Hertz and leads to a linear service
differentiation in aggregated throughput value.

    

### [[2108.11697] Revenue Maximization through Cell Switching and Spectrum Leasing in 5G HetNets](http://arxiv.org/abs/2108.11697)


  One of the ways of achieving improved capacity in mobile cellular networks is
via network densification. Even though densification increases the capacity of
the network, it also leads to increased energy consumption which can be curbed
by dynamically switching off some base stations (BSs) during periods of low
traffic. However, dynamic cell switching has the challenge of spectrum
under-utilizationas the spectrum originally occupied by the BSs that are turned
off remains dormant. This dormant spectrum can be leased by the primary network
(PN) operators, who hold the license, to the secondary network (SN) operators
who cannot afford to purchase the spectrum license. Thus enabling the PN to
gain additional revenue from spectrum leasing as well as from electricity cost
savings due to reduced energy consumption. Therefore, in this work, we propose
a cell switching and spectrum leasing framework based on simulated annealing
(SA) algorithm to maximize the revenue of the PN while respecting the
quality-of-service constraints. The performance evaluation reveals that the
proposed method is very close to optimal exhaustive search method with a
significant reduction in the computation complexity.

    

### [[2108.11807] Human readable network troubleshooting based on anomaly detection and feature scoring](http://arxiv.org/abs/2108.11807)


  Network troubleshooting is still a heavily human-intensive process. To reduce
the time spent by human operators in the diagnosis process, we present a system
based on (i) unsupervised learning methods for detecting anomalies in the time
domain, (ii) an attention mechanism to rank features in the feature space and
finally (iii) an expert knowledge module able to seamlessly incorporate
previously collected domain-knowledge. In this paper, we thoroughly evaluate
the performance of the full system and of its individual building blocks:
particularly, we consider (i) 10 anomaly detection algorithms as well as (ii)
10 attention mechanisms, that comprehensively represent the current state of
the art in the respective fields. Leveraging a unique collection of
expert-labeled datasets worth several months of real router telemetry data, we
perform a thorough performance evaluation contrasting practical results in
constrained stream-mode settings, with the results achievable by an ideal
oracle in academic settings. Our experimental evaluation shows that (i) the
proposed system is effective in achieving high levels of agreement with the
expert, and (ii) that even a simple statistical approach is able to extract
useful information from expert knowledge gained in past cases, significantly
improving troubleshooting performance.

    

### [[2108.11861] Security and privacy for 6G: A survey on prospective technologies and challenges](http://arxiv.org/abs/2108.11861)


  Sixth-generation (6G) mobile networks will have to cope with diverse threats
on a space-air-ground integrated network environment, novel technologies, and
an accessible user information explosion. However, for now, security and
privacy issues for 6G remain largely in concept. This survey provides a
systematic overview of security and privacy issues based on prospective
technologies for 6G in the physical, connection, and service layers, as well as
through lessons learned from the failures of existing security architectures
and state-of-the-art defenses. Two key lessons learned are as follows. First,
other than inheriting vulnerabilities from the previous generations, 6G has new
threat vectors from new radio technologies, such as the exposed location of
radio stripes in ultra-massive MIMO systems at Terahertz bands and attacks
against pervasive intelligence. Second, physical layer protection, deep network
slicing, quantum-safe communications, artificial intelligence (AI) security,
platform-agnostic security, real-time adaptive security, and novel data
protection mechanisms such as distributed ledgers and differential privacy are
the top promising techniques to mitigate the attack magnitude and personal data
breaches substantially.

    

### [[2108.11879] Green Internet of Vehicles (IoV) in the 6G Era: Toward Sustainable Vehicular Communications and Networking](http://arxiv.org/abs/2108.11879)


  As one of the most promising applications in future Internet of Things,
Internet of Vehicles (IoV) has been acknowledged as a fundamental technology
for developing the Intelligent Transportation Systems in smart cities. With the
emergence of the sixth generation (6G) communications technologies, massive
network infrastructures will be densely deployed and the number of network
nodes will increase exponentially, leading to extremely high energy
consumption. There has been an upsurge of interest to develop the green IoV
towards sustainable vehicular communication and networking in the 6G era. In
this paper, we present the main considerations for green IoV from five
different scenarios, including the communication, computation, traffic,
Electric Vehicles (EVs), and energy harvesting management. The literatures
relevant to each of the scenarios are compared from the perspective of energy
optimization (e.g., with respect to resource allocation, workload scheduling,
routing design, traffic control, charging management, energy harvesting and
sharing, etc.) and the related factors affecting energy efficiency (e.g.,
resource limitation, channel state, network topology, traffic condition, etc.).
In addition, we introduce the potential challenges and the emerging
technologies in 6G for developing green IoV systems. Finally, we discuss the
research trends in designing energy-efficient IoV systems.

    

### [[2108.11374] Machine Learning for Sensor Transducer Conversion Routines](http://arxiv.org/abs/2108.11374)


  Sensors with digital outputs require software conversion routines to
transform the unitless ADC samples to physical quantities with the correct
units. These conversion routines are computationally complex given the limited
computational resources of low-power embedded systems. This article presents a
set of machine learning methods to learn new, less-complex conversion routines
that do not sacrifice accuracy for the BME680 environmental sensor. We present
a Pareto analysis of the tradeoff between accuracy and computational overhead
for the models and present models that reduce the computational overhead of the
existing industry-standard conversion routines for temperature, pressure, and
humidity by 62 %, 71 %, and 18 % respectively. The corresponding RMS errors for
these methods are 0.0114 $^\circ$C, 0.0280 KPa, and 0.0337 %. These results
show that machine learning methods for learning conversion routines can produce
conversion routines with reduced computational overhead while maintaining good
accuracy.

    

### [[2108.11417] Unsupervised Reservoir Computing for Solving Ordinary Differential Equations](http://arxiv.org/abs/2108.11417)


  There is a wave of interest in using unsupervised neural networks for solving
differential equations. The existing methods are based on feed-forward
networks, {while} recurrent neural network differential equation solvers have
not yet been reported. We introduce an unsupervised reservoir computing (RC),
an echo-state recurrent neural network capable of discovering approximate
solutions that satisfy ordinary differential equations (ODEs). We suggest an
approach to calculate time derivatives of recurrent neural network outputs
without using backpropagation. The internal weights of an RC are fixed, while
only a linear output layer is trained, yielding efficient training. However, RC
performance strongly depends on finding the optimal hyper-parameters, which is
a computationally expensive process. We use Bayesian optimization to
efficiently discover optimal sets in a high-dimensional hyper-parameter space
and numerically show that one set is robust and can be used to solve an ODE for
different initial conditions and time ranges. A closed-form formula for the
optimal output weights is derived to solve first order linear equations in a
backpropagation-free learning process. We extend the RC approach by solving
nonlinear system of ODEs using a hybrid optimization method consisting of
gradient descent and Bayesian optimization. Evaluation of linear and nonlinear
systems of equations demonstrates the efficiency of the RC ODE solver.

    

### [[2108.11430] Towards Memory-Efficient Neural Networks via Multi-Level in situ Generation](http://arxiv.org/abs/2108.11430)


  Deep neural networks (DNN) have shown superior performance in a variety of
tasks. As they rapidly evolve, their escalating computation and memory demands
make it challenging to deploy them on resource-constrained edge devices. Though
extensive efficient accelerator designs, from traditional electronics to
emerging photonics, have been successfully demonstrated, they are still
bottlenecked by expensive memory accesses due to tremendous gaps between the
bandwidth/power/latency of electrical memory and computing cores. Previous
solutions fail to fully-leverage the ultra-fast computational speed of emerging
DNN accelerators to break through the critical memory bound. In this work, we
propose a general and unified framework to trade expensive memory transactions
with ultra-fast on-chip computations, directly translating to performance
improvement. We are the first to jointly explore the intrinsic correlations and
bit-level redundancy within DNN kernels and propose a multi-level in situ
generation mechanism with mixed-precision bases to achieve on-the-fly recovery
of high-resolution parameters with minimum hardware overhead. Extensive
experiments demonstrate that our proposed joint method can boost the memory
efficiency by 10-20x with comparable accuracy over four state-of-the-art
designs, when benchmarked on ResNet-18/DenseNet-121/MobileNetV2/V3 with various
tasks.

    

### [[2108.11436] Integrated Speech and Gesture Synthesis](http://arxiv.org/abs/2108.11436)


  Text-to-speech and co-speech gesture synthesis have until now been treated as
separate areas by two different research communities, and applications merely
stack the two technologies using a simple system-level pipeline. This can lead
to modeling inefficiencies and may introduce inconsistencies that limit the
achievable naturalness. We propose to instead synthesize the two modalities in
a single model, a new problem we call integrated speech and gesture synthesis
(ISG). We also propose a set of models modified from state-of-the-art neural
speech-synthesis engines to achieve this goal. We evaluate the models in three
carefully-designed user studies, two of which evaluate the synthesized speech
and gesture in isolation, plus a combined study that evaluates the models like
they will be used in real-world applications -- speech and gesture presented
together. The results show that participants rate one of the proposed
integrated synthesis models as being as good as the state-of-the-art pipeline
system we compare against, in all three tests. The model is able to achieve
this with faster synthesis time and greatly reduced parameter count compared to
the pipeline system, illustrating some of the potential benefits of treating
speech and gesture synthesis together as a single, unified problem. Videos and
code are available on our project page at this https URL


### [[2108.11463] With One Voice: Composing a Travel Voice Assistant from Re-purposed Models](http://arxiv.org/abs/2108.11463)


  Voice assistants provide users a new way of interacting with digital
products, allowing them to retrieve information and complete tasks with an
increased sense of control and flexibility. Such products are comprised of
several machine learning models, like Speech-to-Text transcription, Named
Entity Recognition and Resolution, and Text Classification. Building a voice
assistant from scratch takes the prolonged efforts of several teams
constructing numerous models and orchestrating between components. Alternatives
such as using third-party vendors or re-purposing existing models may be
considered to shorten time-to-market and development costs. However, each
option has its benefits and drawbacks. We present key insights from building a
voice search assistant for this http URL search and recommendation system. Our
paper compares the achieved performance and development efforts in dedicated
tailor-made solutions against existing re-purposed models. We share and discuss
our data-driven decisions about implementation trade-offs and their estimated
outcomes in hindsight, showing that a fully functional machine learning product
can be built from existing models.

    

### [[2108.11468] SomnNET: An SpO2 Based Deep Learning Network for Sleep Apnea Detection in Smartwatches](http://arxiv.org/abs/2108.11468)


  The abnormal pause or rate reduction in breathing is known as the sleep-apnea
hypopnea syndrome and affects the quality of sleep of an individual. A novel
method for the detection of sleep apnea events (pause in breathing) from
peripheral oxygen saturation (SpO2) signals obtained from wearable devices is
discussed in this paper. The paper details an apnea detection algorithm of a
very high resolution on a per-second basis for which a 1-dimensional
convolutional neural network -- which we termed SomnNET -- is developed. This
network exhibits an accuracy of 97.08% and outperforms several lower resolution
state-of-the-art apnea detection methods. The feasibility of model pruning and
binarization to reduce the computational complexity is explored. The pruned
network with 80% sparsity exhibited an accuracy of 89.75%, and the binarized
network exhibited an accuracy of 68.22%. The performance of the proposed
networks is compared against several state-of-the-art algorithms.

    

### [[2108.11481] Learning to discover: expressive Gaussian mixture models for multi-dimensional simulation and parameter inference in the physical sciences](http://arxiv.org/abs/2108.11481)


  We show that density models describing multiple observables with (i) hard
boundaries and (ii) dependence on external parameters may be created using an
auto-regressive Gaussian mixture model. The model is designed to capture how
observable spectra are deformed by hypothesis variations, and is made more
expressive by projecting data onto a configurable latent space. It may be used
as a statistical model for scientific discovery in interpreting experimental
observations, for example when constraining the parameters of a physical model
or tuning simulation parameters according to calibration data. The model may
also be sampled for use within a Monte Carlo simulation chain, or used to
estimate likelihood ratios for event classification. The method is demonstrated
on simulated high-energy particle physics data considering the anomalous
electroweak production of a $Z$ boson in association with a dijet system at the
Large Hadron Collider, and the accuracy of inference is tested using a
realistic toy example. The developed methods are domain agnostic; they may be
used within any field to perform simulation or inference where a dataset
consisting of many real-valued observables has conditional dependence on
external parameters.

    

### [[2108.11482] ETA Prediction with Graph Neural Networks in Google Maps](http://arxiv.org/abs/2108.11482)


  Travel-time prediction constitutes a task of high importance in
transportation networks, with web mapping services like Google Maps regularly
serving vast quantities of travel time queries from users and enterprises
alike. Further, such a task requires accounting for complex spatiotemporal
interactions (modelling both the topological properties of the road network and
anticipating events -- such as rush hours -- that may occur in the future).
Hence, it is an ideal target for graph representation learning at scale. Here
we present a graph neural network estimator for estimated time of arrival (ETA)
which we have deployed in production at Google Maps. While our main
architecture consists of standard GNN building blocks, we further detail the
usage of training schedule methods such as MetaGradients in order to make our
model robust and production-ready. We also provide prescriptive studies:
ablating on various architectural decisions and training regimes, and
qualitative analyses on real-world situations where our model provides a
competitive edge. Our GNN proved powerful when deployed, significantly reducing
negative ETA outcomes in several regions compared to the previous production
baseline (40+% in cities like Sydney).

    

### [[2108.11483] Heavy-tailed Streaming Statistical Estimation](http://arxiv.org/abs/2108.11483)


  We consider the task of heavy-tailed statistical estimation given streaming
$p$-dimensional samples. This could also be viewed as stochastic optimization
under heavy-tailed distributions, with an additional $O(p)$ space complexity
constraint. We design a clipped stochastic gradient descent algorithm and
provide an improved analysis, under a more nuanced condition on the noise of
the stochastic gradients, which we show is critical when analyzing stochastic
optimization problems arising from general statistical estimation problems. Our
results guarantee convergence not just in expectation but with exponential
concentration, and moreover does so using $O(1)$ batch size. We provide
consequences of our results for mean estimation and linear regression. Finally,
we provide empirical corroboration of our results and algorithms via synthetic
experiments for mean estimation and linear regression.

    

### [[2108.11489] The Interplay Between Implicit Bias and Benign Overfitting in Two-Layer Linear Networks](http://arxiv.org/abs/2108.11489)


  The recent success of neural network models has shone light on a rather
surprising statistical phenomenon: statistical models that perfectly fit noisy
data can generalize well to unseen test data. Understanding this phenomenon of
$\textit{benign overfitting}$ has attracted intense theoretical and empirical
study. In this paper, we consider interpolating two-layer linear neural
networks trained with gradient flow on the squared loss and derive bounds on
the excess risk when the covariates satisfy sub-Gaussianity and
anti-concentration properties, and the noise is independent and sub-Gaussian.
By leveraging recent results that characterize the implicit bias of this
estimator, our bounds emphasize the role of both the quality of the
initialization as well as the properties of the data covariance matrix in
achieving low excess risk.

    

### [[2108.11498] Physics-informed neural networks for improving cerebral hemodynamics predictions](http://arxiv.org/abs/2108.11498)


  Determining brain hemodynamics plays a critical role in the diagnosis and
treatment of various cerebrovascular diseases. In this work, we put forth a
physics-informed deep learning framework that augments sparse clinical
measurements with fast computational fluid dynamics (CFD) simulations to
generate physically consistent and high spatiotemporal resolution of brain
hemodynamic parameters. Transcranial Doppler (TCD) ultrasound is one of the
most common techniques in the current clinical workflow that enables
noninvasive and instantaneous evaluation of blood flow velocity within the
cerebral arteries. However, it is spatially limited to only a handful of
locations across the cerebrovasculature due to the constrained accessibility
through the skull's acoustic windows. Our deep learning framework employs
in-vivo real-time TCD velocity measurements at several locations in the brain
and the baseline vessel cross-sectional areas acquired from 3D angiography
images, and provides high-resolution maps of velocity, area, and pressure in
the entire vasculature. We validated the predictions of our model against
in-vivo velocity measurements obtained via 4D flow MRI scans. We then showcased
the clinical significance of this technique in diagnosing the cerebral
vasospasm (CVS) by successfully predicting the changes in vasospastic local
vessel diameters based on corresponding sparse velocities measurements. The key
finding here is that the combined effects of uncertainties in outlet boundary
condition subscription and modeling physics deficiencies render the
conventional purely physics-based computational models unsuccessful in
recovering accurate brain hemodynamics. Nonetheless, fusing these models with
clinical measurements through a data-driven approach ameliorates predictions of
brain hemodynamic variables.

    

### [[2108.11505] Generalized Real-World Super-Resolution through Adversarial Robustness](http://arxiv.org/abs/2108.11505)


  Real-world Super-Resolution (SR) has been traditionally tackled by first
learning a specific degradation model that resembles the noise and corruption
artifacts in low-resolution imagery. Thus, current methods lack generalization
and lose their accuracy when tested on unseen types of corruption. In contrast
to the traditional proposal, we present Robust Super-Resolution (RSR), a method
that leverages the generalization capability of adversarial attacks to tackle
real-world SR. Our novel framework poses a paradigm shift in the development of
real-world SR methods. Instead of learning a dataset-specific degradation, we
employ adversarial attacks to create difficult examples that target the model's
weaknesses. Afterward, we use these adversarial examples during training to
improve our model's capacity to process noisy inputs. We perform extensive
experimentation on synthetic and real-world images and empirically demonstrate
that our RSR method generalizes well across datasets without re-training for
specific noise priors. By using a single robust model, we outperform
state-of-the-art specialized methods on real-world benchmarks.

    

### [[2108.11513] Learning Effective and Efficient Embedding via an Adaptively-Masked Twins-based Layer](http://arxiv.org/abs/2108.11513)


  Embedding learning for categorical features is crucial for the deep
learning-based recommendation models (DLRMs). Each feature value is mapped to
an embedding vector via an embedding learning process. Conventional methods
configure a fixed and uniform embedding size to all feature values from the
same feature field. However, such a configuration is not only sub-optimal for
embedding learning but also memory costly. Existing methods that attempt to
resolve these problems, either rule-based or neural architecture search
(NAS)-based, need extensive efforts on the human design or network training.
They are also not flexible in embedding size selection or in warm-start-based
applications. In this paper, we propose a novel and effective embedding size
selection scheme. Specifically, we design an Adaptively-Masked Twins-based
Layer (AMTL) behind the standard embedding layer. AMTL generates a mask vector
to mask the undesired dimensions for each embedding vector. The mask vector
brings flexibility in selecting the dimensions and the proposed layer can be
easily added to either untrained or trained DLRMs. Extensive experimental
evaluations show that the proposed scheme outperforms competitive baselines on
all the benchmark tasks, and is also memory-efficient, saving 60\% memory usage
without compromising any performance metrics.

    

### [[2108.11514] Bilateral Denoising Diffusion Models](http://arxiv.org/abs/2108.11514)


  Denoising diffusion probabilistic models (DDPMs) have emerged as competitive
generative models yet brought challenges to efficient sampling. In this paper,
we propose novel bilateral denoising diffusion models (BDDMs), which take
significantly fewer steps to generate high-quality samples. From a bilateral
modeling objective, BDDMs parameterize the forward and reverse processes with a
score network and a scheduling network, respectively. We show that a new lower
bound tighter than the standard evidence lower bound can be derived as a
surrogate objective for training the two networks. In particular, BDDMs are
efficient, simple-to-train, and capable of further improving any pre-trained
DDPM by optimizing the inference noise schedules. Our experiments demonstrated
that BDDMs can generate high-fidelity samples with as few as 3 sampling steps
and produce comparable or even higher quality samples than DDPMs using 1000
steps with only 16 sampling steps (a 62x speedup).

    

### [[2108.11523] SOMTimeS: Self Organizing Maps for Time Series Clustering and its Application to Serious Illness Conversations](http://arxiv.org/abs/2108.11523)


  There is an increasing demand for scalable algorithms capable of clustering
and analyzing large time series datasets. The Kohonen self-organizing map (SOM)
is a type of unsupervised artificial neural network for visualizing and
clustering complex data, reducing the dimensionality of data, and selecting
influential features. Like all clustering methods, the SOM requires a measure
of similarity between input data (in this work time series). Dynamic time
warping (DTW) is one such measure, and a top performer given that it
accommodates the distortions when aligning time series. Despite its use in
clustering, DTW is limited in practice because it is quadratic in runtime
complexity with the length of the time series data. To address this, we present
a new DTW-based clustering method, called SOMTimeS (a Self-Organizing Map for
TIME Series), that scales better and runs faster than other DTW-based
clustering algorithms, and has similar performance accuracy. The computational
performance of SOMTimeS stems from its ability to prune unnecessary DTW
computations during the SOM's training phase. We also implemented a similar
pruning strategy for K-means for comparison with one of the top performing
clustering algorithms. We evaluated the pruning effectiveness, accuracy,
execution time and scalability on 112 benchmark time series datasets from the
University of California, Riverside classification archive. We showed that for
similar accuracy, the speed-up achieved for SOMTimeS and K-means was 1.8x on
average; however, rates varied between 1x and 18x depending on the dataset.
SOMTimeS and K-means pruned 43% and 50% of the total DTW computations,
respectively. We applied SOMtimeS to natural language conversation data
collected as part of a large healthcare cohort study of patient-clinician
serious illness conversations to demonstrate the algorithm's utility with
complex, temporally sequenced phenomena.

    

### [[2108.11530] A New Interpolation Approach and Corresponding Instance-Based Learning](http://arxiv.org/abs/2108.11530)


  Starting from finding approximate value of a function, introduces the measure
of approximation-degree between two numerical values, proposes the concepts of
"strict approximation" and "strict approximation region", then, derives the
corresponding one-dimensional interpolation methods and formulas, and then
presents a calculation model called "sum-times-difference formula" for
high-dimensional interpolation, thus develops a new interpolation approach,
that is, ADB interpolation. ADB interpolation is applied to the interpolation
of actual functions with satisfactory results. Viewed from principle and
effect, the interpolation approach is of novel idea, and has the advantages of
simple calculation, stable accuracy, facilitating parallel processing, very
suiting for high-dimensional interpolation, and easy to be extended to the
interpolation of vector valued functions. Applying the approach to
instance-based learning, a new instance-based learning method, learning using
ADB interpolation, is obtained. The learning method is of unique technique,
which has also the advantages of definite mathematical basis, implicit distance
weights, avoiding misclassification, high efficiency, and wide range of
applications, as well as being interpretable, etc. In principle, this method is
a kind of learning by analogy, which and the deep learning that belongs to
inductive learning can complement each other, and for some problems, the two
can even have an effect of "different approaches but equal results" in big data
and cloud computing environment. Thus, the learning using ADB interpolation can
also be regarded as a kind of "wide learning" that is dual to deep learning.

    

### [[2108.11535] ChessMix: Spatial Context Data Augmentation for Remote Sensing Semantic Segmentation](http://arxiv.org/abs/2108.11535)


  Labeling semantic segmentation datasets is a costly and laborious process if
compared with tasks like image classification and object detection. This is
especially true for remote sensing applications that not only work with
extremely high spatial resolution data but also commonly require the knowledge
of experts of the area to perform the manual labeling. Data augmentation
techniques help to improve deep learning models under the circumstance of few
and imbalanced labeled samples. In this work, we propose a novel data
augmentation method focused on exploring the spatial context of remote sensing
semantic segmentation. This method, ChessMix, creates new synthetic images from
the existing training set by mixing transformed mini-patches across the dataset
in a chessboard-like grid. ChessMix prioritizes patches with more examples of
the rarest classes to alleviate the imbalance problems. The results in three
diverse well-known remote sensing datasets show that this is a promising
approach that helps to improve the networks' performance, working especially
well in datasets with few available data. The results also show that ChessMix
is capable of improving the segmentation of objects with few labeled pixels
when compared to the most common data augmentation methods widely used.

    

### [[2108.11550] The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation](http://arxiv.org/abs/2108.11550)


  It is fundamental for personal robots to reliably navigate to a specified
goal. To study this task, PointGoal navigation has been introduced in simulated
Embodied AI environments. Recent advances solve this PointGoal navigation task
with near-perfect accuracy (99.6% success) in photo-realistically simulated
environments, assuming noiseless egocentric vision, noiseless actuation, and
most importantly, perfect localization. However, under realistic noise models
for visual sensors and actuation, and without access to a "GPS and Compass
sensor," the 99.6%-success agents for PointGoal navigation only succeed with
0.3%. In this work, we demonstrate the surprising effectiveness of visual
odometry for the task of PointGoal navigation in this realistic setting, i.e.,
with realistic noise models for perception and actuation and without access to
GPS and Compass sensors. We show that integrating visual odometry techniques
into navigation policies improves the state-of-the-art on the popular Habitat
PointNav benchmark by a large margin, improving success from 64.5% to 71.7%
while executing 6.4 times faster.

    

### [[2108.11556] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](http://arxiv.org/abs/2108.11556)


  We propose a latent space energy-based prior model for text generation and
classification. The model stands on a generator network that generates the text
sequence based on a continuous latent vector. The energy term of the prior
model couples a continuous latent vector and a symbolic one-hot vector, so that
discrete category can be inferred from the observed example based on the
continuous latent vector. Such a latent space coupling naturally enables
incorporation of information bottleneck regularization to encourage the
continuous latent vector to extract information from the observed example that
is informative of the underlying category. In our learning method, the
symbol-vector coupling, the generator network and the inference network are
learned jointly. Our model can be learned in an unsupervised setting where no
category labels are provided. It can also be learned in semi-supervised setting
where category labels are provided for a subset of training examples. Our
experiments demonstrate that the proposed model learns well-structured and
meaningful latent space, which (1) guides the generator to generate text with
high quality, diversity, and interpretability, and (2) effectively classifies
text.

    

### [[2108.11561] CoSEM: Contextual and Semantic Embedding for App Usage Prediction](http://arxiv.org/abs/2108.11561)


  App usage prediction is important for smartphone system optimization to
enhance user experience. Existing modeling approaches utilize historical app
usage logs along with a wide range of semantic information to predict the app
usage; however, they are only effective in certain scenarios and cannot be
generalized across different situations. This paper address this problem by
developing a model called Contextual and Semantic Embedding model for App Usage
Prediction (CoSEM) for app usage prediction that leverages integration of 1)
semantic information embedding and 2) contextual information embedding based on
historical app usage of individuals. Extensive experiments show that the
combination of semantic information and history app usage information enables
our model to outperform the baselines on three real-world datasets, achieving
an MRR score over 0.55,0.57,0.86 and Hit rate scores of more than 0.71, 0.75,
and 0.95, respectively.

    

### [[2108.11563] Adaptive Control of Differentially Private Linear Quadratic Systems](http://arxiv.org/abs/2108.11563)


  In this paper, we study the problem of regret minimization in reinforcement
learning (RL) under differential privacy constraints. This work is motivated by
the wide range of RL applications for providing personalized service, where
privacy concerns are becoming paramount. In contrast to previous works, we take
the first step towards non-tabular RL settings, while providing a rigorous
privacy guarantee. In particular, we consider the adaptive control of
differentially private linear quadratic (LQ) systems. We develop the first
private RL algorithm, PRL, which is able to attain a sub-linear regret while
guaranteeing privacy protection. More importantly, the additional cost due to
privacy is only on the order of $\frac{\ln(1/\delta)^{1/4}}{\epsilon^{1/2}}$
given privacy parameters $\epsilon, \delta > 0$. Through this process, we also
provide a general procedure for adaptive control of LQ systems under changing
regularizers, which not only generalizes previous non-private controls, but
also serves as the basis for general private controls.

    

### [[2108.11569] Robust Long-Tailed Learning under Label Noise](http://arxiv.org/abs/2108.11569)


  Long-tailed learning has attracted much attention recently, with the goal of
improving generalisation for tail classes. Most existing works use supervised
learning without considering the prevailing noise in the training dataset. To
move long-tailed learning towards more realistic scenarios, this work
investigates the label noise problem under long-tailed label distribution. We
first observe the negative impact of noisy labels on the performance of
existing methods, revealing the intrinsic challenges of this problem. As the
most commonly used approach to cope with noisy labels in previous literature,
we then find that the small-loss trick fails under long-tailed label
distribution. The reason is that deep neural networks cannot distinguish
correctly-labeled and mislabeled examples on tail classes. To overcome this
limitation, we establish a new prototypical noise detection method by designing
a distance-based metric that is resistant to label noise. Based on the above
findings, we propose a robust framework,~\algo, that realizes noise detection
for long-tailed learning, followed by soft pseudo-labeling via both label
smoothing and diverse label guessing. Moreover, our framework can naturally
leverage semi-supervised learning algorithms to further improve the
generalisation. Extensive experiments on benchmark and real-world datasets
demonstrate the superiority of our methods over existing baselines. In
particular, our method outperforms DivideMix by 3\% in test accuracy. Source
code will be released soon.

    

### [[2108.11571] GNNSampler: Bridging the Gap between Sampling Algorithms of GNN and Hardware](http://arxiv.org/abs/2108.11571)


  Sampling is a critical operation in the training of Graph Neural Network
(GNN) that helps reduce the cost. Previous works have explored improving
sampling algorithms through mathematical and statistical methods. However,
there is a gap between sampling algorithms and hardware. Without consideration
of hardware, algorithm designers merely optimize sampling at the algorithm
level, missing the great potential of promoting the efficiency of existing
sampling algorithms by leveraging hardware features. In this paper, we first
propose a unified programming model for mainstream sampling algorithms, termed
GNNSampler, covering the key processes for sampling algorithms in various
categories. Second, we explore the data locality among nodes and their
neighbors (i.e., the hardware feature) in real-world datasets for alleviating
the irregular memory access in sampling. Third, we implement locality-aware
optimizations in GNNSampler for diverse sampling algorithms to optimize the
general sampling process in the training of GNN. Finally, we emphatically
conduct experiments on large graph datasets to analyze the relevance between
the training time, model accuracy, and hardware-level metrics, which helps
achieve a good trade-off between time and accuracy in GNN training. Extensive
experimental results show that our method is universal to mainstream sampling
algorithms and reduces the training time of GNN (range from 4.83% with
layer-wise sampling to 44.92% with subgraph-based sampling) with comparable
accuracy.

    

### [[2108.11577] Machine Unlearning of Features and Labels](http://arxiv.org/abs/2108.11577)


  Removing information from a machine learning model is a non-trivial task that
requires to partially revert the training process. This task is unavoidable
when sensitive data, such as credit card numbers or passwords, accidentally
enter the model and need to be removed afterwards. Recently, different concepts
for machine unlearning have been proposed to address this problem. While these
approaches are effective in removing individual data points, they do not scale
to scenarios where larger groups of features and labels need to be reverted.
In this paper, we propose a method for unlearning features and labels. Our
approach builds on the concept of influence functions and realizes unlearning
through closed-form updates of model parameters. It enables to adapt the
influence of training data on a learning model retrospectively, thereby
correcting data leaks and privacy issues. For learning models with strongly
convex loss functions, our method provides certified unlearning with
theoretical guarantees. For models with non-convex losses, we empirically show
that unlearning features and labels is effective and significantly faster than
other strategies.

    

### [[2108.11579] Modeling Item Response Theory with Stochastic Variational Inference](http://arxiv.org/abs/2108.11579)


  Item Response Theory (IRT) is a ubiquitous model for understanding human
behaviors and attitudes based on their responses to questions. Large modern
datasets offer opportunities to capture more nuances in human behavior,
potentially improving psychometric modeling leading to improved scientific
understanding and public policy. However, while larger datasets allow for more
flexible approaches, many contemporary algorithms for fitting IRT models may
also have massive computational demands that forbid real-world application. To
address this bottleneck, we introduce a variational Bayesian inference
algorithm for IRT, and show that it is fast and scalable without sacrificing
accuracy. Applying this method to five large-scale item response datasets from
cognitive science and education yields higher log likelihoods and higher
accuracy in imputing missing data than alternative inference algorithms. Using
this new inference approach we then generalize IRT with expressive Bayesian
models of responses, leveraging recent advances in deep learning to capture
nonlinear item characteristic curves (ICC) with neural networks. Using an
eigth-grade mathematics test from TIMSS, we show our nonlinear IRT models can
capture interesting asymmetric ICCs. The algorithm implementation is
open-source, and easily usable.

    

### [[2108.11604] Identification of the Resting Position Based on EGG, ECG, Respiration Rate and SpO2 Using Stacked Ensemble Learning](http://arxiv.org/abs/2108.11604)


  Rest is essential for a high-level physiological and psychological
performance. It is also necessary for the muscles to repair, rebuild, and
strengthen. There is a significant correlation between the quality of rest and
the resting posture. Therefore, identification of the resting position is of
paramount importance to maintain a healthy life. Resting postures can be
classified into four basic categories: Lying on the back (supine), facing of
the left / right sides and free-fall position. The later position is already
considered to be an unhealthy posture by researchers equivocally and hence can
be eliminated. In this paper, we analyzed the other three states of resting
position based on the data collected from the physiological parameters:
Electrogastrogram (EGG), Electrocardiogram (ECG), Respiration Rate, Heart Rate,
and Oxygen Saturation (SpO2). Based on these parameters, the resting position
is classified using a hybrid stacked ensemble machine learning model designed
using the Decision tree, Random Forest, and Xgboost algorithms. Our study
demonstrates a 100% accurate prediction of the resting position using the
hybrid model. The proposed method of identifying the resting position based on
physiological parameters has the potential to be integrated into wearable
devices. This is a low cost, highly accurate and autonomous technique to
monitor the body posture while maintaining the user privacy by eliminating the
use of RGB camera conventionally used to conduct the polysomnography (sleep
Monitoring) or resting position studies.

    

### [[2108.11623] Model-based Chance-Constrained Reinforcement Learning via Separated Proportional-Integral Lagrangian](http://arxiv.org/abs/2108.11623)


  Safety is essential for reinforcement learning (RL) applied in the real
world. Adding chance constraints (or probabilistic constraints) is a suitable
way to enhance RL safety under uncertainty. Existing chance-constrained RL
methods like the penalty methods and the Lagrangian methods either exhibit
periodic oscillations or learn an over-conservative or unsafe policy. In this
paper, we address these shortcomings by proposing a separated
proportional-integral Lagrangian (SPIL) algorithm. We first review the
constrained policy optimization process from a feedback control perspective,
which regards the penalty weight as the control input and the safe probability
as the control output. Based on this, the penalty method is formulated as a
proportional controller, and the Lagrangian method is formulated as an integral
controller. We then unify them and present a proportional-integral Lagrangian
method to get both their merits, with an integral separation technique to limit
the integral value in a reasonable range. To accelerate training, the gradient
of safe probability is computed in a model-based manner. We demonstrate our
method can reduce the oscillations and conservatism of RL policy in a
car-following simulation. To prove its practicality, we also apply our method
to a real-world mobile robot navigation task, where our robot successfully
avoids a moving obstacle with highly uncertain or even aggressive behaviors.

    

### [[2108.11637] Self-Attention for Audio Super-Resolution](http://arxiv.org/abs/2108.11637)


  Convolutions operate only locally, thus failing to model global interactions.
Self-attention is, however, able to learn representations that capture
long-range dependencies in sequences. We propose a network architecture for
audio super-resolution that combines convolution and self-attention.
Attention-based Feature-Wise Linear Modulation (AFiLM) uses self-attention
mechanism instead of recurrent neural networks to modulate the activations of
the convolutional model. Extensive experiments show that our model outperforms
existing approaches on standard benchmarks. Moreover, it allows for more
parallelization resulting in significantly faster training.

    

### [[2108.11644] Training a discrete variational autoencoder for generative chemistry and drug design on a quantum annealer](http://arxiv.org/abs/2108.11644)


  Deep generative chemistry models emerge as powerful tools to expedite drug
discovery. However, the immense size and complexity of the structural space of
all possible drug-like molecules pose significant obstacles, which could be
overcome with hybrid architectures combining quantum computers with deep
classical networks. We built a compact discrete variational autoencoder (DVAE)
with a Restricted Boltzmann Machine (RBM) of reduced size in its latent layer.
The size of the proposed model was small enough to fit on a state-of-the-art
D-Wave quantum annealer and allowed training on a subset of the ChEMBL dataset
of biologically active compounds. Finally, we generated $4290$ novel chemical
structures with medicinal chemistry and synthetic accessibility properties in
the ranges typical for molecules from ChEMBL. The experimental results point
towards the feasibility of using already existing quantum annealing devices for
drug discovery problems, which opens the way to building quantum generative
models for practically relevant applications.

    

### [[2108.11673] Why Adversarial Reprogramming Works, When It Fails, and How to Tell the Difference](http://arxiv.org/abs/2108.11673)


  Adversarial reprogramming allows repurposing a machine-learning model to
perform a different task. For example, a model trained to recognize animals can
be reprogrammed to recognize digits by embedding an adversarial program in the
digit images provided as input. Recent work has shown that adversarial
reprogramming may not only be used to abuse machine-learning models provided as
a service, but also beneficially, to improve transfer learning when training
data is scarce. However, the factors affecting its success are still largely
unexplained. In this work, we develop a first-order linear model of adversarial
reprogramming to show that its success inherently depends on the size of the
average input gradient, which grows when input gradients are more aligned, and
when inputs have higher dimensionality. The results of our experimental
analysis, involving fourteen distinct reprogramming tasks, show that the above
factors are correlated with the success and the failure of adversarial
reprogramming.

    

### [[2108.11674] Network Module Detection from Multi-Modal Node Features with a Greedy Decision Forest for Actionable Explainable AI](http://arxiv.org/abs/2108.11674)


  Network-based algorithms are used in most domains of research and industry in
a wide variety of applications and are of great practical use. In this work, we
demonstrate subnetwork detection based on multi-modal node features using a new
Greedy Decision Forest for better interpretability. The latter will be a
crucial factor in retaining experts and gaining their trust in such algorithms
in the future. To demonstrate a concrete application example, we focus in this
paper on bioinformatics and systems biology with a special focus on
biomedicine. However, our methodological approach is applicable in many other
domains as well. Systems biology serves as a very good example of a field in
which statistical data-driven machine learning enables the analysis of large
amounts of multi-modal biomedical data. This is important to reach the future
goal of precision medicine, where the complexity of patients is modeled on a
system level to best tailor medical decisions, health practices and therapies
to the individual patient. Our glass-box approach could help to uncover
disease-causing network modules from multi-omics data to better understand
diseases such as cancer.

    

### [[2108.11683] Estimation of Riemannian distances between covariance operators and Gaussian processes](http://arxiv.org/abs/2108.11683)


  In this work we study two Riemannian distances between infinite-dimensional
positive definite Hilbert-Schmidt operators, namely affine-invariant Riemannian
and Log-Hilbert-Schmidt distances, in the context of covariance operators
associated with functional stochastic processes, in particular Gaussian
processes. Our first main results show that both distances converge in the
Hilbert-Schmidt norm. Using concentration results for Hilbert space-valued
random variables, we then show that both distances can be consistently and
efficiently estimated from (i) sample covariance operators, (ii) finite,
normalized covariance matrices, and (iii) finite samples generated by the given
processes, all with dimension-independent convergence. Our theoretical analysis
exploits extensively the methodology of reproducing kernel Hilbert space (RKHS)
covariance and cross-covariance operators. The theoretical formulation is
illustrated with numerical experiments on covariance operators of Gaussian
processes.

    

### [[2108.11684] Disentangling ODE parameters from dynamics in VAEs](http://arxiv.org/abs/2108.11684)


  Deep networks have become increasingly of interest in dynamical system
prediction, but generalization remains elusive. In this work, we consider the
physical parameters of ODEs as factors of variation of the data generating
process. By leveraging ideas from supervised disentanglement in VAEs, we aim to
separate the ODE parameters from the dynamics in the latent space. Experiments
show that supervised disentanglement allows VAEs to capture the variability in
the dynamics and extrapolate better to ODE parameter spaces that were not
present in the training data.

    

### [[2108.11693] Improving the Reliability of Semantic Segmentation of Medical Images by Uncertainty Modeling with Bayesian Deep Networks and Curriculum Learning](http://arxiv.org/abs/2108.11693)


  In this paper we propose a novel method which leverages the uncertainty
measures provided by Bayesian deep networks through curriculum learning so that
the uncertainty estimates are fed back to the system to resample the training
data more densely in areas where uncertainty is high. We show in the concrete
setting of a semantic segmentation task (iPS cell colony segmentation) that the
proposed system is able to increase significantly the reliability of the model.

    

### [[2108.11694] PoissonSeg: Semi-Supervised Few-Shot Medical Image Segmentation via Poisson Learning](http://arxiv.org/abs/2108.11694)


  The application of deep learning to medical image segmentation has been
hampered due to the lack of abundant pixel-level annotated data. Few-shot
Semantic Segmentation (FSS) is a promising strategy for breaking the deadlock.
However, a high-performing FSS model still requires sufficient pixel-level
annotated classes for training to avoid overfitting, which leads to its
performance bottleneck in medical image segmentation due to the unmet need for
annotations. Thus, semi-supervised FSS for medical images is accordingly
proposed to utilize unlabeled data for further performance improvement.
Nevertheless, existing semi-supervised FSS methods has two obvious defects: (1)
neglecting the relationship between the labeled and unlabeled data; (2) using
unlabeled data directly for end-to-end training leads to degenerated
representation learning. To address these problems, we propose a novel
semi-supervised FSS framework for medical image segmentation. The proposed
framework employs Poisson learning for modeling data relationship and
propagating supervision signals, and Spatial Consistency Calibration for
encouraging the model to learn more coherent representations. In this process,
unlabeled samples do not involve in end-to-end training, but provide
supervisory information for query image segmentation through graph-based
learning. We conduct extensive experiments on three medical image segmentation
datasets (i.e. ISIC skin lesion segmentation, abdominal organs segmentation for
MRI and abdominal organs segmentation for CT) to demonstrate the
state-of-the-art performance and broad applicability of the proposed framework.

    

### [[2108.11713] The Number of Steps Needed for Nonconvex Optimization of a Deep Learning Optimizer is a Rational Function of Batch Size](http://arxiv.org/abs/2108.11713)


  Recently, convergence as well as convergence rate analyses of deep learning
optimizers for nonconvex optimization have been widely studied. Meanwhile,
numerical evaluations for the optimizers have precisely clarified the
relationship between batch size and the number of steps needed for training
deep neural networks. The main contribution of this paper is to show
theoretically that the number of steps needed for nonconvex optimization of
each of the optimizers can be expressed as a rational function of batch size.
Having these rational functions leads to two particularly important facts,
which were validated numerically in previous studies. The first fact is that
there exists an optimal batch size such that the number of steps needed for
nonconvex optimization is minimized. This implies that using larger batch sizes
than the optimal batch size does not decrease the number of steps needed for
nonconvex optimization. The second fact is that the optimal batch size depends
on the optimizer. In particular, it is shown theoretically that momentum and
Adam-type optimizers can exploit larger optimal batches and further reduce the
minimum number of steps needed for nonconvex optimization than can the
stochastic gradient descent optimizer.

    

### [[2108.11730] Deep learning based dictionary learning and tomographic image reconstruction](http://arxiv.org/abs/2108.11730)


  This work presents an approach for image reconstruction in clinical low-dose
tomography that combines principles from sparse signal processing with ideas
from deep learning. First, we describe sparse signal representation in terms of
dictionaries from a statistical perspective and interpret dictionary learning
as a process of aligning distribution that arises from a generative model with
empirical distribution of true signals. As a result we can see that sparse
coding with learned dictionaries resembles a specific variational autoencoder,
where the decoder is a linear function and the encoder is a sparse coding
algorithm. Next, we show that dictionary learning can also benefit from
computational advancements introduced in the context of deep learning, such as
parallelism and as stochastic optimization. Finally, we show that
regularization by dictionaries achieves competitive performance in computed
tomography (CT) reconstruction comparing to state-of-the-art model based and
data driven approaches.

    

### [[2108.11751] Local Exceptionality Detection in Time Series Using Subgroup Discovery](http://arxiv.org/abs/2108.11751)


  In this paper, we present a novel approach for local exceptionality detection
on time series data. This method provides the ability to discover interpretable
patterns in the data, which can be used to understand and predict the
progression of a time series. This being an exploratory approach, the results
can be used to generate hypotheses about the relationships between the
variables describing a specific process and its dynamics. We detail our
approach in a concrete instantiation and exemplary implementation, specifically
in the field of teamwork research. Using a real-world dataset of team
interactions we include results from an example data analytics application of
our proposed approach, showcase novel analysis options, and discuss possible
implications of the results from the perspective of teamwork research.

    

### [[2108.11753] A survey on Bayesian inference for Gaussian mixture model](http://arxiv.org/abs/2108.11753)


  Clustering has become a core technology in machine learning, largely due to
its application in the field of unsupervised learning, clustering,
classification, and density estimation. A frequentist approach exists to hand
clustering based on mixture model which is known as the EM algorithm where the
parameters of the mixture model are usually estimated into a maximum likelihood
estimation framework. Bayesian approach for finite and infinite Gaussian
mixture model generates point estimates for all variables as well as associated
uncertainty in the form of the whole estimates' posterior distribution.
The sole aim of this survey is to give a self-contained introduction to
concepts and mathematical tools in Bayesian inference for finite and infinite
Gaussian mixture model in order to seamlessly introduce their applications in
subsequent sections. However, we clearly realize our inability to cover all the
useful and interesting results concerning this field and given the paucity of
scope to present this discussion, e.g., the separated analysis of the
generation of Dirichlet samples by stick-breaking and Polya's Urn approaches.
We refer the reader to literature in the field of the Dirichlet process mixture
model for a much detailed introduction to the related fields. Some excellent
examples include (Frigyik et al., 2010; Murphy, 2012; Gelman et al., 2014;
Hoff, 2009).
This survey is primarily a summary of purpose, significance of important
background and techniques for Gaussian mixture model, e.g., Dirichlet prior,
Chinese restaurant process, and most importantly the origin and complexity of
the methods which shed light on their modern applications. The mathematical
prerequisite is a first course in probability. Other than this modest
background, the development is self-contained, with rigorous proofs provided
throughout.

    

### [[2108.11754] Training and Profiling a Pediatric Emotion Recognition Classifier on Mobile Devices](http://arxiv.org/abs/2108.11754)


  Implementing automated emotion recognition on mobile devices could provide an
accessible diagnostic and therapeutic tool for those who struggle to recognize
emotion, including children with developmental behavioral conditions such as
autism. Although recent advances have been made in building more accurate
emotion classifiers, existing models are too computationally expensive to be
deployed on mobile devices. In this study, we optimized and profiled various
machine learning models designed for inference on edge devices and were able to
match previous state of the art results for emotion recognition on children.
Our best model, a MobileNet-V2 network pre-trained on ImageNet, achieved 65.11%
balanced accuracy and 64.19% F1-score on CAFE, while achieving a 45-millisecond
inference latency on a Motorola Moto G6 phone. This balanced accuracy is only
1.79% less than the current state of the art for CAFE, which used a model that
contains 26.62x more parameters and was unable to run on the Moto G6, even when
fully optimized. This work validates that with specialized design and
optimization techniques, machine learning models can become lightweight enough
for deployment on mobile devices and still achieve high accuracies on difficult
image classification tasks.

    

### [[2108.11757] Fast Accurate Defect Detection in Wafer Fabrication](http://arxiv.org/abs/2108.11757)


  A generic fast method for object classification is proposed. In addition, a
method for dimensional reduction is presented. The presented algorithms have
been applied to real-world data from chip fabrication successfully to the task
of predicting defect states of tens of thousands of chips of several products
based on measurements or even just part of measurements. Unlike typical neural
networks with a large number of weights to optimize over, the presented
algorithm tries optimizing only over a very small number of variables in order
to increase chances to find a global optimum. Our approach is interesting in
that it is fast, led to good to very good performance with real-world wafer
data, allows for short implementations and computes values which have a clear
meaning easy to explain.

    

### [[2108.11758] Determining the origin of impulsive noise events using paired wireless sound sensors](http://arxiv.org/abs/2108.11758)


  This work investigates how to identify the source of impulsive noise events
using a pair of wireless noise sensors. One sensor is placed at a known noise
source, and another sensor is placed at the noise receiver. Machine learning
models receive data from the two sensors and estimate whether a given noise
event originates from the known noise source or another source. To avoid
privacy issues, the approach uses on-edge preprocessing that converts the sound
into privacy compatible spectrograms. The system was evaluated at a shooting
range and explosives training facility, using data collected during noise
emission testing. The combination of convolutional neural networks with
cross-correlation achieved the best results. We created multiple alternative
models using different spectrogram representations. The best model detected
70.8\% of the impulsive noise events and correctly predicted 90.3\% of the
noise events in the optimal trade-off between recall and precision.

    

### [[2108.11761] Inducing Semantic Grouping of Latent Concepts for Explanations: An Ante-Hoc Approach](http://arxiv.org/abs/2108.11761)


  Self-explainable deep models are devised to represent the hidden concepts in
the dataset without requiring any posthoc explanation generation technique. We
worked with one of such models motivated by explicitly representing the
classifier function as a linear function and showed that by exploiting
probabilistic latent and properly modifying different parts of the model can
result better explanation as well as provide superior predictive performance.
Apart from standard visualization techniques, we proposed a new technique which
can strengthen human understanding towards hidden concepts. We also proposed a
technique of using two different self-supervision techniques to extract
meaningful concepts related to the type of self-supervision considered and
achieved significant performance boost. The most important aspect of our method
is that it works nicely in a low data regime and reaches the desired accuracy
in a few number of epochs. We reported exhaustive results with CIFAR10,
CIFAR100, and AWA2 datasets to show effect of our method with moderate and
relatively complex datasets.

    

### [[2108.11763] Attention-based Neural Load Forecasting: A Dynamic Feature Selection Approach](http://arxiv.org/abs/2108.11763)


  Encoder-decoder-based recurrent neural network (RNN) has made significant
progress in sequence-to-sequence learning tasks such as machine translation and
conversational models. Recent works have shown the advantage of this type of
network in dealing with various time series forecasting tasks. The present
paper focuses on the problem of multi-horizon short-term load forecasting,
which plays a key role in the power system's planning and operation. Leveraging
the encoder-decoder RNN, we develop an attention model to select the relevant
features and similar temporal information adaptively. First, input features are
assigned with different weights by a feature selection attention layer, while
the updated historical features are encoded by a bi-directional long short-term
memory (BiLSTM) layer. Then, a decoder with hierarchical temporal attention
enables a similar day selection, which re-evaluates the importance of
historical information at each time step. Numerical results tested on the
dataset of the global energy forecasting competition 2014 show that our
proposed model significantly outperforms some existing forecasting schemes.

    

### [[2108.11769] Byzantine Fault-Tolerance in Federated Local SGD under 2f-Redundancy](http://arxiv.org/abs/2108.11769)


  We consider the problem of Byzantine fault-tolerance in federated machine
learning. In this problem, the system comprises multiple agents each with local
data, and a trusted centralized coordinator. In fault-free setting, the agents
collaborate with the coordinator to find a minimizer of the aggregate of their
local cost functions defined over their local data. We consider a scenario
where some agents ($f$ out of $N$) are Byzantine faulty. Such agents need not
follow a prescribed algorithm correctly, and may communicate arbitrary
incorrect information to the coordinator. In the presence of Byzantine agents,
a more reasonable goal for the non-faulty agents is to find a minimizer of the
aggregate cost function of only the non-faulty agents. This particular goal is
commonly referred as exact fault-tolerance. Recent work has shown that exact
fault-tolerance is achievable if only if the non-faulty agents satisfy the
property of $2f$-redundancy. Now, under this property, techniques are known to
impart exact fault-tolerance to the distributed implementation of the classical
stochastic gradient-descent (SGD) algorithm. However, we do not know of any
such techniques for the federated local SGD algorithm - a more commonly used
method for federated machine learning. To address this issue, we propose a
novel technique named comparative elimination (CE). We show that, under
$2f$-redundancy, the federated local SGD algorithm with CE can indeed obtain
exact fault-tolerance in the deterministic setting when the non-faulty agents
can accurately compute gradients of their local cost functions. In the general
stochastic case, when agents can only compute unbiased noisy estimates of their
local gradients, our algorithm achieves approximate fault-tolerance with
approximation error proportional to the variance of stochastic gradients and
the fraction of Byzantine agents.

    

### [[2108.11775] Parallelised Diffeomorphic Sampling-based Motion Planning](http://arxiv.org/abs/2108.11775)


  We propose Parallelised Diffeomorphic Sampling-based Motion Planning (PDMP).
PDMP is a novel parallelised framework that uses bijective and differentiable
mappings, or diffeomorphisms, to transform sampling distributions of
sampling-based motion planners, in a manner akin to normalising flows. Unlike
normalising flow models which use invertible neural network structures to
represent these diffeomorphisms, we develop them from gradient information of
desired costs, and encode desirable behaviour, such as obstacle avoidance.
These transformed sampling distributions can then be used for sampling-based
motion planning. A particular example is when we wish to imbue the sampling
distribution with knowledge of the environment geometry, such that drawn
samples are less prone to be in collisions. To this end, we propose to learn a
continuous occupancy representation from environment occupancy data, such that
gradients of the representation defines a valid diffeomorphism and is amenable
to fast parallel evaluation. We use this to "morph" the sampling distribution
to draw far fewer collision-prone samples. PDMP is able to leverage gradient
information of costs, to inject specifications, in a manner similar to
optimisation-based motion planning methods, but relies on drawing from a
sampling distribution, retaining the tendency to find more global solutions,
thereby bridging the gap between trajectory optimisation and sampling-based
planning methods.

    

### [[2108.11781] On the use of test smells for prediction of flaky tests](http://arxiv.org/abs/2108.11781)


  Regression testing is an important phase to deliver software with quality.
However, flaky tests hamper the evaluation of test results and can increase
costs. This is because a flaky test may pass or fail non-deterministically and
to identify properly the flakiness of a test requires rerunning the test suite
multiple times. To cope with this challenge, approaches have been proposed
based on prediction models and machine learning. Existing approaches based on
the use of the test case vocabulary may be context-sensitive and prone to
overfitting, presenting low performance when executed in a cross-project
scenario. To overcome these limitations, we investigate the use of test smells
as predictors of flaky tests. We conducted an empirical study to understand if
test smells have good performance as a classifier to predict the flakiness in
the cross-project context, and analyzed the information gain of each test
smell. We also compared the test smell-based approach with the vocabulary-based
one. As a result, we obtained a classifier that had a reasonable performance
(Random Forest, 0.83%) to predict the flakiness in the testing phase. This
classifier presented better performance than vocabulary-based model for
cross-project prediction. The Assertion Roulette and Sleepy Test test smell
types are the ones associated with the best information gain values.

    

### [[2108.11785] A Hierarchical Assessment of Adversarial Severity](http://arxiv.org/abs/2108.11785)


  Adversarial Robustness is a growing field that evidences the brittleness of
neural networks. Although the literature on adversarial robustness is vast, a
dimension is missing in these studies: assessing how severe the mistakes are.
We call this notion "Adversarial Severity" since it quantifies the downstream
impact of adversarial corruptions by computing the semantic error between the
misclassification and the proper label. We propose to study the effects of
adversarial noise by measuring the Robustness and Severity into a large-scale
dataset: iNaturalist-H. Our contributions are: (i) we introduce novel
Hierarchical Attacks that harness the rich structured space of labels to create
adversarial examples. (ii) These attacks allow us to benchmark the Adversarial
Robustness and Severity of classification models. (iii) We enhance the
traditional adversarial training with a simple yet effective Hierarchical
Curriculum Training to learn these nodes gradually within the hierarchical
tree. We perform extensive experiments showing that hierarchical defenses allow
deep models to boost the adversarial Robustness by 1.85% and reduce the
severity of all attacks by 0.17, on average.

    

### [[2108.11800] Efficient Out-of-Distribution Detection Using Latent Space of $β$-VAE for Cyber-Physical Systems](http://arxiv.org/abs/2108.11800)


  Deep Neural Networks are actively being used in the design of autonomous
Cyber-Physical Systems (CPSs). The advantage of these models is their ability
to handle high-dimensional state-space and learn compact surrogate
representations of the operational state spaces. However, the problem is that
the sampled observations used for training the model may never cover the entire
state space of the physical environment, and as a result, the system will
likely operate in conditions that do not belong to the training distribution.
These conditions that do not belong to training distribution are referred to as
Out-of-Distribution (OOD). Detecting OOD conditions at runtime is critical for
the safety of CPS. In addition, it is also desirable to identify the context or
the feature(s) that are the source of OOD to select an appropriate control
action to mitigate the consequences that may arise because of the OOD
condition. In this paper, we study this problem as a multi-labeled time series
OOD detection problem over images, where the OOD is defined both sequentially
across short time windows (change points) as well as across the training data
distribution. A common approach to solving this problem is the use of
multi-chained one-class classifiers. However, this approach is expensive for
CPSs that have limited computational resources and require short inference
times. Our contribution is an approach to design and train a single
$\beta$-Variational Autoencoder detector with a partially disentangled latent
space sensitive to variations in image features. We use the feature sensitive
latent variables in the latent space to detect OOD images and identify the most
likely feature(s) responsible for the OOD. We demonstrate our approach using an
Autonomous Vehicle in the CARLA simulator and a real-world automotive dataset
called nuImages.

    

### [[2108.11809] Fine-tuning Pretrained Language Models with Label Attention for Explainable Biomedical Text Classification](http://arxiv.org/abs/2108.11809)


  The massive growth of digital biomedical data is making biomedical text
indexing and classification increasingly important. Accordingly, previous
research has devised numerous techniques ranging from rule-based systems to
deep neural networks, with most focusing on feedforward, convolutional or
recurrent neural architectures. More recently, fine-tuned transformers-based
pretrained models (PTMs) have demonstrated superior performance in many natural
language processing tasks. However, the direct use of PTMs in the biomedical
domain is only limited to the target documents, ignoring the rich semantic
information in the label descriptions. In this paper, we develop an improved
label attention-based architecture to inject semantic label description into
the fine-tuning process of PTMs. Results on two public medical datasets show
that the proposed fine-tuning scheme outperforms the conventionally fine-tuned
PTMs and prior state-of-the-art models. Furthermore, we show that fine-tuning
with the label attention mechanism is interpretable in the interpretability
study.

    

### [[2108.11811] When should agents explore?](http://arxiv.org/abs/2108.11811)


  Exploration remains a central challenge for reinforcement learning (RL).
Virtually all existing methods share the feature of a monolithic behaviour
policy that changes only gradually (at best). In contrast, the exploratory
behaviours of animals and humans exhibit a rich diversity, namely including
forms of switching between modes. This paper presents an initial study of
mode-switching, non-monolithic exploration for RL. We investigate different
modes to switch between, at what timescales it makes sense to switch, and what
signals make for good switching triggers. We also propose practical algorithmic
components that make the switching mechanism adaptive and robust, which enables
flexibility without an accompanying hyper-parameter-tuning burden. Finally, we
report a promising and detailed analysis on Atari, using two-mode exploration
and switching at sub-episodic time-scales.

    

### [[2108.11832] Subgradient methods near active manifolds: saddle point avoidance, local convergence, and asymptotic normality](http://arxiv.org/abs/2108.11832)


  Nonsmooth optimization problems arising in practice tend to exhibit
beneficial smooth substructure: their domains stratify into "active manifolds"
of smooth variation, which common proximal algorithms "identify" in finite
time. Identification then entails a transition to smooth dynamics, and
accommodates second-order acceleration techniques. While identification is
clearly useful algorithmically, empirical evidence suggests that even those
algorithms that do not identify the active manifold in finite time -- notably
the subgradient method -- are nonetheless affected by it. This work seeks to
explain this phenomenon, asking: how do active manifolds impact the subgradient
method in nonsmooth optimization?
In this work, we answer this question by introducing two algorithmically
useful properties -- aiming and subgradient approximation -- that fully expose
the smooth substructure of the problem. We show that these properties imply
that the shadow of the (stochastic) subgradient method along the active
manifold is precisely an inexact Riemannian gradient method with an implicit
retraction. We prove that these properties hold for a wide class of problems,
including cone reducible/decomposable functions and generic semialgebraic
problems. Moreover, we develop a thorough calculus, proving such properties are
preserved under smooth deformations and spectral lifts. This viewpoint then
leads to several algorithmic consequences that parallel results in smooth
optimization, despite the nonsmoothness of the problem: local rates of
convergence, asymptotic normality, and saddle point avoidance. The asymptotic
normality results appear to be new even in the most classical setting of
stochastic nonlinear programming. The results culminate in the following
observation: the perturbed subgradient method on generic, Clarke regular
semialgebraic problems, converges only to local minimizers.

    

### [[2108.11845] Consistent Relative Confidence and Label-Free Model Selection for Convolutional Neural Networks](http://arxiv.org/abs/2108.11845)


  This paper is concerned with image classification based on deep convolutional
neural networks (CNNs). The focus is centered around the following question:
given a set of candidate CNN models, how to select the right one that has the
best generalization property for the current task? Present model selection
methods require access to a batch of labeled data for defining a performance
metric, such as the cross-entropy loss, the classification error rate, the
negative log-likelihood, and so on. In many practical cases, however, labeled
data are not available in time as labeling itself is a time-consuming and
expensive task. To this end, this paper presents an approach to CNN model
selection using only unlabeled data. This method is developed based on a
principle termed consistent relative confidence (CRC). The effectiveness and
efficiency of the presented method are demonstrated by extensive experimental
studies based on datasets MNIST and FasionMNIST.

    

### [[2108.11872] Comparing Classes of Estimators: When does Gradient Descent Beat Ridge Regression in Linear Models?](http://arxiv.org/abs/2108.11872)


  Modern methods for learning from data depend on many tuning parameters, such
as the stepsize for optimization methods, and the regularization strength for
regularized learning methods. Since performance can depend strongly on these
parameters, it is important to develop comparisons between \emph{classes of
methods}, not just for particularly tuned ones. Here, we take aim to compare
classes of estimators via the relative performance of the \emph{best method in
the class}. This allows us to rigorously quantify the tuning sensitivity of
learning algorithms. As an illustration, we investigate the statistical
estimation performance of ridge regression with a uniform grid of
regularization parameters, and of gradient descent iterates with a fixed
stepsize, in the standard linear model with a random isotropic ground truth
parameter.
(1) For orthogonal designs, we find the \emph{exact minimax optimal classes
of estimators}, showing they are equal to gradient descent with a polynomially
decaying learning rate. We find the exact suboptimalities of ridge regression
and gradient descent with a fixed stepsize, showing that they decay as either
$1/k$ or $1/k^2$ for specific ranges of $k$ estimators.
(2) For general designs with a large number of non-zero eigenvalues, we find
that gradient descent outperforms ridge regression when the eigenvalues decay
slowly, as a power law with exponent less than unity. If instead the
eigenvalues decay quickly, as a power law with exponent greater than unity or
exponentially, we find that ridge regression outperforms gradient descent.
Our results highlight the importance of tuning parameters. In particular,
while optimally tuned ridge regression is the best estimator in our case, it
can be outperformed by gradient descent when both are restricted to being tuned
over a finite regularization grid.

    

### [[2108.11873] Spatio-Temporal Graph Contrastive Learning](http://arxiv.org/abs/2108.11873)


  Deep learning models are modern tools for spatio-temporal graph (STG)
forecasting. Despite their effectiveness, they require large-scale datasets to
achieve better performance and are vulnerable to noise perturbation. To
alleviate these limitations, an intuitive idea is to use the popular data
augmentation and contrastive learning techniques. However, existing graph
contrastive learning methods cannot be directly applied to STG forecasting due
to three reasons. First, we empirically discover that the forecasting task is
unable to benefit from the pretrained representations derived from contrastive
learning. Second, data augmentations that are used for defeating noise are less
explored for STG data. Third, the semantic similarity of samples has been
overlooked. In this paper, we propose a Spatio-Temporal Graph Contrastive
Learning framework (STGCL) to tackle these issues. Specifically, we improve the
performance by integrating the forecasting loss with an auxiliary contrastive
loss rather than using a pretrained paradigm. We elaborate on four types of
data augmentations, which disturb data in terms of graph structure, time
domain, and frequency domain. We also extend the classic contrastive loss
through a rule-based strategy that filters out the most semantically similar
negatives. Our framework is evaluated across three real-world datasets and four
state-of-the-art models. The consistent improvements demonstrate that STGCL can
be used as an off-the-shelf plug-in for existing deep models.

    

### [[2108.11875] A spatio-temporal LSTM model to forecast across multiple temporal and spatial scales](http://arxiv.org/abs/2108.11875)


  This paper presents a novel spatio-temporal LSTM (SPATIAL) architecture for
time series forecasting applied to environmental datasets. The framework was
evaluated across multiple sensors and for three different oceanic variables:
current speed, temperature, and dissolved oxygen. Network implementation
proceeded in two directions that are nominally separated but connected as part
of a natural environmental system -- across the spatial (between individual
sensors) and temporal components of the sensor data. Data from four sensors
sampling current speed, and eight measuring both temperature and dissolved
oxygen evaluated the framework. Results were compared against RF and XGB
baseline models that learned on the temporal signal of each sensor
independently by extracting the date-time features together with the past
history of data using sliding window matrix. Results demonstrated ability to
accurately replicate complex signals and provide comparable performance to
state-of-the-art benchmarks. Notably, the novel framework provided a simpler
pre-processing and training pipeline that handles missing values via a simple
masking layer. Enabling learning across the spatial and temporal directions,
this paper addresses two fundamental challenges of ML applications to
environmental science: 1) data sparsity and the challenges and costs of
collecting measurements of environmental conditions such as ocean dynamics, and
2) environmental datasets are inherently connected in the spatial and temporal
directions while classical ML approaches only consider one of these directions.
Furthermore, sharing of parameters across all input steps makes SPATIAL a fast,
scalable, and easily-parameterized forecasting framework.

    

### [[2108.11883] DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN](http://arxiv.org/abs/2108.11883)


  In the information explosion era, recommender systems (RSs) are widely
studied and applied to discover user-preferred information. A RS performs
poorly when suffering from the cold-start issue, which can be alleviated if
incorporating Knowledge Graphs (KGs) as side information. However, most
existing works neglect the facts that node degrees in KGs are skewed and
massive amount of interactions in KGs are recommendation-irrelevant. To address
these problems, in this paper, we propose Differentiable Sampling on Knowledge
Graph for Recommendation with Relational GNN (DSKReG) that learns the relevance
distribution of connected items from KGs and samples suitable items for
recommendation following this distribution. We devise a differentiable sampling
strategy, which enables the selection of relevant items to be jointly optimized
with the model training procedure. The experimental results demonstrate that
our model outperforms state-of-the-art KG-based recommender systems. The code
is available online at this https URL.

    

### [[2108.11884] Enabling SQL-based Training Data Debugging for Federated Learning](http://arxiv.org/abs/2108.11884)


  How can we debug a logistical regression model in a federated learning
setting when seeing the model behave unexpectedly (e.g., the model rejects all
high-income customers' loan applications)? The SQL-based training data
debugging framework has proved effective to fix this kind of issue in a
non-federated learning setting. Given an unexpected query result over model
predictions, this framework automatically removes the label errors from
training data such that the unexpected behavior disappears in the retrained
model. In this paper, we enable this powerful framework for federated learning.
The key challenge is how to develop a security protocol for federated debugging
which is proved to be secure, efficient, and accurate. Achieving this goal
requires us to investigate how to seamlessly integrate the techniques from
multiple fields (Databases, Machine Learning, and Cybersecurity). We first
propose FedRain, which extends Rain, the state-of-the-art SQL-based training
data debugging framework, to our federated learning setting. We address several
technical challenges to make FedRain work and analyze its security guarantee
and time complexity. The analysis results show that FedRain falls short in
terms of both efficiency and security. To overcome these limitations, we
redesign our security protocol and propose Frog, a novel SQL-based training
data debugging framework tailored for federated learning. Our theoretical
analysis shows that Frog is more secure, more accurate, and more efficient than
FedRain. We conduct extensive experiments using several real-world datasets and
a case study. The experimental results are consistent with our theoretical
analysis and validate the effectiveness of Frog in practice.

    

### [[2108.11887] Federated Reinforcement Learning: Techniques, Applications, and Open Challenges](http://arxiv.org/abs/2108.11887)


  This paper presents a comprehensive survey of Federated Reinforcement
Learning (FRL), an emerging and promising field in Reinforcement Learning (RL).
Starting with a tutorial of Federated Learning (FL) and RL, we then focus on
the introduction of FRL as a new method with great potential by leveraging the
basic idea of FL to improve the performance of RL while preserving
data-privacy. According to the distribution characteristics of the agents in
the framework, FRL algorithms can be divided into two categories, i.e.
Horizontal Federated Reinforcement Learning (HFRL) and Vertical Federated
Reinforcement Learning (VFRL). We provide the detailed definitions of each
category by formulas, investigate the evolution of FRL from a technical
perspective, and highlight its advantages over previous RL algorithms. In
addition, the existing works on FRL are summarized by application fields,
including edge computing, communication, control optimization, and attack
detection. Finally, we describe and discuss several key research directions
that are crucial to solving the open problems within FRL.

    

### [[2108.11894] Machine Learning for Discovering Effective Interaction Kernels between Celestial Bodies from Ephemerides](http://arxiv.org/abs/2108.11894)


  Building accurate and predictive models of the underlying mechanisms of
celestial motion has inspired fundamental developments in theoretical physics.
Candidate theories seek to explain observations and predict future positions of
planets, stars, and other astronomical bodies as faithfully as possible. We use
a data-driven learning approach, extending that developed in Lu et al. ($2019$)
and extended in Zhong et al. ($2020$), to a derive stable and accurate model
for the motion of celestial bodies in our Solar System. Our model is based on a
collective dynamics framework, and is learned from the NASA Jet Propulsion
Lab's development ephemerides. By modeling the major astronomical bodies in the
Solar System as pairwise interacting agents, our learned model generate
extremely accurate dynamics that preserve not only intrinsic geometric
properties of the orbits, but also highly sensitive features of the dynamics,
such as perihelion precession rates. Our learned model can provide a unified
explanation to the observation data, especially in terms of reproducing the
perihelion precession of Mars, Mercury, and the Moon. Moreover, Our model
outperforms Newton's Law of Universal Gravitation in all cases and performs
similarly to, and exceeds on the Moon, the Einstein-Infeld-Hoffman equations
derived from Einstein's theory of general relativity.

    

### [[2108.11898] Supervised Compression for Resource-constrained Edge Computing Systems](http://arxiv.org/abs/2108.11898)


  There has been much interest in deploying deep learning algorithms on
low-powered devices, including smartphones, drones, and medical sensors.
However, full-scale deep neural networks are often too resource-intensive in
terms of energy and storage. As a result, the bulk part of the machine learning
operation is therefore often carried out on an edge server, where the data is
compressed and transmitted. However, compressing data (such as images) leads to
transmitting information irrelevant to the supervised task. Another popular
approach is to split the deep network between the device and the server while
compressing intermediate features. To date, however, such split computing
strategies have barely outperformed the aforementioned naive data compression
baselines due to their inefficient approaches to feature compression. This
paper adopts ideas from knowledge distillation and neural image compression to
compress intermediate feature representations more efficiently. Our supervised
compression approach uses a teacher model and a student model with a stochastic
bottleneck and learnable prior for entropy coding. We compare our approach to
various neural image and feature compression baselines in three vision tasks
and found that it achieves better supervised rate-distortion performance while
also maintaining smaller end-to-end latency. We furthermore show that the
learned feature representations can be tuned to serve multiple downstream
tasks.

    

### [[2108.11923] Sketches for Time-Dependent Machine Learning](http://arxiv.org/abs/2108.11923)


  Time series data can be subject to changes in the underlying process that
generates them and, because of these changes, models built on old samples can
become obsolete or perform poorly. In this work, we present a way to
incorporate information about the current data distribution and its evolution
across time into machine learning algorithms. Our solution is based on
efficiently maintaining statistics, particularly the mean and the variance, of
data features at different time resolutions. These data summarisations can be
performed over the input attributes, in which case they can then be fed into
the model as additional input features, or over latent representations learned
by models, such as those of Recurrent Neural Networks. In classification tasks,
the proposed techniques can significantly outperform the prediction
capabilities of equivalent architectures with no feature / latent
summarisations. Furthermore, these modifications do not introduce notable
computational and memory overhead when properly adjusted.

    

### [[2108.11939] Understanding and Accelerating Neural Architecture Search with Training-Free and Theory-Grounded Metrics](http://arxiv.org/abs/2108.11939)


  This work targets designing a principled and unified training-free framework
for Neural Architecture Search (NAS), with high performance, low cost, and
in-depth interpretation. NAS has been explosively studied to automate the
discovery of top-performer neural networks, but suffers from heavy resource
consumption and often incurs search bias due to truncated training or
approximations. Recent NAS works start to explore indicators that can predict a
network's performance without training. However, they either leveraged limited
properties of deep networks, or the benefits of their training-free indicators
are not applied to more extensive search methods. By rigorous correlation
analysis, we present a unified framework to understand and accelerate NAS, by
disentangling "TEG" characteristics of searched networks - Trainability,
Expressivity, Generalization - all assessed in a training-free manner. The TEG
indicators could be scaled up and integrated with various NAS search methods,
including both supernet and single-path approaches. Extensive studies validate
the effective and efficient guidance from our TEG-NAS framework, leading to
both improved search accuracy and over 2.3x reduction in search time cost.
Moreover, we visualize search trajectories on three landscapes of "TEG"
characteristics, observing that while a good local minimum is easier to find on
NAS-Bench-201 given its simple topology, balancing "TEG" characteristics is
much harder on the DARTS search space due to its complex landscape geometry.
Our code is available at this https URL.

    

### [[2108.11942] Machine Learning for Mediation in Armed Conflicts](http://arxiv.org/abs/2108.11942)


  Today's conflicts are becoming increasingly complex, fluid and fragmented,
often involving a host of national and international actors with multiple and
often divergent interests. This development poses significant challenges for
conflict mediation, as mediators struggle to make sense of conflict dynamics,
such as the range of conflict parties and the evolution of their political
positions, the distinction between relevant and less relevant actors in peace
making, or the identification of key conflict issues and their interdependence.
International peace efforts appear increasingly ill-equipped to successfully
address these challenges. While technology is being increasingly used in a
range of conflict related fields, such as conflict predicting or information
gathering, less attention has been given to how technology can contribute to
conflict mediation. This case study is the first to apply state-of-the-art
machine learning technologies to data from an ongoing mediation process. Using
dialogue transcripts from peace negotiations in Yemen, this study shows how
machine-learning tools can effectively support international mediators by
managing knowledge and offering additional conflict analysis tools to assess
complex information. Apart from illustrating the potential of machine learning
tools in conflict mediation, the paper also emphasises the importance of
interdisciplinary and participatory research design for the development of
context-sensitive and targeted tools and to ensure meaningful and responsible
implementation.

    

### [[1902.04294] Unpriortized Autoencoder For Image Generation](http://arxiv.org/abs/1902.04294)


  In this paper, we treat the image generation task using an autoencoder, a
representative latent model. Unlike many studies regularizing the latent
variable's distribution by assuming a manually specified prior, we approach the
image generation task using an autoencoder by directly estimating the latent
distribution. To this end, we introduce 'latent density estimator' which
captures latent distribution explicitly and propose its structure. Through
experiments, we show that our generative model generates images with the
improved visual quality compared to previous autoencoder-based generative
models.

    

### [[2003.01852] q-VAE for Disentangled Representation Learning and Latent Dynamical Systems](http://arxiv.org/abs/2003.01852)


  A variational autoencoder (VAE) derived from Tsallis statistics called q-VAE
is proposed. In the proposed method, a standard VAE is employed to
statistically extract latent space hidden in sampled data, and this latent
space helps make robots controllable in feasible computational time and cost.
To improve the usefulness of the latent space, this paper focuses on
disentangled representation learning, e.g., $\beta$-VAE, which is the baseline
for it. Starting from a Tsallis statistics perspective, a new lower bound for
the proposed q-VAE is derived to maximize the likelihood of the sampled data,
which can be considered an adaptive $\beta$-VAE with deformed Kullback-Leibler
divergence. To verify the benefits of the proposed q-VAE, a benchmark task to
extract the latent space from the MNIST dataset was performed. The results
demonstrate that the proposed q-VAE improved disentangled representation while
maintaining the reconstruction accuracy of the data. In addition, it relaxes
the independency condition between data, which is demonstrated by learning the
latent dynamics of nonlinear dynamical systems. By combining disentangled
representation, the proposed q-VAE achieves stable and accurate long-term state
prediction from the initial state and the action sequence.
The dataset for hexapod walking is available on IEEE Dataport, doi:
this https URL.

    

### [[2003.07859] Stop-and-Go: Exploring Backdoor Attacks on Deep Reinforcement Learning-based Traffic Congestion Control Systems](http://arxiv.org/abs/2003.07859)


  Recent work has shown that the introduction of autonomous vehicles (AVs) in
traffic could help reduce traffic jams. Deep reinforcement learning methods
demonstrate good performance in complex control problems, including autonomous
vehicle control, and have been used in state-of-the-art AV controllers.
However, deep neural networks (DNNs) render automated driving vulnerable to
machine learning-based attacks. In this work, we explore the
backdooring/trojanning of DRL-based AV controllers. We develop a trigger design
methodology that is based on well-established principles of traffic physics.
The malicious actions include vehicle deceleration and acceleration to cause
stop-and-go traffic waves to emerge (congestion attacks) or AV acceleration
resulting in the AV crashing into the vehicle in front (insurance attack). We
test our attack on single-lane and two-lane circuits. Our experimental results
show that the backdoored model does not compromise normal operation
performance, with the maximum decrease in cumulative rewards being 1%. Still,
it can be maliciously activated to cause a crash or congestion when the
corresponding triggers appear.

    

### [[2003.11246] FastDTW is approximate and Generally Slower than the Algorithm it Approximates](http://arxiv.org/abs/2003.11246)


  Many time series data mining problems can be solved with repeated use of
distance measure. Examples of such tasks include similarity search, clustering,
classification, anomaly detection and segmentation. For over two decades it has
been known that the Dynamic Time Warping (DTW) distance measure is the best
measure to use for most tasks, in most domains. Because the classic DTW
algorithm has quadratic time complexity, many ideas have been introduced to
reduce its amortized time, or to quickly approximate it. One of the most cited
approximate approaches is FastDTW. The FastDTW algorithm has well over a
thousand citations and has been explicitly used in several hundred research
efforts. In this work, we make a surprising claim. In any realistic data mining
application, the approximate FastDTW is much slower than the exact DTW. This
fact clearly has implications for the community that uses this algorithm:
allowing it to address much larger datasets, get exact results, and do so in
less time.

    

### [[2005.13291] Deep Sensory Substitution: Noninvasively Enabling Biological Neural Networks to Receive Input from Artificial Neural Networks](http://arxiv.org/abs/2005.13291)


  As is expressed in the adage "a picture is worth a thousand words", when
using spoken language to communicate visual information, brevity can be a
challenge. This work describes a novel technique for leveraging machine-learned
feature embeddings to sonify visual (and other types of) information into a
perceptual audio domain, allowing users to perceive this information using only
their aural faculty. The system uses a pretrained image embedding network to
extract visual features and embed them in a compact subset of Euclidean space
-- this converts the images into feature vectors whose $L^2$ distances can be
used as a meaningful measure of similarity. A generative adversarial network
(GAN) is then used to find a distance preserving map from this metric space of
feature vectors into the metric space defined by a target audio dataset
equipped with either the Euclidean metric or a mel-frequency cepstrum-based
psychoacoustic distance metric. We demonstrate this technique by sonifying
images of faces into human speech-like audio. For both target audio metrics,
the GAN successfully found a metric preserving mapping, and in human subject
tests, users were able to accurately classify audio sonifications of faces.

    

### [[2009.05487] The Intriguing Relation Between Counterfactual Explanations and Adversarial Examples](http://arxiv.org/abs/2009.05487)


  The same method that creates adversarial examples (AEs) to fool
image-classifiers can be used to generate counterfactual explanations (CEs)
that explain algorithmic decisions. This observation has led researchers to
consider CEs as AEs by another name. We argue that the relationship to the true
label and the tolerance with respect to proximity are two properties that
formally distinguish CEs and AEs. Based on these arguments, we introduce CEs,
AEs, and related concepts mathematically in a common framework. Furthermore, we
show connections between current methods for generating CEs and AEs, and
estimate that the fields will merge more and more as the number of common
use-cases grows.

    

### [[2009.14332] Multi-hop Attention Graph Neural Network](http://arxiv.org/abs/2009.14332)


  Self-attention mechanism in graph neural networks (GNNs) led to
state-of-the-art performance on many graph representation learning tasks.
Currently, at every layer, attention is computed between connected pairs of
nodes and depends solely on the representation of the two nodes. However, such
attention mechanism does not account for nodes that are not directly connected
but provide important network context. Here we propose Multi-hop Attention
Graph Neural Network (MAGNA), a principled way to incorporate multi-hop context
information into every layer of attention computation. MAGNA diffuses the
attention scores across the network, which increases the receptive field for
every layer of the GNN. Unlike previous approaches, MAGNA uses a diffusion
prior on attention values, to efficiently account for all paths between the
pair of disconnected nodes. We demonstrate in theory and experiments that MAGNA
captures large-scale structural information in every layer, and has a low-pass
effect that eliminates noisy high-frequency information from graph data.
Experimental results on node classification as well as the knowledge graph
completion benchmarks show that MAGNA achieves state-of-the-art results: MAGNA
achieves up to 5.7 percent relative error reduction over the previous
state-of-the-art on Cora, Citeseer, and Pubmed. MAGNA also obtains the best
performance on a large-scale Open Graph Benchmark dataset. On knowledge graph
completion MAGNA advances state-of-the-art on WN18RR and FB15k-237 across four
different performance metrics.

    

### [[2009.14397] Deep Equals Shallow for ReLU Networks in Kernel Regimes](http://arxiv.org/abs/2009.14397)


  Deep networks are often considered to be more expressive than shallow ones in
terms of approximation. Indeed, certain functions can be approximated by deep
networks provably more efficiently than by shallow ones, however, no tractable
algorithms are known for learning such deep models. Separately, a recent line
of work has shown that deep networks trained with gradient descent may behave
like (tractable) kernel methods in a certain over-parameterized regime, where
the kernel is determined by the architecture and initialization, and this paper
focuses on approximation for such kernels. We show that for ReLU activations,
the kernels derived from deep fully-connected networks have essentially the
same approximation properties as their shallow two-layer counterpart, namely
the same eigenvalue decay for the corresponding integral operator. This
highlights the limitations of the kernel framework for understanding the
benefits of such deep architectures. Our main theoretical result relies on
characterizing such eigenvalue decays through differentiability properties of
the kernel function, which also easily applies to the study of other kernels
defined on the sphere.

    

### [[2010.12877] EEGsig: an open-source machine learning-based toolbox for EEG signal processing](http://arxiv.org/abs/2010.12877)


  In the quest to realize a comprehensive EEG signal processing framework, in
this paper, we demonstrate a toolbox and graphic user interface, EEGsig, for
the full process of EEG signals. Our goal is to provide a comprehensive suite,
free and open-source framework for EEG signal processing where the users
especially physicians who do not have programming experience can focus on their
practical requirements to speed up the medical projects. Developed on MATLAB
software, we have aggregated all the three EEG signal processing steps,
including preprocessing, feature extraction, and classification into EEGsig. In
addition to a varied list of useful features, in EEGsig, we have implemented
three popular classification algorithms (K-NN, SVM, and ANN) to assess the
performance of the features. Our experimental results demonstrate that our
novel framework for EEG signal processing attained excellent classification
results and feature extraction robustness under different machine learning
classifier algorithms. Besides, in EEGsig, for selecting the best feature
extracted, all EEG signal channels can be visible simultaneously; thus, the
effect of each task on the signal can be visible. We believe that our
user-centered MATLAB package is an encouraging platform for novice users as
well as offering the highest level of control to expert users

    

### [[2011.10737] Neural Network iLQR: A Reinforcement Learning Architecture for Trajectory Optimization](http://arxiv.org/abs/2011.10737)


  As a notable machine learning paradigm, the research efforts in the context
of reinforcement learning have certainly progressed leaps and bounds. When
compared with reinforcement learning methods with the given system model, the
methodology of the reinforcement learning architecture based on the unknown
model generally exhibits significantly broader universality and applicability.
In this work, a new reinforcement learning architecture based on iterative
linear quadratic regulator (iLQR) is developed and presented without the
requirement of any prior knowledge of the system model, which is termed as an
approach of a "neural network iterative linear quadratic regulator (NNiLQR)".
Depending solely on measurement data, this method yields a completely new
non-parametric routine for the establishment of the optimal policy (without the
necessity of system modeling) through iterative refinements of the neural
network system. Rather importantly, this approach significantly outperforms the
classical iLQR method in terms of the given objective function because of the
innovative utilization of further exploration in the methodology. As clearly
indicated from the results attained in two illustrative examples, these
significant merits of the NNiLQR method are demonstrated rather evidently.

    

### [[2011.12130] CASU2Net: Cascaded Unification Network by a Two-step Early Fusion for Fault Detection in Offshore Wind Turbines](http://arxiv.org/abs/2011.12130)


  This paper presents a novel feature fusion-based deep learning model (called
CASU2Net) for fault detection in offshore wind turbines. The proposed CASU2Net
model benefits of a two-step early fusion to enrich features in the final
stage. Moreover, since previous studies did not consider uncertainty while
model developing and also predictions, we take advantage of Monte Carlo dropout
(MC dropout) to enhance the certainty of the results. To design fault detection
model, we use five sensors and a sliding window to exploit the inherent
temporal information contained in the raw time-series data obtained from
sensors. The proposed model uses the nonlinear relationships among multiple
sensor variables and the temporal dependency of each sensor on others which
considerably increases the performance of fault detection model. A 10-fold
cross-validation approach is used to verify the generalization of the model and
evaluate the classification metrics. To evaluate the performance of the model,
simulated data from a benchmark floating offshore wind turbine (FOWT) with
supervisory control and data acquisition (SCADA) are used. The results
illustrate that the proposed model would accurately disclose and classify more
than 99% of the faults. Moreover, it is generalizable and can be used to detect
faults for different types of systems.

    

### [[2012.03918] NeRD: Neural Reflectance Decomposition from Image Collections](http://arxiv.org/abs/2012.03918)


  Decomposing a scene into its shape, reflectance, and illumination is a
challenging but important problem in computer vision and graphics. This problem
is inherently more challenging when the illumination is not a single light
source under laboratory conditions but is instead an unconstrained
environmental illumination. Though recent work has shown that implicit
representations can be used to model the radiance field of an object, most of
these techniques only enable view synthesis and not relighting. Additionally,
evaluating these radiance fields is resource and time-intensive. We propose a
neural reflectance decomposition (NeRD) technique that uses physically-based
rendering to decompose the scene into spatially varying BRDF material
properties. In contrast to existing techniques, our input images can be
captured under different illumination conditions. In addition, we also propose
techniques to convert the learned reflectance volume into a relightable
textured mesh enabling fast real-time rendering with novel illuminations. We
demonstrate the potential of the proposed approach with experiments on both
synthetic and real datasets, where we are able to obtain high-quality
relightable 3D assets from image collections. The datasets and code is
available on the project page: this https URL


### [[2012.10961] Recent Developments in Detection of Central Serous Retinopathy through Imaging and Artificial Intelligence Techniques A Review](http://arxiv.org/abs/2012.10961)


  Central Serous Retinopathy (CSR) or Central Serous Chorioretinopathy (CSC) is
a significant disease that causes blindness and vision loss among millions of
people worldwide. It transpires as a result of accumulation of watery fluids
behind the retina. Therefore, detection of CSR at early stages allows
preventive measures to avert any impairment to the human eye. Traditionally,
several manual methods for detecting CSR have been developed in the past;
however, they have shown to be imprecise and unreliable. Consequently,
Artificial Intelligence (AI) services in the medical field, including automated
CSR detection, are now possible to detect and cure this disease. This review
assessed a variety of innovative technologies and researches that contribute to
the automatic detection of CSR. In this review, various CSR disease detection
techniques, broadly classified into two categories: a) CSR detection based on
classical imaging technologies, and b) CSR detection based on Machine/Deep
Learning methods, have been reviewed after an elaborated evaluation of 29
different relevant articles. Additionally, it also goes over the advantages,
drawbacks and limitations of a variety of traditional imaging techniques, such
as Optical Coherence Tomography Angiography (OCTA), Fundus Imaging and more
recent approaches that utilize Artificial Intelligence techniques. Finally, it
is concluded that the most recent Deep Learning (DL) classifiers deliver
accurate, fast, and reliable CSR detection. However, more research needs to be
conducted on publicly available datasets to improve computation complexity for
the reliable detection and diagnosis of CSR disease.

    

### [[2101.03013] Multistage BiCross encoder for multilingual access to COVID-19 health information](http://arxiv.org/abs/2101.03013)


  The Coronavirus (COVID-19) pandemic has led to a rapidly growing 'infodemic'
of health information online. This has motivated the need for accurate semantic
search and retrieval of reliable COVID-19 information across millions of
documents, in multiple languages. To address this challenge, this paper
proposes a novel high precision and high recall neural Multistage BiCross
encoder approach. It is a sequential three-stage ranking pipeline which uses
the Okapi BM25 retrieval algorithm and transformer-based bi-encoder and
cross-encoder to effectively rank the documents with respect to the given
query. We present experimental results from our participation in the
Multilingual Information Access (MLIA) shared task on COVID-19 multilingual
semantic search. The independently evaluated MLIA results validate our approach
and demonstrate that it outperforms other state-of-the-art approaches according
to nearly all evaluation metrics in cases of both monolingual and bilingual
runs.

    

### [[2101.05151] Temporal Knowledge Graph Forecasting with Neural ODE](http://arxiv.org/abs/2101.05151)


  Learning node representation on dynamically-evolving, multi-relational graph
data has gained great research interest. However, most of the existing models
for temporal knowledge graph forecasting use Recurrent Neural Network (RNN)
with discrete depth to capture temporal information, while time is a continuous
variable. Inspired by Neural Ordinary Differential Equation (NODE), we extend
the idea of continuum-depth models to time-evolving multi-relational graph
data, and propose a novel Temporal Knowledge Graph Forecasting model with NODE.
Our model captures temporal information through NODE and structural information
through a Graph Neural Network (GNN). Thus, our graph ODE model achieves a
continuous model in time and efficiently learns node representation for future
prediction. We evaluate our model on six temporal knowledge graph datasets by
performing link forecasting. Experiment results show the superiority of our
model.

    

### [[2101.11075] Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization](http://arxiv.org/abs/2101.11075)


  We introduce MADGRAD, a novel optimization method in the family of AdaGrad
adaptive gradient methods. MADGRAD shows excellent performance on deep learning
optimization problems from multiple fields, including classification and
image-to-image tasks in vision, and recurrent and bidirectionally-masked models
in natural language processing. For each of these tasks, MADGRAD matches or
outperforms both SGD and ADAM in test set performance, even on problems for
which adaptive methods normally perform poorly.

    

### [[2101.11513] State estimation with limited sensors -- A deep learning based approach](http://arxiv.org/abs/2101.11513)


  The importance of state estimation in fluid mechanics is well-established; it
is required for accomplishing several tasks including design/optimization,
active control, and future state prediction. A common tactic in this regards is
to rely on reduced order models. Such approaches, in general, use measurement
data of one-time instance. However, oftentimes data available from sensors is
sequential and ignoring it results in information loss. In this paper, we
propose a novel deep learning based state estimation framework that learns from
sequential data. The proposed model structure consists of the recurrent cell to
pass information from different time steps enabling utilization of this
information to recover the full state. We illustrate that utilizing sequential
data allows for state recovery from only one or two sensors. For efficient
recovery of the state, the proposed approached is coupled with an auto-encoder
based reduced order model. We illustrate the performance of the proposed
approach using two examples and it is found to outperform other alternatives
existing in the literature.

    

### [[2102.09603] Towards Solving the DeepFake Problem : An Analysis on Improving DeepFake Detection using Dynamic Face Augmentation](http://arxiv.org/abs/2102.09603)


  The creation of altered and manipulated faces has become more common due to
the improvement of DeepFake generation methods. Simultaneously, we have seen
detection models' development for differentiating between a manipulated and
original face from image or video content. In this paper, we focus on
identifying the limitations and shortcomings of existing deepfake detection
frameworks. We identified some key problems surrounding deepfake detection
through quantitative and qualitative analysis of existing methods and datasets.
We found that deepfake datasets are highly oversampled, causing models to
become easily overfitted. The datasets are created using a small set of real
faces to generate multiple fake samples. When trained on these datasets, models
tend to memorize the actors' faces and labels instead of learning fake
features. To mitigate this problem, we propose a simple data augmentation
method termed Face-Cutout. Our method dynamically cuts out regions of an image
using the face landmark information. It helps the model selectively attend to
only the relevant regions of the input. Our evaluation experiments show that
Face-Cutout can successfully improve the data variation and alleviate the
problem of overfitting. Our method achieves a reduction in LogLoss of 15.2% to
35.3% on different datasets, compared to other occlusion-based techniques.
Moreover, we also propose a general-purpose data pre-processing guideline to
train and evaluate existing architectures allowing us to improve the
generalizability of these models for deepfake detection.

    

### [[2102.11887] Quantum Cross Entropy and Maximum Likelihood Principle](http://arxiv.org/abs/2102.11887)


  Quantum machine learning is an emerging field at the intersection of machine
learning and quantum computing. Classical cross entropy plays a central role in
machine learning. We define its quantum generalization, the quantum cross
entropy, prove its lower bounds, and investigate its relation to quantum
fidelity. In the classical case, minimizing cross entropy is equivalent to
maximizing likelihood. In the quantum case, when the quantum cross entropy is
constructed from quantum data undisturbed by quantum measurements, this
relation holds. Classical cross entropy is equal to negative log-likelihood.
When we obtain quantum cross entropy through empirical density matrix based on
measurement outcomes, the quantum cross entropy is lower-bounded by negative
log-likelihood. These two different scenarios illustrate the information loss
when making quantum measurements. We conclude that to achieve the goal of full
quantum machine learning, it is crucial to utilize the deferred measurement
principle.

    

### [[2103.04710] Cluster-based Input Weight Initialization for Echo State Networks](http://arxiv.org/abs/2103.04710)


  Echo State Networks (ESNs) are a special type of recurrent neural networks
(RNNs), in which the input and recurrent connections are traditionally
generated randomly, and only the output weights are trained. Despite the recent
success of ESNs in various tasks of audio, image and radar recognition, we
postulate that a purely random initialization is not the ideal way of
initializing ESNs. The aim of this work is to propose an unsupervised
initialization of the input connections using the K-Means algorithm on the
training data. We show that for a large variety of datasets this initialization
performs equivalently or superior than a randomly initialized ESN whilst
needing significantly less reservoir neurons. Furthermore, we discuss that this
approach provides the opportunity to estimate a suitable size of the reservoir
based on prior knowledge about the data.

    

### [[2103.09504] PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](http://arxiv.org/abs/2103.09504)


  The predictive learning of spatiotemporal sequences aims to generate future
images by learning from the historical context, where the visual dynamics are
believed to have modular structures that can be learned with compositional
subsystems. This paper models these structures by presenting PredRNN, a new
recurrent network, in which a pair of memory cells are explicitly decoupled,
operate in nearly independent transition manners, and finally form unified
representations of the complex environment. Concretely, besides the original
memory cell of LSTM, this network is featured by a zigzag memory flow that
propagates in both bottom-up and top-down directions across all layers,
enabling the learned visual dynamics at different levels of RNNs to
communicate. It also leverages a memory decoupling loss to keep the memory
cells from learning redundant features. We further propose a new curriculum
learning strategy to force PredRNN to learn long-term dynamics from context
frames, which can be generalized to most sequence-to-sequence models. We
provide detailed ablation studies to verify the effectiveness of each
component. Our approach is shown to obtain highly competitive results on five
datasets for both action-free and action-conditioned predictive learning
scenarios.

    

### [[2103.13822] FedGP: Correlation-Based Active Client Selection for Heterogeneous Federated Learning](http://arxiv.org/abs/2103.13822)


  Client-wise heterogeneity is one of the major issues that hinder effective
training in federated learning (FL). Since the data distribution on each client
may vary dramatically, the client selection strategy can largely influence the
convergence rate of the FL process. Active client selection strategies are
popularly adopted in recent studies. However, they neglect the loss
correlations between the clients and achieve marginal improvement compared to
the uniform selection strategy. In this work, we propose FedGP -- a federated
learning framework built on a correlation-based client selection strategy, to
boost the convergence rate of FL. Specifically, we first model the loss
correlations between the clients with a Gaussian Process (GP). To make the GP
training practical in the communication-bounded FL process, we develop a GP
training method to reduce the communication cost by utilizing the covariance
stationarity. Finally, based on the correlations we learned, we derive a client
selection strategy with an enlarged reduction of expected global loss in each
round. Our experimental results show that compared to the latest active client
selection strategy, FedGP can improve the convergence rates by
$1.3\sim2.0\times$ and $1.2\sim1.5\times$ on FMNIST and CIFAR-10, respectively.

    

### [[2103.16774] Towards understanding the power of quantum kernels in the NISQ era](http://arxiv.org/abs/2103.16774)


  A key problem in the field of quantum computing is understanding whether
quantum machine learning (QML) models implemented on noisy intermediate-scale
quantum (NISQ) machines can achieve quantum advantages. Recently, Huang et al.
[Nat Commun 12, 2631] partially answered this question by the lens of quantum
kernel learning. Namely, they exhibited that quantum kernels can learn specific
datasets with lower generalization error over the optimal classical kernel
methods. However, most of their results are established on the ideal setting
and ignore the caveats of near-term quantum machines. To this end, a crucial
open question is: does the power of quantum kernels still hold under the NISQ
setting? In this study, we fill this knowledge gap by exploiting the power of
quantum kernels when the quantum system noise and sample error are considered.
Concretely, we first prove that the advantage of quantum kernels is vanished
for large size of datasets, few number of measurements, and large system noise.
With the aim of preserving the superiority of quantum kernels in the NISQ era,
we further devise an effective method via indefinite kernel learning. Numerical
simulations accord with our theoretical results. Our work provides theoretical
guidance of exploring advanced quantum kernels to attain quantum advantages on
NISQ devices.

    

### [[2104.06399] Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399)


  In this paper, we present Co-scale conv-attentional image Transformers
(CoaT), a Transformer-based image classifier equipped with co-scale and
conv-attentional mechanisms. First, the co-scale mechanism maintains the
integrity of Transformers' encoder branches at individual scales, while
allowing representations learned at different scales to effectively communicate
with each other; we design a series of serial and parallel blocks to realize
the co-scale mechanism. Second, we devise a conv-attentional mechanism by
realizing a relative position embedding formulation in the factorized attention
module with an efficient convolution-like implementation. CoaT empowers image
Transformers with enriched multi-scale and contextual modeling capabilities. On
ImageNet, relatively small CoaT models attain superior classification results
compared with similar-sized convolutional neural networks and image/vision
Transformers. The effectiveness of CoaT's backbone is also illustrated on
object detection and instance segmentation, demonstrating its applicability to
downstream computer vision tasks.

    

### [[2105.01883] RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](http://arxiv.org/abs/2105.01883)


  We propose RepMLP, a multi-layer-perceptron-style neural network building
block for image recognition, which is composed of a series of fully-connected
(FC) layers. Compared to convolutional layers, FC layers are more efficient,
better at modeling the long-range dependencies and positional patterns, but
worse at capturing the local structures, hence usually less favored for image
recognition. We propose a structural re-parameterization technique that adds
local prior into an FC to make it powerful for image recognition. Specifically,
we construct convolutional layers inside a RepMLP during training and merge
them into the FC for inference. On CIFAR, a simple pure-MLP model shows
performance very close to CNN. By inserting RepMLP in traditional CNN, we
improve ResNets by 1.8% accuracy on ImageNet, 2.9% for face recognition, and
2.3% mIoU on Cityscapes with lower FLOPs. Our intriguing findings highlight
that combining the global representational capacity and positional perception
of FC with the local prior of convolution can improve the performance of neural
network with faster speed on both the tasks with translation invariance (e.g.,
semantic segmentation) and those with aligned images and positional patterns
(e.g., face recognition). The code and models are available at
this https URL.

    

### [[2105.07142] Move2Hear: Active Audio-Visual Source Separation](http://arxiv.org/abs/2105.07142)


  We introduce the active audio-visual source separation problem, where an
agent must move intelligently in order to better isolate the sounds coming from
an object of interest in its environment. The agent hears multiple audio
sources simultaneously (e.g., a person speaking down the hall in a noisy
household) and it must use its eyes and ears to automatically separate out the
sounds originating from a target object within a limited time budget. Towards
this goal, we introduce a reinforcement learning approach that trains movement
policies controlling the agent's camera and microphone placement over time,
guided by the improvement in predicted audio separation quality. We demonstrate
our approach in scenarios motivated by both augmented reality (system is
already co-located with the target object) and mobile robotics (agent begins
arbitrarily far from the target object). Using state-of-the-art realistic
audio-visual simulations in 3D environments, we demonstrate our model's ability
to find minimal movement sequences with maximal payoff for audio source
separation. Project: this http URL.

    

### [[2105.08450] Implementation and Evaluation of a Multivariate Abstraction-Based, Interval-Based Dynamic Time-Warping Method as a Similarity Measure for Longitudinal Medical Records](http://arxiv.org/abs/2105.08450)


  We extended dynamic time warping (DTW) into interval-based dynamic time
warping (iDTW), including (A) interval-based representation (iRep): [1]
abstracting raw, time-stamped data into interval-based abstractions, [2]
comparison-period scoping, [3] partitioning abstract intervals into a given
temporal granularity; (B) interval-based matching (iMatch): matching
partitioned, abstract-concepts records, using a modified DTW. Using domain
knowledge, we abstracted the raw data of medical records, for up to three
concepts out of four or five relevant concepts, into two interval types: State
abstractions (e.g. LOW, HIGH) and Gradient abstractions (e.g. INCREASING,
DECREASING). We created all uni-dimensional (State or Gradient) or
multi-dimensional (State and Gradient) abstraction combinations. Tasks:
Classifying 161 oncology patients records as autologous or allogenic
bone-marrow transplantation; classifying 125 hepatitis patients records as B or
C hepatitis; predicting micro- or macro-albuminuria in the next year for 151
Type 2 diabetes patients. We used a k-Nearest-Neighbors majority, k = an odd
number from 1 to SQRT(N), N = set size. 75,936 10-fold cross-validation
experiments were performed: 33,600 (Oncology), 28,800 (Hepatitis), 13,536
(Diabetes). Measures: Area Under the Curve (AUC), optimal Youden's Index.
Paired t-tests compared result vectors for equivalent configurations other than
a tested variable, to determine a significant mean accuracy difference
(P<0.05). Mean classification and prediction using abstractions was
significantly better than using only raw time-stamped data. In each domain, at
least one abstraction combination led to a significantly better mean
performance than raw data. Increasing feature number and using
Multi-dimensional abstractions enhanced performance. Unlike when using raw
data, optimal mean performance was often reached with k=5, using abstractions.

    

### [[2106.03640] Making EfficientNet More Efficient: Exploring Batch-Independent Normalization, Group Convolutions and Reduced Resolution Training](http://arxiv.org/abs/2106.03640)


  Much recent research has been dedicated to improving the efficiency of
training and inference for image classification. This effort has commonly
focused on explicitly improving theoretical efficiency, often measured as
ImageNet validation accuracy per FLOP. These theoretical savings have, however,
proven challenging to achieve in practice, particularly on high-performance
training accelerators.
In this work, we focus on improving the practical efficiency of the
state-of-the-art EfficientNet models on a new class of accelerator, the
Graphcore IPU. We do this by extending this family of models in the following
ways: (i) generalising depthwise convolutions to group convolutions; (ii)
adding proxy-normalized activations to match batch normalization performance
with batch-independent statistics; (iii) reducing compute by lowering the
training resolution and inexpensively fine-tuning at higher resolution. We find
that these three methods improve the practical efficiency for both training and
inference. Code available at
this https URL .

    

### [[2106.05285] CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows](http://arxiv.org/abs/2106.05285)


  We introduce CaloFlow, a fast detector simulation framework based on
normalizing flows. For the first time, we demonstrate that normalizing flows
can reproduce many-channel calorimeter showers with extremely high fidelity,
providing a fresh alternative to computationally expensive GEANT4 simulations,
as well as other state-of-the-art fast simulation frameworks based on GANs and
VAEs. Besides the usual histograms of physical features and images of
calorimeter showers, we introduce a new metric for judging the quality of
generative modeling: the performance of a classifier trained to differentiate
real from generated images. We show that GAN-generated images can be identified
by the classifier with nearly 100% accuracy, while images generated from
CaloFlow are better able to fool the classifier. More broadly, normalizing
flows offer several advantages compared to other state-of-the-art approaches
(GANs and VAEs), including: tractable likelihoods; stable and convergent
training; and principled model selection. Normalizing flows also provide a
bijective mapping between data and the latent space, which could have other
applications beyond simulation, for example, to detector unfolding.

    

### [[2106.07030] The Backpropagation Algorithm Implemented on Spiking Neuromorphic Hardware](http://arxiv.org/abs/2106.07030)


  The capabilities of natural neural systems have inspired new generations of
machine learning algorithms as well as neuromorphic very large-scale integrated
(VLSI) circuits capable of fast, low-power information processing. However, it
has been argued that most modern machine learning algorithms are not
neurophysiologically plausible. In particular, the workhorse of modern deep
learning, the backpropagation algorithm, has proven difficult to translate to
neuromorphic hardware. In this study, we present a neuromorphic, spiking
backpropagation algorithm based on synfire-gated dynamical information
coordination and processing, implemented on Intel's Loihi neuromorphic research
processor. We demonstrate a proof-of-principle three-layer circuit that learns
to classify digits from the MNIST dataset. To our knowledge, this is the first
work to show a Spiking Neural Network (SNN) implementation of the
backpropagation algorithm that is fully on-chip, without a computer in the
loop. It is competitive in accuracy with off-chip trained SNNs and achieves an
energy-delay product suitable for edge computing. This implementation shows a
path for using in-memory, massively parallel neuromorphic processors for
low-power, low-latency implementation of modern deep learning applications.

    

### [[2106.16213] On the Power of Saturated Transformers: A View from Circuit Complexity](http://arxiv.org/abs/2106.16213)


  Transformers have become a standard architecture for many NLP problems. This
has motivated theoretically analyzing their capabilities as models of language,
in order to understand what makes them successful, and what their potential
weaknesses might be. Recent work has shown that transformers with hard
attention are quite limited in capacity, and in fact can be simulated by
constant-depth circuits. However, hard attention is a restrictive assumption,
which may complicate the relevance of these results for practical transformers.
In this work, we analyze the circuit complexity of transformers with saturated
attention: a generalization of hard attention that more closely captures the
attention patterns learnable in practical transformers. We show that saturated
transformers transcend the limitations of hard-attention transformers. With
some minor assumptions, we prove that the number of bits needed to represent a
saturated transformer memory vector is $O(\log n)$, which implies saturated
transformers can be simulated by log-depth circuits. Thus, the jump from hard
to saturated attention can be understood as increasing the transformer's
effective circuit depth by a factor of $O(\log n)$.

    

### [[2108.11441] Design and Scaffolded Training of an Efficient DNN Operator for Computer Vision on the Edge](http://arxiv.org/abs/2108.11441)


  Massively parallel systolic arrays and resource-efficient depthwise separable
convolutions are two promising techniques to accelerate DNN inference on the
edge. Interestingly, their combination is inefficient: Computational patterns
of depthwise separable convolutions do not exhibit a rhythmic systolic flow and
lack sufficient data reuse to saturate systolic arrays. We formally analyse
this inefficiency and propose an efficient operator, an optimal hardware
dataflow, and a superior training methodology towards alleviating this. The
efficient operator, called FuSeConv, is a drop-in replacement for depthwise
separable convolutions. FuSeConv factorizes convolution fully along their
spatial and depth dimensions. The resultant computation efficiently maps to
systolic arrays. The optimal dataflow, called Spatial-Tiled Output Stationary
(ST-OS), maximizes the efficiency of FuSeConv on systolic arrays. It maps
independent convolutions to rows of the array to maximize resource utilization
with negligible VLSI overheads. Neural Operator Scaffolding (NOS) scaffolds the
training of FuSeConv by distilling knowledge from the expensive depthwise
separable convolutions. This bridges the accuracy gap between FuSeConv networks
and baselines. Additionally, NOS can be combined with Neural Architecture
Search (NAS) to trade-off latency and accuracy. The HW/SW co-design of FuSeConv
with ST-OS achieves a significant speedup of 4.1-9.25X with state-of-the-art
efficient networks for ImageNet. The parameter efficiency of FuSeConv and its
significant out-performance over depthwise separable convolutions on systolic
arrays illustrates their promise as a strong solution on the edge. Training
FuSeConv networks with NOS achieves accuracy comparable to the baselines.
Further, by combining NOS with NAS, we design networks that define
state-of-the-art models improving on both accuracy and latency on systolic
arrays.

    

### [[2108.11521] Efficient On-Chip Communication for Parallel Graph-Analytics on Spatial Architectures](http://arxiv.org/abs/2108.11521)


  Large-scale graph processing has drawn great attention in recent years. Most
of the modern-day datacenter workloads can be represented in the form of Graph
Processing such as MapReduce etc. Consequently, a lot of designs for
Domain-Specific Accelerators have been proposed for Graph Processing. Spatial
Architectures have been promising in the execution of Graph Processing, where
the graph is partitioned into several nodes and each node works in parallel. We
conduct experiments to analyze the on-chip movement of data in graph processing
on a Spatial Architecture. Based on the observations, we identify a data
movement bottleneck, in the execution of such highly parallel processing
accelerators. To mitigate the bottleneck we propose a novel power-law aware
Graph Partitioning and Data Mapping scheme to reduce the communication latency
by minimizing the hop counts on a scalable network-on-chip. The experimental
results on popular graph algorithms show that our implementation makes the
execution 2-5x faster and 2.7-4x energy-efficient by reducing the data movement
time in comparison to a baseline implementation.

    

### [[2108.11444] PIVODL: Privacy-preserving vertical federated learning over distributed labels](http://arxiv.org/abs/2108.11444)


  Federated learning (FL) is an emerging privacy preserving machine learning
protocol that allows multiple devices to collaboratively train a shared global
model without revealing their private local data. Non-parametric models like
gradient boosting decision trees (GBDT) have been commonly used in FL for
vertically partitioned data. However, all these studies assume that all the
data labels are stored on only one client, which may be unrealistic for
real-world applications. Therefore, in this work, we propose a secure vertical
FL framework, named PIVODL, to train GBDT with data labels distributed on
multiple devices. Both homomorphic encryption and differential privacy are
adopted to prevent label information from being leaked through transmitted
gradients and leaf values. Our experimental results show that both information
leakage and model performance degradation of the proposed PIVODL are
negligible.

    

### [[2108.11503] Maneuver Identification Challenge](http://arxiv.org/abs/2108.11503)


  AI algorithms that identify maneuvers from trajectory data could play an
important role in improving flight safety and pilot training. AI challenges
allow diverse teams to work together to solve hard problems and are an
effective tool for developing AI solutions. AI challenges are also a key driver
of AI computational requirements. The Maneuver Identification Challenge hosted
at this http URL provides thousands of trajectories collected from pilots
practicing in flight simulators, descriptions of maneuvers, and examples of
these maneuvers performed by experienced pilots. Each trajectory consists of
positions, velocities, and aircraft orientations normalized to a common
coordinate system. Construction of the data set required significant data
architecture to transform flight simulator logs into AI ready data, which
included using a supercomputer for deduplication and data conditioning. There
are three proposed challenges. The first challenge is separating physically
plausible (good) trajectories from unfeasible (bad) trajectories. Human labeled
good and bad trajectories are provided to aid in this task. Subsequent
challenges are to label trajectories with their intended maneuvers and to
assess the quality of those maneuvers.

    

### [[2108.11507] Hardware-assisted Trusted Memory Disaggregation for Secure Far Memory](http://arxiv.org/abs/2108.11507)


  Memory disaggregation provides efficient memory utilization across
network-connected systems. It allows a node to use part of memory in remote
nodes in the same cluster. Recent studies have improved RDMA-based memory
disaggregation systems, supporting lower latency and higher bandwidth than the
prior generation of disaggregated memory. However, the current disaggregated
memory systems manage remote memory only at coarse granularity due to the
limitation of the access validation mechanism of RDMA. In such systems, to
support fine-grained remote page allocation, the trustworthiness of all
participating systems needs to be assumed, and thus a security breach in a node
can propagate to the entire cluster. From the security perspective, the
memory-providing node must protect its memory from memory-requesting nodes. On
the other hand, the memory-requesting node requires the confidentiality and
integrity protection of its memory contents even if they are stored in remote
nodes. To address the weak isolation support in the current system, this study
proposes a novel hardware-assisted memory disaggregation system. Based on the
security features of FPGA, the logic in each per-node FPGA board provides a
secure memory disaggregation engine. With its own networks, a set of FPGA-based
engines form a trusted memory disaggregation system, which is isolated from the
privileged software of each participating node. The secure memory
disaggregation system allows fine-grained memory management in memory-providing
nodes, while the access validation is guaranteed with the hardware-hardened
mechanism. In addition, the proposed system hides the memory access patterns
observable from remote nodes, supporting obliviousness. Our evaluation with
FPGA implementation shows that such fine-grained secure disaggregated memory is
feasible with comparable performance to the latest software-based techniques.

    

### [[2108.11525] Supercomputing Enabled Deployable Analytics for Disaster Response](http://arxiv.org/abs/2108.11525)


  First responders and other forward deployed essential workers can benefit
from advanced analytics. Limited network access and software security
requirements prevent the usage of standard cloud based microservice analytic
platforms that are typically used in industry. One solution is to precompute a
wide range of analytics as files that can be used with standard preinstalled
software that does not require network access or additional software and can
run on a wide range of legacy hardware. In response to the COVID-19 pandemic,
this approach was tested for providing geo-spatial census data to allow quick
analysis of demographic data for better responding to emergencies. These data
were processed using the MIT SuperCloud to create several thousand Google Earth
and Microsoft Excel files representative of many advanced analytics. The fast
mapping of census data using Google Earth and Microsoft Excel has the potential
to give emergency responders a powerful tool to improve emergency preparedness.
Our approach displays relevant census data (total population, population under
15, population over 65, median age) per census block, sorted by county, through
a Microsoft Excel spreadsheet (xlsx file) and Google Earth map (kml file). The
spreadsheet interface includes features that allow users to convert between
different longitude and latitude coordinate units. For the Google Earth files,
a variety of absolute and relative colors maps of population density have been
explored to provide an intuitive and meaningful interface. Using several
hundred cores on the MIT SuperCloud, new analytics can be generated in a few
minutes.

    

### [[2108.11613] On Truly Parallel Time in Population Protocols](http://arxiv.org/abs/2108.11613)


  The {\em parallel time} of a population protocol is defined as the average
number of required interactions that an agent in the protocol participates,
i.e., the quotient between the total number of interactions required by the
protocol and the total number $n$ of agents, or just roughly the number of
required rounds with $n$ interactions. This naming triggers an intuition that
at least on the average a round of $n$ interactions can be implemented in
$O(1)$ parallel steps. We show that when the transition function of a
population protocol is treated as a black box then the expected maximum number
of parallel steps necessary to implement a round of $n$ interactions is $\Omega
(\frac {\log n}{\log \log n})$. We also provide a combinatorial argument for a
matching upper bound on the number of parallel steps in the average case under
additional assumptions.

    

### [[2108.11633] Online Service Placement and Request Scheduling in MEC Networks](http://arxiv.org/abs/2108.11633)


  Mobile edge computing (MEC) emerges as a promising solution for servicing
delay-sensitive tasks at the edge network. A body of recent literature started
to focus on cost-efficient service placement and request scheduling. This work
investigates the joint optimization of service placement and request scheduling
in a dense MEC network, and develops an efficient online algorithm that
achieves close-to-optimal performance. Our online algorithm consists of two
basic modules: (1) a regularization with look-ahead approach from competitive
online convex optimization, for decomposing the offline relaxed minimization
problem into multiple sub-problems, each of which can be efficiently solved in
each time slot; (2) a randomized rounding method to transform the fractional
solution of offline relaxed problem into integer solution of the original
minimization problem, guaranteeing a low competitive ratio. Both theoretical
analysis and simulation studies corroborate the efficacy of our proposed online
MEC optimization algorithm.

    

### [[2108.11932] H2OPUS-TLR: High Performance Tile Low Rank Symmetric Factorizations using Adaptive Randomized Approximation](http://arxiv.org/abs/2108.11932)


  Tile low rank representations of dense matrices partition them into blocks of
roughly uniform size, where each off-diagonal tile is compressed and stored as
its own low rank factorization. They offer an attractive representation for
many data-sparse dense operators that appear in practical applications, where
substantial compression and a much smaller memory footprint can be achieved.
TLR matrices are a compromise between the simplicity of a regular
perfectly-strided data structure and the optimal complexity of the unbalanced
trees of hierarchically low rank matrices, and provide a convenient
performance-tuning parameter through their tile size that can be proportioned
to take into account the cache size where the tiles reside in the memory
hierarchy.
There are currently no high-performance algorithms that can generate Cholesky
and $LDL^T$ factorizations, particularly on GPUs. The difficulties in achieving
high performance when factoring TLR matrices come from the expensive
compression operations that must be performed during the factorization process
and the adaptive rank distribution of the tiles that causes an irregular work
pattern for the processing cores. In this work, we develop a dynamic batching
operation and combine it with batched adaptive randomized approximations to
achieve high performance both on GPUs and CPUs.
Our implementation attains over 1.2 TFLOP/s in double precision on the V100
GPU, and is limited by the performance of batched GEMM operations. The Cholesky
factorization of covariance matrix of size $N = 131K$ arising in spatial
statistics can be factored to an accuracy $\epsilon=10^{-2}$ in just a few
seconds. We believe the proposed GEMM-centric algorithm allows it to be readily
ported to newer hardware such as the tensor cores that are optimized for small
GEMM operations.

    

### [[2108.11420] Model-based Decision Making with Imagination for Autonomous Parking](http://arxiv.org/abs/2108.11420)


  Autonomous parking technology is a key concept within autonomous driving
research. This paper will propose an imaginative autonomous parking algorithm
to solve issues concerned with parking. The proposed algorithm consists of
three parts: an imaginative model for anticipating results before parking, an
improved rapid-exploring random tree (RRT) for planning a feasible trajectory
from a given start point to a parking lot, and a path smoothing module for
optimizing the efficiency of parking tasks. Our algorithm is based on a real
kinematic vehicle model; which makes it more suitable for algorithm application
on real autonomous cars. Furthermore, due to the introduction of the
imagination mechanism, the processing speed of our algorithm is ten times
faster than that of traditional methods, permitting the realization of
real-time planning simultaneously. In order to evaluate the algorithm's
effectiveness, we have compared our algorithm with traditional RRT, within
three different parking scenarios. Ultimately, results show that our algorithm
is more stable than traditional RRT and performs better in terms of efficiency
and quality.

    

### [[2108.11451] From Statistical Relational to Neural Symbolic Artificial Intelligence: a Survey](http://arxiv.org/abs/2108.11451)


  Neural-symbolic and statistical relational artificial intelligence both
integrate frameworks for learning with logical reasoning. This survey
identifies several parallels across seven different dimensions between these
two fields. These cannot only be used to characterize and position
neural-symbolic artificial intelligence approaches but also to identify a
number of directions for further research.

    

### [[2108.11510] Deep Reinforcement Learning in Computer Vision: A Comprehensive Survey](http://arxiv.org/abs/2108.11510)


  Deep reinforcement learning augments the reinforcement learning framework and
utilizes the powerful representation of deep neural networks. Recent works have
demonstrated the remarkable successes of deep reinforcement learning in various
domains including finance, medicine, healthcare, video games, robotics, and
computer vision. In this work, we provide a detailed review of recent and
state-of-the-art research advances of deep reinforcement learning in computer
vision. We start with comprehending the theories of deep learning,
reinforcement learning, and deep reinforcement learning. We then propose a
categorization of deep reinforcement learning methodologies and discuss their
advantages and limitations. In particular, we divide deep reinforcement
learning into seven main categories according to their applications in computer
vision, i.e. (i)landmark localization (ii) object detection; (iii) object
tracking; (iv) registration on both 2D image and 3D image volumetric data (v)
image segmentation; (vi) videos analysis; and (vii) other applications. Each of
these categories is further analyzed with reinforcement learning techniques,
network design, and performance. Moreover, we provide a comprehensive analysis
of the existing publicly available datasets and examine source code
availability. Finally, we present some open issues and discuss future research
directions on deep reinforcement learning in computer vision

    

### [[2108.11539] TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios](http://arxiv.org/abs/2108.11539)


  Object detection on drone-captured scenarios is a recent popular task. As
drones always navigate in different altitudes, the object scale varies
violently, which burdens the optimization of networks. Moreover, high-speed and
low-altitude flight bring in the motion blur on the densely packed objects,
which leads to great challenge of object distinction. To solve the two issues
mentioned above, we propose TPH-YOLOv5. Based on YOLOv5, we add one more
prediction head to detect different-scale objects. Then we replace the original
prediction heads with Transformer Prediction Heads (TPH) to explore the
prediction potential with self-attention mechanism. We also integrate
convolutional block attention model (CBAM) to find attention region on
scenarios with dense objects. To achieve more improvement of our proposed
TPH-YOLOv5, we provide bags of useful strategies such as data augmentation,
multiscale testing, multi-model integration and utilizing extra classifier.
Extensive experiments on dataset VisDrone2021 show that TPH-YOLOv5 have good
performance with impressive interpretability on drone-captured scenarios. On
DET-test-challenge dataset, the AP result of TPH-YOLOv5 are 39.18%, which is
better than previous SOTA method (DPNetV3) by 1.81%. On VisDrone Challenge
2021, TPHYOLOv5 wins 5th place and achieves well-matched results with 1st place
model (AP 39.43%). Compared to baseline model (YOLOv5), TPH-YOLOv5 improves
about 7%, which is encouraging and competitive.

    

### [[2108.11554] XCI-Sketch: Extraction of Color Information from Images for Generation of Colored Outlines and Sketches](http://arxiv.org/abs/2108.11554)


  Sketches are a medium to convey a visual scene from an individual's creative
perspective. The addition of color substantially enhances the overall
expressivity of a sketch. This paper proposes two methods to mimic human-drawn
colored sketches by utilizing the Contour Drawing Dataset. Our first approach
renders colored outline sketches by applying image processing techniques aided
by k-means color clustering. The second method uses a generative adversarial
network to develop a model that can generate colored sketches from previously
unobserved images. We assess the results obtained through quantitative and
qualitative evaluations.

    

### [[2108.11574] Understanding Attention in Machine Reading Comprehension](http://arxiv.org/abs/2108.11574)


  Achieving human-level performance on some of Machine Reading Comprehension
(MRC) datasets is no longer challenging with the help of powerful Pre-trained
Language Models (PLMs). However, the internal mechanism of these artifacts
still remains unclear, placing an obstacle for further understanding these
models. This paper focuses on conducting a series of analytical experiments to
examine the relations between the multi-head self-attention and the final
performance, trying to analyze the potential explainability in PLM-based MRC
models. We perform quantitative analyses on SQuAD (English) and CMRC 2018
(Chinese), two span-extraction MRC datasets, on top of BERT, ALBERT, and
ELECTRA in various aspects. We discover that {\em passage-to-question} and {\em
passage understanding} attentions are the most important ones, showing strong
correlations to the final performance than other parts. Through visualizations
and case studies, we also observe several general findings on the attention
maps, which could be helpful to understand how these models solve the
questions.

    

### [[2108.11609] Unsupervised Dense Deformation Embedding Network for Template-Free Shape Correspondence](http://arxiv.org/abs/2108.11609)


  Shape correspondence from 3D deformation learning has attracted appealing
academy interests recently. Nevertheless, current deep learning based methods
require the supervision of dense annotations to learn per-point translations,
which severely overparameterize the deformation process. Moreover, they fail to
capture local geometric details of original shape via global feature embedding.
To address these challenges, we develop a new Unsupervised Dense Deformation
Embedding Network (i.e., UD^2E-Net), which learns to predict deformations
between non-rigid shapes from dense local features. Since it is non-trivial to
match deformation-variant local features for deformation prediction, we develop
an Extrinsic-Intrinsic Autoencoder to frst encode extrinsic geometric features
from source into intrinsic coordinates in a shared canonical shape, with which
the decoder then synthesizes corresponding target features. Moreover, a bounded
maximum mean discrepancy loss is developed to mitigate the distribution
divergence between the synthesized and original features. To learn natural
deformation without dense supervision, we introduce a coarse parameterized
deformation graph, for which a novel trace and propagation algorithm is
proposed to improve both the quality and effciency of the deformation. Our
UD^2E-Net outperforms state-of-the-art unsupervised methods by 24% on Faust
Inter challenge and even supervised methods by 13% on Faust Intra challenge.

    

### [[2108.11618] Few-shot Visual Relationship Co-localization](http://arxiv.org/abs/2108.11618)


  In this paper, given a small bag of images, each containing a common but
latent predicate, we are interested in localizing visual subject-object pairs
connected via the common predicate in each of the images. We refer to this
novel problem as visual relationship co-localization or VRC as an abbreviation.
VRC is a challenging task, even more so than the well-studied object
co-localization task. This becomes further challenging when using just a few
images, the model has to learn to co-localize visual subject-object pairs
connected via unseen predicates. To solve VRC, we propose an optimization
framework to select a common visual relationship in each image of the bag. The
goal of the optimization framework is to find the optimal solution by learning
visual relationship similarity across images in a few-shot setting. To obtain
robust visual relationship representation, we utilize a simple yet effective
technique that learns relationship embedding as a translation vector from
visual subject to visual object in a shared space. Further, to learn visual
relationship similarity, we utilize a proven meta-learning technique commonly
used for few-shot classification tasks. Finally, to tackle the combinatorial
complexity challenge arising from an exponential number of feasible solutions,
we use a greedy approximation inference algorithm that selects approximately
the best solution.
We extensively evaluate our proposed framework on variations of bag sizes
obtained from two challenging public datasets, namely VrR-VG and VG-150, and
achieve impressive visual co-localization performance.

    

### [[2108.11626] CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation](http://arxiv.org/abs/2108.11626)


  As the use of interactive machines grow, the task of Emotion Recognition in
Conversation (ERC) became more important. If the machine generated sentences
reflect emotion, more human-like sympathetic conversations are possible. Since
emotion recognition in conversation is inaccurate if the previous utterances
are not taken into account, many studies reflect the dialogue context to
improve the performances. We introduce CoMPM, a context embedding module (CoM)
combined with a pre-trained memory module (PM) that tracks memory of the
speaker's previous utterances within the context, and show that the pre-trained
memory significantly improves the final accuracy of emotion recognition. We
experimented on both the multi-party datasets (MELD, EmoryNLP) and the
dyadic-party datasets (IEMOCAP, DailyDialog), showing that our approach achieve
competitive performance on all datasets.

    

### [[2108.11635] MCML: A Novel Memory-based Contrastive Meta-Learning Method for Few Shot Slot Tagging](http://arxiv.org/abs/2108.11635)


  Meta-learning is widely used for few-shot slot tagging in the task of
few-shot learning. The performance of existing methods is, however, seriously
affected by catastrophic forgetting. This phenomenon is common in deep learning
as the training and testing modules fail to take into account historical
information, i.e. previously trained episodes in the metric-based
meta-learning. To overcome this predicament, we propose the Memory-based
Contrastive Meta-learning (MCML) method. Specifically, we propose a
learn-from-memory mechanism that use explicit memory to keep track of the label
representations of previously trained episodes and propose a contrastive
learning method to compare the current label embedded in the few shot episode
with the historic ones stored in the memory, and an adaption-from memory
mechanism to determine the output label based on the contrast between the input
labels embedded in the test episode and the label clusters in the memory.
Experimental results show that MCML is scalable and outperforms metric-based
meta-learning and optimization-based meta-learning on all 1shot, 5-shot,
10-shot, and 20-shot scenarios of the SNIPS dataset.

    

### [[2108.11645] Robust Model-based Reinforcement Learning for Autonomous Greenhouse Control](http://arxiv.org/abs/2108.11645)


  Due to the high efficiency and less weather dependency, autonomous
greenhouses provide an ideal solution to meet the increasing demand for fresh
food. However, managers are faced with some challenges in finding appropriate
control strategies for crop growth, since the decision space of the greenhouse
control problem is an astronomical number. Therefore, an intelligent
closed-loop control framework is highly desired to generate an automatic
control policy. As a powerful tool for optimal control, reinforcement learning
(RL) algorithms can surpass human beings' decision-making and can also be
seamlessly integrated into the closed-loop control framework. However, in
complex real-world scenarios such as agricultural automation control, where the
interaction with the environment is time-consuming and expensive, the
application of RL algorithms encounters two main challenges, i.e., sample
efficiency and safety. Although model-based RL methods can greatly mitigate the
efficiency problem of greenhouse control, the safety problem has not got too
much attention. In this paper, we present a model-based robust RL framework for
autonomous greenhouse control to meet the sample efficiency and safety
challenges. Specifically, our framework introduces an ensemble of environment
models to work as a simulator and assist in policy optimization, thereby
addressing the low sample efficiency problem. As for the safety concern, we
propose a sample dropout module to focus more on worst-case samples, which can
help improve the adaptability of the greenhouse planting policy in extreme
cases. Experimental results demonstrate that our approach can learn a more
effective greenhouse planting policy with better robustness than existing
methods.

    

### [[2108.11663] Convolutional Neural Networks Demystified: A Matched Filtering Perspective Based Tutorial](http://arxiv.org/abs/2108.11663)


  Deep Neural Networks (DNN) and especially Convolutional Neural Networks (CNN)
are a de-facto standard for the analysis of large volumes of signals and
images. Yet, their development and underlying principles have been largely
performed in an ad-hoc and black box fashion. To help demystify CNNs, we
revisit their operation from first principles and a matched filtering
perspective. We establish that the convolution operation within CNNs, their
very backbone, represents a matched filter which examines the input
signal/image for the presence of pre-defined features. This perspective is
shown to be physically meaningful, and serves as a basis for a step-by-step
tutorial on the operation of CNNs, including pooling, zero padding, various
ways of dimensionality reduction. Starting from first principles, both the
feed-forward pass and the learning stage (via back-propagation) are illuminated
in detail, both through a worked-out numerical example and the corresponding
visualizations. It is our hope that this tutorial will help shed new light and
physical intuition into the understanding and further development of deep
neural networks.

    

### [[2108.11711] SLIM: Explicit Slot-Intent Mapping with BERT for Joint Multi-Intent Detection and Slot Filling](http://arxiv.org/abs/2108.11711)


  Utterance-level intent detection and token-level slot filling are two key
tasks for natural language understanding (NLU) in task-oriented systems. Most
existing approaches assume that only a single intent exists in an utterance.
However, there are often multiple intents within an utterance in real-life
scenarios. In this paper, we propose a multi-intent NLU framework, called SLIM,
to jointly learn multi-intent detection and slot filling based on BERT. To
fully exploit the existing annotation data and capture the interactions between
slots and intents, SLIM introduces an explicit slot-intent classifier to learn
the many-to-one mapping between slots and intents. Empirical results on three
public multi-intent datasets demonstrate (1) the superior performance of SLIM
compared to the current state-of-the-art for NLU with multiple intents and (2)
the benefits obtained from the slot-intent classifier.

    

### [[2108.11714] Photos Are All You Need for Reciprocal Recommendation in Online Dating](http://arxiv.org/abs/2108.11714)


  Recommender Systems are algorithms that predict a user's preference for an
item. Reciprocal Recommenders are a subset of recommender systems, where the
items in question are people, and the objective is therefore to predict a
bidirectional preference relation. They are used in settings such as online
dating services and social networks. In particular, images provided by users
are a crucial part of user preference, and one that is not exploited much in
the literature. We present a novel method of interpreting user image preference
history and using this to make recommendations. We train a recurrent neural
network to learn a user's preferences and make predictions of reciprocal
preference relations that can be used to make recommendations that satisfy both
users. We show that our proposed system achieves an F1 score of 0.87 when using
only photographs to produce reciprocal recommendations on a large real world
online dating dataset. Our system significantly outperforms on the state of the
art in both content-based and collaborative filtering systems.

    

### [[2108.11762] Disentangling What and Where for 3D Object-Centric Representations Through Active Inference](http://arxiv.org/abs/2108.11762)


  Although modern object detection and classification models achieve high
accuracy, these are typically constrained in advance on a fixed train set and
are therefore not flexible to deal with novel, unseen object categories.
Moreover, these models most often operate on a single frame, which may yield
incorrect classifications in case of ambiguous viewpoints. In this paper, we
propose an active inference agent that actively gathers evidence for object
classifications, and can learn novel object categories over time. Drawing
inspiration from the human brain, we build object-centric generative models
composed of two information streams, a what- and a where-stream. The
what-stream predicts whether the observed object belongs to a specific
category, while the where-stream is responsible for representing the object in
its internal 3D reference frame. We show that our agent (i) is able to learn
representations for many object categories in an unsupervised way, (ii)
achieves state-of-the-art classification accuracies, actively resolving
ambiguity when required and (iii) identifies novel object categories.
Furthermore, we validate our system in an end-to-end fashion where the agent is
able to search for an object at a given pose from a pixel-based rendering. We
believe that this is a first step towards building modular, intelligent systems
that can be used for a wide range of tasks involving three dimensional objects.

    

### [[2108.11791] Multiple Sclerosis Lesions Identification/Segmentation in Magnetic Resonance Imaging using Ensemble CNN and Uncertainty Classification](http://arxiv.org/abs/2108.11791)


  To date, several automated strategies for identification/segmentation of
Multiple Sclerosis (MS) lesions by Magnetic Resonance Imaging (MRI) have been
presented which are either outperformed by human experts or, at least, whose
results are well distinguishable from humans. This is due to the ambiguity
originated by MRI instabilities, peculiar MS Heterogeneity and MRI unspecific
nature with respect to MS. Physicians partially treat the uncertainty generated
by ambiguity relying on personal radiological/clinical/anatomical background
and experience.
We present an automated framework for MS lesions identification/segmentation
based on three pivotal concepts to better emulate human reasoning: the modeling
of uncertainty; the proposal of two, separately trained, CNN, one optimized
with respect to lesions themselves and the other to the environment surrounding
lesions, respectively repeated for axial, coronal and sagittal directions; the
ensemble of the CNN output.
The proposed framework is trained, validated and tested on the 2016 MSSEG
benchmark public data set from a single imaging modality, FLuid-Attenuated
Inversion Recovery (FLAIR). The comparison, performed on the segmented lesions
by means of most of the metrics normally used with respect to the ground-truth
and the 7 human raters in MSSEG, prove that there is no significant difference
between the proposed framework and the other raters. Results are also shown for
the uncertainty, though a comparison with the other raters is impossible.

    

### [[2108.11824] Magnetic Field Sensing for Pedestrian and Robot Indoor Positioning](http://arxiv.org/abs/2108.11824)


  In this paper we address the problem of indoor localization using magnetic
field data in two setups, when data is collected by (i) human-held mobile phone
and (ii) by localization robots that perturb magnetic data with their own
electromagnetic field. For the first setup, we revise the state of the art
approaches and propose a novel extended pipeline to benefit from the presence
of magnetic anomalies in indoor environment created by different ferromagnetic
objects. We capture changes of the Earth's magnetic field due to indoor
magnetic anomalies and transform them in multi-variate times series. We then
convert temporal patterns into visual ones. We use methods of Recurrence Plots,
Gramian Angular Fields and Markov Transition Fields to represent magnetic field
time series as image sequences. We regress the continuous values of user
position in a deep neural network that combines convolutional and recurrent
layers. For the second setup, we analyze how magnetic field data get perturbed
by robots' electromagnetic field. We add an alignment step to the main
pipeline, in order to compensate the mismatch between train and test sets
obtained by different robots. We test our methods on two public (MagPie and
IPIN'20) and one proprietary (Hyundai department store) datasets. We report
evaluation results and show that our methods outperform the state of the art
methods by a large margin.

    

### [[2108.11833] Gene Transformer: Transformers for the Gene Expression-based Classification of Cancer Subtypes](http://arxiv.org/abs/2108.11833)


  Adenocarcinoma and squamous cell carcinoma constitute approximately 40% and
30% of all lung cancer subtypes, respectively, and display broad heterogeneity
in terms of clinical and molecular responses to therapy. Molecular subtyping
has enabled precision medicine to overcome these challenges and provide
significant biological insights to predict prognosis and improve clinical
decision making. Over the past decade, conventional ML algorithms and DL-based
CNNs have been espoused for the classification of cancer subtypes from gene
expression datasets. However, these methods are potentially biased toward
identification of cancer biomarkers. Recently proposed transformer-based
architectures that leverage the self-attention mechanism encode high throughput
gene expressions and learn representations that are computationally complex and
parametrically expensive. However, compared to the datasets for natural
language processing applications, gene expression consists of several hundreds
of thousands of genes from a limited number of observations, making it
difficult to efficiently train transformers for bioinformatics applications.
Hence, we propose an end-to-end deep learning approach, Gene Transformer, which
addresses the complexity of high-dimensional gene expression with a multi-head
self-attention module by identifying relevant biomarkers across multiple cancer
subtypes without requiring feature selection as a prerequisite for the current
classification algorithms. The proposed architecture achieved an overall
improved performance for all evaluation metrics and had fewer misclassification
errors than the commonly used traditional classification algorithms. The
classification results show that Gene Transformer can be an efficient approach
for classifying cancer subtypes, indicating that any improvement in deep
learning models in computational biology can also be reflected well in this
domain.

    

### [[2108.11838] Geometry Based Machining Feature Retrieval with Inductive Transfer Learning](http://arxiv.org/abs/2108.11838)


  Manufacturing industries have widely adopted the reuse of machine parts as a
method to reduce costs and as a sustainable manufacturing practice.
Identification of reusable features from the design of the parts and finding
their similar features from the database is an important part of this process.
In this project, with the help of fully convolutional geometric features, we
are able to extract and learn the high level semantic features from CAD models
with inductive transfer learning. The extracted features are then compared with
that of other CAD models from the database using Frobenius norm and identical
features are retrieved. Later we passed the extracted features to a deep
convolutional neural network with a spatial pyramid pooling layer and the
performance of the feature retrieval increased significantly. It was evident
from the results that the model could effectively capture the geometrical
elements from machining features.

    

### [[2108.11844] AI at work -- Mitigating safety and discriminatory risk with technical standards](http://arxiv.org/abs/2108.11844)


  The use of artificial intelligence (AI) and AI methods in the workplace holds
both great opportunities as well as risks to occupational safety and
discrimination. In addition to legal regulation, technical standards will play
a key role in mitigating such risk by defining technical requirements for
development and testing of AI systems. This paper provides an overview and
assessment of existing international, European and German standards as well as
those currently under development. The paper is part of the research project
"ExamAI - Testing and Auditing of AI systems" and focusses on the use of AI in
an industrial production environment as well as in the realm of human resource
management (HR).

    

### [[2108.11885] Human operator cognitive availability aware Mixed-Initiative control](http://arxiv.org/abs/2108.11885)


  This paper presents a Cognitive Availability Aware Mixed-Initiative
Controller for remotely operated mobile robots. The controller enables dynamic
switching between different levels of autonomy (LOA), initiated by either the
AI or the human operator. The controller leverages a state-of-the-art computer
vision method and an off-the-shelf web camera to infer the cognitive
availability of the operator and inform the AI-initiated LOA switching. This
constitutes a qualitative advancement over previous Mixed-Initiative (MI)
controllers. The controller is evaluated in a disaster response experiment, in
which human operators have to conduct an exploration task with a remote robot.
MI systems are shown to effectively assist the operators, as demonstrated by
quantitative and qualitative results in performance and workload. Additionally,
some insights into the experimental difficulties of evaluating complex MI
controllers are presented.

    

### [[2108.11949] Weisfeiler-Leman in the BAMBOO: Novel AMR Graph Metrics and a Benchmark for AMR Graph Similarity](http://arxiv.org/abs/2108.11949)


  Several metrics have been proposed for assessing the similarity of (abstract)
meaning representations (AMRs), but little is known about how they relate to
human similarity ratings. Moreover, the current metrics have complementary
strengths and weaknesses: some emphasize speed, while others make the alignment
of graph structures explicit, at the price of a costly alignment step.
In this work we propose new Weisfeiler-Leman AMR similarity metrics that
unify the strengths of previous metrics, while mitigating their weaknesses.
Specifically, our new metrics are able to match contextualized substructures
and induce n:m alignments between their nodes. Furthermore, we introduce a
Benchmark for AMR Metrics based on Overt Objectives (BAMBOO), the first
benchmark to support empirical assessment of graph-based MR similarity metrics.
BAMBOO maximizes the interpretability of results by defining multiple overt
objectives that range from sentence similarity objectives to stress tests that
probe a metric's robustness against meaning-altering and meaning-preserving
graph transformations. We show the benefits of BAMBOO by profiling previous
metrics and our own metrics. Results indicate that our novel metrics may serve
as a strong baseline for future work.

    

### [[2104.04670] Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections](http://arxiv.org/abs/2104.04670)


  Large pre-trained language models (LMs) such as GPT-3 have acquired a
surprising ability to perform zero-shot learning. For example, to classify
sentiment without any training examples, we can "prompt" the LM with the review
and the label description "Does the user like this movie?", and ask whether the
next word is "yes" or "no". However, the next word prediction training
objective is still misaligned with the target zero-shot learning objective. To
address this weakness, we propose meta-tuning, which directly optimizes the
zero-shot learning objective by fine-tuning pre-trained language models on a
collection of datasets. We focus on classification tasks, and construct the
meta-dataset by aggregating 43 existing datasets and annotating 441 label
descriptions in a question-answering (QA) format. When evaluated on unseen
tasks, meta-tuned models outperform a same-sized QA model and the previous SOTA
zero-shot learning system based on natural language inference. Additionally,
increasing parameter count from 220M to 770M improves AUC-ROC scores by 6.3%,
and we forecast that even larger models would perform better. Therefore,
measuring zero-shot learning performance on language models out-of-the-box
might underestimate their true potential, and community-wide efforts on
aggregating datasets and unifying their formats can help build models that
answer prompts better.

    

### [[2104.05755] Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning Workloads](http://arxiv.org/abs/2104.05755)


  During the past decade, novel Deep Learning (DL) algorithms/workloads and
hardware have been developed to tackle a wide range of problems. Despite the
advances in workload/hardware ecosystems, the programming methodology of DL
systems is stagnant. DL workloads leverage either highly optimized, yet
platform-specific and inflexible kernels from DL libraries, or in the case of
novel operators, reference implementations are built via DL framework
primitives with underwhelming performance. This work introduces the Tensor
Processing Primitives (TPP), a programming abstraction striving for efficient,
portable implementation of DL workloads with high productivity. TPPs define a
compact, yet versatile set of 2D-tensor operators (or a virtual Tensor ISA),
which subsequently can be utilized as building blocks to construct complex
operators on high-dimensional tensors. The TPP specification is
platform-agnostic, thus code expressed via TPPs is portable, whereas the TPP
implementation is highly optimized and platform-specific. We demonstrate the
efficacy of our approach using standalone kernels and end-to-end DL workloads
expressed entirely via TPPs that outperform state-of-the-art implementations on
multiple platforms.

    

### [[2108.11426] Visualizing JIT Compiler Graphs](http://arxiv.org/abs/2108.11426)


  Just-in-time (JIT) compilers are used by many modern programming systems in
order to improve performance. Bugs in JIT compilers provide exploitable
security vulnerabilities and debugging them is difficult as they are large,
complex, and dynamic. Current debugging and visualization tools deal with
static code and are not suitable in this domain. We describe a new approach for
simplifying the large and complex intermediate representation, generated by a
JIT compiler and visualize it with a metro map metaphor to aid developers in
debugging.

    

### [[2108.11651] Scalable and Modular Robustness Analysis of Deep Neural Networks](http://arxiv.org/abs/2108.11651)


  As neural networks are trained to be deeper and larger, the scalability of
neural network analyzers is urgently required. The main technical insight of
our method is modularly analyzing neural networks by segmenting a network into
blocks and conduct the analysis for each block. In particular, we propose the
network block summarization technique to capture the behaviors within a network
block using a block summary and leverage the summary to speed up the analysis
process. We instantiate our method in the context of a CPU-version of the
state-of-the-art analyzer DeepPoly and name our system as Bounded-Block Poly
(BBPoly). We evaluate BBPoly extensively on various experiment settings. The
experimental result indicates that our method yields comparable precision as
DeepPoly but runs faster and requires less computational resources. For
example, BBPoly can analyze really large neural networks like SkipNet or ResNet
which contain up to one million neurons in less than around 1 hour per input
image, while DeepPoly needs to spend even 40 hours to analyze one image.

    

### [[2108.11679] A Program Instrumentation for Prefix-Based Tracing in Message-Passing Concurrency](http://arxiv.org/abs/2108.11679)


  The execution of concurrent programs generally involves some degree of
nondeterminism, mostly due to the relative speeds of the concurrent processes.
As a consequence, reproducibility is often challenging. This problem has been
traditionally tackled by a combination of tracing and replay. In this paper, we
introduce a program instrumentation for "prefix-based tracing" that combines
both tracing and replay. In the general case, the program is instrumented with
a partial trace, so that the execution first follows the partial trace (replay)
and, then, proceeds nondeterministically, eventually producing a trace of the
complete execution as a side effect. Observe that traditional tracing and
replay are particular cases of our approach when an empty trace is provided
(pure tracing) and when a full trace is provided (pure replay), respectively.

    

### [[2108.11867] A Typed Programmatic Interface to Contracts on the Blockchain](http://arxiv.org/abs/2108.11867)


  Smart contract applications on the blockchain can only reach their full
potential if they integrate seamlessly with traditional software systems via a
programmatic interface. This interface should provide for originating and
invoking contracts as well as observing the state of the blockchain. We propose
a typed API for this purpose and establish some properties of the combined
system. Specifically, we provide an execution model that enables us to prove
type-safe interaction between programs and the blockchain. We establish further
properties of the model that give rise to requirements on the API. A prototype
of the interface is implemented in OCaml for the Tezos blockchain.

    