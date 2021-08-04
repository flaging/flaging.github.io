
## 2021-8-4

### [[2108.01544] Controlled Deep Reinforcement Learning for Optimized Slice Placement](http://arxiv.org/abs/2108.01544)


  We present a hybrid ML-heuristic approach that we name "Heuristically
Assisted Deep Reinforcement Learning (HA-DRL)" to solve the problem of Network
Slice Placement Optimization. The proposed approach leverages recent works on
Deep Reinforcement Learning (DRL) for slice placement and Virtual Network
Embedding (VNE) and uses a heuristic function to optimize the exploration of
the action space by giving priority to reliable actions indicated by an
efficient heuristic algorithm. The evaluation results show that the proposed
HA-DRL algorithm can accelerate the learning of an efficient slice placement
policy improving slice acceptance ratio when compared with state-of-the-art
approaches that are based only on reinforcement learning.

    

### [[2108.01598] Secure and Efficient Blockchain based Knowledge Sharing for Connected Autonomous Vehicles](http://arxiv.org/abs/2108.01598)


  The emergence of Connected Autonomous Vehicles (CAVs) shows great potential
for future intelligent traffic systems, enhancing both traffic safety and road
efficiency. However, the CAVs relying on data driven perception and driving
models face many challenges, including the lack of comprehensive knowledge to
deal with complicated driving context. In this paper, we are motivated to
investigate cooperative knowledge sharing for CAVs. We propose a secure and
efficient directed acyclic graph (DAG) blockchain based knowledge sharing
framework, aiming to cater for the micro-transaction based vehicular networks.
The framework can realize both local and cross-regional knowledge sharing.
Then, the framework is applied to autonomous driving applications, wherein
machine learning based models for autonomous driving control can be shared. A
lightweight tip selection algorithm (TSA) is proposed for the DAG based
knowledge sharing framework to achieve consensus and identity verification for
cross-regional vehicles. To enhance model accuracy as well as minimizing
bandwidth consumption, an adaptive asynchronous distributed learning (ADL)
based scheme is proposed for model uploading and downloading. Experiment
results show that the blockchain based knowledge sharing is secure, and it can
resist attacks from malicious users. In addition, the proposed adaptive ADL
scheme can enhance driving safety related performance compared to several
existing algorithms.

    

### [[2108.01077] Generating Master Faces for Dictionary Attacks with a Network-Assisted Latent Space Evolution](http://arxiv.org/abs/2108.01077)


  A master face is a face image that passes face-based identity-authentication
for a large portion of the population. These faces can be used to impersonate,
with a high probability of success, any user, without having access to any user
information. We optimize these faces, by using an evolutionary algorithm in the
latent embedding space of the StyleGAN face generator. Multiple evolutionary
strategies are compared, and we propose a novel approach that employs a neural
network in order to direct the search in the direction of promising samples,
without adding fitness evaluations. The results we present demonstrate that it
is possible to obtain a high coverage of the population (over 40%) with less
than 10 master faces, for three leading deep face recognition systems.

    

### [[2108.01080] Learning-based Preference Prediction for Constrained Multi-Criteria Path-Planning](http://arxiv.org/abs/2108.01080)


  Learning-based methods are increasingly popular for search algorithms in
single-criterion optimization problems. In contrast, for multiple-criteria
optimization there are significantly fewer approaches despite the existence of
numerous applications. Constrained path-planning for Autonomous Ground Vehicles
(AGV) is one such application, where an AGV is typically deployed in disaster
relief or search and rescue applications in off-road environments. The agent
can be faced with the following dilemma : optimize a source-destination path
according to a known criterion and an uncertain criterion under operational
constraints. The known criterion is associated to the cost of the path,
representing the distance. The uncertain criterion represents the feasibility
of driving through the path without requiring human intervention. It depends on
various external parameters such as the physics of the vehicle, the state of
the explored terrains or weather conditions. In this work, we leverage
knowledge acquired through offline simulations by training a neural network
model to predict the uncertain criterion. We integrate this model inside a
path-planner which can solve problems online. Finally, we conduct experiments
on realistic AGV scenarios which illustrate that the proposed framework
requires human intervention less frequently, trading for a limited increase in
the path distance.

    

### [[2108.01099] Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training data](http://arxiv.org/abs/2108.01099)


  There has been a recent surge of interest in designing Graph Neural Networks
(GNNs) for semi-supervised learning tasks. Unfortunately this work has assumed
that the nodes labeled for use in training were selected uniformly at random
(i.e. are an IID sample). However in many real world scenarios gathering labels
for graph nodes is both expensive and inherently biased -- so this assumption
can not be met. GNNs can suffer poor generalization when this occurs, by
overfitting to superfluous regularities present in the training data. In this
work we present a method, Shift-Robust GNN (SR-GNN), designed to account for
distributional differences between biased training data and the graph's true
inference distribution. SR-GNN adapts GNN models for the presence of
distributional shifts between the nodes which have had labels provided for
training and the rest of the dataset. We illustrate the effectiveness of SR-GNN
in a variety of experiments with biased training datasets on common GNN
benchmark datasets for semi-supervised learning, where we see that SR-GNN
outperforms other GNN baselines by accuracy, eliminating at least (~40%) of the
negative effects introduced by biased training data. On the largest dataset we
consider, ogb-arxiv, we observe an 2% absolute improvement over the baseline
and reduce 30% of the negative effects.

    

### [[2108.01110] Batch Normalization Preconditioning for Neural Network Training](http://arxiv.org/abs/2108.01110)


  Batch normalization (BN) is a popular and ubiquitous method in deep learning
that has been shown to decrease training time and improve generalization
performance of neural networks. Despite its success, BN is not theoretically
well understood. It is not suitable for use with very small mini-batch sizes or
online learning. In this paper, we propose a new method called Batch
Normalization Preconditioning (BNP). Instead of applying normalization
explicitly through a batch normalization layer as is done in BN, BNP applies
normalization by conditioning the parameter gradients directly during training.
This is designed to improve the Hessian matrix of the loss function and hence
convergence during training. One benefit is that BNP is not constrained on the
mini-batch size and works in the online learning setting. Furthermore, its
connection to BN provides theoretical insights on how BN improves training and
how BN is applied to special architectures such as convolutional neural
networks.

    

### [[2108.01111] Pre-trained Models for Sonar Images](http://arxiv.org/abs/2108.01111)


  Machine learning and neural networks are now ubiquitous in sonar perception,
but it lags behind the computer vision field due to the lack of data and
pre-trained models specifically for sonar images. In this paper we present the
Marine Debris Turntable dataset and produce pre-trained neural networks trained
on this dataset, meant to fill the gap of missing pre-trained models for sonar
images. We train Resnet 20, MobileNets, DenseNet121, SqueezeNet, MiniXception,
and an Autoencoder, over several input image sizes, from 32 x 32 to 96 x 96, on
the Marine Debris turntable dataset. We evaluate these models using transfer
learning for low-shot classification in the Marine Debris Watertank and another
dataset captured using a Gemini 720i sonar. Our results show that in both
datasets the pre-trained models produce good features that allow good
classification accuracy with low samples (10-30 samples per class). The Gemini
dataset validates that the features transfer to other kinds of sonar sensors.
We expect that the community benefits from the public release of our
pre-trained models and the turntable dataset.

    

### [[2108.01123] Metodos de Agrupamentos em dois Estagios](http://arxiv.org/abs/2108.01123)


  This work investigates the use of two-stage clustering methods. Four
techniques were proposed: SOMK, SOMAK, ASCAK and SOINAK. SOMK is composed of a
SOM (Self-Organizing Maps) followed by the K-means algorithm, SOMAK is a
combination of SOM followed by the Ant K-means (AK) algorithm, ASCAK is
composed by the ASCA (Ant System-based Clustering Algorithm) and AK algorithms,
SOINAK is composed by the Self-Organizing Incremental Neural Network (SOINN)
and AK. SOINAK presented a better performance among the four proposed
techniques when applied to pattern recognition problems.

    

### [[2108.01124] Efficacy of Statistical and Artificial Intelligence-based False Information Cyberattack Detection Models for Connected Vehicles](http://arxiv.org/abs/2108.01124)


  Connected vehicles (CVs), because of the external connectivity with other CVs
and connected infrastructure, are vulnerable to cyberattacks that can instantly
compromise the safety of the vehicle itself and other connected vehicles and
roadway infrastructure. One such cyberattack is the false information attack,
where an external attacker injects inaccurate information into the connected
vehicles and eventually can cause catastrophic consequences by compromising
safety-critical applications like the forward collision warning. The occurrence
and target of such attack events can be very dynamic, making real-time and
near-real-time detection challenging. Change point models, can be used for
real-time anomaly detection caused by the false information attack. In this
paper, we have evaluated three change point-based statistical models;
Expectation Maximization, Cumulative Summation, and Bayesian Online Change
Point Algorithms for cyberattack detection in the CV data. Also, data-driven
artificial intelligence (AI) models, which can be used to detect known and
unknown underlying patterns in the dataset, have the potential of detecting a
real-time anomaly in the CV data. We have used six AI models to detect false
information attacks and compared the performance for detecting the attacks with
our developed change point models. Our study shows that change points models
performed better in real-time false information attack detection compared to
the performance of the AI models. Change point models having the advantage of
no training requirements can be a feasible and computationally efficient
alternative to AI models for false information attack detection in connected
vehicles.

    

### [[2108.01125] Hybrid Classical-Quantum Deep Learning Models for Autonomous Vehicle Traffic Image Classification Under Adversarial Attack](http://arxiv.org/abs/2108.01125)


  Image classification must work for autonomous vehicles (AV) operating on
public roads, and actions performed based on image misclassification can have
serious consequences. Traffic sign images can be misclassified by an
adversarial attack on machine learning models used by AVs for traffic sign
recognition. To make classification models resilient against adversarial
attacks, we used a hybrid deep-learning model with both the quantum and
classical layers. Our goal is to study the hybrid deep-learning architecture
for classical-quantum transfer learning models to support the current era of
intermediate-scale quantum technology. We have evaluated the impacts of various
white box adversarial attacks on these hybrid models. The classical part of
hybrid models includes a convolution network from the pre-trained Resnet18
model, which extracts informative features from a high dimensional LISA traffic
sign image dataset. The output from the classical processor is processed
further through the quantum layer, which is composed of various quantum gates
and provides support to various quantum mechanical features like entanglement
and superposition. We have tested multiple combinations of quantum circuits to
provide better classification accuracy with decreasing training data and found
better resiliency for our hybrid classical-quantum deep learning model during
attacks compared to the classical-only machine learning models.

    

### [[2108.01127] Hybrid Quantum-Classical Neural Network for Incident Detection](http://arxiv.org/abs/2108.01127)


  The efficiency and reliability of real-time incident detection models
directly impact the affected corridors' traffic safety and operational
conditions. The recent emergence of cloud-based quantum computing
infrastructure and innovations in noisy intermediate-scale quantum devices have
revealed a new era of quantum-enhanced algorithms that can be leveraged to
improve real-time incident detection accuracy. In this research, a hybrid
machine learning model, which includes classical and quantum machine learning
(ML) models, is developed to identify incidents using the connected vehicle
(CV) data. The incident detection performance of the hybrid model is evaluated
against baseline classical ML models. The framework is evaluated using data
from a microsimulation tool for different incident scenarios. The results
indicate that a hybrid neural network containing a 4-qubit quantum layer
outperforms all other baseline models when there is a lack of training data. We
have created three datasets; DS-1 with sufficient training data, and DS-2 and
DS-3 with insufficient training data. The hybrid model achieves a recall of
98.9%, 98.3%, and 96.6% for DS-1, DS-2, and DS-3, respectively. For DS-2 and
DS-3, the average improvement in F2-score (measures model's performance to
correctly identify incidents) achieved by the hybrid model is 1.9% and 7.8%,
respectively, compared to the classical models. It shows that with insufficient
data, which may be common for CVs, the hybrid ML model will perform better than
the classical models. With the continuing improvements of quantum computing
infrastructure, the quantum ML models could be a promising alternative for
CV-related applications when the available data is insufficient.

    

### [[2108.01139] PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors](http://arxiv.org/abs/2108.01139)


  EuroVoc is a multilingual thesaurus that was built for organizing the
legislative documentary of the European Union institutions. It contains
thousands of categories at different levels of specificity and its descriptors
are targeted by legal texts in almost thirty languages. In this work we propose
a unified framework for EuroVoc classification on 22 languages by fine-tuning
modern Transformer-based pretrained language models. We study extensively the
performance of our trained models and show that they significantly improve the
results obtained by a similar tool - JEX - on the same dataset. The code and
the fine-tuned models were open sourced, together with a programmatic interface
that eases the process of loading the weights of a trained model and of
classifying a new document.

    

### [[2108.01141] Correcting Arabic Soft Spelling Mistakes using BiLSTM-based Machine Learning](http://arxiv.org/abs/2108.01141)


  Soft spelling errors are a class of spelling mistakes that is widespread
among native Arabic speakers and foreign learners alike. Some of these errors
are typographical in nature. They occur due to orthographic variations of some
Arabic letters and the complex rules that dictate their correct usage. Many
people forgo these rules, and given the identical phonetic sounds, they often
confuse such letters. In this paper, we propose a bidirectional long short-term
memory network that corrects this class of errors. We develop, train, evaluate,
and compare a set of BiLSTM networks. We approach the spelling correction
problem at the character level. We handle Arabic texts from both classical and
modern standard Arabic. We treat the problem as a one-to-one sequence
transcription problem. Since the soft Arabic errors class encompasses omission
and addition mistakes, to preserve the one-to-one sequence transcription, we
propose a simple low-resource yet effective technique that maintains the
one-to-one sequencing and avoids using a costly encoder-decoder architecture.
We train the BiLSTM models to correct the spelling mistakes using transformed
input and stochastic error injection approaches. We recommend a configuration
that has two BiLSTM layers, uses the dropout regularization, and is trained
using the latter training approach with error injection rate of 40%. The best
model corrects 96.4% of the injected errors and achieves a low character error
rate of 1.28% on a real test set of soft spelling mistakes.

    

### [[2108.01152] Pure Exploration in Multi-armed Bandits with Graph Side Information](http://arxiv.org/abs/2108.01152)


  We study pure exploration in multi-armed bandits with graph side-information.
In particular, we consider the best arm (and near-best arm) identification
problem in the fixed confidence setting under the assumption that the arm
rewards are smooth with respect to a given arbitrary graph. This captures a
range of real world pure-exploration scenarios where one often has information
about the similarity of the options or actions under consideration. We propose
a novel algorithm GRUB (GRaph based UcB) for this problem and provide a
theoretical characterization of its performance that elicits the benefit of the
graph-side information. We complement our theory with experimental results that
show that capitalizing on available graph side information yields significant
improvements over pure exploration methods that are unable to use this
information.

    

### [[2108.01181] Waveform Selection for Radar Tracking in Target Channels With Memory via Universal Learning](http://arxiv.org/abs/2108.01181)


  In tracking radar, the sensing environment often varies significantly over a
track duration due to the target's trajectory and dynamic interference.
Adapting the radar's waveform using partial information about the state of the
scene has been shown to provide performance benefits in many practical
scenarios. Moreover, radar measurements generally exhibit strong temporal
correlation, allowing memory-based learning algorithms to effectively learn
waveform selection strategies. This work examines a radar system which builds a
compressed model of the radar-environment interface in the form of a
context-tree. The radar uses this context tree-based model to select waveforms
in a signal-dependent target channel, which may respond adversarially to the
radar's strategy. This approach is guaranteed to asymptotically converge to the
average-cost optimal policy for any stationary target channel that can be
represented as a Markov process of order U < $\infty$, where the constant U is
unknown to the radar. The proposed approach is tested in a simulation study,
and is shown to provide tracking performance improvements over two
state-of-the-art waveform selection schemes.

    

### [[2108.01192] Multi-objective Recurrent Neural Networks Optimization for the Edge -- a Quantization-based Approach](http://arxiv.org/abs/2108.01192)


  The compression of deep learning models is of fundamental importance in
deploying such models to edge devices. Incorporating hardware model and
application constraints during compression maximizes the benefits but makes it
specifically designed for one case. Therefore, the compression needs to be
automated. Searching for the optimal compression method parameters is
considered an optimization problem. This article introduces a Multi-Objective
Hardware-Aware Quantization (MOHAQ) method, which considers both hardware
efficiency and inference error as objectives for mixed-precision quantization.
The proposed method makes the evaluation of candidate solutions in a large
search space feasible by relying on two steps. First, post-training
quantization is applied for fast solution evaluation. Second, we propose a
search technique named "beacon-based search" to retrain selected solutions only
in the search space and use them as beacons to know the effect of retraining on
other solutions. To evaluate the optimization potential, we chose a speech
recognition model using the TIMIT dataset. The model is based on Simple
Recurrent Unit (SRU) due to its considerable speedup over other recurrent
units. We applied our method to run on two platforms: SiLago and Bitfusion.
Experimental evaluations showed that SRU can be compressed up to 8x by
post-training quantization without any significant increase in the error and up
to 12x with only a 1.5 percentage point increase in error. On SiLago, the
inference-only search found solutions that achieve 80\% and 64\% of the maximum
possible speedup and energy saving, respectively, with a 0.5 percentage point
increase in the error. On Bitfusion, with a constraint of a small SRAM size,
beacon-based search reduced the error gain of inference-only search by 4
percentage points and increased the possible reached speedup to be 47x compared
to the Bitfusion baseline.

    

### [[2108.01204] The RareDis corpus: a corpus annotated with rare diseases, their signs and symptoms](http://arxiv.org/abs/2108.01204)


  The RareDis corpus contains more than 5,000 rare diseases and almost 6,000
clinical manifestations are annotated. Moreover, the Inter Annotator Agreement
evaluation shows a relatively high agreement (F1-measure equal to 83.5% under
exact match criteria for the entities and equal to 81.3% for the relations).
Based on these results, this corpus is of high quality, supposing a significant
step for the field since there is a scarcity of available corpus annotated with
rare diseases. This could open the door to further NLP applications, which
would facilitate the diagnosis and treatment of these rare diseases and,
therefore, would improve dramatically the quality of life of these patients.

    

### [[2108.01210] Representation learning for neural population activity with Neural Data Transformers](http://arxiv.org/abs/2108.01210)


  Neural population activity is theorized to reflect an underlying dynamical
structure. This structure can be accurately captured using state space models
with explicit dynamics, such as those based on recurrent neural networks
(RNNs). However, using recurrence to explicitly model dynamics necessitates
sequential processing of data, slowing real-time applications such as
brain-computer interfaces. Here we introduce the Neural Data Transformer (NDT),
a non-recurrent alternative. We test the NDT's ability to capture autonomous
dynamical systems by applying it to synthetic datasets with known dynamics and
data from monkey motor cortex during a reaching task well-modeled by RNNs. The
NDT models these datasets as well as state-of-the-art recurrent models.
Further, its non-recurrence enables 3.9ms inference, well within the loop time
of real-time applications and more than 6 times faster than recurrent baselines
on the monkey reaching dataset. These results suggest that an explicit dynamics
model is not necessary to model autonomous neural population dynamics. Code:
this https URL


### [[2108.01215] Variational Actor-Critic Algorithms](http://arxiv.org/abs/2108.01215)


  We introduce a class of variational actor-critic algorithms based on a
variational formulation over both the value function and the policy. The
objective function of the variational formulation consists of two parts: one
for maximizing the value function and the other for minimizing the Bellman
residual. Besides the vanilla gradient descent with both the value function and
the policy updates, we propose two variants, the clipping method and the
flipping method, in order to speed up the convergence. We also prove that, when
the prefactor of the Bellman residual is sufficiently large, the fixed point of
the algorithm is close to the optimal policy.

    

### [[2108.01219] Computing the Newton-step faster than Hessian accumulation](http://arxiv.org/abs/2108.01219)


  Computing the Newton-step of a generic function with $N$ decision variables
takes $O(N^3)$ flops. In this paper, we show that given the computational graph
of the function, this bound can be reduced to $O(m\tau^3)$, where $\tau, m$ are
the width and size of a tree-decomposition of the graph. The proposed algorithm
generalizes nonlinear optimal-control methods based on LQR to general
optimization problems and provides non-trivial gains in iteration-complexity
even in cases where the Hessian is dense.

    

### [[2108.01220] OVERT: An Algorithm for Safety Verification of Neural Network Control Policies for Nonlinear Systems](http://arxiv.org/abs/2108.01220)


  Deep learning methods can be used to produce control policies, but certifying
their safety is challenging. The resulting networks are nonlinear and often
very large. In response to this challenge, we present OVERT: a sound algorithm
for safety verification of nonlinear discrete-time closed loop dynamical
systems with neural network control policies. The novelty of OVERT lies in
combining ideas from the classical formal methods literature with ideas from
the newer neural network verification literature. The central concept of OVERT
is to abstract nonlinear functions with a set of optimally tight piecewise
linear bounds. Such piecewise linear bounds are designed for seamless
integration into ReLU neural network verification tools. OVERT can be used to
prove bounded-time safety properties by either computing reachable sets or
solving feasibility queries directly. We demonstrate various examples of safety
verification for several classical benchmark examples. OVERT compares favorably
to existing methods both in computation time and in tightness of the reachable
set.

    

### [[2108.01224] Elastic Architecture Search for Diverse Tasks with Different Resources](http://arxiv.org/abs/2108.01224)


  We study a new challenging problem of efficient deployment for diverse tasks
with different resources, where the resource constraint and task of interest
corresponding to a group of classes are dynamically specified at testing time.
Previous NAS approaches seek to design architectures for all classes
simultaneously, which may not be optimal for some individual tasks. A
straightforward solution is to search an architecture from scratch for each
deployment scenario, which however is computation-intensive and impractical. To
address this, we present a novel and general framework, called Elastic
Architecture Search (EAS), permitting instant specializations at runtime for
diverse tasks with various resource constraints. To this end, we first propose
to effectively train the over-parameterized network via a task dropout strategy
to disentangle the tasks during training. In this way, the resulting model is
robust to the subsequent task dropping at inference time. Based on the
well-trained over-parameterized network, we then propose an efficient
architecture generator to obtain optimal architectures within a single forward
pass. Experiments on two image classification datasets show that EAS is able to
find more compact networks with better performance while remarkably being
orders of magnitude faster than state-of-the-art NAS methods. For example, our
proposed EAS finds compact architectures within 0.1 second for 50 deployment
scenarios.

    

### [[2108.01250] Your fairness may vary: Group fairness of pretrained language models in toxic text classification](http://arxiv.org/abs/2108.01250)


  We study the performance-fairness trade-off in more than a dozen fine-tuned
LMs for toxic text classification. We empirically show that no blanket
statement can be made with respect to the bias of large versus regular versus
compressed models. Moreover, we find that focusing on fairness-agnostic
performance metrics can lead to models with varied fairness characteristics.

    

### [[2108.01262] SABER: Data-Driven Motion Planner for Autonomously Navigating Heterogeneous Robots](http://arxiv.org/abs/2108.01262)


  We present an end-to-end online motion planning framework that uses a
data-driven approach to navigate a heterogeneous robot team towards a global
goal while avoiding obstacles in uncertain environments. First, we use
stochastic model predictive control (SMPC) to calculate control inputs that
satisfy robot dynamics, and consider uncertainty during obstacle avoidance with
chance constraints. Second, recurrent neural networks are used to provide a
quick estimate of future state uncertainty considered in the SMPC finite-time
horizon solution, which are trained on uncertainty outputs of various
simultaneous localization and mapping algorithms. When two or more robots are
in communication range, these uncertainties are then updated using a
distributed Kalman filtering approach. Lastly, a Deep Q-learning agent is
employed to serve as a high-level path planner, providing the SMPC with target
positions that move the robots towards a desired global goal. Our complete
methods are demonstrated on a ground and aerial robot simultaneously (code
available at: this https URL).

    

### [[2108.01265] Memorize, Factorize, or be Naïve: Learning Optimal Feature Interaction Methods for CTR Prediction](http://arxiv.org/abs/2108.01265)


  Click-through rate prediction is one of the core tasks in commercial
recommender systems. It aims to predict the probability of a user clicking a
particular item given user and item features. As feature interactions bring in
non-linearity, they are widely adopted to improve the performance of CTR
prediction models. Therefore, effectively modelling feature interactions has
attracted much attention in both the research and industry field. The current
approaches can generally be categorized into three classes: (1) naïve
methods, which do not model feature interactions and only use original
features; (2) memorized methods, which memorize feature interactions by
explicitly viewing them as new features and assigning trainable embeddings; (3)
factorized methods, which learn latent vectors for original features and
implicitly model feature interactions through factorization functions. Studies
have shown that modelling feature interactions by one of these methods alone
are suboptimal due to the unique characteristics of different feature
interactions. To address this issue, we first propose a general framework
called OptInter which finds the most suitable modelling method for each feature
interaction. Different state-of-the-art deep CTR models can be viewed as
instances of OptInter. To realize the functionality of OptInter, we also
introduce a learning algorithm that automatically searches for the optimal
modelling method. We conduct extensive experiments on four large datasets. Our
experiments show that OptInter improves the best performed state-of-the-art
baseline deep CTR models by up to 2.21%. Compared to the memorized method,
which also outperforms baselines, we reduce up to 91% parameters. In addition,
we conduct several ablation studies to investigate the influence of different
components of OptInter. Finally, we provide interpretable discussions on the
results of OptInter.

    

### [[2108.01267] Process Mining Model to Predict Mortality in Paralytic Ileus Patients](http://arxiv.org/abs/2108.01267)


  Paralytic Ileus (PI) patients are at high risk of death when admitted to the
Intensive care unit (ICU), with mortality as high as 40\%. There is minimal
research concerning PI patient mortality prediction. There is a need for more
accurate prediction modeling for ICU patients diagnosed with PI. This paper
demonstrates performance improvements in predicting the mortality of ICU
patients diagnosed with PI after 24 hours of being admitted. The proposed
framework, PMPI(Process Mining Model to predict mortality of PI patients), is a
modification of the work used for prediction of in-hospital mortality for ICU
patients with diabetes. PMPI demonstrates similar if not better performance
with an Area under the ROC Curve (AUC) score of 0.82 compared to the best
results of the existing literature. PMPI uses patient medical history, the time
related to the events, and demographic information for prediction. The PMPI
prediction framework has the potential to help medical teams in making better
decisions for treatment and care for ICU patients with PI to increase their
life expectancy.

    

### [[2108.01285] Toward Spatially Unbiased Generative Models](http://arxiv.org/abs/2108.01285)


  Recent image generation models show remarkable generation performance.
However, they mirror strong location preference in datasets, which we call
spatial bias. Therefore, generators render poor samples at unseen locations and
scales. We argue that the generators rely on their implicit positional encoding
to render spatial content. From our observations, the generator's implicit
positional encoding is translation-variant, making the generator spatially
biased. To address this issue, we propose injecting explicit positional
encoding at each scale of the generator. By learning the spatially unbiased
generator, we facilitate the robust use of generators in multiple tasks, such
as GAN inversion, multi-scale generation, generation of arbitrary sizes and
aspect ratios. Furthermore, we show that our method can also be applied to
denoising diffusion probabilistic models.

    

### [[2108.01289] AdvRush: Searching for Adversarially Robust Neural Architectures](http://arxiv.org/abs/2108.01289)


  Deep neural networks continue to awe the world with their remarkable
performance. Their predictions, however, are prone to be corrupted by
adversarial examples that are imperceptible to humans. Current efforts to
improve the robustness of neural networks against adversarial examples are
focused on developing robust training methods, which update the weights of a
neural network in a more robust direction. In this work, we take a step beyond
training of the weight parameters and consider the problem of designing an
adversarially robust neural architecture with high intrinsic robustness. We
propose AdvRush, a novel adversarial robustness-aware neural architecture
search algorithm, based upon a finding that independent of the training method,
the intrinsic robustness of a neural network can be represented with the
smoothness of its input loss landscape. Through a regularizer that favors a
candidate architecture with a smoother input loss landscape, AdvRush
successfully discovers an adversarially robust neural architecture. Along with
a comprehensive theoretical motivation for AdvRush, we conduct an extensive
amount of experiments to demonstrate the efficacy of AdvRush on various
benchmark datasets. Notably, on CIFAR-10, AdvRush achieves 55.91% robust
accuracy under FGSM attack after standard training and 50.04% robust accuracy
under AutoAttack after 7-step PGD adversarial training.

    

### [[2108.01295] MBDP: A Model-based Approach to Achieve both Robustness and Sample Efficiency via Double Dropout Planning](http://arxiv.org/abs/2108.01295)


  Model-based reinforcement learning is a widely accepted solution for solving
excessive sample demands. However, the predictions of the dynamics models are
often not accurate enough, and the resulting bias may incur catastrophic
decisions due to insufficient robustness. Therefore, it is highly desired to
investigate how to improve the robustness of model-based RL algorithms while
maintaining high sampling efficiency. In this paper, we propose Model-Based
Double-dropout Planning (MBDP) to balance robustness and efficiency. MBDP
consists of two kinds of dropout mechanisms, where the rollout-dropout aims to
improve the robustness with a small cost of sample efficiency, while the
model-dropout is designed to compensate for the lost efficiency at a slight
expense of robustness. By combining them in a complementary way, MBDP provides
a flexible control mechanism to meet different demands of robustness and
efficiency by tuning two corresponding dropout ratios. The effectiveness of
MBDP is demonstrated both theoretically and experimentally.

    

### [[2108.01301] Visualizing Data using GTSNE](http://arxiv.org/abs/2108.01301)


  We present a new method GTSNE to visualize high-dimensional data points in
the two dimensional map. The technique is a variation of t-SNE that produces
better visualizations by capturing both the local neighborhood structure and
the macro structure in the data. This is particularly important for
high-dimensional data that lie on continuous low-dimensional manifolds. We
illustrate the performance of GTSNE on a wide variety of datasets and compare
it the state of art methods, including t-SNE and UMAP. The visualizations
produced by GTSNE are better than those produced by the other techniques on
almost all of the datasets on the macro structure preservation.

    

### [[2108.01312] Learning Causal Relationships from Conditional Moment Conditions by Importance Weighting](http://arxiv.org/abs/2108.01312)


  We consider learning causal relationships under conditional moment
conditions. Unlike causal inference under unconditional moment conditions,
conditional moment conditions pose serious challenges for causal inference,
especially in complex, high-dimensional settings. To address this issue, we
propose a method that transforms conditional moment conditions to unconditional
moment conditions through importance weighting using the conditional density
ratio. Then, using this transformation, we propose a method that successfully
approximates conditional moment conditions. Our proposed approach allows us to
employ methods for estimating causal parameters from unconditional moment
conditions, such as generalized method of moments, adequately in a
straightforward manner. In experiments, we confirm that our proposed method
performs well compared to existing methods.

    

### [[2108.01314] Solving Fashion Recommendation -- The Farfetch Challenge](http://arxiv.org/abs/2108.01314)


  Recommendation engines are integral to the modern e-commerce experience, both
for the seller and the end user. Accurate recommendations lead to higher
revenue and better user experience. In this paper, we are presenting our
solution to ECML PKDD Farfetch Fashion Recommendation Challenge.The goal of
this challenge is to maximize the chances of a click when the users are
presented with set of fashion items. We have approached this problem as a
binary classification problem. Our winning solution utilizes Catboost as the
classifier and Bayesian Optimization for hyper parameter tuning. Our baseline
model achieved MRR of 0.5153 on the validation set. Bayesian optimization of
hyper parameters improved the MRR to 0.5240 on the validation set. Our final
submission on the test set achieved a MRR of 0.5257.

    

### [[2108.01316] RAIN: Reinforced Hybrid Attention Inference Network for Motion Forecasting](http://arxiv.org/abs/2108.01316)


  Motion forecasting plays a significant role in various domains (e.g.,
autonomous driving, human-robot interaction), which aims to predict future
motion sequences given a set of historical observations. However, the observed
elements may be of different levels of importance. Some information may be
irrelevant or even distracting to the forecasting in certain situations. To
address this issue, we propose a generic motion forecasting framework (named
RAIN) with dynamic key information selection and ranking based on a hybrid
attention mechanism. The general framework is instantiated to handle
multi-agent trajectory prediction and human motion forecasting tasks,
respectively. In the former task, the model learns to recognize the relations
between agents with a graph representation and to determine their relative
significance. In the latter task, the model learns to capture the temporal
proximity and dependency in long-term human motions. We also propose an
effective double-stage training pipeline with an alternating training strategy
to optimize the parameters in different modules of the framework. We validate
the framework on both synthetic simulations and motion forecasting benchmarks
in different domains, demonstrating that our method not only achieves
state-of-the-art forecasting performance, but also provides interpretable and
reasonable hybrid attention weights.

    

### [[2108.01317] Deep Reinforcement Learning Based Networked Control with Network Delays for Signal Temporal Logic Specifications](http://arxiv.org/abs/2108.01317)


  We present a novel deep reinforcement learning (DRL)-based design of a
networked controller with network delays for signal temporal logic (STL)
specifications. We consider the case in which both the system dynamics and
network delays are unknown. Because the satisfaction of an STL formula is based
not only on the current state but also on the behavior of the system, we
propose an extension of the Markov decision process (MDP), which is called a
$\tau\delta$-MDP, such that we can evaluate the satisfaction of the STL formula
under the network delays using the $\tau\delta$-MDP. Thereafter, we construct
deep neural networks based on the $\tau\delta$-MDP and propose a learning
algorithm. Through simulations, we also demonstrate the learning performance of
the proposed algorithm.

    

### [[2108.01335] Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability](http://arxiv.org/abs/2108.01335)


  Conventional saliency maps highlight input features to which neural network
predictions are highly sensitive. We take a different approach to saliency, in
which we identify and analyze the network parameters, rather than inputs, which
are responsible for erroneous decisions. We find that samples which cause
similar parameters to malfunction are semantically similar. We also show that
pruning the most salient parameters for a wrongly classified sample often
improves model behavior. Furthermore, fine-tuning a small number of the most
salient parameters on a single sample results in error correction on other
samples that are misclassified for similar reasons. Based on our parameter
saliency method, we also introduce an input-space saliency technique that
reveals how image features cause specific network components to malfunction.
Further, we rigorously validate the meaningfulness of our saliency maps on both
the dataset and case-study levels.

    

### [[2108.01358] Accelerating the Convergence of Human-in-the-Loop Reinforcement Learning with Counterfactual Explanations](http://arxiv.org/abs/2108.01358)


  The capability to interactively learn from human feedback would enable robots
in new social settings. For example, novice users could train service robots in
new tasks naturally and interactively. Human-in-the-loop Reinforcement Learning
(HRL) addresses this issue by combining human feedback and reinforcement
learning (RL) techniques. State-of-the-art interactive learning techniques
suffer from slow convergence, thus leading to a frustrating experience for the
human. This work approaches this problem by extending the existing TAMER
Framework with the possibility to enhance human feedback with two different
types of counterfactual explanations. We demonstrate our extensions' success in
improving the convergence, especially in the crucial early phases of the
training.

    

### [[2108.01368] Robust Compressed Sensing MRI with Deep Generative Priors](http://arxiv.org/abs/2108.01368)


  The CSGM framework (Bora-Jalal-Price-Dimakis'17) has shown that deep
generative priors can be powerful tools for solving inverse problems. However,
to date this framework has been empirically successful only on certain datasets
(for example, human faces and MNIST digits), and it is known to perform poorly
on out-of-distribution samples. In this paper, we present the first successful
application of the CSGM framework on clinical MRI data. We train a generative
prior on brain scans from the fastMRI dataset, and show that posterior sampling
via Langevin dynamics achieves high quality reconstructions. Furthermore, our
experiments and theory show that posterior sampling is robust to changes in the
ground-truth distribution and measurement process. Our code and models are
available at: \url{this https URL}.

    

### [[2108.01375] Classifying action correctness in physical rehabilitation exercises](http://arxiv.org/abs/2108.01375)


  The work in this paper focuses on the role of machine learning in assessing
the correctness of a human motion or action. This task proves to be more
challenging than the gesture and action recognition ones. We will demonstrate,
through a set of experiments on a recent dataset, that machine learning
algorithms can produce good results for certain actions, but can also fall into
the trap of classifying an incorrect execution of an action as a correct
execution of another action.

    

### [[2108.01393] Electrical peak demand forecasting- A review](http://arxiv.org/abs/2108.01393)


  The power system is undergoing rapid evolution with the roll-out of advanced
metering infrastructure and local energy applications (e.g. electric vehicles)
as well as the increasing penetration of intermittent renewable energy at both
transmission and distribution level, which characterizes the peak load demand
with stronger randomness and less predictability and therefore poses a threat
to the power grid security. Since storing large quantities of electricity to
satisfy load demand is neither economically nor environmentally friendly,
effective peak demand management strategies and reliable peak load forecast
methods become essential for optimizing the power system operations. To this
end, this paper provides a timely and comprehensive overview of peak load
demand forecast methods in the literature. To our best knowledge, this is the
first comprehensive review on such topic. In this paper we first give a precise
and unified problem definition of peak load demand forecast. Second, 139 papers
on peak load forecast methods were systematically reviewed where methods were
classified into different stages based on the timeline. Thirdly, a comparative
analysis of peak load forecast methods are summarized and different optimizing
methods to improve the forecast performance are discussed. The paper ends with
a comprehensive summary of the reviewed papers and a discussion of potential
future research directions.

    

### [[2108.01407] GalaxAI: Machine learning toolbox for interpretable analysis of spacecraft telemetry data](http://arxiv.org/abs/2108.01407)


  We present GalaxAI - a versatile machine learning toolbox for efficient and
interpretable end-to-end analysis of spacecraft telemetry data. GalaxAI employs
various machine learning algorithms for multivariate time series analyses,
classification, regression and structured output prediction, capable of
handling high-throughput heterogeneous data. These methods allow for the
construction of robust and accurate predictive models, that are in turn applied
to different tasks of spacecraft monitoring and operations planning. More
importantly, besides the accurate building of models, GalaxAI implements a
visualisation layer, providing mission specialists and operators with a full,
detailed and interpretable view of the data analysis process. We show the
utility and versatility of GalaxAI on two use-cases concerning two different
spacecraft: i) analysis and planning of Mars Express thermal power consumption
and ii) predicting of INTEGRAL's crossings through Van Allen belts.

    

### [[2108.01431] Noise-Resistant Deep Metric Learning with Probabilistic Instance Filtering](http://arxiv.org/abs/2108.01431)


  Noisy labels are commonly found in real-world data, which cause performance
degradation of deep neural networks. Cleaning data manually is labour-intensive
and time-consuming. Previous research mostly focuses on enhancing
classification models against noisy labels, while the robustness of deep metric
learning (DML) against noisy labels remains less well-explored. In this paper,
we bridge this important gap by proposing Probabilistic Ranking-based Instance
Selection with Memory (PRISM) approach for DML. PRISM calculates the
probability of a label being clean, and filters out potentially noisy samples.
Specifically, we propose three methods to calculate this probability: 1)
Average Similarity Method (AvgSim), which calculates the average similarity
between potentially noisy data and clean data; 2) Proxy Similarity Method
(ProxySim), which replaces the centers maintained by AvgSim with the proxies
trained by proxy-based method; and 3) von Mises-Fisher Distribution Similarity
(vMF-Sim), which estimates a von Mises-Fisher distribution for each data class.
With such a design, the proposed approach can deal with challenging DML
situations in which the majority of the samples are noisy. Extensive
experiments on both synthetic and real-world noisy dataset show that the
proposed approach achieves up to 8.37% higher Precision@1 compared with the
best performing state-of-the-art baseline approaches, within reasonable
training time.

    

### [[2108.01440] Adaptively Optimize Content Recommendation Using Multi Armed Bandit Algorithms in E-commerce](http://arxiv.org/abs/2108.01440)


  E-commerce sites strive to provide users the most timely relevant information
in order to reduce shopping frictions and increase customer satisfaction. Multi
armed bandit models (MAB) as a type of adaptive optimization algorithms provide
possible approaches for such purposes. In this paper, we analyze using three
classic MAB algorithms, epsilon-greedy, Thompson sampling (TS), and upper
confidence bound 1 (UCB1) for dynamic content recommendations, and walk through
the process of developing these algorithms internally to solve a real world
e-commerce use case. First, we analyze the three MAB algorithms using simulated
purchasing datasets with non-stationary reward distributions to simulate the
possible time-varying customer preferences, where the traffic allocation
dynamics and the accumulative rewards of different algorithms are studied.
Second, we compare the accumulative rewards of the three MAB algorithms with
more than 1,000 trials using actual historical A/B test datasets. We find that
the larger difference between the success rates of competing recommendations
the more accumulative rewards the MAB algorithms can achieve. In addition, we
find that TS shows the highest average accumulative rewards under different
testing scenarios. Third, we develop a batch-updated MAB algorithm to overcome
the delayed reward issue in e-commerce and enable an online content
optimization on our App homepage. For a state-of-the-art comparison, a real A/B
test among our batch-updated MAB algorithm, a third-party MAB solution, and the
default business logic are conducted. The result shows that our batch-updated
MAB algorithm outperforms the counterparts and achieves 6.13% relative
click-through rate (CTR) increase and 16.1% relative conversion rate (CVR)
increase compared to the default experience, and 2.9% relative CTR increase and
1.4% relative CVR increase compared to the external MAB service.

    

### [[2108.01441] Using Query Expansion in Manifold Ranking for Query-Oriented Multi-Document Summarization](http://arxiv.org/abs/2108.01441)


  Manifold ranking has been successfully applied in query-oriented
multi-document summarization. It not only makes use of the relationships among
the sentences, but also the relationships between the given query and the
sentences. However, the information of original query is often insufficient. So
we present a query expansion method, which is combined in the manifold ranking
to resolve this problem. Our method not only utilizes the information of the
query term itself and the knowledge base WordNet to expand it by synonyms, but
also uses the information of the document set itself to expand the query in
various ways (mean expansion, variance expansion and TextRank expansion).
Compared with the previous query expansion methods, our method combines
multiple query expansion methods to better represent query information, and at
the same time, it makes a useful attempt on manifold ranking. In addition, we
use the degree of word overlap and the proximity between words to calculate the
similarity between sentences. We performed experiments on the datasets of DUC
2006 and DUC2007, and the evaluation results show that the proposed query
expansion method can significantly improve the system performance and make our
system comparable to the state-of-the-art systems.

    

### [[2108.01442] Sequence Adaptation via Reinforcement Learning in Recommender Systems](http://arxiv.org/abs/2108.01442)


  Accounting for the fact that users have different sequential patterns, the
main drawback of state-of-the-art recommendation strategies is that a fixed
sequence length of user-item interactions is required as input to train the
models. This might limit the recommendation accuracy, as in practice users
follow different trends on the sequential recommendations. Hence, baseline
strategies might ignore important sequential interactions or add noise to the
models with redundant interactions, depending on the variety of users'
sequential behaviours. To overcome this problem, in this study we propose the
SAR model, which not only learns the sequential patterns but also adjusts the
sequence length of user-item interactions in a personalized manner. We first
design an actor-critic framework, where the RL agent tries to compute the
optimal sequence length as an action, given the user's state representation at
a certain time step. In addition, we optimize a joint loss function to align
the accuracy of the sequential recommendations with the expected cumulative
rewards of the critic network, while at the same time we adapt the sequence
length with the actor network in a personalized manner. Our experimental
evaluation on four real-world datasets demonstrates the superiority of our
proposed model over several baseline approaches. Finally, we make our
implementation publicly available at this https URL.

    

### [[2108.01450] Is Disentanglement enough? On Latent Representations for Controllable Music Generation](http://arxiv.org/abs/2108.01450)


  Improving controllability or the ability to manipulate one or more attributes
of the generated data has become a topic of interest in the context of deep
generative models of music. Recent attempts in this direction have relied on
learning disentangled representations from data such that the underlying
factors of variation are well separated. In this paper, we focus on the
relationship between disentanglement and controllability by conducting a
systematic study using different supervised disentanglement learning algorithms
based on the Variational Auto-Encoder (VAE) architecture. Our experiments show
that a high degree of disentanglement can be achieved by using different forms
of supervision to train a strong discriminative encoder. However, in the
absence of a strong generative decoder, disentanglement does not necessarily
imply controllability. The structure of the latent space with respect to the
VAE-decoder plays an important role in boosting the ability of a generative
model to manipulate different attributes. To this end, we also propose methods
and metrics to help evaluate the quality of a latent space with respect to the
afforded degree of controllability.

    

### [[2108.01455] FEBR: Expert-Based Recommendation Framework for beneficial and personalized content](http://arxiv.org/abs/2108.01455)


  So far, most research on recommender systems focused on maintaining long-term
user engagement and satisfaction, by promoting relevant and personalized
content. However, it is still very challenging to evaluate the quality and the
reliability of this content. In this paper, we propose FEBR (Expert-Based
Recommendation Framework), an apprenticeship learning framework to assess the
quality of the recommended content on online platforms. The framework exploits
the demonstrated trajectories of an expert (assumed to be reliable) in a
recommendation evaluation environment, to recover an unknown utility function.
This function is used to learn an optimal policy describing the expert's
behavior, which is then used in the framework to provide high-quality and
personalized recommendations. We evaluate the performance of our solution
through a user interest simulation environment (using RecSim). We simulate
interactions under the aforementioned expert policy for videos recommendation,
and compare its efficiency with standard recommendation methods. The results
show that our approach provides a significant gain in terms of content quality,
evaluated by experts and watched by users, while maintaining almost the same
watch time as the baseline approaches.

    

### [[2108.01466] Risk Adversarial Learning System for Connected and Autonomous Vehicle Charging](http://arxiv.org/abs/2108.01466)


  In this paper, the design of a rational decision support system (RDSS) for a
connected and autonomous vehicle charging infrastructure (CAV-CI) is studied.
In the considered CAV-CI, the distribution system operator (DSO) deploys
electric vehicle supply equipment (EVSE) to provide an EV charging facility for
human-driven connected vehicles (CVs) and autonomous vehicles (AVs). The
charging request by the human-driven EV becomes irrational when it demands more
energy and charging period than its actual need. Therefore, the scheduling
policy of each EVSE must be adaptively accumulated the irrational charging
request to satisfy the charging demand of both CVs and AVs. To tackle this, we
formulate an RDSS problem for the DSO, where the objective is to maximize the
charging capacity utilization by satisfying the laxity risk of the DSO. Thus,
we devise a rational reward maximization problem to adapt the irrational
behavior by CVs in a data-informed manner. We propose a novel risk adversarial
multi-agent learning system (RAMALS) for CAV-CI to solve the formulated RDSS
problem. In RAMALS, the DSO acts as a centralized risk adversarial agent (RAA)
for informing the laxity risk to each EVSE. Subsequently, each EVSE plays the
role of a self-learner agent to adaptively schedule its own EV sessions by
coping advice from RAA. Experiment results show that the proposed RAMALS
affords around 46.6% improvement in charging rate, about 28.6% improvement in
the EVSE's active charging time and at least 33.3% more energy utilization, as
compared to a currently deployed ACN EVSE system, and other baselines.

    

### [[2108.01468] Quantum Neural Networks: Concepts, Applications, and Challenges](http://arxiv.org/abs/2108.01468)


  Quantum deep learning is a research field for the use of quantum computing
techniques for training deep neural networks. The research topics and
directions of deep learning and quantum computing have been separated for long
time, however by discovering that quantum circuits can act like artificial
neural networks, quantum deep learning research is widely adopted. This paper
explains the backgrounds and basic principles of quantum deep learning and also
introduces major achievements. After that, this paper discusses the challenges
of quantum deep learning research in multiple perspectives. Lastly, this paper
presents various future research directions and application fields of quantum
deep learning.

    

### [[2108.01469] Creation and Detection of German Voice Deepfakes](http://arxiv.org/abs/2108.01469)


  Synthesizing voice with the help of machine learning techniques has made
rapid progress over the last years [1] and first high profile fraud cases have
been recently reported [2]. Given the current increase in using conferencing
tools for online teaching, we question just how easy (i.e. needed data,
hardware, skill set) it would be to create a convincing voice fake. We analyse
how much training data a participant (e.g. a student) would actually need to
fake another participants voice (e.g. a professor). We provide an analysis of
the existing state of the art in creating voice deep fakes, as well as offer
detailed technical guidance and evidence of just how much effort is needed to
copy a voice. A user study with more than 100 participants shows how difficult
it is to identify real and fake voice (on avg. only 37 percent can distinguish
between real and fake voice of a professor). With a focus on German language
and an online teaching environment we discuss the societal implications as well
as demonstrate how to use machine learning techniques to possibly detect such
fakes.

    

### [[2108.01473] A Hinge-Loss based Codebook Transfer for Cross-Domain Recommendation with Nonoverlapping Data](http://arxiv.org/abs/2108.01473)


  Recommender systems(RS), especially collaborative filtering(CF) based RS, has
been playing an important role in many e-commerce applications. As the
information being searched over the internet is rapidly increasing, users often
face the difficulty of finding items of his/her own interest and RS often
provides help in such tasks. Recent studies show that, as the item space
increases, and the number of items rated by the users become very less, issues
like sparsity arise. To mitigate the sparsity problem, transfer learning
techniques are being used wherein the data from dense domain(source) is
considered in order to predict the missing entries in the sparse
domain(target). In this paper, we propose a transfer learning approach for
cross-domain recommendation when both domains have no overlap of users and
items. In our approach the transferring of knowledge from source to target
domain is done in a novel way. We make use of co-clustering technique to obtain
the codebook (cluster-level rating pattern) of source domain. By making use of
hinge loss function we transfer the learnt codebook of the source domain to
target. The use of hinge loss as a loss function is novel and has not been
tried before in transfer learning. We demonstrate that our technique improves
the approximation of the target matrix on benchmark datasets.

    

### [[2108.01485] Fast Estimation Method for the Stability of Ensemble Feature Selectors](http://arxiv.org/abs/2108.01485)


  It is preferred that feature selectors be \textit{stable} for better
interpretabity and robust prediction. Ensembling is known to be effective for
improving the stability of feature selectors. Since ensembling is
time-consuming, it is desirable to reduce the computational cost to estimate
the stability of the ensemble feature selectors. We propose a simulator of a
feature selector, and apply it to a fast estimation of the stability of
ensemble feature selectors. To the best of our knowledge, this is the first
study that estimates the stability of ensemble feature selectors and reduces
the computation time theoretically and empirically.

    

### [[2108.01512] Task Agnostic Metrics for Reservoir Computing](http://arxiv.org/abs/2108.01512)


  Physical reservoir computing is a computational paradigm that enables
temporal pattern recognition to be performed directly in physical matter. By
exciting non-linear dynamical systems and linearly classifying their changes in
state, we can create highly energy-efficient devices capable of solving machine
learning tasks without the need to build a modular system consisting of
millions of neurons interconnected by synapses. The chosen dynamical system
must have three desirable properties: non-linearity, complexity, and fading
memory to act as an effective reservoir. We present task agnostic quantitative
measures for each of these three requirements and exemplify them for two
reservoirs: an echo state network and a simulated magnetic skyrmion-based
reservoir. We show that, in general, systems with lower damping reach higher
values in all three performance metrics. Whilst for input signal strength,
there is a natural trade-off between memory capacity and non-linearity of the
reservoir's behaviour. In contrast to typical task-dependent reservoir
computing benchmarks, these metrics can be evaluated in parallel from a single
input signal, drastically speeding up the parameter search to design efficient
and high-performance reservoirs.

    

### [[2108.01513] SphereFace2: Binary Classification is All You Need for Deep Face Recognition](http://arxiv.org/abs/2108.01513)


  State-of-the-art deep face recognition methods are mostly trained with a
softmax-based multi-class classification framework. Despite being popular and
effective, these methods still have a few shortcomings that limit empirical
performance. In this paper, we first identify the discrepancy between training
and evaluation in the existing multi-class classification framework and then
discuss the potential limitations caused by the "competitive" nature of softmax
normalization. Motivated by these limitations, we propose a novel binary
classification training framework, termed SphereFace2. In contrast to existing
methods, SphereFace2 circumvents the softmax normalization, as well as the
corresponding closed-set assumption. This effectively bridges the gap between
training and evaluation, enabling the representations to be improved
individually by each binary classification task. Besides designing a specific
well-performing loss function, we summarize a few general principles for this
"one-vs-all" binary classification framework so that it can outperform current
competitive methods. We conduct comprehensive experiments on popular benchmarks
to demonstrate that SphereFace2 can consistently outperform current
state-of-the-art deep face recognition methods.

    

### [[2108.01518] Non-local Graph Convolutional Network for joint Activity Recognition and Motion Prediction](http://arxiv.org/abs/2108.01518)


  3D skeleton-based motion prediction and activity recognition are two
interwoven tasks in human behaviour analysis. In this work, we propose a motion
context modeling methodology that provides a new way to combine the advantages
of both graph convolutional neural networks and recurrent neural networks for
joint human motion prediction and activity recognition. Our approach is based
on using an LSTM encoder-decoder and a non-local feature extraction attention
mechanism to model the spatial correlation of human skeleton data and temporal
correlation among motion frames. The proposed network can easily include two
output branches, one for Activity Recognition and one for Future Motion
Prediction, which can be jointly trained for enhanced performance. Experimental
results on Human 3.6M, CMU Mocap and NTU RGB-D datasets show that our proposed
approach provides the best prediction capability among baseline LSTM-based
methods, while achieving comparable performance to other state-of-the-art
methods.

    

### [[2108.01527] Double-Dot Network for Antipodal Grasp Detection](http://arxiv.org/abs/2108.01527)


  This paper proposes a new deep learning approach to antipodal grasp
detection, named Double-Dot Network (DD-Net). It follows the recent anchor-free
object detection framework, which does not depend on empirically pre-set
anchors and thus allows more generalized and flexible prediction on unseen
objects. Specifically, unlike the widely used 5-dimensional rectangle, the
gripper configuration is defined as a pair of fingertips. An effective CNN
architecture is introduced to localize such fingertips, and with the help of
auxiliary centers for refinement, it accurately and robustly infers grasp
candidates. Additionally, we design a specialized loss function to measure the
quality of grasps, and in contrast to the IoU scores of bounding boxes adopted
in object detection, it is more consistent to the grasp detection task. Both
the simulation and robotic experiments are executed and state of the art
accuracies are achieved, showing that DD-Net is superior to the counterparts in
handling unseen objects.

    

### [[2108.01529] Neural Calibration for Scalable Beamforming in FDD Massive MIMO with Implicit Channel Estimation](http://arxiv.org/abs/2108.01529)


  Channel estimation and beamforming play critical roles in frequency-division
duplexing (FDD) massive multiple-input multiple-output (MIMO) systems. However,
these two modules have been treated as two stand-alone components, which makes
it difficult to achieve a global system optimality. In this paper, we propose a
deep learning-based approach that directly optimizes the beamformers at the
base station according to the received uplink pilots, thereby, bypassing the
explicit channel estimation. Different from the existing fully data-driven
approach where all the modules are replaced by deep neural networks (DNNs), a
neural calibration method is proposed to improve the scalability of the
end-to-end design. In particular, the backbone of conventional time-efficient
algorithms, i.e., the least-squares (LS) channel estimator and the zero-forcing
(ZF) beamformer, is preserved and DNNs are leveraged to calibrate their inputs
for better performance. The permutation equivariance property of the formulated
resource allocation problem is then identified to design a low-complexity
neural network architecture. Simulation results will show the superiority of
the proposed neural calibration method over benchmark schemes in terms of both
the spectral efficiency and scalability in large-scale wireless networks.

    

### [[2108.01538] Geometry of Linear Convolutional Networks](http://arxiv.org/abs/2108.01538)


  We study the family of functions that are represented by a linear
convolutional neural network (LCN). These functions form a semi-algebraic
subset of the set of linear maps from input space to output space. In contrast,
the families of functions represented by fully-connected linear networks form
algebraic sets. We observe that the functions represented by LCNs can be
identified with polynomials that admit certain factorizations, and we use this
perspective to describe the impact of the network's architecture on the
geometry of the resulting function space. We further study the optimization of
an objective function over an LCN, analyzing critical points in function space
and in parameter space, and describing dynamical invariants for gradient
descent. Overall, our theory predicts that the optimized parameters of an LCN
will often correspond to repeated filters across layers, or filters that can be
decomposed as repeated filters. We also conduct numerical and symbolic
experiments that illustrate our results and present an in-depth analysis of the
landscape for small architectures.

    

### [[2108.01548] Inference via Sparse Coding in a Hierarchical Vision Model](http://arxiv.org/abs/2108.01548)


  Sparse coding has been incorporated in models of the visual cortex for its
computational advantages and connection to biology. But how the level of
sparsity contributes to performance on visual tasks is not well understood. In
this work, sparse coding has been integrated into an existing hierarchical V2
model (Hosoya and Hyvärinen, 2015), but replacing the Independent Component
Analysis (ICA) with an explicit sparse coding in which the degree of sparsity
can be controlled. After training, the sparse coding basis functions with a
higher degree of sparsity resembled qualitatively different structures, such as
curves and corners. The contributions of the models were assessed with image
classification tasks, including object classification, and tasks associated
with mid-level vision including figure-ground classification, texture
classification, and angle prediction between two line stimuli. In addition, the
models were assessed in comparison to a texture sensitivity measure that has
been reported in V2 (Freeman et al., 2013), and a deleted-region inference
task. The results from the experiments show that while sparse coding performed
worse than ICA at classifying images, only sparse coding was able to better
match the texture sensitivity level of V2 and infer deleted image regions, both
by increasing the degree of sparsity in sparse coding. Higher degrees of
sparsity allowed for inference over larger deleted image regions. The mechanism
that allows for this inference capability in sparse coding is described here.

    

### [[2108.01584] Numerical Solution of Stiff Ordinary Differential Equations with Random Projection Neural Networks](http://arxiv.org/abs/2108.01584)


  We propose a numerical scheme based on Random Projection Neural Networks
(RPNN) for the solution of Ordinary Differential Equations (ODEs) with a focus
on stiff problems. In particular, we use an Extreme Learning Machine, a
single-hidden layer Feedforward Neural Network with Radial Basis Functions
which widths are uniformly distributed random variables, while the values of
the weights between the input and the hidden layer are set equal to one. The
numerical solution is obtained by constructing a system of nonlinear algebraic
equations, which is solved with respect to the output weights using the
Gauss-Newton method. For our illustrations, we apply the proposed machine
learning approach to solve two benchmark stiff problems, namely the Rober and
the van der Pol ones (the latter with large values of the stiffness parameter),
and we perform a comparison with well-established methods such as the adaptive
Runge-Kutta method based on the Dormand-Prince pair, and a variable-step
variable-order multistep solver based on numerical differentiation formulas, as
implemented in the \texttt{ode45} and \texttt{ode15s} MATLAB functions,
respectively. We show that our proposed scheme yields good numerical
approximation accuracy without being affected by the stiffness, thus
outperforming in same cases the \texttt{ode45} and \texttt{ode15s} functions.
Importantly, upon training using a fixed number of collocation points, the
proposed scheme approximates the solution in the whole domain in contrast to
the classical time integration methods.

    

### [[2108.01621] Domain Generalization via Gradient Surgery](http://arxiv.org/abs/2108.01621)


  In real-life applications, machine learning models often face scenarios where
there is a change in data distribution between training and test domains. When
the aim is to make predictions on distributions different from those seen at
training, we incur in a domain generalization problem. Methods to address this
issue learn a model using data from multiple source domains, and then apply
this model to the unseen target domain. Our hypothesis is that when training
with multiple domains, conflicting gradients within each mini-batch contain
information specific to the individual domains which is irrelevant to the
others, including the test domain. If left untouched, such disagreement may
degrade generalization performance. In this work, we characterize the
conflicting gradients emerging in domain shift scenarios and devise novel
gradient agreement strategies based on gradient surgery to alleviate their
effect. We validate our approach in image classification tasks with three
multi-domain datasets, showing the value of the proposed agreement strategy in
enhancing the generalization capability of deep learning models in domain shift
scenarios.

    

### [[2108.01624] Large-Scale Differentially Private BERT](http://arxiv.org/abs/2108.01624)


  In this work, we study the large-scale pretraining of BERT-Large with
differentially private SGD (DP-SGD). We show that combined with a careful
implementation, scaling up the batch size to millions (i.e., mega-batches)
improves the utility of the DP-SGD step for BERT; we also enhance its
efficiency by using an increasing batch size schedule. Our implementation
builds on the recent work of [SVK20], who demonstrated that the overhead of a
DP-SGD step is minimized with effective use of JAX [BFH+18, FJL18] primitives
in conjunction with the XLA compiler [XLA17]. Our implementation achieves a
masked language model accuracy of 60.5% at a batch size of 2M, for $\epsilon =
5.36$. To put this number in perspective, non-private BERT models achieve an
accuracy of $\sim$70%.

    

### [[2108.01625] From augmented microscopy to the topological transformer: a new approach in cell image analysis for Alzheimer's research](http://arxiv.org/abs/2108.01625)


  Cell image analysis is crucial in Alzheimer's research to detect the presence
of A$\beta$ protein inhibiting cell function. Deep learning speeds up the
process by making only low-level data sufficient for fruitful inspection. We
first found Unet is most suitable in augmented microscopy by comparing
performance in multi-class semantics segmentation. We develop the augmented
microscopy method to capture nuclei in a brightfield image and the transformer
using Unet model to convert an input image into a sequence of topological
information. The performance regarding Intersection-over-Union is consistent
concerning the choice of image preprocessing and ground-truth generation.
Training model with data of a specific cell type demonstrates transfer learning
applies to some extent.
The topological transformer aims to extract persistence silhouettes or
landscape signatures containing geometric information of a given image of
cells. This feature extraction facilitates studying an image as a collection of
one-dimensional data, substantially reducing computational costs. Using the
transformer, we attempt grouping cell images by their cell type relying solely
on topological features. Performances of the transformers followed by SVM,
XGBoost, LGBM, and simple convolutional neural network classifiers are inferior
to the conventional image classification. However, since this research
initiates a new perspective in biomedical research by combining deep learning
and topology for image analysis, we speculate follow-up investigation will
reinforce our genuine regime.

    

### [[2108.01640] Automatic classification of eclipsing binary stars using deep learning methods](http://arxiv.org/abs/2108.01640)


  In the last couple of decades, tremendous progress has been achieved in
developing robotic telescopes and, as a result, sky surveys (both terrestrial
and space) have become the source of a substantial amount of new observational
data. These data contain a lot of information about binary stars, hidden in
their light curves. With the huge amount of astronomical data gathered, it is
not reasonable to expect all the data to be manually processed and analyzed.
Therefore, in this paper, we focus on the automatic classification of eclipsing
binary stars using deep learning methods. Our classifier provides a tool for
the categorization of light curves of binary stars into two classes: detached
and over-contact. We used the ELISa software to obtain synthetic data, which we
then used for the training of the classifier. For evaluation purposes, we
collected 100 light curves of observed binary stars, in order to evaluate a
number of classifiers. We evaluated semi-detached eclipsing binary stars as
detached. The best-performing classifier combines bidirectional Long Short-Term
Memory (LSTM) and a one-dimensional convolutional neural network, which
achieved 98% accuracy on the evaluation set. Omitting semi-detached eclipsing
binary stars, we could obtain 100% accuracy in classification.

    

### [[2108.01644] The Devil is in the GAN: Defending Deep Generative Models Against Backdoor Attacks](http://arxiv.org/abs/2108.01644)


  Deep Generative Models (DGMs) allow users to synthesize data from complex,
high-dimensional manifolds. Industry applications of DGMs include data
augmentation to boost performance of (semi-)supervised machine learning, or to
mitigate fairness or privacy concerns. Large-scale DGMs are notoriously hard to
train, requiring expert skills, large amounts of data and extensive
computational resources. Thus, it can be expected that many enterprises will
resort to sourcing pre-trained DGMs from potentially unverified third parties,
e.g.~open source model repositories.
As we show in this paper, such a deployment scenario poses a new attack
surface, which allows adversaries to potentially undermine the integrity of
entire machine learning development pipelines in a victim organization.
Specifically, we describe novel training-time attacks resulting in corrupted
DGMs that synthesize regular data under normal operations and designated target
outputs for inputs sampled from a trigger distribution. Depending on the
control that the adversary has over the random number generation, this imposes
various degrees of risk that harmful data may enter the machine learning
development pipelines, potentially causing material or reputational damage to
the victim organization.
Our attacks are based on adversarial loss functions that combine the dual
objectives of attack stealth and fidelity. We show its effectiveness for a
variety of DGM architectures (Generative Adversarial Networks (GANs),
Variational Autoencoders (VAEs)) and data domains (images, audio). Our
experiments show that - even for large-scale industry-grade DGMs - our attack
can be mounted with only modest computational efforts. We also investigate the
effectiveness of different defensive approaches (based on static/dynamic model
and output inspections) and prescribe a practical defense strategy that paves
the way for safe usage of DGMs.

    

### [[2108.01660] Spectral Graph Convolutional Networks WithLifting-based Adaptive Graph Wavelets](http://arxiv.org/abs/2108.01660)


  Spectral graph convolutional networks (SGCNs) have been attracting increasing
attention in graph representation learning partly due to their interpretability
through the prism of the established graph signal processing framework.
However, existing SGCNs are limited in implementing graph convolutions with
rigid transforms that could not adapt to signals residing on graphs and tasks
at hand. In this paper, we propose a novel class of spectral graph
convolutional networks that implement graph convolutions with adaptive graph
wavelets. Specifically, the adaptive graph wavelets are learned with neural
network-parameterized lifting structures, where structure-aware attention-based
lifting operations are developed to jointly consider graph structures and node
features. We propose to lift based on diffusion wavelets to alleviate the
structural information loss induced by partitioning non-bipartite graphs. By
design, the locality and sparsity of the resulting wavelet transform as well as
the scalability of the lifting structure for large and varying-size graphs are
guaranteed. We further derive a soft-thresholding filtering operation by
learning sparse graph representations in terms of the learned wavelets, which
improves the scalability and interpretablity, and yield a localized, efficient
and scalable spectral graph convolution. To ensure that the learned graph
representations are invariant to node permutations, a layer is employed at the
input of the networks to reorder the nodes according to their local topology
information. We evaluate the proposed networks in both node-level and
graph-level representation learning tasks on benchmark citation and
bioinformatics graph datasets. Extensive experiments demonstrate the
superiority of the proposed networks over existing SGCNs in terms of accuracy,
efficiency and scalability.

    

### [[2108.01661] Grounding Representation Similarity with Statistical Testing](http://arxiv.org/abs/2108.01661)


  To understand neural network behavior, recent works quantitatively compare
different networks' learned representations using canonical correlation
analysis (CCA), centered kernel alignment (CKA), and other dissimilarity
measures. Unfortunately, these widely used measures often disagree on
fundamental observations, such as whether deep networks differing only in
random initialization learn similar representations. These disagreements raise
the question: which, if any, of these dissimilarity measures should we believe?
We provide a framework to ground this question through a concrete test:
measures should have sensitivity to changes that affect functional behavior,
and specificity against changes that do not. We quantify this through a variety
of functional behaviors including probing accuracy and robustness to
distribution shift, and examine changes such as varying random initialization
and deleting principal components. We find that current metrics exhibit
different weaknesses, note that a classical baseline performs surprisingly
well, and highlight settings where all metrics appear to fail, thus providing a
challenge set for further improvement.

    

### [[2108.01662] Uniform Sampling over Episode Difficulty](http://arxiv.org/abs/2108.01662)


  Episodic training is a core ingredient of few-shot learning to train models
on tasks with limited labelled data. Despite its success, episodic training
remains largely understudied, prompting us to ask the question: what is the
best way to sample episodes? In this paper, we first propose a method to
approximate episode sampling distributions based on their difficulty. Building
on this method, we perform an extensive analysis and find that sampling
uniformly over episode difficulty outperforms other sampling schemes, including
curriculum and easy-/hard-mining. As the proposed sampling method is algorithm
agnostic, we can leverage these insights to improve few-shot learning
accuracies across many episodic training algorithms. We demonstrate the
efficacy of our method across popular few-shot learning datasets, algorithms,
network architectures, and protocols.

    

### [[1906.04870] Communication-Efficient Accurate Statistical Estimation](http://arxiv.org/abs/1906.04870)


  When the data are stored in a distributed manner, direct application of
traditional statistical inference procedures is often prohibitive due to
communication cost and privacy concerns. This paper develops and investigates
two Communication-Efficient Accurate Statistical Estimators (CEASE),
implemented through iterative algorithms for distributed optimization. In each
iteration, node machines carry out computation in parallel and communicate with
the central processor, which then broadcasts aggregated information to node
machines for new updates. The algorithms adapt to the similarity among loss
functions on node machines, and converge rapidly when each node machine has
large enough sample size. Moreover, they do not require good initialization and
enjoy linear converge guarantees under general conditions. The contraction rate
of optimization errors is presented explicitly, with dependence on the local
sample size unveiled. In addition, the improved statistical accuracy per
iteration is derived. By regarding the proposed method as a multi-step
statistical estimator, we show that statistical efficiency can be achieved in
finite steps in typical statistical applications. In addition, we give the
conditions under which the one-step CEASE estimator is statistically efficient.
Extensive numerical experiments on both synthetic and real data validate the
theoretical results and demonstrate the superior performance of our algorithms.

    

### [[1908.03097] Variational Bayes on Manifolds](http://arxiv.org/abs/1908.03097)


  Variational Bayes (VB) has become a widely-used tool for Bayesian inference
in statistics and machine learning. Nonetheless, the development of the
existing VB algorithms is so far generally restricted to the case where the
variational parameter space is Euclidean, which hinders the potential broad
application of VB methods. This paper extends the scope of VB to the case where
the variational parameter space is a Riemannian manifold. We develop an
efficient manifold-based VB algorithm that exploits both the geometric
structure of the constraint parameter space and the information geometry of the
manifold of VB approximating probability distributions. Our algorithm is
provably convergent and achieves a convergence rate of order $\mathcal
O(1/\sqrt{T})$ and $\mathcal O(1/T^{2-2\epsilon})$ for a non-convex evidence
lower bound function and a strongly retraction-convex evidence lower bound
function, respectively. We develop in particular two manifold VB algorithms,
Manifold Gaussian VB and Manifold Neural Net VB, and demonstrate through
numerical experiments that the proposed algorithms are stable, less sensitive
to initialization and compares favourably to existing VB methods.

    

### [[2002.05079] Efficient Structure-preserving Support Tensor Train Machine](http://arxiv.org/abs/2002.05079)


  An increasing amount of collected data are high-dimensional multi-way arrays
(tensors), and it is crucial for efficient learning algorithms to exploit this
tensorial structure as much as possible. The ever-present curse of
dimensionality for high dimensional data and the loss of structure when
vectorizing the data motivates the use of tailored low-rank tensor
classification methods. In the presence of small amounts of training data,
kernel methods offer an attractive choice as they provide the possibility for a
nonlinear decision boundary. We develop the Tensor Train Multi-way Multi-level
Kernel (TT-MMK), which combines the simplicity of the Canonical Polyadic
decomposition, the classification power of the Dual Structure-preserving
Support Vector Machine, and the reliability of the Tensor Train (TT)
approximation. We show by experiments that the TT-MMK method is usually more
reliable computationally, less sensitive to tuning parameters, and gives higher
prediction accuracy in the SVM classification when benchmarked against other
state-of-the-art techniques.

    

### [[2002.08536] Debiased Off-Policy Evaluation for Recommendation Systems](http://arxiv.org/abs/2002.08536)


  Efficient methods to evaluate new algorithms are critical for improving
interactive bandit and reinforcement learning systems such as recommendation
systems. A/B tests are reliable, but are time- and money-consuming, and entail
a risk of failure. In this paper, we develop an alternative method, which
predicts the performance of algorithms given historical data that may have been
generated by a different algorithm. Our estimator has the property that its
prediction converges in probability to the true performance of a counterfactual
algorithm at a rate of $\sqrt{N}$, as the sample size $N$ increases. We also
show a correct way to estimate the variance of our prediction, thus allowing
the analyst to quantify the uncertainty in the prediction. These properties
hold even when the analyst does not know which among a large number of
potentially important state variables are actually important. We validate our
method by a simulation experiment about reinforcement learning. We finally
apply it to improve advertisement design by a major advertisement company. We
find that our method produces smaller mean squared errors than state-of-the-art
methods.

    

### [[2004.05768] Sequential Weakly Labeled Multi-Activity Localization and Recognition on Wearable Sensors using Recurrent Attention Networks](http://arxiv.org/abs/2004.05768)


  With the popularity and development of the wearable devices such as
smartphones, human activity recognition (HAR) based on sensors has become as a
key research area in human computer interaction and ubiquitous computing. The
emergence of deep learning leads to a recent shift in the research of HAR,
which requires massive strictly labeled data. In comparison with video data,
activity data recorded from accelerometer or gyroscope is often more difficult
to interpret and segment. Recently, several attention mechanisms are proposed
to handle the weakly labeled human activity data, which do not require accurate
data annotation. However, these attention-based models can only handle the
weakly labeled dataset whose sample includes one target activity, as a result
it limits efficiency and practicality. In the paper, we propose a recurrent
attention networks (RAN) to handle sequential weakly labeled multi-activity
recognition and location tasks. The model can repeatedly perform steps of
attention on multiple activities of one sample and each step is corresponding
to the current focused activity. The effectiveness of the RAN model is
validated on a collected sequential weakly labeled multi-activity dataset and
the other two public datasets. The experiment results show that our RAN model
can simultaneously infer multi-activity types from the coarse-grained
sequential weak labels and determine specific locations of every target
activity with only knowledge of which types of activities contained in the long
sequence. It will greatly reduce the burden of manual labeling. The code of our
work is available at this https URL.

    

### [[2004.07054] JCS: An Explainable COVID-19 Diagnosis System by Joint Classification and Segmentation](http://arxiv.org/abs/2004.07054)


  Recently, the coronavirus disease 2019 (COVID-19) has caused a pandemic
disease in over 200 countries, influencing billions of humans. To control the
infection, identifying and separating the infected people is the most crucial
step. The main diagnostic tool is the Reverse Transcription Polymerase Chain
Reaction (RT-PCR) test. Still, the sensitivity of the RT-PCR test is not high
enough to effectively prevent the pandemic. The chest CT scan test provides a
valuable complementary tool to the RT-PCR test, and it can identify the
patients in the early-stage with high sensitivity. However, the chest CT scan
test is usually time-consuming, requiring about 21.5 minutes per case. This
paper develops a novel Joint Classification and Segmentation (JCS) system to
perform real-time and explainable COVID-19 chest CT diagnosis. To train our JCS
system, we construct a large scale COVID-19 Classification and Segmentation
(COVID-CS) dataset, with 144,167 chest CT images of 400 COVID-19 patients and
350 uninfected cases. 3,855 chest CT images of 200 patients are annotated with
fine-grained pixel-level labels of opacifications, which are increased
attenuation of the lung parenchyma. We also have annotated lesion counts,
opacification areas, and locations and thus benefit various diagnosis aspects.
Extensive experiments demonstrate that the proposed JCS diagnosis system is
very efficient for COVID-19 classification and segmentation. It obtains an
average sensitivity of 95.0% and a specificity of 93.0% on the classification
test set, and 78.5% Dice score on the segmentation test set of our COVID-CS
dataset. The COVID-CS dataset and code are available at
this https URL.

    

### [[2005.06284] Artificial Neural Network Pruning to Extract Knowledge](http://arxiv.org/abs/2005.06284)


  Artificial Neural Networks (NN) are widely used for solving complex problems
from medical diagnostics to face recognition. Despite notable successes, the
main disadvantages of NN are also well known: the risk of overfitting, lack of
explainability (inability to extract algorithms from trained NN), and high
consumption of computing resources. Determining the appropriate specific NN
structure for each problem can help overcome these difficulties: Too poor NN
cannot be successfully trained, but too rich NN gives unexplainable results and
may have a high chance of overfitting. Reducing precision of NN parameters
simplifies the implementation of these NN, saves computing resources, and makes
the NN skills more transparent. This paper lists the basic NN simplification
problems and controlled pruning procedures to solve these problems. All the
described pruning procedures can be implemented in one framework. The developed
procedures, in particular, find the optimal structure of NN for each task,
measure the influence of each input signal and NN parameter, and provide a
detailed verbal description of the algorithms and skills of NN. The described
methods are illustrated by a simple example: the generation of explicit
algorithms for predicting the results of the US presidential election.

    

### [[2006.04593] ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing](http://arxiv.org/abs/2006.04593)


  We propose AriaNN, a low-interaction privacy-preserving framework for private
neural network training and inference on sensitive data. Our semi-honest
2-party computation protocol leverages function secret sharing, a recent
lightweight cryptographic protocol that allows us to achieve an efficient
online phase. We design optimized primitives for the building blocks of neural
networks such as ReLU, MaxPool and BatchNorm. For instance, we perform private
comparison for ReLU operations with a single message of the size of the input
during the online phase, and with preprocessing keys close to 4X smaller than
previous work. Last, we propose an extension to support n-party private
federated learning. We implement our framework as an extensible system on top
of PyTorch that leverages CPU and GPU hardware acceleration for cryptographic
and machine learning operations. We evaluate our end-to-end system for private
inference and training on standard neural networks such as AlexNet, VGG16 or
ResNet18 between distant servers. We show that computation rather than
communication is the main bottleneck and that using GPUs together with reduced
key size is a promising solution to overcome this barrier.

    

### [[2006.05573] Global Data Science Project for COVID-19](http://arxiv.org/abs/2006.05573)


  This paper aims at providing the summary of the Global Data Science Project
(GDSC) for COVID-19. as on May 31 2020. COVID-19 has largely impacted on our
societies through both direct and indirect effects transmitted by the policy
measures to counter the spread of viruses. We quantitatively analysed the
multifaceted impacts of the COVID-19 pandemic on our societies including
people's mobility, health, and social behaviour changes. People's mobility has
changed significantly due to the implementation of travel restriction and
quarantine measurements. Indeed, the physical distance has widened at
international (cross-border), national and regional level. At international
level, due to the travel restrictions, the number of international flights has
plunged overall at around 88 percent during March. In particular, the number of
flights connecting Europe dropped drastically in mid of March after the United
States announced travel restrictions to Europe and the EU and participating
countries agreed to close borders, at 84 percent decline compared to March
10th. Similarly, we examined the impacts of quarantine measures in the major
city: Tokyo (Japan), New York City (the United States), and Barcelona (Spain).
Within all three cities, we found the significant decline in traffic volume. We
also identified the increased concern for mental health through the analysis of
posts on social networking services such as Twitter and Instagram. Notably, in
the beginning of April 2020, the number of post with #depression on Instagram
doubled, which might reflect the rise in mental health awareness among
Instagram users. Besides, we identified the changes in a wide range of people's
social behaviors, as well as economic impacts through the analysis of Instagram
data and primary survey data.

    

### [[2007.12371] Dopant Network Processing Units: Towards Efficient Neural-network Emulators with High-capacity Nanoelectronic Nodes](http://arxiv.org/abs/2007.12371)


  The rapidly growing computational demands of deep neural networks require
novel hardware designs. Recently, tunable nanoelectronic devices were developed
based on hopping electrons through a network of dopant atoms in silicon. These
"Dopant Network Processing Units" (DNPUs) are highly energy-efficient and have
potentially very high throughput. By adapting the control voltages applied to
its terminals, a single DNPU can solve a variety of linearly non-separable
classification problems. However, using a single device has limitations due to
the implicit single-node architecture. This paper presents a promising novel
approach to neural information processing by introducing DNPUs as high-capacity
neurons and moving from a single to a multi-neuron framework. By implementing
and testing a small multi-DNPU classifier in hardware, we show that
feed-forward DNPU networks improve the performance of a single DNPU from 77% to
94% test accuracy on a binary classification task with concentric classes on a
plane. Furthermore, motivated by the integration of DNPUs with memristor
arrays, we study the potential of using DNPUs in combination with linear
layers. We show by simulation that a single-layer MNIST classifier with only 10
DNPUs achieves over 96% test accuracy. Our results pave the road towards
hardware neural-network emulators that offer atomic-scale information
processing with low latency and energy consumption.

    

### [[2008.09381] A Survey on Assessing the Generalization Envelope of Deep Neural Networks: Predictive Uncertainty, Out-of-distribution and Adversarial Samples](http://arxiv.org/abs/2008.09381)


  Deep Neural Networks (DNNs) achieve state-of-the-art performance on numerous
applications. However, it is difficult to tell beforehand if a DNN receiving an
input will deliver the correct output since their decision criteria are usually
nontransparent. A DNN delivers the correct output if the input is within the
area enclosed by its generalization envelope. In this case, the information
contained in the input sample is processed reasonably by the network. It is of
large practical importance to assess at inference time if a DNN generalizes
correctly. Currently, the approaches to achieve this goal are investigated in
different problem set-ups rather independently from one another, leading to
three main research and literature fields: predictive uncertainty,
out-of-distribution detection and adversarial example detection. This survey
connects the three fields within the larger framework of investigating the
generalization performance of machine learning methods and in particular DNNs.
We underline the common ground, point at the most promising approaches and give
a structured overview of the methods that provide at inference time means to
establish if the current input is within the generalization envelope of a DNN.

    

### [[2009.12740] STAN: Synthetic Network Traffic Generation with Generative Neural Models](http://arxiv.org/abs/2009.12740)


  Deep learning models have achieved great success in recent years but progress
in some domains like cybersecurity is stymied due to a paucity of realistic
datasets. Organizations are reluctant to share such data, even internally, due
to privacy reasons. An alternative is to use synthetically generated data but
existing methods are limited in their ability to capture complex dependency
structures, between attributes and across time. This paper presents STAN
(Synthetic network Traffic generation with Autoregressive Neural models), a
tool to generate realistic synthetic network traffic datasets for subsequent
downstream applications. Our novel neural architecture captures both temporal
dependencies and dependence between attributes at any given time. It integrates
convolutional neural layers with mixture density neural layers and softmax
layers, and models both continuous and discrete variables. We evaluate the
performance of STAN in terms of the quality of data generated, by training it
on both a simulated dataset and a real network traffic data set. Finally, to
answer the question - can real network traffic data be substituted with
synthetic data to train models of comparable accuracy? We train two anomaly
detection models based on self-supervision. The results show only a small
decline in the accuracy of models trained solely on synthetic data. While
current results are encouraging in terms of quality of data generated and
absence of any obvious data leakage from training data, in the future we plan
to further validate this fact by conducting privacy attacks on the generated
data. Other future work includes validating capture of long term dependencies
and making model training

    

### [[2010.05328] Three-Dimensional Swarming Using Cyclic Stochastic Optimization](http://arxiv.org/abs/2010.05328)


  In this paper we simulate an ensemble of cooperating, mobile sensing agents
that implement the cyclic stochastic optimization (CSO) algorithm in an attempt
to survey and track multiple targets. In the CSO algorithm proposed, each agent
uses its sensed measurements, its shared information, and its predictions of
others' future motion to decide on its next action. This decision is selected
to minimize a loss function that decreases as the uncertainty in the targets'
state estimates decreases. Only noisy measurements of this loss function are
available to each agent, and in this study, each agent attempts to minimize
this function by calculating its stochastic gradient. This paper examines, via
simulation-based experiments, the implications and applicability of CSO
convergence in three dimensions.

    

### [[2010.05337] DistDGL: Distributed Graph Neural Network Training for Billion-Scale Graphs](http://arxiv.org/abs/2010.05337)


  Graph neural networks (GNN) have shown great success in learning from
graph-structured data. They are widely used in various applications, such as
recommendation, fraud detection, and search. In these domains, the graphs are
typically large, containing hundreds of millions of nodes and several billions
of edges. To tackle this challenge, we develop DistDGL, a system for training
GNNs in a mini-batch fashion on a cluster of machines. DistDGL is based on the
Deep Graph Library (DGL), a popular GNN development framework. DistDGL
distributes the graph and its associated data (initial features and embeddings)
across the machines and uses this distribution to derive a computational
decomposition by following an owner-compute rule. DistDGL follows a synchronous
training approach and allows ego-networks forming the mini-batches to include
non-local nodes. To minimize the overheads associated with distributed
computations, DistDGL uses a high-quality and light-weight min-cut graph
partitioning algorithm along with multiple balancing constraints. This allows
it to reduce communication overheads and statically balance the computations.
It further reduces the communication by replicating halo nodes and by using
sparse embedding updates. The combination of these design choices allows
DistDGL to train high-quality models while achieving high parallel efficiency
and memory scalability. We demonstrate our optimizations on both inductive and
transductive GNN models. Our results show that DistDGL achieves linear speedup
without compromising model accuracy and requires only 13 seconds to complete a
training epoch for a graph with 100 million nodes and 3 billion edges on a
cluster with 16 machines. DistDGL is now publicly available as part of
DGL:this https URL.

    

### [[2010.05906] Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning](http://arxiv.org/abs/2010.05906)


  Abductive and counterfactual reasoning, core abilities of everyday human
cognition, require reasoning about what might have happened at time t, while
conditioning on multiple contexts from the relative past and future. However,
simultaneous incorporation of past and future contexts using generative
language models (LMs) can be challenging, as they are trained either to
condition only on the past context or to perform narrowly scoped
text-infilling. In this paper, we propose DeLorean, a new unsupervised decoding
algorithm that can flexibly incorporate both the past and future contexts using
only off-the-shelf, left-to-right language models and no supervision. The key
intuition of our algorithm is incorporating the future through
back-propagation, during which, we only update the internal representation of
the output while fixing the model parameters. By alternating between forward
and backward propagation, DeLorean can decode the output representation that
reflects both the left and right contexts. We demonstrate that our approach is
general and applicable to two nonmonotonic reasoning tasks: abductive text
generation and counterfactual story revision, where DeLorean outperforms a
range of unsupervised and some supervised methods, based on automatic and human
evaluation.

    

### [[2010.07873] Neograd: Near-Ideal Gradient Descent](http://arxiv.org/abs/2010.07873)


  The purpose of this paper is to improve upon existing variants of gradient
descent by solving two problems: (1) removing (or reducing) the plateau that
occurs while minimizing the cost function, (2) continually adjusting the
learning rate to an "ideal" value. The approach taken is to approximately solve
for the learning rate as a function of a trust metric. When this technique is
hybridized with momentum, it creates an especially effective gradient descent
variant, called NeogradM. It is shown to outperform Adam on several test
problems, and can easily reach cost function values that are smaller by a
factor of $10^8$, for example.

    

### [[2010.08920] Average-reward model-free reinforcement learning: a systematic review and literature mapping](http://arxiv.org/abs/2010.08920)


  Reinforcement learning is important part of artificial intelligence. In this
paper, we review model-free reinforcement learning that utilizes the average
reward optimality criterion in the infinite horizon setting. Motivated by the
solo survey by Mahadevan (1996a), we provide an updated review of work in this
area and extend it to cover policy-iteration and function approximation methods
(in addition to the value-iteration and tabular counterparts). We present a
comprehensive literature mapping. We also identify and discuss opportunities
for future work.

    

### [[2012.00452] Counting People by Estimating People Flows](http://arxiv.org/abs/2012.00452)


  Modern methods for counting people in crowded scenes rely on deep networks to
estimate people densities in individual images. As such, only very few take
advantage of temporal consistency in video sequences, and those that do only
impose weak smoothness constraints across consecutive frames. In this paper, we
advocate estimating people flows across image locations between consecutive
images and inferring the people densities from these flows instead of directly
regressing them. This enables us to impose much stronger constraints encoding
the conservation of the number of people. As a result, it significantly boosts
performance without requiring a more complex architecture. Furthermore, it
allows us to exploit the correlation between people flow and optical flow to
further improve the results. We also show that leveraging people conservation
constraints in both a spatial and temporal manner makes it possible to train a
deep crowd counting model in an active learning setting with much fewer
annotations. This significantly reduces the annotation cost while still leading
to similar performance to the full supervision case.

    

### [[2012.08883] Multi-type Disentanglement without Adversarial Training](http://arxiv.org/abs/2012.08883)


  Controlling the style of natural language by disentangling the latent space
is an important step towards interpretable machine learning. After the latent
space is disentangled, the style of a sentence can be transformed by tuning the
style representation without affecting other features of the sentence. Previous
works usually use adversarial training to guarantee that disentangled vectors
do not affect each other. However, adversarial methods are difficult to train.
Especially when there are multiple features (e.g., sentiment, or tense, which
we call style types in this paper), each feature requires a separate
discriminator for extracting a disentangled style vector corresponding to that
feature. In this paper, we propose a unified distribution-controlling method,
which provides each specific style value (the value of style types, e.g.,
positive sentiment, or past tense) with a unique representation. This method
contributes a solid theoretical basis to avoid adversarial training in
multi-type disentanglement. We also propose multiple loss functions to achieve
a style-content disentanglement as well as a disentanglement among multiple
style types. In addition, we observe that if two different style types always
have some specific style values that occur together in the dataset, they will
affect each other when transferring the style values. We call this phenomenon
training bias, and we propose a loss function to alleviate such training bias
while disentangling multiple types. We conduct experiments on two datasets
(Yelp service reviews and Amazon product reviews) to evaluate the
style-disentangling effect and the unsupervised style transfer performance on
two style types: sentiment and tense. The experimental results show the
effectiveness of our model.

    

### [[2101.02966] Infinite-dimensional Folded-in-time Deep Neural Networks](http://arxiv.org/abs/2101.02966)


  The method recently introduced in arXiv:2011.10115 realizes a deep neural
network with just a single nonlinear element and delayed feedback. It is
applicable for the description of physically implemented neural networks. In
this work, we present an infinite-dimensional generalization, which allows for
a more rigorous mathematical analysis and a higher flexibility in choosing the
weight functions. Precisely speaking, the weights are described by Lebesgue
integrable functions instead of step functions. We also provide a functional
back-propagation algorithm, which enables gradient descent training of the
weights. In addition, with a slight modification, our concept realizes
recurrent neural networks.

    

### [[2101.03581] Curvature-based Feature Selection with Application in Classifying Electronic Health Records](http://arxiv.org/abs/2101.03581)


  Disruptive technologies provides unparalleled opportunities to contribute to
the identifications of many aspects in pervasive healthcare, from the adoption
of the Internet of Things through to Machine Learning (ML) techniques. As a
powerful tool, ML has been widely applied in patient-centric healthcare
solutions. To further improve the quality of patient care, Electronic Health
Records (EHRs) are widely applied in healthcare facilities nowadays. Due to the
inherent heterogeneity, unbalanced, incompleteness, and high-dimensional nature
of EHRs, it is a challenging task to employ machine learning algorithms to
analyse such EHRs for prediction and diagnostics within the scope of precision
medicine. Dimensionality reduction is an efficient data preprocessing technique
for the analysis of high dimensional data that reduces the number of features
while improving the performance of the data analysis, e.g. classification. In
this paper, we propose an efficient curvature-based feature selection method
for supporting more precise diagnosis. The proposed method is a filter-based
feature selection method, which directly utilises the Menger Curvature for
ranking all the attributes in the given data set. We evaluate the performance
of our method against conventional PCA and recent ones including BPCM, GSAM,
WCNN, BLS II, VIBES, 2L-MJFA, RFGA, and VAF. Our method achieves
state-of-the-art performance on four benchmark healthcare data sets including
CCRFDS, BCCDS, BTDS, and DRDDS with impressive 24.73% and 13.93% improvements
respectively on BTDS and CCRFDS, 7.97% improvement on BCCDS, and 3.63%
improvement on DRDDS. Our CFS source code is publicly available at
this https URL.

    

### [[2101.08685] ItNet: iterative neural networks with small graphs for accurate, efficient and anytime semantic segmentation](http://arxiv.org/abs/2101.08685)


  Deep neural networks have usually to be compressed and accelerated for their
usage in low-power, e.g. mobile, devices. Recently, massively-parallel hardware
accelerators were developed that offer high throughput and low latency at low
power by utilizing in-memory computation. However, to exploit these benefits
the computational graph of a neural network has to fit into the in-computation
memory of these hardware systems that is usually rather limited in size. In
this study, we introduce a class of network models that have a small memory
footprint in terms of their computational graphs. To this end, the graph is
designed to contain loops by iteratively executing a single network building
block. Furthermore, the trade-off between accuracy and latency of these
so-called iterative neural networks is improved by adding multiple intermediate
outputs during both training and inference. We show state-of-the-art results
for semantic segmentation on the CamVid and Cityscapes datasets that are
especially demanding in terms of computational resources. In ablation studies,
the improvement of network training by intermediate network outputs as well as
the trade-off between weight sharing over iterations and the network size are
investigated.

    

### [[2101.11605] Bottleneck Transformers for Visual Recognition](http://arxiv.org/abs/2101.11605)


  We present BoTNet, a conceptually simple yet powerful backbone architecture
that incorporates self-attention for multiple computer vision tasks including
image classification, object detection and instance segmentation. By just
replacing the spatial convolutions with global self-attention in the final
three bottleneck blocks of a ResNet and no other changes, our approach improves
upon the baselines significantly on instance segmentation and object detection
while also reducing the parameters, with minimal overhead in latency. Through
the design of BoTNet, we also point out how ResNet bottleneck blocks with
self-attention can be viewed as Transformer blocks. Without any bells and
whistles, BoTNet achieves 44.4% Mask AP and 49.7% Box AP on the COCO Instance
Segmentation benchmark using the Mask R-CNN framework; surpassing the previous
best published single model and single scale results of ResNeSt evaluated on
the COCO validation set. Finally, we present a simple adaptation of the BoTNet
design for image classification, resulting in models that achieve a strong
performance of 84.7% top-1 accuracy on the ImageNet benchmark while being up to
1.64x faster in compute time than the popular EfficientNet models on TPU-v3
hardware. We hope our simple and effective approach will serve as a strong
baseline for future research in self-attention models for vision

    

### [[2102.06191] Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals](http://arxiv.org/abs/2102.06191)


  Being able to learn dense semantic representations of images without
supervision is an important problem in computer vision. However, despite its
significance, this problem remains rather unexplored, with a few exceptions
that considered unsupervised semantic segmentation on small-scale datasets with
a narrow visual domain. In this paper, we make a first attempt to tackle the
problem on datasets that have been traditionally utilized for the supervised
case. To achieve this, we introduce a two-step framework that adopts a
predetermined mid-level prior in a contrastive optimization objective to learn
pixel embeddings. This marks a large deviation from existing works that relied
on proxy tasks or end-to-end clustering. Additionally, we argue about the
importance of having a prior that contains information about objects, or their
parts, and discuss several possibilities to obtain such a prior in an
unsupervised manner.
Experimental evaluation shows that our method comes with key advantages over
existing works. First, the learned pixel embeddings can be directly clustered
in semantic groups using K-Means on PASCAL. Under the fully unsupervised
setting, there is no precedent in solving the semantic segmentation task on
such a challenging benchmark. Second, our representations can improve over
strong baselines when transferred to new datasets, e.g. COCO and DAVIS. The
code is available.

    

### [[2102.10859] Recursive Least Squares Based Refinement Network for the Rollout Trajectory Prediction Methods](http://arxiv.org/abs/2102.10859)


  Trajectory prediction plays a pivotal role in the field of intelligent
vehicles. It currently suffers from several challenges,e.g., accumulative error
in rollout process and weak adaptability in various scenarios. This paper
proposes a parametric-learning recursive least squares (RLS) estimation based
on deep neural network for trajectory prediction. We design a flexible plug-in
module which can be readily implanted into rollout approaches. Goal points are
proposed to capture the long-term prediction stability from the global
perspective. We carried experiments out on the NGSIM dataset. The promising
results indicate that our method could improve rollout trajectory prediction
methods effectively.

    

### [[2103.02383] Nonlinear MPC for Offset-Free Tracking of systems learned by GRU Neural Networks](http://arxiv.org/abs/2103.02383)


  The use of Recurrent Neural Networks (RNNs) for system identification has
recently gathered increasing attention, thanks to their black-box modeling
capabilities.Albeit RNNs have been fruitfully adopted in many applications,
only few works are devoted to provide rigorous theoretical foundations that
justify their use for control purposes. The aim of this paper is to describe
how stable Gated Recurrent Units (GRUs), a particular RNN architecture, can be
trained and employed in a Nonlinear MPC framework to perform offset-free
tracking of constant references with guaranteed closed-loop stability. The
proposed approach is tested on a pH neutralization process benchmark, showing
remarkable performances.

    

### [[2103.03111] Robust Binary Neural Network Operation from 233 K to 398 K via Gate Stack and Bias Optimization of Ferroelectric FinFET Synapses](http://arxiv.org/abs/2103.03111)


  A synergistic approach for optimizing devices, circuits, and neural network
architectures was used to abate junction-temperature-change-induced performance
degradation of a Fe-FinFET-based artificial neural network. We demonstrated
that the digital nature of the binarized neural network, with the "0" state
programmed deep in the subthreshold and the "1" state in strong inversion, is
crucial for robust DNN inference. The performance of a purely software-based
binary neural network (BNN), with 96.1% accuracy for Modified National
Institute of Standards and Technology (MNIST) handwritten digit recognition,
was used as a baseline. The Fe-FinFET-based BNN (including device-to-device
variation at 300 K) achieved 95.7% inference accuracy on the MNIST dataset.
Although substantial inference accuracy degradation with temperature change was
observed in a nonbinary neural network, the BNN with optimized Fe-FinFETs as
synaptic devices had excellent resistance to temperature change effects and
maintained a minimum inference accuracy of 95.2% within a temperature range of
-233K to 398K after gate stack and bias optimization. However, reprogramming to
adjust device conductance was necessary for temperatures higher than 398K.

    

### [[2103.05127] Model Complexity of Deep Learning: A Survey](http://arxiv.org/abs/2103.05127)


  Model complexity is a fundamental problem in deep learning. In this paper we
conduct a systematic overview of the latest studies on model complexity in deep
learning. Model complexity of deep learning can be categorized into expressive
capacity and effective model complexity. We review the existing studies on
those two categories along four important factors, including model framework,
model size, optimization process and data complexity. We also discuss the
applications of deep learning model complexity including understanding model
generalization, model optimization, and model selection and design. We conclude
by proposing several interesting future directions.

    

### [[2104.04853] Beyond Pointwise Submodularity: Non-Monotone Adaptive Submodular Maximization subject to Knapsack and $k$-System Constraints](http://arxiv.org/abs/2104.04853)


  In this paper, we study the non-monotone adaptive submodular maximization
problem subject to a knapsack and a $k$-system constraints. The input of our
problem is a set of items, where each item has a particular state drawn from a
known prior distribution. However, the state of an item is initially unknown,
one must select an item in order to reveal the state of that item. There is a
utility function which is defined over items and states. Our objective is to
sequentially select a group of items to maximize the expected utility. Although
the cardinality-constrained non-monotone adaptive submodular maximization has
been well studied in the literature, whether there exists a constant
approximation solution for the knapsack-constrained or $k$-system constrained
adaptive submodular maximization problem remains an open problem. It fact, it
has only been settled given the additional assumption of pointwise
submodularity. In this paper, we remove the common assumption on pointwise
submodularity and propose the first constant approximation solutions for both
cases. Inspired by two recent studies on non-monotone adaptive submodular
maximization, we develop a sampling-based randomized algorithm that achieves a
$\frac{1}{10}$ approximation for the case of a knapsack constraint and that
achieves a $\frac{1}{2k+4}$ approximation ratio for the case of a $k$-system
constraint.

    

### [[2104.05632] Augmented World Models Facilitate Zero-Shot Dynamics Generalization From a Single Offline Environment](http://arxiv.org/abs/2104.05632)


  Reinforcement learning from large-scale offline datasets provides us with the
ability to learn policies without potentially unsafe or impractical
exploration. Significant progress has been made in the past few years in
dealing with the challenge of correcting for differing behavior between the
data collection and learned policies. However, little attention has been paid
to potentially changing dynamics when transferring a policy to the online
setting, where performance can be up to 90% reduced for existing methods. In
this paper we address this problem with Augmented World Models (AugWM). We
augment a learned dynamics model with simple transformations that seek to
capture potential changes in physical properties of the robot, leading to more
robust policies. We not only train our policy in this new setting, but also
provide it with the sampled augmentation as a context, allowing it to adapt to
changes in the environment. At test time we learn the context in a
self-supervised fashion by approximating the augmentation which corresponds to
the new environment. We rigorously evaluate our approach on over 100 different
changed dynamics settings, and show that this simple approach can significantly
improve the zero-shot generalization of a recent state-of-the-art baseline,
often achieving successful policies where the baseline fails.

    

### [[2105.08970] Disentanglement Learning for Variational Autoencoders Applied to Audio-Visual Speech Enhancement](http://arxiv.org/abs/2105.08970)


  Recently, the standard variational autoencoder has been successfully used to
learn a probabilistic prior over speech signals, which is then used to perform
speech enhancement. Variational autoencoders have then been conditioned on a
label describing a high-level speech attribute (e.g. speech activity) that
allows for a more explicit control of speech generation. However, the label is
not guaranteed to be disentangled from the other latent variables, which
results in limited performance improvements compared to the standard
variational autoencoder. In this work, we propose to use an adversarial
training scheme for variational autoencoders to disentangle the label from the
other latent variables. At training, we use a discriminator that competes with
the encoder of the variational autoencoder. Simultaneously, we also use an
additional encoder that estimates the label for the decoder of the variational
autoencoder, which proves to be crucial to learn disentanglement. We show the
benefit of the proposed disentanglement learning when a voice activity label,
estimated from visual data, is used for speech enhancement.

    

### [[2105.13336] TENSILE: A Tensor granularity dynamic GPU memory scheduling method towards multiple dynamic workloads system](http://arxiv.org/abs/2105.13336)


  Recently, deep learning has been an area of intense researching. However, as
a kind of computing intensive task, deep learning highly relies on the scale of
GPU memory, which is usually prohibitive and scarce. Although there are some
extensive works have been proposed for dynamic GPU memory management, they are
hard to be applied to systems with multiple dynamic workloads, such as
in-database machine learning system.
In this paper, we demonstrated TENSILE, a method of managing GPU memory in
tensor granularity to reduce the GPU memory peak, with taking the multiple
dynamic workloads into consideration. As far as we know, TENSILE is the first
method which is designed to manage multiple workloads' GPU memory using. We
implement TENSILE on a deep learning framework built by ourselves, and
evaluated its performance. The experiment results show that TENSILE can save
more GPU memory with less extra time overhead than prior works in both single
and multiple dynamic workloads scenarios.

    

### [[2106.03702] Can a single neuron learn quantiles?](http://arxiv.org/abs/2106.03702)


  A novel non-parametric quantile estimation method for continuous random
variables is introduced, based on a minimal neural network architecture
consisting of a single unit. Its advantage over estimations from ranking the
order statistics is shown, specifically for small sample size. In a regression
context, the method can be used to quantify predictive uncertainty under the
split conformal prediction setting, where prediction intervals are estimated
from the residuals of a pre-trained model on a held-out validation set to
quantify the uncertainty in future predictions. Benchmarking experiments
demonstrate that the method is competitive in quality and coverage with
state-of-the-art solutions, with the added benefit of being more
computationally efficient.

    

### [[2106.09764] A probabilistic database approach to autoencoder-based data cleaning](http://arxiv.org/abs/2106.09764)


  Data quality problems are a large threat in data science. In this paper, we
propose a data-cleaning autoencoder capable of near-automatic data quality
improvement. It learns the structure and dependencies in the data and uses it
as evidence to identify and correct doubtful values. We apply a probabilistic
database approach to represent weak and strong evidence for attribute value
repairs. A theoretical framework is provided, and experiments show that it can
remove significant amounts of noise (i.e., data quality problems) from
categorical and numeric probabilistic data. Our method does not require clean
data. We do, however, show that manually cleaning a small fraction of the data
significantly improves performance.

    

### [[2106.15962] On the Generative Utility of Cyclic Conditionals](http://arxiv.org/abs/2106.15962)


  We study whether and how can we model a joint distribution $p(x,z)$ using two
conditional models $p(x|z)$ and $q(z|x)$ that form a cycle. This is motivated
by the observation that deep generative models, in addition to a likelihood
model $p(x|z)$, often also use an inference model $q(z|x)$ for data
representation, but they rely on a usually uninformative prior distribution
$p(z)$ to define a joint distribution, which may render problems like posterior
collapse and manifold mismatch. To explore the possibility to model a joint
distribution using only $p(x|z)$ and $q(z|x)$, we study their compatibility and
determinacy, corresponding to the existence and uniqueness of a joint
distribution whose conditional distributions coincide with them. We develop a
general theory for novel and operable equivalence criteria for compatibility,
and sufficient conditions for determinacy. Based on the theory, we propose the
CyGen framework for cyclic-conditional generative modeling, including methods
to enforce compatibility and use the determined distribution to fit and
generate data. With the prior constraint removed, CyGen better fits data and
captures more representative features, supported by experiments showing better
generation and downstream classification performance.

    

### [[2107.00798] Near-optimal Algorithms for Explainable k-Medians and k-Means](http://arxiv.org/abs/2107.00798)


  We consider the problem of explainable $k$-medians and $k$-means introduced
by Dasgupta, Frost, Moshkovitz, and Rashtchian~(ICML 2020). In this problem,
our goal is to find a threshold decision tree that partitions data into $k$
clusters and minimizes the $k$-medians or $k$-means objective. The obtained
clustering is easy to interpret because every decision node of a threshold tree
splits data based on a single feature into two groups. We propose a new
algorithm for this problem which is $\tilde O(\log k)$ competitive with
$k$-medians with $\ell_1$ norm and $\tilde O(k)$ competitive with $k$-means.
This is an improvement over the previous guarantees of $O(k)$ and $O(k^2)$ by
Dasgupta et al (2020). We also provide a new algorithm which is $O(\log^{3/2}
k)$ competitive for $k$-medians with $\ell_2$ norm. Our first algorithm is
near-optimal: Dasgupta et al (2020) showed a lower bound of $\Omega(\log k)$
for $k$-medians; in this work, we prove a lower bound of $\tilde\Omega(k)$ for
$k$-means. We also provide a lower bound of $\Omega(\log k)$ for $k$-medians
with $\ell_2$ norm.

    

### [[2108.00977] Multilevel Knowledge Transfer for Cross-Domain Object Detection](http://arxiv.org/abs/2108.00977)


  Domain shift is a well known problem where a model trained on a particular
domain (source) does not perform well when exposed to samples from a different
domain (target). Unsupervised methods that can adapt to domain shift are highly
desirable as they allow effective utilization of the source data without
requiring additional annotated training data from the target. Practically,
obtaining sufficient amount of annotated data from the target domain can be
both infeasible and extremely expensive. In this work, we address the domain
shift problem for the object detection task. Our approach relies on gradually
removing the domain shift between the source and the target domains. The key
ingredients to our approach are -- (a) mapping the source to the target domain
on pixel-level; (b) training a teacher network on the mapped source and the
unannotated target domain using adversarial feature alignment; and (c) finally
training a student network using the pseudo-labels obtained from the teacher.
Experimentally, when tested on challenging scenarios involving domain shift, we
consistently obtain significantly large performance gains over various recent
state of the art approaches.

    

### [[2108.01005] Sequoia: A Software Framework to Unify Continual Learning Research](http://arxiv.org/abs/2108.01005)


  The field of Continual Learning (CL) seeks to develop algorithms that
accumulate knowledge and skills over time through interaction with
non-stationary environments and data distributions. Measuring progress in CL
can be difficult because a plethora of evaluation procedures (ettings) and
algorithmic solutions (methods) have emerged, each with their own potentially
disjoint set of assumptions about the CL problem. In this work, we view each
setting as a set of assumptions. We then create a tree-shaped hierarchy of the
research settings in CL, in which more general settings become the parents of
those with more restrictive assumptions. This makes it possible to use
inheritance to share and reuse research, as developing a method for a given
setting also makes it directly applicable onto any of its children. We
instantiate this idea as a publicly available software framework called
Sequoia, which features a variety of settings from both the Continual
Supervised Learning (CSL) and Continual Reinforcement Learning (CRL) domains.
Sequoia also includes a growing suite of methods which are easy to extend and
customize, in addition to more specialized methods from third-party libraries.
We hope that this new paradigm and its first implementation can serve as a
foundation for the unification and acceleration of research in CL. You can help
us grow the tree by visiting this http URL.

    

### [[2108.01298] Synthesizing Brain-Network-Inspired Interconnections for Large-Scale Network-on-Chips](http://arxiv.org/abs/2108.01298)


  Brain network is a large-scale complex network with scale-free, small-world,
and modularity properties, which largely supports this high-efficiency massive
system. In this paper, we propose to synthesize brain-network-inspired
interconnections for large-scale network-on-chips. Firstly, we propose a method
to generate brain-network-inspired topologies with limited scale-free and
power-law small-world properties, which have a low total link length and
extremely low average hop count approximately proportional to the logarithm of
the network size. In addition, given the large-scale applications and the
modular topology, we present an application mapping method, including task
mapping and deterministic deadlock-free routing, to minimize the power
consumption and hop count. Finally, a cycle-accurate simulator BookSim2 is used
to validate the architecture performance with different synthetic traffic
patterns and large-scale test cases, including real-world communication
networks for the graph processing application. Experiments show that, compared
with other topologies and methods, the NoC design generated by the proposed
method presents significantly lower average hop count and lower average
latency. Especially in graph processing applications with a power-law and
tightly coupled inter-core communication, the brain-network-inspired NoC has up
to 70% lower average hop count and 75% lower average latency than mesh-based
NoCs.

    

### [[2108.01565] Hardware-aware Design of Multiplierless Second-Order IIR Filters with Minimum Adders](http://arxiv.org/abs/2108.01565)


  In this work, we optimally solve the problem of multiplierless design of
second-order Infinite Impulse Response filters with minimum number of adders.
Given a frequency specification, we design a stable direct form filter with
hardware-aware fixed-point coefficients that yielding minimal number of adders
when replacing all the multiplications by bit shifts and additions. The
coefficient design, quantization and implementation, typically conducted
independently, are now gathered into one global optimization problem, modeled
through integer linear programming and efficiently solved using generic
solvers. We guarantee the frequency-domain specifications and stability, which
together with optimal number of adders will significantly simplify design-space
exploration for filter designers. The optimal filters are implemented within
the FloPoCo IP core generator and synthesized for Field Programmable Gate
Arrays. With respect to state-of-the-art three-step filter design methods, our
one-step design approach achieves, on average, 42% reduction in the number of
lookup tables and 21% improvement in delay.

    

### [[2108.01330] Frugal Byzantine Computing](http://arxiv.org/abs/2108.01330)


  Traditional techniques for handling Byzantine failures are expensive: digital
signatures are too costly, while using $3f{+}1$ replicas is uneconomical ($f$
denotes the maximum number of Byzantine processes). We seek algorithms that
reduce the number of replicas to $2f{+}1$ and minimize the number of
signatures. While the first goal can be achieved in the message-and-memory
model, accomplishing the second goal simultaneously is challenging. We first
address this challenge for the problem of broadcasting messages reliably. We
consider two variants of this problem, Consistent Broadcast and Reliable
Broadcast, typically considered very close. Perhaps surprisingly, we establish
a separation between them in terms of signatures required. In particular, we
show that Consistent Broadcast requires at least 1 signature in some execution,
while Reliable Broadcast requires $O(n)$ signatures in some execution. We
present matching upper bounds for both primitives within constant factors. We
then turn to the problem of consensus and argue that this separation matters
for solving consensus with Byzantine failures: we present a practical consensus
algorithm that uses Consistent Broadcast as its main communication primitive.
This algorithm works for $n=2f{+}1$ and avoids signatures in the common-case --
properties that have not been simultaneously achieved previously. Overall, our
work approaches Byzantine computing in a frugal manner and motivates the use of
Consistent Broadcast -- rather than Reliable Broadcast -- as a key primitive
for reaching agreement.

    

### [[2108.01341] Using Throughput-Centric Byzantine Broadcast to Tolerate Malicious Majority in Blockchains](http://arxiv.org/abs/2108.01341)


  Fault tolerance of a blockchain is often characterized by the fraction $f$ of
``adversarial power'' that it can tolerate in the system. Despite the fast
progress in blockchain designs in recent years, existing blockchain systems can
still only tolerate $f$ below $\frac{1}{2}$. Can practically usable blockchains
tolerate a malicious majority, i.e., $f \ge \frac{1}{2}$?
This work presents a positive answer to this question. We first note that the
well-known impossibility of {\em byzantine consensus} under $f \ge \frac{1}{2}$
does not carry over to blockchains. To tolerate $f \ge \frac{1}{2}$, we use
{\em byzantine broadcast}, instead of byzantine consensus, as the core of the
blockchain. A major obstacle in doing so, however, is that the resulting
blockchain may have extremely low throughput. To overcome this central
technical challenge, we propose a novel byzantine broadcast protocol OverlayBB,
that can tolerate $f \ge \frac{1}{2}$ while achieving good throughput. Using
OverlayBB as the core, we present the design, implementation, and evaluation of
a novel Proof-of-Stake blockchain called BCube. BCube can tolerate a malicious
majority, while achieving practically usable transaction throughput and
confirmation latency in our experiments with $10000$ nodes and under $f = 0.7$.
To our knowledge, BCube is the first blockchain that can achieve such
properties.

    

### [[2108.01651] An Impossibility Result on Strong Linearizability in Message-Passing Systems](http://arxiv.org/abs/2108.01651)


  We prove that there is no 1-resilient strongly linearizable implementation of
a weak object in asynchronous message-passing systems. This object, that we
call Test-\emph{or}-Set (ToS), allows a single distinguished process to apply
the set operation once, and a \emph{different} distinguished process to apply
the test operation also once. Since this weak object can be directly
implemented by a single-writer single-reader (SWSR) register (and other common
objects such as max-register, snapshot, counter, test-and-set, queue, stack,
etc...), this result implies that there is no 1-resilient strongly linearizable
implementation of a SWSR register (and of these other objects) in
message-passing systems.

    

### [[1912.08950] Slim Graph: Practical Lossy Graph Compression for Approximate Graph Processing, Storage, and Analytics](http://arxiv.org/abs/1912.08950)


  We propose Slim Graph: the first programming model and framework for
practical lossy graph compression that facilitates high-performance approximate
graph processing, storage, and analytics. Slim Graph enables the developer to
express numerous compression schemes using small and programmable compression
kernels that can access and modify local parts of input graphs. Such kernels
are executed in parallel by the underlying engine, isolating developers from
complexities of parallel programming. Our kernels implement novel graph
compression schemes that preserve numerous graph properties, for example
connected components, minimum spanning trees, or graph spectra. Finally, Slim
Graph uses statistical divergences and other metrics to analyze the accuracy of
lossy graph compression. We illustrate both theoretically and empirically that
Slim Graph accelerates numerous graph algorithms, reduces storage used by graph
datasets, and ensures high accuracy of results. Slim Graph may become the
common ground for developing, executing, and analyzing emerging lossy graph
compression schemes.

    

### [[2101.06139] CPU Scheduling in Data Centers Using Asynchronous Finite-Time Distributed Coordination Mechanisms](http://arxiv.org/abs/2101.06139)


  We propose an asynchronous iterative scheme which allows a set of
interconnected nodes to distributively reach an agreement within a
pre-specified bound in a finite number of steps. While this scheme could be
adopted in a wide variety of applications, we discuss it within the context of
task scheduling for data centers. In this context, the algorithm is guaranteed
to approximately converge to the optimal scheduling plan, given the available
resources, in a finite number of steps. Furthermore, being asynchronous, the
proposed scheme is able to take into account the uncertainty that can be
introduced from straggler nodes or communication issues in the form of latency
variability while still converging to the target objective. In addition, by
using extensive empirical evaluation through simulations we show that the
proposed method exhibits state-of-the-art performance.

    

### [[2105.12929] Characterizing Impacts of Storage Faults on HPC Applications: A Methodology and Insights](http://arxiv.org/abs/2105.12929)


  In recent years, the increasing complexity in scientific simulations and
emerging demands for training heavy artificial intelligence models require
massive and fast data accesses, which urges high-performance computing (HPC)
platforms to equip with more advanced storage infrastructures such as
solid-state disks (SSDs). While SSDs offer high-performance I/O, the
reliability challenges faced by the HPC applications under the SSD-related
failures remains unclear, in particular for failures resulting in data
corruptions. The goal of this paper is to understand the impact of SSD-related
faults on the behaviors of complex HPC applications. To this end, we propose
FFIS, a FUSE-based fault injection framework that systematically introduces
storage faults into the application layer to model the errors originated from
SSDs. FFIS is able to plant different I/O related faults into the data returned
from underlying file systems, which enables the investigation on the error
resilience characteristics of the scientific file format. We demonstrate the
use of FFIS with three representative real HPC applications, showing how each
application reacts to the data corruptions, and provide insights on the error
resilience of the widely adopted HDF5 file format for the HPC applications.

    

### [[2108.01174] Knowledge-intensive Language Understanding for Explainable AI](http://arxiv.org/abs/2108.01174)


  AI systems have seen significant adoption in various domains. At the same
time, further adoption in some domains is hindered by inability to fully trust
an AI system that it will not harm a human. Besides the concerns for fairness,
privacy, transparency, and explainability are key to developing trusts in AI
systems. As stated in describing trustworthy AI "Trust comes through
understanding. How AI-led decisions are made and what determining factors were
included are crucial to understand." The subarea of explaining AI systems has
come to be known as XAI. Multiple aspects of an AI system can be explained;
these include biases that the data might have, lack of data points in a
particular region of the example space, fairness of gathering the data, feature
importances, etc. However, besides these, it is critical to have human-centered
explanations that are directly related to decision-making similar to how a
domain expert makes decisions based on "domain knowledge," that also include
well-established, peer-validated explicit guidelines. To understand and
validate an AI system's outcomes (such as classification, recommendations,
predictions), that lead to developing trust in the AI system, it is necessary
to involve explicit domain knowledge that humans understand and use.

    

### [[2108.01176] Hierarchical Representations and Explicit Memory: Learning Effective Navigation Policies on 3D Scene Graphs using Graph Neural Networks](http://arxiv.org/abs/2108.01176)


  Representations are crucial for a robot to learn effective navigation
policies. Recent work has shown that mid-level perceptual abstractions, such as
depth estimates or 2D semantic segmentation, lead to more effective policies
when provided as observations in place of raw sensor data (e.g., RGB images).
However, such policies must still learn latent three-dimensional scene
properties from mid-level abstractions. In contrast, high-level, hierarchical
representations such as 3D scene graphs explicitly provide a scene's geometry,
topology, and semantics, making them compelling representations for navigation.
In this work, we present a reinforcement learning framework that leverages
high-level hierarchical representations to learn navigation policies. Towards
this goal, we propose a graph neural network architecture and show how to embed
a 3D scene graph into an agent-centric feature space, which enables the robot
to learn policies for low-level action in an end-to-end manner. For each node
in the scene graph, our method uses features that capture occupancy and
semantic content, while explicitly retaining memory of the robot trajectory. We
demonstrate the effectiveness of our method against commonly used visuomotor
policies in a challenging object search task. These experiments and supporting
ablation studies show that our method leads to more effective object search
behaviors, exhibits improved long-term memory, and successfully leverages
hierarchical information to guide its navigation objectives.

    

### [[2108.01229] Taking Cognition Seriously: A generalised physics of cognition](http://arxiv.org/abs/2108.01229)


  The study of complex systems through the lens of category theory consistently
proves to be a powerful approach. We propose that cognition deserves the same
category-theoretic treatment. We show that by considering a highly-compact
cognitive system, there are fundamental physical trade-offs resulting in a
utility problem. We then examine how to do this systematically, and propose
some requirements for "cognitive categories", before investigating the
phenomenona of topological defects in gauge fields over conceptual spaces.

    

### [[2108.01234] AGAR a microbial colony dataset for deep learning detection](http://arxiv.org/abs/2108.01234)


  The Annotated Germs for Automated Recognition (AGAR) dataset is an image
database of microbial colonies cultured on agar plates. It contains 18000
photos of five different microorganisms as single or mixed cultures, taken
under diverse lighting conditions with two different cameras. All the images
are classified into "countable", "uncountable", and "empty", with the
"countable" class labeled by microbiologists with colony location and species
identification (336442 colonies in total). This study describes the dataset
itself and the process of its development. In the second part, the performance
of selected deep neural network architectures for object detection, namely
Faster R-CNN and Cascade R-CNN, was evaluated on the AGAR dataset. The results
confirmed the great potential of deep learning methods to automate the process
of microbe localization and classification based on Petri dish photos.
Moreover, AGAR is the first publicly available dataset of this kind and size
and will facilitate the future development of machine learning models. The data
used in these studies can be found at this https URL.

    

### [[2108.01254] Desk Organization: Effect of Multimodal Inputs on Spatial Relational Learning](http://arxiv.org/abs/2108.01254)


  For robots to operate in a three dimensional world and interact with humans,
learning spatial relationships among objects in the surrounding is necessary.
Reasoning about the state of the world requires inputs from many different
sensory modalities including vision ($V$) and haptics ($H$). We examine the
problem of desk organization: learning how humans spatially position different
objects on a planar surface according to organizational ''preference''. We
model this problem by examining how humans position objects given multiple
features received from vision and haptic modalities. However, organizational
habits vary greatly between people both in structure and adherence. To deal
with user organizational preferences, we add an additional modality,
''utility'' ($U$), which informs on a particular human's perceived usefulness
of a given object. Models were trained as generalized (over many different
people) or tailored (per person). We use two types of models: random forests,
which focus on precise multi-task classification, and Markov logic networks,
which provide an easily interpretable insight into organizational habits. The
models were applied to both synthetic data, which proved to be learnable when
using fixed organizational constraints, and human-study data, on which the
random forest achieved over 90% accuracy. Over all combinations of $\{H, U,
V\}$ modalities, $UV$ and $HUV$ were the most informative for organization. In
a follow-up study, we gauged participants preference of desk organizations by a
generalized random forest organization vs. by a random model. On average,
participants rated the random forest models as 4.15 on a 5-point Likert scale
compared to 1.84 for the random model

    

### [[2108.01266] More but Correct: Generating Diversified and Entity-revised Medical Response](http://arxiv.org/abs/2108.01266)


  Medical Dialogue Generation (MDG) is intended to build a medical dialogue
system for intelligent consultation, which can communicate with patients in
real-time, thereby improving the efficiency of clinical diagnosis with broad
application prospects. This paper presents our proposed framework for the
Chinese MDG organized by the 2021 China conference on knowledge graph and
semantic computing (CCKS) competition, which requires generating
context-consistent and medically meaningful responses conditioned on the
dialogue history. In our framework, we propose a pipeline system composed of
entity prediction and entity-aware dialogue generation, by adding predicted
entities to the dialogue model with a fusion mechanism, thereby utilizing
information from different sources. At the decoding stage, we propose a new
decoding mechanism named Entity-revised Diverse Beam Search (EDBS) to improve
entity correctness and promote the length and quality of the final response.
The proposed method wins both the CCKS and the International Conference on
Learning Representations (ICLR) 2021 Workshop Machine Learning for Preventing
and Combating Pandemics (MLPCP) Track 1 Entity-aware MED competitions, which
demonstrate the practicality and effectiveness of our method.

    

### [[2108.01309] Skeleton Split Strategies for Spatial Temporal Graph Convolution Networks](http://arxiv.org/abs/2108.01309)


  A skeleton representation of the human body has been proven to be effective
for this task. The skeletons are presented in graphs form-like. However, the
topology of a graph is not structured like Euclidean-based data. Therefore, a
new set of methods to perform the convolution operation upon the skeleton graph
is presented. Our proposal is based upon the ST-GCN framework proposed by Yan
et al. [1]. In this study, we present an improved set of label mapping methods
for the ST-GCN framework. We introduce three split processes (full distance
split, connection split, and index split) as an alternative approach for the
convolution operation. To evaluate the performance, the experiments presented
in this study have been trained using two benchmark datasets: NTU-RGB+D and
Kinetics. Our results indicate that all of our split processes outperform the
previous partition strategies and are more stable during training without using
the edge importance weighting additional training parameter. Therefore, our
proposal can provide a more realistic solution for real-time applications
centred on daily living recognition systems activities for indoor environments.

    

### [[2108.01326] Predicting Popularity of Images Over 30 Days](http://arxiv.org/abs/2108.01326)


  The current work deals with the problem of attempting to predict the
popularity of images before even being uploaded. This method is specifically
focused on Flickr images. Social features of each image as well as that of the
user who had uploaded it, have been recorded. The dataset also includes the
engagement score of each image which is the ground truth value of the views
obtained by each image over a period of 30 days. The work aims to predict the
popularity of images on Flickr over a period of 30 days using the social
features of the user and the image, as well as the visual features of the
images. The method states that the engagement sequence of an image can be said
to depend on two independent quantities, namely scale and shape of an image.
Once the shape and scale of an image have been predicted, combining them the
predicted sequence of an image over 30 days is obtained. The current work
follows a previous work done in the same direction, with certain speculations
and suggestions of improvement.

    

### [[2108.01360] Understanding Human Reading Comprehension with brain signals](http://arxiv.org/abs/2108.01360)


  Reading comprehension is a complex cognitive process involving many human
brain activities. Plenty of works have studied the reading patterns and
attention allocation mechanisms in the reading process. However, little is
known about what happens in human brain during reading comprehension and how we
can utilize this information as implicit feedback to facilitate information
acquisition performance. With the advances in brain imaging techniques such as
EEG, it is possible to collect high-precision brain signals in almost real
time. With neuroimaging techniques, we carefully design a lab-based user study
to investigate brain activities during reading comprehension. Our findings show
that neural responses vary with different types of contents, i.e., contents
that can satisfy users' information needs and contents that cannot. We suggest
that various cognitive activities, e.g., cognitive loading, semantic-thematic
understanding, and inferential processing, at the micro-time scale during
reading comprehension underpin these neural responses. Inspired by these
detectable differences in cognitive activities, we construct supervised
learning models based on EEG features for two reading comprehension tasks:
answer sentence classification and answer extraction. Results show that it is
feasible to improve their performance with brain signals. These findings imply
that brain signals are valuable feedback for enhancing human-computer
interactions during reading comprehension.

    

### [[2108.01380] Dynamic communication topologies for distributed heuristics in energy system optimization algorithms](http://arxiv.org/abs/2108.01380)


  The communication topology is an essential aspect in designing distributed
optimization heuristics. It can influence the exploration and exploitation of
the search space and thus the optimization performance in terms of solution
quality, convergence speed and collaboration costs, all relevant aspects for
applications operating critical infrastructure in energy systems. In this work,
we present an approach for adapting the communication topology during runtime,
based on the principles of simulated annealing. We compare the approach to
common static topologies regarding the performance of an exemplary distributed
optimization heuristic. Finally, we investigate the correlations between
fitness landscape properties and defined performance metrics.

    

### [[2108.01383] On the descriptive power of LiDAR intensity images for segment-based loop closing in 3-D SLAM](http://arxiv.org/abs/2108.01383)


  We propose an extension to the segment-based global localization method for
LiDAR SLAM using descriptors learned considering the visual context of the
segments. A new architecture of the deep neural network is presented that
learns the visual context acquired from synthetic LiDAR intensity images. This
approach allows a single multi-beam LiDAR to produce rich and highly
descriptive location signatures. The method is tested on two public datasets,
demonstrating an improved descriptiveness of the new descriptors, and more
reliable loop closure detection in SLAM. Attention analysis of the network is
used to show the importance of focusing on the broader context rather than only
on the 3-D segment.

    

### [[2108.01425] sarcasm detection and quantification in arabic tweets](http://arxiv.org/abs/2108.01425)


  The role of predicting sarcasm in the text is known as automatic sarcasm
detection. Given the prevalence and challenges of sarcasm in sentiment-bearing
text, this is a critical phase in most sentiment analysis tasks. With the
increasing popularity and usage of different social media platforms among users
around the world, people are using sarcasm more and more in their day-to-day
conversations, social media posts and tweets, and it is considered as a way for
people to express their sentiment about some certain topics or issues. As a
result of the increasing popularity, researchers started to focus their
research endeavors on detecting sarcasm from a text in different languages
especially the English language. However, the task of sarcasm detection is a
challenging task due to the nature of sarcastic texts; which can be relative
and significantly differs from one person to another depending on the topic,
region, the user's mentality and other factors. In addition to these
challenges, sarcasm detection in the Arabic language has its own challenges due
to the complexity of the Arabic language, such as being morphologically rich,
with many dialects that significantly vary between each other, while also being
lowly resourced. In recent years, only few research attempts started tackling
the task of sarcasm detection in Arabic, including creating and collecting
corpora, organizing workshops and establishing baseline models. This paper
intends to create a new humanly annotated Arabic corpus for sarcasm detection
collected from tweets, and implementing a new approach for sarcasm detection
and quantification in Arabic tweets. The annotation technique followed in this
paper is unique in sarcasm detection and the proposed approach tackles the
problem as a regression problem instead of classification; i.e., the model
attempts to predict the level of sarcasm instead of binary classification.

    

### [[2108.01453] PhotoChat: A Human-Human Dialogue Dataset with Photo Sharing Behavior for Joint Image-Text Modeling](http://arxiv.org/abs/2108.01453)


  We present a new human-human dialogue dataset - PhotoChat, the first dataset
that casts light on the photo sharing behavior in onlin emessaging. PhotoChat
contains 12k dialogues, each of which is paired with a user photo that is
shared during the conversation. Based on this dataset, we propose two tasks to
facilitate research on image-text modeling: a photo-sharing intent prediction
task that predicts whether one intends to share a photo in the next
conversation turn, and a photo retrieval task that retrieves the most relevant
photo according to the dialogue context. In addition, for both tasks, we
provide baseline models using the state-of-the-art models and report their
benchmark performances. The best image retrieval model achieves 10.4% recall@1
(out of 1000 candidates) and the best photo intent prediction model achieves
58.1% F1 score, indicating that the dataset presents interesting yet
challenging real-world problems. We are releasing PhotoChat to facilitate
future research work among the community.

    

### [[2108.01483] Research Challenges and Progress in Robotic Grasping and Manipulation Competitions](http://arxiv.org/abs/2108.01483)


  This paper discusses recent research progress in robotic grasping and
manipulation in the light of the latest Robotic Grasping and Manipulation
Competitions (RGMCs). We first provide an overview of past benchmarks and
competitions related to the robotics manipulation field. Then, we discuss the
methodology behind designing the manipulation tasks in RGMCs. We provide a
detailed analysis of key challenges for each task and identify the most
difficult aspects based on the competing teams' performance in recent years. We
believe that such an analysis is insightful to determine the future research
directions for the robotic manipulation domain.

    

### [[2108.01508] Learning Nonlinear Waves in Plasmon-induced Transparency](http://arxiv.org/abs/2108.01508)


  Plasmon-induced transparency (PIT) displays complex nonlinear dynamics that
find critical phenomena in areas such as nonlinear waves. However, such a
nonlinear solution depends sensitively on the selection of parameters and
different potentials in the Schrödinger equation. Despite this complexity,
the machine learning community has developed remarkable efficiencies in
predicting complicated datasets by regression. Here, we consider a recurrent
neural network (RNN) approach to predict the complex propagation of nonlinear
solitons in plasmon-induced transparency metamaterial systems with applied
potentials bypassing the need for analytical and numerical approaches of a
guiding model. We demonstrate the success of this scheme on the prediction of
the propagation of the nonlinear solitons solely from a given initial condition
and potential. We prove the prominent agreement of results in simulation and
prediction by long short-term memory (LSTM) artificial neural networks. The
framework presented in this work opens up a new perspective for the application
of RNN in quantum systems and nonlinear waves using Schrödinger-type
equations, for example, the nonlinear dynamics in cold-atom systems and
nonlinear fiber optics.

    

### [[2108.01547] EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training](http://arxiv.org/abs/2108.01547)


  Although pre-trained language models have remarkably enhanced the generation
ability of dialogue systems, open-domain Chinese dialogue systems are still
limited by the dialogue data and the model size compared with English ones. In
this paper, we propose EVA, a Chinese dialogue system that contains the largest
Chinese pre-trained dialogue model with 2.8B parameters. To build this model,
we collect the largest Chinese dialogue dataset named WDC-Dialogue from various
public social media. This dataset contains 1.4B context-response pairs and is
used as the pre-training corpus of EVA. Extensive experiments on automatic and
human evaluation show that EVA outperforms other Chinese pre-trained dialogue
models especially in the multi-turn interaction of human-bot conversations.

    

### [[2108.01573] Classification of Discrete Dynamical Systems Based on Transients](http://arxiv.org/abs/2108.01573)


  In order to develop systems capable of artificial evolution, we need to
identify which systems can produce complex behavior. We present a novel
classification method applicable to any class of deterministic discrete space
and time dynamical systems. The method is based on classifying the asymptotic
behavior of the average computation time in a given system before entering a
loop. We were able to identify a critical region of behavior that corresponds
to a phase transition from ordered behavior to chaos across various classes of
dynamical systems. To show that our approach can be applied to many different
computational systems, we demonstrate the results of classifying cellular
automata, Turing machines, and random Boolean networks. Further, we use this
method to classify 2D cellular automata to automatically find those with
interesting, complex dynamics.
We believe that our work can be used to design systems in which complex
structures emerge. Also, it can be used to compare various versions of existing
attempts to model open-ended evolution (Ray (1991), Ofria et al. (2004),
Channon (2006)).

    

### [[2108.01591] The application of artificial intelligence in software engineering: a review challenging conventional wisdom](http://arxiv.org/abs/2108.01591)


  The field of artificial intelligence (AI) is witnessing a recent upsurge in
research, tools development, and deployment of applications. Multiple software
companies are shifting their focus to developing intelligent systems; and many
others are deploying AI paradigms to their existing processes. In parallel, the
academic research community is injecting AI paradigms to provide solutions to
traditional engineering problems. Similarly, AI has evidently been proved
useful to software engineering (SE). When one observes the SE phases
(requirements, design, development, testing, release, and maintenance), it
becomes clear that multiple AI paradigms (such as neural networks, machine
learning, knowledge-based systems, natural language processing) could be
applied to improve the process and eliminate many of the major challenges that
the SE field has been facing. This survey chapter is a review of the most
commonplace methods of AI applied to SE. The review covers methods between
years 1975-2017, for the requirements phase, 46 major AI-driven methods are
found, 19 for design, 15 for development, 68 for testing, and 15 for release
and maintenance. Furthermore, the purpose of this chapter is threefold;
firstly, to answer the following questions: is there sufficient intelligence in
the SE lifecycle? What does applying AI to SE entail? Secondly, to measure,
formulize, and evaluate the overlap of SE phases and AI disciplines. Lastly,
this chapter aims to provide serious questions to challenging the current
conventional wisdom (i.e., status quo) of the state-of-the-art, craft a call
for action, and to redefine the path forward.

    

### [[2108.01608] Scheduling Aerial Vehicles in an Urban Air Mobility Scheme](http://arxiv.org/abs/2108.01608)


  Highly populated cities face several challenges, one of them being the
intense traffic congestion. In recent years, the concept of Urban Air Mobility
has been put forward by large companies and organizations as a way to address
this problem, and this approach has been rapidly gaining ground. This
disruptive technology involves aerial vehicles (AVs) for hire than can be
utilized by customers to travel between locations within large cities. This
concept has the potential to drastically decrease traffic congestion and reduce
air pollution, since these vehicles typically use electric motors powered by
batteries. This work studies the problem of scheduling the assignment of AVs to
customers, having as a goal to maximize the serviced customers and minimize the
energy consumption of the AVs by forcing them to fly at the lowest possible
altitude. Initially, an Integer Linear Program (ILP) formulation is presented,
that is solved offline and optimally, followed by a near-optimal algorithm,
that solves the problem incrementally, one AV at a time, to address scalability
issues, allowing scheduling in problems involving large numbers of locations,
AVs, and customer requests.

    

### [[2012.07464] Online Action Recognition](http://arxiv.org/abs/2012.07464)


  Recognition in planning seeks to find agent intentions, goals or activities
given a set of observations and a knowledge library (e.g. goal states, plans or
domain theories). In this work we introduce the problem of Online Action
Recognition. It consists in recognizing, in an open world, the planning action
that best explains a partially observable state transition from a knowledge
library of first-order STRIPS actions, which is initially empty. We frame this
as an optimization problem, and propose two algorithms to address it: Action
Unification (AU) and Online Action Recognition through Unification (OARU). The
former builds on logic unification and generalizes two input actions using
weighted partial MaxSAT. The latter looks for an action within the library that
explains an observed transition. If there is such action, it generalizes it
making use of AU, building in this way an AU hierarchy. Otherwise, OARU inserts
a Trivial Grounded Action (TGA) in the library that explains just that
transition. We report results on benchmarks from the International Planning
Competition and PDDLGym, where OARU recognizes actions accurately with respect
to expert knowledge, and shows real-time performance.

    

### [[2101.10284] Reinforcement Learning Based Temporal Logic Control with Soft Constraints Using Limit-deterministic Generalized Buchi Automata](http://arxiv.org/abs/2101.10284)


  This paper studies the control synthesis of motion planning subject to
uncertainties. The uncertainties are considered in robot motion and environment
properties, giving rise to the probabilistic labeled Markov decision process
(MDP). A model-free reinforcement learning (RL) is developed to generate a
finite-memory control policy to satisfy high-level tasks expressed in linear
temporal logic (LTL) formulas. One of the novelties is to translate LTL into a
limit deterministic generalized Büchi automaton (LDGBA) and develop a
corresponding embedded LDGBA (E-LDGBA) by incorporating a tracking-frontier
function to overcome the issue of sparse accepting rewards, resulting in
improved learning performance without increasing computational complexity. Due
to potentially conflicting tasks, a relaxed product MDP is developed to allow
the agent to revise its motion plan without strictly following the desired LTL
constraints if the desired tasks can only be partially fulfilled. An expected
return composed of violation rewards and accepting rewards is developed. The
designed violation function quantifies the differences between the revised and
the desired motion planning, while the accepting rewards are designed to
enforce the satisfaction of the acceptance condition of the relaxed product
MDP. Rigorous analysis shows that any RL algorithm that optimizes the expected
return is guaranteed to find policies that, in decreasing order, can 1) satisfy
acceptance condition of relaxed product MDP and 2) reduce the violation cost
over long-term behaviors. Also, we validate the control synthesis approach via
simulation and experimental results.

    

### [[2102.08827] A Knowledge-based Approach for the Automatic Construction of Skill Graphs for Online Monitoring](http://arxiv.org/abs/2102.08827)


  Automated vehicles need to be aware of the capabilities they currently
possess. Skill graphs are directed acylic graphs in which a vehicle's
capabilities and the dependencies between these capabilities are modeled. The
skills a vehicle requires depend on the behaviors the vehicle has to perform
and the operational design domain (ODD) of the vehicle. Skill graphs were
originally proposed for online monitoring of the current capabilities of an
automated vehicle. They have also been shown to be useful during other parts of
the development process, e.g. system design, system verification. Skill graph
construction is an iterative, expert-based, manual process with little to no
guidelines. This process is, thus, prone to errors and inconsistencies
especially regarding the propagation of changes in the vehicle's intended ODD
into the skill graphs. In order to circumnavigate this problem, we propose to
formalize expert knowledge regarding skill graph construction into a knowledge
base and automate the construction process. Thus, all changes in the vehicle's
ODD are reflected in the skill graphs automatically leading to a reduction in
inconsistencies and errors in the constructed skill graphs.

    

### [[2108.00903] Extending Sticky-Datalog+/- via Finite-Position Selection Functions: Tractability, Algorithms, and Optimization](http://arxiv.org/abs/2108.00903)


  Weakly-Sticky(WS) Datalog+/- is an expressive member of the family of
Datalog+/- program classes that is defined on the basis of the conditions of
stickiness and weak-acyclicity. Conjunctive query answering (QA) over the WS
programs has been investigated, and its tractability in data complexity has
been established. However, the design and implementation of practical QA
algorithms and their optimizations have been open. In order to fill this gap,
we first study Sticky and WS programs from the point of view of the behavior of
the chase procedure. We extend the stickiness property of the chase to that of
generalized stickiness of the chase (GSCh) modulo an oracle that selects (and
provides) the predicate positions where finitely values appear during the
chase. Stickiness modulo a selection function S that provides only a subset of
those positions defines sch(S), a semantic subclass of GSCh. Program classes
with selection functions include Sticky and WS, and another syntactic class
that we introduce and characterize, namely JWS, of jointly-weakly-sticky
programs, which contains WS. The selection functions for these last three
classes are computable, and no external, possibly non-computable oracle is
needed. We propose a bottom-up QA algorithm for programs in the class sch(S),
for a general selection function S. As a particular case, we obtain a
polynomial-time QA algorithm for JWS and weakly-sticky programs. Unlike WS, JWS
turns out to be closed under magic-sets query optimization. As a consequence,
both the generic polynomial-time QA algorithm and its magic-set optimization
can be particularized and applied to WS.

    

### [[2108.01292] Energy Management in Data Centers with Server Setup Delay: A Semi-MDP Approximation](http://arxiv.org/abs/2108.01292)


  The energy management schemes in multi-server data centers with setup time
mostly consider thresholds on the number of idle servers or waiting jobs to
switch servers $\textit{on}$ or $\textit{off}$. An optimal energy management
policy can be characterized as a $\textit{Markov decision process}$ (MDP) at
large, given that the system parameters evolve Markovian. The resulting optimal
reward can be defined as the weighted sum of mean power usage and mean delay of
requested jobs. For large-scale data centers however, these models become
intractable due to the colossal state-action space, thus making conventional
algorithms inefficient in finding the optimal policy. In this paper, we propose
an approximate $\textit{semi-MDP}$ (SMDP) approach, known as
`$\textit{multi-level SMDP}$', based on state aggregation and Markovian
analysis of the system behavior. Rather than averaging the transition
probabilities of aggregated states as in typical methods, we introduce an
approximate Markovian framework for calculating the transition probabilities of
the proposed multi-level SMDP accurately. Moreover, near-optimal performance
can be attained at the expense of increased state-space dimensionality by
tuning the number of levels in the multi-level approach. Simulation results
show that the proposed approach reduces the SMDP size while yielding better
rewards as against existing fixed threshold-based policies and aggregation
methods.

    

### [[2108.01470] FIRESTARTER 2: Dynamic Code Generation for Processor Stress Tests](http://arxiv.org/abs/2108.01470)


  Processor stress tests target to maximize processor power consumption by
executing highly demanding workloads. They are typically used to test the
cooling and electrical infrastructure of compute nodes or larger systems in
labs or data centers. While multiple of these tools already exists, they have
to be re-evaluated and updated regularly to match the developments in computer
architecture. This paper presents the first major update of FIRESTARTER, an
Open Source tool specifically designed to create near-peak power consumption.
The main new features concern the online generation of workloads and automatic
self-tuning for specific hardware configurations. We further apply these new
features on an AMD Rome system and demonstrate the optimization process. Our
analysis shows how accesses to the different levels of the memory hierarchy
contribute to the overall power consumption. Finally, we demonstrate how the
auto-tuning algorithm can cope with different processor configurations and how
these influence the effectiveness of the created workload.

    

### [[2005.03459] AIBench Scenario: Scenario-distilling AI Benchmarking](http://arxiv.org/abs/2005.03459)


  Modern real-world application scenarios like Internet services consist of a
diversity of AI and non-AI modules with huge code sizes and long and
complicated execution paths, which raises serious benchmarking or evaluating
challenges. Using AI components or micro benchmarks alone can lead to
error-prone conclusions. This paper presents a methodology to attack the above
challenge. We formalize a real-world application scenario as a Directed Acyclic
Graph-based model and propose the rules to distill it into a permutation of
essential AI and non-AI tasks, which we call a scenario benchmark. Together
with seventeen industry partners, we extract nine typical scenario benchmarks.
We design and implement an extensible, configurable, and flexible benchmark
framework. We implement two Internet service AI scenario benchmarks based on
the framework as proxies to two real-world application scenarios. We consider
scenario, component, and micro benchmarks as three indispensable parts for
evaluating. Our evaluation shows the advantage of our methodology against using
component or micro AI benchmarks alone. The specifications, source code,
testbed, and results are publicly available from
\url{this https URL}.

    

### [[2108.01418] Owicki-Gries Reasoning for C11 Programs with Relaxed Dependencies (Extended Version)](http://arxiv.org/abs/2108.01418)


  Deductive verification techniques for C11 programs have advanced
significantly in recent years with the development of operational semantics and
associated logics for increasingly large fragments of C11. However, these
semantics and logics have been developed in a restricted setting to avoid the
thin-air-read problem. In this paper, we propose an operational semantics that
leverages an intra-thread partial order (called semantic dependencies) induced
by a recently developed denotational event-structure-based semantics. We prove
that our operational semantics is sound and complete with respect to the
denotational semantics. We present an associated logic that generalises a
recent Owicki-Gries framework for RC11 (repaired C11), and demonstrate the use
of this logic over several example proofs.

    

### [[2108.01610] Towards Substructural Property-Based Testing](http://arxiv.org/abs/2108.01610)


  We propose to extend property-based testing to substructural logics to
overcome the current lack of reasoning tools in the field. We take the first
step by implementing a property-based testing system for specifications written
in the linear logic programming language Lolli. We employ the foundational
proof certificates architecture to model various data generation strategies. We
validate our approach by encoding a model of a simple imperative programming
language and its compilation and by testing its meta-theory via mutation
analysis.

    