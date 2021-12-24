
## 2021-12-24

### [[2112.12321] Physics Constrained Flow Neural Network for Short-Timescale Predictions in Data Communications Networks](http://arxiv.org/abs/2112.12321)


  Machine learning is gaining growing momentum in various recent models for the
dynamic analysis of information flows in data communications networks. These
preliminary models often rely on off-the-shelf learning models to predict from
historical statistics while disregarding the physics governing the generating
behaviors of these flows. This paper instead introduces Flow Neural Network
(FlowNN) to improve the feature representation with learned physical bias. This
is implemented by an induction layer, working upon the embedding layer, to
impose the physics connected data correlations, and a self-supervised learning
strategy with stop-gradient to make the learned physics universal. For the
short-timescale network prediction tasks, FlowNN achieves 17% - 71% of loss
decrease than the state-of-the-art baselines on both synthetic and real-world
networking datasets, which shows the strength of this new approach. Code will
be made available.

    

### [[2112.12388] Reservoir: Named Data for Pervasive Computation Reuse at the Network Edge](http://arxiv.org/abs/2112.12388)


  In edge computing use cases (e.g., smart cities), where several users and
devices may be in close proximity to each other, computational tasks with
similar input data for the same services (e.g., image or video annotation) may
be offloaded to the edge. The execution of such tasks often yields the same
results (output) and thus duplicate (redundant) computation. Based on this
observation, prior work has advocated for "computation reuse", a paradigm where
the results of previously executed tasks are stored at the edge and are reused
to satisfy incoming tasks with similar input data, instead of executing these
incoming tasks from scratch. However, realizing computation reuse in practical
edge computing deployments, where services may be offered by multiple
(distributed) edge nodes (servers) for scalability and fault tolerance, is
still largely unexplored. To tackle this challenge, in this paper, we present
Reservoir, a framework to enable pervasive computation reuse at the edge, while
imposing marginal overheads on user devices and the operation of the edge
network infrastructure. Reservoir takes advantage of Locality Sensitive Hashing
(LSH) and runs on top of Named-Data Networking (NDN), extending the NDN
architecture for the realization of the computation reuse semantics in the
network. Our evaluation demonstrated that Reservoir can reuse computation with
up to an almost perfect accuracy, achieving 4.25-21.34x lower task completion
times compared to cases without computation reuse.

    

### [[2112.12478] Hierarchical Multi-Building And Multi-Floor Indoor Localization Based On Recurrent Neural Networks](http://arxiv.org/abs/2112.12478)


  There has been an increasing tendency to move from outdoor to indoor
lifestyle in modern cities. The emergence of big shopping malls, indoor sports
complexes, factories, and warehouses is accelerating this tendency. In such an
environment, indoor localization becomes one of the essential services, and the
indoor localization systems to be deployed should be scalable enough to cover
the expected expansion of those indoor facilities. One of the most economical
and practical approaches to indoor localization is Wi-Fi fingerprinting, which
exploits the widely-deployed Wi-Fi networks using mobile devices (e.g.,
smartphones) without any modification of the existing infrastructure.
Traditional Wi-Fi fingerprinting schemes rely on complicated data
pre/post-processing and time-consuming manual parameter tuning. In this paper,
we propose hierarchical multi-building and multi-floor indoor localization
based on a recurrent neural network (RNN) using Wi-Fi fingerprinting,
eliminating the need of complicated data pre/post-processing and with less
parameter tuning. The RNN in the proposed scheme estimates locations in a
sequential manner from a general to a specific one (e.g.,
building->floor->location) in order to exploit the hierarchical nature of the
localization in multi-building and multi-floor environments. The experimental
results with the UJIIndoorLoc dataset demonstrate that the proposed scheme
estimates building and floor with 100% and 95.24% accuracy, respectively, and
provides three-dimensional positioning error of 8.62 m, which outperforms
existing deep neural network-based schemes.

    

### [[2112.12546] Collaborative adversary nodes learning on the logs of IoT devices in an IoT network](http://arxiv.org/abs/2112.12546)


  Artificial Intelligence (AI) development has encouraged many new research
areas, including AI-enabled Internet of Things (IoT) network. AI analytics and
intelligent paradigms greatly improve learning efficiency and accuracy.
Applying these learning paradigms to network scenarios provide technical
advantages of new networking solutions. In this paper, we propose an improved
approach for IoT security from data perspective. The network traffic of IoT
devices can be analyzed using AI techniques. The Adversary Learning (AdLIoTLog)
model is proposed using Recurrent Neural Network (RNN) with attention mechanism
on sequences of network events in the network traffic. We define network events
as a sequence of the time series packets of protocols captured in the log. We
have considered different packets TCP packets, UDP packets, and HTTP packets in
the network log to make the algorithm robust. The distributed IoT devices can
collaborate to cripple our world which is extending to Internet of
Intelligence. The time series packets are converted into structured data by
removing noise and adding timestamps. The resulting data set is trained by RNN
and can detect the node pairs collaborating with each other. We used the BLEU
score to evaluate the model performance. Our results show that the predicting
performance of the AdLIoTLog model trained by our method degrades by 3-4% in
the presence of attack in comparison to the scenario when the network is not
under attack. AdLIoTLog can detect adversaries because when adversaries are
present the model gets duped by the collaborative events and therefore predicts
the next event with a biased event rather than a benign event. We conclude that
AI can provision ubiquitous learning for the new generation of Internet of
Things.

    

### [[1901.06786] On the Capacity Region of Bipartite and Tripartite Entanglement Switching](http://arxiv.org/abs/1901.06786)


  We study a quantum entanglement distribution switch serving a set of users in
a star topology with equal-length links. The quantum switch, much like a
quantum repeater, can perform entanglement swapping to extend entanglement
across longer distances. Additionally, the switch is equipped with entanglement
switching logic, enabling it to implement switching policies to better serve
the needs of the network. In this work, the function of the switch is to create
bipartite or tripartite entangled states among users at the highest possible
rates at a fixed ratio. Using Markov chains, we model a set of randomized
switching policies. Discovering that some are better than others, we present
analytical results for the case where the switch stores one qubit per user, and
find that the best policies outperform a time division multiplexing (TDM)
policy for sharing the switch between bipartite and tripartite state
generation. This performance improvement decreases as the number of users
grows. The model is easily augmented to study the capacity region in the
presence of quantum state decoherence and associated cut-off times for qubit
storage, obtaining similar results. Moreover, decoherence-associated quantum
storage cut-off times appear to have little effect on capacity in our
identical-link system. We also study a smaller class of policies when the
switch stores two qubits per user.

    

### [[2112.12165] The Universal $\ell^p$-Metric on Merge Trees](http://arxiv.org/abs/2112.12165)


  Adapting a definition given by Bjerkevik and Lesnick for multiparameter
persistence modules, we introduce an $\ell^p$-type extension of the
interleaving distance on merge trees. We show that our distance is a metric,
and that it upper-bounds the $p$-Wasserstein distance between the associated
barcodes. For each $p\in[1,\infty]$, we prove that this distance is stable with
respect to cellular sublevel filtrations and that it is the universal (i.e.,
largest) distance satisfying this stability property. In the $p=\infty$ case,
this gives a novel proof of universality for the interleaving distance on merge
trees.

    

### [[2112.12181] Simple and near-optimal algorithms for hidden stratification and multi-group learning](http://arxiv.org/abs/2112.12181)


  Multi-group agnostic learning is a formal learning criterion that is
concerned with the conditional risks of predictors within subgroups of a
population. The criterion addresses recent practical concerns such as subgroup
fairness and hidden stratification. This paper studies the structure of
solutions to the multi-group learning problem, and provides simple and
near-optimal algorithms for the learning problem.

    

### [[2112.12194] Surrogate Likelihoods for Variational Annealed Importance Sampling](http://arxiv.org/abs/2112.12194)


  Variational inference is a powerful paradigm for approximate Bayesian
inference with a number of appealing properties, including support for model
learning and data subsampling. By contrast MCMC methods like Hamiltonian Monte
Carlo do not share these properties but remain attractive since, contrary to
parametric methods, MCMC is asymptotically unbiased. For these reasons
researchers have sought to combine the strengths of both classes of algorithms,
with recent approaches coming closer to realizing this vision in practice.
However, supporting data subsampling in these hybrid methods can be a
challenge, a shortcoming that we address by introducing a surrogate likelihood
that can be learned jointly with other variational parameters. We argue
theoretically that the resulting algorithm permits the user to make an
intuitive trade-off between inference fidelity and computational cost. In an
extensive empirical comparison we show that our method performs well in
practice and that it is well-suited for black-box inference in probabilistic
programming frameworks.

    

### [[2112.12210] ProBF: Learning Probabilistic Safety Certificates with Barrier Functions](http://arxiv.org/abs/2112.12210)


  Safety-critical applications require controllers/policies that can guarantee
safety with high confidence. The control barrier function is a useful tool to
guarantee safety if we have access to the ground-truth system dynamics. In
practice, we have inaccurate knowledge of the system dynamics, which can lead
to unsafe behaviors due to unmodeled residual dynamics. Learning the residual
dynamics with deterministic machine learning models can prevent the unsafe
behavior but can fail when the predictions are imperfect. In this situation, a
probabilistic learning method that reasons about the uncertainty of its
predictions can help provide robust safety margins. In this work, we use a
Gaussian process to model the projection of the residual dynamics onto a
control barrier function. We propose a novel optimization procedure to generate
safe controls that can guarantee safety with high probability. The safety
filter is provided with the ability to reason about the uncertainty of the
predictions from the GP. We show the efficacy of this method through
experiments on Segway and Quadrotor simulations. Our proposed probabilistic
approach is able to reduce the number of safety violations significantly as
compared to the deterministic approach with a neural network.

    

### [[2112.12218] Maximum Entropy on Erroneous Predictions (MEEP): Improving model calibration for medical image segmentation](http://arxiv.org/abs/2112.12218)


  Modern deep neural networks have achieved remarkable progress in medical
image segmentation tasks. However, it has recently been observed that they tend
to produce overconfident estimates, even in situations of high uncertainty,
leading to poorly calibrated and unreliable models. In this work we introduce
Maximum Entropy on Erroneous Predictions (MEEP), a training strategy for
segmentation networks which selectively penalizes overconfident predictions,
focusing only on misclassified pixels. In particular, we design a
regularization term that encourages high entropy posteriors for wrong
predictions, increasing the network uncertainty in complex scenarios. Our
method is agnostic to the neural architecture, does not increase model
complexity and can be coupled with multiple segmentation loss functions. We
benchmark the proposed strategy in two challenging medical image segmentation
tasks: white matter hyperintensity lesions in magnetic resonance images (MRI)
of the brain, and atrial segmentation in cardiac MRI. The experimental results
demonstrate that coupling MEEP with standard segmentation losses leads to
improvements not only in terms of model calibration, but also in segmentation
quality.

    

### [[2112.12219] MC-DGCNN: A Novel DNN Architecture for Multi-Category Point Set Classification](http://arxiv.org/abs/2112.12219)


  Point set classification aims to build a representation learning model that
distinguishes between spatial and categorical configurations of point set data.
This problem is societally important since in many applications domains such as
immunology, and microbial ecology. This problem is challenging since the
interactions between different categories of points are not always equal; as a
result, the representation learning model must selectively learn the most
relevant multi-categorical relationships. The related works are limited (1) in
learning the importance of different multi-categorical relationships,
especially for high-order interactions, and (2) do not fully exploit the
spatial distribution of points beyond simply measuring relative distance or
applying a feed-forward neural network to coordinates. To overcome these
limitations, we leverage the dynamic graph convolutional neural network (DGCNN)
architecture to design a novel multi-category DGCNN (MC-DGCNN), contributing
location representation and point pair attention layers for multi-categorical
point set classification. MC-DGCNN has the ability to identify the categorical
importance of each point pair and extends this to N-way spatial relationships,
while still preserving all the properties and benefits of DGCNN (e.g.,
differentiability). Experimental results show that the proposed architecture is
computationally efficient and significantly outperforms current deep learning
architectures on real-world datasets.

    

### [[2112.12228] Direct Behavior Specification via Constrained Reinforcement Learning](http://arxiv.org/abs/2112.12228)


  The standard formulation of Reinforcement Learning lacks a practical way of
specifying what are admissible and forbidden behaviors. Most often,
practitioners go about the task of behavior specification by manually
engineering the reward function, a counter-intuitive process that requires
several iterations and is prone to reward hacking by the agent. In this work,
we argue that constrained RL, which has almost exclusively been used for safe
RL, also has the potential to significantly reduce the amount of work spent for
reward specification in applied Reinforcement Learning projects. To this end,
we propose to specify behavioral preferences in the CMDP framework and to use
Lagrangian methods, which seek to solve a min-max problem between the agent's
policy and the Lagrangian multipliers, to automatically weigh each of the
behavioral constraints. Specifically, we investigate how CMDPs can be adapted
in order to solve goal-based tasks while adhering to a set of behavioral
constraints and propose modifications to the SAC-Lagrangian algorithm to handle
the challenging case of several constraints. We evaluate this framework on a
set of continuous control tasks relevant to the application of Reinforcement
Learning for NPC design in video games.

    

### [[2112.12245] Combinations of Adaptive Filters](http://arxiv.org/abs/2112.12245)


  Adaptive filters are at the core of many signal processing applications,
ranging from acoustic noise supression to echo cancelation, array beamforming,
channel equalization, to more recent sensor network applications in
surveillance, target localization, and tracking. A trending approach in this
direction is to recur to in-network distributed processing in which individual
nodes implement adaptation rules and diffuse their estimation to the network.
When the a priori knowledge about the filtering scenario is limited or
imprecise, selecting the most adequate filter structure and adjusting its
parameters becomes a challenging task, and erroneous choices can lead to
inadequate performance. To address this difficulty, one useful approach is to
rely on combinations of adaptive structures.
The combination of adaptive filters exploits to some extent the same divide
and conquer principle that has also been successfully exploited by the
machine-learning community (e.g., in bagging or boosting). In particular, the
problem of combining the outputs of several learning algorithms (mixture of
experts) has been studied in the computational learning field under a different
perspective: rather than studying the expected performance of the mixture,
deterministic bounds are derived that apply to individual sequences and,
therefore, reflect worst-case scenarios. These bounds require assumptions
different from the ones typically used in adaptive filtering, which is the
emphasis of this overview article. We review the key ideas and principles
behind these combination schemes, with emphasis on design rules. We also
illustrate their performance with a variety of examples.

    

### [[2112.12249] Regularized Multivariate Analysis Framework for Interpretable High-Dimensional Variable Selection](http://arxiv.org/abs/2112.12249)


  Multivariate Analysis (MVA) comprises a family of well-known methods for
feature extraction which exploit correlations among input variables
representing the data. One important property that is enjoyed by most such
methods is uncorrelation among the extracted features. Recently, regularized
versions of MVA methods have appeared in the literature, mainly with the goal
to gain interpretability of the solution. In these cases, the solutions can no
longer be obtained in a closed manner, and more complex optimization methods
that rely on the iteration of two steps are frequently used. This paper recurs
to an alternative approach to solve efficiently this iterative problem. The
main novelty of this approach lies in preserving several properties of the
original methods, most notably the uncorrelation of the extracted features.
Under this framework, we propose a novel method that takes advantage of the
l-21 norm to perform variable selection during the feature extraction process.
Experimental results over different problems corroborate the advantages of the
proposed formulation in comparison to state of the art formulations.

    

### [[2112.12251] ML4CO: Is GCNN All You Need? Graph Convolutional Neural Networks Produce Strong Baselines For Combinatorial Optimization Problems, If Tuned and Trained Properly, on Appropriate Data](http://arxiv.org/abs/2112.12251)


  The 2021 NeurIPS Machine Learning for Combinatorial Optimization (ML4CO)
competition was designed with the goal of improving state-of-the-art
combinatorial optimization solvers by replacing key heuristic components with
machine learning models. The competition's main scientific question was the
following: is machine learning a viable option for improving traditional
combinatorial optimization solvers on specific problem distributions, when
historical data is available? This was motivated by the fact that in many
practical scenarios, the data changes only slightly between the repetitions of
a combinatorial optimization problem, and this is an area where machine
learning models are particularly powerful at. This paper summarizes the
solution and lessons learned by the Huawei EI-OROAS team in the dual task of
the competition. The submission of our team achieved the second place in the
final ranking, with a very close distance to the first spot. In addition, our
solution was ranked first consistently for several weekly leaderboard updates
before the final evaluation. We provide insights gained from a large number of
experiments, and argue that a simple Graph Convolutional Neural Network (GCNNs)
can achieve state-of-the-art results if trained and tuned properly.

    

### [[2112.12262] Morphological classifiers](http://arxiv.org/abs/2112.12262)


  This work proposes a new type of classifier called Morphological Classifier
(MC). MCs aggregate concepts from mathematical morphology and supervised
learning. The outcomes of this aggregation are classifiers that may preserve
shape characteristics of classes, subject to the choice of a stopping criterion
and structuring element. MCs are fundamentally based on set theory, and their
classification model can be a mathematical set itself. Two types of
morphological classifiers are proposed in the current work, namely,
Morphological k-NN (MkNN) and Morphological Dilation Classifier (MDC), which
demonstrate the feasibility of the approach. This work provides evidence
regarding the advantages of MCs, e.g., very fast classification times as well
as competitive accuracy rates. The performance of MkNN and MDC was tested using
p -dimensional datasets. MCs tied or outperformed 14 well established
classifiers in 5 out of 8 datasets. In all occasions, the obtained accuracies
were higher than the average accuracy obtained with all classifiers. Moreover,
the proposed implementations utilize the power of the Graphics Processing Units
(GPUs) to speed up processing.

    

### [[2112.12263] Crash Data Augmentation Using Conditional Generative Adversarial Networks (CGAN) for Improving Safety Performance Functions](http://arxiv.org/abs/2112.12263)


  In this paper, we present a crash frequency data augmentation method based on
Conditional Generative Adversarial Networks to improve crash frequency models.
The proposed method is evaluated by comparing the performance of Base SPFs
(developed using original data) and Augmented SPFs (developed using original
data plus synthesised data) in terms of hotspot identification performance,
model prediction accuracy, and dispersion parameter estimation accuracy. The
experiments are conducted using simulated and real-world crash data sets. The
results indicate that the synthesised crash data by CGAN have the same
distribution as the original data and the Augmented SPFs outperforms Base SPFs
in almost all aspects especially when the dispersion parameter is low.

    

### [[2112.12272] Human Activity Recognition on wrist-worn accelerometers using self-supervised neural networks](http://arxiv.org/abs/2112.12272)


  Measures of Activity of Daily Living (ADL) are an important indicator of
overall health but difficult to measure in-clinic. Automated and accurate human
activity recognition (HAR) using wrist-worn accelerometers enables practical
and cost efficient remote monitoring of ADL. Key obstacles in developing high
quality HAR is the lack of large labeled datasets and the performance loss when
applying models trained on small curated datasets to the continuous stream of
heterogeneous data in real-life. In this work we design a self-supervised
learning paradigm to create a robust representation of accelerometer data that
can generalize across devices and subjects. We demonstrate that this
representation can separate activities of daily living and achieve strong HAR
accuracy (on multiple benchmark datasets) using very few labels. We also
propose a segmentation algorithm which can identify segments of salient
activity and boost HAR accuracy on continuous real-life data.

    

### [[2112.12275] Algorithmic Probability of Large Datasets and the Simplicity Bubble Problem in Machine Learning](http://arxiv.org/abs/2112.12275)


  When mining large datasets in order to predict new data, limitations of the
principles behind statistical machine learning pose a serious challenge not
only to the Big Data deluge, but also to the traditional assumptions that data
generating processes are biased toward low algorithmic complexity. Even when
one assumes an underlying algorithmic-informational bias toward simplicity in
finite dataset generators, we show that fully automated, with or without access
to pseudo-random generators, computable learning algorithms, in particular
those of statistical nature used in current approaches to machine learning
(including deep learning), can always be deceived, naturally or artificially,
by sufficiently large datasets. In particular, we demonstrate that, for every
finite learning algorithm, there is a sufficiently large dataset size above
which the algorithmic probability of an unpredictable deceiver is an upper
bound (up to a multiplicative constant that only depends on the learning
algorithm) for the algorithmic probability of any other larger dataset. In
other words, very large and complex datasets are as likely to deceive learning
algorithms into a "simplicity bubble" as any other particular dataset. These
deceiving datasets guarantee that any prediction will diverge from the
high-algorithmic-complexity globally optimal solution while converging toward
the low-algorithmic-complexity locally optimal solution. We discuss the
framework and empirical conditions for circumventing this deceptive phenomenon,
moving away from statistical machine learning towards a stronger type of
machine learning based on, or motivated by, the intrinsic power of algorithmic
information theory and computability theory.

    

### [[2112.12280] Nonnegative OPLS for Supervised Design of Filter Banks: Application to Image and Audio Feature Extraction](http://arxiv.org/abs/2112.12280)


  Audio or visual data analysis tasks usually have to deal with
high-dimensional and nonnegative signals. However, most data analysis methods
suffer from overfitting and numerical problems when data have more than a few
dimensions needing a dimensionality reduction preprocessing. Moreover,
interpretability about how and why filters work for audio or visual
applications is a desired property, especially when energy or spectral signals
are involved. In these cases, due to the nature of these signals, the
nonnegativity of the filter weights is a desired property to better understand
its working. Because of these two necessities, we propose different methods to
reduce the dimensionality of data while the nonnegativity and interpretability
of the solution are assured. In particular, we propose a generalized
methodology to design filter banks in a supervised way for applications dealing
with nonnegative data, and we explore different ways of solving the proposed
objective function consisting of a nonnegative version of the orthonormalized
partial least-squares method. We analyze the discriminative power of the
features obtained with the proposed methods for two different and widely
studied applications: texture and music genre classification. Furthermore, we
compare the filter banks achieved by our methods with other state-of-the-art
methods specifically designed for feature extraction.

    

### [[2112.12281] Improving the Efficiency of Off-Policy Reinforcement Learning by Accounting for Past Decisions](http://arxiv.org/abs/2112.12281)


  Off-policy learning from multistep returns is crucial for sample-efficient
reinforcement learning, particularly in the experience replay setting now
commonly used with deep neural networks. Classically, off-policy estimation
bias is corrected in a per-decision manner: past temporal-difference errors are
re-weighted by the instantaneous Importance Sampling (IS) ratio (via
eligibility traces) after each action. Many important off-policy algorithms
such as Tree Backup and Retrace rely on this mechanism along with differing
protocols for truncating ("cutting") the ratios ("traces") to counteract the
excessive variance of the IS estimator. Unfortunately, cutting traces on a
per-decision basis is not necessarily efficient; once a trace has been cut
according to local information, the effect cannot be reversed later,
potentially resulting in the premature truncation of estimated returns and
slower learning. In the interest of motivating efficient off-policy algorithms,
we propose a multistep operator that permits arbitrary past-dependent traces.
We prove that our operator is convergent for policy evaluation, and for optimal
control when targeting greedy-in-the-limit policies. Our theorems establish the
first convergence guarantees for many existing algorithms including Truncated
IS, Non-Markov Retrace, and history-dependent TD($\lambda$). Our theoretical
results also provide guidance for the development of new algorithms that
jointly consider multiple past decisions for better credit assignment and
faster learning.

    

### [[2112.12288] Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning](http://arxiv.org/abs/2112.12288)


  Reach-avoid optimal control problems, in which the system must reach certain
goal conditions while staying clear of unacceptable failure modes, are central
to safety and liveness assurance for autonomous robotic systems, but their
exact solutions are intractable for complex dynamics and environments. Recent
successes in reinforcement learning methods to approximately solve optimal
control problems with performance objectives make their application to
certification problems attractive; however, the Lagrange-type objective used in
reinforcement learning is not suitable to encode temporal logic requirements.
Recent work has shown promise in extending the reinforcement learning machinery
to safety-type problems, whose objective is not a sum, but a minimum (or
maximum) over time. In this work, we generalize the reinforcement learning
formulation to handle all optimal control problems in the reach-avoid category.
We derive a time-discounted reach-avoid Bellman backup with contraction mapping
properties and prove that the resulting reach-avoid Q-learning algorithm
converges under analogous conditions to the traditional Lagrange-type problem,
yielding an arbitrarily tight conservative approximation to the reach-avoid
set. We further demonstrate the use of this formulation with deep reinforcement
learning methods, retaining zero-violation guarantees by treating the
approximate solutions as untrusted oracles in a model-predictive supervisory
control framework. We evaluate our proposed framework on a range of nonlinear
systems, validating the results against analytic and numerical solutions, and
through Monte Carlo simulation in previously intractable problems. Our results
open the door to a range of learning-based methods for safe-and-live autonomous
behavior, with applications across robotics and automation. See
this https URL for code and supplementary
material.

    

### [[2112.12297] Batch Processing and Data Streaming Fourier-based Convolutional Neural Network Accelerator](http://arxiv.org/abs/2112.12297)


  Decision-making by artificial neural networks with minimal latency is
paramount for numerous applications such as navigation, tracking, and real-time
machine action systems. This requires the machine learning hardware to handle
multidimensional data with a high throughput. Processing convolution operations
being the major computational tool for data classification tasks,
unfortunately, follows a challenging run-time complexity scaling law. However,
implementing the convolution theorem homomorphically in a Fourier-optic
display-light-processor enables a non-iterative O(1) runtime complexity for
data inputs beyond 1,000 x 1,000 large matrices. Following this approach, here
we demonstrate data streaming multi-kernel image batch-processing with a
Fourier Convolutional Neural Network (FCNN) accelerator. We show image batch
processing of large-scale matrices as passive 2-million dot-product
multiplications performed by digital light-processing modules in the Fourier
domain. In addition, we parallelize this optical FCNN system further by
utilizing multiple spatio-parallel diffraction orders, thus achieving a
98-times throughput improvement over state-of-art FCNN accelerators. The
comprehensive discussion of the practical challenges related to working on the
edge of the system's capabilities highlights issues of crosstalk in the Fourier
domain and resolution scaling laws. Accelerating convolutions by utilizing the
massive parallelism in display technology brings forth a non-van Neuman-based
machine learning acceleration.

    

### [[2112.12298] Analysis of ECG data to detect Atrial Fibrillation](http://arxiv.org/abs/2112.12298)


  Atrial fibrillation(termed as AF/Afib henceforth) is a discrete and often
rapid heart rhythm that can lead to clots near the heart. We can detect Afib by
ECG signal by the absence of p and inconsistent intervals between R waves as
shown in fig(1). Existing methods revolve around CNN that are used to detect
afib but most of them work with 12 point lead ECG data where in our case the
health gauge watch deals with single-point ECG data. Twelve-point lead ECG data
is more accurate than a single point. Furthermore, the health gauge watch data
is much noisier. Implementing a model to detect Afib for the watch is a test of
how the CNN is changed/modified to work with real life data

    

### [[2112.12299] A Robust Initialization of Residual Blocks for Effective ResNet Training without Batch Normalization](http://arxiv.org/abs/2112.12299)


  Batch Normalization is an essential component of all state-of-the-art neural
networks architectures. However, since it introduces many practical issues,
much recent research has been devoted to designing normalization-free
architectures. In this paper, we show that weights initialization is key to
train ResNet-like normalization-free networks. In particular, we propose a
slight modification to the summation operation of a block output to the skip
connection branch, so that the whole network is correctly initialized. We show
that this modified architecture achieves competitive results on CIFAR-10
without further regularization nor algorithmic modifications.

    

### [[2112.12303] Learning with Proper Partial Labels](http://arxiv.org/abs/2112.12303)


  Partial-label learning is a kind of weakly-supervised learning with inexact
labels, where for each training example, we are given a set of candidate labels
instead of only one true label. Recently, various approaches on partial-label
learning have been proposed under different generation models of candidate
label sets. However, these methods require relatively strong distributional
assumptions on the generation models. When the assumptions do not hold, the
performance of the methods is not guaranteed theoretically. In this paper, we
propose the notion of properness on partial labels. We show that this proper
partial-label learning framework includes many previous partial-label learning
settings as special cases. We then derive a unified unbiased estimator of the
classification risk. We prove that our estimator is risk-consistent by
obtaining its estimation error bound. Finally, we validate the effectiveness of
our algorithm through experiments.

    

### [[2112.12306] Selective Multiple Power Iteration: from Tensor PCA to gradient-based exploration of landscapes](http://arxiv.org/abs/2112.12306)


  We propose Selective Multiple Power Iterations (SMPI), a new algorithm to
address the important Tensor PCA problem that consists in recovering a spike
$\bf{v_0}^{\otimes k}$ corrupted by a Gaussian noise tensor $\bf{Z} \in
(\mathbb{R}^n)^{\otimes k}$ such that $\bf{T}=\sqrt{n} \beta \bf{v_0}^{\otimes
k} + \bf{Z}$ where $\beta$ is the signal-to-noise ratio (SNR). SMPI consists in
generating a polynomial number of random initializations, performing a
polynomial number of symmetrized tensor power iterations on each
initialization, then selecting the one that maximizes $\langle \bf{T},
\bf{v}^{\otimes k} \rangle$. Various numerical simulations for $k=3$ in the
conventionally considered range $n \leq 1000$ show that the experimental
performances of SMPI improve drastically upon existent algorithms and becomes
comparable to the theoretical optimal recovery. We show that these unexpected
performances are due to a powerful mechanism in which the noise plays a key
role for the signal recovery and that takes place at low $\beta$. Furthermore,
this mechanism results from five essential features of SMPI that distinguish it
from previous algorithms based on power iteration. These remarkable results may
have strong impact on both practical and theoretical applications of Tensor
PCA. (i) We provide a variant of this algorithm to tackle low-rank CP tensor
decomposition. These proposed algorithms also outperforms existent methods even
on real data which shows a huge potential impact for practical applications.
(ii) We present new theoretical insights on the behavior of SMPI and gradient
descent methods for the optimization in high-dimensional non-convex landscapes
that are present in various machine learning problems. (iii) We expect that
these results may help the discussion concerning the existence of the
conjectured statistical-algorithmic gap.

    

### [[2112.12320] Model Selection in Batch Policy Optimization](http://arxiv.org/abs/2112.12320)


  We study the problem of model selection in batch policy optimization: given a
fixed, partial-feedback dataset and $M$ model classes, learn a policy with
performance that is competitive with the policy derived from the best model
class. We formalize the problem in the contextual bandit setting with linear
model classes by identifying three sources of error that any model selection
algorithm should optimally trade-off in order to be competitive: (1)
approximation error, (2) statistical complexity, and (3) coverage. The first
two sources are common in model selection for supervised learning, where
optimally trading-off these properties is well-studied. In contrast, the third
source is unique to batch policy optimization and is due to dataset shift
inherent to the setting. We first show that no batch policy optimization
algorithm can achieve a guarantee addressing all three simultaneously,
revealing a stark contrast between difficulties in batch policy optimization
and the positive results available in supervised learning. Despite this
negative result, we show that relaxing any one of the three error sources
enables the design of algorithms achieving near-oracle inequalities for the
remaining two. We conclude with experiments demonstrating the efficacy of these
algorithms.

    

### [[2112.12340] Learning with distributional inverters](http://arxiv.org/abs/2112.12340)


  We generalize the "indirect learning" technique of Furst et. al., 1991 to
reduce from learning a concept class over a samplable distribution $\mu$ to
learning the same concept class over the uniform distribution. The reduction
succeeds when the sampler for $\mu$ is both contained in the target concept
class and efficiently invertible in the sense of Impagliazzo & Luby, 1989. We
give two applications.
- We show that AC0[q] is learnable over any succinctly-described product
distribution. AC0[q] is the class of constant-depth Boolean circuits of
polynomial size with AND, OR, NOT, and counting modulo $q$ gates of unbounded
fanins. Our algorithm runs in randomized quasi-polynomial time and uses
membership queries.
- If there is a strongly useful natural property in the sense of Razborov &
Rudich 1997 -- an efficient algorithm that can distinguish between random
strings and strings of non-trivial circuit complexity -- then general
polynomial-sized Boolean circuits are learnable over any efficiently samplable
distribution in randomized polynomial time, given membership queries to the
target function

    

### [[2112.12345] Revisiting Transformation Invariant Geometric Deep Learning: Are Initial Representations All You Need?](http://arxiv.org/abs/2112.12345)


  Geometric deep learning, i.e., designing neural networks to handle the
ubiquitous geometric data such as point clouds and graphs, have achieved great
successes in the last decade. One critical inductive bias is that the model can
maintain invariance towards various transformations such as translation,
rotation, and scaling. The existing graph neural network (GNN) approaches can
only maintain permutation-invariance, failing to guarantee invariance with
respect to other transformations. Besides GNNs, other works design
sophisticated transformation-invariant layers, which are computationally
expensive and difficult to be extended. To solve this problem, we revisit why
the existing neural networks cannot maintain transformation invariance when
handling geometric data. Our findings show that transformation-invariant and
distance-preserving initial representations are sufficient to achieve
transformation invariance rather than needing sophisticated neural layer
designs. Motivated by these findings, we propose Transformation Invariant
Neural Networks (TinvNN), a straightforward and general framework for geometric
data. Specifically, we realize transformation-invariant and distance-preserving
initial point representations by modifying multi-dimensional scaling before
feeding the representations into neural networks. We prove that TinvNN can
strictly guarantee transformation invariance, being general and flexible enough
to be combined with the existing neural networks. Extensive experimental
results on point cloud analysis and combinatorial optimization demonstrate the
effectiveness and general applicability of our proposed method. Based on the
experimental results, we advocate that TinvNN should be considered a new
starting point and an essential baseline for further studies of
transformation-invariant geometric deep learning.

    

### [[2112.12346] Statistical Feature-based Personal Information Detection in Mobile Network Traffic](http://arxiv.org/abs/2112.12346)


  With the popularity of smartphones, mobile applications (apps) have
penetrated the daily life of people. Although apps provide rich
functionalities, they also access a large amount of personal information
simultaneously. As a result, privacy concerns are raised. To understand what
personal information the apps collect, many solutions are presented to detect
privacy leaks in apps. Recently, the traffic monitoring-based privacy leak
detection method has shown promising performance and strong scalability.
However, it still has some shortcomings. Firstly, it suffers from detecting the
leakage of personal information with obfuscation. Secondly, it cannot discover
the privacy leaks of undefined type. Aiming at solving the above problems, a
new personal information detection method based on traffic monitoring is
proposed in this paper. In this paper, statistical features of personal
information are designed to depict the occurrence patterns of personal
information in the traffic, including local patterns and global patterns. Then
a detector is trained based on machine learning algorithms to discover
potential personal information with similar patterns. Since the statistical
features are independent of the value and type of personal information, the
trained detector is capable of identifying various types of privacy leaks and
obfuscated privacy leaks. As far as we know, this is the first work that
detects personal information based on statistical features. Finally, the
experimental results show that the proposed method could achieve better
performance than the state-of-the-art.

    

### [[2112.12353] LAME: Layout Aware Metadata Extraction Approach for Research Articles](http://arxiv.org/abs/2112.12353)


  The volume of academic literature, such as academic conference papers and
journals, has increased rapidly worldwide, and research on metadata extraction
is ongoing. However, high-performing metadata extraction is still challenging
due to diverse layout formats according to journal publishers. To accommodate
the diversity of the layouts of academic journals, we propose a novel
LAyout-aware Metadata Extraction (LAME) framework equipped with the three
characteristics (e.g., design of an automatic layout analysis, construction of
a large meta-data training set, and construction of Layout-MetaBERT). We
designed an automatic layout analysis using PDFMiner. Based on the layout
analysis, a large volume of metadata-separated training data, including the
title, abstract, author name, author affiliated organization, and keywords,
were automatically extracted. Moreover, we constructed Layout-MetaBERT to
extract the metadata from academic journals with varying layout formats. The
experimental results with Layout-MetaBERT exhibited robust performance
(Macro-F1, 93.27%) in metadata extraction for unseen journals with different
layout formats.

    

### [[2112.12371] A Practical Data-Free Approach to One-shot Federated Learning with Heterogeneity](http://arxiv.org/abs/2112.12371)


  One-shot Federated Learning (FL) has recently emerged as a promising
approach, which allows the central server to learn a model in a single
communication round. Despite the low communication cost, existing one-shot FL
methods are mostly impractical or face inherent limitations, e.g., a public
dataset is required, clients' models are homogeneous, need to upload additional
data/model information. To overcome these issues, we propose a more practical
data-free approach named FedSyn for one-shot FL framework with heterogeneity.
Our FedSyn trains the global model by a data generation stage and a model
distillation stage. To the best of our knowledge, FedSyn is the first method
that can be practically applied to various real-world applications due to the
following advantages: (1) FedSyn requires no additional information (except the
model parameters) to be transferred between clients and the server; (2) FedSyn
does not require any auxiliary dataset for training; (3) FedSyn is the first to
consider both model and statistical heterogeneities in FL, i.e., the clients'
data are non-iid and different clients may have different model architectures.
Experiments on a variety of real-world datasets demonstrate the superiority of
our FedSyn. For example, FedSyn outperforms the best baseline method Fed-ADI by
5.08% on CIFAR10 dataset when data are non-iid.

    

### [[2112.12373] Decentralized Multi-Task Stochastic Optimization With Compressed Communications](http://arxiv.org/abs/2112.12373)


  We consider a multi-agent network where each node has a stochastic (local)
cost function that depends on the decision variable of that node and a random
variable, and further the decision variables of neighboring nodes are pairwise
constrained. There is an aggregate objective function for the network, composed
additively of the expected values of the local cost functions at the nodes, and
the overall goal of the network is to obtain the minimizing solution to this
aggregate objective function subject to all the pairwise constraints. This is
to be achieved at the node level using decentralized information and local
computation, with exchanges of only compressed information allowed by
neighboring nodes. The paper develops algorithms and obtains performance bounds
for two different models of local information availability at the nodes: (i)
sample feedback, where each node has direct access to samples of the local
random variable to evaluate its local cost, and (ii) bandit feedback, where
samples of the random variables are not available, but only the values of the
local cost functions at two random points close to the decision are available
to each node. For both models, with compressed communication between neighbors,
we have developed decentralized saddle-point algorithms that deliver
performances no different (in order sense) from those without communication
compression; specifically, we show that deviation from the global minimum value
and violations of the constraints are upper-bounded by
$\mathcal{O}(T^{-\frac{1}{2}})$ and $\mathcal{O}(T^{-\frac{1}{4}})$,
respectively, where $T$ is the number of iterations. Numerical examples
provided in the paper corroborate these bounds and demonstrate the
communication efficiency of the proposed method.

    

### [[2112.12376] Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization](http://arxiv.org/abs/2112.12376)


  Adversarial training (AT) has become a widely recognized defense mechanism to
improve the robustness of deep neural networks against adversarial attacks. It
solves a min-max optimization problem, where the minimizer (i.e., defender)
seeks a robust model to minimize the worst-case training loss in the presence
of adversarial examples crafted by the maximizer (i.e., attacker). However, the
min-max nature makes AT computationally intensive and thus difficult to scale.
Meanwhile, the FAST-AT algorithm, and in fact many recent algorithms that
improve AT, simplify the min-max based AT by replacing its maximization step
with the simple one-shot gradient sign based attack generation step. Although
easy to implement, FAST-AT lacks theoretical guarantees, and its practical
performance can be unsatisfactory, suffering from the robustness catastrophic
overfitting when training with strong adversaries.
In this paper, we propose to design FAST-AT from the perspective of bi-level
optimization (BLO). We first make the key observation that the most
commonly-used algorithmic specification of FAST-AT is equivalent to using some
gradient descent-type algorithm to solve a bi-level problem involving a sign
operation. However, the discrete nature of the sign operation makes it
difficult to understand the algorithm performance. Based on the above
observation, we propose a new tractable bi-level optimization problem, design
and analyze a new set of algorithms termed Fast Bi-level AT (FAST-BAT).
FAST-BAT is capable of defending sign-based projected gradient descent (PGD)
attacks without calling any gradient sign method and explicit robust
regularization. Furthermore, we empirically show that our method outperforms
state-of-the-art FAST-AT baselines, by achieving superior model robustness
without inducing robustness catastrophic overfitting, or suffering from any
loss of standard accuracy.

    

### [[2112.12411] Mitigating Leakage from Data Dependent Communications in Decentralized Computing using Differential Privacy](http://arxiv.org/abs/2112.12411)


  Imagine a group of citizens willing to collectively contribute their personal
data for the common good to produce socially useful information, resulting from
data analytics or machine learning computations. Sharing raw personal data with
a centralized server performing the computation could raise concerns about
privacy and a perceived risk of mass surveillance. Instead, citizens may trust
each other and their own devices to engage into a decentralized computation to
collaboratively produce an aggregate data release to be shared. In the context
of secure computing nodes exchanging messages over secure channels at runtime,
a key security issue is to protect against external attackers observing the
traffic, whose dependence on data may reveal personal information. Existing
solutions are designed for the cloud setting, with the goal of hiding all
properties of the underlying dataset, and do not address the specific privacy
and efficiency challenges that arise in the above context. In this paper, we
define a general execution model to control the data-dependence of
communications in user-side decentralized computations, in which differential
privacy guarantees for communication patterns in global execution plans can be
analyzed by combining guarantees obtained on local clusters of nodes. We
propose a set of algorithms which allow to trade-off between privacy, utility
and efficiency. Our formal privacy guarantees leverage and extend recent
results on privacy amplification by shuffling. We illustrate the usefulness of
our proposal on two representative examples of decentralized execution plans
with data-dependent communications.

    

### [[2112.12431] Adaptive Modeling Against Adversarial Attacks](http://arxiv.org/abs/2112.12431)


  Adversarial training, the process of training a deep learning model with
adversarial data, is one of the most successful adversarial defense methods for
deep learning models. We have found that the robustness to white-box attack of
an adversarially trained model can be further improved if we fine tune this
model in inference stage to adapt to the adversarial input, with the extra
information in it. We introduce an algorithm that "post trains" the model at
inference stage between the original output class and a "neighbor" class, with
existing training data. The accuracy of pre-trained Fast-FGSM CIFAR10
classifier base model against white-box projected gradient attack (PGD) can be
significantly improved from 46.8% to 64.5% with our algorithm.

    

### [[2112.12433] Sparse-softmax: A Simpler and Faster Alternative Softmax Transformation](http://arxiv.org/abs/2112.12433)


  The softmax function is widely used in artificial neural networks for the
multiclass classification problems, where the softmax transformation enforces
the output to be positive and sum to one, and the corresponding loss function
allows to use maximum likelihood principle to optimize the model. However,
softmax leaves a large margin for loss function to conduct optimizing operation
when it comes to high-dimensional classification, which results in
low-performance to some extent. In this paper, we provide an empirical study on
a simple and concise softmax variant, namely sparse-softmax, to alleviate the
problem that occurred in traditional softmax in terms of high-dimensional
classification problems. We evaluate our approach in several interdisciplinary
tasks, the experimental results show that sparse-softmax is simpler, faster,
and produces better results than the baseline models.

    

### [[2112.12438] Using Sequential Statistical Tests to Improve the Performance of Random Search in hyperparameter Tuning](http://arxiv.org/abs/2112.12438)


  Hyperparamter tuning is one of the the most time-consuming parts in machine
learning: The performance of a large number of different hyperparameter
settings has to be evaluated to find the best one. Although modern optimization
algorithms exist that minimize the number of evaluations needed, the evaluation
of a single setting is still expensive: Using a resampling technique, the
machine learning method has to be fitted a fixed number of $K$ times on
different training data sets. As an estimator for the performance of the
setting the respective mean value of the $K$ fits is used. Many hyperparameter
settings could be discarded after less than $K$ resampling iterations, because
they already are clearly inferior to high performing settings. However, in
practice, the resampling is often performed until the very end, wasting a lot
of computational effort.
We propose to use a sequential testing procedure to minimize the number of
resampling iterations to detect inferior parameter setting. To do so, we first
analyze the distribution of resampling errors, we will find out, that a
log-normal distribution is promising. Afterwards, we build a sequential testing
procedure assuming this distribution. This sequential test procedure is
utilized within a random search algorithm.
We compare a standard random search with our enhanced sequential random
search in some realistic data situation. It can be shown that the sequential
random search is able to find comparably good hyperparameter settings, however,
the computational time needed to find those settings is roughly halved.

    

### [[2112.12455] Your Face Mirrors Your Deepest Beliefs-Predicting Personality and Morals through Facial Emotion Recognition](http://arxiv.org/abs/2112.12455)


  Can we really "read the mind in the eyes"? Moreover, can AI assist us in this
task? This paper answers these two questions by introducing a machine learning
system that predicts personality characteristics of individuals on the basis of
their face. It does so by tracking the emotional response of the individual's
face through facial emotion recognition (FER) while watching a series of 15
short videos of different genres. To calibrate the system, we invited 85 people
to watch the videos, while their emotional responses were analyzed through
their facial expression. At the same time, these individuals also took four
well-validated surveys of personality characteristics and moral values: the
revised NEO FFI personality inventory, the Haidt moral foundations test, the
Schwartz personal value system, and the domain-specific risk-taking scale
(DOSPERT). We found that personality characteristics and moral values of an
individual can be predicted through their emotional response to the videos as
shown in their face, with an accuracy of up to 86% using gradient-boosted
trees. We also found that different personality characteristics are better
predicted by different videos, in other words, there is no single video that
will provide accurate predictions for all personality characteristics, but it
is the response to the mix of different videos that allows for accurate
prediction.

    

### [[2112.12458] Local Advantage Networks for Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2112.12458)


  Multi-agent reinforcement learning (MARL) enables us to create adaptive
agents in challenging environments, even when the agents have limited
observation. Modern MARL methods have hitherto focused on finding factorized
value functions. While this approach has proven successful, the resulting
methods have convoluted network structures. We take a radically different
approach, and build on the structure of independent Q-learners. Inspired by
influence-based abstraction, we start from the observation that compact
representations of the observation-action histories can be sufficient to learn
close to optimal decentralized policies. Combining this observation with a
dueling architecture, our algorithm, LAN, represents these policies as separate
individual advantage functions w.r.t. a centralized critic. These local
advantage networks condition only on a single agent's local observation-action
history. The centralized value function conditions on the agents'
representations as well as the full state of the environment. The value
function, which is cast aside before execution, serves as a stabilizer that
coordinates the learning and to formulate DQN targets during learning. In
contrast with other methods, this enables LAN to keep the number of network
parameters of its centralized network independent in the number of agents,
without imposing additional constraints like monotonic value functions. When
evaluated on the StarCraft multi-agent challenge benchmark, LAN shows
state-of-the-art performance and scores more than 80% wins in two previously
unsolved maps `corridor' and `3s5z_vs_3s6z', leading to an improvement of 10%
over QPLEX on average performance on the 14 maps. Moreover when the number of
agents becomes large, LAN uses significantly fewer parameters than QPLEX or
even QMIX. We thus show that LAN's structure forms a key improvement that helps
MARL methods remain scalable.

    

### [[2112.12463] Comprehensive Movie Recommendation System](http://arxiv.org/abs/2112.12463)


  A recommender system, also known as a recommendation system, is a type of
information filtering system that attempts to forecast a user's rating or
preference for an item. This article designs and implements a complete movie
recommendation system prototype based on the Genre, Pearson Correlation
Coefficient, Cosine Similarity, KNN-Based, Content-Based Filtering using TFIDF
and SVD, Collaborative Filtering using TFIDF and SVD, Surprise Library based
recommendation system technology. Apart from that in this paper, we present a
novel idea that applies machine learning techniques to construct a cluster for
the movie based on genres and then observes the inertia value number of
clusters were defined. The constraints of the approaches discussed in this work
have been described, as well as how one strategy overcomes the disadvantages of
another. The whole work has been done on the dataset Movie Lens present at the
group lens website which contains 100836 ratings and 3683 tag applications
across 9742 movies. These data were created by 610 users between March 29,
1996, and September 24, 2018.

    

### [[2112.12465] The Impact of Missing Velocity Information in Dynamic Obstacle Avoidance based on Deep Reinforcement Learning](http://arxiv.org/abs/2112.12465)


  We introduce a novel approach to dynamic obstacle avoidance based on Deep
Reinforcement Learning by defining a traffic type independent environment with
variable complexity. Filling a gap in the current literature, we thoroughly
investigate the effect of missing velocity information on an agent's
performance in obstacle avoidance tasks. This is a crucial issue in practice
since several sensors yield only positional information of objects or vehicles.
We evaluate frequently-applied approaches in scenarios of partial
observability, namely the incorporation of recurrency in the deep neural
networks and simple frame-stacking. For our analysis, we rely on
state-of-the-art model-free deep RL algorithms. The lack of velocity
information is found to significantly impact the performance of an agent. Both
approaches - recurrency and frame-stacking - cannot consistently replace
missing velocity information in the observation space. However, in simplified
scenarios, they can significantly boost performance and stabilize the overall
training procedure.

    

### [[2112.12474] Generalization capabilities of neural networks in lattice applications](http://arxiv.org/abs/2112.12474)


  In recent years, the use of machine learning has become increasingly popular
in the context of lattice field theories. An essential element of such theories
is represented by symmetries, whose inclusion in the neural network properties
can lead to high reward in terms of performance and generalizability. A
fundamental symmetry that usually characterizes physical systems on a lattice
with periodic boundary conditions is equivariance under spacetime translations.
Here we investigate the advantages of adopting translationally equivariant
neural networks in favor of non-equivariant ones. The system we consider is a
complex scalar field with quartic interaction on a two-dimensional lattice in
the flux representation, on which the networks carry out various regression and
classification tasks. Promising equivariant and non-equivariant architectures
are identified with a systematic search. We demonstrate that in most of these
tasks our best equivariant architectures can perform and generalize
significantly better than their non-equivariant counterparts, which applies not
only to physical parameters beyond those represented in the training set, but
also to different lattice sizes.

    

### [[2112.12482] Self-supervised Representation Learning of Neuronal Morphologies](http://arxiv.org/abs/2112.12482)


  Understanding the diversity of cell types and their function in the brain is
one of the key challenges in neuroscience. The advent of large-scale datasets
has given rise to the need of unbiased and quantitative approaches to cell type
classification. We present GraphDINO, a purely data-driven approach to learning
a low dimensional representation of the 3D morphology of neurons. GraphDINO is
a novel graph representation learning method for spatial graphs utilizing
self-supervised learning on transformer models. It smoothly interpolates
between attention-based global interaction between nodes and classic graph
convolutional processing. We show that this method is able to yield
morphological cell type clustering that is comparable to manual feature-based
classification and shows a good correspondence to expert-labeled cell types in
two different species and cortical areas. Our method is applicable beyond
neuroscience in settings where samples in a dataset are graphs and graph-level
embeddings are desired.

    

### [[2112.12490] Curriculum Learning for Safe Mapless Navigation](http://arxiv.org/abs/2112.12490)


  This work investigates the effects of Curriculum Learning (CL)-based
approaches on the agent's performance. In particular, we focus on the safety
aspect of robotic mapless navigation, comparing over a standard end-to-end
(E2E) training strategy. To this end, we present a CL approach that leverages
Transfer of Learning (ToL) and fine-tuning in a Unity-based simulation with the
Robotnik Kairos as a robotic agent. For a fair comparison, our evaluation
considers an equal computational demand for every learning approach (i.e., the
same number of interactions and difficulty of the environments) and confirms
that our CL-based method that uses ToL outperforms the E2E methodology. In
particular, we improve the average success rate and the safety of the trained
policy, resulting in 10% fewer collisions in unseen testing scenarios. To
further confirm these results, we employ a formal verification tool to quantify
the number of correct behaviors of Reinforcement Learning policies over desired
specifications.

    

### [[2112.12493] Equivariance and generalization in neural networks](http://arxiv.org/abs/2112.12493)


  The crucial role played by the underlying symmetries of high energy physics
and lattice field theories calls for the implementation of such symmetries in
the neural network architectures that are applied to the physical system under
consideration. In these proceedings, we focus on the consequences of
incorporating translational equivariance among the network properties,
particularly in terms of performance and generalization. The benefits of
equivariant networks are exemplified by studying a complex scalar field theory,
on which various regression and classification tasks are examined. For a
meaningful comparison, promising equivariant and non-equivariant architectures
are identified by means of a systematic search. The results indicate that in
most of the tasks our best equivariant architectures can perform and generalize
significantly better than their non-equivariant counterparts, which applies not
only to physical parameters beyond those represented in the training set, but
also to different lattice sizes.

    

### [[2112.12506] Attentive Multi-View Deep Subspace Clustering Net](http://arxiv.org/abs/2112.12506)


  In this paper, we propose a novel Attentive Multi-View Deep Subspace Nets
(AMVDSN), which deeply explores underlying consistent and view-specific
information from multiple views and fuse them by considering each view's
dynamic contribution obtained by attention mechanism. Unlike most multi-view
subspace learning methods that they directly reconstruct data points on raw
data or only consider consistency or complementarity when learning
representation in deep or shallow space, our proposed method seeks to find a
joint latent representation that explicitly considers both consensus and
view-specific information among multiple views, and then performs subspace
clustering on learned joint latent representation.Besides, different views
contribute differently to representation learning, we therefore introduce
attention mechanism to derive dynamic weight for each view, which performs much
better than previous fusion methods in the field of multi-view subspace
clustering. The proposed algorithm is intuitive and can be easily optimized
just by using Stochastic Gradient Descent (SGD) because of the neural network
framework, which also provides strong non-linear characterization capability
compared with traditional subspace clustering approaches. The experimental
results on seven real-world data sets have demonstrated the effectiveness of
our proposed algorithm against some state-of-the-art subspace learning
approaches.

    

### [[2112.12509] Integrating Quantum Processor Device and Control Optimization in a Gradient-based Framework](http://arxiv.org/abs/2112.12509)


  In a quantum processor, the device design and external controls together
contribute to the quality of the target quantum operations. As we continuously
seek better alternative qubit platforms, we explore the increasingly large
device and control design space. Thus, optimization becomes more and more
challenging. In this work, we demonstrate that the figure of merit reflecting a
design goal can be made differentiable with respect to the device and control
parameters. In addition, we can compute the gradient of the design objective
efficiently in a similar manner to the back-propagation algorithm and then
utilize the gradient to optimize the device and the control parameters jointly
and efficiently. This extends the scope of the quantum optimal control to
superconducting device design. We also demonstrate the viability of
gradient-based joint optimization over the device and control parameters
through a few examples.

    

### [[2112.12510] Neuroevolution deep learning architecture search for estimation of river surface elevation from photogrammetric Digital Surface Models](http://arxiv.org/abs/2112.12510)


  Development of the new methods of surface water observation is crucial in the
perspective of increasingly frequent extreme hydrological events related to
global warming and increasing demand for water. Orthophotos and digital surface
models (DSMs) obtained using UAV photogrammetry can be used to determine the
Water Surface Elevation (WSE) of a river. However, this task is difficult due
to disturbances of the water surface on DSMs caused by limitations of
photogrammetric algorithms. In this study, machine learning was used to extract
a WSE value from disturbed photogrammetric data. A brand new dataset has been
prepared specifically for this purpose by hydrology and photogrammetry experts.
The new method is an important step toward automating water surface level
measurements with high spatial and temporal resolution. Such data can be used
to validate and calibrate of hydrological, hydraulic and hydrodynamic models
making hydrological forecasts more accurate, in particular predicting extreme
and dangerous events such as floods or droughts. For our knowledge this is the
first approach in which dataset was created for this purpose and deep learning
models were used for this task. Additionally, neuroevolution algorithm was set
to explore different architectures to find local optimal models and
non-gradient search was performed to fine-tune the model parameters. The
achieved results have better accuracy compared to manual methods of determining
WSE from photogrammetric DSMs.

    

### [[2112.12524] Emulation of greenhouse-gas sensitivities using variational autoencoders](http://arxiv.org/abs/2112.12524)


  Flux inversion is the process by which sources and sinks of a gas are
identified from observations of gas mole fraction. The inversion often involves
running a Lagrangian particle dispersion model (LPDM) to generate sensitivities
between observations and fluxes over a spatial domain of interest. The LPDM
must be run backward in time for every gas measurement, and this can be
computationally prohibitive. To address this problem, here we develop a novel
spatio-temporal emulator for LPDM sensitivities that is built using a
convolutional variational autoencoder (CVAE). With the encoder segment of the
CVAE, we obtain approximate (variational) posterior distributions over latent
variables in a low-dimensional space. We then use a spatio-temporal Gaussian
process emulator on the low-dimensional space to emulate new variables at
prediction locations and time points. Emulated variables are then passed
through the decoder segment of the CVAE to yield emulated sensitivities. We
show that our CVAE-based emulator outperforms the more traditional emulator
built using empirical orthogonal functions and that it can be used with
different LPDMs. We conclude that our emulation-based approach can be used to
reliably reduce the computing time needed to generate LPDM outputs for use in
high-resolution flux inversions.

    

### [[2112.12533] PyCIL: A Python Toolbox for Class-Incremental Learning](http://arxiv.org/abs/2112.12533)


  Traditional machine learning systems are deployed under the closed-world
setting, which requires the entire training data before the offline training
process. However, real-world applications often face the incoming new classes,
and a model should incorporate them continually. The learning paradigm is
called Class-Incremental Learning (CIL). We propose a Python toolbox that
implements several key algorithms for class-incremental learning to ease the
burden of researchers in the machine learning community. The toolbox contains
implementations of a number of founding works of CIL such as EWC and iCaRL, but
also provides current state-of-the-art algorithms that can be used for
conducting novel fundamental research. This toolbox, named PyCIL for Python
Class-Incremental Learning, is available at this https URL


### [[2112.12535] FourierMask: Instance Segmentation using Fourier Mapping in Implicit Neural Networks](http://arxiv.org/abs/2112.12535)


  We present FourierMask, which employs Fourier series combined with implicit
neural representations to generate instance segmentation masks. We apply a
Fourier mapping (FM) to the coordinate locations and utilize the mapped
features as inputs to an implicit representation (coordinate-based multi-layer
perceptron (MLP)). FourierMask learns to predict the coefficients of the FM for
a particular instance, and therefore adapts the FM to a specific object. This
allows FourierMask to be generalized to predict instance segmentation masks
from natural images. Since implicit functions are continuous in the domain of
input coordinates, we illustrate that by sub-sampling the input pixel
coordinates, we can generate higher resolution masks during inference.
Furthermore, we train a renderer MLP (FourierRend) on the uncertain predictions
of FourierMask and illustrate that it significantly improves the quality of the
masks. FourierMask shows competitive results on the MS COCO dataset compared to
the baseline Mask R-CNN at the same output resolution and surpasses it on
higher resolution.

    

### [[2112.12542] How Much of the Chemical Space Has Been Covered? Measuring and Improving the Variety of Candidate Set in Molecular Generation](http://arxiv.org/abs/2112.12542)


  Forming a high-quality molecular candidate set that contains a wide range of
dissimilar compounds is crucial to the success of drug discovery. However,
comparing to the research aiming at optimizing chemical properties, how to
measure and improve the variety of drug candidates is relatively understudied.
In this paper, we first investigate the problem of properly measuring the
molecular variety through both an axiomatic analysis framework and an empirical
study. Our analysis suggests that many existing measures are not suitable for
evaluating the variety of molecules. We also propose new variety measures based
on our analysis. We further explicitly integrate the proposed variety measures
into the optimization objective of molecular generation models. Our experiment
results demonstrate that this new optimization objective can guide molecular
generation models to find compounds that cover a lager chemical space,
providing the downstream phases with more distinctive drug candidate choices.

    

### [[2112.12544] Newsvendor Model with Deep Reinforcement Learning](http://arxiv.org/abs/2112.12544)


  I present a deep reinforcement learning (RL) solution to the mathematical
problem known as the Newsvendor model, which seeks to optimize profit given a
probabilistic demand distribution. To reflect a more realistic and complex
situation, the demand distribution can change for different days of the week,
thus changing the optimum behavior. I used a Twin-Delayed Deep Deterministic
Policy Gradient agent (written as completely original code) with both an actor
and critic network to solve this problem. The agent was able to learn optimal
behavior consistent with the analytical solution of the problem, and could
identify separate probability distributions for different days of the week and
behave accordingly.

    

### [[2112.12545] A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](http://arxiv.org/abs/2112.12545)


  Reinforcement learning has recently shown promise in learning quality
solutions in many combinatorial optimization problems. In particular, the
attention-based encoder-decoder models show high effectiveness on various
routing problems, including the Traveling Salesman Problem (TSP).
Unfortunately, they perform poorly for the TSP with Drone (TSP-D), requiring
routing a heterogeneous fleet of vehicles in coordination -- a truck and a
drone. In TSP-D, the two vehicles are moving in tandem and may need to wait at
a node for the other vehicle to join. State-less attention-based decoder fails
to make such coordination between vehicles. We propose an attention
encoder-LSTM decoder hybrid model, in which the decoder's hidden state can
represent the sequence of actions made. We empirically demonstrate that such a
hybrid model improves upon a purely attention-based model for both solution
quality and computational efficiency. Our experiments on the min-max
Capacitated Vehicle Routing Problem (mmCVRP) also confirm that the hybrid model
is more suitable for coordinated routing of multiple vehicles than the
attention-based model.

    

### [[2112.12549] Combining Minkowski and Chebyshev: New distance proposal and survey of distance metrics using k-nearest neighbours classifier](http://arxiv.org/abs/2112.12549)


  This work proposes a distance that combines Minkowski and Chebyshev distances
and can be seen as an intermediary distance. This combination not only achieves
efficient run times in neighbourhood iteration tasks in Z^2, but also obtains
good accuracies when coupled with the k-Nearest Neighbours (k-NN) classifier.
The proposed distance is approximately 1.3 times faster than Manhattan distance
and 329.5 times faster than Euclidean distance in discrete neighbourhood
iterations. An accuracy analysis of the k-NN classifier using a total of 33
datasets from the UCI repository, 15 distances and values assigned to k that
vary from 1 to 200 is presented. In this experiment, the proposed distance
obtained accuracies that were better than the average more often than its
counterparts (in 26 cases out of 33), and also obtained the best accuracy more
frequently (in 9 out of 33 cases).

    

### [[2112.12551] Preprocessing in Inductive Logic Programming](http://arxiv.org/abs/2112.12551)


  Inductive logic programming is a type of machine learning in which logic
programs are learned from examples. This learning typically occurs relative to
some background knowledge provided as a logic program. This dissertation
introduces bottom preprocessing, a method for generating initial constraints on
the programs an ILP system must consider. Bottom preprocessing applies ideas
from inverse entailment to modern ILP systems. Inverse entailment is an
influential early ILP approach introduced with Progol. This dissertation also
presents $\bot$-Popper, an implementation of bottom preprocessing for the
modern ILP system Popper. It is shown experimentally that bottom preprocessing
can reduce learning times of ILP systems on hard problems. This reduction can
be especially significant when the amount of background knowledge in the
problem is large.

    

### [[2112.12554] Beyond Low Earth Orbit: Biomonitoring, Artificial Intelligence, and Precision Space Health](http://arxiv.org/abs/2112.12554)


  Human space exploration beyond low Earth orbit will involve missions of
significant distance and duration. To effectively mitigate myriad space health
hazards, paradigm shifts in data and space health systems are necessary to
enable Earth-independence, rather than Earth-reliance. Promising developments
in the fields of artificial intelligence and machine learning for biology and
health can address these needs. We propose an appropriately autonomous and
intelligent Precision Space Health system that will monitor, aggregate, and
assess biomedical statuses; analyze and predict personalized adverse health
outcomes; adapt and respond to newly accumulated data; and provide preventive,
actionable, and timely insights to individual deep space crew members and
iterative decision support to their crew medical officer. Here we present a
summary of recommendations from a workshop organized by the National
Aeronautics and Space Administration, on future applications of artificial
intelligence in space biology and health. In the next decade, biomonitoring
technology, biomarker science, spacecraft hardware, intelligent software, and
streamlined data management must mature and be woven together into a Precision
Space Health system to enable humanity to thrive in deep space.

    

### [[2112.12555] Optimal learning of high-dimensional classification problems using deep neural networks](http://arxiv.org/abs/2112.12555)


  We study the problem of learning classification functions from noiseless
training samples, under the assumption that the decision boundary is of a
certain regularity. We establish universal lower bounds for this estimation
problem, for general classes of continuous decision boundaries. For the class
of locally Barron-regular decision boundaries, we find that the optimal
estimation rates are essentially independent of the underlying dimension and
can be realized by empirical risk minimization methods over a suitable class of
deep neural networks. These results are based on novel estimates of the $L^1$
and $L^\infty$ entropies of the class of Barron-regular functions.

    

### [[2112.12563] Scalable Variational Quantum Circuits for Autoencoder-based Drug Discovery](http://arxiv.org/abs/2112.12563)


  The de novo design of drug molecules is recognized as a time-consuming and
costly process, and computational approaches have been applied in each stage of
the drug discovery pipeline. Variational autoencoder is one of the
computer-aided design methods which explores the chemical space based on
existing molecular dataset. Quantum machine learning has emerged as an atypical
learning method that may speed up some classical learning tasks because of its
strong expressive power. However, near-term quantum computers suffer from
limited number of qubits which hinders the representation learning in high
dimensional spaces. We present a scalable quantum generative autoencoder
(SQ-VAE) for simultaneously reconstructing and sampling drug molecules, and a
corresponding vanilla variant (SQ-AE) for better reconstruction. The
architectural strategies in hybrid quantum classical networks such as,
adjustable quantum layer depth, heterogeneous learning rates, and patched
quantum circuits are proposed to learn high dimensional dataset such as,
ligand-targeted drugs. Extensive experimental results are reported for
different dimensions including 8x8 and 32x32 after choosing suitable
architectural strategies. The performance of quantum generative autoencoder is
compared with the corresponding classical counterpart throughout all
experiments. The results show that quantum computing advantages can be achieved
for normalized low-dimension molecules, and that high-dimension molecules
generated from quantum generative autoencoders have better drug properties
within the same learning period.

    

### [[2112.12566] Integrating Material Selection with Design Optimization via Neural Networks](http://arxiv.org/abs/2112.12566)


  The engineering design process often entails optimizing the underlying
geometry while simultaneously selecting a suitable material. For a certain
class of simple problems, the two are separable where, for example, one can
first select an optimal material, and then optimize the geometry. However, in
general, the two are not separable. Furthermore, the discrete nature of
material selection is not compatible with gradient-based geometry optimization,
making simultaneous optimization challenging.
In this paper, we propose the use of variational autoencoders (VAE) for
simultaneous optimization. First, a data-driven VAE is used to project the
discrete material database onto a continuous and differentiable latent space.
This is then coupled with a fully-connected neural network, embedded with a
finite-element solver, to simultaneously optimize the material and geometry.
The neural-network's built-in gradient optimizer and back-propagation are
exploited during optimization.
The proposed framework is demonstrated using trusses, where an optimal
material needs to be chosen from a database, while simultaneously optimizing
the cross-sectional areas of the truss members. Several numerical examples
illustrate the efficacy of the proposed framework. The Python code used in
these experiments is available at this http URL


### [[2112.12582] Beyond Low Earth Orbit: Biological Research, Artificial Intelligence, and Self-Driving Labs](http://arxiv.org/abs/2112.12582)


  Space biology research aims to understand fundamental effects of spaceflight
on organisms, develop foundational knowledge to support deep space exploration,
and ultimately bioengineer spacecraft and habitats to stabilize the ecosystem
of plants, crops, microbes, animals, and humans for sustained multi-planetary
life. To advance these aims, the field leverages experiments, platforms, data,
and model organisms from both spaceborne and ground-analog studies. As research
is extended beyond low Earth orbit, experiments and platforms must be maximally
autonomous, light, agile, and intelligent to expedite knowledge discovery. Here
we present a summary of recommendations from a workshop organized by the
National Aeronautics and Space Administration on artificial intelligence,
machine learning, and modeling applications which offer key solutions toward
these space biology challenges. In the next decade, the synthesis of artificial
intelligence into the field of space biology will deepen the biological
understanding of spaceflight effects, facilitate predictive modeling and
analytics, support maximally autonomous and reproducible experiments, and
efficiently manage spaceborne data and metadata, all with the goal to enable
life to thrive in deep space.

    

### [[2112.12584] Attention Based Communication and Control for Multi-UAV Path Planning](http://arxiv.org/abs/2112.12584)


  Inspired by the multi-head attention (MHA) mechanism in natural language
processing, this letter proposes an iterative single-head attention (ISHA)
mechanism for multi-UAV path planning. The ISHA mechanism is run by a
communication helper collecting the state embeddings of UAVs and distributing
an attention score vector to each UAV. The attention scores computed by ISHA
identify how many interactions with other UAVs should be considered in each
UAV's control decision-making. Simulation results corroborate that the
ISHA-based communication and control framework achieves faster travel with
lower inter-UAV collision risks than an MHA-aided baseline, particularly under
limited communication resources.

    

### [[2112.12589] A deep reinforcement learning model for predictive maintenance planning of road assets: Integrating LCA and LCCA](http://arxiv.org/abs/2112.12589)


  Road maintenance planning is an integral part of road asset management. One
of the main challenges in Maintenance and Rehabilitation (M&R) practices is to
determine maintenance type and timing. This research proposes a framework using
Reinforcement Learning (RL) based on the Long Term Pavement Performance (LTPP)
database to determine the type and timing of M&R practices. A predictive DNN
model is first developed in the proposed algorithm, which serves as the
Environment for the RL algorithm. For the Policy estimation of the RL model,
both DQN and PPO models are developed. However, PPO has been selected in the
end due to better convergence and higher sample efficiency. Indicators used in
this study are International Roughness Index (IRI) and Rutting Depth (RD).
Initially, we considered Cracking Metric (CM) as the third indicator, but it
was then excluded due to the much fewer data compared to other indicators,
which resulted in lower accuracy of the results. Furthermore, in
cost-effectiveness calculation (reward), we considered both the economic and
environmental impacts of M&R treatments. Costs and environmental impacts have
been evaluated with paLATE 2.0 software. Our method is tested on a hypothetical
case study of a six-lane highway with 23 kilometers length located in Texas,
which has a warm and wet climate. The results propose a 20-year M&R plan in
which road condition remains in an excellent condition range. Because the early
state of the road is at a good level of service, there is no need for heavy
maintenance practices in the first years. Later, after heavy M&R actions, there
are several 1-2 years of no need for treatments. All of these show that the
proposed plan has a logical result. Decision-makers and transportation agencies
can use this scheme to conduct better maintenance practices that can prevent
budget waste and, at the same time, minimize the environmental impacts.

    

### [[2112.12591] Black-Box Testing of Deep Neural Networks through Test Case Diversity](http://arxiv.org/abs/2112.12591)


  Deep Neural Networks (DNNs) have been extensively used in many areas
including image processing, medical diagnostics, and autonomous driving.
However, DNNs can exhibit erroneous behaviours that may lead to critical
errors, especially when used in safety-critical systems. Inspired by testing
techniques for traditional software systems, researchers have proposed neuron
coverage criteria, as an analogy to source code coverage, to guide the testing
of DNN models. Despite very active research on DNN coverage, several recent
studies have questioned the usefulness of such criteria in guiding DNN testing.
Further, from a practical standpoint, these criteria are white-box as they
require access to the internals or training data of DNN models, which is in
many contexts not feasible or convenient. In this paper, we investigate
black-box input diversity metrics as an alternative to white-box coverage
criteria. To this end, we first select and adapt three diversity metrics and
study, in a controlled manner, their capacity to measure actual diversity in
input sets. We then analyse their statistical association with fault detection
using two datasets and three DNN models. We further compare diversity with
state-of-the-art white-box coverage criteria. Our experiments show that relying
on the diversity of image features embedded in test input sets is a more
reliable indicator than coverage criteria to effectively guide the testing of
DNNs. Indeed, we found that one of our selected black-box diversity metrics far
outperforms existing coverage criteria in terms of fault-revealing capability
and computational time. Results also confirm the suspicions that
state-of-the-art coverage metrics are not adequate to guide the construction of
test input sets to detect as many faults as possible with natural inputs.

    

### [[2112.12596] INTRPRT: A Systematic Review of and Guidelines for Designing and Validating Transparent AI in Medical Image Analysis](http://arxiv.org/abs/2112.12596)


  Transparency in Machine Learning (ML), attempts to reveal the working
mechanisms of complex models. Transparent ML promises to advance human factors
engineering goals of human-centered AI in the target users. From a
human-centered design perspective, transparency is not a property of the ML
model but an affordance, i.e. a relationship between algorithm and user; as a
result, iterative prototyping and evaluation with users is critical to
attaining adequate solutions that afford transparency. However, following
human-centered design principles in healthcare and medical image analysis is
challenging due to the limited availability of and access to end users. To
investigate the state of transparent ML in medical image analysis, we conducted
a systematic review of the literature. Our review reveals multiple severe
shortcomings in the design and validation of transparent ML for medical image
analysis applications. We find that most studies to date approach transparency
as a property of the model itself, similar to task performance, without
considering end users during neither development nor evaluation. Additionally,
the lack of user research, and the sporadic validation of transparency claims
put contemporary research on transparent ML for medical image analysis at risk
of being incomprehensible to users, and thus, clinically irrelevant. To
alleviate these shortcomings in forthcoming research while acknowledging the
challenges of human-centered design in healthcare, we introduce the INTRPRT
guideline, a systematic design directive for transparent ML systems in medical
image analysis. The INTRPRT guideline suggests formative user research as the
first step of transparent model design to understand user needs and domain
requirements. Following this process produces evidence to support design
choices, and ultimately, increases the likelihood that the algorithms afford
transparency.

    

### [[2112.12616] Deep Filtering with DNN, CNN and RNN](http://arxiv.org/abs/2112.12616)


  This paper is about a deep learning approach for linear and nonlinear
filtering. The idea is to train a neural network with Monte Carlo samples
generated from a nominal dynamic model. Then the network weights are applied to
Monte Carlo samples from an actual dynamic model. A main focus of this paper is
on the deep filters with three major neural network architectures (DNN, CNN,
RNN). Our deep filter compares favorably to the traditional Kalman filter in
linear cases and outperform the extended Kalman filter in nonlinear cases. Then
a switching model with jumps is studied to show the adaptiveness and power of
our deep filtering. Among the three major NNs, the CNN outperform the others on
average. while the RNN does not seem to be suitable for the filtering problem.
One advantage of the deep filter is its robustness when the nominal model and
actual model differ. The other advantage of deep filtering is real data can be
used directly to train the deep neutral network. Therefore, model calibration
can be by-passed all together.

    

### [[2112.12618] Manifold Learning Benefits GANs](http://arxiv.org/abs/2112.12618)


  In this paper, we improve Generative Adversarial Networks by incorporating a
manifold learning step into the discriminator. We consider locality-constrained
linear and subspace-based manifolds, and locality-constrained non-linear
manifolds. In our design, the manifold learning and coding steps are
intertwined with layers of the discriminator, with the goal of attracting
intermediate feature representations onto manifolds. We adaptively balance the
discrepancy between feature representations and their manifold view, which
represents a trade-off between denoising on the manifold and refining the
manifold. We conclude that locality-constrained non-linear manifolds have the
upper hand over linear manifolds due to their non-uniform density and
smoothness. We show substantial improvements over different recent
state-of-the-art baselines.

    

### [[2112.12630] A Survey of Near-Data Processing Architectures for Neural Networks](http://arxiv.org/abs/2112.12630)


  Data-intensive workloads and applications, such as machine learning (ML), are
fundamentally limited by traditional computing systems based on the von-Neumann
architecture. As data movement operations and energy consumption become key
bottlenecks in the design of computing systems, the interest in unconventional
approaches such as Near-Data Processing (NDP), machine learning, and especially
neural network (NN)-based accelerators has grown significantly. Emerging memory
technologies, such as ReRAM and 3D-stacked, are promising for efficiently
architecting NDP-based accelerators for NN due to their capabilities to work as
both: High-density/low-energy storage and in/near-memory computation/search
engine. In this paper, we present a survey of techniques for designing NDP
architectures for NN. By classifying the techniques based on the memory
technology employed, we underscore their similarities and differences. Finally,
we discuss open challenges and future perspectives that need to be explored in
order to improve and extend the adoption of NDP architectures for future
computing platforms. This paper will be valuable for computer architects, chip
designers and researchers in the area of machine learning.

    

### [[2112.12635] AcME -- Accelerated Model-agnostic Explanations: Fast Whitening of the Machine-Learning Black Box](http://arxiv.org/abs/2112.12635)


  In the context of human-in-the-loop Machine Learning applications, like
Decision Support Systems, interpretability approaches should provide actionable
insights without making the users wait. In this paper, we propose Accelerated
Model-agnostic Explanations (AcME), an interpretability approach that quickly
provides feature importance scores both at the global and the local level. AcME
can be applied a posteriori to each regression or classification model. Not
only does AcME compute feature ranking, but it also provides a what-if analysis
tool to assess how changes in features values would affect model predictions.
We evaluated the proposed approach on synthetic and real-world datasets, also
in comparison with SHapley Additive exPlanations (SHAP), the approach we drew
inspiration from, which is currently one of the state-of-the-art model-agnostic
interpretability approaches. We achieved comparable results in terms of quality
of produced explanations while reducing dramatically the computational time and
providing consistent visualization for global and local interpretations. To
foster research in this field, and for the sake of reproducibility, we also
provide a repository with the code used for the experiments.

    

### [[2112.12641] Prolog-based agnostic explanation module for structured pattern classification](http://arxiv.org/abs/2112.12641)


  This paper presents a Prolog-based reasoning module to generate
counterfactual explanations given the predictions computed by a black-box
classifier. The proposed symbolic reasoning module can also resolve what-if
queries using the ground-truth labels instead of the predicted ones. Overall,
our approach comprises four well-defined stages that can be applied to any
structured pattern classification problem. Firstly, we pre-process the given
dataset by imputing missing values and normalizing the numerical features.
Secondly, we transform numerical features into symbolic ones using fuzzy
clustering such that extracted fuzzy clusters are mapped to an ordered set of
predefined symbols. Thirdly, we encode instances as a Prolog rule using the
nominal values, the predefined symbols, the decision classes, and the
confidence values. Fourthly, we compute the overall confidence of each Prolog
rule using fuzzy-rough set theory to handle the uncertainty caused by
transforming numerical quantities into symbols. This step comes with an
additional theoretical contribution to a new similarity function to compare the
previously defined Prolog rules involving confidence values. Finally, we
implement a chatbot as a proxy between human beings and the Prolog-based
reasoning module to resolve natural language queries and generate
counterfactual explanations. During the numerical simulations using synthetic
datasets, we study the performance of our system when using different fuzzy
operators and similarity functions. Towards the end, we illustrate how our
reasoning module works using different use cases.

    

### [[2112.12650] Distilling the Knowledge of Romanian BERTs Using Multiple Teachers](http://arxiv.org/abs/2112.12650)


  As transfer learning from large-scale pre-trained language models has become
prevalent in Natural Language Processing, running these models in
computationally constrained environments remains a challenging problem yet to
address. Several solutions including knowledge distillation, network
quantization or network pruning have been proposed; however, these approaches
focus mostly on the English language, thus widening the gap when considering
low-resource languages. In this work, we introduce three light and fast
versions of distilled BERT models for the Romanian language:
Distil-BERT-base-ro, Distil-RoBERT-base and DistilMulti-BERT-base-ro. The first
two models resulted from individually distilling the knowledge of the two base
versions of Romanian BERTs available in literature, while the last one was
obtained by distilling their ensemble. To our knowledge, this is the first
attempt to create publicly available Romanian distilled BERT models, which were
thoroughly evaluated on five tasks: part-of-speech tagging, named entity
recognition, sentiment analysis, semantic textual similarity and dialect
identification. The experimental results on these benchmarks proved that our
three distilled models maintain most performance in terms of accuracy with
their teachers, while being twice as fast on a GPU and ~35\% smaller. In
addition, we further test the similarity between our students and their
teachers prediction by measuring their label and probability loyalty, together
with regression loyalty - a new metric introduced in this work.

    

### [[2112.12668] 3D Skeleton-based Few-shot Action Recognition with JEANIE is not so Naïve](http://arxiv.org/abs/2112.12668)


  In this paper, we propose a Few-shot Learning pipeline for 3D skeleton-based
action recognition by Joint tEmporal and cAmera viewpoiNt alIgnmEnt (JEANIE).
To factor out misalignment between query and support sequences of 3D body
joints, we propose an advanced variant of Dynamic Time Warping which jointly
models each smooth path between the query and support frames to achieve
simultaneously the best alignment in the temporal and simulated camera
viewpoint spaces for end-to-end learning under the limited few-shot training
data. Sequences are encoded with a temporal block encoder based on Simple
Spectral Graph Convolution, a lightweight linear Graph Neural Network backbone
(we also include a setting with a transformer). Finally, we propose a
similarity-based loss which encourages the alignment of sequences of the same
class while preventing the alignment of unrelated sequences. We demonstrate
state-of-the-art results on NTU-60, NTU-120, Kinetics-skeleton and UWA3D
Multiview Activity II.

    

### [[2112.12705] Explainable Artificial Intelligence Methods in Combating Pandemics: A Systematic Review](http://arxiv.org/abs/2112.12705)


  Despite the myriad peer-reviewed papers demonstrating novel Artificial
Intelligence (AI)-based solutions to COVID-19 challenges during the pandemic,
few have made significant clinical impact. The impact of artificial
intelligence during the COVID-19 pandemic was greatly limited by lack of model
transparency. This systematic review examines the use of Explainable Artificial
Intelligence (XAI) during the pandemic and how its use could overcome barriers
to real-world success. We find that successful use of XAI can improve model
performance, instill trust in the end-user, and provide the value needed to
affect user decision-making. We introduce the reader to common XAI techniques,
their utility, and specific examples of their application. Evaluation of XAI
results is also discussed as an important step to maximize the value of
AI-based clinical decision support systems. We illustrate the classical,
modern, and potential future trends of XAI to elucidate the evolution of novel
XAI techniques. Finally, we provide a checklist of suggestions during the
experimental design process supported by recent publications. Common challenges
during the implementation of AI solutions are also addressed with specific
examples of potential solutions. We hope this review may serve as a guide to
improve the clinical impact of future AI-based solutions.

    

### [[2112.12707] Improving Robustness and Uncertainty Modelling in Neural Ordinary Differential Equations](http://arxiv.org/abs/2112.12707)


  Neural ordinary differential equations (NODE) have been proposed as a
continuous depth generalization to popular deep learning models such as
Residual networks (ResNets). They provide parameter efficiency and automate the
model selection process in deep learning models to some extent. However, they
lack the much-required uncertainty modelling and robustness capabilities which
are crucial for their use in several real-world applications such as autonomous
driving and healthcare. We propose a novel and unique approach to model
uncertainty in NODE by considering a distribution over the end-time $T$ of the
ODE solver. The proposed approach, latent time NODE (LT-NODE), treats $T$ as a
latent variable and apply Bayesian learning to obtain a posterior distribution
over $T$ from the data. In particular, we use variational inference to learn an
approximate posterior and the model parameters. Prediction is done by
considering the NODE representations from different samples of the posterior
and can be done efficiently using a single forward pass. As $T$ implicitly
defines the depth of a NODE, posterior distribution over $T$ would also help in
model selection in NODE. We also propose, adaptive latent time NODE (ALT-NODE),
which allow each data point to have a distinct posterior distribution over
end-times. ALT-NODE uses amortized variational inference to learn an
approximate posterior using inference networks. We demonstrate the
effectiveness of the proposed approaches in modelling uncertainty and
robustness through experiments on synthetic and several real-world image
classification data.

    

### [[2112.12713] Modeling Implicit Bias with Fuzzy Cognitive Maps](http://arxiv.org/abs/2112.12713)


  This paper presents a Fuzzy Cognitive Map model to quantify implicit bias in
structured datasets where features can be numeric or discrete. In our proposal,
problem features are mapped to neural concepts that are initially activated by
experts when running what-if simulations, whereas weights connecting the neural
concepts represent absolute correlation/association patterns between features.
In addition, we introduce a new reasoning mechanism equipped with a
normalization-like transfer function that prevents neurons from saturating.
Another advantage of this new reasoning mechanism is that it can easily be
controlled by regulating nonlinearity when updating neurons' activation values
in each iteration. Finally, we study the convergence of our model and derive
analytical conditions concerning the existence and unicity of fixed-point
attractors.

    

### [[2112.12717] Forward Composition Propagation for Explainable Neural Reasoning](http://arxiv.org/abs/2112.12717)


  This paper proposes an algorithm called Forward Composition Propagation (FCP)
to explain the predictions of feed-forward neural networks operating on
structured pattern recognition problems. In the proposed FCP algorithm, each
neuron is described by a composition vector indicating the role of each problem
feature in that neuron. Composition vectors are initialized using a given input
instance and subsequently propagated through the whole network until we reach
the output layer. It is worth mentioning that the algorithm is executed once
the network's training network is done. The sign of each composition value
indicates whether the corresponding feature excites or inhibits the neuron,
while the absolute value quantifies such an impact. Aiming to validate the FCP
algorithm's correctness, we develop a case study concerning bias detection in a
state-of-the-art problem in which the ground truth is known. The simulation
results show that the composition values closely align with the expected
behavior of protected features.

    

### [[2112.12728] Latent Time Neural Ordinary Differential Equations](http://arxiv.org/abs/2112.12728)


  Neural ordinary differential equations (NODE) have been proposed as a
continuous depth generalization to popular deep learning models such as
Residual networks (ResNets). They provide parameter efficiency and automate the
model selection process in deep learning models to some extent. However, they
lack the much-required uncertainty modelling and robustness capabilities which
are crucial for their use in several real-world applications such as autonomous
driving and healthcare. We propose a novel and unique approach to model
uncertainty in NODE by considering a distribution over the end-time $T$ of the
ODE solver. The proposed approach, latent time NODE (LT-NODE), treats $T$ as a
latent variable and apply Bayesian learning to obtain a posterior distribution
over $T$ from the data. In particular, we use variational inference to learn an
approximate posterior and the model parameters. Prediction is done by
considering the NODE representations from different samples of the posterior
and can be done efficiently using a single forward pass. As $T$ implicitly
defines the depth of a NODE, posterior distribution over $T$ would also help in
model selection in NODE. We also propose, adaptive latent time NODE (ALT-NODE),
which allow each data point to have a distinct posterior distribution over
end-times. ALT-NODE uses amortized variational inference to learn an
approximate posterior using inference networks. We demonstrate the
effectiveness of the proposed approaches in modelling uncertainty and
robustness through experiments on synthetic and several real-world image
classification data.

    

### [[2112.12740] Learning Cooperative Multi-Agent Policies with Partial Reward Decoupling](http://arxiv.org/abs/2112.12740)


  One of the preeminent obstacles to scaling multi-agent reinforcement learning
to large numbers of agents is assigning credit to individual agents' actions.
In this paper, we address this credit assignment problem with an approach that
we call \textit{partial reward decoupling} (PRD), which attempts to decompose
large cooperative multi-agent RL problems into decoupled subproblems involving
subsets of agents, thereby simplifying credit assignment. We empirically
demonstrate that decomposing the RL problem using PRD in an actor-critic
algorithm results in lower variance policy gradient estimates, which improves
data efficiency, learning stability, and asymptotic performance across a wide
array of multi-agent RL tasks, compared to various other actor-critic
approaches. Additionally, we relate our approach to counterfactual multi-agent
policy gradient (COMA), a state-of-the-art MARL algorithm, and empirically show
that our approach outperforms COMA by making better use of information in
agents' reward streams, and by enabling recent advances in advantage estimation
to be used.

    

### [[2112.12748] Assessing the Impact of Attention and Self-Attention Mechanisms on the Classification of Skin Lesions](http://arxiv.org/abs/2112.12748)


  Attention mechanisms have raised significant interest in the research
community, since they promise significant improvements in the performance of
neural network architectures. However, in any specific problem, we still lack a
principled way to choose specific mechanisms and hyper-parameters that lead to
guaranteed improvements. More recently, self-attention has been proposed and
widely used in transformer-like architectures, leading to significant
breakthroughs in some applications. In this work we focus on two forms of
attention mechanisms: attention modules and self-attention. Attention modules
are used to reweight the features of each layer input tensor. Different modules
have different ways to perform this reweighting in fully connected or
convolutional layers. The attention models studied are completely modular and
in this work they will be used with the popular ResNet architecture.
Self-Attention, originally proposed in the area of Natural Language Processing
makes it possible to relate all the items in an input sequence. Self-Attention
is becoming increasingly popular in Computer Vision, where it is sometimes
combined with convolutional layers, although some recent architectures do away
entirely with convolutions. In this work, we study and perform an objective
comparison of a number of different attention mechanisms in a specific computer
vision task, the classification of samples in the widely used Skin Cancer MNIST
dataset. The results show that attention modules do sometimes improve the
performance of convolutional neural network architectures, but also that this
improvement, although noticeable and statistically significant, is not
consistent in different settings. The results obtained with self-attention
mechanisms, on the other hand, show consistent and significant improvements,
leading to the best results even in architectures with a reduced number of
parameters.

    

### [[2112.12770] Optimal and instance-dependent guarantees for Markovian linear stochastic approximation](http://arxiv.org/abs/2112.12770)


  We study stochastic approximation procedures for approximately solving a
$d$-dimensional linear fixed point equation based on observing a trajectory of
length $n$ from an ergodic Markov chain. We first exhibit a non-asymptotic
bound of the order $t_{\mathrm{mix}} \tfrac{d}{n}$ on the squared error of the
last iterate of a standard scheme, where $t_{\mathrm{mix}}$ is a mixing time.
We then prove a non-asymptotic instance-dependent bound on a suitably averaged
sequence of iterates, with a leading term that matches the local asymptotic
minimax limit, including sharp dependence on the parameters $(d,
t_{\mathrm{mix}})$ in the higher order terms. We complement these upper bounds
with a non-asymptotic minimax lower bound that establishes the
instance-optimality of the averaged SA estimator. We derive corollaries of
these results for policy evaluation with Markov noise -- covering the
TD($\lambda$) family of algorithms for all $\lambda \in [0, 1)$ -- and linear
autoregressive models. Our instance-dependent characterizations open the door
to the design of fine-grained model selection procedures for hyperparameter
tuning (e.g., choosing the value of $\lambda$ when running the TD($\lambda$)
algorithm).

    

### [[2112.12782] SeMask: Semantically Masked Transformers for Semantic Segmentation](http://arxiv.org/abs/2112.12782)


  Finetuning a pretrained backbone in the encoder part of an image transformer
network has been the traditional approach for the semantic segmentation task.
However, such an approach leaves out the semantic context that an image
provides during the encoding stage. This paper argues that incorporating
semantic information of the image into pretrained hierarchical
transformer-based backbones while finetuning improves the performance
considerably. To achieve this, we propose SeMask, a simple and effective
framework that incorporates semantic information into the encoder with the help
of a semantic attention operation. In addition, we use a lightweight semantic
decoder during training to provide supervision to the intermediate semantic
prior maps at every stage. Our experiments demonstrate that incorporating
semantic priors enhances the performance of the established hierarchical
encoders with a slight increase in the number of FLOPs. We provide empirical
proof by integrating SeMask into each variant of the Swin-Transformer as our
encoder paired with different decoders. Our framework achieves a new
state-of-the-art of 58.22% mIoU on the ADE20K dataset and improvements of over
3% in the mIoU metric on the Cityscapes dataset. The code and checkpoints are
publicly available at
this https URL .

    

### [[1802.07382] Generic Coreset for Scalable Learning of Monotonic Kernels: Logistic Regression, Sigmoid and more](http://arxiv.org/abs/1802.07382)


  Coreset (or core-set) is a small weighted \emph{subset} $Q$ of an input set
$P$ with respect to a given \emph{monotonic} function
$f:\mathbb{R}\to\mathbb{R}$ that \emph{provably} approximates its fitting loss
$\sum_{p\in P}f(p\cdot x)$ to \emph{any} given $x\in\mathbb{R}^d$. Using $Q$ we
can obtain approximation of $x^*$ that minimizes this loss, by running
\emph{existing} optimization algorithms on $Q$. In this work we provide: (i) A
lower bound which proves that there are sets with no coresets smaller than
$n=|P|$ for general monotonic loss functions. (ii) A proof that, under a
natural assumption that holds e.g. for logistic regression and the sigmoid
activation functions, a small coreset exists for \emph{any} input $P$. (iii) A
generic coreset construction algorithm that computes such a small coreset $Q$
in $O(nd+n\log n)$ time, and (iv) Experimental results which demonstrate that
our coresets are effective and are much smaller in practice than predicted in
theory.

    

### [[1902.10658] Regularity Normalization: Neuroscience-Inspired Unsupervised Attention across Neural Network Layers](http://arxiv.org/abs/1902.10658)


  Inspired by the adaptation phenomenon of neuronal firing, we propose the
regularity normalization (RN) as an unsupervised attention mechanism (UAM)
which computes the statistical regularity in the implicit space of neural
networks under the Minimum Description Length (MDL) principle. Treating the
neural network optimization process as a partially observable model selection
problem, the regularity normalization constrains the implicit space by a
normalization factor, the universal code length. We compute this universal code
incrementally across neural network layers and demonstrate the flexibility to
include data priors such as top-down attention and other oracle information.
Empirically, our approach outperforms existing normalization methods in
tackling limited, imbalanced and non-stationary input distribution in image
classification, classic control, procedurally-generated reinforcement learning,
generative modeling, handwriting generation and question answering tasks with
various neural network architectures. Lastly, the unsupervised attention
mechanisms is a useful probing tool for neural networks by tracking the
dependency and critical learning stages across layers and recurrent time steps
of deep networks.

    

### [[1905.11027] A Geometric Modeling of Occam's Razor in Deep Learning](http://arxiv.org/abs/1905.11027)


  Why do deep neural networks (DNNs) benefit from very high dimensional
parameter spaces? Their huge parameter complexities vs stunning performances in
practice is all the more intriguing and not explainable using the standard
theory of regular models. In this work, we propose a geometrically flavored
information-theoretic approach to study this phenomenon. Namely, we introduce
the locally varying dimensionality of the parameter space of neural network
models by considering the number of significant dimensions of the Fisher
information matrix, and model the parameter space as a manifold using the
framework of singular semi-Riemannian geometry. We derive model complexity
measures which yield short description lengths for deep neural network models
based on their singularity analysis thus explaining the good performance of
DNNs despite their large number of parameters.

    

### [[1905.11797] ROI Maximization in Stochastic Online Decision-Making](http://arxiv.org/abs/1905.11797)


  We introduce a novel theoretical framework for Return On Investment (ROI)
maximization in repeated decision-making. Our setting is motivated by the use
case of companies that regularly receive proposals for technological
innovations and want to quickly decide whether they are worth implementing. We
design an algorithm for learning ROI-maximizing decision-making policies over a
sequence of innovation proposals. Our algorithm provably converges to an
optimal policy in class $\Pi$ at a rate of order
$\min\big\{1/(N\Delta^2),N^{-1/3}\}$, where $N$ is the number of innovations
and $\Delta$ is the suboptimality gap in $\Pi$. A significant hurdle of our
formulation, which sets it aside from other online learning problems such as
bandits, is that running a policy does not provide an unbiased estimate of its
performance.

    

### [[2006.14514] Taming neural networks with TUSLA: Non-convex learning via adaptive stochastic gradient Langevin algorithms](http://arxiv.org/abs/2006.14514)


  Artificial neural networks (ANNs) are typically highly nonlinear systems
which are finely tuned via the optimization of their associated, non-convex
loss functions. In many cases, the gradient of any such loss function has
superlinear growth, making the use of the widely-accepted (stochastic) gradient
descent methods, which are based on Euler numerical schemes, problematic. We
offer a new learning algorithm based on an appropriately constructed variant of
the popular stochastic gradient Langevin dynamics (SGLD), which is called tamed
unadjusted stochastic Langevin algorithm (TUSLA). We also provide a
nonasymptotic analysis of the new algorithm's convergence properties in the
context of non-convex learning problems with the use of ANNs. Thus, we provide
finite-time guarantees for TUSLA to find approximate minimizers of both
empirical and population risks. The roots of the TUSLA algorithm are based on
the taming technology for diffusion processes with superlinear coefficients as
developed in \citet{tamed-euler, SabanisAoAP} and for MCMC algorithms in
\citet{tula}. Numerical experiments are presented which confirm the theoretical
findings and illustrate the need for the use of the new algorithm in comparison
to vanilla SGLD within the framework of ANNs.

    

### [[2007.09541] Same-Day Delivery with Fairness](http://arxiv.org/abs/2007.09541)


  The demand for same-day delivery (SDD) has increased rapidly in the last few
years and has particularly boomed during the COVID-19 pandemic. The fast growth
is not without its challenge. In 2016, due to low concentrations of memberships
and far distance from the depot, certain minority neighborhoods were excluded
from receiving Amazon's SDD service, raising concerns about fairness. In this
paper, we study the problem of offering fair SDD-service to customers. The
service area is partitioned into different regions. Over the course of a day,
customers request for SDD service, and the timing of requests and delivery
locations are not known in advance. The dispatcher dynamically assigns vehicles
to make deliveries to accepted customers before their delivery deadline. In
addition to the overall service rate (utility), we maximize the minimal
regional service rate across all regions (fairness). We model the problem as a
multi-objective Markov decision process and develop a deep Q-learning solution
approach. We introduce a novel transformation of learning from rates to actual
services, which creates a stable and efficient learning process. Computational
results demonstrate the effectiveness of our approach in alleviating unfairness
both spatially and temporally in different customer geographies. We also show
this effectiveness is valid with different depot locations, providing
businesses with an opportunity to achieve better fairness from any location.
Further, we consider the impact of ignoring fairness in service, and results
show that our policies eventually outperform the utility-driven baseline when
customers have a high expectation on service level.

    

### [[2009.02400] The Area Under the ROC Curve as a Measure of Clustering Quality](http://arxiv.org/abs/2009.02400)


  The Area Under the the Receiver Operating Characteristics (ROC) Curve,
referred to as AUC, is a well-known performance measure in the supervised
learning domain. Due to its compelling features, it has been employed in a
number of studies to evaluate and compare the performance of different
classifiers. In this work, we explore AUC as a performance measure in the
unsupervised learning domain, more specifically, in the context of cluster
analysis. In particular, we elaborate on the use of AUC as an internal/relative
measure of clustering quality, which we refer to as Area Under the Curve for
Clustering (AUCC). We show that the AUCC of a given candidate clustering
solution has an expected value under a null model of random clustering
solutions, regardless of the size of the dataset and, more importantly,
regardless of the number or the (im)balance of clusters under evaluation. In
addition, we elaborate on the fact that, in the context of internal/relative
clustering validation as we consider, AUCC is actually a linear transformation
of the Gamma criterion from Baker and Hubert (1975), for which we also formally
derive a theoretical expected value for chance clusterings. We also discuss the
computational complexity of these criteria and show that, while an ordinary
implementation of Gamma can be computationally prohibitive and impractical for
most real applications of cluster analysis, its equivalence with AUCC actually
unveils a much more efficient algorithmic procedure. Our theoretical findings
are supported by experimental results. These results show that, in addition to
an effective and robust quantitative evaluation provided by AUCC, visual
inspection of the ROC curves themselves can be useful to further assess a
candidate clustering solution from a broader, qualitative perspective as well.

    

### [[2012.13635] Logic Tensor Networks](http://arxiv.org/abs/2012.13635)


  Artificial Intelligence agents are required to learn from their surroundings
and to reason about the knowledge that has been learned in order to make
decisions. While state-of-the-art learning from data typically uses
sub-symbolic distributed representations, reasoning is normally useful at a
higher level of abstraction with the use of a first-order logic language for
knowledge representation. As a result, attempts at combining symbolic AI and
neural computation into neural-symbolic systems have been on the increase. In
this paper, we present Logic Tensor Networks (LTN), a neurosymbolic formalism
and computational model that supports learning and reasoning through the
introduction of a many-valued, end-to-end differentiable first-order logic
called Real Logic as a representation language for deep learning. We show that
LTN provides a uniform language for the specification and the computation of
several AI tasks such as data clustering, multi-label classification,
relational learning, query answering, semi-supervised learning, regression and
embedding learning. We implement and illustrate each of the above tasks with a
number of simple explanatory examples using TensorFlow 2. Keywords:
Neurosymbolic AI, Deep Learning and Reasoning, Many-valued Logic.

    

### [[2102.06757] Multimodal Data Visualization and Denoising with Integrated Diffusion](http://arxiv.org/abs/2102.06757)


  We propose a method called integrated diffusion for combining multimodal
datasets, or data gathered via several different measurements on the same
system, to create a joint data diffusion operator. As real world data suffers
from both local and global noise, we introduce mechanisms to optimally
calculate a diffusion operator that reflects the combined information from both
modalities. We show the utility of this joint operator in data denoising,
visualization and clustering, performing better than other methods to integrate
and analyze multimodal data. We apply our method to multi-omic data generated
from blood cells, measuring both gene expression and chromatin accessibility.
Our approach better visualizes the geometry of the joint data, captures known
cross-modality associations and identifies known cellular populations. More
generally, integrated diffusion is broadly applicable to multimodal datasets
generated in many medical and biological systems.

    

### [[2102.08360] Interpretable COVID-19 Chest X-Ray Classification via Orthogonality Constraint](http://arxiv.org/abs/2102.08360)


  Deep neural networks have increasingly been used as an auxiliary tool in
healthcare applications, due to their ability to improve performance of several
diagnosis tasks. However, these methods are not widely adopted in clinical
settings due to the practical limitations in the reliability, generalizability,
and interpretability of deep learning based systems. As a result, methods have
been developed that impose additional constraints during network training to
gain more control as well as improve interpretabilty, facilitating their
acceptance in healthcare community. In this work, we investigate the benefit of
using Orthogonal Spheres (OS) constraint for classification of COVID-19 cases
from chest X-ray images. The OS constraint can be written as a simple
orthonormality term which is used in conjunction with the standard
cross-entropy loss during classification network training. Previous studies
have demonstrated significant benefits in applying such constraints to deep
learning models. Our findings corroborate these observations, indicating that
the orthonormality loss function effectively produces improved semantic
localization via GradCAM visualizations, enhanced classification performance,
and reduced model calibration error. Our approach achieves an improvement in
accuracy of 1.6% and 4.8% for two- and three-class classification,
respectively; similar results are found for models with data augmentation
applied. In addition to these findings, our work also presents a new
application of the OS regularizer in healthcare, increasing the post-hoc
interpretability and performance of deep learning models for COVID-19
classification to facilitate adoption of these methods in clinical settings. We
also identify the limitations of our strategy that can be explored for further
research in future.

    

### [[2104.07213] Attentive max feature map and joint training for acoustic scene classification](http://arxiv.org/abs/2104.07213)


  Various attention mechanisms are being widely applied to acoustic scene
classification. However, we empirically found that the attention mechanism can
excessively discard potentially valuable information, despite improving
performance. We propose the attentive max feature map that combines two
effective techniques, attention and a max feature map, to further elaborate the
attention mechanism and mitigate the above-mentioned phenomenon. We also
explore various joint training methods, including multi-task learning, that
allocate additional abstract labels for each audio recording. Our proposed
system demonstrates state-of-the-art performance for single systems on Subtask
A of the DCASE 2020 challenge by applying the two proposed techniques using
relatively fewer parameters. Furthermore, adopting the proposed attentive max
feature map, our team placed fourth in the recent DCASE 2021 challenge.

    

### [[2105.09016] E(n) Equivariant Normalizing Flows](http://arxiv.org/abs/2105.09016)


  This paper introduces a generative model equivariant to Euclidean symmetries:
E(n) Equivariant Normalizing Flows (E-NFs). To construct E-NFs, we take the
discriminative E(n) graph neural networks and integrate them as a differential
equation to obtain an invertible equivariant function: a continuous-time
normalizing flow. We demonstrate that E-NFs considerably outperform baselines
and existing methods from the literature on particle systems such as DW4 and
LJ13, and on molecules from QM9 in terms of log-likelihood. To the best of our
knowledge, this is the first flow that jointly generates molecule features and
positions in 3D.

    

### [[2106.03287] Stein ICP for Uncertainty Estimation in Point Cloud Matching](http://arxiv.org/abs/2106.03287)


  Quantification of uncertainty in point cloud matching is critical in many
tasks such as pose estimation, sensor fusion, and grasping. Iterative closest
point (ICP) is a commonly used pose estimation algorithm which provides a point
estimate of the transformation between two point clouds. There are many sources
of uncertainty in this process that may arise due to sensor noise, ambiguous
environment, and occlusion. However, for safety critical problems such as
autonomous driving, a point estimate of the pose transformation is not
sufficient as it does not provide information about the multiple solutions.
Current probabilistic ICP methods usually do not capture all sources of
uncertainty and may provide unreliable transformation estimates which can have
a detrimental effect in state estimation or decision making tasks that use this
information. In this work we propose a new algorithm to align two point clouds
that can precisely estimate the uncertainty of ICP's transformation parameters.
We develop a Stein variational inference framework with gradient based
optimization of ICP's cost function. The method provides a non-parametric
estimate of the transformation, can model complex multi-modal distributions,
and can be effectively parallelized on a GPU. Experiments using 3D kinect data
as well as sparse indoor/outdoor LiDAR data show that our method is capable of
efficiently producing accurate pose uncertainty estimates.

    

### [[2106.06044] Convergence and Alignment of Gradient Descent with Random Backpropagation Weights](http://arxiv.org/abs/2106.06044)


  Stochastic gradient descent with backpropagation is the workhorse of
artificial neural networks. It has long been recognized that backpropagation
fails to be a biologically plausible algorithm. Fundamentally, it is a
non-local procedure -- updating one neuron's synaptic weights requires
knowledge of synaptic weights or receptive fields of downstream neurons. This
limits the use of artificial neural networks as a tool for understanding the
biological principles of information processing in the brain. Lillicrap et al.
(2016) propose a more biologically plausible "feedback alignment" algorithm
that uses random and fixed backpropagation weights, and show promising
simulations. In this paper we study the mathematical properties of the feedback
alignment procedure by analyzing convergence and alignment for two-layer
networks under squared error loss. In the overparameterized setting, we prove
that the error converges to zero exponentially fast, and also that
regularization is necessary in order for the parameters to become aligned with
the random backpropagation weights. Simulations are given that are consistent
with this analysis and suggest further generalizations. These results
contribute to our understanding of how biologically plausible algorithms might
carry out weight learning in a manner different from Hebbian learning, with
performance that is comparable with the full non-local backpropagation
algorithm.

    

### [[2112.10872] Calabi-Yau Metrics, Energy Functionals and Machine-Learning](http://arxiv.org/abs/2112.10872)


  We apply machine learning to the problem of finding numerical Calabi-Yau
metrics. We extend previous work on learning approximate Ricci-flat metrics
calculated using Donaldson's algorithm to the much more accurate "optimal"
metrics of Headrick and Nassar. We show that machine learning is able to
predict the Kähler potential of a Calabi-Yau metric having seen only a small
sample of training data.

    

### [[2112.12520] Dependability Analysis of Data Storage Systems in Presence of Soft Errors](http://arxiv.org/abs/2112.12520)


  In recent years, high availability and reliability of Data Storage Systems
(DSS) have been significantly threatened by soft errors occurring in storage
controllers. Due to their specific functionality and hardware-software stack,
error propagation and manifestation in DSS is quite different from
general-purpose computing architectures. To our knowledge, no previous study
has examined the system-level effects of soft errors on the availability and
reliability of data storage systems. In this paper, we first analyze the
effects of soft errors occurring in the server processors of storage
controllers on the entire storage system dependability. To this end, we
implemented the major functions of a typical data storage system controller,
running on a full stack of storage system operating system, and developed a
framework to perform fault injection experiments using a full system simulator.
We then propose a new metric, Storage System Vulnerability Factor (SSVF), to
accurately capture the impact of soft errors in storage systems. By conducting
extensive experiments, it is revealed that depending on the controller
configuration, up to 40% of cache memory contains end-user data where any
unrecoverable soft errors in this part will result in Data Loss (DL) in an
irreversible manner. However, soft errors in the rest of cache memory filled by
Operating System (OS) and storage applications will result in Data
Unavailability (DU) at the storage system level. Our analysis also shows that
Detectable Unrecoverable Errors (DUEs) on the cache data field are the major
cause of DU in storage systems, while Silent Data Corruptions (SDCs) in the
cache tag and data field are mainly the cause of DL in storage systems.

    

### [[2112.12667] Using Silent Writes in Low-Power Traffic-Aware ECC](http://arxiv.org/abs/2112.12667)


  Using Error Detection Code (EDC) and Error Correction Code (ECC) is a
noteworthy way to increase cache memories robustness against soft errors. EDC
enables detecting errors in cache memory while ECC is used to correct erroneous
cache blocks. ECCs are often costly as they impose considerable area and energy
overhead on cache memory. Reducing this overhead has been the subject of many
studies. In particular, a previous study has suggested mapping ECC to the main
memory at the expense of high cache traffic and energy. A major source of this
excessive traffic and energy is the high frequency of cache writes. In this
work, we show that a significant portion of cache writes are silent, i.e., they
write the same data already existing. We build on this observation and
introduce Traffic-aware ECC (or simply TCC). TCC detects silent writes by an
efficient mechanism. Once such writes are detected updating their ECC is
avoided effectively reducing L2 cache traffic and access frequency. Using our
solution, we reduce L2 cache access frequency by 8% while maintaining
performance. We reduce L2 cache dynamic and overall cache energy by up to 32%
and 8%, respectively. Furthermore, TCC reduces L2 cache miss rate by 3%.

    

### [[2112.12415] In-storage Processing of I/O Intensive Applications on Computational Storage Drives](http://arxiv.org/abs/2112.12415)


  Computational storage drives (CSD) are solid-state drives (SSD) empowered by
general-purpose processors that can perform in-storage processing. They have
the potential to improve both performance and energy significantly for big-data
analytics by bringing compute to data, thereby eliminating costly data transfer
while offering better privacy. In this work, we introduce Solana, the
first-ever high-capacity(12-TB) CSD in E1.S form factor, and present an actual
prototype for evaluation. To demonstrate the benefits of in-storage processing
on CSD, we deploy several natural language processing (NLP) applications on
datacenter-grade storage servers comprised of clusters of the Solana.
Experimental results show up to 3.1x speedup in processing while reducing the
energy consumption and data transfer by 67% and 68%, respectively, compared to
regular enterprise SSDs.

    

### [[2112.12685] Dynamic Page Placement on Real Persistent Memory Systems](http://arxiv.org/abs/2112.12685)


  As persistent memory (PM) technologies emerge, hybrid memory architectures
combining DRAM with PM bring the potential to provide a tiered,
byte-addressable main memory of unprecedented capacity. Nearly a decade after
the first proposals for these hybrid architectures, the real technology has
finally reached commercial availability with Intel Optane(TM) DC Persistent
Memory (DCPMM). This raises the challenge of designing systems that realize
this potential in practice, namely through effective approaches that
dynamically decide at which memory tier should pages be placed. In this paper,
we are the first, to our knowledge, to systematically analyze tiered page
placement on real DCPMM-based systems. To this end, we start by revisiting the
assumptions of state-of-the-art proposals, and confronting them with the
idiosyncrasies of today's off-the-shelf DCPMM-equipped architectures. This
empirical study reveals that some of the key design choices in the literature
rely on important assumptions that are not verified in present-day DRAM-DCPMM
memory architectures. Based on the lessons from this study, we design and
implement HyPlacer, a tool for tiered page placement in off-the-shelf
Linux-based systems equipped with DRAM+DCPMM. In contrast to previous
proposals, HyPlacer follows an approach guided by two main practicality
principles: 1) it is tailored to the performance idiosyncrasies of off-theshelf
DRAM+DCPMM systems; and 2) it can be seamlessly integrated into Linux with
minimal kernel-mode components, while ensuring extensibility to other HMAs and
other data placement policies. Our experimental evaluation of HyPlacer shows
that it outperforms both solutions proposed in past literature and placement
options that are currently available in off-the-shelf DCPMM-equipped Linux
systems, reaching an improvement of up to 11x when compared to the default
memory policy in Linux.

    

### [[2112.12704] Deterministic Parallel Hypergraph Partitioning](http://arxiv.org/abs/2112.12704)


  Balanced hypergraph partitioning is a classical NP-hard optimization problem
with applications in various domains such as VLSI design, simulating quantum
circuits, optimizing data placement in distributed databases or minimizing
communication volume in high-performance computing. Engineering parallel
heuristics for this problem is a topic of recent research. Most of them are
non-deterministic though. In this work, we design and implement a highly
scalable deterministic algorithm in the state-of-the-art parallel partitioning
framework Mt-KaHyPar. On our extensive set of benchmark instances, it achieves
similar partition quality and performance as a comparable but non-deterministic
configuration of Mt-KaHyPar and outperforms the only other existing parallel
deterministic algorithm BiPart regarding partition quality, running time and
parallel speedups.

    

### [[1911.01858] A GenEO Domain Decomposition method for Saddle Point problems](http://arxiv.org/abs/1911.01858)


  We introduce an adaptive element-based domain decomposition (DD) method for
solving saddle point problems defined as a block two by two matrix. The
algorithm does not require any knowledge of the constrained space. We assume
that all sub matrices are sparse and that the diagonal blocks are spectrally
equivalent to a sum of positive semi definite matrices. The latter assumption
enables the design of adaptive coarse space for DD methods that extends the
GenEO theory to saddle point problems. Numerical results on three dimensional
elasticity problems for steel-rubber structures discretized by a finite element
with continuous pressure are shown for up to one billion degrees of freedom.

    

### [[2112.11393] A Survey on Perfectly-Secure Verifiable Secret-Sharing](http://arxiv.org/abs/2112.11393)


  Verifiable Secret-Sharing (VSS) is a fundamental primitive in secure
distributed computing. It is used as a building block in several distributed
computing tasks, such as Byzantine agreement and secure multi-party
computation. In this article, we consider VSS schemes with perfect security,
tolerating computationally unbounded adversaries. We comprehensively survey the
existing perfectly-secure VSS schemes in three different communication
settings, namely synchronous, asynchronous and hybrid setting and provide full
details of the existing schemes in these settings. The aim of this survey is to
provide a clear knowledge and foundation to researchers who are interested in
knowing and extending the state-of-the-art perfectly-secure VSS schemes.

    

### [[2112.12180] Multimodal Personality Recognition using Cross-Attention Transformer and Behaviour Encoding](http://arxiv.org/abs/2112.12180)


  Personality computing and affective computing have gained recent interest in
many research areas. The datasets for the task generally have multiple
modalities like video, audio, language and bio-signals. In this paper, we
propose a flexible model for the task which exploits all available data. The
task involves complex relations and to avoid using a large model for video
processing specifically, we propose the use of behaviour encoding which boosts
performance with minimal change to the model. Cross-attention using
transformers has become popular in recent times and is utilised for fusion of
different modalities. Since long term relations may exist, breaking the input
into chunks is not desirable, thus the proposed model processes the entire
input together. Our experiments show the importance of each of the above
contributions

    

### [[2112.12182] Fine-grained Multi-Modal Self-Supervised Learning](http://arxiv.org/abs/2112.12182)


  Multi-Modal Self-Supervised Learning from videos has been shown to improve
model's performance on various downstream tasks. However, such Self-Supervised
pre-training requires large batch sizes and a large amount of computation
resources due to the noise present in the uncurated data. This is partly due to
the fact that the prevalent training scheme is trained on coarse-grained
setting, in which vectors representing the whole video clips or natural
language sentences are used for computing similarity. Such scheme makes
training noisy as part of the video clips can be totally not correlated with
the other-modality input such as text description. In this paper, we propose a
fine-grained multi-modal self-supervised training scheme that computes the
similarity between embeddings at finer-scale (such as individual feature map
embeddings and embeddings of phrases), and uses attention mechanisms to reduce
noisy pairs' weighting in the loss function. We show that with the proposed
pre-training scheme, we can train smaller models, with smaller batch-size and
much less computational resources to achieve downstream tasks performances
comparable to State-Of-The-Art, for tasks including action recognition and
text-image retrievals.

    

### [[2112.12255] Entropy-Regularized Partially Observed Markov Decision Processes](http://arxiv.org/abs/2112.12255)


  We investigate partially observed Markov decision processes (POMDPs) with
cost functions regularized by entropy terms describing state, observation, and
control uncertainty. Standard POMDP techniques are shown to offer bounded-error
solutions to these entropy-regularized POMDPs, with exact solutions when the
regularization involves the joint entropy of the state, observation, and
control trajectories. Our joint-entropy result is particularly surprising since
it constitutes a novel, tractable formulation of active state estimation.

    

### [[2112.12310] Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art](http://arxiv.org/abs/2112.12310)


  The malware has been being one of the most damaging threats to computers that
span across multiple operating systems and various file formats. To defend
against the ever-increasing and ever-evolving threats of malware, tremendous
efforts have been made to propose a variety of malware detection methods that
attempt to effectively and efficiently detect malware. Recent studies have
shown that, on the one hand, existing ML and DL enable the superior detection
of newly emerging and previously unseen malware. However, on the other hand, ML
and DL models are inherently vulnerable to adversarial attacks in the form of
adversarial examples, which are maliciously generated by slightly and carefully
perturbing the legitimate inputs to confuse the targeted models. Basically,
adversarial attacks are initially extensively studied in the domain of computer
vision, and some quickly expanded to other domains, including NLP, speech
recognition and even malware detection. In this paper, we focus on malware with
the file format of portable executable (PE) in the family of Windows operating
systems, namely Windows PE malware, as a representative case to study the
adversarial attack methods in such adversarial settings. To be specific, we
start by first outlining the general learning framework of Windows PE malware
detection based on ML/DL and subsequently highlighting three unique challenges
of performing adversarial attacks in the context of PE malware. We then conduct
a comprehensive and systematic review to categorize the state-of-the-art
adversarial attacks against PE malware detection, as well as corresponding
defenses to increase the robustness of PE malware detection. We conclude the
paper by first presenting other related attacks against Windows PE malware
detection beyond the adversarial attacks and then shedding light on future
research directions and opportunities.

    

### [[2112.12318] Investigating Effect of Dialogue History in Multilingual Task Oriented Dialogue Systems](http://arxiv.org/abs/2112.12318)


  While the English virtual assistants have achieved exciting performance with
an enormous amount of training resources, the needs of non-English-speakers
have not been satisfied well. Up to Dec 2021, Alexa, one of the most popular
smart speakers around the world, is able to support 9 different languages [1],
while there are thousands of languages in the world, 91 of which are spoken by
more than 10 million people according to statistics published in 2019 [2].
However, training a virtual assistant in other languages than English is often
more difficult, especially for those low-resource languages. The lack of
high-quality training data restricts the performance of models, resulting in
poor user satisfaction. Therefore, we devise an efficient and effective
training solution for multilingual task-orientated dialogue systems, using the
same dataset generation pipeline and end-to-end dialogue system architecture as
BiToD[5], which adopted some key design choices for a minimalistic natural
language design where formal dialogue states are used in place of natural
language inputs. This reduces the room for error brought by weaker natural
language models, and ensures the model can correctly extract the essential slot
values needed to perform dialogue state tracking (DST). Our goal is to reduce
the amount of natural language encoded at each turn, and the key parameter we
investigate is the number of turns (H) to feed as history to model. We first
explore the turning point where increasing H begins to yield limiting returns
on the overall performance. Then we examine whether the examples a model with
small H gets wrong can be categorized in a way for the model to do few-shot
finetuning on. Lastly, will explore the limitations of this approach, and
whether there is a certain type of examples that this approach will not be able
to resolve.

    

### [[2112.12327] Making sense of electrical vehicle discussions using sentiment analysis on closely related news and user comments](http://arxiv.org/abs/2112.12327)


  We used a token-wise and document-wise sentiment analysis using both
unsupervised and supervised models applied to both news and user reviews
dataset. And our token-wise sentiment analysis found a statistically
significant difference in sentiment between the two groups (both of which were
very large N), our document-wise supervised sentiment analysis found no
significant difference in sentiment.

    

### [[2112.12496] FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition](http://arxiv.org/abs/2112.12496)


  Current state-of-the-art deep learning based face recognition (FR) models
require a large number of face identities for central training. However, due to
the growing privacy awareness, it is prohibited to access the face images on
user devices to continually improve face recognition models. Federated Learning
(FL) is a technique to address the privacy issue, which can collaboratively
optimize the model without sharing the data between clients. In this work, we
propose a FL based framework called FedFR to improve the generic face
representation in a privacy-aware manner. Besides, the framework jointly
optimizes personalized models for the corresponding clients via the proposed
Decoupled Feature Customization module. The client-specific personalized model
can serve the need of optimized face recognition experience for registered
identities at the local device. To the best of our knowledge, we are the first
to explore the personalized face recognition in FL setup. The proposed
framework is validated to be superior to previous approaches on several generic
and personalized face recognition benchmarks with diverse FL scenarios. The
source codes and our proposed personalized FR benchmark under FL setup are
available at this https URL.

    

### [[2112.12508] From Procedures, Objects, Actors, Components, Services, to Agents -- A Comparative Analysis of the History and Evolution of Programming Abstractions](http://arxiv.org/abs/2112.12508)


  The objective of this chapter is to propose some retrospective analysis of
the evolution of programming abstractions, from {\em procedures}, {\em
objects}, {\em actors}, {\em components}, {\em services}, up to {\em agents},
%have some compare concepts of software component and of agent (and multi-agent
system), %The method chosen is to by replacing them within a general historical
perspective. Some common referential with three axes/dimensions is chosen: {\em
action selection} at the level of one entity, {\em coupling flexibility}
between entities, and {\em abstraction level}. We indeed may observe some
continuous quest for higher flexibility (through notions such as {\em late
binding}, or {\em reification} of {\em connections}) and higher level of {\em
abstraction}. Concepts of components, services and agents have some common
objectives (notably, {\em software modularity and reconfigurability}), with
multi-agent systems raising further concepts of {\em autonomy} and {\em
coordination}. notably through the notion of {\em auto-organization} and the
use of {\em knowledge}. We hope that this analysis helps at highlighting some
of the basic forces motivating the progress of programming abstractions and
therefore that it may provide some seeds for the reflection about future
programming abstractions.

    

### [[2112.12702] TagLab: A human-centric AI system for interactive semantic segmentation](http://arxiv.org/abs/2112.12702)


  Fully automatic semantic segmentation of highly specific semantic classes and
complex shapes may not meet the accuracy standards demanded by scientists. In
such cases, human-centered AI solutions, able to assist operators while
preserving human control over complex tasks, are a good trade-off to speed up
image labeling while maintaining high accuracy levels. TagLab is an open-source
AI-assisted software for annotating large orthoimages which takes advantage of
different degrees of automation; it speeds up image annotation from scratch
through assisted tools, creates custom fully automatic semantic segmentation
models, and, finally, allows the quick edits of automatic predictions. Since
the orthoimages analysis applies to several scientific disciplines, TagLab has
been designed with a flexible labeling pipeline. We report our results in two
different scenarios, marine ecology, and architectural heritage.

    

### [[2112.12744] AI-based Reconstruction for Fast MRI -- A Systematic Review and Meta-analysis](http://arxiv.org/abs/2112.12744)


  Compressed sensing (CS) has been playing a key role in accelerating the
magnetic resonance imaging (MRI) acquisition process. With the resurgence of
artificial intelligence, deep neural networks and CS algorithms are being
integrated to redefine the state of the art of fast MRI. The past several years
have witnessed substantial growth in the complexity, diversity, and performance
of deep learning-based CS techniques that are dedicated to fast MRI. In this
meta-analysis, we systematically review the deep learning-based CS techniques
for fast MRI, describe key model designs, highlight breakthroughs, and discuss
promising directions. We have also introduced a comprehensive analysis
framework and a classification system to assess the pivotal role of deep
learning in CS-based acceleration for MRI.

    

### [[2112.12754] Toward a New Science of Common Sense](http://arxiv.org/abs/2112.12754)


  Common sense has always been of interest in AI, but has rarely taken center
stage. Despite its mention in one of John McCarthy's earliest papers and years
of work by dedicated researchers, arguably no AI system with a serious amount
of general common sense has ever emerged. Why is that? What's missing? Examples
of AI systems' failures of common sense abound, and they point to AI's frequent
focus on expertise as the cause. Those attempting to break the brittleness
barrier, even in the context of modern deep learning, have tended to invest
their energy in large numbers of small bits of commonsense knowledge. But all
the commonsense knowledge fragments in the world don't add up to a system that
actually demonstrates common sense in a human-like way. We advocate examining
common sense from a broader perspective than in the past. Common sense is more
complex than it has been taken to be and is worthy of its own scientific
exploration.

    

### [[2112.12768] An Ontological Knowledge Representation for Smart Agriculture](http://arxiv.org/abs/2112.12768)


  In order to provide the agricultural industry with the infrastructure it
needs to take advantage of advanced technology, such as big data, the cloud,
and the internet of things (IoT); smart farming is a management concept that
focuses on providing the infrastructure necessary to track, monitor, automate,
and analyse operations. To represent the knowledge extracted from the primary
data collected is of utmost importance. An agricultural ontology framework for
smart agriculture systems is presented in this study. The knowledge graph is
represented as a lattice to capture and perform reasoning on spatio-temporal
agricultural data.

    

### [[2112.12786] ELSA: Enhanced Local Self-Attention for Vision Transformer](http://arxiv.org/abs/2112.12786)


  Self-attention is powerful in modeling long-range dependencies, but it is
weak in local finer-level feature learning. The performance of local
self-attention (LSA) is just on par with convolution and inferior to dynamic
filters, which puzzles researchers on whether to use LSA or its counterparts,
which one is better, and what makes LSA mediocre. To clarify these, we
comprehensively investigate LSA and its counterparts from two sides:
\emph{channel setting} and \emph{spatial processing}. We find that the devil
lies in the generation and application of spatial attention, where relative
position embeddings and the neighboring filter application are key factors.
Based on these findings, we propose the enhanced local self-attention (ELSA)
with Hadamard attention and the ghost head. Hadamard attention introduces the
Hadamard product to efficiently generate attention in the neighboring case,
while maintaining the high-order mapping. The ghost head combines attention
maps with static matrices to increase channel capacity. Experiments demonstrate
the effectiveness of ELSA. Without architecture / hyperparameter modification,
drop-in replacing LSA with ELSA boosts Swin Transformer \cite{swin} by up to
+1.4 on top-1 accuracy. ELSA also consistently benefits VOLO \cite{volo} from
D1 to D5, where ELSA-VOLO-D5 achieves 87.2 on the ImageNet-1K without extra
training images. In addition, we evaluate ELSA in downstream tasks. ELSA
significantly improves the baseline by up to +1.9 box Ap / +1.3 mask Ap on the
COCO, and by up to +1.9 mIoU on the ADE20K. Code is available at
\url{this https URL}.

    

### [[2101.00058] Conflict-driven Inductive Logic Programming](http://arxiv.org/abs/2101.00058)


  The goal of Inductive Logic Programming (ILP) is to learn a program that
explains a set of examples. Until recently, most research on ILP targeted
learning Prolog programs. The ILASP system instead learns Answer Set Programs
(ASP). Learning such expressive programs widens the applicability of ILP
considerably; for example, enabling preference learning, learning common-sense
knowledge, including defaults and exceptions, and learning non-deterministic
theories.
Early versions of ILASP can be considered meta-level ILP approaches, which
encode a learning task as a logic program and delegate the search to an ASP
solver. More recently, ILASP has shifted towards a new method, inspired by
conflict-driven SAT and ASP solvers. The fundamental idea of the approach,
called Conflict-driven ILP (CDILP), is to iteratively interleave the search for
a hypothesis with the generation of constraints which explain why the current
hypothesis does not cover a particular example. These coverage constraints
allow ILASP to rule out not just the current hypothesis, but an entire class of
hypotheses that do not satisfy the coverage constraint.
This paper formalises the CDILP approach and presents the ILASP3 and ILASP4
systems for CDILP, which are demonstrated to be more scalable than previous
ILASP systems, particularly in the presence of noise.
Under consideration in Theory and Practice of Logic Programming (TPLP).

    

### [[2110.12981] Neural ODE and DAE Modules for Power System Dynamic Component Modeling](http://arxiv.org/abs/2110.12981)


  The time-domain simulation is fundamental for power system transient
stability analysis. Accurate and reliable simulations depend on accurate
dynamic component modeling. In practical power systems, dynamic component
modeling has long faced the challenges of model determination and model
calibration, especially with the rapid development of renewable generation and
power electronics. In this paper, based on the general framework of neural
ordinary differential equations (ODEs), a modified neural ODE module and a
neural differential-algebraic equations (DAEs) module for power system dynamic
component modeling are proposed. Autoencoder-based frameworks of the proposed
modules are put forward to upgrade model performance. The methodology of
integrating neural dynamic models trained by the proposed neural modules into
transient stability simulations is demonstrated. With datasets consisting of
sampled curves of input variables and output variables, the proposed modules
can be used to fulfill the tasks of black-box modeling, physics-data-integrated
modeling, parameter inference, etc. Tests are carried out in the IEEE-39 system
to prove the validity and potentiality of the proposed modules.

    

### [[2112.10321] English-to-Chinese Transliteration with Phonetic Back-transliteration](http://arxiv.org/abs/2112.10321)


  Transliteration is a task of translating named entities from a language to
another, based on phonetic similarity. The task has embraced deep learning
approaches in recent years, yet, most ignore the phonetic features of the
involved languages. In this work, we incorporate phonetic information into
neural networks in two ways: we synthesize extra data using forward and
back-translation but in a phonetic manner; and we pre-train models on a
phonetic task before learning transliteration. Our experiments include three
language pairs and six directions, namely English to and from Chinese, Hebrew
and Thai. Results indicate that our proposed approach brings benefits to the
model and achieves better or similar performance when compared to state of the
art.

    

### [[2112.12575] A Modeling Framework for Reliability of Erasure Codes in SSD Arrays](http://arxiv.org/abs/2112.12575)


  To help reliability of SSD arrays, Redundant Array of Independent Disks
(RAID) are commonly employed. However, the conventional reliability models of
HDD RAID cannot be applied to SSD arrays, as the nature of failures in SSDs are
different from HDDs. Previous studies on the reliability of SSD arrays are
based on the deprecated SSD failure data, and only focus on limited failure
types, device failures, and page failures caused by the bit errors, while
recent field studies have reported other failure types including bad blocks and
bad chips, and a high correlation between failures. In this paper, we explore
the reliability of SSD arrays using field storage traces and real-system
implementation of conventional and emerging erasure codes. The reliability is
evaluated by statistical fault injections that post-process the usage logs from
the real-system implementation, while the fault/failure attributes are obtained
from field data. As a case study, we examine conventional and emerging erasure
codes in terms of both reliability and performance using Linux MD RAID and
commercial SSDs. Our analysis shows that a) emerging erasure codes fail to
replace RAID6 in terms of reliability, b) row-wise erasure codes are the most
efficient choices for contemporary SSD devices, and c) previous models
overestimate the SSD array reliability by up to six orders of magnitude, as
they focus on the coincidence of bad pages and bad chips that roots the
minority of Data Loss (DL) in SSD arrays. Our experiments show that the
combination of bad chips with bad blocks is the major source of DL in RAID5 and
emerging codes (contributing more than 54% and 90% of DL in RAID5 and emerging
codes, respectively), while RAID6 remains robust under these failure
combinations. Finally, the fault injection results show that SSD array
reliability, as well as the failure breakdown is significantly correlated with
SSD type.

    

### [[1904.06480] Dynamic scheduling in a partially fluid, partially lossy queueing system](http://arxiv.org/abs/1904.06480)


  We consider a single server queueing system with two classes of jobs: eager
jobs with small sizes that require service to begin almost immediately upon
arrival, and tolerant jobs with larger sizes that can wait for service. While
blocking probability is the relevant performance metric for the eager class,
the tolerant class seeks to minimize its mean sojourn time. In this paper, we
discuss the performance of each class under dynamic scheduling policies, where
the scheduling of both classes depends on the instantaneous state of the
system. This analysis is carried out under a certain fluid limit, where the
arrival rate and service rate of the eager class are scaled to infinity,
holding the offered load constant. Our performance characterizations reveal a
(dynamic) pseudo-conservation law that ties the performance of both the classes
to the standalone blocking probabilities of the eager class. Further, the
performance is robust to other specifics of the scheduling policies. We also
characterize the Pareto frontier of the achievable region of performance
vectors under the same fluid limit, and identify a (two-parameter) class of
Pareto-complete scheduling policies.

    

### [[2004.11847] Age of Information for Single Buffer Systems with Vacation Server](http://arxiv.org/abs/2004.11847)


  In this research, we study the information freshness in M/G/1 queueing system
with a single buffer and the server taking multiple vacations. This system has
wide applications in communication systems. We aim to evaluate the information
freshness in this system with both i.i.d. and non-i.i.d. vacations under three
different scheduling policies, namely Conventional Buffer System (CBS), Buffer
Relaxation System (BRS), and Conventional Buffer System with Preemption in
Service (CBS-P). For the systems with i.i.d. vacations, we derive the
closed-form expressions of information freshness metrics such as the expected
Age of Information (AoI), the expected Peak Age of Information (PAoI), and the
variance of peak age under each policy. For systems with non-i.i.d. vacations,
we use the polling system as an example and provide the closed-form expression
of its PAoI under each policy. We explore the conditions under which one of
these policies has advantages over the others for each information freshness
metric. We further perform numerical studies to validate our results and
develop insights.

    

### [[2107.12246] On the Quantum Performance Evaluation of Two Distributed Quantum Architectures](http://arxiv.org/abs/2107.12246)


  Distributed quantum applications impose requirements on the quality of the
quantum states that they consume. When analyzing architecture implementations
of quantum hardware, characterizing this quality forms an important factor in
understanding their performance. Fundamental characteristics of quantum
hardware lead to inherent tradeoffs between the quality of states and
traditional performance metrics such as throughput. Furthermore, any real-world
implementation of quantum hardware exhibits time-dependent noise that degrades
the quality of quantum states over time. Here, we study the performance of two
possible architectures for interfacing a quantum processor with a quantum
network. The first corresponds to the current experimental state of the art in
which the same device functions both as a processor and a network device. The
second corresponds to a future architecture that separates these two functions
over two distinct devices. We model these architectures as Markov chains and
compare their quality of executing quantum operations and producing entangled
quantum states as functions of their memory lifetimes, as well as the time that
it takes to perform various operations within each architecture. As an
illustrative example, we apply our analysis to architectures based on
Nitrogen-Vacancy centers in diamond, where we find that for present-day device
parameters one architecture is more suited to computation-heavy applications,
and the other for network-heavy ones. Besides the detailed study of these
architectures, a novel contribution of our work are several formulas that
connect an understanding of waiting time distributions to the decay of quantum
quality over time for the most common noise models employed in quantum
technologies. This provides a valuable new tool for performance evaluation
experts, and its applications extend beyond the two architectures studied in
this work.

    

### [[2112.12398] Towards Fully Declarative Program Analysis via Source Code Transformation](http://arxiv.org/abs/2112.12398)


  Advances in logic programming and increasing industrial uptake of
Datalog-inspired approaches demonstrate the emerging need to express powerful
code analyses more easily. Declarative program analysis frameworks (e.g., using
logic programming like Datalog) significantly ease defining analyses compared
to imperative implementations. However, the declarative benefits of these
frameworks only materialize after parsing and translating source code to
generate facts. Fact generation remains a non-declarative precursor to analysis
where imperative implementations first parse and interpret program structures
(e.g., abstract syntax trees and control-flow graphs). The procedure of fact
generation thus remains opaque and difficult for non-experts to understand or
modify. We present a new perspective on this analysis workflow by proposing
declarative fact generation to ease specification and exploration of
lightweight declarative analyses. Our approach demonstrates the first venture
towards fully declarative analysis specification across multiple languages. The
key idea is to translate source code directly to Datalog facts in the analysis
domain using declarative syntax transformation. We then reuse existing Datalog
analyses over generated facts, yielding an end-to-end declarative pipeline. As
a first approximation we pursue a syntax-driven approach and demonstrate the
feasibility of generating and using lightweight versions of liveness and call
graph reachability properties. We then discuss the workability of extending
declarative fact generation to also incorporate semantic information.

    

### [[2112.12693] Deadlock-free asynchronous message reordering in Rust with multiparty session types](http://arxiv.org/abs/2112.12693)


  Rust is a modern systems language focused on performance and reliability.
Complementing Rust's promise to provide "fearless concurrency", developers
frequently exploit asynchronous message passing. Unfortunately, arbitrarily
ordering sending and receiving messages to maximise computation-communication
overlap (a popular optimisation to message-passing applications) opens up a
Pandora's box of further subtle concurrency bugs.
To guarantee deadlock-freedom by construction, we present Rumpsteak: a new
Rust framework based on multiparty session types. Previous session type
implementations in Rust are either built upon synchronous and blocking
communication and/or limited to two-party interactions. Crucially, none support
the arbitrary ordering of messages for efficiency.
Rumpsteak instead targets asynchronous async/await code. Its unique ability
is allowing developers to arbitrarily order send/receive messages while
preserving deadlock-freedom. For this, Rumpsteak incorporates two recent
advanced session type theories: (1) k-multiparty compatibility (kmc), which
globally verifies the safety of a set of participants, and (2) asynchronous
multiparty session subtyping, which locally verifies optimisations in the
context of a single participant. Specifically, we propose a novel algorithm for
asynchronous subtyping that is both sound and decidable.
We first evaluate the performance and expressiveness of Rumpsteak against
three previous Rust implementations. We discover that Rumpsteak is around
1.7--8.6x more efficient and can safely express many more examples by virtue of
offering arbitrary message ordering. Secondly, we analyse the complexity of our
new algorithm and benchmark it against kmc and a binary session subtyping
algorithm. We find they are exponentially slower than Rumpsteak's.

    